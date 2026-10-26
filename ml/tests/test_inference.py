import json
from copy import copy
from dataclasses import replace

import numpy as np
import pytest

from nids_ml import (InputValidationError, ModelUnavailableError, PredictionOptions, PredictionResult,
                     SchemaVersionError, predict)
from nids_ml.contracts import REASON_CODES, severity_for_score
from nids_ml.inference import decide_row
from nids_ml.risk import RiskPolicy
from conftest import FIXED_NOW, fixed_clock
from test_features import GOOD, rec

OPTS = PredictionOptions(clock=fixed_clock)
FIELDS = {"row_index", "prediction_label", "prediction_type", "confidence", "anomaly_score", "risk_score",
          "severity", "reason_codes", "model_version", "feature_schema_version", "prediction_timestamp"}


def D(**kw):
    base = dict(cls_label="port_scan", confidence=0.9, anomaly_score=0.2, known_confidence=0.7, anomaly_threshold=0.8,
                benign_label="benign", classifier_available=True, anomaly_available=True, risk_policy=RiskPolicy())
    base.update(kw); return decide_row(**base)


# ---- pure decision policy: known-class and anomaly paths independently and together ----
def test_known_attack_path_only():
    d = D()
    assert (d["prediction_type"], d["prediction_label"]) == ("known_attack", "port_scan")
    assert d["reason_codes"] == ["KNOWN_CLASS_HIGH_SCORE"]


def test_high_confidence_known_attack_is_not_overwritten_by_anomaly():
    d = D(anomaly_score=0.97)
    assert (d["prediction_type"], d["prediction_label"]) == ("known_attack", "port_scan")
    assert "KNOWN_CLASS_HIGH_SCORE" in d["reason_codes"] and "ANOMALY_HIGH" in d["reason_codes"]


def test_low_confidence_is_uncertain_unknown():
    d = D(confidence=0.4)
    assert (d["prediction_type"], d["prediction_label"]) == ("uncertain", "unknown")
    assert d["reason_codes"] == ["KNOWN_CLASS_LOW_CONFIDENCE"]


def test_low_confidence_plus_anomaly_is_explicitly_promoted():
    d = D(confidence=0.4, anomaly_score=0.85)
    assert (d["prediction_type"], d["prediction_label"]) == ("anomaly", "unknown")
    assert d["risk_score"] == 64 and d["severity"] == "high"
    assert d["reason_codes"] == ["KNOWN_CLASS_LOW_CONFIDENCE", "ANOMALY_ELEVATED", "RISK_POLICY_HIGH"]


def test_benign_with_high_anomaly_is_anomaly_unknown_never_benign():
    d = D(cls_label="benign", anomaly_score=0.9)
    assert (d["prediction_type"], d["prediction_label"]) == ("anomaly", "unknown")


def test_benign_confident_and_quiet_is_benign():
    d = D(cls_label="benign", anomaly_score=0.1, confidence=0.95)
    assert (d["prediction_type"], d["prediction_label"], d["reason_codes"]) == ("benign", "benign", [])
    assert d["severity"] == "info"


def test_anomaly_only_model_never_says_benign():
    d = D(classifier_available=False, cls_label=None, confidence=None, anomaly_score=0.1)
    assert (d["prediction_type"], d["prediction_label"]) == ("uncertain", "unknown")
    assert "MODEL_COMPONENT_UNAVAILABLE" in d["reason_codes"] and d["confidence"] is None
    d2 = D(classifier_available=False, cls_label=None, confidence=None, anomaly_score=0.95)
    assert d2["prediction_type"] == "anomaly"


def test_classifier_only_model_reports_missing_anomaly_component():
    d = D(anomaly_available=False, anomaly_score=None)
    assert d["anomaly_score"] is None and "MODEL_COMPONENT_UNAVAILABLE" in d["reason_codes"]
    assert d["prediction_type"] == "known_attack"


def test_reason_codes_only_from_allowlist_and_risk_policy_high_is_consistent():
    for conf in (0.3, 0.8, 0.99):
        for an in (0.0, 0.85, 1.0):
            d = D(confidence=conf, anomaly_score=an, incomplete=True, out_of_distribution=True)
            assert set(d["reason_codes"]) <= set(REASON_CODES)
            assert ("RISK_POLICY_HIGH" in d["reason_codes"]) == (d["risk_score"] >= 60)
            assert d["severity"] == severity_for_score(d["risk_score"])


# ---- predict() against a real fitted bundle ----
def test_output_count_order_schema_and_finite_json(bundle, test_records):
    recs, _ = test_records
    out = predict(recs[:200], bundle, OPTS)
    assert len(out) == 200 and [r.row_index for r in out] == list(range(200))
    for r in out:
        assert isinstance(r, PredictionResult) and set(r.model_dump()) == FIELDS
        assert r.model_version == bundle.version and r.feature_schema_version == "1.0.0"
        assert r.prediction_timestamp == "2026-10-25T12:00:00Z"
        assert r.severity == severity_for_score(r.risk_score)
    json.dumps([r.model_dump() for r in out], allow_nan=False)


def test_predict_is_deterministic(bundle, test_records):
    recs, _ = test_records
    a = [r.model_dump() for r in predict(recs[:100], bundle, OPTS)]
    b = [r.model_dump() for r in predict(recs[:100], bundle, OPTS)]
    assert a == b


def test_row_result_does_not_depend_on_batch_composition(bundle, test_records):
    recs, _ = test_records
    solo = predict([recs[3]], bundle, OPTS)[0].model_dump(); batch = predict(recs[:10], bundle, OPTS)[3].model_dump()
    solo["row_index"] = batch["row_index"] = 0
    assert solo == batch


def test_input_validation_errors_propagate(bundle):
    with pytest.raises(InputValidationError):
        predict([], bundle)
    with pytest.raises(InputValidationError):
        predict([rec(duration=-1)], bundle)
    with pytest.raises(InputValidationError):
        predict([GOOD] * 5, bundle, PredictionOptions(max_batch_size=4))
    with pytest.raises(InputValidationError):
        predict([dict(GOOD, label="dos_flood")], bundle)  # label leakage attempt: rejected, not silently used
    with pytest.raises(SchemaVersionError):
        predict({"feature_schema_version": "2.0.0", "observations": [GOOD]}, bundle)


def test_missing_features_and_unseen_categories_do_not_crash(bundle):
    rows = [rec(flow_bytes=None, flow_packets=None, packets_per_second=None, bytes_per_second=None, source_port=None),
            rec(protocol="icmp", connection_state="brand-new", destination_port=40404, direction="internal")]
    out = predict(rows, bundle, OPTS)
    assert "FEATURES_INCOMPLETE" in out[0].reason_codes
    assert "OUT_OF_DISTRIBUTION_WARNING" in out[1].reason_codes
    assert all(r.risk_score in range(101) for r in out)


def test_no_components_raises_model_unavailable_not_benign(bundle):
    empty = replace(bundle, classifier=None, anomaly=None)
    with pytest.raises(ModelUnavailableError) as ei:
        predict([GOOD], empty)
    assert ei.value.code == "MODEL_UNAVAILABLE"


class _Boom:
    classes_ = ["benign", "x"]
    def scores(self, X): raise RuntimeError("secret internals /srv/model/path")


def test_component_failure_policy_fail_is_controlled_and_leak_free(bundle):
    broken = replace(bundle, classifier=_Boom())
    with pytest.raises(ModelUnavailableError) as ei:
        predict([GOOD], broken, PredictionOptions(clock=fixed_clock))
    assert "secret" not in ei.value.message and "/srv" not in ei.value.message


def test_component_failure_policy_uncertain_never_benign(bundle):
    broken = replace(bundle, classifier=_Boom())
    out = predict([GOOD, GOOD], broken, PredictionOptions(clock=fixed_clock, component_failure_policy="uncertain"))
    assert all(r.prediction_type == "uncertain" and r.prediction_label == "unknown" and r.confidence is None
               and r.anomaly_score is None and r.reason_codes == ["MODEL_COMPONENT_UNAVAILABLE"] for r in out)


def test_classifier_only_and_anomaly_only_bundles_respect_policy(bundle, test_records):
    recs, _ = test_records
    c_only = replace(bundle, anomaly=None)
    out = predict(recs[:50], c_only, OPTS)
    assert all(r.anomaly_score is None and "MODEL_COMPONENT_UNAVAILABLE" in r.reason_codes for r in out)
    a_only = replace(bundle, classifier=None)
    out = predict(recs[:50], a_only, OPTS)
    assert all(r.prediction_type in ("anomaly", "uncertain") and r.prediction_label == "unknown" for r in out)


def test_prediction_result_model_rejects_inconsistent_values():
    base = dict(row_index=0, prediction_label="benign", prediction_type="benign", confidence=0.5, anomaly_score=None,
                risk_score=10, severity="info", reason_codes=[], model_version="m", feature_schema_version="1.0.0",
                prediction_timestamp="2026-10-25T12:00:00Z")
    PredictionResult(**base)
    for patch in ({"severity": "high"}, {"confidence": float("nan")}, {"confidence": 1.2}, {"risk_score": 101},
                  {"prediction_type": "known_attack"}, {"prediction_label": "Port Scan"}, {"reason_codes": ["MADE_UP"]},
                  {"prediction_timestamp": "2026-10-25 12:00:00"}, {"extra_field": 1}):
        with pytest.raises(Exception):
            PredictionResult(**{**base, **patch})
