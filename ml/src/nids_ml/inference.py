"""Prediction decision policy (contract section 6) and the public `predict` function.

Policy `decision-policy-1.0.0` (per row; thresholds come from the bundle, never from code):

 1. Classifier available -> (label, confidence). Anomaly available -> anomaly_score.
 2. confidence < known_confidence:
        type=uncertain, label=unknown, code KNOWN_CLASS_LOW_CONFIDENCE.
        If anomaly_score >= anomaly_threshold the row is explicitly PROMOTED to type=anomaly.
 3. confidence >= known_confidence and class is an attack:
        type=known_attack, label=<class>, code KNOWN_CLASS_HIGH_SCORE.
        Arbitration rule: a high anomaly score ADDS ANOMALY_* codes but never overwrites the label.
 4. confidence >= known_confidence and class is benign:
        anomaly_score >= anomaly_threshold -> type=anomaly, label=unknown (never "benign": a flow the
        baseline finds unusual must not be reported as confidently benign);
        otherwise type=benign, label=benign.
 5. Classifier missing (anomaly only): anomaly >= threshold -> anomaly/unknown, otherwise
        uncertain/unknown. A low anomaly score alone is not evidence of benign traffic.
 6. Both components missing -> ModelUnavailableError (MODEL_UNAVAILABLE). Never defaults to benign.
 ANOMALY_HIGH is emitted when anomaly_score >= threshold + (1 - threshold) / 2, else ANOMALY_ELEVATED.
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Optional, Sequence

import numpy as np

from .artifacts import ModelBundle
from .contracts import (
    BENIGN_LABEL, UNKNOWN_LABEL, InputValidationError, ModelUnavailableError, PredictionOptions,
    PredictionResult, SchemaVersionError, severity_for_score, utc_now_iso,
)
from .features import build_model_frame, incomplete_mask, validate_features
from .risk import RiskPolicy, compute_risk_score

DECISION_POLICY_VERSION = "decision-policy-1.0.0"
RISK_HIGH_THRESHOLD = 60  # severity high/critical => RISK_POLICY_HIGH


def decide_row(
    *, cls_label: Optional[str], confidence: Optional[float], anomaly_score: Optional[float],
    known_confidence: float, anomaly_threshold: float, benign_label: str,
    classifier_available: bool, anomaly_available: bool, risk_policy: RiskPolicy,
    incomplete: bool = False, out_of_distribution: bool = False,
) -> dict[str, Any]:
    """Pure decision function for one row. Returns the label/type/scores/risk/reason codes."""
    codes: list[str] = []
    anomalous = anomaly_available and anomaly_score is not None and anomaly_score >= anomaly_threshold

    def add_anomaly_code() -> None:
        high_cut = anomaly_threshold + (1.0 - anomaly_threshold) / 2.0
        codes.append("ANOMALY_HIGH" if anomaly_score >= high_cut else "ANOMALY_ELEVATED")

    if classifier_available:
        assert cls_label is not None and confidence is not None
        if confidence < known_confidence:
            codes.append("KNOWN_CLASS_LOW_CONFIDENCE")
            if anomalous:
                ptype, label = "anomaly", UNKNOWN_LABEL
                add_anomaly_code()
            else:
                ptype, label = "uncertain", UNKNOWN_LABEL
        elif cls_label == benign_label:
            if anomalous:
                ptype, label = "anomaly", UNKNOWN_LABEL
                add_anomaly_code()
            else:
                ptype, label = "benign", BENIGN_LABEL
        else:
            ptype, label = "known_attack", cls_label
            codes.append("KNOWN_CLASS_HIGH_SCORE")
            if anomalous:
                add_anomaly_code()
        if not anomaly_available:
            codes.append("MODEL_COMPONENT_UNAVAILABLE")
    else:  # anomaly-only model
        codes.append("MODEL_COMPONENT_UNAVAILABLE")
        if anomalous:
            ptype, label = "anomaly", UNKNOWN_LABEL
            add_anomaly_code()
        else:
            ptype, label = "uncertain", UNKNOWN_LABEL

    risk = compute_risk_score(ptype, label, confidence if classifier_available else None, anomaly_score, risk_policy)
    if incomplete:
        codes.append("FEATURES_INCOMPLETE")
    if out_of_distribution:
        codes.append("OUT_OF_DISTRIBUTION_WARNING")
    if risk >= RISK_HIGH_THRESHOLD:
        codes.append("RISK_POLICY_HIGH")
    return {
        "prediction_label": label, "prediction_type": ptype,
        "confidence": confidence if classifier_available else None,
        "anomaly_score": anomaly_score if anomaly_available else None,
        "risk_score": risk, "severity": severity_for_score(risk), "reason_codes": codes,
    }


def run_components(bundle: ModelBundle, X: np.ndarray):
    """Raw component outputs (labels, confidence, score matrix, anomaly) or raise on failure."""
    labels = conf = P = anom = None
    if bundle.classifier is not None:
        labels, conf, P = bundle.classifier.scores(X)
    if bundle.anomaly is not None:
        anom = bundle.anomaly.scores(X)
    for arr in (conf, anom):
        if arr is not None and not np.all(np.isfinite(arr)):
            raise ValueError("non-finite model output")
    return labels, conf, P, anom


def predict(records: Any, bundle: ModelBundle, options: Optional[PredictionOptions] = None) -> list[PredictionResult]:
    """Contract entry point. Output i corresponds to input i; length always equals input length.

    Raises: InputValidationError, SchemaVersionError, ModelUnavailableError. Never returns a
    fabricated 'benign' result when the model cannot run.
    """
    opts = options or PredictionOptions()
    meta = bundle.metadata
    if not (bundle.classifier or bundle.anomaly):
        raise ModelUnavailableError("model bundle has no usable component")

    report = validate_features(records, meta.feature_schema_version, max_batch_size=opts.max_batch_size,
                               extra_fields=opts.extra_fields)
    report.raise_if_invalid()
    rows: list[dict] = [r for r in report.records if r is not None]
    frame = build_model_frame(rows)
    X = bundle.preprocessing.transform_frame(frame)
    ood = bundle.preprocessing.unseen_category_mask(frame)
    inc = incomplete_mask(rows)
    now = utc_now_iso(opts.clock() if opts.clock else None)

    th = meta.thresholds
    common = dict(
        known_confidence=th.known_confidence, anomaly_threshold=th.anomaly_threshold,
        benign_label=meta.benign_label, classifier_available=bundle.classifier is not None,
        anomaly_available=bundle.anomaly is not None, risk_policy=bundle.risk_policy,
    )
    try:
        labels, conf, _, anom = run_components(bundle, X)
        failed = False
    except Exception:  # component failure: controlled failure or explicit uncertain, per policy
        if opts.component_failure_policy == "fail":
            raise ModelUnavailableError("a model component failed during inference") from None
        failed = True

    out: list[PredictionResult] = []
    for i in range(len(rows)):
        if failed:
            risk = compute_risk_score("uncertain", UNKNOWN_LABEL, None, None, bundle.risk_policy)
            body = {"prediction_label": UNKNOWN_LABEL, "prediction_type": "uncertain", "confidence": None,
                    "anomaly_score": None, "risk_score": risk, "severity": severity_for_score(risk),
                    "reason_codes": ["MODEL_COMPONENT_UNAVAILABLE"]}
        else:
            body = decide_row(
                cls_label=None if labels is None else str(labels[i]),
                confidence=None if conf is None else float(conf[i]),
                anomaly_score=None if anom is None else float(anom[i]),
                incomplete=bool(inc[i]), out_of_distribution=bool(ood[i]), **common,
            )
        out.append(PredictionResult(
            row_index=i, model_version=meta.model_version, feature_schema_version=meta.feature_schema_version,
            prediction_timestamp=now, **body,
        ))
    return out
