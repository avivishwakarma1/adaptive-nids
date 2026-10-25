import math
from datetime import datetime, timezone

import numpy as np
import pytest

from nids_ml import InputValidationError, SchemaVersionError, validate_features
from nids_ml.features import MODEL_INPUT_COLUMNS, build_model_frame

GOOD = {
    "duration": 0.2, "protocol": "tcp", "source_port": 51515, "destination_port": 22, "flow_packets": 12,
    "flow_bytes": 1400, "packets_per_second": 60.0, "bytes_per_second": 7000.0, "connection_state": "S0",
    "direction": "unknown", "observed_at": "2026-10-24T12:30:00Z",
}


def rec(**kw):
    r = dict(GOOD); r.update(kw); return r


def test_contract_example_is_valid_and_normalized():
    rep = validate_features({"feature_schema_version": "1.0.0", "observations": [GOOD]})
    assert rep.valid and rep.n_records == 1
    r = rep.records[0]
    assert r["connection_state"] == "s0" and r["source_id"] is None
    assert r["observed_at"] == datetime(2026, 10, 24, 12, 30, tzinfo=timezone.utc)


@pytest.mark.parametrize("bad", [[], (), None, "abc", 5, b"x", {}, {"observations": []}])
def test_empty_or_wrong_container_raises(bad):
    with pytest.raises(InputValidationError):
        validate_features(bad)


def test_batch_size_bound():
    with pytest.raises(InputValidationError):
        validate_features([GOOD] * 4, max_batch_size=3)
    assert validate_features([GOOD] * 3, max_batch_size=3).valid


def test_schema_version_mismatch():
    with pytest.raises(SchemaVersionError):
        validate_features([GOOD], schema_version="2.0.0")
    with pytest.raises(SchemaVersionError):
        validate_features({"feature_schema_version": "0.9", "observations": [GOOD]})


@pytest.mark.parametrize("field,value,code", [
    ("duration", -1, "OUT_OF_RANGE"), ("duration", float("nan"), "NON_FINITE"), ("duration", float("inf"), "NON_FINITE"),
    ("duration", None, "NULL_NOT_ALLOWED"), ("duration", "1.0", "NOT_NUMERIC"), ("duration", True, "NOT_NUMERIC"),
    ("source_port", 70000, "OUT_OF_RANGE"), ("source_port", -1, "OUT_OF_RANGE"), ("destination_port", 22.5, "NOT_INTEGER"),
    ("flow_bytes", -5, "OUT_OF_RANGE"), ("flow_packets", float("inf"), "NON_FINITE"),
    ("protocol", "gre", "INVALID_ENUM"), ("direction", "sideways", "INVALID_ENUM"), ("protocol", 6, "NOT_A_STRING"),
    ("connection_state", "../../etc/passwd", "INVALID_FORMAT"), ("connection_state", "x" * 40, "INVALID_FORMAT"),
    ("observed_at", "2026-10-24T12:30:00", "TIMESTAMP_NOT_UTC"), ("observed_at", "yesterday", "INVALID_TIMESTAMP"),
    ("observed_at", 12345, "INVALID_TIMESTAMP"), ("source_id", ["a"], "INVALID_FORMAT"),
])
def test_invalid_values_are_rejected_without_echoing_input(field, value, code):
    rep = validate_features([rec(**{field: value})])
    assert not rep.valid and rep.records == [None]
    assert code in {i.code for i in rep.issues if i.field == field}
    assert str(value) not in " ".join(i.message for i in rep.issues) or str(value) in ("None", "True")
    with pytest.raises(InputValidationError):
        rep.raise_if_invalid()


def test_nullable_fields_accept_null_and_nan():
    rep = validate_features([rec(source_port=None, destination_port=None, flow_packets=None,
                                 flow_bytes=float("nan"), packets_per_second=None, bytes_per_second=None,
                                 observed_at=None)])
    assert rep.valid
    r = rep.records[0]
    assert r["flow_bytes"] is None and r["observed_at"] is None


def test_missing_required_fields_reported_per_field():
    r = rec(); del r["protocol"]; del r["duration"]
    rep = validate_features([r])
    assert {(i.field, i.code) for i in rep.issues} == {("protocol", "MISSING_FIELD"), ("duration", "MISSING_FIELD")}


def test_null_category_normalizes_to_unknown_with_warning():
    rep = validate_features([rec(protocol=None, direction="", connection_state=None)])
    assert rep.valid
    r = rep.records[0]
    assert (r["protocol"], r["direction"], r["connection_state"]) == ("unknown",) * 3
    assert {w.code for w in rep.warnings} == {"NORMALIZED_TO_UNKNOWN"}


def test_extra_fields_rejected_or_stripped_with_warning():
    assert not validate_features([rec(label="dos")]).valid
    rep = validate_features([rec(label="dos")], extra_fields="strip")
    assert rep.valid and "label" not in rep.records[0]
    assert rep.warnings[0].code == "FIELD_STRIPPED"


def test_non_mapping_row_and_row_alignment():
    rep = validate_features([GOOD, "oops", GOOD])
    assert rep.invalid_row_indices == [1]
    assert rep.records[0] is not None and rep.records[1] is None and rep.records[2] is not None


def test_issue_cap_sets_truncated_flag():
    rep = validate_features([rec(duration=-1)] * 50, max_issues=5)
    assert len(rep.issues) == 5 and rep.issues_truncated and not rep.valid


def test_numpy_scalars_accepted():
    rep = validate_features([rec(duration=np.float32(0.5), source_port=np.int64(80), flow_packets=np.int32(3))])
    assert rep.valid


def test_source_id_and_time_never_enter_model_frame():
    rep = validate_features([rec(source_id="flow-123")])
    frame = build_model_frame([rep.records[0]])
    assert list(frame.columns) == list(MODEL_INPUT_COLUMNS)
    assert "source_id" not in frame.columns and "observed_at" not in frame.columns and "source_port" not in frame.columns


def test_ports_are_bucketed_not_raw():
    rep = validate_features([rec(source_port=51515), rec(source_port=80), rec(source_port=None)])
    f = build_model_frame(rep.records)
    assert f["source_port_bucket"].tolist() == ["dynamic", "well_known", "missing"]
