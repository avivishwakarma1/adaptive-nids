import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nids_ml.contracts import ConfigError
from nids_ml.features import ColumnMapping, apply_mapping, load_mapping

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "mappings"


def _mapping(**over):
    m = {
        "dataset_name": "toy-test-mapping", "reviewed": True, "label_column": "cat", "benign_labels": ["Normal"],
        "label_map": {"backdoors": "backdoor"},
        "fields": {
            "duration": {"column": "dur_us", "scale": 1e-6},
            "protocol": {"column": "proto", "value_map": {"tcp": "tcp", "udp": "udp"}, "default": "other"},
            "source_port": {}, "destination_port": {"column": "dport"},
            "flow_packets": {"sum_of": ["fp", "bp"]}, "flow_bytes": {"sum_of": ["fb", "bb"]},
            "packets_per_second": {"derive_rate_of": "flow_packets"}, "bytes_per_second": {"derive_rate_of": "flow_bytes"},
            "connection_state": {"column": "st"}, "direction": {"constant": "unknown"}, "observed_at": {},
        },
    }
    m.update(over); return ColumnMapping.model_validate(m)


def _raw():
    return pd.DataFrame({
        " dur_us": [2_000_000, 500_000, 0, -5, 1_000_000, 1_000_000], "proto": ["TCP", "gre", "udp", "tcp", "tcp", "tcp"],
        "dport": [80, 53, 22, 22, 8080, 8080], "fp": [10, 4, 1, 1, 2, 2], "bp": [5, 4, 1, 1, 2, 2],
        "fb": [1000, 400, 40, 40, 1, 1], "bb": [500, 400, 40, 40, 1, 1], "st": ["FIN", "CON", "INT", "INT", "REQ", "REQ"],
        "cat": ["Normal", "Backdoors", "Normal", "Normal", None, "  Normal "],
    })


def test_conversion_units_sums_value_maps_and_rates():
    out, labels, groups, rep = apply_mapping(_raw(), _mapping())
    assert groups is None
    r0 = out.iloc[0]
    assert r0["duration"] == pytest.approx(2.0) and r0["flow_packets"] == 15 and r0["flow_bytes"] == 1500
    assert r0["packets_per_second"] == pytest.approx(7.5) and r0["bytes_per_second"] == pytest.approx(750.0)
    assert out["protocol"].tolist()[:2] == ["tcp", "other"]           # unlisted protocol -> documented default
    assert out["connection_state"].iloc[0] == "fin" and out["direction"].iloc[0] == "unknown"
    assert labels.tolist()[:2] == ["benign", "backdoor"]               # benign label normalised, label_map applied


def test_zero_duration_gives_null_rates_not_infinity():
    out, *_ = apply_mapping(_raw(), _mapping())
    row = out[out["duration"] == 0.0].iloc[0]
    assert pd.isna(row["packets_per_second"]) and pd.isna(row["bytes_per_second"])


def test_invalid_rows_are_dropped_and_counted_never_repaired():
    out, labels, _, rep = apply_mapping(_raw(), _mapping())
    assert rep.rows_in == 6 and rep.rows_out == 4 and rep.rows_dropped == 2
    assert rep.dropped_by_reason["duration:OUT_OF_RANGE"] == 1 and rep.dropped_by_reason["label:MISSING_OR_UNUSABLE"] == 1
    assert (out["duration"] >= 0).all() and len(out) == len(labels)
    assert sum(rep.label_counts.values()) == rep.rows_out


def test_labels_are_separate_from_features():
    out, labels, _, _ = apply_mapping(_raw(), _mapping())
    assert "label" not in out.columns and "cat" not in out.columns
    assert set(out.columns) == {"duration", "protocol", "source_port", "destination_port", "flow_packets", "flow_bytes",
                                "packets_per_second", "bytes_per_second", "connection_state", "direction", "observed_at", "source_id"}


def test_infinite_values_become_null():
    raw = _raw(); raw["fb"] = [np.inf] + list(raw["fb"][1:])
    out, *_ = apply_mapping(raw, _mapping())
    assert pd.isna(out.iloc[0]["flow_bytes"]) or out.iloc[0]["flow_bytes"] == 500


def test_missing_column_in_header_is_a_config_error():
    with pytest.raises(ConfigError):
        apply_mapping(_raw().drop(columns=["dport"]), _mapping())
    with pytest.raises(ConfigError):
        apply_mapping(_raw().drop(columns=["cat"]), _mapping())


def test_mapping_file_validation():
    bad = json.loads(json.dumps(_mapping().model_dump(exclude_none=True)))
    del bad["fields"]["protocol"]
    with pytest.raises(ValueError):
        ColumnMapping.model_validate(bad)
    with pytest.raises(ValueError):
        ColumnMapping.model_validate({**_mapping().model_dump(), "fields": {**_mapping().model_dump()["fields"],
                                      "duration": {"column": "a", "constant": 1}}})


@pytest.mark.parametrize("name", ["unsw_nb15.draft.json", "cicids2017.draft.json", "ton_iot_network.draft.json"])
def test_draft_mappings_are_unreviewed_and_refused(name):
    assert json.loads((CONFIGS / name).read_text())["reviewed"] is False
    assert "UNVERIFIED" in json.loads((CONFIGS / name).read_text())["review_notes"]
    with pytest.raises(ConfigError, match="reviewed=false"):
        load_mapping(CONFIGS / name)


def test_identity_mapping_is_reviewed_and_loads():
    assert load_mapping(CONFIGS / "canonical_identity.json").reviewed
