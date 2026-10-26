"""CLI smoke tests on SYNTHETIC canonical CSVs. They prove the scripts run end-to-end and enforce their
guard rails; they do not (and cannot) say anything about real-dataset performance."""
import json
from pathlib import Path

import pandas as pd
import pytest

import continual_experiment, evaluate as evaluate_script, inspect_dataset, train as train_script, unknown_attack
from nids_ml import load_bundle
from nids_ml.lifecycle import LocalModelRegistry
from synthetic_fixture import make_frame

CFG = Path(__file__).resolve().parents[1] / "configs"
IDENT = str(CFG / "mappings" / "canonical_identity.json")


@pytest.fixture(scope="module")
def csv_path(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    df, y = make_frame(seed=21)
    df = df.assign(label=y.to_numpy()); df["observed_at"] = df["observed_at"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    df.loc[3, "duration"] = -1.0                       # one invalid row: must be dropped and counted
    p = d / "canonical.csv"; df.to_csv(p, index=False); return str(p)


@pytest.fixture(scope="module")
def small_config(tmp_path_factory):
    p = tmp_path_factory.mktemp("cfg") / "train.json"
    base = json.loads((CFG / "train_default.json").read_text())
    base["classifier"]["n_estimators"] = 30; base["anomaly"]["n_estimators"] = 30
    p.write_text(json.dumps(base)); return str(p)


@pytest.fixture(scope="module")
def gates(tmp_path_factory):
    """Bootstrap gates (no previous model exists, so no regression gate)."""
    p = tmp_path_factory.mktemp("g") / "gates.json"
    p.write_text(json.dumps({"minimum_samples": 1, "maximum_false_positive_rate": 1.0})); return str(p)


@pytest.fixture(scope="module")
def regression_gates(tmp_path_factory):
    p = tmp_path_factory.mktemp("g2") / "gates_reg.json"
    p.write_text(json.dumps({"minimum_samples": 1, "maximum_false_positive_rate": 1.0, "maximum_macro_f1_drop": 1.0})); return str(p)


def run_train(tmp_path, csv_path, small_config, version="base-1", extra=()):
    return train_script.main(["--csv", csv_path, "--mapping", IDENT, "--dataset-name", "synthetic-test-fixture", "--version", version,
                              "--artifact-dir", str(tmp_path / "art"), "--run-dir", str(tmp_path / "runs" / version),
                              "--config", small_config, *extra])


def test_inspect_reports_facts_without_modelling(tmp_path, csv_path):
    out = tmp_path / "inspect.json"
    assert inspect_dataset.main(["--csv", csv_path, "--mapping", IDENT, "--out", str(out)]) == 0
    r = json.loads(out.read_text())
    assert r["raw"]["rows_read"] == 2160 and any(c["name"] == "flow_bytes" and c["missing"] > 0 for c in r["raw"]["columns"])
    assert r["mapping"]["report"]["rows_dropped"] == 1 and r["mapping"]["report"]["dropped_by_reason"]["duration:OUT_OF_RANGE"] == 1
    assert abs(sum(v["share"] for v in r["class_distribution"].values()) - 1) < 1e-9
    assert "feature_vectors_with_conflicting_labels" in r["model_feature_duplicates"]


def test_inspect_without_mapping_works_on_any_csv(tmp_path):
    p = tmp_path / "x.csv"; pd.DataFrame({" a ": [1, 2, 2], "Label": ["x", "y", "y"]}).to_csv(p, index=False)
    out = tmp_path / "o.json"
    assert inspect_dataset.main(["--csv", str(p), "--out", str(out)]) == 0
    r = json.loads(out.read_text())["raw"]
    assert r["exact_duplicate_rows"] == 1 and [c["name"] for c in r["columns"]] == ["a", "Label"]


def test_train_end_to_end_writes_verifiable_artifact_and_reports(tmp_path, csv_path, small_config, capsys):
    assert run_train(tmp_path, csv_path, small_config) == 0
    sha = json.loads((tmp_path / "runs/base-1/training_summary.json").read_text())["artifact_sha256"]
    assert f"sha256={sha}" in capsys.readouterr().out
    b = load_bundle(tmp_path / "art", "base-1", expected_sha256=sha)
    rep = json.loads((tmp_path / "runs/base-1/evaluation_test.json").read_text())
    assert rep["artifact_sha256"] == sha and rep["dataset_name"] == "synthetic-test-fixture" and rep["split_strategy"] == "stratified"
    summ = json.loads((tmp_path / "runs/base-1/training_summary.json").read_text())
    assert summ["mapping_report"]["rows_dropped"] == 1 and "leakage_checks" in summ and b.metadata.thresholds.known_confidence > 0
    assert run_train(tmp_path, csv_path, small_config) == 2          # same version again: never overwritten


def test_train_refuses_unreviewed_mapping_and_creates_nothing(tmp_path, csv_path, small_config, capsys):
    rc = train_script.main(["--csv", csv_path, "--mapping", str(CFG / "mappings" / "unsw_nb15.draft.json"), "--dataset-name", "x",
                            "--version", "base-1", "--artifact-dir", str(tmp_path / "art"), "--run-dir", str(tmp_path / "r")])
    assert rc == 2 and "reviewed=false" in capsys.readouterr().err and not (tmp_path / "art").exists()


def test_train_promotion_needs_explicit_flags_and_a_passing_validation(tmp_path, csv_path, small_config, gates, capsys):
    reg = str(tmp_path / "reg.json")
    assert run_train(tmp_path, csv_path, small_config, "base-1", ["--registry", reg, "--gates", gates, "--promote"]) == 2     # no approver
    assert not (tmp_path / "art").exists()
    assert run_train(tmp_path, csv_path, small_config, "base-1", ["--registry", reg, "--gates", gates]) == 0
    assert LocalModelRegistry(reg).status("base-1") == "validated" and LocalModelRegistry(reg).active_version is None   # not promoted by default
    assert run_train(tmp_path, csv_path, small_config, "base-2", ["--registry", reg]) == 0                                 # no gates -> rejected
    assert LocalModelRegistry(reg).status("base-2") == "rejected"
    assert run_train(tmp_path, csv_path, small_config, "base-3", ["--registry", reg, "--gates", gates, "--promote", "--approved-by", "admin1"]) == 0
    assert LocalModelRegistry(reg).active_version == "base-3"


def test_evaluate_script_verifies_checksum_and_writes_report(tmp_path, csv_path, small_config):
    assert run_train(tmp_path, csv_path, small_config) == 0
    sha = json.loads((tmp_path / "runs/base-1/training_summary.json").read_text())["artifact_sha256"]
    args = ["--csv", csv_path, "--mapping", IDENT, "--dataset-name", "other-file", "--artifact-dir", str(tmp_path / "art"),
            "--version", "base-1", "--out", str(tmp_path / "e.json")]
    assert evaluate_script.main([*args, "--expected-sha256", "0" * 64]) == 2 and not (tmp_path / "e.json").exists()
    assert evaluate_script.main([*args, "--expected-sha256", sha]) == 0
    assert json.loads((tmp_path / "e.json").read_text())["split_strategy"] == "external_labelled_file"


def test_unknown_attack_script_output_states_its_limits(tmp_path, csv_path, small_config):
    out = tmp_path / "u.json"
    assert unknown_attack.main(["--csv", csv_path, "--mapping", IDENT, "--dataset-name", "synthetic-test-fixture", "--out", str(out),
                                "--held-out", "port_scan", "vol", "--family", "vol=dos_flood,exfiltration", "--config", small_config]) == 0
    r = json.loads(out.read_text())
    assert len(r["folds"]) == 2
    assert "unknown attacks in general" in r["what_this_does_not_establish"] and r["what_this_establishes"]
    assert unknown_attack.main(["--csv", csv_path, "--mapping", IDENT, "--dataset-name", "x", "--out", str(out), "--held-out", "benign"]) == 2


def test_continual_script_runs_a_cycle_without_promoting(tmp_path, csv_path, small_config, gates, regression_gates, capsys):
    reg = str(tmp_path / "reg.json")
    assert run_train(tmp_path, csv_path, small_config, "base-1", ["--registry", reg, "--gates", gates, "--promote", "--approved-by", "admin1"]) == 0
    fb = tmp_path / "fb.jsonl"
    df, y = make_frame(seed=77, sizes={"benign": 30, "port_scan": 30})
    lines = []
    for (_, r), lab in zip(df.iterrows(), y):
        obs = {k: (None if pd.isna(v) else (v.strftime("%Y-%m-%dT%H:%M:%SZ") if hasattr(v, "strftime") else (v.item() if hasattr(v, "item") else v)))
               for k, v in r.items()}
        lines.append(json.dumps({"observation": obs, "analyst_label": lab, "training_eligible": True,
                                 "reviewed_by": "analyst-1", "reviewed_at": "2026-10-20T09:00:00Z"}))
    fb.write_text("\n".join(lines))
    capsys.readouterr()
    base = ["--csv", csv_path, "--mapping", IDENT, "--dataset-name", "synthetic-test-fixture", "--artifact-dir", str(tmp_path / "art"),
            "--registry", reg, "--feedback", str(fb), "--gates", regression_gates, "--out-dir", str(tmp_path / "cycle"), "--min-replay-per-class", "10"]
    assert continual_experiment.main([*base, "--candidate-version", "cand-1"]) == 0
    out = capsys.readouterr().out
    assert "validation: PASSED" in out and "not promoted" in out
    r = LocalModelRegistry(reg)
    assert r.active_version == "base-1" and r.status("cand-1") == "validated"
    man = json.loads((tmp_path / "cycle/candidate_dataset_manifest.json").read_text())
    assert man["feedback_accepted"] > 0 and (tmp_path / "cycle/active_model_evaluation.json").exists()
    assert continual_experiment.main([*base, "--candidate-version", "cand-2", "--promote"]) == 2     # --promote needs --approved-by
    assert continual_experiment.main([*base, "--candidate-version", "cand-3", "--promote", "--approved-by", "admin1"]) == 0
    assert LocalModelRegistry(reg).active_version == "cand-3"
