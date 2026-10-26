import json
from dataclasses import replace

import pandas as pd
import pytest

from nids_ml import (EvaluationConfig, LifecycleError, PromotionGates, evaluate, load_bundle, save_bundle,
                     validate_candidate)
from nids_ml.contracts import ClassMetrics
from nids_ml.features import build_model_frame, frame_to_records, validate_features
from nids_ml.lifecycle import (CandidateDatasetConfig, FeedbackExample, LocalModelRegistry,
                               build_candidate_training_set, run_candidate_cycle)
from conftest import fixed_clock, small_cfg
from synthetic_fixture import make_records

LENIENT = dict(minimum_samples=1, maximum_false_positive_rate=1.0)


def _labeled(recs, labs):
    return [{"observation": o, "label": l} for o, l in zip(recs, labs)]


@pytest.fixture()
def saved(bundle, tmp_path):
    b = replace(bundle); s = save_bundle(b, tmp_path / "art"); return b, s, tmp_path / "art"


@pytest.fixture()
def report(saved, test_records):
    b, _, _ = saved
    recs, labs = test_records
    return evaluate(b, _labeled(recs, labs), EvaluationConfig(dataset_name="synthetic", clock=fixed_clock))


def _with_class_recall(rep, **recalls):
    cms = [c.model_copy(update={"recall": recalls[c.label]}) if c.label in recalls else c for c in rep.class_metrics]
    return rep.model_copy(update={"class_metrics": cms})


# ------------------------------ validate_candidate ------------------------------
def test_no_quantitative_gate_means_failure(saved, report):
    r = validate_candidate(saved[0], report, PromotionGates(), clock=fixed_clock)
    assert not r.passed and any("at_least_one_quantitative_gate_configured" in x for x in r.reasons)


def test_lenient_gates_pass_and_result_is_bound_to_artifact(saved, report):
    b, s, _ = saved
    r = validate_candidate(b, report, PromotionGates(**LENIENT), clock=fixed_clock)
    assert r.passed and r.candidate_artifact_sha256 == s.sha256 and r.validated_at == "2026-10-25T12:00:00Z"
    assert r.evaluation_dataset_fingerprint == report.dataset_fingerprint
    assert {c.name for c in r.checks} >= {"report_bound_to_artifact", "feature_schema_compatible", "minimum_samples"}
    json.dumps(r.model_dump(mode="json"), allow_nan=False)


@pytest.mark.parametrize("patch,check", [
    ({"model_version": "other-9"}, "report_matches_candidate_version"),
    ({"artifact_sha256": "e" * 64}, "report_bound_to_artifact"),
    ({"artifact_sha256": None}, "report_bound_to_artifact"),
    ({"feature_schema_version": "9.9.9"}, "feature_schema_compatible"),
    ({"n_samples": 0}, "report_is_evaluated"),
    ({"evaluated": False}, "report_is_evaluated"),
    ({"dataset_fingerprint": ""}, "dataset_fingerprint_present"),
])
def test_structural_checks_block_validation(saved, report, patch, check):
    r = validate_candidate(saved[0], report.model_copy(update=patch), PromotionGates(**LENIENT))
    assert not r.passed and check in " ".join(r.reasons)


def test_unsaved_bundle_cannot_be_validated(bundle, report):
    unsaved = replace(bundle, artifact_sha256=None)
    r = validate_candidate(unsaved, report, PromotionGates(**LENIENT))
    assert not r.passed and "artifact_checksum_recorded" in " ".join(r.reasons)


def test_quality_gates_fail_when_not_met_and_when_not_evaluable(saved, report):
    b = saved[0]
    bad = report.model_copy(update={"metrics": {**report.metrics, "false_positive_rate": 0.30, "f1_macro": 0.50}})
    assert not validate_candidate(b, bad, PromotionGates(maximum_false_positive_rate=0.10)).passed
    assert not validate_candidate(b, bad, PromotionGates(minimum_macro_f1=0.60)).passed
    assert validate_candidate(b, bad, PromotionGates(minimum_macro_f1=0.40)).passed
    undefined = report.model_copy(update={"metrics": {**report.metrics, "false_positive_rate": None}})
    r = validate_candidate(b, undefined, PromotionGates(maximum_false_positive_rate=0.5))
    assert not r.passed and "not evaluated" in " ".join(r.reasons)


def test_minority_recall_gate_catches_what_aggregate_accuracy_hides(saved, report):
    b = saved[0]
    skewed = _with_class_recall(report.model_copy(update={"metrics": {**report.metrics, "accuracy": 0.99}}), exfiltration=0.10)
    r = validate_candidate(b, skewed, PromotionGates(minimum_per_class_recall=0.5))
    assert not r.passed and "exfiltration" in " ".join(r.reasons)


def test_attack_detection_recall_gate_uses_prediction_types(saved, report):
    b = saved[0]
    miss_all = report.model_copy(update={"type_by_truth": {k: ({"known_attack": 0, "benign": sum(v.values()), "anomaly": 0, "uncertain": 0}
                                                              if k != "benign" else v) for k, v in report.type_by_truth.items()}})
    assert not validate_candidate(b, miss_all, PromotionGates(minimum_recall=0.01)).passed
    assert validate_candidate(b, report, PromotionGates(minimum_recall=0.0)).passed


def test_regression_gates_require_a_comparable_baseline(saved, report):
    b = saved[0]
    g = dict(maximum_macro_f1_drop=0.05)
    assert not validate_candidate(b, report, PromotionGates(**g)).passed                                   # no baseline
    other = report.model_copy(update={"dataset_fingerprint": "different", "model_version": "prev-1"})
    r = validate_candidate(b, report, PromotionGates(**g, baseline_report=other))
    assert not r.passed and "baseline_report_comparable" in " ".join(r.reasons)                           # other eval set


def test_regression_gates_detect_drops_against_previous_model(saved, report):
    b = saved[0]
    prev = report.model_copy(update={"model_version": "prev-1", "metrics": {**report.metrics, "f1_macro": 0.90, "false_positive_rate": 0.02}})
    prev = _with_class_recall(prev, port_scan=0.95, brute_force=0.90)
    cand = report.model_copy(update={"metrics": {**report.metrics, "f1_macro": 0.80, "false_positive_rate": 0.10}})
    cand = _with_class_recall(cand, port_scan=0.95, brute_force=0.60)
    base = dict(baseline_report=prev)
    assert not validate_candidate(b, cand, PromotionGates(maximum_macro_f1_drop=0.05, **base)).passed
    assert validate_candidate(b, cand, PromotionGates(maximum_macro_f1_drop=0.15, **base)).passed
    r = validate_candidate(b, cand, PromotionGates(maximum_per_class_recall_drop=0.10, **base))
    assert not r.passed and "brute_force" in " ".join(r.reasons)
    assert not validate_candidate(b, cand, PromotionGates(maximum_false_positive_rate_increase=0.05, **base)).passed
    assert validate_candidate(b, cand, PromotionGates(maximum_false_positive_rate_increase=0.10, **base)).passed


def test_regression_gate_fails_if_a_class_disappears_from_candidate_report(saved, report):
    b = saved[0]
    prev = report.model_copy(update={"model_version": "prev-1"})
    cand = report.model_copy(update={"class_metrics": [c for c in report.class_metrics if c.label != "exfiltration"]})
    r = validate_candidate(b, cand, PromotionGates(maximum_per_class_recall_drop=0.5, baseline_report=prev))
    assert not r.passed


def test_gate_values_must_be_in_unit_interval():
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError):
            PromotionGates(minimum_recall=bad)


# ------------------------------ registry: promotion & rollback ------------------------------
def _bundle_named(bundle, name, tmp, schema=None):
    md = bundle.metadata.model_copy(update={"model_version": name, **({"feature_schema_version": schema} if schema else {})})
    b = replace(bundle, metadata=md); s = save_bundle(b, tmp); return b, s


def _validated(reg, b, s, report, passed=True, actor="t"):
    reg.register_candidate(b, s_ := s, actor=actor)
    rep = report.model_copy(update={"model_version": b.version, "artifact_sha256": s.sha256})
    res = validate_candidate(b, rep, PromotionGates(**(LENIENT if passed else {"minimum_macro_f1": 1.01 - 0.0001 if False else 0.9999999})), clock=fixed_clock)
    if not passed:
        res = validate_candidate(b, rep.model_copy(update={"metrics": {**rep.metrics, "f1_macro": 0.1}}), PromotionGates(minimum_macro_f1=0.9), clock=fixed_clock)
    reg.record_validation(res, actor=actor)
    return res


def test_full_state_machine_with_audit_and_persistence(bundle, report, tmp_path):
    reg = LocalModelRegistry(tmp_path / "reg.json", clock=fixed_clock)
    b1, s1 = _bundle_named(bundle, "m-1", tmp_path / "art"); b2, s2 = _bundle_named(bundle, "m-2", tmp_path / "art")
    r1 = _validated(reg, b1, s1, report)
    assert reg.status("m-1") == "validated" and reg.active_version is None
    reg.promote("m-1", validation=r1, expected_active_version=None, actor="admin", reason="bootstrap")
    r2 = _validated(reg, b2, s2, report)
    assert reg.active_version == "m-1"                                   # validation alone never changes the active model
    out = reg.promote("m-2", validation=r2, expected_active_version="m-1", actor="admin", reason="passed gates")
    assert out == {"active_model_version": "m-2", "previous_active_version": "m-1"}
    assert reg.status("m-1") == "retired" and reg.status("m-2") == "active"
    back = reg.rollback("m-1", expected_active_version="m-2", actor="admin", reason="regression observed")
    assert back["active_model_version"] == "m-1" and reg.status("m-2") == "retired"
    assert [e["event"] for e in reg.audit_log()] == ["candidate_created", "validation_passed", "promoted", "candidate_created",
                                                      "validation_passed", "promoted", "rolled_back"]
    again = LocalModelRegistry(tmp_path / "reg.json")                     # persisted state survives reload
    assert again.active_version == "m-1" and again.status("m-2") == "retired"
    loaded = again.load_active_bundle(tmp_path / "art")                   # checksum verified from the registry value
    assert loaded.version == "m-1" and loaded.artifact_sha256 == s1.sha256


def test_promotion_is_impossible_without_a_passing_validation(bundle, report, tmp_path):
    reg = LocalModelRegistry(clock=fixed_clock)
    b, s = _bundle_named(bundle, "m-1", tmp_path)
    reg.register_candidate(b, s, actor="t")
    rep = report.model_copy(update={"model_version": "m-1", "artifact_sha256": s.sha256})
    ok = validate_candidate(b, rep, PromotionGates(**LENIENT))
    with pytest.raises(LifecycleError):                                   # still 'candidate': validation not recorded
        reg.promote("m-1", validation=ok, expected_active_version=None, actor="a", reason="x")
    failing = validate_candidate(b, rep, PromotionGates())
    assert reg.record_validation(failing, actor="t") == "rejected"
    with pytest.raises(LifecycleError):
        reg.promote("m-1", validation=ok, expected_active_version=None, actor="a", reason="x")   # rejected stays rejected
    assert reg.active_version is None


def test_promote_guards_stale_state_reason_and_wrong_artifact(bundle, report, tmp_path):
    reg = LocalModelRegistry(clock=fixed_clock)
    b1, s1 = _bundle_named(bundle, "m-1", tmp_path); b2, s2 = _bundle_named(bundle, "m-2", tmp_path)
    r1 = _validated(reg, b1, s1, report); reg.promote("m-1", validation=r1, expected_active_version=None, actor="a", reason="ok")
    r2 = _validated(reg, b2, s2, report)
    with pytest.raises(LifecycleError):
        reg.promote("m-2", validation=r2, expected_active_version=None, actor="a", reason="stale")
    with pytest.raises(LifecycleError):
        reg.promote("m-2", validation=r2, expected_active_version="m-1", actor="a", reason="  ")
    with pytest.raises(LifecycleError):                                   # a passing result for a DIFFERENT artifact
        reg.promote("m-2", validation=r1, expected_active_version="m-1", actor="a", reason="swap")
    forged = r2.model_copy(update={"candidate_artifact_sha256": "0" * 64})
    with pytest.raises(LifecycleError):
        reg.promote("m-2", validation=forged, expected_active_version="m-1", actor="a", reason="forged")
    assert reg.active_version == "m-1"


def test_candidate_schema_must_match_active_to_promote_or_rollback(bundle, report, tmp_path):
    reg = LocalModelRegistry(clock=fixed_clock)
    b1, s1 = _bundle_named(bundle, "m-1", tmp_path); b2, s2 = _bundle_named(bundle, "m-2", tmp_path)
    r1 = _validated(reg, b1, s1, report); reg.promote("m-1", validation=r1, expected_active_version=None, actor="a", reason="ok")
    r2 = _validated(reg, b2, s2, report)
    reg.state["entries"]["m-2"]["feature_schema_version"] = "2.0.0"
    with pytest.raises(LifecycleError):
        reg.promote("m-2", validation=r2, expected_active_version="m-1", actor="a", reason="schema")
    reg.state["entries"]["m-2"]["feature_schema_version"] = "1.0.0"
    reg.promote("m-2", validation=r2, expected_active_version="m-1", actor="a", reason="ok")
    reg.state["entries"]["m-1"]["feature_schema_version"] = "2.0.0"
    with pytest.raises(LifecycleError):
        reg.rollback("m-1", expected_active_version="m-2", actor="a", reason="incompatible")
    assert reg.active_version == "m-2"


def test_rollback_guards(bundle, report, tmp_path):
    reg = LocalModelRegistry(clock=fixed_clock)
    b1, s1 = _bundle_named(bundle, "m-1", tmp_path); b2, s2 = _bundle_named(bundle, "m-2", tmp_path); b3, s3 = _bundle_named(bundle, "m-3", tmp_path)
    r1 = _validated(reg, b1, s1, report); reg.promote("m-1", validation=r1, expected_active_version=None, actor="a", reason="ok")
    r2 = _validated(reg, b2, s2, report); reg.promote("m-2", validation=r2, expected_active_version="m-1", actor="a", reason="ok")
    _validated(reg, b3, s3, report, passed=False)
    assert reg.status("m-3") == "rejected"
    for kwargs in (dict(target_version="m-3", expected_active_version="m-2", reason="to rejected"),
                   dict(target_version="m-1", expected_active_version="m-1", reason="stale"),
                   dict(target_version="m-2", expected_active_version="m-2", reason="already active"),
                   dict(target_version="m-1", expected_active_version="m-2", reason=""),
                   dict(target_version="nope-1", expected_active_version="m-2", reason="unknown")):
        with pytest.raises(LifecycleError):
            reg.rollback(actor="a", **kwargs)
    assert reg.active_version == "m-2"


def test_register_rules(bundle, tmp_path):
    reg = LocalModelRegistry(clock=fixed_clock)
    b, s = _bundle_named(bundle, "m-1", tmp_path)
    reg.register_candidate(b, s, actor="t")
    with pytest.raises(LifecycleError):
        reg.register_candidate(b, s, actor="t")
    b2, s2 = _bundle_named(bundle, "m-2", tmp_path)
    with pytest.raises(LifecycleError):
        reg.register_candidate(b2, s, actor="t")     # checksum belongs to another artifact


# ------------------------------ reviewed feedback -> candidate dataset ------------------------------
def _fb(rec, label, **kw):
    base = dict(observation=rec, analyst_label=label, training_eligible=True, reviewed_by="analyst-1", reviewed_at="2026-10-20T09:00:00Z")
    base.update(kw); return FeedbackExample(**base)


@pytest.fixture()
def base_sets(outcome):
    sp, mf, y = outcome.split, outcome.model_frame, outcome.labels
    return (mf.iloc[sp.train].reset_index(drop=True), y.iloc[sp.train].tolist(),
            mf.iloc[sp.validation].reset_index(drop=True), mf.iloc[sp.test].reset_index(drop=True))


def _fresh(n=6, seed=99):
    recs, labs = make_records(seed=seed, sizes={"benign": n, "port_scan": n})
    return recs, labs


def test_only_reviewed_eligible_valid_feedback_is_admitted(base_sets):
    trf, trl, vaf, tef = base_sets
    recs, labs = _fresh(4)
    fb = [_fb(recs[0], labs[0]), _fb(recs[1], labs[1], training_eligible=False), _fb(recs[2], labs[2], reviewed_by=None),
          _fb(recs[3], labs[3], reviewed_at=None), _fb(recs[4], "totally_new_class"), _fb(dict(recs[5], duration=-1), labs[5]),
          _fb(recs[6], "!!!")]
    ds = build_candidate_training_set(trf, trl, fb, CandidateDatasetConfig(allowed_labels=sorted(set(trl)), min_replay_per_class=0),
                                      exclude_frames=[vaf, tef])
    m = ds.manifest
    assert m["feedback_received"] == 7 and m["feedback_accepted"] == 1
    assert m["feedback_dropped_by_reason"] == {"not_training_eligible": 1, "missing_review_record": 2, "label_not_in_allowed_set": 1,
                                               "invalid_observation": 1, "invalid_label": 1}
    assert len(ds.frame) == len(trf) + 1 and len(ds.frame) == len(ds.labels)


def test_duplicates_conflicts_and_holdout_overlap_are_removed(base_sets, outcome):
    trf, trl, vaf, tef = base_sets
    recs, labs = _fresh(4)
    canon, lab = outcome.canonical_frame, outcome.labels
    held_rec = frame_to_records(canon.iloc[outcome.split.test[:1]])[0]
    r_same, r_flip = (frame_to_records(canon.iloc[outcome.split.train[i:i + 1]])[0] for i in (0, 1))
    l_same, l_flip = lab.iloc[outcome.split.train[0]], lab.iloc[outcome.split.train[1]]
    flipped = "port_scan" if l_flip != "port_scan" else "benign"
    fb = [_fb(recs[0], "benign"), _fb(recs[0], "benign"),                        # exact duplicate
          _fb(recs[1], "benign"), _fb(recs[1], "port_scan"),                     # contradictory labels
          _fb(held_rec, "benign"),                                               # same features as a holdout row
          _fb(r_same, l_same),                                                   # already in training data, same label
          _fb(r_flip, flipped)]                                                  # contradicts the label in training data
    ds = build_candidate_training_set(trf, trl, fb, CandidateDatasetConfig(allowed_labels=sorted(set(trl)), min_replay_per_class=0),
                                      exclude_frames=[vaf, tef])
    r = ds.manifest["feedback_dropped_by_reason"]
    assert r["duplicate_feedback"] == 1 and r["conflicting_feedback_labels"] == 2 and r["overlaps_validation_or_holdout"] == 1
    assert r["already_in_training_data"] == 1 and r["contradicts_training_label_needs_adjudication"] == 1
    assert ds.manifest["feedback_accepted"] == 1          # only the first of the duplicate pair survives


def test_replay_is_deterministic_stratified_and_protects_minorities(base_sets):
    trf, trl, vaf, tef = base_sets
    cfg = CandidateDatasetConfig(allowed_labels=sorted(set(trl)), replay_fraction=0.2, min_replay_per_class=25, seed=5)
    a = build_candidate_training_set(trf, trl, [], cfg); b = build_candidate_training_set(trf, trl, [], cfg)
    assert a.manifest["candidate_dataset_fingerprint"] == b.manifest["candidate_dataset_fingerprint"]
    counts = a.manifest["class_counts"]
    assert set(counts) == set(trl) and all(v >= min(25, trl.count(c)) for c, v in counts.items())
    assert counts["benign"] < trl.count("benign")                                # actually sub-sampled
    c = build_candidate_training_set(trf, trl, [], cfg.model_copy(update={"seed": 6}))
    assert c.manifest["candidate_dataset_fingerprint"] != a.manifest["candidate_dataset_fingerprint"]


def test_feedback_share_can_be_capped(base_sets):
    trf, trl, vaf, tef = base_sets
    recs, labs = make_records(seed=123, sizes={"benign": 60, "port_scan": 60})
    fb = [_fb(r, l) for r, l in zip(recs, labs)]
    cfg = CandidateDatasetConfig(allowed_labels=sorted(set(trl)), min_replay_per_class=0, max_feedback_fraction=0.05)
    ds = build_candidate_training_set(trf, trl, fb, cfg, exclude_frames=[vaf, tef])
    n_fb = ds.manifest["feedback_accepted"]
    assert 0 < n_fb <= 0.05 / 0.95 * ds.manifest["replay_rows"] + 1
    assert ds.manifest["feedback_dropped_by_reason"]["exceeds_max_feedback_fraction"] > 0


def test_unreviewed_feedback_by_default_changes_nothing(base_sets):
    trf, trl, vaf, tef = base_sets
    recs, labs = _fresh(5)
    fb = [FeedbackExample(observation=r, analyst_label=l) for r, l in zip(recs, labs)]      # defaults: training_eligible=False
    ds = build_candidate_training_set(trf, trl, fb, CandidateDatasetConfig(allowed_labels=sorted(set(trl)), min_replay_per_class=0))
    assert ds.manifest["feedback_accepted"] == 0 and len(ds.frame) == len(trf)


# ------------------------------ whole cycle ------------------------------
def test_candidate_cycle_never_promotes_and_supports_explicit_promotion_and_rollback(outcome, bundle, tmp_path):
    art = tmp_path / "art"; reg = LocalModelRegistry(tmp_path / "reg.json", clock=fixed_clock)
    active = replace(bundle); s0 = save_bundle(active, art)
    holdout_recs, holdout_labs = outcome.subset("test")
    rep0 = evaluate(active, _labeled(holdout_recs, holdout_labs), EvaluationConfig(dataset_name="syn", clock=fixed_clock))
    reg.register_candidate(active, s0, actor="boot")
    v0 = validate_candidate(active, rep0, PromotionGates(**LENIENT), clock=fixed_clock)
    reg.record_validation(v0, actor="boot")
    reg.promote(active.version, validation=v0, expected_active_version=None, actor="admin", reason="bootstrap")

    sp, mf, y = outcome.split, outcome.model_frame, outcome.labels
    recs, labs = make_records(seed=555, sizes={"benign": 40, "port_scan": 40, "brute_force": 20})
    fb = [_fb(r, l) for r, l in zip(recs, labs)]
    kw = dict(active_bundle=active, registry=reg, artifact_dir=art,
              base_train_frame=mf.iloc[sp.train].reset_index(drop=True), base_train_labels=y.iloc[sp.train].tolist(),
              validation_frame=mf.iloc[sp.validation].reset_index(drop=True), validation_labels=y.iloc[sp.validation].tolist(),
              holdout_records=holdout_recs, holdout_labels=holdout_labs, holdout_dataset_name="syn:holdout",
              feedback=fb, candidate_dataset_cfg=CandidateDatasetConfig(allowed_labels=sorted(set(y)), replay_fraction=0.5, min_replay_per_class=20),
              actor="exp", clock=fixed_clock)

    res = run_candidate_cycle(training_cfg=small_cfg("cand-1"), gates=PromotionGates(**LENIENT, maximum_macro_f1_drop=1.0), **kw)
    assert reg.active_version == active.version, "running a cycle must never change the active model"
    assert res.saved.directory.name == "cand-1" and (art / active.version).exists()           # separate artifact, nothing overwritten
    assert res.candidate_report.dataset_fingerprint == res.baseline_report.dataset_fingerprint
    assert res.candidate_report.artifact_sha256 == res.saved.sha256
    assert res.validation.baseline_version == active.version and reg.status("cand-1") == "validated"
    assert res.dataset_manifest["feedback_accepted"] > 0

    bad = run_candidate_cycle(training_cfg=small_cfg("cand-2"), gates=PromotionGates(), **kw)   # no gates configured
    assert not bad.validation.passed and reg.status("cand-2") == "rejected"
    with pytest.raises(LifecycleError):
        reg.promote("cand-2", validation=bad.validation, expected_active_version=active.version, actor="admin", reason="try")

    reg.promote("cand-1", validation=res.validation, expected_active_version=active.version, actor="admin", reason="gates passed")
    assert reg.active_version == "cand-1" and reg.load_active_bundle(art).version == "cand-1"
    reg.rollback(active.version, expected_active_version="cand-1", actor="admin", reason="drill")
    assert reg.load_active_bundle(art).version == active.version
