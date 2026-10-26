import json

import numpy as np
import pandas as pd
import pytest

from nids_ml import EvaluationConfig, InputValidationError, PredictionOptions, evaluate, predict
from nids_ml.anomaly import AnomalyConfig
from nids_ml.classification import ClassifierConfig
from nids_ml.contracts import ConfigError, EvaluationReport
from nids_ml.evaluation import (UnknownAttackConfig, calibration_summary, class_metrics_from_confusion,
                                confusion_from_labels, macro, run_unknown_attack_protocol)
from nids_ml.features import build_model_frame, validate_features
from nids_ml.preprocessing import fit_preprocessing
from nids_ml.training import (SplitConfig, TrainingConfig, compare_classifier_candidates, dataset_fingerprint,
                              deduplicate, duplicate_report, fit_bundle, prepare_frames, split_indices,
                              train_bundle)
from conftest import fixed_clock, small_cfg
from synthetic_fixture import make_frame, make_records


# ----------------------------- splitting & leakage -----------------------------
def _disjoint_cover(sp, n):
    parts = [set(sp.train.tolist()), set(sp.validation.tolist()), set(sp.test.tolist())]
    assert not (parts[0] & parts[1]) and not (parts[0] & parts[2]) and not (parts[1] & parts[2])
    assert set().union(*parts) == set(range(n))


def test_stratified_split_is_disjoint_complete_and_seeded():
    _, y = make_frame()
    a = split_indices(y.tolist(), SplitConfig()); b = split_indices(y.tolist(), SplitConfig())
    _disjoint_cover(a, len(y))
    np.testing.assert_array_equal(a.train, b.train); np.testing.assert_array_equal(a.test, b.test)
    assert not np.array_equal(a.test, split_indices(y.tolist(), SplitConfig(seed=1)).test)
    for split in ("train", "validation", "test"):
        assert set(a.metadata["class_counts"][split]) == set(y)
        assert all(v > 0 for v in a.metadata["class_counts"][split].values())


def test_rare_class_goes_to_train_with_warning():
    y = ["benign"] * 50 + ["rare"] * 2
    sp = split_indices(y, SplitConfig())
    assert any("rare" in w for w in sp.metadata["warnings"])
    assert all(y[i] != "rare" for i in np.concatenate([sp.validation, sp.test]))


def test_time_split_is_chronological_and_needs_timestamps():
    df, y = make_frame()
    sp = split_indices(y.tolist(), SplitConfig(strategy="time"), timestamps=df["observed_at"].tolist())
    _disjoint_cover(sp, len(y))
    ts = df["observed_at"]
    assert ts.iloc[sp.train].max() <= ts.iloc[sp.validation].min() and ts.iloc[sp.validation].max() <= ts.iloc[sp.test].min()
    with pytest.raises(ConfigError):
        split_indices(y.tolist(), SplitConfig(strategy="time"))
    with pytest.raises(ConfigError):
        split_indices(y.tolist(), SplitConfig(strategy="time"), timestamps=[None] * len(y))


def test_group_split_never_shares_a_group_across_partitions():
    df, y = make_frame()
    groups = [f"day-{i // 50}" for i in range(len(y))]
    sp = split_indices(y.tolist(), SplitConfig(strategy="group"), groups=groups)
    _disjoint_cover(sp, len(y))
    g = np.asarray(groups)
    sets = [set(g[i]) for i in (sp.train, sp.validation, sp.test)]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
    with pytest.raises(ConfigError):
        split_indices(y.tolist(), SplitConfig(strategy="group"))


def test_split_fractions_must_sum_to_one():
    with pytest.raises(ValueError):
        SplitConfig(train_fraction=0.5, validation_fraction=0.2, test_fraction=0.2)


def test_duplicates_and_conflicts_are_found_and_removed():
    recs, labels = make_records(seed=3, sizes={"benign": 60, "port_scan": 30})
    recs = recs + [dict(recs[0]), dict(recs[1])]; labels = labels + [labels[0], "port_scan" if labels[1] == "benign" else "benign"]
    df = pd.DataFrame(recs); df["observed_at"] = pd.to_datetime(df["observed_at"], utc=True)
    mf, labs = prepare_frames(df, pd.Series(labels))
    rep = duplicate_report(mf, labs.tolist())
    assert rep["duplicate_rows_same_features_and_label"] >= 1 and rep["feature_vectors_with_conflicting_labels"] >= 1
    keep, info = deduplicate(mf, labs, "model_features")
    assert info["dropped"] >= 3 and info["dropped_conflicting"] >= 2
    assert duplicate_report(mf.iloc[keep], labs.iloc[keep].tolist())["duplicate_rows_same_features_any_label"] == 0


def test_train_bundle_fits_only_on_train_and_selects_thresholds_on_validation(outcome):
    sp, b = outcome.split, outcome.bundle
    mf, y = outcome.model_frame, outcome.labels
    assert b.preprocessing.fit_n_rows == len(sp.train)
    ref = fit_preprocessing(mf.iloc[sp.train], b.preprocessing.config)
    assert ref.describe()["numeric_medians_log1p_space"] == b.preprocessing.describe()["numeric_medians_log1p_space"]
    leaky = fit_preprocessing(mf, b.preprocessing.config)           # what leakage would have looked like
    assert leaky.describe()["numeric_medians_log1p_space"] != b.preprocessing.describe()["numeric_medians_log1p_space"]
    assert b.anomaly.fitted_on_n_ == int((y.iloc[sp.train] == "benign").sum())
    assert b.anomaly.reference_n_ == int((y.iloc[sp.validation] == "benign").sum())     # normalisation: validation only
    assert b.metadata.training_dataset_fingerprint == dataset_fingerprint(mf.iloc[sp.train], y.iloc[sp.train].tolist())
    assert b.metadata.split_metadata["sizes"]["test"] == len(sp.test)
    assert all(v == 0 for v in outcome.leakage["rows_in_split_also_in_train"].values())  # dedupe removed overlap


def test_anomaly_threshold_equals_validation_quantile_not_test(outcome):
    b, sp = outcome.bundle, outcome.split
    Xv = b.preprocessing.transform_frame(outcome.model_frame.iloc[sp.validation])
    ben = Xv[(outcome.labels.iloc[sp.validation] == "benign").to_numpy()]
    fpr = float((b.anomaly.scores(ben) >= b.metadata.thresholds.anomaly_threshold).mean())
    assert fpr <= b.anomaly.config.target_false_positive_rate + 1e-9
    assert "validation_benign_quantile" in b.metadata.threshold_selection["anomaly_threshold"]


def test_fingerprint_is_stable_and_sensitive():
    mf, y = prepare_frames(*make_frame(seed=5, sizes={"benign": 40, "port_scan": 20}))
    assert dataset_fingerprint(mf, y.tolist()) == dataset_fingerprint(mf.copy(), y.tolist())
    y2 = y.copy(); y2.iloc[0] = "port_scan" if y2.iloc[0] == "benign" else "benign"
    assert dataset_fingerprint(mf, y.tolist()) != dataset_fingerprint(mf, y2.tolist())


def test_training_requires_two_classes_and_benign_for_anomaly():
    df, y = make_frame(seed=2, sizes={"port_scan": 120, "dos_flood": 120})
    with pytest.raises(ConfigError):
        train_bundle(df, y, small_cfg("nobenign-1"))   # anomaly detector configured but no benign rows
    df, y = make_frame(seed=2, sizes={"benign": 150})
    with pytest.raises(ConfigError):
        train_bundle(df, y, small_cfg("one-class-1"))


def test_invalid_training_records_fail_closed():
    df, y = make_frame(seed=2, sizes={"benign": 60, "port_scan": 40})
    df.loc[0, "duration"] = -5
    with pytest.raises(InputValidationError):
        train_bundle(df, y, small_cfg("bad-1"))


def test_classifier_comparison_uses_validation_only(outcome):
    b, sp = outcome.bundle, outcome.split
    mf, y = outcome.model_frame, outcome.labels
    Xtr = b.preprocessing.transform_frame(mf.iloc[sp.train]); Xva = b.preprocessing.transform_frame(mf.iloc[sp.validation])
    rows = compare_classifier_candidates(Xtr, y.iloc[sp.train], Xva, y.iloc[sp.validation], {
        "rf": ClassifierConfig(n_estimators=20), "lr": ClassifierConfig(kind="logistic_regression", max_iter=100)})
    assert [r["name"] for r in rows] == ["rf", "lr"] and all(0 <= r["validation_macro_f1"] <= 1 for r in rows)


# ----------------------------- metric primitives (hand-computed) -----------------------------
def test_class_metrics_match_hand_calculation():
    # truth: benign x4, scan x3, dos x3 ; predictions given below
    t = ["benign"] * 4 + ["scan"] * 3 + ["dos"] * 3
    p = ["benign", "benign", "benign", "scan",   "scan", "scan", "unknown",   "dos", "benign", "unknown"]
    tl = ["benign", "dos", "scan"]; pl = ["benign", "dos", "scan", "unknown"]
    M = confusion_from_labels(t, p, tl, pl)
    assert M.tolist() == [[3, 0, 1, 0], [1, 1, 0, 1], [0, 0, 2, 1]]
    m = {c.label: c for c in class_metrics_from_confusion(M, tl, pl)}
    assert m["benign"].precision == pytest.approx(3 / 4) and m["benign"].recall == pytest.approx(3 / 4)
    assert m["benign"].false_positive_rate == pytest.approx(1 / 6)            # 1 FP / (1 FP + 5 TN)
    assert m["scan"].precision == pytest.approx(2 / 3) and m["scan"].recall == pytest.approx(2 / 3)
    assert m["scan"].f1 == pytest.approx(2 * 2 / (2 * 2 + 1 + 1))
    assert m["dos"].precision == pytest.approx(1.0) and m["dos"].recall == pytest.approx(1 / 3)
    assert macro([c.recall for c in m.values()], [c.support for c in m.values()]) == pytest.approx((3 / 4 + 1 / 3 + 2 / 3) / 3)


def test_undefined_metrics_are_none_not_zero():
    M = confusion_from_labels(["a", "a"], ["a", "a"], ["a", "b"], ["a", "b", "unknown"])
    m = {c.label: c for c in class_metrics_from_confusion(M, ["a", "b"], ["a", "b", "unknown"])}
    assert m["b"].support == 0 and m["b"].recall is None and m["b"].precision is None and m["b"].f1 is None
    assert m["a"].false_positive_rate is None   # no negatives in the data


def test_calibration_summary_perfect_and_overconfident():
    conf = np.array([0.9] * 10 + [0.1] * 10)
    ok = calibration_summary(conf, np.array([1.0] * 9 + [0.0] + [1.0] + [0.0] * 9), 10)
    assert ok["expected_calibration_error"] == pytest.approx(0.0)
    bad = calibration_summary(np.full(10, 0.99), np.array([1.0] * 5 + [0.0] * 5), 10)
    assert bad["expected_calibration_error"] == pytest.approx(0.49)


# ----------------------------- evaluate() -----------------------------
def _labeled(records, labels):
    return [{"observation": o, "label": l} for o, l in zip(records, labels)]


def test_evaluate_report_is_complete_json_safe_and_provenanced(bundle, test_records):
    recs, labs = test_records
    rep = evaluate(bundle, _labeled(recs, labs), EvaluationConfig(
        dataset_name="synthetic", split_strategy="stratified_holdout", split_metadata={"seed": 20260101}, clock=fixed_clock))
    assert isinstance(rep, EvaluationReport)
    json.dumps(rep.model_dump(mode="json"), allow_nan=False)
    assert rep.n_samples == len(recs) == sum(rep.class_support.values())
    assert rep.dataset_fingerprint and rep.model_version == bundle.version and rep.evaluated_at == "2026-10-25T12:00:00Z"
    assert set(rep.metrics) == {"accuracy", "precision_macro", "recall_macro", "f1_macro", "false_positive_rate"}
    assert set(rep.metric_definitions) == set(rep.metrics)
    assert {"scikit-learn", "numpy", "pandas", "nids_ml"} <= set(rep.package_versions)
    assert rep.thresholds["known_confidence"] == bundle.metadata.thresholds.known_confidence
    assert np.asarray(rep.confusion_matrix).sum() == len(recs)
    assert rep.anomaly_metrics["threshold_selection"] and rep.anomaly_metrics["pr_auc"] is not None
    assert rep.anomaly_metrics["attack_prevalence"] == pytest.approx(1 - labs.count("benign") / len(labs))
    assert rep.calibration["interpretation"] and rep.classifier_closed_set["n_rows"] == len(recs)
    assert rep.metrics["f1_macro"] is not None and 0 <= rep.metrics["f1_macro"] <= 1    # structural only, no value asserted


def test_evaluate_agrees_with_predict_on_types_and_labels(bundle, test_records):
    recs, labs = test_records
    rep = evaluate(bundle, _labeled(recs[:300], labs[:300]), EvaluationConfig(clock=fixed_clock))
    pred = predict(recs[:300], bundle)
    from collections import Counter
    by_type = Counter(p.prediction_type for p in pred)
    got = Counter()
    for d in rep.type_by_truth.values():
        got.update(d)
    assert {k: v for k, v in got.items() if v} == dict(by_type)


def test_evaluate_without_benign_rows_reports_undefined_fpr(bundle, test_records):
    recs, labs = test_records
    idx = [i for i, l in enumerate(labs) if l != "benign"][:40]
    rep = evaluate(bundle, _labeled([recs[i] for i in idx], [labs[i] for i in idx]))
    assert rep.metrics["false_positive_rate"] is None
    assert rep.anomaly_metrics["roc_auc"] is None and rep.anomaly_metrics["pr_auc"] is None
    assert any("no benign" in w for w in rep.warnings)


def test_evaluate_flags_classes_the_model_never_saw(bundle, test_records):
    recs, labs = test_records
    idx = [i for i, l in enumerate(labs) if l == "port_scan"][:20] + [i for i, l in enumerate(labs) if l == "benign"][:40]
    rep = evaluate(bundle, _labeled([recs[i] for i in idx], ["zero_day" if labs[i] == "port_scan" else labs[i] for i in idx]))
    assert rep.class_support["zero_day"] == 20
    assert any("never seen" in w and "zero_day" in w for w in rep.warnings)
    zd = next(c for c in rep.class_metrics if c.label == "zero_day")
    assert zd.recall == 0.0   # a label the classifier cannot emit can never be recalled


@pytest.mark.parametrize("bad", [[], "x", None, [{"observation": {}}], [{"observation": {}, "label": "a", "x": 1}], [5]])
def test_evaluate_rejects_malformed_labeled_input(bundle, bad):
    with pytest.raises(InputValidationError):
        evaluate(bundle, bad)


def test_evaluate_rejects_invalid_observation_and_label(bundle, test_records):
    recs, labs = test_records
    with pytest.raises(InputValidationError):
        evaluate(bundle, _labeled([dict(recs[0], duration=-1)], [labs[0]]))
    with pytest.raises(InputValidationError):
        evaluate(bundle, _labeled([recs[0]], ["!!!"]))


def test_evaluate_does_not_change_the_bundle(bundle, test_records):
    recs, labs = test_records
    before = (bundle.metadata.thresholds.model_copy(), bundle.metadata.metrics)
    evaluate(bundle, _labeled(recs[:100], labs[:100]))
    assert (bundle.metadata.thresholds, bundle.metadata.metrics) == before


# ----------------------------- unknown-attack protocol -----------------------------
def test_unknown_attack_protocol_structure_and_isolation():
    df, y = make_frame(seed=11)
    cfg = UnknownAttackConfig(held_out=["port_scan", "volumetric"], families={"volumetric": ["dos_flood", "exfiltration"]},
                              training=small_cfg("uk-1"), min_heldout_support=30)
    rep = run_unknown_attack_protocol(df, y, cfg)
    json.dumps(rep.model_dump(mode="json"), allow_nan=False)
    assert "unknown attacks in general" in rep.what_this_does_not_establish and rep.what_this_establishes
    by = {f.held_out: f for f in rep.folds}
    assert by["port_scan"].held_out_classes == ["port_scan"] and "port_scan" not in by["port_scan"].trained_classes
    assert set(by["volumetric"].held_out_classes) == {"dos_flood", "exfiltration"}
    assert not ({"dos_flood", "exfiltration"} & set(by["volumetric"].trained_classes))
    for f in rep.folds:
        t = f.heldout_prediction_types
        assert sum(t.values()) == f.n_heldout_rows
        assert f.heldout_flagged_rate == pytest.approx(1 - t["benign"] / f.n_heldout_rows)
        assert all(v is None or 0 <= v <= 1 for v in (f.benign_alert_rate, f.heldout_benign_pass_rate))
        assert f.anomaly_pr_auc_baseline_prevalence is not None


def test_unknown_attack_protocol_rejects_bad_requests():
    df, y = make_frame(seed=11)
    with pytest.raises(ConfigError):
        run_unknown_attack_protocol(df, y, UnknownAttackConfig(held_out=["benign"], training=small_cfg("uk-2")))
    with pytest.raises(ConfigError):
        run_unknown_attack_protocol(df, y, UnknownAttackConfig(held_out=["not_a_class"], training=small_cfg("uk-3")))


def test_unknown_attack_warns_on_small_support():
    df, y = make_frame(seed=11)
    cfg = UnknownAttackConfig(held_out=["exfiltration"], training=small_cfg("uk-4"), min_heldout_support=500)
    f = run_unknown_attack_protocol(df, y, cfg).folds[0]
    assert any("held-out rows" in w for w in f.warnings)


def test_training_is_reproducible_for_same_data_config_and_seed(synthetic, outcome, test_records):
    df, y = synthetic
    again = train_bundle(df, y, small_cfg())
    a, b = outcome.bundle, again.bundle
    assert a.metadata.thresholds == b.metadata.thresholds
    assert a.metadata.training_dataset_fingerprint == b.metadata.training_dataset_fingerprint
    assert a.metadata.transformed_feature_names == b.metadata.transformed_feature_names
    recs, _ = test_records
    opts = PredictionOptions(clock=fixed_clock)
    assert [p.model_dump() for p in predict(recs[:150], a, opts)] == [p.model_dump() for p in predict(recs[:150], b, opts)]
    different_seed = train_bundle(df, y, small_cfg().model_copy(update={"split": SplitConfig(seed=1)}))
    assert different_seed.bundle.metadata.training_dataset_fingerprint != a.metadata.training_dataset_fingerprint
