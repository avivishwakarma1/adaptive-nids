"""Evaluation (contract section 9) and the held-out-class ("unknown attack") protocol.

Nothing here invents numbers: every metric is computed from the supplied labelled records, and is
`None` when mathematically undefined (e.g. FPR with no benign rows). Thresholds are read from the
bundle; evaluation never tunes them.
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from sklearn.metrics import average_precision_score, roc_auc_score

from .artifacts import ModelBundle, runtime_versions
from .contracts import (
    BENIGN_LABEL, UNKNOWN_LABEL, ClassMetrics, ConfigError, EvaluationConfig, EvaluationReport,
    InputValidationError, LabeledObservation, ModelUnavailableError, normalize_label, utc_now_iso,
)
from .features import build_model_frame, validate_features
from .inference import decide_row, run_components
from .training import (
    SplitConfig, TrainingConfig, dataset_fingerprint, deduplicate, fit_bundle, prepare_frames, split_indices,
)

METRIC_DEFINITIONS = {
    "accuracy": "share of rows whose final prediction_label equals the true label; 'unknown' counts as wrong",
    "precision_macro": "mean over true classes with support>0 of TP/(TP+FP); undefined (never predicted) counts as 0",
    "recall_macro": "mean over true classes with support>0 of TP/support",
    "f1_macro": "mean over true classes with support>0 of 2TP/(2TP+FP+FN)",
    "false_positive_rate": "binary FP/(FP+TN): true-benign rows whose prediction_type is not 'benign' (anomaly, "
                           "uncertain and known_attack all count as alerts needing review)",
}


def _f(x: Any) -> Optional[float]:
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _safe_div(a: float, b: float) -> Optional[float]:
    return None if b == 0 else float(a) / float(b)


# --------------------------------------------------------------------------- #
# Scoring helper shared by evaluate() and the unknown-attack protocol
# --------------------------------------------------------------------------- #
class _Scored:
    def __init__(self, ptype, plabel, conf, anom, raw_label, P):
        self.ptype, self.plabel, self.conf, self.anom, self.raw_label, self.P = ptype, plabel, conf, anom, raw_label, P


def score_model_frame(bundle: ModelBundle, model_frame: pd.DataFrame) -> _Scored:
    """Same decision policy as `predict` (shared `decide_row`), vectorised over a whole frame."""
    if not (bundle.classifier or bundle.anomaly):
        raise ModelUnavailableError("model bundle has no usable component")
    X = bundle.preprocessing.transform_frame(model_frame)
    raw_label, conf, P, anom = run_components(bundle, X)
    th = bundle.metadata.thresholds
    ptype, plabel = [], []
    for i in range(len(model_frame)):
        d = decide_row(
            cls_label=None if raw_label is None else str(raw_label[i]),
            confidence=None if conf is None else float(conf[i]),
            anomaly_score=None if anom is None else float(anom[i]),
            known_confidence=th.known_confidence, anomaly_threshold=th.anomaly_threshold,
            benign_label=bundle.metadata.benign_label, classifier_available=bundle.classifier is not None,
            anomaly_available=bundle.anomaly is not None, risk_policy=bundle.risk_policy,
        )
        ptype.append(d["prediction_type"]); plabel.append(d["prediction_label"])
    return _Scored(np.asarray(ptype, dtype=object), np.asarray(plabel, dtype=object), conf, anom, raw_label, P)


# --------------------------------------------------------------------------- #
# Metric primitives
# --------------------------------------------------------------------------- #
def confusion_from_labels(y_true: Sequence[str], y_pred: Sequence[str], true_labels: Sequence[str],
                          pred_labels: Sequence[str]) -> np.ndarray:
    ti = {l: i for i, l in enumerate(true_labels)}; pi = {l: i for i, l in enumerate(pred_labels)}
    M = np.zeros((len(true_labels), len(pred_labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        M[ti[t], pi[p]] += 1
    return M


def class_metrics_from_confusion(M: np.ndarray, true_labels: Sequence[str], pred_labels: Sequence[str]) -> list[ClassMetrics]:
    n = int(M.sum())
    out = []
    for i, lab in enumerate(true_labels):
        support = int(M[i].sum())
        j = list(pred_labels).index(lab) if lab in pred_labels else None
        tp = int(M[i, j]) if j is not None else 0
        col = int(M[:, j].sum()) if j is not None else 0
        fp, fn = col - tp, support - tp
        tn = n - support - fp
        out.append(ClassMetrics(
            label=lab, precision=_safe_div(tp, tp + fp), recall=_safe_div(tp, support),
            f1=_safe_div(2 * tp, 2 * tp + fp + fn), support=support,
            false_positive_rate=_safe_div(fp, fp + tn),
        ))
    return out


def macro(values: Sequence[Optional[float]], supports: Sequence[int]) -> Optional[float]:
    vs = [0.0 if v is None else v for v, s in zip(values, supports) if s > 0]
    return float(np.mean(vs)) if vs else None


def calibration_summary(conf: np.ndarray, correct: np.ndarray, n_bins: int) -> dict[str, Any]:
    """Reliability of the classifier's confidence. Reported as a diagnostic, never as a guarantee."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=False), 0, n_bins - 1)
    bins, ece = [], 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        mc, acc = float(conf[m].mean()), float(correct[m].mean())
        bins.append({"lower": float(edges[b]), "upper": float(edges[b + 1]), "count": int(m.sum()),
                     "mean_confidence": mc, "accuracy": acc})
        ece += m.mean() * abs(mc - acc)
    return {"expected_calibration_error": float(ece), "n_bins": n_bins, "bins": bins,
            "interpretation": "Large ECE means 'confidence' must not be read as a probability. Small ECE on this "
                              "set does not establish calibration on other data."}


# --------------------------------------------------------------------------- #
# evaluate()  (contract entry point)
# --------------------------------------------------------------------------- #
def _coerce_labeled(items: Any) -> tuple[list[dict], list[str]]:
    if isinstance(items, (str, bytes)) or not isinstance(items, Sequence) or len(items) == 0:
        raise InputValidationError("labeled_records must be a non-empty list")
    obs, labs = [], []
    for k, it in enumerate(items):
        if isinstance(it, Mapping):
            if set(it) != {"observation", "label"}:
                raise InputValidationError(f"labeled record {k} must have exactly 'observation' and 'label'")
            it = LabeledObservation(observation=dict(it["observation"]), label=it["label"])
        if not isinstance(it, LabeledObservation):
            raise InputValidationError(f"labeled record {k} has an unsupported type")
        try:
            labs.append(normalize_label(it.label))
        except ValueError:
            raise InputValidationError(f"labeled record {k} has an invalid label") from None
        obs.append(it.observation)
    return obs, labs


def evaluate(bundle: ModelBundle, labeled_records: Any, config: Optional[EvaluationConfig] = None) -> EvaluationReport:
    """Score `labeled_records` with `bundle` and report measured metrics. Does not tune anything."""
    cfg = config or EvaluationConfig()
    obs, labs = _coerce_labeled(labeled_records)
    rep = validate_features(obs, bundle.metadata.feature_schema_version, max_batch_size=None)
    rep.raise_if_invalid()
    mf = build_model_frame([r for r in rep.records if r is not None])
    sc = score_model_frame(bundle, mf)
    y = np.asarray(labs, dtype=object)
    n = len(y)
    warnings: list[str] = []

    true_labels = sorted(set(labs) | set(bundle.metadata.class_labels))
    pred_labels = sorted(set(true_labels) - {UNKNOWN_LABEL}) + [UNKNOWN_LABEL]
    M = confusion_from_labels(y, sc.plabel, true_labels, pred_labels)
    cms = class_metrics_from_confusion(M, true_labels, pred_labels)
    supports = [c.support for c in cms]
    acc = float(np.mean(sc.plabel == y))
    absent = sorted(set(labs) - set(bundle.metadata.class_labels))
    if absent and bundle.classifier is not None:
        warnings.append(f"true classes never seen by the classifier (recall is necessarily 0 for known-class labels): {absent}")
    small = [c.label for c in cms if 0 < c.support < cfg.min_class_support_warning]
    if small:
        warnings.append(f"classes with support < {cfg.min_class_support_warning}: {small}; their metrics are unstable")
    if BENIGN_LABEL not in set(labs):
        warnings.append("no benign rows in this evaluation set: false positive rate is undefined")

    is_attack = y != BENIGN_LABEL
    alert = sc.ptype != "benign"
    fp = int((alert & ~is_attack).sum()); tn = int((~alert & ~is_attack).sum())
    fpr = _safe_div(fp, fp + tn)

    type_by_truth: dict[str, dict[str, int]] = {}
    for lab in sorted(set(labs)):
        m = y == lab
        type_by_truth[lab] = {t: int((sc.ptype[m] == t).sum()) for t in ("known_attack", "benign", "anomaly", "uncertain")}

    anomaly_metrics = None
    if bundle.anomaly is not None and sc.anom is not None:
        thr = bundle.metadata.thresholds.anomaly_threshold
        flagged = sc.anom >= thr
        tp_a = int((flagged & is_attack).sum()); fp_a = int((flagged & ~is_attack).sum())
        fn_a = int((~flagged & is_attack).sum()); tn_a = int((~flagged & ~is_attack).sum())
        both = is_attack.any() and (~is_attack).any()
        pr = _safe_div(tp_a, tp_a + fp_a); rc = _safe_div(tp_a, tp_a + fn_a)
        anomaly_metrics = {
            "threshold": thr, "threshold_selection": bundle.metadata.threshold_selection.get("anomaly_threshold"),
            "roc_auc": _f(roc_auc_score(is_attack, sc.anom)) if both else None,
            "pr_auc": _f(average_precision_score(is_attack, sc.anom)) if both else None,
            "attack_prevalence": _safe_div(int(is_attack.sum()), n),
            "confusion": {"tp": tp_a, "fp": fp_a, "fn": fn_a, "tn": tn_a},
            "precision": pr, "recall": rc,
            "f1": _safe_div(2 * tp_a, 2 * tp_a + fp_a + fn_a),
            "false_positive_rate": _safe_div(fp_a, fp_a + tn_a), "alert_rate": float(flagged.mean()),
            "recall_by_attack_class": {c: _safe_div(int((flagged & (y == c)).sum()), int((y == c).sum()))
                                       for c in sorted(set(labs) - {BENIGN_LABEL})},
            "note": "positive = any non-benign label. PR-AUC baseline equals attack_prevalence. An anomaly "
                    "score is not an attack probability; benign novelty also scores high.",
        }
        if not both:
            warnings.append("anomaly ROC/PR-AUC undefined: evaluation set lacks benign or attack rows")

    closed = calib = None
    if bundle.classifier is not None and sc.raw_label is not None:
        in_model = np.isin(y, bundle.metadata.class_labels)
        if in_model.any():
            yt, yp = y[in_model], sc.raw_label[in_model]
            lbl = sorted(set(yt.tolist()) | set(yp.tolist()))
            Mc = confusion_from_labels(yt, yp, lbl, lbl)
            cmc = class_metrics_from_confusion(Mc, lbl, lbl)
            sup = [c.support for c in cmc]
            closed = {
                "n_rows": int(in_model.sum()), "n_rows_excluded_unknown_true_class": int((~in_model).sum()),
                "accuracy": float(np.mean(yt == yp)),
                "precision_macro": macro([c.precision for c in cmc], sup),
                "recall_macro": macro([c.recall for c in cmc], sup), "f1_macro": macro([c.f1 for c in cmc], sup),
                "note": "raw argmax, no confidence rejection; for comparing classifiers, not the operational output",
            }
            if BENIGN_LABEL in bundle.metadata.class_labels and sc.P is not None:
                bi = bundle.metadata.class_labels.index(BENIGN_LABEL)
                att_score = 1.0 - sc.P[:, bi]
                both = is_attack.any() and (~is_attack).any()
                closed["attack_score_roc_auc"] = _f(roc_auc_score(is_attack, att_score)) if both else None
                closed["attack_score_pr_auc"] = _f(average_precision_score(is_attack, att_score)) if both else None
            calib = calibration_summary(sc.conf[in_model], (yt == yp).astype(float), cfg.calibration_bins)

    fingerprint = cfg.dataset_fingerprint or dataset_fingerprint(mf, labs)
    return EvaluationReport(
        model_version=bundle.version, artifact_sha256=bundle.artifact_sha256,
        feature_schema_version=bundle.metadata.feature_schema_version, dataset_name=cfg.dataset_name,
        dataset_fingerprint=fingerprint, split_strategy=cfg.split_strategy, split_metadata=cfg.split_metadata,
        n_samples=n, class_support={c.label: c.support for c in cms if c.support > 0},
        thresholds={"known_confidence": bundle.metadata.thresholds.known_confidence,
                    "anomaly_threshold": bundle.metadata.thresholds.anomaly_threshold},
        risk_policy_version=bundle.metadata.risk_policy_version, package_versions=runtime_versions(),
        evaluated_at=utc_now_iso(cfg.clock() if cfg.clock else None),
        metrics={"accuracy": acc, "precision_macro": macro([c.precision for c in cms], supports),
                 "recall_macro": macro([c.recall for c in cms], supports),
                 "f1_macro": macro([c.f1 for c in cms], supports), "false_positive_rate": fpr},
        metric_definitions=METRIC_DEFINITIONS, class_metrics=cms, confusion_labels_true=list(true_labels),
        confusion_labels_pred=list(pred_labels), confusion_matrix=M.tolist(), type_by_truth=type_by_truth,
        alert_rate=float(alert.mean()), anomaly_metrics=anomaly_metrics, classifier_closed_set=closed,
        calibration=calib, warnings=warnings,
    )


# --------------------------------------------------------------------------- #
# Held-out attack class / family protocol
# --------------------------------------------------------------------------- #
ESTABLISHES = (
    "For this dataset, feature set, seed and thresholds: how often rows of a class the model never saw during "
    "training (classifier AND anomaly detector) still raise an alert (anomaly or uncertain), how often they are "
    "misattributed to a known class, and how often they pass as benign, next to the false-alarm rate on benign "
    "test traffic under the same decision policy."
)
DOES_NOT_ESTABLISH = (
    "That unknown attacks in general are detected. One held-out class (or family) from one dataset measures "
    "detection of that class under that distribution shift only. Classes in the same dataset share capture "
    "tooling, hosts and timing artefacts, so results may be optimistic or pessimistic for real novel attacks; "
    "family grouping reduces but cannot remove near-duplicate leakage. Results must not be compared across "
    "datasets, and benign novelty can raise the same alerts."
)


class UnknownAttackConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    held_out: list[str] = Field(min_length=1, description="class names or family names, one fold each")
    families: dict[str, list[str]] = Field(default_factory=dict)
    training: TrainingConfig
    min_heldout_support: int = Field(default=30, ge=1)


class UnknownAttackFold(BaseModel):
    model_config = ConfigDict(extra="forbid")
    held_out: str
    held_out_classes: list[str]
    n_heldout_rows: int
    n_benign_test_rows: int
    n_known_attack_test_rows: int
    heldout_prediction_types: dict[str, int]
    heldout_flagged_rate: Optional[float]
    heldout_anomaly_flag_rate: Optional[float]
    heldout_benign_pass_rate: Optional[float]
    heldout_misattributed_to: dict[str, int]
    benign_alert_rate: Optional[float]
    benign_anomaly_alert_rate: Optional[float]
    anomaly_roc_auc_heldout_vs_benign: Optional[float]
    anomaly_pr_auc_heldout_vs_benign: Optional[float]
    anomaly_pr_auc_baseline_prevalence: Optional[float]
    known_attack_macro_recall_control: Optional[float]
    thresholds: dict[str, float]
    trained_classes: list[str]
    warnings: list[str]


class UnknownAttackReport(BaseModel):
    model_config = ConfigDict(extra="forbid")
    protocol: str = "leave-one-class-or-family-out"
    what_this_establishes: str = ESTABLISHES
    what_this_does_not_establish: str = DOES_NOT_ESTABLISH
    dataset_fingerprint_full: str
    split_metadata: dict[str, Any]
    folds: list[UnknownAttackFold]
    package_versions: dict[str, str]


def run_unknown_attack_protocol(canonical: pd.DataFrame, labels: pd.Series, cfg: UnknownAttackConfig) -> UnknownAttackReport:
    """For each held-out class/family H: train on everything except H, evaluate on H + benign/known TEST rows.

    Thresholds are selected on validation data that contains no H rows. H rows are never used for
    fitting, normalisation or threshold selection in that fold.
    """
    mf_all, labs_all = prepare_frames(canonical, labels)
    keep, _ = deduplicate(mf_all, labs_all, cfg.training.split.dedupe)
    mf = mf_all.iloc[keep].reset_index(drop=True); labs = labs_all.iloc[keep].reset_index(drop=True)
    y = labs.to_numpy(dtype=object)
    sp = split_indices(labs.tolist(), cfg.training.split)
    present = set(y.tolist())
    folds: list[UnknownAttackFold] = []

    for name in cfg.held_out:
        classes = [normalize_label(c) for c in cfg.families.get(name, [name])]
        if BENIGN_LABEL in classes:
            raise ConfigError("benign cannot be held out")
        missing = [c for c in classes if c not in present]
        if missing:
            raise ConfigError(f"held-out classes not present in data: {missing}")
        is_h = np.isin(y, classes)
        tr = sp.train[~is_h[sp.train]]; va = sp.validation[~is_h[sp.validation]]; te = sp.test[~is_h[sp.test]]
        bundle = fit_bundle(
            mf.iloc[tr].reset_index(drop=True), y[tr], mf.iloc[va].reset_index(drop=True), y[va],
            cfg.training.model_copy(update={"model_version": f"{cfg.training.model_version}-ho-{normalize_label(name)}"[:60]}),
            split_metadata={"held_out": classes},
        )
        hrows = np.flatnonzero(is_h)
        sh = score_model_frame(bundle, mf.iloc[hrows].reset_index(drop=True))
        te_y = y[te]
        st = score_model_frame(bundle, mf.iloc[te].reset_index(drop=True))
        b = te_y == BENIGN_LABEL; ka = ~b
        warn: list[str] = []
        if len(hrows) < cfg.min_heldout_support:
            warn.append(f"only {len(hrows)} held-out rows (< {cfg.min_heldout_support}); rates are unstable")
        if b.sum() < cfg.min_heldout_support:
            warn.append(f"only {int(b.sum())} benign test rows; false-alarm rate is unstable")
        types = {t: int((sh.ptype == t).sum()) for t in ("known_attack", "benign", "anomaly", "uncertain")}
        mis: dict[str, int] = {}
        for lab in sh.plabel[sh.ptype == "known_attack"]:
            mis[str(lab)] = mis.get(str(lab), 0) + 1
        auc = prauc = prev = None
        if bundle.anomaly is not None and b.sum() > 0 and len(hrows) > 0:
            yy = np.concatenate([np.ones(len(hrows)), np.zeros(int(b.sum()))])
            ss = np.concatenate([sh.anom, st.anom[b]])
            auc, prauc, prev = _f(roc_auc_score(yy, ss)), _f(average_precision_score(yy, ss)), float(yy.mean())
        thr = bundle.metadata.thresholds
        ctl = None
        if ka.any():
            rec = [float(np.mean(st.plabel[te_y == c] == c)) for c in sorted(set(te_y[ka].tolist()))]
            ctl = float(np.mean(rec))
        nh = len(hrows)
        folds.append(UnknownAttackFold(
            held_out=name, held_out_classes=classes, n_heldout_rows=nh, n_benign_test_rows=int(b.sum()),
            n_known_attack_test_rows=int(ka.sum()), heldout_prediction_types=types,
            heldout_flagged_rate=_safe_div(nh - types["benign"], nh),
            heldout_anomaly_flag_rate=_safe_div(types["anomaly"], nh),
            heldout_benign_pass_rate=_safe_div(types["benign"], nh), heldout_misattributed_to=mis,
            benign_alert_rate=_safe_div(int((st.ptype[b] != "benign").sum()), int(b.sum())),
            benign_anomaly_alert_rate=_safe_div(int((st.ptype[b] == "anomaly").sum()), int(b.sum())),
            anomaly_roc_auc_heldout_vs_benign=auc, anomaly_pr_auc_heldout_vs_benign=prauc,
            anomaly_pr_auc_baseline_prevalence=prev, known_attack_macro_recall_control=ctl,
            thresholds={"known_confidence": thr.known_confidence, "anomaly_threshold": thr.anomaly_threshold},
            trained_classes=bundle.metadata.class_labels, warnings=warn,
        ))
    return UnknownAttackReport(
        dataset_fingerprint_full=dataset_fingerprint(mf, labs.tolist()), split_metadata=sp.metadata,
        folds=folds, package_versions=runtime_versions(),
    )
