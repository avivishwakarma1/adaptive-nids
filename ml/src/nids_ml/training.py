"""Offline training: split -> fit preprocessing on TRAIN only -> fit models -> select thresholds on VALIDATION.

The held-out test split is never touched here (no fitting, no threshold selection, no model
comparison). Evaluation of the test split happens in `evaluation.evaluate`.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Optional, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sklearn.metrics import f1_score, recall_score

from .anomaly import AnomalyConfig, AnomalyDetector
from .artifacts import BundleMetadata, ModelBundle, Thresholds, check_bundle_consistency, runtime_versions
from .classification import ClassifierConfig, KnownClassClassifier
from .contracts import (
    BENIGN_LABEL, SCHEMA_VERSION, ConfigError, InputValidationError, normalize_label, validate_version_string,
)
from .features import MODEL_INPUT_COLUMNS, build_model_frame, frame_to_records, validate_features
from .preprocessing import PreprocessingConfig, fit_preprocessing
from .risk import RiskPolicy


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
class SplitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    strategy: Literal["stratified", "time", "group"] = "stratified"
    train_fraction: float = Field(default=0.6, gt=0, lt=1)
    validation_fraction: float = Field(default=0.2, gt=0, lt=1)
    test_fraction: float = Field(default=0.2, gt=0, lt=1)
    seed: int = 20260101
    dedupe: Literal["model_features", "none"] = "model_features"

    @model_validator(mode="after")
    def _sum(self) -> "SplitConfig":
        if abs(self.train_fraction + self.validation_fraction + self.test_fraction - 1.0) > 1e-9:
            raise ValueError("split fractions must sum to 1")
        return self


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_version: str
    split: SplitConfig = Field(default_factory=SplitConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    classifier: Optional[ClassifierConfig] = Field(default_factory=ClassifierConfig)
    anomaly: Optional[AnomalyConfig] = Field(default_factory=AnomalyConfig)
    risk_policy: RiskPolicy = Field(default_factory=RiskPolicy)
    # known_confidence is selected on validation: smallest t whose accepted predictions reach this accuracy.
    target_accepted_accuracy: float = Field(default=0.95, gt=0, le=1)
    min_validation_rows: int = Field(default=50, ge=1)
    max_train_rows: Optional[int] = Field(default=None, ge=100)  # deterministic subsample for fast experiments


# --------------------------------------------------------------------------- #
# Fingerprint / duplicates / leakage
# --------------------------------------------------------------------------- #
def dataset_fingerprint(model_frame: pd.DataFrame, labels: Sequence[str]) -> str:
    """SHA-256 over model-input columns + labels, in row order (order-sensitive on purpose)."""
    h = hashlib.sha256()
    h.update(("|".join(model_frame.columns) + f"|n={len(model_frame)}\n").encode())
    df = model_frame.copy()
    df["__label__"] = list(labels)
    for start in range(0, len(df), 50_000):
        h.update(df.iloc[start:start + 50_000].to_csv(index=False, header=False, float_format="%.17g").encode())
    return h.hexdigest()


def _row_hashes(model_frame: pd.DataFrame) -> np.ndarray:
    return pd.util.hash_pandas_object(model_frame, index=False).to_numpy()


def duplicate_report(model_frame: pd.DataFrame, labels: Sequence[str]) -> dict[str, Any]:
    """Exact duplicates on model-input columns, and feature-identical rows with conflicting labels."""
    h = pd.Series(_row_hashes(model_frame))
    lab = pd.Series(list(labels))
    dup_feature = int(h.duplicated(keep="first").sum())
    dup_full = int(pd.DataFrame({"h": h, "l": lab}).duplicated(keep="first").sum())
    nun = pd.DataFrame({"h": h, "l": lab}).groupby("h")["l"].nunique()
    return {
        "n_rows": int(len(model_frame)),
        "duplicate_rows_same_features_and_label": dup_full,
        "duplicate_rows_same_features_any_label": dup_feature,
        "feature_vectors_with_conflicting_labels": int((nun > 1).sum()),
    }


def deduplicate(model_frame: pd.DataFrame, labels: pd.Series, mode: str) -> tuple[np.ndarray, dict[str, Any]]:
    """Return indices to keep. Feature-identical rows with conflicting labels are dropped entirely
    (their label is not trustworthy for this feature set)."""
    n = len(model_frame)
    if mode == "none":
        return np.arange(n), {"mode": "none", "dropped": 0, "dropped_conflicting": 0}
    h = pd.Series(_row_hashes(model_frame))
    lab = pd.Series(labels.to_numpy())
    df = pd.DataFrame({"h": h, "l": lab})
    conflicting = df.groupby("h")["l"].transform("nunique") > 1
    dup = df.duplicated(keep="first")
    keep = ~(dup | conflicting)
    return np.flatnonzero(keep.to_numpy()), {
        "mode": mode, "dropped": int((~keep).sum()),
        "dropped_conflicting": int(conflicting.sum()), "dropped_duplicates": int((dup & ~conflicting).sum()),
    }


def cross_split_overlap(frames: dict[str, pd.DataFrame]) -> dict[str, int]:
    """Number of rows in each later split whose feature vector also appears in TRAIN (leakage check)."""
    train_h = set(_row_hashes(frames["train"]).tolist())
    return {name: int(sum(h in train_h for h in _row_hashes(f).tolist()))
            for name, f in frames.items() if name != "train"}


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
@dataclass
class SplitIndices:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


def split_indices(labels: Sequence[str], cfg: SplitConfig, *, groups: Optional[Sequence[Any]] = None,
                  timestamps: Optional[Sequence[Any]] = None) -> SplitIndices:
    y = np.asarray(list(labels), dtype=object)
    n = len(y)
    if n < 10:
        raise ConfigError("dataset too small to split")
    rng = np.random.default_rng(cfg.seed)
    warnings: list[str] = []
    meta: dict[str, Any] = {"strategy": cfg.strategy, "seed": cfg.seed, "n_rows": int(n),
                            "fractions": [cfg.train_fraction, cfg.validation_fraction, cfg.test_fraction]}

    if cfg.strategy == "stratified":
        tr: list[int] = []; va: list[int] = []; te: list[int] = []
        for cls in sorted(set(y.tolist())):
            idx = np.flatnonzero(y == cls)
            rng.shuffle(idx)
            m = len(idx)
            if m < 3:
                tr += idx.tolist()
                warnings.append(f"class '{cls}' has {m} rows (<3): all placed in train; it cannot be evaluated")
                continue
            n_te = max(1, int(round(m * cfg.test_fraction)))
            n_va = max(1, int(round(m * cfg.validation_fraction)))
            if n_te + n_va >= m:
                n_te, n_va = 1, 1
            te += idx[:n_te].tolist(); va += idx[n_te:n_te + n_va].tolist(); tr += idx[n_te + n_va:].tolist()
        out = [np.sort(np.asarray(a, dtype=int)) for a in (tr, va, te)]
        meta["note"] = "per-class shuffled split; flows from one host/session may still straddle splits (see limitations)"
    elif cfg.strategy == "time":
        if timestamps is None:
            raise ConfigError("time split requires observed_at timestamps")
        ts = pd.to_datetime(pd.Series(list(timestamps)), utc=True, errors="coerce")
        if ts.isna().any():
            raise ConfigError("time split requires observed_at for every row")
        order = np.argsort(ts.to_numpy(), kind="stable")
        a = int(round(n * cfg.train_fraction)); b = a + int(round(n * cfg.validation_fraction))
        out = [np.sort(order[:a]), np.sort(order[a:b]), np.sort(order[b:])]
        meta["time_boundaries"] = {
            "train_end": ts.iloc[order[a - 1]].isoformat(), "validation_end": ts.iloc[order[b - 1]].isoformat()}
        meta["note"] = "chronological split: later traffic is never used to fit earlier traffic"
    else:  # group
        if groups is None:
            raise ConfigError("group split requires a group key (e.g. capture file, day, host)")
        g = pd.Series(list(groups)).astype(str).to_numpy()
        uniq = np.array(sorted(set(g.tolist())), dtype=object)
        if len(uniq) < 3:
            raise ConfigError("group split needs at least 3 distinct groups")
        rng.shuffle(uniq)
        sizes = {u: int((g == u).sum()) for u in uniq}
        targets = [cfg.train_fraction * n, cfg.validation_fraction * n, cfg.test_fraction * n]
        buckets: list[list[Any]] = [[], [], []]; filled = [0, 0, 0]
        for u in uniq:  # greedy: put the group where it leaves the least relative deficit
            j = int(np.argmax([(targets[k] - filled[k]) / targets[k] for k in range(3)]))
            buckets[j].append(u); filled[j] += sizes[u]
        out = [np.flatnonzero(np.isin(g, np.asarray(b, dtype=object))) for b in buckets]
        if any(len(b) == 0 for b in out):
            raise ConfigError("group split produced an empty partition; use fewer/smaller groups")
        meta["n_groups"] = int(len(uniq)); meta["note"] = "no group appears in more than one split"

    names = ("train", "validation", "test")
    meta["sizes"] = {k: int(len(v)) for k, v in zip(names, out)}
    meta["class_counts"] = {k: {str(c): int((y[v] == c).sum()) for c in sorted(set(y.tolist()))}
                            for k, v in zip(names, out)}
    for k, v in zip(names, out):
        missing = [c for c, cnt in meta["class_counts"][k].items() if cnt == 0]
        if missing:
            warnings.append(f"split '{k}' has no rows of classes {missing}")
    meta["warnings"] = warnings
    return SplitIndices(out[0], out[1], out[2], meta)


# --------------------------------------------------------------------------- #
# Threshold selection (validation data only)
# --------------------------------------------------------------------------- #
def select_known_confidence(conf_val: np.ndarray, correct_val: np.ndarray, target_accuracy: float,
                            n_classes: int) -> tuple[float, str]:
    """Smallest threshold whose *accepted* validation predictions reach `target_accuracy`.

    Falls back to the highest observed confidence when the target is unreachable; the returned
    method string says which happened so the report never overstates the guarantee.
    """
    lo = 1.0 / max(n_classes, 2)
    cands = np.unique(np.clip(np.round(conf_val, 4), lo, 1.0))
    for t in cands:
        acc = correct_val[conf_val >= t]
        if len(acc) >= 10 and acc.mean() >= target_accuracy:
            return float(t), f"validation_accepted_accuracy>={target_accuracy}"
    return float(cands[-1]), "fallback_target_unreachable_on_validation"


# --------------------------------------------------------------------------- #
# Fit a bundle from pre-split model frames
# --------------------------------------------------------------------------- #
def fit_bundle(train_frame: pd.DataFrame, y_train: Sequence[str], val_frame: pd.DataFrame, y_val: Sequence[str],
               cfg: TrainingConfig, *, split_metadata: Optional[dict[str, Any]] = None,
               label_mapping: Optional[dict[str, str]] = None, trained_at: Optional[datetime] = None) -> ModelBundle:
    validate_version_string(cfg.model_version)
    if cfg.classifier is None and cfg.anomaly is None:
        raise ConfigError("at least one of classifier/anomaly must be configured")
    y_tr = np.asarray(list(y_train), dtype=object); y_va = np.asarray(list(y_val), dtype=object)
    if list(train_frame.columns) != list(MODEL_INPUT_COLUMNS) or list(val_frame.columns) != list(MODEL_INPUT_COLUMNS):
        raise ConfigError("frames must be model-input frames")

    if cfg.max_train_rows and len(train_frame) > cfg.max_train_rows:
        rng = np.random.default_rng(cfg.split.seed)
        keep = np.sort(rng.choice(len(train_frame), cfg.max_train_rows, replace=False))
        train_frame, y_tr = train_frame.iloc[keep].reset_index(drop=True), y_tr[keep]

    pre = fit_preprocessing(train_frame, cfg.preprocessing, SCHEMA_VERSION)   # TRAIN ONLY
    X_tr, X_va = pre.transform_frame(train_frame), pre.transform_frame(val_frame)

    classifier = None; known_conf = 1.0; selection: dict[str, str] = {}
    if cfg.classifier is not None:
        classifier = KnownClassClassifier(cfg.classifier).fit(X_tr, y_tr)
        if len(y_va) < cfg.min_validation_rows:
            raise ConfigError("validation split too small to select the confidence threshold")
        lab_va, conf_va, _ = classifier.scores(X_va)
        known_conf, how = select_known_confidence(conf_va, lab_va == y_va, cfg.target_accepted_accuracy,
                                                  len(classifier.classes_))
        selection["known_confidence"] = how

    anomaly = None; anom_thr = 1.0
    if cfg.anomaly is not None:
        benign_tr, benign_va = X_tr[y_tr == BENIGN_LABEL], X_va[y_va == BENIGN_LABEL]
        if len(benign_tr) == 0 or len(benign_va) < 20:
            raise ConfigError("anomaly detector needs benign rows in train and >=20 in validation")
        anomaly = AnomalyDetector(cfg.anomaly).fit(benign_tr).fit_reference(benign_va)
        anom_thr = anomaly.threshold_for_fpr(benign_va)
        selection["anomaly_threshold"] = (
            f"validation_benign_quantile(target_fpr={cfg.anomaly.target_false_positive_rate}); "
            f"reference_n={anomaly.reference_n_}")

    kind = ("classifier_plus_anomaly" if classifier and anomaly else "classifier_only" if classifier else "anomaly_only")
    meta = BundleMetadata(
        model_version=cfg.model_version, model_kind=kind, feature_schema_version=SCHEMA_VERSION,
        preprocessing_version=pre.preprocessing_version, input_features=pre.input_columns,
        transformed_feature_names=pre.feature_names_out,
        class_labels=classifier.classes_ if classifier else [], label_mapping=label_mapping or {},
        thresholds=Thresholds(known_confidence=known_conf, anomaly_threshold=anom_thr),
        threshold_selection=selection, risk_policy_version=cfg.risk_policy.policy_version,
        risk_policy=cfg.risk_policy.model_dump(),
        training_dataset_fingerprint=dataset_fingerprint(train_frame, list(y_tr)),
        split_metadata=split_metadata or {}, training_config=cfg.model_dump(mode="json"),
        library_versions=runtime_versions(),
        trained_at=(trained_at or datetime.now(timezone.utc)).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    bundle = ModelBundle(metadata=meta, preprocessing=pre, classifier=classifier, anomaly=anomaly,
                         risk_policy=cfg.risk_policy)
    check_bundle_consistency(bundle)
    return bundle


# --------------------------------------------------------------------------- #
# End-to-end: canonical frame + labels -> dedupe -> split -> fit
# --------------------------------------------------------------------------- #
@dataclass
class TrainingOutcome:
    bundle: ModelBundle
    split: SplitIndices
    canonical_frame: pd.DataFrame        # deduplicated canonical records (positional index aligned with split)
    labels: pd.Series
    model_frame: pd.DataFrame
    dedupe: dict[str, Any]
    duplicate_report_before: dict[str, Any]
    leakage: dict[str, Any]

    def subset(self, name: str) -> tuple[list[dict], list[str]]:
        idx = getattr(self.split, name)
        return frame_to_records(self.canonical_frame.iloc[idx]), self.labels.iloc[idx].tolist()


def prepare_frames(canonical: pd.DataFrame, labels: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Validate canonical records (fail closed) and build the model-input frame."""
    if len(canonical) != len(labels):
        raise ConfigError("records and labels differ in length")
    rep = validate_features(frame_to_records(canonical), SCHEMA_VERSION, max_batch_size=None)
    if not rep.valid:
        raise InputValidationError("training records failed validation", rep.issues)
    labs = pd.Series([normalize_label(x) for x in labels], index=labels.index, dtype="object")
    return build_model_frame([r for r in rep.records if r is not None]), labs


def train_bundle(canonical: pd.DataFrame, labels: pd.Series, cfg: TrainingConfig, *,
                 groups: Optional[pd.Series] = None, label_mapping: Optional[dict[str, str]] = None) -> TrainingOutcome:
    model_frame, labs = prepare_frames(canonical, labels)
    before = duplicate_report(model_frame, labs.tolist())
    keep, dd = deduplicate(model_frame, labs, cfg.split.dedupe)
    canonical = canonical.iloc[keep].reset_index(drop=True)
    model_frame = model_frame.iloc[keep].reset_index(drop=True)
    labs = labs.iloc[keep].reset_index(drop=True)
    g = None if groups is None else groups.iloc[keep].reset_index(drop=True)

    sp = split_indices(labs.tolist(), cfg.split, groups=None if g is None else g.tolist(),
                       timestamps=canonical["observed_at"].tolist() if cfg.split.strategy == "time" else None)
    frames = {"train": model_frame.iloc[sp.train], "validation": model_frame.iloc[sp.validation],
              "test": model_frame.iloc[sp.test]}
    leakage = {"rows_in_split_also_in_train": cross_split_overlap(frames),
               "dedupe": dd, "duplicates_before_dedupe": before}
    sp.metadata["dataset_fingerprint_full"] = dataset_fingerprint(model_frame, labs.tolist())
    sp.metadata["leakage_checks"] = leakage
    bundle = fit_bundle(frames["train"], labs.iloc[sp.train], frames["validation"], labs.iloc[sp.validation], cfg,
                        split_metadata=sp.metadata, label_mapping=label_mapping)
    return TrainingOutcome(bundle, sp, canonical, labs, model_frame, dd, before, leakage)


def compare_classifier_candidates(X_train: np.ndarray, y_train: Sequence[str], X_val: np.ndarray,
                                  y_val: Sequence[str], configs: dict[str, ClassifierConfig]) -> list[dict[str, Any]]:
    """Model comparison on the VALIDATION split only (never the test split)."""
    rows = []
    yv = np.asarray(list(y_val), dtype=object)
    for name, c in configs.items():
        clf = KnownClassClassifier(c).fit(X_train, np.asarray(list(y_train), dtype=object))
        lab, conf, _ = clf.scores(X_val)
        labels_present = sorted(set(yv.tolist()))
        rec = recall_score(yv, lab, labels=labels_present, average=None, zero_division=0)
        rows.append({
            "name": name, "kind": c.kind,
            "validation_macro_f1": float(f1_score(yv, lab, labels=labels_present, average="macro", zero_division=0)),
            "validation_min_class_recall": float(np.min(rec)),
            "mean_confidence": float(np.mean(conf)),
        })
    return rows
