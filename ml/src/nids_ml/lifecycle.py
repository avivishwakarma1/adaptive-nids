"""Controlled continual learning (contract section 10): reviewed feedback -> candidate -> validation ->
explicit promotion -> rollback. Nothing here promotes or retrains automatically.

* `build_candidate_training_set` admits only feedback that is `training_eligible` AND has a recorded
  reviewer, de-duplicates it, drops contradictions, and keeps it disjoint from validation/holdout rows.
* `validate_candidate` applies explicit gates (including regression checks against the previous
  model on the SAME evaluation set). With no quantitative gate configured it FAILS (fail closed).
* `LocalModelRegistry` is an offline-experiment registry with the same state machine as the backend
  `model_registry` (candidate -> validated|rejected -> active -> retired). Production state lives in
  the backend; this class lets the ML package prove the transitions without MongoDB.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from .artifacts import ModelBundle, SavedArtifact, load_bundle, save_bundle
from .contracts import (
    BENIGN_LABEL, SUPPORTED_SCHEMA_VERSIONS, ConfigError, EvaluationConfig, EvaluationReport, GateCheck,
    LabeledObservation, LifecycleError, PromotionGates, ValidationResult, normalize_label, utc_now_iso,
    validate_version_string,
)
from .evaluation import evaluate
from .features import MODEL_INPUT_COLUMNS, build_model_frame, validate_features
from .training import TrainingConfig, _row_hashes, dataset_fingerprint, fit_bundle


def _now(clock: Optional[Callable[[], datetime]]) -> str:
    return utc_now_iso(clock() if clock else None)


# --------------------------------------------------------------------------- #
# Validation gates
# --------------------------------------------------------------------------- #
def _attack_detection_recall(rep: EvaluationReport) -> Optional[float]:
    att = [v for k, v in rep.type_by_truth.items() if k != BENIGN_LABEL]
    total = sum(sum(v.values()) for v in att)
    if total == 0:
        return None
    return sum(sum(v.values()) - v.get("benign", 0) for v in att) / total


def _class_recalls(rep: EvaluationReport, attack_only: bool = False) -> dict[str, Optional[float]]:
    return {c.label: c.recall for c in rep.class_metrics
            if c.support > 0 and not (attack_only and c.label == BENIGN_LABEL)}


def validate_candidate(candidate_bundle: ModelBundle, evaluation_report: EvaluationReport, gates: PromotionGates,
                       *, clock: Optional[Callable[[], datetime]] = None) -> ValidationResult:
    """Structural integrity checks + configured quality gates. Never mutates anything.

    Gate semantics (also in README):
      minimum_recall                attack-detection recall: share of true attack rows whose prediction_type != benign
      minimum_per_class_recall      lowest operational recall over attack classes present in the evaluation set
      maximum_false_positive_rate   benign rows with prediction_type != benign / benign rows
      maximum_*_drop / *_increase   regression versus `gates.baseline_report` on the same dataset fingerprint
    """
    r, b = evaluation_report, candidate_bundle
    checks: list[GateCheck] = []

    def add(name: str, ok: bool, observed=None, required=None, detail: str = "") -> None:
        checks.append(GateCheck(name=name, passed=bool(ok), observed=observed, required=required, detail=detail))

    # -- structural checks (always applied) --
    add("report_is_evaluated", r.evaluated and r.n_samples > 0, r.n_samples, ">0", "an evaluation must have run on real rows")
    add("report_matches_candidate_version", r.model_version == b.version, r.model_version, b.version)
    add("artifact_checksum_recorded", b.artifact_sha256 is not None, detail="bundle must be saved/loaded so its checksum is known")
    add("report_bound_to_artifact", b.artifact_sha256 is not None and r.artifact_sha256 == b.artifact_sha256,
        detail="the report must describe exactly this artifact")
    add("feature_schema_compatible",
        b.metadata.feature_schema_version in SUPPORTED_SCHEMA_VERSIONS and r.feature_schema_version == b.metadata.feature_schema_version,
        r.feature_schema_version, b.metadata.feature_schema_version)
    add("dataset_fingerprint_present", bool(r.dataset_fingerprint), detail="provenance is required")

    # -- quantitative gates --
    names = gates.quantitative_gate_names()
    add("at_least_one_quantitative_gate_configured", len(names) > 0, len(names), ">=1",
        "validation cannot pass without configured quality gates")
    m = r.metrics
    if gates.minimum_samples is not None:
        add("minimum_samples", r.n_samples >= gates.minimum_samples, r.n_samples, gates.minimum_samples)
    if gates.minimum_recall is not None:
        v = _attack_detection_recall(r)
        add("minimum_recall", v is not None and v >= gates.minimum_recall, v, gates.minimum_recall,
            "attack-detection recall" if v is not None else "no attack rows: not evaluated")
    if gates.maximum_false_positive_rate is not None:
        v = m.get("false_positive_rate")
        add("maximum_false_positive_rate", v is not None and v <= gates.maximum_false_positive_rate, v,
            gates.maximum_false_positive_rate, "" if v is not None else "no benign rows: not evaluated")
    if gates.minimum_macro_f1 is not None:
        v = m.get("f1_macro")
        add("minimum_macro_f1", v is not None and v >= gates.minimum_macro_f1, v, gates.minimum_macro_f1)
    if gates.minimum_per_class_recall is not None:
        rec = _class_recalls(r, attack_only=True)
        low = min((x if x is not None else 0.0 for x in rec.values()), default=None)
        worst = None if not rec else min(rec, key=lambda k: rec[k] if rec[k] is not None else 0.0)
        add("minimum_per_class_recall", low is not None and low >= gates.minimum_per_class_recall, low,
            gates.minimum_per_class_recall, f"worst class: {worst}" if worst else "no attack classes: not evaluated")

    # -- regression versus previous model --
    regress_names = ("maximum_macro_f1_drop", "maximum_per_class_recall_drop", "maximum_false_positive_rate_increase")
    base = gates.baseline_report
    if any(getattr(gates, n) is not None for n in regress_names):
        comparable = base is not None and base.dataset_fingerprint == r.dataset_fingerprint
        add("baseline_report_comparable", comparable, detail="regression gates need the previous model scored on the SAME dataset fingerprint")
        if comparable:
            if gates.maximum_macro_f1_drop is not None:
                a, c = base.metrics.get("f1_macro"), m.get("f1_macro")
                ok = a is not None and c is not None and (a - c) <= gates.maximum_macro_f1_drop
                add("maximum_macro_f1_drop", ok, None if a is None or c is None else a - c, gates.maximum_macro_f1_drop)
            if gates.maximum_per_class_recall_drop is not None:
                cand = _class_recalls(r); old = _class_recalls(base)
                worst_drop, worst_cls, ok = 0.0, None, True
                for cls, ov in old.items():
                    if ov is None:
                        continue
                    cv = cand.get(cls)
                    drop = ov - (cv if cv is not None else 0.0)
                    if cls not in cand or cv is None:
                        ok = False
                    if drop > worst_drop:
                        worst_drop, worst_cls = drop, cls
                ok = ok and worst_drop <= gates.maximum_per_class_recall_drop
                add("maximum_per_class_recall_drop", ok, worst_drop, gates.maximum_per_class_recall_drop,
                    f"largest regression in class: {worst_cls}" if worst_cls else "")
            if gates.maximum_false_positive_rate_increase is not None:
                a, c = base.metrics.get("false_positive_rate"), m.get("false_positive_rate")
                ok = a is not None and c is not None and (c - a) <= gates.maximum_false_positive_rate_increase
                add("maximum_false_positive_rate_increase", ok, None if a is None or c is None else c - a,
                    gates.maximum_false_positive_rate_increase)

    failed = [c for c in checks if not c.passed]
    return ValidationResult(
        passed=not failed, candidate_version=b.version, candidate_artifact_sha256=b.artifact_sha256,
        feature_schema_version=b.metadata.feature_schema_version, evaluation_dataset_fingerprint=r.dataset_fingerprint,
        baseline_version=None if base is None else base.model_version, checks=checks,
        reasons=[f"{c.name}: {c.detail or 'failed'}" for c in failed],
        gates=gates.model_dump(exclude={"baseline_report"}), validated_at=_now(clock),
    )


# --------------------------------------------------------------------------- #
# Registry (offline experiment registry mirroring the backend state machine)
# --------------------------------------------------------------------------- #
STATUSES = ("candidate", "validated", "active", "retired", "rejected")


class LocalModelRegistry:
    def __init__(self, path: Optional[str | Path] = None, *, clock: Optional[Callable[[], datetime]] = None):
        self.path = None if path is None else Path(path)
        self.clock = clock
        self.state: dict[str, Any] = {"format": "nids-ml-registry/1", "active_version": None, "entries": {}, "audit": []}
        if self.path is not None and self.path.exists():
            self.state = json.loads(self.path.read_text(encoding="utf-8"))

    # -- internals --
    def _audit(self, event: str, version: Optional[str], actor: str, reason: str = "", **details: Any) -> None:
        self.state["audit"].append({"event": event, "version": version, "actor": actor, "at": _now(self.clock),
                                    "reason": reason, "details": details})

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".registry.")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(self.state, fh, indent=2, sort_keys=True, allow_nan=False)
        os.replace(tmp, self.path)

    def _entry(self, version: str) -> dict[str, Any]:
        validate_version_string(version)
        try:
            return self.state["entries"][version]
        except KeyError:
            raise LifecycleError(f"unknown model version {version!r}") from None

    # -- queries --
    @property
    def active_version(self) -> Optional[str]:
        return self.state["active_version"]

    def status(self, version: str) -> str:
        return self._entry(version)["status"]

    def audit_log(self) -> list[dict[str, Any]]:
        return list(self.state["audit"])

    # -- transitions --
    def register_candidate(self, bundle: ModelBundle, saved: SavedArtifact, *, actor: str) -> None:
        if bundle.version in self.state["entries"]:
            raise LifecycleError(f"version {bundle.version!r} already registered")
        if bundle.artifact_sha256 != saved.sha256:
            raise LifecycleError("bundle checksum does not match the saved artifact")
        m = bundle.metadata
        self.state["entries"][bundle.version] = {
            "status": "candidate", "model_kind": m.model_kind, "feature_schema_version": m.feature_schema_version,
            "artifact_sha256": saved.sha256, "training_dataset_fingerprint": m.training_dataset_fingerprint,
            "thresholds": m.thresholds.model_dump(), "created_at": _now(self.clock), "validated_at": None,
            "activated_at": None, "validation": None,
        }
        self._audit("candidate_created", bundle.version, actor, artifact_sha256=saved.sha256)
        self.save()

    def record_validation(self, result: ValidationResult, *, actor: str) -> str:
        e = self._entry(result.candidate_version)
        if e["status"] != "candidate":
            raise LifecycleError(f"cannot validate a model in status {e['status']!r}")
        if result.candidate_artifact_sha256 != e["artifact_sha256"]:
            raise LifecycleError("validation result does not describe the registered artifact")
        e["status"] = "validated" if result.passed else "rejected"
        e["validated_at"] = result.validated_at
        e["validation"] = {"passed": result.passed, "reasons": result.reasons,
                           "evaluation_dataset_fingerprint": result.evaluation_dataset_fingerprint,
                           "baseline_version": result.baseline_version}
        self._audit("validation_passed" if result.passed else "validation_failed", result.candidate_version, actor,
                    reasons=result.reasons)
        self.save()
        return e["status"]

    def promote(self, version: str, *, validation: ValidationResult, expected_active_version: Optional[str],
                actor: str, reason: str) -> dict[str, Optional[str]]:
        """Explicit, human-triggered promotion. Fails closed unless a passing validation matches this exact artifact."""
        e = self._entry(version)
        if e["status"] != "validated" or not validation.passed:
            raise LifecycleError("promotion requires a passing validation of this candidate")
        if validation.candidate_version != version or validation.candidate_artifact_sha256 != e["artifact_sha256"]:
            raise LifecycleError("validation result does not match the candidate artifact")
        if expected_active_version != self.active_version:
            raise LifecycleError(f"stale request: active version is {self.active_version!r}, expected {expected_active_version!r}")
        if not reason.strip():
            raise LifecycleError("a promotion reason is required")
        prev = self.active_version
        if prev is not None:
            pe = self._entry(prev)
            if pe["feature_schema_version"] != e["feature_schema_version"]:
                raise LifecycleError("candidate feature schema differs from the active model; a declared migration is required")
            pe["status"] = "retired"
        e["status"] = "active"; e["activated_at"] = _now(self.clock)
        self.state["active_version"] = version
        self._audit("promoted", version, actor, reason, previous_active_version=prev)
        self.save()
        return {"active_model_version": version, "previous_active_version": prev}

    def rollback(self, target_version: str, *, expected_active_version: str, actor: str, reason: str) -> dict[str, Optional[str]]:
        """Restore a previously validated version. The artifact checksum is re-verified when it is loaded."""
        cur = self.active_version
        if cur is None or expected_active_version != cur:
            raise LifecycleError(f"stale request: active version is {cur!r}, expected {expected_active_version!r}")
        if target_version == cur:
            raise LifecycleError("target is already active")
        t, c = self._entry(target_version), self._entry(cur)
        if t["status"] not in ("retired", "validated") or not (t.get("validation") or {}).get("passed"):
            raise LifecycleError("rollback target must be a previously validated version")
        if t["feature_schema_version"] != c["feature_schema_version"]:
            raise LifecycleError("rollback target has an incompatible feature schema")
        if not reason.strip():
            raise LifecycleError("a rollback reason is required")
        c["status"] = "retired"; t["status"] = "active"; t["activated_at"] = _now(self.clock)
        self.state["active_version"] = target_version
        self._audit("rolled_back", target_version, actor, reason, previous_active_version=cur)
        self.save()
        return {"active_model_version": target_version, "previous_active_version": cur}

    def load_active_bundle(self, artifact_dir: str | Path, **kw: Any) -> ModelBundle:
        v = self.active_version
        if v is None:
            raise LifecycleError("no active model")
        return load_bundle(artifact_dir, v, expected_sha256=self._entry(v)["artifact_sha256"], **kw)


# --------------------------------------------------------------------------- #
# Reviewed feedback -> candidate training set (with replay)
# --------------------------------------------------------------------------- #
class FeedbackExample(BaseModel):
    """One analyst-labelled observation. Only reviewed, eligible examples can enter training."""
    model_config = ConfigDict(extra="forbid")
    feedback_id: Optional[str] = None
    observation: dict[str, Any]
    analyst_label: str
    training_eligible: bool = False
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[str] = None


class CandidateDatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    allowed_labels: list[str] = Field(description="known classes + benign; new classes need an explicit decision")
    replay_fraction: float = Field(default=1.0, gt=0, le=1)
    min_replay_per_class: int = Field(default=50, ge=0)
    max_feedback_fraction: Optional[float] = Field(default=None, gt=0, lt=1)
    seed: int = 20260101


@dataclass
class CandidateDataset:
    frame: pd.DataFrame
    labels: list[str]
    manifest: dict[str, Any] = field(default_factory=dict)


def build_candidate_training_set(base_train_frame: pd.DataFrame, base_train_labels: Sequence[str],
                                 feedback: Sequence[FeedbackExample], cfg: CandidateDatasetConfig, *,
                                 exclude_frames: Sequence[pd.DataFrame] = ()) -> CandidateDataset:
    """base_train_frame/exclude_frames are model-input frames. `exclude_frames` = validation + holdout."""
    allowed = {normalize_label(a) for a in cfg.allowed_labels}
    reasons: dict[str, int] = {}

    def drop(reason: str) -> None:
        reasons[reason] = reasons.get(reason, 0) + 1

    accepted_rows: list[dict] = []; accepted_labels: list[str] = []
    for fb in feedback:
        if not fb.training_eligible:
            drop("not_training_eligible"); continue
        if not fb.reviewed_by or not fb.reviewed_at:
            drop("missing_review_record"); continue
        try:
            lab = normalize_label(fb.analyst_label)
        except ValueError:
            drop("invalid_label"); continue
        if lab not in allowed:
            drop("label_not_in_allowed_set"); continue
        rep = validate_features([fb.observation], max_batch_size=None)
        if not rep.valid or rep.records[0] is None:
            drop("invalid_observation"); continue
        accepted_rows.append(rep.records[0]); accepted_labels.append(lab)

    fb_frame = build_model_frame(accepted_rows) if accepted_rows else pd.DataFrame(columns=list(MODEL_INPUT_COLUMNS))
    for col in MODEL_INPUT_COLUMNS:  # keep dtypes aligned for empty feedback
        if len(fb_frame) == 0 and col in base_train_frame:
            fb_frame[col] = fb_frame[col].astype(base_train_frame[col].dtype)
    keep = np.ones(len(fb_frame), dtype=bool)
    if len(fb_frame):
        h = pd.Series(_row_hashes(fb_frame)); lab_s = pd.Series(accepted_labels)
        conflict = (pd.DataFrame({"h": h, "l": lab_s}).groupby("h")["l"].transform("nunique") > 1).to_numpy()
        for _ in range(int(conflict.sum())): drop("conflicting_feedback_labels")
        dup = pd.DataFrame({"h": h, "l": lab_s}).duplicated(keep="first").to_numpy() & ~conflict
        for _ in range(int(dup.sum())): drop("duplicate_feedback")
        keep &= ~conflict & ~dup
        held = set()
        for f in exclude_frames:
            held |= set(_row_hashes(f).tolist())
        in_hold = np.array([x in held for x in h.tolist()], dtype=bool) & keep
        for _ in range(int(in_hold.sum())): drop("overlaps_validation_or_holdout")
        keep &= ~in_hold
        base_map: dict[int, set] = {}
        for hv, lv in zip(_row_hashes(base_train_frame).tolist(), base_train_labels):
            base_map.setdefault(hv, set()).add(lv)
        for i in np.flatnonzero(keep):
            known = base_map.get(int(h.iloc[i]))
            if known is None:
                continue
            if known == {accepted_labels[i]}:
                drop("already_in_training_data"); keep[i] = False
            else:
                drop("contradicts_training_label_needs_adjudication"); keep[i] = False
    fb_idx = np.flatnonzero(keep)

    # replay: deterministic, stratified, minority classes protected by min_replay_per_class
    rng = np.random.default_rng(cfg.seed)
    y = np.asarray(list(base_train_labels), dtype=object)
    chosen: list[int] = []
    for cls in sorted(set(y.tolist())):
        idx = np.flatnonzero(y == cls); rng.shuffle(idx)
        k = min(len(idx), max(int(math.ceil(cfg.replay_fraction * len(idx))), cfg.min_replay_per_class))
        chosen += idx[:k].tolist()
    chosen = sorted(chosen)

    n_fb = len(fb_idx)
    if cfg.max_feedback_fraction is not None and n_fb > 0:
        cap = int(math.floor(cfg.max_feedback_fraction / (1 - cfg.max_feedback_fraction) * len(chosen)))
        if n_fb > cap:
            for _ in range(n_fb - cap): drop("exceeds_max_feedback_fraction")
            fb_idx = fb_idx[:cap]

    frame = pd.concat([base_train_frame.iloc[chosen], fb_frame.iloc[fb_idx]], ignore_index=True)
    labels = [y[i] for i in chosen] + [accepted_labels[i] for i in fb_idx]
    manifest = {
        "feedback_received": len(feedback), "feedback_accepted": int(len(fb_idx)), "feedback_dropped_by_reason": reasons,
        "replay_rows": len(chosen), "base_train_rows": int(len(base_train_frame)),
        "class_counts": {c: int(labels.count(c)) for c in sorted(set(labels))},
        "feedback_class_counts": {c: int(sum(1 for i in fb_idx if accepted_labels[i] == c)) for c in sorted({accepted_labels[i] for i in fb_idx})},
        "candidate_dataset_fingerprint": dataset_fingerprint(frame, labels), "seed": cfg.seed,
        "replay_fraction": cfg.replay_fraction, "min_replay_per_class": cfg.min_replay_per_class,
    }
    return CandidateDataset(frame, labels, manifest)


# --------------------------------------------------------------------------- #
# One full candidate cycle (never promotes)
# --------------------------------------------------------------------------- #
@dataclass
class CandidateCycleResult:
    candidate: ModelBundle
    saved: SavedArtifact
    candidate_report: EvaluationReport
    baseline_report: EvaluationReport
    validation: ValidationResult
    dataset_manifest: dict[str, Any]


def run_candidate_cycle(
    *, active_bundle: ModelBundle, registry: LocalModelRegistry, artifact_dir: str | Path,
    base_train_frame: pd.DataFrame, base_train_labels: Sequence[str],
    validation_frame: pd.DataFrame, validation_labels: Sequence[str],
    holdout_records: list[dict], holdout_labels: list[str], holdout_dataset_name: str,
    feedback: Sequence[FeedbackExample], candidate_dataset_cfg: CandidateDatasetConfig,
    training_cfg: TrainingConfig, gates: PromotionGates, actor: str,
    clock: Optional[Callable[[], datetime]] = None,
) -> CandidateCycleResult:
    """Train a candidate in-process, evaluate candidate AND active model on the same untouched holdout,
    validate against gates (incl. regression), and record the outcome. Does NOT promote."""
    h_rep = validate_features(holdout_records, max_batch_size=None); h_rep.raise_if_invalid()
    holdout_frame = build_model_frame([r for r in h_rep.records if r is not None])
    cand_ds = build_candidate_training_set(
        base_train_frame, base_train_labels, feedback, candidate_dataset_cfg,
        exclude_frames=[validation_frame, holdout_frame])
    candidate = fit_bundle(cand_ds.frame, cand_ds.labels, validation_frame, validation_labels, training_cfg,
                           split_metadata={"candidate_dataset": cand_ds.manifest,
                                           "validation_fixed_from_previous_cycle": True})
    saved = save_bundle(candidate, artifact_dir, saved_at=clock() if clock else None)
    registry.register_candidate(candidate, saved, actor=actor)

    fp = dataset_fingerprint(holdout_frame, holdout_labels)
    labeled = [LabeledObservation(observation=o, label=l) for o, l in zip(holdout_records, holdout_labels)]
    ecfg = EvaluationConfig(dataset_name=holdout_dataset_name, dataset_fingerprint=fp,
                            split_strategy="fixed_untouched_holdout", clock=clock)
    cand_report = evaluate(candidate, labeled, ecfg)
    base_report = evaluate(active_bundle, labeled, ecfg)
    result = validate_candidate(candidate, cand_report, gates.model_copy(update={"baseline_report": base_report}), clock=clock)
    registry.record_validation(result, actor=actor)
    return CandidateCycleResult(candidate, saved, cand_report, base_report, result, cand_ds.manifest)
