#!/usr/bin/env python
"""Controlled continual-learning experiment: reviewed feedback + replay -> CANDIDATE -> validation -> (optional) explicit promotion.

  python scripts/continual_experiment.py --csv data/X.csv --mapping configs/mappings/x.json --dataset-name x \
      --artifact-dir artifacts --registry runs/registry.json --feedback runs/reviewed_feedback.jsonl \
      --gates configs/gates.json --candidate-version candidate-2 --actor you --out-dir runs/cycle-2 \
      [--replay-fraction 1.0 --min-replay-per-class 50 --max-feedback-fraction 0.2] \
      [--promote --approved-by you]

Protocol
  1. Active model comes from the registry (checksum-verified). The base data are re-split with the ACTIVE
     model's recorded split config; the run aborts if the data fingerprint differs from the one it was trained on.
  2. Feedback JSONL: one FeedbackExample per line. Only training_eligible rows with a recorded reviewer are used;
     duplicates, contradictions, rows overlapping validation/holdout, and unknown labels are dropped and counted.
  3. Replay: stratified, seeded sample of the original TRAIN split (minority classes protected).
  4. Candidate is trained in-process on replay+feedback, thresholds re-selected on the SAME validation split.
  5. Candidate and active model are both scored on the SAME untouched holdout (the original test split).
  6. Gates (explicit file; null = not enforced; none configured = FAIL) include regression checks against the active model.
  7. The candidate is registered as validated/rejected. It is promoted only with --promote AND --approved-by AND a passing
     validation. Rollback: LocalModelRegistry.rollback(...) restores the previous version (see README).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _common import DEFAULT_MAX_ROWS, fail, load_dataset, read_json_config, write_json
from nids_ml import PromotionGates
from nids_ml.contracts import ConfigError, NidsMLError
from nids_ml.features import frame_to_records
from nids_ml.lifecycle import CandidateDatasetConfig, FeedbackExample, LocalModelRegistry, run_candidate_cycle
from nids_ml.training import SplitConfig, TrainingConfig, dataset_fingerprint, deduplicate, prepare_frames, split_indices


def read_feedback(path: str) -> list[FeedbackExample]:
    out = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                out.append(FeedbackExample.model_validate(json.loads(line)))
            except ValueError:
                raise ConfigError(f"feedback line {n} is not a valid FeedbackExample") from None
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("csv", "mapping", "dataset_name", "artifact_dir", "registry", "feedback", "gates", "candidate_version", "out_dir"):
        ap.add_argument("--" + k.replace("_", "-"), required=True)
    ap.add_argument("--actor", default="offline-script"); ap.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS)
    ap.add_argument("--replay-fraction", type=float, default=1.0); ap.add_argument("--min-replay-per-class", type=int, default=50)
    ap.add_argument("--max-feedback-fraction", type=float); ap.add_argument("--promote", action="store_true")
    ap.add_argument("--approved-by")
    a = ap.parse_args(argv)
    try:
        if a.promote and not a.approved_by:
            return fail("--promote requires --approved-by (explicit human approval)")
        reg = LocalModelRegistry(a.registry)
        active = reg.load_active_bundle(a.artifact_dir)
        tcfg = TrainingConfig.model_validate({**active.metadata.training_config, "model_version": a.candidate_version})

        frame, labels, _, _, _ = load_dataset(a.csv, a.mapping, a.max_rows)
        mf, labs = prepare_frames(frame, labels)
        keep, _ = deduplicate(mf, labs, tcfg.split.dedupe)
        frame, mf, labs = frame.iloc[keep].reset_index(drop=True), mf.iloc[keep].reset_index(drop=True), labs.iloc[keep].reset_index(drop=True)
        if dataset_fingerprint(mf, labs.tolist()) != active.metadata.split_metadata.get("dataset_fingerprint_full"):
            return fail("dataset differs from the data the active model was trained on; the fixed holdout would not be comparable")
        sp = split_indices(labs.tolist(), tcfg.split)

        feedback = read_feedback(a.feedback)
        res = run_candidate_cycle(
            active_bundle=active, registry=reg, artifact_dir=a.artifact_dir,
            base_train_frame=mf.iloc[sp.train].reset_index(drop=True), base_train_labels=labs.iloc[sp.train].tolist(),
            validation_frame=mf.iloc[sp.validation].reset_index(drop=True), validation_labels=labs.iloc[sp.validation].tolist(),
            holdout_records=frame_to_records(frame.iloc[sp.test]), holdout_labels=labs.iloc[sp.test].tolist(),
            holdout_dataset_name=a.dataset_name + ":holdout", feedback=feedback,
            candidate_dataset_cfg=CandidateDatasetConfig(
                allowed_labels=sorted(set(labs)), replay_fraction=a.replay_fraction,
                min_replay_per_class=a.min_replay_per_class, max_feedback_fraction=a.max_feedback_fraction),
            training_cfg=tcfg, gates=PromotionGates.model_validate(read_json_config(a.gates)), actor=a.actor)

        out = Path(a.out_dir)
        write_json(out / "candidate_dataset_manifest.json", res.dataset_manifest)
        write_json(out / "candidate_evaluation.json", res.candidate_report.model_dump(mode="json"))
        write_json(out / "active_model_evaluation.json", res.baseline_report.model_dump(mode="json"))
        write_json(out / "validation_result.json", res.validation.model_dump(mode="json"))
        print(f"candidate {a.candidate_version} sha256={res.saved.sha256}")
        print(f"validation: {'PASSED' if res.validation.passed else 'FAILED'}")
        for r in res.validation.reasons:
            print(f"  - {r}")
        if a.promote and res.validation.passed:
            reg.promote(a.candidate_version, validation=res.validation, expected_active_version=reg.active_version,
                        actor=a.approved_by, reason="explicit approval via continual_experiment.py --promote")
            print(f"promoted {a.candidate_version}; previous version retained for rollback")
        else:
            print("not promoted (active model unchanged)")
    except (NidsMLError, OSError, ValueError) as exc:
        return fail(getattr(exc, "message", None) or f"{type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
