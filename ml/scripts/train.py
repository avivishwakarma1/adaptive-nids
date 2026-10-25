#!/usr/bin/env python
"""Train a baseline bundle on a REAL dataset, evaluate on the untouched test split, save immutably.

  python scripts/train.py --csv data/UNSW.csv --mapping configs/mappings/unsw_nb15.json \
      --dataset-name unsw-nb15 --version baseline-1 --artifact-dir artifacts --run-dir runs/baseline-1 \
      [--config configs/train_default.json] [--compare-classifiers] \
      [--registry runs/registry.json --gates configs/gates.json --actor you [--promote --approved-by you]]

Order of operations (no step may look at the test split before the last one):
  validate -> dedupe -> split -> fit preprocessing on TRAIN -> fit models on TRAIN -> select thresholds on
  VALIDATION -> (optional classifier comparison on VALIDATION) -> save artifact -> evaluate TEST once.
Prints the artifact SHA-256: store it in the backend model_registry (`artifact_sha256`).
Nothing is promoted unless --promote AND --approved-by are given AND validation passes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _common import DEFAULT_MAX_ROWS, fail, load_dataset, read_json_config, write_json
from nids_ml import EvaluationConfig, PromotionGates, evaluate, save_bundle, validate_candidate
from nids_ml.classification import ClassifierConfig
from nids_ml.contracts import NidsMLError
from nids_ml.lifecycle import LocalModelRegistry
from nids_ml.training import TrainingConfig, compare_classifier_candidates, train_bundle


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True); ap.add_argument("--mapping", required=True)
    ap.add_argument("--dataset-name", required=True); ap.add_argument("--version", required=True)
    ap.add_argument("--artifact-dir", required=True); ap.add_argument("--run-dir", required=True)
    ap.add_argument("--config"); ap.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS)
    ap.add_argument("--compare-classifiers", action="store_true")
    ap.add_argument("--registry"); ap.add_argument("--gates"); ap.add_argument("--actor", default="offline-script")
    ap.add_argument("--promote", action="store_true"); ap.add_argument("--approved-by")
    a = ap.parse_args(argv)
    try:
        if a.promote and not (a.registry and a.gates and a.approved_by):
            return fail("--promote requires --registry, --gates and --approved-by (explicit human approval)")
        cfg = TrainingConfig.model_validate({**read_json_config(a.config), "model_version": a.version})
        frame, labels, groups, map_rep, mapping = load_dataset(a.csv, a.mapping, a.max_rows)
        if cfg.split.strategy == "group" and groups is None:
            return fail("group split needs 'group_column' in the mapping")
        out = train_bundle(frame, labels, cfg, groups=groups, label_mapping=mapping.label_map)
        b = out.bundle
        run = Path(a.run_dir)

        comparison = None
        if a.compare_classifiers:
            sp, mf, y = out.split, out.model_frame, out.labels
            comparison = compare_classifier_candidates(
                b.preprocessing.transform_frame(mf.iloc[sp.train]), y.iloc[sp.train],
                b.preprocessing.transform_frame(mf.iloc[sp.validation]), y.iloc[sp.validation],
                {"random_forest": ClassifierConfig(kind="random_forest"),
                 "logistic_regression": ClassifierConfig(kind="logistic_regression"),
                 "hist_gradient_boosting": ClassifierConfig(kind="hist_gradient_boosting")})
            write_json(run / "classifier_comparison_validation_only.json", comparison)

        saved = save_bundle(b, a.artifact_dir)
        recs, labs = out.subset("test")
        report = evaluate(b, [{"observation": o, "label": l} for o, l in zip(recs, labs)], EvaluationConfig(
            dataset_name=a.dataset_name, split_strategy=cfg.split.strategy,
            split_metadata={k: v for k, v in out.split.metadata.items() if k in ("strategy", "seed", "sizes", "fractions", "note")}))
        write_json(run / "evaluation_test.json", report.model_dump(mode="json"))
        write_json(run / "training_summary.json", {
            "dataset_name": a.dataset_name, "mapping_report": map_rep.model_dump(), "dedupe": out.dedupe,
            "duplicates_before_dedupe": out.duplicate_report_before, "leakage_checks": out.leakage,
            "split": out.split.metadata, "thresholds": b.metadata.thresholds.model_dump(),
            "threshold_selection": b.metadata.threshold_selection, "model_kind": b.metadata.model_kind,
            "class_labels": b.metadata.class_labels, "library_versions": b.metadata.library_versions,
            "artifact_sha256": saved.sha256, "training_dataset_fingerprint": b.metadata.training_dataset_fingerprint,
            "license_note": mapping.license_note,
        })

        if a.registry:
            reg = LocalModelRegistry(a.registry)
            reg.register_candidate(b, saved, actor=a.actor)
            gates = PromotionGates.model_validate(read_json_config(a.gates)) if a.gates else PromotionGates()
            result = validate_candidate(b, report, gates)
            status = reg.record_validation(result, actor=a.actor)
            write_json(run / "validation_result.json", result.model_dump(mode="json"))
            print(f"validation: {status}" + ("" if result.passed else f" ({'; '.join(result.reasons)})"))
            if a.promote and result.passed:
                reg.promote(a.version, validation=result, expected_active_version=reg.active_version,
                            actor=a.approved_by, reason="explicit approval via train.py --promote")
                print(f"promoted {a.version} (approved by {a.approved_by})")
            elif a.promote:
                print("NOT promoted: validation failed")
        print(f"saved {a.version}  sha256={saved.sha256}")
        print(f"reports in {run}/ (evaluation_test.json, training_summary.json)")
    except (NidsMLError, OSError, ValueError) as exc:
        return fail(getattr(exc, "message", None) or f"{type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
