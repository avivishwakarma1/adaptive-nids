#!/usr/bin/env python
"""Evaluate a saved bundle on a separate labelled CSV (e.g. cross-dataset or a fixed holdout file).

  python scripts/evaluate.py --csv data/holdout.csv --mapping configs/mappings/x.json --dataset-name x-holdout \
      --artifact-dir artifacts --version baseline-1 --expected-sha256 <registry value> --out runs/eval.json

The checksum must come from the model registry. The script cannot know whether the CSV overlaps the
training data: evaluating on training rows gives meaningless numbers, so use a file that was never used
for fitting (for the training dataset's own test split, use the report written by train.py).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _common import DEFAULT_MAX_ROWS, fail, load_dataset, write_json
from nids_ml import EvaluationConfig, evaluate, load_bundle
from nids_ml.contracts import NidsMLError
from nids_ml.features import frame_to_records
from nids_ml.training import prepare_frames


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("csv", "mapping", "dataset_name", "artifact_dir", "version", "expected_sha256", "out"):
        ap.add_argument("--" + k.replace("_", "-"), required=True)
    ap.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS)
    a = ap.parse_args(argv)
    try:
        bundle = load_bundle(a.artifact_dir, a.version, expected_sha256=a.expected_sha256)
        frame, labels, _, map_rep, _ = load_dataset(a.csv, a.mapping, a.max_rows)
        _, labs = prepare_frames(frame, labels)
        rep = evaluate(bundle, [{"observation": o, "label": l} for o, l in zip(frame_to_records(frame), labs)],
                       EvaluationConfig(dataset_name=a.dataset_name, split_strategy="external_labelled_file",
                                        split_metadata={"mapping_report": map_rep.model_dump()}))
        write_json(Path(a.out), rep.model_dump(mode="json"))
    except (NidsMLError, OSError, ValueError) as exc:
        return fail(getattr(exc, "message", None) or type(exc).__name__)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
