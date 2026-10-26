#!/usr/bin/env python
"""Objective 1: inspect a REAL dataset file before any modelling. Reports facts only.

  python scripts/inspect_dataset.py --csv PATH --out runs/inspect.json [--mapping MAPPING.json]

Without --mapping: raw header, dtypes, missing values, exact-duplicate rows, label-column candidates.
With a reviewed mapping: also canonical conversion results, rows dropped by reason, class distribution
and duplicate/conflict counts on the model-input features.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from _common import DEFAULT_MAX_ROWS, fail, load_dataset, write_json
from nids_ml.contracts import NidsMLError
from nids_ml.training import duplicate_report, prepare_frames


def inspect_raw(path: Path, max_rows: int) -> dict:
    raw = pd.read_csv(path, nrows=max_rows + 1, low_memory=False, skipinitialspace=True)
    truncated = len(raw) > max_rows
    raw = raw.iloc[:max_rows]
    raw.columns = [str(c).strip() for c in raw.columns]
    num = raw.select_dtypes("number")
    inf_cols = {c: int(((num[c] == float("inf")) | (num[c] == float("-inf"))).sum()) for c in num.columns}
    cand = [c for c in raw.columns if raw[c].dtype == object and raw[c].nunique() <= 60]
    return {
        "file": path.name, "rows_read": int(len(raw)), "truncated_at_max_rows": bool(truncated),
        "columns": [{"name": c, "dtype": str(raw[c].dtype), "missing": int(raw[c].isna().sum()),
                     "n_unique": int(raw[c].nunique(dropna=True))} for c in raw.columns],
        "infinite_values_by_column": {k: v for k, v in inf_cols.items() if v},
        "exact_duplicate_rows": int(raw.duplicated().sum()),
        "low_cardinality_text_columns_possible_labels_or_categoricals": cand,
        "label_distribution_candidates": {c: {str(k): int(v) for k, v in raw[c].value_counts().head(60).items()}
                                          for c in cand if c.lower() in ("label", "attack_cat", "type", "class", "attack")},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--mapping"); ap.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS)
    a = ap.parse_args(argv)
    try:
        out: dict = {"raw": inspect_raw(Path(a.csv), a.max_rows)}
        if a.mapping:
            frame, labels, _, rep, mapping = load_dataset(a.csv, a.mapping, a.max_rows)
            mf, labs = prepare_frames(frame, labels)
            out["mapping"] = {"report": rep.model_dump(), "license_note": mapping.license_note, "review_notes": mapping.review_notes}
            out["model_feature_duplicates"] = duplicate_report(mf, labs.tolist())
            total = len(labs)
            out["class_distribution"] = {k: {"count": int(v), "share": float(v) / total}
                                         for k, v in labs.value_counts().items()}
        write_json(Path(a.out), out)
    except (NidsMLError, OSError, ValueError) as exc:
        return fail(getattr(exc, "message", type(exc).__name__))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
