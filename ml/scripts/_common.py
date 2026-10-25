"""Shared helpers for the offline scripts. No network, no subprocess, no shell: CSV in, JSON out."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from nids_ml.contracts import ConfigError
from nids_ml.features import apply_mapping, load_mapping

DEFAULT_MAX_ROWS = 1_000_000


def load_dataset(csv_path: str, mapping_path: str, max_rows: int = DEFAULT_MAX_ROWS):
    """CSV + REVIEWED mapping -> (canonical frame, labels, groups, mapping report, mapping)."""
    p = Path(csv_path)
    if not p.is_file():
        raise ConfigError("dataset file not found")
    mapping = load_mapping(mapping_path)  # refuses reviewed=false
    raw = pd.read_csv(p, nrows=max_rows + 1, low_memory=False, skipinitialspace=True)
    if len(raw) > max_rows:
        raise ConfigError(f"dataset exceeds --max-rows={max_rows}; sample or split it first (nothing was truncated silently)")
    frame, labels, groups, rep = apply_mapping(raw, mapping)
    return frame, labels, groups, rep, mapping


def read_json_config(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise ConfigError("cannot read JSON config file") from None
    return {k: v for k, v in data.items() if not k.startswith("_")}  # allow "_comment" keys


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False, default=str), encoding="utf-8")


def fail(msg: str) -> int:
    print(f"ERROR: {msg}", file=sys.stderr)
    return 2
