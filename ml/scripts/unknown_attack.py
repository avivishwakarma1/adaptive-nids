#!/usr/bin/env python
"""Held-out attack class / family protocol (objective 6).

  python scripts/unknown_attack.py --csv data/X.csv --mapping configs/mappings/x.json --dataset-name x \
      --held-out backdoor dos --family dos=dos,ddos --out runs/unknown_attack.json [--config configs/train_default.json]

Each held-out name (class or --family alias) is one fold: it is removed from training AND validation, thresholds
are selected without it, and it is scored together with benign/known TEST rows. The output states what the
experiment does and does not establish. Treat results as dataset-specific evidence, never as proof of
general unknown-attack detection.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _common import DEFAULT_MAX_ROWS, fail, load_dataset, read_json_config, write_json
from nids_ml.contracts import NidsMLError
from nids_ml.evaluation import UnknownAttackConfig, run_unknown_attack_protocol
from nids_ml.training import TrainingConfig


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True); ap.add_argument("--mapping", required=True)
    ap.add_argument("--dataset-name", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--held-out", nargs="+", required=True); ap.add_argument("--family", action="append", default=[])
    ap.add_argument("--config"); ap.add_argument("--min-heldout-support", type=int, default=30)
    ap.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS)
    a = ap.parse_args(argv)
    try:
        fam = {}
        for f in a.family:
            name, _, members = f.partition("=")
            fam[name] = [m for m in members.split(",") if m]
        cfg = UnknownAttackConfig(
            held_out=a.held_out, families=fam, min_heldout_support=a.min_heldout_support,
            training=TrainingConfig.model_validate({**read_json_config(a.config), "model_version": "unknown-attack-fold"}))
        frame, labels, _, map_rep, _ = load_dataset(a.csv, a.mapping, a.max_rows)
        rep = run_unknown_attack_protocol(frame, labels, cfg)
        write_json(Path(a.out), {"dataset_name": a.dataset_name, "mapping_report": map_rep.model_dump(),
                                 **rep.model_dump(mode="json")})
    except (NidsMLError, OSError, ValueError) as exc:
        return fail(getattr(exc, "message", None) or f"{type(exc).__name__}: {exc}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
