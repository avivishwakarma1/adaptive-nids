"""SYNTHETIC TEST FIXTURE - NOT REAL NETWORK TRAFFIC.

Generated from fixed random seeds with invented distributions purely to exercise code paths
(validation, preprocessing, splitting, thresholds, lifecycle). Nothing measured on this data says
anything about detection quality on UNSW-NB15, CICIDS2017, ToN-IoT or any live network, and no
number derived from it may be reported as a result.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

SYNTHETIC_BANNER = "synthetic-test-fixture (NOT real traffic)"
CLASS_SIZES = {"benign": 1400, "port_scan": 260, "dos_flood": 300, "brute_force": 140, "exfiltration": 60}


def _row(rng: np.random.Generator, cls: str, t: datetime) -> dict:
    # 6% of attack rows imitate benign traffic so classes overlap and nothing is trivially separable.
    mimic = cls != "benign" and rng.random() < 0.06
    k = "benign" if mimic else cls
    if k == "benign":
        dur = rng.gamma(2.0, 1.5); pk = max(1.0, rng.gamma(3.0, 6.0)); by = pk * rng.uniform(80, 900)
        proto = rng.choice(["tcp", "udp", "icmp"], p=[0.72, 0.25, 0.03])
        dport = int(rng.choice([80, 443, 53, 123, 8080, 22, 25, 3306], p=[.25, .3, .2, .05, .08, .05, .03, .04]))
        state = rng.choice(["sf", "s0", "rej", "con", "fin"], p=[.7, .05, .05, .15, .05])
        direction = rng.choice(["outbound", "internal", "inbound"], p=[.6, .3, .1])
    elif k == "port_scan":
        dur = rng.exponential(0.05); pk = float(rng.integers(1, 4)); by = pk * rng.uniform(40, 70)
        proto = "tcp"; dport = int(rng.integers(1, 65535)); state = rng.choice(["s0", "rej"], p=[.7, .3]); direction = "inbound"
    elif k == "dos_flood":
        dur = rng.exponential(2.0); pk = rng.gamma(20, 200); by = pk * rng.uniform(40, 120)
        proto = rng.choice(["udp", "tcp", "icmp"], p=[.5, .4, .1]); dport = int(rng.choice([80, 443, 53])); state = rng.choice(["s0", "sf"], p=[.8, .2]); direction = "inbound"
    elif k == "brute_force":
        dur = rng.gamma(2, 0.8); pk = rng.gamma(6, 4); by = pk * rng.uniform(100, 300)
        proto = "tcp"; dport = int(rng.choice([22, 21, 3389], p=[.7, .15, .15])); state = rng.choice(["sf", "rej", "s0"], p=[.5, .3, .2]); direction = "inbound"
    else:  # exfiltration
        dur = rng.gamma(6, 20); pk = rng.gamma(30, 60); by = pk * rng.uniform(900, 1400)
        proto = "tcp"; dport = int(rng.choice([443, 8443, 21])); state = "sf"; direction = "outbound"
    dur = float(max(dur, 1e-4))
    rec = {
        "duration": dur, "protocol": str(proto), "source_port": int(rng.integers(1024, 65535)),
        "destination_port": dport, "flow_packets": float(pk), "flow_bytes": float(by),
        "packets_per_second": float(pk / dur), "bytes_per_second": float(by / dur),
        "connection_state": str(state), "direction": str(direction),
        "observed_at": t.strftime("%Y-%m-%dT%H:%M:%SZ"), "source_id": None,
    }
    for f in ("flow_packets", "flow_bytes", "packets_per_second", "bytes_per_second"):
        if rng.random() < 0.04:
            rec[f] = None  # missing values
    return rec


def make_records(seed: int = 7, sizes: dict[str, int] | None = None) -> tuple[list[dict], list[str]]:
    rng = np.random.default_rng(seed)
    sizes = sizes or CLASS_SIZES
    labels = [c for c, n in sizes.items() for _ in range(n)]
    rng.shuffle(labels)
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    recs = [_row(rng, c, t0 + timedelta(seconds=int(i * 3))) for i, c in enumerate(labels)]
    for i, r in enumerate(recs):
        r["source_id"] = f"syn-{seed}-{i}"
    return recs, labels


def make_frame(seed: int = 7, sizes: dict[str, int] | None = None) -> tuple[pd.DataFrame, pd.Series]:
    recs, labels = make_records(seed, sizes)
    df = pd.DataFrame(recs)
    df["observed_at"] = pd.to_datetime(df["observed_at"], utc=True)
    return df, pd.Series(labels, dtype="object")
