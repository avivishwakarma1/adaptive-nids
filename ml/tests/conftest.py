"""Shared fixtures. All data comes from synthetic_fixture.py (SYNTHETIC, not real traffic)."""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

# Make `pytest ml/tests` work from the repository root (INTEGRATION_GUIDE) as well as from ml/.
_ML = Path(__file__).resolve().parents[1]
for _p in (_ML / "scripts", _ML / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from nids_ml.anomaly import AnomalyConfig
from nids_ml.classification import ClassifierConfig
from nids_ml.training import TrainingConfig, train_bundle
from synthetic_fixture import make_frame

FIXED_NOW = datetime(2026, 10, 25, 12, 0, 0, tzinfo=timezone.utc)


def fixed_clock():
    return FIXED_NOW


def small_cfg(version: str = "base-1", **kw) -> TrainingConfig:
    return TrainingConfig(model_version=version, classifier=ClassifierConfig(n_estimators=40),
                          anomaly=AnomalyConfig(n_estimators=40), **kw)


@pytest.fixture(scope="session")
def synthetic():
    return make_frame(seed=7)


@pytest.fixture(scope="session")
def outcome(synthetic):
    df, y = synthetic
    return train_bundle(df, y, small_cfg())


@pytest.fixture()
def bundle(outcome):
    return outcome.bundle


@pytest.fixture()
def test_records(outcome):
    return outcome.subset("test")
