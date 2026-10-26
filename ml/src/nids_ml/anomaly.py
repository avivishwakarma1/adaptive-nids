"""Anomaly detector: Isolation Forest fitted on BENIGN training flows (a baseline of normal).

`anomaly_score` in [0, 1] is a *rank* score: the fraction of held-out benign validation flows whose
raw Isolation-Forest score is lower than this flow's (empirical CDF, piecewise-linear). 0.95 means
"more anomalous than ~95% of held-out benign validation flows". It is neither a probability of
attack nor comparable across models/datasets. A high score can be benign novelty.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from sklearn.ensemble import IsolationForest

from .contracts import ConfigError

_N_QUANTILES = 1001


class AnomalyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed: int = 20260101
    n_estimators: int = Field(default=200, ge=10)
    max_samples: int = Field(default=256, ge=16)
    # Operating point: share of held-out benign validation flows allowed to alert. This is a
    # configuration choice, not a tuned result; the actual threshold is computed from validation data.
    target_false_positive_rate: float = Field(default=0.05, gt=0, lt=1)


class AnomalyDetector:
    def __init__(self, config: AnomalyConfig):
        self.config = config
        self.forest = IsolationForest(
            n_estimators=config.n_estimators, max_samples=config.max_samples, contamination="auto",
            random_state=config.seed, n_jobs=1,
        )
        self.ref_values_: Optional[np.ndarray] = None
        self.ref_levels_: Optional[np.ndarray] = None
        self.fitted_on_n_: int = 0
        self.reference_n_: int = 0

    def fit(self, X_benign_train: np.ndarray) -> "AnomalyDetector":
        if len(X_benign_train) < 2 * self.config.max_samples // 16:
            raise ConfigError("too few benign training rows to fit the anomaly detector")
        self.forest.fit(X_benign_train)
        self.fitted_on_n_ = int(len(X_benign_train))
        return self

    def raw_scores(self, X: np.ndarray) -> np.ndarray:
        r = -np.asarray(self.forest.score_samples(X), dtype="float64")  # higher = more anomalous
        if not np.all(np.isfinite(r)):
            raise ValueError("anomaly detector produced non-finite scores")
        return r

    def fit_reference(self, X_benign_validation: np.ndarray) -> "AnomalyDetector":
        """Fit the score normalisation on held-out benign VALIDATION data (never on test data)."""
        if len(X_benign_validation) < 20:
            raise ConfigError("need at least 20 held-out benign validation rows to normalise anomaly scores")
        raw = self.raw_scores(X_benign_validation)
        levels = np.linspace(0.0, 1.0, _N_QUANTILES)
        vals = np.quantile(raw, levels)
        # np.interp needs strictly increasing x: collapse ties to the highest level they reach.
        uniq, last_idx = np.unique(vals[::-1], return_index=True)
        order_vals = uniq
        order_levels = levels[::-1][last_idx]
        self.ref_values_ = order_vals.astype("float64")
        self.ref_levels_ = order_levels.astype("float64")
        self.reference_n_ = int(len(raw))
        return self

    def scores(self, X: np.ndarray) -> np.ndarray:
        if self.ref_values_ is None:
            raise ConfigError("anomaly detector has no fitted score normalisation")
        raw = self.raw_scores(X)
        s = np.interp(raw, self.ref_values_, self.ref_levels_, left=0.0, right=1.0)
        return np.clip(s, 0.0, 1.0)

    def threshold_for_fpr(self, X_benign_validation: np.ndarray, target_fpr: Optional[float] = None) -> float:
        """Smallest threshold t such that at most `target_fpr` of validation benign flows score >= t."""
        fpr = self.config.target_false_positive_rate if target_fpr is None else target_fpr
        s = self.scores(X_benign_validation)
        return float(np.clip(np.quantile(s, 1.0 - fpr, method="higher"), 0.0, 1.0))
