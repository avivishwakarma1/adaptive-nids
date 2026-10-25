"""Known-class classifier wrapper (scikit-learn).

`confidence` is the highest per-class score the estimator outputs. It is NOT a calibrated
probability: tree ensembles and boosted models are routinely over/under-confident. Reliability is
measured (not assumed) in `evaluation.py` and reported next to the metrics.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .contracts import ConfigError


class ClassifierConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["random_forest", "logistic_regression", "hist_gradient_boosting"] = "random_forest"
    seed: int = 20260101
    n_estimators: int = Field(default=200, ge=10)          # random_forest only
    max_depth: Optional[int] = Field(default=None, ge=1)    # random_forest only
    min_samples_leaf: int = Field(default=2, ge=1)          # random_forest only
    C: float = Field(default=1.0, gt=0)                     # logistic_regression only
    max_iter: int = Field(default=300, ge=50)               # logistic_regression / hist_gradient_boosting


def build_estimator(cfg: ClassifierConfig):
    # Class imbalance: every estimator uses balanced class weights. n_jobs=1 keeps results
    # bit-for-bit reproducible across machines for a given library version.
    if cfg.kind == "random_forest":
        return RandomForestClassifier(
            n_estimators=cfg.n_estimators, max_depth=cfg.max_depth, min_samples_leaf=cfg.min_samples_leaf,
            class_weight="balanced_subsample", random_state=cfg.seed, n_jobs=1,
        )
    if cfg.kind == "logistic_regression":
        return LogisticRegression(C=cfg.C, max_iter=cfg.max_iter, class_weight="balanced", random_state=cfg.seed)
    if cfg.kind == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(max_iter=cfg.max_iter, class_weight="balanced", random_state=cfg.seed)
    raise ConfigError(f"unknown classifier kind {cfg.kind!r}")


class KnownClassClassifier:
    """Fitted classifier + class list. `scores()` returns (labels, confidence, full score matrix)."""

    def __init__(self, config: ClassifierConfig):
        self.config = config
        self.estimator = build_estimator(config)
        self.classes_: list[str] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KnownClassClassifier":
        y = np.asarray(y, dtype=object)
        if len(set(y.tolist())) < 2:
            raise ConfigError("classifier training needs at least two classes")
        self.estimator.fit(X, y)
        self.classes_ = [str(c) for c in self.estimator.classes_]
        return self

    def score_matrix(self, X: np.ndarray) -> np.ndarray:
        P = np.asarray(self.estimator.predict_proba(X), dtype="float64")
        if P.shape[1] != len(self.classes_) or not np.all(np.isfinite(P)):
            raise ValueError("classifier produced malformed or non-finite scores")
        return np.clip(P, 0.0, 1.0)

    def scores(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        P = self.score_matrix(X)
        idx = np.argmax(P, axis=1)  # ties -> first class in sorted order (deterministic)
        labels = np.asarray(self.classes_, dtype=object)[idx]
        return labels, P[np.arange(len(P)), idx], P
