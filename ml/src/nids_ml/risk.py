"""Risk score and severity (contract section 7). Policy version: risk-policy-1.0.0.

`risk_score` is a *priority score*, not a probability. The formula below is a documented design
choice with UNVALIDATED weights: it must be reviewed against real labels and analyst workload
before any production claim (contract section 7). Per-class weights are configuration; the default
for an attack class that has no explicit weight is `default_attack_weight`.

    known_attack : 100 * w_class * (w_conf*confidence + w_anom*anomaly_score)
                   (anomaly term omitted and w_conf renormalised to 1 when no detector is available)
    anomaly      : 100 * anomaly_only_weight * anomaly_score
    uncertain    : 100 * uncertain_weight * (anomaly_score, or anomaly_missing_standin if unavailable)
    benign       : 100 * benign_weight * (anomaly_score, or 0 if unavailable)

The result is rounded half-up, clamped to [0, 100] and mapped with `severity_for_score`
(0-19 info, 20-39 low, 40-59 medium, 60-79 high, 80-100 critical). The backend must not recompute it.
"""
from __future__ import annotations

import math
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .contracts import severity_for_score  # re-exported on purpose

RISK_POLICY_VERSION = "risk-policy-1.0.0"
__all__ = ["RISK_POLICY_VERSION", "RiskPolicy", "compute_risk_score", "severity_for_score"]


class RiskPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    policy_version: str = RISK_POLICY_VERSION
    class_weights: dict[str, float] = Field(default_factory=dict)
    default_attack_weight: float = Field(default=0.7, ge=0, le=1)
    w_conf: float = Field(default=0.7, ge=0, le=1)
    w_anom: float = Field(default=0.3, ge=0, le=1)
    anomaly_only_weight: float = Field(default=0.75, ge=0, le=1)
    uncertain_weight: float = Field(default=0.4, ge=0, le=1)
    benign_weight: float = Field(default=0.15, ge=0, le=1)
    anomaly_missing_standin: float = Field(default=0.5, ge=0, le=1)

    @field_validator("class_weights")
    @classmethod
    def _weights(cls, v: dict[str, float]) -> dict[str, float]:
        for k, w in v.items():
            if not (isinstance(w, (int, float)) and math.isfinite(w) and 0 <= w <= 1):
                raise ValueError(f"class weight for {k!r} must be within [0, 1]")
        return v

    @model_validator(mode="after")
    def _sum(self) -> "RiskPolicy":
        if abs(self.w_conf + self.w_anom - 1.0) > 1e-9:
            raise ValueError("w_conf + w_anom must equal 1")
        return self

    def weight_for(self, label: str) -> float:
        return self.class_weights.get(label, self.default_attack_weight)


def _half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


def compute_risk_score(prediction_type: str, label: str, confidence: Optional[float],
                       anomaly_score: Optional[float], policy: RiskPolicy) -> int:
    for name, v in (("confidence", confidence), ("anomaly_score", anomaly_score)):
        if v is not None and (not math.isfinite(v) or not 0.0 <= v <= 1.0):
            raise ValueError(f"{name} must be finite and within [0, 1]")
    if prediction_type == "known_attack":
        if confidence is None:
            raise ValueError("known_attack requires a classifier confidence")
        base = confidence if anomaly_score is None else policy.w_conf * confidence + policy.w_anom * anomaly_score
        raw = 100.0 * policy.weight_for(label) * base
    elif prediction_type == "anomaly":
        raw = 100.0 * policy.anomaly_only_weight * (anomaly_score or 0.0)
    elif prediction_type == "uncertain":
        a = policy.anomaly_missing_standin if anomaly_score is None else anomaly_score
        raw = 100.0 * policy.uncertain_weight * a
    elif prediction_type == "benign":
        raw = 100.0 * policy.benign_weight * (anomaly_score or 0.0)
    else:
        raise ValueError(f"unknown prediction_type {prediction_type!r}")
    return max(0, min(100, _half_up(raw)))
