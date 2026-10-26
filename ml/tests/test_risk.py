import math

import pytest

from nids_ml import severity_for_score
from nids_ml.risk import RiskPolicy, compute_risk_score

@pytest.mark.parametrize("score,sev", [
    (0, "info"), (19, "info"), (20, "low"), (39, "low"), (40, "medium"), (59, "medium"),
    (60, "high"), (79, "high"), (80, "critical"), (100, "critical"),
])
def test_severity_boundaries(score, sev):
    assert severity_for_score(score) == sev


def test_severity_clamps_and_type_checks():
    assert severity_for_score(-5) == "info" and severity_for_score(500) == "critical"
    for bad in (1.5, "50", None, True):
        with pytest.raises(TypeError):
            severity_for_score(bad)


def test_known_attack_formula_matches_documentation():
    p = RiskPolicy(class_weights={"port_scan": 0.8})
    assert compute_risk_score("known_attack", "port_scan", 0.9, 0.5, p) == round(100 * 0.8 * (0.7 * 0.9 + 0.3 * 0.5))
    assert compute_risk_score("known_attack", "other", 0.9, None, p) == round(100 * 0.7 * 0.9)


def test_policy_bounds_and_monotonicity():
    p = RiskPolicy()
    prev = -1
    for c in [i / 20 for i in range(21)]:
        r = compute_risk_score("known_attack", "x", c, 0.5, p)
        assert 0 <= r <= 100 and r >= prev
        prev = r
    assert compute_risk_score("anomaly", "unknown", None, 1.0, p) <= 75
    assert compute_risk_score("benign", "benign", 0.99, 1.0, p) < 20  # benign never rises above 'info'


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.1])
def test_non_finite_or_out_of_range_inputs_rejected(bad):
    with pytest.raises(ValueError):
        compute_risk_score("known_attack", "x", bad, 0.5, RiskPolicy())


def test_known_attack_requires_confidence_and_policy_validates():
    with pytest.raises(ValueError):
        compute_risk_score("known_attack", "x", None, 0.5, RiskPolicy())
    with pytest.raises(ValueError):
        RiskPolicy(w_conf=0.5, w_anom=0.6)
    with pytest.raises(ValueError):
        RiskPolicy(class_weights={"x": 1.5})


def test_rounding_is_half_up_and_deterministic():
    p = RiskPolicy(default_attack_weight=1.0, w_conf=1.0, w_anom=0.0)
    assert compute_risk_score("known_attack", "x", 0.795, None, p) == 80  # 79.5 -> 80
    assert compute_risk_score("known_attack", "x", 0.795, None, p) == 80
