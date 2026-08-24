from __future__ import annotations

import numpy as np

from e01_onset_discovery.heredity_phi_incremental import (
    composition_controls,
    fit_binomial_ridge,
    metric_summary,
    predict_probability,
    probability_metrics,
)


def test_metric_summary_current_and_slope() -> None:
    result = metric_summary(np.arange(10, dtype=float), recent=8)
    assert result.current == 9
    assert abs(result.recent_slope - 1) < 1e-12
    assert result.finite_fraction == 1


def test_metric_summary_retains_nonfinite_status() -> None:
    result = metric_summary(np.array([0.0, 1.0, np.nan]))
    assert np.isnan(result.current)
    assert result.finite_fraction == 2 / 3


def test_composition_controls_identity_and_change() -> None:
    identity = composition_controls(np.array([[1, 1], [2, 2]]))
    changed = composition_controls(np.array([[1, 1], [2, 0]]))
    assert abs(identity["currentAdjacentMolecularH"] - 1) < 1e-12
    assert identity["currentCompositionChange"] == 0
    assert changed["currentAdjacentMolecularH"] < 1
    assert changed["currentCompositionChange"] > 0


def test_binomial_ridge_replays_and_orders_signal() -> None:
    x = np.arange(12, dtype=float)[:, None]
    successes = np.array([1, 1, 1, 1, 2, 3, 5, 7, 8, 9, 9, 9])
    trials = np.full(12, 10)
    first = fit_binomial_ridge(x, successes, trials, seed=42)
    second = fit_binomial_ridge(x, successes, trials, seed=42)
    p1 = predict_probability(first, x)
    p2 = predict_probability(second, x)
    assert np.array_equal(p1, p2)
    assert p1[-1] > p1[0]


def test_probability_metrics_perfect_q_prediction() -> None:
    q = np.array([0.2, 0.5, 0.8])
    k = np.array([2, 5, 8])
    n = np.array([10, 10, 10])
    result = probability_metrics(q, q, k, n)
    assert result["qBrier"] == 0
    assert result["spearman"] == 1
    assert abs(result["calibrationSlope"] - 1) < 1e-12
    assert abs(result["calibrationIntercept"]) < 1e-12
