from __future__ import annotations

import numpy as np

from e01_attractor_onset_early_warning.core import (
    FEATURE_GROUPS,
    build_landmark_target,
    extract_past_features,
    metric_summary,
)


def test_landmark_target_boundaries() -> None:
    labels = np.zeros(220, dtype=bool)
    labels[64] = True
    result = build_landmark_target(labels)
    assert result["atRiskAtLandmark"] is True
    assert result["eventWithinHorizon"] is True
    labels[:] = False
    labels[63] = True
    result = build_landmark_target(labels)
    assert result["atRiskAtLandmark"] is False
    assert result["eventWithinHorizon"] is None
    labels[:] = False
    labels[192] = True
    result = build_landmark_target(labels)
    assert result["atRiskAtLandmark"] is True
    assert result["eventWithinHorizon"] is False


def test_metric_summary_is_deterministic_and_finite_aware() -> None:
    values = np.array([1.0, np.nan, -1.0, 3.0])
    result = metric_summary(values, "x")
    assert result["x_finite_fraction"] == 0.75
    assert result["x_last"] == 3.0
    assert result["x_positive_fraction"] == 2.0 / 3.0


def test_past_features_have_exact_registered_schema() -> None:
    rng = np.random.default_rng(9)
    states = rng.integers(0, 5, size=(64, 100), dtype=np.int64)
    states[:, 0] += 1
    generations = np.repeat(np.arange(8), 8)
    kinds = ["post_fission" if i % 8 == 0 else "molecular_update" for i in range(64)]
    first = extract_past_features(states, generations, kinds)
    second = extract_past_features(states.copy(), generations.copy(), list(kinds))
    expected = set().union(*map(set, FEATURE_GROUPS.values()))
    assert set(first) == expected
    assert first == second
    assert all(np.isfinite(value) for value in first.values())


def test_suffix_cannot_change_past_features() -> None:
    rng = np.random.default_rng(10)
    states = rng.integers(0, 4, size=(230, 100), dtype=np.int64)
    states[:, 0] += 1
    generations = np.arange(230) // 5
    kinds = np.array(["molecular_update"] * 230, dtype=object)
    before = extract_past_features(states[:64], generations[:64], kinds[:64])
    rng.shuffle(states[64:], axis=0)
    after = extract_past_features(states[:64], generations[:64], kinds[:64])
    assert before == after
