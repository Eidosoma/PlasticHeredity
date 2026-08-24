from __future__ import annotations

import numpy as np

from e01_onset_discovery.reaction_coordinate import (
    EXACT_H_WINDOW_FEATURES,
    ORDINARY_WINDOW_FEATURES,
    REACTION_FEATURES,
    extract_window_features,
)


def fixture() -> np.ndarray:
    rng = np.random.default_rng(2401)
    values = rng.poisson(2.0, size=(32, 100)).astype(np.int64)
    values[:, 0] += 1
    return values


def test_reaction_feature_schema_and_replay() -> None:
    values = fixture()
    first = extract_window_features(values)
    second = extract_window_features(values.copy())
    assert tuple(first) == REACTION_FEATURES
    assert first == second
    assert set(EXACT_H_WINDOW_FEATURES) < set(ORDINARY_WINDOW_FEATURES) < set(
        REACTION_FEATURES
    )


def test_molecule_label_permutation_invariance() -> None:
    values = fixture()
    order = np.random.default_rng(2402).permutation(100)
    first = extract_window_features(values)
    second = extract_window_features(values[:, order])
    assert all(
        np.isclose(first[name], second[name], atol=1e-10, rtol=1e-10)
        for name in first
    )


def test_temporal_order_sensitivity() -> None:
    values = fixture()
    first = extract_window_features(values)
    second = extract_window_features(values[::-1])
    assert any(first[name] != second[name] for name in REACTION_FEATURES)
