from __future__ import annotations

import numpy as np

from e01_onset_discovery.operator_change import (
    CHANNEL_SHIFT_FEATURES,
    EXACT_H_CHANGE_FEATURES,
    OPERATOR_CHANGE_FEATURES,
    OPERATOR_ONLY_FEATURES,
    extract_operator_change_features,
)


def fixture() -> np.ndarray:
    rng = np.random.default_rng(2501)
    values = rng.poisson(2.0, size=(64, 100)).astype(np.int64)
    values[:, 0] += 1
    return values


def test_schema_replay_and_finite() -> None:
    values = fixture()
    first = extract_operator_change_features(values)
    replay = extract_operator_change_features(values.copy())
    assert tuple(first) == OPERATOR_CHANGE_FEATURES
    assert first == replay
    assert len(CHANNEL_SHIFT_FEATURES) == 33
    assert len(EXACT_H_CHANGE_FEATURES) == 15
    assert len(OPERATOR_ONLY_FEATURES) == 13
    assert np.isfinite(list(first.values())).all()


def test_molecule_relabelling_invariance() -> None:
    values = fixture()
    order = np.random.default_rng(2502).permutation(100)
    first = extract_operator_change_features(values)
    relabelled = extract_operator_change_features(values[:, order])
    assert all(
        np.isclose(first[name], relabelled[name], atol=1e-10, rtol=1e-10)
        for name in first
    )


def test_half_order_is_directional() -> None:
    values = fixture()
    first = extract_operator_change_features(values)
    swapped = extract_operator_change_features(np.r_[values[32:], values[:32]])
    assert any(first[name] != swapped[name] for name in OPERATOR_CHANGE_FEATURES)
