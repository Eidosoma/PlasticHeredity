from __future__ import annotations

import numpy as np

from e01_onset_discovery.core import (
    DMD_FEATURES,
    EWS_FEATURES,
    RQA_FEATURES,
    extract_organization_warning_features,
)


def _states(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.poisson(2.0, size=(64, 100)).astype(np.int64)
    values[:, 0] += 1
    return values


def test_feature_schema_and_replay() -> None:
    states = _states()
    first = extract_organization_warning_features(states)
    second = extract_organization_warning_features(states.copy())
    expected = set(EWS_FEATURES) | set(RQA_FEATURES) | set(DMD_FEATURES)
    assert set(first) == expected
    assert first == second
    assert all(np.isfinite(list(first.values())))


def test_suffix_independence() -> None:
    rng = np.random.default_rng(11)
    prefix = _states(3)
    left = np.vstack([prefix, rng.poisson(2.0, size=(40, 100))])
    right = np.vstack([prefix, rng.poisson(8.0, size=(40, 100))])
    assert extract_organization_warning_features(
        left[:64]
    ) == extract_organization_warning_features(right[:64])


def test_order_sensitive_families_change() -> None:
    states = _states(19)
    permutation = np.r_[0, np.arange(1, 64)[::-1]]
    original = extract_organization_warning_features(states)
    changed = extract_organization_warning_features(states[permutation])
    assert any(original[name] != changed[name] for name in RQA_FEATURES + DMD_FEATURES)


def test_invalid_shape_is_rejected() -> None:
    try:
        extract_organization_warning_features(np.ones((63, 100), dtype=np.int64))
    except ValueError as exc:
        assert "64-by-100" in str(exc)
    else:
        raise AssertionError("invalid shape was accepted")
