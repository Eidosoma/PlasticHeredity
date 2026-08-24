from __future__ import annotations

import numpy as np

from e01_onset_discovery.multiscale_geometry import (
    INTRINSIC_GEOMETRY_FEATURES,
    PATH_GEOMETRY_FEATURES,
    TOPOLOGY_FEATURES,
    chord_distance_matrix,
    extract_multiscale_geometry_features,
)


def _states(seed: int = 17) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.poisson(2.0, size=(64, 100)).astype(np.int64)
    values[:, 0] += 1
    return values


def test_schema_finite_and_exact_replay() -> None:
    states = _states()
    first = extract_multiscale_geometry_features(states)
    second = extract_multiscale_geometry_features(states.copy())
    assert set(first) == (
        set(TOPOLOGY_FEATURES)
        | set(INTRINSIC_GEOMETRY_FEATURES)
        | set(PATH_GEOMETRY_FEATURES)
    )
    assert first == second
    assert np.isfinite(list(first.values())).all()


def test_feature_permutation_and_closure_scaling() -> None:
    states = _states(19)
    permutation = np.random.default_rng(3).permutation(100)
    original = extract_multiscale_geometry_features(states)
    permuted = extract_multiscale_geometry_features(states[:, permutation])
    for field in TOPOLOGY_FEATURES + INTRINSIC_GEOMETRY_FEATURES + PATH_GEOMETRY_FEATURES:
        assert np.isclose(original[field], permuted[field], atol=1e-10, rtol=1e-10)
    scaled = extract_multiscale_geometry_features(states * 7)
    for field in original:
        assert np.isclose(original[field], scaled[field], atol=1e-10, rtol=1e-10)


def test_temporal_order_changes_only_path_family() -> None:
    states = _states(23)
    order = np.r_[0, np.arange(1, 64)[::-1]]
    original = extract_multiscale_geometry_features(states)
    permuted = extract_multiscale_geometry_features(states[order])
    for field in TOPOLOGY_FEATURES + INTRINSIC_GEOMETRY_FEATURES:
        if field.endswith("_full"):
            assert np.isclose(original[field], permuted[field], atol=1e-10, rtol=1e-10)
    order_sensitive = tuple(
        field
        for field in TOPOLOGY_FEATURES
        + INTRINSIC_GEOMETRY_FEATURES
        + PATH_GEOMETRY_FEATURES
        if not field.endswith("_full") or field in PATH_GEOMETRY_FEATURES
    )
    assert any(not np.isclose(original[field], permuted[field]) for field in order_sensitive)


def test_chord_distance_is_symmetric_metric_input() -> None:
    states = _states(29).astype(float)
    compositions = states / states.sum(axis=1, keepdims=True)
    distance = chord_distance_matrix(compositions)
    assert np.allclose(distance, distance.T)
    assert np.allclose(np.diag(distance), 0.0)
    assert np.all(distance >= 0.0)


def test_prefix_extractor_rejects_wrong_shape() -> None:
    try:
        extract_multiscale_geometry_features(np.ones((63, 100), dtype=np.int64))
    except ValueError as exc:
        assert "64-by-100" in str(exc)
    else:
        raise AssertionError("wrong prefix shape was accepted")
