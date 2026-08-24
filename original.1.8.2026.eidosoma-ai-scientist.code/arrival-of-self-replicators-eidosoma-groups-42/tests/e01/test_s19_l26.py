from __future__ import annotations

import numpy as np

from e01_onset_discovery.analog_committor import (
    all_analog_representations,
    deterministic_knn_probability,
    exact_h_trace_vector,
    ordinary_path_vector,
    recurrence_map_vector,
)


def fixture() -> np.ndarray:
    rng = np.random.default_rng(2601)
    values = rng.poisson(2.0, size=(64, 100)).astype(np.int64)
    values[:, 0] += 1
    return values


def test_representations_are_finite_and_exactly_replayable() -> None:
    values = fixture()
    first = all_analog_representations(values)
    replay = all_analog_representations(values.copy())
    assert recurrence_map_vector(values).shape == (1953,)
    assert exact_h_trace_vector(values).shape == (320,)
    assert ordinary_path_vector(values).shape == (384,)
    assert first.keys() == replay.keys()
    assert all(np.array_equal(first[key], replay[key]) for key in first)
    assert all(np.isfinite(value).all() for value in first.values())


def test_representations_are_molecule_relabelling_invariant() -> None:
    values = fixture()
    order = np.random.default_rng(2602).permutation(100)
    first = all_analog_representations(values)
    second = all_analog_representations(values[:, order])
    assert all(
        np.allclose(first[key], second[key], atol=1e-12, rtol=1e-12) for key in first
    )


def test_knn_probability_and_ties_are_deterministic() -> None:
    reference = np.asarray([[0.0], [0.0], [1.0], [2.0]], dtype=float)
    labels = np.asarray([0, 1, 1, 0])
    keys = [("B", 2), ("A", 1), ("C", 3), ("D", 4)]
    first = deterministic_knn_probability(
        np.asarray([0.0]), reference, labels, keys, k=2
    )
    replay = deterministic_knn_probability(
        np.asarray([0.0]), reference, labels, keys, k=2
    )
    assert first == replay
    assert first[0] == 0.5
    assert first[1] == (1, 0)


def test_temporal_order_is_retained() -> None:
    values = fixture()
    assert not np.array_equal(
        recurrence_map_vector(values), recurrence_map_vector(values[::-1])
    )
