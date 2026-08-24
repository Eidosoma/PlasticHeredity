from __future__ import annotations

import numpy as np

from e01_onset_discovery.transition_tube import TUBE_VIEWS, transition_tube_views


def fixture() -> np.ndarray:
    rng = np.random.default_rng(2701)
    values = rng.poisson(2.0, size=(32, 100)).astype(np.int64)
    values[:, 0] += 1
    return values


def test_schema_finite_and_replay() -> None:
    values = fixture()
    first = transition_tube_views(values)
    replay = transition_tube_views(values.copy())
    assert tuple(first) == TUBE_VIEWS
    assert [len(first[key]) for key in TUBE_VIEWS] == [693, 315, 378]
    assert all(np.array_equal(first[key], replay[key]) for key in TUBE_VIEWS)
    assert all(np.isfinite(first[key]).all() for key in TUBE_VIEWS)


def test_molecule_relabelling_invariance() -> None:
    values = fixture()
    order = np.random.default_rng(2702).permutation(100)
    first = transition_tube_views(values)
    second = transition_tube_views(values[:, order])
    assert all(
        np.allclose(first[key], second[key], atol=1e-12, rtol=1e-12)
        for key in TUBE_VIEWS
    )


def test_direction_is_retained() -> None:
    values = fixture()
    first = transition_tube_views(values)
    reverse = transition_tube_views(values[::-1])
    assert all(not np.array_equal(first[key], reverse[key]) for key in TUBE_VIEWS)
