from __future__ import annotations

import numpy as np

from e01_onset_discovery.full_state_graph import (
    ORACLE_VIEW,
    PRIMARY_VIEW,
    VIEWS,
    feature_names,
    graph_views,
)


def _fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3401)
    state = rng.poisson(1.2, size=100).astype(np.int64)
    state[0] += 1
    beta = np.exp(rng.normal(-3.0, 1.0, size=(100, 100)))
    target = rng.random(100)
    target /= target.sum()
    return state, beta, target


def _views(state: np.ndarray, beta: np.ndarray, target: np.ndarray):
    return graph_views(
        state,
        beta,
        target,
        generation_local_step=7,
        observation_kind="molecular_update",
        completed_fissions=11,
        batch_step=52,
        landmark=96,
        target_component_fraction=0.42,
    )


def test_schema_finite_and_replay() -> None:
    state, beta, target = _fixture()
    first = _views(state, beta, target)
    replay = _views(state.copy(), beta.copy(), target.copy())
    assert tuple(first) == VIEWS
    assert all(np.isfinite(first[key]).all() for key in VIEWS)
    assert all(np.array_equal(first[key], replay[key]) for key in VIEWS)
    assert all(len(first[key]) == len(feature_names()[key]) for key in VIEWS)


def test_simultaneous_molecule_permutation_invariance() -> None:
    state, beta, target = _fixture()
    order = np.random.default_rng(3402).permutation(100)
    first = _views(state, beta, target)
    permuted = _views(
        state[order], beta[np.ix_(order, order)], target[order]
    )
    assert all(
        np.allclose(first[key], permuted[key], atol=1e-10, rtol=1e-10)
        for key in VIEWS
    )


def test_primary_is_target_invariant_but_oracle_is_not() -> None:
    state, beta, target = _fixture()
    alternative = np.roll(target, 1)
    first = _views(state, beta, target)
    second = _views(state, beta, alternative)
    assert np.array_equal(first[PRIMARY_VIEW], second[PRIMARY_VIEW])
    assert not np.array_equal(first[ORACLE_VIEW], second[ORACLE_VIEW])


def test_beta_change_affects_primary_graph_signature() -> None:
    state, beta, target = _fixture()
    changed = beta.copy()
    changed[0, 1] *= 100
    assert not np.array_equal(
        _views(state, beta, target)[PRIMARY_VIEW],
        _views(state, changed, target)[PRIMARY_VIEW],
    )
