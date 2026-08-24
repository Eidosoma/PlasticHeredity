from __future__ import annotations

import numpy as np

from e01_latent_timebase.core import ExposureDefinition, SimulationDefinition
from e01_onset_discovery.operator_memory import (
    VIEWS,
    feature_names,
    operator_memory_views,
)


def _definition() -> SimulationDefinition:
    return SimulationDefinition(
        daughter_rule="FIRST_DAUGHTER",
        overshoot_rule="TRIM_NEW_ENTRANTS_TO_NMAX",
        exposure=ExposureDefinition(family="FIXED_COMMON_EXPOSURE", h=0.6),
    )


def _fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(3301)
    states = rng.poisson(1.5, size=(8, 100)).astype(np.int64)
    states[:, 0] += 1
    beta = np.exp(rng.normal(-4.0, 0.4, size=(100, 100)))
    target = rng.random(100)
    target /= target.sum()
    kinds = ["molecular_update"] * 7 + ["post_fission"]
    return states, beta, target, kinds


def _views(
    states: np.ndarray, beta: np.ndarray, target: np.ndarray, kinds: list[str]
) -> dict[str, np.ndarray]:
    return operator_memory_views(
        states,
        beta,
        target,
        _definition(),
        observation_kinds=kinds,
        generation_local_steps=list(range(1, 9)),
        growth_generations=list(range(3, 11)),
        batch_steps=list(range(40, 48)),
        target_component_fraction=0.37,
    )


def test_schema_finite_and_exact_replay() -> None:
    states, beta, target, kinds = _fixture()
    first = _views(states, beta, target, kinds)
    replay = _views(states.copy(), beta.copy(), target.copy(), list(kinds))
    assert tuple(first) == VIEWS
    assert [len(first[key]) for key in VIEWS] == [15, 35, 51]
    assert all(np.array_equal(first[key], replay[key]) for key in VIEWS)
    assert all(len(first[key]) == len(feature_names()[key]) for key in VIEWS)


def test_simultaneous_molecule_permutation_invariance() -> None:
    states, beta, target, kinds = _fixture()
    order = np.random.default_rng(3302).permutation(100)
    first = _views(states, beta, target, kinds)
    permuted = _views(
        states[:, order], beta[np.ix_(order, order)], target[order], kinds
    )
    assert all(
        np.allclose(first[key], permuted[key], atol=1e-12, rtol=1e-12)
        for key in VIEWS
    )


def test_primary_basin_blind_view_is_exactly_target_invariant() -> None:
    states, beta, target, kinds = _fixture()
    alternative = np.random.default_rng(3303).random(100)
    alternative /= alternative.sum()
    first = _views(states, beta, target, kinds)
    second = _views(states, beta, alternative, kinds)
    assert np.array_equal(first["PHASE_MEMORY"], second["PHASE_MEMORY"])
    assert np.array_equal(
        first["BASIN_BLIND_OPERATOR_MEMORY"],
        second["BASIN_BLIND_OPERATOR_MEMORY"],
    )
    assert not np.array_equal(
        first["TARGET_CONDITIONED_OPERATOR_MEMORY"],
        second["TARGET_CONDITIONED_OPERATOR_MEMORY"],
    )


def test_history_order_changes_slopes_but_not_endpoints() -> None:
    states, beta, target, kinds = _fixture()
    first = _views(states, beta, target, kinds)
    order = np.asarray([6, 5, 4, 3, 2, 1, 0, 7])
    second = operator_memory_views(
        states[order],
        beta,
        target,
        _definition(),
        observation_kinds=[kinds[index] for index in order],
        generation_local_steps=[list(range(1, 9))[index] for index in order],
        growth_generations=[list(range(3, 11))[index] for index in order],
        batch_steps=[list(range(40, 48))[index] for index in order],
        target_component_fraction=0.37,
    )
    assert all(not np.array_equal(first[key], second[key]) for key in VIEWS)
    assert np.array_equal(first["PHASE_MEMORY"][:5], second["PHASE_MEMORY"][:5])
