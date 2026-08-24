from __future__ import annotations

import numpy as np

from e01_latent_timebase.core import ExposureDefinition, SimulationDefinition
from e01_onset_discovery.empirical_committor import (
    RestoredState,
    corrected_between_state_variance,
    dominant_component_centroid,
    simulate_branch,
)


def definition() -> SimulationDefinition:
    return SimulationDefinition(
        daughter_rule="FIRST_DAUGHTER",
        overshoot_rule="TRIM_NEW_ENTRANTS_TO_NMAX",
        exposure=ExposureDefinition(family="FIXED_COMMON_EXPOSURE", h=0.6),
    )


def streams(seed: int) -> tuple[np.random.Generator, ...]:
    return tuple(np.random.Generator(np.random.PCG64DXSM(seed + i)) for i in range(4))


def test_dominant_component_centroid_is_deterministic() -> None:
    states = np.zeros((6, 100), dtype=np.int64)
    states[:4, 0] = 39
    states[:4, 1] = 1
    states[4:, 2] = 40
    first, component = dominant_component_centroid(states)
    second, replay_component = dominant_component_centroid(states.copy())
    assert component == (0, 1, 2, 3)
    assert replay_component == component
    assert np.array_equal(first, second)
    assert np.isclose(first.sum(), 1.0)


def test_branch_exact_replay_and_horizon() -> None:
    state = np.zeros(100, dtype=np.int64)
    state[:40] = 1
    target = state.astype(np.float64) / state.sum()
    beta = np.exp(np.full((100, 100), -4.0, dtype=np.float64))
    restored = RestoredState(tuple(map(int, state)), "post_fission", 1, 1, 0, 4)
    first = simulate_branch(
        restored=restored,
        beta=beta,
        definition=definition(),
        target_centroid=target,
        event_rng=streams(101)[0],
        trim_rng=streams(101)[1],
        fission_rng=streams(101)[2],
        daughter_rng=streams(101)[3],
    )
    replay = simulate_branch(
        restored=restored,
        beta=beta.copy(),
        definition=definition(),
        target_centroid=target.copy(),
        event_rng=streams(101)[0],
        trim_rng=streams(101)[1],
        fission_rng=streams(101)[2],
        daughter_rng=streams(101)[3],
    )
    assert first == replay
    assert first.selected_observations_generated == 32
    assert first.path_sha256 == replay.path_sha256


def test_boundary_fissions_before_next_growth_update() -> None:
    state = np.zeros(100, dtype=np.int64)
    state[:80] = 1
    target = state.astype(np.float64) / state.sum()
    beta = np.exp(np.full((100, 100), -4.0, dtype=np.float64))
    restored = RestoredState(tuple(map(int, state)), "molecular_update", 0, 1, 9, 9)
    result = simulate_branch(
        restored=restored,
        beta=beta,
        definition=definition(),
        target_centroid=target,
        event_rng=streams(202)[0],
        trim_rng=streams(202)[1],
        fission_rng=streams(202)[2],
        daughter_rng=streams(202)[3],
        horizon=1,
    )
    assert result.fissions == 1
    assert result.molecular_updates == 0
    assert result.selected_observations_generated == 1


def test_corrected_variance_subtracts_binomial_noise() -> None:
    q = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
    result = corrected_between_state_variance(q, 128)
    expected = np.var(q, ddof=1) - np.mean(q * (1 - q) / 127)
    assert np.isclose(result["correctedBetweenStateVariance"], expected)
