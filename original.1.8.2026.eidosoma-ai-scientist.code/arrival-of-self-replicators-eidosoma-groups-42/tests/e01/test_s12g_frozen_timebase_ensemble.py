from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from e01_frozen_timebase_ensemble.core import (
    derive_seed,
    frozen_clr,
    post_fission_endpoint_records,
    selected_clock_observations,
)


def _observation(index: int, kind: str, generation: int, state: list[int]):
    return SimpleNamespace(
        observation_index=index,
        observation_kind=kind,
        growth_generation_one_based=generation,
        state=tuple(state),
    )


def _trajectory(generations: int = 2):
    state = [0] * 100
    state[0] = 40
    observations = [_observation(0, "initial_selected_state", 0, state)]
    index = 1
    for generation in range(1, generations + 1):
        for _ in range(2):
            state = state.copy()
            state[0] += 1
            observations.append(_observation(index, "molecular_update", generation, state))
            index += 1
        state = state.copy()
        state[0] //= 2
        observations.append(_observation(index, "post_fission", generation, state))
        index += 1
    return SimpleNamespace(
        trajectory_id="fixture",
        configuration_id="S12F-CANDIDATE-01",
        matrix_index=0,
        observations=tuple(observations),
        total_batch_updates=2 * generations,
        completed_fissions=generations,
    )


def test_locked_clock_cardinalities_and_kinds() -> None:
    trajectory = _trajectory()
    c0 = selected_clock_observations(trajectory, "C0_BATCH_UPDATES_ONLY")
    c1 = selected_clock_observations(trajectory, "C1_SELECTED_DAUGHTER_RETAINED")
    assert len(c0) == 5
    assert len(c1) == 7
    assert all(item.observation_kind != "post_fission" for item in c0)
    assert sum(item.observation_kind == "post_fission" for item in c1) == 2


def test_endpoint_mapping_respects_clock_without_synthetic_daughter() -> None:
    trajectory = _trajectory()
    c0 = post_fission_endpoint_records(
        trajectory, "C0_BATCH_UPDATES_ONLY", minimum_prior_transitions=0
    )
    c1 = post_fission_endpoint_records(
        trajectory, "C1_SELECTED_DAUGHTER_RETAINED", minimum_prior_transitions=0
    )
    assert [item.observation_kind for item in c0] == ["molecular_update"] * 2
    assert [item.observation_kind for item in c1] == ["post_fission"] * 2
    assert [item.raw_observation_index for item in c0] == [2, 5]
    assert [item.raw_observation_index for item in c1] == [3, 6]
    assert [item.selected_sequence_index for item in c0] == [2, 4]
    assert [item.selected_sequence_index for item in c1] == [3, 6]


def test_frozen_clr_is_finite_closed_and_drops_original_component_100() -> None:
    states = np.zeros((3, 100), dtype=np.int64)
    states[0, 0] = 40
    states[1, :40] = 1
    states[2, 99] = 80
    clr, masses, closure_error = frozen_clr(states)
    assert clr.shape == (3, 99)
    assert np.all(np.isfinite(clr))
    assert np.array_equal(masses, [40.0, 40.0, 80.0])
    assert np.max(closure_error) <= 2e-15
    # Dropping occurs after full 100-component centering, so a molecule only in
    # component 100 changes every retained coordinate.
    assert np.all(clr[2] < 0)


def test_seed_derivation_is_exact_and_domain_separated() -> None:
    first = derive_seed("source_preprocessing", "candidate-1", 0, "IIGR")
    assert first == derive_seed("source_preprocessing", "candidate-1", 0, "IIGR")
    assert first != derive_seed("source_partition", "candidate-1", 0, "IIGR")
    assert 0 <= first < 2**32
