from __future__ import annotations

import numpy as np

from e01_latent_timebase.core import ExposureDefinition, SimulationDefinition
from e01_onset_discovery.empirical_committor import RestoredState
from e01_onset_discovery.fission_clock_recurrence import simulate_fission_clock
from e01_onset_discovery.functional_heredity_regime import (
    FissionInterval,
    functional_profile,
    growth_signature,
    mean_pairwise_cosine,
    mean_pairwise_distance,
    simulate_functional_fission_clock,
)


def _definition() -> SimulationDefinition:
    return SimulationDefinition(
        daughter_rule="FIRST_DAUGHTER",
        overshoot_rule="TRIM_NEW_ENTRANTS_TO_NMAX",
        exposure=ExposureDefinition(family="FIXED_COMMON_EXPOSURE", h=0.6),
    )


def _restored() -> RestoredState:
    state = np.zeros(100, dtype=np.int64)
    state[:40] = 1
    return RestoredState(
        state=tuple(map(int, state)),
        observation_kind="initial_selected_state",
        completed_fissions=0,
        growth_generation_one_based=1,
        generation_local_step=0,
        batch_step=0,
    )


def test_functional_trace_is_exact_l41_replay() -> None:
    beta = np.full((100, 100), 0.02, dtype=np.float64)
    seeds = (101, 102, 103, 104)
    canonical = simulate_fission_clock(
        restored=_restored(),
        beta=beta,
        definition=_definition(),
        event_rng=np.random.default_rng(seeds[0]),
        trim_rng=np.random.default_rng(seeds[1]),
        fission_rng=np.random.default_rng(seeds[2]),
        daughter_rng=np.random.default_rng(seeds[3]),
        future_fissions=3,
    )
    audited = simulate_functional_fission_clock(
        restored=_restored(),
        beta=beta,
        definition=_definition(),
        event_rng=np.random.default_rng(seeds[0]),
        trim_rng=np.random.default_rng(seeds[1]),
        fission_rng=np.random.default_rng(seeds[2]),
        daughter_rng=np.random.default_rng(seeds[3]),
        future_fissions=3,
    )
    assert audited.future_states == canonical.future_states
    assert audited.parent_daughter_h == canonical.parent_daughter_h
    assert audited.path_sha256 == canonical.path_sha256
    assert audited.final_state_sha256 == canonical.final_state_sha256
    assert audited.selected_observations_generated == canonical.selected_observations_generated
    assert sum(row.molecular_updates for row in audited.intervals) == audited.molecular_updates


def test_functional_profiles_are_feature_permutation_equivariant() -> None:
    rng = np.random.default_rng(88)
    state = rng.integers(1, 5, size=100, dtype=np.int64)
    beta = np.exp(rng.normal(-4, 1, size=(100, 100)))
    order = rng.permutation(100)
    original = functional_profile(state, beta)
    permuted = functional_profile(state[order], beta[order][:, order])
    np.testing.assert_allclose(
        permuted.catalytic_activation,
        original.catalytic_activation[order],
        atol=1e-14,
        rtol=1e-14,
    )
    np.testing.assert_allclose(
        permuted.expected_net_exchange,
        original.expected_net_exchange[order],
        atol=1e-14,
        rtol=1e-14,
    )


def test_pairwise_summaries_and_growth_signature() -> None:
    vectors = np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert np.isclose(mean_pairwise_cosine(vectors), 1.0 / 3.0)
    assert np.isclose(mean_pairwise_distance(vectors, np.ones(2)), np.sqrt(2) * 2 / 3 / np.sqrt(2))
    interval = FissionInterval(
        boundary_one_based=1,
        generation=2,
        selected_offset_one_based=12,
        start_state=(1,) * 100,
        pre_fission_state=(1,) * 100,
        daughter_state=(1,) * 100,
        parent_daughter_h=1.0,
        molecular_updates=10,
        nonzero_reaction_type_count=20,
        gross_sampled_event_count=30,
        maximum_overshoot=0,
        mean_exposure=0.6,
        pre_fission_mass=80,
        daughter_mass=40,
        complete_growth_interval=True,
    )
    value = growth_signature(interval)
    np.testing.assert_allclose(value, [np.log1p(10), np.log1p(30), 2.0, 0.5])
