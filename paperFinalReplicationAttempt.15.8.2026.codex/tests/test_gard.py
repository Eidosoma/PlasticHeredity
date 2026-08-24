from dataclasses import replace

import numpy as np

from aor_replication.config import GardConfig
from aor_replication.gard import (
    PHASE_FISSION,
    InterventionDecision,
    gard_propensities,
    simulate_gard,
)


def small_config() -> GardConfig:
    return GardConfig(
        n_types=12,
        initial_size=6,
        max_size=12,
        generations=8,
        max_steps_per_generation=100,
        beta_log_sigma=1.0,
        tau=2.0,
    )


def test_gard_is_reproducible_and_respects_invariants() -> None:
    config = small_config()
    first = simulate_gard(config, 17)
    second = simulate_gard(config, 17)
    np.testing.assert_array_equal(first.counts, second.counts)
    np.testing.assert_array_equal(first.beta, second.beta)
    assert first.counts.shape[1] == config.n_types
    assert np.all(first.counts >= 0)
    assert np.all(first.counts.sum(axis=1) >= 1)
    assert np.all(first.counts.sum(axis=1) <= config.max_size)
    assert np.sum(first.phases == PHASE_FISSION) == config.generations
    first.validate(config)


def test_zero_event_recording_convention_is_configurable() -> None:
    recorded = GardConfig(
        n_types=8,
        initial_size=4,
        max_size=8,
        generations=3,
        max_steps_per_generation=20,
        forward_rate=1e-8,
        backward_rate=1e-8,
        tau=0.1,
        record_zero_event_steps=True,
    )
    omitted = replace(recorded, record_zero_event_steps=False)
    with_zeroes = simulate_gard(recorded, 99)
    without_zeroes = simulate_gard(omitted, 99)
    assert len(with_zeroes.counts) > len(without_zeroes.counts)
    assert len(without_zeroes.counts) == omitted.generations + 1


def test_beta_orientation_matches_gard_equation() -> None:
    config = replace(
        small_config(),
        n_types=2,
        initial_size=2,
        max_size=4,
        forward_rate=1.0,
        backward_rate=1.0,
        environment_concentration=1.0,
    )
    counts = np.array([1, 1], dtype=np.int64)
    beta = np.array([[2.0, 4.0], [6.0, 8.0]])
    joins, leaves = gard_propensities(counts, beta, config)
    np.testing.assert_allclose(joins, [8.0, 16.0])
    np.testing.assert_allclose(leaves, [4.0, 8.0])


def test_intervention_is_applied_once_per_generation() -> None:
    config = small_config()

    def add_zero(parent, daughter, history, generation):
        del parent, history, generation
        if daughter.sum() < config.max_size:
            return InterventionDecision(species=0, delta=1, score=2.5)
        return InterventionDecision()

    trace = simulate_gard(config, 4, intervention=add_zero)
    fission_rows = trace.phases == PHASE_FISSION
    assert np.all(trace.intervention_delta[fission_rows] == 1)
    assert np.all(trace.intervention_species[fission_rows] == 0)
    assert np.all(trace.intervention_score[fission_rows] == 2.5)
