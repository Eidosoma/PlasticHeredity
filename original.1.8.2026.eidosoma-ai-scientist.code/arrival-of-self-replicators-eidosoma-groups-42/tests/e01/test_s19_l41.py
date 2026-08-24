from __future__ import annotations

import numpy as np
import pytest

from e01_latent_timebase.core import ExposureDefinition, SimulationDefinition
from e01_onset_discovery.empirical_committor import RestoredState
from e01_onset_discovery.fission_clock_recurrence import (
    score_repeated_recurrence,
    simulate_fission_clock,
)

A = np.asarray([10, 0], dtype=np.int64)
B = np.asarray([0, 10], dtype=np.int64)
C = np.asarray([8, 2], dtype=np.int64)


def score(prefix, future):
    return score_repeated_recurrence(
        prefix_states=np.asarray(prefix, dtype=np.int64).reshape((-1, 2)),
        prefix_generations=np.arange(1, len(prefix) + 1, dtype=np.int64),
        future_states=np.asarray(future, dtype=np.int64).reshape((-1, 2)),
        future_generations=np.arange(
            len(prefix) + 1, len(prefix) + len(future) + 1, dtype=np.int64
        ),
        future_offsets_one_based=np.arange(2, 2 + 2 * len(future), 2),
    )


def test_two_far_to_near_returns_certify_online() -> None:
    result = score([A, B], [A, B, A])
    assert result.event
    assert result.return_boundary_count == 3
    assert result.first_return_boundary_one_based == 1
    assert result.certification_boundary_one_based == 2


def test_continuous_membership_is_not_repeated_return() -> None:
    result = score([A, B], [A, A, A])
    assert not result.event
    assert result.return_boundary_count == 1
    assert result.membership_only_event


def test_nonadjacent_generation_is_required() -> None:
    result = score([A], [A, B, A])
    assert result.return_boundary_count == 1
    assert not result.event


def test_order_changes_return_but_not_state_multiset() -> None:
    ordered = score([A, B], [A, B, A])
    permuted = score([A, B], [B, A, A])
    assert ordered.event
    assert not permuted.event


def test_empty_future_is_status_bearing() -> None:
    result = score([A, B], [])
    assert not result.event
    assert result.future_boundary_count == 0


def test_invalid_clock_rejected() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        score_repeated_recurrence(
            prefix_states=np.asarray([A, B]),
            prefix_generations=np.asarray([1, 2]),
            future_states=np.asarray([A, B]),
            future_generations=np.asarray([3, 3]),
            future_offsets_one_based=np.asarray([1, 2]),
        )


def test_fission_clock_exact_replay_and_horizon() -> None:
    restored = RestoredState(
        state=tuple([1] * 100),
        observation_kind="initial_selected_state",
        completed_fissions=0,
        growth_generation_one_based=1,
        generation_local_step=0,
        batch_step=0,
    )
    beta = np.zeros((100, 100), dtype=np.float64)
    definition = SimulationDefinition(
        daughter_rule="FIRST_DAUGHTER",
        overshoot_rule="TRIM_NEW_ENTRANTS_TO_NMAX",
        exposure=ExposureDefinition(family="FIXED_COMMON_EXPOSURE", h=0.6),
    )

    def run():
        streams = [np.random.default_rng(seed) for seed in (1, 2, 3, 4)]
        return simulate_fission_clock(
            restored=restored,
            beta=beta,
            definition=definition,
            event_rng=streams[0],
            trim_rng=streams[1],
            fission_rng=streams[2],
            daughter_rng=streams[3],
            future_fissions=4,
        )

    first = run()
    second = run()
    assert first == second
    assert first.fissions == 4
    assert len(first.future_states) == 4
