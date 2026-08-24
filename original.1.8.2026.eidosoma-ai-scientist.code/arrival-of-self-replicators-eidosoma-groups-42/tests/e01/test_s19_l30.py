from __future__ import annotations

import numpy as np

from e01_latent_timebase.core import ExposureDefinition, SimulationDefinition
from e01_onset_discovery.empirical_committor import RestoredState, simulate_branch


def _definition() -> SimulationDefinition:
    return SimulationDefinition(
        daughter_rule="FIRST_DAUGHTER",
        overshoot_rule="TRIM_NEW_ENTRANTS_TO_NMAX",
        exposure=ExposureDefinition(family="FIXED_COMMON_EXPOSURE", h=0.6),
    )


def _streams(seed: int) -> tuple[np.random.Generator, ...]:
    return tuple(
        np.random.Generator(np.random.PCG64DXSM(seed + offset)) for offset in range(4)
    )


def test_eight_step_branch_exact_replay() -> None:
    state = np.zeros(100, dtype=np.int64)
    state[:40] = 1
    target = state.astype(np.float64) / state.sum()
    beta = np.exp(np.full((100, 100), -4.0, dtype=np.float64))
    restored = RestoredState(tuple(map(int, state)), "post_fission", 1, 1, 0, 4)

    def run() -> object:
        streams = _streams(8301)
        return simulate_branch(
            restored=restored,
            beta=beta,
            definition=_definition(),
            target_centroid=target,
            event_rng=streams[0],
            trim_rng=streams[1],
            fission_rng=streams[2],
            daughter_rng=streams[3],
            horizon=8,
        )

    first = run()
    replay = run()
    assert first == replay
    assert first.selected_observations_generated == 8
    assert first.path_sha256 == replay.path_sha256


def test_jeffreys_short_branch_estimate_is_bounded() -> None:
    estimates = np.asarray([(successes + 0.5) / 65 for successes in range(65)])
    assert np.all((estimates > 0) & (estimates < 1))
    assert np.all(np.diff(estimates) > 0)
