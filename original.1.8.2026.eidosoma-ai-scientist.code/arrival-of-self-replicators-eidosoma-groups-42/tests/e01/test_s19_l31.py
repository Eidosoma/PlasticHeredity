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


def _run(horizon: int, seed: int) -> object:
    state = np.zeros(100, dtype=np.int64)
    state[:40] = 1
    target = state.astype(np.float64) / state.sum()
    beta = np.exp(np.full((100, 100), -4.0, dtype=np.float64))
    restored = RestoredState(tuple(map(int, state)), "post_fission", 1, 1, 0, 4)
    streams = _streams(seed)
    return simulate_branch(
        restored=restored,
        beta=beta,
        definition=_definition(),
        target_centroid=target,
        event_rng=streams[0],
        trim_rng=streams[1],
        fission_rng=streams[2],
        daughter_rng=streams[3],
        horizon=horizon,
    )


def test_confirmation_horizons_and_exact_replay() -> None:
    h8 = _run(8, 3108)
    h32 = _run(32, 3132)
    assert h8 == _run(8, 3108)
    assert h32 == _run(32, 3132)
    assert h8.selected_observations_generated == 8
    assert h32.selected_observations_generated == 32


def test_h8_and_h32_streams_are_domain_separated() -> None:
    h8 = _run(8, 4108)
    h32 = _run(32, 4132)
    assert h8.path_sha256 != h32.path_sha256
