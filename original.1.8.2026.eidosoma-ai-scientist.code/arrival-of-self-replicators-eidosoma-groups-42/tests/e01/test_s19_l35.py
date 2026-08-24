from __future__ import annotations

import numpy as np

from e01_latent_timebase.core import ExposureDefinition, SimulationDefinition
from e01_onset_discovery.branch_trace import simulate_branch_trace
from e01_onset_discovery.empirical_committor import RestoredState, simulate_branch


def test_trace_exactly_reproduces_compact_branch() -> None:
    rng = np.random.default_rng(3501)
    state = rng.poisson(1.2, size=100).astype(np.int64)
    state[0] += 1
    beta = np.exp(rng.normal(-3.0, 0.8, size=(100, 100)))
    target = rng.random(100)
    target /= target.sum()
    restored = RestoredState(
        tuple(map(int, state)), "molecular_update", 5, 6, 7, 31
    )
    definition = SimulationDefinition(
        daughter_rule="FIRST_DAUGHTER",
        overshoot_rule="TRIM_NEW_ENTRANTS_TO_NMAX",
        exposure=ExposureDefinition(family="FIXED_COMMON_EXPOSURE", h=0.6),
    )
    seeds = [3502, 3503, 3504, 3505]
    compact = simulate_branch(
        restored=restored,
        beta=beta,
        definition=definition,
        target_centroid=target,
        event_rng=np.random.default_rng(seeds[0]),
        trim_rng=np.random.default_rng(seeds[1]),
        fission_rng=np.random.default_rng(seeds[2]),
        daughter_rng=np.random.default_rng(seeds[3]),
        horizon=8,
    )
    trace = simulate_branch_trace(
        restored=restored,
        beta=beta,
        definition=definition,
        target_centroid=target,
        event_rng=np.random.default_rng(seeds[0]),
        trim_rng=np.random.default_rng(seeds[1]),
        fission_rng=np.random.default_rng(seeds[2]),
        daughter_rng=np.random.default_rng(seeds[3]),
        horizon=8,
    )
    assert trace.compact == compact
    assert len(trace.observations) == compact.selected_observations_generated
    assert [row.offset for row in trace.observations] == list(
        range(1, len(trace.observations) + 1)
    )


def test_trace_exact_replay() -> None:
    rng = np.random.default_rng(3506)
    state = rng.poisson(1.1, size=100).astype(np.int64)
    state[0] += 1
    beta = np.exp(rng.normal(-3.0, 0.7, size=(100, 100)))
    target = rng.random(100)
    target /= target.sum()
    restored = RestoredState(tuple(map(int, state)), "post_fission", 4, 4, 3, 24)
    definition = SimulationDefinition(
        daughter_rule="RANDOM_NONEMPTY",
        overshoot_rule="TRIM_NEW_ENTRANTS_TO_NMAX",
        exposure=ExposureDefinition(family="FIXED_COMMON_EXPOSURE", h=0.56),
    )

    def run():
        return simulate_branch_trace(
            restored=restored,
            beta=beta,
            definition=definition,
            target_centroid=target,
            event_rng=np.random.default_rng(3507),
            trim_rng=np.random.default_rng(3508),
            fission_rng=np.random.default_rng(3509),
            daughter_rng=np.random.default_rng(3510),
            horizon=8,
        )

    first = run()
    second = run()
    assert first.compact == second.compact
    assert len(first.observations) == len(second.observations)
    for left, right in zip(first.observations, second.observations, strict=True):
        for field in left.__dataclass_fields__:
            left_value = getattr(left, field)
            right_value = getattr(right, field)
            if isinstance(left_value, float) and np.isnan(left_value):
                assert isinstance(right_value, float) and np.isnan(right_value)
            else:
                assert left_value == right_value
