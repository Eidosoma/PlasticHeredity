"""Pure mechanics for prospective strict-eight matching and switch-lock forks."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from math import cos, pi
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from plastic_heredity.config import GardConfig, SimulationContract
from plastic_heredity.regime_confirmation import evaluate_regime
from plastic_heredity.simulator import (
    FissionRecord,
    SimulationError,
    advance_fission,
    cosine_similarity,
    generate_initial_composition,
)
from reviewer_lineage_identity_response.followup_carrier_cleanroom.carrier_core import (
    advance_carrier_fission,
    influence_mask,
    writer_signal,
)


IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]
INHERITANCE_THRESHOLD = 0.90
DISTINCTNESS_THRESHOLD = 0.85
F12_HORIZON = 12
STRICT_WINDOW = 8
CHECKPOINTS = (8, 16, 32)
WAVE_PERIOD = 4
WAVE_COUPLING = 2.0
WAVE_COORDINATES = 32


@dataclass(frozen=True)
class ProspectiveEvent:
    f12: bool
    strict_extension: bool
    f12_only: bool
    any_strict8: bool
    first_break: int
    run3_start: int
    run3_end: int
    b_state: IntArray
    match_features: FloatArray


@dataclass(frozen=True)
class LockArm:
    name: str
    quench: bool = False
    field: str = "none"
    phase: float = 0.0
    release_after: int = 0


@dataclass(frozen=True)
class Trajectory:
    states: IntArray
    boundary_h: FloatArray
    observed: int
    digest: str


ARMS = (
    LockArm("control"),
    LockArm("quench", quench=True),
    LockArm("wave", field="wave"),
    LockArm("quench_wave", quench=True, field="wave"),
    LockArm("quench_static", quench=True, field="static"),
    LockArm("quench_shuffled", quench=True, field="shuffled"),
    LockArm("quench_phase_pi", quench=True, field="wave", phase=pi),
    LockArm("pulse_release", quench=True, field="wave", release_after=8),
)


def _minimum_pairwise(states: Sequence[NDArray]) -> float:
    if len(states) < 2:
        return np.nan
    return min(
        cosine_similarity(states[left], states[right])
        for left in range(len(states))
        for right in range(left + 1, len(states))
    )


def _first_true_run(values: NDArray, length: int, start: int) -> int:
    array = np.asarray(values, dtype=bool)
    for index in range(start, array.size - length + 1):
        if bool(array[index : index + length].all()):
            return index
    return -1


def score_prospective_event(records: Sequence[FissionRecord], n_types: int) -> ProspectiveEvent:
    """Capture strict and F12-only B at the same local event age (run three)."""

    values = list(records)
    empty = np.zeros(n_types, dtype=np.int64)
    no_features = np.full(5, np.nan, dtype=np.float64)
    if not values:
        return ProspectiveEvent(False, False, False, False, -1, -1, -1, empty, no_features)
    inherited = np.asarray([record.h > INHERITANCE_THRESHOLD for record in values], dtype=bool)
    first = inherited[:F12_HORIZON]
    breaks = np.flatnonzero(~first)
    first_break = int(breaks[0]) if breaks.size else -1
    f12_run3_start = _first_true_run(first, 3, first_break + 1) if first_break >= 0 else -1
    f12_run3_end = f12_run3_start + 2 if f12_run3_start >= 0 else -1
    f12 = bool(f12_run3_end >= 0 and f12_run3_end < F12_HORIZON)
    regime = evaluate_regime(values)
    any_strict = bool(regime.primary_all8)
    if any_strict:
        run3_start = int(regime.primary_all8_onset)
        first_break_for_features = int(regime.first_break_index)
    elif f12:
        run3_start = f12_run3_start
        first_break_for_features = first_break
    else:
        return ProspectiveEvent(False, False, False, False, first_break, -1, -1, empty, no_features)
    run3_end = run3_start + 2
    anchor = values[first_break_for_features].parent
    first_three = [record.daughter for record in values[run3_start : run3_start + 3]]
    b_state = values[run3_end].daughter.copy()
    features = np.asarray(
        [
            first_break_for_features,
            run3_start,
            _minimum_pairwise(first_three),
            max(cosine_similarity(anchor, state) for state in first_three),
            int(b_state.sum()),
        ],
        dtype=np.float64,
    )
    return ProspectiveEvent(
        f12=True,
        strict_extension=any_strict,
        f12_only=bool(f12 and not any_strict),
        any_strict8=any_strict,
        first_break=first_break_for_features,
        run3_start=run3_start,
        run3_end=run3_end,
        b_state=b_state,
        match_features=features,
    )


def simulate_donor_lineage(
    beta: NDArray,
    config: GardConfig,
    contract: SimulationContract,
    *,
    seed: int,
    horizon: int = 32,
) -> tuple[ProspectiveEvent, int]:
    rng = np.random.default_rng(seed)
    current = generate_initial_composition(config, rng)
    records: list[FissionRecord] = []
    for _ in range(horizon):
        try:
            record = advance_fission(current, beta, config, contract, rng)
        except SimulationError:
            break
        records.append(record)
        current = record.daughter
    return score_prospective_event(records, config.n_types), len(records)


def shuffled_writer_signal(signal: NDArray, mask: NDArray, seed: int) -> FloatArray:
    values = np.asarray(signal, dtype=np.float64).copy()
    active = np.flatnonzero(np.asarray(mask, dtype=bool))
    if active.size:
        values[active] = values[active][np.random.default_rng(seed).permutation(active.size)]
    values[~np.asarray(mask, dtype=bool)] = 0.0
    return values


def wave_amplitude(generation: int, phase: float = 0.0) -> float:
    if generation < 1:
        raise ValueError("generation is one-indexed")
    return float(0.5 * (1.0 + cos(2.0 * pi * (generation - 1) / WAVE_PERIOD + phase)))


def simulate_lock_future(
    start: NDArray,
    target: NDArray,
    beta: NDArray,
    anchor_config: GardConfig,
    lock_config: GardConfig,
    contract: SimulationContract,
    arm: LockArm,
    *,
    dynamics_seed: int,
    shuffle_seed: int,
    horizon: int = 32,
) -> Trajectory:
    """Run one fork; control follows the frozen simulator bit for bit."""

    dynamics_rng = np.random.default_rng(dynamics_seed)
    current = np.asarray(start, dtype=np.int64).copy()
    states = np.zeros((horizon, anchor_config.n_types), dtype=np.int64)
    boundary = np.full(horizon, np.nan, dtype=np.float64)
    mask = influence_mask(beta, WAVE_COORDINATES)
    correct = writer_signal(target, mask)
    shuffled = shuffled_writer_signal(correct, mask, shuffle_seed)
    observed = 0
    for generation in range(1, horizon + 1):
        supported = arm.release_after == 0 or generation <= arm.release_after
        config = lock_config if arm.quench and supported else anchor_config
        field = arm.field if supported else "none"
        if field == "none":
            carrier = np.zeros(anchor_config.n_types, dtype=np.float64)
        elif field == "static":
            carrier = correct
        elif field == "shuffled":
            carrier = wave_amplitude(generation, arm.phase) * shuffled
        elif field == "wave":
            carrier = wave_amplitude(generation, arm.phase) * correct
        else:
            raise ValueError(f"unknown field: {field}")
        try:
            if field == "none" or not np.any(carrier):
                record = advance_fission(current, beta, config, contract, dynamics_rng)
            else:
                record = advance_carrier_fission(
                    current,
                    beta,
                    config,
                    contract,
                    dynamics_rng,
                    carrier,
                    WAVE_COUPLING,
                    reader=True,
                )
        except SimulationError:
            break
        states[generation - 1] = record.daughter
        boundary[generation - 1] = record.h
        current = record.daughter
        observed += 1
    digest = hashlib.sha256()
    digest.update(np.asarray(states[:observed], dtype="<i8").tobytes(order="C"))
    digest.update(int(observed).to_bytes(4, "little", signed=False))
    return Trajectory(states, boundary, observed, digest.hexdigest())


def static_similarity(left: Trajectory, right: Trajectory, checkpoint: int) -> float:
    if left.observed < checkpoint or right.observed < checkpoint:
        return np.nan
    return cosine_similarity(left.states[checkpoint - 1], right.states[checkpoint - 1])


def phase_aligned_similarity(
    left: Trajectory,
    right: Trajectory,
    checkpoint: int,
    *,
    window: int = STRICT_WINDOW,
    period: int = WAVE_PERIOD,
) -> float:
    if checkpoint < window or left.observed < checkpoint or right.observed < checkpoint:
        return np.nan
    a = left.states[checkpoint - window : checkpoint]
    b = right.states[checkpoint - window : checkpoint]
    scores = []
    for lag in range(period):
        shifted = np.roll(b, lag, axis=0)
        scores.append(float(np.mean([cosine_similarity(x, y) for x, y in zip(a, shifted, strict=True)])))
    return max(scores)


def terminal_target_capture(trajectory: Trajectory, target: NDArray, checkpoint: int) -> float:
    if checkpoint < STRICT_WINDOW or trajectory.observed < checkpoint:
        return np.nan
    block = trajectory.states[checkpoint - STRICT_WINDOW : checkpoint]
    return float(
        all(cosine_similarity(state, target) > INHERITANCE_THRESHOLD for state in block)
        and _minimum_pairwise(list(block)) > INHERITANCE_THRESHOLD
    )


def target_occupancy(trajectory: Trajectory, target: NDArray) -> float:
    if not trajectory.observed:
        return np.nan
    return float(np.mean([cosine_similarity(state, target) > INHERITANCE_THRESHOLD for state in trajectory.states[: trajectory.observed]]))


def target_maximum_residence(trajectory: Trajectory, target: NDArray) -> int:
    longest = 0
    current = 0
    for state in trajectory.states[: trajectory.observed]:
        if cosine_similarity(state, target) > INHERITANCE_THRESHOLD:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def fork_pair_metrics(
    a0: Trajectory,
    a1: Trajectory,
    b0: Trajectory,
    b1: Trajectory,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    same_static = []
    cross_static = []
    same_phase = []
    cross_phase = []
    for checkpoint in CHECKPOINTS:
        same_static.append(np.nanmean([static_similarity(a0, a1, checkpoint), static_similarity(b0, b1, checkpoint)]))
        cross_static.append(np.nanmean([static_similarity(a0, b0, checkpoint), static_similarity(a1, b1, checkpoint)]))
        same_phase.append(np.nanmean([phase_aligned_similarity(a0, a1, checkpoint), phase_aligned_similarity(b0, b1, checkpoint)]))
        cross_phase.append(np.nanmean([phase_aligned_similarity(a0, b0, checkpoint), phase_aligned_similarity(a1, b1, checkpoint)]))
    return tuple(np.asarray(values, dtype=np.float64) for values in (same_static, cross_static, same_phase, cross_phase))
