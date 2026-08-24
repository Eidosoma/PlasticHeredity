"""Pure mechanisms and readouts for the clean-room GARD carrier study.

This module has no filesystem access.  It depends only on the frozen local GARD
reconstruction; in particular, it never imports anything under ``NewIdeas``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from math import sqrt
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from plastic_heredity.config import GardConfig, SimulationContract
from plastic_heredity.simulator import (
    FissionRecord,
    SimulationError,
    _fission,
    _sample_without_replacement,
    _trim_whole_assembly,
    advance_fission,
    cosine_similarity,
)


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]

INHERITANCE_THRESHOLD = 0.90
DEPARTURE_THRESHOLD = 0.85
CAPTURE_WINDOW = 8


@dataclass(frozen=True)
class CarrierSetting:
    """A registered carrier engineering setting."""

    k: int
    half_life: int
    coupling: float
    copy_mode: str

    @property
    def setting_id(self) -> str:
        coupling = str(self.coupling).replace(".", "p")
        return f"k{self.k:03d}_l{self.half_life:02d}_u{coupling}_{self.copy_mode}"

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


@dataclass(frozen=True)
class ArmPolicy:
    """Carrier interventions applied to one stochastic future."""

    name: str
    initial: str = "correct"
    reader: bool = True
    renewal: bool = True
    no_carrier: bool = False
    erase_after_generation: int = 0
    rescue_generation: int = 0


@dataclass(frozen=True)
class FutureReadout:
    observed: int
    completed: bool
    extinct: bool
    first_arrival: int
    arrival_f4: bool
    arrival_f8: bool
    arrival_f16: bool
    capture_any_f16: bool
    capture_any_f32: bool
    capture_any_f64: bool
    terminal8_f16: int
    terminal8_f32: int
    terminal8_f64: int
    occupancy: float
    maximum_residence: int
    departed: bool
    reentered: bool
    final_target_h: float
    final_other_h: float
    origin_correct: int
    carrier_target_h: float
    carrier_other_h: float
    carrier_origin_correct: int
    maximum_boundary_h_error: float
    state_digest: str

    def to_dict(self) -> dict[str, int | float | bool | str]:
        return asdict(self)


def raw_cosine(left: NDArray, right: NDArray) -> float:
    """Cosine for signed carrier vectors (unlike compositional cosine)."""

    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0:
        return 0.0
    return float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0))


def writer_signal(composition: NDArray, mask: NDArray | None = None) -> FloatArray:
    """Map an adult composition to a bounded centered molecule-indexed signal."""

    values = np.asarray(composition, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("composition must be a non-empty vector")
    mass = float(values.sum())
    if mass <= 0.0:
        return np.zeros(values.size, dtype=np.float64)
    signal = values / mass - (1.0 / values.size)
    scale = float(np.max(np.abs(signal)))
    if scale > 0.0:
        signal = signal / scale
    if mask is not None:
        active = np.asarray(mask, dtype=bool)
        if active.shape != signal.shape:
            raise ValueError("mask and composition widths differ")
        signal = np.where(active, signal, 0.0)
    return np.clip(signal, -1.0, 1.0).astype(np.float64, copy=False)


def influence_mask(beta: NDArray, k: int) -> BoolArray:
    """Beta-only mask using total incoming plus outgoing catalytic influence."""

    matrix = np.asarray(beta, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("beta must be square")
    if not 1 <= k <= matrix.shape[0]:
        raise ValueError("k outside beta width")
    influence = matrix.sum(axis=0) + matrix.sum(axis=1)
    # lexsort makes the molecule index the deterministic descending-score tie break.
    order = np.lexsort((np.arange(matrix.shape[0]), -influence))
    mask = np.zeros(matrix.shape[0], dtype=bool)
    mask[order[:k]] = True
    return mask


def random_mask(width: int, k: int, seed: int) -> BoolArray:
    if not 1 <= k <= width:
        raise ValueError("k outside mask width")
    rng = np.random.default_rng(seed)
    mask = np.zeros(width, dtype=bool)
    mask[rng.choice(width, size=k, replace=False)] = True
    return mask


def reservoir_field(carrier: NDArray, coupling: float) -> FloatArray:
    """Return the normalized softmax reservoir field registered in the protocol."""

    values = np.asarray(carrier, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("carrier must be a non-empty vector")
    if coupling < 0.0:
        raise ValueError("coupling must be non-negative")
    logits = coupling * values
    logits = logits - float(np.max(logits))
    weights = np.exp(logits)
    return weights / float(weights.sum())


def update_carrier(
    carrier: NDArray,
    adult_parent: NDArray,
    setting: CarrierSetting,
    mask: NDArray,
    rng: np.random.Generator,
    *,
    renewal: bool,
) -> FloatArray:
    """Write, copy, damage, mask, and bound the register for one fission."""

    previous = np.asarray(carrier, dtype=np.float64)
    active = np.asarray(mask, dtype=bool)
    if previous.shape != active.shape:
        raise ValueError("carrier and mask widths differ")
    decay = 2.0 ** (-1.0 / float(setting.half_life))
    written = writer_signal(adult_parent, active) if renewal else np.zeros_like(previous)
    mixed = decay * previous + (1.0 - decay) * written
    if setting.copy_mode == "ideal":
        copied = mixed
    elif setting.copy_mode == "nominal":
        copied = 0.95 * mixed
        dropout = rng.random(previous.size) < 0.02
        copied = np.where(dropout, 0.0, copied)
        copied = copied + rng.normal(0.0, 0.05, size=previous.size)
    else:
        raise ValueError(f"unknown copy mode: {setting.copy_mode}")
    copied = np.where(active, copied, 0.0)
    return np.clip(copied, -1.0, 1.0).astype(np.float64, copy=False)


def _advance_with_field(
    composition: NDArray,
    beta: NDArray,
    config: GardConfig,
    contract: SimulationContract,
    rng: np.random.Generator,
    rho: NDArray,
) -> FissionRecord:
    """Frozen GARD growth with only its uniform reservoir replaced by ``rho``."""

    current = np.asarray(composition, dtype=np.int64).copy()
    matrix = np.asarray(beta, dtype=np.float64)
    field = np.asarray(rho, dtype=np.float64)
    if field.shape != (config.n_types,) or not np.isclose(field.sum(), 1.0):
        raise ValueError("rho must be a normalized molecule field")
    for step in range(1, config.max_growth_steps + 1):
        mass = int(current.sum())
        if mass <= 0:
            raise SimulationError("assembly became extinct")
        if mass >= config.n_max:
            parent = _trim_whole_assembly(current, config.n_max, rng)
            daughter = _fission(parent, config, contract, rng)
            return FissionRecord(
                parent=parent,
                daughter=daughter,
                h=cosine_similarity(parent, daughter),
                growth_steps=step - 1,
            )
        catalytic_boost = 1.0 + (matrix @ current) / mass
        join_rate = config.k_join * field * mass * catalytic_boost
        leave_rate = config.k_leave * current * catalytic_boost
        exposure = contract.poisson_exposure
        joins = np.asarray(rng.poisson(join_rate * exposure), dtype=np.int64)
        leaves = np.minimum(
            np.asarray(rng.poisson(leave_rate * exposure), dtype=np.int64), current
        )
        survivors = current - leaves
        if contract.overshoot_rule == "admit_joiners_to_capacity":
            capacity = config.n_max - int(survivors.sum())
            if int(joins.sum()) > capacity:
                joins = _sample_without_replacement(joins, capacity, rng)
            current = survivors + joins
        elif contract.overshoot_rule == "trim_whole_assembly":
            current = survivors + joins
            if int(current.sum()) >= config.n_max:
                current = _trim_whole_assembly(current, config.n_max, rng)
        else:
            raise SimulationError(f"unknown overshoot rule: {contract.overshoot_rule}")
        if int(current.sum()) >= config.n_max:
            daughter = _fission(current, config, contract, rng)
            return FissionRecord(
                parent=current,
                daughter=daughter,
                h=cosine_similarity(current, daughter),
                growth_steps=step,
            )
    raise SimulationError(
        f"growth did not reach mass {config.n_max} in {config.max_growth_steps} steps"
    )


def advance_carrier_fission(
    composition: NDArray,
    beta: NDArray,
    config: GardConfig,
    contract: SimulationContract,
    rng: np.random.Generator,
    carrier: NDArray,
    coupling: float,
    *,
    reader: bool,
) -> FissionRecord:
    """Advance once, preserving the exact base path whenever the reader is null."""

    values = np.asarray(carrier, dtype=np.float64)
    if not reader or coupling == 0.0 or not np.any(values):
        return advance_fission(composition, beta, config, contract, rng)
    return _advance_with_field(
        composition,
        beta,
        config,
        contract,
        rng,
        reservoir_field(values, coupling),
    )


def _all_pairwise_above(values: NDArray, threshold: float) -> bool:
    array = np.asarray(values, dtype=np.float64)
    for left in range(array.shape[0]):
        for right in range(left + 1, array.shape[0]):
            if cosine_similarity(array[left], array[right]) <= threshold:
                return False
    return True


def _capture_block(
    daughters: NDArray,
    target: NDArray,
    start: int,
    stop: int,
    threshold: float = INHERITANCE_THRESHOLD,
) -> bool:
    block = np.asarray(daughters[start:stop])
    return bool(
        block.shape[0] == CAPTURE_WINDOW
        and all(cosine_similarity(row, target) > threshold for row in block)
        and _all_pairwise_above(block, threshold)
    )


def _any_capture(daughters: NDArray, target: NDArray, observed: int, horizon: int) -> bool:
    stop = min(observed, horizon)
    return any(
        _capture_block(daughters, target, start, start + CAPTURE_WINDOW)
        for start in range(0, stop - CAPTURE_WINDOW + 1)
    )


def _terminal_capture(daughters: NDArray, target: NDArray, observed: int, horizon: int) -> int:
    if horizon < CAPTURE_WINDOW or observed < horizon or daughters.shape[0] < horizon:
        return -1
    return int(_capture_block(daughters, target, horizon - CAPTURE_WINDOW, horizon))


def _maximum_residence(similarities: NDArray, observed: int) -> int:
    longest = 0
    current = 0
    for value in np.asarray(similarities)[:observed]:
        if value > INHERITANCE_THRESHOLD:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def score_trajectory(
    daughters: NDArray,
    boundary_h: NDArray,
    carrier: NDArray,
    target: NDArray,
    other_target: NDArray | None,
    mask: NDArray,
    *,
    observed: int,
) -> FutureReadout:
    states = np.asarray(daughters, dtype=np.int64)
    similarities = np.asarray(
        [cosine_similarity(states[index], target) for index in range(observed)],
        dtype=np.float64,
    )
    arrivals = np.flatnonzero(similarities > INHERITANCE_THRESHOLD)
    first_arrival = int(arrivals[0] + 1) if arrivals.size else -1
    departed_indices = np.flatnonzero(similarities <= DEPARTURE_THRESHOLD)
    reentered = False
    if departed_indices.size:
        first_departure = int(departed_indices[0])
        reentered = any(
            _capture_block(states, target, start, start + CAPTURE_WINDOW)
            for start in range(first_departure + 1, observed - CAPTURE_WINDOW + 1)
        )
    final_target_h = cosine_similarity(states[observed - 1], target) if observed else 0.0
    final_other_h = 0.0
    origin_correct = -1
    carrier_other_h = 0.0
    carrier_origin_correct = -1
    carrier_target = writer_signal(target, mask)
    carrier_target_h = raw_cosine(carrier, carrier_target)
    if other_target is not None:
        final_other_h = cosine_similarity(states[observed - 1], other_target) if observed else 0.0
        origin_correct = int(final_target_h > final_other_h)
        carrier_other = writer_signal(other_target, mask)
        carrier_other_h = raw_cosine(carrier, carrier_other)
        carrier_origin_correct = int(carrier_target_h > carrier_other_h)
    digest = hashlib.sha256()
    digest.update(np.asarray(states[:observed], dtype="<i8").tobytes(order="C"))
    digest.update(int(observed).to_bytes(4, "little", signed=False))
    return FutureReadout(
        observed=int(observed),
        completed=bool(observed == states.shape[0]),
        extinct=bool(observed < states.shape[0]),
        first_arrival=first_arrival,
        arrival_f4=bool(first_arrival != -1 and first_arrival <= 4),
        arrival_f8=bool(first_arrival != -1 and first_arrival <= 8),
        arrival_f16=bool(first_arrival != -1 and first_arrival <= 16),
        capture_any_f16=_any_capture(states, target, observed, 16),
        capture_any_f32=_any_capture(states, target, observed, 32),
        capture_any_f64=_any_capture(states, target, observed, 64),
        terminal8_f16=_terminal_capture(states, target, observed, 16),
        terminal8_f32=_terminal_capture(states, target, observed, 32),
        terminal8_f64=_terminal_capture(states, target, observed, 64),
        occupancy=float(similarities.mean()) if similarities.size else 0.0,
        maximum_residence=_maximum_residence(similarities, observed),
        departed=bool(departed_indices.size),
        reentered=bool(reentered),
        final_target_h=float(final_target_h),
        final_other_h=float(final_other_h),
        origin_correct=origin_correct,
        carrier_target_h=float(carrier_target_h),
        carrier_other_h=float(carrier_other_h),
        carrier_origin_correct=carrier_origin_correct,
        maximum_boundary_h_error=0.0,
        state_digest=digest.hexdigest(),
    )


def simulate_carrier_future(
    start: NDArray,
    target: NDArray,
    other_target: NDArray | None,
    beta: NDArray,
    config: GardConfig,
    contract: SimulationContract,
    setting: CarrierSetting,
    mask: NDArray,
    policy: ArmPolicy,
    *,
    dynamics_seed: int,
    carrier_seed: int,
    horizon: int,
    initial_override: NDArray | None = None,
) -> tuple[FutureReadout, FloatArray, FloatArray, FloatArray]:
    """Run one future and return registered outcomes plus replay audit traces."""

    active = np.asarray(mask, dtype=bool)
    correct = writer_signal(target, active)
    if initial_override is not None:
        initial = np.asarray(initial_override, dtype=np.float64).copy()
    elif policy.initial == "correct":
        initial = correct.copy()
    elif policy.initial == "zero":
        initial = np.zeros(config.n_types, dtype=np.float64)
    elif policy.initial in {"opposite", "shuffled"}:
        raise ValueError(f"{policy.initial} requires initial_override")
    else:
        raise ValueError(f"unknown initial carrier: {policy.initial}")
    if policy.no_carrier:
        initial = np.zeros(config.n_types, dtype=np.float64)
    carrier = np.where(active, initial, 0.0).astype(np.float64, copy=False)
    dynamics_rng = np.random.default_rng(dynamics_seed)
    carrier_rng = np.random.default_rng(carrier_seed)
    current = np.asarray(start, dtype=np.int64).copy()
    daughters = np.zeros((horizon, config.n_types), dtype=np.int64)
    boundary_h = np.full(horizon, np.nan, dtype=np.float64)
    target_h = np.full(horizon, np.nan, dtype=np.float64)
    carrier_target_h = np.full(horizon, np.nan, dtype=np.float64)
    observed = 0
    for generation in range(1, horizon + 1):
        if policy.erase_after_generation and (
            (not policy.rescue_generation and generation > policy.erase_after_generation)
            or (policy.rescue_generation and generation == policy.rescue_generation)
        ):
            carrier = np.zeros_like(carrier)
        if policy.rescue_generation and generation == policy.rescue_generation:
            carrier = correct.copy()
        try:
            record = advance_carrier_fission(
                current,
                beta,
                config,
                contract,
                dynamics_rng,
                carrier,
                setting.coupling,
                reader=bool(policy.reader and not policy.no_carrier),
            )
        except SimulationError:
            break
        daughters[generation - 1] = record.daughter
        boundary_h[generation - 1] = record.h
        target_h[generation - 1] = cosine_similarity(record.daughter, target)
        current = record.daughter
        observed += 1
        if policy.no_carrier or (
            policy.erase_after_generation
            and not policy.rescue_generation
            and generation >= policy.erase_after_generation
        ):
            carrier = np.zeros_like(carrier)
        else:
            carrier = update_carrier(
                carrier,
                record.parent,
                setting,
                active,
                carrier_rng,
                renewal=policy.renewal,
            )
        carrier_target_h[generation - 1] = raw_cosine(carrier, correct)
    readout = score_trajectory(
        daughters,
        boundary_h,
        carrier,
        target,
        other_target,
        active,
        observed=observed,
    )
    return readout, boundary_h, target_h, carrier_target_h


def choose_low_similarity_permutation(
    state: NDArray,
    *,
    seed: int,
    proposals: int,
) -> tuple[NDArray[np.int16], float]:
    values = np.asarray(state)
    rng = np.random.default_rng(seed)
    best: NDArray[np.int16] | None = None
    best_h = float("inf")
    for _ in range(proposals):
        permutation = np.asarray(rng.permutation(values.size), dtype=np.int16)
        h = cosine_similarity(values, values[permutation])
        if h < best_h:
            best = permutation
            best_h = h
    assert best is not None
    return best, float(best_h)


def paired_bootstrap_ci(
    left: NDArray,
    right: NDArray,
    *,
    seed: int,
    repetitions: int = 10_000,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Equal-rule paired bootstrap interval for a mean difference."""

    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or a.size == 0:
        raise ValueError("paired bootstrap requires equal non-empty vectors")
    values = a - b
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, values.size, size=(repetitions, values.size))
    statistics = values[draws].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return (
        float(values.mean()),
        float(np.quantile(statistics, alpha)),
        float(np.quantile(statistics, 1.0 - alpha)),
    )


def bootstrap_mean_ci(
    values: NDArray,
    *,
    seed: int,
    repetitions: int = 10_000,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("bootstrap requires a non-empty vector")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, array.size, size=(repetitions, array.size))
    statistics = array[draws].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return (
        float(array.mean()),
        float(np.quantile(statistics, alpha)),
        float(np.quantile(statistics, 1.0 - alpha)),
    )


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    """Dependency-free Wilson interval (z=1.95996 at the registered 95%)."""

    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("invalid binomial counts")
    if confidence != 0.95:
        raise ValueError("only the registered 95% Wilson interval is supported")
    z = 1.959963984540054
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    radius = z * sqrt((p * (1.0 - p) + z * z / (4.0 * trials)) / trials) / denominator
    return float(max(0.0, center - radius)), float(min(1.0, center + radius))


def permutation_equivariance(
    beta: NDArray,
    composition: NDArray,
    carrier: NDArray,
    mask: NDArray,
    permutation: Sequence[int],
) -> tuple[FloatArray, IntArray, FloatArray, BoolArray]:
    """Jointly relabel the rule, state, carrier, and mask."""

    order = np.asarray(permutation, dtype=np.int64)
    matrix = np.asarray(beta, dtype=np.float64)
    return (
        matrix[np.ix_(order, order)].copy(),
        np.asarray(composition, dtype=np.int64)[order].copy(),
        np.asarray(carrier, dtype=np.float64)[order].copy(),
        np.asarray(mask, dtype=bool)[order].copy(),
    )
