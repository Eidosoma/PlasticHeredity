"""Pure mechanics for the exploratory GARD dynamic-regime experiment.

This module deliberately has no filesystem access.  The stochastic twin
coupling is implemented here rather than inferred from independent futures:
identical states receive identical events, while unequal rates receive a
shared Poisson component plus independent residuals.  Molecule-token random
priorities couple capacity trimming and fission without changing either
simulator contract's marginal rule.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
from math import log
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from plastic_heredity.config import GardConfig, SimulationContract
from plastic_heredity.processes import evaluate_process
from plastic_heredity.simulator import (
    FissionRecord,
    SimulationError,
    advance_fission,
    cosine_similarity,
    generate_initial_composition,
)


IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]

INHERITANCE_THRESHOLD = 0.90
DISTINCTNESS_THRESHOLD = 0.85
DAMAGE_FLOOR = 1.0 / 160.0


@dataclass(frozen=True)
class TwinReadout:
    damage_tv: FloatArray
    damage_cosine: FloatArray
    exponent_1_8: float
    cosine_exponent_1_8: float
    coalesced_by_8: bool
    coalesced_by_32: bool
    survival_f32: bool
    saturated_f32: bool
    integrated_damage: float
    maximum_damage: float
    identical_path: bool
    left_digest: str
    right_digest: str

    def scalars(self) -> dict[str, float | bool | str]:
        result = asdict(self)
        result.pop("damage_tv")
        result.pop("damage_cosine")
        return result


@dataclass(frozen=True)
class PHReadout:
    f12: bool
    strict8: bool
    break_event: bool
    break_time: int
    inherited_boundaries: int
    terminal_within8_h: float
    terminal_digest: str

    def to_dict(self) -> dict[str, float | bool | int | str]:
        return asdict(self)


def scaled_config(base: GardConfig, leave_multiplier: float) -> GardConfig:
    if leave_multiplier <= 0.0:
        raise ValueError("leave multiplier must be positive")
    return replace(base, k_leave=base.k_leave * float(leave_multiplier))


def scaled_beta(beta: NDArray, multiplier: float) -> FloatArray:
    if multiplier <= 0.0:
        raise ValueError("beta multiplier must be positive")
    return np.asarray(beta, dtype=np.float64) * float(multiplier)


def composition_probabilities(values: NDArray) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    mass = float(array.sum())
    if array.ndim != 1 or mass <= 0.0:
        raise ValueError("composition must be a non-empty positive-mass vector")
    return array / mass


def total_variation(left: NDArray, right: NDArray) -> float:
    return float(
        0.5
        * np.abs(composition_probabilities(left) - composition_probabilities(right)).sum()
    )


def one_molecule_substitution(state: NDArray, rng: np.random.Generator) -> IntArray:
    """Make an outcome-blind, mass-preserving one-token perturbation."""

    values = np.asarray(state, dtype=np.int64).copy()
    if values.ndim != 1 or int(values.sum()) <= 0 or np.any(values < 0):
        raise ValueError("invalid source composition")
    token = int(rng.integers(0, int(values.sum())))
    cumulative = np.cumsum(values)
    source = int(np.searchsorted(cumulative, token, side="right"))
    destination_draw = int(rng.integers(0, values.size - 1))
    destination = destination_draw if destination_draw < source else destination_draw + 1
    values[source] -= 1
    values[destination] += 1
    return values


def _coupled_poisson(
    left_rate: NDArray, right_rate: NDArray, rng: np.random.Generator
) -> tuple[IntArray, IntArray]:
    left = np.maximum(np.asarray(left_rate, dtype=np.float64), 0.0)
    right = np.maximum(np.asarray(right_rate, dtype=np.float64), 0.0)
    if left.shape != right.shape:
        raise ValueError("coupled Poisson shapes differ")
    common_rate = np.minimum(left, right)
    common = np.asarray(rng.poisson(common_rate), dtype=np.int64)
    left_extra = np.asarray(rng.poisson(left - common_rate), dtype=np.int64)
    right_extra = np.asarray(rng.poisson(right - common_rate), dtype=np.int64)
    return common + left_extra, common + right_extra


def _token_keys(
    left: IntArray, right: IntArray, rng: np.random.Generator
) -> tuple[list[tuple[float, int]], list[tuple[float, int]]]:
    left_tokens: list[tuple[float, int]] = []
    right_tokens: list[tuple[float, int]] = []
    for molecule, (left_count, right_count) in enumerate(zip(left, right, strict=True)):
        common = min(int(left_count), int(right_count))
        if common:
            keys = rng.random(common)
            left_tokens.extend((float(key), molecule) for key in keys)
            right_tokens.extend((float(key), molecule) for key in keys)
        left_extra = int(left_count) - common
        right_extra = int(right_count) - common
        if left_extra:
            left_tokens.extend((float(key), molecule) for key in rng.random(left_extra))
        if right_extra:
            right_tokens.extend((float(key), molecule) for key in rng.random(right_extra))
    return left_tokens, right_tokens


def _counts_from_tokens(
    tokens: list[tuple[float, int]], sample_size: int, width: int
) -> IntArray:
    if not 0 <= sample_size <= len(tokens):
        raise SimulationError("invalid coupled token sample size")
    chosen = sorted(tokens, key=lambda item: item[0])[:sample_size]
    result = np.zeros(width, dtype=np.int64)
    for _, molecule in chosen:
        result[molecule] += 1
    return result


def _coupled_sample(
    left: IntArray,
    right: IntArray,
    left_size: int,
    right_size: int,
    rng: np.random.Generator,
) -> tuple[IntArray, IntArray]:
    left_tokens, right_tokens = _token_keys(left, right, rng)
    return (
        _counts_from_tokens(left_tokens, left_size, left.size),
        _counts_from_tokens(right_tokens, right_size, right.size),
    )


def _coupled_binomial_half(
    left: IntArray, right: IntArray, rng: np.random.Generator
) -> tuple[IntArray, IntArray]:
    selected_left = np.zeros_like(left)
    selected_right = np.zeros_like(right)
    for molecule, (left_count, right_count) in enumerate(zip(left, right, strict=True)):
        common = min(int(left_count), int(right_count))
        if common:
            values = rng.random(common) < 0.5
            number = int(values.sum())
            selected_left[molecule] += number
            selected_right[molecule] += number
        if int(left_count) > common:
            selected_left[molecule] += int(
                (rng.random(int(left_count) - common) < 0.5).sum()
            )
        if int(right_count) > common:
            selected_right[molecule] += int(
                (rng.random(int(right_count) - common) < 0.5).sum()
            )
    return selected_left, selected_right


def _coupled_trim(
    left: IntArray,
    right: IntArray,
    left_target: int,
    right_target: int,
    rng: np.random.Generator,
) -> tuple[IntArray, IntArray]:
    left_excess = max(0, int(left.sum()) - left_target)
    right_excess = max(0, int(right.sum()) - right_target)
    removed_left, removed_right = _coupled_sample(
        left, right, left_excess, right_excess, rng
    )
    return left - removed_left, right - removed_right


def _coupled_growth(
    left: IntArray,
    right: IntArray,
    beta: FloatArray,
    config: GardConfig,
    contract: SimulationContract,
    rng: np.random.Generator,
) -> tuple[IntArray, IntArray, int, int]:
    states = [np.asarray(left, dtype=np.int64).copy(), np.asarray(right, dtype=np.int64).copy()]
    parents: list[IntArray | None] = [None, None]
    steps = [0, 0]
    rho = 1.0 / config.n_types
    for step in range(1, config.max_growth_steps + 1):
        active = [parents[index] is None for index in range(2)]
        for index in range(2):
            if active[index] and int(states[index].sum()) <= 0:
                raise SimulationError("assembly became extinct")
        if all(not value for value in active):
            return parents[0], parents[1], steps[0], steps[1]  # type: ignore[return-value]

        join_rates: list[FloatArray] = []
        leave_rates: list[FloatArray] = []
        for index in range(2):
            current = states[index]
            if not active[index]:
                join_rates.append(np.zeros(config.n_types, dtype=np.float64))
                leave_rates.append(np.zeros(config.n_types, dtype=np.float64))
                continue
            mass = int(current.sum())
            if mass >= config.n_max:
                parents[index] = current.copy()
                steps[index] = step - 1
                active[index] = False
                join_rates.append(np.zeros(config.n_types, dtype=np.float64))
                leave_rates.append(np.zeros(config.n_types, dtype=np.float64))
                continue
            boost = 1.0 + (beta @ current) / mass
            join_rates.append(config.k_join * rho * mass * boost * contract.poisson_exposure)
            leave_rates.append(config.k_leave * current * boost * contract.poisson_exposure)

        joins = _coupled_poisson(join_rates[0], join_rates[1], rng)
        leaves = _coupled_poisson(leave_rates[0], leave_rates[1], rng)
        proposed: list[IntArray] = []
        for index in range(2):
            if not active[index]:
                proposed.append(states[index])
                continue
            actual_leaves = np.minimum(leaves[index], states[index])
            proposed.append(states[index] - actual_leaves + joins[index])

        if contract.overshoot_rule == "trim_whole_assembly":
            proposed[0], proposed[1] = _coupled_trim(
                proposed[0], proposed[1], config.n_max, config.n_max, rng
            )
        elif contract.overshoot_rule == "admit_joiners_to_capacity":
            # Reconstruct survivor/joins and uniformly admit only available join tokens.
            accepted: list[IntArray] = []
            survivor_values: list[IntArray] = []
            join_values: list[IntArray] = []
            for index in range(2):
                if not active[index]:
                    survivor_values.append(states[index])
                    join_values.append(np.zeros_like(states[index]))
                    continue
                actual_leaves = np.minimum(leaves[index], states[index])
                survivor_values.append(states[index] - actual_leaves)
                join_values.append(joins[index])
            capacities = [
                max(0, config.n_max - int(survivor_values[index].sum()))
                for index in range(2)
            ]
            sample_sizes = [min(int(join_values[index].sum()), capacities[index]) for index in range(2)]
            admitted = _coupled_sample(
                join_values[0], join_values[1], sample_sizes[0], sample_sizes[1], rng
            )
            for index in range(2):
                accepted.append(survivor_values[index] + admitted[index])
            proposed = accepted
        else:
            raise SimulationError(f"unknown overshoot rule: {contract.overshoot_rule}")

        for index in range(2):
            if not active[index]:
                continue
            states[index] = proposed[index]
            if int(states[index].sum()) >= config.n_max:
                parents[index] = states[index].copy()
                steps[index] = step
    raise SimulationError("coupled growth exceeded max_growth_steps")


def _coupled_fission(
    left: IntArray,
    right: IntArray,
    config: GardConfig,
    contract: SimulationContract,
    rng: np.random.Generator,
) -> tuple[IntArray, IntArray]:
    if contract.fission_rule == "fixed_size":
        first_left, first_right = _coupled_sample(
            left, right, config.n_min, config.n_min, rng
        )
    elif contract.fission_rule == "binomial":
        first_left, first_right = _coupled_binomial_half(left, right, rng)
    else:
        raise SimulationError(f"unknown fission rule: {contract.fission_rule}")
    daughters = (
        (first_left, first_right)
        if contract.daughter_rule == "first"
        else (left - first_left, right - first_right)
    )
    if int(daughters[0].sum()) == 0 or int(daughters[1].sum()) == 0:
        raise SimulationError("selected coupled daughter was empty")
    return daughters


def advance_coupled_fission(
    left: NDArray,
    right: NDArray,
    beta: NDArray,
    config: GardConfig,
    contract: SimulationContract,
    rng: np.random.Generator,
) -> tuple[FissionRecord, FissionRecord]:
    parent_left, parent_right, steps_left, steps_right = _coupled_growth(
        np.asarray(left, dtype=np.int64),
        np.asarray(right, dtype=np.int64),
        np.asarray(beta, dtype=np.float64),
        config,
        contract,
        rng,
    )
    daughter_left, daughter_right = _coupled_fission(
        parent_left, parent_right, config, contract, rng
    )
    return (
        FissionRecord(parent_left, daughter_left, cosine_similarity(parent_left, daughter_left), steps_left),
        FissionRecord(parent_right, daughter_right, cosine_similarity(parent_right, daughter_right), steps_right),
    )


def _trajectory_digest(states: Sequence[NDArray]) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(states, dtype="<i8").tobytes(order="C"))
    digest.update(len(states).to_bytes(4, "little", signed=False))
    return digest.hexdigest()


def simulate_twins(
    left: NDArray,
    right: NDArray,
    beta: NDArray,
    config: GardConfig,
    contract: SimulationContract,
    *,
    seed: int,
    horizon: int = 32,
) -> TwinReadout:
    current_left = np.asarray(left, dtype=np.int64).copy()
    current_right = np.asarray(right, dtype=np.int64).copy()
    tv = [total_variation(current_left, current_right)]
    cosine = [1.0 - cosine_similarity(current_left, current_right)]
    states_left: list[IntArray] = [current_left.copy()]
    states_right: list[IntArray] = [current_right.copy()]
    rng = np.random.default_rng(seed)
    for _ in range(horizon):
        record_left, record_right = advance_coupled_fission(
            current_left, current_right, beta, config, contract, rng
        )
        current_left = record_left.daughter
        current_right = record_right.daughter
        states_left.append(current_left.copy())
        states_right.append(current_right.copy())
        tv.append(total_variation(current_left, current_right))
        cosine.append(1.0 - cosine_similarity(current_left, current_right))
    tv_values = np.asarray(tv, dtype=np.float64)
    cosine_values = np.asarray(cosine, dtype=np.float64)
    endpoint = min(8, horizon)
    exponent = log((tv_values[endpoint] + DAMAGE_FLOOR) / (tv_values[0] + DAMAGE_FLOOR)) / endpoint
    cosine_exponent = log(
        (cosine_values[endpoint] + DAMAGE_FLOOR)
        / (cosine_values[0] + DAMAGE_FLOOR)
    ) / endpoint
    identical = bool(np.array_equal(states_left, states_right))
    return TwinReadout(
        damage_tv=tv_values,
        damage_cosine=cosine_values,
        exponent_1_8=float(exponent),
        cosine_exponent_1_8=float(cosine_exponent),
        coalesced_by_8=bool(np.any(tv_values[: endpoint + 1] == 0.0)),
        coalesced_by_32=bool(np.any(tv_values == 0.0)),
        survival_f32=bool(tv_values[-1] > 0.0),
        saturated_f32=bool(tv_values[-1] >= 0.5),
        integrated_damage=float(np.mean(tv_values[1:])),
        maximum_damage=float(np.max(tv_values)),
        identical_path=identical,
        left_digest=_trajectory_digest(states_left),
        right_digest=_trajectory_digest(states_right),
    )


def burn_in_state(
    config: GardConfig,
    contract: SimulationContract,
    beta: NDArray,
    *,
    seed: int,
    generations: int = 64,
) -> IntArray:
    rng = np.random.default_rng(seed)
    current = generate_initial_composition(config, rng)
    for _ in range(generations):
        current = advance_fission(current, beta, config, contract, rng).daughter
    return current


def _all_pairwise_above(states: Sequence[NDArray], threshold: float) -> bool:
    return all(
        cosine_similarity(states[left], states[right]) > threshold
        for left in range(len(states))
        for right in range(left + 1, len(states))
    )


def score_strict8(records: Sequence[FissionRecord]) -> bool:
    inherited = [record.h > INHERITANCE_THRESHOLD for record in records]
    breaks = [index for index, value in enumerate(inherited) if not value]
    if not breaks:
        return False
    first_break = breaks[0]
    anchor = records[first_break].parent
    daughters = [record.daughter for record in records]
    for start in range(first_break + 1, len(records) - 7):
        block = daughters[start : start + 8]
        if _all_pairwise_above(block, INHERITANCE_THRESHOLD) and all(
            cosine_similarity(anchor, daughter) <= DISTINCTNESS_THRESHOLD
            for daughter in block
        ):
            return True
    return False


def simulate_ph_lineage(
    beta: NDArray,
    config: GardConfig,
    contract: SimulationContract,
    *,
    seed: int,
    horizon: int = 32,
) -> tuple[PHReadout, IntArray]:
    rng = np.random.default_rng(seed)
    current = generate_initial_composition(config, rng)
    records: list[FissionRecord] = []
    daughters: list[IntArray] = []
    for _ in range(horizon):
        record = advance_fission(current, beta, config, contract, rng)
        records.append(record)
        current = record.daughter
        daughters.append(current.copy())
    process = evaluate_process(records[:12], INHERITANCE_THRESHOLD)
    break_locations = [index for index, record in enumerate(records) if record.h <= INHERITANCE_THRESHOLD]
    block = daughters[-8:]
    within = np.mean(
        [
            cosine_similarity(block[left], block[right])
            for left in range(len(block))
            for right in range(left + 1, len(block))
        ]
    )
    readout = PHReadout(
        f12=bool(process.joint_break_run3),
        strict8=score_strict8(records),
        break_event=bool(break_locations),
        break_time=int(break_locations[0] + 1) if break_locations else -1,
        inherited_boundaries=sum(record.h > INHERITANCE_THRESHOLD for record in records),
        terminal_within8_h=float(within),
        terminal_digest=_trajectory_digest([current]),
    )
    return readout, current


def mean_field_flow(
    composition: NDArray,
    beta: NDArray,
    k_join: float,
    k_leave: float,
) -> FloatArray:
    x = composition_probabilities(composition)
    matrix = np.asarray(beta, dtype=np.float64)
    rho = 1.0 / x.size
    boost = 1.0 + matrix @ x
    raw = (k_join * rho - k_leave * x) * boost
    return raw - x * float(raw.sum())


def flow_jacobian(
    composition: NDArray,
    beta: NDArray,
    k_join: float,
    k_leave: float,
) -> FloatArray:
    x = composition_probabilities(composition)
    matrix = np.asarray(beta, dtype=np.float64)
    basal = k_join / x.size
    boost = 1.0 + matrix @ x
    raw = (basal - k_leave * x) * boost
    derivative = (basal - k_leave * x)[:, None] * matrix
    derivative[np.diag_indices_from(derivative)] -= k_leave * boost
    total_derivative = derivative.sum(axis=0)
    return derivative - np.outer(x, total_derivative) - np.eye(x.size) * float(raw.sum())


def tangent_stability_margin(
    composition: NDArray,
    beta: NDArray,
    k_join: float,
    k_leave: float,
) -> float:
    x = composition_probabilities(composition)
    width = x.size
    # Orthonormal basis for the sum-zero simplex tangent space.
    raw_basis = np.vstack((np.eye(width - 1), -np.ones(width - 1)))
    basis, _ = np.linalg.qr(raw_basis)
    reduced = basis.T @ flow_jacobian(x, beta, k_join, k_leave) @ basis
    return -float(np.max(np.linalg.eigvals(reduced).real))


def relax_mean_field(
    start: NDArray,
    beta: NDArray,
    k_join: float,
    k_leave: float,
    *,
    step_size: float = 0.5,
    tolerance: float = 1e-10,
    maximum_iterations: int = 10_000,
) -> tuple[FloatArray, int, float]:
    del step_size  # Kept in the interface for a stable preregistered call shape.
    current = composition_probabilities(start)[None, :]
    matrix = np.asarray(beta, dtype=np.float64)
    basal = k_join / current.shape[1]
    residual = float("inf")
    for iteration in range(1, maximum_iterations + 1):
        boost = 1.0 + current @ matrix.T
        low = np.zeros(1, dtype=np.float64)
        high = 2.0 * basal * boost.sum(axis=1) + k_leave * boost.max(axis=1)
        for _ in range(80):
            middle = 0.5 * (low + high)
            total = (basal * boost / (middle[:, None] + k_leave * boost)).sum(axis=1)
            low = np.where(total > 1.0, middle, low)
            high = np.where(total > 1.0, high, middle)
        proposed = basal * boost / (high[:, None] + k_leave * boost)
        proposed /= proposed.sum(axis=1, keepdims=True)
        update = float(np.max(np.abs(proposed - current)))
        current = proposed
        residual = float(
            np.max(np.abs(mean_field_flow(current[0], matrix, k_join, k_leave)))
        )
        if update < tolerance and residual < tolerance:
            return current[0], iteration, residual
    return current[0], maximum_iterations, residual
