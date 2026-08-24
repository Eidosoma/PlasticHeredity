from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .config import GardConfig, SimulationContract

IntVector = NDArray[np.int64]
FloatMatrix = NDArray[np.float64]


class SimulationError(RuntimeError):
    pass


@dataclass(frozen=True)
class Snapshot:
    composition: IntVector
    generation: int
    inheritance: tuple[bool, ...]
    boundary_h: tuple[float, ...]
    # Observational clocks captured at the post-fission restoration boundary.
    # Neither field is read by future dynamics; they are retained solely for
    # the prospective history-versus-state ablation.
    previous_growth_steps: int = 0
    cumulative_growth_steps: int = 0


@dataclass(frozen=True)
class FissionRecord:
    parent: IntVector
    daughter: IntVector
    h: float
    growth_steps: int


def cosine_similarity(left: NDArray, right: NDArray) -> float:
    left_f = np.asarray(left, dtype=np.float64)
    right_f = np.asarray(right, dtype=np.float64)
    denominator = float(np.linalg.norm(left_f) * np.linalg.norm(right_f))
    if denominator == 0.0:
        return 0.0
    return float(np.clip(np.dot(left_f, right_f) / denominator, 0.0, 1.0))


def generate_beta(config: GardConfig, rng: np.random.Generator) -> FloatMatrix:
    normal = rng.standard_normal((config.n_types, config.n_types))
    return np.exp(config.beta_log_mean + config.beta_log_sd * normal)


def generate_initial_composition(
    config: GardConfig, rng: np.random.Generator
) -> IntVector:
    composition = np.zeros(config.n_types, dtype=np.int64)
    composition[rng.choice(config.n_types, size=config.n_min, replace=False)] = 1
    return composition


def _sample_without_replacement(
    counts: IntVector, sample_size: int, rng: np.random.Generator
) -> IntVector:
    total = int(counts.sum())
    if not 0 <= sample_size <= total:
        raise SimulationError(f"invalid sample size {sample_size} for mass {total}")
    if sample_size == 0:
        return np.zeros_like(counts)
    return np.asarray(
        rng.multivariate_hypergeometric(counts, sample_size), dtype=np.int64
    )


def _trim_whole_assembly(
    composition: IntVector, target: int, rng: np.random.Generator
) -> IntVector:
    excess = int(composition.sum()) - target
    if excess <= 0:
        return composition
    return composition - _sample_without_replacement(composition, excess, rng)


def _grow_to_fission(
    composition: IntVector,
    beta: FloatMatrix,
    config: GardConfig,
    contract: SimulationContract,
    rng: np.random.Generator,
) -> tuple[IntVector, int]:
    current = np.asarray(composition, dtype=np.int64).copy()
    rho = 1.0 / config.n_types

    for step in range(1, config.max_growth_steps + 1):
        mass = int(current.sum())
        if mass <= 0:
            raise SimulationError("assembly became extinct")
        if mass >= config.n_max:
            return _trim_whole_assembly(current, config.n_max, rng), step - 1

        catalytic_boost = 1.0 + (beta @ current) / mass
        join_rate = config.k_join * rho * mass * catalytic_boost
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
            return current, step

    raise SimulationError(
        f"growth did not reach mass {config.n_max} in {config.max_growth_steps} steps"
    )


def _fission(
    parent: IntVector,
    config: GardConfig,
    contract: SimulationContract,
    rng: np.random.Generator,
) -> IntVector:
    if contract.fission_rule == "fixed_size":
        first = _sample_without_replacement(parent, config.n_min, rng)
    elif contract.fission_rule == "binomial":
        first = np.asarray(rng.binomial(parent, 0.5), dtype=np.int64)
    else:
        raise SimulationError(f"unknown fission rule: {contract.fission_rule}")
    second = parent - first
    daughter = first if contract.daughter_rule == "first" else second
    if int(daughter.sum()) == 0:
        raise SimulationError("selected daughter was empty")
    return daughter


def advance_fission(
    composition: IntVector,
    beta: FloatMatrix,
    config: GardConfig,
    contract: SimulationContract,
    rng: np.random.Generator,
) -> FissionRecord:
    parent, steps = _grow_to_fission(composition, beta, config, contract, rng)
    daughter = _fission(parent, config, contract, rng)
    return FissionRecord(
        parent=parent,
        daughter=daughter,
        h=cosine_similarity(parent, daughter),
        growth_steps=steps,
    )


def simulate_lineage(
    initial: IntVector,
    beta: FloatMatrix,
    config: GardConfig,
    contract: SimulationContract,
    rng: np.random.Generator,
) -> list[Snapshot]:
    current = np.asarray(initial, dtype=np.int64).copy()
    inheritance: list[bool] = []
    boundary_h: list[float] = []
    snapshots: list[Snapshot] = []
    cumulative_growth_steps = 0
    for generation in range(1, config.generations + 1):
        record = advance_fission(current, beta, config, contract, rng)
        cumulative_growth_steps += record.growth_steps
        boundary_h.append(record.h)
        inheritance.append(record.h > config.inheritance_threshold)
        current = record.daughter
        snapshots.append(
            Snapshot(
                composition=current.copy(),
                generation=generation,
                inheritance=tuple(inheritance),
                boundary_h=tuple(boundary_h),
                previous_growth_steps=record.growth_steps,
                cumulative_growth_steps=cumulative_growth_steps,
            )
        )
    return snapshots


def simulate_future(
    snapshot: Snapshot,
    beta: FloatMatrix,
    config: GardConfig,
    contract: SimulationContract,
    horizon: int,
    rng: np.random.Generator,
) -> list[FissionRecord]:
    current = np.asarray(snapshot.composition, dtype=np.int64).copy()
    records: list[FissionRecord] = []
    for _ in range(horizon):
        record = advance_fission(current, beta, config, contract, rng)
        records.append(record)
        current = record.daughter
    return records


def simulate_future_absorbing(
    snapshot: Snapshot,
    beta: FloatMatrix,
    config: GardConfig,
    contract: SimulationContract,
    horizon: int,
    rng: np.random.Generator,
) -> tuple[list[FissionRecord], bool]:
    """Simulate until the horizon or absorbing extinction.

    Extinction is not treated as a fission or an inheritance break. A process
    event certified before extinction remains positive; an uncertified event is
    negative because no later fissions occur in the absorbing future.
    """

    current = np.asarray(snapshot.composition, dtype=np.int64).copy()
    records: list[FissionRecord] = []
    for _ in range(horizon):
        try:
            record = advance_fission(current, beta, config, contract, rng)
        except SimulationError:
            return records, False
        records.append(record)
        current = record.daughter
    return records, True
