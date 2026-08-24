"""Independent information instruments for the Chapter 5 Phi-r program.

This module is a clean-room implementation from the public typeset equation
and the public PhiRL method.  It contains no Fable source, data, seeds, or
serialized objects.  The existing Codex simulator is not modified: the traced
helpers below mirror its frozen stochastic contract and are required to agree
bitwise with it on every validation fixture.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh

from .config import GardConfig, SimulationContract
from .simulator import (
    FissionRecord,
    SimulationError,
    _fission,
    _sample_without_replacement,
    _trim_whole_assembly,
    cosine_similarity,
)


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]

GRAPH_FLOOR = 1e-6
ACTIVE_STD_EPS = 1e-8
COVARIANCE_RIDGE = 1e-6
ZERO_REPLACEMENT = 0.5

# The four antichains of the non-empty subsets of a two-element system.
REDUNDANT = ((0,), (1,))
UNIQUE_0 = ((0,),)
UNIQUE_1 = ((1,),)
SYNERGISTIC = ((0, 1),)
ANTICHAINS = (REDUNDANT, UNIQUE_0, UNIQUE_1, SYNERGISTIC)
Atom = tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]

# Public PhiRL's local_phi_r is this initial atom plus the following eight.
PHIR_ATOMS: tuple[Atom, ...] = (
    (UNIQUE_0, SYNERGISTIC),
    (REDUNDANT, SYNERGISTIC),
    (UNIQUE_1, SYNERGISTIC),
    (SYNERGISTIC, UNIQUE_0),
    (SYNERGISTIC, REDUNDANT),
    (SYNERGISTIC, UNIQUE_1),
    (SYNERGISTIC, SYNERGISTIC),
    (UNIQUE_0, UNIQUE_1),
    (UNIQUE_1, UNIQUE_0),
)


def atom_name(atom: Atom) -> str:
    def side(value: tuple[tuple[int, ...], ...]) -> str:
        lookup = {
            REDUNDANT: "r",
            UNIQUE_0: "u0",
            UNIQUE_1: "u1",
            SYNERGISTIC: "s",
        }
        return lookup[value]

    return f"{side(atom[0])}_to_{side(atom[1])}"


ATOM_NAMES = tuple(atom_name((source, target)) for source in ANTICHAINS for target in ANTICHAINS)


@dataclass(frozen=True)
class PhiWindowScore:
    """All registered readings for one past-only observation window."""

    revised_phi_r: float
    typeset_phi_r: float
    text_normalized_phi_r: float
    causation: float
    emergence: float
    synergy_persistence: float
    atoms: FloatArray
    active_dimensions: int
    partition_a: tuple[int, ...]
    partition_b: tuple[int, ...]
    observations: int
    transitions: int
    digest: str


@dataclass(frozen=True)
class TracedFission:
    record: FissionRecord
    growth_observations: tuple[IntArray, ...]


def _canonical_array_digest(*arrays: NDArray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def close_clr_drop_last(counts: NDArray, pseudocount: float = ZERO_REPLACEMENT) -> FloatArray:
    """Close count rows, apply CLR, and remove the final coordinate."""

    values = np.asarray(counts, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 3:
        raise ValueError("counts must be a time-by-types matrix with at least 3 types")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("counts must be finite and nonnegative")
    if pseudocount <= 0.0:
        raise ValueError("pseudocount must be positive")
    replaced = values + float(pseudocount)
    closed = replaced / replaced.sum(axis=1, keepdims=True)
    logged = np.log(closed)
    clr = logged - logged.mean(axis=1, keepdims=True)
    output = np.asarray(clr[:, :-1].T, dtype=np.float64)
    if output.shape != (values.shape[1] - 1, values.shape[0]):
        raise AssertionError("CLR shape contract failed")
    return output


def _active_zscore(data: FloatArray) -> tuple[FloatArray, IntArray]:
    values = np.asarray(data, dtype=np.float64)
    standard_deviation = values.std(axis=1)
    active = np.flatnonzero(np.isfinite(standard_deviation) & (standard_deviation > ACTIVE_STD_EPS))
    if active.size < 2:
        raise ValueError("fewer than two active dimensions in Phi window")
    selected = values[active]
    output = (selected - selected.mean(axis=1, keepdims=True)) / selected.std(
        axis=1, keepdims=True
    )
    if not np.isfinite(output).all():
        raise ValueError("non-finite standardized Phi window")
    return np.asarray(output, dtype=np.float64), active.astype(np.int64)


def _paired(data: FloatArray, valid_pairs: BoolArray | None) -> tuple[FloatArray, FloatArray]:
    values = np.asarray(data, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 3:
        raise ValueError("Phi windows require at least three observations")
    if valid_pairs is None:
        mask = np.ones(values.shape[1] - 1, dtype=bool)
    else:
        mask = np.asarray(valid_pairs, dtype=bool)
        if mask.shape != (values.shape[1] - 1,):
            raise ValueError("pair mask does not align with observations")
    if int(mask.sum()) < 2:
        raise ValueError("Phi windows require at least two valid transitions")
    return values[:, :-1][:, mask], values[:, 1:][:, mask]


def lagged_gaussian_mi_graph(
    data: FloatArray, valid_pairs: BoolArray | None = None
) -> FloatArray:
    """Public-PhiRL-style symmetric lag-one Gaussian-MI graph."""

    past, future = _paired(data, valid_pairs)
    dimensions = past.shape[0]
    first = np.corrcoef(np.concatenate((past, future), axis=0))[:dimensions, dimensions:]
    second = np.corrcoef(np.concatenate((future, past), axis=0))[:dimensions, dimensions:]
    correlation = 0.5 * (first + second)
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.999999, neginf=-0.999999)
    correlation = np.clip(correlation, -0.999999, 0.999999)
    mutual_information = -0.5 * np.log1p(-(correlation * correlation))
    mutual_information = 0.5 * (mutual_information + mutual_information.T)
    np.fill_diagonal(mutual_information, 0.0)
    return np.asarray(mutual_information, dtype=np.float64)


def fiedler_bipartition(weight_matrix: FloatArray) -> tuple[IntArray, IntArray]:
    matrix = np.asarray(weight_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2:
        raise ValueError("Fiedler graph must be square with at least two nodes")
    if not np.isfinite(matrix).all() or np.any(matrix < 0.0):
        raise ValueError("Fiedler graph weights must be finite and nonnegative")
    connected = matrix.copy()
    connected += GRAPH_FLOOR
    np.fill_diagonal(connected, 0.0)
    laplacian = np.diag(connected.sum(axis=1)) - connected
    _, vectors = eigh(laplacian, check_finite=True, driver="evr")
    vector = np.asarray(vectors[:, 1], dtype=np.float64)
    pivot = int(np.argmax(np.abs(vector)))
    if vector[pivot] < 0.0:
        vector = -vector
    tolerance = 64.0 * np.finfo(np.float64).eps * max(1.0, float(np.max(np.abs(vector))))
    positive = np.flatnonzero(vector > tolerance)
    negative = np.flatnonzero(vector < -tolerance)
    zeros = np.flatnonzero(np.abs(vector) <= tolerance)
    for index in zeros:
        if positive.size <= negative.size:
            positive = np.append(positive, index)
        else:
            negative = np.append(negative, index)
    if positive.size == 0 or negative.size == 0:
        order = np.lexsort((np.arange(vector.size), vector))
        split = vector.size // 2
        negative, positive = order[:split], order[split:]
    return np.sort(positive).astype(np.int64), np.sort(negative).astype(np.int64)


def _regularized_covariance(data: FloatArray) -> FloatArray:
    values = np.atleast_2d(np.asarray(data, dtype=np.float64))
    if values.shape[1] < 2:
        raise ValueError("covariance requires at least two samples")
    covariance = np.atleast_2d(np.cov(values, ddof=0))
    trace = float(np.trace(covariance))
    ridge = max(1e-8, COVARIANCE_RIDGE * trace / covariance.shape[0])
    output = covariance + np.eye(covariance.shape[0], dtype=np.float64) * ridge
    if not np.isfinite(output).all():
        raise ValueError("non-finite covariance")
    return output


def _logdet_covariance(data: FloatArray) -> float:
    sign, value = np.linalg.slogdet(_regularized_covariance(data))
    if sign <= 0.0 or not np.isfinite(value):
        raise ValueError("regularized covariance is not positive definite")
    return float(value)


def gaussian_mutual_information(left: FloatArray, right: FloatArray) -> float:
    x = np.atleast_2d(np.asarray(left, dtype=np.float64))
    y = np.atleast_2d(np.asarray(right, dtype=np.float64))
    if x.shape[1] != y.shape[1]:
        raise ValueError("mutual-information sample counts differ")
    value = 0.5 * (
        _logdet_covariance(x)
        + _logdet_covariance(y)
        - _logdet_covariance(np.vstack((x, y)))
    )
    return float(value)


def typeset_whole_minus_parts(
    data: FloatArray,
    partition_a: IntArray,
    partition_b: IntArray,
    valid_pairs: BoolArray | None = None,
) -> tuple[float, float]:
    """Return the verbatim unnormalized numerator and whole-system MI."""

    past, future = _paired(data, valid_pairs)
    whole = gaussian_mutual_information(past, future)
    part_a = gaussian_mutual_information(past[partition_a], future)
    part_b = gaussian_mutual_information(past[partition_b], future)
    return float(whole - part_a - part_b), float(whole)


def _local_surprisal(data: FloatArray) -> FloatArray:
    values = np.atleast_2d(np.asarray(data, dtype=np.float64))
    if values.shape[0] == 1:
        mean = float(values[0].mean())
        standard_deviation = float(values[0].std())
        if standard_deviation <= 0.0 or not np.isfinite(standard_deviation):
            raise ValueError("one-dimensional local entropy has zero variance")
        centered = (values[0] - mean) / standard_deviation
        return (
            0.5 * np.log(2.0 * np.pi)
            + np.log(standard_deviation)
            + 0.5 * centered * centered
        )
    covariance = np.atleast_2d(np.cov(values, ddof=0))
    ridge = COVARIANCE_RIDGE * float(np.trace(covariance)) / covariance.shape[0]
    covariance = covariance + np.eye(covariance.shape[0], dtype=np.float64) * ridge
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0.0:
        raise ValueError("local Gaussian covariance is not positive definite")
    centered = values - values.mean(axis=1, keepdims=True)
    solved = np.linalg.solve(covariance, centered)
    mahalanobis = np.sum(centered * solved, axis=0)
    return 0.5 * (
        values.shape[0] * np.log(2.0 * np.pi) + logdet + mahalanobis
    )


def _antichain_leq(
    lower: tuple[tuple[int, ...], ...], upper: tuple[tuple[int, ...], ...]
) -> bool:
    return all(any(set(left).issubset(right) for left in lower) for right in upper)


def _atom_leq(lower: Atom, upper: Atom) -> bool:
    return _antichain_leq(lower[0], upper[0]) and _antichain_leq(lower[1], upper[1])


ALL_ATOMS: tuple[Atom, ...] = tuple(
    (source, target) for source in ANTICHAINS for target in ANTICHAINS
)


def _atom_rank(atom: Atom) -> tuple[int, str]:
    strict_lower = sum(other != atom and _atom_leq(other, atom) for other in ALL_ATOMS)
    return strict_lower, atom_name(atom)


ATOM_ORDER = tuple(sorted(ALL_ATOMS, key=_atom_rank))


def _local_redundancy(
    atom: Atom, past_macro: FloatArray, future_macro: FloatArray
) -> FloatArray:
    informative = np.full(past_macro.shape[1], np.inf, dtype=np.float64)
    misinformative = np.full(past_macro.shape[1], np.inf, dtype=np.float64)
    for source in atom[0]:
        source_data = past_macro[np.asarray(source, dtype=np.int64)]
        informative = np.minimum(informative, _local_surprisal(source_data))
        for target in atom[1]:
            target_data = future_macro[np.asarray(target, dtype=np.int64)]
            joint = np.vstack((source_data, target_data))
            conditional = _local_surprisal(joint) - _local_surprisal(target_data)
            misinformative = np.minimum(misinformative, conditional)
    return informative - misinformative


def local_phi_id_atoms(
    past_macro: FloatArray, future_macro: FloatArray
) -> dict[Atom, FloatArray]:
    past = np.asarray(past_macro, dtype=np.float64)
    future = np.asarray(future_macro, dtype=np.float64)
    if past.shape != future.shape or past.ndim != 2 or past.shape[0] != 2:
        raise ValueError("PhiID macro variables must have matching shape (2, samples)")
    partial: dict[Atom, FloatArray] = {}
    for atom in ATOM_ORDER:
        redundancy = _local_redundancy(atom, past, future)
        lower = [
            partial[other]
            for other in partial
            if other != atom and _atom_leq(other, atom)
        ]
        partial[atom] = redundancy - (np.sum(lower, axis=0) if lower else 0.0)
    if set(partial) != set(ALL_ATOMS):
        raise AssertionError("PhiID lattice did not produce all 16 atoms")
    return partial


def revised_phi_from_partition(
    data: FloatArray,
    partition_a: IntArray,
    partition_b: IntArray,
    valid_pairs: BoolArray | None = None,
) -> tuple[float, float, float, float, FloatArray]:
    macro = np.vstack((data[partition_a].mean(axis=0), data[partition_b].mean(axis=0)))
    past, future = _paired(macro, valid_pairs)
    atoms = local_phi_id_atoms(past, future)
    means = {atom: float(np.mean(value)) for atom, value in atoms.items()}
    revised = float(sum(means[atom] for atom in PHIR_ATOMS))
    synergy = means[(SYNERGISTIC, SYNERGISTIC)]
    causation = means[(SYNERGISTIC, UNIQUE_0)] + means[(SYNERGISTIC, UNIQUE_1)]
    emergence = synergy + causation
    atom_vector = np.asarray(
        [means[(source, target)] for source in ANTICHAINS for target in ANTICHAINS],
        dtype=np.float64,
    )
    return revised, causation, emergence, synergy, atom_vector


def score_phi_window(
    counts: NDArray,
    valid_pairs: BoolArray | None = None,
    *,
    include_typeset: bool = True,
) -> PhiWindowScore:
    raw = np.asarray(counts, dtype=np.float64)
    clr = close_clr_drop_last(raw)
    data, active = _active_zscore(clr)
    if valid_pairs is not None:
        pair_mask = np.asarray(valid_pairs, dtype=bool)
    else:
        pair_mask = None
    graph = lagged_gaussian_mi_graph(data, pair_mask)
    partition_a, partition_b = fiedler_bipartition(graph)
    if include_typeset:
        typeset, whole = typeset_whole_minus_parts(
            data, partition_a, partition_b, pair_mask
        )
    else:
        typeset, whole = float("nan"), float("nan")
    revised, causation, emergence, synergy, atoms = revised_phi_from_partition(
        data, partition_a, partition_b, pair_mask
    )
    normalized = float(typeset / whole) if np.isfinite(whole) and abs(whole) > 1e-12 else float("nan")
    digest = _canonical_array_digest(raw, data, graph, partition_a, partition_b, atoms)
    transitions = raw.shape[0] - 1 if pair_mask is None else int(pair_mask.sum())
    return PhiWindowScore(
        revised_phi_r=float(revised),
        typeset_phi_r=float(typeset),
        text_normalized_phi_r=normalized,
        causation=float(causation),
        emergence=float(emergence),
        synergy_persistence=float(synergy),
        atoms=atoms,
        active_dimensions=int(active.size),
        partition_a=tuple(int(active[index]) for index in partition_a),
        partition_b=tuple(int(active[index]) for index in partition_b),
        observations=int(raw.shape[0]),
        transitions=transitions,
        digest=digest,
    )


def advance_fission_traced(
    composition: NDArray,
    beta: NDArray,
    config: GardConfig,
    contract: SimulationContract,
    rng: np.random.Generator,
) -> TracedFission:
    """Mirror one frozen Codex fission while exposing growth observations."""

    current = np.asarray(composition, dtype=np.int64).copy()
    matrix = np.asarray(beta, dtype=np.float64)
    rho = 1.0 / config.n_types
    observations: list[IntArray] = []
    parent: IntArray | None = None
    steps = 0
    for step in range(1, config.max_growth_steps + 1):
        mass = int(current.sum())
        if mass <= 0:
            raise SimulationError("assembly became extinct")
        if mass >= config.n_max:
            parent = _trim_whole_assembly(current, config.n_max, rng)
            steps = step - 1
            break
        catalytic_boost = 1.0 + (matrix @ current) / mass
        join_rate = config.k_join * rho * mass * catalytic_boost
        leave_rate = config.k_leave * current * catalytic_boost
        joins = np.asarray(
            rng.poisson(join_rate * contract.poisson_exposure), dtype=np.int64
        )
        leaves = np.minimum(
            np.asarray(
                rng.poisson(leave_rate * contract.poisson_exposure), dtype=np.int64
            ),
            current,
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
        observations.append(current.copy())
        if int(current.sum()) >= config.n_max:
            parent = current.copy()
            steps = step
            break
    if parent is None:
        raise SimulationError(
            f"growth did not reach mass {config.n_max} in {config.max_growth_steps} steps"
        )
    daughter = _fission(parent, config, contract, rng)
    record = FissionRecord(
        parent=parent,
        daughter=daughter,
        h=cosine_similarity(parent, daughter),
        growth_steps=steps,
    )
    return TracedFission(record, tuple(observations))


def rng_states_equal(left: dict, right: dict) -> bool:
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def records_equal(left: FissionRecord, right: FissionRecord) -> bool:
    return bool(
        np.array_equal(left.parent, right.parent)
        and np.array_equal(left.daughter, right.daughter)
        and np.asarray(left.h, dtype=np.float64).tobytes()
        == np.asarray(right.h, dtype=np.float64).tobytes()
        and left.growth_steps == right.growth_steps
    )


def trailing_run(values: Sequence[bool]) -> int:
    count = 0
    for value in reversed(values):
        if not value:
            break
        count += 1
    return count


def observations_digest(
    observations: Iterable[NDArray], transition_kinds: Iterable[int]
) -> str:
    values = tuple(np.asarray(item, dtype=np.int64) for item in observations)
    kinds = np.asarray(tuple(transition_kinds), dtype=np.int8)
    return _canonical_array_digest(*(values + (kinds,)))
