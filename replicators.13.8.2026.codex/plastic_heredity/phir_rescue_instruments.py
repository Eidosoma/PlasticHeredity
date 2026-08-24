"""Theory-motivated Phi-r rescue instruments.

These functions are additive to the sealed Chapter 5 instruments.  The legacy
implementation remains untouched and is used as an exact replay control.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np
from numpy.typing import NDArray
from scipy.special import ndtr, ndtri
from scipy.stats import rankdata

from .phir_instruments import (
    ACTIVE_STD_EPS,
    ALL_ATOMS,
    ANTICHAINS,
    ATOM_NAMES,
    ATOM_ORDER,
    PHIR_ATOMS,
    REDUNDANT,
    SYNERGISTIC,
    UNIQUE_0,
    UNIQUE_1,
    _atom_leq,
    _local_surprisal,
    _paired,
    fiedler_bipartition,
    gaussian_mutual_information,
)


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class MacroPhiScore:
    revised: float
    whole_mi: float
    causation: float
    emergence: float
    synergy_persistence: float
    atoms: FloatArray


@dataclass(frozen=True)
class FullBlockScore:
    revised: float
    whole_mi: float
    aa_mi: float
    ab_mi: float
    ba_mi: float
    bb_mi: float
    double_redundancy: float


@dataclass(frozen=True)
class NullScore:
    observed: float
    null_mean: float
    null_std: float
    z_score: float
    percentile: float
    draws: int


@dataclass(frozen=True)
class NumitCalibration:
    percentile: float
    probit: float
    observed_whole_mi: float
    reference_min_mi: float
    reference_max_mi: float
    neighbors: int
    valid: bool


def close_all_clr(counts: NDArray, pseudocount: float = 0.5) -> FloatArray:
    """Return all CLR coordinates as dimensions by observations."""

    values = np.asarray(counts, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 3:
        raise ValueError("counts must be observations by at least three types")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("counts must be finite and nonnegative")
    if pseudocount <= 0.0:
        raise ValueError("pseudocount must be positive")
    replaced = values + float(pseudocount)
    # CLR(log(close(x))) is algebraically log(x)-mean(log(x)).  Computing it
    # directly avoids a redundant closure and sorting before the reduction
    # makes the floating-point result invariant to a molecule-label permutation.
    logged = np.log(replaced)
    sorted_logged = np.sort(logged, axis=1)
    center = np.asarray(
        [math.fsum(float(item) for item in row) / row.size for row in sorted_logged],
        dtype=np.float64,
    )[:, None]
    return np.asarray((logged - center).T, dtype=np.float64)


def rank_gaussianize(data: NDArray) -> tuple[FloatArray, IntArray]:
    """Average-rank Gaussian copula transform with deterministic inactive filtering."""

    values = np.asarray(data, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 3 or not np.isfinite(values).all():
        raise ValueError("rank Gaussianization requires finite dimensions by samples")
    standard_deviation = values.std(axis=1)
    active = np.flatnonzero(standard_deviation > ACTIVE_STD_EPS).astype(np.int64)
    if active.size < 2:
        raise ValueError("fewer than two active dimensions")
    selected = values[active]
    output = np.empty_like(selected)
    samples = selected.shape[1]
    for index, row in enumerate(selected):
        ranks = rankdata(row, method="average")
        probabilities = (ranks - 0.5) / samples
        output[index] = ndtri(probabilities)
    output -= output.mean(axis=1, keepdims=True)
    scales = output.std(axis=1, keepdims=True)
    if np.any(scales <= 0.0) or not np.isfinite(scales).all():
        raise ValueError("rank Gaussianization produced an inactive coordinate")
    output /= scales
    return np.asarray(output, dtype=np.float64), active


def beta_physical_partition(beta: NDArray) -> tuple[IntArray, IntArray]:
    """Deterministic arm-independent Fiedler split of the symmetrized log beta web."""

    matrix = np.asarray(beta, dtype=np.float64)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or matrix.shape[0] < 2
        or not np.isfinite(matrix).all()
        or np.any(matrix <= 0.0)
    ):
        raise ValueError("beta must be a finite positive square matrix")
    logged = np.log1p(matrix)
    graph = 0.5 * (logged + logged.T)
    np.fill_diagonal(graph, 0.0)
    return fiedler_bipartition(graph)


def active_partition(
    active: IntArray, partition_a: IntArray, partition_b: IntArray
) -> tuple[IntArray, IntArray]:
    """Map species-index partitions into an active-coordinate array."""

    active_values = np.asarray(active, dtype=np.int64)
    lookup = {int(species): index for index, species in enumerate(active_values)}
    first = np.asarray(
        [lookup[int(species)] for species in partition_a if int(species) in lookup],
        dtype=np.int64,
    )
    second = np.asarray(
        [lookup[int(species)] for species in partition_b if int(species) in lookup],
        dtype=np.int64,
    )
    if first.size == 0 or second.size == 0:
        raise ValueError("physical partition lost a complete active block")
    if first.size + second.size != active_values.size:
        raise AssertionError("active physical partition is incomplete")
    return first, second


def _cached_local_phi_id_atoms(
    past_macro: FloatArray, future_macro: FloatArray
) -> dict[tuple, FloatArray]:
    """Exact public local PhiID lattice with cached Gaussian surprisals."""

    past = np.asarray(past_macro, dtype=np.float64)
    future = np.asarray(future_macro, dtype=np.float64)
    if past.shape != future.shape or past.ndim != 2 or past.shape[0] != 2:
        raise ValueError("macro variables must have matching shape (2, samples)")
    subsets = ((0,), (1,), (0, 1))
    past_h = {
        subset: _local_surprisal(past[np.asarray(subset, dtype=np.int64)])
        for subset in subsets
    }
    future_h = {
        subset: _local_surprisal(future[np.asarray(subset, dtype=np.int64)])
        for subset in subsets
    }
    joint_h = {
        (source, target): _local_surprisal(
            np.vstack(
                (
                    past[np.asarray(source, dtype=np.int64)],
                    future[np.asarray(target, dtype=np.int64)],
                )
            )
        )
        for source in subsets
        for target in subsets
    }
    partial: dict[tuple, FloatArray] = {}
    for atom in ATOM_ORDER:
        informative = np.minimum.reduce([past_h[source] for source in atom[0]])
        misinformative = np.minimum.reduce(
            [
                joint_h[(source, target)] - future_h[target]
                for source in atom[0]
                for target in atom[1]
            ]
        )
        redundancy = informative - misinformative
        lower = [
            partial[other]
            for other in partial
            if other != atom and _atom_leq(other, atom)
        ]
        partial[atom] = redundancy - (np.sum(lower, axis=0) if lower else 0.0)
    if set(partial) != set(ALL_ATOMS):
        raise AssertionError("cached PhiID lattice did not produce all atoms")
    return partial


def macro_phi_score(
    data: FloatArray,
    partition_a: IntArray,
    partition_b: IntArray,
    valid_pairs: BoolArray | None = None,
) -> MacroPhiScore:
    macro = np.vstack(
        (data[np.asarray(partition_a, dtype=np.int64)].mean(axis=0),
         data[np.asarray(partition_b, dtype=np.int64)].mean(axis=0))
    )
    past, future = _paired(macro, valid_pairs)
    atom_series = _cached_local_phi_id_atoms(past, future)
    means = {atom: float(np.mean(value)) for atom, value in atom_series.items()}
    atoms = np.asarray(
        [means[(source, target)] for source in ANTICHAINS for target in ANTICHAINS],
        dtype=np.float64,
    )
    revised = float(sum(means[atom] for atom in PHIR_ATOMS))
    synergy = means[(SYNERGISTIC, SYNERGISTIC)]
    causation = means[(SYNERGISTIC, UNIQUE_0)] + means[(SYNERGISTIC, UNIQUE_1)]
    return MacroPhiScore(
        revised=revised,
        whole_mi=gaussian_mutual_information(past, future),
        causation=float(causation),
        emergence=float(causation + synergy),
        synergy_persistence=float(synergy),
        atoms=atoms,
    )


def full_block_revised(
    data: FloatArray,
    partition_a: IntArray,
    partition_b: IntArray,
    valid_pairs: BoolArray | None = None,
) -> FullBlockScore:
    past, future = _paired(np.asarray(data, dtype=np.float64), valid_pairs)
    first = np.asarray(partition_a, dtype=np.int64)
    second = np.asarray(partition_b, dtype=np.int64)
    whole = gaussian_mutual_information(past, future)
    aa = gaussian_mutual_information(past[first], future[first])
    ab = gaussian_mutual_information(past[first], future[second])
    ba = gaussian_mutual_information(past[second], future[first])
    bb = gaussian_mutual_information(past[second], future[second])
    redundancy = min(aa, ab, ba, bb)
    return FullBlockScore(
        revised=float(whole - aa - bb + redundancy),
        whole_mi=float(whole),
        aa_mi=float(aa),
        ab_mi=float(ab),
        ba_mi=float(ba),
        bb_mi=float(bb),
        double_redundancy=float(redundancy),
    )


def matched_partition_null(
    data: FloatArray,
    size_a: int,
    draws: int,
    rng: np.random.Generator,
    observed: float,
) -> tuple[NullScore, FloatArray]:
    dimensions = int(np.asarray(data).shape[0])
    if not 1 <= size_a < dimensions or draws < 1:
        raise ValueError("invalid matched-partition null dimensions or draws")
    values = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        order = rng.permutation(dimensions)
        first = np.sort(order[:size_a]).astype(np.int64)
        second = np.sort(order[size_a:]).astype(np.int64)
        values[draw] = full_block_revised(data, first, second).revised
    mean = float(values.mean())
    standard_deviation = float(values.std(ddof=0))
    z_score = (
        float((observed - mean) / standard_deviation)
        if standard_deviation > 0.0
        else float("nan")
    )
    below = np.count_nonzero(values < observed)
    tied = np.count_nonzero(values == observed)
    percentile = float((below + 0.5 * tied + 0.5) / (draws + 1.0))
    return (
        NullScore(observed, mean, standard_deviation, z_score, percentile, draws),
        values,
    )


def _stable_var_parameters(
    rng: np.random.Generator, systems: int
) -> tuple[FloatArray, FloatArray]:
    raw = rng.normal(size=(systems, 2, 2))
    eigenvalues = np.linalg.eigvals(raw)
    radii = np.max(np.abs(eigenvalues), axis=1)
    targets = rng.uniform(0.0, 0.995, size=systems)
    transitions = raw * (targets / np.maximum(radii, 1e-12))[:, None, None]
    factors = np.zeros((systems, 2, 2), dtype=np.float64)
    factors[:, 0, 0] = np.exp(rng.normal(0.0, 0.5, size=systems))
    factors[:, 1, 0] = rng.normal(0.0, 0.5, size=systems)
    factors[:, 1, 1] = np.exp(rng.normal(0.0, 0.5, size=systems))
    covariance = factors @ np.swapaxes(factors, 1, 2)
    covariance *= (2.0 / np.trace(covariance, axis1=1, axis2=2))[:, None, None]
    innovations = np.linalg.cholesky(covariance)
    return np.asarray(transitions), np.asarray(innovations)


def generate_numit_library(
    transitions: int,
    systems: int,
    rng: np.random.Generator,
    burn: int = 512,
) -> dict[str, FloatArray]:
    """Generate a finite-sample bivariate VAR reference library."""

    if transitions < 2 or systems < 1 or burn < 0:
        raise ValueError("invalid NuMIT library dimensions")
    matrices, factors = _stable_var_parameters(rng, systems)
    current = np.zeros((systems, 2), dtype=np.float64)
    kept = np.empty((systems, 2, transitions + 1), dtype=np.float64)
    for step in range(burn + transitions + 1):
        noise = rng.normal(size=(systems, 2))
        innovation = np.einsum("mij,mj->mi", factors, noise, optimize=True)
        current = np.einsum("mij,mj->mi", matrices, current, optimize=True) + innovation
        if step >= burn:
            kept[:, :, step - burn] = current
    whole = np.empty(systems, dtype=np.float64)
    revised = np.empty(systems, dtype=np.float64)
    for index in range(systems):
        score = macro_phi_score(
            kept[index], np.asarray([0], dtype=np.int64), np.asarray([1], dtype=np.int64)
        )
        whole[index] = score.whole_mi
        revised[index] = score.revised
    return {
        "whole_mi": whole,
        "revised": revised,
        "transition_count": np.asarray([transitions], dtype=np.int64),
        "systems": np.asarray([systems], dtype=np.int64),
    }


def calibrate_numit(
    observed_revised: float,
    observed_whole_mi: float,
    library: Mapping[str, NDArray],
    neighbors: int = 256,
) -> NumitCalibration:
    whole = np.asarray(library["whole_mi"], dtype=np.float64)
    revised = np.asarray(library["revised"], dtype=np.float64)
    if whole.shape != revised.shape or whole.ndim != 1 or neighbors < 1:
        raise ValueError("invalid NuMIT library")
    finite = np.isfinite(whole) & np.isfinite(revised)
    whole = whole[finite]
    revised = revised[finite]
    if whole.size < neighbors:
        raise ValueError("NuMIT library has too few finite systems")
    minimum = float(whole.min())
    maximum = float(whole.max())
    valid = bool(
        np.isfinite(observed_revised)
        and np.isfinite(observed_whole_mi)
        and minimum <= observed_whole_mi <= maximum
    )
    if not valid:
        return NumitCalibration(
            float("nan"), float("nan"), float(observed_whole_mi), minimum, maximum,
            neighbors, False,
        )
    order = np.argsort(np.abs(whole - observed_whole_mi), kind="stable")[:neighbors]
    local = revised[order]
    below = np.count_nonzero(local < observed_revised)
    tied = np.count_nonzero(local == observed_revised)
    percentile = float((below + 0.5 * tied + 0.5) / (neighbors + 1.0))
    clipped = float(np.clip(percentile, 0.5 / (neighbors + 1), 1.0 - 0.5 / (neighbors + 1)))
    return NumitCalibration(
        percentile=percentile,
        probit=float(ndtri(clipped)),
        observed_whole_mi=float(observed_whole_mi),
        reference_min_mi=minimum,
        reference_max_mi=maximum,
        neighbors=neighbors,
        valid=True,
    )


def numit_uniformity_pvalue(calibrations: NDArray) -> float:
    """Small validation helper: two-sided Kolmogorov distance approximation."""

    values = np.sort(np.asarray(calibrations, dtype=np.float64))
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float("nan")
    empirical = np.arange(1, values.size + 1, dtype=np.float64) / values.size
    distance = max(
        float(np.max(np.abs(empirical - values))),
        float(np.max(np.abs((empirical - 1.0 / values.size) - values))),
    )
    return float(min(1.0, 2.0 * np.exp(-2.0 * values.size * distance * distance)))


def atom_mapping(score: MacroPhiScore) -> dict[str, float]:
    return {
        name: float(value)
        for name, value in zip(ATOM_NAMES, score.atoms, strict=True)
    }


__all__ = [
    "FullBlockScore",
    "MacroPhiScore",
    "NullScore",
    "NumitCalibration",
    "active_partition",
    "atom_mapping",
    "beta_physical_partition",
    "calibrate_numit",
    "close_all_clr",
    "full_block_revised",
    "generate_numit_library",
    "macro_phi_score",
    "matched_partition_null",
    "numit_uniformity_pvalue",
    "rank_gaussianize",
]
