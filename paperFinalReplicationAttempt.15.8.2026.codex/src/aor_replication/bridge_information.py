"""Frozen information instruments for the Arrivals formulation bridge.

The original estimator in :mod:`aor_replication.information` is called rather
than reimplemented.  The two PX instruments are a small, standalone port of
the non-author implementations source-hashed in ``FORMULATION_BRIDGE_PROTOCOL``.
They expose local transition series so the global PX formulas can be compared
to the original local estimator on identical trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Dict, Mapping, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import linalg
from scipy.special import ndtri
from scipy.stats import rankdata

from .config import CausalConfig
from .information import fit_causal_trajectory


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

ACTIVE_STD_EPS = 1e-8
COVARIANCE_RIDGE = 1e-6
GRAPH_FLOOR = 1e-6
ZERO_REPLACEMENT = 0.5

ESTIMATOR_ORDER = (
    "macro_wms",
    "macro_mmi",
    "public_nine_atom",
    "full_revised",
)

# Antichains of the non-empty subsets of a two-element macro system.
REDUNDANT = ((0,), (1,))
UNIQUE_0 = ((0,),)
UNIQUE_1 = ((1,),)
SYNERGISTIC = ((0, 1),)
ANTICHAINS = (REDUNDANT, UNIQUE_0, UNIQUE_1, SYNERGISTIC)
Atom = Tuple[Tuple[Tuple[int, ...], ...], Tuple[Tuple[int, ...], ...]]

# Public PhiRL's revised Phi-r is the sum of these nine PhiID atoms.
PHIR_ATOMS: Tuple[Atom, ...] = (
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


def _atom_name(atom: Atom) -> str:
    lookup = {
        REDUNDANT: "r",
        UNIQUE_0: "u0",
        UNIQUE_1: "u1",
        SYNERGISTIC: "s",
    }
    return f"{lookup[atom[0]]}_to_{lookup[atom[1]]}"


ALL_ATOMS: Tuple[Atom, ...] = tuple(
    (source, target) for source in ANTICHAINS for target in ANTICHAINS
)
ATOM_NAMES = tuple(_atom_name(atom) for atom in ALL_ATOMS)


def _antichain_leq(
    lower: Tuple[Tuple[int, ...], ...], upper: Tuple[Tuple[int, ...], ...]
) -> bool:
    return all(any(set(left).issubset(right) for left in lower) for right in upper)


def _atom_leq(lower: Atom, upper: Atom) -> bool:
    return _antichain_leq(lower[0], upper[0]) and _antichain_leq(
        lower[1], upper[1]
    )


def _atom_rank(atom: Atom) -> Tuple[int, str]:
    strict_lower = sum(
        other != atom and _atom_leq(other, atom) for other in ALL_ATOMS
    )
    return strict_lower, _atom_name(atom)


ATOM_ORDER = tuple(sorted(ALL_ATOMS, key=_atom_rank))


@dataclass(frozen=True)
class BridgeEstimatorTrajectory:
    """One frozen instrument's local scores and fitted scalar diagnostics."""

    name: str
    values: FloatArray
    time_indices: IntArray
    partition_a: IntArray
    partition_b: IntArray
    active_dimensions: int
    components: Mapping[str, float]
    redundancy_channel: str = "none"

    def validate(self) -> None:
        if self.name not in ESTIMATOR_ORDER:
            raise ValueError(f"unknown bridge estimator: {self.name}")
        if self.values.ndim != 1 or self.time_indices.shape != self.values.shape:
            raise ValueError("score values and time indices must be aligned vectors")
        if not np.isfinite(self.values).all():
            raise ValueError(f"{self.name} produced non-finite local scores")
        if not self.partition_a.size or not self.partition_b.size:
            raise ValueError(f"{self.name} produced an empty partition")
        if self.active_dimensions != self.partition_a.size + self.partition_b.size:
            raise ValueError(f"{self.name} partition does not cover active dimensions")
        if not all(np.isfinite(float(value)) for value in self.components.values()):
            raise ValueError(f"{self.name} produced non-finite components")


def close_all_clr(
    counts: ArrayLike, pseudocount: float = ZERO_REPLACEMENT
) -> FloatArray:
    """Return all CLR coordinates as dimensions by observations.

    The sorted summation makes the center invariant to a molecule-label
    permutation down to deterministic floating-point ordering.
    """

    values = np.asarray(counts, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 3 or values.shape[1] < 3:
        raise ValueError("counts must be observations by at least three types")
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("counts must be finite and nonnegative")
    if pseudocount <= 0.0:
        raise ValueError("pseudocount must be positive")
    logged = np.log(values + float(pseudocount))
    sorted_logged = np.sort(logged, axis=1)
    center = np.asarray(
        [math.fsum(float(item) for item in row) / row.size for row in sorted_logged],
        dtype=np.float64,
    )[:, None]
    return np.asarray((logged - center).T, dtype=np.float64)


def rank_gaussianize(data: ArrayLike) -> Tuple[FloatArray, IntArray]:
    """Average-rank Gaussian copula transform with frozen inactive filtering."""

    values = np.asarray(data, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 3 or not np.isfinite(values).all():
        raise ValueError("rank Gaussianization requires finite dimensions by samples")
    standard_deviation = values.std(axis=1, ddof=0)
    active = np.flatnonzero(standard_deviation > ACTIVE_STD_EPS).astype(np.int64)
    if active.size < 2:
        raise ValueError("fewer than two active molecular dimensions")
    selected = values[active]
    output = np.empty_like(selected)
    samples = selected.shape[1]
    for index, row in enumerate(selected):
        ranks = rankdata(row, method="average")
        output[index] = ndtri((ranks - 0.5) / samples)
    output -= output.mean(axis=1, keepdims=True)
    scales = output.std(axis=1, keepdims=True, ddof=0)
    if np.any(scales <= 0.0) or not np.isfinite(scales).all():
        raise ValueError("rank Gaussianization produced an inactive coordinate")
    output /= scales
    return np.asarray(output, dtype=np.float64), active


def _fiedler_index_partition(weight_matrix: ArrayLike) -> Tuple[IntArray, IntArray]:
    matrix = np.asarray(weight_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2:
        raise ValueError("Fiedler graph must be square with at least two nodes")
    if not np.isfinite(matrix).all() or np.any(matrix < 0.0):
        raise ValueError("Fiedler graph weights must be finite and nonnegative")
    connected = matrix.copy() + GRAPH_FLOOR
    np.fill_diagonal(connected, 0.0)
    laplacian = np.diag(connected.sum(axis=1)) - connected
    _, vectors = linalg.eigh(laplacian, check_finite=True, driver="evr")
    vector = np.asarray(vectors[:, 1], dtype=np.float64)
    pivot = int(np.argmax(np.abs(vector)))
    if vector[pivot] < 0.0:
        vector = -vector
    tolerance = (
        64.0
        * np.finfo(np.float64).eps
        * max(1.0, float(np.max(np.abs(vector))))
    )
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
    return (
        np.sort(positive).astype(np.int64),
        np.sort(negative).astype(np.int64),
    )


def beta_physical_partition(beta: ArrayLike) -> Tuple[IntArray, IntArray]:
    """Arm-independent Fiedler split of the symmetrized log catalytic web."""

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
    return _fiedler_index_partition(graph)


def _active_partition(
    active: IntArray, physical_a: IntArray, physical_b: IntArray
) -> Tuple[IntArray, IntArray]:
    lookup = {int(species): index for index, species in enumerate(active)}
    first = np.asarray(
        [lookup[int(species)] for species in physical_a if int(species) in lookup],
        dtype=np.int64,
    )
    second = np.asarray(
        [lookup[int(species)] for species in physical_b if int(species) in lookup],
        dtype=np.int64,
    )
    if not first.size or not second.size:
        raise ValueError("physical partition lost a complete active block")
    if first.size + second.size != active.size:
        raise AssertionError("active physical partition is incomplete")
    return first, second


def _regularized_covariance(data: ArrayLike) -> FloatArray:
    values = np.atleast_2d(np.asarray(data, dtype=np.float64))
    if values.shape[1] < 2:
        raise ValueError("covariance requires at least two samples")
    covariance = np.atleast_2d(np.cov(values, ddof=0))
    trace = float(np.trace(covariance))
    ridge = max(1e-8, COVARIANCE_RIDGE * trace / covariance.shape[0])
    output = covariance + np.eye(covariance.shape[0], dtype=np.float64) * ridge
    if not np.isfinite(output).all():
        raise ValueError("non-finite covariance")
    return np.asarray(output, dtype=np.float64)


def _logdet_covariance(data: ArrayLike) -> float:
    sign, value = np.linalg.slogdet(_regularized_covariance(data))
    if sign <= 0.0 or not np.isfinite(value):
        raise ValueError("regularized covariance is not positive definite")
    return float(value)


def gaussian_mutual_information(left: ArrayLike, right: ArrayLike) -> float:
    """Ridge-regularized plug-in multivariate Gaussian mutual information."""

    x = np.atleast_2d(np.asarray(left, dtype=np.float64))
    y = np.atleast_2d(np.asarray(right, dtype=np.float64))
    if x.shape[1] != y.shape[1] or x.shape[1] < 2:
        raise ValueError("mutual-information sample counts differ or are too small")
    return float(
        0.5
        * (
            _logdet_covariance(x)
            + _logdet_covariance(y)
            - _logdet_covariance(np.vstack((x, y)))
        )
    )


def _public_local_surprisal(data: ArrayLike) -> FloatArray:
    """Local Gaussian surprisal used by the frozen public-PhiRL port."""

    values = np.atleast_2d(np.asarray(data, dtype=np.float64))
    if values.shape[0] == 1:
        mean = float(values[0].mean())
        standard_deviation = float(values[0].std(ddof=0))
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
    covariance = covariance + np.eye(covariance.shape[0]) * ridge
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0.0 or not np.isfinite(logdet):
        raise ValueError("public local Gaussian covariance is not positive definite")
    centered = values - values.mean(axis=1, keepdims=True)
    solved = np.linalg.solve(covariance, centered)
    mahalanobis = np.sum(centered * solved, axis=0)
    return 0.5 * (
        values.shape[0] * np.log(2.0 * np.pi) + logdet + mahalanobis
    )


def local_phi_id_atoms(
    past_macro: ArrayLike, future_macro: ArrayLike
) -> Dict[Atom, FloatArray]:
    """Return all 16 local PhiID atoms for a two-variable Gaussian system."""

    past = np.asarray(past_macro, dtype=np.float64)
    future = np.asarray(future_macro, dtype=np.float64)
    if past.shape != future.shape or past.ndim != 2 or past.shape[0] != 2:
        raise ValueError("macro variables must have matching shape (2, samples)")
    subsets = ((0,), (1,), (0, 1))
    past_h = {
        subset: _public_local_surprisal(past[np.asarray(subset, dtype=np.int64)])
        for subset in subsets
    }
    future_h = {
        subset: _public_local_surprisal(future[np.asarray(subset, dtype=np.int64)])
        for subset in subsets
    }
    joint_h = {
        (source, target): _public_local_surprisal(
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
    partial: Dict[Atom, FloatArray] = {}
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
        raise AssertionError("PhiID lattice did not produce all 16 atoms")
    return partial


def _entropy_matched_local_surprisal(data: ArrayLike) -> FloatArray:
    """Local Gaussian surprisal centered to the registered plug-in entropy.

    Ridge regularization means raw fitted-model log likelihoods do not average
    exactly to the analytic Gaussian entropy.  The additive centering below
    makes every local mutual-information channel average exactly to the global
    log-determinant estimator while preserving all within-channel rankings.
    """

    values = np.atleast_2d(np.asarray(data, dtype=np.float64))
    covariance = _regularized_covariance(values)
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0.0 or not np.isfinite(logdet):
        raise ValueError("local Gaussian covariance is not positive definite")
    centered = values - values.mean(axis=1, keepdims=True)
    solved = np.linalg.solve(covariance, centered)
    mahalanobis = np.sum(centered * solved, axis=0)
    dimension = values.shape[0]
    raw = 0.5 * (dimension * np.log(2.0 * np.pi) + logdet + mahalanobis)
    entropy = 0.5 * (dimension * (1.0 + np.log(2.0 * np.pi)) + logdet)
    return np.asarray(raw + (entropy - float(raw.mean())), dtype=np.float64)


def _local_gaussian_information(left: ArrayLike, right: ArrayLike) -> FloatArray:
    x = np.atleast_2d(np.asarray(left, dtype=np.float64))
    y = np.atleast_2d(np.asarray(right, dtype=np.float64))
    if x.shape[1] != y.shape[1]:
        raise ValueError("local mutual-information sample counts differ")
    return (
        _entropy_matched_local_surprisal(x)
        + _entropy_matched_local_surprisal(y)
        - _entropy_matched_local_surprisal(np.vstack((x, y)))
    )


def full_block_local_scores(
    data: ArrayLike, partition_a: ArrayLike, partition_b: ArrayLike
) -> Tuple[FloatArray, Dict[str, float], str]:
    """Localize the frozen full-block revised formula without pointwise MMI."""

    values = np.asarray(data, dtype=np.float64)
    first = np.asarray(partition_a, dtype=np.int64)
    second = np.asarray(partition_b, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] < 3:
        raise ValueError("full-block data must be dimensions by observations")
    if not first.size or not second.size:
        raise ValueError("both full-block partitions must be nonempty")
    if first.size + second.size != values.shape[0]:
        raise ValueError("full-block partitions must cover every dimension")
    past = values[:, :-1]
    future = values[:, 1:]
    local = {
        "whole": _local_gaussian_information(past, future),
        "aa": _local_gaussian_information(past[first], future[first]),
        "ab": _local_gaussian_information(past[first], future[second]),
        "ba": _local_gaussian_information(past[second], future[first]),
        "bb": _local_gaussian_information(past[second], future[second]),
    }
    global_channels = {
        "whole": gaussian_mutual_information(past, future),
        "aa": gaussian_mutual_information(past[first], future[first]),
        "ab": gaussian_mutual_information(past[first], future[second]),
        "ba": gaussian_mutual_information(past[second], future[first]),
        "bb": gaussian_mutual_information(past[second], future[second]),
    }
    redundancy_channel = min(("aa", "ab", "ba", "bb"), key=global_channels.get)
    revised_global = (
        global_channels["whole"]
        - global_channels["aa"]
        - global_channels["bb"]
        + global_channels[redundancy_channel]
    )
    revised_local = (
        local["whole"] - local["aa"] - local["bb"] + local[redundancy_channel]
    )
    components = {
        "whole_mi": global_channels["whole"],
        "aa_mi": global_channels["aa"],
        "ab_mi": global_channels["ab"],
        "ba_mi": global_channels["ba"],
        "bb_mi": global_channels["bb"],
        "double_redundancy": global_channels[redundancy_channel],
        "global_revised": float(revised_global),
        "local_mean": float(np.mean(revised_local)),
        "mean_identity_error": float(np.mean(revised_local) - revised_global),
    }
    if not np.isclose(
        components["local_mean"], revised_global, rtol=1e-10, atol=1e-10
    ):
        raise AssertionError("localized full-block score failed its mean identity")
    return np.asarray(revised_local, dtype=np.float64), components, redundancy_channel


def _px_transform(
    counts: ArrayLike, beta: ArrayLike
) -> Tuple[FloatArray, IntArray, IntArray, IntArray]:
    raw = np.asarray(counts, dtype=np.float64)
    matrix = np.asarray(beta, dtype=np.float64)
    if raw.ndim != 2 or matrix.shape != (raw.shape[1], raw.shape[1]):
        raise ValueError("counts and beta molecular dimensions do not agree")
    data, active = rank_gaussianize(close_all_clr(raw))
    physical_a, physical_b = beta_physical_partition(matrix)
    first, second = _active_partition(active, physical_a, physical_b)
    return data, active, first, second


def fit_bridge_estimators(
    counts: ArrayLike,
    beta: ArrayLike,
    causal_config: CausalConfig = CausalConfig(),
) -> Dict[str, BridgeEstimatorTrajectory]:
    """Fit all four frozen bridge instruments to one observation window."""

    raw = np.asarray(counts)
    if raw.ndim != 2 or raw.shape[0] < 6:
        raise ValueError("bridge estimators require at least six observations")
    if causal_config.lag != 1:
        raise ValueError("the frozen bridge uses adjacent lag-one transitions")
    base_config = replace(causal_config, measure="wms")
    macro = fit_causal_trajectory(raw, base_config)
    past_grouped = macro.grouped[:-1]
    future_grouped = macro.grouped[1:]
    macro_information = macro.model.local_information(
        np.column_stack((past_grouped, future_grouped))
    )
    macro_mmi = macro.model.score_transitions(
        past_grouped, future_grouped, measure="mmi_synergy"
    )
    macro_a = np.flatnonzero(~macro.partition).astype(np.int64)
    macro_b = np.flatnonzero(macro.partition).astype(np.int64)
    macro_components = {
        "whole_mi": float(np.mean(macro_information["whole_to_future"])),
        "part_a_to_whole_future_mi": float(
            np.mean(macro_information["part1_to_future"])
        ),
        "part_b_to_whole_future_mi": float(
            np.mean(macro_information["part2_to_future"])
        ),
        "local_mean": float(np.mean(macro.values)),
    }
    macro_mmi_components = dict(macro_components)
    macro_mmi_components["local_mean"] = float(np.mean(macro_mmi))
    macro_redundancy = (
        "part_a_to_whole_future"
        if macro.model.redundancy_source == 0
        else "part_b_to_whole_future"
    )

    data, active, first, second = _px_transform(raw, beta)
    macro_data = np.vstack(
        (data[first].mean(axis=0), data[second].mean(axis=0))
    )
    atoms = local_phi_id_atoms(macro_data[:, :-1], macro_data[:, 1:])
    public_values = np.sum([atoms[atom] for atom in PHIR_ATOMS], axis=0)
    public_components = {
        f"atom_{_atom_name(atom)}": float(np.mean(atoms[atom]))
        for atom in ALL_ATOMS
    }
    public_components.update(
        {
            "global_revised": float(np.mean(public_values)),
            "local_mean": float(np.mean(public_values)),
            "whole_mi": gaussian_mutual_information(
                macro_data[:, :-1], macro_data[:, 1:]
            ),
        }
    )
    full_values, full_components, full_redundancy = full_block_local_scores(
        data, first, second
    )
    time_indices = np.arange(1, raw.shape[0], dtype=np.int64)
    px_a = active[first]
    px_b = active[second]
    estimators = {
        "macro_wms": BridgeEstimatorTrajectory(
            name="macro_wms",
            values=np.asarray(macro.values, dtype=np.float64),
            time_indices=np.asarray(macro.time_indices, dtype=np.int64),
            partition_a=macro_a,
            partition_b=macro_b,
            active_dimensions=int(macro.partition.size),
            components=macro_components,
        ),
        "macro_mmi": BridgeEstimatorTrajectory(
            name="macro_mmi",
            values=np.asarray(macro_mmi, dtype=np.float64),
            time_indices=np.asarray(macro.time_indices, dtype=np.int64),
            partition_a=macro_a,
            partition_b=macro_b,
            active_dimensions=int(macro.partition.size),
            components=macro_mmi_components,
            redundancy_channel=macro_redundancy,
        ),
        "public_nine_atom": BridgeEstimatorTrajectory(
            name="public_nine_atom",
            values=np.asarray(public_values, dtype=np.float64),
            time_indices=time_indices,
            partition_a=np.asarray(px_a, dtype=np.int64),
            partition_b=np.asarray(px_b, dtype=np.int64),
            active_dimensions=int(active.size),
            components=public_components,
        ),
        "full_revised": BridgeEstimatorTrajectory(
            name="full_revised",
            values=np.asarray(full_values, dtype=np.float64),
            time_indices=time_indices,
            partition_a=np.asarray(px_a, dtype=np.int64),
            partition_b=np.asarray(px_b, dtype=np.int64),
            active_dimensions=int(active.size),
            components=full_components,
            redundancy_channel=full_redundancy,
        ),
    }
    if tuple(estimators) != ESTIMATOR_ORDER:
        raise AssertionError("bridge estimator order drifted")
    for result in estimators.values():
        result.validate()
        np.testing.assert_array_equal(result.time_indices, time_indices)
    np.testing.assert_array_equal(
        estimators["public_nine_atom"].partition_a,
        estimators["full_revised"].partition_a,
    )
    np.testing.assert_array_equal(
        estimators["public_nine_atom"].partition_b,
        estimators["full_revised"].partition_b,
    )
    return estimators


__all__ = [
    "ALL_ATOMS",
    "ATOM_NAMES",
    "ESTIMATOR_ORDER",
    "PHIR_ATOMS",
    "BridgeEstimatorTrajectory",
    "beta_physical_partition",
    "close_all_clr",
    "fit_bridge_estimators",
    "full_block_local_scores",
    "gaussian_mutual_information",
    "local_phi_id_atoms",
    "rank_gaussianize",
]
