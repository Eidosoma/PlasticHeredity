"""Fail-closed high-dimensional partition search for E01 S11."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from sklearn.metrics import adjusted_rand_score

from e01_information_dynamics.validation import (
    gaussian_entropy,
    gaussian_mutual_information,
)

from .estimator import oas_covariance

AFFINITY_ID = "E01-S11-PARTITION-AFFINITY-ABS-CORR-LAG-v1.0.0"
SEARCH_ID = "E01-S11-MIB-SEARCH-STABILITY-SPECTRAL-SINGLE-FLIP-v1.0.0"
GROUP_MEAN_ID = "E01-S11-PARTMAP-ZSCORE-GROUP-MEAN-v1.0.0"
PC1_ID = "E01-S11-PARTMAP-ZSCORE-PC1-v1.0.0"


class PartitionError(ValueError):
    """A partition request violates the frozen S11 contract."""


@dataclass(frozen=True, slots=True)
class StablePartitionResult:
    """A candidate set and stability diagnostics, possibly ineligible."""

    status: str
    reason: str | None
    dimension: int
    tau: int
    selected_part_a: tuple[int, ...] | None
    candidate_parts: tuple[tuple[int, ...], ...]
    bootstrap_parts: tuple[tuple[int, ...], ...]
    diagnostics: dict[str, Any]


def canonical_part(part_a: tuple[int, ...] | list[int], dimension: int) -> tuple[int, ...]:
    """Canonicalize an unordered bipartition by placing component zero in A."""

    selected = tuple(sorted({int(value) for value in part_a}))
    if not selected or len(selected) == dimension or min(selected) < 0 or max(selected) >= dimension:
        raise PartitionError("Partition must be nontrivial and within the component range.")
    if 0 in selected:
        return selected
    selected_set = set(selected)
    return tuple(index for index in range(dimension) if index not in selected_set)


def complement(part_a: tuple[int, ...], dimension: int) -> tuple[int, ...]:
    selected = set(part_a)
    return tuple(index for index in range(dimension) if index not in selected)


def labels_from_part(part_a: tuple[int, ...], dimension: int) -> np.ndarray:
    labels = np.ones(dimension, dtype=np.int8)
    labels[list(part_a)] = 0
    return labels


def partition_ari(first: tuple[int, ...], second: tuple[int, ...], dimension: int) -> float:
    return float(
        adjusted_rand_score(labels_from_part(first, dimension), labels_from_part(second, dimension))
    )


def _standardize(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] < 2:
        raise PartitionError("Partition data must have shape observations by D>=2.")
    if not np.all(np.isfinite(data)):
        raise PartitionError("NONFINITE_PARTITION_INPUT_NO_ROW_DELETION")
    sd = np.std(data, axis=0, ddof=1)
    if np.any(~np.isfinite(sd)) or np.any(sd <= 1.0e-12):
        raise PartitionError("PARTITION_COMPONENT_CONSTANT")
    return (data - np.mean(data, axis=0, keepdims=True)) / sd, sd


def _weighted_corr(first: np.ndarray, second: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1 or weights.shape[0] != first.shape[0] or np.any(weights <= 0):
        raise PartitionError("Bayesian bootstrap weights must be finite and strictly positive.")
    weights = weights / np.sum(weights)
    first_centered = first - np.sum(weights[:, None] * first, axis=0, keepdims=True)
    second_centered = second - np.sum(weights[:, None] * second, axis=0, keepdims=True)
    covariance = (first_centered * weights[:, None]).T @ second_centered
    first_sd = np.sqrt(np.sum(weights[:, None] * first_centered**2, axis=0))
    second_sd = np.sqrt(np.sum(weights[:, None] * second_centered**2, axis=0))
    denominator = first_sd[:, None] * second_sd[None, :]
    if np.any(denominator <= 1.0e-14):
        raise PartitionError("PARTITION_COMPONENT_CONSTANT")
    return covariance / denominator


def lagged_affinity(
    data: np.ndarray,
    *,
    tau: int,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Construct the frozen absolute contemporaneous/directed-lag affinity."""

    standardized, _ = _standardize(data)
    if not isinstance(tau, int) or isinstance(tau, bool) or tau < 1 or tau >= standardized.shape[0]:
        raise PartitionError("INVALID_LAG")
    past = standardized[:-tau]
    future = standardized[tau:]
    if weights is None:
        weights = np.ones(past.shape[0], dtype=np.float64)
    corr_past = _weighted_corr(past, past, weights)
    corr_future = _weighted_corr(future, future, weights)
    corr_lag = _weighted_corr(past, future, weights)
    contemporaneous = 0.5 * (np.abs(corr_past) + np.abs(corr_future))
    affinity = (contemporaneous + np.abs(corr_lag) + np.abs(corr_lag.T)) / 3.0
    affinity = 0.5 * (affinity + affinity.T)
    np.fill_diagonal(affinity, 0.0)
    if not np.all(np.isfinite(affinity)) or np.any(affinity < 0):
        raise PartitionError("NONFINITE_PARTITION_AFFINITY")
    return affinity


def spectral_split(affinity: np.ndarray) -> tuple[tuple[int, ...], dict[str, float]]:
    """Split a nonnegative affinity by the normalized-Laplacian Fiedler vector."""

    affinity = np.asarray(affinity, dtype=np.float64)
    if affinity.ndim != 2 or affinity.shape[0] != affinity.shape[1] or affinity.shape[0] < 2:
        raise PartitionError("Affinity must be a square D>=2 matrix.")
    if not np.all(np.isfinite(affinity)) or np.any(affinity < 0):
        raise PartitionError("Affinity must be finite and nonnegative.")
    dimension = affinity.shape[0]
    degree = np.sum(affinity, axis=1)
    if np.any(degree <= np.finfo(np.float64).tiny):
        raise PartitionError("ZERO_AFFINITY_DEGREE")
    inverse = 1.0 / np.sqrt(degree)
    laplacian = np.eye(dimension) - inverse[:, None] * affinity * inverse[None, :]
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    vector = eigenvectors[:, 1]
    negative = tuple(int(index) for index in np.flatnonzero(vector < 0.0))
    if not negative or len(negative) == dimension:
        order = np.argsort(vector, kind="stable")
        negative = tuple(int(index) for index in order[: dimension // 2])
    part = canonical_part(negative, dimension)
    third = float(eigenvalues[2]) if dimension > 2 else float(eigenvalues[-1])
    second = float(eigenvalues[1])
    relative_gap = float((third - second) / max(abs(third), np.finfo(float).tiny))
    return part, {
        "lambda1": float(eigenvalues[0]),
        "lambda2": second,
        "lambda3": third,
        "relativeFiedlerEigengap": relative_gap,
    }


def _coassignment(parts: tuple[tuple[int, ...], ...], dimension: int) -> np.ndarray:
    coassignment = np.zeros((dimension, dimension), dtype=np.float64)
    for part in parts:
        labels = labels_from_part(part, dimension)
        coassignment += labels[:, None] == labels[None, :]
    coassignment /= len(parts)
    np.fill_diagonal(coassignment, 0.0)
    return coassignment


def _aligned_confidence(
    consensus: tuple[int, ...], parts: tuple[tuple[int, ...], ...], dimension: int
) -> float:
    reference = labels_from_part(consensus, dimension)
    aligned: list[np.ndarray] = []
    for part in parts:
        labels = labels_from_part(part, dimension)
        if np.mean(labels == reference) < np.mean((1 - labels) == reference):
            labels = 1 - labels
        aligned.append(labels)
    probabilities = np.mean(np.stack(aligned), axis=0)
    return float(np.mean(np.maximum(probabilities, 1.0 - probabilities)))


def _affinity_contrast(affinity: np.ndarray, part_a: tuple[int, ...]) -> float:
    dimension = affinity.shape[0]
    part_b = complement(part_a, dimension)
    within_a = affinity[np.ix_(part_a, part_a)]
    within_b = affinity[np.ix_(part_b, part_b)]
    within_values = np.concatenate(
        [within_a[np.triu_indices(len(part_a), 1)], within_b[np.triu_indices(len(part_b), 1)]]
    )
    between = affinity[np.ix_(part_a, part_b)].ravel()
    if within_values.size == 0 or between.size == 0:
        return -math.inf
    return float(np.mean(within_values) - np.mean(between))


def _candidate_set(
    consensus: tuple[int, ...],
    base: tuple[int, ...],
    bootstrap_parts: tuple[tuple[int, ...], ...],
    dimension: int,
) -> tuple[tuple[int, ...], ...]:
    candidates = {canonical_part(part, dimension) for part in (base, consensus, *bootstrap_parts)}
    consensus_set = set(consensus)
    for component in range(dimension):
        changed = set(consensus_set)
        if component in changed:
            changed.remove(component)
        else:
            changed.add(component)
        if 0 < len(changed) < dimension:
            candidates.add(canonical_part(tuple(changed), dimension))
    return tuple(sorted(candidates))


def stable_partition_candidates(
    data: np.ndarray,
    *,
    tau: int,
    rng: np.random.Generator,
    bootstrap_replicates: int = 8,
) -> StablePartitionResult:
    """Generate a stable candidate set and fail closed when structure is absent."""

    dimension = int(np.asarray(data).shape[1]) if np.asarray(data).ndim == 2 else 0
    try:
        base_affinity = lagged_affinity(data, tau=tau)
        base, base_spectral = spectral_split(base_affinity)
        bootstrap: list[tuple[int, ...]] = []
        effective = np.asarray(data).shape[0] - tau
        for _ in range(bootstrap_replicates):
            weights = rng.exponential(scale=1.0, size=effective)
            affinity = lagged_affinity(data, tau=tau, weights=weights)
            part, _ = spectral_split(affinity)
            bootstrap.append(part)
        bootstrap_parts = tuple(bootstrap)
        consensus_affinity = _coassignment(bootstrap_parts, dimension)
        consensus, consensus_spectral = spectral_split(consensus_affinity)
        pairwise = [
            partition_ari(bootstrap_parts[first], bootstrap_parts[second], dimension)
            for first in range(len(bootstrap_parts))
            for second in range(first + 1, len(bootstrap_parts))
        ]
        mean_pairwise_ari = float(np.mean(pairwise)) if pairwise else 1.0
        confidence = _aligned_confidence(consensus, bootstrap_parts, dimension)
        contrast = _affinity_contrast(base_affinity, consensus)
        minimum_fraction = min(len(consensus), dimension - len(consensus)) / dimension
        diagnostics = {
            "affinityId": AFFINITY_ID,
            "searchId": SEARCH_ID,
            "bootstrapReplicates": bootstrap_replicates,
            "meanPairwiseBootstrapAri": mean_pairwise_ari,
            "consensusConfidence": confidence,
            "withinMinusBetweenAffinity": contrast,
            "minimumPartFraction": minimum_fraction,
            "basePartA": list(base),
            "consensusPartA": list(consensus),
            "baseSpectral": base_spectral,
            "consensusSpectral": consensus_spectral,
        }
        failures: list[str] = []
        if minimum_fraction < 0.10:
            failures.append("MINIMUM_PART_FRACTION_GATE_FAILED")
        if mean_pairwise_ari < 0.75:
            failures.append("BOOTSTRAP_ARI_GATE_FAILED")
        if confidence < 0.85:
            failures.append("CONSENSUS_CONFIDENCE_GATE_FAILED")
        if contrast < 0.20:
            failures.append("AFFINITY_CONTRAST_GATE_FAILED")
        if base_spectral["relativeFiedlerEigengap"] < 0.05:
            failures.append("FIEDLER_EIGENGAP_GATE_FAILED")
        candidates = _candidate_set(consensus, base, bootstrap_parts, dimension)
        if failures:
            return StablePartitionResult(
                "INELIGIBLE",
                ";".join(failures),
                dimension,
                tau,
                None,
                candidates,
                bootstrap_parts,
                diagnostics,
            )
        return StablePartitionResult(
            "ELIGIBLE",
            None,
            dimension,
            tau,
            consensus,
            candidates,
            bootstrap_parts,
            diagnostics,
        )
    except (PartitionError, np.linalg.LinAlgError) as error:
        return StablePartitionResult(
            "INELIGIBLE", str(error), dimension, tau, None, (), (), {}
        )


def map_partition(
    data: np.ndarray,
    part_a: tuple[int, ...],
    *,
    mapping: Literal["zscore_group_mean", "zscore_pc1"],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Map a component split to two scalar series using a named S11 branch."""

    standardized, _ = _standardize(data)
    dimension = standardized.shape[1]
    part_a = canonical_part(part_a, dimension)
    part_b = complement(part_a, dimension)

    def pc1(block: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        _, singular_values, right = np.linalg.svd(block, full_matrices=False)
        if singular_values.size < 2:
            relative_gap = math.inf
        else:
            relative_gap = float(
                (singular_values[0] ** 2 - singular_values[1] ** 2)
                / max(singular_values[0] ** 2, np.finfo(float).tiny)
            )
        if relative_gap <= 1.0e-8:
            raise PartitionError("PC1_LEADING_EIGENGAP_GATE_FAILED")
        loading = right[0].copy()
        maximum = np.max(np.abs(loading))
        pivot = int(np.flatnonzero(np.abs(loading) == maximum)[0])
        if loading[pivot] < 0:
            loading = -loading
        return block @ loading, {
            "relativeEigengap": relative_gap,
            "leadingSingularValue": float(singular_values[0]),
            "signPivotWithinPart": pivot,
        }

    block_a = standardized[:, part_a]
    block_b = standardized[:, part_b]
    if mapping == "zscore_group_mean":
        return np.mean(block_a, axis=1), np.mean(block_b, axis=1), {
            "mappingId": GROUP_MEAN_ID
        }
    if mapping == "zscore_pc1":
        score_a, diagnostics_a = pc1(block_a)
        score_b, diagnostics_b = pc1(block_b)
        return score_a, score_b, {
            "mappingId": PC1_ID,
            "partA": diagnostics_a,
            "partB": diagnostics_b,
        }
    raise PartitionError(f"Unknown mapping {mapping!r}.")


def _regularized_covariance(blocks: list[np.ndarray]) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    arrays: list[np.ndarray] = []
    groups: list[tuple[int, ...]] = []
    offset = 0
    for block in blocks:
        block = np.asarray(block, dtype=np.float64)
        if block.ndim == 1:
            block = block[:, None]
        arrays.append(block)
        groups.append(tuple(range(offset, offset + block.shape[1])))
        offset += block.shape[1]
    matrix = np.concatenate(arrays, axis=1)
    matrix, _ = _standardize(matrix)
    covariance, _ = oas_covariance(matrix, backend="numpy", shrinkage_multiplier=1.0)
    return np.asarray(covariance, dtype=np.float64), groups


def partition_objective(
    data: np.ndarray,
    part_a: tuple[int, ...],
    *,
    tau: int,
    mapping: Literal["zscore_group_mean", "zscore_pc1"],
    objective: Literal["synchronous_mi", "bidirectional_lagged_mi", "abs_paper_equation"],
    normalization: Literal["none", "min_part_entropy", "geometric_part_size"],
) -> dict[str, Any]:
    """Score one explicit partition candidate with regularized Gaussian moments."""

    dimension = int(np.asarray(data).shape[1])
    try:
        part_a = canonical_part(part_a, dimension)
        part_b = complement(part_a, dimension)
        first, second, mapping_diagnostics = map_partition(data, part_a, mapping=mapping)
        if objective == "synchronous_mi":
            covariance, groups = _regularized_covariance([first, second])
            raw = gaussian_mutual_information(covariance, groups[0], groups[1])
        elif objective in ("bidirectional_lagged_mi", "abs_paper_equation"):
            if tau < 1 or tau >= first.size:
                raise PartitionError("INVALID_LAG")
            covariance, groups = _regularized_covariance(
                [first[:-tau], second[:-tau], first[tau:], second[tau:]]
            )
            if objective == "bidirectional_lagged_mi":
                raw = gaussian_mutual_information(covariance, groups[0], groups[3])
                raw += gaussian_mutual_information(covariance, groups[1], groups[2])
            else:
                past = groups[0] + groups[1]
                future = groups[2] + groups[3]
                total = gaussian_mutual_information(covariance, past, future)
                first_source = gaussian_mutual_information(covariance, groups[0], future)
                second_source = gaussian_mutual_information(covariance, groups[1], future)
                raw = abs(total - first_source - second_source)
        else:
            raise PartitionError(f"Unknown objective {objective!r}.")
        if normalization == "none":
            denominator = 1.0
        elif normalization == "geometric_part_size":
            denominator = math.sqrt(len(part_a) * len(part_b))
        elif normalization == "min_part_entropy":
            covariance, groups = _regularized_covariance([first, second])
            denominator = min(
                gaussian_entropy(covariance, groups[0]),
                gaussian_entropy(covariance, groups[1]),
            )
            if denominator <= 0 or not np.isfinite(denominator):
                raise PartitionError("NONPOSITIVE_PART_ENTROPY_NORMALIZATION")
        else:
            raise PartitionError(f"Unknown normalization {normalization!r}.")
        normalized = float(raw / denominator)
        if not np.isfinite(normalized):
            raise PartitionError("NONFINITE_PARTITION_OBJECTIVE")
        return {
            "status": "ELIGIBLE",
            "reason": None,
            "partA": list(part_a),
            "partB": list(part_b),
            "mapping": mapping,
            "objective": objective,
            "normalization": normalization,
            "rawObjective": float(raw),
            "normalizationDenominator": float(denominator),
            "normalizedObjective": normalized,
            "mappingDiagnostics": mapping_diagnostics,
        }
    except (PartitionError, np.linalg.LinAlgError) as error:
        return {
            "status": "INELIGIBLE",
            "reason": str(error),
            "partA": list(part_a),
            "partB": list(complement(canonical_part(part_a, dimension), dimension)),
            "mapping": mapping,
            "objective": objective,
            "normalization": normalization,
            "rawObjective": None,
            "normalizationDenominator": None,
            "normalizedObjective": None,
            "mappingDiagnostics": {},
        }


def select_partition_candidate(
    data: np.ndarray,
    stable: StablePartitionResult,
    *,
    tau: int,
    mapping: Literal["zscore_group_mean", "zscore_pc1"],
    objective: Literal["synchronous_mi", "bidirectional_lagged_mi", "abs_paper_equation"],
    normalization: Literal["none", "min_part_entropy", "geometric_part_size"],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate the frozen approximation candidate set and select its minimum."""

    if stable.status != "ELIGIBLE":
        return {
            "status": "INELIGIBLE",
            "reason": stable.reason,
            "partA": None,
            "partB": None,
            "normalizedObjective": None,
        }, []
    scores = [
        partition_objective(
            data,
            part,
            tau=tau,
            mapping=mapping,
            objective=objective,
            normalization=normalization,
        )
        for part in stable.candidate_parts
    ]
    eligible = [score for score in scores if score["status"] == "ELIGIBLE"]
    if len(eligible) != len(scores):
        return {
            "status": "INELIGIBLE",
            "reason": "ONE_OR_MORE_CANDIDATE_SCORES_INELIGIBLE",
            "partA": None,
            "partB": None,
            "normalizedObjective": None,
        }, scores
    winner = min(
        eligible,
        key=lambda item: (item["normalizedObjective"], tuple(item["partA"])),
    )
    return winner, scores


def evaluate_candidate_grid(
    data: np.ndarray,
    stable: StablePartitionResult,
    *,
    tau: int,
    mappings: tuple[str, ...] = ("zscore_group_mean", "zscore_pc1"),
    objectives: tuple[str, ...] = (
        "synchronous_mi",
        "bidirectional_lagged_mi",
        "abs_paper_equation",
    ),
    normalizations: tuple[str, ...] = (
        "none",
        "min_part_entropy",
        "geometric_part_size",
    ),
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str], dict[str, Any]]]:
    """Score the complete named grid while caching each mapped candidate.

    This function is algebraically identical to repeated ``partition_objective``
    calls, but avoids recomputing PCA mappings and OAS covariance matrices for
    every normalization branch.
    """

    if stable.status != "ELIGIBLE":
        return [], {
            (mapping, objective, normalization): {
                "status": "INELIGIBLE",
                "reason": stable.reason,
                "partA": None,
                "partB": None,
                "normalizedObjective": None,
            }
            for mapping in mappings
            for objective in objectives
            for normalization in normalizations
        }
    dimension = np.asarray(data).shape[1]
    rows: list[dict[str, Any]] = []
    for part_a in stable.candidate_parts:
        part_b = complement(part_a, dimension)
        for mapping in mappings:
            try:
                first, second, mapping_diagnostics = map_partition(
                    data, part_a, mapping=mapping  # type: ignore[arg-type]
                )
                synchronous_covariance, synchronous_groups = _regularized_covariance(
                    [first, second]
                )
                synchronous = gaussian_mutual_information(
                    synchronous_covariance,
                    synchronous_groups[0],
                    synchronous_groups[1],
                )
                entropy_denominator = min(
                    gaussian_entropy(synchronous_covariance, synchronous_groups[0]),
                    gaussian_entropy(synchronous_covariance, synchronous_groups[1]),
                )
                lag_covariance, lag_groups = _regularized_covariance(
                    [first[:-tau], second[:-tau], first[tau:], second[tau:]]
                )
                bidirectional = gaussian_mutual_information(
                    lag_covariance, lag_groups[0], lag_groups[3]
                ) + gaussian_mutual_information(
                    lag_covariance, lag_groups[1], lag_groups[2]
                )
                past = lag_groups[0] + lag_groups[1]
                future = lag_groups[2] + lag_groups[3]
                total = gaussian_mutual_information(lag_covariance, past, future)
                source_first = gaussian_mutual_information(
                    lag_covariance, lag_groups[0], future
                )
                source_second = gaussian_mutual_information(
                    lag_covariance, lag_groups[1], future
                )
                raw_by_objective = {
                    "synchronous_mi": synchronous,
                    "bidirectional_lagged_mi": bidirectional,
                    "abs_paper_equation": abs(total - source_first - source_second),
                }
                denominator_by_normalization = {
                    "none": 1.0,
                    "min_part_entropy": entropy_denominator,
                    "geometric_part_size": math.sqrt(len(part_a) * len(part_b)),
                }
                if entropy_denominator <= 0 or not np.isfinite(entropy_denominator):
                    raise PartitionError("NONPOSITIVE_PART_ENTROPY_NORMALIZATION")
                for objective in objectives:
                    for normalization in normalizations:
                        raw = float(raw_by_objective[objective])
                        denominator = float(denominator_by_normalization[normalization])
                        normalized = raw / denominator
                        if not np.isfinite(normalized):
                            raise PartitionError("NONFINITE_PARTITION_OBJECTIVE")
                        rows.append(
                            {
                                "status": "ELIGIBLE",
                                "reason": None,
                                "partA": list(part_a),
                                "partB": list(part_b),
                                "mapping": mapping,
                                "objective": objective,
                                "normalization": normalization,
                                "rawObjective": raw,
                                "normalizationDenominator": denominator,
                                "normalizedObjective": float(normalized),
                                "mappingDiagnostics": mapping_diagnostics,
                            }
                        )
            except (PartitionError, np.linalg.LinAlgError) as error:
                for objective in objectives:
                    for normalization in normalizations:
                        rows.append(
                            {
                                "status": "INELIGIBLE",
                                "reason": str(error),
                                "partA": list(part_a),
                                "partB": list(part_b),
                                "mapping": mapping,
                                "objective": objective,
                                "normalization": normalization,
                                "rawObjective": None,
                                "normalizationDenominator": None,
                                "normalizedObjective": None,
                                "mappingDiagnostics": {},
                            }
                        )
    winners: dict[tuple[str, str, str], dict[str, Any]] = {}
    for mapping in mappings:
        for objective in objectives:
            for normalization in normalizations:
                key = (mapping, objective, normalization)
                selected = [
                    row
                    for row in rows
                    if row["mapping"] == mapping
                    and row["objective"] == objective
                    and row["normalization"] == normalization
                    and row["status"] == "ELIGIBLE"
                ]
                if len(selected) != len(stable.candidate_parts):
                    winners[key] = {
                        "status": "INELIGIBLE",
                        "reason": "ONE_OR_MORE_CANDIDATE_SCORES_INELIGIBLE",
                        "partA": None,
                        "partB": None,
                        "normalizedObjective": None,
                    }
                else:
                    winners[key] = min(
                        selected,
                        key=lambda item: (
                            item["normalizedObjective"],
                            tuple(item["partA"]),
                        ),
                    )
    return rows, winners
