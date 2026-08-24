"""Permutation-equivariant threshold-component partition branch for E01 S11R."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import adjusted_rand_score

AFFINITY_ID = "E01-S11R-PARTITION-AFFINITY-LAG-ALIGNED-ABS-CORR-v1.0.0"
SEARCH_ID = "E01-S11R-PARTITION-THRESHOLD-COMPONENTS-v1.0.0"
EDGE_THRESHOLD = 0.90
TIE_TOLERANCE = 1.0e-12


class RepairPartitionError(ValueError):
    """A partition request violates the frozen S11R contract."""


@dataclass(frozen=True, slots=True)
class ThresholdPartitionResult:
    """A status-bearing unordered component split and stability diagnostics."""

    status: str
    reason: str | None
    dimension: int
    tau: int
    selected_part_a: tuple[int, ...] | None
    bootstrap_parts: tuple[tuple[int, ...], ...]
    diagnostics: dict[str, Any]


def canonical_part(
    part_a: tuple[int, ...] | list[int], dimension: int
) -> tuple[int, ...]:
    """Canonicalize only for serialization; selection never uses this ordering."""

    selected = tuple(sorted({int(value) for value in part_a}))
    if (
        not selected
        or len(selected) == dimension
        or selected[0] < 0
        or selected[-1] >= dimension
    ):
        raise RepairPartitionError("Partition must be nontrivial and in range.")
    if 0 in selected:
        return selected
    selected_set = set(selected)
    return tuple(index for index in range(dimension) if index not in selected_set)


def labels_from_part(part_a: tuple[int, ...], dimension: int) -> np.ndarray:
    labels = np.ones(dimension, dtype=np.int8)
    labels[list(part_a)] = 0
    return labels


def partition_ari(
    first: tuple[int, ...], second: tuple[int, ...], dimension: int
) -> float:
    return float(
        adjusted_rand_score(
            labels_from_part(first, dimension), labels_from_part(second, dimension)
        )
    )


def _standardize(data: np.ndarray) -> np.ndarray:
    values = np.asarray(data, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 2:
        raise RepairPartitionError("Partition data must be observations by D>=2.")
    if not np.all(np.isfinite(values)):
        raise RepairPartitionError("NONFINITE_PARTITION_INPUT_NO_ROW_DELETION")
    sd = np.std(values, axis=0, ddof=1)
    if np.any(~np.isfinite(sd)) or np.any(sd <= 1.0e-12):
        raise RepairPartitionError("PARTITION_COMPONENT_CONSTANT")
    return (values - np.mean(values, axis=0, keepdims=True)) / sd


def _weighted_self_corr(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    if (
        weights.ndim != 1
        or weights.shape[0] != values.shape[0]
        or np.any(~np.isfinite(weights))
        or np.any(weights <= 0)
    ):
        raise RepairPartitionError("Bayesian weights must be finite and positive.")
    weights = weights / np.sum(weights)
    centered = values - np.sum(weights[:, None] * values, axis=0, keepdims=True)
    covariance = (centered * weights[:, None]).T @ centered
    sd = np.sqrt(np.diag(covariance))
    if np.any(sd <= 1.0e-14) or np.any(~np.isfinite(sd)):
        raise RepairPartitionError("PARTITION_COMPONENT_CONSTANT")
    correlation = covariance / (sd[:, None] * sd[None, :])
    correlation = np.clip(correlation, -1.0, 1.0)
    return correlation


def lagged_affinity(
    data: np.ndarray, *, tau: int, weights: np.ndarray | None = None
) -> np.ndarray:
    """Mean absolute past/future Pearson affinity at the exact lag alignment."""

    standardized = _standardize(data)
    if (
        not isinstance(tau, int)
        or isinstance(tau, bool)
        or tau < 1
        or tau >= len(standardized)
    ):
        raise RepairPartitionError("INVALID_LAG")
    past = standardized[:-tau]
    future = standardized[tau:]
    if weights is None:
        weights = np.ones(past.shape[0], dtype=np.float64)
    affinity = 0.5 * (
        np.abs(_weighted_self_corr(past, weights))
        + np.abs(_weighted_self_corr(future, weights))
    )
    affinity = 0.5 * (affinity + affinity.T)
    np.fill_diagonal(affinity, 0.0)
    if not np.all(np.isfinite(affinity)):
        raise RepairPartitionError("NONFINITE_PARTITION_AFFINITY")
    return affinity


def _components(affinity: np.ndarray) -> tuple[tuple[int, ...], ...]:
    upper = affinity[np.triu_indices(affinity.shape[0], 1)]
    if np.any(np.abs(upper - EDGE_THRESHOLD) <= TIE_TOLERANCE):
        raise RepairPartitionError("THRESHOLD_EDGE_TIE")
    adjacency = affinity > EDGE_THRESHOLD
    np.fill_diagonal(adjacency, False)
    unseen = set(range(affinity.shape[0]))
    components: list[tuple[int, ...]] = []
    while unseen:
        seed = next(iter(unseen))
        stack = [seed]
        component: set[int] = set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            unseen.discard(node)
            stack.extend(
                int(item)
                for item in np.flatnonzero(adjacency[node])
                if item not in component
            )
        components.append(tuple(sorted(component)))
    return tuple(components)


def _contrast(
    affinity: np.ndarray, part_a: tuple[int, ...]
) -> tuple[float, float, float]:
    selected = set(part_a)
    part_b = tuple(index for index in range(affinity.shape[0]) if index not in selected)
    within: list[np.ndarray] = []
    for part in (part_a, part_b):
        block = affinity[np.ix_(part, part)]
        values = block[np.triu_indices(len(part), 1)]
        if values.size:
            within.append(values)
    if not within:
        raise RepairPartitionError("SINGLETON_ONLY_WITHIN_AFFINITY")
    minimum_within = float(np.min(np.concatenate(within)))
    maximum_between = float(np.max(affinity[np.ix_(part_a, part_b)]))
    return minimum_within, maximum_between, minimum_within - maximum_between


def threshold_component_partition(
    data: np.ndarray,
    *,
    tau: int,
    rng: np.random.Generator,
    bootstrap_replicates: int = 8,
) -> ThresholdPartitionResult:
    """Return the frozen two-component split or fail closed with a reason."""

    values = np.asarray(data)
    dimension = int(values.shape[1]) if values.ndim == 2 else 0
    bootstrap: list[tuple[int, ...]] = []
    try:
        affinity = lagged_affinity(values, tau=tau)
        components = _components(affinity)
        if len(components) != 2:
            raise RepairPartitionError(
                f"THRESHOLD_GRAPH_COMPONENT_COUNT_{len(components)}"
            )
        base = canonical_part(components[0], dimension)
        minimum_within, maximum_between, contrast = _contrast(affinity, base)
        effective = values.shape[0] - tau
        for _ in range(bootstrap_replicates):
            weighted_affinity = lagged_affinity(
                values, tau=tau, weights=rng.exponential(scale=1.0, size=effective)
            )
            weighted_components = _components(weighted_affinity)
            if len(weighted_components) != 2:
                raise RepairPartitionError(
                    f"BOOTSTRAP_THRESHOLD_GRAPH_COMPONENT_COUNT_{len(weighted_components)}"
                )
            bootstrap.append(canonical_part(weighted_components[0], dimension))
        aris = [partition_ari(base, item, dimension) for item in bootstrap]
        mean_bootstrap_ari = float(np.mean(aris)) if aris else 1.0
        minimum_fraction = min(len(base), dimension - len(base)) / dimension
        failures: list[str] = []
        if minimum_fraction < 0.10:
            failures.append("MINIMUM_PART_FRACTION_GATE_FAILED")
        if mean_bootstrap_ari < 0.75:
            failures.append("BOOTSTRAP_ARI_GATE_FAILED")
        if minimum_within < 0.90:
            failures.append("MINIMUM_WITHIN_AFFINITY_GATE_FAILED")
        if contrast < 0.10:
            failures.append("AFFINITY_CONTRAST_GATE_FAILED")
        diagnostics = {
            "affinityId": AFFINITY_ID,
            "searchId": SEARCH_ID,
            "edgeThreshold": EDGE_THRESHOLD,
            "thresholdTieTolerance": TIE_TOLERANCE,
            "componentCount": len(components),
            "bootstrapReplicates": bootstrap_replicates,
            "meanBootstrapAri": mean_bootstrap_ari,
            "minimumPartFraction": minimum_fraction,
            "minimumWithinAffinity": minimum_within,
            "maximumBetweenAffinity": maximum_between,
            "withinMinusMaximumBetweenAffinity": contrast,
            "permutationEquivariantByConstruction": True,
            "indexUsedForSelectionOrTieBreak": False,
        }
        if failures:
            return ThresholdPartitionResult(
                "INELIGIBLE",
                ";".join(failures),
                dimension,
                tau,
                None,
                tuple(bootstrap),
                diagnostics,
            )
        return ThresholdPartitionResult(
            "ELIGIBLE", None, dimension, tau, base, tuple(bootstrap), diagnostics
        )
    except (RepairPartitionError, np.linalg.LinAlgError) as error:
        return ThresholdPartitionResult(
            "INELIGIBLE", str(error), dimension, tau, None, tuple(bootstrap), {}
        )
