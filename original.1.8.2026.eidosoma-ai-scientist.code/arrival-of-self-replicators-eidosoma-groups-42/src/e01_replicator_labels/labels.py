"""Explicit, fail-closed label families for E01 S08.

Only the historical H/non-drift branch delegates to source-traceable S04
behavior.  The graph cluster branches are deliberately named reconstruction
configurations and must not be interpreted as recovered author settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from e01_gard_historical import (
    HistoricalReferenceError,
    HistoricalSourceDomainError,
    historical_h,
    historical_nondrift_technique1,
    historical_nondrift_technique2,
)

MetricName = Literal["cosine", "euclidean", "aitchison"]
TemporalScope = Literal["paper_retrospective_full_trace", "past_only_online"]


class LabelContractError(ValueError):
    """Input or configuration violates the frozen S08 label contract."""


@dataclass(frozen=True, slots=True)
class ClusterConfiguration:
    """A complete graph-clustering configuration with no model defaults."""

    configuration_id: str
    family_id: str
    family_name: str
    evidence_class: str
    metric: MetricName
    representation: str
    threshold: float
    comparator: str
    minimum_cluster_size: int
    temporal_scope: TemporalScope
    zero_policy: str

    def __post_init__(self) -> None:
        for name in (
            "configuration_id",
            "family_id",
            "family_name",
            "evidence_class",
            "representation",
            "comparator",
            "zero_policy",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise LabelContractError(f"{name} must be an explicit nonempty string.")
        if self.metric not in ("cosine", "euclidean", "aitchison"):
            raise LabelContractError(f"Unsupported metric {self.metric!r}.")
        expected = (
            "strict_greater_than" if self.metric == "cosine" else "strict_less_than"
        )
        if self.comparator != expected:
            raise LabelContractError(
                f"{self.metric} requires comparator={expected!r}, got {self.comparator!r}."
            )
        if not np.isfinite(self.threshold) or self.threshold <= 0:
            raise LabelContractError("threshold must be finite and positive.")
        if self.metric == "cosine" and self.threshold > 1:
            raise LabelContractError("cosine threshold may not exceed one.")
        if (
            not isinstance(self.minimum_cluster_size, int)
            or isinstance(self.minimum_cluster_size, bool)
            or self.minimum_cluster_size < 2
        ):
            raise LabelContractError("minimum_cluster_size must be an integer >= 2.")
        if self.temporal_scope not in (
            "paper_retrospective_full_trace",
            "past_only_online",
        ):
            raise LabelContractError(
                "temporal_scope must explicitly select retrospective or past-only."
            )


@dataclass(frozen=True, slots=True)
class LabelRecord:
    """One status-bearing label row; ineligible observations are never dropped."""

    trajectory_id: str
    observation_id: str
    observation_index_one_based: int
    configuration_id: str
    family_id: str
    temporal_scope: str
    evidence_class: str
    label_status: str
    is_replicator: bool | None
    cluster_id: str | None
    component_id: str | None
    reference_observation_id: str | None
    metric_to_reference: float | None
    historical_incoming_h: float | None
    historical_local_score: float | None
    ineligibility_reason: str | None
    source_padding: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "trajectoryId": self.trajectory_id,
            "observationId": self.observation_id,
            "observationIndexOneBased": self.observation_index_one_based,
            "configurationId": self.configuration_id,
            "familyId": self.family_id,
            "temporalScope": self.temporal_scope,
            "evidenceClass": self.evidence_class,
            "labelStatus": self.label_status,
            "isReplicator": self.is_replicator,
            "clusterId": self.cluster_id,
            "componentId": self.component_id,
            "referenceObservationId": self.reference_observation_id,
            "metricToReference": self.metric_to_reference,
            "historicalIncomingH": self.historical_incoming_h,
            "historicalLocalScore": self.historical_local_score,
            "ineligibilityReason": self.ineligibility_reason,
            "sourcePadding": self.source_padding,
        }


@dataclass(frozen=True, slots=True)
class MetricResult:
    """Pairwise metric with explicit eligibility for every input row."""

    metric: MetricName
    values: NDArray[np.float64]
    distances: NDArray[np.float64]
    eligible: tuple[bool, ...]
    reasons: tuple[str | None, ...]
    representation: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class LabelTraceResult:
    """Complete label output for one trajectory/configuration pair."""

    trajectory_id: str
    configuration_id: str
    family_id: str
    result_status: str
    result_reason: str | None
    rows: tuple[LabelRecord, ...]
    metric_result: MetricResult | None


def _state_matrix(values: ArrayLike) -> NDArray[np.float64]:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise LabelContractError(
            "states must be a nonempty observations-by-components matrix."
        )
    if not np.all(np.isfinite(matrix)):
        raise LabelContractError("states must contain only finite values.")
    if np.any(matrix < 0):
        raise LabelContractError("states must be nonnegative compositions.")
    return matrix


def _observation_ids(
    count: int, observation_ids: tuple[str, ...] | list[str]
) -> tuple[str, ...]:
    ids = tuple(observation_ids)
    if len(ids) != count:
        raise LabelContractError(
            "observation_ids length must equal the state row count."
        )
    if any(not isinstance(item, str) or not item for item in ids):
        raise LabelContractError("Every observation ID must be a nonempty string.")
    if len(set(ids)) != len(ids):
        raise LabelContractError("observation IDs must be unique within a trajectory.")
    return ids


def strict_similarity_adjacency(
    values: ArrayLike,
    *,
    threshold: float,
    eligible: ArrayLike,
) -> NDArray[np.bool_]:
    """Build an undirected graph using the literal strict ``>`` comparator."""

    matrix = np.asarray(values, dtype=np.float64)
    mask = np.asarray(eligible, dtype=bool)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise LabelContractError("similarity values must be a square matrix.")
    if mask.shape != (matrix.shape[0],):
        raise LabelContractError("eligible mask length mismatch.")
    if not np.isfinite(threshold):
        raise LabelContractError("threshold must be finite.")
    adjacency = matrix > float(threshold)
    adjacency &= mask[:, None] & mask[None, :]
    np.fill_diagonal(adjacency, False)
    return adjacency


def strict_distance_adjacency(
    values: ArrayLike,
    *,
    threshold: float,
    eligible: ArrayLike,
) -> NDArray[np.bool_]:
    """Build an undirected graph using the literal strict ``<`` comparator."""

    matrix = np.asarray(values, dtype=np.float64)
    mask = np.asarray(eligible, dtype=bool)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise LabelContractError("distance values must be a square matrix.")
    if mask.shape != (matrix.shape[0],):
        raise LabelContractError("eligible mask length mismatch.")
    if not np.isfinite(threshold):
        raise LabelContractError("threshold must be finite.")
    adjacency = matrix < float(threshold)
    adjacency &= mask[:, None] & mask[None, :]
    np.fill_diagonal(adjacency, False)
    return adjacency


def _euclidean_pairwise(values: NDArray[np.float64]) -> NDArray[np.float64]:
    differences = values[:, None, :] - values[None, :, :]
    return np.sqrt(np.sum(differences * differences, axis=2))


def metric_result(states: ArrayLike, *, metric: MetricName) -> MetricResult:
    """Compute one frozen metric while retaining all ineligible rows explicitly."""

    matrix = _state_matrix(states)
    row_sums = np.sum(matrix, axis=1)
    positive_mass = row_sums > 0
    reasons: list[str | None] = [None] * matrix.shape[0]
    for index in np.flatnonzero(~positive_mass):
        reasons[int(index)] = "ZERO_SUM_COMPOSITION"

    if metric == "aitchison":
        strictly_positive = np.all(matrix > 0, axis=1)
        eligible = positive_mass & strictly_positive
        for index in np.flatnonzero(positive_mass & ~strictly_positive):
            reasons[int(index)] = "ZERO_COMPONENT_STRICT_POSITIVE_AITCHISON"
    elif metric in ("cosine", "euclidean"):
        eligible = positive_mass
    else:
        raise LabelContractError(f"Unsupported metric {metric!r}.")

    representation = np.full_like(matrix, np.nan, dtype=np.float64)
    if metric == "cosine":
        representation[eligible] = matrix[eligible]
        values = np.full((matrix.shape[0], matrix.shape[0]), np.nan, dtype=np.float64)
        if np.any(eligible):
            indices = np.flatnonzero(eligible)
            subset = historical_h(matrix[indices].T)
            values[np.ix_(indices, indices)] = subset
        distances = 1.0 - values
    else:
        closed = np.full_like(matrix, np.nan, dtype=np.float64)
        closed[eligible] = matrix[eligible] / row_sums[eligible, None]
        if metric == "euclidean":
            representation[eligible] = closed[eligible]
        else:
            logs = np.log(closed[eligible])
            representation[eligible] = logs - np.mean(logs, axis=1, keepdims=True)
        values = np.full((matrix.shape[0], matrix.shape[0]), np.nan, dtype=np.float64)
        if np.any(eligible):
            indices = np.flatnonzero(eligible)
            subset = _euclidean_pairwise(representation[indices])
            values[np.ix_(indices, indices)] = subset
        distances = values.copy()

    return MetricResult(
        metric=metric,
        values=values,
        distances=distances,
        eligible=tuple(bool(value) for value in eligible),
        reasons=tuple(reasons),
        representation=representation,
    )


def _components(
    adjacency: NDArray[np.bool_], eligible: NDArray[np.bool_]
) -> list[tuple[int, ...]]:
    remaining = {int(index) for index in np.flatnonzero(eligible)}
    result: list[tuple[int, ...]] = []
    while remaining:
        root = min(remaining)
        stack = [root]
        found: set[int] = set()
        while stack:
            current = stack.pop()
            if current in found:
                continue
            found.add(current)
            neighbors = np.flatnonzero(adjacency[current])
            stack.extend(int(item) for item in neighbors if int(item) not in found)
        remaining.difference_update(found)
        result.append(tuple(sorted(found)))
    return result


def _adjacency(metric: MetricResult, threshold: float) -> NDArray[np.bool_]:
    eligible = np.asarray(metric.eligible, dtype=bool)
    if metric.metric == "cosine":
        return strict_similarity_adjacency(
            metric.values, threshold=threshold, eligible=eligible
        )
    return strict_distance_adjacency(
        metric.values, threshold=threshold, eligible=eligible
    )


def _component_details(
    component: tuple[int, ...],
    *,
    ids: tuple[str, ...],
    distances: NDArray[np.float64],
) -> tuple[str, int, float]:
    canonical_member = min(component, key=lambda index: ids[index])
    component_id = f"component::{ids[canonical_member]}"
    distance_block = distances[np.ix_(component, component)]
    sums = np.sum(distance_block, axis=1)
    minimum = float(np.min(sums))
    tied = [
        component[index]
        for index, value in enumerate(sums)
        if np.isclose(float(value), minimum, rtol=0.0, atol=1e-15)
    ]
    medoid = min(tied, key=lambda index: ids[index])
    return component_id, medoid, minimum


def _cluster_rows_for_prefix(
    matrix: NDArray[np.float64],
    *,
    ids: tuple[str, ...],
    trajectory_id: str,
    config: ClusterConfiguration,
    current_only: bool,
) -> tuple[list[LabelRecord], MetricResult]:
    metric = metric_result(matrix, metric=config.metric)
    eligible = np.asarray(metric.eligible, dtype=bool)
    adjacency = _adjacency(metric, config.threshold)
    components = _components(adjacency, eligible)
    component_for: dict[int, tuple[int, ...]] = {
        member: component for component in components for member in component
    }
    indices = [matrix.shape[0] - 1] if current_only else list(range(matrix.shape[0]))
    rows: list[LabelRecord] = []
    for index in indices:
        if not eligible[index]:
            rows.append(
                LabelRecord(
                    trajectory_id=trajectory_id,
                    observation_id=ids[index],
                    observation_index_one_based=index + 1,
                    configuration_id=config.configuration_id,
                    family_id=config.family_id,
                    temporal_scope=config.temporal_scope,
                    evidence_class=config.evidence_class,
                    label_status="INELIGIBLE",
                    is_replicator=None,
                    cluster_id=None,
                    component_id=None,
                    reference_observation_id=None,
                    metric_to_reference=None,
                    historical_incoming_h=None,
                    historical_local_score=None,
                    ineligibility_reason=metric.reasons[index],
                    source_padding=False,
                )
            )
            continue
        component = component_for[index]
        component_id, medoid, _ = _component_details(
            component, ids=ids, distances=metric.distances
        )
        is_replicator = len(component) >= config.minimum_cluster_size
        rows.append(
            LabelRecord(
                trajectory_id=trajectory_id,
                observation_id=ids[index],
                observation_index_one_based=index + 1,
                configuration_id=config.configuration_id,
                family_id=config.family_id,
                temporal_scope=config.temporal_scope,
                evidence_class=config.evidence_class,
                label_status=(
                    "LABELED_REPLICATOR" if is_replicator else "LABELED_DRIFT"
                ),
                is_replicator=is_replicator,
                cluster_id=component_id if is_replicator else None,
                component_id=component_id,
                reference_observation_id=ids[medoid] if is_replicator else None,
                metric_to_reference=(
                    float(metric.distances[index, medoid]) if is_replicator else None
                ),
                historical_incoming_h=None,
                historical_local_score=None,
                ineligibility_reason=None,
                source_padding=False,
            )
        )
    return rows, metric


def cluster_labels(
    states: ArrayLike,
    *,
    trajectory_id: str,
    observation_ids: tuple[str, ...] | list[str],
    configuration: ClusterConfiguration,
) -> LabelTraceResult:
    """Label one trace using a complete explicit retrospective or online branch."""

    matrix = _state_matrix(states)
    ids = _observation_ids(matrix.shape[0], observation_ids)
    if not isinstance(trajectory_id, str) or not trajectory_id:
        raise LabelContractError("trajectory_id must be a nonempty string.")

    if configuration.temporal_scope == "paper_retrospective_full_trace":
        rows, metric = _cluster_rows_for_prefix(
            matrix,
            ids=ids,
            trajectory_id=trajectory_id,
            config=configuration,
            current_only=False,
        )
    else:
        rows = []
        for stop in range(1, matrix.shape[0] + 1):
            prefix_rows, _ = _cluster_rows_for_prefix(
                matrix[:stop],
                ids=ids[:stop],
                trajectory_id=trajectory_id,
                config=configuration,
                current_only=True,
            )
            rows.append(prefix_rows[0])
        # A full-trace metric matrix would expose future observations through
        # this return object even though the labels above are prefix-only.
        metric = None
    return LabelTraceResult(
        trajectory_id=trajectory_id,
        configuration_id=configuration.configuration_id,
        family_id=configuration.family_id,
        result_status="OK",
        result_reason=None,
        rows=tuple(rows),
        metric_result=metric,
    )


def historical_technique1_labels(
    states: ArrayLike,
    *,
    trajectory_id: str,
    observation_ids: tuple[str, ...] | list[str],
    configuration_id: str,
    threshold: float,
    evidence_class: str,
) -> LabelTraceResult:
    """Wrap the S04 source-traceable technique-1 behavior with complete statuses."""

    matrix = _state_matrix(states)
    ids = _observation_ids(matrix.shape[0], observation_ids)
    try:
        historical = historical_nondrift_technique1(matrix.T, threshold=threshold)
    except (HistoricalReferenceError, HistoricalSourceDomainError) as exc:
        rows = tuple(
            LabelRecord(
                trajectory_id=trajectory_id,
                observation_id=observation_id,
                observation_index_one_based=index + 1,
                configuration_id=configuration_id,
                family_id="Y_H",
                temporal_scope="local_adjacent_with_next_observation_for_interior_scores",
                evidence_class=evidence_class,
                label_status="ERROR_SOURCE_DOMAIN",
                is_replicator=None,
                cluster_id=None,
                component_id=None,
                reference_observation_id=None,
                metric_to_reference=None,
                historical_incoming_h=None,
                historical_local_score=None,
                ineligibility_reason=str(exc),
                source_padding=False,
            )
            for index, observation_id in enumerate(ids)
        )
        return LabelTraceResult(
            trajectory_id=trajectory_id,
            configuration_id=configuration_id,
            family_id="Y_H",
            result_status="ERROR_SOURCE_DOMAIN",
            result_reason=str(exc),
            rows=rows,
            metric_result=None,
        )

    rows: list[LabelRecord] = []
    for index, observation_id in enumerate(ids):
        padded = index >= historical.active_generation_count
        is_replicator = historical.is_non_drift[index]
        rows.append(
            LabelRecord(
                trajectory_id=trajectory_id,
                observation_id=observation_id,
                observation_index_one_based=index + 1,
                configuration_id=configuration_id,
                family_id="Y_H",
                temporal_scope="local_adjacent_with_next_observation_for_interior_scores",
                evidence_class=evidence_class,
                label_status=(
                    "LABELED_DRIFT_SOURCE_PADDED_AFTER_FIRST_ZERO_SUM"
                    if padded
                    else ("LABELED_REPLICATOR" if is_replicator else "LABELED_DRIFT")
                ),
                is_replicator=is_replicator,
                cluster_id=None,
                component_id=None,
                reference_observation_id=None,
                metric_to_reference=None,
                historical_incoming_h=float(historical.angles[index]),
                historical_local_score=float(historical.local_scores[index]),
                ineligibility_reason=None,
                source_padding=padded,
            )
        )
    return LabelTraceResult(
        trajectory_id=trajectory_id,
        configuration_id=configuration_id,
        family_id="Y_H",
        result_status="OK",
        result_reason=None,
        rows=tuple(rows),
        metric_result=None,
    )


def historical_technique2_diagnostic(
    states: ArrayLike,
    *,
    trajectory_id: str,
    configuration_id: str,
    threshold: float,
    drift_size: int,
) -> dict[str, Any]:
    """Return optional source behavior or an explicit, unrepaired source-domain error."""

    matrix = _state_matrix(states)
    try:
        result = historical_nondrift_technique2(
            matrix.T, threshold=threshold, drift_size=drift_size
        )
    except (HistoricalReferenceError, HistoricalSourceDomainError) as exc:
        return {
            "trajectoryId": trajectory_id,
            "configurationId": configuration_id,
            "status": "ERROR_SOURCE_DOMAIN",
            "errorType": type(exc).__name__,
            "reason": str(exc),
            "sourceRepairApplied": False,
        }
    return {
        "trajectoryId": trajectory_id,
        "configurationId": configuration_id,
        "status": "OK",
        "threshold": result.threshold,
        "driftSize": result.drift_size,
        "labels": list(result.is_non_drift),
        "angles": list(result.angles),
        "activeGenerationCount": result.active_generation_count,
        "firstZeroSumGenerationOneBased": result.first_zero_sum_generation_one_based,
        "sourceRepairApplied": False,
    }


def continuous_past_recurrence(states: ArrayLike) -> tuple[float | None, ...]:
    """Return R_g=max_{h<g} H(n_g,n_h) without creating another binary label."""

    metric = metric_result(states, metric="cosine")
    recurrence: list[float | None] = []
    for index, is_eligible in enumerate(metric.eligible):
        if not is_eligible or index == 0:
            recurrence.append(None)
            continue
        prior = [
            float(metric.values[index, previous])
            for previous in range(index)
            if metric.eligible[previous]
        ]
        recurrence.append(max(prior) if prior else None)
    return tuple(recurrence)
