"""Frozen Phase-3 source-informed information branches for E01 S12E."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from e01_pigozzi_source_audit.core import (
    SourceImplementation,
    _iigr_preprocess,
    _phirl_mi_matrix,
    _phirl_preprocess,
    _source_partition,
)
from e01_pigozzi_source_equivalence_confirmation.core import iigr_pairwise_source_mi
from e01_source_emergence_metric_identity.core import component_arrays

METRIC_IDS = (
    "M1_IIGR_EMERGENCE_CLR_FULL",
    "M2_IIGR_EMERGENCE_CLR_GSR_AR1_FULL",
    "M3_IIGR_LOCAL_PHIR_CLR_FULL",
    "M4_PHIRL_EMERGENCE_CLR_FULL",
)


@dataclass(frozen=True, slots=True)
class SourceMetricResult:
    metric_id: str
    status: str
    reason: str | None
    retained_variables: tuple[int, ...]
    partition_1: tuple[int, ...]
    partition_2: tuple[int, ...]
    fiedler_vector: NDArray[np.float64] | None
    mi_matrix: NDArray[np.float64] | None
    partition_average: NDArray[np.float64] | None
    synergy: NDArray[np.float64] | None
    downward_causation: NDArray[np.float64] | None
    emergence: NDArray[np.float64] | None
    local_phi_r: NDArray[np.float64] | None
    scalar: NDArray[np.float64] | None
    local_offset: int
    covariance_condition_number: float | None
    nonfinite_count: int


def common_clr_drop100(states: NDArray[np.int64]) -> NDArray[np.float64]:
    """Apply additive-0.5 closure, full CLR, then remove component 100."""

    counts = np.asarray(states, dtype=np.float64)
    if counts.ndim != 2 or counts.shape[1] != 100:
        raise ValueError("S12E common substrate requires observations by 100 counts")
    if np.any(counts < 0) or not np.all(np.isfinite(counts)):
        raise ValueError("counts must be finite and nonnegative")
    masses = counts.sum(axis=1)
    closed = (counts + 0.5) / (masses[:, None] + 50.0)
    logs = np.log(closed)
    clr = logs - logs.mean(axis=1, keepdims=True)
    return np.asarray(clr[:, :99], dtype=np.float64)


def _empty(metric_id: str, status: str, reason: str, offset: int) -> SourceMetricResult:
    return SourceMetricResult(
        metric_id,
        status,
        reason,
        (),
        (),
        (),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        offset,
        None,
        0,
    )


def run_metric_branch(
    common_clr: NDArray[np.float64],
    metric_id: str,
    safe_lattice_path: str | Path,
    *,
    preprocessing_seed: int,
    partition_seed: int,
) -> SourceMetricResult:
    """Run one frozen full-array or prefix branch with status-bearing failures."""

    if metric_id not in METRIC_IDS:
        raise ValueError(f"unknown S12E metric branch {metric_id!r}")
    raw = np.asarray(common_clr, dtype=np.float64)
    offset = 2 if metric_id == "M2_IIGR_EMERGENCE_CLR_GSR_AR1_FULL" else 1
    if raw.ndim != 2 or raw.shape[0] < offset + 2 or raw.shape[1] != 99:
        return _empty(metric_id, "INELIGIBLE_INPUT_SHAPE", "requires_time_by_99_CLR", offset)
    if not np.all(np.isfinite(raw)):
        return _empty(metric_id, "INELIGIBLE_NONFINITE_INPUT", "common_CLR_nonfinite", offset)

    try:
        if metric_id in {
            "M1_IIGR_EMERGENCE_CLR_FULL",
            "M3_IIGR_LOCAL_PHIR_CLR_FULL",
        }:
            implementation = SourceImplementation.IIGR
            processed = raw.T.copy()
            retained = tuple(range(99))
            mi = iigr_pairwise_source_mi(processed, alpha=1, lag=1)
        elif metric_id == "M2_IIGR_EMERGENCE_CLR_GSR_AR1_FULL":
            implementation = SourceImplementation.IIGR
            processed, retained = _iigr_preprocess(raw.T, preprocessing_seed)
            mi = iigr_pairwise_source_mi(processed, alpha=1, lag=1)
        else:
            implementation = SourceImplementation.PHIRL
            processed, retained = _phirl_preprocess(raw.T)
            if len(retained) < 2:
                return _empty(
                    metric_id,
                    "INELIGIBLE_TOO_FEW_ACTIVE_DIMENSIONS",
                    "fewer_than_two_dimensions_above_std_1e-8",
                    offset,
                )
            mi = _phirl_mi_matrix(processed)
        if not np.all(np.isfinite(processed)) or not np.all(np.isfinite(mi)):
            return _empty(
                metric_id,
                "INELIGIBLE_NONFINITE_PREPROCESSING_OR_MI",
                "processed_array_or_mi_nonfinite",
                offset,
            )
        fiedler, p1_local, p2_local = _source_partition(mi, partition_seed)
        if not p1_local or not p2_local:
            return _empty(
                metric_id,
                "INELIGIBLE_FIEDLER_PARTITION_EMPTY",
                "strict_sign_partition_has_empty_side",
                offset,
            )
        if np.any(fiedler == 0.0):
            return _empty(
                metric_id,
                "INELIGIBLE_FIEDLER_PARTITION_AMBIGUOUS",
                "one_or_more_fiedler_entries_equal_zero",
                offset,
            )
        reduced = np.vstack(
            (
                processed[list(p1_local)].mean(axis=0),
                processed[list(p2_local)].mean(axis=0),
            )
        )
        synergy, downward, emergence, phi_r = component_arrays(
            reduced, implementation, safe_lattice_path
        )
        scalar = (
            phi_r
            if metric_id == "M3_IIGR_LOCAL_PHIR_CLR_FULL"
            else emergence
        )
        covariance = np.cov(reduced, ddof=0)
        condition = float(np.linalg.cond(covariance))
        nonfinite = int(np.count_nonzero(~np.isfinite(scalar)))
        status = "ELIGIBLE" if nonfinite == 0 else "ELIGIBLE_PARTIAL_NONFINITE_LOCAL_VALUES"
        reason = None if nonfinite == 0 else "one_or_more_local_scalar_values_nonfinite"
        p1 = tuple(retained[index] for index in p1_local)
        p2 = tuple(retained[index] for index in p2_local)
        return SourceMetricResult(
            metric_id=metric_id,
            status=status,
            reason=reason,
            retained_variables=retained,
            partition_1=p1,
            partition_2=p2,
            fiedler_vector=fiedler,
            mi_matrix=mi,
            partition_average=reduced,
            synergy=synergy,
            downward_causation=downward,
            emergence=emergence,
            local_phi_r=phi_r,
            scalar=scalar,
            local_offset=offset,
            covariance_condition_number=condition,
            nonfinite_count=nonfinite,
        )
    except Exception as exc:  # noqa: BLE001 - failures must remain status-bearing.
        return _empty(
            metric_id,
            "INELIGIBLE_SOURCE_PIPELINE_EXCEPTION",
            f"{type(exc).__name__}:{exc}",
            offset,
        )


def source_result_replay_equal(
    left: SourceMetricResult, right: SourceMetricResult
) -> bool:
    scalars = (
        left.metric_id == right.metric_id
        and left.status == right.status
        and left.reason == right.reason
        and left.retained_variables == right.retained_variables
        and left.partition_1 == right.partition_1
        and left.partition_2 == right.partition_2
        and left.local_offset == right.local_offset
        and left.covariance_condition_number == right.covariance_condition_number
        and left.nonfinite_count == right.nonfinite_count
    )
    if not scalars:
        return False
    for name in (
        "fiedler_vector",
        "mi_matrix",
        "partition_average",
        "synergy",
        "downward_causation",
        "emergence",
        "local_phi_r",
        "scalar",
    ):
        a, b = getattr(left, name), getattr(right, name)
        if (a is None) != (b is None):
            return False
        if a is not None and not np.array_equal(a, b, equal_nan=True):
            return False
    return True


def metric_value_rows(
    *,
    pipeline_id: str,
    trajectory_id: str,
    matrix_index: int,
    metric: SourceMetricResult,
    observation_indices: NDArray[np.int64],
    observation_kinds: list[str],
    generations: NDArray[np.int64],
    molecular_steps: NDArray[np.int64],
    temporal_mode: str,
) -> list[dict[str, object]]:
    """Emit one status-bearing row for each expected aligned local value."""

    output: list[dict[str, object]] = []
    scalar = metric.scalar
    for local_index, observation_index in enumerate(observation_indices):
        value: float | None = None
        status = metric.status
        reason = metric.reason
        if scalar is not None and local_index < scalar.size:
            candidate = float(scalar[local_index])
            if np.isfinite(candidate):
                value = candidate
                status = "ELIGIBLE"
                reason = None
            else:
                status = "INELIGIBLE_NONFINITE_LOCAL_VALUE"
                reason = "source_local_scalar_nonfinite"
        output.append(
            {
                "pipelineId": pipeline_id,
                "trajectoryId": trajectory_id,
                "matrixIndex": matrix_index,
                "metricId": metric.metric_id,
                "temporalMode": temporal_mode,
                "localIndex": local_index,
                "observationIndex": int(observation_index),
                "observationKind": observation_kinds[int(observation_index)],
                "generation": int(generations[int(observation_index)]),
                "molecularStep": int(molecular_steps[int(observation_index)]),
                "status": status,
                "reason": reason,
                "value": value,
                "conditionNumber": metric.covariance_condition_number,
            }
        )
    return output
