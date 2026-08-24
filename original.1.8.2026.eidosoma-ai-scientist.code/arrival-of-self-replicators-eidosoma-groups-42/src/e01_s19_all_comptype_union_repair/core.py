"""Frozen numerical repair for E01/S19-L11R.

This module changes one implementation detail relative to the immutable L11
code: U2 centroids use the material-negative tolerance already established in
L10.  Every other scientific primitive, seed contract, label definition,
control, statistic, and gate is imported unchanged from L11.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from e01_s19_all_comptype_union.core import (
    BOOTSTRAP_REPLICATES,
    PAPER_TARGETS,
    PIPELINE_IDS,
    RANDOM_CENTROID_DRAWS,
    THRESHOLD,
    U1_ID,
    U2_ID,
    UnionLabelResult,
    _empty_result,
    array_sha256,
    bootstrap_indices,
    close_rows,
    deterministic_seed,
    historical_h,
    label_fingerprint,
    materialize_u1,
    paper_distance,
    project_boundary_values,
    run_descriptors,
    serialize_worker_exception,
)
from e01_s19_matlab_attractor.core import fit_r2_euclidean

VERSION = "E01-S19-L11R-CENTROID-NUMERICAL-TOLERANCE-CONFIRMATION-v1.0.0"
LOOP_ID = "S19-L11R"
MATERIAL_NEGATIVE_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class CentroidRepairAudit:
    """Machine-readable evidence for one centroid sanitation operation."""

    row_count: int
    coordinate_count: int
    finite_count: int
    minimum_raw_coordinate: float
    negative_coordinate_count: int
    clamped_coordinate_count: int
    material_negative_coordinate_count: int
    zero_sum_row_count_before_reclosure: int
    maximum_unit_sum_error_after_reclosure: float
    raw_sha256: str
    repaired_sha256: str


def repair_u2_centroids(
    centroids: NDArray[Any],
) -> tuple[NDArray[np.float64], CentroidRepairAudit]:
    """Apply exactly the prospectively authorized L10 centroid policy.

    A finite coordinate below ``-1e-12`` is material and fails. Coordinates
    from ``-1e-12`` through zero are set to zero, and each row is then closed
    to a positive unit-sum composition.  No other tolerance or transformation
    is applied.
    """

    raw = np.asarray(centroids, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[0] == 0 or raw.shape[1] == 0:
        raise ValueError("U2 centroid repair requires a nonempty matrix")
    if np.any(~np.isfinite(raw)):
        raise ValueError("U2 centroid repair requires finite coordinates")
    material = raw < -MATERIAL_NEGATIVE_TOLERANCE
    if np.any(material):
        raise ValueError("U2 centroid contains a material negative coordinate")
    repaired_unclosed = np.where(raw <= 0.0, 0.0, raw)
    totals = np.sum(repaired_unclosed, axis=1)
    zero_sum = totals <= 0.0
    if np.any(zero_sum):
        raise ValueError("U2 centroid repair produced a zero-sum composition")
    repaired = close_rows(repaired_unclosed)
    audit = CentroidRepairAudit(
        row_count=int(raw.shape[0]),
        coordinate_count=int(raw.size),
        finite_count=int(np.count_nonzero(np.isfinite(raw))),
        minimum_raw_coordinate=float(np.min(raw)),
        negative_coordinate_count=int(np.count_nonzero(raw < 0.0)),
        clamped_coordinate_count=int(np.count_nonzero(raw <= 0.0)),
        material_negative_coordinate_count=int(np.count_nonzero(material)),
        zero_sum_row_count_before_reclosure=int(np.count_nonzero(zero_sum)),
        maximum_unit_sum_error_after_reclosure=float(
            np.max(np.abs(np.sum(repaired, axis=1) - 1.0))
        ),
        raw_sha256=array_sha256(raw),
        repaired_sha256=array_sha256(repaired),
    )
    return np.ascontiguousarray(repaired, dtype=np.float64), audit


def direct_union_scores(
    compositions: NDArray[Any], centroids: NDArray[Any]
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Return strict-H090 union membership after the sole L11R repair."""

    values = close_rows(compositions)
    references, _ = repair_u2_centroids(centroids)
    scores = np.max(historical_h(values, references), axis=1)
    return np.asarray(scores, dtype=np.float64), np.asarray(
        scores > THRESHOLD, dtype=bool
    )


def materialize_u2(
    boundary_compositions: NDArray[Any],
    molecular_compositions: NDArray[Any],
    trajectory_identity: str,
) -> UnionLabelResult:
    """Materialize unchanged U2 with repaired centroids before H scoring."""

    boundary = close_rows(boundary_compositions)
    molecular = close_rows(molecular_compositions)
    fit = fit_r2_euclidean(boundary, trajectory_identity)
    if fit.labels is None or fit.centroids is None or fit.selected_k is None:
        return _empty_result(U2_ID, fit.status, fit)
    cluster_sizes = tuple(
        int(np.count_nonzero(np.asarray(fit.labels) == cluster_id))
        for cluster_id in range(len(fit.centroids))
    )
    if cluster_sizes != fit.cluster_sizes:
        raise ValueError("U2 cluster-size replay mismatch")
    repaired_all, _ = repair_u2_centroids(fit.centroids)
    recurring_ids = tuple(
        index for index, size in enumerate(cluster_sizes) if size >= 2
    )
    singleton_ids = tuple(
        index for index, size in enumerate(cluster_sizes) if size == 1
    )
    if not recurring_ids:
        return _empty_result(U2_ID, "NO_RECURRING_CLUSTER_UNION", fit)
    recurring = np.asarray(repaired_all[list(recurring_ids)], dtype=np.float64)
    singleton = (
        np.asarray(repaired_all[list(singleton_ids)], dtype=np.float64)
        if singleton_ids
        else None
    )
    molecular_scores, molecular_labels = direct_union_scores(molecular, recurring)
    boundary_scores, boundary_labels = direct_union_scores(boundary, recurring)
    tags = np.asarray(fit.labels, dtype=np.int64) + 1
    boundary_sizes = np.asarray(
        [cluster_sizes[int(tag) - 1] for tag in tags], dtype=np.int64
    )
    return UnionLabelResult(
        pipeline_id=U2_ID,
        status="ELIGIBLE_RECURRING_CENTROID_UNION",
        fit=fit,
        boundary_labels=boundary_labels,
        molecular_labels=molecular_labels,
        boundary_scores=boundary_scores,
        molecular_scores=molecular_scores,
        boundary_tags=tags,
        molecular_tags=None,
        boundary_cluster_sizes=boundary_sizes,
        molecular_cluster_sizes=None,
        recurring_cluster_ids=recurring_ids,
        singleton_cluster_ids=singleton_ids,
        recurring_centroids=recurring,
        singleton_centroids=singleton,
        singleton_positive_fraction=None,
    )


def materialize_pipeline(
    pipeline_id: str,
    boundary_compositions: NDArray[Any],
    molecular_compositions: NDArray[Any],
    boundary_positions: NDArray[Any],
    trajectory_identity: str,
) -> UnionLabelResult:
    """Dispatch the unchanged U1 or one-repair U2 pipeline."""

    if pipeline_id == U1_ID:
        return materialize_u1(
            boundary_compositions,
            molecular_compositions,
            boundary_positions,
            trajectory_identity,
        )
    if pipeline_id == U2_ID:
        return materialize_u2(
            boundary_compositions, molecular_compositions, trajectory_identity
        )
    raise ValueError(f"unregistered L11R primary pipeline {pipeline_id!r}")


__all__ = [
    "BOOTSTRAP_REPLICATES",
    "LOOP_ID",
    "MATERIAL_NEGATIVE_TOLERANCE",
    "PAPER_TARGETS",
    "PIPELINE_IDS",
    "RANDOM_CENTROID_DRAWS",
    "THRESHOLD",
    "U1_ID",
    "U2_ID",
    "VERSION",
    "CentroidRepairAudit",
    "UnionLabelResult",
    "array_sha256",
    "bootstrap_indices",
    "close_rows",
    "deterministic_seed",
    "direct_union_scores",
    "historical_h",
    "label_fingerprint",
    "materialize_pipeline",
    "materialize_u1",
    "materialize_u2",
    "paper_distance",
    "project_boundary_values",
    "repair_u2_centroids",
    "run_descriptors",
    "serialize_worker_exception",
]
