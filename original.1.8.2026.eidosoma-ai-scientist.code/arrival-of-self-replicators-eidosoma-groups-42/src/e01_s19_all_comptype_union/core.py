"""Frozen scientific primitives for E01/S19-L11.

L11 asks one question only: whether the binary self-replicator state is the
union of all selected compotype clusters rather than membership in one
dominant cluster.  U1 is the source-literal historical tag union projected
from each selected daughter to the next boundary.  U2 is direct molecular
strict-H membership in every Euclidean cluster with at least two boundary
members.  The module imports no emergence, prediction, intervention, or
metric-distinctiveness code.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from e01_s19_matlab_attractor.core import (
    ClusterFit,
    array_sha256,
    close_rows,
    fit_r1_matlab_historical,
    fit_r2_euclidean,
    historical_h,
    label_fingerprint,
    paper_distance,
    run_descriptors,
)

VERSION = "E01-S19-L11-ALL-COMPTYPE-UNION-LABEL-RECONSTRUCTION-v1.0.0"
LOOP_ID = "S19-L11"
U1_ID = "U1_HISTORICAL_ALL_COMPTYPE_TAGS_H090"
U2_ID = "U2_PAPER_EUCLIDEAN_ALL_RECURRING_CENTROIDS_H090"
PIPELINE_IDS = (U1_ID, U2_ID)
THRESHOLD = 0.9
BOOTSTRAP_REPLICATES = 4096
RANDOM_CENTROID_DRAWS = 64

CONTROL_ROOT_HEX = "fe3872bc0e2774f42fb76008a0f6e50fa27266e7ec48949c29b315f0ad149247"
BOOTSTRAP_ROOT_HEX = "4ddc59d7a3489ea463a0676d768fa6927a2c334a2edb4b80279e387e01ca0833"

PAPER_TARGETS: dict[str, tuple[float, float]] = {
    "selectedClockLength": (716.0 / 0.88, 225.0),
    "persistence": (716.0, 198.0),
    "occupancy": (0.88, 0.03),
    "consistency": (0.38, 0.06),
    "firstOnsetRawStep1": (37.0, 27.0),
    "firstOnsetNormalized": (0.37, 0.27),
}


@dataclass(frozen=True, slots=True)
class UnionLabelResult:
    """Complete status-bearing output of one registered primary pipeline."""

    pipeline_id: str
    status: str
    fit: ClusterFit
    boundary_labels: NDArray[np.bool_] | None
    molecular_labels: NDArray[np.bool_] | None
    boundary_scores: NDArray[np.float64] | None
    molecular_scores: NDArray[np.float64] | None
    boundary_tags: NDArray[np.int64] | None
    molecular_tags: NDArray[np.int64] | None
    boundary_cluster_sizes: NDArray[np.int64] | None
    molecular_cluster_sizes: NDArray[np.int64] | None
    recurring_cluster_ids: tuple[int, ...]
    singleton_cluster_ids: tuple[int, ...]
    recurring_centroids: NDArray[np.float64] | None
    singleton_centroids: NDArray[np.float64] | None
    singleton_positive_fraction: float | None


def deterministic_seed(
    *parts: object, bits: int = 128, root_hex: str = CONTROL_ROOT_HEX
) -> int:
    """Return a domain-separated PCG seed under the frozen L11 identity."""

    material = "\x1f".join([VERSION, root_hex, *map(str, parts)]).encode("utf-8")
    digest = hashlib.sha256(material).digest()
    if bits == 32:
        return int.from_bytes(digest[:4], "big")
    if bits == 128:
        return int.from_bytes(digest[:16], "big")
    if bits == 256:
        return int.from_bytes(digest, "big")
    raise ValueError("seed bits must be 32, 128, or 256")


def serialize_worker_exception(
    *,
    matrix_id: int,
    candidate_id: str,
    pipeline_id: str,
    generation: int | None,
    selected_k: int | None,
    cluster_sizes: tuple[int, ...] | list[int],
    tag_counts: dict[str, int],
    seed_identity: str,
    error: BaseException,
) -> dict[str, Any]:
    """Serialize every field required by the frozen L11 worker-failure contract."""

    return {
        "matrixId": int(matrix_id),
        "candidateId": str(candidate_id),
        "pipelineId": str(pipeline_id),
        "generation": None if generation is None else int(generation),
        "selectedK": None if selected_k is None else int(selected_k),
        "clusterSizes": [int(value) for value in cluster_sizes],
        "tagCounts": {str(key): int(value) for key, value in tag_counts.items()},
        "seedIdentity": str(seed_identity),
        "exceptionClass": type(error).__name__,
        "exceptionMessage": str(error),
    }


def project_boundary_values(
    boundary_values: NDArray[Any],
    boundary_positions: NDArray[Any],
    molecular_length: int,
    *,
    prefix_value: int | bool = 0,
) -> NDArray[Any]:
    """Project a boundary value from its daughter to before the next boundary."""

    values = np.asarray(boundary_values)
    positions = np.asarray(boundary_positions, dtype=np.int64)
    if values.ndim != 1 or positions.ndim != 1 or len(values) != len(positions):
        raise ValueError("boundary values and positions must be aligned vectors")
    if molecular_length <= 0 or len(values) == 0:
        raise ValueError("projection requires nonempty molecular and boundary clocks")
    if np.any(positions < 0) or np.any(positions >= molecular_length):
        raise ValueError("boundary position outside molecular clock")
    if np.any(np.diff(positions) <= 0):
        raise ValueError("boundary positions must be strictly increasing")
    output = np.full(molecular_length, prefix_value, dtype=values.dtype)
    for index, start in enumerate(positions):
        stop = positions[index + 1] if index + 1 < len(positions) else molecular_length
        output[start:stop] = values[index]
    return output


def _empty_result(pipeline_id: str, status: str, fit: ClusterFit) -> UnionLabelResult:
    return UnionLabelResult(
        pipeline_id=pipeline_id,
        status=status,
        fit=fit,
        boundary_labels=None,
        molecular_labels=None,
        boundary_scores=None,
        molecular_scores=None,
        boundary_tags=None,
        molecular_tags=None,
        boundary_cluster_sizes=None,
        molecular_cluster_sizes=None,
        recurring_cluster_ids=(),
        singleton_cluster_ids=(),
        recurring_centroids=None,
        singleton_centroids=None,
        singleton_positive_fraction=None,
    )


def materialize_u1(
    boundary_compositions: NDArray[Any],
    molecular_compositions: NDArray[Any],
    boundary_positions: NDArray[Any],
    trajectory_identity: str,
) -> UnionLabelResult:
    """Materialize the historical source-tag union and its fixed projection."""

    boundary = close_rows(boundary_compositions)
    molecular = close_rows(molecular_compositions)
    fit = fit_r1_matlab_historical(boundary, trajectory_identity)
    if fit.labels is None or fit.centroids is None or fit.selected_k is None:
        return _empty_result(U1_ID, fit.status, fit)

    tags = np.zeros(len(boundary), dtype=np.int64)
    eligible_positions = np.flatnonzero(fit.eligible_mask)
    if len(eligible_positions) != len(fit.labels):
        raise ValueError("U1 eligible-mask and selected-tag cardinality mismatch")
    tags[eligible_positions] = np.asarray(fit.labels, dtype=np.int64) + 1
    if np.any(tags[~fit.eligible_mask] != 0) or np.any(tags[fit.eligible_mask] <= 0):
        raise ValueError("U1 failed source tag-zero/tag-positive invariant")

    cluster_sizes = tuple(
        int(np.count_nonzero(np.asarray(fit.labels) == cluster_id))
        for cluster_id in range(len(fit.centroids))
    )
    if cluster_sizes != fit.cluster_sizes:
        raise ValueError("U1 cluster-size replay mismatch")
    boundary_sizes = np.zeros(len(boundary), dtype=np.int64)
    for position, tag in enumerate(tags):
        if tag > 0:
            boundary_sizes[position] = cluster_sizes[int(tag) - 1]

    boundary_labels = tags > 0
    molecular_tags = project_boundary_values(
        tags, boundary_positions, len(molecular), prefix_value=0
    ).astype(np.int64)
    molecular_sizes = project_boundary_values(
        boundary_sizes, boundary_positions, len(molecular), prefix_value=0
    ).astype(np.int64)
    molecular_labels = molecular_tags > 0
    singleton_only = molecular_sizes == 1
    positive_count = int(np.count_nonzero(molecular_labels))
    singleton_fraction = (
        float(np.count_nonzero(singleton_only) / positive_count)
        if positive_count
        else 0.0
    )
    recurring_ids = tuple(index for index, size in enumerate(cluster_sizes) if size >= 2)
    singleton_ids = tuple(index for index, size in enumerate(cluster_sizes) if size == 1)
    boundary_scores = np.asarray(
        np.max(historical_h(boundary, fit.centroids), axis=1), dtype=np.float64
    )
    molecular_scores = project_boundary_values(
        boundary_scores, boundary_positions, len(molecular), prefix_value=0.0
    ).astype(np.float64)
    return UnionLabelResult(
        pipeline_id=U1_ID,
        status="ELIGIBLE_SOURCE_TAG_UNION",
        fit=fit,
        boundary_labels=np.asarray(boundary_labels, dtype=bool),
        molecular_labels=np.asarray(molecular_labels, dtype=bool),
        boundary_scores=boundary_scores,
        molecular_scores=molecular_scores,
        boundary_tags=tags,
        molecular_tags=molecular_tags,
        boundary_cluster_sizes=boundary_sizes,
        molecular_cluster_sizes=molecular_sizes,
        recurring_cluster_ids=recurring_ids,
        singleton_cluster_ids=singleton_ids,
        recurring_centroids=(
            np.asarray(fit.centroids[list(recurring_ids)], dtype=np.float64)
            if recurring_ids
            else None
        ),
        singleton_centroids=(
            np.asarray(fit.centroids[list(singleton_ids)], dtype=np.float64)
            if singleton_ids
            else None
        ),
        singleton_positive_fraction=singleton_fraction,
    )


def direct_union_scores(
    compositions: NDArray[Any], centroids: NDArray[Any]
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Return max historical H and strict-H090 union membership."""

    values = close_rows(compositions)
    references = close_rows(centroids)
    scores = np.max(historical_h(values, references), axis=1)
    return np.asarray(scores, dtype=np.float64), np.asarray(scores > THRESHOLD, dtype=bool)


def materialize_u2(
    boundary_compositions: NDArray[Any],
    molecular_compositions: NDArray[Any],
    trajectory_identity: str,
) -> UnionLabelResult:
    """Materialize direct H membership in all recurring Euclidean centroids."""

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
    recurring_ids = tuple(index for index, size in enumerate(cluster_sizes) if size >= 2)
    singleton_ids = tuple(index for index, size in enumerate(cluster_sizes) if size == 1)
    if not recurring_ids:
        return _empty_result(U2_ID, "NO_RECURRING_CLUSTER_UNION", fit)
    recurring = np.asarray(fit.centroids[list(recurring_ids)], dtype=np.float64)
    singleton = (
        np.asarray(fit.centroids[list(singleton_ids)], dtype=np.float64)
        if singleton_ids
        else None
    )
    molecular_scores, molecular_labels = direct_union_scores(molecular, recurring)
    boundary_scores, boundary_labels = direct_union_scores(boundary, recurring)
    tags = np.asarray(fit.labels, dtype=np.int64) + 1
    boundary_sizes = np.asarray([cluster_sizes[int(tag) - 1] for tag in tags], dtype=np.int64)
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
    raise ValueError(f"unregistered L11 primary pipeline {pipeline_id!r}")


def bootstrap_indices(candidate_id: str, pipeline_id: str) -> NDArray[np.int64]:
    seed = deterministic_seed(
        "bootstrap",
        candidate_id,
        pipeline_id,
        root_hex=BOOTSTRAP_ROOT_HEX,
    )
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    return rng.integers(0, 100, size=(BOOTSTRAP_REPLICATES, 100), dtype=np.int64)


__all__ = [
    "BOOTSTRAP_REPLICATES",
    "LOOP_ID",
    "PAPER_TARGETS",
    "PIPELINE_IDS",
    "RANDOM_CENTROID_DRAWS",
    "THRESHOLD",
    "U1_ID",
    "U2_ID",
    "UnionLabelResult",
    "VERSION",
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
    "run_descriptors",
    "serialize_worker_exception",
]
