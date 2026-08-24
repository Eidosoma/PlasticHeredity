"""Frozen scientific primitives for E01/S19-L10.

This module implements exactly two recurring-attractor pipelines.  It imports
no emergence, prediction, or intervention machinery.  R1 differs from the
failed-closed L09 implementation only by using the documented MATLAB
singleton-silhouette convention; R2 retains the frozen paper-Euclidean
clustering specification.  A separate scientific recurrence gate prevents a
software-optimal all-singleton or tied-largest solution from becoming a
replicator reference.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

from e01_s19_recurring_attractor.core import (
    _cluster_order,
    _cosine_kmeans_one,
    _historical_k1_score,
    close_rows,
    historical_nondrift_technique1,
    run_descriptors,
)
from e01_s19_recurring_attractor.core import historical_h as _historical_h
from e01_s19_recurring_attractor.core import (
    label_against_reference as _label_against_reference,
)


def historical_h(left: NDArray[Any], right: NDArray[Any]) -> NDArray[np.float64]:
    """Expose the pinned historical H primitive through the L10 namespace."""

    return np.asarray(_historical_h(left, right), dtype=np.float64)


def label_against_reference(
    values: NDArray[Any], reference: NDArray[Any]
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Expose the unchanged strict-H090 direct-membership primitive."""

    scores, labels = _label_against_reference(values, reference)
    return np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=bool)


VERSION = "E01-S19-L10-MATLAB-COMPATIBLE-RECURRING-ATTRACTOR-RECONSTRUCTION-v1.0.0"
LOOP_ID = "S19-L10"
R1_ID = "R1_MATLAB_COMPATIBLE_HISTORICAL_DOMINANT_COMPTYPE_H090"
R2_ID = "R2_PAPER_EUCLIDEAN_DOMINANT_ATTRACTOR_H090"
PIPELINE_IDS = (R1_ID, R2_ID)
THRESHOLD = 0.9
K_VALUES = tuple(range(1, 11))
REPLICAS = 10
EARLY_STOP_STREAK = 4
MAX_ITERATIONS = 300
COSINE_TOLERANCE = 1e-12
EUCLIDEAN_TOLERANCE = 1e-4
BOOTSTRAP_REPLICATES = 4096
RANDOM_REFERENCE_DRAWS = 64
TIME_PERMUTATION_DRAWS = 1
CLUSTER_ROOT_HEX = "3f0c61b5af62626d24d561d7ee730eadb9d23eb8b138fafd3dce3464dd30112f"
BOOTSTRAP_ROOT_HEX = "167c00bcb513f2c8dd98d26b160f335b40ed250bdeb7582d100c657fee36249b"

EXPECTED_SCIENTIFIC_STATUSES = {
    "NO_NONDRIFT_COMPOSITIONS",
    "NO_RECURRING_COMPTYPE",
    "NO_UNIQUE_RECURRING_COMPTYPE",
    "NO_VALID_CLUSTERING",
    "LABEL_CONSTANT_ZERO",
    "LABEL_CONSTANT_ONE",
    "UNDEFINED_CONSISTENCY",
}

PAPER_TARGETS: dict[str, tuple[float, float]] = {
    "selectedClockLength": (716.0 / 0.88, 225.0),
    "persistence": (716.0, 198.0),
    "occupancy": (0.88, 0.03),
    "consistency": (0.38, 0.06),
    "firstOnsetRawStep1": (37.0, 27.0),
    "firstOnsetNormalized": (0.37, 0.27),
}


@dataclass(frozen=True, slots=True)
class ClusterFit:
    pipeline_id: str
    status: str
    selected_k: int | None
    selected_score: float | None
    labels: NDArray[np.int64] | None
    centroids: NDArray[np.float64] | None
    eligible_mask: NDArray[np.bool_]
    local_scores: NDArray[np.float64] | None
    cluster_sizes: tuple[int, ...]
    dominant_cluster_id: int | None
    second_cluster_id: int | None
    dominant_centroid: NDArray[np.float64] | None
    second_centroid: NDArray[np.float64] | None
    k_records: tuple[dict[str, Any], ...]
    selected_k_equals_n: bool
    all_singleton_selected: bool
    tied_largest_selected: bool


def deterministic_seed(
    *parts: object, bits: int = 32, root_hex: str = CLUSTER_ROOT_HEX
) -> int:
    material = "\x1f".join([VERSION, root_hex, *map(str, parts)]).encode("utf-8")
    digest = hashlib.sha256(material).digest()
    if bits == 32:
        return int.from_bytes(digest[:4], "big")
    if bits == 128:
        return int.from_bytes(digest[:16], "big")
    if bits == 256:
        return int.from_bytes(digest, "big")
    raise ValueError("seed bits must be 32, 128, or 256")


def array_sha256(values: NDArray[Any]) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _pairwise_distance(values: NDArray[np.float64], metric: str) -> NDArray[np.float64]:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or len(x) == 0 or np.any(~np.isfinite(x)):
        raise ValueError("silhouette input must be a nonempty finite matrix")
    if metric == "cosine":
        norms = np.linalg.norm(x, axis=1)
        if np.any(norms <= 0):
            raise ValueError("cosine silhouette requires nonzero observations")
        similarity = (x / norms[:, None]) @ (x / norms[:, None]).T
        distances = 1.0 - np.clip(similarity, -1.0, 1.0)
    elif metric == "euclidean":
        delta = x[:, None, :] - x[None, :, :]
        distances = np.linalg.norm(delta, axis=2)
    else:
        raise ValueError(f"unsupported metric {metric!r}")
    if np.any(~np.isfinite(distances)):
        raise ValueError("pairwise distances must be finite")
    if float(np.min(distances)) < -COSINE_TOLERANCE:
        raise ValueError("materially negative distance")
    distances = np.maximum(distances, 0.0)
    np.fill_diagonal(distances, 0.0)
    return np.asarray(distances, dtype=np.float64)


def matlab_compatible_silhouette(
    values: NDArray[Any], labels: NDArray[Any], metric: str
) -> NDArray[np.float64]:
    """Clean-room silhouette with documented MATLAB singleton semantics.

    Singleton observations receive exactly 1.  For non-singletons, ``a`` is
    the mean distance to other members and ``b`` is the smallest mean distance
    to another cluster.  If all relevant distances are identically zero, the
    frozen value is 0 rather than propagating 0/0.
    """

    x = np.asarray(values, dtype=np.float64)
    raw_labels = np.asarray(labels)
    if raw_labels.ndim != 1 or len(raw_labels) != len(x):
        raise ValueError("labels must be one-dimensional and aligned")
    if any(value is None for value in raw_labels.tolist()):
        raise ValueError("cluster labels must be finite and nonmissing")
    unique = sorted(np.unique(raw_labels).tolist(), key=lambda value: str(value))
    if len(unique) < 2:
        raise ValueError("multi-cluster silhouette requires at least two clusters")
    remap = {value: index for index, value in enumerate(unique)}
    canonical = np.asarray(
        [remap[value] for value in raw_labels.tolist()], dtype=np.int64
    )
    distances = _pairwise_distance(x, metric)
    result = np.empty(len(x), dtype=np.float64)
    for index in range(len(x)):
        own = canonical == canonical[index]
        own_count = int(np.count_nonzero(own))
        if own_count == 1:
            result[index] = 1.0
            continue
        own_without_self = own.copy()
        own_without_self[index] = False
        a_value = float(np.mean(distances[index, own_without_self]))
        b_value = min(
            float(np.mean(distances[index, canonical == cluster_id]))
            for cluster_id in range(len(unique))
            if cluster_id != canonical[index]
        )
        denominator = max(a_value, b_value)
        result[index] = 0.0 if denominator == 0.0 else (b_value - a_value) / denominator
    if np.any(~np.isfinite(result)):
        raise ValueError("silhouette calculation produced nonfinite output")
    if np.any(result < -1.0 - COSINE_TOLERANCE) or np.any(
        result > 1.0 + COSINE_TOLERANCE
    ):
        raise ValueError("silhouette outside [-1, 1]")
    return np.clip(result, -1.0, 1.0)


def serialize_worker_exception(
    *,
    candidate_id: str,
    matrix_id: int,
    pipeline_id: str,
    k: int | None,
    n: int | None,
    cluster_sizes: tuple[int, ...] | list[int],
    seed_identity: str,
    error: BaseException,
) -> dict[str, Any]:
    return {
        "candidateId": candidate_id,
        "matrixId": int(matrix_id),
        "pipelineId": pipeline_id,
        "k": None if k is None else int(k),
        "n": None if n is None else int(n),
        "clusterSizeVector": [int(value) for value in cluster_sizes],
        "seedIdentity": str(seed_identity),
        "exceptionClass": type(error).__name__,
        "exceptionMessage": str(error),
    }


def _fit_historical_k(
    values: NDArray[np.float64], k: int, trajectory_identity: str
) -> tuple[
    NDArray[np.int64],
    NDArray[np.float64],
    float,
    NDArray[np.float64],
    tuple[float, ...],
    tuple[int, ...],
]:
    if k == 1:
        labels, centroids, loss, iterations = _cosine_kmeans_one(
            values, 1, deterministic_seed("R1", trajectory_identity, "k", 1, "rep", 0)
        )
        score = _historical_k1_score(values)
        return (
            labels,
            centroids,
            score,
            np.full(len(values), score),
            (loss,),
            (iterations,),
        )
    candidates: list[
        tuple[float, int, NDArray[np.int64], NDArray[np.float64], int]
    ] = []
    for replica in range(REPLICAS):
        labels, centroids, loss, iterations = _cosine_kmeans_one(
            values,
            k,
            deterministic_seed("R1", trajectory_identity, "k", k, "rep", replica),
        )
        candidates.append((loss, replica, labels, centroids, iterations))
    loss, _, labels, centroids, _ = min(candidates, key=lambda item: (item[0], item[1]))
    labels, centroids = _cluster_order(labels, centroids)
    local = matlab_compatible_silhouette(values, labels, "cosine")
    return (
        labels,
        centroids,
        float(np.mean(local)),
        local,
        tuple(float(item[0]) for item in candidates),
        tuple(int(item[4]) for item in candidates),
    )


def _fit_euclidean_k(
    values: NDArray[np.float64], k: int, trajectory_identity: str
) -> tuple[
    NDArray[np.int64] | None,
    NDArray[np.float64] | None,
    float | None,
    NDArray[np.float64] | None,
    tuple[float, ...],
    tuple[int, ...],
    str,
]:
    if k == 1:
        centroid = np.mean(values, axis=0, keepdims=True)
        return (
            np.zeros(len(values), dtype=np.int64),
            centroid,
            None,
            None,
            (float(np.sum((values - centroid) ** 2)),),
            (1,),
            "EVALUATED_SILHOUETTE_UNDEFINED_K1",
        )
    fits: list[tuple[float, int, NDArray[np.int64], NDArray[np.float64], int]] = []
    for replica in range(REPLICAS):
        model = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=1,
            max_iter=MAX_ITERATIONS,
            tol=EUCLIDEAN_TOLERANCE,
            algorithm="lloyd",
            random_state=deterministic_seed(
                "R2", trajectory_identity, "k", k, "rep", replica
            ),
        )
        labels = model.fit_predict(values).astype(np.int64)
        if len(np.unique(labels)) != k:
            continue
        labels, centroids = _cluster_order(
            labels, np.asarray(model.cluster_centers_, dtype=np.float64)
        )
        fits.append(
            (float(model.inertia_), replica, labels, centroids, int(model.n_iter_))
        )
    if not fits:
        return None, None, None, None, (), (), "INVALID_REALIZED_CLUSTER_COUNT"
    _, _, labels, centroids, _ = min(fits, key=lambda item: (item[0], item[1]))
    local = np.asarray(
        silhouette_samples(values, labels, metric="euclidean"), dtype=np.float64
    )
    return (
        labels,
        centroids,
        float(np.mean(local)),
        local,
        tuple(float(item[0]) for item in fits),
        tuple(int(item[4]) for item in fits),
        "ELIGIBLE",
    )


def scientific_recurrence_gate(
    labels: NDArray[Any], centroids: NDArray[Any]
) -> dict[str, Any]:
    assigned = np.asarray(labels, dtype=np.int64)
    center = np.asarray(centroids, dtype=np.float64)
    if assigned.ndim != 1 or center.ndim != 2 or len(center) == 0:
        raise ValueError("recurrence gate requires labels and centroids")
    cluster_sizes = tuple(
        int(np.count_nonzero(assigned == index)) for index in range(len(center))
    )
    maximum = max(cluster_sizes)
    all_singleton = maximum == 1
    largest = tuple(
        index for index, size in enumerate(cluster_sizes) if size == maximum
    )
    if all_singleton:
        return {
            "status": "NO_RECURRING_COMPTYPE",
            "clusterSizes": cluster_sizes,
            "dominantClusterId": None,
            "secondClusterId": None,
            "allSingleton": True,
            "tiedLargest": len(largest) > 1,
        }
    if len(largest) != 1:
        return {
            "status": "NO_UNIQUE_RECURRING_COMPTYPE",
            "clusterSizes": cluster_sizes,
            "dominantClusterId": None,
            "secondClusterId": None,
            "allSingleton": False,
            "tiedLargest": True,
        }
    dominant = int(largest[0])
    lower_sizes = sorted(
        {size for size in cluster_sizes if size < maximum and size >= 2}, reverse=True
    )
    second: int | None = None
    if lower_sizes:
        candidates = tuple(
            index for index, size in enumerate(cluster_sizes) if size == lower_sizes[0]
        )
        if len(candidates) == 1:
            second = int(candidates[0])
    return {
        "status": "ELIGIBLE",
        "clusterSizes": cluster_sizes,
        "dominantClusterId": dominant,
        "secondClusterId": second,
        "allSingleton": False,
        "tiedLargest": False,
    }


def _finalize_fit(
    *,
    pipeline_id: str,
    eligible_mask: NDArray[np.bool_],
    local_scores: NDArray[np.float64] | None,
    selected_k: int,
    selected_score: float,
    selected_labels: NDArray[np.int64],
    selected_centroids: NDArray[np.float64],
    k_records: list[dict[str, Any]],
) -> ClusterFit:
    gate = scientific_recurrence_gate(selected_labels, selected_centroids)
    dominant = gate["dominantClusterId"]
    second = gate["secondClusterId"]
    n = int(np.count_nonzero(eligible_mask))
    return ClusterFit(
        pipeline_id=pipeline_id,
        status=str(gate["status"]),
        selected_k=int(selected_k),
        selected_score=float(selected_score),
        labels=selected_labels,
        centroids=selected_centroids,
        eligible_mask=eligible_mask,
        local_scores=local_scores,
        cluster_sizes=tuple(gate["clusterSizes"]),
        dominant_cluster_id=dominant,
        second_cluster_id=second,
        dominant_centroid=None
        if dominant is None
        else np.asarray(selected_centroids[dominant], dtype=np.float64),
        second_centroid=None
        if second is None
        else np.asarray(selected_centroids[second], dtype=np.float64),
        k_records=tuple(k_records),
        selected_k_equals_n=bool(selected_k == n),
        all_singleton_selected=bool(gate["allSingleton"]),
        tied_largest_selected=bool(gate["tiedLargest"]),
    )


def _empty_fit(
    pipeline_id: str,
    status: str,
    eligible_mask: NDArray[np.bool_],
    local_scores: NDArray[np.float64] | None,
    records: tuple[dict[str, Any], ...] = (),
) -> ClusterFit:
    return ClusterFit(
        pipeline_id,
        status,
        None,
        None,
        None,
        None,
        eligible_mask,
        local_scores,
        (),
        None,
        None,
        None,
        None,
        records,
        False,
        False,
        False,
    )


def fit_r1_matlab_historical(
    boundary_compositions: NDArray[Any], trajectory_identity: str
) -> ClusterFit:
    values = close_rows(boundary_compositions)
    nondrift, _, local_scores = historical_nondrift_technique1(values)
    eligible = values[nondrift]
    if len(eligible) == 0:
        return _empty_fit(R1_ID, "NO_NONDRIFT_COMPOSITIONS", nondrift, local_scores)
    records: list[dict[str, Any]] = []
    fitted: list[
        tuple[int, float, NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]
    ] = []
    streak = 0
    best = -math.inf
    for k in K_VALUES:
        if k > len(eligible):
            records.append(
                {"k": k, "n": len(eligible), "status": "INELIGIBLE_K_EXCEEDS_POINTS"}
            )
            score = -math.inf
        else:
            labels, centroids, score, local, losses, iterations = _fit_historical_k(
                eligible, k, trajectory_identity
            )
            records.append(
                {
                    "k": k,
                    "n": len(eligible),
                    "status": "ELIGIBLE",
                    "selectionScore": score,
                    "localSilhouetteSha256": array_sha256(local),
                    "singletonCount": int(
                        sum(
                            np.count_nonzero(labels == index) == 1 for index in range(k)
                        )
                    ),
                    "clusterSizes": [
                        int(np.count_nonzero(labels == index)) for index in range(k)
                    ],
                    "replicaLosses": list(losses),
                    "replicaIterations": list(iterations),
                    "selectedLoss": min(losses),
                    "realizedClusterCount": len(centroids),
                }
            )
            fitted.append((k, score, labels, centroids, local))
        streak += 1
        if score >= best:
            best = score
            streak = 0
        if streak >= EARLY_STOP_STREAK:
            break
    if not fitted:
        return _empty_fit(
            R1_ID, "NO_VALID_CLUSTERING", nondrift, local_scores, tuple(records)
        )
    selected_k, score, labels, centroids, local = max(
        fitted, key=lambda item: (item[1], -item[0])
    )
    return _finalize_fit(
        pipeline_id=R1_ID,
        eligible_mask=nondrift,
        local_scores=local_scores,
        selected_k=selected_k,
        selected_score=score,
        selected_labels=labels,
        selected_centroids=centroids,
        k_records=records,
    )


def fit_r2_euclidean(
    boundary_compositions: NDArray[Any], trajectory_identity: str
) -> ClusterFit:
    values = close_rows(boundary_compositions)
    eligible_mask = np.ones(len(values), dtype=bool)
    records: list[dict[str, Any]] = []
    fitted: list[
        tuple[int, float, NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]
    ] = []
    for k in K_VALUES:
        labels, centroids, score, local, losses, iterations, status = _fit_euclidean_k(
            values, k, trajectory_identity
        )
        records.append(
            {
                "k": k,
                "n": len(values),
                "status": status,
                "selectionScore": score,
                "localSilhouetteSha256": None if local is None else array_sha256(local),
                "singletonCount": None
                if labels is None
                else int(
                    sum(np.count_nonzero(labels == index) == 1 for index in range(k))
                ),
                "clusterSizes": []
                if labels is None
                else [int(np.count_nonzero(labels == index)) for index in range(k)],
                "replicaLosses": list(losses),
                "replicaIterations": list(iterations),
                "selectedLoss": min(losses) if losses else None,
                "realizedClusterCount": None if centroids is None else len(centroids),
            }
        )
        if (
            score is not None
            and labels is not None
            and centroids is not None
            and local is not None
        ):
            fitted.append((k, score, labels, centroids, local))
    if not fitted:
        return _empty_fit(
            R2_ID, "NO_VALID_CLUSTERING", eligible_mask, None, tuple(records)
        )
    selected_k, score, labels, centroids, local = max(
        fitted, key=lambda item: (item[1], -item[0])
    )
    return _finalize_fit(
        pipeline_id=R2_ID,
        eligible_mask=eligible_mask,
        local_scores=None,
        selected_k=selected_k,
        selected_score=score,
        selected_labels=labels,
        selected_centroids=centroids,
        k_records=records,
    )


def fit_pipeline(
    pipeline_id: str, boundary_compositions: NDArray[Any], trajectory_identity: str
) -> ClusterFit:
    if pipeline_id == R1_ID:
        return fit_r1_matlab_historical(boundary_compositions, trajectory_identity)
    if pipeline_id == R2_ID:
        return fit_r2_euclidean(boundary_compositions, trajectory_identity)
    raise ValueError(f"unregistered pipeline {pipeline_id!r}")


def label_fingerprint(
    labels: NDArray[Any], generation_indices: NDArray[Any]
) -> dict[str, Any]:
    values = np.asarray(labels, dtype=bool)
    generations = np.asarray(generation_indices, dtype=np.int64)
    if values.ndim != 1 or generations.shape != values.shape or len(values) == 0:
        raise ValueError("fingerprint requires aligned nonempty arrays")
    positives = np.flatnonzero(values)
    onset0 = int(positives[0]) if len(positives) else None
    consistency: float | None = None
    if len(values) >= 3 and np.ptp(values.astype(np.int8)) > 0:
        with np.errstate(invalid="ignore", divide="ignore"):
            estimate = float(
                np.corrcoef(values[:-1].astype(float), values[1:].astype(float))[0, 1]
            )
        if np.isfinite(estimate):
            consistency = estimate
    positive_episodes = run_descriptors(values, True)
    negative_episodes = run_descriptors(values, False)
    fractions: dict[str, Any] = {}
    for percent in (10, 20, 25, 33):
        count = max(1, math.floor((percent / 100.0) * len(values)))
        fractions[f"nonreplicatingAt{percent}Percent"] = bool(not values[count - 1])
        fractions[f"noReplicatorThrough{percent}Percent"] = bool(
            not np.any(values[:count])
        )
    label_status = "ELIGIBLE"
    if np.all(values):
        label_status = "LABEL_CONSTANT_ONE"
    elif not np.any(values):
        label_status = "LABEL_CONSTANT_ZERO"
    consistency_status = (
        "DEFINED" if consistency is not None else "UNDEFINED_CONSISTENCY"
    )
    return {
        "fingerprintStatus": label_status,
        "selectedClockLength": len(values),
        "persistence": int(np.count_nonzero(values)),
        "occupancy": float(np.mean(values)),
        "consistency": consistency,
        "consistencyStatus": consistency_status,
        "firstOnsetRawIndex0": onset0,
        "firstOnsetRawStep1": None if onset0 is None else onset0 + 1,
        "firstOnsetNormalized": None
        if onset0 is None
        else float(onset0 / max(1, len(values) - 1)),
        "firstOnsetGeneration": None if onset0 is None else int(generations[onset0]),
        "preOnsetNonreplicatingDuration": int(
            len(values) if onset0 is None else onset0
        ),
        "transitionCount": int(np.count_nonzero(values[1:] != values[:-1])),
        "positiveEpisodeCount": len(positive_episodes),
        "negativeEpisodeCount": len(negative_episodes),
        "positiveMeanEpisodeDuration": float(
            np.mean([row["duration"] for row in positive_episodes])
        )
        if positive_episodes
        else None,
        "negativeMeanEpisodeDuration": float(
            np.mean([row["duration"] for row in negative_episodes])
        )
        if negative_episodes
        else None,
        "positiveLongestEpisodeDuration": max(
            (row["duration"] for row in positive_episodes), default=0
        ),
        "negativeLongestEpisodeDuration": max(
            (row["duration"] for row in negative_episodes), default=0
        ),
        "labelSha256": array_sha256(values.astype(np.int8)),
        **fractions,
    }


def paper_distance(
    summary: dict[str, Any], onset_mode: Literal["RAW", "NORMALIZED"]
) -> float | None:
    onset = "firstOnsetRawStep1" if onset_mode == "RAW" else "firstOnsetNormalized"
    metrics = ("selectedClockLength", "persistence", "occupancy", "consistency", onset)
    errors: list[float] = []
    for metric in metrics:
        value = summary.get(metric)
        if value is None or not np.isfinite(float(value)):
            return None
        target, scale = PAPER_TARGETS[metric]
        errors.append((float(value) - target) / scale)
    return float(np.sqrt(np.mean(np.square(errors))))


def bootstrap_indices(candidate_id: str, pipeline_id: str) -> NDArray[np.int64]:
    seed = deterministic_seed(
        "bootstrap", candidate_id, pipeline_id, bits=128, root_hex=BOOTSTRAP_ROOT_HEX
    )
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    return rng.integers(0, 100, size=(BOOTSTRAP_REPLICATES, 100), dtype=np.int64)


def holm_adjust(values: list[float]) -> list[float]:
    if not values:
        return []
    raw = np.asarray(values, dtype=np.float64)
    order = np.argsort(raw, kind="stable")
    adjusted = np.empty_like(raw)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(raw) - rank) * float(raw[index])))
        adjusted[index] = running
    return adjusted.tolist()
