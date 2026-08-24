"""Outcome-blind contracts for E01/S19-L09.

This module implements exactly two registered dominant-recurring-composition
pipelines.  It never simulates GARD and contains no emergence, prediction, or
intervention code.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

VERSION = "E01-S19-L09-RECURRING-ATTRACTOR-LABEL-RECONSTRUCTION-v1.0.0"
LOOP_ID = "S19-L09"
R1_ID = "R1_HISTORICAL_DOMINANT_COMPTYPE_H090"
R2_ID = "R2_PAPER_EUCLIDEAN_DOMINANT_ATTRACTOR_H090"
LABEL_IDS = (R1_ID, R2_ID)
THRESHOLD = 0.9
K_VALUES = tuple(range(1, 11))
REPLICAS = 10
EARLY_STOP_STREAK = 4
MAX_ITERATIONS = 300
COSINE_TOLERANCE = 1e-10
EUCLIDEAN_TOLERANCE = 1e-4
MINIMUM_VALID_CLUSTER_SIZE = 2
MINIMUM_REFERENCE_MEMBERSHIP_VISITS = 2
RANDOM_REFERENCE_DRAWS = 64
BOOTSTRAP_REPLICATES = 4096
ROOT_SEED_HEX = "1af2599e7e4c6a35e6b4bb9fc11f813484157798160a304c4012a3e26e564f3d"

PAPER_TARGETS: dict[str, tuple[float, float]] = {
    "occupancy": (0.88, 0.03),
    "persistence": (716.0, 198.0),
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
    valid_cluster_ids: tuple[int, ...]
    dominant_cluster_id: int | None
    second_cluster_id: int | None
    dominant_centroid: NDArray[np.float64] | None
    second_centroid: NDArray[np.float64] | None
    dominant_member_count: int
    second_member_count: int
    k_records: tuple[dict[str, Any], ...]


def deterministic_seed(*parts: object, bits: int = 32) -> int:
    material = "\x1f".join([VERSION, ROOT_SEED_HEX, *map(str, parts)]).encode("utf-8")
    digest = hashlib.sha256(material).digest()
    if bits == 32:
        return int.from_bytes(digest[:4], "big")
    if bits == 128:
        return int.from_bytes(digest[:16], "big")
    raise ValueError("seed bits must be 32 or 128")


def array_sha256(values: NDArray[Any]) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def close_rows(states: NDArray[Any]) -> NDArray[np.float64]:
    values = np.asarray(states, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or np.any(~np.isfinite(values)):
        raise ValueError("compositions must be a nonempty finite matrix")
    if np.any(values < 0):
        raise ValueError("compositions must be nonnegative")
    totals = np.sum(values, axis=1)
    if np.any(totals <= 0):
        raise ValueError("zero-mass compositions are ineligible")
    return np.ascontiguousarray(values / totals[:, None], dtype=np.float64)


def historical_h(left: NDArray[Any], right: NDArray[Any]) -> NDArray[np.float64]:
    """Pinned tgs_H semantics: column-vector cosine clipped to [0, 1]."""

    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.ndim == 1:
        a = a[None, :]
    if b.ndim == 1:
        b = b[None, :]
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("H operands must have matching feature dimensions")
    an = np.maximum(np.linalg.norm(a, axis=1), 1e-7)
    bn = np.maximum(np.linalg.norm(b, axis=1), 1e-7)
    values = (a / an[:, None]) @ (b / bn[:, None]).T
    return np.clip(values, 0.0, 1.0)


def historical_nondrift_technique1(
    boundary_compositions: NDArray[Any], threshold: float = THRESHOLD
) -> tuple[NDArray[np.bool_], NDArray[np.float64], NDArray[np.float64]]:
    """Exact vectorized reconstruction of pinned tgs_nondrift technique 1."""

    values = np.asarray(boundary_compositions, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("historical technique 1 requires at least two boundaries")
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms <= 0) or np.any(~np.isfinite(values)):
        raise ValueError("historical technique 1 requires finite nonzero states")
    normalized = values / norms[:, None]
    adjacent = np.sum(normalized[:-1] * normalized[1:], axis=1)
    angles = np.concatenate(([adjacent[0]], adjacent)).astype(np.float64)
    local = (
        np.concatenate(([adjacent[0]], adjacent))
        + np.concatenate((adjacent, [adjacent[-1]]))
    ) / 2.0
    return local > float(threshold), angles, local


def _cosine_kmeans_one(
    values: NDArray[np.float64], k: int, seed: int
) -> tuple[NDArray[np.int64], NDArray[np.float64], float, int]:
    """Deterministic spherical k-means reconstruction of MATLAB cosine kmeans."""

    n, _ = values.shape
    unit = values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-15)
    if k == 1:
        centroid = np.mean(values, axis=0, keepdims=True)
        centroid /= np.maximum(np.linalg.norm(centroid, axis=1, keepdims=True), 1e-15)
        distance = 1.0 - (unit @ centroid.T).ravel()
        return np.zeros(n, dtype=np.int64), centroid, float(np.sum(distance)), 1

    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    chosen = [int(rng.integers(0, n))]
    min_distance = 1.0 - (unit @ unit[chosen[0]])
    for _ in range(1, k):
        weights = np.maximum(min_distance, 0.0) ** 2
        if float(np.sum(weights)) <= 0:
            remaining = [index for index in range(n) if index not in chosen]
            chosen.append(int(remaining[0]))
        else:
            chosen.append(int(rng.choice(n, p=weights / np.sum(weights))))
        min_distance = np.minimum(min_distance, 1.0 - unit @ unit[chosen[-1]])
    centroids = unit[np.asarray(chosen, dtype=np.int64)].copy()
    labels = np.full(n, -1, dtype=np.int64)
    iterations = 0
    for iterations in range(1, MAX_ITERATIONS + 1):
        similarities = unit @ centroids.T
        next_labels = np.argmax(similarities, axis=1).astype(np.int64)
        counts = np.bincount(next_labels, minlength=k)
        if np.any(counts == 0):
            assigned_similarity = similarities[np.arange(n), next_labels]
            for empty in np.flatnonzero(counts == 0):
                donor = int(np.argmin(assigned_similarity))
                old = int(next_labels[donor])
                next_labels[donor] = int(empty)
                counts[old] -= 1
                counts[int(empty)] += 1
                assigned_similarity[donor] = math.inf
        next_centroids = np.empty_like(centroids)
        for cluster_id in range(k):
            mean = np.mean(values[next_labels == cluster_id], axis=0)
            norm = float(np.linalg.norm(mean))
            if norm <= 0:
                raise ValueError("zero cosine centroid")
            next_centroids[cluster_id] = mean / norm
        shift = float(np.max(1.0 - np.sum(centroids * next_centroids, axis=1)))
        converged = np.array_equal(labels, next_labels) or shift <= COSINE_TOLERANCE
        labels, centroids = next_labels, next_centroids
        if converged:
            break
    loss = float(np.sum(1.0 - (unit @ centroids.T)[np.arange(n), labels]))
    return labels, centroids, loss, iterations


def _historical_k1_score(values: NDArray[np.float64]) -> float:
    # tgs_kmeans calls mean(tgs_carpet(trace,'none')(:)) for k=1.
    return float(np.mean(historical_h(values, values)))


def _cluster_order(labels: NDArray[np.int64], centroids: NDArray[np.float64]) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Canonicalize arbitrary cluster IDs by earliest assigned observation."""

    ids = sorted(np.unique(labels).tolist(), key=lambda item: int(np.flatnonzero(labels == item)[0]))
    remap = {old: new for new, old in enumerate(ids)}
    ordered_labels = np.asarray([remap[int(value)] for value in labels], dtype=np.int64)
    ordered_centroids = np.asarray([centroids[old] for old in ids], dtype=np.float64)
    return ordered_labels, ordered_centroids


def _fit_historical_k(
    values: NDArray[np.float64], k: int, trajectory_identity: str
) -> tuple[NDArray[np.int64], NDArray[np.float64], float, tuple[float, ...], tuple[int, ...]]:
    if k == 1:
        labels, centroids, loss, iterations = _cosine_kmeans_one(
            values, 1, deterministic_seed("R1", trajectory_identity, "k", 1, "rep", 0)
        )
        return labels, centroids, _historical_k1_score(values), (loss,), (iterations,)
    candidates: list[tuple[float, int, NDArray[np.int64], NDArray[np.float64], int]] = []
    for replica in range(REPLICAS):
        labels, centroids, loss, iterations = _cosine_kmeans_one(
            values,
            k,
            deterministic_seed("R1", trajectory_identity, "k", k, "rep", replica),
        )
        candidates.append((loss, replica, labels, centroids, iterations))
    loss, _, labels, centroids, _ = min(candidates, key=lambda item: (item[0], item[1]))
    labels, centroids = _cluster_order(labels, centroids)
    score = float(silhouette_score(values, labels, metric="cosine"))
    return (
        labels,
        centroids,
        score,
        tuple(float(item[0]) for item in candidates),
        tuple(int(item[4]) for item in candidates),
    )


def _fit_euclidean_k(
    values: NDArray[np.float64], k: int, trajectory_identity: str
) -> tuple[NDArray[np.int64] | None, NDArray[np.float64] | None, float | None, tuple[float, ...], tuple[int, ...], str]:
    if k == 1:
        centroid = np.mean(values, axis=0, keepdims=True)
        return (
            np.zeros(len(values), dtype=np.int64),
            centroid,
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
        labels, centroids = _cluster_order(labels, np.asarray(model.cluster_centers_))
        fits.append((float(model.inertia_), replica, labels, centroids, int(model.n_iter_)))
    if not fits:
        return None, None, None, (), (), "INVALID_REALIZED_CLUSTER_COUNT"
    inertia, _, labels, centroids, _ = min(fits, key=lambda item: (item[0], item[1]))
    score = float(silhouette_score(values, labels, metric="euclidean"))
    return (
        labels,
        centroids,
        score,
        tuple(float(item[0]) for item in fits),
        tuple(int(item[4]) for item in fits),
        "ELIGIBLE",
    )


def _finalize_cluster_fit(
    *,
    pipeline_id: str,
    all_boundary_values: NDArray[np.float64],
    eligible_mask: NDArray[np.bool_],
    selected_k: int,
    selected_score: float,
    selected_labels: NDArray[np.int64],
    selected_centroids: NDArray[np.float64],
    local_scores: NDArray[np.float64] | None,
    k_records: list[dict[str, Any]],
) -> ClusterFit:
    eligible_values = all_boundary_values[eligible_mask]
    sizes = tuple(
        int(np.count_nonzero(selected_labels == cluster_id))
        for cluster_id in range(len(selected_centroids))
    )
    member_counts = tuple(
        int(np.count_nonzero(historical_h(eligible_values, centroid).ravel() > THRESHOLD))
        for centroid in selected_centroids
    )
    valid = tuple(
        cluster_id
        for cluster_id, (size, membership) in enumerate(zip(sizes, member_counts, strict=True))
        if size >= MINIMUM_VALID_CLUSTER_SIZE
        and membership >= MINIMUM_REFERENCE_MEMBERSHIP_VISITS
    )
    if not valid:
        return ClusterFit(
            pipeline_id,
            "INELIGIBLE_NO_VALID_RECURRING_CLUSTER",
            selected_k,
            selected_score,
            selected_labels,
            selected_centroids,
            eligible_mask,
            local_scores,
            sizes,
            valid,
            None,
            None,
            None,
            None,
            0,
            0,
            tuple(k_records),
        )
    ranked = sorted(valid, key=lambda cluster_id: (-sizes[cluster_id], cluster_id))
    dominant = int(ranked[0])
    second = int(ranked[1]) if len(ranked) > 1 else None
    return ClusterFit(
        pipeline_id,
        "ELIGIBLE",
        selected_k,
        selected_score,
        selected_labels,
        selected_centroids,
        eligible_mask,
        local_scores,
        sizes,
        valid,
        dominant,
        second,
        np.asarray(selected_centroids[dominant], dtype=np.float64),
        None if second is None else np.asarray(selected_centroids[second], dtype=np.float64),
        int(member_counts[dominant]),
        0 if second is None else int(member_counts[second]),
        tuple(k_records),
    )


def fit_r1_historical(
    boundary_compositions: NDArray[Any], trajectory_identity: str
) -> ClusterFit:
    values = close_rows(boundary_compositions)
    nondrift, _, local = historical_nondrift_technique1(values)
    eligible_values = values[nondrift]
    if len(eligible_values) == 0:
        return ClusterFit(
            R1_ID,
            "INELIGIBLE_NO_NONDRIFT_BOUNDARIES",
            None,
            None,
            None,
            None,
            nondrift,
            local,
            (),
            (),
            None,
            None,
            None,
            None,
            0,
            0,
            (),
        )
    records: list[dict[str, Any]] = []
    fitted: list[tuple[int, float, NDArray[np.int64], NDArray[np.float64]]] = []
    streak = 0
    best_score = -math.inf
    for k in K_VALUES:
        if k > len(eligible_values):
            records.append({"k": k, "status": "INELIGIBLE_K_EXCEEDS_POINTS"})
            score = -math.inf
        else:
            labels, centroids, score, losses, iterations = _fit_historical_k(
                eligible_values, k, trajectory_identity
            )
            records.append(
                {
                    "k": k,
                    "status": "ELIGIBLE",
                    "selectionScore": score,
                    "replicaLosses": list(losses),
                    "replicaIterations": list(iterations),
                    "selectedLoss": min(losses),
                    "realizedClusterCount": int(len(centroids)),
                }
            )
            fitted.append((k, score, labels, centroids))
        streak += 1
        if score >= best_score:
            best_score = score
            streak = 0
        if streak >= EARLY_STOP_STREAK:
            break
    if not fitted:
        raise RuntimeError("R1 did not produce any evaluated k")
    selected_k, score, labels, centroids = max(fitted, key=lambda item: (item[1], -item[0]))
    return _finalize_cluster_fit(
        pipeline_id=R1_ID,
        all_boundary_values=values,
        eligible_mask=nondrift,
        selected_k=selected_k,
        selected_score=score,
        selected_labels=labels,
        selected_centroids=centroids,
        local_scores=local,
        k_records=records,
    )


def fit_r2_euclidean(
    boundary_compositions: NDArray[Any], trajectory_identity: str
) -> ClusterFit:
    values = close_rows(boundary_compositions)
    eligible = np.ones(len(values), dtype=bool)
    records: list[dict[str, Any]] = []
    fitted: list[tuple[int, float, NDArray[np.int64], NDArray[np.float64]]] = []
    for k in K_VALUES:
        labels, centroids, score, losses, iterations, status = _fit_euclidean_k(
            values, k, trajectory_identity
        )
        records.append(
            {
                "k": k,
                "status": status,
                "selectionScore": score,
                "replicaLosses": list(losses),
                "replicaIterations": list(iterations),
                "selectedLoss": min(losses) if losses else None,
                "realizedClusterCount": None if centroids is None else int(len(centroids)),
            }
        )
        if score is not None and labels is not None and centroids is not None:
            fitted.append((k, score, labels, centroids))
    if not fitted:
        return ClusterFit(
            R2_ID,
            "INELIGIBLE_NO_FINITE_SILHOUETTE_SOLUTION",
            None,
            None,
            None,
            None,
            eligible,
            None,
            (),
            (),
            None,
            None,
            None,
            None,
            0,
            0,
            tuple(records),
        )
    selected_k, score, labels, centroids = max(fitted, key=lambda item: (item[1], -item[0]))
    return _finalize_cluster_fit(
        pipeline_id=R2_ID,
        all_boundary_values=values,
        eligible_mask=eligible,
        selected_k=selected_k,
        selected_score=score,
        selected_labels=labels,
        selected_centroids=centroids,
        local_scores=None,
        k_records=records,
    )


def label_against_reference(
    compositions: NDArray[Any], reference: NDArray[Any]
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    closed = close_rows(compositions)
    raw_ref = np.asarray(reference, dtype=np.float64).reshape(1, -1)
    # Arithmetic centroids of nonnegative compositions can contain tiny
    # negative coordinates from floating subtraction in Lloyd updates.  Clamp
    # only machine-scale residue under the frozen numerical contract; any
    # material negative remains an error.
    if np.any(raw_ref < -1e-12):
        raise ValueError("reference contains a material negative coordinate")
    ref = close_rows(np.maximum(raw_ref, 0.0))[0]
    scores = historical_h(closed, ref).ravel()
    return scores, scores > THRESHOLD


def run_descriptors(labels: NDArray[Any], desired: bool) -> list[dict[str, int]]:
    sequence = np.asarray(labels, dtype=bool)
    mask = sequence if desired else ~sequence
    padded = np.concatenate(([False], mask, [False]))
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return [
        {"startIndex0": int(start), "endIndex0": int(end - 1), "duration": int(end - start)}
        for start, end in zip(starts, ends, strict=True)
    ]


def label_fingerprint(
    labels: NDArray[Any], generation_indices: NDArray[Any]
) -> dict[str, Any]:
    values = np.asarray(labels, dtype=bool)
    generations = np.asarray(generation_indices, dtype=np.int64)
    if values.ndim != 1 or generations.shape != values.shape or len(values) == 0:
        raise ValueError("fingerprint requires aligned nonempty one-dimensional arrays")
    positives = np.flatnonzero(values)
    onset0 = int(positives[0]) if len(positives) else None
    consistency: float | None = None
    if len(values) >= 3 and np.ptp(values.astype(np.int8)) > 0:
        result = float(np.corrcoef(values[:-1].astype(float), values[1:].astype(float))[0, 1])
        if np.isfinite(result):
            consistency = result
    positive_episodes = run_descriptors(values, True)
    negative_episodes = run_descriptors(values, False)
    cutoff_count = int(math.floor(0.25 * len(values)))
    cutoff_index = max(0, cutoff_count - 1)
    return {
        "fingerprintStatus": "ELIGIBLE",
        "selectedClockLength": int(len(values)),
        "persistence": int(np.count_nonzero(values)),
        "occupancy": float(np.mean(values)),
        "consistency": consistency,
        "consistencyStatus": "DEFINED" if consistency is not None else "UNDEFINED_CONSTANT_LABEL",
        "firstOnsetRawIndex0": onset0,
        "firstOnsetRawStep1": None if onset0 is None else onset0 + 1,
        "firstOnsetNormalized": None if onset0 is None else float(onset0 / max(1, len(values) - 1)),
        "firstOnsetGeneration": None if onset0 is None else int(generations[onset0]),
        "preOnsetNonreplicatingDuration": int(len(values) if onset0 is None else onset0),
        "isNonreplicatingAtQuarterCutoff": bool(not values[cutoff_index]),
        "noReplicatorThroughQuarterCutoff": bool(not np.any(values[:cutoff_count])),
        "positiveEpisodeCount": int(len(positive_episodes)),
        "negativeEpisodeCount": int(len(negative_episodes)),
        "transitionCount": int(np.count_nonzero(values[1:] != values[:-1])),
        "positiveMeanEpisodeDuration": (
            float(np.mean([row["duration"] for row in positive_episodes]))
            if positive_episodes
            else None
        ),
        "negativeMeanEpisodeDuration": (
            float(np.mean([row["duration"] for row in negative_episodes]))
            if negative_episodes
            else None
        ),
        "positiveLongestEpisodeDuration": max(
            (row["duration"] for row in positive_episodes), default=0
        ),
        "negativeLongestEpisodeDuration": max(
            (row["duration"] for row in negative_episodes), default=0
        ),
        "labelSha256": array_sha256(values.astype(np.int8)),
    }


def paper_distance(summary: dict[str, Any], onset_mode: Literal["RAW", "NORMALIZED"]) -> float | None:
    onset_key = "firstOnsetRawStep1" if onset_mode == "RAW" else "firstOnsetNormalized"
    keys = ("occupancy", "persistence", "consistency", onset_key)
    errors: list[float] = []
    for key in keys:
        value = summary.get(key)
        if value is None or not np.isfinite(float(value)):
            return None
        target, scale = PAPER_TARGETS[key]
        errors.append((float(value) - target) / scale)
    return float(np.sqrt(np.mean(np.square(errors))))


def bootstrap_indices(candidate_id: str, pipeline_id: str) -> NDArray[np.int64]:
    rng = np.random.Generator(
        np.random.PCG64DXSM(
            deterministic_seed("bootstrap", candidate_id, pipeline_id, bits=128)
        )
    )
    return rng.integers(0, 100, size=(BOOTSTRAP_REPLICATES, 100), dtype=np.int64)


def holm_adjust(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    raw = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(raw, kind="stable")
    adjusted = np.empty_like(raw)
    running = 0.0
    count = len(raw)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * float(raw[index])))
        adjusted[index] = running
    return adjusted.tolist()
