"""Frozen Phase-2 self-replicator label candidates for E01 S12E."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .core import GardTrajectory

LABEL_IDS = (
    "L0_HISTORICAL_ADJACENT_H090",
    "L1_DOMINANT_CENTROID_H090",
    "L2_EUCLIDEAN_COMPTYPE",
    "L3_EUCLIDEAN_DOMINANT_CENTROID",
)


@dataclass(frozen=True, slots=True)
class KMeansSelection:
    status: str
    k: int | None
    replica: int | None
    silhouette: float | None
    cluster_labels: tuple[int, ...]
    persistent_cluster_ids: tuple[int, ...]
    dominant_cluster_id: int | None
    centroid: tuple[float, ...] | None
    radius: float | None


@dataclass(frozen=True, slots=True)
class LabelResult:
    trajectory_id: str
    engine_id: str
    label_id: str
    status: str
    reason: str | None
    observation_labels: tuple[bool, ...]
    generation_labels: tuple[bool, ...]
    generation_cluster_ids: tuple[int | None, ...]
    persistent_cluster_count: int
    dominant_cluster_size: int
    centroid: tuple[float, ...] | None
    radius: float | None
    kmeans: KMeansSelection | None


def relative_compositions(states: NDArray[np.int64]) -> NDArray[np.float64]:
    values = np.asarray(states, dtype=np.float64)
    masses = values.sum(axis=1)
    result = np.full_like(values, np.nan, dtype=np.float64)
    nonempty = masses > 0
    result[nonempty] = values[nonempty] / masses[nonempty, None]
    return result


def h_similarity(left: NDArray[np.float64], right: NDArray[np.float64]) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(left, right) / denom)


def h_matrix(compositions: NDArray[np.float64]) -> NDArray[np.float64]:
    norms = np.linalg.norm(compositions, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = compositions / norms[:, None]
    return normalized @ normalized.T


def historical_adjacent_labels(
    compositions: NDArray[np.float64], threshold: float = 0.9
) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
    """Reproduce public historical non-drift technique 1 on row observations."""

    n = compositions.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=np.float64)
    if n == 1:
        return np.zeros(1, dtype=bool), np.full(1, np.nan)
    norms = np.linalg.norm(compositions, axis=1)
    normalized = compositions / norms[:, None]
    adjacent = np.sum(normalized[:-1] * normalized[1:], axis=1)
    padded_left = np.concatenate(([adjacent[0]], adjacent))
    padded_right = np.concatenate((adjacent, [adjacent[-1]]))
    score = (padded_left + padded_right) / 2.0
    return score > threshold, score


def connected_components(adjacency: NDArray[np.bool_]) -> list[tuple[int, ...]]:
    remaining = set(range(adjacency.shape[0]))
    output: list[tuple[int, ...]] = []
    while remaining:
        root = min(remaining)
        stack = [root]
        found: set[int] = set()
        while stack:
            item = stack.pop()
            if item in found:
                continue
            found.add(item)
            stack.extend(
                int(value)
                for value in np.flatnonzero(adjacency[item])
                if int(value) not in found
            )
        remaining.difference_update(found)
        output.append(tuple(sorted(found)))
    return output


def _post_fission_indices(trajectory: GardTrajectory) -> NDArray[np.int64]:
    return np.asarray(
        [
            row.observation_index
            for row in trajectory.observations
            if row.observation_kind == "post_fission"
        ],
        dtype=np.int64,
    )


def _propagate_generation_labels(
    trajectory: GardTrajectory, generation_labels: NDArray[np.bool_]
) -> NDArray[np.bool_]:
    output = np.zeros(len(trajectory.observations), dtype=bool)
    for index, row in enumerate(trajectory.observations):
        if row.generation > 0 and row.generation <= generation_labels.size:
            output[index] = bool(generation_labels[row.generation - 1])
    return output


def _episode_summary(binary: NDArray[np.bool_]) -> tuple[int, float, int, int]:
    if binary.size == 0:
        return 0, 0.0, 0, 0
    starts = np.flatnonzero(binary & ~np.concatenate(([False], binary[:-1])))
    ends = np.flatnonzero(binary & ~np.concatenate((binary[1:], [False])))
    lengths = ends - starts + 1
    longest = int(lengths.max()) if lengths.size else 0
    mean = float(lengths.mean()) if lengths.size else 0.0
    entries = int(starts.size)
    exits = int(np.count_nonzero(binary[:-1] & ~binary[1:]))
    return longest, mean, entries, exits


def _direct_centroid_labels(
    compositions: NDArray[np.float64],
    centroid: NDArray[np.float64],
    *,
    similarity_threshold: float | None = None,
    radius: float | None = None,
) -> NDArray[np.bool_]:
    output = np.zeros(compositions.shape[0], dtype=bool)
    finite = np.all(np.isfinite(compositions), axis=1)
    if similarity_threshold is not None:
        centroid_norm = np.linalg.norm(centroid)
        norms = np.linalg.norm(compositions, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            similarity = (compositions @ centroid) / (norms * centroid_norm)
        output[finite] = similarity[finite] >= similarity_threshold
    else:
        if radius is None:
            raise ValueError("radius is required for Euclidean centroid labels")
        distances = np.linalg.norm(compositions - centroid[None, :], axis=1)
        output[finite] = distances[finite] <= radius
    return output


def _kmeans_selection(
    compositions: NDArray[np.float64],
    nondrift: NDArray[np.bool_],
    seed_for: Callable[[int, int], int],
) -> KMeansSelection:
    indices = np.flatnonzero(nondrift)
    data = compositions[indices]
    if data.shape[0] == 0:
        return KMeansSelection(
            "NO_NONDRIFT_GENERATIONS", None, None, None, (), (), None, None, None
        )
    candidates: list[tuple[float, int, int, NDArray[np.int64]]] = []
    for k in range(2, min(10, data.shape[0] - 1) + 1):
        for replica in range(10):
            model = KMeans(
                n_clusters=k,
                n_init=1,
                algorithm="lloyd",
                random_state=seed_for(k, replica),
            )
            labels = model.fit_predict(data).astype(np.int64, copy=False)
            if np.unique(labels).size != k:
                continue
            try:
                score = float(silhouette_score(data, labels, metric="euclidean"))
            except ValueError:
                continue
            if np.isfinite(score):
                candidates.append((score, k, replica, labels.copy()))
    positive = [row for row in candidates if row[0] > 0.0]
    if positive:
        score, k, replica, selected = min(
            positive, key=lambda row: (-row[0], row[1], row[2])
        )
    else:
        score, k, replica = 0.0, 1, 0
        selected = np.zeros(data.shape[0], dtype=np.int64)
    full = np.full(compositions.shape[0], -1, dtype=np.int64)
    full[indices] = selected
    counts = {
        int(cluster): int(np.count_nonzero(selected == cluster))
        for cluster in np.unique(selected)
    }
    persistent = tuple(sorted(cluster for cluster, count in counts.items() if count >= 3))
    dominant: int | None = None
    centroid: tuple[float, ...] | None = None
    radius: float | None = None
    if persistent:
        dominant = min(persistent, key=lambda cluster: (-counts[cluster], cluster))
        members = data[selected == dominant]
        center = members.mean(axis=0)
        center /= center.sum()
        distances = np.linalg.norm(members - center[None, :], axis=1)
        median = float(np.median(distances))
        mad = float(np.median(np.abs(distances - median)))
        cap = median + 3.0 * 1.4826 * mad
        radius = float(distances.max()) if mad == 0.0 else float(min(distances.max(), cap))
        centroid = tuple(map(float, center))
    return KMeansSelection(
        status="ELIGIBLE",
        k=k,
        replica=replica,
        silhouette=score,
        cluster_labels=tuple(map(int, full)),
        persistent_cluster_ids=persistent,
        dominant_cluster_id=dominant,
        centroid=centroid,
        radius=radius,
    )


def label_trajectory(
    trajectory: GardTrajectory,
    label_id: str,
    *,
    kmeans_seed_for: Callable[[int, int], int],
) -> LabelResult:
    if label_id not in LABEL_IDS:
        raise ValueError(f"unknown S12E label {label_id!r}")
    states = trajectory.states
    compositions = relative_compositions(states)
    post_indices = _post_fission_indices(trajectory)
    post = compositions[post_indices]
    if post.shape[0] == 0 or not np.all(np.isfinite(post)):
        return LabelResult(
            trajectory.trajectory_id,
            trajectory.engine_id,
            label_id,
            "INELIGIBLE_POST_FISSION_SUBSTRATE",
            "no_finite_post_fission_compositions",
            tuple(False for _ in trajectory.observations),
            (),
            (),
            0,
            0,
            None,
            None,
            None,
        )

    kmeans: KMeansSelection | None = None
    centroid: tuple[float, ...] | None = None
    radius: float | None = None
    cluster_ids: tuple[int | None, ...]
    persistent_count = 0
    dominant_size = 0

    if label_id == "L0_HISTORICAL_ADJACENT_H090":
        generation_labels, _ = historical_adjacent_labels(post, 0.9)
        observation_labels = _propagate_generation_labels(trajectory, generation_labels)
        cluster_ids = tuple(None for _ in range(post.shape[0]))
    elif label_id == "L1_DOMINANT_CENTROID_H090":
        similarity = h_matrix(post)
        adjacency = similarity > 0.9
        np.fill_diagonal(adjacency, False)
        components = connected_components(adjacency)
        dominant = min(components, key=lambda item: (-len(item), min(item)))
        center = post[list(dominant)].mean(axis=0)
        center /= center.sum()
        centroid = tuple(map(float, center))
        observation_labels = _direct_centroid_labels(
            compositions, center, similarity_threshold=0.9
        )
        generation_labels = observation_labels[post_indices]
        component_by_index: dict[int, int] = {}
        for component_id, members in enumerate(components):
            for member in members:
                component_by_index[member] = component_id
        cluster_ids = tuple(component_by_index[index] for index in range(post.shape[0]))
        persistent_count = len(components)
        dominant_size = len(dominant)
    else:
        nondrift, _ = historical_adjacent_labels(post, 0.9)
        kmeans = _kmeans_selection(post, nondrift, kmeans_seed_for)
        full_clusters = np.asarray(kmeans.cluster_labels, dtype=np.int64)
        persistent = set(kmeans.persistent_cluster_ids)
        generation_labels = np.asarray(
            [int(value) in persistent for value in full_clusters], dtype=bool
        )
        cluster_ids = tuple(None if value < 0 else int(value) for value in full_clusters)
        persistent_count = len(persistent)
        if kmeans.dominant_cluster_id is not None:
            dominant_size = int(
                np.count_nonzero(full_clusters == kmeans.dominant_cluster_id)
            )
        if label_id == "L2_EUCLIDEAN_COMPTYPE":
            observation_labels = _propagate_generation_labels(
                trajectory, generation_labels
            )
        else:
            if kmeans.centroid is None or kmeans.radius is None:
                observation_labels = np.zeros(len(trajectory.observations), dtype=bool)
                generation_labels = np.zeros(post.shape[0], dtype=bool)
            else:
                center = np.asarray(kmeans.centroid, dtype=np.float64)
                centroid = kmeans.centroid
                radius = kmeans.radius
                observation_labels = _direct_centroid_labels(
                    compositions, center, radius=radius
                )
                generation_labels = observation_labels[post_indices]

    status = "ELIGIBLE"
    reason = None
    return LabelResult(
        trajectory_id=trajectory.trajectory_id,
        engine_id=trajectory.engine_id,
        label_id=label_id,
        status=status,
        reason=reason,
        observation_labels=tuple(map(bool, observation_labels)),
        generation_labels=tuple(map(bool, generation_labels)),
        generation_cluster_ids=cluster_ids,
        persistent_cluster_count=persistent_count,
        dominant_cluster_size=dominant_size,
        centroid=centroid,
        radius=radius,
        kmeans=kmeans,
    )


def label_fingerprint(
    trajectory: GardTrajectory, result: LabelResult
) -> dict[str, object]:
    binary_all = np.asarray(result.observation_labels, dtype=bool)
    update_indices = np.asarray(
        [
            index
            for index, row in enumerate(trajectory.observations)
            if row.observation_kind == "molecular_update"
        ],
        dtype=np.int64,
    )
    binary = binary_all[update_indices]
    steps = np.asarray(
        [trajectory.observations[index].molecular_step for index in update_indices],
        dtype=np.int64,
    )
    persistence = int(binary.sum())
    fraction = float(binary.mean()) if binary.size else np.nan
    first = int(steps[np.flatnonzero(binary)[0]]) if np.any(binary) else np.nan
    longest, mean_episode, entries, exits = _episode_summary(binary)
    if binary.size >= 3 and np.unique(binary).size > 1:
        consistency = float(np.corrcoef(binary[:-1].astype(float), binary[1:].astype(float))[0, 1])
    else:
        consistency = np.nan
    generation = np.asarray(result.generation_labels, dtype=bool)
    return {
        "trajectoryId": trajectory.trajectory_id,
        "phase": trajectory.phase,
        "matrixIndex": trajectory.matrix_index,
        "engineId": trajectory.engine_id,
        "labelId": result.label_id,
        "status": result.status,
        "reason": result.reason,
        "replicatingLifetime": persistence,
        "replicatingFraction": fraction,
        "firstReplicatorBatchStep": first,
        "longestEpisode": longest,
        "meanEpisode": mean_episode,
        "entries": entries,
        "exits": exits,
        "consecutiveBinaryPearson": consistency,
        "replicatingGenerationFraction": (
            float(generation.mean()) if generation.size else np.nan
        ),
        "persistentClusterCount": result.persistent_cluster_count,
        "dominantClusterSize": result.dominant_cluster_size,
        "selectedK": result.kmeans.k if result.kmeans else None,
        "selectedReplica": result.kmeans.replica if result.kmeans else None,
        "selectedSilhouette": result.kmeans.silhouette if result.kmeans else None,
        "radius": result.radius,
    }


def label_rows(
    trajectory: GardTrajectory, result: LabelResult, pipeline_id: str
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for index, row in enumerate(trajectory.observations):
        output.append(
            {
                "pipelineId": pipeline_id,
                "trajectoryId": trajectory.trajectory_id,
                "phase": trajectory.phase,
                "matrixIndex": trajectory.matrix_index,
                "engineId": trajectory.engine_id,
                "labelId": result.label_id,
                "observationIndex": row.observation_index,
                "observationKind": row.observation_kind,
                "generation": row.generation,
                "molecularStep": row.molecular_step,
                "labelStatus": result.status,
                "isReplicator": bool(result.observation_labels[index]),
                "reason": result.reason,
            }
        )
    return output
