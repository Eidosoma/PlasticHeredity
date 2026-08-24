"""Deterministic multi-lineage recurring-attractor atlas primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class AtlasComponent:
    component_id: int
    member_indices: tuple[int, ...]
    lineage_counts: tuple[int, ...]
    lineage_generation_spans: tuple[int, ...]
    centroid: NDArray[np.float64]
    mean_within_h: float
    minimum_within_h: float


@dataclass(frozen=True, slots=True)
class AttractorAtlas:
    status: str
    components: tuple[AtlasComponent, ...]
    centroids: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class AtlasTrajectorySummary:
    occupancy: float
    positive_count: int
    first_entry_zero_based: int | None
    transition_count: int
    positive_episode_count: int
    longest_positive_episode: int
    recurrent_positive: bool
    self_transition_probability: float | None


def relative_compositions(states: NDArray[np.integer]) -> NDArray[np.float64]:
    values = np.asarray(states, dtype=np.float64)
    if values.ndim != 2 or np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("states must be a finite nonnegative two-dimensional array")
    masses = values.sum(axis=1)
    if np.any(masses <= 0):
        raise ValueError("each state must have positive mass")
    return np.ascontiguousarray(values / masses[:, None], dtype=np.float64)


def cosine_similarity_matrix(compositions: NDArray[np.floating]) -> NDArray[np.float64]:
    values = np.asarray(compositions, dtype=np.float64)
    if values.ndim != 2 or np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("compositions must be finite and nonnegative")
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms <= 0):
        raise ValueError("composition norm must be positive")
    normalized = values / norms[:, None]
    return np.clip(normalized @ normalized.T, -1.0, 1.0)


def _connected_components(adjacency: NDArray[np.bool_]) -> tuple[tuple[int, ...], ...]:
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be square")
    remaining = set(range(len(adjacency)))
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
    return tuple(output)


def build_cross_lineage_atlas(
    states: NDArray[np.integer],
    lineage_ids: NDArray[np.integer],
    generations: NDArray[np.integer],
    *,
    lineage_count: int = 2,
    threshold: float = 0.9,
    minimum_visits_per_lineage: int = 2,
    minimum_generation_span: int = 2,
) -> AttractorAtlas:
    """Build an atlas from components recurring in every reference lineage."""

    compositions = relative_compositions(states)
    lineage = np.asarray(lineage_ids, dtype=np.int64)
    generation = np.asarray(generations, dtype=np.int64)
    if lineage.shape != (len(compositions),) or generation.shape != (len(compositions),):
        raise ValueError("lineage and generation vectors must match the states")
    if set(np.unique(lineage)) != set(range(lineage_count)):
        raise ValueError("every registered reference lineage must be represented")
    similarity = cosine_similarity_matrix(compositions)
    components = _connected_components(similarity >= float(threshold))
    valid: list[AtlasComponent] = []
    for members in components:
        counts = tuple(int(np.sum(lineage[list(members)] == item)) for item in range(lineage_count))
        spans = []
        for item in range(lineage_count):
            values = generation[np.asarray(members)[lineage[list(members)] == item]]
            spans.append(int(values.max() - values.min()) if len(values) else 0)
        if any(value < minimum_visits_per_lineage for value in counts) or any(
            value < minimum_generation_span for value in spans
        ):
            continue
        member_values = compositions[list(members)]
        centroid = member_values.mean(axis=0)
        centroid /= centroid.sum()
        within = similarity[np.ix_(members, members)]
        off_diagonal = within[~np.eye(len(members), dtype=bool)]
        valid.append(
            AtlasComponent(
                component_id=len(valid),
                member_indices=members,
                lineage_counts=counts,
                lineage_generation_spans=tuple(spans),
                centroid=np.ascontiguousarray(centroid, dtype=np.float64),
                mean_within_h=float(np.mean(off_diagonal)) if len(off_diagonal) else 1.0,
                minimum_within_h=float(np.min(off_diagonal)) if len(off_diagonal) else 1.0,
            )
        )
    if not valid:
        return AttractorAtlas(
            "NO_CROSS_LINEAGE_RECURRING_BASIN",
            (),
            np.empty((0, compositions.shape[1]), dtype=np.float64),
        )
    centroids = np.stack([component.centroid for component in valid])
    return AttractorAtlas("ELIGIBLE", tuple(valid), centroids)


def score_atlas(
    states: NDArray[np.integer], centroids: NDArray[np.floating], *, threshold: float = 0.9
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.bool_]]:
    """Score states against a union of atlas centroids."""

    compositions = relative_compositions(states)
    references = np.asarray(centroids, dtype=np.float64)
    if references.ndim != 2 or references.shape[1] != compositions.shape[1] or not len(references):
        raise ValueError("at least one dimension-matched centroid is required")
    references = relative_compositions(references)
    state_norm = np.linalg.norm(compositions, axis=1)
    ref_norm = np.linalg.norm(references, axis=1)
    values = np.clip(
        (compositions @ references.T) / (state_norm[:, None] * ref_norm[None, :]),
        -1.0,
        1.0,
    )
    assignment = np.argmax(values, axis=1).astype(np.int64)
    scores = values[np.arange(len(values)), assignment]
    labels = scores >= float(threshold)
    assignment = np.where(labels, assignment, -1).astype(np.int64)
    return scores, assignment, labels


def summarize_atlas_labels(
    labels: NDArray[np.bool_], assignments: NDArray[np.integer]
) -> AtlasTrajectorySummary:
    values = np.asarray(labels, dtype=bool)
    basins = np.asarray(assignments, dtype=np.int64)
    if values.ndim != 1 or basins.shape != values.shape or not len(values):
        raise ValueError("labels and assignments must be nonempty matching vectors")
    indices = np.flatnonzero(values)
    transitions = int(np.sum(values[1:] != values[:-1]))
    lengths: list[int] = []
    current = 0
    for value in values:
        if value:
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    paired = values[:-1] & values[1:]
    self_probability = (
        float(np.mean(basins[:-1][paired] == basins[1:][paired]))
        if paired.any()
        else None
    )
    recurrent = bool(len(indices) >= 2 and int(indices[-1] - indices[0]) >= 2)
    return AtlasTrajectorySummary(
        occupancy=float(np.mean(values)),
        positive_count=int(np.sum(values)),
        first_entry_zero_based=int(indices[0]) if len(indices) else None,
        transition_count=transitions,
        positive_episode_count=len(lengths),
        longest_positive_episode=max(lengths, default=0),
        recurrent_positive=recurrent,
        self_transition_probability=self_probability,
    )
