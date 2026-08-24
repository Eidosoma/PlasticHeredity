"""Dominant recurring-composition detector for GARD self-replicator states."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from .composition import cosine_similarity, relative_composition
from .config import ReplicatorConfig
from .gard import RunTrace


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class ReplicatorResult:
    """Binary molecular-step labels relative to the dominant compotype medoid."""

    labels: BoolArray
    similarity: FloatArray
    reference: FloatArray
    reference_trace_index: int
    support: int
    reference_indices: IntArray


def _pairwise_cosine(values: FloatArray) -> FloatArray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    safe = np.where(norms > 0, norms, 1.0)
    normalized = values / safe
    return np.clip(
        np.einsum("ik,jk->ij", normalized, normalized, optimize=True),
        -1.0,
        1.0,
    )


def _similarity(
    reference: FloatArray, values: FloatArray, metric: str
) -> FloatArray:
    if metric == "cosine":
        return cosine_similarity(reference, values)
    if metric == "euclidean":
        distances = np.linalg.norm(np.asarray(values, dtype=float) - reference, axis=-1)
        # The diameter of the relative-composition simplex is sqrt(2).
        return np.clip(1.0 - distances / np.sqrt(2.0), 0.0, 1.0)
    raise ValueError("metric must be 'cosine' or 'euclidean'")


def _pairwise_similarity(values: FloatArray, metric: str) -> FloatArray:
    if metric == "cosine":
        return _pairwise_cosine(values)
    if metric == "euclidean":
        differences = values[:, None, :] - values[None, :, :]
        distances = np.linalg.norm(differences, axis=2)
        return np.clip(1.0 - distances / np.sqrt(2.0), 0.0, 1.0)
    raise ValueError("metric must be 'cosine' or 'euclidean'")


def detect_replicators(
    trace: RunTrace, config: ReplicatorConfig = ReplicatorConfig()
) -> ReplicatorResult:
    """Find the densest recurring composition and label similar molecular steps.

    This reconstructs the paper's phrase "similarity threshold relative to the
    most recurring composition" using standard GARD cosine/H similarity.
    """

    config.validate()
    compositions = relative_composition(trace.counts)
    if config.reference_states == "generation_end":
        reference_indices = trace.generation_end_indices()
    else:
        reference_indices = np.arange(trace.counts.shape[0], dtype=np.int64)
    if reference_indices.size == 0:
        raise ValueError("trace contains no candidate reference states")
    candidates = compositions[reference_indices]
    pairwise = _pairwise_similarity(candidates, config.similarity_metric)
    neighbors = pairwise >= config.similarity_threshold
    support_by_candidate = neighbors.sum(axis=1)
    max_support = int(support_by_candidate.max())
    tied = np.flatnonzero(support_by_candidate == max_support)
    if tied.size > 1:
        mean_neighbor_similarity = np.array(
            [pairwise[i, neighbors[i]].mean() for i in tied]
        )
        chosen_local = int(tied[np.argmax(mean_neighbor_similarity)])
    else:
        chosen_local = int(tied[0])
    chosen_trace = int(reference_indices[chosen_local])
    if config.reference_method == "neighbor_centroid":
        reference = np.mean(candidates[neighbors[chosen_local]], axis=0)
        total = float(reference.sum())
        reference = reference / total if total > 0 else compositions[chosen_trace]
    else:
        reference = compositions[chosen_trace]
    similarity = _similarity(reference, compositions, config.similarity_metric)
    labels = similarity >= config.similarity_threshold
    if max_support < config.min_recurrences:
        labels[:] = False
    return ReplicatorResult(
        labels=labels,
        similarity=similarity,
        reference=reference,
        reference_trace_index=chosen_trace,
        support=max_support,
        reference_indices=reference_indices,
    )


@dataclass(frozen=True)
class ReplicatorMetrics:
    persistence: int
    probability: float
    consistency: float
    time_to_first: float


def replicator_metrics(labels: NDArray[np.bool_]) -> ReplicatorMetrics:
    """Compute the four intervention outcomes reported in Table 1."""

    values = np.asarray(labels, dtype=bool)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("labels must be a non-empty vector")
    persistence = int(values.sum())
    probability = float(values.mean())
    if values.size > 2 and np.unique(values[:-1]).size > 1 and np.unique(values[1:]).size > 1:
        consistency = float(stats.pearsonr(values[:-1].astype(float), values[1:].astype(float)).statistic)
    else:
        consistency = float("nan")
    locations = np.flatnonzero(values)
    time_to_first = float(locations[0] / (values.size - 1)) if locations.size and values.size > 1 else float("nan")
    return ReplicatorMetrics(
        persistence=persistence,
        probability=probability,
        consistency=consistency,
        time_to_first=time_to_first,
    )
