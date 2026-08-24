"""Frozen recurrence-map analogue predictors for S19-L26.

The module deliberately contains no outcome-dependent fitting.  It converts a
64-observation past-only composition window into three fixed geometric views;
the L26 runner fits development-only scalers and evaluates a fixed-k analogue
committor on held-out matrices.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from e01_onset_discovery.outcome_blind_representation import (
    LANDMARK_COUNT,
    MOLECULE_TYPES,
    organization_channel_sequence,
)

ANALOG_NEIGHBORS = 15
RECURRENCE_OFFSET = 2


def _closed_compositions(
    states: NDArray[np.integer[Any]],
) -> NDArray[np.float64]:
    counts = np.asarray(states, dtype=np.float64)
    if counts.shape != (LANDMARK_COUNT, MOLECULE_TYPES):
        raise ValueError("analogue window must be 64-by-100")
    if np.any(counts < 0.0):
        raise ValueError("negative molecular count")
    mass = np.sum(counts, axis=1)
    if np.any(mass <= 0.0):
        raise ValueError("empty molecular composition")
    return counts / mass[:, None]


def recurrence_map_vector(
    states: NDArray[np.integer[Any]],
) -> NDArray[np.float64]:
    """Vectorize all nonadjacent cosine similarities in temporal order."""

    composition = _closed_compositions(states)
    norm = np.linalg.norm(composition, axis=1)
    similarity = (composition @ composition.T) / (norm[:, None] * norm[None, :])
    similarity = np.clip(similarity, -1.0, 1.0)
    row, column = np.triu_indices(LANDMARK_COUNT, k=RECURRENCE_OFFSET)
    result = similarity[row, column].astype(np.float64, copy=False)
    if len(result) != 1953 or not np.isfinite(result).all():
        raise RuntimeError("invalid recurrence-map representation")
    return result


def exact_h_trace_vector(
    states: NDArray[np.integer[Any]],
) -> NDArray[np.float64]:
    """Return the five frozen H-derived organization channels."""

    result = organization_channel_sequence(states)[:, 6:].reshape(-1)
    if result.shape != (320,) or not np.isfinite(result).all():
        raise RuntimeError("invalid exact-H trace representation")
    return result


def ordinary_path_vector(
    states: NDArray[np.integer[Any]],
) -> NDArray[np.float64]:
    """Return non-H mass/diversity/composition-summary path channels."""

    result = organization_channel_sequence(states)[:, :6].reshape(-1)
    if result.shape != (384,) or not np.isfinite(result).all():
        raise RuntimeError("invalid ordinary-path representation")
    return result


def all_analog_representations(
    states: NDArray[np.integer[Any]],
) -> dict[str, NDArray[np.float64]]:
    return {
        "RECURRENCE_MAP_ANALOG": recurrence_map_vector(states),
        "EXACT_H_TRACE_ANALOG": exact_h_trace_vector(states),
        "ORDINARY_PATH_ANALOG": ordinary_path_vector(states),
    }


def deterministic_knn_probability(
    query: NDArray[np.floating[Any]],
    reference: NDArray[np.floating[Any]],
    labels: NDArray[np.integer[Any]],
    tie_keys: list[tuple[str, int]],
    k: int = ANALOG_NEIGHBORS,
) -> tuple[float, tuple[int, ...], tuple[float, ...]]:
    """Uniform fixed-k analogue probability with outcome-blind tie ordering."""

    left = np.asarray(query, dtype=np.float64)
    right = np.asarray(reference, dtype=np.float64)
    target = np.asarray(labels, dtype=np.int8)
    if right.ndim != 2 or left.shape != (right.shape[1],):
        raise ValueError("query/reference shape mismatch")
    if len(right) != len(target) or len(tie_keys) != len(target):
        raise ValueError("reference metadata length mismatch")
    if len(right) < k:
        raise ValueError("insufficient analogue library")
    distances = np.sqrt(np.mean(np.square(right - left[None, :]), axis=1))
    if not np.isfinite(distances).all():
        raise RuntimeError("nonfinite analogue distance")
    order = sorted(
        range(len(distances)),
        key=lambda index: (
            float(distances[index]),
            tie_keys[index][0],
            tie_keys[index][1],
        ),
    )
    selected = tuple(order[:k])
    probability = float(np.mean(target[list(selected)]))
    return probability, selected, tuple(float(distances[index]) for index in selected)
