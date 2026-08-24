"""Historical GARD H similarity and non-drift marking routines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .engine import HistoricalReferenceError, HistoricalSourceDomainError


@dataclass(frozen=True)
class NonDriftResult:
    """Historical non-drift mask with source-compatible angle alignment."""

    is_non_drift: tuple[bool, ...]
    angles: tuple[float, ...]
    local_scores: tuple[float, ...]
    active_generation_count: int
    first_zero_sum_generation_one_based: int | None
    technique: str
    threshold: float
    drift_size: int | None


def _matrix(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise HistoricalReferenceError(
            f"{name} must be a nonempty two-dimensional matrix."
        )
    if not np.all(np.isfinite(matrix)):
        raise HistoricalReferenceError(f"{name} must contain finite values.")
    return matrix


def historical_h(set1: ArrayLike, set2: ArrayLike | None = None) -> NDArray[np.float64]:
    """Reproduce ``tgs_H`` column-wise cosine similarity with clipping."""

    first = _matrix(set1, name="set1")
    second = first if set2 is None else _matrix(set2, name="set2")
    if first.shape[0] != second.shape[0]:
        raise HistoricalReferenceError("set1 and set2 must have the same row count.")
    epsilon = 1e-7
    first_norm = first / np.maximum(np.sqrt(np.sum(first**2, axis=0)), epsilon)
    second_norm = second / np.maximum(np.sqrt(np.sum(second**2, axis=0)), epsilon)
    if first_norm.shape[0] == 1:
        first_norm = first_norm.T
    if second_norm.shape[0] == 1:
        second_norm = second_norm.T
    return np.clip(first_norm.T @ second_norm, 0.0, 1.0)


def _active_trace(
    trace: ArrayLike,
) -> tuple[NDArray[np.float64], int, int | None]:
    matrix = _matrix(trace, name="trace")
    original_size = matrix.shape[1]
    zero_columns = np.flatnonzero(np.sum(matrix, axis=0) == 0)
    first_zero = int(zero_columns[0]) if zero_columns.size else None
    active = matrix if first_zero is None else matrix[:, :first_zero]
    if active.shape[1] < 2:
        raise HistoricalSourceDomainError(
            "tgs_nondrift accesses the first pairwise similarity and is undefined with fewer "
            "than two active generations."
        )
    norms = np.sqrt(np.sum(active * active, axis=0))
    if np.any(norms == 0):
        raise HistoricalSourceDomainError(
            "Historical non-drift normalization divided by zero."
        )
    normalized = active / norms
    return normalized, original_size, first_zero


def _pad(
    values: NDArray[np.float64] | NDArray[np.bool_],
    *,
    size: int,
    fill: float | bool,
) -> NDArray[np.float64] | NDArray[np.bool_]:
    if values.size == size:
        return values
    result = np.full(size, fill, dtype=values.dtype)
    result[: values.size] = values
    return result


def historical_nondrift_technique1(
    trace: ArrayLike,
    *,
    threshold: float,
) -> NonDriftResult:
    """Reproduce historical adjacent-generation averaged-H marking.

    The threshold is mandatory because the MATLAB default path misspells the
    variable name.  Comparison is strictly ``>`` as in the source.
    """

    if not np.isfinite(threshold):
        raise HistoricalReferenceError("threshold must be finite.")
    normalized, original_size, first_zero = _active_trace(trace)
    pairwise = np.sum(normalized[:, :-1] * normalized[:, 1:], axis=0)
    angles_active = np.concatenate(([pairwise[0]], pairwise))
    local_scores_active = 0.5 * (
        np.concatenate(([pairwise[0]], pairwise))
        + np.concatenate((pairwise, [pairwise[-1]]))
    )
    mask_active = local_scores_active > float(threshold)
    angles = _pad(angles_active, size=original_size, fill=0.0)
    scores = _pad(local_scores_active, size=original_size, fill=0.0)
    mask = _pad(mask_active, size=original_size, fill=False)
    return NonDriftResult(
        is_non_drift=tuple(bool(value) for value in mask),
        angles=tuple(float(value) for value in angles),
        local_scores=tuple(float(value) for value in scores),
        active_generation_count=int(normalized.shape[1]),
        first_zero_sum_generation_one_based=None
        if first_zero is None
        else first_zero + 1,
        technique="historical_tgs_nondrift_technique1",
        threshold=float(threshold),
        drift_size=None,
    )


def historical_nondrift_technique2(
    trace: ArrayLike,
    *,
    threshold: float,
    drift_size: int,
) -> NonDriftResult:
    """Reproduce the optional historical consecutive-similarity technique.

    The source shifts accepted runs one generation earlier.  If an accepted
    run starts at the first element, its MATLAB index expression begins at
    zero; this port raises instead of silently repairing that source-domain
    failure.
    """

    if not np.isfinite(threshold):
        raise HistoricalReferenceError("threshold must be finite.")
    if (
        not isinstance(drift_size, int)
        or isinstance(drift_size, bool)
        or drift_size <= 0
    ):
        raise HistoricalReferenceError("drift_size must be a positive integer.")
    normalized, original_size, first_zero = _active_trace(trace)
    pairwise = np.sum(normalized[:, :-1] * normalized[:, 1:], axis=0)
    similarities = np.concatenate(([pairwise[0]], pairwise))
    above = similarities > float(threshold)

    starts = np.flatnonzero(above & ~np.concatenate(([False], above[:-1])))
    ends = np.flatnonzero(above & ~np.concatenate((above[1:], [False])))
    mask_active = np.zeros(normalized.shape[1], dtype=bool)
    for start, end in zip(starts, ends, strict=True):
        if end - start + 1 < drift_size:
            continue
        if start == 0:
            raise HistoricalSourceDomainError(
                "Historical technique-2 accepted run starts at MATLAB index 1, making "
                "the source assignment begin at invalid index 0."
            )
        mask_active[start - 1 : end] = True

    similarities_padded = _pad(similarities, size=original_size, fill=0.0)
    mask = _pad(mask_active, size=original_size, fill=False)
    return NonDriftResult(
        is_non_drift=tuple(bool(value) for value in mask),
        angles=tuple(float(value) for value in similarities_padded),
        local_scores=tuple(float(value) for value in similarities_padded),
        active_generation_count=int(normalized.shape[1]),
        first_zero_sum_generation_one_based=None
        if first_zero is None
        else first_zero + 1,
        technique="historical_tgs_nondrift_technique2",
        threshold=float(threshold),
        drift_size=drift_size,
    )
