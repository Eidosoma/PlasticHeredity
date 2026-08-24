"""Deterministic helpers for independent-lineage target-basin transfer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class TraceTargetSummary:
    entered: bool
    first_entry_offset_one_based: int | None
    minimum_score: float | None
    maximum_score: float | None
    final_score: float | None


@dataclass(frozen=True, slots=True)
class NumericalEquivalence:
    """Diagnostics for a value-preserving float64 replay comparison."""

    passed: bool
    finite_masks_identical: bool
    nonfinite_values_identical: bool
    max_absolute_error: float
    max_relative_error: float
    max_ulp_error: int


def close_compositions(states: NDArray[np.integer]) -> NDArray[np.float64]:
    """Close integer or nonnegative states to relative compositions."""

    values = np.asarray(states, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("states must be a two-dimensional array")
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("states must be finite and nonnegative")
    masses = values.sum(axis=1)
    if np.any(masses <= 0):
        raise ValueError("each state must have positive mass")
    return values / masses[:, None]


def cosine_scores(
    states: NDArray[np.integer], centroids: NDArray[np.floating]
) -> NDArray[np.float64]:
    """Score every state against every centroid using historical cosine H."""

    compositions = close_compositions(states)
    references = np.asarray(centroids, dtype=np.float64)
    if references.ndim == 1:
        references = references[None, :]
    if references.ndim != 2 or references.shape[1] != compositions.shape[1]:
        raise ValueError("centroid dimensions do not match states")
    if np.any(~np.isfinite(references)) or np.any(references < 0):
        raise ValueError("centroids must be finite and nonnegative")
    totals = references.sum(axis=1)
    if np.any(totals <= 0):
        raise ValueError("centroids must have positive mass")
    references = references / totals[:, None]
    denominator = np.linalg.norm(compositions, axis=1)[:, None] * np.linalg.norm(
        references, axis=1
    )[None, :]
    if np.any(denominator <= 0):
        raise ValueError("cosine denominator must be positive")
    return np.clip(compositions @ references.T / denominator, -1.0, 1.0)


def centroid_similarity(
    left: NDArray[np.floating], right: NDArray[np.floating]
) -> float:
    """Historical cosine H between two target centroids."""

    values = cosine_scores(
        np.asarray(left, dtype=np.float64)[None, :],
        np.asarray(right, dtype=np.float64)[None, :],
    )
    return float(values[0, 0])


def summarize_scores(
    scores: NDArray[np.floating], *, threshold: float = 0.9
) -> TraceTargetSummary:
    """Summarize finite target scores along one frozen branch trace."""

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    finite = np.isfinite(values)
    if not finite.any():
        return TraceTargetSummary(False, None, None, None, None)
    labels = finite & (values >= threshold)
    indices = np.flatnonzero(labels)
    return TraceTargetSummary(
        entered=bool(len(indices)),
        first_entry_offset_one_based=int(indices[0] + 1) if len(indices) else None,
        minimum_score=float(np.min(values[finite])),
        maximum_score=float(np.max(values[finite])),
        final_score=float(values[np.flatnonzero(finite)[-1]]),
    )


def numerical_equivalence(
    left: NDArray[np.floating],
    right: NDArray[np.floating],
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
    maximum_ulp_error: int,
) -> NumericalEquivalence:
    """Compare float64 arrays under an explicit numerical replay contract."""

    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("replay arrays must have identical shapes")
    finite_a = np.isfinite(a)
    finite_b = np.isfinite(b)
    finite_masks_identical = bool(np.array_equal(finite_a, finite_b))
    nonfinite_values_identical = bool(
        np.array_equal(np.isnan(a), np.isnan(b))
        and np.array_equal(np.isposinf(a), np.isposinf(b))
        and np.array_equal(np.isneginf(a), np.isneginf(b))
    )
    if not finite_masks_identical:
        return NumericalEquivalence(False, False, nonfinite_values_identical, float("inf"), float("inf"), 2**63 - 1)
    finite = finite_a
    if not finite.any():
        return NumericalEquivalence(nonfinite_values_identical, True, nonfinite_values_identical, 0.0, 0.0, 0)
    a_finite = a[finite]
    b_finite = b[finite]
    absolute = np.abs(a_finite - b_finite)
    scale = np.maximum(np.abs(a_finite), np.abs(b_finite))
    relative = np.divide(
        absolute,
        scale,
        out=np.zeros_like(absolute),
        where=scale > 0,
    )

    # Map IEEE-754 bit patterns into monotonically ordered unsigned integers,
    # including negative values, before measuring representable-value distance.
    sign = np.uint64(1 << 63)
    a_bits = a_finite.view(np.uint64)
    b_bits = b_finite.view(np.uint64)
    a_ordered = np.where((a_bits & sign) != 0, ~a_bits, a_bits | sign)
    b_ordered = np.where((b_bits & sign) != 0, ~b_bits, b_bits | sign)
    ulp = np.maximum(a_ordered, b_ordered) - np.minimum(a_ordered, b_ordered)
    max_absolute = float(np.max(absolute))
    max_relative = float(np.max(relative))
    max_ulp = int(np.max(ulp))
    passed = bool(
        nonfinite_values_identical
        and max_absolute <= absolute_tolerance
        and max_relative <= relative_tolerance
        and max_ulp <= maximum_ulp_error
    )
    return NumericalEquivalence(
        passed,
        finite_masks_identical,
        nonfinite_values_identical,
        max_absolute,
        max_relative,
        max_ulp,
    )
