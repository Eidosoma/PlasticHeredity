"""Observer summaries and Plastic Heredity endpoints."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Sequence

from .config import ObserverThresholds


Vector = tuple[float, ...]


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("cosine vectors must have the same dimension")
    dot = sum(a * b for a, b in zip(left, right))
    norm_left = sum(a * a for a in left)
    norm_right = sum(b * b for b in right)
    if norm_left <= 0.0 or norm_right <= 0.0:
        return 0.0
    return dot / sqrt(norm_left * norm_right)


def cyclic_kmer_spectrum(row: int, width: int, k: int, *, drop_zero: bool = False) -> Vector:
    if k <= 0 or k > width:
        raise ValueError("k must lie between 1 and width")
    counts = [0.0] * (1 << k)
    mask = (1 << width) - 1
    for offset in range(width):
        rotated = ((row >> offset) | ((row << (width - offset)) & mask)) & mask
        counts[rotated & ((1 << k) - 1)] += 1.0
    if drop_zero:
        counts = counts[1:]
    return tuple(counts)


def normalize(vector: Sequence[float]) -> Vector:
    total = sum(vector)
    if total <= 0.0:
        return tuple(0.0 for _ in vector)
    return tuple(value / total for value in vector)


def mass_support(vector: Sequence[float], quantile: float) -> int:
    """Encode the smallest deterministic top-mass support as a bit mask."""

    if not 0.0 < quantile <= 1.0:
        raise ValueError("quantile must lie in (0, 1]")
    total = sum(max(0.0, value) for value in vector)
    if total <= 0.0:
        return 0
    ordered = sorted(range(len(vector)), key=lambda index: (-vector[index], index))
    threshold = quantile * total
    cumulative = 0.0
    support = 0
    for index in ordered:
        if vector[index] <= 0.0:
            continue
        cumulative += vector[index]
        support |= 1 << index
        if cumulative >= threshold:
            break
    return support


def jaccard_bits(left: int, right: int, *, empty_empty: float = 1.0) -> float:
    union = left | right
    if union == 0:
        return empty_empty
    return (left & right).bit_count() / union.bit_count()


@dataclass(frozen=True)
class StrictEvent:
    occurred: bool
    first_break: int | None
    run_start: int | None
    similarities: tuple[float, ...]


def strict_coherent_event(
    compositions: Sequence[Sequence[float]],
    thresholds: ObserverThresholds,
) -> StrictEvent:
    """Evaluate the strict coherent-eight contract from the preprint.

    Composition index 0 is the launch anchor.  Similarity index ``i`` is the
    boundary from composition ``i`` to composition ``i + 1``.
    """

    similarities = tuple(cosine(a, b) for a, b in zip(compositions, compositions[1:]))
    bounded = similarities[: thresholds.horizon]
    first_break = next(
        (index for index, value in enumerate(bounded) if value <= thresholds.inherit),
        None,
    )
    if first_break is None:
        return StrictEvent(False, None, None, similarities)

    old_anchor = compositions[first_break]
    run = thresholds.strict_run
    # A candidate begins strictly after the break boundary.
    final_start = min(len(bounded) - run, thresholds.horizon - run)
    for start in range(first_break + 1, final_start + 1):
        if not all(value > thresholds.inherit for value in bounded[start : start + run]):
            continue
        daughters = compositions[start + 1 : start + run + 1]
        if len(daughters) != run:
            continue
        coherent = all(
            cosine(daughters[i], daughters[j]) > thresholds.coherence
            for i in range(run)
            for j in range(i)
        )
        distinct = all(cosine(daughter, old_anchor) <= thresholds.distinct for daughter in daughters)
        if coherent and distinct:
            return StrictEvent(True, first_break, start, similarities)
    return StrictEvent(False, first_break, None, similarities)


def break_by(similarities: Sequence[float], horizon: int, cutoff: float) -> bool:
    return any(value <= cutoff for value in similarities[:horizon])

