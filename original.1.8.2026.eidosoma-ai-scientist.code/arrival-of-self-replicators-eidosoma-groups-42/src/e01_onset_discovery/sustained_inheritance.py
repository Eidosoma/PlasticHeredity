"""Online scoring for sustained parent-to-daughter compositional heredity.

The event is certified only when a fixed run of qualifying future fissions has
actually been observed.  It uses no completed trajectory, basin, centroid, or
future backfill.  The first fission in the certified run is retained only as a
retrospective physical-onset diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from math import comb

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class SustainedInheritanceResult:
    event: bool
    certification_boundary_one_based: int | None
    certification_generation: int | None
    certification_offset_one_based: int | None
    retrospective_onset_boundary_one_based: int | None
    retrospective_onset_generation: int | None
    retrospective_onset_offset_one_based: int | None
    future_boundary_count: int
    inherited_future_boundary_count: int
    maximum_consecutive_inherited: int
    fission_opportunity: bool
    exact_order_null_event_probability: float


def maximum_true_run(values: NDArray[np.bool_]) -> int:
    """Return the maximum number of consecutive true values."""

    sequence = np.asarray(values, dtype=np.bool_)
    if sequence.ndim != 1:
        raise ValueError("inheritance sequence must be one-dimensional")
    best = 0
    current = 0
    for value in sequence:
        current = current + 1 if bool(value) else 0
        best = max(best, current)
    return best


def exact_order_null_probability(length: int, successes: int, run_length: int) -> float:
    """Probability of a qualifying run under a uniformly permuted binary order.

    The count of qualifying fissions and the number of future fission
    opportunities remain fixed.  Only their temporal order is randomized.
    """

    if length < 0 or successes < 0 or successes > length:
        raise ValueError("invalid sequence length or success count")
    if run_length < 2:
        raise ValueError("run length must be at least two")
    if length == 0 or successes < run_length:
        return 0.0
    if run_length > length:
        return 0.0

    @cache
    def count_without_run(
        remaining_positions: int,
        remaining_successes: int,
        trailing_successes: int,
    ) -> int:
        if remaining_successes < 0 or remaining_successes > remaining_positions:
            return 0
        if remaining_positions == 0:
            return int(remaining_successes == 0)
        total = count_without_run(
            remaining_positions - 1,
            remaining_successes,
            0,
        )
        if remaining_successes and trailing_successes + 1 < run_length:
            total += count_without_run(
                remaining_positions - 1,
                remaining_successes - 1,
                trailing_successes + 1,
            )
        return total

    total_orders = comb(length, successes)
    avoiding_orders = count_without_run(length, successes, 0)
    probability = 1.0 - avoiding_orders / total_orders
    return float(np.clip(probability, 0.0, 1.0))


def score_sustained_inheritance(
    *,
    inherited: NDArray[np.bool_],
    generations: NDArray[np.integer],
    offsets_one_based: NDArray[np.integer],
    required_run: int = 3,
) -> SustainedInheritanceResult:
    """Score online certification of a sustained future inheritance run."""

    sequence = np.asarray(inherited, dtype=np.bool_)
    generation_values = np.asarray(generations, dtype=np.int64)
    offset_values = np.asarray(offsets_one_based, dtype=np.int64)
    if sequence.ndim != 1:
        raise ValueError("inheritance sequence must be one-dimensional")
    if len(sequence) != len(generation_values) or len(sequence) != len(offset_values):
        raise ValueError("inheritance, generation and offset arrays must align")
    if required_run < 2:
        raise ValueError("required run must be at least two")
    if len(generation_values) and np.any(np.diff(generation_values) <= 0):
        raise ValueError("future generations must be strictly increasing")
    if len(offset_values) and (
        np.any(offset_values <= 0) or np.any(np.diff(offset_values) <= 0)
    ):
        raise ValueError("future offsets must be positive and strictly increasing")

    certification_index: int | None = None
    current_run = 0
    best_run = 0
    for index, value in enumerate(sequence):
        current_run = current_run + 1 if bool(value) else 0
        best_run = max(best_run, current_run)
        if current_run >= required_run and certification_index is None:
            certification_index = index

    if certification_index is None:
        certification_boundary = None
        certification_generation = None
        certification_offset = None
        physical_index = None
        physical_boundary = None
        physical_generation = None
        physical_offset = None
    else:
        physical_index = certification_index - required_run + 1
        certification_boundary = certification_index + 1
        certification_generation = int(generation_values[certification_index])
        certification_offset = int(offset_values[certification_index])
        physical_boundary = physical_index + 1
        physical_generation = int(generation_values[physical_index])
        physical_offset = int(offset_values[physical_index])

    return SustainedInheritanceResult(
        event=certification_index is not None,
        certification_boundary_one_based=certification_boundary,
        certification_generation=certification_generation,
        certification_offset_one_based=certification_offset,
        retrospective_onset_boundary_one_based=physical_boundary,
        retrospective_onset_generation=physical_generation,
        retrospective_onset_offset_one_based=physical_offset,
        future_boundary_count=len(sequence),
        inherited_future_boundary_count=int(sequence.sum()),
        maximum_consecutive_inherited=best_run,
        fission_opportunity=len(sequence) >= required_run,
        exact_order_null_event_probability=exact_order_null_probability(
            len(sequence), int(sequence.sum()), required_run
        ),
    )
