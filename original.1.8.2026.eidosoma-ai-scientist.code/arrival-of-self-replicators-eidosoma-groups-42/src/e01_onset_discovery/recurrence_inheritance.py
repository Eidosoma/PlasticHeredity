"""Past-only recurrence and parent/daughter inheritance event scoring.

The event is evaluated sequentially.  A completed trajectory, future centroid,
or backfilled label is never needed: an inherited selected daughter is positive
only when it recurs with an earlier inherited selected daughter separated by at
least one intervening generation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class RecurrenceInheritanceResult:
    event: bool
    first_event_boundary_one_based: int | None
    first_event_generation: int | None
    matched_reference_generation: int | None
    future_boundary_count: int
    inherited_future_boundary_count: int
    eligible_comparison_count: int
    recurrence_match_count: int
    maximum_eligible_h: float | None
    inheritance_only_event: bool


def cosine_h(left: NDArray[np.number], right: NDArray[np.number]) -> float:
    """Return historical cosine H for two finite, nonnegative count vectors."""

    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("H inputs must be one-dimensional and shape matched")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("H inputs must be finite")
    if np.any(x < 0) or np.any(y < 0):
        raise ValueError("H inputs must be nonnegative")
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denominator <= 0:
        return float("nan")
    return float(np.clip(np.dot(x, y) / denominator, -1.0, 1.0))


def _validate_history(
    states: NDArray[np.number],
    generations: NDArray[np.integer],
    inherited: NDArray[np.bool_],
    *,
    name: str,
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.bool_]]:
    values = np.asarray(states, dtype=np.int64)
    generation_values = np.asarray(generations, dtype=np.int64)
    inheritance_values = np.asarray(inherited, dtype=np.bool_)
    if values.ndim != 2:
        raise ValueError(f"{name} states must be a two-dimensional array")
    if len(values) != len(generation_values) or len(values) != len(inheritance_values):
        raise ValueError(f"{name} arrays must have equal row counts")
    if np.any(values < 0) or (len(values) and np.any(values.sum(axis=1) <= 0)):
        raise ValueError(f"{name} states must have positive mass")
    if len(generation_values) and np.any(np.diff(generation_values) <= 0):
        raise ValueError(f"{name} generations must be strictly increasing")
    return values, generation_values, inheritance_values


def score_recurrence_inheritance(
    *,
    prefix_states: NDArray[np.number],
    prefix_generations: NDArray[np.integer],
    prefix_inherited: NDArray[np.bool_],
    future_states: NDArray[np.number],
    future_generations: NDArray[np.integer],
    future_inherited: NDArray[np.bool_],
    threshold: float = 0.9,
    minimum_generation_gap: int = 2,
) -> RecurrenceInheritanceResult:
    """Score the first sequential recurrence/inheritance event.

    At each future boundary ``g``, only boundary states already present in the
    prefix or generated earlier in this same continuation are eligible.  Both
    the current and matched earlier boundary must pass the parent/daughter
    inheritance threshold outside this function.  Recurrence uses strict
    ``H > threshold`` and requires ``g - h >= minimum_generation_gap``.
    """

    if not 0 < threshold < 1:
        raise ValueError("threshold must lie strictly between zero and one")
    if minimum_generation_gap < 2:
        raise ValueError("minimum generation gap must preserve an intervening generation")
    prefix, prefix_g, prefix_i = _validate_history(
        prefix_states, prefix_generations, prefix_inherited, name="prefix"
    )
    future, future_g, future_i = _validate_history(
        future_states, future_generations, future_inherited, name="future"
    )
    if prefix.shape[1:] != future.shape[1:]:
        raise ValueError("prefix and future states must have matching feature counts")
    if len(prefix_g) and len(future_g) and future_g[0] <= prefix_g[-1]:
        raise ValueError("future generations must follow the observed prefix")

    history_states = [row.copy() for row in prefix]
    history_generations = [int(value) for value in prefix_g]
    history_inherited = [bool(value) for value in prefix_i]
    first_boundary: int | None = None
    first_generation: int | None = None
    first_reference_generation: int | None = None
    comparisons = 0
    matches = 0
    maximum: float | None = None

    for boundary_index, (state, generation, inherited) in enumerate(
        zip(future, future_g, future_i, strict=True), start=1
    ):
        if inherited:
            candidates = [
                (prior, prior_generation)
                for prior, prior_generation, prior_inherited in zip(
                    history_states,
                    history_generations,
                    history_inherited,
                    strict=True,
                )
                if prior_inherited
                and int(generation) - prior_generation >= minimum_generation_gap
            ]
            scores = [(cosine_h(state, prior), prior_generation) for prior, prior_generation in candidates]
            finite = [(score, prior_g) for score, prior_g in scores if np.isfinite(score)]
            comparisons += len(finite)
            if finite:
                local_score, local_generation = max(finite, key=lambda item: item[0])
                maximum = local_score if maximum is None else max(maximum, local_score)
                local_matches = sum(score > threshold for score, _ in finite)
                matches += local_matches
                if local_matches and first_boundary is None:
                    first_boundary = boundary_index
                    first_generation = int(generation)
                    # Choose the most similar eligible reference deterministically.
                    first_reference_generation = int(local_generation)
        history_states.append(state.copy())
        history_generations.append(int(generation))
        history_inherited.append(bool(inherited))

    return RecurrenceInheritanceResult(
        event=first_boundary is not None,
        first_event_boundary_one_based=first_boundary,
        first_event_generation=first_generation,
        matched_reference_generation=first_reference_generation,
        future_boundary_count=len(future),
        inherited_future_boundary_count=int(np.sum(future_i)),
        eligible_comparison_count=comparisons,
        recurrence_match_count=matches,
        maximum_eligible_h=maximum,
        inheritance_only_event=bool(np.any(future_i)),
    )
