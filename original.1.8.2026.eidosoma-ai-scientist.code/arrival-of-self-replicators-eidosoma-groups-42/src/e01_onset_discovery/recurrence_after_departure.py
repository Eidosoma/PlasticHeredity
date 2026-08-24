"""Past-defined online recurrence after a certified post-fission departure."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np
from numpy.typing import NDArray

from e01_onset_discovery.recurrence_inheritance import cosine_h


@dataclass(frozen=True, slots=True)
class RecurrenceAfterDepartureResult:
    event: bool
    departure_boundary_one_based: int | None
    departure_generation: int | None
    departure_offset_one_based: int | None
    certification_boundary_one_based: int | None
    certification_generation: int | None
    certification_offset_one_based: int | None
    future_boundary_count: int
    near_anchor_count: int
    departed_count: int
    departure_observed: bool
    mixed_membership_opportunity: bool
    maximum_postdeparture_h: float | None
    return_progress: float
    exact_order_null_event_probability: float
    scores: tuple[float, ...]


def exact_departure_return_order_probability(length: int, near_count: int) -> float:
    """Return P(a departed boundary precedes a near-anchor boundary).

    The total boundary count and the near/departed counts are fixed.  Under a
    uniformly random order, the only ordering without a departed-to-near
    transition has every near boundary before every departed boundary.
    """

    if length < 0 or near_count < 0 or near_count > length:
        raise ValueError("invalid length or near count")
    departed = length - near_count
    if near_count == 0 or departed == 0:
        return 0.0
    return float(1.0 - 1.0 / comb(length, near_count))


def score_recurrence_after_departure(
    *,
    anchor: NDArray[np.number],
    future_states: NDArray[np.number],
    generations: NDArray[np.integer],
    offsets_one_based: NDArray[np.integer],
    threshold: float = 0.9,
) -> RecurrenceAfterDepartureResult:
    """Certify return only after a future boundary has first departed."""

    reference = np.asarray(anchor, dtype=np.int64)
    states = np.asarray(future_states, dtype=np.int64)
    generation_values = np.asarray(generations, dtype=np.int64)
    offset_values = np.asarray(offsets_one_based, dtype=np.int64)
    if reference.ndim != 1 or np.any(reference < 0) or reference.sum() <= 0:
        raise ValueError("anchor must be a positive-mass nonnegative vector")
    if states.ndim != 2 or states.shape[1] != len(reference):
        raise ValueError("future states must be a feature-matched matrix")
    if len(states) != len(generation_values) or len(states) != len(offset_values):
        raise ValueError("future state, generation and offset arrays must align")
    if np.any(states < 0) or (len(states) and np.any(states.sum(axis=1) <= 0)):
        raise ValueError("future states must have positive mass")
    if not 0 < threshold < 1:
        raise ValueError("threshold must lie strictly between zero and one")
    if len(generation_values) and np.any(np.diff(generation_values) <= 0):
        raise ValueError("future generations must be strictly increasing")
    if len(offset_values) and (
        np.any(offset_values <= 0) or np.any(np.diff(offset_values) <= 0)
    ):
        raise ValueError("future offsets must be positive and strictly increasing")

    scores = np.asarray([cosine_h(reference, state) for state in states], dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError("all anchor similarities must be finite")
    near = scores > threshold
    departed_indices = np.flatnonzero(~near)
    departure_index = int(departed_indices[0]) if len(departed_indices) else None
    certification_index: int | None = None
    maximum_postdeparture: float | None = None
    if departure_index is not None:
        postdeparture = scores[departure_index + 1 :]
        if len(postdeparture):
            maximum_postdeparture = float(np.max(postdeparture))
            returns = np.flatnonzero(postdeparture > threshold)
            if len(returns):
                certification_index = departure_index + 1 + int(returns[0])

    progress = 0.0 if maximum_postdeparture is None else maximum_postdeparture
    return RecurrenceAfterDepartureResult(
        event=certification_index is not None,
        departure_boundary_one_based=(departure_index + 1 if departure_index is not None else None),
        departure_generation=(
            int(generation_values[departure_index]) if departure_index is not None else None
        ),
        departure_offset_one_based=(
            int(offset_values[departure_index]) if departure_index is not None else None
        ),
        certification_boundary_one_based=(
            certification_index + 1 if certification_index is not None else None
        ),
        certification_generation=(
            int(generation_values[certification_index])
            if certification_index is not None
            else None
        ),
        certification_offset_one_based=(
            int(offset_values[certification_index])
            if certification_index is not None
            else None
        ),
        future_boundary_count=len(states),
        near_anchor_count=int(near.sum()),
        departed_count=int((~near).sum()),
        departure_observed=departure_index is not None,
        mixed_membership_opportunity=bool(np.any(near) and np.any(~near)),
        maximum_postdeparture_h=maximum_postdeparture,
        return_progress=progress,
        exact_order_null_event_probability=exact_departure_return_order_probability(
            len(states), int(near.sum())
        ),
        scores=tuple(map(float, scores)),
    )
