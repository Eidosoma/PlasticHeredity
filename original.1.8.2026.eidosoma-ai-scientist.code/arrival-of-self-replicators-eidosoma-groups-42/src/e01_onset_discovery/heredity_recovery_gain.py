"""Continuous homeostatic gain after a genuine fission-boundary disruption."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from e01_onset_discovery.recurrence_inheritance import cosine_h


@dataclass(frozen=True, slots=True)
class HeredityRecoveryGainResult:
    break_observed: bool
    break_boundary_one_based: int | None
    break_generation: int | None
    break_offset_one_based: int | None
    resumption_observed: bool
    resumption_certification_boundary_one_based: int | None
    resumption_certification_generation: int | None
    resumption_certification_offset_one_based: int | None
    postbreak_opportunities: int
    inherited_postbreak_count: int
    maximum_consecutive_inherited_postbreak: int
    break_anchor_h: float | None
    certification_anchor_h: float | None
    recovery_gain: float | None
    maximum_postbreak_anchor_h: float | None
    maximum_inherited_postbreak_anchor_h: float | None
    maximum_recovery_gain: float | None
    break_to_certification_h: float | None
    first_inherited_anchor_h: float | None
    second_inherited_anchor_h: float | None
    second_inherited_gain: float | None
    resumption_lag_fissions: int | None
    inherited_flags: tuple[bool, ...]


def _first_run_certification(values: NDArray[np.bool_], run: int) -> tuple[int, int | None]:
    current = best = 0
    certification: int | None = None
    for index, value in enumerate(values):
        current = current + 1 if bool(value) else 0
        best = max(best, current)
        if current >= run and certification is None:
            certification = index
    return best, certification


def score_heredity_recovery_gain(
    *,
    latest_prefix_daughter: NDArray[np.number],
    future_daughters: NDArray[np.number],
    parent_daughter_h: NDArray[np.floating],
    future_generations: NDArray[np.integer],
    future_offsets_one_based: NDArray[np.integer],
    recovery_anchor_override: NDArray[np.number] | None = None,
    threshold: float = 0.9,
    required_resumption_run: int = 2,
) -> HeredityRecoveryGainResult:
    """Measure continuous return toward the pre-break state at heredity resumption.

    The first genuine break and first two-fission inheritance resumption are
    determined from the physical path and are invariant to an optional anchor
    control. Only the H values and gains change under an anchor override.
    """

    if not 0 < threshold < 1:
        raise ValueError("threshold must lie strictly between zero and one")
    if required_resumption_run < 2:
        raise ValueError("resumption run must be at least two")
    prefix = np.asarray(latest_prefix_daughter, dtype=np.int64)
    future = np.asarray(future_daughters, dtype=np.int64)
    inheritance = np.asarray(parent_daughter_h, dtype=np.float64)
    generations = np.asarray(future_generations, dtype=np.int64)
    offsets = np.asarray(future_offsets_one_based, dtype=np.int64)
    if prefix.ndim != 1 or np.any(prefix < 0) or prefix.sum() <= 0:
        raise ValueError("latest prefix daughter must have positive mass")
    if future.ndim != 2 or future.shape[1] != len(prefix):
        raise ValueError("future daughters must be feature matched")
    if not (
        len(future) == len(inheritance) == len(generations) == len(offsets)
    ):
        raise ValueError("future arrays must align")
    if np.any(future < 0) or (len(future) and np.any(future.sum(axis=1) <= 0)):
        raise ValueError("future daughters must have positive mass")
    if not np.isfinite(inheritance).all():
        raise ValueError("parent-daughter H must be finite")
    if len(generations) and np.any(np.diff(generations) <= 0):
        raise ValueError("future generations must be strictly increasing")
    if len(offsets) and (np.any(offsets <= 0) or np.any(np.diff(offsets) <= 0)):
        raise ValueError("future offsets must be positive and strictly increasing")

    previous = prefix
    break_index: int | None = None
    primary_anchor: NDArray[np.int64] | None = None
    for index, daughter in enumerate(future):
        if cosine_h(previous, daughter) <= threshold and inheritance[index] <= threshold:
            break_index = index
            primary_anchor = previous.copy()
            break
        previous = daughter

    if break_index is None or primary_anchor is None:
        return HeredityRecoveryGainResult(
            False,
            None,
            None,
            None,
            False,
            None,
            None,
            None,
            0,
            0,
            0,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            (),
        )

    anchor = (
        primary_anchor
        if recovery_anchor_override is None
        else np.asarray(recovery_anchor_override, dtype=np.int64)
    )
    if anchor.shape != prefix.shape or np.any(anchor < 0) or anchor.sum() <= 0:
        raise ValueError("recovery anchor must be a positive-mass matched vector")

    break_state = future[break_index]
    break_anchor_h = cosine_h(break_state, anchor)
    post_states = future[break_index + 1 :]
    post_inherited = inheritance[break_index + 1 :] > threshold
    best_run, certification_relative = _first_run_certification(
        post_inherited, required_resumption_run
    )
    anchor_h = np.asarray([cosine_h(state, anchor) for state in post_states])
    inherited_anchor_h = anchor_h[post_inherited]
    maximum_h = float(anchor_h.max()) if len(anchor_h) else None
    maximum_inherited_h = (
        float(inherited_anchor_h.max()) if len(inherited_anchor_h) else None
    )
    first_inherited_h = (
        float(inherited_anchor_h[0]) if len(inherited_anchor_h) else None
    )
    second_inherited_h = (
        float(inherited_anchor_h[1]) if len(inherited_anchor_h) >= 2 else None
    )
    second_inherited_gain = (
        second_inherited_h - break_anchor_h if second_inherited_h is not None else None
    )

    if certification_relative is None:
        return HeredityRecoveryGainResult(
            True,
            break_index + 1,
            int(generations[break_index]),
            int(offsets[break_index]),
            False,
            None,
            None,
            None,
            len(post_states),
            int(post_inherited.sum()),
            best_run,
            break_anchor_h,
            None,
            None,
            maximum_h,
            maximum_inherited_h,
            maximum_inherited_h - break_anchor_h
            if maximum_inherited_h is not None
            else None,
            None,
            first_inherited_h,
            second_inherited_h,
            second_inherited_gain,
            None,
            tuple(map(bool, post_inherited)),
        )

    certification_index = break_index + 1 + certification_relative
    certification_h = cosine_h(future[certification_index], anchor)
    return HeredityRecoveryGainResult(
        True,
        break_index + 1,
        int(generations[break_index]),
        int(offsets[break_index]),
        True,
        certification_index + 1,
        int(generations[certification_index]),
        int(offsets[certification_index]),
        len(post_states),
        int(post_inherited.sum()),
        best_run,
        break_anchor_h,
        certification_h,
        certification_h - break_anchor_h,
        maximum_h,
        maximum_inherited_h,
        maximum_inherited_h - break_anchor_h
        if maximum_inherited_h is not None
        else None,
        cosine_h(break_state, future[certification_index]),
        first_inherited_h,
        second_inherited_h,
        second_inherited_gain,
        certification_index - break_index,
        tuple(map(bool, post_inherited)),
    )
