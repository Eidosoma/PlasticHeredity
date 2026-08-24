"""Online fission-conditioned heredity-break and homeostatic-recovery scoring."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from e01_onset_discovery.recurrence_inheritance import cosine_h
from e01_onset_discovery.sustained_inheritance import exact_order_null_probability


@dataclass(frozen=True, slots=True)
class HeredityRecoveryResult:
    break_observed: bool
    break_boundary_one_based: int | None
    break_generation: int | None
    break_offset_one_based: int | None
    event: bool
    certification_boundary_one_based: int | None
    certification_generation: int | None
    certification_offset_one_based: int | None
    recovery_opportunities: int
    qualifying_recovery_count: int
    maximum_consecutive_recovery: int
    first_qualifying_recovery_boundary_one_based: int | None
    inheritance_resumption_event: bool
    inheritance_resumption_certification_boundary_one_based: int | None
    inherited_postbreak_count: int
    maximum_consecutive_inherited_postbreak: int
    maximum_postbreak_anchor_h: float | None
    maximum_inherited_postbreak_anchor_h: float | None
    recovery_progress: float
    exact_recovery_order_null_probability: float
    exact_resumption_order_null_probability: float
    recovery_flags: tuple[bool, ...]
    resumption_flags: tuple[bool, ...]


def _maximum_run_and_certification(
    values: NDArray[np.bool_], required_run: int
) -> tuple[int, int | None]:
    best = current = 0
    certification: int | None = None
    for index, value in enumerate(values):
        current = current + 1 if bool(value) else 0
        best = max(best, current)
        if current >= required_run and certification is None:
            certification = index
    return best, certification


def score_heredity_recovery(
    *,
    latest_prefix_daughter: NDArray[np.number],
    future_daughters: NDArray[np.number],
    parent_daughter_h: NDArray[np.floating],
    future_generations: NDArray[np.integer],
    future_offsets_one_based: NDArray[np.integer],
    recovery_anchor_override: NDArray[np.number] | None = None,
    threshold: float = 0.9,
    required_recovery_run: int = 2,
) -> HeredityRecoveryResult:
    """Score recovery only after the first genuine future heredity break.

    The break is detected using the actual preceding daughter.  An optional
    anchor override changes only the recovery reference, so molecule-permuted
    and unrelated-composition controls share the exact same break and future
    fission opportunities as the primary branch.
    """

    if not 0 < threshold < 1:
        raise ValueError("threshold must lie strictly between zero and one")
    if required_recovery_run < 2:
        raise ValueError("recovery run must be at least two")
    prefix = np.asarray(latest_prefix_daughter, dtype=np.int64)
    future = np.asarray(future_daughters, dtype=np.int64)
    inheritance = np.asarray(parent_daughter_h, dtype=np.float64)
    generations = np.asarray(future_generations, dtype=np.int64)
    offsets = np.asarray(future_offsets_one_based, dtype=np.int64)
    if prefix.ndim != 1 or np.any(prefix < 0) or prefix.sum() <= 0:
        raise ValueError("latest prefix daughter must have positive mass")
    if future.ndim != 2 or future.shape[1] != len(prefix):
        raise ValueError("future daughters must be feature matched")
    if len(future) != len(inheritance) or len(future) != len(generations) or len(future) != len(offsets):
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
        departed = cosine_h(previous, daughter) <= threshold
        inheritance_broken = inheritance[index] <= threshold
        if departed and inheritance_broken:
            break_index = index
            primary_anchor = previous.copy()
            break
        previous = daughter

    if break_index is None or primary_anchor is None:
        return HeredityRecoveryResult(
            break_observed=False,
            break_boundary_one_based=None,
            break_generation=None,
            break_offset_one_based=None,
            event=False,
            certification_boundary_one_based=None,
            certification_generation=None,
            certification_offset_one_based=None,
            recovery_opportunities=0,
            qualifying_recovery_count=0,
            maximum_consecutive_recovery=0,
            first_qualifying_recovery_boundary_one_based=None,
            inheritance_resumption_event=False,
            inheritance_resumption_certification_boundary_one_based=None,
            inherited_postbreak_count=0,
            maximum_consecutive_inherited_postbreak=0,
            maximum_postbreak_anchor_h=None,
            maximum_inherited_postbreak_anchor_h=None,
            recovery_progress=0.0,
            exact_recovery_order_null_probability=0.0,
            exact_resumption_order_null_probability=0.0,
            recovery_flags=(),
            resumption_flags=(),
        )

    anchor = (
        primary_anchor
        if recovery_anchor_override is None
        else np.asarray(recovery_anchor_override, dtype=np.int64)
    )
    if anchor.shape != prefix.shape or np.any(anchor < 0) or anchor.sum() <= 0:
        raise ValueError("recovery anchor must be a positive-mass matched vector")
    post_states = future[break_index + 1 :]
    post_inheritance = inheritance[break_index + 1 :] > threshold
    anchor_h = np.asarray([cosine_h(state, anchor) for state in post_states], dtype=np.float64)
    if not np.isfinite(anchor_h).all():
        raise ValueError("post-break anchor similarities must be finite")
    recovery = post_inheritance & (anchor_h > threshold)
    recovery_best, recovery_certification = _maximum_run_and_certification(
        recovery, required_recovery_run
    )
    resumption_best, resumption_certification = _maximum_run_and_certification(
        post_inheritance, required_recovery_run
    )
    first_recovery = np.flatnonzero(recovery)
    certification_index = (
        break_index + 1 + recovery_certification
        if recovery_certification is not None
        else None
    )
    resumption_index = (
        break_index + 1 + resumption_certification
        if resumption_certification is not None
        else None
    )
    inherited_anchor_h = anchor_h[post_inheritance]
    progress = float(np.max(inherited_anchor_h)) if len(inherited_anchor_h) else 0.0
    return HeredityRecoveryResult(
        break_observed=True,
        break_boundary_one_based=break_index + 1,
        break_generation=int(generations[break_index]),
        break_offset_one_based=int(offsets[break_index]),
        event=certification_index is not None,
        certification_boundary_one_based=(
            certification_index + 1 if certification_index is not None else None
        ),
        certification_generation=(
            int(generations[certification_index]) if certification_index is not None else None
        ),
        certification_offset_one_based=(
            int(offsets[certification_index]) if certification_index is not None else None
        ),
        recovery_opportunities=len(post_states),
        qualifying_recovery_count=int(recovery.sum()),
        maximum_consecutive_recovery=recovery_best,
        first_qualifying_recovery_boundary_one_based=(
            break_index + 2 + int(first_recovery[0]) if len(first_recovery) else None
        ),
        inheritance_resumption_event=resumption_index is not None,
        inheritance_resumption_certification_boundary_one_based=(
            resumption_index + 1 if resumption_index is not None else None
        ),
        inherited_postbreak_count=int(post_inheritance.sum()),
        maximum_consecutive_inherited_postbreak=resumption_best,
        maximum_postbreak_anchor_h=float(np.max(anchor_h)) if len(anchor_h) else None,
        maximum_inherited_postbreak_anchor_h=(
            float(np.max(inherited_anchor_h)) if len(inherited_anchor_h) else None
        ),
        recovery_progress=progress,
        exact_recovery_order_null_probability=exact_order_null_probability(
            len(recovery), int(recovery.sum()), required_recovery_run
        ),
        exact_resumption_order_null_probability=exact_order_null_probability(
            len(post_inheritance), int(post_inheritance.sum()), required_recovery_run
        ),
        recovery_flags=tuple(map(bool, recovery)),
        resumption_flags=tuple(map(bool, post_inheritance)),
    )
