"""Online break-and-renewed-heredity process scoring for S19-L49.

The event is process based and bounded entirely by a fixed future fission
horizon.  It never refers to a completed-run centroid or another future-defined
destination.  A branch first has to exhibit a genuine parent/daughter
inheritance break and then certify a new hereditary episode through a fixed
run of inherited fissions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class NewHereditaryEpisode:
    """Bounded online process outcome on the fission clock."""

    break_observed: bool
    break_boundary_one_based: int | None
    postbreak_opportunities: int
    postbreak_inherited_count: int
    maximum_postbreak_run: int
    event: bool
    certification_boundary_one_based: int | None
    inheritance_flags: tuple[bool, ...]
    postbreak_flags: tuple[bool, ...]


def _run_summary(values: NDArray[np.bool_], required_run: int) -> tuple[int, int | None]:
    best = 0
    current = 0
    certification: int | None = None
    for index, value in enumerate(values):
        current = current + 1 if bool(value) else 0
        best = max(best, current)
        if current >= required_run and certification is None:
            certification = index
    return best, certification


def score_new_hereditary_episode(
    parent_daughter_h: NDArray[np.floating] | list[float] | tuple[float, ...],
    *,
    threshold: float = 0.9,
    required_run: int = 3,
) -> NewHereditaryEpisode:
    """Score the first break followed by a fixed inherited-fission run.

    Strict ``H > threshold`` denotes inheritance.  The first non-inherited
    fission is the break.  Only later fissions can certify the new episode, so
    uninterrupted inheritance is explicitly ineligible rather than a positive
    continuation event.
    """

    if not 0 < threshold < 1:
        raise ValueError("threshold must lie strictly between zero and one")
    if required_run < 2:
        raise ValueError("required_run must be at least two")
    scores = np.asarray(parent_daughter_h, dtype=np.float64)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("parent_daughter_h must be a finite one-dimensional array")
    inherited = scores > threshold
    breaks = np.flatnonzero(~inherited)
    if not len(breaks):
        return NewHereditaryEpisode(
            break_observed=False,
            break_boundary_one_based=None,
            postbreak_opportunities=0,
            postbreak_inherited_count=0,
            maximum_postbreak_run=0,
            event=False,
            certification_boundary_one_based=None,
            inheritance_flags=tuple(map(bool, inherited)),
            postbreak_flags=(),
        )
    break_index = int(breaks[0])
    postbreak = inherited[break_index + 1 :]
    best, certification = _run_summary(postbreak, required_run)
    absolute_certification = (
        break_index + 1 + certification if certification is not None else None
    )
    return NewHereditaryEpisode(
        break_observed=True,
        break_boundary_one_based=break_index + 1,
        postbreak_opportunities=len(postbreak),
        postbreak_inherited_count=int(postbreak.sum()),
        maximum_postbreak_run=best,
        event=certification is not None,
        certification_boundary_one_based=(
            absolute_certification + 1
            if absolute_certification is not None
            else None
        ),
        inheritance_flags=tuple(map(bool, inherited)),
        postbreak_flags=tuple(map(bool, postbreak)),
    )


def trailing_true_run(values: NDArray[np.bool_] | list[bool] | tuple[bool, ...]) -> int:
    """Return the number of consecutive true values at the end of a sequence."""

    array = np.asarray(values, dtype=np.bool_)
    if array.ndim != 1:
        raise ValueError("values must be one-dimensional")
    count = 0
    for value in array[::-1]:
        if not bool(value):
            break
        count += 1
    return count


def jeffreys_mean(successes: int, trials: int) -> float:
    """Jeffreys posterior mean for a Bernoulli probability."""

    if successes < 0 or trials < 0 or successes > trials:
        raise ValueError("invalid Bernoulli counts")
    return float((successes + 0.5) / (trials + 1.0))
