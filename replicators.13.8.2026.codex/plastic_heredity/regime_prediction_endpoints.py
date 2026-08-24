"""Richer, prospectively usable summaries of strict hereditary episodes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .regime_confirmation import (
    COHERENCE_THRESHOLD,
    DISTINCTNESS_THRESHOLD,
    ENDPOINTS,
    FIRST5_LENGTH,
    INHERITANCE_THRESHOLD,
    PRIMARY_ENDPOINT,
    RUN_LENGTH,
    SECONDARY_CENTROID,
    SECONDARY_FIRST5,
    _window_geometry,
)
from .simulator import FissionRecord


WINDOW_METRIC_NAMES = (
    "start",
    "minimum_pairwise_all8",
    "minimum_pairwise_first5",
    "maximum_anchor_all8",
    "maximum_anchor_first5",
    "minimum_centroid_all8",
    "strict_margin",
    "first5_margin",
    "centroid_margin",
)


@dataclass(frozen=True)
class WindowMeasurement:
    start: int
    minimum_pairwise_all8: float
    minimum_pairwise_first5: float
    maximum_anchor_all8: float
    maximum_anchor_first5: float
    minimum_centroid_all8: float
    strict_margin: float
    first5_margin: float
    centroid_margin: float

    def to_row(self) -> NDArray[np.float64]:
        return np.asarray(tuple(asdict(self).values()), dtype=np.float64)


@dataclass(frozen=True)
class RichRegimeOutcome:
    break_event: bool
    any_run8_after_break: bool
    primary_all8: bool
    secondary_first5: bool
    secondary_centroid: bool
    primary_all8_onset: int
    secondary_first5_onset: int
    secondary_centroid_onset: int
    first_break_index: int
    first_run8_start: int
    longest_post_break_inheritance_run: int
    run8_window_count: int
    best_strict_margin: float
    best_first5_margin: float
    best_centroid_margin: float
    windows: tuple[WindowMeasurement, ...]

    @property
    def targets(self) -> tuple[bool, bool, bool]:
        return (
            self.primary_all8,
            self.secondary_first5,
            self.secondary_centroid,
        )

    @property
    def onsets(self) -> tuple[int, int, int]:
        return (
            self.primary_all8_onset,
            self.secondary_first5_onset,
            self.secondary_centroid_onset,
        )

    @property
    def best_margins(self) -> tuple[float, float, float]:
        return (
            self.best_strict_margin,
            self.best_first5_margin,
            self.best_centroid_margin,
        )


def _inclusive_distinctness_margin(
    maximum_similarity: float, threshold: float
) -> float:
    # The registered rule is inclusive (H <= threshold), while a zero margin
    # would not satisfy the desired exact ``best_margin > 0`` representation.
    # Moving the threshold by one representable float preserves all possible
    # float64 comparisons and makes the equivalence literal.
    return float(np.nextafter(threshold, np.inf) - maximum_similarity)


def _measurement(
    records: list[FissionRecord],
    start: int,
    anchor: NDArray,
    coherence_threshold: float,
    distinctness_threshold: float,
) -> WindowMeasurement:
    geometry = _window_geometry(records, start, anchor)
    strict_margin = min(
        geometry.minimum_pairwise_all8 - coherence_threshold,
        _inclusive_distinctness_margin(
            geometry.maximum_anchor_all8, distinctness_threshold
        ),
    )
    first5_margin = min(
        geometry.minimum_pairwise_first5 - coherence_threshold,
        _inclusive_distinctness_margin(
            geometry.maximum_anchor_first5, distinctness_threshold
        ),
    )
    centroid_margin = min(
        geometry.minimum_centroid_all8 - coherence_threshold,
        _inclusive_distinctness_margin(
            geometry.maximum_anchor_all8, distinctness_threshold
        ),
    )
    return WindowMeasurement(
        start=start,
        minimum_pairwise_all8=geometry.minimum_pairwise_all8,
        minimum_pairwise_first5=geometry.minimum_pairwise_first5,
        maximum_anchor_all8=geometry.maximum_anchor_all8,
        maximum_anchor_first5=geometry.maximum_anchor_first5,
        minimum_centroid_all8=geometry.minimum_centroid_all8,
        strict_margin=float(strict_margin),
        first5_margin=float(first5_margin),
        centroid_margin=float(centroid_margin),
    )


def _longest_true_run(values: Sequence[bool]) -> int:
    longest = 0
    current = 0
    for value in values:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def evaluate_rich_regime(
    records: list[FissionRecord],
    inheritance_threshold: float = INHERITANCE_THRESHOLD,
    coherence_threshold: float = COHERENCE_THRESHOLD,
    distinctness_threshold: float = DISTINCTNESS_THRESHOLD,
) -> RichRegimeOutcome:
    """Evaluate stage labels and every eligible post-break eight-run window."""

    inherited = np.asarray(
        [record.h > inheritance_threshold for record in records], dtype=bool
    )
    breaks = np.flatnonzero(~inherited)
    if breaks.size == 0:
        return RichRegimeOutcome(
            break_event=False,
            any_run8_after_break=False,
            primary_all8=False,
            secondary_first5=False,
            secondary_centroid=False,
            primary_all8_onset=-1,
            secondary_first5_onset=-1,
            secondary_centroid_onset=-1,
            first_break_index=-1,
            first_run8_start=-1,
            longest_post_break_inheritance_run=0,
            run8_window_count=0,
            best_strict_margin=np.nan,
            best_first5_margin=np.nan,
            best_centroid_margin=np.nan,
            windows=(),
        )

    first_break = int(breaks[0])
    anchor = records[first_break].parent
    tail = inherited[first_break + 1 :]
    windows: list[WindowMeasurement] = []
    last_start = len(records) - RUN_LENGTH
    for start in range(first_break + 1, last_start + 1):
        if bool(inherited[start : start + RUN_LENGTH].all()):
            windows.append(
                _measurement(
                    records,
                    start,
                    anchor,
                    coherence_threshold,
                    distinctness_threshold,
                )
            )

    def summarize(name: str) -> tuple[bool, int, float]:
        if not windows:
            return False, -1, float("nan")
        margins = np.asarray([getattr(item, name) for item in windows])
        qualifying = np.flatnonzero(margins > 0.0)
        onset = windows[int(qualifying[0])].start if qualifying.size else -1
        best = float(np.max(margins))
        return best > 0.0, onset, best

    primary, primary_onset, best_primary = summarize("strict_margin")
    first5, first5_onset, best_first5 = summarize("first5_margin")
    centroid, centroid_onset, best_centroid = summarize("centroid_margin")
    outcome = RichRegimeOutcome(
        break_event=True,
        any_run8_after_break=bool(windows),
        primary_all8=primary,
        secondary_first5=first5,
        secondary_centroid=centroid,
        primary_all8_onset=primary_onset,
        secondary_first5_onset=first5_onset,
        secondary_centroid_onset=centroid_onset,
        first_break_index=first_break,
        first_run8_start=windows[0].start if windows else -1,
        longest_post_break_inheritance_run=_longest_true_run(tail.tolist()),
        run8_window_count=len(windows),
        best_strict_margin=best_primary,
        best_first5_margin=best_first5,
        best_centroid_margin=best_centroid,
        windows=tuple(windows),
    )
    if outcome.primary_all8 != bool(outcome.best_strict_margin > 0.0):
        raise AssertionError("strict endpoint and continuous margin diverged")
    if outcome.secondary_first5 != bool(outcome.best_first5_margin > 0.0):
        raise AssertionError("first-five endpoint and continuous margin diverged")
    if outcome.secondary_centroid != bool(outcome.best_centroid_margin > 0.0):
        raise AssertionError("centroid endpoint and continuous margin diverged")
    return outcome


if ENDPOINTS != (PRIMARY_ENDPOINT, SECONDARY_FIRST5, SECONDARY_CENTROID):
    raise AssertionError("upstream endpoint order changed")
if FIRST5_LENGTH != 5 or RUN_LENGTH != 8:
    raise AssertionError("upstream regime-window contract changed")
