"""Causality-safe time-window indexing and status-bearing estimate records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class TemporalIndexError(ValueError):
    """A window would violate the frozen temporal scope."""


@dataclass(frozen=True, slots=True)
class WindowIndex:
    window_start: int
    window_end: int
    window_length: int
    lag: int
    past_index_min: int
    past_index_max: int
    future_index_min: int
    future_index_max: int
    effective_sample_count: int
    prospective: bool

    def to_payload(self) -> dict[str, Any]:
        return {
            "windowStart": self.window_start,
            "windowEnd": self.window_end,
            "windowLength": self.window_length,
            "lag": self.lag,
            "pastIndexMin": self.past_index_min,
            "pastIndexMax": self.past_index_max,
            "futureIndexMin": self.future_index_min,
            "futureIndexMax": self.future_index_max,
            "effectiveSampleCount": self.effective_sample_count,
            "prospective": self.prospective,
            "usesFutureBeyondWindowEnd": self.future_index_max > self.window_end,
        }


def fixed_window_index(*, window_end: int, window_length: int, lag: int) -> WindowIndex:
    """Return the exact past/future index bounds for an inclusive past-only window."""

    if any(isinstance(value, bool) or not isinstance(value, int) for value in (window_end, window_length, lag)):
        raise TemporalIndexError("Window indices, length, and lag must be integers.")
    if window_length < 2 or lag < 1 or lag >= window_length:
        raise TemporalIndexError("INVALID_WINDOW_OR_LAG")
    window_start = window_end - window_length + 1
    if window_start < 0:
        raise TemporalIndexError("WINDOW_START_BEFORE_ZERO")
    index = WindowIndex(
        window_start=window_start,
        window_end=window_end,
        window_length=window_length,
        lag=lag,
        past_index_min=window_start,
        past_index_max=window_end - lag,
        future_index_min=window_start + lag,
        future_index_max=window_end,
        effective_sample_count=window_length - lag,
        prospective=True,
    )
    if index.future_index_max > index.window_end:
        raise TemporalIndexError("FUTURE_INDEX_BEYOND_WINDOW_END")
    return index


def sliding_endpoints(total_length: int, window_length: int) -> tuple[int, ...]:
    """Freeze first-complete, half-window cadence, and final evaluation endpoints."""

    if total_length < window_length:
        raise TemporalIndexError("Trajectory is shorter than the requested window.")
    first = window_length - 1
    cadence = window_length // 2
    endpoints = list(range(first, total_length, cadence))
    if endpoints[-1] != total_length - 1:
        endpoints.append(total_length - 1)
    return tuple(endpoints)


def whole_trajectory_index(*, total_length: int, lag: int) -> dict[str, Any]:
    """Return explicit non-prospective metadata for a whole-trajectory description."""

    if total_length < 2 or lag < 1 or lag >= total_length:
        raise TemporalIndexError("INVALID_WHOLE_TRAJECTORY_LENGTH_OR_LAG")
    return {
        "windowStart": 0,
        "windowEnd": total_length - 1,
        "windowLength": total_length,
        "lag": lag,
        "pastIndexMin": 0,
        "pastIndexMax": total_length - lag - 1,
        "futureIndexMin": lag,
        "futureIndexMax": total_length - 1,
        "effectiveSampleCount": total_length - lag,
        "prospective": False,
        "scopeLabel": "NON_PROSPECTIVE_WHOLE_TRAJECTORY_DESCRIPTION",
        "usesFutureBeyondWindowEnd": False,
    }
