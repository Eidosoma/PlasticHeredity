"""Time-resolved transition-tube representations for S19-L27."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from e01_onset_discovery.outcome_blind_representation import (
    CHANNEL_NAMES,
    organization_channel_sequence,
)

WINDOW_COUNT = 32
TUBE_VIEWS = (
    "FULL_TRANSITION_TUBE",
    "EXACT_H_TRANSITION_TUBE",
    "ORDINARY_TRANSITION_TUBE",
)


def _level_and_current(channels: NDArray[np.float64]) -> NDArray[np.float64]:
    current = np.diff(channels, axis=0)
    result = np.concatenate([channels.reshape(-1), current.reshape(-1)])
    if not np.isfinite(result).all():
        raise RuntimeError("transition-tube representation is nonfinite")
    return result


def transition_tube_views(
    states: NDArray[np.integer[Any]],
) -> dict[str, NDArray[np.float64]]:
    counts = np.asarray(states, dtype=np.int64)
    if counts.shape != (WINDOW_COUNT, 100):
        raise ValueError("transition-tube window must be 32-by-100")
    # The shared organization routine is frozen at 64 rows. Duplicate the
    # 32-state window only to invoke that audited channel calculation, then
    # retain its first half; no duplicated value enters the representation.
    channels = organization_channel_sequence(np.concatenate([counts, counts], axis=0))[
        :WINDOW_COUNT
    ]
    result = {
        "FULL_TRANSITION_TUBE": _level_and_current(channels),
        "EXACT_H_TRANSITION_TUBE": _level_and_current(channels[:, 6:]),
        "ORDINARY_TRANSITION_TUBE": _level_and_current(channels[:, :6]),
    }
    expected = {
        "FULL_TRANSITION_TUBE": 32 * 11 + 31 * 11,
        "EXACT_H_TRANSITION_TUBE": 32 * 5 + 31 * 5,
        "ORDINARY_TRANSITION_TUBE": 32 * 6 + 31 * 6,
    }
    if tuple(result) != TUBE_VIEWS or any(
        result[key].shape != (expected[key],) for key in result
    ):
        raise RuntimeError("transition-tube schema changed")
    return result


def channel_schema() -> dict[str, tuple[str, ...]]:
    return {
        "FULL_TRANSITION_TUBE": CHANNEL_NAMES,
        "EXACT_H_TRANSITION_TUBE": CHANNEL_NAMES[6:],
        "ORDINARY_TRANSITION_TUBE": CHANNEL_NAMES[:6],
    }
