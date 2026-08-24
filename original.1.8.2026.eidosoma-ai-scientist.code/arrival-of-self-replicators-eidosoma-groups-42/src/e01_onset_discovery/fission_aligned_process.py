"""Helpers for fission-aligned online heredity-process outcomes."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from e01_onset_discovery.longitudinal_process_risk import (
    NewHereditaryEpisode,
    score_new_hereditary_episode,
)


def nested_process_scores(
    parent_daughter_h: Iterable[float],
    horizons: Sequence[int] = (4, 8, 12),
    *,
    threshold: float = 0.9,
    required_run: int = 3,
) -> dict[int, NewHereditaryEpisode]:
    """Score one fixed process definition at nested fission horizons."""
    values = np.asarray(tuple(parent_daughter_h), dtype=np.float64)
    if tuple(horizons) != tuple(sorted(set(horizons))):
        raise ValueError("horizons must be strictly increasing and unique")
    if not horizons or horizons[0] < required_run + 1:
        raise ValueError("every horizon must allow a break followed by the run")
    if len(values) < horizons[-1]:
        raise ValueError("insufficient fission observations for largest horizon")
    return {
        int(horizon): score_new_hereditary_episode(
            values[:horizon], threshold=threshold, required_run=required_run
        )
        for horizon in horizons
    }


def post_fission_index(selected: Sequence[Any], completed_fissions: int) -> int:
    """Return the unique selected-clock post-fission index for a generation."""
    matches = [
        index
        for index, observation in enumerate(selected)
        if observation.observation_kind == "post_fission"
        and int(observation.completed_fissions) == int(completed_fissions)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one post-fission observation for generation {completed_fissions}, "
            f"found {len(matches)}"
        )
    return matches[0]


def future_post_fission_count(selected: Sequence[Any], current_index: int) -> int:
    """Count future post-fission observations after a selected-clock state."""
    return sum(
        observation.observation_kind == "post_fission"
        for observation in selected[current_index + 1 :]
    )
