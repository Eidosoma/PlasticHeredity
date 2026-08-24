from dataclasses import dataclass

import pytest

from e01_onset_discovery.fission_aligned_process import (
    future_post_fission_count,
    nested_process_scores,
    post_fission_index,
)


@dataclass(frozen=True)
class Observation:
    observation_kind: str
    completed_fissions: int


def test_nested_events_are_monotone_in_horizon() -> None:
    scores = nested_process_scores(
        [0.2, 0.91, 0.92, 0.93, 0.1, 0.91, 0.92, 0.93, 0.95, 0.95, 0.95, 0.95]
    )
    assert scores[4].event
    assert scores[8].event
    assert scores[12].event


def test_late_certification_enters_only_later_horizon() -> None:
    scores = nested_process_scores(
        [0.95, 0.95, 0.2, 0.91, 0.92, 0.93, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95]
    )
    assert not scores[4].event
    assert scores[8].event
    assert scores[12].event


def test_horizon_contract_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        nested_process_scores([0.2] * 12, (8, 4, 12))
    with pytest.raises(ValueError):
        nested_process_scores([0.2] * 8, (4, 8, 12))


def test_unique_post_fission_index_and_future_count() -> None:
    selected = [
        Observation("molecular_update", 0),
        Observation("post_fission", 1),
        Observation("molecular_update", 1),
        Observation("post_fission", 2),
        Observation("molecular_update", 2),
        Observation("post_fission", 3),
    ]
    assert post_fission_index(selected, 2) == 3
    assert future_post_fission_count(selected, 3) == 1
    with pytest.raises(ValueError):
        post_fission_index(selected, 4)
