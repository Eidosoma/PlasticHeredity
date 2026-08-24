from __future__ import annotations

import numpy as np
import pytest

from e01_onset_discovery.attractor_atlas import (
    build_cross_lineage_atlas,
    score_atlas,
    summarize_atlas_labels,
)


def test_cross_lineage_atlas_retains_multiple_recurring_components() -> None:
    a = np.array([9, 1, 0, 0], dtype=np.int64)
    b = np.array([0, 0, 1, 9], dtype=np.int64)
    states = np.stack([a, a, b, b, a, a, b, b])
    lineage = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    generations = np.array([0, 3, 1, 4, 0, 3, 1, 4])
    atlas = build_cross_lineage_atlas(states, lineage, generations)
    assert atlas.status == "ELIGIBLE"
    assert len(atlas.components) == 2
    assert {component.lineage_counts for component in atlas.components} == {(2, 2)}


def test_atlas_excludes_single_lineage_and_short_span_components() -> None:
    states = np.array(
        [[9, 1, 0], [9, 1, 0], [0, 1, 9], [0, 1, 9]], dtype=np.int64
    )
    lineage = np.array([0, 0, 1, 1])
    generations = np.array([0, 3, 0, 1])
    atlas = build_cross_lineage_atlas(states, lineage, generations)
    assert atlas.status == "NO_CROSS_LINEAGE_RECURRING_BASIN"


def test_score_and_summary_union_semantics() -> None:
    centroids = np.array([[1.0, 0, 0], [0, 0, 1.0]])
    states = np.array([[8, 1, 0], [0, 1, 8], [1, 1, 1], [0, 1, 8]])
    scores, assignments, labels = score_atlas(states, centroids)
    assert labels.tolist() == [True, True, False, True]
    assert assignments.tolist() == [0, 1, -1, 1]
    assert np.all(scores[:2] > 0.9)
    summary = summarize_atlas_labels(labels, assignments)
    assert summary.occupancy == pytest.approx(0.75)
    assert summary.positive_episode_count == 2
    assert summary.recurrent_positive


def test_fail_closed_shapes() -> None:
    with pytest.raises(ValueError):
        build_cross_lineage_atlas(
            np.ones((3, 2), dtype=np.int64),
            np.zeros(3, dtype=np.int64),
            np.arange(3),
        )
    with pytest.raises(ValueError):
        score_atlas(np.ones((2, 3), dtype=np.int64), np.empty((0, 3)))
