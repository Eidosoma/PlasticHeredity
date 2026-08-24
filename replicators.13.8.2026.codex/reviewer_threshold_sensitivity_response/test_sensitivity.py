from __future__ import annotations

import numpy as np

from plastic_heredity.simulator import FissionRecord
from reviewer_threshold_sensitivity_response.sensitivity_core import (
    F12_DEFINITIONS,
    F32_DEFINITIONS,
    F12Definition,
    F32Definition,
    dominant_h_component_centroid,
    score_f12_definition,
    score_f12_grid,
    score_f32_definition,
    score_f32_grid,
)


def _record(h: float, parent_marker: int, daughter_marker: int) -> FissionRecord:
    parent = np.asarray([4, 1, parent_marker, 0], dtype=np.int64)
    daughter = np.asarray([4, 1, daughter_marker, 0], dtype=np.int64)
    return FissionRecord(parent=parent, daughter=daughter, h=h, growth_steps=1)


def test_f12_strict_inheritance_and_inclusive_break() -> None:
    definition = F12Definition(0.90, 8, 3)
    assert score_f12_definition(
        np.asarray([0.90, 0.91, 0.92, 0.93, np.nan, np.nan, np.nan, np.nan]),
        definition,
    )
    assert not score_f12_definition(
        np.asarray([0.91, 0.90, 0.91, 0.92, 0.90, np.nan, np.nan, np.nan]),
        definition,
    )


def test_f12_horizon_and_run_length_boundaries() -> None:
    values = np.asarray(
        [0.80, 0.91, 0.92, 0.80, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96]
    )
    assert score_f12_definition(values, F12Definition(0.90, 8, 4))
    assert not score_f12_definition(values, F12Definition(0.90, 6, 3))
    assert score_f12_grid(np.pad(values, (0, 6), constant_values=np.nan)).shape == (
        len(F12_DEFINITIONS),
    )


def test_f32_strict_pairwise_and_inclusive_anchor() -> None:
    # The old-anchor similarity is exactly 0.80, so the cutoff's inclusive
    # semantics are exercised while all renewal daughters have pairwise H=1.
    anchor = np.asarray([6, 8, 0, 0], dtype=np.int64)
    daughter = np.asarray([0, 10, 0, 0], dtype=np.int64)
    daughters = [daughter.copy() for _ in range(7)]
    records = [
        FissionRecord(anchor, daughter, 0.88, 1),
        *[
            FissionRecord(daughter, daughter, 0.95, 1)
            for daughter in daughters
        ],
    ]
    positive, onset = score_f32_definition(records, F32Definition(0.88, 7, 0.80))
    assert positive
    assert onset == 1
    labels, onsets = score_f32_grid(records)
    assert labels.shape == (len(F32_DEFINITIONS),)
    assert onsets.shape == (len(F32_DEFINITIONS),)


def test_optimized_f32_grid_matches_definition_scorer() -> None:
    rng = np.random.default_rng(20260818)
    compositions = rng.integers(0, 12, size=(32, 10), dtype=np.int64)
    compositions[:, 0] += 1
    records = [
        FissionRecord(
            parent=compositions[max(0, index - 1)],
            daughter=compositions[index],
            h=float(rng.uniform(0.75, 0.99)),
            growth_steps=1,
        )
        for index in range(32)
    ]
    expected = [score_f32_definition(records, item) for item in F32_DEFINITIONS]
    labels, onsets = score_f32_grid(records)
    assert np.array_equal(labels, np.asarray([item[0] for item in expected], dtype=np.int8))
    assert np.array_equal(onsets, np.asarray([item[1] for item in expected], dtype=np.int16))


def test_missing_fissions_do_not_create_a_break() -> None:
    definition = F12Definition(0.90, 8, 2)
    values = np.asarray([0.95, 0.96, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    assert not score_f12_definition(values, definition)


def test_dominant_h_component_centroid_uses_size_then_earliest_tie_break() -> None:
    states = np.asarray(
        [
            [10, 0, 0],
            [9, 1, 0],
            [0, 10, 0],
            [0, 9, 1],
        ],
        dtype=np.int64,
    )
    centroid, members = dominant_h_component_centroid(states, threshold=0.90)
    assert members == (0, 1)
    assert np.allclose(centroid, np.asarray([0.95, 0.05, 0.0]))
