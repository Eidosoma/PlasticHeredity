from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from sensitivity_core import (
    F12_DEFINITIONS,
    F12_THRESHOLDS,
    F32_DEFINITIONS,
    F32_THRESHOLDS,
    bray_curtis_similarity,
    quantile_matched_cutoffs,
    score_f12_array,
    score_f32_records,
)


def _identity_mapping(values):
    return {float(value): float(value) for value in values}


def test_bray_curtis_is_scale_invariant_and_bounded() -> None:
    assert bray_curtis_similarity(np.array([2, 0, 2]), np.array([1, 0, 1])) == 1.0
    assert bray_curtis_similarity(np.array([1, 0]), np.array([0, 1])) == 0.0
    observed = bray_curtis_similarity(np.array([3, 1]), np.array([1, 3]))
    assert 0.0 <= observed <= 1.0
    assert np.isclose(observed, 0.5)


def test_quantile_mapping_matches_empirical_percentiles() -> None:
    cosine = np.array([0.1, 0.2, 0.3, 0.4])
    alternative = np.array([0.4, 0.1, 0.3, 0.2])
    mapping, rows = quantile_matched_cutoffs(cosine, alternative, (0.2, 0.3))
    assert mapping == {0.2: 0.2, 0.3: 0.3}
    assert rows[0]["paired_observations"] == 4
    assert rows[0]["cosine_fraction_le"] == 0.5
    assert rows[0]["bray_curtis_fraction_le"] == 0.5


def test_f12_requires_a_break_before_the_renewed_run() -> None:
    boundary = np.array(
        [
            [0.80, 0.91, 0.92, 0.93, 0.94, 0.95] + [np.nan] * 10,
            [0.91] * 16,
        ]
    )
    labels = score_f12_array(boundary, _identity_mapping(F12_THRESHOLDS))
    index = next(
        i
        for i, item in enumerate(F12_DEFINITIONS)
        if item.source_threshold == 0.90
        and item.horizon == 8
        and item.run_length == 5
    )
    assert labels.shape == (2, len(F12_DEFINITIONS))
    assert labels[0, index] == 1
    assert labels[1, index] == 0


def test_f32_strict_windows_use_post_break_daughters_and_inclusive_anchor() -> None:
    old = np.array([10, 0, 0], dtype=np.int64)
    new = np.array([0, 10, 0], dtype=np.int64)
    records = [SimpleNamespace(parent=old, daughter=new)]
    records.extend(SimpleNamespace(parent=new, daughter=new) for _ in range(10))
    labels, onsets, boundary = score_f32_records(
        records,
        "cosine",
        _identity_mapping(F32_THRESHOLDS),
        {0.80: 0.80, 0.85: 0.85, 0.90: 0.90},
    )
    assert boundary[0] == 0.0
    assert np.all(labels == 1)
    assert set(onsets.tolist()) == {1}


def test_f32_registered_shape_is_present() -> None:
    matches = [item for item in F32_DEFINITIONS if item.registered_shape]
    assert len(matches) == 1
