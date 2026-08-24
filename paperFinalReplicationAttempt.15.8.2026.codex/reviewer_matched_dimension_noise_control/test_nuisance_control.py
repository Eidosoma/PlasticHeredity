from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


TASK_ROOT = Path(__file__).resolve().parent
if str(TASK_ROOT) not in sys.path:
    sys.path.insert(0, str(TASK_ROOT))

from nuisance_core import event_from_h, grouped_derangement, holm_adjust, sattolo_indices  # noqa: E402


def test_sattolo_is_one_to_one_and_fixed_point_free() -> None:
    for size in (2, 3, 10, 100):
        order = sattolo_indices(size, np.random.default_rng(91 + size))
        assert np.array_equal(np.sort(order), np.arange(size))
        assert not np.any(order == np.arange(size))


def test_grouped_derangement_preserves_group_and_moves_matrix() -> None:
    matrices = np.tile(np.arange(8), 3)
    groups = np.repeat(np.arange(3), 8)
    order = np.lexsort((matrices, groups))
    matrices, groups = matrices[order], groups[order]
    donors = grouped_derangement(matrices, groups, seed=1234)
    assert np.array_equal(np.sort(donors), np.arange(donors.size))
    assert np.array_equal(groups[donors], groups)
    assert not np.any(matrices[donors] == matrices)


def test_derangement_preserves_feature_multiset_and_covariance() -> None:
    matrices = np.tile(np.arange(12), 2)
    groups = np.repeat(np.arange(2), 12)
    features = np.arange(24 * 7, dtype=float).reshape(24, 7)
    donors = grouped_derangement(matrices, groups, seed=817)
    assert np.array_equal(np.sort(features[donors], axis=0), np.sort(features, axis=0))
    assert np.array_equal(np.cov(features[donors], rowvar=False), np.cov(features, rowvar=False))


def test_event_requires_break_then_later_three_run() -> None:
    assert not event_from_h([0.95] * 12)
    assert not event_from_h([0.80, 0.95, 0.95])
    assert event_from_h([0.95, 0.80, 0.95, 0.95, 0.95])
    assert not event_from_h([0.95, 0.95, 0.95, 0.80])


def test_holm_is_monotone_in_rank_and_bounded() -> None:
    ranked = sorted(zip([0.04, 0.001, 0.02, 0.5], holm_adjust([0.04, 0.001, 0.02, 0.5])))
    assert all(0 <= adjusted <= 1 for _, adjusted in ranked)
    assert all(ranked[index][1] <= ranked[index + 1][1] for index in range(len(ranked) - 1))

