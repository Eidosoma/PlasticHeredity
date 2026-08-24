from __future__ import annotations

import numpy as np

from reviewer_lineage_identity_response.lineage_identity_core import (
    Episode,
    attractor_census,
    complete_link_clusters,
    coherent_residences,
    empirical_range_overlap,
    find_earliest_episode,
    fork_scores,
    nearest_identity_accuracy,
    probability_superiority,
    select_capable_rules,
    sibling_stranger_values,
)


def _trajectory() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    parents = np.tile(np.asarray([10, 0, 0], dtype=np.uint8), (64, 1))
    daughters = np.tile(np.asarray([10, 0, 0], dtype=np.uint8), (64, 1))
    h = np.full(64, 0.95, dtype=np.float64)
    h[32] = 0.90
    daughters[32] = np.asarray([9, 1, 0], dtype=np.uint8)
    daughters[33:41] = np.asarray([0, 10, 0], dtype=np.uint8)
    return parents, daughters, h


def _episode(marker: int) -> Episode:
    daughters = np.zeros((8, 4), dtype=np.uint8)
    daughters[:, marker] = 10
    anchor = np.zeros(4, dtype=np.uint8)
    anchor[(marker + 1) % 4] = 10
    return Episode("strict", 0, 32, 33, daughters, anchor)


def test_strict_episode_uses_registered_equalities() -> None:
    parents, daughters, h = _trajectory()
    result = find_earliest_episode(parents, daughters, h, kind="strict")
    assert result is not None
    assert result.break_index == 32
    assert result.run_start == 33
    assert np.array_equal(result.final, np.asarray([0, 10, 0], dtype=np.uint8))


def test_strict_coherence_equality_fails() -> None:
    parents, daughters, h = _trajectory()
    # Replace one daughter with a composition outside the coherent neighborhood.
    daughters[40] = np.asarray([9, 0, 0], dtype=np.uint8)
    assert find_earliest_episode(parents, daughters, h, kind="strict") is None


def test_f12_does_not_require_coherent_daughters() -> None:
    parents, daughters, h = _trajectory()
    daughters[33] = np.asarray([10, 0, 0], dtype=np.uint8)
    daughters[34] = np.asarray([0, 10, 0], dtype=np.uint8)
    daughters[35] = np.asarray([0, 0, 10], dtype=np.uint8)
    result = find_earliest_episode(parents, daughters, h, kind="f12")
    assert result is not None
    assert result.daughters.shape == (3, 3)


def test_f12_control_does_not_borrow_a_run_from_the_next_window() -> None:
    parents = np.tile(np.asarray([10, 0], dtype=np.uint8), (64, 1))
    daughters = parents.copy()
    h = np.full(64, 0.95, dtype=np.float64)
    h[32] = 0.80
    h[33:44] = 0.80
    # A run begins only after the first fixed F12 window has ended. The next
    # window contains no break, so it must not certify an event.
    h[44:47] = 0.95
    assert find_earliest_episode(
        parents, daughters, h, kind="f12", window=12
    ) is None


def test_sibling_stranger_uses_identical_split_axes() -> None:
    episodes = [_episode(0), _episode(1), _episode(2)]
    within, cross = sibling_stranger_values(episodes)
    assert np.array_equal(within, np.ones(3))
    assert np.array_equal(cross, np.zeros(6))
    assert probability_superiority(within, cross) == 1.0
    assert nearest_identity_accuracy(episodes) == 1.0
    overlap, fraction = empirical_range_overlap(within, cross)
    assert not overlap
    assert fraction == 0.0


def test_probability_superiority_gives_half_credit_to_ties() -> None:
    assert probability_superiority(np.asarray([1.0]), np.asarray([1.0])) == 0.5
    assert probability_superiority(np.asarray([2.0]), np.asarray([1.0, 2.0, 3.0])) == 0.5


def test_capable_rule_selection_is_shared_and_deterministic() -> None:
    counts = {
        "02": {index: 1 for index in range(12)},
        "03": {index: int(index != 7) for index in range(12)},
    }
    first = select_capable_rules(counts, count=5, selection_seed="fixed")
    second = select_capable_rules(counts, count=5, selection_seed="fixed")
    assert first == second
    assert len(first) == 5
    assert 7 not in first


def test_residence_windows_merge_and_record_full_duration() -> None:
    trajectory = np.zeros((24, 3), dtype=np.uint8)
    trajectory[:, 0] = 10
    episodes = coherent_residences(
        trajectory, lineage=4, burn_in=0, residence_length=8
    )
    assert len(episodes) == 1
    assert episodes[0].start == 0
    assert episodes[0].end == 23
    assert episodes[0].duration == 24


def test_complete_link_clusters_keep_distinct_textures_apart() -> None:
    values = np.asarray(
        [[10, 0, 0], [9, 1, 0], [0, 10, 0], [0, 9, 1]], dtype=np.uint8
    )
    clusters = complete_link_clusters(values, threshold=0.90)
    assert clusters == [[0, 1], [2, 3]]


def test_attractor_census_recovers_two_stable_forms() -> None:
    trajectories = np.zeros((4, 32, 3), dtype=np.uint8)
    trajectories[0:2, :, 0] = 10
    trajectories[2:4, :, 1] = 10
    result = attractor_census(
        trajectories,
        burn_in=0,
        residence_length=4,
        start_support=2,
        durable_support=2,
        separation=0.85,
    )
    assert len(result.stable_forms) == 2
    assert len(result.distinct_forms) == 2


def test_attractor_census_does_not_call_one_form_multistable() -> None:
    trajectories = np.zeros((4, 32, 3), dtype=np.uint8)
    trajectories[:, :, 0] = 10
    result = attractor_census(
        trajectories,
        burn_in=0,
        residence_length=4,
        start_support=2,
        durable_support=2,
        separation=0.85,
    )
    assert len(result.distinct_forms) == 1


def test_fork_scores_use_minimum_sibling_and_maximum_stranger() -> None:
    starts = np.asarray([[10, 0], [0, 10]], dtype=np.uint8)
    fork_a = np.repeat(starts[:, None, :], 8, axis=1)
    fork_b = fork_a.copy()
    sibling, stranger, pairs = fork_scores(starts, fork_a, fork_b)
    assert np.array_equal(sibling, np.ones(2))
    assert np.array_equal(stranger, np.zeros(2))
    assert pairs == [(0, 1), (1, 0)]


def test_fork_without_distinguishable_stranger_is_explicit() -> None:
    starts = np.asarray([[10, 0], [10, 0]], dtype=np.uint8)
    fork_a = np.repeat(starts[:, None, :], 8, axis=1)
    fork_b = fork_a.copy()
    sibling, stranger, pairs = fork_scores(starts, fork_a, fork_b)
    assert np.array_equal(sibling, np.ones(2))
    assert stranger.size == 0
    assert pairs == []
