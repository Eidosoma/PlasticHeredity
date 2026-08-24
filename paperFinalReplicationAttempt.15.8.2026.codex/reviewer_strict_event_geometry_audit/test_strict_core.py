from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from strict_core import (
    DOMINANCE_SHARE,
    GATE_BREAK_NO_RUN,
    GATE_COHERENCE_ANCHOR_FAIL,
    GATE_EVENT,
    GATE_NO_BREAK,
    GATE_RUN_NO_COHERENCE,
    EndpointSpec,
    calibration_comparisons,
    match_event_controls,
    quantile_match,
    score_all_specs,
    window_statistics,
)


def _record(parent, daughter, growth=4):
    return SimpleNamespace(
        parent=np.asarray(parent, dtype=np.int64),
        daughter=np.asarray(daughter, dtype=np.int64),
        growth_steps=growth,
    )


def _event_records() -> list:
    old = [10, 0, 0, 0]
    new = [0, 10, 0, 0]
    return [_record(old, new)] + [_record(new, new, growth=5) for _ in range(8)]


def test_strict_event_and_inclusive_anchor_are_scored() -> None:
    spec = EndpointSpec("test", "cosine", 0.90, 0.90, 0.0)
    outcomes, cross = score_all_specs(_event_records(), (spec,))
    outcome = outcomes[0]
    assert outcome.event
    assert outcome.onset == 1
    assert outcome.deepest_gate == GATE_EVENT
    assert outcome.eligible_windows == 1
    assert outcome.coherent_windows == 1
    assert cross.shape == (1, 1, 4)
    assert np.all(cross[0, 0] >= 0.0)


def test_failure_gates_are_distinct() -> None:
    no_break = [_record([10, 0, 0], [10, 0, 0]) for _ in range(10)]
    break_no_run = [_record([10, 0, 0], [0, 10, 0])] + [
        _record([0, 10, 0], [0, 0, 10]) for _ in range(8)
    ]
    incoherent = [_record([10, 0, 0], [0, 10, 0])] + [
        _record([0, 10, 0], [0, 10, 0]) if index % 2 == 0
        else _record([0, 0, 10], [0, 0, 10])
        for index in range(8)
    ]
    anchor_fail = [_record([10, 0, 0], [0, 10, 0])] + [
        _record([10, 0, 0], [10, 0, 0]) for _ in range(8)
    ]
    standard = EndpointSpec("standard", "cosine", 0.90, 0.90, 0.85)
    assert score_all_specs(no_break, (standard,))[0][0].deepest_gate == GATE_NO_BREAK
    assert score_all_specs(break_no_run, (standard,))[0][0].deepest_gate == GATE_BREAK_NO_RUN
    # Force the alternating daughter sequence to count as inherited while its
    # pairwise geometry remains incoherent.
    # A 0.5 boundary threshold gives the initial orthogonal break and accepts
    # the following identical parent/daughter boundaries.
    altered = EndpointSpec("altered", "cosine", 0.5, 0.90, 0.85)
    assert score_all_specs(incoherent, (altered,))[0][0].deepest_gate == GATE_RUN_NO_COHERENCE
    assert score_all_specs(anchor_fail, (standard,))[0][0].deepest_gate == GATE_COHERENCE_ANCHOR_FAIL


def test_window_statistics_detect_concentration_and_turnover() -> None:
    records = []
    for index in range(8):
        daughter = [8, 2, 0] if index % 2 == 0 else [8, 0, 2]
        records.append(_record(daughter, daughter, growth=index + 1))
    values = window_statistics(records, 0)
    assert np.isclose(values[4], DOMINANCE_SHARE)
    assert np.isclose(values[8], 1.0)
    assert np.isclose(values[9], 1.0)
    assert values[10] > 0.0
    assert values[12] > 0.0
    assert values[14] == sum(range(1, 9))


def test_calibration_objects_are_paired_and_use_unique_window_pairs() -> None:
    values = calibration_comparisons(_event_records())
    assert len(values["boundary_cosine"]) == 9
    assert len(values["boundary_bray_curtis"]) == 9
    assert len(values["coherence_cosine"]) == 28
    assert len(values["coherence_bray_curtis"]) == 28
    assert len(values["anchor_cosine"]) == 8
    assert len(values["anchor_bray_curtis"]) == 8


def test_quantile_mapping_and_control_matching_are_deterministic() -> None:
    mapping = quantile_match(
        np.asarray([0.1, 0.2, 0.3, 0.4]),
        np.asarray([0.4, 0.1, 0.3, 0.2]),
        0.2,
    )
    assert mapping["target_cutoff"] == 0.2
    labels = np.zeros((1, 6, 1), dtype=np.int8)
    labels[0, [1, 4], 0] = 1
    runs = np.zeros_like(labels, dtype=np.int16)
    runs[0, 5, 0] = -1
    left = match_event_controls(labels, runs, ["state"], ("metric",))
    right = match_event_controls(labels, runs, ["state"], ("metric",))
    assert left == right
    assert len(left) == 2
    assert len({row["control_branch"] for row in left}) == 2
