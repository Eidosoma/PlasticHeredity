from __future__ import annotations

from pathlib import Path

import numpy as np

from e01_intervention_reconstruction import core
from e01_intervention_reconstruction.core import (
    ActionSpec,
    apply_action,
    enumerate_actions,
    first_state_divergence,
    label_h900,
    simulate_condition,
    source_seeds,
    tie_rank,
    trajectory_outcomes,
    trajectory_replay_equal,
)


def test_action_set_is_exact_and_has_no_noop() -> None:
    state = np.zeros(100, dtype=np.int64)
    state[[0, 7, 99]] = [2, 1, 4]
    actions = enumerate_actions(state)
    assert len(actions) == 103
    assert [row.action_order for row in actions] == list(range(103))
    assert actions[0] == ActionSpec(0, "ADD_001", "ADD", 0)
    assert actions[99] == ActionSpec(99, "ADD_100", "ADD", 99)
    assert [row.action_id for row in actions[100:]] == [
        "DELETE_001",
        "DELETE_008",
        "DELETE_100",
    ]
    assert all(row.action_id != "NO_OP" for row in actions)
    assert int(apply_action(state, actions[0]).sum()) == 8
    assert int(apply_action(state, actions[100]).sum()) == 6


def test_source_and_tie_domains_are_deterministic_and_paired() -> None:
    assert source_seeds("S12F-CANDIDATE-02", 3, 8) == source_seeds(
        "S12F-CANDIDATE-02", 3, 8
    )
    a = tie_rank("S12F-CANDIDATE-02", 3, "MAX", 8, "ADD_001")
    b = tie_rank("S12F-CANDIDATE-02", 3, "MAX", 8, "ADD_002")
    assert len(a) == 64 and a != b


def test_literal_source_fit_uses_candidate_as_final_endpoint() -> None:
    assert Path(core.SAFE_LATTICE).is_file()
    rng = np.random.default_rng(20260807)
    states = rng.poisson(1.2, size=(40, 100)).astype(np.int64)
    states[:, 0] += 1
    candidate = states[-1].copy()
    candidate[4] += 1
    result, metadata = core._score_fit(
        decision_states=states,
        candidate_state=candidate,
        preprocessing_seed=17,
        partition_seed=23,
    )
    assert metadata["inputObservationCount"] == 41
    assert metadata["fitStateSha256"] == core.states_sha256(
        np.vstack((states, candidate))
    )
    assert result.status in core.ELIGIBLE_SOURCE_STATUSES
    assert metadata["emergence"] == float(result.emergence[-1])
    assert metadata["closureMaxAbsError"] <= 4e-16


def test_control_replay_and_label_identity(monkeypatch) -> None:
    monkeypatch.setattr(core, "N_GENERATIONS", 3)
    first = simulate_condition(
        candidate_id="S12F-CANDIDATE-02", matrix_index=9182, condition="CONTROL"
    )
    second = simulate_condition(
        candidate_id="S12F-CANDIDATE-02", matrix_index=9182, condition="CONTROL"
    )
    assert trajectory_replay_equal(first.trajectory, second.trajectory)
    h, labels = label_h900(first.trajectory)
    assert np.array_equal(labels, h > 0.9)
    outcomes = trajectory_outcomes(first.trajectory)
    assert outcomes["selectedObservationCount"] == len(first.trajectory.observations)
    assert outcomes["exactLabelIdentityMismatchCount"] == 0


def test_fixed_schedule_replay_and_first_divergence(monkeypatch) -> None:
    monkeypatch.setattr(core, "N_GENERATIONS", 2)
    control = simulate_condition(
        candidate_id="S12F-CANDIDATE-02", matrix_index=9183, condition="CONTROL"
    )
    schedule = {1: "ADD_001", 2: "ADD_001"}
    treated = simulate_condition(
        candidate_id="S12F-CANDIDATE-02",
        matrix_index=9183,
        condition="MAX",
        frozen_action_schedule=schedule,
    )
    replay = simulate_condition(
        candidate_id="S12F-CANDIDATE-02",
        matrix_index=9183,
        condition="MAX",
        frozen_action_schedule=schedule,
    )
    assert trajectory_replay_equal(treated.trajectory, replay.trajectory)
    divergence = first_state_divergence(control.trajectory, treated.trajectory)
    assert divergence["diverged"]
    assert divergence["leftObservationKind"] == "post_fission"
    assert divergence["rightObservationKind"] == "post_fission"
