from __future__ import annotations

import numpy as np
import pytest

from plastic_heredity import intervention_p3b_dose_bridge as original
from plastic_heredity import intervention_p3b_singleton_recovery as recovery
from plastic_heredity import intervention_replication as base
from plastic_heredity.experiment import StateCase
from plastic_heredity.simulator import Snapshot


def _singleton_case() -> StateCase:
    composition = np.zeros(100, dtype=np.int64)
    composition[7] = 40
    beta = np.full((100, 100), 0.02, dtype=np.float64)
    np.fill_diagonal(beta, 0.04)
    snapshot = Snapshot(
        composition=composition,
        generation=60,
        inheritance=(True, False, True, True, True),
        boundary_h=(0.95, 0.80, 0.92, 0.93, 0.94),
        previous_growth_steps=17,
        cumulative_growth_steps=81,
    )
    return StateCase(
        recovery.EXPECTED_NEXT_CASE,
        original.LABEL,
        "02",
        30,
        60,
        beta,
        snapshot,
    )


def test_original_balanced_surgery_rejects_singleton() -> None:
    case = _singleton_case()
    with pytest.raises(ValueError, match="at least two present types"):
        original.select_surgeries(
            case.snapshot.composition,
            case.beta,
            np.random.default_rng(1),
        )


def test_recovery_protocol_freezes_conservative_all_arm_no_action() -> None:
    prefix = recovery._checkpoint_prefix_audit(
        recovery.DEFAULT_WORK, require_interrupted_boundary=False
    )
    protocol = recovery._protocol(prefix)
    rule = protocol["recovery_rule"]
    assert rule["eligibility"] == "number of occupied types >= 2"
    assert rule["ineligible_action"] == "STRUCTURAL_NO_ACTION"
    assert rule["all_seven_arm_labels_use_original_beta"] is True
    assert rule["paired_contribution_to_every_contrast"] == 0.0
    assert rule["state_retained"] is True
    assert rule["matrix_retained"] is True
    assert rule["future_retried"] is False
    assert protocol["unchanged"]["eligible_state_interventions"] is True
    assert protocol["unchanged"]["inference_and_gates"] is True


def test_interrupted_prefix_is_complete_and_hashed_without_loading_outcomes() -> None:
    audit = recovery._checkpoint_prefix_audit(
        recovery.DEFAULT_WORK, require_interrupted_boundary=False
    )
    assert audit["immutable_state_checkpoint_count"] == 363
    assert len(audit["immutable_state_checkpoint_hashes"]) == 363
    assert audit["completed_futures"] == 81_312
    assert audit["next_state_id"] == recovery.EXPECTED_NEXT_CASE
    assert audit["checkpoints_deserialized"] is False
    assert audit["branch_outcomes_loaded"] is False


def test_singleton_worker_produces_bitwise_identical_all_arm_noop() -> None:
    case = _singleton_case()
    spec = original.BridgeSpec(branches=2)
    experiment = original._experiment(spec)
    batch = recovery._structural_no_action_batch(
        (
            case,
            experiment,
            spec,
            str(recovery.ORIGINAL_REGISTRATION / "frozen_full_predictor.npz"),
        )
    )
    assert batch.arm_names == original.ARMS
    assert all(edit is None for edit in batch.selected_edits)
    assert all(surgery is None for surgery in batch.surgeries)
    assert np.array_equal(
        batch.predictions,
        np.full(len(original.ARMS), batch.predictions[0]),
    )
    for branch in range(spec.branches):
        digests = {arm[branch].record_digest for arm in batch.outcomes}
        assert len(digests) == 1


def test_structural_audit_reports_exact_zero_contribution() -> None:
    case = _singleton_case()
    spec = original.BridgeSpec(branches=2)
    experiment = original._experiment(spec)
    singleton = recovery._structural_no_action_batch(
        (
            case,
            experiment,
            spec,
            str(recovery.ORIGINAL_REGISTRATION / "frozen_full_predictor.npz"),
        )
    )

    eligible_snapshot = base._fixture_snapshot()
    eligible_beta = np.full((4, 4), 0.2, dtype=np.float64)
    eligible = StateCase(
        "eligible-fixture",
        "FIX",
        "02",
        31,
        60,
        eligible_beta,
        eligible_snapshot,
    )
    surgeries = original.select_surgeries(
        eligible.snapshot.composition,
        eligible.beta,
        np.random.default_rng(19),
    )
    eligible_batch = base.PhaseBatch(
        state_id=eligible.state_id,
        state_digest=base._snapshot_digest(eligible),
        arm_names=original.ARMS,
        predictions=np.full(len(original.ARMS), 0.5, dtype=np.float64),
        selected_edits=tuple(None for _ in original.ARMS),
        surgeries=surgeries,
        scored_edits=tuple(),
        catalytic_support=np.empty(0, dtype=np.float64),
        outcomes=singleton.outcomes,
    )
    surgery_rows, structural_rows, audit = recovery._audit_interventions(
        [eligible, case], [eligible_batch, singleton]
    )
    assert not surgery_rows.empty
    assert len(structural_rows) == 1
    assert audit["structural_no_action_states"] == 1
    assert audit["structural_no_action_audit_pass"] is True
    assert audit["all_registered_states_retained"] is True


def test_recovery_contract_keeps_arm_free_future_and_original_eligible_worker() -> None:
    case = _singleton_case()
    spec = original.BridgeSpec(branches=2)
    contract = recovery._recovery_contract(
        [case], spec, recovery.EXPECTED_ORIGINAL_REGISTRATION_ID, "fixture-amendment", "generate"
    )
    assert contract["structural_no_action_condition"] == "occupied_types < 2"
    assert contract["eligible_worker_unchanged"] is True
    assert contract["future_seed_includes_arm"] is False
    assert contract["case_ids"] == [recovery.EXPECTED_NEXT_CASE]
