from __future__ import annotations

from pathlib import Path

import numpy as np

from plastic_heredity import intervention_cr5 as cr5
from plastic_heredity import intervention_replication as base
from plastic_heredity.experiment import StateCase
from plastic_heredity.intervention_cr5r import (
    ACQUISITION_LIMIT,
    ARMS,
    BOOTSTRAP_REPETITIONS,
    BRANCHES,
    DEFAULT_CPU_BUDGET_HOURS,
    EQUIVALENCE_MARGIN,
    HORIZON,
    LANDMARKS,
    MATRICES,
    MAXIMUM_CPU_BUDGET_HOURS,
    MINIMUM_CPU_BUDGET_HOURS,
    MINIMUM_ELIGIBLE_MATRICES,
    RANDOMIZATION_REPETITIONS,
    RANDOM_RATIO_LIMIT,
    SEEDS,
    _prepare_work,
    compute_inference,
    eligibility_summary,
    future_seed,
    generate_candidate_draws,
    phase_spec,
    protocol,
    selection_seed,
)
from plastic_heredity.simulator import FissionRecord, Snapshot


def _case(candidate: str, matrix_id: int, landmark: int = 20) -> StateCase:
    snapshot = Snapshot(
        composition=np.asarray([2, 1, 1, 0], dtype=np.int64),
        generation=landmark + 1,
        inheritance=(True, False),
        boundary_h=(0.95, 0.80),
        previous_growth_steps=7,
        cumulative_growth_steps=43,
    )
    return StateCase(
        state_id=f"fixture-c{candidate}-m{matrix_id}-g{landmark}",
        cohort="ARTIFICIAL_FIXTURE",
        candidate=candidate,
        matrix_id=matrix_id,
        landmark=landmark,
        beta=np.eye(4, dtype=np.float64),
        snapshot=snapshot,
    )


def _record(h: float) -> FissionRecord:
    return FissionRecord(
        parent=np.asarray([2, 1, 1, 0], dtype=np.int64),
        daughter=np.asarray([1, 1, 0, 0], dtype=np.int64),
        h=h,
        growth_steps=3,
    )


def test_cr5r_design_is_separate_frozen_and_bounded() -> None:
    frozen = protocol()
    assert MATRICES == 250
    assert MINIMUM_ELIGIBLE_MATRICES == 200
    assert LANDMARKS == (20, 35, 50, 65, 80)
    assert BRANCHES == 64
    assert HORIZON == 8
    assert ACQUISITION_LIMIT == 60
    assert ARMS == ("RENEWAL_UP", "RENEWAL_DOWN", "RANDOM", "NOOP")
    assert MINIMUM_CPU_BUDGET_HOURS == 8.0
    assert DEFAULT_CPU_BUDGET_HOURS == 12.0
    assert MAXIMUM_CPU_BUDGET_HOURS == 14.0
    assert frozen["relationship_to_cr5"]["cr5_result_unchanged"] is True
    assert frozen["operational"]["cr6_not_launched_automatically"] is True


def test_cr5r_uses_only_the_frozen_renewal_target() -> None:
    spec = phase_spec()
    assert spec.target == "renewal"
    assert spec.stage == "resilience"
    assert spec.horizon == 8
    assert spec.arms == ARMS
    assert (
        protocol()["frozen_model"]["refit_recalibration_search_or_threshold_change"]
        is False
    )


def test_cr5r_seed_domains_are_unique_disjoint_and_arm_free() -> None:
    case = _case("02", 3)
    assert len(SEEDS) == len(set(SEEDS.values()))
    assert set(SEEDS.values()).isdisjoint(cr5.SEEDS.values())
    assert set(SEEDS.values()).isdisjoint(base.SEED_DOMAINS.values())
    assert len({future_seed(case, 9) for _arm in ARMS}) == 1
    assert selection_seed(case) != future_seed(case, 9)


def test_eligibility_is_candidate_specific_and_never_subselects() -> None:
    eligible = [
        *[_case("02", matrix_id) for matrix_id in range(200)],
        *[_case("03", matrix_id) for matrix_id in range(201)],
        _case("02", 0, 35),
    ]
    summary = eligibility_summary(eligible, 2_500, True)
    assert summary["intervention_futures_authorized"] is True
    assert summary["eligible_by_candidate"]["02"] == {
        "states": 201,
        "matrices": 200,
    }
    assert summary["eligible_by_candidate"]["03"] == {
        "states": 201,
        "matrices": 201,
    }
    assert summary["all_eligible_states_retained"] is True
    assert summary["no_retry_replacement_or_subselection"] is True


def test_eligibility_shortfall_is_inconclusive_and_launches_nothing() -> None:
    eligible = [
        *[_case("02", matrix_id) for matrix_id in range(200)],
        *[_case("03", matrix_id) for matrix_id in range(199)],
    ]
    summary = eligibility_summary(eligible, 2_500, True)
    assert summary["intervention_futures_authorized"] is False
    assert (
        summary["classification"]
        == "inconclusive_insufficient_eligible_matrix_coverage"
    )


def test_candidate_specific_matrix_block_inference_and_gates() -> None:
    cases = [
        *[_case("02", matrix_id) for matrix_id in range(200)],
        *[_case("03", matrix_id) for matrix_id in range(1, 201)],
    ]
    targets = np.zeros((len(cases), len(ARMS), BRANCHES), dtype=np.int8)
    targets[:, ARMS.index("RENEWAL_UP")] = 1
    predictions = np.full((len(cases), len(ARMS)), 0.5, dtype=np.float64)
    draws = generate_candidate_draws(cases)
    metrics, rows, stored = compute_inference(cases, targets, predictions, draws)
    assert metrics["cr5r_all_four_cells_pass"] is True
    assert len(metrics["cells"]) == 4
    assert {row["matrix_id"] for row in rows if row["candidate"] == "02"} == {
        *range(200),
    }
    assert {row["matrix_id"] for row in rows if row["candidate"] == "03"} == {
        *range(1, 201),
    }
    assert stored["c02_bootstrap_indices"].shape == (
        BOOTSTRAP_REPETITIONS,
        200,
    )
    assert stored["c03_randomization_signs"].shape == (
        RANDOMIZATION_REPETITIONS,
        200,
    )
    assert EQUIVALENCE_MARGIN == 0.025
    assert RANDOM_RATIO_LIMIT == 0.25


def test_cr5r_endpoint_is_run3_after_the_shared_break() -> None:
    inherited = np.nextafter(0.9, 1.0)
    launch = _case("02", 0).snapshot
    positive = cr5._stage_outcome(
        "resilience",
        launch,
        [_record(inherited), _record(inherited), _record(inherited)],
        True,
        HORIZON,
        0.9,
    )
    threshold_break = cr5._stage_outcome(
        "resilience",
        launch,
        [_record(inherited), _record(0.9), _record(inherited)],
        True,
        HORIZON,
        0.9,
    )
    assert positive.joint_break_run3 is True
    assert positive.renewal_certification_time == 3
    assert threshold_break.joint_break_run3 is False


def test_cr5r_protocol_excludes_strict_eight_and_broad_claims() -> None:
    frozen = protocol()
    assert frozen["endpoint"]["strict_eight_excluded"] is True
    assert "biological memory" in frozen["claim_boundary"]["prohibited"]
    assert "Phi or PhiID intervention" in frozen["claim_boundary"]["prohibited"]


def test_work_contract_enforces_cpu_bounds_and_is_resume_stable(
    tmp_path: Path,
) -> None:
    work = tmp_path / "work"
    output = tmp_path / "result"
    _prepare_work(work, output, "fixture-registration", 12.0)
    _prepare_work(work, output, "fixture-registration", 12.0)
    assert (work / "campaign_contract.json").is_file()
    try:
        _prepare_work(tmp_path / "too-small", output, "fixture", 7.99)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("CR5R accepted a CPU declaration below its bound")
