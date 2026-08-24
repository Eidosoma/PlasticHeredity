from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from plastic_heredity import intervention_replication as base
from plastic_heredity.config import CohortConfig, ExperimentConfig, GardConfig
from plastic_heredity.experiment import StateCase, build_cohort
from plastic_heredity.intervention_cr5 import (
    BREAK_HORIZON,
    CONFIRMATION_BRANCHES,
    CV_FOLDS,
    DEVELOPMENT_BRANCHES,
    EQUIVALENCE_MARGIN,
    LANDMARKS,
    MATRICES,
    MAXIMUM_CPU_HOURS,
    NATURAL_BREAK_ACQUISITION_LIMIT,
    PCA_COMPONENTS,
    RANDOM_RATIO_LIMIT,
    RENEWAL_HORIZON,
    RESILIENCE_ARMS,
    RESISTANCE_ARMS,
    RIDGE_PENALTIES,
    SEEDS,
    _first_run,
    _phase_future_seed,
    _phase_selection_seed,
    _stage_outcome,
    _inference,
    _outcome_arrays,
    add_cr5_gate_fields,
    fit_cr5_student,
    load_students,
    protocol,
    resistance_spec,
    run_phase_batches,
    resilience_spec,
    save_students,
    select_student_edits,
)
from plastic_heredity.intervention_core import MolecularEdit, ScoredEdit
from plastic_heredity.intervention_metrics import generate_inference_draws
from plastic_heredity.simulator import FissionRecord, Snapshot


def _snapshot() -> Snapshot:
    return Snapshot(
        composition=np.asarray([2, 1, 1, 0], dtype=np.int64),
        generation=20,
        inheritance=(True, False),
        boundary_h=(0.95, 0.8),
        previous_growth_steps=9,
        cumulative_growth_steps=40,
    )


def _record(h: float) -> FissionRecord:
    return FissionRecord(
        parent=np.asarray([2, 1, 1, 0], dtype=np.int64),
        daughter=np.asarray([1, 1, 0, 0], dtype=np.int64),
        h=h,
        growth_steps=3,
    )


def _passing_cell() -> dict:
    return {
        "contrasts": {
            "up_minus_down": {"estimate": 0.1, "bootstrap_ci95": (0.05, 0.15)}
        },
        "up_down_randomization_p_holm": 0.01,
        "random_noop_equivalence": {
            "tost_equivalent": True,
            "absolute_difference_within_ratio": True,
        },
    }


def test_cr5_design_is_frozen_and_bounded() -> None:
    frozen = protocol()
    assert MATRICES == 200
    assert LANDMARKS == (20, 35, 50, 65, 80)
    assert DEVELOPMENT_BRANCHES == 32
    assert CONFIRMATION_BRANCHES == 64
    assert BREAK_HORIZON == 6
    assert RENEWAL_HORIZON == 8
    assert NATURAL_BREAK_ACQUISITION_LIMIT == 60
    assert PCA_COMPONENTS == 12
    assert RIDGE_PENALTIES == (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
    assert CV_FOLDS == 5
    assert MAXIMUM_CPU_HOURS == 30.0
    assert frozen["resistance"]["primary_futures"] == 512_000
    assert frozen["operational"]["cr6_not_launched_automatically"] is True


def test_cr5_arms_and_primary_targets_are_separate() -> None:
    assert RESISTANCE_ARMS == ("BREAK_UP", "BREAK_DOWN", "RANDOM", "NOOP")
    assert RESILIENCE_ARMS == ("RENEWAL_UP", "RENEWAL_DOWN", "RANDOM", "NOOP")
    assert resistance_spec().target == "break"
    assert resilience_spec().target == "renewal"


def test_first_run_and_strict_threshold_edges() -> None:
    inherited = np.nextafter(0.9, 1.0)
    snapshot = _snapshot()
    assert _first_run(np.asarray([True, True, False, True, True, True]), 3) == 6
    assert _first_run(np.asarray([True, False, True]), 3) == -1
    resistance = _stage_outcome("resistance", snapshot, [_record(0.9)], True, 1, 0.9)
    renewal = _stage_outcome(
        "resilience",
        snapshot,
        [_record(inherited), _record(inherited), _record(inherited)],
        True,
        3,
        0.9,
    )
    uninterrupted_break = _stage_outcome(
        "resistance", snapshot, [_record(inherited)], True, 1, 0.9
    )
    assert resistance.joint_break_run3 is True
    assert uninterrupted_break.joint_break_run3 is False
    assert renewal.joint_break_run3 is True
    assert renewal.renewal_certification_time == 3


def test_cr5_seed_domains_are_unique_arm_free_and_separated() -> None:
    assert len(SEEDS) == len(set(SEEDS.values()))
    assert set(SEEDS.values()).isdisjoint(base.SEED_DOMAINS.values())
    case = StateCase("seed-fixture", "FIX", "02", 5, 20, np.eye(4), _snapshot())
    for spec in (resistance_spec(), resilience_spec()):
        seeds = {_phase_future_seed(spec, case, 2) for _arm in spec.arms}
        assert len(seeds) == 1
        assert _phase_selection_seed(spec, case) not in seeds


def test_extreme_and_random_selection_are_deterministic() -> None:
    scores = (
        ScoredEdit(MolecularEdit(0, 1), 0.2, -0.3),
        ScoredEdit(MolecularEdit(0, 2), 0.8, 0.3),
        ScoredEdit(MolecularEdit(2, 1), 0.5, 0.0),
    )
    left = select_student_edits(0.5, scores, np.random.default_rng(17))
    right = select_student_edits(0.5, scores, np.random.default_rng(17))
    assert np.array_equal(left[0], right[0])
    assert left[1] == right[1]
    assert left[1][0] == MolecularEdit(0, 2)
    assert left[1][1] == MolecularEdit(0, 1)
    assert left[1][-1] is None


def test_cr5_gate_does_not_require_up_noop_or_noop_down() -> None:
    metrics = {"cells": [_passing_cell() for _ in range(4)]}
    add_cr5_gate_fields(metrics, "resistance")
    assert metrics["cr5_all_four_cells_pass"] is True
    assert set(metrics["cells"][0]["cr5_registered_gates"]) == {
        "up_minus_down_positive",
        "up_minus_down_bootstrap_lower_positive",
        "holm_randomization_below_0_05",
        "random_tost_equivalent_to_noop",
        "random_absolute_difference_within_effect_ratio",
    }


def test_cr5_gate_requires_random_equivalence_and_ratio() -> None:
    cells = [_passing_cell() for _ in range(4)]
    cells[2]["random_noop_equivalence"]["tost_equivalent"] = False
    metrics = {"cells": cells}
    add_cr5_gate_fields(metrics, "resilience")
    assert metrics["cr5_all_four_cells_pass"] is False
    assert EQUIVALENCE_MARGIN == 0.025
    assert RANDOM_RATIO_LIMIT == 0.25


def test_candidate_separated_student_round_trip_is_exact(tmp_path: Path) -> None:
    rng = np.random.default_rng(23)
    states = rng.normal(size=(40, 195))
    history = rng.normal(size=(40, 9))
    labels = rng.binomial(1, 0.4, size=(40, 8)).astype(np.int8)
    matrix_ids = np.arange(40, dtype=np.int64)
    student, diagnostics = fit_cr5_student(
        "break", "02", states, history, labels, matrix_ids
    )
    archive = tmp_path / "models.npz"
    contract = tmp_path / "contract.json"
    save_students(
        archive,
        contract,
        {("break", "02"): student},
        {"break__c02": diagnostics},
    )
    restored = load_students(archive, contract)[("break", "02")]
    assert np.array_equal(
        student.predict_features(states, history),
        restored.predict_features(states, history),
    )
    assert student.ridge_penalty in RIDGE_PENALTIES
    assert student.coefficient.size == PCA_COMPONENTS + history.shape[1]


def test_protocol_prohibits_strict_eight_and_replacement() -> None:
    frozen = protocol()
    assert "strict-eight excluded" in frozen["target"]
    assert frozen["randomness"]["no_matrix_or_source_replacement"] is True
    assert frozen["resilience"]["all_200_matrices_per_candidate_required"] is True


def test_non_scientific_worker_checkpoint_replay_and_matrix_inference(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(41)
    states = rng.normal(size=(40, 195))
    history = rng.normal(size=(40, 9))
    labels = rng.binomial(1, 0.4, size=(40, 4)).astype(np.int8)
    ids = np.arange(40, dtype=np.int64)
    students = {}
    diagnostics = {}
    for target in ("break", "renewal"):
        for candidate in ("02", "03"):
            student, item = fit_cr5_student(
                target, candidate, states, history, labels, ids
            )
            students[(target, candidate)] = student
            diagnostics[f"{target}__c{candidate}"] = item
    model = tmp_path / "models.npz"
    contract = tmp_path / "contract.json"
    save_students(model, contract, students, diagnostics)

    gard = GardConfig(
        n_types=8,
        n_min=4,
        n_max=8,
        beta_log_mean=-1.0,
        beta_log_sd=1.0,
        k_join=0.05,
        k_leave=0.0,
        max_growth_steps=2_000,
        generations=5,
    )
    cohort = CohortConfig(2, 2, (5,))
    experiment = ExperimentConfig(
        gard=gard,
        development=cohort,
        confirmation=cohort,
        horizon=2,
        bootstrap_repetitions=32,
        permutation_repetitions=32,
        master_seed=SEEDS["validation"],
    )
    cases = build_cohort(experiment, "CR5_ARTIFICIAL_TEST", cohort)
    spec = replace(
        resistance_spec(),
        branches=2,
        horizon=2,
        selection_seed=SEEDS["smoke_selection"],
        future_seed=SEEDS["smoke_future"],
    )
    generated = run_phase_batches(
        cases,
        gard,
        spec,
        model,
        contract,
        "non-scientific-registration",
        tmp_path / "generate",
        1,
        "generate",
    )
    replayed = run_phase_batches(
        cases,
        gard,
        spec,
        model,
        contract,
        "non-scientific-registration",
        tmp_path / "replay",
        1,
        "replay",
    )
    assert base.replay_audit(generated, replayed)[
        "state_edit_endpoint_and_process_digests_exact"
    ]
    arrays = _outcome_arrays(cases, generated, spec)
    draws = generate_inference_draws(
        2,
        32,
        32,
        np.random.default_rng(43),
        np.random.default_rng(47),
    )
    metrics, rows = _inference(cases, arrays, spec, draws)
    assert len(metrics["cells"]) == 4
    assert len(rows) == 8
    assert arrays["targets"].shape == (4, 4, 2)
