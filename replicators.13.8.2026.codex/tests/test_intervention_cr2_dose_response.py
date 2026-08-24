from types import SimpleNamespace

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from plastic_heredity import intervention_replication as base
from plastic_heredity.config import CohortConfig, ExperimentConfig
from plastic_heredity.experiment import build_cohort
from plastic_heredity.intervention_core import MolecularEdit, ScoredEdit
from plastic_heredity.intervention_cr2_dose_response import (
    ARMS,
    BOOTSTRAP_REPETITIONS,
    BRANCHES,
    LANDMARKS,
    MATRICES,
    MINIMUM_AVAILABLE_CPU_HOURS,
    QUANTILES,
    RANDOMIZATION_REPETITIONS,
    SEEDS,
    DoseSpec,
    _future_seed,
    _readback_audit,
    _write_inference_arrays,
    _write_selection_artifacts,
    compute_dose_inference,
    phase_spec,
    protocol,
    run_batches,
    select_quantile_edits,
    state_spearman,
)
from plastic_heredity.intervention_metrics import generate_inference_draws


def _score(index: int, probability: float) -> ScoredEdit:
    return ScoredEdit(
        MolecularEdit(index // 10, index % 10),
        probability,
        probability - 0.5,
    )


def test_full_cr2_design_matches_directive() -> None:
    frozen = protocol()
    assert MATRICES == 200
    assert BRANCHES == 64
    assert LANDMARKS == (20, 35, 50, 65, 80)
    assert QUANTILES == (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    assert ARMS == ("Q00", "Q20", "Q40", "Q60", "Q80", "Q100")
    assert frozen["futures"]["primary_futures"] == 768_000
    assert frozen["futures"]["replay_futures"] == 768_000
    assert MINIMUM_AVAILABLE_CPU_HOURS == 18.0


def test_quantile_selection_uses_frozen_nearest_order_statistics() -> None:
    scores = tuple(_score(index, float(index)) for index in range(11))
    selected, ranks = select_quantile_edits(scores)
    assert np.array_equal(ranks, [0, 2, 4, 6, 8, 10])
    assert [item.predicted_probability for item in selected] == [
        0.0,
        2.0,
        4.0,
        6.0,
        8.0,
        10.0,
    ]


def test_quantile_ties_choose_lexicographically_first_edit() -> None:
    scores = (
        ScoredEdit(MolecularEdit(2, 3), 0.1, -0.4),
        ScoredEdit(MolecularEdit(0, 3), 0.1, -0.4),
        ScoredEdit(MolecularEdit(4, 1), 0.9, 0.4),
        ScoredEdit(MolecularEdit(1, 4), 0.9, 0.4),
    )
    selected, _ = select_quantile_edits(scores)
    assert selected[0].edit == MolecularEdit(0, 3)
    assert selected[-1].edit == MolecularEdit(1, 4)


def test_constant_state_spearman_is_registered_zero() -> None:
    assert state_spearman(np.arange(6), np.ones(6)) == 0.0
    assert state_spearman(np.ones(6), np.arange(6)) == 0.0
    assert state_spearman(np.arange(6), np.arange(6)) == 1.0


def test_cr2_seeds_are_new_and_future_seed_is_arm_free() -> None:
    spec = phase_spec()
    assert len(SEEDS) == len(set(SEEDS.values()))
    assert not set(SEEDS.values()).intersection(base.SEED_DOMAINS.values())
    case = SimpleNamespace(candidate="02", matrix_id=7, landmark=35)
    seed = _future_seed(spec, case, 4)  # type: ignore[arg-type]
    assert seed == _future_seed(spec, case, 4)  # type: ignore[arg-type]
    assert seed != _future_seed(spec, case, 5)  # type: ignore[arg-type]


def test_matrix_block_dose_inference_recovers_positive_gradient() -> None:
    cases = [
        SimpleNamespace(candidate=candidate, matrix_id=matrix_id)
        for matrix_id in range(2)
        for candidate in ("02", "03")
    ]
    predictions = np.tile(np.linspace(0.1, 0.9, 6), (len(cases), 1))
    targets = np.zeros((len(cases), 6, 4), dtype=np.int8)
    targets[:, 2:4, (1, 3)] = 1
    targets[:, 4:6, :] = 1
    spec = DoseSpec(
        role="fixture",
        matrices=2,
        branches=4,
        cohort_seed="fixture-cohort",
        selection_seed="fixture-selection",
        future_seed="fixture-future",
        bootstrap_seed="fixture-bootstrap",
        randomization_seed="fixture-randomization",
    )
    draws = generate_inference_draws(
        2,
        BOOTSTRAP_REPETITIONS,
        RANDOMIZATION_REPETITIONS,
        np.random.default_rng(3),
        np.random.default_rng(5),
    )
    metrics, rows, stored = compute_dose_inference(
        cases, targets, predictions, draws, spec  # type: ignore[arg-type]
    )
    assert metrics["registered_all_four_cells_pass"]
    assert len(metrics["cells"]) == 4
    assert len(rows) == 8
    assert stored["bootstrap_indices"].shape == (BOOTSTRAP_REPETITIONS, 2)
    for cell in metrics["cells"]:
        assert cell["mean_within_state_spearman"] > 0.0
        assert cell["spearman_bootstrap_ci95"][0] > 0.0
        assert cell["state_centered_calibration_slope"] > 0.0
        assert cell["slope_bootstrap_ci95"][0] > 0.0


def test_non_scientific_checkpoint_and_artifact_roundtrip(tmp_path) -> None:
    cohort = CohortConfig(2, 2, (5,))
    spec = DoseSpec(
        role="test fixture",
        matrices=2,
        branches=2,
        cohort_seed=SEEDS["smoke_cohort"],
        selection_seed=SEEDS["selection_audit"],
        future_seed=SEEDS["smoke_future"],
        bootstrap_seed=SEEDS["validation"],
        randomization_seed=SEEDS["replay"],
    )
    current_experiment = ExperimentConfig(
        development=cohort,
        confirmation=cohort,
        horizon=12,
        master_seed=spec.cohort_seed,
        bootstrap_repetitions=BOOTSTRAP_REPETITIONS,
        permutation_repetitions=RANDOMIZATION_REPETITIONS,
    )
    with threadpool_limits(limits=1):
        cases = build_cohort(current_experiment, "INTCR2_TEST_FIXTURE", cohort)
    model_path = (
        base.RESULT_ROOT / "cr1_confirmation_registration/frozen_full_predictor.npz"
    )
    generated = run_batches(
        cases,
        current_experiment,
        spec,
        model_path,
        "test-registration",
        tmp_path / "generate",
        1,
        "generate",
    )
    replayed = run_batches(
        cases,
        current_experiment,
        spec,
        model_path,
        "test-registration",
        tmp_path / "replay",
        1,
        "replay",
    )
    assert base.replay_audit(generated, replayed)[
        "state_edit_endpoint_and_process_digests_exact"
    ]
    arrays = base._outcome_arrays(cases, generated, spec)  # type: ignore[arg-type]
    draws = generate_inference_draws(
        2,
        BOOTSTRAP_REPETITIONS,
        RANDOMIZATION_REPETITIONS,
        np.random.default_rng(13),
        np.random.default_rng(17),
    )
    metrics, rows, inference_arrays = compute_dose_inference(
        cases, arrays["targets"], arrays["predictions"], draws, spec
    )
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    np.savez_compressed(artifacts / "branch_arrays.npz", **arrays)
    selected = _write_selection_artifacts(artifacts, cases, generated)
    _write_inference_arrays(artifacts / "inference_arrays.npz", inference_arrays)
    pd.DataFrame(rows).to_csv(
        artifacts / "matrix_effects.csv", index=False, float_format="%.17g"
    )
    audit = _readback_audit(
        artifacts, cases, generated, spec, metrics, rows, selected
    )
    assert all(audit.values())
