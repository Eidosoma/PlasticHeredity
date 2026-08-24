from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from e01_paper_pipeline_detective.core import (
    ENGINE_IDS,
    GardTrajectory,
    GenerationSummary,
    Observation,
    derive_seed,
    fission,
    generate_beta,
    initialize_distinct_state,
    poisson_vector_update,
    rates,
    select_daughter,
    simulate_trajectory,
    trajectory_replay_equal,
)
from e01_paper_pipeline_detective.information import (
    METRIC_IDS,
    common_clr_drop100,
    run_metric_branch,
    source_result_replay_equal,
)
from e01_paper_pipeline_detective.labels import LABEL_IDS, label_trajectory

CONFIG = REPO / "configs/e01/s12e_paper_pipeline_detective_preregistration.yaml"
SAFE_LATTICE = Path("/artifacts/research_steps/S12B/safe_phi_lattice.json")
ROOT = "ab" * 32


def test_preregistration_has_exact_frozen_candidate_counts_and_seed_domains() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["researchStepId"] == "E01-S12E-PAPER-PIPELINE-DETECTIVE-RECONSTRUCTION-v1.0.0"
    assert tuple(config["phase1"]["candidates"]) == ENGINE_IDS
    assert tuple(config["phase2"]["candidates"]) == LABEL_IDS
    assert tuple(config["phase3"]["branches"]) == METRIC_IDS
    assert len(config["phase4"]["scoringSemantics"]) == 3
    assert set(config["randomness"]["roots"]) == {"development", "confirmation", "intervention"}
    assert len(set(config["randomness"]["roots"].values())) == 3
    assert config["s13Status"] == "BLOCKED_PENDING_S12E_HUMAN_REVIEW"


def test_seed_derivation_and_paper_initialization_are_exact_and_replayable() -> None:
    matrix_seed = derive_seed(ROOT, "test", "catalytic_matrix", 0)
    init_seed = derive_seed(ROOT, "test", "initial_state", 0)
    assert matrix_seed == derive_seed(ROOT, "test", "catalytic_matrix", 0)
    assert matrix_seed.seed_material_sha256 != init_seed.seed_material_sha256
    beta = generate_beta(matrix_seed)
    state = initialize_distinct_state(init_seed)
    assert beta.shape == (100, 100)
    assert np.all(beta > 0)
    assert state.dtype == np.int64
    assert state.sum() == 40
    assert np.count_nonzero(state) == 40
    assert set(np.unique(state)) <= {0, 1}


def test_rate_equation_fixture_and_poisson_nonnegative_invariant() -> None:
    state = np.zeros(100, dtype=np.int64)
    state[:40] = 1
    beta = np.zeros((100, 100), dtype=np.float64)
    boost, joins, losses = rates(state, beta, 0.01)
    assert np.array_equal(boost, np.ones(100))
    assert np.allclose(joins, np.full(100, 0.004), rtol=0, atol=0)
    assert np.array_equal(losses[:40], np.full(40, 0.0001))
    assert np.array_equal(losses[40:], np.zeros(60))
    rng = np.random.Generator(np.random.PCG64DXSM(123))
    for _ in range(100):
        state = poisson_vector_update(state, beta, 0.01, rng)
        assert np.all(state >= 0)


def test_binomial_fission_and_daughter_rules_preserve_declared_semantics() -> None:
    parent = np.arange(1, 101, dtype=np.int64)
    first_rng = np.random.Generator(np.random.PCG64DXSM(456))
    a, b = fission(parent, first_rng)
    assert np.array_equal(a + b, parent)
    selected, name = select_daughter(a, b, "first_literal", first_rng)
    assert name == "A" and np.array_equal(selected, a)
    empty = np.zeros(100, dtype=np.int64)
    selected, name = select_daughter(empty, b, "uniform_nonempty", first_rng)
    assert name == "B" and np.array_equal(selected, b)


def test_complete_fast_engine_fixture_replays_and_retains_boundaries() -> None:
    first, first_seeds = simulate_trajectory(
        phase="validation_fixture",
        root_hex=ROOT,
        matrix_index=17,
        engine_id="K4_PAPER_POISSON_RHO_ONE",
    )
    replay, replay_seeds = simulate_trajectory(
        phase="validation_fixture",
        root_hex=ROOT,
        matrix_index=17,
        engine_id="K4_PAPER_POISSON_RHO_ONE",
    )
    assert trajectory_replay_equal(first, replay)
    assert first_seeds == replay_seeds
    assert first.completed_fissions == 100
    assert first.terminal_status == "requested_fissions_completed"
    assert sum(row.observation_kind == "post_fission" for row in first.observations) == 100
    assert all(row.pre_fission_mass is not None and row.pre_fission_mass >= 80 for row in first.generations)
    assert all(row.child_a_mass + row.child_b_mass == row.pre_fission_mass for row in first.generations)


def _synthetic_trajectory() -> GardTrajectory:
    states: list[np.ndarray] = []
    for generation in range(6):
        state = np.zeros(100, dtype=np.int64)
        if generation < 3:
            state[:4] = [10, 10, 10, 10]
        else:
            state[4:8] = [10, 10, 10, 10]
        states.append(state)
    observations = tuple(
        Observation(
            observation_index=index,
            observation_kind="initial_selected_state" if index == 0 else "post_fission",
            generation=index,
            growth_generation_one_based=index,
            molecular_step=index * 10,
            generation_local_step=0,
            state=tuple(map(int, state)),
        )
        for index, state in enumerate(states)
    )
    generations = tuple(
        GenerationSummary(index, "n_max_reached", 10, 80, 40, 40, "A", 40, 0)
        for index in range(1, 6)
    )
    return GardTrajectory(
        "SYNTHETIC",
        "test",
        0,
        "K1_PAPER_POISSON_RANDOM_NONEMPTY",
        "0" * 64,
        "1" * 64,
        observations,
        generations,
        5,
        50,
        "requested_fissions_completed",
        None,
        "2" * 64,
    )


def test_all_label_candidates_are_status_bearing_on_a_two_cluster_fixture() -> None:
    trajectory = _synthetic_trajectory()
    for label_id in LABEL_IDS:
        result = label_trajectory(
            trajectory,
            label_id,
            kmeans_seed_for=lambda k, replica: 1000 * k + replica,
        )
        assert result.status == "ELIGIBLE"
        assert len(result.observation_labels) == len(trajectory.observations)
        assert len(result.generation_labels) == 5


def test_common_preprocessing_is_finite_closed_clr_with_dropped_component() -> None:
    rng = np.random.Generator(np.random.PCG64DXSM(789))
    states = rng.poisson(1.5, size=(32, 100)).astype(np.int64)
    states[0] = 0
    transformed = common_clr_drop100(states)
    assert transformed.shape == (32, 99)
    assert np.all(np.isfinite(transformed))
    full_closed = (states + 0.5) / (states.sum(axis=1, keepdims=True) + 50.0)
    full_clr = np.log(full_closed) - np.log(full_closed).mean(axis=1, keepdims=True)
    assert np.allclose(transformed, full_clr[:, :99], rtol=0, atol=0)
    assert np.allclose(full_clr.sum(axis=1), 0.0, atol=2e-13)


def test_m1_m3_share_partition_and_source_replay() -> None:
    assert SAFE_LATTICE.is_file()
    rng = np.random.Generator(np.random.PCG64DXSM(999))
    states = rng.poisson(3.0, size=(40, 100)).astype(np.int64)
    clr = common_clr_drop100(states)
    first = run_metric_branch(
        clr,
        "M1_IIGR_EMERGENCE_CLR_FULL",
        SAFE_LATTICE,
        preprocessing_seed=123,
        partition_seed=456,
    )
    replay = run_metric_branch(
        clr,
        "M1_IIGR_EMERGENCE_CLR_FULL",
        SAFE_LATTICE,
        preprocessing_seed=123,
        partition_seed=456,
    )
    comparator = run_metric_branch(
        clr,
        "M3_IIGR_LOCAL_PHIR_CLR_FULL",
        SAFE_LATTICE,
        preprocessing_seed=123,
        partition_seed=456,
    )
    assert source_result_replay_equal(first, replay)
    assert first.status.startswith("ELIGIBLE")
    assert first.partition_1 == comparator.partition_1
    assert first.partition_2 == comparator.partition_2
    assert np.array_equal(first.partition_average, comparator.partition_average)


def test_phase1_selection_is_gate_only_and_never_uses_label_or_metric_fields() -> None:
    # Importing the runner is safe: it has no import-time outcome access.
    from scripts.e01.run_s12e_paper_pipeline_detective import phase1_summary

    rows = []
    for engine_id in ENGINE_IDS:
        for matrix_index in range(24):
            rows.append(
                {
                    "engineId": engine_id,
                    "matrixIndex": matrix_index,
                    "completedFissions": 100,
                    "terminalStatus": "requested_fissions_completed",
                    "totalBatchSteps": 800 + matrix_index,
                    "totalSourceObservations": 901 + matrix_index,
                    "medianPostFissionMass": 40.0,
                    "fractionGenerationsReachingNMax": 1.0,
                    "meanOvershoot": 0.1,
                    "maxstepsTerminations": 0,
                    "exactReplayPassed": True,
                }
            )
    result = phase1_summary(pd.DataFrame(rows))
    assert result["phase1Eligible"].all()
    assert result["selectedForPhase2"].sum() == 2


def test_safe_lattice_is_json_not_pickle() -> None:
    payload = json.loads(SAFE_LATTICE.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    assert SAFE_LATTICE.suffix == ".json"
