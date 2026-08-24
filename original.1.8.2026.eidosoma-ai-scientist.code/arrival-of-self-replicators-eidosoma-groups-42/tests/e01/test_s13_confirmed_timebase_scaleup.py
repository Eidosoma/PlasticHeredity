from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from e01_confirmed_timebase_scaleup.core import (
    ANALYSIS_ROOT_SEED_HEX,
    CANDIDATE_IDS,
    DEFINED_AT_LEAST,
    POSITIVE_ASSOCIATIONS_AT_LEAST,
    POSITIVE_DRIFT_AT_LEAST,
    RAW_LJUNG_BOX_AT_LEAST,
    SPIKED_RUNS_AT_LEAST,
    association_gate,
    candidate_registry,
    combined_classification,
    derive_analysis_seed,
    drift_gate,
)
from e01_latent_timebase.core import (
    ExposureDefinition,
    SimulationDefinition,
    derive_seed,
    generate_beta,
    initialize_distinct_state,
    simulate_trajectory,
)
from e01_replay_repair.comparator import compare_seed_tuples, compare_trajectories

REPO = Path(__file__).resolve().parents[2]


def test_preregistration_parses_and_freezes_exact_scope() -> None:
    config = yaml.safe_load(
        (
            REPO
            / "configs/e01/s13_confirmed_timebase_baseline_scaleup_preregistration.yaml"
        ).read_text(encoding="utf-8")
    )
    assert config["versionedStepId"] == (
        "E01-S13-CONFIRMED-TIMEBASE-BASELINE-SCALEUP-v1.0.0"
    )
    assert config["simulation"]["independentUnits"] == 100
    assert config["simulation"]["retainedTrajectoryCount"] == 200
    assert config["simulation"]["validationReplayExecutionCount"] == 200
    assert config["compute"]["cumulativeProjectedCpuHours"] == 144.0
    assert config["forbiddenWork"] == [
        "candidate_1_execution_or_reclassification",
        "new_candidate_or_exposure_search",
        "prediction",
        "MLP",
        "intervention",
        "estimator_repair",
        "S14_through_S18",
        "report_bundle_progression",
        "E02",
        "further_scaleup",
    ]


def test_candidate_registry_contains_only_two_exact_confirmed_candidates() -> None:
    rows = candidate_registry()
    assert tuple(row["candidateId"] for row in rows) == CANDIDATE_IDS
    assert {row["h"] for row in rows} == {
        0.6031526490073492,
        0.5613315384859516,
    }
    assert {row["clockId"] for row in rows} == {
        "C1_SELECTED_DAUGHTER_RETAINED"
    }
    assert {row["overshootRule"] for row in rows} == {
        "TRIM_NEW_ENTRANTS_TO_NMAX"
    }
    assert all(row["evidenceStatus"] == "S12FR_UPSTREAM_CONFIRMED" for row in rows)
    assert all("CANDIDATE-01" not in row["candidateId"] for row in rows)


def test_seed_domain_is_deterministic_and_distinct_from_s12g() -> None:
    assert len(ANALYSIS_ROOT_SEED_HEX) == 64
    int(ANALYSIS_ROOT_SEED_HEX, 16)
    first = derive_analysis_seed("statistics", CANDIDATE_IDS[0], "bootstrap")
    assert first == derive_analysis_seed("statistics", CANDIDATE_IDS[0], "bootstrap")
    assert first != derive_analysis_seed("statistics", CANDIDATE_IDS[1], "bootstrap")
    assert first != derive_analysis_seed("suffix", CANDIDATE_IDS[0], "bootstrap")


def test_scaled_counts_are_frozen_upward_from_s12j_contract() -> None:
    assert DEFINED_AT_LEAST == 80
    assert POSITIVE_ASSOCIATIONS_AT_LEAST == 75
    assert POSITIVE_DRIFT_AT_LEAST == 59
    assert SPIKED_RUNS_AT_LEAST == 75
    assert RAW_LJUNG_BOX_AT_LEAST == 88


def test_scaled_association_and_drift_gates_preserve_all_continuous_rules() -> None:
    association = SimpleNamespace(
        defined_count=80,
        positive_count=75,
        median=0.01,
        bootstrap_lower_95=0.001,
        circular_shift_positive_p=0.05,
    )
    drift = SimpleNamespace(
        defined_count=80,
        positive_count=59,
        median_mean_difference=0.01,
        bootstrap_lower_95=0.001,
        block_aware_positive_p=0.05,
    )
    assert association_gate(association, 0.80)
    assert drift_gate(drift)
    association.positive_count = 74
    drift.positive_count = 58
    assert not association_gate(association, 0.80)
    assert not drift_gate(drift)


def test_two_candidate_adjudication_requires_both_retrospective_and_prefix() -> None:
    positive = [
        {
            "candidateId": candidate,
            "primaryFullCoherent": True,
            "primaryPrefixGate": True,
        }
        for candidate in CANDIDATE_IDS
    ]
    assert combined_classification(positive).endswith("PROSPECTIVE_SUPPORT")
    positive[1]["primaryPrefixGate"] = False
    assert combined_classification(positive).endswith("UNDERDETERMINATION")
    positive[0]["primaryFullCoherent"] = False
    assert combined_classification(positive).endswith(
        "NON_SUPPORT_WITHIN_SOURCE_INFORMED_SCOPE"
    )
    assert combined_classification(positive[:1]) == "S13_VALIDATION_FAILED_CLOSED"


def test_small_exact_simulator_replay_uses_shared_matrix_and_initial_state() -> None:
    root = ANALYSIS_ROOT_SEED_HEX
    phase = "s13_unit_test"
    matrix_index = 0
    beta = generate_beta(derive_seed(root, phase, "catalytic_matrix", matrix_index))
    initial = initialize_distinct_state(
        derive_seed(root, phase, "initial_state", matrix_index)
    )
    definition = SimulationDefinition(
        daughter_rule="FIRST_DAUGHTER",
        overshoot_rule="TRIM_NEW_ENTRANTS_TO_NMAX",
        exposure=ExposureDefinition(family="FIXED_COMMON_EXPOSURE", h=0.6031526490073492),
    )
    kwargs = {
        "phase": phase,
        "root_hex": root,
        "matrix_index": matrix_index,
        "definition": definition,
        "stream_identity": CANDIDATE_IDS[0],
        "beta": beta,
        "initial_state": initial,
    }
    first, first_seeds = simulate_trajectory(**kwargs)
    second, second_seeds = simulate_trajectory(**kwargs)
    comparison = compare_trajectories(first, second)
    seeds_equal, seed_differences = compare_seed_tuples(first_seeds, second_seeds)
    assert first.completed_fissions == 100
    assert first.trajectory_sha256 == second.trajectory_sha256
    assert comparison.repaired_comparator_passed
    assert comparison.discrete_divergence_count == 0
    assert comparison.finite_numeric_divergence_count == 0
    assert seeds_equal and not seed_differences
    assert np.array_equal(np.asarray(first.observations[0].state), initial)
