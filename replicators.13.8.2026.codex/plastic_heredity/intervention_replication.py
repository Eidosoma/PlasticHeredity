"""Registered clean-room causal-intervention replication for JOINT_BREAK_RUN3.

The module is deliberately additive.  It imports the sealed Codex simulator,
endpoint, feature map, and 5x frozen predictor but never modifies or refits
them.  The first executable scientific stage is the budgeted 40-matrix CR1
pilot; later mechanisms have independent pilot seed domains and stop points.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from threadpoolctl import threadpool_limits

from .config import CANDIDATES, CohortConfig, ExperimentConfig, GardConfig
from .experiment import StateCase, _json_ready, _runtime_manifest, build_cohort
from .features import history_features, state_graph_features
from .intervention_core import (
    BetaSurgery,
    FrozenFullPredictor,
    InterventionOutcome,
    MolecularEdit,
    ScoredEdit,
    apply_molecular_edit,
    catalytic_support,
    edited_snapshot,
    enumerate_legal_edits,
    outcome_from_records,
    random_beta_surgery,
    score_legal_edits,
    select_rule_edits,
    select_scored_edits,
    simulate_controlled,
    simulate_one_shot,
    state_graph_features_many,
    targeted_beta_surgery,
)
from .intervention_metrics import (
    compute_one_shot_inference,
    generate_inference_draws,
)
from .mechanistic import (
    _atomic_destination,
    _canonical_digest,
    sha256_file,
    verify_checksums,
    write_checksums,
)
from .models import predict_frozen_archive
from .processes import JOINT_BREAK_RUN3
from .seeds import derive_seed
from .simulator import (
    FissionRecord,
    Snapshot,
    _fission,
    generate_beta,
    simulate_future_absorbing,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = REPOSITORY_ROOT / "results_intervention_replication"
FROZEN_MODEL_SOURCE = REPOSITORY_ROOT / "results/scaled5/frozen_models.npz"
FROZEN_ARRAY_SOURCE = REPOSITORY_ROOT / "results/scaled5/analysis_arrays.npz"
FROZEN_MANIFEST_SOURCE = REPOSITORY_ROOT / "results/scaled5/manifest.json"
FROZEN_CHECKSUM_SOURCE = REPOSITORY_ROOT / "results/scaled5/SHA256SUMS"
FROZEN_STATE_TABLE_SOURCE = (
    REPOSITORY_ROOT / "results/scaled5/confirmation_states.csv"
)
EXPECTED_MODEL_SHA256 = (
    "9b3305a7fed11f432651926d34903443e9413ed299c5d0f1056a0b5fde9990af"
)

PROGRAM_FORMAT = "codex-joint-break-run3-intervention-program-v1"
REGISTRATION_FORMAT = "codex-intervention-pilot-registration-v1"
RESULT_FORMAT = "codex-intervention-one-shot-pilot-v1"
CHECKPOINT_FORMAT = "codex-intervention-checkpoint-v1"
LANDMARKS = (20, 35, 50, 65, 80)
PILOT_MATRICES = 40
PILOT_BRANCHES = 32
CONFIRMATION_MATRICES = 160
CONFIRMATION_BRANCHES = 32
HORIZON = 12
BOOTSTRAP_REPETITIONS = 4_096
RANDOMIZATION_REPETITIONS = 4_096
EQUIVALENCE_MARGIN = 0.025
RANDOM_RATIO_LIMIT = 0.25
MODEL_TOLERANCE = 1e-12
BATCH_FEATURE_TOLERANCE = 1e-11


def _seed_value(label: str) -> str:
    return hashlib.sha256(
        f"codex-clean-room-intervention-v1::{label}".encode("utf-8")
    ).hexdigest()


SEED_DOMAINS = {
    label: _seed_value(label)
    for label in (
        "validation",
        "smoke_cohort",
        "smoke_selection",
        "smoke_future",
        "p1_cohort",
        "p1_selection",
        "p1_future",
        "p1_bootstrap",
        "p1_randomization",
        "p2_cohort",
        "p2_selection",
        "p2_future",
        "p2_bootstrap",
        "p2_randomization",
        "p3_cohort",
        "p3_selection",
        "p3_future",
        "p3_bootstrap",
        "p3_randomization",
        "chosen_confirmation_cohort",
        "chosen_confirmation_selection",
        "chosen_confirmation_future",
        "chosen_confirmation_bootstrap",
        "chosen_confirmation_randomization",
        "replay",
    )
}

PHASE_ARMS = {
    "p1": ("MODEL_UP", "MODEL_DOWN", "RANDOM", "NOOP"),
    "p2": ("RULE_UP", "RULE_DOWN", "RANDOM", "NOOP"),
    "p3": ("LOOSEN", "TIGHTEN", "RANDOM_SURGERY", "NOOP"),
}
PHASE_CONTRAST = {
    "p1": ("MODEL_UP", "MODEL_DOWN"),
    "p2": ("RULE_UP", "RULE_DOWN"),
    "p3": ("LOOSEN", "TIGHTEN"),
}
PHASE_LABEL = {"p1": "INTP1", "p2": "INTP2", "p3": "INTP3"}

SOURCE_FILES = (
    "plastic_heredity/config.py",
    "plastic_heredity/experiment.py",
    "plastic_heredity/features.py",
    "plastic_heredity/intervention_core.py",
    "plastic_heredity/intervention_metrics.py",
    "plastic_heredity/intervention_replication.py",
    "plastic_heredity/mechanistic.py",
    "plastic_heredity/mechanistic_metrics.py",
    "plastic_heredity/models.py",
    "plastic_heredity/processes.py",
    "plastic_heredity/seeds.py",
    "plastic_heredity/simulator.py",
    "tests/test_intervention_replication.py",
    "pyproject.toml",
    "requirements-lock.txt",
)


@dataclass(frozen=True)
class PhaseSpec:
    phase: str
    role: str
    matrices: int
    branches: int
    cohort_seed: str
    selection_seed: str
    future_seed: str
    bootstrap_seed: str
    randomization_seed: str

    @property
    def arms(self) -> tuple[str, ...]:
        return PHASE_ARMS[self.phase]

    @property
    def contrast(self) -> tuple[str, str]:
        return PHASE_CONTRAST[self.phase]


@dataclass(frozen=True)
class PhaseBatch:
    state_id: str
    state_digest: str
    arm_names: tuple[str, ...]
    predictions: NDArray[np.float64]
    selected_edits: tuple[MolecularEdit | None, ...]
    surgeries: tuple[BetaSurgery | None, ...]
    scored_edits: tuple[ScoredEdit, ...]
    catalytic_support: NDArray[np.float64]
    outcomes: tuple[tuple[InterventionOutcome, ...], ...]


def pilot_spec(phase: str) -> PhaseSpec:
    if phase not in PHASE_ARMS:
        raise ValueError(f"unknown pilot phase {phase}")
    return PhaseSpec(
        phase=phase,
        role={
            "p1": "CR1 predictor-guided molecular pilot",
            "p2": "CR3 externally specified catalytic-support-rule pilot",
            "p3": "CR4 fixed-composition catalytic-network-surgery pilot",
        }[phase],
        matrices=PILOT_MATRICES,
        branches=PILOT_BRANCHES,
        cohort_seed=SEED_DOMAINS[f"{phase}_cohort"],
        selection_seed=SEED_DOMAINS[f"{phase}_selection"],
        future_seed=SEED_DOMAINS[f"{phase}_future"],
        bootstrap_seed=SEED_DOMAINS[f"{phase}_bootstrap"],
        randomization_seed=SEED_DOMAINS[f"{phase}_randomization"],
    )


def _source_hashes() -> dict[str, str]:
    return {name: sha256_file(REPOSITORY_ROOT / name) for name in SOURCE_FILES}


def _protocol() -> dict[str, Any]:
    value: dict[str, Any] = {
        "format": PROGRAM_FORMAT,
        "status": "frozen_before_any_scientific_intervention_matrix",
        "mission": (
            "independent clean-room causal intervention tests on the validated "
            "Codex JOINT_BREAK_RUN3 process"
        ),
        "strict_eight_program": {
            "status": "closed",
            "occurrence_result_preserved": True,
            "common_prospective_predictor_gate_passed": False,
            "additional_predictor_search_prohibited": True,
            "not_an_endpoint_in_this_program": True,
        },
        "clean_room": {
            "uses_codex_simulator_contracts_only": True,
            "uses_codex_frozen_predictor_only": True,
            "fable_code_models_matrices_states_seeds_edits_and_results_imported": False,
            "external_values_available_only_after_each_codex_result_is_sealed": True,
        },
        "simulator_contracts": {
            candidate: {
                "poisson_exposure": contract.poisson_exposure,
                "overshoot_rule": contract.overshoot_rule,
                "fission_rule": contract.fission_rule,
                "daughter_rule": contract.daughter_rule,
            }
            for candidate, contract in CANDIDATES.items()
        },
        "endpoint": {
            "name": JOINT_BREAK_RUN3,
            "horizon_fissions": HORIZON,
            "inheritance": "unrounded float64 parent-selected-daughter cosine H > 0.9",
            "break": "H <= 0.9",
            "positive": (
                "a break followed strictly later by three consecutive inherited "
                "fissions within F12"
            ),
            "uninterrupted_existing_run_qualifies": False,
            "positive_before_later_extinction_remains_positive": True,
            "extinction_before_certification": "negative",
        },
        "molecular_edit": {
            "definition": "one mass-preserving remove-i/add-j substitution",
            "requires_present_source": True,
            "requires_distinct_types": True,
            "history_held_fixed_instantaneously": True,
        },
        "frozen_predictor": {
            "source": "results/scaled5/frozen_models.npz",
            "sha256": EXPECTED_MODEL_SHA256,
            "family": "candidate-separated full 5x-development composite",
            "refitting_recalibration_or_threshold_change": False,
            "archived_prediction_tolerance": MODEL_TOLERANCE,
        },
        "randomness": {
            "purpose_keyed_domains": sorted(SEED_DOMAINS),
            "arm_identity_in_future_seed": False,
            "paired_description": "common random streams, not identical realized futures",
            "random_selection_independent_of_future_stream": True,
            "matrix_shared_across_candidates": True,
        },
        "inference": {
            "unit": "whole catalytic matrix",
            "states_arms_halves_and_landmarks_travel_with_matrix": True,
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "paired_sign_randomizations": RANDOMIZATION_REPETITIONS,
            "holm_family": "four candidate-by-fixed-branch-half cells",
            "random_noop_tost_margin": [-EQUIVALENCE_MARGIN, EQUIVALENCE_MARGIN],
            "random_effect_ratio_limit": RANDOM_RATIO_LIMIT,
            "prediction_log_loss": "ordinary branch-level Bernoulli log loss",
            "prediction_brier": "ordinary branch-level squared error",
            "complete_exact_replay": True,
        },
        "budgeted_serial_program": {
            "elapsed_runtime_goal_hours": 30,
            "hard_mid_phase_kill": False,
            "stopping_policy": (
                "finish, replay, seal, and report each stage; stop for user review "
                "before the next scientific stage"
            ),
            "p1": {
                "role": "CR1 predictor-guided pilot",
                "matrices": PILOT_MATRICES,
                "candidates": 2,
                "landmarks": list(LANDMARKS),
                "branches_per_arm": PILOT_BRANCHES,
                "arms": list(PHASE_ARMS["p1"]),
                "futures_per_pass": 2 * PILOT_MATRICES * len(LANDMARKS) * 4 * PILOT_BRANCHES,
                "full_replay": True,
                "mandatory_stop_after_seal": True,
            },
            "p2": {
                "role": "CR3 physical-rule pilot",
                "matrices": PILOT_MATRICES,
                "branches_per_arm": PILOT_BRANCHES,
                "arms": list(PHASE_ARMS["p2"]),
                "runs_only_after_separate_user_continue_instruction": True,
                "mandatory_stop_after_seal": True,
            },
            "p3": {
                "role": "CR4 beta-surgery pilot",
                "matrices": PILOT_MATRICES,
                "branches_per_arm": PILOT_BRANCHES,
                "arms": list(PHASE_ARMS["p3"]),
                "primary_delta_fraction": 0.05,
                "runs_only_after_separate_user_continue_instruction": True,
                "mandatory_stop_after_seal": True,
            },
            "chosen_mechanism_confirmation": {
                "role": "separately registered untouched confirmation",
                "matrices": CONFIRMATION_MATRICES,
                "branches_per_arm": CONFIRMATION_BRANCHES,
                "mechanism_chosen_only_after_all_three_pilots": True,
                "new_seed_domain": True,
                "pilot_outcomes_not_confirmation_data": True,
            },
            "conditional_compact_extensions": [
                "graded dose response",
                "closed-loop maintenance",
                "steer-release challenge",
            ],
            "deferred_from_this_budget": [
                "full resistance/resilience decomposition",
                "parameter-regime transfer",
                "control-half-life ladder",
                "internalization ladder",
            ],
        },
        "pilot_decision_boundary": {
            "registered_confirmatory_gates_still_reported": True,
            "pilot_eligibility_only": [
                "up-minus-down point estimate positive in all four cells",
                "random-minus-noop point estimate within +/-0.025 in all four cells",
                "exact replay",
            ],
            "pilot_pass_is_not_cross_clean_room_confirmation": True,
        },
        "phase_advancement": {
            "next_automatic_scientific_phase": None,
            "user_instruction_required_after_each_pilot": True,
            "failed_result_retained_at_full_prominence": True,
            "matrices_never_replaced": True,
            "future_branches_never_retried": True,
        },
        "claim_boundaries": {
            "may_eventually_support": (
                "small molecular or network interventions causally alter the "
                "Codex JOINT_BREAK_RUN3 probability under common random streams"
            ),
            "prohibited": [
                "Phi or PhiID intervention",
                "strict-eight control",
                "autonomous agency",
                "biological memory",
                "error correction",
                "a living organism",
                "a universal origin-of-life mechanism",
                "real prebiotic chemistry",
                "cross-clean-room replication before the relevant untouched gate passes",
            ],
        },
    }
    value["protocol_id"] = _canonical_digest(value)
    return value


def _experiment(spec: PhaseSpec) -> ExperimentConfig:
    cohort = CohortConfig(
        matrices=spec.matrices,
        branches_per_state=spec.branches,
        landmarks=LANDMARKS,
    )
    return ExperimentConfig(
        development=cohort,
        confirmation=cohort,
        horizon=HORIZON,
        bootstrap_repetitions=BOOTSTRAP_REPETITIONS,
        permutation_repetitions=RANDOMIZATION_REPETITIONS,
        regenerate_confirmation=True,
        master_seed=spec.cohort_seed,
    )


def _model_prediction_audit() -> tuple[dict[str, Any], dict[str, NDArray]]:
    if sha256_file(FROZEN_MODEL_SOURCE) != EXPECTED_MODEL_SHA256:
        raise ValueError("the frozen 5x predictor archive hash changed")
    verify_checksums(FROZEN_MODEL_SOURCE.parent)
    predictor = FrozenFullPredictor.load(FROZEN_MODEL_SOURCE)
    source_manifest = json.loads(
        FROZEN_MANIFEST_SOURCE.read_text(encoding="utf-8")
    )
    expected_experiment = json.loads(
        json.dumps(ExperimentConfig.scaled5().to_dict())
    )
    if source_manifest.get("experiment") != expected_experiment:
        raise ValueError("the archived 5x simulator/model contract changed")
    archived_states = pd.read_csv(FROZEN_STATE_TABLE_SOURCE)
    if archived_states.shape[0] != 2_000:
        raise ValueError("the archived confirmation state table changed")
    audit: dict[str, Any] = {
        "source_model_sha256": sha256_file(FROZEN_MODEL_SOURCE),
        "source_arrays_sha256": sha256_file(FROZEN_ARRAY_SOURCE),
        "source_manifest_sha256": sha256_file(FROZEN_MANIFEST_SOURCE),
        "source_state_table_sha256": sha256_file(FROZEN_STATE_TABLE_SOURCE),
        "archived_experiment_contract_exact": True,
        "candidate_errors": {},
        "tolerance": MODEL_TOLERANCE,
    }
    saved: dict[str, NDArray] = {}
    with np.load(FROZEN_ARRAY_SOURCE, allow_pickle=False) as arrays:
        state = arrays["confirmation_state_graph"]
        history = arrays["confirmation_history"]
        beta = arrays["confirmation_beta"]
        row = np.arange(state.shape[0]) % (2 * len(LANDMARKS))
        for candidate, mask in (
            ("02", row < len(LANDMARKS)),
            ("03", row >= len(LANDMARKS)),
        ):
            portable = predictor.predict_features(candidate, state[mask], history[mask])
            reference = predict_frozen_archive(
                FROZEN_MODEL_SOURCE,
                candidate,
                state[mask],
                history[mask],
                beta[mask],
            )["full"]
            error = float(np.max(np.abs(portable - reference)))
            archived = archived_states.loc[
                archived_states["candidate"].astype(str).str.zfill(2) == candidate,
                "prediction_full",
            ].to_numpy(dtype=np.float64)
            if archived.shape != portable.shape:
                raise ValueError(
                    f"archived prediction rows changed for candidate {candidate}"
                )
            archived_error = float(np.max(np.abs(portable - archived)))
            if error > MODEL_TOLERANCE:
                raise ValueError(
                    f"portable frozen predictor changed for candidate {candidate}: {error}"
                )
            if archived_error > MODEL_TOLERANCE:
                raise ValueError(
                    "portable predictor no longer reproduces archived confirmation "
                    f"predictions for candidate {candidate}: {archived_error}"
                )
            audit["candidate_errors"][candidate] = {
                "states": int(mask.sum()),
                "maximum_absolute_error_vs_archive_implementation": error,
                "maximum_absolute_error_vs_archived_confirmation_table": (
                    archived_error
                ),
                "within_tolerance": True,
            }
            saved[f"c{candidate}_portable"] = portable
            saved[f"c{candidate}_reference"] = reference
            saved[f"c{candidate}_archived_table"] = archived
    audit["all_within_tolerance"] = True
    return audit, saved


def _fixture_snapshot() -> Snapshot:
    return Snapshot(
        composition=np.asarray([1, 1, 0, 0], dtype=np.int64),
        generation=5,
        inheritance=(True, False, True, True, True),
        boundary_h=(0.95, 0.80, 0.92, 0.93, 0.94),
        previous_growth_steps=17,
        cumulative_growth_steps=81,
    )


def _fixture_gard() -> GardConfig:
    return GardConfig(
        n_types=4,
        n_min=2,
        n_max=4,
        beta_log_mean=0.0,
        beta_log_sd=0.1,
        k_join=0.05,
        k_leave=0.0,
        max_growth_steps=2_000,
        generations=20,
        inheritance_threshold=0.9,
    )


def _fixture_record(h: float, marker: int = 0) -> FissionRecord:
    parent = np.asarray([2, 1, marker, 1], dtype=np.int64)
    daughter = np.asarray([1, 1, marker, 0], dtype=np.int64)
    return FissionRecord(parent=parent, daughter=daughter, h=h, growth_steps=3)


def _records_bitwise_equal(
    left: tuple[FissionRecord, ...] | list[FissionRecord],
    right: tuple[FissionRecord, ...] | list[FissionRecord],
) -> bool:
    if len(left) != len(right):
        return False
    return all(
        np.array_equal(a.parent, b.parent)
        and np.array_equal(a.daughter, b.daughter)
        and np.asarray(a.h, dtype=np.float64).tobytes()
        == np.asarray(b.h, dtype=np.float64).tobytes()
        and a.growth_steps == b.growth_steps
        for a, b in zip(left, right, strict=True)
    )


def _future_seed(spec: PhaseSpec, case: StateCase, branch: int) -> int:
    """Return an arm-free seed shared by every arm of one paired branch."""

    return derive_seed(
        spec.future_seed,
        f"{PHASE_LABEL[spec.phase]}.future",
        case.candidate,
        case.matrix_id,
        case.landmark,
        branch,
    )


def _selection_seed(spec: PhaseSpec, case: StateCase, purpose: str) -> int:
    return derive_seed(
        spec.selection_seed,
        f"{PHASE_LABEL[spec.phase]}.selection.{purpose}",
        case.candidate,
        case.matrix_id,
        case.landmark,
    )


def validation_checks() -> dict[str, Any]:
    """Run the mandatory pre-scientific checks without generating a cohort."""

    checks: dict[str, dict[str, Any]] = {}

    def record(name: str, condition: bool, detail: Any = None) -> None:
        if not condition:
            raise AssertionError(f"validation failed: {name}: {detail}")
        checks[name] = {"passed": True, "detail": detail}

    snapshot = _fixture_snapshot()
    config = _fixture_gard()
    beta = np.asarray(
        [
            [4.0, 2.0, 1.0, 3.0],
            [1.5, 5.0, 2.5, 1.0],
            [2.0, 1.0, 6.0, 2.0],
            [3.0, 2.0, 1.0, 5.0],
        ],
        dtype=np.float64,
    )
    edit = MolecularEdit(0, 2)
    edited = apply_molecular_edit(snapshot.composition, edit)
    record(
        "01_legal_substitution_preserves_mass",
        int(edited.sum()) == int(snapshot.composition.sum()),
    )
    try:
        apply_molecular_edit(snapshot.composition, MolecularEdit(0, 0))
    except ValueError:
        same_rejected = True
    else:
        same_rejected = False
    record("02_same_type_substitution_rejected", same_rejected)
    try:
        apply_molecular_edit(snapshot.composition, MolecularEdit(3, 2))
    except ValueError:
        absent_rejected = True
    else:
        absent_rejected = False
    record("03_absent_source_rejected", absent_rejected)
    legal_results = [
        apply_molecular_edit(snapshot.composition, item)
        for item in enumerate_legal_edits(snapshot.composition)
    ]
    record(
        "04_legal_edits_nonnegative_integer",
        all(
            np.issubdtype(value.dtype, np.integer) and np.all(value >= 0)
            for value in legal_results
        ),
        {"legal_edits": len(legal_results)},
    )

    permutation = np.asarray([2, 0, 3, 1], dtype=np.int64)
    original_features = state_graph_features(edited, beta, config)
    permuted_features = state_graph_features(
        edited[permutation], beta[np.ix_(permutation, permutation)], config
    )
    permutation_error = float(np.max(np.abs(original_features - permuted_features)))
    record(
        "05_edited_features_permutation_invariant",
        permutation_error <= 1e-12,
        {"maximum_absolute_error": permutation_error},
    )
    new_snapshot = edited_snapshot(snapshot, edit)
    record(
        "06_history_unchanged_by_instantaneous_edit",
        new_snapshot.generation == snapshot.generation
        and new_snapshot.inheritance == snapshot.inheritance
        and new_snapshot.boundary_h == snapshot.boundary_h
        and new_snapshot.previous_growth_steps == snapshot.previous_growth_steps
        and new_snapshot.cumulative_growth_steps == snapshot.cumulative_growth_steps
        and np.array_equal(
            history_features(new_snapshot, config), history_features(snapshot, config)
        ),
    )

    model_audit, _ = _model_prediction_audit()
    predictor = FrozenFullPredictor.load(FROZEN_MODEL_SOURCE)
    with tempfile.TemporaryDirectory(prefix="codex-intervention-model-") as temporary:
        serialized = Path(temporary) / "predictor.npz"
        np.savez_compressed(serialized, **predictor.arrays)
        reloaded = FrozenFullPredictor.load(serialized)
        with np.load(FROZEN_ARRAY_SOURCE, allow_pickle=False) as arrays:
            left = predictor.predict_features(
                "02",
                arrays["confirmation_state_graph"][:5],
                arrays["confirmation_history"][:5],
            )
            right = reloaded.predict_features(
                "02",
                arrays["confirmation_state_graph"][:5],
                arrays["confirmation_history"][:5],
            )
        serialization_exact = np.array_equal(left, right)
    record(
        "07_frozen_predictor_serialization_exact",
        serialization_exact and model_audit["all_within_tolerance"],
        model_audit,
    )

    small = np.asarray([1, 0, 2], dtype=np.int64)
    observed_edits = enumerate_legal_edits(small)
    expected_edits = (
        MolecularEdit(0, 1),
        MolecularEdit(0, 2),
        MolecularEdit(2, 0),
        MolecularEdit(2, 1),
    )
    record("08_exhaustive_legal_enumeration_exact", observed_edits == expected_edits)
    tied = tuple(ScoredEdit(item, 0.5, 0.0) for item in expected_edits)
    selected_tie = select_scored_edits(0.5, tied, np.random.default_rng(7))
    record(
        "09_extreme_tie_selection_deterministic",
        selected_tie.model_up.edit == expected_edits[0]
        and selected_tie.model_down.edit == expected_edits[0],
    )
    random_rng = np.random.default_rng(11)
    counts = {item: 0 for item in expected_edits}
    for _ in range(20_000):
        choice = select_scored_edits(0.5, tied, random_rng).random.edit
        counts[choice] += 1
    expected_count = 20_000 / len(expected_edits)
    maximum_relative_deviation = max(
        abs(value - expected_count) / expected_count for value in counts.values()
    )
    record(
        "10_random_edit_selection_uniform",
        maximum_relative_deviation < 0.04,
        {"counts": [counts[item] for item in expected_edits]},
    )

    spec = pilot_spec("p1")
    dummy_case = StateCase("fixture", "FIX", "02", 3, 20, beta, snapshot)
    selection_seed = _selection_seed(spec, dummy_case, "random_edit")
    future_seed = _future_seed(spec, dummy_case, 0)
    record(
        "11_random_selection_stream_distinct_from_future_stream",
        selection_seed != future_seed,
    )
    arm_seeds = {
        arm: _future_seed(spec, dummy_case, 0) for arm in PHASE_ARMS["p1"]
    }
    record(
        "12_future_streams_paired_across_arms",
        len(set(arm_seeds.values())) == 1,
        {"arm_identity_in_seed_key": False},
    )

    simulation_beta = np.full((4, 4), 100.0, dtype=np.float64)
    seed = derive_seed(SEED_DOMAINS["validation"], "noop.future")
    direct_records, direct_completed = simulate_future_absorbing(
        snapshot,
        simulation_beta,
        config,
        CANDIDATES["02"],
        4,
        np.random.default_rng(seed),
    )
    noop_outcome = simulate_one_shot(
        snapshot,
        simulation_beta,
        "02",
        config,
        4,
        np.random.default_rng(seed),
        None,
    )
    direct_outcome = outcome_from_records(
        snapshot, direct_records, direct_completed, 4
    )
    record(
        "13_noop_one_shot_bitwise_plain_simulator",
        noop_outcome.record_digest == direct_outcome.record_digest
        and np.array_equal(noop_outcome.final_composition, direct_outcome.final_composition),
    )

    threshold_records = [
        _fixture_record(0.9, 0),
        _fixture_record(np.nextafter(0.9, 1.0), 1),
        _fixture_record(np.nextafter(0.9, 1.0), 2),
        _fixture_record(np.nextafter(0.9, 1.0), 3),
    ]
    threshold_outcome = outcome_from_records(
        snapshot, threshold_records, True, 4
    )
    uninterrupted = outcome_from_records(
        snapshot, [_fixture_record(0.95, index) for index in range(4)], True, 4
    )
    record(
        "14_endpoint_threshold_and_horizon_edges",
        threshold_outcome.joint_break_run3
        and threshold_outcome.renewal_certification_time == 4
        and not uninterrupted.joint_break_run3,
    )
    positive_extinct = outcome_from_records(
        snapshot, threshold_records, False, 5
    )
    record(
        "15_positive_before_extinction_remains_positive",
        positive_extinct.joint_break_run3 and not positive_extinct.completed_horizon,
    )
    negative_extinct = outcome_from_records(
        snapshot, threshold_records[:3], False, 5
    )
    record(
        "16_extinction_before_certification_negative",
        not negative_extinct.joint_break_run3,
    )

    class FixedFissionRng:
        def multivariate_hypergeometric(self, counts: NDArray, size: int) -> NDArray:
            return np.asarray([1, 1, 0, 0], dtype=np.int64)

        def binomial(self, counts: NDArray, probability: float) -> NDArray:
            return np.asarray([1, 0, 1, 0], dtype=np.int64)

    parent = np.asarray([2, 1, 1, 0], dtype=np.int64)
    fixed_rng = FixedFissionRng()
    daughter_02 = _fission(parent, config, CANDIDATES["02"], fixed_rng)  # type: ignore[arg-type]
    daughter_03 = _fission(parent, config, CANDIDATES["03"], fixed_rng)  # type: ignore[arg-type]
    record(
        "17_candidate_selected_daughter_semantics_preserved",
        np.array_equal(daughter_02, [1, 1, 0, 0])
        and np.array_equal(daughter_03, [1, 1, 0, 0]),
        {
            "candidate_02": "first fixed-size daughter",
            "candidate_03": "second binomial daughter",
        },
    )

    draw_rng = np.random.default_rng(19)
    draws = generate_inference_draws(4, 32, 32, draw_rng, np.random.default_rng(23))
    record(
        "18_matrix_bootstrap_uses_whole_blocks",
        draws["bootstrap_indices"].shape == (32, 4)
        and np.all(draws["bootstrap_indices"] < 4),
    )
    record(
        "19_matrix_sign_randomization_preserves_pairing",
        draws["randomization_signs"].shape == (32, 4)
        and np.isin(draws["randomization_signs"], (-1.0, 1.0)).all(),
    )

    replay_left = simulate_one_shot(
        snapshot,
        simulation_beta,
        "02",
        config,
        4,
        np.random.default_rng(seed),
        edit,
    )
    replay_right = simulate_one_shot(
        snapshot,
        simulation_beta,
        "02",
        config,
        4,
        np.random.default_rng(seed),
        edit,
    )
    record(
        "20_replay_state_edit_endpoint_and_process_exact",
        replay_left.record_digest == replay_right.record_digest
        and replay_left.joint_break_run3 == replay_right.joint_break_run3
        and replay_left.break_event == replay_right.break_event
        and replay_left.run3_after_break == replay_right.run3_after_break
        and replay_left.inherited_boundary_count
        == replay_right.inherited_boundary_count
        and replay_left.first_break_time == replay_right.first_break_time
        and replay_left.renewal_certification_time
        == replay_right.renewal_certification_time
        and replay_left.completed_horizon == replay_right.completed_horizon
        and np.array_equal(
            replay_left.final_composition, replay_right.final_composition
        )
        and np.array_equal(replay_left.boundary_h, replay_right.boundary_h)
        and np.array_equal(replay_left.growth_updates, replay_right.growth_updates),
    )

    tighten = targeted_beta_surgery(snapshot.composition, beta, 0.05, True)
    loosen = targeted_beta_surgery(snapshot.composition, beta, 0.05, False)
    norm_tolerance = 1e-12 * max(1.0, tighten.requested_norm)
    record(
        "21_beta_surgery_positive_and_norm_preserving",
        np.all(tighten.beta > 0.0)
        and np.all(loosen.beta > 0.0)
        and abs(tighten.observed_norm - tighten.requested_norm) <= norm_tolerance
        and abs(loosen.observed_norm - loosen.requested_norm) <= norm_tolerance,
    )
    random_surgery = random_beta_surgery(
        snapshot.composition, beta, 0.05, np.random.default_rng(29)
    )
    record(
        "22_random_beta_surgery_norm_matched",
        np.all(random_surgery.beta > 0.0)
        and abs(random_surgery.observed_norm - tighten.requested_norm)
        <= norm_tolerance,
    )

    experiment = ExperimentConfig(
        gard=config,
        development=CohortConfig(2, 2, (5,)),
        confirmation=CohortConfig(2, 2, (5,)),
        horizon=4,
        master_seed=SEED_DOMAINS["validation"],
    )
    controlled_seed = derive_seed(SEED_DOMAINS["validation"], "controlled.noop")
    controlled_plain = simulate_controlled(
        snapshot,
        simulation_beta,
        "02",
        experiment,
        4,
        np.random.default_rng(controlled_seed),
        None,
    )
    controlled_callback = simulate_controlled(
        snapshot,
        simulation_beta,
        "02",
        experiment,
        4,
        np.random.default_rng(controlled_seed),
        lambda *_: None,
    )
    record(
        "23_closed_loop_noop_callback_bitwise_plain",
        _records_bitwise_equal(controlled_plain.records, controlled_callback.records)
        and np.array_equal(
            controlled_plain.final_snapshot.composition,
            controlled_callback.final_snapshot.composition,
        ),
    )
    callback_steps: list[int] = []

    def releasing_controller(
        current: Snapshot, current_beta: NDArray, candidate: str, step: int
    ) -> MolecularEdit | None:
        callback_steps.append(step)
        return None

    released = simulate_controlled(
        snapshot,
        simulation_beta,
        "02",
        experiment,
        4,
        np.random.default_rng(controlled_seed),
        releasing_controller,
        release_after=2,
    )
    record(
        "24_release_applies_zero_interventions_after_release",
        callback_steps == [0, 1] and released.interventions_applied == 0,
    )

    actual_config = GardConfig()
    actual_rng = np.random.default_rng(
        derive_seed(SEED_DOMAINS["validation"], "batch.features")
    )
    actual_beta = generate_beta(actual_config, actual_rng)
    actual_composition = np.zeros(actual_config.n_types, dtype=np.int64)
    actual_composition[: actual_config.n_min] = 1
    actual_edits = enumerate_legal_edits(actual_composition)[:64]
    compositions = np.vstack(
        [apply_molecular_edit(actual_composition, item) for item in actual_edits]
    )
    batched = state_graph_features_many(compositions, actual_beta, actual_config)
    scalar = np.vstack(
        [
            state_graph_features(value, actual_beta, actual_config)
            for value in compositions
        ]
    )
    batch_error = float(np.max(np.abs(batched - scalar)))
    zero_history = np.zeros(9, dtype=np.float64)
    batch_probability = predictor.predict_features(
        "02",
        batched,
        np.broadcast_to(zero_history, (batched.shape[0], zero_history.size)),
    )
    scalar_probability = np.asarray(
        [
            predictor.predict_features("02", row, zero_history)[0]
            for row in scalar
        ],
        dtype=np.float64,
    )
    probability_error = float(
        np.max(np.abs(batch_probability - scalar_probability))
    )
    record(
        "25_batched_exhaustive_scoring_is_not_an_approximation",
        batch_error <= BATCH_FEATURE_TOLERANCE
        and probability_error <= MODEL_TOLERANCE,
        {
            "actual_dimension": actual_config.n_types,
            "edits_compared": len(actual_edits),
            "maximum_feature_absolute_error": batch_error,
            "maximum_probability_absolute_error": probability_error,
        },
    )
    support = catalytic_support(snapshot.composition, beta)
    record(
        "26_beta_orientation_matches_codex_propensity_equation",
        np.array_equal(support, beta @ snapshot.composition),
        {"storage": "beta[target,catalyst]", "expression": "beta @ x"},
    )
    return {
        "format": "codex-intervention-validation-v1",
        "checks": checks,
        "required_checks_passed": all(
            checks[f"{index:02d}_" + next(
                key.split("_", 1)[1]
                for key in checks
                if key.startswith(f"{index:02d}_")
            )]["passed"]
            for index in range(1, 25)
        ),
        "all_checks_passed": all(value["passed"] for value in checks.values()),
        "check_count": len(checks),
        "scientific_cohort_generated": False,
    }


def run_validation(output_directory: Path) -> None:
    output_directory = output_directory.resolve()
    if output_directory.exists():
        raise FileExistsError(f"refusing to overwrite {output_directory}")
    validation = validation_checks()
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_intervention_replication.py",
    ]
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "intervention pytest validation failed\n"
            + completed.stdout
            + "\n"
            + completed.stderr
        )
    with _atomic_destination(output_directory) as output:
        (output / "validation.json").write_text(
            json.dumps(_json_ready(validation), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output / "pytest_output.txt").write_text(
            "$ " + " ".join(command) + "\n" + completed.stdout + completed.stderr,
            encoding="utf-8",
        )
        (output / "manifest.json").write_text(
            json.dumps(
                {
                    "format": "codex-intervention-cr0-validation-v1",
                    "required_checks_passed": validation["required_checks_passed"],
                    "all_checks_passed": validation["all_checks_passed"],
                    "pytest_returncode": completed.returncode,
                    "scientific_cohort_generated": False,
                    "runtime": _runtime_manifest(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        write_checksums(output)
    verify_checksums(output_directory)
    print(f"CR0 validation sealed at {output_directory}", flush=True)


def register_program(validation_directory: Path, output_directory: Path) -> None:
    validation_directory = validation_directory.resolve()
    output_directory = output_directory.resolve()
    if output_directory.exists():
        raise FileExistsError(f"refusing to overwrite {output_directory}")
    verify_checksums(validation_directory)
    validation = json.loads(
        (validation_directory / "validation.json").read_text(encoding="utf-8")
    )
    if not validation.get("required_checks_passed") or not validation.get(
        "all_checks_passed"
    ):
        raise ValueError("mandatory intervention validation is incomplete")
    if len(SEED_DOMAINS) != len(set(SEED_DOMAINS.values())):
        raise ValueError("intervention seed-domain collision")
    model_audit, predictions = _model_prediction_audit()
    protocol = _protocol()
    with _atomic_destination(output_directory) as output:
        (output / "intervention_protocol.json").write_text(
            json.dumps(_json_ready(protocol), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output / "intervention_seed_registry.json").write_text(
            json.dumps(
                {
                    "format": "codex-intervention-seed-registry-v1",
                    "domains": SEED_DOMAINS,
                    "all_values_unique": True,
                    "future_keys_exclude_arm_identity": True,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        shutil.copy2(FROZEN_MODEL_SOURCE, output / "frozen_full_predictor.npz")
        np.savez_compressed(output / "archived_prediction_audit.npz", **predictions)
        (output / "frozen_model_audit.json").write_text(
            json.dumps(_json_ready(model_audit), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        shutil.copy2(validation_directory / "validation.json", output / "validation.json")
        shutil.copy2(
            validation_directory / "pytest_output.txt", output / "pytest_output.txt"
        )
        payload: dict[str, Any] = {
            "format": REGISTRATION_FORMAT,
            "status": "sealed_before_any_scientific_intervention_matrix",
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": sha256_file(output / "intervention_protocol.json"),
            "seed_registry_sha256": sha256_file(
                output / "intervention_seed_registry.json"
            ),
            "source_hashes": _source_hashes(),
            "frozen_model_source": {
                "path": str(FROZEN_MODEL_SOURCE),
                "sha256": EXPECTED_MODEL_SHA256,
                "copied_sha256": sha256_file(output / "frozen_full_predictor.npz"),
            },
            "source_scaled5_checksum_manifest_sha256": sha256_file(
                FROZEN_CHECKSUM_SOURCE
            ),
            "validation_source": {
                "path": str(validation_directory),
                "checksum_manifest_sha256": sha256_file(
                    validation_directory / "SHA256SUMS"
                ),
                "all_checks_passed": True,
            },
            "model_audit_sha256": sha256_file(output / "frozen_model_audit.json"),
            "prediction_audit_sha256": sha256_file(
                output / "archived_prediction_audit.npz"
            ),
            "no_scientific_intervention_matrix_generated": True,
        }
        payload["registration_id"] = _canonical_digest(payload)
        (output / "registration.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_checksums(output)
    verified = verify_registration(output_directory)
    print(
        f"Intervention program sealed: {verified['registration_id']}", flush=True
    )


def verify_registration(directory: Path) -> dict[str, Any]:
    directory = directory.resolve()
    verify_checksums(directory)
    payload = json.loads((directory / "registration.json").read_text(encoding="utf-8"))
    identifier = payload.pop("registration_id")
    if (
        payload.get("format") != REGISTRATION_FORMAT
        or payload.get("status")
        != "sealed_before_any_scientific_intervention_matrix"
        or _canonical_digest(payload) != identifier
    ):
        raise ValueError("invalid intervention registration")
    payload["registration_id"] = identifier
    if payload["source_hashes"] != _source_hashes():
        current = _source_hashes()
        changed = [
            name
            for name, digest in payload["source_hashes"].items()
            if current.get(name) != digest
        ]
        raise ValueError(f"registered intervention source changed: {changed}")
    protocol = json.loads(
        (directory / "intervention_protocol.json").read_text(encoding="utf-8")
    )
    if protocol != json.loads(json.dumps(_json_ready(_protocol()))):
        raise ValueError("intervention protocol implementation diverged")
    if (
        protocol["protocol_id"] != payload["protocol_id"]
        or sha256_file(directory / "intervention_protocol.json")
        != payload["protocol_sha256"]
    ):
        raise ValueError("intervention protocol digest changed")
    seeds = json.loads(
        (directory / "intervention_seed_registry.json").read_text(encoding="utf-8")
    )
    if seeds["domains"] != SEED_DOMAINS or not seeds["all_values_unique"]:
        raise ValueError("intervention seed registry changed")
    if (
        sha256_file(directory / "frozen_full_predictor.npz")
        != EXPECTED_MODEL_SHA256
        or sha256_file(FROZEN_MODEL_SOURCE) != EXPECTED_MODEL_SHA256
    ):
        raise ValueError("frozen predictor changed")
    model_audit, predictions = _model_prediction_audit()
    if sha256_file(directory / "frozen_model_audit.json") != payload[
        "model_audit_sha256"
    ]:
        raise ValueError("frozen model audit changed")
    with np.load(directory / "archived_prediction_audit.npz", allow_pickle=False) as saved:
        for name, value in predictions.items():
            if not np.array_equal(saved[name], value):
                raise ValueError(f"registered prediction audit changed: {name}")
    if not model_audit["all_within_tolerance"]:
        raise ValueError("portable model verification failed")
    validation = json.loads((directory / "validation.json").read_text(encoding="utf-8"))
    if not validation["required_checks_passed"] or not validation["all_checks_passed"]:
        raise ValueError("registered validation no longer passes")
    return payload


def _predict_edit_arms(
    predictor: FrozenFullPredictor,
    candidate: str,
    snapshot: Snapshot,
    beta: NDArray,
    config: GardConfig,
    edits: tuple[MolecularEdit | None, ...],
) -> NDArray[np.float64]:
    direct = history_features(snapshot, config)
    compositions = np.vstack(
        [
            snapshot.composition
            if edit is None
            else apply_molecular_edit(snapshot.composition, edit)
            for edit in edits
        ]
    )
    features = state_graph_features_many(compositions, beta, config)
    histories = np.broadcast_to(direct, (len(edits), direct.size))
    return predictor.predict_features(candidate, features, histories)


def _select_phase_arms(
    case: StateCase,
    experiment: ExperimentConfig,
    spec: PhaseSpec,
    predictor: FrozenFullPredictor,
) -> tuple[
    NDArray[np.float64],
    tuple[MolecularEdit | None, ...],
    tuple[BetaSurgery | None, ...],
    tuple[ScoredEdit, ...],
    NDArray[np.float64],
]:
    random_rng = np.random.default_rng(
        _selection_seed(spec, case, "random_arm")
    )
    if spec.phase == "p1":
        noop, scores = score_legal_edits(
            predictor,
            case.candidate,
            case.snapshot,
            case.beta,
            experiment.gard,
        )
        selected = select_scored_edits(noop, scores, random_rng)
        by_name = {
            "MODEL_UP": selected.model_up,
            "MODEL_DOWN": selected.model_down,
            "RANDOM": selected.random,
        }
        edits = tuple(
            None if arm == "NOOP" else by_name[arm].edit for arm in spec.arms
        )
        predictions = np.asarray(
            [
                noop if arm == "NOOP" else by_name[arm].predicted_probability
                for arm in spec.arms
            ],
            dtype=np.float64,
        )
        return (
            predictions,
            edits,
            tuple(None for _ in spec.arms),
            scores,
            np.empty(0, dtype=np.float64),
        )

    if spec.phase == "p2":
        rules = select_rule_edits(case.snapshot.composition, case.beta)
        legal = enumerate_legal_edits(case.snapshot.composition)
        random_edit = legal[int(random_rng.integers(0, len(legal)))]
        by_name = {
            "RULE_UP": rules["RULE_UP"],
            "RULE_DOWN": rules["RULE_DOWN"],
            "RANDOM": random_edit,
        }
        edits = tuple(
            None if arm == "NOOP" else by_name[arm] for arm in spec.arms
        )
        predictions = _predict_edit_arms(
            predictor,
            case.candidate,
            case.snapshot,
            case.beta,
            experiment.gard,
            edits,
        )
        return (
            predictions,
            edits,
            tuple(None for _ in spec.arms),
            tuple(),
            catalytic_support(case.snapshot.composition, case.beta),
        )

    if spec.phase == "p3":
        surgeries_by_name = {
            "LOOSEN": targeted_beta_surgery(
                case.snapshot.composition, case.beta, 0.05, False
            ),
            "TIGHTEN": targeted_beta_surgery(
                case.snapshot.composition, case.beta, 0.05, True
            ),
            "RANDOM_SURGERY": random_beta_surgery(
                case.snapshot.composition, case.beta, 0.05, random_rng
            ),
        }
        surgeries = tuple(
            None if arm == "NOOP" else surgeries_by_name[arm] for arm in spec.arms
        )
        predictions = np.asarray(
            [
                predictor.predict_snapshot(
                    case.candidate,
                    case.snapshot,
                    case.beta if surgery is None else surgery.beta,
                    experiment.gard,
                )
                for surgery in surgeries
            ],
            dtype=np.float64,
        )
        return (
            predictions,
            tuple(None for _ in spec.arms),
            surgeries,
            tuple(),
            np.empty(0, dtype=np.float64),
        )
    raise AssertionError(spec.phase)


def _phase_worker(
    arguments: tuple[StateCase, ExperimentConfig, PhaseSpec, str]
) -> PhaseBatch:
    case, experiment, spec, model_path = arguments
    limiter = threadpool_limits(limits=1)
    try:
        predictor = FrozenFullPredictor.load(model_path)
        predictions, edits, surgeries, scores, support = _select_phase_arms(
            case, experiment, spec, predictor
        )
        arm_outcomes: list[list[InterventionOutcome | None]] = [
            [None] * spec.branches for _ in spec.arms
        ]
        for branch in range(spec.branches):
            seed = _future_seed(spec, case, branch)
            for arm_index, _arm in enumerate(spec.arms):
                surgery = surgeries[arm_index]
                outcome = simulate_one_shot(
                    case.snapshot,
                    case.beta if surgery is None else surgery.beta,
                    case.candidate,
                    experiment.gard,
                    HORIZON,
                    np.random.default_rng(seed),
                    edits[arm_index],
                )
                arm_outcomes[arm_index][branch] = outcome
        outcomes = tuple(
            tuple(item for item in arm if item is not None) for arm in arm_outcomes
        )
        if any(len(arm) != spec.branches for arm in outcomes):
            raise AssertionError("future branch worker dropped an outcome")
        return PhaseBatch(
            state_id=case.state_id,
            state_digest=_snapshot_digest(case),
            arm_names=spec.arms,
            predictions=predictions,
            selected_edits=edits,
            surgeries=surgeries,
            scored_edits=scores,
            catalytic_support=support,
            outcomes=outcomes,
        )
    finally:
        limiter.restore_original_limits()


def _update_hash(digest: Any, value: NDArray) -> None:
    array = np.asarray(value)
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(array).tobytes())


def _edit_bytes(edit: MolecularEdit | None) -> bytes:
    values = (-1, -1) if edit is None else (edit.remove_type, edit.add_type)
    return np.asarray(values, dtype=np.int16).tobytes()


def _batch_digest(batch: PhaseBatch) -> str:
    digest = hashlib.sha256()
    digest.update(batch.state_id.encode("utf-8"))
    digest.update(batch.state_digest.encode("ascii"))
    digest.update("|".join(batch.arm_names).encode("utf-8"))
    _update_hash(digest, batch.predictions)
    for edit in batch.selected_edits:
        digest.update(_edit_bytes(edit))
    for surgery in batch.surgeries:
        if surgery is None:
            digest.update(b"NO_SURGERY")
            continue
        digest.update(surgery.name.encode("ascii"))
        _update_hash(digest, surgery.beta)
        _update_hash(digest, surgery.flat_indices)
        _update_hash(digest, surgery.before)
        _update_hash(digest, surgery.after)
        _update_hash(
            digest,
            np.asarray(
                (surgery.requested_norm, surgery.observed_norm), dtype=np.float64
            ),
        )
    for score in batch.scored_edits:
        digest.update(_edit_bytes(score.edit))
        _update_hash(
            digest,
            np.asarray(
                (score.predicted_probability, score.predicted_shift),
                dtype=np.float64,
            ),
        )
    _update_hash(digest, batch.catalytic_support)
    for arm in batch.outcomes:
        for outcome in arm:
            digest.update(outcome.record_digest.encode("ascii"))
            _update_hash(digest, outcome.final_composition)
            _update_hash(digest, outcome.boundary_h)
            _update_hash(digest, outcome.growth_updates)
            _update_hash(
                digest,
                np.asarray(
                    (
                        int(outcome.joint_break_run3),
                        int(outcome.break_event),
                        int(outcome.run3_after_break),
                        outcome.inherited_boundary_count,
                        outcome.first_break_time,
                        outcome.renewal_certification_time,
                        int(outcome.completed_horizon),
                        outcome.observed_fissions,
                        outcome.total_growth_updates,
                        outcome.mean_growth_updates,
                        outcome.final_entropy,
                        outcome.final_occupied_types,
                    ),
                    dtype=np.float64,
                ),
            )
    return digest.hexdigest()


def _snapshot_digest(case: StateCase) -> str:
    digest = hashlib.sha256()
    digest.update(case.state_id.encode("utf-8"))
    _update_hash(digest, case.beta)
    _update_hash(digest, case.snapshot.composition)
    _update_hash(digest, np.asarray(case.snapshot.boundary_h, dtype=np.float64))
    _update_hash(digest, np.asarray(case.snapshot.inheritance, dtype=np.int8))
    _update_hash(
        digest,
        np.asarray(
            (
                case.snapshot.generation,
                case.snapshot.previous_growth_steps,
                case.snapshot.cumulative_growth_steps,
            ),
            dtype=np.int64,
        ),
    )
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(_json_ready(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_pickle(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def _phase_checkpoint_contract(
    cases: list[StateCase], spec: PhaseSpec, registration_id: str, stage: str
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "registration_id": registration_id,
        "phase": spec.phase,
        "role": spec.role,
        "stage": stage,
        "matrices": spec.matrices,
        "branches": spec.branches,
        "horizon": HORIZON,
        "arms": list(spec.arms),
        "case_ids": [case.state_id for case in cases],
        "case_digests": [_snapshot_digest(case) for case in cases],
        "future_seed": spec.future_seed,
        "future_seed_includes_arm": False,
        "selection_seed": spec.selection_seed,
        "source_hashes": _source_hashes(),
    }
    value["contract_id"] = _canonical_digest(value)
    return value


def run_phase_batches(
    cases: list[StateCase],
    experiment: ExperimentConfig,
    spec: PhaseSpec,
    model_path: Path,
    registration_id: str,
    checkpoint_directory: Path,
    workers: int,
    stage: str,
) -> list[PhaseBatch]:
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    contract = _phase_checkpoint_contract(cases, spec, registration_id, stage)
    contract_path = checkpoint_directory / "checkpoint_contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text(encoding="utf-8")) != json.loads(
            json.dumps(_json_ready(contract))
        ):
            raise ValueError(f"checkpoint contract changed: {checkpoint_directory}")
    else:
        _atomic_json(contract_path, contract)

    batches: list[PhaseBatch | None] = [None] * len(cases)
    missing: list[int] = []
    for index, case in enumerate(cases):
        path = checkpoint_directory / f"state_{index:04d}.pkl"
        if path.is_file():
            with path.open("rb") as handle:
                batch = pickle.load(handle)
            if (
                not isinstance(batch, PhaseBatch)
                or batch.state_id != case.state_id
                or batch.state_digest != _snapshot_digest(case)
                or batch.arm_names != spec.arms
            ):
                raise ValueError(f"invalid checkpoint {path}")
            batches[index] = batch
        else:
            missing.append(index)

    def status(state: str) -> None:
        complete = sum(batch is not None for batch in batches)
        _atomic_json(
            checkpoint_directory / "status.json",
            {
                "format": CHECKPOINT_FORMAT,
                "phase": spec.phase,
                "stage": stage,
                "state": state,
                "states_complete": complete,
                "states_total": len(cases),
                "percent_complete": 100.0 * complete / len(cases),
                "futures_complete": complete * len(spec.arms) * spec.branches,
                "futures_total": len(cases) * len(spec.arms) * spec.branches,
                "checkpoint_directory": str(checkpoint_directory),
            },
        )

    status("running" if missing else "complete")
    arguments = [
        (cases[index], experiment, spec, str(model_path)) for index in missing
    ]
    if workers <= 1:
        generated = map(_phase_worker, arguments)
        for index, batch in zip(missing, generated, strict=True):
            batches[index] = batch
            _atomic_pickle(
                checkpoint_directory / f"state_{index:04d}.pkl", batch
            )
            status("running")
    elif missing:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            generated = executor.map(_phase_worker, arguments, chunksize=1)
            for index, batch in zip(missing, generated, strict=True):
                batches[index] = batch
                _atomic_pickle(
                    checkpoint_directory / f"state_{index:04d}.pkl", batch
                )
                status("running")
    status("complete")
    if any(batch is None for batch in batches):
        raise AssertionError("checkpointed phase has missing states")
    return [batch for batch in batches if batch is not None]


def replay_audit(
    generated: list[PhaseBatch], replayed: list[PhaseBatch]
) -> dict[str, Any]:
    if len(generated) != len(replayed):
        raise ValueError("generation and replay state counts differ")
    generated_digests = [_batch_digest(batch) for batch in generated]
    replay_digests = [_batch_digest(batch) for batch in replayed]
    prediction_error = max(
        float(np.max(np.abs(left.predictions - right.predictions)))
        for left, right in zip(generated, replayed, strict=True)
    )
    exact_by_state = [
        left == right
        for left, right in zip(generated_digests, replay_digests, strict=True)
    ]
    return {
        "format": "codex-intervention-replay-audit-v1",
        "states": len(generated),
        "state_edit_endpoint_and_process_digests_exact": bool(all(exact_by_state)),
        "states_exact": int(sum(exact_by_state)),
        "maximum_prediction_absolute_error": prediction_error,
        "future_seed_domain_reused_for_exact_replay": True,
        "replay_seed_domain_does_not_perturb_future_stream": True,
        "generated_campaign_digest": hashlib.sha256(
            "".join(generated_digests).encode("ascii")
        ).hexdigest(),
        "replay_campaign_digest": hashlib.sha256(
            "".join(replay_digests).encode("ascii")
        ).hexdigest(),
        "mismatched_state_ids": [
            generated[index].state_id
            for index, exact in enumerate(exact_by_state)
            if not exact
        ],
    }


def _outcome_arrays(
    cases: list[StateCase], batches: list[PhaseBatch], spec: PhaseSpec
) -> dict[str, NDArray]:
    shape = (len(cases), len(spec.arms), spec.branches)
    targets = np.empty(shape, dtype=np.int8)
    break_event = np.empty(shape, dtype=np.int8)
    run3 = np.empty(shape, dtype=np.int8)
    inherited_count = np.empty(shape, dtype=np.int8)
    first_break = np.empty(shape, dtype=np.int8)
    renewal_time = np.empty(shape, dtype=np.int8)
    completed = np.empty(shape, dtype=np.int8)
    observed = np.empty(shape, dtype=np.int8)
    total_growth = np.empty(shape, dtype=np.int32)
    mean_growth = np.empty(shape, dtype=np.float64)
    entropy = np.empty(shape, dtype=np.float64)
    occupied = np.empty(shape, dtype=np.int16)
    boundary_h = np.empty((*shape, HORIZON), dtype=np.float64)
    growth_updates = np.empty((*shape, HORIZON), dtype=np.int32)
    final_composition = np.empty(
        (*shape, cases[0].snapshot.composition.size), dtype=np.int16
    )
    predictions = np.vstack([batch.predictions for batch in batches])
    for state_index, batch in enumerate(batches):
        for arm_index, arm in enumerate(batch.outcomes):
            for branch, outcome in enumerate(arm):
                location = (state_index, arm_index, branch)
                targets[location] = int(outcome.joint_break_run3)
                break_event[location] = int(outcome.break_event)
                run3[location] = int(outcome.run3_after_break)
                inherited_count[location] = outcome.inherited_boundary_count
                first_break[location] = outcome.first_break_time
                renewal_time[location] = outcome.renewal_certification_time
                completed[location] = int(outcome.completed_horizon)
                observed[location] = outcome.observed_fissions
                total_growth[location] = outcome.total_growth_updates
                mean_growth[location] = outcome.mean_growth_updates
                entropy[location] = outcome.final_entropy
                occupied[location] = outcome.final_occupied_types
                boundary_h[location] = outcome.boundary_h
                growth_updates[location] = outcome.growth_updates
                final_composition[location] = outcome.final_composition
    return {
        "targets": targets,
        "break_event": break_event,
        "run3_after_break": run3,
        "inherited_boundary_count": inherited_count,
        "first_break_time": first_break,
        "renewal_certification_time": renewal_time,
        "completed_horizon": completed,
        "observed_fissions": observed,
        "total_growth_updates": total_growth,
        "mean_growth_updates": mean_growth,
        "final_entropy": entropy,
        "final_occupied_types": occupied,
        "boundary_h": boundary_h,
        "growth_updates": growth_updates,
        "final_composition": final_composition,
        "predictions": predictions,
    }


def _write_state_artifacts(
    output: Path,
    cases: list[StateCase],
    batches: list[PhaseBatch],
    arrays: dict[str, NDArray],
) -> None:
    max_history = max(len(case.snapshot.inheritance) for case in cases)
    inheritance = np.full((len(cases), max_history), -1, dtype=np.int8)
    boundary_h = np.full((len(cases), max_history), np.nan, dtype=np.float64)
    for index, case in enumerate(cases):
        length = len(case.snapshot.inheritance)
        inheritance[index, :length] = np.asarray(
            case.snapshot.inheritance, dtype=np.int8
        )
        boundary_h[index, :length] = np.asarray(
            case.snapshot.boundary_h, dtype=np.float64
        )
    matrix_ids = np.unique([case.matrix_id for case in cases])
    beta = []
    for matrix_id in matrix_ids:
        selected = [case.beta for case in cases if case.matrix_id == matrix_id]
        if not all(np.array_equal(selected[0], item) for item in selected[1:]):
            raise AssertionError("matrix beta differs across its candidates or landmarks")
        beta.append(selected[0])
    np.savez_compressed(
        output / "state_and_matrix_arrays.npz",
        state_ids=np.asarray([case.state_id for case in cases]),
        candidates=np.asarray([case.candidate for case in cases]),
        matrix_ids=np.asarray([case.matrix_id for case in cases], dtype=np.int16),
        landmarks=np.asarray([case.landmark for case in cases], dtype=np.int16),
        compositions=np.vstack([case.snapshot.composition for case in cases]).astype(
            np.int16
        ),
        generations=np.asarray(
            [case.snapshot.generation for case in cases], dtype=np.int16
        ),
        previous_growth_steps=np.asarray(
            [case.snapshot.previous_growth_steps for case in cases], dtype=np.int32
        ),
        cumulative_growth_steps=np.asarray(
            [case.snapshot.cumulative_growth_steps for case in cases], dtype=np.int64
        ),
        history_lengths=np.asarray(
            [len(case.snapshot.inheritance) for case in cases], dtype=np.int16
        ),
        inheritance=inheritance,
        boundary_h=boundary_h,
        beta_matrix_ids=matrix_ids.astype(np.int16),
        beta=np.stack(beta),
    )
    rows: list[dict[str, Any]] = []
    half = arrays["targets"].shape[2] // 2
    for state_index, (case, batch) in enumerate(
        zip(cases, batches, strict=True)
    ):
        row: dict[str, Any] = {
            "state_id": case.state_id,
            "candidate": case.candidate,
            "matrix_id": case.matrix_id,
            "landmark": case.landmark,
            "mass": int(case.snapshot.composition.sum()),
            "occupied_types": int(np.count_nonzero(case.snapshot.composition)),
            "previous_growth_steps": case.snapshot.previous_growth_steps,
            "cumulative_growth_steps": case.snapshot.cumulative_growth_steps,
        }
        for arm_index, arm in enumerate(batch.arm_names):
            row[f"prediction_{arm}"] = float(batch.predictions[arm_index])
            row[f"q_all_{arm}"] = float(
                arrays["targets"][state_index, arm_index].mean()
            )
            row[f"q_half_A_{arm}"] = float(
                arrays["targets"][state_index, arm_index, :half].mean()
            )
            row[f"q_half_B_{arm}"] = float(
                arrays["targets"][state_index, arm_index, half:].mean()
            )
        rows.append(row)
    pd.DataFrame(rows).to_csv(output / "state_probabilities.csv", index=False)


def _write_branch_table(
    path: Path,
    cases: list[StateCase],
    batches: list[PhaseBatch],
) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "state_id",
                "candidate",
                "matrix_id",
                "landmark",
                "arm",
                "branch",
                "branch_half",
                "joint_break_run3",
                "break_event",
                "run3_after_break",
                "inherited_boundary_count",
                "first_break_time",
                "renewal_certification_time",
                "completed_horizon",
                "observed_fissions",
                "total_growth_updates",
                "mean_growth_updates",
                "final_entropy",
                "final_occupied_types",
                "record_digest",
                "boundary_h_json",
                "growth_updates_json",
                "final_composition_json",
            )
        )
        for case, batch in zip(cases, batches, strict=True):
            for arm_index, arm in enumerate(batch.arm_names):
                for branch, outcome in enumerate(batch.outcomes[arm_index]):
                    writer.writerow(
                        (
                            case.state_id,
                            case.candidate,
                            case.matrix_id,
                            case.landmark,
                            arm,
                            branch,
                            "A" if branch < len(batch.outcomes[arm_index]) // 2 else "B",
                            int(outcome.joint_break_run3),
                            int(outcome.break_event),
                            int(outcome.run3_after_break),
                            outcome.inherited_boundary_count,
                            outcome.first_break_time,
                            outcome.renewal_certification_time,
                            int(outcome.completed_horizon),
                            outcome.observed_fissions,
                            outcome.total_growth_updates,
                            f"{outcome.mean_growth_updates:.17g}",
                            f"{outcome.final_entropy:.17g}",
                            outcome.final_occupied_types,
                            outcome.record_digest,
                            json.dumps(_json_ready(outcome.boundary_h.tolist())),
                            json.dumps(outcome.growth_updates.tolist()),
                            json.dumps(outcome.final_composition.tolist()),
                        )
                    )


def _write_selection_artifacts(
    output: Path, cases: list[StateCase], batches: list[PhaseBatch], spec: PhaseSpec
) -> None:
    selected_rows: list[dict[str, Any]] = []
    surgery_rows: list[dict[str, Any]] = []
    score_offsets = [0]
    score_remove: list[int] = []
    score_add: list[int] = []
    score_probability: list[float] = []
    score_shift: list[float] = []
    support = np.full(
        (len(cases), cases[0].snapshot.composition.size), np.nan, dtype=np.float64
    )
    for state_index, (case, batch) in enumerate(
        zip(cases, batches, strict=True)
    ):
        if batch.catalytic_support.size:
            support[state_index] = batch.catalytic_support
        for arm_index, arm in enumerate(batch.arm_names):
            edit = batch.selected_edits[arm_index]
            surgery = batch.surgeries[arm_index]
            selected_rows.append(
                {
                    "state_id": case.state_id,
                    "candidate": case.candidate,
                    "matrix_id": case.matrix_id,
                    "landmark": case.landmark,
                    "arm": arm,
                    "remove_type": -1 if edit is None else edit.remove_type,
                    "add_type": -1 if edit is None else edit.add_type,
                    "predicted_probability": float(batch.predictions[arm_index]),
                    "predicted_shift_from_noop": float(
                        batch.predictions[arm_index]
                        - batch.predictions[batch.arm_names.index("NOOP")]
                    ),
                    "is_noop": edit is None and surgery is None,
                    "is_beta_surgery": surgery is not None,
                }
            )
            if surgery is not None:
                for flat_index, before, after in zip(
                    surgery.flat_indices,
                    surgery.before,
                    surgery.after,
                    strict=True,
                ):
                    surgery_rows.append(
                        {
                            "state_id": case.state_id,
                            "arm": arm,
                            "flat_beta_index": int(flat_index),
                            "target_type": int(flat_index // case.beta.shape[1]),
                            "catalyst_type": int(flat_index % case.beta.shape[1]),
                            "before": float(before),
                            "after": float(after),
                            "requested_frobenius_norm": surgery.requested_norm,
                            "observed_frobenius_norm": surgery.observed_norm,
                        }
                    )
        for score in batch.scored_edits:
            score_remove.append(score.edit.remove_type)
            score_add.append(score.edit.add_type)
            score_probability.append(score.predicted_probability)
            score_shift.append(score.predicted_shift)
        score_offsets.append(len(score_remove))
    pd.DataFrame(selected_rows).to_csv(output / "selected_interventions.csv", index=False)
    pd.DataFrame(
        surgery_rows,
        columns=(
            "state_id",
            "arm",
            "flat_beta_index",
            "target_type",
            "catalyst_type",
            "before",
            "after",
            "requested_frobenius_norm",
            "observed_frobenius_norm",
        ),
    ).to_csv(output / "beta_surgery_edges.csv.gz", index=False, compression="gzip")
    np.savez_compressed(
        output / "selection_arrays.npz",
        state_ids=np.asarray([case.state_id for case in cases]),
        score_offsets=np.asarray(score_offsets, dtype=np.int64),
        score_remove=np.asarray(score_remove, dtype=np.int16),
        score_add=np.asarray(score_add, dtype=np.int16),
        score_probability=np.asarray(score_probability, dtype=np.float64),
        score_shift=np.asarray(score_shift, dtype=np.float64),
        catalytic_support=support,
    )


def _secondary_descriptives(
    cases: list[StateCase], arrays: dict[str, NDArray], spec: PhaseSpec
) -> dict[str, Any]:
    metrics = (
        "break_event",
        "run3_after_break",
        "inherited_boundary_count",
        "completed_horizon",
        "observed_fissions",
        "total_growth_updates",
        "mean_growth_updates",
        "final_entropy",
        "final_occupied_types",
    )
    rows: list[dict[str, Any]] = []
    branch_half = spec.branches // 2
    for candidate in CANDIDATES:
        selected = np.asarray(
            [case.candidate == candidate for case in cases], dtype=bool
        )
        for half, branch_slice in (
            ("A", slice(0, branch_half)),
            ("B", slice(branch_half, spec.branches)),
        ):
            for arm_index, arm in enumerate(spec.arms):
                row: dict[str, Any] = {
                    "candidate": candidate,
                    "branch_half": half,
                    "arm": arm,
                }
                for name in metrics:
                    row[f"mean_{name}"] = float(
                        np.nanmean(arrays[name][selected, arm_index, branch_slice])
                    )
                first = arrays["first_break_time"][selected, arm_index, branch_slice]
                renewal = arrays["renewal_certification_time"][
                    selected, arm_index, branch_slice
                ]
                row["mean_first_break_time_given_break"] = (
                    float(first[first >= 0].mean()) if np.any(first >= 0) else None
                )
                row["mean_renewal_time_given_certification"] = (
                    float(renewal[renewal >= 0].mean())
                    if np.any(renewal >= 0)
                    else None
                )
                rows.append(row)
    return {
        "registered_secondary_outcomes": list(metrics)
        + [
            "first_break_time_given_break",
            "renewal_time_given_certification",
        ],
        "cells": rows,
        "conditional_time_denominators_not_imputed": True,
    }


def _write_inference_arrays(
    path: Path, draws: dict[str, NDArray], metrics: dict[str, Any]
) -> None:
    stored = metrics.pop("stored_inference_arrays")
    arrays: dict[str, NDArray] = {
        "bootstrap_indices": np.asarray(draws["bootstrap_indices"], dtype=np.int64),
        "randomization_signs": np.asarray(
            draws["randomization_signs"], dtype=np.float64
        ),
    }
    for cell, contrasts in stored["cell_bootstrap_effects"].items():
        for contrast, values in contrasts.items():
            arrays[f"{cell}__bootstrap__{contrast}"] = np.asarray(
                values, dtype=np.float64
            )
    for cell, values in stored["cell_randomization_nulls"].items():
        arrays[f"{cell}__randomization_null"] = np.asarray(
            values, dtype=np.float64
        )
    np.savez_compressed(path, **arrays)
    metrics["stored_inference_arrays"] = {
        "path": path.name,
        "bootstrap_indices_shape": stored["bootstrap_indices_shape"],
        "randomization_signs_shape": stored["randomization_signs_shape"],
        "all_cell_bootstrap_and_randomization_arrays_stored": True,
    }


def _technical_report(
    phase: str,
    spec: PhaseSpec,
    metrics: dict[str, Any],
    replay: dict[str, Any],
    registration: dict[str, Any],
) -> str:
    title = {
        "p1": "P1 / CR1 predictor-guided molecular intervention pilot",
        "p2": "P2 / CR3 catalytic-support rule pilot",
        "p3": "P3 / CR4 beta-surgery pilot",
    }[phase]
    rows = []
    for cell in metrics["cells"]:
        effect = cell["contrasts"]["up_minus_down"]
        random_effect = cell["contrasts"]["random_minus_noop"]
        rows.append(
            "| {cell} | {effect:.6f} | [{low:.6f}, {high:.6f}] | "
            "{p:.6g} | {random:.6f} | {full} |".format(
                cell=cell["cell"],
                effect=effect["estimate"],
                low=effect["bootstrap_ci95"][0],
                high=effect["bootstrap_ci95"][1],
                p=cell["up_down_randomization_p_holm"],
                random=random_effect["estimate"],
                full=cell["registered_cell_pass"],
            )
        )
    pilot_eligible = bool(
        metrics["pilot_eligibility_without_replay"]
        and replay["state_edit_endpoint_and_process_digests_exact"]
    )
    return "\n".join(
        [
            f"# {title}",
            "",
            "## Outcome",
            "",
            f"Pilot eligibility for a later untouched confirmation: **{pilot_eligible}**.",
            f"The original full four-cell confirmatory gate passed: **{metrics['registered_all_four_cells_pass']}**.",
            f"Complete deterministic replay passed: **{replay['state_edit_endpoint_and_process_digests_exact']}**.",
            "",
            "A pilot eligibility result is developmental evidence only. It is not the separately registered 160-matrix confirmation and cannot establish cross-clean-room replication.",
            "",
            "## Primary cells",
            "",
            "| Cell | Up−down | 95% matrix-bootstrap CI | Holm p | Random−no-op | Full gate |",
            "|---|---:|---:|---:|---:|---:|",
            *rows,
            "",
            "The inference unit was the catalytic matrix. All landmarks, arms, fixed branch halves, and repeated states from a matrix remained together. Bootstrap and sign-randomization draws were shared across all four cells.",
            "",
            "## Design and audit",
            "",
            f"- Matrices: {spec.matrices} fresh matrices shared across candidates.",
            f"- Restored states: {2 * spec.matrices * len(LANDMARKS)}.",
            f"- Futures per pass: {2 * spec.matrices * len(LANDMARKS) * len(spec.arms) * spec.branches:,} F12 futures.",
            "- Every scientific future was replayed; futures were never retried.",
            "- Paired arms used common random streams whose seed key omitted arm identity.",
            f"- Registration ID: `{registration['registration_id']}`.",
            "",
            "## Claim boundary",
            "",
            "This pilot tests causal movement of the operational break-and-renewal probability under the registered intervention family. It does not test Phi/PhiID, strict-eight control, biological memory, autonomy, life, real chemistry, or a universal origin-of-life mechanism. A positive pilot does not by itself establish a common confirmed control law.",
            "",
            "## Mandatory stop",
            "",
            "This stage is sealed and the workflow stops here. No subsequent scientific pilot or confirmation is launched without a new user instruction.",
            "",
        ]
    )


def _lay_report(
    phase: str,
    metrics: dict[str, Any],
    replay: dict[str, Any],
) -> str:
    eligible = bool(
        metrics["pilot_eligibility_without_replay"]
        and replay["state_edit_endpoint_and_process_digests_exact"]
    )
    intervention = {
        "p1": "tiny one-molecule changes chosen by our already frozen risk predictor",
        "p2": "tiny one-molecule changes chosen by a simple catalytic-support rule",
        "p3": "small changes to the catalytic network while holding the molecules fixed",
    }[phase]
    verdict = (
        "The pilot moved in the required direction consistently enough to be eligible for an untouched confirmation."
        if eligible
        else "The pilot did not meet the prewritten eligibility rule for an untouched confirmation."
    )
    return "\n".join(
        [
            "# Lay summary",
            "",
            f"We tested whether {intervention} can change how often an assembly loses heredity and then rebuilds a short inherited run. The compared versions started from the same saved assembly and received matched streams of random numbers, so their average difference is attributable to the registered intervention rather than different luck at launch.",
            "",
            verdict,
            " Every future was run a second time and reproduced exactly. Because this is a 40-matrix pilot, it is a screening result—not the final causal replication. A later result can be called confirmed only after one chosen mechanism is frozen again and succeeds on 160 entirely new matrices.",
            "",
            "Even a positive result would mean only that this computer model has an externally controllable break-and-renewal process. It would not mean that the simulation is alive, remembers biologically, proves PhiID, or demonstrates real prebiotic chemistry.",
            "",
        ]
    )


def _campaign_status(work: Path, phase: str, state: str, detail: str) -> None:
    _atomic_json(
        work / "campaign_status.json",
        {
            "format": CHECKPOINT_FORMAT,
            "phase": phase,
            "state": state,
            "detail": detail,
            "mandatory_stop_after_seal": True,
        },
    )


def _prepare_campaign(
    work: Path,
    output: Path,
    registration: dict[str, Any],
    spec: PhaseSpec,
) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    work.mkdir(parents=True, exist_ok=True)
    contract: dict[str, Any] = {
        "format": "codex-intervention-campaign-contract-v1",
        "registration_id": registration["registration_id"],
        "phase": spec.phase,
        "role": spec.role,
        "output": str(output),
        "matrices": spec.matrices,
        "branches": spec.branches,
        "arms": list(spec.arms),
        "horizon": HORIZON,
        "source_hashes": _source_hashes(),
    }
    contract["campaign_id"] = _canonical_digest(contract)
    path = work / "campaign_contract.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != json.loads(
            json.dumps(_json_ready(contract))
        ):
            raise ValueError("work directory belongs to another campaign")
    else:
        _atomic_json(path, contract)
    _campaign_status(work, spec.phase, "running", "campaign_initialized")


def _readback_metrics(
    output: Path,
    cases: list[StateCase],
    spec: PhaseSpec,
    expected: dict[str, Any],
    expected_matrix_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    with np.load(output / "branch_arrays.npz", allow_pickle=False) as archive:
        targets = archive["targets"]
        predictions = archive["predictions"]
    with np.load(output / "inference_arrays.npz", allow_pickle=False) as archive:
        draws = {
            "bootstrap_indices": archive["bootstrap_indices"],
            "randomization_signs": archive["randomization_signs"],
        }
    up, down = spec.contrast
    observed, matrix_rows = compute_one_shot_inference(
        cases,
        spec.arms,
        targets,
        predictions,
        draws,
        up_arm=up,
        down_arm=down,
        equivalence_margin=EQUIVALENCE_MARGIN,
        random_ratio_limit=RANDOM_RATIO_LIMIT,
    )
    stored = observed.pop("stored_inference_arrays")
    observed["stored_inference_arrays"] = {
        "path": "inference_arrays.npz",
        "bootstrap_indices_shape": stored["bootstrap_indices_shape"],
        "randomization_signs_shape": stored["randomization_signs_shape"],
        "all_cell_bootstrap_and_randomization_arrays_stored": True,
    }
    metrics_exact = _json_ready(observed) == _json_ready(expected)
    matrix_effects_exact = _json_ready(matrix_rows) == _json_ready(
        expected_matrix_rows
    )
    if not metrics_exact or not matrix_effects_exact:
        raise ValueError("round-trip intervention inference changed")
    return {
        "branch_arrays_reloaded": True,
        "inference_draws_reloaded": True,
        "primary_metrics_exact": metrics_exact,
        "matrix_effects_exact": matrix_effects_exact,
        "no_fitting_or_recalibration": True,
    }


def _append_intervention_ledger(
    phase: str,
    output: Path,
    metrics: dict[str, Any],
    replay: dict[str, Any],
    registration_id: str,
) -> None:
    path = REPOSITORY_ROOT / "INTERVENTION_RESULTS_LEDGER.md"
    if path.exists():
        text = path.read_text(encoding="utf-8").rstrip() + "\n"
    else:
        text = "# Codex intervention results ledger\n\n"
        text += (
            "This ledger is separate from strict-eight occurrence and prediction. "
            "Its sole target is `JOINT_BREAK_RUN3`.\n"
        )
    marker = f"<!-- sealed-{phase}-{registration_id} -->"
    if marker in text:
        return
    pilot_eligible = bool(
        metrics["pilot_eligibility_without_replay"]
        and replay["state_edit_endpoint_and_process_digests_exact"]
    )
    lines = [
        "",
        marker,
        f"## {phase.upper()} sealed pilot",
        "",
        f"- Registration: `{registration_id}`",
        f"- Result bundle: `{output.relative_to(REPOSITORY_ROOT)}`",
        f"- Pilot eligibility: **{pilot_eligible}**",
        f"- Full four-cell gate: **{metrics['registered_all_four_cells_pass']}**",
        f"- Exact replay: **{replay['state_edit_endpoint_and_process_digests_exact']}**",
        "- Status: stopped after this stage; no next scientific phase was launched.",
        "- Boundary: pilot/developmental evidence only, not untouched cross-clean-room confirmation.",
        "",
        "| Cell | Up−down | 95% whole-matrix CI | Holm p | Random−no-op |",
        "|---|---:|---:|---:|---:|",
    ]
    for cell in metrics["cells"]:
        effect = cell["contrasts"]["up_minus_down"]
        random_effect = cell["contrasts"]["random_minus_noop"]["estimate"]
        lines.append(
            f"| {cell['cell']} | {effect['estimate']:.6f} | "
            f"[{effect['bootstrap_ci95'][0]:.6f}, {effect['bootstrap_ci95'][1]:.6f}] | "
            f"{cell['up_down_randomization_p_holm']:.6g} | {random_effect:.6f} |"
        )
    path.write_text(text + "\n".join(lines) + "\n", encoding="utf-8")


def run_pilot(
    phase: str,
    registration_directory: Path,
    output_directory: Path,
    workers: int,
    work_directory: Path | None = None,
) -> None:
    if workers < 1:
        raise ValueError("workers must be positive")
    registration_directory = registration_directory.resolve()
    output_directory = output_directory.resolve()
    registration = verify_registration(registration_directory)
    spec = pilot_spec(phase)
    experiment = _experiment(spec)
    work = (
        work_directory.resolve()
        if work_directory is not None
        else RESULT_ROOT / f".{phase}_work"
    )
    _prepare_campaign(work, output_directory, registration, spec)
    print(
        f"[{phase} 1/8] Building {spec.matrices} fresh matrices and "
        f"{2 * spec.matrices * len(LANDMARKS)} natural restored states",
        flush=True,
    )
    _campaign_status(work, phase, "running", "building_natural_trajectories")
    with threadpool_limits(limits=1):
        cases = build_cohort(
            experiment,
            PHASE_LABEL[phase],
            experiment.confirmation,
        )
    expected_states = 2 * spec.matrices * len(LANDMARKS)
    if len(cases) != expected_states:
        raise AssertionError("fresh intervention cohort has the wrong state count")

    model_path = registration_directory / "frozen_full_predictor.npz"
    futures = len(cases) * len(spec.arms) * spec.branches
    print(
        f"[{phase} 2/8] Selecting registered arms and shooting "
        f"{futures:,} F12 futures",
        flush=True,
    )
    _campaign_status(work, phase, "running", "selecting_and_shooting_futures")
    generated = run_phase_batches(
        cases,
        experiment,
        spec,
        model_path,
        registration["registration_id"],
        work / "generate",
        workers,
        "generate",
    )
    print(f"[{phase} 3/8] Replaying all {futures:,} F12 futures", flush=True)
    _campaign_status(work, phase, "running", "exact_replay")
    replayed = run_phase_batches(
        cases,
        experiment,
        spec,
        model_path,
        registration["registration_id"],
        work / "replay",
        workers,
        "replay",
    )
    replay = replay_audit(generated, replayed)
    arrays = _outcome_arrays(cases, generated, spec)
    draws = generate_inference_draws(
        spec.matrices,
        BOOTSTRAP_REPETITIONS,
        RANDOMIZATION_REPETITIONS,
        np.random.default_rng(
            derive_seed(spec.bootstrap_seed, f"{PHASE_LABEL[phase]}.bootstrap")
        ),
        np.random.default_rng(
            derive_seed(
                spec.randomization_seed, f"{PHASE_LABEL[phase]}.randomization"
            )
        ),
    )
    up, down = spec.contrast
    print(f"[{phase} 4/8] Computing frozen whole-matrix inference", flush=True)
    _campaign_status(work, phase, "running", "whole_matrix_inference")
    metrics, matrix_rows = compute_one_shot_inference(
        cases,
        spec.arms,
        arrays["targets"],
        arrays["predictions"],
        draws,
        up_arm=up,
        down_arm=down,
        equivalence_margin=EQUIVALENCE_MARGIN,
        random_ratio_limit=RANDOM_RATIO_LIMIT,
    )
    metrics["pilot_eligibility"] = bool(
        metrics["pilot_eligibility_without_replay"]
        and replay["state_edit_endpoint_and_process_digests_exact"]
    )
    secondary = _secondary_descriptives(cases, arrays, spec)

    print(f"[{phase} 5/8] Writing complete machine-readable artifacts", flush=True)
    with _atomic_destination(output_directory) as output:
        np.savez_compressed(output / "branch_arrays.npz", **arrays)
        _write_branch_table(output / "branches.csv.gz", cases, generated)
        _write_state_artifacts(output, cases, generated, arrays)
        _write_selection_artifacts(output, cases, generated, spec)
        _write_inference_arrays(output / "inference_arrays.npz", draws, metrics)
        pd.DataFrame(matrix_rows).to_csv(output / "matrix_effects.csv", index=False)
        (output / "primary_metrics.json").write_text(
            json.dumps(_json_ready(metrics), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output / "secondary_outcomes.json").write_text(
            json.dumps(_json_ready(secondary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output / "replay_audit.json").write_text(
            json.dumps(_json_ready(replay), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        readback = _readback_metrics(
            output, cases, spec, metrics, matrix_rows
        )
        (output / "readback_audit.json").write_text(
            json.dumps(readback, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        technical = _technical_report(
            phase, spec, metrics, replay, registration
        )
        lay = _lay_report(phase, metrics, replay)
        (output / "SCIENTIFIC_REPORT.md").write_text(technical, encoding="utf-8")
        (output / "LAY_SUMMARY.md").write_text(lay, encoding="utf-8")
        claim_boundary = {
            "supported_at_this_stage": (
                [
                    "pilot eligibility of the registered intervention family for a later untouched confirmation"
                ]
                if metrics["pilot_eligibility"]
                else []
            ),
            "failed_predictions": (
                []
                if metrics["pilot_eligibility"]
                else ["the registered pilot eligibility rule"]
            ),
            "deviations": [],
            "unresolved_questions": [
                "whether the effect passes a separately registered 160-matrix confirmation",
                "whether feedback can maintain the altered hereditary behavior",
                "whether any maintained organization persists autonomously after release",
            ],
            "prohibited_interpretations": _protocol()["claim_boundaries"][
                "prohibited"
            ],
        }
        (output / "claim_boundaries.json").write_text(
            json.dumps(claim_boundary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "format": RESULT_FORMAT,
            "phase": phase,
            "role": spec.role,
            "registration_id": registration["registration_id"],
            "matrices": spec.matrices,
            "candidates": list(CANDIDATES),
            "landmarks": list(LANDMARKS),
            "states": len(cases),
            "arms": list(spec.arms),
            "branches_per_arm_per_state": spec.branches,
            "primary_futures": futures,
            "replay_futures": futures,
            "pilot_eligibility": metrics["pilot_eligibility"],
            "full_registered_gate": metrics["registered_all_four_cells_pass"],
            "exact_replay": replay[
                "state_edit_endpoint_and_process_digests_exact"
            ],
            "no_future_retries": True,
            "no_matrix_replacement": True,
            "no_refitting_or_recalibration": True,
            "mandatory_stop_after_this_stage": True,
            "next_scientific_phase_launched": False,
            "runtime": _runtime_manifest(),
            "checkpoint_audit": {
                "work_directory": str(work),
                "campaign_contract_sha256": sha256_file(
                    work / "campaign_contract.json"
                ),
                "generate_contract_sha256": sha256_file(
                    work / "generate/checkpoint_contract.json"
                ),
                "replay_contract_sha256": sha256_file(
                    work / "replay/checkpoint_contract.json"
                ),
            },
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        ledger_snapshot = "\n".join(
            [
                "# Intervention result ledger snapshot",
                "",
                f"Phase: `{phase}`",
                f"Registration: `{registration['registration_id']}`",
                f"Pilot eligibility: **{metrics['pilot_eligibility']}**",
                f"Full registered gate: **{metrics['registered_all_four_cells_pass']}**",
                f"Exact replay: **{replay['state_edit_endpoint_and_process_digests_exact']}**",
                "Next phase: not launched; mandatory review stop.",
                "",
            ]
        )
        (output / "CUMULATIVE_RESULTS_LEDGER.md").write_text(
            ledger_snapshot, encoding="utf-8"
        )
        print(f"[{phase} 6/8] Sealing and checksum-verifying result", flush=True)
        write_checksums(output)
    verify_checksums(output_directory)
    _append_intervention_ledger(
        phase,
        output_directory,
        metrics,
        replay,
        registration["registration_id"],
    )
    _campaign_status(work, phase, "sealed_complete", "mandatory_review_stop")
    print(f"[{phase} 7/8] Result sealed: {output_directory}", flush=True)
    print(
        f"[{phase} 8/8] STOPPED as registered; no later scientific phase launched",
        flush=True,
    )


def run_smoke(
    registration_directory: Path, output_directory: Path, workers: int
) -> None:
    registration = verify_registration(registration_directory)
    output_directory = output_directory.resolve()
    if output_directory.exists():
        raise FileExistsError(f"refusing to overwrite {output_directory}")
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    spec = PhaseSpec(
        phase="p1",
        role="non-scientific I/O and replay smoke",
        matrices=1,
        branches=2,
        cohort_seed=SEED_DOMAINS["smoke_cohort"],
        selection_seed=SEED_DOMAINS["smoke_selection"],
        future_seed=SEED_DOMAINS["smoke_future"],
        bootstrap_seed=SEED_DOMAINS["validation"],
        randomization_seed=SEED_DOMAINS["replay"],
    )
    cohort = CohortConfig(1, 2, (20,))
    experiment = ExperimentConfig(
        development=cohort,
        confirmation=cohort,
        horizon=HORIZON,
        master_seed=spec.cohort_seed,
        bootstrap_repetitions=8,
        permutation_repetitions=8,
    )
    with tempfile.TemporaryDirectory(
        prefix="codex-intervention-smoke-", dir=output_directory.parent
    ) as temporary:
        temporary_path = Path(temporary)
        with threadpool_limits(limits=1):
            cases = build_cohort(experiment, "INTSMOKE", cohort)
        model_path = registration_directory.resolve() / "frozen_full_predictor.npz"
        generated = run_phase_batches(
            cases,
            experiment,
            spec,
            model_path,
            registration["registration_id"],
            temporary_path / "generate",
            workers,
            "generate",
        )
        replayed = run_phase_batches(
            cases,
            experiment,
            spec,
            model_path,
            registration["registration_id"],
            temporary_path / "replay",
            workers,
            "replay",
        )
        audit = replay_audit(generated, replayed)
        if not audit["state_edit_endpoint_and_process_digests_exact"]:
            raise AssertionError("non-scientific smoke replay failed")
    with _atomic_destination(output_directory) as output:
        (output / "manifest.json").write_text(
            json.dumps(
                {
                    "format": "codex-intervention-smoke-v1",
                    "registration_id": registration["registration_id"],
                    "scientific_result": False,
                    "scientific_matrix_count": 0,
                    "io_legality_checkpoint_and_replay_paths_passed": True,
                    "replay_exact": True,
                    "effect_sizes_disclosed": False,
                    "arm_ordering_disclosed": False,
                    "event_rates_disclosed": False,
                    "candidate_differences_disclosed": False,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        write_checksums(output)
    verify_checksums(output_directory)
    print(f"Non-scientific smoke passed: {output_directory}", flush=True)


def read_status(work_directory: Path) -> dict[str, Any]:
    work = work_directory.resolve()
    if not work.is_dir():
        raise FileNotFoundError(f"work directory does not exist: {work}")
    value: dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "work_directory": str(work),
        "campaign": None,
        "stages": {},
    }
    campaign = work / "campaign_status.json"
    if campaign.is_file():
        value["campaign"] = json.loads(campaign.read_text(encoding="utf-8"))
    for stage in ("generate", "replay"):
        status = work / stage / "status.json"
        if status.is_file():
            value["stages"][stage] = json.loads(status.read_text(encoding="utf-8"))
    if value["campaign"] is None and not value["stages"]:
        raise ValueError("work directory has no readable status")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clean-room JOINT_BREAK_RUN3 intervention replication"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument(
        "--output", type=Path, default=RESULT_ROOT / "cr0_validation"
    )
    register = commands.add_parser("register")
    register.add_argument(
        "--validation", type=Path, default=RESULT_ROOT / "cr0_validation"
    )
    register.add_argument(
        "--output", type=Path, default=RESULT_ROOT / "registration"
    )
    verify = commands.add_parser("verify")
    verify.add_argument(
        "--registration", type=Path, default=RESULT_ROOT / "registration"
    )
    smoke = commands.add_parser("smoke")
    smoke.add_argument(
        "--registration", type=Path, default=RESULT_ROOT / "registration"
    )
    smoke.add_argument(
        "--output", type=Path, default=RESULT_ROOT / "smoke"
    )
    smoke.add_argument("--workers", type=int, default=1)
    pilot = commands.add_parser("run-pilot")
    pilot.add_argument("--phase", choices=tuple(PHASE_ARMS), required=True)
    pilot.add_argument(
        "--registration", type=Path, default=RESULT_ROOT / "registration"
    )
    pilot.add_argument("--output", type=Path, default=None)
    pilot.add_argument("--work-dir", type=Path, default=None)
    pilot.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 14))
    status = commands.add_parser("status")
    status.add_argument("--work-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    if arguments.command == "validate":
        run_validation(arguments.output)
    elif arguments.command == "register":
        register_program(arguments.validation, arguments.output)
    elif arguments.command == "verify":
        payload = verify_registration(arguments.registration)
        print(
            json.dumps(
                {
                    "registration_id": payload["registration_id"],
                    "status": payload["status"],
                    "source_hashes_current": True,
                    "frozen_predictor_current": True,
                },
                indent=2,
                sort_keys=True,
            )
        )
    elif arguments.command == "smoke":
        run_smoke(arguments.registration, arguments.output, arguments.workers)
    elif arguments.command == "run-pilot":
        output = (
            arguments.output
            if arguments.output is not None
            else RESULT_ROOT
            / {
                "p1": "p1_cr1_model_guided_pilot",
                "p2": "p2_cr3_physical_rule_pilot",
                "p3": "p3_cr4_beta_surgery_pilot",
            }[arguments.phase]
        )
        run_pilot(
            arguments.phase,
            arguments.registration,
            output,
            arguments.workers,
            arguments.work_dir,
        )
    elif arguments.command == "status":
        print(json.dumps(read_status(arguments.work_dir), indent=2, sort_keys=True))
    else:  # pragma: no cover
        raise AssertionError(arguments.command)


if __name__ == "__main__":
    main()
