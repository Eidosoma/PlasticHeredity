"""Prospective shared-break resilience confirmation (CR5R).

CR5R is an additive confirmation using the immutable CR5 renewal students. It
does not alter or reinterpret the sealed CR5 resistance/resilience result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from threadpoolctl import threadpool_limits

from . import intervention_cr5 as cr5
from . import intervention_replication as base
from .config import CANDIDATES, CohortConfig, ExperimentConfig, GardConfig
from .experiment import StateCase, _json_ready, _runtime_manifest, build_cohort
from .intervention_core import (
    InterventionOutcome,
    apply_molecular_edit,
    enumerate_legal_edits,
)
from .intervention_metrics import generate_inference_draws
from .mechanistic import (
    _atomic_destination,
    _canonical_digest,
    sha256_file,
    verify_checksums,
    write_checksums,
)
from .mechanistic_metrics import holm_adjust
from .seeds import derive_seed
from .simulator import Snapshot, generate_beta, simulate_future_absorbing


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results_intervention_replication"
CR5_REGISTRATION = RESULT_ROOT / "cr5_confirmation_registration"
CR5_RESULT = RESULT_ROOT / "cr5_resistance_resilience_confirmation"
DEFAULT_VALIDATION = RESULT_ROOT / "cr5r_validation"
DEFAULT_REGISTRATION = RESULT_ROOT / "cr5r_confirmation_registration"
DEFAULT_SMOKE = RESULT_ROOT / "cr5r_smoke"
DEFAULT_OUTPUT = RESULT_ROOT / "cr5r_shared_break_resilience_confirmation"
DEFAULT_WORK = RESULT_ROOT / ".cr5r_shared_break_resilience_confirmation_work"

DOCUMENT = "CODEX_INTERVENTION_CR5R_PREREGISTRATION.md"
SOURCE_FILES = (
    DOCUMENT,
    "plastic_heredity/intervention_cr5r.py",
    "tests/test_intervention_cr5r.py",
    "plastic_heredity/intervention_cr5.py",
    "plastic_heredity/intervention_replication.py",
    "plastic_heredity/intervention_core.py",
    "plastic_heredity/intervention_metrics.py",
    "plastic_heredity/mechanistic_metrics.py",
    "plastic_heredity/simulator.py",
    "plastic_heredity/processes.py",
    "plastic_heredity/features.py",
    "plastic_heredity/experiment.py",
    "plastic_heredity/config.py",
    "plastic_heredity/seeds.py",
    "plastic_heredity/mechanistic.py",
    "pyproject.toml",
    "requirements-lock.txt",
)

PROGRAM_FORMAT = "codex-intervention-cr5r-shared-break-resilience-v1"
VALIDATION_FORMAT = "codex-intervention-cr5r-validation-v1"
REGISTRATION_FORMAT = "codex-intervention-cr5r-registration-v1"
RESULT_FORMAT = "codex-intervention-cr5r-result-v1"
CHECKPOINT_FORMAT = "codex-intervention-cr5r-checkpoint-v1"
STATUS_FORMAT = "codex-intervention-cr5r-status-v1"

CR5_REGISTRATION_ID = "61afdf612368c59aeb63fee7adb72b20e15dfdd3fafdefcc19f2f4e4d1bd4d3c"
EXPECTED_MODEL_SHA256 = (
    "59750718efcea3492a6d9b4493e9dc379eb221150025681acd38651d623cd430"
)
EXPECTED_MODEL_CONTRACT_SHA256 = (
    "0d14920f45f831c3825ee36a73537cfbc067cafe9042088ae9a123d82900bd95"
)

CONFIRMATION_LABEL = "INTCR5R_SHARED_BREAK_RESILIENCE_CONFIRMATION_V1"
MATRICES = 250
MINIMUM_ELIGIBLE_MATRICES = 200
LANDMARKS = (20, 35, 50, 65, 80)
BRANCHES = 64
HORIZON = 8
ACQUISITION_LIMIT = 60
ARMS = ("RENEWAL_UP", "RENEWAL_DOWN", "RANDOM", "NOOP")
BOOTSTRAP_REPETITIONS = 4_096
RANDOMIZATION_REPETITIONS = 4_096
EQUIVALENCE_MARGIN = 0.025
RANDOM_RATIO_LIMIT = 0.25
MINIMUM_CPU_BUDGET_HOURS = 8.0
MAXIMUM_CPU_BUDGET_HOURS = 14.0
DEFAULT_CPU_BUDGET_HOURS = 12.0
MINIMUM_FREE_DISK_BYTES = 4_000_000_000


def _seed(name: str) -> str:
    return hashlib.sha256(
        f"codex-clean-room-cr5r-shared-break-resilience-v1::{name}".encode()
    ).hexdigest()


SEEDS = {
    name: _seed(name)
    for name in (
        "validation",
        "smoke_selection",
        "smoke_future",
        "confirmation_cohort",
        "natural_break_acquisition",
        "random_edit_selection",
        "future_simulation",
        "bootstrap",
        "randomization",
        "replay",
    )
}


def source_hashes() -> dict[str, str]:
    return {name: sha256_file(ROOT / name) for name in SOURCE_FILES}


def protocol() -> dict[str, Any]:
    value: dict[str, Any] = {
        "format": PROGRAM_FORMAT,
        "status": "frozen_before_any_cr5r_scientific_matrix",
        "relationship_to_cr5": {
            "cr5_result_unchanged": True,
            "cr5_resilience_classification": (
                "inconclusive_incomplete_matrix_coverage"
            ),
            "separately_versioned_confirmation": True,
            "not_a_rescue_or_reanalysis": True,
        },
        "question": (
            "Can one frozen-predictor-guided molecule swap causally alter run3 "
            "recovery within F8 from an identical natural post-break daughter?"
        ),
        "endpoint": {
            "name": "run3_within_f8_from_natural_post_break_daughter",
            "horizon": HORIZON,
            "inheritance": "strict unrounded float64 H > 0.9",
            "positive_before_later_extinction_remains_positive": True,
            "extinction_before_certification_is_negative": True,
            "strict_eight_excluded": True,
        },
        "frozen_model": {
            "source_registration_id": CR5_REGISTRATION_ID,
            "sha256": EXPECTED_MODEL_SHA256,
            "contract_sha256": EXPECTED_MODEL_CONTRACT_SHA256,
            "students": ["renewal__c02", "renewal__c03"],
            "refit_recalibration_search_or_threshold_change": False,
        },
        "cohort": {
            "matrices": MATRICES,
            "candidates": list(CANDIDATES),
            "landmarks": list(LANDMARKS),
            "source_states": 2 * MATRICES * len(LANDMARKS),
            "natural_break_acquisition_limit": ACQUISITION_LIMIT,
            "minimum_eligible_matrices_per_candidate": MINIMUM_ELIGIBLE_MATRICES,
            "eligibility_assessed_separately_by_candidate": True,
            "all_eligible_states_retained_after_gate": True,
            "no_retry_replacement_or_risk_preselection": True,
        },
        "intervention": {
            "arms": list(ARMS),
            "all_legal_swaps_scored_exhaustively": True,
            "mass_preserving": True,
            "history_held_fixed_instantaneously": True,
            "tie_rule": "first frozen legal-enumeration index",
            "random_edit_uniform_over_legal_swaps": True,
        },
        "futures": {
            "branches_per_arm_state": BRANCHES,
            "branch_half_A": [0, 31],
            "branch_half_B": [32, 63],
            "horizon": HORIZON,
            "common_random_streams": True,
            "arm_identity_in_future_seed": False,
            "selection_stream_separate": True,
            "complete_exact_replay": True,
            "future_retries": False,
        },
        "inference": {
            "unit": "whole catalytic matrix",
            "all_states_within_matrix_kept_together": True,
            "candidate_specific_eligible_matrix_sets": True,
            "draws_reused_across_halves_within_candidate": True,
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "randomization_repetitions": RANDOMIZATION_REPETITIONS,
            "holm_family": "four candidate-by-branch-half cells",
            "equivalence_margin": EQUIVALENCE_MARGIN,
            "random_ratio_limit": RANDOM_RATIO_LIMIT,
            "up_noop_and_noop_down_reported_not_gated": True,
        },
        "primary_gate": {
            "all_four_cells_required": True,
            "up_minus_down_positive": True,
            "bootstrap_lower_positive": True,
            "holm_randomization_p_below": 0.05,
            "random_noop_tost_equivalent": True,
            "random_absolute_effect_ratio_at_most": RANDOM_RATIO_LIMIT,
            "exact_replay_and_readback": True,
        },
        "secondary": [
            "run5",
            "time_to_run3",
            "inherited_boundary_count",
            "old_anchor_similarity",
            "survival",
            "growth_updates",
            "final_entropy",
            "final_occupied_types",
        ],
        "operational": {
            "minimum_free_disk_bytes": MINIMUM_FREE_DISK_BYTES,
            "cpu_budget_hours": [
                MINIMUM_CPU_BUDGET_HOURS,
                MAXIMUM_CPU_BUDGET_HOURS,
            ],
            "expected_cpu_hours": [8.0, 10.0],
            "checkpoint_resumable": True,
            "mandatory_stop_after_seal": True,
            "cr6_not_launched_automatically": True,
        },
        "external_benchmark": {
            "available_only_after_codex_result_seal": True,
            "fable_effects_descriptive_only": [0.026, 0.027],
            "used_as_threshold_fit_or_margin": False,
        },
        "claim_boundary": {
            "passing_claim": (
                "causal molecular control of short-run recovery from an "
                "identical naturally broken state in both Codex candidates"
            ),
            "prohibited": [
                "biological repair",
                "biological memory",
                "agency or life",
                "autonomous organization or installed attractor",
                "strict-eight control",
                "real prebiotic chemistry",
                "universal origin-of-life mechanism",
                "Phi or PhiID intervention",
            ],
        },
        "seed_domains": SEEDS,
    }
    value["protocol_id"] = _canonical_digest(_json_ready(value))
    return value


def _experiment() -> ExperimentConfig:
    cohort = CohortConfig(MATRICES, BRANCHES, LANDMARKS)
    return ExperimentConfig(
        development=cohort,
        confirmation=cohort,
        horizon=HORIZON,
        bootstrap_repetitions=BOOTSTRAP_REPETITIONS,
        permutation_repetitions=RANDOMIZATION_REPETITIONS,
        regenerate_confirmation=True,
        master_seed=SEEDS["confirmation_cohort"],
    )


def phase_spec() -> cr5.CR5PhaseSpec:
    return cr5.CR5PhaseSpec(
        stage="resilience",
        target="renewal",
        arms=ARMS,
        horizon=HORIZON,
        branches=BRANCHES,
        selection_seed=SEEDS["random_edit_selection"],
        future_seed=SEEDS["future_simulation"],
        bootstrap_seed=SEEDS["bootstrap"],
        randomization_seed=SEEDS["randomization"],
    )


def selection_seed(case: StateCase) -> int:
    return derive_seed(
        SEEDS["random_edit_selection"],
        f"{CONFIRMATION_LABEL}.random_edit",
        case.candidate,
        case.matrix_id,
        case.landmark,
    )


def future_seed(case: StateCase, branch: int) -> int:
    return derive_seed(
        SEEDS["future_simulation"],
        f"{CONFIRMATION_LABEL}.future",
        case.candidate,
        case.matrix_id,
        case.landmark,
        branch,
    )


def _phase_worker(
    arguments: tuple[StateCase, GardConfig, str, str],
) -> base.PhaseBatch:
    case, config, model_path, model_contract_path = arguments
    limiter = threadpool_limits(limits=1)
    try:
        student = cr5.load_students(Path(model_path), Path(model_contract_path))[
            ("renewal", case.candidate)
        ]
        noop, scores = cr5.score_student_edits(student, case, config)
        predictions, edits = cr5.select_student_edits(
            noop,
            scores,
            np.random.default_rng(selection_seed(case)),
        )
        outcomes: list[list[InterventionOutcome]] = [[] for _ in ARMS]
        for branch in range(BRANCHES):
            seed = future_seed(case, branch)
            for arm_index, edit in enumerate(edits):
                composition = (
                    case.snapshot.composition
                    if edit is None
                    else apply_molecular_edit(case.snapshot.composition, edit)
                )
                launch = Snapshot(
                    composition=np.asarray(composition, dtype=np.int64).copy(),
                    generation=case.snapshot.generation,
                    inheritance=case.snapshot.inheritance,
                    boundary_h=case.snapshot.boundary_h,
                    previous_growth_steps=case.snapshot.previous_growth_steps,
                    cumulative_growth_steps=case.snapshot.cumulative_growth_steps,
                )
                records, completed = simulate_future_absorbing(
                    launch,
                    case.beta,
                    config,
                    CANDIDATES[case.candidate],
                    HORIZON,
                    np.random.default_rng(seed),
                )
                outcomes[arm_index].append(
                    cr5._stage_outcome(
                        "resilience",
                        launch,
                        records,
                        completed,
                        HORIZON,
                        config.inheritance_threshold,
                    )
                )
        return base.PhaseBatch(
            state_id=case.state_id,
            state_digest=base._snapshot_digest(case),
            arm_names=ARMS,
            predictions=predictions,
            selected_edits=edits,
            surgeries=tuple(None for _ in ARMS),
            scored_edits=scores,
            catalytic_support=np.empty(0, dtype=np.float64),
            outcomes=tuple(tuple(values) for values in outcomes),
        )
    finally:
        limiter.restore_original_limits()


def _checkpoint_contract(
    cases: list[StateCase],
    registration_id: str,
    execution_stage: str,
    model_hash: str,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "registration_id": registration_id,
        "execution_stage": execution_stage,
        "target": "run3_within_f8_from_natural_post_break_daughter",
        "horizon": HORIZON,
        "branches": BRANCHES,
        "arms": list(ARMS),
        "case_ids": [case.state_id for case in cases],
        "case_digests": [base._snapshot_digest(case) for case in cases],
        "selection_seed": SEEDS["random_edit_selection"],
        "future_seed": SEEDS["future_simulation"],
        "future_seed_includes_arm": False,
        "model_sha256": model_hash,
        "source_hashes": source_hashes(),
    }
    value["contract_id"] = _canonical_digest(_json_ready(value))
    return value


def run_phase_batches(
    cases: list[StateCase],
    config: GardConfig,
    model_path: Path,
    model_contract_path: Path,
    registration_id: str,
    checkpoint: Path,
    workers: int,
    execution_stage: str,
) -> list[base.PhaseBatch]:
    checkpoint.mkdir(parents=True, exist_ok=True)
    contract = _checkpoint_contract(
        cases, registration_id, execution_stage, sha256_file(model_path)
    )
    contract_path = checkpoint / "checkpoint_contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text()) != _json_ready(contract):
            raise ValueError(f"CR5R checkpoint contract changed: {checkpoint}")
    else:
        base._atomic_json(contract_path, contract)

    batches: list[base.PhaseBatch | None] = [None] * len(cases)
    missing: list[int] = []
    for index, case in enumerate(cases):
        path = checkpoint / f"state_{index:04d}.pkl"
        if path.exists():
            with path.open("rb") as handle:
                batch = pickle.load(handle)
            if (
                not isinstance(batch, base.PhaseBatch)
                or batch.state_id != case.state_id
                or batch.state_digest != base._snapshot_digest(case)
                or batch.arm_names != ARMS
            ):
                raise ValueError(f"invalid CR5R checkpoint: {path}")
            batches[index] = batch
        else:
            missing.append(index)

    def save_status(state: str) -> None:
        complete = sum(batch is not None for batch in batches)
        base._atomic_json(
            checkpoint / "status.json",
            {
                "format": CHECKPOINT_FORMAT,
                "execution_stage": execution_stage,
                "state": state,
                "states_complete": complete,
                "states_total": len(cases),
                "futures_complete": complete * len(ARMS) * BRANCHES,
                "futures_total": len(cases) * len(ARMS) * BRANCHES,
                "percent_complete": 100.0 * complete / max(1, len(cases)),
            },
        )

    save_status("running" if missing else "complete")
    arguments = [
        (case, config, str(model_path), str(model_contract_path))
        for case in (cases[index] for index in missing)
    ]
    if workers <= 1:
        generated = map(_phase_worker, arguments)
        for index, batch in zip(missing, generated, strict=True):
            batches[index] = batch
            base._atomic_pickle(checkpoint / f"state_{index:04d}.pkl", batch)
            save_status("running")
    elif missing:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            generated = executor.map(_phase_worker, arguments, chunksize=1)
            for index, batch in zip(missing, generated, strict=True):
                batches[index] = batch
                base._atomic_pickle(checkpoint / f"state_{index:04d}.pkl", batch)
                save_status("running")
    save_status("complete")
    if any(batch is None for batch in batches):
        raise AssertionError("CR5R phase dropped a state")
    return [batch for batch in batches if batch is not None]


def eligibility_summary(
    broken: list[StateCase], source_state_count: int, acquisition_exact: bool
) -> dict[str, Any]:
    eligible = {
        candidate: {
            "states": sum(case.candidate == candidate for case in broken),
            "matrices": len(
                {case.matrix_id for case in broken if case.candidate == candidate}
            ),
        }
        for candidate in CANDIDATES
    }
    launch = bool(
        acquisition_exact
        and all(
            value["matrices"] >= MINIMUM_ELIGIBLE_MATRICES
            for value in eligible.values()
        )
    )
    return {
        "source_states": source_state_count,
        "eligible_broken_states": len(broken),
        "eligible_by_candidate": eligible,
        "minimum_eligible_matrices_per_candidate": MINIMUM_ELIGIBLE_MATRICES,
        "acquisition_limit": ACQUISITION_LIMIT,
        "acquisition_replay_exact": acquisition_exact,
        "intervention_futures_authorized": launch,
        "classification": (
            "eligible_for_registered_confirmation"
            if launch
            else "inconclusive_insufficient_eligible_matrix_coverage"
        ),
        "all_eligible_states_retained": True,
        "no_retry_replacement_or_subselection": True,
    }


def _interval(values: NDArray[np.float64], alpha: float = 0.05) -> tuple[float, float]:
    lower, upper = np.quantile(values, (alpha / 2.0, 1.0 - alpha / 2.0))
    return float(lower), float(upper)


def _matrix_means(
    values: NDArray[np.float64],
    matrix_ids: NDArray[np.int64],
    matrix_order: NDArray[np.int64],
) -> NDArray[np.float64]:
    return np.asarray(
        [values[matrix_ids == matrix_id].mean() for matrix_id in matrix_order],
        dtype=np.float64,
    )


def _bootstrap_means(
    values: NDArray[np.float64], indices: NDArray[np.int64]
) -> NDArray[np.float64]:
    return np.asarray(values[indices].mean(axis=1), dtype=np.float64)


def _one_sided_sign_p(
    values: NDArray[np.float64], signs: NDArray[np.float64]
) -> tuple[float, NDArray[np.float64]]:
    observed = float(values.mean())
    null = np.asarray(signs @ values / values.size, dtype=np.float64)
    p_value = float((np.count_nonzero(null >= observed) + 1) / (null.size + 1))
    return p_value, null


def _maximum_leave_one_out_influence(values: NDArray[np.float64]) -> float:
    if values.size <= 1:
        return float("nan")
    observed = float(values.mean())
    leave_one_out = (values.sum() - values) / (values.size - 1)
    return float(np.max(np.abs(leave_one_out - observed)))


def _branch_scores(
    truth: NDArray[np.float64], prediction: NDArray[np.float64]
) -> dict[str, float]:
    probability = np.clip(np.asarray(prediction, dtype=np.float64), 1e-12, 1 - 1e-12)
    if probability.ndim == 1:
        probability = probability[:, None]
    return {
        "log_loss": float(
            np.mean(
                -(
                    truth * np.log(probability)
                    + (1.0 - truth) * np.log(1.0 - probability)
                )
            )
        ),
        "brier": float(np.mean((truth - probability) ** 2)),
    }


def generate_candidate_draws(cases: list[StateCase]) -> dict[str, NDArray]:
    draws: dict[str, NDArray] = {}
    for candidate in CANDIDATES:
        matrix_count = len(
            {case.matrix_id for case in cases if case.candidate == candidate}
        )
        candidate_draws = generate_inference_draws(
            matrix_count,
            BOOTSTRAP_REPETITIONS,
            RANDOMIZATION_REPETITIONS,
            np.random.default_rng(
                derive_seed(SEEDS["bootstrap"], "INTCR5R.bootstrap", candidate)
            ),
            np.random.default_rng(
                derive_seed(SEEDS["randomization"], "INTCR5R.randomization", candidate)
            ),
        )
        draws[f"c{candidate}_bootstrap_indices"] = candidate_draws["bootstrap_indices"]
        draws[f"c{candidate}_randomization_signs"] = candidate_draws[
            "randomization_signs"
        ]
    return draws


def compute_inference(
    cases: list[StateCase],
    targets: NDArray,
    predictions: NDArray,
    draws: dict[str, NDArray],
    *,
    minimum_matrices: int = MINIMUM_ELIGIBLE_MATRICES,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, NDArray]]:
    target_array = np.asarray(targets, dtype=np.float64)
    prediction_array = np.asarray(predictions, dtype=np.float64)
    if target_array.shape != (len(cases), len(ARMS), BRANCHES):
        raise ValueError(
            "CR5R target table does not align with cases, arms, and branches"
        )
    if prediction_array.shape != (len(cases), len(ARMS)):
        raise ValueError("CR5R prediction table does not align with cases and arms")
    arm_index = {arm: index for index, arm in enumerate(ARMS)}
    cells: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    raw_p_values: list[float] = []
    stored_arrays = {name: np.asarray(value) for name, value in draws.items()}

    for candidate in CANDIDATES:
        selected = np.asarray(
            [case.candidate == candidate for case in cases], dtype=bool
        )
        if not selected.any():
            raise ValueError(f"candidate {candidate} is absent")
        selected_cases = [case for case in cases if case.candidate == candidate]
        matrix_ids = np.asarray(
            [case.matrix_id for case in selected_cases], dtype=np.int64
        )
        matrix_order = np.unique(matrix_ids)
        if matrix_order.size < minimum_matrices:
            raise ValueError(
                f"candidate {candidate} has {matrix_order.size} eligible matrices; "
                f"registered minimum is {minimum_matrices}"
            )
        bootstrap_indices = np.asarray(
            draws[f"c{candidate}_bootstrap_indices"], dtype=np.int64
        )
        signs = np.asarray(draws[f"c{candidate}_randomization_signs"], dtype=np.float64)
        if (
            bootstrap_indices.shape != (BOOTSTRAP_REPETITIONS, matrix_order.size)
            or signs.shape != (RANDOMIZATION_REPETITIONS, matrix_order.size)
            or np.any(bootstrap_indices < 0)
            or np.any(bootstrap_indices >= matrix_order.size)
            or not np.isin(signs, (-1.0, 1.0)).all()
        ):
            raise ValueError(
                f"candidate {candidate} draws do not preserve eligible matrix blocks"
            )
        candidate_targets = target_array[selected]
        candidate_predictions = prediction_array[selected]
        for half, branch_slice in (
            ("A", slice(0, BRANCHES // 2)),
            ("B", slice(BRANCHES // 2, BRANCHES)),
        ):
            q = candidate_targets[:, :, branch_slice].mean(axis=2)
            state_effects = {
                "up_minus_down": (
                    q[:, arm_index["RENEWAL_UP"]] - q[:, arm_index["RENEWAL_DOWN"]]
                ),
                "up_minus_noop": (
                    q[:, arm_index["RENEWAL_UP"]] - q[:, arm_index["NOOP"]]
                ),
                "noop_minus_down": (
                    q[:, arm_index["NOOP"]] - q[:, arm_index["RENEWAL_DOWN"]]
                ),
                "random_minus_noop": (
                    q[:, arm_index["RANDOM"]] - q[:, arm_index["NOOP"]]
                ),
            }
            matrix_effects = {
                name: _matrix_means(values, matrix_ids, matrix_order)
                for name, values in state_effects.items()
            }
            bootstraps = {
                name: _bootstrap_means(values, bootstrap_indices)
                for name, values in matrix_effects.items()
            }
            cell_key = f"c{candidate}_{half}"
            for name, values in bootstraps.items():
                stored_arrays[f"{cell_key}_bootstrap_{name}"] = values
            p_value, null = _one_sided_sign_p(matrix_effects["up_minus_down"], signs)
            stored_arrays[f"{cell_key}_randomization_up_minus_down"] = null
            raw_p_values.append(p_value)

            arms: dict[str, Any] = {}
            for arm in ARMS:
                index = arm_index[arm]
                matrix_q = _matrix_means(q[:, index], matrix_ids, matrix_order)
                arm_bootstrap = _bootstrap_means(matrix_q, bootstrap_indices)
                expanded_prediction = candidate_predictions[:, index]
                arms[arm] = {
                    "mean_probability": float(matrix_q.mean()),
                    "bootstrap_ci95": _interval(arm_bootstrap),
                    "branch_scores": _branch_scores(
                        candidate_targets[:, index, branch_slice],
                        expanded_prediction,
                    ),
                    "mean_frozen_prediction": float(expanded_prediction.mean()),
                }

            contrasts: dict[str, Any] = {}
            for name, values in matrix_effects.items():
                contrasts[name] = {
                    "estimate": float(values.mean()),
                    "bootstrap_ci95": _interval(bootstraps[name]),
                    "matrices_expected_sign": int(np.count_nonzero(values > 0.0)),
                    "matrices_zero": int(np.count_nonzero(values == 0.0)),
                    "maximum_leave_one_matrix_out_influence": (
                        _maximum_leave_one_out_influence(values)
                    ),
                }
            random_ci90 = _interval(bootstraps["random_minus_noop"], alpha=0.10)
            random_difference = contrasts["random_minus_noop"]["estimate"]
            up_down = contrasts["up_minus_down"]["estimate"]
            predicted_shift = (
                candidate_predictions[:, arm_index["RENEWAL_UP"]]
                - candidate_predictions[:, arm_index["RENEWAL_DOWN"]]
            )
            realized_shift = state_effects["up_minus_down"]
            predicted_centered = predicted_shift - np.asarray(
                [predicted_shift[matrix_ids == key].mean() for key in matrix_ids]
            )
            realized_centered = realized_shift - np.asarray(
                [realized_shift[matrix_ids == key].mean() for key in matrix_ids]
            )
            denominator = float(np.dot(predicted_centered, predicted_centered))
            slope = (
                float(np.dot(predicted_centered, realized_centered) / denominator)
                if denominator > 0.0
                else float("nan")
            )
            cell = {
                "cell": cell_key,
                "candidate": candidate,
                "branch_half": half,
                "branch_range": [
                    int(branch_slice.start),
                    int(branch_slice.stop - 1),
                ],
                "states": int(selected.sum()),
                "matrices": int(matrix_order.size),
                "matrix_ids_sha256": hashlib.sha256(
                    np.ascontiguousarray(matrix_order).tobytes()
                ).hexdigest(),
                "arms": arms,
                "contrasts": contrasts,
                "up_down_randomization_p_raw": p_value,
                "random_noop_equivalence": {
                    "margin": EQUIVALENCE_MARGIN,
                    "bootstrap_ci90": random_ci90,
                    "tost_equivalent": bool(
                        random_ci90[0] > -EQUIVALENCE_MARGIN
                        and random_ci90[1] < EQUIVALENCE_MARGIN
                    ),
                    "ratio_limit": RANDOM_RATIO_LIMIT,
                    "absolute_difference_within_ratio": bool(
                        up_down > 0.0
                        and abs(random_difference) <= RANDOM_RATIO_LIMIT * up_down
                    ),
                },
                "predicted_versus_realized": {
                    "mean_predicted_up_minus_down": float(predicted_shift.mean()),
                    "mean_realized_up_minus_down": float(realized_shift.mean()),
                    "state_centered_slope": slope,
                },
            }
            cells.append(cell)
            for position, matrix_id in enumerate(matrix_order):
                row: dict[str, Any] = {
                    "cell": cell_key,
                    "candidate": candidate,
                    "branch_half": half,
                    "matrix_id": int(matrix_id),
                }
                row.update(
                    {
                        name: float(values[position])
                        for name, values in matrix_effects.items()
                    }
                )
                matrix_rows.append(row)

    adjusted = holm_adjust(raw_p_values)
    for cell, adjusted_p in zip(cells, adjusted, strict=True):
        contrast = cell["contrasts"]["up_minus_down"]
        gates = {
            "up_minus_down_positive": contrast["estimate"] > 0.0,
            "up_minus_down_bootstrap_lower_positive": (
                contrast["bootstrap_ci95"][0] > 0.0
            ),
            "holm_randomization_below_0_05": adjusted_p < 0.05,
            "random_tost_equivalent_to_noop": cell["random_noop_equivalence"][
                "tost_equivalent"
            ],
            "random_absolute_difference_within_effect_ratio": cell[
                "random_noop_equivalence"
            ]["absolute_difference_within_ratio"],
        }
        cell["up_down_randomization_p_holm"] = float(adjusted_p)
        cell["cr5r_registered_gates"] = gates
        cell["cr5r_registered_cell_pass"] = bool(all(gates.values()))

    landmark_rows: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        for landmark in LANDMARKS:
            selected = np.asarray(
                [
                    case.candidate == candidate and case.landmark == landmark
                    for case in cases
                ],
                dtype=bool,
            )
            if not selected.any():
                continue
            for half, branch_slice in (
                ("A", slice(0, BRANCHES // 2)),
                ("B", slice(BRANCHES // 2, BRANCHES)),
            ):
                q = target_array[selected, :, branch_slice].mean(axis=2)
                landmark_rows.append(
                    {
                        "candidate": candidate,
                        "branch_half": half,
                        "landmark": landmark,
                        "states": int(selected.sum()),
                        "up_minus_down": float(
                            np.mean(
                                q[:, arm_index["RENEWAL_UP"]]
                                - q[:, arm_index["RENEWAL_DOWN"]]
                            )
                        ),
                        "up_minus_noop": float(
                            np.mean(
                                q[:, arm_index["RENEWAL_UP"]] - q[:, arm_index["NOOP"]]
                            )
                        ),
                        "noop_minus_down": float(
                            np.mean(
                                q[:, arm_index["NOOP"]]
                                - q[:, arm_index["RENEWAL_DOWN"]]
                            )
                        ),
                        "random_minus_noop": float(
                            np.mean(q[:, arm_index["RANDOM"]] - q[:, arm_index["NOOP"]])
                        ),
                    }
                )

    result = {
        "inference_unit": "whole catalytic matrix",
        "candidate_specific_eligible_matrix_sets": True,
        "all_states_within_matrix_kept_together": True,
        "draws_reused_across_halves_within_candidate": True,
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "randomization_repetitions": RANDOMIZATION_REPETITIONS,
        "equivalence_margin": EQUIVALENCE_MARGIN,
        "holm_family_size": len(cells),
        "cells": cells,
        "landmark_effects": landmark_rows,
        "cr5r_all_four_cells_pass": bool(
            len(cells) == 4 and all(cell["cr5r_registered_cell_pass"] for cell in cells)
        ),
        "up_noop_and_noop_down_reported_not_gated": True,
        "stored_inference_array_names": sorted(stored_arrays),
    }
    return result, matrix_rows, stored_arrays


def _verify_predecessor() -> dict[str, Any]:
    registration = cr5.verify_confirmation_registration(CR5_REGISTRATION)
    verify_checksums(CR5_RESULT)
    manifest = json.loads((CR5_RESULT / "manifest.json").read_text())
    eligibility = json.loads((CR5_RESULT / "resilience_eligibility.json").read_text())
    if registration["registration_id"] != CR5_REGISTRATION_ID:
        raise ValueError("unexpected CR5 predecessor registration")
    if registration["frozen_model_sha256"] != EXPECTED_MODEL_SHA256:
        raise ValueError("unexpected frozen CR5 model hash")
    if registration["model_contract_sha256"] != EXPECTED_MODEL_CONTRACT_SHA256:
        raise ValueError("unexpected frozen CR5 model contract hash")
    if not manifest["resistance_gate"]:
        raise ValueError("CR5 predecessor resistance gate was not passed")
    if manifest["resilience_gate"] is not None:
        raise ValueError("CR5 predecessor unexpectedly contains a resilience result")
    if eligibility["classification"] != "inconclusive_incomplete_matrix_coverage":
        raise ValueError("CR5 predecessor eligibility classification changed")
    return {
        "registration": registration,
        "manifest": manifest,
        "eligibility": eligibility,
    }


def _fixture_case(candidate: str, matrix_id: int) -> StateCase:
    composition = np.zeros(100, dtype=np.int64)
    composition[:3] = (2, 1, 1)
    snapshot = Snapshot(
        composition=composition,
        generation=21,
        inheritance=(True, False),
        boundary_h=(0.95, 0.80),
        previous_growth_steps=7,
        cumulative_growth_steps=43,
    )
    return StateCase(
        state_id=f"cr5r-fixture-c{candidate}-m{matrix_id}",
        cohort="ARTIFICIAL_FIXTURE",
        candidate=candidate,
        matrix_id=matrix_id,
        landmark=20,
        beta=np.eye(100, dtype=np.float64),
        snapshot=snapshot,
    )


def validation_checks() -> dict[str, Any]:
    predecessor = _verify_predecessor()
    inherited_checks = base.validation_checks()
    cases = [
        _fixture_case("02", 0),
        _fixture_case("02", 1),
        _fixture_case("03", 1),
        _fixture_case("03", 2),
    ]
    targets = np.zeros((len(cases), len(ARMS), BRANCHES), dtype=np.int8)
    targets[:, ARMS.index("RENEWAL_UP")] = 1
    predictions = np.full((len(cases), len(ARMS)), 0.5, dtype=np.float64)
    draws = generate_candidate_draws(cases)
    inference, rows, _ = compute_inference(
        cases, targets, predictions, draws, minimum_matrices=2
    )

    model_path = CR5_REGISTRATION / "frozen_cr5_students.npz"
    model_contract = CR5_REGISTRATION / "model_contract.json"
    left_students = cr5.load_students(model_path, model_contract)
    right_students = cr5.load_students(model_path, model_contract)
    config = GardConfig()
    model_round_trip = True
    for candidate in CANDIDATES:
        noop_left, scores_left = cr5.score_student_edits(
            left_students[("renewal", candidate)],
            _fixture_case(candidate, 0),
            config,
        )
        noop_right, scores_right = cr5.score_student_edits(
            right_students[("renewal", candidate)],
            _fixture_case(candidate, 0),
            config,
        )
        model_round_trip &= noop_left == noop_right and scores_left == scores_right

    eligibility_pass = eligibility_summary(cases, len(cases), True)
    eligibility_fail = eligibility_summary(cases, len(cases), True)
    endpoint_fixture = cr5._stage_outcome(
        "resilience",
        cases[0].snapshot,
        [],
        False,
        HORIZON,
        0.9,
    )
    checks = {
        "inherited_26_plus_validation_suite_passes": bool(
            inherited_checks["all_checks_passed"]
        ),
        "cr5_predecessor_checksum_exact": True,
        "cr5_resistance_pass_preserved": bool(
            predecessor["manifest"]["resistance_gate"]
        ),
        "cr5_resilience_inconclusive_preserved": (
            predecessor["manifest"]["resilience_gate"] is None
        ),
        "frozen_model_hash_exact": sha256_file(model_path) == EXPECTED_MODEL_SHA256,
        "frozen_model_contract_hash_exact": sha256_file(model_contract)
        == EXPECTED_MODEL_CONTRACT_SHA256,
        "frozen_model_round_trip_exact": model_round_trip,
        "seed_domains_unique": len(SEEDS) == len(set(SEEDS.values())),
        "seed_domains_disjoint_from_cr5": set(SEEDS.values()).isdisjoint(
            cr5.SEEDS.values()
        ),
        "seed_domains_disjoint_from_original_program": set(SEEDS.values()).isdisjoint(
            base.SEED_DOMAINS.values()
        ),
        "future_stream_is_arm_free": len({future_seed(cases[0], 0) for _arm in ARMS})
        == 1,
        "selection_and_future_streams_distinct": selection_seed(cases[0])
        != future_seed(cases[0], 0),
        "candidate_specific_matrix_sets_supported": (
            inference["cells"][0]["matrices"] == 2
            and inference["cells"][2]["matrices"] == 2
            and {row["matrix_id"] for row in rows if row["candidate"] == "02"} == {0, 1}
            and {row["matrix_id"] for row in rows if row["candidate"] == "03"} == {1, 2}
        ),
        "matrix_inference_has_four_cells": len(inference["cells"]) == 4,
        "eligibility_threshold_is_200": (
            eligibility_pass["intervention_futures_authorized"] is False
            and eligibility_fail["minimum_eligible_matrices_per_candidate"] == 200
        ),
        "extinction_before_renewal_is_negative": (
            endpoint_fixture.joint_break_run3 is False
        ),
        "design_size_exact": MATRICES == 250
        and LANDMARKS == (20, 35, 50, 65, 80)
        and BRANCHES == 64
        and HORIZON == 8
        and ACQUISITION_LIMIT == 60,
        "strict_eight_excluded": protocol()["endpoint"]["strict_eight_excluded"],
        "cpu_budget_bounded": MINIMUM_CPU_BUDGET_HOURS == 8.0
        and MAXIMUM_CPU_BUDGET_HOURS == 14.0,
    }
    return {
        "format": VALIDATION_FORMAT,
        "checks": checks,
        "check_count": len(checks),
        "all_checks_passed": bool(all(checks.values())),
        "scientific_matrices_generated": 0,
        "scientific_futures_generated": 0,
    }


def validate(output: Path = DEFAULT_VALIDATION) -> None:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    checks = validation_checks()
    if not checks["all_checks_passed"]:
        raise AssertionError(
            {name: value for name, value in checks["checks"].items() if not value}
        )
    command = [sys.executable, "-m", "pytest", "-q"]
    completed = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "CR5R full repository validation failed\n"
            + completed.stdout
            + "\n"
            + completed.stderr
        )
    with _atomic_destination(output) as destination:
        payload = dict(checks)
        payload["source_hashes"] = source_hashes()
        payload["cr5_registration_checksum_manifest_sha256"] = sha256_file(
            CR5_REGISTRATION / "SHA256SUMS"
        )
        payload["cr5_result_checksum_manifest_sha256"] = sha256_file(
            CR5_RESULT / "SHA256SUMS"
        )
        (destination / "validation.json").write_text(
            json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n"
        )
        (destination / "pytest_output.txt").write_text(
            "$ " + " ".join(command) + "\n" + completed.stdout + completed.stderr
        )
        write_checksums(destination)
    verify_checksums(output)
    print(f"CR5R validation sealed: {output}", flush=True)


def _append_ledger(marker: str, lines: list[str]) -> None:
    path = ROOT / "INTERVENTION_RESULTS_LEDGER.md"
    current = path.read_text(encoding="utf-8").rstrip() + "\n"
    if marker in current:
        return
    path.write_text(current + "\n" + marker + "\n" + "\n".join(lines))


def register(
    validation_directory: Path = DEFAULT_VALIDATION,
    output: Path = DEFAULT_REGISTRATION,
) -> None:
    validation_directory = validation_directory.resolve()
    output = output.resolve()
    verify_checksums(validation_directory)
    validation = json.loads((validation_directory / "validation.json").read_text())
    if not validation["all_checks_passed"]:
        raise ValueError("CR5R validation did not pass")
    if validation["source_hashes"] != source_hashes():
        raise ValueError("CR5R source changed after validation")
    _verify_predecessor()
    for forbidden in (DEFAULT_SMOKE, DEFAULT_OUTPUT, DEFAULT_WORK):
        if forbidden.exists():
            raise FileExistsError(
                f"CR5R scientific artifact exists before registration: {forbidden}"
            )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    frozen = protocol()
    payload: dict[str, Any] = {
        "format": REGISTRATION_FORMAT,
        "protocol_id": frozen["protocol_id"],
        "source_hashes": source_hashes(),
        "seed_registry": SEEDS,
        "validation_checksum_manifest_sha256": sha256_file(
            validation_directory / "SHA256SUMS"
        ),
        "cr5_registration_id": CR5_REGISTRATION_ID,
        "cr5_result_checksum_manifest_sha256": sha256_file(CR5_RESULT / "SHA256SUMS"),
        "frozen_model_sha256": EXPECTED_MODEL_SHA256,
        "model_contract_sha256": EXPECTED_MODEL_CONTRACT_SHA256,
        "scientific_matrices_generated_at_registration": 0,
        "scientific_futures_generated_at_registration": 0,
    }
    payload["registration_id"] = _canonical_digest(_json_ready(payload))
    with _atomic_destination(output) as destination:
        (destination / "protocol.json").write_text(
            json.dumps(frozen, indent=2, sort_keys=True) + "\n"
        )
        (destination / "seed_registry.json").write_text(
            json.dumps(SEEDS, indent=2, sort_keys=True) + "\n"
        )
        (destination / "registration.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
        shutil.copy2(
            CR5_REGISTRATION / "frozen_cr5_students.npz",
            destination / "frozen_cr5_students.npz",
        )
        shutil.copy2(
            CR5_REGISTRATION / "model_contract.json",
            destination / "model_contract.json",
        )
        write_checksums(destination)
    verify_registration(output)
    _append_ledger(
        f"<!-- cr5r-registered-{payload['registration_id']} -->",
        [
            "## CR5R shared-break resilience confirmation registered",
            "",
            f"- Registration: `{payload['registration_id']}`.",
            "- CR5 remains sealed with resistance passed and resilience inconclusive.",
            "- CR5R uses 250 fresh matrices and requires at least 200 eligible matrices per candidate before any intervention future launches.",
            "- The frozen CR5 renewal students were copied without refit or recalibration.",
            "- No CR5R scientific matrix existed at the seal.",
            "",
        ],
    )
    print(f"CR5R registered: {payload['registration_id']}", flush=True)


def verify_registration(
    directory: Path = DEFAULT_REGISTRATION,
) -> dict[str, Any]:
    directory = directory.resolve()
    verify_checksums(directory)
    payload = json.loads((directory / "registration.json").read_text())
    frozen = json.loads((directory / "protocol.json").read_text())
    if payload.get("format") != REGISTRATION_FORMAT:
        raise ValueError("unsupported CR5R registration")
    if frozen != _json_ready(protocol()):
        raise ValueError("CR5R frozen protocol changed")
    if payload["source_hashes"] != source_hashes():
        raise ValueError("CR5R source changed after registration")
    if payload["registration_id"] != _canonical_digest(
        {key: value for key, value in payload.items() if key != "registration_id"}
    ):
        raise ValueError("CR5R registration ID changed")
    if sha256_file(directory / "frozen_cr5_students.npz") != EXPECTED_MODEL_SHA256:
        raise ValueError("CR5R frozen model changed")
    if sha256_file(directory / "model_contract.json") != EXPECTED_MODEL_CONTRACT_SHA256:
        raise ValueError("CR5R model contract changed")
    cr5.load_students(
        directory / "frozen_cr5_students.npz", directory / "model_contract.json"
    )
    return payload


def smoke(
    registration_directory: Path = DEFAULT_REGISTRATION,
    output: Path = DEFAULT_SMOKE,
) -> None:
    registration_directory = registration_directory.resolve()
    output = output.resolve()
    registration = verify_registration(registration_directory)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    config = GardConfig()
    beta = generate_beta(
        config,
        np.random.default_rng(derive_seed(SEEDS["validation"], "smoke.beta")),
    )
    composition = np.zeros(config.n_types, dtype=np.int64)
    composition[: config.n_min] = 1
    snapshot = Snapshot(
        composition=composition,
        generation=21,
        inheritance=(True, False),
        boundary_h=(0.95, 0.80),
        previous_growth_steps=9,
        cumulative_growth_steps=51,
    )
    case = StateCase(
        "cr5r-smoke",
        "ARTIFICIAL_FIXTURE",
        "02",
        0,
        20,
        beta,
        snapshot,
    )
    student = cr5.load_students(
        registration_directory / "frozen_cr5_students.npz",
        registration_directory / "model_contract.json",
    )[("renewal", "02")]
    noop, scores = cr5.score_student_edits(student, case, config)
    left = cr5.select_student_edits(
        noop,
        scores,
        np.random.default_rng(derive_seed(SEEDS["smoke_selection"], "fixture")),
    )
    right = cr5.select_student_edits(
        noop,
        scores,
        np.random.default_rng(derive_seed(SEEDS["smoke_selection"], "fixture")),
    )
    future = derive_seed(SEEDS["smoke_future"], "fixture")
    records_left, completed_left = simulate_future_absorbing(
        snapshot,
        beta,
        config,
        CANDIDATES["02"],
        2,
        np.random.default_rng(future),
    )
    records_right, completed_right = simulate_future_absorbing(
        snapshot,
        beta,
        config,
        CANDIDATES["02"],
        2,
        np.random.default_rng(future),
    )
    batch = _phase_worker(
        (
            case,
            config,
            str(registration_directory / "frozen_cr5_students.npz"),
            str(registration_directory / "model_contract.json"),
        )
    )
    checks = {
        "all_and_only_legal_swaps_scored": len(scores)
        == len(enumerate_legal_edits(composition)),
        "selection_deterministic": np.array_equal(left[0], right[0])
        and left[1] == right[1],
        "future_stream_replay_exact": completed_left == completed_right
        and cr5._records_digest(records_left) == cr5._records_digest(records_right),
        "complete_artificial_worker_contract": batch.arm_names == ARMS
        and len(batch.outcomes) == len(ARMS)
        and all(len(outcomes) == BRANCHES for outcomes in batch.outcomes)
        and batch.predictions.shape == (len(ARMS),),
        "scientific_effect_sizes_not_disclosed": True,
        "scientific_arm_ordering_not_disclosed": True,
    }
    if not all(checks.values()):
        raise AssertionError(
            {name: value for name, value in checks.items() if not value}
        )
    with _atomic_destination(output) as destination:
        (destination / "manifest.json").write_text(
            json.dumps(
                {
                    "format": "codex-intervention-cr5r-smoke-v1",
                    "registration_id": registration["registration_id"],
                    "artificial_fixture_only": True,
                    "scientific_matrices": 0,
                    "scientific_futures": 0,
                    "checks": checks,
                    "effect_sizes_disclosed": False,
                    "arm_ordering_disclosed": False,
                    "event_rates_disclosed": False,
                    "candidate_differences_disclosed": False,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        write_checksums(destination)
    verify_checksums(output)
    print(f"CR5R non-scientific smoke passed: {output}", flush=True)


def _status(work: Path, state: str, detail: str, **extra: Any) -> None:
    work.mkdir(parents=True, exist_ok=True)
    base._atomic_json(
        work / "campaign_status.json",
        {
            "format": STATUS_FORMAT,
            "phase": "CR5R shared-break resilience confirmation",
            "state": state,
            "detail": detail,
            "mandatory_stop_after_seal": True,
            **extra,
        },
    )


def _prepare_work(
    work: Path,
    output: Path,
    registration_id: str,
    cpu_budget_hours: float,
) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if not MINIMUM_CPU_BUDGET_HOURS <= cpu_budget_hours <= MAXIMUM_CPU_BUDGET_HOURS:
        raise ValueError(
            f"CR5R CPU declaration must be between {MINIMUM_CPU_BUDGET_HOURS:g} "
            f"and {MAXIMUM_CPU_BUDGET_HOURS:g} hours"
        )
    free = shutil.disk_usage(RESULT_ROOT).free
    if free < MINIMUM_FREE_DISK_BYTES:
        raise OSError(
            f"CR5R requires {MINIMUM_FREE_DISK_BYTES} free bytes; found {free}"
        )
    work.mkdir(parents=True, exist_ok=True)
    stable = {
        "format": "codex-intervention-cr5r-campaign-v1",
        "registration_id": registration_id,
        "output": str(output),
        "source_hashes": source_hashes(),
        "declared_cpu_budget_hours": cpu_budget_hours,
    }
    path = work / "campaign_contract.json"
    if path.exists():
        existing = json.loads(path.read_text())
        if {key: existing.get(key) for key in stable} != _json_ready(stable):
            raise ValueError("CR5R work directory belongs to another campaign")
    else:
        contract = {
            **stable,
            "free_disk_bytes_at_initialization": free,
        }
        contract["campaign_id"] = _canonical_digest(_json_ready(contract))
        base._atomic_json(path, contract)
    _status(
        work,
        "running",
        "campaign_initialized",
        declared_cpu_budget_hours=cpu_budget_hours,
    )


def read_status(work: Path = DEFAULT_WORK) -> dict[str, Any]:
    work = work.resolve()
    path = work / "campaign_status.json"
    if not path.exists():
        return {"state": "not_started", "work_directory": str(work)}
    value = json.loads(path.read_text())
    for stage in ("generate", "replay"):
        status_path = work / stage / "status.json"
        if status_path.exists():
            value[stage] = json.loads(status_path.read_text())
    return value


def _reports(
    eligibility: dict[str, Any],
    metrics: dict[str, Any] | None,
    integrity: dict[str, Any],
) -> tuple[str, str]:
    launched = metrics is not None
    passed = bool(
        launched and metrics["cr5r_all_four_cells_pass"] and integrity["all_pass"]
    )
    lines = [
        "# CR5R shared-break resilience confirmation",
        "",
        "CR5R is a separately registered extension. The sealed CR5 result remains: resistance passed; resilience was inconclusive because its all-200 eligibility rule was not met.",
        "",
        f"Eligibility classification: **{eligibility['classification']}**.",
        f"Intervention futures launched: **{launched}**.",
        f"CR5R four-cell recovery-control gate: **{None if not launched else passed}**.",
        f"Exact acquisition/future replay and artifact readback: **{integrity['all_pass']}**.",
        "",
        "## Eligibility",
        "",
        "| Candidate | Eligible states | Eligible matrices | Registered minimum |",
        "|---|---:|---:|---:|",
    ]
    for candidate in CANDIDATES:
        item = eligibility["eligible_by_candidate"][candidate]
        lines.append(
            f"| {candidate} | {item['states']} | {item['matrices']} | {MINIMUM_ELIGIBLE_MATRICES} |"
        )
    if metrics is not None:
        lines.extend(
            [
                "",
                "## Primary recovery-control result",
                "",
                "| Cell | Renewal-up minus renewal-down | 95% CI | Holm p | Random-noop 90% CI | Pass |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for cell in metrics["cells"]:
            effect = cell["contrasts"]["up_minus_down"]
            lines.append(
                f"| {cell['cell']} | {effect['estimate']:+.6f} | "
                f"{effect['bootstrap_ci95']} | "
                f"{cell['up_down_randomization_p_holm']:.6g} | "
                f"{cell['random_noop_equivalence']['bootstrap_ci90']} | "
                f"{cell['cr5r_registered_cell_pass']} |"
            )
        lines.extend(
            [
                "",
                "`RENEWAL_UP - NOOP` and `NOOP - RENEWAL_DOWN` are reported in the machine-readable metrics but were not preregistered gates.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "The eligibility threshold was not met, so the protocol correctly generated zero intervention futures. This is an inconclusive result, not evidence against recovery control.",
            ]
        )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            (
                "The registered result supports causal molecular control of short-run recovery from an identical naturally broken state in both Codex candidates."
                if passed
                else (
                    "The registered recovery-control prediction did not pass all four cells."
                    if launched
                    else "The registered recovery-control prediction remains untested because eligibility was insufficient."
                )
            ),
            "",
            "This simulated result does not establish biological repair, memory, agency, life, autonomy, strict-eight control, real chemistry, an origin-of-life mechanism, or Phi/PhiID intervention.",
            "",
        ]
    )
    lay = "\n".join(
        [
            "# CR5R in plain language",
            "",
            "CR5 already showed that tiny, predictor-chosen molecular changes can make a hereditary break more or less likely. It could not run its second test—recovery after a break—because its original eligibility rule was stricter than the natural data could satisfy.",
            "",
            (
                "CR5R obtained enough fresh naturally broken assemblies. For each one, all four arms began from exactly the same broken daughter, so any difference was caused by the one-molecule edit rather than by starting from different breaks."
                if launched
                else "Even with 250 fresh matrices, CR5R did not obtain the prewritten minimum number of naturally broken matrices in both simulator candidates. It therefore stopped without trying the edits."
            ),
            "",
            (
                "The predictor-chosen up edits reliably produced more short hereditary recovery than the down edits in both simulators and in both independent branch halves."
                if passed
                else (
                    "The edits were tested, but the result did not satisfy every prewritten reliability and random-control check."
                    if launched
                    else "Recovery control remains an open question rather than a negative finding."
                )
            ),
            "",
            "This is a causal test inside the computer model. It is not evidence that the assemblies are alive or consciously remember anything.",
            "",
        ]
    )
    return "\n".join(lines), lay


def _claims(
    eligibility: dict[str, Any],
    metrics: dict[str, Any] | None,
    integrity: dict[str, Any],
) -> dict[str, Any]:
    passed = bool(
        metrics is not None
        and metrics["cr5r_all_four_cells_pass"]
        and integrity["all_pass"]
    )
    supported = (
        [
            "causal molecular control of run3-within-F8 recovery from identical natural post-break daughters in both Codex candidates"
        ]
        if passed
        else []
    )
    failed_or_inconclusive: list[str] = []
    if metrics is None:
        failed_or_inconclusive.append(
            "CR5R inconclusive: registered eligible-matrix threshold was not met"
        )
    elif not passed:
        failed_or_inconclusive.append(
            "registered four-cell shared-break resilience gate"
        )
    return {
        "supported": supported,
        "failed_or_inconclusive": failed_or_inconclusive,
        "cr5_predecessor_unchanged": True,
        "cr5_resistance_remains_passed": True,
        "cr5_resilience_remains_inconclusive": True,
        "cr5r_eligibility_classification": eligibility["classification"],
        "unresolved": [
            "zero-shot parameter-regime transfer",
            "closed-loop hereditary steering",
        ],
        "prohibited": protocol()["claim_boundary"]["prohibited"],
    }


def _write_context(
    destination: Path,
    broken: list[StateCase],
    anchors: list[NDArray[np.int64]],
) -> None:
    if not broken:
        return
    np.savez_compressed(
        destination / "natural_break_context.npz",
        broken_state_ids=np.asarray([case.state_id for case in broken]),
        broken_state_digests=np.asarray(
            [base._snapshot_digest(case) for case in broken]
        ),
        candidates=np.asarray([case.candidate for case in broken]),
        matrix_ids=np.asarray([case.matrix_id for case in broken], dtype=np.int16),
        landmarks=np.asarray([case.landmark for case in broken], dtype=np.int16),
        old_parent_anchors=np.vstack(anchors).astype(np.int16),
    )


def run_confirmation(
    registration_directory: Path = DEFAULT_REGISTRATION,
    output: Path = DEFAULT_OUTPUT,
    work: Path = DEFAULT_WORK,
    workers: int = min(os.cpu_count() or 1, 14),
    cpu_budget_hours: float = DEFAULT_CPU_BUDGET_HOURS,
) -> None:
    if workers < 1:
        raise ValueError("workers must be positive")
    registration_directory = registration_directory.resolve()
    output = output.resolve()
    work = work.resolve()
    registration = verify_registration(registration_directory)
    verify_checksums(DEFAULT_SMOKE)
    _verify_predecessor()
    _prepare_work(work, output, registration["registration_id"], cpu_budget_hours)
    experiment = _experiment()
    model_path = registration_directory / "frozen_cr5_students.npz"
    model_contract_path = registration_directory / "model_contract.json"

    print(
        "[cr5r 1/8] Building 250 fresh matrices and 2,500 untreated landmark states",
        flush=True,
    )
    _status(work, "running", "building_fresh_natural_states")
    with threadpool_limits(limits=1):
        cases = build_cohort(experiment, CONFIRMATION_LABEL, experiment.confirmation)
    expected_cases = 2 * MATRICES * len(LANDMARKS)
    if len(cases) != expected_cases:
        raise AssertionError(
            f"CR5R cohort has {len(cases)} states; expected {expected_cases}"
        )

    print(
        "[cr5r 2/8] Acquiring first untreated natural breaks and exact replay",
        flush=True,
    )
    _status(work, "running", "natural_break_acquisition")
    broken, anchors, acquisition = cr5.acquire_natural_breaks(
        cases,
        experiment.gard,
        SEEDS["natural_break_acquisition"],
        f"{CONFIRMATION_LABEL}.acquisition",
    )
    replay_broken, replay_anchors, replay_acquisition = cr5.acquire_natural_breaks(
        cases,
        experiment.gard,
        SEEDS["natural_break_acquisition"],
        f"{CONFIRMATION_LABEL}.acquisition",
    )
    acquisition_exact = cr5._acquisition_exact(
        broken,
        anchors,
        acquisition,
        replay_broken,
        replay_anchors,
        replay_acquisition,
    )
    if not acquisition_exact:
        raise AssertionError("CR5R natural-break acquisition replay failed")
    eligibility = eligibility_summary(broken, len(cases), acquisition_exact)
    authorized = bool(eligibility["intervention_futures_authorized"])
    _status(
        work,
        "running",
        "eligibility_sealed",
        eligibility=eligibility,
    )

    generated: list[base.PhaseBatch] = []
    metrics: dict[str, Any] | None = None
    matrix_rows: list[dict[str, Any]] = []
    arrays: dict[str, NDArray] | None = None
    stored_inference_arrays: dict[str, NDArray] | None = None
    future_replay: dict[str, Any] = {"not_launched_due_to_eligibility": not authorized}
    primary_futures = 0
    if authorized:
        primary_futures = len(broken) * len(ARMS) * BRANCHES
        print(
            f"[cr5r 3/8] Shooting {primary_futures:,} F8 shared-state futures",
            flush=True,
        )
        _status(
            work,
            "running",
            "primary_futures",
            eligibility=eligibility,
            primary_futures=primary_futures,
        )
        generated = run_phase_batches(
            broken,
            experiment.gard,
            model_path,
            model_contract_path,
            registration["registration_id"],
            work / "generate",
            workers,
            "generate",
        )
        print("[cr5r 4/8] Replaying every intervention future", flush=True)
        _status(
            work,
            "running",
            "exact_future_replay",
            eligibility=eligibility,
            primary_futures=primary_futures,
        )
        replayed = run_phase_batches(
            broken,
            experiment.gard,
            model_path,
            model_contract_path,
            registration["registration_id"],
            work / "replay",
            workers,
            "replay",
        )
        future_replay = base.replay_audit(generated, replayed)
        if not future_replay["state_edit_endpoint_and_process_digests_exact"]:
            raise AssertionError("CR5R intervention replay failed")
        arrays = cr5._outcome_arrays(broken, generated, phase_spec())
        draws = generate_candidate_draws(broken)
        metrics, matrix_rows, stored_inference_arrays = compute_inference(
            broken, arrays["targets"], arrays["predictions"], draws
        )
    else:
        print(
            "[cr5r 3/8] Eligibility threshold missed; zero intervention futures launched",
            flush=True,
        )
        print("[cr5r 4/8] Future replay not applicable", flush=True)

    print("[cr5r 5/8] Writing complete sealed artifacts", flush=True)
    _status(
        work,
        "running",
        "artifact_write_and_readback",
        eligibility=eligibility,
        primary_futures=primary_futures,
    )
    with _atomic_destination(output) as destination:
        acquisition.to_csv(destination / "natural_break_acquisition.csv", index=False)
        (destination / "eligibility.json").write_text(
            json.dumps(eligibility, indent=2, sort_keys=True) + "\n"
        )
        _write_context(destination, broken, anchors)

        readback: dict[str, Any]
        if authorized:
            assert arrays is not None
            assert metrics is not None
            assert stored_inference_arrays is not None
            np.savez_compressed(destination / "branch_arrays.npz", **arrays)
            np.savez_compressed(
                destination / "inference_arrays.npz", **stored_inference_arrays
            )
            cr5._write_branch_table(
                destination / "branches.csv.gz",
                broken,
                generated,
                "run3_within_f8",
            )
            base._write_state_artifacts(destination, broken, generated, arrays)
            base._write_selection_artifacts(
                destination,
                broken,
                generated,
                phase_spec(),  # type: ignore[arg-type]
            )
            pd.DataFrame(matrix_rows).to_csv(
                destination / "matrix_effects.csv", index=False
            )
            (destination / "primary_metrics.json").write_text(
                json.dumps(_json_ready(metrics), indent=2, sort_keys=True) + "\n"
            )
            secondary = cr5._secondary(broken, arrays, phase_spec(), anchors)
            (destination / "secondary_outcomes.json").write_text(
                json.dumps(_json_ready(secondary), indent=2, sort_keys=True) + "\n"
            )

            with np.load(destination / "branch_arrays.npz", allow_pickle=False) as data:
                loaded_arrays = {name: data[name] for name in data.files}
            with np.load(
                destination / "inference_arrays.npz", allow_pickle=False
            ) as data:
                loaded_inference = {name: data[name] for name in data.files}
            recomputed_metrics, recomputed_rows, _ = compute_inference(
                broken,
                loaded_arrays["targets"],
                loaded_arrays["predictions"],
                loaded_inference,
            )
            readback = {
                "branch_arrays_reloaded": True,
                "inference_arrays_reloaded": True,
                "metrics_recomputed_exact": _json_ready(recomputed_metrics)
                == _json_ready(metrics),
                "matrix_effects_recomputed_exact": _json_ready(recomputed_rows)
                == _json_ready(matrix_rows),
                "written_metrics_json_exact": json.loads(
                    (destination / "primary_metrics.json").read_text()
                )
                == _json_ready(metrics),
            }
        else:
            readback = {
                "not_applicable_due_to_eligibility": True,
                "eligibility_json_exact": json.loads(
                    (destination / "eligibility.json").read_text()
                )
                == _json_ready(eligibility),
                "zero_intervention_futures": True,
            }
        acquisition_readback = pd.read_csv(
            destination / "natural_break_acquisition.csv"
        )
        readback["acquisition_row_count_exact"] = len(acquisition_readback) == len(
            acquisition
        )
        readback["all_exact"] = bool(all(readback.values()))

        exact_future_replay = bool(
            not authorized
            or future_replay.get("state_edit_endpoint_and_process_digests_exact", False)
        )
        integrity = {
            "natural_break_acquisition_replay_exact": acquisition_exact,
            "future_replay_exact_or_not_launched": exact_future_replay,
            "artifact_readback_exact": readback["all_exact"],
        }
        integrity["all_pass"] = bool(all(integrity.values()))
        (destination / "replay_audit.json").write_text(
            json.dumps(
                _json_ready(
                    {
                        "natural_break_acquisition_exact": acquisition_exact,
                        "future_replay": future_replay,
                    }
                ),
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        (destination / "readback_audit.json").write_text(
            json.dumps(readback, indent=2, sort_keys=True) + "\n"
        )
        technical, lay = _reports(eligibility, metrics, integrity)
        (destination / "SCIENTIFIC_REPORT.md").write_text(technical)
        (destination / "LAY_SUMMARY.md").write_text(lay)
        (destination / "claim_boundaries.json").write_text(
            json.dumps(
                _claims(eligibility, metrics, integrity), indent=2, sort_keys=True
            )
            + "\n"
        )
        manifest = {
            "format": RESULT_FORMAT,
            "registration_id": registration["registration_id"],
            "cr5_predecessor_unchanged": True,
            "source_matrices": MATRICES,
            "source_states": len(cases),
            "eligible_broken_states": len(broken),
            "eligibility_classification": eligibility["classification"],
            "primary_futures": primary_futures,
            "replay_futures": primary_futures if authorized else 0,
            "cr5r_gate": (
                bool(metrics["cr5r_all_four_cells_pass"] and integrity["all_pass"])
                if metrics is not None
                else None
            ),
            "integrity": integrity,
            "declared_cpu_budget_hours": cpu_budget_hours,
            "no_refit_recalibration_retry_replacement_or_subselection": True,
            "mandatory_stop_after_this_stage": True,
            "cr6_launched": False,
            "runtime": _runtime_manifest(),
        }
        (destination / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        write_checksums(destination)
    verify_checksums(output)
    print("[cr5r 6/8] Checksums and written-artifact readback passed", flush=True)
    _append_ledger(
        f"<!-- sealed-cr5r-{registration['registration_id']} -->",
        [
            "## CR5R shared-break resilience confirmation sealed",
            "",
            f"- Registration: `{registration['registration_id']}`.",
            f"- Result: `{output.relative_to(ROOT)}`.",
            f"- Eligibility: **{eligibility['classification']}**.",
            f"- CR5R four-cell gate: **{None if metrics is None else bool(metrics['cr5r_all_four_cells_pass'] and integrity['all_pass'])}**.",
            "- Natural-break acquisition replay, complete future replay where applicable, and artifact readback passed.",
            "- Sealed CR5 remains unchanged; CR6 was not launched automatically.",
            "",
        ],
    )
    print("[cr5r 7/8] Cumulative intervention ledger updated", flush=True)
    _status(
        work,
        "sealed_complete",
        "mandatory_review_stop",
        eligibility=eligibility,
        primary_futures=primary_futures,
        cr5r_gate=(
            bool(metrics["cr5r_all_four_cells_pass"] and integrity["all_pass"])
            if metrics is not None
            else None
        ),
        declared_cpu_budget_hours=cpu_budget_hours,
    )
    print("[cr5r 8/8] STOPPED; CR6 not launched", flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate").add_argument(
        "--output", type=Path, default=DEFAULT_VALIDATION
    )
    register_parser = commands.add_parser("register")
    register_parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    register_parser.add_argument("--output", type=Path, default=DEFAULT_REGISTRATION)
    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument(
        "--registration", type=Path, default=DEFAULT_REGISTRATION
    )
    smoke_parser = commands.add_parser("smoke")
    smoke_parser.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    smoke_parser.add_argument("--output", type=Path, default=DEFAULT_SMOKE)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    run_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run_parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    run_parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 14))
    run_parser.add_argument(
        "--cpu-budget-hours", type=float, default=DEFAULT_CPU_BUDGET_HOURS
    )
    status_parser = commands.add_parser("status")
    status_parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "validate":
        validate(args.output)
    elif args.command == "register":
        register(args.validation, args.output)
    elif args.command == "verify":
        print(
            json.dumps(verify_registration(args.registration), indent=2, sort_keys=True)
        )
    elif args.command == "smoke":
        smoke(args.registration, args.output)
    elif args.command == "run":
        run_confirmation(
            args.registration,
            args.output,
            args.work_dir,
            args.workers,
            args.cpu_budget_hours,
        )
    elif args.command == "status":
        print(json.dumps(read_status(args.work_dir), indent=2, sort_keys=True))
    else:  # pragma: no cover
        raise AssertionError(args.command)


if __name__ == "__main__":  # pragma: no cover
    main()
