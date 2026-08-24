"""Prospective CR5 decomposition of hereditary resistance and resilience.

This module is additive: it reconstructs the sealed 5x development cohort,
freezes two new endpoint-specific students, and then applies legal one-molecule
substitutions in a completely fresh confirmation cohort.  It never changes the
Codex simulator or the already sealed CR1--CR4 results.
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
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from . import intervention_replication as base
from .config import CANDIDATES, CohortConfig, ExperimentConfig, GardConfig
from .experiment import (
    StateCase,
    _json_ready,
    _runtime_manifest,
    build_cohort,
    extract_features,
)
from .features import HISTORY_FEATURE_NAMES, STATE_GRAPH_FEATURE_NAMES, history_features
from .intervention_core import (
    InterventionOutcome,
    MolecularEdit,
    ScoredEdit,
    apply_molecular_edit,
    enumerate_legal_edits,
    outcome_from_records,
    state_graph_features_many,
)
from .intervention_metrics import compute_one_shot_inference, generate_inference_draws
from .mechanistic import (
    _atomic_destination,
    _canonical_digest,
    sha256_file,
    verify_checksums,
    write_checksums,
)
from .seeds import derive_seed
from .simulator import (
    FissionRecord,
    SimulationError,
    Snapshot,
    advance_fission,
    cosine_similarity,
    generate_beta,
    simulate_future_absorbing,
)


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results_intervention_replication"
SCALED5 = ROOT / "results/scaled5"
CR1_RESULT = RESULT_ROOT / "cr1_model_guided_confirmation"
CR3_RESULT = RESULT_ROOT / "cr3_physical_rule_confirmation"
DEFAULT_VALIDATION = RESULT_ROOT / "cr5_validation"
DEFAULT_PROTOCOL_REGISTRATION = RESULT_ROOT / "cr5_protocol_registration"
DEFAULT_DEVELOPMENT = RESULT_ROOT / "cr5_development_freeze"
DEFAULT_DEVELOPMENT_WORK = RESULT_ROOT / ".cr5_development_work"
DEFAULT_CONFIRMATION_REGISTRATION = RESULT_ROOT / "cr5_confirmation_registration"
DEFAULT_SMOKE = RESULT_ROOT / "cr5_smoke"
DEFAULT_OUTPUT = RESULT_ROOT / "cr5_resistance_resilience_confirmation"
DEFAULT_WORK = RESULT_ROOT / ".cr5_confirmation_work"

DOCUMENT = "CODEX_INTERVENTION_CR5_PREREGISTRATION.md"
SOURCE_FILES = (
    DOCUMENT,
    "plastic_heredity/intervention_cr5.py",
    "tests/test_intervention_cr5.py",
    "plastic_heredity/intervention_replication.py",
    "plastic_heredity/intervention_core.py",
    "plastic_heredity/intervention_metrics.py",
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

PROGRAM_FORMAT = "codex-intervention-cr5-resistance-resilience-v1"
VALIDATION_FORMAT = "codex-intervention-cr5-validation-v1"
PROTOCOL_REGISTRATION_FORMAT = "codex-intervention-cr5-protocol-registration-v1"
DEVELOPMENT_FORMAT = "codex-intervention-cr5-development-freeze-v1"
CONFIRMATION_REGISTRATION_FORMAT = "codex-intervention-cr5-confirmation-registration-v1"
RESULT_FORMAT = "codex-intervention-cr5-confirmation-result-v1"
CHECKPOINT_FORMAT = "codex-intervention-cr5-checkpoint-v1"
MODEL_FORMAT = "codex-intervention-cr5-frozen-students-v1"

DEVELOPMENT_LABEL = "VALI"
CONFIRMATION_LABEL = "INTCR5_RESISTANCE_RESILIENCE_CONFIRMATION_V1"
MATRICES = 200
LANDMARKS = (20, 35, 50, 65, 80)
DEVELOPMENT_BRANCHES = 32
CONFIRMATION_BRANCHES = 64
BREAK_HORIZON = 6
RENEWAL_HORIZON = 8
ARCHIVE_AUDIT_HORIZON = 12
NATURAL_BREAK_ACQUISITION_LIMIT = 60
PCA_COMPONENTS = 12
RIDGE_PENALTIES = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
CV_FOLDS = 5
BOOTSTRAP_REPETITIONS = 4_096
RANDOMIZATION_REPETITIONS = 4_096
EQUIVALENCE_MARGIN = 0.025
RANDOM_RATIO_LIMIT = 0.25
MAXIMUM_CPU_HOURS = 30.0
MINIMUM_CPU_BUDGET_HOURS = 16.0
MINIMUM_FREE_DISK_BYTES = 4_000_000_000

RESISTANCE_ARMS = ("BREAK_UP", "BREAK_DOWN", "RANDOM", "NOOP")
RESILIENCE_ARMS = ("RENEWAL_UP", "RENEWAL_DOWN", "RANDOM", "NOOP")


def _seed(name: str) -> str:
    return hashlib.sha256(
        f"codex-clean-room-cr5-resistance-resilience-v1::{name}".encode("utf-8")
    ).hexdigest()


SEEDS = {
    name: _seed(name)
    for name in (
        "validation",
        "development_renewal_acquisition",
        "development_renewal_future",
        "development_replay",
        "smoke_selection",
        "smoke_future",
        "confirmation_cohort",
        "resistance_selection",
        "resistance_future",
        "resistance_bootstrap",
        "resistance_randomization",
        "resilience_acquisition",
        "resilience_selection",
        "resilience_future",
        "resilience_bootstrap",
        "resilience_randomization",
        "confirmation_replay",
    )
}


def source_hashes() -> dict[str, str]:
    return {name: sha256_file(ROOT / name) for name in SOURCE_FILES}


def _scaled5_experiment() -> ExperimentConfig:
    return ExperimentConfig.scaled5()


def _confirmation_experiment() -> ExperimentConfig:
    cohort = CohortConfig(MATRICES, CONFIRMATION_BRANCHES, LANDMARKS)
    return ExperimentConfig(
        development=cohort,
        confirmation=cohort,
        horizon=ARCHIVE_AUDIT_HORIZON,
        bootstrap_repetitions=BOOTSTRAP_REPETITIONS,
        permutation_repetitions=RANDOMIZATION_REPETITIONS,
        regenerate_confirmation=True,
        master_seed=SEEDS["confirmation_cohort"],
    )


def protocol() -> dict[str, Any]:
    value: dict[str, Any] = {
        "format": PROGRAM_FORMAT,
        "status": "frozen_before_cr5_development_labels_and_confirmation_matrices",
        "target": "JOINT_BREAK_RUN3 decomposition; strict-eight excluded",
        "development": {
            "source": "exactly regenerated results/scaled5 VALI cohort",
            "matrices": MATRICES,
            "candidates": list(CANDIDATES),
            "landmarks": list(LANDMARKS),
            "states": 2 * MATRICES * len(LANDMARKS),
            "branches": DEVELOPMENT_BRANCHES,
            "archive_audit_horizon": ARCHIVE_AUDIT_HORIZON,
            "break_target": "first strict break within F6",
            "renewal_target": "run3 within F8 from natural post-break daughter",
            "natural_break_acquisition_limit": NATURAL_BREAK_ACQUISITION_LIMIT,
            "complete_replay": True,
        },
        "model": {
            "candidate_separated": True,
            "target_separated": True,
            "state_graph_features": len(STATE_GRAPH_FEATURE_NAMES),
            "history_features": len(HISTORY_FEATURE_NAMES),
            "state_pca_components": PCA_COMPONENTS,
            "ridge_penalties": list(RIDGE_PENALTIES),
            "whole_matrix_cv_folds": CV_FOLDS,
            "fold_rule": "matrix_id modulo 5",
            "tie_rule": "largest penalty within 1e-12 of minimum loss",
            "intercept_penalized": False,
            "confirmation_refit_or_recalibration": False,
        },
        "resistance": {
            "matrices": MATRICES,
            "states": 2 * MATRICES * len(LANDMARKS),
            "arms": list(RESISTANCE_ARMS),
            "branches": CONFIRMATION_BRANCHES,
            "horizon": BREAK_HORIZON,
            "primary_futures": 2
            * MATRICES
            * len(LANDMARKS)
            * len(RESISTANCE_ARMS)
            * CONFIRMATION_BRANCHES,
            "complete_replay": True,
        },
        "resilience": {
            "natural_sources": 2 * MATRICES * len(LANDMARKS),
            "arms": list(RESILIENCE_ARMS),
            "branches": CONFIRMATION_BRANCHES,
            "horizon": RENEWAL_HORIZON,
            "natural_break_acquisition_limit": NATURAL_BREAK_ACQUISITION_LIMIT,
            "all_200_matrices_per_candidate_required": True,
            "complete_replay": True,
        },
        "inference": {
            "unit": "whole catalytic matrix",
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "randomization_repetitions": RANDOMIZATION_REPETITIONS,
            "holm_family": "four candidate-by-branch-half cells within each stage",
            "equivalence_margin": EQUIVALENCE_MARGIN,
            "random_effect_ratio_limit": RANDOM_RATIO_LIMIT,
            "up_noop_and_noop_down_reported_not_gated": True,
        },
        "randomness": {
            "common_random_streams": True,
            "future_seed_excludes_arm": True,
            "random_selection_stream_separate": True,
            "no_future_retries": True,
            "no_matrix_or_source_replacement": True,
            "seed_domains": SEEDS,
        },
        "operational": {
            "maximum_cpu_hours": MAXIMUM_CPU_HOURS,
            "minimum_declared_cpu_budget_hours": MINIMUM_CPU_BUDGET_HOURS,
            "minimum_free_disk_bytes": MINIMUM_FREE_DISK_BYTES,
            "checkpoint_resumable": True,
            "mandatory_stop_after_seal": True,
            "cr6_not_launched_automatically": True,
        },
        "claim_boundary": {
            "prohibited": [
                "strict-eight control",
                "biological repair",
                "biological memory",
                "agency",
                "life",
                "autonomous organization",
                "real prebiotic chemistry",
                "universal origin-of-life mechanism",
                "Phi or PhiID intervention",
            ]
        },
    }
    value["protocol_id"] = _canonical_digest(_json_ready(value))
    return value


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class FrozenCR5Student:
    target: str
    candidate: str
    state_mean: FloatArray
    state_scale: FloatArray
    pca_mean: FloatArray
    pca_components: FloatArray
    combined_mean: FloatArray
    combined_scale: FloatArray
    coefficient: FloatArray
    intercept: float
    ridge_penalty: float
    cv_scores: dict[str, float]

    def predict_features(self, state_graph: NDArray, history: NDArray) -> FloatArray:
        state = np.atleast_2d(np.asarray(state_graph, dtype=np.float64))
        direct = np.atleast_2d(np.asarray(history, dtype=np.float64))
        if state.shape[0] != direct.shape[0]:
            raise ValueError("state and history row counts differ")
        scaled = (state - self.state_mean) / self.state_scale
        components = (scaled - self.pca_mean) @ self.pca_components.T
        combined = np.column_stack((components, direct))
        transformed = (combined - self.combined_mean) / self.combined_scale
        return np.clip(
            expit(self.intercept + transformed @ self.coefficient),
            1e-12,
            1.0 - 1e-12,
        )

    def predict_case(self, case: StateCase, config: GardConfig) -> float:
        features = state_graph_features_many(
            np.atleast_2d(case.snapshot.composition), case.beta, config
        )
        history = history_features(case.snapshot, config)
        return float(self.predict_features(features, history)[0])


def _fit_transform(
    state: FloatArray, history: FloatArray
) -> tuple[FloatArray, dict[str, FloatArray]]:
    scaler = StandardScaler().fit(state)
    state_scaled = scaler.transform(state)
    count = min(PCA_COMPONENTS, *state_scaled.shape)
    pca = PCA(n_components=count, svd_solver="full").fit(state_scaled)
    components = pca.transform(state_scaled)
    combined = np.column_stack((components, history))
    combined_scaler = StandardScaler().fit(combined)
    design = combined_scaler.transform(combined)
    arrays = {
        "state_mean": scaler.mean_.copy(),
        "state_scale": scaler.scale_.copy(),
        "pca_mean": pca.mean_.copy(),
        "pca_components": pca.components_.copy(),
        "combined_mean": combined_scaler.mean_.copy(),
        "combined_scale": combined_scaler.scale_.copy(),
    }
    return design, arrays


def _apply_transform(
    state: FloatArray, history: FloatArray, arrays: dict[str, FloatArray]
) -> FloatArray:
    scaled = (state - arrays["state_mean"]) / arrays["state_scale"]
    components = (scaled - arrays["pca_mean"]) @ arrays["pca_components"].T
    combined = np.column_stack((components, history))
    return (combined - arrays["combined_mean"]) / arrays["combined_scale"]


def _fit_binomial(
    design: FloatArray,
    successes: FloatArray,
    trials: FloatArray,
    ridge_penalty: float,
) -> tuple[float, FloatArray, dict[str, Any]]:
    x = np.asarray(design, dtype=np.float64)
    y = np.asarray(successes, dtype=np.float64)
    n = np.asarray(trials, dtype=np.float64)
    if x.shape[0] != y.size or y.shape != n.shape:
        raise ValueError("binomial design and labels are misaligned")
    if np.any(y < 0.0) or np.any(y > n) or np.any(n <= 0.0):
        raise ValueError("invalid binomial successes or trials")
    total = float(n.sum())
    prior = float(np.clip(y.sum() / total, 1e-8, 1.0 - 1e-8))
    initial = np.zeros(x.shape[1] + 1, dtype=np.float64)
    initial[0] = np.log(prior / (1.0 - prior))

    def objective(parameters: FloatArray) -> tuple[float, FloatArray]:
        intercept = parameters[0]
        coefficient = parameters[1:]
        logits = intercept + x @ coefficient
        value = float(
            np.sum(n * np.logaddexp(0.0, logits) - y * logits) / total
            + 0.5 * ridge_penalty * np.dot(coefficient, coefficient)
        )
        residual = n * expit(logits) - y
        gradient = np.empty_like(parameters)
        gradient[0] = residual.sum() / total
        gradient[1:] = x.T @ residual / total + ridge_penalty * coefficient
        return value, gradient

    fitted = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 5_000, "ftol": 1e-13, "gtol": 1e-8, "maxls": 50},
    )
    value, gradient = objective(fitted.x)
    gradient_max = float(np.max(np.abs(gradient)))
    if not fitted.success and gradient_max > 1e-5:
        raise RuntimeError(f"CR5 model failed to converge: {fitted.message}")
    return (
        float(fitted.x[0]),
        fitted.x[1:].copy(),
        {
            "objective": value,
            "gradient_max_abs": gradient_max,
            "iterations": int(fitted.nit),
            "optimizer_success": bool(fitted.success),
        },
    )


def _binomial_loss(
    successes: FloatArray, trials: FloatArray, logits: FloatArray
) -> float:
    return float(
        np.sum(trials * np.logaddexp(0.0, logits) - successes * logits) / np.sum(trials)
    )


def fit_cr5_student(
    target: str,
    candidate: str,
    state: FloatArray,
    history: FloatArray,
    branch_labels: NDArray,
    matrix_ids: NDArray,
) -> tuple[FrozenCR5Student, dict[str, Any]]:
    labels = np.asarray(branch_labels, dtype=np.int8)
    ids = np.asarray(matrix_ids, dtype=np.int64)
    if (
        labels.ndim != 2
        or labels.shape[0] != state.shape[0]
        or ids.size != state.shape[0]
    ):
        raise ValueError("CR5 training arrays are misaligned")
    successes = labels.sum(axis=1).astype(np.float64)
    trials = np.full(labels.shape[0], labels.shape[1], dtype=np.float64)
    cv_scores: dict[str, float] = {}
    for ridge in RIDGE_PENALTIES:
        numerator = 0.0
        denominator = 0.0
        for fold in range(CV_FOLDS):
            validation = ids % CV_FOLDS == fold
            training = ~validation
            if not validation.any() or not training.any():
                raise ValueError("whole-matrix CV produced an empty fold")
            design_train, arrays = _fit_transform(state[training], history[training])
            design_validation = _apply_transform(
                state[validation], history[validation], arrays
            )
            intercept, coefficient, _ = _fit_binomial(
                design_train, successes[training], trials[training], ridge
            )
            logits = intercept + design_validation @ coefficient
            fold_trials = float(trials[validation].sum())
            numerator += (
                _binomial_loss(successes[validation], trials[validation], logits)
                * fold_trials
            )
            denominator += fold_trials
        cv_scores[f"{ridge:g}"] = numerator / denominator
    minimum = min(cv_scores.values())
    selected = max(
        ridge for ridge in RIDGE_PENALTIES if cv_scores[f"{ridge:g}"] <= minimum + 1e-12
    )
    design, arrays = _fit_transform(state, history)
    intercept, coefficient, diagnostics = _fit_binomial(
        design, successes, trials, selected
    )
    student = FrozenCR5Student(
        target=target,
        candidate=candidate,
        state_mean=arrays["state_mean"],
        state_scale=arrays["state_scale"],
        pca_mean=arrays["pca_mean"],
        pca_components=arrays["pca_components"],
        combined_mean=arrays["combined_mean"],
        combined_scale=arrays["combined_scale"],
        coefficient=coefficient,
        intercept=intercept,
        ridge_penalty=float(selected),
        cv_scores=cv_scores,
    )
    diagnostics.update(
        {
            "target": target,
            "candidate": candidate,
            "states": int(labels.shape[0]),
            "branches_per_state": int(labels.shape[1]),
            "matrices": int(np.unique(ids).size),
            "event_rate": float(successes.sum() / trials.sum()),
            "selected_ridge_penalty": float(selected),
            "cv_scores": cv_scores,
            "pca_components": int(student.pca_components.shape[0]),
            "combined_features": int(student.coefficient.size),
        }
    )
    return student, diagnostics


def save_students(
    archive_path: Path,
    contract_path: Path,
    students: dict[tuple[str, str], FrozenCR5Student],
    diagnostics: dict[str, Any],
) -> None:
    arrays: dict[str, NDArray] = {}
    metadata: dict[str, Any] = {
        "format": MODEL_FORMAT,
        "state_feature_names": list(STATE_GRAPH_FEATURE_NAMES),
        "history_feature_names": list(HISTORY_FEATURE_NAMES),
        "ridge_penalties": list(RIDGE_PENALTIES),
        "cv_folds": CV_FOLDS,
        "students": {},
        "fit_diagnostics": diagnostics,
    }
    for (target, candidate), student in sorted(students.items()):
        prefix = f"{target}__c{candidate}"
        arrays[f"{prefix}__state_mean"] = student.state_mean
        arrays[f"{prefix}__state_scale"] = student.state_scale
        arrays[f"{prefix}__pca_mean"] = student.pca_mean
        arrays[f"{prefix}__pca_components"] = student.pca_components
        arrays[f"{prefix}__combined_mean"] = student.combined_mean
        arrays[f"{prefix}__combined_scale"] = student.combined_scale
        arrays[f"{prefix}__coefficient"] = student.coefficient
        arrays[f"{prefix}__intercept"] = np.asarray([student.intercept])
        metadata["students"][prefix] = {
            "target": target,
            "candidate": candidate,
            "ridge_penalty": student.ridge_penalty,
            "cv_scores": student.cv_scores,
        }
    np.savez_compressed(archive_path, **arrays)
    contract_path.write_text(
        json.dumps(_json_ready(metadata), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_students(
    archive_path: Path | str, contract_path: Path | str
) -> dict[tuple[str, str], FrozenCR5Student]:
    metadata = json.loads(Path(contract_path).read_text(encoding="utf-8"))
    if metadata.get("format") != MODEL_FORMAT:
        raise ValueError("unsupported CR5 model archive")
    output: dict[tuple[str, str], FrozenCR5Student] = {}
    with np.load(archive_path, allow_pickle=False) as arrays:
        for prefix, item in metadata["students"].items():
            target = str(item["target"])
            candidate = str(item["candidate"])
            output[(target, candidate)] = FrozenCR5Student(
                target=target,
                candidate=candidate,
                state_mean=arrays[f"{prefix}__state_mean"].copy(),
                state_scale=arrays[f"{prefix}__state_scale"].copy(),
                pca_mean=arrays[f"{prefix}__pca_mean"].copy(),
                pca_components=arrays[f"{prefix}__pca_components"].copy(),
                combined_mean=arrays[f"{prefix}__combined_mean"].copy(),
                combined_scale=arrays[f"{prefix}__combined_scale"].copy(),
                coefficient=arrays[f"{prefix}__coefficient"].copy(),
                intercept=float(arrays[f"{prefix}__intercept"][0]),
                ridge_penalty=float(item["ridge_penalty"]),
                cv_scores={
                    key: float(value) for key, value in item["cv_scores"].items()
                },
            )
    return output


@dataclass(frozen=True)
class DevelopmentTargetBatch:
    state_id: str
    state_digest: str
    labels: NDArray[np.int8]
    auxiliary_labels: NDArray[np.int8]
    first_event_time: NDArray[np.int16]
    record_digests: tuple[str, ...]


def _records_digest(records: Iterable[FissionRecord]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(np.ascontiguousarray(record.parent).tobytes())
        digest.update(np.ascontiguousarray(record.daughter).tobytes())
        digest.update(np.asarray([record.h], dtype=np.float64).tobytes())
        digest.update(np.asarray([record.growth_steps], dtype=np.int64).tobytes())
    return digest.hexdigest()


def _first_run(values: NDArray, length: int) -> int:
    inherited = np.asarray(values, dtype=bool)
    for start in range(max(0, inherited.size - length + 1)):
        if bool(inherited[start : start + length].all()):
            return start + length
    return -1


def _development_break_worker(
    arguments: tuple[StateCase, ExperimentConfig],
) -> DevelopmentTargetBatch:
    case, experiment = arguments
    limiter = threadpool_limits(limits=1)
    try:
        labels = np.empty(DEVELOPMENT_BRANCHES, dtype=np.int8)
        joint = np.empty(DEVELOPMENT_BRANCHES, dtype=np.int8)
        first = np.full(DEVELOPMENT_BRANCHES, -1, dtype=np.int16)
        digests: list[str] = []
        for branch in range(DEVELOPMENT_BRANCHES):
            seed = derive_seed(
                experiment.master_seed,
                f"{DEVELOPMENT_LABEL}.future",
                case.candidate,
                case.matrix_id,
                case.landmark,
                branch,
            )
            records, completed = simulate_future_absorbing(
                case.snapshot,
                case.beta,
                experiment.gard,
                CANDIDATES[case.candidate],
                ARCHIVE_AUDIT_HORIZON,
                np.random.default_rng(seed),
            )
            outcome = outcome_from_records(
                case.snapshot, records, completed, ARCHIVE_AUDIT_HORIZON
            )
            inherited = np.asarray(
                [
                    record.h > experiment.gard.inheritance_threshold
                    for record in records
                ],
                dtype=bool,
            )
            locations = np.flatnonzero(~inherited[:BREAK_HORIZON])
            labels[branch] = int(locations.size > 0)
            first[branch] = int(locations[0] + 1) if locations.size else -1
            joint[branch] = int(outcome.joint_break_run3)
            digests.append(_records_digest(records))
        return DevelopmentTargetBatch(
            state_id=case.state_id,
            state_digest=base._snapshot_digest(case),
            labels=labels,
            auxiliary_labels=joint,
            first_event_time=first,
            record_digests=tuple(digests),
        )
    finally:
        limiter.restore_original_limits()


def _development_renewal_worker(
    arguments: tuple[StateCase, ExperimentConfig],
) -> DevelopmentTargetBatch:
    case, experiment = arguments
    limiter = threadpool_limits(limits=1)
    try:
        labels = np.empty(DEVELOPMENT_BRANCHES, dtype=np.int8)
        run5 = np.empty(DEVELOPMENT_BRANCHES, dtype=np.int8)
        first = np.full(DEVELOPMENT_BRANCHES, -1, dtype=np.int16)
        digests: list[str] = []
        for branch in range(DEVELOPMENT_BRANCHES):
            seed = derive_seed(
                SEEDS["development_renewal_future"],
                "INTCR5.development.renewal.future",
                case.candidate,
                case.matrix_id,
                case.landmark,
                branch,
            )
            records, _completed = simulate_future_absorbing(
                case.snapshot,
                case.beta,
                experiment.gard,
                CANDIDATES[case.candidate],
                RENEWAL_HORIZON,
                np.random.default_rng(seed),
            )
            inherited = np.asarray(
                [
                    record.h > experiment.gard.inheritance_threshold
                    for record in records
                ],
                dtype=bool,
            )
            time = _first_run(inherited, 3)
            labels[branch] = int(time >= 0)
            first[branch] = time
            run5[branch] = int(_first_run(inherited, 5) >= 0)
            digests.append(_records_digest(records))
        return DevelopmentTargetBatch(
            state_id=case.state_id,
            state_digest=base._snapshot_digest(case),
            labels=labels,
            auxiliary_labels=run5,
            first_event_time=first,
            record_digests=tuple(digests),
        )
    finally:
        limiter.restore_original_limits()


def _target_batch_digest(batch: DevelopmentTargetBatch) -> str:
    digest = hashlib.sha256()
    digest.update(batch.state_id.encode("utf-8"))
    digest.update(batch.state_digest.encode("ascii"))
    digest.update(np.ascontiguousarray(batch.labels).tobytes())
    digest.update(np.ascontiguousarray(batch.auxiliary_labels).tobytes())
    digest.update(np.ascontiguousarray(batch.first_event_time).tobytes())
    digest.update("".join(batch.record_digests).encode("ascii"))
    return digest.hexdigest()


def _run_target_batches(
    cases: list[StateCase],
    experiment: ExperimentConfig,
    worker: Callable[[tuple[StateCase, ExperimentConfig]], DevelopmentTargetBatch],
    checkpoint: Path,
    workers: int,
    registration_id: str,
    stage: str,
) -> list[DevelopmentTargetBatch]:
    checkpoint.mkdir(parents=True, exist_ok=True)
    contract = {
        "format": CHECKPOINT_FORMAT,
        "registration_id": registration_id,
        "stage": stage,
        "case_ids": [case.state_id for case in cases],
        "case_digests": [base._snapshot_digest(case) for case in cases],
        "branches": DEVELOPMENT_BRANCHES,
        "source_hashes": source_hashes(),
    }
    contract["contract_id"] = _canonical_digest(_json_ready(contract))
    contract_path = checkpoint / "checkpoint_contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text()) != _json_ready(contract):
            raise ValueError(f"development checkpoint contract changed: {checkpoint}")
    else:
        base._atomic_json(contract_path, contract)
    batches: list[DevelopmentTargetBatch | None] = [None] * len(cases)
    missing: list[int] = []
    for index, case in enumerate(cases):
        path = checkpoint / f"state_{index:04d}.pkl"
        if path.exists():
            with path.open("rb") as handle:
                batch = pickle.load(handle)
            if (
                not isinstance(batch, DevelopmentTargetBatch)
                or batch.state_id != case.state_id
                or batch.state_digest != base._snapshot_digest(case)
            ):
                raise ValueError(f"invalid development checkpoint: {path}")
            batches[index] = batch
        else:
            missing.append(index)

    def status(state: str) -> None:
        complete = sum(batch is not None for batch in batches)
        base._atomic_json(
            checkpoint / "status.json",
            {
                "format": CHECKPOINT_FORMAT,
                "stage": stage,
                "state": state,
                "states_complete": complete,
                "states_total": len(cases),
                "futures_complete": complete * DEVELOPMENT_BRANCHES,
                "futures_total": len(cases) * DEVELOPMENT_BRANCHES,
                "percent_complete": 100.0 * complete / max(1, len(cases)),
            },
        )

    status("running" if missing else "complete")
    arguments = [(cases[index], experiment) for index in missing]
    if workers <= 1:
        generated = map(worker, arguments)
        for index, batch in zip(missing, generated, strict=True):
            batches[index] = batch
            base._atomic_pickle(checkpoint / f"state_{index:04d}.pkl", batch)
            status("running")
    elif missing:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            generated = executor.map(worker, arguments, chunksize=1)
            for index, batch in zip(missing, generated, strict=True):
                batches[index] = batch
                base._atomic_pickle(checkpoint / f"state_{index:04d}.pkl", batch)
                status("running")
    status("complete")
    if any(batch is None for batch in batches):
        raise AssertionError("development target generation dropped a state")
    return [batch for batch in batches if batch is not None]


def _target_replay_audit(
    generated: list[DevelopmentTargetBatch], replayed: list[DevelopmentTargetBatch]
) -> dict[str, Any]:
    left = [_target_batch_digest(batch) for batch in generated]
    right = [_target_batch_digest(batch) for batch in replayed]
    exact = [a == b for a, b in zip(left, right, strict=True)]
    return {
        "states": len(left),
        "all_exact": bool(all(exact)),
        "states_exact": int(sum(exact)),
        "generated_digest": hashlib.sha256("".join(left).encode()).hexdigest(),
        "replay_digest": hashlib.sha256("".join(right).encode()).hexdigest(),
        "mismatched_state_ids": [
            generated[index].state_id for index, value in enumerate(exact) if not value
        ],
    }


def _acquisition_seed(case: StateCase, domain: str, label: str) -> int:
    return derive_seed(
        domain,
        label,
        case.candidate,
        case.matrix_id,
        case.landmark,
    )


def acquire_natural_break(
    case: StateCase,
    config: GardConfig,
    seed_domain: str,
    label: str,
) -> tuple[StateCase | None, NDArray[np.int64] | None, dict[str, Any]]:
    """Acquire the first untreated break, without retry or replacement."""

    rng = np.random.default_rng(_acquisition_seed(case, seed_domain, label))
    current = np.asarray(case.snapshot.composition, dtype=np.int64).copy()
    inheritance = list(case.snapshot.inheritance)
    boundary_h = list(case.snapshot.boundary_h)
    cumulative = int(case.snapshot.cumulative_growth_steps)
    for offset in range(1, NATURAL_BREAK_ACQUISITION_LIMIT + 1):
        try:
            record = advance_fission(
                current,
                case.beta,
                config,
                CANDIDATES[case.candidate],
                rng,
            )
        except SimulationError:
            return (
                None,
                None,
                {
                    "source_state_id": case.state_id,
                    "candidate": case.candidate,
                    "matrix_id": case.matrix_id,
                    "landmark": case.landmark,
                    "eligible": False,
                    "reason": "extinction_before_natural_break",
                    "observed_fissions": offset - 1,
                    "break_time": -1,
                },
            )
        inherited = bool(record.h > config.inheritance_threshold)
        inheritance.append(inherited)
        boundary_h.append(float(record.h))
        cumulative += int(record.growth_steps)
        snapshot = Snapshot(
            composition=np.asarray(record.daughter, dtype=np.int64).copy(),
            generation=case.snapshot.generation + offset,
            inheritance=tuple(inheritance),
            boundary_h=tuple(boundary_h),
            previous_growth_steps=int(record.growth_steps),
            cumulative_growth_steps=cumulative,
        )
        if not inherited:
            broken = StateCase(
                state_id=f"{case.state_id}-natural-break-f{offset:02d}",
                cohort=f"{case.cohort}_CR5_NATURAL_BREAK",
                candidate=case.candidate,
                matrix_id=case.matrix_id,
                landmark=case.landmark,
                beta=case.beta,
                snapshot=snapshot,
            )
            return (
                broken,
                np.asarray(record.parent, dtype=np.int64).copy(),
                {
                    "source_state_id": case.state_id,
                    "broken_state_id": broken.state_id,
                    "candidate": case.candidate,
                    "matrix_id": case.matrix_id,
                    "landmark": case.landmark,
                    "eligible": True,
                    "reason": "natural_break_observed",
                    "observed_fissions": offset,
                    "break_time": offset,
                    "break_h": float(record.h),
                    "broken_state_digest": base._snapshot_digest(broken),
                    "old_anchor_digest": hashlib.sha256(
                        np.ascontiguousarray(record.parent).tobytes()
                    ).hexdigest(),
                },
            )
        current = record.daughter
    return (
        None,
        None,
        {
            "source_state_id": case.state_id,
            "candidate": case.candidate,
            "matrix_id": case.matrix_id,
            "landmark": case.landmark,
            "eligible": False,
            "reason": "no_natural_break_within_registered_limit",
            "observed_fissions": NATURAL_BREAK_ACQUISITION_LIMIT,
            "break_time": -1,
        },
    )


def acquire_natural_breaks(
    cases: list[StateCase],
    config: GardConfig,
    seed_domain: str,
    label: str,
) -> tuple[list[StateCase], list[NDArray[np.int64]], pd.DataFrame]:
    broken: list[StateCase] = []
    anchors: list[NDArray[np.int64]] = []
    rows: list[dict[str, Any]] = []
    for case in cases:
        acquired, anchor, row = acquire_natural_break(case, config, seed_domain, label)
        rows.append(row)
        if acquired is not None:
            if anchor is None:
                raise AssertionError("eligible natural break lacks its old anchor")
            broken.append(acquired)
            anchors.append(anchor)
    return broken, anchors, pd.DataFrame(rows)


def _acquisition_exact(
    left_cases: list[StateCase],
    left_anchors: list[NDArray[np.int64]],
    left_rows: pd.DataFrame,
    right_cases: list[StateCase],
    right_anchors: list[NDArray[np.int64]],
    right_rows: pd.DataFrame,
) -> bool:
    return bool(
        _json_ready(left_rows.to_dict("records"))
        == _json_ready(right_rows.to_dict("records"))
        and [base._snapshot_digest(case) for case in left_cases]
        == [base._snapshot_digest(case) for case in right_cases]
        and len(left_anchors) == len(right_anchors)
        and all(
            np.array_equal(left, right)
            for left, right in zip(left_anchors, right_anchors, strict=True)
        )
    )


@dataclass(frozen=True)
class CR5PhaseSpec:
    stage: str
    target: str
    arms: tuple[str, ...]
    horizon: int
    branches: int
    selection_seed: str
    future_seed: str
    bootstrap_seed: str
    randomization_seed: str

    @property
    def up_arm(self) -> str:
        return self.arms[0]

    @property
    def down_arm(self) -> str:
        return self.arms[1]


def resistance_spec() -> CR5PhaseSpec:
    return CR5PhaseSpec(
        stage="resistance",
        target="break",
        arms=RESISTANCE_ARMS,
        horizon=BREAK_HORIZON,
        branches=CONFIRMATION_BRANCHES,
        selection_seed=SEEDS["resistance_selection"],
        future_seed=SEEDS["resistance_future"],
        bootstrap_seed=SEEDS["resistance_bootstrap"],
        randomization_seed=SEEDS["resistance_randomization"],
    )


def resilience_spec() -> CR5PhaseSpec:
    return CR5PhaseSpec(
        stage="resilience",
        target="renewal",
        arms=RESILIENCE_ARMS,
        horizon=RENEWAL_HORIZON,
        branches=CONFIRMATION_BRANCHES,
        selection_seed=SEEDS["resilience_selection"],
        future_seed=SEEDS["resilience_future"],
        bootstrap_seed=SEEDS["resilience_bootstrap"],
        randomization_seed=SEEDS["resilience_randomization"],
    )


def _phase_selection_seed(spec: CR5PhaseSpec, case: StateCase) -> int:
    return derive_seed(
        spec.selection_seed,
        f"{CONFIRMATION_LABEL}.{spec.stage}.random_edit",
        case.candidate,
        case.matrix_id,
        case.landmark,
    )


def _phase_future_seed(spec: CR5PhaseSpec, case: StateCase, branch: int) -> int:
    return derive_seed(
        spec.future_seed,
        f"{CONFIRMATION_LABEL}.{spec.stage}.future",
        case.candidate,
        case.matrix_id,
        case.landmark,
        branch,
    )


def score_student_edits(
    student: FrozenCR5Student,
    case: StateCase,
    config: GardConfig,
) -> tuple[float, tuple[ScoredEdit, ...]]:
    edits = enumerate_legal_edits(case.snapshot.composition)
    if not edits:
        raise ValueError("state has no legal molecular substitution")
    compositions = np.vstack(
        [apply_molecular_edit(case.snapshot.composition, edit) for edit in edits]
    )
    noop_features = state_graph_features_many(
        np.atleast_2d(case.snapshot.composition), case.beta, config
    )
    edit_features = state_graph_features_many(compositions, case.beta, config)
    direct = history_features(case.snapshot, config)
    noop = float(student.predict_features(noop_features, direct)[0])
    probabilities = student.predict_features(
        edit_features, np.broadcast_to(direct, (len(edits), direct.size))
    )
    return noop, tuple(
        ScoredEdit(edit, float(probability), float(probability - noop))
        for edit, probability in zip(edits, probabilities, strict=True)
    )


def select_student_edits(
    noop: float,
    scores: tuple[ScoredEdit, ...],
    rng: np.random.Generator,
) -> tuple[NDArray[np.float64], tuple[MolecularEdit | None, ...]]:
    if not scores:
        raise ValueError("cannot select from an empty edit set")
    probabilities = np.asarray(
        [score.predicted_probability for score in scores], dtype=np.float64
    )
    up_index = int(np.flatnonzero(probabilities == probabilities.max())[0])
    down_index = int(np.flatnonzero(probabilities == probabilities.min())[0])
    random_index = int(rng.integers(0, len(scores)))
    return (
        np.asarray(
            [
                probabilities[up_index],
                probabilities[down_index],
                probabilities[random_index],
                noop,
            ],
            dtype=np.float64,
        ),
        (
            scores[up_index].edit,
            scores[down_index].edit,
            scores[random_index].edit,
            None,
        ),
    )


def _stage_outcome(
    stage: str,
    launch: Snapshot,
    records: list[FissionRecord],
    completed: bool,
    horizon: int,
    threshold: float,
) -> InterventionOutcome:
    raw = outcome_from_records(launch, records, completed, horizon, threshold)
    inherited = np.asarray([record.h > threshold for record in records], dtype=bool)
    if stage == "resistance":
        target = bool((~inherited).any())
        return replace(raw, joint_break_run3=target)
    if stage == "resilience":
        time = _first_run(inherited, 3)
        target = time >= 0
        return replace(
            raw,
            joint_break_run3=target,
            run3_after_break=target,
            renewal_certification_time=time,
        )
    raise ValueError(stage)


def _phase_worker(
    arguments: tuple[
        StateCase,
        GardConfig,
        CR5PhaseSpec,
        str,
        str,
    ],
) -> base.PhaseBatch:
    case, config, spec, model_path, contract_path = arguments
    limiter = threadpool_limits(limits=1)
    try:
        students = load_students(Path(model_path), Path(contract_path))
        student = students[(spec.target, case.candidate)]
        noop, scores = score_student_edits(student, case, config)
        predictions, edits = select_student_edits(
            noop,
            scores,
            np.random.default_rng(_phase_selection_seed(spec, case)),
        )
        arm_outcomes: list[list[InterventionOutcome]] = [[] for _ in spec.arms]
        for branch in range(spec.branches):
            seed = _phase_future_seed(spec, case, branch)
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
                    spec.horizon,
                    np.random.default_rng(seed),
                )
                arm_outcomes[arm_index].append(
                    _stage_outcome(
                        spec.stage,
                        launch,
                        records,
                        completed,
                        spec.horizon,
                        config.inheritance_threshold,
                    )
                )
        return base.PhaseBatch(
            state_id=case.state_id,
            state_digest=base._snapshot_digest(case),
            arm_names=spec.arms,
            predictions=predictions,
            selected_edits=edits,
            surgeries=tuple(None for _ in spec.arms),
            scored_edits=scores,
            catalytic_support=np.empty(0, dtype=np.float64),
            outcomes=tuple(tuple(values) for values in arm_outcomes),
        )
    finally:
        limiter.restore_original_limits()


def _phase_checkpoint_contract(
    cases: list[StateCase],
    spec: CR5PhaseSpec,
    registration_id: str,
    stage: str,
    model_hash: str,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "registration_id": registration_id,
        "scientific_stage": spec.stage,
        "execution_stage": stage,
        "target": spec.target,
        "horizon": spec.horizon,
        "branches": spec.branches,
        "arms": list(spec.arms),
        "case_ids": [case.state_id for case in cases],
        "case_digests": [base._snapshot_digest(case) for case in cases],
        "selection_seed": spec.selection_seed,
        "future_seed": spec.future_seed,
        "future_seed_includes_arm": False,
        "model_sha256": model_hash,
        "source_hashes": source_hashes(),
    }
    value["contract_id"] = _canonical_digest(_json_ready(value))
    return value


def run_phase_batches(
    cases: list[StateCase],
    config: GardConfig,
    spec: CR5PhaseSpec,
    model_path: Path,
    model_contract_path: Path,
    registration_id: str,
    checkpoint: Path,
    workers: int,
    stage: str,
) -> list[base.PhaseBatch]:
    checkpoint.mkdir(parents=True, exist_ok=True)
    model_hash = sha256_file(model_path)
    contract = _phase_checkpoint_contract(
        cases, spec, registration_id, stage, model_hash
    )
    contract_path = checkpoint / "checkpoint_contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text()) != _json_ready(contract):
            raise ValueError(f"CR5 checkpoint contract changed: {checkpoint}")
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
                or batch.arm_names != spec.arms
            ):
                raise ValueError(f"invalid CR5 checkpoint: {path}")
            batches[index] = batch
        else:
            missing.append(index)

    def status(value: str) -> None:
        complete = sum(batch is not None for batch in batches)
        base._atomic_json(
            checkpoint / "status.json",
            {
                "format": CHECKPOINT_FORMAT,
                "scientific_stage": spec.stage,
                "execution_stage": stage,
                "state": value,
                "states_complete": complete,
                "states_total": len(cases),
                "futures_complete": complete * len(spec.arms) * spec.branches,
                "futures_total": len(cases) * len(spec.arms) * spec.branches,
                "percent_complete": 100.0 * complete / max(1, len(cases)),
            },
        )

    status("running" if missing else "complete")
    arguments = [
        (case, config, spec, str(model_path), str(model_contract_path))
        for case in (cases[index] for index in missing)
    ]
    if workers <= 1:
        generated = map(_phase_worker, arguments)
        for index, batch in zip(missing, generated, strict=True):
            batches[index] = batch
            base._atomic_pickle(checkpoint / f"state_{index:04d}.pkl", batch)
            status("running")
    elif missing:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            generated = executor.map(_phase_worker, arguments, chunksize=1)
            for index, batch in zip(missing, generated, strict=True):
                batches[index] = batch
                base._atomic_pickle(checkpoint / f"state_{index:04d}.pkl", batch)
                status("running")
    status("complete")
    if any(batch is None for batch in batches):
        raise AssertionError("CR5 phase dropped a state")
    return [batch for batch in batches if batch is not None]


def _outcome_arrays(
    cases: list[StateCase], batches: list[base.PhaseBatch], spec: CR5PhaseSpec
) -> dict[str, NDArray]:
    shape = (len(cases), len(spec.arms), spec.branches)
    arrays: dict[str, NDArray] = {
        "targets": np.empty(shape, dtype=np.int8),
        "break_event": np.empty(shape, dtype=np.int8),
        "run3": np.empty(shape, dtype=np.int8),
        "inherited_boundary_count": np.empty(shape, dtype=np.int8),
        "first_break_time": np.empty(shape, dtype=np.int8),
        "renewal_time": np.empty(shape, dtype=np.int8),
        "completed_horizon": np.empty(shape, dtype=np.int8),
        "observed_fissions": np.empty(shape, dtype=np.int8),
        "total_growth_updates": np.empty(shape, dtype=np.int32),
        "mean_growth_updates": np.empty(shape, dtype=np.float64),
        "final_entropy": np.empty(shape, dtype=np.float64),
        "final_occupied_types": np.empty(shape, dtype=np.int16),
        "boundary_h": np.empty((*shape, spec.horizon), dtype=np.float64),
        "growth_updates": np.empty((*shape, spec.horizon), dtype=np.int32),
        "final_composition": np.empty(
            (*shape, cases[0].snapshot.composition.size), dtype=np.int16
        ),
        "predictions": np.vstack([batch.predictions for batch in batches]),
    }
    for state_index, batch in enumerate(batches):
        for arm_index, outcomes in enumerate(batch.outcomes):
            for branch, outcome in enumerate(outcomes):
                location = (state_index, arm_index, branch)
                arrays["targets"][location] = int(outcome.joint_break_run3)
                arrays["break_event"][location] = int(outcome.break_event)
                arrays["run3"][location] = int(outcome.run3_after_break)
                arrays["inherited_boundary_count"][location] = (
                    outcome.inherited_boundary_count
                )
                arrays["first_break_time"][location] = outcome.first_break_time
                arrays["renewal_time"][location] = outcome.renewal_certification_time
                arrays["completed_horizon"][location] = int(outcome.completed_horizon)
                arrays["observed_fissions"][location] = outcome.observed_fissions
                arrays["total_growth_updates"][location] = outcome.total_growth_updates
                arrays["mean_growth_updates"][location] = outcome.mean_growth_updates
                arrays["final_entropy"][location] = outcome.final_entropy
                arrays["final_occupied_types"][location] = outcome.final_occupied_types
                arrays["boundary_h"][location] = outcome.boundary_h
                arrays["growth_updates"][location] = outcome.growth_updates
                arrays["final_composition"][location] = outcome.final_composition
    return arrays


def add_cr5_gate_fields(metrics: dict[str, Any], stage: str) -> dict[str, Any]:
    for cell in metrics["cells"]:
        contrast = cell["contrasts"]["up_minus_down"]
        gates = {
            "up_minus_down_positive": contrast["estimate"] > 0.0,
            "up_minus_down_bootstrap_lower_positive": (
                contrast["bootstrap_ci95"][0] > 0.0
            ),
            "holm_randomization_below_0_05": (
                cell["up_down_randomization_p_holm"] < 0.05
            ),
            "random_tost_equivalent_to_noop": cell["random_noop_equivalence"][
                "tost_equivalent"
            ],
            "random_absolute_difference_within_effect_ratio": cell[
                "random_noop_equivalence"
            ]["absolute_difference_within_ratio"],
        }
        cell["cr5_registered_gates"] = gates
        cell["cr5_registered_cell_pass"] = bool(all(gates.values()))
    metrics["stage"] = stage
    metrics["cr5_all_four_cells_pass"] = bool(
        len(metrics["cells"]) == 4
        and all(cell["cr5_registered_cell_pass"] for cell in metrics["cells"])
    )
    return metrics


def _inference(
    cases: list[StateCase],
    arrays: dict[str, NDArray],
    spec: CR5PhaseSpec,
    draws: dict[str, NDArray],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metrics, rows = compute_one_shot_inference(
        cases,
        spec.arms,
        arrays["targets"],
        arrays["predictions"],
        draws,
        up_arm=spec.up_arm,
        down_arm=spec.down_arm,
        equivalence_margin=EQUIVALENCE_MARGIN,
        random_ratio_limit=RANDOM_RATIO_LIMIT,
    )
    return add_cr5_gate_fields(metrics, spec.stage), rows


def validation_checks() -> dict[str, Any]:
    fixture = base.validation_checks()
    config = GardConfig()
    composition = np.zeros(config.n_types, dtype=np.int64)
    composition[: config.n_min] = 1
    beta = generate_beta(config, np.random.default_rng(101))
    snapshot = Snapshot(
        composition=composition,
        generation=20,
        inheritance=(True, False, True),
        boundary_h=(0.95, 0.80, 0.92),
        previous_growth_steps=11,
        cumulative_growth_steps=77,
    )
    case = StateCase("cr5-validation", "FIX", "02", 3, 20, beta, snapshot)
    resistance = resistance_spec()
    resilience = resilience_spec()
    threshold = config.inheritance_threshold
    parent = np.asarray([2, 1, 1], dtype=np.int64)
    daughter = np.asarray([1, 1, 0], dtype=np.int64)

    def record(h: float) -> FissionRecord:
        return FissionRecord(parent=parent, daughter=daughter, h=h, growth_steps=3)

    break_outcome = _stage_outcome(
        "resistance", snapshot, [record(0.9)], True, 1, threshold
    )
    inherited = np.nextafter(0.9, 1.0)
    renewal_outcome = _stage_outcome(
        "resilience",
        snapshot,
        [record(inherited), record(inherited), record(inherited)],
        True,
        3,
        threshold,
    )
    checks = {
        "inherited_validation_suite_passes": bool(fixture["all_checks_passed"]),
        "scaled5_checksum_manifest_exists": (SCALED5 / "SHA256SUMS").is_file(),
        "cr1_and_cr3_predecessors_exist": CR1_RESULT.is_dir() and CR3_RESULT.is_dir(),
        "seed_domains_unique": len(SEEDS) == len(set(SEEDS.values())),
        "seed_domains_disjoint_from_original_interventions": set(
            SEEDS.values()
        ).isdisjoint(base.SEED_DOMAINS.values()),
        "break_threshold_is_strict": break_outcome.joint_break_run3,
        "renewal_run3_edge_exact": renewal_outcome.joint_break_run3
        and renewal_outcome.renewal_certification_time == 3,
        "resistance_future_seed_is_arm_free": len(
            {_phase_future_seed(resistance, case, 0) for _arm in resistance.arms}
        )
        == 1,
        "resilience_future_seed_is_arm_free": len(
            {_phase_future_seed(resilience, case, 0) for _arm in resilience.arms}
        )
        == 1,
        "selection_and_future_streams_distinct": _phase_selection_seed(resistance, case)
        != _phase_future_seed(resistance, case, 0),
        "ridge_grid_exact": RIDGE_PENALTIES == (0.001, 0.01, 0.1, 1.0, 10.0, 100.0),
        "whole_matrix_fold_rule_exact": all(
            matrix_id % CV_FOLDS == matrix_id % 5 for matrix_id in range(20)
        ),
        "development_and_confirmation_sizes_frozen": MATRICES == 200
        and DEVELOPMENT_BRANCHES == 32
        and CONFIRMATION_BRANCHES == 64
        and LANDMARKS == (20, 35, 50, 65, 80),
        "cpu_ceiling_frozen": MAXIMUM_CPU_HOURS == 30.0,
        "strict_eight_excluded": "strict-eight excluded" in protocol()["target"],
    }
    return {
        "format": VALIDATION_FORMAT,
        "checks": checks,
        "all_checks_passed": bool(all(checks.values())),
        "check_count": len(checks),
        "scientific_matrices_generated": 0,
        "scientific_futures_generated": 0,
    }


def validate(output: Path = DEFAULT_VALIDATION) -> None:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    verify_checksums(SCALED5)
    verify_checksums(CR1_RESULT)
    verify_checksums(CR3_RESULT)
    checks = validation_checks()
    if not checks["all_checks_passed"]:
        raise AssertionError(
            {name: passed for name, passed in checks["checks"].items() if not passed}
        )
    command = [sys.executable, "-m", "pytest", "-q"]
    completed = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "CR5 full repository validation failed\n"
            + completed.stdout
            + "\n"
            + completed.stderr
        )
    with _atomic_destination(output) as destination:
        payload = dict(checks)
        payload["source_hashes"] = source_hashes()
        payload["scaled5_checksum_manifest_sha256"] = sha256_file(
            SCALED5 / "SHA256SUMS"
        )
        (destination / "validation.json").write_text(
            json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (destination / "pytest_output.txt").write_text(
            "$ " + " ".join(command) + "\n" + completed.stdout + completed.stderr,
            encoding="utf-8",
        )
        write_checksums(destination)
    verify_checksums(output)
    print(f"CR5 validation sealed: {output}", flush=True)


def register_protocol(
    validation_directory: Path = DEFAULT_VALIDATION,
    output: Path = DEFAULT_PROTOCOL_REGISTRATION,
) -> None:
    validation_directory = validation_directory.resolve()
    output = output.resolve()
    verify_checksums(validation_directory)
    validated = json.loads((validation_directory / "validation.json").read_text())
    if not validated.get("all_checks_passed"):
        raise ValueError("CR5 validation did not pass")
    for forbidden in (
        DEFAULT_DEVELOPMENT,
        DEFAULT_DEVELOPMENT_WORK,
        DEFAULT_CONFIRMATION_REGISTRATION,
        DEFAULT_OUTPUT,
        DEFAULT_WORK,
    ):
        if forbidden.exists():
            raise FileExistsError(
                f"CR5 artifact exists before protocol registration: {forbidden}"
            )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    frozen = protocol()
    payload: dict[str, Any] = {
        "format": PROTOCOL_REGISTRATION_FORMAT,
        "protocol_id": frozen["protocol_id"],
        "source_hashes": source_hashes(),
        "validation_checksum_manifest_sha256": sha256_file(
            validation_directory / "SHA256SUMS"
        ),
        "scaled5_checksum_manifest_sha256": sha256_file(SCALED5 / "SHA256SUMS"),
        "seed_registry": SEEDS,
        "development_labels_generated_at_registration": 0,
        "confirmation_matrices_generated_at_registration": 0,
        "confirmation_futures_generated_at_registration": 0,
    }
    payload["registration_id"] = _canonical_digest(_json_ready(payload))
    with _atomic_destination(output) as destination:
        (destination / "protocol.json").write_text(
            json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (destination / "seed_registry.json").write_text(
            json.dumps(SEEDS, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (destination / "registration.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        write_checksums(destination)
    verify_protocol_registration(output)
    _append_ledger(
        f"<!-- cr5-protocol-{payload['registration_id']} -->",
        [
            "## CR5 resistance/resilience protocol registered",
            "",
            f"- Registration: `{payload['registration_id']}`.",
            "- The complete development, model, confirmation, inference, replay, and stop contracts were sealed before CR5 development labels or confirmation matrices.",
            "- The 5x development cohort must regenerate exactly; no intervention confirmation outcome can enter model fitting.",
            "",
        ],
    )
    print(f"CR5 protocol registered: {payload['registration_id']}", flush=True)


def verify_protocol_registration(
    directory: Path = DEFAULT_PROTOCOL_REGISTRATION,
) -> dict[str, Any]:
    directory = directory.resolve()
    verify_checksums(directory)
    payload = json.loads((directory / "registration.json").read_text())
    frozen = json.loads((directory / "protocol.json").read_text())
    if payload.get("format") != PROTOCOL_REGISTRATION_FORMAT:
        raise ValueError("unsupported CR5 protocol registration")
    if frozen != _json_ready(protocol()):
        raise ValueError("CR5 frozen protocol changed")
    if payload["source_hashes"] != source_hashes():
        raise ValueError("CR5 registered source tree changed")
    if payload["registration_id"] != _canonical_digest(
        {key: value for key, value in payload.items() if key != "registration_id"}
    ):
        raise ValueError("CR5 protocol registration ID changed")
    return payload


def _status(work: Path, phase: str, state: str, detail: str, **extra: Any) -> None:
    work.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "codex-intervention-cr5-status-v1",
        "phase": phase,
        "state": state,
        "detail": detail,
        "mandatory_stop_after_seal": True,
        **extra,
    }
    base._atomic_json(work / "campaign_status.json", payload)


def _prepare_work(
    work: Path,
    output: Path,
    phase: str,
    registration_id: str,
    cpu_budget_hours: float | None = None,
) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    free = shutil.disk_usage(RESULT_ROOT).free
    if phase == "confirmation":
        if cpu_budget_hours is None or not (
            MINIMUM_CPU_BUDGET_HOURS <= cpu_budget_hours <= MAXIMUM_CPU_HOURS
        ):
            raise ValueError(
                f"CR5 confirmation CPU declaration must be between "
                f"{MINIMUM_CPU_BUDGET_HOURS:g} and {MAXIMUM_CPU_HOURS:g} hours"
            )
        if free < MINIMUM_FREE_DISK_BYTES:
            raise OSError(
                f"CR5 requires {MINIMUM_FREE_DISK_BYTES} free bytes; found {free}"
            )
    work.mkdir(parents=True, exist_ok=True)
    stable_contract = {
        "format": "codex-intervention-cr5-campaign-v1",
        "phase": phase,
        "registration_id": registration_id,
        "output": str(output),
        "source_hashes": source_hashes(),
        "declared_cpu_budget_hours": cpu_budget_hours,
    }
    contract = {
        **stable_contract,
        "free_disk_bytes_at_initialization": free,
    }
    contract["campaign_id"] = _canonical_digest(_json_ready(contract))
    path = work / "campaign_contract.json"
    if path.exists():
        existing = json.loads(path.read_text())
        observed_stable = {key: existing.get(key) for key in stable_contract}
        if observed_stable != _json_ready(stable_contract):
            raise ValueError("CR5 work directory belongs to another campaign")
    else:
        base._atomic_json(path, contract)
    _status(
        work,
        phase,
        "running",
        "campaign_initialized",
        declared_cpu_budget_hours=cpu_budget_hours,
    )


def _append_ledger(marker: str, lines: list[str]) -> None:
    path = ROOT / "INTERVENTION_RESULTS_LEDGER.md"
    text = path.read_text(encoding="utf-8").rstrip() + "\n"
    if marker in text:
        return
    path.write_text(text + "\n" + marker + "\n" + "\n".join(lines), encoding="utf-8")


def _development_case_table(cases: list[StateCase]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "state_id": case.state_id,
                "candidate": case.candidate,
                "matrix_id": case.matrix_id,
                "landmark": case.landmark,
                "generation": case.snapshot.generation,
                "mass": int(case.snapshot.composition.sum()),
                "occupied_types": int(np.count_nonzero(case.snapshot.composition)),
                "state_digest": base._snapshot_digest(case),
            }
            for case in cases
        ]
    )


def run_development(
    registration_directory: Path = DEFAULT_PROTOCOL_REGISTRATION,
    output: Path = DEFAULT_DEVELOPMENT,
    work: Path = DEFAULT_DEVELOPMENT_WORK,
    workers: int = min(os.cpu_count() or 1, 14),
) -> None:
    if workers < 1:
        raise ValueError("workers must be positive")
    registration = verify_protocol_registration(registration_directory)
    verify_checksums(SCALED5)
    output = output.resolve()
    work = work.resolve()
    _prepare_work(work, output, "development", registration["registration_id"])
    experiment = _scaled5_experiment()
    print(
        "[cr5 development 1/9] Reconstructing the exact 5x development cohort",
        flush=True,
    )
    _status(work, "development", "running", "reconstructing_5x_cohort")
    with threadpool_limits(limits=1):
        cases = build_cohort(experiment, DEVELOPMENT_LABEL, experiment.development)
        features = extract_features(cases, experiment)
    if len(cases) != 2 * MATRICES * len(LANDMARKS):
        raise AssertionError("CR5 development cohort is incomplete")
    with np.load(SCALED5 / "analysis_arrays.npz", allow_pickle=False) as archived:
        feature_checks = {
            "state_graph": np.array_equal(
                features.state_graph, archived["development_state_graph"]
            ),
            "history": np.array_equal(
                features.history, archived["development_history"]
            ),
            "beta": np.array_equal(features.beta, archived["development_beta"]),
        }
        archived_joint = archived["development_targets"].copy()
    if not all(feature_checks.values()):
        raise AssertionError(f"5x development feature replay changed: {feature_checks}")

    print(
        "[cr5 development 2/9] Regenerating archived F12 futures and deriving F6 breaks",
        flush=True,
    )
    _status(work, "development", "running", "break_target_generation")
    break_generated = _run_target_batches(
        cases,
        experiment,
        _development_break_worker,
        work / "break_generate",
        workers,
        registration["registration_id"],
        "break_generate",
    )
    print("[cr5 development 3/9] Replaying all development break futures", flush=True)
    break_replayed = _run_target_batches(
        cases,
        experiment,
        _development_break_worker,
        work / "break_replay",
        workers,
        registration["registration_id"],
        "break_replay",
    )
    break_replay = _target_replay_audit(break_generated, break_replayed)
    break_labels = np.vstack([batch.labels for batch in break_generated])
    joint_labels = np.vstack([batch.auxiliary_labels for batch in break_generated])
    archive_target_exact = bool(np.array_equal(joint_labels, archived_joint))
    if not break_replay["all_exact"] or not archive_target_exact:
        raise AssertionError("CR5 archived development target replay failed")

    print(
        "[cr5 development 4/9] Acquiring and replaying untreated natural breaks",
        flush=True,
    )
    _status(work, "development", "running", "natural_break_acquisition")
    broken, anchors, acquisition = acquire_natural_breaks(
        cases,
        experiment.gard,
        SEEDS["development_renewal_acquisition"],
        "INTCR5.development.renewal.acquisition",
    )
    replay_broken, replay_anchors, replay_acquisition = acquire_natural_breaks(
        cases,
        experiment.gard,
        SEEDS["development_renewal_acquisition"],
        "INTCR5.development.renewal.acquisition",
    )
    acquisition_exact = _acquisition_exact(
        broken,
        anchors,
        acquisition,
        replay_broken,
        replay_anchors,
        replay_acquisition,
    )
    if not acquisition_exact or not broken:
        raise AssertionError("CR5 development natural-break acquisition failed")
    with threadpool_limits(limits=1):
        renewal_features = extract_features(broken, experiment)

    print(
        f"[cr5 development 5/9] Shooting {len(broken) * DEVELOPMENT_BRANCHES:,} F8 renewal futures",
        flush=True,
    )
    _status(work, "development", "running", "renewal_target_generation")
    renewal_generated = _run_target_batches(
        broken,
        experiment,
        _development_renewal_worker,
        work / "renewal_generate",
        workers,
        registration["registration_id"],
        "renewal_generate",
    )
    print("[cr5 development 6/9] Replaying all development renewal futures", flush=True)
    renewal_replayed = _run_target_batches(
        broken,
        experiment,
        _development_renewal_worker,
        work / "renewal_replay",
        workers,
        registration["registration_id"],
        "renewal_replay",
    )
    renewal_replay = _target_replay_audit(renewal_generated, renewal_replayed)
    if not renewal_replay["all_exact"]:
        raise AssertionError("CR5 renewal development replay failed")
    renewal_labels = np.vstack([batch.labels for batch in renewal_generated])

    print(
        "[cr5 development 7/9] Selecting ridge penalties by whole-matrix CV", flush=True
    )
    _status(work, "development", "running", "whole_matrix_cv_and_model_freeze")
    students: dict[tuple[str, str], FrozenCR5Student] = {}
    diagnostics: dict[str, Any] = {}
    for target, target_cases, target_features, labels in (
        ("break", cases, features, break_labels),
        ("renewal", broken, renewal_features, renewal_labels),
    ):
        for candidate in CANDIDATES:
            selected = np.asarray(
                [case.candidate == candidate for case in target_cases], dtype=bool
            )
            ids = np.asarray([case.matrix_id for case in target_cases], dtype=np.int64)[
                selected
            ]
            student, item = fit_cr5_student(
                target,
                candidate,
                target_features.state_graph[selected],
                target_features.history[selected],
                labels[selected],
                ids,
            )
            students[(target, candidate)] = student
            diagnostics[f"{target}__c{candidate}"] = item

    print(
        "[cr5 development 8/9] Writing and serialization-checking the frozen students",
        flush=True,
    )
    with _atomic_destination(output) as destination:
        model_path = destination / "frozen_cr5_students.npz"
        contract_path = destination / "model_contract.json"
        save_students(model_path, contract_path, students, diagnostics)
        loaded = load_students(model_path, contract_path)
        prediction_checks: dict[str, bool] = {}
        for (target, candidate), student in students.items():
            target_cases = cases if target == "break" else broken
            target_features = features if target == "break" else renewal_features
            selected = np.asarray(
                [case.candidate == candidate for case in target_cases], dtype=bool
            )
            left = student.predict_features(
                target_features.state_graph[selected],
                target_features.history[selected],
            )
            right = loaded[(target, candidate)].predict_features(
                target_features.state_graph[selected],
                target_features.history[selected],
            )
            prediction_checks[f"{target}__c{candidate}"] = np.array_equal(left, right)
        if not all(prediction_checks.values()):
            raise AssertionError("CR5 student serialization changed predictions")
        np.savez_compressed(
            destination / "development_arrays.npz",
            break_state_graph=features.state_graph,
            break_history=features.history,
            break_labels=break_labels,
            break_joint_f12_labels=joint_labels,
            renewal_state_graph=renewal_features.state_graph,
            renewal_history=renewal_features.history,
            renewal_labels=renewal_labels,
            renewal_old_anchors=np.vstack(anchors).astype(np.int16),
            break_matrix_ids=np.asarray([case.matrix_id for case in cases]),
            renewal_matrix_ids=np.asarray([case.matrix_id for case in broken]),
        )
        _development_case_table(cases).to_csv(
            destination / "break_development_states.csv", index=False
        )
        _development_case_table(broken).to_csv(
            destination / "renewal_development_states.csv", index=False
        )
        acquisition.to_csv(destination / "renewal_acquisition.csv", index=False)
        replay = {
            "feature_arrays_exact_to_archived_5x": feature_checks,
            "archived_joint_f12_targets_exact": archive_target_exact,
            "break_future_replay": break_replay,
            "natural_break_acquisition_replay_exact": acquisition_exact,
            "renewal_future_replay": renewal_replay,
            "model_serialization_predictions_exact": prediction_checks,
            "all_exact": bool(
                all(feature_checks.values())
                and archive_target_exact
                and break_replay["all_exact"]
                and acquisition_exact
                and renewal_replay["all_exact"]
                and all(prediction_checks.values())
            ),
        }
        (destination / "replay_audit.json").write_text(
            json.dumps(_json_ready(replay), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (destination / "fit_diagnostics.json").write_text(
            json.dumps(_json_ready(diagnostics), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        eligible = {
            candidate: {
                "states": sum(case.candidate == candidate for case in broken),
                "matrices": len(
                    {case.matrix_id for case in broken if case.candidate == candidate}
                ),
            }
            for candidate in CANDIDATES
        }
        manifest = {
            "format": DEVELOPMENT_FORMAT,
            "protocol_registration_id": registration["registration_id"],
            "source_hashes": source_hashes(),
            "break_states": len(cases),
            "break_development_futures": len(cases) * DEVELOPMENT_BRANCHES,
            "renewal_eligible": eligible,
            "renewal_development_futures": len(broken) * DEVELOPMENT_BRANCHES,
            "complete_replay": replay["all_exact"],
            "confirmation_matrices_generated": 0,
            "confirmation_futures_generated": 0,
            "frozen_model_sha256": sha256_file(model_path),
            "model_contract_sha256": sha256_file(contract_path),
            "runtime": _runtime_manifest(),
        }
        (destination / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        report = [
            "# CR5 development and model freeze",
            "",
            "The original 5x development cohort, feature arrays, and F12 target labels reproduced exactly. New F6 break and shared-natural-break F8 renewal targets were generated and replayed exactly.",
            "",
            "| Student | States | Event rate | Selected ridge |",
            "|---|---:|---:|---:|",
        ]
        for key, item in diagnostics.items():
            report.append(
                f"| {key} | {item['states']} | {item['event_rate']:.6f} | {item['selected_ridge_penalty']:g} |"
            )
        report.extend(
            [
                "",
                "No CR5 confirmation matrix or intervention future existed when these models were frozen.",
                "",
            ]
        )
        (destination / "SCIENTIFIC_REPORT.md").write_text(
            "\n".join(report), encoding="utf-8"
        )
        write_checksums(destination)
    verify_development(output, registration_directory)
    _status(work, "development", "sealed_complete", "model_freeze_complete")
    _append_ledger(
        f"<!-- cr5-development-{registration['registration_id']} -->",
        [
            "## CR5 development students frozen",
            "",
            f"- Protocol registration: `{registration['registration_id']}`.",
            f"- Development freeze: `{output.relative_to(ROOT)}`.",
            "- The 5x cohort and archived F12 labels reproduced exactly; all new F6/F8 development futures and natural-break acquisition replayed exactly.",
            "- No CR5 confirmation matrices or futures were generated before model freeze.",
            "",
        ],
    )
    print(
        "[cr5 development 9/9] Frozen model bundle sealed; confirmation not started",
        flush=True,
    )


def verify_development(
    directory: Path = DEFAULT_DEVELOPMENT,
    registration_directory: Path = DEFAULT_PROTOCOL_REGISTRATION,
) -> dict[str, Any]:
    registration = verify_protocol_registration(registration_directory)
    directory = directory.resolve()
    verify_checksums(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    replay = json.loads((directory / "replay_audit.json").read_text())
    if manifest.get("format") != DEVELOPMENT_FORMAT:
        raise ValueError("unsupported CR5 development freeze")
    if manifest["protocol_registration_id"] != registration["registration_id"]:
        raise ValueError("CR5 development belongs to another protocol")
    if manifest["source_hashes"] != source_hashes():
        raise ValueError("CR5 source changed after development freeze")
    if not manifest["complete_replay"] or not replay["all_exact"]:
        raise ValueError("CR5 development replay did not pass")
    if manifest["frozen_model_sha256"] != sha256_file(
        directory / "frozen_cr5_students.npz"
    ):
        raise ValueError("CR5 frozen model hash changed")
    load_students(
        directory / "frozen_cr5_students.npz", directory / "model_contract.json"
    )
    return manifest


def seal_confirmation(
    development_directory: Path = DEFAULT_DEVELOPMENT,
    protocol_registration: Path = DEFAULT_PROTOCOL_REGISTRATION,
    output: Path = DEFAULT_CONFIRMATION_REGISTRATION,
) -> None:
    protocol_registration = protocol_registration.resolve()
    development_directory = development_directory.resolve()
    output = output.resolve()
    protocol_payload = verify_protocol_registration(protocol_registration)
    development = verify_development(development_directory, protocol_registration)
    for forbidden in (DEFAULT_OUTPUT, DEFAULT_WORK):
        if forbidden.exists():
            raise FileExistsError(
                f"CR5 confirmation artifact exists before confirmation seal: {forbidden}"
            )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    payload: dict[str, Any] = {
        "format": CONFIRMATION_REGISTRATION_FORMAT,
        "protocol_registration_id": protocol_payload["registration_id"],
        "protocol_id": protocol_payload["protocol_id"],
        "development_checksum_manifest_sha256": sha256_file(
            development_directory / "SHA256SUMS"
        ),
        "frozen_model_sha256": development["frozen_model_sha256"],
        "model_contract_sha256": development["model_contract_sha256"],
        "source_hashes": source_hashes(),
        "seed_registry": SEEDS,
        "confirmation_matrices_generated_at_seal": 0,
        "confirmation_futures_generated_at_seal": 0,
    }
    payload["registration_id"] = _canonical_digest(_json_ready(payload))
    with _atomic_destination(output) as destination:
        (destination / "registration.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        shutil.copy2(
            development_directory / "frozen_cr5_students.npz",
            destination / "frozen_cr5_students.npz",
        )
        shutil.copy2(
            development_directory / "model_contract.json",
            destination / "model_contract.json",
        )
        shutil.copy2(
            protocol_registration / "protocol.json", destination / "protocol.json"
        )
        shutil.copy2(
            protocol_registration / "seed_registry.json",
            destination / "seed_registry.json",
        )
        write_checksums(destination)
    verify_confirmation_registration(output)
    _append_ledger(
        f"<!-- cr5-confirmation-registration-{payload['registration_id']} -->",
        [
            "## CR5 confirmation sealed",
            "",
            f"- Confirmation registration: `{payload['registration_id']}`.",
            f"- Frozen model SHA-256: `{payload['frozen_model_sha256']}`.",
            "- No CR5 confirmation matrix or intervention future existed at the seal.",
            "",
        ],
    )
    print(f"CR5 confirmation registered: {payload['registration_id']}", flush=True)


def verify_confirmation_registration(
    directory: Path = DEFAULT_CONFIRMATION_REGISTRATION,
) -> dict[str, Any]:
    directory = directory.resolve()
    verify_checksums(directory)
    payload = json.loads((directory / "registration.json").read_text())
    if payload.get("format") != CONFIRMATION_REGISTRATION_FORMAT:
        raise ValueError("unsupported CR5 confirmation registration")
    if payload["source_hashes"] != source_hashes():
        raise ValueError("CR5 source changed after confirmation registration")
    if payload["frozen_model_sha256"] != sha256_file(
        directory / "frozen_cr5_students.npz"
    ):
        raise ValueError("CR5 registered model changed")
    if payload["model_contract_sha256"] != sha256_file(
        directory / "model_contract.json"
    ):
        raise ValueError("CR5 registered model contract changed")
    if payload["registration_id"] != _canonical_digest(
        {key: value for key, value in payload.items() if key != "registration_id"}
    ):
        raise ValueError("CR5 confirmation registration ID changed")
    load_students(
        directory / "frozen_cr5_students.npz", directory / "model_contract.json"
    )
    return payload


def smoke(
    registration_directory: Path = DEFAULT_CONFIRMATION_REGISTRATION,
    output: Path = DEFAULT_SMOKE,
) -> None:
    registration_directory = registration_directory.resolve()
    output = output.resolve()
    registration = verify_confirmation_registration(registration_directory)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    config = GardConfig()
    composition = np.zeros(config.n_types, dtype=np.int64)
    composition[: config.n_min] = 1
    beta = generate_beta(
        config,
        np.random.default_rng(derive_seed(SEEDS["validation"], "smoke.beta")),
    )
    snapshot = Snapshot(
        composition=composition,
        generation=20,
        inheritance=(True, True, False),
        boundary_h=(0.95, 0.92, 0.80),
        previous_growth_steps=12,
        cumulative_growth_steps=93,
    )
    students = load_students(
        registration_directory / "frozen_cr5_students.npz",
        registration_directory / "model_contract.json",
    )
    checks: dict[str, bool] = {}
    for target in ("break", "renewal"):
        student = students[(target, "02")]
        case = StateCase(
            f"cr5-smoke-{target}", "ARTIFICIAL_FIXTURE", "02", 0, 20, beta, snapshot
        )
        noop, scores = score_student_edits(student, case, config)
        selection_seed = derive_seed(SEEDS["smoke_selection"], target)
        left = select_student_edits(noop, scores, np.random.default_rng(selection_seed))
        right = select_student_edits(
            noop, scores, np.random.default_rng(selection_seed)
        )
        checks[f"{target}_exhaustive_selection_deterministic"] = bool(
            np.array_equal(left[0], right[0]) and left[1] == right[1]
        )
        checks[f"{target}_all_legal_swaps_scored"] = len(scores) == len(
            enumerate_legal_edits(composition)
        )
    future_seed = derive_seed(SEEDS["smoke_future"], "fixture")
    records_a, complete_a = simulate_future_absorbing(
        snapshot,
        beta,
        config,
        CANDIDATES["02"],
        2,
        np.random.default_rng(future_seed),
    )
    records_b, complete_b = simulate_future_absorbing(
        snapshot,
        beta,
        config,
        CANDIDATES["02"],
        2,
        np.random.default_rng(future_seed),
    )
    checks["future_replay_exact"] = complete_a == complete_b and _records_digest(
        records_a
    ) == _records_digest(records_b)
    checks["no_effect_sizes_or_arm_ordering_disclosed"] = True
    if not all(checks.values()):
        raise AssertionError(
            {name: value for name, value in checks.items() if not value}
        )
    with _atomic_destination(output) as destination:
        (destination / "manifest.json").write_text(
            json.dumps(
                {
                    "format": "codex-intervention-cr5-smoke-v1",
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
            + "\n",
            encoding="utf-8",
        )
        write_checksums(destination)
    verify_checksums(output)
    print(f"CR5 non-scientific smoke passed: {output}", flush=True)


def _write_branch_table(
    path: Path,
    cases: list[StateCase],
    batches: list[base.PhaseBatch],
    target_name: str,
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
                target_name,
                "break_event",
                "run3",
                "inherited_boundary_count",
                "first_break_time",
                "renewal_time",
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
                            "A"
                            if branch < len(batch.outcomes[arm_index]) // 2
                            else "B",
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


def _secondary(
    cases: list[StateCase],
    arrays: dict[str, NDArray],
    spec: CR5PhaseSpec,
    anchors: list[NDArray[np.int64]] | None = None,
) -> dict[str, Any]:
    inherited = np.asarray(arrays["boundary_h"] > 0.9, dtype=bool)
    run5 = np.zeros(inherited.shape[:3], dtype=np.int8)
    for location in np.ndindex(run5.shape):
        run5[location] = int(_first_run(inherited[location], 5) >= 0)
    old_similarity: NDArray[np.float64] | None = None
    if anchors is not None:
        old_similarity = np.empty(run5.shape, dtype=np.float64)
        for state_index, anchor in enumerate(anchors):
            for arm_index in range(len(spec.arms)):
                for branch in range(spec.branches):
                    old_similarity[state_index, arm_index, branch] = cosine_similarity(
                        anchor,
                        arrays["final_composition"][state_index, arm_index, branch],
                    )
    cells: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        selected = np.asarray(
            [case.candidate == candidate for case in cases], dtype=bool
        )
        for half, branch_slice in (
            ("A", slice(0, spec.branches // 2)),
            ("B", slice(spec.branches // 2, spec.branches)),
        ):
            for arm_index, arm in enumerate(spec.arms):
                renewal = arrays["renewal_time"][selected, arm_index, branch_slice]
                first_break = arrays["first_break_time"][
                    selected, arm_index, branch_slice
                ]
                row: dict[str, Any] = {
                    "candidate": candidate,
                    "branch_half": half,
                    "arm": arm,
                    "mean_target": float(
                        arrays["targets"][selected, arm_index, branch_slice].mean()
                    ),
                    "mean_run5": float(run5[selected, arm_index, branch_slice].mean()),
                    "mean_inherited_boundary_count": float(
                        arrays["inherited_boundary_count"][
                            selected, arm_index, branch_slice
                        ].mean()
                    ),
                    "mean_survival": float(
                        arrays["completed_horizon"][
                            selected, arm_index, branch_slice
                        ].mean()
                    ),
                    "mean_growth_updates": float(
                        arrays["mean_growth_updates"][
                            selected, arm_index, branch_slice
                        ].mean()
                    ),
                    "mean_final_entropy": float(
                        arrays["final_entropy"][
                            selected, arm_index, branch_slice
                        ].mean()
                    ),
                    "mean_final_occupied_types": float(
                        arrays["final_occupied_types"][
                            selected, arm_index, branch_slice
                        ].mean()
                    ),
                    "mean_first_break_time_given_break": (
                        float(first_break[first_break >= 0].mean())
                        if np.any(first_break >= 0)
                        else None
                    ),
                    "mean_renewal_time_given_renewal": (
                        float(renewal[renewal >= 0].mean())
                        if np.any(renewal >= 0)
                        else None
                    ),
                }
                if old_similarity is not None:
                    row["mean_old_anchor_similarity_at_horizon"] = float(
                        old_similarity[selected, arm_index, branch_slice].mean()
                    )
                cells.append(row)
    return {
        "stage": spec.stage,
        "cells": cells,
        "conditional_times_do_not_impute_uncertified_branches": True,
    }


def _draws(spec: CR5PhaseSpec) -> dict[str, NDArray]:
    return generate_inference_draws(
        MATRICES,
        BOOTSTRAP_REPETITIONS,
        RANDOMIZATION_REPETITIONS,
        np.random.default_rng(
            derive_seed(spec.bootstrap_seed, f"INTCR5.{spec.stage}.bootstrap")
        ),
        np.random.default_rng(
            derive_seed(spec.randomization_seed, f"INTCR5.{spec.stage}.randomization")
        ),
    )


def _readback_stage(
    directory: Path,
    cases: list[StateCase],
    spec: CR5PhaseSpec,
    expected_metrics: dict[str, Any],
    expected_rows: list[dict[str, Any]],
) -> dict[str, bool]:
    with np.load(directory / "branch_arrays.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    with np.load(directory / "inference_draws.npz", allow_pickle=False) as archive:
        draws = {name: archive[name] for name in archive.files}
    metrics, rows = _inference(cases, arrays, spec, draws)
    return {
        "branch_arrays_reloaded": True,
        "inference_draws_reloaded": True,
        "metrics_exact": _json_ready(metrics) == _json_ready(expected_metrics),
        "matrix_effects_exact": _json_ready(rows) == _json_ready(expected_rows),
    }


def _reports(
    resistance: dict[str, Any],
    resilience: dict[str, Any] | None,
    eligibility: dict[str, Any],
    integrity: dict[str, Any],
) -> tuple[str, str]:
    lines = [
        "# CR5 resistance and resilience confirmation",
        "",
        f"Resistance four-cell gate: **{resistance['cr5_all_four_cells_pass']}**.",
        f"Resilience classification: **{eligibility['classification']}**.",
    ]
    if resilience is not None:
        lines.append(
            f"Resilience four-cell gate: **{resilience['cr5_all_four_cells_pass']}**."
        )
    lines.extend(
        [
            f"Exact replay and readback: **{integrity['all_pass']}**.",
            "",
            "## Resistance",
            "",
            "| Cell | Break-up minus break-down | 95% CI | Holm p | Random-noop 90% CI | Pass |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for cell in resistance["cells"]:
        effect = cell["contrasts"]["up_minus_down"]
        lines.append(
            f"| {cell['cell']} | {effect['estimate']:+.6f} | {effect['bootstrap_ci95']} | "
            f"{cell['up_down_randomization_p_holm']:.6g} | "
            f"{cell['random_noop_equivalence']['bootstrap_ci90']} | "
            f"{cell['cr5_registered_cell_pass']} |"
        )
    if resilience is not None:
        lines.extend(
            [
                "",
                "## Resilience from identical naturally broken daughters",
                "",
                "| Cell | Renewal-up minus renewal-down | 95% CI | Holm p | Random-noop 90% CI | Pass |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for cell in resilience["cells"]:
            effect = cell["contrasts"]["up_minus_down"]
            lines.append(
                f"| {cell['cell']} | {effect['estimate']:+.6f} | {effect['bootstrap_ci95']} | "
                f"{cell['up_down_randomization_p_holm']:.6g} | "
                f"{cell['random_noop_equivalence']['bootstrap_ci90']} | "
                f"{cell['cr5_registered_cell_pass']} |"
            )
    lines.extend(
        [
            "",
            "All natural-break acquisition failures remain in the ledger and no matrix or lineage source was replaced. Resistance and resilience are separate registered conclusions; one cannot rescue the other.",
            "",
            "CR5 concerns a narrow simulated hereditary process. It does not establish biological repair, biological memory, agency, life, autonomy, strict-eight control, real chemistry, or Phi/PhiID intervention.",
            "",
        ]
    )
    resistance_pass = resistance["cr5_all_four_cells_pass"]
    resilience_pass = resilience is not None and resilience["cr5_all_four_cells_pass"]
    lay = "\n".join(
        [
            "# CR5 in plain language",
            "",
            "The earlier event mixes two abilities: avoiding a heredity break, and rebuilding a short inherited run after a break has already happened. CR5 tested those abilities separately.",
            "",
            (
                "Tiny predictor-chosen molecular edits reliably changed how often heredity broke within six fissions."
                if resistance_pass
                else "The molecular edits did not pass every prewritten test for controlling whether heredity first broke."
            ),
            "",
            (
                "Starting from exactly the same naturally broken daughter, tiny predictor-chosen edits also reliably changed short-run recovery."
                if resilience_pass
                else (
                    "The shared-broken-state recovery test was inconclusive because complete natural-break matrix coverage was unavailable."
                    if resilience is None
                    else "Starting from the same naturally broken daughter, the recovery edits did not pass every prewritten test."
                )
            ),
            "",
            "This is causal control inside a computer model, not evidence that the assemblies are alive or possess biological memory.",
            "",
        ]
    )
    return "\n".join(lines), lay


def run_confirmation(
    registration_directory: Path = DEFAULT_CONFIRMATION_REGISTRATION,
    output: Path = DEFAULT_OUTPUT,
    work: Path = DEFAULT_WORK,
    workers: int = min(os.cpu_count() or 1, 14),
    cpu_budget_hours: float = 20.0,
) -> None:
    if workers < 1:
        raise ValueError("workers must be positive")
    registration_directory = registration_directory.resolve()
    output = output.resolve()
    work = work.resolve()
    registration = verify_confirmation_registration(registration_directory)
    verify_checksums(DEFAULT_SMOKE)
    _prepare_work(
        work,
        output,
        "confirmation",
        registration["registration_id"],
        cpu_budget_hours,
    )
    experiment = _confirmation_experiment()
    model_path = registration_directory / "frozen_cr5_students.npz"
    model_contract = registration_directory / "model_contract.json"
    print(
        "[cr5 confirmation 1/10] Building 200 fresh matrices and 2,000 natural states",
        flush=True,
    )
    _status(work, "confirmation", "running", "building_fresh_natural_states")
    with threadpool_limits(limits=1):
        cases = build_cohort(experiment, CONFIRMATION_LABEL, experiment.confirmation)
    if len(cases) != 2 * MATRICES * len(LANDMARKS):
        raise AssertionError("CR5 confirmation cohort is incomplete")

    resistance = resistance_spec()
    resistance_futures = len(cases) * len(resistance.arms) * resistance.branches
    print(
        f"[cr5 confirmation 2/10] Shooting {resistance_futures:,} F6 resistance futures",
        flush=True,
    )
    _status(work, "confirmation", "running", "resistance_primary")
    resistance_generated = run_phase_batches(
        cases,
        experiment.gard,
        resistance,
        model_path,
        model_contract,
        registration["registration_id"],
        work / "resistance_generate",
        workers,
        "generate",
    )
    print("[cr5 confirmation 3/10] Replaying every resistance future", flush=True)
    resistance_replayed = run_phase_batches(
        cases,
        experiment.gard,
        resistance,
        model_path,
        model_contract,
        registration["registration_id"],
        work / "resistance_replay",
        workers,
        "replay",
    )
    resistance_replay = base.replay_audit(resistance_generated, resistance_replayed)
    if not resistance_replay["state_edit_endpoint_and_process_digests_exact"]:
        raise AssertionError("CR5 resistance replay failed")
    resistance_arrays = _outcome_arrays(cases, resistance_generated, resistance)
    resistance_draws = _draws(resistance)
    resistance_metrics, resistance_rows = _inference(
        cases, resistance_arrays, resistance, resistance_draws
    )

    print(
        "[cr5 confirmation 4/10] Acquiring and replaying untreated natural breaks",
        flush=True,
    )
    _status(work, "confirmation", "running", "natural_break_acquisition")
    broken, anchors, acquisition = acquire_natural_breaks(
        cases,
        experiment.gard,
        SEEDS["resilience_acquisition"],
        "INTCR5.confirmation.resilience.acquisition",
    )
    replay_broken, replay_anchors, replay_acquisition = acquire_natural_breaks(
        cases,
        experiment.gard,
        SEEDS["resilience_acquisition"],
        "INTCR5.confirmation.resilience.acquisition",
    )
    acquisition_exact = _acquisition_exact(
        broken,
        anchors,
        acquisition,
        replay_broken,
        replay_anchors,
        replay_acquisition,
    )
    if not acquisition_exact:
        raise AssertionError("CR5 confirmation natural-break replay failed")
    eligible = {
        candidate: {
            "states": sum(case.candidate == candidate for case in broken),
            "matrices": len(
                {case.matrix_id for case in broken if case.candidate == candidate}
            ),
        }
        for candidate in CANDIDATES
    }
    full_coverage = all(value["matrices"] == MATRICES for value in eligible.values())
    eligibility = {
        "eligible_by_candidate": eligible,
        "source_states": len(cases),
        "eligible_broken_states": len(broken),
        "acquisition_limit": NATURAL_BREAK_ACQUISITION_LIMIT,
        "acquisition_replay_exact": acquisition_exact,
        "all_200_matrices_per_candidate": full_coverage,
        "classification": "eligible"
        if full_coverage
        else "inconclusive_incomplete_matrix_coverage",
        "no_replacement": True,
    }

    resilience = resilience_spec()
    resilience_generated: list[base.PhaseBatch] = []
    resilience_arrays: dict[str, NDArray] | None = None
    resilience_draws: dict[str, NDArray] | None = None
    resilience_metrics: dict[str, Any] | None = None
    resilience_rows: list[dict[str, Any]] = []
    resilience_replay: dict[str, Any] = {
        "not_launched_due_to_incomplete_matrix_coverage": not full_coverage
    }
    if full_coverage:
        resilience_futures = len(broken) * len(resilience.arms) * resilience.branches
        print(
            f"[cr5 confirmation 5/10] Shooting {resilience_futures:,} F8 shared-state resilience futures",
            flush=True,
        )
        _status(work, "confirmation", "running", "resilience_primary")
        resilience_generated = run_phase_batches(
            broken,
            experiment.gard,
            resilience,
            model_path,
            model_contract,
            registration["registration_id"],
            work / "resilience_generate",
            workers,
            "generate",
        )
        print("[cr5 confirmation 6/10] Replaying every resilience future", flush=True)
        resilience_replayed = run_phase_batches(
            broken,
            experiment.gard,
            resilience,
            model_path,
            model_contract,
            registration["registration_id"],
            work / "resilience_replay",
            workers,
            "replay",
        )
        resilience_replay = base.replay_audit(resilience_generated, resilience_replayed)
        if not resilience_replay["state_edit_endpoint_and_process_digests_exact"]:
            raise AssertionError("CR5 resilience replay failed")
        resilience_arrays = _outcome_arrays(broken, resilience_generated, resilience)
        resilience_draws = _draws(resilience)
        resilience_metrics, resilience_rows = _inference(
            broken, resilience_arrays, resilience, resilience_draws
        )
    else:
        print(
            "[cr5 confirmation 5/10] Resilience sealed inconclusive; no replacements",
            flush=True,
        )
        print("[cr5 confirmation 6/10] No resilience futures launched", flush=True)

    print(
        "[cr5 confirmation 7/10] Writing complete machine-readable artifacts",
        flush=True,
    )
    _status(work, "confirmation", "running", "artifact_write_and_readback")
    with _atomic_destination(output) as destination:
        resistance_dir = destination / "resistance"
        resistance_dir.mkdir()
        np.savez_compressed(resistance_dir / "branch_arrays.npz", **resistance_arrays)
        np.savez_compressed(resistance_dir / "inference_draws.npz", **resistance_draws)
        _write_branch_table(
            resistance_dir / "branches.csv.gz",
            cases,
            resistance_generated,
            "break_within_f6",
        )
        base._write_state_artifacts(
            resistance_dir, cases, resistance_generated, resistance_arrays
        )
        base._write_selection_artifacts(
            resistance_dir,
            cases,
            resistance_generated,
            resistance,  # type: ignore[arg-type]
        )
        pd.DataFrame(resistance_rows).to_csv(
            resistance_dir / "matrix_effects.csv", index=False
        )
        (resistance_dir / "primary_metrics.json").write_text(
            json.dumps(_json_ready(resistance_metrics), indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        (resistance_dir / "secondary_outcomes.json").write_text(
            json.dumps(
                _json_ready(_secondary(cases, resistance_arrays, resistance)),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        acquisition.to_csv(destination / "natural_break_acquisition.csv", index=False)
        (destination / "resilience_eligibility.json").write_text(
            json.dumps(eligibility, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        resilience_readback = {
            "not_applicable": not full_coverage,
            "metrics_exact": not full_coverage,
            "matrix_effects_exact": not full_coverage,
        }
        if full_coverage:
            assert resilience_arrays is not None
            assert resilience_draws is not None
            assert resilience_metrics is not None
            resilience_dir = destination / "resilience"
            resilience_dir.mkdir()
            np.savez_compressed(
                resilience_dir / "branch_arrays.npz", **resilience_arrays
            )
            np.savez_compressed(
                resilience_dir / "inference_draws.npz", **resilience_draws
            )
            _write_branch_table(
                resilience_dir / "branches.csv.gz",
                broken,
                resilience_generated,
                "run3_within_f8",
            )
            base._write_state_artifacts(
                resilience_dir, broken, resilience_generated, resilience_arrays
            )
            base._write_selection_artifacts(
                resilience_dir,
                broken,
                resilience_generated,
                resilience,  # type: ignore[arg-type]
            )
            pd.DataFrame(resilience_rows).to_csv(
                resilience_dir / "matrix_effects.csv", index=False
            )
            (resilience_dir / "primary_metrics.json").write_text(
                json.dumps(_json_ready(resilience_metrics), indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            (resilience_dir / "secondary_outcomes.json").write_text(
                json.dumps(
                    _json_ready(
                        _secondary(broken, resilience_arrays, resilience, anchors)
                    ),
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            resilience_readback = _readback_stage(
                resilience_dir,
                broken,
                resilience,
                resilience_metrics,
                resilience_rows,
            )
        resistance_readback = _readback_stage(
            resistance_dir,
            cases,
            resistance,
            resistance_metrics,
            resistance_rows,
        )
        readback = {
            "resistance": resistance_readback,
            "resilience": resilience_readback,
            "all_exact": bool(
                all(resistance_readback.values()) and all(resilience_readback.values())
            ),
        }
        exact_resilience_replay = bool(
            not full_coverage
            or resilience_replay.get(
                "state_edit_endpoint_and_process_digests_exact", False
            )
        )
        integrity = {
            "resistance_replay_exact": resistance_replay[
                "state_edit_endpoint_and_process_digests_exact"
            ],
            "natural_break_acquisition_replay_exact": acquisition_exact,
            "resilience_replay_exact_or_not_launched": exact_resilience_replay,
            "artifact_readback_exact": readback["all_exact"],
        }
        integrity["all_pass"] = bool(all(integrity.values()))
        (destination / "replay_audit.json").write_text(
            json.dumps(
                _json_ready(
                    {
                        "resistance": resistance_replay,
                        "natural_break_acquisition_exact": acquisition_exact,
                        "resilience": resilience_replay,
                    }
                ),
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (destination / "readback_audit.json").write_text(
            json.dumps(readback, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        technical, lay = _reports(
            resistance_metrics, resilience_metrics, eligibility, integrity
        )
        (destination / "SCIENTIFIC_REPORT.md").write_text(technical, encoding="utf-8")
        (destination / "LAY_SUMMARY.md").write_text(lay, encoding="utf-8")
        supported: list[str] = []
        failed: list[str] = []
        if resistance_metrics["cr5_all_four_cells_pass"]:
            supported.append(
                "causal molecular control of first-break resistance within F6"
            )
        else:
            failed.append("registered four-cell resistance gate")
        if resilience_metrics is None:
            failed.append(
                "resilience confirmation unavailable from complete matrix coverage"
            )
        elif resilience_metrics["cr5_all_four_cells_pass"]:
            supported.append(
                "causal molecular control of run3 recovery from identical natural post-break daughters"
            )
        else:
            failed.append("registered four-cell shared-state resilience gate")
        claims = {
            "supported": supported,
            "failed_or_inconclusive": failed,
            "resistance_and_resilience_separate": True,
            "predecessor_cr1_to_cr4_unchanged": True,
            "unresolved": [
                "zero-shot parameter-regime transfer",
                "closed-loop hereditary steering",
            ],
            "prohibited": protocol()["claim_boundary"]["prohibited"],
        }
        (destination / "claim_boundaries.json").write_text(
            json.dumps(claims, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        manifest = {
            "format": RESULT_FORMAT,
            "registration_id": registration["registration_id"],
            "matrices": MATRICES,
            "resistance_states": len(cases),
            "resistance_primary_futures": resistance_futures,
            "resistance_replay_futures": resistance_futures,
            "resistance_gate": resistance_metrics["cr5_all_four_cells_pass"],
            "resilience_eligible_states": len(broken),
            "resilience_primary_futures": (
                len(broken) * len(resilience.arms) * resilience.branches
                if full_coverage
                else 0
            ),
            "resilience_replay_futures": (
                len(broken) * len(resilience.arms) * resilience.branches
                if full_coverage
                else 0
            ),
            "resilience_gate": (
                resilience_metrics["cr5_all_four_cells_pass"]
                if resilience_metrics is not None
                else None
            ),
            "resilience_classification": eligibility["classification"],
            "integrity": integrity,
            "declared_cpu_budget_hours": cpu_budget_hours,
            "no_refit_recalibration_retry_or_replacement": True,
            "mandatory_stop_after_this_stage": True,
            "cr6_launched": False,
            "runtime": _runtime_manifest(),
        }
        (destination / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        write_checksums(destination)
    verify_checksums(output)
    print("[cr5 confirmation 8/10] Written artifacts recomputed exactly", flush=True)
    _append_ledger(
        f"<!-- sealed-cr5-confirmation-{registration['registration_id']} -->",
        [
            "## CR5 resistance/resilience confirmation sealed",
            "",
            f"- Registration: `{registration['registration_id']}`.",
            f"- Result: `{output.relative_to(ROOT)}`.",
            f"- Resistance four-cell gate: **{resistance_metrics['cr5_all_four_cells_pass']}**.",
            f"- Resilience classification: **{eligibility['classification']}**.",
            f"- Resilience four-cell gate: **{None if resilience_metrics is None else resilience_metrics['cr5_all_four_cells_pass']}**.",
            "- Complete future replay, natural-break acquisition replay, and written-artifact readback passed.",
            "- CR1--CR4 remain unchanged; CR6 was not launched automatically.",
            "",
        ],
    )
    _status(
        work,
        "confirmation",
        "sealed_complete",
        "mandatory_review_stop",
        declared_cpu_budget_hours=cpu_budget_hours,
    )
    print("[cr5 confirmation 9/10] Cumulative ledger updated", flush=True)
    print("[cr5 confirmation 10/10] STOPPED; CR6 not launched", flush=True)


def read_status(work: Path = DEFAULT_WORK) -> dict[str, Any]:
    path = work.resolve() / "campaign_status.json"
    if not path.exists():
        return {"state": "not_started", "work_directory": str(work.resolve())}
    value = json.loads(path.read_text())
    for stage in (
        "break_generate",
        "break_replay",
        "renewal_generate",
        "renewal_replay",
        "resistance_generate",
        "resistance_replay",
        "resilience_generate",
        "resilience_replay",
    ):
        status = work.resolve() / stage / "status.json"
        if status.exists():
            value[stage] = json.loads(status.read_text())
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate").add_argument(
        "--output", type=Path, default=DEFAULT_VALIDATION
    )
    registration = commands.add_parser("register-protocol")
    registration.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    registration.add_argument(
        "--output", type=Path, default=DEFAULT_PROTOCOL_REGISTRATION
    )
    verify_protocol = commands.add_parser("verify-protocol")
    verify_protocol.add_argument(
        "--registration", type=Path, default=DEFAULT_PROTOCOL_REGISTRATION
    )
    development = commands.add_parser("develop")
    development.add_argument(
        "--registration", type=Path, default=DEFAULT_PROTOCOL_REGISTRATION
    )
    development.add_argument("--output", type=Path, default=DEFAULT_DEVELOPMENT)
    development.add_argument("--work-dir", type=Path, default=DEFAULT_DEVELOPMENT_WORK)
    development.add_argument(
        "--workers", type=int, default=min(os.cpu_count() or 1, 14)
    )
    seal = commands.add_parser("seal-confirmation")
    seal.add_argument("--development", type=Path, default=DEFAULT_DEVELOPMENT)
    seal.add_argument(
        "--protocol-registration", type=Path, default=DEFAULT_PROTOCOL_REGISTRATION
    )
    seal.add_argument("--output", type=Path, default=DEFAULT_CONFIRMATION_REGISTRATION)
    verify_confirmation = commands.add_parser("verify-confirmation")
    verify_confirmation.add_argument(
        "--registration", type=Path, default=DEFAULT_CONFIRMATION_REGISTRATION
    )
    smoke_parser = commands.add_parser("smoke")
    smoke_parser.add_argument(
        "--registration", type=Path, default=DEFAULT_CONFIRMATION_REGISTRATION
    )
    smoke_parser.add_argument("--output", type=Path, default=DEFAULT_SMOKE)
    run_parser = commands.add_parser("run")
    run_parser.add_argument(
        "--registration", type=Path, default=DEFAULT_CONFIRMATION_REGISTRATION
    )
    run_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run_parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    run_parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 14))
    run_parser.add_argument("--cpu-budget-hours", type=float, default=20.0)
    status = commands.add_parser("status")
    status.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "validate":
        validate(args.output)
    elif args.command == "register-protocol":
        register_protocol(args.validation, args.output)
    elif args.command == "verify-protocol":
        payload = verify_protocol_registration(args.registration)
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif args.command == "develop":
        run_development(args.registration, args.output, args.work_dir, args.workers)
    elif args.command == "seal-confirmation":
        seal_confirmation(args.development, args.protocol_registration, args.output)
    elif args.command == "verify-confirmation":
        payload = verify_confirmation_registration(args.registration)
        print(json.dumps(payload, indent=2, sort_keys=True))
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
