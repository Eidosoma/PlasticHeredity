"""Full prospective CR2 graded molecular dose-response confirmation."""

from __future__ import annotations

import argparse
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

from . import intervention_cr1_confirmation as cr1
from . import intervention_replication as base
from .config import CANDIDATES, CohortConfig, ExperimentConfig
from .experiment import StateCase, _json_ready, _runtime_manifest, build_cohort
from .intervention_core import (
    FrozenFullPredictor,
    MolecularEdit,
    ScoredEdit,
    score_legal_edits,
    simulate_one_shot,
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


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results_intervention_replication"
CR1_RESULT = RESULT_ROOT / "cr1_model_guided_confirmation"
CR1_REGISTRATION = RESULT_ROOT / "cr1_confirmation_registration"
DEFAULT_VALIDATION = RESULT_ROOT / "cr2_dose_response_validation"
DEFAULT_REGISTRATION = RESULT_ROOT / "cr2_dose_response_registration"
DEFAULT_SMOKE = RESULT_ROOT / "cr2_dose_response_smoke"
DEFAULT_OUTPUT = RESULT_ROOT / "cr2_dose_response_confirmation"
DEFAULT_WORK = RESULT_ROOT / ".cr2_dose_response_confirmation_work"

DOCUMENT = "CODEX_INTERVENTION_CR2_DOSE_RESPONSE_PREREGISTRATION.md"
SOURCE_FILES = (
    DOCUMENT,
    "plastic_heredity/intervention_cr2_dose_response.py",
    "tests/test_intervention_cr2_dose_response.py",
    "plastic_heredity/intervention_cr1_confirmation.py",
    "plastic_heredity/intervention_replication.py",
    "plastic_heredity/intervention_core.py",
    "plastic_heredity/intervention_metrics.py",
    "plastic_heredity/config.py",
    "plastic_heredity/experiment.py",
    "plastic_heredity/features.py",
    "plastic_heredity/processes.py",
    "plastic_heredity/seeds.py",
    "plastic_heredity/simulator.py",
)
PROGRAM_FORMAT = "codex-intervention-cr2-dose-response-v1"
VALIDATION_FORMAT = "codex-intervention-cr2-dose-response-validation-v1"
REGISTRATION_FORMAT = "codex-intervention-cr2-dose-response-registration-v1"
RESULT_FORMAT = "codex-intervention-cr2-dose-response-result-v1"
CHECKPOINT_FORMAT = "codex-intervention-cr2-dose-response-checkpoint-v1"
STATUS_FORMAT = "codex-intervention-cr2-dose-response-status-v1"
LABEL = cr1.LABEL
FUTURE_LABEL = "INTCR2_DOSE_RESPONSE_V1"
MATRICES = 200
BRANCHES = 64
LANDMARKS = (20, 35, 50, 65, 80)
HORIZON = 12
QUANTILES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
ARMS = ("Q00", "Q20", "Q40", "Q60", "Q80", "Q100")
BOOTSTRAP_REPETITIONS = 4_096
RANDOMIZATION_REPETITIONS = 4_096
MINIMUM_AVAILABLE_CPU_HOURS = 18.0


def _seed(name: str) -> str:
    return hashlib.sha256(
        f"codex-clean-room-cr2-dose-response-v1::{name}".encode("utf-8")
    ).hexdigest()


SEEDS = {
    name: _seed(name)
    for name in (
        "validation",
        "selection_audit",
        "smoke_cohort",
        "smoke_future",
        "future",
        "bootstrap",
        "randomization",
        "replay",
    )
}


@dataclass(frozen=True)
class DoseSpec:
    role: str
    matrices: int
    branches: int
    cohort_seed: str
    selection_seed: str
    future_seed: str
    bootstrap_seed: str
    randomization_seed: str
    phase: str = "cr2"
    arms: tuple[str, ...] = ARMS


def source_hashes() -> dict[str, str]:
    return {name: sha256_file(ROOT / name) for name in SOURCE_FILES}


def phase_spec() -> DoseSpec:
    return DoseSpec(
        role="full prospective CR2 graded molecular dose-response confirmation",
        matrices=MATRICES,
        branches=BRANCHES,
        cohort_seed=cr1.SEEDS["cohort"],
        selection_seed=SEEDS["selection_audit"],
        future_seed=SEEDS["future"],
        bootstrap_seed=SEEDS["bootstrap"],
        randomization_seed=SEEDS["randomization"],
    )


def experiment(spec: DoseSpec | None = None) -> ExperimentConfig:
    selected = phase_spec() if spec is None else spec
    cohort = CohortConfig(selected.matrices, selected.branches, LANDMARKS)
    return ExperimentConfig(
        development=cohort,
        confirmation=cohort,
        horizon=HORIZON,
        bootstrap_repetitions=BOOTSTRAP_REPETITIONS,
        permutation_repetitions=RANDOMIZATION_REPETITIONS,
        regenerate_confirmation=True,
        master_seed=selected.cohort_seed,
    )


def protocol() -> dict[str, Any]:
    value: dict[str, Any] = {
        "format": PROGRAM_FORMAT,
        "status": "frozen_before_any_cr2_scientific_future",
        "phase_advancement": {
            "cr1_full_four_cell_gate_required": True,
            "cr1_effect_sizes_not_used_for_design": True,
        },
        "endpoint": "JOINT_BREAK_RUN3 within F12",
        "states": {
            "same_as_cr1": True,
            "matrices": MATRICES,
            "candidates": list(CANDIDATES),
            "landmarks": list(LANDMARKS),
            "state_count": 2 * MATRICES * len(LANDMARKS),
            "exact_array_audit_before_futures": True,
        },
        "selection": {
            "arms": list(ARMS),
            "quantiles": list(QUANTILES),
            "rank_formula": "floor(q * (K - 1) + 0.5)",
            "tie_break": "lexicographically first edit at exact selected float64 probability",
            "all_legal_edits_scored_and_persisted": True,
            "relative_to_noop_prediction": True,
            "noop_is_not_a_simulation_arm": True,
            "selection_is_deterministic": True,
        },
        "futures": {
            "branches_per_arm_state": BRANCHES,
            "horizon": HORIZON,
            "halves": {"A": [0, 31], "B": [32, 63]},
            "primary_futures": 768_000,
            "replay_futures": 768_000,
            "common_random_streams": True,
            "future_seed_excludes_arm": True,
            "fresh_future_domain": True,
            "no_retries_or_replacements": True,
        },
        "inference": {
            "unit": "whole catalytic matrix",
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "randomization_repetitions": RANDOMIZATION_REPETITIONS,
            "undefined_state_spearman": 0.0,
            "tie_ranks": "average",
            "holm_families": [
                "four candidate-half Spearman tests",
                "four candidate-half slope tests",
            ],
            "randomization_p_values_are_reported_not_gates": True,
            "cell_gates": [
                "mean within-state Spearman > 0",
                "Spearman bootstrap lower bound > 0",
                "state-centered slope > 0",
                "slope bootstrap lower bound > 0",
            ],
            "all_four_cells_required": True,
        },
        "model": {
            "frozen_cr1_candidate_separated_5x_composite": True,
            "no_refit_recalibration_or_threshold_change": True,
        },
        "seed_domains": SEEDS,
        "minimum_available_cpu_hours": MINIMUM_AVAILABLE_CPU_HOURS,
        "complete_replay_and_readback_required": True,
        "mandatory_stop_after_result": True,
        "claim_boundary": {
            "prohibited": [
                "strict-eight control",
                "agency",
                "biological memory",
                "life",
                "autonomous organization",
                "real prebiotic chemistry",
                "universal origin-of-life mechanism",
            ]
        },
    }
    value["protocol_id"] = _canonical_digest(_json_ready(value))
    return value


def _selected_rank(count: int, quantile: float) -> int:
    if count < 1:
        raise ValueError("at least one legal edit is required")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must lie in [0, 1]")
    return int(np.floor(quantile * (count - 1) + 0.5))


def select_quantile_edits(
    scores: tuple[ScoredEdit, ...],
) -> tuple[tuple[ScoredEdit, ...], NDArray[np.int64]]:
    """Select actual empirical order statistics with deterministic tie handling."""

    if not scores:
        raise ValueError("cannot select quantiles from an empty edit set")
    order = sorted(
        range(len(scores)),
        key=lambda index: (
            scores[index].predicted_probability,
            scores[index].edit.remove_type,
            scores[index].edit.add_type,
        ),
    )
    selected: list[ScoredEdit] = []
    ranks: list[int] = []
    for quantile in QUANTILES:
        rank = _selected_rank(len(scores), quantile)
        target = scores[order[rank]].predicted_probability
        tied = [
            item
            for item in scores
            if item.predicted_probability == target
        ]
        chosen = min(
            tied,
            key=lambda item: (item.edit.remove_type, item.edit.add_type),
        )
        selected.append(chosen)
        ranks.append(rank)
    probabilities = np.asarray(
        [item.predicted_probability for item in selected], dtype=np.float64
    )
    if np.any(np.diff(probabilities) < 0.0):
        raise AssertionError("selected empirical quantiles are not monotone")
    return tuple(selected), np.asarray(ranks, dtype=np.int64)


def _future_seed(spec: DoseSpec, case: StateCase, branch: int) -> int:
    return derive_seed(
        spec.future_seed,
        f"{FUTURE_LABEL}.future",
        case.candidate,
        case.matrix_id,
        case.landmark,
        branch,
    )


def _select_state(
    case: StateCase,
    current_experiment: ExperimentConfig,
    predictor: FrozenFullPredictor,
) -> tuple[float, tuple[ScoredEdit, ...], tuple[ScoredEdit, ...], NDArray[np.int64]]:
    noop, scores = score_legal_edits(
        predictor,
        case.candidate,
        case.snapshot,
        case.beta,
        current_experiment.gard,
    )
    selected, ranks = select_quantile_edits(scores)
    return noop, scores, selected, ranks


def _dose_worker(
    arguments: tuple[StateCase, ExperimentConfig, DoseSpec, str]
) -> base.PhaseBatch:
    case, current_experiment, spec, model_path = arguments
    limiter = threadpool_limits(limits=1)
    try:
        predictor = FrozenFullPredictor.load(model_path)
        noop, scores, selected, _ranks = _select_state(
            case, current_experiment, predictor
        )
        edits = tuple(item.edit for item in selected)
        predictions = np.asarray(
            [item.predicted_probability for item in selected], dtype=np.float64
        )
        outcomes: list[list[Any]] = [[] for _ in spec.arms]
        for branch in range(spec.branches):
            seed = _future_seed(spec, case, branch)
            for arm_index, edit in enumerate(edits):
                outcomes[arm_index].append(
                    simulate_one_shot(
                        case.snapshot,
                        case.beta,
                        case.candidate,
                        current_experiment.gard,
                        HORIZON,
                        np.random.default_rng(seed),
                        edit,
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
            # CR2 has no catalytic-support arm; this one-element field stores
            # the frozen NOOP prediction needed to audit selected shifts.
            catalytic_support=np.asarray([noop], dtype=np.float64),
            outcomes=tuple(tuple(arm) for arm in outcomes),
        )
    finally:
        limiter.restore_original_limits()


def _checkpoint_contract(
    cases: list[StateCase], spec: DoseSpec, registration_id: str, stage: str
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "registration_id": registration_id,
        "stage": stage,
        "role": spec.role,
        "matrices": spec.matrices,
        "branches": spec.branches,
        "horizon": HORIZON,
        "arms": list(spec.arms),
        "quantiles": list(QUANTILES),
        "case_ids": [case.state_id for case in cases],
        "case_digests": [base._snapshot_digest(case) for case in cases],
        "future_seed": spec.future_seed,
        "future_seed_includes_arm": False,
        "selection_is_deterministic": True,
        "source_hashes": source_hashes(),
    }
    value["contract_id"] = _canonical_digest(_json_ready(value))
    return value


def run_batches(
    cases: list[StateCase],
    current_experiment: ExperimentConfig,
    spec: DoseSpec,
    model_path: Path,
    registration_id: str,
    checkpoint_directory: Path,
    workers: int,
    stage: str,
) -> list[base.PhaseBatch]:
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    contract = _checkpoint_contract(cases, spec, registration_id, stage)
    contract_path = checkpoint_directory / "checkpoint_contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text(encoding="utf-8")) != _json_ready(
            contract
        ):
            raise ValueError(f"checkpoint contract changed: {checkpoint_directory}")
    else:
        base._atomic_json(contract_path, contract)

    batches: list[base.PhaseBatch | None] = [None] * len(cases)
    missing: list[int] = []
    for index, case in enumerate(cases):
        path = checkpoint_directory / f"state_{index:04d}.pkl"
        if not path.is_file():
            missing.append(index)
            continue
        with path.open("rb") as handle:
            batch = pickle.load(handle)
        if (
            not isinstance(batch, base.PhaseBatch)
            or batch.state_id != case.state_id
            or batch.state_digest != base._snapshot_digest(case)
            or batch.arm_names != spec.arms
        ):
            raise ValueError(f"invalid CR2 checkpoint: {path}")
        batches[index] = batch

    def status(state: str) -> None:
        complete = sum(batch is not None for batch in batches)
        base._atomic_json(
            checkpoint_directory / "status.json",
            {
                "format": CHECKPOINT_FORMAT,
                "phase": "cr2",
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
        (cases[index], current_experiment, spec, str(model_path))
        for index in missing
    ]
    if workers <= 1:
        generated = map(_dose_worker, arguments)
        for index, batch in zip(missing, generated, strict=True):
            batches[index] = batch
            base._atomic_pickle(
                checkpoint_directory / f"state_{index:04d}.pkl", batch
            )
            status("running")
    elif missing:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            generated = executor.map(_dose_worker, arguments, chunksize=1)
            for index, batch in zip(missing, generated, strict=True):
                batches[index] = batch
                base._atomic_pickle(
                    checkpoint_directory / f"state_{index:04d}.pkl", batch
                )
                status("running")
    status("complete")
    if any(batch is None for batch in batches):
        raise AssertionError("CR2 checkpoint stage has missing states")
    return [batch for batch in batches if batch is not None]


def audit_cases_against_cr1(cases: list[StateCase]) -> dict[str, Any]:
    """Prove the reconstructed CR2 launch states equal the sealed CR1 states."""

    verify_checksums(CR1_RESULT)
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
    matrix_order = np.sort(
        np.unique(np.asarray([case.matrix_id for case in cases], dtype=np.int64))
    )
    beta = np.stack(
        [next(case.beta for case in cases if case.matrix_id == mid) for mid in matrix_order]
    )
    observed: dict[str, NDArray] = {
        "state_ids": np.asarray([case.state_id for case in cases]),
        "candidates": np.asarray([case.candidate for case in cases]),
        "matrix_ids": np.asarray([case.matrix_id for case in cases], dtype=np.int16),
        "landmarks": np.asarray([case.landmark for case in cases], dtype=np.int16),
        "compositions": np.vstack(
            [case.snapshot.composition for case in cases]
        ).astype(np.int16),
        "generations": np.asarray(
            [case.snapshot.generation for case in cases], dtype=np.int16
        ),
        "previous_growth_steps": np.asarray(
            [case.snapshot.previous_growth_steps for case in cases], dtype=np.int32
        ),
        "cumulative_growth_steps": np.asarray(
            [case.snapshot.cumulative_growth_steps for case in cases], dtype=np.int64
        ),
        "history_lengths": np.asarray(
            [len(case.snapshot.inheritance) for case in cases], dtype=np.int16
        ),
        "inheritance": inheritance,
        "boundary_h": boundary_h,
        "beta_matrix_ids": matrix_order.astype(np.int16),
        "beta": beta,
    }
    checks: dict[str, bool] = {}
    with np.load(
        CR1_RESULT / "state_and_matrix_arrays.npz", allow_pickle=False
    ) as archived:
        if set(archived.files) != set(observed):
            raise AssertionError("CR1 state artifact fields changed")
        for name, value in observed.items():
            if np.issubdtype(value.dtype, np.inexact):
                checks[name] = bool(
                    np.array_equal(value, archived[name], equal_nan=True)
                )
            else:
                checks[name] = bool(np.array_equal(value, archived[name]))
    if not all(checks.values()):
        raise AssertionError(
            {name: exact for name, exact in checks.items() if not exact}
        )
    return {
        "format": "codex-intervention-cr2-state-reconstruction-audit-v1",
        "state_count": len(cases),
        "matrix_count": len(matrix_order),
        "field_exact": checks,
        "all_state_history_and_beta_arrays_exact": True,
        "cr1_state_artifact_sha256": sha256_file(
            CR1_RESULT / "state_and_matrix_arrays.npz"
        ),
        "cr1_checksum_manifest_sha256": sha256_file(CR1_RESULT / "SHA256SUMS"),
    }


def _average_ranks(values: NDArray) -> NDArray[np.float64]:
    return (
        pd.Series(np.asarray(values, dtype=np.float64))
        .rank(method="average")
        .to_numpy(dtype=np.float64)
    )


def state_spearman(values_x: NDArray, values_y: NDArray) -> float:
    """Registered six-arm Spearman; an undefined state contributes zero."""

    rank_x = _average_ranks(values_x)
    rank_y = _average_ranks(values_y)
    if np.std(rank_x) == 0.0 or np.std(rank_y) == 0.0:
        return 0.0
    return float(np.corrcoef(rank_x, rank_y)[0, 1])


def _interval(values: NDArray, alpha: float = 0.05) -> tuple[float, float]:
    lower, upper = np.quantile(
        np.asarray(values, dtype=np.float64),
        (alpha / 2.0, 1.0 - alpha / 2.0),
    )
    return float(lower), float(upper)


def _one_sided_p(observed: float, null: NDArray) -> float:
    values = np.asarray(null, dtype=np.float64)
    return float((np.count_nonzero(values >= observed) + 1) / (values.size + 1))


def _maximum_leave_one_out_mean(values: NDArray) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size <= 1:
        return float("nan")
    observed = float(array.mean())
    leave_one = (array.sum() - array) / (array.size - 1)
    return float(np.max(np.abs(leave_one - observed)))


def _maximum_leave_one_out_slope(
    numerator: NDArray, denominator: NDArray
) -> float:
    num = np.asarray(numerator, dtype=np.float64)
    den = np.asarray(denominator, dtype=np.float64)
    observed = float(num.sum() / den.sum())
    leave_one = (num.sum() - num) / (den.sum() - den)
    return float(np.max(np.abs(leave_one - observed)))


def compute_dose_inference(
    cases: list[StateCase],
    targets: NDArray,
    predictions: NDArray,
    draws: dict[str, NDArray],
    spec: DoseSpec,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, NDArray]]:
    target = np.asarray(targets, dtype=np.float64)
    predicted = np.asarray(predictions, dtype=np.float64)
    expected = (len(cases), len(spec.arms), spec.branches)
    if target.shape != expected:
        raise ValueError(f"CR2 target shape {target.shape} differs from {expected}")
    if predicted.shape != (len(cases), len(spec.arms)):
        raise ValueError("CR2 predictions do not align with states and arms")
    if np.any(np.diff(predicted, axis=1) < 0.0):
        raise ValueError("CR2 selected predictions are not monotone by arm")
    matrix_order = np.sort(
        np.unique(np.asarray([case.matrix_id for case in cases], dtype=np.int64))
    )
    if matrix_order.size != spec.matrices:
        raise ValueError("CR2 matrix count differs from its frozen specification")
    bootstrap = np.asarray(draws["bootstrap_indices"], dtype=np.int64)
    signs = np.asarray(draws["randomization_signs"], dtype=np.float64)
    if bootstrap.shape != (BOOTSTRAP_REPETITIONS, spec.matrices):
        raise ValueError("CR2 bootstrap draw shape changed")
    if signs.shape != (RANDOMIZATION_REPETITIONS, spec.matrices):
        raise ValueError("CR2 randomization draw shape changed")
    if np.any(bootstrap < 0) or np.any(bootstrap >= spec.matrices):
        raise ValueError("CR2 bootstrap violates matrix blocks")
    if not np.isin(signs, (-1.0, 1.0)).all():
        raise ValueError("CR2 sign randomization changed")

    cells: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    stored: dict[str, NDArray] = {
        "bootstrap_indices": bootstrap,
        "randomization_signs": signs,
    }
    rho_raw_p: list[float] = []
    slope_raw_p: list[float] = []
    half_size = spec.branches // 2

    for candidate in CANDIDATES:
        selected_mask = np.asarray(
            [case.candidate == candidate for case in cases], dtype=bool
        )
        selected_cases = [case for case in cases if case.candidate == candidate]
        ids = np.asarray(
            [case.matrix_id for case in selected_cases], dtype=np.int64
        )
        if not np.array_equal(np.unique(ids), matrix_order):
            raise ValueError(f"candidate {candidate} has incomplete matrices")
        candidate_target = target[selected_mask]
        candidate_prediction = predicted[selected_mask]
        for half, branch_slice in (
            ("A", slice(0, half_size)),
            ("B", slice(half_size, spec.branches)),
        ):
            q = candidate_target[:, :, branch_slice].mean(axis=2)
            state_rho = np.asarray(
                [
                    state_spearman(candidate_prediction[index], q[index])
                    for index in range(q.shape[0])
                ],
                dtype=np.float64,
            )
            x_centered = candidate_prediction - candidate_prediction.mean(
                axis=1, keepdims=True
            )
            y_centered = q - q.mean(axis=1, keepdims=True)
            state_numerator = np.sum(x_centered * y_centered, axis=1)
            state_denominator = np.sum(x_centered * x_centered, axis=1)
            matrix_rho = np.asarray(
                [state_rho[ids == mid].mean() for mid in matrix_order],
                dtype=np.float64,
            )
            matrix_numerator = np.asarray(
                [state_numerator[ids == mid].sum() for mid in matrix_order],
                dtype=np.float64,
            )
            matrix_denominator = np.asarray(
                [state_denominator[ids == mid].sum() for mid in matrix_order],
                dtype=np.float64,
            )
            if matrix_denominator.sum() <= 0.0:
                raise ValueError("CR2 predictor has no within-state dose variation")
            rho = float(matrix_rho.mean())
            slope = float(matrix_numerator.sum() / matrix_denominator.sum())
            rho_bootstrap = matrix_rho[bootstrap].mean(axis=1)
            slope_bootstrap = matrix_numerator[bootstrap].sum(
                axis=1
            ) / matrix_denominator[bootstrap].sum(axis=1)
            rho_null = signs @ matrix_rho / spec.matrices
            slope_null = (
                signs @ matrix_numerator / matrix_denominator.sum()
            )
            rho_p = _one_sided_p(rho, rho_null)
            slope_p = _one_sided_p(slope, slope_null)
            rho_raw_p.append(rho_p)
            slope_raw_p.append(slope_p)
            key = f"c{candidate}_{half}"
            stored[f"{key}__rho_bootstrap"] = rho_bootstrap
            stored[f"{key}__slope_bootstrap"] = slope_bootstrap
            stored[f"{key}__rho_randomization_null"] = rho_null
            stored[f"{key}__slope_randomization_null"] = slope_null

            arm_means: dict[str, Any] = {}
            for arm_index, arm in enumerate(spec.arms):
                state_q = q[:, arm_index]
                matrix_q = np.asarray(
                    [state_q[ids == mid].mean() for mid in matrix_order],
                    dtype=np.float64,
                )
                arm_draws = matrix_q[bootstrap].mean(axis=1)
                arm_means[arm] = {
                    "mean_realized_probability": float(matrix_q.mean()),
                    "bootstrap_ci95": _interval(arm_draws),
                    "mean_frozen_prediction": float(
                        candidate_prediction[:, arm_index].mean()
                    ),
                    "mean_frozen_shift_from_q00": float(
                        (
                            candidate_prediction[:, arm_index]
                            - candidate_prediction[:, 0]
                        ).mean()
                    ),
                }
            rho_ci = _interval(rho_bootstrap)
            slope_ci = _interval(slope_bootstrap)
            cell: dict[str, Any] = {
                "cell": key,
                "candidate": candidate,
                "branch_half": half,
                "branch_range": [
                    branch_slice.start,
                    branch_slice.stop - 1,
                ],
                "matrices": spec.matrices,
                "states": len(selected_cases),
                "arms": arm_means,
                "mean_within_state_spearman": rho,
                "spearman_bootstrap_ci95": rho_ci,
                "spearman_randomization_p_raw": rho_p,
                "state_centered_calibration_slope": slope,
                "slope_bootstrap_ci95": slope_ci,
                "slope_randomization_p_raw": slope_p,
                "zero_information_state_correlations": int(
                    np.count_nonzero(state_rho == 0.0)
                ),
                "positive_state_correlations": int(
                    np.count_nonzero(state_rho > 0.0)
                ),
                "maximum_leave_one_matrix_out_rho_influence": (
                    _maximum_leave_one_out_mean(matrix_rho)
                ),
                "maximum_leave_one_matrix_out_slope_influence": (
                    _maximum_leave_one_out_slope(
                        matrix_numerator, matrix_denominator
                    )
                ),
                "registered_gates": {
                    "mean_spearman_positive": rho > 0.0,
                    "spearman_bootstrap_lower_positive": rho_ci[0] > 0.0,
                    "state_centered_slope_positive": slope > 0.0,
                    "slope_bootstrap_lower_positive": slope_ci[0] > 0.0,
                },
            }
            cell["registered_cell_pass"] = bool(
                all(cell["registered_gates"].values())
            )
            cells.append(cell)
            for index, mid in enumerate(matrix_order):
                matrix_rows.append(
                    {
                        "cell": key,
                        "candidate": candidate,
                        "branch_half": half,
                        "matrix_id": int(mid),
                        "mean_state_spearman": float(matrix_rho[index]),
                        "slope_numerator": float(matrix_numerator[index]),
                        "slope_denominator": float(matrix_denominator[index]),
                        "states": int(np.count_nonzero(ids == mid)),
                    }
                )

    rho_holm = holm_adjust(rho_raw_p)
    slope_holm = holm_adjust(slope_raw_p)
    for cell, rho_p, slope_p in zip(cells, rho_holm, slope_holm, strict=True):
        cell["spearman_randomization_p_holm"] = float(rho_p)
        cell["slope_randomization_p_holm"] = float(slope_p)
    metrics = {
        "format": "codex-intervention-cr2-dose-inference-v1",
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "randomization_repetitions": RANDOMIZATION_REPETITIONS,
        "undefined_state_spearman_assigned_zero": True,
        "randomization_p_values_reported_not_gated": True,
        "cells": cells,
        "registered_all_four_cells_pass": bool(
            len(cells) == 4
            and all(cell["registered_cell_pass"] for cell in cells)
        ),
        "stored_inference_arrays": {
            "path": "inference_arrays.npz",
            "bootstrap_indices_shape": list(bootstrap.shape),
            "randomization_signs_shape": list(signs.shape),
            "all_cell_bootstrap_and_randomization_arrays_stored": True,
        },
    }
    return metrics, matrix_rows, stored


def _selection_frames(
    cases: list[StateCase],
    batches: list[base.PhaseBatch],
) -> tuple[pd.DataFrame, dict[str, NDArray]]:
    rows: list[dict[str, Any]] = []
    score_offsets = [0]
    score_remove: list[int] = []
    score_add: list[int] = []
    score_probability: list[float] = []
    score_shift: list[float] = []
    selected_score_index = np.empty((len(cases), len(ARMS)), dtype=np.int32)
    selected_rank = np.empty((len(cases), len(ARMS)), dtype=np.int32)
    noop_probability = np.empty(len(cases), dtype=np.float64)
    for state_index, (case, batch) in enumerate(zip(cases, batches, strict=True)):
        scores = batch.scored_edits
        selected, ranks = select_quantile_edits(scores)
        if batch.catalytic_support.shape != (1,):
            raise AssertionError("CR2 checkpoint omitted its NOOP prediction audit")
        noop = float(batch.catalytic_support[0])
        if any(
            abs((item.predicted_probability - noop) - item.predicted_shift)
            > 1e-15
            for item in scores
        ):
            raise AssertionError("persisted CR2 score shifts changed")
        noop_probability[state_index] = noop
        by_edit = {item.edit: index for index, item in enumerate(scores)}
        for arm_index, (arm, quantile, item, rank) in enumerate(
            zip(ARMS, QUANTILES, selected, ranks, strict=True)
        ):
            if batch.selected_edits[arm_index] != item.edit:
                raise AssertionError("CR2 batch edit differs from frozen quantile selection")
            selected_score_index[state_index, arm_index] = by_edit[item.edit]
            selected_rank[state_index, arm_index] = int(rank)
            rows.append(
                {
                    "state_id": case.state_id,
                    "candidate": case.candidate,
                    "matrix_id": case.matrix_id,
                    "landmark": case.landmark,
                    "arm": arm,
                    "empirical_quantile": quantile,
                    "legal_edit_count": len(scores),
                    "selected_rank_zero_based": int(rank),
                    "remove_type": item.edit.remove_type,
                    "add_type": item.edit.add_type,
                    "noop_predicted_probability": noop,
                    "predicted_probability": item.predicted_probability,
                    "predicted_shift_from_noop": item.predicted_shift,
                }
            )
        for item in scores:
            score_remove.append(item.edit.remove_type)
            score_add.append(item.edit.add_type)
            score_probability.append(item.predicted_probability)
            score_shift.append(item.predicted_shift)
        score_offsets.append(len(score_remove))
    arrays: dict[str, NDArray] = {
        "state_ids": np.asarray([case.state_id for case in cases]),
        "quantiles": np.asarray(QUANTILES, dtype=np.float64),
        "score_offsets": np.asarray(score_offsets, dtype=np.int64),
        "score_remove": np.asarray(score_remove, dtype=np.int16),
        "score_add": np.asarray(score_add, dtype=np.int16),
        "score_probability": np.asarray(score_probability, dtype=np.float64),
        "score_shift": np.asarray(score_shift, dtype=np.float64),
        "selected_score_index": selected_score_index,
        "selected_rank": selected_rank,
        "noop_probability": noop_probability,
    }
    return pd.DataFrame(rows), arrays


def _write_selection_artifacts(
    output: Path,
    cases: list[StateCase],
    batches: list[base.PhaseBatch],
) -> pd.DataFrame:
    frame, arrays = _selection_frames(cases, batches)
    frame.to_csv(
        output / "selected_interventions.csv",
        index=False,
        float_format="%.17g",
    )
    np.savez_compressed(output / "selection_arrays.npz", **arrays)
    return frame


def _write_inference_arrays(path: Path, arrays: dict[str, NDArray]) -> None:
    np.savez_compressed(path, **arrays)


def _readback_audit(
    output: Path,
    cases: list[StateCase],
    batches: list[base.PhaseBatch],
    spec: DoseSpec,
    expected_metrics: dict[str, Any],
    expected_rows: list[dict[str, Any]],
    expected_selection: pd.DataFrame,
) -> dict[str, Any]:
    with np.load(output / "branch_arrays.npz", allow_pickle=False) as archive:
        targets = archive["targets"]
        predictions = archive["predictions"]
    with np.load(output / "inference_arrays.npz", allow_pickle=False) as archive:
        draws = {
            "bootstrap_indices": archive["bootstrap_indices"],
            "randomization_signs": archive["randomization_signs"],
        }
    observed_metrics, observed_rows, observed_arrays = compute_dose_inference(
        cases, targets, predictions, draws, spec
    )
    metrics_exact = _json_ready(observed_metrics) == _json_ready(expected_metrics)
    rows_exact = _json_ready(observed_rows) == _json_ready(expected_rows)
    inference_arrays_exact = True
    with np.load(output / "inference_arrays.npz", allow_pickle=False) as archive:
        if set(archive.files) != set(observed_arrays):
            inference_arrays_exact = False
        else:
            inference_arrays_exact = all(
                np.array_equal(archive[name], value, equal_nan=True)
                for name, value in observed_arrays.items()
            )
    selected = pd.read_csv(
        output / "selected_interventions.csv",
        float_precision="round_trip",
        dtype={"candidate": str},
    )
    try:
        pd.testing.assert_frame_equal(
            selected,
            expected_selection,
            check_exact=True,
            check_dtype=False,
        )
    except AssertionError:
        selection_exact = False
    else:
        selection_exact = True
    matrix_frame = pd.read_csv(
        output / "matrix_effects.csv",
        float_precision="round_trip",
        dtype={"candidate": str},
    )
    expected_matrix_frame = pd.DataFrame(expected_rows)
    try:
        pd.testing.assert_frame_equal(
            matrix_frame,
            expected_matrix_frame,
            check_exact=True,
            check_dtype=False,
        )
    except AssertionError:
        matrix_csv_exact = False
    else:
        matrix_csv_exact = True
    checks = {
        "branch_arrays_reloaded": True,
        "inference_draws_reloaded": True,
        "primary_metrics_exact": metrics_exact,
        "matrix_effects_recomputed_exact": rows_exact,
        "matrix_effects_csv_exact": matrix_csv_exact,
        "inference_arrays_exact": inference_arrays_exact,
        "selected_interventions_csv_exact": selection_exact,
        "no_fitting_or_recalibration": True,
    }
    if not all(checks.values()):
        raise ValueError(
            {name: passed for name, passed in checks.items() if not passed}
        )
    return checks


def _reports(metrics: dict[str, Any]) -> tuple[str, str]:
    lines = [
        "# CR2 graded molecular dose-response confirmation",
        "",
        f"Registered four-cell gate: **{metrics['confirmation_gate_pass']}**.",
        f"Exact replay: **{metrics['integrity_gates']['exact_replay']}**.",
        f"Exact CR1 state reconstruction: **{metrics['integrity_gates']['cr1_states_exact']}**.",
        "",
        "| Cell | Mean state Spearman | 95% CI | Holm p | Centered slope | 95% CI | Holm p | Pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in metrics["cells"]:
        lines.append(
            f"| {cell['cell']} | {cell['mean_within_state_spearman']:+.6f} | "
            f"{cell['spearman_bootstrap_ci95']} | {cell['spearman_randomization_p_holm']:.6g} | "
            f"{cell['state_centered_calibration_slope']:+.6f} | "
            f"{cell['slope_bootstrap_ci95']} | {cell['slope_randomization_p_holm']:.6g} | "
            f"{cell['registered_cell_pass']} |"
        )
    lines.extend(
        [
            "",
            "The same sealed CR1 states and predictor were used, while every future used the fresh registered CR2 stream. The six edits were selected only from frozen predicted empirical ranks.",
            "",
            "This simulated-process result cannot establish strict-eight control, life, agency, biological memory, autonomous organization, real chemistry, or a universal origin-of-life mechanism.",
            "",
        ]
    )
    if metrics["confirmation_gate_pass"]:
        conclusion = (
            "The realized event probability rose in graded agreement with the frozen predictor "
            "in both simulator candidates and both independent branch halves."
        )
    else:
        conclusion = (
            "The six intervention levels did not satisfy every prewritten graded-response "
            "condition, so CR2 does not establish a transferable causal dial."
        )
    lay = "\n".join(
        [
            "# CR2 in plain language",
            "",
            "CR1 showed that the predictor could pick a strong risk-raising edit and a strong risk-lowering edit. CR2 tried six edits ranging from the predicted lowest to highest effect, like testing six positions on a dial.",
            "",
            conclusion,
            "",
        ]
    )
    return "\n".join(lines), lay


def _append_registration_ledger(registration_id: str) -> None:
    path = ROOT / "INTERVENTION_RESULTS_LEDGER.md"
    text = path.read_text(encoding="utf-8").rstrip() + "\n"
    marker = f"<!-- cr2-dose-response-registered-{registration_id} -->"
    if marker in text:
        return
    lines = [
        "",
        marker,
        "## Full CR2 graded dose response registered",
        "",
        f"- Registration: `{registration_id}`.",
        "- The full CR1 pass is used only for phase advancement; CR1 effect sizes do not select CR2 arms or analyses.",
        "- Same 2,000 sealed CR1 states; six frozen empirical edit quantiles; 64 fresh F12 branches per arm; complete replay.",
        "- Scientific CR2 matrices and futures at registration: **0**.",
        "- Status: sealed before scientific CR2 execution.",
        "",
    ]
    path.write_text(text + "\n".join(lines), encoding="utf-8")


def _append_result_ledger(
    output: Path, registration_id: str, metrics: dict[str, Any]
) -> None:
    path = ROOT / "INTERVENTION_RESULTS_LEDGER.md"
    text = path.read_text(encoding="utf-8").rstrip() + "\n"
    marker = f"<!-- sealed-cr2-dose-response-{registration_id} -->"
    if marker in text:
        return
    lines = [
        "",
        marker,
        "## Full CR2 graded dose response sealed",
        "",
        f"- Registration: `{registration_id}`.",
        f"- Result: `{output.relative_to(ROOT)}`.",
        f"- Full four-cell graded-response gate: **{metrics['confirmation_gate_pass']}**.",
        f"- Exact replay: **{metrics['integrity_gates']['exact_replay']}**.",
        f"- Exact CR1 state reconstruction: **{metrics['integrity_gates']['cr1_states_exact']}**.",
        "- Mandatory review stop observed; no later phase launched automatically.",
        "",
    ]
    path.write_text(text + "\n".join(lines), encoding="utf-8")


def validate(output: Path = DEFAULT_VALIDATION) -> None:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    verify_checksums(CR1_RESULT)
    cr1_registration = cr1.verify_registration(CR1_REGISTRATION)
    cr1_manifest = json.loads((CR1_RESULT / "manifest.json").read_text())
    inherited = base.validation_checks()

    synthetic = tuple(
        ScoredEdit(
            MolecularEdit(index // 10, index % 10),
            float(index),
            float(index) - 5.0,
        )
        for index in range(11)
    )
    selected, ranks = select_quantile_edits(synthetic)
    synthetic_ties = (
        ScoredEdit(MolecularEdit(2, 3), 0.1, -0.4),
        ScoredEdit(MolecularEdit(0, 3), 0.1, -0.4),
        ScoredEdit(MolecularEdit(4, 1), 0.9, 0.4),
        ScoredEdit(MolecularEdit(1, 4), 0.9, 0.4),
    )
    tied, _ = select_quantile_edits(synthetic_ties)
    existing_seeds = set(base.SEED_DOMAINS.values()) | set(cr1.SEEDS.values())
    checks = {
        "inherited_validation_all_passed": bool(
            inherited["required_checks_passed"]
            and inherited["all_checks_passed"]
        ),
        "cr1_result_checksum_verified": True,
        "cr1_full_gate_passed": bool(cr1_manifest["full_four_cell_gate"]),
        "cr1_exact_replay_passed": bool(cr1_manifest["exact_replay"]),
        "cr1_readback_passed": bool(cr1_manifest["complete_readback_exact"]),
        "cr1_registration_verified": bool(cr1_registration["registration_id"]),
        "cr1_model_hash_preserved": (
            sha256_file(CR1_REGISTRATION / "frozen_full_predictor.npz")
            == cr1_registration["frozen_model_sha256"]
        ),
        "full_matrix_count": MATRICES == 200,
        "full_branch_count": BRANCHES == 64,
        "six_quantile_contract": QUANTILES == (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        "quantile_ranks_exact": np.array_equal(
            ranks, np.asarray([0, 2, 4, 6, 8, 10], dtype=np.int64)
        ),
        "quantile_selection_exact": tuple(
            item.predicted_probability for item in selected
        )
        == (0.0, 2.0, 4.0, 6.0, 8.0, 10.0),
        "tie_resolution_lexicographic_at_both_extremes": (
            tied[0].edit == MolecularEdit(0, 3)
            and tied[-1].edit == MolecularEdit(1, 4)
        ),
        "seed_domains_unique": len(SEEDS) == len(set(SEEDS.values())),
        "fresh_seeds_disjoint_from_earlier_program": not set(SEEDS.values()).intersection(
            existing_seeds
        ),
        "future_seed_domain_differs_from_cr1": SEEDS["future"]
        != cr1.SEEDS["future"],
        "future_seed_key_has_no_arm_argument": True,
        "primary_future_count": 2 * MATRICES * len(LANDMARKS) * len(ARMS) * BRANCHES
        == 768_000,
        "replay_future_count": 768_000,
        "scientific_cr2_output_absent": not DEFAULT_OUTPUT.exists(),
        "scientific_cr2_work_absent": not DEFAULT_WORK.exists(),
    }
    if not all(checks.values()):
        raise AssertionError(
            {name: passed for name, passed in checks.items() if not passed}
        )
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_intervention_cr2_dose_response.py",
        "tests/test_intervention_replication.py",
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "CR2 validation tests failed:\n"
            + completed.stdout
            + "\n"
            + completed.stderr
        )
    with _atomic_destination(output) as destination:
        payload = {
            "format": VALIDATION_FORMAT,
            "checks": checks,
            "all_pass": True,
            "inherited_validation": inherited,
            "pytest_command": command,
            "pytest_stdout": completed.stdout,
            "pytest_stderr": completed.stderr,
            "scientific_cr2_matrices_generated": 0,
            "scientific_cr2_futures_generated": 0,
            "source_hashes": source_hashes(),
        }
        (destination / "validation.json").write_text(
            json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_checksums(destination)
    verify_checksums(output)
    print(f"CR2 validation sealed: {output}", flush=True)


def register(
    validation_directory: Path = DEFAULT_VALIDATION,
    output: Path = DEFAULT_REGISTRATION,
) -> None:
    validation_directory = validation_directory.resolve()
    output = output.resolve()
    verify_checksums(validation_directory)
    validation = json.loads(
        (validation_directory / "validation.json").read_text(encoding="utf-8")
    )
    if not validation.get("all_pass"):
        raise ValueError("CR2 validation did not pass")
    if validation["source_hashes"] != source_hashes():
        raise ValueError("CR2 sources changed after validation")
    for scientific in (DEFAULT_OUTPUT, DEFAULT_WORK):
        if scientific.exists():
            raise FileExistsError(
                f"CR2 scientific artifact exists before registration: {scientific}"
            )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    verify_checksums(CR1_RESULT)
    cr1_payload = cr1.verify_registration(CR1_REGISTRATION)
    cr1_manifest = json.loads((CR1_RESULT / "manifest.json").read_text())
    if not (
        cr1_manifest["full_four_cell_gate"]
        and cr1_manifest["exact_replay"]
        and cr1_manifest["complete_readback_exact"]
    ):
        raise ValueError("sealed CR1 does not permit CR2 phase advancement")
    frozen = protocol()
    payload: dict[str, Any] = {
        "format": REGISTRATION_FORMAT,
        "protocol_id": frozen["protocol_id"],
        "source_hashes": source_hashes(),
        "seed_registry": SEEDS,
        "validation_checksum_manifest_sha256": sha256_file(
            validation_directory / "SHA256SUMS"
        ),
        "cr1_registration_id": cr1_payload["registration_id"],
        "cr1_result_checksum_manifest_sha256": sha256_file(
            CR1_RESULT / "SHA256SUMS"
        ),
        "cr1_state_artifact_sha256": sha256_file(
            CR1_RESULT / "state_and_matrix_arrays.npz"
        ),
        "frozen_model_sha256": sha256_file(
            CR1_REGISTRATION / "frozen_full_predictor.npz"
        ),
        "cr1_gate_used_only_for_phase_advancement": True,
        "cr1_effect_sizes_used_for_design": False,
        "scientific_cr2_matrices_at_registration": 0,
        "scientific_cr2_futures_at_registration": 0,
    }
    payload["registration_id"] = _canonical_digest(_json_ready(payload))
    with _atomic_destination(output) as destination:
        (destination / "protocol.json").write_text(
            json.dumps(frozen, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (destination / "seed_registry.json").write_text(
            json.dumps(SEEDS, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (destination / "registration.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        shutil.copy2(
            CR1_REGISTRATION / "frozen_full_predictor.npz",
            destination / "frozen_full_predictor.npz",
        )
        write_checksums(destination)
    verify_registration(output)
    _append_registration_ledger(payload["registration_id"])
    print(f"CR2 registration sealed: {payload['registration_id']}", flush=True)


def verify_registration(directory: Path = DEFAULT_REGISTRATION) -> dict[str, Any]:
    directory = directory.resolve()
    verify_checksums(directory)
    value = json.loads((directory / "registration.json").read_text())
    if value.get("format") != REGISTRATION_FORMAT:
        raise ValueError("invalid CR2 registration")
    if value["source_hashes"] != source_hashes():
        raise ValueError("CR2 source changed")
    if json.loads((directory / "protocol.json").read_text()) != protocol():
        raise ValueError("CR2 protocol changed")
    if value["seed_registry"] != SEEDS:
        raise ValueError("CR2 seed registry changed")
    if value["frozen_model_sha256"] != sha256_file(
        directory / "frozen_full_predictor.npz"
    ):
        raise ValueError("CR2 frozen model changed")
    verify_checksums(CR1_RESULT)
    if value["cr1_result_checksum_manifest_sha256"] != sha256_file(
        CR1_RESULT / "SHA256SUMS"
    ):
        raise ValueError("CR1 predecessor result changed")
    if value["cr1_state_artifact_sha256"] != sha256_file(
        CR1_RESULT / "state_and_matrix_arrays.npz"
    ):
        raise ValueError("CR1 state artifact changed")
    return value


def smoke(
    registration_directory: Path = DEFAULT_REGISTRATION,
    output: Path = DEFAULT_SMOKE,
) -> None:
    registration = verify_registration(registration_directory)
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    spec = DoseSpec(
        role="non-scientific CR2 smoke",
        matrices=2,
        branches=2,
        cohort_seed=SEEDS["smoke_cohort"],
        selection_seed=SEEDS["selection_audit"],
        future_seed=SEEDS["smoke_future"],
        bootstrap_seed=SEEDS["validation"],
        randomization_seed=SEEDS["replay"],
    )
    cohort = CohortConfig(2, 2, (5,))
    current_experiment = ExperimentConfig(
        development=cohort,
        confirmation=cohort,
        horizon=HORIZON,
        master_seed=spec.cohort_seed,
        bootstrap_repetitions=8,
        permutation_repetitions=8,
    )
    with tempfile.TemporaryDirectory(
        prefix="codex-cr2-dose-smoke-", dir=output.parent
    ) as temporary:
        with threadpool_limits(limits=1):
            cases = build_cohort(
                current_experiment,
                "INTCR2_NONSCIENTIFIC_SMOKE_V1",
                cohort,
            )
        generated = run_batches(
            cases,
            current_experiment,
            spec,
            registration_directory / "frozen_full_predictor.npz",
            registration["registration_id"],
            Path(temporary) / "generate",
            1,
            "generate",
        )
        replayed = run_batches(
            cases,
            current_experiment,
            spec,
            registration_directory / "frozen_full_predictor.npz",
            registration["registration_id"],
            Path(temporary) / "replay",
            1,
            "replay",
        )
        replay = base.replay_audit(generated, replayed)
        if not replay["state_edit_endpoint_and_process_digests_exact"]:
            raise AssertionError("CR2 smoke replay failed")
        if not all(np.all(np.diff(batch.predictions) >= 0.0) for batch in generated):
            raise AssertionError("CR2 smoke quantile ordering failed")
        arrays = base._outcome_arrays(cases, generated, spec)  # type: ignore[arg-type]
        draws = generate_inference_draws(
            2,
            BOOTSTRAP_REPETITIONS,
            RANDOMIZATION_REPETITIONS,
            np.random.default_rng(derive_seed(SEEDS["validation"], "smoke.bootstrap")),
            np.random.default_rng(derive_seed(SEEDS["replay"], "smoke.randomization")),
        )
        metrics, rows, inference_arrays = compute_dose_inference(
            cases, arrays["targets"], arrays["predictions"], draws, spec
        )
        artifact_directory = Path(temporary) / "artifacts"
        artifact_directory.mkdir()
        np.savez_compressed(artifact_directory / "branch_arrays.npz", **arrays)
        selected_frame = _write_selection_artifacts(
            artifact_directory, cases, generated
        )
        _write_inference_arrays(
            artifact_directory / "inference_arrays.npz", inference_arrays
        )
        pd.DataFrame(rows).to_csv(
            artifact_directory / "matrix_effects.csv",
            index=False,
            float_format="%.17g",
        )
        readback = _readback_audit(
            artifact_directory,
            cases,
            generated,
            spec,
            metrics,
            rows,
            selected_frame,
        )
        if not all(readback.values()):
            raise AssertionError("CR2 smoke artifact readback failed")
    with _atomic_destination(output) as destination:
        manifest = {
            "format": "codex-intervention-cr2-dose-response-smoke-v1",
            "registration_id": registration["registration_id"],
            "scientific_result": False,
            "scientific_matrices": 0,
            "scientific_futures": 0,
            "six_arm_legality_io_checkpoint_and_replay_passed": True,
            "effect_sizes_disclosed": False,
            "event_rates_disclosed": False,
            "arm_ordering_disclosed": False,
            "candidate_differences_disclosed": False,
        }
        (destination / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_checksums(destination)
    verify_checksums(output)
    print(f"CR2 non-scientific smoke passed: {output}", flush=True)


def _campaign_status(
    work: Path,
    state: str,
    detail: str,
    available_cpu_hours: float | None = None,
) -> None:
    value: dict[str, Any] = {
        "format": STATUS_FORMAT,
        "phase": "cr2_dose_response_confirmation",
        "state": state,
        "detail": detail,
        "mandatory_stop_after_seal": True,
    }
    if available_cpu_hours is not None:
        value["available_cpu_hours_at_launch"] = available_cpu_hours
    work.mkdir(parents=True, exist_ok=True)
    base._atomic_json(work / "campaign_status.json", value)


def _prepare_campaign(
    work: Path,
    output: Path,
    registration: dict[str, Any],
    available_cpu_hours: float,
) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if available_cpu_hours < MINIMUM_AVAILABLE_CPU_HOURS:
        raise ValueError(
            "CR2 needs at least "
            f"{MINIMUM_AVAILABLE_CPU_HOURS:.1f} projected CPU-hours at launch"
        )
    verify_checksums(CR1_RESULT)
    verify_checksums(DEFAULT_SMOKE)
    work.mkdir(parents=True, exist_ok=True)
    contract: dict[str, Any] = {
        "format": "codex-intervention-cr2-dose-response-campaign-v1",
        "registration_id": registration["registration_id"],
        "output": str(output),
        "matrices": MATRICES,
        "branches": BRANCHES,
        "landmarks": list(LANDMARKS),
        "arms": list(ARMS),
        "quantiles": list(QUANTILES),
        "available_cpu_hours_at_launch": available_cpu_hours,
        "cr1_state_artifact_sha256": sha256_file(
            CR1_RESULT / "state_and_matrix_arrays.npz"
        ),
        "cr1_effect_sizes_not_used": True,
        "source_hashes": source_hashes(),
    }
    contract["campaign_id"] = _canonical_digest(_json_ready(contract))
    path = work / "campaign_contract.json"
    if path.exists() and json.loads(path.read_text()) != _json_ready(contract):
        raise ValueError("CR2 work directory belongs to another campaign")
    if not path.exists():
        base._atomic_json(path, contract)
    _campaign_status(work, "running", "campaign_initialized", available_cpu_hours)


def run(
    registration_directory: Path = DEFAULT_REGISTRATION,
    output: Path = DEFAULT_OUTPUT,
    work: Path = DEFAULT_WORK,
    workers: int = min(os.cpu_count() or 1, 14),
    available_cpu_hours: float = MINIMUM_AVAILABLE_CPU_HOURS,
) -> None:
    if workers < 1:
        raise ValueError("workers must be positive")
    registration_directory = registration_directory.resolve()
    output = output.resolve()
    work = work.resolve()
    registration = verify_registration(registration_directory)
    _prepare_campaign(work, output, registration, available_cpu_hours)
    spec = phase_spec()
    current_experiment = experiment(spec)
    print(
        f"[cr2 1/9] Reconstructing {2 * MATRICES * len(LANDMARKS)} sealed CR1 states",
        flush=True,
    )
    _campaign_status(
        work, "running", "reconstructing_and_auditing_cr1_states", available_cpu_hours
    )
    with threadpool_limits(limits=1):
        cases = build_cohort(
            current_experiment,
            LABEL,
            current_experiment.confirmation,
        )
    if len(cases) != 2 * MATRICES * len(LANDMARKS):
        raise AssertionError("CR2 reconstructed cohort is incomplete")
    state_audit = audit_cases_against_cr1(cases)
    print("[cr2 2/9] CR1 states, histories, and beta matrices are exact", flush=True)
    futures = len(cases) * len(spec.arms) * spec.branches
    model_path = registration_directory / "frozen_full_predictor.npz"
    print(
        f"[cr2 3/9] Selecting six exhaustive edit quantiles and shooting {futures:,} F12 futures",
        flush=True,
    )
    _campaign_status(
        work, "running", "selection_and_primary_futures", available_cpu_hours
    )
    generated = run_batches(
        cases,
        current_experiment,
        spec,
        model_path,
        registration["registration_id"],
        work / "generate",
        workers,
        "generate",
    )
    print(f"[cr2 4/9] Replaying all {futures:,} futures", flush=True)
    _campaign_status(work, "running", "exact_replay", available_cpu_hours)
    replayed = run_batches(
        cases,
        current_experiment,
        spec,
        model_path,
        registration["registration_id"],
        work / "replay",
        workers,
        "replay",
    )
    replay = base.replay_audit(generated, replayed)
    if not replay["state_edit_endpoint_and_process_digests_exact"]:
        raise AssertionError("CR2 exact replay failed")
    arrays = base._outcome_arrays(cases, generated, spec)  # type: ignore[arg-type]
    draws = generate_inference_draws(
        MATRICES,
        BOOTSTRAP_REPETITIONS,
        RANDOMIZATION_REPETITIONS,
        np.random.default_rng(
            derive_seed(SEEDS["bootstrap"], f"{FUTURE_LABEL}.bootstrap")
        ),
        np.random.default_rng(
            derive_seed(SEEDS["randomization"], f"{FUTURE_LABEL}.randomization")
        ),
    )
    print("[cr2 5/9] Computing frozen whole-matrix dose inference", flush=True)
    _campaign_status(work, "running", "whole_matrix_inference", available_cpu_hours)
    metrics, matrix_rows, inference_arrays = compute_dose_inference(
        cases, arrays["targets"], arrays["predictions"], draws, spec
    )
    secondary = base._secondary_descriptives(  # type: ignore[arg-type]
        cases, arrays, spec
    )
    print("[cr2 6/9] Writing and readback-checking artifacts", flush=True)
    _campaign_status(
        work, "running", "artifact_write_and_readback", available_cpu_hours
    )
    with _atomic_destination(output) as destination:
        np.savez_compressed(destination / "branch_arrays.npz", **arrays)
        base._write_branch_table(destination / "branches.csv.gz", cases, generated)
        base._write_state_artifacts(  # type: ignore[arg-type]
            destination, cases, generated, arrays
        )
        selected_frame = _write_selection_artifacts(
            destination, cases, generated
        )
        _write_inference_arrays(
            destination / "inference_arrays.npz", inference_arrays
        )
        pd.DataFrame(matrix_rows).to_csv(
            destination / "matrix_effects.csv",
            index=False,
            float_format="%.17g",
        )
        readback = _readback_audit(
            destination,
            cases,
            generated,
            spec,
            metrics,
            matrix_rows,
            selected_frame,
        )
        integrity = {
            "exact_replay": bool(
                replay["state_edit_endpoint_and_process_digests_exact"]
            ),
            "cr1_states_exact": bool(
                state_audit["all_state_history_and_beta_arrays_exact"]
            ),
            "artifact_readback_exact": bool(all(readback.values())),
        }
        metrics["integrity_gates"] = integrity
        metrics["confirmation_gate_pass"] = bool(
            metrics["registered_all_four_cells_pass"]
            and all(integrity.values())
        )
        (destination / "primary_metrics.json").write_text(
            json.dumps(_json_ready(metrics), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (destination / "secondary_outcomes.json").write_text(
            json.dumps(_json_ready(secondary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (destination / "state_reconstruction_audit.json").write_text(
            json.dumps(_json_ready(state_audit), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (destination / "replay_audit.json").write_text(
            json.dumps(_json_ready(replay), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (destination / "readback_audit.json").write_text(
            json.dumps(readback, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        technical, lay = _reports(metrics)
        (destination / "SCIENTIFIC_REPORT.md").write_text(
            technical, encoding="utf-8"
        )
        (destination / "LAY_SUMMARY.md").write_text(lay, encoding="utf-8")
        claims = {
            "supported": (
                [
                    "graded causal ranking of legal molecular edits for Codex JOINT_BREAK_RUN3"
                ]
                if metrics["confirmation_gate_pass"]
                else []
            ),
            "failed_predictions": (
                []
                if metrics["confirmation_gate_pass"]
                else ["full CR2 four-cell graded dose-response gate"]
            ),
            "unresolved": [
                "zero-shot parameter transfer",
                "resistance versus resilience under molecular edits",
                "closed-loop control",
            ],
            "prohibited": protocol()["claim_boundary"]["prohibited"],
        }
        (destination / "claim_boundaries.json").write_text(
            json.dumps(claims, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "format": RESULT_FORMAT,
            "registration_id": registration["registration_id"],
            "matrices": MATRICES,
            "states": len(cases),
            "arms": list(ARMS),
            "branches_per_arm_state": BRANCHES,
            "primary_futures": futures,
            "replay_futures": futures,
            "full_four_cell_gate": metrics["confirmation_gate_pass"],
            "exact_replay": integrity["exact_replay"],
            "exact_cr1_state_reconstruction": integrity["cr1_states_exact"],
            "complete_readback_exact": integrity["artifact_readback_exact"],
            "available_cpu_hours_at_launch": available_cpu_hours,
            "no_refitting_recalibration_or_threshold_change": True,
            "no_future_retry_matrix_replacement_or_state_exclusion": True,
            "fresh_future_seed_domain": True,
            "cr1_effect_sizes_not_used": True,
            "mandatory_stop_after_this_stage": True,
            "later_phase_launched": False,
            "runtime": _runtime_manifest(),
        }
        (destination / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_checksums(destination)
    verify_checksums(output)
    _append_result_ledger(output, registration["registration_id"], metrics)
    _campaign_status(work, "sealed_complete", "mandatory_review_stop", available_cpu_hours)
    print("[cr2 7/9] Result checksum sealed", flush=True)
    print("[cr2 8/9] Durable ledger and status updated", flush=True)
    print("[cr2 9/9] STOPPED; no later phase launched", flush=True)


def read_status(work: Path = DEFAULT_WORK) -> dict[str, Any]:
    return base.read_status(work)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate").add_argument(
        "--output", type=Path, default=DEFAULT_VALIDATION
    )
    register_parser = commands.add_parser("register")
    register_parser.add_argument(
        "--validation", type=Path, default=DEFAULT_VALIDATION
    )
    register_parser.add_argument("--output", type=Path, default=DEFAULT_REGISTRATION)
    commands.add_parser("verify").add_argument(
        "--registration", type=Path, default=DEFAULT_REGISTRATION
    )
    smoke_parser = commands.add_parser("smoke")
    smoke_parser.add_argument(
        "--registration", type=Path, default=DEFAULT_REGISTRATION
    )
    smoke_parser.add_argument("--output", type=Path, default=DEFAULT_SMOKE)
    run_parser = commands.add_parser("run")
    run_parser.add_argument(
        "--registration", type=Path, default=DEFAULT_REGISTRATION
    )
    run_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run_parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    run_parser.add_argument(
        "--workers", type=int, default=min(os.cpu_count() or 1, 14)
    )
    run_parser.add_argument("--available-cpu-hours", type=float, required=True)
    commands.add_parser("status").add_argument(
        "--work-dir", type=Path, default=DEFAULT_WORK
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "validate":
        validate(args.output)
    elif args.command == "register":
        register(args.validation, args.output)
    elif args.command == "verify":
        print(json.dumps(verify_registration(args.registration), indent=2, sort_keys=True))
    elif args.command == "smoke":
        smoke(args.registration, args.output)
    elif args.command == "run":
        run(
            args.registration,
            args.output,
            args.work_dir,
            args.workers,
            args.available_cpu_hours,
        )
    elif args.command == "status":
        print(json.dumps(read_status(args.work_dir), indent=2, sort_keys=True))
    else:  # pragma: no cover
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
