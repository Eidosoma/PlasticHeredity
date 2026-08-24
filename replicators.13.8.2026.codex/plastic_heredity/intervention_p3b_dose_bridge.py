"""Prospectively registered P3b beta-surgery dose-and-contract bridge."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from threadpoolctl import threadpool_limits

from . import intervention_p3_inference_recovery as p3_recovery
from . import intervention_replication as base
from .config import CANDIDATES, CohortConfig, ExperimentConfig
from .experiment import StateCase, _json_ready, _runtime_manifest, build_cohort
from .intervention_core import BetaSurgery, FrozenFullPredictor, simulate_one_shot
from .intervention_metrics import (
    _bernoulli_scores,
    _bootstrap_means,
    _interval,
    _matrix_means,
    _maximum_leave_one_out_influence,
    _one_sided_sign_p,
    generate_inference_draws,
)
from .mechanistic import (
    _atomic_destination,
    _canonical_digest,
    sha256_file,
    verify_checksums,
    write_checksums,
)
from .mechanistic_metrics import holm_adjust
from .seeds import derive_seed


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = REPOSITORY_ROOT / "results_intervention_replication"
ORIGINAL_REGISTRATION = RESULT_ROOT / "registration"
P3_RESULT = RESULT_ROOT / "p3_cr4_beta_surgery_pilot"
P3_RECOVERY_AMENDMENT = RESULT_ROOT / "p3_inference_recovery_amendment"
DEFAULT_VALIDATION = RESULT_ROOT / "p3b_dose_bridge_validation"
DEFAULT_REGISTRATION = RESULT_ROOT / "p3b_dose_bridge_registration"
DEFAULT_OUTPUT = RESULT_ROOT / "p3b_beta_surgery_dose_bridge"
DEFAULT_WORK = RESULT_ROOT / ".p3b_dose_bridge_work"

DOCUMENT = "CODEX_INTERVENTION_P3B_DOSE_BRIDGE_PREREGISTRATION.md"
PROGRAM_FORMAT = "codex-intervention-p3b-dose-bridge-v1"
VALIDATION_FORMAT = "codex-intervention-p3b-dose-bridge-validation-v1"
REGISTRATION_FORMAT = "codex-intervention-p3b-dose-bridge-registration-v1"
RESULT_FORMAT = "codex-intervention-p3b-dose-bridge-result-v1"
CHECKPOINT_FORMAT = "codex-intervention-p3b-dose-bridge-checkpoint-v1"
LABEL = "INTP3B_DOSE_BRIDGE_V1"

EXPECTED_ORIGINAL_REGISTRATION_ID = (
    "f61e0340dcd8c9ae6b606c8133ca3d8fb1de2e13fe863719aa67b649e8b74531"
)
EXPECTED_P3_RECOVERY_AMENDMENT_ID = (
    "86e0cda5d5403fed3601fcd65110472ed76305ab9f5a1b139a0347abf238e185"
)

MATRICES = 80
BRANCHES = 32
LANDMARKS = (20, 35, 50, 60, 65, 80)
PRIMARY_LANDMARKS = (60,)
GENERALIZATION_LANDMARKS = (20, 35, 50, 65, 80)
HORIZON = 12
BOOTSTRAP_REPETITIONS = 4_096
RANDOMIZATION_REPETITIONS = 4_096
EQUIVALENCE_MARGIN = 0.025
SMALL_TIGHTEN_FACTOR = 1.05
SMALL_LOOSEN_FACTOR = 0.95
FABLE_TIGHTEN_FACTOR = 1.5
FABLE_LOOSEN_FACTOR = 1.0 / 1.5

ARMS = (
    "SMALL_LOOSEN",
    "SMALL_TIGHTEN",
    "SMALL_RANDOM_PP",
    "FABLE_LOOSEN",
    "FABLE_TIGHTEN",
    "FABLE_RANDOM_PP",
    "NOOP",
)

SOURCE_FILES = (
    DOCUMENT,
    "plastic_heredity/intervention_p3b_dose_bridge.py",
    "tests/test_intervention_p3b_dose_bridge.py",
)


def _seed_value(label: str) -> str:
    return hashlib.sha256(
        f"codex-clean-room-p3b-dose-bridge-v1::{label}".encode("utf-8")
    ).hexdigest()


SEED_DOMAINS = {
    name: _seed_value(name)
    for name in (
        "validation",
        "smoke_cohort",
        "smoke_selection",
        "smoke_future",
        "cohort",
        "selection",
        "future",
        "bootstrap",
        "randomization",
        "replay",
    )
}


@dataclass(frozen=True)
class BridgeSpec:
    phase: str = "p3b_dose_bridge"
    role: str = "Fable-strength beta-surgery replication and two-dose bridge"
    matrices: int = MATRICES
    branches: int = BRANCHES
    cohort_seed: str = SEED_DOMAINS["cohort"]
    selection_seed: str = SEED_DOMAINS["selection"]
    future_seed: str = SEED_DOMAINS["future"]
    bootstrap_seed: str = SEED_DOMAINS["bootstrap"]
    randomization_seed: str = SEED_DOMAINS["randomization"]
    arms: tuple[str, ...] = ARMS


def phase_spec() -> BridgeSpec:
    return BridgeSpec()


def _source_hashes() -> dict[str, str]:
    return {name: sha256_file(REPOSITORY_ROOT / name) for name in SOURCE_FILES}


def _present_block(
    composition: NDArray, beta: NDArray
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    values = np.asarray(composition)
    matrix = np.asarray(beta, dtype=np.float64)
    if values.ndim != 1 or matrix.shape != (values.size, values.size):
        raise ValueError("beta and composition dimensions differ")
    if not np.isfinite(matrix).all() or np.any(matrix <= 0.0):
        raise ValueError("beta surgery requires a finite positive matrix")
    present = np.flatnonzero(values > 0).astype(np.int64)
    if present.size < 2:
        raise ValueError("P3b surgery requires at least two present types")
    rows, columns = np.meshgrid(present, present, indexing="ij")
    flat = np.ravel_multi_index((rows.ravel(), columns.ravel()), matrix.shape)
    before = matrix.ravel()[flat].copy()
    return present, flat.astype(np.int64), before


def multiplicative_pp_surgery(
    composition: NDArray,
    beta: NDArray,
    factor: float,
    name: str,
) -> BetaSurgery:
    """Multiply every present-present edge by one frozen positive factor."""

    if not np.isfinite(factor) or factor <= 0.0 or factor == 1.0:
        raise ValueError("multiplicative surgery factor must be positive and nonunit")
    matrix = np.asarray(beta, dtype=np.float64)
    _present, flat, before = _present_block(composition, matrix)
    altered = matrix.copy()
    altered.ravel()[flat] = before * factor
    after = altered.ravel()[flat].copy()
    observed = float(np.linalg.norm(altered - matrix))
    requested = abs(factor - 1.0) * float(np.linalg.norm(before))
    return BetaSurgery(
        name=name,
        beta=altered,
        flat_indices=flat,
        before=before,
        after=after,
        requested_norm=requested,
        observed_norm=observed,
    )


def balanced_log_direction(count: int, rng: np.random.Generator) -> NDArray[np.float64]:
    """Draw a deterministic nonzero direction whose log changes sum to zero."""

    if count < 2:
        raise ValueError("balanced direction requires at least two entries")
    direction = np.asarray(rng.standard_normal(count), dtype=np.float64)
    direction -= direction.mean()
    if np.linalg.norm(direction) == 0.0:
        direction = np.linspace(-1.0, 1.0, count, dtype=np.float64)
        direction -= direction.mean()
    direction /= np.linalg.norm(direction)
    direction -= direction.mean()
    if not np.any(direction > 0.0) or not np.any(direction < 0.0):
        raise AssertionError("balanced direction lacks both signs")
    return direction


def audited_random_pp_surgery(
    composition: NDArray,
    beta: NDArray,
    target_norm: float,
    direction: NDArray,
    name: str,
) -> BetaSurgery:
    """Perturb all and only P x P edges at an exact achieved norm."""

    matrix = np.asarray(beta, dtype=np.float64)
    _present, flat, before = _present_block(composition, matrix)
    vector = np.asarray(direction, dtype=np.float64)
    if vector.shape != before.shape or not np.isfinite(vector).all():
        raise ValueError("random surgery direction has the wrong shape")
    if not np.isclose(vector.sum(), 0.0, atol=1e-12, rtol=0.0):
        raise ValueError("random surgery direction is not log balanced")
    if not np.any(vector > 0.0) or not np.any(vector < 0.0):
        raise ValueError("random surgery direction must have both signs")
    if not np.isfinite(target_norm) or target_norm <= 0.0:
        raise ValueError("random surgery target norm must be positive")

    def distance(scale: float) -> float:
        with np.errstate(over="ignore"):
            changed = before * np.exp(scale * vector)
        if not np.isfinite(changed).all():
            return float("inf")
        return float(np.linalg.norm(changed - before))

    lower = 0.0
    upper = 1.0
    for _ in range(256):
        if distance(upper) >= target_norm:
            break
        upper *= 2.0
    else:  # pragma: no cover
        raise ValueError("could not bracket random surgery magnitude")
    for _ in range(192):
        middle = 0.5 * (lower + upper)
        if distance(middle) < target_norm:
            lower = middle
        else:
            upper = middle
    scale = 0.5 * (lower + upper)
    after = before * np.exp(scale * vector)
    altered = matrix.copy()
    altered.ravel()[flat] = after
    observed = float(np.linalg.norm(altered - matrix))
    tolerance = 1e-12 * max(1.0, target_norm)
    if abs(observed - target_norm) > tolerance:
        raise AssertionError("random surgery failed its exact norm audit")
    if np.any(altered <= 0.0) or not np.isfinite(altered).all():
        raise AssertionError("random surgery violated positivity")
    return BetaSurgery(
        name=name,
        beta=altered,
        flat_indices=flat,
        before=before,
        after=after,
        requested_norm=float(target_norm),
        observed_norm=observed,
    )


def select_surgeries(
    composition: NDArray, beta: NDArray, rng: np.random.Generator
) -> tuple[BetaSurgery | None, ...]:
    """Construct every frozen arm; both random doses share one direction."""

    _present, flat, before = _present_block(composition, beta)
    block_norm = float(np.linalg.norm(before))
    direction = balanced_log_direction(flat.size, rng)
    by_name: dict[str, BetaSurgery | None] = {
        "SMALL_LOOSEN": multiplicative_pp_surgery(
            composition, beta, SMALL_LOOSEN_FACTOR, "SMALL_LOOSEN"
        ),
        "SMALL_TIGHTEN": multiplicative_pp_surgery(
            composition, beta, SMALL_TIGHTEN_FACTOR, "SMALL_TIGHTEN"
        ),
        "SMALL_RANDOM_PP": audited_random_pp_surgery(
            composition,
            beta,
            0.05 * block_norm,
            direction,
            "SMALL_RANDOM_PP",
        ),
        "FABLE_LOOSEN": multiplicative_pp_surgery(
            composition, beta, FABLE_LOOSEN_FACTOR, "FABLE_LOOSEN"
        ),
        "FABLE_TIGHTEN": multiplicative_pp_surgery(
            composition, beta, FABLE_TIGHTEN_FACTOR, "FABLE_TIGHTEN"
        ),
        "FABLE_RANDOM_PP": audited_random_pp_surgery(
            composition,
            beta,
            0.5 * block_norm,
            direction,
            "FABLE_RANDOM_PP",
        ),
        "NOOP": None,
    }
    return tuple(by_name[arm] for arm in ARMS)


def _experiment(spec: BridgeSpec) -> ExperimentConfig:
    cohort = CohortConfig(spec.matrices, spec.branches, LANDMARKS)
    return ExperimentConfig(
        development=cohort,
        confirmation=cohort,
        horizon=HORIZON,
        bootstrap_repetitions=BOOTSTRAP_REPETITIONS,
        permutation_repetitions=RANDOMIZATION_REPETITIONS,
        regenerate_confirmation=True,
        master_seed=spec.cohort_seed,
    )


def _selection_seed(spec: BridgeSpec, case: StateCase) -> int:
    return derive_seed(
        spec.selection_seed,
        f"{LABEL}.selection.random_pp_direction",
        case.candidate,
        case.matrix_id,
        case.landmark,
    )


def _future_seed(spec: BridgeSpec, case: StateCase, branch: int) -> int:
    return derive_seed(
        spec.future_seed,
        f"{LABEL}.future",
        case.candidate,
        case.matrix_id,
        case.landmark,
        branch,
    )


def _phase_worker(
    arguments: tuple[StateCase, ExperimentConfig, BridgeSpec, str]
) -> base.PhaseBatch:
    case, experiment, spec, model_path = arguments
    limiter = threadpool_limits(limits=1)
    try:
        predictor = FrozenFullPredictor.load(model_path)
        surgeries = select_surgeries(
            case.snapshot.composition,
            case.beta,
            np.random.default_rng(_selection_seed(spec, case)),
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
        arm_outcomes: list[list[Any | None]] = [
            [None] * spec.branches for _ in spec.arms
        ]
        for branch in range(spec.branches):
            seed = _future_seed(spec, case, branch)
            for arm_index, surgery in enumerate(surgeries):
                arm_outcomes[arm_index][branch] = simulate_one_shot(
                    case.snapshot,
                    case.beta if surgery is None else surgery.beta,
                    case.candidate,
                    experiment.gard,
                    HORIZON,
                    np.random.default_rng(seed),
                    None,
                )
        outcomes = tuple(
            tuple(item for item in arm if item is not None) for arm in arm_outcomes
        )
        if any(len(arm) != spec.branches for arm in outcomes):
            raise AssertionError("P3b worker dropped an outcome")
        return base.PhaseBatch(
            state_id=case.state_id,
            state_digest=base._snapshot_digest(case),
            arm_names=spec.arms,
            predictions=predictions,
            selected_edits=tuple(None for _ in spec.arms),
            surgeries=surgeries,
            scored_edits=tuple(),
            catalytic_support=np.empty(0, dtype=np.float64),
            outcomes=outcomes,
        )
    finally:
        limiter.restore_original_limits()


def _checkpoint_contract(
    cases: list[StateCase], spec: BridgeSpec, registration_id: str, stage: str
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "registration_id": registration_id,
        "scientific_label": LABEL,
        "phase": spec.phase,
        "role": spec.role,
        "stage": stage,
        "matrices": spec.matrices,
        "branches": spec.branches,
        "landmarks": list(LANDMARKS),
        "horizon": HORIZON,
        "arms": list(spec.arms),
        "case_ids": [case.state_id for case in cases],
        "case_digests": [base._snapshot_digest(case) for case in cases],
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
    spec: BridgeSpec,
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
        if json.loads(contract_path.read_text(encoding="utf-8")) != json.loads(
            json.dumps(_json_ready(contract))
        ):
            raise ValueError(f"P3b checkpoint contract changed: {checkpoint_directory}")
    else:
        base._atomic_json(contract_path, contract)

    batches: list[base.PhaseBatch | None] = [None] * len(cases)
    missing: list[int] = []
    for index, case in enumerate(cases):
        path = checkpoint_directory / f"state_{index:04d}.pkl"
        if path.is_file():
            with path.open("rb") as handle:
                batch = pickle.load(handle)
            if (
                not isinstance(batch, base.PhaseBatch)
                or batch.state_id != case.state_id
                or batch.state_digest != base._snapshot_digest(case)
                or batch.arm_names != spec.arms
            ):
                raise ValueError(f"invalid P3b checkpoint {path}")
            batches[index] = batch
        else:
            missing.append(index)

    def status(state: str) -> None:
        complete = sum(batch is not None for batch in batches)
        base._atomic_json(
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
            base._atomic_pickle(checkpoint_directory / f"state_{index:04d}.pkl", batch)
            status("running")
    elif missing:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            generated = executor.map(_phase_worker, arguments, chunksize=1)
            for index, batch in zip(missing, generated, strict=True):
                batches[index] = batch
                base._atomic_pickle(
                    checkpoint_directory / f"state_{index:04d}.pkl", batch
                )
                status("running")
    status("complete")
    if any(batch is None for batch in batches):
        raise AssertionError("P3b checkpoint stage has missing states")
    return [batch for batch in batches if batch is not None]


def _contrast_summary(
    matrix_values: NDArray[np.float64], bootstrap_indices: NDArray[np.int64]
) -> tuple[dict[str, Any], NDArray[np.float64]]:
    bootstrap = _bootstrap_means(matrix_values, bootstrap_indices)
    summary = {
        "estimate": float(matrix_values.mean()),
        "bootstrap_ci95": _interval(bootstrap),
        "matrices_expected_sign": int(np.count_nonzero(matrix_values > 0.0)),
        "matrices_zero": int(np.count_nonzero(matrix_values == 0.0)),
        "maximum_leave_one_matrix_out_influence": (
            _maximum_leave_one_out_influence(matrix_values)
        ),
    }
    return summary, bootstrap


def _analyze_scope(
    scope_name: str,
    scope_landmarks: tuple[int, ...],
    cases: list[StateCase],
    targets: NDArray,
    predictions: NDArray,
    draws: dict[str, NDArray],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, NDArray]]:
    targets_array = np.asarray(targets, dtype=np.float64)
    predictions_array = np.asarray(predictions, dtype=np.float64)
    expected = (len(cases), len(ARMS), BRANCHES)
    if targets_array.shape != expected:
        raise ValueError(f"P3b target shape {targets_array.shape} differs from {expected}")
    if predictions_array.shape != (len(cases), len(ARMS)):
        raise ValueError("P3b predictions do not align with states and arms")
    bootstrap_indices = np.asarray(draws["bootstrap_indices"], dtype=np.int64)
    signs = np.asarray(draws["randomization_signs"], dtype=np.float64)
    if bootstrap_indices.shape != (BOOTSTRAP_REPETITIONS, MATRICES):
        raise ValueError("P3b bootstrap draw shape changed")
    if signs.shape != (RANDOMIZATION_REPETITIONS, MATRICES):
        raise ValueError("P3b randomization draw shape changed")
    if not np.isin(signs, (-1.0, 1.0)).all():
        raise ValueError("P3b randomization signs changed")

    arm_index = {name: index for index, name in enumerate(ARMS)}
    matrix_order = np.arange(MATRICES, dtype=np.int64)
    cells: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    stored: dict[str, NDArray] = {}
    raw_high_p: list[float] = []
    raw_dose_p: list[float] = []

    for candidate in ("02", "03"):
        candidate_mask = np.asarray(
            [
                case.candidate == candidate and case.landmark in scope_landmarks
                for case in cases
            ],
            dtype=bool,
        )
        selected_cases = [case for case, keep in zip(cases, candidate_mask) if keep]
        ids = np.asarray([case.matrix_id for case in selected_cases], dtype=np.int64)
        expected_states = MATRICES * len(scope_landmarks)
        if candidate_mask.sum() != expected_states:
            raise ValueError(f"scope {scope_name} candidate {candidate} is incomplete")
        if not np.array_equal(np.unique(ids), matrix_order):
            raise ValueError(f"scope {scope_name} candidate {candidate} lacks matrices")
        candidate_targets = targets_array[candidate_mask]
        candidate_predictions = predictions_array[candidate_mask]

        for half, branch_slice in (
            ("A", slice(0, BRANCHES // 2)),
            ("B", slice(BRANCHES // 2, BRANCHES)),
        ):
            q = candidate_targets[:, :, branch_slice].mean(axis=2)
            state_values = {
                "fable_effect": (
                    q[:, arm_index["FABLE_LOOSEN"]]
                    - q[:, arm_index["FABLE_TIGHTEN"]]
                ),
                "small_effect": (
                    q[:, arm_index["SMALL_LOOSEN"]]
                    - q[:, arm_index["SMALL_TIGHTEN"]]
                ),
                "fable_random_minus_noop": (
                    q[:, arm_index["FABLE_RANDOM_PP"]]
                    - q[:, arm_index["NOOP"]]
                ),
                "small_random_minus_noop": (
                    q[:, arm_index["SMALL_RANDOM_PP"]]
                    - q[:, arm_index["NOOP"]]
                ),
                "fable_loosen_minus_noop": (
                    q[:, arm_index["FABLE_LOOSEN"]]
                    - q[:, arm_index["NOOP"]]
                ),
                "noop_minus_fable_tighten": (
                    q[:, arm_index["NOOP"]]
                    - q[:, arm_index["FABLE_TIGHTEN"]]
                ),
            }
            state_values["fable_minus_small_effect"] = (
                state_values["fable_effect"] - state_values["small_effect"]
            )
            matrix_values = {
                name: _matrix_means(values, ids, matrix_order)
                for name, values in state_values.items()
            }
            summaries: dict[str, Any] = {}
            bootstraps: dict[str, NDArray] = {}
            for name, values in matrix_values.items():
                summaries[name], bootstraps[name] = _contrast_summary(
                    values, bootstrap_indices
                )

            high_p, high_null = _one_sided_sign_p(
                matrix_values["fable_effect"], signs
            )
            dose_p, dose_null = _one_sided_sign_p(
                matrix_values["fable_minus_small_effect"], signs
            )
            raw_high_p.append(high_p)
            raw_dose_p.append(dose_p)
            cell_key = f"{scope_name}__c{candidate}_{half}"
            stored[f"{cell_key}__randomization__fable_effect"] = high_null
            stored[f"{cell_key}__randomization__fable_minus_small"] = dose_null
            for contrast_name, values in bootstraps.items():
                stored[f"{cell_key}__bootstrap__{contrast_name}"] = values

            arms: dict[str, Any] = {}
            for arm in ARMS:
                index = arm_index[arm]
                state_q = q[:, index]
                matrix_q = _matrix_means(state_q, ids, matrix_order)
                arm_bootstrap = _bootstrap_means(matrix_q, bootstrap_indices)
                arms[arm] = {
                    "mean_probability": float(matrix_q.mean()),
                    "bootstrap_ci95": _interval(arm_bootstrap),
                    "mean_frozen_prediction": float(
                        _matrix_means(
                            candidate_predictions[:, index], ids, matrix_order
                        ).mean()
                    ),
                    "branch_scores": _bernoulli_scores(
                        candidate_targets[:, index, branch_slice],
                        candidate_predictions[:, index],
                    ),
                }

            fable_random_ci90 = _interval(
                bootstraps["fable_random_minus_noop"], alpha=0.10
            )
            small_random_ci90 = _interval(
                bootstraps["small_random_minus_noop"], alpha=0.10
            )
            cell = {
                "cell": f"c{candidate}_{half}",
                "scope_cell": cell_key,
                "scope": scope_name,
                "scope_landmarks": list(scope_landmarks),
                "candidate": candidate,
                "branch_half": half,
                "branch_range": [branch_slice.start, branch_slice.stop - 1],
                "states": int(candidate_mask.sum()),
                "matrices": MATRICES,
                "arms": arms,
                "contrasts": summaries,
                "fable_effect_randomization_p_raw": high_p,
                "two_dose_randomization_p_raw": dose_p,
                "fable_random_noop_equivalence": {
                    "margin": EQUIVALENCE_MARGIN,
                    "bootstrap_ci90": fable_random_ci90,
                    "tost_equivalent": bool(
                        fable_random_ci90[0] > -EQUIVALENCE_MARGIN
                        and fable_random_ci90[1] < EQUIVALENCE_MARGIN
                    ),
                },
                "small_random_noop_equivalence": {
                    "margin": EQUIVALENCE_MARGIN,
                    "bootstrap_ci90": small_random_ci90,
                    "tost_equivalent": bool(
                        small_random_ci90[0] > -EQUIVALENCE_MARGIN
                        and small_random_ci90[1] < EQUIVALENCE_MARGIN
                    ),
                },
                "predicted_versus_realized": {
                    "mean_predicted_fable_effect": float(
                        (
                            candidate_predictions[:, arm_index["FABLE_LOOSEN"]]
                            - candidate_predictions[:, arm_index["FABLE_TIGHTEN"]]
                        ).mean()
                    ),
                    "mean_realized_fable_effect": summaries["fable_effect"][
                        "estimate"
                    ],
                    "mean_predicted_small_effect": float(
                        (
                            candidate_predictions[:, arm_index["SMALL_LOOSEN"]]
                            - candidate_predictions[:, arm_index["SMALL_TIGHTEN"]]
                        ).mean()
                    ),
                    "mean_realized_small_effect": summaries["small_effect"][
                        "estimate"
                    ],
                },
            }
            cells.append(cell)
            for position, matrix_id in enumerate(matrix_order):
                row: dict[str, Any] = {
                    "scope": scope_name,
                    "cell": f"c{candidate}_{half}",
                    "candidate": candidate,
                    "branch_half": half,
                    "matrix_id": int(matrix_id),
                }
                row.update(
                    {
                        name: float(values[position])
                        for name, values in matrix_values.items()
                    }
                )
                matrix_rows.append(row)

    high_adjusted = holm_adjust(raw_high_p)
    dose_adjusted = holm_adjust(raw_dose_p)
    for cell, high_p, dose_p in zip(
        cells, high_adjusted, dose_adjusted, strict=True
    ):
        cell["fable_effect_randomization_p_holm"] = float(high_p)
        cell["two_dose_randomization_p_holm"] = float(dose_p)
        high = cell["contrasts"]["fable_effect"]
        dose = cell["contrasts"]["fable_minus_small_effect"]
        cell["primary_fable_gates"] = {
            "fable_effect_positive": high["estimate"] > 0.0,
            "fable_effect_bootstrap_lower_positive": high["bootstrap_ci95"][0]
            > 0.0,
            "fable_effect_holm_p_below_0_05": high_p < 0.05,
            "fable_random_tost_equivalent_to_noop": cell[
                "fable_random_noop_equivalence"
            ]["tost_equivalent"],
        }
        cell["primary_fable_cell_pass_without_replay"] = bool(
            all(cell["primary_fable_gates"].values())
        )
        cell["two_dose_gates"] = {
            "fable_minus_small_positive": dose["estimate"] > 0.0,
            "fable_minus_small_bootstrap_lower_positive": dose[
                "bootstrap_ci95"
            ][0]
            > 0.0,
            "fable_minus_small_holm_p_below_0_05": dose_p < 0.05,
        }
        cell["two_dose_cell_pass"] = bool(
            all(cell["two_dose_gates"].values())
        )

    scope_result = {
        "scope": scope_name,
        "landmarks": list(scope_landmarks),
        "cells": cells,
        "holm_family_size": 4,
        "all_fable_cells_pass_without_replay": bool(
            all(cell["primary_fable_cell_pass_without_replay"] for cell in cells)
        ),
        "all_two_dose_cells_pass": bool(
            all(cell["two_dose_cell_pass"] for cell in cells)
        ),
        "all_small_random_cells_tost_equivalent": bool(
            all(
                cell["small_random_noop_equivalence"]["tost_equivalent"]
                for cell in cells
            )
        ),
    }
    return scope_result, matrix_rows, stored


def compute_bridge_inference(
    cases: list[StateCase],
    targets: NDArray,
    predictions: NDArray,
    draws: dict[str, NDArray],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    primary, primary_rows, primary_stored = _analyze_scope(
        "landmark60", PRIMARY_LANDMARKS, cases, targets, predictions, draws
    )
    generalization, generalization_rows, generalization_stored = _analyze_scope(
        "five_landmark_generalization",
        GENERALIZATION_LANDMARKS,
        cases,
        targets,
        predictions,
        draws,
    )

    arm_index = {name: index for index, name in enumerate(ARMS)}
    targets_array = np.asarray(targets, dtype=np.float64)
    landmark_effects: list[dict[str, Any]] = []
    for candidate in ("02", "03"):
        for landmark in LANDMARKS:
            selected = np.asarray(
                [
                    case.candidate == candidate and case.landmark == landmark
                    for case in cases
                ],
                dtype=bool,
            )
            for half, branch_slice in (
                ("A", slice(0, BRANCHES // 2)),
                ("B", slice(BRANCHES // 2, BRANCHES)),
            ):
                q = targets_array[selected, :, branch_slice].mean(axis=2)
                landmark_effects.append(
                    {
                        "candidate": candidate,
                        "branch_half": half,
                        "landmark": landmark,
                        "fable_effect": float(
                            np.mean(
                                q[:, arm_index["FABLE_LOOSEN"]]
                                - q[:, arm_index["FABLE_TIGHTEN"]]
                            )
                        ),
                        "small_effect": float(
                            np.mean(
                                q[:, arm_index["SMALL_LOOSEN"]]
                                - q[:, arm_index["SMALL_TIGHTEN"]]
                            )
                        ),
                        "fable_random_minus_noop": float(
                            np.mean(
                                q[:, arm_index["FABLE_RANDOM_PP"]]
                                - q[:, arm_index["NOOP"]]
                            )
                        ),
                        "small_random_minus_noop": float(
                            np.mean(
                                q[:, arm_index["SMALL_RANDOM_PP"]]
                                - q[:, arm_index["NOOP"]]
                            )
                        ),
                    }
                )

    stored = dict(primary_stored)
    stored.update(generalization_stored)
    result = {
        "format": "codex-intervention-p3b-dose-bridge-inference-v1",
        "inference_unit": "whole catalytic matrix",
        "state_replicates_within_matrix_kept_together": True,
        "shared_bootstrap_draws_across_scopes_and_cells": True,
        "shared_randomization_signs_across_scopes_and_cells": True,
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "randomization_repetitions": RANDOMIZATION_REPETITIONS,
        "equivalence_margin": EQUIVALENCE_MARGIN,
        "primary": primary,
        "generalization": generalization,
        "landmark_effects": landmark_effects,
        "stored_inference_arrays": {
            "bootstrap_indices_shape": list(
                np.asarray(draws["bootstrap_indices"]).shape
            ),
            "randomization_signs_shape": list(
                np.asarray(draws["randomization_signs"]).shape
            ),
            "arrays": {name: value.tolist() for name, value in stored.items()},
        },
    }
    return result, primary_rows + generalization_rows


def add_replay_gates(metrics: dict[str, Any], replay_exact: bool) -> dict[str, Any]:
    for scope_name in ("primary", "generalization"):
        scope = metrics[scope_name]
        scope["all_fable_cells_pass"] = bool(
            scope["all_fable_cells_pass_without_replay"] and replay_exact
        )
        scope["all_two_dose_cells_pass_with_replay"] = bool(
            scope["all_two_dose_cells_pass"] and replay_exact
        )
    metrics["primary_replication_gate_pass"] = metrics["primary"][
        "all_fable_cells_pass"
    ]
    metrics["five_landmark_generalization_gate_pass"] = metrics[
        "generalization"
    ]["all_fable_cells_pass"]
    metrics["landmark60_two_dose_gate_pass"] = metrics["primary"][
        "all_two_dose_cells_pass_with_replay"
    ]
    metrics["five_landmark_two_dose_gate_pass"] = metrics["generalization"][
        "all_two_dose_cells_pass_with_replay"
    ]
    metrics["exact_replay_in_gate"] = bool(replay_exact)
    return metrics


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
    for name, values in stored["arrays"].items():
        arrays[name] = np.asarray(values, dtype=np.float64)
    np.savez_compressed(path, **arrays)
    metrics["stored_inference_arrays"] = {
        "path": path.name,
        "bootstrap_indices_shape": stored["bootstrap_indices_shape"],
        "randomization_signs_shape": stored["randomization_signs_shape"],
        "stored_array_names": sorted(stored["arrays"]),
        "all_scope_bootstrap_and_randomization_arrays_stored": True,
    }


def _readback_metrics(
    output: Path,
    cases: list[StateCase],
    expected: dict[str, Any],
    expected_matrix_rows: list[dict[str, Any]],
    replay_exact: bool,
) -> dict[str, Any]:
    with np.load(output / "branch_arrays.npz", allow_pickle=False) as archive:
        targets = archive["targets"]
        predictions = archive["predictions"]
    with np.load(output / "inference_arrays.npz", allow_pickle=False) as archive:
        draws = {
            "bootstrap_indices": archive["bootstrap_indices"],
            "randomization_signs": archive["randomization_signs"],
        }
    observed, matrix_rows = compute_bridge_inference(
        cases, targets, predictions, draws
    )
    stored = observed.pop("stored_inference_arrays")
    observed["stored_inference_arrays"] = {
        "path": "inference_arrays.npz",
        "bootstrap_indices_shape": stored["bootstrap_indices_shape"],
        "randomization_signs_shape": stored["randomization_signs_shape"],
        "stored_array_names": sorted(stored["arrays"]),
        "all_scope_bootstrap_and_randomization_arrays_stored": True,
    }
    add_replay_gates(observed, replay_exact)
    metrics_exact = _json_ready(observed) == _json_ready(expected)
    matrix_effects_exact = _json_ready(matrix_rows) == _json_ready(
        expected_matrix_rows
    )
    if not metrics_exact or not matrix_effects_exact:
        raise ValueError("P3b written-artifact inference changed")
    return {
        "branch_arrays_reloaded": True,
        "inference_draws_reloaded": True,
        "primary_metrics_exact": metrics_exact,
        "matrix_effects_exact": matrix_effects_exact,
        "phase_specific_cr4_gate_recomputed": True,
        "no_fitting_or_recalibration": True,
    }


def _protocol() -> dict[str, Any]:
    spec = phase_spec()
    futures = 2 * spec.matrices * len(LANDMARKS) * len(ARMS) * spec.branches
    value: dict[str, Any] = {
        "format": PROGRAM_FORMAT,
        "status": "frozen_before_any_p3b_scientific_matrix",
        "scientific_label": LABEL,
        "reason": {
            "sealed_p3_preserved": True,
            "sealed_p3_classification": (
                "valid unintended small-dose experiment; not a failed replication "
                "of Fable's delta=0.5 surgery"
            ),
            "instruction_assembly_error": (
                "0.05 block-norm target was assembled in error from Fable delta=0.5"
            ),
            "external_clarification_received_after_p3_seal": True,
            "external_fable_contract": {
                "tighten": "beta[P,P] *= 1.5",
                "loosen": "beta[P,P] /= 1.5",
                "log_symmetric": True,
                "frobenius_symmetric": False,
                "tighten_distance_fraction": 0.5,
                "loosen_distance_fraction": 1.0 / 3.0,
            },
            "external_random_control_limitation": (
                "historical Fable whole-matrix random control was only approximately "
                "norm matched; P3b uses an exact within-PxP control"
            ),
        },
        "design": {
            "role": spec.role,
            "matrices": spec.matrices,
            "candidates": list(CANDIDATES),
            "landmarks": list(LANDMARKS),
            "primary_landmark": 60,
            "generalization_landmarks": list(GENERALIZATION_LANDMARKS),
            "states": 2 * spec.matrices * len(LANDMARKS),
            "arms": list(ARMS),
            "branches_per_arm_per_state": spec.branches,
            "branch_halves": {"A": [0, 15], "B": [16, 31]},
            "horizon_fissions": HORIZON,
            "primary_futures": futures,
            "replay_futures": futures,
            "fresh_matrices_states_and_seed_domains": True,
            "future_seed_includes_arm": False,
            "intervention_future_retries": 0,
            "matrix_replacement": False,
        },
        "arms": {
            "SMALL_LOOSEN": {"factor": SMALL_LOOSEN_FACTOR},
            "SMALL_TIGHTEN": {"factor": SMALL_TIGHTEN_FACTOR},
            "SMALL_RANDOM_PP": {
                "location": "all and only P x P",
                "target_norm_fraction": 0.05,
                "balanced_log_direction": True,
            },
            "FABLE_LOOSEN": {"factor": FABLE_LOOSEN_FACTOR},
            "FABLE_TIGHTEN": {"factor": FABLE_TIGHTEN_FACTOR},
            "FABLE_RANDOM_PP": {
                "location": "all and only P x P",
                "target_norm_fraction": 0.5,
                "balanced_log_direction": True,
            },
            "NOOP": {"factor": 1.0},
            "random_direction_shared_across_doses_within_state": True,
            "random_directions_domain_separated_from_futures": True,
        },
        "endpoint": {
            "name": "JOINT_BREAK_RUN3",
            "inheritance": "unrounded float64 H > 0.9",
            "break": "H <= 0.9",
            "horizon": HORIZON,
            "positive": "break followed strictly later by run3",
            "extinction_before_certification": "negative",
            "certification_before_later_extinction": "positive",
        },
        "primary_inference": {
            "scope": "landmark 60 only",
            "contrast": "FABLE_LOOSEN - FABLE_TIGHTEN",
            "cells": ["c02_A", "c02_B", "c03_A", "c03_B"],
            "unit": "whole catalytic matrix",
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "randomization_repetitions": RANDOMIZATION_REPETITIONS,
            "holm_family_size": 4,
            "gates": [
                "contrast > 0",
                "95% bootstrap lower bound > 0",
                "Holm-adjusted sign-randomization p < 0.05",
                "FABLE_RANDOM_PP TOST-equivalent to NOOP within +/-0.025",
                "exact replay",
            ],
            "cr1_only_gates_excluded": [
                "each target arm separately differs from NOOP",
                "random effect no greater than 25% of target contrast",
            ],
        },
        "secondary_inference": {
            "five_landmark_generalization": {
                "landmarks": list(GENERALIZATION_LANDMARKS),
                "same_high_dose_gate_separate_holm_family": True,
                "cannot_rescue_primary": True,
            },
            "two_dose_ordering": {
                "small_effect": "SMALL_LOOSEN - SMALL_TIGHTEN",
                "dose_contrast": "Fable-strength effect - small effect",
                "gates": [
                    "dose contrast > 0",
                    "95% bootstrap lower bound > 0",
                    "Holm-adjusted sign-randomization p < 0.05",
                    "exact replay",
                ],
                "claim_boundary": "two-dose ordering, not a complete dose-response curve",
            },
        },
        "audit": {
            "persist_every_changed_edge": True,
            "per_state_achieved_norm": True,
            "per_state_changed_edge_set": True,
            "per_state_log_balance": True,
            "complete_deterministic_replay": True,
            "complete_written_artifact_readback": True,
            "predictions_descriptive_only": True,
            "refitting_or_recalibration": False,
        },
        "seed_domains": SEED_DOMAINS,
        "lifecycle": {
            "mandatory_stop_after_seal": True,
            "next_scientific_phase_launched": False,
        },
        "claim_boundary": {
            "passing_primary_may_support": (
                "qualitative cross-clean-room replication of Fable-strength "
                "present-present beta-surgery control under Codex contracts"
            ),
            "prohibited": base._protocol()["claim_boundaries"]["prohibited"],
        },
    }
    value["protocol_id"] = _canonical_digest(value)
    return value


def validation_checks() -> dict[str, Any]:
    original = base.validation_checks()
    checks = dict(original["checks"])

    def record(name: str, condition: bool, detail: Any = None) -> None:
        if not condition:
            raise AssertionError(f"validation failed: {name}: {detail}")
        checks[name] = {"passed": True, "detail": detail}

    composition = np.asarray([3, 0, 2, 1], dtype=np.int64)
    beta = np.asarray(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ],
        dtype=np.float64,
    )
    surgeries = select_surgeries(
        composition, beta, np.random.default_rng(derive_seed(SEED_DOMAINS["validation"], "fixture"))
    )
    by_name = dict(zip(ARMS, surgeries, strict=True))
    _present, expected_flat, before = _present_block(composition, beta)
    block_norm = float(np.linalg.norm(before))
    tolerance = 1e-12 * max(1.0, block_norm)

    record(
        "27_p3b_targeted_factors_exact",
        np.array_equal(
            by_name["SMALL_LOOSEN"].after,
            before * SMALL_LOOSEN_FACTOR,
        )
        and np.array_equal(
            by_name["SMALL_TIGHTEN"].after,
            before * SMALL_TIGHTEN_FACTOR,
        )
        and np.array_equal(
            by_name["FABLE_LOOSEN"].after,
            before * FABLE_LOOSEN_FACTOR,
        )
        and np.array_equal(
            by_name["FABLE_TIGHTEN"].after,
            before * FABLE_TIGHTEN_FACTOR,
        ),
    )
    record(
        "28_p3b_fable_pair_log_symmetric",
        np.isclose(
            np.log(FABLE_TIGHTEN_FACTOR),
            -np.log(FABLE_LOOSEN_FACTOR),
            atol=1e-15,
            rtol=0.0,
        ),
    )
    record(
        "29_p3b_registered_frobenius_asymmetry_exact",
        abs(by_name["FABLE_TIGHTEN"].observed_norm - 0.5 * block_norm)
        <= tolerance
        and abs(by_name["FABLE_LOOSEN"].observed_norm - block_norm / 3.0)
        <= tolerance
        and np.isclose(
            by_name["FABLE_LOOSEN"].observed_norm
            / by_name["FABLE_TIGHTEN"].observed_norm,
            2.0 / 3.0,
            atol=1e-14,
            rtol=0.0,
        ),
    )
    random_arms = ("SMALL_RANDOM_PP", "FABLE_RANDOM_PP")
    record(
        "30_p3b_random_controls_change_all_and_only_pp",
        all(
            np.array_equal(by_name[name].flat_indices, expected_flat)
            and np.count_nonzero(by_name[name].after != by_name[name].before)
            == expected_flat.size
            for name in random_arms
        ),
    )
    record(
        "31_p3b_random_controls_positive_and_exact_norm",
        all(
            np.all(by_name[name].beta > 0.0)
            and abs(by_name[name].observed_norm - by_name[name].requested_norm)
            <= tolerance
            for name in random_arms
        )
        and abs(by_name["SMALL_RANDOM_PP"].observed_norm - 0.05 * block_norm)
        <= tolerance
        and abs(by_name["FABLE_RANDOM_PP"].observed_norm - 0.5 * block_norm)
        <= tolerance,
    )
    small_log = np.log(
        by_name["SMALL_RANDOM_PP"].after
        / by_name["SMALL_RANDOM_PP"].before
    )
    high_log = np.log(
        by_name["FABLE_RANDOM_PP"].after
        / by_name["FABLE_RANDOM_PP"].before
    )
    record(
        "32_p3b_random_controls_log_balanced_same_direction",
        abs(float(small_log.sum())) <= 1e-12
        and abs(float(high_log.sum())) <= 1e-12
        and np.allclose(
            small_log / np.linalg.norm(small_log),
            high_log / np.linalg.norm(high_log),
            atol=1e-12,
            rtol=0.0,
        ),
    )
    record(
        "33_p3b_noop_is_exactly_none",
        by_name["NOOP"] is None,
    )
    spec = phase_spec()
    dummy = StateCase(
        "p3b-seed-fixture",
        "FIX",
        "02",
        7,
        60,
        beta,
        base._fixture_snapshot(),
    )
    futures = [_future_seed(spec, dummy, branch) for branch in range(4)]
    record(
        "34_p3b_future_streams_unique_and_arm_free",
        len(set(futures)) == len(futures),
        {"arm_identity_in_key": False},
    )
    prior_domains = set(base.SEED_DOMAINS.values())
    record(
        "35_p3b_seed_domains_fresh_and_unique",
        len(set(SEED_DOMAINS.values())) == len(SEED_DOMAINS)
        and set(SEED_DOMAINS.values()).isdisjoint(prior_domains),
    )
    record(
        "36_p3b_phase_specific_gate_excludes_cr1_extras",
        _protocol()["primary_inference"]["cr1_only_gates_excluded"]
        == [
            "each target arm separately differs from NOOP",
            "random effect no greater than 25% of target contrast",
        ],
    )
    verify_checksums(P3_RESULT)
    p3_manifest = json.loads((P3_RESULT / "manifest.json").read_text(encoding="utf-8"))
    record(
        "37_sealed_p3_preserved_as_small_dose_result",
        p3_manifest["phase"] == "p3"
        and p3_manifest["exact_replay"] is True
        and p3_manifest["primary_futures"] == 51_200,
    )
    return {
        "format": VALIDATION_FORMAT,
        "checks": checks,
        "original_required_checks_passed": original["required_checks_passed"],
        "all_checks_passed": all(value["passed"] for value in checks.values()),
        "check_count": len(checks),
        "scientific_cohort_generated": False,
        "scientific_effect_sizes_computed": False,
    }


def run_validation(output_directory: Path) -> None:
    output_directory = output_directory.resolve()
    if output_directory.exists():
        raise FileExistsError(f"refusing to overwrite {output_directory}")
    validation = validation_checks()
    command = [
        str(REPOSITORY_ROOT / ".venv/bin/python"),
        "-m",
        "pytest",
        "-q",
        "tests/test_intervention_p3b_dose_bridge.py",
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
            "P3b pytest validation failed\n"
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
            completed.stdout + completed.stderr, encoding="utf-8"
        )
        write_checksums(output)
    verify_checksums(output_directory)
    print(f"P3b validation passed: {output_directory}", flush=True)


def _append_registration_notice(registration_id: str) -> None:
    path = REPOSITORY_ROOT / "INTERVENTION_RESULTS_LEDGER.md"
    if path.exists():
        current = path.read_text(encoding="utf-8").rstrip() + "\n"
    else:
        current = "# Codex intervention results ledger\n"
    marker = f"<!-- p3b-dose-bridge-registered-{registration_id} -->"
    if marker in current:
        return
    lines = [
        "",
        marker,
        "## P3b beta-surgery dose bridge registered",
        "",
        f"- Registration: `{registration_id}`",
        "- The sealed P3 result remains unchanged and is retained as a valid unintended 5% dose experiment.",
        "- External post-seal clarification established that Fable used `beta[P,P] *= 1.5` versus `beta[P,P] /= 1.5`, not a 5% Frobenius perturbation.",
        "- P3b prospectively tests that actual contract and repeats the 5% pair on the same 80 entirely fresh matrices.",
        "- Random controls change all and only present-present edges and are audited to their exact registered Frobenius norms.",
        "- No P3b scientific matrix existed when this registration was sealed.",
        "",
    ]
    path.write_text(current + "\n".join(lines), encoding="utf-8")


def register_program(validation_directory: Path, output_directory: Path) -> None:
    validation_directory = validation_directory.resolve()
    output_directory = output_directory.resolve()
    if output_directory.exists():
        raise FileExistsError(f"refusing to overwrite {output_directory}")
    if DEFAULT_OUTPUT.exists() or DEFAULT_WORK.exists():
        raise FileExistsError(
            "P3b scientific output/work already exists before registration"
        )
    verify_checksums(validation_directory)
    validation = json.loads(
        (validation_directory / "validation.json").read_text(encoding="utf-8")
    )
    if (
        not validation["all_checks_passed"]
        or validation["scientific_cohort_generated"]
        or validation["scientific_effect_sizes_computed"]
    ):
        raise ValueError("P3b validation is not registration-eligible")
    original = base.verify_registration(ORIGINAL_REGISTRATION)
    if original["registration_id"] != EXPECTED_ORIGINAL_REGISTRATION_ID:
        raise ValueError("unexpected original intervention registration")
    verify_checksums(P3_RESULT)
    recovery = p3_recovery.verify_amendment(P3_RECOVERY_AMENDMENT)
    if recovery["amendment_id"] != EXPECTED_P3_RECOVERY_AMENDMENT_ID:
        raise ValueError("unexpected P3 recovery amendment")
    model_path = ORIGINAL_REGISTRATION / "frozen_full_predictor.npz"
    if sha256_file(model_path) != base.EXPECTED_MODEL_SHA256:
        raise ValueError("unexpected frozen JOINT_BREAK_RUN3 predictor")

    protocol = _protocol()
    with _atomic_destination(output_directory) as output:
        (output / "dose_bridge_protocol.json").write_text(
            json.dumps(_json_ready(protocol), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        seed_registry = {
            "format": "codex-intervention-p3b-seed-registry-v1",
            "domains": SEED_DOMAINS,
            "all_values_unique": len(set(SEED_DOMAINS.values()))
            == len(SEED_DOMAINS),
            "disjoint_from_original": set(SEED_DOMAINS.values()).isdisjoint(
                base.SEED_DOMAINS.values()
            ),
            "future_seed_includes_arm": False,
            "random_selection_stream_separate_from_future_stream": True,
        }
        (output / "dose_bridge_seed_registry.json").write_text(
            json.dumps(seed_registry, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        shutil.copy2(validation_directory / "validation.json", output / "validation.json")
        shutil.copy2(
            validation_directory / "pytest_output.txt", output / "pytest_output.txt"
        )
        shutil.copy2(model_path, output / "frozen_full_predictor.npz")
        payload: dict[str, Any] = {
            "format": REGISTRATION_FORMAT,
            "status": "sealed_before_any_p3b_scientific_matrix",
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": sha256_file(output / "dose_bridge_protocol.json"),
            "seed_registry_sha256": sha256_file(
                output / "dose_bridge_seed_registry.json"
            ),
            "source_hashes": _source_hashes(),
            "original_registration_id": original["registration_id"],
            "original_registration_checksum_manifest_sha256": sha256_file(
                ORIGINAL_REGISTRATION / "SHA256SUMS"
            ),
            "sealed_p3_checksum_manifest_sha256": sha256_file(
                P3_RESULT / "SHA256SUMS"
            ),
            "p3_recovery_amendment_id": recovery["amendment_id"],
            "p3_recovery_checksum_manifest_sha256": sha256_file(
                P3_RECOVERY_AMENDMENT / "SHA256SUMS"
            ),
            "frozen_predictor_sha256": sha256_file(
                output / "frozen_full_predictor.npz"
            ),
            "validation_checksum_manifest_sha256": sha256_file(
                validation_directory / "SHA256SUMS"
            ),
            "external_instruction_file_sha256_at_registration": (
                sha256_file(REPOSITORY_ROOT / "FULL_FABLE_REPLICATION_INSTRUCTIONS.md")
            ),
            "p3b_scientific_matrices_generated": False,
            "p3b_effect_sizes_computed": False,
        }
        payload["registration_id"] = _canonical_digest(payload)
        (output / "registration.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_checksums(output)
    registered = verify_registration(output_directory)
    _append_registration_notice(registered["registration_id"])
    print(f"P3b registration sealed: {registered['registration_id']}", flush=True)


def verify_registration(directory: Path) -> dict[str, Any]:
    directory = directory.resolve()
    verify_checksums(directory)
    payload = json.loads((directory / "registration.json").read_text(encoding="utf-8"))
    identifier = payload.pop("registration_id")
    if (
        payload.get("format") != REGISTRATION_FORMAT
        or payload.get("status") != "sealed_before_any_p3b_scientific_matrix"
        or _canonical_digest(payload) != identifier
    ):
        raise ValueError("invalid P3b registration")
    payload["registration_id"] = identifier
    if payload["source_hashes"] != _source_hashes():
        raise ValueError("P3b source changed after registration")
    if payload["frozen_predictor_sha256"] != base.EXPECTED_MODEL_SHA256:
        raise ValueError("P3b frozen predictor hash changed")
    original = base.verify_registration(ORIGINAL_REGISTRATION)
    if original["registration_id"] != payload["original_registration_id"]:
        raise ValueError("original intervention registration changed")
    verify_checksums(P3_RESULT)
    recovery = p3_recovery.verify_amendment(P3_RECOVERY_AMENDMENT)
    if recovery["amendment_id"] != payload["p3_recovery_amendment_id"]:
        raise ValueError("P3 recovery amendment changed")
    protocol = json.loads(
        (directory / "dose_bridge_protocol.json").read_text(encoding="utf-8")
    )
    if protocol != json.loads(json.dumps(_json_ready(_protocol()))):
        raise ValueError("P3b protocol implementation diverged")
    if (
        protocol["protocol_id"] != payload["protocol_id"]
        or sha256_file(directory / "dose_bridge_protocol.json")
        != payload["protocol_sha256"]
    ):
        raise ValueError("P3b protocol digest changed")
    seeds = json.loads(
        (directory / "dose_bridge_seed_registry.json").read_text(encoding="utf-8")
    )
    if (
        seeds["domains"] != SEED_DOMAINS
        or not seeds["all_values_unique"]
        or not seeds["disjoint_from_original"]
        or seeds["future_seed_includes_arm"]
        or not seeds["random_selection_stream_separate_from_future_stream"]
    ):
        raise ValueError("P3b seed registry changed")
    if sha256_file(directory / "frozen_full_predictor.npz") != base.EXPECTED_MODEL_SHA256:
        raise ValueError("registered P3b predictor bytes changed")
    validation = json.loads((directory / "validation.json").read_text(encoding="utf-8"))
    if not validation["all_checks_passed"]:
        raise ValueError("P3b validation no longer passes")
    return payload


def run_smoke(
    registration_directory: Path, output_directory: Path, workers: int
) -> None:
    if workers < 1:
        raise ValueError("workers must be positive")
    registration = verify_registration(registration_directory)
    output_directory = output_directory.resolve()
    if output_directory.exists():
        raise FileExistsError(f"refusing to overwrite {output_directory}")
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    spec = BridgeSpec(
        phase="p3b_smoke",
        role="non-scientific P3b legality, checkpoint, I/O, and replay smoke",
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
        prefix="codex-p3b-smoke-", dir=output_directory.parent
    ) as temporary:
        temporary_path = Path(temporary)
        with threadpool_limits(limits=1):
            cases = build_cohort(experiment, "INTP3B_SMOKE", cohort)
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
        replay = base.replay_audit(generated, replayed)
        if not replay["state_edit_endpoint_and_process_digests_exact"]:
            raise AssertionError("P3b non-scientific smoke replay failed")
        for batch in generated:
            for surgery in batch.surgeries:
                if surgery is not None and not np.isclose(
                    surgery.requested_norm,
                    surgery.observed_norm,
                    atol=1e-12 * max(1.0, surgery.requested_norm),
                    rtol=0.0,
                ):
                    raise AssertionError("P3b smoke norm audit failed")
    with _atomic_destination(output_directory) as output:
        (output / "manifest.json").write_text(
            json.dumps(
                {
                    "format": "codex-intervention-p3b-smoke-v1",
                    "registration_id": registration["registration_id"],
                    "scientific_result": False,
                    "scientific_matrix_count": 0,
                    "legality_checkpoint_io_norm_and_replay_paths_passed": True,
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
    print(f"P3b non-scientific smoke passed: {output_directory}", flush=True)


def _campaign_status(work: Path, state: str, detail: str) -> None:
    base._atomic_json(
        work / "campaign_status.json",
        {
            "format": CHECKPOINT_FORMAT,
            "phase": "p3b_dose_bridge",
            "state": state,
            "detail": detail,
            "mandatory_stop_after_seal": True,
        },
    )


def _prepare_campaign(
    work: Path,
    output: Path,
    registration: dict[str, Any],
    spec: BridgeSpec,
) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    work.mkdir(parents=True, exist_ok=True)
    contract: dict[str, Any] = {
        "format": "codex-intervention-p3b-campaign-contract-v1",
        "registration_id": registration["registration_id"],
        "scientific_label": LABEL,
        "phase": spec.phase,
        "role": spec.role,
        "output": str(output),
        "matrices": spec.matrices,
        "branches": spec.branches,
        "landmarks": list(LANDMARKS),
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
            raise ValueError("P3b work directory belongs to another campaign")
    else:
        base._atomic_json(path, contract)
    _campaign_status(work, "running", "campaign_initialized")


def _surgery_audit(
    cases: list[StateCase], batches: list[base.PhaseBatch]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case, batch in zip(cases, batches, strict=True):
        present = np.flatnonzero(case.snapshot.composition > 0)
        rows_pp, columns_pp = np.meshgrid(present, present, indexing="ij")
        pp = set(
            np.ravel_multi_index(
                (rows_pp.ravel(), columns_pp.ravel()), case.beta.shape
            ).tolist()
        )
        for arm, surgery in zip(batch.arm_names, batch.surgeries, strict=True):
            if surgery is None:
                continue
            changed = set(np.asarray(surgery.flat_indices, dtype=np.int64).tolist())
            log_changes = np.log(surgery.after / surgery.before)
            rows.append(
                {
                    "state_id": case.state_id,
                    "candidate": case.candidate,
                    "matrix_id": case.matrix_id,
                    "landmark": case.landmark,
                    "arm": arm,
                    "present_types": int(present.size),
                    "present_present_edges": len(pp),
                    "distinct_changed_edges": len(changed),
                    "all_and_only_present_present": changed == pp,
                    "requested_frobenius_norm": surgery.requested_norm,
                    "observed_frobenius_norm": surgery.observed_norm,
                    "norm_absolute_error": abs(
                        surgery.observed_norm - surgery.requested_norm
                    ),
                    "norm_ratio": surgery.observed_norm / surgery.requested_norm,
                    "log_change_sum": float(log_changes.sum()),
                    "minimum_beta_after": float(surgery.beta.min()),
                    "all_beta_positive": bool(np.all(surgery.beta > 0.0)),
                    "constant_multiplicative_ratio": bool(
                        arm not in {"SMALL_RANDOM_PP", "FABLE_RANDOM_PP"}
                        and np.array_equal(
                            surgery.after,
                            surgery.before
                            * (
                                SMALL_LOOSEN_FACTOR
                                if arm == "SMALL_LOOSEN"
                                else SMALL_TIGHTEN_FACTOR
                                if arm == "SMALL_TIGHTEN"
                                else FABLE_LOOSEN_FACTOR
                                if arm == "FABLE_LOOSEN"
                                else FABLE_TIGHTEN_FACTOR
                            ),
                        )
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    random_rows = frame[frame["arm"].isin(("SMALL_RANDOM_PP", "FABLE_RANDOM_PP"))]
    targeted_rows = frame[~frame["arm"].isin(("SMALL_RANDOM_PP", "FABLE_RANDOM_PP"))]
    summary = {
        "format": "codex-intervention-p3b-surgery-audit-v1",
        "states": len(cases),
        "surgery_rows": len(frame),
        "expected_surgery_rows": len(cases) * 6,
        "all_and_only_present_present": bool(frame["all_and_only_present_present"].all()),
        "all_beta_positive": bool(frame["all_beta_positive"].all()),
        "maximum_norm_absolute_error": float(frame["norm_absolute_error"].max()),
        "maximum_norm_relative_error": float(np.max(np.abs(frame["norm_ratio"] - 1.0))),
        "random_maximum_absolute_log_change_sum": float(
            random_rows["log_change_sum"].abs().max()
        ),
        "all_targeted_ratios_exact": bool(
            targeted_rows["constant_multiplicative_ratio"].all()
        ),
        "random_control_location_norm_positivity_audit_pass": bool(
            random_rows["all_and_only_present_present"].all()
            and random_rows["all_beta_positive"].all()
            and np.max(np.abs(random_rows["norm_ratio"] - 1.0)) <= 1e-12
            and random_rows["log_change_sum"].abs().max() <= 1e-10
        ),
    }
    if not (
        summary["surgery_rows"] == summary["expected_surgery_rows"]
        and summary["all_and_only_present_present"]
        and summary["all_beta_positive"]
        and summary["all_targeted_ratios_exact"]
        and summary["random_control_location_norm_positivity_audit_pass"]
    ):
        raise AssertionError("P3b surgery audit failed")
    return frame, summary


def _technical_report(
    metrics: dict[str, Any],
    replay: dict[str, Any],
    audit: dict[str, Any],
    registration_id: str,
) -> str:
    def scope_table(scope: dict[str, Any]) -> list[str]:
        lines = [
            "| Cell | Fable-strength effect | 95% CI | Holm p | Random−no-op 90% CI | Cell pass |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for cell in scope["cells"]:
            effect = cell["contrasts"]["fable_effect"]
            random_ci = cell["fable_random_noop_equivalence"]["bootstrap_ci90"]
            lines.append(
                f"| {cell['cell']} | {effect['estimate']:.6f} | "
                f"[{effect['bootstrap_ci95'][0]:.6f}, {effect['bootstrap_ci95'][1]:.6f}] | "
                f"{cell['fable_effect_randomization_p_holm']:.6g} | "
                f"[{random_ci[0]:.6f}, {random_ci[1]:.6f}] | "
                f"{cell['primary_fable_cell_pass_without_replay']} |"
            )
        return lines

    return "\n".join(
        [
            "# P3b beta-surgery dose-and-contract bridge",
            "",
            "## Registered outcome",
            "",
            f"Landmark-60 Fable-strength replication gate: **{metrics['primary_replication_gate_pass']}**.",
            f"Five-landmark generalization gate: **{metrics['five_landmark_generalization_gate_pass']}**.",
            f"Landmark-60 two-dose ordering gate: **{metrics['landmark60_two_dose_gate_pass']}**.",
            f"Five-landmark two-dose ordering gate: **{metrics['five_landmark_two_dose_gate_pass']}**.",
            f"Exact replay: **{replay['state_edit_endpoint_and_process_digests_exact']}**.",
            "",
            "The causal sign is `LOOSEN − TIGHTEN`: positive values mean weakening the occupied catalytic web causes more break-and-renewal than strengthening it.",
            "",
            "## Primary: landmark 60",
            "",
            *scope_table(metrics["primary"]),
            "",
            "## Registered five-landmark generalization",
            "",
            *scope_table(metrics["generalization"]),
            "",
            "## Contract and audit",
            "",
            "- 80 fresh matrices, two candidates, six landmarks, seven arms, and 32 branches per arm/state.",
            "- The Fable-strength pair is `×1.5` versus `÷1.5`; the small pair is `×1.05` versus `×0.95`.",
            "- Random controls changed all and only the same present-present block and achieved their registered norm exactly within numerical tolerance.",
            f"- Maximum random/target norm relative audit error: `{audit['maximum_norm_relative_error']:.3g}`.",
            "- Common random streams were paired across arms; arm identity was absent from future seed keys.",
            f"- Registration: `{registration_id}`.",
            "",
            "## Interpretation boundary",
            "",
            "The sealed P3 5% finding remains unchanged. P3b is its separately registered contract correction and dose bridge. A passing gate supports causal control of the simulated JOINT_BREAK_RUN3 process under Codex's contracts; it does not establish life, biological memory, agency, autonomous organization, real chemistry, or strict-eight control.",
            "",
            "## Mandatory stop",
            "",
            "P3b is sealed without automatically launching another intervention phase.",
            "",
        ]
    )


def _lay_report(metrics: dict[str, Any], replay: dict[str, Any]) -> str:
    primary = (
        "The correctly sized intervention passed every prewritten candidate-and-half test at the matching generation-60 state."
        if metrics["primary_replication_gate_pass"]
        else "The correctly sized intervention did not pass every prewritten candidate-and-half test at the matching generation-60 state."
    )
    generalization = (
        "It also generalized across the five ordinary saved-state ages."
        if metrics["five_landmark_generalization_gate_pass"]
        else "It did not satisfy the full generalization test across all five ordinary saved-state ages."
    )
    dose = (
        "The large intervention was reliably stronger than the small one, giving a registered two-dose ordering."
        if metrics["landmark60_two_dose_gate_pass"]
        else "The large intervention was not reliably stronger in every required cell, so we do not claim a graded two-dose result."
    )
    return "\n".join(
        [
            "# Lay summary",
            "",
            "Our earlier experiment nudged the catalytic rulebook by only about 5%. We later learned that Fable's real test made a much larger symmetric-in-log-space change: strengthening the links among molecules currently present by 50%, or weakening those links by dividing by 1.5. P3b tested both sizes prospectively on entirely new simulated assemblies.",
            "",
            primary,
            generalization,
            dose,
            f"Every simulated future was repeated exactly: **{replay['state_edit_endpoint_and_process_digests_exact']}**.",
            "",
            "The clean causal question is whether changing only the catalytic links—while leaving the molecules and their past untouched—changes the chance of losing heredity and then rebuilding a short inherited run. Random changes to the same links at the same size serve as the specificity control.",
            "",
            "Whatever the outcome, this concerns control inside a deliberately simplified computer model. It does not show that the assemblies are alive or autonomously remember and repair themselves.",
            "",
        ]
    )


def _append_result_ledger(
    output: Path,
    metrics: dict[str, Any],
    replay: dict[str, Any],
    registration_id: str,
) -> None:
    path = REPOSITORY_ROOT / "INTERVENTION_RESULTS_LEDGER.md"
    text = path.read_text(encoding="utf-8").rstrip() + "\n"
    marker = f"<!-- sealed-p3b-dose-bridge-{registration_id} -->"
    if marker in text:
        return
    lines = [
        "",
        marker,
        "## P3b beta-surgery dose bridge sealed",
        "",
        f"- Registration: `{registration_id}`",
        f"- Result bundle: `{output.relative_to(REPOSITORY_ROOT)}`",
        f"- Landmark-60 Fable-strength gate: **{metrics['primary_replication_gate_pass']}**",
        f"- Five-landmark generalization gate: **{metrics['five_landmark_generalization_gate_pass']}**",
        f"- Landmark-60 two-dose gate: **{metrics['landmark60_two_dose_gate_pass']}**",
        f"- Five-landmark two-dose gate: **{metrics['five_landmark_two_dose_gate_pass']}**",
        f"- Exact replay: **{replay['state_edit_endpoint_and_process_digests_exact']}**",
        "- The sealed P3 result remains the unchanged 5% predecessor; P3b used the clarified `×1.5`/`÷1.5` Fable-strength contract.",
        "- Status: sealed and stopped; no later phase launched automatically.",
        "",
        "| Scope | Cell | Fable-strength effect | 95% whole-matrix CI | Holm p | Random TOST |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for scope in (metrics["primary"], metrics["generalization"]):
        for cell in scope["cells"]:
            effect = cell["contrasts"]["fable_effect"]
            lines.append(
                f"| {scope['scope']} | {cell['cell']} | {effect['estimate']:.6f} | "
                f"[{effect['bootstrap_ci95'][0]:.6f}, {effect['bootstrap_ci95'][1]:.6f}] | "
                f"{cell['fable_effect_randomization_p_holm']:.6g} | "
                f"{cell['fable_random_noop_equivalence']['tost_equivalent']} |"
            )
    path.write_text(text + "\n".join(lines) + "\n", encoding="utf-8")


def run_campaign(
    registration_directory: Path,
    output_directory: Path,
    workers: int,
    work_directory: Path,
) -> None:
    if workers < 1:
        raise ValueError("workers must be positive")
    registration_directory = registration_directory.resolve()
    output_directory = output_directory.resolve()
    work = work_directory.resolve()
    registration = verify_registration(registration_directory)
    spec = phase_spec()
    experiment = _experiment(spec)
    _prepare_campaign(work, output_directory, registration, spec)

    expected_states = 2 * spec.matrices * len(LANDMARKS)
    print(
        f"[p3b 1/8] Building {spec.matrices} fresh matrices and "
        f"{expected_states} natural restored states",
        flush=True,
    )
    _campaign_status(work, "running", "building_natural_trajectories")
    with threadpool_limits(limits=1):
        cases = build_cohort(experiment, LABEL, experiment.confirmation)
    if len(cases) != expected_states:
        raise AssertionError("fresh P3b cohort has the wrong state count")

    model_path = registration_directory / "frozen_full_predictor.npz"
    futures = len(cases) * len(spec.arms) * spec.branches
    print(
        f"[p3b 2/8] Applying seven frozen arms and shooting {futures:,} F12 futures",
        flush=True,
    )
    _campaign_status(work, "running", "selecting_surgeries_and_shooting_futures")
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
    print(f"[p3b 3/8] Replaying all {futures:,} F12 futures", flush=True)
    _campaign_status(work, "running", "complete_exact_replay")
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
    replay = base.replay_audit(generated, replayed)
    replay_exact = replay["state_edit_endpoint_and_process_digests_exact"]
    if not replay_exact:
        raise AssertionError("P3b exact replay failed")

    print("[p3b 4/8] Computing frozen whole-matrix inference and surgery audits", flush=True)
    _campaign_status(work, "running", "whole_matrix_inference_and_surgery_audit")
    arrays = base._outcome_arrays(cases, generated, spec)
    draws = generate_inference_draws(
        spec.matrices,
        BOOTSTRAP_REPETITIONS,
        RANDOMIZATION_REPETITIONS,
        np.random.default_rng(derive_seed(spec.bootstrap_seed, f"{LABEL}.bootstrap")),
        np.random.default_rng(
            derive_seed(spec.randomization_seed, f"{LABEL}.randomization")
        ),
    )
    metrics, matrix_rows = compute_bridge_inference(
        cases, arrays["targets"], arrays["predictions"], draws
    )
    add_replay_gates(metrics, replay_exact)
    secondary = base._secondary_descriptives(cases, arrays, spec)
    surgery_rows, surgery_summary = _surgery_audit(cases, generated)

    print("[p3b 5/8] Writing and readback-checking complete artifacts", flush=True)
    _campaign_status(work, "running", "writing_and_readback_checking_artifacts")
    with _atomic_destination(output_directory) as output:
        np.savez_compressed(output / "branch_arrays.npz", **arrays)
        base._write_branch_table(output / "branches.csv.gz", cases, generated)
        base._write_state_artifacts(output, cases, generated, arrays)
        base._write_selection_artifacts(output, cases, generated, spec)
        surgery_rows.to_csv(output / "surgery_norm_audit.csv", index=False)
        (output / "surgery_audit_summary.json").write_text(
            json.dumps(_json_ready(surgery_summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
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
            output, cases, metrics, matrix_rows, replay_exact
        )
        (output / "readback_audit.json").write_text(
            json.dumps(readback, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output / "SCIENTIFIC_REPORT.md").write_text(
            _technical_report(
                metrics, replay, surgery_summary, registration["registration_id"]
            ),
            encoding="utf-8",
        )
        (output / "LAY_SUMMARY.md").write_text(
            _lay_report(metrics, replay), encoding="utf-8"
        )
        supported: list[str] = []
        failed: list[str] = []
        decisions = (
            (
                "landmark-60 qualitative cross-clean-room beta-surgery replication",
                metrics["primary_replication_gate_pass"],
            ),
            (
                "five-landmark beta-surgery generalization",
                metrics["five_landmark_generalization_gate_pass"],
            ),
            (
                "landmark-60 graded two-dose ordering",
                metrics["landmark60_two_dose_gate_pass"],
            ),
            (
                "five-landmark graded two-dose ordering",
                metrics["five_landmark_two_dose_gate_pass"],
            ),
        )
        for statement, passed in decisions:
            (supported if passed else failed).append(statement)
        claim_boundary = {
            "supported_claims": supported,
            "failed_predictions": failed,
            "deviations": [],
            "unresolved_questions": [
                "whether network surgery acts mainly on break resistance or post-break renewal",
                "whether repeated physical intervention can maintain heredity over long horizons",
                "whether any maintained organization persists autonomously after release",
            ],
            "prohibited_interpretations": _protocol()["claim_boundary"]["prohibited"],
        }
        (output / "claim_boundaries.json").write_text(
            json.dumps(claim_boundary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "format": RESULT_FORMAT,
            "phase": spec.phase,
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
            "landmark60_primary_gate": metrics["primary_replication_gate_pass"],
            "five_landmark_generalization_gate": metrics[
                "five_landmark_generalization_gate_pass"
            ],
            "landmark60_two_dose_gate": metrics["landmark60_two_dose_gate_pass"],
            "five_landmark_two_dose_gate": metrics[
                "five_landmark_two_dose_gate_pass"
            ],
            "exact_replay": replay_exact,
            "complete_readback_exact": True,
            "surgery_audit_pass": True,
            "sealed_p3_preserved": True,
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
        (output / "CUMULATIVE_RESULTS_LEDGER.md").write_text(
            "\n".join(
                [
                    "# Intervention result ledger snapshot",
                    "",
                    "Phase: `p3b_dose_bridge`",
                    f"Registration: `{registration['registration_id']}`",
                    f"Landmark-60 primary gate: **{metrics['primary_replication_gate_pass']}**",
                    f"Five-landmark generalization gate: **{metrics['five_landmark_generalization_gate_pass']}**",
                    f"Landmark-60 two-dose gate: **{metrics['landmark60_two_dose_gate_pass']}**",
                    f"Five-landmark two-dose gate: **{metrics['five_landmark_two_dose_gate_pass']}**",
                    "Exact replay: **True**",
                    "Next phase: not launched; mandatory review stop.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        print("[p3b 6/8] Sealing and checksum-verifying result", flush=True)
        write_checksums(output)
    verify_checksums(output_directory)
    _append_result_ledger(
        output_directory, metrics, replay, registration["registration_id"]
    )
    _campaign_status(work, "sealed_complete", "mandatory_review_stop")
    print(f"[p3b 7/8] Result sealed: {output_directory}", flush=True)
    print("[p3b 8/8] STOPPED as registered; no later phase launched", flush=True)


def read_status(work_directory: Path) -> dict[str, Any]:
    return base.read_status(work_directory)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="P3b beta-surgery dose bridge")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--output", type=Path, default=DEFAULT_VALIDATION)
    register = commands.add_parser("register")
    register.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    register.add_argument("--output", type=Path, default=DEFAULT_REGISTRATION)
    verify = commands.add_parser("verify")
    verify.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    smoke = commands.add_parser("smoke")
    smoke.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    smoke.add_argument(
        "--output", type=Path, default=RESULT_ROOT / "p3b_dose_bridge_smoke"
    )
    smoke.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 4))
    run = commands.add_parser("run")
    run.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    run.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    run.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 14))
    status = commands.add_parser("status")
    status.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    return parser


def main(argv: list[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    if arguments.command == "validate":
        run_validation(arguments.output)
    elif arguments.command == "register":
        register_program(arguments.validation, arguments.output)
    elif arguments.command == "verify":
        print(
            json.dumps(
                verify_registration(arguments.registration), indent=2, sort_keys=True
            )
        )
    elif arguments.command == "smoke":
        run_smoke(arguments.registration, arguments.output, arguments.workers)
    elif arguments.command == "run":
        run_campaign(
            arguments.registration,
            arguments.output,
            arguments.workers,
            arguments.work_dir,
        )
    elif arguments.command == "status":
        print(json.dumps(read_status(arguments.work_dir), indent=2, sort_keys=True))
    else:  # pragma: no cover
        raise AssertionError(arguments.command)


if __name__ == "__main__":
    main()
