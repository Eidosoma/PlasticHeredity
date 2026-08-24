"""Prospective zero-shot parameter-regime transfer confirmation (CR6)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from threadpoolctl import threadpool_limits

from . import intervention_cr1_confirmation as cr1
from . import intervention_replication as base
from .config import CANDIDATES, CohortConfig, ExperimentConfig, GardConfig
from .experiment import StateCase, _json_ready, _runtime_manifest, build_cohort
from .intervention_core import FrozenFullPredictor
from .intervention_metrics import compute_one_shot_inference, generate_inference_draws
from .mechanistic import (
    _atomic_destination,
    _canonical_digest,
    sha256_file,
    verify_checksums,
    write_checksums,
)
from .seeds import derive_seed
from .simulator import Snapshot


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results_intervention_replication"
CR1_REGISTRATION = RESULT_ROOT / "cr1_confirmation_registration"
CR1_RESULT = RESULT_ROOT / "cr1_model_guided_confirmation"
DEFAULT_VALIDATION = RESULT_ROOT / "cr6_transfer_validation"
DEFAULT_REGISTRATION = RESULT_ROOT / "cr6_transfer_registration"
DEFAULT_SMOKE = RESULT_ROOT / "cr6_transfer_smoke"
DEFAULT_OUTPUT = RESULT_ROOT / "cr6_zero_shot_transfer"
DEFAULT_WORK = RESULT_ROOT / ".cr6_zero_shot_transfer_work"

DOCUMENT = "CODEX_INTERVENTION_CR6_PREREGISTRATION.md"
SOURCE_FILES = (
    DOCUMENT,
    "plastic_heredity/intervention_cr6_transfer.py",
    "tests/test_intervention_cr6_transfer.py",
    "plastic_heredity/intervention_cr1_confirmation.py",
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

PROGRAM_FORMAT = "codex-intervention-cr6-zero-shot-transfer-v1"
VALIDATION_FORMAT = "codex-intervention-cr6-validation-v1"
REGISTRATION_FORMAT = "codex-intervention-cr6-registration-v1"
RESULT_FORMAT = "codex-intervention-cr6-result-v1"
STATUS_FORMAT = "codex-intervention-cr6-status-v1"

CR1_REGISTRATION_ID = "a8743234235e82133d2938c15ead062c7c85004c5f640d7359e5b075cb31368e"
EXPECTED_MODEL_SHA256 = (
    "9b3305a7fed11f432651926d34903443e9413ed299c5d0f1056a0b5fde9990af"
)

REGIMES: dict[str, tuple[float, float, str]] = {
    "POS_A_M4_S5": (-4.0, 5.0, "positive_transfer"),
    "POS_A_M3_S4": (-3.0, 4.0, "positive_transfer"),
    "POS_A_M5_S4": (-5.0, 4.0, "positive_transfer"),
    "NULL_A_M4_S3": (-4.0, 3.0, "predicted_null"),
}
POSITIVE_REGIMES = tuple(
    key for key, value in REGIMES.items() if value[2] == "positive_transfer"
)
NULL_REGIME = "NULL_A_M4_S3"
MATRICES = 40
LANDMARKS = (35, 65)
BRANCHES = 48
HORIZON = 12
ARMS = ("MODEL_UP", "MODEL_DOWN", "RANDOM", "NOOP")
BOOTSTRAP_REPETITIONS = 4_096
RANDOMIZATION_REPETITIONS = 4_096
RANDOM_EQUIVALENCE_MARGIN = 0.025
NULL_EQUIVALENCE_MARGIN = 0.04
MINIMUM_CPU_BUDGET_HOURS = 3.0
MAXIMUM_CPU_BUDGET_HOURS = 6.0
DEFAULT_CPU_BUDGET_HOURS = 5.0
MINIMUM_FREE_DISK_BYTES = 3_000_000_000


def _seed(name: str) -> str:
    return hashlib.sha256(
        f"codex-clean-room-cr6-zero-shot-transfer-v1::{name}".encode()
    ).hexdigest()


SEEDS = {
    name: _seed(name)
    for name in (
        "validation",
        "smoke_cohort",
        "smoke_selection",
        "smoke_future",
        "replay",
        *(
            f"{regime}__{purpose}"
            for regime in REGIMES
            for purpose in (
                "cohort",
                "selection",
                "future",
                "bootstrap",
                "randomization",
            )
        ),
    )
}


def source_hashes() -> dict[str, str]:
    return {name: sha256_file(ROOT / name) for name in SOURCE_FILES}


def protocol() -> dict[str, Any]:
    value: dict[str, Any] = {
        "format": PROGRAM_FORMAT,
        "status": "frozen_before_any_cr6_scientific_matrix",
        "upstream": {
            "cr1_registration_id": CR1_REGISTRATION_ID,
            "cr1_full_four_cell_gate_passed": True,
            "cr5r_not_used_to_select_cr6_design": True,
        },
        "endpoint": {
            "name": "JOINT_BREAK_RUN3",
            "horizon": HORIZON,
            "inheritance": "strict unrounded float64 H > 0.9",
            "break": "H <= 0.9",
            "strict_eight_excluded": True,
        },
        "frozen_model": {
            "source": "sealed CR1 candidate-separated 5x full composite",
            "sha256": EXPECTED_MODEL_SHA256,
            "zero_shot": True,
            "refit_recalibration_search_or_regime_switching": False,
        },
        "regimes": {
            key: {"A": a, "sigma": sigma, "role": role}
            for key, (a, sigma, role) in REGIMES.items()
        },
        "cohort_per_regime": {
            "matrices": MATRICES,
            "candidates": list(CANDIDATES),
            "landmarks": list(LANDMARKS),
            "states": 2 * MATRICES * len(LANDMARKS),
            "arms": list(ARMS),
            "branches": BRANCHES,
            "halves": {"A": [0, 23], "B": [24, 47]},
            "primary_futures": 2 * MATRICES * len(LANDMARKS) * len(ARMS) * BRANCHES,
            "complete_replay": True,
            "no_retry_or_replacement": True,
        },
        "selection": {
            "every_legal_swap_scored": True,
            "deterministic_extreme_ties": True,
            "random_uniform_over_legal_swaps": True,
            "optional_rule_family_included": False,
        },
        "randomness": {
            "purpose_and_regime_keyed_domains": True,
            "future_seed_excludes_arm": True,
            "common_random_streams": True,
            "random_selection_stream_separate": True,
            "seed_domains": SEEDS,
        },
        "positive_transfer_gate": {
            "regimes": list(POSITIVE_REGIMES),
            "cells_per_regime": "candidate by fixed branch half",
            "holm_family_per_regime": 4,
            "up_minus_down_positive": True,
            "bootstrap_ci95_lower_positive": True,
            "holm_randomization_p_below": 0.05,
            "random_noop_tost_margin": RANDOM_EQUIVALENCE_MARGIN,
            "up_noop_noop_down_and_ratio_reported_not_gated": True,
        },
        "predicted_null_gate": {
            "regime": NULL_REGIME,
            "candidate_pooled_across_all_48_branches": True,
            "tost_interval": "90% whole-matrix bootstrap",
            "margin": NULL_EQUIVALENCE_MARGIN,
            "both_candidates_required": True,
            "ci_crossing_zero_is_not_equivalence": True,
            "candidate_half_results_descriptive": True,
        },
        "inference": {
            "unit": "whole catalytic matrix within regime",
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "randomization_repetitions": RANDOMIZATION_REPETITIONS,
            "regimes_never_pooled": True,
            "candidates_never_pooled": True,
        },
        "operational": {
            "minimum_free_disk_bytes": MINIMUM_FREE_DISK_BYTES,
            "cpu_budget_hours": [
                MINIMUM_CPU_BUDGET_HOURS,
                MAXIMUM_CPU_BUDGET_HOURS,
            ],
            "expected_cpu_hours": [3.0, 5.0],
            "checkpoint_resumable": True,
            "mandatory_stop_after_seal": True,
            "cr7_not_launched_automatically": True,
        },
        "claim_boundary": {
            "prohibited": [
                "universal transfer",
                "strict-eight control",
                "biological memory",
                "agency or life",
                "autonomous organization or installed attractor",
                "real prebiotic chemistry",
                "universal origin-of-life mechanism",
                "Phi or PhiID intervention",
            ]
        },
    }
    value["protocol_id"] = _canonical_digest(_json_ready(value))
    return value


def regime_gard(regime: str) -> GardConfig:
    if regime not in REGIMES:
        raise ValueError(f"unknown CR6 regime: {regime}")
    a, sigma, _ = REGIMES[regime]
    return replace(GardConfig(), beta_log_mean=a, beta_log_sd=sigma)


def phase_spec(regime: str) -> base.PhaseSpec:
    if regime not in REGIMES:
        raise ValueError(f"unknown CR6 regime: {regime}")
    return base.PhaseSpec(
        phase="p1",
        role=f"CR6 zero-shot transfer {regime}",
        matrices=MATRICES,
        branches=BRANCHES,
        cohort_seed=SEEDS[f"{regime}__cohort"],
        selection_seed=SEEDS[f"{regime}__selection"],
        future_seed=SEEDS[f"{regime}__future"],
        bootstrap_seed=SEEDS[f"{regime}__bootstrap"],
        randomization_seed=SEEDS[f"{regime}__randomization"],
    )


def experiment(regime: str) -> ExperimentConfig:
    spec = phase_spec(regime)
    cohort = CohortConfig(MATRICES, BRANCHES, LANDMARKS)
    return ExperimentConfig(
        gard=regime_gard(regime),
        development=cohort,
        confirmation=cohort,
        horizon=HORIZON,
        bootstrap_repetitions=BOOTSTRAP_REPETITIONS,
        permutation_repetitions=RANDOMIZATION_REPETITIONS,
        regenerate_confirmation=True,
        master_seed=spec.cohort_seed,
    )


def cohort_label(regime: str) -> str:
    return f"INTCR6_ZERO_SHOT_{regime}_V1"


def _interval(samples: NDArray[np.float64], alpha: float = 0.05) -> tuple[float, float]:
    low, high = np.quantile(samples, (alpha / 2.0, 1.0 - alpha / 2.0))
    return float(low), float(high)


def _tost_equivalent(interval: tuple[float, float], margin: float) -> bool:
    """Return true only when the complete 90% interval is within the margin."""

    return bool(interval[0] > -margin and interval[1] < margin)


def _matrix_means(
    values: NDArray[np.float64],
    matrix_ids: NDArray[np.int64],
    matrix_order: NDArray[np.int64],
) -> NDArray[np.float64]:
    return np.asarray(
        [values[matrix_ids == matrix_id].mean() for matrix_id in matrix_order],
        dtype=np.float64,
    )


def _maximum_leave_one_out_influence(values: NDArray[np.float64]) -> float:
    if values.size < 2:
        return float("nan")
    observed = float(values.mean())
    leave_one_out = (values.sum() - values) / (values.size - 1)
    return float(np.max(np.abs(leave_one_out - observed)))


def _candidate_pooled_inference(
    cases: list[StateCase],
    targets: NDArray,
    predictions: NDArray,
    draws: dict[str, NDArray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[float]]]:
    """Compute the registered all-branch candidate summaries.

    The predicted-null decision is based on the 90% bootstrap interval for
    MODEL_UP minus MODEL_DOWN.  Both landmarks are reduced to one value per
    catalytic matrix before resampling.
    """

    target_array = np.asarray(targets, dtype=np.float64)
    prediction_array = np.asarray(predictions, dtype=np.float64)
    if target_array.shape != (len(cases), len(ARMS), BRANCHES):
        raise ValueError("CR6 target array does not match the frozen design")
    if prediction_array.shape != (len(cases), len(ARMS)):
        raise ValueError("CR6 prediction array does not match the frozen design")
    bootstrap_indices = np.asarray(draws["bootstrap_indices"], dtype=np.int64)
    matrix_order = np.unique(
        np.asarray([case.matrix_id for case in cases], dtype=np.int64)
    )
    if matrix_order.size != MATRICES or bootstrap_indices.shape[1] != MATRICES:
        raise ValueError("CR6 pooled inference lost a whole-matrix block")
    arm_index = {arm: index for index, arm in enumerate(ARMS)}
    summaries: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    stored: dict[str, list[float]] = {}
    for candidate in CANDIDATES:
        selected = np.asarray(
            [case.candidate == candidate for case in cases], dtype=bool
        )
        candidate_cases = [case for case in cases if case.candidate == candidate]
        ids = np.asarray([case.matrix_id for case in candidate_cases], dtype=np.int64)
        if not np.array_equal(np.unique(ids), matrix_order):
            raise ValueError(f"candidate {candidate} lacks a complete CR6 cohort")
        q = target_array[selected].mean(axis=2)
        effects = {
            "up_minus_down": q[:, arm_index["MODEL_UP"]]
            - q[:, arm_index["MODEL_DOWN"]],
            "up_minus_noop": q[:, arm_index["MODEL_UP"]] - q[:, arm_index["NOOP"]],
            "noop_minus_down": q[:, arm_index["NOOP"]] - q[:, arm_index["MODEL_DOWN"]],
            "random_minus_noop": q[:, arm_index["RANDOM"]] - q[:, arm_index["NOOP"]],
        }
        matrix_effects = {
            name: _matrix_means(values, ids, matrix_order)
            for name, values in effects.items()
        }
        bootstraps = {
            name: values[bootstrap_indices].mean(axis=1)
            for name, values in matrix_effects.items()
        }
        for name, values in bootstraps.items():
            stored[f"c{candidate}__pooled__bootstrap__{name}"] = values.tolist()

        arms: dict[str, Any] = {}
        for arm, index in arm_index.items():
            matrix_q = _matrix_means(q[:, index], ids, matrix_order)
            arm_bootstrap = matrix_q[bootstrap_indices].mean(axis=1)
            stored[f"c{candidate}__pooled__bootstrap__q_{arm}"] = arm_bootstrap.tolist()
            arms[arm] = {
                "mean_probability": float(matrix_q.mean()),
                "bootstrap_ci95": _interval(arm_bootstrap),
                "mean_frozen_prediction": float(
                    prediction_array[selected, index].mean()
                ),
            }

        contrasts: dict[str, Any] = {}
        for name, values in matrix_effects.items():
            contrasts[name] = {
                "estimate": float(values.mean()),
                "bootstrap_ci95": _interval(bootstraps[name]),
                "bootstrap_ci90": _interval(bootstraps[name], alpha=0.10),
                "matrices_expected_sign": int(np.count_nonzero(values > 0.0)),
                "matrices_zero": int(np.count_nonzero(values == 0.0)),
                "maximum_leave_one_matrix_out_influence": (
                    _maximum_leave_one_out_influence(values)
                ),
            }
        targeted_ci90 = contrasts["up_minus_down"]["bootstrap_ci90"]
        random_ci90 = contrasts["random_minus_noop"]["bootstrap_ci90"]
        predicted_shift = (
            prediction_array[selected, arm_index["MODEL_UP"]]
            - prediction_array[selected, arm_index["MODEL_DOWN"]]
        )
        summary = {
            "candidate": candidate,
            "states": int(selected.sum()),
            "matrices": int(matrix_order.size),
            "branches": BRANCHES,
            "arms": arms,
            "contrasts": contrasts,
            "targeted_null_equivalence": {
                "margin": NULL_EQUIVALENCE_MARGIN,
                "bootstrap_ci90": targeted_ci90,
                "tost_equivalent": _tost_equivalent(
                    targeted_ci90, NULL_EQUIVALENCE_MARGIN
                ),
                "ci_crossing_zero_alone_not_equivalence": True,
            },
            "random_noop_equivalence": {
                "margin": RANDOM_EQUIVALENCE_MARGIN,
                "bootstrap_ci90": random_ci90,
                "tost_equivalent": _tost_equivalent(
                    random_ci90, RANDOM_EQUIVALENCE_MARGIN
                ),
            },
            "mean_predicted_up_minus_down": float(predicted_shift.mean()),
        }
        summaries.append(summary)
        for position, matrix_id in enumerate(matrix_order):
            row: dict[str, Any] = {
                "cell": f"c{candidate}_ALL",
                "candidate": candidate,
                "branch_half": "ALL",
                "matrix_id": int(matrix_id),
            }
            row.update(
                {
                    name: float(values[position])
                    for name, values in matrix_effects.items()
                }
            )
            rows.append(row)
    return summaries, rows, stored


def compute_regime_inference(
    regime: str,
    cases: list[StateCase],
    targets: NDArray,
    predictions: NDArray,
    draws: dict[str, NDArray],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Apply the preregistered positive-transfer or predicted-null gate."""

    if regime not in REGIMES:
        raise ValueError(f"unknown CR6 regime: {regime}")
    metrics, matrix_rows = compute_one_shot_inference(
        cases,
        ARMS,
        targets,
        predictions,
        draws,
        up_arm="MODEL_UP",
        down_arm="MODEL_DOWN",
        equivalence_margin=RANDOM_EQUIVALENCE_MARGIN,
        random_ratio_limit=0.25,
    )
    role = REGIMES[regime][2]
    for cell in metrics["cells"]:
        targeted = cell["contrasts"]["up_minus_down"]
        gates = {
            "up_minus_down_positive": targeted["estimate"] > 0.0,
            "up_minus_down_bootstrap_lower_positive": (
                targeted["bootstrap_ci95"][0] > 0.0
            ),
            "holm_randomization_below_0_05": (
                cell["up_down_randomization_p_holm"] < 0.05
            ),
            "random_tost_equivalent_to_noop": cell["random_noop_equivalence"][
                "tost_equivalent"
            ],
        }
        cell["cr6_positive_transfer_gates"] = gates
        cell["cr6_positive_transfer_cell_pass"] = bool(all(gates.values()))

    pooled, pooled_rows, pooled_stored = _candidate_pooled_inference(
        cases, targets, predictions, draws
    )
    raw_stored = metrics["stored_inference_arrays"]
    raw_stored["candidate_pooled_bootstraps"] = pooled_stored
    matrix_rows.extend(pooled_rows)
    metrics.update(
        {
            "regime": regime,
            "A": REGIMES[regime][0],
            "sigma": REGIMES[regime][1],
            "registered_role": role,
            "candidate_pooled": pooled,
            "positive_transfer_gate_pass": (
                bool(
                    len(metrics["cells"]) == 4
                    and all(
                        cell["cr6_positive_transfer_cell_pass"]
                        for cell in metrics["cells"]
                    )
                )
                if role == "positive_transfer"
                else None
            ),
            "predicted_null_gate_pass": (
                bool(
                    len(pooled) == 2
                    and all(
                        candidate["targeted_null_equivalence"]["tost_equivalent"]
                        for candidate in pooled
                    )
                )
                if role == "predicted_null"
                else None
            ),
            "positive_gate_uses_four_candidate_half_cells": True,
            "null_gate_uses_candidate_pooled_all_branches": True,
            "up_noop_noop_down_and_ratio_reported_not_gated": True,
        }
    )
    metrics["registered_regime_gate_pass"] = (
        metrics["positive_transfer_gate_pass"]
        if role == "positive_transfer"
        else metrics["predicted_null_gate_pass"]
    )
    return metrics, matrix_rows


def _normalized_stored_arrays(
    stored: dict[str, Any], path_name: str = "inference_arrays.npz"
) -> dict[str, Any]:
    return {
        "path": path_name,
        "bootstrap_indices_shape": stored["bootstrap_indices_shape"],
        "randomization_signs_shape": stored["randomization_signs_shape"],
        "all_cell_bootstrap_and_randomization_arrays_stored": True,
        "all_candidate_pooled_bootstrap_arrays_stored": True,
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
        arrays[f"{cell}__randomization_null"] = np.asarray(values, dtype=np.float64)
    for name, values in stored["candidate_pooled_bootstraps"].items():
        arrays[name] = np.asarray(values, dtype=np.float64)
    np.savez_compressed(path, **arrays)
    metrics["stored_inference_arrays"] = _normalized_stored_arrays(stored, path.name)


def _readback_regime(
    output: Path,
    regime: str,
    cases: list[StateCase],
    expected: dict[str, Any],
    expected_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    with np.load(output / "branch_arrays.npz", allow_pickle=False) as archive:
        targets = archive["targets"]
        predictions = archive["predictions"]
    with np.load(output / "inference_arrays.npz", allow_pickle=False) as archive:
        draws = {
            "bootstrap_indices": archive["bootstrap_indices"],
            "randomization_signs": archive["randomization_signs"],
        }
    observed, observed_rows = compute_regime_inference(
        regime, cases, targets, predictions, draws
    )
    stored = observed.pop("stored_inference_arrays")
    observed["stored_inference_arrays"] = _normalized_stored_arrays(stored)
    metrics_exact = _json_ready(observed) == _json_ready(expected)
    rows_exact = _json_ready(observed_rows) == _json_ready(expected_rows)
    if not metrics_exact or not rows_exact:
        raise ValueError(f"CR6 {regime} written-artifact inference changed")
    return {
        "branch_arrays_reloaded": True,
        "inference_draws_reloaded": True,
        "primary_metrics_exact": metrics_exact,
        "matrix_effects_exact": rows_exact,
        "no_fitting_or_recalibration": True,
    }


def _verify_predecessor() -> dict[str, Any]:
    registration = cr1.verify_registration(CR1_REGISTRATION)
    verify_checksums(CR1_RESULT)
    manifest = json.loads((CR1_RESULT / "manifest.json").read_text())
    if registration["registration_id"] != CR1_REGISTRATION_ID:
        raise ValueError("unexpected CR1 registration ID")
    if registration["frozen_model_sha256"] != EXPECTED_MODEL_SHA256:
        raise ValueError("unexpected CR1 frozen-model hash")
    if sha256_file(CR1_REGISTRATION / "frozen_full_predictor.npz") != (
        EXPECTED_MODEL_SHA256
    ):
        raise ValueError("CR1 frozen-model bytes changed")
    if not (
        manifest["full_four_cell_gate"]
        and manifest["exact_replay"]
        and manifest["complete_readback_exact"]
    ):
        raise ValueError("CR1 did not authorize zero-shot transfer")
    return {"registration": registration, "manifest": manifest}


def _fixture_case(candidate: str, matrix_id: int, landmark: int) -> StateCase:
    composition = np.zeros(100, dtype=np.int64)
    composition[:4] = (2, 1, 1, 1)
    snapshot = Snapshot(
        composition=composition,
        generation=landmark + 1,
        inheritance=(True, False),
        boundary_h=(0.95, 0.80),
        previous_growth_steps=7,
        cumulative_growth_steps=43,
    )
    return StateCase(
        state_id=f"cr6-fixture-c{candidate}-m{matrix_id}-g{landmark}",
        cohort="ARTIFICIAL_FIXTURE",
        candidate=candidate,
        matrix_id=matrix_id,
        landmark=landmark,
        beta=np.eye(100, dtype=np.float64),
        snapshot=snapshot,
    )


def _fixture_inference(
    *, targeted_effect: bool
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    cases = [
        _fixture_case(candidate, matrix_id, landmark)
        for matrix_id in range(MATRICES)
        for candidate in CANDIDATES
        for landmark in LANDMARKS
    ]
    targets = np.zeros((len(cases), len(ARMS), BRANCHES), dtype=np.int8)
    if targeted_effect:
        targets[:, ARMS.index("MODEL_UP")] = 1
    predictions = np.full((len(cases), len(ARMS)), 0.5, dtype=np.float64)
    draws = generate_inference_draws(
        MATRICES,
        128,
        128,
        np.random.default_rng(11),
        np.random.default_rng(13),
    )
    return compute_regime_inference(
        "POS_A_M4_S5" if targeted_effect else NULL_REGIME,
        cases,
        targets,
        predictions,
        draws,
    )


def validation_checks() -> dict[str, Any]:
    predecessor = _verify_predecessor()
    inherited = base.validation_checks()
    positive, positive_rows = _fixture_inference(targeted_effect=True)
    null, null_rows = _fixture_inference(targeted_effect=False)

    from .intervention_cr2_dose_response import SEEDS as CR2_SEEDS
    from .intervention_cr3_confirmation import SEEDS as CR3_SEEDS
    from .intervention_cr4_confirmation import SEEDS as CR4_SEEDS
    from .intervention_cr5 import SEEDS as CR5_SEEDS
    from .intervention_cr5r import SEEDS as CR5R_SEEDS

    earlier_seed_values = set(base.SEED_DOMAINS.values())
    for registry in (
        cr1.SEEDS,
        CR2_SEEDS,
        CR3_SEEDS,
        CR4_SEEDS,
        CR5_SEEDS,
        CR5R_SEEDS,
    ):
        earlier_seed_values.update(registry.values())

    baseline_gard = asdict(GardConfig())
    config_changes_are_exact = True
    for regime, (a, sigma, _role) in REGIMES.items():
        current = asdict(regime_gard(regime))
        differences = {key for key in current if current[key] != baseline_gard[key]}
        expected = {
            key
            for key, value in (
                ("beta_log_mean", a),
                ("beta_log_sd", sigma),
            )
            if value != baseline_gard[key]
        }
        config_changes_are_exact &= differences == expected
        config_changes_are_exact &= current["beta_log_mean"] == a
        config_changes_are_exact &= current["beta_log_sd"] == sigma

    model = FrozenFullPredictor.load(CR1_REGISTRATION / "frozen_full_predictor.npz")
    model_round_trip = FrozenFullPredictor.load(
        CR1_REGISTRATION / "frozen_full_predictor.npz"
    )
    model_arrays_exact = model.arrays.keys() == model_round_trip.arrays.keys() and all(
        np.array_equal(model.arrays[key], model_round_trip.arrays[key])
        for key in model.arrays
    )

    checks = {
        "inherited_mandatory_validation_suite_passes": bool(
            inherited["all_checks_passed"]
        ),
        "cr1_predecessor_registration_exact": predecessor["registration"][
            "registration_id"
        ]
        == CR1_REGISTRATION_ID,
        "cr1_full_gate_replay_and_readback_passed": all(
            predecessor["manifest"][key]
            for key in (
                "full_four_cell_gate",
                "exact_replay",
                "complete_readback_exact",
            )
        ),
        "frozen_model_hash_exact": sha256_file(
            CR1_REGISTRATION / "frozen_full_predictor.npz"
        )
        == EXPECTED_MODEL_SHA256,
        "frozen_model_round_trip_arrays_exact": model_arrays_exact,
        "four_regimes_and_roles_exact": tuple(REGIMES)
        == (
            "POS_A_M4_S5",
            "POS_A_M3_S4",
            "POS_A_M5_S4",
            "NULL_A_M4_S3",
        )
        and POSITIVE_REGIMES == ("POS_A_M4_S5", "POS_A_M3_S4", "POS_A_M5_S4")
        and NULL_REGIME == "NULL_A_M4_S3",
        "only_beta_distribution_parameters_change": config_changes_are_exact,
        "design_size_exact": MATRICES == 40
        and LANDMARKS == (35, 65)
        and BRANCHES == 48
        and HORIZON == 12
        and ARMS == ("MODEL_UP", "MODEL_DOWN", "RANDOM", "NOOP"),
        "all_phase_specs_reuse_frozen_model_guided_algorithm": all(
            phase_spec(regime).phase == "p1"
            and phase_spec(regime).arms == ARMS
            and phase_spec(regime).contrast == ("MODEL_UP", "MODEL_DOWN")
            for regime in REGIMES
        ),
        "seed_domains_unique": len(SEEDS) == len(set(SEEDS.values())),
        "seed_domains_disjoint_from_all_predecessors": set(SEEDS.values()).isdisjoint(
            earlier_seed_values
        ),
        "positive_fixture_passes_all_four_cells": positive[
            "positive_transfer_gate_pass"
        ]
        is True
        and len(positive_rows) == (4 + 2) * MATRICES,
        "null_fixture_passes_both_candidate_tost_gates": null[
            "predicted_null_gate_pass"
        ]
        is True
        and len(null_rows) == (4 + 2) * MATRICES,
        "ci_crossing_zero_is_not_equivalence": not _tost_equivalent(
            (-0.05, 0.01), NULL_EQUIVALENCE_MARGIN
        ),
        "null_equivalence_margin_exact": NULL_EQUIVALENCE_MARGIN == 0.04,
        "positive_random_control_margin_exact": RANDOM_EQUIVALENCE_MARGIN == 0.025,
        "whole_matrix_draw_counts_exact": BOOTSTRAP_REPETITIONS == 4_096
        and RANDOMIZATION_REPETITIONS == 4_096,
        "strict_eight_excluded": protocol()["endpoint"]["strict_eight_excluded"],
        "cpu_budget_bounded": MINIMUM_CPU_BUDGET_HOURS == 3.0
        and DEFAULT_CPU_BUDGET_HOURS == 5.0
        and MAXIMUM_CPU_BUDGET_HOURS == 6.0,
        "mandatory_stop_and_no_automatic_cr7": protocol()["operational"][
            "mandatory_stop_after_seal"
        ]
        and protocol()["operational"]["cr7_not_launched_automatically"],
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
            "CR6 full repository validation failed\n"
            + completed.stdout
            + "\n"
            + completed.stderr
        )
    with _atomic_destination(output) as destination:
        payload = dict(checks)
        payload["source_hashes"] = source_hashes()
        payload["cr1_registration_checksum_manifest_sha256"] = sha256_file(
            CR1_REGISTRATION / "SHA256SUMS"
        )
        payload["cr1_result_checksum_manifest_sha256"] = sha256_file(
            CR1_RESULT / "SHA256SUMS"
        )
        (destination / "validation.json").write_text(
            json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n"
        )
        (destination / "pytest_output.txt").write_text(
            "$ " + " ".join(command) + "\n" + completed.stdout + completed.stderr
        )
        write_checksums(destination)
    verify_checksums(output)
    print(f"CR6 validation sealed: {output}", flush=True)


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
        raise ValueError("CR6 validation did not pass")
    if validation["source_hashes"] != source_hashes():
        raise ValueError("CR6 source changed after validation")
    _verify_predecessor()
    for forbidden in (DEFAULT_SMOKE, DEFAULT_OUTPUT, DEFAULT_WORK):
        if forbidden.exists():
            raise FileExistsError(
                f"CR6 scientific artifact exists before registration: {forbidden}"
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
        "cr1_registration_id": CR1_REGISTRATION_ID,
        "cr1_result_checksum_manifest_sha256": sha256_file(CR1_RESULT / "SHA256SUMS"),
        "frozen_model_sha256": EXPECTED_MODEL_SHA256,
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
            CR1_REGISTRATION / "frozen_full_predictor.npz",
            destination / "frozen_full_predictor.npz",
        )
        write_checksums(destination)
    verify_registration(output)
    _append_ledger(
        f"<!-- cr6-registered-{payload['registration_id']} -->",
        [
            "## CR6 zero-shot parameter-regime transfer registered",
            "",
            f"- Registration: `{payload['registration_id']}`.",
            "- The prospectively confirmed CR1 predictor was copied byte-for-byte and will not be refit or recalibrated.",
            "- Three positive-transfer regimes and one predicted-null regime were frozen before any CR6 scientific matrix.",
            "- CR6 is bounded to 40 fresh matrices per regime and a declared 3--6 CPU-hour budget.",
            "- No CR6 scientific matrix or future existed at the seal.",
            "",
        ],
    )
    print(f"CR6 registered: {payload['registration_id']}", flush=True)


def verify_registration(
    directory: Path = DEFAULT_REGISTRATION,
) -> dict[str, Any]:
    directory = directory.resolve()
    verify_checksums(directory)
    payload = json.loads((directory / "registration.json").read_text())
    frozen = json.loads((directory / "protocol.json").read_text())
    if payload.get("format") != REGISTRATION_FORMAT:
        raise ValueError("unsupported CR6 registration")
    if frozen != _json_ready(protocol()):
        raise ValueError("CR6 frozen protocol changed")
    if payload["source_hashes"] != source_hashes():
        raise ValueError("CR6 source changed after registration")
    if payload["seed_registry"] != SEEDS:
        raise ValueError("CR6 seed registry changed")
    if payload["registration_id"] != _canonical_digest(
        {key: value for key, value in payload.items() if key != "registration_id"}
    ):
        raise ValueError("CR6 registration ID changed")
    if sha256_file(directory / "frozen_full_predictor.npz") != (EXPECTED_MODEL_SHA256):
        raise ValueError("CR6 frozen model changed")
    FrozenFullPredictor.load(directory / "frozen_full_predictor.npz")
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
    smoke_spec = base.PhaseSpec(
        phase="p1",
        role="non-scientific CR6 artificial smoke",
        matrices=1,
        branches=2,
        cohort_seed=SEEDS["smoke_cohort"],
        selection_seed=SEEDS["smoke_selection"],
        future_seed=SEEDS["smoke_future"],
        bootstrap_seed=SEEDS["validation"],
        randomization_seed=SEEDS["replay"],
    )
    cohort = CohortConfig(1, 2, (5,))
    smoke_experiment = ExperimentConfig(
        gard=GardConfig(),
        development=cohort,
        confirmation=cohort,
        horizon=HORIZON,
        bootstrap_repetitions=8,
        permutation_repetitions=8,
        regenerate_confirmation=True,
        master_seed=smoke_spec.cohort_seed,
    )
    with tempfile.TemporaryDirectory(
        prefix="codex-cr6-smoke-", dir=output.parent
    ) as temporary:
        temporary_path = Path(temporary)
        with threadpool_limits(limits=1):
            cases = build_cohort(smoke_experiment, "INTCR6_ARTIFICIAL_SMOKE", cohort)
        generated = base.run_phase_batches(
            cases,
            smoke_experiment,
            smoke_spec,
            registration_directory / "frozen_full_predictor.npz",
            registration["registration_id"],
            temporary_path / "generate",
            1,
            "generate",
        )
        replayed = base.run_phase_batches(
            cases,
            smoke_experiment,
            smoke_spec,
            registration_directory / "frozen_full_predictor.npz",
            registration["registration_id"],
            temporary_path / "replay",
            1,
            "replay",
        )
        replay = base.replay_audit(generated, replayed)
        checks = {
            "complete_artificial_worker_contract": len(generated) == 2
            and all(batch.arm_names == ARMS for batch in generated)
            and all(
                all(len(outcomes) == 2 for outcomes in batch.outcomes)
                for batch in generated
            ),
            "exact_replay": replay["state_edit_endpoint_and_process_digests_exact"],
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
                    "format": "codex-intervention-cr6-smoke-v1",
                    "registration_id": registration["registration_id"],
                    "artificial_fixture_only": True,
                    "scientific_matrices": 0,
                    "scientific_futures": 0,
                    "checks": checks,
                    "effect_sizes_disclosed": False,
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
    print(f"CR6 non-scientific smoke passed: {output}", flush=True)


def _status(
    work: Path,
    state: str,
    detail: str,
    *,
    regime: str | None = None,
    completed_regimes: int | None = None,
    available_cpu_hours: float | None = None,
) -> None:
    value: dict[str, Any] = {
        "format": STATUS_FORMAT,
        "phase": "cr6_zero_shot_parameter_transfer",
        "state": state,
        "detail": detail,
        "current_regime": regime,
        "regimes_total": len(REGIMES),
        "mandatory_stop_after_seal": True,
        "cr7_launched": False,
    }
    if completed_regimes is not None:
        value["regimes_complete"] = completed_regimes
    if available_cpu_hours is not None:
        value["available_cpu_hours_at_launch"] = available_cpu_hours
    work.mkdir(parents=True, exist_ok=True)
    base._atomic_json(work / "campaign_status.json", value)


def _prepare_work(
    work: Path,
    output: Path,
    registration_id: str,
    available_cpu_hours: float,
) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if not (
        MINIMUM_CPU_BUDGET_HOURS <= available_cpu_hours <= MAXIMUM_CPU_BUDGET_HOURS
    ):
        raise ValueError(
            "CR6 requires a declared CPU budget between "
            f"{MINIMUM_CPU_BUDGET_HOURS:.1f} and {MAXIMUM_CPU_BUDGET_HOURS:.1f} hours"
        )
    free_bytes = shutil.disk_usage(work.parent).free
    if free_bytes < MINIMUM_FREE_DISK_BYTES:
        raise OSError(
            f"CR6 requires {MINIMUM_FREE_DISK_BYTES:,} free bytes; found {free_bytes:,}"
        )
    work.mkdir(parents=True, exist_ok=True)
    path = work / "campaign_contract.json"
    existing = json.loads(path.read_text()) if path.exists() else None
    launch_free_bytes = (
        int(existing["free_disk_bytes_at_launch"])
        if existing is not None
        else free_bytes
    )
    contract: dict[str, Any] = {
        "format": "codex-intervention-cr6-campaign-v1",
        "registration_id": registration_id,
        "output": str(output),
        "regimes": list(REGIMES),
        "matrices_per_regime": MATRICES,
        "landmarks": list(LANDMARKS),
        "branches": BRANCHES,
        "arms": list(ARMS),
        "primary_futures": len(REGIMES)
        * 2
        * MATRICES
        * len(LANDMARKS)
        * len(ARMS)
        * BRANCHES,
        "replay_futures": len(REGIMES)
        * 2
        * MATRICES
        * len(LANDMARKS)
        * len(ARMS)
        * BRANCHES,
        "available_cpu_hours_at_launch": available_cpu_hours,
        "free_disk_bytes_at_launch": launch_free_bytes,
        "source_hashes": source_hashes(),
        "checkpoint_resumable": True,
    }
    contract["campaign_id"] = _canonical_digest(_json_ready(contract))
    if path.exists():
        if json.loads(path.read_text()) != _json_ready(contract):
            raise ValueError("CR6 work directory belongs to another campaign")
    else:
        base._atomic_json(path, contract)
    _status(
        work,
        "running",
        "campaign_initialized",
        completed_regimes=0,
        available_cpu_hours=available_cpu_hours,
    )


def _inference_draws(regime: str) -> dict[str, NDArray]:
    label = cohort_label(regime)
    return generate_inference_draws(
        MATRICES,
        BOOTSTRAP_REPETITIONS,
        RANDOMIZATION_REPETITIONS,
        np.random.default_rng(
            derive_seed(SEEDS[f"{regime}__bootstrap"], f"{label}.bootstrap")
        ),
        np.random.default_rng(
            derive_seed(SEEDS[f"{regime}__randomization"], f"{label}.randomization")
        ),
    )


def _regime_artifact_is_complete(
    artifact: Path, regime: str, registration_id: str
) -> bool:
    if not artifact.is_dir():
        return False
    verify_checksums(artifact)
    manifest = json.loads((artifact / "manifest.json").read_text())
    if (
        manifest.get("format") != "codex-intervention-cr6-regime-result-v1"
        or manifest.get("regime") != regime
        or manifest.get("registration_id") != registration_id
        or not manifest.get("exact_replay")
        or not manifest.get("complete_readback_exact")
    ):
        raise ValueError(f"invalid completed CR6 staging artifact: {artifact}")
    return True


def _write_regime_artifact(
    artifact: Path,
    regime: str,
    cases: list[StateCase],
    generated: list[base.PhaseBatch],
    replay: dict[str, Any],
    registration: dict[str, Any],
) -> dict[str, Any]:
    spec = phase_spec(regime)
    arrays = base._outcome_arrays(cases, generated, spec)
    draws = _inference_draws(regime)
    metrics, matrix_rows = compute_regime_inference(
        regime, cases, arrays["targets"], arrays["predictions"], draws
    )
    secondary = base._secondary_descriptives(cases, arrays, spec)
    for row in matrix_rows:
        row["regime"] = regime
    with _atomic_destination(artifact) as destination:
        np.savez_compressed(destination / "branch_arrays.npz", **arrays)
        base._write_branch_table(destination / "branches.csv.gz", cases, generated)
        base._write_state_artifacts(destination, cases, generated, arrays)
        base._write_selection_artifacts(destination, cases, generated, spec)
        _write_inference_arrays(destination / "inference_arrays.npz", draws, metrics)
        pd.DataFrame(matrix_rows).to_csv(
            destination / "matrix_effects.csv", index=False
        )
        readback = _readback_regime(destination, regime, cases, metrics, matrix_rows)
        integrity = {
            "exact_replay": bool(
                replay["state_edit_endpoint_and_process_digests_exact"]
            ),
            "artifact_readback_exact": bool(
                readback["primary_metrics_exact"] and readback["matrix_effects_exact"]
            ),
        }
        metrics["integrity_gates"] = integrity
        metrics["registered_regime_gate_with_integrity"] = bool(
            metrics["registered_regime_gate_pass"] and all(integrity.values())
        )
        (destination / "primary_metrics.json").write_text(
            json.dumps(_json_ready(metrics), indent=2, sort_keys=True) + "\n"
        )
        (destination / "secondary_outcomes.json").write_text(
            json.dumps(_json_ready(secondary), indent=2, sort_keys=True) + "\n"
        )
        (destination / "replay_audit.json").write_text(
            json.dumps(_json_ready(replay), indent=2, sort_keys=True) + "\n"
        )
        (destination / "readback_audit.json").write_text(
            json.dumps(readback, indent=2, sort_keys=True) + "\n"
        )
        (destination / "manifest.json").write_text(
            json.dumps(
                {
                    "format": "codex-intervention-cr6-regime-result-v1",
                    "registration_id": registration["registration_id"],
                    "regime": regime,
                    "A": REGIMES[regime][0],
                    "sigma": REGIMES[regime][1],
                    "registered_role": REGIMES[regime][2],
                    "matrices": MATRICES,
                    "states": len(cases),
                    "branches_per_arm_state": BRANCHES,
                    "primary_futures": len(cases) * len(ARMS) * BRANCHES,
                    "replay_futures": len(cases) * len(ARMS) * BRANCHES,
                    "registered_regime_gate": metrics[
                        "registered_regime_gate_with_integrity"
                    ],
                    "exact_replay": integrity["exact_replay"],
                    "complete_readback_exact": integrity["artifact_readback_exact"],
                    "no_refit_recalibration_or_regime_switching": True,
                    "no_future_retry_or_matrix_replacement": True,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        write_checksums(destination)
    verify_checksums(artifact)
    return metrics


def _reports(summary: dict[str, Any]) -> tuple[str, str]:
    lines = [
        "# CR6 zero-shot parameter-regime transfer",
        "",
        f"Complete registered CR6 gate: **{summary['complete_cr6_gate']}**.",
        f"All three positive-transfer regimes passed: **{summary['all_positive_regimes_pass']}**.",
        f"Predicted weak-control null passed: **{summary['predicted_null_pass']}**.",
        "",
        "| Regime | Role | Candidate/half | Up-down | 95% CI | Holm p | Random TOST | Cell pass |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for regime in REGIMES:
        metrics = summary["regimes"][regime]
        for cell in metrics["cells"]:
            targeted = cell["contrasts"]["up_minus_down"]
            lines.append(
                f"| {regime} | {metrics['registered_role']} | {cell['cell']} | "
                f"{targeted['estimate']:+.6f} | {targeted['bootstrap_ci95']} | "
                f"{cell['up_down_randomization_p_holm']:.6g} | "
                f"{cell['random_noop_equivalence']['tost_equivalent']} | "
                f"{cell['cr6_positive_transfer_cell_pass']} |"
            )
    lines.extend(
        [
            "",
            "The candidate-pooled predicted-null decision uses the full 90% whole-matrix interval, not merely whether an interval crosses zero.",
            "No regime, candidate, or branch half was pooled to rescue another. The home-regime predictor and exhaustive edit selection were reused without refitting or recalibration.",
            "",
            "This is a causal result inside the two Codex simulator contracts. It does not establish universal transfer, life, agency, biological memory, autonomous organization, or real prebiotic chemistry.",
            "",
        ]
    )
    lay_lines = [
        "# CR6 in plain language",
        "",
        "We took the already frozen molecular-edit predictor out of the environment where it was developed and gave it four new kinds of catalytic networks. It was not allowed to learn from these new simulations.",
        "",
        (
            "The predictor passed all three predeclared transfer tests."
            if summary["all_positive_regimes_pass"]
            else "The predictor did not pass every predeclared positive-transfer test; the successful and failed environments remain reported separately."
        ),
        (
            "It also correctly identified the weak-heterogeneity environment as having an effect small enough to fit entirely inside the predeclared ±0.04 practical-null range."
            if summary["predicted_null_pass"]
            else "The weak-heterogeneity environment did not meet the strict practical-null test; crossing zero alone was not counted as success."
        ),
        "",
        "This asks whether the control signal generalizes across versions of the simulated chemistry. It does not mean the rule is universal or that a real chemical system has been controlled.",
        "",
    ]
    return "\n".join(lines), "\n".join(lay_lines)


def _assemble_result(
    work: Path,
    output: Path,
    registration: dict[str, Any],
    available_cpu_hours: float,
) -> dict[str, Any]:
    regime_metrics: dict[str, Any] = {}
    regime_manifests: dict[str, Any] = {}
    for regime in REGIMES:
        artifact = work / "artifacts" / regime
        if not _regime_artifact_is_complete(
            artifact, regime, registration["registration_id"]
        ):
            raise ValueError(f"CR6 regime artifact is incomplete: {regime}")
        regime_metrics[regime] = json.loads(
            (artifact / "primary_metrics.json").read_text()
        )
        regime_manifests[regime] = json.loads((artifact / "manifest.json").read_text())

    positive_pass = bool(
        all(
            regime_metrics[regime]["registered_regime_gate_with_integrity"]
            for regime in POSITIVE_REGIMES
        )
    )
    null_pass = bool(
        regime_metrics[NULL_REGIME]["registered_regime_gate_with_integrity"]
    )
    all_integrity = bool(
        all(
            manifest["exact_replay"] and manifest["complete_readback_exact"]
            for manifest in regime_manifests.values()
        )
    )
    summary = {
        "format": "codex-intervention-cr6-summary-v1",
        "registration_id": registration["registration_id"],
        "regimes": regime_metrics,
        "all_positive_regimes_pass": positive_pass,
        "predicted_null_pass": null_pass,
        "all_regime_replays_and_readbacks_pass": all_integrity,
        "complete_cr6_gate": bool(positive_pass and null_pass and all_integrity),
        "regimes_never_pooled": True,
        "candidates_never_pooled": True,
        "mandatory_stop_observed": True,
        "cr7_launched": False,
    }
    technical, lay = _reports(summary)
    failed = [
        regime
        for regime in REGIMES
        if not regime_metrics[regime]["registered_regime_gate_with_integrity"]
    ]
    claims = {
        "supported": [
            *(
                [
                    "zero-shot transfer of frozen molecular control across the three registered positive beta regimes"
                ]
                if positive_pass
                else []
            ),
            *(
                [
                    "the registered weak-heterogeneity regime is practically equivalent to zero within +/-0.04 in both candidates"
                ]
                if null_pass
                else []
            ),
        ],
        "failed_predictions": [
            f"CR6 registered regime gate: {name}" for name in failed
        ],
        "unresolved": [
            "transfer beyond the four registered beta distributions",
            "long-horizon closed-loop control",
            "autonomous persistence after release",
        ],
        "prohibited": protocol()["claim_boundary"]["prohibited"],
    }
    with _atomic_destination(output) as destination:
        for regime in REGIMES:
            shutil.copytree(
                work / "artifacts" / regime,
                destination / regime,
                copy_function=os.link,
            )
        (destination / "primary_metrics.json").write_text(
            json.dumps(_json_ready(summary), indent=2, sort_keys=True) + "\n"
        )
        (destination / "SCIENTIFIC_REPORT.md").write_text(technical)
        (destination / "LAY_SUMMARY.md").write_text(lay)
        (destination / "claim_boundaries.json").write_text(
            json.dumps(claims, indent=2, sort_keys=True) + "\n"
        )
        rows = []
        for regime, metrics in regime_metrics.items():
            for cell in metrics["cells"]:
                rows.append(
                    {
                        "regime": regime,
                        "role": metrics["registered_role"],
                        "cell": cell["cell"],
                        "up_minus_down": cell["contrasts"]["up_minus_down"]["estimate"],
                        "ci95_lower": cell["contrasts"]["up_minus_down"][
                            "bootstrap_ci95"
                        ][0],
                        "ci95_upper": cell["contrasts"]["up_minus_down"][
                            "bootstrap_ci95"
                        ][1],
                        "holm_p": cell["up_down_randomization_p_holm"],
                        "random_tost": cell["random_noop_equivalence"][
                            "tost_equivalent"
                        ],
                        "positive_cell_pass": cell["cr6_positive_transfer_cell_pass"],
                    }
                )
        pd.DataFrame(rows).to_csv(destination / "regime_summary.csv", index=False)
        manifest = {
            "format": RESULT_FORMAT,
            "registration_id": registration["registration_id"],
            "regimes": list(REGIMES),
            "matrices_per_regime": MATRICES,
            "total_fresh_matrices": MATRICES * len(REGIMES),
            "total_states": 2 * MATRICES * len(LANDMARKS) * len(REGIMES),
            "branches_per_arm_state": BRANCHES,
            "primary_futures": 122_880,
            "replay_futures": 122_880,
            "all_positive_regimes_pass": positive_pass,
            "predicted_null_pass": null_pass,
            "complete_cr6_gate": summary["complete_cr6_gate"],
            "exact_replay_all_regimes": all_integrity,
            "complete_readback_all_regimes": all_integrity,
            "available_cpu_hours_at_launch": available_cpu_hours,
            "no_refit_recalibration_or_regime_switching": True,
            "no_future_retry_or_matrix_replacement": True,
            "mandatory_stop_after_this_stage": True,
            "cr7_launched": False,
            "runtime": _runtime_manifest(),
        }
        (destination / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        write_checksums(destination)
    verify_checksums(output)
    return summary


def run(
    registration_directory: Path = DEFAULT_REGISTRATION,
    output: Path = DEFAULT_OUTPUT,
    work: Path = DEFAULT_WORK,
    workers: int = min(os.cpu_count() or 1, 14),
    available_cpu_hours: float = DEFAULT_CPU_BUDGET_HOURS,
) -> None:
    if workers < 1:
        raise ValueError("workers must be positive")
    registration_directory = registration_directory.resolve()
    output = output.resolve()
    work = work.resolve()
    registration = verify_registration(registration_directory)
    verify_checksums(DEFAULT_SMOKE)
    _verify_predecessor()
    _prepare_work(
        work,
        output,
        registration["registration_id"],
        available_cpu_hours,
    )
    model_path = registration_directory / "frozen_full_predictor.npz"
    artifacts = work / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    completed = 0
    for position, regime in enumerate(REGIMES, start=1):
        artifact = artifacts / regime
        if _regime_artifact_is_complete(
            artifact, regime, registration["registration_id"]
        ):
            completed += 1
            print(
                f"[cr6 {position}/4] Reusing sealed staging artifact for {regime}",
                flush=True,
            )
            continue
        _status(
            work,
            "running",
            "building_fresh_natural_states",
            regime=regime,
            completed_regimes=completed,
            available_cpu_hours=available_cpu_hours,
        )
        current_experiment = experiment(regime)
        current_spec = phase_spec(regime)
        print(
            f"[cr6 {position}/4] {regime}: building {MATRICES} fresh matrices and 160 states",
            flush=True,
        )
        with threadpool_limits(limits=1):
            cases = build_cohort(
                current_experiment,
                cohort_label(regime),
                current_experiment.confirmation,
            )
        if len(cases) != 2 * MATRICES * len(LANDMARKS):
            raise AssertionError(f"CR6 {regime} cohort is incomplete")
        futures = len(cases) * len(ARMS) * BRANCHES
        _status(
            work,
            "running",
            "selection_and_primary_futures",
            regime=regime,
            completed_regimes=completed,
            available_cpu_hours=available_cpu_hours,
        )
        print(
            f"[cr6 {position}/4] {regime}: exhaustive edit selection and {futures:,} primary F12 futures",
            flush=True,
        )
        generated = base.run_phase_batches(
            cases,
            current_experiment,
            current_spec,
            model_path,
            registration["registration_id"],
            work / regime / "generate",
            workers,
            "generate",
        )
        _status(
            work,
            "running",
            "complete_exact_replay",
            regime=regime,
            completed_regimes=completed,
            available_cpu_hours=available_cpu_hours,
        )
        print(
            f"[cr6 {position}/4] {regime}: replaying all {futures:,} futures",
            flush=True,
        )
        replayed = base.run_phase_batches(
            cases,
            current_experiment,
            current_spec,
            model_path,
            registration["registration_id"],
            work / regime / "replay",
            workers,
            "replay",
        )
        replay = base.replay_audit(generated, replayed)
        if not replay["state_edit_endpoint_and_process_digests_exact"]:
            raise AssertionError(f"CR6 {regime} exact replay failed")
        del replayed
        _status(
            work,
            "running",
            "whole_matrix_inference_and_artifact_readback",
            regime=regime,
            completed_regimes=completed,
            available_cpu_hours=available_cpu_hours,
        )
        _write_regime_artifact(artifact, regime, cases, generated, replay, registration)
        completed += 1
        _status(
            work,
            "running",
            "regime_staging_artifact_sealed",
            regime=regime,
            completed_regimes=completed,
            available_cpu_hours=available_cpu_hours,
        )
        del generated, cases

    print("[cr6] Assembling final four-regime result", flush=True)
    _status(
        work,
        "running",
        "assembling_final_result",
        completed_regimes=completed,
        available_cpu_hours=available_cpu_hours,
    )
    summary = _assemble_result(work, output, registration, available_cpu_hours)
    _append_ledger(
        f"<!-- sealed-cr6-{registration['registration_id']} -->",
        [
            "## CR6 zero-shot parameter-regime transfer sealed",
            "",
            f"- Registration: `{registration['registration_id']}`.",
            f"- Result: `{output.relative_to(ROOT)}`.",
            f"- All three positive-transfer regimes passed: **{summary['all_positive_regimes_pass']}**.",
            f"- Predicted weak-control null passed: **{summary['predicted_null_pass']}**.",
            f"- Complete CR6 gate: **{summary['complete_cr6_gate']}**.",
            "- All primary futures, replays, written-artifact readbacks, and checksums passed.",
            "- Mandatory review stop observed; CR7 was not launched automatically.",
            "",
        ],
    )
    _status(
        work,
        "sealed_complete",
        "mandatory_review_stop",
        completed_regimes=completed,
        available_cpu_hours=available_cpu_hours,
    )
    print("[cr6] Result sealed; STOPPED; CR7 not launched", flush=True)


def read_status(work: Path = DEFAULT_WORK) -> dict[str, Any]:
    work = work.resolve()
    if not work.is_dir():
        raise FileNotFoundError(f"work directory does not exist: {work}")
    campaign = work / "campaign_status.json"
    value: dict[str, Any] = {
        "format": STATUS_FORMAT,
        "work_directory": str(work),
        "campaign": (json.loads(campaign.read_text()) if campaign.is_file() else None),
        "regimes": {},
    }
    registration_id = None
    contract = work / "campaign_contract.json"
    if contract.is_file():
        registration_id = json.loads(contract.read_text())["registration_id"]
    for regime in REGIMES:
        item: dict[str, Any] = {"generate": None, "replay": None, "artifact": False}
        for stage in ("generate", "replay"):
            status_path = work / regime / stage / "status.json"
            if status_path.is_file():
                item[stage] = json.loads(status_path.read_text())
        artifact = work / "artifacts" / regime
        if artifact.is_dir() and registration_id is not None:
            item["artifact"] = _regime_artifact_is_complete(
                artifact, regime, registration_id
            )
        value["regimes"][regime] = item
    if value["campaign"] is None and not any(
        item["generate"] or item["replay"] for item in value["regimes"].values()
    ):
        raise ValueError("CR6 work directory has no readable status")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate").add_argument(
        "--output", type=Path, default=DEFAULT_VALIDATION
    )
    register_parser = commands.add_parser("register")
    register_parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    register_parser.add_argument("--output", type=Path, default=DEFAULT_REGISTRATION)
    commands.add_parser("verify").add_argument(
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
        "--available-cpu-hours", type=float, default=DEFAULT_CPU_BUDGET_HOURS
    )
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
        print(
            json.dumps(verify_registration(args.registration), indent=2, sort_keys=True)
        )
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
