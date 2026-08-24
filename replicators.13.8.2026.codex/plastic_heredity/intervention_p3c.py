"""Prospective P3c catalytic-throughput clarification program.

The module is additive to the sealed P3b campaign.  Registration is blocked
until the external Fable response is archived, and confirmation is blocked
until the separately sealed development pilot passes its frozen advancement
rule.
"""

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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from threadpoolctl import threadpool_limits

from . import intervention_p3b_dose_bridge as p3b
from . import intervention_p3b_singleton_recovery as p3b_recovery
from . import intervention_replication as base
from .config import CANDIDATES, CohortConfig, ExperimentConfig
from .experiment import StateCase, _json_ready, _runtime_manifest, build_cohort
from .intervention_core import (
    BetaSurgery,
    FrozenFullPredictor,
    InterventionOutcome,
    outcome_from_records,
    simulate_one_shot,
)
from .intervention_metrics import (
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
from .simulator import (
    SimulationError,
    Snapshot,
    advance_fission,
    cosine_similarity,
    simulate_future_absorbing,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = REPOSITORY_ROOT / "results_intervention_replication"
ORIGINAL_REGISTRATION = RESULT_ROOT / "registration"
P3B_RESULT = RESULT_ROOT / "p3b_beta_surgery_dose_bridge"
P3B_REGISTRATION = RESULT_ROOT / "p3b_dose_bridge_registration"
P3B_RECOVERY = RESULT_ROOT / "p3b_singleton_recovery_amendment"
GEOMETRY_AUDIT = RESULT_ROOT / "p3c_geometry_audit"
DEFAULT_FABLE_ARCHIVE = RESULT_ROOT / "p3c_fable_response"
DEFAULT_VALIDATION = RESULT_ROOT / "p3c_validation"
DEFAULT_REGISTRATION = RESULT_ROOT / "p3c_registration"
DEFAULT_SMOKE = RESULT_ROOT / "p3c_smoke"
DEFAULT_PILOT_OUTPUT = RESULT_ROOT / "p3c_throughput_pilot"
DEFAULT_PILOT_WORK = RESULT_ROOT / ".p3c_throughput_pilot_work"
DEFAULT_CONFIRMATION_OUTPUT = RESULT_ROOT / "p3c_throughput_confirmation"
DEFAULT_CONFIRMATION_WORK = RESULT_ROOT / ".p3c_throughput_confirmation_work"
DEFAULT_RESILIENCE_OUTPUT = RESULT_ROOT / "p3c_shared_break_resilience"
DEFAULT_RESILIENCE_WORK = RESULT_ROOT / ".p3c_shared_break_resilience_work"

DOCUMENT = "CODEX_INTERVENTION_P3C_PREREGISTRATION.md"
FABLE_REQUEST = "FABLE_P3C_GEOMETRY_REQUEST.md"
SOURCE_FILES = (
    DOCUMENT,
    FABLE_REQUEST,
    "plastic_heredity/intervention_p3c.py",
    "tests/test_intervention_p3c.py",
)
PROGRAM_FORMAT = "codex-intervention-p3c-throughput-v1"
VALIDATION_FORMAT = "codex-intervention-p3c-validation-v1"
REGISTRATION_FORMAT = "codex-intervention-p3c-registration-v1"
RESULT_FORMAT = "codex-intervention-p3c-result-v1"
CHECKPOINT_FORMAT = "codex-intervention-p3c-checkpoint-v1"

ARMS = (
    "LOOSEN",
    "TIGHTEN",
    "BALANCED_LOG_RANDOM",
    "THROUGHPUT_NEUTRAL_RANDOM",
    "NOOP",
)
LANDMARKS = (20, 35, 50, 60, 65, 80)
PRIMARY_LANDMARKS = (20, 35, 50, 65, 80)
COMPATIBILITY_LANDMARKS = (60,)
BRANCHES = 32
HORIZON = 12
PILOT_MATRICES = 40
CONFIRMATION_MATRICES = 160
BOOTSTRAP_REPETITIONS = 4_096
RANDOMIZATION_REPETITIONS = 4_096
EQUIVALENCE_MARGIN = 0.025
SURGERY_NORM_FRACTION = 0.5
NEUTRAL_MAX_ATTEMPTS = 4_096
NEUTRAL_ABSOLUTE_TOLERANCE = 1e-10
NEUTRAL_RELATIVE_TOLERANCE = 1e-12
LABELS = {
    "smoke": "INTP3C_NONSCIENTIFIC_SMOKE_V1",
    "pilot": "INTP3C_THROUGHPUT_PILOT_V1",
    "confirmation": "INTP3C_THROUGHPUT_CONFIRMATION_V1",
    "resilience": "INTP3C_SHARED_BREAK_RESILIENCE_V1",
}


def _seed_value(name: str) -> str:
    return hashlib.sha256(
        f"codex-clean-room-p3c-throughput-v1::{name}".encode("utf-8")
    ).hexdigest()


SEED_DOMAINS = {
    name: _seed_value(name)
    for name in (
        "validation",
        "smoke_cohort",
        "smoke_balanced_selection",
        "smoke_neutral_selection",
        "smoke_future",
        "pilot_cohort",
        "pilot_balanced_selection",
        "pilot_neutral_selection",
        "pilot_future",
        "pilot_bootstrap",
        "pilot_randomization",
        "confirmation_cohort",
        "confirmation_balanced_selection",
        "confirmation_neutral_selection",
        "confirmation_future",
        "confirmation_bootstrap",
        "confirmation_randomization",
        "resilience_acquisition",
        "resilience_balanced_selection",
        "resilience_neutral_selection",
        "resilience_future",
        "resilience_bootstrap",
        "resilience_randomization",
        "replay",
    )
}


@dataclass(frozen=True)
class P3CSpec:
    stage: str
    role: str
    matrices: int
    branches: int
    landmarks: tuple[int, ...]
    horizon: int
    cohort_seed: str
    balanced_selection_seed: str
    neutral_selection_seed: str
    future_seed: str
    bootstrap_seed: str
    randomization_seed: str
    arms: tuple[str, ...] = ARMS

    @property
    def phase(self) -> str:
        return f"p3c_{self.stage}"


def phase_spec(stage: str) -> P3CSpec:
    if stage == "pilot":
        matrices = PILOT_MATRICES
    elif stage == "confirmation":
        matrices = CONFIRMATION_MATRICES
    else:
        raise ValueError(f"unknown P3c stage {stage}")
    return P3CSpec(
        stage=stage,
        role=(
            "development pilot for catalytic-throughput clarification"
            if stage == "pilot"
            else "untouched confirmation of catalytic-throughput control"
        ),
        matrices=matrices,
        branches=BRANCHES,
        landmarks=LANDMARKS,
        horizon=HORIZON,
        cohort_seed=SEED_DOMAINS[f"{stage}_cohort"],
        balanced_selection_seed=SEED_DOMAINS[f"{stage}_balanced_selection"],
        neutral_selection_seed=SEED_DOMAINS[f"{stage}_neutral_selection"],
        future_seed=SEED_DOMAINS[f"{stage}_future"],
        bootstrap_seed=SEED_DOMAINS[f"{stage}_bootstrap"],
        randomization_seed=SEED_DOMAINS[f"{stage}_randomization"],
    )


def resilience_spec() -> P3CSpec:
    return P3CSpec(
        stage="resilience",
        role="causal renewal from identical naturally broken daughter states",
        matrices=CONFIRMATION_MATRICES,
        branches=BRANCHES,
        landmarks=PRIMARY_LANDMARKS,
        horizon=8,
        cohort_seed=SEED_DOMAINS["confirmation_cohort"],
        balanced_selection_seed=SEED_DOMAINS["resilience_balanced_selection"],
        neutral_selection_seed=SEED_DOMAINS["resilience_neutral_selection"],
        future_seed=SEED_DOMAINS["resilience_future"],
        bootstrap_seed=SEED_DOMAINS["resilience_bootstrap"],
        randomization_seed=SEED_DOMAINS["resilience_randomization"],
    )


def _source_hashes() -> dict[str, str]:
    return {name: sha256_file(REPOSITORY_ROOT / name) for name in SOURCE_FILES}


def _verify_geometry_audit() -> dict[str, Any]:
    verify_checksums(GEOMETRY_AUDIT)
    manifest = json.loads(
        (GEOMETRY_AUDIT / "manifest.json").read_text(encoding="utf-8")
    )
    for name, expected in manifest.get("source_hashes", {}).items():
        if sha256_file(REPOSITORY_ROOT / name) != expected:
            raise ValueError(f"P3c geometry-audit source changed: {name}")
    if (
        manifest.get("classification")
        != "posthoc_exploratory_existing_data_audit"
        or manifest.get("new_scientific_matrices") != 0
        or manifest.get("new_simulated_futures") != 0
        or manifest.get("p3b_result_modified")
    ):
        raise ValueError("invalid P3c geometry-audit manifest")
    return manifest


def _present_block(
    composition: NDArray, beta: NDArray
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    values = np.asarray(composition)
    matrix = np.asarray(beta, dtype=np.float64)
    if values.ndim != 1 or matrix.shape != (values.size, values.size):
        raise ValueError("beta and composition dimensions differ")
    if np.any(values < 0) or int(values.sum()) <= 0:
        raise ValueError("composition must be nonnegative and nonempty")
    if not np.isfinite(matrix).all() or np.any(matrix <= 0.0):
        raise ValueError("beta must be finite and strictly positive")
    present = np.flatnonzero(values > 0).astype(np.int64)
    rows, columns = np.meshgrid(present, present, indexing="ij")
    flat = np.ravel_multi_index((rows.ravel(), columns.ravel()), matrix.shape)
    return present, flat.astype(np.int64), matrix.ravel()[flat].copy()


def catalytic_throughput(composition: NDArray, beta: NDArray) -> float:
    values = np.asarray(composition, dtype=np.float64)
    if values.sum() <= 0.0:
        raise ValueError("throughput requires nonempty composition")
    x = values / values.sum()
    return float(x @ np.asarray(beta, dtype=np.float64) @ x)


def throughput_weights(composition: NDArray) -> NDArray[np.float64]:
    values = np.asarray(composition, dtype=np.float64)
    present = np.flatnonzero(values > 0)
    x = values / values.sum()
    return np.outer(x[present], x[present]).ravel()


def _surgery(
    name: str,
    beta: NDArray,
    flat: NDArray[np.int64],
    before: NDArray[np.float64],
    after: NDArray[np.float64],
    target_norm: float,
) -> BetaSurgery:
    matrix = np.asarray(beta, dtype=np.float64)
    altered = matrix.copy()
    altered.ravel()[flat] = after
    observed = float(np.linalg.norm(after - before))
    tolerance = 1e-11 * max(1.0, target_norm)
    if abs(observed - target_norm) > tolerance:
        raise AssertionError(f"{name} failed exact Frobenius audit")
    if np.any(altered <= 0.0) or not np.isfinite(altered).all():
        raise AssertionError(f"{name} violated beta positivity")
    return BetaSurgery(
        name=name,
        beta=altered,
        flat_indices=flat,
        before=before,
        after=np.asarray(after, dtype=np.float64),
        requested_norm=float(target_norm),
        observed_norm=observed,
    )


def multiplicative_surgery(
    composition: NDArray, beta: NDArray, factor: float, name: str
) -> BetaSurgery:
    if not np.isfinite(factor) or factor <= 0.0 or factor == 1.0:
        raise ValueError("factor must be positive and nonunit")
    _present, flat, before = _present_block(composition, beta)
    target = abs(factor - 1.0) * float(np.linalg.norm(before))
    return _surgery(name, beta, flat, before, before * factor, target)


def balanced_log_surgery(
    composition: NDArray,
    beta: NDArray,
    rng: np.random.Generator,
    name: str = "BALANCED_LOG_RANDOM",
) -> BetaSurgery:
    _present, flat, before = _present_block(composition, beta)
    direction = p3b.balanced_log_direction(before.size, rng)
    return p3b.audited_random_pp_surgery(
        composition,
        beta,
        SURGERY_NORM_FRACTION * float(np.linalg.norm(before)),
        direction,
        name,
    )


def throughput_neutral_pp_surgery(
    composition: NDArray,
    beta: NDArray,
    rng: np.random.Generator,
    *,
    name: str = "THROUGHPUT_NEUTRAL_RANDOM",
    max_attempts: int = NEUTRAL_MAX_ATTEMPTS,
) -> BetaSurgery:
    """Draw a positive exact-norm P x P perturbation orthogonal to throughput."""

    present, flat, before = _present_block(composition, beta)
    if present.size < 2:
        raise ValueError("throughput-neutral surgery requires two present types")
    weights = throughput_weights(composition)
    weight_norm_squared = float(weights @ weights)
    target_norm = SURGERY_NORM_FRACTION * float(np.linalg.norm(before))
    if weight_norm_squared <= 0.0 or target_norm <= 0.0:
        raise ValueError("invalid neutral-surgery geometry")
    for _attempt in range(max_attempts):
        z = np.asarray(rng.standard_normal(before.size), dtype=np.float64)
        direction = before * z
        direction -= weights * float(weights @ direction) / weight_norm_squared
        magnitude = float(np.linalg.norm(direction))
        if not np.isfinite(magnitude) or magnitude == 0.0:
            continue
        direction *= target_norm / magnitude
        first_sign = 1.0 if int(rng.integers(0, 2)) else -1.0
        for sign in (first_sign, -first_sign):
            after = before + sign * direction
            if np.all(after > 0.0) and np.isfinite(after).all():
                # Remove the final floating residual along w and renormalize.
                delta = after - before
                delta -= weights * float(weights @ delta) / weight_norm_squared
                delta *= target_norm / float(np.linalg.norm(delta))
                after = before + delta
                if np.any(after <= 0.0):
                    continue
                surgery = _surgery(
                    name, beta, flat, before, after, target_norm
                )
                before_throughput = catalytic_throughput(composition, beta)
                after_throughput = catalytic_throughput(composition, surgery.beta)
                tolerance = max(
                    NEUTRAL_ABSOLUTE_TOLERANCE,
                    NEUTRAL_RELATIVE_TOLERANCE * abs(before_throughput),
                )
                if abs(after_throughput - before_throughput) <= tolerance:
                    return surgery
    raise RuntimeError(
        f"no positive throughput-neutral direction in {max_attempts} attempts"
    )


def select_surgeries(
    composition: NDArray,
    beta: NDArray,
    balanced_rng: np.random.Generator,
    neutral_rng: np.random.Generator,
) -> tuple[BetaSurgery | None, ...]:
    present, _flat, _before = _present_block(composition, beta)
    if present.size < 2:
        return tuple(None for _ in ARMS)
    by_name: dict[str, BetaSurgery | None] = {
        "LOOSEN": multiplicative_surgery(composition, beta, 1.0 / 1.5, "LOOSEN"),
        "TIGHTEN": multiplicative_surgery(composition, beta, 1.5, "TIGHTEN"),
        "BALANCED_LOG_RANDOM": balanced_log_surgery(
            composition, beta, balanced_rng
        ),
        "THROUGHPUT_NEUTRAL_RANDOM": throughput_neutral_pp_surgery(
            composition, beta, neutral_rng
        ),
        "NOOP": None,
    }
    return tuple(by_name[name] for name in ARMS)


def _protocol() -> dict[str, Any]:
    pilot = phase_spec("pilot")
    confirmation = phase_spec("confirmation")
    value: dict[str, Any] = {
        "format": PROGRAM_FORMAT,
        "status": "frozen_before_any_p3c_scientific_matrix",
        "classification": "new additive prospective clarification program",
        "sealed_predecessor": {
            "p3b_unchanged": True,
            "p3b_target_contrast_succeeded": True,
            "p3b_formal_specificity_gate_passed": False,
            "p3b_not_rescued_by_p3c": True,
        },
        "endpoint": {
            "name": "JOINT_BREAK_RUN3",
            "horizon": HORIZON,
            "inheritance": "unrounded float64 H > 0.9",
            "break": "H <= 0.9",
            "renewal": "three consecutive inherited fissions strictly after break",
        },
        "arms": {
            "order": list(ARMS),
            "LOOSEN": {"factor": 1.0 / 1.5},
            "TIGHTEN": {"factor": 1.5},
            "BALANCED_LOG_RANDOM": {
                "location": "all and only P x P",
                "frobenius_fraction": SURGERY_NORM_FRACTION,
                "sum_log_change": 0.0,
                "required_null": False,
            },
            "THROUGHPUT_NEUTRAL_RANDOM": {
                "location": "all and only P x P",
                "frobenius_fraction": SURGERY_NORM_FRACTION,
                "throughput": "x.T @ beta @ x preserved",
                "positivity": "strict; no clipping",
                "maximum_attempts": NEUTRAL_MAX_ATTEMPTS,
                "required_null": True,
            },
            "NOOP": {"changed_edges": 0},
            "singleton_contract": "all-arm structural NOOP when |P| < 2",
        },
        "pilot": {
            "matrices": pilot.matrices,
            "candidates": list(CANDIDATES),
            "landmarks": list(pilot.landmarks),
            "branches": pilot.branches,
            "branch_halves": {"A": [0, 15], "B": [16, 31]},
            "primary_futures": 2
            * pilot.matrices
            * len(pilot.landmarks)
            * len(ARMS)
            * pilot.branches,
            "complete_replay": True,
            "advancement_scope": list(PRIMARY_LANDMARKS),
            "gates_in_every_candidate_half": [
                "LOOSEN - TIGHTEN > 0",
                "state-centered event slope on log throughput ratio < 0",
                "abs(THROUGHPUT_NEUTRAL_RANDOM - NOOP) <= 0.025",
                "surgery audit, replay, and readback exact",
            ],
            "cannot_support_confirmation_claim": True,
        },
        "confirmation": {
            "frozen_before_pilot_outcomes": True,
            "matrices": confirmation.matrices,
            "candidates": list(CANDIDATES),
            "landmarks": list(confirmation.landmarks),
            "primary_landmarks": list(PRIMARY_LANDMARKS),
            "compatibility_landmark": list(COMPATIBILITY_LANDMARKS),
            "branches": confirmation.branches,
            "primary_futures": 2
            * confirmation.matrices
            * len(confirmation.landmarks)
            * len(ARMS)
            * confirmation.branches,
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "randomization_repetitions": RANDOMIZATION_REPETITIONS,
            "holm_family": 4,
            "equivalence_margin": EQUIVALENCE_MARGIN,
            "gates_in_every_candidate_half": [
                "LOOSEN - TIGHTEN > 0, 95% lower > 0, Holm p < 0.05",
                "state-centered event slope on log throughput ratio < 0, 95% upper < 0",
                "mean within-state Spearman < 0, 95% upper < 0",
                "neutral-minus-NOOP 90% CI strictly inside +/-0.025",
                "exact replay and readback",
            ],
        },
        "resistance": {
            "reuses_confirmation_futures": True,
            "endpoint": "first break within F6",
            "registered_before_confirmation": True,
        },
        "resilience": {
            "run_only_after_confirmation_pass": True,
            "natural_break_acquisition_horizon": 12,
            "identical_post_break_daughter_across_arms": True,
            "future_horizon": 8,
            "branches": 32,
            "minimum_eligible_matrices_per_candidate": 120,
            "no_retries_or_replacements": True,
        },
        "inference": {
            "unit": "whole catalytic matrix",
            "states_and_landmarks_stay_with_matrix": True,
            "candidate_pooling": False,
            "future_seed_includes_arm": False,
            "matrix_replacement": False,
            "intervention_future_retries": 0,
        },
        "external_response": {
            "required_before_registration": True,
            "used_as_fitting_target": False,
            "may_change_protocol_after_archive": False,
        },
        "lifecycle": {
            "mandatory_stops": ["pilot", "confirmation", "resilience"],
            "cpu_budget_expected_hours": [14, 20],
            "cpu_budget_ceiling_hours": 30,
        },
        "seed_domains": SEED_DOMAINS,
        "claim_boundary": {
            "passing_confirmation_may_support": (
                "Codex-specific causal catalytic-throughput control of simulated "
                "JOINT_BREAK_RUN3"
            ),
            "cross_clean_room_requires_external_compatibility": True,
            "prohibited": [
                "repair or rescue of P3b",
                "control of strict-eight",
                "life or living organism",
                "agency or autonomous organization",
                "biological memory or error correction",
                "real prebiotic chemistry",
                "universal origin-of-life mechanism",
            ],
        },
    }
    value["protocol_id"] = _canonical_digest(value)
    return value


def archive_fable_response(input_path: Path, output: Path) -> None:
    input_path = input_path.resolve()
    output = output.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    request_path = REPOSITORY_ROOT / FABLE_REQUEST
    response = input_path.read_bytes()
    if not response.strip():
        raise ValueError("external response is empty")
    with _atomic_destination(output) as destination:
        shutil.copy2(request_path, destination / "request.md")
        (destination / "response.txt").write_bytes(response)
        manifest: dict[str, Any] = {
            "format": "codex-intervention-p3c-external-response-v1",
            "classification": "external hypothesis context; not a fitting target",
            "request_sha256": sha256_file(destination / "request.md"),
            "response_sha256": sha256_file(destination / "response.txt"),
            "archived_before_p3c_registration": True,
            "protocol_tuning_authorized": False,
        }
        manifest["archive_id"] = _canonical_digest(manifest)
        (destination / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_checksums(destination)
    verify_fable_archive(output)
    print(f"P3c external response archived: {output}", flush=True)


def verify_fable_archive(directory: Path) -> dict[str, Any]:
    directory = directory.resolve()
    verify_checksums(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    archive_id = manifest.pop("archive_id")
    if (
        manifest.get("format") != "codex-intervention-p3c-external-response-v1"
        or _canonical_digest(manifest) != archive_id
        or manifest["request_sha256"] != sha256_file(REPOSITORY_ROOT / FABLE_REQUEST)
        or manifest["request_sha256"] != sha256_file(directory / "request.md")
        or manifest["response_sha256"] != sha256_file(directory / "response.txt")
    ):
        raise ValueError("invalid P3c external-response archive")
    manifest["archive_id"] = archive_id
    return manifest


def validation_checks() -> dict[str, Any]:
    original = base.validation_checks()
    checks: dict[str, dict[str, Any]] = dict(original["checks"])

    def record(name: str, condition: bool, detail: Any = None) -> None:
        if not condition:
            raise AssertionError(f"P3c validation failed: {name}: {detail}")
        checks[name] = {"passed": True, "detail": detail}

    composition = np.asarray([4, 2, 1, 0], dtype=np.int64)
    beta = np.asarray(
        [
            [2.0, 0.7, 1.3, 0.4],
            [1.1, 3.0, 0.8, 0.9],
            [0.6, 1.4, 2.5, 1.2],
            [0.5, 0.3, 0.4, 1.8],
        ],
        dtype=np.float64,
    )
    neutral = throughput_neutral_pp_surgery(
        composition,
        beta,
        np.random.default_rng(derive_seed(SEED_DOMAINS["validation"], "neutral")),
    )
    present, expected_flat, before = _present_block(composition, beta)
    target = SURGERY_NORM_FRACTION * float(np.linalg.norm(before))
    record(
        "27_p3c_neutral_changes_all_and_only_present_present",
        np.array_equal(neutral.flat_indices, expected_flat)
        and np.count_nonzero(neutral.after != neutral.before) == expected_flat.size,
    )
    record(
        "28_p3c_neutral_exact_norm",
        np.isclose(neutral.observed_norm, target, atol=1e-11 * max(1.0, target), rtol=0.0),
    )
    before_t = catalytic_throughput(composition, beta)
    after_t = catalytic_throughput(composition, neutral.beta)
    record(
        "29_p3c_neutral_exact_throughput",
        abs(after_t - before_t)
        <= max(NEUTRAL_ABSOLUTE_TOLERANCE, NEUTRAL_RELATIVE_TOLERANCE * abs(before_t)),
        {"before": before_t, "after": after_t},
    )
    record("30_p3c_neutral_strict_positivity_without_clipping", bool(np.all(neutral.beta > 0.0)))
    neutral_repeat = throughput_neutral_pp_surgery(
        composition,
        beta,
        np.random.default_rng(derive_seed(SEED_DOMAINS["validation"], "neutral")),
    )
    record("31_p3c_neutral_selection_deterministic", np.array_equal(neutral.beta, neutral_repeat.beta))
    surgeries = select_surgeries(
        composition,
        beta,
        np.random.default_rng(derive_seed(SEED_DOMAINS["validation"], "balanced")),
        np.random.default_rng(derive_seed(SEED_DOMAINS["validation"], "neutral2")),
    )
    by_name = dict(zip(ARMS, surgeries, strict=True))
    record(
        "32_p3c_targeted_contract_exact",
        np.array_equal(by_name["TIGHTEN"].after, before * 1.5)
        and np.array_equal(by_name["LOOSEN"].after, before * (1.0 / 1.5)),
    )
    log_change = np.log(
        by_name["BALANCED_LOG_RANDOM"].after
        / by_name["BALANCED_LOG_RANDOM"].before
    )
    record("33_p3c_balanced_log_contract_exact", abs(float(log_change.sum())) <= 1e-12)
    singleton = np.asarray([0, 5, 0, 0], dtype=np.int64)
    singleton_arms = select_surgeries(
        singleton,
        beta,
        np.random.default_rng(1),
        np.random.default_rng(2),
    )
    record("34_p3c_singleton_is_all_arm_structural_noop", all(item is None for item in singleton_arms))
    record("35_p3c_noop_is_none", by_name["NOOP"] is None)
    record(
        "36_p3c_selection_and_future_domains_disjoint",
        len(set(SEED_DOMAINS.values())) == len(SEED_DOMAINS),
    )
    record(
        "37_p3c_pilot_and_confirmation_domains_disjoint",
        not {
            value for key, value in SEED_DOMAINS.items() if key.startswith("pilot_")
        }.intersection(
            value
            for key, value in SEED_DOMAINS.items()
            if key.startswith("confirmation_")
        ),
    )
    verify_checksums(P3B_RESULT)
    _verify_geometry_audit()
    record("38_p3c_sealed_p3b_and_posthoc_audit_verify", True)
    protocol = _protocol()
    record(
        "39_p3c_confirmation_frozen_inside_registration_protocol",
        protocol["confirmation"]["frozen_before_pilot_outcomes"],
    )
    record(
        "40_p3c_budget_below_user_ceiling",
        protocol["lifecycle"]["cpu_budget_expected_hours"][1]
        < protocol["lifecycle"]["cpu_budget_ceiling_hours"]
        and protocol["lifecycle"]["cpu_budget_ceiling_hours"] == 30,
    )
    record(
        "41_p3c_fable_response_required_before_registration",
        protocol["external_response"]["required_before_registration"],
    )
    return {
        "format": VALIDATION_FORMAT,
        "checks": checks,
        "original_required_checks_passed": original["required_checks_passed"],
        "all_checks_passed": all(item["passed"] for item in checks.values()),
        "check_count": len(checks),
        "scientific_cohort_generated": False,
        "scientific_effect_sizes_computed": False,
    }


def run_validation(output: Path) -> None:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    validation = validation_checks()
    command = [
        str(REPOSITORY_ROOT / ".venv/bin/python"),
        "-m",
        "pytest",
        "-q",
    ]
    completed = subprocess.run(
        command, cwd=REPOSITORY_ROOT, capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stdout + completed.stderr)
    with _atomic_destination(output) as destination:
        (destination / "validation.json").write_text(
            json.dumps(_json_ready(validation), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (destination / "pytest_output.txt").write_text(
            completed.stdout + completed.stderr, encoding="utf-8"
        )
        write_checksums(destination)
    verify_checksums(output)
    print(f"P3c validation sealed: {output}", flush=True)


def register_program(
    validation_directory: Path,
    fable_archive: Path,
    output: Path,
) -> None:
    validation_directory = validation_directory.resolve()
    fable_archive = fable_archive.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    for scientific in (
        DEFAULT_PILOT_OUTPUT,
        DEFAULT_PILOT_WORK,
        DEFAULT_CONFIRMATION_OUTPUT,
        DEFAULT_CONFIRMATION_WORK,
        DEFAULT_RESILIENCE_OUTPUT,
        DEFAULT_RESILIENCE_WORK,
    ):
        if scientific.exists():
            raise FileExistsError(
                f"P3c scientific artifact exists before registration: {scientific}"
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
        raise ValueError("P3c validation is not registration eligible")
    external = verify_fable_archive(fable_archive)
    original = base.verify_registration(ORIGINAL_REGISTRATION)
    p3b_registration = p3b.verify_registration(P3B_REGISTRATION)
    p3b_amendment = p3b_recovery.verify_amendment(P3B_RECOVERY)
    verify_checksums(P3B_RESULT)
    _verify_geometry_audit()
    model_path = ORIGINAL_REGISTRATION / "frozen_full_predictor.npz"
    if sha256_file(model_path) != base.EXPECTED_MODEL_SHA256:
        raise ValueError("unexpected frozen JOINT_BREAK_RUN3 predictor")
    protocol = _protocol()
    with _atomic_destination(output) as destination:
        (destination / "protocol.json").write_text(
            json.dumps(_json_ready(protocol), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        confirmation_protocol = protocol["confirmation"] | {
            "format": "codex-intervention-p3c-frozen-confirmation-protocol-v1",
            "parent_protocol_id": protocol["protocol_id"],
            "immutable_after_pilot": True,
        }
        (destination / "frozen_confirmation_protocol.json").write_text(
            json.dumps(_json_ready(confirmation_protocol), indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        seed_registry = {
            "format": "codex-intervention-p3c-seed-registry-v1",
            "domains": SEED_DOMAINS,
            "all_unique": len(set(SEED_DOMAINS.values())) == len(SEED_DOMAINS),
            "future_seed_includes_arm": False,
            "selection_streams_distinct_from_future": True,
            "pilot_confirmation_resilience_domains_disjoint": True,
        }
        (destination / "seed_registry.json").write_text(
            json.dumps(seed_registry, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        shutil.copy2(validation_directory / "validation.json", destination / "validation.json")
        shutil.copy2(validation_directory / "pytest_output.txt", destination / "pytest_output.txt")
        shutil.copy2(model_path, destination / "frozen_full_predictor.npz")
        shutil.copy2(fable_archive / "request.md", destination / "external_request.md")
        shutil.copy2(fable_archive / "response.txt", destination / "external_response.txt")
        payload: dict[str, Any] = {
            "format": REGISTRATION_FORMAT,
            "status": "sealed_before_any_p3c_scientific_matrix",
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": sha256_file(destination / "protocol.json"),
            "frozen_confirmation_protocol_sha256": sha256_file(
                destination / "frozen_confirmation_protocol.json"
            ),
            "seed_registry_sha256": sha256_file(destination / "seed_registry.json"),
            "source_hashes": _source_hashes(),
            "frozen_predictor_sha256": sha256_file(
                destination / "frozen_full_predictor.npz"
            ),
            "original_registration_id": original["registration_id"],
            "p3b_registration_id": p3b_registration["registration_id"],
            "p3b_singleton_amendment_id": p3b_amendment["amendment_id"],
            "p3b_result_checksum_manifest_sha256": sha256_file(
                P3B_RESULT / "SHA256SUMS"
            ),
            "geometry_audit_checksum_manifest_sha256": sha256_file(
                GEOMETRY_AUDIT / "SHA256SUMS"
            ),
            "external_archive_id": external["archive_id"],
            "external_response_sha256": external["response_sha256"],
            "validation_checksum_manifest_sha256": sha256_file(
                validation_directory / "SHA256SUMS"
            ),
            "p3c_scientific_matrices_generated": False,
            "p3c_effect_sizes_computed": False,
        }
        payload["registration_id"] = _canonical_digest(payload)
        (destination / "registration.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_checksums(destination)
    registration = verify_registration(output)
    print(f"P3c registration sealed: {registration['registration_id']}", flush=True)


def verify_registration(directory: Path) -> dict[str, Any]:
    directory = directory.resolve()
    verify_checksums(directory)
    payload = json.loads((directory / "registration.json").read_text(encoding="utf-8"))
    registration_id = payload.pop("registration_id")
    if (
        payload.get("format") != REGISTRATION_FORMAT
        or payload.get("status") != "sealed_before_any_p3c_scientific_matrix"
        or _canonical_digest(payload) != registration_id
    ):
        raise ValueError("invalid P3c registration")
    payload["registration_id"] = registration_id
    if payload["source_hashes"] != _source_hashes():
        raise ValueError("P3c source changed after registration")
    protocol = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
    if protocol != json.loads(json.dumps(_json_ready(_protocol()))):
        raise ValueError("P3c protocol implementation diverged")
    if (
        protocol["protocol_id"] != payload["protocol_id"]
        or sha256_file(directory / "protocol.json") != payload["protocol_sha256"]
    ):
        raise ValueError("P3c protocol hash changed")
    if sha256_file(directory / "frozen_full_predictor.npz") != base.EXPECTED_MODEL_SHA256:
        raise ValueError("P3c frozen predictor changed")
    seed_registry = json.loads(
        (directory / "seed_registry.json").read_text(encoding="utf-8")
    )
    if (
        seed_registry["domains"] != SEED_DOMAINS
        or not seed_registry["all_unique"]
        or seed_registry["future_seed_includes_arm"]
    ):
        raise ValueError("P3c seed registry changed")
    if payload["external_response_sha256"] != sha256_file(
        directory / "external_response.txt"
    ):
        raise ValueError("archived external response changed")
    verify_checksums(P3B_RESULT)
    _verify_geometry_audit()
    return payload


def _experiment(spec: P3CSpec) -> ExperimentConfig:
    cohort = CohortConfig(spec.matrices, spec.branches, spec.landmarks)
    return ExperimentConfig(
        development=cohort,
        confirmation=cohort,
        horizon=spec.horizon,
        bootstrap_repetitions=BOOTSTRAP_REPETITIONS,
        permutation_repetitions=RANDOMIZATION_REPETITIONS,
        regenerate_confirmation=True,
        master_seed=spec.cohort_seed,
    )


def _balanced_seed(spec: P3CSpec, case: StateCase) -> int:
    return derive_seed(
        spec.balanced_selection_seed,
        f"{LABELS[spec.stage]}.selection.balanced_log",
        case.candidate,
        case.matrix_id,
        case.landmark,
    )


def _neutral_seed(spec: P3CSpec, case: StateCase) -> int:
    return derive_seed(
        spec.neutral_selection_seed,
        f"{LABELS[spec.stage]}.selection.throughput_neutral",
        case.candidate,
        case.matrix_id,
        case.landmark,
    )


def _future_seed(spec: P3CSpec, case: StateCase, branch: int) -> int:
    return derive_seed(
        spec.future_seed,
        f"{LABELS[spec.stage]}.future",
        case.candidate,
        case.matrix_id,
        case.landmark,
        branch,
    )


def _has_run(values: NDArray, length: int) -> tuple[bool, int]:
    flags = np.asarray(values, dtype=bool)
    if flags.size < length:
        return False, -1
    for start in range(flags.size - length + 1):
        if bool(flags[start : start + length].all()):
            return True, start + length
    return False, -1


def simulate_resilience_one_shot(
    snapshot: Snapshot,
    beta: NDArray,
    candidate: str,
    experiment: ExperimentConfig,
    horizon: int,
    rng: np.random.Generator,
) -> InterventionOutcome:
    records, completed = simulate_future_absorbing(
        snapshot,
        beta,
        experiment.gard,
        CANDIDATES[candidate],
        horizon,
        rng,
    )
    ordinary = outcome_from_records(snapshot, records, completed, horizon)
    inherited = np.asarray(
        [record.h > experiment.gard.inheritance_threshold for record in records],
        dtype=bool,
    )
    run3, certification = _has_run(inherited, 3)
    return replace(
        ordinary,
        joint_break_run3=run3,
        run3_after_break=run3,
        renewal_certification_time=certification,
    )


def _phase_worker(
    arguments: tuple[StateCase, ExperimentConfig, P3CSpec, str]
) -> base.PhaseBatch:
    case, experiment, spec, model_path = arguments
    limiter = threadpool_limits(limits=1)
    try:
        predictor = FrozenFullPredictor.load(model_path)
        surgeries = select_surgeries(
            case.snapshot.composition,
            case.beta,
            np.random.default_rng(_balanced_seed(spec, case)),
            np.random.default_rng(_neutral_seed(spec, case)),
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
        outcomes: list[list[Any]] = [[] for _ in spec.arms]
        for branch in range(spec.branches):
            seed = _future_seed(spec, case, branch)
            for arm_index, surgery in enumerate(surgeries):
                if spec.stage == "resilience":
                    outcome = simulate_resilience_one_shot(
                        case.snapshot,
                        case.beta if surgery is None else surgery.beta,
                        case.candidate,
                        experiment,
                        spec.horizon,
                        np.random.default_rng(seed),
                    )
                else:
                    outcome = simulate_one_shot(
                        case.snapshot,
                        case.beta if surgery is None else surgery.beta,
                        case.candidate,
                        experiment.gard,
                        spec.horizon,
                        np.random.default_rng(seed),
                        None,
                    )
                outcomes[arm_index].append(outcome)
        frozen_outcomes = tuple(tuple(arm) for arm in outcomes)
        if all(surgery is None for surgery in surgeries):
            for branch in range(spec.branches):
                if len(
                    {
                        frozen_outcomes[arm][branch].record_digest
                        for arm in range(len(spec.arms))
                    }
                ) != 1:
                    raise AssertionError("structural no-op arms diverged")
        return base.PhaseBatch(
            state_id=case.state_id,
            state_digest=base._snapshot_digest(case),
            arm_names=spec.arms,
            predictions=predictions,
            selected_edits=tuple(None for _ in spec.arms),
            surgeries=surgeries,
            scored_edits=tuple(),
            catalytic_support=np.empty(0, dtype=np.float64),
            outcomes=frozen_outcomes,
        )
    finally:
        limiter.restore_original_limits()


def _checkpoint_contract(
    cases: list[StateCase], spec: P3CSpec, registration_id: str, stage: str
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "registration_id": registration_id,
        "scientific_label": LABELS[spec.stage],
        "phase": spec.phase,
        "stage": stage,
        "matrices": spec.matrices,
        "branches": spec.branches,
        "landmarks": list(spec.landmarks),
        "horizon": spec.horizon,
        "arms": list(spec.arms),
        "case_ids": [case.state_id for case in cases],
        "case_digests": [base._snapshot_digest(case) for case in cases],
        "future_seed": spec.future_seed,
        "future_seed_includes_arm": False,
        "balanced_selection_seed": spec.balanced_selection_seed,
        "neutral_selection_seed": spec.neutral_selection_seed,
        "source_hashes": _source_hashes(),
    }
    value["contract_id"] = _canonical_digest(value)
    return value


def run_phase_batches(
    cases: list[StateCase],
    experiment: ExperimentConfig,
    spec: P3CSpec,
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
            raise ValueError("P3c checkpoint contract changed")
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
                raise ValueError(f"invalid P3c checkpoint {path}")
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
        raise AssertionError("P3c checkpoint stage has missing states")
    return [batch for batch in batches if batch is not None]


def _rank(values: NDArray) -> NDArray[np.float64]:
    return pd.Series(np.asarray(values, dtype=np.float64)).rank(method="average").to_numpy()


def _spearman(x: NDArray, y: NDArray) -> float:
    xr = _rank(x)
    yr = _rank(y)
    if np.std(xr) == 0.0 or np.std(yr) == 0.0:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def _geometry_arrays(
    cases: list[StateCase], batches: list[base.PhaseBatch], spec: P3CSpec
) -> tuple[dict[str, NDArray], pd.DataFrame, dict[str, Any]]:
    throughput = np.empty((len(cases), len(spec.arms)), dtype=np.float64)
    log_ratio = np.empty_like(throughput)
    rows: list[dict[str, Any]] = []
    all_pp = True
    all_positive = True
    neutral_norm_exact = True
    neutral_throughput_exact = True
    maximum_neutral_throughput_error = 0.0
    maximum_norm_relative_error = 0.0
    structural_states = 0
    for state_index, (case, batch) in enumerate(zip(cases, batches, strict=True)):
        composition = case.snapshot.composition
        baseline = catalytic_throughput(composition, case.beta)
        present, expected_flat, before = _present_block(composition, case.beta)
        expected_set = set(expected_flat.tolist())
        singleton = present.size < 2
        if singleton:
            structural_states += 1
        for arm_index, (arm, surgery) in enumerate(
            zip(spec.arms, batch.surgeries, strict=True)
        ):
            altered = case.beta if surgery is None else surgery.beta
            observed_throughput = catalytic_throughput(composition, altered)
            throughput[state_index, arm_index] = observed_throughput
            log_ratio[state_index, arm_index] = np.log(observed_throughput / baseline)
            if surgery is None:
                changed_set: set[int] = set()
                requested = 0.0
                observed_norm = 0.0
                norm_relative_error = 0.0
                minimum_after = float(case.beta.min())
            else:
                changed_set = set(np.asarray(surgery.flat_indices, dtype=np.int64).tolist())
                requested = float(surgery.requested_norm)
                observed_norm = float(surgery.observed_norm)
                norm_relative_error = abs(observed_norm - requested) / requested
                minimum_after = float(surgery.beta.min())
                all_pp = all_pp and changed_set == expected_set
                all_positive = all_positive and minimum_after > 0.0
                maximum_norm_relative_error = max(
                    maximum_norm_relative_error, norm_relative_error
                )
            throughput_error = observed_throughput - baseline
            if arm == "THROUGHPUT_NEUTRAL_RANDOM":
                tolerance = max(
                    NEUTRAL_ABSOLUTE_TOLERANCE,
                    NEUTRAL_RELATIVE_TOLERANCE * abs(baseline),
                )
                neutral_norm_exact = neutral_norm_exact and (
                    surgery is None
                    or abs(observed_norm - requested)
                    <= 1e-11 * max(1.0, requested)
                )
                neutral_throughput_exact = neutral_throughput_exact and (
                    surgery is None or abs(throughput_error) <= tolerance
                )
                maximum_neutral_throughput_error = max(
                    maximum_neutral_throughput_error, abs(throughput_error)
                )
            rows.append(
                {
                    "state_id": case.state_id,
                    "candidate": case.candidate,
                    "matrix_id": case.matrix_id,
                    "landmark": case.landmark,
                    "arm": arm,
                    "present_types": int(present.size),
                    "present_present_edges": int(before.size),
                    "structural_no_action": bool(singleton),
                    "changed_edges": len(changed_set),
                    "all_and_only_present_present": bool(
                        surgery is None or changed_set == expected_set
                    ),
                    "requested_frobenius_norm": requested,
                    "observed_frobenius_norm": observed_norm,
                    "norm_relative_error": norm_relative_error,
                    "throughput_noop": baseline,
                    "throughput_arm": observed_throughput,
                    "throughput_difference": throughput_error,
                    "log_throughput_ratio": log_ratio[state_index, arm_index],
                    "minimum_beta_after": minimum_after,
                }
            )
    summary = {
        "format": "codex-intervention-p3c-surgery-audit-v1",
        "states": len(cases),
        "rows": len(rows),
        "structural_no_action_states": structural_states,
        "all_nonnoop_surgeries_change_all_and_only_present_present": all_pp,
        "all_beta_strictly_positive": all_positive,
        "neutral_exact_registered_norm": neutral_norm_exact,
        "neutral_exact_registered_throughput": neutral_throughput_exact,
        "maximum_neutral_throughput_absolute_error": maximum_neutral_throughput_error,
        "maximum_surgery_norm_relative_error": maximum_norm_relative_error,
        "no_clipping": True,
    }
    summary["all_audits_pass"] = bool(
        all_pp
        and all_positive
        and neutral_norm_exact
        and neutral_throughput_exact
    )
    return {"throughput": throughput, "log_throughput_ratio": log_ratio}, pd.DataFrame(rows), summary


def _outcome_arrays(
    cases: list[StateCase], batches: list[base.PhaseBatch], spec: P3CSpec
) -> dict[str, NDArray]:
    shape = (len(cases), len(spec.arms), spec.branches)
    result: dict[str, NDArray] = {
        "targets": np.empty(shape, dtype=np.int8),
        "break_event": np.empty(shape, dtype=np.int8),
        "run3_after_break": np.empty(shape, dtype=np.int8),
        "inherited_boundary_count": np.empty(shape, dtype=np.int8),
        "first_break_time": np.empty(shape, dtype=np.int8),
        "renewal_certification_time": np.empty(shape, dtype=np.int8),
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
    field_map = {
        "targets": "joint_break_run3",
        "break_event": "break_event",
        "run3_after_break": "run3_after_break",
        "inherited_boundary_count": "inherited_boundary_count",
        "first_break_time": "first_break_time",
        "renewal_certification_time": "renewal_certification_time",
        "completed_horizon": "completed_horizon",
        "observed_fissions": "observed_fissions",
        "total_growth_updates": "total_growth_updates",
        "mean_growth_updates": "mean_growth_updates",
        "final_entropy": "final_entropy",
        "final_occupied_types": "final_occupied_types",
        "boundary_h": "boundary_h",
        "growth_updates": "growth_updates",
        "final_composition": "final_composition",
    }
    for state_index, batch in enumerate(batches):
        for arm_index, arm in enumerate(batch.outcomes):
            for branch, outcome in enumerate(arm):
                location = (state_index, arm_index, branch)
                for array_name, attribute in field_map.items():
                    result[array_name][location] = getattr(outcome, attribute)
    return result


def _slope_and_rank_statistics(
    x: NDArray,
    y: NDArray,
    matrix_ids: NDArray[np.int64],
    bootstrap_indices: NDArray[np.int64],
) -> tuple[dict[str, Any], dict[str, NDArray]]:
    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    ids = np.asarray(matrix_ids, dtype=np.int64)
    if x_values.shape != y_values.shape or x_values.ndim != 2:
        raise ValueError("slope inputs must be aligned state by arm matrices")
    x_centered = x_values - x_values.mean(axis=1, keepdims=True)
    y_centered = y_values - y_values.mean(axis=1, keepdims=True)
    state_numerator = np.sum(x_centered * y_centered, axis=1)
    state_denominator = np.sum(x_centered * x_centered, axis=1)
    matrix_order = np.sort(np.unique(ids))
    numerator = np.asarray(
        [state_numerator[ids == mid].sum() for mid in matrix_order],
        dtype=np.float64,
    )
    denominator = np.asarray(
        [state_denominator[ids == mid].sum() for mid in matrix_order],
        dtype=np.float64,
    )
    slope = float(numerator.sum() / denominator.sum())
    slope_draws = numerator[bootstrap_indices].sum(axis=1) / denominator[
        bootstrap_indices
    ].sum(axis=1)
    state_rho = np.asarray(
        [_spearman(x_values[index], y_values[index]) for index in range(len(ids))],
        dtype=np.float64,
    )
    matrix_rho = np.asarray(
        [
            np.nanmean(state_rho[ids == mid])
            if np.isfinite(state_rho[ids == mid]).any()
            else np.nan
            for mid in matrix_order
        ],
        dtype=np.float64,
    )
    if not np.isfinite(matrix_rho).any():
        raise ValueError("no finite within-state rank correlations")
    with np.errstate(invalid="ignore"):
        rho_draws = np.nanmean(matrix_rho[bootstrap_indices], axis=1)
    if not np.isfinite(rho_draws).all():
        raise ValueError("bootstrap rank draw lacks informative matrices")
    summary = {
        "state_centered_slope": slope,
        "slope_bootstrap_ci95": _interval(slope_draws),
        "mean_within_state_spearman": float(np.nanmean(matrix_rho)),
        "spearman_bootstrap_ci95": _interval(rho_draws),
        "informative_state_correlations": int(np.count_nonzero(np.isfinite(state_rho))),
        "total_states": int(state_rho.size),
        "matrices_with_defined_rank_mean": int(np.count_nonzero(np.isfinite(matrix_rho))),
    }
    return summary, {"slope": slope_draws, "spearman": rho_draws}


def _analyze_scope(
    scope: str,
    landmarks: tuple[int, ...],
    cases: list[StateCase],
    endpoint: NDArray,
    log_throughput_ratio: NDArray,
    draws: dict[str, NDArray],
    stage: str,
    endpoint_name: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, NDArray]]:
    values = np.asarray(endpoint, dtype=np.float64)
    geometry = np.asarray(log_throughput_ratio, dtype=np.float64)
    bootstrap_indices = np.asarray(draws["bootstrap_indices"], dtype=np.int64)
    signs = np.asarray(draws["randomization_signs"], dtype=np.float64)
    arm_index = {arm: index for index, arm in enumerate(ARMS)}
    matrix_order = np.arange(phase_spec(stage).matrices, dtype=np.int64)
    cells: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    arrays: dict[str, NDArray] = {}
    raw_p: list[float] = []
    for candidate in CANDIDATES:
        selected_mask = np.asarray(
            [
                case.candidate == candidate and case.landmark in landmarks
                for case in cases
            ],
            dtype=bool,
        )
        selected_cases = [case for case, keep in zip(cases, selected_mask) if keep]
        ids = np.asarray([case.matrix_id for case in selected_cases], dtype=np.int64)
        if not np.array_equal(np.unique(ids), matrix_order):
            raise ValueError(f"{scope} candidate {candidate} lacks a matrix")
        candidate_values = values[selected_mask]
        candidate_geometry = geometry[selected_mask]
        for half, branch_slice in (
            ("A", slice(0, BRANCHES // 2)),
            ("B", slice(BRANCHES // 2, BRANCHES)),
        ):
            q = candidate_values[:, :, branch_slice].mean(axis=2)
            y = q - q[:, [arm_index["NOOP"]]]
            association, association_draws = _slope_and_rank_statistics(
                candidate_geometry, y, ids, bootstrap_indices
            )
            target_state = q[:, arm_index["LOOSEN"]] - q[:, arm_index["TIGHTEN"]]
            neutral_state = (
                q[:, arm_index["THROUGHPUT_NEUTRAL_RANDOM"]]
                - q[:, arm_index["NOOP"]]
            )
            balanced_state = (
                q[:, arm_index["BALANCED_LOG_RANDOM"]]
                - q[:, arm_index["NOOP"]]
            )
            matrix_target = _matrix_means(target_state, ids, matrix_order)
            matrix_neutral = _matrix_means(neutral_state, ids, matrix_order)
            matrix_balanced = _matrix_means(balanced_state, ids, matrix_order)
            target_boot = _bootstrap_means(matrix_target, bootstrap_indices)
            neutral_boot = _bootstrap_means(matrix_neutral, bootstrap_indices)
            balanced_boot = _bootstrap_means(matrix_balanced, bootstrap_indices)
            p_raw, randomization_null = _one_sided_sign_p(matrix_target, signs)
            raw_p.append(p_raw)
            ci90 = _interval(neutral_boot, alpha=0.10)
            cell_key = f"{endpoint_name}__{scope}__c{candidate}_{half}"
            arrays[f"{cell_key}__target_bootstrap"] = target_boot
            arrays[f"{cell_key}__neutral_bootstrap"] = neutral_boot
            arrays[f"{cell_key}__balanced_bootstrap"] = balanced_boot
            arrays[f"{cell_key}__target_randomization"] = randomization_null
            arrays[f"{cell_key}__slope_bootstrap"] = association_draws["slope"]
            arrays[f"{cell_key}__spearman_bootstrap"] = association_draws["spearman"]
            arm_means = {
                arm: float(
                    _matrix_means(q[:, index], ids, matrix_order).mean()
                )
                for arm, index in arm_index.items()
            }
            cell: dict[str, Any] = {
                "cell": f"c{candidate}_{half}",
                "scope": scope,
                "endpoint": endpoint_name,
                "candidate": candidate,
                "branch_half": half,
                "matrices": len(matrix_order),
                "states": len(selected_cases),
                "arm_means": arm_means,
                "target_loosen_minus_tighten": {
                    "estimate": float(matrix_target.mean()),
                    "bootstrap_ci95": _interval(target_boot),
                    "randomization_p_raw": p_raw,
                    "matrices_expected_sign": int(np.count_nonzero(matrix_target > 0.0)),
                    "matrices_zero": int(np.count_nonzero(matrix_target == 0.0)),
                    "maximum_leave_one_matrix_out_influence": _maximum_leave_one_out_influence(matrix_target),
                },
                "neutral_minus_noop": {
                    "estimate": float(matrix_neutral.mean()),
                    "bootstrap_ci95": _interval(neutral_boot),
                    "bootstrap_ci90": ci90,
                    "tost_equivalent_margin_0_025": bool(
                        ci90[0] > -EQUIVALENCE_MARGIN
                        and ci90[1] < EQUIVALENCE_MARGIN
                    ),
                },
                "balanced_log_minus_noop": {
                    "estimate": float(matrix_balanced.mean()),
                    "bootstrap_ci95": _interval(balanced_boot),
                    "required_null": False,
                },
                "throughput_association": association,
            }
            cells.append(cell)
            for position, matrix_id in enumerate(matrix_order):
                matrix_rows.append(
                    {
                        "endpoint": endpoint_name,
                        "scope": scope,
                        "candidate": candidate,
                        "branch_half": half,
                        "matrix_id": int(matrix_id),
                        "target_loosen_minus_tighten": float(matrix_target[position]),
                        "neutral_minus_noop": float(matrix_neutral[position]),
                        "balanced_log_minus_noop": float(matrix_balanced[position]),
                    }
                )
    adjusted = holm_adjust(raw_p)
    for cell, adjusted_p in zip(cells, adjusted, strict=True):
        cell["target_loosen_minus_tighten"]["randomization_p_holm"] = float(adjusted_p)
        target = cell["target_loosen_minus_tighten"]
        neutral = cell["neutral_minus_noop"]
        association = cell["throughput_association"]
        if stage == "pilot":
            gates = {
                "target_positive": target["estimate"] > 0.0,
                "throughput_slope_negative": association["state_centered_slope"] < 0.0,
                "neutral_absolute_point_within_margin": abs(neutral["estimate"])
                <= EQUIVALENCE_MARGIN,
            }
        else:
            gates = {
                "target_positive": target["estimate"] > 0.0,
                "target_bootstrap_lower_positive": target["bootstrap_ci95"][0] > 0.0,
                "target_holm_p_below_0_05": adjusted_p < 0.05,
                "throughput_slope_negative": association["state_centered_slope"] < 0.0,
                "throughput_slope_bootstrap_upper_negative": association[
                    "slope_bootstrap_ci95"
                ][1]
                < 0.0,
                "within_state_spearman_negative": association[
                    "mean_within_state_spearman"
                ]
                < 0.0,
                "spearman_bootstrap_upper_negative": association[
                    "spearman_bootstrap_ci95"
                ][1]
                < 0.0,
                "neutral_tost_equivalent": neutral[
                    "tost_equivalent_margin_0_025"
                ],
            }
        cell["statistical_gates"] = gates
        cell["statistical_cell_pass"] = bool(all(gates.values()))
    return (
        {
            "scope": scope,
            "landmarks": list(landmarks),
            "endpoint": endpoint_name,
            "cells": cells,
            "all_statistical_cells_pass": bool(
                all(cell["statistical_cell_pass"] for cell in cells)
            ),
        },
        matrix_rows,
        arrays,
    )


def compute_inference(
    cases: list[StateCase],
    arrays: dict[str, NDArray],
    geometry: dict[str, NDArray],
    draws: dict[str, NDArray],
    stage: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, NDArray]]:
    primary, rows, stored = _analyze_scope(
        "five_standard_landmarks",
        PRIMARY_LANDMARKS,
        cases,
        arrays["targets"],
        geometry["log_throughput_ratio"],
        draws,
        stage,
        "JOINT_BREAK_RUN3_F12",
    )
    compatibility, compatibility_rows, compatibility_stored = _analyze_scope(
        "landmark60_compatibility",
        COMPATIBILITY_LANDMARKS,
        cases,
        arrays["targets"],
        geometry["log_throughput_ratio"],
        draws,
        stage,
        "JOINT_BREAK_RUN3_F12",
    )
    first_break = np.asarray(arrays["first_break_time"])
    break_f6 = ((first_break >= 1) & (first_break <= 6)).astype(np.int8)
    resistance, resistance_rows, resistance_stored = _analyze_scope(
        "five_standard_landmarks",
        PRIMARY_LANDMARKS,
        cases,
        break_f6,
        geometry["log_throughput_ratio"],
        draws,
        stage,
        "FIRST_BREAK_F6",
    )
    stored.update(compatibility_stored)
    stored.update(resistance_stored)
    result = {
        "format": "codex-intervention-p3c-inference-v1",
        "stage": stage,
        "inference_unit": "whole catalytic matrix",
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "randomization_repetitions": RANDOMIZATION_REPETITIONS,
        "equivalence_margin": EQUIVALENCE_MARGIN,
        "primary": primary,
        "landmark60_compatibility": compatibility,
        "resistance": resistance,
        "resistance_is_registered_secondary_mechanism": True,
        "balanced_log_random_required_null": False,
    }
    return result, rows + compatibility_rows + resistance_rows, stored


def _acquisition_seed(case: StateCase) -> int:
    return derive_seed(
        SEED_DOMAINS["resilience_acquisition"],
        "INTP3C_SHARED_BREAK_RESILIENCE_V1.natural_acquisition",
        case.candidate,
        case.matrix_id,
        case.landmark,
    )


def acquire_natural_broken_state(
    case: StateCase, experiment: ExperimentConfig, horizon: int = 12
) -> tuple[StateCase | None, NDArray[np.int64] | None, dict[str, Any]]:
    rng = np.random.default_rng(_acquisition_seed(case))
    current = np.asarray(case.snapshot.composition, dtype=np.int64).copy()
    inheritance = list(case.snapshot.inheritance)
    boundary_h = list(case.snapshot.boundary_h)
    cumulative = int(case.snapshot.cumulative_growth_steps)
    for offset in range(1, horizon + 1):
        try:
            record = advance_fission(
                current,
                case.beta,
                experiment.gard,
                CANDIDATES[case.candidate],
                rng,
            )
        except SimulationError:
            return None, None, {
                "source_state_id": case.state_id,
                "candidate": case.candidate,
                "matrix_id": case.matrix_id,
                "landmark": case.landmark,
                "eligible": False,
                "reason": "extinction_before_natural_break",
                "observed_fissions": offset - 1,
                "break_time": -1,
            }
        inherited = bool(record.h > experiment.gard.inheritance_threshold)
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
                cohort="INTP3C_SHARED_NATURAL_BREAK",
                candidate=case.candidate,
                matrix_id=case.matrix_id,
                landmark=case.landmark,
                beta=case.beta,
                snapshot=snapshot,
            )
            return broken, np.asarray(record.parent, dtype=np.int64).copy(), {
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
            }
        current = record.daughter
    return None, None, {
        "source_state_id": case.state_id,
        "candidate": case.candidate,
        "matrix_id": case.matrix_id,
        "landmark": case.landmark,
        "eligible": False,
        "reason": "no_natural_break_within_f12",
        "observed_fissions": horizon,
        "break_time": -1,
    }


def acquire_resilience_cohort(
    cases: list[StateCase], experiment: ExperimentConfig
) -> tuple[list[StateCase], list[NDArray[np.int64]], pd.DataFrame]:
    broken: list[StateCase] = []
    anchors: list[NDArray[np.int64]] = []
    rows: list[dict[str, Any]] = []
    for case in cases:
        acquired, anchor, row = acquire_natural_broken_state(case, experiment)
        rows.append(row)
        if acquired is not None:
            if anchor is None:
                raise AssertionError("eligible broken state lacks its old anchor")
            broken.append(acquired)
            anchors.append(anchor)
    return broken, anchors, pd.DataFrame(rows)


def _resilience_inference(
    cases: list[StateCase],
    arrays: dict[str, NDArray],
    geometry: dict[str, NDArray],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, NDArray]]:
    arm_index = {arm: index for index, arm in enumerate(ARMS)}
    targets = np.asarray(arrays["targets"], dtype=np.float64)
    log_throughput = np.asarray(geometry["log_throughput_ratio"], dtype=np.float64)
    cells: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    stored: dict[str, NDArray] = {}
    raw_p: list[float] = []
    eligible_counts: dict[str, int] = {}
    for candidate in CANDIDATES:
        mask = np.asarray([case.candidate == candidate for case in cases], dtype=bool)
        candidate_cases = [case for case, keep in zip(cases, mask) if keep]
        ids = np.asarray([case.matrix_id for case in candidate_cases], dtype=np.int64)
        matrix_order = np.sort(np.unique(ids))
        eligible_counts[candidate] = int(matrix_order.size)
        if matrix_order.size < 2:
            raise ValueError("resilience inference requires two eligible matrices")
        draws = generate_inference_draws(
            matrix_order.size,
            BOOTSTRAP_REPETITIONS,
            RANDOMIZATION_REPETITIONS,
            np.random.default_rng(
                derive_seed(
                    SEED_DOMAINS["resilience_bootstrap"],
                    f"P3C.resilience.bootstrap.c{candidate}",
                )
            ),
            np.random.default_rng(
                derive_seed(
                    SEED_DOMAINS["resilience_randomization"],
                    f"P3C.resilience.randomization.c{candidate}",
                )
            ),
        )
        bootstrap_indices = draws["bootstrap_indices"]
        signs = draws["randomization_signs"]
        stored[f"c{candidate}__bootstrap_indices"] = bootstrap_indices
        stored[f"c{candidate}__randomization_signs"] = signs
        q_candidate = targets[mask]
        x_candidate = log_throughput[mask]
        for half, branch_slice in (
            ("A", slice(0, BRANCHES // 2)),
            ("B", slice(BRANCHES // 2, BRANCHES)),
        ):
            q = q_candidate[:, :, branch_slice].mean(axis=2)
            y = q - q[:, [arm_index["NOOP"]]]
            association, association_draws = _slope_and_rank_statistics(
                x_candidate, y, ids, bootstrap_indices
            )
            target_state = q[:, arm_index["TIGHTEN"]] - q[:, arm_index["LOOSEN"]]
            neutral_state = (
                q[:, arm_index["THROUGHPUT_NEUTRAL_RANDOM"]]
                - q[:, arm_index["NOOP"]]
            )
            matrix_target = _matrix_means(target_state, ids, matrix_order)
            matrix_neutral = _matrix_means(neutral_state, ids, matrix_order)
            target_boot = _bootstrap_means(matrix_target, bootstrap_indices)
            neutral_boot = _bootstrap_means(matrix_neutral, bootstrap_indices)
            p_raw, null = _one_sided_sign_p(matrix_target, signs)
            raw_p.append(p_raw)
            ci90 = _interval(neutral_boot, alpha=0.10)
            key = f"c{candidate}_{half}"
            stored[f"{key}__target_bootstrap"] = target_boot
            stored[f"{key}__neutral_bootstrap"] = neutral_boot
            stored[f"{key}__target_randomization"] = null
            stored[f"{key}__slope_bootstrap"] = association_draws["slope"]
            stored[f"{key}__spearman_bootstrap"] = association_draws["spearman"]
            cell = {
                "cell": key,
                "candidate": candidate,
                "branch_half": half,
                "eligible_matrices": int(matrix_order.size),
                "eligible_states": len(candidate_cases),
                "arm_means": {
                    arm: float(
                        _matrix_means(q[:, index], ids, matrix_order).mean()
                    )
                    for arm, index in arm_index.items()
                },
                "target_tighten_minus_loosen": {
                    "estimate": float(matrix_target.mean()),
                    "bootstrap_ci95": _interval(target_boot),
                    "randomization_p_raw": p_raw,
                },
                "neutral_minus_noop": {
                    "estimate": float(matrix_neutral.mean()),
                    "bootstrap_ci90": ci90,
                    "tost_equivalent_margin_0_025": bool(
                        ci90[0] > -EQUIVALENCE_MARGIN
                        and ci90[1] < EQUIVALENCE_MARGIN
                    ),
                },
                "throughput_association": association,
            }
            cells.append(cell)
            for position, matrix_id in enumerate(matrix_order):
                matrix_rows.append(
                    {
                        "candidate": candidate,
                        "branch_half": half,
                        "matrix_id": int(matrix_id),
                        "target_tighten_minus_loosen": float(matrix_target[position]),
                        "neutral_minus_noop": float(matrix_neutral[position]),
                    }
                )
    adjusted = holm_adjust(raw_p)
    eligibility_pass = all(count >= 120 for count in eligible_counts.values())
    for cell, adjusted_p in zip(cells, adjusted, strict=True):
        target = cell["target_tighten_minus_loosen"]
        association = cell["throughput_association"]
        target["randomization_p_holm"] = float(adjusted_p)
        gates = {
            "minimum_eligible_matrices_per_candidate": eligibility_pass,
            "target_positive": target["estimate"] > 0.0,
            "target_bootstrap_lower_positive": target["bootstrap_ci95"][0] > 0.0,
            "target_holm_p_below_0_05": adjusted_p < 0.05,
            "throughput_slope_positive": association["state_centered_slope"] > 0.0,
            "throughput_slope_bootstrap_lower_positive": association[
                "slope_bootstrap_ci95"
            ][0]
            > 0.0,
            "neutral_tost_equivalent": cell["neutral_minus_noop"][
                "tost_equivalent_margin_0_025"
            ],
        }
        cell["statistical_gates"] = gates
        cell["statistical_cell_pass"] = bool(all(gates.values()))
    result = {
        "format": "codex-intervention-p3c-resilience-inference-v1",
        "endpoint": "RUN3_WITHIN_F8_FROM_IDENTICAL_NATURAL_POST_BREAK_DAUGHTER",
        "eligible_matrices_by_candidate": eligible_counts,
        "minimum_required_per_candidate": 120,
        "eligibility_gate_pass": eligibility_pass,
        "cells": cells,
        "all_statistical_cells_pass": bool(
            all(cell["statistical_cell_pass"] for cell in cells)
        ),
        "inference_unit": "whole catalytic matrix among prospectively eligible natural-break matrices",
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "randomization_repetitions": RANDOMIZATION_REPETITIONS,
    }
    return result, matrix_rows, stored


def add_integrity_gates(
    metrics: dict[str, Any], replay_exact: bool, readback_exact: bool, surgery_audit: bool
) -> dict[str, Any]:
    integrity = {
        "exact_replay": bool(replay_exact),
        "exact_artifact_readback": bool(readback_exact),
        "surgery_audit_pass": bool(surgery_audit),
    }
    metrics["integrity_gates"] = integrity
    metrics["primary_gate_pass"] = bool(
        metrics["primary"]["all_statistical_cells_pass"]
        and all(integrity.values())
    )
    metrics["resistance_gate_pass"] = bool(
        metrics["resistance"]["all_statistical_cells_pass"]
        and all(integrity.values())
    )
    return metrics


def _write_inference_arrays(
    path: Path, draws: dict[str, NDArray], stored: dict[str, NDArray]
) -> None:
    arrays: dict[str, NDArray] = {
        "bootstrap_indices": np.asarray(draws["bootstrap_indices"], dtype=np.int64),
        "randomization_signs": np.asarray(
            draws["randomization_signs"], dtype=np.float64
        ),
    }
    arrays.update({name: np.asarray(value) for name, value in stored.items()})
    np.savez_compressed(path, **arrays)


def _readback_inference(
    output: Path,
    cases: list[StateCase],
    expected_metrics: dict[str, Any],
    expected_matrix_rows: list[dict[str, Any]],
    stage: str,
) -> dict[str, Any]:
    with np.load(output / "branch_arrays.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    with np.load(output / "surgery_geometry_arrays.npz", allow_pickle=False) as archive:
        geometry = {name: archive[name] for name in archive.files}
    with np.load(output / "inference_arrays.npz", allow_pickle=False) as archive:
        draws = {
            "bootstrap_indices": archive["bootstrap_indices"],
            "randomization_signs": archive["randomization_signs"],
        }
    observed_metrics, observed_rows, _stored = compute_inference(
        cases, arrays, geometry, draws, stage
    )
    metrics_exact = _json_ready(observed_metrics) == _json_ready(expected_metrics)
    rows_exact = _json_ready(observed_rows) == _json_ready(expected_matrix_rows)
    if not metrics_exact or not rows_exact:
        raise ValueError("P3c written-artifact inference changed")
    return {
        "branch_arrays_reloaded": True,
        "geometry_arrays_reloaded": True,
        "inference_draws_reloaded": True,
        "primary_metrics_exact": metrics_exact,
        "matrix_effects_exact": rows_exact,
        "no_refitting_or_recalibration": True,
    }


def _technical_report(
    metrics: dict[str, Any], replay: dict[str, Any], audit: dict[str, Any], stage: str
) -> str:
    lines = [
        f"# P3c catalytic-throughput {stage}",
        "",
        f"Registered primary gate: **{metrics['primary_gate_pass']}**.",
        f"Registered F6 resistance gate: **{metrics['resistance_gate_pass']}**.",
        f"Exact replay: **{replay['state_edit_endpoint_and_process_digests_exact']}**.",
        "",
        "Positive target effects mean weakening the occupied catalytic web caused more break-and-renewal than strengthening it. Negative slopes mean greater starting catalytic throughput was associated with less break-and-renewal across the prospectively frozen arms.",
        "",
        "## Five-landmark primary cells",
        "",
        "| Cell | Loosen-tighten | 95% CI | Holm p | Throughput slope | 95% CI | Spearman | 95% CI | Neutral-NOOP 90% CI | Pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in metrics["primary"]["cells"]:
        target = cell["target_loosen_minus_tighten"]
        association = cell["throughput_association"]
        neutral = cell["neutral_minus_noop"]
        lines.append(
            f"| {cell['cell']} | {target['estimate']:+.6f} | {target['bootstrap_ci95']} | "
            f"{target['randomization_p_holm']:.6g} | {association['state_centered_slope']:+.6f} | "
            f"{association['slope_bootstrap_ci95']} | {association['mean_within_state_spearman']:+.6f} | "
            f"{association['spearman_bootstrap_ci95']} | {neutral['bootstrap_ci90']} | "
            f"{cell['statistical_cell_pass']} |"
        )
    lines.extend(
        [
            "",
            "## Integrity and boundary",
            "",
            f"- Throughput-neutral constraint passed: `{audit['neutral_exact_registered_throughput']}`.",
            f"- Exact norm and positivity audit passed: `{audit['all_audits_pass']}`.",
            "- The balanced-log random arm is diagnostic and is not required to be null.",
            "- Landmark 60 and resistance are reported separately and cannot rescue the five-landmark primary gate.",
            "- This simulated causal result cannot establish life, agency, biological memory, real chemistry, or strict-eight control.",
            "",
            "## Mandatory stop",
            "",
            f"The {stage} result is sealed. No next scientific stage was launched automatically.",
            "",
        ]
    )
    return "\n".join(lines)


def _lay_report(metrics: dict[str, Any], stage: str) -> str:
    if metrics["primary_gate_pass"]:
        outcome = (
            "The test passed every prewritten check: loosening the catalytic web "
            "increased break-and-renewal, strengthening it reduced the event, and "
            "a random change engineered to preserve total starting catalytic support "
            "behaved like no intervention."
        )
    else:
        outcome = (
            "The test did not pass every prewritten check. Any successful individual "
            "contrast remains reported, but P3c does not claim a confirmed common "
            "catalytic-throughput control axis from this stage."
        )
    return "\n".join(
        [
            f"# P3c {stage}: lay summary",
            "",
            "P3c asks whether the useful physical dial is the total catalytic support shared by the molecules currently in an assembly. It compares strengthening and weakening that web with a genuinely random rewiring that leaves the starting support unchanged.",
            "",
            outcome,
            "",
            "The earlier P3b result is unchanged. P3c is a new experiment designed to clarify why its old random comparison was not neutral.",
            "",
        ]
    )


def _campaign_status(work: Path, spec: P3CSpec, state: str, detail: str) -> None:
    work.mkdir(parents=True, exist_ok=True)
    base._atomic_json(
        work / "campaign_status.json",
        {
            "format": CHECKPOINT_FORMAT,
            "phase": spec.phase,
            "state": state,
            "detail": detail,
            "mandatory_stop_after_seal": True,
        },
    )


def _prepare_campaign(
    work: Path, output: Path, registration: dict[str, Any], spec: P3CSpec
) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    work.mkdir(parents=True, exist_ok=True)
    contract: dict[str, Any] = {
        "format": "codex-intervention-p3c-campaign-contract-v1",
        "registration_id": registration["registration_id"],
        "stage": spec.stage,
        "label": LABELS[spec.stage],
        "output": str(output),
        "matrices": spec.matrices,
        "branches": spec.branches,
        "landmarks": list(spec.landmarks),
        "arms": list(spec.arms),
        "horizon": spec.horizon,
        "source_hashes": _source_hashes(),
    }
    contract["campaign_id"] = _canonical_digest(contract)
    path = work / "campaign_contract.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != json.loads(
            json.dumps(_json_ready(contract))
        ):
            raise ValueError("P3c work directory belongs to another campaign")
    else:
        base._atomic_json(path, contract)
    _campaign_status(work, spec, "running", "campaign_initialized")


def verify_stage_result(
    directory: Path, registration_directory: Path, expected_stage: str
) -> dict[str, Any]:
    directory = directory.resolve()
    verify_checksums(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    registration = verify_registration(registration_directory)
    if (
        manifest.get("format") != RESULT_FORMAT
        or manifest.get("stage") != expected_stage
        or manifest.get("registration_id") != registration["registration_id"]
        or not manifest.get("exact_replay")
        or not manifest.get("complete_readback_exact")
    ):
        raise ValueError(f"invalid P3c {expected_stage} result")
    return manifest


def run_smoke(registration_directory: Path, output: Path, workers: int) -> None:
    registration_directory = registration_directory.resolve()
    output = output.resolve()
    registration = verify_registration(registration_directory)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    spec = P3CSpec(
        stage="smoke",
        role="non-scientific legality, I/O, and replay smoke",
        matrices=1,
        branches=2,
        landmarks=(20,),
        horizon=HORIZON,
        cohort_seed=SEED_DOMAINS["smoke_cohort"],
        balanced_selection_seed=SEED_DOMAINS["smoke_balanced_selection"],
        neutral_selection_seed=SEED_DOMAINS["smoke_neutral_selection"],
        future_seed=SEED_DOMAINS["smoke_future"],
        bootstrap_seed=SEED_DOMAINS["validation"],
        randomization_seed=SEED_DOMAINS["replay"],
    )
    experiment = _experiment(spec)
    with tempfile.TemporaryDirectory(prefix="codex-p3c-smoke-", dir=output.parent) as temporary:
        temporary_path = Path(temporary)
        with threadpool_limits(limits=1):
            cases = build_cohort(experiment, LABELS["smoke"], experiment.confirmation)
        generated = run_phase_batches(
            cases,
            experiment,
            spec,
            registration_directory / "frozen_full_predictor.npz",
            registration["registration_id"],
            temporary_path / "generate",
            workers,
            "generate",
        )
        replayed = run_phase_batches(
            cases,
            experiment,
            spec,
            registration_directory / "frozen_full_predictor.npz",
            registration["registration_id"],
            temporary_path / "replay",
            workers,
            "replay",
        )
        replay = base.replay_audit(generated, replayed)
        _geometry, _rows, audit = _geometry_arrays(cases, generated, spec)
        passed = bool(
            replay["state_edit_endpoint_and_process_digests_exact"]
            and audit["all_audits_pass"]
        )
        if not passed:
            raise AssertionError("P3c non-scientific smoke failed")
    with _atomic_destination(output) as destination:
        (destination / "manifest.json").write_text(
            json.dumps(
                {
                    "format": "codex-intervention-p3c-smoke-v1",
                    "registration_id": registration["registration_id"],
                    "scientific_result": False,
                    "scientific_matrix_count": 0,
                    "legality_io_neutral_geometry_and_replay_passed": True,
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
    print(f"P3c non-scientific smoke passed: {output}", flush=True)


def verify_smoke(directory: Path, registration_id: str) -> None:
    verify_checksums(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("format") != "codex-intervention-p3c-smoke-v1"
        or manifest.get("registration_id") != registration_id
        or not manifest.get("legality_io_neutral_geometry_and_replay_passed")
        or manifest.get("scientific_result")
    ):
        raise ValueError("invalid P3c non-scientific smoke")


def _append_ledger(
    output: Path, registration_id: str, metrics: dict[str, Any], stage: str
) -> None:
    path = REPOSITORY_ROOT / "INTERVENTION_RESULTS_LEDGER.md"
    current = path.read_text(encoding="utf-8").rstrip() + "\n"
    marker = f"<!-- sealed-p3c-{stage}-{registration_id} -->"
    if marker in current:
        return
    lines = [
        "",
        marker,
        f"## P3c catalytic-throughput {stage} sealed",
        "",
        f"- Registration: `{registration_id}`",
        f"- Result: `{output.relative_to(REPOSITORY_ROOT)}`",
        f"- Primary gate: **{metrics['primary_gate_pass']}**",
    ]
    if stage != "resilience":
        lines.append(
            f"- Registered resistance gate: **{metrics['resistance_gate_pass']}**"
        )
    else:
        lines.append(
            f"- Eligible matrices by candidate: `{metrics['eligible_matrices_by_candidate']}`"
        )
    lines.extend(
        [
            "- P3b remains unchanged; P3c is an additive clarification experiment.",
            "- Exact replay and written-artifact readback passed.",
            "- Mandatory stop observed; no next stage launched automatically.",
            "",
        ]
    )
    path.write_text(current + "\n".join(lines), encoding="utf-8")


def run_campaign(
    registration_directory: Path,
    output: Path,
    work: Path,
    workers: int,
    stage: str,
    pilot_result: Path | None = None,
) -> None:
    if workers < 1:
        raise ValueError("workers must be positive")
    registration_directory = registration_directory.resolve()
    output = output.resolve()
    work = work.resolve()
    registration = verify_registration(registration_directory)
    verify_smoke(DEFAULT_SMOKE, registration["registration_id"])
    spec = phase_spec(stage)
    if stage == "confirmation":
        if pilot_result is None:
            raise ValueError("confirmation requires the sealed pilot result")
        pilot_manifest = verify_stage_result(
            pilot_result, registration_directory, "pilot"
        )
        if not pilot_manifest["primary_gate_pass"]:
            raise ValueError("P3c pilot did not pass its frozen advancement rule")
        if sha256_file(registration_directory / "frozen_confirmation_protocol.json") != registration[
            "frozen_confirmation_protocol_sha256"
        ]:
            raise ValueError("frozen confirmation protocol changed after pilot")
    _prepare_campaign(work, output, registration, spec)
    experiment = _experiment(spec)
    expected_states = 2 * spec.matrices * len(spec.landmarks)
    print(
        f"[p3c {stage} 1/7] Building {spec.matrices} fresh matrices and {expected_states} states",
        flush=True,
    )
    _campaign_status(work, spec, "running", "building_natural_trajectories")
    with threadpool_limits(limits=1):
        cases = build_cohort(experiment, LABELS[stage], experiment.confirmation)
    if len(cases) != expected_states:
        raise AssertionError("P3c cohort state count changed")
    model_path = registration_directory / "frozen_full_predictor.npz"
    futures = len(cases) * len(spec.arms) * spec.branches
    print(f"[p3c {stage} 2/7] Shooting {futures:,} F12 futures", flush=True)
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
    print(f"[p3c {stage} 3/7] Replaying all {futures:,} futures", flush=True)
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
        raise AssertionError("P3c exact replay failed")
    print(f"[p3c {stage} 4/7] Auditing geometry and whole-matrix inference", flush=True)
    arrays = _outcome_arrays(cases, generated, spec)
    geometry, geometry_rows, surgery_audit = _geometry_arrays(cases, generated, spec)
    if not surgery_audit["all_audits_pass"]:
        raise AssertionError("P3c surgery audit failed")
    draws = generate_inference_draws(
        spec.matrices,
        BOOTSTRAP_REPETITIONS,
        RANDOMIZATION_REPETITIONS,
        np.random.default_rng(
            derive_seed(spec.bootstrap_seed, f"{LABELS[stage]}.bootstrap")
        ),
        np.random.default_rng(
            derive_seed(spec.randomization_seed, f"{LABELS[stage]}.randomization")
        ),
    )
    raw_metrics, matrix_rows, stored = compute_inference(
        cases, arrays, geometry, draws, stage
    )
    secondary = base._secondary_descriptives(cases, arrays, spec)
    print(f"[p3c {stage} 5/7] Writing and readback-checking artifacts", flush=True)
    with _atomic_destination(output) as destination:
        np.savez_compressed(destination / "branch_arrays.npz", **arrays)
        np.savez_compressed(destination / "surgery_geometry_arrays.npz", **geometry)
        _write_inference_arrays(destination / "inference_arrays.npz", draws, stored)
        base._write_branch_table(destination / "branches.csv.gz", cases, generated)
        base._write_state_artifacts(destination, cases, generated, arrays)
        base._write_selection_artifacts(destination, cases, generated, spec)
        geometry_rows.to_csv(destination / "surgery_geometry_audit.csv.gz", index=False)
        (destination / "surgery_audit_summary.json").write_text(
            json.dumps(_json_ready(surgery_audit), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        pd.DataFrame(matrix_rows).to_csv(destination / "matrix_effects.csv", index=False)
        readback = _readback_inference(
            destination, cases, raw_metrics, matrix_rows, stage
        )
        readback_exact = bool(
            readback["primary_metrics_exact"] and readback["matrix_effects_exact"]
        )
        metrics = add_integrity_gates(
            raw_metrics,
            replay_exact,
            readback_exact,
            surgery_audit["all_audits_pass"],
        )
        (destination / "primary_metrics.json").write_text(
            json.dumps(_json_ready(metrics), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (destination / "secondary_outcomes.json").write_text(
            json.dumps(_json_ready(secondary), indent=2, sort_keys=True) + "\n",
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
        (destination / "SCIENTIFIC_REPORT.md").write_text(
            _technical_report(metrics, replay, surgery_audit, stage),
            encoding="utf-8",
        )
        (destination / "LAY_SUMMARY.md").write_text(
            _lay_report(metrics, stage), encoding="utf-8"
        )
        claim_boundary = {
            "supported_claims": (
                [
                    "P3c stage passed its registered common catalytic-throughput gate under Codex contracts"
                ]
                if metrics["primary_gate_pass"]
                else []
            ),
            "failed_predictions": (
                []
                if metrics["primary_gate_pass"]
                else ["P3c registered common catalytic-throughput gate"]
            ),
            "deviations": [],
            "unresolved_questions": [
                "causal post-break resilience from an identical naturally broken state",
                "cross-clean-room compatibility of the local throughput-neutral control",
            ],
            "prohibited_interpretations": _protocol()["claim_boundary"]["prohibited"],
        }
        (destination / "claim_boundaries.json").write_text(
            json.dumps(claim_boundary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "format": RESULT_FORMAT,
            "stage": stage,
            "registration_id": registration["registration_id"],
            "matrices": spec.matrices,
            "candidates": list(CANDIDATES),
            "landmarks": list(spec.landmarks),
            "states": len(cases),
            "arms": list(spec.arms),
            "branches": spec.branches,
            "primary_futures": futures,
            "replay_futures": futures,
            "primary_gate_pass": metrics["primary_gate_pass"],
            "resistance_gate_pass": metrics["resistance_gate_pass"],
            "exact_replay": replay_exact,
            "complete_readback_exact": readback_exact,
            "surgery_audit_pass": surgery_audit["all_audits_pass"],
            "p3b_preserved": True,
            "no_matrix_replacement": True,
            "no_future_retries": True,
            "mandatory_stop_after_this_stage": True,
            "next_scientific_stage_launched": False,
            "runtime": _runtime_manifest(),
            "checkpoint_audit": {
                "work_directory": str(work),
                "campaign_contract_sha256": sha256_file(work / "campaign_contract.json"),
                "generate_contract_sha256": sha256_file(
                    work / "generate/checkpoint_contract.json"
                ),
                "replay_contract_sha256": sha256_file(
                    work / "replay/checkpoint_contract.json"
                ),
            },
        }
        (destination / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (destination / "CUMULATIVE_RESULTS_LEDGER.md").write_text(
            "\n".join(
                [
                    "# P3c result ledger snapshot",
                    "",
                    f"Stage: `{stage}`",
                    f"Registration: `{registration['registration_id']}`",
                    f"Primary gate: **{metrics['primary_gate_pass']}**",
                    f"Resistance gate: **{metrics['resistance_gate_pass']}**",
                    "Exact replay and readback: **True**",
                    "Next phase: not launched; mandatory review stop.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        write_checksums(destination)
    verify_checksums(output)
    _append_ledger(output, registration["registration_id"], metrics, stage)
    _campaign_status(work, spec, "sealed_complete", "mandatory_review_stop")
    print(f"[p3c {stage} 6/7] Result sealed: {output}", flush=True)
    print(f"[p3c {stage} 7/7] STOPPED; no next stage launched", flush=True)


def _verify_regenerated_confirmation_states(
    cases: list[StateCase], confirmation_result: Path
) -> None:
    with np.load(
        confirmation_result / "state_and_matrix_arrays.npz", allow_pickle=False
    ) as archive:
        ids = archive["state_ids"]
        compositions = archive["compositions"]
        candidates = archive["candidates"]
        matrix_ids = archive["matrix_ids"]
        landmarks = archive["landmarks"]
    if not np.array_equal(ids, np.asarray([case.state_id for case in cases])):
        raise ValueError("regenerated confirmation state IDs changed")
    if not np.array_equal(
        compositions, np.vstack([case.snapshot.composition for case in cases])
    ):
        raise ValueError("regenerated confirmation compositions changed")
    if not np.array_equal(candidates, np.asarray([case.candidate for case in cases])):
        raise ValueError("regenerated confirmation candidates changed")
    if not np.array_equal(matrix_ids, np.asarray([case.matrix_id for case in cases])):
        raise ValueError("regenerated confirmation matrix IDs changed")
    if not np.array_equal(landmarks, np.asarray([case.landmark for case in cases])):
        raise ValueError("regenerated confirmation landmarks changed")


def _resilience_secondary(
    cases: list[StateCase],
    anchors: list[NDArray[np.int64]],
    arrays: dict[str, NDArray],
) -> tuple[dict[str, Any], dict[str, NDArray]]:
    inherited = np.asarray(arrays["boundary_h"] > 0.9, dtype=bool)
    run5 = np.zeros(inherited.shape[:3], dtype=np.int8)
    for location in np.ndindex(run5.shape):
        run5[location] = int(_has_run(inherited[location], 5)[0])
    old_anchor_similarity = np.empty(run5.shape, dtype=np.float64)
    for state_index, anchor in enumerate(anchors):
        for arm_index in range(len(ARMS)):
            for branch in range(BRANCHES):
                old_anchor_similarity[state_index, arm_index, branch] = cosine_similarity(
                    anchor, arrays["final_composition"][state_index, arm_index, branch]
                )
    cells: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        mask = np.asarray([case.candidate == candidate for case in cases], dtype=bool)
        for half, branch_slice in (
            ("A", slice(0, BRANCHES // 2)),
            ("B", slice(BRANCHES // 2, BRANCHES)),
        ):
            for arm_index, arm in enumerate(ARMS):
                renewal = arrays["renewal_certification_time"][
                    mask, arm_index, branch_slice
                ]
                cells.append(
                    {
                        "candidate": candidate,
                        "branch_half": half,
                        "arm": arm,
                        "mean_run3": float(
                            arrays["targets"][mask, arm_index, branch_slice].mean()
                        ),
                        "mean_run5": float(run5[mask, arm_index, branch_slice].mean()),
                        "mean_inherited_boundary_count": float(
                            arrays["inherited_boundary_count"][
                                mask, arm_index, branch_slice
                            ].mean()
                        ),
                        "mean_old_anchor_similarity_at_horizon": float(
                            old_anchor_similarity[mask, arm_index, branch_slice].mean()
                        ),
                        "mean_survival": float(
                            arrays["completed_horizon"][
                                mask, arm_index, branch_slice
                            ].mean()
                        ),
                        "mean_final_entropy": float(
                            arrays["final_entropy"][mask, arm_index, branch_slice].mean()
                        ),
                        "mean_final_occupied_types": float(
                            arrays["final_occupied_types"][
                                mask, arm_index, branch_slice
                            ].mean()
                        ),
                        "mean_time_to_run3_given_run3": (
                            float(renewal[renewal >= 0].mean())
                            if np.any(renewal >= 0)
                            else None
                        ),
                    }
                )
    return {
        "format": "codex-intervention-p3c-resilience-secondary-v1",
        "cells": cells,
        "conditional_times_do_not_impute_nonrenewal": True,
    }, {"run5": run5, "old_anchor_similarity": old_anchor_similarity}


def run_resilience(
    registration_directory: Path,
    confirmation_result: Path,
    output: Path,
    work: Path,
    workers: int,
) -> None:
    if workers < 1:
        raise ValueError("workers must be positive")
    registration_directory = registration_directory.resolve()
    confirmation_result = confirmation_result.resolve()
    output = output.resolve()
    work = work.resolve()
    registration = verify_registration(registration_directory)
    verify_smoke(DEFAULT_SMOKE, registration["registration_id"])
    confirmation = verify_stage_result(
        confirmation_result, registration_directory, "confirmation"
    )
    if not confirmation["primary_gate_pass"]:
        raise ValueError("resilience requires a passing P3c confirmation")
    spec = resilience_spec()
    _prepare_campaign(work, output, registration, spec)
    confirmation_spec = phase_spec("confirmation")
    confirmation_experiment = _experiment(confirmation_spec)
    print("[p3c resilience 1/8] Regenerating and verifying confirmation states", flush=True)
    with threadpool_limits(limits=1):
        all_cases = build_cohort(
            confirmation_experiment,
            LABELS["confirmation"],
            confirmation_experiment.confirmation,
        )
    _verify_regenerated_confirmation_states(all_cases, confirmation_result)
    natural_cases = [case for case in all_cases if case.landmark in PRIMARY_LANDMARKS]
    if len(natural_cases) != 2 * CONFIRMATION_MATRICES * len(PRIMARY_LANDMARKS):
        raise AssertionError("resilience natural source cohort is incomplete")
    print("[p3c resilience 2/8] Acquiring one fixed natural break per source state", flush=True)
    experiment = _experiment(spec)
    broken_cases, anchors, acquisition_rows = acquire_resilience_cohort(
        natural_cases, experiment
    )
    replay_cases, replay_anchors, replay_rows = acquire_resilience_cohort(
        natural_cases, experiment
    )
    acquisition_exact = bool(
        _json_ready(acquisition_rows.to_dict("records"))
        == _json_ready(replay_rows.to_dict("records"))
        and [base._snapshot_digest(case) for case in broken_cases]
        == [base._snapshot_digest(case) for case in replay_cases]
        and all(
            np.array_equal(left, right)
            for left, right in zip(anchors, replay_anchors, strict=True)
        )
    )
    if not acquisition_exact:
        raise AssertionError("natural-break acquisition replay failed")
    eligible_counts = {
        candidate: int(
            len(
                {
                    case.matrix_id
                    for case in broken_cases
                    if case.candidate == candidate
                }
            )
        )
        for candidate in CANDIDATES
    }
    if not all(count >= 120 for count in eligible_counts.values()):
        inconclusive_metrics = {
            "format": "codex-intervention-p3c-resilience-inconclusive-v1",
            "primary_gate_pass": False,
            "classification": "inconclusive_insufficient_natural_break_matrices",
            "eligible_matrices_by_candidate": eligible_counts,
            "minimum_required_per_candidate": 120,
            "natural_acquisition_replay_exact": acquisition_exact,
        }
        with _atomic_destination(output) as destination:
            acquisition_rows.to_csv(destination / "natural_break_acquisition.csv", index=False)
            manifest = {
                "format": RESULT_FORMAT,
                "stage": "resilience",
                "registration_id": registration["registration_id"],
                "classification": "inconclusive_insufficient_natural_break_matrices",
                "eligible_matrices_by_candidate": eligible_counts,
                "minimum_required_per_candidate": 120,
                "natural_acquisition_replay_exact": acquisition_exact,
                "primary_futures": 0,
                "replay_futures": 0,
                "primary_gate_pass": False,
                "exact_replay": True,
                "complete_readback_exact": True,
                "mandatory_stop_after_this_stage": True,
            }
            (destination / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (destination / "SCIENTIFIC_REPORT.md").write_text(
                "# P3c shared-break resilience\n\nThe registered minimum of 120 naturally eligible matrices per candidate was not reached. The result is inconclusive and no intervention future was launched.\n",
                encoding="utf-8",
            )
            (destination / "LAY_SUMMARY.md").write_text(
                "# Lay summary\n\nToo few independent simulated catalytic matrices naturally produced a usable break under the frozen acquisition rule. We therefore stopped without testing any intervention and call this question inconclusive.\n",
                encoding="utf-8",
            )
            (destination / "primary_metrics.json").write_text(
                json.dumps(inconclusive_metrics, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            (destination / "replay_audit.json").write_text(
                json.dumps(
                    {
                        "natural_acquisition_replay_exact": acquisition_exact,
                        "intervention_futures_launched": False,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            (destination / "readback_audit.json").write_text(
                json.dumps(
                    {
                        "acquisition_table_written": True,
                        "no_branch_or_inference_artifact_expected": True,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            (destination / "claim_boundaries.json").write_text(
                json.dumps(
                    {
                        "supported_claims": [],
                        "failed_predictions": [],
                        "inconclusive_questions": [
                            "causal resilience after an identical natural break"
                        ],
                        "prohibited_interpretations": _protocol()["claim_boundary"][
                            "prohibited"
                        ],
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            (destination / "CUMULATIVE_RESULTS_LEDGER.md").write_text(
                "# P3c resilience result\n\nClassification: **inconclusive** because the registered natural-break eligibility minimum was not met. No intervention future was launched.\n",
                encoding="utf-8",
            )
            write_checksums(destination)
        verify_checksums(output)
        _append_ledger(
            output,
            registration["registration_id"],
            inconclusive_metrics,
            "resilience",
        )
        _campaign_status(work, spec, "sealed_inconclusive", "insufficient_natural_break_matrices")
        print("P3c resilience sealed inconclusive; STOPPED", flush=True)
        return
    futures = len(broken_cases) * len(ARMS) * BRANCHES
    print(
        f"[p3c resilience 3/8] Shooting {futures:,} F8 futures from shared broken daughters",
        flush=True,
    )
    generated = run_phase_batches(
        broken_cases,
        experiment,
        spec,
        registration_directory / "frozen_full_predictor.npz",
        registration["registration_id"],
        work / "generate",
        workers,
        "generate",
    )
    print(f"[p3c resilience 4/8] Replaying all {futures:,} F8 futures", flush=True)
    replayed = run_phase_batches(
        broken_cases,
        experiment,
        spec,
        registration_directory / "frozen_full_predictor.npz",
        registration["registration_id"],
        work / "replay",
        workers,
        "replay",
    )
    replay = base.replay_audit(generated, replayed)
    replay_exact = bool(
        acquisition_exact
        and replay["state_edit_endpoint_and_process_digests_exact"]
    )
    if not replay_exact:
        raise AssertionError("P3c resilience replay failed")
    print("[p3c resilience 5/8] Computing whole-matrix inference", flush=True)
    arrays = _outcome_arrays(broken_cases, generated, spec)
    geometry, geometry_rows, surgery_audit = _geometry_arrays(
        broken_cases, generated, spec
    )
    metrics, matrix_rows, stored = _resilience_inference(
        broken_cases, arrays, geometry
    )
    secondary, secondary_arrays = _resilience_secondary(
        broken_cases, anchors, arrays
    )
    print("[p3c resilience 6/8] Writing and readback-checking artifacts", flush=True)
    with _atomic_destination(output) as destination:
        acquisition_rows.to_csv(destination / "natural_break_acquisition.csv", index=False)
        np.savez_compressed(destination / "branch_arrays.npz", **arrays)
        np.savez_compressed(destination / "surgery_geometry_arrays.npz", **geometry)
        np.savez_compressed(destination / "secondary_arrays.npz", **secondary_arrays)
        np.savez_compressed(destination / "inference_arrays.npz", **stored)
        base._write_branch_table(destination / "branches.csv.gz", broken_cases, generated)
        base._write_state_artifacts(destination, broken_cases, generated, arrays)
        base._write_selection_artifacts(destination, broken_cases, generated, spec)
        geometry_rows.to_csv(destination / "surgery_geometry_audit.csv.gz", index=False)
        pd.DataFrame(matrix_rows).to_csv(destination / "matrix_effects.csv", index=False)
        (destination / "secondary_outcomes.json").write_text(
            json.dumps(_json_ready(secondary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with np.load(destination / "branch_arrays.npz", allow_pickle=False) as archive:
            read_arrays = {name: archive[name] for name in archive.files}
        with np.load(
            destination / "surgery_geometry_arrays.npz", allow_pickle=False
        ) as archive:
            read_geometry = {name: archive[name] for name in archive.files}
        read_metrics, read_rows, _ = _resilience_inference(
            broken_cases, read_arrays, read_geometry
        )
        readback_exact = bool(
            _json_ready(read_metrics) == _json_ready(metrics)
            and _json_ready(read_rows) == _json_ready(matrix_rows)
        )
        if not readback_exact:
            raise AssertionError("P3c resilience artifact readback failed")
        integrity = {
            "natural_acquisition_replay_exact": acquisition_exact,
            "future_replay_exact": replay_exact,
            "artifact_readback_exact": readback_exact,
            "surgery_audit_pass": surgery_audit["all_audits_pass"],
        }
        metrics["integrity_gates"] = integrity
        metrics["primary_gate_pass"] = bool(
            metrics["all_statistical_cells_pass"] and all(integrity.values())
        )
        (destination / "primary_metrics.json").write_text(
            json.dumps(_json_ready(metrics), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (destination / "surgery_audit_summary.json").write_text(
            json.dumps(_json_ready(surgery_audit), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (destination / "replay_audit.json").write_text(
            json.dumps(_json_ready(replay), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (destination / "readback_audit.json").write_text(
            json.dumps(
                {
                    "natural_acquisition_replay_exact": acquisition_exact,
                    "primary_metrics_exact": readback_exact,
                    "matrix_effects_exact": readback_exact,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (destination / "SCIENTIFIC_REPORT.md").write_text(
            "\n".join(
                [
                    "# P3c shared-natural-break resilience",
                    "",
                    f"Registered resilience gate: **{metrics['primary_gate_pass']}**.",
                    f"Eligible matrices: `{eligible_counts}`.",
                    "",
                    "Every arm began from the identical selected daughter immediately after one naturally occurring break. Positive TIGHTEN-LOOSEN effects therefore measure causal recovery rather than treatment-created selection into breaking.",
                    "",
                    "This cannot establish autonomous repair, biological memory, or life.",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        (destination / "LAY_SUMMARY.md").write_text(
            "# Lay summary\n\nThis test waits for nature to produce the same break first, copies that exact broken state into every arm, and only then changes the catalytic web. It asks whether stronger catalytic support helps the assembly rebuild a short inherited run.\n",
            encoding="utf-8",
        )
        manifest = {
            "format": RESULT_FORMAT,
            "stage": "resilience",
            "registration_id": registration["registration_id"],
            "eligible_matrices_by_candidate": eligible_counts,
            "eligible_states": len(broken_cases),
            "primary_futures": futures,
            "replay_futures": futures,
            "primary_gate_pass": metrics["primary_gate_pass"],
            "exact_replay": replay_exact,
            "complete_readback_exact": readback_exact,
            "natural_acquisition_replay_exact": acquisition_exact,
            "surgery_audit_pass": surgery_audit["all_audits_pass"],
            "no_acquisition_retries": True,
            "no_matrix_or_state_replacement": True,
            "mandatory_stop_after_this_stage": True,
            "next_scientific_stage_launched": False,
            "runtime": _runtime_manifest(),
        }
        (destination / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_checksums(destination)
    verify_checksums(output)
    _append_ledger(output, registration["registration_id"], metrics, "resilience")
    _campaign_status(work, spec, "sealed_complete", "mandatory_review_stop")
    print("[p3c resilience 7/8] Result sealed", flush=True)
    print("[p3c resilience 8/8] STOPPED; no feedback stage launched", flush=True)


def read_status(work: Path) -> dict[str, Any]:
    return base.read_status(work)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--output", type=Path, default=DEFAULT_VALIDATION)
    archive = commands.add_parser("archive-fable-response")
    archive.add_argument("--input", type=Path, required=True)
    archive.add_argument("--output", type=Path, default=DEFAULT_FABLE_ARCHIVE)
    register = commands.add_parser("register")
    register.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    register.add_argument("--fable-archive", type=Path, default=DEFAULT_FABLE_ARCHIVE)
    register.add_argument("--output", type=Path, default=DEFAULT_REGISTRATION)
    verify = commands.add_parser("verify")
    verify.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    smoke = commands.add_parser("smoke")
    smoke.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    smoke.add_argument("--output", type=Path, default=DEFAULT_SMOKE)
    smoke.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 4))
    pilot = commands.add_parser("run-pilot")
    pilot.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    pilot.add_argument("--output", type=Path, default=DEFAULT_PILOT_OUTPUT)
    pilot.add_argument("--work-dir", type=Path, default=DEFAULT_PILOT_WORK)
    pilot.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 14))
    confirmation = commands.add_parser("run-confirmation")
    confirmation.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    confirmation.add_argument("--pilot-result", type=Path, default=DEFAULT_PILOT_OUTPUT)
    confirmation.add_argument("--output", type=Path, default=DEFAULT_CONFIRMATION_OUTPUT)
    confirmation.add_argument("--work-dir", type=Path, default=DEFAULT_CONFIRMATION_WORK)
    confirmation.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 14))
    resilience = commands.add_parser("run-resilience")
    resilience.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    resilience.add_argument(
        "--confirmation-result", type=Path, default=DEFAULT_CONFIRMATION_OUTPUT
    )
    resilience.add_argument("--output", type=Path, default=DEFAULT_RESILIENCE_OUTPUT)
    resilience.add_argument("--work-dir", type=Path, default=DEFAULT_RESILIENCE_WORK)
    resilience.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 14))
    status = commands.add_parser("status")
    status.add_argument("--work-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    if arguments.command == "validate":
        run_validation(arguments.output)
    elif arguments.command == "archive-fable-response":
        archive_fable_response(arguments.input, arguments.output)
    elif arguments.command == "register":
        register_program(arguments.validation, arguments.fable_archive, arguments.output)
    elif arguments.command == "verify":
        print(json.dumps(verify_registration(arguments.registration), indent=2, sort_keys=True))
    elif arguments.command == "smoke":
        run_smoke(arguments.registration, arguments.output, arguments.workers)
    elif arguments.command == "run-pilot":
        run_campaign(
            arguments.registration,
            arguments.output,
            arguments.work_dir,
            arguments.workers,
            "pilot",
        )
    elif arguments.command == "run-confirmation":
        run_campaign(
            arguments.registration,
            arguments.output,
            arguments.work_dir,
            arguments.workers,
            "confirmation",
            arguments.pilot_result,
        )
    elif arguments.command == "run-resilience":
        run_resilience(
            arguments.registration,
            arguments.confirmation_result,
            arguments.output,
            arguments.work_dir,
            arguments.workers,
        )
    elif arguments.command == "status":
        print(json.dumps(read_status(arguments.work_dir), indent=2, sort_keys=True))
    else:  # pragma: no cover
        raise AssertionError(arguments.command)


if __name__ == "__main__":
    main()
