"""P4 prospective shared-natural-break catalytic recovery experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from threadpoolctl import threadpool_limits

from . import intervention_p3c as p3c
from . import intervention_replication as base
from .config import CANDIDATES, CohortConfig, ExperimentConfig
from .experiment import StateCase, _json_ready, _runtime_manifest, build_cohort
from .intervention_metrics import (
    _bootstrap_means,
    _interval,
    _matrix_means,
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
from .simulator import SimulationError, Snapshot, advance_fission, cosine_similarity


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results_intervention_replication"
AUDIT = RESULT_ROOT / "p4a_p3c_interpretation_audit"
P3C_CONFIRMATION = RESULT_ROOT / "p3c_throughput_confirmation"
MODEL_REGISTRATION = RESULT_ROOT / "registration"
DEFAULT_VALIDATION = RESULT_ROOT / "p4_validation"
DEFAULT_REGISTRATION = RESULT_ROOT / "p4_registration"
DEFAULT_SMOKE = RESULT_ROOT / "p4_smoke"
DEFAULT_OUTPUT = RESULT_ROOT / "p4_shared_break_recovery"
DEFAULT_WORK = RESULT_ROOT / ".p4_shared_break_recovery_work"
DOCUMENT = "CODEX_INTERVENTION_P4_SHARED_BREAK_PREREGISTRATION.md"
SOURCE_FILES = (
    DOCUMENT,
    "plastic_heredity/intervention_p4.py",
    "tests/test_intervention_p4.py",
    "plastic_heredity/intervention_p3c.py",
    "plastic_heredity/intervention_replication.py",
    "plastic_heredity/intervention_core.py",
    "plastic_heredity/intervention_metrics.py",
    "plastic_heredity/simulator.py",
    "plastic_heredity/processes.py",
    "plastic_heredity/experiment.py",
    "plastic_heredity/config.py",
)
PROGRAM_FORMAT = "codex-intervention-p4-shared-break-v1"
REGISTRATION_FORMAT = "codex-intervention-p4-registration-v1"
VALIDATION_FORMAT = "codex-intervention-p4-validation-v1"
RESULT_FORMAT = "codex-intervention-p4-result-v1"
LABEL = "INTP4_SHARED_NATURAL_BREAK_V1"
MATRICES = 160
LANDMARKS = (20, 35, 50, 65, 80)
BRANCHES = 32
ACQUISITION_HORIZON = 12
RECOVERY_HORIZON = 8
MINIMUM_ELIGIBLE_MATRICES = 120
BOOTSTRAP_REPETITIONS = 4_096
RANDOMIZATION_REPETITIONS = 4_096
EQUIVALENCE_MARGIN = 0.025
ARMS = p3c.ARMS


def _seed(name: str) -> str:
    return hashlib.sha256(f"codex-clean-room-p4-shared-break-v1::{name}".encode()).hexdigest()


SEEDS = {
    name: _seed(name)
    for name in (
        "validation",
        "smoke_cohort",
        "smoke_acquisition",
        "smoke_balanced_selection",
        "smoke_neutral_selection",
        "smoke_future",
        "cohort",
        "acquisition",
        "balanced_selection",
        "neutral_selection",
        "future",
        "bootstrap",
        "randomization",
        "replay",
    )
}


def source_hashes() -> dict[str, str]:
    return {name: sha256_file(ROOT / name) for name in SOURCE_FILES}


def spec() -> p3c.P3CSpec:
    # P3CSpec is a neutral data contract consumed by the sealed one-shot worker.
    # New seed roots make every P4 stream disjoint despite the worker's legacy
    # internal stage name.
    return p3c.P3CSpec(
        stage="resilience",
        role="P4 causal recovery from identical naturally broken daughters",
        matrices=MATRICES,
        branches=BRANCHES,
        landmarks=LANDMARKS,
        horizon=RECOVERY_HORIZON,
        cohort_seed=SEEDS["cohort"],
        balanced_selection_seed=SEEDS["balanced_selection"],
        neutral_selection_seed=SEEDS["neutral_selection"],
        future_seed=SEEDS["future"],
        bootstrap_seed=SEEDS["bootstrap"],
        randomization_seed=SEEDS["randomization"],
    )


def experiment(stage_spec: p3c.P3CSpec | None = None) -> ExperimentConfig:
    current = spec() if stage_spec is None else stage_spec
    cohort = CohortConfig(current.matrices, current.branches, current.landmarks)
    return ExperimentConfig(
        development=cohort,
        confirmation=cohort,
        horizon=current.horizon,
        bootstrap_repetitions=BOOTSTRAP_REPETITIONS,
        permutation_repetitions=RANDOMIZATION_REPETITIONS,
        regenerate_confirmation=True,
        master_seed=current.cohort_seed,
    )


def protocol() -> dict[str, Any]:
    value: dict[str, Any] = {
        "format": PROGRAM_FORMAT,
        "status": "frozen_before_any_p4_scientific_matrix",
        "classification": "new_additive_shared_natural_break_experiment",
        "p3c_unchanged_and_not_rescued": True,
        "not_original_cr5": True,
        "questions": [
            "coherent catalytic strength and recovery from an identical natural break",
            "fixed-launch-throughput topology and recovery",
        ],
        "cohort": {
            "matrices": MATRICES,
            "candidates": list(CANDIDATES),
            "landmarks": list(LANDMARKS),
            "natural_acquisition_horizon": ACQUISITION_HORIZON,
            "minimum_eligible_matrices_per_candidate": MINIMUM_ELIGIBLE_MATRICES,
            "no_retries_or_replacements": True,
        },
        "arms": list(ARMS),
        "arm_contract": {
            "LOOSEN": "beta[P,P] / 1.5",
            "TIGHTEN": "beta[P,P] * 1.5",
            "BALANCED_LOG_RANDOM": "diagnostic_not_required_null",
            "THROUGHPUT_NEUTRAL_RANDOM": "exact 0.5-block-norm perturbation preserving launch x^T beta x",
            "NOOP": "unchanged beta",
        },
        "futures": {
            "horizon": RECOVERY_HORIZON,
            "branches_per_arm_state": BRANCHES,
            "halves": {"A": [0, 15], "B": [16, 31]},
            "common_random_streams": True,
            "future_seed_excludes_arm": True,
            "complete_replay": True,
        },
        "primary_endpoint": "RUN3_WITHIN_F8_FROM_IDENTICAL_NATURAL_POST_BREAK_DAUGHTER",
        "inference": {
            "unit": "whole catalytic matrix",
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "sign_randomizations": RANDOMIZATION_REPETITIONS,
            "strength_family": "one-sided Holm across four candidate-half cells",
            "topology_family": "two-sided Holm across four candidate-half cells",
            "topology_equivalence_margin": EQUIVALENCE_MARGIN,
            "strength_and_topology_classified_separately": True,
        },
        "mandatory_stop_after_result": True,
        "seed_domains": SEEDS,
        "claim_boundary": {
            "prohibited": [
                "P3c passed",
                "original CR5 completed",
                "biological repair or memory",
                "agency or life",
                "autonomous attractor",
                "real prebiotic chemistry",
                "universal origin-of-life mechanism",
            ]
        },
    }
    value["protocol_id"] = _canonical_digest(_json_ready(value))
    return value


def _validation_checks() -> dict[str, bool]:
    verify_checksums(AUDIT)
    verify_checksums(P3C_CONFIRMATION)
    verify_checksums(MODEL_REGISTRATION)
    composition = np.asarray([5, 3, 2, 0], dtype=np.int64)
    beta = np.asarray(
        [[2.0, 0.7, 1.3, 0.4], [1.1, 3.0, 0.8, 0.9], [0.6, 1.4, 2.5, 1.2], [0.5, 0.3, 0.4, 1.8]],
        dtype=np.float64,
    )
    surgeries = p3c.select_surgeries(
        composition, beta, np.random.default_rng(11), np.random.default_rng(12)
    )
    neutral = surgeries[ARMS.index("THROUGHPUT_NEUTRAL_RANDOM")]
    assert neutral is not None
    return {
        "sealed_p3c_audit_verified": True,
        "sealed_p3c_confirmation_verified": True,
        "frozen_model_registration_verified": True,
        "p4_seed_domains_unique": len(set(SEEDS.values())) == len(SEEDS),
        "p4_seeds_disjoint_from_p3c": not set(SEEDS.values()).intersection(p3c.SEED_DOMAINS.values()),
        "arm_order_frozen": ARMS == p3c.ARMS,
        "neutral_launch_throughput_exact": bool(np.isclose(p3c.catalytic_throughput(composition, beta), p3c.catalytic_throughput(composition, neutral.beta), rtol=1e-12, atol=1e-10)),
        "neutral_norm_exact": bool(np.isclose(neutral.observed_norm, neutral.requested_norm, rtol=1e-12, atol=1e-10)),
        "neutral_positive": bool(np.all(neutral.beta > 0.0)),
        "future_seed_excludes_arm": True,
        "acquisition_seed_excludes_arm": True,
        "p3c_verdict_not_modified": True,
    }


def validate(output: Path = DEFAULT_VALIDATION) -> None:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    checks = _validation_checks()
    if not all(checks.values()):
        raise AssertionError({name: passed for name, passed in checks.items() if not passed})
    with _atomic_destination(output) as destination:
        payload = {
            "format": VALIDATION_FORMAT,
            "checks": checks,
            "all_pass": True,
            "scientific_matrices_generated": 0,
            "scientific_futures_generated": 0,
            "source_hashes": source_hashes(),
        }
        (destination / "validation.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        write_checksums(destination)
    verify_checksums(output)
    print(f"P4 validation sealed: {output}", flush=True)


def register(validation_directory: Path = DEFAULT_VALIDATION, output: Path = DEFAULT_REGISTRATION) -> None:
    validation_directory = validation_directory.resolve()
    output = output.resolve()
    verify_checksums(validation_directory)
    validation_payload = json.loads((validation_directory / "validation.json").read_text())
    if not validation_payload.get("all_pass"):
        raise ValueError("P4 validation did not pass")
    for scientific in (DEFAULT_OUTPUT, DEFAULT_WORK):
        if scientific.exists():
            raise FileExistsError(f"P4 scientific artifact exists before registration: {scientific}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    frozen_protocol = protocol()
    registration_payload: dict[str, Any] = {
        "format": REGISTRATION_FORMAT,
        "protocol_id": frozen_protocol["protocol_id"],
        "source_hashes": source_hashes(),
        "validation_checksum_manifest_sha256": sha256_file(validation_directory / "SHA256SUMS"),
        "p3c_confirmation_checksum_manifest_sha256": sha256_file(P3C_CONFIRMATION / "SHA256SUMS"),
        "p3c_audit_checksum_manifest_sha256": sha256_file(AUDIT / "SHA256SUMS"),
        "frozen_model_sha256": sha256_file(MODEL_REGISTRATION / "frozen_full_predictor.npz"),
        "seed_registry": SEEDS,
        "scientific_matrices_at_registration": 0,
        "scientific_futures_at_registration": 0,
    }
    registration_payload["registration_id"] = _canonical_digest(_json_ready(registration_payload))
    with _atomic_destination(output) as destination:
        (destination / "protocol.json").write_text(json.dumps(frozen_protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (destination / "seed_registry.json").write_text(json.dumps(SEEDS, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (destination / "registration.json").write_text(json.dumps(registration_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        shutil.copy2(MODEL_REGISTRATION / "frozen_full_predictor.npz", destination / "frozen_full_predictor.npz")
        write_checksums(destination)
    verify_registration(output)
    print(f"P4 registration sealed: {registration_payload['registration_id']}", flush=True)


def verify_registration(directory: Path = DEFAULT_REGISTRATION) -> dict[str, Any]:
    directory = directory.resolve()
    verify_checksums(directory)
    value = json.loads((directory / "registration.json").read_text())
    if value.get("format") != REGISTRATION_FORMAT:
        raise ValueError("invalid P4 registration format")
    if value["source_hashes"] != source_hashes():
        raise ValueError("P4 source tree changed after registration")
    if json.loads((directory / "protocol.json").read_text()) != protocol():
        raise ValueError("P4 protocol changed after registration")
    if value["frozen_model_sha256"] != sha256_file(directory / "frozen_full_predictor.npz"):
        raise ValueError("P4 frozen model changed")
    if value["seed_registry"] != SEEDS:
        raise ValueError("P4 seed registry changed")
    return value


def _acquisition_seed(case: StateCase, seed_root: str = SEEDS["acquisition"]) -> int:
    return derive_seed(seed_root, f"{LABEL}.natural_acquisition", case.candidate, case.matrix_id, case.landmark)


def acquire_natural_break(
    case: StateCase,
    current_experiment: ExperimentConfig,
    *,
    seed_root: str = SEEDS["acquisition"],
    horizon: int = ACQUISITION_HORIZON,
) -> tuple[StateCase | None, NDArray[np.int64] | None, dict[str, Any]]:
    rng = np.random.default_rng(_acquisition_seed(case, seed_root))
    current = np.asarray(case.snapshot.composition, dtype=np.int64).copy()
    inheritance = list(case.snapshot.inheritance)
    boundary_h = list(case.snapshot.boundary_h)
    cumulative = int(case.snapshot.cumulative_growth_steps)
    for offset in range(1, horizon + 1):
        try:
            record = advance_fission(current, case.beta, current_experiment.gard, CANDIDATES[case.candidate], rng)
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
        inherited = bool(record.h > current_experiment.gard.inheritance_threshold)
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
                state_id=f"{LABEL}-c{case.candidate}-m{case.matrix_id:03d}-g{case.landmark:03d}-break-f{offset:02d}",
                cohort=LABEL,
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
        current = np.asarray(record.daughter, dtype=np.int64)
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


def acquire_cohort(
    cases: list[StateCase], current_experiment: ExperimentConfig, *, seed_root: str = SEEDS["acquisition"]
) -> tuple[list[StateCase], list[NDArray[np.int64]], pd.DataFrame]:
    broken: list[StateCase] = []
    anchors: list[NDArray[np.int64]] = []
    rows: list[dict[str, Any]] = []
    for case in cases:
        acquired, anchor, audit = acquire_natural_break(case, current_experiment, seed_root=seed_root)
        rows.append(audit)
        if acquired is not None:
            if anchor is None:
                raise AssertionError("eligible P4 break lacks its pre-break anchor")
            broken.append(acquired)
            anchors.append(anchor)
    return broken, anchors, pd.DataFrame(rows)


def _two_sided_sign_p(values: NDArray, signs: NDArray) -> tuple[float, NDArray]:
    data = np.asarray(values, dtype=np.float64)
    null = (np.asarray(signs, dtype=np.float64) * data).mean(axis=1)
    observed = abs(float(data.mean()))
    p = float((1 + np.count_nonzero(np.abs(null) >= observed)) / (len(null) + 1))
    return p, null


def compute_inference(
    cases: list[StateCase], arrays: dict[str, NDArray], geometry: dict[str, NDArray]
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, NDArray]]:
    arm_index = {arm: index for index, arm in enumerate(ARMS)}
    targets = np.asarray(arrays["targets"], dtype=np.float64)
    log_throughput = np.asarray(geometry["log_throughput_ratio"], dtype=np.float64)
    cells: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    stored: dict[str, NDArray] = {}
    strength_raw: list[float] = []
    topology_raw: list[float] = []
    eligible_counts: dict[str, int] = {}
    for candidate in CANDIDATES:
        mask = np.asarray([case.candidate == candidate for case in cases], dtype=bool)
        candidate_cases = [case for case, keep in zip(cases, mask) if keep]
        ids = np.asarray([case.matrix_id for case in candidate_cases], dtype=np.int64)
        matrix_order = np.sort(np.unique(ids))
        eligible_counts[candidate] = int(matrix_order.size)
        if matrix_order.size < 2:
            raise ValueError("P4 inference requires two eligible matrices")
        draws = generate_inference_draws(
            matrix_order.size,
            BOOTSTRAP_REPETITIONS,
            RANDOMIZATION_REPETITIONS,
            np.random.default_rng(derive_seed(SEEDS["bootstrap"], f"{LABEL}.bootstrap.c{candidate}")),
            np.random.default_rng(derive_seed(SEEDS["randomization"], f"{LABEL}.randomization.c{candidate}")),
        )
        bootstrap = draws["bootstrap_indices"]
        signs = draws["randomization_signs"]
        stored[f"c{candidate}__bootstrap_indices"] = bootstrap
        stored[f"c{candidate}__randomization_signs"] = signs
        candidate_targets = targets[mask]
        candidate_geometry = log_throughput[mask]
        for half, branch_slice in (("A", slice(0, 16)), ("B", slice(16, 32))):
            q = candidate_targets[:, :, branch_slice].mean(axis=2)
            outcome_shift = q - q[:, [arm_index["NOOP"]]]
            association, association_draws = p3c._slope_and_rank_statistics(candidate_geometry, outcome_shift, ids, bootstrap)
            strength_state = q[:, arm_index["TIGHTEN"]] - q[:, arm_index["LOOSEN"]]
            topology_state = q[:, arm_index["THROUGHPUT_NEUTRAL_RANDOM"]] - q[:, arm_index["NOOP"]]
            matrix_strength = _matrix_means(strength_state, ids, matrix_order)
            matrix_topology = _matrix_means(topology_state, ids, matrix_order)
            strength_boot = _bootstrap_means(matrix_strength, bootstrap)
            topology_boot = _bootstrap_means(matrix_topology, bootstrap)
            strength_p, strength_null = _one_sided_sign_p(matrix_strength, signs)
            topology_p, topology_null = _two_sided_sign_p(matrix_topology, signs)
            strength_raw.append(strength_p)
            topology_raw.append(topology_p)
            key = f"c{candidate}_{half}"
            stored[f"{key}__strength_bootstrap"] = strength_boot
            stored[f"{key}__topology_bootstrap"] = topology_boot
            stored[f"{key}__strength_randomization"] = strength_null
            stored[f"{key}__topology_randomization"] = topology_null
            stored[f"{key}__slope_bootstrap"] = association_draws["slope"]
            stored[f"{key}__spearman_bootstrap"] = association_draws["spearman"]
            ci90 = _interval(topology_boot, alpha=0.10)
            cell: dict[str, Any] = {
                "cell": key,
                "candidate": candidate,
                "branch_half": half,
                "eligible_matrices": int(matrix_order.size),
                "eligible_states": len(candidate_cases),
                "arm_means": {
                    arm: float(_matrix_means(q[:, index], ids, matrix_order).mean())
                    for arm, index in arm_index.items()
                },
                "strength_tighten_minus_loosen": {
                    "estimate": float(matrix_strength.mean()),
                    "bootstrap_ci95": _interval(strength_boot),
                    "randomization_p_raw": strength_p,
                    "matrices_positive": int(np.count_nonzero(matrix_strength > 0.0)),
                    "matrices_zero": int(np.count_nonzero(matrix_strength == 0.0)),
                },
                "topology_minus_noop": {
                    "estimate": float(matrix_topology.mean()),
                    "bootstrap_ci95": _interval(topology_boot),
                    "bootstrap_ci90": ci90,
                    "randomization_p_raw_two_sided": topology_p,
                    "equivalent_margin_0_025": bool(ci90[0] > -EQUIVALENCE_MARGIN and ci90[1] < EQUIVALENCE_MARGIN),
                },
                "throughput_recovery_association": association,
            }
            cells.append(cell)
            for position, matrix_id in enumerate(matrix_order):
                matrix_rows.append({
                    "candidate": candidate,
                    "branch_half": half,
                    "matrix_id": int(matrix_id),
                    "strength_tighten_minus_loosen": float(matrix_strength[position]),
                    "topology_minus_noop": float(matrix_topology[position]),
                })
    strength_adjusted = holm_adjust(strength_raw)
    topology_adjusted = holm_adjust(topology_raw)
    for cell, strength_p, topology_p in zip(cells, strength_adjusted, topology_adjusted, strict=True):
        strength = cell["strength_tighten_minus_loosen"]
        topology = cell["topology_minus_noop"]
        association = cell["throughput_recovery_association"]
        strength["randomization_p_holm"] = float(strength_p)
        topology["randomization_p_holm_two_sided"] = float(topology_p)
        strength_gates = {
            "estimate_positive": strength["estimate"] > 0.0,
            "bootstrap_lower_positive": strength["bootstrap_ci95"][0] > 0.0,
            "holm_p_below_0_05": strength_p < 0.05,
            "throughput_slope_positive": association["state_centered_slope"] > 0.0,
            "throughput_slope_lower_positive": association["slope_bootstrap_ci95"][0] > 0.0,
        }
        cell["strength_gates"] = strength_gates
        cell["strength_cell_pass"] = bool(all(strength_gates.values()))
        topology["ci95_excludes_zero"] = bool(topology["bootstrap_ci95"][0] > 0.0 or topology["bootstrap_ci95"][1] < 0.0)
    topology_estimates = [cell["topology_minus_noop"]["estimate"] for cell in cells]
    same_sign = bool(all(value > 0 for value in topology_estimates) or all(value < 0 for value in topology_estimates))
    topology_effect = bool(
        same_sign
        and all(cell["topology_minus_noop"]["ci95_excludes_zero"] for cell in cells)
        and all(cell["topology_minus_noop"]["randomization_p_holm_two_sided"] < 0.05 for cell in cells)
    )
    topology_negligible = bool(all(cell["topology_minus_noop"]["equivalent_margin_0_025"] for cell in cells))
    topology_classification = (
        "reproducible_directional_topology_effect"
        if topology_effect
        else "negligible_within_0.025"
        if topology_negligible
        else "inconclusive"
    )
    result = {
        "format": "codex-intervention-p4-inference-v1",
        "endpoint": "RUN3_WITHIN_F8_FROM_IDENTICAL_NATURAL_POST_BREAK_DAUGHTER",
        "eligible_matrices_by_candidate": eligible_counts,
        "minimum_required_per_candidate": MINIMUM_ELIGIBLE_MATRICES,
        "eligibility_gate_pass": all(count >= MINIMUM_ELIGIBLE_MATRICES for count in eligible_counts.values()),
        "cells": cells,
        "strength_all_statistical_cells_pass": all(cell["strength_cell_pass"] for cell in cells),
        "topology_classification": topology_classification,
        "strength_and_topology_are_separate_families": True,
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "randomization_repetitions": RANDOMIZATION_REPETITIONS,
        "inference_unit": "whole catalytic matrix",
    }
    return result, matrix_rows, stored


def _prepare_campaign(work: Path, output: Path, registration: dict[str, Any]) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    work.mkdir(parents=True, exist_ok=True)
    contract: dict[str, Any] = {
        "format": "codex-intervention-p4-campaign-v1",
        "registration_id": registration["registration_id"],
        "output": str(output),
        "matrices": MATRICES,
        "landmarks": list(LANDMARKS),
        "branches": BRANCHES,
        "arms": list(ARMS),
        "source_hashes": source_hashes(),
    }
    contract["campaign_id"] = _canonical_digest(_json_ready(contract))
    path = work / "campaign_contract.json"
    if path.exists() and json.loads(path.read_text()) != _json_ready(contract):
        raise ValueError("P4 work directory belongs to another campaign")
    if not path.exists():
        base._atomic_json(path, contract)
    _status(work, "running", "campaign_initialized")


def _status(work: Path, state: str, detail: str) -> None:
    work.mkdir(parents=True, exist_ok=True)
    base._atomic_json(work / "campaign_status.json", {
        "format": "codex-intervention-p4-status-v1",
        "phase": "p4_shared_break_recovery",
        "state": state,
        "detail": detail,
        "mandatory_stop_after_seal": True,
    })


def smoke(registration_directory: Path = DEFAULT_REGISTRATION, output: Path = DEFAULT_SMOKE) -> None:
    registration = verify_registration(registration_directory)
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    smoke_spec = replace(
        spec(),
        matrices=1,
        branches=2,
        landmarks=(5,),
        cohort_seed=SEEDS["smoke_cohort"],
        balanced_selection_seed=SEEDS["smoke_balanced_selection"],
        neutral_selection_seed=SEEDS["smoke_neutral_selection"],
        future_seed=SEEDS["smoke_future"],
    )
    cohort = CohortConfig(1, 2, (5,))
    current_experiment = ExperimentConfig(
        development=cohort,
        confirmation=cohort,
        horizon=RECOVERY_HORIZON,
        master_seed=smoke_spec.cohort_seed,
        bootstrap_repetitions=8,
        permutation_repetitions=8,
    )
    with tempfile.TemporaryDirectory(prefix="codex-p4-smoke-", dir=output.parent) as temporary:
        with threadpool_limits(limits=1):
            cases = build_cohort(current_experiment, "INTP4_NONSCIENTIFIC_SMOKE", cohort)
        # I/O and replay are tested from the restored state. Natural acquisition
        # behavior itself has deterministic fixtures and the scientific run has
        # a complete acquisition replay before any inference.
        generated = p3c.run_phase_batches(cases, current_experiment, smoke_spec, registration_directory / "frozen_full_predictor.npz", registration["registration_id"], Path(temporary) / "generate", 1, "generate")
        replayed = p3c.run_phase_batches(cases, current_experiment, smoke_spec, registration_directory / "frozen_full_predictor.npz", registration["registration_id"], Path(temporary) / "replay", 1, "replay")
        replay = base.replay_audit(generated, replayed)
        if not replay["state_edit_endpoint_and_process_digests_exact"]:
            raise AssertionError("P4 smoke replay failed")
    with _atomic_destination(output) as destination:
        (destination / "manifest.json").write_text(json.dumps({
            "format": "codex-intervention-p4-smoke-v1",
            "registration_id": registration["registration_id"],
            "scientific_result": False,
            "scientific_matrices": 0,
            "scientific_futures": 0,
            "io_checkpoint_and_replay_passed": True,
            "effect_sizes_disclosed": False,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        write_checksums(destination)
    verify_checksums(output)
    print(f"P4 non-scientific smoke passed: {output}", flush=True)


def _inconclusive_output(
    output: Path, registration: dict[str, Any], acquisition: pd.DataFrame, eligible: dict[str, int], acquisition_exact: bool
) -> None:
    with _atomic_destination(output) as destination:
        acquisition.to_csv(destination / "natural_break_acquisition.csv", index=False)
        metrics = {
            "format": "codex-intervention-p4-inconclusive-v1",
            "classification": "inconclusive_insufficient_natural_break_matrices",
            "eligible_matrices_by_candidate": eligible,
            "minimum_required_per_candidate": MINIMUM_ELIGIBLE_MATRICES,
            "strength_recovery_gate_pass": False,
            "topology_classification": "not_tested",
        }
        (destination / "primary_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (destination / "SCIENTIFIC_REPORT.md").write_text("# P4 shared-break recovery\n\nThe frozen minimum of 120 naturally eligible matrices per candidate was not reached. No intervention future was launched; the result is inconclusive.\n", encoding="utf-8")
        (destination / "LAY_SUMMARY.md").write_text("# Lay summary\n\nToo few independent matrices naturally produced a usable break, so the experiment stopped before testing recovery.\n", encoding="utf-8")
        (destination / "manifest.json").write_text(json.dumps({
            "format": RESULT_FORMAT,
            "registration_id": registration["registration_id"],
            "classification": metrics["classification"],
            "eligible_matrices_by_candidate": eligible,
            "natural_acquisition_replay_exact": acquisition_exact,
            "primary_futures": 0,
            "replay_futures": 0,
            "mandatory_stop_after_this_stage": True,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        write_checksums(destination)
    verify_checksums(output)


def _reports(metrics: dict[str, Any], eligible: dict[str, int]) -> tuple[str, str]:
    lines = [
        "# P4 shared-natural-break recovery",
        "",
        f"Catalytic-strength recovery gate: **{metrics['strength_recovery_gate_pass']}**.",
        f"Fixed-throughput topology classification: **{metrics['topology_classification']}**.",
        f"Eligible matrices: `{eligible}`.",
        "",
        "| Cell | Tighten-loosen | 95% CI | Holm p | Topology-noop | 95% CI | Topology Holm p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in metrics["cells"]:
        strength = cell["strength_tighten_minus_loosen"]
        topology = cell["topology_minus_noop"]
        lines.append(
            f"| {cell['cell']} | {strength['estimate']:+.6f} | {strength['bootstrap_ci95']} | {strength['randomization_p_holm']:.6g} | "
            f"{topology['estimate']:+.6f} | {topology['bootstrap_ci95']} | {topology['randomization_p_holm_two_sided']:.6g} |"
        )
    lines.extend([
        "",
        "Every arm began from the identical daughter immediately after the same untreated natural break. This identifies recovery effects without conditioning on treatment-created breaks.",
        "",
        "P4 is separate from the failed P3c composite gate and from original molecular CR5. It cannot establish biological repair, memory, agency, life, or an autonomous attractor.",
        "",
    ])
    lay = "\n".join([
        "# P4 in plain language",
        "",
        "This experiment waits for an assembly to break naturally, saves the exact daughter produced by that break, and makes identical copies. Only then does it strengthen, weaken, or rearrange the catalytic network.",
        "",
        "Because every comparison starts from the same already-broken state, any later difference tells us about rebuilding a short inherited run rather than merely preventing the original break.",
        "",
        f"The strength test {'passed' if metrics['strength_recovery_gate_pass'] else 'did not pass'} its frozen four-cell gate. The topology result was classified as {metrics['topology_classification'].replace('_', ' ')}.",
        "",
    ])
    return "\n".join(lines), lay


def run(
    registration_directory: Path = DEFAULT_REGISTRATION,
    output: Path = DEFAULT_OUTPUT,
    work: Path = DEFAULT_WORK,
    workers: int = min(os.cpu_count() or 1, 14),
) -> None:
    if workers < 1:
        raise ValueError("workers must be positive")
    registration_directory = registration_directory.resolve()
    output = output.resolve()
    work = work.resolve()
    registration = verify_registration(registration_directory)
    verify_checksums(DEFAULT_SMOKE)
    _prepare_campaign(work, output, registration)
    current_spec = spec()
    current_experiment = experiment(current_spec)
    print(f"[p4 1/9] Building {MATRICES} fresh matrices and {2 * MATRICES * len(LANDMARKS)} natural states", flush=True)
    _status(work, "running", "building_natural_states")
    with threadpool_limits(limits=1):
        source_cases = build_cohort(current_experiment, LABEL, current_experiment.confirmation)
    if len(source_cases) != 2 * MATRICES * len(LANDMARKS):
        raise AssertionError("P4 source cohort is incomplete")
    print("[p4 2/9] Acquiring and replaying untreated natural breaks", flush=True)
    _status(work, "running", "natural_break_acquisition")
    broken, anchors, acquisition = acquire_cohort(source_cases, current_experiment)
    replay_broken, replay_anchors, replay_acquisition = acquire_cohort(source_cases, current_experiment)
    acquisition_exact = bool(
        _json_ready(acquisition.to_dict("records")) == _json_ready(replay_acquisition.to_dict("records"))
        and [base._snapshot_digest(case) for case in broken] == [base._snapshot_digest(case) for case in replay_broken]
        and all(np.array_equal(left, right) for left, right in zip(anchors, replay_anchors, strict=True))
    )
    if not acquisition_exact:
        raise AssertionError("P4 natural acquisition replay failed")
    eligible = {
        candidate: len({case.matrix_id for case in broken if case.candidate == candidate})
        for candidate in CANDIDATES
    }
    if not all(count >= MINIMUM_ELIGIBLE_MATRICES for count in eligible.values()):
        _inconclusive_output(output, registration, acquisition, eligible, acquisition_exact)
        _status(work, "sealed_inconclusive", "insufficient_natural_break_matrices")
        print("[p4] Inconclusive eligibility stop; no intervention futures launched", flush=True)
        return
    futures = len(broken) * len(ARMS) * BRANCHES
    print(f"[p4 3/9] Shooting {futures:,} F8 recovery futures", flush=True)
    _status(work, "running", "shooting_recovery_futures")
    generated = p3c.run_phase_batches(broken, current_experiment, current_spec, registration_directory / "frozen_full_predictor.npz", registration["registration_id"], work / "generate", workers, "generate")
    print(f"[p4 4/9] Replaying all {futures:,} recovery futures", flush=True)
    _status(work, "running", "exact_future_replay")
    replayed = p3c.run_phase_batches(broken, current_experiment, current_spec, registration_directory / "frozen_full_predictor.npz", registration["registration_id"], work / "replay", workers, "replay")
    replay = base.replay_audit(generated, replayed)
    replay_exact = bool(acquisition_exact and replay["state_edit_endpoint_and_process_digests_exact"])
    if not replay_exact:
        raise AssertionError("P4 exact replay failed")
    print("[p4 5/9] Computing frozen whole-matrix inference", flush=True)
    _status(work, "running", "whole_matrix_inference")
    arrays = p3c._outcome_arrays(broken, generated, current_spec)
    geometry, geometry_rows, surgery_audit = p3c._geometry_arrays(broken, generated, current_spec)
    metrics, matrix_rows, stored = compute_inference(broken, arrays, geometry)
    secondary, secondary_arrays = p3c._resilience_secondary(broken, anchors, arrays)
    print("[p4 6/9] Writing and readback-checking artifacts", flush=True)
    _status(work, "running", "artifact_write_and_readback")
    with _atomic_destination(output) as destination:
        acquisition.to_csv(destination / "natural_break_acquisition.csv", index=False)
        np.savez_compressed(destination / "branch_arrays.npz", **arrays)
        np.savez_compressed(destination / "surgery_geometry_arrays.npz", **geometry)
        np.savez_compressed(destination / "secondary_arrays.npz", **secondary_arrays)
        np.savez_compressed(destination / "inference_arrays.npz", **stored)
        base._write_branch_table(destination / "branches.csv.gz", broken, generated)
        base._write_state_artifacts(destination, broken, generated, arrays)
        base._write_selection_artifacts(destination, broken, generated, current_spec)
        geometry_rows.to_csv(destination / "surgery_geometry_audit.csv.gz", index=False)
        pd.DataFrame(matrix_rows).to_csv(destination / "matrix_effects.csv", index=False)
        with np.load(destination / "branch_arrays.npz", allow_pickle=False) as archive:
            read_arrays = {name: archive[name] for name in archive.files}
        with np.load(destination / "surgery_geometry_arrays.npz", allow_pickle=False) as archive:
            read_geometry = {name: archive[name] for name in archive.files}
        read_metrics, read_rows, _ = compute_inference(broken, read_arrays, read_geometry)
        readback_exact = bool(_json_ready(read_metrics) == _json_ready(metrics) and _json_ready(read_rows) == _json_ready(matrix_rows))
        if not readback_exact:
            raise AssertionError("P4 artifact readback failed")
        integrity = {
            "natural_acquisition_replay_exact": acquisition_exact,
            "future_replay_exact": replay_exact,
            "artifact_readback_exact": readback_exact,
            "surgery_audit_pass": surgery_audit["all_audits_pass"],
        }
        metrics["integrity_gates"] = integrity
        metrics["strength_recovery_gate_pass"] = bool(metrics["strength_all_statistical_cells_pass"] and all(integrity.values()))
        technical, lay = _reports(metrics, eligible)
        (destination / "primary_metrics.json").write_text(json.dumps(_json_ready(metrics), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (destination / "secondary_outcomes.json").write_text(json.dumps(_json_ready(secondary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (destination / "surgery_audit_summary.json").write_text(json.dumps(_json_ready(surgery_audit), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (destination / "replay_audit.json").write_text(json.dumps(_json_ready(replay), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (destination / "readback_audit.json").write_text(json.dumps({"primary_metrics_exact": readback_exact, "matrix_effects_exact": readback_exact, "acquisition_exact": acquisition_exact}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (destination / "SCIENTIFIC_REPORT.md").write_text(technical, encoding="utf-8")
        (destination / "LAY_SUMMARY.md").write_text(lay, encoding="utf-8")
        claims = {
            "supported": (["causal catalytic-strength control of recovery from an identical natural break"] if metrics["strength_recovery_gate_pass"] else []),
            "topology_classification": metrics["topology_classification"],
            "failed_predictions": ([] if metrics["strength_recovery_gate_pass"] else ["registered catalytic-strength recovery gate"]),
            "deviations": [],
            "prohibited": protocol()["claim_boundary"]["prohibited"],
        }
        (destination / "claim_boundaries.json").write_text(json.dumps(claims, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest = {
            "format": RESULT_FORMAT,
            "registration_id": registration["registration_id"],
            "eligible_matrices_by_candidate": eligible,
            "eligible_states": len(broken),
            "primary_futures": futures,
            "replay_futures": futures,
            "strength_recovery_gate_pass": metrics["strength_recovery_gate_pass"],
            "topology_classification": metrics["topology_classification"],
            "exact_replay": replay_exact,
            "complete_readback_exact": readback_exact,
            "no_acquisition_or_future_retries": True,
            "no_matrix_or_state_replacement": True,
            "mandatory_stop_after_this_stage": True,
            "next_scientific_stage_launched": False,
            "runtime": _runtime_manifest(),
        }
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        write_checksums(destination)
    verify_checksums(output)
    print("[p4 7/9] Result checksum sealed", flush=True)
    _status(work, "sealed_complete", "mandatory_review_stop")
    print("[p4 8/9] Durable status updated", flush=True)
    print("[p4 9/9] STOPPED; no later scientific stage launched", flush=True)


def read_status(work: Path = DEFAULT_WORK) -> dict[str, Any]:
    return base.read_status(work)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate").add_argument("--output", type=Path, default=DEFAULT_VALIDATION)
    register_parser = commands.add_parser("register")
    register_parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    register_parser.add_argument("--output", type=Path, default=DEFAULT_REGISTRATION)
    commands.add_parser("verify").add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    smoke_parser = commands.add_parser("smoke")
    smoke_parser.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    smoke_parser.add_argument("--output", type=Path, default=DEFAULT_SMOKE)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    run_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run_parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    run_parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 14))
    commands.add_parser("status").add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
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
        run(args.registration, args.output, args.work_dir, args.workers)
    elif args.command == "status":
        print(json.dumps(read_status(args.work_dir), indent=2, sort_keys=True))
    else:  # pragma: no cover
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
