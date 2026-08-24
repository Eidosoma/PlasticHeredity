"""Full prospective CR3 confirmation of the outgoing catalytic rule."""

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
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from threadpoolctl import threadpool_limits

from . import intervention_replication as base
from .config import CANDIDATES, CohortConfig, ExperimentConfig
from .experiment import StateCase, _json_ready, _runtime_manifest, build_cohort
from .intervention_core import (
    FrozenFullPredictor,
    MolecularEdit,
    enumerate_legal_edits,
    simulate_one_shot,
)
from .intervention_metrics import compute_one_shot_inference, generate_inference_draws
from .intervention_outgoing_rule import (
    SEED_DOMAINS as P2B_SEEDS,
    outgoing_catalytic_influence,
    select_outgoing_rule_edits,
    validation_checks as outgoing_validation_checks,
)
from .mechanistic import (
    _atomic_destination,
    _canonical_digest,
    sha256_file,
    verify_checksums,
    write_checksums,
)
from .seeds import derive_seed


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results_intervention_replication"
ORIGINAL_REGISTRATION = RESULT_ROOT / "registration"
P2B_PILOT = RESULT_ROOT / "p2b_cr3_outgoing_rule_pilot"
CR1_RESULT = RESULT_ROOT / "cr1_model_guided_confirmation"
DEFAULT_VALIDATION = RESULT_ROOT / "cr3_confirmation_validation"
DEFAULT_REGISTRATION = RESULT_ROOT / "cr3_confirmation_registration"
DEFAULT_SMOKE = RESULT_ROOT / "cr3_confirmation_smoke"
DEFAULT_OUTPUT = RESULT_ROOT / "cr3_physical_rule_confirmation"
DEFAULT_WORK = RESULT_ROOT / ".cr3_physical_rule_confirmation_work"

DOCUMENT = "CODEX_INTERVENTION_CR3_CONFIRMATION_PREREGISTRATION.md"
SOURCE_FILES = (
    DOCUMENT,
    "REPOSITORY_RELOCATION_AUDIT.md",
    "plastic_heredity/intervention_cr3_confirmation.py",
    "plastic_heredity/archive_paths.py",
    "tests/test_intervention_cr3_confirmation.py",
    "tests/test_archive_paths.py",
    "plastic_heredity/intervention_outgoing_rule.py",
    "plastic_heredity/intervention_readback_recovery.py",
    "plastic_heredity/intervention_p2_readback_recovery.py",
    "plastic_heredity/intervention_p3_inference_recovery.py",
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
    "plastic_heredity/models.py",
    "pyproject.toml",
    "requirements-lock.txt",
)

PROGRAM_FORMAT = "codex-intervention-cr3-confirmation-v1"
VALIDATION_FORMAT = "codex-intervention-cr3-confirmation-validation-v1"
REGISTRATION_FORMAT = "codex-intervention-cr3-confirmation-registration-v1"
RESULT_FORMAT = "codex-intervention-cr3-confirmation-result-v1"
CHECKPOINT_FORMAT = "codex-intervention-cr3-confirmation-checkpoint-v1"
LABEL = "INTCR3_OUTGOING_RULE_CONFIRMATION_V1"

MATRICES = 200
BRANCHES = 64
LANDMARKS = (20, 35, 50, 65, 80)
HORIZON = 12
BOOTSTRAP_REPETITIONS = 4_096
RANDOMIZATION_REPETITIONS = 4_096
EQUIVALENCE_MARGIN = 0.025
MINIMUM_CPU_BUDGET_HOURS = 15.0
MINIMUM_FREE_DISK_BYTES = 2_500_000_000


def _seed(name: str) -> str:
    return hashlib.sha256(
        f"codex-clean-room-cr3-full-confirmation-v1::{name}".encode("utf-8")
    ).hexdigest()


SEEDS = {
    name: _seed(name)
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


def source_hashes() -> dict[str, str]:
    return {name: sha256_file(ROOT / name) for name in SOURCE_FILES}


def phase_spec() -> base.PhaseSpec:
    return base.PhaseSpec(
        phase="p2",
        role="full prospective CR3 outgoing catalytic-rule confirmation",
        matrices=MATRICES,
        branches=BRANCHES,
        cohort_seed=SEEDS["cohort"],
        selection_seed=SEEDS["selection"],
        future_seed=SEEDS["future"],
        bootstrap_seed=SEEDS["bootstrap"],
        randomization_seed=SEEDS["randomization"],
    )


def experiment(spec: base.PhaseSpec | None = None) -> ExperimentConfig:
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


def _future_seed(spec: base.PhaseSpec, case: StateCase, branch: int) -> int:
    return derive_seed(
        spec.future_seed,
        f"{LABEL}.future",
        case.candidate,
        case.matrix_id,
        case.landmark,
        branch,
    )


def _selection_seed(spec: base.PhaseSpec, case: StateCase) -> int:
    return derive_seed(
        spec.selection_seed,
        f"{LABEL}.selection.random_arm",
        case.candidate,
        case.matrix_id,
        case.landmark,
    )


def protocol() -> dict[str, Any]:
    spec = phase_spec()
    value: dict[str, Any] = {
        "format": PROGRAM_FORMAT,
        "status": "frozen_before_any_cr3_confirmation_matrix",
        "endpoint": "JOINT_BREAK_RUN3 within F12",
        "predecessors": {
            "p2_incoming_negative_control_preserved": True,
            "p2b_corrected_outgoing_pilot_is_developmental": True,
            "p2b_outcomes_seen_before_registration": True,
            "p2b_outcomes_do_not_select_design_seeds_or_gate": True,
            "cr1_model_guided_confirmation_already_sealed": True,
        },
        "cohort": {
            "matrices": MATRICES,
            "candidates": list(CANDIDATES),
            "landmarks": list(LANDMARKS),
            "states": 2 * MATRICES * len(LANDMARKS),
            "fresh_matrices": True,
            "natural_untreated_landmarks": True,
            "no_risk_or_outcome_state_selection": True,
        },
        "arms": list(spec.arms),
        "rule": {
            "beta_storage": "beta[target,catalyst]",
            "kinetic_boost": "beta @ n",
            "normalization": "x = n / sum(n)",
            "outgoing_quantity": "x @ beta == beta.T @ x",
            "incoming_quantity_not_used": "beta @ x",
            "rule_up": "legal edit minimizing outgoing[add]-outgoing[remove]",
            "rule_down": "legal edit maximizing outgoing[add]-outgoing[remove]",
            "all_legal_swaps_enumerated": True,
            "tie_breaking": "first fixed lexicographic legal edit",
            "random_uniform_over_legal_swaps": True,
        },
        "futures": {
            "branches_per_arm_state": BRANCHES,
            "horizon": HORIZON,
            "halves": {"A": [0, 31], "B": [32, 63]},
            "primary_futures": 512_000,
            "replay_futures": 512_000,
            "common_random_streams": True,
            "future_seed_excludes_arm": True,
            "random_selection_stream_separate": True,
            "no_retries_or_matrix_replacement": True,
        },
        "inference": {
            "unit": "whole catalytic matrix",
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "randomization_repetitions": RANDOMIZATION_REPETITIONS,
            "holm_family": ["c02_A", "c02_B", "c03_A", "c03_B"],
            "target_contrast": "RULE_UP - RULE_DOWN",
            "target_positive": True,
            "target_bootstrap_lower_positive": True,
            "target_holm_randomization_p_below": 0.05,
            "random_noop_tost_margin": [-EQUIVALENCE_MARGIN, EQUIVALENCE_MARGIN],
            "random_noop_tost_method": "90% whole-matrix bootstrap interval strictly inside margin",
            "cr1_only_up_noop_and_noop_down_checks_are_not_gates": True,
            "cr1_only_random_effect_ratio_check_is_not_a_gate": True,
            "exact_replay_and_readback_required": True,
        },
        "descriptive": {
            "registered_secondary_process_outcomes": True,
            "landmark_and_matrix_effects": True,
            "ratio_to_sealed_cr1_model_guided_effect": True,
            "external_numerical_benchmarks_used_for_fitting_or_gating": False,
        },
        "operational": {
            "estimated_cpu_hours": [11.0, 13.0],
            "minimum_declared_cpu_budget_hours": MINIMUM_CPU_BUDGET_HOURS,
            "minimum_free_disk_bytes": MINIMUM_FREE_DISK_BYTES,
            "no_mid_phase_kill": True,
            "mandatory_review_stop_after_seal": True,
        },
        "seed_domains": SEEDS,
        "claim_boundary": {
            "prohibited": [
                "strict-eight control",
                "agency",
                "biological memory",
                "error correction",
                "life",
                "autonomous organization",
                "autonomous attractor",
                "real prebiotic chemistry",
                "Phi or PhiID intervention",
                "universal origin-of-life mechanism",
            ]
        },
    }
    value["protocol_id"] = _canonical_digest(_json_ready(value))
    return value


def add_cr3_gate_fields(metrics: dict[str, Any]) -> dict[str, Any]:
    """Add the frozen CR3 gates without importing stricter CR1-only gates."""

    for cell in metrics["cells"]:
        effect = cell["contrasts"]["up_minus_down"]
        gates = {
            "rule_up_minus_down_positive": effect["estimate"] > 0.0,
            "rule_up_minus_down_bootstrap_lower_positive": (
                effect["bootstrap_ci95"][0] > 0.0
            ),
            "holm_randomization_below_0_05": (
                cell["up_down_randomization_p_holm"] < 0.05
            ),
            "random_tost_equivalent_to_noop": cell["random_noop_equivalence"][
                "tost_equivalent"
            ],
        }
        cell["cr3_registered_gates"] = gates
        cell["cr3_registered_cell_pass"] = bool(all(gates.values()))
    metrics["cr3_all_four_cells_scientific_pass"] = bool(
        all(cell["cr3_registered_cell_pass"] for cell in metrics["cells"])
    )
    return metrics


def _phase_worker(
    arguments: tuple[StateCase, ExperimentConfig, base.PhaseSpec, str]
) -> base.PhaseBatch:
    case, current_experiment, spec, model_path = arguments
    limiter = threadpool_limits(limits=1)
    try:
        predictor = FrozenFullPredictor.load(model_path)
        selected = select_outgoing_rule_edits(case.snapshot.composition, case.beta)
        legal = enumerate_legal_edits(case.snapshot.composition)
        random_rng = np.random.default_rng(_selection_seed(spec, case))
        random_edit = legal[int(random_rng.integers(0, len(legal)))]
        by_name = {
            "RULE_UP": selected["RULE_UP"],
            "RULE_DOWN": selected["RULE_DOWN"],
            "RANDOM": random_edit,
        }
        edits: tuple[MolecularEdit | None, ...] = tuple(
            None if arm == "NOOP" else by_name[arm] for arm in spec.arms
        )
        predictions = base._predict_edit_arms(
            predictor,
            case.candidate,
            case.snapshot,
            case.beta,
            current_experiment.gard,
            edits,
        )
        arm_outcomes: list[list[Any | None]] = [
            [None] * spec.branches for _ in spec.arms
        ]
        for branch in range(spec.branches):
            seed = _future_seed(spec, case, branch)
            for arm_index, _arm in enumerate(spec.arms):
                arm_outcomes[arm_index][branch] = simulate_one_shot(
                    case.snapshot,
                    case.beta,
                    case.candidate,
                    current_experiment.gard,
                    HORIZON,
                    np.random.default_rng(seed),
                    edits[arm_index],
                )
        outcomes = tuple(
            tuple(value for value in arm if value is not None)
            for arm in arm_outcomes
        )
        if any(len(arm) != spec.branches for arm in outcomes):
            raise AssertionError("CR3 worker dropped an outcome")
        return base.PhaseBatch(
            state_id=case.state_id,
            state_digest=base._snapshot_digest(case),
            arm_names=spec.arms,
            predictions=np.asarray(predictions, dtype=np.float64),
            selected_edits=edits,
            surgeries=tuple(None for _ in spec.arms),
            scored_edits=tuple(),
            catalytic_support=outgoing_catalytic_influence(
                case.snapshot.composition, case.beta
            ),
            outcomes=outcomes,
        )
    finally:
        limiter.restore_original_limits()


def _checkpoint_contract(
    cases: list[StateCase],
    spec: base.PhaseSpec,
    registration_id: str,
    stage: str,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "registration_id": registration_id,
        "scientific_label": LABEL,
        "phase": "cr3_physical_rule_confirmation",
        "stage": stage,
        "matrices": spec.matrices,
        "branches": spec.branches,
        "horizon": HORIZON,
        "arms": list(spec.arms),
        "case_ids": [case.state_id for case in cases],
        "case_digests": [base._snapshot_digest(case) for case in cases],
        "future_seed": spec.future_seed,
        "future_seed_includes_arm": False,
        "selection_seed": spec.selection_seed,
        "rule_expression": "x @ beta == beta.T @ x",
        "source_hashes": source_hashes(),
    }
    value["contract_id"] = _canonical_digest(_json_ready(value))
    return value


def run_phase_batches(
    cases: list[StateCase],
    current_experiment: ExperimentConfig,
    spec: base.PhaseSpec,
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
        if path.is_file():
            with path.open("rb") as handle:
                batch = pickle.load(handle)
            if (
                not isinstance(batch, base.PhaseBatch)
                or batch.state_id != case.state_id
                or batch.state_digest != base._snapshot_digest(case)
                or batch.arm_names != spec.arms
            ):
                raise ValueError(f"invalid CR3 checkpoint {path}")
            batches[index] = batch
        else:
            missing.append(index)

    def status(state: str) -> None:
        complete = sum(batch is not None for batch in batches)
        base._atomic_json(
            checkpoint_directory / "status.json",
            {
                "format": CHECKPOINT_FORMAT,
                "phase": "cr3_physical_rule_confirmation",
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
        generated = map(_phase_worker, arguments)
        for index, batch in zip(missing, generated, strict=True):
            batches[index] = batch
            base._atomic_pickle(
                checkpoint_directory / f"state_{index:04d}.pkl", batch
            )
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
        raise AssertionError("checkpointed CR3 phase has missing states")
    return [batch for batch in batches if batch is not None]


def validation_checks() -> dict[str, Any]:
    from .intervention_cr1_confirmation import SEEDS as CR1_SEEDS
    from .intervention_cr2_dose_response import SEEDS as CR2_SEEDS
    from .intervention_p3b_dose_bridge import SEED_DOMAINS as P3B_SEEDS
    from .intervention_p3c import SEED_DOMAINS as P3C_SEEDS
    from .intervention_p4 import SEEDS as P4_SEEDS

    inherited = outgoing_validation_checks()
    all_prior_seed_values = set(base.SEED_DOMAINS.values())
    for prior in (
        P2B_SEEDS,
        CR1_SEEDS,
        CR2_SEEDS,
        P3B_SEEDS,
        P3C_SEEDS,
        P4_SEEDS,
    ):
        all_prior_seed_values.update(prior.values())
    checks: dict[str, Any] = {
        "inherited_34_checks_pass": inherited["all_checks_passed"],
        "inherited_validation_generated_no_scientific_cohort": not inherited[
            "scientific_cohort_generated"
        ],
        "full_matrix_count": MATRICES == 200,
        "full_branch_count": BRANCHES == 64,
        "five_registered_landmarks": LANDMARKS == (20, 35, 50, 65, 80),
        "full_primary_future_count": (
            2 * MATRICES * len(LANDMARKS) * 4 * BRANCHES == 512_000
        ),
        "arm_order_exact": phase_spec().arms
        == ("RULE_UP", "RULE_DOWN", "RANDOM", "NOOP"),
        "contrast_exact": phase_spec().contrast == ("RULE_UP", "RULE_DOWN"),
        "seed_domains_unique": len(SEEDS) == len(set(SEEDS.values())),
        "seeds_disjoint_from_base": set(SEEDS.values()).isdisjoint(
            base.SEED_DOMAINS.values()
        ),
        "seeds_disjoint_from_p2b": set(SEEDS.values()).isdisjoint(
            P2B_SEEDS.values()
        ),
        "seeds_disjoint_from_all_known_prior_campaigns": set(
            SEEDS.values()
        ).isdisjoint(all_prior_seed_values),
        "p2b_result_checksum_verified": True,
        "cr1_result_checksum_verified": True,
        "cr1_passed": True,
        "cr3_gate_excludes_cr1_only_side_gates": protocol()["inference"][
            "cr1_only_up_noop_and_noop_down_checks_are_not_gates"
        ],
        "cr3_gate_excludes_cr1_only_ratio_gate": protocol()["inference"][
            "cr1_only_random_effect_ratio_check_is_not_a_gate"
        ],
        "no_scientific_cohort_generated": True,
    }
    verify_checksums(P2B_PILOT)
    verify_checksums(CR1_RESULT)
    cr1_manifest = json.loads((CR1_RESULT / "manifest.json").read_text())
    checks["cr1_passed"] = bool(cr1_manifest["full_four_cell_gate"])

    fixture_metrics = {
        "cells": [
            {
                "contrasts": {
                    "up_minus_down": {
                        "estimate": 0.1,
                        "bootstrap_ci95": (0.05, 0.15),
                    }
                },
                "up_down_randomization_p_holm": 0.01,
                "random_noop_equivalence": {"tost_equivalent": True},
            }
            for _ in range(4)
        ]
    }
    add_cr3_gate_fields(fixture_metrics)
    checks["cr3_gate_fixture_passes"] = fixture_metrics[
        "cr3_all_four_cells_scientific_pass"
    ]
    if not all(bool(value) for value in checks.values()):
        raise AssertionError(
            {name: value for name, value in checks.items() if not bool(value)}
        )
    return {
        "format": VALIDATION_FORMAT,
        "checks": checks,
        "inherited_check_count": inherited["check_count"],
        "all_checks_passed": True,
        "scientific_matrices_generated": 0,
        "scientific_futures_generated": 0,
        "source_hashes": source_hashes(),
    }


def validate(output: Path = DEFAULT_VALIDATION) -> None:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    for scientific in (DEFAULT_OUTPUT, DEFAULT_WORK):
        if scientific.exists():
            raise FileExistsError(
                f"CR3 scientific artifact exists before validation: {scientific}"
            )
    validation = validation_checks()
    command = [str(ROOT / ".venv/bin/python"), "-m", "pytest", "-q"]
    completed = subprocess.run(
        command, cwd=ROOT, check=False, capture_output=True, text=True
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "complete repository validation failed\n"
            + completed.stdout
            + "\n"
            + completed.stderr
        )
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
    print(f"CR3 confirmation validation sealed: {output}", flush=True)


def _append_registration_notice(registration_id: str) -> None:
    path = ROOT / "INTERVENTION_RESULTS_LEDGER.md"
    text = path.read_text(encoding="utf-8").rstrip() + "\n"
    marker = f"<!-- cr3-confirmation-registered-{registration_id} -->"
    if marker in text:
        return
    rows = [
        "",
        marker,
        "## Full CR3 outgoing physical-rule confirmation registered",
        "",
        f"- Registration: `{registration_id}`.",
        "- The corrected outgoing rule is frozen as `x @ beta`, equivalently `beta.T @ x` under Codex storage.",
        "- 200 fresh matrices, both candidates, five landmarks, 64 F12 branches per arm, and complete replay.",
        "- The CR3 gate uses targeted separation plus RANDOM/NOOP equivalence; CR1-only side-arm and effect-ratio gates are not imported.",
        "- Scientific CR3 matrices and futures at registration: **0**.",
        "- Status: sealed before scientific CR3 execution.",
        "",
    ]
    path.write_text(text + "\n".join(rows), encoding="utf-8")


def register(
    validation_directory: Path = DEFAULT_VALIDATION,
    output: Path = DEFAULT_REGISTRATION,
) -> None:
    validation_directory = validation_directory.resolve()
    output = output.resolve()
    verify_checksums(validation_directory)
    validation = json.loads((validation_directory / "validation.json").read_text())
    if not validation.get("all_checks_passed"):
        raise ValueError("CR3 confirmation validation did not pass")
    for scientific in (DEFAULT_OUTPUT, DEFAULT_WORK):
        if scientific.exists():
            raise FileExistsError(
                f"CR3 scientific artifact exists before registration: {scientific}"
            )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    verify_checksums(P2B_PILOT)
    verify_checksums(CR1_RESULT)
    frozen = protocol()
    payload: dict[str, Any] = {
        "format": REGISTRATION_FORMAT,
        "protocol_id": frozen["protocol_id"],
        "source_hashes": source_hashes(),
        "validation_checksum_manifest_sha256": sha256_file(
            validation_directory / "SHA256SUMS"
        ),
        "p2b_pilot_checksum_manifest_sha256": sha256_file(
            P2B_PILOT / "SHA256SUMS"
        ),
        "cr1_result_checksum_manifest_sha256": sha256_file(
            CR1_RESULT / "SHA256SUMS"
        ),
        "frozen_model_sha256": sha256_file(
            ORIGINAL_REGISTRATION / "frozen_full_predictor.npz"
        ),
        "seed_registry": SEEDS,
        "scientific_matrices_at_registration": 0,
        "scientific_futures_at_registration": 0,
        "p2b_outcomes_seen_before_registration": True,
        "p2b_outcomes_used_to_change_frozen_rule_or_gate": False,
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
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        shutil.copy2(
            ORIGINAL_REGISTRATION / "frozen_full_predictor.npz",
            destination / "frozen_full_predictor.npz",
        )
        shutil.copy2(
            validation_directory / "pytest_output.txt",
            destination / "pytest_output.txt",
        )
        write_checksums(destination)
    verify_registration(output)
    _append_registration_notice(payload["registration_id"])
    print(f"CR3 confirmation registration sealed: {payload['registration_id']}", flush=True)


def verify_registration(directory: Path = DEFAULT_REGISTRATION) -> dict[str, Any]:
    directory = directory.resolve()
    verify_checksums(directory)
    payload = json.loads((directory / "registration.json").read_text())
    if payload.get("format") != REGISTRATION_FORMAT:
        raise ValueError("invalid CR3 confirmation registration")
    unsigned = dict(payload)
    registration_id = unsigned.pop("registration_id", None)
    if registration_id is None or _canonical_digest(_json_ready(unsigned)) != registration_id:
        raise ValueError("invalid CR3 confirmation registration ID")
    if payload["source_hashes"] != source_hashes():
        raise ValueError("CR3 confirmation source changed after registration")
    if json.loads((directory / "protocol.json").read_text()) != protocol():
        raise ValueError("CR3 confirmation protocol changed after registration")
    if payload["seed_registry"] != SEEDS:
        raise ValueError("CR3 confirmation seed registry changed")
    if payload["frozen_model_sha256"] != sha256_file(
        directory / "frozen_full_predictor.npz"
    ) or payload["frozen_model_sha256"] != base.EXPECTED_MODEL_SHA256:
        raise ValueError("CR3 frozen predictor changed")
    verify_checksums(P2B_PILOT)
    verify_checksums(CR1_RESULT)
    if payload["p2b_pilot_checksum_manifest_sha256"] != sha256_file(
        P2B_PILOT / "SHA256SUMS"
    ):
        raise ValueError("P2b predecessor result changed")
    if payload["cr1_result_checksum_manifest_sha256"] != sha256_file(
        CR1_RESULT / "SHA256SUMS"
    ):
        raise ValueError("CR1 comparison result changed")
    return payload


def smoke(
    registration_directory: Path = DEFAULT_REGISTRATION,
    output: Path = DEFAULT_SMOKE,
) -> None:
    registration = verify_registration(registration_directory)
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    smoke_spec = base.PhaseSpec(
        phase="p2",
        role="non-scientific CR3 confirmation smoke",
        matrices=1,
        branches=2,
        cohort_seed=SEEDS["smoke_cohort"],
        selection_seed=SEEDS["smoke_selection"],
        future_seed=SEEDS["smoke_future"],
        bootstrap_seed=SEEDS["validation"],
        randomization_seed=SEEDS["replay"],
    )
    cohort = CohortConfig(1, 2, (5,))
    current_experiment = ExperimentConfig(
        development=cohort,
        confirmation=cohort,
        horizon=HORIZON,
        bootstrap_repetitions=8,
        permutation_repetitions=8,
        regenerate_confirmation=True,
        master_seed=smoke_spec.cohort_seed,
    )
    with tempfile.TemporaryDirectory(
        prefix="codex-cr3-confirmation-smoke-", dir=output.parent
    ) as temporary:
        with threadpool_limits(limits=1):
            cases = build_cohort(
                current_experiment, "INTCR3_CONFIRMATION_SMOKE", cohort
            )
        model = registration_directory / "frozen_full_predictor.npz"
        generated = run_phase_batches(
            cases,
            current_experiment,
            smoke_spec,
            model,
            registration["registration_id"],
            Path(temporary) / "generate",
            1,
            "generate",
        )
        replayed = run_phase_batches(
            cases,
            current_experiment,
            smoke_spec,
            model,
            registration["registration_id"],
            Path(temporary) / "replay",
            1,
            "replay",
        )
        replay = base.replay_audit(generated, replayed)
        if not replay["state_edit_endpoint_and_process_digests_exact"]:
            raise AssertionError("CR3 confirmation smoke replay failed")
    with _atomic_destination(output) as destination:
        (destination / "manifest.json").write_text(
            json.dumps(
                {
                    "format": "codex-intervention-cr3-confirmation-smoke-v1",
                    "registration_id": registration["registration_id"],
                    "scientific_result": False,
                    "scientific_matrices": 0,
                    "scientific_futures": 0,
                    "legality_io_checkpoint_and_replay_passed": True,
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
    print(f"CR3 confirmation smoke passed: {output}", flush=True)


def _status(
    work: Path,
    state: str,
    detail: str,
    cpu_budget_hours: float | None = None,
) -> None:
    value: dict[str, Any] = {
        "format": "codex-intervention-cr3-confirmation-status-v1",
        "phase": "cr3_physical_rule_confirmation",
        "state": state,
        "detail": detail,
        "mandatory_stop_after_seal": True,
    }
    if cpu_budget_hours is not None:
        value["declared_cpu_budget_hours"] = cpu_budget_hours
    work.mkdir(parents=True, exist_ok=True)
    base._atomic_json(work / "campaign_status.json", value)


def _prepare_campaign(
    work: Path,
    output: Path,
    registration: dict[str, Any],
    cpu_budget_hours: float,
) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if cpu_budget_hours < MINIMUM_CPU_BUDGET_HOURS:
        raise ValueError(
            f"CR3 confirmation requires at least {MINIMUM_CPU_BUDGET_HOURS:.1f} CPU-hours"
        )
    free = shutil.disk_usage(RESULT_ROOT).free
    if free < MINIMUM_FREE_DISK_BYTES:
        raise OSError(
            f"CR3 confirmation needs at least {MINIMUM_FREE_DISK_BYTES} free bytes; found {free}"
        )
    work.mkdir(parents=True, exist_ok=True)
    contract: dict[str, Any] = {
        "format": "codex-intervention-cr3-confirmation-campaign-v1",
        "registration_id": registration["registration_id"],
        "output": str(output),
        "matrices": MATRICES,
        "branches": BRANCHES,
        "landmarks": list(LANDMARKS),
        "arms": list(phase_spec().arms),
        "declared_cpu_budget_hours": cpu_budget_hours,
        "free_disk_bytes_at_launch": free,
        "source_hashes": source_hashes(),
    }
    contract["campaign_id"] = _canonical_digest(_json_ready(contract))
    path = work / "campaign_contract.json"
    if path.exists() and json.loads(path.read_text()) != _json_ready(contract):
        raise ValueError("CR3 work directory belongs to another campaign")
    if not path.exists():
        base._atomic_json(path, contract)
    _status(work, "running", "campaign_initialized", cpu_budget_hours)


def _readback_metrics(
    output: Path,
    cases: list[StateCase],
    spec: base.PhaseSpec,
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
    observed, matrix_rows = compute_one_shot_inference(
        cases,
        spec.arms,
        targets,
        predictions,
        draws,
        up_arm="RULE_UP",
        down_arm="RULE_DOWN",
        equivalence_margin=EQUIVALENCE_MARGIN,
        random_ratio_limit=base.RANDOM_RATIO_LIMIT,
    )
    add_cr3_gate_fields(observed)
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
        raise ValueError("CR3 round-trip inference changed")
    return {
        "branch_arrays_reloaded": True,
        "inference_draws_reloaded": True,
        "primary_metrics_exact": metrics_exact,
        "matrix_effects_exact": matrix_effects_exact,
        "cr3_specific_gate_recomputed": True,
        "no_fitting_or_recalibration": True,
    }


def _add_cr1_efficiency(metrics: dict[str, Any]) -> None:
    cr1 = json.loads((CR1_RESULT / "primary_metrics.json").read_text())
    cr1_effects = {
        cell["cell"]: cell["contrasts"]["up_minus_down"]["estimate"]
        for cell in cr1["cells"]
    }
    rows: list[dict[str, Any]] = []
    for cell in metrics["cells"]:
        rule = cell["contrasts"]["up_minus_down"]["estimate"]
        model = cr1_effects[cell["cell"]]
        rows.append(
            {
                "cell": cell["cell"],
                "rule_up_minus_down": rule,
                "sealed_cr1_model_up_minus_down": model,
                "descriptive_efficiency_ratio": rule / model if model != 0.0 else None,
            }
        )
    metrics["descriptive_efficiency_relative_to_sealed_cr1"] = {
        "inference_class": "descriptive_only",
        "cr1_checksum_manifest_sha256": sha256_file(CR1_RESULT / "SHA256SUMS"),
        "cells": rows,
    }


def _reports(metrics: dict[str, Any]) -> tuple[str, str]:
    lines = [
        "# Full CR3 outgoing physical-rule confirmation",
        "",
        f"Registered four-cell CR3 gate: **{metrics['confirmation_gate_pass']}**.",
        f"Exact replay: **{metrics['integrity_gates']['exact_replay']}**.",
        "",
        "| Cell | Rule up-down | 95% CI | Holm p | Random-noop 90% CI | CR3 pass | Rule/model ratio |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    ratios = {
        row["cell"]: row["descriptive_efficiency_ratio"]
        for row in metrics["descriptive_efficiency_relative_to_sealed_cr1"][
            "cells"
        ]
    }
    for cell in metrics["cells"]:
        effect = cell["contrasts"]["up_minus_down"]
        random_ci = cell["random_noop_equivalence"]["bootstrap_ci90"]
        ratio = ratios[cell["cell"]]
        ratio_text = "NA" if ratio is None else f"{ratio:.3f}"
        lines.append(
            f"| {cell['cell']} | {effect['estimate']:+.6f} | {effect['bootstrap_ci95']} | "
            f"{cell['up_down_randomization_p_holm']:.6g} | {random_ci} | "
            f"{cell['cr3_registered_cell_pass']} | {ratio_text} |"
        )
    lines.extend(
        [
            "",
            "The rule, orientation, cohort, seeds, arms, threshold, and CR3-specific inference gate were frozen before these matrices existed. Candidates were not pooled.",
            "",
            "The efficiency ratio to the sealed CR1 model-guided effect is descriptive and was not a gate.",
            "",
            "This simulated-process result cannot establish strict-eight control, life, agency, biological memory, autonomous organization, real chemistry, or Phi/PhiID intervention.",
            "",
        ]
    )
    lay = "\n".join(
        [
            "# CR3 confirmation in plain language",
            "",
            "A simple physical rule looked only at how strongly each molecular type helps catalyze the molecules already present. It chose one one-molecule change expected to destabilize heredity and the opposite change expected to stabilize it. Random changes and no change were controls.",
            "",
            (
                "The simple rule passed every prewritten test in both simulator candidates and both independent branch halves."
                if metrics["confirmation_gate_pass"]
                else "The simple rule did not pass every prewritten test across both simulator candidates and both branch halves. Individual effects remain reported, but the full CR3 claim is not confirmed."
            ),
            "",
            "This concerns causal control of a narrowly defined simulated break-and-renewal event. It is not evidence that the assemblies are alive or possess biological memory or agency.",
            "",
        ]
    )
    return "\n".join(lines), lay


def _append_sealed_ledger(
    output: Path, registration_id: str, metrics: dict[str, Any]
) -> None:
    path = ROOT / "INTERVENTION_RESULTS_LEDGER.md"
    text = path.read_text(encoding="utf-8").rstrip() + "\n"
    marker = f"<!-- sealed-cr3-confirmation-{registration_id} -->"
    if marker in text:
        return
    rows = [
        "",
        marker,
        "## Full CR3 outgoing physical-rule confirmation sealed",
        "",
        f"- Registration: `{registration_id}`.",
        f"- Result: `{output.relative_to(ROOT)}`.",
        f"- Full four-cell CR3 gate: **{metrics['confirmation_gate_pass']}**.",
        f"- Exact replay: **{metrics['integrity_gates']['exact_replay']}**.",
        "- Mandatory review stop observed; CR4 was not launched automatically.",
        "",
    ]
    path.write_text(text + "\n".join(rows), encoding="utf-8")


def run(
    registration_directory: Path = DEFAULT_REGISTRATION,
    output: Path = DEFAULT_OUTPUT,
    work: Path = DEFAULT_WORK,
    workers: int = min(os.cpu_count() or 1, 14),
    cpu_budget_hours: float = MINIMUM_CPU_BUDGET_HOURS,
) -> None:
    if workers < 1:
        raise ValueError("workers must be positive")
    registration_directory = registration_directory.resolve()
    output = output.resolve()
    work = work.resolve()
    registration = verify_registration(registration_directory)
    verify_checksums(DEFAULT_SMOKE)
    _prepare_campaign(work, output, registration, cpu_budget_hours)
    spec = phase_spec()
    current_experiment = experiment(spec)
    print(
        f"[cr3 confirmation 1/8] Building {MATRICES} fresh matrices and "
        f"{2 * MATRICES * len(LANDMARKS)} states",
        flush=True,
    )
    _status(work, "running", "building_natural_states", cpu_budget_hours)
    with threadpool_limits(limits=1):
        cases = build_cohort(
            current_experiment, LABEL, current_experiment.confirmation
        )
    if len(cases) != 2 * MATRICES * len(LANDMARKS):
        raise AssertionError("CR3 confirmation cohort is incomplete")
    model = registration_directory / "frozen_full_predictor.npz"
    futures = len(cases) * len(spec.arms) * BRANCHES
    print(
        f"[cr3 confirmation 2/8] Selecting physical-rule edits and shooting {futures:,} F12 futures",
        flush=True,
    )
    _status(work, "running", "selection_and_primary_futures", cpu_budget_hours)
    generated = run_phase_batches(
        cases,
        current_experiment,
        spec,
        model,
        registration["registration_id"],
        work / "generate",
        workers,
        "generate",
    )
    print(f"[cr3 confirmation 3/8] Replaying all {futures:,} futures", flush=True)
    _status(work, "running", "exact_replay", cpu_budget_hours)
    replayed = run_phase_batches(
        cases,
        current_experiment,
        spec,
        model,
        registration["registration_id"],
        work / "replay",
        workers,
        "replay",
    )
    replay = base.replay_audit(generated, replayed)
    if not replay["state_edit_endpoint_and_process_digests_exact"]:
        raise AssertionError("CR3 confirmation exact replay failed")
    arrays = base._outcome_arrays(cases, generated, spec)
    draws = generate_inference_draws(
        MATRICES,
        BOOTSTRAP_REPETITIONS,
        RANDOMIZATION_REPETITIONS,
        np.random.default_rng(derive_seed(SEEDS["bootstrap"], f"{LABEL}.bootstrap")),
        np.random.default_rng(
            derive_seed(SEEDS["randomization"], f"{LABEL}.randomization")
        ),
    )
    print("[cr3 confirmation 4/8] Computing frozen whole-matrix inference", flush=True)
    _status(work, "running", "whole_matrix_inference", cpu_budget_hours)
    metrics, matrix_rows = compute_one_shot_inference(
        cases,
        spec.arms,
        arrays["targets"],
        arrays["predictions"],
        draws,
        up_arm="RULE_UP",
        down_arm="RULE_DOWN",
        equivalence_margin=EQUIVALENCE_MARGIN,
        random_ratio_limit=base.RANDOM_RATIO_LIMIT,
    )
    add_cr3_gate_fields(metrics)
    secondary = base._secondary_descriptives(cases, arrays, spec)
    print("[cr3 confirmation 5/8] Writing and readback-checking artifacts", flush=True)
    _status(work, "running", "artifact_write_and_readback", cpu_budget_hours)
    with _atomic_destination(output) as destination:
        np.savez_compressed(destination / "branch_arrays.npz", **arrays)
        base._write_branch_table(destination / "branches.csv.gz", cases, generated)
        base._write_state_artifacts(destination, cases, generated, arrays)
        base._write_selection_artifacts(destination, cases, generated, spec)
        base._write_inference_arrays(
            destination / "inference_arrays.npz", draws, metrics
        )
        pd.DataFrame(matrix_rows).to_csv(
            destination / "matrix_effects.csv", index=False
        )
        readback = _readback_metrics(
            destination, cases, spec, metrics, matrix_rows
        )
        integrity = {
            "exact_replay": replay[
                "state_edit_endpoint_and_process_digests_exact"
            ],
            "artifact_readback_exact": bool(
                readback["primary_metrics_exact"]
                and readback["matrix_effects_exact"]
            ),
        }
        metrics["integrity_gates"] = integrity
        metrics["confirmation_gate_pass"] = bool(
            metrics["cr3_all_four_cells_scientific_pass"]
            and all(integrity.values())
        )
        _add_cr1_efficiency(metrics)
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
        (destination / "rule_definition.json").write_text(
            json.dumps(protocol()["rule"], indent=2, sort_keys=True) + "\n",
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
                    "prospective outgoing physical-rule control of Codex JOINT_BREAK_RUN3"
                ]
                if metrics["confirmation_gate_pass"]
                else []
            ),
            "failed_predictions": (
                []
                if metrics["confirmation_gate_pass"]
                else ["full CR3 four-cell physical-rule confirmation gate"]
            ),
            "unresolved": [
                "catalytic-network surgery confirmation",
                "resistance versus resilience under molecular edits",
                "parameter-regime transfer",
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
            "branches_per_arm_state": BRANCHES,
            "primary_futures": futures,
            "replay_futures": futures,
            "full_four_cell_cr3_gate": metrics["confirmation_gate_pass"],
            "exact_replay": integrity["exact_replay"],
            "complete_readback_exact": integrity["artifact_readback_exact"],
            "rule_expression": "x @ beta == beta.T @ x",
            "declared_cpu_budget_hours": cpu_budget_hours,
            "no_rule_search_refitting_or_threshold_change": True,
            "no_future_retry_or_matrix_replacement": True,
            "mandatory_stop_after_this_stage": True,
            "cr4_launched": False,
            "runtime": _runtime_manifest(),
        }
        (destination / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_checksums(destination)
    verify_checksums(output)
    _append_sealed_ledger(output, registration["registration_id"], metrics)
    _status(work, "sealed_complete", "mandatory_review_stop", cpu_budget_hours)
    print("[cr3 confirmation 6/8] Result checksum sealed", flush=True)
    print("[cr3 confirmation 7/8] Durable ledger and status updated", flush=True)
    print("[cr3 confirmation 8/8] STOPPED; CR4 not launched", flush=True)


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
    run_parser.add_argument("--cpu-budget-hours", type=float, required=True)
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
            json.dumps(
                verify_registration(args.registration), indent=2, sort_keys=True
            )
        )
    elif args.command == "smoke":
        smoke(args.registration, args.output)
    elif args.command == "run":
        run(
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


if __name__ == "__main__":
    main()
