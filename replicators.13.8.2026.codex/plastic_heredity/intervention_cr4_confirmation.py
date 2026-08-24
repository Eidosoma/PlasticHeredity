"""Full prospective CR4 fixed-composition catalytic-network confirmation."""

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

from . import intervention_p3c as p3c
from . import intervention_replication as base
from .config import CANDIDATES, CohortConfig, ExperimentConfig
from .experiment import StateCase, _json_ready, _runtime_manifest, build_cohort
from .intervention_core import (
    BetaSurgery,
    FrozenFullPredictor,
    random_beta_surgery,
    simulate_one_shot,
)
from .intervention_metrics import (
    _bootstrap_means,
    _interval,
    _matrix_means,
    compute_one_shot_inference,
    generate_inference_draws,
    holm_adjust,
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
CR3_RESULT = RESULT_ROOT / "cr3_physical_rule_confirmation"
P3B_RESULT = RESULT_ROOT / "p3b_beta_surgery_dose_bridge"
P3C_RESULT = RESULT_ROOT / "p3c_throughput_confirmation"
P4_RESULT = RESULT_ROOT / "p4_shared_break_recovery"
FABLE_ARCHIVE = RESULT_ROOT / "p3c_fable_response"
DEFAULT_VALIDATION = RESULT_ROOT / "cr4_confirmation_validation"
DEFAULT_REGISTRATION = RESULT_ROOT / "cr4_confirmation_registration"
DEFAULT_SMOKE = RESULT_ROOT / "cr4_confirmation_smoke"
DEFAULT_OUTPUT = RESULT_ROOT / "cr4_beta_surgery_confirmation"
DEFAULT_WORK = RESULT_ROOT / ".cr4_beta_surgery_confirmation_work"

DOCUMENT = "CODEX_INTERVENTION_CR4_CONFIRMATION_PREREGISTRATION.md"
SOURCE_FILES = (
    DOCUMENT,
    "plastic_heredity/intervention_cr4_confirmation.py",
    "tests/test_intervention_cr4_confirmation.py",
    "plastic_heredity/intervention_p3c.py",
    "plastic_heredity/intervention_p4.py",
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

PROGRAM_FORMAT = "codex-intervention-cr4-confirmation-v1"
VALIDATION_FORMAT = "codex-intervention-cr4-confirmation-validation-v1"
REGISTRATION_FORMAT = "codex-intervention-cr4-confirmation-registration-v1"
RESULT_FORMAT = "codex-intervention-cr4-confirmation-result-v1"
CHECKPOINT_FORMAT = "codex-intervention-cr4-confirmation-checkpoint-v1"
LABEL = "INTCR4_FIXED_COMPOSITION_BETA_CONFIRMATION_V1"

MATRICES = 200
BRANCHES = 64
LANDMARKS = (20, 35, 50, 65, 80)
HORIZON = 12
BOOTSTRAP_REPETITIONS = 4_096
RANDOMIZATION_REPETITIONS = 4_096
EQUIVALENCE_MARGIN = 0.025
SURGERY_NORM_FRACTION = 0.5
TIGHTEN_FACTOR = 1.5
LOOSEN_FACTOR = 1.0 / 1.5
MINIMUM_CPU_BUDGET_HOURS = 20.0
MINIMUM_FREE_DISK_BYTES = 3_000_000_000

ARMS = (
    "LOOSEN",
    "TIGHTEN",
    "GLOBAL_RANDOM_SURGERY",
    "THROUGHPUT_NEUTRAL_TOPOLOGY",
    "NOOP",
)


def _seed(name: str) -> str:
    return hashlib.sha256(
        f"codex-clean-room-cr4-full-confirmation-v1::{name}".encode("utf-8")
    ).hexdigest()


SEEDS = {
    name: _seed(name)
    for name in (
        "validation",
        "smoke_cohort",
        "smoke_global_selection",
        "smoke_topology_selection",
        "smoke_future",
        "cohort",
        "global_selection",
        "topology_selection",
        "future",
        "bootstrap",
        "randomization",
        "replay",
    )
}


@dataclass(frozen=True)
class CR4Spec:
    role: str
    matrices: int
    branches: int
    landmarks: tuple[int, ...]
    cohort_seed: str
    global_selection_seed: str
    topology_selection_seed: str
    future_seed: str
    bootstrap_seed: str
    randomization_seed: str
    arms: tuple[str, ...] = ARMS

    @property
    def phase(self) -> str:
        return "cr4_beta_surgery_confirmation"


def source_hashes() -> dict[str, str]:
    return {name: sha256_file(ROOT / name) for name in SOURCE_FILES}


def phase_spec() -> CR4Spec:
    return CR4Spec(
        role="full prospective fixed-composition catalytic-network confirmation",
        matrices=MATRICES,
        branches=BRANCHES,
        landmarks=LANDMARKS,
        cohort_seed=SEEDS["cohort"],
        global_selection_seed=SEEDS["global_selection"],
        topology_selection_seed=SEEDS["topology_selection"],
        future_seed=SEEDS["future"],
        bootstrap_seed=SEEDS["bootstrap"],
        randomization_seed=SEEDS["randomization"],
    )


def experiment(spec: CR4Spec | None = None) -> ExperimentConfig:
    selected = phase_spec() if spec is None else spec
    cohort = CohortConfig(selected.matrices, selected.branches, selected.landmarks)
    return ExperimentConfig(
        development=cohort,
        confirmation=cohort,
        horizon=HORIZON,
        bootstrap_repetitions=BOOTSTRAP_REPETITIONS,
        permutation_repetitions=RANDOMIZATION_REPETITIONS,
        regenerate_confirmation=True,
        master_seed=selected.cohort_seed,
    )


def _global_selection_seed(spec: CR4Spec, case: StateCase) -> int:
    return derive_seed(
        spec.global_selection_seed,
        f"{LABEL}.selection.global_random",
        case.candidate,
        case.matrix_id,
        case.landmark,
    )


def _topology_selection_seed(spec: CR4Spec, case: StateCase) -> int:
    return derive_seed(
        spec.topology_selection_seed,
        f"{LABEL}.selection.throughput_neutral_topology",
        case.candidate,
        case.matrix_id,
        case.landmark,
    )


def _future_seed(spec: CR4Spec, case: StateCase, branch: int) -> int:
    return derive_seed(
        spec.future_seed,
        f"{LABEL}.future",
        case.candidate,
        case.matrix_id,
        case.landmark,
        branch,
    )


def _present_block(
    composition: NDArray, beta: NDArray
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64]]:
    values = np.asarray(composition)
    matrix = np.asarray(beta, dtype=np.float64)
    if values.ndim != 1 or matrix.shape != (values.size, values.size):
        raise ValueError("composition and beta dimensions differ")
    if np.any(values < 0) or int(values.sum()) <= 0:
        raise ValueError("composition must be nonnegative and nonempty")
    if not np.isfinite(matrix).all() or np.any(matrix <= 0.0):
        raise ValueError("beta must be finite and strictly positive")
    present = np.flatnonzero(values > 0).astype(np.int64)
    rows, columns = np.meshgrid(present, present, indexing="ij")
    flat = np.ravel_multi_index((rows.ravel(), columns.ravel()), matrix.shape)
    return present, flat.astype(np.int64), matrix.ravel()[flat].copy()


def select_surgeries(
    composition: NDArray,
    beta: NDArray,
    global_rng: np.random.Generator,
    topology_rng: np.random.Generator,
) -> tuple[BetaSurgery | None, ...]:
    """Construct all frozen CR4 arms without using future randomness."""

    present, _flat, _before = _present_block(composition, beta)
    if present.size < 2:
        return tuple(None for _ in ARMS)
    global_surgery = replace(
        random_beta_surgery(
            composition, beta, SURGERY_NORM_FRACTION, global_rng
        ),
        name="GLOBAL_RANDOM_SURGERY",
    )
    by_name: dict[str, BetaSurgery | None] = {
        "LOOSEN": p3c.multiplicative_surgery(
            composition, beta, LOOSEN_FACTOR, "LOOSEN"
        ),
        "TIGHTEN": p3c.multiplicative_surgery(
            composition, beta, TIGHTEN_FACTOR, "TIGHTEN"
        ),
        "GLOBAL_RANDOM_SURGERY": global_surgery,
        "THROUGHPUT_NEUTRAL_TOPOLOGY": p3c.throughput_neutral_pp_surgery(
            composition,
            beta,
            topology_rng,
            name="THROUGHPUT_NEUTRAL_TOPOLOGY",
        ),
        "NOOP": None,
    }
    return tuple(by_name[arm] for arm in ARMS)


def protocol() -> dict[str, Any]:
    spec = phase_spec()
    value: dict[str, Any] = {
        "format": PROGRAM_FORMAT,
        "status": "frozen_before_any_cr4_confirmation_matrix",
        "endpoint": "JOINT_BREAK_RUN3 within F12",
        "predecessors": {
            "full_cr3_pass_required": True,
            "p3_p3b_p3c_and_p4_results_preserved": True,
            "p3c_failed_omnibus_gate_not_rescued": True,
            "predecessor_outcomes_seen_before_registration": True,
            "controls_motivated_by_predecessor_geometry": True,
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
        "arms": {
            "order": list(spec.arms),
            "LOOSEN": {
                "location": "all and only beta[P,P]",
                "factor": LOOSEN_FACTOR,
                "frobenius_fraction": 1.0 / 3.0,
            },
            "TIGHTEN": {
                "location": "all and only beta[P,P]",
                "factor": TIGHTEN_FACTOR,
                "frobenius_fraction": SURGERY_NORM_FRACTION,
            },
            "GLOBAL_RANDOM_SURGERY": {
                "location_count": "exactly |P|^2 distinct whole-matrix edges",
                "location_selection_independent_of_present_identities": True,
                "balanced_log": True,
                "strictly_positive": True,
                "frobenius_fraction_of_present_block": SURGERY_NORM_FRACTION,
                "required_noop_equivalence": True,
            },
            "THROUGHPUT_NEUTRAL_TOPOLOGY": {
                "location": "all and only beta[P,P]",
                "strictly_positive": True,
                "frobenius_fraction": SURGERY_NORM_FRACTION,
                "launch_throughput": "x.T @ beta @ x exactly preserved",
                "separate_classification_not_required_null": True,
            },
            "NOOP": {"changed_edges": 0},
            "singleton_contract": "all-arm structural NOOP when |P| < 2",
            "fable_strength_pair_log_symmetric": True,
            "fable_strength_pair_frobenius_symmetric": False,
        },
        "futures": {
            "branches_per_arm_state": BRANCHES,
            "horizon": HORIZON,
            "halves": {"A": [0, 31], "B": [32, 63]},
            "primary_futures": 640_000,
            "replay_futures": 640_000,
            "common_random_streams": True,
            "future_seed_excludes_arm": True,
            "selection_streams_separate_from_future": True,
            "no_retries_or_matrix_replacement": True,
        },
        "inference": {
            "unit": "whole catalytic matrix",
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "randomization_repetitions": RANDOMIZATION_REPETITIONS,
            "strength_holm_family": ["c02_A", "c02_B", "c03_A", "c03_B"],
            "target_contrast": "LOOSEN - TIGHTEN",
            "target_bootstrap_lower_positive": True,
            "target_holm_randomization_p_below": 0.05,
            "global_random_noop_tost_margin": [
                -EQUIVALENCE_MARGIN,
                EQUIVALENCE_MARGIN,
            ],
            "global_random_tost_method": (
                "90% whole-matrix bootstrap interval strictly inside margin"
            ),
            "side_arm_and_effect_ratio_checks_descriptive_only": True,
            "topology_family": "two-sided Holm across four cells",
            "topology_classifications": [
                "reproducible_directional_topology_effect",
                "negligible_within_0.025",
                "inconclusive",
            ],
            "topology_cannot_rescue_or_invalidate_strength_gate": True,
            "exact_surgery_replay_and_readback_required": True,
        },
        "operational": {
            "estimated_cpu_hours": [14.0, 17.0],
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


def add_cr4_gate_fields(metrics: dict[str, Any]) -> dict[str, Any]:
    """Apply only the preregistered CR4 strength and location-control gates."""

    for cell in metrics["cells"]:
        effect = cell["contrasts"]["up_minus_down"]
        gates = {
            "loosen_minus_tighten_positive": effect["estimate"] > 0.0,
            "loosen_minus_tighten_bootstrap_lower_positive": (
                effect["bootstrap_ci95"][0] > 0.0
            ),
            "holm_randomization_below_0_05": (
                cell["up_down_randomization_p_holm"] < 0.05
            ),
            "global_random_tost_equivalent_to_noop": cell[
                "random_noop_equivalence"
            ]["tost_equivalent"],
        }
        cell["cr4_registered_gates"] = gates
        cell["cr4_registered_cell_pass"] = bool(all(gates.values()))
    metrics["cr4_all_four_cells_scientific_pass"] = bool(
        all(cell["cr4_registered_cell_pass"] for cell in metrics["cells"])
    )
    return metrics


def _phase_worker(
    arguments: tuple[StateCase, ExperimentConfig, CR4Spec, str]
) -> base.PhaseBatch:
    case, current_experiment, spec, model_path = arguments
    limiter = threadpool_limits(limits=1)
    try:
        predictor = FrozenFullPredictor.load(model_path)
        surgeries = select_surgeries(
            case.snapshot.composition,
            case.beta,
            np.random.default_rng(_global_selection_seed(spec, case)),
            np.random.default_rng(_topology_selection_seed(spec, case)),
        )
        predictions = np.asarray(
            [
                predictor.predict_snapshot(
                    case.candidate,
                    case.snapshot,
                    case.beta if surgery is None else surgery.beta,
                    current_experiment.gard,
                )
                for surgery in surgeries
            ],
            dtype=np.float64,
        )
        outcomes: list[list[Any]] = [[] for _ in spec.arms]
        for branch in range(spec.branches):
            seed = _future_seed(spec, case, branch)
            for arm_index, surgery in enumerate(surgeries):
                outcomes[arm_index].append(
                    simulate_one_shot(
                        case.snapshot,
                        case.beta if surgery is None else surgery.beta,
                        case.candidate,
                        current_experiment.gard,
                        HORIZON,
                        np.random.default_rng(seed),
                        None,
                    )
                )
        frozen = tuple(tuple(arm) for arm in outcomes)
        if all(surgery is None for surgery in surgeries):
            for branch in range(spec.branches):
                if len(
                    {
                        frozen[arm][branch].record_digest
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
            outcomes=frozen,
        )
    finally:
        limiter.restore_original_limits()


def _checkpoint_contract(
    cases: list[StateCase], spec: CR4Spec, registration_id: str, stage: str
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "registration_id": registration_id,
        "scientific_label": LABEL,
        "phase": spec.phase,
        "stage": stage,
        "matrices": spec.matrices,
        "branches": spec.branches,
        "landmarks": list(spec.landmarks),
        "horizon": HORIZON,
        "arms": list(spec.arms),
        "case_ids": [case.state_id for case in cases],
        "case_digests": [base._snapshot_digest(case) for case in cases],
        "future_seed": spec.future_seed,
        "future_seed_includes_arm": False,
        "global_selection_seed": spec.global_selection_seed,
        "topology_selection_seed": spec.topology_selection_seed,
        "source_hashes": source_hashes(),
    }
    value["contract_id"] = _canonical_digest(_json_ready(value))
    return value


def run_phase_batches(
    cases: list[StateCase],
    current_experiment: ExperimentConfig,
    spec: CR4Spec,
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
                raise ValueError(f"invalid CR4 checkpoint {path}")
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
        raise AssertionError("checkpointed CR4 phase has missing states")
    return [batch for batch in batches if batch is not None]


def _geometry_audit(
    cases: list[StateCase], batches: list[base.PhaseBatch], spec: CR4Spec
) -> tuple[dict[str, NDArray], pd.DataFrame, dict[str, Any]]:
    throughput = np.empty((len(cases), len(spec.arms)), dtype=np.float64)
    log_ratio = np.empty_like(throughput)
    rows: list[dict[str, Any]] = []
    checks = {
        "all_beta_strictly_positive": True,
        "all_nonnoop_norms_exact": True,
        "targeted_arms_all_and_only_present_present": True,
        "targeted_factors_exact": True,
        "global_locations_distinct_and_count_matched": True,
        "global_norm_matches_tighten": True,
        "topology_all_and_only_present_present": True,
        "topology_norm_matches_tighten": True,
        "topology_throughput_exact": True,
        "noop_unchanged": True,
        "singleton_all_arm_structural_noop": True,
    }
    structural_states = 0
    maximum_norm_relative_error = 0.0
    maximum_topology_throughput_error = 0.0
    for state_index, (case, batch) in enumerate(zip(cases, batches, strict=True)):
        composition = case.snapshot.composition
        present, pp_flat, pp_before = _present_block(composition, case.beta)
        pp_set = set(pp_flat.tolist())
        baseline = p3c.catalytic_throughput(composition, case.beta)
        tighten_norm = SURGERY_NORM_FRACTION * float(np.linalg.norm(pp_before))
        singleton = present.size < 2
        if singleton:
            structural_states += 1
            checks["singleton_all_arm_structural_noop"] &= all(
                surgery is None for surgery in batch.surgeries
            )
        for arm_index, (arm, surgery) in enumerate(
            zip(spec.arms, batch.surgeries, strict=True)
        ):
            altered = case.beta if surgery is None else surgery.beta
            observed_throughput = p3c.catalytic_throughput(composition, altered)
            throughput[state_index, arm_index] = observed_throughput
            log_ratio[state_index, arm_index] = np.log(
                observed_throughput / baseline
            )
            changed_set = (
                set()
                if surgery is None
                else set(np.asarray(surgery.flat_indices, dtype=np.int64).tolist())
            )
            requested = 0.0 if surgery is None else float(surgery.requested_norm)
            observed = 0.0 if surgery is None else float(surgery.observed_norm)
            relative_error = (
                0.0
                if requested == 0.0
                else abs(observed - requested) / requested
            )
            maximum_norm_relative_error = max(
                maximum_norm_relative_error, relative_error
            )
            if surgery is not None:
                checks["all_beta_strictly_positive"] &= bool(
                    np.isfinite(surgery.beta).all() and np.all(surgery.beta > 0.0)
                )
                checks["all_nonnoop_norms_exact"] &= bool(
                    abs(observed - requested)
                    <= 1e-11 * max(1.0, requested)
                )
            if arm in ("LOOSEN", "TIGHTEN") and surgery is not None:
                factor = LOOSEN_FACTOR if arm == "LOOSEN" else TIGHTEN_FACTOR
                checks["targeted_arms_all_and_only_present_present"] &= (
                    changed_set == pp_set
                )
                checks["targeted_factors_exact"] &= bool(
                    np.array_equal(surgery.after, surgery.before * factor)
                )
            elif arm == "GLOBAL_RANDOM_SURGERY" and surgery is not None:
                checks["global_locations_distinct_and_count_matched"] &= bool(
                    len(changed_set) == pp_before.size
                    and len(changed_set) == len(surgery.flat_indices)
                )
                checks["global_norm_matches_tighten"] &= bool(
                    abs(requested - tighten_norm)
                    <= 1e-11 * max(1.0, tighten_norm)
                )
            elif arm == "THROUGHPUT_NEUTRAL_TOPOLOGY" and surgery is not None:
                checks["topology_all_and_only_present_present"] &= (
                    changed_set == pp_set
                )
                checks["topology_norm_matches_tighten"] &= bool(
                    abs(requested - tighten_norm)
                    <= 1e-11 * max(1.0, tighten_norm)
                )
                error = abs(observed_throughput - baseline)
                maximum_topology_throughput_error = max(
                    maximum_topology_throughput_error, error
                )
                tolerance = max(
                    p3c.NEUTRAL_ABSOLUTE_TOLERANCE,
                    p3c.NEUTRAL_RELATIVE_TOLERANCE * abs(baseline),
                )
                checks["topology_throughput_exact"] &= error <= tolerance
            elif arm == "NOOP":
                checks["noop_unchanged"] &= surgery is None
            rows.append(
                {
                    "state_id": case.state_id,
                    "candidate": case.candidate,
                    "matrix_id": case.matrix_id,
                    "landmark": case.landmark,
                    "arm": arm,
                    "present_types": int(present.size),
                    "present_present_edges": int(pp_before.size),
                    "structural_no_action": bool(singleton),
                    "changed_edges": len(changed_set),
                    "all_and_only_present_present": bool(
                        surgery is not None and changed_set == pp_set
                    ),
                    "requested_frobenius_norm": requested,
                    "observed_frobenius_norm": observed,
                    "norm_relative_error": relative_error,
                    "throughput_noop": baseline,
                    "throughput_arm": observed_throughput,
                    "throughput_difference": observed_throughput - baseline,
                    "log_throughput_ratio": log_ratio[state_index, arm_index],
                    "minimum_beta_after": float(altered.min()),
                }
            )
    summary: dict[str, Any] = {
        "format": "codex-intervention-cr4-surgery-audit-v1",
        "states": len(cases),
        "rows": len(rows),
        "structural_no_action_states": structural_states,
        "checks": checks,
        "maximum_surgery_norm_relative_error": maximum_norm_relative_error,
        "maximum_topology_throughput_absolute_error": (
            maximum_topology_throughput_error
        ),
        "no_clipping": True,
        "global_locations_sampled_without_replacement": True,
    }
    summary["all_audits_pass"] = bool(all(checks.values()))
    return (
        {"throughput": throughput, "log_throughput_ratio": log_ratio},
        pd.DataFrame(rows),
        summary,
    )


def _two_sided_sign_p(
    values: NDArray, signs: NDArray
) -> tuple[float, NDArray[np.float64]]:
    data = np.asarray(values, dtype=np.float64)
    null = (np.asarray(signs, dtype=np.float64) * data).mean(axis=1)
    observed = abs(float(data.mean()))
    p = float((1 + np.count_nonzero(np.abs(null) >= observed)) / (len(null) + 1))
    return p, null


def compute_topology_inference(
    cases: list[StateCase],
    targets: NDArray,
    draws: dict[str, NDArray],
    spec: CR4Spec,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, NDArray]]:
    values = np.asarray(targets, dtype=np.float64)
    arm_index = {arm: index for index, arm in enumerate(spec.arms)}
    bootstrap = np.asarray(draws["bootstrap_indices"], dtype=np.int64)
    signs = np.asarray(draws["randomization_signs"], dtype=np.float64)
    matrix_order = np.arange(spec.matrices, dtype=np.int64)
    cells: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    stored: dict[str, NDArray] = {}
    raw_p: list[float] = []
    for candidate in CANDIDATES:
        selected = np.asarray(
            [case.candidate == candidate for case in cases], dtype=bool
        )
        selected_cases = [case for case in cases if case.candidate == candidate]
        ids = np.asarray([case.matrix_id for case in selected_cases], dtype=np.int64)
        if not np.array_equal(np.unique(ids), matrix_order):
            raise ValueError(f"candidate {candidate} lacks a complete CR4 cohort")
        candidate_values = values[selected]
        for half, branch_slice in (
            ("A", slice(0, spec.branches // 2)),
            ("B", slice(spec.branches // 2, spec.branches)),
        ):
            q = candidate_values[:, :, branch_slice].mean(axis=2)
            state_effect = (
                q[:, arm_index["THROUGHPUT_NEUTRAL_TOPOLOGY"]]
                - q[:, arm_index["NOOP"]]
            )
            matrix_effect = _matrix_means(state_effect, ids, matrix_order)
            boot = _bootstrap_means(matrix_effect, bootstrap)
            p_raw, null = _two_sided_sign_p(matrix_effect, signs)
            raw_p.append(p_raw)
            key = f"c{candidate}_{half}"
            stored[f"{key}__topology_bootstrap"] = boot
            stored[f"{key}__topology_randomization"] = null
            ci95 = _interval(boot)
            ci90 = _interval(boot, alpha=0.10)
            cells.append(
                {
                    "cell": key,
                    "candidate": candidate,
                    "branch_half": half,
                    "estimate": float(matrix_effect.mean()),
                    "bootstrap_ci95": ci95,
                    "bootstrap_ci90": ci90,
                    "randomization_p_raw_two_sided": p_raw,
                    "ci95_excludes_zero": bool(
                        ci95[0] > 0.0 or ci95[1] < 0.0
                    ),
                    "equivalent_margin_0_025": bool(
                        ci90[0] > -EQUIVALENCE_MARGIN
                        and ci90[1] < EQUIVALENCE_MARGIN
                    ),
                    "matrices_positive": int(np.count_nonzero(matrix_effect > 0.0)),
                    "matrices_negative": int(np.count_nonzero(matrix_effect < 0.0)),
                    "matrices_zero": int(np.count_nonzero(matrix_effect == 0.0)),
                }
            )
            for position, matrix_id in enumerate(matrix_order):
                rows.append(
                    {
                        "cell": key,
                        "candidate": candidate,
                        "branch_half": half,
                        "matrix_id": int(matrix_id),
                        "topology_minus_noop": float(matrix_effect[position]),
                    }
                )
    adjusted = holm_adjust(raw_p)
    for cell, adjusted_p in zip(cells, adjusted, strict=True):
        cell["randomization_p_holm_two_sided"] = float(adjusted_p)
    estimates = [cell["estimate"] for cell in cells]
    same_sign = bool(
        all(value > 0.0 for value in estimates)
        or all(value < 0.0 for value in estimates)
    )
    directional = bool(
        same_sign
        and all(cell["ci95_excludes_zero"] for cell in cells)
        and all(
            cell["randomization_p_holm_two_sided"] < 0.05 for cell in cells
        )
    )
    negligible = bool(
        all(cell["equivalent_margin_0_025"] for cell in cells)
    )
    classification = (
        "reproducible_directional_topology_effect"
        if directional
        else "negligible_within_0.025"
        if negligible
        else "inconclusive"
    )
    return (
        {
            "inference_family": "separate two-sided topology family",
            "contrast": "THROUGHPUT_NEUTRAL_TOPOLOGY - NOOP",
            "cells": cells,
            "classification": classification,
            "cannot_rescue_or_invalidate_primary_strength_gate": True,
        },
        rows,
        stored,
    )


def _inference(
    cases: list[StateCase],
    arrays: dict[str, NDArray],
    draws: dict[str, NDArray],
    spec: CR4Spec,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, NDArray],
]:
    metrics, matrix_rows = compute_one_shot_inference(
        cases,
        spec.arms,
        arrays["targets"],
        arrays["predictions"],
        draws,
        up_arm="LOOSEN",
        down_arm="TIGHTEN",
        random_arm="GLOBAL_RANDOM_SURGERY",
        noop_arm="NOOP",
        equivalence_margin=EQUIVALENCE_MARGIN,
        random_ratio_limit=base.RANDOM_RATIO_LIMIT,
    )
    add_cr4_gate_fields(metrics)
    topology, topology_rows, topology_arrays = compute_topology_inference(
        cases, arrays["targets"], draws, spec
    )
    metrics["topology"] = topology
    return metrics, matrix_rows, topology_rows, topology_arrays


def _write_inference_arrays(
    path: Path,
    draws: dict[str, NDArray],
    metrics: dict[str, Any],
    topology_arrays: dict[str, NDArray],
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
    arrays.update(topology_arrays)
    np.savez_compressed(path, **arrays)
    metrics["stored_inference_arrays"] = {
        "path": path.name,
        "bootstrap_indices_shape": stored["bootstrap_indices_shape"],
        "randomization_signs_shape": stored["randomization_signs_shape"],
        "all_strength_and_topology_arrays_stored": True,
        "topology_array_names": sorted(topology_arrays),
    }


def _normalized_stored_inference(
    metrics: dict[str, Any], topology_arrays: dict[str, NDArray]
) -> None:
    stored = metrics.pop("stored_inference_arrays")
    metrics["stored_inference_arrays"] = {
        "path": "inference_arrays.npz",
        "bootstrap_indices_shape": stored["bootstrap_indices_shape"],
        "randomization_signs_shape": stored["randomization_signs_shape"],
        "all_strength_and_topology_arrays_stored": True,
        "topology_array_names": sorted(topology_arrays),
    }


def validation_checks() -> dict[str, Any]:
    from .intervention_cr1_confirmation import SEEDS as CR1_SEEDS
    from .intervention_cr2_dose_response import SEEDS as CR2_SEEDS
    from .intervention_cr3_confirmation import SEEDS as CR3_SEEDS
    from .intervention_p3b_dose_bridge import SEED_DOMAINS as P3B_SEEDS
    from .intervention_p3c import SEED_DOMAINS as P3C_SEEDS
    from .intervention_p4 import SEEDS as P4_SEEDS

    inherited = base.validation_checks()
    all_prior = set(base.SEED_DOMAINS.values())
    for prior in (CR1_SEEDS, CR2_SEEDS, CR3_SEEDS, P3B_SEEDS, P3C_SEEDS, P4_SEEDS):
        all_prior.update(prior.values())

    composition = np.asarray([4, 2, 0, 2], dtype=np.int64)
    beta = np.asarray(
        [
            [1.0, 4.0, 2.0, 3.0],
            [7.0, 1.0, 5.0, 2.0],
            [8.0, 4.0, 1.0, 9.0],
            [2.0, 3.0, 6.0, 1.0],
        ],
        dtype=np.float64,
    )
    surgeries = select_surgeries(
        composition,
        beta,
        np.random.default_rng(101),
        np.random.default_rng(103),
    )
    by_name = dict(zip(ARMS, surgeries, strict=True))
    present, flat, before = _present_block(composition, beta)
    tighten = by_name["TIGHTEN"]
    loosen = by_name["LOOSEN"]
    global_surgery = by_name["GLOBAL_RANDOM_SURGERY"]
    topology = by_name["THROUGHPUT_NEUTRAL_TOPOLOGY"]
    if any(value is None for value in (tighten, loosen, global_surgery, topology)):
        raise AssertionError("validation fixture unexpectedly produced a no-op")
    assert tighten is not None and loosen is not None
    assert global_surgery is not None and topology is not None
    checks: dict[str, Any] = {
        "inherited_validation_pass": inherited["all_checks_passed"],
        "full_matrix_count": MATRICES == 200,
        "full_branch_count": BRANCHES == 64,
        "five_registered_landmarks": LANDMARKS == (20, 35, 50, 65, 80),
        "five_arm_primary_future_count": (
            2 * MATRICES * len(LANDMARKS) * len(ARMS) * BRANCHES == 640_000
        ),
        "arm_order_exact": phase_spec().arms == ARMS,
        "fable_factors_exact": (
            TIGHTEN_FACTOR == 1.5 and LOOSEN_FACTOR == 1.0 / 1.5
        ),
        "targeted_factor_application_exact": (
            np.array_equal(tighten.after, before * 1.5)
            and np.array_equal(loosen.after, before * LOOSEN_FACTOR)
        ),
        "global_count_distinct": (
            len(set(global_surgery.flat_indices.tolist())) == present.size**2
        ),
        "global_exact_norm": abs(
            global_surgery.observed_norm - 0.5 * np.linalg.norm(before)
        )
        <= 1e-11 * max(1.0, float(np.linalg.norm(before))),
        "topology_exact_norm": abs(
            topology.observed_norm - 0.5 * np.linalg.norm(before)
        )
        <= 1e-11 * max(1.0, float(np.linalg.norm(before))),
        "topology_all_and_only_pp": set(topology.flat_indices.tolist())
        == set(flat.tolist()),
        "topology_exact_throughput": np.isclose(
            p3c.catalytic_throughput(composition, topology.beta),
            p3c.catalytic_throughput(composition, beta),
            rtol=p3c.NEUTRAL_RELATIVE_TOLERANCE,
            atol=p3c.NEUTRAL_ABSOLUTE_TOLERANCE,
        ),
        "all_surgeries_positive": all(
            surgery is None or np.all(surgery.beta > 0.0)
            for surgery in surgeries
        ),
        "singleton_structural_noop": all(
            surgery is None
            for surgery in select_surgeries(
                np.asarray([8, 0, 0, 0]),
                beta,
                np.random.default_rng(107),
                np.random.default_rng(109),
            )
        ),
        "seed_domains_unique": len(SEEDS) == len(set(SEEDS.values())),
        "seeds_disjoint_from_prior_campaigns": set(SEEDS.values()).isdisjoint(
            all_prior
        ),
        "cr3_result_checksum_verified": True,
        "cr3_passed": True,
        "predecessor_checksums_verified": True,
        "no_scientific_cohort_generated": True,
    }
    for predecessor in (CR3_RESULT, P3B_RESULT, P3C_RESULT, P4_RESULT, FABLE_ARCHIVE):
        verify_checksums(predecessor)
    cr3_manifest = json.loads((CR3_RESULT / "manifest.json").read_text())
    checks["cr3_passed"] = bool(cr3_manifest["full_four_cell_cr3_gate"])

    fixture = {
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
    add_cr4_gate_fields(fixture)
    checks["cr4_gate_fixture_passes"] = fixture[
        "cr4_all_four_cells_scientific_pass"
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
                f"CR4 scientific artifact exists before validation: {scientific}"
            )
    validation = validation_checks()
    completed = subprocess.run(
        [str(ROOT / ".venv/bin/python"), "-m", "pytest", "-q"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
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
    print(f"CR4 confirmation validation sealed: {output}", flush=True)


def _append_registration_notice(registration_id: str) -> None:
    path = ROOT / "INTERVENTION_RESULTS_LEDGER.md"
    text = path.read_text(encoding="utf-8").rstrip() + "\n"
    marker = f"<!-- cr4-confirmation-registered-{registration_id} -->"
    if marker in text:
        return
    rows = [
        "",
        marker,
        "## Full CR4 fixed-composition beta-surgery confirmation registered",
        "",
        f"- Registration: `{registration_id}`.",
        "- The corrected Fable-strength pair is frozen as `beta[P,P] *= 1.5` versus `beta[P,P] /= 1.5`.",
        "- The exact-norm whole-matrix random arm is the required location control; fixed-throughput occupied-block topology is classified separately and is not assumed null.",
        "- 200 fresh matrices, both candidates, five landmarks, 64 F12 branches per arm, and complete replay.",
        "- Scientific CR4 matrices and futures at registration: **0**.",
        "- Status: sealed before scientific CR4 execution.",
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
        raise ValueError("CR4 confirmation validation did not pass")
    for scientific in (DEFAULT_OUTPUT, DEFAULT_WORK):
        if scientific.exists():
            raise FileExistsError(
                f"CR4 scientific artifact exists before registration: {scientific}"
            )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    predecessors = (CR3_RESULT, P3B_RESULT, P3C_RESULT, P4_RESULT, FABLE_ARCHIVE)
    for predecessor in predecessors:
        verify_checksums(predecessor)
    frozen = protocol()
    payload: dict[str, Any] = {
        "format": REGISTRATION_FORMAT,
        "protocol_id": frozen["protocol_id"],
        "source_hashes": source_hashes(),
        "validation_checksum_manifest_sha256": sha256_file(
            validation_directory / "SHA256SUMS"
        ),
        "predecessor_checksum_manifests": {
            str(path.relative_to(ROOT)): sha256_file(path / "SHA256SUMS")
            for path in predecessors
        },
        "frozen_model_sha256": sha256_file(
            ORIGINAL_REGISTRATION / "frozen_full_predictor.npz"
        ),
        "seed_registry": SEEDS,
        "scientific_matrices_at_registration": 0,
        "scientific_futures_at_registration": 0,
        "predecessor_outcomes_seen_before_registration": True,
        "predecessors_preserved_not_rescued": True,
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
    print(f"CR4 confirmation registration sealed: {payload['registration_id']}", flush=True)


def verify_registration(directory: Path = DEFAULT_REGISTRATION) -> dict[str, Any]:
    directory = directory.resolve()
    verify_checksums(directory)
    payload = json.loads((directory / "registration.json").read_text())
    if payload.get("format") != REGISTRATION_FORMAT:
        raise ValueError("invalid CR4 confirmation registration")
    unsigned = dict(payload)
    registration_id = unsigned.pop("registration_id", None)
    if registration_id is None or _canonical_digest(_json_ready(unsigned)) != registration_id:
        raise ValueError("invalid CR4 confirmation registration ID")
    if payload["source_hashes"] != source_hashes():
        raise ValueError("CR4 confirmation source changed after registration")
    if json.loads((directory / "protocol.json").read_text()) != protocol():
        raise ValueError("CR4 confirmation protocol changed after registration")
    if payload["seed_registry"] != SEEDS:
        raise ValueError("CR4 confirmation seed registry changed")
    if payload["frozen_model_sha256"] != sha256_file(
        directory / "frozen_full_predictor.npz"
    ) or payload["frozen_model_sha256"] != base.EXPECTED_MODEL_SHA256:
        raise ValueError("CR4 frozen predictor changed")
    for relative, expected in payload["predecessor_checksum_manifests"].items():
        predecessor = ROOT / relative
        verify_checksums(predecessor)
        if sha256_file(predecessor / "SHA256SUMS") != expected:
            raise ValueError(f"CR4 predecessor changed: {relative}")
    return payload


def smoke(
    registration_directory: Path = DEFAULT_REGISTRATION,
    output: Path = DEFAULT_SMOKE,
) -> None:
    registration = verify_registration(registration_directory)
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    smoke_spec = CR4Spec(
        role="non-scientific CR4 I/O and replay smoke",
        matrices=1,
        branches=2,
        landmarks=(5,),
        cohort_seed=SEEDS["smoke_cohort"],
        global_selection_seed=SEEDS["smoke_global_selection"],
        topology_selection_seed=SEEDS["smoke_topology_selection"],
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
        prefix="codex-cr4-confirmation-smoke-", dir=output.parent
    ) as temporary:
        with threadpool_limits(limits=1):
            cases = build_cohort(
                current_experiment, "INTCR4_CONFIRMATION_NONSCIENTIFIC_SMOKE", cohort
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
        _geometry, _rows, audit = _geometry_audit(cases, generated, smoke_spec)
        if not replay["state_edit_endpoint_and_process_digests_exact"]:
            raise AssertionError("CR4 confirmation smoke replay failed")
        if not audit["all_audits_pass"]:
            raise AssertionError("CR4 confirmation smoke surgery audit failed")
    with _atomic_destination(output) as destination:
        (destination / "manifest.json").write_text(
            json.dumps(
                {
                    "format": "codex-intervention-cr4-confirmation-smoke-v1",
                    "registration_id": registration["registration_id"],
                    "scientific_result": False,
                    "scientific_matrices": 0,
                    "scientific_futures": 0,
                    "surgery_io_checkpoint_and_replay_passed": True,
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
    print(f"CR4 confirmation smoke passed: {output}", flush=True)


def _status(
    work: Path,
    state: str,
    detail: str,
    cpu_budget_hours: float | None = None,
) -> None:
    value: dict[str, Any] = {
        "format": "codex-intervention-cr4-confirmation-status-v1",
        "phase": "cr4_beta_surgery_confirmation",
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
            f"CR4 confirmation requires at least {MINIMUM_CPU_BUDGET_HOURS:.1f} CPU-hours"
        )
    free = shutil.disk_usage(RESULT_ROOT).free
    if free < MINIMUM_FREE_DISK_BYTES:
        raise OSError(
            f"CR4 confirmation needs at least {MINIMUM_FREE_DISK_BYTES} free bytes; found {free}"
        )
    work.mkdir(parents=True, exist_ok=True)
    contract: dict[str, Any] = {
        "format": "codex-intervention-cr4-confirmation-campaign-v1",
        "registration_id": registration["registration_id"],
        "output": str(output),
        "matrices": MATRICES,
        "branches": BRANCHES,
        "landmarks": list(LANDMARKS),
        "arms": list(ARMS),
        "declared_cpu_budget_hours": cpu_budget_hours,
        "free_disk_bytes_at_launch": free,
        "source_hashes": source_hashes(),
    }
    contract["campaign_id"] = _canonical_digest(_json_ready(contract))
    path = work / "campaign_contract.json"
    if path.exists() and json.loads(path.read_text()) != _json_ready(contract):
        raise ValueError("CR4 work directory belongs to another campaign")
    if not path.exists():
        base._atomic_json(path, contract)
    _status(work, "running", "campaign_initialized", cpu_budget_hours)


def _readback(
    output: Path,
    cases: list[StateCase],
    spec: CR4Spec,
    expected: dict[str, Any],
    expected_rows: list[dict[str, Any]],
    expected_topology_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    with np.load(output / "branch_arrays.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    with np.load(output / "inference_arrays.npz", allow_pickle=False) as archive:
        draws = {
            "bootstrap_indices": archive["bootstrap_indices"],
            "randomization_signs": archive["randomization_signs"],
        }
    observed, rows, topology_rows, topology_arrays = _inference(
        cases, arrays, draws, spec
    )
    _normalized_stored_inference(observed, topology_arrays)
    metrics_exact = _json_ready(observed) == _json_ready(expected)
    rows_exact = _json_ready(rows) == _json_ready(expected_rows)
    topology_rows_exact = _json_ready(topology_rows) == _json_ready(
        expected_topology_rows
    )
    if not metrics_exact or not rows_exact or not topology_rows_exact:
        raise ValueError("CR4 written-artifact inference changed")
    return {
        "branch_arrays_reloaded": True,
        "inference_draws_reloaded": True,
        "primary_metrics_exact": metrics_exact,
        "matrix_effects_exact": rows_exact,
        "topology_matrix_effects_exact": topology_rows_exact,
        "cr4_specific_gate_and_topology_recomputed": True,
        "no_fitting_or_recalibration": True,
    }


def _reports(metrics: dict[str, Any]) -> tuple[str, str]:
    lines = [
        "# Full CR4 fixed-composition catalytic-network confirmation",
        "",
        f"Registered four-cell strength gate: **{metrics['confirmation_gate_pass']}**.",
        f"Fixed-throughput topology classification: **{metrics['topology']['classification']}**.",
        f"Exact replay: **{metrics['integrity_gates']['exact_replay']}**.",
        "",
        "| Cell | Loosen-tighten | 95% CI | Holm p | Global random-noop 90% CI | Strength pass | Topology-noop | Topology 95% CI | Topology Holm p |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    topology_by_cell = {
        cell["cell"]: cell for cell in metrics["topology"]["cells"]
    }
    for cell in metrics["cells"]:
        effect = cell["contrasts"]["up_minus_down"]
        random_ci = cell["random_noop_equivalence"]["bootstrap_ci90"]
        topology = topology_by_cell[cell["cell"]]
        lines.append(
            f"| {cell['cell']} | {effect['estimate']:+.6f} | {effect['bootstrap_ci95']} | "
            f"{cell['up_down_randomization_p_holm']:.6g} | {random_ci} | "
            f"{cell['cr4_registered_cell_pass']} | {topology['estimate']:+.6f} | "
            f"{topology['bootstrap_ci95']} | {topology['randomization_p_holm_two_sided']:.6g} |"
        )
    lines.extend(
        [
            "",
            "Every arm began with identical composition and observed history; only beta differed. The exact-norm whole-matrix location control and the occupied-block topology treatment answer different questions.",
            "",
            "The earlier P3/P3b/P3c/P4 results remain unchanged. CR4 cannot rescue their failed registered gates.",
            "",
            "This simulated-process result cannot establish strict-eight control, life, agency, biological memory, autonomous organization, real chemistry, or Phi/PhiID intervention.",
            "",
        ]
    )
    lay = "\n".join(
        [
            "# CR4 confirmation in plain language",
            "",
            "CR4 makes identical copies of each simulated assembly, keeps every molecule and all of its past history fixed, and changes only the catalytic rulebook connecting the molecular types.",
            "",
            (
                "Strengthening versus weakening the occupied catalytic web passed every prewritten test in both simulator candidates and both independent branch halves."
                if metrics["confirmation_gate_pass"]
                else "Strengthening versus weakening the occupied catalytic web did not pass every prewritten test across both simulator candidates and both branch halves."
            ),
            "",
            f"A separate control rearranged the occupied web while keeping its starting total support fixed. Its registered classification was {metrics['topology']['classification'].replace('_', ' ')}. That result is reported separately rather than being forced to behave like no intervention.",
            "",
            "This tests causal control of one narrow simulated break-and-renewal event. It does not show that the assemblies are alive, autonomous, or biologically remembering.",
            "",
        ]
    )
    return "\n".join(lines), lay


def _append_sealed_ledger(
    output: Path, registration_id: str, metrics: dict[str, Any]
) -> None:
    path = ROOT / "INTERVENTION_RESULTS_LEDGER.md"
    text = path.read_text(encoding="utf-8").rstrip() + "\n"
    marker = f"<!-- sealed-cr4-confirmation-{registration_id} -->"
    if marker in text:
        return
    rows = [
        "",
        marker,
        "## Full CR4 fixed-composition beta-surgery confirmation sealed",
        "",
        f"- Registration: `{registration_id}`.",
        f"- Result: `{output.relative_to(ROOT)}`.",
        f"- Full four-cell CR4 strength gate: **{metrics['confirmation_gate_pass']}**.",
        f"- Fixed-throughput topology classification: **{metrics['topology']['classification']}**.",
        f"- Exact replay: **{metrics['integrity_gates']['exact_replay']}**.",
        "- P3/P3b/P3c/P4 predecessor verdicts remain unchanged.",
        "- Mandatory review stop observed; CR5 was not launched automatically.",
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
    expected_states = 2 * MATRICES * len(LANDMARKS)
    print(
        f"[cr4 confirmation 1/8] Building {MATRICES} fresh matrices and {expected_states} states",
        flush=True,
    )
    _status(work, "running", "building_natural_states", cpu_budget_hours)
    with threadpool_limits(limits=1):
        cases = build_cohort(
            current_experiment, LABEL, current_experiment.confirmation
        )
    if len(cases) != expected_states:
        raise AssertionError("CR4 confirmation cohort is incomplete")
    model = registration_directory / "frozen_full_predictor.npz"
    futures = len(cases) * len(spec.arms) * BRANCHES
    print(
        f"[cr4 confirmation 2/8] Selecting beta surgeries and shooting {futures:,} F12 futures",
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
    print(f"[cr4 confirmation 3/8] Replaying all {futures:,} futures", flush=True)
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
        raise AssertionError("CR4 confirmation exact replay failed")
    arrays = base._outcome_arrays(cases, generated, spec)  # type: ignore[arg-type]
    geometry, geometry_rows, surgery_audit = _geometry_audit(
        cases, generated, spec
    )
    if not surgery_audit["all_audits_pass"]:
        raise AssertionError("CR4 confirmation surgery audit failed")
    draws = generate_inference_draws(
        MATRICES,
        BOOTSTRAP_REPETITIONS,
        RANDOMIZATION_REPETITIONS,
        np.random.default_rng(
            derive_seed(SEEDS["bootstrap"], f"{LABEL}.bootstrap")
        ),
        np.random.default_rng(
            derive_seed(SEEDS["randomization"], f"{LABEL}.randomization")
        ),
    )
    print("[cr4 confirmation 4/8] Computing frozen whole-matrix inference", flush=True)
    _status(work, "running", "whole_matrix_inference", cpu_budget_hours)
    metrics, matrix_rows, topology_rows, topology_arrays = _inference(
        cases, arrays, draws, spec
    )
    secondary = base._secondary_descriptives(cases, arrays, spec)  # type: ignore[arg-type]
    print("[cr4 confirmation 5/8] Writing and readback-checking artifacts", flush=True)
    _status(work, "running", "artifact_write_and_readback", cpu_budget_hours)
    with _atomic_destination(output) as destination:
        np.savez_compressed(destination / "branch_arrays.npz", **arrays)
        np.savez_compressed(destination / "surgery_geometry_arrays.npz", **geometry)
        base._write_branch_table(destination / "branches.csv.gz", cases, generated)
        base._write_state_artifacts(destination, cases, generated, arrays)
        base._write_selection_artifacts(destination, cases, generated, spec)  # type: ignore[arg-type]
        geometry_rows.to_csv(
            destination / "surgery_geometry_audit.csv.gz",
            index=False,
            compression="gzip",
        )
        pd.DataFrame(matrix_rows).to_csv(
            destination / "matrix_effects.csv", index=False
        )
        pd.DataFrame(topology_rows).to_csv(
            destination / "topology_matrix_effects.csv", index=False
        )
        _write_inference_arrays(
            destination / "inference_arrays.npz", draws, metrics, topology_arrays
        )
        readback = _readback(
            destination,
            cases,
            spec,
            metrics,
            matrix_rows,
            topology_rows,
        )
        integrity = {
            "exact_replay": replay[
                "state_edit_endpoint_and_process_digests_exact"
            ],
            "artifact_readback_exact": bool(
                readback["primary_metrics_exact"]
                and readback["matrix_effects_exact"]
                and readback["topology_matrix_effects_exact"]
            ),
            "surgery_audit_pass": surgery_audit["all_audits_pass"],
        }
        metrics["integrity_gates"] = integrity
        metrics["confirmation_gate_pass"] = bool(
            metrics["cr4_all_four_cells_scientific_pass"]
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
        (destination / "surgery_audit_summary.json").write_text(
            json.dumps(_json_ready(surgery_audit), indent=2, sort_keys=True)
            + "\n",
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
                    "causal occupied catalytic-strength control of Codex JOINT_BREAK_RUN3 at fixed composition"
                ]
                if metrics["confirmation_gate_pass"]
                else []
            ),
            "topology_classification": metrics["topology"]["classification"],
            "failed_predictions": (
                []
                if metrics["confirmation_gate_pass"]
                else ["full CR4 four-cell catalytic-strength gate"]
            ),
            "predecessor_results_preserved": True,
            "unresolved": [
                "molecular resistance versus resilience students",
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
            "arms": list(ARMS),
            "primary_futures": futures,
            "replay_futures": futures,
            "full_four_cell_cr4_gate": metrics["confirmation_gate_pass"],
            "topology_classification": metrics["topology"]["classification"],
            "exact_replay": integrity["exact_replay"],
            "complete_readback_exact": integrity["artifact_readback_exact"],
            "surgery_audit_pass": integrity["surgery_audit_pass"],
            "declared_cpu_budget_hours": cpu_budget_hours,
            "no_surgery_search_refitting_or_threshold_change": True,
            "no_future_retry_or_matrix_replacement": True,
            "predecessor_verdicts_unchanged": True,
            "mandatory_stop_after_this_stage": True,
            "cr5_launched": False,
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
    print("[cr4 confirmation 6/8] Result checksum sealed", flush=True)
    print("[cr4 confirmation 7/8] Durable ledger and status updated", flush=True)
    print("[cr4 confirmation 8/8] STOPPED; CR5 not launched", flush=True)


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
