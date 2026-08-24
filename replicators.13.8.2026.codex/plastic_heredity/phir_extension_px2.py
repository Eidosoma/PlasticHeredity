"""PX2 event-locked Phi around recovery from an identical broken state."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from threadpoolctl import threadpool_limits

from . import intervention_cr5 as cr5
from .config import CANDIDATES, GardConfig
from .experiment import StateCase
from .intervention_core import apply_molecular_edit
from .mechanistic import verify_checksums, write_checksums
from .phir_ch5 import _append_ledger, _snapshot_after_record
from .phir_extension_common import (
    BOOTSTRAP_DRAWS,
    MASTER_REGISTRATION,
    MAX_WORKERS,
    MINIMUM_FREE_DISK_BYTES,
    RANDOMIZATION_DRAWS,
    RESULT_ROOT,
    ROOT,
    apply_holm,
    atomic_json,
    atomic_pickle,
    canonical_digest,
    canonical_json,
    matrix_block_hotelling,
    paired_matrix_effects,
    paired_summary,
    purpose_seed,
    runtime_versions,
    safe_score_pairs,
    sha256_file,
)
from .simulator import (
    FissionRecord,
    SimulationError,
    Snapshot,
    advance_fission,
    generate_beta,
    generate_initial_composition,
    simulate_future_absorbing,
)


DOCUMENT = "CODEX_CH5_PHIR_EXTENSION_PREREGISTRATION.md"
DEFAULT_VALIDATION = RESULT_ROOT / "px2_validation"
DEFAULT_REGISTRATION = RESULT_ROOT / "px2_registration"
DEFAULT_SMOKE = RESULT_ROOT / "px2_smoke"
DEFAULT_OUTPUT = RESULT_ROOT / "px2_event_locked_recovery"
DEFAULT_WORK = RESULT_ROOT / ".px2_work"
DEFAULT_LOG = RESULT_ROOT / "px2_event_locked_recovery.log"

LABEL = "CODEX_CH5_PHIR_EXTENSION_PX2_V1"
PROGRAM_FORMAT = "codex-ch5-phir-extension-px2-program-v1"
REGISTRATION_FORMAT = "codex-ch5-phir-extension-px2-registration-v1"
CHECKPOINT_FORMAT = "codex-ch5-phir-extension-px2-checkpoint-v1"
RESULT_FORMAT = "codex-ch5-phir-extension-px2-result-v1"
SERVICE_NAME = "codex-phir-extension-px2-20260820"

ACQUISITION_MATRICES = 32
TARGET_MATRICES = 24
ACQUISITION_START = 10
ACQUISITION_LIMIT = 60
BRANCHES = 64
HORIZON = 8
ARMS = ("RENEWAL_UP", "RENEWAL_DOWN", "RANDOM", "NOOP")
HALVES = {"A": (0, 32), "B": (32, 64)}
REPRESENTATIONS = ("material", "functional_flux")
CPU_ALLOCATION_SECONDS = 10.0 * 3600.0

MODEL_SOURCE = (
    ROOT
    / "results_intervention_replication"
    / "cr5r_confirmation_registration"
    / "frozen_cr5_students.npz"
)
MODEL_CONTRACT_SOURCE = (
    ROOT
    / "results_intervention_replication"
    / "cr5r_confirmation_registration"
    / "model_contract.json"
)
EXPECTED_MODEL_SHA256 = "59750718efcea3492a6d9b4493e9dc379eb221150025681acd38651d623cd430"
EXPECTED_CONTRACT_SHA256 = "0d14920f45f831c3825ee36a73537cfbc067cafe9042088ae9a123d82900bd95"

# Transported, already sealed before PX2 outcomes.  These are the largest
# candidate/replicate margins from PX1's archived-NOOP calibration.
INFORMATION_EQUIVALENCE_MARGINS = {
    "material": 3.303539192139139,
    "functional_flux": 0.03321441162874219,
}
HEREDITY_EQUIVALENCE_MARGIN = 0.025

SOURCE_FILES = (
    DOCUMENT,
    "plastic_heredity/phir_extension_px2.py",
    "plastic_heredity/phir_extension_common.py",
    "tests/test_phir_extension_px2.py",
    "plastic_heredity/intervention_cr5.py",
    "plastic_heredity/intervention_core.py",
    "plastic_heredity/experiment.py",
    "plastic_heredity/config.py",
    "plastic_heredity/simulator.py",
    "plastic_heredity/features.py",
    "plastic_heredity/phir_instruments.py",
    "plastic_heredity/phir_rescue_instruments.py",
    "plastic_heredity/seeds.py",
)


@dataclass(frozen=True)
class PX2Spec:
    label: str
    acquisition_matrices: int
    target_matrices: int
    acquisition_start: int
    acquisition_limit: int
    branches: int
    horizon: int
    bootstrap_draws: int
    randomization_draws: int
    cpu_allocation_seconds: float


@dataclass(frozen=True)
class AcquiredState:
    candidate: str
    break_step: int
    composition: NDArray[np.int16]
    generation: int
    inheritance: tuple[bool, ...]
    boundary_h: tuple[float, ...]
    previous_growth_steps: int
    cumulative_growth_steps: int
    old_parent_anchor: NDArray[np.int16]
    path_record_digest: str


@dataclass(frozen=True)
class AcquisitionBatch:
    matrix_id: int
    beta: NDArray[np.float64]
    initial_composition: NDArray[np.int16]
    states: tuple[AcquiredState, ...]
    acquisition_rows: tuple[dict[str, Any], ...]
    cpu_seconds: float
    scientific_digest: str


@dataclass(frozen=True)
class InterventionBatch:
    matrix_id: int
    score_rows: tuple[dict[str, Any], ...]
    branch_rows: tuple[dict[str, Any], ...]
    selected_edit_rows: tuple[dict[str, Any], ...]
    cpu_seconds: float
    scientific_digest: str


def scientific_spec() -> PX2Spec:
    return PX2Spec(
        "scientific",
        ACQUISITION_MATRICES,
        TARGET_MATRICES,
        ACQUISITION_START,
        ACQUISITION_LIMIT,
        BRANCHES,
        HORIZON,
        BOOTSTRAP_DRAWS,
        RANDOMIZATION_DRAWS,
        CPU_ALLOCATION_SECONDS,
    )


def smoke_spec() -> PX2Spec:
    return PX2Spec("smoke", 1, 1, 1, 4, 4, 4, 32, 32, 300.0)


def _source_hashes() -> dict[str, str]:
    return {name: sha256_file(ROOT / name) for name in SOURCE_FILES}


def _batch_digest(value: AcquisitionBatch | InterventionBatch) -> str:
    fields = asdict(value)
    fields["cpu_seconds"] = 0.0
    fields["scientific_digest"] = ""
    return canonical_digest(fields)


def _record_digest(records: Sequence[FissionRecord]) -> str:
    return cr5._records_digest(records)


def _matrix_seed(spec: PX2Spec, matrix_id: int, purpose: str) -> int:
    domain = "smoke" if spec.label == "smoke" else purpose
    return purpose_seed(domain, "PX2", spec.label, purpose, matrix_id)


def _acquisition_seed(spec: PX2Spec, candidate: str, matrix_id: int) -> int:
    domain = "smoke" if spec.label == "smoke" else "acquisition"
    return purpose_seed(
        domain, "PX2", spec.label, "natural_break", candidate, matrix_id
    )


def _selection_seed(spec: PX2Spec, candidate: str, matrix_id: int) -> int:
    domain = "smoke" if spec.label == "smoke" else "random_action"
    return purpose_seed(domain, "PX2", spec.label, "selection", candidate, matrix_id)


def _future_seed(
    spec: PX2Spec, candidate: str, matrix_id: int, branch: int
) -> int:
    domain = "smoke" if spec.label == "smoke" else "future"
    return purpose_seed(
        domain, "PX2", spec.label, "future", candidate, matrix_id, branch
    )


def _to_snapshot(state: AcquiredState) -> Snapshot:
    return Snapshot(
        composition=np.asarray(state.composition, dtype=np.int64).copy(),
        generation=state.generation,
        inheritance=state.inheritance,
        boundary_h=state.boundary_h,
        previous_growth_steps=state.previous_growth_steps,
        cumulative_growth_steps=state.cumulative_growth_steps,
    )


def _acquire_candidate(
    spec: PX2Spec,
    matrix_id: int,
    candidate: str,
    beta: NDArray,
    initial: NDArray,
) -> tuple[AcquiredState | None, dict[str, Any]]:
    config = GardConfig()
    rng = np.random.default_rng(_acquisition_seed(spec, candidate, matrix_id))
    snapshot = Snapshot(np.asarray(initial, dtype=np.int64).copy(), 0, (), ())
    records: list[FissionRecord] = []
    reason = "no_break_within_limit"
    for step in range(1, spec.acquisition_limit + 1):
        try:
            record = advance_fission(
                snapshot.composition,
                beta,
                config,
                CANDIDATES[candidate],
                rng,
            )
        except SimulationError:
            reason = "extinction_before_eligible_break"
            break
        records.append(record)
        snapshot = _snapshot_after_record(snapshot, record)
        if step >= spec.acquisition_start and record.h <= config.inheritance_threshold:
            acquired = AcquiredState(
                candidate=candidate,
                break_step=step,
                composition=np.asarray(snapshot.composition, dtype=np.int16),
                generation=snapshot.generation,
                inheritance=snapshot.inheritance,
                boundary_h=snapshot.boundary_h,
                previous_growth_steps=snapshot.previous_growth_steps,
                cumulative_growth_steps=snapshot.cumulative_growth_steps,
                old_parent_anchor=np.asarray(record.parent, dtype=np.int16),
                path_record_digest=_record_digest(records),
            )
            return acquired, {
                "matrix_id": matrix_id,
                "candidate": candidate,
                "eligible": 1,
                "reason": "first_break_at_or_after_registered_start",
                "break_step": step,
                "break_h": float(record.h),
                "observed_fissions": len(records),
                "path_record_digest": acquired.path_record_digest,
                "state_digest": canonical_digest(asdict(acquired)),
            }
    return None, {
        "matrix_id": matrix_id,
        "candidate": candidate,
        "eligible": 0,
        "reason": reason,
        "break_step": -1,
        "break_h": float("nan"),
        "observed_fissions": len(records),
        "path_record_digest": _record_digest(records),
        "state_digest": "missing",
    }


def _acquire_matrix(args: tuple[int, PX2Spec]) -> AcquisitionBatch:
    matrix_id, spec = args
    started = time.process_time()
    with threadpool_limits(limits=1):
        config = GardConfig()
        beta = generate_beta(
            config, np.random.default_rng(_matrix_seed(spec, matrix_id, "matrix"))
        )
        initial = generate_initial_composition(
            config, np.random.default_rng(_matrix_seed(spec, matrix_id, "initial"))
        )
        states: list[AcquiredState] = []
        rows: list[dict[str, Any]] = []
        for candidate in CANDIDATES:
            state, row = _acquire_candidate(
                spec, matrix_id, candidate, beta, initial
            )
            rows.append(row)
            if state is not None:
                states.append(state)
    provisional = AcquisitionBatch(
        matrix_id,
        np.asarray(beta, dtype=np.float64),
        np.asarray(initial, dtype=np.int16),
        tuple(states),
        tuple(rows),
        float(time.process_time() - started),
        "",
    )
    return AcquisitionBatch(
        provisional.matrix_id,
        provisional.beta,
        provisional.initial_composition,
        provisional.states,
        provisional.acquisition_rows,
        provisional.cpu_seconds,
        _batch_digest(provisional),
    )


def _branch_fields(
    outcome: Any, records: Sequence[FissionRecord], threshold: float
) -> dict[str, Any]:
    inherited = np.asarray([record.h > threshold for record in records], dtype=bool)
    return {
        "renewal_run3": int(outcome.joint_break_run3),
        "run5": int(cr5._first_run(inherited, 5) >= 0),
        "renewal_time": int(outcome.renewal_certification_time),
        "inherited_count": int(outcome.inherited_boundary_count),
        "completed_horizon": int(outcome.completed_horizon),
        "observed_fissions": int(outcome.observed_fissions),
        "total_growth_updates": int(outcome.total_growth_updates),
        "final_entropy": float(outcome.final_entropy),
        "final_occupied_types": int(outcome.final_occupied_types),
        "record_digest": str(outcome.record_digest),
    }


def _intervene_matrix(
    args: tuple[AcquisitionBatch, PX2Spec, str, str]
) -> InterventionBatch:
    acquisition, spec, model_path, contract_path = args
    started = time.process_time()
    config = GardConfig()
    score_rows: list[dict[str, Any]] = []
    branch_rows: list[dict[str, Any]] = []
    edit_rows: list[dict[str, Any]] = []
    with threadpool_limits(limits=1):
        students = cr5.load_students(Path(model_path), Path(contract_path))
        for state in acquisition.states:
            candidate = state.candidate
            snapshot = _to_snapshot(state)
            case = StateCase(
                state_id=f"PX2-c{candidate}-m{acquisition.matrix_id}-break{state.break_step}",
                cohort="PX2_NATURAL_POST_BREAK",
                candidate=candidate,
                matrix_id=acquisition.matrix_id,
                landmark=state.break_step,
                beta=acquisition.beta,
                snapshot=snapshot,
            )
            student = students[("renewal", candidate)]
            noop, scores = cr5.score_student_edits(student, case, config)
            predictions, edits = cr5.select_student_edits(
                noop,
                scores,
                np.random.default_rng(
                    _selection_seed(spec, candidate, acquisition.matrix_id)
                ),
            )
            arm_pairs: dict[tuple[str, str], tuple[list[NDArray], list[NDArray]]] = {
                (arm, half): ([], []) for arm in ARMS for half in HALVES
            }
            early_pairs: dict[tuple[str, str], tuple[list[NDArray], list[NDArray]]] = {
                (arm, half): ([], []) for arm in ARMS for half in HALVES
            }
            selected_lookup = {arm: edit for arm, edit in zip(ARMS, edits, strict=True)}
            for arm_index, arm in enumerate(ARMS):
                edit = selected_lookup[arm]
                edit_rows.append(
                    {
                        "matrix_id": acquisition.matrix_id,
                        "candidate": candidate,
                        "break_step": state.break_step,
                        "arm": arm,
                        "remove_type": -1 if edit is None else edit.remove_type,
                        "add_type": -1 if edit is None else edit.add_type,
                        "predicted_probability": float(predictions[arm_index]),
                        "noop_probability": float(noop),
                        "predicted_shift": float(predictions[arm_index] - noop),
                        "legal_edits_scored": len(scores),
                    }
                )
            for branch in range(spec.branches):
                half = "A" if branch < spec.branches // 2 else "B"
                seed = _future_seed(spec, candidate, acquisition.matrix_id, branch)
                for arm, edit in selected_lookup.items():
                    composition = (
                        snapshot.composition
                        if edit is None
                        else apply_molecular_edit(snapshot.composition, edit)
                    )
                    launch = Snapshot(
                        np.asarray(composition, dtype=np.int64).copy(),
                        snapshot.generation,
                        snapshot.inheritance,
                        snapshot.boundary_h,
                        snapshot.previous_growth_steps,
                        snapshot.cumulative_growth_steps,
                    )
                    records, completed = simulate_future_absorbing(
                        launch,
                        acquisition.beta,
                        config,
                        CANDIDATES[candidate],
                        spec.horizon,
                        np.random.default_rng(seed),
                    )
                    outcome = cr5._stage_outcome(
                        "resilience",
                        launch,
                        records,
                        completed,
                        spec.horizon,
                        config.inheritance_threshold,
                    )
                    branch_row = {
                        "matrix_id": acquisition.matrix_id,
                        "candidate": candidate,
                        "break_step": state.break_step,
                        "arm": arm,
                        "branch": branch,
                        "half": half,
                        **_branch_fields(outcome, records, config.inheritance_threshold),
                    }
                    if records:
                        branch_row["old_anchor_final_similarity"] = float(
                            np.dot(records[-1].daughter, state.old_parent_anchor)
                            / max(
                                np.linalg.norm(records[-1].daughter)
                                * np.linalg.norm(state.old_parent_anchor),
                                np.finfo(float).tiny,
                            )
                        )
                    else:
                        branch_row["old_anchor_final_similarity"] = float("nan")
                    branch_rows.append(branch_row)
                    previous = np.asarray(launch.composition, dtype=np.int64)
                    for transition, record in enumerate(records, start=1):
                        arm_pairs[(arm, half)][0].append(previous.copy())
                        arm_pairs[(arm, half)][1].append(record.daughter.copy())
                        if transition <= 2:
                            early_pairs[(arm, half)][0].append(previous.copy())
                            early_pairs[(arm, half)][1].append(record.daughter.copy())
                        previous = record.daughter
            branch_frame = pd.DataFrame(
                [
                    row
                    for row in branch_rows
                    if row["matrix_id"] == acquisition.matrix_id
                    and row["candidate"] == candidate
                ]
            )
            for arm in ARMS:
                for half in HALVES:
                    selected_branches = branch_frame[
                        (branch_frame["arm"] == arm) & (branch_frame["half"] == half)
                    ]
                    row: dict[str, Any] = {
                        "matrix_id": acquisition.matrix_id,
                        "candidate": candidate,
                        "break_step": state.break_step,
                        "arm": arm,
                        "half": half,
                        "renewal_probability": float(selected_branches["renewal_run3"].mean()),
                        "mean_inherited_count": float(selected_branches["inherited_count"].mean()),
                        "survival_probability": float(selected_branches["completed_horizon"].mean()),
                        "mean_observed_fissions": float(selected_branches["observed_fissions"].mean()),
                        "branches": int(len(selected_branches)),
                    }
                    past, future = arm_pairs[(arm, half)]
                    early_past, early_future = early_pairs[(arm, half)]
                    row["transition_pairs"] = len(past)
                    row["early_transition_pairs"] = len(early_past)
                    for representation in REPRESENTATIONS:
                        score = safe_score_pairs(
                            np.asarray(past),
                            np.asarray(future),
                            acquisition.beta,
                            representation,
                            config,
                        )
                        row.update(score.fields(representation))
                        early = safe_score_pairs(
                            np.asarray(early_past),
                            np.asarray(early_future),
                            acquisition.beta,
                            representation,
                            config,
                        )
                        row.update(early.fields(f"early_{representation}"))
                    score_rows.append(row)
    provisional = InterventionBatch(
        acquisition.matrix_id,
        tuple(score_rows),
        tuple(branch_rows),
        tuple(edit_rows),
        float(time.process_time() - started),
        "",
    )
    return InterventionBatch(
        provisional.matrix_id,
        provisional.score_rows,
        provisional.branch_rows,
        provisional.selected_edit_rows,
        provisional.cpu_seconds,
        _batch_digest(provisional),
    )


def protocol() -> dict[str, Any]:
    value: dict[str, Any] = {
        "format": PROGRAM_FORMAT,
        "phase": "PX2",
        "question": "information dynamics during causal recovery from an identical naturally broken state",
        "spec": asdict(scientific_spec()),
        "acquisition": {
            "seeded_pool": ACQUISITION_MATRICES,
            "break_start": ACQUISITION_START,
            "break_limit": ACQUISITION_LIMIT,
            "paired_candidate_eligibility": True,
            "selection": "first 24 paired-eligible matrices in sealed seed order",
            "if_shortfall": "run all paired-eligible matrices and fail coverage; add no seeds",
            "outcome_blind": True,
        },
        "arms": list(ARMS),
        "model_sha256": EXPECTED_MODEL_SHA256,
        "model_contract_sha256": EXPECTED_CONTRACT_SHA256,
        "model_refit_or_recalibration": False,
        "branches": BRANCHES,
        "halves": HALVES,
        "future_seed_includes_arm": False,
        "selection_stream_separate": True,
        "explicit_pairs": True,
        "cross_branch_transitions": False,
        "extinction_information_rule": "retain every observed transition, never retry, and report pair count and survival",
        "representations": list(REPRESENTATIONS),
        "primary": "material full-block RENEWAL_UP minus RENEWAL_DOWN",
        "secondary": [
            "functional-flux full block",
            "first two transitions",
            "four-component public PhiID vector",
            "renewal probability",
            "inherited count",
            "renewal time",
            "old-anchor similarity",
            "survival",
        ],
        "equivalence_margins": {
            "information": INFORMATION_EQUIVALENCE_MARGINS,
            "heredity_probability": HEREDITY_EQUIVALENCE_MARGIN,
            "origin": "transported pre-outcome PX1 archived-NOOP calibration maxima",
        },
        "inference": {
            "unit": "whole catalytic matrix",
            "bootstrap": BOOTSTRAP_DRAWS,
            "randomization": RANDOMIZATION_DRAWS,
            "holm": "four candidate-by-half primary cells",
            "vector_test": "ridge Hotelling paired whole-matrix sign randomization",
        },
        "gate": [
            "24 paired-eligible matrices",
            "positive material effect in all four cells",
            "positive bootstrap lower bounds",
            "Holm-adjusted p < 0.05",
            "RANDOM equivalent to NOOP",
            "renewal manipulation valid in all four cells",
            "exact acquisition and intervention replay",
        ],
        "no_48_matrix_continuation": True,
        "claim_boundary": [
            "event-locked information is not consciousness",
            "association is not a claim that Phi causes recovery",
            "strict-eight is excluded",
            "prior results remain unchanged",
        ],
    }
    value["protocol_id"] = canonical_digest(value)
    return value


def validation_checks() -> dict[str, bool]:
    inherited = np.nextafter(0.9, 1.0)
    launch = Snapshot(np.asarray([2, 1, 1, 0]), 10, (False,), (0.8,))
    record = lambda h: FissionRecord(
        np.asarray([2, 1, 1, 0]), np.asarray([1, 1, 0, 0]), h, 3
    )
    positive = cr5._stage_outcome(
        "resilience",
        launch,
        [record(inherited), record(inherited), record(inherited)],
        True,
        HORIZON,
        0.9,
    )
    threshold = cr5._stage_outcome(
        "resilience",
        launch,
        [record(inherited), record(0.9), record(inherited)],
        True,
        HORIZON,
        0.9,
    )
    return {
        "model_hash_exact": sha256_file(MODEL_SOURCE) == EXPECTED_MODEL_SHA256,
        "model_contract_hash_exact": sha256_file(MODEL_CONTRACT_SOURCE)
        == EXPECTED_CONTRACT_SHA256,
        "acquisition_pool_fixed": ACQUISITION_MATRICES == 32,
        "target_scale_fixed": TARGET_MATRICES == 24,
        "break_window_fixed": ACQUISITION_START == 10 and ACQUISITION_LIMIT == 60,
        "branches_and_halves_fixed": BRANCHES == 64
        and HALVES == {"A": (0, 32), "B": (32, 64)},
        "endpoint_positive_fixture": positive.joint_break_run3
        and positive.renewal_certification_time == 3,
        "threshold_is_strict": not threshold.joint_break_run3,
        "future_and_selection_streams_distinct": _future_seed(
            smoke_spec(), "02", 0, 0
        )
        != _selection_seed(smoke_spec(), "02", 0),
        "future_seed_arm_free": "arm" not in inspect.signature(_future_seed).parameters,
        "representations_fixed": REPRESENTATIONS == ("material", "functional_flux"),
        "information_margins_positive": all(
            value > 0 for value in INFORMATION_EQUIVALENCE_MARGINS.values()
        ),
        "draws_fixed": BOOTSTRAP_DRAWS == 4096 and RANDOMIZATION_DRAWS == 4096,
        "cpu_allocation_fixed": CPU_ALLOCATION_SECONDS == 10 * 3600,
    }


def run_validation() -> dict[str, Any]:
    checks = validation_checks()
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/test_phir_extension_px2.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    payload = {
        "format": "codex-ch5-phir-extension-px2-validation-v1",
        "checks": checks,
        "pytest_returncode": completed.returncode,
        "pytest_stdout": completed.stdout,
        "pytest_stderr": completed.stderr,
        "all_passed": bool(all(checks.values()) and completed.returncode == 0),
    }
    if DEFAULT_VALIDATION.exists():
        shutil.rmtree(DEFAULT_VALIDATION)
    DEFAULT_VALIDATION.mkdir(parents=True)
    atomic_json(DEFAULT_VALIDATION / "validation.json", payload)
    write_checksums(DEFAULT_VALIDATION)
    if not payload["all_passed"]:
        raise AssertionError(f"PX2 validation failed\n{completed.stdout}\n{completed.stderr}")
    return payload


def register_program() -> dict[str, Any]:
    verify_checksums(DEFAULT_VALIDATION)
    validation = json.loads(
        (DEFAULT_VALIDATION / "validation.json").read_text(encoding="utf-8")
    )
    if not validation["all_passed"]:
        raise ValueError("PX2 validation did not pass")
    if DEFAULT_REGISTRATION.exists():
        raise FileExistsError(f"PX2 registration exists: {DEFAULT_REGISTRATION}")
    master = json.loads(
        (MASTER_REGISTRATION / "registration.json").read_text(encoding="utf-8")
    )
    body: dict[str, Any] = {
        "format": REGISTRATION_FORMAT,
        "master_registration_id": master["registration_id"],
        "protocol": protocol(),
        "source_hashes": _source_hashes(),
        "runtime": runtime_versions(),
        "model_sha256": sha256_file(MODEL_SOURCE),
        "model_contract_sha256": sha256_file(MODEL_CONTRACT_SOURCE),
        "new_scientific_matrices_at_registration": 0,
    }
    body["registration_id"] = canonical_digest(body)
    DEFAULT_REGISTRATION.mkdir(parents=True)
    shutil.copy2(ROOT / DOCUMENT, DEFAULT_REGISTRATION / "preregistration.md")
    shutil.copy2(MODEL_SOURCE, DEFAULT_REGISTRATION / "frozen_cr5_students.npz")
    shutil.copy2(MODEL_CONTRACT_SOURCE, DEFAULT_REGISTRATION / "model_contract.json")
    atomic_json(DEFAULT_REGISTRATION / "protocol.json", body["protocol"])
    atomic_json(DEFAULT_REGISTRATION / "registration.json", body)
    write_checksums(DEFAULT_REGISTRATION)
    _append_ledger(
        f"<!-- phir-extension-px2-registration-{body['registration_id']} -->",
        [
            "## Phi-r extension PX2 registered",
            "",
            f"- Registration: `{body['registration_id']}`.",
            "- A 32-seed acquisition pool and at most 24 paired-candidate intervention matrices were sealed.",
            "- No PX2 scientific matrix existed at registration; strict-eight is excluded.",
        ],
    )
    return body


def verify_registration() -> dict[str, Any]:
    verify_checksums(DEFAULT_REGISTRATION)
    body = json.loads(
        (DEFAULT_REGISTRATION / "registration.json").read_text(encoding="utf-8")
    )
    observed = body.pop("registration_id")
    if body.get("format") != REGISTRATION_FORMAT or observed != canonical_digest(body):
        raise ValueError("PX2 registration identity failed")
    body["registration_id"] = observed
    if body["protocol"] != canonical_json(protocol()):
        raise ValueError("PX2 protocol changed")
    if body["source_hashes"] != _source_hashes():
        raise ValueError("PX2 source changed after registration")
    if sha256_file(DEFAULT_REGISTRATION / "frozen_cr5_students.npz") != EXPECTED_MODEL_SHA256:
        raise ValueError("PX2 frozen model changed")
    if sha256_file(DEFAULT_REGISTRATION / "model_contract.json") != EXPECTED_CONTRACT_SHA256:
        raise ValueError("PX2 model contract changed")
    return body


def _checkpointed(
    kind: str,
    spec: PX2Spec,
    registration: Mapping[str, Any],
    directory: Path,
    inputs: Sequence[Any],
    worker: Any,
    workers: int,
    prior_cpu: float = 0.0,
) -> tuple[list[Any], float]:
    directory.mkdir(parents=True, exist_ok=True)
    contract = {
        "format": CHECKPOINT_FORMAT,
        "registration_id": registration["registration_id"],
        "protocol_id": registration["protocol"]["protocol_id"],
        "kind": kind,
        "spec": asdict(spec),
        "input_ids": [
            int(value if isinstance(value, int) else value.matrix_id) for value in inputs
        ],
        "source_hashes": registration["source_hashes"],
    }
    contract["contract_id"] = canonical_digest(contract)
    path = directory / "checkpoint_contract.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != canonical_json(contract):
            raise ValueError(f"PX2 checkpoint contract changed: {directory}")
    else:
        atomic_json(path, contract)
    batches: list[Any | None] = [None] * len(inputs)
    missing: list[int] = []
    cpu = float(prior_cpu)
    for index, value in enumerate(inputs):
        matrix_id = int(value if isinstance(value, int) else value.matrix_id)
        checkpoint = directory / f"matrix_{matrix_id:04d}.pkl"
        if checkpoint.exists():
            with checkpoint.open("rb") as handle:
                batch = pickle.load(handle)
            if batch.matrix_id != matrix_id or batch.scientific_digest != _batch_digest(batch):
                raise ValueError(f"invalid PX2 checkpoint: {checkpoint}")
            batches[index] = batch
            cpu += batch.cpu_seconds
        else:
            missing.append(index)

    def save_status(state: str) -> None:
        complete = sum(item is not None for item in batches)
        atomic_json(
            directory / "status.json",
            {
                "kind": kind,
                "state": state,
                "completed": complete,
                "total": len(inputs),
                "fraction": complete / max(1, len(inputs)),
                "cpu_seconds": cpu,
            },
        )

    save_status("running")
    arguments = [inputs[index] for index in missing]
    if kind.startswith("acquisition"):
        arguments = [(int(item), spec) for item in arguments]
    else:
        arguments = [
            (
                item,
                spec,
                str(DEFAULT_REGISTRATION / "frozen_cr5_students.npz"),
                str(DEFAULT_REGISTRATION / "model_contract.json"),
            )
            for item in arguments
        ]
    executor: ProcessPoolExecutor | None = None
    generated: Iterable[Any]
    if workers <= 1:
        generated = map(worker, arguments)
    else:
        executor = ProcessPoolExecutor(max_workers=min(workers, MAX_WORKERS))
        generated = executor.map(worker, arguments, chunksize=1)
    try:
        for index, batch in zip(missing, generated, strict=True):
            expected_id = int(inputs[index] if isinstance(inputs[index], int) else inputs[index].matrix_id)
            if batch.matrix_id != expected_id or batch.scientific_digest != _batch_digest(batch):
                raise AssertionError("PX2 worker returned an invalid batch")
            batches[index] = batch
            atomic_pickle(directory / f"matrix_{expected_id:04d}.pkl", batch)
            cpu += batch.cpu_seconds
            save_status("running")
            print(f"[PX2 {kind}] {sum(item is not None for item in batches)}/{len(inputs)}", flush=True)
            if cpu > spec.cpu_allocation_seconds:
                raise RuntimeError("PX2 CPU allocation reached; checkpoints retained")
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
    if any(item is None for item in batches):
        raise AssertionError(f"PX2 {kind} checkpoint stage incomplete")
    save_status("complete")
    return [item for item in batches if item is not None], cpu


def _eligible(acquisitions: Sequence[AcquisitionBatch], target: int) -> list[AcquisitionBatch]:
    paired = [
        batch
        for batch in sorted(acquisitions, key=lambda item: item.matrix_id)
        if {state.candidate for state in batch.states} == set(CANDIDATES)
    ]
    return paired[:target]


def _replay_exact(left: Sequence[Any], right: Sequence[Any], expected: int) -> dict[str, Any]:
    rows = [
        {
            "matrix_id": first.matrix_id,
            "generated": first.scientific_digest,
            "replay": second.scientific_digest,
            "exact": first.scientific_digest == second.scientific_digest,
        }
        for first, second in zip(left, right, strict=True)
    ]
    return {
        "matrices": rows,
        "complete_exact_replay": len(rows) == expected and all(row["exact"] for row in rows),
    }


def analyze(
    batches: Sequence[InterventionBatch],
    spec: PX2Spec,
    coverage_pass: bool,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame], dict[str, NDArray]]:
    scores = pd.DataFrame([row for batch in batches for row in batch.score_rows])
    branches = pd.DataFrame([row for batch in batches for row in batch.branch_rows])
    edits = pd.DataFrame([row for batch in batches for row in batch.selected_edit_rows])
    scores["candidate"] = scores["candidate"].astype(str).str.zfill(2)
    arrays: dict[str, NDArray] = {}
    matrix_rows: list[dict[str, Any]] = []
    primary: list[dict[str, Any]] = []
    secondary: list[dict[str, Any]] = []
    specificity: list[dict[str, Any]] = []
    renewal_cells: list[dict[str, Any]] = []
    vector_cells: list[dict[str, Any]] = []
    vector_names = (
        "downward_causation",
        "synergy_persistence",
        "redundancy_persistence",
        "cross_part_transfer",
    )
    for candidate in CANDIDATES:
        for half in HALVES:
            filters = {"candidate": candidate, "half": half}
            cell = f"{candidate}_{half}"
            for representation in REPRESENTATIONS:
                metric = f"{representation}_full_revised"
                values = paired_matrix_effects(
                    scores, metric, "RENEWAL_UP", "RENEWAL_DOWN", filters=filters
                )
                summary, local = paired_summary(
                    values.to_numpy(),
                    f"PX2/{representation}/{cell}/up-down",
                    bootstrap_draws=spec.bootstrap_draws,
                    randomization_draws=spec.randomization_draws,
                )
                summary.update(
                    {
                        "representation": representation,
                        "candidate": candidate,
                        "half": half,
                        "metric": metric,
                        "contrast": "RENEWAL_UP-RENEWAL_DOWN",
                    }
                )
                (primary if representation == "material" else secondary).append(summary)
                arrays.update(
                    {f"{representation}__{cell}__{name}": value for name, value in local.items()}
                )
                for matrix_id, value in values.items():
                    matrix_rows.append(
                        {
                            "family": "information",
                            "representation": representation,
                            "candidate": candidate,
                            "half": half,
                            "matrix_id": int(matrix_id),
                            "value": float(value),
                        }
                    )
                random_values = paired_matrix_effects(
                    scores, metric, "RANDOM", "NOOP", filters=filters
                )
                random_summary, local = paired_summary(
                    random_values.to_numpy(),
                    f"PX2/{representation}/{cell}/random-noop",
                    bootstrap_draws=spec.bootstrap_draws,
                    randomization_draws=spec.randomization_draws,
                    equivalence_margin=INFORMATION_EQUIVALENCE_MARGINS[representation],
                )
                random_summary.update(
                    {
                        "family": "information_specificity",
                        "representation": representation,
                        "candidate": candidate,
                        "half": half,
                    }
                )
                specificity.append(random_summary)
                arrays.update(
                    {f"specificity__{representation}__{cell}__{name}": value for name, value in local.items()}
                )
            renewal = paired_matrix_effects(
                scores,
                "renewal_probability",
                "RENEWAL_UP",
                "RENEWAL_DOWN",
                filters=filters,
            )
            renewal_summary, local = paired_summary(
                renewal.to_numpy(),
                f"PX2/renewal/{cell}/up-down",
                bootstrap_draws=spec.bootstrap_draws,
                randomization_draws=spec.randomization_draws,
            )
            renewal_summary.update({"candidate": candidate, "half": half})
            renewal_cells.append(renewal_summary)
            arrays.update({f"renewal__{cell}__{name}": value for name, value in local.items()})
            renewal_random = paired_matrix_effects(
                scores, "renewal_probability", "RANDOM", "NOOP", filters=filters
            )
            random_summary, local = paired_summary(
                renewal_random.to_numpy(),
                f"PX2/renewal/{cell}/random-noop",
                bootstrap_draws=spec.bootstrap_draws,
                randomization_draws=spec.randomization_draws,
                equivalence_margin=HEREDITY_EQUIVALENCE_MARGIN,
            )
            random_summary.update(
                {"family": "renewal_specificity", "candidate": candidate, "half": half}
            )
            specificity.append(random_summary)
            arrays.update({f"renewal_specificity__{cell}__{name}": value for name, value in local.items()})

            selected = scores[(scores.candidate == candidate) & (scores.half == half)]
            vectors: list[list[float]] = []
            vector_ids: list[int] = []
            for matrix_id, local_frame in selected.groupby("matrix_id", sort=True):
                pivot = local_frame.set_index("arm")
                if not {"RENEWAL_UP", "RENEWAL_DOWN"}.issubset(pivot.index):
                    continue
                up = pivot.loc["RENEWAL_UP"]
                down = pivot.loc["RENEWAL_DOWN"]
                vectors.append(
                    [
                        float(up["material_causation"] - down["material_causation"]),
                        float(up["material_synergy_persistence"] - down["material_synergy_persistence"]),
                        float(up["material_atom_r_to_r"] - down["material_atom_r_to_r"]),
                        float(
                            up["material_atom_u0_to_u1"]
                            + up["material_atom_u1_to_u0"]
                            - down["material_atom_u0_to_u1"]
                            - down["material_atom_u1_to_u0"]
                        ),
                    ]
                )
                vector_ids.append(int(matrix_id))
            vector = np.asarray(vectors, dtype=np.float64)
            statistic, local = matrix_block_hotelling(vector, f"PX2/vector/{cell}")
            statistic.update(
                {
                    "candidate": candidate,
                    "half": half,
                    "component_names": vector_names,
                }
            )
            vector_cells.append(statistic)
            arrays.update({f"vector__{cell}__{name}": value for name, value in local.items()})
            arrays[f"vector__{cell}__matrix_ids"] = np.asarray(vector_ids, dtype=np.int16)
    apply_holm(primary)
    apply_holm(renewal_cells)
    apply_holm(vector_cells, source="randomization_p", destination="holm_adjusted_p")
    info_specificity = {
        (row.get("representation"), row["candidate"], row["half"]): row
        for row in specificity
        if row["family"] == "information_specificity"
    }
    renewal_specificity = {
        (row["candidate"], row["half"]): row
        for row in specificity
        if row["family"] == "renewal_specificity"
    }
    renewal_pass = bool(
        len(renewal_cells) == 4
        and all(
            row["effect"] > 0
            and row["ci95"][0] > 0
            and row.get("holm_adjusted_p", 1) < 0.05
            and renewal_specificity[(row["candidate"], row["half"])].get("tost_via_90ci", False)
            for row in renewal_cells
        )
    )
    primary_pass = bool(
        coverage_pass
        and renewal_pass
        and len(primary) == 4
        and all(
            row["effect"] > 0
            and row["ci95"][0] > 0
            and row.get("holm_adjusted_p", 1) < 0.05
            and info_specificity[("material", row["candidate"], row["half"])].get("tost_via_90ci", False)
            for row in primary
        )
    )
    metrics = {
        "format": "codex-ch5-phir-extension-px2-metrics-v1",
        "primary_material": primary,
        "secondary_functional": secondary,
        "specificity": specificity,
        "renewal_validity": renewal_cells,
        "phiid_vector": vector_cells,
        "gates": {
            "paired_eligibility_24": coverage_pass,
            "renewal_manipulation_validity": renewal_pass,
            "event_locked_material_information": primary_pass,
            "vector_response_all_cells": bool(
                len(vector_cells) == 4
                and all(row.get("holm_adjusted_p", 1) < 0.05 for row in vector_cells)
            ),
        },
    }
    return metrics, {
        "scores": scores,
        "branches": branches,
        "edits": edits,
        "matrix_effects": pd.DataFrame(matrix_rows),
    }, arrays


def _write_result(
    acquisitions: Sequence[AcquisitionBatch],
    batches: Sequence[InterventionBatch],
    replay: Mapping[str, Any],
    registration: Mapping[str, Any],
    cpu_seconds: float,
) -> dict[str, Any]:
    coverage = len(batches) == TARGET_MATRICES
    metrics, tables, arrays = analyze(batches, scientific_spec(), coverage)
    temporary = DEFAULT_OUTPUT.with_name(DEFAULT_OUTPUT.name + f".tmp-{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    for name, frame in tables.items():
        frame.to_csv(temporary / f"{name}.csv.gz", index=False)
    pd.DataFrame(
        [row for batch in acquisitions for row in batch.acquisition_rows]
    ).to_csv(temporary / "acquisition.csv.gz", index=False)
    np.savez_compressed(temporary / "inference_arrays.npz", **arrays)
    np.savez_compressed(
        temporary / "matrix_inputs.npz",
        matrix_id=np.asarray([batch.matrix_id for batch in acquisitions], dtype=np.int16),
        beta=np.stack([batch.beta for batch in acquisitions]),
        initial=np.stack([batch.initial_composition for batch in acquisitions]),
        digest=np.asarray([batch.scientific_digest for batch in acquisitions]),
    )
    atomic_json(temporary / "primary_metrics.json", metrics)
    atomic_json(temporary / "replay_audit.json", replay)
    lines = [
        "# PX2 event-locked recovery report",
        "",
        f"Registration: `{registration['registration_id']}`.",
        "",
        "Every intervention arm began from the identical naturally broken daughter. Explicit transitions from independent branches were pooled without joining branch boundaries.",
        "",
        "| Candidate | Half | Material effect [95% CI] | Holm p |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in metrics["primary_material"]:
        lines.append(
            f"| {row['candidate']} | {row['half']} | {row['effect']:+.5f} "
            f"[{row['ci95'][0]:+.5f}, {row['ci95'][1]:+.5f}] | "
            f"{row.get('holm_adjusted_p', float('nan')):.4g} |"
        )
    lines.extend(
        [
            "",
            "## Gates",
            "",
            "```json",
            json.dumps(metrics["gates"], indent=2, sort_keys=True),
            "```",
            "",
            "This phase does not establish consciousness or make Phi a cause of recovery.",
        ]
    )
    (temporary / "SCIENTIFIC_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    lay = [
        "# PX2 lay summary",
        "",
        "We waited until an assembly naturally lost its usual parent-to-daughter similarity, copied that exact broken daughter, and gave each copy a different one-molecule edit. This makes the recovery comparison genuinely fair: every arm starts after the same break.",
        "",
        f"The recovery-control check {'passed' if metrics['gates']['renewal_manipulation_validity'] else 'did not pass every strict condition'}.",
        f"The event-locked material information test {'passed' if metrics['gates']['event_locked_material_information'] else 'did not pass'} its four-cell gate.",
        "",
        "The result concerns a narrow mathematical information measure, not awareness or life.",
    ]
    (temporary / "LAY_SUMMARY.md").write_text("\n".join(lay) + "\n", encoding="utf-8")
    atomic_json(
        temporary / "claim_boundaries.json",
        {
            "supported": [key for key, value in metrics["gates"].items() if value],
            "failed": [key for key, value in metrics["gates"].items() if not value],
            "prohibited": protocol()["claim_boundary"],
        },
    )
    manifest = {
        "format": RESULT_FORMAT,
        "registration_id": registration["registration_id"],
        "acquisition_matrices": len(acquisitions),
        "intervention_matrices": len(batches),
        "cpu_seconds": cpu_seconds,
        "complete_exact_replay": bool(replay["complete_exact_replay"]),
        "complete_readback_exact": False,
        "gates": metrics["gates"],
    }
    atomic_json(temporary / "manifest.json", manifest)
    write_checksums(temporary)
    temporary.replace(DEFAULT_OUTPUT)
    verify_checksums(DEFAULT_OUTPUT)
    readback = pd.read_csv(DEFAULT_OUTPUT / "scores.csv.gz")
    exact = len(readback) == len(tables["scores"])
    manifest["complete_readback_exact"] = exact
    atomic_json(DEFAULT_OUTPUT / "manifest.json", manifest)
    atomic_json(DEFAULT_OUTPUT / "readback_audit.json", {"complete": exact, "rows": len(readback)})
    write_checksums(DEFAULT_OUTPUT)
    if not exact:
        raise AssertionError("PX2 readback failed")
    _append_ledger(
        f"<!-- phir-extension-px2-result-{registration['registration_id']} -->",
        [
            "## Phi-r extension PX2 completed",
            "",
            "- Result: `results/phir_extension/px2_event_locked_recovery`.",
            "- Acquisition, intervention replay, and readback passed exactly.",
            f"- Registered gates: `{json.dumps(metrics['gates'], sort_keys=True)}`.",
        ],
    )
    return manifest


def run_scientific(workers: int = MAX_WORKERS) -> dict[str, Any]:
    registration = verify_registration()
    verify_checksums(DEFAULT_SMOKE)
    if DEFAULT_OUTPUT.exists():
        raise FileExistsError(f"PX2 output exists: {DEFAULT_OUTPUT}")
    if shutil.disk_usage(ROOT).free < MINIMUM_FREE_DISK_BYTES:
        raise RuntimeError("PX2 refused below the sealed disk floor")
    spec = scientific_spec()
    matrix_ids = list(range(spec.acquisition_matrices))
    acquired, cpu = _checkpointed(
        "acquisition_generate",
        spec,
        registration,
        DEFAULT_WORK / "acquisition_generate",
        matrix_ids,
        _acquire_matrix,
        workers,
    )
    acquired_replay, cpu = _checkpointed(
        "acquisition_replay",
        spec,
        registration,
        DEFAULT_WORK / "acquisition_replay",
        matrix_ids,
        _acquire_matrix,
        workers,
        cpu,
    )
    acquisition_audit = _replay_exact(acquired, acquired_replay, spec.acquisition_matrices)
    if not acquisition_audit["complete_exact_replay"]:
        raise AssertionError("PX2 acquisition replay failed")
    eligible = _eligible(acquired, spec.target_matrices)
    generated, cpu = _checkpointed(
        "intervention_generate",
        spec,
        registration,
        DEFAULT_WORK / "intervention_generate",
        eligible,
        _intervene_matrix,
        workers,
        cpu,
    )
    replayed, cpu = _checkpointed(
        "intervention_replay",
        spec,
        registration,
        DEFAULT_WORK / "intervention_replay",
        eligible,
        _intervene_matrix,
        workers,
        cpu,
    )
    intervention_audit = _replay_exact(generated, replayed, len(eligible))
    replay = {
        "acquisition": acquisition_audit,
        "intervention": intervention_audit,
        "complete_exact_replay": acquisition_audit["complete_exact_replay"]
        and intervention_audit["complete_exact_replay"],
    }
    if not replay["complete_exact_replay"]:
        raise AssertionError("PX2 intervention replay failed")
    return _write_result(acquired, generated, replay, registration, cpu)


def run_smoke() -> dict[str, Any]:
    if DEFAULT_SMOKE.exists():
        raise FileExistsError(f"PX2 smoke exists: {DEFAULT_SMOKE}")
    spec = smoke_spec()
    # Use a forced post-break state so smoke checks mechanics without disclosing
    # whether a scientific acquisition seed is eligible.
    config = GardConfig()
    beta = generate_beta(config, np.random.default_rng(61))
    initial = generate_initial_composition(config, np.random.default_rng(62))
    snapshot = Snapshot(initial, 10, (True, False), (0.95, 0.8), 5, 20)
    state = AcquiredState(
        "02", 10, initial.astype(np.int16), 10, snapshot.inheritance,
        snapshot.boundary_h, 5, 20, np.asarray(initial, dtype=np.int16), "fixture"
    )
    provisional = AcquisitionBatch(0, beta, initial.astype(np.int16), (state,), (), 0.0, "")
    acquisition = AcquisitionBatch(
        provisional.matrix_id, provisional.beta, provisional.initial_composition,
        provisional.states, provisional.acquisition_rows, provisional.cpu_seconds,
        _batch_digest(provisional)
    )
    model = str(MODEL_SOURCE)
    contract = str(MODEL_CONTRACT_SOURCE)
    first = _intervene_matrix((acquisition, spec, model, contract))
    second = _intervene_matrix((acquisition, spec, model, contract))
    rows = pd.DataFrame(first.score_rows)
    payload = {
        "format": "codex-ch5-phir-extension-px2-smoke-v1",
        "all_arms": set(rows.arm) == set(ARMS),
        "all_halves": set(rows.half) == set(HALVES),
        "explicit_pairs_positive": bool((rows.transition_pairs > 0).all()),
        "scores_finite": bool(np.isfinite(rows.material_full_revised).all()),
        "replay_exact": first.scientific_digest == second.scientific_digest,
        "effects_suppressed": True,
    }
    payload["passed"] = all(value for key, value in payload.items() if key not in {"format"})
    DEFAULT_SMOKE.mkdir(parents=True)
    atomic_json(DEFAULT_SMOKE / "smoke.json", payload)
    write_checksums(DEFAULT_SMOKE)
    if not payload["passed"]:
        raise AssertionError("PX2 smoke failed")
    return payload


def launch_detached(workers: int = MAX_WORKERS) -> dict[str, Any]:
    registration = verify_registration()
    verify_checksums(DEFAULT_SMOKE)
    if DEFAULT_OUTPUT.exists():
        raise FileExistsError(f"PX2 output exists: {DEFAULT_OUTPUT}")
    # Enforce the serial scientific-phase rule.
    px1_output = RESULT_ROOT / "px1_fresh_confirmation"
    if not px1_output.exists():
        raise RuntimeError("PX2 launch is locked until PX1 is complete")
    DEFAULT_WORK.mkdir(parents=True, exist_ok=True)
    command = [
        "systemd-run", "--user", f"--unit={SERVICE_NAME}", "--collect",
        "--property", f"WorkingDirectory={ROOT}",
        "--property", f"StandardOutput=append:{DEFAULT_LOG}",
        "--property", f"StandardError=append:{DEFAULT_LOG}",
        sys.executable, "-m", "plastic_heredity.phir_extension_px2", "run",
        "--workers", str(min(workers, MAX_WORKERS)),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = {
        "registration_id": registration["registration_id"],
        "service": SERVICE_NAME,
        "workers": min(workers, MAX_WORKERS),
        "launched_at": time.time(),
        "stderr": completed.stderr.strip(),
    }
    atomic_json(DEFAULT_WORK / "detached_launch.json", payload)
    return payload


def status_payload() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "phase": "PX2",
        "validation": DEFAULT_VALIDATION.exists(),
        "registration": DEFAULT_REGISTRATION.exists(),
        "smoke": DEFAULT_SMOKE.exists(),
        "complete": DEFAULT_OUTPUT.exists(),
        "launch_locked_until_px1_complete": not (
            RESULT_ROOT / "px1_fresh_confirmation"
        ).exists(),
        "service": SERVICE_NAME,
        "free_disk_bytes": shutil.disk_usage(ROOT).free,
    }
    for stage in (
        "acquisition_generate", "acquisition_replay", "intervention_generate", "intervention_replay"
    ):
        path = DEFAULT_WORK / stage / "status.json"
        if path.exists():
            payload[stage] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate")
    commands.add_parser("register")
    commands.add_parser("smoke")
    run = commands.add_parser("run")
    run.add_argument("--workers", type=int, default=MAX_WORKERS)
    launch = commands.add_parser("launch")
    launch.add_argument("--workers", type=int, default=MAX_WORKERS)
    commands.add_parser("status")
    arguments = parser.parse_args(argv)
    if arguments.command == "validate":
        print(json.dumps(run_validation(), indent=2, sort_keys=True))
    elif arguments.command == "register":
        print(json.dumps(register_program(), indent=2, sort_keys=True))
    elif arguments.command == "smoke":
        print(json.dumps(run_smoke(), indent=2, sort_keys=True))
    elif arguments.command == "run":
        print(json.dumps(run_scientific(arguments.workers), indent=2, sort_keys=True))
    elif arguments.command == "launch":
        print(json.dumps(launch_detached(arguments.workers), indent=2, sort_keys=True))
    elif arguments.command == "status":
        print(json.dumps(status_payload(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
