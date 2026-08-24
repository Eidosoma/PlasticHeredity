"""PX1 fresh generational and functional Phi-r confirmation.

This module is additive.  It uses new matrices and streams, preserves all
sealed Chapter 5 code, and exposes validate/register/smoke/run/launch/status
commands for a detached, checkpointed 24-matrix campaign.
"""

from __future__ import annotations

import argparse
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

from .config import CANDIDATES, GardConfig
from .intervention_core import (
    FrozenFullPredictor,
    MolecularEdit,
    _records_digest as sealed_records_digest,
    apply_molecular_edit,
    edited_snapshot,
    enumerate_legal_edits,
    score_legal_edits,
)
from .mechanistic import verify_checksums, write_checksums
from .phir_ch5 import _append_ledger, _snapshot_after_record
from .phir_extension_common import (
    BOOTSTRAP_DRAWS,
    CPU_BUDGET_HOURS,
    MASTER_DOCUMENT,
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
    master_protocol,
    paired_matrix_effects,
    paired_summary,
    purpose_seed,
    register_master,
    runtime_versions,
    safe_score_sequence,
    sha256_file,
    verify_master,
)
from .simulator import (
    FissionRecord,
    SimulationError,
    Snapshot,
    advance_fission,
    generate_beta,
    generate_initial_composition,
)


DOCUMENT = "CODEX_CH5_PHIR_EXTENSION_PREREGISTRATION.md"
DEFAULT_VALIDATION = RESULT_ROOT / "px0_validation"
DEFAULT_REGISTRATION = RESULT_ROOT / "px1_registration"
DEFAULT_SMOKE = RESULT_ROOT / "px1_smoke"
DEFAULT_OUTPUT = RESULT_ROOT / "px1_fresh_confirmation"
DEFAULT_WORK = RESULT_ROOT / ".px1_work"
DEFAULT_LOG = RESULT_ROOT / "px1_fresh_confirmation.log"

LABEL = "CODEX_CH5_PHIR_EXTENSION_PX1_V1"
PHASE_FORMAT = "codex-ch5-phir-extension-px1-program-v1"
REGISTRATION_FORMAT = "codex-ch5-phir-extension-px1-registration-v1"
CHECKPOINT_FORMAT = "codex-ch5-phir-extension-px1-checkpoint-v1"
RESULT_FORMAT = "codex-ch5-phir-extension-px1-result-v1"
STATUS_FORMAT = "codex-ch5-phir-extension-px1-status-v1"
SERVICE_NAME = "codex-phir-extension-px1-20260819"

MATRICES = 24
REPLICATES = 2
HORIZON = 60
FINAL_START = 30
CPU_ALLOCATION_SECONDS = 8.0 * 3600.0
ARMS = ("STABILIZE", "DESTABILIZE", "RANDOM", "NOOP")
PRIMARY_REPRESENTATIONS = ("material", "functional_flux")
MODEL_SOURCE = ROOT / "results" / "phir_protocol_adjudication_registration" / "frozen_full_predictor.npz"
EXPECTED_MODEL_SHA256 = "9b3305a7fed11f432651926d34903443e9413ed299c5d0f1056a0b5fde9990af"

ARCHIVED_INPUTS = ROOT / "results" / "phir_protocol_adjudication24" / "matrix_inputs.npz"
ARCHIVED_R0 = ROOT / "results" / "phir_rescue_r0" / "lineages.csv.gz"

MASTER_SOURCE_FILES = (
    DOCUMENT,
    "plastic_heredity/phir_extension_common.py",
    "tests/test_phir_extension_common.py",
)

SOURCE_FILES = (
    DOCUMENT,
    "plastic_heredity/phir_extension_px1.py",
    "plastic_heredity/phir_extension_common.py",
    "tests/test_phir_extension_px1.py",
    "tests/test_phir_extension_common.py",
    "plastic_heredity/config.py",
    "plastic_heredity/simulator.py",
    "plastic_heredity/features.py",
    "plastic_heredity/intervention_core.py",
    "plastic_heredity/phir_instruments.py",
    "plastic_heredity/phir_rescue_instruments.py",
    "plastic_heredity/phir_ch5.py",
    "plastic_heredity/seeds.py",
)


@dataclass(frozen=True)
class PX1Spec:
    label: str
    matrices: int
    replicates: int
    horizon: int
    final_start: int
    bootstrap_draws: int
    randomization_draws: int
    cpu_allocation_seconds: float


@dataclass(frozen=True)
class PX1Batch:
    matrix_id: int
    beta: NDArray[np.float64]
    initial_composition: NDArray[np.int16]
    lineage_rows: tuple[dict[str, Any], ...]
    selected_edit_rows: tuple[dict[str, Any], ...]
    cpu_seconds: float
    scientific_digest: str


def _batch_digest(batch: PX1Batch) -> str:
    """Digest scientific content while excluding nondeterministic CPU timing."""

    value = asdict(batch)
    value["cpu_seconds"] = 0.0
    value["scientific_digest"] = ""
    return canonical_digest(value)


def scientific_spec() -> PX1Spec:
    return PX1Spec(
        "scientific",
        MATRICES,
        REPLICATES,
        HORIZON,
        FINAL_START,
        BOOTSTRAP_DRAWS,
        RANDOMIZATION_DRAWS,
        CPU_ALLOCATION_SECONDS,
    )


def smoke_spec() -> PX1Spec:
    return PX1Spec("smoke", 1, 1, 6, 3, 32, 32, 300.0)


def _source_hashes() -> dict[str, str]:
    return {name: sha256_file(ROOT / name) for name in SOURCE_FILES}


def _matrix_seed(spec: PX1Spec, matrix_id: int, purpose: str) -> int:
    domain = "smoke" if spec.label == "smoke" else purpose
    return purpose_seed(domain, "PX1", spec.label, purpose, matrix_id)


def _future_seed(spec: PX1Spec, candidate: str, matrix_id: int, replicate: int) -> int:
    domain = "smoke" if spec.label == "smoke" else "future"
    return purpose_seed(
        domain, "PX1", spec.label, "future", candidate, matrix_id, replicate
    )


def _random_action_seed(
    spec: PX1Spec, candidate: str, matrix_id: int, replicate: int
) -> int:
    domain = "smoke" if spec.label == "smoke" else "random_action"
    return purpose_seed(
        domain, "PX1", spec.label, "random_action", candidate, matrix_id, replicate
    )


def _records_digest(records: Sequence[FissionRecord]) -> str:
    return canonical_digest(
        [
            {
                "parent": record.parent,
                "daughter": record.daughter,
                "h_bytes": np.asarray(record.h, dtype=np.float64).tobytes().hex(),
                "growth_steps": record.growth_steps,
            }
            for record in records
        ]
    )


def _extreme_choice(scores: Sequence[Any], stabilize: bool) -> Any:
    if not scores:
        raise ValueError("cannot choose an extreme from no legal edits")
    if stabilize:
        return min(
            scores,
            key=lambda item: (
                item.predicted_probability,
                item.edit.remove_type,
                item.edit.add_type,
            ),
        )
    return min(
        scores,
        key=lambda item: (
            -item.predicted_probability,
            item.edit.remove_type,
            item.edit.add_type,
        ),
    )


def _entropy(composition: NDArray) -> float:
    values = np.asarray(composition, dtype=np.float64)
    fraction = values / values.sum()
    positive = fraction[fraction > 0.0]
    return float(-np.sum(positive * np.log(positive)))


def _run_arm(
    matrix_id: int,
    candidate: str,
    replicate: int,
    arm: str,
    beta: NDArray,
    initial: NDArray,
    predictor: FrozenFullPredictor,
    spec: PX1Spec,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = GardConfig()
    rng = np.random.default_rng(_future_seed(spec, candidate, matrix_id, replicate))
    action_rng = np.random.default_rng(
        _random_action_seed(spec, candidate, matrix_id, replicate)
    )
    snapshot = Snapshot(np.asarray(initial, dtype=np.int64).copy(), 0, (), ())
    records: list[FissionRecord] = []
    boundary_states: list[NDArray[np.int64]] = []
    actions: list[dict[str, Any]] = []
    first_record_digest = ""
    for step in range(1, spec.horizon + 1):
        try:
            record = advance_fission(
                snapshot.composition,
                beta,
                config,
                CANDIDATES[candidate],
                rng,
            )
        except SimulationError:
            break
        if step == 1:
            first_record_digest = _records_digest((record,))
        records.append(record)
        snapshot = _snapshot_after_record(snapshot, record)
        edit: MolecularEdit | None = None
        noop_probability = float("nan")
        selected_probability = float("nan")
        legal_count = 0
        if step < spec.horizon and arm in ("STABILIZE", "DESTABILIZE"):
            noop_probability, scores = score_legal_edits(
                predictor, candidate, snapshot, beta, config
            )
            choice = _extreme_choice(scores, arm == "STABILIZE")
            edit = choice.edit
            selected_probability = choice.predicted_probability
            legal_count = len(scores)
        elif step < spec.horizon and arm == "RANDOM":
            legal = enumerate_legal_edits(snapshot.composition)
            legal_count = len(legal)
            if not legal:
                raise SimulationError("random controller reached a state with no legal edit")
            edit = legal[int(action_rng.integers(0, len(legal)))]
        if edit is not None:
            snapshot = edited_snapshot(snapshot, edit)
        boundary_states.append(snapshot.composition.copy())
        action = {
            "matrix_id": matrix_id,
            "candidate": candidate,
            "replicate": replicate,
            "arm": arm,
            "step": step,
            "edit_applied": int(edit is not None),
            "remove_type": -1 if edit is None else edit.remove_type,
            "add_type": -1 if edit is None else edit.add_type,
            "legal_edits": legal_count,
            "noop_probability": noop_probability,
            "selected_probability": selected_probability,
            "predicted_shift": selected_probability - noop_probability,
        }
        action["action_digest"] = canonical_digest(action)
        actions.append(action)
    complete = len(records) == spec.horizon
    information_eligible = complete and len(boundary_states) >= spec.final_start
    row: dict[str, Any] = {
        "matrix_id": matrix_id,
        "candidate": candidate,
        "replicate": replicate,
        "arm": arm,
        "completed_horizon": int(complete),
        "information_eligible": int(information_eligible),
        "completed_fissions": len(records),
        "extinct": int(not complete),
        "first_record_digest": first_record_digest,
        "record_digest": _records_digest(records),
        "final_rng_state_digest": canonical_digest(rng.bit_generator.state),
        "final_composition": snapshot.composition.astype(int).tolist(),
        "edits_applied": int(sum(item["edit_applied"] for item in actions)),
        "inherited_1_60": (
            float(np.mean([record.h > config.inheritance_threshold for record in records]))
            if records
            else float("nan")
        ),
        "inherited_31_60": (
            float(
                np.mean(
                    [
                        record.h > config.inheritance_threshold
                        for record in records[spec.final_start :]
                    ]
                )
            )
            if len(records) > spec.final_start
            else float("nan")
        ),
        "breaks_31_60": int(
            np.count_nonzero(
                [
                    record.h <= config.inheritance_threshold
                    for record in records[spec.final_start :]
                ]
            )
        ),
        "mean_growth_updates_31_60": (
            float(np.mean([record.growth_steps for record in records[spec.final_start :]]))
            if len(records) > spec.final_start
            else float("nan")
        ),
        "final_entropy": _entropy(snapshot.composition),
        "final_occupied_types": int(np.count_nonzero(snapshot.composition)),
        "final_top1_share": float(snapshot.composition.max() / snapshot.composition.sum()),
    }
    if information_eligible:
        counts = np.asarray(boundary_states[spec.final_start - 1 :], dtype=np.int64)
        # final_start=30 yields the fission-30 boundary through fission 60:
        # 31 unique observations and 30 adjacent transitions, exactly as R0.
        for representation in PRIMARY_REPRESENTATIONS:
            score = safe_score_sequence(counts, beta, representation, config)
            row.update(score.fields(representation))
        row["generational_observations"] = int(counts.shape[0])
    else:
        for representation in PRIMARY_REPRESENTATIONS:
            row.update(safe_score_sequence(np.empty((0, config.n_types)), beta, representation).fields(representation))
        row["generational_observations"] = 0
    return row, actions


def _run_matrix(args: tuple[int, PX1Spec, str]) -> PX1Batch:
    matrix_id, spec, model_path = args
    started = time.process_time()
    with threadpool_limits(limits=1):
        config = GardConfig()
        beta = generate_beta(
            config, np.random.default_rng(_matrix_seed(spec, matrix_id, "matrix"))
        )
        initial = generate_initial_composition(
            config, np.random.default_rng(_matrix_seed(spec, matrix_id, "initial"))
        )
        predictor = FrozenFullPredictor.load(model_path)
        rows: list[dict[str, Any]] = []
        edits: list[dict[str, Any]] = []
        for candidate in CANDIDATES:
            for replicate in range(spec.replicates):
                for arm in ARMS:
                    row, local = _run_arm(
                        matrix_id,
                        candidate,
                        replicate,
                        arm,
                        beta,
                        initial,
                        predictor,
                        spec,
                    )
                    rows.append(row)
                    edits.extend(local)
    provisional = PX1Batch(
        matrix_id,
        np.asarray(beta, dtype=np.float64),
        np.asarray(initial, dtype=np.int16),
        tuple(rows),
        tuple(edits),
        float(time.process_time() - started),
        "",
    )
    return PX1Batch(
        provisional.matrix_id,
        provisional.beta,
        provisional.initial_composition,
        provisional.lineage_rows,
        provisional.selected_edit_rows,
        provisional.cpu_seconds,
        _batch_digest(provisional),
    )


def _calibration_worker(matrix_id: int) -> tuple[int, tuple[dict[str, Any], ...]]:
    """Replay archived R0 NOOP paths to set margins without arm contrasts."""

    from .phir_protocol_adjudication import _future_seed as old_future_seed
    from .phir_protocol_adjudication import scientific_spec as old_specification

    with threadpool_limits(limits=1), np.load(ARCHIVED_INPUTS, allow_pickle=False) as archive:
        identifiers = archive["pab24_matrix_id"].astype(int)
        location = int(np.flatnonzero(identifiers == matrix_id)[0])
        beta = np.asarray(archive["pab24_beta"][location], dtype=np.float64)
        initial = np.asarray(archive["pab24_initial"][location], dtype=np.int64)
    config = GardConfig()
    old_spec = old_specification()
    rows: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        for replicate in range(REPLICATES):
            rng = np.random.default_rng(
                old_future_seed(old_spec, candidate, matrix_id, replicate)
            )
            snapshot = Snapshot(initial.copy(), 0, (), ())
            states: list[NDArray[np.int64]] = []
            records: list[FissionRecord] = []
            for step in range(1, HORIZON + 1):
                record = advance_fission(
                    snapshot.composition,
                    beta,
                    config,
                    CANDIDATES[candidate],
                    rng,
                )
                records.append(record)
                snapshot = _snapshot_after_record(snapshot, record)
                if step >= FINAL_START:
                    states.append(snapshot.composition.copy())
            counts = np.asarray(states, dtype=np.int64)
            material = safe_score_sequence(counts, beta, "material", config)
            flux = safe_score_sequence(counts, beta, "functional_flux", config)
            rows.append(
                {
                    "matrix_id": matrix_id,
                    "candidate": candidate,
                    "replicate": replicate,
                    "material": material.full_revised,
                    "functional_flux": flux.full_revised,
                    "record_digest": sealed_records_digest(records),
                }
            )
    return matrix_id, tuple(rows)


def calibrate_equivalence_margins(workers: int = MAX_WORKERS) -> dict[str, Any]:
    archived = pd.read_csv(ARCHIVED_R0)
    archived["candidate"] = archived["candidate"].astype(str).str.zfill(2)
    archived = archived[archived["arm"] == "FRESH__NOOP"].copy()
    arguments = range(MATRICES)
    if workers <= 1:
        generated = map(_calibration_worker, arguments)
    else:
        with ProcessPoolExecutor(max_workers=min(workers, MAX_WORKERS)) as executor:
            generated = executor.map(_calibration_worker, arguments, chunksize=1)
            generated = list(generated)
    rows = [row for _, local in generated for row in local]
    frame = pd.DataFrame(rows)
    joined = frame.merge(
        archived[
            [
                "matrix_id",
                "candidate",
                "replicate",
                "generational_full_block_raw",
                "controlled_record_digest",
            ]
        ],
        on=["matrix_id", "candidate", "replicate"],
        validate="one_to_one",
    )
    maximum_score_error = float(
        np.max(
            np.abs(
                joined["material"].to_numpy(float)
                - joined["generational_full_block_raw"].to_numpy(float)
            )
        )
    )
    record_mismatches = int(
        np.count_nonzero(
            joined["record_digest"].astype(str).to_numpy()
            != joined["controlled_record_digest"].astype(str).to_numpy()
        )
    )
    if record_mismatches or maximum_score_error > 1e-7:
        raise AssertionError("R0 NOOP calibration replay did not reproduce its archive")
    cells: dict[str, Any] = {}
    for candidate in CANDIDATES:
        for replicate in range(REPLICATES):
            selected = frame[
                (frame["candidate"] == candidate)
                & (frame["replicate"] == replicate)
            ]
            key = f"{candidate}_r{replicate}"
            cells[key] = {}
            for metric in PRIMARY_REPRESENTATIONS:
                standard_deviation = float(selected[metric].std(ddof=1))
                cells[key][metric] = {
                    "noop_matrix_sd": standard_deviation,
                    "equivalence_margin": 0.2 * standard_deviation,
                }
    return {
        "format": "codex-ch5-phir-extension-px1-calibration-v1",
        "source": "exact replay of archived R0 FRESH__NOOP lineages only",
        "intervention_contrasts_inspected": False,
        "matrices": MATRICES,
        "maximum_material_archive_absolute_error": maximum_score_error,
        "record_digest_mismatches": record_mismatches,
        "cells": cells,
    }


def phase_protocol(calibration: Mapping[str, Any]) -> dict[str, Any]:
    value: dict[str, Any] = {
        "format": PHASE_FORMAT,
        "phase": "PX1",
        "question": "fresh generational material and catalytic-flux confirmation",
        "spec": asdict(scientific_spec()),
        "arms": list(ARMS),
        "representations": list(PRIMARY_REPRESENTATIONS),
        "launch": "fresh mass-40 composition",
        "edits": "after fissions 1-59; no edit after fission 60",
        "model": {
            "source": str(MODEL_SOURCE.relative_to(ROOT)),
            "sha256": EXPECTED_MODEL_SHA256,
            "selection": "exhaustive legal substitution extrema",
            "stabilize": "minimum frozen JOINT_BREAK_RUN3 probability",
            "destabilize": "maximum frozen JOINT_BREAK_RUN3 probability",
        },
        "future_stream_key": ["phase", "candidate", "matrix", "replicate"],
        "arm_in_future_stream_key": False,
        "random_action_stream_separate": True,
        "measurement": "31 unique boundaries at fissions 30-60; 30 transitions",
        "primary_contrasts": "STABILIZE minus DESTABILIZE in four candidate-by-replicate cells",
        "primary_family": "eight representation-by-cell comparisons",
        "specificity": {
            "random_minus_noop": True,
            "heredity_margin": 0.025,
            "information_margins": calibration["cells"],
        },
        "gate": [
            "positive point effect",
            "positive 95% whole-matrix bootstrap lower bound",
            "Holm-adjusted one-sided randomization p < 0.05",
            "all leave-one-matrix-out effects positive",
            "RANDOM equivalent to NOOP",
            "heredity manipulation validity",
            "complete exact replay and readback",
        ],
        "matrix_inference": True,
        "candidate_pooling": False,
        "replicate_pooling": False,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "randomization_draws": RANDOMIZATION_DRAWS,
        "no_48_matrix_continuation": True,
        "claim_boundary": master_protocol()["claim_boundaries"],
    }
    value["protocol_id"] = canonical_digest(value)
    return value


def _ledger_entry(marker: str, lines: Sequence[str]) -> None:
    _append_ledger(marker, list(lines))


def register_program(workers: int = MAX_WORKERS) -> dict[str, Any]:
    if not DEFAULT_VALIDATION.exists():
        raise FileNotFoundError("PX0 validation must pass before registration")
    validation = json.loads(
        (DEFAULT_VALIDATION / "validation.json").read_text(encoding="utf-8")
    )
    if not validation.get("all_passed"):
        raise ValueError("PX0 validation is not registration-eligible")
    if not MASTER_REGISTRATION.exists():
        master = register_master(MASTER_SOURCE_FILES)
        shutil.copy2(MASTER_DOCUMENT, MASTER_REGISTRATION / "preregistration.md")
        write_checksums(MASTER_REGISTRATION)
        _ledger_entry(
            f"phir-extension-master-{master['registration_id']}",
            [
                "## Chapter 5 Phi-r extension master program registered",
                "",
                f"- Registration: `{master['registration_id']}`.",
                "- Six additive 24-matrix-or-smaller phases were fixed; no 48-matrix campaign is authorized.",
                "- Prior Chapter 5 results remain unchanged.",
            ],
        )
    master = verify_master(MASTER_SOURCE_FILES)
    if DEFAULT_REGISTRATION.exists():
        raise FileExistsError(f"PX1 registration exists: {DEFAULT_REGISTRATION}")
    calibration = calibrate_equivalence_margins(workers)
    protocol = phase_protocol(calibration)
    body: dict[str, Any] = {
        "format": REGISTRATION_FORMAT,
        "master_registration_id": master["registration_id"],
        "protocol": protocol,
        "source_hashes": _source_hashes(),
        "runtime": runtime_versions(),
        "model_sha256": sha256_file(MODEL_SOURCE),
        "archived_inputs_sha256": sha256_file(ARCHIVED_INPUTS),
        "archived_r0_sha256": sha256_file(ARCHIVED_R0),
        "calibration": calibration,
        "new_scientific_matrices_at_registration": 0,
    }
    if body["model_sha256"] != EXPECTED_MODEL_SHA256:
        raise ValueError("frozen PX1 predictor hash changed")
    body["registration_id"] = canonical_digest(body)
    DEFAULT_REGISTRATION.mkdir(parents=True, exist_ok=False)
    shutil.copy2(MASTER_DOCUMENT, DEFAULT_REGISTRATION / "preregistration.md")
    shutil.copy2(MODEL_SOURCE, DEFAULT_REGISTRATION / "frozen_full_predictor.npz")
    atomic_json(DEFAULT_REGISTRATION / "protocol.json", protocol)
    atomic_json(DEFAULT_REGISTRATION / "calibration.json", calibration)
    atomic_json(DEFAULT_REGISTRATION / "registration.json", body)
    write_checksums(DEFAULT_REGISTRATION)
    _ledger_entry(
        f"phir-extension-px1-registration-{body['registration_id']}",
        [
            "## Phi-r extension PX1 registered",
            "",
            f"- Registration: `{body['registration_id']}`.",
            "- Twenty-four fresh matrices, both candidates, two replicates, material and functional-flux readings.",
            "- Scientific output did not exist at registration; no 48-matrix continuation is authorized.",
        ],
    )
    return body


def verify_registration() -> dict[str, Any]:
    verify_checksums(DEFAULT_REGISTRATION)
    master = verify_master(MASTER_SOURCE_FILES)
    body = json.loads(
        (DEFAULT_REGISTRATION / "registration.json").read_text(encoding="utf-8")
    )
    observed = body.pop("registration_id")
    if body.get("format") != REGISTRATION_FORMAT or observed != canonical_digest(body):
        raise ValueError("PX1 registration identity failed")
    body["registration_id"] = observed
    if body["master_registration_id"] != master["registration_id"]:
        raise ValueError("PX1 master registration changed")
    if body["source_hashes"] != _source_hashes():
        raise ValueError("PX1 source changed after registration")
    if body["model_sha256"] != sha256_file(
        DEFAULT_REGISTRATION / "frozen_full_predictor.npz"
    ):
        raise ValueError("PX1 frozen model changed")
    if body["protocol"] != canonical_json(phase_protocol(body["calibration"])):
        raise ValueError("PX1 protocol implementation changed")
    return body


def _batch_valid(batch: Any, matrix_id: int) -> bool:
    return bool(
        isinstance(batch, PX1Batch)
        and batch.matrix_id == matrix_id
        and batch.scientific_digest == _batch_digest(batch)
    )


def _checkpoint_contract(
    spec: PX1Spec, registration: Mapping[str, Any], stage: str
) -> dict[str, Any]:
    value = {
        "format": CHECKPOINT_FORMAT,
        "registration_id": registration["registration_id"],
        "protocol_id": registration["protocol"]["protocol_id"],
        "stage": stage,
        "spec": asdict(spec),
        "source_hashes": registration["source_hashes"],
    }
    value["contract_id"] = canonical_digest(value)
    return value


def _write_status(stage: str, completed: int, total: int, **extra: Any) -> None:
    started = DEFAULT_WORK / f"started_{stage}.txt"
    if not started.exists():
        started.parent.mkdir(parents=True, exist_ok=True)
        started.write_text(str(time.time()), encoding="ascii")
    elapsed = max(0.0, time.time() - float(started.read_text(encoding="ascii")))
    rate = completed / elapsed if completed and elapsed else 0.0
    atomic_json(
        DEFAULT_WORK / "campaign_status.json",
        {
            "format": STATUS_FORMAT,
            "stage": stage,
            "completed": completed,
            "total": total,
            "fraction": completed / total if total else 1.0,
            "elapsed_seconds": elapsed,
            "eta_seconds": (total - completed) / rate if rate else None,
            "pid": os.getpid(),
            "free_disk_bytes": shutil.disk_usage(ROOT).free,
            **extra,
        },
    )


def _run_checkpointed(
    spec: PX1Spec,
    registration: Mapping[str, Any],
    directory: Path,
    stage: str,
    workers: int,
    prior_cpu_seconds: float = 0.0,
) -> tuple[list[PX1Batch], float]:
    directory.mkdir(parents=True, exist_ok=True)
    contract = _checkpoint_contract(spec, registration, stage)
    contract_path = directory / "checkpoint_contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text(encoding="utf-8")) != canonical_json(contract):
            raise ValueError(f"PX1 checkpoint contract changed: {directory}")
    else:
        atomic_json(contract_path, contract)
    batches: list[PX1Batch | None] = [None] * spec.matrices
    missing: list[int] = []
    cpu_seconds = float(prior_cpu_seconds)
    for matrix_id in range(spec.matrices):
        path = directory / f"matrix_{matrix_id:04d}.pkl"
        if not path.exists():
            missing.append(matrix_id)
            continue
        with path.open("rb") as handle:
            batch = pickle.load(handle)
        if not _batch_valid(batch, matrix_id):
            raise ValueError(f"invalid PX1 checkpoint: {path}")
        batches[matrix_id] = batch
        cpu_seconds += batch.cpu_seconds
    completed = spec.matrices - len(missing)
    _write_status(stage, completed, spec.matrices, cpu_seconds=cpu_seconds, reused=completed)
    model_path = str(DEFAULT_REGISTRATION / "frozen_full_predictor.npz")
    arguments = [(matrix_id, spec, model_path) for matrix_id in missing]
    executor: ProcessPoolExecutor | None = None
    generated: Iterable[PX1Batch]
    if workers <= 1:
        generated = map(_run_matrix, arguments)
    else:
        executor = ProcessPoolExecutor(max_workers=min(workers, MAX_WORKERS))
        generated = executor.map(_run_matrix, arguments, chunksize=1)
    try:
        for matrix_id, batch in zip(missing, generated, strict=True):
            if not _batch_valid(batch, matrix_id):
                raise AssertionError("PX1 worker returned an invalid batch")
            batches[matrix_id] = batch
            atomic_pickle(directory / f"matrix_{matrix_id:04d}.pkl", batch)
            cpu_seconds += batch.cpu_seconds
            completed += 1
            _write_status(
                stage,
                completed,
                spec.matrices,
                cpu_seconds=cpu_seconds,
                cpu_budget_seconds=spec.cpu_allocation_seconds,
                reused=spec.matrices - len(missing),
            )
            print(f"[{stage}] {completed}/{spec.matrices} matrices", flush=True)
            if cpu_seconds > spec.cpu_allocation_seconds:
                raise RuntimeError("PX1 CPU allocation reached; checkpoints preserved")
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
    if any(batch is None for batch in batches):
        raise AssertionError("PX1 checkpoint stage incomplete")
    return [batch for batch in batches if batch is not None], cpu_seconds


def _replay_audit(
    generated: Sequence[PX1Batch], replayed: Sequence[PX1Batch]
) -> dict[str, Any]:
    rows = [
        {
            "matrix_id": left.matrix_id,
            "generated_digest": left.scientific_digest,
            "replay_digest": right.scientific_digest,
            "exact": left.scientific_digest == right.scientific_digest,
        }
        for left, right in zip(generated, replayed, strict=True)
    ]
    return {
        "format": "codex-ch5-phir-extension-px1-replay-v1",
        "matrices": rows,
        "complete_exact_replay": bool(
            len(rows) == MATRICES and all(row["exact"] for row in rows)
        ),
    }


def _normalized_frame(batches: Sequence[PX1Batch]) -> pd.DataFrame:
    frame = pd.DataFrame([row for batch in batches for row in batch.lineage_rows])
    frame["candidate"] = frame["candidate"].astype(str).str.zfill(2)
    return frame


def analyze_batches(
    batches: Sequence[PX1Batch], spec: PX1Spec, calibration: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, pd.DataFrame], dict[str, NDArray]]:
    frame = _normalized_frame(batches)
    arrays: dict[str, NDArray] = {}
    matrix_rows: list[dict[str, Any]] = []
    primary: list[dict[str, Any]] = []
    specificity: list[dict[str, Any]] = []
    heredity: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        for replicate in range(spec.replicates):
            cell = f"{candidate}_r{replicate}"
            filters = {"candidate": candidate, "replicate": replicate}
            for representation in PRIMARY_REPRESENTATIONS:
                metric = f"{representation}_full_revised"
                values = paired_matrix_effects(
                    frame, metric, "STABILIZE", "DESTABILIZE", filters=filters
                )
                summary, local_arrays = paired_summary(
                    values.to_numpy(),
                    f"PX1/{representation}/{cell}/stabilize-destabilize",
                    bootstrap_draws=spec.bootstrap_draws,
                    randomization_draws=spec.randomization_draws,
                )
                summary.update(
                    {
                        "family": "primary_information",
                        "representation": representation,
                        "metric": metric,
                        "candidate": candidate,
                        "replicate": replicate,
                        "contrast": "STABILIZE-DESTABILIZE",
                    }
                )
                primary.append(summary)
                for matrix_id, value in values.items():
                    matrix_rows.append(
                        {
                            "family": "primary_information",
                            "representation": representation,
                            "candidate": candidate,
                            "replicate": replicate,
                            "matrix_id": int(matrix_id),
                            "contrast": "STABILIZE-DESTABILIZE",
                            "value": float(value),
                        }
                    )
                arrays.update(
                    {
                        f"primary__{representation}__{cell}__{name}": value
                        for name, value in local_arrays.items()
                    }
                )
                random_values = paired_matrix_effects(
                    frame, metric, "RANDOM", "NOOP", filters=filters
                )
                margin = float(calibration["cells"][cell][representation]["equivalence_margin"])
                random_summary, local_arrays = paired_summary(
                    random_values.to_numpy(),
                    f"PX1/{representation}/{cell}/random-noop",
                    bootstrap_draws=spec.bootstrap_draws,
                    randomization_draws=spec.randomization_draws,
                    equivalence_margin=margin,
                )
                random_summary.update(
                    {
                        "family": "specificity",
                        "representation": representation,
                        "metric": metric,
                        "candidate": candidate,
                        "replicate": replicate,
                        "contrast": "RANDOM-NOOP",
                    }
                )
                specificity.append(random_summary)
                arrays.update(
                    {
                        f"specificity__{representation}__{cell}__{name}": value
                        for name, value in local_arrays.items()
                    }
                )
            inherited_values = paired_matrix_effects(
                frame,
                "inherited_31_60",
                "STABILIZE",
                "DESTABILIZE",
                filters=filters,
            )
            inherited_summary, local_arrays = paired_summary(
                inherited_values.to_numpy(),
                f"PX1/heredity/{cell}/stabilize-destabilize",
                bootstrap_draws=spec.bootstrap_draws,
                randomization_draws=spec.randomization_draws,
            )
            inherited_summary.update(
                {
                    "family": "heredity_validity",
                    "metric": "inherited_31_60",
                    "candidate": candidate,
                    "replicate": replicate,
                    "contrast": "STABILIZE-DESTABILIZE",
                }
            )
            heredity.append(inherited_summary)
            arrays.update(
                {
                    f"heredity__{cell}__{name}": value
                    for name, value in local_arrays.items()
                }
            )
            random_values = paired_matrix_effects(
                frame,
                "inherited_31_60",
                "RANDOM",
                "NOOP",
                filters=filters,
            )
            random_summary, local_arrays = paired_summary(
                random_values.to_numpy(),
                f"PX1/heredity/{cell}/random-noop",
                bootstrap_draws=spec.bootstrap_draws,
                randomization_draws=spec.randomization_draws,
                equivalence_margin=0.025,
            )
            random_summary.update(
                {
                    "family": "heredity_specificity",
                    "metric": "inherited_31_60",
                    "candidate": candidate,
                    "replicate": replicate,
                    "contrast": "RANDOM-NOOP",
                }
            )
            specificity.append(random_summary)
            arrays.update(
                {
                    f"heredity_specificity__{cell}__{name}": value
                    for name, value in local_arrays.items()
                }
            )
    apply_holm(primary)
    apply_holm(heredity)
    heredity_specificity = {
        (row["candidate"], row["replicate"]): row
        for row in specificity
        if row["family"] == "heredity_specificity"
    }
    information_specificity = {
        (row["representation"], row["candidate"], row["replicate"]): row
        for row in specificity
        if row["family"] == "specificity"
    }
    heredity_pass = all(
        row["effect"] > 0.0
        and row["ci95"][0] > 0.0
        and row.get("holm_adjusted_p", 1.0) < 0.05
        and heredity_specificity[(row["candidate"], row["replicate"])].get(
            "tost_via_90ci", False
        )
        for row in heredity
    ) and len(heredity) == 4
    representation_gates: dict[str, bool] = {}
    for representation in PRIMARY_REPRESENTATIONS:
        selected = [row for row in primary if row["representation"] == representation]
        representation_gates[representation] = bool(
            heredity_pass
            and len(selected) == 4
            and all(
                row["effect"] > 0.0
                and row["ci95"][0] > 0.0
                and row.get("holm_adjusted_p", 1.0) < 0.05
                and row["leave_one_matrix_out_all_positive"]
                and information_specificity[
                    (representation, row["candidate"], row["replicate"])
                ].get("tost_via_90ci", False)
                for row in selected
            )
        )
    arm_means = (
        frame.groupby(["candidate", "replicate", "arm"], sort=True)[
            [
                "inherited_31_60",
                "material_full_revised",
                "functional_flux_full_revised",
            ]
        ]
        .mean()
        .reset_index()
    )
    metrics = {
        "format": "codex-ch5-phir-extension-px1-metrics-v1",
        "primary": primary,
        "specificity": specificity,
        "heredity_validity": heredity,
        "arm_means": arm_means.to_dict(orient="records"),
        "gates": {
            "heredity_manipulation_validity": heredity_pass,
            "material_confirmation": representation_gates["material"],
            "functional_flux_confirmation": representation_gates["functional_flux"],
        },
        "completion": {
            "expected_lineages": spec.matrices * len(CANDIDATES) * spec.replicates * len(ARMS),
            "observed_lineages": int(len(frame)),
            "information_eligible": int(frame["information_eligible"].sum()),
            "extinct": int(frame["extinct"].sum()),
        },
    }
    tables = {
        "lineages": frame,
        "matrix_effects": pd.DataFrame(matrix_rows),
        "arm_means": arm_means,
    }
    return metrics, tables, arrays


def _report(metrics: Mapping[str, Any], registration_id: str) -> tuple[str, str]:
    lines = [
        "# PX1 fresh generational and functional Phi-r confirmation",
        "",
        f"Registration: `{registration_id}`.",
        "",
        "All effects are STABILIZE minus DESTABILIZE. Catalytic matrix is the inference unit.",
        "",
        "| Representation | Candidate | Replicate | Effect [95% CI] | Holm p | LOO+ |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in metrics["primary"]:
        lines.append(
            f"| {row['representation']} | {row['candidate']} | {row['replicate']} | "
            f"{row['effect']:+.5f} [{row['ci95'][0]:+.5f}, {row['ci95'][1]:+.5f}] | "
            f"{row.get('holm_adjusted_p', float('nan')):.4g} | "
            f"{row['leave_one_matrix_out_all_positive']} |"
        )
    lines.extend(
        [
            "",
            "## Registered gates",
            "",
            "```json",
            json.dumps(metrics["gates"], sort_keys=True, indent=2),
            "```",
            "",
            "## Claim boundary",
            "",
            "A positive result is a fresh confirmation only for the named representation and scale. It does not overwrite the negative public nine-atom result and does not establish consciousness, agency, life, or Phi as a cause.",
        ]
    )
    gates = metrics["gates"]
    lay = [
        "# PX1 lay summary",
        "",
        "We again pushed otherwise matched molecular assemblies toward more-stable or less-stable heredity. We then asked two versions of the information question: one based on which molecules were present and one based on the catalytic joining/leaving activity those molecules generated.",
        "",
        f"The hereditary manipulation itself {'worked cleanly' if gates['heredity_manipulation_validity'] else 'did not meet every strict validity condition'}.",
        f"The material-composition Phi test {'passed' if gates['material_confirmation'] else 'did not pass'} its complete four-cell gate.",
        f"The catalytic-flux Phi test {'passed' if gates['functional_flux_confirmation'] else 'did not pass'} its complete four-cell gate.",
        "",
        "These are narrow information measurements, not a test of consciousness or life.",
    ]
    return "\n".join(lines) + "\n", "\n".join(lay) + "\n"


def _write_result(
    batches: Sequence[PX1Batch],
    replay: Mapping[str, Any],
    registration: Mapping[str, Any],
    total_cpu_seconds: float,
) -> dict[str, Any]:
    metrics, tables, arrays = analyze_batches(
        batches, scientific_spec(), registration["calibration"]
    )
    temporary = DEFAULT_OUTPUT.with_name(DEFAULT_OUTPUT.name + f".tmp-{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    tables["lineages"].to_csv(temporary / "lineages.csv.gz", index=False)
    tables["matrix_effects"].to_csv(temporary / "matrix_effects.csv.gz", index=False)
    tables["arm_means"].to_csv(temporary / "arm_means.csv.gz", index=False)
    edits = pd.DataFrame(
        [row for batch in batches for row in batch.selected_edit_rows]
    )
    edits.to_csv(temporary / "selected_edits.csv.gz", index=False)
    np.savez_compressed(
        temporary / "matrix_inputs.npz",
        matrix_id=np.asarray([batch.matrix_id for batch in batches], dtype=np.int16),
        beta=np.stack([batch.beta for batch in batches]),
        initial=np.stack([batch.initial_composition for batch in batches]),
        digest=np.asarray([batch.scientific_digest for batch in batches]),
    )
    np.savez_compressed(temporary / "inference_arrays.npz", **arrays)
    atomic_json(temporary / "primary_metrics.json", metrics)
    atomic_json(temporary / "replay_audit.json", replay)
    scientific, lay = _report(metrics, registration["registration_id"])
    (temporary / "SCIENTIFIC_REPORT.md").write_text(scientific, encoding="utf-8")
    (temporary / "LAY_SUMMARY.md").write_text(lay, encoding="utf-8")
    claims = {
        "supported": [
            name
            for name, passed in metrics["gates"].items()
            if passed
        ],
        "failed": [
            name
            for name, passed in metrics["gates"].items()
            if not passed
        ],
        "prohibited": master_protocol()["claim_boundaries"],
    }
    atomic_json(temporary / "claim_boundaries.json", claims)
    manifest = {
        "format": RESULT_FORMAT,
        "registration_id": registration["registration_id"],
        "matrices": MATRICES,
        "lineages": int(len(tables["lineages"])),
        "cpu_seconds": float(total_cpu_seconds),
        "complete_exact_replay": bool(replay["complete_exact_replay"]),
        "complete_readback_exact": False,
        "gates": metrics["gates"],
    }
    atomic_json(temporary / "manifest.json", manifest)
    write_checksums(temporary)
    if DEFAULT_OUTPUT.exists():
        raise FileExistsError(f"PX1 output exists: {DEFAULT_OUTPUT}")
    temporary.replace(DEFAULT_OUTPUT)
    verify_checksums(DEFAULT_OUTPUT)
    readback = pd.read_csv(DEFAULT_OUTPUT / "lineages.csv.gz")
    readback_exact = bool(
        len(readback) == len(tables["lineages"])
        and set(readback["record_digest"].astype(str))
        == set(tables["lineages"]["record_digest"].astype(str))
    )
    manifest["complete_readback_exact"] = readback_exact
    atomic_json(DEFAULT_OUTPUT / "manifest.json", manifest)
    atomic_json(
        DEFAULT_OUTPUT / "readback_audit.json",
        {"complete_readback_exact": readback_exact, "rows": int(len(readback))},
    )
    write_checksums(DEFAULT_OUTPUT)
    if not readback_exact:
        raise AssertionError("PX1 output readback failed")
    _ledger_entry(
        f"phir-extension-px1-result-{registration['registration_id']}",
        [
            "## Phi-r extension PX1 completed",
            "",
            "- Result: `results/phir_extension/px1_fresh_confirmation`.",
            "- Complete exact replay and readback passed.",
            f"- Registered gates: `{json.dumps(metrics['gates'], sort_keys=True)}`.",
            "- Prior Chapter 5 results remain unchanged; no 48-matrix run was launched.",
        ],
    )
    return manifest


def run_scientific(workers: int = MAX_WORKERS) -> dict[str, Any]:
    registration = verify_registration()
    if DEFAULT_OUTPUT.exists():
        raise FileExistsError(f"PX1 output exists: {DEFAULT_OUTPUT}")
    if shutil.disk_usage(ROOT).free < MINIMUM_FREE_DISK_BYTES:
        raise RuntimeError("PX1 refused below the sealed 1.5 GB disk floor")
    spec = scientific_spec()
    try:
        generated, cpu_seconds = _run_checkpointed(
            spec, registration, DEFAULT_WORK / "generation", "generation", workers
        )
        replayed, total_cpu = _run_checkpointed(
            spec,
            registration,
            DEFAULT_WORK / "replay",
            "replay",
            workers,
            prior_cpu_seconds=cpu_seconds,
        )
        replay = _replay_audit(generated, replayed)
        if not replay["complete_exact_replay"]:
            raise AssertionError("PX1 complete exact replay failed")
        manifest = _write_result(generated, replay, registration, total_cpu)
        _write_status("complete", MATRICES, MATRICES, cpu_seconds=total_cpu)
        return manifest
    except BaseException as error:
        _write_status(
            "failed",
            0,
            MATRICES,
            error_type=type(error).__name__,
            error=str(error),
        )
        raise


def _plain_noop_fixture() -> bool:
    spec = smoke_spec()
    config = GardConfig()
    beta = generate_beta(config, np.random.default_rng(100))
    initial = generate_initial_composition(config, np.random.default_rng(101))
    seed = _future_seed(spec, "02", 0, 0)
    left_rng = np.random.default_rng(seed)
    right_rng = np.random.default_rng(seed)
    left = np.asarray(initial, dtype=np.int64).copy()
    right = np.asarray(initial, dtype=np.int64).copy()
    for _ in range(spec.horizon):
        left_record = advance_fission(left, beta, config, CANDIDATES["02"], left_rng)
        right_record = advance_fission(right, beta, config, CANDIDATES["02"], right_rng)
        if _records_digest((left_record,)) != _records_digest((right_record,)):
            return False
        left = left_record.daughter
        right = right_record.daughter
    return bool(
        np.array_equal(left, right)
        and canonical_digest(left_rng.bit_generator.state)
        == canonical_digest(right_rng.bit_generator.state)
    )


def validation_checks() -> dict[str, bool]:
    config = GardConfig()
    composition = np.zeros(config.n_types, dtype=np.int64)
    composition[:4] = (10, 12, 9, 9)
    legal = enumerate_legal_edits(composition)
    edit = legal[0]
    changed = apply_molecular_edit(composition, edit)
    return {
        "master_document_exists": MASTER_DOCUMENT.exists(),
        "old_model_hash_exact": MODEL_SOURCE.exists()
        and sha256_file(MODEL_SOURCE) == EXPECTED_MODEL_SHA256,
        "archived_r0_inputs_exist": ARCHIVED_INPUTS.exists() and ARCHIVED_R0.exists(),
        "scientific_matrix_scale_24": scientific_spec().matrices == 24,
        "no_48_matrix_campaign": master_protocol()["no_48_matrix_campaign"],
        "all_phases_run_without_evidence_gate": master_protocol()[
            "run_all_phases_without_evidence_gating"
        ],
        "cpu_budget_exact": CPU_BUDGET_HOURS == 80.0,
        "px1_cpu_allocation_exact": scientific_spec().cpu_allocation_seconds
        == 8.0 * 3600.0,
        "arms_exact": ARMS == ("STABILIZE", "DESTABILIZE", "RANDOM", "NOOP"),
        "representations_exact": PRIMARY_REPRESENTATIONS
        == ("material", "functional_flux"),
        "legal_edit_mass_preserved": int(changed.sum()) == int(composition.sum()),
        "legal_edit_nonnegative_integer": np.issubdtype(changed.dtype, np.integer)
        and bool(np.all(changed >= 0)),
        "random_and_future_streams_distinct": _future_seed(smoke_spec(), "02", 0, 0)
        != _random_action_seed(smoke_spec(), "02", 0, 0),
        "future_seed_arm_free": _future_seed(smoke_spec(), "02", 0, 0)
        == _future_seed(smoke_spec(), "02", 0, 0),
        "noop_plain_bitwise_identity": _plain_noop_fixture(),
        "final_edit_forbidden": HORIZON == 60,
        "bootstrap_fixed": BOOTSTRAP_DRAWS == 4096,
        "randomization_fixed": RANDOMIZATION_DRAWS == 4096,
        "workers_bounded": MAX_WORKERS == 12,
        "disk_floor_fixed": MINIMUM_FREE_DISK_BYTES == 1_500_000_000,
    }


def run_validation() -> dict[str, Any]:
    checks = validation_checks()
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_phir_extension_common.py",
            "tests/test_phir_extension_px1.py",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    payload = {
        "format": "codex-ch5-phir-extension-px0-validation-v1",
        "checks": checks,
        "pytest_returncode": completed.returncode,
        "pytest_stdout": completed.stdout,
        "pytest_stderr": completed.stderr,
        "all_passed": bool(all(checks.values()) and completed.returncode == 0),
        "runtime": runtime_versions(),
    }
    if DEFAULT_VALIDATION.exists():
        shutil.rmtree(DEFAULT_VALIDATION)
    DEFAULT_VALIDATION.mkdir(parents=True)
    atomic_json(DEFAULT_VALIDATION / "validation.json", payload)
    write_checksums(DEFAULT_VALIDATION)
    if not payload["all_passed"]:
        raise AssertionError(f"PX0 validation failed\n{completed.stdout}\n{completed.stderr}")
    return payload


def run_smoke() -> dict[str, Any]:
    if DEFAULT_SMOKE.exists():
        raise FileExistsError(f"PX1 smoke exists: {DEFAULT_SMOKE}")
    spec = smoke_spec()
    batch = _run_matrix((0, spec, str(MODEL_SOURCE)))
    replay = _run_matrix((0, spec, str(MODEL_SOURCE)))
    rows = pd.DataFrame(batch.lineage_rows)
    payload = {
        "format": "codex-ch5-phir-extension-px1-smoke-v1",
        "matrices": 1,
        "lineages": int(len(rows)),
        "all_arms_present": set(rows["arm"]) == set(ARMS),
        "future_prefix_paired": bool(
            rows.groupby(["candidate", "replicate"])["first_record_digest"]
            .nunique()
            .eq(1)
            .all()
        ),
        "complete": bool(rows["completed_horizon"].all()),
        "information_finite": bool(
            np.isfinite(rows["material_full_revised"]).all()
            and np.isfinite(rows["functional_flux_full_revised"]).all()
        ),
        "replay_exact": batch.scientific_digest == replay.scientific_digest,
        "effect_sizes_suppressed": True,
    }
    payload["passed"] = bool(
        payload["all_arms_present"]
        and payload["future_prefix_paired"]
        and payload["complete"]
        and payload["information_finite"]
        and payload["replay_exact"]
    )
    DEFAULT_SMOKE.mkdir(parents=True)
    atomic_json(DEFAULT_SMOKE / "smoke.json", payload)
    write_checksums(DEFAULT_SMOKE)
    if not payload["passed"]:
        raise AssertionError("PX1 smoke failed")
    return payload


def launch_detached(workers: int = MAX_WORKERS) -> dict[str, Any]:
    registration = verify_registration()
    verify_checksums(DEFAULT_SMOKE)
    smoke = json.loads((DEFAULT_SMOKE / "smoke.json").read_text(encoding="utf-8"))
    if not smoke.get("passed"):
        raise ValueError("PX1 smoke did not pass")
    if DEFAULT_OUTPUT.exists():
        raise FileExistsError(f"PX1 output exists: {DEFAULT_OUTPUT}")
    if shutil.disk_usage(ROOT).free < MINIMUM_FREE_DISK_BYTES:
        raise RuntimeError("PX1 detached launch refused below disk floor")
    DEFAULT_WORK.mkdir(parents=True, exist_ok=True)
    command = [
        "systemd-run",
        "--user",
        f"--unit={SERVICE_NAME}",
        "--collect",
        "--property",
        f"WorkingDirectory={ROOT}",
        "--property",
        f"StandardOutput=append:{DEFAULT_LOG}",
        "--property",
        f"StandardError=append:{DEFAULT_LOG}",
        sys.executable,
        "-m",
        "plastic_heredity.phir_extension_px1",
        "run",
        "--workers",
        str(min(workers, MAX_WORKERS)),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = {
        "format": "codex-ch5-phir-extension-px1-detached-launch-v1",
        "registration_id": registration["registration_id"],
        "service": SERVICE_NAME,
        "workers": min(workers, MAX_WORKERS),
        "launched_at_unix": time.time(),
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }
    atomic_json(DEFAULT_WORK / "detached_launch.json", payload)
    return payload


def status_payload() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "phase": "PX1",
        "validation": DEFAULT_VALIDATION.exists(),
        "master_registration": MASTER_REGISTRATION.exists(),
        "registration": DEFAULT_REGISTRATION.exists(),
        "smoke": DEFAULT_SMOKE.exists(),
        "complete": DEFAULT_OUTPUT.exists(),
        "service": SERVICE_NAME,
        "free_disk_bytes": shutil.disk_usage(ROOT).free,
        "minimum_free_disk_bytes": MINIMUM_FREE_DISK_BYTES,
        "no_48_matrix_campaign": True,
    }
    status = DEFAULT_WORK / "campaign_status.json"
    if status.exists():
        payload["campaign"] = json.loads(status.read_text(encoding="utf-8"))
    launch = DEFAULT_WORK / "detached_launch.json"
    if launch.exists():
        payload["detached_launch"] = json.loads(launch.read_text(encoding="utf-8"))
    return payload


def verify_result() -> dict[str, Any]:
    verify_checksums(DEFAULT_OUTPUT)
    registration = verify_registration()
    manifest = json.loads((DEFAULT_OUTPUT / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != RESULT_FORMAT:
        raise ValueError("unsupported PX1 result")
    if manifest.get("registration_id") != registration["registration_id"]:
        raise ValueError("PX1 result registration mismatch")
    if not manifest.get("complete_exact_replay") or not manifest.get(
        "complete_readback_exact"
    ):
        raise ValueError("PX1 result integrity failed")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate")
    register = subparsers.add_parser("register")
    register.add_argument("--workers", type=int, default=MAX_WORKERS)
    subparsers.add_parser("smoke")
    run = subparsers.add_parser("run")
    run.add_argument("--workers", type=int, default=MAX_WORKERS)
    launch = subparsers.add_parser("launch")
    launch.add_argument("--workers", type=int, default=MAX_WORKERS)
    subparsers.add_parser("status")
    subparsers.add_parser("verify")
    arguments = parser.parse_args(argv)
    if arguments.command == "validate":
        print(json.dumps(run_validation(), sort_keys=True, indent=2))
    elif arguments.command == "register":
        print(json.dumps(register_program(arguments.workers), sort_keys=True, indent=2))
    elif arguments.command == "smoke":
        print(json.dumps(run_smoke(), sort_keys=True, indent=2))
    elif arguments.command == "run":
        print(json.dumps(run_scientific(arguments.workers), sort_keys=True, indent=2))
    elif arguments.command == "launch":
        print(json.dumps(launch_detached(arguments.workers), sort_keys=True, indent=2))
    elif arguments.command == "status":
        print(json.dumps(status_payload(), sort_keys=True, indent=2))
    elif arguments.command == "verify":
        print(json.dumps(verify_result(), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
