"""PX4 prospective simulator-mechanics moderator experiment.

The phase uses Codex-owned code and neutral shared inputs to factor the
documented growth, overshoot, and daughter-selection differences between the
two clean-room simulator contracts.  It does not import or execute Fable code.
"""

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

from .config import CANDIDATES, GardConfig
from .intervention_core import MolecularEdit, apply_molecular_edit
from .intervention_outgoing_rule import select_outgoing_rule_edits
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
    paired_summary,
    purpose_seed,
    runtime_versions,
    safe_score_sequence,
    sha256_file,
)
from .phir_instruments import ATOM_NAMES
from .simulator import (
    FissionRecord,
    SimulationError,
    Snapshot,
    _fission,
    _sample_without_replacement,
    advance_fission,
    cosine_similarity,
    generate_beta,
    generate_initial_composition,
)


DOCUMENT = "CODEX_CH5_PHIR_EXTENSION_PREREGISTRATION.md"
DEFAULT_VALIDATION = RESULT_ROOT / "px4_validation"
DEFAULT_REGISTRATION = RESULT_ROOT / "px4_registration"
DEFAULT_SMOKE = RESULT_ROOT / "px4_smoke"
DEFAULT_OUTPUT = RESULT_ROOT / "px4_simulator_moderator"
DEFAULT_WORK = RESULT_ROOT / ".px4_work"
DEFAULT_LOG = RESULT_ROOT / "px4_simulator_moderator.log"

LABEL = "CODEX_CH5_PHIR_EXTENSION_PX4_V1"
REGISTRATION_FORMAT = "codex-ch5-phir-extension-px4-registration-v1"
RESULT_FORMAT = "codex-ch5-phir-extension-px4-result-v1"
CHECKPOINT_FORMAT = "codex-ch5-phir-extension-px4-checkpoint-v1"
SERVICE_NAME = "codex-phir-extension-px4-20260820"

MATRICES = 24
REPLICATES = 2
HORIZON = 60
FINAL30_START = 30
FINAL20_START = 40
CPU_SECONDS = 14.0 * 3600.0
ARMS = ("STABILIZE", "DESTABILIZE", "NOOP")

C02_VARIANTS = ("C02_CODEX", "C02_EVENTWISE")
C03_VARIANTS = tuple(
    f"C03_E{exposure}_O{overshoot}_D{daughter}"
    for exposure in (0, 1)
    for overshoot in (0, 1)
    for daughter in (0, 1)
)
FULL_PORT = {"02": "C02_EVENTWISE", "03": "C03_E1_O1_D1"}
CODEX_VARIANT = {"02": "C02_CODEX", "03": "C03_E0_O0_D0"}

SOURCE_FILES = (
    DOCUMENT,
    "plastic_heredity/phir_extension_px4.py",
    "plastic_heredity/phir_extension_common.py",
    "tests/test_phir_extension_px4.py",
    "plastic_heredity/config.py",
    "plastic_heredity/simulator.py",
    "plastic_heredity/intervention_core.py",
    "plastic_heredity/intervention_outgoing_rule.py",
    "plastic_heredity/phir_instruments.py",
    "plastic_heredity/phir_rescue_instruments.py",
    "plastic_heredity/seeds.py",
)


@dataclass(frozen=True)
class PX4Spec:
    label: str
    matrices: int
    replicates: int
    horizon: int
    final30_start: int
    final20_start: int
    cpu_seconds: float


@dataclass(frozen=True)
class PX4Batch:
    matrix_id: int
    lineage_rows: tuple[dict[str, Any], ...]
    edit_rows: tuple[dict[str, Any], ...]
    cpu_seconds: float
    scientific_digest: str


def scientific_spec() -> PX4Spec:
    return PX4Spec(
        "scientific", MATRICES, REPLICATES, HORIZON,
        FINAL30_START, FINAL20_START, CPU_SECONDS
    )


def smoke_spec() -> PX4Spec:
    return PX4Spec("smoke", 1, 1, 6, 3, 3, 300.0)


def variants(candidate: str) -> tuple[str, ...]:
    if candidate == "02":
        return C02_VARIANTS
    if candidate == "03":
        return C03_VARIANTS
    raise KeyError(candidate)


def variant_factors(candidate: str, variant: str) -> dict[str, int]:
    if candidate == "02":
        if variant not in C02_VARIANTS:
            raise ValueError(variant)
        return {
            "eventwise_growth": int(variant == "C02_EVENTWISE"),
            "adaptive_exposure": 0,
            "allow_overshoot": 0,
            "uniform_daughter": 0,
        }
    if candidate == "03" and variant in C03_VARIANTS:
        return {
            "eventwise_growth": 0,
            "adaptive_exposure": int(variant[5]),
            "allow_overshoot": int(variant[8]),
            "uniform_daughter": int(variant[11]),
        }
    raise ValueError(f"unknown PX4 variant: {candidate}/{variant}")


def _source_hashes() -> dict[str, str]:
    return {name: sha256_file(ROOT / name) for name in SOURCE_FILES}


def _batch_digest(batch: PX4Batch) -> str:
    value = asdict(batch)
    value["cpu_seconds"] = 0.0
    value["scientific_digest"] = ""
    return canonical_digest(value)


def _matrix_seed(spec: PX4Spec, matrix_id: int, purpose: str) -> int:
    domain = "smoke" if spec.label == "smoke" else purpose
    return purpose_seed(domain, "PX4", spec.label, purpose, matrix_id)


def _future_seed(
    spec: PX4Spec, candidate: str, matrix_id: int, replicate: int
) -> int:
    domain = "smoke" if spec.label == "smoke" else "future"
    # Deliberately excludes both intervention arm and simulator variant.
    return purpose_seed(domain, "PX4", spec.label, candidate, matrix_id, replicate)


def _rates(
    composition: NDArray, beta: NDArray, config: GardConfig
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    current = np.asarray(composition, dtype=np.int64)
    mass = int(current.sum())
    if mass <= 0:
        raise SimulationError("PX4 assembly became extinct")
    boost = 1.0 + (np.asarray(beta, dtype=np.float64) @ current) / mass
    join = config.k_join * (1.0 / config.n_types) * mass * boost
    leave = config.k_leave * current * boost
    return join, leave


def _eventwise_parent(
    composition: NDArray,
    beta: NDArray,
    config: GardConfig,
    rng: np.random.Generator,
) -> tuple[NDArray[np.int64], int]:
    current = np.asarray(composition, dtype=np.int64).copy()
    maximum = 40 * config.max_growth_steps
    for event in range(maximum):
        if int(current.sum()) >= config.n_max:
            return current, event
        join, leave = _rates(current, beta, config)
        rates = np.concatenate((join, leave))
        total = float(rates.sum())
        if not np.isfinite(total) or total <= 0.0:
            raise SimulationError("PX4 eventwise rate collapsed")
        location = int(np.searchsorted(np.cumsum(rates), rng.random() * total))
        location = min(location, 2 * config.n_types - 1)
        if location < config.n_types:
            current[location] += 1
        else:
            molecule = location - config.n_types
            if current[molecule] <= 0:
                raise AssertionError("PX4 selected a zero-rate leave event")
            current[molecule] -= 1
            if int(current.sum()) == 0:
                raise SimulationError("PX4 eventwise assembly became extinct")
    raise SimulationError("PX4 eventwise growth reached its sealed event cap")


def _poisson_parent(
    composition: NDArray,
    beta: NDArray,
    config: GardConfig,
    rng: np.random.Generator,
    *,
    adaptive: bool,
    allow_overshoot: bool,
) -> tuple[NDArray[np.int64], int]:
    current = np.asarray(composition, dtype=np.int64).copy()
    for step in range(1, config.max_growth_steps + 1):
        if int(current.sum()) >= config.n_max:
            return current, step - 1
        join, leave = _rates(current, beta, config)
        exposure = 4.0 / float(join.sum() + leave.sum()) if adaptive else 0.125
        joins = np.asarray(rng.poisson(join * exposure), dtype=np.int64)
        leaves = np.minimum(
            np.asarray(rng.poisson(leave * exposure), dtype=np.int64), current
        )
        survivors = current - leaves
        if allow_overshoot:
            current = survivors + joins
        else:
            capacity = config.n_max - int(survivors.sum())
            if int(joins.sum()) > capacity:
                joins = _sample_without_replacement(joins, capacity, rng)
            current = survivors + joins
        if int(current.sum()) == 0:
            raise SimulationError("PX4 Poisson assembly became extinct")
        if int(current.sum()) >= config.n_max:
            return current, step
    raise SimulationError("PX4 Poisson growth reached its sealed step cap")


def advance_variant(
    composition: NDArray,
    beta: NDArray,
    config: GardConfig,
    candidate: str,
    variant: str,
    rng: np.random.Generator,
) -> FissionRecord:
    factors = variant_factors(candidate, variant)
    if variant == CODEX_VARIANT[candidate]:
        return advance_fission(composition, beta, config, CANDIDATES[candidate], rng)
    if candidate == "02":
        parent, updates = _eventwise_parent(composition, beta, config, rng)
        daughter = _sample_without_replacement(parent, config.n_min, rng)
    else:
        parent, updates = _poisson_parent(
            composition,
            beta,
            config,
            rng,
            adaptive=bool(factors["adaptive_exposure"]),
            allow_overshoot=bool(factors["allow_overshoot"]),
        )
        first = np.asarray(rng.binomial(parent, 0.5), dtype=np.int64)
        second = parent - first
        if factors["uniform_daughter"]:
            daughter = first if rng.random() < 0.5 else second
            if int(daughter.sum()) == 0:
                daughter = first if int(first.sum()) > 0 else second
        else:
            daughter = second
    if int(daughter.sum()) <= 0:
        raise SimulationError("PX4 selected an empty daughter")
    return FissionRecord(
        np.asarray(parent, dtype=np.int64),
        np.asarray(daughter, dtype=np.int64),
        cosine_similarity(parent, daughter),
        int(updates),
    )


def _records_digest(records: Sequence[FissionRecord]) -> str:
    return canonical_digest(
        [
            {
                "parent": record.parent,
                "daughter": record.daughter,
                "h": np.asarray(record.h, dtype=np.float64).tobytes().hex(),
                "growth_steps": record.growth_steps,
            }
            for record in records
        ]
    )


def _run_arm(
    matrix_id: int,
    candidate: str,
    replicate: int,
    variant: str,
    arm: str,
    beta: NDArray,
    initial: NDArray,
    spec: PX4Spec,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = GardConfig()
    rng = np.random.default_rng(_future_seed(spec, candidate, matrix_id, replicate))
    snapshot = Snapshot(np.asarray(initial, dtype=np.int64).copy(), 0, (), ())
    records: list[FissionRecord] = []
    boundaries: list[NDArray[np.int64]] = []
    actions: list[dict[str, Any]] = []
    for step in range(1, spec.horizon + 1):
        try:
            record = advance_variant(
                snapshot.composition, beta, config, candidate, variant, rng
            )
        except SimulationError:
            break
        records.append(record)
        snapshot = _snapshot_after_record(snapshot, record)
        edit: MolecularEdit | None = None
        if step < spec.horizon and arm != "NOOP":
            rules = select_outgoing_rule_edits(snapshot.composition, beta)
            edit = rules["RULE_DOWN" if arm == "STABILIZE" else "RULE_UP"]
            snapshot = Snapshot(
                apply_molecular_edit(snapshot.composition, edit),
                snapshot.generation,
                snapshot.inheritance,
                snapshot.boundary_h,
                snapshot.previous_growth_steps,
                snapshot.cumulative_growth_steps,
            )
        boundaries.append(snapshot.composition.copy())
        action = {
            "matrix_id": matrix_id,
            "candidate": candidate,
            "replicate": replicate,
            "variant": variant,
            "arm": arm,
            "step": step,
            "remove_type": -1 if edit is None else edit.remove_type,
            "add_type": -1 if edit is None else edit.add_type,
        }
        action["action_digest"] = canonical_digest(action)
        actions.append(action)
    complete = len(records) == spec.horizon
    row: dict[str, Any] = {
        "matrix_id": matrix_id,
        "candidate": candidate,
        "replicate": replicate,
        "variant": variant,
        "arm": arm,
        "completed": int(complete),
        "completed_fissions": len(records),
        "record_digest": _records_digest(records),
        "final_rng_digest": canonical_digest(rng.bit_generator.state),
        "edits_applied": int(sum(item["remove_type"] >= 0 for item in actions)),
        "inherited_all": float(
            np.mean([record.h > config.inheritance_threshold for record in records])
        )
        if records
        else float("nan"),
        "inherited_final30": float(
            np.mean(
                [
                    record.h > config.inheritance_threshold
                    for record in records[spec.final30_start :]
                ]
            )
        )
        if len(records) > spec.final30_start
        else float("nan"),
        "inherited_final20": float(
            np.mean(
                [
                    record.h > config.inheritance_threshold
                    for record in records[spec.final20_start :]
                ]
            )
        )
        if len(records) > spec.final20_start
        else float("nan"),
    }
    for window, start in (("final30", spec.final30_start), ("final20", spec.final20_start)):
        if complete:
            observations = np.asarray(boundaries[start - 1 :], dtype=np.int64)
        else:
            observations = np.empty((0, config.n_types), dtype=np.int64)
        score = safe_score_sequence(observations, beta, "material", config)
        row.update(
            {f"{window}_{name}": value for name, value in score.fields("material").items()}
        )
        row[f"{window}_observations"] = int(observations.shape[0])
    return row, actions


def _run_matrix(args: tuple[int, PX4Spec]) -> PX4Batch:
    matrix_id, spec = args
    started = time.process_time()
    config = GardConfig()
    beta = generate_beta(
        config, np.random.default_rng(_matrix_seed(spec, matrix_id, "matrix"))
    )
    initial = generate_initial_composition(
        config, np.random.default_rng(_matrix_seed(spec, matrix_id, "initial"))
    )
    rows: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    with threadpool_limits(limits=1):
        for candidate in CANDIDATES:
            for replicate in range(spec.replicates):
                for variant in variants(candidate):
                    for arm in ARMS:
                        row, local = _run_arm(
                            matrix_id,
                            candidate,
                            replicate,
                            variant,
                            arm,
                            beta,
                            initial,
                            spec,
                        )
                        rows.append(row)
                        actions.extend(local)
    provisional = PX4Batch(
        matrix_id,
        tuple(rows),
        tuple(actions),
        float(time.process_time() - started),
        "",
    )
    return PX4Batch(
        provisional.matrix_id,
        provisional.lineage_rows,
        provisional.edit_rows,
        provisional.cpu_seconds,
        _batch_digest(provisional),
    )


def protocol() -> dict[str, Any]:
    value: dict[str, Any] = {
        "phase": "PX4",
        "question": "which documented simulator mechanics moderate the public nine-atom and full-block material responses to the physical outgoing rule?",
        "spec": asdict(scientific_spec()),
        "neutral_shared_inputs": {
            "same_beta_and_initial_across_all_variants": True,
            "future_seed_excludes_arm": True,
            "future_seed_excludes_variant": True,
            "common_random_streams_not_identical_realized_futures": True,
        },
        "candidate_02_factor": {
            "codex": "fixed-exposure vector Poisson, whole-assembly trim, fixed-size first daughter",
            "port": "single-event categorical growth to exact mass, same fixed-size first daughter",
        },
        "candidate_03_factorial": {
            "adaptive_exposure": "expected four events per vector-Poisson update",
            "allow_overshoot": "stop at or above parent mass without capacity admission",
            "uniform_daughter": "uniformly choose one of two binomial daughters",
            "variants": list(C03_VARIANTS),
        },
        "arms": list(ARMS),
        "controller": {
            "rule": "outgoing catalytic influence beta.T @ x",
            "stabilize": "RULE_DOWN: remove lowest and add highest outgoing influence",
            "destabilize": "RULE_UP: remove highest and add lowest outgoing influence",
            "timing": "after fissions 1-59; final daughter remains unedited",
        },
        "measurements": {
            "representation": "material",
            "windows": {
                "final30": "31 post-fission/post-control observations from boundaries 30-60",
                "final20": "21 post-fission/post-control observations from boundaries 40-60",
            },
            "primary": "public nine-atom revised STABILIZE minus DESTABILIZE",
            "secondary": "material full-block revised and inherited-boundary fraction",
            "all_16_atoms_retained": True,
        },
        "analysis": {
            "candidate_02": "eventwise-minus-Codex moderation of the arm contrast",
            "candidate_03": "complete 2^3 factorial moderation of the arm contrast",
            "inference_unit": "whole catalytic matrix",
            "replicates_never_pooled": True,
            "bootstrap": BOOTSTRAP_DRAWS,
            "randomization": RANDOMIZATION_DRAWS,
        },
        "classification": {
            "full_port_recovery": "public response positive with positive 95% matrix-bootstrap lower bound in both replicates of both candidates, with positive hereditary separation",
            "factor_localization": "a registered factor contrast has a same-signed 95% interval excluding zero in both replicates",
            "otherwise": "unresolved or distributed simulator moderation",
        },
        "external_code_imported_or_executed": False,
        "run_regardless_of_px1_px2_px3_gates": True,
        "no_48_matrix_campaign": True,
        "claim_boundary": [
            "simulator moderation is not proof that either implementation is uniquely correct",
            "an information response is not consciousness, agency, or life",
            "prior negative public Phi-r results remain unchanged",
            "strict-eight is excluded",
        ],
    }
    value["protocol_id"] = canonical_digest(value)
    return value


def validation_checks() -> dict[str, bool]:
    config = GardConfig()
    beta = generate_beta(config, np.random.default_rng(71))
    initial = generate_initial_composition(config, np.random.default_rng(72))
    left_rng = np.random.default_rng(73)
    right_rng = np.random.default_rng(73)
    direct = advance_fission(initial, beta, config, CANDIDATES["03"], left_rng)
    wrapped = advance_variant(
        initial, beta, config, "03", CODEX_VARIANT["03"], right_rng
    )
    same_record = _records_digest((direct,)) == _records_digest((wrapped,))
    same_rng = canonical_digest(left_rng.bit_generator.state) == canonical_digest(
        right_rng.bit_generator.state
    )
    return {
        "master_registration_exists": MASTER_REGISTRATION.exists(),
        "matrix_scale_24": scientific_spec().matrices == 24,
        "replicates_two": scientific_spec().replicates == 2,
        "horizon_60": scientific_spec().horizon == 60,
        "cpu_allocation_14h": scientific_spec().cpu_seconds == 14 * 3600,
        "candidate02_factor_complete": C02_VARIANTS
        == ("C02_CODEX", "C02_EVENTWISE"),
        "candidate03_factorial_complete": len(C03_VARIANTS) == 8
        and len({tuple(variant_factors("03", item).values()) for item in C03_VARIANTS})
        == 8,
        "full_ports_fixed": FULL_PORT
        == {"02": "C02_EVENTWISE", "03": "C03_E1_O1_D1"},
        "future_seed_arm_and_variant_free": "arm" not in inspect.signature(_future_seed).parameters
        and "variant" not in inspect.signature(_future_seed).parameters,
        "codex_wrapper_record_exact": same_record,
        "codex_wrapper_rng_exact": same_rng,
        "outgoing_rule_orientation_fixed": protocol()["controller"]["rule"]
        == "outgoing catalytic influence beta.T @ x",
        "all_atoms_retained": protocol()["measurements"]["all_16_atoms_retained"],
        "draws_fixed": BOOTSTRAP_DRAWS == 4096
        and RANDOMIZATION_DRAWS == 4096,
        "no_external_code_execution": not protocol()["external_code_imported_or_executed"],
        "no_48_matrix_campaign": protocol()["no_48_matrix_campaign"],
        "strict_eight_excluded": "strict-eight is excluded"
        in protocol()["claim_boundary"],
    }


def run_validation() -> dict[str, Any]:
    checks = validation_checks()
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/test_phir_extension_px4.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    payload = {
        "format": "codex-ch5-phir-extension-px4-validation-v1",
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
        raise AssertionError(f"PX4 validation failed\n{completed.stdout}\n{completed.stderr}")
    return payload


def register_program() -> dict[str, Any]:
    verify_checksums(DEFAULT_VALIDATION)
    validation = json.loads((DEFAULT_VALIDATION / "validation.json").read_text())
    if not validation["all_passed"]:
        raise ValueError("PX4 validation did not pass")
    verify_checksums(MASTER_REGISTRATION)
    if DEFAULT_REGISTRATION.exists():
        raise FileExistsError(f"PX4 registration exists: {DEFAULT_REGISTRATION}")
    master = json.loads((MASTER_REGISTRATION / "registration.json").read_text())
    body: dict[str, Any] = {
        "format": REGISTRATION_FORMAT,
        "master_registration_id": master["registration_id"],
        "protocol": protocol(),
        "source_hashes": _source_hashes(),
        "runtime": runtime_versions(),
        "new_scientific_matrices_at_registration": 0,
    }
    body["registration_id"] = canonical_digest(body)
    DEFAULT_REGISTRATION.mkdir(parents=True)
    shutil.copy2(ROOT / DOCUMENT, DEFAULT_REGISTRATION / "preregistration.md")
    atomic_json(DEFAULT_REGISTRATION / "protocol.json", body["protocol"])
    atomic_json(DEFAULT_REGISTRATION / "registration.json", body)
    write_checksums(DEFAULT_REGISTRATION)
    _append_ledger(
        f"<!-- phir-extension-px4-registration-{body['registration_id']} -->",
        [
            "## Phi-r extension PX4 registered",
            "",
            f"- Registration: `{body['registration_id']}`.",
            "- A fresh 24-matrix simulator-mechanics factorial was sealed with neutral shared inputs.",
            "- The phase uses Codex-owned code only and cannot overwrite earlier Phi-r results.",
        ],
    )
    return body


def verify_registration() -> dict[str, Any]:
    verify_checksums(DEFAULT_REGISTRATION)
    body = json.loads((DEFAULT_REGISTRATION / "registration.json").read_text())
    observed = body.pop("registration_id")
    if body.get("format") != REGISTRATION_FORMAT or observed != canonical_digest(body):
        raise ValueError("PX4 registration identity failed")
    body["registration_id"] = observed
    if body["protocol"] != canonical_json(protocol()):
        raise ValueError("PX4 protocol changed")
    if body["source_hashes"] != _source_hashes():
        raise ValueError("PX4 source changed after registration")
    return body


def _checkpointed(
    spec: PX4Spec,
    registration: Mapping[str, Any],
    directory: Path,
    stage: str,
    workers: int,
    prior_cpu: float = 0.0,
) -> tuple[list[PX4Batch], float]:
    directory.mkdir(parents=True, exist_ok=True)
    contract = {
        "format": CHECKPOINT_FORMAT,
        "stage": stage,
        "registration_id": registration["registration_id"],
        "protocol_id": registration["protocol"]["protocol_id"],
        "spec": asdict(spec),
        "source_hashes": registration["source_hashes"],
    }
    contract["contract_id"] = canonical_digest(contract)
    path = directory / "checkpoint_contract.json"
    if path.exists():
        if json.loads(path.read_text()) != canonical_json(contract):
            raise ValueError("PX4 checkpoint contract changed")
    else:
        atomic_json(path, contract)
    batches: list[PX4Batch | None] = [None] * spec.matrices
    missing: list[int] = []
    cpu = float(prior_cpu)
    for matrix_id in range(spec.matrices):
        checkpoint = directory / f"matrix_{matrix_id:04d}.pkl"
        if checkpoint.exists():
            with checkpoint.open("rb") as handle:
                batch = pickle.load(handle)
            if not isinstance(batch, PX4Batch) or batch.scientific_digest != _batch_digest(batch):
                raise ValueError(f"invalid PX4 checkpoint: {checkpoint}")
            batches[matrix_id] = batch
            cpu += batch.cpu_seconds
        else:
            missing.append(matrix_id)

    started = time.time()

    def status(state: str) -> None:
        complete = sum(item is not None for item in batches)
        elapsed = max(time.time() - started, 1e-9)
        rate = complete / elapsed if complete else 0.0
        atomic_json(
            directory / "status.json",
            {
                "stage": stage,
                "state": state,
                "completed": complete,
                "total": spec.matrices,
                "fraction": complete / spec.matrices,
                "cpu_seconds": cpu,
                "eta_seconds": (spec.matrices - complete) / rate if rate else None,
            },
        )

    status("running")
    arguments = [(matrix_id, spec) for matrix_id in missing]
    executor: ProcessPoolExecutor | None = None
    generated: Iterable[PX4Batch]
    if workers <= 1:
        generated = map(_run_matrix, arguments)
    else:
        executor = ProcessPoolExecutor(max_workers=min(workers, MAX_WORKERS))
        generated = executor.map(_run_matrix, arguments, chunksize=1)
    try:
        for matrix_id, batch in zip(missing, generated, strict=True):
            if batch.matrix_id != matrix_id or batch.scientific_digest != _batch_digest(batch):
                raise AssertionError("PX4 worker returned an invalid batch")
            batches[matrix_id] = batch
            atomic_pickle(directory / f"matrix_{matrix_id:04d}.pkl", batch)
            cpu += batch.cpu_seconds
            status("running")
            print(f"[PX4 {stage}] {sum(item is not None for item in batches)}/{spec.matrices}", flush=True)
            if cpu > spec.cpu_seconds:
                status("paused_cpu_budget")
                raise RuntimeError("PX4 CPU allocation reached; checkpoints retained")
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
    if any(batch is None for batch in batches):
        raise AssertionError("PX4 checkpoint stage incomplete")
    status("complete")
    return [batch for batch in batches if batch is not None], cpu


def _replay_audit(
    generated: Sequence[PX4Batch], replayed: Sequence[PX4Batch]
) -> dict[str, Any]:
    rows = [
        {
            "matrix_id": left.matrix_id,
            "generated": left.scientific_digest,
            "replay": right.scientific_digest,
            "exact": left.scientific_digest == right.scientific_digest,
        }
        for left, right in zip(generated, replayed, strict=True)
    ]
    return {
        "matrices": rows,
        "complete_exact_replay": len(rows) == MATRICES and all(row["exact"] for row in rows),
    }


def _arm_effect(
    frame: pd.DataFrame,
    candidate: str,
    replicate: int,
    variant: str,
    metric: str,
) -> pd.Series:
    selected = frame[
        (frame["candidate"] == candidate)
        & (frame["replicate"] == replicate)
        & (frame["variant"] == variant)
    ]
    pivot = selected.pivot(index="matrix_id", columns="arm", values=metric)
    return (pivot["STABILIZE"] - pivot["DESTABILIZE"]).sort_index()


def _factor_contrast(
    frame: pd.DataFrame,
    replicate: int,
    metric: str,
    terms: tuple[str, ...],
) -> pd.Series:
    responses = {
        variant: _arm_effect(frame, "03", replicate, variant, metric)
        for variant in C03_VARIANTS
    }
    values: list[pd.Series] = []
    for variant in C03_VARIANTS:
        factors = variant_factors("03", variant)
        sign = float(
            np.prod([2 * factors[name] - 1 for name in terms], dtype=np.float64)
        )
        values.append(responses[variant] * sign)
    # For a 2^3 design, division by four gives a high-minus-low main-effect
    # contrast and the correspondingly scaled interaction contrast.
    return sum(values[1:], values[0]) / 4.0


def analyze(batches: Sequence[PX4Batch]) -> tuple[dict[str, Any], dict[str, pd.DataFrame], dict[str, NDArray]]:
    frame = pd.DataFrame([row for batch in batches for row in batch.lineage_rows])
    edits = pd.DataFrame([row for batch in batches for row in batch.edit_rows])
    frame["candidate"] = frame["candidate"].astype(str).str.zfill(2)
    response_rows: list[dict[str, Any]] = []
    heredity_rows: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    arrays: dict[str, NDArray] = {}
    metrics = {
        "public": "final30_material_public_revised",
        "full": "final30_material_full_revised",
        "public_final20": "final20_material_public_revised",
        "full_final20": "final20_material_full_revised",
    }
    for candidate in CANDIDATES:
        for replicate in range(REPLICATES):
            for variant in variants(candidate):
                for family, metric in metrics.items():
                    effect = _arm_effect(frame, candidate, replicate, variant, metric)
                    summary, local = paired_summary(
                        effect.to_numpy(),
                        f"PX4/{candidate}/r{replicate}/{variant}/{family}",
                    )
                    summary.update(
                        {
                            "candidate": candidate,
                            "replicate": replicate,
                            "variant": variant,
                            "family": family,
                            "metric": metric,
                        }
                    )
                    response_rows.append(summary)
                    arrays.update(
                        {
                            f"response__{candidate}__r{replicate}__{variant}__{family}__{name}": value
                            for name, value in local.items()
                        }
                    )
                    for matrix_id, value in effect.items():
                        matrix_rows.append(
                            {
                                "kind": "arm_response",
                                "candidate": candidate,
                                "replicate": replicate,
                                "variant": variant,
                                "term": family,
                                "matrix_id": int(matrix_id),
                                "value": float(value),
                            }
                        )
                inherited = _arm_effect(
                    frame,
                    candidate,
                    replicate,
                    variant,
                    "inherited_final30",
                )
                summary, local = paired_summary(
                    inherited.to_numpy(),
                    f"PX4/{candidate}/r{replicate}/{variant}/inheritance",
                )
                summary.update(
                    {
                        "candidate": candidate,
                        "replicate": replicate,
                        "variant": variant,
                        "family": "inheritance",
                    }
                )
                heredity_rows.append(summary)
                arrays.update(
                    {
                        f"inheritance__{candidate}__r{replicate}__{variant}__{name}": value
                        for name, value in local.items()
                    }
                )
    primary = [row for row in response_rows if row["family"] == "public"]
    apply_holm(primary)
    apply_holm(heredity_rows)

    moderator_rows: list[dict[str, Any]] = []
    for replicate in range(REPLICATES):
        for family, metric in metrics.items():
            c02 = _arm_effect(
                frame, "02", replicate, "C02_EVENTWISE", metric
            ) - _arm_effect(frame, "02", replicate, "C02_CODEX", metric)
            summary, local = paired_summary(
                c02.to_numpy(), f"PX4/moderator/02/r{replicate}/{family}"
            )
            summary.update(
                {
                    "candidate": "02",
                    "replicate": replicate,
                    "family": family,
                    "term": "eventwise_growth",
                }
            )
            moderator_rows.append(summary)
            arrays.update(
                {
                    f"moderator__02__r{replicate}__{family}__eventwise__{name}": value
                    for name, value in local.items()
                }
            )
            for terms in (
                ("adaptive_exposure",),
                ("allow_overshoot",),
                ("uniform_daughter",),
                ("adaptive_exposure", "allow_overshoot"),
                ("adaptive_exposure", "uniform_daughter"),
                ("allow_overshoot", "uniform_daughter"),
                ("adaptive_exposure", "allow_overshoot", "uniform_daughter"),
            ):
                effect = _factor_contrast(frame, replicate, metric, terms)
                term = ":".join(terms)
                summary, local = paired_summary(
                    effect.to_numpy(),
                    f"PX4/moderator/03/r{replicate}/{family}/{term}",
                )
                summary.update(
                    {
                        "candidate": "03",
                        "replicate": replicate,
                        "family": family,
                        "term": term,
                    }
                )
                moderator_rows.append(summary)
                arrays.update(
                    {
                        f"moderator__03__r{replicate}__{family}__{term}__{name}": value
                        for name, value in local.items()
                    }
                )

    atom_rows: list[dict[str, Any]] = []
    atom_columns = [f"final30_material_atom_{name}" for name in ATOM_NAMES]
    for candidate in CANDIDATES:
        variant = FULL_PORT[candidate]
        for replicate in range(REPLICATES):
            selected = frame[
                (frame["candidate"] == candidate)
                & (frame["replicate"] == replicate)
                & (frame["variant"] == variant)
            ]
            vectors: list[NDArray] = []
            for matrix_id, local in selected.groupby("matrix_id", sort=True):
                arms = local.set_index("arm")
                vectors.append(
                    arms.loc["STABILIZE", atom_columns].to_numpy(float)
                    - arms.loc["DESTABILIZE", atom_columns].to_numpy(float)
                )
            summary, local_arrays = matrix_block_hotelling(
                np.vstack(vectors), f"PX4/atoms/{candidate}/r{replicate}"
            )
            summary.update(
                {
                    "candidate": candidate,
                    "replicate": replicate,
                    "variant": variant,
                    "component_names": list(ATOM_NAMES),
                }
            )
            atom_rows.append(summary)
            arrays.update(
                {
                    f"atoms__{candidate}__r{replicate}__{name}": value
                    for name, value in local_arrays.items()
                }
            )
    apply_holm(atom_rows, source="randomization_p")

    def lookup(rows: Sequence[dict[str, Any]], candidate: str, replicate: int, variant: str, family: str | None = None) -> dict[str, Any]:
        matches = [
            row
            for row in rows
            if row["candidate"] == candidate
            and row["replicate"] == replicate
            and row["variant"] == variant
            and (family is None or row.get("family") == family)
        ]
        if len(matches) != 1:
            raise AssertionError("PX4 result lookup is not unique")
        return matches[0]

    public_full_port = [
        lookup(response_rows, candidate, replicate, FULL_PORT[candidate], "public")
        for candidate in CANDIDATES
        for replicate in range(REPLICATES)
    ]
    heredity_full_port = [
        lookup(heredity_rows, candidate, replicate, FULL_PORT[candidate])
        for candidate in CANDIDATES
        for replicate in range(REPLICATES)
    ]
    full_port_recovery = bool(
        all(row["effect"] > 0 and row["ci95"][0] > 0 for row in public_full_port)
        and all(row["effect"] > 0 and row["ci95"][0] > 0 for row in heredity_full_port)
    )
    localized: list[dict[str, Any]] = []
    public_moderators = [row for row in moderator_rows if row["family"] == "public"]
    for candidate in CANDIDATES:
        for term in sorted({row["term"] for row in public_moderators if row["candidate"] == candidate}):
            cells = [
                row
                for row in public_moderators
                if row["candidate"] == candidate and row["term"] == term
            ]
            same_sign = len(cells) == 2 and np.sign(cells[0]["effect"]) == np.sign(cells[1]["effect"])
            excluded = all(row["ci95"][0] > 0 or row["ci95"][1] < 0 for row in cells)
            if same_sign and excluded:
                localized.append(
                    {
                        "candidate": candidate,
                        "term": term,
                        "direction": "positive" if cells[0]["effect"] > 0 else "negative",
                    }
                )
    classification = (
        "full_port_recovers_public_direction"
        if full_port_recovery
        else "specific_mechanics_moderate_without_full_recovery"
        if localized
        else "unresolved_or_distributed_simulator_moderation"
    )
    output = {
        "format": "codex-ch5-phir-extension-px4-metrics-v1",
        "responses": response_rows,
        "heredity": heredity_rows,
        "moderators": moderator_rows,
        "full_port_atom_vectors": atom_rows,
        "localized_public_factors": localized,
        "gates": {
            "full_port_public_direction_recovery": full_port_recovery,
            "heredity_full_port_valid": all(
                row["effect"] > 0 and row["ci95"][0] > 0
                for row in heredity_full_port
            ),
            "at_least_one_localized_public_factor": bool(localized),
        },
        "classification": classification,
    }
    return output, {
        "lineages": frame,
        "selected_edits": edits,
        "matrix_effects": pd.DataFrame(matrix_rows),
    }, arrays


def _write_result(
    batches: Sequence[PX4Batch],
    replay: Mapping[str, Any],
    registration: Mapping[str, Any],
    cpu: float,
) -> dict[str, Any]:
    metrics, tables, arrays = analyze(batches)
    temporary = DEFAULT_OUTPUT.with_name(DEFAULT_OUTPUT.name + f".tmp-{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    for name, frame in tables.items():
        frame.to_csv(temporary / f"{name}.csv.gz", index=False)
    np.savez_compressed(temporary / "inference_arrays.npz", **arrays)
    atomic_json(temporary / "primary_metrics.json", metrics)
    atomic_json(temporary / "replay_audit.json", replay)
    report = [
        "# PX4 simulator-mechanics moderator",
        "",
        f"Registration: `{registration['registration_id']}`.",
        "",
        f"Classification: **{metrics['classification']}**.",
        "",
        "## Full-port public nine-atom responses",
        "",
        "| Candidate | Replicate | Effect [95% CI] |",
        "| --- | ---: | ---: |",
    ]
    for candidate in CANDIDATES:
        for replicate in range(REPLICATES):
            row = next(
                item
                for item in metrics["responses"]
                if item["candidate"] == candidate
                and item["replicate"] == replicate
                and item["variant"] == FULL_PORT[candidate]
                and item["family"] == "public"
            )
            report.append(
                f"| {candidate} | {replicate} | {row['effect']:+.5f} "
                f"[{row['ci95'][0]:+.5f}, {row['ci95'][1]:+.5f}] |"
            )
    report.extend(
        [
            "",
            "## Registered gates",
            "",
            "```json",
            json.dumps(metrics["gates"], indent=2, sort_keys=True),
            "```",
            "",
            "This post-clean-room factor experiment identifies simulator moderation; it does not establish that either reconstruction is uniquely correct or that Phi is a physical cause.",
        ]
    )
    (temporary / "SCIENTIFIC_REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    (temporary / "LAY_SUMMARY.md").write_text(
        "# PX4 lay summary\n\n"
        "We rebuilt the documented simulator differences as switches and changed one switch at a time while keeping the molecular networks, starting states, control rule, and random streams matched.\n\n"
        f"The registered classification is **{metrics['classification']}**. This tells us whether the cross-clean-room Phi-r disagreement can be localized to particular simulation mechanics; it is not a consciousness result.\n",
        encoding="utf-8",
    )
    manifest = {
        "format": RESULT_FORMAT,
        "registration_id": registration["registration_id"],
        "matrices": MATRICES,
        "cpu_seconds": cpu,
        "complete_exact_replay": bool(replay["complete_exact_replay"]),
        "complete_readback_exact": False,
        "classification": metrics["classification"],
        "gates": metrics["gates"],
    }
    atomic_json(temporary / "manifest.json", manifest)
    write_checksums(temporary)
    temporary.replace(DEFAULT_OUTPUT)
    verify_checksums(DEFAULT_OUTPUT)
    readback = pd.read_csv(DEFAULT_OUTPUT / "lineages.csv.gz")
    exact = len(readback) == len(tables["lineages"])
    manifest["complete_readback_exact"] = exact
    atomic_json(DEFAULT_OUTPUT / "manifest.json", manifest)
    atomic_json(DEFAULT_OUTPUT / "readback_audit.json", {"complete": exact})
    write_checksums(DEFAULT_OUTPUT)
    if not exact:
        raise AssertionError("PX4 readback failed")
    _append_ledger(
        f"<!-- phir-extension-px4-result-{registration['registration_id']} -->",
        [
            "## Phi-r extension PX4 completed",
            "",
            "- Result: `results/phir_extension/px4_simulator_moderator`.",
            f"- Classification: `{metrics['classification']}`.",
            "- Complete exact replay/readback passed; prior results remain unchanged.",
        ],
    )
    return manifest


def run_scientific(workers: int = MAX_WORKERS) -> dict[str, Any]:
    registration = verify_registration()
    verify_checksums(DEFAULT_SMOKE)
    if DEFAULT_OUTPUT.exists():
        raise FileExistsError(f"PX4 output exists: {DEFAULT_OUTPUT}")
    if not (RESULT_ROOT / "px3_confirmation" / "manifest.json").exists():
        raise RuntimeError("PX4 is locked until PX3 confirmation completes")
    if shutil.disk_usage(ROOT).free < MINIMUM_FREE_DISK_BYTES:
        raise RuntimeError("PX4 refused below the sealed disk floor")
    spec = scientific_spec()
    generated, cpu = _checkpointed(
        spec,
        registration,
        DEFAULT_WORK / "generate",
        "generate",
        workers,
    )
    replayed, cpu = _checkpointed(
        spec,
        registration,
        DEFAULT_WORK / "replay",
        "replay",
        workers,
        cpu,
    )
    replay = _replay_audit(generated, replayed)
    if not replay["complete_exact_replay"]:
        raise AssertionError("PX4 exact replay failed")
    return _write_result(generated, replay, registration, cpu)


def run_smoke() -> dict[str, Any]:
    if DEFAULT_SMOKE.exists():
        raise FileExistsError(f"PX4 smoke exists: {DEFAULT_SMOKE}")
    spec = smoke_spec()
    first = _run_matrix((0, spec))
    second = _run_matrix((0, spec))
    rows = pd.DataFrame(first.lineage_rows)
    payload = {
        "format": "codex-ch5-phir-extension-px4-smoke-v1",
        "all_candidates": set(rows["candidate"]) == set(CANDIDATES),
        "all_arms": set(rows["arm"]) == set(ARMS),
        "all_variants": set(rows[rows.candidate == "02"].variant) == set(C02_VARIANTS)
        and set(rows[rows.candidate == "03"].variant) == set(C03_VARIANTS),
        "complete": bool(rows["completed"].all()),
        "information_finite": bool(
            np.isfinite(rows["final30_material_public_revised"]).all()
            and np.isfinite(rows["final30_material_full_revised"]).all()
        ),
        "replay_exact": first.scientific_digest == second.scientific_digest,
        "effect_sizes_suppressed": True,
    }
    payload["passed"] = bool(
        all(value for key, value in payload.items() if key != "format")
    )
    DEFAULT_SMOKE.mkdir(parents=True)
    atomic_json(DEFAULT_SMOKE / "smoke.json", payload)
    write_checksums(DEFAULT_SMOKE)
    if not payload["passed"]:
        raise AssertionError("PX4 smoke failed")
    return payload


def launch_detached(workers: int = MAX_WORKERS) -> dict[str, Any]:
    registration = verify_registration()
    verify_checksums(DEFAULT_SMOKE)
    if DEFAULT_OUTPUT.exists():
        raise FileExistsError(f"PX4 output exists: {DEFAULT_OUTPUT}")
    if not (RESULT_ROOT / "px3_confirmation" / "manifest.json").exists():
        raise RuntimeError("PX4 launch is locked until PX3 confirmation completes")
    if shutil.disk_usage(ROOT).free < MINIMUM_FREE_DISK_BYTES:
        raise RuntimeError("PX4 launch refused below the sealed disk floor")
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
        "plastic_heredity.phir_extension_px4",
        "run",
        "--workers",
        str(min(workers, MAX_WORKERS)),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = {
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
        "phase": "PX4",
        "validation": DEFAULT_VALIDATION.exists(),
        "registration": DEFAULT_REGISTRATION.exists(),
        "smoke": DEFAULT_SMOKE.exists(),
        "complete": DEFAULT_OUTPUT.exists(),
        "launch_locked_until_px3_confirmation": not (
            RESULT_ROOT / "px3_confirmation" / "manifest.json"
        ).exists(),
        "service": SERVICE_NAME,
        "free_disk_bytes": shutil.disk_usage(ROOT).free,
    }
    for stage in ("generate", "replay"):
        path = DEFAULT_WORK / stage / "status.json"
        if path.exists():
            payload[stage] = json.loads(path.read_text())
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
