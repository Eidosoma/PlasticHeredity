"""Preparation, sealing, execution, reporting, status, and verification."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import io
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

from .cohorts import allocate, donor_ids_from_pairs
from .contract import (
    CONDITIONS,
    CONTRACT,
    DEFAULT_ARTIFACTS,
    FIXED_CONFIGURATION,
    NAMESPACE,
    PAIRING_NAMESPACE,
    PACKAGE_ROOT,
    PROFILE,
    SCHEMA_VERSION,
    atomic_write_json,
    atomic_write_text,
    implementation_manifest,
    load_json,
    read_checkpoint,
    seal_registration,
    sha256_bytes,
    sha256_file,
    sha256_json,
    verify_registration,
    write_checkpoint,
)
from .engine import (
    decode_state_hex,
    encode_state_hex,
    motif_addresses_batch,
    motif_counts,
    motif_counts_batch,
    parent_statistics,
    pooled_reference,
    read_motif_energy_batch,
    simulate_pair_lineages,
    step_rule31649,
    step_rule31649_batch,
    texture2x2_counts_batch,
    write_carriers_batch,
)
from .inference import adjudicate
from .snapshot import prepare_snapshot, verify_snapshot


SOURCE_CANDIDATE = "simple--strict-49-64--gain-050"


def _artifact_root(value: Path | None) -> Path:
    return (value or DEFAULT_ARTIFACTS).resolve()


def _context(artifacts: Path) -> dict[str, Any]:
    input_root = artifacts / "input"
    manifest = verify_snapshot(input_root)
    local = input_root / "local"
    context = input_root / "context"
    donors = load_json(local / "DONORS.json")["donors"]
    return {
        "manifest": manifest,
        "input_root": input_root,
        "local_root": local,
        "context_root": context,
        "donors": donors,
        "donor_index": {str(donor["donor_id"]): donor for donor in donors},
        "hypothesis": load_json(local / "HYPOTHESIS.json"),
        "launch_resets": load_json(local / "LAUNCH_RESETS.json"),
        "stage1_calibration": load_json(local / "STAGE1_CALIBRATION.json"),
        "stage1_registration": load_json(local / "STAGE1_REGISTRATION.json"),
        "stage1_decision": load_json(local / "STAGE1_DECISION.json"),
        "stage2_registration": load_json(local / "STAGE2_REGISTRATION.json"),
        "stage2_decision": load_json(local / "STAGE2_DECISION.json"),
        "source_cohorts": load_json(context / "STAGE3R_COHORTS.json"),
        "source_design": load_json(context / "STAGE3R_DESIGN.json"),
        "source_results": load_json(context / "STAGE3R_RESULTS.json"),
        "source_decision": load_json(context / "STAGE3R_STAGE_DECISION.json"),
        "v1_registration": load_json(input_root / "v1/REGISTRATION.json"),
        "v1_results": load_json(input_root / "v1/RESULTS.json"),
        "forensic": load_json(input_root / "forensic/FORENSIC_RECORD.json"),
    }


def _verify_external_registration(registration: Mapping[str, Any], name: str) -> None:
    digest = registration.get("design_digest")
    body = {key: value for key, value in registration.items() if key != "design_digest"}
    if digest != sha256_json(body):
        raise ValueError(f"{name} registration digest mismatch")


def _cohorts(context: Mapping[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    return allocate(
        context["donors"],
        context["stage1_registration"],
        context["stage2_registration"],
        context["source_cohorts"],
        context["v1_registration"],
    )


def _calibration_pairs(registration: Mapping[str, Any]) -> list[dict[str, Any]]:
    value = registration["cohorts"]["calibration"]
    if len(value) != 64:
        raise ValueError("frozen Stage-1 calibration cohort must contain 64 pairs")
    return [dict(pair) for pair in value]


def _recompute_reference(context: Mapping[str, Any]) -> np.ndarray:
    histories: list[np.ndarray] = []
    for pair in _calibration_pairs(context["stage1_registration"]):
        for key in ("a_donor_id", "b_donor_id"):
            donor = context["donor_index"][str(pair[key])]
            histories.append(
                parent_statistics(decode_state_hex(donor["donor_state_hex"]), 32)
            )
    return pooled_reference(histories, alpha=float(CONTRACT["jeffreys_alpha"]))


def _frozen_reference(context: Mapping[str, Any]) -> np.ndarray:
    calibration = context["stage1_calibration"]
    if calibration.get("binding_sha256") != context["stage1_registration"]["design_digest"]:
        raise ValueError("frozen Stage-1 calibration binding mismatch")
    payload = calibration.get("payload")
    if calibration.get("payload_sha256") != sha256_json(payload):
        raise ValueError("frozen Stage-1 calibration checksum mismatch")
    return np.asarray(payload["reference"]["32"]["motif_probability"], dtype=np.float64)


def _reference_document(context: Mapping[str, Any]) -> dict[str, Any]:
    recalculated = _recompute_reference(context)
    frozen = _frozen_reference(context)
    error = float(np.max(np.abs(recalculated - frozen)))
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": "label_blind_v2_reference_recalibration",
        "write_window": 32,
        "calibration_pairs": 64,
        "calibration_histories": 128,
        "state_and_motif_convention": "row-major least-significant-bit first",
        "motif_probability": recalculated.tolist(),
        "frozen_local_stage1_max_abs_error": error,
        "frozen_local_stage1_exact_tolerance": 1e-15,
        "labels_accessed": False,
        "source_outcomes_accessed": False,
    }


def _targets(context: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    targets = context["hypothesis"]["targets"]
    primary = {
        label: np.asarray(targets["primary"][label], dtype=np.float64)
        for label in ("A", "B")
    }
    terminal = {
        label: np.asarray(targets["primary_terminal"][label], dtype=np.float64)
        for label in ("A", "B")
    }
    return primary, terminal


def _reset_for_pair(
    pair: Mapping[str, Any], context: Mapping[str, Any]
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    launch = int(pair["launch_index"])
    reset = str(context["launch_resets"][f"launch{launch}"])
    donor_a = context["donor_index"][str(pair["a_donor_id"])]
    donor_b = context["donor_index"][str(pair["b_donor_id"])]
    if int(donor_a["launch_index"]) != launch or int(donor_b["launch_index"]) != launch:
        raise ValueError("pair launch does not match its donors")
    if donor_a["prototype_label"] != "A" or donor_b["prototype_label"] != "B":
        raise ValueError("pair label order must be A then B")
    if donor_a["initial_state_hex"] != reset or donor_b["initial_state_hex"] != reset:
        raise ValueError("pair donors do not share the registered launch reset")
    return reset, donor_a, donor_b


def parity_check(artifacts_root: Path | None = None) -> dict[str, Any]:
    artifacts = _artifact_root(artifacts_root)
    context = _context(artifacts)
    rng = np.random.default_rng(20260823)
    boards = rng.integers(0, 2, size=(3, 16, 16), dtype=np.uint8)
    scalar_steps = np.stack([step_rule31649(board) for board in boards])
    batch_steps = step_rule31649_batch(boards)
    scalar_motifs = np.stack([motif_counts(board) for board in boards])
    batch_motifs = motif_counts_batch(boards)
    reference = _recompute_reference(context)
    batch_carriers = write_carriers_batch(batch_motifs, reference)
    uniforms = rng.random(boards.shape)
    scalar_reads = np.stack(
        [
            read_motif_energy_batch(board, carrier, 0.25, uniform)
            for board, carrier, uniform in zip(
                batch_steps, batch_carriers, uniforms, strict=True
            )
        ]
    )
    batch_reads = read_motif_energy_batch(
        batch_steps, batch_carriers, 0.25, uniforms
    )
    roundtrip_failures = 0
    for donor in context["donors"]:
        for key in ("initial_state_hex", "donor_state_hex"):
            value = donor[key]
            roundtrip_failures += int(encode_state_hex(decode_state_hex(value)) != value)
    reference_error = float(
        np.max(np.abs(reference - _frozen_reference(context)))
    )
    checks = {
        "state_hex_roundtrip_all_donors": roundtrip_failures == 0,
        "scalar_batch_rule_exact": bool(np.array_equal(scalar_steps, batch_steps)),
        "scalar_batch_motif_counts_exact": bool(
            np.array_equal(scalar_motifs, batch_motifs)
        ),
        "scalar_batch_reader_exact": bool(np.array_equal(scalar_reads, batch_reads)),
        "calibration_reference_within_1e-15": reference_error <= 1e-15,
        "reader_transfer_function_frozen": (
            FIXED_CONFIGURATION["transfer_function"]
            == "strength*tanh(max(energy_advantage,0)/9)"
        ),
        "sweep_order_frozen": CONTRACT["sweep_order"]
        == ["CA step", "reader", "process noise", "write/observe"],
    }
    if not all(checks.values()):
        raise AssertionError(f"v2 parity failed: {checks}")
    report = {
        "schema_version": SCHEMA_VERSION,
        "valid": True,
        "checks": checks,
        "reference_max_abs_error": reference_error,
        "source_trajectory_parity": "not performed; source outcomes are non-evidential",
        "fresh_outcomes_generated": False,
    }
    atomic_write_json(artifacts / "PARITY.json", report)
    return report


def validate_cleanroom(artifacts_root: Path | None = None) -> dict[str, Any]:
    artifacts = _artifact_root(artifacts_root)
    context = _context(artifacts)
    for name in ("SOURCE_SPEC.md", "FORENSIC_AUDIT.md"):
        if sha256_file(context["input_root"] / "forensic" / name) != sha256_file(
            PACKAGE_ROOT / name
        ):
            raise ValueError(f"frozen forensic document differs from package {name}")
    for name in ("stage1_registration", "stage2_registration", "v1_registration"):
        _verify_external_registration(context[name], name)
    cohorts, audit = _cohorts(context)
    if len(cohorts["confirmation"]) != int(PROFILE["confirmation_pairs"]):
        raise AssertionError(
            f"expected exactly 92 untouched pairs; found {len(cohorts['confirmation'])}"
        )
    if audit["untouched_donors"] != 431:
        raise AssertionError(f"expected 431 untouched donors; found {audit['untouched_donors']}")
    expected_launches = {"launch0": 20, "launch1": 30, "launch2": 21, "launch3": 21}
    if audit["confirmation_pairs_by_launch"] != expected_launches:
        raise AssertionError("unexpected same-launch matching cardinalities")
    if audit["maximum_density_difference"] > float(CONTRACT["density_caliper"]):
        raise AssertionError("confirmation density caliper violated")
    confirmation_donors = donor_ids_from_pairs(cohorts["confirmation"])
    quarantine_donors = donor_ids_from_pairs(cohorts["quarantine"])
    if confirmation_donors & quarantine_donors:
        raise AssertionError("confirmation donor leaked into exposed engineering cohort")
    reset_audit: dict[str, Any] = {}
    expected_live = {0: 5, 1: 3, 2: 6, 3: 4}
    for launch in range(4):
        reset_hex = context["launch_resets"][f"launch{launch}"]
        reset = decode_state_hex(reset_hex)
        if int(reset.sum()) != expected_live[launch]:
            raise AssertionError("launch reset live-cell count changed")
        reset_audit[f"launch{launch}"] = {
            "state_hex": reset_hex,
            "array_sha256": sha256_bytes(reset.tobytes(order="C")),
            "live_cells": int(reset.sum()),
        }
    for cohort in cohorts.values():
        for pair in cohort:
            _reset_for_pair(pair, context)
    reference_document = _reference_document(context)
    if reference_document["frozen_local_stage1_max_abs_error"] > 1e-15:
        raise AssertionError("corrected reference failed the frozen calibration reconstruction")
    atomic_write_json(artifacts / "REFERENCE.json", reference_document)
    source_contract = context["source_design"]["contract"]
    contract_checks = {
        "rule": source_contract.get("rule") == CONTRACT["rule"],
        "generation_sweeps": source_contract.get("generation_sweeps")
        == CONTRACT["generation_sweeps"],
        "read_sweeps": source_contract.get("read_sweeps") == CONTRACT["read_sweeps"],
        "observe_start": source_contract.get("observe_start")
        == CONTRACT["observation_window"][0],
        "process_noise": source_contract.get("process_noise")
        == CONTRACT["process_noise"],
        "stale_retention": source_contract.get("stale_retention")
        == CONTRACT["stale_retention"],
        "decoder_splits": source_contract.get("decoder_splits")
        == CONTRACT["decoder_splits"],
    }
    if not all(contract_checks.values()):
        raise ValueError("corrected v2 contract diverges from frozen protocol data")
    if context["stage1_decision"].get("verdict") != "ROBUST_LOCAL_MOTIF_CONTROLLABILITY":
        raise ValueError("local Stage-1 prerequisite did not pass")
    if context["stage2_decision"].get("verdict") != "DENSITY_ROBUST_GENERAL_MOTIF_CHANNEL":
        raise ValueError("local Stage-2 prerequisite did not pass")
    if context["source_decision"].get("verdict") != "STRICT_RENEWED_CA_PLASTIC_HEREDITY":
        raise ValueError("frozen positive hypothesis binding changed")
    source_candidate = context["source_results"]["adjudication"]["candidates"].get(
        SOURCE_CANDIDATE
    )
    if source_candidate is None or source_candidate["model"].get("gain") != 0.5:
        raise ValueError("frozen repair candidate binding changed")
    v1_quarantine = {
        "schema_version": SCHEMA_VERSION,
        "classification": context["forensic"]["v1_disposition"],
        "v1_design_digest": context["v1_registration"]["design_digest"],
        "v1_verdict": context["v1_results"]["adjudication"]["verdict"],
        "v1_files_modified": False,
        "use_for_v2_intended_mechanism_verdict": False,
        "reasons": context["forensic"]["comparability_breakers"],
    }
    atomic_write_json(artifacts / "V1_QUARANTINE.json", v1_quarantine)
    report = {
        "schema_version": SCHEMA_VERSION,
        "valid": True,
        "snapshot_digest": context["manifest"]["snapshot_digest"],
        "one_time_source_specification_frozen": True,
        "further_source_code_access": False,
        "source_and_v1_outcomes_used_as_evidence": False,
        "contract_checks": contract_checks,
        "cohort_audit": audit,
        "cohort_counts": {key: len(value) for key, value in cohorts.items()},
        "confirmation_donors_unique": len(confirmation_donors) == 184,
        "confirmation_donor_overlap_with_any_exposure": 0,
        "reset_audit": reset_audit,
        "reference_recalibration": {
            "sha256": sha256_file(artifacts / "REFERENCE.json"),
            "max_abs_error": reference_document[
                "frozen_local_stage1_max_abs_error"
            ],
        },
        "v1_disposition": v1_quarantine["classification"],
        "additional_scientific_experiments": [],
    }
    atomic_write_json(artifacts / "VALIDATION.json", report)
    atomic_write_json(
        artifacts / "COHORTS.json",
        {
            "schema_version": SCHEMA_VERSION,
            "snapshot_digest": context["manifest"]["snapshot_digest"],
            "allocation_namespace": PAIRING_NAMESPACE,
            "audit": audit,
            "cohorts": cohorts,
        },
    )
    parity_check(artifacts)
    return report


def audit_tests(artifacts_root: Path | None = None) -> dict[str, Any]:
    artifacts = _artifact_root(artifacts_root)
    verify_snapshot(artifacts / "input")
    started = time.monotonic()
    command = [sys.executable, "-m", "pytest", str(PACKAGE_ROOT / "tests"), "-q"]
    result = subprocess.run(
        command,
        cwd=PACKAGE_ROOT.parent,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    audit = {
        "schema_version": SCHEMA_VERSION,
        "passed": result.returncode == 0,
        "returncode": result.returncode,
        "command": command,
        "elapsed_seconds": time.monotonic() - started,
        "output": result.stdout[-20_000:],
        "implementation_manifest": implementation_manifest(),
    }
    atomic_write_json(artifacts / "TEST_AUDIT.json", audit)
    if not audit["passed"]:
        raise RuntimeError("v2 test audit failed")
    return audit


def register(artifacts_root: Path | None = None) -> dict[str, Any]:
    artifacts = _artifact_root(artifacts_root)
    path = artifacts / "REGISTRATION.json"
    if path.exists():
        return _load_registration(artifacts)
    for phase in ("quarantine", "confirmation"):
        checkpoint_root = artifacts / phase / "checkpoints"
        if checkpoint_root.exists() and any(checkpoint_root.glob("*.json")):
            raise RuntimeError("unregistered lineage outcomes already exist")
    validation = load_json(artifacts / "VALIDATION.json")
    parity = load_json(artifacts / "PARITY.json")
    test_audit = load_json(artifacts / "TEST_AUDIT.json")
    if not validation.get("valid") or not parity.get("valid") or not test_audit.get("passed"):
        raise ValueError("validation, parity, and the complete test audit must pass")
    current_manifest = implementation_manifest()
    if test_audit["implementation_manifest"] != current_manifest:
        raise ValueError("implementation changed after the passing test audit")
    context = _context(artifacts)
    cohorts, cohort_audit = _cohorts(context)
    reference = load_json(artifacts / "REFERENCE.json")
    registration = seal_registration(
        {
            "schema_version": SCHEMA_VERSION,
            "experiment": "corrected_independent_ca_lineage_renewal_stage3r_replication",
            "registration_state": "sealed_before_any_v2_lineage_outcome",
            "frozen_date_utc": "2026-08-23",
            "namespace": NAMESPACE,
            "snapshot_digest": context["manifest"]["snapshot_digest"],
            "snapshot_manifest_sha256": sha256_file(
                context["input_root"] / "MANIFEST.json"
            ),
            "validation_sha256": sha256_file(artifacts / "VALIDATION.json"),
            "parity_sha256": sha256_file(artifacts / "PARITY.json"),
            "test_audit_sha256": sha256_file(artifacts / "TEST_AUDIT.json"),
            "reference_sha256": sha256_file(artifacts / "REFERENCE.json"),
            "implementation_manifest": current_manifest,
            "configuration": FIXED_CONFIGURATION,
            "repair": {
                "source_candidate": SOURCE_CANDIDATE,
                "mechanism_class": "simple",
                "kind": "gain-050",
                "gain": 0.5,
                "daughter_write_window": [49, 64],
            },
            "contract": CONTRACT,
            "conditions": CONDITIONS,
            "profile": PROFILE,
            "cohorts": cohorts,
            "cohort_audit": cohort_audit,
            "reference": {
                "phase": reference["phase"],
                "calibration_pairs": reference["calibration_pairs"],
                "sha256": sha256_file(artifacts / "REFERENCE.json"),
            },
            "source_bindings": {
                name: context["manifest"]["files"][name]
                for name in (
                    "context/CA_MOTIF_LINEAGE_STAGE3R_PROTOCOL.md",
                    "context/STAGE3R_COHORTS.json",
                    "context/STAGE3R_DESIGN.json",
                    "context/STAGE3R_RESULTS.json",
                    "context/STAGE3R_STAGE_DECISION.json",
                    "forensic/SOURCE_SPEC.md",
                )
            },
            "source_evidential_role": "none; hypothesis and recovered specification only",
            "v1_disposition": "NON_COMPARABLE_MODEL_RUN",
            "v1_design_digest": context["v1_registration"]["design_digest"],
            "independent_unit": "matched founder pair",
            "quarantine_role": "already-exposed engineering pairs; never enters inference",
            "automatic_confirmation_launch": False,
            "additional_scientific_experiments": [],
            "fresh_outcome_files_present_at_seal": [],
        }
    )
    atomic_write_json(path, registration)
    _write_status(artifacts)
    return registration


def _load_registration(artifacts: Path) -> dict[str, Any]:
    registration = load_json(artifacts / "REGISTRATION.json")
    verify_registration(registration)
    if registration["implementation_manifest"] != implementation_manifest():
        raise ValueError("implementation changed after v2 registration")
    context = _context(artifacts)
    if registration["snapshot_digest"] != context["manifest"]["snapshot_digest"]:
        raise ValueError("input snapshot changed after v2 registration")
    for key, name in (
        ("validation_sha256", "VALIDATION.json"),
        ("parity_sha256", "PARITY.json"),
        ("test_audit_sha256", "TEST_AUDIT.json"),
        ("reference_sha256", "REFERENCE.json"),
    ):
        if registration[key] != sha256_file(artifacts / name):
            raise ValueError(f"sealed {name} changed after registration")
    return registration


def _reference(artifacts: Path) -> np.ndarray:
    value = load_json(artifacts / "REFERENCE.json")["motif_probability"]
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (512,) or not np.isfinite(result).all():
        raise ValueError("invalid registered reference")
    return result


def _worker(argument: Mapping[str, Any]) -> dict[str, Any]:
    payload = simulate_pair_lineages(
        pair_id=str(argument["pair"]["pair_id"]),
        donor_state_hex=argument["donor_state_hex"],
        donor_initial_state_hex=argument["donor_initial_state_hex"],
        reset_state_hex=str(argument["reset_state_hex"]),
        reference_probability=np.asarray(argument["reference_probability"], dtype=np.float64),
        targets_primary={
            key: np.asarray(value, dtype=np.float64)
            for key, value in argument["targets_primary"].items()
        },
        targets_terminal={
            key: np.asarray(value, dtype=np.float64)
            for key, value in argument["targets_terminal"].items()
        },
        replicates=int(argument["replicates"]),
        generations=int(argument["generations"]),
        conditions=argument["conditions"],
    )
    return {"phase": argument["phase"], "pair": dict(argument["pair"]), **payload}


def _arguments(
    artifacts: Path,
    registration: Mapping[str, Any],
    cohort_name: str,
    *,
    replicates: int,
    generations: int,
) -> list[dict[str, Any]]:
    context = _context(artifacts)
    primary, terminal = _targets(context)
    reference = _reference(artifacts)
    arguments: list[dict[str, Any]] = []
    for pair in registration["cohorts"][cohort_name]:
        reset, donor_a, donor_b = _reset_for_pair(pair, context)
        arguments.append(
            {
                "phase": cohort_name,
                "pair": pair,
                "donor_state_hex": [
                    donor_a["donor_state_hex"],
                    donor_b["donor_state_hex"],
                ],
                "donor_initial_state_hex": [
                    donor_a["initial_state_hex"],
                    donor_b["initial_state_hex"],
                ],
                "reset_state_hex": reset,
                "reference_probability": reference.tolist(),
                "targets_primary": {
                    key: value.tolist() for key, value in primary.items()
                },
                "targets_terminal": {
                    key: value.tolist() for key, value in terminal.items()
                },
                "replicates": replicates,
                "generations": generations,
                "conditions": registration["conditions"],
            }
        )
    return arguments


def _checkpoint_payloads(
    directory: Path,
    cohort: Sequence[Mapping[str, Any]],
    binding: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    payloads: list[dict[str, Any]] = []
    missing: list[str] = []
    for pair in cohort:
        path = directory / f"{pair['pair_id']}.json"
        if not path.exists():
            missing.append(str(pair["pair_id"]))
            continue
        payload = read_checkpoint(path, binding)
        if payload.get("pair_id") != pair["pair_id"] or payload.get("pair") != pair:
            raise ValueError(f"checkpoint pair identity mismatch in {path}")
        payloads.append(payload)
    return payloads, missing


def _run_checkpoint_tasks(
    arguments: Sequence[Mapping[str, Any]],
    checkpoint_dir: Path,
    binding: str,
    workers: int,
    artifacts: Path,
) -> int:
    if workers < 1 or workers > 20:
        raise ValueError("workers must be between 1 and 20")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    progress_path = checkpoint_dir.parent / "PROGRESS.json"
    pending = [
        argument
        for argument in arguments
        if not (checkpoint_dir / f"{argument['pair']['pair_id']}.json").exists()
    ]
    for argument in arguments:
        path = checkpoint_dir / f"{argument['pair']['pair_id']}.json"
        if path.exists():
            payload = read_checkpoint(path, binding)
            if payload.get("pair_id") != argument["pair"]["pair_id"]:
                raise ValueError(f"checkpoint pair identity mismatch in {path}")
    if not pending:
        atomic_write_json(
            progress_path,
            {
                "state": "complete",
                "completed": len(arguments),
                "total": len(arguments),
                "eta_seconds": 0.0,
                "last_update_unix": time.time(),
            },
        )
        return 0
    already_complete = len(arguments) - len(pending)
    started_monotonic = time.monotonic()
    started_unix = time.time()
    atomic_write_json(
        progress_path,
        {
            "state": "running",
            "completed": already_complete,
            "total": len(arguments),
            "eta_seconds": None,
            "session_started_unix": started_unix,
            "last_update_unix": started_unix,
            "resume_safe": True,
        },
    )

    def record_progress(session_completed: int) -> None:
        elapsed = max(time.monotonic() - started_monotonic, 1e-9)
        rate = session_completed / elapsed
        complete_count = already_complete + session_completed
        remaining = len(arguments) - complete_count
        atomic_write_json(
            progress_path,
            {
                "state": "complete" if remaining == 0 else "running",
                "completed": complete_count,
                "total": len(arguments),
                "pairs_per_second": rate,
                "eta_seconds": remaining / rate if rate else None,
                "session_started_unix": started_unix,
                "last_update_unix": time.time(),
                "resume_safe": True,
            },
        )
        _write_status(artifacts)

    completed = 0
    try:
        if workers == 1:
            for argument in pending:
                payload = _worker(argument)
                write_checkpoint(
                    checkpoint_dir / f"{argument['pair']['pair_id']}.json",
                    binding,
                    payload,
                )
                completed += 1
                record_progress(completed)
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(_worker, argument): argument for argument in pending
                }
                for future in as_completed(futures):
                    argument = futures[future]
                    payload = future.result()
                    write_checkpoint(
                        checkpoint_dir / f"{argument['pair']['pair_id']}.json",
                        binding,
                        payload,
                    )
                    completed += 1
                    record_progress(completed)
    except BaseException as error:
        atomic_write_json(
            progress_path,
            {
                "state": "interrupted",
                "completed": already_complete + completed,
                "total": len(arguments),
                "eta_seconds": None,
                "session_started_unix": started_unix,
                "last_update_unix": time.time(),
                "error_type": type(error).__name__,
                "resume_safe": True,
            },
        )
        _write_status(artifacts)
        raise
    return completed


def run_quarantine(
    artifacts_root: Path | None = None, *, workers: int = 1
) -> dict[str, Any]:
    artifacts = _artifact_root(artifacts_root)
    registration = _load_registration(artifacts)
    arguments = _arguments(
        artifacts, registration, "quarantine", replicates=2, generations=4
    )
    _run_checkpoint_tasks(
        arguments,
        artifacts / "quarantine/checkpoints",
        registration["design_digest"],
        workers,
        artifacts,
    )
    payloads, missing = _checkpoint_payloads(
        artifacts / "quarantine/checkpoints",
        registration["cohorts"]["quarantine"],
        registration["design_digest"],
    )
    passed = not missing and len(payloads) == 2
    for payload in payloads:
        reset_hashes = {
            condition["reset_sha256"]
            for condition in payload["conditions"].values()
        }
        passed = bool(
            passed
            and set(payload["conditions"]) == set(CONDITIONS)
            and all(
                set(condition["outcomes"]) == {"1", "2", "4"}
                for condition in payload["conditions"].values()
            )
            and reset_hashes == {payload["reset"]["array_sha256"]}
            and payload["reset"]["live_cells"] in {3, 4, 5, 6}
        )
    audit = {
        "phase": "engineering_quarantine",
        "passed": passed,
        "pair_count": len(payloads),
        "missing": missing,
        "pairs_previously_exposed_in_v1": True,
        "scientific_inference": False,
        "design_changes_permitted": False,
    }
    write_checkpoint(
        artifacts / "quarantine/SMOKE_AUDIT.json",
        registration["design_digest"],
        audit,
    )
    _write_status(artifacts)
    if not passed:
        raise RuntimeError("v2 engineering quarantine failed")
    return audit


def run_confirmation(
    artifacts_root: Path | None = None, *, workers: int = 8
) -> dict[str, Any]:
    artifacts = _artifact_root(artifacts_root)
    registration = _load_registration(artifacts)
    smoke_path = artifacts / "quarantine/SMOKE_AUDIT.json"
    if not smoke_path.exists():
        raise RuntimeError("run the registered engineering quarantine first")
    smoke = read_checkpoint(smoke_path, registration["design_digest"])
    if not smoke.get("passed"):
        raise RuntimeError("engineering quarantine did not pass")
    arguments = _arguments(
        artifacts,
        registration,
        "confirmation",
        replicates=int(registration["profile"]["replicates"]),
        generations=int(registration["profile"]["generations"]),
    )
    _run_checkpoint_tasks(
        arguments,
        artifacts / "confirmation/checkpoints",
        registration["design_digest"],
        workers,
        artifacts,
    )
    _write_status(artifacts)
    return status(artifacts)


def _raw_rows(payloads: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        for condition, condition_result in payload["conditions"].items():
            for generation, outcome in condition_result["outcomes"].items():
                for observer in ("primary", "terminal"):
                    metric = outcome[observer]
                    rows.append(
                        {
                            "pair_id": payload["pair_id"],
                            "condition": condition,
                            "generation": generation,
                            "observer": observer,
                            "p_a_given_a": metric["p_a_given_a"],
                            "p_a_given_b": metric["p_a_given_b"],
                            "p_b_given_a": metric["p_b_given_a"],
                            "p_b_given_b": metric["p_b_given_b"],
                            "direction_a": metric["direction_a"],
                            "direction_b": metric["direction_b"],
                            "crossover": metric["crossover"],
                            "correct": metric["correct"],
                            "resolved": metric["resolved"],
                            "survival": outcome["survival"],
                        }
                    )
    return rows


def _secondary_rows(payloads: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "pair_id": payload["pair_id"],
            "feature_kind": kind,
            "condition": condition,
            "generation": 16,
            "balanced_accuracy": value,
            "opposite_rescue_branch": condition == "rescue_opposite_enter_g4",
        }
        for payload in payloads
        for kind in ("carrier", "phenotype")
        for condition, value in payload["secondary_decoder"][kind].items()
    ]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())


def _metric_text(metric: Mapping[str, Any]) -> str:
    return f"{metric['mean']:.4f} [{metric['ci'][0]:.4f}, {metric['ci'][1]:.4f}]"


def _ratio_text(ratio: Mapping[str, Any]) -> str:
    return f"{ratio['value']:.4f}" if ratio["defined"] else f"undefined ({ratio['reason']})"


def report(artifacts_root: Path | None = None) -> dict[str, Any]:
    artifacts = _artifact_root(artifacts_root)
    registration = _load_registration(artifacts)
    confirmation = artifacts / "confirmation"
    payloads, missing = _checkpoint_payloads(
        confirmation / "checkpoints",
        registration["cohorts"]["confirmation"],
        registration["design_digest"],
    )
    complete = not missing and len(payloads) == int(PROFILE["confirmation_pairs"])
    if payloads:
        adjudication = adjudicate(
            payloads,
            complete=complete,
            expected_pairs=int(PROFILE["confirmation_pairs"]),
            resamples=int(PROFILE["bootstrap_resamples"]),
        )
    else:
        adjudication = {
            "state": "incomplete",
            "strict_primary_passed": False,
            "verdict": "INCOMPLETE",
            "n_pairs": 0,
            "claim_boundary": CONTRACT["claim_boundary"],
        }
    result = {
        "schema_version": SCHEMA_VERSION,
        "experiment": registration["experiment"],
        "state": "complete" if complete else "incomplete",
        "complete": complete,
        "design_digest": registration["design_digest"],
        "pair_count": len(payloads),
        "missing_pair_ids": missing,
        "profile": registration["profile"],
        "v1_disposition": registration["v1_disposition"],
        "adjudication": adjudication,
    }
    atomic_write_json(confirmation / "RESULTS.json", result)
    _write_csv(confirmation / "RAW_OUTCOMES.csv", _raw_rows(payloads))
    if complete:
        _write_csv(confirmation / "SECONDARY_OUTCOMES.csv", _secondary_rows(payloads))
        carrier = adjudication["carrier_decoder"]
        phenotype = adjudication["phenotype_decoder"]
        report_text = f"""# Corrected independent CA lineage-renewal replication

Verdict: `{adjudication['verdict']}`.

The corrected amplitude-sensitive reader and strict-49--64 writer with gain
0.5 were tested on 92 untouched same-launch founder pairs, 64 futures per
history, and 16 sparse-reset generations. Intervals use 10,000 pair-cluster
bootstrap draws at alpha 0.0125.

## Strict original-form endpoints

- Generation 4: {_metric_text(adjudication['intact_generation4'])}
- Generation 8: {_metric_text(adjudication['intact_generation8'])}
- Generation 16: {_metric_text(adjudication['intact_generation16'])}
- Terminal observer at generation 8: {_metric_text(adjudication['terminal_generation8'])}
- No-rewrite loss: {_ratio_text(adjudication['no_rewrite_loss_fraction'])}
- Same-history rescue restoration: {_ratio_text(adjudication['rescue_restoration_fraction'])}

## Registered secondary endpoints

- Carrier decoder, intact generation 16: {_metric_text(carrier['conditions']['intact'])}; passed: `{carrier['passed']}`
- Visible 41-feature decoder, intact generation 16: {_metric_text(phenotype['conditions']['intact'])}; passed: `{phenotype['passed']}`

The complete causal panel and every gate are recorded in `RESULTS.json`. The
sealed v1 result remains preserved but is classified as a non-comparable model
run and does not enter this verdict. Source results are contextual only.

Claim boundary: {CONTRACT['claim_boundary']}.
"""
        lay_text = f"""# Lay summary

The corrected replication is complete. Its verdict is
`{adjudication['verdict']}`.

Every daughter began from the same tiny launch pattern. Its only history cue
was a 512-number carrier. Cells responded more strongly to a strong favourable
carrier signal than to a weak one; after growth, each daughter measured its own
late pattern and wrote the carrier for the next generation. We also removed,
scrambled, stopped, faded, ablated, rescued, reversed, and slightly corrupted
that carrier to distinguish active copying from a leftover founder imprint.

This v2 result is the relevant local test of the intended mechanism. The older
v1 run tested a materially different reader/reset/order and is not counted
against it. This remains a synthetic cellular-automaton result, not a claim
about biological life, agency, or heredity outside the automaton.
"""
    else:
        report_text = "# Corrected CA lineage-renewal replication\n\nThe registered confirmation is incomplete; no verdict is available.\n"
        lay_text = "# Lay summary\n\nThe corrected run is incomplete, so there is no result yet.\n"
    atomic_write_text(confirmation / "REPORT.md", report_text)
    atomic_write_text(confirmation / "LAY_SUMMARY.md", lay_text)
    atomic_write_json(
        confirmation / "STAGE_DECISION.json",
        {
            "design_digest": registration["design_digest"],
            "verdict": adjudication["verdict"],
            "decision": "review_completed_v2" if complete else "finish_registered_confirmation",
            "review_required": True,
            "automatic_continuation": False,
        },
    )
    atomic_write_json(
        confirmation / "QUEUE.json",
        {
            "state": "awaiting_review" if complete else "confirmation_incomplete",
            "automatic_launch": False,
            "added_experiments": [],
        },
    )
    atomic_write_text(
        confirmation / "REGISTRATION.json",
        (artifacts / "REGISTRATION.json").read_text(encoding="utf-8"),
    )
    filenames = [
        "REGISTRATION.json",
        "RESULTS.json",
        "RAW_OUTCOMES.csv",
        "REPORT.md",
        "LAY_SUMMARY.md",
        "STAGE_DECISION.json",
        "QUEUE.json",
    ]
    if complete:
        filenames.append("SECONDARY_OUTCOMES.csv")
    files = {name: sha256_file(confirmation / name) for name in filenames}
    checkpoints = {
        path.name: sha256_file(path)
        for path in sorted((confirmation / "checkpoints").glob("*.json"))
    }
    atomic_write_json(
        confirmation / "MANIFEST.json",
        {
            "schema_version": SCHEMA_VERSION,
            "files": files,
            "checkpoints": checkpoints,
            "seal_digest": sha256_json({"files": files, "checkpoints": checkpoints}),
        },
    )
    _write_status(artifacts)
    return result


def _progress(artifacts: Path, phase: str, total: int) -> dict[str, Any]:
    path = artifacts / phase / "PROGRESS.json"
    saved = load_json(path) if path.exists() else {}
    checkpoint_dir = artifacts / phase / "checkpoints"
    completed = len(list(checkpoint_dir.glob("*.json"))) if checkpoint_dir.exists() else 0
    state = saved.get("state", "not_started" if completed == 0 else "interrupted")
    if completed == total:
        state = "complete"
    return {
        **saved,
        "state": state,
        "completed": completed,
        "total": total,
        "eta_seconds": 0.0 if completed == total else saved.get("eta_seconds"),
    }


def status(artifacts_root: Path | None = None) -> dict[str, Any]:
    artifacts = _artifact_root(artifacts_root)
    registration_path = artifacts / "REGISTRATION.json"
    if not registration_path.exists():
        return {
            "schema_version": SCHEMA_VERSION,
            "state": "unregistered",
            "next_action": "prepare, validate, audit-tests, then register",
        }
    registration = load_json(registration_path)
    verify_registration(registration)
    quarantine = _progress(
        artifacts, "quarantine", len(registration["cohorts"]["quarantine"])
    )
    confirmation = _progress(
        artifacts, "confirmation", len(registration["cohorts"]["confirmation"])
    )
    result_path = artifacts / "confirmation/RESULTS.json"
    result = load_json(result_path) if result_path.exists() else None
    if result and result.get("complete"):
        state, next_action = "complete", "review the sealed corrected result"
    elif confirmation["state"] == "running":
        state, next_action = "confirmation_running", "monitor or safely resume after interruption"
    elif quarantine["state"] != "complete":
        state, next_action = "registered_awaiting_quarantine", "run the engineering quarantine"
    else:
        state, next_action = "registered_awaiting_confirmation", "launch or resume confirmation"
    return {
        "schema_version": SCHEMA_VERSION,
        "state": state,
        "design_digest": registration["design_digest"],
        "v1_disposition": registration["v1_disposition"],
        "quarantine": quarantine,
        "confirmation": confirmation,
        "verdict": None if result is None else result["adjudication"]["verdict"],
        "automatic_launch": False,
        "next_action": next_action,
    }


def _write_status(artifacts: Path) -> None:
    atomic_write_json(artifacts / "STATUS.json", status(artifacts))


def _validate_confirmation_payload(
    payload: Mapping[str, Any], pair: Mapping[str, Any], context: Mapping[str, Any]
) -> None:
    if payload["pair_id"] != pair["pair_id"] or payload["pair"] != pair:
        raise ValueError("confirmation payload pair mismatch")
    if payload["replicates"] != 64 or payload["generations"] != 16:
        raise ValueError("confirmation profile mismatch")
    if set(payload["conditions"]) != set(CONDITIONS):
        raise ValueError("confirmation condition panel mismatch")
    reset, _, _ = _reset_for_pair(pair, context)
    board = decode_state_hex(reset)
    if payload["reset"] != {
        "state_hex": reset,
        "array_sha256": sha256_bytes(board.tobytes(order="C")),
        "live_cells": int(board.sum()),
    }:
        raise ValueError("confirmation reset mismatch")
    for condition in payload["conditions"].values():
        if set(condition["outcomes"]) != {"1", "2", "4", "8", "16"}:
            raise ValueError("confirmation checkpoint panel incomplete")
        if condition["reset_sha256"] != payload["reset"]["array_sha256"]:
            raise ValueError("condition reset hash mismatch")
    if set(payload["secondary_decoder"]) != {"generation", "carrier", "phenotype"}:
        raise ValueError("secondary endpoint panel absent")
    for kind in ("carrier", "phenotype"):
        if set(payload["secondary_decoder"][kind]) != {
            "intact",
            "no_rewrite",
            "read_disabled",
            "ablate_after_g2",
            "rescue_same_enter_g4",
            "rescue_opposite_enter_g4",
        }:
            raise ValueError("secondary endpoint conditions incomplete")


def verify_all(artifacts_root: Path | None = None) -> dict[str, Any]:
    artifacts = _artifact_root(artifacts_root)
    snapshot = verify_snapshot(artifacts / "input")
    registration = _load_registration(artifacts)
    context = _context(artifacts)
    checkpoint_counts: dict[str, int] = {}
    for phase in ("quarantine", "confirmation"):
        cohort = registration["cohorts"][phase]
        expected = {str(pair["pair_id"]) for pair in cohort}
        directory = artifacts / phase / "checkpoints"
        actual = {path.stem for path in directory.glob("*.json")} if directory.exists() else set()
        if not actual <= expected:
            raise ValueError(f"unexpected {phase} checkpoint identities")
        selected = [pair for pair in cohort if pair["pair_id"] in actual]
        payloads, missing = _checkpoint_payloads(
            directory, selected, registration["design_digest"]
        )
        if missing:
            raise AssertionError("selected checkpoint unexpectedly missing")
        if phase == "confirmation":
            pair_index = {pair["pair_id"]: pair for pair in cohort}
            for payload in payloads:
                _validate_confirmation_payload(
                    payload, pair_index[payload["pair_id"]], context
                )
        checkpoint_counts[phase] = len(actual)
    manifest_path = artifacts / "confirmation/MANIFEST.json"
    report_seal = "absent"
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        files = {
            name: sha256_file(artifacts / "confirmation" / name)
            for name in manifest["files"]
        }
        checkpoints = {
            name: sha256_file(artifacts / "confirmation/checkpoints" / name)
            for name in manifest["checkpoints"]
        }
        if files != manifest["files"] or checkpoints != manifest["checkpoints"]:
            raise ValueError("confirmation report/checkpoint seal mismatch")
        if sha256_json({"files": files, "checkpoints": checkpoints}) != manifest["seal_digest"]:
            raise ValueError("confirmation seal digest mismatch")
        report_seal = "valid"
    result = {
        "schema_version": SCHEMA_VERSION,
        "valid": True,
        "snapshot_digest": snapshot["snapshot_digest"],
        "design_digest": registration["design_digest"],
        "implementation_manifest": "valid",
        "test_audit": "valid",
        "checkpoint_counts": checkpoint_counts,
        "report_seal": report_seal,
        "confirmation_expected": int(PROFILE["confirmation_pairs"]),
        "source_and_v1_evidence_boundary": "valid",
    }
    atomic_write_json(artifacts / "VERIFY.json", result)
    return result


__all__ = [
    "prepare_snapshot",
    "validate_cleanroom",
    "parity_check",
    "audit_tests",
    "register",
    "run_quarantine",
    "run_confirmation",
    "report",
    "status",
    "verify_all",
]
