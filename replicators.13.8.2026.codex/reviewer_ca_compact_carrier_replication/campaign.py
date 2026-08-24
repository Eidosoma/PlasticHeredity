"""Lifecycle orchestration for the sealed compact-carrier replication."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .acquisition import acquire, historical_state_set
from .codec import load_codecs, validate_codecs
from .contract import (
    CANDIDATE_IDS,
    CHECKPOINT_GENERATIONS,
    CONDITIONS,
    CONTRACT,
    DEFAULT_ARTIFACTS,
    ENVIRONMENTS,
    NAMESPACE,
    PACKAGE_ROOT,
    PROFILE,
    SCHEMA_VERSION,
    atomic_write_json,
    atomic_write_text,
    implementation_manifest,
    load_json,
    read_checkpoint,
    seal_registration,
    sha256_file,
    sha256_json,
    verify_registration,
    write_checkpoint,
)
from .engine import simulate_pair_cell, worker_cell
from .inference import adjudicate_campaign
from .snapshot import verify_snapshot


def _root(value: Path | None) -> Path:
    return (value or DEFAULT_ARTIFACTS).resolve()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _context(artifacts: Path) -> dict[str, Any]:
    input_root = artifacts / "input"
    manifest = verify_snapshot(input_root)
    acquisition = load_json(artifacts / "ACQUISITION.json")
    cohorts = load_json(artifacts / "COHORTS.json")
    hypothesis = load_json(input_root / "local/HYPOTHESIS.json")
    launches = load_json(input_root / "local/LAUNCH_RESETS.json")
    reference = load_json(input_root / "local/REFERENCE.json")
    source_design = load_json(input_root / "hypothesis/STAGE4_DESIGN.json")
    source_confirmation = load_json(
        input_root / "hypothesis/STAGE4_CONFIRMATION_DESIGN.json"
    )
    donors = {str(donor["donor_id"]): donor for donor in acquisition["donors"]}
    if len(donors) != len(acquisition["donors"]):
        raise ValueError("duplicate fresh donor identifiers")
    return {
        "artifacts": artifacts,
        "input_root": input_root,
        "manifest": manifest,
        "acquisition": acquisition,
        "cohorts_document": cohorts,
        "cohorts": cohorts["cohorts"],
        "hypothesis": hypothesis,
        "launches": launches,
        "reference": reference,
        "source_design": source_design,
        "source_confirmation": source_confirmation,
        "donors": donors,
        "codecs": load_codecs(input_root),
    }


def _targets(context: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    values = context["hypothesis"]["targets"]
    return (
        {label: np.asarray(values["primary"][label], dtype=np.float64) for label in ("A", "B")},
        {
            label: np.asarray(values["primary_terminal"][label], dtype=np.float64)
            for label in ("A", "B")
        },
    )


def _reference(context: Mapping[str, Any]) -> np.ndarray:
    value = np.asarray(context["reference"]["motif_probability"], dtype=np.float64)
    if value.shape != (512,) or not np.isfinite(value).all() or np.any(value <= 0):
        raise ValueError("frozen motif reference must contain 512 positive finite values")
    if abs(float(value.sum()) - 1.0) > 1e-12:
        raise ValueError("frozen motif reference does not sum to one")
    return value


def _pair_donors(pair: Mapping[str, Any], context: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], str]:
    a = dict(context["donors"][str(pair["a_donor_id"])])
    b = dict(context["donors"][str(pair["b_donor_id"])])
    launch = int(pair["launch_index"])
    reset = str(context["launches"][f"launch{launch}"])
    if (a["prototype_label"], b["prototype_label"]) != ("A", "B"):
        raise ValueError("cohort pair does not preserve A/B order")
    if int(a["launch_index"]) != launch or int(b["launch_index"]) != launch:
        raise ValueError("cohort donor launch mismatch")
    if a["initial_state_hex"] != reset or b["initial_state_hex"] != reset:
        raise ValueError("cohort reset mismatch")
    return a, b, reset


def physics_dependency_manifest() -> dict[str, str]:
    root = PACKAGE_ROOT.parent / "reviewer_ca_lineage_renewal_replication_v2"
    paths = [root / "engine.py", root / "contract.py", root / "__init__.py"]
    return {str(path.relative_to(PACKAGE_ROOT.parent)): sha256_file(path) for path in paths}


def validate(artifacts_root: Path | None = None) -> dict[str, Any]:
    artifacts = _root(artifacts_root)
    context = _context(artifacts)
    codec_audit = validate_codecs(context["input_root"])
    source_contract = context["source_design"]["contract"]
    contract_checks = {
        "rule": source_contract.get("rule") == CONTRACT["rule"],
        "generation_sweeps": source_contract.get("generation_sweeps")
        == CONTRACT["generation_sweeps"],
        "read_sweeps": source_contract.get("read_sweeps") == CONTRACT["read_sweeps"],
        "write_start": source_contract.get("write_start") == CONTRACT["write_window"][0],
        "write_end": source_contract.get("write_end") == CONTRACT["write_window"][1],
        "observe_start": source_contract.get("observe_start")
        == CONTRACT["observation_window"][0],
        "process_noise": source_contract.get("process_noise")
        == CONTRACT["ordinary_process_noise"],
        "repair_gain": source_contract.get("repair_gain") == CONTRACT["repair_gain"],
        "confirmation_alpha": source_contract.get("confirmation_alpha_per_codec")
        == CONTRACT["confirmation_alpha_per_codec"],
    }
    if not all(contract_checks.values()):
        raise ValueError("local contract diverges from the frozen data/document specification")
    confirmation = context["source_confirmation"]
    if tuple(confirmation["candidate_ids"]) != CANDIDATE_IDS:
        raise ValueError("frozen candidate order changed")
    if tuple(confirmation["environments"]) != ENVIRONMENTS:
        raise ValueError("frozen environment panel changed")
    if int(confirmation["replicates"]) != PROFILE["confirmation_replicates"]:
        raise ValueError("frozen replicate count changed")
    if int(confirmation["generations"]) != PROFILE["confirmation_generations"]:
        raise ValueError("frozen generation count changed")
    if float(confirmation["alpha_per_codec"]) != CONTRACT["confirmation_alpha_per_codec"]:
        raise ValueError("frozen alpha changed")

    if context["acquisition"].get("state") != "READY":
        raise ValueError("fresh acquisition did not reach its preregistered sample size")
    if context["cohorts_document"].get("state") != "READY":
        raise ValueError("fresh cohorts are not ready")
    expected_counts = {
        "engineering": int(PROFILE["engineering_pairs"]),
        "confirmation": int(PROFILE["confirmation_pairs"]),
        "audit_reserve": int(PROFILE["audit_reserve_pairs"]),
    }
    observed_counts = {name: len(context["cohorts"][name]) for name in expected_counts}
    if observed_counts != expected_counts:
        raise ValueError("fresh cohort allocation count mismatch")
    pair_ids: list[str] = []
    donor_ids: list[str] = []
    maximum_density_difference = 0.0
    for cohort in context["cohorts"].values():
        for pair in cohort:
            a, b, _ = _pair_donors(pair, context)
            pair_ids.append(str(pair["pair_id"]))
            donor_ids.extend((str(a["donor_id"]), str(b["donor_id"])))
            difference = abs(float(a["density"]) - float(b["density"]))
            if abs(difference - float(pair["density_difference"])) > 1e-15:
                raise ValueError("recorded density distance is inconsistent")
            maximum_density_difference = max(maximum_density_difference, difference)
    if len(pair_ids) != len(set(pair_ids)) or len(donor_ids) != len(set(donor_ids)):
        raise ValueError("cohorts are not pair- and donor-disjoint")
    if maximum_density_difference > float(CONTRACT["density_caliper"]) + 1e-15:
        raise ValueError("fresh pairing exceeds density caliper")
    historical = load_json(context["input_root"] / "local/DONORS.json")
    old_states = historical_state_set(historical)
    overlap = sum(
        context["donors"][donor_id]["donor_state_hex"] in old_states
        for donor_id in donor_ids
    )
    if overlap:
        raise ValueError("fresh cohort contains a historically exposed state")
    if not all(
        context["donors"][donor_id]["best_similarity"]
        >= float(CONTRACT["acquisition_similarity"])
        and context["donors"][donor_id]["assignment_margin"]
        >= float(CONTRACT["acquisition_margin"])
        for donor_id in donor_ids
    ):
        raise ValueError("fresh cohort violates admission thresholds")
    cohort_audit = {
        "valid": True,
        "cohort_counts": observed_counts,
        "pair_ids_unique": True,
        "allocated_donors_unique": True,
        "historical_state_overlap": overlap,
        "maximum_density_difference": maximum_density_difference,
        "thresholds": {
            "similarity": CONTRACT["acquisition_similarity"],
            "margin": CONTRACT["acquisition_margin"],
            "density_caliper": CONTRACT["density_caliper"],
        },
        "confirmation_outcomes_accessed": False,
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "valid": True,
        "snapshot_digest": context["manifest"]["snapshot_digest"],
        "source_code_accessed": False,
        "source_results_or_checkpoints_imported": False,
        "source_outcomes_used_as_evidence": False,
        "contract_checks": contract_checks,
        "reference_valid": bool(_reference(context).shape == (512,)),
        "model_audit_sha256": None,
        "cohort_audit_sha256": None,
        "additional_scientific_experiments": [],
    }
    atomic_write_json(artifacts / "MODEL_AUDIT.json", codec_audit)
    atomic_write_json(artifacts / "COHORT_AUDIT.json", cohort_audit)
    report["model_audit_sha256"] = sha256_file(artifacts / "MODEL_AUDIT.json")
    report["cohort_audit_sha256"] = sha256_file(artifacts / "COHORT_AUDIT.json")
    atomic_write_json(artifacts / "VALIDATION.json", report)
    return report


def smoke(artifacts_root: Path | None = None) -> dict[str, Any]:
    artifacts = _root(artifacts_root)
    if (artifacts / "REGISTRATION.json").exists():
        existing = artifacts / "SMOKE.json"
        if existing.exists():
            return load_json(existing)
        raise RuntimeError("cannot create a post-registration engineering outcome")
    context = _context(artifacts)
    pair = context["cohorts"]["engineering"][0]
    donor_a, donor_b, reset = _pair_donors(pair, context)
    primary, terminal = _targets(context)
    results: dict[str, str] = {}
    for candidate_id in CANDIDATE_IDS:
        for environment in ENVIRONMENTS:
            payload = simulate_pair_cell(
                pair=pair,
                donors={donor_a["donor_id"]: donor_a, donor_b["donor_id"]: donor_b},
                reset_state_hex=reset,
                reference_probability=_reference(context),
                targets_primary=primary,
                targets_terminal=terminal,
                codec=context["codecs"][candidate_id],
                environment=environment,
                replicates=2,
                generations=4,
            )
            results[f"{candidate_id}:{environment}"] = sha256_json(payload)
    report = {
        "schema_version": SCHEMA_VERSION,
        "valid": True,
        "role": "non-evidential engineering smoke test",
        "pair_id": pair["pair_id"],
        "replicates": 2,
        "generations": 4,
        "all_candidates_and_environments": True,
        "cell_digests": results,
    }
    atomic_write_json(artifacts / "SMOKE.json", report)
    return report


def audit_tests(artifacts_root: Path | None = None) -> dict[str, Any]:
    artifacts = _root(artifacts_root)
    if (artifacts / "REGISTRATION.json").exists():
        existing = artifacts / "TEST_AUDIT.json"
        if existing.exists():
            return load_json(existing)
        raise RuntimeError("cannot create the test audit after registration")
    command = [sys.executable, "-m", "pytest", "-q", str(PACKAGE_ROOT / "tests")]
    completed = subprocess.run(
        command,
        cwd=PACKAGE_ROOT.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "passed": completed.returncode == 0,
        "returncode": completed.returncode,
        "command": command,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "implementation_manifest": implementation_manifest(),
        "physics_dependency_manifest": physics_dependency_manifest(),
    }
    atomic_write_json(artifacts / "TEST_AUDIT.json", report)
    if completed.returncode:
        raise RuntimeError(f"test audit failed:\n{completed.stdout}\n{completed.stderr}")
    return report


def register(artifacts_root: Path | None = None) -> dict[str, Any]:
    artifacts = _root(artifacts_root)
    path = artifacts / "REGISTRATION.json"
    if path.exists():
        return _load_registration(artifacts)
    checkpoint_root = artifacts / "confirmation/checkpoints"
    if checkpoint_root.exists() and any(checkpoint_root.glob("*.json")):
        raise RuntimeError("unregistered confirmation outcomes already exist")
    validation = load_json(artifacts / "VALIDATION.json")
    tests = load_json(artifacts / "TEST_AUDIT.json")
    smoke_report = load_json(artifacts / "SMOKE.json")
    if not validation.get("valid") or not tests.get("passed") or not smoke_report.get("valid"):
        raise ValueError("validation, tests, and engineering smoke must pass before sealing")
    if tests["implementation_manifest"] != implementation_manifest():
        raise ValueError("implementation changed after the passing test audit")
    if tests["physics_dependency_manifest"] != physics_dependency_manifest():
        raise ValueError("local physics dependency changed after the test audit")
    context = _context(artifacts)
    candidate_models = {
        key: context["codecs"][key].metadata() for key in CANDIDATE_IDS
    }
    registration = seal_registration(
        {
            "schema_version": SCHEMA_VERSION,
            "experiment": "cleanroom_stage4_compact_carrier_replication",
            "registration_state": "sealed_before_any_fresh_confirmation_outcome",
            "frozen_date_utc": "2026-08-24",
            "namespace": NAMESPACE,
            "snapshot_digest": context["manifest"]["snapshot_digest"],
            "snapshot_manifest_sha256": sha256_file(artifacts / "input/MANIFEST.json"),
            "acquisition_sha256": sha256_file(artifacts / "ACQUISITION.json"),
            "cohorts_sha256": sha256_file(artifacts / "COHORTS.json"),
            "acquisition_audit_sha256": sha256_file(artifacts / "ACQUISITION_AUDIT.json"),
            "model_audit_sha256": sha256_file(artifacts / "MODEL_AUDIT.json"),
            "cohort_audit_sha256": sha256_file(artifacts / "COHORT_AUDIT.json"),
            "validation_sha256": sha256_file(artifacts / "VALIDATION.json"),
            "test_audit_sha256": sha256_file(artifacts / "TEST_AUDIT.json"),
            "smoke_sha256": sha256_file(artifacts / "SMOKE.json"),
            "implementation_manifest": implementation_manifest(),
            "physics_dependency_manifest": physics_dependency_manifest(),
            "candidate_models": candidate_models,
            "candidate_ids": list(CANDIDATE_IDS),
            "environments": list(ENVIRONMENTS),
            "conditions": list(CONDITIONS),
            "contract": CONTRACT,
            "profile": PROFILE,
            "cohorts": context["cohorts"],
            "independent_unit": CONTRACT["independent_unit"],
            "registered_secondary": [
                "Walsh-minus-identity intact crossover at generations 8 and 16"
            ],
            "secondary_is_gating": False,
            "target_replication_gate": (
                "identity ordinary and Walsh-r16-q04 ordinary plus moderate; "
                "PCA cannot substitute"
            ),
            "source_evidential_role": "none; outcome-known hypothesis specification only",
            "source_results_or_checkpoints_in_snapshot": False,
            "confirmation_requires_resume_and_authorization_flags": True,
            "fresh_confirmation_outcomes_present_at_seal": [],
            "additional_scientific_experiments": [],
        }
    )
    atomic_write_json(path, registration)
    _write_status(artifacts)
    return registration


def _load_registration(artifacts: Path) -> dict[str, Any]:
    registration = load_json(artifacts / "REGISTRATION.json")
    verify_registration(registration)
    if registration["implementation_manifest"] != implementation_manifest():
        raise ValueError("implementation changed after registration")
    if registration["physics_dependency_manifest"] != physics_dependency_manifest():
        raise ValueError("physics dependency changed after registration")
    verify_snapshot(artifacts / "input")
    sealed_files = {
        "snapshot_manifest_sha256": "input/MANIFEST.json",
        "acquisition_sha256": "ACQUISITION.json",
        "cohorts_sha256": "COHORTS.json",
        "acquisition_audit_sha256": "ACQUISITION_AUDIT.json",
        "model_audit_sha256": "MODEL_AUDIT.json",
        "cohort_audit_sha256": "COHORT_AUDIT.json",
        "validation_sha256": "VALIDATION.json",
        "test_audit_sha256": "TEST_AUDIT.json",
        "smoke_sha256": "SMOKE.json",
    }
    for key, name in sealed_files.items():
        if registration[key] != sha256_file(artifacts / name):
            raise ValueError(f"sealed artifact changed: {name}")
    return registration


def _cell_id(candidate_id: str, environment: str, pair_id: str) -> str:
    digest = hashlib.sha256(pair_id.encode()).hexdigest()[:20]
    return f"{candidate_id}--{environment}--{digest}"


def _arguments(artifacts: Path, registration: Mapping[str, Any]) -> list[dict[str, Any]]:
    context = _context(artifacts)
    primary, terminal = _targets(context)
    reference = _reference(context)
    arguments: list[dict[str, Any]] = []
    for pair in registration["cohorts"]["confirmation"]:
        donor_a, donor_b, reset = _pair_donors(pair, context)
        selected_donors = {
            str(donor_a["donor_id"]): donor_a,
            str(donor_b["donor_id"]): donor_b,
        }
        for candidate_id in CANDIDATE_IDS:
            for environment in ENVIRONMENTS:
                arguments.append(
                    {
                        "task_id": _cell_id(candidate_id, environment, str(pair["pair_id"])),
                        "artifacts": str(artifacts),
                        "pair": pair,
                        "donors": selected_donors,
                        "reset_state_hex": reset,
                        "reference_probability": reference.tolist(),
                        "targets_primary": {key: value.tolist() for key, value in primary.items()},
                        "targets_terminal": {key: value.tolist() for key, value in terminal.items()},
                        "candidate_id": candidate_id,
                        "environment": environment,
                        "replicates": int(PROFILE["confirmation_replicates"]),
                        "generations": int(PROFILE["confirmation_generations"]),
                        "conditions": list(CONDITIONS),
                    }
                )
    return arguments


def _validate_payload(payload: Mapping[str, Any], argument: Mapping[str, Any]) -> None:
    if payload.get("pair_id") != argument["pair"]["pair_id"]:
        raise ValueError("checkpoint pair mismatch")
    if payload.get("pair") != argument["pair"]:
        raise ValueError("checkpoint pair record mismatch")
    if payload.get("candidate", {}).get("candidate_id") != argument["candidate_id"]:
        raise ValueError("checkpoint candidate mismatch")
    if payload.get("environment") != argument["environment"]:
        raise ValueError("checkpoint environment mismatch")
    if int(payload.get("replicates", -1)) != int(argument["replicates"]):
        raise ValueError("checkpoint replicate mismatch")
    if int(payload.get("generations", -1)) != int(argument["generations"]):
        raise ValueError("checkpoint generation mismatch")
    if set(payload.get("conditions", {})) != set(CONDITIONS):
        raise ValueError("checkpoint condition order mismatch")
    for condition in CONDITIONS:
        if {int(value) for value in payload["conditions"][condition]["outcomes"]} != set(CHECKPOINT_GENERATIONS):
            raise ValueError("checkpoint generation panel mismatch")


def _load_cells(
    artifacts: Path,
    registration: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    checkpoint_root = artifacts / "confirmation/checkpoints"
    existing: dict[str, dict[str, Any]] = {}
    missing: list[dict[str, Any]] = []
    for argument in _arguments(artifacts, registration):
        path = checkpoint_root / f"{argument['task_id']}.json"
        if not path.exists():
            missing.append(argument)
            continue
        payload = read_checkpoint(path, str(registration["design_digest"]))
        _validate_payload(payload, argument)
        existing[str(argument["task_id"])] = payload
    return existing, missing


def run_confirmation(
    artifacts_root: Path | None = None,
    *,
    workers: int | None = None,
    resume: bool = False,
    authorize_confirmation: bool = False,
) -> dict[str, Any]:
    artifacts = _root(artifacts_root)
    if not resume or not authorize_confirmation:
        raise PermissionError("confirmation requires both --resume and --authorize-confirmation")
    registration = _load_registration(artifacts)
    existing, missing = _load_cells(artifacts, registration)
    total = len(existing) + len(missing)
    checkpoint_root = artifacts / "confirmation/checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    state_path = artifacts / "confirmation/RUN_STATE.json"
    if state_path.exists():
        run_state = load_json(state_path)
        started_epoch = float(run_state["started_epoch"])
    else:
        started_epoch = time.time()
    run_state = {
        "schema_version": SCHEMA_VERSION,
        "state": "running" if missing else "complete",
        "started_at": datetime.fromtimestamp(started_epoch, timezone.utc).isoformat(),
        "started_epoch": started_epoch,
        "design_digest": registration["design_digest"],
        "total_cells": total,
        "completed_cells": len(existing),
        "workers": int(workers or PROFILE["default_workers"]),
    }
    atomic_write_json(state_path, run_state)
    atomic_write_json(
        artifacts / "confirmation/QUEUE.json",
        {
            "schema_version": SCHEMA_VERSION,
            "cell_definition": "founder pair x codec x environment",
            "total_cells": total,
            "candidate_ids": list(CANDIDATE_IDS),
            "environments": list(ENVIRONMENTS),
            "confirmation_pairs": len(registration["cohorts"]["confirmation"]),
            "checkpoint_directory": "checkpoints",
        },
    )
    if missing:
        worker_count = max(1, int(workers or PROFILE["default_workers"]))
        budget = float(PROFILE["wall_budget_hours"]) * 3600.0
        reserve = float(PROFILE["reserve_minutes"]) * 60.0
        deadline = started_epoch + budget - reserve
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {executor.submit(worker_cell, argument): argument for argument in missing}
            for future in as_completed(futures):
                argument = futures[future]
                payload = future.result()
                _validate_payload(payload, argument)
                write_checkpoint(
                    checkpoint_root / f"{argument['task_id']}.json",
                    str(registration["design_digest"]),
                    payload,
                )
                existing[str(argument["task_id"])] = payload
                run_state["completed_cells"] = len(existing)
                run_state["last_checkpoint_at"] = _utc_now()
                atomic_write_json(state_path, run_state)
                if time.time() >= deadline:
                    for pending in futures:
                        pending.cancel()
                    run_state["state"] = "paused_at_registered_wall_reserve"
                    run_state["stopped_at"] = _utc_now()
                    atomic_write_json(state_path, run_state)
                    break
    _, remaining = _load_cells(artifacts, registration)
    run_state["completed_cells"] = total - len(remaining)
    run_state["state"] = "complete" if not remaining else run_state.get("state", "incomplete")
    run_state["updated_at"] = _utc_now()
    run_state["elapsed_seconds"] = time.time() - started_epoch
    atomic_write_json(state_path, run_state)
    _write_status(artifacts)
    if not remaining:
        return report(artifacts)
    return status(artifacts)


def _grouped_payloads(
    artifacts: Path, registration: Mapping[str, Any]
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    existing, _ = _load_cells(artifacts, registration)
    grouped = {(candidate, environment): [] for candidate in CANDIDATE_IDS for environment in ENVIRONMENTS}
    for payload in existing.values():
        key = (str(payload["candidate"]["candidate_id"]), str(payload["environment"]))
        grouped[key].append(payload)
    for values in grouped.values():
        values.sort(key=lambda payload: str(payload["pair_id"]))
    return grouped


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        atomic_write_text(path, "")
        return
    columns = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _raw_rows(grouped: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate_id in CANDIDATE_IDS:
        for environment in ENVIRONMENTS:
            for payload in grouped[(candidate_id, environment)]:
                for condition in CONDITIONS:
                    for generation in CHECKPOINT_GENERATIONS:
                        outcome = payload["conditions"][condition]["outcomes"][str(generation)]
                        primary = outcome["primary"]
                        terminal = outcome["terminal"]
                        rows.append(
                            {
                                "pair_id": payload["pair_id"],
                                "candidate_id": candidate_id,
                                "environment": environment,
                                "condition": condition,
                                "generation": generation,
                                "primary_crossover": primary["crossover"],
                                "primary_direction_a": primary["direction_a"],
                                "primary_direction_b": primary["direction_b"],
                                "primary_correct": primary["correct"],
                                "primary_resolved": primary["resolved"],
                                "terminal_crossover": terminal["crossover"],
                                "survival": outcome["survival"],
                            }
                        )
    return rows


def _aggregate_rows(adjudication: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate_id in CANDIDATE_IDS:
        for environment in ENVIRONMENTS:
            strict = adjudication["candidates"][candidate_id]["environments"][environment]["strict"]
            if "conditions" not in strict:
                continue
            for condition in CONDITIONS:
                for generation in CHECKPOINT_GENERATIONS:
                    outcome = strict["conditions"][condition][str(generation)]
                    rows.append(
                        {
                            "candidate_id": candidate_id,
                            "environment": environment,
                            "condition": condition,
                            "generation": generation,
                            "primary_mean": outcome["primary"]["mean"],
                            "primary_ci_low": outcome["primary"]["ci"][0],
                            "primary_ci_high": outcome["primary"]["ci"][1],
                            "terminal_mean": outcome["terminal"]["mean"],
                            "survival_mean": outcome["survival"]["mean"],
                            "n_pairs": outcome["primary"]["n_pairs"],
                        }
                    )
    return rows


def _metric(value: Mapping[str, Any]) -> str:
    return f"{value['mean']:.4f} [{value['ci'][0]:.4f}, {value['ci'][1]:.4f}]"


def _render_reports(adjudication: Mapping[str, Any]) -> tuple[str, str, str]:
    lines = [
        "# Clean-room compact-carrier replication",
        "",
        f"Verdict: `{adjudication['verdict']}`.",
        "",
        f"Target Walsh replication passed: **{adjudication['target_replication_passed']}**.",
        "",
        "All intervals are founder-pair cluster bootstraps at alpha 0.005 per codec.",
        "",
        "| Candidate | Environment | G8 crossover | G16 crossover | Strict ladder |",
        "|---|---|---:|---:|:---:|",
    ]
    table = [
        "# Registered confirmation table",
        "",
        "| Carrier | Payload | Shared codebook | Environment | G8 crossover (99.5% CI) | G16 crossover (99.5% CI) | Full causal ladder |",
        "|---|---:|---:|---|---:|---:|:---:|",
    ]
    for candidate_id in CANDIDATE_IDS:
        candidate = adjudication["candidates"][candidate_id]
        model = candidate["model"]
        for environment in ENVIRONMENTS:
            strict = candidate["environments"][environment]["strict"]
            if "intact_generation8" not in strict:
                g8 = g16 = "incomplete"
            else:
                g8 = _metric(strict["intact_generation8"])
                g16 = _metric(strict["intact_generation16"])
            passed = strict.get("stage4_renewed_gate", False)
            lines.append(f"| {candidate_id} | {environment} | {g8} | {g16} | {passed} |")
            table.append(
                f"| {candidate_id} | {model['payload_bits']} bits | {model['codebook_bits']} bits | "
                f"{environment} | {g8} | {g16} | {passed} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            CONTRACT["claim_boundary"] + ".",
            "",
            "The NewIdeas outcomes were not imported into this evidence bundle. The run is an "
            "outcome-known direct replication using newly generated founders.",
        ]
    )
    lay = [
        "# Lay summary",
        "",
        "This test asks whether a tiny inherited signal can repeatedly steer a fresh cellular "
        "automaton lineage toward one of two ancestral patterns.",
        "",
    ]
    if adjudication["state"] != "complete":
        lay.append("The confirmation is incomplete, so no scientific conclusion is available yet.")
    elif adjudication["target_replication_passed"]:
        lay.append(
            "It worked: the full carrier replicated under ordinary conditions, and the 64-bit "
            "Walsh carrier passed every registered persistence, break, rescue, reversal, and "
            "moderate-damage test. This supports a compact causal carrier inside this CA system."
        )
    else:
        lay.append(
            "The exact target did not fully replicate. The detailed table shows whether the "
            "failure was in the full-carrier anchor, ordinary compact carrier, moderate stress, "
            "or a causal steering control."
        )
    lay.extend(
        [
            "",
            "It does not show transfer to a different substrate: all evidence here remains inside "
            "the same cellular-automaton substrate.",
        ]
    )
    return "\n".join(lines) + "\n", "\n".join(lay) + "\n", "\n".join(table) + "\n"


def report(artifacts_root: Path | None = None) -> dict[str, Any]:
    artifacts = _root(artifacts_root)
    registration = _load_registration(artifacts)
    grouped = _grouped_payloads(artifacts, registration)
    adjudication = adjudicate_campaign(
        grouped,
        candidate_models=registration["candidate_models"],
        expected_pairs=int(PROFILE["confirmation_pairs"]),
        resamples=int(PROFILE["bootstrap_resamples"]),
    )
    completed = sum(len(values) for values in grouped.values())
    results = {
        "schema_version": SCHEMA_VERSION,
        "experiment": registration["experiment"],
        "design_digest": registration["design_digest"],
        "state": adjudication["state"],
        "completed_cells": completed,
        "expected_cells": int(PROFILE["confirmation_pairs"]) * len(CANDIDATE_IDS) * len(ENVIRONMENTS),
        "adjudication": adjudication,
    }
    confirmation = artifacts / "confirmation"
    atomic_write_json(confirmation / "RESULTS.json", results)
    _write_csv(confirmation / "RAW_OUTCOMES.csv", _raw_rows(grouped))
    _write_csv(confirmation / "AGGREGATE.csv", _aggregate_rows(adjudication))
    report_text, lay_text, table_text = _render_reports(adjudication)
    atomic_write_text(confirmation / "REPORT.md", report_text)
    atomic_write_text(confirmation / "LAY_SUMMARY.md", lay_text)
    atomic_write_text(confirmation / "PREPRINT_TABLE.md", table_text)
    atomic_write_json(
        confirmation / "STAGE_DECISION.json",
        {
            "schema_version": SCHEMA_VERSION,
            "state": adjudication["state"],
            "verdict": adjudication["verdict"],
            "target_replication_passed": adjudication["target_replication_passed"],
            "claim_boundary": CONTRACT["claim_boundary"],
        },
    )
    checkpoint_files = sorted((confirmation / "checkpoints").glob("*.json"))
    checkpoint_manifest = {
        "schema_version": SCHEMA_VERSION,
        "design_digest": registration["design_digest"],
        "count": len(checkpoint_files),
        "files": {path.name: sha256_file(path) for path in checkpoint_files},
    }
    checkpoint_manifest["aggregate_sha256"] = sha256_json(checkpoint_manifest["files"])
    atomic_write_json(confirmation / "CHECKPOINT_MANIFEST.json", checkpoint_manifest)
    recomputation = {
        "schema_version": SCHEMA_VERSION,
        "deterministic_pair_bootstrap": True,
        "results_sha256": sha256_file(confirmation / "RESULTS.json"),
        "raw_outcomes_sha256": sha256_file(confirmation / "RAW_OUTCOMES.csv"),
        "aggregate_sha256": sha256_file(confirmation / "AGGREGATE.csv"),
        "checkpoint_aggregate_sha256": checkpoint_manifest["aggregate_sha256"],
    }
    atomic_write_json(confirmation / "RECOMPUTATION.json", recomputation)
    output_names = (
        "RESULTS.json",
        "RAW_OUTCOMES.csv",
        "AGGREGATE.csv",
        "REPORT.md",
        "LAY_SUMMARY.md",
        "PREPRINT_TABLE.md",
        "STAGE_DECISION.json",
        "CHECKPOINT_MANIFEST.json",
        "RECOMPUTATION.json",
    )
    atomic_write_json(
        confirmation / "MANIFEST.json",
        {
            "schema_version": SCHEMA_VERSION,
            "design_digest": registration["design_digest"],
            "files": {name: sha256_file(confirmation / name) for name in output_names},
        },
    )
    _write_status(artifacts)
    return results


def status(artifacts_root: Path | None = None) -> dict[str, Any]:
    artifacts = _root(artifacts_root)
    registration_path = artifacts / "REGISTRATION.json"
    if not registration_path.exists():
        return {
            "schema_version": SCHEMA_VERSION,
            "state": "not_registered",
            "next": "prepare, acquire, validate, audit-tests, smoke, register",
        }
    registration = _load_registration(artifacts)
    existing, missing = _load_cells(artifacts, registration)
    total = len(existing) + len(missing)
    run_state_path = artifacts / "confirmation/RUN_STATE.json"
    elapsed = None
    eta = None
    if run_state_path.exists():
        run_state = load_json(run_state_path)
        elapsed = max(0.0, time.time() - float(run_state["started_epoch"]))
        if existing and missing:
            eta = elapsed / len(existing) * len(missing)
    result = {
        "schema_version": SCHEMA_VERSION,
        "state": "complete" if not missing else ("running_or_resumable" if existing else "registered_not_started"),
        "completed_cells": len(existing),
        "total_cells": total,
        "remaining_cells": len(missing),
        "fraction_complete": len(existing) / total if total else 0.0,
        "elapsed_seconds": elapsed,
        "eta_seconds": eta,
        "confirmation_pair_count": int(PROFILE["confirmation_pairs"]),
        "cell_definition": "founder pair x codec x environment",
    }
    results_path = artifacts / "confirmation/RESULTS.json"
    if results_path.exists():
        results = load_json(results_path)
        result["verdict"] = results["adjudication"]["verdict"]
        result["target_replication_passed"] = results["adjudication"]["target_replication_passed"]
    return result


def _write_status(artifacts: Path) -> None:
    atomic_write_json(artifacts / "STATUS.json", status(artifacts))


def verify_all(artifacts_root: Path | None = None) -> dict[str, Any]:
    artifacts = _root(artifacts_root)
    registration = _load_registration(artifacts)
    existing, missing = _load_cells(artifacts, registration)
    grouped = _grouped_payloads(artifacts, registration)
    results_path = artifacts / "confirmation/RESULTS.json"
    recomputation_matches = None
    if results_path.exists():
        recorded = load_json(results_path)
        recomputed_adjudication = adjudicate_campaign(
            grouped,
            candidate_models=registration["candidate_models"],
            expected_pairs=int(PROFILE["confirmation_pairs"]),
            resamples=int(PROFILE["bootstrap_resamples"]),
        )
        recomputation_matches = sha256_json(recomputed_adjudication) == sha256_json(
            recorded["adjudication"]
        )
        if not recomputation_matches:
            raise ValueError("recorded adjudication does not exactly recompute")
    manifest_path = artifacts / "confirmation/MANIFEST.json"
    manifest_valid = None
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        actual = {
            name: sha256_file(artifacts / "confirmation" / name)
            for name in manifest["files"]
        }
        manifest_valid = actual == manifest["files"]
        if not manifest_valid:
            raise ValueError("confirmation output manifest mismatch")
    report_value = {
        "schema_version": SCHEMA_VERSION,
        "valid": not missing and recomputation_matches is True and manifest_valid is True,
        "registration_valid": True,
        "snapshot_valid": True,
        "checkpoint_count": len(existing),
        "expected_checkpoint_count": len(existing) + len(missing),
        "missing_checkpoints": len(missing),
        "checkpoint_payloads_valid": True,
        "adjudication_recomputes_exactly": recomputation_matches,
        "output_manifest_valid": manifest_valid,
        "source_results_or_checkpoints_used": False,
    }
    atomic_write_json(artifacts / "VERIFY.json", report_value)
    return report_value
