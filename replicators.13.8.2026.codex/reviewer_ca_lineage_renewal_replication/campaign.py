"""Registration, execution, reporting, status, and verification workflows."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import io
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from reviewer_motif_channel_replication.contract import (
    read_checkpoint as read_base_checkpoint,
    verify_registration as verify_base_registration,
)
from reviewer_motif_channel_replication.engine import (
    deterministic_board,
    motif_counts,
    read_motif_energy,
    step_rule31649,
    texture2x2_counts,
    write_carrier,
)

from .cohorts import allocate, eligible_pairs
from .contract import (
    CONDITIONS,
    CONTRACT,
    DEFAULT_ARTIFACTS,
    FIXED_CONFIGURATION,
    NAMESPACE,
    PAIRING_NAMESPACE,
    PROFILE,
    SCHEMA_VERSION,
    atomic_write_json,
    atomic_write_text,
    dependency_manifest,
    implementation_manifest,
    load_json,
    read_checkpoint,
    seal_registration,
    sha256_file,
    sha256_json,
    verify_registration,
    write_checkpoint,
)
from .engine import (
    motif_counts_batch,
    read_motif_energy_batch,
    simulate_pair_lineages,
    step_rule31649_batch,
    texture2x2_counts_batch,
    write_carriers_batch,
)
from .inference import adjudicate
from .snapshot import verify_snapshot


SOURCE_CANDIDATE = "simple--strict-49-64--gain-050"


def _artifact_root(value: Path | None) -> Path:
    return (value or DEFAULT_ARTIFACTS).resolve()


def _context(artifacts: Path) -> dict[str, Any]:
    input_root = artifacts / "input"
    manifest = verify_snapshot(input_root)
    local = input_root / "local"
    newideas = input_root / "newideas"
    donors = load_json(local / "DONORS.json")["donors"]
    pool_document = load_json(local / "FRESH_PAIR_POOL.json")
    stage1_registration = load_json(local / "STAGE1_REGISTRATION.json")
    stage2_registration = load_json(local / "STAGE2_REGISTRATION.json")
    return {
        "manifest": manifest,
        "input_root": input_root,
        "local_root": local,
        "newideas_root": newideas,
        "donors": donors,
        "donor_index": {str(donor["donor_id"]): donor for donor in donors},
        "pool": pool_document["pairs"],
        "pool_document": pool_document,
        "hypothesis": load_json(local / "HYPOTHESIS.json"),
        "stage1_registration": stage1_registration,
        "stage2_registration": stage2_registration,
        "stage1_decision": load_json(local / "STAGE1_DECISION.json"),
        "stage2_decision": load_json(local / "STAGE2_DECISION.json"),
        "source_cohorts": load_json(newideas / "stage3r/COHORTS.json"),
        "source_design": load_json(newideas / "stage3r/DESIGN.json"),
        "source_results": load_json(newideas / "stage3r/RESULTS.json"),
        "source_decision": load_json(newideas / "stage3r/STAGE_DECISION.json"),
    }


def _eligible(context: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return eligible_pairs(
        context["pool"],
        context["stage1_registration"],
        context["stage2_registration"],
        context["source_cohorts"],
    )


def validate_cleanroom(artifacts_root: Path | None = None) -> dict[str, Any]:
    artifacts = _artifact_root(artifacts_root)
    context = _context(artifacts)
    verify_base_registration(context["stage1_registration"])
    verify_base_registration(context["stage2_registration"])
    sealed_dependencies = {
        name: context["stage1_registration"]["implementation_manifest"][name]
        for name in ("__init__.py", "contract.py", "engine.py")
    }
    current_dependencies = dependency_manifest()
    if current_dependencies != sealed_dependencies:
        raise ValueError("the reused local scalar primitives changed after Stage 1")
    eligible, audit = _eligible(context)
    cohorts = allocate(eligible)

    donor_ids = set(context["donor_index"])
    allocated_donors = [
        str(pair[key])
        for cohort in cohorts.values()
        for pair in cohort
        for key in ("a_donor_id", "b_donor_id")
    ]
    if len(allocated_donors) != len(set(allocated_donors)):
        raise AssertionError("a donor is reused in the Stage-3R replication")
    if not set(allocated_donors) <= donor_ids:
        raise AssertionError("an allocated donor is absent from the frozen acquisition data")
    if len(eligible) != 98:
        raise AssertionError(f"expected exactly 98 fully fresh pairs; found {len(eligible)}")
    if audit["max_density_difference"] > 0.02:
        raise AssertionError("fresh-pair density caliper violated")
    if context["stage1_decision"].get("verdict") != "ROBUST_LOCAL_MOTIF_CONTROLLABILITY":
        raise ValueError("the frozen local Stage-1 prerequisite did not pass")
    if context["stage2_decision"].get("verdict") != "DENSITY_ROBUST_GENERAL_MOTIF_CHANNEL":
        raise ValueError("the frozen local Stage-2 prerequisite did not pass")
    if context["source_decision"].get("verdict") != "STRICT_RENEWED_CA_PLASTIC_HEREDITY":
        raise ValueError("the data/docs hypothesis source is not the final positive result")
    source_candidate = context["source_results"]["adjudication"]["candidates"].get(
        SOURCE_CANDIDATE
    )
    if source_candidate is None or source_candidate["model"].get("gain") != 0.5:
        raise ValueError("the fixed Stage-3R source candidate is absent or changed")
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
    }
    if not all(contract_checks.values()):
        raise ValueError("the frozen local lifecycle diverges from the data/docs contract")

    report = {
        "schema_version": SCHEMA_VERSION,
        "valid": True,
        "snapshot_digest": context["manifest"]["snapshot_digest"],
        "data_docs_only_source": True,
        "source_code_opened_hashed_imported_or_executed": False,
        "source_candidate": SOURCE_CANDIDATE,
        "source_verdict_used_as_evidence": False,
        "source_role": "hypothesis and fixed protocol only",
        "local_prerequisites": {
            "stage1": context["stage1_decision"]["verdict"],
            "stage2": context["stage2_decision"]["verdict"],
        },
        "contract_checks": contract_checks,
        "sealed_dependency_checks": {
            name: current_dependencies[name] == sealed_dependencies[name]
            for name in sealed_dependencies
        },
        "donor_audit": audit,
        "cohort_counts": {key: len(value) for key, value in cohorts.items()},
        "allocated_donors_unique": True,
        "confirmation_donor_overlap_with_prior_local_or_newideas": 0,
        "new_scientific_experiments_added": False,
        "secondary_decoder_status": "excluded because data/docs do not operationally specify it",
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
    return report


def _reference(context: Mapping[str, Any]) -> dict[str, np.ndarray]:
    registration = context["stage1_registration"]
    verify_base_registration(registration)
    payload = read_base_checkpoint(
        context["local_root"] / "STAGE1_CALIBRATION.json",
        registration["design_digest"],
    )
    return {
        key: np.asarray(value, dtype=np.float64)
        for key, value in payload["reference"]["32"].items()
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


def parity_check(artifacts_root: Path | None = None) -> dict[str, Any]:
    """Check local vectorized primitives against the sealed scalar primitives."""

    artifacts = _artifact_root(artifacts_root)
    context = _context(artifacts)
    reference = _reference(context)
    boards = np.stack(
        [
            deterministic_board("stage3r-parity", index, density=0.35 + 0.1 * index)
            for index in range(3)
        ]
    )
    scalar_steps = np.stack([step_rule31649(board) for board in boards])
    batch_steps = step_rule31649_batch(boards)
    scalar_motifs = np.stack([motif_counts(board) for board in boards])
    batch_motifs = motif_counts_batch(boards)
    scalar_textures = np.stack([texture2x2_counts(board) for board in boards])
    batch_textures = texture2x2_counts_batch(boards)
    scalar_carriers = np.stack(
        [
            write_carrier(
                {
                    "motif": counts,
                    "context_total": np.zeros(256),
                    "context_live": np.zeros(256),
                },
                reference,
                "motif_energy512",
            )
            for counts in scalar_motifs
        ]
    )
    batch_carriers = write_carriers_batch(
        batch_motifs, reference["motif_probability"]
    )
    uniform = np.random.default_rng(7813).random(boards.shape)
    scalar_reads = np.stack(
        [
            read_motif_energy(board, carrier, 0.25, field)
            for board, carrier, field in zip(
                scalar_steps, scalar_carriers, uniform, strict=True
            )
        ]
    )
    batch_reads = read_motif_energy_batch(
        batch_steps, batch_carriers, 0.25, uniform
    )
    checks = {
        "rule_batch_exact": bool(np.array_equal(scalar_steps, batch_steps)),
        "motif_counts_batch_exact": bool(np.array_equal(scalar_motifs, batch_motifs)),
        "texture_counts_batch_exact": bool(
            np.array_equal(scalar_textures, batch_textures)
        ),
        "writer_batch_exact": bool(np.array_equal(scalar_carriers, batch_carriers)),
        "reader_batch_exact": bool(np.array_equal(scalar_reads, batch_reads)),
        "zero_reader_inert": bool(
            np.array_equal(
                batch_steps,
                read_motif_energy_batch(
                    batch_steps, np.zeros_like(batch_carriers), 0.25, uniform
                ),
            )
        ),
    }
    if not all(checks.values()):
        raise AssertionError(f"deterministic parity failed: {checks}")
    report = {
        "schema_version": SCHEMA_VERSION,
        "valid": True,
        "checks": checks,
        "trajectory_parity_against_source": "not performed; source code is prohibited",
        "fresh_outcomes_generated": False,
        "evidential_role": "implementation validation only",
    }
    atomic_write_json(artifacts / "PARITY.json", report)
    return report


def register(artifacts_root: Path | None = None) -> dict[str, Any]:
    artifacts = _artifact_root(artifacts_root)
    path = artifacts / "REGISTRATION.json"
    if path.exists():
        return _load_registration(artifacts)
    checkpoint_root = artifacts / "confirmation/checkpoints"
    if checkpoint_root.exists() and any(checkpoint_root.glob("*.json")):
        raise RuntimeError("unregistered confirmation outcomes already exist")
    smoke_root = artifacts / "quarantine/checkpoints"
    if smoke_root.exists() and any(smoke_root.glob("*.json")):
        raise RuntimeError("unregistered quarantine outcomes already exist")
    validation = load_json(artifacts / "VALIDATION.json")
    parity = load_json(artifacts / "PARITY.json")
    if not validation.get("valid") or not parity.get("valid"):
        raise ValueError("validation and parity must pass before registration")
    context = _context(artifacts)
    eligible, audit = _eligible(context)
    cohorts = allocate(eligible)
    registration = seal_registration(
        {
            "schema_version": SCHEMA_VERSION,
            "experiment": "independent_ca_lineage_renewal_stage3r_replication",
            "registration_state": "sealed_before_any_local_lineage_outcome",
            "frozen_date_utc": "2026-08-23",
            "namespace": NAMESPACE,
            "snapshot_digest": context["manifest"]["snapshot_digest"],
            "snapshot_manifest_sha256": sha256_file(
                context["input_root"] / "MANIFEST.json"
            ),
            "validation_sha256": sha256_file(artifacts / "VALIDATION.json"),
            "parity_sha256": sha256_file(artifacts / "PARITY.json"),
            "implementation_manifest": implementation_manifest(),
            "dependency_manifest": dependency_manifest(),
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
            "cohort_audit": audit,
            "source_bindings": {
                "design_sha256": context["manifest"]["newideas_data_docs"][
                    "stage3r/DESIGN.json"
                ],
                "results_sha256": context["manifest"]["newideas_data_docs"][
                    "stage3r/RESULTS.json"
                ],
                "decision_sha256": context["manifest"]["newideas_data_docs"][
                    "stage3r/STAGE_DECISION.json"
                ],
                "evidential_role": "none; hypothesis and protocol source only",
            },
            "local_upstream_bindings": {
                name: context["manifest"]["local_frozen_evidence"][name]
                for name in (
                    "STAGE1_CALIBRATION.json",
                    "STAGE1_REGISTRATION.json",
                    "STAGE1_RESULTS.json",
                    "STAGE1_DECISION.json",
                    "STAGE2_REGISTRATION.json",
                    "STAGE2_RESULTS.json",
                    "STAGE2_DECISION.json",
                    "FRESH_PAIR_POOL.json",
                )
            },
            "independent_unit": "matched founder pair",
            "quarantine_role": "engineering only; never enters inference",
            "secondary_decoder": {
                "status": "not_replicated",
                "reason": "independent texture descriptor is underspecified in data/docs",
                "gating": False,
            },
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
        raise ValueError("implementation changed after registration; outcomes are blocked")
    if registration["dependency_manifest"] != dependency_manifest():
        raise ValueError("sealed scalar dependency changed after registration")
    context = _context(artifacts)
    if registration["snapshot_digest"] != context["manifest"]["snapshot_digest"]:
        raise ValueError("data/docs snapshot changed after registration")
    if registration["validation_sha256"] != sha256_file(artifacts / "VALIDATION.json"):
        raise ValueError("clean-room validation changed after registration")
    if registration["parity_sha256"] != sha256_file(artifacts / "PARITY.json"):
        raise ValueError("parity record changed after registration")
    return registration


def _worker(argument: Mapping[str, Any]) -> dict[str, Any]:
    pair = argument["pair"]
    donor_index = argument["donor_index"]
    reference = {
        key: np.asarray(value, dtype=np.float64)
        for key, value in argument["reference"].items()
    }
    primary = {
        key: np.asarray(value, dtype=np.float64)
        for key, value in argument["targets_primary"].items()
    }
    terminal = {
        key: np.asarray(value, dtype=np.float64)
        for key, value in argument["targets_terminal"].items()
    }
    payload = simulate_pair_lineages(
        pair_id=str(pair["pair_id"]),
        donor_state_hex=[
            donor_index[str(pair["a_donor_id"])]["donor_state_hex"],
            donor_index[str(pair["b_donor_id"])]["donor_state_hex"],
        ],
        reference=reference,
        targets_primary=primary,
        targets_terminal=terminal,
        replicates=int(argument["replicates"]),
        generations=int(argument["generations"]),
        conditions=argument["conditions"],
    )
    return {
        "phase": argument["phase"],
        "pair": dict(pair),
        **payload,
    }


def _arguments(
    artifacts: Path,
    registration: Mapping[str, Any],
    cohort_name: str,
    *,
    replicates: int,
    generations: int,
) -> list[dict[str, Any]]:
    context = _context(artifacts)
    reference = _reference(context)
    primary, terminal = _targets(context)
    common = {
        "phase": cohort_name,
        "reference": {key: value.tolist() for key, value in reference.items()},
        "targets_primary": {key: value.tolist() for key, value in primary.items()},
        "targets_terminal": {key: value.tolist() for key, value in terminal.items()},
        "replicates": replicates,
        "generations": generations,
        "conditions": registration["conditions"],
    }
    arguments = []
    for pair in registration["cohorts"][cohort_name]:
        donor_index = {
            donor_id: context["donor_index"][donor_id]
            for donor_id in (pair["a_donor_id"], pair["b_donor_id"])
        }
        arguments.append({**common, "pair": pair, "donor_index": donor_index})
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
        if payload.get("pair_id") != pair["pair_id"]:
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

    def record_interruption(session_completed: int, error: BaseException) -> None:
        atomic_write_json(
            progress_path,
            {
                "state": "interrupted",
                "completed": already_complete + session_completed,
                "total": len(arguments),
                "eta_seconds": None,
                "session_started_unix": started_unix,
                "last_update_unix": time.time(),
                "error_type": type(error).__name__,
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
        record_interruption(completed, error)
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
    passed = not missing
    for payload in payloads:
        passed = bool(
            passed
            and set(payload["conditions"]) == set(CONDITIONS)
            and all(
                set(value["outcomes"]) == {"1", "2", "4"}
                for value in payload["conditions"].values()
            )
            and np.isfinite(
                [
                    value["outcomes"][generation][observer]["crossover"]
                    for value in payload["conditions"].values()
                    for generation in ("1", "2", "4")
                    for observer in ("primary", "terminal")
                ]
            ).all()
        )
    audit = {
        "phase": "engineering_quarantine",
        "passed": passed,
        "pair_count": len(payloads),
        "missing": missing,
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
        raise RuntimeError("engineering quarantine failed")
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


def report(artifacts_root: Path | None = None) -> dict[str, Any]:
    artifacts = _artifact_root(artifacts_root)
    registration = _load_registration(artifacts)
    confirmation_root = artifacts / "confirmation"
    payloads, missing = _checkpoint_payloads(
        confirmation_root / "checkpoints",
        registration["cohorts"]["confirmation"],
        registration["design_digest"],
    )
    complete = not missing
    if payloads:
        adjudication = adjudicate(
            payloads,
            complete=complete,
            expected_pairs=int(registration["profile"]["confirmation_pairs"]),
            resamples=int(registration["profile"]["bootstrap_resamples"]),
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
        "experiment": "independent_ca_lineage_renewal_stage3r_replication",
        "state": "complete" if complete else "incomplete",
        "complete": complete,
        "design_digest": registration["design_digest"],
        "pair_count": len(payloads),
        "missing_pair_ids": missing,
        "profile": registration["profile"],
        "adjudication": adjudication,
    }
    atomic_write_json(confirmation_root / "RESULTS.json", result)
    _write_csv(confirmation_root / "RAW_OUTCOMES.csv", _raw_rows(payloads))

    if complete:
        verdict = adjudication["verdict"]
        report_text = f"""# Independent CA lineage-renewal replication

Verdict: `{verdict}`.

The fixed `motif_energy512-w32-s025-d32` reader and the preregistered
strict-49--64 daughter writer with universal gain 0.5 were tested on 96 fully
fresh matched founder pairs, 64 futures per history, and 16 visibly reset
generations. The independent unit was the founder pair; intervals are 10,000
pair-cluster bootstrap draws at alpha 0.0125.

## Original-form persistence

- Generation 4: {_metric_text(adjudication['intact_generation4'])}
- Generation 8: {_metric_text(adjudication['intact_generation8'])}
- Generation 16: {_metric_text(adjudication['intact_generation16'])}
- Terminal observer at generation 8: {_metric_text(adjudication['terminal_generation8'])}

## Causal renewal

- No-rewrite loss fraction: {adjudication['no_rewrite_loss_fraction']:.4f}
- Active rewrite advantage at generation 8: {_metric_text(adjudication['active_rewrite_advantage_generation8'])}
- Ablation loss fraction at generation 4: {adjudication['ablation_loss_fraction']:.4f}
- Same-history rescue restoration: {adjudication['rescue_restoration_fraction']:.4f}
- Opposite rescue at generation 4: {_metric_text(adjudication['opposite_rescue_generation4'])}
- Opposite founder at generation 8: {_metric_text(adjudication['opposite_founder_generation8'])}
- One-percent corruption at generation 8: {_metric_text(adjudication['carrier_corruption_generation8'])}

Every registered gate must pass for the strict verdict. The secondary drift
decoder was not replicated because its independent texture descriptor was not
operationally specified in the source data/docs; it is non-gating for this
strict original-form claim.

Claim boundary: {CONTRACT['claim_boundary']}.
"""
        lay_text = f"""# Lay summary

We repeatedly erased the daughter's visible starting pattern, gave it only a
hidden 512-number carrier from its parent, and let it grow. Near the end of each
generation the daughter measured its own pattern and wrote a new carrier for
the next daughter. The local verdict is `{verdict}`.

The key distinction is active copying versus a fading founder imprint. The
registered controls ask whether the signal disappears when the carrier is
removed, scrambled, unread, or not rewritten; whether removing it after two
generations breaks the lineage; and whether inserting the matching or opposite
carrier steers the next generation in the predicted direction.

This is a synthetic cellular-automaton lineage-memory result. It is not a claim
about biological heredity, life, agency, or memory outside the automaton.
"""
    else:
        report_text = "# Independent CA lineage-renewal replication\n\nThe confirmation run is incomplete. No scientific verdict is available.\n"
        lay_text = "# Lay summary\n\nThe run is incomplete, so there is no result yet.\n"
    atomic_write_text(confirmation_root / "REPORT.md", report_text)
    atomic_write_text(confirmation_root / "LAY_SUMMARY.md", lay_text)
    decision = {
        "design_digest": registration["design_digest"],
        "verdict": adjudication["verdict"],
        "decision": (
            "review_completed_replication"
            if complete
            else "complete_registered_confirmation_before_review"
        ),
        "review_required": True,
        "automatic_continuation": False,
    }
    atomic_write_json(confirmation_root / "STAGE_DECISION.json", decision)
    atomic_write_json(
        confirmation_root / "QUEUE.json",
        {
            "state": "awaiting_review" if complete else "confirmation_incomplete",
            "automatic_launch": False,
            "added_experiments": [],
        },
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
    atomic_write_text(
        confirmation_root / "REGISTRATION.json",
        (artifacts / "REGISTRATION.json").read_text(encoding="utf-8"),
    )
    files = {
        name: sha256_file(confirmation_root / name)
        for name in filenames
        if (confirmation_root / name).exists()
    }
    atomic_write_json(
        confirmation_root / "MANIFEST.json",
        {
            "schema_version": SCHEMA_VERSION,
            "files": files,
            "seal_digest": sha256_json(files),
        },
    )
    _write_status(artifacts)
    return result


def _progress(artifacts: Path, phase: str, total: int) -> dict[str, Any]:
    path = artifacts / phase / "PROGRESS.json"
    if path.exists():
        return load_json(path)
    checkpoint_dir = artifacts / phase / "checkpoints"
    completed = len(list(checkpoint_dir.glob("*.json"))) if checkpoint_dir.exists() else 0
    return {
        "state": "not_started" if completed == 0 else "interrupted",
        "completed": completed,
        "total": total,
        "eta_seconds": None,
    }


def status(artifacts_root: Path | None = None) -> dict[str, Any]:
    artifacts = _artifact_root(artifacts_root)
    registration_path = artifacts / "REGISTRATION.json"
    if not registration_path.exists():
        return {
            "schema_version": SCHEMA_VERSION,
            "state": "unregistered",
            "next_action": "snapshot, validate, parity, then register",
        }
    registration = load_json(registration_path)
    verify_registration(registration)
    quarantine = _progress(
        artifacts, "quarantine", len(registration["cohorts"]["quarantine"])
    )
    confirmation = _progress(
        artifacts, "confirmation", len(registration["cohorts"]["confirmation"])
    )
    results_path = artifacts / "confirmation/RESULTS.json"
    result = load_json(results_path) if results_path.exists() else None
    if result and result.get("complete"):
        state = "complete"
        next_action = "review the sealed replication result"
    elif confirmation["state"] == "running":
        state = "confirmation_running"
        next_action = "monitor or safely resume if interrupted"
    elif quarantine["state"] != "complete":
        state = "registered_awaiting_quarantine"
        next_action = "explicitly run the engineering quarantine"
    else:
        state = "registered_awaiting_confirmation"
        next_action = "explicitly launch or resume confirmation"
    return {
        "schema_version": SCHEMA_VERSION,
        "state": state,
        "design_digest": registration["design_digest"],
        "source_candidate": registration["repair"]["source_candidate"],
        "quarantine": quarantine,
        "confirmation": confirmation,
        "verdict": None if result is None else result["adjudication"]["verdict"],
        "automatic_launch": False,
        "next_action": next_action,
    }


def _write_status(artifacts: Path) -> None:
    atomic_write_json(artifacts / "STATUS.json", status(artifacts))


def verify_all(artifacts_root: Path | None = None) -> dict[str, Any]:
    artifacts = _artifact_root(artifacts_root)
    snapshot = verify_snapshot(artifacts / "input")
    registration = _load_registration(artifacts)
    checkpoint_counts: dict[str, int] = {}
    for phase in ("quarantine", "confirmation"):
        cohort = registration["cohorts"][phase]
        expected = {str(pair["pair_id"]) for pair in cohort}
        directory = artifacts / phase / "checkpoints"
        actual = {path.stem for path in directory.glob("*.json")} if directory.exists() else set()
        if not actual <= expected:
            raise ValueError(f"unexpected {phase} checkpoint identities")
        _checkpoint_payloads(directory, [pair for pair in cohort if pair["pair_id"] in actual], registration["design_digest"])
        checkpoint_counts[phase] = len(actual)
    manifest_path = artifacts / "confirmation/MANIFEST.json"
    report_seal = "absent"
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        actual_files = {
            name: sha256_file(artifacts / "confirmation" / name)
            for name in manifest["files"]
        }
        if actual_files != manifest["files"] or sha256_json(actual_files) != manifest["seal_digest"]:
            raise ValueError("confirmation report seal mismatch")
        report_seal = "valid"
    result = {
        "schema_version": SCHEMA_VERSION,
        "valid": True,
        "snapshot_digest": snapshot["snapshot_digest"],
        "design_digest": registration["design_digest"],
        "implementation_manifest": "valid",
        "dependency_manifest": "valid",
        "checkpoint_counts": checkpoint_counts,
        "report_seal": report_seal,
        "data_docs_only_source_boundary": "valid",
    }
    atomic_write_json(artifacts / "VERIFY.json", result)
    return result
