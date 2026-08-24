"""Full prospective CR1 model-guided molecular confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from . import intervention_replication as base
from .config import CANDIDATES, CohortConfig, ExperimentConfig
from .experiment import _json_ready, _runtime_manifest, build_cohort
from .intervention_metrics import compute_one_shot_inference, generate_inference_draws
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
P1_PILOT = RESULT_ROOT / "p1_cr1_model_guided_pilot"
P4_RESULT = RESULT_ROOT / "p4_shared_break_recovery"
DEFAULT_VALIDATION = RESULT_ROOT / "cr1_confirmation_validation"
DEFAULT_REGISTRATION = RESULT_ROOT / "cr1_confirmation_registration"
DEFAULT_SMOKE = RESULT_ROOT / "cr1_confirmation_smoke"
DEFAULT_OUTPUT = RESULT_ROOT / "cr1_model_guided_confirmation"
DEFAULT_WORK = RESULT_ROOT / ".cr1_model_guided_confirmation_work"
DOCUMENT = "CODEX_INTERVENTION_CR1_CONFIRMATION_PREREGISTRATION.md"
SOURCE_FILES = (
    DOCUMENT,
    "plastic_heredity/intervention_cr1_confirmation.py",
    "tests/test_intervention_cr1_confirmation.py",
    "plastic_heredity/intervention_replication.py",
    "plastic_heredity/intervention_core.py",
    "plastic_heredity/intervention_metrics.py",
    "plastic_heredity/simulator.py",
    "plastic_heredity/processes.py",
    "plastic_heredity/features.py",
    "plastic_heredity/experiment.py",
    "plastic_heredity/config.py",
)
PROGRAM_FORMAT = "codex-intervention-cr1-confirmation-v1"
REGISTRATION_FORMAT = "codex-intervention-cr1-confirmation-registration-v1"
VALIDATION_FORMAT = "codex-intervention-cr1-confirmation-validation-v1"
RESULT_FORMAT = "codex-intervention-cr1-confirmation-result-v1"
LABEL = "INTCR1_MODEL_GUIDED_CONFIRMATION_V1"
MATRICES = 200
BRANCHES = 64
LANDMARKS = (20, 35, 50, 65, 80)
HORIZON = 12
BOOTSTRAP_REPETITIONS = 4_096
RANDOMIZATION_REPETITIONS = 4_096
EQUIVALENCE_MARGIN = 0.025
RANDOM_RATIO_LIMIT = 0.25
MINIMUM_AVAILABLE_CPU_HOURS = 17.0


def _seed(name: str) -> str:
    return hashlib.sha256(f"codex-clean-room-cr1-full-confirmation-v1::{name}".encode()).hexdigest()


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
        phase="p1",
        role="full prospective CR1 model-guided molecular confirmation",
        matrices=MATRICES,
        branches=BRANCHES,
        cohort_seed=SEEDS["cohort"],
        selection_seed=SEEDS["selection"],
        future_seed=SEEDS["future"],
        bootstrap_seed=SEEDS["bootstrap"],
        randomization_seed=SEEDS["randomization"],
    )


def experiment(current: base.PhaseSpec | None = None) -> ExperimentConfig:
    selected = phase_spec() if current is None else current
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


def protocol() -> dict[str, Any]:
    value: dict[str, Any] = {
        "format": PROGRAM_FORMAT,
        "status": "frozen_before_any_cr1_confirmation_matrix",
        "predecessor_p1_is_developmental_only": True,
        "p4_outcomes_not_used": True,
        "endpoint": "JOINT_BREAK_RUN3 within F12",
        "cohort": {
            "matrices": MATRICES,
            "candidates": list(CANDIDATES),
            "landmarks": list(LANDMARKS),
            "states": 2 * MATRICES * len(LANDMARKS),
        },
        "arms": list(base.PHASE_ARMS["p1"]),
        "selection": {
            "every_legal_mass_preserving_swap_scored": True,
            "frozen_predictor": "immutable candidate-separated 5x full composite",
            "deterministic_tie_breaking": True,
            "random_uniform_over_legal_swaps": True,
            "selection_stream_separate_from_future": True,
        },
        "futures": {
            "branches_per_arm_state": BRANCHES,
            "horizon": HORIZON,
            "halves": {"A": [0, 31], "B": [32, 63]},
            "primary_futures": 512_000,
            "replay_futures": 512_000,
            "common_random_streams": True,
            "future_seed_excludes_arm": True,
            "no_retries": True,
        },
        "inference": {
            "unit": "whole catalytic matrix",
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "randomization_repetitions": RANDOMIZATION_REPETITIONS,
            "holm_family": "four candidate-half cells",
            "random_noop_equivalence_margin": EQUIVALENCE_MARGIN,
            "random_effect_ratio_limit": RANDOM_RATIO_LIMIT,
            "all_original_cr1_gates_unchanged": True,
        },
        "operational_gate": {
            "p4_terminal_checksum_seal_required": True,
            "minimum_available_cpu_hours_before_launch": MINIMUM_AVAILABLE_CPU_HOURS,
            "no_mid_phase_kill": True,
        },
        "mandatory_stop_after_result": True,
        "seed_domains": SEEDS,
        "claim_boundary": {
            "prohibited": [
                "strict-eight control",
                "agency",
                "biological memory",
                "life",
                "autonomous organization",
                "real prebiotic chemistry",
                "universal origin-of-life mechanism",
            ]
        },
    }
    value["protocol_id"] = _canonical_digest(_json_ready(value))
    return value


def validate(output: Path = DEFAULT_VALIDATION) -> None:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    verify_checksums(ORIGINAL_REGISTRATION)
    verify_checksums(P1_PILOT)
    original = base.verify_registration(ORIGINAL_REGISTRATION)
    pilot_metrics = json.loads((P1_PILOT / "primary_metrics.json").read_text())
    checks = {
        "original_registration_verified": bool(original["registration_id"]),
        "frozen_model_hash_preserved": sha256_file(ORIGINAL_REGISTRATION / "frozen_full_predictor.npz") == base.EXPECTED_MODEL_SHA256,
        "p1_pilot_eligible_for_confirmation": bool(pilot_metrics["pilot_eligibility"]),
        "p1_pilot_retained_as_developmental": True,
        "fresh_seed_domains_unique": len(SEEDS) == len(set(SEEDS.values())),
        "fresh_seeds_disjoint_from_original": not set(SEEDS.values()).intersection(base.SEED_DOMAINS.values()),
        "full_directive_matrix_count": MATRICES == 200,
        "full_directive_branch_count": BRANCHES == 64,
        "original_arm_order": base.PHASE_ARMS["p1"] == ("MODEL_UP", "MODEL_DOWN", "RANDOM", "NOOP"),
        "original_equivalence_margin": EQUIVALENCE_MARGIN == base.EQUIVALENCE_MARGIN,
        "original_random_ratio_limit": RANDOM_RATIO_LIMIT == base.RANDOM_RATIO_LIMIT,
        "p4_outcomes_unavailable_to_design": not P4_RESULT.exists(),
    }
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
    print(f"CR1 confirmation validation sealed: {output}", flush=True)


def register(validation_directory: Path = DEFAULT_VALIDATION, output: Path = DEFAULT_REGISTRATION) -> None:
    validation_directory = validation_directory.resolve()
    output = output.resolve()
    verify_checksums(validation_directory)
    validation_payload = json.loads((validation_directory / "validation.json").read_text())
    if not validation_payload.get("all_pass"):
        raise ValueError("CR1 confirmation validation did not pass")
    for scientific in (DEFAULT_OUTPUT, DEFAULT_WORK):
        if scientific.exists():
            raise FileExistsError(f"CR1 scientific artifact exists before registration: {scientific}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    frozen = protocol()
    payload: dict[str, Any] = {
        "format": REGISTRATION_FORMAT,
        "protocol_id": frozen["protocol_id"],
        "source_hashes": source_hashes(),
        "validation_checksum_manifest_sha256": sha256_file(validation_directory / "SHA256SUMS"),
        "p1_pilot_checksum_manifest_sha256": sha256_file(P1_PILOT / "SHA256SUMS"),
        "frozen_model_sha256": sha256_file(ORIGINAL_REGISTRATION / "frozen_full_predictor.npz"),
        "seed_registry": SEEDS,
        "scientific_matrices_at_registration": 0,
        "scientific_futures_at_registration": 0,
        "p4_outcomes_seen_at_registration": False,
    }
    payload["registration_id"] = _canonical_digest(_json_ready(payload))
    with _atomic_destination(output) as destination:
        (destination / "protocol.json").write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (destination / "seed_registry.json").write_text(json.dumps(SEEDS, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (destination / "registration.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        shutil.copy2(ORIGINAL_REGISTRATION / "frozen_full_predictor.npz", destination / "frozen_full_predictor.npz")
        write_checksums(destination)
    verify_registration(output)
    print(f"CR1 confirmation registration sealed: {payload['registration_id']}", flush=True)


def verify_registration(directory: Path = DEFAULT_REGISTRATION) -> dict[str, Any]:
    directory = directory.resolve()
    verify_checksums(directory)
    value = json.loads((directory / "registration.json").read_text())
    if value.get("format") != REGISTRATION_FORMAT:
        raise ValueError("invalid CR1 confirmation registration")
    if value["source_hashes"] != source_hashes():
        raise ValueError("CR1 confirmation source changed")
    if json.loads((directory / "protocol.json").read_text()) != protocol():
        raise ValueError("CR1 confirmation protocol changed")
    if value["seed_registry"] != SEEDS:
        raise ValueError("CR1 confirmation seed registry changed")
    if value["frozen_model_sha256"] != sha256_file(directory / "frozen_full_predictor.npz"):
        raise ValueError("CR1 confirmation frozen model changed")
    return value


def smoke(registration_directory: Path = DEFAULT_REGISTRATION, output: Path = DEFAULT_SMOKE) -> None:
    registration = verify_registration(registration_directory)
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    smoke_spec = base.PhaseSpec(
        phase="p1",
        role="non-scientific CR1 confirmation smoke",
        matrices=1,
        branches=2,
        cohort_seed=SEEDS["smoke_cohort"],
        selection_seed=SEEDS["smoke_selection"],
        future_seed=SEEDS["smoke_future"],
        bootstrap_seed=SEEDS["validation"],
        randomization_seed=SEEDS["replay"],
    )
    cohort = CohortConfig(1, 2, (5,))
    current_experiment = ExperimentConfig(development=cohort, confirmation=cohort, horizon=HORIZON, master_seed=smoke_spec.cohort_seed, bootstrap_repetitions=8, permutation_repetitions=8)
    with tempfile.TemporaryDirectory(prefix="codex-cr1-confirmation-smoke-", dir=output.parent) as temporary:
        with threadpool_limits(limits=1):
            cases = build_cohort(current_experiment, "INTCR1_CONFIRMATION_SMOKE", cohort)
        generated = base.run_phase_batches(cases, current_experiment, smoke_spec, registration_directory / "frozen_full_predictor.npz", registration["registration_id"], Path(temporary) / "generate", 1, "generate")
        replayed = base.run_phase_batches(cases, current_experiment, smoke_spec, registration_directory / "frozen_full_predictor.npz", registration["registration_id"], Path(temporary) / "replay", 1, "replay")
        replay = base.replay_audit(generated, replayed)
        if not replay["state_edit_endpoint_and_process_digests_exact"]:
            raise AssertionError("CR1 confirmation smoke replay failed")
    with _atomic_destination(output) as destination:
        (destination / "manifest.json").write_text(json.dumps({
            "format": "codex-intervention-cr1-confirmation-smoke-v1",
            "registration_id": registration["registration_id"],
            "scientific_result": False,
            "scientific_matrices": 0,
            "scientific_futures": 0,
            "exhaustive_selection_io_checkpoint_and_replay_passed": True,
            "effect_sizes_disclosed": False,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        write_checksums(destination)
    verify_checksums(output)
    print(f"CR1 confirmation smoke passed: {output}", flush=True)


def _status(work: Path, state: str, detail: str, available_cpu_hours: float | None = None) -> None:
    value: dict[str, Any] = {
        "format": "codex-intervention-cr1-confirmation-status-v1",
        "phase": "cr1_model_guided_confirmation",
        "state": state,
        "detail": detail,
        "mandatory_stop_after_seal": True,
    }
    if available_cpu_hours is not None:
        value["available_cpu_hours_at_launch"] = available_cpu_hours
    work.mkdir(parents=True, exist_ok=True)
    base._atomic_json(work / "campaign_status.json", value)


def _prepare_campaign(work: Path, output: Path, registration: dict[str, Any], available_cpu_hours: float) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if available_cpu_hours < MINIMUM_AVAILABLE_CPU_HOURS:
        raise ValueError(f"CR1 confirmation needs at least {MINIMUM_AVAILABLE_CPU_HOURS:.1f} projected CPU-hours at launch")
    verify_checksums(P4_RESULT)
    p4_manifest = json.loads((P4_RESULT / "manifest.json").read_text())
    work.mkdir(parents=True, exist_ok=True)
    contract: dict[str, Any] = {
        "format": "codex-intervention-cr1-confirmation-campaign-v1",
        "registration_id": registration["registration_id"],
        "output": str(output),
        "matrices": MATRICES,
        "branches": BRANCHES,
        "landmarks": list(LANDMARKS),
        "arms": list(base.PHASE_ARMS["p1"]),
        "available_cpu_hours_at_launch": available_cpu_hours,
        "p4_terminal_manifest_sha256": sha256_file(P4_RESULT / "SHA256SUMS"),
        "p4_terminal_classification": p4_manifest.get("topology_classification", p4_manifest.get("classification")),
        "p4_outcome_not_used_by_frozen_protocol": True,
        "source_hashes": source_hashes(),
    }
    contract["campaign_id"] = _canonical_digest(_json_ready(contract))
    path = work / "campaign_contract.json"
    if path.exists() and json.loads(path.read_text()) != _json_ready(contract):
        raise ValueError("CR1 confirmation work directory belongs to another campaign")
    if not path.exists():
        base._atomic_json(path, contract)
    _status(work, "running", "campaign_initialized", available_cpu_hours)


def _reports(metrics: dict[str, Any]) -> tuple[str, str]:
    lines = [
        "# Full CR1 model-guided molecular confirmation",
        "",
        f"Registered four-cell gate: **{metrics['confirmation_gate_pass']}**.",
        f"Exact replay: **{metrics['integrity_gates']['exact_replay']}**.",
        "",
        "| Cell | Up-down | 95% CI | Holm p | Up-noop | Noop-down | Random-noop 90% CI | Pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in metrics["cells"]:
        contrasts = cell["contrasts"]
        lines.append(
            f"| {cell['cell']} | {contrasts['up_minus_down']['estimate']:+.6f} | {contrasts['up_minus_down']['bootstrap_ci95']} | {cell['up_down_randomization_p_holm']:.6g} | "
            f"{contrasts['up_minus_noop']['estimate']:+.6f} | {contrasts['noop_minus_down']['estimate']:+.6f} | {cell['random_noop_equivalence']['bootstrap_ci90']} | {cell['registered_cell_pass']} |"
        )
    lines.extend([
        "",
        "The predictor, preprocessing, exhaustive legal-edit search, thresholds, and inference were frozen before this cohort existed. Candidates were not pooled.",
        "",
        "This operational simulated-process result cannot establish life, agency, biological memory, real chemistry, or strict-eight control.",
        "",
    ])
    lay = "\n".join([
        "# CR1 confirmation in plain language",
        "",
        "For each simulated assembly, the frozen predictor chose one tiny one-molecule change expected to raise break-and-renewal risk and one expected to lower it. A random one-molecule change and no change were included as controls.",
        "",
        ("The frozen predictor passed every prewritten check in both simulator candidates and both independent branch halves." if metrics["confirmation_gate_pass"] else "The frozen predictor did not pass every prewritten four-cell check. Successful individual effects remain visible, but the full prospective control claim is not confirmed."),
        "",
    ])
    return "\n".join(lines), lay


def _append_ledger(output: Path, registration_id: str, metrics: dict[str, Any]) -> None:
    path = ROOT / "INTERVENTION_RESULTS_LEDGER.md"
    text = path.read_text(encoding="utf-8").rstrip() + "\n"
    marker = f"<!-- sealed-cr1-confirmation-{registration_id} -->"
    if marker in text:
        return
    rows = ["", marker, "## Full CR1 model-guided confirmation sealed", "", f"- Registration: `{registration_id}`", f"- Result: `{output.relative_to(ROOT)}`", f"- Full four-cell gate: **{metrics['confirmation_gate_pass']}**", f"- Exact replay: **{metrics['integrity_gates']['exact_replay']}**", "- Mandatory stop observed; CR2 was not launched automatically.", ""]
    path.write_text(text + "\n".join(rows), encoding="utf-8")


def run(
    registration_directory: Path = DEFAULT_REGISTRATION,
    output: Path = DEFAULT_OUTPUT,
    work: Path = DEFAULT_WORK,
    workers: int = min(os.cpu_count() or 1, 14),
    available_cpu_hours: float = MINIMUM_AVAILABLE_CPU_HOURS,
) -> None:
    if workers < 1:
        raise ValueError("workers must be positive")
    registration_directory = registration_directory.resolve()
    output = output.resolve()
    work = work.resolve()
    registration = verify_registration(registration_directory)
    verify_checksums(DEFAULT_SMOKE)
    _prepare_campaign(work, output, registration, available_cpu_hours)
    current_spec = phase_spec()
    current_experiment = experiment(current_spec)
    print(f"[cr1 confirmation 1/8] Building {MATRICES} fresh matrices and {2 * MATRICES * len(LANDMARKS)} states", flush=True)
    _status(work, "running", "building_natural_states", available_cpu_hours)
    with threadpool_limits(limits=1):
        cases = build_cohort(current_experiment, LABEL, current_experiment.confirmation)
    if len(cases) != 2 * MATRICES * len(LANDMARKS):
        raise AssertionError("CR1 confirmation cohort is incomplete")
    model_path = registration_directory / "frozen_full_predictor.npz"
    futures = len(cases) * len(current_spec.arms) * BRANCHES
    print(f"[cr1 confirmation 2/8] Exhaustively selecting edits and shooting {futures:,} F12 futures", flush=True)
    _status(work, "running", "selection_and_primary_futures", available_cpu_hours)
    generated = base.run_phase_batches(cases, current_experiment, current_spec, model_path, registration["registration_id"], work / "generate", workers, "generate")
    print(f"[cr1 confirmation 3/8] Replaying all {futures:,} futures", flush=True)
    _status(work, "running", "exact_replay", available_cpu_hours)
    replayed = base.run_phase_batches(cases, current_experiment, current_spec, model_path, registration["registration_id"], work / "replay", workers, "replay")
    replay = base.replay_audit(generated, replayed)
    if not replay["state_edit_endpoint_and_process_digests_exact"]:
        raise AssertionError("CR1 confirmation exact replay failed")
    arrays = base._outcome_arrays(cases, generated, current_spec)
    draws = generate_inference_draws(
        MATRICES,
        BOOTSTRAP_REPETITIONS,
        RANDOMIZATION_REPETITIONS,
        np.random.default_rng(derive_seed(SEEDS["bootstrap"], f"{LABEL}.bootstrap")),
        np.random.default_rng(derive_seed(SEEDS["randomization"], f"{LABEL}.randomization")),
    )
    print("[cr1 confirmation 4/8] Computing frozen whole-matrix inference", flush=True)
    _status(work, "running", "whole_matrix_inference", available_cpu_hours)
    metrics, matrix_rows = compute_one_shot_inference(
        cases,
        current_spec.arms,
        arrays["targets"],
        arrays["predictions"],
        draws,
        up_arm="MODEL_UP",
        down_arm="MODEL_DOWN",
        equivalence_margin=EQUIVALENCE_MARGIN,
        random_ratio_limit=RANDOM_RATIO_LIMIT,
    )
    secondary = base._secondary_descriptives(cases, arrays, current_spec)
    print("[cr1 confirmation 5/8] Writing and readback-checking artifacts", flush=True)
    _status(work, "running", "artifact_write_and_readback", available_cpu_hours)
    with _atomic_destination(output) as destination:
        np.savez_compressed(destination / "branch_arrays.npz", **arrays)
        base._write_branch_table(destination / "branches.csv.gz", cases, generated)
        base._write_state_artifacts(destination, cases, generated, arrays)
        base._write_selection_artifacts(destination, cases, generated, current_spec)
        base._write_inference_arrays(destination / "inference_arrays.npz", draws, metrics)
        pd.DataFrame(matrix_rows).to_csv(destination / "matrix_effects.csv", index=False)
        readback = base._readback_metrics(destination, cases, current_spec, metrics, matrix_rows)
        integrity = {
            "exact_replay": replay["state_edit_endpoint_and_process_digests_exact"],
            "artifact_readback_exact": bool(readback["primary_metrics_exact"] and readback["matrix_effects_exact"]),
        }
        metrics["integrity_gates"] = integrity
        metrics["confirmation_gate_pass"] = bool(metrics["registered_all_four_cells_pass"] and all(integrity.values()))
        (destination / "primary_metrics.json").write_text(json.dumps(_json_ready(metrics), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (destination / "secondary_outcomes.json").write_text(json.dumps(_json_ready(secondary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (destination / "replay_audit.json").write_text(json.dumps(_json_ready(replay), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (destination / "readback_audit.json").write_text(json.dumps(readback, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        technical, lay = _reports(metrics)
        (destination / "SCIENTIFIC_REPORT.md").write_text(technical, encoding="utf-8")
        (destination / "LAY_SUMMARY.md").write_text(lay, encoding="utf-8")
        claims = {
            "supported": (["prospective frozen-predictor molecular control of Codex JOINT_BREAK_RUN3"] if metrics["confirmation_gate_pass"] else []),
            "failed_predictions": ([] if metrics["confirmation_gate_pass"] else ["full CR1 four-cell confirmation gate"]),
            "unresolved": ["graded dose response", "parameter transfer", "closed-loop control"],
            "prohibited": protocol()["claim_boundary"]["prohibited"],
        }
        (destination / "claim_boundaries.json").write_text(json.dumps(claims, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest = {
            "format": RESULT_FORMAT,
            "registration_id": registration["registration_id"],
            "matrices": MATRICES,
            "states": len(cases),
            "branches_per_arm_state": BRANCHES,
            "primary_futures": futures,
            "replay_futures": futures,
            "full_four_cell_gate": metrics["confirmation_gate_pass"],
            "exact_replay": integrity["exact_replay"],
            "complete_readback_exact": integrity["artifact_readback_exact"],
            "available_cpu_hours_at_launch": available_cpu_hours,
            "no_refitting_recalibration_or_threshold_change": True,
            "no_future_retry_or_matrix_replacement": True,
            "mandatory_stop_after_this_stage": True,
            "cr2_launched": False,
            "runtime": _runtime_manifest(),
        }
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        write_checksums(destination)
    verify_checksums(output)
    _append_ledger(output, registration["registration_id"], metrics)
    _status(work, "sealed_complete", "mandatory_review_stop", available_cpu_hours)
    print("[cr1 confirmation 6/8] Result checksum sealed", flush=True)
    print("[cr1 confirmation 7/8] Durable ledger and status updated", flush=True)
    print("[cr1 confirmation 8/8] STOPPED; CR2 not launched", flush=True)


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
    run_parser.add_argument("--available-cpu-hours", type=float, required=True)
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
        run(args.registration, args.output, args.work_dir, args.workers, args.available_cpu_hours)
    elif args.command == "status":
        print(json.dumps(read_status(args.work_dir), indent=2, sort_keys=True))
    else:  # pragma: no cover
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()

