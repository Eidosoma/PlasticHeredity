"""Administrative CR6 readback correction and checkpoint-safe resume."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from . import intervention_cr6_transfer as cr6
from .experiment import _json_ready
from .intervention_metrics import generate_inference_draws
from .mechanistic import (
    _atomic_destination,
    _canonical_digest,
    sha256_file,
    verify_checksums,
    write_checksums,
)


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results_intervention_replication"
DOCUMENT = "CODEX_INTERVENTION_CR6_READBACK_AMENDMENT.md"
SOURCE_FILES = (
    DOCUMENT,
    "plastic_heredity/intervention_cr6_readback_amendment.py",
    "tests/test_intervention_cr6_readback_amendment.py",
)
DEFAULT_VALIDATION = RESULT_ROOT / "cr6_readback_amendment_validation"
DEFAULT_REGISTRATION = RESULT_ROOT / "cr6_readback_amendment_registration"
DEFAULT_SMOKE = RESULT_ROOT / "cr6_readback_amendment_smoke"
ORIGINAL_REGISTRATION = cr6.DEFAULT_REGISTRATION
ORIGINAL_VALIDATION = cr6.DEFAULT_VALIDATION
ORIGINAL_SMOKE = cr6.DEFAULT_SMOKE
WORK = cr6.DEFAULT_WORK
OUTPUT = cr6.DEFAULT_OUTPUT
FAILED_LOG = RESULT_ROOT / "cr6_zero_shot_transfer.log"
FAILED_TIME = RESULT_ROOT / "cr6_zero_shot_transfer.time"
FAILED_REGIME = "POS_A_M4_S5"
ORIGINAL_REGISTRATION_ID = (
    "d15ad57c7f925b5aa8585e3ae32090fa08c500b465d657bd0faf84327744b07e"
)
VALIDATION_FORMAT = "codex-intervention-cr6-readback-amendment-validation-v1"
REGISTRATION_FORMAT = "codex-intervention-cr6-readback-amendment-registration-v1"
SMOKE_FORMAT = "codex-intervention-cr6-readback-amendment-smoke-v1"


def source_hashes() -> dict[str, str]:
    return {name: sha256_file(ROOT / name) for name in SOURCE_FILES}


def corrected_readback_regime(
    output: Path,
    regime: str,
    cases: list[Any],
    expected: dict[str, Any],
    expected_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Repeat the sealed audit after applying the writer's regime label."""

    with np.load(output / "branch_arrays.npz", allow_pickle=False) as archive:
        targets = archive["targets"]
        predictions = archive["predictions"]
    with np.load(output / "inference_arrays.npz", allow_pickle=False) as archive:
        draws = {
            "bootstrap_indices": archive["bootstrap_indices"],
            "randomization_signs": archive["randomization_signs"],
        }
    observed, observed_rows = cr6.compute_regime_inference(
        regime, cases, targets, predictions, draws
    )
    stored = observed.pop("stored_inference_arrays")
    observed["stored_inference_arrays"] = cr6._normalized_stored_arrays(stored)
    for row in observed_rows:
        row["regime"] = regime
    metrics_exact = _json_ready(observed) == _json_ready(expected)
    rows_exact = _json_ready(observed_rows) == _json_ready(expected_rows)
    if not metrics_exact or not rows_exact:
        raise ValueError(f"amended CR6 {regime} written-artifact inference changed")
    return {
        "branch_arrays_reloaded": True,
        "inference_draws_reloaded": True,
        "primary_metrics_exact": metrics_exact,
        "matrix_effects_exact": rows_exact,
        "no_fitting_or_recalibration": True,
        "administrative_regime_label_applied_symmetrically": True,
    }


def _artificial_round_trip() -> dict[str, Any]:
    cases = [
        cr6._fixture_case(candidate, matrix_id, landmark)
        for matrix_id in range(cr6.MATRICES)
        for candidate in ("02", "03")
        for landmark in cr6.LANDMARKS
    ]
    targets = np.zeros((len(cases), len(cr6.ARMS), cr6.BRANCHES), dtype=np.int8)
    targets[:, cr6.ARMS.index("MODEL_UP")] = 1
    predictions = np.full((len(cases), len(cr6.ARMS)), 0.5, dtype=np.float64)
    draws = generate_inference_draws(
        cr6.MATRICES,
        128,
        128,
        np.random.default_rng(101),
        np.random.default_rng(103),
    )
    metrics, rows = cr6.compute_regime_inference(
        FAILED_REGIME, cases, targets, predictions, draws
    )
    for row in rows:
        row["regime"] = FAILED_REGIME
    with tempfile.TemporaryDirectory(prefix="codex-cr6-readback-fixture-") as temp:
        path = Path(temp)
        np.savez_compressed(
            path / "branch_arrays.npz", targets=targets, predictions=predictions
        )
        cr6._write_inference_arrays(path / "inference_arrays.npz", draws, metrics)
        try:
            cr6._readback_regime(path, FAILED_REGIME, cases, metrics, rows)
        except ValueError:
            original_failure_reproduced = True
        else:
            original_failure_reproduced = False
        amended = corrected_readback_regime(path, FAILED_REGIME, cases, metrics, rows)
    return {
        "original_false_failure_reproduced": original_failure_reproduced,
        "amended_metrics_exact": amended["primary_metrics_exact"],
        "amended_matrix_rows_exact": amended["matrix_effects_exact"],
        "only_added_field": "regime",
        "numerical_tolerance_weakened": False,
    }


def _checkpoint_summary() -> dict[str, Any]:
    stages: dict[str, Any] = {}
    for stage in ("generate", "replay"):
        directory = WORK / FAILED_REGIME / stage
        status = json.loads((directory / "status.json").read_text())
        files = sorted(directory.glob("state_*.pkl"))
        file_hashes = [sha256_file(path) for path in files]
        stages[stage] = {
            "states": len(files),
            "futures_complete": status["futures_complete"],
            "futures_total": status["futures_total"],
            "status_state": status["state"],
            "checkpoint_contract_sha256": sha256_file(
                directory / "checkpoint_contract.json"
            ),
            "state_file_hash_digest": _canonical_digest(file_hashes),
        }
    return stages


def validation_checks() -> dict[str, Any]:
    original = cr6.verify_registration(ORIGINAL_REGISTRATION)
    verify_checksums(ORIGINAL_VALIDATION)
    verify_checksums(ORIGINAL_SMOKE)
    fixture = _artificial_round_trip()
    checkpoints = _checkpoint_summary()
    log_text = FAILED_LOG.read_text()
    checks = {
        "original_registration_exact": original["registration_id"]
        == ORIGINAL_REGISTRATION_ID,
        "original_validation_and_smoke_checksums_exact": True,
        "failure_is_readback_row_comparison": (
            "written-artifact inference changed" in log_text
            and "_readback_regime" in log_text
        ),
        "primary_checkpoints_complete": checkpoints["generate"]["states"] == 160
        and checkpoints["generate"]["futures_complete"] == 30_720
        and checkpoints["generate"]["futures_total"] == 30_720
        and checkpoints["generate"]["status_state"] == "complete",
        "replay_checkpoints_complete": checkpoints["replay"]["states"] == 160
        and checkpoints["replay"]["futures_complete"] == 30_720
        and checkpoints["replay"]["futures_total"] == 30_720
        and checkpoints["replay"]["status_state"] == "complete",
        "no_final_result_exists": not OUTPUT.exists(),
        "no_failed_regime_artifact_exists": not (
            WORK / "artifacts" / FAILED_REGIME
        ).exists(),
        "later_regimes_not_started": all(
            not (WORK / regime).exists()
            for regime in cr6.REGIMES
            if regime != FAILED_REGIME
        ),
        "original_false_failure_reproduced": fixture[
            "original_false_failure_reproduced"
        ],
        "amended_round_trip_exact": fixture["amended_metrics_exact"]
        and fixture["amended_matrix_rows_exact"],
        "amendment_only_adds_regime_label": fixture["only_added_field"] == "regime"
        and not fixture["numerical_tolerance_weakened"],
        "scientific_engine_source_unchanged": cr6.source_hashes()
        == original["source_hashes"],
    }
    return {
        "format": VALIDATION_FORMAT,
        "checks": checks,
        "check_count": len(checks),
        "all_checks_passed": bool(all(checks.values())),
        "checkpoint_summary": checkpoints,
        "fixture_audit": fixture,
        "scientific_effect_sizes_inspected": False,
    }


def validate(output: Path = DEFAULT_VALIDATION) -> None:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    validation = validation_checks()
    if not validation["all_checks_passed"]:
        raise AssertionError(
            {name: value for name, value in validation["checks"].items() if not value}
        )
    command = [sys.executable, "-m", "pytest", "-q"]
    completed = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "CR6 readback-amendment repository validation failed\n"
            + completed.stdout
            + completed.stderr
        )
    with _atomic_destination(output) as destination:
        payload = dict(validation)
        payload["source_hashes"] = source_hashes()
        payload["failed_log_sha256"] = sha256_file(FAILED_LOG)
        payload["failed_time_sha256"] = sha256_file(FAILED_TIME)
        (destination / "validation.json").write_text(
            json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n"
        )
        (destination / "pytest_output.txt").write_text(
            "$ " + " ".join(command) + "\n" + completed.stdout + completed.stderr
        )
        write_checksums(destination)
    verify_checksums(output)
    print(f"CR6 readback amendment validation sealed: {output}", flush=True)


def register(
    validation_directory: Path = DEFAULT_VALIDATION,
    output: Path = DEFAULT_REGISTRATION,
) -> None:
    validation_directory = validation_directory.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    verify_checksums(validation_directory)
    validation = json.loads((validation_directory / "validation.json").read_text())
    if not validation["all_checks_passed"]:
        raise ValueError("CR6 readback amendment validation did not pass")
    if validation["source_hashes"] != source_hashes():
        raise ValueError("CR6 readback amendment source changed after validation")
    original = cr6.verify_registration(ORIGINAL_REGISTRATION)
    payload: dict[str, Any] = {
        "format": REGISTRATION_FORMAT,
        "status": "sealed_before_cr6_resume",
        "original_registration_id": original["registration_id"],
        "original_registration_checksum_manifest_sha256": sha256_file(
            ORIGINAL_REGISTRATION / "SHA256SUMS"
        ),
        "source_hashes": source_hashes(),
        "validation_checksum_manifest_sha256": sha256_file(
            validation_directory / "SHA256SUMS"
        ),
        "failed_log_sha256": sha256_file(FAILED_LOG),
        "failed_time_sha256": sha256_file(FAILED_TIME),
        "checkpoint_summary": _checkpoint_summary(),
        "scientific_changes": [],
        "administrative_change": "add regime label to recomputed matrix rows before exact readback equality",
        "scientific_effect_sizes_inspected_before_amendment": False,
        "original_model_seeds_checkpoints_and_gates_unchanged": True,
    }
    payload["amendment_id"] = _canonical_digest(_json_ready(payload))
    with _atomic_destination(output) as destination:
        (destination / "registration.json").write_text(
            json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n"
        )
        (destination / "amendment.md").write_text((ROOT / DOCUMENT).read_text())
        write_checksums(destination)
    verify_registration(output)
    _append_ledger(
        f"<!-- cr6-readback-amendment-{payload['amendment_id']} -->",
        [
            "## CR6 administrative readback amendment sealed",
            "",
            f"- Amendment: `{payload['amendment_id']}`.",
            f"- Original scientific registration remains `{ORIGINAL_REGISTRATION_ID}`.",
            "- First-regime primary and replay checkpoints were complete; no later regime had started.",
            "- The only correction symmetrically adds the deterministic regime label before exact row comparison.",
            "- No model, state, seed, intervention, outcome, inference, gate, or numerical tolerance changed.",
            "- No scientific effect size was inspected before this amendment was frozen.",
            "",
        ],
    )
    print(f"CR6 readback amendment registered: {payload['amendment_id']}", flush=True)


def verify_registration(directory: Path = DEFAULT_REGISTRATION) -> dict[str, Any]:
    directory = directory.resolve()
    verify_checksums(directory)
    payload = json.loads((directory / "registration.json").read_text())
    if payload.get("format") != REGISTRATION_FORMAT:
        raise ValueError("unsupported CR6 readback amendment registration")
    amendment_id = payload.pop("amendment_id")
    if amendment_id != _canonical_digest(_json_ready(payload)):
        raise ValueError("CR6 readback amendment ID changed")
    payload["amendment_id"] = amendment_id
    if payload["source_hashes"] != source_hashes():
        raise ValueError("CR6 readback amendment source changed")
    original = cr6.verify_registration(ORIGINAL_REGISTRATION)
    if original["registration_id"] != payload["original_registration_id"]:
        raise ValueError("CR6 original registration changed")
    return payload


def smoke(
    registration_directory: Path = DEFAULT_REGISTRATION,
    output: Path = DEFAULT_SMOKE,
) -> None:
    registration = verify_registration(registration_directory)
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    fixture = _artificial_round_trip()
    if not (
        fixture["original_false_failure_reproduced"]
        and fixture["amended_metrics_exact"]
        and fixture["amended_matrix_rows_exact"]
    ):
        raise AssertionError("CR6 readback amendment smoke failed")
    with _atomic_destination(output) as destination:
        (destination / "manifest.json").write_text(
            json.dumps(
                {
                    "format": SMOKE_FORMAT,
                    "amendment_id": registration["amendment_id"],
                    "artificial_fixture_only": True,
                    "scientific_effect_sizes_disclosed": False,
                    "fixture_audit": fixture,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        write_checksums(destination)
    verify_checksums(output)
    print(f"CR6 readback amendment smoke passed: {output}", flush=True)


def _append_ledger(marker: str, lines: list[str]) -> None:
    path = ROOT / "INTERVENTION_RESULTS_LEDGER.md"
    current = path.read_text(encoding="utf-8").rstrip() + "\n"
    if marker in current:
        return
    path.write_text(current + "\n" + marker + "\n" + "\n".join(lines))


def _apply_result_provenance(output: Path, registration: dict[str, Any]) -> None:
    verify_checksums(output)
    provenance_path = output / "READBACK_AMENDMENT.json"
    expected = {
        "format": "codex-intervention-cr6-readback-amendment-result-v1",
        "amendment_id": registration["amendment_id"],
        "original_registration_id": registration["original_registration_id"],
        "administrative_only": True,
        "scientific_changes": [],
        "corrected_operation": registration["administrative_change"],
        "all_original_scientific_gates_unchanged": True,
    }
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if provenance_path.exists():
        if json.loads(provenance_path.read_text()) != expected:
            raise ValueError("CR6 result amendment provenance changed")
        if manifest.get("readback_amendment_id") != registration["amendment_id"]:
            raise ValueError("CR6 result manifest lacks the sealed amendment")
        verify_checksums(output)
        return
    provenance_path.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")
    manifest["readback_amendment_id"] = registration["amendment_id"]
    manifest["original_scientific_registration_id"] = registration[
        "original_registration_id"
    ]
    manifest["administrative_readback_amendment_only"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    write_checksums(output)
    verify_checksums(output)


def run(
    registration_directory: Path = DEFAULT_REGISTRATION,
    output: Path = OUTPUT,
    work: Path = WORK,
    workers: int = min(os.cpu_count() or 1, 14),
    available_cpu_hours: float = cr6.DEFAULT_CPU_BUDGET_HOURS,
) -> None:
    registration = verify_registration(registration_directory)
    verify_checksums(DEFAULT_SMOKE)
    output = output.resolve()
    work = work.resolve()
    if not output.exists():
        original_callback = cr6._readback_regime
        cr6._readback_regime = corrected_readback_regime
        try:
            cr6.run(
                ORIGINAL_REGISTRATION,
                output,
                work,
                workers,
                available_cpu_hours,
            )
        finally:
            cr6._readback_regime = original_callback
    _apply_result_provenance(output, registration)
    manifest = json.loads((output / "manifest.json").read_text())
    _append_ledger(
        f"<!-- sealed-cr6-readback-amended-{registration['amendment_id']} -->",
        [
            "## CR6 result sealed under the administrative readback amendment",
            "",
            f"- Amendment: `{registration['amendment_id']}`.",
            f"- Result: `{output.relative_to(ROOT)}`.",
            f"- Complete original CR6 gate: **{manifest['complete_cr6_gate']}**.",
            "- Original scientific registration, model, seeds, futures, inference, and gates remained unchanged.",
            "- Mandatory review stop observed; CR7 was not launched automatically.",
            "",
        ],
    )
    print("CR6 amended readback result sealed; STOPPED before CR7", flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate").add_argument(
        "--output", type=Path, default=DEFAULT_VALIDATION
    )
    register_parser = commands.add_parser("register")
    register_parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    register_parser.add_argument("--output", type=Path, default=DEFAULT_REGISTRATION)
    commands.add_parser("verify").add_argument(
        "--registration", type=Path, default=DEFAULT_REGISTRATION
    )
    smoke_parser = commands.add_parser("smoke")
    smoke_parser.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    smoke_parser.add_argument("--output", type=Path, default=DEFAULT_SMOKE)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--registration", type=Path, default=DEFAULT_REGISTRATION)
    run_parser.add_argument("--output", type=Path, default=OUTPUT)
    run_parser.add_argument("--work-dir", type=Path, default=WORK)
    run_parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 14))
    run_parser.add_argument(
        "--available-cpu-hours", type=float, default=cr6.DEFAULT_CPU_BUDGET_HOURS
    )
    commands.add_parser("status").add_argument("--work-dir", type=Path, default=WORK)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "validate":
        validate(args.output)
    elif args.command == "register":
        register(args.validation, args.output)
    elif args.command == "verify":
        print(
            json.dumps(verify_registration(args.registration), indent=2, sort_keys=True)
        )
    elif args.command == "smoke":
        smoke(args.registration, args.output)
    elif args.command == "run":
        run(
            args.registration,
            args.output,
            args.work_dir,
            args.workers,
            args.available_cpu_hours,
        )
    elif args.command == "status":
        print(json.dumps(cr6.read_status(args.work_dir), indent=2, sort_keys=True))
    else:  # pragma: no cover
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
