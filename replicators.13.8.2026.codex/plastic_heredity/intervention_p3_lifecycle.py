"""Additive lifecycle amendment for the already registered P3 pilot."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from . import intervention_replication as base
from .experiment import _json_ready
from .intervention_metrics import compute_one_shot_inference
from .mechanistic import (
    _atomic_destination,
    _canonical_digest,
    sha256_file,
    verify_checksums,
    write_checksums,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = REPOSITORY_ROOT / "results_intervention_replication"
ORIGINAL_REGISTRATION = RESULT_ROOT / "registration"
DEFAULT_VALIDATION = RESULT_ROOT / "p3_lifecycle_validation"
DEFAULT_AMENDMENT = RESULT_ROOT / "p3_lifecycle_amendment"
DEFAULT_OUTPUT = RESULT_ROOT / "p3_cr4_beta_surgery_pilot"
DEFAULT_WORK = RESULT_ROOT / ".p3_work"
DEFAULT_AUDIT = RESULT_ROOT / "p3_cr4_beta_surgery_pilot_lifecycle_audit"

DOCUMENT = "CODEX_INTERVENTION_P3_LIFECYCLE_AMENDMENT.md"
VALIDATION_FORMAT = "codex-intervention-p3-lifecycle-validation-v1"
AMENDMENT_FORMAT = "codex-intervention-p3-lifecycle-amendment-v1"
AUDIT_FORMAT = "codex-intervention-p3-lifecycle-result-audit-v1"
EXPECTED_ORIGINAL_REGISTRATION_ID = (
    "f61e0340dcd8c9ae6b606c8133ca3d8fb1de2e13fe863719aa67b649e8b74531"
)
SOURCE_FILES = (
    DOCUMENT,
    "conftest.py",
    "plastic_heredity/intervention_p3_lifecycle.py",
    "tests/test_intervention_p3_lifecycle.py",
)


def _source_hashes() -> dict[str, str]:
    return {name: sha256_file(REPOSITORY_ROOT / name) for name in SOURCE_FILES}


def add_derived_pilot_eligibility(
    metrics: dict[str, Any], replay_exact: bool
) -> dict[str, Any]:
    metrics["pilot_eligibility"] = bool(
        metrics["pilot_eligibility_without_replay"] and replay_exact
    )
    return metrics


def corrected_readback_metrics(
    output: Path,
    cases: list[Any],
    spec: base.PhaseSpec,
    expected: dict[str, Any],
    expected_matrix_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Recompute all inference and the omitted replay-dependent field."""

    with np.load(output / "branch_arrays.npz", allow_pickle=False) as archive:
        targets = archive["targets"]
        predictions = archive["predictions"]
    with np.load(output / "inference_arrays.npz", allow_pickle=False) as archive:
        draws = {
            "bootstrap_indices": archive["bootstrap_indices"],
            "randomization_signs": archive["randomization_signs"],
        }
    replay = json.loads((output / "replay_audit.json").read_text(encoding="utf-8"))
    replay_exact = bool(replay["state_edit_endpoint_and_process_digests_exact"])
    up, down = spec.contrast
    observed, matrix_rows = compute_one_shot_inference(
        cases,
        spec.arms,
        targets,
        predictions,
        draws,
        up_arm=up,
        down_arm=down,
        equivalence_margin=base.EQUIVALENCE_MARGIN,
        random_ratio_limit=base.RANDOM_RATIO_LIMIT,
    )
    stored = observed.pop("stored_inference_arrays")
    observed["stored_inference_arrays"] = {
        "path": "inference_arrays.npz",
        "bootstrap_indices_shape": stored["bootstrap_indices_shape"],
        "randomization_signs_shape": stored["randomization_signs_shape"],
        "all_cell_bootstrap_and_randomization_arrays_stored": True,
    }
    add_derived_pilot_eligibility(observed, replay_exact)
    metrics_exact = _json_ready(observed) == _json_ready(expected)
    matrix_effects_exact = _json_ready(matrix_rows) == _json_ready(
        expected_matrix_rows
    )
    if not metrics_exact or not matrix_effects_exact:
        raise ValueError("amended P3 round-trip intervention inference changed")
    return {
        "branch_arrays_reloaded": True,
        "inference_draws_reloaded": True,
        "derived_pilot_eligibility_recomputed": True,
        "primary_metrics_exact": metrics_exact,
        "matrix_effects_exact": matrix_effects_exact,
        "no_fitting_or_recalibration": True,
        "lifecycle_amendment": AMENDMENT_FORMAT,
    }


def _protocol() -> dict[str, Any]:
    spec = base.pilot_spec("p3")
    value: dict[str, Any] = {
        "format": AMENDMENT_FORMAT,
        "status": "sealed_before_any_p3_scientific_matrix",
        "phase": "p3",
        "original_registration_id": EXPECTED_ORIGINAL_REGISTRATION_ID,
        "only_change": (
            "derive readback pilot_eligibility from readback inference and the "
            "written exact-replay audit before dictionary comparison"
        ),
        "scientific_contract_changes": [],
        "original_scientific_design": {
            "role": spec.role,
            "matrices": spec.matrices,
            "branches": spec.branches,
            "landmarks": list(base.LANDMARKS),
            "arms": list(spec.arms),
            "contrast": list(spec.contrast),
            "horizon": base.HORIZON,
            "cohort_seed": spec.cohort_seed,
            "selection_seed": spec.selection_seed,
            "future_seed": spec.future_seed,
            "bootstrap_seed": spec.bootstrap_seed,
            "randomization_seed": spec.randomization_seed,
            "bootstrap_repetitions": base.BOOTSTRAP_REPETITIONS,
            "randomization_repetitions": base.RANDOMIZATION_REPETITIONS,
            "equivalence_margin": base.EQUIVALENCE_MARGIN,
            "random_ratio_limit": base.RANDOM_RATIO_LIMIT,
        },
        "orientation_note": {
            "outgoing_rule_correction_changes_p3": False,
            "reason": "the full present-present beta edge set is transpose invariant",
        },
        "execution": {
            "calls_original_registered_p3_entry_point": True,
            "temporarily_replaces_only_in_memory_readback_callback": True,
            "restores_callback_after_run": True,
            "creates_separate_checksum_sealed_result_audit": True,
            "mandatory_stop_after_p3": True,
            "confirmation_launched": False,
        },
        "claim_boundary": base._protocol()["claim_boundaries"],
    }
    value["protocol_id"] = _canonical_digest(value)
    return value


def validation_checks() -> dict[str, Any]:
    original = base.validation_checks()
    checks = dict(original["checks"])

    def record(name: str, condition: bool, detail: Any = None) -> None:
        if not condition:
            raise AssertionError(f"validation failed: {name}: {detail}")
        checks[name] = {"passed": True, "detail": detail}

    spec = base.pilot_spec("p3")
    record(
        "27_p3_original_contract_unchanged",
        spec.arms == ("LOOSEN", "TIGHTEN", "RANDOM_SURGERY", "NOOP")
        and spec.contrast == ("LOOSEN", "TIGHTEN")
        and spec.matrices == 40
        and spec.branches == 32,
    )
    metrics = {"pilot_eligibility_without_replay": True}
    add_derived_pilot_eligibility(metrics, True)
    record(
        "28_p3_readback_field_derived",
        metrics["pilot_eligibility"] is True,
    )
    present = np.asarray([0, 2, 4], dtype=np.int64)
    rows, columns = np.meshgrid(present, present, indexing="ij")
    block = set(zip(rows.ravel().tolist(), columns.ravel().tolist(), strict=True))
    transpose = {(column, row) for row, column in block}
    record(
        "29_present_present_surgery_is_transpose_invariant",
        block == transpose,
    )
    return {
        "format": VALIDATION_FORMAT,
        "checks": checks,
        "original_required_checks_passed": original["required_checks_passed"],
        "all_checks_passed": all(value["passed"] for value in checks.values()),
        "check_count": len(checks),
        "scientific_cohort_generated": False,
    }


def run_validation(output_directory: Path) -> None:
    output_directory = output_directory.resolve()
    if output_directory.exists():
        raise FileExistsError(f"refusing to overwrite {output_directory}")
    validation = validation_checks()
    command = [
        str(REPOSITORY_ROOT / ".venv/bin/python"),
        "-m",
        "pytest",
        "-q",
        "tests/test_intervention_p3_lifecycle.py",
    ]
    completed = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "P3 lifecycle pytest validation failed\n"
            + completed.stdout
            + "\n"
            + completed.stderr
        )
    with _atomic_destination(output_directory) as output:
        (output / "validation.json").write_text(
            json.dumps(_json_ready(validation), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (output / "pytest_output.txt").write_text(
            completed.stdout + completed.stderr, encoding="utf-8"
        )
        write_checksums(output)
    verify_checksums(output_directory)
    print(f"P3 lifecycle validation passed: {output_directory}", flush=True)


def _append_amendment_notice(amendment_id: str) -> None:
    path = REPOSITORY_ROOT / "INTERVENTION_RESULTS_LEDGER.md"
    text = path.read_text(encoding="utf-8").rstrip() + "\n"
    marker = f"<!-- p3-lifecycle-amendment-{amendment_id} -->"
    if marker in text:
        return
    lines = [
        "",
        marker,
        "## P3 lifecycle amendment registered",
        "",
        f"- Amendment: `{amendment_id}`",
        "- Scientific P3 design, source, seed, surgery, endpoint, and gates remain unchanged.",
        "- The only repair derives `pilot_eligibility` before exact readback comparison.",
        "- No P3 scientific matrix existed when the amendment was sealed.",
        "",
    ]
    path.write_text(text + "\n".join(lines), encoding="utf-8")


def register_amendment(
    validation_directory: Path, output_directory: Path
) -> None:
    validation_directory = validation_directory.resolve()
    output_directory = output_directory.resolve()
    if output_directory.exists():
        raise FileExistsError(f"refusing to overwrite {output_directory}")
    if DEFAULT_OUTPUT.exists() or DEFAULT_WORK.exists():
        raise ValueError("P3 already started; prospective lifecycle amendment is ineligible")
    verify_checksums(validation_directory)
    validation = json.loads(
        (validation_directory / "validation.json").read_text(encoding="utf-8")
    )
    if not validation["all_checks_passed"] or validation["scientific_cohort_generated"]:
        raise ValueError("P3 lifecycle validation is not registration-eligible")
    original = base.verify_registration(ORIGINAL_REGISTRATION)
    if original["registration_id"] != EXPECTED_ORIGINAL_REGISTRATION_ID:
        raise ValueError("unexpected original intervention registration")
    protocol = _protocol()
    with _atomic_destination(output_directory) as output:
        (output / "amendment_protocol.json").write_text(
            json.dumps(_json_ready(protocol), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        shutil.copy2(validation_directory / "validation.json", output / "validation.json")
        shutil.copy2(
            validation_directory / "pytest_output.txt", output / "pytest_output.txt"
        )
        payload: dict[str, Any] = {
            "format": AMENDMENT_FORMAT,
            "status": "sealed_before_any_p3_scientific_matrix",
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": sha256_file(output / "amendment_protocol.json"),
            "source_hashes": _source_hashes(),
            "original_registration_id": original["registration_id"],
            "original_registration_checksum_manifest_sha256": sha256_file(
                ORIGINAL_REGISTRATION / "SHA256SUMS"
            ),
            "validation_checksum_manifest_sha256": sha256_file(
                validation_directory / "SHA256SUMS"
            ),
            "p3_scientific_matrices_generated": False,
        }
        payload["amendment_id"] = _canonical_digest(payload)
        (output / "amendment_registration.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_checksums(output)
    amendment = verify_amendment(output_directory)
    _append_amendment_notice(amendment["amendment_id"])
    print(f"P3 lifecycle amendment sealed: {amendment['amendment_id']}", flush=True)


def verify_amendment(directory: Path) -> dict[str, Any]:
    directory = directory.resolve()
    verify_checksums(directory)
    payload = json.loads(
        (directory / "amendment_registration.json").read_text(encoding="utf-8")
    )
    identifier = payload.pop("amendment_id")
    if (
        payload.get("format") != AMENDMENT_FORMAT
        or payload.get("status") != "sealed_before_any_p3_scientific_matrix"
        or _canonical_digest(payload) != identifier
    ):
        raise ValueError("invalid P3 lifecycle amendment")
    payload["amendment_id"] = identifier
    if payload["source_hashes"] != _source_hashes():
        raise ValueError("P3 lifecycle source changed after sealing")
    original = base.verify_registration(ORIGINAL_REGISTRATION)
    if original["registration_id"] != payload["original_registration_id"]:
        raise ValueError("original intervention registration changed")
    protocol = json.loads(
        (directory / "amendment_protocol.json").read_text(encoding="utf-8")
    )
    if protocol != json.loads(json.dumps(_json_ready(_protocol()))):
        raise ValueError("P3 lifecycle protocol implementation diverged")
    if (
        protocol["protocol_id"] != payload["protocol_id"]
        or sha256_file(directory / "amendment_protocol.json")
        != payload["protocol_sha256"]
    ):
        raise ValueError("P3 lifecycle protocol digest changed")
    validation = json.loads((directory / "validation.json").read_text(encoding="utf-8"))
    if not validation["all_checks_passed"]:
        raise ValueError("P3 lifecycle validation no longer passes")
    return payload


def _append_result_audit_notice(amendment_id: str, audit_directory: Path) -> None:
    path = REPOSITORY_ROOT / "INTERVENTION_RESULTS_LEDGER.md"
    text = path.read_text(encoding="utf-8").rstrip() + "\n"
    marker = f"<!-- p3-lifecycle-result-{amendment_id} -->"
    if marker in text:
        return
    lines = [
        "",
        marker,
        "## P3 lifecycle result audit",
        "",
        f"- Amendment: `{amendment_id}`",
        f"- Audit: `{audit_directory.relative_to(REPOSITORY_ROOT)}`",
        "- P3 result used the unchanged original scientific contract.",
        "- Corrected readback, exact replay, and result checksums passed.",
        "",
    ]
    path.write_text(text + "\n".join(lines), encoding="utf-8")


def _write_result_audit(
    amendment: dict[str, Any], result: Path, audit_directory: Path
) -> None:
    result = result.resolve()
    audit_directory = audit_directory.resolve()
    if audit_directory.exists():
        raise FileExistsError(f"refusing to overwrite {audit_directory}")
    verify_checksums(result)
    manifest = json.loads((result / "manifest.json").read_text(encoding="utf-8"))
    replay = json.loads((result / "replay_audit.json").read_text(encoding="utf-8"))
    readback = json.loads((result / "readback_audit.json").read_text(encoding="utf-8"))
    payload: dict[str, Any] = {
        "format": AUDIT_FORMAT,
        "amendment_id": amendment["amendment_id"],
        "original_registration_id": amendment["original_registration_id"],
        "result_directory": str(result),
        "result_checksum_manifest_sha256": sha256_file(result / "SHA256SUMS"),
        "phase": manifest["phase"],
        "primary_futures": manifest["primary_futures"],
        "replay_futures": manifest["replay_futures"],
        "exact_replay": replay["state_edit_endpoint_and_process_digests_exact"],
        "complete_readback_exact": bool(
            readback["primary_metrics_exact"]
            and readback["matrix_effects_exact"]
            and readback["derived_pilot_eligibility_recomputed"]
        ),
        "scientific_contract_changes": [],
        "confirmation_launched": False,
    }
    payload["audit_id"] = _canonical_digest(payload)
    with _atomic_destination(audit_directory) as output:
        (output / "result_audit.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_checksums(output)
    verify_checksums(audit_directory)
    _append_result_audit_notice(amendment["amendment_id"], audit_directory)


def run_p3(
    amendment_directory: Path,
    output_directory: Path,
    work_directory: Path,
    audit_directory: Path,
    workers: int,
) -> None:
    amendment = verify_amendment(amendment_directory)
    if output_directory.exists():
        raise FileExistsError(f"refusing to overwrite {output_directory}")
    original_readback = base._readback_metrics
    base._readback_metrics = corrected_readback_metrics
    try:
        base.run_pilot(
            "p3",
            ORIGINAL_REGISTRATION,
            output_directory,
            workers,
            work_directory,
        )
    finally:
        base._readback_metrics = original_readback
    _write_result_audit(amendment, output_directory, audit_directory)
    print(
        "P3 lifecycle audit sealed; mandatory stop before confirmation",
        flush=True,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="P3 readback lifecycle amendment")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--output", type=Path, default=DEFAULT_VALIDATION)
    register = commands.add_parser("register")
    register.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    register.add_argument("--output", type=Path, default=DEFAULT_AMENDMENT)
    verify = commands.add_parser("verify")
    verify.add_argument("--amendment", type=Path, default=DEFAULT_AMENDMENT)
    run = commands.add_parser("run")
    run.add_argument("--amendment", type=Path, default=DEFAULT_AMENDMENT)
    run.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    run.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    run.add_argument("--workers", type=int, default=14)
    status = commands.add_parser("status")
    status.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    return parser


def main(argv: list[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    if arguments.command == "validate":
        run_validation(arguments.output)
    elif arguments.command == "register":
        register_amendment(arguments.validation, arguments.output)
    elif arguments.command == "verify":
        print(
            json.dumps(
                verify_amendment(arguments.amendment), indent=2, sort_keys=True
            )
        )
    elif arguments.command == "run":
        run_p3(
            arguments.amendment,
            arguments.output,
            arguments.work_dir,
            arguments.audit,
            arguments.workers,
        )
    elif arguments.command == "status":
        print(json.dumps(base.read_status(arguments.work_dir), indent=2, sort_keys=True))
    else:  # pragma: no cover
        raise AssertionError(arguments.command)


if __name__ == "__main__":
    main()
