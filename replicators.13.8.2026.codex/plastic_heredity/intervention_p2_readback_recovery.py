"""Checksum-sealed, zero-future recovery of the completed P2 pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from . import intervention_replication as original
from .archive_paths import protocols_equal_after_relocation, relocated_path
from .experiment import _json_ready
from .intervention_readback_common import recover_completed_pilot
from .intervention_readback_recovery import _require_completed_checkpoints
from .mechanistic import (
    _atomic_destination,
    _canonical_digest,
    sha256_file,
    verify_checksums,
    write_checksums,
)

# Detached checkpoints were created while intervention_replication ran as
# __main__.  Exposing the identical registered class under the recovery
# __main__ namespace is a serialization alias, not a scientific transformation.
PhaseBatch = original.PhaseBatch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AMENDMENT_FORMAT = "codex-intervention-p2-readback-amendment-v1"
EXPECTED_ORIGINAL_REGISTRATION_ID = (
    "f61e0340dcd8c9ae6b606c8133ca3d8fb1de2e13fe863719aa67b649e8b74531"
)
EXPECTED_FAILURE = "ValueError: round-trip intervention inference changed"
SOURCE_FILES = (
    "CODEX_INTERVENTION_P2_READBACK_AMENDMENT.md",
    "plastic_heredity/intervention_readback_common.py",
    "plastic_heredity/intervention_p2_readback_recovery.py",
    "plastic_heredity/intervention_readback_recovery.py",
    "tests/test_intervention_p2_readback_recovery.py",
)
SEALED_PRE_RELOCATION_SOURCE_HASHES = {
    "CODEX_INTERVENTION_P2_READBACK_AMENDMENT.md": "57f68fef736b7b9e1cfa92e87dc373ff3c817608747bec5330f3647814102019",
    "plastic_heredity/intervention_readback_common.py": "3e2f65029f3951cfd488a6965bdb0822db0884e739cabc19df0599eac413d8c9",
    "plastic_heredity/intervention_p2_readback_recovery.py": "e019d93b384d8cea90615d455d04d3382b4908bc8cc2b4775824b1bdb1f9eb28",
    "plastic_heredity/intervention_readback_recovery.py": "9a33ec3dd04b416739df9d4604674eb6d16e042434a98b0088ced1cc527fb531",
    "tests/test_intervention_p2_readback_recovery.py": "60a9492ce88eca15ab01fce1e5b3c8755faaec2574a097343b4f2301035c0e56",
}


def _source_hashes() -> dict[str, str]:
    return {name: sha256_file(REPOSITORY_ROOT / name) for name in SOURCE_FILES}


def _protocol(
    original_registration: dict[str, Any],
    original_registration_directory: Path,
    work: Path,
    failed_log: Path,
    intended_output: Path,
    checkpoint_record: dict[str, Any],
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "format": AMENDMENT_FORMAT,
        "status": "sealed_before_recovery_loaded_any_p2_checkpoint_outcome",
        "phase": "p2",
        "original_registration": {
            "id": original_registration["registration_id"],
            "path": str(original_registration_directory.resolve()),
            "checksum_manifest_sha256": sha256_file(
                original_registration_directory / "SHA256SUMS"
            ),
            "all_original_registered_source_hashes_current": True,
        },
        "failure": {
            "log_path": str(failed_log.resolve()),
            "log_sha256": sha256_file(failed_log),
            "exception": EXPECTED_FAILURE,
            "primary_futures_complete": 51_200,
            "replay_futures_complete": 51_200,
            "result_bundle_sealed": False,
            "p3_launched": False,
        },
        "static_diagnosis": (
            "readback omitted the generation path's derived, replay-dependent "
            "pilot_eligibility field before complete dictionary comparison"
        ),
        "only_repair": (
            "derive readback pilot_eligibility as readback eligibility_without_replay "
            "AND exact_replay before comparison"
        ),
        "serialization_alias": (
            "map historical __main__.PhaseBatch to the byte-identical registered "
            "intervention_replication.PhaseBatch class"
        ),
        "scientific_contract_changes": [],
        "checkpoint_record": checkpoint_record,
        "work_directory": str(work.resolve()),
        "intended_result_directory": str(intended_output.resolve()),
        "recovery_futures": 0,
        "checkpoint_outcomes_read_during_preparation": False,
        "mandatory_stop_after_recovery": True,
    }
    value["protocol_id"] = _canonical_digest(value)
    return value


def prepare_amendment(
    original_registration_directory: Path,
    work: Path,
    failed_log: Path,
    intended_output: Path,
    output_directory: Path,
) -> None:
    original_registration_directory = original_registration_directory.resolve()
    work = work.resolve()
    failed_log = failed_log.resolve()
    intended_output = intended_output.resolve()
    output_directory = output_directory.resolve()
    if output_directory.exists():
        raise FileExistsError(f"refusing to overwrite {output_directory}")
    if intended_output.exists():
        raise ValueError("a P2 result already exists; recovery is ineligible")
    registration = original.verify_registration(original_registration_directory)
    if registration["registration_id"] != EXPECTED_ORIGINAL_REGISTRATION_ID:
        raise ValueError("unexpected original intervention registration")
    if not failed_log.is_file() or EXPECTED_FAILURE not in failed_log.read_text(
        encoding="utf-8"
    ):
        raise ValueError("the registered P2 readback failure is absent")
    checkpoints = _require_completed_checkpoints(work)
    protocol = _protocol(
        registration,
        original_registration_directory,
        work,
        failed_log,
        intended_output,
        checkpoints,
    )
    with _atomic_destination(output_directory) as output:
        (output / "recovery_protocol.json").write_text(
            json.dumps(_json_ready(protocol), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        payload: dict[str, Any] = {
            "format": AMENDMENT_FORMAT,
            "status": "sealed_before_recovery_loaded_any_p2_checkpoint_outcome",
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": sha256_file(output / "recovery_protocol.json"),
            "source_hashes": _source_hashes(),
            "original_registration_id": registration["registration_id"],
            "original_registration_checksum_manifest_sha256": sha256_file(
                original_registration_directory / "SHA256SUMS"
            ),
            "failed_log_sha256": sha256_file(failed_log),
            "generation_checkpoint_aggregate_sha256": checkpoints["generate"][
                "checkpoint_digest"
            ]["aggregate_sha256"],
            "replay_checkpoint_aggregate_sha256": checkpoints["replay"][
                "checkpoint_digest"
            ]["aggregate_sha256"],
            "p2_checkpoint_outcomes_loaded": False,
        }
        payload["amendment_id"] = _canonical_digest(payload)
        (output / "amendment_registration.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_checksums(output)
    verify_amendment(output_directory)
    print(f"P2 readback amendment sealed: {payload['amendment_id']}", flush=True)


def verify_amendment(directory: Path) -> dict[str, Any]:
    directory = directory.resolve()
    verify_checksums(directory)
    payload = json.loads(
        (directory / "amendment_registration.json").read_text(encoding="utf-8")
    )
    identifier = payload.pop("amendment_id")
    if (
        payload.get("format") != AMENDMENT_FORMAT
        or payload.get("status")
        != "sealed_before_recovery_loaded_any_p2_checkpoint_outcome"
        or _canonical_digest(payload) != identifier
    ):
        raise ValueError("invalid P2 recovery amendment")
    payload["amendment_id"] = identifier
    current_source_hashes = _source_hashes()
    if payload["source_hashes"] != current_source_hashes and payload[
        "source_hashes"
    ] != SEALED_PRE_RELOCATION_SOURCE_HASHES:
        raise ValueError("P2 recovery source changed after sealing")
    protocol = json.loads(
        (directory / "recovery_protocol.json").read_text(encoding="utf-8")
    )
    archived_unsigned = dict(protocol)
    archived_protocol_id = archived_unsigned.pop("protocol_id")
    if _canonical_digest(archived_unsigned) != archived_protocol_id:
        raise ValueError("invalid archived P2 recovery protocol ID")
    original_directory = relocated_path(protocol["original_registration"]["path"])
    registration = original.verify_registration(original_directory)
    work = relocated_path(protocol["work_directory"])
    failed_log = relocated_path(protocol["failure"]["log_path"])
    intended_output = relocated_path(
        protocol["intended_result_directory"], require_exists=False
    )
    checkpoints = _require_completed_checkpoints(work)
    expected = _protocol(
        registration,
        original_directory,
        work,
        failed_log,
        intended_output,
        checkpoints,
    )
    if not protocols_equal_after_relocation(
        protocol, json.loads(json.dumps(_json_ready(expected)))
    ):
        raise ValueError("P2 recovery protocol changed")
    if (
        protocol["protocol_id"] != payload["protocol_id"]
        or sha256_file(directory / "recovery_protocol.json")
        != payload["protocol_sha256"]
        or sha256_file(failed_log) != payload["failed_log_sha256"]
    ):
        raise ValueError("P2 recovery provenance changed")
    return payload


def recover(amendment_directory: Path) -> None:
    amendment_directory = amendment_directory.resolve()
    amendment = verify_amendment(amendment_directory)
    protocol = json.loads(
        (amendment_directory / "recovery_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    recover_completed_pilot(
        phase="p2",
        amendment_id=amendment["amendment_id"],
        original_registration_directory=relocated_path(
            protocol["original_registration"]["path"]
        ),
        work=relocated_path(protocol["work_directory"]),
        output_directory=relocated_path(
            protocol["intended_result_directory"], require_exists=False
        ),
        registered_checkpoint_record=protocol["checkpoint_record"],
        failure_log_sha256=amendment["failed_log_sha256"],
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="P2 readback recovery")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument(
        "--original-registration",
        type=Path,
        default=Path("results_intervention_replication/registration"),
    )
    prepare.add_argument(
        "--work", type=Path, default=Path("results_intervention_replication/.p2_work")
    )
    prepare.add_argument(
        "--failed-log",
        type=Path,
        default=Path("results_intervention_replication/p2_cr3_run.log"),
    )
    prepare.add_argument(
        "--intended-output",
        type=Path,
        default=Path("results_intervention_replication/p2_cr3_physical_rule_pilot"),
    )
    prepare.add_argument(
        "--output",
        type=Path,
        default=Path("results_intervention_replication/p2_readback_amendment"),
    )
    verify = commands.add_parser("verify")
    verify.add_argument(
        "--amendment",
        type=Path,
        default=Path("results_intervention_replication/p2_readback_amendment"),
    )
    recover_command = commands.add_parser("recover")
    recover_command.add_argument(
        "--amendment",
        type=Path,
        default=Path("results_intervention_replication/p2_readback_amendment"),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    if arguments.command == "prepare":
        prepare_amendment(
            arguments.original_registration,
            arguments.work,
            arguments.failed_log,
            arguments.intended_output,
            arguments.output,
        )
    elif arguments.command == "verify":
        print(
            json.dumps(
                verify_amendment(arguments.amendment), indent=2, sort_keys=True
            )
        )
    elif arguments.command == "recover":
        recover(arguments.amendment)
    else:  # pragma: no cover
        raise AssertionError(arguments.command)


if __name__ == "__main__":
    main()
