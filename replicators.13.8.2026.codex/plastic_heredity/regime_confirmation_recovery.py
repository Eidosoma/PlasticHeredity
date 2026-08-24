"""Audited recovery for the regime-confirmation CSV round-trip failure.

The prospective design and development model bundles hash
``regime_confirmation.py``.  Changing that sealed source after confirmation
shooting would invalidate both bundles.  This narrowly scoped runner therefore
leaves every registered scientific source byte-for-byte unchanged and applies
one operational correction while artifact readback is active: pandas parses
CSV floats with ``float_precision="round_trip"``.

The correction affects only reconstruction of already-written prediction
floats.  It does not change matrices, trajectories, futures, endpoints,
features, fitted models, predictions, seeds, or inference rules.  An amendment
bundle records the failed log and hashes this runner before a recovery run can
start.  The final confirmation manifest also records that amendment.
"""

from __future__ import annotations

import argparse
import json
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from . import regime_confirmation as regime
from .mechanistic import (
    _atomic_destination,
    _canonical_digest,
    sha256_file,
    verify_checksums,
    write_checksums,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AMENDMENT_FORMAT = "plastic-heredity-regime-operational-amendment-v1"
EXPECTED_REGIME_SOURCE_SHA256 = (
    "034c70adf4571408ad2e0215e8a7bddf4aa03c39199f2e8c080ac6e50ad60c0a"
)
FAILURE_SIGNATURE = (
    "metrics.endpoints.primary_all8.candidates.02.models.beta_only."
    "centered_spearman[0]"
)


def _runner_relative_path() -> str:
    return str(Path(__file__).resolve().relative_to(REPOSITORY_ROOT))


def _runner_sha256() -> str:
    return sha256_file(Path(__file__).resolve())


def _verify_original_source_unchanged() -> None:
    source = REPOSITORY_ROOT / "plastic_heredity/regime_confirmation.py"
    observed = sha256_file(source)
    if observed != EXPECTED_REGIME_SOURCE_SHA256:
        raise ValueError(
            "sealed regime confirmation source changed before operational recovery"
        )


def register_amendment(
    registration: Path,
    development: Path,
    failure_log: Path,
    output_directory: Path,
) -> None:
    """Seal the parser-only amendment before restarting confirmation."""

    output_directory = output_directory.resolve()
    if output_directory.exists():
        raise FileExistsError(f"refusing to overwrite {output_directory}")
    _verify_original_source_unchanged()
    design = regime.verify_design(registration.resolve())
    model_seal = regime.verify_development(development.resolve())
    if model_seal["design"]["registration_id"] != design["registration_id"]:
        raise ValueError("development models reference a different design")
    failure_log = failure_log.resolve()
    failure_text = failure_log.read_text(encoding="utf-8")
    required_markers = (
        "[confirm-generate] states 2000/2000",
        "[confirm-replay] states 2000/2000",
        "[confirm 6/9] Writing complete confirmation artifacts",
        FAILURE_SIGNATURE,
    )
    missing = [marker for marker in required_markers if marker not in failure_text]
    if missing:
        raise ValueError(f"failure log lacks required recovery markers: {missing}")
    if "[confirm 7/9]" in failure_text:
        raise ValueError("failure log indicates checksumming had already begun")

    with _atomic_destination(output_directory) as output:
        retained_log = output / "failed_confirmation.log"
        shutil.copyfile(failure_log, retained_log)
        payload: dict[str, Any] = {
            "format": AMENDMENT_FORMAT,
            "status": "sealed_after_operational_failure_before_recovery_rerun",
            "scope": "CSV prediction-float readback only",
            "correction": {
                "reader": "pandas.read_csv",
                "parameter": "float_precision",
                "value": "round_trip",
                "reason": (
                    "default CSV parsing did not reproduce binary64 prediction "
                    "values exactly; centered ranks of theoretically static beta-only "
                    "predictions were therefore unstable"
                ),
            },
            "scientific_contract_changes": [],
            "unchanged": [
                "registered endpoints and thresholds",
                "development models and coefficients",
                "matrix, trajectory, future, bootstrap, and randomization seeds",
                "simulator and feature code",
                "confirmation cohort and inference gates",
            ],
            "failed_run": {
                "all_confirmation_futures_completed": True,
                "all_confirmation_replays_completed": True,
                "failed_before_checksums_and_atomic_seal": True,
                "no_confirmation_bundle_sealed": True,
                "error_signature": FAILURE_SIGNATURE,
                "retained_log": retained_log.name,
                "retained_log_sha256": sha256_file(retained_log),
            },
            "design_registration_id": design["registration_id"],
            "design_checksum_digest": sha256_file(
                registration.resolve() / "SHA256SUMS"
            ),
            "model_seal_id": model_seal["model_seal_id"],
            "development_checksum_digest": sha256_file(
                development.resolve() / "SHA256SUMS"
            ),
            "sealed_regime_source_sha256": EXPECTED_REGIME_SOURCE_SHA256,
            "recovery_runner": _runner_relative_path(),
            "recovery_runner_sha256": _runner_sha256(),
        }
        payload["amendment_id"] = _canonical_digest(payload)
        (output / "amendment.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        write_checksums(output)
    print(f"Operational recovery amendment sealed at {output_directory}", flush=True)


def verify_amendment(
    amendment_directory: Path,
    registration: Path,
    development: Path,
) -> dict[str, Any]:
    amendment_directory = amendment_directory.resolve()
    verify_checksums(amendment_directory)
    payload = json.loads(
        (amendment_directory / "amendment.json").read_text(encoding="utf-8")
    )
    if payload.get("format") != AMENDMENT_FORMAT:
        raise ValueError("unsupported regime operational amendment")
    amendment_id = payload.pop("amendment_id")
    if _canonical_digest(payload) != amendment_id:
        raise ValueError("regime operational amendment identifier mismatch")
    payload["amendment_id"] = amendment_id
    _verify_original_source_unchanged()
    if payload["recovery_runner"] != _runner_relative_path():
        raise ValueError("operational amendment names a different recovery runner")
    if payload["recovery_runner_sha256"] != _runner_sha256():
        raise ValueError("recovery runner changed after amendment sealing")
    retained_log = amendment_directory / payload["failed_run"]["retained_log"]
    if payload["failed_run"]["retained_log_sha256"] != sha256_file(retained_log):
        raise ValueError("retained failure log changed after amendment sealing")

    design = regime.verify_design(registration.resolve())
    model_seal = regime.verify_development(development.resolve())
    if payload["design_registration_id"] != design["registration_id"]:
        raise ValueError("operational amendment references a different design")
    if payload["model_seal_id"] != model_seal["model_seal_id"]:
        raise ValueError("operational amendment references different models")
    if payload["design_checksum_digest"] != sha256_file(
        registration.resolve() / "SHA256SUMS"
    ):
        raise ValueError("design checksum seal changed after amendment")
    if payload["development_checksum_digest"] != sha256_file(
        development.resolve() / "SHA256SUMS"
    ):
        raise ValueError("development checksum seal changed after amendment")
    return payload


@contextmanager
def round_trip_csv_readback() -> Iterator[None]:
    """Make pandas reconstruct CSV binary64 values exactly within this scope."""

    original = regime.pd.read_csv

    def read_csv(*args: Any, **kwargs: Any):
        kwargs.setdefault("float_precision", "round_trip")
        return original(*args, **kwargs)

    regime.pd.read_csv = read_csv
    try:
        yield
    finally:
        regime.pd.read_csv = original


@contextmanager
def recovery_runtime_manifest(
    amendment_directory: Path, payload: dict[str, Any]
) -> Iterator[None]:
    """Record the operational amendment inside the sealed result manifest."""

    original = regime._runtime_manifest

    def runtime_manifest() -> dict[str, Any]:
        manifest = original()
        manifest["operational_recovery"] = {
            "format": AMENDMENT_FORMAT,
            "amendment_id": payload["amendment_id"],
            "amendment_json_sha256": sha256_file(
                amendment_directory.resolve() / "amendment.json"
            ),
            "amendment_checksum_digest": sha256_file(
                amendment_directory.resolve() / "SHA256SUMS"
            ),
            "recovery_runner": _runner_relative_path(),
            "recovery_runner_sha256": _runner_sha256(),
            "correction": payload["correction"],
        }
        return manifest

    regime._runtime_manifest = runtime_manifest
    try:
        yield
    finally:
        regime._runtime_manifest = original


def run_recovery_confirmation(
    amendment: Path,
    registration: Path,
    development: Path,
    output: Path,
    workers: int,
) -> None:
    payload = verify_amendment(amendment, registration, development)
    with round_trip_csv_readback(), recovery_runtime_manifest(amendment, payload):
        regime.run_confirmation(registration, development, output, workers)
    verify_checksums(output.resolve())
    manifest = json.loads((output.resolve() / "manifest.json").read_text())
    recovery = manifest.get("runtime", {}).get("operational_recovery", {})
    if recovery.get("amendment_id") != payload["amendment_id"]:
        raise ValueError("sealed confirmation omitted the operational amendment")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audited recovery for regime confirmation CSV readback"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    register = commands.add_parser(
        "register-amendment", help="seal the parser-only operational correction"
    )
    register.add_argument(
        "--registration",
        type=Path,
        default=Path("results/regime_design_registration"),
    )
    register.add_argument(
        "--development", type=Path, default=Path("results/regime_development")
    )
    register.add_argument(
        "--failure-log",
        type=Path,
        default=Path("/tmp/plastic_heredity_regime_campaign.20260813.log"),
    )
    register.add_argument(
        "--output",
        type=Path,
        default=Path("results/regime_confirmation_roundtrip_amendment"),
    )
    confirm = commands.add_parser(
        "confirm", help="rerun confirmation under the sealed correction"
    )
    confirm.add_argument(
        "--amendment",
        type=Path,
        default=Path("results/regime_confirmation_roundtrip_amendment"),
    )
    confirm.add_argument(
        "--registration",
        type=Path,
        default=Path("results/regime_design_registration"),
    )
    confirm.add_argument(
        "--development", type=Path, default=Path("results/regime_development")
    )
    confirm.add_argument(
        "--output", type=Path, default=Path("results/regime_confirmation")
    )
    confirm.add_argument("--workers", type=int, default=1)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "register-amendment":
        register_amendment(
            args.registration, args.development, args.failure_log, args.output
        )
    else:
        run_recovery_confirmation(
            args.amendment,
            args.registration,
            args.development,
            args.output,
            args.workers,
        )


if __name__ == "__main__":
    main()
