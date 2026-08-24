"""Command-line interface for the clean-room replication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .campaign import (
    parity_check,
    register_stage1,
    register_stage2,
    run_stage1,
    run_stage2,
    stage1_report,
    stage2_report,
    status,
    validate_cleanroom,
    verify_all,
)
from .contract import DEFAULT_ARTIFACTS


def _print(value: Any) -> None:
    print(json.dumps(value, sort_keys=True, indent=2, allow_nan=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Independent Rule-31649 motif-channel replication"
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=DEFAULT_ARTIFACTS,
        help="local snapshot, registrations, checkpoints, and reports",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    snapshot = subparsers.add_parser(
        "snapshot", help="copy only allowlisted historical data/documents"
    )
    snapshot.add_argument("--source", type=Path, required=True)
    subparsers.add_parser("validate", help="validate firewall and fresh cohort capacity")
    subparsers.add_parser("parity", help="run non-evidential deterministic parity")
    subparsers.add_parser("register-stage1", help="seal Stage 1 before fresh outcomes")
    stage1 = subparsers.add_parser("stage1", help="run/resume registered Stage 1")
    stage1.add_argument(
        "--phase",
        choices=("calibration", "screen", "validation", "all"),
        default="all",
    )
    stage1.add_argument("--workers", type=int, default=8)
    subparsers.add_parser("stage1-report", help="adjudicate and report Stage 1")
    subparsers.add_parser(
        "register-stage2",
        help="explicitly seal Stage 2 after a reviewed robust Stage-1 pass",
    )
    stage2 = subparsers.add_parser("stage2", help="run/resume registered Stage 2")
    stage2.add_argument("--workers", type=int, default=8)
    subparsers.add_parser("stage2-report", help="adjudicate and report Stage 2")
    subparsers.add_parser("status", help="show checkpoint progress and gates")
    subparsers.add_parser("verify", help="verify snapshot, registrations, and checkpoints")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    artifacts = args.artifacts
    if args.command == "snapshot":
        from .snapshot import build_snapshot

        value = build_snapshot(args.source, artifacts)
    elif args.command == "validate":
        value = validate_cleanroom(artifacts)
    elif args.command == "parity":
        value = parity_check(artifacts)
    elif args.command == "register-stage1":
        value = register_stage1(artifacts)
    elif args.command == "stage1":
        value = run_stage1(artifacts, phase=args.phase, workers=args.workers)
    elif args.command == "stage1-report":
        value = stage1_report(artifacts)
    elif args.command == "register-stage2":
        value = register_stage2(artifacts)
    elif args.command == "stage2":
        value = run_stage2(artifacts, workers=args.workers)
    elif args.command == "stage2-report":
        value = stage2_report(artifacts)
    elif args.command == "status":
        value = status(artifacts)
    elif args.command == "verify":
        value = verify_all(artifacts)
    else:  # pragma: no cover
        parser.error(f"unhandled command: {args.command}")
    _print(value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
