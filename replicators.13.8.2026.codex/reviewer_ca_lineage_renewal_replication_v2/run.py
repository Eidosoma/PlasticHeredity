"""CLI for the corrected v2 lineage-renewal replication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .campaign import (
    audit_tests,
    parity_check,
    register,
    report,
    run_confirmation,
    run_quarantine,
    status,
    validate_cleanroom,
    verify_all,
)
from .contract import DEFAULT_ARTIFACTS, DEFAULT_UPSTREAM
from .snapshot import prepare_snapshot


def _print(value: Any) -> None:
    print(json.dumps(value, sort_keys=True, indent=2, allow_nan=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Corrected independent Rule-31649 lineage-renewal replication"
    )
    parser.add_argument(
        "--artifacts", type=Path, default=DEFAULT_ARTIFACTS, help="v2 artifact root"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare", help="snapshot only already-frozen local data/doc artifacts"
    )
    prepare.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    commands.add_parser("validate", help="validate the recovered spec and untouched cohort")
    commands.add_parser("parity", help="rerun deterministic non-evidential parity checks")
    commands.add_parser("audit-tests", help="run and record the complete v2 test suite")
    commands.add_parser("register", help="seal code, tests, inputs, cohort, and endpoints")
    quarantine = commands.add_parser(
        "quarantine", help="run two already-exposed engineering pairs"
    )
    quarantine.add_argument("--workers", type=int, default=1)
    confirmation = commands.add_parser(
        "confirm", help="explicitly run or resume the sealed 92-pair confirmation"
    )
    confirmation.add_argument("--workers", type=int, default=8)
    commands.add_parser("report", help="adjudicate the available registered checkpoints")
    commands.add_parser("status", help="show checkpoint progress and ETA")
    commands.add_parser("verify", help="verify input, design, checkpoint, and report seals")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    artifacts = args.artifacts
    if args.command == "prepare":
        value = prepare_snapshot(args.upstream, artifacts)
    elif args.command == "validate":
        value = validate_cleanroom(artifacts)
    elif args.command == "parity":
        value = parity_check(artifacts)
    elif args.command == "audit-tests":
        value = audit_tests(artifacts)
    elif args.command == "register":
        value = register(artifacts)
    elif args.command == "quarantine":
        value = run_quarantine(artifacts, workers=args.workers)
    elif args.command == "confirm":
        value = run_confirmation(artifacts, workers=args.workers)
    elif args.command == "report":
        value = report(artifacts)
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
