"""Command-line interface for the clean-room lineage-renewal replication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .campaign import (
    parity_check,
    register,
    report,
    run_confirmation,
    run_quarantine,
    status,
    validate_cleanroom,
    verify_all,
)
from .contract import DEFAULT_ARTIFACTS


def _print(value: Any) -> None:
    print(json.dumps(value, sort_keys=True, indent=2, allow_nan=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Independent Rule-31649 lineage-renewal replication"
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=DEFAULT_ARTIFACTS,
        help="local snapshot, registration, checkpoints, and reports",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    snapshot = commands.add_parser(
        "snapshot", help="copy only explicitly allowlisted data and documents"
    )
    snapshot.add_argument("--source", type=Path, required=True)
    commands.add_parser("validate", help="validate firewall and fully fresh cohorts")
    commands.add_parser("parity", help="check local deterministic primitives")
    commands.add_parser("register", help="seal all tests before fresh lineage outcomes")
    quarantine = commands.add_parser(
        "quarantine", help="run the non-evidential two-pair engineering check"
    )
    quarantine.add_argument("--workers", type=int, default=1)
    confirmation = commands.add_parser(
        "confirm", help="explicitly run or resume the sealed 96-pair confirmation"
    )
    confirmation.add_argument("--workers", type=int, default=8)
    commands.add_parser("report", help="adjudicate and seal available confirmation data")
    commands.add_parser("status", help="show checkpoint progress and ETA")
    commands.add_parser("verify", help="verify all hashes and checkpoints")
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
