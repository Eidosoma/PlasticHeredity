"""Command-line interface for the clean-room compact-carrier replication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from .acquisition import acquire
from .campaign import (
    audit_tests,
    register,
    report,
    run_confirmation,
    smoke,
    status,
    validate,
    verify_all,
)
from .contract import DEFAULT_ARTIFACTS, DEFAULT_LOCAL_INPUT, DEFAULT_SOURCE_ROOT
from .snapshot import prepare_snapshot


def _print(value: Any) -> None:
    print(json.dumps(value, sort_keys=True, indent=2, allow_nan=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fresh clean-room replication of the compact Rule-31649 carrier"
    )
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="snapshot allow-listed data/documents only")
    prepare.add_argument("--local-root", type=Path, default=DEFAULT_LOCAL_INPUT)
    prepare.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    commands.add_parser("acquire", help="generate and freeze the prospective donor bank")
    commands.add_parser("validate", help="validate models, acquisition, cohorts, and contract")
    commands.add_parser("audit-tests", help="run and record the complete package test suite")
    commands.add_parser("smoke", help="run the non-evidential engineering cohort smoke test")
    commands.add_parser("register", help="seal the design before confirmation outcomes")
    confirm = commands.add_parser("confirm", help="run or resume the 768 registered cells")
    confirm.add_argument("--workers", type=int, default=20)
    confirm.add_argument("--resume", action="store_true")
    confirm.add_argument("--authorize-confirmation", action="store_true")
    commands.add_parser("report", help="adjudicate all available sealed checkpoints")
    commands.add_parser("verify", help="verify seals and exactly recompute the result")
    commands.add_parser("status", help="show progress and ETA")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifacts = args.artifacts
    if args.command == "prepare":
        value = prepare_snapshot(
            artifacts_root=artifacts,
            local_root=args.local_root,
            source_root=args.source_root,
        )
    elif args.command == "acquire":
        value = acquire(artifacts)
    elif args.command == "validate":
        value = validate(artifacts)
    elif args.command == "audit-tests":
        value = audit_tests(artifacts)
    elif args.command == "smoke":
        value = smoke(artifacts)
    elif args.command == "register":
        value = register(artifacts)
    elif args.command == "confirm":
        value = run_confirmation(
            artifacts,
            workers=args.workers,
            resume=args.resume,
            authorize_confirmation=args.authorize_confirmation,
        )
    elif args.command == "report":
        value = report(artifacts)
    elif args.command == "verify":
        value = verify_all(artifacts)
    elif args.command == "status":
        value = status(artifacts)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    _print(value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
