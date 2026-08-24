from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .campaign import run_benchmark, run_campaign, seal_run, verify_run
from .config import load_registration
from .validation import validate
from .worker import run_worker


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wagner-memory")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--profile", choices=("smoke", "quick", "full"), default="smoke")
    for name in ("benchmark", "campaign"):
        command = subparsers.add_parser(name)
        command.add_argument("--run", type=Path, required=True)
        command.add_argument("--profile", choices=("smoke", "quick", "full"), default="full")
    status = subparsers.add_parser("status")
    status.add_argument("--run", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--run", type=Path, required=True)
    worker = subparsers.add_parser("worker", help=argparse.SUPPRESS)
    worker.add_argument("--run", type=Path, required=True)
    worker.add_argument("--profile", choices=("smoke", "quick", "full"), required=True)
    worker.add_argument("--stage", required=True)
    worker.add_argument("--worker-index", type=int, required=True)
    worker.add_argument("--worker-count", type=int, default=2)
    worker.add_argument("--benchmark", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "validate":
        result = validate(load_registration(args.profile))
        print(json.dumps(result, indent=2, sort_keys=True))
        if not result["valid"]:
            raise SystemExit(1)
        return
    if args.command == "status":
        status_path = args.run / "STATUS.json"
        if status_path.exists():
            print(status_path.read_text(), end="")
        else:
            print(json.dumps({"state": "not_started", "run": str(args.run)}, indent=2))
        return
    if args.command == "verify":
        result = verify_run(args.run)
        if result["verified"]:
            seal_run(args.run)
        print(json.dumps(result, indent=2, sort_keys=True))
        if not result["verified"]:
            raise SystemExit(1)
        return
    if args.command == "worker":
        result = run_worker(args.run, args.profile, args.stage, args.worker_index, args.worker_count, args.benchmark)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    registration = load_registration(args.profile)
    if args.command == "benchmark":
        result = run_benchmark(args.run, registration)
        print(json.dumps(result, indent=2, sort_keys=True))
        if not result["admitted"]:
            raise SystemExit(1)
        return
    if args.command == "campaign":
        result = run_campaign(args.run, registration)
        print(json.dumps({"overall_verdict": result["overall_verdict"]}, indent=2))
        return
    raise AssertionError(args.command)


if __name__ == "__main__":
    main(sys.argv[1:])

