from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from .campaign import run_benchmark, run_campaign, verify_run
from .config import load_registration
from .validation import validate
from .worker import run_worker


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wagner-memory-v2")
    commands = parser.add_subparsers(dest="command", required=True)
    validate_parser = commands.add_parser("validate")
    validate_parser.add_argument("--profile", choices=("smoke", "quick", "full"), default="smoke")
    for name in ("benchmark", "campaign"):
        command = commands.add_parser(name)
        command.add_argument("--run", type=Path, required=True)
        command.add_argument("--profile", choices=("smoke", "quick", "full"), default="full")
    status = commands.add_parser("status")
    status.add_argument("--run", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--run", type=Path, required=True)
    worker = commands.add_parser("worker", help=argparse.SUPPRESS)
    worker.add_argument("--run", type=Path, required=True)
    worker.add_argument("--stage", required=True)
    worker.add_argument("--worker-index", type=int, required=True)
    worker.add_argument("--worker-count", type=int, required=True)
    worker.add_argument("--benchmark", action="store_true")
    return parser


def status_snapshot(run_dir: Path) -> dict[str, Any]:
    status_path = run_dir / "STATUS.json"
    if not status_path.exists():
        return {"format": "wagner-memory-status-v2", "state": "not_started", "run": str(run_dir)}
    status = json.loads(status_path.read_text())
    progress: list[dict[str, Any]] = []
    phase = str(status.get("phase", ""))
    if phase == "benchmark":
        paths = sorted((run_dir / "benchmark").glob("worker-*.progress.json"))
    elif phase:
        paths = sorted((run_dir / "stages" / phase).glob("worker-*.progress.json"))
    else:
        paths = []
    for path in paths:
        try:
            progress.append(json.loads(path.read_text()))
        except (OSError, json.JSONDecodeError):
            continue
    status["worker_progress"] = progress
    return status


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "validate":
        result = validate(load_registration(args.profile))
        print(json.dumps(result, indent=2, sort_keys=True))
        if not result["valid"]:
            raise SystemExit(1)
        return
    if args.command == "status":
        print(json.dumps(status_snapshot(args.run), indent=2, sort_keys=True))
        return
    if args.command == "verify":
        result = verify_run(args.run)
        print(json.dumps(result, indent=2, sort_keys=True))
        if not result["verified"]:
            raise SystemExit(1)
        return
    if args.command == "worker":
        result = run_worker(
            args.run,
            args.stage,
            args.worker_index,
            args.worker_count,
            args.benchmark,
        )
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
