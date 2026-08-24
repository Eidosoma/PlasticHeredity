from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Sequence

from .analysis import analyze_predictor, analyze_primary
from .campaign import benchmark, run_campaign
from .experiment import run_primary
from .predictor import run_predictor_cohort
from .protocol import load_protocol
from .verification import replay_primary, validate_environment, verify_run


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wagner-cleanroom")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate", help="validate dependencies, protocols, and clean-room boundary")

    benchmark_parser = sub.add_parser("benchmark", help="run the discarded admission benchmark")
    benchmark_parser.add_argument("--output", type=Path, required=True)
    benchmark_parser.add_argument("--workers", type=int, default=12)
    benchmark_parser.add_argument("--profile", choices=("full", "smoke"), default="full")

    primary_parser = sub.add_parser("run-primary", help="run or resume the exact-state replication")
    primary_parser.add_argument("--output", type=Path, required=True)
    primary_parser.add_argument("--workers", type=int, default=12)
    primary_parser.add_argument("--profile", choices=("full", "smoke"), default="full")

    predictor_parser = sub.add_parser("run-predictor", help="run or resume both predictor cohorts")
    predictor_parser.add_argument("--output", type=Path, required=True)
    predictor_parser.add_argument("--workers", type=int, default=12)
    predictor_parser.add_argument("--profile", choices=("full", "smoke"), default="full")

    campaign_parser = sub.add_parser("campaign", help="run the admitted, wall-limited campaign")
    campaign_parser.add_argument("--output", type=Path, required=True)
    campaign_parser.add_argument("--workers", type=int, default=12)
    campaign_parser.add_argument("--profile", choices=("full", "smoke"), default="full")

    analysis_parser = sub.add_parser("analyze", help="analyze completed campaign shards")
    analysis_parser.add_argument("run", type=Path)
    analysis_parser.add_argument("--profile", choices=("full", "smoke"), default="full")

    verify_parser = sub.add_parser("verify", help="verify registrations, completion, and checksums")
    verify_parser.add_argument("run", type=Path)

    replay_parser = sub.add_parser("replay", help="exactly replay one retained primary future")
    replay_parser.add_argument("run", type=Path)
    replay_parser.add_argument("future_id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    args = build_parser().parse_args(argv)
    if args.command == "validate":
        result = validate_environment()
        _print(result)
        return 0 if result["ok"] else 1
    if args.command == "benchmark":
        result = benchmark(args.output, args.workers, args.profile)
    elif args.command == "run-primary":
        protocol = load_protocol("primary", args.profile)
        result = run_primary(args.output, protocol, args.workers)
    elif args.command == "run-predictor":
        protocol = load_protocol("predictor", args.profile)
        development = run_predictor_cohort(args.output / "development", protocol, "development", args.workers)
        evaluation = run_predictor_cohort(args.output / "evaluation", protocol, "evaluation", args.workers)
        result = {"development": development, "evaluation": evaluation}
        if development["complete"] and evaluation["complete"]:
            result["analysis"] = analyze_predictor(args.output, protocol)
    elif args.command == "campaign":
        result = run_campaign(args.output, args.workers, args.profile)
    elif args.command == "analyze":
        primary = analyze_primary(args.run / "primary", load_protocol("primary", args.profile))
        predictor = analyze_predictor(args.run / "predictor", load_protocol("predictor", args.profile))
        result = {"primary": primary, "predictor": predictor}
    elif args.command == "verify":
        result = verify_run(args.run)
    elif args.command == "replay":
        result = replay_primary(args.run, args.future_id)
    else:
        raise AssertionError(args.command)
    _print(result)
    return 0 if result.get("ok", result.get("complete", True)) else 1
