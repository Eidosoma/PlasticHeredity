"""Command-line interface for the replication."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Optional, Sequence

from .config import ExperimentConfig, InterventionConfig
from .formulation_bridge import (
    register_formulation_bridge,
    run_formulation_bridge_pilot,
)
from .pipeline import run_replication, smoke_config
from .probe import ProbeConfig, run_probe


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aor-replicate",
        description="Independent clean-room replication of arXiv:2607.28250",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("run", "run or resume the paper-scale experiment"),
        ("smoke", "run a small end-to-end verification"),
    ):
        child = subparsers.add_parser(command, help=help_text)
        child.add_argument("--output", type=Path, default=Path("results") / command)
        child.add_argument("--runs", type=int, default=None)
        child.add_argument("--generations", type=int, default=None)
        child.add_argument("--seed", type=int, default=1729)
        child.add_argument(
            "--intervention-estimator",
            choices=("online_initial", "online_history", "matched_control"),
            default=None,
            help="reference distribution used to score molecule interventions",
        )
        child.add_argument(
            "--source-pdf",
            type=Path,
            default=Path("2607.28250v1.pdf"),
            help="user-supplied arXiv:2607.28250v1 PDF (not distributed here)",
        )
        child.add_argument("--skip-interventions", action="store_true")
        child.add_argument("--skip-forecast", action="store_true")
        child.add_argument("--skip-sensitivity", action="store_true")
        child.add_argument("--no-plots", action="store_true")
        child.add_argument("--overwrite", action="store_true")
    probe = subparsers.add_parser(
        "probe", help="genetically search under-specified method choices"
    )
    probe.add_argument("--output", type=Path, default=Path("results") / "probe")
    probe.add_argument("--population", type=int, default=24)
    probe.add_argument("--ga-generations", type=int, default=10)
    probe.add_argument("--calibration-runs", type=int, default=8)
    probe.add_argument("--holdout-runs", type=int, default=24)
    probe.add_argument("--calibration-seed", type=int, default=30_000)
    probe.add_argument("--holdout-seed", type=int, default=90_000)
    probe.add_argument("--ga-seed", type=int, default=2_718_281)
    probe.add_argument("--workers", type=int, default=1)
    probe.add_argument("--objective", choices=("control", "full"), default="control")
    probe.add_argument("--gard-generations", type=int, default=100)
    probe.add_argument("--max-trace-steps", type=int, default=5_000)
    probe.add_argument("--overwrite", action="store_true")
    bridge_registration = subparsers.add_parser(
        "bridge-register",
        help="validate and seal the frozen Arrivals formulation bridge",
    )
    bridge_registration.add_argument(
        "--output",
        type=Path,
        default=Path("results") / "formulation-bridge-registration",
    )
    bridge_pilot = subparsers.add_parser(
        "bridge-pilot",
        help="run the registered 12-seed observational formulation pilot",
    )
    bridge_pilot.add_argument(
        "--output",
        type=Path,
        default=Path("results") / "formulation-bridge-pilot12",
    )
    bridge_pilot.add_argument(
        "--registration",
        type=Path,
        default=Path("results") / "formulation-bridge-registration",
    )
    bridge_pilot.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "bridge-register":
        registration = register_formulation_bridge(args.output)
        print(
            f"formulation bridge registered as {registration['registration_id']} "
            f"in {args.output.resolve()}"
        )
        return 0
    if args.command == "bridge-pilot":
        result = run_formulation_bridge_pilot(
            args.output,
            args.registration,
            overwrite=args.overwrite,
        )
        print(
            f"formulation bridge results written to {args.output.resolve()}; "
            f"next action: {result['next_action']}"
        )
        return 0
    if args.command == "probe":
        probe = ProbeConfig(
            population_size=args.population,
            ga_generations=args.ga_generations,
            calibration_runs=args.calibration_runs,
            holdout_runs=args.holdout_runs,
            calibration_seed=args.calibration_seed,
            holdout_seed=args.holdout_seed,
            ga_seed=args.ga_seed,
            workers=args.workers,
            objective=args.objective,
            gard_generations=args.gard_generations,
            max_trace_steps=args.max_trace_steps,
        )
        run_probe(probe, args.output, overwrite=args.overwrite)
        print(f"probe results written to {args.output.resolve()}")
        return 0
    config = ExperimentConfig(base_seed=args.seed)
    if args.command == "smoke":
        config = smoke_config(config)
    if args.runs is not None:
        config = replace(config, runs=args.runs)
    if args.generations is not None:
        config = replace(config, gard=replace(config.gard, generations=args.generations))
    if args.intervention_estimator is not None:
        config = replace(
            config,
            intervention=InterventionConfig(estimator=args.intervention_estimator),
        )
    run_replication(
        config,
        args.output,
        include_interventions=not args.skip_interventions,
        include_forecast=not args.skip_forecast,
        include_sensitivity=not args.skip_sensitivity,
        make_plots=not args.no_plots,
        overwrite=args.overwrite,
        source_pdf=args.source_pdf,
    )
    print(f"results written to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
