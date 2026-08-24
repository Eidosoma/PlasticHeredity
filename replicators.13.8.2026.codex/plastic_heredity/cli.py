from __future__ import annotations

import argparse
from pathlib import Path

from .config import ExperimentConfig
from .experiment import run_replication
from .plotting import create_figures
from .reporting import write_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clean-room replication of the GARD plastic-heredity process-risk result"
    )
    parser.add_argument(
        "--profile",
        choices=("quick", "full", "scaled5"),
        default="quick",
        help=(
            "quick is a smoke test; full uses the supplied confirmation design; "
            "scaled5 uses five times as many independent matrices"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/quick"),
        help="artifact directory",
    )
    parser.add_argument("--workers", type=int, default=None)
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    if arguments.profile == "quick":
        experiment = ExperimentConfig.quick()
    elif arguments.profile == "scaled5":
        experiment = ExperimentConfig.scaled5()
    else:
        experiment = ExperimentConfig()
    artifacts = run_replication(experiment, arguments.output, arguments.workers)
    create_figures(
        artifacts.metrics,
        artifacts.process_summary,
        artifacts.state_table,
        arguments.output,
    )
    write_report(
        arguments.output,
        artifacts.metrics,
        artifacts.process_summary,
        artifacts.comparison_table,
        artifacts.replay_exact,
    )
    print(f"Replication artifacts written to {arguments.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
