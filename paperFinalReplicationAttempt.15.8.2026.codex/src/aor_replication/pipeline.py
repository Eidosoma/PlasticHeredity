"""Checkpointed end-to-end replication pipeline."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib
import networkx
import numpy as np
import pandas as pd
import scipy
import seaborn
import sklearn
import statsmodels

from .analysis import (
    AnalyzedRun,
    analyze_run,
    intervention_generation_trends,
    intervention_table,
    intervention_tests,
    records_frame,
    summarize_control_runs,
    spike_correlations,
)
from .config import ExperimentConfig
from .comparators import established_metric_correlations, established_metric_values
from .forecast import (
    forecast_tests,
    run_forecast_experiment,
    run_forecast_threshold_sensitivity,
)
from .gard import RunTrace, simulate_gard
from .interventions import PhiDirectedIntervention, OnlinePhiDirectedIntervention
from .plots import (
    plot_figure2,
    plot_figure3,
    plot_figure4,
    plot_figure5,
    plot_figure6,
    plot_sensitivity,
)
from .sensitivity import (
    DEFAULT_THRESHOLDS,
    causal_estimator_sensitivity,
    detector_threshold_sensitivity,
    gard_tau_sensitivity,
    intervention_label_sensitivity,
)
from .storage import load_trace, save_analysis, save_trace, write_json


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_provenance(source_pdf: Optional[Path] = None) -> Dict[str, Any]:
    """Collect environment and source hashes without running the experiment."""

    project_root = Path(__file__).resolve().parents[2]
    source_files = [project_root / "pyproject.toml", project_root / "README.md"]
    for relative in ("src/aor_replication", "tests", "docs"):
        source_files.extend(sorted((project_root / relative).glob("*")))
    code_hashes = {
        str(path.relative_to(project_root)): _sha256(path)
        for path in source_files
        if path.is_file() and path.suffix in {".py", ".toml", ".md"}
    }
    provenance: Dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy_default_rng": "PCG64",
        "clean_room_boundary": (
            "No source code by the preprint authors or their older projects "
            "was inspected or reused."
        ),
        "packages": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
            "networkx": networkx.__version__,
            "scikit_learn": sklearn.__version__,
            "statsmodels": statsmodels.__version__,
            "matplotlib": matplotlib.__version__,
            "seaborn": seaborn.__version__,
        },
        "source_sha256": code_hashes,
    }
    if source_pdf is not None and source_pdf.exists():
        provenance["source_pdf"] = {
            "path": str(source_pdf.resolve()),
            "sha256": _sha256(source_pdf),
        }
    return provenance


def _trace_path(output: Path, treatment: str, run_index: int) -> Path:
    return output / "traces" / treatment / f"run-{run_index:03d}.npz"


def _analysis_path(output: Path, treatment: str, run_index: int) -> Path:
    return output / "analysis" / treatment / f"run-{run_index:03d}.npz"


def _get_or_simulate_control(
    config: ExperimentConfig,
    output: Path,
    run_index: int,
    seed: int,
    overwrite: bool,
) -> RunTrace:
    path = _trace_path(output, "control", run_index)
    if path.exists() and not overwrite:
        trace = load_trace(path)
        trace.validate(config.gard)
        return trace
    trace = simulate_gard(config.gard, seed)
    save_trace(path, trace)
    return trace


def _get_or_simulate_treatment(
    config: ExperimentConfig,
    output: Path,
    run_index: int,
    seed: int,
    treatment: str,
    control: AnalyzedRun,
    overwrite: bool,
) -> RunTrace:
    path = _trace_path(output, treatment, run_index)
    if path.exists() and not overwrite:
        trace = load_trace(path)
        trace.validate(config.gard)
        return trace
    if config.intervention.estimator in {"online_initial", "online_history"}:
        policy = OnlinePhiDirectedIntervention(
            config=config.causal,
            direction=treatment,  # type: ignore[arg-type]
            max_size=config.gard.max_size,
            refit_each_generation=(
                config.intervention.estimator == "online_history"
            ),
        )
    else:
        policy = PhiDirectedIntervention(
            reference=control.causal,
            direction=treatment,  # type: ignore[arg-type]
            max_size=config.gard.max_size,
        )
    trace = simulate_gard(
        config.gard,
        seed,
        beta=control.trace.beta,
        intervention=policy,
    )
    save_trace(path, trace)
    return trace


def _write_readable_summary(
    path: Path,
    config: ExperimentConfig,
    control_summary: Dict[str, Any],
    table: Optional[pd.DataFrame],
    forecast: Optional[pd.DataFrame],
    detector_sensitivity: Optional[pd.DataFrame],
) -> None:
    lines = [
        "# Replication run summary",
        "",
        f"Completed {control_summary['run_count']} matched control runs with base seed {config.base_seed}.",
        "",
        "## Control results",
        "",
        f"- Positive Phi-r/self-replication correlations: {control_summary['spearman']['positive_runs']}/{control_summary['run_count']}.",
        f"- Evaluable Phi-r/self-replication correlations: {control_summary['spearman']['evaluable_runs']}/{control_summary['run_count']}.",
        f"- Positive and individually significant correlations: {control_summary['spearman']['positive_significant_runs']}/{control_summary['run_count']}.",
        f"- Runs with higher mean Phi-r during self-replication: {control_summary['replicating_phi_higher_runs']}/{control_summary['run_count']}.",
        f"- Runs rejecting white noise by Ljung-Box: {control_summary['ljung_box_significant_runs']}/{control_summary['run_count']}.",
        f"- Runs retaining Ljung-Box significance after differencing: {control_summary['differenced_ljung_box_significant_runs']}/{control_summary['run_count']}.",
        "",
    ]
    if table is not None:
        lines.extend(
            [
                "## Intervention outcomes",
                "",
                "```text",
                table.to_string(index=False),
                "```",
                "",
            ]
        )
    if forecast is not None:
        means = forecast.groupby("model", sort=False).accuracy.mean().reset_index()
        lines.extend(
            [
                "## Forecast accuracy",
                "",
                "```text",
                means.to_string(index=False),
                "```",
                "",
            ]
        )
    if detector_sensitivity is not None:
        primary = detector_sensitivity[
            (detector_sensitivity.reference_states == config.replicator.reference_states)
            & np.isclose(
                detector_sensitivity.similarity_threshold,
                config.replicator.similarity_threshold,
            )
        ].iloc[0]
        lines.extend(
            [
                "## Under-specification sensitivity",
                "",
                f"At the registered standard similarity cutoff ({config.replicator.similarity_threshold:.2f}), mean reconstructed self-replication probability was {primary.probability_mean:.1%}; the preprint reports 88% for controls.",
                "The complete cutoff, causal-estimator, and Poisson-time-scale sweeps are in `sensitivity/`. The primary analysis was not retuned after seeing this discrepancy.",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation guardrail",
            "",
            "These are outputs of the documented clean-room reconstruction. Numerical disagreement with the preprint must be interpreted together with the under-specification ledger in `docs/REPLICATION_SPEC.md`.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        stream.write("\n".join(lines))


def run_replication(
    config: ExperimentConfig,
    output: Path,
    *,
    include_interventions: bool = True,
    include_forecast: bool = True,
    include_sensitivity: bool = True,
    make_plots: bool = True,
    overwrite: bool = False,
    source_pdf: Optional[Path] = None,
) -> List[AnalyzedRun]:
    """Run or resume the full matched-seed experiment."""

    config.validate()
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "config.json"
    if config_path.exists() and not overwrite:
        with config_path.open("r", encoding="utf-8") as stream:
            previous_config = json.load(stream)
        if previous_config != config.to_dict():
            raise RuntimeError(
                "output directory contains checkpoints from a different "
                "configuration; choose a new output path or pass --overwrite"
            )
    write_json(config_path, config.to_dict())
    provenance = collect_provenance(source_pdf)
    write_json(output / "provenance.json", provenance)

    all_runs: List[AnalyzedRun] = []
    controls: List[AnalyzedRun] = []
    for run_index in range(config.runs):
        seed = config.base_seed + run_index
        control_trace = _get_or_simulate_control(
            config, output, run_index, seed, overwrite
        )
        control = analyze_run(
            control_trace,
            run_index=run_index,
            treatment="control",
            causal_config=config.causal,
            replicator_config=config.replicator,
        )
        save_analysis(_analysis_path(output, "control", run_index), control)
        controls.append(control)
        all_runs.append(control)

        if include_interventions:
            for treatment in ("max", "min"):
                treatment_trace = _get_or_simulate_treatment(
                    config,
                    output,
                    run_index,
                    seed,
                    treatment,
                    control,
                    overwrite,
                )
                analyzed = analyze_run(
                    treatment_trace,
                    run_index=run_index,
                    treatment=treatment,
                    causal_config=config.causal,
                    replicator_config=config.replicator,
                )
                save_analysis(_analysis_path(output, treatment, run_index), analyzed)
                all_runs.append(analyzed)
        print(f"completed run {run_index + 1}/{config.runs}", flush=True)

    metrics = records_frame(all_runs)
    metrics.to_csv(output / "run_metrics.csv", index=False)
    control_summary = summarize_control_runs(controls)
    write_json(output / "control_summary.json", control_summary)
    write_json(output / "spike_correlations.json", spike_correlations(controls))
    comparator_values = established_metric_values(controls)
    comparator_values.to_csv(output / "established_metric_values.csv", index=False)
    established_metric_correlations(comparator_values).to_csv(
        output / "established_metric_correlations.csv", index=False
    )

    detector_summary: Optional[pd.DataFrame] = None
    causal_summary: Optional[pd.DataFrame] = None
    tau_summary: Optional[pd.DataFrame] = None
    if include_sensitivity:
        sensitivity_dir = output / "sensitivity"
        sensitivity_dir.mkdir(parents=True, exist_ok=True)
        detector_summary, detector_detail = detector_threshold_sensitivity(
            controls, config.replicator
        )
        detector_summary.to_csv(
            sensitivity_dir / "replicator_threshold_summary.csv", index=False
        )
        detector_detail.to_csv(
            sensitivity_dir / "replicator_threshold_runs.csv", index=False
        )
        causal_summary, causal_detail = causal_estimator_sensitivity(
            controls, config.causal
        )
        causal_summary.to_csv(
            sensitivity_dir / "causal_estimator_summary.csv", index=False
        )
        causal_detail.to_csv(
            sensitivity_dir / "causal_estimator_runs.csv", index=False
        )
        tau_summary, tau_detail = gard_tau_sensitivity(config, controls)
        tau_summary.to_csv(sensitivity_dir / "gard_tau_summary.csv", index=False)
        tau_detail.to_csv(sensitivity_dir / "gard_tau_runs.csv", index=False)
    else:
        existing_detector = (
            output / "sensitivity" / "replicator_threshold_summary.csv"
        )
        if existing_detector.exists():
            detector_summary = pd.read_csv(existing_detector)

    table: Optional[pd.DataFrame] = None
    if include_interventions:
        table = intervention_table(all_runs)
        table.to_csv(output / "table1_interventions.csv", index=False)
        write_json(output / "intervention_tests.json", intervention_tests(all_runs))
        write_json(
            output / "intervention_generation_trends.json",
            intervention_generation_trends(all_runs),
        )
        intervention_sensitivity_dir = output / "sensitivity"
        intervention_sensitivity_dir.mkdir(parents=True, exist_ok=True)
        intervention_label_summary, intervention_label_detail, intervention_label_tests = (
            intervention_label_sensitivity(all_runs, config.replicator)
        )
        intervention_label_summary.to_csv(
            intervention_sensitivity_dir / "intervention_threshold_summary.csv",
            index=False,
        )
        intervention_label_detail.to_csv(
            intervention_sensitivity_dir / "intervention_threshold_runs.csv",
            index=False,
        )
        intervention_label_tests.to_csv(
            intervention_sensitivity_dir / "intervention_threshold_tests.csv",
            index=False,
        )

    forecast: Optional[pd.DataFrame] = None
    if include_forecast:
        forecast = run_forecast_experiment(
            controls,
            repetitions=config.bootstrap_repetitions,
            test_fraction=config.test_fraction,
            input_fraction=config.forecast_input_fraction,
            grid_points=config.forecast_grid_points,
            base_seed=config.base_seed + 100_000,
        )
        forecast.to_csv(output / "figure5_forecast.csv", index=False)
        write_json(output / "forecast_tests.json", forecast_tests(forecast))
        forecast_sensitivity_dir = output / "sensitivity"
        forecast_sensitivity_dir.mkdir(parents=True, exist_ok=True)
        forecast_threshold_detail, forecast_threshold_summary, forecast_threshold_tests = (
            run_forecast_threshold_sensitivity(
                controls,
                config.replicator,
                thresholds=DEFAULT_THRESHOLDS,
                repetitions=config.bootstrap_repetitions,
                test_fraction=config.test_fraction,
                input_fraction=config.forecast_input_fraction,
                grid_points=config.forecast_grid_points,
                base_seed=config.base_seed + 100_000,
            )
        )
        forecast_threshold_detail.to_csv(
            forecast_sensitivity_dir / "forecast_threshold_runs.csv", index=False
        )
        forecast_threshold_summary.to_csv(
            forecast_sensitivity_dir / "forecast_threshold_summary.csv", index=False
        )
        forecast_threshold_tests.to_csv(
            forecast_sensitivity_dir / "forecast_threshold_tests.csv", index=False
        )

    if make_plots:
        figure_dir = output / "figures"
        plot_figure2(controls, figure_dir / "figure2_phi_trajectories.png")
        plot_figure3(controls, figure_dir / "figure3_correlations.png")
        plot_figure4(controls, figure_dir / "figure4_state_comparison.png")
        if forecast is not None:
            plot_figure5(forecast, figure_dir / "figure5_forecast.png")
        if include_interventions:
            plot_figure6(all_runs, figure_dir / "figure6_interventions.png")
        if (
            detector_summary is not None
            and causal_summary is not None
            and tau_summary is not None
        ):
            plot_sensitivity(
                detector_summary,
                causal_summary,
                tau_summary,
                figure_dir / "supplement_sensitivity.png",
            )

    _write_readable_summary(
        output / "SUMMARY.md",
        config,
        control_summary,
        table,
        forecast,
        detector_summary,
    )
    return all_runs


def smoke_config(base: ExperimentConfig) -> ExperimentConfig:
    """Small real-dimensionality configuration for installation verification."""

    return replace(
        base,
        runs=6,
        bootstrap_repetitions=2,
        forecast_grid_points=32,
        gard=replace(base.gard, generations=12),
    )
