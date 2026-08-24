"""Genetic calibration of method choices omitted from the preprint.

The probe is deliberately separated from the registered replication. It fits
aggregate values reported by the paper on one deterministic seed cohort and
then evaluates the winning genome on untouched seeds. A close fit is therefore
a candidate reconstruction, not independent evidence for the paper's claims.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .analysis import AnalyzedRun, aggregate_phi, analyze_run, spike_correlations
from .config import CausalConfig, GardConfig, InterventionConfig, ReplicatorConfig
from .forecast import forecast_tests, run_forecast_experiment
from .gard import RunTrace, simulate_gard
from .interventions import PhiDirectedIntervention, OnlinePhiDirectedIntervention
from .replicators import detect_replicators, replicator_metrics
from .storage import write_json


EVALUATOR_VERSION = 3


@dataclass(frozen=True)
class TargetSpec:
    value: float
    tolerance: float
    weight: float
    provenance: str


# Targets are restricted to values stated in the manuscript or directly
# implied by Table 1. The inferred control step count is 716 / 0.88.
CONTROL_TARGETS: Mapping[str, TargetSpec] = {
    "molecular_steps_mean": TargetSpec(716 / 0.88, 225.0, 0.5, "Table 1 implied"),
    "control_persistence_mean": TargetSpec(716.0, 198.0, 1.0, "Table 1"),
    "control_probability_mean": TargetSpec(0.88, 0.03, 2.0, "Table 1"),
    "control_consistency_mean": TargetSpec(0.38, 0.06, 1.0, "Table 1"),
    "control_time_to_first_mean": TargetSpec(0.37, 0.27, 0.5, "Table 1"),
    "positive_correlation_fraction": TargetSpec(0.73, 0.10, 1.0, "Results"),
    "positive_significant_fraction": TargetSpec(0.54, 0.10, 1.0, "Results"),
    "mean_spearman_rho": TargetSpec(0.139, 0.06, 1.0, "Figure 3A annotation"),
    "negative_significant_fraction": TargetSpec(
        0.05, 0.05, 0.25, "Figure 3B digitized"
    ),
    "phi_higher_fraction": TargetSpec(0.57, 0.10, 1.0, "Results"),
    "median_mean_phi_drift": TargetSpec(
        0.2, 0.5, 0.4, "Figure 4B digitized"
    ),
    "median_mean_phi_replicating": TargetSpec(
        0.85, 0.5, 0.4, "Figure 4B digitized"
    ),
    "raw_ljung_significant_fraction": TargetSpec(0.86, 0.10, 0.5, "Results"),
    "differenced_ljung_significant_fraction": TargetSpec(
        1.0, 0.05, 0.5, "Results"
    ),
    "spike_time_rho": TargetSpec(0.66, 0.20, 0.25, "Results"),
    "spike_distance_rho": TargetSpec(0.71, 0.20, 0.25, "Results"),
    "evaluable_correlation_fraction": TargetSpec(1.0, 0.05, 0.5, "Figure 3"),
}


FULL_TARGETS: Mapping[str, TargetSpec] = {
    **CONTROL_TARGETS,
    "max_persistence_mean": TargetSpec(874.0, 233.0, 0.75, "Table 1"),
    "max_probability_mean": TargetSpec(0.88, 0.03, 1.0, "Table 1"),
    "max_consistency_mean": TargetSpec(0.52, 0.04, 0.75, "Table 1"),
    "max_time_to_first_mean": TargetSpec(0.36, 0.26, 0.4, "Table 1"),
    "min_persistence_mean": TargetSpec(559.0, 99.0, 0.75, "Table 1"),
    "min_probability_mean": TargetSpec(0.80, 0.03, 1.0, "Table 1"),
    "min_consistency_mean": TargetSpec(0.42, 0.04, 0.75, "Table 1"),
    "min_time_to_first_mean": TargetSpec(0.40, 0.28, 0.4, "Table 1"),
    # Figure 6C reports percentage-point slopes. They are divided by 100 here
    # because the reconstruction represents probabilities on [0, 1].
    "max_probability_slope": TargetSpec(
        0.00041, 0.00025, 0.4, "Figure 6C annotation"
    ),
    "control_probability_slope": TargetSpec(
        0.00008, 0.00025, 0.2, "Figure 6C annotation"
    ),
    "min_probability_slope": TargetSpec(
        -0.00030, 0.00025, 0.4, "Figure 6C annotation"
    ),
}


# Figure 5 supplies no numerical table. These medians were read from the plot
# and are used only to contextualize the winning genome on held-out runs, never
# as genetic-search fitness targets.
FIGURE5_DIGITIZED_TARGETS: Mapping[str, TargetSpec] = {
    "phi": TargetSpec(0.845, 0.020, 0.0, "Figure 5 digitized median"),
    "composition_change": TargetSpec(
        0.805, 0.025, 0.0, "Figure 5 digitized median"
    ),
    "compositions": TargetSpec(0.795, 0.025, 0.0, "Figure 5 digitized median"),
    "fluxes": TargetSpec(0.790, 0.025, 0.0, "Figure 5 digitized median"),
    "baseline": TargetSpec(0.610, 0.025, 0.0, "Figure 5 digitized median"),
}


@dataclass(frozen=True)
class ProbeConfig:
    """Genetic-search and fixed paper-model settings."""

    population_size: int = 24
    ga_generations: int = 10
    calibration_runs: int = 8
    holdout_runs: int = 24
    calibration_seed: int = 30_000
    holdout_seed: int = 90_000
    ga_seed: int = 2_718_281
    workers: int = 1
    objective: str = "control"
    elite_count: int = 2
    tournament_size: int = 3
    crossover_rate: float = 0.9
    mutation_rate: float = 0.25
    mutation_scale: float = 0.12
    max_trace_steps: int = 5_000
    n_types: int = 100
    initial_size: int = 40
    max_size: int = 80
    gard_generations: int = 100
    max_steps_per_generation: int = 1_000
    beta_log_mean: float = -4.0
    beta_log_sigma: float = 4.0

    def validate(self) -> None:
        if self.population_size < 4:
            raise ValueError("population_size must be at least 4")
        if self.ga_generations < 1:
            raise ValueError("ga_generations must be positive")
        if self.calibration_runs < 2 or self.holdout_runs < 2:
            raise ValueError("calibration_runs and holdout_runs must be at least 2")
        if not 1 <= self.elite_count < self.population_size:
            raise ValueError("elite_count must be in [1, population_size)")
        if not 2 <= self.tournament_size <= self.population_size:
            raise ValueError("invalid tournament_size")
        if not 0 <= self.crossover_rate <= 1 or not 0 <= self.mutation_rate <= 1:
            raise ValueError("crossover_rate and mutation_rate must be in [0, 1]")
        if self.mutation_scale <= 0:
            raise ValueError("mutation_scale must be positive")
        if self.workers < 1:
            raise ValueError("workers must be positive")
        if self.objective not in {"control", "full"}:
            raise ValueError("objective must be 'control' or 'full'")
        if not 2 <= self.initial_size <= self.max_size:
            raise ValueError("invalid initial/max size")
        if self.initial_size > self.n_types:
            raise ValueError("initial_size cannot exceed n_types")
        if self.gard_generations < 2 or self.max_steps_per_generation < 1:
            raise ValueError("invalid GARD duration")
        if self.max_trace_steps < self.gard_generations + 1:
            raise ValueError("max_trace_steps is too small for fission records")


@dataclass(frozen=True)
class GeneSpec:
    name: str
    kind: str
    default: Any
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Tuple[Any, ...] = ()

    def sample(self, rng: np.random.Generator) -> Any:
        if self.kind == "float":
            return float(rng.uniform(float(self.low), float(self.high)))
        if self.kind == "int":
            return int(rng.integers(int(self.low), int(self.high) + 1))
        if self.kind == "bool":
            return bool(rng.integers(0, 2))
        if self.kind == "categorical":
            return self.choices[int(rng.integers(0, len(self.choices)))]
        raise ValueError(f"unknown gene kind {self.kind!r}")

    def clip(self, value: Any) -> Any:
        if self.kind == "float":
            return float(np.clip(float(value), float(self.low), float(self.high)))
        if self.kind == "int":
            return int(np.clip(int(round(value)), int(self.low), int(self.high)))
        if self.kind == "bool":
            return bool(value)
        if self.kind == "categorical":
            if value not in self.choices:
                raise ValueError(f"invalid value {value!r} for gene {self.name}")
            return value
        raise ValueError(f"unknown gene kind {self.kind!r}")


def gene_specs(objective: str = "control") -> Tuple[GeneSpec, ...]:
    """Return the bounded search space, keeping reported paper values fixed."""

    specs = [
        # These two exposure genes are identifiable in a Poisson leap, unlike
        # kf, rho, kb, and tau separately.
        GeneSpec("log10_join_exposure", "float", math.log10(5e-5), -5.2, -3.5),
        GeneSpec("log10_leave_exposure", "float", math.log10(5e-5), -5.5, -3.2),
        GeneSpec("record_zero_event_steps", "bool", True),
        GeneSpec("log10_pseudocount", "float", math.log10(0.5), -3.0, 0.3),
        GeneSpec("partition_cut", "categorical", "zero", choices=("zero", "median")),
        GeneSpec("measure", "categorical", "wms", choices=("wms", "mmi_synergy")),
        GeneSpec("log10_covariance_ridge", "float", -8.0, -10.0, -5.0),
        GeneSpec("similarity_threshold", "float", 0.95, 0.30, 0.995),
        GeneSpec("min_recurrences", "int", 3, 2, 12),
        GeneSpec(
            "reference_states",
            "categorical",
            "generation_end",
            choices=("generation_end", "all"),
        ),
        GeneSpec(
            "similarity_metric",
            "categorical",
            "cosine",
            choices=("cosine", "euclidean"),
        ),
        GeneSpec(
            "reference_method",
            "categorical",
            "medoid",
            choices=("medoid", "neighbor_centroid"),
        ),
    ]
    if objective == "full":
        specs.append(
            GeneSpec(
                "intervention_estimator",
                "categorical",
                "online_initial",
                choices=("online_initial", "matched_control"),
            )
        )
    elif objective != "control":
        raise ValueError("objective must be 'control' or 'full'")
    return tuple(specs)


def default_genome(objective: str = "control") -> Dict[str, Any]:
    return {spec.name: spec.default for spec in gene_specs(objective)}


def canonical_genome(
    genome: Mapping[str, Any], objective: str = "control"
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for spec in gene_specs(objective):
        value = spec.clip(genome.get(spec.name, spec.default))
        result[spec.name] = round(value, 12) if isinstance(value, float) else value
    return result


def genome_id(genome: Mapping[str, Any], objective: str = "control") -> str:
    encoded = json.dumps(
        canonical_genome(genome, objective), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def resolved_configs(
    genome: Mapping[str, Any], probe: ProbeConfig
) -> Tuple[GardConfig, CausalConfig, ReplicatorConfig, InterventionConfig]:
    """Decode one genome into model configurations."""

    values = canonical_genome(genome, probe.objective)
    tau = 0.5
    environment = 1e-2
    join_exposure = 10 ** float(values["log10_join_exposure"])
    leave_exposure = 10 ** float(values["log10_leave_exposure"])
    gard = GardConfig(
        n_types=probe.n_types,
        initial_size=probe.initial_size,
        beta_log_mean=probe.beta_log_mean,
        beta_log_sigma=probe.beta_log_sigma,
        generations=probe.gard_generations,
        max_size=probe.max_size,
        max_steps_per_generation=probe.max_steps_per_generation,
        forward_rate=join_exposure / (environment * tau),
        backward_rate=leave_exposure / tau,
        environment_concentration=environment,
        tau=tau,
        record_zero_event_steps=bool(values["record_zero_event_steps"]),
    )
    causal = CausalConfig(
        pseudocount=10 ** float(values["log10_pseudocount"]),
        partition_cut=str(values["partition_cut"]),
        covariance_ridge=10 ** float(values["log10_covariance_ridge"]),
        measure=str(values["measure"]),
    )
    replicator = ReplicatorConfig(
        similarity_threshold=float(values["similarity_threshold"]),
        min_recurrences=int(values["min_recurrences"]),
        reference_states=str(values["reference_states"]),
        similarity_metric=str(values["similarity_metric"]),
        reference_method=str(values["reference_method"]),
    )
    intervention = InterventionConfig(
        estimator=str(values.get("intervention_estimator", "online_initial"))
    )
    gard.validate()
    causal.validate()
    replicator.validate()
    intervention.validate()
    return gard, causal, replicator, intervention


def _targets(objective: str) -> Mapping[str, TargetSpec]:
    return CONTROL_TARGETS if objective == "control" else FULL_TARGETS


def score_metrics(
    metrics: Mapping[str, Any], objective: str = "control"
) -> Tuple[float, Dict[str, float]]:
    """Robust normalized distance from reported aggregate values."""

    components: Dict[str, float] = {}
    weighted = 0.0
    total_weight = 0.0
    for name, target in _targets(objective).items():
        raw = metrics.get(name)
        if raw is None or not np.isfinite(float(raw)):
            loss = 25.0
        else:
            z = (float(raw) - target.value) / target.tolerance
            # Pseudo-Huber loss limits domination by a single badly matched
            # target while remaining quadratic close to the target.
            loss = float(math.sqrt(1.0 + z * z) - 1.0)
        components[name] = loss
        weighted += target.weight * loss
        total_weight += target.weight
    return weighted / total_weight, components


def _finite_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.mean(finite)) if finite.size else float("nan")


def _finite_median(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.median(finite)) if finite.size else float("nan")


def _control_metrics(runs: Sequence[AnalyzedRun]) -> Dict[str, float]:
    count = len(runs)
    rho = np.asarray([run.spearman_rho for run in runs], dtype=float)
    rho_p = np.asarray([run.spearman_p for run in runs], dtype=float)
    evaluable = np.isfinite(rho) & np.isfinite(rho_p)
    median, _, contributors = aggregate_phi(runs)
    minimum_contributors = max(3, len(runs) // 10)
    valid = np.isfinite(median) & (contributors >= minimum_contributors)
    aggregate_p = (
        float(stats.linregress(np.flatnonzero(valid), median[valid]).pvalue)
        if valid.sum() >= 3
        else float("nan")
    )
    spike_summary = spike_correlations(runs)
    return {
        "molecular_steps_mean": float(
            np.mean([run.trace.counts.shape[0] for run in runs])
        ),
        "control_persistence_mean": float(
            np.mean([run.metrics.persistence for run in runs])
        ),
        "control_probability_mean": float(
            np.mean([run.metrics.probability for run in runs])
        ),
        "control_consistency_mean": _finite_mean(
            run.metrics.consistency for run in runs
        ),
        "control_time_to_first_mean": _finite_mean(
            run.metrics.time_to_first for run in runs
        ),
        "positive_correlation_fraction": float(np.sum(evaluable & (rho > 0)) / count),
        "positive_significant_fraction": float(
            np.sum(evaluable & (rho > 0) & (rho_p < 0.05)) / count
        ),
        "mean_spearman_rho": _finite_mean(rho),
        "negative_significant_fraction": float(
            np.sum(evaluable & (rho < 0) & (rho_p < 0.05)) / count
        ),
        "evaluable_correlation_fraction": float(np.sum(evaluable) / count),
        "phi_higher_fraction": float(
            np.sum(
                [
                    np.isfinite(run.mean_phi_replicating)
                    and np.isfinite(run.mean_phi_drift)
                    and run.mean_phi_replicating > run.mean_phi_drift
                    for run in runs
                ]
            )
            / count
        ),
        "median_mean_phi_drift": _finite_median(
            run.mean_phi_drift for run in runs
        ),
        "median_mean_phi_replicating": _finite_median(
            run.mean_phi_replicating for run in runs
        ),
        "raw_ljung_significant_fraction": float(
            np.sum(
                [
                    np.isfinite(run.ljung_box_p) and run.ljung_box_p < 0.05
                    for run in runs
                ]
            )
            / count
        ),
        "differenced_ljung_significant_fraction": float(
            np.sum(
                [
                    np.isfinite(run.differenced_ljung_box_p)
                    and run.differenced_ljung_box_p < 0.05
                    for run in runs
                ]
            )
            / count
        ),
        "spike_time_rho": float(spike_summary["mean_spike_time"]["rho"]),
        "spike_distance_rho": float(
            spike_summary["mean_spike_distance"]["rho"]
        ),
        "spike_height_rho": float(spike_summary["mean_spike_height"]["rho"]),
        "spike_height_pvalue": float(
            spike_summary["mean_spike_height"]["pvalue"]
        ),
        "aggregate_phi_trend_pvalue": aggregate_p,
        "median_ljung_box_pvalue": _finite_median(
            run.ljung_box_p for run in runs
        ),
    }


def _figure5_validation(
    controls: Sequence[AnalyzedRun], probe: ProbeConfig
) -> Dict[str, Any]:
    if len(controls) < 5:
        return {
            "status": "skipped",
            "reason": "at least five holdout runs are required",
            "runs": len(controls),
        }
    # Match the manuscript's sample size when the validation cohort is larger.
    selected = list(controls[:100])
    try:
        frame = run_forecast_experiment(
            selected,
            repetitions=10,
            test_fraction=0.2,
            input_fraction=0.25,
            grid_points=128,
            base_seed=probe.holdout_seed + 500_000,
        )
        observed: Dict[str, Dict[str, float]] = {}
        for model, values in frame.groupby("model", sort=False).accuracy:
            observed[str(model)] = {
                "median": float(values.median()),
                "mean": float(values.mean()),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        return {
            "status": "ok",
            "runs": len(selected),
            "repetitions": 10,
            "observed": observed,
            "digitized_targets": {
                name: asdict(target)
                for name, target in FIGURE5_DIGITIZED_TARGETS.items()
            },
            "mann_whitney_greater_pvalues": forecast_tests(frame),
            "note": (
                "Targets were digitized from Figure 5 and are validation-only; "
                "they do not affect genetic fitness."
            ),
        }
    except Exception as error:
        return {
            "status": "failed",
            "runs": len(selected),
            "error": f"{type(error).__name__}: {error}",
        }


def _intervention_policy(
    treatment: str,
    control: AnalyzedRun,
    causal: CausalConfig,
    intervention: InterventionConfig,
    max_size: int,
) -> Any:
    if intervention.estimator == "matched_control":
        return PhiDirectedIntervention(control.causal, treatment, max_size)
    return OnlinePhiDirectedIntervention(
        causal,
        treatment,
        max_size,
        refit_each_generation=intervention.estimator == "online_history",
    )


def _probability_slope(
    traces_and_labels: Sequence[Tuple[RunTrace, np.ndarray]],
) -> float:
    rows: List[np.ndarray] = []
    for trace, labels in traces_and_labels:
        probabilities = np.full(int(trace.generations.max()) + 1, np.nan)
        for generation in range(probabilities.size):
            selected = trace.generations == generation
            if selected.any():
                probabilities[generation] = np.mean(labels[selected])
        rows.append(probabilities)
    width = max(row.size for row in rows)
    matrix = np.full((len(rows), width), np.nan)
    for index, row in enumerate(rows):
        matrix[index, : row.size] = row
    generation = np.broadcast_to(np.arange(width), matrix.shape)
    finite = np.isfinite(matrix)
    if finite.sum() < 3 or np.unique(generation[finite]).size < 2:
        return float("nan")
    return float(stats.linregress(generation[finite], matrix[finite]).slope)


def _treatment_metrics(
    traces: Mapping[str, Sequence[RunTrace]],
    controls: Sequence[AnalyzedRun],
    replicator: ReplicatorConfig,
) -> Dict[str, float]:
    result: Dict[str, float] = {}
    detected: Dict[str, List[Tuple[RunTrace, np.ndarray]]] = {
        "control": [(run.trace, run.replicator.labels) for run in controls]
    }
    for treatment in ("max", "min"):
        treatment_pairs: List[Tuple[RunTrace, np.ndarray]] = []
        metrics = []
        for trace in traces[treatment]:
            labels = detect_replicators(trace, replicator).labels
            treatment_pairs.append((trace, labels))
            metrics.append(replicator_metrics(labels))
        detected[treatment] = treatment_pairs
        result[f"{treatment}_persistence_mean"] = float(
            np.mean([value.persistence for value in metrics])
        )
        result[f"{treatment}_probability_mean"] = float(
            np.mean([value.probability for value in metrics])
        )
        result[f"{treatment}_consistency_mean"] = _finite_mean(
            value.consistency for value in metrics
        )
        result[f"{treatment}_time_to_first_mean"] = _finite_mean(
            value.time_to_first for value in metrics
        )
    for treatment in ("max", "control", "min"):
        result[f"{treatment}_probability_slope"] = _probability_slope(
            detected[treatment]
        )
    return result


def evaluate_genome(
    genome: Mapping[str, Any],
    probe: ProbeConfig,
    seeds: Sequence[int],
    *,
    include_figure5: bool = False,
) -> Dict[str, Any]:
    """Evaluate one candidate deterministically on the supplied seeds."""

    started = time.monotonic()
    values = canonical_genome(genome, probe.objective)
    try:
        gard, causal, replicator, intervention = resolved_configs(values, probe)
        controls: List[AnalyzedRun] = []
        treatment_traces: Dict[str, List[RunTrace]] = {"max": [], "min": []}
        for run_index, seed in enumerate(seeds):
            trace = simulate_gard(gard, int(seed))
            if trace.counts.shape[0] > probe.max_trace_steps:
                raise RuntimeError(
                    f"trace length {trace.counts.shape[0]} exceeds probe cap "
                    f"{probe.max_trace_steps}"
                )
            control = analyze_run(
                trace,
                run_index=run_index,
                treatment="control",
                causal_config=causal,
                replicator_config=replicator,
            )
            controls.append(control)
            if probe.objective == "full":
                for treatment in ("max", "min"):
                    policy = _intervention_policy(
                        treatment, control, causal, intervention, gard.max_size
                    )
                    treated = simulate_gard(
                        gard,
                        int(seed),
                        beta=trace.beta,
                        intervention=policy,
                    )
                    if treated.counts.shape[0] > probe.max_trace_steps:
                        raise RuntimeError(
                            f"{treatment} trace length {treated.counts.shape[0]} "
                            f"exceeds probe cap {probe.max_trace_steps}"
                        )
                    treatment_traces[treatment].append(treated)
        metrics = _control_metrics(controls)
        if probe.objective == "full":
            metrics.update(_treatment_metrics(treatment_traces, controls, replicator))
        score, components = score_metrics(metrics, probe.objective)
        result = {
            "status": "ok",
            "score": score,
            "metrics": metrics,
            "component_losses": components,
            "elapsed_seconds": time.monotonic() - started,
        }
        if include_figure5:
            result["figure5_validation"] = _figure5_validation(controls, probe)
            result["elapsed_seconds"] = time.monotonic() - started
        return result
    except Exception as error:  # Candidate failures are data, not search crashes.
        return {
            "status": "failed",
            "score": 1_000_000.0,
            "metrics": {},
            "component_losses": {},
            "error": f"{type(error).__name__}: {error}",
            "elapsed_seconds": time.monotonic() - started,
        }


def _cache_key(
    genome: Mapping[str, Any],
    probe: ProbeConfig,
    seeds: Sequence[int],
    include_figure5: bool,
) -> str:
    payload = {
        "evaluator_version": EVALUATOR_VERSION,
        "genome": canonical_genome(genome, probe.objective),
        "probe": asdict(probe),
        "seeds": list(map(int, seeds)),
        "include_figure5": include_figure5,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_json(temporary, payload)
    os.replace(temporary, path)


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _worker_evaluate(
    args: Tuple[Dict[str, Any], ProbeConfig, Tuple[int, ...], bool]
) -> Dict[str, Any]:
    genome, probe, seeds, include_figure5 = args
    return evaluate_genome(
        genome, probe, seeds, include_figure5=include_figure5
    )


def evaluate_population(
    population: Sequence[Mapping[str, Any]],
    probe: ProbeConfig,
    seeds: Sequence[int],
    cache_dir: Path,
    *,
    include_figure5: bool = False,
) -> List[Dict[str, Any]]:
    """Evaluate a population with persistent cache and optional processes."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    results: List[Optional[Dict[str, Any]]] = [None] * len(population)
    pending: Dict[str, Tuple[Dict[str, Any], List[int]]] = {}
    for index, genome in enumerate(population):
        canonical = canonical_genome(genome, probe.objective)
        key = _cache_key(canonical, probe, seeds, include_figure5)
        path = cache_dir / f"{key}.json"
        if path.exists():
            cached = _load_json(path)
            cached["cache_hit"] = True
            cached["cache_key"] = key
            results[index] = cached
        elif key in pending:
            pending[key][1].append(index)
        else:
            pending[key] = (canonical, [index])

    items = list(pending.items())
    arguments = [
        (genome, probe, tuple(map(int, seeds)), include_figure5)
        for _, (genome, _) in items
    ]
    if probe.workers > 1 and len(arguments) > 1:
        with ProcessPoolExecutor(max_workers=probe.workers) as executor:
            evaluated = list(executor.map(_worker_evaluate, arguments))
    else:
        evaluated = [_worker_evaluate(argument) for argument in arguments]

    for (key, (_, indices)), result in zip(items, evaluated):
        result["cache_hit"] = False
        result["cache_key"] = key
        _atomic_json(cache_dir / f"{key}.json", result)
        for index in indices:
            results[index] = dict(result)
    if any(result is None for result in results):
        raise RuntimeError("internal error: population evaluation is incomplete")
    return [dict(result) for result in results if result is not None]


def mutate_genome(
    genome: Mapping[str, Any], probe: ProbeConfig, rng: np.random.Generator
) -> Dict[str, Any]:
    child = canonical_genome(genome, probe.objective)
    for spec in gene_specs(probe.objective):
        if rng.random() >= probe.mutation_rate:
            continue
        current = child[spec.name]
        if spec.kind == "float":
            width = float(spec.high) - float(spec.low)
            current = float(current) + rng.normal(0.0, probe.mutation_scale * width)
        elif spec.kind == "int":
            width = max(1.0, float(spec.high) - float(spec.low))
            current = int(round(float(current) + rng.normal(0.0, probe.mutation_scale * width)))
        elif spec.kind == "bool":
            current = not bool(current)
        else:
            alternatives = [choice for choice in spec.choices if choice != current]
            current = alternatives[int(rng.integers(0, len(alternatives)))]
        child[spec.name] = spec.clip(current)
    return canonical_genome(child, probe.objective)


def crossover_genomes(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    probe: ProbeConfig,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    left = canonical_genome(first, probe.objective)
    right = canonical_genome(second, probe.objective)
    if rng.random() >= probe.crossover_rate:
        return dict(left if rng.random() < 0.5 else right)
    child: Dict[str, Any] = {}
    for spec in gene_specs(probe.objective):
        if spec.kind == "float":
            alpha = float(rng.uniform(0.0, 1.0))
            child[spec.name] = spec.clip(
                alpha * float(left[spec.name]) + (1 - alpha) * float(right[spec.name])
            )
        else:
            child[spec.name] = (
                left[spec.name] if rng.random() < 0.5 else right[spec.name]
            )
    return canonical_genome(child, probe.objective)


def _initial_population(probe: ProbeConfig, rng: np.random.Generator) -> List[Dict[str, Any]]:
    specs = gene_specs(probe.objective)
    registered = default_genome(probe.objective)
    seeds = [registered]
    for threshold in (0.50, 0.70):
        candidate = dict(registered)
        candidate["similarity_threshold"] = threshold
        seeds.append(candidate)
    candidate = dict(registered)
    candidate["record_zero_event_steps"] = False
    seeds.append(candidate)
    population = [canonical_genome(value, probe.objective) for value in seeds]
    while len(population) < probe.population_size:
        population.append({spec.name: spec.sample(rng) for spec in specs})
    return population[: probe.population_size]


def _select_parent(
    population: Sequence[Dict[str, Any]],
    results: Sequence[Mapping[str, Any]],
    probe: ProbeConfig,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    indices = rng.integers(0, len(population), size=probe.tournament_size)
    best = min(indices, key=lambda index: float(results[int(index)]["score"]))
    return population[int(best)]


def _next_population(
    population: Sequence[Dict[str, Any]],
    results: Sequence[Mapping[str, Any]],
    probe: ProbeConfig,
    rng: np.random.Generator,
) -> List[Dict[str, Any]]:
    ranked = sorted(range(len(population)), key=lambda index: float(results[index]["score"]))
    next_population = [dict(population[index]) for index in ranked[: probe.elite_count]]
    seen = {
        json.dumps(value, sort_keys=True, separators=(",", ":"))
        for value in next_population
    }
    attempts = 0
    while len(next_population) < probe.population_size:
        first = _select_parent(population, results, probe, rng)
        second = _select_parent(population, results, probe, rng)
        child = crossover_genomes(first, second, probe, rng)
        child = mutate_genome(child, probe, rng)
        encoded = json.dumps(child, sort_keys=True, separators=(",", ":"))
        attempts += 1
        if encoded in seen and attempts < 10 * probe.population_size:
            continue
        if encoded in seen:
            child = {
                spec.name: spec.sample(rng) for spec in gene_specs(probe.objective)
            }
            child = canonical_genome(child, probe.objective)
            encoded = json.dumps(child, sort_keys=True, separators=(",", ":"))
        next_population.append(child)
        seen.add(encoded)
    return next_population


def _history_record(
    generation: int,
    index: int,
    genome: Mapping[str, Any],
    result: Mapping[str, Any],
    objective: str,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "generation": generation,
        "candidate_index": index,
        "candidate_id": genome_id(genome, objective),
        "score": result["score"],
        "status": result["status"],
        "cache_hit": result.get("cache_hit", False),
        "elapsed_seconds": result.get("elapsed_seconds"),
        "error": result.get("error"),
    }
    record.update({f"gene_{key}": value for key, value in genome.items()})
    record.update(
        {f"metric_{key}": value for key, value in result.get("metrics", {}).items()}
    )
    record.update(
        {
            f"loss_{key}": value
            for key, value in result.get("component_losses", {}).items()
        }
    )
    return record


def _plot_history(history: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    finite = history[np.isfinite(history.score)]
    grouped = finite.groupby("generation").score
    summary = grouped.agg(["min", "median", "max"])
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.plot(summary.index, summary["min"], marker="o", label="best")
    ax.plot(summary.index, summary["median"], marker="o", label="median")
    ax.fill_between(
        summary.index,
        summary["min"],
        summary["max"],
        color="#2a7fbb",
        alpha=0.12,
        label="population range",
    )
    ax.set(
        xlabel="genetic generation",
        ylabel="paper-distance score (lower is better)",
        title="Genetic settings probe",
    )
    ax.legend(frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_summary(path: Path, best: Mapping[str, Any], objective: str) -> None:
    calibration = best["calibration"]
    holdout = best["holdout"]
    targets = _targets(objective)
    lines = [
        "# Genetic settings probe",
        "",
        f"Objective: `{objective}`.",
        f"Best calibration score: {float(calibration['score']):.6g}.",
        f"Held-out score: {float(holdout['score']):.6g}.",
        "",
        "## Candidate genome",
        "",
        "```json",
        json.dumps(best["genome"], indent=2, sort_keys=True),
        "```",
        "",
        "## Target comparison",
        "",
        "| Metric | Paper target | Calibration | Holdout |",
        "|---|---:|---:|---:|",
    ]
    for name, target in targets.items():
        cal = calibration.get("metrics", {}).get(name)
        held = holdout.get("metrics", {}).get(name)
        cal_text = "NA" if cal is None else f"{float(cal):.6g}"
        held_text = "NA" if held is None else f"{float(held):.6g}"
        lines.append(f"| {name} | {target.value:.6g} | {cal_text} | {held_text} |")
    lines.append("")
    figure5 = holdout.get("figure5_validation", {})
    if figure5.get("status") == "ok":
        lines.extend(
            [
                "## Figure 5 held-out validation",
                "",
                "These plot targets are visually digitized approximations and did not affect genetic fitness.",
                "",
                "| Predictor | Paper median (approx.) | Held-out median |",
                "|---|---:|---:|",
            ]
        )
        observed = figure5.get("observed", {})
        for name, target in FIGURE5_DIGITIZED_TARGETS.items():
            value = observed.get(name, {}).get("median")
            held_text = "NA" if value is None else f"{float(value):.3%}"
            lines.append(f"| {name} | {target.value:.1%} | {held_text} |")
        lines.append("")
    lines.extend(
        [
            "## Guardrail",
            "",
            "This candidate was selected to resemble published aggregates. It is a hypothesis about omitted settings, not independent confirmation of the manuscript. The held-out seed cohort is the appropriate generalization check.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_probe(
    probe: ProbeConfig, output: Path, *, overwrite: bool = False
) -> Dict[str, Any]:
    """Run or resume the genetic search and validate its winner on holdout seeds."""

    probe.validate()
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "probe_config.json"
    current_config = asdict(probe)
    if config_path.exists() and not overwrite:
        if _load_json(config_path) != current_config:
            raise RuntimeError(
                "probe output contains a different configuration; use a new "
                "output directory or --overwrite"
            )
    write_json(config_path, current_config)
    write_json(
        output / "paper_targets.json",
        {name: asdict(target) for name, target in _targets(probe.objective).items()},
    )
    write_json(
        output / "figure5_validation_targets.json",
        {
            name: asdict(target)
            for name, target in FIGURE5_DIGITIZED_TARGETS.items()
        },
    )
    runtime_path = output / "runtime.json"
    started_at = datetime.now(timezone.utc).isoformat()
    runtime = {
        "status": "running",
        "pid": os.getpid(),
        "started_at_utc": started_at,
        "finished_at_utc": None,
        "objective": probe.objective,
    }
    _atomic_json(runtime_path, runtime)

    checkpoint_path = output / "checkpoint.json"
    history_path = output / "generation_history.csv"
    rng = np.random.default_rng(probe.ga_seed)
    if checkpoint_path.exists() and not overwrite:
        checkpoint = _load_json(checkpoint_path)
        if checkpoint.get("complete") and (output / "best_candidate.json").exists():
            runtime.update(
                {
                    "status": "complete",
                    "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                    "resumed_existing_result": True,
                }
            )
            _atomic_json(runtime_path, runtime)
            return _load_json(output / "best_candidate.json")
        population = [
            canonical_genome(value, probe.objective)
            for value in checkpoint["population"]
        ]
        start_generation = int(checkpoint["next_generation"])
        rng.bit_generator.state = checkpoint["rng_state"]
        history_records = (
            pd.read_csv(history_path).to_dict(orient="records")
            if history_path.exists()
            else []
        )
    else:
        population = _initial_population(probe, rng)
        start_generation = 0
        history_records: List[Dict[str, Any]] = []

    calibration_seeds = tuple(
        range(probe.calibration_seed, probe.calibration_seed + probe.calibration_runs)
    )
    global_best: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None
    for generation in range(start_generation, probe.ga_generations):
        results = evaluate_population(
            population, probe, calibration_seeds, output / "cache"
        )
        for index, (genome, result) in enumerate(zip(population, results)):
            history_records.append(
                _history_record(generation, index, genome, result, probe.objective)
            )
        best_index = min(range(len(population)), key=lambda i: float(results[i]["score"]))
        generation_best = (dict(population[best_index]), dict(results[best_index]))
        if global_best is None or float(generation_best[1]["score"]) < float(
            global_best[1]["score"]
        ):
            global_best = generation_best

        history = pd.DataFrame(history_records)
        history.to_csv(history_path, index=False)
        next_population = (
            _next_population(population, results, probe, rng)
            if generation + 1 < probe.ga_generations
            else population
        )
        _atomic_json(
            checkpoint_path,
            {
                "complete": False,
                "next_generation": generation + 1,
                "population": next_population,
                "rng_state": rng.bit_generator.state,
            },
        )
        print(
            f"probe generation {generation + 1}/{probe.ga_generations}: "
            f"best score={float(generation_best[1]['score']):.6g}",
            flush=True,
        )
        population = next_population

    if global_best is None:
        # This occurs only if a prior checkpoint reached the generation limit
        # without being finalized. Recover the best genome from history.
        history = pd.read_csv(history_path)
        row = history.loc[history.score.idxmin()]
        genome = {
            spec.name: row[f"gene_{spec.name}"]
            for spec in gene_specs(probe.objective)
        }
        genome = canonical_genome(genome, probe.objective)
        calibration = evaluate_population(
            [genome], probe, calibration_seeds, output / "cache"
        )[0]
        global_best = (genome, calibration)

    best_genome, calibration = global_best
    holdout_seeds = tuple(range(probe.holdout_seed, probe.holdout_seed + probe.holdout_runs))
    holdout = evaluate_population(
        [best_genome],
        probe,
        holdout_seeds,
        output / "cache",
        include_figure5=True,
    )[0]
    gard, causal, replicator, intervention = resolved_configs(best_genome, probe)
    best: Dict[str, Any] = {
        "genome": best_genome,
        "genome_id": genome_id(best_genome, probe.objective),
        "resolved_config": {
            "gard": asdict(gard),
            "causal": asdict(causal),
            "replicator": asdict(replicator),
            "intervention": asdict(intervention),
            "join_exposure": 10 ** float(best_genome["log10_join_exposure"]),
            "leave_exposure": 10 ** float(best_genome["log10_leave_exposure"]),
        },
        "calibration_seeds": list(calibration_seeds),
        "holdout_seeds": list(holdout_seeds),
        "calibration": calibration,
        "holdout": holdout,
    }
    _atomic_json(output / "best_candidate.json", best)
    history = pd.read_csv(history_path)
    (
        history.sort_values("score")
        .drop_duplicates("candidate_id")
        .head(25)
        .to_csv(output / "top_candidates.csv", index=False)
    )
    _plot_history(history, output / "convergence.png")
    _write_summary(output / "SUMMARY.md", best, probe.objective)
    _atomic_json(
        checkpoint_path,
        {
            "complete": True,
            "next_generation": probe.ga_generations,
            "population": population,
            "rng_state": rng.bit_generator.state,
            "best_genome_id": best["genome_id"],
        },
    )
    runtime.update(
        {
            "status": "complete",
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "resumed_existing_result": False,
        }
    )
    _atomic_json(runtime_path, runtime)
    # Return exactly the durable representation. This normalizes non-finite
    # diagnostic values to JSON null and makes completed resume byte-stable.
    return _load_json(output / "best_candidate.json")
