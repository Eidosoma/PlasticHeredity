"""Repeated held-out-run forecasting corresponding to the preprint's Figure 5."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy import stats
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .analysis import AnalyzedRun
from .composition import relative_composition
from .config import ReplicatorConfig
from .replicators import detect_replicators, replicator_metrics


FloatArray = NDArray[np.float64]


def _resample_continuous(values: ArrayLike, points: int) -> FloatArray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    if array.shape[0] == 1:
        return np.repeat(array, points, axis=0)
    old = np.linspace(0.0, 1.0, array.shape[0])
    new = np.linspace(0.0, 1.0, points)
    return np.column_stack([np.interp(new, old, array[:, column]) for column in range(array.shape[1])])


def _resample_binary(values: ArrayLike, points: int) -> NDArray[np.int64]:
    array = np.asarray(values, dtype=np.int64)
    positions = np.rint(np.linspace(0, array.size - 1, points)).astype(int)
    return array[positions]


def _features_for_run(run: AnalyzedRun) -> Mapping[str, FloatArray]:
    indices = run.causal.time_indices
    lag = run.causal.config.lag
    relative = relative_composition(run.trace.counts)
    change = np.linalg.norm(relative[indices] - relative[indices - lag], axis=1)
    return {
        "phi": run.causal.values[:, None],
        "composition_change": change[:, None],
        "compositions": relative[indices],
        "fluxes": run.trace.net_flux[indices].astype(float),
    }


@dataclass(frozen=True)
class ForecastDataset:
    inputs: Dict[str, FloatArray]
    targets: NDArray[np.int64]


def build_forecast_dataset(
    runs: Sequence[AnalyzedRun], *, input_fraction: float = 0.25, grid_points: int = 128
) -> ForecastDataset:
    """Time-normalize variable-length runs without using future input features."""

    if not runs:
        raise ValueError("at least one run is required")
    input_points = max(2, int(round(grid_points * input_fraction)))
    output_points = grid_points - input_points
    inputs: Dict[str, List[FloatArray]] = {
        "phi": [],
        "composition_change": [],
        "compositions": [],
        "fluxes": [],
    }
    targets: List[NDArray[np.int64]] = []
    for run in runs:
        labels = run.aligned_labels.astype(np.int64)
        cutoff = max(2, min(labels.size - 2, int(np.floor(input_fraction * labels.size))))
        feature_map = _features_for_run(run)
        for name, values in feature_map.items():
            prefix = _resample_continuous(values[:cutoff], input_points)
            inputs[name].append(prefix.reshape(-1))
        targets.append(_resample_binary(labels[cutoff:], output_points))
    return ForecastDataset(
        inputs={name: np.vstack(rows) for name, rows in inputs.items()},
        targets=np.vstack(targets),
    )


def run_forecast_experiment(
    runs: Sequence[AnalyzedRun],
    *,
    repetitions: int = 10,
    test_fraction: float = 0.2,
    input_fraction: float = 0.25,
    grid_points: int = 128,
    base_seed: int = 1729,
) -> pd.DataFrame:
    """Train one MLP per input family on repeated held-out run splits."""

    dataset = build_forecast_dataset(
        runs, input_fraction=input_fraction, grid_points=grid_points
    )
    n_runs = dataset.targets.shape[0]
    test_size = max(1, int(round(test_fraction * n_runs)))
    if n_runs - test_size < 2:
        raise ValueError("forecasting requires at least two training runs")
    rows = []
    for repetition in range(repetitions):
        rng = np.random.default_rng(base_seed + repetition)
        order = rng.permutation(n_runs)
        test_indices = order[:test_size]
        train_indices = order[test_size:]
        y_train = dataset.targets[train_indices]
        y_test = dataset.targets[test_indices]
        for name, values in dataset.inputs.items():
            model = make_pipeline(
                StandardScaler(),
                MLPClassifier(
                    hidden_layer_sizes=(64,),
                    activation="relu",
                    solver="adam",
                    alpha=1e-3,
                    batch_size="auto",
                    learning_rate_init=1e-3,
                    max_iter=500,
                    early_stopping=True,
                    validation_fraction=min(0.2, max(0.1, 2 / len(train_indices))),
                    n_iter_no_change=30,
                    random_state=base_seed + repetition,
                ),
            )
            model.fit(values[train_indices], y_train)
            prediction = model.predict(values[test_indices])
            accuracy = float(np.mean(prediction == y_test))
            rows.append(
                {"repetition": repetition, "model": name, "accuracy": accuracy}
            )
        majority = int(np.mean(y_train) >= 0.5)
        dummy_prediction = np.full_like(y_test, majority)
        rows.append(
            {
                "repetition": repetition,
                "model": "baseline",
                "accuracy": float(np.mean(dummy_prediction == y_test)),
            }
        )
    return pd.DataFrame(rows)


def forecast_tests(frame: pd.DataFrame) -> Dict[str, float]:
    """Mann-Whitney p-values comparing Phi-r against every baseline."""

    phi = frame.loc[frame.model == "phi", "accuracy"].to_numpy()
    results: Dict[str, float] = {}
    for model in sorted(set(frame.model) - {"phi"}):
        other = frame.loc[frame.model == model, "accuracy"].to_numpy()
        results[f"phi_vs_{model}"] = float(
            stats.mannwhitneyu(phi, other, alternative="greater", method="auto").pvalue
        )
    return results


def run_forecast_threshold_sensitivity(
    runs: Sequence[AnalyzedRun],
    replicator_config: ReplicatorConfig,
    *,
    thresholds: Sequence[float],
    repetitions: int,
    test_fraction: float,
    input_fraction: float,
    grid_points: int,
    base_seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Repeat the held-out forecast after relabeling identical trajectories."""

    frames: List[pd.DataFrame] = []
    for threshold in thresholds:
        detector_config = replace(
            replicator_config, similarity_threshold=float(threshold)
        )
        relabeled: List[AnalyzedRun] = []
        for run in runs:
            result = detect_replicators(run.trace, detector_config)
            relabeled.append(
                replace(
                    run,
                    replicator=result,
                    aligned_labels=result.labels[run.causal.time_indices],
                    metrics=replicator_metrics(result.labels),
                )
            )
        frame = run_forecast_experiment(
            relabeled,
            repetitions=repetitions,
            test_fraction=test_fraction,
            input_fraction=input_fraction,
            grid_points=grid_points,
            base_seed=base_seed,
        )
        frame.insert(0, "similarity_threshold", float(threshold))
        frames.append(frame)
    detail = pd.concat(frames, ignore_index=True)
    summary = (
        detail.groupby(["similarity_threshold", "model"], sort=False)
        .accuracy.agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    tests = []
    for threshold, selected in detail.groupby("similarity_threshold", sort=False):
        for comparison, pvalue in forecast_tests(selected).items():
            tests.append(
                {
                    "similarity_threshold": float(threshold),
                    "comparison": comparison,
                    "mann_whitney_greater_p": pvalue,
                }
            )
    return detail, summary, pd.DataFrame(tests)
