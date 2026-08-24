"""Matrix-aware prospective tests for mechanistic model contrasts."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from .config import CANDIDATES, ExperimentConfig
from .experiment import BranchBatch, StateCase, _candidate_indices, _stack_targets
from .metrics import centered_spearman, spearman
from .seeds import derive_seed

PRIMARY_CONTRASTS: dict[str, tuple[str, str]] = {
    "state": ("h10", "h10_state"),
    "network": ("h10_state", "h10_state_beta"),
    "interaction": ("h10_state_beta", "h10_state_beta_interaction"),
}

DESCRIPTIVE_CONTRASTS: dict[str, tuple[str, str]] = {
    "clock_history": ("h8", "h10"),
    "corrected_duplicate": ("h10", "h10_duplicate_corrected"),
    "ridge_duplicate": ("ridge_h10", "ridge_h10_duplicate"),
    "beta_beyond_history": ("h10", "h10_beta"),
    "legacy_composite": ("legacy_h9", "legacy_full"),
}


def holm_adjust(values: list[float]) -> list[float]:
    """Return Holm step-down family-wise adjusted p-values."""

    count = len(values)
    order = np.argsort(np.asarray(values, dtype=np.float64), kind="mergesort")
    adjusted = np.empty(count, dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * values[int(index)])
        adjusted[int(index)] = min(1.0, running)
    return adjusted.tolist()


def _state_log_loss(q: NDArray, prediction: NDArray) -> NDArray[np.float64]:
    prediction = np.clip(np.asarray(prediction, dtype=np.float64), 1e-12, 1 - 1e-12)
    q = np.asarray(q, dtype=np.float64)
    return -(q * np.log(prediction) + (1.0 - q) * np.log(1.0 - prediction))


def _state_brier(q: NDArray, prediction: NDArray) -> NDArray[np.float64]:
    return (np.asarray(q) - np.asarray(prediction)) ** 2


def _matrix_means(values: NDArray, matrix_ids: NDArray) -> NDArray[np.float64]:
    return np.asarray(
        [np.mean(values[matrix_ids == key]) for key in np.unique(matrix_ids)],
        dtype=np.float64,
    )


def _paired_gain(
    q: NDArray,
    baseline: NDArray,
    enhanced: NDArray,
    matrix_ids: NDArray,
    loss: Callable[[NDArray, NDArray], NDArray[np.float64]],
    bootstrap_repetitions: int,
    rng: np.random.Generator,
) -> tuple[float, tuple[float, float]]:
    difference = loss(q, baseline) - loss(q, enhanced)
    group_values = _matrix_means(difference, matrix_ids)
    observed = float(group_values.mean())
    sampled = rng.integers(
        0,
        group_values.size,
        size=(bootstrap_repetitions, group_values.size),
    )
    bootstraps = group_values[sampled].mean(axis=1)
    lower, upper = np.quantile(bootstraps, (0.025, 0.975))
    return observed, (float(lower), float(upper))


def paired_matrix_randomization_p(
    q: NDArray,
    baseline: NDArray,
    enhanced: NDArray,
    matrix_ids: NDArray,
    repetitions: int,
    rng: np.random.Generator,
) -> float:
    """Swap baseline/enhanced assignments for every state in a matrix block."""

    difference = _state_log_loss(q, baseline) - _state_log_loss(q, enhanced)
    group_values = _matrix_means(difference, matrix_ids)
    observed = float(group_values.mean())
    signs = rng.integers(0, 2, size=(repetitions, group_values.size), dtype=np.int8)
    signs = signs.astype(np.float64) * 2.0 - 1.0
    null = (signs @ group_values) / group_values.size
    return float((np.count_nonzero(null >= observed) + 1) / (repetitions + 1))


def _rank_metrics(
    prediction: NDArray, q_a: NDArray, q_b: NDArray, matrix_ids: NDArray
) -> dict[str, Any]:
    overall = [spearman(prediction, q_a), spearman(prediction, q_b)]
    centered = [
        centered_spearman(prediction, q_a, matrix_ids),
        centered_spearman(prediction, q_b, matrix_ids),
    ]
    return {
        "overall_spearman": overall,
        "overall_spearman_mean": float(np.nanmean(overall)),
        "centered_spearman": centered,
        "centered_spearman_mean": float(np.nanmean(centered)),
    }


def _reliability_bootstrap(
    left: NDArray,
    right: NDArray,
    matrix_ids: NDArray,
    centered: bool,
    repetitions: int,
    rng: np.random.Generator,
) -> tuple[float, tuple[float, float]]:
    unique = np.unique(matrix_ids)
    locations = [np.flatnonzero(matrix_ids == key) for key in unique]
    samples = np.empty(repetitions, dtype=np.float64)
    for repetition in range(repetitions):
        chosen = rng.integers(0, unique.size, size=unique.size)
        indices = np.concatenate([locations[index] for index in chosen])
        groups = np.repeat(
            np.arange(unique.size), [locations[index].size for index in chosen]
        )
        samples[repetition] = (
            centered_spearman(left[indices], right[indices], groups)
            if centered
            else spearman(left[indices], right[indices])
        )
    observed = (
        centered_spearman(left, right, matrix_ids)
        if centered
        else spearman(left, right)
    )
    finite = samples[np.isfinite(samples)]
    interval = np.quantile(finite, (0.025, 0.975))
    return observed, (float(interval[0]), float(interval[1]))


def compute_mechanistic_metrics(
    cases: list[StateCase],
    batches: list[BranchBatch],
    predictions: dict[str, dict[str, NDArray[np.float64]]],
    experiment: ExperimentConfig,
) -> dict[str, Any]:
    labels = _stack_targets(batches)
    metrics: dict[str, Any] = {
        "candidates": {},
        "primary_tests": [],
        "descriptive_tests": [],
        "primary_contrasts": PRIMARY_CONTRASTS,
        "descriptive_contrasts": DESCRIPTIVE_CONTRASTS,
    }
    primary_rows: list[dict[str, Any]] = []
    descriptive_rows: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        selected = _candidate_indices(cases, candidate)
        candidate_labels = labels[selected]
        split = candidate_labels.shape[1] // 2
        q_a = candidate_labels[:, :split].mean(axis=1)
        q_b = candidate_labels[:, split:].mean(axis=1)
        q_all = candidate_labels.mean(axis=1)
        matrix_ids = np.asarray([cases[index].matrix_id for index in selected])
        reliability_rng = np.random.default_rng(
            derive_seed(
                experiment.master_seed, "MECHCONF.metrics.reliability", candidate
            )
        )
        reliability, reliability_ci = _reliability_bootstrap(
            q_a,
            q_b,
            matrix_ids,
            False,
            experiment.bootstrap_repetitions,
            reliability_rng,
        )
        centered_reliability, centered_reliability_ci = _reliability_bootstrap(
            q_a,
            q_b,
            matrix_ids,
            True,
            experiment.bootstrap_repetitions,
            reliability_rng,
        )
        candidate_metrics: dict[str, Any] = {
            "states": int(selected.size),
            "transition_region_states": int(((q_all > 0.1) & (q_all < 0.9)).sum()),
            "branch_half_reliability": reliability,
            "branch_half_reliability_ci95": reliability_ci,
            "centered_branch_half_reliability": centered_reliability,
            "centered_branch_half_reliability_ci95": centered_reliability_ci,
            "models": {
                name: _rank_metrics(prediction, q_a, q_b, matrix_ids)
                for name, prediction in predictions[candidate].items()
            },
        }
        metrics["candidates"][candidate] = candidate_metrics

        for contrast, (baseline_name, enhanced_name) in PRIMARY_CONTRASTS.items():
            baseline = predictions[candidate][baseline_name]
            enhanced = predictions[candidate][enhanced_name]
            for direction, q in (("A", q_a), ("B", q_b)):
                seed_parts = (contrast, candidate, direction)
                log_gain, log_ci = _paired_gain(
                    q,
                    baseline,
                    enhanced,
                    matrix_ids,
                    _state_log_loss,
                    experiment.bootstrap_repetitions,
                    np.random.default_rng(
                        derive_seed(
                            experiment.master_seed,
                            "MECHCONF.metrics.bootstrap.log_loss",
                            *seed_parts,
                        )
                    ),
                )
                brier_gain, brier_ci = _paired_gain(
                    q,
                    baseline,
                    enhanced,
                    matrix_ids,
                    _state_brier,
                    experiment.bootstrap_repetitions,
                    np.random.default_rng(
                        derive_seed(
                            experiment.master_seed,
                            "MECHCONF.metrics.bootstrap.brier",
                            *seed_parts,
                        )
                    ),
                )
                p_value = paired_matrix_randomization_p(
                    q,
                    baseline,
                    enhanced,
                    matrix_ids,
                    experiment.permutation_repetitions,
                    np.random.default_rng(
                        derive_seed(
                            experiment.master_seed,
                            "MECHCONF.metrics.randomization",
                            *seed_parts,
                        )
                    ),
                )
                primary_rows.append(
                    {
                        "contrast": contrast,
                        "baseline": baseline_name,
                        "enhanced": enhanced_name,
                        "candidate": candidate,
                        "direction": direction,
                        "log_loss_gain": log_gain,
                        "log_loss_gain_ci95": log_ci,
                        "q_brier_gain": brier_gain,
                        "q_brier_gain_ci95": brier_ci,
                        "randomization_p_raw": p_value,
                    }
                )

        for contrast, (baseline_name, enhanced_name) in DESCRIPTIVE_CONTRASTS.items():
            baseline = predictions[candidate][baseline_name]
            enhanced = predictions[candidate][enhanced_name]
            for direction, q in (("A", q_a), ("B", q_b)):
                seed_parts = (contrast, candidate, direction)
                log_gain, log_ci = _paired_gain(
                    q,
                    baseline,
                    enhanced,
                    matrix_ids,
                    _state_log_loss,
                    experiment.bootstrap_repetitions,
                    np.random.default_rng(
                        derive_seed(
                            experiment.master_seed,
                            "MECHCONF.metrics.descriptive.log_loss",
                            *seed_parts,
                        )
                    ),
                )
                brier_gain, brier_ci = _paired_gain(
                    q,
                    baseline,
                    enhanced,
                    matrix_ids,
                    _state_brier,
                    experiment.bootstrap_repetitions,
                    np.random.default_rng(
                        derive_seed(
                            experiment.master_seed,
                            "MECHCONF.metrics.descriptive.brier",
                            *seed_parts,
                        )
                    ),
                )
                descriptive_rows.append(
                    {
                        "contrast": contrast,
                        "baseline": baseline_name,
                        "enhanced": enhanced_name,
                        "candidate": candidate,
                        "direction": direction,
                        "log_loss_gain": log_gain,
                        "log_loss_gain_ci95": log_ci,
                        "q_brier_gain": brier_gain,
                        "q_brier_gain_ci95": brier_ci,
                    }
                )

    adjusted = holm_adjust([row["randomization_p_raw"] for row in primary_rows])
    for row, adjusted_value in zip(primary_rows, adjusted):
        row["randomization_p_holm"] = adjusted_value
        row["passes_gate"] = bool(
            row["log_loss_gain"] > 0.0
            and row["log_loss_gain_ci95"][0] > 0.0
            and adjusted_value < 0.05
        )
    support = {
        contrast: all(
            row["passes_gate"]
            for row in primary_rows
            if row["contrast"] == contrast
        )
        for contrast in PRIMARY_CONTRASTS
    }
    metrics["primary_tests"] = primary_rows
    metrics["descriptive_tests"] = descriptive_rows
    metrics["support"] = support
    metrics["family_size"] = len(primary_rows)
    metrics["decision_rule"] = (
        "positive paired log-loss gain, matrix-bootstrap lower 95% bound > 0, "
        "and Holm-adjusted paired whole-matrix randomization p < 0.05 in both "
        "candidates and both preassigned branch-half directions"
    )
    return metrics
