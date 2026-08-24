"""Matrix-aware tests for the beta-complete prospective correction."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .config import CANDIDATES, ExperimentConfig
from .experiment import StateCase, _candidate_indices
from .mechanistic_metrics import (
    _paired_gain,
    _rank_metrics,
    _reliability_bootstrap,
    _state_brier,
    _state_log_loss,
    holm_adjust,
    paired_matrix_randomization_p,
)
from .seeds import derive_seed

FloatArray = NDArray[np.float64]

PRIMARY_CONTRASTS = {
    "state": ("h10", "h10_state"),
    "network": ("h10_state", "h10_state_beta"),
    "interaction": ("h10_state_beta", "h10_state_beta_interaction"),
}
DESCRIPTIVE_CONTRASTS = {
    "beta_beyond_history": ("h10", "h10_beta"),
}


def compute_mechanistic_v2_metrics(
    cases: list[StateCase],
    labels: NDArray[np.int8],
    predictions: dict[str, dict[str, FloatArray]],
    experiment: ExperimentConfig,
    seed_namespace: str,
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int8)
    if labels.shape != (len(cases), experiment.confirmation.branches_per_state):
        raise ValueError("v2 labels do not match the registered confirmation design")
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
                experiment.master_seed,
                f"{seed_namespace}.metrics.reliability",
                candidate,
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
        centered, centered_ci = _reliability_bootstrap(
            q_a,
            q_b,
            matrix_ids,
            True,
            experiment.bootstrap_repetitions,
            reliability_rng,
        )
        metrics["candidates"][candidate] = {
            "states": int(selected.size),
            "transition_region_states": int(((q_all > 0.1) & (q_all < 0.9)).sum()),
            "branch_half_reliability": reliability,
            "branch_half_reliability_ci95": reliability_ci,
            "centered_branch_half_reliability": centered,
            "centered_branch_half_reliability_ci95": centered_ci,
            "models": {
                name: _rank_metrics(prediction, q_a, q_b, matrix_ids)
                for name, prediction in predictions[candidate].items()
            },
        }

        for family, contrasts, rows in (
            ("primary", PRIMARY_CONTRASTS, primary_rows),
            ("descriptive", DESCRIPTIVE_CONTRASTS, descriptive_rows),
        ):
            for contrast, (baseline_name, enhanced_name) in contrasts.items():
                baseline = predictions[candidate][baseline_name]
                enhanced = predictions[candidate][enhanced_name]
                for direction, q in (("A", q_a), ("B", q_b)):
                    parts = (contrast, candidate, direction)
                    gain, interval = _paired_gain(
                        q,
                        baseline,
                        enhanced,
                        matrix_ids,
                        _state_log_loss,
                        experiment.bootstrap_repetitions,
                        np.random.default_rng(
                            derive_seed(
                                experiment.master_seed,
                                f"{seed_namespace}.metrics.{family}.bootstrap.log_loss",
                                *parts,
                            )
                        ),
                    )
                    brier, brier_interval = _paired_gain(
                        q,
                        baseline,
                        enhanced,
                        matrix_ids,
                        _state_brier,
                        experiment.bootstrap_repetitions,
                        np.random.default_rng(
                            derive_seed(
                                experiment.master_seed,
                                f"{seed_namespace}.metrics.{family}.bootstrap.brier",
                                *parts,
                            )
                        ),
                    )
                    row: dict[str, Any] = {
                        "contrast": contrast,
                        "baseline": baseline_name,
                        "enhanced": enhanced_name,
                        "candidate": candidate,
                        "direction": direction,
                        "log_loss_gain": gain,
                        "log_loss_gain_ci95": interval,
                        "q_brier_gain": brier,
                        "q_brier_gain_ci95": brier_interval,
                    }
                    if family == "primary":
                        row["randomization_p_raw"] = paired_matrix_randomization_p(
                            q,
                            baseline,
                            enhanced,
                            matrix_ids,
                            experiment.permutation_repetitions,
                            np.random.default_rng(
                                derive_seed(
                                    experiment.master_seed,
                                    f"{seed_namespace}.metrics.primary.randomization",
                                    *parts,
                                )
                            ),
                        )
                    rows.append(row)

    adjusted = holm_adjust([row["randomization_p_raw"] for row in primary_rows])
    for row, adjusted_value in zip(primary_rows, adjusted):
        row["randomization_p_holm"] = adjusted_value
        row["passes_gate"] = bool(
            row["log_loss_gain"] > 0.0
            and row["log_loss_gain_ci95"][0] > 0.0
            and adjusted_value < 0.05
        )
    metrics["primary_tests"] = primary_rows
    metrics["descriptive_tests"] = descriptive_rows
    metrics["support"] = {
        contrast: all(
            row["passes_gate"]
            for row in primary_rows
            if row["contrast"] == contrast
        )
        for contrast in PRIMARY_CONTRASTS
    }
    metrics["family_size"] = len(primary_rows)
    metrics["decision_rule"] = (
        "positive paired log-loss gain, matrix-bootstrap lower 95% bound > 0, "
        "and Holm-adjusted whole-matrix randomization p < 0.05 in both candidates "
        "and both preassigned branch-half directions"
    )
    return metrics


def independently_recompute_primary_gains(
    cases: list[StateCase],
    labels: NDArray[np.int8],
    predictions: dict[str, dict[str, FloatArray]],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    expected = {
        (row["contrast"], row["candidate"], row["direction"]): float(
            row["log_loss_gain"]
        )
        for row in metrics["primary_tests"]
    }
    errors: dict[str, float] = {}
    for candidate in CANDIDATES:
        selected = _candidate_indices(cases, candidate)
        candidate_labels = labels[selected]
        split = candidate_labels.shape[1] // 2
        matrix_ids = np.asarray([cases[index].matrix_id for index in selected])
        for direction, q in (
            ("A", candidate_labels[:, :split].mean(axis=1)),
            ("B", candidate_labels[:, split:].mean(axis=1)),
        ):
            for contrast, (baseline_name, enhanced_name) in PRIMARY_CONTRASTS.items():
                difference = _state_log_loss(
                    q, predictions[candidate][baseline_name]
                ) - _state_log_loss(q, predictions[candidate][enhanced_name])
                matrix_values = np.asarray(
                    [difference[matrix_ids == key].mean() for key in np.unique(matrix_ids)]
                )
                observed = float(matrix_values.mean())
                key = (contrast, candidate, direction)
                errors["/".join(key)] = abs(observed - expected[key])
    maximum = max(errors.values(), default=0.0)
    return {
        "maximum_absolute_error": maximum,
        "all_within_1e-14": maximum <= 1e-14,
        "errors": errors,
    }
