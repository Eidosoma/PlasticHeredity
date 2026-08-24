"""Whole-matrix inference for the clean-room intervention campaigns."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .experiment import StateCase
from .mechanistic_metrics import holm_adjust

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _matrix_means(
    values: FloatArray, matrix_ids: IntArray, matrix_order: IntArray
) -> FloatArray:
    return np.asarray(
        [np.mean(values[matrix_ids == matrix_id]) for matrix_id in matrix_order],
        dtype=np.float64,
    )


def _interval(samples: FloatArray, alpha: float = 0.05) -> tuple[float, float]:
    lower, upper = np.quantile(samples, (alpha / 2.0, 1.0 - alpha / 2.0))
    return float(lower), float(upper)


def _bootstrap_means(matrix_values: FloatArray, indices: IntArray) -> FloatArray:
    return np.asarray(matrix_values[indices].mean(axis=1), dtype=np.float64)


def _one_sided_sign_p(
    matrix_values: FloatArray, signs: FloatArray
) -> tuple[float, FloatArray]:
    observed = float(matrix_values.mean())
    null = np.asarray(signs @ matrix_values / matrix_values.size, dtype=np.float64)
    value = float((np.count_nonzero(null >= observed) + 1) / (null.size + 1))
    return value, null


def _bernoulli_scores(y: FloatArray, prediction: FloatArray) -> dict[str, float]:
    truth = np.asarray(y, dtype=np.float64)
    probability = np.clip(np.asarray(prediction, dtype=np.float64), 1e-12, 1 - 1e-12)
    if probability.ndim == 1:
        probability = probability[:, None]
    return {
        "log_loss": float(
            np.mean(
                -(truth * np.log(probability) + (1.0 - truth) * np.log(1.0 - probability))
            )
        ),
        "brier": float(np.mean((truth - probability) ** 2)),
    }


def _maximum_leave_one_out_influence(matrix_values: FloatArray) -> float:
    if matrix_values.size <= 1:
        return float("nan")
    observed = float(matrix_values.mean())
    leave_one_out = (
        matrix_values.sum() - matrix_values
    ) / (matrix_values.size - 1)
    return float(np.max(np.abs(leave_one_out - observed)))


def generate_inference_draws(
    matrix_count: int,
    bootstrap_repetitions: int,
    randomization_repetitions: int,
    bootstrap_rng: np.random.Generator,
    randomization_rng: np.random.Generator,
) -> dict[str, NDArray]:
    """Generate draws once so every candidate/half uses identical matrix draws."""

    if matrix_count < 2:
        raise ValueError("matrix-block inference requires at least two matrices")
    bootstrap_indices = bootstrap_rng.integers(
        0,
        matrix_count,
        size=(bootstrap_repetitions, matrix_count),
        dtype=np.int64,
    )
    signs = randomization_rng.integers(
        0,
        2,
        size=(randomization_repetitions, matrix_count),
        dtype=np.int8,
    )
    randomization_signs = signs.astype(np.float64) * 2.0 - 1.0
    return {
        "bootstrap_indices": bootstrap_indices,
        "randomization_signs": randomization_signs,
    }


def _validate_inputs(
    cases: list[StateCase],
    arm_names: tuple[str, ...],
    targets: NDArray,
    predictions: NDArray,
    branches_per_state: int,
) -> None:
    expected_targets = (len(cases), len(arm_names), branches_per_state)
    if np.asarray(targets).shape != expected_targets:
        raise ValueError(
            f"target shape {np.asarray(targets).shape} differs from {expected_targets}"
        )
    if np.asarray(predictions).shape != (len(cases), len(arm_names)):
        raise ValueError("prediction table does not align with cases and arms")
    if branches_per_state < 2 or branches_per_state % 2:
        raise ValueError("fixed branch halves require a positive even branch count")
    if len(set(arm_names)) != len(arm_names):
        raise ValueError("arm names must be unique")


def compute_one_shot_inference(
    cases: list[StateCase],
    arm_names: tuple[str, ...],
    targets: NDArray,
    predictions: NDArray,
    draws: dict[str, NDArray],
    *,
    up_arm: str,
    down_arm: str,
    random_arm: str = "RANDOM",
    noop_arm: str = "NOOP",
    equivalence_margin: float = 0.025,
    random_ratio_limit: float = 0.25,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compute the registered four-cell causal contrast and specificity gates.

    Every observation is first reduced to a state probability and then to a
    whole-matrix mean.  The same sealed bootstrap indices and sign vectors are
    reused in all four primary cells.
    """

    targets_array = np.asarray(targets, dtype=np.float64)
    predictions_array = np.asarray(predictions, dtype=np.float64)
    branches = int(targets_array.shape[2]) if targets_array.ndim == 3 else -1
    _validate_inputs(cases, arm_names, targets_array, predictions_array, branches)
    required = {up_arm, down_arm, random_arm, noop_arm}
    if not required.issubset(arm_names):
        raise ValueError(f"missing registered arms: {sorted(required - set(arm_names))}")
    arm_index = {name: index for index, name in enumerate(arm_names)}
    half_size = branches // 2
    matrix_order = np.unique(
        np.asarray([case.matrix_id for case in cases], dtype=np.int64)
    )
    bootstrap_indices = np.asarray(draws["bootstrap_indices"], dtype=np.int64)
    signs = np.asarray(draws["randomization_signs"], dtype=np.float64)
    expected_draw_width = matrix_order.size
    if (
        bootstrap_indices.ndim != 2
        or signs.ndim != 2
        or bootstrap_indices.shape[1] != expected_draw_width
        or signs.shape[1] != expected_draw_width
        or np.any(bootstrap_indices < 0)
        or np.any(bootstrap_indices >= expected_draw_width)
        or not np.isin(signs, (-1.0, 1.0)).all()
    ):
        raise ValueError("inference draws do not preserve the sealed matrix blocks")

    cells: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    raw_p_values: list[float] = []
    cell_bootstraps: dict[str, dict[str, list[float]]] = {}
    cell_randomization: dict[str, list[float]] = {}

    for candidate in ("02", "03"):
        selected = np.asarray(
            [case.candidate == candidate for case in cases], dtype=bool
        )
        if not selected.any():
            raise ValueError(f"candidate {candidate} is absent")
        selected_cases = [case for case in cases if case.candidate == candidate]
        ids = np.asarray([case.matrix_id for case in selected_cases], dtype=np.int64)
        if not np.array_equal(np.unique(ids), matrix_order):
            raise ValueError(f"candidate {candidate} lacks a complete matrix cohort")
        candidate_targets = targets_array[selected]
        candidate_predictions = predictions_array[selected]
        for half, branch_slice in (
            ("A", slice(0, half_size)),
            ("B", slice(half_size, branches)),
        ):
            q = candidate_targets[:, :, branch_slice].mean(axis=2)
            state_effects = {
                "up_minus_down": q[:, arm_index[up_arm]] - q[:, arm_index[down_arm]],
                "up_minus_noop": q[:, arm_index[up_arm]] - q[:, arm_index[noop_arm]],
                "noop_minus_down": q[:, arm_index[noop_arm]] - q[:, arm_index[down_arm]],
                "random_minus_noop": q[:, arm_index[random_arm]] - q[:, arm_index[noop_arm]],
            }
            matrix_effects = {
                name: _matrix_means(values, ids, matrix_order)
                for name, values in state_effects.items()
            }
            bootstraps = {
                name: _bootstrap_means(values, bootstrap_indices)
                for name, values in matrix_effects.items()
            }
            cell_key = f"c{candidate}_{half}"
            cell_bootstraps[cell_key] = {
                name: values.tolist() for name, values in bootstraps.items()
            }
            p_value, null = _one_sided_sign_p(matrix_effects["up_minus_down"], signs)
            raw_p_values.append(p_value)
            cell_randomization[cell_key] = null.tolist()

            arms: dict[str, Any] = {}
            for name in arm_names:
                index = arm_index[name]
                state_q = q[:, index]
                matrix_q = _matrix_means(state_q, ids, matrix_order)
                arm_bootstrap = _bootstrap_means(matrix_q, bootstrap_indices)
                expanded_prediction = candidate_predictions[:, index]
                arms[name] = {
                    "mean_probability": float(matrix_q.mean()),
                    "bootstrap_ci95": _interval(arm_bootstrap),
                    "branch_scores": _bernoulli_scores(
                        candidate_targets[:, index, branch_slice],
                        expanded_prediction,
                    ),
                    "mean_frozen_prediction": float(expanded_prediction.mean()),
                }

            contrasts: dict[str, Any] = {}
            for name, values in matrix_effects.items():
                contrasts[name] = {
                    "estimate": float(values.mean()),
                    "bootstrap_ci95": _interval(bootstraps[name]),
                    "matrices_expected_sign": int(np.count_nonzero(values > 0.0)),
                    "matrices_zero": int(np.count_nonzero(values == 0.0)),
                    "maximum_leave_one_matrix_out_influence": (
                        _maximum_leave_one_out_influence(values)
                    ),
                }
            random_ci90 = _interval(
                bootstraps["random_minus_noop"], alpha=0.10
            )
            random_difference = contrasts["random_minus_noop"]["estimate"]
            up_down = contrasts["up_minus_down"]["estimate"]
            tost_equivalent = bool(
                random_ci90[0] > -equivalence_margin
                and random_ci90[1] < equivalence_margin
            )
            ratio_specific = bool(
                up_down > 0.0
                and abs(random_difference) <= random_ratio_limit * up_down
            )
            predicted_shift = (
                candidate_predictions[:, arm_index[up_arm]]
                - candidate_predictions[:, arm_index[down_arm]]
            )
            realized_shift = state_effects["up_minus_down"]
            predicted_centered = predicted_shift - np.asarray(
                [predicted_shift[ids == key].mean() for key in ids]
            )
            realized_centered = realized_shift - np.asarray(
                [realized_shift[ids == key].mean() for key in ids]
            )
            denominator = float(np.dot(predicted_centered, predicted_centered))
            slope = (
                float(np.dot(predicted_centered, realized_centered) / denominator)
                if denominator > 0.0
                else float("nan")
            )
            cell = {
                "cell": cell_key,
                "candidate": candidate,
                "branch_half": half,
                "branch_range": [
                    int(branch_slice.start),
                    int(branch_slice.stop - 1),
                ],
                "states": int(selected.sum()),
                "matrices": int(matrix_order.size),
                "arms": arms,
                "contrasts": contrasts,
                "up_down_randomization_p_raw": p_value,
                "random_noop_equivalence": {
                    "margin": equivalence_margin,
                    "bootstrap_ci90": random_ci90,
                    "tost_equivalent": tost_equivalent,
                    "ratio_limit": random_ratio_limit,
                    "absolute_difference_within_ratio": ratio_specific,
                },
                "predicted_versus_realized": {
                    "mean_predicted_up_minus_down": float(predicted_shift.mean()),
                    "mean_realized_up_minus_down": float(realized_shift.mean()),
                    "state_centered_slope": slope,
                },
            }
            cells.append(cell)
            for position, matrix_id in enumerate(matrix_order):
                row: dict[str, Any] = {
                    "cell": cell_key,
                    "candidate": candidate,
                    "branch_half": half,
                    "matrix_id": int(matrix_id),
                }
                row.update(
                    {
                        name: float(values[position])
                        for name, values in matrix_effects.items()
                    }
                )
                matrix_rows.append(row)

    adjusted = holm_adjust(raw_p_values)
    for cell, adjusted_p in zip(cells, adjusted, strict=True):
        contrasts = cell["contrasts"]
        cell["up_down_randomization_p_holm"] = float(adjusted_p)
        cell["registered_gates"] = {
            "up_minus_down_positive": contrasts["up_minus_down"]["estimate"] > 0.0,
            "up_minus_down_bootstrap_lower_positive": (
                contrasts["up_minus_down"]["bootstrap_ci95"][0] > 0.0
            ),
            "holm_randomization_below_0_05": adjusted_p < 0.05,
            "up_minus_noop_bootstrap_lower_positive": (
                contrasts["up_minus_noop"]["bootstrap_ci95"][0] > 0.0
            ),
            "noop_minus_down_bootstrap_lower_positive": (
                contrasts["noop_minus_down"]["bootstrap_ci95"][0] > 0.0
            ),
            "random_tost_equivalent_to_noop": cell["random_noop_equivalence"][
                "tost_equivalent"
            ],
            "random_absolute_difference_within_effect_ratio": cell[
                "random_noop_equivalence"
            ]["absolute_difference_within_ratio"],
        }
        cell["registered_cell_pass"] = bool(
            all(cell["registered_gates"].values())
        )
        cell["pilot_eligibility_gates"] = {
            "up_minus_down_positive": contrasts["up_minus_down"]["estimate"] > 0.0,
            "random_noop_point_within_margin": (
                abs(contrasts["random_minus_noop"]["estimate"])
                <= equivalence_margin
            ),
        }

    landmark_rows: list[dict[str, Any]] = []
    for candidate in ("02", "03"):
        for landmark in sorted({case.landmark for case in cases}):
            selected = np.asarray(
                [
                    case.candidate == candidate and case.landmark == landmark
                    for case in cases
                ],
                dtype=bool,
            )
            for half, branch_slice in (
                ("A", slice(0, half_size)),
                ("B", slice(half_size, branches)),
            ):
                q = targets_array[selected, :, branch_slice].mean(axis=2)
                landmark_rows.append(
                    {
                        "candidate": candidate,
                        "branch_half": half,
                        "landmark": int(landmark),
                        "up_minus_down": float(
                            np.mean(q[:, arm_index[up_arm]] - q[:, arm_index[down_arm]])
                        ),
                        "up_minus_noop": float(
                            np.mean(q[:, arm_index[up_arm]] - q[:, arm_index[noop_arm]])
                        ),
                        "noop_minus_down": float(
                            np.mean(q[:, arm_index[noop_arm]] - q[:, arm_index[down_arm]])
                        ),
                        "random_minus_noop": float(
                            np.mean(q[:, arm_index[random_arm]] - q[:, arm_index[noop_arm]])
                        ),
                    }
                )

    pilot_directional = all(
        bool(cell["pilot_eligibility_gates"]["up_minus_down_positive"])
        for cell in cells
    )
    pilot_random = all(
        bool(cell["pilot_eligibility_gates"]["random_noop_point_within_margin"])
        for cell in cells
    )
    result: dict[str, Any] = {
        "inference_unit": "whole catalytic matrix",
        "state_replicates_within_matrix_kept_together": True,
        "shared_bootstrap_draws_across_cells": True,
        "shared_randomization_signs_across_cells": True,
        "bootstrap_repetitions": int(bootstrap_indices.shape[0]),
        "randomization_repetitions": int(signs.shape[0]),
        "equivalence_margin": equivalence_margin,
        "cells": cells,
        "holm_family_size": len(cells),
        "registered_all_four_cells_pass": bool(
            all(cell["registered_cell_pass"] for cell in cells)
        ),
        "pilot_eligibility_without_replay": bool(pilot_directional and pilot_random),
        "pilot_eligibility_components": {
            "up_down_positive_in_all_four_cells": pilot_directional,
            "random_noop_point_within_margin_in_all_four_cells": pilot_random,
        },
        "landmark_effects": landmark_rows,
        "stored_inference_arrays": {
            "bootstrap_indices_shape": list(bootstrap_indices.shape),
            "randomization_signs_shape": list(signs.shape),
            "cell_bootstrap_effects": cell_bootstraps,
            "cell_randomization_nulls": cell_randomization,
        },
    }
    return result, matrix_rows

