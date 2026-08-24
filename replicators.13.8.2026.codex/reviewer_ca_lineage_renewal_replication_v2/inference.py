"""Pair-cluster inference for strict and complete secondary v2 endpoints."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .contract import CONDITIONS, CONTRACT, NAMESPACE, semantic_seed


PROBABILITY_KEYS = ("p_a_given_a", "p_a_given_b", "p_b_given_a", "p_b_given_b")


def _finite_vector(values: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array):
        raise ValueError(f"{name} must be a nonempty vector")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains a nonfinite value")
    return array


def _bootstrap_mean(
    values: Sequence[float],
    *,
    resamples: int,
    alpha: float,
    seed_parts: tuple[object, ...],
) -> list[float]:
    array = _finite_vector(values, "bootstrap values")
    rng = np.random.default_rng(semantic_seed(NAMESPACE, *seed_parts, "bootstrap"))
    draws = rng.integers(0, len(array), size=(resamples, len(array)))
    estimates = array[draws].mean(axis=1)
    return [
        float(np.quantile(estimates, alpha / 2.0)),
        float(np.quantile(estimates, 1.0 - alpha / 2.0)),
    ]


def summarize_scalar(
    values: Sequence[float],
    *,
    resamples: int,
    alpha: float,
    seed_parts: tuple[object, ...],
) -> dict[str, Any]:
    array = _finite_vector(values, "scalar values")
    return {
        "mean": float(array.mean()),
        "ci": _bootstrap_mean(
            array, resamples=resamples, alpha=alpha, seed_parts=seed_parts
        ),
        "alpha": alpha,
        "n_pairs": len(array),
    }


def summarize_assignments(
    rows: Sequence[Mapping[str, float]],
    *,
    resamples: int,
    alpha: float,
    seed_parts: tuple[object, ...],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty outcome set")
    for row in rows:
        _finite_vector([float(row[key]) for key in (*PROBABILITY_KEYS, "direction_a", "direction_b", "crossover", "correct", "resolved")], "assignment row")
    summary = {
        key: float(np.mean([float(row[key]) for row in rows]))
        for key in (*PROBABILITY_KEYS, "direction_a", "direction_b", "correct", "resolved")
    }
    pair_crossovers = [float(row["crossover"]) for row in rows]
    summary.update(
        {
            "mean": float(np.mean(pair_crossovers)),
            "ci": _bootstrap_mean(
                pair_crossovers,
                resamples=resamples,
                alpha=alpha,
                seed_parts=seed_parts,
            ),
            "alpha": alpha,
            "n_pairs": len(rows),
            "fraction_pairs_positive": float(
                np.mean(np.asarray(pair_crossovers) > 0.0)
            ),
        }
    )
    return summary


def paired_scalar_advantage(
    intact: Sequence[float],
    control: Sequence[float],
    *,
    resamples: int,
    alpha: float,
    seed_parts: tuple[object, ...],
) -> dict[str, Any]:
    left = _finite_vector(intact, "intact paired values")
    right = _finite_vector(control, "control paired values")
    if left.shape != right.shape:
        raise ValueError("paired vectors must have equal nonzero lengths")
    return summarize_scalar(
        left - right,
        resamples=resamples,
        alpha=alpha,
        seed_parts=seed_parts,
    )


def paired_assignment_advantage(
    intact: Sequence[Mapping[str, float]],
    control: Sequence[Mapping[str, float]],
    *,
    resamples: int,
    alpha: float,
    seed_parts: tuple[object, ...],
) -> dict[str, Any]:
    if len(intact) != len(control) or not intact:
        raise ValueError("paired assignment conditions must have equal nonzero counts")
    return paired_scalar_advantage(
        [float(row["crossover"]) for row in intact],
        [float(row["crossover"]) for row in control],
        resamples=resamples,
        alpha=alpha,
        seed_parts=seed_parts,
    )


def safe_fraction(numerator: float, denominator: float) -> dict[str, Any]:
    numerator = float(numerator)
    denominator = float(denominator)
    if not np.isfinite([numerator, denominator]).all():
        raise ValueError("ratio inputs must be finite")
    if denominator <= 0.0:
        return {
            "defined": False,
            "value": None,
            "numerator": numerator,
            "denominator": denominator,
            "reason": "non_positive_denominator",
        }
    return {
        "defined": True,
        "value": numerator / denominator,
        "numerator": numerator,
        "denominator": denominator,
        "reason": None,
    }


def _ratio_at_least(ratio: Mapping[str, Any], threshold: float) -> bool:
    return bool(ratio["defined"] and float(ratio["value"]) >= threshold)


def _rows(
    payloads: Sequence[Mapping[str, Any]],
    condition: str,
    generation: int,
    observer: str = "primary",
) -> list[Mapping[str, float]]:
    return [
        payload["conditions"][condition]["outcomes"][str(generation)][observer]
        for payload in payloads
    ]


def _survival(
    payloads: Sequence[Mapping[str, Any]], condition: str, generation: int
) -> list[float]:
    return [
        float(payload["conditions"][condition]["outcomes"][str(generation)]["survival"])
        for payload in payloads
    ]


def _carrier_values(
    payloads: Sequence[Mapping[str, Any]],
    condition: str,
    generation: int,
    boundary: str,
    metric: str,
) -> list[float]:
    return [
        float(
            payload["conditions"][condition]["carrier_history"][str(generation)][
                boundary
            ][metric]
        )
        for payload in payloads
    ]


def _decoder_panel(
    payloads: Sequence[Mapping[str, Any]],
    kind: str,
    *,
    resamples: int,
    alpha: float,
) -> dict[str, Any]:
    names = (
        "intact",
        "no_rewrite",
        "read_disabled",
        "ablate_after_g2",
        "rescue_same_enter_g4",
        "rescue_opposite_enter_g4",
    )
    values = {
        name: [
            float(payload["secondary_decoder"][kind][name]) for payload in payloads
        ]
        for name in names
    }
    summaries = {
        name: summarize_scalar(
            condition_values,
            resamples=resamples,
            alpha=alpha,
            seed_parts=("secondary", kind, name, 16),
        )
        for name, condition_values in values.items()
    }
    active_advantage = paired_scalar_advantage(
        values["intact"],
        values["no_rewrite"],
        resamples=resamples,
        alpha=alpha,
        seed_parts=("secondary", kind, "active-over-no-rewrite", 16),
    )
    read_advantage = paired_scalar_advantage(
        values["intact"],
        values["read_disabled"],
        resamples=resamples,
        alpha=alpha,
        seed_parts=("secondary", kind, "read-over-disabled", 16),
    )
    restoration = safe_fraction(
        summaries["rescue_same_enter_g4"]["mean"] - 0.5,
        summaries["intact"]["mean"] - 0.5,
    )
    gates = {
        "intact_generation16": bool(
            summaries["intact"]["mean"] >= CONTRACT["decoder_mean_gate"]
            and summaries["intact"]["ci"][0] > CONTRACT["decoder_lower_gate"]
        ),
        "active_advantage_over_no_rewrite": bool(
            active_advantage["mean"] >= CONTRACT["decoder_advantage"]
            and active_advantage["ci"][0] > 0.0
        ),
        "read_advantage": bool(
            read_advantage["mean"] >= CONTRACT["decoder_advantage"]
            and read_advantage["ci"][0] > 0.0
        ),
        "ablation_null": bool(
            summaries["ablate_after_g2"]["mean"]
            <= CONTRACT["decoder_null_ceiling"]
        ),
        "same_history_rescue": _ratio_at_least(
            restoration, float(CONTRACT["rescue_fraction"])
        ),
        "opposite_history_reversal": bool(
            summaries["rescue_opposite_enter_g4"]["mean"]
            >= CONTRACT["decoder_mean_gate"]
            and summaries["rescue_opposite_enter_g4"]["ci"][0]
            > CONTRACT["decoder_lower_gate"]
        ),
    }
    return {
        "generation": 16,
        "conditions": summaries,
        "active_advantage_over_no_rewrite": active_advantage,
        "read_advantage": read_advantage,
        "rescue_restoration_excess_over_chance": restoration,
        "opposite_rescue_scoring": (
            "target-free within-condition history separation after opposite rescue"
        ),
        "gates": gates,
        "passed": all(gates.values()),
    }


def adjudicate(
    pair_payloads: Sequence[Mapping[str, Any]],
    *,
    complete: bool,
    expected_pairs: int,
    resamples: int = 10_000,
) -> dict[str, Any]:
    alpha = float(CONTRACT["confirmation_alpha"])
    checkpoints = (1, 2, 4, 8, 16)
    if not pair_payloads:
        raise ValueError("no confirmation payloads supplied")
    pair_ids = [str(payload["pair_id"]) for payload in pair_payloads]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("duplicate independent pair payload")
    complete = bool(complete and len(pair_payloads) == expected_pairs)

    conditions: dict[str, Any] = {}
    for condition in CONDITIONS:
        conditions[condition] = {}
        for generation in checkpoints:
            conditions[condition][str(generation)] = {
                "primary": summarize_assignments(
                    _rows(pair_payloads, condition, generation),
                    resamples=resamples,
                    alpha=alpha,
                    seed_parts=("condition", condition, generation, "primary"),
                ),
                "terminal": summarize_assignments(
                    _rows(pair_payloads, condition, generation, "terminal"),
                    resamples=resamples,
                    alpha=alpha,
                    seed_parts=("condition", condition, generation, "terminal"),
                ),
                "survival": summarize_scalar(
                    _survival(pair_payloads, condition, generation),
                    resamples=resamples,
                    alpha=alpha,
                    seed_parts=("condition", condition, generation, "survival"),
                ),
            }

    intact4 = conditions["intact"]["4"]["primary"]
    intact8 = conditions["intact"]["8"]["primary"]
    intact16 = conditions["intact"]["16"]["primary"]
    terminal8 = conditions["intact"]["8"]["terminal"]
    survival8 = conditions["intact"]["8"]["survival"]
    survival16 = conditions["intact"]["16"]["survival"]
    intact8_rows = _rows(pair_payloads, "intact", 8)
    controls = (
        "zero_every_boundary",
        "shuffle_every_boundary",
        "read_disabled",
        "founder_write_disabled",
    )
    control_advantages = {
        control: paired_assignment_advantage(
            intact8_rows,
            _rows(pair_payloads, control, 8),
            resamples=resamples,
            alpha=alpha,
            seed_parts=("advantage", "intact", control, 8),
        )
        for control in controls
    }
    active_rewrite_advantage = paired_assignment_advantage(
        intact8_rows,
        _rows(pair_payloads, "no_rewrite", 8),
        resamples=resamples,
        alpha=alpha,
        seed_parts=("advantage", "intact", "no_rewrite", 8),
    )
    rescue_advantage = paired_assignment_advantage(
        _rows(pair_payloads, "rescue_same_enter_g4", 4),
        _rows(pair_payloads, "ablate_after_g2", 4),
        resamples=resamples,
        alpha=alpha,
        seed_parts=("advantage", "rescue_same", "ablation", 4),
    )
    no_rewrite8 = conditions["no_rewrite"]["8"]["primary"]
    ablation4 = conditions["ablate_after_g2"]["4"]["primary"]
    rescue4 = conditions["rescue_same_enter_g4"]["4"]["primary"]
    opposite_rescue4 = conditions["rescue_opposite_enter_g4"]["4"]["primary"]
    opposite_founder8 = conditions["opposite_founder"]["8"]["primary"]
    corruption8 = conditions["carrier_corruption_1"]["8"]["primary"]
    no_rewrite_loss = safe_fraction(
        intact8["mean"] - no_rewrite8["mean"], intact8["mean"]
    )
    ablation_loss = safe_fraction(
        intact4["mean"] - ablation4["mean"], intact4["mean"]
    )
    rescue_restoration = safe_fraction(rescue4["mean"], intact4["mean"])

    strict_gates = {
        f"complete_{expected_pairs}_pairs": complete,
        "generation4_original_form": bool(
            intact4["mean"] >= CONTRACT["primary_crossover_generation4"]
            and intact4["ci"][0] > 0.0
        ),
        "generation8_original_form": bool(
            intact8["mean"] >= CONTRACT["primary_crossover_generation8"]
            and intact8["ci"][0] > 0.0
        ),
        "generation16_original_form": bool(
            intact16["mean"] >= CONTRACT["durable_crossover_generation16"]
            and intact16["ci"][0] > 0.0
        ),
        "directions_and_pair_prevalence": bool(
            intact8["direction_a"] > 0.0
            and intact8["direction_b"] > 0.0
            and intact8["fraction_pairs_positive"] >= 0.5
        ),
        "survival": bool(
            survival8["mean"] >= CONTRACT["survival_gate"]
            and survival16["mean"] >= CONTRACT["survival_gate"]
        ),
        "terminal_observer": bool(
            terminal8["mean"] >= CONTRACT["durable_crossover_generation16"]
            and terminal8["ci"][0] > 0.0
        ),
        "four_controls": bool(
            all(
                value["mean"] >= CONTRACT["control_advantage"]
                and value["ci"][0] > 0.0
                for value in control_advantages.values()
            )
        ),
        "active_rewrite": bool(
            _ratio_at_least(no_rewrite_loss, float(CONTRACT["loss_fraction"]))
            and active_rewrite_advantage["mean"] >= CONTRACT["control_advantage"]
            and active_rewrite_advantage["ci"][0] > 0.0
        ),
        "ablation": _ratio_at_least(ablation_loss, float(CONTRACT["loss_fraction"])),
        "same_history_rescue": bool(
            _ratio_at_least(rescue_restoration, float(CONTRACT["rescue_fraction"]))
            and rescue_advantage["mean"] >= CONTRACT["control_advantage"]
            and rescue_advantage["ci"][0] > 0.0
        ),
        "opposite_history_rescue": bool(
            opposite_rescue4["mean"] <= -CONTRACT["control_advantage"]
            and opposite_rescue4["ci"][1] < 0.0
        ),
        "opposite_founder": bool(
            opposite_founder8["mean"] <= -CONTRACT["control_advantage"]
            and opposite_founder8["ci"][1] < 0.0
        ),
        "carrier_corruption": bool(
            corruption8["mean"] >= CONTRACT["corruption_crossover"]
            and corruption8["ci"][0] > 0.0
        ),
    }
    strict_passed = all(strict_gates.values())
    carrier_decoder = _decoder_panel(
        pair_payloads, "carrier", resamples=resamples, alpha=alpha
    )
    phenotype_decoder = _decoder_panel(
        pair_payloads, "phenotype", resamples=resamples, alpha=alpha
    )

    if not complete:
        verdict = "INCOMPLETE"
    elif strict_passed:
        verdict = "STRICT_RENEWED_CA_PLASTIC_HEREDITY"
    elif carrier_decoder["passed"] and phenotype_decoder["passed"]:
        verdict = "EXPRESSED_DRIFTED_LINEAGE_HEREDITY"
    elif carrier_decoder["passed"]:
        verdict = "CRYPTIC_RENEWED_CARRIER_MEMORY"
    elif (
        strict_gates["generation8_original_form"]
        and not _ratio_at_least(no_rewrite_loss, float(CONTRACT["loss_fraction"]))
    ):
        verdict = "STATIC_HIDDEN_TEMPLATE"
    elif (
        strict_gates["generation8_original_form"]
        and not strict_gates["generation16_original_form"]
    ):
        verdict = "TRANSIENT_LINEAGE_MEMORY"
    else:
        verdict = "NO_DURABLE_RENEWAL"

    carrier_renewal: dict[str, Any] = {}
    for generation in checkpoints:
        carrier_renewal[str(generation)] = {
            "intact_exit_centroid_l2": summarize_scalar(
                _carrier_values(pair_payloads, "intact", generation, "exit", "centroid_l2"),
                resamples=resamples,
                alpha=alpha,
                seed_parts=("carrier", "intact", generation, "exit", "centroid_l2"),
            ),
            "no_rewrite_entry_mean_abs": summarize_scalar(
                _carrier_values(pair_payloads, "no_rewrite", generation, "entry", "mean_abs"),
                resamples=resamples,
                alpha=alpha,
                seed_parts=("carrier", "no_rewrite", generation, "entry", "mean_abs"),
            ),
        }

    return {
        "state": "complete" if complete else "incomplete",
        "strict_primary_passed": strict_passed,
        "verdict": verdict,
        "alpha": alpha,
        "n_pairs": len(pair_payloads),
        "strict_gates": strict_gates,
        "intact_generation4": intact4,
        "intact_generation8": intact8,
        "intact_generation16": intact16,
        "terminal_generation8": terminal8,
        "survival_generation8": survival8,
        "survival_generation16": survival16,
        "control_advantages_generation8": control_advantages,
        "no_rewrite_generation8": no_rewrite8,
        "active_rewrite_advantage_generation8": active_rewrite_advantage,
        "no_rewrite_loss_fraction": no_rewrite_loss,
        "ablation_generation4": ablation4,
        "ablation_loss_fraction": ablation_loss,
        "rescue_generation4": rescue4,
        "rescue_advantage_generation4": rescue_advantage,
        "rescue_restoration_fraction": rescue_restoration,
        "opposite_rescue_generation4": opposite_rescue4,
        "opposite_founder_generation8": opposite_founder8,
        "carrier_corruption_generation8": corruption8,
        "carrier_renewal": carrier_renewal,
        "carrier_decoder": carrier_decoder,
        "phenotype_decoder": phenotype_decoder,
        "conditions": conditions,
        "claim_boundary": CONTRACT["claim_boundary"],
    }
