"""Pair-cluster inference for the frozen Stage-3R replication."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .contract import CONDITIONS, CONTRACT, NAMESPACE, semantic_seed


PROBABILITY_KEYS = ("p_a_given_a", "p_a_given_b", "p_b_given_a", "p_b_given_b")


def _bootstrap_mean(
    values: Sequence[float],
    *,
    resamples: int,
    alpha: float,
    seed_parts: tuple[object, ...],
) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array):
        raise ValueError("bootstrap values must be a nonempty vector")
    rng = np.random.default_rng(semantic_seed(NAMESPACE, *seed_parts, "bootstrap"))
    draws = rng.integers(0, len(array), size=(resamples, len(array)))
    estimates = array[draws].mean(axis=1)
    return [
        float(np.quantile(estimates, alpha / 2.0)),
        float(np.quantile(estimates, 1.0 - alpha / 2.0)),
    ]


def summarize_assignments(
    rows: Sequence[Mapping[str, float]],
    *,
    resamples: int,
    alpha: float,
    seed_parts: tuple[object, ...],
) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize an empty outcome set")
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


def summarize_scalar(
    values: Sequence[float],
    *,
    resamples: int,
    alpha: float,
    seed_parts: tuple[object, ...],
) -> dict[str, Any]:
    return {
        "mean": float(np.mean(values)),
        "ci": _bootstrap_mean(
            values,
            resamples=resamples,
            alpha=alpha,
            seed_parts=seed_parts,
        ),
        "alpha": alpha,
        "n_pairs": len(values),
    }


def paired_advantage(
    intact: Sequence[Mapping[str, float]],
    control: Sequence[Mapping[str, float]],
    *,
    resamples: int,
    alpha: float,
    seed_parts: tuple[object, ...],
) -> dict[str, Any]:
    if len(intact) != len(control) or not intact:
        raise ValueError("paired conditions must have equal nonzero pair counts")
    differences = [
        float(left["crossover"]) - float(right["crossover"])
        for left, right in zip(intact, control, strict=True)
    ]
    return summarize_scalar(
        differences,
        resamples=resamples,
        alpha=alpha,
        seed_parts=seed_parts,
    )


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


def _safe_fraction(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0.0 else float("-inf")


def adjudicate(
    pair_payloads: Sequence[Mapping[str, Any]],
    *,
    complete: bool,
    expected_pairs: int,
    resamples: int = 10_000,
) -> dict[str, Any]:
    """Apply the preregistered strict-form and causal-renewal gates."""

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
            primary_rows = _rows(pair_payloads, condition, generation)
            terminal_rows = _rows(pair_payloads, condition, generation, "terminal")
            conditions[condition][str(generation)] = {
                "primary": summarize_assignments(
                    primary_rows,
                    resamples=resamples,
                    alpha=alpha,
                    seed_parts=("condition", condition, generation, "primary"),
                ),
                "terminal": summarize_assignments(
                    terminal_rows,
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
    control_advantages = {
        control: paired_advantage(
            intact8_rows,
            _rows(pair_payloads, control, 8),
            resamples=resamples,
            alpha=alpha,
            seed_parts=("advantage", "intact", control, 8),
        )
        for control in (
            "zero_every_boundary",
            "shuffle_every_boundary",
            "read_disabled",
            "founder_write_disabled",
        )
    }
    active_rewrite_advantage = paired_advantage(
        intact8_rows,
        _rows(pair_payloads, "no_rewrite", 8),
        resamples=resamples,
        alpha=alpha,
        seed_parts=("advantage", "intact", "no_rewrite", 8),
    )
    rescue_advantage = paired_advantage(
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
    no_rewrite_loss = _safe_fraction(
        intact8["mean"] - no_rewrite8["mean"], intact8["mean"]
    )
    ablation_loss = _safe_fraction(
        intact4["mean"] - ablation4["mean"], intact4["mean"]
    )
    rescue_restoration = _safe_fraction(rescue4["mean"], intact4["mean"])

    gates = {
        "complete_96_pairs": complete,
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
            no_rewrite_loss >= CONTRACT["loss_fraction"]
            and active_rewrite_advantage["mean"] >= CONTRACT["control_advantage"]
            and active_rewrite_advantage["ci"][0] > 0.0
        ),
        "ablation": bool(ablation_loss >= CONTRACT["loss_fraction"]),
        "same_history_rescue": bool(
            rescue_restoration >= CONTRACT["rescue_fraction"]
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
    strict_passed = all(gates.values())
    if strict_passed:
        verdict = "STRICT_RENEWED_CA_PLASTIC_HEREDITY"
    elif not complete:
        verdict = "INCOMPLETE"
    elif (
        gates["generation8_original_form"]
        and no_rewrite_loss < CONTRACT["loss_fraction"]
    ):
        verdict = "STATIC_HIDDEN_TEMPLATE"
    elif gates["generation8_original_form"] and not gates["generation16_original_form"]:
        verdict = "TRANSIENT_LINEAGE_MEMORY"
    else:
        verdict = "NO_DURABLE_RENEWAL"

    carrier_renewal: dict[str, Any] = {}
    for generation in checkpoints:
        carrier_renewal[str(generation)] = {
            "intact_exit_centroid_l2": summarize_scalar(
                _carrier_values(
                    pair_payloads, "intact", generation, "exit", "centroid_l2"
                ),
                resamples=resamples,
                alpha=alpha,
                seed_parts=("carrier", "intact", generation, "exit", "centroid_l2"),
            ),
            "no_rewrite_entry_mean_abs": summarize_scalar(
                _carrier_values(
                    pair_payloads, "no_rewrite", generation, "entry", "mean_abs"
                ),
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
        "gates": gates,
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
        "conditions": conditions,
        "secondary_decoder": {
            "status": "NOT_ADJUDICATED_UNDERSPECIFIED_NON_GATING_FALLBACK",
            "reason": (
                "The data/docs source does not operationally define its independent "
                "texture descriptor; inventing it would violate direct replication."
            ),
            "effect_on_strict_primary_verdict": "none",
        },
        "claim_boundary": CONTRACT["claim_boundary"],
    }
