"""Registered founder-pair inference for the compact-carrier confirmation."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .contract import (
    CANDIDATE_IDS,
    CHECKPOINT_GENERATIONS,
    CONDITIONS,
    CONTRACT,
    ENVIRONMENTS,
    NAMESPACE,
    semantic_seed,
)


ASSIGNMENT_KEYS = (
    "p_a_given_a",
    "p_a_given_b",
    "p_b_given_a",
    "p_b_given_b",
    "direction_a",
    "direction_b",
    "crossover",
    "correct",
    "resolved",
)


def _vector(values: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite nonempty vector")
    return array


def _bootstrap_mean(
    values: Sequence[float],
    *,
    resamples: int,
    alpha: float,
    seed_parts: tuple[object, ...],
) -> list[float]:
    array = _vector(values, "bootstrap values")
    rng = np.random.default_rng(
        semantic_seed(NAMESPACE, "inference", *seed_parts, "pair-bootstrap")
    )
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
    array = _vector(values, "scalar values")
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
        raise ValueError("cannot summarize empty assignments")
    for row in rows:
        _vector([float(row[key]) for key in ASSIGNMENT_KEYS], "assignment row")
    result = {
        key: float(np.mean([float(row[key]) for row in rows]))
        for key in ASSIGNMENT_KEYS
        if key != "crossover"
    }
    crossovers = [float(row["crossover"]) for row in rows]
    result.update(
        {
            "mean": float(np.mean(crossovers)),
            "ci": _bootstrap_mean(
                crossovers,
                resamples=resamples,
                alpha=alpha,
                seed_parts=seed_parts,
            ),
            "alpha": alpha,
            "n_pairs": len(rows),
            "fraction_pairs_positive": float(np.mean(np.asarray(crossovers) > 0.0)),
        }
    )
    return result


def paired_advantage(
    left: Sequence[float],
    right: Sequence[float],
    *,
    resamples: int,
    alpha: float,
    seed_parts: tuple[object, ...],
) -> dict[str, Any]:
    a = _vector(left, "left paired values")
    b = _vector(right, "right paired values")
    if a.shape != b.shape:
        raise ValueError("paired vectors must have identical shape")
    return summarize_scalar(
        a - b,
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


def _crossovers(
    payloads: Sequence[Mapping[str, Any]], condition: str, generation: int
) -> list[float]:
    return [float(row["crossover"]) for row in _rows(payloads, condition, generation)]


def _ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0.0:
        return None
    return float(numerator / denominator)


def _ratio_gate(value: float | None, threshold: float) -> bool:
    return value is not None and value >= threshold


def adjudicate_cell(
    payloads: Sequence[Mapping[str, Any]],
    *,
    candidate_id: str,
    environment: str,
    expected_pairs: int,
    resamples: int,
) -> dict[str, Any]:
    if not payloads:
        return {
            "state": "incomplete",
            "candidate_id": candidate_id,
            "environment": environment,
            "n_pairs": 0,
            "expected_pairs": expected_pairs,
            "stage4_renewed_gate": False,
            "verdict": "INCOMPLETE",
        }
    pair_ids = [str(payload["pair_id"]) for payload in payloads]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("duplicate founder-pair checkpoint in inference")
    alpha = float(CONTRACT["confirmation_alpha_per_codec"])
    complete = len(payloads) == expected_pairs
    conditions: dict[str, Any] = {}
    for condition in CONDITIONS:
        conditions[condition] = {}
        for generation in CHECKPOINT_GENERATIONS:
            conditions[condition][str(generation)] = {
                "primary": summarize_assignments(
                    _rows(payloads, condition, generation),
                    resamples=resamples,
                    alpha=alpha,
                    seed_parts=(candidate_id, environment, condition, generation, "primary"),
                ),
                "terminal": summarize_assignments(
                    _rows(payloads, condition, generation, "terminal"),
                    resamples=resamples,
                    alpha=alpha,
                    seed_parts=(candidate_id, environment, condition, generation, "terminal"),
                ),
                "survival": summarize_scalar(
                    [
                        float(
                            payload["conditions"][condition]["outcomes"][str(generation)][
                                "survival"
                            ]
                        )
                        for payload in payloads
                    ],
                    resamples=resamples,
                    alpha=alpha,
                    seed_parts=(candidate_id, environment, condition, generation, "survival"),
                ),
            }
    intact4 = conditions["intact"]["4"]["primary"]
    intact8 = conditions["intact"]["8"]["primary"]
    intact16 = conditions["intact"]["16"]["primary"]
    terminal8 = conditions["intact"]["8"]["terminal"]
    survival8 = conditions["intact"]["8"]["survival"]
    survival16 = conditions["intact"]["16"]["survival"]
    controls = (
        "zero_every_boundary",
        "decoded_shuffle_every_boundary",
        "read_disabled",
        "founder_write_disabled",
    )
    control_advantages = {
        condition: paired_advantage(
            _crossovers(payloads, "intact", 8),
            _crossovers(payloads, condition, 8),
            resamples=resamples,
            alpha=alpha,
            seed_parts=(candidate_id, environment, "control", condition, 8),
        )
        for condition in controls
    }
    latent_shuffle_advantage = paired_advantage(
        _crossovers(payloads, "intact", 8),
        _crossovers(payloads, "latent_shuffle_every_boundary", 8),
        resamples=resamples,
        alpha=alpha,
        seed_parts=(candidate_id, environment, "latent-shuffle", 8),
    )
    active_rewrite_advantage = paired_advantage(
        _crossovers(payloads, "intact", 8),
        _crossovers(payloads, "no_rewrite", 8),
        resamples=resamples,
        alpha=alpha,
        seed_parts=(candidate_id, environment, "active-rewrite", 8),
    )
    rescue_advantage = paired_advantage(
        _crossovers(payloads, "rescue_same_enter_g4", 4),
        _crossovers(payloads, "ablate_after_g2", 4),
        resamples=resamples,
        alpha=alpha,
        seed_parts=(candidate_id, environment, "rescue-over-ablation", 4),
    )
    no_rewrite8 = conditions["no_rewrite"]["8"]["primary"]
    ablation4 = conditions["ablate_after_g2"]["4"]["primary"]
    rescue4 = conditions["rescue_same_enter_g4"]["4"]["primary"]
    opposite_rescue4 = conditions["rescue_opposite_enter_g4"]["4"]["primary"]
    opposite_founder8 = conditions["opposite_founder"]["8"]["primary"]
    corruption8 = conditions["latent_corruption_1"]["8"]["primary"]
    no_rewrite_loss = _ratio(intact8["mean"] - no_rewrite8["mean"], intact8["mean"])
    ablation_loss = _ratio(intact4["mean"] - ablation4["mean"], intact4["mean"])
    rescue_restoration = _ratio(rescue4["mean"], intact4["mean"])

    renewed_gates = {
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
        "four_controls": all(
            value["mean"] >= CONTRACT["control_advantage"] and value["ci"][0] > 0.0
            for value in control_advantages.values()
        ),
        "active_rewrite": bool(
            _ratio_gate(no_rewrite_loss, float(CONTRACT["loss_fraction"]))
            and active_rewrite_advantage["mean"] >= CONTRACT["control_advantage"]
            and active_rewrite_advantage["ci"][0] > 0.0
        ),
        "ablation": _ratio_gate(ablation_loss, float(CONTRACT["loss_fraction"])),
        "same_history_rescue": bool(
            _ratio_gate(rescue_restoration, float(CONTRACT["rescue_fraction"]))
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
        "latent_corruption": bool(
            corruption8["mean"] >= CONTRACT["corruption_crossover"]
            and corruption8["ci"][0] > 0.0
        ),
    }
    renewed_gate = all(renewed_gates.values())
    latent_gate = bool(
        latent_shuffle_advantage["mean"] >= CONTRACT["control_advantage"]
        and latent_shuffle_advantage["ci"][0] > 0.0
    )
    stage4_gate = renewed_gate and latent_gate
    return {
        "state": "complete" if complete else "incomplete",
        "candidate_id": candidate_id,
        "environment": environment,
        "alpha": alpha,
        "n_pairs": len(payloads),
        "expected_pairs": expected_pairs,
        "intact_generation4": intact4,
        "intact_generation8": intact8,
        "intact_generation16": intact16,
        "terminal_generation8": terminal8,
        "survival_generation8": survival8,
        "survival_generation16": survival16,
        "direction_a_mean": intact8["direction_a"],
        "direction_b_mean": intact8["direction_b"],
        "fraction_pairs_positive": intact8["fraction_pairs_positive"],
        "control_advantages_generation8": control_advantages,
        "latent_shuffle_advantage": latent_shuffle_advantage,
        "latent_shuffle_gate": latent_gate,
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
        "latent_corruption_generation8": corruption8,
        "renewed_gates": renewed_gates,
        "renewed_gate": renewed_gate,
        "stage4_renewed_gate": stage4_gate,
        "verdict": (
            "STRICT_RENEWED_CA_PLASTIC_HEREDITY" if stage4_gate else "NO_STRICT_RENEWAL"
        )
        if complete
        else "INCOMPLETE",
        "conditions": conditions,
    }


def _candidate_model(payloads: Sequence[Mapping[str, Any]], fallback: Mapping[str, Any]) -> dict[str, Any]:
    return dict(payloads[0]["candidate"] if payloads else fallback)


def adjudicate_campaign(
    grouped_payloads: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
    *,
    candidate_models: Mapping[str, Mapping[str, Any]],
    expected_pairs: int,
    resamples: int,
) -> dict[str, Any]:
    candidates: dict[str, Any] = {}
    all_complete = True
    for candidate_id in CANDIDATE_IDS:
        environments: dict[str, Any] = {}
        union: list[Mapping[str, Any]] = []
        for environment in ENVIRONMENTS:
            payloads = list(grouped_payloads.get((candidate_id, environment), []))
            union.extend(payloads)
            strict = adjudicate_cell(
                payloads,
                candidate_id=candidate_id,
                environment=environment,
                expected_pairs=expected_pairs,
                resamples=resamples,
            )
            all_complete &= strict["state"] == "complete"
            environments[environment] = {"strict": strict}
        candidates[candidate_id] = {
            "model": _candidate_model(union, candidate_models[candidate_id]),
            "environments": environments,
        }

    secondary: dict[str, Any] = {}
    for environment in ENVIRONMENTS:
        identity = {
            str(payload["pair_id"]): payload
            for payload in grouped_payloads.get(("identity-r512-f32", environment), [])
        }
        walsh = {
            str(payload["pair_id"]): payload
            for payload in grouped_payloads.get(("walsh-r016-q04", environment), [])
        }
        shared = sorted(set(identity) & set(walsh))
        secondary[environment] = {}
        for generation in (8, 16):
            if not shared:
                secondary[environment][f"walsh_minus_identity_generation{generation}"] = None
                continue
            secondary[environment][f"walsh_minus_identity_generation{generation}"] = paired_advantage(
                [
                    float(walsh[pair]["conditions"]["intact"]["outcomes"][str(generation)]["primary"]["crossover"])
                    for pair in shared
                ],
                [
                    float(identity[pair]["conditions"]["intact"]["outcomes"][str(generation)]["primary"]["crossover"])
                    for pair in shared
                ],
                resamples=resamples,
                alpha=float(CONTRACT["confirmation_alpha_per_codec"]),
                seed_parts=(environment, "walsh-minus-identity", generation),
            )

    def passed(candidate: str, environment: str) -> bool:
        return bool(
            candidates[candidate]["environments"][environment]["strict"].get(
                "stage4_renewed_gate", False
            )
        )

    identity_ordinary = passed("identity-r512-f32", "ordinary")
    walsh_ordinary = passed("walsh-r016-q04", "ordinary")
    walsh_moderate = passed("walsh-r016-q04", "moderate_joint")
    robust_compact = [
        candidate
        for candidate in ("pca-r008-q04", "walsh-r016-q04")
        if passed(candidate, "ordinary") and passed(candidate, "moderate_joint")
    ]
    compact = [
        candidate
        for candidate in ("pca-r008-q04", "walsh-r016-q04")
        if passed(candidate, "ordinary")
    ]
    if not all_complete:
        verdict = "INCOMPLETE"
    elif not identity_ordinary:
        verdict = "NO_FRESH_STAGE3R_REPLICATION"
    elif robust_compact:
        verdict = "ROBUST_COMPACT_RENEWED_CA_PLASTIC_HEREDITY"
    elif compact:
        verdict = "COMPACT_RENEWED_CA_PLASTIC_HEREDITY"
    else:
        verdict = "FULL_CARRIER_ONLY"
    return {
        "state": "complete" if all_complete else "incomplete",
        "verdict": verdict,
        "target_replication_passed": bool(
            all_complete and identity_ordinary and walsh_ordinary and walsh_moderate
        ),
        "target_replication_contract": (
            "identity ordinary plus Walsh-r16-q04 ordinary and moderate strict ladders; "
            "PCA cannot substitute for Walsh"
        ),
        "fresh_anchor_replicated": identity_ordinary,
        "robust_compact_candidate_ids": robust_compact,
        "compact_candidate_ids": compact,
        "candidates": candidates,
        "registered_secondary_non_gating": secondary,
        "claim_boundary": CONTRACT["claim_boundary"],
    }
