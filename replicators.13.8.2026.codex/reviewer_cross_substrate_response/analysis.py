"""Registered pilot and confirmation summaries."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .core import (
    BREAK_HORIZON,
    block_bootstrap_interval,
    block_center,
    derive_seed,
    score_break_renewal,
    sign_randomization_p,
    spearman,
)
from .models import ModelProfile


FULL_RESAMPLES = 4_096
MODEL_ALPHA = 0.025


def pilot_eligibility(
    futures: pd.DataFrame,
    expected_blocks: int,
    profile: ModelProfile,
) -> dict[str, Any]:
    if futures.empty:
        return {
            "profile": profile.name,
            "eligible": False,
            "reason": "no futures",
            "complete_blocks": 0,
            "complete_horizon_fraction": 0.0,
            "breaks": 0,
            "events": 0,
            "event_blocks": 0,
        }
    state_counts = futures.groupby(["block_id", "landmark"]).size()
    complete_blocks = 0
    for block_id in futures["block_id"].unique():
        counts = state_counts.loc[block_id] if block_id in state_counts.index.levels[0] else pd.Series(dtype=int)
        block = futures[futures["block_id"] == block_id]
        if (
            set(int(value) for value in counts.index) == set(profile.landmarks)
            and bool((counts == profile.pilot_branches).all())
            and bool(block.get("main_complete", pd.Series([1])).astype(bool).all())
        ):
            complete_blocks += 1
    complete_fraction = float(futures["complete_horizon"].mean())
    breaks = int(futures["break_index"].notna().sum())
    events = int(futures["event"].sum())
    event_blocks = int(futures.loc[futures["event"] == 1, "block_id"].nunique())
    gates = {
        "complete_blocks": complete_blocks >= 24,
        "complete_horizon_fraction": complete_fraction >= 0.70,
        "break_count": breaks >= 100,
        "event_count": events >= 50,
        "event_blocks": event_blocks >= 8,
    }
    eligible = bool(profile.name == "full" and all(gates.values()))
    return {
        "profile": profile.name,
        "expected_blocks": int(expected_blocks),
        "complete_blocks": complete_blocks,
        "complete_horizon_fraction": complete_fraction,
        "breaks": breaks,
        "events": events,
        "event_blocks": event_blocks,
        "gates": gates,
        "eligible": eligible,
        "smoke_cannot_qualify": profile.name != "full",
    }


def _cell_summary(values: np.ndarray, blocks: np.ndarray, label: str, repetitions: int) -> dict[str, Any]:
    interval = block_bootstrap_interval(
        values,
        blocks,
        repetitions=repetitions,
        seed=derive_seed("analysis", label, "bootstrap"),
        confidence=0.95,
    )
    block_frame = pd.DataFrame({"block": blocks, "value": values}).groupby("block")["value"].mean()
    p_value = sign_randomization_p(
        block_frame.to_numpy(),
        repetitions=repetitions,
        seed=derive_seed("analysis", label, "randomization"),
    )
    return {
        "mean": float(block_frame.mean()),
        "ci97_5_one_sided_lower": interval[0],
        "two_sided_ci95": list(interval),
        "randomization_p_one_sided": p_value,
        "blocks": int(block_frame.size),
    }


def _reliability_summary(
    futures: pd.DataFrame, model: str, repetitions: int
) -> dict[str, Any]:
    state_rates = (
        futures.groupby(["block_id", "landmark", "half"], as_index=False)["event"]
        .mean()
        .pivot(index=["block_id", "landmark"], columns="half", values="event")
        .reset_index()
    )
    if not {"A", "B"}.issubset(state_rates.columns):
        return {"estimate": 0.0, "ci97_5_one_sided_lower": 0.0, "randomization_p_one_sided": 1.0, "passed": False}
    blocks = state_rates["block_id"].to_numpy()
    left = block_center(state_rates["A"].to_numpy(), blocks)
    right = block_center(state_rates["B"].to_numpy(), blocks)
    estimate = spearman(left, right)
    unique = np.unique(blocks)
    rng = np.random.default_rng(derive_seed("analysis", model, "reliability", "bootstrap"))
    draws = np.empty(repetitions, dtype=np.float64)
    grouped = [np.flatnonzero(blocks == block) for block in unique]
    for index in range(repetitions):
        picks = rng.integers(0, len(grouped), size=len(grouped))
        aa: list[float] = []
        bb: list[float] = []
        for pick in picks:
            rows = grouped[int(pick)]
            aa.extend(left[rows])
            bb.extend(right[rows])
        draws[index] = spearman(aa, bb)
    lower = float(np.quantile(draws, 0.025))

    rng = np.random.default_rng(derive_seed("analysis", model, "reliability", "randomization"))
    extreme = 0
    for _ in range(repetitions):
        randomized = right.copy()
        for block in unique:
            if rng.random() < 0.5:
                randomized[blocks == block] *= -1.0
        extreme += int(spearman(left, randomized) >= estimate)
    p_value = float((extreme + 1) / (repetitions + 1))
    return {
        "estimate": estimate,
        "ci97_5_one_sided_lower": lower,
        "randomization_p_one_sided": p_value,
        "states": int(state_rates.shape[0]),
        "blocks": int(unique.size),
        "passed": bool(lower > 0 and p_value <= MODEL_ALPHA),
    }


def confirmation_metrics(
    model: str,
    futures: pd.DataFrame,
    boundaries: pd.DataFrame,
    profile: ModelProfile,
    threshold: float,
) -> dict[str, Any]:
    repetitions = FULL_RESAMPLES if profile.name == "full" else 64
    primary_boundaries = boundaries[
        pd.to_numeric(boundaries.get("boundary", pd.Series(dtype=float)), errors="coerce")
        < BREAK_HORIZON
    ].copy()
    matched = primary_boundaries[
        np.isfinite(pd.to_numeric(primary_boundaries["stranger_similarity"], errors="coerce"))
    ].copy() if not primary_boundaries.empty else primary_boundaries
    if matched.empty:
        fidelity = {
            "mean_similarity_difference": None,
            "ci97_5_one_sided_lower": None,
            "matched_boundaries": 0,
            "passed": False,
        }
    else:
        differences = (matched["similarity"] - matched["stranger_similarity"]).to_numpy()
        matched_blocks = matched["block_id"].to_numpy()
        block_differences = (
            pd.DataFrame({"block": matched_blocks, "difference": differences})
            .groupby("block")["difference"]
            .mean()
        )
        interval = block_bootstrap_interval(
            differences,
            matched_blocks,
            repetitions=repetitions,
            seed=derive_seed("analysis", model, "fidelity"),
            confidence=0.95,
        )
        inherited_fraction = float(
            primary_boundaries.groupby("block_id")["inherited"].mean().mean()
        )
        break_fraction = 1.0 - inherited_fraction
        fidelity = {
            "mean_similarity_difference": float(block_differences.mean()),
            "ci97_5_one_sided_lower": interval[0],
            "matched_boundaries": int(matched.shape[0]),
            "inherited_boundary_fraction": inherited_fraction,
            "break_boundary_fraction": break_fraction,
            "passed": bool(interval[0] > 0 and inherited_fraction >= 0.50 and break_fraction >= 0.05),
        }

    half_metrics: dict[str, Any] = {}
    for half in ("A", "B"):
        cell = futures[futures["half"] == half]
        prevalence = float(
            cell.groupby("block_id")["event"].mean().mean()
        ) if not cell.empty else 0.0
        sequence = _cell_summary(
            cell["event_minus_order_null"].to_numpy(dtype=float),
            cell["block_id"].to_numpy(),
            f"{model}:{half}:sequence",
            repetitions,
        ) if not cell.empty else {
            "mean": 0.0,
            "ci97_5_one_sided_lower": 0.0,
            "randomization_p_one_sided": 1.0,
        }
        sequence["passed"] = bool(
            sequence["ci97_5_one_sided_lower"] > 0
            and sequence["randomization_p_one_sided"] <= MODEL_ALPHA
        )
        half_metrics[half] = {
            "prevalence": prevalence,
            "events": int(cell["event"].sum()),
            "sequence_excess": sequence,
        }

    total_events = int(futures["event"].sum())
    event_blocks = int(futures.loc[futures["event"] == 1, "block_id"].nunique())
    nondegenerate = bool(
        all(half_metrics[half]["prevalence"] >= 0.01 for half in ("A", "B"))
        and total_events >= 100
        and event_blocks >= 16
    )
    reliability = _reliability_summary(futures, model, repetitions)
    sensitivities = sensitivity_summaries(futures, boundaries, threshold)
    model_passed = bool(
        profile.name == "full"
        and fidelity["passed"]
        and nondegenerate
        and all(half_metrics[half]["sequence_excess"]["passed"] for half in ("A", "B"))
        and reliability["passed"]
    )
    return {
        "model": model,
        "profile": profile.name,
        "model_alpha": MODEL_ALPHA,
        "fidelity": fidelity,
        "halves": half_metrics,
        "nondegenerate_event_gate": {
            "total_events": total_events,
            "event_blocks": event_blocks,
            "passed": nondegenerate,
        },
        "state_dependence": reliability,
        "non_rescuing_sensitivities": sensitivities,
        "model_passed": model_passed,
        "smoke_cannot_pass": profile.name != "full",
    }


def sensitivity_summaries(
    futures: pd.DataFrame,
    boundaries: pd.DataFrame,
    threshold: float,
) -> dict[str, Any]:
    """Return preregistered descriptive variants; none enter a pass gate."""

    specifications = {
        "raw_S_gt_0.9_F12_R3": (0.9, 12, 3),
        "calibrated_F8_R3": (threshold, 8, 3),
        "calibrated_F16_R3": (threshold, 16, 3),
        "calibrated_F12_R2": (threshold, 12, 2),
        "calibrated_F12_R4": (threshold, 12, 4),
    }
    if futures.empty:
        return {
            "non_rescuing": True,
            "variants": {
                name: {"events": 0, "prevalence": 0.0, "complete_horizon_fraction": 0.0}
                for name in specifications
            },
        }
    grouped = {
        str(future_id): group.sort_values("boundary")["similarity"].to_numpy(dtype=float)
        for future_id, group in boundaries.groupby("future_id")
    } if not boundaries.empty else {}
    variants: dict[str, Any] = {}
    for name, (variant_threshold, horizon, run_length) in specifications.items():
        events: list[int] = []
        complete: list[int] = []
        halves: dict[str, list[int]] = {"A": [], "B": []}
        for _, row in futures.iterrows():
            similarities = grouped.get(str(row["future_id"]), np.zeros(0, dtype=float))
            outcome = score_break_renewal(
                similarities,
                variant_threshold,
                horizon=horizon,
                run_length=run_length,
                complete_horizon=similarities.size >= horizon,
            )
            event = int(outcome.event)
            events.append(event)
            complete.append(int(outcome.complete_horizon))
            half = str(row["half"])
            if half in halves:
                halves[half].append(event)
        variants[name] = {
            "threshold": float(variant_threshold),
            "horizon": horizon,
            "run_length": run_length,
            "events": int(sum(events)),
            "prevalence": float(np.mean(events)),
            "complete_horizon_fraction": float(np.mean(complete)),
            "half_prevalence": {
                half: float(np.mean(values)) if values else 0.0
                for half, values in halves.items()
            },
        }
    return {"non_rescuing": True, "variants": variants}
