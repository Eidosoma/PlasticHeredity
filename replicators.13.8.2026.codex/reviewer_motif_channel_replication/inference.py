"""Pair-cluster inference and frozen adjudication gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr

from .contract import STAGE1_CONTRACT, STAGE2_CONTRACT, semantic_seed


PROBABILITY_KEYS = ("p_a_given_a", "p_a_given_b", "p_b_given_a", "p_b_given_b")


@dataclass
class AssignmentAccumulator:
    replicates: int = 0
    a_from_a: int = 0
    a_from_b: int = 0
    b_from_a: int = 0
    b_from_b: int = 0
    resolved: int = 0

    def add(self, history: str, assignment: str | None) -> None:
        if history not in {"A", "B"}:
            raise ValueError("history must be A or B")
        if history == "A":
            self.replicates += 1
        if assignment is not None:
            self.resolved += 1
        if history == "A" and assignment == "A":
            self.a_from_a += 1
        elif history == "A" and assignment == "B":
            self.b_from_a += 1
        elif history == "B" and assignment == "A":
            self.a_from_b += 1
        elif history == "B" and assignment == "B":
            self.b_from_b += 1

    def finish(self) -> dict[str, float]:
        n = self.replicates
        if n <= 0:
            raise ValueError("no paired replicates accumulated")
        p_a_a = self.a_from_a / n
        p_a_b = self.a_from_b / n
        p_b_a = self.b_from_a / n
        p_b_b = self.b_from_b / n
        direction_a = p_a_a - p_a_b
        direction_b = p_b_b - p_b_a
        return {
            "p_a_given_a": p_a_a,
            "p_a_given_b": p_a_b,
            "p_b_given_a": p_b_a,
            "p_b_given_b": p_b_b,
            "direction_a": direction_a,
            "direction_b": direction_b,
            "crossover": min(direction_a, direction_b),
            "correct": 0.5 * (p_a_a + p_b_b),
            "resolved": self.resolved / (2.0 * n),
        }


def crossover_from_rows(rows: Sequence[Mapping[str, float]]) -> float:
    if not rows:
        return float("nan")
    means = {key: float(np.mean([row[key] for row in rows])) for key in PROBABILITY_KEYS}
    return min(
        means["p_a_given_a"] - means["p_a_given_b"],
        means["p_b_given_b"] - means["p_b_given_a"],
    )


def aggregate_assignment(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot aggregate an empty outcome set")
    probabilities = {
        key: float(np.mean([row[key] for row in rows])) for key in PROBABILITY_KEYS
    }
    direction_a = probabilities["p_a_given_a"] - probabilities["p_a_given_b"]
    direction_b = probabilities["p_b_given_b"] - probabilities["p_b_given_a"]
    return {
        **probabilities,
        "direction_a": direction_a,
        "direction_b": direction_b,
        "crossover": min(direction_a, direction_b),
        "correct": 0.5
        * (probabilities["p_a_given_a"] + probabilities["p_b_given_b"]),
        "resolved": float(np.mean([row["resolved"] for row in rows])),
        "fraction_pairs_positive": float(
            np.mean([float(row["crossover"] > 0.0) for row in rows])
        ),
    }


def bootstrap_interval(
    rows: Sequence[Any],
    statistic: Callable[[Sequence[Any]], float],
    *,
    resamples: int,
    alpha: float,
    namespace: str,
    seed_parts: tuple[object, ...],
) -> list[float]:
    if not rows:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(semantic_seed(namespace, *seed_parts, "bootstrap"))
    size = len(rows)
    values = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        selection = rng.integers(0, size, size=size)
        sample = [rows[int(position)] for position in selection]
        values[index] = statistic(sample)
    return [
        float(np.quantile(values, alpha / 2.0)),
        float(np.quantile(values, 1.0 - alpha / 2.0)),
    ]


def summarize_rows(
    rows: Sequence[Mapping[str, float]],
    *,
    resamples: int,
    alpha: float,
    namespace: str,
    seed_parts: tuple[object, ...],
) -> dict[str, Any]:
    aggregate = aggregate_assignment(rows)
    aggregate["ci"] = bootstrap_interval(
        rows,
        crossover_from_rows,
        resamples=resamples,
        alpha=alpha,
        namespace=namespace,
        seed_parts=seed_parts,
    )
    return aggregate


def paired_advantage(
    intact: Sequence[Mapping[str, float]],
    control: Sequence[Mapping[str, float]],
    *,
    resamples: int,
    alpha: float,
    namespace: str,
    seed_parts: tuple[object, ...],
) -> dict[str, Any]:
    if len(intact) != len(control) or not intact:
        raise ValueError("paired conditions must have equal nonzero pair counts")
    paired = list(zip(intact, control, strict=True))

    def statistic(sample: Sequence[tuple[Mapping[str, float], Mapping[str, float]]]) -> float:
        return crossover_from_rows([item[0] for item in sample]) - crossover_from_rows(
            [item[1] for item in sample]
        )

    estimate = statistic(paired)
    return {
        "estimate": estimate,
        "ci": bootstrap_interval(
            paired,
            statistic,
            resamples=resamples,
            alpha=alpha,
            namespace=namespace,
            seed_parts=seed_parts,
        ),
    }


def _extract_stage1_rows(
    pair_payloads: Sequence[Mapping[str, Any]],
    configuration_id: str,
    condition: str,
    checkpoint: int,
    observer: str = "primary",
) -> list[Mapping[str, float]]:
    return [
        payload["configurations"][configuration_id]["conditions"][condition][
            "checkpoints"
        ][str(checkpoint)][observer]
        for payload in pair_payloads
    ]


def adjudicate_stage1(
    pair_payloads: Sequence[Mapping[str, Any]],
    configuration_id: str,
    *,
    complete: bool,
    namespace: str,
    resamples: int = 10_000,
    checkpoint: int = 64,
) -> dict[str, Any]:
    alpha = float(STAGE1_CONTRACT["familywise_alpha"])
    conditions = {
        name: _extract_stage1_rows(pair_payloads, configuration_id, name, checkpoint)
        for name in (
            "intact",
            "zero",
            "read_disabled",
            "shuffle",
            "opposite_history",
            "process_noise",
            "carrier_sign_corruption",
        )
    }
    summaries = {
        name: summarize_rows(
            rows,
            resamples=resamples,
            alpha=alpha,
            namespace=namespace,
            seed_parts=(configuration_id, name, checkpoint),
        )
        for name, rows in conditions.items()
    }
    terminal_rows = _extract_stage1_rows(
        pair_payloads, configuration_id, "intact", checkpoint, "terminal"
    )
    terminal = summarize_rows(
        terminal_rows,
        resamples=resamples,
        alpha=alpha,
        namespace=namespace,
        seed_parts=(configuration_id, "terminal", checkpoint),
    )
    advantages = {
        name: paired_advantage(
            conditions["intact"],
            conditions[name],
            resamples=resamples,
            alpha=alpha,
            namespace=namespace,
            seed_parts=(configuration_id, "advantage", name),
        )
        for name in ("zero", "read_disabled", "shuffle")
    }
    survival = float(
        np.mean(
            [
                payload["configurations"][configuration_id]["conditions"]["intact"][
                    "checkpoints"
                ][str(checkpoint)]["survival"]
                for payload in pair_payloads
            ]
        )
    )
    intact = summaries["intact"]
    controllable = bool(
        complete
        and intact["crossover"] >= STAGE1_CONTRACT["crossover_gate"]
        and intact["ci"][0] > 0
        and intact["direction_a"] > 0
        and intact["direction_b"] > 0
        and intact["fraction_pairs_positive"] >= 0.5
        and survival >= STAGE1_CONTRACT["survival_gate"]
        and all(
            value["estimate"] >= STAGE1_CONTRACT["control_advantage_gate"]
            and value["ci"][0] > 0
            for value in advantages.values()
        )
        and summaries["opposite_history"]["crossover"] <= -0.10
        and summaries["opposite_history"]["ci"][1] < 0
        and terminal["crossover"] > 0
        and terminal["ci"][0] > 0
    )
    robust = bool(
        controllable
        and all(
            summaries[name]["crossover"]
            >= STAGE1_CONTRACT["robust_crossover_gate"]
            and summaries[name]["ci"][0] > 0
            for name in ("process_noise", "carrier_sign_corruption")
        )
    )
    verdict = (
        "ROBUST_LOCAL_MOTIF_CONTROLLABILITY"
        if robust
        else "LOCAL_MOTIF_CONTROLLABILITY"
        if controllable
        else "INCOMPLETE"
        if not complete
        else "NO_LOCAL_MOTIF_CONTROLLABILITY"
    )
    return {
        "complete": complete,
        "configuration_id": configuration_id,
        "checkpoint": checkpoint,
        "conditions": summaries,
        "terminal": terminal,
        "survival": survival,
        "control_advantages": advantages,
        "controllable": controllable,
        "robust": robust,
        "verdict": verdict,
    }


def dose_summary(
    pair_rows: Mapping[str, Sequence[Mapping[str, float]]],
    *,
    namespace: str,
    resamples: int,
    alpha: float,
) -> dict[str, Any]:
    doses = sorted(float(value) for value in pair_rows)
    keys = [f"{dose:.2f}" for dose in doses]
    contrasts = {
        key: summarize_rows(
            pair_rows[key],
            resamples=resamples,
            alpha=alpha,
            namespace=namespace,
            seed_parts=("dose", key),
        )
        for key in keys
    }
    effects = np.array([contrasts[key]["crossover"] for key in keys])
    slope = float(np.polyfit(np.asarray(doses), effects, 1)[0])
    rank = float(spearmanr(doses, effects).statistic)
    if not np.isfinite(rank):
        rank = -1.0
    pairs_by_index = [list(zip(*(pair_rows[key] for key in keys), strict=True))][0]

    def slope_stat(sample: Sequence[tuple[Mapping[str, float], ...]]) -> float:
        values = [
            crossover_from_rows([pair[dose_index] for pair in sample])
            for dose_index in range(len(keys))
        ]
        return float(np.polyfit(np.asarray(doses), np.asarray(values), 1)[0])

    slope_ci = bootstrap_interval(
        pairs_by_index,
        slope_stat,
        resamples=resamples,
        alpha=alpha,
        namespace=namespace,
        seed_parts=("dose", "slope"),
    )
    monotone = all(
        effects[index + 1] + STAGE2_CONTRACT["monotonic_tolerance"] >= effects[index]
        for index in range(len(effects) - 1)
    )
    passed = bool(
        monotone
        and rank >= STAGE2_CONTRACT["dose_rank_gate"]
        and slope >= STAGE2_CONTRACT["dose_slope_gate"]
        and slope_ci[0] > 0
    )
    return {
        "contrasts": contrasts,
        "spearman": rank,
        "slope": {"estimate": slope, "ci": slope_ci},
        "monotone": monotone,
        "passed": passed,
    }


def _stage2_rows(
    pair_payloads: Sequence[Mapping[str, Any]],
    environment: str,
    condition: str,
    observer: str = "primary",
) -> list[Mapping[str, float]]:
    return [
        payload["environments"][environment]["conditions"][condition]["checkpoints"][
            "64"
        ][observer]
        for payload in pair_payloads
    ]


def _stage2_environment_summary(
    pair_payloads: Sequence[Mapping[str, Any]],
    environment: str,
    *,
    stress: bool,
    namespace: str,
    resamples: int,
) -> dict[str, Any]:
    alpha = float(STAGE2_CONTRACT["familywise_alpha"])
    condition_names = (
        ("intact", "zero", "opposite_history", "unrelated_pair")
        if stress
        else (
            "intact",
            "zero",
            "read_disabled",
            "shuffle",
            "matched_random",
            "opposite_history",
            "unrelated_pair",
            "midpoint",
        )
    )
    rows = {
        condition: _stage2_rows(pair_payloads, environment, condition)
        for condition in condition_names
    }
    summaries = {
        condition: summarize_rows(
            values,
            resamples=resamples,
            alpha=alpha,
            namespace=namespace,
            seed_parts=("stage2", environment, condition),
        )
        for condition, values in rows.items()
    }
    intact = summaries["intact"]
    zero_advantage = paired_advantage(
        rows["intact"],
        rows["zero"],
        resamples=resamples,
        alpha=alpha,
        namespace=namespace,
        seed_parts=("stage2", environment, "advantage", "zero"),
    )
    advantages = {"zero": zero_advantage}
    if not stress:
        for control in ("read_disabled", "shuffle", "matched_random"):
            advantages[control] = paired_advantage(
                rows["intact"],
                rows[control],
                resamples=resamples,
                alpha=alpha,
                namespace=namespace,
                seed_parts=("stage2", environment, "advantage", control),
            )
    survival = float(
        np.mean(
            [
                payload["environments"][environment]["conditions"]["intact"]
                ["checkpoints"]["64"]["survival"]
                for payload in pair_payloads
            ]
        )
    )
    unrelated = summaries["unrelated_pair"]
    retention = (
        unrelated["crossover"] / intact["crossover"]
        if intact["crossover"] > 0
        else 0.0
    )
    threshold = (
        STAGE2_CONTRACT["stress_crossover"]
        if stress
        else STAGE2_CONTRACT["primary_crossover"]
    )
    passed = bool(
        intact["crossover"] >= threshold
        and intact["ci"][0] > 0
        and intact["direction_a"] > 0
        and intact["direction_b"] > 0
        and survival >= STAGE2_CONTRACT["survival_gate"]
        and all(
            advantage["estimate"] >= STAGE2_CONTRACT["control_advantage"]
            and advantage["ci"][0] > 0
            for advantage in advantages.values()
        )
        and summaries["opposite_history"]["crossover"] <= -0.10
        and summaries["opposite_history"]["ci"][1] < 0
        and unrelated["crossover"] >= 0.10
        and retention >= STAGE2_CONTRACT["unrelated_retention"]
    )
    terminal = None
    midpoint = None
    if not stress:
        terminal_rows = _stage2_rows(
            pair_payloads, environment, "intact", observer="terminal"
        )
        terminal = summarize_rows(
            terminal_rows,
            resamples=resamples,
            alpha=alpha,
            namespace=namespace,
            seed_parts=("stage2", environment, "terminal"),
        )
        midpoint = summaries["midpoint"]
        passed = bool(
            passed
            and terminal["crossover"] >= 0.10
            and terminal["ci"][0] > 0
            and abs(midpoint["crossover"]) <= STAGE2_CONTRACT["midpoint_tolerance"]
        )
    return {
        "conditions": summaries,
        "control_advantages": advantages,
        "survival": survival,
        "terminal": terminal,
        "unrelated_retention": retention,
        "midpoint": midpoint,
        "passed": passed,
    }


def adjudicate_stage2(
    pair_payloads: Sequence[Mapping[str, Any]],
    writer_audit: Mapping[str, Any],
    *,
    complete: bool,
    namespace: str,
    resamples: int = 10_000,
) -> dict[str, Any]:
    primary_names = (
        "native",
        "launch0",
        "launch1",
        "launch2",
        "launch3",
        "native_translate_3_5",
        "native_rot90",
        "native_reflect_x",
    )
    stress_names = (
        "random_density_10",
        "random_density_30",
        "random_density_50",
    )
    primary = {
        environment: _stage2_environment_summary(
            pair_payloads,
            environment,
            stress=False,
            namespace=namespace,
            resamples=resamples,
        )
        for environment in primary_names
    }
    stress = {
        environment: _stage2_environment_summary(
            pair_payloads,
            environment,
            stress=True,
            namespace=namespace,
            resamples=resamples,
        )
        for environment in stress_names
    }
    alpha = float(STAGE2_CONTRACT["familywise_alpha"])
    native_robustness = {}
    intact_native = _stage2_rows(pair_payloads, "native", "intact")
    for condition in ("process_noise", "carrier_sign_corruption"):
        rows = _stage2_rows(pair_payloads, "native", condition)
        summary = summarize_rows(
            rows,
            resamples=resamples,
            alpha=alpha,
            namespace=namespace,
            seed_parts=("stage2", "native", condition),
        )
        advantage = paired_advantage(
            intact_native,
            rows,
            resamples=resamples,
            alpha=alpha,
            namespace=namespace,
            seed_parts=("stage2", "native", "robust-advantage", condition),
        )
        summary["retention_vs_intact"] = (
            summary["crossover"] / primary["native"]["conditions"]["intact"]["crossover"]
            if primary["native"]["conditions"]["intact"]["crossover"] > 0
            else 0.0
        )
        # Robustness is a lower-bound test, not a requirement that noise improve
        # the intact arm; advantage is retained as a diagnostic.
        summary["intact_minus_robust"] = advantage
        summary["passed"] = bool(
            summary["crossover"] >= STAGE2_CONTRACT["stress_crossover"]
            and summary["ci"][0] > 0
        )
        native_robustness[condition] = summary
    dose_rows = {
        f"{dose:.2f}": _stage2_rows(
            pair_payloads, "native", f"dose_{dose:.2f}"
        )
        for dose in (0.0, 0.25, 0.50, 0.75, 1.0)
    }
    dose = dose_summary(
        dose_rows,
        namespace=namespace,
        resamples=resamples,
        alpha=alpha,
    )
    general = bool(
        complete
        and writer_audit.get("passed") is True
        and all(value["passed"] for value in primary.values())
        and all(value["passed"] for value in native_robustness.values())
        and dose["passed"]
    )
    density_robust = bool(general and all(value["passed"] for value in stress.values()))
    verdict = (
        "DENSITY_ROBUST_GENERAL_MOTIF_CHANNEL"
        if density_robust
        else "GENERAL_REUSABLE_MOTIF_CHANNEL"
        if general
        else "INCOMPLETE"
        if not complete
        else "NO_GENERAL_REUSABLE_MOTIF_CHANNEL"
    )
    return {
        "complete": complete,
        "writer_audit": dict(writer_audit),
        "primary_environments": primary,
        "stress_environments": stress,
        "native_robustness": native_robustness,
        "dose_response": dose,
        "general": general,
        "density_robust": density_robust,
        "verdict": verdict,
    }
