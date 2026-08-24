from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from .config import Registration
from .rng import generator
from .storage import read_records


Record = dict[str, Any]
HISTORIES = ("A", "B")
HALVES = (0, 1)


def _matching(records: Iterable[Record], **criteria: Any) -> list[Record]:
    return [row for row in records if all(row.get(key) == value for key, value in criteria.items())]


def _correct_rate(row: Record) -> float:
    count = int(row["hold_a"] if row["history"] == "A" else row["hold_b"])
    return count / max(1, int(row["n"]))


def _wrong_rate(row: Record) -> float:
    count = int(row["hold_b"] if row["history"] == "A" else row["hold_a"])
    return count / max(1, int(row["n"]))


def _direct_crossover(row: Record) -> float:
    # This is the within-treatment form-specific crossover contribution.  It
    # uses the first three-cycle point destination, not strict-8 holding risk.
    if row["history"] == "A":
        signed = int(row["dest_a"]) - int(row["dest_b"])
    else:
        signed = int(row["dest_b"]) - int(row["dest_a"])
    return signed / max(1, int(row["n"]))


def _source_half_values(
    records: Iterable[Record], score: Callable[[Record], float]
) -> dict[tuple[int, int], float]:
    grouped: dict[tuple[int, int], list[tuple[float, int]]] = defaultdict(list)
    for row in records:
        grouped[(int(row["source_id"]), int(row["half"]))].append((score(row), int(row["n"])))
    result: dict[tuple[int, int], float] = {}
    for key, values in grouped.items():
        weights = np.asarray([value[1] for value in values], dtype=float)
        scores = np.asarray([value[0] for value in values], dtype=float)
        result[key] = float(np.average(scores, weights=weights))
    return result


def _paired_half_arrays(
    records: list[Record], treatment: str, control: str,
    *, score: Callable[[Record], float] = _correct_rate, **criteria: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    treated = _source_half_values(_matching(records, arm=treatment, **criteria), score)
    controls = _source_half_values(_matching(records, arm=control, **criteria), score)
    sources = sorted(
        source for source in {key[0] for key in treated} & {key[0] for key in controls}
        if all((source, half) in treated and (source, half) in controls for half in HALVES)
    )
    first = np.asarray([treated[(source, 0)] - controls[(source, 0)] for source in sources], dtype=float)
    second = np.asarray([treated[(source, 1)] - controls[(source, 1)] for source in sources], dtype=float)
    return np.asarray(sources, dtype=int), first, second


def _direct_half_arrays(
    records: list[Record], arm: str,
    *, score: Callable[[Record], float] = _direct_crossover, **criteria: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = _source_half_values(_matching(records, arm=arm, **criteria), score)
    sources = sorted(
        source for source in {key[0] for key in values}
        if all((source, half) in values for half in HALVES)
    )
    return (
        np.asarray(sources, dtype=int),
        np.asarray([values[(source, 0)] for source in sources], dtype=float),
        np.asarray([values[(source, 1)] for source in sources], dtype=float),
    )


def _pooled(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    if first.size != second.size:
        raise ValueError("future halves are not aligned")
    return (np.asarray(first, dtype=float) + np.asarray(second, dtype=float)) / 2.0


def _history_logloss_gain_values(
    records: list[Record], arm: str, **criteria: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Cross-fit a source-specific destination committor against pooled history.

    On one future half we estimate P(destination | history, source) and the
    corresponding history-blind P(destination | source), both with
    Dirichlet-0.5 smoothing over A/B/other.  The other half evaluates proper
    multiclass log loss; halves are then reversed and averaged.  This is the
    registered prediction score and intentionally does not compare losses from
    two different intervention outcome distributions.
    """
    categories = ("dest_a", "dest_b", "dest_other")
    counts: dict[tuple[int, int, str], np.ndarray] = defaultdict(
        lambda: np.zeros(len(categories), dtype=np.int64)
    )
    for row in _matching(records, arm=arm, **criteria):
        key = (int(row["source_id"]), int(row["half"]), str(row["history"]))
        counts[key] += np.asarray([int(row[name]) for name in categories], dtype=np.int64)
    candidate_sources = sorted({key[0] for key in counts})
    sources: list[int] = []
    gains: list[float] = []
    for source in candidate_sources:
        if not all((source, half, history) in counts for half in HALVES for history in HISTORIES):
            continue
        half_gains: list[float] = []
        for evaluation_half in HALVES:
            training_half = 1 - evaluation_half
            training = {
                history: counts[(source, training_half, history)].astype(float)
                for history in HISTORIES
            }
            pooled = training["A"] + training["B"]
            pooled_probabilities = (pooled + 0.5) / (float(np.sum(pooled)) + 1.5)
            informed_loss = 0.0
            pooled_loss = 0.0
            total = 0
            for history in HISTORIES:
                history_counts = training[history]
                probabilities = (history_counts + 0.5) / (
                    float(np.sum(history_counts)) + 1.5
                )
                evaluation = counts[(source, evaluation_half, history)].astype(float)
                informed_loss -= float(np.sum(evaluation * np.log(probabilities)))
                pooled_loss -= float(np.sum(evaluation * np.log(pooled_probabilities)))
                total += int(np.sum(evaluation))
            half_gains.append((pooled_loss - informed_loss) / max(1, total))
        sources.append(source)
        gains.append(float(np.mean(half_gains)))
    return np.asarray(sources, dtype=int), np.asarray(gains, dtype=float)


def _bootstrap_point(values: np.ndarray, repeats: int, rng: np.random.Generator) -> dict[str, Any]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"n": 0, "mean": None, "lower": None, "upper": None}
    indices = rng.integers(0, finite.size, size=(repeats, finite.size))
    bootstrap = np.mean(finite[indices], axis=1)
    return {
        "n": int(finite.size),
        "mean": float(np.mean(finite)),
        "lower": float(np.quantile(bootstrap, 0.025)),
        "upper": float(np.quantile(bootstrap, 0.975)),
    }


def _simultaneous_family(
    registration: Registration,
    family: dict[str, np.ndarray],
    label: tuple[Any, ...],
) -> dict[str, dict[str, Any]]:
    if not family:
        return {}
    sizes = {np.asarray(values).size for values in family.values()}
    if len(sizes) != 1 or next(iter(sizes)) == 0:
        raise ValueError(f"unaligned or empty simultaneous family {label}: {sizes}")
    names = list(family)
    matrix = np.column_stack([np.asarray(family[name], dtype=float) for name in names])
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"non-finite simultaneous family {label}")
    repeats = int(registration.profile["bootstrap_repetitions"])
    rng = generator(str(registration.protocol["master_seed"]), "analysis", *label)
    indices = rng.integers(0, matrix.shape[0], size=(repeats, matrix.shape[0]))
    boot = np.mean(matrix[indices], axis=1)
    points = np.mean(matrix, axis=0)
    max_shortfall = np.max(points[None, :] - boot, axis=1)
    critical = float(np.quantile(max_shortfall, float(registration.protocol["gates"]["interval_level"])))
    return {
        name: {
            "n": int(matrix.shape[0]),
            "mean": float(points[index]),
            "simultaneous_lower": float(points[index] - critical),
            "critical_shortfall": critical,
        }
        for index, name in enumerate(names)
    }


def _reliability(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if first.size < 3 or first.size != second.size:
        return 0.0
    if np.array_equal(first, second):
        return 1.0
    if np.std(first) <= 1e-15 or np.std(second) <= 1e-15:
        return 0.0
    return float(np.clip(np.corrcoef(first, second)[0, 1], -1.0, 1.0))


def _mean_strict_correct(records: list[Record], **criteria: Any) -> float:
    rows = _matching(records, **criteria)
    total = sum(int(row["n"]) for row in rows)
    correct = sum(
        int(row["hold_a"] if row["history"] == "A" else row["hold_b"])
        for row in rows
    )
    return correct / max(1, total)


def _state_writer_analysis(registration: Registration, records: list[Record], condition: str) -> dict[str, Any]:
    gates = registration.protocol["gates"]
    primary = {"condition": condition, "challenge": "neutral_damage", "age": 0}
    _, risk0, risk1 = _paired_half_arrays(
        records, "state_transplant", "reset_both", **primary
    )
    _, cross0, cross1 = _direct_half_arrays(
        records, "state_transplant", **primary
    )
    _, logloss = _history_logloss_gain_values(records, "state_transplant", **primary)
    _, shuffle0, shuffle1 = _paired_half_arrays(
        records, "state_transplant", "state_pattern_shuffle", **primary
    )
    _, age_risk0, age_risk1 = _paired_half_arrays(
        records, "state_transplant", "reset_both", condition=condition, age=1,
        challenge="release_only",
    )
    _, age_cross0, age_cross1 = _direct_half_arrays(
        records, "state_transplant", condition=condition, age=1,
        challenge="release_only",
    )
    values = {
        "risk_gain": _pooled(risk0, risk1),
        "direct_crossover": _pooled(cross0, cross1),
        "history_logloss_gain": logloss,
        "shuffle_contrast": _pooled(shuffle0, shuffle1),
        "age1_risk_gain": _pooled(age_risk0, age_risk1),
        "age1_crossover": _pooled(age_cross0, age_cross1),
    }
    summaries = _simultaneous_family(registration, values, ("state", condition))
    acquisition = {
        history: float(np.mean([
            float(row["acquired_exact"])
            for row in _matching(records, condition=condition, arm="state_transplant", history=history, age=0)
        ]))
        for history in HISTORIES
    }
    hold = {
        history: _mean_strict_correct(
            records, condition=condition, arm="state_transplant", history=history,
            challenge="neutral_damage", age=0,
        )
        for history in HISTORIES
    }
    ceiling = {
        history: _mean_strict_correct(
            records, condition=condition, arm="destination_matched_donor",
            history=history, challenge="neutral_damage", age=0,
        )
        for history in HISTORIES
    }
    ceiling_fraction = {
        history: hold[history] / max(ceiling[history], 1e-12) for history in HISTORIES
    }
    identity_fields = (
        "source_id", "midpoint", "history", "challenge", "age", "half",
        "n", "future_digest", "trajectory_digest",
    )
    self_signature = sorted(
        tuple(row[field] for field in identity_fields)
        for row in _matching(records, condition=condition, arm="self_continuation", challenge="neutral_damage", age=0)
    )
    transplant_signature = sorted(
        tuple(row[field] for field in identity_fields)
        for row in _matching(records, condition=condition, arm="state_transplant", challenge="neutral_damage", age=0)
    )
    identity = bool(self_signature) and self_signature == transplant_signature
    reliability = _reliability(cross0, cross1)
    def lower(name: str) -> float:
        return float(summaries[name]["simultaneous_lower"])
    checks = {
        "acquisition": min(acquisition.values()) >= float(gates["acquisition_min"])
            and abs(acquisition["A"] - acquisition["B"]) <= float(gates["acquisition_imbalance_max"]),
        "absolute_hold8": min(hold.values()) >= float(gates["absolute_hold8_min"]),
        "ceiling_fraction": min(ceiling_fraction.values()) >= float(gates["ceiling_fraction_min"]),
        "risk": summaries["risk_gain"]["mean"] >= float(gates["risk_gain_min"]) and lower("risk_gain") > 0,
        "crossover": summaries["direct_crossover"]["mean"] >= float(gates["crossover_min"]) and lower("direct_crossover") > 0,
        "logloss": summaries["history_logloss_gain"]["mean"] >= float(gates["logloss_gain_min"]) and lower("history_logloss_gain") > 0,
        "reliability": reliability >= float(gates["reliability_min"]),
        "self_transplant_pathwise_identity": identity,
        "beats_shuffle": lower("shuffle_contrast") > 0,
        "age1_risk": lower("age1_risk_gain") > 0,
        "age1_crossover": lower("age1_crossover") > 0,
    }
    return {
        "condition": condition,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
        "acquisition": acquisition,
        "hold8": hold,
        "injection_ceiling": ceiling,
        "ceiling_fraction": ceiling_fraction,
        "metrics": summaries,
        "split_half_crossover_reliability": reliability,
        "self_transplant_pathwise_identity": identity,
        "checks": checks,
        "failed": [name for name, passed in checks.items() if not passed],
    }


def analyze_state(registration: Registration, records: list[Record]) -> dict[str, Any]:
    conditions = [str(row["name"]) for row in registration.protocol["state"]["writers"]]
    results = {name: _state_writer_analysis(registration, records, name) for name in conditions}
    return {
        "format": "wagner-memory-state-analysis-v2",
        "cells": len(records),
        "futures": sum(int(row["n"]) for row in records),
        "writers": results,
        "state_channel_verdict": "STATE_CHANNEL_CONFIRMED" if results["hard-theta-0"]["verdict"] == "PASS" else "STATE_CHANNEL_NOT_CONFIRMED",
        "soft_writer_verdict": "SOFT_WRITER_SUPPORTED" if results["soft-theta-0"]["verdict"] == "PASS" else "SOFT_WRITER_NOT_CONFIRMED",
    }


def analyze_boundary(registration: Registration, records: list[Record]) -> dict[str, Any]:
    theta_values = [float(value) for value in registration.protocol["boundary"]["thetas"]]
    modes: dict[str, Any] = {}
    for mode in registration.protocol["boundary"]["writers"]:
        per_theta: dict[float, np.ndarray] = {}
        summaries: dict[str, Any] = {}
        for theta in theta_values:
            condition = f"{mode}-theta-{theta:g}"
            _, first, second = _paired_half_arrays(
                records, "state_transplant", "reset_both", condition=condition
            )
            values = _pooled(first, second)
            per_theta[theta] = values
            summaries[f"{theta:g}"] = _bootstrap_point(
                values, int(registration.profile["bootstrap_repetitions"]),
                generator(str(registration.protocol["master_seed"]), "boundary", mode, theta),
            )
        matrix = np.column_stack([per_theta[theta] for theta in theta_values])
        slopes = np.asarray([np.polyfit(theta_values, row, 1)[0] for row in matrix])
        endpoint = matrix[:, 0] - matrix[:, -1]
        slope_summary = _bootstrap_point(
            slopes, int(registration.profile["bootstrap_repetitions"]),
            generator(str(registration.protocol["master_seed"]), "boundary-slope", mode),
        )
        endpoint_summary = _bootstrap_point(
            endpoint, int(registration.profile["bootstrap_repetitions"]),
            generator(str(registration.protocol["master_seed"]), "boundary-endpoint", mode),
        )
        passed = bool(slope_summary["upper"] < 0 and endpoint_summary["lower"] > 0)
        modes[mode] = {
            "theta_effects": summaries,
            "slope": slope_summary,
            "theta0_minus_theta0.1": endpoint_summary,
            "pass": passed,
        }
    return {
        "format": "wagner-memory-boundary-analysis-v2",
        "cells": len(records),
        "futures": sum(int(row["n"]) for row in records),
        "modes": modes,
        "verdict": "NOISE_BOUNDARY_REPRODUCED" if modes["persistent_hard"]["pass"] else "NOISE_BOUNDARY_NOT_REPRODUCED",
    }


def analyze_slow_mark(registration: Registration, records: list[Record]) -> dict[str, Any]:
    gates = registration.protocol["gates"]
    settings: dict[str, Any] = {}
    any_pass = False
    for half_life in registration.protocol["slow_mark"]["half_lives"]:
        for coupling in registration.protocol["slow_mark"]["couplings"]:
            condition = f"half-{int(half_life)}.mu-{float(coupling):g}"
            screen_criteria = {
                "condition": condition, "schedule": "screen", "age": 8,
                "challenge": "forced_break",
            }
            _, risk0, risk1 = _paired_half_arrays(
                records, "mark_transplant", "reset_both", **screen_criteria
            )
            _, cross0, cross1 = _direct_half_arrays(
                records, "mark_transplant", **screen_criteria
            )
            _, logloss = _history_logloss_gain_values(
                records, "mark_transplant", **screen_criteria
            )
            mechanism = {"condition": condition, "schedule": "mechanism", "age": 8}
            family: dict[str, np.ndarray] = {
                "risk_gain": _pooled(risk0, risk1),
                "direct_crossover": _pooled(cross0, cross1),
                "history_logloss_gain": logloss,
            }
            controls = (
                "reset_both", "mark_pattern_shuffle", "write_disabled", "mark_inert"
            )
            for control in controls:
                _, first, second = _paired_half_arrays(
                    records, "state_and_mark_transplant", control, **mechanism
                )
                family[f"beats_{control}"] = _pooled(first, second)
            _, targeted0, targeted1 = _paired_half_arrays(
                records, "mark_random_ablation", "mark_ablation", **mechanism
            )
            family["targeted_more_damaging"] = _pooled(targeted0, targeted1)
            _, rescue0, rescue1 = _paired_half_arrays(
                records, "mark_rescue", "mark_ablation", **mechanism
            )
            family["rescue_restores"] = _pooled(rescue0, rescue1)
            summaries = _simultaneous_family(registration, family, ("slow-mark", condition))
            reliability = _reliability(cross0, cross1)
            acquisition = {
                history: float(np.mean([
                    float(row["acquired_exact"])
                    for row in _matching(
                        records, condition=condition, schedule="screen",
                        arm="mark_transplant", history=history, age=0,
                    )
                ]))
                for history in HISTORIES
            }
            hold = {
                history: _mean_strict_correct(
                    records, condition=condition, schedule="screen",
                    arm="mark_transplant", history=history, age=8,
                    challenge="forced_break",
                )
                for history in HISTORIES
            }
            checks = {
                "acquisition": min(acquisition.values()) >= float(gates["acquisition_min"])
                    and abs(acquisition["A"] - acquisition["B"]) <= float(gates["acquisition_imbalance_max"]),
                "absolute_hold8": min(hold.values()) >= float(gates["absolute_hold8_min"]),
                "risk": summaries["risk_gain"]["mean"] >= float(gates["risk_gain_min"])
                    and summaries["risk_gain"]["simultaneous_lower"] > 0,
                "crossover": summaries["direct_crossover"]["mean"] >= float(gates["crossover_min"])
                    and summaries["direct_crossover"]["simultaneous_lower"] > 0,
                "logloss": summaries["history_logloss_gain"]["mean"] >= float(gates["logloss_gain_min"])
                    and summaries["history_logloss_gain"]["simultaneous_lower"] > 0,
                "reliability": reliability >= float(gates["reliability_min"]),
                "registered_controls": all(
                    summaries[f"beats_{control}"]["simultaneous_lower"] > 0 for control in controls
                ),
                "targeted_ablation": summaries["targeted_more_damaging"]["simultaneous_lower"] > 0,
                "rescue": summaries["rescue_restores"]["simultaneous_lower"] > 0,
            }
            passed = all(checks.values())
            any_pass |= passed
            settings[condition] = {
                "acquisition": acquisition,
                "hold8": hold,
                "metrics": summaries,
                "split_half_crossover_reliability": reliability,
                "checks": checks,
                "pass": passed,
            }
    return {
        "format": "wagner-memory-slow-mark-analysis-v2",
        "cells": len(records),
        "futures": sum(int(row["n"]) for row in records),
        "settings": settings,
        "verdict": "SLOW_MARK_CANDIDATE" if any_pass else "NO_SLOW_MARK_CONFIRMED",
    }


def _carrier_risk(
    records: list[Record], arm: str, checkpoint: int, challenge: str | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    criteria: dict[str, Any] = {"checkpoint": checkpoint}
    if challenge is not None:
        criteria["challenge"] = challenge
    return _paired_half_arrays(records, arm, "zero", **criteria)


def analyze_carrier(registration: Registration, records: list[Record]) -> dict[str, Any]:
    gates = registration.protocol["gates"]
    _, risk0, risk1 = _carrier_risk(records, "natural_full", 4)
    _, cross0, cross1 = _direct_half_arrays(records, "natural_full", checkpoint=4)
    _, logloss = _history_logloss_gain_values(records, "natural_full", checkpoint=4)
    family: dict[str, np.ndarray] = {
        "generation4_risk_gain": _pooled(risk0, risk1),
        "generation4_direct_crossover": _pooled(cross0, cross1),
        "generation4_history_logloss_gain": logloss,
    }
    persistence_keys: list[str] = []
    for checkpoint in (8, 16):
        for challenge in registration.protocol["carrier"]["challenges"]:
            _, first, second = _carrier_risk(records, "natural_full", checkpoint, challenge)
            risk_key = f"g{checkpoint}.{challenge}.risk"
            family[risk_key] = _pooled(first, second)
            persistence_keys.append(risk_key)
            _, first, second = _direct_half_arrays(
                records, "natural_full", checkpoint=checkpoint, challenge=challenge
            )
            cross_key = f"g{checkpoint}.{challenge}.crossover"
            family[cross_key] = _pooled(first, second)
            persistence_keys.append(cross_key)
    control_arms = ("pattern_shuffle", "write_disabled", "read_disabled", "no_rewrite", "zero")
    control_keys: list[str] = []
    for control in control_arms:
        _, first, second = _paired_half_arrays(
            records, "natural_full", control, checkpoint=4
        )
        key = f"beats_{control}"
        family[key] = _pooled(first, second)
        control_keys.append(key)
    _, opposite0, opposite1 = _direct_half_arrays(records, "opposite_history", checkpoint=4)
    family["opposite_reversal"] = -_pooled(opposite0, opposite1)
    _, ablate0, ablate1 = _carrier_risk(records, "ablate_generation_2", 4)
    _, rescue0, rescue1 = _carrier_risk(records, "ablate_2_rescue_3", 4)
    family["ablation_effect"] = _pooled(ablate0, ablate1)
    family["rescue_effect"] = _pooled(rescue0, rescue1)
    bottleneck_arms = (
        "targeted_k5", "targeted_k3", "targeted_k1",
        "random_k5", "random_k3", "random_k1",
    )
    for arm in bottleneck_arms:
        _, first, second = _carrier_risk(records, arm, 4)
        family[arm] = _pooled(first, second)
    summaries = _simultaneous_family(registration, family, ("carrier", "primary-family"))
    reliability = _reliability(cross0, cross1)
    acquisition = {
        history: float(np.mean([
            float(row["acquired_exact"])
            for row in _matching(records, arm="natural_full", history=history, checkpoint=0)
        ]))
        for history in HISTORIES
    }
    full_mean = float(summaries["generation4_risk_gain"]["mean"])
    ablation_mean = float(summaries["ablation_effect"]["mean"])
    rescue_mean = float(summaries["rescue_effect"]["mean"])
    k5_mean = float(summaries["targeted_k5"]["mean"])
    ablation_loss = 1.0 - ablation_mean / max(abs(full_mean), 1e-12)
    rescue_fraction = (rescue_mean - ablation_mean) / max(
        abs(full_mean - ablation_mean), 1e-12
    )
    k5_fraction = k5_mean / max(abs(full_mean), 1e-12)
    primary_checks = {
        "acquisition": min(acquisition.values()) >= float(gates["acquisition_min"])
            and abs(acquisition["A"] - acquisition["B"]) <= float(gates["acquisition_imbalance_max"]),
        "generation4_risk": full_mean >= float(gates["risk_gain_min"])
            and summaries["generation4_risk_gain"]["simultaneous_lower"] > 0,
        "generation4_crossover": summaries["generation4_direct_crossover"]["mean"] >= float(gates["crossover_min"])
            and summaries["generation4_direct_crossover"]["simultaneous_lower"] > 0,
        "generation4_logloss": summaries["generation4_history_logloss_gain"]["mean"] >= float(gates["logloss_gain_min"])
            and summaries["generation4_history_logloss_gain"]["simultaneous_lower"] > 0,
        "split_half_reliability": reliability >= float(gates["reliability_min"]),
        "generation8_and_16": all(summaries[key]["simultaneous_lower"] > 0 for key in persistence_keys),
    }
    causal_checks = {
        "registered_controls": all(summaries[key]["simultaneous_lower"] > 0 for key in control_keys),
        "opposite_history_reverses": summaries["opposite_reversal"]["simultaneous_lower"] > 0,
        "ablation_loss": ablation_loss >= float(gates["ablation_loss_fraction_min"]),
        "rescue": rescue_fraction >= float(gates["rescue_fraction_min"]),
    }
    primary_pass = all(primary_checks.values())
    causal_pass = primary_pass and all(causal_checks.values())
    distributed = k5_fraction < float(gates["bottleneck_retention_max"])
    return {
        "format": "wagner-memory-carrier-analysis-v2",
        "cells": len(records),
        "futures": sum(int(row["n"]) for row in records),
        "acquisition": acquisition,
        "metrics": summaries,
        "split_half_crossover_reliability": reliability,
        "ablation_loss_fraction": ablation_loss,
        "rescue_fraction": rescue_fraction,
        "targeted_k5_retention_fraction": k5_fraction,
        "primary_checks": primary_checks,
        "causal_checks": causal_checks,
        "carrier_verdict": "LINEAGE_CARRIER_CONFIRMED" if primary_pass else "LINEAGE_CARRIER_NOT_CONFIRMED",
        "causal_verdict": "CAUSAL_CARRIER_SUPPORTED" if causal_pass else "CAUSAL_CARRIER_NOT_SUPPORTED",
        "distributed_verdict": "DISTRIBUTED_CARRIER_SUPPORTED" if primary_pass and distributed else "DISTRIBUTED_CARRIER_NOT_SUPPORTED",
    }


def load_stage_records(run_dir: Path, stage: str) -> list[Record]:
    paths = [
        path
        for path in sorted((run_dir / "stages" / stage).glob("worker-*.jsonl.gz"))
        if ".sources." not in path.name
    ]
    if not paths:
        raise FileNotFoundError(f"no {stage} worker records in {run_dir}")
    return read_records(paths)


def analyze_all(registration: Registration, run_dir: Path) -> dict[str, Any]:
    state = analyze_state(registration, load_stage_records(run_dir, "state"))
    boundary = analyze_boundary(registration, load_stage_records(run_dir, "boundary"))
    slow_mark = analyze_slow_mark(registration, load_stage_records(run_dir, "slow_mark"))
    carrier = analyze_carrier(registration, load_stage_records(run_dir, "carrier"))
    scientific_confirmation = (
        state["state_channel_verdict"] == "STATE_CHANNEL_CONFIRMED"
        and carrier["carrier_verdict"] == "LINEAGE_CARRIER_CONFIRMED"
        and carrier["causal_verdict"] == "CAUSAL_CARRIER_SUPPORTED"
    )
    return {
        "format": "wagner-memory-campaign-analysis-v2",
        "scientific": registration.scientific,
        "state": state,
        "boundary": boundary,
        "slow_mark": slow_mark,
        "carrier": carrier,
        "overall_verdict": (
            "WAGNER_MEMORY_STACK_CONFIRMED" if scientific_confirmation
            else "WAGNER_MEMORY_STACK_NOT_CONFIRMED"
        ) if registration.scientific else "NON_SCIENTIFIC_DIAGNOSTIC_ONLY",
    }
