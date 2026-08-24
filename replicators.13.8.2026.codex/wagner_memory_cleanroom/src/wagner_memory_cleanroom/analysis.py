from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from scipy.stats import spearmanr

from .config import Registration
from .rng import generator
from .storage import read_records


Record = dict[str, Any]


def _matching(records: Iterable[Record], **criteria: Any) -> list[Record]:
    return [row for row in records if all(row.get(key) == value for key, value in criteria.items())]


def _rate(row: Record, kind: str = "correct") -> float:
    return float(row[kind]) / max(1, int(row["n"]))


def _source_half_values(records: Iterable[Record], score: Callable[[Record], float]) -> dict[tuple[int, int], float]:
    grouped: dict[tuple[int, int], list[tuple[float, int]]] = defaultdict(list)
    for row in records:
        grouped[(int(row["source_id"]), int(row["half"]))].append((score(row), int(row["n"])))
    result: dict[tuple[int, int], float] = {}
    for key, values in grouped.items():
        weights = np.asarray([item[1] for item in values], dtype=float)
        scores = np.asarray([item[0] for item in values], dtype=float)
        result[key] = float(np.average(scores, weights=weights))
    return result


def _paired_effect(
    records: list[Record],
    treatment: str,
    control: str,
    *,
    score: Callable[[Record], float] = _rate,
    **criteria: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    treated = _source_half_values(_matching(records, arm=treatment, **criteria), score)
    controls = _source_half_values(_matching(records, arm=control, **criteria), score)
    sources = sorted({key[0] for key in treated} & {key[0] for key in controls})
    half0: list[float] = []
    half1: list[float] = []
    for source_id in sources:
        if (source_id, 0) in treated and (source_id, 0) in controls and (source_id, 1) in treated and (source_id, 1) in controls:
            half0.append(treated[(source_id, 0)] - controls[(source_id, 0)])
            half1.append(treated[(source_id, 1)] - controls[(source_id, 1)])
    a = np.asarray(half0, dtype=float)
    b = np.asarray(half1, dtype=float)
    return np.asarray(sources[: min(a.size, b.size)], dtype=int), a, b


def _bootstrap_summary(values: np.ndarray, repeats: int, rng: np.random.Generator) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"n": 0, "mean": None, "lower": None, "upper": None}
    indices = rng.integers(0, finite.size, size=(repeats, finite.size))
    boot = np.mean(finite[indices], axis=1)
    return {
        "n": int(finite.size),
        "mean": float(np.mean(finite)),
        "lower": float(np.quantile(boot, 0.025)),
        "upper": float(np.quantile(boot, 0.975)),
    }


def _effect_summary(
    registration: Registration,
    records: list[Record],
    treatment: str,
    control: str,
    *,
    score: Callable[[Record], float] = _rate,
    label: tuple[Any, ...] = (),
    **criteria: Any,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    _, first, second = _paired_effect(records, treatment, control, score=score, **criteria)
    pooled = (first + second) / 2.0
    rng = generator(str(registration.protocol["master_seed"]), "analysis", *label)
    summary = _bootstrap_summary(pooled, int(registration.profile["bootstrap_repetitions"]), rng)
    summary["half_0_mean"] = float(np.mean(first)) if first.size else None
    summary["half_1_mean"] = float(np.mean(second)) if second.size else None
    return summary, first, second


def _reliability(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if first.size < 3 or second.size != first.size:
        return 0.0
    if np.allclose(first, second, atol=1e-12):
        return 1.0
    variance = float(np.var(np.concatenate([first, second]), ddof=1))
    if variance <= 1e-15:
        return 0.0
    error = float(np.var(first - second, ddof=1)) / 2.0
    return float(np.clip(1.0 - error / variance, -1.0, 1.0))


def _crossover(row: Record) -> float:
    return (_rate(row, "correct") - _rate(row, "wrong"))


def _crossfit_logloss_by_source(records: list[Record], arm: str, **criteria: Any) -> dict[int, float]:
    selected = _matching(records, arm=arm, **criteria)
    by_half = {half: [row for row in selected if int(row["half"]) == half] for half in (0, 1)}
    losses: dict[int, list[float]] = defaultdict(list)
    for evaluation_half in (0, 1):
        training = by_half[1 - evaluation_half]
        correct = sum(int(row["correct"]) for row in training)
        wrong = sum(int(row["wrong"]) for row in training)
        probability = (correct + 0.5) / max(1.0, correct + wrong + 1.0)
        probability = float(np.clip(probability, 1e-6, 1 - 1e-6))
        grouped: dict[int, list[Record]] = defaultdict(list)
        for row in by_half[evaluation_half]:
            grouped[int(row["source_id"])].append(row)
        for source_id, rows in grouped.items():
            total = sum(int(row["n"]) for row in rows)
            c = sum(int(row["correct"]) for row in rows)
            w = sum(int(row["wrong"]) for row in rows)
            u = total - c - w
            loss = -(c * math.log(probability) + w * math.log(1 - probability) + u * math.log(0.5)) / max(1, total)
            losses[source_id].append(float(loss))
    return {source: float(np.mean(value)) for source, value in losses.items() if len(value) == 2}


def _logloss_gain(
    registration: Registration,
    records: list[Record],
    treatment: str,
    control: str,
    *,
    label: tuple[Any, ...],
    **criteria: Any,
) -> dict[str, Any]:
    treated = _crossfit_logloss_by_source(records, treatment, **criteria)
    baseline = _crossfit_logloss_by_source(records, control, **criteria)
    sources = sorted(set(treated) & set(baseline))
    values = np.asarray([baseline[source] - treated[source] for source in sources])
    rng = generator(str(registration.protocol["master_seed"]), "analysis-logloss", *label)
    return _bootstrap_summary(values, int(registration.profile["bootstrap_repetitions"]), rng)


def _mean_rate(records: list[Record], **criteria: Any) -> float:
    rows = _matching(records, **criteria)
    total = sum(int(row["n"]) for row in rows)
    return sum(int(row["correct"]) for row in rows) / max(1, total)


def _state_writer_analysis(registration: Registration, records: list[Record], writer: str) -> dict[str, Any]:
    gates = registration.protocol["gates"]
    criteria = {"writer": writer, "challenge": "neutral_damage", "age": 0}
    risk, risk_h0, risk_h1 = _effect_summary(
        registration, records, "state_transplant", "reset", label=("state", writer, "risk"), **criteria
    )
    crossover, cross_h0, cross_h1 = _effect_summary(
        registration, records, "state_transplant", "reset", score=_crossover,
        label=("state", writer, "crossover"), **criteria
    )
    logloss = _logloss_gain(
        registration, records, "state_transplant", "reset", label=("state", writer), **criteria
    )
    shuffle, _, _ = _effect_summary(
        registration, records, "state_transplant", "pattern_shuffle", label=("state", writer, "shuffle"), **criteria
    )
    age_one_risk, _, _ = _effect_summary(
        registration, records, "state_transplant", "reset", label=("state", writer, "age1-risk"),
        writer=writer, age=1
    )
    age_one_cross, _, _ = _effect_summary(
        registration, records, "state_transplant", "reset", score=_crossover,
        label=("state", writer, "age1-cross"), writer=writer, age=1
    )
    acquisition = {
        history: float(np.mean([row["acquired"] for row in _matching(records, writer=writer, arm="state_transplant", history=history, age=0)]))
        for history in HISTORIES_LOCAL
    }
    hold = {
        history: _mean_rate(records, writer=writer, arm="state_transplant", history=history, challenge="neutral_damage", age=0)
        for history in HISTORIES_LOCAL
    }
    self_rows = _matching(records, writer=writer, arm="self", challenge="neutral_damage", age=0)
    transplant_rows = _matching(records, writer=writer, arm="state_transplant", challenge="neutral_damage", age=0)
    identity_fields = ("source_id", "history", "half", "n", "correct", "wrong", "both", "unresolved")
    self_signature = sorted(tuple(row[key] for key in identity_fields) for row in self_rows)
    transplant_signature = sorted(tuple(row[key] for key in identity_fields) for row in transplant_rows)
    identity = self_signature == transplant_signature and bool(self_signature)
    reliability = min(_reliability(risk_h0, risk_h1), _reliability(cross_h0, cross_h1))
    failures: list[str] = []
    checks = {
        "acquisition": min(acquisition.values()) >= float(gates["acquisition_min"]) and abs(acquisition["A"] - acquisition["B"]) <= float(gates["acquisition_imbalance_max"]),
        "hold8": min(hold.values()) >= float(gates["absolute_hold8_min"]),
        "risk": risk["mean"] is not None and risk["mean"] >= float(gates["risk_gain_min"]) and risk["lower"] > 0,
        "crossover": crossover["mean"] is not None and crossover["mean"] >= float(gates["crossover_min"]) and crossover["lower"] > 0,
        "logloss": logloss["mean"] is not None and logloss["mean"] >= float(gates["logloss_gain_min"]) and logloss["lower"] > 0,
        "reliability": reliability >= float(gates["reliability_min"]),
        "self_transplant_identity": identity,
        "beats_shuffle": shuffle["lower"] is not None and shuffle["lower"] > 0,
        "age_one_risk": age_one_risk["lower"] is not None and age_one_risk["lower"] > 0,
        "age_one_crossover": age_one_cross["lower"] is not None and age_one_cross["lower"] > 0,
    }
    failures.extend(name for name, passed in checks.items() if not passed)
    return {
        "writer": writer,
        "verdict": "PASS" if not failures else "FAIL",
        "acquisition": acquisition,
        "hold8": hold,
        "risk_gain": risk,
        "crossover_gain": crossover,
        "logloss_gain": logloss,
        "shuffle_contrast": shuffle,
        "age_one_risk_gain": age_one_risk,
        "age_one_crossover_gain": age_one_cross,
        "split_half_reliability": reliability,
        "self_transplant_pathwise_identity": identity,
        "checks": checks,
        "failed": failures,
    }


HISTORIES_LOCAL = ("A", "B")


def analyze_state(registration: Registration, records: list[Record]) -> dict[str, Any]:
    writers = [row["name"] for row in registration.protocol["state"]["writers"]]
    results = {writer: _state_writer_analysis(registration, records, writer) for writer in writers}
    return {
        "format": "wagner-memory-state-analysis-v1",
        "records": len(records),
        "futures": sum(int(row["n"]) for row in records),
        "writers": results,
        "state_channel_verdict": "STATE_CHANNEL_CONFIRMED" if results["hard-theta-0"]["verdict"] == "PASS" else "STATE_CHANNEL_NOT_CONFIRMED",
        "soft_writer_verdict": "SOFT_WRITER_SUPPORTED" if results["soft-theta-0"]["verdict"] == "PASS" else "SOFT_WRITER_NOT_CONFIRMED",
    }


def analyze_boundary(registration: Registration, records: list[Record]) -> dict[str, Any]:
    theta_values = [float(value) for value in registration.protocol["boundary"]["thetas"]]
    modes: dict[str, Any] = {}
    for mode in registration.protocol["boundary"]["writers"]:
        summaries: dict[str, Any] = {}
        per_theta: list[np.ndarray] = []
        for theta in theta_values:
            summary, first, second = _effect_summary(
                registration, records, "state_transplant", "reset", label=("boundary", mode, theta),
                theta=theta, writer=f"{mode}-theta-{theta:g}"
            )
            summaries[f"{theta:g}"] = summary
            per_theta.append((first + second) / 2.0)
        common = min((values.size for values in per_theta), default=0)
        slopes = np.asarray([
            np.polyfit(theta_values, [per_theta[index][source] for index in range(len(theta_values))], 1)[0]
            for source in range(common)
        ])
        endpoint = per_theta[0][:common] - per_theta[-1][:common] if common else np.asarray([])
        repeats = int(registration.profile["bootstrap_repetitions"])
        slope_summary = _bootstrap_summary(slopes, repeats, generator(str(registration.protocol["master_seed"]), "boundary-slope", mode))
        endpoint_summary = _bootstrap_summary(endpoint, repeats, generator(str(registration.protocol["master_seed"]), "boundary-endpoint", mode))
        passed = bool(slope_summary["upper"] is not None and slope_summary["upper"] < 0 and endpoint_summary["lower"] is not None and endpoint_summary["lower"] > 0)
        modes[mode] = {"theta_effects": summaries, "slope": slope_summary, "theta0_minus_theta01": endpoint_summary, "pass": passed}
    return {
        "format": "wagner-memory-boundary-analysis-v1",
        "records": len(records),
        "futures": sum(int(row["n"]) for row in records),
        "modes": modes,
        "verdict": "NOISE_BOUNDARY_REPRODUCED" if modes["hard"]["pass"] else "NOISE_BOUNDARY_NOT_REPRODUCED",
    }


def analyze_slow_mark(registration: Registration, records: list[Record]) -> dict[str, Any]:
    gates = registration.protocol["gates"]
    settings: dict[str, Any] = {}
    any_pass = False
    for half_life in registration.protocol["slow_mark"]["half_lives"]:
        for coupling in registration.protocol["slow_mark"]["couplings"]:
            label = f"half-{half_life}.mu-{coupling:g}"
            risk, first, second = _effect_summary(
                registration, records, "mark", "reset", label=("mark", half_life, coupling, "risk"),
                half_life=int(half_life), coupling=float(coupling)
            )
            crossover, _, _ = _effect_summary(
                registration, records, "mark", "reset", score=_crossover,
                label=("mark", half_life, coupling, "cross"), half_life=int(half_life), coupling=float(coupling)
            )
            reliability = _reliability(first, second)
            passed = bool(
                risk["mean"] is not None and risk["mean"] >= float(gates["risk_gain_min"]) and risk["lower"] > 0
                and crossover["mean"] >= float(gates["crossover_min"]) and crossover["lower"] > 0
                and reliability >= float(gates["reliability_min"])
            )
            any_pass |= passed
            settings[label] = {"risk_gain": risk, "crossover_gain": crossover, "split_half_reliability": reliability, "pass": passed}
    return {
        "format": "wagner-memory-slow-mark-analysis-v1",
        "records": len(records),
        "futures": sum(int(row["n"]) for row in records),
        "settings": settings,
        "verdict": "SLOW_MARK_CANDIDATE" if any_pass else "NO_SLOW_MARK_CONFIRMED",
    }


def _carrier_effect(
    registration: Registration,
    records: list[Record],
    arm: str,
    checkpoint: int,
    challenge: str | None,
    label: str,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    criteria: dict[str, Any] = {"checkpoint": checkpoint}
    if challenge is not None:
        criteria["challenge"] = challenge
    return _effect_summary(registration, records, arm, "zero", label=("carrier", label), **criteria)


def analyze_carrier(registration: Registration, records: list[Record]) -> dict[str, Any]:
    gates = registration.protocol["gates"]
    risk4, risk_h0, risk_h1 = _carrier_effect(registration, records, "natural_full", 4, None, "risk4")
    cross4, cross_h0, cross_h1 = _effect_summary(
        registration, records, "natural_full", "zero", score=_crossover,
        label=("carrier", "cross4"), checkpoint=4
    )
    logloss4 = _logloss_gain(
        registration, records, "natural_full", "zero", label=("carrier", "logloss4"), checkpoint=4
    )
    reliability = min(_reliability(risk_h0, risk_h1), _reliability(cross_h0, cross_h1))
    persistence: dict[str, Any] = {}
    persistence_pass = True
    for checkpoint in (8, 16):
        for challenge in registration.protocol["carrier"]["challenges"]:
            summary, _, _ = _carrier_effect(registration, records, "natural_full", checkpoint, challenge, f"g{checkpoint}-{challenge}")
            persistence[f"g{checkpoint}.{challenge}"] = summary
            persistence_pass &= summary["lower"] is not None and summary["lower"] > 0
    controls: dict[str, Any] = {}
    controls_pass = True
    for arm in ("pattern_shuffle", "zero", "write_disabled", "read_disabled", "no_rewrite"):
        summary, _, _ = _effect_summary(
            registration, records, "natural_full", arm, label=("carrier", "control", arm), checkpoint=4
        )
        controls[arm] = summary
        controls_pass &= summary["lower"] is not None and summary["lower"] > 0
    opposite_cross = _source_half_values(_matching(records, arm="opposite_history", checkpoint=4), _crossover)
    opposite_values = np.asarray([
        np.mean([opposite_cross[(source, half)] for half in HALVES_LOCAL])
        for source in sorted({key[0] for key in opposite_cross})
        if all((source, half) in opposite_cross for half in HALVES_LOCAL)
    ])
    opposite = _bootstrap_summary(
        opposite_values,
        int(registration.profile["bootstrap_repetitions"]),
        generator(str(registration.protocol["master_seed"]), "carrier-opposite"),
    )
    ablate, _, _ = _carrier_effect(registration, records, "ablate_generation_2", 4, None, "ablate")
    rescue, _, _ = _carrier_effect(registration, records, "ablate_2_rescue_3", 4, None, "rescue")
    k_effects: dict[str, Any] = {}
    for arm in ("targeted_k5", "targeted_k3", "targeted_k1", "random_k5", "random_k3", "random_k1"):
        k_effects[arm], _, _ = _carrier_effect(registration, records, arm, 4, None, arm)
    full_mean = float(risk4["mean"] or 0.0)
    ablation_loss = 1.0 - float(ablate["mean"] or 0.0) / max(abs(full_mean), 1e-12)
    rescue_fraction = float(rescue["mean"] or 0.0) / max(abs(full_mean), 1e-12)
    k5_fraction = float(k_effects["targeted_k5"]["mean"] or 0.0) / max(abs(full_mean), 1e-12)
    acquisition = {
        history: float(np.mean([row["acquired"] for row in _matching(records, arm="natural_full", history=history, checkpoint=0)]))
        for history in HISTORIES_LOCAL
    }
    primary_checks = {
        "acquisition": min(acquisition.values()) >= float(gates["acquisition_min"]) and abs(acquisition["A"] - acquisition["B"]) <= float(gates["acquisition_imbalance_max"]),
        "generation4_risk": risk4["mean"] is not None and risk4["mean"] >= float(gates["risk_gain_min"]) and risk4["lower"] > 0,
        "generation4_crossover": cross4["mean"] is not None and cross4["mean"] >= float(gates["crossover_min"]) and cross4["lower"] > 0,
        "generation4_logloss": logloss4["mean"] is not None and logloss4["mean"] >= float(gates["logloss_gain_min"]) and logloss4["lower"] > 0,
        "reliability": reliability >= float(gates["reliability_min"]),
        "generation8_and_16": persistence_pass,
    }
    causal_checks = {
        "negative_controls": controls_pass,
        "opposite_history_reverses": opposite["upper"] is not None and opposite["upper"] < 0,
        "ablation_loss": ablation_loss >= float(gates["ablation_loss_fraction_min"]),
        "rescue": rescue_fraction >= float(gates["rescue_fraction_min"]),
    }
    distributed = k5_fraction < float(gates["bottleneck_retention_max"])
    primary_pass = all(primary_checks.values())
    causal_pass = primary_pass and all(causal_checks.values())
    return {
        "format": "wagner-memory-carrier-analysis-v1",
        "records": len(records),
        "futures": sum(int(row["n"]) for row in records),
        "acquisition": acquisition,
        "generation4_risk_gain": risk4,
        "generation4_crossover_gain": cross4,
        "generation4_logloss_gain": logloss4,
        "split_half_reliability": reliability,
        "persistence": persistence,
        "control_contrasts": controls,
        "opposite_history_crossover": opposite,
        "ablation_effect": ablate,
        "rescue_effect": rescue,
        "ablation_loss_fraction": ablation_loss,
        "rescue_fraction": rescue_fraction,
        "bottlenecks": k_effects,
        "targeted_k5_retention_fraction": k5_fraction,
        "primary_checks": primary_checks,
        "causal_checks": causal_checks,
        "carrier_verdict": "LINEAGE_CARRIER_CONFIRMED" if primary_pass else "LINEAGE_CARRIER_NOT_CONFIRMED",
        "causal_verdict": "CAUSAL_CARRIER_SUPPORTED" if causal_pass else "CAUSAL_CARRIER_NOT_SUPPORTED",
        "distributed_verdict": "DISTRIBUTED_CARRIER_SUPPORTED" if primary_pass and distributed else "DISTRIBUTED_CARRIER_NOT_SUPPORTED",
    }


HALVES_LOCAL = (0, 1)


def load_stage_records(run_dir: Path, stage: str) -> list[Record]:
    paths = sorted((run_dir / "stages" / stage).glob("worker-*.jsonl.gz"))
    if not paths:
        raise FileNotFoundError(f"no {stage} worker records in {run_dir}")
    return read_records(paths)


def analyze_all(registration: Registration, run_dir: Path) -> dict[str, Any]:
    state = analyze_state(registration, load_stage_records(run_dir, "state"))
    boundary = analyze_boundary(registration, load_stage_records(run_dir, "boundary"))
    slow_mark = analyze_slow_mark(registration, load_stage_records(run_dir, "slow_mark"))
    carrier = analyze_carrier(registration, load_stage_records(run_dir, "carrier"))
    overall = (
        state["state_channel_verdict"] == "STATE_CHANNEL_CONFIRMED"
        and carrier["carrier_verdict"] == "LINEAGE_CARRIER_CONFIRMED"
        and carrier["causal_verdict"] == "CAUSAL_CARRIER_SUPPORTED"
    )
    return {
        "format": "wagner-memory-campaign-analysis-v1",
        "scientific": registration.scientific,
        "state": state,
        "boundary": boundary,
        "slow_mark": slow_mark,
        "carrier": carrier,
        "overall_verdict": "WAGNER_MEMORY_STACK_CONFIRMED" if overall else "WAGNER_MEMORY_STACK_NOT_CONFIRMED",
    }

