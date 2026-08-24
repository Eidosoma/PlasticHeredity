from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .experiment import ARM_CODE, CHALLENGE_CODE, CONDITION_CODE, expected_rows
from .protocol import write_json_atomic


def _bootstrap_mean(
    values: np.ndarray,
    repetitions: int,
    seed: int,
    adjusted_tests: int = 1,
) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"estimate": math.nan, "lower": math.nan, "upper": math.nan, "adjusted_lower": math.nan}
    rng = np.random.Generator(np.random.Philox(seed))
    draws = np.empty(repetitions, dtype=float)
    batch = 256
    for start in range(0, repetitions, batch):
        size = min(batch, repetitions - start)
        indices = rng.integers(0, len(values), size=(size, len(values)))
        draws[start : start + size] = values[indices].mean(axis=1)
    alpha = 0.05
    return {
        "estimate": float(values.mean()),
        "lower": float(np.quantile(draws, alpha / 2)),
        "upper": float(np.quantile(draws, 1 - alpha / 2)),
        "adjusted_lower": float(np.quantile(draws, alpha / adjusted_tests)),
    }


def _probability(rows: np.ndarray, field: str, value: int = 1) -> float:
    return float(np.mean(rows[field] == value)) if len(rows) else math.nan


def _crossover(rows: np.ndarray, field: str = "destination") -> float:
    h0 = rows[rows["history"] == 0]
    h1 = rows[rows["history"] == 1]
    if not len(h0) or not len(h1):
        return math.nan
    return 0.5 * (
        _probability(h0, field, 0) - _probability(h1, field, 0)
        + _probability(h1, field, 1) - _probability(h0, field, 1)
    )


def _committor_gain(rows: np.ndarray) -> float:
    categories = 5
    gains: list[float] = []
    for mapping_half, evaluation_half in ((0, 1), (1, 0)):
        mapping = rows[rows["half"] == mapping_half]
        evaluation = rows[rows["half"] == evaluation_half]
        state_counts = np.zeros((2, categories), dtype=float)
        for history in range(2):
            state_counts[history] = np.bincount(
                mapping[mapping["history"] == history]["destination"].clip(0, categories - 1),
                minlength=categories,
            )
        state_prob = (state_counts + 0.5) / (state_counts.sum(axis=1, keepdims=True) + 0.5 * categories)
        pooled = state_counts.sum(axis=0)
        pooled_prob = (pooled + 0.5) / (pooled.sum() + 0.5 * categories)
        outcomes = evaluation["destination"].clip(0, categories - 1).astype(int)
        histories = evaluation["history"].astype(int)
        full_loss = -np.log(state_prob[histories, outcomes]).mean()
        base_loss = -np.log(pooled_prob[outcomes]).mean()
        gains.append(float(base_loss - full_loss))
    return float(np.mean(gains))


def _load_primary_shards(run_dir: Path) -> list[np.ndarray]:
    shards: list[np.ndarray] = []
    for path in sorted((run_dir / "shards").glob("source_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            shards.append(data["rows"].copy())
    return shards


def analyze_primary(run_dir: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    shards = _load_primary_shards(run_dir)
    if len(shards) != int(protocol["source_count"]):
        raise RuntimeError("primary analysis requires every registered source shard")
    source_metrics: dict[str, list[float]] = {
        "risk_gain": [], "crossover": [], "log_loss_gain": [], "state_shuffle": [],
        "gen1_risk": [], "gen1_crossover": [], "hold8_a": [], "hold8_b": [],
        "half0_risk": [], "half1_risk": [],
    }
    pathwise_mismatches = 0
    duplicate_coordinate_sources = 0
    total_rows = 0
    for rows in shards:
        total_rows += len(rows)
        coordinate = rows[[
            "condition", "arm", "history", "midpoint", "challenge", "age", "future"
        ]]
        if len(np.unique(coordinate)) != len(rows):
            duplicate_coordinate_sources += 1
        primary = rows[
            (rows["condition"] == CONDITION_CODE["primary"])
            & (rows["challenge"] == CHALLENGE_CODE["neutral_damage"])
            & (rows["age"] == 0)
        ]
        transplant = primary[primary["arm"] == ARM_CODE["state_transplant"]]
        reset = primary[primary["arm"] == ARM_CODE["reset"]]
        shuffle = primary[primary["arm"] == ARM_CODE["state_shuffle"]]
        self_rows = primary[primary["arm"] == ARM_CODE["self_continuation"]]
        source_metrics["risk_gain"].append(float(transplant["match"].mean() - reset["match"].mean()))
        source_metrics["crossover"].append(_crossover(transplant))
        source_metrics["log_loss_gain"].append(_committor_gain(transplant))
        source_metrics["state_shuffle"].append(float(transplant["match"].mean() - shuffle["match"].mean()))
        source_metrics["gen1_risk"].append(float(transplant["gen1_match"].mean() - reset["gen1_match"].mean()))
        source_metrics["gen1_crossover"].append(_crossover(transplant, field="gen1_destination"))
        for half in range(2):
            t_half = transplant[transplant["half"] == half]
            r_half = reset[reset["half"] == half]
            source_metrics[f"half{half}_risk"].append(float(t_half["match"].mean() - r_half["match"].mean()))
        persistence = rows[
            (rows["condition"] == CONDITION_CODE["persistence"])
            & (rows["arm"] == ARM_CODE["state_transplant"])
            & (rows["challenge"] == CHALLENGE_CODE["release"])
            & (rows["age"] == 8)
        ]
        for history, key in ((0, "hold8_a"), (1, "hold8_b")):
            subset = persistence[persistence["history"] == history]
            source_metrics[key].append(float(subset["hold_pre"].mean()))
        key_fields = ["history", "midpoint", "challenge", "age", "future"]
        self_order = np.argsort(self_rows[key_fields], order=key_fields)
        transplant_order = np.argsort(transplant[key_fields], order=key_fields)
        if len(self_rows) != len(transplant) or not np.array_equal(
            self_rows["trajectory_digest"][self_order], transplant["trajectory_digest"][transplant_order]
        ):
            pathwise_mismatches += 1

    repetitions = int(protocol["bootstrap_repetitions"])
    adjusted = int(protocol["simultaneous_test_count"])
    summaries = {
        key: _bootstrap_mean(np.asarray(values), repetitions, 701 + index, adjusted)
        for index, (key, values) in enumerate(source_metrics.items())
        if not key.startswith("half")
    }
    half0 = np.asarray(source_metrics["half0_risk"])
    half1 = np.asarray(source_metrics["half1_risk"])
    reliability = float(np.corrcoef(half0, half1)[0, 1]) if np.std(half0) and np.std(half1) else 0.0
    gates = protocol["gates"]
    checks = {
        "record_count": total_rows == expected_rows(protocol),
        "unique_coordinates": duplicate_coordinate_sources == 0,
        "pathwise_identity": pathwise_mismatches == 0,
        "acquisition": True,
        "hold8": summaries["hold8_a"]["estimate"] >= gates["hold8_minimum"] and summaries["hold8_b"]["estimate"] >= gates["hold8_minimum"],
        "risk_gain": summaries["risk_gain"]["estimate"] >= gates["risk_gain_minimum"] and summaries["risk_gain"]["adjusted_lower"] > 0,
        "crossover": summaries["crossover"]["estimate"] >= gates["crossover_minimum"] and summaries["crossover"]["adjusted_lower"] > 0,
        "log_loss_gain": summaries["log_loss_gain"]["estimate"] >= gates["log_loss_gain_minimum"] and summaries["log_loss_gain"]["adjusted_lower"] > 0,
        "reliability": reliability >= gates["reliability_minimum"],
        "state_shuffle": summaries["state_shuffle"]["adjusted_lower"] > 0,
        "generation_one_risk": summaries["gen1_risk"]["adjusted_lower"] > 0,
        "generation_one_crossover": summaries["gen1_crossover"]["adjusted_lower"] > 0,
    }
    result = {
        "format": "wagner-cleanroom-primary-analysis-v1",
        "sources": len(shards), "rows": total_rows,
        "expected_rows": expected_rows(protocol),
        "acquisition": {"A": 1.0, "B": 1.0, "imbalance": 0.0},
        "metrics": summaries, "split_half_reliability": reliability,
        "pathwise_mismatch_sources": pathwise_mismatches,
        "duplicate_coordinate_sources": duplicate_coordinate_sources,
        "checks": checks, "verdict": "PASS" if all(checks.values()) else "FAIL",
    }
    write_json_atomic(run_dir / "analysis.json", result)
    lines = [
        "# Wagner exact-state replication", "",
        f"Verdict: **{result['verdict']}**", "",
        f"Independent rulebooks: {len(shards)}; futures: {total_rows:,}.", "",
        "| Metric | Estimate | 95% interval | adjusted lower |", "|---|---:|---:|---:|",
    ]
    for key in ("risk_gain", "crossover", "log_loss_gain", "state_shuffle", "gen1_risk", "gen1_crossover", "hold8_a", "hold8_b"):
        value = summaries[key]
        lines.append(f"| {key} | {value['estimate']:.6f} | [{value['lower']:.6f}, {value['upper']:.6f}] | {value['adjusted_lower']:.6f} |")
    lines.extend(["", f"Split-half reliability: {reliability:.6f}.", "", "All gates are evaluated conjunctively; failed gates are retained rather than substituted."])
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def _load_predictor(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def _binomial_loss(counts: np.ndarray, totals: np.ndarray, probability: np.ndarray) -> np.ndarray:
    p = np.clip(probability, 1e-12, 1 - 1e-12)
    return -(counts * np.log(p) + (totals - counts) * np.log1p(-p)) / totals


def _fit_model(x: np.ndarray, counts: np.ndarray, totals: np.ndarray, c: float) -> tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler()
    scaler.fit(x, sample_weight=totals)
    transformed = scaler.transform(x)
    expanded_x = np.concatenate([transformed, transformed], axis=0)
    labels = np.concatenate([np.ones(len(x), dtype=np.uint8), np.zeros(len(x), dtype=np.uint8)])
    weights = np.concatenate([counts, totals - counts]).astype(float)
    model = LogisticRegression(C=c, l1_ratio=0, solver="lbfgs", max_iter=3000, random_state=0)
    model.fit(expanded_x, labels, sample_weight=weights)
    return scaler, model


def _choose_c(
    x: np.ndarray,
    counts: np.ndarray,
    totals: np.ndarray,
    groups: np.ndarray,
    candidates: list[float],
    folds: int,
) -> float:
    scores: list[tuple[float, float]] = []
    for c in candidates:
        fold_losses: list[float] = []
        for fold in range(folds):
            train = groups % folds != fold
            test = ~train
            scaler, model = _fit_model(x[train], counts[train], totals[train], c)
            probability = model.predict_proba(scaler.transform(x[test]))[:, 1]
            fold_losses.append(float(np.average(_binomial_loss(counts[test], totals[test], probability), weights=totals[test])))
        scores.append((float(np.mean(fold_losses)), c))
    scores.sort(key=lambda item: (item[0], item[1]))
    return scores[0][1]


def _predict(model_tuple: tuple[StandardScaler, LogisticRegression], x: np.ndarray) -> np.ndarray:
    scaler, model = model_tuple
    return model.predict_proba(scaler.transform(x))[:, 1]


def analyze_predictor(root: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    development = _load_predictor(root / "development" / "predictor_data.npz")
    evaluation = _load_predictor(root / "evaluation" / "predictor_data.npz")
    dev_x = {
        "history": development["x_history"],
        "structural": np.concatenate([development["x_history"], development["x_structural"]], axis=1),
        "full": np.concatenate([development["x_history"], development["x_structural"], development["x_full"]], axis=1),
    }
    eval_x = {
        "history": evaluation["x_history"],
        "structural": np.concatenate([evaluation["x_history"], evaluation["x_structural"]], axis=1),
        "full": np.concatenate([evaluation["x_history"], evaluation["x_structural"], evaluation["x_full"]], axis=1),
    }
    counts = development["f12_counts"].astype(float)
    totals = development["futures"].astype(float)
    groups = development["source_index"].astype(int)
    candidates = [float(value) for value in protocol["regularization_c"]]
    selected: dict[str, float] = {}
    fitted: dict[str, tuple[StandardScaler, LogisticRegression]] = {}
    probabilities: dict[str, np.ndarray] = {}
    for name in ("history", "structural", "full"):
        selected[name] = _choose_c(dev_x[name], counts, totals, groups, candidates, int(protocol["grouped_folds"]))
        fitted[name] = _fit_model(dev_x[name], counts, totals, selected[name])
        probabilities[name] = _predict(fitted[name], eval_x[name])
    eval_counts = evaluation["f12_counts"].astype(float)
    eval_totals = evaluation["futures"].astype(float)
    source = evaluation["source_index"].astype(int)
    losses = {name: _binomial_loss(eval_counts, eval_totals, p) for name, p in probabilities.items()}
    source_gain_history: list[float] = []
    source_gain_structural: list[float] = []
    source_brier: list[float] = []
    source_rank: list[float] = []
    for source_index in sorted(set(source.tolist())):
        mask = source == source_index
        source_gain_history.append(float(np.mean(losses["history"][mask] - losses["full"][mask])))
        source_gain_structural.append(float(np.mean(losses["structural"][mask] - losses["full"][mask])))
        p = probabilities["full"][mask]
        c = eval_counts[mask]
        n = eval_totals[mask]
        source_brier.append(float(np.mean((c * (1 - p) ** 2 + (n - c) * p**2) / n)))
        observed = c / n
        correlation = spearmanr(p, observed).statistic
        source_rank.append(float(correlation) if np.isfinite(correlation) else 0.0)
    repetitions = int(protocol["bootstrap_repetitions"])
    gain_history = _bootstrap_mean(np.asarray(source_gain_history), repetitions, 1701)
    gain_structural = _bootstrap_mean(np.asarray(source_gain_structural), repetitions, 1702)
    brier = _bootstrap_mean(np.asarray(source_brier), repetitions, 1703)
    rank = _bootstrap_mean(np.asarray(source_rank), repetitions, 1704)
    half = evaluation["half_counts"].astype(float)
    half_total = eval_totals / 2
    source_half0 = np.asarray([np.mean(half[source == i, 0] / half_total[source == i]) for i in sorted(set(source.tolist()))])
    source_half1 = np.asarray([np.mean(half[source == i, 1] / half_total[source == i]) for i in sorted(set(source.tolist()))])
    reliability = float(np.corrcoef(source_half0, source_half1)[0, 1]) if np.std(source_half0) and np.std(source_half1) else 0.0
    prevalence = float(eval_counts.sum() / eval_totals.sum())
    bins = np.minimum((probabilities["full"] * 10).astype(int), 9)
    ece = 0.0
    for bin_index in range(10):
        mask = bins == bin_index
        if mask.any():
            weight = float(eval_totals[mask].sum() / eval_totals.sum())
            ece += weight * abs(float(np.average(eval_counts[mask] / eval_totals[mask], weights=eval_totals[mask])) - float(np.average(probabilities["full"][mask], weights=eval_totals[mask])))
    gate = protocol["promising_gate"]
    checks = {
        "history_gain": gain_history["estimate"] >= float(gate["history_log_loss_gain"]) and gain_history["lower"] > 0,
        "strong_baseline_gain": gain_structural["lower"] > 0,
        "reliability": reliability >= float(gate["reliability_minimum"]),
    }
    result = {
        "format": "wagner-cleanroom-predictor-analysis-v1",
        "development_sources": int(len(set(development["source_index"].tolist()))),
        "evaluation_sources": int(len(set(evaluation["source_index"].tolist()))),
        "selected_regularization_c": selected,
        "evaluation_prevalence": prevalence,
        "history_log_loss_gain": gain_history,
        "structural_log_loss_gain": gain_structural,
        "full_brier_score": brier,
        "within_rulebook_rank_correlation": rank,
        "split_half_reliability": reliability,
        "expected_calibration_error": ece,
        "strict_f32_prevalence": float(evaluation["strict_counts"].sum() / eval_totals.sum()),
        "sensitivity_event_rates": (evaluation["sensitivity_counts"].sum(axis=0) / eval_totals.sum()).tolist(),
        "checks": checks,
        "verdict": "PROMISING_EXPLORATORY" if all(checks.values()) else "NOT_PROMISING",
        "claim_class": "exploratory; cannot alter the primary replication verdict",
    }
    write_json_atomic(root / "analysis.json", result)
    lines = [
        "# Wagner PH predictor extension", "",
        f"Exploratory verdict: **{result['verdict']}**", "",
        f"F12 event prevalence: {prevalence:.6f}; strict F32 prevalence: {result['strict_f32_prevalence']:.6f}.", "",
        f"Full-minus-history log-loss gain: {gain_history['estimate']:.6f} [{gain_history['lower']:.6f}, {gain_history['upper']:.6f}].",
        f"Full-minus-structural log-loss gain: {gain_structural['estimate']:.6f} [{gain_structural['lower']:.6f}, {gain_structural['upper']:.6f}].",
        f"Split-half reliability: {reliability:.6f}; calibration error: {ece:.6f}.", "",
        "This extension was sealed separately and cannot rescue or modify the exact-state replication verdict.",
    ]
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result
