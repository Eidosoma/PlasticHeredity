from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .statistics import (
    binomial_brier, binomial_log_loss, bootstrap_mean, permutation_tests,
    prediction_correlations, split_half_reliability,
)
from .storage import load_npz, verify_run, write_json_atomic


def _contrast_by_network(events: np.ndarray, total: float, better: np.ndarray, comparator: np.ndarray, metric: str) -> np.ndarray:
    function = binomial_log_loss if metric == "logloss" else binomial_brier
    difference = function(events, total, comparator) - function(events, total, better)
    return difference.mean(axis=1)


def analyze_tier(run_dir: str | Path, tier: str, protocol: dict[str, Any]) -> dict[str, Any]:
    root = Path(run_dir)
    predictions = load_npz(root / "predictions" / f"{tier}.npz")
    networks, states = predictions["event_count"].shape
    futures = int(predictions["futures"][0])
    half_total = futures // 2
    master = str(protocol["master_seed_label"])
    repetitions = int(protocol["inference"]["bootstrap_repetitions"])
    adjusted_tests = 4
    contrasts: dict[str, dict[str, float]] = {}
    structural_contrasts: dict[str, dict[str, float]] = {}
    for half in range(2):
        events = predictions[f"event_half{half}"]
        for metric in ("logloss", "brier"):
            name = f"{metric}_half{half}"
            per_network = _contrast_by_network(
                events, half_total, predictions["full_event"], predictions["history_event"], metric
            )
            contrasts[name] = bootstrap_mean(
                per_network, repetitions, master, f"{tier}|full-history|{name}", adjusted_tests
            )
            structural_values = _contrast_by_network(
                events, half_total, predictions["full_event"], predictions["structural_event"], metric
            )
            structural_contrasts[name] = bootstrap_mean(
                structural_values, repetitions, master, f"{tier}|full-structural|{name}", adjusted_tests
            )
    rate = predictions["event_count"] / float(futures)
    reliability = split_half_reliability(
        predictions["event_half0"] / float(half_total),
        predictions["event_half1"] / float(half_total),
    )
    correlations = prediction_correlations(predictions["full_event"], rate, predictions["cue_index"])
    permutation = permutation_tests(
        predictions["event_half0"], predictions["event_half1"], half_total,
        predictions["full_event"], predictions["history_event"],
        int(protocol["inference"]["permutation_repetitions"]), master, tier,
    )
    audit_path = root / "audit" / f"{tier}.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {
        "complete": False, "pass": False, "error": "independent regeneration missing"
    }
    expected_networks = int(protocol["tiers"][tier]["confirmation_networks"])
    coordinates = np.stack((
        np.repeat(predictions["network_index"], states),
        predictions["cue_index"].reshape(-1), predictions["age"].reshape(-1),
    ), axis=1)
    complete = networks == expected_networks and len(np.unique(coordinates, axis=0)) == networks * states
    gates = protocol["gates"]
    checks = {
        "complete_unique_records": bool(complete),
        "independent_regeneration": bool(audit.get("pass", False)),
        "reliability": reliability["spearman_brown_q"] >= float(gates["reliability_minimum"]),
        "logloss_effect_half0": contrasts["logloss_half0"]["estimate"] >= float(gates["log_loss_gain_minimum"]),
        "logloss_effect_half1": contrasts["logloss_half1"]["estimate"] >= float(gates["log_loss_gain_minimum"]),
        "logloss_bound_half0": contrasts["logloss_half0"]["adjusted_lower"] > 0.0,
        "logloss_bound_half1": contrasts["logloss_half1"]["adjusted_lower"] > 0.0,
        "brier_bound_half0": contrasts["brier_half0"]["adjusted_lower"] > 0.0,
        "brier_bound_half1": contrasts["brier_half1"]["adjusted_lower"] > 0.0,
        "spearman_overall": correlations["overall"] >= float(gates["spearman_overall_minimum"]),
        "spearman_within_network": correlations["median_within_network"] >= float(gates["spearman_within_network_minimum"]),
        "cue_A_positive": correlations["cue_A"] > 0.0,
        "cue_B_positive": correlations["cue_B"] > 0.0,
        "permutation": max(permutation["holm_p"]["logloss_half0"], permutation["holm_p"]["logloss_half1"]) <= float(gates["permutation_p_maximum"]),
    }
    result: dict[str, Any] = {
        "format": "grn-f12-tier-analysis-v1", "tier": tier,
        "networks": networks, "states": networks * states, "futures_per_state": futures,
        "event_prevalence": float(predictions["event_count"].sum() / (networks * states * futures)),
        "break_prevalence": float(predictions["break_count"].sum() / (networks * states * futures)),
        "secondary_endpoints": {
            "threshold_q025_prevalence": float(predictions["event_count_q025"].sum() / (networks * states * futures)),
            "threshold_q10_prevalence": float(predictions["event_count_q10"].sum() / (networks * states * futures)),
            "run5_F12_prevalence": float(predictions["run5_count"].sum() / (networks * states * futures)),
            "run3_F24_prevalence": float(predictions["f24_count"].sum() / (networks * states * futures)),
            "mean_coherence": float(np.mean(predictions["coherence_mean"])),
            "mean_old_anchor_separation": float(np.mean(predictions["old_anchor_separation"])),
        },
        "reliability": reliability, "correlations": correlations,
        "full_vs_history": contrasts, "full_vs_structural": structural_contrasts,
        "permutation": permutation, "checks": checks,
        "verdict": "PASS" if all(checks.values()) else "FAIL",
    }
    if "continuous_zero_shot_event" in predictions:
        result["continuous_zero_shot"] = {
            "correlations": prediction_correlations(
                predictions["continuous_zero_shot_event"], rate, predictions["cue_index"]
            ),
            "logloss": float(np.mean(binomial_log_loss(
                predictions["event_count"], futures, predictions["continuous_zero_shot_event"]
            ))),
        }
    output_dir = root / "analysis"
    write_json_atomic(output_dir / f"{tier}.json", result)
    return result


def analyze_campaign(run_dir: str | Path, protocol: dict[str, Any]) -> dict[str, Any]:
    root = Path(run_dir)
    continuous = analyze_tier(root, "continuous", protocol)
    molecular = analyze_tier(root, "molecular", protocol)
    controls_path = root / "analysis" / "controls.json"
    controls = json.loads(controls_path.read_text(encoding="utf-8")) if controls_path.exists() else {
        "pass": False, "status": "missing"
    }
    scientific = bool(protocol.get("scientific", False)) and protocol.get("profile", "full") == "full"
    if not scientific:
        verdict = "NON_SCIENTIFIC_PROFILE"
    elif continuous["verdict"] == "PASS" and molecular["verdict"] == "PASS":
        verdict = "CROSS_REALISM_CONFIRMED"
    elif continuous["verdict"] == "PASS":
        verdict = "CONTINUOUS_CONFIRMED"
    else:
        verdict = "NOT_CONFIRMED"
    result = {
        "format": "grn-f12-campaign-analysis-v1", "scientific": scientific,
        "prediction_verdict": verdict,
        "continuous_verdict": continuous["verdict"], "molecular_verdict": molecular["verdict"],
        "mechanistic_verdict": "MECHANISTIC_SUPPORT" if controls.get("pass") else "NOT_SUPPORTED",
    }
    write_json_atomic(root / "analysis" / "summary.json", result)
    lines = [
        "# Realistic GRN F12 replication", "",
        f"Prediction verdict: **{verdict}**", "",
        f"Mechanistic controls: **{result['mechanistic_verdict']}**", "",
        "| Tier | Prediction verdict | F12 prevalence | q reliability | Spearman |",
        "|---|---|---:|---:|---:|",
    ]
    for tier_result in (continuous, molecular):
        lines.append(
            f"| {tier_result['tier']} | {tier_result['verdict']} | {tier_result['event_prevalence']:.4f} | "
            f"{tier_result['reliability']['spearman_brown_q']:.4f} | {tier_result['correlations']['overall']:.4f} |"
        )
    lines.extend([
        "", "The continuous tier is primary. Molecular success can upgrade but molecular failure cannot overturn it.",
        "All failed gates remain in the tier JSON reports; no endpoint or comparator is substituted.",
    ])
    (root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    prefix = "This was a reduced engineering run, not evidence" if not scientific else "This was the registered confirmation run"
    lay_paragraphs = [
        "# Lay summary",
        "",
        f"{prefix}. We grew many independently sampled gene-regulatory networks through two different cue histories, removed the cue, and saved molecular states at five later ages. From each saved state we launched many noisy daughter lineages. We asked whether a lineage could first lose parent–daughter phenotype continuity and then regain stable continuity for three generations within a twelve-generation window—the GRN version of PH F12.",
        "",
        f"In the continuous 32-gene model, the F12 event occurred in {100 * continuous['event_prevalence']:.1f}% of futures. The graph predictor’s overall rank correlation with state-level risk was {continuous['correlations']['overall']:.2f}, and the continuous confirmatory verdict was {continuous['verdict']}. This verdict includes both frozen future halves, whole-network uncertainty, a network-level randomization test, and independent regeneration of every confirmation future.",
        "",
        f"In the explicit 16-gene mRNA/protein model, F12 occurred in {100 * molecular['event_prevalence']:.1f}% of futures. Its independently trained predictor had rank correlation {molecular['correlations']['overall']:.2f}, giving a molecular verdict of {molecular['verdict']}. This tier is an realism bridge: success strengthens generality, while failure cannot erase a continuous-tier confirmation.",
        "",
        f"The combined prediction verdict was {verdict}; the separate intervention verdict was {result['mechanistic_verdict']}. Exact state transplant, basal reset, gene-state shuffling, and erased inheritance were evaluated only as pre-registered mechanistic controls. Failed gates remain failures in the machine-readable reports, so this summary never upgrades a suggestive pattern into confirmation.",
    ]
    (root / "LAY_SUMMARY.md").write_text("\n".join(lay_paragraphs) + "\n", encoding="utf-8")
    return result
