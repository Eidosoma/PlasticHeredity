"""ECA atlas execution, adjudication, and reference comparison."""

from __future__ import annotations

import csv
import json
import math
import os
import platform
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import mean, median
from typing import Iterable, Sequence

from .config import ECAConfig
from .eca import (
    CLASS_1,
    CLASS_3_CORE,
    CLASS_4,
    DISPUTED,
    RAW_CHAMPIONS,
    canonical_rule,
    canonical_rules,
    rule_descriptors,
    simulate_lineage,
    wolfram_class,
)
from .metrics import jaccard_bits, mass_support
from .rng import stream
from .stats import average_ranks, quantile, spearman


def environment_record() -> dict[str, object]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "implementation": "standard-library integer bitboards",
    }


def _mean_vector(total: list[float], count: int) -> tuple[float, ...]:
    if count <= 0:
        return tuple(0.0 for _ in total)
    return tuple(value / count for value in total)


@dataclass(frozen=True)
class RuleResult:
    rule: int
    wolfram_class: int
    disputed: bool
    strict: float
    break_by_8: float
    median_gen_sweeps: float
    mean_survival: float
    form_supports: tuple[int, ...]
    descriptors: dict[str, float]
    n_futures: int

    @property
    def library(self) -> frozenset[int]:
        return frozenset(self.form_supports)


def evaluate_rule(rule: int, config: ECAConfig, *, experiment: str = "atlas") -> RuleResult:
    strict_count = 0
    break_count = 0
    survival: list[int] = []
    sweep_counts: list[int] = []
    seed_totals = [[0.0] * 16 for _ in range(config.n_seeds)]
    seed_counts = [0] * config.n_seeds

    for seed_index in range(config.n_seeds):
        for future_index in range(config.futures_per_seed):
            lineage = simulate_lineage(
                rule,
                seed_index,
                future_index,
                config,
                experiment=experiment,
            )
            strict_count += lineage.strict
            break_count += lineage.break_by_8
            survival.append(lineage.survived)
            sweep_counts.extend(lineage.sweeps)
            for composition in lineage.post_break_daughters:
                for index, value in enumerate(composition):
                    seed_totals[seed_index][index] += value
                seed_counts[seed_index] += 1

    forms = tuple(
        mass_support(_mean_vector(seed_totals[index], seed_counts[index]), config.form_mass_quantile)
        for index in range(config.n_seeds)
        if seed_counts[index] > 0
    )
    n_futures = config.n_seeds * config.futures_per_seed
    return RuleResult(
        rule=rule,
        wolfram_class=wolfram_class(rule),
        disputed=rule in DISPUTED,
        strict=strict_count / n_futures,
        break_by_8=break_count / n_futures,
        median_gen_sweeps=float(median(sweep_counts)) if sweep_counts else float(config.max_sweeps),
        mean_survival=mean(survival) if survival else 0.0,
        form_supports=forms,
        descriptors=rule_descriptors(rule),
        n_futures=n_futures,
    )


def _library_jaccard(left: frozenset[int], right: frozenset[int], *, empty_empty: float = 1.0) -> float:
    union = left | right
    if not union:
        return empty_empty
    return len(left & right) / len(union)


def canonical_hypercube_edges() -> tuple[tuple[int, int], ...]:
    edges: set[tuple[int, int]] = set()
    for rule in range(256):
        source = canonical_rule(rule)
        for bit in range(8):
            target = canonical_rule(rule ^ (1 << bit))
            if source != target:
                edges.add(tuple(sorted((source, target))))
    return tuple(sorted(edges))


def _descriptor_distance(left: RuleResult, right: RuleResult) -> float:
    keys = ("lambda", "asymmetry", "quiescent_defect", "sensitivity")
    return math.sqrt(sum((left.descriptors[key] - right.descriptors[key]) ** 2 for key in keys))


def adjudicate_atlas(results: Sequence[RuleResult], namespace: str) -> dict[str, object]:
    by_rule = {result.rule: result for result in results}
    clean_12 = [
        result.break_by_8
        for result in results
        if result.wolfram_class in (1, 2) and not result.disputed
    ]
    class3_min = min(by_rule[rule].break_by_8 for rule in CLASS_3_CORE)
    clean_median = median(clean_12)

    descending = sorted(results, key=lambda item: (-item.strict, item.rule))
    rank_values = average_ranks([-item.strict for item in results])
    ranks = {result.rule: rank for result, rank in zip(results, rank_values)}

    edges = canonical_hypercube_edges()
    edge_jaccard = mean(_library_jaccard(by_rule[a].library, by_rule[b].library) for a, b in edges)
    all_pairs = [(a, b) for index, a in enumerate(by_rule) for b in list(by_rule)[index + 1 :]]
    rng = stream(namespace, "atlas-random-pairs")
    random_pairs = rng.sample(all_pairs, min(len(edges), len(all_pairs)))
    random_jaccard = mean(_library_jaccard(by_rule[a].library, by_rule[b].library) for a, b in random_pairs)
    smoothness = edge_jaccard / random_jaccard if random_jaccard else float("inf")

    census = Counter(support for result in results for support in result.form_supports if support)
    n_top = max(1, math.ceil(0.1 * len(census))) if census else 0
    top_share = sum(count for _, count in census.most_common(n_top)) / sum(census.values()) if census else 0.0

    pair_divergence: list[float] = []
    hamming: list[float] = []
    descriptor: list[float] = []
    for a, b in all_pairs:
        pair_divergence.append(1.0 - _library_jaccard(by_rule[a].library, by_rule[b].library))
        hamming.append(float((a ^ b).bit_count()))
        descriptor.append(_descriptor_distance(by_rule[a], by_rule[b]))
    rho_hamming = spearman(hamming, pair_divergence)
    rho_descriptor = spearman(descriptor, pair_divergence)

    class_medians = {
        str(cls): median(result.break_by_8 for result in results if result.wolfram_class == cls)
        for cls in (1, 2, 3, 4)
    }
    return {
        "gate_class3_separation": class3_min > clean_median,
        "class3_min_break_by_8": class3_min,
        "clean_class12_median_break_by_8": clean_median,
        "class_break_medians": class_medians,
        "gate_rule110_top_decile": ranks[110] <= 9,
        "rule110_strict_rank": ranks[110],
        "rule110_strict": by_rule[110].strict,
        "raw_class4_strict": {str(rule): by_rule[rule].strict for rule in sorted(CLASS_4)},
        "champion_strict": {str(rule): by_rule[rule].strict for rule in RAW_CHAMPIONS},
        "gate_smoothness": smoothness >= 2.0,
        "smoothness_ratio": smoothness,
        "edge_mean_jaccard": edge_jaccard,
        "random_mean_jaccard": random_jaccard,
        "gate_heavy_tail": top_share >= 0.35,
        "heavy_tail_share": top_share,
        "n_forms": len(census),
        "gate_metric": rho_descriptor > rho_hamming,
        "metric_rho_descriptors": rho_descriptor,
        "metric_rho_hamming": rho_hamming,
    }


def _evaluate_rule_task(arguments: tuple[int, ECAConfig, str]) -> RuleResult:
    rule, config, experiment = arguments
    return evaluate_rule(rule, config, experiment=experiment)


def run_atlas(
    config: ECAConfig,
    output: Path,
    *,
    rules: Sequence[int] | None = None,
    workers: int = 1,
    experiment: str = "atlas",
) -> dict[str, object]:
    started = time.time()
    selected = tuple(rules) if rules is not None else canonical_rules()
    if any(canonical_rule(rule) != rule for rule in selected):
        raise ValueError("atlas rules must be canonical representatives")
    results: list[RuleResult] = []
    if workers <= 1:
        results = [evaluate_rule(rule, config, experiment=experiment) for rule in selected]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_evaluate_rule_task, (rule, config, experiment)): rule
                for rule in selected
            }
            for future in as_completed(futures):
                results.append(future.result())
    results.sort(key=lambda item: item.rule)

    registry = {
        support: index
        for index, support in enumerate(sorted({support for result in results for support in result.library if support}))
    }
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "eca_rules.csv"
    fieldnames = [
        "rule",
        "wolfram_class",
        "disputed",
        "strict",
        "break_by_8",
        "median_gen_sweeps",
        "mean_survival",
        "library_size",
        "library",
        "support_masks",
        "d_lambda",
        "d_asymmetry",
        "d_quiescent",
        "d_sensitivity",
        "n_futures",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            library = sorted(result.library)
            writer.writerow(
                {
                    "rule": result.rule,
                    "wolfram_class": result.wolfram_class,
                    "disputed": result.disputed,
                    "strict": result.strict,
                    "break_by_8": result.break_by_8,
                    "median_gen_sweeps": result.median_gen_sweeps,
                    "mean_survival": result.mean_survival,
                    "library_size": len(library),
                    "library": "|".join(str(registry[value]) for value in library),
                    "support_masks": "|".join(str(value) for value in library),
                    "d_lambda": result.descriptors["lambda"],
                    "d_asymmetry": result.descriptors["asymmetry"],
                    "d_quiescent": result.descriptors["quiescent_defect"],
                    "d_sensitivity": result.descriptors["sensitivity"],
                    "n_futures": result.n_futures,
                }
            )

    gates = adjudicate_atlas(results, config.seed_namespace) if set(selected) == set(canonical_rules()) else {}
    summary = {
        "experiment": experiment,
        "elapsed_seconds": time.time() - started,
        "environment": environment_record(),
        "config": config.to_dict(),
        "n_rules": len(results),
        "n_futures": sum(result.n_futures for result in results),
        "gates": gates,
        "cleanroom_semantics": {
            "launch_anchor": config.semantics.launch_anchor,
            "launch_preparation": config.semantics.launch_preparation,
            "launch_burnin_sweeps": config.launch_burnin_sweeps,
            "seed_mode": config.semantics.seed_mode,
            "process_noise": config.semantics.process_noise,
            "activity_count": config.semantics.activity_count,
            "monochrome_death": config.semantics.monochrome_death,
            "observed_daughter": config.semantics.observed_daughter,
            "form_definition": "per-seed mean of live post-first-break daughters, top-50%-mass support",
        },
    }
    (output / "atlas_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output / "COMPLETE").write_text("complete\n")
    return summary


def _read_numeric_rows(path: Path) -> dict[int, dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        int(row["rule"]): {
            "strict": float(row["strict"]),
            "break_by_8": float(row["break_by_8"]),
            "wolfram_class": float(row["wolfram_class"]),
        }
        for row in rows
    }


def compare_atlas(ours_path: Path, reference_path: Path) -> dict[str, object]:
    ours = _read_numeric_rows(ours_path)
    reference = _read_numeric_rows(reference_path)
    common = sorted(set(ours) & set(reference))
    if not common:
        raise ValueError("the two atlas files have no common rules")
    strict_ours = [ours[rule]["strict"] for rule in common]
    strict_ref = [reference[rule]["strict"] for rule in common]
    break_ours = [ours[rule]["break_by_8"] for rule in common]
    break_ref = [reference[rule]["break_by_8"] for rule in common]
    class4_raw_failure = all(ours.get(rule, {}).get("strict", 1.0) < 0.005 for rule in CLASS_4 if rule in ours)
    core = [rule for rule in CLASS_3_CORE if rule in ours]
    clean = [
        rule
        for rule in common
        if int(ours[rule]["wolfram_class"]) in (1, 2) and rule not in DISPUTED
    ]
    class3_separation = min(ours[rule]["break_by_8"] for rule in core) > median(
        ours[rule]["break_by_8"] for rule in clean
    ) if core and clean else None
    return {
        "n_common_rules": len(common),
        "strict_spearman": spearman(strict_ours, strict_ref),
        "break_by_8_spearman": spearman(break_ours, break_ref),
        "strict_mean_absolute_error": mean(abs(a - b) for a, b in zip(strict_ours, strict_ref)),
        "break_by_8_mean_absolute_error": mean(abs(a - b) for a, b in zip(break_ours, break_ref)),
        "directional_checks": {
            "class3_separation": class3_separation,
            "raw_class4_strict_below_0.005": class4_raw_failure,
            "rule110_strict_below_0.005": ours.get(110, {}).get("strict", 1.0) < 0.005,
            "at_least_three_raw_champions_nonzero": sum(
                ours.get(rule, {}).get("strict", 0.0) >= 0.005 for rule in RAW_CHAMPIONS
            ) >= 3,
        },
        "selected_rules": {
            str(rule): {"ours": ours.get(rule), "reference": reference.get(rule)}
            for rule in sorted(set(RAW_CHAMPIONS) | CLASS_4 | {30, 90, 150})
            if rule in common
        },
    }


def _phase_task(arguments: tuple[int, float, ECAConfig]) -> tuple[float, RuleResult]:
    rule, eta, base = arguments
    config = replace(base, flip_noise=eta, copy_error=1.5 * eta)
    return eta, evaluate_rule(rule, config, experiment=f"phase-eta-{eta:g}")


def _unimodal_with_tolerance(values: Sequence[float], tolerance: float) -> bool:
    """Return true if some peak makes both flanks monotone within tolerance."""

    for peak in range(len(values)):
        left_ok = all(values[index + 1] + tolerance >= values[index] for index in range(peak))
        right_ok = all(values[index + 1] <= values[index] + tolerance for index in range(peak, len(values) - 1))
        if left_ok and right_ok:
            return True
    return False


def run_phase(
    config: ECAConfig,
    output: Path,
    *,
    etas: Sequence[float] = (0.0025, 0.005, 0.01, 0.02, 0.04),
    workers: int = 1,
) -> dict[str, object]:
    started = time.time()
    tasks = [(rule, eta, config) for eta in etas for rule in canonical_rules()]
    evaluated: list[tuple[float, RuleResult]] = []
    if workers <= 1:
        evaluated = [_phase_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_phase_task, task) for task in tasks]
            for future in as_completed(futures):
                evaluated.append(future.result())
    evaluated.sort(key=lambda item: (item[0], item[1].rule))

    output.mkdir(parents=True, exist_ok=True)
    with (output / "phase.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("break_by_8", "eta", "median_gen_sweeps", "rule", "strict", "in_band", "wolfram_class"),
        )
        writer.writeheader()
        for eta, result in evaluated:
            writer.writerow(
                {
                    "break_by_8": result.break_by_8,
                    "eta": eta,
                    "median_gen_sweeps": result.median_gen_sweeps,
                    "rule": result.rule,
                    "strict": result.strict,
                    "in_band": 0.005 <= result.strict <= 0.5,
                    "wolfram_class": result.wolfram_class,
                }
            )

    by_eta = {
        eta: {result.rule: result for point, result in evaluated if point == eta}
        for eta in etas
    }
    per_eta = []
    robustness = True
    for eta in etas:
        rows = by_eta[eta]
        band = [result for result in rows.values() if 0.005 <= result.strict <= 0.5]
        class2_fraction = (
            sum(result.wolfram_class == 2 for result in band) / len(band) if band else 0.0
        )
        core_in_band = sum(rule in CLASS_3_CORE for rule in (result.rule for result in band))
        cell_ok = bool(band) and class2_fraction > 0.5 and core_in_band == 0
        robustness &= cell_ok
        per_eta.append(
            {
                "eta": eta,
                "n_band": len(band),
                "band_rules": [result.rule for result in band],
                "band_class2_fraction": class2_fraction,
                "core3_in_band": core_in_band,
                "cell_pass": cell_ok,
            }
        )

    ever_capable = sorted(
        rule
        for rule in canonical_rules()
        if any(0.005 <= by_eta[eta][rule].strict <= 0.5 for eta in etas)
    )
    profiles = {
        rule: [by_eta[eta][rule].strict for eta in etas]
        for rule in ever_capable
    }
    unimodal = {rule: _unimodal_with_tolerance(values, 0.01) for rule, values in profiles.items()}
    fraction = sum(unimodal.values()) / len(unimodal) if unimodal else 0.0
    peak_eta = {
        str(rule): etas[max(range(len(etas)), key=lambda index: profiles[rule][index])]
        for rule in ever_capable
    }
    awakened = sorted(
        rule
        for rule in CLASS_1
        if any(0.005 <= by_eta[eta][rule].strict <= 0.5 for eta in etas)
    )
    gates = {
        "gate_regime_robustness": robustness,
        "gate_ridge_law": fraction >= 2 / 3,
        "gate_class1_awakening": bool(awakened),
        "awakened_class1": awakened,
        "n_ever_capable": len(ever_capable),
        "unimodal_fraction": fraction,
        "peak_eta_by_rule": peak_eta,
        "per_eta": per_eta,
        "rule110_by_eta": [
            {
                "eta": eta,
                "strict": by_eta[eta][110].strict,
                "break_by_8": by_eta[eta][110].break_by_8,
            }
            for eta in etas
        ],
    }
    summary = {
        "experiment": "eca_phase",
        "elapsed_seconds": time.time() - started,
        "environment": environment_record(),
        "config": config.to_dict(),
        "n_points": len(evaluated),
        "gates": gates,
    }
    (output / "phase_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output / "COMPLETE").write_text("complete\n")
    return summary
