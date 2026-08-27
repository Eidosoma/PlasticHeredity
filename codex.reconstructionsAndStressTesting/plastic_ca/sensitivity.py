"""Resumable clean-room reconciliation across underspecified ECA semantics."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import mean, median
from typing import Iterable, Sequence

from .config import ECAConfig, ECASemantics, config_for_profile
from .eca import CLASS_3_CORE, CLASS_4, RAW_CHAMPIONS, canonical_rules, wolfram_class
from .evolution import run_evolution
from .experiments import RuleResult, compare_atlas, evaluate_rule, run_atlas, run_phase
from .life import life_config_for_profile, named_patterns, run_life, simulate_life_lineage
from .metrics import cosine, mass_support, normalize, strict_coherent_event
from .particle import run_particle
from .rng import derive_seed
from .stats import spearman


REFERENCE_CHAMPIONS = frozenset(RAW_CHAMPIONS)
FALSE_CHAMPIONS = frozenset({13, 28, 156, 172})
PREPARATIONS = ("sweeps_0", "sweeps_1", "sweeps_4", "sweeps_16", "sweeps_64", "noiseless_generation")


@dataclass(frozen=True)
class SensitivitySetting:
    launch_anchor: str
    launch_preparation: str
    seed_mode: str
    process_noise: str
    activity_count: str
    monochrome_death: str
    observed_daughter: str

    @property
    def setting_id(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()[:16]

    @property
    def contract_plausible(self) -> bool:
        return self.seed_mode != "density_stratified" and self.process_noise != "terminal_once"

    def build_config(self, n_seeds: int, futures: int, namespace: str) -> ECAConfig:
        if self.launch_preparation == "noiseless_generation":
            preparation = "noiseless_generation"
            burnin = 0
        else:
            preparation = "sweeps"
            burnin = int(self.launch_preparation.removeprefix("sweeps_"))
        semantics = ECASemantics(
            launch_anchor=self.launch_anchor,
            launch_preparation=preparation,
            seed_mode=self.seed_mode,
            process_noise=self.process_noise,
            activity_count=self.activity_count,
            monochrome_death=self.monochrome_death,
            observed_daughter=self.observed_daughter,
        )
        return replace(
            config_for_profile("smoke"),
            n_seeds=n_seeds,
            futures_per_seed=futures,
            seed_namespace=namespace,
            launch_burnin_sweeps=burnin,
            semantics=semantics,
        )


def enumerate_settings() -> tuple[SensitivitySetting, ...]:
    settings: list[SensitivitySetting] = []
    for anchor in ("prepared_seed", "first_completed_generation"):
        for preparation in PREPARATIONS:
            for seed_mode in ("expected_half_hash", "exact_half", "density_stratified"):
                for process_noise in ("pre_rule_each_sweep", "post_rule_each_sweep", "terminal_once"):
                    activity_values = ("realized",) if process_noise == "terminal_once" else ("realized", "deterministic")
                    for activity_count in activity_values:
                        for death in (
                            "terminal_only",
                            "realized_immediate",
                            "realized_after_minimum",
                            "deterministic_immediate",
                        ):
                            for observed in ("pre_copy_terminal", "post_copy_offspring"):
                                settings.append(
                                    SensitivitySetting(
                                        anchor,
                                        preparation,
                                        seed_mode,
                                        process_noise,
                                        activity_count,
                                        death,
                                        observed,
                                    )
                                )
    return tuple(settings)


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _status(root: Path, state: str, stage: str, **extra: object) -> None:
    _atomic_json(
        root / "STATUS.json",
        {"state": state, "stage": stage, "pid": os.getpid(), "updated_unix": time.time(), **extra},
    )
    print(f"[{state}] {stage}", flush=True)


def _reference_rows(path: Path) -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rule = int(row["rule"])
            rows[rule] = {
                "strict": float(row["strict"]),
                "break_by_8": float(row["break_by_8"]),
                "median_gen_sweeps": float(row["median_gen_sweeps"]),
                "mean_survival": float(row["mean_survival"]),
                "wolfram_class": float(row["wolfram_class"]),
            }
    return rows


def stratified_rule_split(reference: dict[int, dict[str, float]]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    development: list[int] = []
    holdout: list[int] = []
    for cls in (1, 2, 3, 4):
        rules = sorted(
            (rule for rule, row in reference.items() if int(row["wolfram_class"]) == cls),
            key=lambda rule: (reference[rule]["strict"], derive_seed("sensitivity-split-v1", rule)),
        )
        for index, rule in enumerate(rules):
            (development if index % 2 == 0 else holdout).append(rule)
    while len(development) > len(holdout):
        movable = sorted(
            (rule for rule in development if int(reference[rule]["wolfram_class"]) == 2),
            key=lambda rule: derive_seed("sensitivity-split-balance-v1", rule),
        )
        rule = movable[0]
        development.remove(rule)
        holdout.append(rule)
    while len(holdout) > len(development):
        movable = sorted(
            (rule for rule in holdout if int(reference[rule]["wolfram_class"]) == 2),
            key=lambda rule: derive_seed("sensitivity-split-balance-v1", rule),
        )
        rule = movable[0]
        holdout.remove(rule)
        development.append(rule)
    return tuple(sorted(development)), tuple(sorted(holdout))


def _rule_result_dict(result: RuleResult) -> dict[str, object]:
    return {
        "rule": result.rule,
        "wolfram_class": result.wolfram_class,
        "disputed": result.disputed,
        "strict": result.strict,
        "break_by_8": result.break_by_8,
        "median_gen_sweeps": result.median_gen_sweeps,
        "mean_survival": result.mean_survival,
        "form_supports": list(result.form_supports),
        "descriptors": result.descriptors,
        "n_futures": result.n_futures,
    }


def _setting_task(payload: tuple[dict[str, str], tuple[int, ...], int, int, str, str]) -> dict[str, object]:
    raw_setting, rules, n_seeds, futures, namespace, experiment = payload
    setting = SensitivitySetting(**raw_setting)
    config = setting.build_config(n_seeds, futures, namespace)
    started = time.time()
    results = [evaluate_rule(rule, config, experiment=experiment) for rule in rules]
    return {
        "setting": asdict(setting),
        "setting_id": setting.setting_id,
        "contract_plausible": setting.contract_plausible,
        "n_seeds": n_seeds,
        "futures_per_seed": futures,
        "seed_namespace": namespace,
        "elapsed_seconds": time.time() - started,
        "results": [_rule_result_dict(result) for result in results],
    }


def _run_cells(
    root: Path,
    stage: str,
    settings: Sequence[SensitivitySetting],
    rules: tuple[int, ...],
    n_seeds: int,
    futures: int,
    namespace: str,
    workers: int,
) -> None:
    cells = root / stage / "cells"
    cells.mkdir(parents=True, exist_ok=True)
    todo = [setting for setting in settings if not (cells / f"{setting.setting_id}.json").exists()]
    total = len(settings)
    complete_before = total - len(todo)
    _status(root, "running", stage, complete=complete_before, total=total)
    if not todo:
        _status(root, "completed_stage", stage, complete=total, total=total)
        return
    payloads = [
        (asdict(setting), rules, n_seeds, futures, namespace, f"sensitivity:{stage}")
        for setting in todo
    ]
    completed = complete_before
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures_by_id = {
            pool.submit(_setting_task, payload): setting.setting_id
            for setting, payload in zip(todo, payloads)
        }
        for future in as_completed(futures_by_id):
            setting_id = futures_by_id[future]
            payload = future.result()
            _atomic_json(cells / f"{setting_id}.json", payload)
            completed += 1
            if completed == total or completed % max(1, min(25, total // 10 or 1)) == 0:
                _status(root, "running", stage, complete=completed, total=total)
    _status(root, "completed_stage", stage, complete=total, total=total)


def _load_cell(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def score_cell(cell: dict[str, object], reference: dict[int, dict[str, float]]) -> dict[str, object]:
    results = list(cell["results"])
    results.sort(key=lambda row: int(row["rule"]))
    rules = [int(row["rule"]) for row in results]
    strict = [float(row["strict"]) for row in results]
    breaks = [float(row["break_by_8"]) for row in results]
    survival = [float(row["mean_survival"]) for row in results]
    sweeps = [float(row["median_gen_sweeps"]) for row in results]
    ref_strict = [reference[rule]["strict"] for rule in rules]
    ref_break = [reference[rule]["break_by_8"] for rule in rules]
    ref_survival = [reference[rule]["mean_survival"] for rule in rules]
    ref_sweeps = [reference[rule]["median_gen_sweeps"] for rule in rules]

    def mae(left: Sequence[float], right: Sequence[float]) -> float:
        return mean(abs(a - b) for a, b in zip(left, right))

    strict_rho = spearman(strict, ref_strict)
    break_rho = spearman(breaks, ref_break)
    survival_rho = spearman(survival, ref_survival)
    sweep_rho = spearman(sweeps, ref_sweeps)
    strict_mae = mae(strict, ref_strict)
    break_mae = mae(breaks, ref_break)
    survival_mae = mae(survival, ref_survival)
    sweep_log_mae = mae([math.log1p(value) for value in sweeps], [math.log1p(value) for value in ref_sweeps])
    ours_by_rule = {int(row["rule"]): row for row in results}
    available_core = sorted(CLASS_3_CORE & set(rules))
    clean = [
        float(row["break_by_8"])
        for row in results
        if int(row["wolfram_class"]) in (1, 2) and not bool(row["disputed"])
    ]
    class3_separation = bool(available_core and clean) and min(
        float(ours_by_rule[rule]["break_by_8"]) for rule in available_core
    ) > median(clean)
    class4_values = [float(ours_by_rule[rule]["strict"]) for rule in CLASS_4 if rule in ours_by_rule]
    class4_max = max(class4_values, default=0.0)
    ours_top10 = {
        int(row["rule"])
        for row in sorted(results, key=lambda row: (-float(row["strict"]), int(row["rule"])))[:10]
    }
    reference_top = {
        rule
        for rule in sorted(rules, key=lambda rule: (-reference[rule]["strict"], rule))[: min(5, len(rules))]
    }
    top_recall = len(ours_top10 & reference_top) / len(reference_top) if reference_top else 0.0
    loss = (
        0.25 * min(1.0, max(0.0, (1.0 - strict_rho) / 2.0))
        + 0.20 * min(1.0, max(0.0, (1.0 - break_rho) / 2.0))
        + 0.15 * min(1.0, strict_mae / 0.25)
        + 0.15 * min(1.0, break_mae / 0.50)
        + 0.10 * min(1.0, survival_mae / 32.0)
        + 0.10 * min(1.0, sweep_log_mae / math.log(129.0))
        + 0.05 * (1.0 - top_recall)
    )
    class_medians = {
        str(cls): median(float(row["break_by_8"]) for row in results if int(row["wolfram_class"]) == cls)
        for cls in (1, 2, 3, 4)
        if any(int(row["wolfram_class"]) == cls for row in results)
    }
    return {
        "setting_id": str(cell["setting_id"]),
        "setting": cell["setting"],
        "contract_plausible": bool(cell["contract_plausible"]),
        "n_rules": len(rules),
        "strict_rho": strict_rho,
        "break_rho": break_rho,
        "survival_rho": survival_rho,
        "sweep_rho": sweep_rho,
        "strict_mae": strict_mae,
        "break_mae": break_mae,
        "survival_mae": survival_mae,
        "sweep_log_mae": sweep_log_mae,
        "class3_separation": class3_separation,
        "class4_max_strict": class4_max,
        "class_break_medians": class_medians,
        "top10_reference_recall": top_recall,
        "reconciliation_loss": loss,
        "top10_rules": sorted(ours_top10),
    }


def _score_stage(root: Path, stage: str, reference: dict[int, dict[str, float]]) -> list[dict[str, object]]:
    metrics = [score_cell(_load_cell(path), reference) for path in sorted((root / stage / "cells").glob("*.json"))]
    metrics.sort(key=lambda row: (float(row["reconciliation_loss"]), str(row["setting_id"])))
    _atomic_json(root / stage / "metrics.json", metrics)
    return metrics


def _pareto(metrics: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    dimensions = ("strict_rho", "break_rho", "strict_mae", "break_mae", "survival_mae", "sweep_log_mae")

    def vector(row: dict[str, object]) -> tuple[float, ...]:
        return (
            -float(row[dimensions[0]]),
            -float(row[dimensions[1]]),
            *(float(row[name]) for name in dimensions[2:]),
        )

    front: list[dict[str, object]] = []
    for candidate in metrics:
        cvec = vector(candidate)
        dominated = False
        for other in metrics:
            if other is candidate:
                continue
            ovec = vector(other)
            if all(a <= b for a, b in zip(ovec, cvec)) and any(a < b for a, b in zip(ovec, cvec)):
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    return sorted(front, key=lambda row: (float(row["reconciliation_loss"]), str(row["setting_id"])))


def _setting_from_metric(metric: dict[str, object]) -> SensitivitySetting:
    return SensitivitySetting(**dict(metric["setting"]))


def _select_semifinalists(metrics: Sequence[dict[str, object]], count: int) -> list[SensitivitySetting]:
    eligible = [
        row for row in metrics
        if bool(row["contract_plausible"])
        and bool(row["class3_separation"])
        and float(row["class4_max_strict"]) <= 0.05
    ]
    if not eligible:
        eligible = [row for row in metrics if bool(row["contract_plausible"])] or list(metrics)
    front = _pareto(eligible)
    chosen = list(front[:count])
    chosen_ids = {str(row["setting_id"]) for row in chosen}
    for row in eligible:
        if len(chosen) >= count:
            break
        if str(row["setting_id"]) not in chosen_ids:
            chosen.append(row)
            chosen_ids.add(str(row["setting_id"]))
    return [_setting_from_metric(row) for row in chosen]


def _holdout_pass(metric: dict[str, object]) -> bool:
    return (
        bool(metric["contract_plausible"])
        and float(metric["strict_rho"]) >= 0.75
        and float(metric["break_rho"]) >= 0.85
        and float(metric["strict_mae"]) <= 0.05
        and float(metric["break_mae"]) <= 0.10
        and float(metric["survival_rho"]) >= 0.75
        and float(metric["sweep_rho"]) >= 0.80
        and float(metric["class4_max_strict"]) <= 0.005
        and bool(metric["class3_separation"])
    )


def _full_pass(metric: dict[str, object], cell: dict[str, object], reference: dict[int, dict[str, float]]) -> tuple[bool, dict[str, object]]:
    results = {int(row["rule"]): row for row in cell["results"]}
    top10 = {
        rule for rule, _ in sorted(results.items(), key=lambda item: (-float(item[1]["strict"]), item[0]))[:10]
    }
    champion_recall = len(top10 & REFERENCE_CHAMPIONS)
    false_ok = all(float(results[rule]["strict"]) <= 0.05 for rule in FALSE_CHAMPIONS)
    reference_medians = {
        cls: median(row["break_by_8"] for row in reference.values() if int(row["wolfram_class"]) == cls)
        for cls in (1, 2, 3, 4)
    }
    observed_medians = {int(key): float(value) for key, value in dict(metric["class_break_medians"]).items()}
    medians_ok = all(abs(observed_medians[cls] - reference_medians[cls]) <= 0.10 for cls in (1, 2, 3, 4))
    details = {
        "champions_in_top10": champion_recall,
        "top10": sorted(top10),
        "false_champions_ok": false_ok,
        "class_medians_ok": medians_ok,
        "reference_class_medians": reference_medians,
    }
    passed = (
        bool(metric["contract_plausible"])
        and float(metric["strict_rho"]) >= 0.80
        and float(metric["break_rho"]) >= 0.90
        and float(metric["strict_mae"]) <= 0.04
        and float(metric["break_mae"]) <= 0.08
        and champion_recall >= 4
        and false_ok
        and medians_ok
        and float(metric["class4_max_strict"]) <= 0.005
        and bool(metric["class3_separation"])
    )
    return passed, details


def _mean_vectors(vectors: Sequence[Sequence[float]]) -> tuple[float, ...]:
    if not vectors:
        return (0.0,) * 15
    return tuple(mean(vector[index] for vector in vectors) for index in range(15))


def run_life_form_sensitivity(output: Path, *, profile: str = "reference") -> dict[str, object]:
    config = life_config_for_profile(profile)
    patterns = named_patterns(config.width, config.height)
    named_lineages = {
        name: [
            simulate_life_lineage(patterns[name], future, config, f"life-form-sensitivity:{name}")
            for future in range(config.named_futures)
        ]
        for name in ("glider", "blinker", "toad")
    }
    modes = ("first_1", "first_2", "first_4", "first_8", "all_pooled", "equal_future", "post_break")
    rows: list[dict[str, object]] = []
    for normalized in (False, True):
        for mode in modes:
            spectra: dict[str, tuple[float, ...]] = {}
            supports: dict[str, int] = {}
            for name, lineages in named_lineages.items():
                vectors: list[tuple[float, ...]] = []
                if mode == "equal_future":
                    for lineage in lineages:
                        selected = list(lineage.compositions)
                        if normalized:
                            selected = [normalize(value) for value in selected]
                        if selected:
                            vectors.append(_mean_vectors(selected))
                else:
                    for lineage in lineages:
                        if mode.startswith("first_"):
                            selected = list(lineage.compositions[: int(mode.split("_")[1])])
                        elif mode == "post_break":
                            event = strict_coherent_event(lineage.compositions, config.thresholds)
                            selected = list(lineage.compositions[event.first_break + 1 :]) if event.first_break is not None else []
                        else:
                            selected = list(lineage.compositions)
                        vectors.extend(normalize(value) if normalized else tuple(value) for value in selected)
                spectrum = _mean_vectors(vectors)
                spectra[name] = spectrum
                supports[name] = mass_support(spectrum, config.form_mass_quantile)
            between = {
                f"{left}|{right}": cosine(spectra[left], spectra[right])
                for left, right in (("glider", "blinker"), ("glider", "toad"), ("blinker", "toad"))
            }
            rows.append(
                {
                    "mode": mode,
                    "weighting": "normalized_each_generation" if normalized else "mass_weighted",
                    "supports": supports,
                    "between_object_cosines": between,
                    "gate_distinct_forms": len(set(supports.values())) == 3 and all(value < 0.999 for value in between.values()),
                    "contract_plausible": mode in {"first_4", "all_pooled", "equal_future"},
                }
            )
    summary = {
        "experiment": "life_form_sensitivity",
        "config": config.to_dict(),
        "rows": rows,
        "gate_any_plausible_distinct": any(row["contract_plausible"] and row["gate_distinct_forms"] for row in rows),
    }
    _atomic_json(output / "life_form_sensitivity.json", summary)
    return summary


def _run_downstream(
    root: Path,
    setting: SensitivitySetting,
    reference_root: Path,
    workers: int,
    profile: str = "reference",
) -> dict[str, object]:
    downstream = root / "downstream"
    n_seeds, futures = (4, 4) if profile == "smoke" else (16, 128)
    config = setting.build_config(n_seeds, futures, "plastic-ca-sensitivity-downstream-v1")
    atlas_dir = downstream / "atlas"
    if (atlas_dir / "COMPLETE").exists() and (atlas_dir / "reference_comparison.json").exists():
        atlas = json.loads((atlas_dir / "atlas_summary.json").read_text(encoding="utf-8"))
        comparison = json.loads((atlas_dir / "reference_comparison.json").read_text(encoding="utf-8"))
    else:
        _status(root, "running", "downstream_atlas")
        atlas = run_atlas(config, atlas_dir, workers=workers, experiment="sensitivity:downstream-atlas")
        comparison = compare_atlas(atlas_dir / "eca_rules.csv", reference_root / "eca_atlas/results/full/eca_rules.csv")
        _atomic_json(atlas_dir / "reference_comparison.json", comparison)

    phase_dir = downstream / "phase"
    if (phase_dir / "COMPLETE").exists():
        phase = json.loads((phase_dir / "phase_summary.json").read_text(encoding="utf-8"))
    else:
        _status(root, "running", "downstream_phase")
        phase = run_phase(config, phase_dir, workers=workers)

    particle_dir = downstream / "particle"
    if (particle_dir / "COMPLETE").exists():
        particle = json.loads((particle_dir / "particle_gates.json").read_text(encoding="utf-8"))
    else:
        _status(root, "running", "downstream_particle")
        particle = run_particle(config, particle_dir, workers=workers)

    life_dir = downstream / "life"
    if (life_dir / "COMPLETE").exists():
        life = json.loads((life_dir / "observer.json").read_text(encoding="utf-8"))
    else:
        _status(root, "running", "downstream_life")
        life = run_life(life_config_for_profile(profile), life_dir)
    if (life_dir / "life_form_sensitivity.json").exists():
        life_forms = json.loads((life_dir / "life_form_sensitivity.json").read_text(encoding="utf-8"))
    else:
        life_forms = run_life_form_sensitivity(life_dir, profile=profile)

    evolution_dir = downstream / "evolution"
    if (evolution_dir / "COMPLETE").exists():
        evolution = json.loads((evolution_dir / "evolution_summary.json").read_text(encoding="utf-8"))
    else:
        _status(root, "running", "downstream_evolution")
        evolution = run_evolution(atlas_dir / "eca_rules.csv", evolution_dir)

    reference_particle = json.loads(
        (reference_root / "particle_observer/results/full/particle_gates.json").read_text(encoding="utf-8")
    )
    coverage = particle["gates"]["dict_coverage_by_class"]
    reference_coverage = reference_particle["gates"]["dict_coverage_by_class"]
    coverage_ok = all(abs(float(coverage[key]) - float(reference_coverage[key])) <= 0.02 for key in reference_coverage)
    phase_gates = phase["gates"]
    particle_gates = particle["gates"]
    life_gates = life["gates"]
    evolution_gates = evolution["evolution"]
    gates = {
        "phase_parity": (
            bool(phase_gates["gate_regime_robustness"])
            and bool(phase_gates["gate_ridge_law"])
            and not bool(phase_gates["gate_class1_awakening"])
            and max(float(row["strict"]) for row in phase_gates["rule110_by_eta"]) <= 0.005
        ),
        "particle_redemption_and_coverage": bool(particle_gates["gate_redemption_110"]) and coverage_ok,
        "life_fidelity_persistence_forms": (
            bool(life_gates["gate_fidelity_absolute"])
            and bool(life_gates["gate_persistence"])
            and bool(life_forms["gate_any_plausible_distinct"])
        ),
        "evolution_parity": (
            bool(evolution_gates["gate_selection_finds_boundary"])
            and not bool(evolution_gates["gate_sticky_walk"])
        ),
    }
    summary = {
        "setting": asdict(setting),
        "atlas_gates": atlas["gates"],
        "reference_comparison": comparison,
        "gates": gates,
        "all_downstream_gates": all(gates.values()),
    }
    _atomic_json(downstream / "adjudication.json", summary)
    (downstream / "COMPLETE").write_text("complete\n", encoding="utf-8")
    _status(root, "completed_stage", "downstream")
    return summary


def _fable_questions(adjudication: dict[str, object], best_metrics: Sequence[dict[str, object]]) -> str:
    best_lines = "\n".join(
        f"- `{row['setting_id']}`: strict rho {float(row['strict_rho']):.3f}, break rho "
        f"{float(row['break_rho']):.3f}, strict MAE {float(row['strict_mae']):.3f}, "
        f"break MAE {float(row['break_mae']):.3f}; settings `{json.dumps(row['setting'], sort_keys=True)}`."
        for row in best_metrics[:5]
    )
    return f"""# Questions for Fable — ECA Clean-Room Reconciliation

The exhaustive clean-room sensitivity round did not meet the frozen strong-match
bar. No sibling source, tests, scripts, or seeds were inspected.

## Best surviving settings

{best_lines}

## Minimal executable-contract questions

1. Are the 16 launch rows observable parents, or is composition zero the first
   completed activity-gated generation?
2. Is there deterministic launch preparation or burn-in? If so, how many sweeps
   or what stopping rule is used?
3. How are the 16 frozen launch rows constructed (exact density, PRNG/hash
   family, and whether rows are shared across rules)? Providing the rows or
   their hex digests would resolve the largest remaining ensemble ambiguity.
4. Within one sweep, is process noise applied before or after the ECA rule?
5. Does the 256-change activity counter include noise flips, deterministic rule
   changes only, or the realized stored-row difference?
6. Is monochrome death checked during a generation, only when its activity
   clock stops, before process noise, or after process noise?
7. Is the observed daughter the terminal row before copy error or the offspring
   after copy error?
8. For the Life round-5 ensemble form, which generations/futures are pooled,
   and is each generation normalized before the 0.75 mass support is taken?

## Most useful code-free golden traces

For rules 8, 13, 35, 110, and 172 on one frozen launch row, please provide the
launch row, terminal row, sweep count, death flag, and observed final4 vector
for the first two generations at `(eta, epsilon)=(0,0)` and at
`(0.01,0.015)`. A supplied sequence of process/copy masks would let us replay
the stochastic trace without seeing implementation code.

## Automated adjudication snapshot

```json
{json.dumps(adjudication, indent=2, sort_keys=True)}
```
"""


def _report(adjudication: dict[str, object], best_metrics: Sequence[dict[str, object]]) -> str:
    verdict = "STRONG MATCH" if adjudication["overall_success"] else "ESCALATE CONTRACT QUESTIONS"
    lines = [
        "# Sensitivity Round 1 — Automated Report",
        "",
        f"Verdict: **{verdict}**",
        "",
        f"Raw strong-match pass: `{adjudication['raw_strong_match']}`.",
        f"Downstream all-gates pass: `{adjudication['downstream']['all_downstream_gates']}`.",
        "",
        "## Leading confirmation settings",
        "",
    ]
    for row in best_metrics[:5]:
        lines.append(
            f"- `{row['setting_id']}` — strict rho `{float(row['strict_rho']):.3f}`, "
            f"break rho `{float(row['break_rho']):.3f}`, strict MAE `{float(row['strict_mae']):.3f}`, "
            f"break MAE `{float(row['break_mae']):.3f}`."
        )
    lines.extend(["", "Full machine-readable adjudication: `adjudication.json`.", ""])
    return "\n".join(lines)


def run_sensitivity(
    output: Path,
    reference_root: Path,
    *,
    workers: int = 16,
    resume: bool = False,
    design_name: str = "overnight",
    max_hours: float = 12.0,
) -> dict[str, object] | None:
    started = time.time()
    reference_path = reference_root / "eca_atlas/results/full/eca_rules.csv"
    reference = _reference_rows(reference_path)
    development, holdout = stratified_rule_split(reference)
    all_settings = list(enumerate_settings())
    if design_name == "tiny":
        all_settings = [
            SensitivitySetting("prepared_seed", "sweeps_1", "expected_half_hash", "post_rule_each_sweep", "realized", "terminal_only", "pre_copy_terminal"),
            SensitivitySetting("first_completed_generation", "sweeps_0", "exact_half", "post_rule_each_sweep", "realized", "realized_immediate", "pre_copy_terminal"),
            SensitivitySetting("first_completed_generation", "sweeps_4", "expected_half_hash", "pre_rule_each_sweep", "deterministic", "deterministic_immediate", "post_copy_offspring"),
        ]
        development = development[:8]
        holdout = holdout[:8]
        samples = {"screen": (2, 2), "semifinal": (2, 3), "holdout": (2, 4), "confirmation": (2, 4)}
        semifinal_count, finalist_count, confirmation_count = 2, 2, 1
    else:
        samples = {"screen": (4, 8), "semifinal": (16, 32), "holdout": (16, 128), "confirmation": (16, 128)}
        semifinal_count, finalist_count, confirmation_count = 24, 4, 2
    design = {
        "design_name": design_name,
        "cleanroom_boundary": "No sibling source, tests, scripts, executables, or seeds may be read.",
        "development_rules": development,
        "holdout_rules": holdout,
        "n_settings": len(all_settings),
        "settings": [{**asdict(setting), "setting_id": setting.setting_id, "contract_plausible": setting.contract_plausible} for setting in all_settings],
        "samples": samples,
        "max_hours": max_hours,
    }
    design_digest = hashlib.sha256(json.dumps(design, sort_keys=True).encode()).hexdigest()
    design["design_digest"] = design_digest
    if (output / "design.json").exists():
        existing = json.loads((output / "design.json").read_text(encoding="utf-8"))
        if existing.get("design_digest") != design_digest:
            raise ValueError("existing sensitivity design differs; choose a new output directory")
        if not resume:
            raise FileExistsError("sensitivity output already exists; pass --resume")
        if (output / "COMPLETE").exists() and (output / "adjudication.json").exists():
            adjudication = json.loads((output / "adjudication.json").read_text(encoding="utf-8"))
            _status(output, "complete", "sensitivity_round", resumed=True)
            return adjudication
    else:
        output.mkdir(parents=True, exist_ok=True)
        _atomic_json(output / "design.json", design)

    try:
        _run_cells(output, "screen", all_settings, development, *samples["screen"], "plastic-ca-sensitivity-dev-v1", workers)
        screen_metrics = _score_stage(output, "screen", reference)
        semifinalists = _select_semifinalists(screen_metrics, semifinal_count)
        _atomic_json(output / "semifinalists.json", [{**asdict(value), "setting_id": value.setting_id} for value in semifinalists])
        if time.time() - started > max_hours * 3600:
            _status(output, "paused", "time_limit_after_screen")
            return None

        _run_cells(output, "semifinal", semifinalists, development, *samples["semifinal"], "plastic-ca-sensitivity-dev-v1", workers)
        semifinal_metrics = _score_stage(output, "semifinal", reference)
        finalists = _select_semifinalists(semifinal_metrics, finalist_count)
        _atomic_json(output / "finalists.json", [{**asdict(value), "setting_id": value.setting_id} for value in finalists])
        if time.time() - started > max_hours * 3600:
            _status(output, "paused", "time_limit_after_semifinal")
            return None

        _run_cells(output, "holdout", finalists, holdout, *samples["holdout"], "plastic-ca-sensitivity-holdout-v1", workers)
        holdout_metrics = _score_stage(output, "holdout", reference)
        accepted = [row for row in holdout_metrics if _holdout_pass(row)]
        accepted_ids = {str(row["setting_id"]) for row in accepted}
        # Confirm every holdout-accepted setting first, then fill any remaining
        # slots with the leading settings.  This preserves the frozen "best two
        # accepted-or-leading" rule even when exactly one finalist clears the
        # holdout bar.
        ranked = accepted + [
            row for row in holdout_metrics
            if str(row["setting_id"]) not in accepted_ids
        ]
        confirmations = [_setting_from_metric(row) for row in ranked[:confirmation_count]]
        _atomic_json(output / "confirmation_settings.json", [{**asdict(value), "setting_id": value.setting_id} for value in confirmations])

        _run_cells(output, "confirmation", confirmations, tuple(canonical_rules()), *samples["confirmation"], "plastic-ca-sensitivity-confirm-v1", workers)
        confirmation_metrics = _score_stage(output, "confirmation", reference)
        full_details: dict[str, object] = {}
        raw_passes: list[str] = []
        for metric in confirmation_metrics:
            cell = _load_cell(output / "confirmation" / "cells" / f"{metric['setting_id']}.json")
            passed, details = _full_pass(metric, cell, reference)
            full_details[str(metric["setting_id"])] = {"passed": passed, **details}
            if passed:
                raw_passes.append(str(metric["setting_id"]))
        # A downstream success must belong to the same semantic setting that
        # passed the raw all-rule bar.  If nothing passes, propagate the
        # leading setting so the failure diagnosis is still maximally useful.
        best_metric = next(
            (metric for metric in confirmation_metrics if str(metric["setting_id"]) in raw_passes),
            confirmation_metrics[0],
        )
        best_setting = _setting_from_metric(best_metric)
        downstream = _run_downstream(
            output,
            best_setting,
            reference_root,
            workers,
            profile="smoke" if design_name == "tiny" else "reference",
        )
        overall_success = bool(raw_passes) and bool(downstream["all_downstream_gates"])
        adjudication = {
            "design_digest": design_digest,
            "elapsed_seconds": time.time() - started,
            "best_setting_id": best_setting.setting_id,
            "best_setting": asdict(best_setting),
            "holdout_passes": [str(row["setting_id"]) for row in accepted],
            "raw_strong_match": bool(raw_passes),
            "raw_pass_settings": raw_passes,
            "confirmation_details": full_details,
            "downstream": downstream,
            "overall_success": overall_success,
        }
        _atomic_json(output / "adjudication.json", adjudication)
        (output / "SENSITIVITY_REPORT.md").write_text(_report(adjudication, confirmation_metrics), encoding="utf-8")
        if not overall_success:
            questions = _fable_questions(adjudication, confirmation_metrics)
            (output / "FABLE_QUESTIONS.md").write_text(questions, encoding="utf-8")
            if design_name != "tiny":
                (Path.cwd() / "FABLE_QUESTIONS.md").write_text(questions, encoding="utf-8")
        (output / "COMPLETE").write_text("complete\n", encoding="utf-8")
        _status(output, "complete", "sensitivity_round", overall_success=overall_success)
        return adjudication
    except BaseException as error:
        _status(output, "failed", "sensitivity_round", error=repr(error))
        raise
