"""Figure/ground particle observer for elementary cellular automata."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
import json
import platform
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Sequence

from .config import ECAConfig
from .eca import CLASS_3_CORE, CLASS_4, RAW_CHAMPIONS, eca_step, prepare_launch, simulate_lineage, wolfram_class
from .metrics import Vector, cosine, cyclic_kmer_spectrum
from .rng import fixed_density_bits


PARTICLE_GATE_RULES = tuple(sorted(CLASS_4 | CLASS_3_CORE | set(RAW_CHAMPIONS) | {0, 8, 26, 73, 154}))
PARTICLE_DEV_RULES = (4, 232, 50, 62, 178, 108)


def spacetime_codes(history: tuple[int, int, int], width: int) -> tuple[int, ...]:
    """Return one row-major 3x3 spacetime code per spatial cell."""

    codes: list[int] = []
    for position in range(width):
        code = 0
        bit_index = 0
        for row in history:
            for delta in (-1, 0, 1):
                source = (position + delta) % width
                code |= ((row >> source) & 1) << bit_index
                bit_index += 1
        codes.append(code)
    return tuple(codes)


@dataclass(frozen=True)
class DomainDictionary:
    codes: frozenset[int]
    coverage: float
    n_distinct_codes: int
    counts: tuple[tuple[int, int], ...]


def build_domain_dictionary(
    rule: int,
    config: ECAConfig,
    *,
    n_seeds: int = 4,
    burnin: int = 64,
    collect: int = 16,
    coverage_target: float = 0.9,
    cap: int = 64,
) -> DomainDictionary:
    counts: Counter[int] = Counter()
    width = config.width
    density_grid = (0.2, 0.4, 0.6, 0.8)
    for seed_index in range(n_seeds):
        density = density_grid[seed_index % len(density_grid)]
        row = fixed_density_bits(
            config.seed_namespace + ":particle-domain",
            seed_index,
            width,
            density,
        )
        history = [row, row, row]
        for _ in range(burnin):
            row = eca_step(row, rule, width)
            history = [history[-2], history[-1], row]
        for _ in range(collect):
            row = eca_step(row, rule, width)
            history = [history[-2], history[-1], row]
            counts.update(spacetime_codes(tuple(history), width))
    total = sum(counts.values())
    selected: list[int] = []
    selected_mass = 0
    for code, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        if len(selected) >= cap:
            break
        selected.append(code)
        selected_mass += count
        if total and selected_mass / total >= coverage_target:
            break
    return DomainDictionary(
        frozenset(selected),
        selected_mass / total if total else 0.0,
        len(counts),
        tuple(sorted(counts.items())),
    )


def figure_mask(history: tuple[int, int, int], dictionary: DomainDictionary, width: int) -> int:
    result = 0
    for position, code in enumerate(spacetime_codes(history, width)):
        if code not in dictionary.codes:
            result |= 1 << position
    return result


def particle_observer(dictionary: DomainDictionary, width: int):
    def observe(row: int, history: tuple[int, int, int]) -> Vector | None:
        mask = figure_mask(history, dictionary, width)
        if mask == 0:
            return None
        return cyclic_kmer_spectrum(mask, width, 4)

    return observe


def evaluate_particle_rule(rule: int, config: ECAConfig, *, experiment: str = "particle") -> dict[str, object]:
    dictionary = build_domain_dictionary(rule, config)
    observe = particle_observer(dictionary, config.width)
    particle_config = replace(config, observer="particle4")
    strict = 0
    breaks = 0
    survival: list[int] = []
    similarities: list[float] = []
    n = config.n_seeds * config.futures_per_seed
    for seed_index in range(config.n_seeds):
        for future_index in range(config.futures_per_seed):
            lineage = simulate_lineage(
                rule,
                seed_index,
                future_index,
                particle_config,
                observe,
                experiment=experiment,
                observe_launch=False,
            )
            strict += lineage.strict
            breaks += lineage.break_by_8
            survival.append(lineage.survived)
            similarities.extend(cosine(a, b) for a, b in zip(lineage.compositions, lineage.compositions[1:]))
    return {
        "rule": rule,
        "wolfram_class": wolfram_class(rule),
        "strict": strict / n,
        "break_by_8": breaks / n,
        "mean_survival": mean(survival) if survival else 0.0,
        "h_above": sum(value > config.thresholds.inherit for value in similarities) / len(similarities) if similarities else 0.0,
        "dict_coverage": dictionary.coverage,
        "dict_dict_size": len(dictionary.codes),
        "dict_n_distinct_codes": dictionary.n_distinct_codes,
        "n_futures": n,
    }


def _particle_task(arguments: tuple[int, ECAConfig]) -> dict[str, object]:
    rule, config = arguments
    return evaluate_particle_rule(rule, config)


def run_particle(
    config: ECAConfig,
    output: Path,
    *,
    rules: Sequence[int] = PARTICLE_GATE_RULES,
    workers: int = 1,
) -> dict[str, object]:
    started = time.time()
    rows: list[dict[str, object]] = []
    tasks = [(rule, config) for rule in rules]
    if workers <= 1:
        rows = [_particle_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_particle_task, task) for task in tasks]
            for future in as_completed(futures):
                rows.append(future.result())
    rows.sort(key=lambda row: int(row["rule"]))
    by_rule = {int(row["rule"]): row for row in rows}
    class4 = {str(rule): float(by_rule[rule]["strict"]) for rule in sorted(CLASS_4) if rule in by_rule}
    core = {str(rule): float(by_rule[rule]["strict"]) for rule in sorted(CLASS_3_CORE) if rule in by_rule}
    champions = {str(rule): float(by_rule[rule]["strict"]) for rule in RAW_CHAMPIONS if rule in by_rule}
    gates = {
        "gate_redemption_110": 110 in by_rule and 0.005 <= float(by_rule[110]["strict"]) <= 0.5,
        "gate_chaos_stays_chaos": bool(core) and all(value < 0.005 for value in core.values()),
        "gate_champions_stable": len(champions) == len(RAW_CHAMPIONS) and all(value >= 0.005 for value in champions.values()),
        "class4_strict": class4,
        "champions_strict": champions,
        "core3_strict": core,
        "core3_strict_max": max(core.values()) if core else None,
        "dict_coverage_by_class": {
            str(cls): mean(float(row["dict_coverage"]) for row in rows if int(row["wolfram_class"]) == cls)
            for cls in (1, 2, 3, 4)
            if any(int(row["wolfram_class"]) == cls for row in rows)
        },
    }
    summary = {
        "experiment": "particle_observer",
        "elapsed_seconds": time.time() - started,
        "environment": {"python": sys.version, "platform": platform.platform()},
        "config": replace(config, observer="particle4").to_dict(),
        "gates": gates,
        "rows": rows,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "particle_gates.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output / "COMPLETE").write_text("complete\n")
    return summary
