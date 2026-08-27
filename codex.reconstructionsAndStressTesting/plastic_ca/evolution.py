"""Evolutionary search on the eight-bit ECA rule hypercube."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from statistics import mean, median
import time

from .eca import CLASS_3_CORE, canonical_rule, wolfram_class
from .rng import stream
from .stats import quantile


@dataclass(frozen=True)
class EvolutionConfig:
    population: int = 24
    generations: int = 40
    repeats: int = 32
    tournament_k: int = 3
    measurement_budget: int = 256
    seed_namespace: str = "plastic-ca-cleanroom-evolution-v1"


def _read_atlas(path: Path) -> tuple[dict[int, float], dict[int, frozenset[int]]]:
    fitness: dict[int, float] = {}
    libraries: dict[int, frozenset[int]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rule = int(row["rule"])
            fitness[rule] = float(row["strict"])
            # Keep the exact 16-bit supports.  The former ``mask % 4096``
            # shortcut silently collided distinct forms and biased Jaccard.
            libraries[rule] = frozenset(
                int(value) for value in row.get("support_masks", "").split("|") if value
            )
    return fitness, libraries


def _library_jaccard(left: frozenset[int], right: frozenset[int], *, empty_empty: float) -> float:
    if not left and not right:
        return empty_empty
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _observed_fitness(rng, probability: float, budget: int) -> float:
    if hasattr(rng, "binomialvariate"):
        return rng.binomialvariate(budget, probability) / budget
    return sum(rng.random() < probability for _ in range(budget)) / budget


def _true_fitness(rule: int, fitness: dict[int, float]) -> float:
    return fitness[canonical_rule(rule)]


def _run_arm(
    arm: str,
    repeat: int,
    config: EvolutionConfig,
    fitness: dict[int, float],
    shuffled: dict[int, float],
) -> tuple[float, float, int, list[tuple[int, int]]]:
    rng = stream(config.seed_namespace, arm, repeat)
    population = [rng.randrange(256) for _ in range(config.population)]
    visits = 0
    total_visits = 0
    transitions: list[tuple[int, int]] = []
    for _ in range(config.generations):
        visits += sum(canonical_rule(rule) in CLASS_3_CORE for rule in population)
        total_visits += len(population)
        next_population: list[int] = []
        for index in range(config.population):
            if arm == "random_walk":
                parent = population[index]
            elif arm == "drift":
                parent = rng.choice(population)
            else:
                contenders = rng.sample(population, config.tournament_k)
                landscape = shuffled if arm == "fitness_shuffled" else fitness
                parent = max(
                    contenders,
                    key=lambda rule: _observed_fitness(
                        rng,
                        landscape[canonical_rule(rule)],
                        config.measurement_budget,
                    ),
                )
            child = parent ^ (1 << rng.randrange(8))
            transitions.append((parent, child))
            next_population.append(child)
        population = next_population
    final_mean = mean(_true_fitness(rule, fitness) for rule in population)
    final_class = int(median(wolfram_class(rule) for rule in population))
    return final_mean, visits / total_visits, final_class, transitions


def run_evolution(atlas_path: Path, output: Path, config: EvolutionConfig = EvolutionConfig()) -> dict[str, object]:
    started = time.time()
    fitness, libraries = _read_atlas(atlas_path)
    rules = sorted(fitness)
    shuffle_rng = stream(config.seed_namespace, "shuffled-landscape")
    shuffled_values = [fitness[rule] for rule in rules]
    shuffle_rng.shuffle(shuffled_values)
    shuffled = dict(zip(rules, shuffled_values))
    arms = ("selection", "drift", "random_walk", "fitness_shuffled")
    records: dict[str, list[tuple[float, float, int, list[tuple[int, int]]]]] = {arm: [] for arm in arms}
    for arm in arms:
        for repeat in range(config.repeats):
            records[arm].append(_run_arm(arm, repeat, config, fitness, shuffled))

    summaries = {}
    for arm in arms:
        values = [record[0] for record in records[arm]]
        summaries[arm] = {
            "mean": mean(values),
            "ci95": [quantile(values, 0.025), quantile(values, 0.975)],
            "median_class": median(record[2] for record in records[arm]),
        }
    selection_low = summaries["selection"]["ci95"][0]
    gate_selection = (
        all(selection_low > summaries[arm]["mean"] for arm in arms if arm != "selection")
        and summaries["selection"]["median_class"] == 2
    )

    selected_edges = [edge for record in records["selection"] for edge in record[3]]
    evolved_jaccard = mean(
        _library_jaccard(
            libraries[canonical_rule(parent)],
            libraries[canonical_rule(child)],
            empty_empty=1.0,
        )
        for parent, child in selected_edges
    )
    null_rng = stream(config.seed_namespace, "stickiness-null")
    random_pairs = [(null_rng.randrange(256), null_rng.randrange(256)) for _ in range(len(selected_edges))]
    null_jaccard = mean(
        _library_jaccard(
            libraries[canonical_rule(left)],
            libraries[canonical_rule(right)],
            empty_empty=1.0,
        )
        for left, right in random_pairs
    )
    nonempty_edges = [
        edge
        for edge in selected_edges
        if libraries[canonical_rule(edge[0])] or libraries[canonical_rule(edge[1])]
    ]
    nonempty_pairs = [
        pair
        for pair in random_pairs
        if libraries[canonical_rule(pair[0])] or libraries[canonical_rule(pair[1])]
    ]
    evolved_nonempty = mean(
        _library_jaccard(libraries[canonical_rule(a)], libraries[canonical_rule(b)], empty_empty=0.0)
        for a, b in nonempty_edges
    ) if nonempty_edges else 0.0
    null_nonempty = mean(
        _library_jaccard(libraries[canonical_rule(a)], libraries[canonical_rule(b)], empty_empty=0.0)
        for a, b in nonempty_pairs
    ) if nonempty_pairs else 0.0

    # The retained post-hoc lesson concerns the atlas geometry itself: every
    # one-bit edge, excluding empty/empty degeneracy, versus matched random
    # pairs.  Keep it separate from the failed predeclared selection-path gate.
    full_libraries = {rule: libraries[canonical_rule(rule)] for rule in range(256)}
    all_edges = [
        (rule, rule ^ (1 << bit))
        for rule in range(256)
        for bit in range(8)
        if rule < (rule ^ (1 << bit))
    ]
    nonempty_all_edges = [
        pair for pair in all_edges if full_libraries[pair[0]] or full_libraries[pair[1]]
    ]
    global_null_rng = stream(config.seed_namespace, "global-stickiness-null")
    global_random_pairs: list[tuple[int, int]] = []
    while len(global_random_pairs) < len(all_edges):
        pair = tuple(global_null_rng.sample(range(256), 2))
        if full_libraries[pair[0]] or full_libraries[pair[1]]:
            global_random_pairs.append(pair)
    global_edge_mean = mean(
        _library_jaccard(full_libraries[a], full_libraries[b], empty_empty=0.0)
        for a, b in nonempty_all_edges
    )
    global_null_mean = mean(
        _library_jaccard(full_libraries[a], full_libraries[b], empty_empty=0.0)
        for a, b in global_random_pairs
    )

    summary = {
        "experiment": "rulial_evolution",
        "elapsed_seconds": time.time() - started,
        "config": asdict(config),
        "evolution": {
            "arm_summaries": summaries,
            "gate_selection_finds_boundary": gate_selection,
            "class3_core_visit_fraction": {
                "selection": mean(record[1] for record in records["selection"]),
                "random_walk": mean(record[1] for record in records["random_walk"]),
            },
            "gate_sticky_walk": evolved_jaccard > null_jaccard,
            "stickiness": {"evolved": evolved_jaccard, "null": null_jaccard},
            "stickiness_nonempty_reanalysis": {
                "evolved": evolved_nonempty,
                "null": null_nonempty,
                "ratio": evolved_nonempty / null_nonempty if null_nonempty else None,
            },
            "stickiness_all_edges_nonempty_reanalysis": {
                "n_edges": len(nonempty_all_edges),
                "edge_mean": global_edge_mean,
                "random_mean": global_null_mean,
                "ratio": global_edge_mean / global_null_mean if global_null_mean else None,
                "direction_recovered": global_edge_mean > global_null_mean,
            },
        },
        "boundary": "GPS omitted: the clean-room atlas has no independent fresh-stream truth table",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "evolution_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (output / "COMPLETE").write_text("complete\n")
    return summary
