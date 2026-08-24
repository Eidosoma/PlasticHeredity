import json
from dataclasses import asdict

import numpy as np
import pandas as pd

from aor_replication.probe import (
    CONTROL_TARGETS,
    ProbeConfig,
    crossover_genomes,
    default_genome,
    gene_specs,
    mutate_genome,
    resolved_configs,
    run_probe,
    score_metrics,
)


def tiny_probe() -> ProbeConfig:
    return ProbeConfig(
        population_size=4,
        ga_generations=2,
        calibration_runs=2,
        holdout_runs=2,
        calibration_seed=100,
        holdout_seed=200,
        ga_seed=300,
        workers=1,
        elite_count=1,
        tournament_size=2,
        n_types=10,
        initial_size=5,
        max_size=10,
        gard_generations=6,
        max_steps_per_generation=40,
        max_trace_steps=500,
        beta_log_sigma=1.0,
    )


def test_exact_paper_targets_have_zero_score() -> None:
    metrics = {name: target.value for name, target in CONTROL_TARGETS.items()}
    score, components = score_metrics(metrics)
    assert score == 0
    assert set(components) == set(CONTROL_TARGETS)
    assert all(value == 0 for value in components.values())


def test_genetic_operators_remain_in_bounded_valid_space() -> None:
    probe = tiny_probe()
    rng = np.random.default_rng(71)
    left = default_genome()
    right = {spec.name: spec.sample(rng) for spec in gene_specs()}
    for _ in range(50):
        child = crossover_genomes(left, right, probe, rng)
        child = mutate_genome(child, probe, rng)
        assert set(child) == {spec.name for spec in gene_specs()}
        gard, causal, replicator, intervention = resolved_configs(child, probe)
        for config in (gard, causal, replicator, intervention):
            config.validate()
        left, right = right, child


def test_probe_writes_artifacts_and_completed_run_resumes(tmp_path) -> None:
    probe = tiny_probe()
    output = tmp_path / "probe"
    first = run_probe(probe, output)

    assert first["calibration"]["status"] == "ok"
    assert first["holdout"]["status"] == "ok"
    assert first["holdout"]["figure5_validation"]["status"] == "skipped"
    assert (output / "best_candidate.json").is_file()
    assert (output / "generation_history.csv").is_file()
    assert (output / "top_candidates.csv").is_file()
    assert (output / "convergence.png").stat().st_size > 0
    assert (output / "SUMMARY.md").is_file()
    assert (output / "figure5_validation_targets.json").is_file()

    history = pd.read_csv(output / "generation_history.csv")
    assert len(history) == probe.population_size * probe.ga_generations
    assert history.generation.tolist() == [0] * 4 + [1] * 4
    with (output / "checkpoint.json").open(encoding="utf-8") as stream:
        assert json.load(stream)["complete"] is True
    with (output / "runtime.json").open(encoding="utf-8") as stream:
        assert json.load(stream)["status"] == "complete"

    second = run_probe(probe, output)
    assert second == first
    with (output / "probe_config.json").open(encoding="utf-8") as stream:
        assert json.load(stream) == asdict(probe)
