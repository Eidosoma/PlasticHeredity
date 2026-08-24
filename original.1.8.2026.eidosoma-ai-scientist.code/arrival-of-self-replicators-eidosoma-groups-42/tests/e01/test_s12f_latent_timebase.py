from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from e01_latent_timebase.core import (
    N_MAX,
    ExposureDefinition,
    SimulationDefinition,
    _trim_new_entrants,
    clock_length,
    derive_seed,
    exposure_for_rates,
    rates,
    simulate_trajectory,
    trajectory_replay_equal,
)
from e01_latent_timebase.inference import (
    candidate_groups,
    initial_particles,
    particle_summary_and_distance,
    phase1_clock_audit,
)

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs/e01/s12f_latent_timebase_preregistration.yaml"
TARGET = REPO / "configs/e01/s12f/paper_timebase_targets.yaml"
ROOT = "12" * 32


def test_preregistration_freezes_scope_schedule_and_blocked_s13() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["researchStepId"] == "E01-S12F-LATENT-TIMEBASE-INFERENCE-v1.0.0"
    assert config["s13Status"] == "BLOCKED_PENDING_S12F_HUMAN_REVIEW"
    assert [row["particlesEvaluated"] for row in config["phase2"]["abcSmc"]["rounds"]] == [256, 128, 64]
    assert config["phase2"]["developmentMatricesPerParticle"] == 8
    assert config["phase3"]["matricesPerCandidate"] == 32
    assert config["phase3"]["maximumCandidates"] == 3
    assert len(config["benchmark"]["configurations"]) == 16
    assert not config["interpretationBoundary"]["labelsPermitted"]
    assert not config["interpretationBoundary"]["causalEmergencePermitted"]
    assert not config["interpretationBoundary"]["interventionsPermitted"]


def test_paper_targets_are_interval_bearing_and_table_ratio_is_soft() -> None:
    target = yaml.safe_load(TARGET.read_text(encoding="utf-8"))
    assert target["figure2"]["panelB"]["finalPlottedTimePoint"] == 800.0
    assert target["figure2"]["panelC"]["finalPlottedTimePoint"] == 800.0
    assert target["figure2"]["panelD"]["finalPlottedTimePoint"] == 1000.0
    assert target["figure2"]["panelA"]["visibleAggregateTraceTerminalInterval"] == [1090.0, 1120.0]
    assert np.isclose(target["table1"]["descriptiveRatio"], 716.0 / 0.88)
    assert target["table1"]["evidentiaryWeight"] == "SOFT_SECONDARY_ONLY"
    uncertainty = json.loads(
        (REPO / "configs/e01/s12f/figure_digitization_uncertainty.json").read_text()
    )
    assert len(uncertainty["methods"]) == 2
    assert uncertainty["aggregate"]["status"] == "AGREED_WITH_INTERVAL_CENSORING"


def test_seed_derivation_is_domain_separated_and_exact() -> None:
    shared = derive_seed(ROOT, "test", "catalytic_matrix", 0)
    event = derive_seed(ROOT, "test", "poisson_update", 0, "CONFIG-A")
    trim = derive_seed(ROOT, "test", "overshoot_trim", 0, "CONFIG-A")
    other = derive_seed(ROOT, "test", "poisson_update", 0, "CONFIG-B")
    assert shared == derive_seed(ROOT, "test", "catalytic_matrix", 0)
    assert len({seed.seed_material_sha256 for seed in (shared, event, trim, other)}) == 4


def test_rate_equation_and_exposure_fixtures() -> None:
    state = np.zeros(100, dtype=np.int64)
    state[:40] = 1
    beta = np.zeros((100, 100), dtype=np.float64)
    joins, losses = rates(state, beta)
    assert np.array_equal(joins, np.full(100, 0.004))
    assert np.array_equal(losses[:40], np.full(40, 0.0001))
    assert np.array_equal(losses[40:], np.zeros(60))
    fixed = ExposureDefinition("FIXED_COMMON_EXPOSURE", h=0.5)
    assert exposure_for_rates(fixed, joins, losses) == 0.5
    adaptive = ExposureDefinition(
        "ADAPTIVE_GROSS_EVENT_EXPOSURE", c=1.0, h_max=2.0
    )
    assert np.isclose(
        exposure_for_rates(adaptive, joins, losses),
        min(2.0, 1.0 / (joins.sum() + losses.sum())),
    )


def test_trim_removes_only_new_entrants_and_hits_exact_mass() -> None:
    old = np.zeros(100, dtype=np.int64)
    old[:79] = 1
    joins = np.zeros(100, dtype=np.int64)
    joins[79:89] = 1
    proposed = old + joins
    rng = np.random.Generator(np.random.PCG64DXSM(12345))
    trimmed, removed = _trim_new_entrants(proposed, joins, rng)
    assert removed == 9
    assert trimmed.sum() == N_MAX
    assert np.array_equal(trimmed[:79], old[:79])
    assert np.all(trimmed >= 0)


def test_complete_trajectory_replay_and_clock_cardinalities() -> None:
    definition = SimulationDefinition(
        "RANDOM_NONEMPTY",
        "TRIM_NEW_ENTRANTS_TO_NMAX",
        ExposureDefinition("FIXED_COMMON_EXPOSURE", h=1.25),
    )
    first, first_seeds = simulate_trajectory(
        phase="test",
        root_hex=ROOT,
        matrix_index=3,
        definition=definition,
        stream_identity="CLOCK-FIXTURE",
    )
    replay, replay_seeds = simulate_trajectory(
        phase="test",
        root_hex=ROOT,
        matrix_index=3,
        definition=definition,
        stream_identity="CLOCK-FIXTURE",
    )
    assert trajectory_replay_equal(first, replay)
    assert first_seeds == replay_seeds
    assert first.completed_fissions == 100
    assert clock_length(first, "C0_BATCH_UPDATES_ONLY") == first.total_batch_updates
    assert clock_length(first, "C1_SELECTED_DAUGHTER_RETAINED") == first.total_batch_updates + 100
    assert clock_length(first, "C2_EXPLICIT_PRE_AND_POST_FISSION") == first.total_batch_updates + 200
    assert len(first.observations) == 1 + first.total_batch_updates + 100
    assert all(row.pre_fission_mass == 80 for row in first.generations)


def test_phase1_clock_audit_is_read_only_and_rejects_synthetic_c2() -> None:
    rows, summary = phase1_clock_audit()
    assert rows.shape[0] == 3 * 24 * 5
    assert set(rows["clockId"]) == {
        "C0_BATCH_UPDATES_ONLY", "C1_SELECTED_DAUGHTER_RETAINED",
        "C2_EXPLICIT_PRE_AND_POST_FISSION", "C3_NONZERO_REACTION_CHANNEL",
        "C4_GROSS_MOLECULAR_EVENT",
    }
    c2 = summary[summary["clockId"] == "C2_EXPLICIT_PRE_AND_POST_FISSION"]
    assert not c2["clockOnlyGatePassed"].any()
    assert c2["requiresSyntheticDuplicate"].all()
    diagnostics = summary[summary["clockId"].isin(["C3_NONZERO_REACTION_CHANNEL", "C4_GROSS_MOLECULAR_EVENT"])]
    assert (diagnostics["status"] == "NOT_RECOVERABLE_FROM_S12E_CACHE").all()


def test_abc_prior_is_stratified_and_distance_uses_no_downstream_fields() -> None:
    particles = initial_particles("FIXED_COMMON_EXPOSURE", ROOT, 256)
    assert len(particles) == 256
    assert len({particle.discrete_group for particle in particles}) == 18
    counts = pd.Series([particle.discrete_group for particle in particles]).value_counts()
    assert counts.max() - counts.min() <= 1
    particle = particles[0]
    lengths = np.linspace(700, 1100, 8)
    frame = pd.DataFrame(
        {
            "clockC0": lengths,
            "clockC1": lengths + 100,
            "clockC2": lengths + 200,
            "medianPreFissionMass": np.full(8, 80.0),
            "medianPostFissionMass": np.full(8, 40.0),
            "q95Overshoot": np.full(8, 10.0),
            "completedFissions": np.full(8, 100),
            "maxstepsTerminations": np.zeros(8),
        }
    )
    result = particle_summary_and_distance(particle, frame)
    assert set(result).isdisjoint({"label", "emergence", "phi_r", "intervention"})
    assert np.isfinite(result["distance"])


def test_candidate_group_selection_is_deterministic_and_capped() -> None:
    particles = initial_particles("FIXED_COMMON_EXPOSURE", ROOT, 64)
    rows = []
    for index, particle in enumerate(particles):
        row = {
            "particle_id": particle.particle_id,
            "discreteGroup": particle.discrete_group,
            "family": particle.family,
            "daughter_rule": particle.daughter_rule,
            "overshoot_rule": particle.overshoot_rule,
            "clock_id": particle.clock_id,
            "h": particle.h,
            "c": particle.c,
            "h_max": particle.h_max,
            "distance": 0.1 + index / 1000.0,
            "complexity": 1.0,
            "developmentAcceptanceEnvelopePassed": particle.clock_id != "C2_EXPLICIT_PRE_AND_POST_FISSION",
        }
        rows.append(row)
    frame, selected = candidate_groups(
        particles, pd.DataFrame(rows), np.full(64, 1.0 / 64), maximum=3
    )
    assert len(selected) <= 3
    assert frame["selectedForConfirmation"].sum() <= 3
    assert all(particle.clock_id != "C2_EXPLICIT_PRE_AND_POST_FISSION" for particle in selected)
