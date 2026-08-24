from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import jsonschema
import numpy as np
import yaml

from e01_latent_timebase.core import (
    ExposureDefinition,
    GenerationSummary,
    SeedIdentity,
    SimulationDefinition,
    StateObservation,
    TimebaseTrajectory,
)
from e01_replay_repair.audit import RecordingGenerator, canonical_sha256
from e01_replay_repair.comparator import (
    COMPARATOR_VERSION,
    compare_seed_tuples,
    compare_trajectories,
)

REPO = Path(__file__).resolve().parents[2]


def trajectory(*, updates: int = 0, maximum: float = math.nan, state_value: int = 1) -> TimebaseTrajectory:
    definition = SimulationDefinition(
        "RANDOM_NONEMPTY",
        "RETAIN_OVERSHOOT",
        ExposureDefinition("FIXED_COMMON_EXPOSURE", h=0.25),
    )
    state = (state_value,) + (0,) * 99
    generation = GenerationSummary(
        generation_one_based=1,
        terminal_status="n_max_reached",
        update_count=updates,
        nonzero_reaction_type_count=0,
        gross_sampled_event_count=0,
        pre_fission_mass=80,
        post_fission_mass=40,
        child_a_mass=40,
        child_b_mass=40,
        selected_daughter="A",
        overshoot_before_trim=0,
        trimmed_new_entrants=0,
        maximum_exposure=maximum,
        minimum_exposure=maximum,
    )
    return TimebaseTrajectory(
        trajectory_id="fixture",
        phase="development",
        matrix_index=0,
        configuration_id="fixture",
        definition=definition,
        beta_sha256="a" * 64,
        initial_state_sha256="b" * 64,
        observations=(StateObservation(0, "initial_selected_state", 0, 0, 0, 0, state),),
        generations=(generation,),
        completed_fissions=1,
        total_batch_updates=updates,
        total_nonzero_reaction_types=0,
        total_gross_sampled_events=0,
        terminal_status="requested_fissions_completed",
        extinction_generation=None,
        trajectory_sha256="c" * 64,
    )


def test_only_zero_update_exposure_nans_are_normalized() -> None:
    left = trajectory()
    right = trajectory()
    result = compare_trajectories(left, right)
    assert not result.old_comparator_passed
    assert result.repaired_comparator_passed
    assert result.permitted_paired_nan_count == 2
    assert {row.path for row in result.differences} == {
        "trajectory.generations[0].maximum_exposure",
        "trajectory.generations[0].minimum_exposure",
    }
    assert all(row.permitted for row in result.differences)


def test_nan_with_updates_is_forbidden() -> None:
    result = compare_trajectories(trajectory(updates=1), trajectory(updates=1))
    assert not result.repaired_comparator_passed
    assert result.forbidden_nonfinite_difference_count == 2
    assert not any(row.permitted for row in result.differences)


def test_finite_float_is_bit_exact_without_tolerance() -> None:
    left = trajectory(updates=1, maximum=0.25)
    right_generation = replace(
        trajectory(updates=1, maximum=0.25).generations[0],
        maximum_exposure=float(np.nextafter(0.25, 1.0)),
    )
    right = replace(left, generations=(right_generation,))
    result = compare_trajectories(left, right)
    assert not result.repaired_comparator_passed
    assert result.finite_numeric_divergence_count == 1


def test_discrete_sequence_difference_fails() -> None:
    left = trajectory(updates=1, maximum=0.25)
    right_observation = replace(left.observations[0], state=(2,) + (0,) * 99)
    right = replace(left, observations=(right_observation,))
    result = compare_trajectories(left, right)
    assert not result.repaired_comparator_passed
    assert result.discrete_divergence_count == 1


def test_seed_identity_is_exact() -> None:
    seed = SeedIdentity("development", "d" * 64, "poisson_update", 0, "p", (), 7, "e" * 64)
    passed, differences = compare_seed_tuples((seed,), (seed,))
    assert passed and not differences
    changed = replace(seed, derived_seed=8)
    passed, differences = compare_seed_tuples((seed,), (changed,))
    assert not passed
    assert differences[0].category == "DISCRETE_VALUE_DIVERGENCE"


def test_recording_generator_is_replay_exact_and_counts_calls() -> None:
    seed = SeedIdentity("development", "d" * 64, "poisson_update", 0, "p", (), 7, "e" * 64)
    left = RecordingGenerator(seed)
    right = RecordingGenerator(seed)
    left_value = left.poisson(np.asarray([0.1, 0.2], dtype=np.float64))
    right_value = right.poisson(np.asarray([0.1, 0.2], dtype=np.float64))
    assert np.array_equal(left_value, right_value)
    assert left.calls == right.calls
    assert left.start_state_sha256 == right.start_state_sha256
    assert left.end_state_sha256 == right.end_state_sha256
    assert left.calls[0].finite_float_argument_count == 2
    assert left.calls[0].nonfinite_float_argument_count == 0


def test_canonical_hash_distinguishes_signed_zero() -> None:
    assert canonical_sha256(0.0) != canonical_sha256(-0.0)


def test_contract_and_schema_are_valid_and_narrow() -> None:
    contract = yaml.safe_load(
        (REPO / "configs/e01/s12fr/comparator_contract.yaml").read_text()
    )
    assert contract["repairedComparator"]["identifier"] == COMPARATOR_VERSION
    assert contract["repairedComparator"]["finiteFloatRule"] == "exact_IEEE754_binary64_bit_pattern"
    assert contract["repairedComparator"]["permittedNormalization"]["fields"] == [
        "generations[*].maximum_exposure",
        "generations[*].minimum_exposure",
    ]
    schema = json.loads(
        (REPO / "configs/e01/s12fr/pair_diagnostic_schema.json").read_text()
    )
    jsonschema.Draft202012Validator.check_schema(schema)


def test_preregistration_preserves_original_s12f_design() -> None:
    repair = yaml.safe_load(
        (REPO / "configs/e01/s12fr_replay_comparator_repair_preregistration.yaml").read_text()
    )
    original = yaml.safe_load(
        (REPO / "configs/e01/s12f_latent_timebase_preregistration.yaml").read_text()
    )
    assert repair["originalPairCampaign"]["inferenceRoot"] == original["randomness"]["roots"]["inference"]
    assert repair["originalPairCampaign"]["simulatorRoot"] == original["randomness"]["roots"]["development"]
    assert repair["benchmarkCampaign"]["root"] == original["randomness"]["roots"]["benchmark"]
    assert repair["conditionalAbcResume"]["originalS12FParticleCounts"] == [256, 128, 64]
    assert repair["conditionalAbcResume"]["originalS12FConfirmationMatricesPerCandidate"] == 32
    assert not repair["immutability"]["s12fSuppressedDistancesMayBeOpened"]
