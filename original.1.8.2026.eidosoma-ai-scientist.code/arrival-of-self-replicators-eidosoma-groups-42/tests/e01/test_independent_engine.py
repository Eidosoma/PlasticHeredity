from __future__ import annotations

import hashlib
import inspect
import json
import os
import subprocess
import sys
from dataclasses import MISSING, fields
from pathlib import Path

import numpy as np
import pytest
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from e01_gard_independent import (
    BatchLossError,
    CatalyticMatrixBranch,
    DaughterSelection,
    EmptyDaughterError,
    GardSpecification,
    GrowthLimitError,
    IndependentGardError,
    RNGInput,
    RNGStreams,
    SpecificationError,
    advance_generation,
    calculate_propensities,
    fission,
    generate_catalytic_matrix,
    generator_state_sha256,
    grow,
    initialize_state,
    integer_state,
    sample_update,
    simulate_lineage,
    specification_from_mapping,
)

PROFILES_PATH = REPOSITORY_ROOT / "configs/e01/s05_specification_profiles.yaml"
CONTRACT_PATH = REPOSITORY_ROOT / "configs/e01/s05_independent_contract.yaml"


def _raw_profile(profile_id: str) -> dict[str, object]:
    profiles = yaml.safe_load(PROFILES_PATH.read_text())
    raw = dict(profiles["profiles"][profile_id])
    raw.pop("evidenceBoundary")
    return raw


def _spec(profile_id: str, **updates: object) -> GardSpecification:
    raw = _raw_profile(profile_id)
    raw.update(updates)
    return specification_from_mapping(raw)


def _streams(*seeds: int) -> RNGStreams:
    if not seeds:
        seeds = (101, 102, 103, 104, 105, 106)
    assert len(seeds) == 6
    names = ("beta", "init", "events", "waiting", "fission", "daughter")
    inputs = [
        RNGInput(f"test-{name}-{seed}", np.random.default_rng(seed))
        for name, seed in zip(names, seeds, strict=True)
    ]
    return RNGStreams(*inputs)


HISTORICAL_PROFILE = "E01-S05-HISTORICAL-DISTRIBUTION-COMPARISON-v1.0.0"
GILLESPIE_PROFILE = "E01-S05-MODERN-GILLESPIE-FIXTURE-v1.0.0"
POISSON_PROFILE = "E01-S05-PAPER-POISSON-FIXTURE-v1.0.0"


def test_specification_is_complete_fail_closed_and_registry_safe() -> None:
    assert all(
        field.default is MISSING and field.default_factory is MISSING
        for field in fields(GardSpecification)
    )
    for profile_id in (HISTORICAL_PROFILE, GILLESPIE_PROFILE, POISSON_PROFILE):
        assert _spec(profile_id).specification_id == profile_id

    missing = _raw_profile(HISTORICAL_PROFILE)
    missing.pop("k_f")
    with pytest.raises(SpecificationError, match="missing=.*k_f"):
        specification_from_mapping(missing)

    extra = _raw_profile(HISTORICAL_PROFILE)
    extra["unstated_default"] = 1
    with pytest.raises(SpecificationError, match="extra=.*unstated_default"):
        specification_from_mapping(extra)

    sentinel = _raw_profile(HISTORICAL_PROFILE)
    sentinel["k_f"] = "UNRESOLVED::E01-A009"
    with pytest.raises(SpecificationError, match="forbidden registry sentinel"):
        specification_from_mapping(sentinel)

    raw_branch_set = _raw_profile(HISTORICAL_PROFILE)
    raw_branch_set["daughter_selection"] = "BRANCH_SET::first|uniform_random"
    with pytest.raises(SpecificationError, match="forbidden registry sentinel"):
        specification_from_mapping(raw_branch_set)

    direct = _spec(HISTORICAL_PROFILE)
    with pytest.raises(SpecificationError, match="explicit ProfileRole"):
        GardSpecification(
            **{
                **{field.name: getattr(direct, field.name) for field in fields(direct)},
                "profile_role": "historical_distribution_comparison",
            }
        )


def test_rng_inputs_are_modern_separated_and_caller_owned() -> None:
    streams = _streams()
    descriptions = streams.descriptions()
    assert len(descriptions) == 6
    assert {item["purpose"] for item in descriptions} == {
        "catalytic_matrix",
        "initialization",
        "events",
        "waiting_time",
        "fission",
        "daughter",
    }
    assert {item["bitGenerator"] for item in descriptions} == {"PCG64"}

    shared = np.random.default_rng(99)
    with pytest.raises(SpecificationError, match="distinct objects"):
        RNGStreams(
            RNGInput("a", shared),
            RNGInput("b", shared),
            RNGInput("c", np.random.default_rng(3)),
            RNGInput("d", np.random.default_rng(4)),
            RNGInput("e", np.random.default_rng(5)),
            RNGInput("f", np.random.default_rng(6)),
        )


def test_integer_state_and_initialization_invariants() -> None:
    np.testing.assert_array_equal(integer_state([2, 0, 1], name="state"), [2, 0, 1])
    for invalid in ([1.5, 0], [-1, 2], [np.nan, 1], [], [[1, 2]]):
        with pytest.raises(IndependentGardError):
            integer_state(invalid, name="state")

    paper = _spec(POISSON_PROFILE)
    distinct = initialize_state(paper, _streams().initialization)
    assert sum(distinct) == paper.n_min
    assert set(distinct) <= {0, 1}

    historical = _spec(HISTORICAL_PROFILE)
    counts = initialize_state(historical, _streams().initialization)
    assert sum(counts) == historical.n_min
    assert all(isinstance(value, int) and value >= 0 for value in counts)


def test_catalytic_matrix_and_hand_propensity_arrays() -> None:
    spec = _spec(
        HISTORICAL_PROFILE,
        specification_id="E01-S05-HAND-PROPENSITY-v1",
        n_species=2,
        n_min=1,
        n_max=4,
        k_f=0.1,
        k_b=0.2,
        rho=[0.25, 0.75],
    )
    beta = [[1.0, 2.0], [0.5, 0.0]]
    props = calculate_propensities([2, 1], beta=beta, specification=spec)
    np.testing.assert_allclose(props.boost, [7 / 3, 4 / 3])
    np.testing.assert_allclose(props.join, [0.175, 0.3])
    np.testing.assert_allclose(props.leave, [14 / 15, 4 / 15])
    np.testing.assert_allclose(
        props.probabilities,
        np.asarray(props.concatenated) / props.total,
    )
    assert props.total == pytest.approx(1.675)

    first = generate_catalytic_matrix(
        spec, RNGInput("beta-a", np.random.default_rng(7))
    )
    second = generate_catalytic_matrix(
        spec, RNGInput("beta-b", np.random.default_rng(7))
    )
    np.testing.assert_array_equal(first, second)
    assert np.all(first > 0)

    transposed = _spec(
        HISTORICAL_PROFILE,
        specification_id="E01-S05-TRANSPOSE-v1",
        n_species=2,
        n_min=1,
        n_max=4,
        k_f=0.1,
        k_b=0.2,
        rho=[0.25, 0.75],
        catalytic_matrix_branch=CatalyticMatrixBranch.TRANSPOSED_WITH_DIAGONAL.value,
    )
    transposed_props = calculate_propensities(
        [2, 1], beta=beta, specification=transposed
    )
    assert transposed_props.boost != props.boost


def test_categorical_and_gillespie_events_have_complete_separated_logs() -> None:
    historical = _spec(HISTORICAL_PROFILE)
    streams = _streams()
    waiting_before = generator_state_sha256(streams.waiting_time.generator)
    fission_before = generator_state_sha256(streams.fission.generator)
    event = sample_update(
        [1, 1, 0],
        beta=np.zeros((3, 3)),
        specification=historical,
        rng_streams=streams,
        generation_index_one_based=1,
        step_index_one_based=1,
        model_time_before=None,
    )
    assert event.mass_delta in {-1, 1}
    assert event.event_kind in {"join", "leave"}
    assert sum(event.applied_join_counts) + sum(event.applied_loss_counts) == 1
    assert event.pre_mass + event.mass_delta == event.post_mass
    assert sum(event.event_probabilities) == pytest.approx(1.0)
    assert event.waiting_rng_stream_id is None
    assert generator_state_sha256(streams.waiting_time.generator) == waiting_before
    assert generator_state_sha256(streams.fission.generator) == fission_before
    assert event.event_rng_state_sha256_before != event.event_rng_state_sha256_after

    gillespie = _spec(GILLESPIE_PROFILE)
    gillespie_streams = _streams(201, 202, 203, 204, 205, 206)
    timed = sample_update(
        [1, 1, 0],
        beta=np.zeros((3, 3)),
        specification=gillespie,
        rng_streams=gillespie_streams,
        generation_index_one_based=1,
        step_index_one_based=1,
        model_time_before=0.0,
    )
    assert timed.time_increment is not None and timed.time_increment >= 0
    assert timed.model_time_after == timed.time_increment
    assert timed.waiting_rng_stream_id == gillespie_streams.waiting_time.stream_id
    assert timed.waiting_rng_state_sha256_before != timed.waiting_rng_state_sha256_after


def test_vector_poisson_integer_nonnegativity_and_explicit_loss_failure() -> None:
    poisson = _spec(POISSON_PROFILE)
    event = sample_update(
        [2, 1, 0],
        beta=np.zeros((3, 3)),
        specification=poisson,
        rng_streams=_streams(301, 302, 303, 304, 305, 306),
        generation_index_one_based=1,
        step_index_one_based=1,
        model_time_before=0.0,
    )
    assert event.event_kind == "vector_poisson_batch"
    assert all(isinstance(value, int) and value >= 0 for value in event.post_state)
    assert event.time_increment == poisson.poisson_exposure
    assert event.post_mass - event.pre_mass == event.mass_delta

    failing = _spec(
        POISSON_PROFILE,
        specification_id="E01-S05-POISSON-LOSS-ERROR-v1",
        k_f=0.0,
        k_b=100.0,
        poisson_exposure=1.0,
        loss_nonnegativity="error_on_batch_excess",
    )
    with pytest.raises(BatchLossError):
        sample_update(
            [1, 0, 0],
            beta=np.zeros((3, 3)),
            specification=failing,
            rng_streams=_streams(311, 312, 313, 314, 315, 316),
            generation_index_one_based=1,
            step_index_one_based=1,
            model_time_before=0.0,
        )


def test_growth_limits_exact_mass_and_raise_branch() -> None:
    historical = _spec(
        HISTORICAL_PROFILE,
        specification_id="E01-S05-GROWTH-EXACT-v1",
        k_f=1.0,
        k_b=0.0,
    )
    result = grow(
        [1, 1, 0],
        beta=np.zeros((3, 3)),
        specification=historical,
        rng_streams=_streams(),
        generation_index_one_based=1,
    )
    assert sum(result.final_state) == historical.n_max
    assert result.terminal_status == "n_max_reached"
    assert len(result.events) == historical.n_max - 2
    assert {event.mass_delta for event in result.events} == {1}

    bounded = _spec(
        GILLESPIE_PROFILE,
        specification_id="E01-S05-GROWTH-LIMIT-v1",
        n_max=20,
        max_steps=1,
        max_steps_semantics="stop_without_fission",
        k_f=1.0,
        k_b=0.0,
    )
    stopped = grow(
        [1, 1, 0],
        beta=np.zeros((3, 3)),
        specification=bounded,
        rng_streams=_streams(),
        generation_index_one_based=1,
    )
    assert stopped.terminal_status == "max_steps_reached"
    assert len(stopped.events) == 1

    raising = _spec(
        GILLESPIE_PROFILE,
        specification_id="E01-S05-GROWTH-RAISE-v1",
        n_max=20,
        max_steps=1,
        max_steps_semantics="raise",
        k_f=1.0,
        k_b=0.0,
    )
    with pytest.raises(GrowthLimitError) as error:
        grow(
            [1, 1, 0],
            beta=np.zeros((3, 3)),
            specification=raising,
            rng_streams=_streams(),
            generation_index_one_based=1,
        )
    assert error.value.result.terminal_status == "max_steps_reached"


def test_fission_branches_conserve_mass_and_separate_daughter_rng() -> None:
    historical = _spec(HISTORICAL_PROFILE)
    streams = _streams()
    even = fission(
        [2, 1, 1],
        specification=historical,
        rng_streams=streams,
        generation_index_one_based=1,
    )
    np.testing.assert_array_equal(
        np.asarray(even.child_first)
        + np.asarray(even.child_second)
        + np.asarray(even.discarded),
        even.parent,
    )
    assert sum(even.child_first) == sum(even.child_second) == 2
    assert not even.daughter_rng_consumed
    assert even.daughter_rng_state_sha256_before == even.daughter_rng_state_sha256_after

    odd = fission(
        [2, 1, 0],
        specification=historical,
        rng_streams=_streams(401, 402, 403, 404, 405, 406),
        generation_index_one_based=1,
    )
    assert sum(odd.child_first) == sum(odd.child_second) == 1
    assert sum(odd.discarded) == 1

    gillespie = _spec(GILLESPIE_PROFILE)
    binomial = fission(
        [3, 2, 1],
        specification=gillespie,
        rng_streams=_streams(411, 412, 413, 414, 415, 416),
        generation_index_one_based=1,
    )
    np.testing.assert_array_equal(
        np.asarray(binomial.child_first) + np.asarray(binomial.child_second),
        binomial.parent,
    )
    assert sum(binomial.discarded) == 0
    assert binomial.daughter_rng_consumed


def test_empty_daughter_policy_and_complete_lineage_logs() -> None:
    empty_rejecting = _spec(
        GILLESPIE_PROFILE,
        specification_id="E01-S05-EMPTY-DAUGHTER-v1",
        fission_probability=0.0,
        daughter_selection=DaughterSelection.FIRST.value,
        post_fission_semantics="error_if_selected_empty",
    )
    with pytest.raises(EmptyDaughterError):
        fission(
            [1, 0, 0],
            specification=empty_rejecting,
            rng_streams=_streams(),
            generation_index_one_based=1,
        )

    lineage_spec = _spec(
        GILLESPIE_PROFILE,
        specification_id="E01-S05-LINEAGE-LOG-v1",
        k_f=1.0,
        k_b=0.0,
        daughter_selection="first",
        fission_probability=1.0,
    )
    result = simulate_lineage(
        [1, 1, 0],
        beta=np.zeros((3, 3)),
        specification=lineage_spec,
        rng_streams=_streams(),
    )
    assert len(result.generations) == lineage_spec.n_generations
    assert len(result.fissions) == lineage_spec.n_generations
    assert result.completed_fissions == lineage_spec.n_generations
    assert result.events
    for generation in result.generations:
        assert all(
            event.generation_index_one_based == generation.generation_index_one_based
            for event in generation.growth.events
        )
        assert [
            event.step_index_one_based for event in generation.growth.events
        ] == list(range(1, len(generation.growth.events) + 1))
    assert all(
        event.specification_id == lineage_spec.specification_id
        for event in result.events
    )


def test_independent_package_has_no_s04_import_and_preserves_frozen_evidence() -> None:
    package = REPOSITORY_ROOT / "src/e01_gard_independent"
    for path in package.glob("*.py"):
        assert "e01_gard_historical" not in path.read_text()

    contract = yaml.safe_load(CONTRACT_PATH.read_text())
    assert (
        contract["implementationBoundary"]["forbiddenImport"] == "e01_gard_historical"
    )
    registry_path = Path(contract["frozenEvidence"]["specificationRegistry"]["path"])
    registry_hash = hashlib.sha256(registry_path.read_bytes()).hexdigest()
    assert (
        registry_hash == contract["frozenEvidence"]["specificationRegistry"]["sha256"]
    )
    registry = yaml.safe_load(registry_path.read_text())
    assert registry["executionGate"]["executable"] is False
    assert registry["executionGate"]["noSilentDefaults"] is True
    assert registry["executionGate"]["unresolvedParameterCount"] == 64
    assert registry["executionGate"]["unexpandedBranchSetCount"] == 21

    signature = inspect.signature(advance_generation)
    assert all(
        parameter.default is inspect.Parameter.empty
        for parameter in signature.parameters.values()
    )


def test_s05_artifact_builder_quick_round_trip_and_s06_absence(
    tmp_path: Path,
) -> None:
    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "scripts/e01/build_independent_engine_artifacts.py"),
        "--artifacts-dir",
        str(tmp_path),
        "--quick",
        "--workers",
        "1",
    ]
    environment = {
        **dict(os.environ),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(REPOSITORY_ROOT / "src"),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMBA_NUM_THREADS": "1",
    }
    result = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert '"success": true' in result.stdout.lower()
    step_dir = tmp_path / "research_steps/S05"
    shared_dir = (
        tmp_path / "E01_forensic_replication_bundle/software/independent_engine"
    )
    required = [
        step_dir / "validation_summary.json",
        step_dir / "distributional_agreement.csv",
        step_dir / "distributional_agreement_details.json",
        step_dir / "unit_invariants.json",
        step_dir / "registry_preservation.json",
        step_dir / "artifact_manifest.json",
        shared_dir / "engine_pointer.json",
        shared_dir / "independent_engine_contract.yaml",
        shared_dir / "validation_profiles.yaml",
        shared_dir / "branch_catalog.csv",
        shared_dir / "api_surface.json",
        shared_dir / "diagnostic_event_log_fixture.json",
        shared_dir / "benchmark.json",
    ]
    assert all(path.is_file() and path.stat().st_size > 0 for path in required)
    validation = json.loads((step_dir / "validation_summary.json").read_text())
    assert validation["researchStepId"] == "S05"
    assert validation["stepNumber"] == 5
    assert validation["success"] is True
    assert validation["quickMode"] is True
    assert validation["unitValidation"] == {"checkCount": 22, "passedCount": 22}
    assert validation["distributionalValidation"]["gateCount"] == 8
    assert not (tmp_path / "research_steps/S06").exists()
