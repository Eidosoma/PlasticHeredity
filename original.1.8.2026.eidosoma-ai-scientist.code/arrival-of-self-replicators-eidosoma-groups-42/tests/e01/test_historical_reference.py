from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from e01_gard_historical import (
    HistoricalReferenceError,
    HistoricalSourceDomainError,
    RandomTapeExhausted,
    UniformTape,
    advance_one_generation,
    catalytic_matrix_from_numpy_rng_explicit,
    catalytic_matrix_from_standard_normals,
    compute_propensities,
    grow_to_split_size,
    historical_h,
    historical_initial_state_with_replacement,
    historical_nondrift_technique1,
    historical_nondrift_technique2,
    historical_single_event,
    historical_weighted_index,
    simulate_lineage,
    split_fixed_size_without_replacement,
)


def test_catalytic_matrix_and_hand_propensities() -> None:
    beta = catalytic_matrix_from_standard_normals(
        [[0.0, 1.0], [-1.0, 0.5]], a=-4.0, sigma=4.0
    )
    np.testing.assert_allclose(
        beta,
        [[np.exp(-4.0), 1.0], [np.exp(-8.0), np.exp(-2.0)]],
        rtol=1e-14,
        atol=0.0,
    )
    assert np.all(np.diag(beta) > 0)

    propensities = compute_propensities(
        [2, 1],
        beta=[[1.0, 2.0], [0.5, 0.0]],
        rho=[0.25, 0.75],
        k_f=0.1,
        k_b=0.2,
    )
    np.testing.assert_allclose(propensities.boost, [7 / 3, 4 / 3])
    np.testing.assert_allclose(propensities.join, [0.175, 0.3])
    np.testing.assert_allclose(propensities.leave, [14 / 15, 4 / 15])
    assert propensities.total == pytest.approx(1.675)


def test_numpy_beta_path_is_explicit_and_distribution_compatible() -> None:
    first = catalytic_matrix_from_numpy_rng_explicit(
        3, a=-4.0, sigma=4.0, generator=np.random.default_rng(123)
    )
    second = catalytic_matrix_from_numpy_rng_explicit(
        3, a=-4.0, sigma=4.0, generator=np.random.default_rng(123)
    )
    np.testing.assert_array_equal(first, second)
    assert np.all(first > 0)


def test_weighted_boundaries_zero_redraw_and_mass_changes() -> None:
    assert historical_weighted_index([1.0, 0.0, 2.0], 1 / 3)[0] == 0
    assert historical_weighted_index([1.0, 0.0, 2.0], 0.5)[0] == 2
    assert historical_weighted_index([1.0, 0.0, 2.0], 1.0)[0] == 2

    common = {
        "beta": [[1.0, 2.0], [0.5, 0.0]],
        "rho": [0.25, 0.75],
        "k_f": 0.1,
        "k_b": 0.2,
    }
    joined = historical_single_event(
        [2, 1], uniform_source=UniformTape((0.0, 0.2)), **common
    )
    left = historical_single_event([2, 1], uniform_source=UniformTape((0.7,)), **common)
    assert (joined.kind, joined.post_state, joined.mass_delta) == ("join", (2, 2), 1)
    assert (left.kind, left.post_state, left.mass_delta) == ("leave", (1, 1), -1)


def test_growth_rate_sum_boundary_extinction_and_guard() -> None:
    common = {
        "beta": np.zeros((2, 2)),
        "rho": [0.5, 0.5],
        "k_f": 1.0,
        "k_b": 0.0,
        "n_max": 4,
    }
    result = grow_to_split_size(
        [1, 1], uniform_source=UniformTape((0.25, 0.75)), event_guard=10, **common
    )
    assert result.final_state == (2, 2)
    assert result.terminal_status == "split_size_reached"
    assert [event.mass_delta for event in result.events] == [1, 1]
    assert [event.total_rate for event in result.events] == [2.0, 3.0]
    assert result.legacy_dt_accumulator == 5.0
    assert result.legacy_inverse_rate_sum == 0.2

    extinct = grow_to_split_size(
        [1],
        beta=[[0.0]],
        rho=[0.0],
        k_f=0.0,
        k_b=1.0,
        n_max=2,
        uniform_source=UniformTape((0.5,)),
        event_guard=10,
    )
    assert extinct.final_state == (0,)
    assert extinct.terminal_status == "extinct"
    assert extinct.events[0].mass_delta == -1

    with pytest.raises(HistoricalReferenceError, match="event_guard"):
        grow_to_split_size(
            [1, 1], uniform_source=UniformTape((0.25, 0.75)), event_guard=1, **common
        )
    with pytest.raises(RandomTapeExhausted):
        grow_to_split_size(
            [1, 1], uniform_source=UniformTape((0.25,)), event_guard=10, **common
        )


def test_fission_even_odd_and_daughter_selection() -> None:
    even = split_fixed_size_without_replacement(
        [2, 1, 1], uniform_source=UniformTape((0.1, 0.9))
    )
    assert even.child_a == (1, 0, 1)
    assert even.child_b == (1, 1, 0)
    assert even.discarded == (0, 0, 0)
    assert even.followed_daughter == even.child_a
    assert (
        even.daughter_selection_rule == "FIRST_OUTPUT_CHILD_A_NO_ADDITIONAL_RANDOM_DRAW"
    )

    odd = split_fixed_size_without_replacement(
        [2, 1], uniform_source=UniformTape((0.9, 0.2))
    )
    assert odd.child_a == (0, 1)
    assert odd.child_b == (1, 0)
    assert odd.discarded == (1, 0)
    np.testing.assert_array_equal(
        np.asarray(odd.child_a) + np.asarray(odd.child_b) + np.asarray(odd.discarded),
        [2, 1],
    )

    tape = UniformTape((0.25, 0.75, 0.1, 0.9))
    generation = advance_one_generation(
        [1, 1],
        beta=np.zeros((2, 2)),
        rho=[0.5, 0.5],
        k_f=1.0,
        k_b=0.0,
        n_max=4,
        uniform_source=tape,
        event_guard=10,
    )
    assert generation.next_state == (1, 1)
    assert generation.terminal_status == "continued_from_child_a"
    assert tape.consumed == 4


def test_two_generation_lineage_records_pre_fission_trace() -> None:
    tape = UniformTape((0.25, 0.75, 0.1, 0.9, 0.25, 0.75, 0.1, 0.9))
    result = simulate_lineage(
        [1, 1],
        beta=np.zeros((2, 2)),
        rho=[0.5, 0.5],
        k_f=1.0,
        k_b=0.0,
        n_max=4,
        n_generations=2,
        uniform_source=tape,
        event_guard_per_generation=10,
    )
    assert result.pre_fission_trace == ((2, 2), (2, 2))
    assert result.final_state == (1, 1)
    assert result.completed_fissions == 2
    assert result.terminal_status == "requested_generations_completed"
    assert tape.consumed == 8


def test_historical_initializer_is_with_replacement_only() -> None:
    result = historical_initial_state_with_replacement(
        n_g=3,
        n_min=4,
        uniform_source=UniformTape((0.0, 0.2, 0.4, 0.99)),
    )
    assert result == (2, 1, 1)
    assert sum(result) == 4


def test_h_and_nondrift_technique1_exact_alignment() -> None:
    h = historical_h([[1.0, 0.0], [0.0, 1.0]], [[0.8], [0.6]])
    np.testing.assert_allclose(h, [[0.8], [0.6]])

    result = historical_nondrift_technique1(
        [[1.0, 1.0, 0.8, 0.0, 0.0], [0.0, 0.0, 0.6, 1.0, 0.0]],
        threshold=0.9,
    )
    np.testing.assert_allclose(result.angles, [1.0, 1.0, 0.8, 0.6, 0.0])
    np.testing.assert_allclose(result.local_scores, [1.0, 0.9, 0.7, 0.6, 0.0])
    assert result.is_non_drift == (True, False, False, False, False)
    assert result.first_zero_sum_generation_one_based == 5


def test_nondrift_technique2_shift_and_source_edge_failure() -> None:
    result = historical_nondrift_technique2(
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 1.0]],
        threshold=0.9,
        drift_size=2,
    )
    assert result.is_non_drift == (False, True, True, False)
    np.testing.assert_allclose(result.angles, [0.0, 0.0, 1.0, 1.0])

    with pytest.raises(HistoricalSourceDomainError, match="index 0"):
        historical_nondrift_technique2(
            [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]],
            threshold=0.9,
            drift_size=2,
        )


def test_contract_maps_every_s04_parameter_and_pinned_source_line() -> None:
    contract = yaml.safe_load(
        (REPOSITORY_ROOT / "configs/e01/s04_historical_contract.yaml").read_text()
    )
    registry_path = Path(contract["specificationRegistry"]["path"])
    registry = yaml.safe_load(registry_path.read_text())
    s04_parameters = {
        item["parameter"]
        for item in registry["parameters"]
        if item.get("ownerStep") == "S04"
    }
    mapped = {item["parameter"] for item in contract["s04RegistryMappings"]}
    assert len(s04_parameters) == 19
    assert mapped == s04_parameters
    assert registry["executionGate"]["executable"] is False
    assert registry["executionGate"]["noSilentDefaults"] is True

    source_root = Path(contract["sourceIdentity"]["localPath"])
    for item in contract["sourceFiles"]:
        assert (source_root / item["path"]).is_file()
    for mapping in contract["sourceLineMappings"]:
        lines = (source_root / mapping["sourceFile"]).read_text().splitlines()
        assert 1 <= mapping["lineStart"] <= mapping["lineEnd"] <= len(lines)


def test_artifact_builder_round_trip_and_scope(tmp_path: Path) -> None:
    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "scripts/e01/build_historical_reference_artifacts.py"),
        "--artifacts-dir",
        str(tmp_path),
    ]
    result = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert '"success": true' in result.stdout.lower()
    step_dir = tmp_path / "research_steps/S04"
    shared_dir = (
        tmp_path / "E01_forensic_replication_bundle/software/historical_reference"
    )
    required = [
        step_dir / "validation_summary.json",
        step_dir / "registry_preservation.json",
        step_dir / "artifact_manifest.json",
        shared_dir / "engine_pointer.json",
        shared_dir / "historical_behavior_contract.yaml",
        shared_dir / "compatibility_notes.md",
        shared_dir / "source_traceability.csv",
        shared_dir / "compatibility_matrix.csv",
        shared_dir / "verified_small_cases.json",
    ]
    assert all(path.is_file() and path.stat().st_size > 0 for path in required)
    validation = json.loads((step_dir / "validation_summary.json").read_text())
    assert validation["researchStepId"] == "S04"
    assert validation["success"] is True
    assert validation["fixtureValidation"]["passedCount"] == 15
    assert validation["registryValidation"]["s04OwnedParameterCount"] == 19
    cases = json.loads((shared_dir / "verified_small_cases.json").read_text())
    assert cases["summary"]["allPassed"] is True
    assert not (tmp_path / "research_steps/S05").exists()
