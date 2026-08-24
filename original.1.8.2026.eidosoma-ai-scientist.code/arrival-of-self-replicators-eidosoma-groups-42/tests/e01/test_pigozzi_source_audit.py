from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from e01_pigozzi_source_audit.analysis import (
    association_summary,
    percentile_ranks,
    prospective_candidate,
    retrospective_coherent,
    spike_thresholds,
)
from e01_pigozzi_source_audit.core import (
    SourceImplementation,
    derive_seed,
    load_safe_lattice,
    run_source_pipeline,
)

SAFE = Path("/artifacts/research_steps/S12B/safe_phi_lattice.json")


def structured_array() -> np.ndarray:
    rng = np.random.RandomState(712)
    result = rng.normal(size=(320, 8))
    result[:, 4:] += 0.3 * result[:, :4]
    return result


def test_safe_lattice_has_complete_source_order() -> None:
    order, descendants = load_safe_lattice(SAFE)
    assert len(order) == 16
    assert len(descendants) == 16
    assert order[0] == (((0,), (1,)), ((0,), (1,)))


def test_both_source_wrappers_replay_exactly_and_preserve_offsets() -> None:
    observations = structured_array()
    expected = {SourceImplementation.IIGR: 318, SourceImplementation.PHIRL: 319}
    for implementation in SourceImplementation:
        first = run_source_pipeline(observations, implementation, SAFE, preprocessing_seed=19, partition_seed=23)
        second = run_source_pipeline(observations, implementation, SAFE, preprocessing_seed=19, partition_seed=23)
        assert first.status == "ELIGIBLE"
        assert first.retained_variables == second.retained_variables
        assert first.partition_1 == second.partition_1
        assert first.partition_2 == second.partition_2
        assert np.array_equal(first.mi_matrix, second.mi_matrix)
        assert np.array_equal(first.local_phi_r, second.local_phi_r)
        assert first.local_phi_r is not None and len(first.local_phi_r) == expected[implementation]


def test_phirl_constant_input_fails_closed_without_hidden_dimension() -> None:
    result = run_source_pipeline(np.ones((320, 8)), SourceImplementation.PHIRL, SAFE, preprocessing_seed=1, partition_seed=2)
    assert result.status == "INELIGIBLE_TOO_FEW_ACTIVE_DIMENSIONS"
    assert result.retained_variables == ()


def test_prefix_endpoint_is_structurally_suffix_invariant() -> None:
    observations = structured_array()
    endpoint = 200
    seed_a = derive_seed("ab" * 32, "prefix", endpoint, "preprocess")
    seed_b = derive_seed("ab" * 32, "prefix", endpoint, "partition")
    base = run_source_pipeline(observations[: endpoint + 1], SourceImplementation.PHIRL, SAFE, preprocessing_seed=seed_a, partition_seed=seed_b)
    changed = observations.copy()
    changed[endpoint + 1 :] = 1e6
    test = run_source_pipeline(changed[: endpoint + 1], SourceImplementation.PHIRL, SAFE, preprocessing_seed=seed_a, partition_seed=seed_b)
    assert base.status == test.status
    assert base.partition_1 == test.partition_1
    assert base.partition_2 == test.partition_2
    assert np.array_equal(base.local_phi_r, test.local_phi_r)


def test_frozen_association_and_spike_rules_are_deterministic() -> None:
    rows = []
    for trajectory in range(12):
        for generation in range(1, 21):
            label = generation >= 11
            rows.append({"trajectoryId": f"T{trajectory:02d}", "generation": generation, "phiR": float(label) + generation * 1e-3 + trajectory * 1e-4, "label": label})
    frame = pd.DataFrame(rows)
    first = association_summary(frame, value_column="phiR", label_column="label", bootstrap_seed=1, circular_seed=2, replicates=128)
    second = association_summary(frame, value_column="phiR", label_column="label", bootstrap_seed=1, circular_seed=2, replicates=128)
    assert first == second
    coherent, gates = retrospective_coherent(first, finite_coverage=1.0, runs_higher=12)
    assert coherent and all(gates.values())
    candidate, candidate_gates = prospective_candidate(first, coverage=1.0, replay_passed=True, suffix_passed=True, other_opposite=False)
    assert candidate and all(candidate_gates.values())
    thresholds = spike_thresholds(np.arange(20.0))
    assert thresholds["positive3Sigma"] > thresholds["mean"]
    assert np.array_equal(percentile_ranks(np.array([2.0, 1.0, 3.0])), np.array([0.5, 1 / 6, 5 / 6]))
