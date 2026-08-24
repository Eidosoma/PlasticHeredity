from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

from plastic_heredity.config import GardConfig
from plastic_heredity.phir_extension_common import (
    CPU_BUDGET_HOURS,
    MAX_WORKERS,
    expected_flux_observations,
    master_protocol,
    paired_matrix_effects,
    paired_summary,
    purpose_seed,
    score_explicit_pairs,
    score_sequence,
)
from plastic_heredity.phir_rescue_instruments import (
    active_partition,
    beta_physical_partition,
    close_all_clr,
    full_block_revised,
    rank_gaussianize,
)


def _fixture(types: int = 100, observations: int = 34) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3201)
    counts = rng.poisson(0.7, size=(observations, types)).astype(np.int64)
    counts[:, 0] += 20
    counts[:, 1] += np.arange(observations) % 7
    beta = np.exp(rng.normal(-4.0, 1.5, size=(types, types)))
    return counts, beta


def test_material_sequence_exactly_matches_r0_full_block_pipeline() -> None:
    counts, beta = _fixture()
    data, active = rank_gaussianize(close_all_clr(counts))
    physical_a, physical_b = beta_physical_partition(beta)
    part_a, part_b = active_partition(active, physical_a, physical_b)
    expected = full_block_revised(data, part_a, part_b)
    observed = score_sequence(counts, beta, "material")
    assert observed.transitions == counts.shape[0] - 1
    np.testing.assert_allclose(observed.full_revised, expected.revised, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(observed.whole_mi, expected.whole_mi, atol=0.0, rtol=0.0)


def test_explicit_pairs_do_not_create_cross_branch_transitions() -> None:
    counts, beta = _fixture(observations=16)
    past = counts[::2]
    future = counts[1::2]
    explicit = score_explicit_pairs(past, future, beta, "material")
    stitched = score_sequence(np.vstack([pair for pair in zip(past, future) for pair in pair]), beta, "material")
    assert explicit.transitions == len(past)
    assert stitched.transitions == 2 * len(past) - 1
    assert explicit.full_revised != stitched.full_revised


def test_functional_flux_matches_frozen_kinetic_equations() -> None:
    counts, beta = _fixture(observations=5)
    config = GardConfig()
    observed = expected_flux_observations(counts, beta, config)
    masses = counts.sum(axis=1).astype(float)
    boost = 1.0 + (counts @ beta.T) / masses[:, None]
    join = config.k_join * (1.0 / config.n_types) * masses[:, None] * boost
    leave = config.k_leave * counts * boost
    first, second = beta_physical_partition(beta)
    expected = np.column_stack(
        (join[:, first].sum(1), leave[:, first].sum(1), join[:, second].sum(1), leave[:, second].sum(1))
    )
    np.testing.assert_allclose(observed, expected, atol=0.0, rtol=0.0)


def test_material_and_flux_scores_are_simultaneously_label_invariant() -> None:
    counts, beta = _fixture()
    permutation = np.random.default_rng(88).permutation(counts.shape[1])
    pcounts = counts[:, permutation]
    pbeta = beta[np.ix_(permutation, permutation)]
    for representation in ("material", "functional_flux"):
        original = score_sequence(counts, beta, representation)
        permuted = score_sequence(pcounts, pbeta, representation)
        np.testing.assert_allclose(original.full_revised, permuted.full_revised, atol=1e-8)
        np.testing.assert_allclose(original.public_revised, permuted.public_revised, atol=1e-8)


def test_matrix_effects_keep_repeated_observations_inside_matrix() -> None:
    rows = []
    for matrix_id in range(4):
        for landmark in (20, 40):
            rows.extend(
                [
                    {"matrix_id": matrix_id, "candidate": "02", "landmark": landmark, "arm": "UP", "value": matrix_id + 2.0},
                    {"matrix_id": matrix_id, "candidate": "02", "landmark": landmark, "arm": "DOWN", "value": matrix_id + 1.0},
                ]
            )
    effects = paired_matrix_effects(
        pd.DataFrame(rows), "value", "UP", "DOWN", filters={"candidate": "02"}, within=("landmark",)
    )
    np.testing.assert_array_equal(effects.to_numpy(), np.ones(4))
    summary, arrays = paired_summary(effects.to_numpy(), "fixture", bootstrap_draws=32, randomization_draws=32)
    assert summary["matrices"] == 4
    assert arrays["matrix_values"].shape == (4,)


def test_future_seed_interface_has_no_arm_or_variant() -> None:
    assert "arm" not in inspect.signature(purpose_seed).parameters
    assert "variant" not in inspect.signature(purpose_seed).parameters
    assert purpose_seed("future", "PX1", "02", 3, 0) != purpose_seed(
        "random_action", "PX1", "02", 3, 0
    )


def test_master_protocol_is_bounded_and_runs_all_phases() -> None:
    protocol = master_protocol()
    assert protocol["matrix_scale"] == 24
    assert protocol["no_48_matrix_campaign"]
    assert protocol["run_all_phases_without_evidence_gating"]
    assert sum(protocol["phase_allocations_cpu_hours"].values()) == CPU_BUDGET_HOURS
    assert MAX_WORKERS == 12
