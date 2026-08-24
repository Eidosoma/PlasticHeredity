from __future__ import annotations

from dataclasses import replace

import numpy as np

from aor_replication.bridge_information import (
    ALL_ATOMS,
    ESTIMATOR_ORDER,
    PHIR_ATOMS,
    fit_bridge_estimators,
    full_block_local_scores,
    gaussian_mutual_information,
    local_phi_id_atoms,
)
from aor_replication.config import CausalConfig
from aor_replication.formulation_bridge import PILOT_SEEDS, frozen_pilot_config
from aor_replication.information import fit_causal_trajectory


def _count_fixture(
    seed: int = 3201, observations: int = 180, types: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    counts = rng.poisson(1.2, size=(observations, types)).astype(np.int64)
    counts[:, 0] += 8 + np.arange(observations) % 5
    counts[:, 1] += (np.arange(observations) // 3) % 4
    beta = np.exp(rng.normal(-4.0, 1.5, size=(types, types)))
    return counts, beta


def _var(
    transition: np.ndarray, *, seed: int, samples: int = 6000
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = np.zeros((transition.shape[0], samples), dtype=np.float64)
    noise = rng.normal(size=values.shape)
    for index in range(1, samples):
        values[:, index] = transition @ values[:, index - 1] + noise[:, index]
    return values[:, 500:]


def test_original_macro_cells_exactly_replay_existing_implementation() -> None:
    counts, beta = _count_fixture()
    observed = fit_bridge_estimators(counts, beta)
    wms = fit_causal_trajectory(counts, CausalConfig(measure="wms"))
    mmi = fit_causal_trajectory(counts, CausalConfig(measure="mmi_synergy"))
    np.testing.assert_allclose(observed["macro_wms"].values, wms.values, atol=0, rtol=0)
    np.testing.assert_allclose(observed["macro_mmi"].values, mmi.values, atol=0, rtol=0)
    np.testing.assert_array_equal(observed["macro_wms"].time_indices, wms.time_indices)
    np.testing.assert_array_equal(observed["macro_mmi"].time_indices, mmi.time_indices)


def test_full_block_local_mean_matches_all_global_channel_identities() -> None:
    data = _var(
        np.asarray(
            [
                [0.55, 0.10, 0.00, 0.00],
                [0.05, 0.45, 0.00, 0.05],
                [0.00, 0.15, 0.50, 0.10],
                [0.10, 0.00, 0.05, 0.40],
            ]
        ),
        seed=88,
    )
    first = np.asarray([0, 1], dtype=np.int64)
    second = np.asarray([2, 3], dtype=np.int64)
    local, components, channel = full_block_local_scores(data, first, second)
    past, future = data[:, :-1], data[:, 1:]
    expected = {
        "whole_mi": gaussian_mutual_information(past, future),
        "aa_mi": gaussian_mutual_information(past[first], future[first]),
        "ab_mi": gaussian_mutual_information(past[first], future[second]),
        "ba_mi": gaussian_mutual_information(past[second], future[first]),
        "bb_mi": gaussian_mutual_information(past[second], future[second]),
    }
    for name, value in expected.items():
        np.testing.assert_allclose(components[name], value, atol=1e-12, rtol=0)
    minimum = min(("aa", "ab", "ba", "bb"), key=lambda name: expected[f"{name}_mi"])
    assert channel == minimum
    revised = (
        expected["whole_mi"]
        - expected["aa_mi"]
        - expected["bb_mi"]
        + expected[f"{minimum}_mi"]
    )
    np.testing.assert_allclose(local.mean(), revised, atol=1e-10, rtol=0)
    np.testing.assert_allclose(components["global_revised"], revised, atol=1e-12, rtol=0)


def test_full_block_scalar_is_partition_swap_invariant() -> None:
    data = _var(np.diag([0.6, 0.5, 0.4, 0.3]), seed=19, samples=2500)
    first = np.asarray([0, 1], dtype=np.int64)
    second = np.asarray([2, 3], dtype=np.int64)
    left, left_components, _ = full_block_local_scores(data, first, second)
    right, right_components, _ = full_block_local_scores(data, second, first)
    np.testing.assert_allclose(
        left_components["global_revised"],
        right_components["global_revised"],
        atol=1e-12,
        rtol=0,
    )
    np.testing.assert_allclose(left.mean(), right.mean(), atol=1e-10, rtol=0)


def test_public_estimator_selects_exactly_nine_of_all_sixteen_atoms() -> None:
    rng = np.random.default_rng(4)
    past = rng.normal(size=(2, 800))
    future = np.asarray(
        [
            0.35 * past[0] + 0.25 * past[1],
            -0.20 * past[0] + 0.45 * past[1],
        ]
    ) + 0.7 * rng.normal(size=(2, 800))
    atoms = local_phi_id_atoms(past, future)
    assert set(atoms) == set(ALL_ATOMS)
    assert len(atoms) == 16
    assert len(PHIR_ATOMS) == 9
    revised = np.sum([atoms[atom] for atom in PHIR_ATOMS], axis=0)
    np.testing.assert_allclose(
        revised.mean(),
        sum(float(atoms[atom].mean()) for atom in PHIR_ATOMS),
        atol=1e-14,
        rtol=0,
    )


def test_px_port_matches_source_hashed_synthetic_parity_fixture() -> None:
    rng = np.random.default_rng(404)
    past = rng.normal(size=(2, 800))
    future = np.asarray(
        [
            0.35 * past[0] + 0.25 * past[1],
            -0.20 * past[0] + 0.45 * past[1],
        ]
    ) + 0.7 * rng.normal(size=(2, 800))
    atoms = local_phi_id_atoms(past, future)
    public_revised = sum(float(atoms[atom].mean()) for atom in PHIR_ATOMS)
    _, full, channel = full_block_local_scores(
        np.vstack((past, future)),
        np.asarray([0, 1], dtype=np.int64),
        np.asarray([2, 3], dtype=np.int64),
    )
    # Frozen from the three source-hashed plastic-heredity reference modules.
    np.testing.assert_allclose(public_revised, 0.2946904999004378, atol=1e-14, rtol=0)
    np.testing.assert_allclose(
        full["global_revised"], 0.008803353462591645, atol=1e-14, rtol=0
    )
    assert channel == "aa"


def test_px_estimators_are_simultaneously_molecule_label_invariant() -> None:
    counts, beta = _count_fixture(types=14)
    original = fit_bridge_estimators(counts, beta)
    permutation = np.random.default_rng(91).permutation(counts.shape[1])
    permuted = fit_bridge_estimators(
        counts[:, permutation], beta[np.ix_(permutation, permutation)]
    )
    for estimator in ("public_nine_atom", "full_revised"):
        np.testing.assert_allclose(
            original[estimator].values,
            permuted[estimator].values,
            atol=2e-8,
            rtol=0,
        )


def test_full_block_distinguishes_independent_from_cross_coupled_parts() -> None:
    independent = _var(np.diag([0.65, 0.50]), seed=7)
    cross = _var(np.asarray([[0.0, 0.75], [0.75, 0.0]]), seed=8)
    first = np.asarray([0], dtype=np.int64)
    second = np.asarray([1], dtype=np.int64)
    _, independent_components, _ = full_block_local_scores(
        independent, first, second
    )
    _, cross_components, _ = full_block_local_scores(cross, first, second)
    assert abs(independent_components["global_revised"]) < 0.02
    assert cross_components["global_revised"] > 0.20


def test_future_changes_cannot_alter_prefix_fitted_inputs() -> None:
    counts, beta = _count_fixture(observations=200)
    boundary = 50
    changed = counts.copy()
    changed[boundary:] = np.random.default_rng(999).poisson(
        5.0, size=changed[boundary:].shape
    )
    first = fit_bridge_estimators(counts[:boundary], beta)
    second = fit_bridge_estimators(changed[:boundary], beta)
    for estimator in ESTIMATOR_ORDER:
        np.testing.assert_array_equal(first[estimator].values, second[estimator].values)
        np.testing.assert_array_equal(
            first[estimator].partition_a, second[estimator].partition_a
        )
        np.testing.assert_array_equal(
            first[estimator].partition_b, second[estimator].partition_b
        )


def test_scientific_pilot_configuration_is_fixed_and_intervention_free() -> None:
    config = frozen_pilot_config()
    config.validate(require_frozen_pilot=True)
    assert config.seeds == PILOT_SEEDS
    assert len(config.seeds) == 12
    assert config.early_fraction == 0.25
    assert config.grid_points == 128
    assert config.replicator.similarity_threshold == 0.95
    assert not hasattr(config, "intervention")
    assert replace(config, early_fraction=0.30) != frozen_pilot_config()
