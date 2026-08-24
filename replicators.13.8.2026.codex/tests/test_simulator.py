import numpy as np

from plastic_heredity.config import CANDIDATES, ExperimentConfig, GardConfig
from plastic_heredity.seeds import derive_seed
from plastic_heredity.simulator import (
    Snapshot,
    advance_fission,
    cosine_similarity,
    generate_beta,
    generate_initial_composition,
    simulate_future_absorbing,
)


def test_cosine_similarity_boundaries():
    assert cosine_similarity(np.array([1, 0]), np.array([1, 0])) == 1.0
    assert cosine_similarity(np.array([1, 0]), np.array([0, 1])) == 0.0
    assert cosine_similarity(np.array([0, 0]), np.array([1, 0])) == 0.0


def test_fission_is_deterministic_for_each_seed_and_contract():
    config = GardConfig()
    beta = generate_beta(config, np.random.default_rng(10))
    initial = generate_initial_composition(config, np.random.default_rng(11))
    for contract in CANDIDATES.values():
        left = advance_fission(
            initial, beta, config, contract, np.random.default_rng(999)
        )
        right = advance_fission(
            initial, beta, config, contract, np.random.default_rng(999)
        )
        np.testing.assert_array_equal(left.parent, right.parent)
        np.testing.assert_array_equal(left.daughter, right.daughter)
        assert left.h == right.h
        assert left.parent.sum() == config.n_max
        assert left.daughter.sum() > 0


def test_seed_domains_are_stable_and_separate():
    master = "abc"
    assert derive_seed(master, "beta", 1) == derive_seed(master, "beta", 1)
    assert derive_seed(master, "beta", 1) != derive_seed(master, "future", 1)


def test_absorbing_future_reports_completion():
    config = GardConfig()
    beta = generate_beta(config, np.random.default_rng(15))
    initial = generate_initial_composition(config, np.random.default_rng(16))
    snapshot = Snapshot(initial, 0, (), ())
    records, completed = simulate_future_absorbing(
        snapshot,
        beta,
        config,
        CANDIDATES["02"],
        2,
        np.random.default_rng(17),
    )
    assert completed
    assert len(records) == 2


def test_scaled5_profile_multiplies_matrix_inferential_units():
    baseline = ExperimentConfig()
    scaled = ExperimentConfig.scaled5()
    assert scaled.development.matrices == 5 * baseline.development.matrices
    assert scaled.confirmation.matrices == 5 * baseline.confirmation.matrices
    assert scaled.development.branches_per_state == baseline.development.branches_per_state
    assert scaled.confirmation.branches_per_state == baseline.confirmation.branches_per_state
