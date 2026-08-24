import numpy as np

from plastic_heredity.config import ExperimentConfig, GardConfig
from plastic_heredity.experiment import StateCase
from plastic_heredity.features import state_graph_features
from plastic_heredity.mechanistic_v2_features import (
    BETA_DESCRIPTOR_NAMES,
    BETA_ONLY_SPECS,
    FEATURE_NAMES,
    H10_SPECS,
    INTERACTION_INDICES,
    INTERACTION_SPECS,
    LEGACY_BETA_ONLY_SPECS,
    STATE_ONLY_INDICES,
    STATE_ONLY_SPECS,
    comprehensive_beta_features,
    extract_mechanistic_v2_features,
)
from plastic_heredity.simulator import Snapshot


def test_provenance_selects_semantic_blocks_and_covers_every_value():
    assert len(H10_SPECS) == len(FEATURE_NAMES["h10"]) == 10
    assert len(STATE_ONLY_SPECS) == len(FEATURE_NAMES["state"])
    assert len(INTERACTION_SPECS) == len(FEATURE_NAMES["interaction"])
    assert len(BETA_ONLY_SPECS) == len(FEATURE_NAMES["beta"])
    assert all(
        spec.provenance.depends_on_state
        and sum(spec.provenance.to_dict().values()) == 1
        for spec in STATE_ONLY_SPECS
    )
    assert all(
        spec.provenance.depends_on_state
        and spec.provenance.depends_on_beta
        and sum(spec.provenance.to_dict().values()) == 2
        for spec in INTERACTION_SPECS
    )
    assert all(
        spec.provenance.depends_on_beta
        and sum(spec.provenance.to_dict().values()) == 1
        for spec in BETA_ONLY_SPECS
    )
    assert len(LEGACY_BETA_ONLY_SPECS) > 31


def test_beta_panel_contains_required_threshold_free_families_and_full_spectrum():
    required = {
        "beta_log_row_strength__q05",
        "beta_log_row_strength__median",
        "beta_log_row_strength__q95",
        "beta_log_column_strength__q05",
        "beta_log_column_strength__median",
        "beta_log_column_strength__q95",
        "beta_log_strength__row_column_correlation",
        "beta_log_entries__reciprocity_correlation",
        "beta_matrix__normalized_asymmetry",
        "beta_singular__stable_rank",
        "beta_singular__normalized_spectral_entropy",
    }
    assert required <= set(BETA_DESCRIPTOR_NAMES)
    singular = {
        name
        for name in BETA_DESCRIPTOR_NAMES
        if name.removeprefix("beta_singular__normalized_").isdigit()
    }
    assert len(singular) == 100
    assert not any("threshold" in name for name in BETA_DESCRIPTOR_NAMES)


def test_comprehensive_beta_panel_is_deterministic_finite_and_relabel_invariant():
    rng = np.random.default_rng(701)
    config = GardConfig()
    beta = np.exp(-4.0 + 4.0 * rng.standard_normal((100, 100)))
    permutation = rng.permutation(100)
    first = comprehensive_beta_features(beta, config)
    second = comprehensive_beta_features(beta, config)
    relabeled = comprehensive_beta_features(
        beta[np.ix_(permutation, permutation)], config
    )
    assert first.shape == (len(BETA_ONLY_SPECS),)
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first, relabeled, rtol=2e-12, atol=2e-12)


def test_state_and_interaction_blocks_are_mass_free_by_construction():
    rng = np.random.default_rng(702)
    config = GardConfig()
    beta = np.exp(-4.0 + 4.0 * rng.standard_normal((100, 100)))
    composition = rng.multinomial(40, np.full(100, 0.01)).astype(np.int64)
    doubled = composition * 2
    first = state_graph_features(composition, beta, config)
    second = state_graph_features(doubled, beta, config)
    np.testing.assert_allclose(
        first[list(STATE_ONLY_INDICES)], second[list(STATE_ONLY_INDICES)], atol=1e-14
    )
    np.testing.assert_allclose(
        first[list(INTERACTION_INDICES)],
        second[list(INTERACTION_INDICES)],
        atol=1e-14,
    )


def test_feature_blocks_respond_only_to_their_declared_input_families():
    rng = np.random.default_rng(703)
    experiment = ExperimentConfig()
    beta = np.exp(-4.0 + 4.0 * rng.standard_normal((100, 100)))
    changed_beta = beta.copy()
    changed_beta[0, 0] *= 2.0
    composition = np.r_[
        np.full(20, 2, dtype=np.int64), np.zeros(80, dtype=np.int64)
    ]
    changed_state = composition.copy()
    changed_state[0] -= 1
    changed_state[20] += 1

    def snapshot(
        values=composition,
        generation=20,
        inheritance=(True, False, True, True),
        boundary_h=(0.95, 0.60, 0.94, 0.96),
        previous=120,
        cumulative=2_500,
    ):
        return Snapshot(
            values.copy(),
            generation,
            inheritance,
            boundary_h,
            previous_growth_steps=previous,
            cumulative_growth_steps=cumulative,
        )

    snapshots = (
        snapshot(),
        snapshot(values=changed_state),
        snapshot(values=composition * 2),
        snapshot(
            inheritance=(True, True, False, False),
            boundary_h=(0.95, 0.94, 0.60, 0.55),
        ),
        snapshot(previous=250, cumulative=8_000),
        snapshot(generation=35),
        snapshot(),
    )
    betas = (beta, beta, beta, beta, beta, beta, changed_beta)
    matrix_ids = (0, 0, 0, 0, 0, 0, 1)
    cases = [
        StateCase(
            state_id=f"semantic-{index}",
            cohort="SEMANTIC",
            candidate="02",
            matrix_id=matrix_id,
            landmark=item.generation,
            beta=case_beta,
            snapshot=item,
        )
        for index, (item, case_beta, matrix_id) in enumerate(
            zip(snapshots, betas, matrix_ids)
        )
    ]
    raw = extract_mechanistic_v2_features(cases, experiment)
    base, state, mass, history, clock, phase, beta_index = range(7)

    assert not np.allclose(raw.state[base], raw.state[state])
    np.testing.assert_allclose(raw.state[base], raw.state[mass], atol=1e-14)
    np.testing.assert_array_equal(raw.state[base], raw.state[history])
    np.testing.assert_array_equal(raw.state[base], raw.state[clock])
    np.testing.assert_array_equal(raw.state[base], raw.state[phase])
    np.testing.assert_array_equal(raw.state[base], raw.state[beta_index])

    for index in (state, mass, history, clock, phase):
        np.testing.assert_array_equal(raw.beta[base], raw.beta[index])
    assert not np.allclose(raw.beta[base], raw.beta[beta_index])

    assert not np.allclose(raw.interaction[base], raw.interaction[state])
    np.testing.assert_allclose(
        raw.interaction[base], raw.interaction[mass], atol=1e-14
    )
    np.testing.assert_array_equal(raw.interaction[base], raw.interaction[history])
    np.testing.assert_array_equal(raw.interaction[base], raw.interaction[clock])
    np.testing.assert_array_equal(raw.interaction[base], raw.interaction[phase])
    assert not np.allclose(raw.interaction[base], raw.interaction[beta_index])

    np.testing.assert_array_equal(raw.h10[base], raw.h10[state])
    np.testing.assert_array_equal(raw.h10[base], raw.h10[beta_index])
    for index in (mass, history, clock, phase):
        assert not np.array_equal(raw.h10[base], raw.h10[index])
