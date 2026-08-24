from __future__ import annotations

from dataclasses import replace
import os

import numpy as np
import pytest

from plastic_heredity.config import CANDIDATES, GardConfig
from plastic_heredity.simulator import generate_beta, generate_initial_composition
from reviewer_lineage_identity_response.followup_dynamic_regime_cleanroom import run_campaign
from reviewer_lineage_identity_response.followup_dynamic_regime_cleanroom.regime_core import (
    _coupled_poisson,
    _coupled_sample,
    advance_coupled_fission,
    composition_probabilities,
    mean_field_flow,
    one_molecule_substitution,
    relax_mean_field,
    scaled_beta,
    scaled_config,
    simulate_twins,
    tangent_stability_margin,
    total_variation,
)


def _fixture() -> tuple[GardConfig, np.ndarray, np.ndarray]:
    config = GardConfig(max_growth_steps=1_000)
    rng = np.random.default_rng(20260822)
    return config, generate_beta(config, rng), generate_initial_composition(config, rng)


def test_substitution_preserves_mass_and_changes_exactly_two_bins() -> None:
    _, _, state = _fixture()
    changed = one_molecule_substitution(state, np.random.default_rng(7))
    assert changed.sum() == state.sum()
    assert np.abs(changed - state).sum() == 2
    assert np.all(changed >= 0)


def test_total_variation_is_symmetric_bounded_and_zero_on_identity() -> None:
    _, _, state = _fixture()
    changed = one_molecule_substitution(state, np.random.default_rng(8))
    assert total_variation(state, state) == 0.0
    assert total_variation(state, changed) == pytest.approx(total_variation(changed, state))
    assert 0.0 < total_variation(state, changed) <= 1.0


def test_coupled_poisson_is_exact_on_equal_rates_and_reproducible() -> None:
    rates = np.linspace(0.0, 4.0, 100)
    left, right = _coupled_poisson(rates, rates, np.random.default_rng(10))
    replay_left, replay_right = _coupled_poisson(rates, rates, np.random.default_rng(10))
    assert np.array_equal(left, right)
    assert np.array_equal(left, replay_left)
    assert np.array_equal(right, replay_right)


def test_token_sampling_is_identical_for_identical_multisets() -> None:
    counts = np.asarray([3, 0, 8, 2, 7], dtype=np.int64)
    left, right = _coupled_sample(counts, counts, 9, 9, np.random.default_rng(11))
    assert np.array_equal(left, right)
    assert left.sum() == 9
    assert np.all(left <= counts)


@pytest.mark.parametrize("candidate", ["02", "03"])
def test_identical_twins_are_pathwise_identical(candidate: str) -> None:
    config, beta, state = _fixture()
    result = simulate_twins(
        state, state, beta, config, CANDIDATES[candidate], seed=51, horizon=6
    )
    assert result.identical_path
    assert result.left_digest == result.right_digest
    assert np.array_equal(result.damage_tv, np.zeros(7))


@pytest.mark.parametrize("candidate", ["02", "03"])
def test_coupled_fission_is_seed_replayable_and_nonnegative(candidate: str) -> None:
    config, beta, state = _fixture()
    left = advance_coupled_fission(
        state, state, beta, config, CANDIDATES[candidate], np.random.default_rng(52)
    )
    right = advance_coupled_fission(
        state, state, beta, config, CANDIDATES[candidate], np.random.default_rng(52)
    )
    assert np.array_equal(left[0].daughter, left[1].daughter)
    assert np.array_equal(left[0].daughter, right[0].daughter)
    assert np.all(left[0].daughter >= 0)


def test_scaling_leaves_base_objects_unchanged() -> None:
    config, beta, _ = _fixture()
    beta_copy = beta.copy()
    altered_beta = scaled_beta(beta, 2.0)
    altered_config = scaled_config(config, 2.0)
    assert np.array_equal(beta, beta_copy)
    assert np.array_equal(altered_beta, 2.0 * beta)
    assert config.k_leave == 1e-4
    assert altered_config.k_leave == 2e-4


def test_mean_field_solver_returns_normalized_low_residual_form() -> None:
    config, beta, state = _fixture()
    form, iterations, residual = relax_mean_field(state, beta, config.k_join, config.k_leave)
    assert np.isclose(form.sum(), 1.0)
    assert np.all(form >= 0.0)
    assert iterations <= 10_000
    assert residual < 1e-8
    assert np.max(np.abs(mean_field_flow(form, beta, config.k_join, config.k_leave))) < 1e-8
    assert np.isfinite(tangent_stability_margin(form, beta, config.k_join, config.k_leave))


def test_grid_contains_original_and_has_21_unique_cells() -> None:
    assert len(run_campaign.GRID) == 21
    assert len({row[0] for row in run_campaign.GRID}) == 21
    assert run_campaign.CURRENT_GRID_ID in {row[0] for row in run_campaign.GRID}


def test_seed_domains_and_fresh_cohorts_are_distinct() -> None:
    development = run_campaign._matrix_beta("development", 0)
    confirmation = run_campaign._matrix_beta("confirmation", 0)
    assert not np.array_equal(development, confirmation)


def test_clean_room_firewall_and_claim_boundary() -> None:
    audit = run_campaign._firewall_audit()
    assert audit["passed"]
    protocol = run_campaign._protocol_payload()
    assert protocol["reporting_boundary"]["not_current_preprint_evidence"]
    assert protocol["reporting_boundary"]["wagner_code_used"] is False


def test_atomic_checkpoint_roundtrip() -> None:
    scratch = run_campaign.ARTIFACT_ROOT / f"unit_test_scratch_{os.getpid()}"
    scratch.mkdir(parents=True, exist_ok=False)
    path = scratch / "test.npz"
    try:
        run_campaign._atomic_npz(path, values=np.arange(7))
        assert np.array_equal(run_campaign._load_npz(path)["values"], np.arange(7))
    finally:
        if path.is_file():
            path.unlink()
        scratch.rmdir()


def test_walltime_tiers_and_restart_guard_are_nested() -> None:
    assert run_campaign.PROJECTION_BUDGET_SECONDS < run_campaign.SOFT_LIMIT_SECONDS < run_campaign.HARD_LIMIT_SECONDS
    assert run_campaign.TIERS["A"]["confirmation_matrices"] > run_campaign.TIERS["B"]["confirmation_matrices"] > run_campaign.TIERS["C"]["confirmation_matrices"]
    source = run_campaign.Path(run_campaign.__file__).read_text(encoding="utf-8")
    assert "active_cumulative_at_start" in source
    assert "recovered_interrupted_run_at_epoch" in source


def test_elapsed_reader_does_not_recover_or_rewrite_active_ledger(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    ledger_path = tmp_path / "runtime_ledger.json"
    monkeypatch.setattr(run_campaign, "LEDGER_PATH", ledger_path)
    active = run_campaign._initial_ledger()
    active["active_started_epoch"] = run_campaign.time.time() - 2.0
    active["active_cumulative_at_start"] = 7.0
    ledger_path.write_text(run_campaign.json.dumps(active), encoding="utf-8")
    before = ledger_path.read_bytes()
    assert run_campaign._ledger_elapsed() >= 9.0
    assert ledger_path.read_bytes() == before
