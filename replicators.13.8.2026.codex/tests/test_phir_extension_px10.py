from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

from plastic_heredity import phir_extension_px9 as px9
from plastic_heredity.config import CANDIDATES, GardConfig
from plastic_heredity.phir_extension_px10 import (
    ALL_LAGS,
    ARMS,
    ATOM_ARMS,
    ATOM_GROUPS,
    ATOM_SUPPORT_BRANCHES,
    BRANCHES,
    CANONICAL_ATOM_SIGNATURES,
    INFORMATION_EQUIVALENCE_MARGIN_BITS,
    MATRICES,
    PRIMARY_LAGS,
    QUANTILE_ARMS,
    RANDOM_PARTITIONS,
    SECONDARY_LAGS,
    AtomCasePayload,
    GrainModel,
    _atom_score,
    _binary_channel_capacity,
    _canonical_samples,
    _channel_matrix_values,
    _derange_source_arm_labels,
    _derangement,
    _exact_discrete_atoms,
    _gaussian_fixture,
    _kernel_metrics,
    _lag_pairs,
    _macro_from_micro,
    _matrix_random_partitions,
    _mutual_information_discrete,
    _px9_seed_context,
    _sample_clr_surrogate,
    _sample_wms,
    _signature_error,
    _uniform_binary_channel_information,
    fit_grain_model,
    load_grain_model,
    protocol,
    save_grain_model,
    scientific_spec,
    smoke_spec,
)
from plastic_heredity.phir_instruments import ATOM_NAMES
from plastic_heredity.phir_rescue_instruments import beta_physical_partition
from plastic_heredity.simulator import Snapshot


def _case_and_payload() -> tuple[px9.ResilienceCase, AtomCasePayload]:
    rng = np.random.default_rng(1010)
    beta = np.exp(rng.normal(-4.0, 1.0, size=(100, 100)))
    composition = np.zeros(100, dtype=np.int64)
    composition[:40] = 1
    case = px9.ResilienceCase(
        "PX10-test",
        "02",
        0,
        20,
        beta,
        Snapshot(composition, 20, (True,) * 20, (0.95,) * 20),
        np.tile(composition, (32, 1)).astype(np.int16),
    )
    blocks: list[px9.PairBlock] = []
    for branch in range(6):
        future = np.tile(composition, (4, 1)).astype(np.int16)
        for depth in range(4):
            future[depth, (branch * 7 + depth) % 100] += depth + 1
        blocks.append(
            px9.PairBlock(
                future[:-1],
                future[1:],
                future[:-1],
                future,
            )
        )
    payload = AtomCasePayload(
        case,
        {"Q100": composition},
        {"Q100": tuple(blocks)},
    )
    return case, payload


def _channel_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for matrix_id in range(4):
        for state in range(2):
            state_id = f"m{matrix_id}-s{state}"
            for arm_index, arm in enumerate(QUANTILE_ARMS):
                outcome = int(arm_index >= 3)
                for branch in range(32):
                    rows.append(
                        {
                            "candidate": "02",
                            "matrix_id": matrix_id,
                            "state_id": state_id,
                            "arm": arm,
                            "branch": branch,
                            "half": "A" if branch < 16 else "B",
                            "primary": outcome,
                        }
                    )
    return pd.DataFrame(rows)


def test_px10_protocol_freezes_codex_only_shared_48_matrix_cohort() -> None:
    value = protocol()
    assert MATRICES == 48
    assert BRANCHES == 256
    assert value["fable_artifacts_used"] is False
    assert value["cohort"]["matrices"] == 48
    assert value["cohort"]["minimum_eligible_matrices"] == 40
    assert value["operational"]["detached_science"] is True
    assert value["operational"]["automatic_continuation"] is False


def test_px10_registered_arms_lags_and_atom_groups_are_exact() -> None:
    assert ARMS == (*QUANTILE_ARMS, "RANDOM", "NOOP")
    assert PRIMARY_LAGS == (1, 2, 3, 4)
    assert SECONDARY_LAGS == (8,)
    assert ALL_LAGS == (1, 2, 3, 4, 8)
    assert ATOM_ARMS == ("Q00", "Q100")
    assert ATOM_SUPPORT_BRANCHES == 64
    assert ATOM_GROUPS == ("causation", "synergy_persistence", "emergence")
    assert INFORMATION_EQUIVALENCE_MARGIN_BITS == 0.0005


def test_px9_seed_context_is_fresh_and_restored() -> None:
    old_label = px9.LABEL
    old_domains = px9.SEED_DOMAINS
    old_matrices = px9.MATRICES
    with _px9_seed_context():
        assert px9.LABEL != old_label
        assert px9.SEED_DOMAINS != old_domains
        assert px9.MATRICES == 48
    assert px9.LABEL == old_label
    assert px9.SEED_DOMAINS is old_domains
    assert px9.MATRICES == old_matrices


def test_future_seed_contract_remains_arm_free() -> None:
    assert "arm" not in inspect.signature(px9._future_seed).parameters
    assert protocol()["randomness"]["arm_in_future_seed"] is False


def test_exact_discrete_reference_closes_and_recovers_registered_signatures() -> None:
    for name in CANONICAL_ATOM_SIGNATURES:
        past, future = _canonical_samples(name, 1024, np.random.default_rng(17))
        atoms = _exact_discrete_atoms(past, future)
        assert np.isclose(
            sum(atoms.values()),
            _mutual_information_discrete(past, future),
            atol=1e-10,
            rtol=0,
        )
        assert _signature_error(name, atoms) <= 1e-10


def test_exact_independent_fixture_has_zero_atoms() -> None:
    past, future = _canonical_samples("independent", 1024, np.random.default_rng(18))
    atoms = _exact_discrete_atoms(past, future)
    assert max(abs(value) for value in atoms.values()) <= 1e-10


def test_analytic_gaussian_fixture_has_positive_recoverable_score() -> None:
    covariance, truth, partition = _gaussian_fixture()
    estimate = _sample_wms(
        covariance, partition, 4096, np.random.default_rng(19)
    )
    assert truth > 0
    assert estimate > 0
    assert abs(estimate - truth) / truth < 0.15


def test_99d_clr_surrogate_is_finite() -> None:
    paired, shuffled = _sample_clr_surrogate(128, np.random.default_rng(20))
    assert np.isfinite(paired)
    assert np.isfinite(shuffled)


def test_lag_derangement_preserves_marginals_without_crossing_branches() -> None:
    _, payload = _case_and_payload()
    past, future, _ = _lag_pairs(payload, "Q100", range(6), 2)
    shifted_past, shifted_future, self_pairs = _lag_pairs(
        payload, "Q100", range(6), 2, 1
    )
    assert self_pairs == 0
    assert np.array_equal(past, shifted_past)
    assert sorted(row.tobytes() for row in future) == sorted(
        row.tobytes() for row in shifted_future
    )
    assert any(
        not np.array_equal(left, right)
        for left, right in zip(future, shifted_future, strict=True)
    )


def test_lag_control_allows_one_surviving_branch_without_false_failure() -> None:
    _, payload = _case_and_payload()
    past, future, self_pairs = _lag_pairs(payload, "Q100", (0,), 3, 1)
    assert self_pairs == 0
    assert past.shape == future.shape
    assert len(past) > 0


def test_random_partitions_are_deterministic_complete_and_size_matched() -> None:
    case, _ = _case_and_payload()
    beta_first, beta_second = beta_physical_partition(case.beta)
    first = _matrix_random_partitions(case.beta, 0, smoke_spec())
    second = _matrix_random_partitions(case.beta, 0, smoke_spec())
    assert len(first) == RANDOM_PARTITIONS
    assert len({tuple(left.tolist()) for left, _ in first}) == RANDOM_PARTITIONS
    for (left, right), (again_left, again_right) in zip(first, second, strict=True):
        assert len(left) == len(beta_first)
        assert len(right) == len(beta_second)
        assert set(left).isdisjoint(right)
        assert set(left) | set(right) == set(range(GardConfig().n_types))
        assert np.array_equal(left, again_left)
        assert np.array_equal(right, again_right)


def test_atom_score_emits_all_atoms_and_frozen_groups() -> None:
    case, _ = _case_and_payload()
    rng = np.random.default_rng(21)
    past = rng.poisson(2.0, size=(256, 100)).astype(np.int16)
    future = rng.poisson(2.0, size=(256, 100)).astype(np.int16)
    first, second = beta_physical_partition(case.beta)
    score = _atom_score(past, future, first, second)
    assert score["partition_digest"] != "invalid"
    assert all(f"atom_{name}" in score for name in ATOM_NAMES)
    assert np.isclose(
        score["emergence"],
        score["causation"] + score["synergy_persistence"],
    )


def test_derangement_is_a_permutation_without_fixed_points() -> None:
    value = _derangement(6, np.random.default_rng(22))
    assert sorted(value.tolist()) == list(range(6))
    assert np.all(value != np.arange(6))


def test_source_label_null_preserves_counts_and_breaks_stable_mapping() -> None:
    source = _channel_frame().query("half == 'A'")
    shuffled = _derange_source_arm_labels(
        source, "02", "A", 0, QUANTILE_ARMS
    )
    original_counts = source.groupby(["state_id", "arm"]).size().sort_index()
    shuffled_counts = shuffled.groupby(["state_id", "arm"]).size().sort_index()
    assert original_counts.equals(shuffled_counts)
    assert not source["arm"].equals(shuffled["arm"])
    assert np.array_equal(source["primary"], shuffled["primary"])


def test_held_out_channel_detects_known_action_information() -> None:
    frame = _channel_frame()
    real, probabilities = _channel_matrix_values(
        frame, "02", "A", "B", QUANTILE_ARMS
    )
    shuffled, _ = _channel_matrix_values(
        frame, "02", "A", "B", QUANTILE_ARMS, 0
    )
    assert len(real) == 4
    assert real.mean() > 0.5
    assert real.mean() > shuffled.mean()
    per_state = probabilities.groupby("state_id")["p_arm"]
    assert per_state.apply(_uniform_binary_channel_information).mean() > 0.5
    assert per_state.apply(_binary_channel_capacity).mean() > 0.5


def test_binary_channel_information_controls_are_exact() -> None:
    assert abs(_binary_channel_capacity([0.4, 0.4, 0.4])) < 1e-10
    assert abs(_uniform_binary_channel_information([0.4, 0.4, 0.4])) < 1e-10
    assert np.isclose(_binary_channel_capacity([0.0, 1.0]), 1.0, atol=1e-8)
    assert np.isclose(
        _uniform_binary_channel_information([0.0, 1.0]), 1.0, atol=1e-8
    )


def test_macro_projection_is_nested_and_stochastic() -> None:
    conditional = np.eye(16, dtype=float)
    macro, active = _macro_from_micro(conditional, range(16))
    assert active == [0, 1, 2, 3]
    assert np.allclose(macro, np.eye(4))
    metrics = _kernel_metrics(macro, active)
    assert np.isclose(metrics["effective_information_bits"], 2.0)
    assert np.isclose(metrics["effectiveness"], 1.0)


def test_grain_model_is_nested_and_serialization_stable(tmp_path) -> None:
    model = fit_grain_model()
    assert isinstance(model, GrainModel)
    path = tmp_path / "grain.npz"
    save_grain_model(model, path)
    restored = load_grain_model(path)
    for candidate in CANDIDATES:
        assert model.components[candidate].shape == (4, 195)
        assert np.array_equal(model.components[candidate], restored.components[candidate])
        features = model.means[candidate][None, :] + np.vstack(
            (np.zeros(195), model.scales[candidate])
        )
        micro, macro = restored.classify(candidate, features)
        assert np.array_equal(macro, micro // 4)
        assert np.all((micro >= 0) & (micro < 16))


def test_scientific_spec_keeps_matrix_as_inference_unit() -> None:
    value = protocol(scientific_spec())
    assert value["inference"]["unit"] == "whole catalytic matrix"
    assert value["cohort"]["halves"]["A"] == list(range(128))
    assert value["cohort"]["halves"]["B"] == list(range(128, 256))
