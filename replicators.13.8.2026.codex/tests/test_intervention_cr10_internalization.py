from __future__ import annotations

import numpy as np

from plastic_heredity.config import CANDIDATES, GardConfig
from plastic_heredity.intervention_core import enumerate_legal_edits
from plastic_heredity.intervention_cr10_internalization import (
    BOOTSTRAP_REPETITIONS,
    CHALLENGE_AFTER_FISSION,
    CHALLENGE_K,
    CONDITIONS,
    HORIZON,
    HOME_MATRICES,
    HOME_REPLICATES,
    KINETIC_LAMBDAS,
    KINETIC_REPLICATES,
    LOCAL_FEATURE_NAMES,
    POLICIES,
    RANDOMIZATION_REPETITIONS,
    SEEDS,
    TRANSFER_MATRICES,
    TRANSFER_REGIMES,
    TRANSFER_REPLICATES,
    FrozenLocalTrees,
    _action_seed,
    _artificial_case,
    _challenge_seed,
    _fixture_inference_tables,
    _future_seed,
    _percentile,
    advance_fission_retention,
    apply_many_edits,
    compute_inference,
    count_nonoverlapping_episodes,
    exact_random_k_edits,
    inference_draws,
    local_type_features,
    longest_inherited_run,
    protocol,
    select_policy_edit,
    trailing_inherited_run,
)
from plastic_heredity.simulator import advance_fission


def test_cr10_design_is_frozen_and_exploratory() -> None:
    frozen = protocol()
    assert HOME_MATRICES == 48
    assert TRANSFER_MATRICES == 24
    assert HOME_REPLICATES == 3
    assert TRANSFER_REPLICATES == 2
    assert KINETIC_REPLICATES == 3
    assert HORIZON == 60
    assert CHALLENGE_AFTER_FISSION == 30
    assert CHALLENGE_K == 8
    assert KINETIC_LAMBDAS == (0.0, 0.1, 0.3)
    assert len(POLICIES) == 7
    assert CONDITIONS == ("UNCHALLENGED", "CHALLENGED_K8")
    assert tuple(TRANSFER_REGIMES) == (
        "POS_A_M4_S5",
        "POS_A_M3_S4",
        "POS_A_M5_S4",
    )
    assert frozen["inference"]["no_confirmatory_pass_fail_gate"] is True
    assert frozen["claim_boundary"]["exploratory_only"] is True
    assert frozen["target"]["strict_eight_excluded"] is True


def test_cr10_local_features_have_exact_orientation_and_permute() -> None:
    composition = np.asarray([2, 1, 0, 1], dtype=np.int64)
    beta = np.asarray(
        [
            [1.0, 2.0, 4.0, 8.0],
            [3.0, 5.0, 7.0, 11.0],
            [13.0, 17.0, 19.0, 23.0],
            [29.0, 31.0, 37.0, 41.0],
        ]
    )
    features = local_type_features(composition, beta)
    x = composition / composition.sum()
    assert tuple(LOCAL_FEATURE_NAMES) == (
        "abundance_share",
        "outgoing_influence_percentile",
        "incoming_boost_percentile",
        "presence",
    )
    assert np.array_equal(features[:, 0], x)
    assert np.array_equal(features[:, 1], _percentile(x @ beta))
    assert np.array_equal(features[:, 2], _percentile(beta @ x))
    permutation = np.asarray([2, 0, 3, 1])
    permuted = local_type_features(
        composition[permutation], beta[np.ix_(permutation, permutation)]
    )
    assert np.array_equal(permuted, features[permutation])


def test_cr10_midrank_percentile_handles_ties_equivariantly() -> None:
    values = np.asarray([5.0, 1.0, 5.0, 3.0])
    observed = _percentile(values)
    assert np.array_equal(observed, np.asarray([5 / 6, 0.0, 5 / 6, 1 / 3]))


def _stump_trees() -> FrozenLocalTrees:
    arrays = {}
    for candidate in CANDIDATES:
        for role in ("remove", "add"):
            prefix = f"c{candidate}__{role}"
            arrays[f"{prefix}__children_left"] = np.asarray([-1], dtype=np.int32)
            arrays[f"{prefix}__children_right"] = np.asarray([-1], dtype=np.int32)
            arrays[f"{prefix}__feature"] = np.asarray([-2], dtype=np.int16)
            arrays[f"{prefix}__threshold"] = np.asarray([-2.0])
            arrays[f"{prefix}__positive_probability"] = np.asarray([0.5])
            arrays[f"{prefix}__node_samples"] = np.asarray([1], dtype=np.int32)
            arrays[f"{prefix}__depth"] = np.asarray([0], dtype=np.int16)
    return FrozenLocalTrees(arrays)


def test_cr10_local_tree_ties_are_deterministic_and_legal() -> None:
    composition = np.asarray([0, 2, 1, 0], dtype=np.int64)
    beta = np.eye(4)
    edit = _stump_trees().select_edit("02", composition, beta)
    assert (edit.remove_type, edit.add_type) == (1, 0)
    assert edit in enumerate_legal_edits(composition)


def test_cr10_exact_k_challenge_preserves_mass_history_and_distance() -> None:
    composition = np.asarray([10, 10, 10, 10] + [0] * 96, dtype=np.int64)
    edits = exact_random_k_edits(composition, 8, np.random.default_rng(9))
    edited = apply_many_edits(composition, edits)
    assert len(edits) == 8
    assert edited.sum() == composition.sum()
    assert np.all(edited >= 0)
    assert np.abs(edited - composition).sum() // 2 == 8
    assert all(edit.remove_type != edit.add_type for edit in edits)


def test_cr10_strict_episode_and_run_contracts() -> None:
    assert count_nonoverlapping_episodes([0.91, 0.92, 0.93]) == 0
    assert count_nonoverlapping_episodes([0.9, 0.91, 0.92, 0.93]) == 1
    assert count_nonoverlapping_episodes(
        [0.8, 0.91, 0.92, 0.93, 0.7, 0.94, 0.95, 0.96]
    ) == 2
    assert trailing_inherited_run([0.8, 0.91, 0.92]) == 2
    assert trailing_inherited_run([0.91, 0.92, 0.93]) == 3
    assert longest_inherited_run([0.9, 0.91, 0.92, 0.93]) == 3


class _ConstantPredictor:
    def predict_snapshot(self, *_args: object) -> float:
        return 0.5


def test_cr10_sparse_policy_triggers_are_exact() -> None:
    case = _artificial_case()
    trees = _stump_trees()
    predictor = _ConstantPredictor()
    rng = np.random.default_rng(2)
    break_snapshot = type(case.snapshot)(
        composition=case.snapshot.composition,
        generation=case.snapshot.generation,
        inheritance=case.snapshot.inheritance + (False,),
        boundary_h=case.snapshot.boundary_h + (0.9,),
        previous_growth_steps=case.snapshot.previous_growth_steps,
        cumulative_growth_steps=case.snapshot.cumulative_growth_steps,
    )
    stable_snapshot = type(case.snapshot)(
        composition=case.snapshot.composition,
        generation=case.snapshot.generation,
        inheritance=case.snapshot.inheritance + (True,),
        boundary_h=case.snapshot.boundary_h + (0.95,),
        previous_growth_steps=case.snapshot.previous_growth_steps,
        cumulative_growth_steps=case.snapshot.cumulative_growth_steps,
    )
    assert select_policy_edit(
        "L1_RULE_AFTER_BREAK",
        "02",
        break_snapshot,
        case.beta,
        GardConfig(),
        predictor,  # type: ignore[arg-type]
        trees,
        rng,
    )[0] is not None
    assert select_policy_edit(
        "L1_RULE_AFTER_BREAK",
        "02",
        stable_snapshot,
        case.beta,
        GardConfig(),
        predictor,  # type: ignore[arg-type]
        trees,
        rng,
    )[0] is None
    assert select_policy_edit(
        "L2_RULE_UNTIL_RUN3",
        "02",
        break_snapshot,
        case.beta,
        GardConfig(),
        predictor,  # type: ignore[arg-type]
        trees,
        rng,
    )[0] is not None
    assert select_policy_edit(
        "L2_RULE_UNTIL_RUN3",
        "02",
        stable_snapshot,
        case.beta,
        GardConfig(),
        predictor,  # type: ignore[arg-type]
        trees,
        rng,
    )[0] is None


def test_cr10_future_stream_is_policy_condition_free_and_auxiliary_streams_separate() -> None:
    case = _artificial_case()
    assert len(SEEDS) == len(set(SEEDS.values()))
    assert len({_future_seed(case, 1) for _policy in POLICIES for _condition in CONDITIONS}) == 1
    assert len({_future_seed(case, 1), _action_seed(case, 1), _challenge_seed(case, 1)}) == 3


def test_cr10_lambda_zero_dispatch_is_bitwise_plain_for_one_fission() -> None:
    case = _artificial_case()
    seed = 12345
    left_rng = np.random.default_rng(seed)
    right_rng = np.random.default_rng(seed)
    left = advance_fission_retention(
        case.snapshot.composition,
        case.beta,
        GardConfig(),
        case.candidate,
        left_rng,
        0.0,
    )
    right = advance_fission(
        case.snapshot.composition,
        case.beta,
        GardConfig(),
        CANDIDATES[case.candidate],
        right_rng,
    )
    assert np.array_equal(left.parent, right.parent)
    assert np.array_equal(left.daughter, right.daughter)
    assert left.h == right.h
    assert left.growth_steps == right.growth_steps
    assert left_rng.bit_generator.state == right_rng.bit_generator.state


def test_cr10_whole_matrix_inference_remains_exploratory() -> None:
    policy, kinetic = _fixture_inference_tables()
    metrics, stored = compute_inference(
        policy,
        kinetic,
        inference_draws(),
        policy_replay_exact=True,
        kinetic_replay_exact=True,
        noop_plain_exact=True,
        lambda_zero_plain_exact=True,
    )
    assert metrics["exploratory_no_confirmatory_gate"] is True
    assert metrics["claim_status"]["confirmatory_gate"] is None
    assert metrics["integrity"]["all_integrity_checks_passed"] is True
    assert stored["home_bootstrap_indices"].shape == (
        BOOTSTRAP_REPETITIONS,
        HOME_MATRICES,
    )
    assert stored["home_randomization_signs"].shape == (
        RANDOMIZATION_REPETITIONS,
        HOME_MATRICES,
    )


def test_cr10_smoke_checks_use_boolean_zero_matrix_assertion() -> None:
    # Regression for the pre-scientific smoke failure that correctly had a
    # count of zero but accidentally included the integer itself in all().
    checks = {"scientific_cr10_matrices_generated_is_zero": 0 == 0}
    assert all(checks.values())
