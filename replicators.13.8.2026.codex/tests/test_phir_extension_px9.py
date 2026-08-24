from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

from plastic_heredity.config import CANDIDATES, GardConfig
from plastic_heredity.phir_extension_px9 import (
    ARMS,
    BRANCHES,
    HALVES,
    QUANTILES,
    RANDOM_PARTITIONS,
    SUPPORT_LEVELS,
    PairBlock,
    PX9Spec,
    ResilienceCase,
    _concatenate_pairs,
    _derived_score_frame,
    _fixed_partition_score,
    _future_seed,
    _random_partitions,
    _selection_seed,
    _simulate_branch,
    _statewise_spearman,
    _temporal_derangement,
    protocol,
    scientific_spec,
    smoke_spec,
)
from plastic_heredity.phir_extension_px7 import _score_pairs
from plastic_heredity.phir_instruments import advance_fission_traced
from plastic_heredity.phir_rescue_instruments import beta_physical_partition
from plastic_heredity.simulator import Snapshot, generate_beta


def _fixture() -> tuple[ResilienceCase, np.ndarray]:
    rng = np.random.default_rng(909)
    beta = np.exp(rng.normal(-4.0, 1.0, size=(100, 100)))
    counts = rng.poisson(2.0, size=(1100, 100)).astype(np.int16)
    counts[counts.sum(axis=1) == 0, 0] = 1
    composition = np.zeros(100, dtype=np.int64)
    composition[:40] = 1
    case = ResilienceCase(
        "PX9-test",
        "02",
        0,
        20,
        beta,
        Snapshot(composition, 20, (True,) * 20, (0.95,) * 20),
        counts[-512:],
    )
    return case, counts


def test_px9_protocol_is_a_manual_24_matrix_pilot() -> None:
    value = protocol()
    assert value["cohort"]["matrices"] == 24
    assert value["cohort"]["pilot_only"] is True
    assert value["cohort"]["automatic_48_matrix_continuation"] is False
    assert value["public_revised_can_win"] is False


def test_px9_dose_arms_and_support_are_exact() -> None:
    assert ARMS == (
        "Q00",
        "Q20",
        "Q40",
        "Q60",
        "Q80",
        "Q100",
        "RANDOM",
        "NOOP",
    )
    assert QUANTILES == (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    assert BRANCHES == 256
    assert SUPPORT_LEVELS == (64, 128)
    assert set(HALVES["A"]).isdisjoint(HALVES["B"])
    assert set(HALVES["A"]).union(HALVES["B"]) == set(range(BRANCHES))
    assert protocol(scientific_spec())["target"]["primary_support"] == 128


def test_future_stream_is_arm_free_and_action_separate() -> None:
    case, _ = _fixture()
    spec = smoke_spec()
    assert "arm" not in inspect.signature(_future_seed).parameters
    assert _future_seed(spec, case, 0) != _selection_seed(spec, case)


def test_temporal_derangement_preserves_depth_marginals_without_self_pairs() -> None:
    _, counts = _fixture()
    blocks = [
        PairBlock(
            counts[index * 4 : index * 4 + 3],
            counts[index * 4 + 1 : index * 4 + 4],
            counts[index * 4 : index * 4 + 3],
            counts[index * 4 + 1 : index * 4 + 4],
        )
        for index in range(8)
    ]
    original_past, original_future = _concatenate_pairs(blocks, "generational")
    shuffled_past, shuffled_future, self_pairs = _temporal_derangement(blocks, 3)
    assert self_pairs == 0
    assert shuffled_past.shape == shuffled_future.shape == original_past.shape
    assert sorted(row.tobytes() for row in shuffled_past) == sorted(
        row.tobytes() for row in original_past
    )
    assert sorted(row.tobytes() for row in shuffled_future) == sorted(
        row.tobytes() for row in original_future
    )
    assert any(
        not np.array_equal(left, right)
        for left, right in zip(shuffled_future, original_future, strict=True)
    )


def test_random_partitions_are_deterministic_complete_and_size_matched() -> None:
    case, _ = _fixture()
    first, second = beta_physical_partition(case.beta)
    left = _random_partitions(case, smoke_spec())
    right = _random_partitions(case, smoke_spec())
    assert len(left) == RANDOM_PARTITIONS
    assert len({tuple(a.tolist()) for a, _ in left}) == RANDOM_PARTITIONS
    for (a, b), (again_a, again_b) in zip(left, right, strict=True):
        assert len(a) == len(first)
        assert len(b) == len(second)
        assert not set(a).intersection(b)
        assert set(a).union(b) == set(range(100))
        assert np.array_equal(a, again_a)
        assert np.array_equal(b, again_b)


def test_fixed_partition_scorer_matches_px8_target_exactly() -> None:
    case, counts = _fixture()
    first, second = beta_physical_partition(case.beta)
    direct = _fixed_partition_score(counts[:1024], counts[1:1025], first, second)
    frozen = _score_pairs(counts[:1024], counts[1:1025], "beta", case)
    assert np.isclose(direct["value"], frozen["typeset"], atol=1e-10, rtol=0)
    assert np.isclose(direct["whole_mi"], frozen["whole_mi"], atol=1e-10, rtol=0)
    assert direct["active_dimensions"] == frozen["active_dimensions"]


def test_fixed_partition_score_is_label_permutation_invariant() -> None:
    case, counts = _fixture()
    first, second = beta_physical_partition(case.beta)
    original = _fixed_partition_score(counts[:900], counts[1:901], first, second)
    rng = np.random.default_rng(41)
    permutation = np.concatenate((rng.permutation(99), np.asarray([99])))
    inverse = np.argsort(permutation)
    permuted_counts = counts[:, permutation]
    permuted_first = inverse[first]
    permuted_second = inverse[second]
    permuted = _fixed_partition_score(
        permuted_counts[:900],
        permuted_counts[1:901],
        permuted_first,
        permuted_second,
    )
    assert np.isclose(original["value"], permuted["value"], atol=1e-8, rtol=0)
    assert np.isclose(original["whole_mi"], permuted["whole_mi"], atol=1e-8, rtol=0)


def test_noop_branch_is_plain_simulator_and_run3_horizon_is_strict() -> None:
    config = GardConfig()
    beta = generate_beta(config, np.random.default_rng(81))
    composition = np.zeros(config.n_types, dtype=np.int64)
    composition[:40] = 1
    case = ResilienceCase(
        "PX9-noop",
        "02",
        0,
        20,
        beta,
        Snapshot(composition, 20, (True,) * 20, (0.95,) * 20),
        np.tile(composition, (32, 1)).astype(np.int16),
    )
    spec = PX9Spec("smoke", 1, (20,), 2, 1, 1, 8, 8)
    row, pairs = _simulate_branch(case, None, 0, spec)
    traced = advance_fission_traced(
        composition,
        beta,
        config,
        CANDIDATES["02"],
        np.random.default_rng(_future_seed(spec, case, 0)),
    )
    assert np.array_equal(pairs.generational_past[0], traced.record.parent)
    assert np.array_equal(pairs.generational_future[0], traced.record.daughter)
    assert row["primary"] == 0


def test_derived_scores_are_registered_differences() -> None:
    rows: list[dict[str, object]] = []
    keys = {
        "matrix_id": 0,
        "candidate": "02",
        "state_id": "s",
        "landmark": 20,
        "arm": "Q100",
        "source_half": "A",
        "support_branches": 128,
        "whole_mi": 5.0,
        "aa_mi": 2.0,
        "bb_mi": 1.0,
        "active_dimensions": 10,
        "part_a_dimensions": 4,
        "part_b_dimensions": 6,
        "transitions": 1024,
    }
    rows.append({**keys, "score_kind": "paired_beta", "control_id": -1, "value": 2.0})
    for index, value in enumerate((0.5, 0.7) * 4):
        rows.append(
            {**keys, "score_kind": "shuffled_beta", "control_id": index, "value": value}
        )
    for index, value in enumerate((1.0, 1.4) * 4):
        rows.append(
            {**keys, "score_kind": "random_partition", "control_id": index, "value": value}
        )
    for index in range(3):
        rows.append(
            {
                **keys,
                "score_kind": "cross_beta_partition",
                "control_id": index,
                "value": 1.1,
            }
        )
    derived = _derived_score_frame(pd.DataFrame(rows), 128).iloc[0]
    assert np.isclose(derived.temporal_value, 1.4)
    assert np.isclose(derived.topology_value, 0.8)
    assert np.isclose(derived.cross_beta_excess, 0.9)
    assert np.isclose(derived.within_sum_mi, 3.0)


def test_statewise_dose_spearman_keeps_matrix_as_unit() -> None:
    rows = []
    for matrix_id in range(3):
        for state in range(2):
            for arm, quantile in zip(
                ("Q00", "Q20", "Q40", "Q60", "Q80", "Q100"),
                QUANTILES,
                strict=True,
            ):
                rows.append(
                    {
                        "matrix_id": matrix_id,
                        "state_id": f"m{matrix_id}s{state}",
                        "arm": arm,
                        "quantile": quantile,
                        "outcome": quantile + 0.01 * state,
                    }
                )
    result = _statewise_spearman(
        pd.DataFrame(rows), "quantile", "outcome", require_quantiles=True
    )
    assert list(result.index) == [0, 1, 2]
    assert np.allclose(result.to_numpy(), 1.0)
