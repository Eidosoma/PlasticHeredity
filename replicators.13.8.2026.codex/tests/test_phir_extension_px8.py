from __future__ import annotations

import numpy as np
import pandas as pd

from plastic_heredity.config import CANDIDATES, GardConfig
from plastic_heredity.phir_extension_px8 import (
    BRANCHES,
    HALVES,
    NEGATIVE_CONTROL,
    PRIMARY_SUPPORT,
    SUPPORT_LEVELS,
    TARGET_FORMULATION,
    PairBlock,
    PX8Spec,
    ResilienceCase,
    _concatenate_pairs,
    _future_seed,
    _score_arm_halves,
    _selection_seed,
    _simulate_branch,
    _support_analysis,
    protocol,
    scientific_spec,
    smoke_spec,
)
from plastic_heredity.phir_instruments import advance_fission_traced
from plastic_heredity.simulator import Snapshot, generate_beta


def _fixture() -> tuple[ResilienceCase, np.ndarray]:
    rng = np.random.default_rng(707)
    beta = np.exp(rng.normal(-4.0, 1.0, size=(100, 100)))
    counts = rng.poisson(2.0, size=(520, 100)).astype(np.int16)
    counts[counts.sum(axis=1) == 0, 0] = 1
    composition = np.zeros(100, dtype=np.int64)
    composition[:40] = 1
    case = ResilienceCase(
        "PX8-test",
        "02",
        0,
        20,
        beta,
        Snapshot(composition, 20, (True,) * 20, (0.95,) * 20),
        counts[-512:],
    )
    return case, counts


def test_px8_has_one_eligible_formulation_and_one_nonwinning_control() -> None:
    value = protocol()
    assert TARGET_FORMULATION == "generational__beta__typeset"
    assert NEGATIVE_CONTROL == "molecular__self__revised"
    assert value["target"]["formulation"] == TARGET_FORMULATION
    assert value["negative_control"]["pass_path"] is False


def test_scientific_halves_and_support_are_exact_and_nested() -> None:
    assert BRANCHES == 256
    assert PRIMARY_SUPPORT == 128
    assert SUPPORT_LEVELS == (16, 32, 64, 128)
    assert set(HALVES["A"]).isdisjoint(HALVES["B"])
    assert set(HALVES["A"]) | set(HALVES["B"]) == set(range(BRANCHES))
    for smaller, larger in zip(SUPPORT_LEVELS[:-1], SUPPORT_LEVELS[1:], strict=True):
        assert set(HALVES["A"][:smaller]).issubset(HALVES["A"][:larger])


def test_future_stream_is_arm_free_and_separate_from_action() -> None:
    case, _ = _fixture()
    spec = smoke_spec()
    assert _future_seed(spec, case, 0) == _future_seed(spec, case, 0)
    assert _future_seed(spec, case, 0) != _selection_seed(spec, case)


def test_pair_concatenation_keeps_explicit_boundaries() -> None:
    _, counts = _fixture()
    first = PairBlock(counts[:3], counts[1:4], counts[:2], counts[1:3])
    second = PairBlock(counts[10:14], counts[11:15], counts[10:13], counts[11:14])
    past, future = _concatenate_pairs([first, second], "generational")
    assert past.shape == future.shape == (5, 100)
    assert np.array_equal(past[1], counts[1])
    assert np.array_equal(past[2], counts[10])


def test_score_ladder_exercises_target_and_control_without_alternatives() -> None:
    case, counts = _fixture()
    blocks = [
        PairBlock(
            counts[index : index + 2],
            counts[index + 1 : index + 3],
            counts[index : index + 1],
            counts[index + 1 : index + 2],
        )
        for index in range(16)
    ]
    rows = _score_arm_halves("NOOP", case, blocks, smoke_spec())
    assert {row["formulation"] for row in rows} == {
        TARGET_FORMULATION,
        NEGATIVE_CONTROL,
    }
    assert {row["support_branches"] for row in rows} == {4, 8}
    assert all(np.isfinite(row["value"]) for row in rows)


def test_noop_branch_matches_plain_traced_simulator() -> None:
    config = GardConfig()
    beta = generate_beta(config, np.random.default_rng(81))
    composition = np.zeros(config.n_types, dtype=np.int64)
    composition[:40] = 1
    case = ResilienceCase(
        "PX8-noop",
        "02",
        0,
        20,
        beta,
        Snapshot(composition, 20, (True,) * 20, (0.95,) * 20),
        np.tile(composition, (32, 1)).astype(np.int16),
    )
    spec = PX8Spec("smoke", 1, (20,), 2, 1, 1, 8, 8)
    row, pairs = _simulate_branch(case, None, 0, spec)
    traced = advance_fission_traced(
        composition,
        beta,
        config,
        CANDIDATES["02"],
        np.random.default_rng(_future_seed(spec, case, 0)),
    )
    assert row["record_digest"]
    assert np.array_equal(pairs.generational_past[0], traced.record.parent)
    assert np.array_equal(pairs.generational_future[0], traced.record.daughter)
    assert row["primary"] == 0  # a run of three cannot certify in one fission


def test_support_convergence_contract_detects_settling_positive_effect() -> None:
    rows: list[dict[str, object]] = []
    effects = {16: -2.0, 32: -1.0, 64: 0.5, 128: 0.7}
    for support, effect in effects.items():
        for candidate in ("02", "03"):
            for half in ("A", "B"):
                for matrix_id in range(6):
                    for arm, value in (
                        ("RENEWAL_UP", effect),
                        ("RENEWAL_DOWN", 0.0),
                        ("RANDOM", 0.0),
                        ("NOOP", 0.0),
                    ):
                        rows.append(
                            {
                                "candidate": candidate,
                                "matrix_id": matrix_id,
                                "state_id": f"s{matrix_id}",
                                "arm": arm,
                                "source_half": half,
                                "support_branches": support,
                                "formulation": TARGET_FORMULATION,
                                "value": value,
                                "transitions": support * 8,
                            }
                        )
    spec = PX8Spec("test", 6, (20,), 256, 8, 60, 32, 32)
    _, _, convergence = _support_analysis(pd.DataFrame(rows), spec, {})
    assert convergence["positive_at_64_and_128_all_cells"] is True
    assert convergence["contracting_final_step"] is True
    assert convergence["pass"] is True


def test_scientific_protocol_keeps_matrix_as_inference_unit() -> None:
    value = protocol(scientific_spec())
    assert value["inference"]["unit"] == "whole catalytic matrix"
    assert value["cohort"]["minimum_eligible_matrices"] == 40
    assert value["randomness"]["arm_in_future_seed"] is False
