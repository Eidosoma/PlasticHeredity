from __future__ import annotations

import numpy as np
import pandas as pd

from plastic_heredity.config import CANDIDATES, GardConfig
from plastic_heredity.phir_extension_px7 import (
    ARMS,
    FORMULATIONS,
    GaugeCase,
    _array_digest,
    _explicit_transform,
    _future_seed,
    _halves,
    _map_partition,
    _matrix_centered_spearman,
    _max_t_adjust,
    _past_partition,
    _score_pairs,
    _selection_seed,
    protocol,
    smoke_spec,
)
from plastic_heredity.phir_instruments import ATOM_NAMES, PHIR_ATOMS, ANTICHAINS
from plastic_heredity.phir_instruments import lagged_gaussian_mi_graph
from plastic_heredity.simulator import Snapshot, advance_fission, generate_beta
from plastic_heredity.phir_instruments import advance_fission_traced


def _fixture() -> tuple[GaugeCase, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(17)
    beta = np.exp(rng.normal(-4.0, 1.0, size=(100, 100)))
    counts = rng.poisson(2.0, size=(300, 100)).astype(np.int16)
    counts[counts.sum(axis=1) == 0, 0] = 1
    case = GaugeCase(
        "fixture",
        "02",
        0,
        20,
        beta,
        Snapshot(counts[-1].astype(np.int64), 20, (True,) * 20, (0.95,) * 20),
        counts[-512:],
    )
    return case, counts[:-1], counts[1:]


def test_px7_family_is_finite_and_frozen() -> None:
    assert len(FORMULATIONS) == 18
    assert len(set(FORMULATIONS)) == 18
    assert ARMS["resistance"] == ("BREAK_UP", "BREAK_DOWN", "RANDOM", "NOOP")
    assert ARMS["resilience"] == (
        "RENEWAL_UP",
        "RENEWAL_DOWN",
        "RANDOM",
        "NOOP",
    )
    assert protocol()["formulations"]["count"] == 18


def test_explicit_pairs_and_all_partitions_score_all_atoms() -> None:
    case, past, future = _fixture()
    for partition in ("self", "past", "beta"):
        score = _score_pairs(past, future, partition, case)
        assert np.isfinite(score["revised"])
        assert np.isfinite(score["typeset"])
        assert np.isfinite(score["ratio"])
        assert len(score["atoms"]) == len(ATOM_NAMES) == 16
        atom_lookup = {
            atom: value
            for atom, value in zip(
                ((source, target) for source in ANTICHAINS for target in ANTICHAINS),
                score["atoms"],
                strict=True,
            )
        }
        assert np.isclose(
            score["revised"], sum(atom_lookup[atom] for atom in PHIR_ATOMS)
        )


def test_fixed_partition_maps_every_active_coordinate_once() -> None:
    case, past, future = _fixture()
    _, _, active = _explicit_transform(past, future)
    first_species, second_species = _past_partition(case)
    first, second = _map_partition(active, first_species, second_species)
    combined = np.concatenate((first, second))
    assert len(np.unique(combined)) == len(active)
    assert set(combined) == set(range(len(active)))


def test_explicit_lag_graph_matches_public_sequence_graph() -> None:
    from plastic_heredity.phir_extension_px7 import _explicit_graph

    rng = np.random.default_rng(23)
    data = rng.normal(size=(7, 80))
    assert np.allclose(
        _explicit_graph(data[:, :-1], data[:, 1:]),
        lagged_gaussian_mi_graph(data),
        atol=1e-14,
        rtol=1e-14,
    )


def test_future_stream_is_arm_free_and_separate_from_selection() -> None:
    case, _, _ = _fixture()
    spec = smoke_spec()
    first = _future_seed(spec, "resistance", case, 0)
    second = _future_seed(spec, "resistance", case, 0)
    assert first == second
    assert first != _selection_seed(spec, "resistance", case)


def test_branch_halves_are_disjoint_and_complete() -> None:
    halves = _halves(smoke_spec())
    assert set(halves["A"]).isdisjoint(halves["B"])
    assert set(halves["A"]) | set(halves["B"]) == set(range(4))


def test_centered_spearman_keeps_matrix_as_unit() -> None:
    frame = pd.DataFrame(
        {
            "matrix_id": [0] * 4 + [1] * 4,
            "x": [0, 1, 2, 3, 3, 2, 1, 0],
            "y": [0, 1, 2, 3, 3, 2, 1, 0],
        }
    )
    values = _matrix_centered_spearman(frame, "x", "y")
    assert values.to_dict() == {0: 1.0, 1: 1.0}


def test_max_t_penalizes_a_family_not_each_test_in_isolation() -> None:
    items = [{"name": "signal"}, {"name": "null"}]
    vectors = [
        pd.Series({index: 1.0 + index / 100 for index in range(24)}),
        pd.Series({index: (-1.0) ** index * 0.1 for index in range(24)}),
    ]
    arrays: dict[str, np.ndarray] = {}
    _max_t_adjust(items, vectors, 256, "unit_fixture", arrays)
    assert items[0]["max_t_adjusted_p"] < 0.05
    assert items[1]["max_t_adjusted_p"] >= items[0]["max_t_adjusted_p"]
    assert "max_t__unit_fixture__maximum" in arrays


def test_noop_traced_simulator_matches_plain_record() -> None:
    config = GardConfig()
    beta = generate_beta(config, np.random.default_rng(81))
    composition = np.zeros(config.n_types, dtype=np.int64)
    composition[:40] = 1
    seed = 991
    plain = advance_fission(
        composition,
        beta,
        config,
        CANDIDATES["02"],
        np.random.default_rng(seed),
    )
    traced = advance_fission_traced(
        composition,
        beta,
        config,
        CANDIDATES["02"],
        np.random.default_rng(seed),
    ).record
    assert np.array_equal(plain.parent, traced.parent)
    assert np.array_equal(plain.daughter, traced.daughter)
    assert plain.h == traced.h
    assert plain.growth_steps == traced.growth_steps
    assert _array_digest(plain.parent, plain.daughter) == _array_digest(
        traced.parent, traced.daughter
    )
