from __future__ import annotations

import numpy as np

from wagner_cleanroom.dynamics import (
    POINT,
    build_landscape,
    decode_state,
    encode_state,
    hamming,
    sample_rulebook,
    sequential_sweep,
    state_matrix,
)


def test_state_encoding_round_trip() -> None:
    matrix = state_matrix(10)
    assert np.array_equal(encode_state(matrix), np.arange(1024, dtype=np.uint16))
    assert np.array_equal(decode_state(np.arange(1024, dtype=np.uint16)), matrix)


def test_hamming_is_normal_bit_distance() -> None:
    assert int(hamming(0, 1023)) == 10
    assert int(hamming(0b1010, 0b0011)) == 2


def test_sequential_update_and_zero_retention() -> None:
    weights = np.asarray([[0.0, 1.0], [1.0, 0.0]])
    assert np.array_equal(sequential_sweep(np.asarray([-1, 1], dtype=np.int8), weights), [1, 1])
    assert np.array_equal(
        sequential_sweep(np.asarray([-1, 1], dtype=np.int8), np.zeros((2, 2))),
        [-1, 1],
    )


def test_identity_landscape_has_only_point_attractors() -> None:
    landscape = build_landscape(np.eye(3), max_sweeps=10)
    assert np.all(landscape.kind == POINT)
    assert np.array_equal(landscape.adult, np.arange(8))
    assert np.array_equal(landscape.basin_sizes, np.ones(8, dtype=np.uint16))


def test_rulebook_sampling_is_deterministic() -> None:
    first = second = None
    for proposal in range(100):
        first = sample_rulebook("unit-test-rulebook", proposal)
        if first is not None:
            second = sample_rulebook("unit-test-rulebook", proposal)
            break
    assert first is not None and second is not None
    assert first.proposal_index == second.proposal_index
    assert np.array_equal(first.weights, second.weights)
    assert np.array_equal(first.midpoints, second.midpoints)
    differing = int(hamming(int(first.targets[0]), int(first.targets[1])))
    assert abs(int(hamming(int(first.midpoints[0]), int(first.targets[0]))) - int(hamming(int(first.midpoints[0]), int(first.targets[1])))) <= 1
    assert differing >= 4

