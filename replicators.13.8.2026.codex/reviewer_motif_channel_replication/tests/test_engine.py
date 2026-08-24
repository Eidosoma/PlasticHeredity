from __future__ import annotations

import numpy as np

from reviewer_motif_channel_replication.contract import ReaderConfiguration
from reviewer_motif_channel_replication.engine import (
    board_diagnostics,
    decode_state_hex,
    deterministic_board,
    encode_state_hex,
    motif_addresses,
    neighbour_count,
    read_contextual,
    read_motif_energy,
    step_rule31649,
    texture2x2,
    transform_board,
    transform_carrier,
    transform_motif_address,
    write_spatial_latch,
)


def test_state_hex_is_big_endian_row_major_and_roundtrips() -> None:
    board = np.zeros((16, 16), dtype=np.uint8)
    board[0, 0] = 1
    board[-1, -1] = 1
    encoded = encode_state_hex(board)
    assert encoded.startswith("80")
    assert encoded.endswith("01")
    np.testing.assert_array_equal(decode_state_hex(encoded), board)


def test_rule_31649_birth_and_survival_table() -> None:
    births = {1, 3, 4, 5, 6}
    survives = {0, 5, 7, 8}
    neighbours = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]
    for alive in (0, 1):
        for count in range(9):
            board = np.zeros((7, 7), dtype=np.uint8)
            board[3, 3] = alive
            for dy, dx in neighbours[:count]:
                board[3 + dy, 3 + dx] = 1
            observed = int(step_rule31649(board)[3, 3])
            expected = int(count in (survives if alive else births))
            assert observed == expected


def test_torus_wraps_at_both_axes() -> None:
    board = np.zeros((5, 5), dtype=np.uint8)
    board[-1, -1] = 1
    board[-1, 0] = 1
    board[0, -1] = 1
    assert neighbour_count(board)[0, 0] == 3


def test_live_2x2_address_convention() -> None:
    board = np.zeros((4, 4), dtype=np.uint8)
    board[0, 0] = 1
    # The top-left-only motif has address binary 1000, which is live bin 8.
    texture = texture2x2(board)
    assert texture[7] > 0


def test_motif_address_and_carrier_transform_are_covariant() -> None:
    board = deterministic_board("test", "covariance", shape=(8, 8), density=0.4)
    addresses = motif_addresses(board)
    rotated_addresses = motif_addresses(transform_board(board, "rot90"))
    expected_centres = np.rot90(addresses)
    expected = np.vectorize(lambda value: transform_motif_address(int(value), "rot90"))(
        expected_centres
    )
    np.testing.assert_array_equal(rotated_addresses, expected)
    carrier = np.arange(512, dtype=np.float64)
    transformed = carrier
    for _ in range(4):
        transformed = transform_carrier(transformed, "rot90")
    np.testing.assert_array_equal(transformed, carrier)
    np.testing.assert_array_equal(
        transform_carrier(transform_carrier(carrier, "reflect_x"), "reflect_x"),
        carrier,
    )


def test_zero_carriers_are_exactly_inert() -> None:
    board = deterministic_board("test", "zero", shape=(9, 9), density=0.5)
    uniform = np.zeros_like(board, dtype=np.float64)
    np.testing.assert_array_equal(
        read_motif_energy(board, np.zeros(512), 1.0, uniform), board
    )
    np.testing.assert_array_equal(
        read_contextual(board, np.zeros(256), 1.0, uniform), board
    )


def test_motif_reader_is_synchronous_and_does_not_mutate_prediction() -> None:
    board = deterministic_board("test", "sync", shape=(7, 7), density=0.5)
    original = board.copy()
    carrier = np.linspace(-2.0, 2.0, 512)
    first = read_motif_energy(board, carrier, 0.5, np.full(board.shape, 0.25))
    second = read_motif_energy(original, carrier, 0.5, np.full(board.shape, 0.25))
    np.testing.assert_array_equal(board, original)
    np.testing.assert_array_equal(first, second)


def test_deterministic_density_and_midpoint_contract() -> None:
    board = deterministic_board("test", "density", density=0.10)
    assert int(board.sum()) == round(0.10 * 256)
    a = np.linspace(-1.0, 1.0, 512)
    b = -a
    midpoint = 0.5 * (a + b)
    np.testing.assert_array_equal(midpoint, np.zeros(512))
    config = ReaderConfiguration("motif_energy512", 32, 0.25, 32)
    assert config.configuration_id == "motif_energy512-w32-s025-d32"


def test_spatial_latch_uses_post_update_occupancy_and_retention() -> None:
    dead = np.zeros((8, 8), dtype=np.uint8)
    latch = write_spatial_latch(dead, window=4, upper=0.6, lower=0.4, retention=0.55)
    np.testing.assert_allclose(latch, -0.55)


def test_non_gating_diagnostics_are_finite_and_translation_invariant() -> None:
    board = deterministic_board("test", "diagnostics", density=0.43)
    translated = transform_board(board, "translate_3_5")
    first = board_diagnostics(board)
    second = board_diagnostics(translated)
    for name in first:
        np.testing.assert_allclose(first[name], second[name], atol=1e-12)
        assert np.all(np.isfinite(first[name]))
