from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np

from reviewer_ca_lineage_renewal_replication_v2.campaign import (
    _context,
    _frozen_reference,
    _recompute_reference,
    _worker,
)
from reviewer_ca_lineage_renewal_replication_v2.contract import (
    CONDITIONS,
    DEFAULT_ARTIFACTS,
    sha256_json,
)
from reviewer_ca_lineage_renewal_replication_v2.engine import (
    MOTIF_OFFSETS,
    MOTIF_WEIGHTS,
    corrected_sweep,
    decode_state_hex,
    encode_state_hex,
    heldout_balanced_accuracy,
    motif_addresses,
    motif_addresses_batch,
    motif_energy_advantage_batch,
    parent_statistics,
    phenotype_features_batch,
    pooled_reference,
    read_motif_energy_batch,
    reader_probability,
    semantic_fields,
    simulate_pair_lineages,
    step_rule31649_batch,
    texture2x2_addresses_batch,
    texture2x2_counts_batch,
)


def _fixture() -> tuple[list[str], str, np.ndarray, dict[str, np.ndarray]]:
    rng = np.random.default_rng(731)
    donors = [rng.integers(0, 2, (16, 16), dtype=np.uint8) for _ in range(2)]
    histories = [parent_statistics(board, 32) for board in donors]
    reference = pooled_reference(histories)
    reset = np.zeros((16, 16), dtype=np.uint8)
    reset[7, 7:10] = 1
    target_counts = [texture2x2_counts_batch(board) for board in donors]
    targets = {
        label: counts / counts.sum()
        for label, counts in zip(("A", "B"), target_counts, strict=True)
    }
    return [encode_state_hex(board) for board in donors], encode_state_hex(reset), reference, targets


def test_lsb_state_roundtrip_and_positions() -> None:
    low = "0" * 63 + "1"
    board = decode_state_hex(low)
    assert board[0, 0] == 1
    assert int(board.sum()) == 1
    assert encode_state_hex(board) == low
    high = "8" + "0" * 63
    board = decode_state_hex(high)
    assert board[-1, -1] == 1
    assert encode_state_hex(board) == high


def test_lsb_motif_and_texture_codes() -> None:
    board = np.zeros((16, 16), dtype=np.uint8)
    board[0, 0] = 1
    assert motif_addresses_batch(board)[1, 1] == 1
    assert texture2x2_addresses_batch(board)[0, 0] == 1
    board.fill(0)
    board[1, 1] = 1
    assert motif_addresses_batch(board)[1, 1] == 1 << 4
    assert texture2x2_addresses_batch(board)[0, 0] == 1 << 3


def test_energy_advantage_matches_brute_force() -> None:
    rng = np.random.default_rng(99)
    board = rng.integers(0, 2, (16, 16), dtype=np.uint8)
    carrier = rng.normal(size=512).astype(np.float32)
    observed = motif_energy_advantage_batch(board, carrier)
    addresses = motif_addresses(board)
    expected = np.zeros((16, 16), dtype=np.float32)
    for row in range(16):
        for column in range(16):
            value = np.float32(0.0)
            for offset in MOTIF_OFFSETS:
                center = ((row - offset[0]) % 16, (column - offset[1]) % 16)
                current = int(addresses[center])
                value += carrier[current ^ MOTIF_WEIGHTS[offset]] - carrier[current]
            expected[row, column] = value
    np.testing.assert_allclose(observed, expected, rtol=0, atol=2e-6)


def test_reader_probability_is_exact_and_amplitude_sensitive() -> None:
    advantage = np.array([-2.0, 0.0, 1.0, 2.0, 4.0])
    expected = 0.25 * np.tanh(np.maximum(advantage, 0.0) / 9.0)
    np.testing.assert_allclose(reader_probability(advantage, 0.25), expected, rtol=0, atol=0)
    probabilities = reader_probability(np.array([0.5, 1.0, 2.0, 4.0]), 0.25)
    assert np.all(np.diff(probabilities) > 0)
    assert reader_probability(np.array([2.0]), 0.25)[0] > reader_probability(np.array([1.0]), 0.25)[0]


def test_carrier_gain_and_decay_change_actual_reader_decisions() -> None:
    rng = np.random.default_rng(314)
    board = rng.integers(0, 2, (16, 16), dtype=np.uint8)
    carrier = rng.normal(size=512).astype(np.float32)
    advantage = motif_energy_advantage_batch(board, carrier)
    low_probability = reader_probability(0.5 * advantage, 0.25)
    high_probability = reader_probability(2.0 * advantage, 0.25)
    eligible = high_probability > low_probability + 1e-10
    assert np.any(eligible)
    uniform = np.ones((16, 16), dtype=np.float64)
    uniform[eligible] = 0.5 * (low_probability[eligible] + high_probability[eligible])
    low = read_motif_energy_batch(board, 0.5 * carrier, 0.25, uniform)
    high = read_motif_energy_batch(board, 2.0 * carrier, 0.25, uniform)
    assert not np.array_equal(low, high)
    assert np.sum(high != board) > np.sum(low != board)


def test_reader_then_noise_order_is_explicit_and_differs_from_old_order() -> None:
    rng = np.random.default_rng(812)
    found_difference = False
    for _ in range(20):
        board = rng.integers(0, 2, (1, 16, 16), dtype=np.uint8)
        carrier = rng.normal(size=(1, 512)).astype(np.float32)
        uniform = rng.random((1, 16, 16))
        noise = rng.random((1, 16, 16)) < 0.08
        observed = corrected_sweep(board, carrier, np.array([True]), uniform, noise)
        predicted = step_rule31649_batch(board)
        expected = read_motif_energy_batch(predicted, carrier, 0.25, uniform) ^ noise
        np.testing.assert_array_equal(observed, expected)
        old_order = read_motif_energy_batch(predicted ^ noise, carrier, 0.25, uniform)
        found_difference |= not np.array_equal(observed, old_order)
    assert found_difference


def test_visible_descriptor_has_exact_41_fields() -> None:
    rng = np.random.default_rng(17)
    boards = rng.integers(0, 2, (2, 3, 16, 16), dtype=np.uint8)
    terminal = texture2x2_counts_batch(boards)
    accumulated = 8 * terminal
    features = phenotype_features_batch(accumulated, terminal, boards)
    assert features.shape == (2, 3, 41)
    np.testing.assert_allclose(features[..., :15].sum(axis=-1), 1.0)
    np.testing.assert_allclose(features[..., 15:30].sum(axis=-1), 1.0)
    np.testing.assert_allclose(features[..., 30], boards.mean(axis=(-2, -1)))


def test_heldout_decoder_uses_condition_local_centroids_and_strict_ties() -> None:
    values = np.zeros((2, 8, 2), dtype=np.float64)
    values[0, :, 0] = -1.0
    values[1, :, 0] = 1.0
    assert heldout_balanced_accuracy(values, pair_id="p", feature_kind="x") == 1.0
    reversed_values = values[::-1].copy()
    assert heldout_balanced_accuracy(
        reversed_values,
        pair_id="p",
        feature_kind="x",
    ) == 1.0
    tied = np.zeros_like(values)
    tied_score = heldout_balanced_accuracy(
        tied, pair_id="ties", feature_kind="x"
    )
    assert tied_score == 0.0


def test_calibration_reference_reconstructs_to_1e_minus_15() -> None:
    context = _context(DEFAULT_ARTIFACTS)
    observed = _recompute_reference(context)
    expected = _frozen_reference(context)
    np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-15)


def test_short_lineage_uses_sparse_explicit_reset_and_decay_lowers_influence() -> None:
    donors, reset, reference, targets = _fixture()
    result = simulate_pair_lineages(
        pair_id="synthetic-pair",
        donor_state_hex=donors,
        donor_initial_state_hex=[reset, reset],
        reset_state_hex=reset,
        reference_probability=reference,
        targets_primary=targets,
        targets_terminal=targets,
        replicates=2,
        generations=4,
    )
    assert set(result["conditions"]) == set(CONDITIONS)
    assert result["reset"]["live_cells"] == 3
    assert all(
        set(condition["outcomes"]) == {"1", "2", "4"}
        for condition in result["conditions"].values()
    )
    reset_hashes = {
        condition["reset_sha256"] for condition in result["conditions"].values()
    }
    assert reset_hashes == {result["reset"]["array_sha256"]}
    no_rewrite = result["conditions"]["no_rewrite"]["carrier_history"]
    assert no_rewrite["2"]["entry"]["mean_abs"] < no_rewrite["1"]["entry"]["mean_abs"]
    assert no_rewrite["4"]["entry"]["mean_abs"] < no_rewrite["2"]["entry"]["mean_abs"]
    assert (
        result["conditions"]["rescue_same_enter_g4"]["outcomes"]["4"]
        == result["conditions"]["intact"]["outcomes"]["4"]
    )


def test_generation16_emits_complete_carrier_and_visible_decoder_panel() -> None:
    donors, reset, reference, targets = _fixture()
    secondary_conditions = [
        "intact",
        "no_rewrite",
        "read_disabled",
        "ablate_after_g2",
        "rescue_same_enter_g4",
        "rescue_opposite_enter_g4",
    ]
    result = simulate_pair_lineages(
        pair_id="synthetic-secondary-pair",
        donor_state_hex=donors,
        donor_initial_state_hex=[reset, reset],
        reset_state_hex=reset,
        reference_probability=reference,
        targets_primary=targets,
        targets_terminal=targets,
        replicates=2,
        generations=16,
        conditions=secondary_conditions,
    )
    assert result["secondary_decoder"]["generation"] == 16
    for kind in ("carrier", "phenotype"):
        assert set(result["secondary_decoder"][kind]) == set(secondary_conditions)
        assert all(
            0.0 <= value <= 1.0
            for value in result["secondary_decoder"][kind].values()
        )


def test_semantic_rng_and_worker_payload_are_reproducible() -> None:
    first = semantic_fields("pair", 3, 2)
    second = semantic_fields("pair", 3, 2)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    donors, reset, reference, targets = _fixture()
    argument = {
        "phase": "test",
        "pair": {
            "pair_id": "worker-pair",
            "a_donor_id": "a",
            "b_donor_id": "b",
            "density_difference": 0.0,
            "launch_index": 0,
        },
        "donor_state_hex": donors,
        "donor_initial_state_hex": [reset, reset],
        "reset_state_hex": reset,
        "reference_probability": reference.tolist(),
        "targets_primary": {key: value.tolist() for key, value in targets.items()},
        "targets_terminal": {key: value.tolist() for key, value in targets.items()},
        "replicates": 1,
        "generations": 1,
        "conditions": ["intact"],
    }
    local = _worker(argument)
    with ProcessPoolExecutor(max_workers=2) as executor:
        remote = executor.submit(_worker, argument).result()
    assert sha256_json(local) == sha256_json(remote)
