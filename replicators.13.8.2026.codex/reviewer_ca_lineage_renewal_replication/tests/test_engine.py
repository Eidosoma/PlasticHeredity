from __future__ import annotations

import numpy as np

from reviewer_ca_lineage_renewal_replication.contract import CONDITIONS
from reviewer_ca_lineage_renewal_replication.engine import (
    motif_counts_batch,
    read_motif_energy_batch,
    semantic_fields,
    simulate_pair_lineages,
    step_rule31649_batch,
    texture2x2_counts_batch,
    write_carriers_batch,
)
from reviewer_motif_channel_replication.engine import (
    deterministic_board,
    encode_state_hex,
    motif_counts,
    parent_statistics,
    pooled_reference,
    read_motif_energy,
    step_rule31649,
    texture2x2,
    texture2x2_counts,
    write_carrier,
)


def _fixture() -> tuple[list[str], dict[str, np.ndarray], dict[str, np.ndarray]]:
    donors = [
        deterministic_board("lineage-test-donor", label, density=density)
        for label, density in (("A", 0.32), ("B", 0.68))
    ]
    histories = [parent_statistics(board, 32) for board in donors]
    reference = pooled_reference(histories)
    targets = {
        label: texture2x2(
            deterministic_board("lineage-test-target", label, density=density)
        )
        for label, density in (("A", 0.25), ("B", 0.75))
    }
    return [encode_state_hex(board) for board in donors], reference, targets


def test_vectorized_primitives_equal_sealed_scalar_primitives() -> None:
    boards = np.stack(
        [deterministic_board("lineage-test", index, density=0.3 + index * 0.2) for index in range(3)]
    )
    np.testing.assert_array_equal(
        step_rule31649_batch(boards), np.stack([step_rule31649(board) for board in boards])
    )
    np.testing.assert_array_equal(
        motif_counts_batch(boards), np.stack([motif_counts(board) for board in boards])
    )
    np.testing.assert_array_equal(
        texture2x2_counts_batch(boards),
        np.stack([texture2x2_counts(board) for board in boards]),
    )
    _, reference, _ = _fixture()
    scalar = np.stack(
        [write_carrier({"motif": motif_counts(board)}, reference, "motif_energy512") for board in boards]
    )
    batch = write_carriers_batch(motif_counts_batch(boards), reference["motif_probability"])
    np.testing.assert_array_equal(batch, scalar)
    fields = np.random.default_rng(42).random(boards.shape)
    expected = np.stack(
        [read_motif_energy(board, carrier, 0.25, field) for board, carrier, field in zip(boards, scalar, fields, strict=True)]
    )
    np.testing.assert_array_equal(
        read_motif_energy_batch(boards, batch, 0.25, fields), expected
    )


def test_semantic_fields_repeat_exactly() -> None:
    first = semantic_fields("pair", 3, 2)
    second = semantic_fields("pair", 3, 2)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_short_lineage_covers_complete_registered_intervention_panel() -> None:
    donor_hex, reference, targets = _fixture()
    result = simulate_pair_lineages(
        pair_id="synthetic-pair",
        donor_state_hex=donor_hex,
        reference=reference,
        targets_primary=targets,
        targets_terminal=targets,
        replicates=1,
        generations=4,
    )
    assert set(result["conditions"]) == set(CONDITIONS)
    assert all(
        set(condition["outcomes"]) == {"1", "2", "4"}
        for condition in result["conditions"].values()
    )
    reset_hashes = {
        condition["reset_sha256"] for condition in result["conditions"].values()
    }
    assert len(reset_hashes) == 1
    no_rewrite = result["conditions"]["no_rewrite"]["carrier_history"]
    assert no_rewrite["2"]["entry"]["mean_abs"] < no_rewrite["1"]["entry"]["mean_abs"]
