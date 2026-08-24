from __future__ import annotations

import numpy as np

from reviewer_motif_channel_replication.campaign import (
    _screen_worker,
    _stage2_worker,
    _validation_worker,
)
from reviewer_motif_channel_replication.contract import FIXED_PRIMARY, as_jsonable_configuration
from reviewer_motif_channel_replication.engine import (
    decode_state_hex,
    deterministic_board,
    encode_state_hex,
    parent_statistics,
    pooled_reference,
    texture2x2,
)


def _synthetic_argument() -> dict:
    donor_index = {}
    for index in range(4):
        board = deterministic_board("synthetic-worker", index, density=0.35 + 0.1 * (index % 2))
        donor_index[f"donor-{index}"] = {"donor_state_hex": encode_state_hex(board)}
    pair = {
        "pair_id": "synthetic-pair",
        "a_donor_id": "donor-0",
        "b_donor_id": "donor-1",
    }
    unrelated = {
        "pair_id": "synthetic-unrelated",
        "a_donor_id": "donor-2",
        "b_donor_id": "donor-3",
    }
    statistics = [
        parent_statistics(decode_state_hex(record["donor_state_hex"]), 32)
        for record in donor_index.values()
    ]
    reference = pooled_reference(statistics)
    target_a = texture2x2(
        deterministic_board("synthetic-target", "A", density=0.25)
    )
    target_b = texture2x2(
        deterministic_board("synthetic-target", "B", density=0.75)
    )
    return {
        "pair": pair,
        "unrelated": unrelated,
        "donor_index": donor_index,
        "references": {
            "32": {key: value.tolist() for key, value in reference.items()}
        },
        "targets_primary": {"A": target_a.tolist(), "B": target_b.tolist()},
        "targets_terminal": {"A": target_a.tolist(), "B": target_b.tolist()},
        "spatial_latch_benchmark": {
            "upper": 0.6,
            "lower": 0.4,
            "decay": 0.55,
            "kappa": 0.05,
        },
        "configurations": [as_jsonable_configuration(FIXED_PRIMARY)],
        "replicates": 1,
    }


def test_synthetic_screen_and_validation_workers_cover_registered_panel() -> None:
    argument = _synthetic_argument()
    screen = _screen_worker(argument)
    assert screen["phase"] == "screen"
    assert set(screen["configurations"]) == {FIXED_PRIMARY.configuration_id}
    validation = _validation_worker(argument)
    conditions = validation["configurations"][FIXED_PRIMARY.configuration_id][
        "conditions"
    ]
    assert set(conditions) == {
        "intact",
        "zero",
        "read_disabled",
        "shuffle",
        "opposite_history",
        "unrelated_same_form",
        "process_noise",
        "carrier_sign_corruption",
        "spatial_latch_benchmark",
        "incomplete_visible64_reset",
    }
    assert conditions["intact"]["reset_asserted_identical"] is True
    assert conditions["incomplete_visible64_reset"]["reset_asserted_identical"] is False


def test_synthetic_stage2_worker_covers_controls_transformation_and_dose() -> None:
    argument = _synthetic_argument()
    stage2_argument = {
        "pair": argument["pair"],
        "unrelated": argument["unrelated"],
        "donor_index": argument["donor_index"],
        "reference": argument["references"]["32"],
        "targets_primary": argument["targets_primary"],
        "targets_terminal": argument["targets_terminal"],
        "launch_resets": {
            f"launch{index}": encode_state_hex(
                deterministic_board("synthetic-launch", index, density=0.5)
            )
            for index in range(4)
        },
        "configuration": as_jsonable_configuration(FIXED_PRIMARY),
        "replicates": 1,
        "primary_environments": ["native", "native_rot90"],
        "core_conditions": [
            "intact",
            "zero",
            "read_disabled",
            "shuffle",
            "matched_random",
            "opposite_history",
            "unrelated_pair",
            "midpoint",
        ],
        "stress_environments": ["random_density_10"],
        "stress_conditions": [
            "intact",
            "zero",
            "opposite_history",
            "unrelated_pair",
        ],
        "dose_contrasts": [0.0, 0.25, 0.50, 0.75, 1.0],
    }
    result = _stage2_worker(stage2_argument)
    assert set(result["environments"]) == {
        "native",
        "native_rot90",
        "random_density_10",
    }
    native = result["environments"]["native"]["conditions"]
    assert "matched_random" in native
    assert "process_noise" in native
    assert "carrier_sign_corruption" in native
    assert all(f"dose_{dose:.2f}" in native for dose in (0.0, 0.25, 0.5, 0.75, 1.0))
