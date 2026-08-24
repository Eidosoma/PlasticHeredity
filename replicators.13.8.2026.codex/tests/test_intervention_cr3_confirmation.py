from __future__ import annotations

import numpy as np

from plastic_heredity import intervention_replication as base
from plastic_heredity.experiment import StateCase
from plastic_heredity.intervention_cr3_confirmation import (
    BRANCHES,
    EQUIVALENCE_MARGIN,
    LANDMARKS,
    MATRICES,
    MINIMUM_CPU_BUDGET_HOURS,
    MINIMUM_FREE_DISK_BYTES,
    SEEDS,
    _future_seed,
    _selection_seed,
    add_cr3_gate_fields,
    phase_spec,
    protocol,
    validation_checks,
)
from plastic_heredity.intervention_outgoing_rule import (
    SEED_DOMAINS as P2B_SEEDS,
    outgoing_catalytic_influence,
    select_outgoing_rule_edits,
)


def _cell(*, equivalent: bool = True) -> dict:
    return {
        "contrasts": {
            "up_minus_down": {
                "estimate": 0.1,
                "bootstrap_ci95": (0.05, 0.15),
            }
        },
        "up_down_randomization_p_holm": 0.01,
        "random_noop_equivalence": {"tost_equivalent": equivalent},
    }


def test_full_cr3_design_matches_external_directive() -> None:
    frozen = protocol()
    assert MATRICES == 200
    assert BRANCHES == 64
    assert LANDMARKS == (20, 35, 50, 65, 80)
    assert frozen["cohort"]["states"] == 2_000
    assert frozen["futures"]["primary_futures"] == 512_000
    assert frozen["futures"]["replay_futures"] == 512_000
    assert frozen["futures"]["halves"] == {"A": [0, 31], "B": [32, 63]}
    assert frozen["inference"]["random_noop_tost_margin"] == [
        -EQUIVALENCE_MARGIN,
        EQUIVALENCE_MARGIN,
    ]


def test_full_cr3_uses_corrected_outgoing_rule_and_arm_order() -> None:
    spec = phase_spec()
    assert spec.phase == "p2"
    assert spec.arms == ("RULE_UP", "RULE_DOWN", "RANDOM", "NOOP")
    assert spec.contrast == ("RULE_UP", "RULE_DOWN")
    assert protocol()["rule"]["outgoing_quantity"] == "x @ beta == beta.T @ x"
    assert protocol()["rule"]["incoming_quantity_not_used"] == "beta @ x"


def test_outgoing_orientation_and_rule_direction_remain_exact() -> None:
    composition = np.asarray([4, 2, 0, 2], dtype=np.int64)
    beta = np.asarray(
        [
            [1.0, 40.0, 2.0, 3.0],
            [7.0, 1.0, 5.0, 2.0],
            [80.0, 4.0, 1.0, 9.0],
            [2.0, 3.0, 60.0, 1.0],
        ],
        dtype=np.float64,
    )
    x = composition / composition.sum()
    influence = outgoing_catalytic_influence(composition, beta)
    selected = select_outgoing_rule_edits(composition, beta)
    assert np.array_equal(influence, x @ beta)
    assert np.array_equal(influence, beta.T @ x)
    assert influence[selected["RULE_DOWN"].add_type] - influence[
        selected["RULE_DOWN"].remove_type
    ] > 0.0
    assert influence[selected["RULE_UP"].add_type] - influence[
        selected["RULE_UP"].remove_type
    ] < 0.0


def test_full_cr3_seed_domains_are_new_and_arm_free() -> None:
    assert len(SEEDS) == len(set(SEEDS.values()))
    assert set(SEEDS.values()).isdisjoint(base.SEED_DOMAINS.values())
    assert set(SEEDS.values()).isdisjoint(P2B_SEEDS.values())
    case = StateCase(
        "cr3-seed-fixture",
        "FIX",
        "02",
        9,
        20,
        np.eye(4),
        base._fixture_snapshot(),
    )
    spec = phase_spec()
    future = [_future_seed(spec, case, branch) for branch in range(4)]
    assert len(set(future)) == 4
    assert _selection_seed(spec, case) not in future


def test_cr3_gate_does_not_import_cr1_only_side_arm_or_ratio_gates() -> None:
    metrics = {"cells": [_cell() for _ in range(4)]}
    add_cr3_gate_fields(metrics)
    assert metrics["cr3_all_four_cells_scientific_pass"] is True
    for cell in metrics["cells"]:
        assert set(cell["cr3_registered_gates"]) == {
            "rule_up_minus_down_positive",
            "rule_up_minus_down_bootstrap_lower_positive",
            "holm_randomization_below_0_05",
            "random_tost_equivalent_to_noop",
        }


def test_cr3_gate_requires_random_noop_equivalence_in_every_cell() -> None:
    cells = [_cell() for _ in range(4)]
    cells[2] = _cell(equivalent=False)
    metrics = {"cells": cells}
    add_cr3_gate_fields(metrics)
    assert metrics["cr3_all_four_cells_scientific_pass"] is False
    assert metrics["cells"][2]["cr3_registered_cell_pass"] is False


def test_cr3_operational_boundary_is_frozen() -> None:
    frozen = protocol()["operational"]
    assert MINIMUM_CPU_BUDGET_HOURS == 15.0
    assert MINIMUM_FREE_DISK_BYTES == 2_500_000_000
    assert frozen["no_mid_phase_kill"] is True
    assert frozen["mandatory_review_stop_after_seal"] is True


def test_cr3_validation_generates_no_scientific_cohort() -> None:
    validated = validation_checks()
    assert validated["all_checks_passed"] is True
    assert validated["scientific_matrices_generated"] == 0
    assert validated["scientific_futures_generated"] == 0
