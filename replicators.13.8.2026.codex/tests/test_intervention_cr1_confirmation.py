from plastic_heredity import intervention_replication as base
from plastic_heredity.intervention_cr1_confirmation import (
    BRANCHES,
    LANDMARKS,
    MATRICES,
    MINIMUM_AVAILABLE_CPU_HOURS,
    SEEDS,
    phase_spec,
    protocol,
)


def test_full_cr1_design_matches_directive() -> None:
    frozen = protocol()
    assert MATRICES == 200
    assert BRANCHES == 64
    assert LANDMARKS == (20, 35, 50, 65, 80)
    assert frozen["futures"]["primary_futures"] == 512_000
    assert frozen["futures"]["replay_futures"] == 512_000
    assert frozen["inference"]["all_original_cr1_gates_unchanged"]


def test_cr1_phase_reuses_frozen_p1_algorithm_with_new_seeds() -> None:
    selected = phase_spec()
    assert selected.phase == "p1"
    assert selected.arms == base.PHASE_ARMS["p1"]
    assert selected.contrast == ("MODEL_UP", "MODEL_DOWN")
    assert not set(SEEDS.values()).intersection(base.SEED_DOMAINS.values())
    assert len(SEEDS) == len(set(SEEDS.values()))


def test_cr1_operational_guard_is_phase_boundary_only() -> None:
    frozen = protocol()["operational_gate"]
    assert MINIMUM_AVAILABLE_CPU_HOURS == 17.0
    assert frozen["p4_terminal_checksum_seal_required"]
    assert frozen["no_mid_phase_kill"]

