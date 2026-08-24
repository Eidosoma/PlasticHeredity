from __future__ import annotations

import numpy as np
import pandas as pd

import plastic_heredity.intervention_cr9m_launch as cr9m
from plastic_heredity.config import GardConfig
from plastic_heredity.intervention_core import MolecularEdit, apply_molecular_edit
from plastic_heredity.simulator import Snapshot


def _snapshot(generation: int = 1) -> Snapshot:
    composition = np.zeros(100, dtype=np.int64)
    composition[:4] = 1
    return Snapshot(
        composition=composition,
        generation=generation,
        inheritance=(True,) if generation else (),
        boundary_h=(0.95,) if generation else (),
        previous_growth_steps=10 if generation else 0,
        cumulative_growth_steps=10 if generation else 0,
    )


def test_cr9m_design_and_claim_boundary_are_frozen() -> None:
    frozen = cr9m.protocol()
    assert cr9m.MATRICES == 48
    assert cr9m.REPLICATES == 3
    assert cr9m.MATURE_LANDMARK == 60
    assert cr9m.RELEASE_HORIZON == 60
    assert cr9m.PULSE_LENGTHS == (1, 2, 4, 8, 16, 32, 60)
    assert cr9m.LAUNCHES == ("NASCENT", "MATURE")
    assert cr9m.CONVENTIONS == ("RELAXED", "POST_EDIT")
    assert cr9m.BOOTSTRAP_REPETITIONS == 4096
    assert cr9m.RANDOMIZATION_REPETITIONS == 4096
    assert frozen["boundary"]["sealed_cr9_unchanged"] is True
    assert frozen["boundary"]["cannot_rescue_or_replace_cr9"] is True
    assert "stop without launching CR10" in frozen["stop_rule"]


def test_future_stream_excludes_all_factor_identities() -> None:
    beta = np.eye(100)
    nascent = cr9m.LaunchCase("n", "02", 7, "NASCENT", beta, _snapshot(0))
    mature = cr9m.LaunchCase("m", "02", 7, "MATURE", beta, _snapshot(60))
    assert cr9m._future_seed(nascent, 2) == cr9m._future_seed(mature, 2)
    assert cr9m._future_seed(nascent, 2) != cr9m._future_seed(nascent, 1)
    assert len(cr9m.SEEDS) == len(set(cr9m.SEEDS.values()))


def test_registered_controller_conventions_have_exact_boundaries(
    monkeypatch: object,
) -> None:
    def fake_down(*_args: object) -> tuple[float, MolecularEdit, float]:
        return 0.2, MolecularEdit(0, 4), 0.1

    monkeypatch.setattr(cr9m, "_model_down", fake_down)  # type: ignore[attr-defined]
    relaxed, relaxed_trace = cr9m._pulse_controller(
        "RELAXED", 1, object(), GardConfig()  # type: ignore[arg-type]
    )
    assert relaxed(_snapshot(), np.eye(100), "02", 0) is None
    assert relaxed_trace.action_steps == []

    post, post_trace = cr9m._pulse_controller(
        "POST_EDIT", 2, object(), GardConfig()  # type: ignore[arg-type]
    )
    assert post(_snapshot(), np.eye(100), "02", 0) == MolecularEdit(0, 4)
    assert post(_snapshot(), np.eye(100), "02", 1) == MolecularEdit(0, 4)
    assert post_trace.action_steps == [1, 2]

    relaxed_two, relaxed_two_trace = cr9m._pulse_controller(
        "RELAXED", 2, object(), GardConfig()  # type: ignore[arg-type]
    )
    assert relaxed_two(_snapshot(), np.eye(100), "02", 0) == MolecularEdit(0, 4)
    assert relaxed_two(_snapshot(), np.eye(100), "02", 1) is None
    assert relaxed_two_trace.action_steps == [1]


def test_legal_mass_preserving_edit_fixture() -> None:
    before = _snapshot().composition
    after = apply_molecular_edit(before, MolecularEdit(0, 4))
    assert int(after.sum()) == int(before.sum())
    assert np.all(after >= 0)
    assert after[0] == 0
    assert after[4] == 1


def test_spearman_and_whole_matrix_draw_contracts() -> None:
    assert cr9m.spearman_constant_zero(np.arange(7), np.ones(7)) == 0.0
    assert cr9m.spearman_constant_zero(np.arange(7), np.arange(7)) == 1.0
    assert cr9m.spearman_constant_zero(np.arange(7), -np.arange(7)) == -1.0
    draws = cr9m.inference_draws()
    assert draws["bootstrap_indices"].shape == (4096, 48)
    assert draws["randomization_signs"].shape == (4096, 48)
    assert np.all(np.isin(draws["randomization_signs"], (-1.0, 1.0)))


def _inference_fixture() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for candidate in ("02", "03"):
        for matrix_id in range(cr9m.MATRICES):
            jitter = (matrix_id % 3) * 0.001
            for launch in cr9m.LAUNCHES:
                for convention in cr9m.CONVENTIONS:
                    for index, pulse in enumerate(cr9m.PULSE_LENGTHS):
                        if launch == "NASCENT":
                            persistence = 5.0 + index + jitter
                            top1 = 0.02 + 0.01 * index
                            entropy_reduction = 0.02 * index
                            occupied_reduction = float(index)
                            throughput_ratio = 0.03 * index
                            risk_reduction = 0.01 * index
                        else:
                            persistence = 10.0 + jitter
                            top1 = 0.001 * index
                            entropy_reduction = 0.001 * index
                            occupied_reduction = 0.0
                            throughput_ratio = 0.001 * index
                            risk_reduction = 0.001 * index
                        rows.append(
                            {
                                "candidate": candidate,
                                "matrix_id": matrix_id,
                                "launch": launch,
                                "convention": convention,
                                "pulse_length": pulse,
                                "persistence": persistence,
                                "top1_increase": top1,
                                "entropy_reduction": entropy_reduction,
                                "occupied_reduction": occupied_reduction,
                                "log_throughput_ratio": throughput_ratio,
                                "risk_reduction": risk_reduction,
                            }
                        )
    return pd.DataFrame(rows)


def test_registered_launch_moderation_fixture_passes() -> None:
    metrics, stored, cells = cr9m.compute_inference(
        _inference_fixture(),
        cr9m.inference_draws(),
        replay_exact=True,
        release_zero_interventions=True,
        readback_exact=True,
    )
    assert metrics["primary_launch_moderation_gate"] is True
    assert metrics["protocol_robust_nascent_hysteresis"] is True
    assert metrics["complete_registered_gate_with_integrity"] is True
    assert len(cells) == 48 * 2 * 2 * 2
    assert stored["c02_launch_moderation_matrix"].shape == (48,)


def test_integrity_failure_does_not_change_scientific_estimand() -> None:
    metrics, _, _ = cr9m.compute_inference(
        _inference_fixture(),
        cr9m.inference_draws(),
        replay_exact=False,
        release_zero_interventions=True,
        readback_exact=True,
    )
    assert metrics["primary_launch_moderation_gate"] is True
    assert metrics["complete_registered_gate_with_integrity"] is False
