from __future__ import annotations

import numpy as np
import pandas as pd

import plastic_heredity.intervention_cr9_feedback as cr9
from plastic_heredity.config import GardConfig
from plastic_heredity.intervention_core import MolecularEdit
from plastic_heredity.simulator import Snapshot


def test_cr9_design_and_claim_boundary_are_frozen() -> None:
    frozen = cr9.protocol()
    assert cr9.MATRICES == 48
    assert cr9.LANDMARK == 60
    assert cr9.REPLICATES == 6
    assert cr9.HORIZON == 60
    assert cr9.PULSE_LENGTHS == (1, 2, 4, 8, 16, 32, 60)
    assert cr9.PERIODS == (1, 2, 4, 8, 16)
    assert cr9.THRESHOLDS == (0.15, 0.25, 0.35)
    assert cr9.BOOTSTRAP_REPETITIONS == 4096
    assert cr9.RANDOMIZATION_REPETITIONS == 4096
    assert frozen["pulse_ladder"]["constant_persistence_vector_spearman"] == 0.0
    assert frozen["stop_rule"] == "seal CR9 and stop before CR10"
    assert "strict-eight or Phi/PhiID control" in frozen["claim_boundary"]["prohibited"]


def test_future_streams_are_policy_free_and_random_action_is_separate() -> None:
    case, _ = cr9._artificial_case()
    assert len(cr9.SEEDS) == len(set(cr9.SEEDS.values()))
    assert len({cr9._pulse_future_seed(case, 3) for _ in cr9.PULSE_LENGTHS}) == 1
    assert len({cr9._periodic_future_seed(case, 3) for _ in cr9.PERIODIC_POLICIES}) == 1
    assert len({cr9._event_future_seed(case, 3) for _ in cr9.EVENT_POLICIES}) == 1
    assert cr9._periodic_action_seed(case, 3, "RANDOM_EVERY_4") != (
        cr9._periodic_future_seed(case, 3)
    )


class _ConstantPredictor:
    def __init__(self, value: float):
        self.value = value

    def predict_snapshot(self, *_args: object) -> float:
        return self.value


def _snapshot() -> Snapshot:
    return Snapshot(
        composition=np.asarray([1, 1, 0], dtype=np.int64),
        generation=60,
        inheritance=(True,),
        boundary_h=(0.95,),
        previous_growth_steps=10,
        cumulative_growth_steps=600,
    )


def test_periodic_schedule_uses_one_based_successful_boundaries() -> None:
    callback, trace = cr9._periodic_controller(
        "RANDOM_EVERY_2",
        _ConstantPredictor(0.2),  # type: ignore[arg-type]
        GardConfig(n_types=3),
        np.random.default_rng(7),
    )
    for zero_based_step in range(5):
        callback(_snapshot(), np.eye(3), "02", zero_based_step)
    assert trace.action_steps == [2, 4]
    assert len(trace.actions) == 2


def test_event_trigger_is_strictly_greater(monkeypatch: object) -> None:
    def fake_down(*_args: object) -> tuple[float, MolecularEdit, float]:
        return 0.1500000000001, MolecularEdit(0, 2), 0.1

    monkeypatch.setattr(cr9, "_model_down", fake_down)  # type: ignore[attr-defined]
    equal, equal_trace = cr9._event_controller(
        "THRESHOLD_015", _ConstantPredictor(0.15), GardConfig(n_types=3)  # type: ignore[arg-type]
    )
    assert equal(_snapshot(), np.eye(3), "02", 0) is None
    assert equal_trace.threshold_excursions == 0

    above, above_trace = cr9._event_controller(
        "THRESHOLD_015",
        _ConstantPredictor(float(np.nextafter(0.15, 1.0))),  # type: ignore[arg-type]
        GardConfig(n_types=3),
    )
    assert above(_snapshot(), np.eye(3), "02", 0) == MolecularEdit(0, 2)
    assert above_trace.threshold_excursions == 1
    assert above_trace.action_steps == [1]


def test_episode_and_longest_run_use_strict_inheritance() -> None:
    assert cr9.count_nonoverlapping_episodes([0.91, 0.92, 0.93]) == 0
    assert cr9.count_nonoverlapping_episodes([0.9, 0.91, 0.92, 0.93]) == 1
    assert cr9.count_nonoverlapping_episodes(
        [0.8, 0.91, 0.92, 0.93, 0.7, 0.94, 0.95, 0.96]
    ) == 2
    assert cr9.longest_inherited_run([0.9, 0.91, 0.92, 0.8, 0.93]) == 2


def test_spearman_constant_rule_is_explicit() -> None:
    assert cr9.spearman_constant_zero(np.arange(7), np.arange(7)) == 1.0
    assert cr9.spearman_constant_zero(np.arange(7), np.ones(7)) == 0.0
    assert cr9.spearman_constant_zero(np.arange(7), -np.arange(7)) == -1.0


def test_whole_matrix_draws_and_holm_are_exact() -> None:
    draws = cr9.inference_draws()
    assert draws["bootstrap_indices"].shape == (4096, 48)
    assert draws["randomization_signs"].shape == (4096, 48)
    assert np.all(np.isin(draws["randomization_signs"], (-1.0, 1.0)))
    raw = [0.04, 0.001, 0.02, 0.5]
    adjusted = cr9._holm_adjust(raw)
    order = np.argsort(raw)
    assert np.all(np.diff(np.asarray(adjusted)[order]) >= 0)
    assert all(raw[index] <= adjusted[index] <= 1.0 for index in range(len(raw)))


def _inference_fixture() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pulse_rows: list[dict[str, object]] = []
    periodic_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    for candidate in ("02", "03"):
        for matrix_id in range(cr9.MATRICES):
            jitter = 0.001 if matrix_id % 2 else -0.001
            for index, length in enumerate(cr9.PULSE_LENGTHS):
                pulse_rows.append(
                    {
                        "candidate": candidate,
                        "matrix_id": matrix_id,
                        "pulse_length": length,
                        "persistence": 5.0 + index,
                        "release_completed": 1.0,
                        "final_similarity": 0.5 + jitter,
                    }
                )
            for policy in cr9.PERIODIC_POLICIES:
                if policy == "NOOP":
                    inheritance, edits = 0.80 + jitter, 0.0
                elif policy.startswith("RANDOM"):
                    period = int(policy.rsplit("_", 1)[1])
                    inheritance, edits = 0.80 + jitter, 60 // period
                else:
                    period = int(policy.rsplit("_", 1)[1])
                    inheritance, edits = 0.92 + jitter, 60 // period
                periodic_rows.append(
                    {
                        "candidate": candidate,
                        "matrix_id": matrix_id,
                        "policy": policy,
                        "inherited_fixed_horizon_fraction": inheritance,
                        "edits_applied": edits,
                    }
                )
            for policy in cr9.EVENT_POLICIES:
                inheritance = 0.80 + jitter if policy == "NOOP" else 0.92 + jitter
                edits = {"NOOP": 0, "CONTINUOUS": 60}.get(policy, 16)
                event_rows.append(
                    {
                        "candidate": candidate,
                        "matrix_id": matrix_id,
                        "policy": policy,
                        "inherited_fixed_horizon_fraction": inheritance,
                        "edits_applied": edits,
                    }
                )
    return pd.DataFrame(pulse_rows), pd.DataFrame(periodic_rows), pd.DataFrame(event_rows)


def test_registered_inference_fixture_passes_hysteresis_gate() -> None:
    pulse, periodic, event = _inference_fixture()
    metrics, stored = cr9.compute_inference(
        pulse,
        periodic,
        event,
        cr9.inference_draws(),
        pulse_replay_exact=True,
        periodic_replay_exact=True,
        event_replay_exact=True,
        noop_plain_exact=True,
        release_zero_interventions=True,
        readback_exact=True,
    )
    assert metrics["pulse"]["complete_two_candidate_hysteresis_gate"] is True
    assert metrics["complete_cr9_registered_gate"] is True
    assert all(
        item["descriptive_minimum_feedback_interval"] == 16
        for item in metrics["periodic"]["candidates"]
    )
    assert stored["bootstrap_indices"].shape == (4096, 48)


def test_integrity_failure_blocks_complete_gate_but_not_efficacy_result() -> None:
    pulse, periodic, event = _inference_fixture()
    metrics, _ = cr9.compute_inference(
        pulse,
        periodic,
        event,
        cr9.inference_draws(),
        pulse_replay_exact=False,
        periodic_replay_exact=True,
        event_replay_exact=True,
        noop_plain_exact=True,
        release_zero_interventions=True,
    )
    assert metrics["pulse"]["complete_two_candidate_hysteresis_gate"] is True
    assert metrics["complete_cr9_registered_gate"] is False
