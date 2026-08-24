from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from plastic_heredity import intervention_p3_lifecycle as lifecycle
from plastic_heredity import intervention_replication as original
from plastic_heredity.intervention_p3_inference_recovery import (
    EXPECTED_FAILURE,
    EXPECTED_LIFECYCLE_AMENDMENT_ID,
    EXPECTED_ORIGINAL_REGISTRATION_ID,
    RANDOM_ARM,
    _protocol,
    compute_registered_p3_inference,
)
from plastic_heredity.intervention_readback_recovery import (
    _checkpoint_digest,
    _require_completed_checkpoints,
)


ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = ROOT / "results_intervention_replication/registration"
LIFECYCLE = ROOT / "results_intervention_replication/p3_lifecycle_amendment"
WORK = ROOT / "results_intervention_replication/.p3_work"
FAILED_LOG = ROOT / "results_intervention_replication/p3_cr4_run.log"
OUTPUT = ROOT / "results_intervention_replication/p3_cr4_beta_surgery_pilot"
AUDIT = ROOT / "results_intervention_replication/p3_cr4_beta_surgery_pilot_lifecycle_audit"


def test_original_registration_and_prospective_amendment_are_intact() -> None:
    registered = original.verify_registration(REGISTRATION)
    assert registered["registration_id"] == EXPECTED_ORIGINAL_REGISTRATION_ID
    amended = lifecycle.verify_amendment(LIFECYCLE)
    assert amended["amendment_id"] == EXPECTED_LIFECYCLE_AMENDMENT_ID


def test_p3_stopped_at_semantic_random_arm_validation_before_output() -> None:
    text = FAILED_LOG.read_text(encoding="utf-8")
    assert EXPECTED_FAILURE in text
    assert "[p3 4/8]" in text
    assert "[p3 5/8]" not in text
    assert not OUTPUT.exists()
    assert not AUDIT.exists()


def test_p3_primary_and_replay_checkpoint_aggregates_are_complete() -> None:
    record = _require_completed_checkpoints(WORK)
    for stage in ("generate", "replay"):
        assert record[stage]["checkpoint_digest"]["states"] == 400
        assert record[stage]["status"]["futures_complete"] == 51_200
        assert _checkpoint_digest(WORK / stage) == record[stage][
            "checkpoint_digest"
        ]


def test_p3_inference_routes_registered_random_surgery_explicitly() -> None:
    spec = original.pilot_spec("p3")
    captured: dict[str, Any] = {}

    def fake_inference(*args: Any, **kwargs: Any) -> tuple[dict[str, Any], list[Any]]:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {"sentinel": True}, []

    metrics, rows = compute_registered_p3_inference(
        [],
        spec,
        np.empty((0, 4, 0)),
        np.empty((0, 4)),
        {},
        inference_function=fake_inference,
    )
    assert metrics == {"sentinel": True}
    assert rows == []
    assert captured["kwargs"]["up_arm"] == "LOOSEN"
    assert captured["kwargs"]["down_arm"] == "TIGHTEN"
    assert captured["kwargs"]["random_arm"] == RANDOM_ARM
    assert captured["kwargs"]["equivalence_margin"] == 0.025
    assert captured["kwargs"]["random_ratio_limit"] == 0.25


def test_recovery_protocol_changes_no_scientific_contract() -> None:
    registration = original.verify_registration(REGISTRATION)
    lifecycle_registration = lifecycle.verify_amendment(LIFECYCLE)
    checkpoints = _require_completed_checkpoints(WORK)
    protocol = _protocol(
        registration,
        lifecycle_registration,
        WORK,
        FAILED_LOG,
        OUTPUT,
        AUDIT,
        checkpoints,
    )
    assert protocol["scientific_contract_changes"] == []
    assert protocol["only_repair"].startswith("pass random_arm='RANDOM_SURGERY'")
    assert protocol["recovery_futures"] == 0
    assert protocol["checkpoint_outcomes_loaded_during_amendment_preparation"] is False
    assert protocol["mandatory_stop_after_recovery"] is True
    assert protocol["confirmation_launched"] is False
    assert protocol["registered_scientific_design"]["arms"] == [
        "LOOSEN",
        "TIGHTEN",
        "RANDOM_SURGERY",
        "NOOP",
    ]


def test_failure_log_is_stable_plain_text() -> None:
    parsed = FAILED_LOG.read_text(encoding="utf-8")
    assert "Traceback (most recent call last):" in parsed
    assert parsed.rstrip().endswith(EXPECTED_FAILURE)
