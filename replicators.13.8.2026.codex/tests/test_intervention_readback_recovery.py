from __future__ import annotations

import json
from pathlib import Path

from plastic_heredity import intervention_replication as original
from plastic_heredity.intervention_readback_recovery import (
    EXPECTED_FAILURE,
    EXPECTED_ORIGINAL_REGISTRATION_ID,
    _checkpoint_digest,
    _require_completed_checkpoints,
    add_derived_pilot_eligibility,
)


ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = ROOT / "results_intervention_replication/registration"
WORK = ROOT / "results_intervention_replication/.p1_work"
FAILED_LOG = ROOT / "results_intervention_replication/p1_cr1_run.log"


def test_original_registration_and_scientific_sources_remain_unchanged() -> None:
    registered = original.verify_registration(REGISTRATION)
    assert registered["registration_id"] == EXPECTED_ORIGINAL_REGISTRATION_ID
    payload = json.loads((REGISTRATION / "registration.json").read_text())
    assert payload["source_hashes"] == original._source_hashes()


def test_failure_is_the_preregistered_derived_field_mismatch() -> None:
    text = FAILED_LOG.read_text(encoding="utf-8")
    assert EXPECTED_FAILURE in text
    assert "[p1 6/8]" not in text
    assert not (ROOT / "results_intervention_replication/p1_cr1_model_guided_pilot").exists()


def test_all_primary_and_replay_checkpoints_are_complete_without_unpickling() -> None:
    record = _require_completed_checkpoints(WORK)
    assert record["generate"]["checkpoint_digest"]["states"] == 400
    assert record["replay"]["checkpoint_digest"]["states"] == 400
    assert _checkpoint_digest(WORK / "generate") == record["generate"][
        "checkpoint_digest"
    ]
    assert _checkpoint_digest(WORK / "replay") == record["replay"][
        "checkpoint_digest"
    ]


def test_recovery_derives_only_the_missing_replay_dependent_field() -> None:
    positive = {"pilot_eligibility_without_replay": True, "untouched": 17}
    returned = add_derived_pilot_eligibility(positive, True)
    assert returned == {
        "pilot_eligibility_without_replay": True,
        "untouched": 17,
        "pilot_eligibility": True,
    }
    negative_replay = {"pilot_eligibility_without_replay": True}
    assert add_derived_pilot_eligibility(negative_replay, False)[
        "pilot_eligibility"
    ] is False
    negative_inference = {"pilot_eligibility_without_replay": False}
    assert add_derived_pilot_eligibility(negative_inference, True)[
        "pilot_eligibility"
    ] is False

