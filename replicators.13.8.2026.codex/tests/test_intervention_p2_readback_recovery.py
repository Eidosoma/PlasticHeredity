from __future__ import annotations

import json
from pathlib import Path

from plastic_heredity import intervention_replication as original
from plastic_heredity.intervention_p2_readback_recovery import (
    EXPECTED_FAILURE,
    EXPECTED_ORIGINAL_REGISTRATION_ID,
    PhaseBatch,
)
from plastic_heredity.intervention_readback_recovery import (
    _checkpoint_digest,
    _require_completed_checkpoints,
)


ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = ROOT / "results_intervention_replication/registration"
WORK = ROOT / "results_intervention_replication/.p2_work"
FAILED_LOG = ROOT / "results_intervention_replication/p2_cr3_run.log"
OUTPUT = ROOT / "results_intervention_replication/p2_cr3_physical_rule_pilot"


def test_original_registration_and_scientific_source_closure_are_intact() -> None:
    registered = original.verify_registration(REGISTRATION)
    assert registered["registration_id"] == EXPECTED_ORIGINAL_REGISTRATION_ID
    payload = json.loads((REGISTRATION / "registration.json").read_text())
    assert payload["source_hashes"] == original._source_hashes()


def test_p2_stopped_only_at_the_known_readback_failure() -> None:
    text = FAILED_LOG.read_text(encoding="utf-8")
    assert EXPECTED_FAILURE in text
    assert "[p2 6/8]" not in text
    assert not OUTPUT.exists()


def test_p2_primary_and_replay_checkpoint_aggregates_are_complete() -> None:
    record = _require_completed_checkpoints(WORK)
    for stage in ("generate", "replay"):
        assert record[stage]["checkpoint_digest"]["states"] == 400
        assert _checkpoint_digest(WORK / stage) == record[stage][
            "checkpoint_digest"
        ]


def test_historical_pickle_alias_is_the_registered_class() -> None:
    assert PhaseBatch is original.PhaseBatch

