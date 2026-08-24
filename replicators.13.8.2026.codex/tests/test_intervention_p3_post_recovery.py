from __future__ import annotations

import json
from pathlib import Path

from plastic_heredity.intervention_p3_inference_recovery import (
    EXPECTED_LIFECYCLE_AMENDMENT_ID,
    EXPECTED_ORIGINAL_REGISTRATION_ID,
    RANDOM_ARM,
    verify_amendment,
)
from plastic_heredity.mechanistic import verify_checksums


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results_intervention_replication"
AMENDMENT = RESULT_ROOT / "p3_inference_recovery_amendment"
RESULT = RESULT_ROOT / "p3_cr4_beta_surgery_pilot"
AUDIT = RESULT_ROOT / "p3_cr4_beta_surgery_pilot_lifecycle_audit"


def test_p3_recovery_bundles_are_checksum_valid() -> None:
    verify_checksums(AMENDMENT)
    verify_checksums(RESULT)
    verify_checksums(AUDIT)


def test_p3_recovery_preserved_provenance_and_generated_zero_futures() -> None:
    amendment = verify_amendment(AMENDMENT)
    manifest = json.loads((RESULT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["registration_id"] == EXPECTED_ORIGINAL_REGISTRATION_ID
    assert (
        manifest["prospective_lifecycle_amendment_id"]
        == EXPECTED_LIFECYCLE_AMENDMENT_ID
    )
    assert manifest["inference_recovery_amendment_id"] == amendment["amendment_id"]
    assert manifest["primary_futures"] == 51_200
    assert manifest["replay_futures"] == 51_200
    assert manifest["recovery_futures"] == 0
    assert manifest["checkpoint_hashes_unchanged"] is True
    assert manifest["scientific_contract_changes"] == []
    assert manifest["next_scientific_phase_launched"] is False


def test_p3_replay_and_corrected_readback_are_exact() -> None:
    replay = json.loads((RESULT / "replay_audit.json").read_text(encoding="utf-8"))
    readback = json.loads(
        (RESULT / "readback_audit.json").read_text(encoding="utf-8")
    )
    assert replay["state_edit_endpoint_and_process_digests_exact"] is True
    assert replay["maximum_prediction_absolute_error"] == 0.0
    assert readback["semantic_random_arm"] == RANDOM_ARM
    assert readback["semantic_random_arm_explicit_in_primary_and_readback"] is True
    assert readback["primary_metrics_exact"] is True
    assert readback["matrix_effects_exact"] is True
    assert readback["derived_pilot_eligibility_recomputed"] is True


def test_p3_manifest_matches_sealed_primary_metrics() -> None:
    manifest = json.loads((RESULT / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads(
        (RESULT / "primary_metrics.json").read_text(encoding="utf-8")
    )
    assert manifest["arms"] == ["LOOSEN", "TIGHTEN", RANDOM_ARM, "NOOP"]
    assert manifest["pilot_eligibility"] == metrics["pilot_eligibility"]
    assert manifest["full_registered_gate"] == metrics[
        "registered_all_four_cells_pass"
    ]
    assert len(metrics["cells"]) == 4
