from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs/e01/s12_strict_mrr_preregistration.yaml"
SCRIPT = REPO / "scripts/e01/freeze_s12_preregistration.py"


def load_freezer():
    spec = importlib.util.spec_from_file_location("freeze_s12_preregistration", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preregistration_validates_against_frozen_inputs() -> None:
    result = load_freezer().validate_preregistration(require_no_outcomes=False)
    assert result["success"], result["errors"]
    assert all(check["passed"] for check in result["checks"])


def test_scope_is_exactly_twelve_and_zero_or_six_triplets() -> None:
    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    scope = data["scopeBoundary"]
    gate = data["baselineFeasibilityGate"]
    assert scope["exactBaselineMatrixCount"] == 12
    assert scope["maximumInterventionTriplets"] == 6
    assert scope["interventionTripletCountRule"] == "exactly_zero_or_exactly_six"
    assert gate["outcome"] == {
        "pass": "run_exactly_six_triplets",
        "fail": "run_zero_triplets_and_retain_every_gate_reason",
    }
    assert scope["nextStepForbidden"] == "S13"


def test_failed_fixed_window_boundaries_and_claim_vocabulary_are_locked() -> None:
    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    scope = data["scopeBoundary"]
    assert scope["fixedWindowRestorationForbidden"]
    assert scope["s11FixedEstimatesForbidden"]
    assert scope["s11rFixedEstimatesForbidden"]
    assert data["lagAndProspectiveIndexing"]["minimumEffectiveSamples"] == 512
    assert data["claimClassification"]["vocabulary"] == [
        "SUPPORTED",
        "DIRECTIONALLY_SUPPORTED",
        "NOT_SUPPORTED_WITHIN_STRICT_SCOPE",
        "UNDERDETERMINED",
        "NOT_EVALUATED",
    ]


def test_intervention_requires_complete_candidates_and_full_set_null() -> None:
    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    intervention = data["interventionDesign"]
    assert intervention["candidates"]["fullSetRequired"]
    assert intervention["scoring"]["everyCandidateMustPassBothPreprocessingStrictGates"]
    assert intervention["separation"]["nullEnvelope"]["families"] == 4096
    assert (
        intervention["suppression"]["exactStatus"] == "INELIGIBLE_ACTION_NOT_SEPARABLE"
    )
    assert intervention["actionConsensus"]["indexOrArrayOrderTieBreakForbidden"]


def test_preoutcome_amendment_closes_action_and_endpoint_details() -> None:
    result = load_freezer().validate_amendment()
    assert result["success"], result["errors"]
    assert all(result["checks"].values())


def test_second_preoutcome_amendment_closes_descriptive_claim_rules() -> None:
    result = load_freezer().validate_amendment_2()
    assert result["success"], result["errors"]
    assert all(result["checks"].values())
