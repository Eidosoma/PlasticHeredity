from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs/e01/s12b_pigozzi_source_audit_preregistration.yaml"
FREEZER = REPO / "scripts/e01/freeze_s12b_preregistration.py"


def load_freezer():
    spec = importlib.util.spec_from_file_location("freeze_s12b_preregistration", FREEZER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preregistration_validates_all_frozen_inputs_and_sources() -> None:
    result = load_freezer().validate_preregistration(require_no_outcomes=True)
    assert result["success"], result["errors"]
    assert all(item["passed"] for item in result["checks"])


def test_scope_and_decision_rules_are_exactly_bounded() -> None:
    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    scope = data["scopeBoundary"]
    assert scope["exactInputTrajectoryCount"] == 12
    assert scope["newGardTrajectories"] == 0
    assert scope["interventionTrajectories"] == 0
    assert scope["exactSourceImplementations"] == ["IIGR_CORRECTED_SOURCE", "PHIRL_REGULARIZED_SOURCE"]
    assert scope["automaticS13Forbidden"]
    assert data["analysis"]["primaryProspectiveEstimand"] == "current_generation_rho_0"
    assert data["classification"]["vocabulary"] == ["SOURCE_FAMILY_NOT_SUPPORTED", "RETROSPECTIVE_SOURCE_FAMILY_RESEMBLANCE", "REGULARIZATION_DEPENDENT_RESEMBLANCE", "SOURCE_FAMILY_PROSPECTIVE_CANDIDATE"]


def test_source_and_temporal_modes_are_frozen_without_author_identity() -> None:
    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert data["sourceRelationship"] == "SOURCE_INFORMED_RECONSTRUCTION"
    assert data["forbiddenIdentities"] == ["AUTHOR_PRIMARY", "PAPER_PRIMARY", "EXACT_GARD_IMPLEMENTATION"]
    assert data["temporalModes"]["full"]["exactLabel"] == "RETROSPECTIVE_FULL_TRAJECTORY_LOCAL"
    assert data["temporalModes"]["prefix"]["minimumPrecedingMolecularTransitions"] == 256
    assert data["sourceEquivalence"]["gates"]["miMaxAbsDifferenceAtMost"] == 1e-10
    assert data["sourceEquivalence"]["gates"]["localPhiRMaxAbsDifferenceAtMost"] == 1e-9
