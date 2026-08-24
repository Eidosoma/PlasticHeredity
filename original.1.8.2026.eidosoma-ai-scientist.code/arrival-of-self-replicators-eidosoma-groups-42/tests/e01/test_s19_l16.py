from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from e01_s19_tensor_architecture_audit.core import (
    REQUIRED_GROUNDING_FIELDS,
    array_sha256,
    assess_hypothesis,
)


def test_preregistration_is_parseable_and_bounded() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "configs/e01/s19_l16_tensor_architecture_audit.yaml"
    )
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config["researchStepId"] == "S19-L16"
    assert len(config["candidateConventionAudit"]) == 3
    assert config["sourceGroundingGate"]["maximumCompleteHypotheses"] == 3
    assert config["frozenInputs"]["generateMatrices"] is False
    assert config["frozenInputs"]["generateTrajectories"] is False


def test_technical_amendment_is_replay_only() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "configs/e01/s19_l16_technical_amendment_001.json"
    )
    amendment = json.loads(path.read_text(encoding="utf-8"))
    assert amendment["authorizedScope"] == "FROZEN_TARGET_REPLAY_VALIDATOR_ONLY"
    assert amendment["outcomeAccessed"] is False
    assert amendment["newModelFitExecuted"] is False
    for field in ("scientificMethodChanged", "scientificValueChanged", "targetChanged", "featureChanged", "splitChanged", "modelChanged", "metricChanged", "gateChanged"):
        assert amendment[field] is False


def test_complete_source_grounding_gate_passes_only_complete_direct_support() -> None:
    evidence = {field: "DIRECT_PAPER_SPECIFICATION" for field in REQUIRED_GROUNDING_FIELDS}
    result = assess_hypothesis(evidence)
    assert result["completeSourceGroundingPassed"] is True
    evidence["scoring_mask"] = "PUBLIC_CODE_MISSING"
    result = assess_hypothesis(evidence)
    assert result["completeSourceGroundingPassed"] is False
    assert "scoring_mask" in result["unsupportedFields"]


def test_partial_plotting_clue_cannot_ground_prediction_tensor() -> None:
    evidence = {
        field: "SOURCE_LINEAGE_INFERENCE" for field in REQUIRED_GROUNDING_FIELDS
    }
    evidence["architecture_topology"] = "DIRECT_PAPER_SPECIFICATION"
    result = assess_hypothesis(evidence)
    assert result["directlyGroundedFieldCount"] == 1
    assert result["registeredForExecution"] is False


def test_unknown_grounding_field_fails_closed() -> None:
    with pytest.raises(ValueError, match="unregistered grounding"):
        assess_hypothesis({"invented_field": "DIRECT_PAPER_SPECIFICATION"})


def test_array_hash_includes_dtype_shape_and_values() -> None:
    a = np.arange(6, dtype=np.float64).reshape(2, 3)
    assert array_sha256(a) == array_sha256(a.copy())
    assert array_sha256(a) != array_sha256(a.astype(np.float32))
    assert array_sha256(a) != array_sha256(a.reshape(3, 2))
