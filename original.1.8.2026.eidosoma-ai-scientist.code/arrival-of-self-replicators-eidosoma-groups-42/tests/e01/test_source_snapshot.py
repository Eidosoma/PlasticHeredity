from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S03 = load_module("build_source_snapshot", "scripts/e01/build_source_snapshot.py")


def make_prior(updates: dict) -> dict:
    parameters = []
    for item in updates["updates"]:
        parameters.append(
            {
                "parameter": item["parameter"],
                "value": f"UNRESOLVED::{item['ambiguityId']}",
                "admissibleValuesOrBranches": "source evidence or explicit sentinel",
                "resolutionStatus": "DEFERRED_EVIDENCE",
                "unresolved": True,
                "sourceEvidence": "prior evidence",
                "resolutionBasis": "prior basis",
                "ownerStep": "S03",
                "ambiguityId": item["ambiguityId"],
                "validationRule": "prior rule",
            }
        )
    parameters.extend(
        [
            {
                "parameter": "untouched.conflict",
                "value": "CONFLICT::a|b",
                "admissibleValuesOrBranches": "a or b",
                "resolutionStatus": "CONFLICT_PRESERVED",
                "unresolved": True,
                "sourceEvidence": "conflicting sources",
                "resolutionBasis": "preserve",
                "ownerStep": "S15",
                "ambiguityId": "E01-A998",
                "validationRule": "reject unqualified",
            },
            {
                "parameter": "untouched.branch",
                "value": "BRANCH_SET::a|b",
                "admissibleValuesOrBranches": "a or b",
                "resolutionStatus": "FROZEN_BRANCH_SET",
                "unresolved": False,
                "sourceEvidence": "planned branches",
                "resolutionBasis": "branch",
                "ownerStep": "S15",
                "ambiguityId": "E01-A999",
                "validationRule": "expand branches",
            },
        ]
    )
    return {
        "schema": "test",
        "researchStepId": "S02",
        "experimentId": "E01",
        "registryVersion": updates["inputRegistryVersion"],
        "ledgerVersion": "test",
        "generatedOn": "2026-08-01",
        "executionGate": {
            "executable": False,
            "rule": "test",
            "unresolvedParameterCount": 7,
            "unexpandedBranchSetCount": 1,
            "executionBlockingParameterCount": 8,
            "noSilentDefaults": True,
        },
        "statusDefinitions": {
            "DEFERRED_EVIDENCE": "test",
            "CONFLICT_PRESERVED": "test",
            "FROZEN_BRANCH_SET": "test",
        },
        "parameters": parameters,
    }


def test_source_only_updates_resolve_exactly_three_and_preserve_other_items() -> None:
    updates = S03.load_yaml(S03.DEFAULT_UPDATES)
    prior = make_prior(updates)
    untouched_before = copy.deepcopy(prior["parameters"][-2:])
    registry, audit, validation = S03.apply_registry_updates(
        prior, updates, "0" * 64
    )

    assert validation["valid"], validation["errors"]
    assert len(audit) == 6
    assert validation["unresolvedCountBefore"] == 7
    assert validation["unresolvedCountAfter"] == 4
    assert validation["branchSetCountAfter"] == 1
    assert validation["executionBlockingCountAfter"] == 5
    assert validation["executionGateOpen"] is False
    assert validation["noSilentDefaults"] is True
    assert validation["preservedNonSourceParameterCount"] == 2
    assert registry["parameters"][-2:] == untouched_before

    by_id = {item["ambiguityId"]: item for item in registry["parameters"]}
    assert {
        ambiguity_id
        for ambiguity_id in updates["allowedAmbiguityIds"]
        if not by_id[ambiguity_id]["unresolved"]
    } == {"E01-A001", "E01-A003", "E01-A004"}
    for ambiguity_id in ("E01-A002", "E01-A043", "E01-A044"):
        assert by_id[ambiguity_id]["value"] == f"UNRESOLVED::{ambiguity_id}"


def test_registry_update_rejects_parameter_mismatch() -> None:
    updates = S03.load_yaml(S03.DEFAULT_UPDATES)
    prior = make_prior(updates)
    prior["parameters"][0]["parameter"] = "wrong.parameter"
    with pytest.raises(ValueError, match="Registry target mismatch"):
        S03.apply_registry_updates(prior, updates, "0" * 64)


def test_source_pin_inventory_has_immutable_ids_hashes_and_license_notes() -> None:
    pins = S03.load_yaml(S03.DEFAULT_PINS)
    assert pins["paper"]["version"] == "v1"
    assert len(pins["paper"]["sha256"]) == 64
    assert pins["paper"]["sizeBytes"] == pins["paper"]["attachmentOriginalSizeBytes"]
    assert len(pins["repositories"]) == 4
    assert {item["sourceId"] for item in pins["repositories"]} == {
        "gard_historical",
        "gard_modern",
        "phyid_reference",
        "omegaid_optional",
    }
    for item in pins["repositories"]:
        assert len(item["commit"]) == 40
        assert len(item["tree"]) == 40
        assert len(item["archiveSha256"]) == 64
        assert item["license"]
        assert item["redistributionPolicy"]
        assert item["files"]


def test_file_hash_verifier_detects_content_change(tmp_path: Path) -> None:
    path = tmp_path / "source.txt"
    path.write_text("frozen\n", encoding="utf-8")
    expected = S03.sha256_file(path)
    assert S03._verify_file(path, expected, path.stat().st_size)["valid"]
    path.write_text("changed\n", encoding="utf-8")
    assert not S03._verify_file(path, expected)["valid"]
