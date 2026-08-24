from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S01 = load_module("build_claim_ledger_for_s02", "scripts/e01/build_claim_ledger.py")
S02 = load_module("build_ambiguity_ledger", "scripts/e01/build_ambiguity_ledger.py")


def make_s01_claim_ledger(tmp_path: Path) -> Path:
    S01.build(tmp_path)
    return tmp_path / "E01_forensic_replication_bundle/ledgers/claim_ledger.csv"


def test_inventory_validates_and_covers_every_claim_and_discrepancy(
    tmp_path: Path,
) -> None:
    claim_path = make_s01_claim_ledger(tmp_path)
    schema = S02.load_yaml(S02.DEFAULT_SCHEMA)
    targets = S02.load_yaml(S02.DEFAULT_TARGETS)
    claims = S02.load_csv(claim_path)
    rows = S02.expand_items(schema, targets, claims)
    validation = S02.validate_rows(schema, targets, claims, rows)
    assert validation["valid"], validation["errors"]
    assert validation["ambiguityCount"] == 105
    assert validation["uniqueParameterCount"] == 105
    assert validation["claimCoverage"]["mappedClaimCount"] == 59
    assert validation["claimCoverage"]["minimumAmbiguitiesPerClaim"] >= 8
    assert set(validation["discrepancyCoverage"]) == {
        f"D{number:02d}" for number in range(1, 13)
    }
    assert validation["noSilentDefaults"] is True
    assert validation["registryExecutable"] is False
    assert validation["unresolvedAmbiguityCount"] > 0
    assert validation["frozenBranchSetCount"] == 21
    assert validation["executionBlockingParameterCount"] == 88


def test_anchor_parameters_use_explicit_values_branches_or_sentinels(
    tmp_path: Path,
) -> None:
    claim_path = make_s01_claim_ledger(tmp_path)
    schema = S02.load_yaml(S02.DEFAULT_SCHEMA)
    targets = S02.load_yaml(S02.DEFAULT_TARGETS)
    rows = S02.expand_items(schema, targets, S02.load_csv(claim_path))
    by_parameter = {row["specification_parameter"]: row for row in rows}

    for parameter in schema["required_plan_parameters"]:
        assert parameter in by_parameter
        assert by_parameter[parameter]["primary_spec_value"]
    assert by_parameter["gard.kinetics.k_f"]["primary_spec_value"].startswith(
        "UNRESOLVED::"
    )
    assert by_parameter["preprocessing.zero.policy"]["primary_spec_value"].startswith(
        "BRANCH_SET::"
    )
    assert (
        by_parameter["association.positive_significant_denominators"][
            "resolution_status"
        ]
        == "RECONCILED"
    )
    assert by_parameter["intervention.time_to_first.unit"][
        "primary_spec_value"
    ].startswith("CONFLICT::")
    assert (
        "D04" in by_parameter["intervention.time_to_first.unit"]["s01_discrepancy_ids"]
    )


def test_build_outputs_registry_and_crosswalks(tmp_path: Path) -> None:
    claim_path = make_s01_claim_ledger(tmp_path)
    result = S02.build(tmp_path, claim_ledger_path=claim_path)
    assert result["valid"]

    paths = [
        tmp_path / "E01_forensic_replication_bundle/ledgers/ambiguity_ledger.csv",
        tmp_path / "E01_forensic_replication_bundle/ledgers/ambiguity_ledger.md",
        tmp_path / "E01_forensic_replication_bundle/ledgers/discrepancy_taxonomy.csv",
        tmp_path
        / "E01_forensic_replication_bundle/specifications/specification_registry.yaml",
        tmp_path / "research_steps/S02/claim_ambiguity_map.csv",
        tmp_path / "research_steps/S02/validation_summary.json",
        tmp_path / "research_steps/S02/artifact_manifest.json",
    ]
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)

    with paths[0].open(encoding="utf-8", newline="") as handle:
        ledger = list(csv.DictReader(handle))
    with paths[2].open(encoding="utf-8", newline="") as handle:
        discrepancies = list(csv.DictReader(handle))
    with paths[4].open(encoding="utf-8", newline="") as handle:
        claim_map = list(csv.DictReader(handle))
    registry = yaml.safe_load(paths[3].read_text(encoding="utf-8"))
    validation = json.loads(paths[5].read_text(encoding="utf-8"))
    manifest = json.loads(paths[6].read_text(encoding="utf-8"))

    assert len(ledger) == 105
    assert len(discrepancies) == 12
    assert len(claim_map) == 59
    assert all(
        not row["s01_discrepancy_ids"] or row["linked_discrepancy_ambiguity_ids"]
        for row in claim_map
    )
    assert len(registry["parameters"]) == 120
    assert registry["executionGate"]["executable"] is False
    assert registry["executionGate"]["noSilentDefaults"] is True
    assert registry["executionGate"]["executionBlockingParameterCount"] == 88
    assert validation["valid"] is True
    assert manifest["researchStepId"] == "S02"
    assert not (tmp_path / "research_steps/S03").exists()
