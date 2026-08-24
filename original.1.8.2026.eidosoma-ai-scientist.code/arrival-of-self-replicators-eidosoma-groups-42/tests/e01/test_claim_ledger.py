from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts/e01/build_claim_ledger.py"
SPEC = importlib.util.spec_from_file_location("build_claim_ledger", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_schema_and_claims_validate() -> None:
    schema = MODULE.load_yaml(MODULE.DEFAULT_SCHEMA)
    targets = MODULE.load_yaml(MODULE.DEFAULT_TARGETS)
    claims = MODULE.expand_claims(schema, targets)
    result = MODULE.validate_claims(schema, targets, claims)
    assert result["valid"], result["errors"]
    assert result["claimCount"] == 59
    assert result["discrepancyCount"] == 12
    assert result["familyCounts"]["intervention_absolute"] == 12
    assert result["explicitlyReconciledClaimCount"] >= 10


def test_anchor_claims_preserve_source_discrepancies() -> None:
    schema = MODULE.load_yaml(MODULE.DEFAULT_SCHEMA)
    targets = MODULE.load_yaml(MODULE.DEFAULT_TARGETS)
    by_id = {row["claim_id"]: row for row in MODULE.expand_claims(schema, targets)}
    assert by_id["E01-C013"]["reported_target"] == "p=0.1995"
    assert "73/100" in by_id["E01-C015"]["reported_target"]
    assert "54/100" in by_id["E01-C016"]["reported_target"]
    assert "54/73" in by_id["E01-C016"]["reported_target"]
    assert "D05" in by_id["E01-C056"]["discrepancy_ids"]
    assert "D04" in by_id["E01-C043"]["discrepancy_ids"]
    assert by_id["E01-C050"]["reported_target"] == "coefficient=0.041; p<0.001"


def test_build_outputs_round_trip(tmp_path: Path) -> None:
    result = MODULE.build(tmp_path)
    assert result["valid"]
    csv_path = tmp_path / "E01_forensic_replication_bundle/ledgers/claim_ledger.csv"
    md_path = tmp_path / "E01_forensic_replication_bundle/ledgers/claim_ledger.md"
    reconciliation = tmp_path / "research_steps/S01/source_reconciliation.csv"
    validation = tmp_path / "research_steps/S01/validation_summary.json"
    manifest = tmp_path / "research_steps/S01/artifact_manifest.json"
    for path in [csv_path, md_path, reconciliation, validation, manifest]:
        assert path.is_file()
        assert path.stat().st_size > 0
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 59
    assert len({row["claim_id"] for row in rows}) == 59
    markdown = md_path.read_text(encoding="utf-8")
    assert all(row["claim_id"] in markdown for row in rows)
    assert "no overall score is defined" in markdown
    validation_payload = json.loads(validation.read_text(encoding="utf-8"))
    assert validation_payload["valid"] is True
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest_payload["researchStepId"] == "S01"
