#!/usr/bin/env python3
"""Build and validate the E01 S01 forensic claim ledger.

The script is deterministic apart from the repository commit recorded in provenance.
It intentionally extracts claims only; it does not resolve S02 method ambiguities or
compute a composite replication score.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
DEFAULT_SCHEMA = REPO_ROOT / "configs/e01/claim_schema.yaml"
DEFAULT_TARGETS = REPO_ROOT / "configs/e01/claim_targets.yaml"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping in {path}")
    return data


def _base_claim() -> dict[str, str]:
    return {
        "claim_id": "",
        "claim_family": "",
        "evidence_layer": "",
        "claim_text": "",
        "primary_source_location": "",
        "corroborating_source_location": "",
        "reported_statistic": "",
        "reported_target": "",
        "expected_direction": "",
        "unit_of_analysis": "",
        "sample_scope": "",
        "reproduction_estimand": "",
        "reproduction_criterion": "",
        "inferential_test": "",
        "specification_status": "",
        "downstream_step": "",
        "discrepancy_ids": "",
        "notes": "",
    }


def expand_claims(
    schema: dict[str, Any], targets: dict[str, Any]
) -> list[dict[str, str]]:
    claims: list[dict[str, str]] = []

    distinct = targets["metric_distinctiveness"]
    for claim_id, metric in distinct["metrics"]:
        row = _base_claim()
        row.update(
            {
                "claim_id": claim_id,
                "claim_family": "metric_distinctiveness",
                "evidence_layer": "associative",
                "claim_text": f"Phi-r has no significant Pearson or Spearman association with {metric}.",
                "primary_source_location": distinct["source"],
                "reported_statistic": "Pearson and Spearman correlation significance",
                "reported_target": "no significant correlation; coefficients and p-values not reported",
                "expected_direction": "no significant association",
                "unit_of_analysis": f"Phi-r versus {metric} comparison",
                "sample_scope": distinct["sample_scope"],
                "reproduction_estimand": f"paper-matched Pearson and Spearman association of Phi-r with {metric}",
                "reproduction_criterion": (
                    "EXACT at reported precision if both paper-matched two-sided tests have p>=0.05; "
                    "DIRECTIONAL if effect sizes are practically small under a preregistered margin; "
                    "NONREPLICATION if a robust association survives the S15 multiplicity policy; "
                    "UNDERDETERMINED if aggregation cannot be reconstructed."
                ),
                "inferential_test": "Pearson and Spearman correlation; multiplicity policy not reported",
                "specification_status": "underdetermined_pending_S02",
                "downstream_step": "S15",
                "notes": "This named metric is a separate target; no across-metric composite pass score is allowed.",
            }
        )
        claims.append(row)

    for raw in targets["claims"]:
        row = _base_claim()
        row.update(
            {key: str(value) if value is not None else "" for key, value in raw.items()}
        )
        claims.append(row)

    table = targets["table1_targets"]
    for claim_id, treatment, outcome, target, direction, point, spread in table["rows"]:
        row = _base_claim()
        is_time = outcome == "time to first replicator"
        if outcome == "persistence":
            unit = "treatment-level persistence summary in molecular steps"
            tolerance = (
                "point estimate within 5% relative and dispersion within 10% relative"
            )
        elif outcome == "probability":
            unit = "treatment-level percent of molecular steps in replication"
            tolerance = "point estimate within 1 percentage point and dispersion within 10% relative"
        elif outcome == "consistency":
            unit = "treatment-level consecutive-step Pearson correlation"
            tolerance = "point estimate within 0.01 and dispersion within 10% relative"
        else:
            unit = "treatment-level time to first replicator; paper unit conflicts"
            tolerance = "point estimate within 1 resolved unit and dispersion within 10% relative"
        discrepancy_ids = "D10"
        if is_time:
            discrepancy_ids = "D04;D10"
        if claim_id == "E01-C042":
            discrepancy_ids = "D05;D10"
        row.update(
            {
                "claim_id": claim_id,
                "claim_family": "intervention_absolute",
                "evidence_layer": "interventional",
                "claim_text": f"Table 1 reports {treatment} {outcome} as {target}.",
                "primary_source_location": table["source"],
                "reported_statistic": "treatment summary: point estimate +/- undefined dispersion",
                "reported_target": target,
                "expected_direction": str(direction),
                "unit_of_analysis": unit,
                "sample_scope": "100 intervention runs per paper Results; pairing and dispersion scope not reported",
                "reproduction_estimand": f"{treatment} treatment mean and reported dispersion for {outcome}",
                "reproduction_criterion": (
                    f"EXACT after S02 resolves the dispersion/unit if {tolerance}; "
                    "DIRECTIONAL if the preregistered treatment ordering agrees; NONREPLICATION if it reverses; "
                    "UNDERDETERMINED while the source unit or +/- definition remains unresolved."
                ),
                "inferential_test": "none for absolute cell; pairwise tests are separate claims",
                "specification_status": "underdetermined_pending_S02",
                "downstream_step": "S17",
                "discrepancy_ids": discrepancy_ids,
                "notes": f"Parsed point={point}, spread={spread}; no assumption is made about SD versus SE.",
            }
        )
        claims.append(row)

    for raw in targets["intervention_claims"]:
        row = _base_claim()
        row.update(
            {
                "claim_id": raw["claim_id"],
                "claim_family": raw["claim_family"],
                "evidence_layer": "interventional",
                "claim_text": raw["claim_text"],
                "primary_source_location": raw["source"],
                "reported_statistic": raw["reported_statistic"],
                "reported_target": raw["reported_target"],
                "expected_direction": raw["expected_direction"],
                "unit_of_analysis": "paired catalytic-matrix treatment contrast"
                if raw["claim_family"] == "intervention_contrast"
                else "treatment-specific generation trend",
                "sample_scope": "100 assemblies/treatment stated; exact pairing and eligible observations not reported",
                "reproduction_estimand": raw["estimand"],
                "reproduction_criterion": raw["criterion"],
                "inferential_test": "Mann-Whitney for contrasts unless absent; linear regression for time trends",
                "specification_status": "underdetermined_pending_S02"
                if raw["discrepancy_ids"]
                else "testable_with_declared_ambiguity",
                "downstream_step": "S17",
                "discrepancy_ids": raw["discrepancy_ids"],
                "notes": "No causal claim is accepted without later paired/randomized intervention validation.",
            }
        )
        claims.append(row)

    claims.sort(key=lambda item: item["claim_id"])
    return claims


def validate_claims(
    schema: dict[str, Any], targets: dict[str, Any], claims: list[dict[str, str]]
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    columns = schema["columns"]
    ids = [claim["claim_id"] for claim in claims]

    if len(ids) != len(set(ids)):
        errors.append("Claim IDs are not unique")
    pattern = re.compile(schema["id_pattern"])
    for claim_id in ids:
        if not pattern.fullmatch(claim_id):
            errors.append(f"Invalid claim ID: {claim_id}")
    expected_count = schema["validation_contract"]["require_claim_count"]
    if len(claims) != expected_count:
        errors.append(f"Expected {expected_count} claims, found {len(claims)}")

    for claim in claims:
        missing_columns = set(columns) - set(claim)
        extra_columns = set(claim) - set(columns)
        if missing_columns:
            errors.append(
                f"{claim['claim_id']} missing columns: {sorted(missing_columns)}"
            )
        if extra_columns:
            errors.append(
                f"{claim['claim_id']} has extra columns: {sorted(extra_columns)}"
            )
        for field in schema["required_nonempty_fields"]:
            if not str(claim.get(field, "")).strip():
                errors.append(f"{claim['claim_id']} has empty required field {field}")
        for field, allowed in schema["enums"].items():
            if claim.get(field) not in allowed:
                errors.append(
                    f"{claim['claim_id']} has invalid {field}: {claim.get(field)}"
                )
        if "composite pass score" in claim["reproduction_criterion"].lower():
            errors.append(
                f"{claim['claim_id']} improperly defines a composite pass score"
            )

    observed_families = {claim["claim_family"] for claim in claims}
    missing_families = set(schema["required_claim_families"]) - observed_families
    if missing_families:
        errors.append(f"Missing claim families: {sorted(missing_families)}")

    discrepancy_ids = set(schema["discrepancies"])
    referenced: set[str] = set()
    for claim in claims:
        for discrepancy in filter(None, claim["discrepancy_ids"].split(";")):
            referenced.add(discrepancy)
            if discrepancy not in discrepancy_ids:
                errors.append(
                    f"{claim['claim_id']} references unknown discrepancy {discrepancy}"
                )
    missing_references = discrepancy_ids - referenced
    if missing_references:
        errors.append(f"Unreferenced discrepancies: {sorted(missing_references)}")

    target_text = "\n".join(claim["reported_target"] for claim in claims)
    for anchor in schema["required_numeric_anchors"]:
        if anchor not in target_text:
            errors.append(f"Missing required numeric anchor: {anchor}")

    table_count = sum(
        claim["claim_family"] == "intervention_absolute" for claim in claims
    )
    expected_table_count = schema["validation_contract"]["require_table_cell_claims"]
    if table_count != expected_table_count:
        errors.append(
            f"Expected {expected_table_count} Table 1 cell claims, found {table_count}"
        )

    reconciliations = targets["reconciliations"]
    if not reconciliations:
        errors.append("No independent Results/caption reconciliation records")
    reconciled_claim_ids: set[str] = set()
    for record in reconciliations:
        for token in str(record[1]).split(";"):
            if pattern.fullmatch(token):
                reconciled_claim_ids.add(token)
    if len(reconciled_claim_ids) < 10:
        errors.append(
            "Fewer than 10 claims received explicit second-pass reconciliation"
        )

    if any(
        claim["claim_id"] == "E01-C056" and "D05" not in claim["discrepancy_ids"]
        for claim in claims
    ):
        errors.append("Consistency direction conflict is not linked to E01-C056")
    if any(
        claim["claim_id"] == "E01-C016" and "54/73" not in claim["reported_target"]
        for claim in claims
    ):
        errors.append("54/100 versus 54/73 reconciliation is incomplete")

    status_counts = Counter(claim["specification_status"] for claim in claims)
    family_counts = Counter(claim["claim_family"] for claim in claims)
    source_counts = Counter(
        "two_source" if claim["corroborating_source_location"] else "single_source"
        for claim in claims
    )
    return {
        "researchStepId": schema["research_step_id"],
        "schemaVersion": schema["schema_version"],
        "ledgerVersion": schema["ledger_version"],
        "valid": not errors,
        "claimCount": len(claims),
        "familyCounts": dict(sorted(family_counts.items())),
        "specificationStatusCounts": dict(sorted(status_counts.items())),
        "sourceCoverageCounts": dict(sorted(source_counts.items())),
        "explicitlyReconciledClaimCount": len(reconciled_claim_ids),
        "reconciliationRecordCount": len(reconciliations),
        "discrepancyCount": len(discrepancy_ids),
        "errors": errors,
        "warnings": warnings,
    }


def write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def md_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_markdown(
    path: Path,
    schema: dict[str, Any],
    claims: list[dict[str, str]],
    validation: dict[str, Any],
) -> None:
    family_counts = validation["familyCounts"]
    status_counts = validation["specificationStatusCounts"]
    lines = [
        "# E01 forensic claim ledger",
        "",
        f"- **Ledger version:** `{schema['ledger_version']}`",
        f"- **Research step:** `{schema['research_step_id']}`",
        f"- **Claim count:** {len(claims)} independent targets",
        f"- **Schema validation:** {'PASS' if validation['valid'] else 'FAIL'}",
        "- **Interpretation:** This is a preregistration ledger, not a replication result and not a composite pass score.",
        "",
        "## Coverage",
        "",
        "| Claim family | Count |",
        "| --- | ---: |",
    ]
    lines.extend(
        f"| {md_escape(key)} | {value} |" for key, value in family_counts.items()
    )
    lines.extend(
        [
            "",
            "## Specification status",
            "",
            "| Status | Count |",
            "| --- | ---: |",
        ]
    )
    lines.extend(
        f"| `{md_escape(key)}` | {value} |" for key, value in status_counts.items()
    )
    lines.extend(
        [
            "",
            "## Source discrepancies retained for adjudication",
            "",
            "| ID | Kind | Summary |",
            "| --- | --- | --- |",
        ]
    )
    for discrepancy_id, detail in schema["discrepancies"].items():
        lines.append(
            f"| `{discrepancy_id}` | {md_escape(detail['kind'])} | {md_escape(detail['summary'])} |"
        )
    lines.extend(
        [
            "",
            "## Claim targets",
            "",
            "Every row is judged separately in its downstream step. `EXACT`, `DIRECTIONAL`, `NONREPLICATION`, and `UNDERDETERMINED` are claim-level outcomes; no overall score is defined.",
            "",
            "| Claim ID | Family | Claim | Reported target | Expected direction | Unit | Source | Reproduction criterion | Status | Discrepancies |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for claim in claims:
        lines.append(
            "| "
            + " | ".join(
                md_escape(value)
                for value in [
                    f"`{claim['claim_id']}`",
                    claim["claim_family"],
                    claim["claim_text"],
                    claim["reported_target"],
                    claim["expected_direction"],
                    claim["unit_of_analysis"],
                    claim["primary_source_location"],
                    claim["reproduction_criterion"],
                    claim["specification_status"],
                    claim["discrepancy_ids"] or "none",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Field-level machine-readable record",
            "",
            "The companion CSV preserves the additional fields `evidence_layer`, `corroborating_source_location`, `reported_statistic`, `sample_scope`, `reproduction_estimand`, `inferential_test`, `downstream_step`, and `notes` for every claim.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_reconciliation(path: Path, targets: dict[str, Any]) -> None:
    columns = [
        "reconciliation_id",
        "claim_ids",
        "results_or_first_pass",
        "caption_table_or_second_pass",
        "status",
        "resolution",
    ]
    rows = [
        dict(zip(columns, record, strict=True)) for record in targets["reconciliations"]
    ]
    write_csv(path, columns, rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unavailable"


def write_manifest(
    path: Path,
    artifacts_dir: Path,
    schema_path: Path,
    targets_path: Path,
    output_paths: list[Path],
) -> None:
    source_paths = [
        WORKSPACE_ROOT / "input-attachments/MANIFEST.json",
        WORKSPACE_ROOT
        / "input-attachments/ed5486bf-a043-485b-a233-d88d8d123759/_metadata/ATTACHMENT.md",
        WORKSPACE_ROOT
        / "input-attachments/ed5486bf-a043-485b-a233-d88d8d123759/pdf-markdown.md",
        schema_path,
        targets_path,
        Path(__file__).resolve(),
    ]
    figure_dir = (
        WORKSPACE_ROOT
        / "input-attachments/ed5486bf-a043-485b-a233-d88d8d123759/figures"
    )
    source_paths.extend(sorted(figure_dir.glob("figure-*.png")))
    report_path = artifacts_dir / "research_steps/S01/research_step_full_results.md"
    if report_path.exists():
        output_paths.append(report_path)
    entries = []
    for item in sorted(set(source_paths + output_paths), key=lambda p: str(p)):
        if item.exists() and item.is_file() and item != path:
            entries.append(
                {
                    "path": str(item),
                    "sizeBytes": item.stat().st_size,
                    "sha256": sha256(item),
                    "role": "input_or_code" if item in source_paths else "S01_output",
                }
            )
    payload = {
        "schema": "eidosoma.e01.s01.artifact_manifest.v1",
        "researchStepId": "S01",
        "experimentId": "E01",
        "generatedOn": "2026-08-01",
        "repository": str(REPO_ROOT),
        "gitCommit": git_commit(),
        "entries": entries,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build(
    artifacts_dir: Path,
    schema_path: Path = DEFAULT_SCHEMA,
    targets_path: Path = DEFAULT_TARGETS,
) -> dict[str, Any]:
    schema = load_yaml(schema_path)
    targets = load_yaml(targets_path)
    claims = expand_claims(schema, targets)
    validation = validate_claims(schema, targets, claims)
    if not validation["valid"]:
        raise ValueError(
            "Claim-ledger validation failed:\n" + "\n".join(validation["errors"])
        )

    ledger_dir = artifacts_dir / "E01_forensic_replication_bundle/ledgers"
    step_dir = artifacts_dir / "research_steps/S01"
    csv_path = ledger_dir / "claim_ledger.csv"
    md_path = ledger_dir / "claim_ledger.md"
    reconciliation_path = step_dir / "source_reconciliation.csv"
    validation_path = step_dir / "validation_summary.json"
    manifest_path = step_dir / "artifact_manifest.json"

    write_csv(csv_path, schema["columns"], claims)
    write_markdown(md_path, schema, claims, validation)
    write_reconciliation(reconciliation_path, targets)
    step_dir.mkdir(parents=True, exist_ok=True)
    validation_path.write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_manifest(
        manifest_path,
        artifacts_dir,
        schema_path,
        targets_path,
        [csv_path, md_path, reconciliation_path, validation_path],
    )
    return validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")),
    )
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build(
        args.artifacts_dir.resolve(), args.schema.resolve(), args.targets.resolve()
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
