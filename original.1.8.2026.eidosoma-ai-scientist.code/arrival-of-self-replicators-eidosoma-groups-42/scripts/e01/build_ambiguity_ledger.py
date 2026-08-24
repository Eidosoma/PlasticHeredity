#!/usr/bin/env python3
"""Build and validate the E01 S02 ambiguity/discrepancy registry.

The registry is intentionally non-executable while any required value is represented
by an ``UNRESOLVED::`` or ``CONFLICT::`` sentinel. Frozen branch sets are explicit
specifications, not defaults: every branch must later receive its own specification ID.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import statistics
import subprocess
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
DEFAULT_SCHEMA = REPO_ROOT / "configs/e01/ambiguity_schema.yaml"
DEFAULT_TARGETS = REPO_ROOT / "configs/e01/ambiguity_targets.yaml"
DEFAULT_CLAIM_SCHEMA = REPO_ROOT / "configs/e01/claim_schema.yaml"
DEFAULT_CLAIM_LEDGER = (
    Path("/artifacts") / "E01_forensic_replication_bundle/ledgers/claim_ledger.csv"
)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping in {path}")
    return data


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _tokens(value: Any) -> list[str]:
    return [token.strip() for token in str(value or "").split(";") if token.strip()]


def expand_claim_references(expression: Any, valid_ids: list[str]) -> list[str]:
    """Expand ALL, individual IDs, and inclusive ``E01-Cnnn:E01-Cnnn`` ranges."""

    valid_set = set(valid_ids)
    expanded: list[str] = []
    for token in _tokens(expression):
        if token == "ALL":
            candidates = valid_ids
        elif ":" in token:
            start, end = token.split(":", maxsplit=1)
            match_start = re.fullmatch(r"E01-C([0-9]{3})", start)
            match_end = re.fullmatch(r"E01-C([0-9]{3})", end)
            if not match_start or not match_end:
                raise ValueError(f"Invalid claim range: {token}")
            first, last = int(match_start.group(1)), int(match_end.group(1))
            if first > last:
                raise ValueError(f"Descending claim range: {token}")
            candidates = [f"E01-C{number:03d}" for number in range(first, last + 1)]
        else:
            candidates = [token]
        for claim_id in candidates:
            if claim_id not in valid_set:
                raise ValueError(f"Unknown claim reference {claim_id} in {expression}")
            if claim_id not in expanded:
                expanded.append(claim_id)
    return expanded


def _risk(item: dict[str, Any]) -> str:
    return str(
        item.get("risk")
        or (
            f"Silently selecting {item['parameter']} can change the estimand, "
            "numerical result, validity, or reproducibility of linked claims."
        )
    )


def _validation_rule(item: dict[str, Any]) -> str:
    status = str(item["status"])
    parameter = str(item["parameter"])
    owner = str(item["owner"])
    if status in {"UNRESOLVED_REQUIRED", "DEFERRED_EVIDENCE"}:
        return (
            f"Configuration validation must reject {parameter}={item['selected']} "
            f"before owner step {owner} executes."
        )
    if status == "CONFLICT_PRESERVED":
        return (
            f"Every conflict branch for {parameter} must remain separate; validation "
            "must reject an unqualified selection."
        )
    if status == "FROZEN_BRANCH_SET":
        return (
            f"Every branch for {parameter} must receive a distinct specification ID; "
            "no branch may be selected after inspecting outcomes."
        )
    return (
        f"Each run manifest must record exactly this {parameter} value and its "
        f"resolution status {status}."
    )


def expand_items(
    schema: dict[str, Any],
    targets: dict[str, Any],
    claims: list[dict[str, str]],
) -> list[dict[str, str]]:
    claim_ids = [row["claim_id"] for row in claims]
    unresolved_statuses = set(schema["unresolved_statuses"])
    rows: list[dict[str, str]] = []
    for item in targets["items"]:
        affected = expand_claim_references(item["claims"], claim_ids)
        status = str(item["status"])
        row = {
            "ambiguity_id": str(item["id"]),
            "category": str(item["category"]),
            "specification_parameter": str(item["parameter"]),
            "materiality": str(item["materiality"]),
            "source_evidence": str(item["evidence"]),
            "ambiguity_description": str(item["issue"]),
            "admissible_values_or_branches": str(item["candidates"]),
            "primary_spec_value": str(item["selected"]),
            "resolution_status": status,
            "unresolved_flag": str(status in unresolved_statuses).lower(),
            "resolution_basis": str(item["basis"]),
            "downstream_owner_step": str(item["owner"]),
            "affected_claim_ids": ";".join(affected),
            "s01_discrepancy_ids": ";".join(_tokens(item.get("discrepancies"))),
            "risk_if_silent": _risk(item),
            "validation_rule": _validation_rule(item),
            "registry_version": str(schema["registry_version"]),
            "notes": str(item.get("notes") or ""),
        }
        rows.append(row)
    return rows


def validate_rows(
    schema: dict[str, Any],
    targets: dict[str, Any],
    claims: list[dict[str, str]],
    rows: list[dict[str, str]],
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    contract = schema["validation_contract"]
    columns = schema["columns"]
    unresolved_statuses = set(schema["unresolved_statuses"])
    expected_discrepancies = set(schema["s01_discrepancy_ids"])
    claim_ids = [claim["claim_id"] for claim in claims]
    claim_set = set(claim_ids)
    ids = [row["ambiguity_id"] for row in rows]
    parameters = [row["specification_parameter"] for row in rows]

    if len(rows) != contract["require_ambiguity_count"]:
        errors.append(
            f"Expected {contract['require_ambiguity_count']} ambiguities, found {len(rows)}"
        )
    if len(ids) != len(set(ids)):
        errors.append("Ambiguity IDs are not unique")
    if contract["require_unique_parameters"] and len(parameters) != len(
        set(parameters)
    ):
        errors.append("Specification parameters are not unique")

    id_pattern = re.compile(schema["id_pattern"])
    discrepancy_coverage: defaultdict[str, list[str]] = defaultdict(list)
    claim_coverage: defaultdict[str, list[str]] = defaultdict(list)
    no_silent_default_violations: list[str] = []

    for row in rows:
        ambiguity_id = row["ambiguity_id"]
        if set(row) != set(columns):
            errors.append(
                f"{ambiguity_id} column mismatch; missing={sorted(set(columns) - set(row))}, "
                f"extra={sorted(set(row) - set(columns))}"
            )
        if not id_pattern.fullmatch(ambiguity_id):
            errors.append(f"Invalid ambiguity ID: {ambiguity_id}")
        for field in schema["required_nonempty_fields"]:
            if not str(row.get(field, "")).strip():
                errors.append(f"{ambiguity_id} has empty required field {field}")
        for field, allowed in schema["enums"].items():
            if row.get(field) not in allowed:
                errors.append(f"{ambiguity_id} has invalid {field}: {row.get(field)}")

        status = row["resolution_status"]
        expected_unresolved = status in unresolved_statuses
        if row["unresolved_flag"] != str(expected_unresolved).lower():
            errors.append(f"{ambiguity_id} unresolved flag disagrees with status")
        required_prefix = schema["sentinel_prefix_by_status"].get(status)
        if required_prefix and not row["primary_spec_value"].startswith(
            required_prefix
        ):
            errors.append(
                f"{ambiguity_id} status {status} requires prefix {required_prefix}"
            )
        if expected_unresolved and not required_prefix:
            errors.append(f"{ambiguity_id} unresolved status has no sentinel rule")
        if (
            contract["require_exact_unresolved_sentinels"]
            and status in {"UNRESOLVED_REQUIRED", "DEFERRED_EVIDENCE"}
            and row["primary_spec_value"] != f"UNRESOLVED::{ambiguity_id}"
        ):
            errors.append(
                f"{ambiguity_id} must use its exact, traceable unresolved sentinel"
            )
        if (
            contract["require_multivalued_branches"]
            and status in {"FROZEN_BRANCH_SET", "CONFLICT_PRESERVED"}
            and "|" not in row["primary_spec_value"]
        ):
            errors.append(f"{ambiguity_id} does not enumerate multiple branches")
        if status in {
            "PAPER_FIXED",
            "PLAN_FIXED",
            "PROVISIONAL_PRIMARY",
            "RECONCILED",
        } and row["primary_spec_value"].startswith(
            ("UNRESOLVED::", "CONFLICT::", "BRANCH_SET::")
        ):
            errors.append(
                f"{ambiguity_id} resolved status cannot use an unresolved/branch sentinel"
            )
        if not row["primary_spec_value"].strip():
            no_silent_default_violations.append(
                f"{ambiguity_id} has blank primary specification value"
            )

        affected = _tokens(row["affected_claim_ids"])
        unknown_claims = set(affected) - claim_set
        if unknown_claims:
            errors.append(
                f"{ambiguity_id} references unknown claims {sorted(unknown_claims)}"
            )
        for claim_id in affected:
            claim_coverage[claim_id].append(ambiguity_id)
        for discrepancy_id in _tokens(row["s01_discrepancy_ids"]):
            if discrepancy_id not in expected_discrepancies:
                errors.append(
                    f"{ambiguity_id} references unknown discrepancy {discrepancy_id}"
                )
            discrepancy_coverage[discrepancy_id].append(ambiguity_id)

    if set(schema["required_categories"]) - {row["category"] for row in rows}:
        errors.append("One or more required ambiguity categories are absent")
    missing_claims = claim_set - set(claim_coverage)
    if contract["require_all_claims_mapped"] and missing_claims:
        errors.append(f"Claims without ambiguity mappings: {sorted(missing_claims)}")
    minimum = contract["require_minimum_ambiguities_per_claim"]
    thin_claims = {
        claim_id: len(claim_coverage[claim_id])
        for claim_id in claim_ids
        if len(claim_coverage[claim_id]) < minimum
    }
    if thin_claims:
        errors.append(
            f"Claims below minimum ambiguity coverage {minimum}: {thin_claims}"
        )

    missing_discrepancies = expected_discrepancies - set(discrepancy_coverage)
    if contract["require_all_s01_discrepancies_mapped"] and missing_discrepancies:
        errors.append(
            f"S01 discrepancies without ambiguity mappings: {sorted(missing_discrepancies)}"
        )
    if contract["require_each_claim_discrepancy_linked"]:
        for claim in claims:
            for discrepancy_id in _tokens(claim.get("discrepancy_ids")):
                matching = [
                    row
                    for row in rows
                    if claim["claim_id"] in _tokens(row["affected_claim_ids"])
                    and discrepancy_id in _tokens(row["s01_discrepancy_ids"])
                ]
                if not matching:
                    errors.append(
                        f"{claim['claim_id']} discrepancy {discrepancy_id} lacks a "
                        "claim-specific ambiguity link"
                    )
    missing_plan_parameters = set(schema["required_plan_parameters"]) - set(parameters)
    if contract["require_all_plan_parameters_mapped"] and missing_plan_parameters:
        errors.append(
            f"Required plan parameters absent: {sorted(missing_plan_parameters)}"
        )
    if contract["require_no_blank_primary_values"] and no_silent_default_violations:
        errors.extend(no_silent_default_violations)

    known_names = [item["parameter"] for item in schema["known_parameters"]]
    if len(known_names) != len(set(known_names)):
        errors.append("Known specification parameters are not unique")
    overlap = set(known_names) & set(parameters)
    if overlap:
        errors.append(f"Known and ambiguity parameters overlap: {sorted(overlap)}")

    unresolved_rows = [
        row for row in rows if row["resolution_status"] in unresolved_statuses
    ]
    branch_rows = [
        row for row in rows if row["resolution_status"] == "FROZEN_BRANCH_SET"
    ]
    execution_blocking_rows = unresolved_rows + branch_rows
    registry_executable = not execution_blocking_rows
    if contract["require_registry_non_executable_while_blocked"]:
        if execution_blocking_rows and registry_executable:
            errors.append(
                "Registry is executable despite unresolved or unexpanded blockers"
            )
        if not execution_blocking_rows:
            warnings.append(
                "Registry unexpectedly has no unresolved or branch-set blockers"
            )

    coverage_counts = [len(claim_coverage[claim_id]) for claim_id in claim_ids]
    status_counts = Counter(row["resolution_status"] for row in rows)
    category_counts = Counter(row["category"] for row in rows)
    materiality_counts = Counter(row["materiality"] for row in rows)
    owner_counts = Counter(row["downstream_owner_step"] for row in rows)
    unresolved_materiality = Counter(row["materiality"] for row in unresolved_rows)
    no_silent_defaults = not no_silent_default_violations and all(
        row["primary_spec_value"].strip() for row in rows
    )
    if contract["require_no_silent_defaults"] and not no_silent_defaults:
        errors.append("No-silent-default contract failed")

    return {
        "researchStepId": schema["research_step_id"],
        "stepNumber": 2,
        "schemaVersion": schema["schema_version"],
        "ledgerVersion": schema["ledger_version"],
        "registryVersion": schema["registry_version"],
        "valid": not errors,
        "ambiguityCount": len(rows),
        "uniqueParameterCount": len(set(parameters)),
        "knownParameterCount": len(schema["known_parameters"]),
        "registryParameterCount": len(rows) + len(schema["known_parameters"]),
        "statusCounts": dict(sorted(status_counts.items())),
        "categoryCounts": dict(sorted(category_counts.items())),
        "materialityCounts": dict(sorted(materiality_counts.items())),
        "downstreamOwnerCounts": dict(sorted(owner_counts.items())),
        "unresolvedAmbiguityCount": len(unresolved_rows),
        "unresolvedMaterialityCounts": dict(sorted(unresolved_materiality.items())),
        "frozenBranchSetCount": len(branch_rows),
        "executionBlockingParameterCount": len(execution_blocking_rows),
        "claimCoverage": {
            "mappedClaimCount": len(claim_coverage),
            "expectedClaimCount": len(claim_ids),
            "minimumAmbiguitiesPerClaim": min(coverage_counts),
            "medianAmbiguitiesPerClaim": statistics.median(coverage_counts),
            "maximumAmbiguitiesPerClaim": max(coverage_counts),
        },
        "discrepancyCoverage": {
            key: sorted(value) for key, value in sorted(discrepancy_coverage.items())
        },
        "requiredPlanParametersCovered": sorted(
            set(schema["required_plan_parameters"]) & set(parameters)
        ),
        "noSilentDefaults": no_silent_defaults,
        "registryExecutable": registry_executable,
        "errors": errors,
        "warnings": warnings,
    }


def write_csv(path: Path, columns: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def md_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_ambiguity_markdown(
    path: Path,
    schema: dict[str, Any],
    rows: list[dict[str, str]],
    validation: dict[str, Any],
) -> None:
    lines = [
        "# E01 ambiguity and discrepancy ledger",
        "",
        f"- **Ledger version:** `{schema['ledger_version']}`",
        f"- **Specification registry:** `{schema['registry_version']}`",
        f"- **Research step:** `{schema['research_step_id']}`",
        f"- **Ambiguity count:** {len(rows)}",
        f"- **Schema validation:** {'PASS' if validation['valid'] else 'FAIL'}",
        f"- **Registry executable:** `{str(validation['registryExecutable']).lower()}`",
        "- **No-silent-default rule:** Every primary value is fixed, an explicit branch set, or an unresolved/conflict sentinel.",
        "",
        "## Resolution status",
        "",
        "| Status | Count | Meaning |",
        "| --- | ---: | --- |",
    ]
    meanings = {
        "PAPER_FIXED": "Explicit in the supplied paper.",
        "PLAN_FIXED": "Frozen prospectively by FULL_PLAN.",
        "PROVISIONAL_PRIMARY": "Explicit provisional choice; must be logged and sensitivity-audited.",
        "FROZEN_BRANCH_SET": "All listed branches are retained; later runs need distinct specification IDs.",
        "RECONCILED": "Source statements can coexist after preserving the exact scopes or denominators.",
        "CONFLICT_PRESERVED": "Contradictory source interpretations remain separate and block unqualified execution.",
        "UNRESOLVED_REQUIRED": "Required method value is absent and execution must reject the sentinel.",
        "DEFERRED_EVIDENCE": "Resolution requires evidence owned by a later authorized step.",
    }
    for status, count in validation["statusCounts"].items():
        lines.append(f"| `{status}` | {count} | {meanings[status]} |")
    lines.extend(
        [
            "",
            "## Ambiguity items",
            "",
            "| ID | Category | Parameter | Materiality | Status | Primary value or sentinel | Owner | Claims | S01 discrepancies |",
            "| --- | --- | --- | --- | --- | --- | --- | ---: | --- |",
        ]
    )
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['ambiguity_id']}`",
                    md_escape(row["category"]),
                    f"`{md_escape(row['specification_parameter'])}`",
                    md_escape(row["materiality"]),
                    f"`{row['resolution_status']}`",
                    f"`{md_escape(row['primary_spec_value'])}`",
                    f"`{row['downstream_owner_step']}`",
                    str(len(_tokens(row["affected_claim_ids"]))),
                    md_escape(row["s01_discrepancy_ids"] or "none"),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "The companion CSV is authoritative for source evidence, ambiguity descriptions, admissible branches, resolution bases, claim IDs, risks, and validation rules.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_claim_map(
    claims: list[dict[str, str]], rows: list[dict[str, str]]
) -> list[dict[str, Any]]:
    by_claim: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        for claim_id in _tokens(row["affected_claim_ids"]):
            by_claim[claim_id].append(row)
    output: list[dict[str, Any]] = []
    for claim in claims:
        linked = by_claim[claim["claim_id"]]
        unresolved = [row for row in linked if row["unresolved_flag"] == "true"]
        source_discrepancies = _tokens(claim.get("discrepancy_ids"))
        discrepancy_rows = [
            row
            for row in linked
            if set(_tokens(row["s01_discrepancy_ids"])) & set(source_discrepancies)
        ]
        output.append(
            {
                "claim_id": claim["claim_id"],
                "claim_family": claim["claim_family"],
                "s01_specification_status": claim["specification_status"],
                "ambiguity_ids": ";".join(row["ambiguity_id"] for row in linked),
                "unresolved_ambiguity_ids": ";".join(
                    row["ambiguity_id"] for row in unresolved
                ),
                "ambiguity_count": len(linked),
                "unresolved_count": len(unresolved),
                "s01_discrepancy_ids": ";".join(source_discrepancies),
                "linked_discrepancy_ambiguity_ids": ";".join(
                    row["ambiguity_id"] for row in discrepancy_rows
                ),
            }
        )
    return output


def _preservation_class(linked: list[dict[str, str]]) -> str:
    statuses = {row["resolution_status"] for row in linked}
    if "CONFLICT_PRESERVED" in statuses:
        return "source_conflict_preserved"
    if "RECONCILED" in statuses:
        return "scope_preserving_reconciliation"
    if statuses & {"UNRESOLVED_REQUIRED", "DEFERRED_EVIDENCE"}:
        return "explicit_unresolved"
    return "versioned_branch_set"


def build_discrepancy_taxonomy(
    claim_schema: dict[str, Any],
    claims: list[dict[str, str]],
    rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for discrepancy_id, detail in claim_schema["discrepancies"].items():
        linked = [
            row for row in rows if discrepancy_id in _tokens(row["s01_discrepancy_ids"])
        ]
        affected_claims = [
            claim["claim_id"]
            for claim in claims
            if discrepancy_id in _tokens(claim.get("discrepancy_ids"))
        ]
        preservation = _preservation_class(linked)
        if preservation == "scope_preserving_reconciliation":
            handling = "Preserve raw count 54 and report both denominators (100 total; 73 positive)."
        elif preservation == "source_conflict_preserved":
            handling = (
                "Keep each source-supported interpretation as a separately named branch; "
                "reject any unqualified primary value."
            )
        elif preservation == "explicit_unresolved":
            handling = "Retain an unresolved sentinel until the owner step supplies independent evidence."
        else:
            handling = "Run every prospectively frozen branch under a distinct specification ID."
        output.append(
            {
                "discrepancy_id": discrepancy_id,
                "kind": str(detail["kind"]),
                "source_summary": str(detail["summary"]),
                "affected_claim_ids": ";".join(affected_claims),
                "ambiguity_ids": ";".join(row["ambiguity_id"] for row in linked),
                "specification_parameters": ";".join(
                    row["specification_parameter"] for row in linked
                ),
                "resolution_statuses": ";".join(
                    sorted({row["resolution_status"] for row in linked})
                ),
                "preservation_class": preservation,
                "primary_handling": handling,
                "silent_default_prohibited": "true",
                "downstream_owner_steps": ";".join(
                    sorted({row["downstream_owner_step"] for row in linked})
                ),
            }
        )
    return output


def write_registry(
    path: Path,
    schema: dict[str, Any],
    rows: list[dict[str, str]],
    validation: dict[str, Any],
) -> None:
    parameters: list[dict[str, Any]] = []
    for item in schema["known_parameters"]:
        parameters.append(
            {
                "parameter": item["parameter"],
                "value": item["value"],
                "resolutionStatus": item["status"],
                "unresolved": False,
                "sourceEvidence": item["source"],
                "ambiguityId": None,
            }
        )
    for row in rows:
        parameters.append(
            {
                "parameter": row["specification_parameter"],
                "value": row["primary_spec_value"],
                "admissibleValuesOrBranches": row["admissible_values_or_branches"],
                "resolutionStatus": row["resolution_status"],
                "unresolved": row["unresolved_flag"] == "true",
                "sourceEvidence": row["source_evidence"],
                "resolutionBasis": row["resolution_basis"],
                "ownerStep": row["downstream_owner_step"],
                "ambiguityId": row["ambiguity_id"],
                "validationRule": row["validation_rule"],
            }
        )
    payload = {
        "schema": "eidosoma.e01.specification_registry.v1",
        "researchStepId": schema["research_step_id"],
        "experimentId": schema["experiment_id"],
        "registryVersion": schema["registry_version"],
        "ledgerVersion": schema["ledger_version"],
        "generatedOn": "2026-08-01",
        "executionGate": {
            "executable": validation["registryExecutable"],
            "rule": (
                "Reject execution while any parameter has unresolved=true. "
                "Expand BRANCH_SET values to separate immutable specification IDs; "
                "never choose a branch from outcome data."
            ),
            "unresolvedParameterCount": validation["unresolvedAmbiguityCount"],
            "unexpandedBranchSetCount": validation["frozenBranchSetCount"],
            "executionBlockingParameterCount": validation[
                "executionBlockingParameterCount"
            ],
            "noSilentDefaults": validation["noSilentDefaults"],
        },
        "statusDefinitions": {
            "PAPER_FIXED": "Explicit in supplied paper.",
            "PLAN_FIXED": "Prospectively fixed by FULL_PLAN.",
            "PROVISIONAL_PRIMARY": "Explicit primary choice requiring sensitivity audit.",
            "FROZEN_BRANCH_SET": "Prospective alternatives; each gets a specification ID.",
            "RECONCILED": "Scopes or denominators preserved without contradiction.",
            "CONFLICT_PRESERVED": "Contradictory source branches; unqualified use rejected.",
            "UNRESOLVED_REQUIRED": "Required value absent; execution rejected.",
            "DEFERRED_EVIDENCE": "Resolution assigned to a later authorized step.",
        },
        "parameters": sorted(parameters, key=lambda item: item["parameter"]),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


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
    input_paths: list[Path],
    output_paths: list[Path],
) -> None:
    report = artifacts_dir / "research_steps/S02/research_step_full_results.md"
    if report.exists():
        output_paths.append(report)
    entries: list[dict[str, Any]] = []
    inputs = set(input_paths)
    for item in sorted(
        set(input_paths + output_paths), key=lambda candidate: str(candidate)
    ):
        if item.exists() and item.is_file() and item != path:
            entries.append(
                {
                    "path": str(item),
                    "sizeBytes": item.stat().st_size,
                    "sha256": sha256(item),
                    "role": "input_or_code" if item in inputs else "S02_output",
                }
            )
    payload = {
        "schema": "eidosoma.e01.s02.artifact_manifest.v1",
        "researchStepId": "S02",
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
    claim_schema_path: Path = DEFAULT_CLAIM_SCHEMA,
    claim_ledger_path: Path | None = None,
) -> dict[str, Any]:
    schema = load_yaml(schema_path)
    targets = load_yaml(targets_path)
    claim_schema = load_yaml(claim_schema_path)
    if claim_ledger_path is None:
        claim_ledger_path = (
            artifacts_dir / "E01_forensic_replication_bundle/ledgers/claim_ledger.csv"
        )
    claims = load_csv(claim_ledger_path)
    rows = expand_items(schema, targets, claims)
    validation = validate_rows(schema, targets, claims, rows)
    if not validation["valid"]:
        raise ValueError(
            "Ambiguity-ledger validation failed:\n" + "\n".join(validation["errors"])
        )

    ledger_dir = artifacts_dir / "E01_forensic_replication_bundle/ledgers"
    registry_dir = artifacts_dir / "E01_forensic_replication_bundle/specifications"
    step_dir = artifacts_dir / "research_steps/S02"
    ledger_csv = ledger_dir / "ambiguity_ledger.csv"
    ledger_md = ledger_dir / "ambiguity_ledger.md"
    discrepancy_csv = ledger_dir / "discrepancy_taxonomy.csv"
    registry_yaml = registry_dir / "specification_registry.yaml"
    claim_map_csv = step_dir / "claim_ambiguity_map.csv"
    validation_json = step_dir / "validation_summary.json"
    manifest_json = step_dir / "artifact_manifest.json"

    claim_map = build_claim_map(claims, rows)
    discrepancy_rows = build_discrepancy_taxonomy(claim_schema, claims, rows)
    write_csv(ledger_csv, schema["columns"], rows)
    write_ambiguity_markdown(ledger_md, schema, rows, validation)
    write_csv(
        discrepancy_csv,
        [
            "discrepancy_id",
            "kind",
            "source_summary",
            "affected_claim_ids",
            "ambiguity_ids",
            "specification_parameters",
            "resolution_statuses",
            "preservation_class",
            "primary_handling",
            "silent_default_prohibited",
            "downstream_owner_steps",
        ],
        discrepancy_rows,
    )
    write_csv(
        claim_map_csv,
        [
            "claim_id",
            "claim_family",
            "s01_specification_status",
            "ambiguity_ids",
            "unresolved_ambiguity_ids",
            "ambiguity_count",
            "unresolved_count",
            "s01_discrepancy_ids",
            "linked_discrepancy_ambiguity_ids",
        ],
        claim_map,
    )
    write_registry(registry_yaml, schema, rows, validation)
    step_dir.mkdir(parents=True, exist_ok=True)
    validation_json.write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    input_paths = [
        WORKSPACE_ROOT / "AGENTS.md",
        WORKSPACE_ROOT / "FULL_PLAN.md",
        WORKSPACE_ROOT / "RESEARCH_PLAN.md",
        WORKSPACE_ROOT / "input-attachments/MANIFEST.json",
        WORKSPACE_ROOT
        / "input-attachments/ed5486bf-a043-485b-a233-d88d8d123759/_metadata/ATTACHMENT.md",
        WORKSPACE_ROOT
        / "input-attachments/ed5486bf-a043-485b-a233-d88d8d123759/pdf-markdown.md",
        artifacts_dir / "research_steps/S01/research_step_full_results.md",
        claim_ledger_path,
        artifacts_dir / "research_steps/S01/source_reconciliation.csv",
        schema_path,
        targets_path,
        claim_schema_path,
        Path(__file__).resolve(),
    ]
    output_paths = [
        ledger_csv,
        ledger_md,
        discrepancy_csv,
        registry_yaml,
        claim_map_csv,
        validation_json,
    ]
    write_manifest(manifest_json, artifacts_dir, input_paths, output_paths)
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
    parser.add_argument("--claim-schema", type=Path, default=DEFAULT_CLAIM_SCHEMA)
    parser.add_argument("--claim-ledger", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build(
        args.artifacts_dir.resolve(),
        args.schema.resolve(),
        args.targets.resolve(),
        args.claim_schema.resolve(),
        args.claim_ledger.resolve() if args.claim_ledger else None,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
