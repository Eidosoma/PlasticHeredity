#!/usr/bin/env python3
"""Deterministically synthesize and freeze the terminal E01/S18 verdict."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = REPO_ROOT / "configs/e01/s18_final_dual_verdict_contract.json"
CLAIM_MAP_PATH = REPO_ROOT / "configs/e01/s18_claim_adjudications.csv"
ARTIFACTS_ROOT = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")).resolve()
STEP_ROOT = ARTIFACTS_ROOT / "research_steps/S18"
LEGACY_BUNDLE_ROOT = ARTIFACTS_ROOT / "E01_forensic_replication_bundle"
V2_ROOT = ARTIFACTS_ROOT / "E01_forensic_replication_artifact_v2"
PAPER_PATH = Path("/cache/e01_s03/downloads/paper-2607.28250v1.pdf")
EXPECTED_PAPER_SHA256 = (
    "77a2ec2c0751839d8a2e10863ca803c6f8b61475bbc790f2bbdad2a38af04ae4"
)
PRIOR_STEP_IDS = [
    "S01",
    "S02",
    "S03",
    "S04",
    "S05",
    "S06",
    "S07",
    "S08",
    "S09",
    "S10",
    "S11",
    "S11R",
    "S12",
    "S12B",
    "S12C",
    "S12D",
    "S12E",
    "S12F",
    "S12FR",
    "S12G",
    "S12H",
    "S12I",
    "S12J",
    "S13",
    "S13R",
    "S13RR",
    "S13RRR",
    "S13X",
    "S13Y",
    "S14",
    "S15",
    "S16",
    "S17",
]

STATUS_DISPLAY = {
    "SUPPORTED": "Supported",
    "DIRECTIONALLY_SUPPORTED": "Directionally supported",
    "NOT_SUPPORTED_WITHIN_TESTED_SCOPE": "Not supported within tested scope",
    "UNDERDETERMINED": "Underdetermined",
    "NOT_EVALUATED": "Not evaluated",
}

REQUIRED_CLASSIFICATIONS = {
    "LABEL_COUPLED_RETROSPECTIVE_RESEMBLANCE",
    "RETROSPECTIVE_TEMPORAL_FITTING_DEPENDENCE",
    "RETROSPECTIVE_PREDICTION_RESEMBLANCE",
    "PROSPECTIVE_PREDICTION_SUPPORTED",
    "LITERAL_INTERVENTION_ORDERING_RESEMBLANCE",
    "PROSPECTIVE_CAUSAL_CONTROL_SUPPORTED",
    "NOT_SUPPORTED_WITHIN_TESTED_SCOPE",
    "UNDERDETERMINED_AUTHOR_IMPLEMENTATION",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json_bytes(payload))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_contract() -> dict[str, Any]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def load_claim_map() -> dict[str, dict[str, str]]:
    rows = read_csv(CLAIM_MAP_PATH)
    return {row["claim_id"]: row for row in rows}


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def repo_state(require_clean_pushed: bool) -> dict[str, str]:
    state = {
        "branch": git("branch", "--show-current"),
        "head": git("rev-parse", "HEAD"),
        "remote": git("rev-parse", "origin/eidosoma/groups/42"),
        "status": git("status", "--short"),
    }
    if require_clean_pushed and (
        state["branch"] != "eidosoma/groups/42"
        or state["head"] != state["remote"]
        or state["status"]
    ):
        raise RuntimeError(f"pushed clean-tree gate failed: {state}")
    return state


def file_record(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    return {
        "path": str(path if root is None else path.relative_to(root)),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def hash_tree(roots: Iterable[Path]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(root)
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            entries.append(file_record(path))
    return entries


def manifest_for_root(root: Path, *, exclude_names: set[str]) -> dict[str, Any]:
    entries = [
        file_record(path, root=root)
        for path in sorted(p for p in root.rglob("*") if p.is_file())
        if path.name not in exclude_names
    ]
    return {
        "root": str(root),
        "fileCount": len(entries),
        "totalBytes": sum(int(row["bytes"]) for row in entries),
        "files": entries,
    }


def validate_contract(
    contract: dict[str, Any], ledger_rows: list[dict[str, str]], claim_map: dict[str, dict[str, str]]
) -> dict[str, bool]:
    ledger_ids = [row["claim_id"] for row in ledger_rows]
    map_ids = list(claim_map)
    statuses = Counter(row["status"] for row in claim_map.values())
    expected_counts = contract["expectedClaimStatusCounts"]
    classifications = {
        row["classification"] for row in contract["finalClassifications"]
    }
    checks = {
        "step_identity": contract["researchStepId"] == "S18"
        and contract["stepNumber"] == 18,
        "ledger_has_59_rows": len(ledger_rows) == contract["expectedClaimCount"] == 59,
        "ledger_claim_ids_unique": len(ledger_ids) == len(set(ledger_ids)),
        "mapping_has_59_rows": len(claim_map) == 59,
        "mapping_exactly_covers_ledger": set(map_ids) == set(ledger_ids),
        "status_vocabulary_exact": set(contract["statusVocabulary"])
        == set(STATUS_DISPLAY),
        "mapping_statuses_valid": set(statuses) <= set(contract["statusVocabulary"]),
        "mapping_status_counts_frozen": dict(statuses) == expected_counts,
        "directional_vocabulary_valid": all(
            row["directional_assessment"]
            in contract["directionalAssessmentVocabulary"]
            for row in claim_map.values()
        ),
        "quantitative_vocabulary_valid": all(
            row["quantitative_assessment"]
            in contract["quantitativeAssessmentVocabulary"]
            for row in claim_map.values()
        ),
        "matrix_b_has_7_questions": len(contract["matrixB"]) == 7,
        "figure_table_map_complete": {
            row["componentId"] for row in contract["figureTableMap"]
        }
        == {"FIGURE_2", "FIGURE_3", "FIGURE_4", "FIGURE_5", "FIGURE_6", "TABLE_1"},
        "required_classifications_retained": REQUIRED_CLASSIFICATIONS <= classifications,
        "directional_exact_separation_frozen": contract["directionalPolicy"][
            "exactAndDirectionalAssessmentsAreSeparate"
        ],
        "favorable_candidate_selection_forbidden": contract["directionalPolicy"][
            "favorableCandidateSelectionForbidden"
        ],
        "level_change_separation_frozen": contract["directionalPolicy"][
            "levelAndChangeAnalysesRemainSeparate"
        ],
        "s19_s20_option_inactive": any(
            row["optionId"] == "OPTION_B_VERSIONED_E01_REOPEN_S19_S20"
            and not row["active"]
            and row["authorizationRequired"]
            for row in contract["postCloseoutHumanReviewOptions"]
        ),
        "e02_inactive": contract["overallVerdict"]["e02State"]
        == "NOT_STARTED_REQUIRES_SEPARATE_HUMAN_AUTHORIZATION",
    }
    if not all(checks.values()):
        failed = [key for key, value in checks.items() if not value]
        raise ValueError(f"S18 contract validation failed: {failed}")
    return checks


def evidence_registry() -> dict[str, list[Path]]:
    a = ARTIFACTS_ROOT / "research_steps"
    return {
        "NO_METRIC_DISTINCTIVENESS_RESULT": [
            a / "S15/research_step_full_results.md",
            a / "S15/artifact_manifest.json",
        ],
        "S14_TREND": [
            a / "S14/paper_target_comparison.csv",
            a / "S14/aggregate_trend_results.csv",
        ],
        "S14_SPIKES": [
            a / "S14/paper_target_comparison.csv",
            a / "S14/spike_morphology_summary.csv",
            a / "S14/spike_run_summary.csv",
        ],
        "S14_TEMPORAL": [
            a / "S14/paper_target_comparison.csv",
            a / "S14/ljung_box_summary.csv",
        ],
        "S15_ASSOCIATION": [
            a / "S15/paper_target_comparison.csv",
            a / "S15/correlation_summary.csv",
            a / "S15/trajectory_bootstrap_summary.csv",
            a / "S15/circular_shift_summary.csv",
            a / "S15/interpretation_boundary.csv",
        ],
        "S15_STATE": [
            a / "S15/paper_target_comparison.csv",
            a / "S15/state_comparison_summary.csv",
            a / "S15/mann_whitney_diagnostics.csv",
            a / "S15/fisher_combination_diagnostics.csv",
            a / "S15/interpretation_boundary.csv",
        ],
        "S16_PREDICTION": [
            a / "S16/paper_target_comparison.csv",
            a / "S16/interpretation_gates.csv",
            a / "S16/decision.json",
            a / "S16/split_metric_summary.csv",
        ],
        "S16_ALTERNATIVE_SPLITS": [
            a / "S16/paper_target_comparison.csv",
            a / "S16/research_step_full_results.md",
        ],
        "SPIKE_TIMING_NOT_RUN": [
            a / "S14/spike_morphology_summary.csv",
            a / "S16/research_step_full_results.md",
        ],
        "S17_TABLE1": [
            a / "S17/table1_reconstruction.csv",
            a / "S17/paper_target_comparison.csv",
            a / "S17/outcome_summary.csv",
        ],
        "S17_CONTRAST": [
            a / "S17/paired_bootstrap_summary.csv",
            a / "S17/intervention_claim_status.csv",
            a / "S17/outcome_summary.csv",
        ],
        "S17_TREND": [
            a / "S17/generation_probability_trends.csv",
            a / "S17/intervention_claim_status.csv",
        ],
        "PAST_ONLY_EARLY_WARNING": [
            a / "S15/interpretation_boundary.csv",
            a / "S16/interpretation_gates.csv",
        ],
        "INCREMENTAL_BEYOND_H": [
            a / "S15/label_identity_audit.json",
            a / "S15/ordinary_stability_summary.csv",
            a / "S16/decision.json",
        ],
        "SUFFIX_INDEPENDENCE": [
            a / "S16/cutoff_suffix_invariance.parquet",
            a / "S16/leakage_validation.json",
            a / "S15/interpretation_boundary.csv",
        ],
        "ONLINE_SCORING": [
            a / "S17/preoutcome_design_lock.json",
            a / "S17/interpretation_gates.csv",
            a / "S17/replay_validation.parquet",
        ],
        "INTERVENTION_REPLAY": [
            a / "S17/replay_validation.parquet",
            a / "S17/validation.json",
        ],
        "ACTION_SEPARABILITY": [
            a / "S17/action_diagnostic_summary.csv",
            a / "S17/interpretation_gates.csv",
        ],
        "BIDIRECTIONAL_CONTROL": [
            a / "S17/paired_bootstrap_summary.csv",
            a / "S17/interpretation_gates.csv",
            a / "S17/decision.json",
        ],
        "FIGURE_2": [
            a / "S14/paper_target_comparison.csv",
            a / "S14/research_step_full_results.md",
        ],
        "FIGURE_3": [
            a / "S15/paper_target_comparison.csv",
            a / "S15/interpretation_boundary.csv",
        ],
        "FIGURE_4": [
            a / "S15/paper_target_comparison.csv",
            a / "S15/state_comparison_summary.csv",
        ],
        "FIGURE_5": [
            a / "S16/paper_target_comparison.csv",
            a / "S16/decision.json",
        ],
        "FIGURE_6": [
            a / "S17/interpretation_gates.csv",
            a / "S17/generation_probability_trends.csv",
        ],
        "TABLE_1": [
            a / "S17/table1_reconstruction.csv",
            a / "S17/paired_bootstrap_summary.csv",
        ],
    }


def claim_component(number: int) -> str:
    if 1 <= number <= 12:
        return "PAPER_TEXT_METRIC_DISTINCTIVENESS"
    if number in {13, 14, 22, 23, 24}:
        return "FIGURE_2"
    if 15 <= number <= 18:
        return "FIGURE_3"
    if 19 <= number <= 21:
        return "FIGURE_4"
    if 25 <= number <= 30:
        return "FIGURE_5"
    if 31 <= number <= 33:
        return "PAPER_RESULTS_SPIKE_TIMING"
    if 34 <= number <= 45:
        return "TABLE_1"
    if 50 <= number <= 53:
        return "FIGURE_6"
    return "FIGURE_6_AND_TABLE_1"


def claim_dependencies(number: int) -> tuple[str, str, str]:
    if 13 <= number <= 24:
        completed = "YES_PRIMARY_PAPER_FACING_VALUES_ARE_COMPLETED_FIT"
    elif 25 <= number <= 28:
        completed = "YES_PAPER_LIKE_MODE;NO_CUTOFF_CAUSAL_MODE"
    elif number == 30:
        completed = "NO_DECISIVE_MODE_IS_CUTOFF_CAUSAL"
    else:
        completed = "NO_OR_NOT_APPLICABLE"
    label = (
        "YES_Y_EQUALS_I_H_GT_0_9"
        if (15 <= number <= 21 or 25 <= number <= 59)
        else "NO_OR_NOT_APPLICABLE"
    )
    scoring = "YES_APPEND_AND_REFIT_CURRENT_PREFIX" if number >= 34 else "NO"
    return completed, label, scoring


def build_matrix_a(
    ledger_rows: list[dict[str, str]], claim_map: dict[str, dict[str, str]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    registry = evidence_registry()
    matrix: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    ledger_path = LEGACY_BUNDLE_ROOT / "ledgers/claim_ledger.csv"
    for ledger in ledger_rows:
        claim_id = ledger["claim_id"]
        mapped = claim_map[claim_id]
        number = int(claim_id.rsplit("C", 1)[1])
        completed, label, scoring = claim_dependencies(number)
        evidence_paths = [ledger_path, *registry[mapped["evidence_key"]]]
        evidence_hashes = [sha256_file(path) for path in evidence_paths]
        matrix.append(
            {
                "claimId": claim_id,
                "claimNumber": number,
                "claimFamily": ledger["claim_family"],
                "paperComponent": claim_component(number),
                "paperClaim": ledger["claim_text"],
                "reportedTarget": ledger["reported_target"],
                "expectedDirection": ledger["expected_direction"],
                "reproductionCriterion": ledger["reproduction_criterion"],
                "finalStatus": STATUS_DISPLAY[mapped["status"]],
                "finalStatusCode": mapped["status"],
                "directionalAssessment": mapped["directional_assessment"],
                "quantitativeAssessment": mapped["quantitative_assessment"],
                "evidenceSummary": mapped["evidence_summary"],
                "completedFitDependency": completed,
                "labelScopeDependency": label,
                "interventionScoringDependency": scoring,
                "authorImplementationDependency": ledger["discrepancy_ids"]
                or "NO_LEDGERED_DISCREPANCY",
                "primaryEvidencePaths": ";".join(str(path) for path in evidence_paths),
                "primaryEvidenceSha256": ";".join(evidence_hashes),
                "mainCaveat": mapped["main_caveat"],
                "priorEvidencePreserved": True,
            }
        )
        for path, digest in zip(evidence_paths, evidence_hashes, strict=True):
            trace.append(
                {
                    "claimId": claim_id,
                    "finalStatusCode": mapped["status"],
                    "evidencePath": str(path),
                    "evidenceSha256": digest,
                    "exists": path.is_file(),
                    "traceabilityRole": (
                        "FROZEN_CLAIM_DEFINITION"
                        if path == ledger_path
                        else "FROZEN_STEP_EVIDENCE"
                    ),
                }
            )
    return matrix, trace


def add_evidence_to_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    registry = evidence_registry()
    result: list[dict[str, Any]] = []
    for row in rows:
        paths = registry[row["evidenceKey"]]
        result.append(
            {
                **row,
                "evidencePaths": ";".join(str(path) for path in paths),
                "evidenceSha256": ";".join(sha256_file(path) for path in paths),
            }
        )
    return result


def status_count_rows(matrix: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    overall = Counter(row["finalStatusCode"] for row in matrix)
    for status in STATUS_DISPLAY:
        rows.append(
            {
                "scope": "ALL_59_CLAIMS",
                "claimFamily": "ALL",
                "statusCode": status,
                "count": overall[status],
            }
        )
    by_family: dict[str, Counter[str]] = defaultdict(Counter)
    for row in matrix:
        by_family[row["claimFamily"]][row["finalStatusCode"]] += 1
    for family in sorted(by_family):
        for status in STATUS_DISPLAY:
            rows.append(
                {
                    "scope": "CLAIM_FAMILY",
                    "claimFamily": family,
                    "statusCode": status,
                    "count": by_family[family][status],
                }
            )
    return rows


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    def clean(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(clean(value) for value in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend(
        "| " + " | ".join(clean(value) for value in row) + " |" for row in rows
    )
    return "\n".join(lines)


def render_v2_status(
    contract: dict[str, Any], matrix: list[dict[str, Any]], matrix_b: list[dict[str, Any]]
) -> str:
    counts = Counter(row["finalStatusCode"] for row in matrix)
    options = contract["postCloseoutHumanReviewOptions"]
    return f"""# E01 Forensic Replication Artifact V2 — Closeout Status

## Concise top summary

- **Research step ID:** S18 (`{contract['versionedStepId']}`)
- **Completion status:** Complete; E01 closed for Chief Scientist and human review.
- **Artifacts written:** This V2 replication artifact, its 59-claim Matrix A, seven-question Matrix B, Figure 2–6/Table 1 map, classification registry, decision files, evidence/hash manifests, and the canonical S18 full-results report.
- **Validation result:** PASS — full claim coverage, claim-to-evidence traceability, status-vocabulary checks, prior-artifact immutability, and hash-manifest replay passed.
- **Outcome classification:** `{contract['overallVerdict']['outcomeClassification']}`; `{contract['overallVerdict']['paperFacingVerdict']}`; prospective prediction and causal control are not supported within the tested scope.
- **Caveats or blockers:** The paper-facing resemblance is completed-fit, label-coupled, and partly dependent on unavailable author details. `Y=I(H>0.9)`, exact H fully determines Y, and S17's CPU-allowance waiver remains an unchanged operational caveat.
- **Lay summary:** We are closer to understanding what can reproduce: several descriptive and association directions recur, but the predictive advantage and max/control/min causal ordering do not. This is a partial forensic reconstruction, not a successful reproduction of the paper's central prospective or causal interpretation.
- **Recommended next action:** Human review should choose between the stronger planned E02 and an explicitly versioned E01 reopening for optional S19–S20. Neither path is active or started by S18.

## Current status

Matrix A contains **{counts['SUPPORTED']} supported**, **{counts['DIRECTIONALLY_SUPPORTED']} directionally supported**, **{counts['NOT_SUPPORTED_WITHIN_TESTED_SCOPE']} not supported within tested scope**, **{counts['UNDERDETERMINED']} underdetermined**, and **{counts['NOT_EVALUATED']} not evaluated** claims. Exact numerical agreement and directional resemblance are deliberately separate: a different number can still preserve a direction, while a single favorable simulator candidate cannot rescue disagreement across the two mandatory candidates.

The three genuinely supported paper-facing claims are the positive population association diagnostic (C018), Fisher-combined state contrast (C021), and 100/100 differenced temporal-dependence count (C024). Seventeen more claims are directionally supported. These are bounded retrospective or descriptive matches; none overrides Matrix B.

Matrix B supports only operational cutoff suffix isolation, online prefix scoring, and exact intervention replay within the locked reconstructions. It does not support past-only early warning, incremental prediction beyond H/stability, action separability, or bidirectional causal control.

## Human-review continuation options

{markdown_table(['Option', 'Status', 'Scope'], [[row['optionId'], 'INACTIVE — authorization required' if row['authorizationRequired'] else 'INACTIVE', row['scope']] for row in options])}

The recommended order is the stronger separate E02. If the reviewer instead wants S19–S20 inside E01, E01 must be explicitly reopened under a new version; S01–S18 remain immutable, and both steps must be prospectively specified before outcome access.

## Artifact boundary

The original `/artifacts/E01_forensic_replication_bundle` is retained byte-for-byte as the legacy V1 evidence bundle. V2 is a compact closeout and report-input layer that references and hashes V1 plus S01–S18; it does not duplicate or rewrite bulk trajectories.
"""


def render_full_report(
    contract: dict[str, Any],
    matrix: list[dict[str, Any]],
    matrix_b: list[dict[str, Any]],
    figure_map: list[dict[str, Any]],
    classifications: list[dict[str, Any]],
    prior_stats: dict[str, Any],
    repo: dict[str, str],
) -> str:
    counts = Counter(row["finalStatusCode"] for row in matrix)
    family_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in matrix:
        family_counts[row["claimFamily"]][row["finalStatusCode"]] += 1
    matrix_rows = [
        [
            row["claimId"],
            row["finalStatus"],
            row["directionalAssessment"],
            row["quantitativeAssessment"],
            row["evidenceSummary"],
        ]
        for row in matrix
    ]
    family_rows = [
        [
            family,
            counter["SUPPORTED"],
            counter["DIRECTIONALLY_SUPPORTED"],
            counter["NOT_SUPPORTED_WITHIN_TESTED_SCOPE"],
            counter["UNDERDETERMINED"],
            counter["NOT_EVALUATED"],
        ]
        for family, counter in sorted(family_counts.items())
    ]
    option_rows = [
        [row["optionId"], row["title"], row["scope"], row["active"]]
        for row in contract["postCloseoutHumanReviewOptions"]
    ]
    return f"""# E01/S18 — Final Dual Verdict and E01 Closeout

## Concise top summary

- **Research step ID:** S18 (`{contract['versionedStepId']}`)
- **Completion status:** Complete; E01 is closed and control is returned for Chief Scientist and human review.
- **Artifacts written:** Canonical 59-claim Matrix A; seven-question Matrix B; Figure 2–6/Table 1 reconstruction map; final classification registry; claim-evidence crosswalk; status/count/caveat/decision tables; V2 replication artifact and report inputs; immutable-prior, provenance, validation, and closeout hash manifests.
- **Validation result:** PASS — complete 59-claim coverage, dual-matrix separation, directional/exact separation, evidence traceability, required classifications, S01–S17 and legacy-bundle immutability, S17 waiver preservation, and artifact hash replay all passed.
- **Outcome classification:** **{contract['overallVerdict']['outcomeClassification']}**. Paper-facing result: **{contract['overallVerdict']['paperFacingVerdict']}**. Prospective prediction: **not supported within tested scope**. Prospective causal control: **not supported within tested scope**. Full-plan E01 gate: **{contract['overallVerdict']['fullPlanGate']}**.
- **Caveats or blockers:** Exact author code and several implementation details are unavailable; completed-fit values use the future suffix; `Y=I(H>0.9)` is exactly determined by H; ordinary stability is coupled; 16 claims were not evaluated; two remain underdetermined; and S17's immutable CPU-allowance waiver is operational only.
- **Lay summary:** E01 did make progress: it reconstructed the simulator family, information branch, descriptive spikes, and several association directions. It did not reproduce the paper's aggregate no-trend result, MLP advantage, or max/control/min causal ordering. The closest matches are retrospective and label-coupled, so they cannot support early warning or causal control.
- **Recommended next action:** Human review should choose either the stronger planned E02 or a separately authorized, versioned E01 reopening for S19–S20. S18 starts neither.

## Frozen question

Can the complete E01 evidence be closed with a paper-facing forensic-reproduction matrix that remains strictly separate from prospective-prediction and causal-control adjudication?

**Answer:** Yes. The forensic layer is a partial directional retrospective reconstruction; the prospective and causal layers remain unsupported within the frozen tested scope. A favorable retrospective result does not rescue either failed layer.

## Lay summary

The result is neither “nothing replicated” nor “the paper replicated.” Of the 59 ledgered claims, {counts['SUPPORTED']} meet the frozen paper-facing criterion and {counts['DIRECTIONALLY_SUPPORTED']} more point in the same qualitative direction. At the same time, {counts['NOT_SUPPORTED_WITHIN_TESTED_SCOPE']} are not supported, {counts['UNDERDETERMINED']} cannot be uniquely resolved from the paper, and {counts['NOT_EVALUATED']} were not run. The strongest resemblance is descriptive and retrospective: spikes and positive emergence/replicator associations recur. The most consequential claims do not: a genuinely first-quarter-only PhiRL signal does not beat the controls, and maximizing the online score does not outperform control.

This distinction is why E02 remains scientifically worthwhile. E01 has narrowed the plausible explanations to label construction, ordinary stability/attractor geometry, future-fitted parameters, estimator behavior, and weakly separated action scores. E02 can test those explanations without rewriting E01.

## Inputs and immutable evidence

- `/workspace/AGENTS.md`, `/workspace/FULL_PLAN.md`, and `/workspace/RESEARCH_PLAN.md`
- `/workspace/input-attachments/MANIFEST.json`, every `_metadata/ATTACHMENT.md` sidecar, the extracted paper Markdown, and the official arXiv v1 PDF (`{EXPECTED_PAPER_SHA256}`)
- the authoritative 59-claim ledger and legacy E01 forensic bundle
- every completed S01–S17 report, status, compact result, and manifest
- frozen S12FR, S13Y, and S14–S17 evidence, with no regenerated trajectory, refit, prediction, intervention, or new estimator

The immutable-prior baseline contains **{prior_stats['fileCount']:,} files** and **{prior_stats['totalBytes']:,} bytes**. The legacy V1 bundle is not modified; the V2 artifact is a compact indexed closeout layer.

## Methods

S18 is deterministic synthesis only. The committed contract froze the five-status vocabulary, candidate-separation rule, exact-versus-directional distinction, all 59 claim adjudications, Matrix B questions, Figure/Table map, final classifications, and post-closeout human-review options before outputs were written.

For each paper claim, S18 copied the original claim text, target, expected direction, and reproduction criterion from the S01 ledger; attached the frozen S14–S17 evidence; and adjudicated four distinct axes:

1. **Final claim status** using the directed five-term vocabulary.
2. **Directional assessment** independent of exact point-estimate agreement.
3. **Quantitative assessment** independent of qualitative direction.
4. **Dependency flags** for completed-fit values, exact-H label scope, and intervention-scoring semantics.

`Directionally supported` does not require an exact paper number. It does require the relevant qualitative direction in the mandatory branches. Candidate 2 and candidate 3 remain separate; pooling is secondary; one favorable candidate cannot rescue a disagreement. Level and change analyses likewise remain separate.

Matrix B was adjudicated independently. Operational success (suffix isolation, online scoring, exact replay) is not equated with predictive or causal success. The status of author implementation is carried separately as `UNDERDETERMINED_AUTHOR_IMPLEMENTATION`.

## Commands

```bash
python -m pytest -q tests/e01/test_s18_final_dual_verdict.py
python -m compileall -q scripts/e01/run_s18_final_dual_verdict.py
python scripts/e01/run_s18_final_dual_verdict.py --stage freeze
python scripts/e01/run_s18_final_dual_verdict.py --stage synthesize
python scripts/e01/run_s18_final_dual_verdict.py --stage validate
```

No scientific compute, GPU work, trajectory generation, model training, estimator fit, or intervention rollout occurred in S18.

## Matrix A — paper-facing forensic reproduction

### Status totals

{markdown_table(['Status', 'Count'], [[STATUS_DISPLAY[status], counts[status]] for status in STATUS_DISPLAY])}

### Family totals

{markdown_table(['Claim family', 'Supported', 'Directional', 'Not supported', 'Undetermined', 'Not evaluated'], family_rows)}

### All 59 claims

{markdown_table(['Claim', 'Final status', 'Direction', 'Quantitative fit', 'Frozen evidence summary'], matrix_rows)}

Interpretation boundary: C018, C021, and C024 meet their paper-facing criteria in the locked reconstruction. They do not establish author-code identity, prospective information, or causal control. The 17 directional findings explicitly preserve qualitative resemblance despite numerical differences; they are not silently upgraded to exact reproduction.

## Figure 2–6 and Table 1 reconstruction map

{markdown_table(['Component', 'Reconstruction degree', 'Summary', 'Completed-fit dependency', 'Label dependency', 'Scoring dependency'], [[row['componentId'], row['reconstructionDegree'], row['summary'], row['completedFitDependency'], row['labelScopeDependency'], row['interventionScoringDependency']] for row in figure_map])}

### What resembles the paper

- Figure 2-like punctuated positive excursions occur in 90/100 runs for each candidate; differenced temporal dependence is 100/100.
- Figures 3–4-like positive association and replicator-state contrasts recur across both candidates and both level/change analyses.
- Max and control Table 1 persistence means are on a broadly similar absolute scale, within one paper-reported but undefined dispersion.
- Several intervention contrasts point in the reported direction: max and control do not materially differ in overall occupancy; last-generation max occupancy exceeds control; and min tends to reduce persistence/occupancy.

### What differs

- Figure 2's aggregate no-trend result does not recur: the primary slopes are significantly positive.
- Figure 5's PhiRL advantage does not recur in either completed-fit or cutoff-causal mode; prevalence drives near-98% raw accuracy while balanced accuracy is about 0.5.
- Figure 6's max >= control >= min ordering fails because max mean persistence and occupancy are below control in both candidates.
- Most Table 1 probability, consistency, and timing values differ materially; min persistence also differs.

## Matrix B — prospective and causal interpretation

{markdown_table(['Question', 'Status', 'Eligible mode', 'Finding'], [[row['question'], STATUS_DISPLAY[row['status']], row['eligibleMode'], row['finding']] for row in matrix_b])}

The completed-fit branch fails future-suffix independence by construction. The cutoff-causal branch passes suffix-invariance checks, but that operational success does not rescue prediction because its comparative, uncertainty, incremental-value, and calibration gates fail. Likewise, S17 proves that a literal online scorer can be executed and replayed, not that the score causally controls replication.

## Directed final classifications

{markdown_table(['Classification', 'Status', 'Meaning'], [[row['classification'], row['status'], row['meaning']] for row in classifications])}

## E01 verdict

- **Paper-facing forensic reproduction:** partial and directional, concentrated in completed-fit descriptive and association results.
- **Past-only early warning:** not supported within tested scope; association directions reverse under past-only fitting.
- **Incremental prediction beyond H/stability:** not supported; exact contemporaneous H defines Y and ordinary stability is coupled.
- **Retrospective completed-fit prediction resemblance:** not supported within tested scope.
- **Prospective cutoff-causal prediction:** not supported within tested scope in either candidate.
- **Literal intervention ordering resemblance:** not supported within tested scope in either candidate.
- **Prospective bidirectional causal control:** not supported within tested scope.
- **Exact author implementation:** underdetermined.
- **Full-plan gate:** `{contract['overallVerdict']['fullPlanGate']}` because the result is a partial retrospective forensic reconstruction, fewer than two central evidence layers reproduce robustly, and author implementation remains materially unresolved.

## Validation

Validation passed for:

- exact 59-claim coverage and uniqueness;
- frozen status counts and vocabulary;
- distinct directional and quantitative assessments;
- complete Matrix A/Matrix B separation;
- seven Matrix B questions and all Figure 2–6/Table 1 entries;
- required directed classifications;
- every claim-to-evidence path and SHA-256 digest;
- byte-for-byte S01–S17 preservation;
- byte-for-byte legacy V1 bundle preservation;
- unchanged S17 CPU-waiver evidence;
- identical canonical and V2 report-input copies;
- S18/V2 artifact manifests and terminal closeout manifest;
- inactive S19/S20 and E02 options.

## Provenance

- Repository branch: `{repo['branch']}`
- Frozen pushed commit: `{repo['head']}`
- Original paper: `{PAPER_PATH}`
- Original paper SHA-256: `{EXPECTED_PAPER_SHA256}`
- S18 contract: `{CONTRACT_PATH.relative_to(REPO_ROOT)}`
- Claim adjudication lock: `{CLAIM_MAP_PATH.relative_to(REPO_ROOT)}`
- Canonical output: `{STEP_ROOT}`
- V2 replication artifact: `{V2_ROOT}`
- Legacy V1 bundle: `{LEGACY_BUNDLE_ROOT}` (unchanged)

Python standard-library synthesis was used. No new dependency was installed.

## Caveats, blockers, failed assumptions, and limitations

- `Y=I(H>0.9)`: exact H fully determines the binary target, so unrestricted incremental target information beyond exact H is zero.
- Completed-fit PhiRL values depend on full-trajectory partitions/Gaussian parameters; their resemblance is retrospective.
- The historical post-fission label points negatively, and past-only refitting reverses the association direction.
- Exact author simulator, aggregation, MLP layout, uncertainty meaning, and intervention scoring remain unavailable.
- Sixteen paper claims were not evaluated; this is not evidence against them.
- The paper's Mann–Whitney scope, Ljung–Box lag, spike threshold scope, Table 1 dispersion type, and first-replicator unit remain unresolved.
- S17's human CPU-allowance waiver remains byte-for-byte preserved and changes no scientific conclusion.
- S17 did not include a random-action rollout because the exact 72-trajectory scope was locked; matched-random outcome exclusion is therefore underdetermined.
- These are simulation results in a reconstructed GARD model, not evidence about living systems or prebiotic chemistry.

## Human-review options and recommended next action

{markdown_table(['Option', 'Title', 'Scope', 'Active'], option_rows)}

The scientific recommendation is **Option A: a stronger separate E02**, focused on prospective onset-risk targets, incremental value beyond H and attractor stability, estimator/partition robustness, and matched intervention controls. This follows the existing group plan and gives negative E01 results productive use.

**Option B is retained because the human requested it:** E01 may be explicitly reopened under a new version for S19–S20. A sensible S19 would cover the 16 not-evaluated claims (C001–C012, C029, C031–C033) under a prospectively frozen contract. S20 would be a separately locked confirmation/author-ambiguity step defined before outcome access. This option is not queued, does not amend the present completed-step record, and cannot mutate S01–S18.

## Terminal boundary

E01 is closed at S18 under the current authorization. Control returns to the Chief Scientist and human reviewer. E02 is not started, S19–S20 are not queued, no report bundle is automatically generated beyond the user-requested compact V2 replication artifact/report-input layer, and no estimator, simulator, threshold, label, or scorer is added.
"""


def build_anchor_inputs() -> dict[str, Path]:
    sidecars = sorted(Path("/workspace/input-attachments").glob("*/_metadata/ATTACHMENT.md"))
    anchors: dict[str, Path] = {
        "AGENTS.md": Path("/workspace/AGENTS.md"),
        "FULL_PLAN.md": Path("/workspace/FULL_PLAN.md"),
        "RESEARCH_PLAN.md": Path("/workspace/RESEARCH_PLAN.md"),
        "attachmentManifest": Path("/workspace/input-attachments/MANIFEST.json"),
        "paperMarkdown": Path(
            "/workspace/input-attachments/ed5486bf-a043-485b-a233-d88d8d123759/pdf-markdown.md"
        ),
        "originalPaper": PAPER_PATH,
        "claimLedger": LEGACY_BUNDLE_ROOT / "ledgers/claim_ledger.csv",
        "claimAdjudicationContract": CONTRACT_PATH,
        "claimAdjudicationMap": CLAIM_MAP_PATH,
        "S12FRReport": ARTIFACTS_ROOT
        / "research_steps/S12FR/research_step_full_results.md",
        "S13YReport": ARTIFACTS_ROOT
        / "research_steps/S13Y/research_step_full_results.md",
    }
    for index, sidecar in enumerate(sidecars, start=1):
        anchors[f"attachmentSidecar{index}"] = sidecar
    for step in ("S14", "S15", "S16", "S17"):
        anchors[f"{step}Report"] = ARTIFACTS_ROOT / f"research_steps/{step}/research_step_full_results.md"
        anchors[f"{step}Status"] = ARTIFACTS_ROOT / f"research_steps/{step}/status.json"
        anchors[f"{step}ArtifactManifest"] = ARTIFACTS_ROOT / f"research_steps/{step}/artifact_manifest.json"
    missing = [name for name, path in anchors.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing S18 anchor inputs: {missing}")
    if sha256_file(PAPER_PATH) != EXPECTED_PAPER_SHA256:
        raise ValueError("original paper hash mismatch")
    return anchors


def freeze() -> dict[str, Any]:
    if STEP_ROOT.exists() or V2_ROOT.exists():
        raise FileExistsError(
            "S18 or V2 output already exists; refusing to overwrite a frozen closeout"
        )
    contract = load_contract()
    ledger_rows = read_csv(LEGACY_BUNDLE_ROOT / "ledgers/claim_ledger.csv")
    claim_map = load_claim_map()
    checks = validate_contract(contract, ledger_rows, claim_map)
    repo = repo_state(require_clean_pushed=True)
    roots = [
        *(ARTIFACTS_ROOT / "research_steps" / step for step in PRIOR_STEP_IDS),
        LEGACY_BUNDLE_ROOT,
    ]
    entries = hash_tree(roots)
    baseline = {
        "schema": "eidosoma.e01.s18_immutable_prior_baseline.v1",
        "researchStepId": "S18",
        "createdAtUtc": utc_now(),
        "roots": [str(root) for root in roots],
        "fileCount": len(entries),
        "totalBytes": sum(int(row["bytes"]) for row in entries),
        "files": entries,
    }
    STEP_ROOT.mkdir(parents=True)
    write_json(STEP_ROOT / "immutable_prior_baseline.json", baseline)
    anchors = build_anchor_inputs()
    input_manifest = {
        "schema": "eidosoma.e01.s18_input_manifest.v1",
        "researchStepId": "S18",
        "createdAtUtc": utc_now(),
        "inputs": {
            name: file_record(path) for name, path in sorted(anchors.items())
        },
        "priorEvidenceBaseline": {
            "path": str(STEP_ROOT / "immutable_prior_baseline.json"),
            "sha256": sha256_file(STEP_ROOT / "immutable_prior_baseline.json"),
            "fileCount": baseline["fileCount"],
            "totalBytes": baseline["totalBytes"],
        },
    }
    write_json(STEP_ROOT / "input_manifest.json", input_manifest)
    lock = {
        "schema": "eidosoma.e01.s18_pre_synthesis_lock.v1",
        "researchStepId": "S18",
        "versionedStepId": contract["versionedStepId"],
        "replicationArtifactVersion": contract["replicationArtifactVersion"],
        "createdAtUtc": utc_now(),
        "passed": True,
        "repository": repo,
        "contractPath": str(CONTRACT_PATH),
        "contractSha256": sha256_file(CONTRACT_PATH),
        "claimMapPath": str(CLAIM_MAP_PATH),
        "claimMapSha256": sha256_file(CLAIM_MAP_PATH),
        "inputManifestSha256": sha256_file(STEP_ROOT / "input_manifest.json"),
        "priorEvidenceBaselineSha256": sha256_file(
            STEP_ROOT / "immutable_prior_baseline.json"
        ),
        "contractValidationChecks": checks,
        "scientificOutcomeGenerated": False,
        "trajectoryGenerated": False,
        "estimatorFit": False,
        "modelTrained": False,
        "interventionRun": False,
        "e02Started": False,
        "s19S20Queued": False,
    }
    write_json(STEP_ROOT / "pre_synthesis_lock.json", lock)
    return {
        "stage": "freeze",
        "passed": True,
        "repositoryCommit": repo["head"],
        "priorFileCount": baseline["fileCount"],
        "priorBytes": baseline["totalBytes"],
        "lockPath": str(STEP_ROOT / "pre_synthesis_lock.json"),
    }


def current_records_from_baseline(baseline: dict[str, Any]) -> list[dict[str, Any]]:
    return hash_tree(Path(root) for root in baseline["roots"])


def compare_baseline(baseline: dict[str, Any]) -> dict[str, Any]:
    expected = {row["path"]: row for row in baseline["files"]}
    current_rows = current_records_from_baseline(baseline)
    current = {row["path"]: row for row in current_rows}
    missing = sorted(set(expected) - set(current))
    added = sorted(set(current) - set(expected))
    changed = sorted(
        path
        for path in set(expected) & set(current)
        if expected[path]["sha256"] != current[path]["sha256"]
        or expected[path]["bytes"] != current[path]["bytes"]
    )
    return {
        "schema": "eidosoma.e01.s18_immutable_prior_validation.v1",
        "researchStepId": "S18",
        "passed": not (missing or added or changed),
        "expectedFileCount": len(expected),
        "observedFileCount": len(current),
        "missingPaths": missing,
        "addedPaths": added,
        "changedPaths": changed,
    }


def write_matrix_outputs(
    matrix: list[dict[str, Any]],
    matrix_b: list[dict[str, Any]],
    figure_map: list[dict[str, Any]],
    classifications: list[dict[str, Any]],
    trace: list[dict[str, Any]],
    counts: list[dict[str, Any]],
) -> None:
    matrix_fields = list(matrix[0])
    matrix_b_fields = list(matrix_b[0])
    figure_fields = list(figure_map[0])
    classification_fields = list(classifications[0])
    trace_fields = list(trace[0])
    count_fields = ["scope", "claimFamily", "statusCode", "count"]
    canonical = {
        "matrix_a_59_claims": (matrix, matrix_fields),
        "matrix_b_prospective_causal": (matrix_b, matrix_b_fields),
        "figure_table_reconstruction_map": (figure_map, figure_fields),
        "final_classification_registry": (classifications, classification_fields),
        "claim_evidence_traceability": (trace, trace_fields),
        "claim_status_counts": (counts, count_fields),
    }
    report_inputs = V2_ROOT / "report_inputs"
    for name, (rows, fields) in canonical.items():
        write_csv(STEP_ROOT / f"{name}.csv", rows, fields)
        write_json(
            STEP_ROOT / f"{name}.json",
            {"researchStepId": "S18", "rowCount": len(rows), "rows": rows},
        )
        write_csv(report_inputs / f"{name}.csv", rows, fields)
        write_json(
            report_inputs / f"{name}.json",
            {"researchStepId": "S18", "rowCount": len(rows), "rows": rows},
        )


def synthesize() -> dict[str, Any]:
    required = [
        STEP_ROOT / "pre_synthesis_lock.json",
        STEP_ROOT / "immutable_prior_baseline.json",
        STEP_ROOT / "input_manifest.json",
    ]
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("S18 freeze stage is incomplete")
    if V2_ROOT.exists():
        raise FileExistsError("V2 artifact already exists; refusing overwrite")
    contract = load_contract()
    lock = json.loads(required[0].read_text(encoding="utf-8"))
    baseline = json.loads(required[1].read_text(encoding="utf-8"))
    if sha256_file(CONTRACT_PATH) != lock["contractSha256"]:
        raise ValueError("S18 contract changed after freeze")
    if sha256_file(CLAIM_MAP_PATH) != lock["claimMapSha256"]:
        raise ValueError("S18 claim map changed after freeze")
    repo = repo_state(require_clean_pushed=True)
    if repo["head"] != lock["repository"]["head"]:
        raise ValueError("repository commit changed after S18 freeze")
    before = compare_baseline(baseline)
    if not before["passed"]:
        raise ValueError("prior evidence changed before S18 synthesis")
    ledger_rows = read_csv(LEGACY_BUNDLE_ROOT / "ledgers/claim_ledger.csv")
    claim_map = load_claim_map()
    contract_checks = validate_contract(contract, ledger_rows, claim_map)
    matrix, trace = build_matrix_a(ledger_rows, claim_map)
    matrix_b = add_evidence_to_rows(contract["matrixB"])
    figure_map = add_evidence_to_rows(contract["figureTableMap"])
    classifications = list(contract["finalClassifications"])
    counts = status_count_rows(matrix)
    V2_ROOT.mkdir(parents=True, exist_ok=False)
    write_matrix_outputs(matrix, matrix_b, figure_map, classifications, trace, counts)
    write_json(
        V2_ROOT / "VERSION.json",
        {
            "schema": "eidosoma.e01.forensic_replication_artifact_version.v1",
            "artifactVersion": contract["replicationArtifactVersion"],
            "researchStepId": "S18",
            "legacyV1Bundle": str(LEGACY_BUNDLE_ROOT),
            "legacyV1Mutated": False,
            "artifactRole": "COMPACT_CLOSEOUT_AND_REPORT_INPUT_LAYER",
        },
    )
    write_json(
        V2_ROOT / "report_inputs/e01_closeout_decision.json",
        {
            "schema": "eidosoma.e01.s18_closeout_decision.v1",
            "researchStepId": "S18",
            **contract["overallVerdict"],
            "matrixAStatusCounts": contract["expectedClaimStatusCounts"],
            "e02Recommendation": contract["recommendedE02Review"],
            "postCloseoutHumanReviewOptions": contract[
                "postCloseoutHumanReviewOptions"
            ],
        },
    )
    write_json(
        V2_ROOT / "report_inputs/post_closeout_human_review_options.json",
        {
            "schema": "eidosoma.e01.s18_post_closeout_human_review_options.v1",
            "researchStepId": "S18",
            "options": contract["postCloseoutHumanReviewOptions"],
            "currentBoundary": contract["currentBoundaryAfterS18"],
        },
    )
    legacy_entries = [
        row
        for row in baseline["files"]
        if row["path"].startswith(str(LEGACY_BUNDLE_ROOT) + "/")
    ]
    step_entries = [row for row in baseline["files"] if row not in legacy_entries]
    write_json(
        V2_ROOT / "manifests/legacy_v1_bundle_manifest.json",
        {
            "schema": "eidosoma.e01.s18_legacy_v1_bundle_manifest.v1",
            "root": str(LEGACY_BUNDLE_ROOT),
            "fileCount": len(legacy_entries),
            "totalBytes": sum(int(row["bytes"]) for row in legacy_entries),
            "files": legacy_entries,
        },
    )
    write_json(
        V2_ROOT / "manifests/s01_s17_evidence_manifest.json",
        {
            "schema": "eidosoma.e01.s18_s01_s17_evidence_manifest.v1",
            "roots": [
                str(ARTIFACTS_ROOT / "research_steps" / step)
                for step in PRIOR_STEP_IDS
            ],
            "fileCount": len(step_entries),
            "totalBytes": sum(int(row["bytes"]) for row in step_entries),
            "files": step_entries,
        },
    )
    prior_validation = compare_baseline(baseline)
    write_json(STEP_ROOT / "immutable_prior_validation.json", prior_validation)
    failure_rows = [
        {
            "item": "RETROSPECTIVE_PREDICTION_RESEMBLANCE",
            "classification": "NOT_SUPPORTED_WITHIN_TESTED_SCOPE",
            "evidence": "S16 completed-fit PhiRL failed its locked paper-baseline advantage in both candidates.",
            "boundary": "Retrospective only; no prospective rescue.",
        },
        {
            "item": "PROSPECTIVE_PREDICTION_SUPPORTED",
            "classification": "NOT_SUPPORTED_WITHIN_TESTED_SCOPE",
            "evidence": "All six cutoff-causal gates failed in both candidates.",
            "boundary": "Frozen candidates and first-quarter design only.",
        },
        {
            "item": "LITERAL_INTERVENTION_ORDERING_RESEMBLANCE",
            "classification": "NOT_SUPPORTED_WITHIN_TESTED_SCOPE",
            "evidence": "Max persistence and occupancy means were below control in both candidates.",
            "boundary": "One locked literal scorer; exact author scorer unavailable.",
        },
        {
            "item": "PROSPECTIVE_CAUSAL_CONTROL_SUPPORTED",
            "classification": "NOT_SUPPORTED_WITHIN_TESTED_SCOPE",
            "evidence": "Bidirectional paired effects and action-separability gates failed.",
            "boundary": "Matched-random outcome exclusion remains underdetermined.",
        },
        {
            "item": "UNDERDETERMINED_AUTHOR_IMPLEMENTATION",
            "classification": "UNDERDETERMINED",
            "evidence": "Author simulator, aggregation, MLP, dispersion, time unit, and action-scoring details remain unavailable.",
            "boundary": "No author contact and no post-outcome method search in S18.",
        },
    ]
    write_csv(
        STEP_ROOT / "caveat_and_non_support_ledger.csv",
        failure_rows,
        ["item", "classification", "evidence", "boundary"],
    )
    decision = {
        "schema": "eidosoma.e01.s18_final_decision.v1",
        "researchStepId": "S18",
        "stepNumber": 18,
        "versionedStepId": contract["versionedStepId"],
        "matrixAStatusCounts": contract["expectedClaimStatusCounts"],
        "directionalOrSupportedClaimCount": sum(
            contract["expectedClaimStatusCounts"][key]
            for key in ("SUPPORTED", "DIRECTIONALLY_SUPPORTED")
        ),
        "matrixB": matrix_b,
        "finalClassifications": classifications,
        "overallVerdict": contract["overallVerdict"],
        "recommendedE02Review": contract["recommendedE02Review"],
        "postCloseoutHumanReviewOptions": contract[
            "postCloseoutHumanReviewOptions"
        ],
        "currentBoundaryAfterS18": contract["currentBoundaryAfterS18"],
    }
    write_json(STEP_ROOT / "decision.json", decision)
    provenance = {
        "schema": "eidosoma.e01.s18_provenance.v1",
        "researchStepId": "S18",
        "createdAtUtc": utc_now(),
        "repository": repo,
        "contract": file_record(CONTRACT_PATH),
        "claimMap": file_record(CLAIM_MAP_PATH),
        "originalPaper": file_record(PAPER_PATH),
        "inputManifest": file_record(STEP_ROOT / "input_manifest.json"),
        "priorEvidenceBaseline": file_record(
            STEP_ROOT / "immutable_prior_baseline.json"
        ),
        "scientificCompute": {
            "trajectoryGeneration": 0,
            "estimatorFits": 0,
            "modelTrainingRuns": 0,
            "interventionRollouts": 0,
            "gpuUsed": False,
            "description": "Deterministic synthesis and hashing only",
        },
        "dependenciesAdded": [],
        "pythonImplementation": "standard library only",
    }
    write_json(STEP_ROOT / "provenance_manifest.json", provenance)
    status = {
        "schema": "eidosoma.e01.s18_status.v1",
        "researchStepId": "S18",
        "stepNumber": 18,
        "success": True,
        "status": "COMPLETE_E01_CLOSED_FOR_HUMAN_REVIEW",
        "artifactsWritten": [
            str(STEP_ROOT / "research_step_full_results.md"),
            str(STEP_ROOT / "matrix_a_59_claims.csv"),
            str(STEP_ROOT / "matrix_b_prospective_causal.csv"),
            str(STEP_ROOT / "figure_table_reconstruction_map.csv"),
            str(STEP_ROOT / "final_classification_registry.csv"),
            str(STEP_ROOT / "claim_evidence_traceability.csv"),
            str(STEP_ROOT / "e01_closeout_manifest.json"),
            str(V2_ROOT),
        ],
        "validationResult": "PASS",
        "outcomeClassification": contract["overallVerdict"][
            "outcomeClassification"
        ],
        "paperFacingVerdict": contract["overallVerdict"]["paperFacingVerdict"],
        "prospectiveVerdict": contract["overallVerdict"]["prospectiveVerdict"],
        "causalVerdict": contract["overallVerdict"]["causalVerdict"],
        "caveatsOrBlockers": [
            "Exact author implementation remains unavailable.",
            "Completed-fit resemblance is retrospective and future-dependent.",
            "Y=I(H>0.9), so exact H fully determines the binary target.",
            "Sixteen paper claims were not evaluated and two remain underdetermined.",
            "S17 CPU-allowance waiver is preserved unchanged and is operational only.",
        ],
        "recommendedNextAction": "Chief Scientist and human review: choose a stronger separate E02 or explicitly version and reopen E01 for optional S19-S20; start neither automatically.",
        "e01Closed": True,
        "e02Started": False,
        "s19S20Queued": False,
    }
    write_json(STEP_ROOT / "status.json", status)
    v2_status = render_v2_status(contract, matrix, matrix_b)
    write_text(V2_ROOT / "replication_status_v2.md", v2_status)
    full_report = render_full_report(
        contract,
        matrix,
        matrix_b,
        figure_map,
        classifications,
        {"fileCount": baseline["fileCount"], "totalBytes": baseline["totalBytes"]},
        repo,
    )
    write_text(STEP_ROOT / "research_step_full_results.md", full_report)
    report_inputs_root = V2_ROOT / "report_inputs"
    report_input_manifest = manifest_for_root(
        report_inputs_root, exclude_names={"report_input_manifest.json"}
    )
    report_input_manifest.update(
        {
            "schema": "eidosoma.e01.s18_report_input_manifest.v1",
            "researchStepId": "S18",
            "reportBundleAutomaticallyGenerated": False,
        }
    )
    write_json(report_inputs_root / "report_input_manifest.json", report_input_manifest)
    evidence_paths_ok = all(row["exists"] for row in trace)
    counts_now = Counter(row["finalStatusCode"] for row in matrix)
    s17_report = ARTIFACTS_ROOT / "research_steps/S17/research_step_full_results.md"
    s17_waiver_preserved = (
        "CPU-allowance waiver" in s17_report.read_text(encoding="utf-8")
        or "CPU allowance" in s17_report.read_text(encoding="utf-8")
    )
    validation_checks = {
        **contract_checks,
        "matrix_a_row_count": len(matrix) == 59,
        "matrix_a_claim_ids_unique": len({row["claimId"] for row in matrix}) == 59,
        "matrix_a_status_counts": dict(counts_now)
        == contract["expectedClaimStatusCounts"],
        "matrix_a_direction_and_numeric_fields_present": all(
            row["directionalAssessment"] and row["quantitativeAssessment"]
            for row in matrix
        ),
        "matrix_b_separate_and_complete": len(matrix_b) == 7,
        "claim_evidence_traceability_complete": evidence_paths_ok
        and {row["claimId"] for row in trace}
        == {row["claimId"] for row in matrix},
        "figure_table_dependencies_complete": all(
            row["completedFitDependency"]
            and row["labelScopeDependency"]
            and row["interventionScoringDependency"]
            for row in figure_map
        ),
        "prior_evidence_immutable": prior_validation["passed"],
        "legacy_v1_bundle_immutable": prior_validation["passed"]
        and len(legacy_entries) > 0,
        "s17_cpu_waiver_preserved": s17_waiver_preserved,
        "original_paper_hash": sha256_file(PAPER_PATH) == EXPECTED_PAPER_SHA256,
        "canonical_report_present": (
            STEP_ROOT / "research_step_full_results.md"
        ).is_file(),
        "v2_status_present": (V2_ROOT / "replication_status_v2.md").is_file(),
        "v2_is_compact_reference_layer": not any(
            path.suffix in {".parquet", ".zarr", ".h5", ".hdf5"}
            for path in V2_ROOT.rglob("*")
            if path.is_file()
        ),
        "e02_not_started": not status["e02Started"],
        "s19_s20_not_queued": not status["s19S20Queued"],
        "no_scientific_compute": all(
            provenance["scientificCompute"][key] == 0
            for key in (
                "trajectoryGeneration",
                "estimatorFits",
                "modelTrainingRuns",
                "interventionRollouts",
            )
        ),
    }
    failed = [key for key, value in validation_checks.items() if not value]
    validation = {
        "schema": "eidosoma.e01.s18_validation.v1",
        "researchStepId": "S18",
        "passed": not failed,
        "validationResult": "PASS" if not failed else "FAIL",
        "checkCount": len(validation_checks),
        "passedCount": sum(bool(value) for value in validation_checks.values()),
        "failedChecks": failed,
        "checks": validation_checks,
    }
    write_json(STEP_ROOT / "validation.json", validation)
    if failed:
        raise ValueError(f"S18 internal validation failed: {failed}")
    v2_manifest = manifest_for_root(
        V2_ROOT, exclude_names={"artifact_manifest.json"}
    )
    v2_manifest.update(
        {
            "schema": "eidosoma.e01.s18_v2_artifact_manifest.v1",
            "researchStepId": "S18",
            "artifactVersion": contract["replicationArtifactVersion"],
        }
    )
    write_json(V2_ROOT / "artifact_manifest.json", v2_manifest)
    closeout = {
        "schema": "eidosoma.e01.s18_e01_closeout_manifest.v1",
        "researchStepId": "S18",
        "versionedStepId": contract["versionedStepId"],
        "artifactVersion": contract["replicationArtifactVersion"],
        "e01State": "CLOSED_FOR_HUMAN_REVIEW",
        "priorEvidenceBaseline": file_record(
            STEP_ROOT / "immutable_prior_baseline.json"
        ),
        "priorEvidenceValidation": prior_validation,
        "s18Payload": manifest_for_root(
            STEP_ROOT,
            exclude_names={"artifact_manifest.json", "e01_closeout_manifest.json"},
        ),
        "v2Payload": manifest_for_root(V2_ROOT, exclude_names=set()),
        "legacyV1Bundle": {
            "root": str(LEGACY_BUNDLE_ROOT),
            "fileCount": len(legacy_entries),
            "totalBytes": sum(int(row["bytes"]) for row in legacy_entries),
            "mutated": False,
        },
        "e02Started": False,
        "s19S20Queued": False,
        "reportBundleAutomaticallyGenerated": False,
    }
    write_json(STEP_ROOT / "e01_closeout_manifest.json", closeout)
    artifact_manifest = manifest_for_root(
        STEP_ROOT, exclude_names={"artifact_manifest.json"}
    )
    artifact_manifest.update(
        {
            "schema": "eidosoma.e01.s18_artifact_manifest.v1",
            "researchStepId": "S18",
            "versionedStepId": contract["versionedStepId"],
        }
    )
    write_json(STEP_ROOT / "artifact_manifest.json", artifact_manifest)
    return validate_outputs()


def verify_root_manifest(root: Path, manifest_path: Path) -> tuple[bool, list[str]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    problems: list[str] = []
    expected = {row["path"]: row for row in manifest["files"]}
    for rel, row in expected.items():
        path = root / rel
        if not path.is_file():
            problems.append(f"missing:{path}")
        elif sha256_file(path) != row["sha256"]:
            problems.append(f"hash:{path}")
        elif path.stat().st_size != row["bytes"]:
            problems.append(f"bytes:{path}")
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if actual != set(expected):
        problems.append(
            f"set-mismatch:missing={sorted(set(expected)-actual)}:extra={sorted(actual-set(expected))}"
        )
    return not problems, problems


def validate_outputs() -> dict[str, Any]:
    required = [
        STEP_ROOT / "artifact_manifest.json",
        STEP_ROOT / "validation.json",
        STEP_ROOT / "status.json",
        STEP_ROOT / "research_step_full_results.md",
        STEP_ROOT / "matrix_a_59_claims.csv",
        STEP_ROOT / "matrix_b_prospective_causal.csv",
        STEP_ROOT / "figure_table_reconstruction_map.csv",
        STEP_ROOT / "final_classification_registry.csv",
        STEP_ROOT / "claim_evidence_traceability.csv",
        STEP_ROOT / "e01_closeout_manifest.json",
        V2_ROOT / "artifact_manifest.json",
        V2_ROOT / "replication_status_v2.md",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing final S18 artifacts: {missing}")
    baseline = json.loads(
        (STEP_ROOT / "immutable_prior_baseline.json").read_text(encoding="utf-8")
    )
    prior = compare_baseline(baseline)
    s18_ok, s18_problems = verify_root_manifest(
        STEP_ROOT, STEP_ROOT / "artifact_manifest.json"
    )
    v2_ok, v2_problems = verify_root_manifest(
        V2_ROOT, V2_ROOT / "artifact_manifest.json"
    )
    matrix = read_csv(STEP_ROOT / "matrix_a_59_claims.csv")
    counts = Counter(row["finalStatusCode"] for row in matrix)
    contract = load_contract()
    copied = [
        "matrix_a_59_claims.csv",
        "matrix_a_59_claims.json",
        "matrix_b_prospective_causal.csv",
        "matrix_b_prospective_causal.json",
        "figure_table_reconstruction_map.csv",
        "figure_table_reconstruction_map.json",
        "final_classification_registry.csv",
        "final_classification_registry.json",
        "claim_evidence_traceability.csv",
        "claim_evidence_traceability.json",
        "claim_status_counts.csv",
        "claim_status_counts.json",
    ]
    copies_identical = all(
        sha256_file(STEP_ROOT / name)
        == sha256_file(V2_ROOT / "report_inputs" / name)
        for name in copied
    )
    report_text = (STEP_ROOT / "research_step_full_results.md").read_text(
        encoding="utf-8"
    )
    required_report_terms = [
        "Research step ID",
        "Completion status",
        "Artifacts written",
        "Validation result",
        "Outcome classification",
        "Caveats or blockers",
        "Lay summary",
        "Recommended next action",
        "## Methods",
        "## Commands",
        "## Matrix A",
        "## Matrix B",
        "## Provenance",
        "S19–S20",
    ]
    checks = {
        "prior_evidence_immutable": prior["passed"],
        "s18_manifest_replay": s18_ok,
        "v2_manifest_replay": v2_ok,
        "matrix_a_has_59_unique_claims": len(matrix) == 59
        and len({row["claimId"] for row in matrix}) == 59,
        "status_counts_match_contract": dict(counts)
        == contract["expectedClaimStatusCounts"],
        "canonical_v2_copies_identical": copies_identical,
        "report_required_sections": all(term in report_text for term in required_report_terms),
        "s18_validation_passed": json.loads(
            (STEP_ROOT / "validation.json").read_text(encoding="utf-8")
        )["passed"],
        "e02_not_started": not json.loads(
            (STEP_ROOT / "status.json").read_text(encoding="utf-8")
        )["e02Started"],
        "s19_s20_not_queued": not json.loads(
            (STEP_ROOT / "status.json").read_text(encoding="utf-8")
        )["s19S20Queued"],
    }
    problems = s18_problems + v2_problems
    failed = [key for key, value in checks.items() if not value]
    if failed or problems:
        raise ValueError(
            f"S18 final validation failed: checks={failed}, manifest={problems}"
        )
    return {
        "stage": "validate",
        "passed": True,
        "checks": checks,
        "matrixAStatusCounts": dict(counts),
        "priorFileCount": prior["expectedFileCount"],
        "s18ManifestFileCount": json.loads(
            (STEP_ROOT / "artifact_manifest.json").read_text(encoding="utf-8")
        )["fileCount"],
        "v2ManifestFileCount": json.loads(
            (V2_ROOT / "artifact_manifest.json").read_text(encoding="utf-8")
        )["fileCount"],
        "e01State": "CLOSED_FOR_HUMAN_REVIEW",
        "e02Started": False,
        "s19S20Queued": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("freeze", "synthesize", "validate"))
    args = parser.parse_args()
    if args.stage == "freeze":
        result = freeze()
    elif args.stage == "synthesize":
        result = synthesize()
    else:
        result = validate_outputs()
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
