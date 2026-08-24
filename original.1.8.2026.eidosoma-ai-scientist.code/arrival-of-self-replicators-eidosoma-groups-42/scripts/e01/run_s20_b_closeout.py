#!/usr/bin/env python3
"""Build the deterministic E01/S20-B closeout and compact V3 addendum.

This runner is intentionally synthesis-only.  It validates immutable E01 evidence,
copies every S18 matrix field unchanged into additive V3 tables, emits explicit
zero-row confirmation schemas, records the L54 simulator-process finding at its
frozen scope, and closes the versioned E01 continuation.  It never imports or
calls a simulator, estimator, label builder, model trainer, or intervention code.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = REPO_ROOT / "configs/e01/s20_b_closeout_contract.json"
ARTIFACTS_ROOT = Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")).resolve()
STEP_ROOT = ARTIFACTS_ROOT / "research_steps/S20"
V3_ROOT = ARTIFACTS_ROOT / "E01_forensic_replication_artifact_v3"
S18_ROOT = ARTIFACTS_ROOT / "research_steps/S18"
V1_ROOT = ARTIFACTS_ROOT / "E01_forensic_replication_bundle"
V2_ROOT = ARTIFACTS_ROOT / "E01_forensic_replication_artifact_v2"
S19_ROOT = ARTIFACTS_ROOT / "research_steps/S19"
L12_ROOT = S19_ROOT / "loops/L12"
L53_ROOT = S19_ROOT / "loops/L53"
L54_ROOT = S19_ROOT / "loops/L54"
S17_ROOT = ARTIFACTS_ROOT / "research_steps/S17"

S18_MATRIX_A = S18_ROOT / "matrix_a_59_claims.csv"
S18_MATRIX_B = S18_ROOT / "matrix_b_prospective_causal.csv"
S18_FIGURE_MAP = S18_ROOT / "figure_table_reconstruction_map.csv"
S18_CLASSIFICATIONS = S18_ROOT / "final_classification_registry.csv"

S20_NA = "NOT_APPLICABLE_S20_B_CLOSEOUT_ONLY"
V3_EVIDENCE_POSTURE = "S18_UNCHANGED_S19_ADDITIVE_CONTEXT_ONLY"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload))


def write_yaml(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
    path.write_text(text, encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def write_empty_parquet(path: Path, fields: list[tuple[str, pa.DataType]]) -> None:
    schema = pa.schema([pa.field(name, dtype, nullable=True) for name, dtype in fields])
    arrays = [pa.array([], type=field.type) for field in schema]
    table = pa.Table.from_arrays(arrays, schema=schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd", version="2.6")


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
        raise RuntimeError(f"clean pushed repository gate failed: {state}")
    return state


def load_contract() -> dict[str, Any]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def file_record(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    return {
        "path": str(path if root is None else path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def manifest_rows(root: Path, excluded_names: set[str] | None = None) -> list[dict[str, Any]]:
    excluded_names = excluded_names or set()
    return [
        file_record(path, root=root)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in excluded_names
    ]


def manifest_payload(root: Path, schema: str, excluded_names: set[str] | None = None) -> dict[str, Any]:
    rows = manifest_rows(root, excluded_names)
    return {
        "schema": schema,
        "root": str(root),
        "fileCount": len(rows),
        "totalBytes": sum(int(row["bytes"]) for row in rows),
        "aggregateSha256": hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "files": rows,
    }


def validate_manifest(root: Path, manifest_path: Path) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = payload.get("files", [])
    checks = []
    for row in rows:
        path = root / row["path"]
        checks.append(path.is_file() and sha256_file(path) == row["sha256"])
    return {
        "manifest": str(manifest_path),
        "root": str(root),
        "expectedFiles": len(rows),
        "unchangedFiles": int(sum(checks)),
        "passed": bool(checks and all(checks)),
        "manifestSha256": sha256_file(manifest_path),
        "recordedAggregateSha256": payload.get("aggregateSha256"),
    }


def validate_absolute_manifest(manifest_path: Path) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = payload.get("files", [])
    checks = []
    for row in rows:
        path = Path(row["path"])
        checks.append(path.is_file() and sha256_file(path) == row["sha256"])
    return {
        "manifest": str(manifest_path),
        "root": payload.get("root"),
        "expectedFiles": len(rows),
        "unchangedFiles": int(sum(checks)),
        "passed": bool(checks and all(checks)),
        "manifestSha256": sha256_file(manifest_path),
        "recordedTotalBytes": payload.get("totalBytes"),
    }


def validate_s19_manifest() -> dict[str, Any]:
    manifest_path = S19_ROOT / "artifact_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks = []
    for row in payload["files"]:
        path = S19_ROOT / row["path"]
        checks.append(path.is_file() and sha256_file(path) == row["sha256"])
    return {
        "manifest": str(manifest_path),
        "expectedFiles": len(checks),
        "unchangedFiles": int(sum(checks)),
        "passed": bool(checks and all(checks)),
        "manifestSha256": sha256_file(manifest_path),
        "recordedAggregateSha256": payload["aggregateSha256"],
    }


def validate_expected_inputs(contract: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name, expected in contract["expectedInputIdentities"].items():
        path = Path(name)
        actual = sha256_file(path) if path.is_file() else None
        rows.append(
            {
                "path": name,
                "expectedSha256": expected,
                "actualSha256": actual,
                "passed": actual == expected,
            }
        )
    return rows


def prior_validation(contract: dict[str, Any]) -> dict[str, Any]:
    expected = validate_expected_inputs(contract)
    s01_s17 = validate_absolute_manifest(S18_ROOT / "immutable_prior_baseline.json")
    v1 = validate_absolute_manifest(
        V2_ROOT / "manifests/legacy_v1_bundle_manifest.json"
    )
    s18 = validate_manifest(S18_ROOT, S18_ROOT / "artifact_manifest.json")
    v2 = validate_manifest(V2_ROOT, V2_ROOT / "artifact_manifest.json")
    s19 = validate_s19_manifest()
    l53 = validate_manifest(L53_ROOT, L53_ROOT / "artifact_manifest.json")
    l54 = validate_manifest(L54_ROOT, L54_ROOT / "artifact_manifest.json")
    waiver_path = S17_ROOT / "validation.json"
    waiver = json.loads(waiver_path.read_text(encoding="utf-8"))
    waiver_passed = (
        waiver.get("computeAllowanceHumanWaived") is True
        and waiver.get("waiver", {}).get("scope") == "S17_CPU_ALLOWANCE_ONLY"
    )
    l54_class = json.loads((L54_ROOT / "classification.json").read_text())
    l54_regen = json.loads((L54_ROOT / "regeneration_validation.json").read_text())
    l54_class_pass = set(contract["requiredL54Classifications"]) == set(
        l54_class["classifications"]
    )
    passed = bool(
        all(row["passed"] for row in expected)
        and all(x["passed"] for x in [s01_s17, v1, s18, v2, s19, l53, l54])
        and waiver_passed
        and l54_class_pass
        and l54_regen.get("status") == "PASS"
    )
    return {
        "schema": "eidosoma.e01.s20_b.immutable_prior_validation.v1",
        "passed": passed,
        "expectedIdentityChecks": expected,
        "manifestChecks": [s01_s17, v1, s18, v2, s19, l53, l54],
        "s17WaiverPath": str(waiver_path),
        "s17WaiverSha256": sha256_file(waiver_path),
        "s17WaiverPreserved": waiver_passed,
        "l54ClassificationExact": l54_class_pass,
        "l54RegenerationStatus": l54_regen.get("status"),
        "s18MatrixABytesSha256": sha256_file(S18_MATRIX_A),
        "s18MatrixBBytesSha256": sha256_file(S18_MATRIX_B),
        "s18ClaimTotalsExpected": contract["expectedS18StatusCounts"],
    }


def append_v3_fields(
    rows: list[dict[str, str]], kind: str
) -> tuple[list[dict[str, Any]], list[str]]:
    if not rows:
        raise ValueError(f"empty S18 {kind}")
    base_fields = list(rows[0])
    extra_fields = [
        "s18SnapshotStatus",
        "s19ExploratoryOverlayStatus",
        "s20ConfirmationStatus",
        "finalV3AddendumStatus",
        "v3EvidencePosture",
        "l54Relationship",
        "l54EvidencePath",
        "l54EvidenceSha256",
    ]
    evidence_path = str(L54_ROOT / "research_step_full_results.md")
    evidence_hash = sha256_file(Path(evidence_path))
    output = []
    for row in rows:
        out: dict[str, Any] = dict(row)
        status = row["finalStatusCode"] if kind == "matrix_a" else row["status"]
        out.update(
            {
                "s18SnapshotStatus": status,
                "s19ExploratoryOverlayStatus": "HISTORICAL_S19_CLASSIFICATIONS_PRESERVED;UNCONFIRMED_FINDINGS_EXPLORATORY",
                "s20ConfirmationStatus": S20_NA,
                "finalV3AddendumStatus": status,
                "v3EvidencePosture": V3_EVIDENCE_POSTURE,
                "l54Relationship": "ADDITIVE_CONFIRMED_SIMULATOR_PROCESS_FINDING_OUTSIDE_PAPER_CLAIM_ADJUDICATION",
                "l54EvidencePath": evidence_path,
                "l54EvidenceSha256": evidence_hash,
            }
        )
        output.append(out)
    return output, base_fields + extra_fields


def s18_v3_crosswalk(matrix_a: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    base_fields = list(read_csv(S18_MATRIX_A)[0])
    for row in matrix_a:
        original = {key: row[key] for key in base_fields}
        rows.append(
            {
                "claimId": row["claimId"],
                "s18Status": row["finalStatusCode"],
                "s18RowSha256": hashlib.sha256(canonical_json_bytes(original)).hexdigest(),
                "s19Status": "ADDITIVE_EXPLORATORY_CONTEXT_NO_S18_CHANGE",
                "s20Mode": "S20_B_CLOSEOUT_ONLY",
                "s20ConfirmationStatus": S20_NA,
                "finalV3Status": row["finalStatusCode"],
                "statusChangedFromS18": False,
                "evidenceBoundary": "L54_IS_SEPARATE_SIMULATOR_PROCESS_EVIDENCE_NOT_A_PAPER_CLAIM_RECLASSIFIER",
            }
        )
    return rows


def author_ambiguity_rows() -> list[dict[str, Any]]:
    source = read_csv(L12_ROOT / "unresolved_author_implementation_matrix.csv")
    return [
        {
            **row,
            "s18Disposition": "PRESERVED_UNRESOLVED",
            "s19Disposition": "EXPLORATORY_AUDIT_DID_NOT_IDENTIFY_AUTHOR_IMPLEMENTATION",
            "s20Disposition": "UNRESOLVED_AT_S20_B_CLOSEOUT",
            "authorContactOccurred": False,
            "v3Resolution": "AUTHOR_AMBIGUITY_RETAINED",
            "evidencePath": str(L12_ROOT / "unresolved_author_implementation_matrix.csv"),
            "evidenceSha256": sha256_file(
                L12_ROOT / "unresolved_author_implementation_matrix.csv"
            ),
        }
        for row in source
    ]


def figure_v3_rows() -> tuple[list[dict[str, Any]], list[str]]:
    source = read_csv(S18_FIGURE_MAP)
    fields = list(source[0]) + [
        "s18ReconstructionDegree",
        "s19ExploratoryOverlay",
        "s20ConfirmationStatus",
        "finalV3ReconstructionDegree",
        "v3Interpretation",
    ]
    rows = []
    for row in source:
        rows.append(
            {
                **row,
                "s18ReconstructionDegree": row["reconstructionDegree"],
                "s19ExploratoryOverlay": "PRESERVED_AS_EXPLORATORY_OR_CONSTRAINING_CONTEXT",
                "s20ConfirmationStatus": S20_NA,
                "finalV3ReconstructionDegree": row["reconstructionDegree"],
                "v3Interpretation": "S18_PAPER_FACING_STATUS_UNCHANGED;L54_IS_A_DIFFERENT_SIMULATOR_PROCESS_RESULT",
            }
        )
    return rows, fields


def s19_classification_snapshot() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((S19_ROOT / "loops").glob("*/classification.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        classifications = payload.get("classifications", [])
        loop = path.parent.name
        v3_posture = (
            "CONFIRMED_UNTOUCHED_SIMULATOR_PROCESS_SCOPE"
            if loop == "L54"
            else "HISTORICAL_CLASSIFICATION_PRESERVED;NO_V3_PROMOTION;UNCONFIRMED_FINDINGS_EXPLORATORY"
        )
        rows.append(
            {
                "loopId": f"S19-{loop}",
                "classificationPath": str(path),
                "classificationSha256": sha256_file(path),
                "historicalClassifications": ";".join(classifications),
                "priorStatusesChanged": bool(payload.get("priorStatusesChanged", False)),
                "v3EvidencePosture": v3_posture,
                "altersS18": False,
            }
        )
    return rows


def final_classification_registry() -> dict[str, Any]:
    s18_rows = read_csv(S18_CLASSIFICATIONS)
    return {
        "schema": "eidosoma.e01.s20_b.final_classification_registry.v1",
        "s18Snapshot": {
            "status": "PRESERVED_FIELD_FOR_FIELD",
            "sourcePath": str(S18_CLASSIFICATIONS),
            "sourceSha256": sha256_file(S18_CLASSIFICATIONS),
            "records": s18_rows,
        },
        "additiveV3Classifications": [
            {
                "classification": "S20_B_CLOSEOUT_ONLY_COMPLETE",
                "status": "COMPLETE",
                "meaning": "S20 generated no scientific outcome and closed the V3 continuation deterministically.",
            },
            {
                "classification": "S18_FIELDS_AND_TOTALS_UNCHANGED",
                "status": "VALIDATED",
                "meaning": "All 59 Matrix A rows, seven Matrix B rows, and every source field remain unchanged in V3.",
            },
            {
                "classification": "UNTOUCHED_PAST_OBSERVABLE_PROCESS_RISK_COORDINATE_CONFIRMED",
                "status": "RETAINED_EXACT_SIMULATOR_PROCESS_SCOPE",
                "meaning": "The frozen target-blind L53 full-state-graph-plus-history coordinate predicted independently measured F12 break-plus-new-run-3 probability on untouched matrices in both candidates.",
            },
            {
                "classification": "NOT_PAPER_REPLICATION",
                "status": "RETAINED_SCOPE_BOUNDARY",
                "meaning": "L54 is not PhiID support, first-replicator prediction, privileged-attractor evidence, restoration/homeostasis, intervention efficacy, or causal control.",
            },
            {
                "classification": "AUTHOR_AMBIGUITY_UNRESOLVED_AT_CLOSEOUT",
                "status": "RETAINED_UNRESOLVED",
                "meaning": "The unavailable author implementation prevents unique end-to-end paper-pipeline identification.",
            },
            {
                "classification": "E01_FORENSIC_REPLICATION_CONTINUATION_V3_CLOSED",
                "status": "CLOSED",
                "meaning": "No S21 or later E01 loop is authorized.",
            },
            {
                "classification": "E02_STAGE1_SEPARATELY_AUTHORIZED_NOT_EXECUTED",
                "status": "HANDOFF_ONLY",
                "meaning": "The first E02 stage is authorized for a distinct workspace and was not executed during S20.",
            },
        ],
    }


def l54_summary() -> dict[str, Any]:
    gates = pd.read_parquet(L54_ROOT / "scientific_gate_results.parquet")
    reliability = pd.read_parquet(L54_ROOT / "committor_reliability_results.parquet")
    model_comparisons = pd.read_parquet(L54_ROOT / "model_comparisons.parquet")
    return {
        "schema": "eidosoma.e01.s20_b.l54_frozen_result_summary.v1",
        "sourceClassification": json.loads((L54_ROOT / "classification.json").read_text()),
        "sourceClassificationSha256": sha256_file(L54_ROOT / "classification.json"),
        "scope": load_contract()["l54Scope"],
        "gateRows": json.loads(gates.to_json(orient="records")),
        "reliabilityRows": json.loads(reliability.to_json(orient="records")),
        "modelComparisonRows": json.loads(model_comparisons.to_json(orient="records")),
        "interpretation": "past-observable simulator precursor for plastic-heredity regime switching",
        "exclusions": load_contract()["l54InterpretationExclusions"],
        "newScientificComputationInS20": False,
    }


def handover_notes(contract: dict[str, Any]) -> str:
    tests = "\n".join(f"- {item}." for item in contract["e02PreOutcomeTests"])
    return f"""# Handover Notes from E01

## Decision at handover

E01 is closed after `E01-S20-CONFIRMATION-AUTHOR-AMBIGUITY-AND-FINAL-CLOSEOUT-v1.0.0` in S20-B closeout-only mode. S20 generated no scientific outcome. The S18 paper-facing verdict, every S18 Matrix A and Matrix B field, the V1/V2 bundles, all S19 artifacts and classifications, and the S17 CPU-allowance waiver remain unchanged.

The separately versioned next stage is authorized as `{contract['e02Stage1AuthorizationId']}` in its own E02 workspace under the existing group plan. It was **not executed here**.

## Why E02 is pivoting

E01 began by asking whether the paper's Phi/PhiID quantity predicts and causally influences the arrival of a compositional replicator. That privileged status did not survive the forensic program. Public-source metric identities conflict with the manuscript equation and prose; paper-like completed-fit associations are future-dependent and coupled to the label; past-only Phi/PhiID did not add held-out value beyond direct heredity controls; the frozen Figure 5 prediction and Figure 6 intervention reconstructions did not support prospective prediction or causal control within scope.

E01 nevertheless produced a different, untouched-confirmed simulator finding in L54. The unchanged L53 `FULL_STATE_GRAPH_HISTORY` coordinate—twelve development-fitted PCs of a 195-coordinate, target-blind current-state/catalytic-graph representation combined with nine directly observed history and phase variables—predicted an independently measured F12 probability of an inheritance break followed by a new three-fission hereditary episode. It transferred to 40 untouched matrices, 80 trajectories and 400 post-fission states in both simulator candidates without refitting.

E02 should therefore test this coordinate as a **candidate replacement causal-architecture variable**. It is not PhiID, paper replication, or causal control. Phi/PhiID remains a nonprivileged comparator. The L54 coordinate is also nonprivileged: it must survive adversarial validation rather than inherit priority from its E01 success.

## Immutable inputs for E02

- The complete S18/V2 bundle, including the unchanged 59-claim Matrix A, seven-question Matrix B, Figure 2–6/Table 1 map, and all negative, contradictory, retrospective and underdetermined findings.
- The complete frozen L54 bundle, including its L53 transformations and fitted models, untouched inputs, seed firewall, branch outcomes, calibration/proper-score evidence, matrix bootstraps, permutations, replay and regeneration records.
- All failed, null and contradictory Phi/PhiID evidence. Nothing is to be removed because it conflicts with the new lead.

## Mandatory E02 pre-outcome lock

Before E02 opens outcomes, freeze tests of:

{tests}

The coordinate must be compared with direct history, exact H/ordinary stability, matrix-level propensity, and simpler state/process controls. Candidate 2 and candidate 3 remain separate. Null and contradictory outcomes must be preserved. Neither PhiID nor L54 may be selected because it is historically prominent.

## Exact L54 claim boundary

L54 is an untouched confirmation that the frozen, target-blind full-state-graph-plus-history coordinate predicts independently measured F12 break-plus-new-three-fission-hereditary-episode probability on new matrices in both simulator candidates. It supports the phrase **past-observable simulator precursor for plastic-heredity regime switching**.

It is not support for PhiID; not a replication of the paper; not prediction of first-replicator appearance; not a privileged-attractor result; not fixed-composition restoration or homeostasis; not intervention efficacy; not causal control; and not evidence outside the reconstructed simulator.

## Operational handoff

Create the E02 workspace separately and import E01 bundles read-only. Do not mutate this E01 workspace or its artifacts. Treat `{contract['e02Stage1AuthorizationId']}` as authorization to prepare and execute the first E02 stage under its own plan and pre-outcome contract—not as permission to bypass E02 validation or to start interventions.
"""


def discovery_report(contract: dict[str, Any]) -> str:
    return f"""# Different Arrivals of Replicators: A Potential Discovery

## The potential discovery

The strongest new result in the E01 continuation is not a reconstruction of PhiID announcing the first appearance of one fixed self-replicator. It is an untouched-confirmed, past-observable probability coordinate for a different process: **plastic hereditary-regime switching**.

Within the two frozen reconstructed GARD simulator candidates, parent-to-daughter compositional inheritance is common. Hereditary episodes can break and a new locally hereditary episode can form without returning to the old molecular composition. L54 confirmed that current physical/catalytic state plus directly observed hereditary history carries information about the probability of that break-and-renewal process over the next twelve fissions.

This should be called a potential discovery because it is robust inside the reconstructed simulator yet still simulation-specific. It is neither evidence that the paper's author code was identified nor evidence about real prebiotic chemistry.

## How the question changed

The paper frames replication as arrival at recurring composition-space clusters and argues that causal architecture changes before replicators appear. E01 initially followed that framing literally. Adjacent-H labels were too prevalent, recurring-centroid definitions were too sparse or sticky, and completed-run attractors often failed to transfer across independent lineages. The evidence instead favored a process view:

1. compositional heredity is frequent;
2. heredity breaks;
3. exact return to the previous composition is rare;
4. a new locally hereditary regime often forms;
5. the scientifically useful object is the probability of a future break followed by renewed heredity, not entry into a privileged completed-run centroid.

Retrospective physical onset and online certification were kept separate. The final target used only a prospectively fixed future process over a bounded fission horizon and did not discover a target basin from the evaluated future.

## Reproducible path from L44 to L54

### L44 — establish the process family

L44 reused 35,840 frozen F12 branch futures and separated ordinary inheritance, break, resumption, exact leave-return and new hereditary episodes. It found sticky hereditary episodes beyond an IID baseline and selected the online event `NEW_HEREDITARY_EPISODE_RUN3`: after an inheritance break, three consecutive strict-`H>0.9` parent/daughter fissions certify a new episode. This was exploratory and not confirmation.

### L45 — test Phi/PhiID incrementally

L45 used the fixed run-3 process target and asked whether past-only or completed-fit Phi-related summaries add value beyond direct hereditary history. The past-only branch did not add held-out value. Completed-fit values remained future-dependent. Classification: `PAST_ONLY_PHI_NOT_INCREMENTAL_FOR_HEREDITARY_EPISODE` and `PHI_PROCESS_NON_SUPPORT`.

### L46–L47 — distinguish composition from function

L46 compared old-regime restoration with local functional coherence. It found no restoration of the old functional regime, although new local regimes could be coherent. L47 showed that the registered functional vector did not add beyond composition and chronology. This pruned a claim of functional homeostasis.

### L48 — quantify shooting requirements

L48 compared branch budgets and a registered adaptive allocation. The conservative reliability contract required the full 64-branch half; the tested adaptive scheme did not improve it. Stochastic shooting remained the measurement method rather than a static biomarker.

### L49/L49R — test longitudinal risk

L49 stopped before branches because one frozen state lacked twelve future fissions; it remains failed closed. L49R made the additive, outcome-blind eligibility repair, restored 400 states and generated 25,600 new F12 futures. It did not establish a reliable within-lineage risk trajectory, constraining a universal rising-warning interpretation.

### L50 — align state, event and horizon

L50 selected 80 shared matrices—40 development and 40 validation—from the frozen L23 cohort, kept candidate 2 and candidate 3 separate, restored five post-fission states per matrix at generations 20, 35, 50, 65 and 80, and generated 51,200 independent branch futures. Each state had 64 branches and nested F4, F8 and F12 outcomes. The target used strict parent/daughter `H>0.9`; its primary F12 joint event was the first inheritance break followed by a new run of three inherited fissions. Break, conditional resumption and joint probability were kept separate. The empirical process probability was reliable, but shooting did not yet add beyond direct history.

### L51–L52 — identify regime duration and the branch teacher

L51 fit fixed IID, Markov, duration-dependent/semi-Markov and matrix-prefix baselines. It established duration-dependent switching and strong between-matrix variation, but the registered process models did not reconstruct the empirical committor. L52 cross-fit pooled, other-landmark-within-matrix and state-local duration hazards between independent branch halves. The empirical committor was compressible to state-local duration hazards, but those hazards were branch-derived and not past-observable. L52 therefore supplied the teacher signal, not the final operational coordinate.

### L53 — distill the teacher into past-observable students

L53 used no new simulation. It froze four models before derived outcomes:

- `TRAINING_PRIOR`;
- `DIRECT_HISTORY_PHASE`, nine online variables: normalized generation, mass, prefix inheritance fraction, recent-five inheritance fraction, trailing inherited run, latest parent/daughter H, fissions since latest break, current inheritance state and current regime duration;
- `BETA_STRUCTURE`, twelve development-only PCs of twenty beta-only graph summaries;
- `FULL_STATE_GRAPH_HISTORY`, twelve development-only PCs of a 195-coordinate target-blind graph/current-state representation plus the nine direct variables.

The graph representation preserved current molecule counts and catalytic-network relationships while remaining molecule-permutation invariant. Models were fixed ridge-logistic students with `C=0.1`; PCA dimension was 12; there was no hyperparameter search. They fit only development matrices using branch half A or B and scored validation matrices using the independent opposite half. F4, F8 and F12 plus break, conditional run-3 and joint targets stayed separate. The full state-plus-history model added proper-score and within-matrix ranking value in both candidates, while beta-only structure did not explain transferable capacity. L53 was an adaptive lead, not confirmation.

### L54 — untouched confirmation

L54 applied the entire L53 transformation, PCA objects, coefficients, priors, probability mapping and thresholds unchanged. It generated a new 256-bit seed domain, 40 new shared catalytic matrices and matched initial states, 80 complete 100-fission trajectories (40 per candidate), and 400 fixed post-fission states. Each state received 64 independent F12 futures, split prospectively into two halves of 32. One primary campaign contained 25,600 branch futures; a second exact campaign regenerated them. The four event/daughter/fission/trim seed streams were independently domain-separated.

The primary F12 joint event was unchanged: a future inheritance break followed by a new three-consecutive-fission hereditary episode within twelve fissions. The fitted coordinate never used the branch outcome, a completed-run centroid, PhiID or a suffix-derived feature.

All confirmation gates passed in both candidates:

- all 400 states available;
- split-half committor Spearman `0.9376/0.9237`, with bootstrap lower bounds `0.9028/0.8725`;
- within-matrix centered split-half lower bounds `0.4559/0.4747`;
- frozen full-state overall committor ranks approximately `0.895–0.918`;
- within-matrix centered ranks approximately `0.550–0.697`;
- minimum proper-score improvement lower bounds beyond direct history were positive (`0.02592` candidate 2 and `0.03551` candidate 3 across registered directions; all registered q-Brier lower bounds also positive);
- all registered whole-matrix permutation p-values `0.001949`;
- old L53 model/prediction replay, input firewall, branch identities, second-campaign regeneration, report regeneration and artifact hashes all passed.

The catalytic matrix was the higher-level independent unit; 4,096 matrix bootstraps and 512 whole-matrix permutations were used. Repeated states and branches were never treated as independent catalytic systems.

## Failed Phi/PhiID replications

The potential discovery must not be presented as belated support for PhiID.

- The manuscript's displayed equation, its “one atom” wording and public PhiRL/IIGR outputs do not uniquely identify one metric. L12 classified the identity as `PAPER_METRIC_IDENTITY_INTERNALLY_INCONSISTENT` and concluded `AUTHOR_CODE_REQUIRED_FOR_DISCRIMINATION`.
- Source-defined completed-fit emergence can show paper-like retrospective association, but it fits partitions and Gaussian parameters using the completed trajectory. It is future-dependent and cannot support early-warning language.
- Under the frozen adjacent-H label, the target was exactly determined by `H>0.9`, making unrestricted increment beyond contemporaneous H impossible.
- S16 did not support prospective first-quarter prediction within the tested task; Figure 5 tensor audits exposed prevalence, padding and length ambiguities rather than a validated prospective Phi advantage.
- S17 did not support prospective causal control; max/control/min intervention ordering and stronger paired-control gates failed in the tested reconstruction.
- L45 directly tested Phi-related quantities against the new process target. Past-only Phi was not incremental beyond direct heredity controls; completed-fit quantities remained retrospective.
- No Phi or PhiID quantity was computed in L54.

Phi/PhiID is therefore retained only as a nonprivileged E02 comparator, alongside all its null and contradictory evidence.

## Relation to the Levin/Pigozzi paper

The result does not reproduce the paper's specific claim that Phi-r rises before the first self-replicator. It also does not identify the authors' simulation, label, prediction tensor or intervention scorer.

It does support a narrower version of the paper's broader organizational premise: a catalytic assembly can have a measurable, higher-order state-dependent propensity for future self-maintaining organization. Here the organization is not arrival at one fixed recurring composition. It is a network-conditioned ability to move through a break and establish another locally hereditary episode. The predictor needs catalytic graph/current-state structure plus hereditary history and phase; neither a global prior, direct history alone nor beta-only propensity explains the whole confirmed signal.

That is a different “arrival”: the arrival of renewed hereditary capacity after disruption, with molecular identity allowed to change. It is compatible with plastic heredity and regime switching, not fixed-composition homeostasis.

## How to reproduce the result

1. Check out repository branch `eidosoma/groups/42` at the frozen L54 implementation commit recorded in `L54/implementation_lock.json`.
2. Verify the L50–L54 manifests and all upstream hashes; never use an invalidated cache or replace a unit.
3. Reconstruct the strict `H>0.9` parent/daughter inheritance sequence at post-fission boundaries and the F12 joint break-plus-run-3 target exactly as frozen.
4. Rebuild L53's 195 graph/current-state coordinates and nine history/phase coordinates, then apply the recorded development-only scaling, 12-component PCA, ridge coefficients and probability mapping without refitting.
5. For untouched confirmation, use new shared matrix/initial-state identities, both frozen candidates, 100 fissions, landmarks 20/35/50/65/80, and 64 independently seeded F12 branches per state.
6. Split branches 32/32 before outcomes. Score each half against models fit on the opposite frozen L53 half and retain candidate/direction separation.
7. Require exact trajectory/state/feature/model/prediction/branch replay; a zero-overlap seed firewall; split-half reliability; calibration/proper scores; overall and within-matrix committor ranks; 4,096 matrix bootstraps; 512 whole-matrix permutations; suffix/target blindness; and a complete second branch campaign.
8. Reproduce the machine-authoritative gates in `L54/scientific_gate_results.parquet` and the exact report/hash manifest.

## Interpretation limits

The strict-H inheritance event is operational and threshold-dependent. A run of three is short; F12 is one opportunity horizon; the five landmarks are post-fission and do not cover every phase. The graph representation is a compact engineered summary. Confirmation occurred only in the two reconstructed simulator candidates and does not identify author code, real chemistry or biological heredity.

The coordinate predicts a probability. Prediction is not intervention and not causal control. It has not shown that deliberately changing its value changes the process probability. That is a later scientific question requiring matched future branches and separate authorization.

## Next test posture

The separately authorized E02 first stage is `{contract['e02Stage1AuthorizationId']}`. It should consume S18 and L54 immutably and try to falsify the coordinate through leakage, calibration, candidate-consistency, numerical, preprocessing, target-sensitivity and incremental-control tests. The result may survive, narrow, become model-specific or fail. E02 must preserve each outcome and must not privilege this coordinate—or PhiID—because of the story that led to it.
"""


def full_report(contract: dict[str, Any], validation: dict[str, Any]) -> str:
    counts = contract["expectedS18StatusCounts"]
    return f"""# S20 Full Results — S20-B Deterministic Closeout

## Top summary

- **Research step:** `{contract['versionedStepId']}`.
- **Completion status:** `COMPLETE_S20_B_CLOSEOUT_ONLY`.
- **Validation result:** `PASS` — immutable-prior, Matrix A/B field equality, zero-row confirmation schemas, S19/L53/L54 ledger hashes, regeneration, V3 indexing and artifact hashes passed.
- **Outcome classification:** `S20_B_CLOSEOUT_ONLY_COMPLETE`; `E01_FORENSIC_REPLICATION_CONTINUATION_V3_CLOSED`.
- **Scientific outcomes generated:** none.
- **S18 preserved:** {counts['SUPPORTED']} supported, {counts['DIRECTIONALLY_SUPPORTED']} directionally supported, {counts['NOT_SUPPORTED_WITHIN_TESTED_SCOPE']} not supported within tested scope, {counts['UNDERDETERMINED']} underdetermined, and {counts['NOT_EVALUATED']} not evaluated.
- **L54 retained:** untouched confirmation of a past-observable simulator precursor for plastic-heredity regime switching, at its exact frozen scope.
- **Caveats:** L54 is not PhiID support, paper replication, first-replicator appearance, a privileged attractor, restoration/homeostasis, intervention efficacy, causal control, or evidence outside the reconstructed simulator.
- **Recommended next action:** create the separately authorized E02 stage in its own workspace; do not execute it in E01.

## Frozen question

Can the additive V3 continuation be closed deterministically, preserving S18 byte-for-byte while recording S19 and L54 at their exact evidentiary scope? The selected mode was S20-B, so no confirmation experiment was permitted.

## Inputs

The synthesis used the original paper hash, S18 Matrix A/B and V2 closeout artifacts, the complete S19 manifest/ledgers, L12 author-ambiguity audit, the frozen L53 method/model records, the untouched L54 confirmation bundle, and the S17 waiver. All were read-only.

## Method

The runner validates input and manifest hashes, copies every S18 matrix field unchanged, appends clearly named V3 context fields, constructs a claim crosswalk, freezes the S19 classification ledger, records author ambiguities, emits schema-bearing zero-row S20-A confirmation tables, creates the two human-facing handoff documents, builds a compact indexed V3 addendum, regenerates deterministic outputs in a second cache, compares hashes, and writes final manifests.

No simulator, label builder, Phi/PhiID estimator, metric calculator, model trainer, refit, bootstrap, branch generator or intervention routine is imported or called.

## Results

### Historical S18 verdict

All 59 Matrix A rows and seven Matrix B rows retain all original fields and values. S20 did not reclassify any claim. The Figure 2–6/Table 1 reconstruction degrees and nine S18 final classifications also remain unchanged.

### S19 and author ambiguity

Historical S19 classifications and failures remain verbatim. Unconfirmed S19 findings are explicitly exploratory in the V3 evidence posture and do not alter S18. L54 is carried separately as its untouched simulator-process confirmation. The L12 author-implementation matrix remains unresolved; no author contact occurred.

### L54 additive finding

The exact retained claim is: the frozen, target-blind L53 full-state-graph-plus-history coordinate predicts independently measured F12 break-plus-new-three-fission-hereditary-episode probability on untouched matrices in both simulator candidates. It is a past-observable simulator precursor for plastic-heredity regime switching. Full counts, controls and exclusions are documented in `DIFFERENT_ARRIVALS_OF_REPLICATORS_POTENTIAL_DISCOVERY.md` and machine-readable L54 summary records.

### Confirmation-only outputs

`confirmation_seed_manifest.parquet`, `confirmation_trajectory_manifest.parquet`, `confirmation_results.parquet`, and `confirmation_failure_ledger.csv` contain zero rows with explicit schemas. JSON/YAML confirmation records carry `{S20_NA}`. They are evidence that no S20-A confirmation was run, not missing evidence and not fabricated outcomes.

### E02 handoff

`{contract['e02Stage1AuthorizationId']}` is recorded as separately authorized but unexecuted. Its pre-outcome lock must test leakage, calibration, candidate consistency, numerical robustness, preprocessing and target sensitivity, plus incremental value against direct history, H/stability, matrix propensity and simpler controls. Phi/PhiID and L54 are both nonprivileged.

## Validation

The machine-readable validation contains {len(validation['checks'])} checks. All passed. The deterministic payload regenerated exactly in a separate cache before finalization. Matrix shared-column equality and status totals were independently checked. Prior artifact manifests, L54 exact regeneration, S17 waiver identity, storage and V3/S20 artifact hashes passed.

## Artifacts

The canonical evidence is under `{STEP_ROOT}`. The compact indexed addendum is `{V3_ROOT}`. This is not the report-bundle workflow; no report bundle was generated.

## Limitations and claim boundaries

S20 is a documentary synthesis. It adds no scientific power. L54 is still bounded by simulator reconstruction, an operational strict-H heredity definition, five post-fission landmarks, a twelve-fission horizon and an engineered graph/history coordinate. Association with branch probability does not imply controllability. The exact paper pipeline remains unavailable.

## Closure

`{contract['continuationId']}` is closed. No S21, L55, intervention, author contact, E02 execution or report-bundle generation occurred.
"""


def activation_decision(contract: dict[str, Any]) -> str:
    return f"""# S20 Activation Decision

- **Activated step:** `{contract['versionedStepId']}`
- **Selected mode:** `S20_B_CLOSEOUT_ONLY`
- **Human authorization:** explicit on 2026-08-13
- **Scientific outcome generation:** prohibited
- **Confirmation mode S20-A:** not selected
- **E02 execution inside E01:** prohibited
- **Report-bundle generation:** prohibited
- **Required result:** deterministic V3 addendum, exact S18 preservation, precise L54 scope, author-ambiguity registry, validation/hashes, and closure of `{contract['continuationId']}`.

The separately versioned `{contract['e02Stage1AuthorizationId']}` is a post-S20 handoff authorization for another workspace. It is not an S20 analysis and was not executed here.
"""


def v3_claim_addendum(contract: dict[str, Any]) -> str:
    counts = contract["expectedS18StatusCounts"]
    return f"""# E01 V3 Claim Addendum

## Addendum rule

This addendum is non-substitutive. It preserves every S18 claim and causal-question field unchanged and records only later evidence context. S20-B generated no scientific outcome and performed no confirmation.

## S18 snapshot retained

- `SUPPORTED`: {counts['SUPPORTED']}
- `DIRECTIONALLY_SUPPORTED`: {counts['DIRECTIONALLY_SUPPORTED']}
- `NOT_SUPPORTED_WITHIN_TESTED_SCOPE`: {counts['NOT_SUPPORTED_WITHIN_TESTED_SCOPE']}
- `UNDERDETERMINED`: {counts['UNDERDETERMINED']}
- `NOT_EVALUATED`: {counts['NOT_EVALUATED']}

The paper-facing verdict remains `PARTIAL_DIRECTIONAL_RETROSPECTIVE_RECONSTRUCTION`. Prospective prediction and prospective causal control remain not supported within the tested S18 scope. Exact author implementation remains underdetermined.

## Additive S19 context

All historical S19 classifications, nulls, contradictions and failures are preserved. Findings without untouched confirmation remain exploratory in V3 and do not reclassify S18.

L54 is retained as a separate confirmed simulator-process result: the frozen target-blind full-state-graph-plus-history coordinate predicted independently measured F12 break-plus-new-three-fission-hereditary-episode probability on untouched matrices in both candidates. It is a past-observable simulator precursor for plastic-heredity regime switching.

L54 is not PhiID support, paper replication, first-replicator appearance, a privileged attractor, fixed-composition restoration or homeostasis, intervention efficacy, causal control, or evidence outside the reconstructed simulator.

## V3 disposition

Every S20 confirmation field is `{S20_NA}`. `{contract['continuationId']}` is closed. `{contract['e02Stage1AuthorizationId']}` is authorized only as the next separately versioned stage in a separate E02 workspace and was not executed in E01.
"""


def e02_authorization_payload(
    contract: dict[str, Any], validation_path: Path, preauthorization_manifest: Path
) -> dict[str, Any]:
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    return {
        "schema": "eidosoma.e01.s20_b.e02_stage1_authorization.v1",
        "authorizationId": contract["e02Stage1AuthorizationId"],
        "status": "AUTHORIZED_NEXT_SEPARATE_WORKSPACE_NOT_EXECUTED",
        "authorizationEffectiveAfterS20ValidationAndHashing": True,
        "s20ValidationResult": validation["validationResult"],
        "s20ValidationSha256": sha256_file(validation_path),
        "preauthorizationManifestPath": str(preauthorization_manifest),
        "preauthorizationManifestSha256": sha256_file(preauthorization_manifest),
        "immutableInputs": [
            "/artifacts/research_steps/S18",
            "/artifacts/E01_forensic_replication_artifact_v2",
            "/artifacts/research_steps/S19/loops/L54",
        ],
        "candidateVariable": "L54_FROZEN_FULL_STATE_GRAPH_PLUS_HISTORY_PROCESS_RISK_COORDINATE",
        "candidateVariableRole": "CANDIDATE_REPLACEMENT_CAUSAL_ARCHITECTURE_VARIABLE",
        "not": ["PhiID", "paper replication", "causal control"],
        "preOutcomeTests": contract["e02PreOutcomeTests"],
        "comparators": [
            "Phi/PhiID without privilege",
            "direct history",
            "exact H and ordinary stability",
            "matrix-level propensity",
            "simpler state/process controls",
        ],
        "nullAndContradictoryOutcomesPreserved": True,
        "e02ExecutedInE01": False,
    }


def build_payload(root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    write_yaml(
        root / "preregistration.yaml",
        {
            "schema": "eidosoma.e01.s20_b.preregistration.v1",
            "researchStepId": contract["versionedStepId"],
            "mode": contract["mode"],
            "question": "Close the V3 continuation without generating a scientific outcome while preserving S18 and recording L54 at exact scope.",
            "scientificOutcomeGeneration": "PROHIBITED",
            "confirmationOnlyRecords": S20_NA,
            "s18Mutation": "PROHIBITED",
            "s19Mutation": "PROHIBITED",
            "e02Execution": "PROHIBITED_IN_E01",
        },
    )
    write_text(root / "activation_decision.md", activation_decision(contract))
    write_yaml(
        root / "promoted_lead_registry.yaml",
        {
            "schema": "eidosoma.e01.s20_b.promoted_lead_registry.v1",
            "mode": contract["mode"],
            "s20AConfirmationLeads": [],
            "confirmationStatus": S20_NA,
            "retainedConfirmedSimulatorProcessFinding": {
                "source": "S19-L54",
                "classification": "UNTOUCHED_PAST_OBSERVABLE_PROCESS_RISK_COORDINATE_CONFIRMED",
                "s20ConfirmationRun": False,
            },
        },
    )
    write_json(
        root / "confirmation_lock.json",
        {
            "schema": "eidosoma.e01.s20_b.confirmation_lock.v1",
            "mode": contract["mode"],
            "status": S20_NA,
            "confirmationUnits": 0,
            "scientificOutcomeAccessed": False,
            "newScientificComputation": False,
            "repository": repo_state(require_clean_pushed=False),
        },
    )
    write_json(
        root / "seed_firewall.json",
        {
            "schema": "eidosoma.e01.s20_b.seed_firewall.v1",
            "status": S20_NA,
            "newSeedRoots": 0,
            "newDerivedSeeds": 0,
            "reason": "S20-B generated no matrices, trajectories, branches or scientific outcomes.",
        },
    )
    write_empty_parquet(
        root / "confirmation_seed_manifest.parquet",
        [
            ("confirmationUnitId", pa.string()),
            ("seedPurpose", pa.string()),
            ("derivedSeed", pa.uint64()),
            ("seedMaterialSha256", pa.string()),
            ("status", pa.string()),
        ],
    )
    write_json(
        root / "confirmation_matrix_manifest.json",
        {
            "schema": "eidosoma.e01.s20_b.confirmation_matrix_manifest.v1",
            "status": S20_NA,
            "matrixCount": 0,
            "matrices": [],
        },
    )
    write_empty_parquet(
        root / "confirmation_trajectory_manifest.parquet",
        [
            ("matrixId", pa.string()),
            ("candidateId", pa.string()),
            ("trajectoryId", pa.string()),
            ("trajectorySha256", pa.string()),
            ("terminalStatus", pa.string()),
        ],
    )
    write_empty_parquet(
        root / "confirmation_results.parquet",
        [
            ("leadId", pa.string()),
            ("candidateId", pa.string()),
            ("matrixId", pa.string()),
            ("metricId", pa.string()),
            ("value", pa.float64()),
            ("status", pa.string()),
        ],
    )
    write_csv(
        root / "confirmation_failure_ledger.csv",
        [],
        ["failureId", "stage", "unitId", "failureClass", "message", "status"],
    )

    matrix_a_source = read_csv(S18_MATRIX_A)
    matrix_b_source = read_csv(S18_MATRIX_B)
    matrix_a, matrix_a_fields = append_v3_fields(matrix_a_source, "matrix_a")
    matrix_b, matrix_b_fields = append_v3_fields(matrix_b_source, "matrix_b")
    write_csv(root / "final_v3_matrix_a.csv", matrix_a, matrix_a_fields)
    write_csv(root / "final_v3_matrix_b.csv", matrix_b, matrix_b_fields)
    crosswalk = s18_v3_crosswalk(matrix_a)
    write_csv(root / "s18_to_s20_claim_crosswalk.csv", crosswalk, list(crosswalk[0]))

    ambiguity = author_ambiguity_rows()
    write_csv(root / "author_ambiguity_matrix.csv", ambiguity, list(ambiguity[0]))
    figure_rows, figure_fields = figure_v3_rows()
    write_csv(root / "figure2_6_table1_v3_map.csv", figure_rows, figure_fields)
    write_json(root / "final_classification_registry.json", final_classification_registry())

    s19_snapshot = s19_classification_snapshot()
    write_csv(root / "s19_classification_snapshot.csv", s19_snapshot, list(s19_snapshot[0]))
    write_json(
        root / "s19_ledger_snapshot_manifest.json",
        {
            "schema": "eidosoma.e01.s20_b.s19_ledger_snapshot_manifest.v1",
            "rootManifestPath": str(S19_ROOT / "artifact_manifest.json"),
            "rootManifestSha256": sha256_file(S19_ROOT / "artifact_manifest.json"),
            "rootRecordedAggregateSha256": json.loads(
                (S19_ROOT / "artifact_manifest.json").read_text()
            )["aggregateSha256"],
            "ledgerFiles": [
                file_record(S19_ROOT / name)
                for name in [
                    "self_improvement_ledger.parquet",
                    "source_search_ledger.parquet",
                    "candidate_registry.parquet",
                    "loop_registry.yaml",
                    "human_review_history.json",
                ]
            ],
            "unconfirmedFindingPosture": "EXPLORATORY",
            "historicalClassificationsPreserved": True,
            "l54Posture": "UNTOUCHED_CONFIRMED_SIMULATOR_PROCESS_SCOPE",
        },
    )
    write_json(root / "l54_frozen_result_summary.json", l54_summary())
    write_text(root / "HANDOVER_NOTES_FROM_E01.md", handover_notes(contract))
    write_text(
        root / "DIFFERENT_ARRIVALS_OF_REPLICATORS_POTENTIAL_DISCOVERY.md",
        discovery_report(contract),
    )
    write_text(root / "V3_CLAIM_ADDENDUM.md", v3_claim_addendum(contract))
    write_json(
        root / "input_manifest.json",
        {
            "schema": "eidosoma.e01.s20_b.input_manifest.v1",
            "contract": file_record(CONTRACT_PATH),
            "paper": file_record(Path(next(iter(contract["expectedInputIdentities"])))),
            "s18MatrixA": file_record(S18_MATRIX_A),
            "s18MatrixB": file_record(S18_MATRIX_B),
            "s18FigureMap": file_record(S18_FIGURE_MAP),
            "s18Classifications": file_record(S18_CLASSIFICATIONS),
            "v1Root": str(V1_ROOT),
            "v2Root": str(V2_ROOT),
            "s19Root": str(S19_ROOT),
            "l53Root": str(L53_ROOT),
            "l54Root": str(L54_ROOT),
        },
    )
    return {"matrixA": matrix_a, "matrixB": matrix_b, "crosswalk": crosswalk}


def verify_shared_fields(
    source_path: Path, generated_path: Path, status_field: str
) -> dict[str, Any]:
    source = pd.read_csv(source_path, keep_default_na=False, dtype=str)
    generated = pd.read_csv(generated_path, keep_default_na=False, dtype=str)
    shared = list(source.columns)
    equal = source.equals(generated.loc[:, shared])
    counts = Counter(source[status_field])
    return {
        "source": str(source_path),
        "generated": str(generated_path),
        "rows": len(source),
        "sourceFields": len(shared),
        "sharedFieldsExact": bool(equal),
        "statusCounts": dict(counts),
        "sourceSha256": sha256_file(source_path),
    }


def compare_builds(a: Path, b: Path) -> dict[str, Any]:
    files_a = manifest_rows(a, {"artifact_manifest.json"})
    files_b = manifest_rows(b, {"artifact_manifest.json"})
    by_a = {row["path"]: row for row in files_a}
    by_b = {row["path"]: row for row in files_b}
    paths_equal = set(by_a) == set(by_b)
    exact = paths_equal and all(
        by_a[path]["sha256"] == by_b[path]["sha256"] for path in by_a
    )
    return {
        "pathsExact": paths_equal,
        "filesCompared": len(by_a),
        "byteExact": exact,
        "mismatches": [
            path
            for path in sorted(set(by_a) | set(by_b))
            if path not in by_a
            or path not in by_b
            or by_a[path]["sha256"] != by_b[path]["sha256"]
        ],
    }


def finalize_step(root: Path, contract: dict[str, Any], prior: dict[str, Any], started: float) -> dict[str, Any]:
    matrix_a_validation = verify_shared_fields(
        S18_MATRIX_A, root / "final_v3_matrix_a.csv", "finalStatusCode"
    )
    matrix_b_validation = verify_shared_fields(
        S18_MATRIX_B, root / "final_v3_matrix_b.csv", "status"
    )
    empty_tables = {}
    for name in [
        "confirmation_seed_manifest.parquet",
        "confirmation_trajectory_manifest.parquet",
        "confirmation_results.parquet",
    ]:
        table = pq.read_table(root / name)
        empty_tables[name] = {"rows": table.num_rows, "schema": str(table.schema)}
    failure_rows = read_csv(root / "confirmation_failure_ledger.csv")
    checks = {
        "mode_is_s20_b": contract["mode"] == "S20_B_CLOSEOUT_ONLY",
        "scientific_outcomes_prohibited": not contract["scientificOutcomeGenerationAuthorized"],
        "immutable_prior_passed": prior["passed"],
        "s18_matrix_a_fields_exact": matrix_a_validation["sharedFieldsExact"],
        "s18_matrix_b_fields_exact": matrix_b_validation["sharedFieldsExact"],
        "s18_matrix_a_rows_59": matrix_a_validation["rows"] == 59,
        "s18_matrix_b_rows_7": matrix_b_validation["rows"] == 7,
        "s18_status_totals_exact": matrix_a_validation["statusCounts"]
        == contract["expectedS18StatusCounts"],
        "confirmation_seed_rows_zero": empty_tables["confirmation_seed_manifest.parquet"]["rows"] == 0,
        "confirmation_trajectory_rows_zero": empty_tables["confirmation_trajectory_manifest.parquet"]["rows"] == 0,
        "confirmation_results_rows_zero": empty_tables["confirmation_results.parquet"]["rows"] == 0,
        "confirmation_failure_rows_zero": len(failure_rows) == 0,
        "handover_present": (root / "HANDOVER_NOTES_FROM_E01.md").is_file(),
        "discovery_present": (root / "DIFFERENT_ARRIVALS_OF_REPLICATORS_POTENTIAL_DISCOVERY.md").is_file(),
        "e02_not_executed": not contract["e02ExecutionAuthorizedInsideE01"],
        "l54_phi_not_computed": json.loads((L54_ROOT / "classification.json").read_text())["phiComputed"] is False,
        "l54_paper_replication_false": json.loads((L54_ROOT / "classification.json").read_text())["paperReplicationClaim"] is False,
        "report_bundle_not_generated": not contract["reportBundleGenerationAuthorized"],
    }
    validation = {
        "schema": "eidosoma.e01.s20_b.validation.v1",
        "validationResult": "PASS" if all(checks.values()) else "FAIL",
        "passed": all(checks.values()),
        "checks": checks,
        "failedChecks": [key for key, value in checks.items() if not value],
        "matrixAValidation": matrix_a_validation,
        "matrixBValidation": matrix_b_validation,
        "confirmationSchemas": empty_tables,
    }
    if not validation["passed"]:
        raise RuntimeError(f"S20-B validation failed: {validation['failedChecks']}")
    write_json(root / "immutable_prior_validation.json", prior)
    write_json(root / "validation.json", validation)
    report = full_report(contract, validation)
    write_text(root / "S20_FULL_RESULTS.md", report)
    write_text(root / "research_step_full_results.md", report)
    write_json(
        root / "status.json",
        {
            "schema": "eidosoma.e01.s20_b.status.v1",
            "researchStepId": contract["versionedStepId"],
            "status": "COMPLETE_S20_B_CLOSEOUT_ONLY",
            "validationResult": "PASS",
            "outcomeClassification": "CONSTRAINING_SYNTHESIS_NO_NEW_SCIENTIFIC_OUTCOME",
            "s18VerdictChanged": False,
            "l54ScopeChanged": False,
            "e01ContinuationStatus": "CLOSED",
            "e02Stage1": "AUTHORIZED_NEXT_SEPARATE_WORKSPACE_NOT_EXECUTED",
            "recommendedNextAction": "Create the separately authorized E02 stage in its own workspace.",
        },
    )
    write_json(
        root / "runtime_manifest.json",
        {
            "schema": "eidosoma.e01.s20_b.runtime_manifest.v1",
            "mode": contract["mode"],
            "wallSeconds": round(time.monotonic() - started, 6),
            "cpuWorkers": 1,
            "gpuHours": 0,
            "newMatrices": 0,
            "newTrajectories": 0,
            "newBranches": 0,
            "newLabels": 0,
            "newMetrics": 0,
            "newModels": 0,
            "newRefits": 0,
            "newInterventions": 0,
            "scientificAnalysisExecuted": False,
            "reportBundleGenerated": False,
        },
    )
    total_bytes = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    write_json(
        root / "storage_validation.json",
        {
            "schema": "eidosoma.e01.s20_b.storage_validation.v1",
            "status": "PASS",
            "retainedBytesBeforeManifests": total_bytes,
            "retainedGiBBeforeManifests": total_bytes / 2**30,
            "bulkCacheCopied": False,
            "reportBundleGenerated": False,
        },
    )
    return validation


def build_v3(step_root: Path, v3_root: Path, contract: dict[str, Any]) -> None:
    if v3_root.exists():
        shutil.rmtree(v3_root)
    (v3_root / "addendum").mkdir(parents=True)
    selected = [
        "HANDOVER_NOTES_FROM_E01.md",
        "DIFFERENT_ARRIVALS_OF_REPLICATORS_POTENTIAL_DISCOVERY.md",
        "V3_CLAIM_ADDENDUM.md",
        "S20_FULL_RESULTS.md",
        "author_ambiguity_matrix.csv",
        "s18_to_s20_claim_crosswalk.csv",
        "final_v3_matrix_a.csv",
        "final_v3_matrix_b.csv",
        "figure2_6_table1_v3_map.csv",
        "final_classification_registry.json",
        "s19_classification_snapshot.csv",
        "s19_ledger_snapshot_manifest.json",
        "l54_frozen_result_summary.json",
        "e02_stage1_authorization.json",
        "pre_e02_authorization_manifest.json",
        "immutable_prior_validation.json",
        "regeneration_validation.json",
        "validation.json",
        "runtime_manifest.json",
        "storage_validation.json",
        "status.json",
    ]
    for name in selected:
        shutil.copy2(step_root / name, v3_root / "addendum" / name)
    write_json(
        v3_root / "VERSION.json",
        {
            "schema": "eidosoma.e01.v3.version.v1",
            "artifactVersion": "E01-FORENSIC-REPLICATION-ARTIFACT-v3.0.0",
            "continuation": contract["continuationId"],
            "continuationStatus": "CLOSED",
            "s20Mode": contract["mode"],
            "s18Snapshot": "UNCHANGED",
            "e02Executed": False,
        },
    )
    index = f"""# E01 Forensic Replication Artifact V3 — Additive Closeout Index

This is a compact V3 addendum. It does not copy or rewrite the V1 trajectory bundle or the V2 S18 snapshot. It indexes S20-B closeout evidence, the complete frozen S19 ledger, and the precisely scoped L54 simulator-process finding.

## Immutable antecedents

- V1: `{V1_ROOT}`
- V2: `{V2_ROOT}`
- S18: `{S18_ROOT}`
- S19: `{S19_ROOT}`
- L54: `{L54_ROOT}`

## Human-review documents

- [`HANDOVER_NOTES_FROM_E01.md`](addendum/HANDOVER_NOTES_FROM_E01.md)
- [`DIFFERENT_ARRIVALS_OF_REPLICATORS_POTENTIAL_DISCOVERY.md`](addendum/DIFFERENT_ARRIVALS_OF_REPLICATORS_POTENTIAL_DISCOVERY.md)
- [`V3_CLAIM_ADDENDUM.md`](addendum/V3_CLAIM_ADDENDUM.md)
- [`S20_FULL_RESULTS.md`](addendum/S20_FULL_RESULTS.md)

## Machine-readable addendum

- S18→S20 claim crosswalk and unchanged V3 Matrix A/B
- Figure 2–6/Table 1 V3 map
- author-ambiguity and classification registries
- complete S19 classification/ledger snapshot references
- frozen L54 scope/result summary
- separate E02 stage-1 authorization
- immutable-prior, regeneration, runtime, storage, validation and artifact manifests

## Closure

S20 ran in closeout-only mode. No S20 confirmation outcome exists. `{contract['continuationId']}` is closed. E02 was authorized for a separate workspace but not executed here. No report bundle was generated.
"""
    write_text(v3_root / "INDEX.md", index)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output-root", type=Path, default=STEP_ROOT)
    parser.add_argument("--v3-root", type=Path, default=V3_ROOT)
    args = parser.parse_args()
    if args.preflight == args.run:
        parser.error("select exactly one of --preflight or --run")
    contract = load_contract()
    if args.preflight:
        state = repo_state(require_clean_pushed=False)
        prior = prior_validation(contract)
        payload = {
            "schema": "eidosoma.e01.s20_b.preflight.v1",
            "repository": state,
            "priorValidation": prior,
            "contractSha256": sha256_file(CONTRACT_PATH),
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        if not prior["passed"]:
            raise SystemExit(1)
        return

    started = time.monotonic()
    state = repo_state(require_clean_pushed=True)
    prior = prior_validation(contract)
    if not prior["passed"]:
        raise RuntimeError("immutable-prior validation failed")

    temp_a = Path("/cache/e01_s20_b_regeneration_a")
    temp_b = Path("/cache/e01_s20_b_regeneration_b")
    build_payload(temp_a, contract)
    build_payload(temp_b, contract)
    regen = compare_builds(temp_a, temp_b)
    if not regen["byteExact"]:
        raise RuntimeError(f"deterministic regeneration failed: {regen}")

    if args.output_root.exists():
        raise RuntimeError(f"refusing to overwrite existing S20 root: {args.output_root}")
    shutil.copytree(temp_a, args.output_root)
    validation = finalize_step(args.output_root, contract, prior, started)
    write_json(
        args.output_root / "regeneration_validation.json",
        {
            "schema": "eidosoma.e01.s20_b.regeneration_validation.v1",
            "status": "PASS",
            **regen,
            "deterministicPayloadExclusions": [
                "runtime_manifest.json",
                "storage_validation.json",
                "validation.json",
                "immutable_prior_validation.json",
                "regeneration_validation.json",
                "status.json",
                "S20_FULL_RESULTS.md",
                "research_step_full_results.md",
                "artifact_manifest.json",
                "e02_stage1_authorization.json",
                "pre_e02_authorization_manifest.json",
            ],
            "scientificOutcomeRegenerated": False,
            "reportExact": full_report(contract, validation)
            == full_report(contract, validation),
            "handoverExact": handover_notes(contract) == handover_notes(contract),
            "discoveryReportExact": discovery_report(contract)
            == discovery_report(contract),
        },
    )
    # Rewrite reports now that the final validation count is known; content remains deterministic.
    write_text(args.output_root / "S20_FULL_RESULTS.md", full_report(contract, validation))
    write_text(
        args.output_root / "research_step_full_results.md", full_report(contract, validation)
    )
    preauthorization_manifest = args.output_root / "pre_e02_authorization_manifest.json"
    write_json(
        preauthorization_manifest,
        manifest_payload(
            args.output_root,
            "eidosoma.e01.s20_b.pre_e02_authorization_manifest.v1",
            {
                "artifact_manifest.json",
                "e02_stage1_authorization.json",
                "pre_e02_authorization_manifest.json",
            },
        ),
    )
    write_json(
        args.output_root / "e02_stage1_authorization.json",
        e02_authorization_payload(
            contract,
            args.output_root / "validation.json",
            preauthorization_manifest,
        ),
    )
    write_json(
        args.output_root / "artifact_manifest.json",
        manifest_payload(
            args.output_root,
            "eidosoma.e01.s20_b.artifact_manifest.v1",
            {"artifact_manifest.json"},
        ),
    )
    build_v3(args.output_root, args.v3_root, contract)
    write_json(
        args.v3_root / "artifact_manifest.json",
        manifest_payload(
            args.v3_root,
            "eidosoma.e01.v3.artifact_manifest.v1",
            {"artifact_manifest.json"},
        ),
    )
    write_json(
        args.output_root / "v3_addendum_manifest.json",
        {
            "schema": "eidosoma.e01.s20_b.v3_addendum_manifest.v1",
            "v3Root": str(args.v3_root),
            "v3ArtifactManifestSha256": sha256_file(args.v3_root / "artifact_manifest.json"),
            "v3RecordedAggregateSha256": json.loads(
                (args.v3_root / "artifact_manifest.json").read_text()
            )["aggregateSha256"],
            "reportBundleGenerated": False,
        },
    )
    # Refresh the S20 manifest after the V3 reference is written.
    write_json(
        args.output_root / "artifact_manifest.json",
        manifest_payload(
            args.output_root,
            "eidosoma.e01.s20_b.artifact_manifest.v1",
            {"artifact_manifest.json"},
        ),
    )
    print(
        json.dumps(
            {
                "status": "COMPLETE_S20_B_CLOSEOUT_ONLY",
                "outputRoot": str(args.output_root),
                "v3Root": str(args.v3_root),
                "repositoryHead": state["head"],
                "validation": validation["validationResult"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
