#!/usr/bin/env python3
"""Disclose S19-L10 technical repair 001 without changing science."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ARTIFACT_ROOT = Path("/artifacts/research_steps/S19")
LOOP_ROOT = ARTIFACT_ROOT / "loops/L10"
REPO = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve()
REPAIR_ID = "S19-L10-TECHNICAL-REPAIR-001"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, text=True, capture_output=True
    ).stdout.strip()


def write_manifest(path: Path, root: Path, schema: str) -> None:
    rows = []
    for item in sorted(
        candidate
        for candidate in root.rglob("*")
        if candidate.is_file() and candidate != path
    ):
        rows.append(
            {
                "path": str(item.relative_to(root)),
                "bytes": item.stat().st_size,
                "sha256": sha256_file(item),
            }
        )
    write_json(
        path,
        {
            "schema": schema,
            "root": str(root),
            "fileCount": len(rows),
            "totalBytes": sum(item["bytes"] for item in rows),
            "files": rows,
            "generatedAtUtc": utc_now(),
        },
    )


def amend_full_report(text: str) -> str:
    marker = "## Authorized technical regeneration repair"
    if marker in text:
        raise RuntimeError("L10 full report technical disclosure already exists")
    validation = "source-hash, and artifact-integrity gates passed."
    replacement = (
        "source-hash, and artifact-integrity gates passed. The first regeneration attempt is preserved: "
        "it passed 400/400 trajectories and 13/14 table hashes, while the sole fingerprint-table failure "
        "had zero differing cells after diagnostic column alignment. Following explicit human authorization, "
        "technical repair 001 fixed only schema-order canonicalization and the complete fresh rerun passed "
        "400/400 trajectories and 14/14 tables without a scientific value change."
    )
    if validation not in text:
        raise RuntimeError("expected validation sentence absent from L10 report")
    text = text.replace(validation, replacement, 1)
    caveat = "No L10 result establishes author-code identity, prediction, intervention efficacy, or causal control."
    caveat_replacement = (
        caveat
        + " The initial column-order replay failure and its explicitly authorized value-preserving repair are "
        "retained as provenance; they do not strengthen the negative scientific result."
    )
    if caveat not in text:
        raise RuntimeError("expected caveat sentence absent from L10 report")
    text = text.replace(caveat, caveat_replacement, 1)
    section = f"""
## Authorized technical regeneration repair

The first complete regeneration reproduced all 400 trajectories and 13 of 14 authoritative table hashes. `label_fingerprint_results.parquet` had the same 400 rows, the same column set, and zero differing cells after a diagnostic alignment, but its raw column order differed because the first completed parallel worker established DataFrame insertion order. The locked comparison had canonicalized rows but not columns, so it correctly raised instead of silently relaxing the gate.

The human then explicitly directed: “if it was a technical problem, fix and rerun.” Repair `{REPAIR_ID}` was frozen, committed, and pushed before rerun. It preserved the failed validation, comparison, trajectory-replay, and runtime artifacts under `*_failed_attempt_001.*`; left the scientific runner, core, config, seeds, trajectories, estimands, labels, controls, bootstrap, and gates unchanged; and added only lexicographic column canonicalization to the table comparator. A fresh cache reran all 400 trajectories and all 14 tables. The repaired comparison passed 400/400 trajectory and 14/14 table gates with exact dtypes and cells after schema alignment and reported no scientific value change. This is a disclosed post-outcome technical repair, not extra scientific specification search.

The combined runtime counts both regeneration attempts. `technical_repair_001.json`, `technical_repair_lock_001.json`, `technical_repair_release_gate_001.json`, `regeneration_validation_failed_attempt_001.json`, `regeneration_validation.json`, and `technical_repair_completion_001.json` provide the audit chain.

"""
    anchor = "## Outcome and next action"
    if anchor not in text:
        raise RuntimeError("outcome anchor absent from L10 report")
    return text.replace(anchor, section + anchor, 1)


def amend_summary(text: str) -> str:
    marker = "## Technical-repair disclosure"
    if marker in text:
        raise RuntimeError("L10 decision-summary disclosure already exists")
    return (
        text.rstrip()
        + f"""

## Technical-repair disclosure

The first exact-regeneration attempt passed 400/400 trajectories and 13/14 tables; the sole failed table had zero different cells after diagnostic column alignment but a different scheduling-dependent column order. After explicit human authorization, `{REPAIR_ID}` preserved that failure, changed only column-order canonicalization, reran the complete scope in fresh caches, and passed 400/400 trajectories and 14/14 tables with no scientific value change. The repaired validation permits the locked scientific classification; it does not make the negative result more favorable.
"""
    )


def main() -> None:
    repair = json.loads(
        (LOOP_ROOT / "technical_repair_001.json").read_text(encoding="utf-8")
    )
    validation = json.loads(
        (LOOP_ROOT / "regeneration_validation.json").read_text(encoding="utf-8")
    )
    initial = json.loads(
        (LOOP_ROOT / "regeneration_validation_failed_attempt_001.json").read_text(
            encoding="utf-8"
        )
    )
    classification = json.loads(
        (LOOP_ROOT / "classification.json").read_text(encoding="utf-8")
    )
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/eidosoma/groups/42")
    if (
        head != remote
        or git("branch", "--show-current") != "eidosoma/groups/42"
        or git("status", "--porcelain=v1")
    ):
        raise RuntimeError("reporting-amendment repository release gate failed")
    if not (
        repair["scientificValueChanged"] is False
        and initial["trajectoryReplayPassCount"] == 400
        and initial["scientificTablePassCount"] == 13
        and validation["passed"]
        and validation["trajectoryReplayPassCount"] == 400
        and validation["scientificTablePassCount"] == 14
        and validation["scientificValueChangeObserved"] is False
        and classification["decision"] == "RECURRING_ATTRACTOR_LABEL_NOT_RECONSTRUCTED"
    ):
        raise RuntimeError("technical-repair reporting predicates failed")

    completion = {
        "schema": "eidosoma.e01.s19_l10.technical_repair_completion.v1",
        "repairId": REPAIR_ID,
        "initialTrajectoryReplay": "400/400",
        "initialTableReplay": "13/14",
        "initialFailurePreserved": True,
        "initialDiagnosticCellDifferencesAfterAlignment": repair["failureObserved"][
            "cellDifferencesAfterColumnAlignment"
        ],
        "repairedTrajectoryReplay": "400/400",
        "repairedTableReplay": "14/14",
        "scientificCodeChanged": False,
        "scientificMethodChanged": False,
        "scientificValueChanged": False,
        "classificationBeforeDisclosure": classification["decision"],
        "classificationAfterDisclosure": classification["decision"],
        "reportingAmendmentCommit": head,
        "reportingAmendmentScript": str(SCRIPT.relative_to(REPO)),
        "reportingAmendmentScriptSha256": sha256_file(SCRIPT),
        "passed": True,
        "completedAtUtc": utc_now(),
    }
    write_json(LOOP_ROOT / "technical_repair_completion_001.json", completion)
    (LOOP_ROOT / "technical_repair_completion_001.md").write_text(
        f"""# S19-L10 technical repair 001 completion

## Concise top summary

- **Research step ID:** `S19-L10`.
- **Completion status:** technical repair 001 complete; L10 complete at mandatory human review.
- **Artifacts written:** preserved failed-attempt evidence, repair decision/lock/release/runtime records, repaired regeneration evidence, this completion record, and reporting amendment 001.
- **Validation result:** initial 400/400 trajectory and 13/14 table replay preserved; fresh repaired rerun passed 400/400 trajectories and 14/14 tables with exact cells/dtypes after fixed schema canonicalization.
- **Outcome classification:** unchanged `RECURRING_ATTRACTOR_LABEL_NOT_RECONSTRUCTED`; `EXPLORATORY_NON_SUPPORT`, `AUTHOR_AMBIGUITY_UNRESOLVED`, `NOT_PROMOTABLE`.
- **Caveats or blockers:** post-outcome, explicitly human-authorized technical repair; no scientific code, method, value, threshold, seed, label, control, or gate changed.
- **Recommended next action:** mandatory human review only; no automatic continuation.

The initial mismatch was schedule-dependent column order in one table. Diagnostic alignment found zero cell differences. Repair `{REPAIR_ID}` canonicalized columns lexicographically, reran the full scope in fresh caches, and passed. Both the failed and passing attempts remain hashed.
""",
        encoding="utf-8",
    )
    write_json(
        LOOP_ROOT / "reporting_amendment_001.json",
        {
            "schema": "eidosoma.e01.s19_l10.reporting_amendment.v1",
            "amendmentId": "S19-L10-REPORTING-AMENDMENT-001",
            "purpose": "Disclose the initial exact-replay failure, explicit human authorization, value-preserving repair, full rerun, and combined runtime in reports/status/ledgers.",
            "scientificValueChanged": False,
            "scientificMethodChanged": False,
            "classificationChanged": False,
            "failureEvidenceRemoved": False,
            "valuePreserving": True,
            "repositoryCommit": head,
            "scriptPath": str(SCRIPT.relative_to(REPO)),
            "scriptSha256": sha256_file(SCRIPT),
            "recordedAtUtc": utc_now(),
        },
    )

    full_path = LOOP_ROOT / "S19_L10_FULL_RESULTS.md"
    amended = amend_full_report(full_path.read_text(encoding="utf-8"))
    full_path.write_text(amended, encoding="utf-8")
    (LOOP_ROOT / "research_step_full_results.md").write_text(amended, encoding="utf-8")
    summary_path = LOOP_ROOT / "loop_decision_summary.md"
    summary_path.write_text(
        amend_summary(summary_path.read_text(encoding="utf-8")), encoding="utf-8"
    )

    status_path = LOOP_ROOT / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    caveat = "initial_exact_replay_column_order_failure_repaired_under_explicit_human_authorization_no_scientific_value_change"
    if caveat not in status["caveatsOrBlockers"]:
        status["caveatsOrBlockers"].append(caveat)
    status["technicalRepair"] = {
        "repairId": REPAIR_ID,
        "initialAttempt": "400_OF_400_TRAJECTORIES_13_OF_14_TABLES",
        "repairedAttempt": "400_OF_400_TRAJECTORIES_14_OF_14_TABLES",
        "scientificValueChanged": False,
        "failureEvidencePreserved": True,
    }
    status["artifactsWritten"].extend(
        [
            str(LOOP_ROOT / "technical_repair_001.json"),
            str(LOOP_ROOT / "technical_repair_completion_001.json"),
            str(LOOP_ROOT / "reporting_amendment_001.json"),
        ]
    )
    status["artifactsWritten"] = list(dict.fromkeys(status["artifactsWritten"]))
    write_json(status_path, status)
    write_json(LOOP_ROOT / "s19_l10_status.json", status)

    ledger_path = ARTIFACT_ROOT / "self_improvement_ledger.parquet"
    ledger = pd.read_parquet(ledger_path)
    phase = "POST_LOOP_TECHNICAL_REPAIR_DISCLOSURE"
    if not (
        (ledger.loopId.astype(str) == "S19-L10")
        & (ledger.recordPhase.astype(str) == phase)
    ).any():
        row = {
            "ledgerSequence": int(ledger.ledgerSequence.max()) + 1,
            "timestampUtc": utc_now(),
            "loopId": "S19-L10",
            "recordPhase": phase,
            "beliefBeforeLoop": "The first regeneration failure appeared to be a schema-order artifact because all trajectories and all fingerprint cells matched after diagnostic column alignment.",
            "motivatingEvidence": "400/400 trajectory replay, 13/14 table hashes, identical fingerprint column sets, and zero aligned cell differences.",
            "failureOrAmbiguityTargeted": "Scheduling-dependent column order in the exact table comparator.",
            "selectedHypotheses": "One explicitly human-authorized value-preserving technical repair only; no scientific branch.",
            "learned": "A fresh full rerun passed 400/400 trajectories and 14/14 tables after lexicographic schema canonicalization, with no scientific value change; the scientific result remained non-support.",
            "weakenedHypotheses": "The idea that the first mismatch reflected scientific nondeterminism; it was localized to schema order.",
            "remainingPlausibleHypotheses": "The recurring-attractor author implementation remains unresolved; neither L10 pipeline matches the complete paper fingerprint.",
            "proposedNextTest": "Mandatory human review only; no automatic next loop or S20.",
            "informationGainRationale": "The repair restored schedule-independent artifact replay without adding a scientific specification or favorable opportunity.",
            "appendOnly": True,
        }
        ledger = pd.concat(
            [ledger, pd.DataFrame([row], columns=ledger.columns)], ignore_index=True
        )
        ledger.to_parquet(ledger_path, index=False, compression="zstd")
        with (ARTIFACT_ROOT / "SELF_IMPROVEMENT_LEDGER.md").open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write(
                "\n\n## Entry 023 — S19-L10 explicitly authorized technical replay repair\n\n"
                "- **Initial failure:** 400/400 trajectories and 13/14 tables; one fingerprint table differed only in column order, with zero aligned cell differences.\n"
                "- **Repair:** preserve the failure, canonicalize schema order only, and rerun the complete scope in fresh caches.\n"
                "- **Result:** 400/400 trajectories and 14/14 tables passed; no scientific value changed.\n"
                "- **Scientific outcome:** unchanged `RECURRING_ATTRACTOR_LABEL_NOT_RECONSTRUCTED`; zero promoted leads.\n"
                "- **Next action:** mandatory human review only.\n"
            )

    registry_path = ARTIFACT_ROOT / "loop_registry.yaml"
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    entry = [item for item in registry["loops"] if item.get("loopId") == "S19-L10"]
    if len(entry) != 1:
        raise RuntimeError("L10 registry entry missing or duplicated")
    entry[0]["technicalRepair"] = {
        "repairId": REPAIR_ID,
        "humanAuthorized": True,
        "initialFailurePreserved": True,
        "initialReplay": "400_TRAJECTORIES_13_OF_14_TABLES",
        "repairedReplay": "400_TRAJECTORIES_14_OF_14_TABLES",
        "scientificValueChanged": False,
        "passed": True,
    }
    registry_path.write_text(
        yaml.safe_dump(registry, sort_keys=False), encoding="utf-8"
    )

    history_path = ARTIFACT_ROOT / "human_review_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    additions = [
        {
            "date": "2026-08-09",
            "decision": "AUTHORIZE_S19_L10_TECHNICAL_REPAIR_001",
            "scope": REPAIR_ID,
            "source": "explicit_human_direction_if_technical_fix_and_rerun",
        },
        {
            "date": "2026-08-09",
            "decision": "S19_L10_TECHNICAL_REPAIR_001_COMPLETE",
            "scope": REPAIR_ID + "::COMPLETE",
            "result": "400_OF_400_TRAJECTORIES_AND_14_OF_14_TABLES_EXACT_NO_SCIENTIFIC_VALUE_CHANGE",
            "source": "validated_value_preserving_technical_rerun",
        },
    ]
    known = {item.get("scope") for item in history["history"]}
    history["history"].extend(item for item in additions if item["scope"] not in known)
    history["pendingDecision"] = "POST_S19_L10_MANDATORY_HUMAN_REVIEW_REQUIRED"
    write_json(history_path, history)
    (ARTIFACT_ROOT / "research_step_full_results.md").write_text(
        amended, encoding="utf-8"
    )
    root_status = dict(status)
    root_status["artifactsWritten"] = list(
        dict.fromkeys(
            [
                *status["artifactsWritten"],
                str(ARTIFACT_ROOT / "research_step_full_results.md"),
            ]
        )
    )
    write_json(ARTIFACT_ROOT / "s19_status.json", root_status)

    expected_count = len(
        [
            path
            for path in LOOP_ROOT.rglob("*")
            if path.is_file() and path.name != "artifact_manifest.json"
        ]
    )
    write_json(
        LOOP_ROOT / "artifact_integrity_validation.json",
        {
            "schema": "eidosoma.e01.s19_l10.artifact_integrity_validation.v2_reporting_amendment",
            "expectedListedFileCount": expected_count,
            "allListedHashesVerified": True,
            "technicalRepairDisclosed": True,
            "passed": True,
            "validatedAtUtc": utc_now(),
        },
    )
    write_manifest(
        LOOP_ROOT / "artifact_manifest.json",
        LOOP_ROOT,
        "eidosoma.e01.s19_l10.artifact_manifest.v2_reporting_amendment",
    )
    manifest = json.loads(
        (LOOP_ROOT / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    hashes_pass = all(
        sha256_file(LOOP_ROOT / item["path"]) == item["sha256"]
        for item in manifest["files"]
    )
    if not hashes_pass or manifest["fileCount"] != expected_count:
        raise RuntimeError("L10 amended artifact manifest failed")
    write_manifest(
        ARTIFACT_ROOT / "artifact_manifest.json",
        ARTIFACT_ROOT,
        "eidosoma.e01.s19.artifact_manifest.v10_amended",
    )
    print(
        json.dumps(
            {
                "status": "REPORTING_AMENDMENT_001_COMPLETE",
                "decision": classification["decision"],
                "repairId": REPAIR_ID,
                "initialReplay": "400/400 trajectories; 13/14 tables",
                "repairedReplay": "400/400 trajectories; 14/14 tables",
                "scientificValueChanged": False,
                "artifactManifestFiles": manifest["fileCount"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
