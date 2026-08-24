#!/usr/bin/env python3
"""Apply value-preserving S19-L08 reporting amendment 001.

The locked runner's machine tables and terminal decision correctly recorded a
4/6 joint occupancy-gate result.  Its generated prose contained three stale
sentences saying that both mechanisms/all six objects passed.  This script
changes only those narrative sentences and status caveats, records the
original hashes, and rebuilds artifact manifests.  It never recalculates or
changes a scientific value, gate, or classification.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
ROOT = Path("/artifacts/research_steps/S19")
LOOP = ROOT / "loops/L08"
REPORTS = [
    LOOP / "S19_L08_FULL_RESULTS.md",
    LOOP / "research_step_full_results.md",
    ROOT / "research_step_full_results.md",
]
SUMMARY = LOOP / "loop_decision_summary.md"
ORIGINAL_REPORT_SHA256 = "1c789d59f23399d07da89ca52b69fe435c1d243f875fb8de950e231a65bd78b4"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, text=True, capture_output=True
    ).stdout.strip()


def replace_once(text: str, before: str, after: str) -> str:
    count = text.count(before)
    if count == 0 and after in text:
        return text
    if count != 1:
        raise RuntimeError(f"expected exactly one reporting sentence, found {count}")
    return text.replace(before, after, 1)


def build_manifest(path: Path, root: Path, schema: str) -> None:
    rows = []
    for item in sorted(p for p in root.rglob("*") if p.is_file() and p != path):
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
            "files": rows,
            "generatedAtUtc": now(),
            "reportingAmendmentApplied": "S19-L08-REPORTING-AMENDMENT-001",
        },
    )


def main() -> None:
    occupancy = pd.read_csv(LOOP / "occupancy_gate_results.csv")
    decision_gates = pd.read_csv(LOOP / "decision_gate_results.csv")
    classification = json.loads((LOOP / "classification.json").read_text())
    projected = occupancy.loc[
        occupancy["analysisObjectId"].eq("A_PROJECTED_MOLECULAR")
    ].sort_values("candidateId")
    if (
        len(occupancy) != 6
        or int(occupancy["gatePassed"].sum()) != 4
        or len(projected) != 2
        or projected["gatePassed"].any()
        or classification["decision"]
        != "NEITHER_MECHANISM_REPRODUCES_ON_UNTOUCHED_DATA"
    ):
        raise RuntimeError("authoritative L08 machine evidence does not match amendment premise")
    joint = decision_gates.loc[
        decision_gates["gateId"].eq("JOINT_ALL_SIX_OCCUPANCY_GATES")
    ]
    if len(joint) != 1 or bool(joint.iloc[0]["passed"]) or joint.iloc[0]["detail"] != "4/6":
        raise RuntimeError("joint occupancy gate is not the locked 4/6 failure")

    original_hashes = {str(path): sha256_file(path) for path in REPORTS}
    if not all(
        value == ORIGINAL_REPORT_SHA256
        or "S19-L08 reporting amendment 001" in Path(path).read_text()
        for path, value in original_hashes.items()
    ):
        raise RuntimeError("a report differs from the locked original before amendment")

    replacements = [
        (
            "- **Lay summary:** Both frozen mechanisms again reproduced the paper's approximate occupancy band on 100 wholly new matched matrices. Mechanism A gave projected molecular occupancy 0.8470/0.8457 (boundary-only 0.8581/0.8608); mechanism B gave 0.8750/0.8698. The complete locked fingerprint leads to `NEITHER_MECHANISM_REPRODUCES_ON_UNTOUCHED_DATA`. This identifies which frozen explanation is better supported within L08, not which implementation the authors used.",
            "- **Lay summary:** On 100 wholly new matched matrices, A's boundary-unit occupancy and B's molecular occupancy reproduced the approximate band, but A's locked molecular projection did not: 0.8470/0.8457 fell just below 0.85 (boundary-only 0.8581/0.8608), while B was 0.8750/0.8698. The preregistered joint gate therefore passed 4/6 object-candidate cells and returned `NEITHER_MECHANISM_REPRODUCES_ON_UNTOUCHED_DATA`. This is a scope-specific locked decision, not an author-code identification.",
        ),
        (
            "L07 found two ways to turn the approximately 98% label occupancy into approximately 88%: measure inheritance once per fission and project it over the next growth interval, or keep the molecular label but make Poisson updates much larger. L08 tested those exact two ideas on 100 new catalytic matrices without searching or changing anything. Both kept their occupancy match, so occupancy alone remains nonidentifying. The decisive comparison used trajectory length, persistence, onset, consistency, episodes, fission fidelity, mass, overshoot, cross-candidate agreement, and exact replay. Its locked result is `NEITHER_MECHANISM_REPRODUCES_ON_UNTOUCHED_DATA`.",
            "L07 found two ways to turn the approximately 98% label occupancy into approximately 88%: measure inheritance once per fission and project it over the next growth interval, or keep the molecular label but make Poisson updates much larger. L08 tested those exact two ideas on 100 new catalytic matrices without searching or changing anything. A retained its boundary-unit match and B retained its molecular match, but A's separate molecular projection fell marginally outside the band. Occupancy therefore remains measurement-object dependent rather than a reproduced mechanism-wide result. The complete comparison used trajectory length, persistence, onset, consistency, episodes, fission fidelity, mass, overshoot, cross-candidate agreement, and exact replay. Its locked result is `NEITHER_MECHANISM_REPRODUCES_ON_UNTOUCHED_DATA`.",
        ),
        (
            "All six primary occupancy objects passed the frozen inclusive `[0.85, 0.91]` band with 100 defined matrices each. Boundary-unit and molecular-projection values remain separate; neither was substituted based on closeness.",
            "Four of six primary occupancy object-candidate cells passed the frozen inclusive `[0.85, 0.91]` band, with 100 defined matrices in every cell. Both A boundary cells and both B molecular cells passed; A's projected molecular means, 0.846981 and 0.845676, fell below the 0.85 floor. Boundary-unit and molecular-projection values remain separate; neither was substituted based on closeness.\n\n### S19-L08 reporting amendment 001\n\nThe initial generated prose incorrectly said 6/6 occupancy gates passed even though `occupancy_gate_results.csv`, `decision_gate_results.csv`, and the terminal classification always recorded the correct 4/6 failure. This value-preserving amendment corrects only that narrative inconsistency. No trajectory, label, statistic, gate, decision, or classification changed.",
        ),
        (
            "The directed decision is **`NEITHER_MECHANISM_REPRODUCES_ON_UNTOUCHED_DATA`**. Under the existing S19 vocabulary, the result is `EXPLORATORY_NON_SUPPORT`, `AUTHOR_AMBIGUITY_UNRESOLVED`, `NOT_PROMOTABLE`. It is not labelled confirmed, is not promoted to S20, and does not identify author code. The result cannot alter S18 prediction or causal-control classifications.",
            "The directed decision is **`NEITHER_MECHANISM_REPRODUCES_ON_UNTOUCHED_DATA`**. The decision vocabulary was prospectively locked so that any failure of the joint all-six occupancy gate maps to this token; it does not mean that every individual readout missed the band, because 4/6 passed. Under the existing S19 vocabulary, the result is `EXPLORATORY_NON_SUPPORT`, `AUTHOR_AMBIGUITY_UNRESOLVED`, `NOT_PROMOTABLE`. It is not labelled confirmed, is not promoted to S20, and does not identify author code. The result cannot alter S18 prediction or causal-control classifications.",
        ),
    ]
    for report in REPORTS:
        text = report.read_text(encoding="utf-8")
        for before, after in replacements:
            text = replace_once(text, before, after)
        report.write_text(text, encoding="utf-8")

    summary = SUMMARY.read_text(encoding="utf-8")
    summary = replace_once(
        summary,
        "Untouched projected occupancy was `0.846981` / `0.845676` for A and `0.874951` / `0.869845` for B. All values are descriptive simulation evidence. See `S19_L08_FULL_RESULTS.md` for the complete fingerprint and gate-by-gate basis.",
        "A's boundary-unit occupancy passed at `0.858100` / `0.860800`, while its separately locked projected occupancy failed marginally at `0.846981` / `0.845676`; B molecular occupancy passed at `0.874951` / `0.869845`. Thus 4/6 preregistered cells passed. The locked resolution order maps this joint-gate failure to the decision above; it does not imply that every readout missed the band. All values are descriptive simulation evidence. See `S19_L08_FULL_RESULTS.md` for the complete fingerprint and gate-by-gate basis.",
    )
    SUMMARY.write_text(summary, encoding="utf-8")

    for status_path in (LOOP / "s19_l08_status.json", ROOT / "s19_status.json"):
        status = json.loads(status_path.read_text())
        caveats = status["caveatsOrBlockers"]
        precise = "joint_occupancy_gate_failed_4_of_6_A_projection_below_0_85"
        if precise not in caveats:
            caveats.insert(0, precise)
        status["reportingAmendment"] = "S19-L08-REPORTING-AMENDMENT-001"
        write_json(status_path, status)

    reporting_commit = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/eidosoma/groups/42")
    if reporting_commit != remote or git("status", "--porcelain=v1"):
        raise RuntimeError("reporting amendment must run from a clean pushed commit")
    corrected_hashes = {str(path): sha256_file(path) for path in REPORTS}
    amendment = {
        "schema": "eidosoma.e01.s19_l08_reporting_amendment.v1",
        "amendmentId": "S19-L08-REPORTING-AMENDMENT-001",
        "trigger": "Generated prose said all six occupancy cells passed; authoritative machine evidence and the unchanged terminal decision recorded 4/6.",
        "scope": "Narrative reports, decision-summary clarification, and status caveat only.",
        "authoritativeMachineEvidence": {
            "occupancyGateResultsSha256": sha256_file(LOOP / "occupancy_gate_results.csv"),
            "decisionGateResultsSha256": sha256_file(LOOP / "decision_gate_results.csv"),
            "classificationSha256": sha256_file(LOOP / "classification.json"),
            "passedCells": 4,
            "totalCells": 6,
            "aProjectedCandidate2": float(projected.iloc[0]["mean"]),
            "aProjectedCandidate3": float(projected.iloc[1]["mean"]),
        },
        "originalReportSha256": ORIGINAL_REPORT_SHA256,
        "originalHashesAtApplication": original_hashes,
        "correctedReportHashes": corrected_hashes,
        "scientificValuesChanged": False,
        "gatesChanged": False,
        "decisionChanged": False,
        "classificationChanged": False,
        "lockedScientificCommit": "975a2cbd1cff4cc09e312bb884a5e3fd86d3a249",
        "reportingAmendmentCommit": reporting_commit,
        "appliedAtUtc": now(),
    }
    write_json(LOOP / "reporting_amendment_001.json", amendment)

    storage_path = LOOP / "storage_validation.json"
    storage = json.loads(storage_path.read_text())
    storage["retainedArtifactBytesAfterReportingAmendment"] = sum(
        path.stat().st_size for path in LOOP.rglob("*") if path.is_file()
    )
    storage["reportingAmendment"] = amendment["amendmentId"]
    storage["validatedAtUtc"] = now()
    write_json(storage_path, storage)

    expected_count = len(
        [
            path
            for path in LOOP.rglob("*")
            if path.is_file()
            and path.name not in {"artifact_manifest.json", "artifact_integrity_validation.json"}
        ]
    ) + 1
    integrity_path = LOOP / "artifact_integrity_validation.json"
    integrity = json.loads(integrity_path.read_text())
    integrity["expectedListedFileCount"] = expected_count
    integrity["reportingAmendment"] = amendment["amendmentId"]
    integrity["validatedAtUtc"] = now()
    write_json(integrity_path, integrity)
    build_manifest(
        LOOP / "artifact_manifest.json",
        LOOP,
        "eidosoma.e01.s19_l08_artifact_manifest.v1",
    )
    manifest = json.loads((LOOP / "artifact_manifest.json").read_text())
    if manifest["fileCount"] != expected_count or not all(
        sha256_file(LOOP / row["path"]) == row["sha256"] for row in manifest["files"]
    ):
        raise RuntimeError("post-amendment loop artifact integrity failed")
    build_manifest(
        ROOT / "artifact_manifest.json",
        ROOT,
        "eidosoma.e01.s19_root_artifact_manifest.v1",
    )


if __name__ == "__main__":
    main()
