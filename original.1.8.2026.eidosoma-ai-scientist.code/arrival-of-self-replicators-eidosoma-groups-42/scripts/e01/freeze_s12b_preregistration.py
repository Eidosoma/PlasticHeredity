#!/usr/bin/env python3
"""Validate and freeze E01-S12B before any trajectory-level outcome audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import networkx
import numpy
import pandas
import pyarrow
import scipy
import yaml

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs/e01/s12b_pigozzi_source_audit_preregistration.yaml"
STEP_ROOT = Path("/artifacts/research_steps/S12B")
CACHE_ROOT = Path("/cache/e01_s12b")
SAFE_LATTICE = STEP_ROOT / "safe_phi_lattice.json"
PREOUTCOME_ALLOWED = {
    "preregistration.yaml",
    "preregistration_record.json",
    "immutable_input_audit.json",
    "source_snapshot_manifest.json",
    "source_audit.md",
    "safe_phi_lattice.json",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_git(checkout: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args], cwd=checkout, check=check, capture_output=True, text=True
    )
    return result.stdout.strip()


def aggregate_paths(root: Path, paths: list[Path]) -> str:
    body = "".join(
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}\n"
        for path in sorted(paths)
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def directory_identity(root: Path) -> dict[str, Any]:
    paths = [path for path in root.rglob("*") if path.is_file()]
    return {
        "fileCount": len(paths),
        "byteCount": sum(path.stat().st_size for path in paths),
        "aggregateSha256": aggregate_paths(root, paths),
    }


def verify_prior_repository(data: dict[str, Any]) -> dict[str, Any]:
    expected = data["immutablePriorEvidence"]["repositoryBeforeS12B"]
    commit = expected["commit"]
    names = run_git(REPO, "ls-tree", "-r", "--name-only", commit).splitlines()
    missing = [name for name in names if not (REPO / name).is_file()]
    paths = [REPO / name for name in names if (REPO / name).is_file()]
    actual = aggregate_paths(REPO, paths) if not missing else None
    return {
        "commit": commit,
        "fileCount": len(names),
        "expectedFileCount": expected["trackedFileCount"],
        "aggregateSha256": actual,
        "expectedAggregateSha256": expected["trackedFileAggregateSha256"],
        "missing": missing,
        "passed": not missing
        and len(names) == expected["trackedFileCount"]
        and actual == expected["trackedFileAggregateSha256"],
    }


def verify_source(source_id: str, specification: dict[str, Any]) -> dict[str, Any]:
    checkout = Path(specification["localCheckout"])
    head = run_git(checkout, "rev-parse", "HEAD")
    tree = run_git(checkout, "rev-parse", "HEAD^{tree}")
    commit_type = run_git(checkout, "cat-file", "-t", specification["commit"])
    status = run_git(checkout, "status", "--porcelain=v1", "--untracked-files=all")
    files: list[dict[str, Any]] = []
    for name, expected in specification["files"].items():
        path = checkout / name
        blob = run_git(checkout, "rev-parse", f"HEAD:{name}")
        actual_sha = sha256_file(path) if path.is_file() else None
        files.append(
            {
                "path": name,
                "bytes": path.stat().st_size if path.is_file() else None,
                "sha256": actual_sha,
                "expectedSha256": expected["sha256"],
                "gitBlob": blob,
                "expectedGitBlob": expected["gitBlob"],
                "passed": actual_sha == expected["sha256"] and blob == expected["gitBlob"],
            }
        )
    license_names = [
        name
        for name in run_git(checkout, "ls-tree", "-r", "--name-only", "HEAD").splitlines()
        if Path(name).name.lower().startswith(("license", "copying"))
    ]
    regularization: dict[str, Any] | None = None
    if source_id == "PHIRL_REGULARIZED_SOURCE":
        reg_commit = specification["covarianceRegularizationIntroducedCommit"]
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", reg_commit, "HEAD"],
            cwd=checkout,
            check=False,
        ).returncode == 0
        regularization = {
            "commit": reg_commit,
            "commitType": run_git(checkout, "cat-file", "-t", reg_commit),
            "isAncestorOfPinnedCommit": ancestor,
            "subject": run_git(checkout, "show", "-s", "--format=%s", reg_commit),
        }
    archive_root = CACHE_ROOT / "source_archives"
    archive_root.mkdir(parents=True, exist_ok=True)
    archive = archive_root / f"{source_id}-{specification['commit']}.tar"
    subprocess.run(
        [
            "git",
            "archive",
            "--format=tar",
            f"--output={archive}",
            specification["commit"],
            *specification["files"].keys(),
        ],
        cwd=checkout,
        check=True,
    )
    passed = (
        head == specification["commit"]
        and tree == specification["tree"]
        and commit_type == "commit"
        and not status
        and all(item["passed"] for item in files)
        and not license_names
        and (regularization is None or regularization["isAncestorOfPinnedCommit"])
    )
    return {
        "sourceId": source_id,
        "repository": specification["repository"],
        "repositoryUrl": specification["repositoryUrl"],
        "checkout": str(checkout),
        "commit": head,
        "expectedCommit": specification["commit"],
        "tree": tree,
        "expectedTree": specification["tree"],
        "commitType": commit_type,
        "workingTreePorcelain": status,
        "files": files,
        "licenseFilesAtPinnedCommit": license_names,
        "licenseStatus": "NO_LICENSE_FILE_FOUND_AT_PINNED_COMMIT",
        "regularizationHistory": regularization,
        "cacheArchive": {
            "path": str(archive),
            "bytes": archive.stat().st_size,
            "sha256": sha256_file(archive),
            "redistributionStatus": "INTERNAL_CACHE_ONLY_NOT_AN_ARTIFACT",
        },
        "passed": passed,
    }


def validate_preregistration(*, require_no_outcomes: bool = True) -> dict[str, Any]:
    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    errors: list[str] = []
    checks: list[dict[str, Any]] = []

    def check(check_id: str, condition: bool, detail: Any) -> None:
        checks.append({"checkId": check_id, "passed": bool(condition), "detail": detail})
        if not condition:
            errors.append(f"{check_id}: {detail}")

    check("schema", data.get("schema") == "eidosoma.e01.s12b_pigozzi_source_audit_preregistration.v1", data.get("schema"))
    check("identity", data.get("researchStepId") == "S12B" and data.get("preregistrationVersion") == "E01-S12B-PIGOZZI-SOURCE-CODE-AUDIT-v1.0.0", [data.get("researchStepId"), data.get("preregistrationVersion")])
    check("preoutcome_status", data.get("status") == "FROZEN_BEFORE_S12B_GARD_SCIENTIFIC_OUTCOMES", data.get("status"))
    scope = data["scopeBoundary"]
    check("bounded_scope", scope["exactInputTrajectoryCount"] == 12 and scope["newGardTrajectories"] == 0 and scope["interventionTrajectories"] == 0 and len(scope["exactSourceImplementations"]) == 2 and scope["automaticS13Forbidden"], scope)
    frozen_results: list[dict[str, Any]] = []
    for item in data["frozenInputs"]:
        path = Path(item["path"])
        actual = sha256_file(path) if path.is_file() else None
        frozen_results.append({"inputId": item["inputId"], "path": str(path), "expectedSha256": item["sha256"], "actualSha256": actual, "passed": actual == item["sha256"]})
    check("frozen_inputs", len(frozen_results) == 30 and all(item["passed"] for item in frozen_results), {"count": len(frozen_results), "failures": [item for item in frozen_results if not item["passed"]]})
    artifact_results: dict[str, Any] = {}
    for step_id, expected in data["immutablePriorEvidence"]["artifactDirectories"].items():
        actual = directory_identity(Path("/artifacts/research_steps") / step_id)
        actual["expectedFileCount"] = expected["fileCount"]
        actual["expectedAggregateSha256"] = expected["aggregateSha256"]
        actual["passed"] = actual["fileCount"] == expected["fileCount"] and actual["aggregateSha256"] == expected["aggregateSha256"]
        artifact_results[step_id] = actual
    check("prior_artifact_immutability", all(item["passed"] for item in artifact_results.values()), artifact_results)
    repository_result = verify_prior_repository(data)
    check("prior_repository_immutability", repository_result["passed"], repository_result)
    source_results = {source_id: verify_source(source_id, specification) for source_id, specification in data["sourceSnapshots"].items() if source_id in scope["exactSourceImplementations"]}
    check("source_snapshots", len(source_results) == 2 and all(item["passed"] for item in source_results.values()), source_results)
    safe = json.loads(SAFE_LATTICE.read_text(encoding="utf-8")) if SAFE_LATTICE.is_file() else {}
    safe_passed = safe.get("schema") == "eidosoma.e01.s12b.safe_phi_lattice.v1" and safe.get("nodeCount") == 16 and safe.get("edgeCount") == 32 and safe.get("rawPickleSha256") == "66cd662640079e9a2a8bc172250b124d59945fd805b7f91d5588e2f7d1d7ea03" and safe.get("conversionIsolation", {}).get("restrictedUnpickler") is True
    check("safe_lattice", safe_passed, {"path": str(SAFE_LATTICE), "sha256": sha256_file(SAFE_LATTICE) if SAFE_LATTICE.is_file() else None, "schema": safe.get("schema"), "nodeCount": safe.get("nodeCount"), "edgeCount": safe.get("edgeCount")})
    equivalence = data["sourceEquivalence"]
    check("equivalence_gates_frozen", equivalence["mustPassBeforeGardProcessing"] and equivalence["gates"]["miMaxAbsDifferenceAtMost"] == 1e-10 and equivalence["gates"]["partitionAverageMaxAbsDifferenceAtMost"] == 1e-10 and equivalence["gates"]["localPhiRMaxAbsDifferenceAtMost"] == 1e-9 and equivalence["failureAction"] == "STOP_BEFORE_GARD_OUTCOMES_NO_REPAIR", equivalence)
    check("analysis_primary_frozen", data["analysis"]["primaryProspectiveEstimand"] == "current_generation_rho_0" and data["analysis"]["bootstrap"]["replicates"] == 4096 and data["analysis"]["circularShiftNull"]["replicates"] == 4096 and data["analysis"]["retrospectiveCoherenceRule"]["allMustPass"]["positiveTrajectoryCorrelationsAtLeast"] == 9, data["analysis"])
    check("classification_precedence", data["classification"]["vocabulary"] == ["SOURCE_FAMILY_NOT_SUPPORTED", "RETROSPECTIVE_SOURCE_FAMILY_RESEMBLANCE", "REGULARIZATION_DEPENDENT_RESEMBLANCE", "SOURCE_FAMILY_PROSPECTIVE_CANDIDATE"] and len(data["classification"]["precedence"]) == 5, data["classification"])
    runtime = data["runtimeAndStorage"]
    check("compute_ceiling", runtime["sourceAnalysisWorkers"] == 6 and runtime["statisticsWorkers"] == 1 and runtime["orchestrationCores"] == 1 and runtime["hardCpuHours"] == 50.0 and runtime["hardWallHours"] == 24.0 and runtime["hardNewArtifactBytes"] == 10737418240 and all(int(value) == 1 for value in runtime["threadEnvironment"].values()), runtime)
    check("required_outputs", len(data["requiredArtifacts"]["files"]) == 22 and len(data["requiredArtifacts"]["figures"]) == 6 and "research_step_full_results.md" in data["requiredArtifacts"]["files"] and "S12B_FULL_RESULTS.md" in data["requiredArtifacts"]["files"], data["requiredArtifacts"])
    existing = {path.relative_to(STEP_ROOT).as_posix() for path in STEP_ROOT.rglob("*") if path.is_file()} if STEP_ROOT.exists() else set()
    if require_no_outcomes:
        check("no_scientific_outcomes_before_freeze", existing.issubset(PREOUTCOME_ALLOWED), sorted(existing))
    return {
        "schema": "eidosoma.e01.s12b_preregistration_validation.v1",
        "researchStepId": "S12B",
        "preregistrationVersion": data["preregistrationVersion"],
        "configPath": str(CONFIG),
        "configSha256": sha256_file(CONFIG),
        "checks": checks,
        "frozenInputs": frozen_results,
        "priorArtifacts": artifact_results,
        "priorRepository": repository_result,
        "sources": source_results,
        "errors": errors,
        "success": not errors,
    }


def source_manifest(validation: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "eidosoma.e01.s12b_source_snapshot_manifest.v1",
        "researchStepId": "S12B",
        "preregistrationVersion": "E01-S12B-PIGOZZI-SOURCE-CODE-AUDIT-v1.0.0",
        "sourceRelationship": "SOURCE_INFORMED_RECONSTRUCTION",
        "forbiddenIdentities": ["AUTHOR_PRIMARY", "PAPER_PRIMARY", "EXACT_GARD_IMPLEMENTATION"],
        "sources": validation["sources"],
        "safeLattice": {"path": str(SAFE_LATTICE), "sha256": sha256_file(SAFE_LATTICE), "rawPickleSha256": "66cd662640079e9a2a8bc172250b124d59945fd805b7f91d5588e2f7d1d7ea03", "rawPickleUsedByScientificRunner": False},
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "networkx": networkx.__version__,
            "pandas": pandas.__version__,
            "pyarrow": pyarrow.__version__,
            "precision": "CPU_float64_authoritative",
            "threadEnvironment": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
        },
        "licenseNote": "No LICENSE or COPYING file was present at either pinned commit. Public visibility is not a redistribution grant; raw source and pickle archives remain cache-only and are not collected artifacts.",
        "success": validation["success"],
    }


def source_audit_markdown(manifest: dict[str, Any]) -> str:
    iigr = manifest["sources"]["IIGR_CORRECTED_SOURCE"]
    phirl = manifest["sources"]["PHIRL_REGULARIZED_SOURCE"]
    return f"""# S12B pinned-source audit

## Top summary

- **Research step ID:** S12B (`E01-S12B-PIGOZZI-SOURCE-CODE-AUDIT-v1.0.0`)
- **Completion status:** Pre-outcome source audit complete; scientific execution not yet started.
- **Artifacts written:** `preregistration.yaml`, `preregistration_record.json`, `immutable_input_audit.json`, `source_snapshot_manifest.json`, `source_audit.md`, and `safe_phi_lattice.json`.
- **Validation result:** PASS — both commits, trees, three required files per repository, raw-pickle identity, safe conversion, regularization ancestry, and prior S10–S12 immutability matched the frozen design.
- **Outcome classification:** Pending; no S12B GARD outcome was inspected for this audit.
- **Caveats or blockers:** This is `SOURCE_INFORMED_RECONSTRUCTION`, no license file was found, the original GARD author implementation remains unavailable, and the raw pickle is barred from scientific execution.
- **Recommended next action:** Commit and push the complete pre-outcome design, then run source-equivalence validation; stop before GARD processing if any equivalence gate fails.

## Pinned identities and source behavior

- IIGR: commit `{iigr['commit']}`, tree `{iigr['tree']}`. `main.py:26–30` defines z-score → global-signal regression → lag-one residualization. `main.py:108–122` defines alpha=1/no-Bonferroni lagged MI, the unnormalized Fiedler split, partition averaging, corrected `local_phi_r`, and diagnostic `emergence`. `information.py:27–32`, `43–53`, `56–118`, `121–148`, and `151–201` provide the traced implementations.
- PhiRL: commit `{phirl['commit']}`, tree `{phirl['tree']}`. `main.py:28–53` removes dimensions at or below `1e-8`, z-scores, applies fast lagged MI, partitions, averages, and decomposes. `information.py:47–59` applies `epsilon=1e-6` trace-scaled covariance regularization; `information.py:189–244` supplies the fast MI and unnormalized Fiedler behavior. Regularization commit `{phirl['regularizationHistory']['commit']}` is an ancestor of the pinned commit.
- Both lattice pickles are byte-identical SHA-256 `{manifest['safeLattice']['rawPickleSha256']}`; the safe JSON artifact is SHA-256 `{manifest['safeLattice']['sha256']}`. The converter inspected every opcode and admitted only `dict`, `DiGraph`, and `NodeView` globals in a `python -I` disposable process.

## Relationship and license boundary

The public code concerns gene-regulatory and reinforcement-learning applications, not a released GARD simulator. It informs the local-Phi reconstruction but cannot establish the paper's unpublished GARD code, data layout, random-state ordering, or an author-primary method. Neither pinned tree contains a detected LICENSE or COPYING file, so no public-source payload is redistributed in S12B artifacts.
"""


def freeze() -> dict[str, Any]:
    STEP_ROOT.mkdir(parents=True, exist_ok=True)
    validation = validate_preregistration(require_no_outcomes=True)
    if not validation["success"]:
        raise RuntimeError("S12B preregistration validation failed: " + "; ".join(validation["errors"]))
    shutil.copyfile(CONFIG, STEP_ROOT / "preregistration.yaml")
    write_json(STEP_ROOT / "preregistration_record.json", validation)
    write_json(
        STEP_ROOT / "immutable_input_audit.json",
        {
            "schema": "eidosoma.e01.s12b_immutable_input_audit.v1",
            "researchStepId": "S12B",
            "preOutcome": True,
            "priorArtifacts": validation["priorArtifacts"],
            "priorRepository": validation["priorRepository"],
            "frozenInputs": validation["frozenInputs"],
            "success": True,
        },
    )
    manifest = source_manifest(validation)
    write_json(STEP_ROOT / "source_snapshot_manifest.json", manifest)
    (STEP_ROOT / "source_audit.md").write_text(source_audit_markdown(manifest), encoding="utf-8")
    return validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--allow-existing-preoutcome", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        result = validate_preregistration(require_no_outcomes=not args.allow_existing_preoutcome)
    else:
        result = freeze()
    print(json.dumps({"success": result["success"], "errors": result["errors"], "configSha256": result["configSha256"]}, sort_keys=True))
    if not result["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
