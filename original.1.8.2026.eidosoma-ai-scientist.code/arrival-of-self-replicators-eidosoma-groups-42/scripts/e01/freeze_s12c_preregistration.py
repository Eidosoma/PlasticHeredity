#!/usr/bin/env python3
"""Validate and freeze S12C before development or any GARD input access."""

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
CONFIG = REPO / "configs/e01/s12c_source_equivalence_confirmation_preregistration.yaml"
STEP_ROOT = Path("/artifacts/research_steps/S12C")
SAFE_LATTICE = Path("/artifacts/research_steps/S12B/safe_phi_lattice.json")
PREOUTCOME_ALLOWED = {
    "preregistration.yaml",
    "preregistration_record.json",
    "immutable_input_audit.json",
    "source_snapshot_manifest.json",
    "safe_lattice_reference.json",
    "repair_delta.md",
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


def verify_repository_snapshot(specification: dict[str, Any]) -> dict[str, Any]:
    commit = specification["commit"]
    names = run_git(REPO, "ls-tree", "-r", "--name-only", commit).splitlines()
    missing = [name for name in names if not (REPO / name).is_file()]
    paths = [REPO / name for name in names if (REPO / name).is_file()]
    actual = aggregate_paths(REPO, paths) if not missing else None
    return {
        "commit": commit,
        "fileCount": len(names),
        "expectedFileCount": specification["trackedFileCount"],
        "aggregateSha256": actual,
        "expectedAggregateSha256": specification["trackedFileAggregateSha256"],
        "missing": missing,
        "passed": not missing
        and len(names) == specification["trackedFileCount"]
        and actual == specification["trackedFileAggregateSha256"],
    }


def verify_source(source_id: str, specification: dict[str, Any]) -> dict[str, Any]:
    checkout = Path(specification["localCheckout"])
    head = run_git(checkout, "rev-parse", "HEAD")
    tree = run_git(checkout, "rev-parse", "HEAD^{tree}")
    status = run_git(checkout, "status", "--porcelain=v1", "--untracked-files=all")
    files: list[dict[str, Any]] = []
    for name, expected in specification["files"].items():
        path = checkout / name
        actual_sha = sha256_file(path) if path.is_file() else None
        blob = run_git(checkout, "rev-parse", f"HEAD:{name}")
        files.append(
            {
                "path": name,
                "sha256": actual_sha,
                "expectedSha256": expected["sha256"],
                "gitBlob": blob,
                "expectedGitBlob": expected["gitBlob"],
                "passed": actual_sha == expected["sha256"] and blob == expected["gitBlob"],
            }
        )
    regularization_ok = True
    if source_id == "PHIRL_REGULARIZED_SOURCE":
        regularization_ok = subprocess.run(
            [
                "git",
                "merge-base",
                "--is-ancestor",
                specification["covarianceRegularizationIntroducedCommit"],
                "HEAD",
            ],
            cwd=checkout,
            check=False,
        ).returncode == 0
    passed = (
        head == specification["commit"]
        and tree == specification["tree"]
        and not status
        and all(item["passed"] for item in files)
        and regularization_ok
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
        "workingTreePorcelain": status,
        "files": files,
        "regularizationAncestorPassed": regularization_ok,
        "licenseStatus": "NO_LICENSE_FILE_FOUND_AT_PINNED_COMMIT",
        "passed": passed,
    }


def design_identity() -> dict[str, Any]:
    head = run_git(REPO, "rev-parse", "HEAD")
    branch = run_git(REPO, "branch", "--show-current")
    remote = run_git(REPO, "rev-parse", "@{upstream}")
    status = run_git(REPO, "status", "--short")
    return {
        "branch": branch,
        "head": head,
        "remote": remote,
        "workingTreeStatus": status,
        "passed": branch == "eidosoma/groups/42" and head == remote and not status,
    }


def validate_preregistration(
    *, require_no_outcomes: bool = True, require_pushed_design: bool = False
) -> dict[str, Any]:
    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    errors: list[str] = []
    checks: list[dict[str, Any]] = []

    def check(check_id: str, condition: bool, detail: Any) -> None:
        checks.append({"checkId": check_id, "passed": bool(condition), "detail": detail})
        if not condition:
            errors.append(f"{check_id}: {detail}")

    check(
        "schema_and_identity",
        data.get("schema")
        == "eidosoma.e01.s12c_source_equivalence_confirmation_preregistration.v1"
        and data.get("researchStepId") == "S12C"
        and data.get("preregistrationVersion")
        == "E01-S12C-SOURCE-EQUIVALENCE-CONFIRMATION-v1.0.0",
        [data.get("schema"), data.get("researchStepId"), data.get("preregistrationVersion")],
    )
    scope = data["scopeBoundary"]
    check(
        "bounded_scope",
        scope["preserveS12BByteExact"]
        and scope["exactInputTrajectoryCount"] == 12
        and scope["newGardTrajectories"] == 0
        and scope["interventionTrajectories"] == 0
        and scope["allBranchesMustPass"]
        and scope["singularFixtureRemovalForbidden"]
        and scope["additionalRepairAfterConfirmationFailureForbidden"]
        and scope["automaticS13Forbidden"],
        scope,
    )
    frozen_results: list[dict[str, Any]] = []
    for item in data["frozenInputs"]:
        path = Path(item["path"])
        actual = sha256_file(path) if path.is_file() else None
        frozen_results.append(
            {
                "inputId": item["inputId"],
                "path": str(path),
                "expectedSha256": item["sha256"],
                "actualSha256": actual,
                "passed": actual == item["sha256"],
            }
        )
    check(
        "frozen_inputs",
        len(frozen_results) >= 35 and all(item["passed"] for item in frozen_results),
        {"count": len(frozen_results), "failures": [x for x in frozen_results if not x["passed"]]},
    )
    prior_results: dict[str, Any] = {}
    for step_id, expected in data["immutablePriorEvidence"]["artifactDirectories"].items():
        actual = directory_identity(Path("/artifacts/research_steps") / step_id)
        actual["expectedFileCount"] = expected["fileCount"]
        actual["expectedAggregateSha256"] = expected["aggregateSha256"]
        actual["passed"] = (
            actual["fileCount"] == expected["fileCount"]
            and actual["aggregateSha256"] == expected["aggregateSha256"]
        )
        prior_results[step_id] = actual
    check("prior_artifact_immutability", all(x["passed"] for x in prior_results.values()), prior_results)
    repository_result = verify_repository_snapshot(
        data["immutablePriorEvidence"]["repositoryBeforeS12C"]
    )
    check("prior_repository_snapshot", repository_result["passed"], repository_result)
    sources = {
        source_id: verify_source(source_id, specification)
        for source_id, specification in data["sourceSnapshots"].items()
        if source_id in scope["exactSourceImplementations"]
    }
    check("pinned_sources", len(sources) == 2 and all(x["passed"] for x in sources.values()), sources)
    safe = json.loads(SAFE_LATTICE.read_text(encoding="utf-8")) if SAFE_LATTICE.is_file() else {}
    safe_ok = (
        sha256_file(SAFE_LATTICE) == data["sourceSnapshots"]["safeLattice"]["sha256"]
        and safe.get("nodeCount") == 16
        and safe.get("edgeCount") == 32
        and safe.get("conversionIsolation", {}).get("restrictedUnpickler") is True
    )
    check("safe_lattice_reference", safe_ok, {"path": str(SAFE_LATTICE), "sha256": sha256_file(SAFE_LATTICE) if SAFE_LATTICE.is_file() else None})
    firewall = data["fixtureFirewall"]
    check(
        "fixture_firewall",
        len(firewall["fixtureIds"]) == 7
        and firewall["exactRowsPerPhase"] == 14
        and firewall["development"]["rootSeedHex"] != firewall["confirmation"]["rootSeedHex"]
        and firewall["confirmation"]["untouched"]
        and firewall["confirmation"]["generationAndAccessRequiresCleanPushedImplementationLock"],
        firewall,
    )
    gates = data["confirmationGates"]
    check(
        "unchanged_global_confirmation_gate",
        gates["allRowsMustPass"]
        and gates["expectedRows"] == 14
        and gates["statusIdentical"]
        and gates["miMaxAbsDifferenceAtMost"] == 1e-10
        and gates["partitionAverageMaxAbsDifferenceAtMost"] == 1e-10
        and gates["localPhiRMaxAbsDifferenceAtMost"] == 1e-9
        and gates["failureAction"] == "CLOSE_E01_NO_GARD_ACCESS_NO_FURTHER_REPAIR",
        gates,
    )
    inherited = Path(data["inheritedS12BAuditContract"]["path"])
    inherited_ok = sha256_file(inherited) == data["inheritedS12BAuditContract"]["sha256"]
    check("inherited_s12b_contract", inherited_ok, {"path": str(inherited), "sha256": sha256_file(inherited)})
    runtime = data["runtimeAndStorage"]
    check(
        "compute_and_threads",
        runtime["sourceAnalysisWorkers"] == 6
        and runtime["hardCpuHours"] == 50.0
        and runtime["hardGpuHours"] == 2.0
        and runtime["hardWallHours"] == 24.0
        and runtime["hardNewArtifactBytes"] == 10737418240
        and all(value == "1" for value in runtime["threadEnvironment"].values()),
        runtime,
    )
    existing = (
        {path.relative_to(STEP_ROOT).as_posix() for path in STEP_ROOT.rglob("*") if path.is_file()}
        if STEP_ROOT.exists()
        else set()
    )
    if require_no_outcomes:
        check("no_development_confirmation_or_gard_outputs", existing.issubset(PREOUTCOME_ALLOWED), sorted(existing))
    design = design_identity()
    if require_pushed_design:
        check("clean_pushed_design", design["passed"], design)
    return {
        "schema": "eidosoma.e01.s12c_preregistration_validation.v1",
        "researchStepId": "S12C",
        "preregistrationVersion": data["preregistrationVersion"],
        "configPath": str(CONFIG),
        "configSha256": sha256_file(CONFIG),
        "checks": checks,
        "frozenInputs": frozen_results,
        "priorArtifacts": prior_results,
        "priorRepository": repository_result,
        "sources": sources,
        "design": design,
        "errors": errors,
        "success": not errors,
    }


def repair_delta_markdown() -> str:
    return """# S12C preregistered repair delta

## Top summary

- **Research step ID:** S12C (`E01-S12C-SOURCE-EQUIVALENCE-CONFIRMATION-v1.0.0`)
- **Completion status:** PREOUTCOME_DESIGN_FROZEN; development, confirmation, and GARD outcomes not yet opened.
- **Artifacts written:** `preregistration.yaml`, `preregistration_record.json`, `immutable_input_audit.json`, `source_snapshot_manifest.json`, `safe_lattice_reference.json`, and this repair delta.
- **Validation result:** PASS if and only if the accompanying preregistration record has `success: true` and the design commit is clean and pushed.
- **Outcome classification:** Pending; this document contains no confirmation or GARD scientific result.
- **Caveats or blockers:** The repair is informed by the known S12B singular failure. Confirmation therefore uses a separate untouched root and remains inaccessible until implementation lock.
- **Recommended next action:** Run only the frozen development suite, audit the wrapper-only delta, lock and push the implementation, and then run untouched confirmation. Do not open GARD input before unanimous confirmation.

## Exactly one permitted repair

S12B used a vectorized cross-correlation block for IIGR lagged MI. On the preserved exact-duplicate fixture it differed from the pinned nested `scipy.stats.pearsonr` loop by about `3.64e-17`; the resulting degenerate Fiedler split changed, and the wrapper reached a singular reduced covariance while the pinned source did not. S12C replaces only that vectorized IIGR MI calculation with the pinned source's pairwise loop, assignment order, and significance comparison. It does not regularize IIGR, change exception policy, alter the Fiedler algorithm, change PhiRL, or weaken any gate.

## Immutable boundaries

S12B remains failed and byte-exact. Both public commits, their file hashes, safe-lattice JSON, S12 trajectories/labels/preprocessing, modes, statistics, tolerances, and classifications remain frozen. A confirmation exception cannot be relabeled as equivalent to source eligibility. Any confirmation failure permanently closes this repair path without GARD access or another repair.
"""


def freeze() -> dict[str, Any]:
    STEP_ROOT.mkdir(parents=True, exist_ok=True)
    validation = validate_preregistration(
        require_no_outcomes=True, require_pushed_design=True
    )
    if not validation["success"]:
        raise RuntimeError("S12C preregistration validation failed: " + "; ".join(validation["errors"]))
    shutil.copyfile(CONFIG, STEP_ROOT / "preregistration.yaml")
    write_json(STEP_ROOT / "preregistration_record.json", validation)
    write_json(
        STEP_ROOT / "immutable_input_audit.json",
        {
            "schema": "eidosoma.e01.s12c_immutable_input_audit.v1",
            "researchStepId": "S12C",
            "preOutcome": {
                "priorArtifacts": validation["priorArtifacts"],
                "priorRepository": validation["priorRepository"],
                "frozenInputs": validation["frozenInputs"],
                "success": True,
            },
            "approvedMutableFiles": ["/workspace/RESEARCH_PLAN.md_after_artifact_finalization_only"],
            "success": True,
        },
    )
    write_json(
        STEP_ROOT / "source_snapshot_manifest.json",
        {
            "schema": "eidosoma.e01.s12c_source_snapshot_manifest.v1",
            "researchStepId": "S12C",
            "preregistrationVersion": "E01-S12C-SOURCE-EQUIVALENCE-CONFIRMATION-v1.0.0",
            "sourceRelationship": "SOURCE_INFORMED_RECONSTRUCTION",
            "sources": validation["sources"],
            "safeLattice": {
                "path": str(SAFE_LATTICE),
                "sha256": sha256_file(SAFE_LATTICE),
                "rawPickleUsedByS12CScientificRunner": False,
            },
            "design": validation["design"],
            "runtime": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": numpy.__version__,
                "scipy": scipy.__version__,
                "networkx": networkx.__version__,
                "pandas": pandas.__version__,
                "pyarrow": pyarrow.__version__,
                "precision": "CPU_float64_authoritative",
                "threadEnvironment": {
                    name: os.environ.get(name)
                    for name in (
                        "OMP_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "NUMEXPR_NUM_THREADS",
                        "VECLIB_MAXIMUM_THREADS",
                    )
                },
            },
            "success": True,
        },
    )
    write_json(
        STEP_ROOT / "safe_lattice_reference.json",
        {
            "schema": "eidosoma.e01.s12c_safe_lattice_reference.v1",
            "researchStepId": "S12C",
            "path": str(SAFE_LATTICE),
            "sha256": sha256_file(SAFE_LATTICE),
            "rawPickleLoadedByS12CScientificCode": False,
            "reuseRelationship": "BYTE_EXACT_IMMUTABLE_S12B_SAFE_JSON",
        },
    )
    (STEP_ROOT / "repair_delta.md").write_text(repair_delta_markdown(), encoding="utf-8")
    return validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--allow-existing-preoutcome", action="store_true")
    parser.add_argument("--require-pushed-design", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        result = validate_preregistration(
            require_no_outcomes=not args.allow_existing_preoutcome,
            require_pushed_design=args.require_pushed_design,
        )
    else:
        result = freeze()
    print(
        json.dumps(
            {
                "success": result["success"],
                "errors": result["errors"],
                "configSha256": result["configSha256"],
                "design": result["design"],
            },
            sort_keys=True,
        )
    )
    if not result["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
