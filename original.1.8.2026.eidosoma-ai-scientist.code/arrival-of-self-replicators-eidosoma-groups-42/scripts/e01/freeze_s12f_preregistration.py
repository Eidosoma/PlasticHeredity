#!/usr/bin/env python3
"""Validate and freeze S12F targets/method without opening simulation outcomes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
WORKSPACE = Path("/workspace")
ARTIFACTS_ROOT = Path("/artifacts")
ARTIFACTS = ARTIFACTS_ROOT / "research_steps/S12F"
CACHE = Path("/cache/e01_s12f")
S12E_CACHE = Path("/cache/e01_s12e")
CONFIG = REPO / "configs/e01/s12f_latent_timebase_preregistration.yaml"
TARGET_DIR = REPO / "configs/e01/s12f"
TARGET_FILES = (
    "paper_timebase_targets.yaml",
    "figure_digitization.csv",
    "figure_digitization_uncertainty.json",
    "table_timebase_fingerprints.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def file_records(root: Path, exclude: Path | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if exclude is not None and (path == exclude or exclude in path.parents):
            continue
        records.append(
            {
                "path": str(path),
                "relativePath": str(path.relative_to(root)),
                "sizeBytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return records


def validate_digitization() -> dict[str, Any]:
    raster = Path(
        "/workspace/input-attachments/ed5486bf-a043-485b-a233-d88d8d123759/figures/figure-02.png"
    )
    if sha256(raster) != "0e4aac507ccf6e10ced31edd6d7e5ba8c876d9d0c8d420b145dfc27c7d040778":
        raise RuntimeError("Figure 2 raster identity changed")
    ticks = {
        "A": ([66, 126, 186, 246, 306, 366, 426], [0, 200, 400, 600, 800, 1000, 1200], 397, 1100.0, 1314.0),
        "B": ([529, 576, 623, 669, 716, 763, 809, 856, 902], list(range(0, 801, 100)), 902, 800.0, 800.0),
        "C": ([66, 113, 160, 207, 254, 301, 347, 394, 441], list(range(0, 801, 100)), 441, 800.0, 800.0),
        "D": ([529, 604, 679, 754, 829, 904], list(range(0, 1001, 200)), 904, 1000.0, 1000.0),
    }
    results: list[dict[str, Any]] = []
    for panel, (x, values, terminal_x, manual, range_value) in ticks.items():
        design = np.column_stack((np.asarray(x, float), np.ones(len(x))))
        slope, intercept = np.linalg.lstsq(design, np.asarray(values, float), rcond=None)[0]
        residual = float(np.max(np.abs(slope * np.asarray(x) + intercept - values)))
        estimate = float(slope * terminal_x + intercept)
        two_pixels = float(2.0 * slope)
        one_percent = float(0.01 * range_value)
        tolerance = max(two_pixels, one_percent)
        difference = abs(estimate - manual)
        passed = bool(difference <= tolerance)
        results.append(
            {
                "panel": panel,
                "slopeUnitsPerPixel": float(slope),
                "intercept": float(intercept),
                "maximumTickFitResidual": residual,
                "terminalPixel": terminal_x,
                "pixelEstimate": estimate,
                "independentEstimate": manual,
                "methodDifference": difference,
                "agreementTolerance": tolerance,
                "passed": passed,
            }
        )
    with (TARGET_DIR / "figure_digitization.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        digitized = list(csv.DictReader(handle))
    if len(digitized) != 10 or not all(row["status"] in {"AGREED", "DETECTED", "INTERVAL_RETAINED"} for row in digitized):
        raise RuntimeError("digitization ledger schema/status failed")
    uncertainty = json.loads(
        (TARGET_DIR / "figure_digitization_uncertainty.json").read_text(encoding="utf-8")
    )
    table = json.loads(
        (TARGET_DIR / "table_timebase_fingerprints.json").read_text(encoding="utf-8")
    )
    if not np.isclose(table["descriptiveRatio"], 716.0 / 0.88, rtol=0, atol=1e-12):
        raise RuntimeError("Table 1 ratio is not exact")
    passed = bool(all(row["passed"] for row in results))
    if not passed:
        raise RuntimeError("independent digitization methods failed their agreement gate")
    return {
        "schemaVersion": "E01-S12F-phase0-validation-v1.0.0",
        "researchStepId": "E01-S12F-LATENT-TIMEBASE-INFERENCE-v1.0.0",
        "validatedAtUtc": datetime.now(UTC).isoformat(),
        "passed": passed,
        "twoIndependentMethods": True,
        "methodAgreement": results,
        "aggregateIntervalRetained": uncertainty["aggregate"]["axisUpperReconciledInterval"],
        "tableRatioValidated": True,
        "simulationOutcomeOpened": False,
        "labelsOrInformationTheoryAccessed": False,
    }


def validate_config(config: dict[str, Any]) -> None:
    if config["researchStepId"] != "E01-S12F-LATENT-TIMEBASE-INFERENCE-v1.0.0":
        raise RuntimeError("wrong research step ID")
    if config["s13Status"] != "BLOCKED_PENDING_S12F_HUMAN_REVIEW":
        raise RuntimeError("S13 is not blocked")
    if config["phase2"]["abcSmc"]["rounds"] != [
        {"round": 1, "particlesEvaluated": 256, "retainForProposal": 128, "logKernelSd": None},
        {"round": 2, "particlesEvaluated": 128, "retainForProposal": 64, "logKernelSd": 0.20},
        {"round": 3, "particlesEvaluated": 64, "retainForProposal": 64, "logKernelSd": 0.10},
    ]:
        raise RuntimeError("ABC-SMC schedule changed")
    if len(config["benchmark"]["configurations"]) != 16:
        raise RuntimeError("benchmark must contain exactly sixteen configurations")
    if config["phase3"]["maximumCandidates"] != 3 or config["phase3"]["matricesPerCandidate"] != 32:
        raise RuntimeError("confirmation ceiling changed")
    prohibited = config["interpretationBoundary"]
    if any(
        prohibited[key]
        for key in (
            "labelsPermitted", "causalEmergencePermitted", "localPhiRPermitted",
            "interventionsPermitted", "s12gPermitted", "s13Permitted",
        )
    ):
        raise RuntimeError("a forbidden downstream action was enabled")
    roots = list(config["randomness"]["roots"].values())
    if len(roots) != len(set(roots)) or not all(len(root) == 64 for root in roots):
        raise RuntimeError("S12F roots are not distinct 256-bit identities")


def source_snapshot(config: dict[str, Any]) -> dict[str, Any]:
    source_records: list[dict[str, Any]] = []
    source_specs = {
        "historicalGard": (
            Path(config["sourcePins"]["historicalGard"]["path"]),
            config["sourcePins"]["historicalGard"]["commit"],
            ["tgs_parameters_v10.m", "tgs_newbeta_v10.m", "tgs_grow_v10.m", "tgs_split_v10.m"],
        ),
        "iigr": (
            Path(config["sourcePins"]["iigr"]["path"]),
            config["sourcePins"]["iigr"]["commit"],
            ["main.py", "information.py", "phi_lattice_22.pickle"],
        ),
        "phirl": (
            Path(config["sourcePins"]["phirl"]["path"]),
            config["sourcePins"]["phirl"]["commit"],
            ["main.py", "information.py", "phi_lattice_22.pickle"],
        ),
    }
    for source_id, (repo, expected, files) in source_specs.items():
        actual = git(repo, "rev-parse", "HEAD^{commit}")
        if actual != expected:
            raise RuntimeError(f"{source_id} commit changed: {actual}")
        for relative in files:
            path = repo / relative
            source_records.append(
                {
                    "sourceId": source_id,
                    "repositoryPath": str(repo),
                    "commit": actual,
                    "tree": git(repo, "rev-parse", "HEAD^{tree}"),
                    "relativePath": relative,
                    "gitBlob": git(repo, "rev-parse", f"HEAD:{relative}"),
                    "sha256": sha256(path),
                    "sizeBytes": path.stat().st_size,
                }
            )
    context_paths = [
        REPO / "src/e01_pigozzi_source_equivalence_confirmation/core.py",
        REPO / "configs/e01/s12c_implementation_lock.yaml",
        Path(config["sourcePins"]["safeLattice"]["path"]),
        Path(config["sourcePins"]["s12e"]["report"]),
        Path(config["sourcePins"]["s12e"]["artifactManifest"]),
        Path(config["sourcePins"]["paper"]["pdfPath"]),
        Path(config["sourcePins"]["paper"]["figure2Path"]),
    ]
    context = []
    for path in context_paths:
        context.append({"path": str(path), "sha256": sha256(path), "sizeBytes": path.stat().st_size})
    if context[-2]["sha256"] != config["sourcePins"]["paper"]["sha256"]:
        raise RuntimeError("paper PDF identity changed")
    if context[-1]["sha256"] != config["sourcePins"]["paper"]["figure2Sha256"]:
        raise RuntimeError("Figure 2 identity changed")
    if sha256(Path(config["sourcePins"]["safeLattice"]["path"])) != config["sourcePins"]["safeLattice"]["sha256"]:
        raise RuntimeError("safe lattice identity changed")
    if sha256(Path(config["sourcePins"]["s12e"]["artifactManifest"])) != config["sourcePins"]["s12e"]["artifactManifestSha256"]:
        raise RuntimeError("S12E artifact manifest identity changed")
    return {
        "schemaVersion": "E01-S12F-source-input-snapshot-v1.0.0",
        "researchStepId": config["researchStepId"],
        "capturedAtUtc": datetime.now(UTC).isoformat(),
        "sourceFiles": source_records,
        "contextFiles": context,
        "paperSourceArchiveStatus": "PDF_ONLY_SOURCE_ENDPOINT_RESPONSE",
        "safeLatticeLoadedDuringS12F": False,
        "informationTheorySourceExecutedDuringS12F": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-commit", action="store_true")
    arguments = parser.parse_args()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    validate_config(config)
    phase0 = validate_digitization()

    # Prior evidence and S12E caches are baselined before any S12F simulation.
    prior_records = file_records(ARTIFACTS_ROOT, exclude=ARTIFACTS)
    prior_payload = {
        "schemaVersion": "E01-S12F-immutable-prior-baseline-v1.0.0",
        "researchStepId": config["researchStepId"],
        "capturedAtUtc": datetime.now(UTC).isoformat(),
        "excludedPath": str(ARTIFACTS),
        "fileCount": len(prior_records),
        "files": prior_records,
        "aggregateSha256": canonical_sha(prior_records),
    }
    cache_records = file_records(S12E_CACHE)
    cache_payload = {
        "schemaVersion": "E01-S12F-s12e-cache-manifest-v1.0.0",
        "researchStepId": config["researchStepId"],
        "capturedAtUtc": datetime.now(UTC).isoformat(),
        "readOnlyRoot": str(S12E_CACHE),
        "fileCount": len(cache_records),
        "files": cache_records,
        "aggregateSha256": canonical_sha(cache_records),
    }

    shutil.copyfile(CONFIG, ARTIFACTS / "preregistration.yaml")
    for filename in TARGET_FILES:
        shutil.copyfile(TARGET_DIR / filename, ARTIFACTS / filename)
    write_json(ARTIFACTS / "phase0_validation.json", phase0)
    write_json(ARTIFACTS / "immutable_prior_baseline.json", prior_payload)
    write_json(ARTIFACTS / "s12e_cache_manifest.json", cache_payload)
    write_json(ARTIFACTS / "source_input_snapshot_manifest.json", source_snapshot(config))

    head = git(REPO, "rev-parse", "HEAD^{commit}")
    branch = git(REPO, "branch", "--show-current")
    remote_head = ""
    try:
        remote_head = git(REPO, "rev-parse", "origin/eidosoma/groups/42^{commit}")
    except subprocess.CalledProcessError:
        pass
    record = {
        "schemaVersion": "E01-S12F-preregistration-record-v1.0.0",
        "researchStepId": config["researchStepId"],
        "frozenAtUtc": datetime.now(UTC).isoformat(),
        "configPath": str(CONFIG),
        "configSha256": sha256(CONFIG),
        "targetFiles": [
            {
                "path": str(TARGET_DIR / filename),
                "sha256": sha256(TARGET_DIR / filename),
            }
            for filename in TARGET_FILES
        ],
        "gitBranch": branch,
        "gitCommit": head,
        "remoteBranchCommit": remote_head,
        "commitRecordedAfterPush": bool(arguments.record_commit),
        "headMatchesRemote": bool(arguments.record_commit and head == remote_head),
        "simulationOutcomeOpened": False,
        "phase0Passed": phase0["passed"],
    }
    if arguments.record_commit and (branch != "eidosoma/groups/42" or head != remote_head):
        raise RuntimeError("cannot record S12F method lock before pushed branch equality")
    write_json(ARTIFACTS / "preregistration_record.json", record)

    manifest_files = [
        "preregistration.yaml", *TARGET_FILES, "phase0_validation.json",
        "immutable_prior_baseline.json", "s12e_cache_manifest.json",
        "source_input_snapshot_manifest.json", "preregistration_record.json",
    ]
    manifest = {
        "schemaVersion": "E01-S12F-preregistration-artifact-manifest-v1.0.0",
        "researchStepId": config["researchStepId"],
        "createdAtUtc": datetime.now(UTC).isoformat(),
        "files": [
            {
                "path": filename,
                "sha256": sha256(ARTIFACTS / filename),
                "sizeBytes": (ARTIFACTS / filename).stat().st_size,
            }
            for filename in manifest_files
        ],
        "simulationOutcomeOpened": False,
    }
    manifest["aggregateSha256"] = canonical_sha(manifest["files"])
    write_json(ARTIFACTS / "preregistration_artifact_manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": "PASS",
                "recordCommit": arguments.record_commit,
                "priorFiles": len(prior_records),
                "s12eCacheFiles": len(cache_records),
                "phase0": phase0["passed"],
                "head": head,
                "remote": remote_head,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
