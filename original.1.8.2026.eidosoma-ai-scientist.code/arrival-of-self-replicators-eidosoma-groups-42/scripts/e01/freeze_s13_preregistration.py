#!/usr/bin/env python3
"""Freeze the outcome-blind E01 S13 held-out scale-up design."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
WORKSPACE = REPO.parent
ARTIFACTS = Path("/artifacts")
STEP_ROOT = ARTIFACTS / "research_steps/S13"
CONFIG = REPO / "configs/e01/s13_confirmed_timebase_baseline_scaleup_preregistration.yaml"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def prior_artifact_files() -> list[Path]:
    paths: list[Path] = []
    root = ARTIFACTS / "research_steps"
    for step in sorted(root.iterdir()):
        if not step.is_dir() or step.name == "S13":
            continue
        paths.extend(path for path in sorted(step.rglob("*")) if path.is_file())
    return paths


def freeze_prior() -> dict[str, Any]:
    rows = [
        {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in prior_artifact_files()
    ]
    payload = {
        "schema": "eidosoma.e01.s13_immutable_prior_baseline.v1",
        "researchStepId": "S13",
        "scope": "all_files_under_research_steps_except_S13_at_freeze_time",
        "fileCount": len(rows),
        "files": rows,
        "passed": bool(rows),
    }
    write_json(STEP_ROOT / "immutable_prior_baseline.json", payload)
    return payload


def compute_ledger() -> dict[str, Any]:
    # These are deliberately conservative envelopes. Exact measured CPU is used
    # when preserved; otherwise artifact-active wall spans are multiplied by the
    # maximum plausible worker count and rounded upward. S12FR sums all component
    # worker CPU fields rather than its known underinclusive top-level field.
    rows = [
        ("S01", 0.250, "conservative_unmetered_envelope"),
        ("S02", 0.250, "conservative_unmetered_envelope"),
        ("S03", 3.000, "conservative_unmetered_environment_build_envelope"),
        ("S04", 0.250, "conservative_unmetered_envelope"),
        ("S05", 0.250, "conservative_unmetered_envelope"),
        ("S06", 0.250, "conservative_unmetered_envelope"),
        ("S07", 2.200, "artifact_span_times_eight_rounded_up"),
        ("S08", 1.250, "artifact_span_times_eight_rounded_up"),
        ("S09", 2.000, "artifact_span_times_eight_rounded_up"),
        ("S10", 0.392, "runtime_wall_seconds_times_eight"),
        ("S11", 1.331, "canonical_wall_seconds_times_eight"),
        ("S11R", 0.373, "summed_stage_wall_seconds_times_eight"),
        ("S12", 0.484, "reported_total_measured_task_cpu_hours_rounded_up"),
        ("S12B", 0.051, "wall_seconds_times_six"),
        ("S12C", 6.010, "reported_worker_cpu_hours_rounded_up"),
        ("S12D", 4.000, "artifact_active_span_times_eight_upper_envelope"),
        ("S12E", 0.032, "reported_wall_hours_times_six_rounded_up"),
        ("S12F", 0.376, "orchestrator_development_wall_times_six_rounded_up"),
        ("S12FR", 5.873, "sum_of_all_preserved_component_worker_cpu_fields"),
        ("S12G", 26.034, "reported_completed_worker_cpu_hours_rounded_up"),
        ("S12H", 0.001, "reported_cpu_seconds_rounded_up"),
        ("S12I", 24.204, "reported_worker_cpu_hours_rounded_up"),
        ("S12J", 0.127, "reported_process_cpu_hours_rounded_up"),
    ]
    entries = [
        {"stepId": step, "cpuEnvelopeHours": hours, "basis": basis}
        for step, hours, basis in rows
    ]
    prior = sum(row[1] for row in rows)
    prior_gpu = 2.0
    gpu_ceiling = 80.0
    proposed = 64.0
    payload = {
        "schema": "eidosoma.e01.s13_cumulative_compute_ledger.v1",
        "researchStepId": "S13",
        "accountingPolicy": {
            "noProjectionDoubleCounting": True,
            "exactMeasuredPreferred": True,
            "missingCpuUsesConservativeEnvelope": True,
            "s12frTopLevelUnderinclusiveFieldIgnored": True,
        },
        "entries": entries,
        "priorCpuEnvelopeHours": prior,
        "priorCpuEnvelopeRoundedForDecision": 80.0,
        "priorGpuEnvelopeHours": prior_gpu,
        "proposedS13CpuEnvelopeHoursIncludingReserve": proposed,
        "proposedS13GpuHours": 0.0,
        "projectedCumulativeCpuHours": 80.0 + proposed,
        "projectedCumulativeGpuHours": prior_gpu,
        "cpuCeilingHours": 250.0,
        "gpuCeilingHours": gpu_ceiling,
        "passed": 80.0 + proposed <= 250.0 and prior_gpu <= gpu_ceiling,
    }
    write_json(STEP_ROOT / "compute_ledger.json", payload)
    gate = {
        "schema": "eidosoma.e01.s13_pre_simulation_compute_gate.v1",
        "researchStepId": "S13",
        "simulationOutcomeOpened": False,
        "priorCpuEnvelopeHours": 80.0,
        "projectedNewCpuHours": proposed,
        "projectedCumulativeCpuHours": 144.0,
        "cpuCeilingHours": 250.0,
        "projectedCumulativeGpuHours": 2.0,
        "gpuCeilingHours": 80.0,
        "headroomCpuHours": 106.0,
        "passed": payload["passed"],
    }
    write_json(STEP_ROOT / "compute_gate.json", gate)
    return payload


def freeze_seed_root(config: dict[str, Any]) -> dict[str, Any]:
    root = str(config["randomness"]["rootHex"])
    matches: list[str] = []
    seed_files = [
        path
        for path in prior_artifact_files()
        if "seed" in path.name.lower() and path.stat().st_size <= 500_000_000
    ]
    for path in seed_files:
        try:
            if path.suffix == ".parquet":
                frame = pd.read_parquet(path)
                if any(
                    root == str(value)
                    for column in frame.columns
                    for value in frame[column].dropna().tolist()
                ):
                    matches.append(str(path))
            elif root.encode("ascii") in path.read_bytes():
                matches.append(str(path))
        except (OSError, TypeError, ValueError):
            # The root must still be checked against every readable seed file;
            # unreadable files are retained as explicit audit caveats.
            matches.append(f"UNREADABLE:{path}")
    payload = {
        "schema": "eidosoma.e01.s13_seed_root_lock.v1",
        "researchStepId": "S13",
        "rootId": config["randomness"]["rootId"],
        "rootHex": root,
        "rootSha256": hashlib.sha256(bytes.fromhex(root)).hexdigest(),
        "priorSeedFilesChecked": len(seed_files),
        "priorRootMatches": matches,
        "bitGenerator": "PCG64DXSM",
        "outcomesOpened": False,
        "passed": len(root) == 64 and not matches,
    }
    write_json(STEP_ROOT / "seed_root_lock.json", payload)
    return payload


def source_snapshot(config: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    repositories = [
        (
            "IIGR_CORRECTED_SOURCE",
            Path("/cache/e01_s12b/sources/IntegratedInformationGeneRegulation"),
            config["sourceIdentities"]["iigrCommit"],
        ),
        (
            "PHIRL_REGULARIZED_SOURCE",
            Path("/cache/e01_s12b/sources/PhiRL"),
            config["sourceIdentities"]["phirlCommit"],
        ),
    ]
    for source_id, path, expected in repositories:
        actual = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=path, text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--short"], cwd=path, text=True
        ).strip()
        checks.append(
            {
                "sourceId": source_id,
                "path": str(path),
                "expectedCommit": expected,
                "actualCommit": actual,
                "workingTreeStatus": dirty,
                "passed": actual == expected and not dirty,
            }
        )
    safe = Path("/artifacts/research_steps/S12B/safe_phi_lattice.json")
    checks.append(
        {
            "sourceId": "SAFE_JSON_LATTICE",
            "path": str(safe),
            "expectedSha256": config["sourceIdentities"]["safeLatticeSha256"],
            "actualSha256": sha256_file(safe),
            "passed": sha256_file(safe)
            == config["sourceIdentities"]["safeLatticeSha256"],
        }
    )
    s12c = pd.read_csv("/artifacts/research_steps/S12C/confirmation_fixture_results.csv")
    s12d = pd.read_csv("/artifacts/research_steps/S12D/source_metric_equivalence.csv")
    payload = {
        "schema": "eidosoma.e01.s13_source_snapshot_manifest.v1",
        "researchStepId": "S13",
        "checks": checks,
        "s12cEquivalenceRows": len(s12c),
        "s12cAllPassed": bool(s12c["allGatesPassed"].astype(bool).all()),
        "s12dEmergenceIdentityRows": len(s12d),
        "s12dAllPassed": bool(s12d["allGatesPassed"].astype(bool).all()),
        "unauditedPickleLoaded": False,
    }
    payload["passed"] = bool(
        all(row["passed"] for row in checks)
        and len(s12c) == 14
        and payload["s12cAllPassed"]
        and len(s12d) == 40
        and payload["s12dAllPassed"]
    )
    write_json(STEP_ROOT / "source_snapshot_manifest.json", payload)
    return payload


def method_files() -> list[Path]:
    return [
        CONFIG,
        REPO / "src/e01_confirmed_timebase_scaleup/__init__.py",
        REPO / "src/e01_confirmed_timebase_scaleup/core.py",
        REPO / "scripts/e01/freeze_s13_preregistration.py",
        REPO / "scripts/e01/run_s13_confirmed_timebase_scaleup.py",
        REPO / "tests/e01/test_s13_confirmed_timebase_scaleup.py",
        REPO / "src/e01_latent_timebase/core.py",
        REPO / "src/e01_replay_repair/comparator.py",
        REPO / "src/e01_frozen_timebase_ensemble/core.py",
        REPO / "src/e01_source_emergence_metric_identity/core.py",
        REPO / "src/e01_source_emergence_metric_identity/analysis.py",
        REPO / "scripts/e01/run_s12g_frozen_timebase_ensemble.py",
        REPO / "configs/e01/s12g_output_schemas.json",
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-commit", action="store_true")
    args = parser.parse_args()
    STEP_ROOT.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    (STEP_ROOT / "preregistration.yaml").write_text(
        CONFIG.read_text(encoding="utf-8"), encoding="utf-8"
    )
    baseline = freeze_prior()
    ledger = compute_ledger()
    root = freeze_seed_root(config)
    sources = source_snapshot(config)
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/eidosoma/groups/42")
    if args.record_commit and (head != remote or git("status", "--short")):
        raise RuntimeError("S13 design must be committed, pushed, and clean")
    files = method_files()
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise RuntimeError(f"S13 method files missing: {missing}")
    lock = {
        "schema": "eidosoma.e01.s13_method_lock.v1",
        "researchStepId": "S13",
        "versionedStepId": config["versionedStepId"],
        "designCommit": head if args.record_commit else None,
        "remoteCommit": remote if args.record_commit else None,
        "branch": git("branch", "--show-current"),
        "outcomesOpened": False,
        "files": [
            {
                "path": str(path.relative_to(REPO)),
                "sha256": sha256_file(path),
            }
            for path in files
        ],
        "priorImmutabilityBaselinePassed": baseline["passed"],
        "computeGatePassed": ledger["passed"],
        "seedRootGatePassed": root["passed"],
        "sourceGatePassed": sources["passed"],
        "passed": bool(
            args.record_commit
            and baseline["passed"]
            and ledger["passed"]
            and root["passed"]
            and sources["passed"]
        ),
    }
    write_json(STEP_ROOT / "method_lock.json", lock)
    record = {
        "schema": "eidosoma.e01.s13_preregistration_record.v1",
        "researchStepId": "S13",
        "versionedStepId": config["versionedStepId"],
        "preregistrationSha256": sha256_file(STEP_ROOT / "preregistration.yaml"),
        "designCommit": lock["designCommit"],
        "remoteCommit": lock["remoteCommit"],
        "completeDesignCommittedAndPushed": lock["passed"],
        "labelOrInformationOutcomeOpened": False,
        "simulationOutcomeOpened": False,
        "passed": lock["passed"],
    }
    write_json(STEP_ROOT / "preregistration_record.json", record)
    access = {
        "schema": "eidosoma.e01.s13_scope_access_ledger.v1",
        "researchStepId": "S13",
        "events": [
            {
                "stage": "PREREGISTRATION_FREEZE",
                "simulationOutcomeOpened": False,
                "labelOutcomeOpened": False,
                "informationTheoryOutcomeOpened": False,
                "status": "PASS" if lock["passed"] else "PENDING_COMMIT",
            }
        ],
        "forbiddenWorkAccessCount": 0,
        "success": lock["passed"],
    }
    write_json(STEP_ROOT / "scope_access_ledger.json", access)
    print(json.dumps({"stage": "S13_preregistration", "passed": lock["passed"], "head": head}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
