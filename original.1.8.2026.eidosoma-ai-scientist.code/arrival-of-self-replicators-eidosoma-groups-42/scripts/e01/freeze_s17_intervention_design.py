#!/usr/bin/env python3
"""Freeze and audit the outcome-blind S17 intervention contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import yaml

from e01_intervention_reconstruction.core import (
    BENCHMARK_ROOT_HEX,
    CANDIDATES,
    HISTORICAL_REPLAY_ENVELOPE,
    ROOT_HEX,
    SAFE_LATTICE,
    VERSION,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs/e01/s17_intervention_reconstruction_preregistration.yaml"
MANIFEST_PATH = REPO_ROOT / "configs/e01/s17_execution_manifest.json"
STEP_ROOT = Path("/artifacts/research_steps/S17")
PAPER_PATH = Path("/cache/e01_s03/downloads/paper-2607.28250v1.pdf")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True
    ).strip()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def build_manifest() -> dict[str, object]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["versionedStepId"] != VERSION:
        raise ValueError("S17 config/version mismatch")
    if config["scientificScope"]["trajectoryCount"] != 72:
        raise ValueError("S17 must freeze exactly 72 scientific trajectories")
    if config["scientificScope"]["sharedMatrixCount"] != 12:
        raise ValueError("S17 must freeze exactly 12 shared matrices")
    if config["computeGate"]["s17AvailableScientificCpuHours"] != 100.52383159377861:
        raise ValueError("S16 carry-forward compute value changed")
    if config["seedAndPairing"]["rootHex"] != ROOT_HEX:
        raise ValueError("scientific root mismatch")
    if config["seedAndPairing"]["benchmarkRootHex"] != BENCHMARK_ROOT_HEX:
        raise ValueError("benchmark root mismatch")
    if (
        config["diagnostics"]["historicalReplayEnvelope"]
        != HISTORICAL_REPLAY_ENVELOPE
    ):
        raise ValueError("historical replay envelope mismatch")
    expected = {
        "S12F-CANDIDATE-02": (0.6031526490073492, "FIRST_DAUGHTER"),
        "S12F-CANDIDATE-03": (0.5613315384859516, "RANDOM_NONEMPTY"),
    }
    for candidate_id, (h, daughter) in expected.items():
        definition = CANDIDATES[candidate_id]
        if definition.exposure.h != h or definition.daughter_rule != daughter:
            raise ValueError(f"candidate definition mismatch: {candidate_id}")
    inputs = {
        "AGENTS.md": Path("/workspace/AGENTS.md"),
        "FULL_PLAN.md": Path("/workspace/FULL_PLAN.md"),
        "RESEARCH_PLAN.md": Path("/workspace/RESEARCH_PLAN.md"),
        "input-attachments/MANIFEST.json": Path(
            "/workspace/input-attachments/MANIFEST.json"
        ),
        "attachmentSidecar": Path(
            "/workspace/input-attachments/ed5486bf-a043-485b-a233-d88d8d123759/_metadata/ATTACHMENT.md"
        ),
        "originalPaper": PAPER_PATH,
        "safeLattice": SAFE_LATTICE,
        "S12FRCandidateLock": Path(
            "/artifacts/research_steps/S12FR/candidate_timebase_pipeline_lock.json"
        ),
        "S13YFixedBranch": Path(
            "/artifacts/research_steps/S13Y/fixed_branch_lock.json"
        ),
        "S13YReport": Path(
            "/artifacts/research_steps/S13Y/research_step_full_results.md"
        ),
        "S14Report": Path(
            "/artifacts/research_steps/S14/research_step_full_results.md"
        ),
        "S15Report": Path(
            "/artifacts/research_steps/S15/research_step_full_results.md"
        ),
        "S16Report": Path(
            "/artifacts/research_steps/S16/research_step_full_results.md"
        ),
        "S16ComputeLedger": Path(
            "/artifacts/research_steps/S16/compute_ledger.json"
        ),
    }
    missing = [name for name, path in inputs.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing frozen S17 input(s): {missing}")
    hashes = {
        name: {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
        for name, path in inputs.items()
    }
    if hashes["originalPaper"]["sha256"] != (
        "77a2ec2c0751839d8a2e10863ca803c6f8b61475bbc790f2bbdad2a38af04ae4"
    ):
        raise ValueError("original paper hash mismatch")
    return {
        "schema": "eidosoma.e01.s17_execution_manifest.v1",
        "researchStepId": "S17",
        "versionedStepId": VERSION,
        "predictionOrInterventionOutcomeAccessed": False,
        "contract": config,
        "frozenInputHashes": hashes,
        "repositoryFiles": {},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-commit", action="store_true")
    args = parser.parse_args()
    payload = build_manifest()
    write_json(MANIFEST_PATH, payload)
    repository_paths = [
        CONFIG_PATH,
        REPO_ROOT / "src/e01_intervention_reconstruction/__init__.py",
        REPO_ROOT / "src/e01_intervention_reconstruction/core.py",
        Path(__file__),
        REPO_ROOT / "scripts/e01/run_s17_intervention_reconstruction.py",
        REPO_ROOT / "tests/e01/test_s17_intervention_reconstruction.py",
    ]
    missing = [str(path) for path in repository_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"S17 repository contract incomplete: {missing}")
    payload["repositoryFiles"] = {
        str(path.relative_to(REPO_ROOT)): sha256(path) for path in repository_paths
    }
    write_json(MANIFEST_PATH, payload)
    if args.record_commit:
        branch = git("branch", "--show-current")
        head = git("rev-parse", "HEAD")
        remote = git("rev-parse", "origin/eidosoma/groups/42")
        status = git("status", "--short")
        if branch != "eidosoma/groups/42" or head != remote or status:
            raise RuntimeError(
                f"S17 pushed lock gate failed: branch={branch!r}, "
                f"head={head}, remote={remote}, status={status!r}"
            )
        lock = {
            "schema": "eidosoma.e01.s17_preoutcome_design_lock.v1",
            "researchStepId": "S17",
            "versionedStepId": VERSION,
            "passed": True,
            "branch": branch,
            "commit": head,
            "remoteCommit": remote,
            "workingTreeStatus": status,
            "executionManifestPath": str(MANIFEST_PATH),
            "executionManifestSha256": sha256(MANIFEST_PATH),
            "benchmarkExecuted": False,
            "scientificMatrixCreated": False,
            "predictionOrInterventionOutcomeAccessed": False,
        }
        write_json(STEP_ROOT / "preoutcome_design_lock.json", lock)
    print(
        json.dumps(
            {
                "manifest": str(MANIFEST_PATH.relative_to(REPO_ROOT)),
                "manifestSha256": sha256(MANIFEST_PATH),
                "recordedCommit": args.record_commit,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
