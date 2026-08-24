#!/usr/bin/env python3
"""Prepare, execute, regenerate, and finalize E01/S19-L07.

Each scientific round is defined in a repository-backed lock before this
runner may open its result.  L07 is intentionally adaptive, but its sole
scientific objective is occupancy closeness to 0.88.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_name] = "1"

import pickle
import platform
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow
import scipy
import yaml

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from e01_clean_directional_confirmation.core import ROOT_SEED_HEX, SIMULATION_PHASE
from e01_frozen_timebase_ensemble.core import selected_clock_observations
from e01_latent_timebase.core import (
    ExposureDefinition,
    SimulationDefinition,
    array_sha256,
    generate_beta,
    initialize_distinct_state,
    simulate_trajectory,
    trajectory_summary,
)
from e01_latent_timebase.core import derive_seed as derive_simulation_seed
from e01_s19_occupancy_search.core import (
    ExploratoryExposureDefinition,
    LOOP_ID,
    PAPER_OCCUPANCY_TARGET,
    PAPER_OCCUPANCY_TOLERANCE,
    VERSION,
    aggregate_occupancy,
    fingerprint,
    materialize_frozen_setting,
    summarize_pairs,
)

ARTIFACT_ROOT = Path("/artifacts/research_steps/S19")
LOOP_ROOT = ARTIFACT_ROOT / "loops/L07"
CACHE_ROOT = Path("/cache/e01_s19_l07")
S13Y_ROOT = Path("/artifacts/research_steps/S13Y")
PROTOCOL_PATH = REPO / "configs/e01/s19_l07_governing_protocol.yaml"
ROUNDS_PATH = REPO / "configs/e01/s19_l07_rounds.yaml"
AMENDMENT_PATH = REPO / "configs/e01/s19_l07_amendment_001.json"
REFINEMENT_PATH = REPO / "configs/e01/s19_l07_round_r04.yaml"
BRACKETING_PATH = REPO / "configs/e01/s19_l07_round_r05.yaml"
VALIDATION_PATH = REPO / "configs/e01/s19_l07_round_r06.yaml"
ADAPTIVE_ROUND_PATHS = {
    "R04_ADAPTIVE_EXPOSURE_REFINEMENT": REFINEMENT_PATH,
    "R05_EXPOSURE_LOCAL_BRACKETING": BRACKETING_PATH,
    "R06_FRESH_SEED_VALIDATION": VALIDATION_PATH,
}
TRAJECTORY_MANIFEST_PATH = S13Y_ROOT / "trajectory_manifest.parquet"
FROZEN_LABEL_PATH = S13Y_ROOT / "label_values.parquet"
IMMUTABLE_BASELINE_PATH = LOOP_ROOT / "immutable_prior_baseline.json"
SCIENTIFIC_ROUNDS = {
    "R01_BOUNDARY_CLOCK",
    "R02_THRESHOLD_TRANSCRIPTION",
    "R03_EXPOSURE_SIMULATOR",
    "R04_ADAPTIVE_EXPOSURE_REFINEMENT",
    "R05_EXPOSURE_LOCAL_BRACKETING",
    "R06_FRESH_SEED_VALIDATION",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression="zstd")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, text=True, capture_output=True
    ).stdout.strip()


def assert_repository_lock() -> dict[str, Any]:
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/eidosoma/groups/42")
    clean = not bool(git("status", "--porcelain=v1"))
    if head != remote or not clean:
        raise RuntimeError("L07 requires pushed HEAD and a clean repository worktree")
    return {"head": head, "remoteHead": remote, "cleanWorktree": clean, "passed": True}


def load_round_registry() -> dict[str, Any]:
    return yaml.safe_load(ROUNDS_PATH.read_text(encoding="utf-8"))


def round_config(round_id: str) -> dict[str, Any]:
    for item in load_round_registry()["rounds"]:
        if item["roundId"] == round_id:
            return item
    adaptive = ADAPTIVE_ROUND_PATHS.get(round_id)
    if adaptive is not None and adaptive.exists():
        return yaml.safe_load(adaptive.read_text(encoding="utf-8"))
    raise KeyError(round_id)


def _hash_prior_files() -> list[dict[str, Any]]:
    roots: list[Path] = []
    step_root = Path("/artifacts/research_steps")
    for item in sorted(step_root.iterdir()):
        if item.name != "S19":
            roots.append(item)
    for loop in ("L01", "L02", "L03", "L04", "L05", "L06", "L06R"):
        roots.append(ARTIFACT_ROOT / "loops" / loop)
    for bundle in (
        Path("/artifacts/E01_forensic_replication_bundle"),
        Path("/artifacts/E01_forensic_replication_artifact_v2"),
    ):
        if bundle.exists():
            roots.append(bundle)
    rows = []
    for root in roots:
        if root.is_file():
            files = [root]
        elif root.is_dir():
            files = sorted(item for item in root.rglob("*") if item.is_file())
        else:
            raise FileNotFoundError(root)
        for path in files:
            rows.append(
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return rows


def validate_prior_baseline() -> dict[str, Any]:
    baseline = json.loads(IMMUTABLE_BASELINE_PATH.read_text(encoding="utf-8"))
    current = _hash_prior_files()
    expected = {(item["path"], item["bytes"], item["sha256"]) for item in baseline["files"]}
    observed = {(item["path"], item["bytes"], item["sha256"]) for item in current}
    missing = sorted(expected - observed)
    added_or_changed = sorted(observed - expected)
    return {
        "schema": "eidosoma.e01.s19_l07_immutable_validation.v1",
        "baselineFileCount": len(expected),
        "currentFileCount": len(observed),
        "missingOrChangedCount": len(missing),
        "addedOrChangedCount": len(added_or_changed),
        "passed": not missing and not added_or_changed,
        "validatedAtUtc": utc_now(),
    }


def _bool_codes(values: pd.Series) -> NDArray[np.int8]:
    return np.asarray([-1 if pd.isna(value) else int(bool(value)) for value in values], dtype=np.int8)


def _positive_float64_ulp_distance(left: NDArray[np.float64], right: NDArray[np.float64]) -> NDArray[np.int64]:
    """ULP distance for nonnegative finite cosine scores."""

    a = np.ascontiguousarray(left, dtype=np.float64)
    b = np.ascontiguousarray(right, dtype=np.float64)
    if np.any(a < 0) or np.any(b < 0) or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("the L07 replay ULP helper requires nonnegative finite scores")
    return np.abs(a.view(np.int64) - b.view(np.int64))


def preanalysis_replay() -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest = pd.read_parquet(TRAJECTORY_MANIFEST_PATH).sort_values(
        ["matrixIndex", "candidateId"], kind="stable"
    )
    frozen = pd.read_parquet(FROZEN_LABEL_PATH)
    frozen = frozen.loc[frozen["labelId"].eq("MOL_ADJACENT_INCOMING_H900")]
    setting = {
        "roundId": "PREANALYSIS_REPLAY",
        "settingId": "FROZEN_COMPARATOR_REPLAY",
        "settingPairId": "FROZEN_COMPARATOR_REPLAY",
        "family": "ADJACENT_CLOCK",
        "clockId": "C1_SELECTED_DAUGHTER_RETAINED",
        "alignment": "INCOMING_DUPLICATE_FIRST",
        "projection": "ALL_OBSERVATIONS",
        "threshold": 0.9,
        "comparator": "STRICT_GT",
    }
    rows = []
    for record in manifest.to_dict(orient="records"):
        path = Path(record["cachePath"])
        cache_hash = sha256_file(path)
        with path.open("rb") as handle:
            trajectory = pickle.load(handle)
        computed = materialize_frozen_setting(trajectory, setting).sort_values(
            "analysisUnitIndex", kind="stable"
        )
        expected = frozen.loc[
            frozen["candidateId"].eq(record["candidateId"])
            & frozen["matrixIndex"].eq(int(record["matrixIndex"]))
        ].sort_values("selectedSequenceIndex", kind="stable")
        clock_pass = len(computed) == len(expected) and np.array_equal(
            computed["rawObservationIndex"].to_numpy(dtype=np.int64),
            expected["rawObservationIndex"].to_numpy(dtype=np.int64),
        )
        labels_pass = np.array_equal(
            _bool_codes(computed["isReplicator"]), _bool_codes(expected["isReplicator"])
        )
        current_scores = computed["score"].to_numpy(dtype=np.float64)
        frozen_scores = expected["labelScore"].to_numpy(dtype=np.float64)
        finite_masks_pass = np.array_equal(np.isfinite(current_scores), np.isfinite(frozen_scores))
        finite = np.isfinite(current_scores) & np.isfinite(frozen_scores)
        absolute = np.abs(current_scores[finite] - frozen_scores[finite])
        relative = absolute / np.maximum(
            np.maximum(np.abs(current_scores[finite]), np.abs(frozen_scores[finite])),
            np.finfo(np.float64).tiny,
        )
        ulp = _positive_float64_ulp_distance(current_scores[finite], frozen_scores[finite])
        maximum_absolute = float(np.max(absolute)) if len(absolute) else 0.0
        maximum_relative = float(np.max(relative)) if len(relative) else 0.0
        maximum_ulp = int(np.max(ulp)) if len(ulp) else 0
        scores_pass = bool(
            finite_masks_pass
            and maximum_absolute <= 1e-12
            and maximum_relative <= 1e-12
            and maximum_ulp <= 8
        )
        identity_pass = bool(
            trajectory.trajectory_id == record["trajectoryId"]
            and trajectory.trajectory_sha256 == record["trajectorySha256"]
            and cache_hash == record["cacheSha256"]
        )
        rows.append(
            {
                "candidateId": record["candidateId"],
                "matrixIndex": int(record["matrixIndex"]),
                "trajectoryId": record["trajectoryId"],
                "cacheSha256": cache_hash,
                "trajectoryIdentityPassed": identity_pass,
                "selectedClockPassed": clock_pass,
                "scoreFiniteMaskPassed": finite_masks_pass,
                "maximumAbsoluteScoreError": maximum_absolute,
                "maximumRelativeScoreError": maximum_relative,
                "maximumUlpDistance": maximum_ulp,
                "adjacentHScorePassed": scores_pass,
                "frozenH900LabelPassed": labels_pass,
                "passed": bool(identity_pass and clock_pass and scores_pass and labels_pass),
            }
        )
    frame = pd.DataFrame(rows)
    summary = {
        "schema": "eidosoma.e01.s19_l07_preanalysis_replay.v1",
        "trajectoryCount": len(frame),
        "passedCount": int(frame["passed"].sum()),
        "selectedClockRows": int(len(frozen)),
        "maximumAbsoluteScoreError": float(frame["maximumAbsoluteScoreError"].max()),
        "maximumRelativeScoreError": float(frame["maximumRelativeScoreError"].max()),
        "maximumUlpDistance": int(frame["maximumUlpDistance"].max()),
        "scorePolicy": "MASK_EXACT_LABEL_EXACT_ABS_LE_1E-12_REL_LE_1E-12_ULP_LE_8",
        "passed": bool(frame["passed"].all() and len(frame) == 200),
        "validatedAtUtc": utc_now(),
    }
    return frame, summary


def expand_frozen_settings(round_id: str) -> list[dict[str, Any]]:
    config = round_config(round_id)
    if "settings" in config:
        return [{"roundId": round_id, **item} for item in config["settings"]]
    contract = config["generatedSettingContract"]
    rows: list[dict[str, Any]] = []
    ordinal = 0
    for clock in contract["clocks"]:
        for alignment in contract["alignments"]:
            for threshold in contract["thresholds"]:
                ordinal += 1
                clock_token = "C1" if clock.startswith("C1") else "C0"
                align_token = "IN" if alignment.startswith("INCOMING") else "AVG"
                threshold_token = f"{int(round(float(threshold) * 10000)):04d}"
                pair = f"{clock_token}_{align_token}_H{threshold_token}"
                rows.append(
                    {
                        "roundId": round_id,
                        "settingId": f"L07-R02-{ordinal:03d}",
                        "settingPairId": pair,
                        "family": "ADJACENT_CLOCK",
                        "clockId": clock,
                        "alignment": alignment,
                        "projection": "ALL_OBSERVATIONS",
                        "threshold": float(threshold),
                        "comparator": contract["comparator"],
                        "sourceTier": contract["sourceTier"],
                    }
                )
    return rows


def expand_simulation_configs(round_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    config = round_config(round_id)
    contract = config["simulationContract"]
    simulations: list[dict[str, Any]] = []
    if round_id == "R03_EXPOSURE_SIMULATOR":
        for h in contract["hValues"]:
            for daughter in contract["daughterRules"]:
                token = f"H{int(round(float(h) * 10000)):05d}"
                candidate = "FIRST" if daughter == "FIRST_DAUGHTER" else "RANDOM"
                simulations.append(
                    {
                        "simulationId": f"L07-R03-{token}-{candidate}",
                        "simulationPairId": f"L07-R03-{token}",
                        "h": float(h),
                        "daughterRule": daughter,
                        "overshootRule": contract["overshootRules"][0],
                        "candidateId": daughter,
                        "phase": "s19_l07_exposure_search",
                        "streamIdentity": f"S19-L07::{token}::{daughter}",
                        "exactFrozenReplay": False,
                    }
                )
        for item in contract.get("exactFrozenReplayConfigurations", []):
            simulations.append(
                {
                    "simulationId": item["configurationId"],
                    "simulationPairId": item["configurationId"],
                    "h": float(item["h"]),
                    "daughterRule": item["daughterRule"],
                    "overshootRule": item["overshootRule"],
                    "candidateId": item["frozenStreamIdentity"],
                    "phase": SIMULATION_PHASE,
                    "streamIdentity": item["frozenStreamIdentity"],
                    "exactFrozenReplay": True,
                }
            )
    else:
        for item in contract["configurations"]:
            simulations.append(dict(item))
    labels = [dict(item) for item in contract["labelSettings"]]
    return simulations, labels


def setting_registry_rows() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for round_id in ("R01_BOUNDARY_CLOCK", "R02_THRESHOLD_TRANSCRIPTION"):
        for item in expand_frozen_settings(round_id):
            rows.append(
                {
                    **item,
                    "trajectorySource": "FROZEN_S13Y",
                    "registeredBeforeOutcome": True,
                    "registrationSource": str(ROUNDS_PATH),
                }
            )
    simulations, labels = expand_simulation_configs("R03_EXPOSURE_SIMULATOR")
    for simulation in simulations:
        for label in labels:
            pair = f"{simulation['simulationPairId']}::{label['suffix']}"
            rows.append(
                {
                    "roundId": "R03_EXPOSURE_SIMULATOR",
                    "settingId": f"{simulation['simulationId']}::{label['suffix']}",
                    "settingPairId": pair,
                    "family": label["family"],
                    "clockId": label["clockId"],
                    "alignment": label["alignment"],
                    "projection": label["projection"],
                    "threshold": label["threshold"],
                    "comparator": label["comparator"],
                    "sourceTier": "PAPER_POISSON_EXPOSURE_MISSING_CONFIGURATION",
                    "trajectorySource": "L07_REGENERATED_SHARED_S13Y_MATRICES",
                    "simulationId": simulation["simulationId"],
                    "h": simulation["h"],
                    "daughterRule": simulation["daughterRule"],
                    "overshootRule": simulation["overshootRule"],
                    "exactFrozenReplay": simulation["exactFrozenReplay"],
                    "registeredBeforeOutcome": True,
                    "registrationSource": str(ROUNDS_PATH),
                }
            )
    for adaptive_round, adaptive_path in ADAPTIVE_ROUND_PATHS.items():
        if not adaptive_path.exists():
            continue
        simulations, labels = expand_simulation_configs(adaptive_round)
        for simulation in simulations:
            for label in labels:
                rows.append(
                    {
                        "roundId": adaptive_round,
                        "settingId": f"{simulation['simulationId']}::{label['suffix']}",
                        "settingPairId": f"{simulation['simulationPairId']}::{label['suffix']}",
                        "family": label["family"],
                        "clockId": label["clockId"],
                        "alignment": label["alignment"],
                        "projection": label["projection"],
                        "threshold": label["threshold"],
                        "comparator": label["comparator"],
                        "sourceTier": str(round_config(adaptive_round).get("sourceBoundary", {}).get("interpretation", "ADAPTIVE_L07_ROUND")),
                        "trajectorySource": str(round_config(adaptive_round).get("matrixContract", {}).get("trajectorySource", "L07_REGENERATED_SHARED_S13Y_MATRICES")),
                        "simulationId": simulation["simulationId"],
                        "h": simulation["h"],
                        "daughterRule": simulation["daughterRule"],
                        "overshootRule": simulation["overshootRule"],
                        "exactFrozenReplay": False,
                        "registeredBeforeOutcome": True,
                        "registrationSource": str(adaptive_path),
                    }
                )
    frame = pd.DataFrame(rows)
    frame.insert(0, "registryOrdinal", np.arange(1, len(frame) + 1))
    return frame


def _benchmark() -> dict[str, Any]:
    manifest = pd.read_parquet(TRAJECTORY_MANIFEST_PATH)
    expected = manifest.loc[
        manifest["candidateId"].eq("S12F-CANDIDATE-02") & manifest["matrixIndex"].eq(0)
    ].iloc[0]
    beta_seed = derive_simulation_seed(ROOT_SEED_HEX, SIMULATION_PHASE, "catalytic_matrix", 0)
    init_seed = derive_simulation_seed(ROOT_SEED_HEX, SIMULATION_PHASE, "initial_state", 0)
    beta = generate_beta(beta_seed)
    initial = initialize_distinct_state(init_seed)
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    trajectory, _ = simulate_trajectory(
        phase="s19_l07_benchmark",
        root_hex=ROOT_SEED_HEX,
        matrix_index=0,
        definition=SimulationDefinition(
            daughter_rule="FIRST_DAUGHTER",
            overshoot_rule="TRIM_NEW_ENTRANTS_TO_NMAX",
            exposure=ExposureDefinition(family="FIXED_COMMON_EXPOSURE", h=0.85),
        ),
        stream_identity="S19-L07-BENCHMARK-NONSCIENTIFIC",
        beta=beta,
        initial_state=initial,
    )
    wall = time.perf_counter() - started_wall
    cpu = time.process_time() - started_cpu
    projected = cpu * 20 * 100 * 2.0
    return {
        "schema": "eidosoma.e01.s19_l07_compute_benchmark.v1",
        "benchmarkScientificOutcomeOpened": False,
        "configurationLockedBeforeBenchmark": True,
        "completedFissions": trajectory.completed_fissions,
        "terminalStatus": trajectory.terminal_status,
        "wallSeconds": wall,
        "cpuSeconds": cpu,
        "projectedR03CpuHoursWithReplayReserve": projected / 3600.0,
        "projectedR03WallHoursAtEightWorkers": projected / 3600.0 / 8.0,
        "betaIdentityPassed": array_sha256(beta) == expected["betaSha256"],
        "initialIdentityPassed": array_sha256(initial) == expected["initialStateSha256"],
        "gatePassed": bool(trajectory.completed_fissions == 100),
        "benchmarkedAtUtc": utc_now(),
    }


def prepare() -> None:
    LOOP_ROOT.mkdir(parents=True, exist_ok=True)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    repository = assert_repository_lock()
    shutil.copy2(PROTOCOL_PATH, LOOP_ROOT / "preregistration.yaml")
    shutil.copy2(ROUNDS_PATH, LOOP_ROOT / "round_registry.yaml")
    baseline_rows = _hash_prior_files()
    baseline = {
        "schema": "eidosoma.e01.s19_l07_immutable_prior_baseline.v1",
        "capturedAtUtc": utc_now(),
        "fileCount": len(baseline_rows),
        "files": baseline_rows,
    }
    write_json(IMMUTABLE_BASELINE_PATH, baseline)
    prior_validation = validate_prior_baseline()
    write_json(LOOP_ROOT / "immutable_prior_validation.json", prior_validation)
    initial_validation = LOOP_ROOT / "preanalysis_replay_validation.json"
    initial_evidence = LOOP_ROOT / "preanalysis_replay_evidence.parquet"
    if initial_validation.exists():
        prior = json.loads(initial_validation.read_text(encoding="utf-8"))
        if not prior.get("passed", False):
            shutil.copy2(
                initial_validation,
                LOOP_ROOT / "preanalysis_replay_validation_attempt_001_failed.json",
            )
            if initial_evidence.exists():
                shutil.copy2(
                    initial_evidence,
                    LOOP_ROOT / "preanalysis_replay_evidence_attempt_001_failed.parquet",
                )
    replay_frame, replay_summary = preanalysis_replay()
    write_parquet(LOOP_ROOT / "preanalysis_replay_evidence.parquet", replay_frame)
    write_json(LOOP_ROOT / "preanalysis_replay_validation.json", replay_summary)
    if not prior_validation["passed"] or not replay_summary["passed"]:
        raise RuntimeError("L07 preanalysis immutable/replay gate failed")

    source_files = [
        Path("/workspace/input-attachments/ed5486bf-a043-485b-a233-d88d8d123759/pdf-markdown.md"),
        Path("/cache/e01_s03/downloads/paper-2607.28250v1.pdf"),
        Path("/cache/e01_s03/sources/gard-historical/tgs_nondrift.m"),
        Path("/cache/e01_s03/sources/gard-historical/tgs_parameters_v10.m"),
        Path("/cache/e01_s03/sources/gard-historical/cluster_traces.m"),
        Path("/cache/e01_s03/sources/gard-historical/tgs_acluster.m"),
        Path("/cache/e01_s03/sources/gard-historical/tgs_agard_v10.m"),
        Path("/cache/e01_s03/sources/gard-historical/README.txt"),
    ]
    source_manifest = {
        "schema": "eidosoma.e01.s19_l07_source_snapshot_manifest.v1",
        "retrievedAtUtc": utc_now(),
        "sources": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in source_files
        ],
        "repositories": [
            {
                "url": "https://github.com/ModelingOriginsofLife/GARD.git",
                "commit": "86dff6320d5ae91b4e831471079ff46749b14df9",
                "licenseStatus": "NO_LICENSE_FILE_FOUND_DO_NOT_REDISTRIBUTE_SOURCE",
            },
            {
                "url": "https://github.com/Amitmiti/GARD-model.git",
                "commit": "19878f6432fdfb30bea5d775175ed42a767eb3ef",
                "licenseStatus": "NO_LICENSE_FILE_FOUND_DO_NOT_REDISTRIBUTE_SOURCE",
            },
        ],
    }
    write_json(LOOP_ROOT / "source_snapshot_manifest.json", source_manifest)
    input_manifest = {
        "schema": "eidosoma.e01.s19_l07_input_manifest.v1",
        "sharedMatrixCount": 100,
        "frozenTrajectoryCount": 200,
        "inputs": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in (
                TRAJECTORY_MANIFEST_PATH,
                FROZEN_LABEL_PATH,
                PROTOCOL_PATH,
                ROUNDS_PATH,
                AMENDMENT_PATH,
            )
        ],
    }
    write_json(LOOP_ROOT / "input_manifest.json", input_manifest)
    benchmark = _benchmark()
    write_json(LOOP_ROOT / "compute_benchmark.json", benchmark)
    if not benchmark["gatePassed"]:
        raise RuntimeError("L07 compute benchmark failed")
    registry = setting_registry_rows()
    registered = utc_now()
    registry["registeredAtUtc"] = registered
    registry["outcomeOpenedAtUtc"] = None
    registry["outcomeStatus"] = "LOCKED_UNOPENED"
    write_parquet(LOOP_ROOT / "setting_registry.parquet", registry)
    method_lock = {
        "schema": "eidosoma.e01.s19_l07_method_lock.v1",
        "version": VERSION,
        "lockedAtUtc": registered,
        "repository": repository,
        "files": [
            {"path": str(path.relative_to(REPO)), "sha256": sha256_file(path)}
            for path in (
                PROTOCOL_PATH,
                ROUNDS_PATH,
                AMENDMENT_PATH,
                REPO / "src/e01_s19_occupancy_search/core.py",
                REPO / "scripts/e01/run_s19_l07.py",
                REPO / "tests/e01/test_s19_l07.py",
            )
        ],
        "scientificTarget": {"occupancy": PAPER_OCCUPANCY_TARGET, "tolerance": PAPER_OCCUPANCY_TOLERANCE},
        "passed": True,
    }
    write_json(LOOP_ROOT / "method_lock.json", method_lock)
    write_json(
        LOOP_ROOT / "search_waiver.json",
        {
            "schema": "eidosoma.e01.s19_l07_human_waiver.v1",
            "soleScientificSuccessTarget": "CLOSENESS_TO_APPROXIMATELY_0.88_OCCUPANCY",
            "exact88Required": False,
            "allOtherPaperFingerprintAndPromotionGatesWaived": True,
            "temporalFingerprintsDescriptiveOnly": True,
            "integrityAndReproducibilityRequirementsWaived": False,
            "authorCodeIdentityMayBeClaimed": False,
            "source": "explicit_human_direction",
        },
    )


def assert_prepared(round_id: str) -> None:
    repository = assert_repository_lock()
    lock = json.loads((LOOP_ROOT / "method_lock.json").read_text(encoding="utf-8"))
    if round_id in ADAPTIVE_ROUND_PATHS:
        lock_name = round_id.split("_", 1)[0].lower() + "_method_lock.json"
        refinement_lock = json.loads((LOOP_ROOT / lock_name).read_text(encoding="utf-8"))
        if repository["head"] != refinement_lock["repository"]["head"]:
            raise RuntimeError(f"repository changed after {round_id} adaptive lock")
        adaptive_path = ADAPTIVE_ROUND_PATHS[round_id]
        if sha256_file(adaptive_path) != refinement_lock["refinementConfigSha256"]:
            raise RuntimeError(f"{round_id} adaptive config changed after lock")
    elif repository["head"] != lock["repository"]["head"]:
        raise RuntimeError("repository changed after L07 method lock")
    for key in ("immutable_prior_validation.json", "preanalysis_replay_validation.json", "compute_benchmark.json"):
        if not json.loads((LOOP_ROOT / key).read_text(encoding="utf-8"))["passed" if key != "compute_benchmark.json" else "gatePassed"]:
            raise RuntimeError(f"prepared gate failed: {key}")


def _fingerprint_row(frame: pd.DataFrame, setting: dict[str, Any], trajectory: Any) -> dict[str, Any]:
    result = fingerprint(frame)
    labels = _bool_codes(frame["isReplicator"])
    scores = frame["score"].to_numpy(dtype=np.float64)
    return {
        "roundId": setting["roundId"],
        "settingId": setting["settingId"],
        "settingPairId": setting["settingPairId"],
        "candidateId": str(trajectory.configuration_id),
        "matrixIndex": int(trajectory.matrix_index),
        "trajectoryId": str(trajectory.trajectory_id),
        "trajectorySha256": str(trajectory.trajectory_sha256),
        "labelSha256": hashlib.sha256(labels.tobytes()).hexdigest(),
        "scoreSha256": hashlib.sha256(np.ascontiguousarray(scores).tobytes()).hexdigest(),
        **result,
    }


def run_frozen(round_id: str, workers: int) -> None:
    assert_prepared(round_id)
    settings = expand_frozen_settings(round_id)
    manifest = pd.read_parquet(TRAJECTORY_MANIFEST_PATH).sort_values(
        ["matrixIndex", "candidateId"], kind="stable"
    )
    started = utc_now()
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for record in manifest.to_dict(orient="records"):
        try:
            with Path(record["cachePath"]).open("rb") as handle:
                trajectory = pickle.load(handle)
            for setting in settings:
                frame = materialize_frozen_setting(trajectory, setting)
                rows.append(_fingerprint_row(frame, setting, trajectory))
        except Exception as error:  # status-bearing exploratory search
            failures.append(
                {
                    "roundId": round_id,
                    "candidateId": record["candidateId"],
                    "matrixIndex": int(record["matrixIndex"]),
                    "failureType": type(error).__name__,
                    "message": str(error),
                }
            )
    frame = pd.DataFrame(rows)
    expected = len(manifest) * len(settings)
    if failures or len(frame) != expected:
        write_csv(LOOP_ROOT / f"{round_id}_failure_ledger.csv", pd.DataFrame(failures))
        raise RuntimeError(f"{round_id} incomplete: {len(frame)}/{expected}; failures={len(failures)}")
    aggregate = aggregate_occupancy(frame)
    ranking = summarize_pairs(aggregate)
    write_parquet(LOOP_ROOT / f"{round_id}_trajectory_results.parquet", frame)
    write_parquet(LOOP_ROOT / f"{round_id}_occupancy_results.parquet", aggregate)
    write_csv(LOOP_ROOT / f"{round_id}_occupancy_ranking.csv", ranking)
    finish_round(
        round_id,
        started,
        wall_start,
        cpu_start,
        attempted=len(settings),
        trajectory_results=len(frame),
        failures=0,
    )


def _simulation_worker(matrix_index: int, round_id: str) -> dict[str, Any]:
    simulations, label_settings = expand_simulation_configs(round_id)
    config = round_config(round_id)
    matrix_contract = config.get("matrixContract", {})
    root_hex = str(matrix_contract.get("rootSeedHex", ROOT_SEED_HEX))
    matrix_phase = str(matrix_contract.get("matrixPhase", SIMULATION_PHASE))
    beta_seed = derive_simulation_seed(root_hex, matrix_phase, "catalytic_matrix", matrix_index)
    init_seed = derive_simulation_seed(root_hex, matrix_phase, "initial_state", matrix_index)
    beta = generate_beta(beta_seed)
    initial = initialize_distinct_state(init_seed)
    output_rows = []
    summary_rows = []
    failure_rows = []
    seed_rows = []
    for simulation in simulations:
        started_wall = time.perf_counter()
        started_cpu = time.process_time()
        try:
            exposure = (
                ExposureDefinition(family="FIXED_COMMON_EXPOSURE", h=float(simulation["h"]))
                if float(simulation["h"]) <= 1.25
                else ExploratoryExposureDefinition(
                    family="FIXED_COMMON_EXPOSURE", h=float(simulation["h"])
                )
            )
            definition = SimulationDefinition(
                daughter_rule=simulation["daughterRule"],
                overshoot_rule=simulation["overshootRule"],
                exposure=exposure,
            )
            trajectory, seeds = simulate_trajectory(
                phase=simulation["phase"],
                root_hex=root_hex,
                matrix_index=matrix_index,
                definition=definition,
                stream_identity=simulation["streamIdentity"],
                beta=beta,
                initial_state=initial,
            )
            summary = trajectory_summary(trajectory)
            summary_rows.append(
                {
                    "roundId": round_id,
                    "simulationId": simulation["simulationId"],
                    "simulationPairId": simulation["simulationPairId"],
                    "candidateId": simulation["candidateId"],
                    "matrixIndex": matrix_index,
                    "exactFrozenReplay": simulation["exactFrozenReplay"],
                    "wallSeconds": time.perf_counter() - started_wall,
                    "cpuSeconds": time.process_time() - started_cpu,
                    **summary,
                }
            )
            for seed in seeds:
                seed_rows.append(
                    {
                        "roundId": round_id,
                        "simulationId": simulation["simulationId"],
                        "candidateId": simulation["candidateId"],
                        "matrixIndex": matrix_index,
                        "purpose": seed.purpose,
                        "derivedSeed": str(seed.derived_seed),
                        "seedMaterialSha256": seed.seed_material_sha256,
                        "rootHex": seed.root_sha256,
                    }
                )
            for label in label_settings:
                setting = {
                    "roundId": round_id,
                    "settingId": f"{simulation['simulationId']}::{label['suffix']}",
                    "settingPairId": f"{simulation['simulationPairId']}::{label['suffix']}",
                    **{key: value for key, value in label.items() if key != "suffix"},
                }
                labels = materialize_frozen_setting(trajectory, setting)
                row = _fingerprint_row(labels, setting, trajectory)
                row["candidateId"] = simulation["candidateId"]
                row["simulationId"] = simulation["simulationId"]
                row["h"] = simulation["h"]
                row["daughterRule"] = simulation["daughterRule"]
                row["overshootRule"] = simulation["overshootRule"]
                row["completedFissions"] = trajectory.completed_fissions
                row["terminalStatus"] = trajectory.terminal_status
                output_rows.append(row)
        except Exception as error:
            failure_rows.append(
                {
                    "roundId": round_id,
                    "simulationId": simulation["simulationId"],
                    "candidateId": simulation["candidateId"],
                    "matrixIndex": matrix_index,
                    "failureType": type(error).__name__,
                    "message": str(error),
                }
            )
    return {
        "fingerprints": output_rows,
        "summaries": summary_rows,
        "failures": failure_rows,
        "seeds": seed_rows,
    }


def validate_exact_s13y_replay(summaries: pd.DataFrame) -> pd.DataFrame:
    expected = pd.read_parquet(TRAJECTORY_MANIFEST_PATH)
    rows = []
    mapping = {
        "S13Y_BASELINE_CANDIDATE_02": "S12F-CANDIDATE-02",
        "S13Y_BASELINE_CANDIDATE_03": "S12F-CANDIDATE-03",
    }
    for simulation_id, candidate in mapping.items():
        current = summaries.loc[summaries["simulationId"].eq(simulation_id)]
        frozen = expected.loc[expected["candidateId"].eq(candidate)]
        merged = current.merge(frozen, on="matrixIndex", suffixes=("", "Frozen"), how="outer", indicator=True)
        for row in merged.to_dict(orient="records"):
            passed = bool(
                row["_merge"] == "both"
                and row["trajectorySha256"] == row["trajectorySha256Frozen"]
                and int(row["completedFissions"]) == int(row["completedFissionsFrozen"])
                and row["terminalStatus"] == row["terminalStatusFrozen"]
            )
            rows.append(
                {
                    "simulationId": simulation_id,
                    "candidateId": candidate,
                    "matrixIndex": int(row["matrixIndex"]),
                    "trajectorySha256": row.get("trajectorySha256"),
                    "frozenTrajectorySha256": row.get("trajectorySha256Frozen"),
                    "passed": passed,
                }
            )
    return pd.DataFrame(rows)


def validate_fresh_seed_firewall(summaries: pd.DataFrame) -> dict[str, Any]:
    """Require fresh R06 catalytic matrices and initial states by exact hash."""

    frozen = pd.read_parquet(TRAJECTORY_MANIFEST_PATH)
    frozen_beta = set(frozen["betaSha256"].astype(str))
    frozen_initial = set(frozen["initialStateSha256"].astype(str))
    current_beta = set(summaries["betaSha256"].astype(str))
    current_initial = set(summaries["initialStateSha256"].astype(str))
    beta_overlap = sorted(current_beta.intersection(frozen_beta))
    initial_overlap = sorted(current_initial.intersection(frozen_initial))
    matrix_counts = summaries.groupby("simulationId")["matrixIndex"].nunique()
    return {
        "schema": "eidosoma.e01.s19_l07_r06_seed_firewall.v1",
        "frozenBetaHashCount": len(frozen_beta),
        "frozenInitialStateHashCount": len(frozen_initial),
        "freshBetaHashCount": len(current_beta),
        "freshInitialStateHashCount": len(current_initial),
        "betaOverlapCount": len(beta_overlap),
        "initialStateOverlapCount": len(initial_overlap),
        "allSimulationsHave100Matrices": bool((matrix_counts == 100).all()),
        "passed": bool(
            not beta_overlap
            and not initial_overlap
            and len(current_beta) == 100
            and len(current_initial) == 100
            and (matrix_counts == 100).all()
        ),
        "validatedAtUtc": utc_now(),
    }


def run_simulation(round_id: str, workers: int) -> None:
    assert_prepared(round_id)
    simulations, labels = expand_simulation_configs(round_id)
    started = utc_now()
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    outputs = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_simulation_worker, matrix, round_id): matrix for matrix in range(100)}
        for future in as_completed(futures):
            outputs.append(future.result())
    fingerprints = pd.DataFrame([row for output in outputs for row in output["fingerprints"]])
    summaries = pd.DataFrame([row for output in outputs for row in output["summaries"]])
    failures = pd.DataFrame([row for output in outputs for row in output["failures"]])
    seeds = pd.DataFrame([row for output in outputs for row in output["seeds"]])
    expected = len(simulations) * len(labels) * 100
    write_csv(LOOP_ROOT / f"{round_id}_failure_ledger.csv", failures)
    if len(fingerprints) != expected or not failures.empty:
        raise RuntimeError(
            f"{round_id} incomplete: {len(fingerprints)}/{expected}, failures={len(failures)}"
        )
    if round_id == "R03_EXPOSURE_SIMULATOR":
        replay = validate_exact_s13y_replay(summaries)
        write_parquet(LOOP_ROOT / "R03_exact_s13y_replay.parquet", replay)
        if len(replay) != 200 or not replay["passed"].all():
            raise RuntimeError("R03 exact S13Y baseline replay failed")
    if round_id == "R06_FRESH_SEED_VALIDATION":
        firewall = validate_fresh_seed_firewall(summaries)
        write_json(LOOP_ROOT / "R06_seed_firewall.json", firewall)
        if not firewall["passed"]:
            raise RuntimeError("R06 fresh-seed firewall failed")
    aggregate = aggregate_occupancy(fingerprints)
    ranking = summarize_pairs(aggregate)
    write_parquet(LOOP_ROOT / f"{round_id}_trajectory_results.parquet", fingerprints)
    write_parquet(LOOP_ROOT / f"{round_id}_simulation_summary.parquet", summaries)
    write_parquet(LOOP_ROOT / f"{round_id}_seed_manifest.parquet", seeds)
    write_parquet(LOOP_ROOT / f"{round_id}_occupancy_results.parquet", aggregate)
    write_csv(LOOP_ROOT / f"{round_id}_occupancy_ranking.csv", ranking)
    finish_round(
        round_id,
        started,
        wall_start,
        cpu_start,
        attempted=len(simulations) * len(labels),
        trajectory_results=len(fingerprints),
        failures=0,
    )


def finish_round(
    round_id: str,
    started: str,
    wall_start: float,
    cpu_start: float,
    *,
    attempted: int,
    trajectory_results: int,
    failures: int,
) -> None:
    finished = utc_now()
    status = {
        "schema": "eidosoma.e01.s19_l07_round_status.v1",
        "roundId": round_id,
        "startedAtUtc": started,
        "outcomeOpenedAtUtc": finished,
        "wallSeconds": time.perf_counter() - wall_start,
        "coordinatorCpuSeconds": time.process_time() - cpu_start,
        "attemptedSettingCount": attempted,
        "trajectoryResultCount": trajectory_results,
        "failureCount": failures,
        "complete": failures == 0,
    }
    write_json(LOOP_ROOT / f"{round_id}_status.json", status)
    registry = pd.read_parquet(LOOP_ROOT / "setting_registry.parquet")
    mask = registry["roundId"].eq(round_id)
    registry.loc[mask, "outcomeOpenedAtUtc"] = finished
    registry.loc[mask, "outcomeStatus"] = "COMPLETE" if failures == 0 else "FAILED"
    write_parquet(LOOP_ROOT / "setting_registry.parquet", registry)
    attempt_path = LOOP_ROOT / "chronological_attempt_ledger.parquet"
    current = registry.loc[mask].copy()
    current["attemptOrderWithinRound"] = np.arange(1, len(current) + 1)
    if attempt_path.exists():
        prior = pd.read_parquet(attempt_path)
        current = pd.concat([prior, current], ignore_index=True)
    write_parquet(attempt_path, current)


def prepare_adaptive_round(round_id: str) -> None:
    repository = assert_repository_lock()
    adaptive_path = ADAPTIVE_ROUND_PATHS[round_id]
    if not adaptive_path.exists():
        raise FileNotFoundError(adaptive_path)
    config = yaml.safe_load(adaptive_path.read_text(encoding="utf-8"))
    basis_round = config["adaptiveBasis"]["openedRound"]
    basis_status = json.loads(
        (LOOP_ROOT / f"{basis_round}_status.json").read_text(encoding="utf-8")
    )
    if not basis_status.get("complete", False):
        raise RuntimeError(f"{round_id} requires complete {basis_round}")
    result_path = Path(config["adaptiveBasis"]["resultPath"])
    if sha256_file(result_path) != config["adaptiveBasis"]["resultSha256"]:
        raise RuntimeError(f"{round_id} adaptive basis result hash mismatch")
    registry = pd.read_parquet(LOOP_ROOT / "setting_registry.parquet")
    if registry["roundId"].eq(round_id).any():
        raise RuntimeError(f"{round_id} registry rows already exist")
    expanded = setting_registry_rows()
    new_rows = expanded.loc[expanded["roundId"].eq(round_id)].copy()
    new_rows["registryOrdinal"] = np.arange(len(registry) + 1, len(registry) + len(new_rows) + 1)
    registered = utc_now()
    new_rows["registeredAtUtc"] = registered
    new_rows["outcomeOpenedAtUtc"] = None
    new_rows["outcomeStatus"] = "LOCKED_UNOPENED"
    all_columns = sorted(set(registry.columns).union(new_rows.columns))
    registry = registry.reindex(columns=all_columns)
    new_rows = new_rows.reindex(columns=all_columns)
    write_parquet(
        LOOP_ROOT / "setting_registry.parquet",
        pd.concat([registry, new_rows], ignore_index=True),
    )
    round_prefix = round_id.split("_", 1)[0]
    shutil.copy2(adaptive_path, LOOP_ROOT / f"round_{round_prefix}_lock.yaml")
    write_json(
        LOOP_ROOT / f"{round_prefix.lower()}_method_lock.json",
        {
            "schema": f"eidosoma.e01.s19_l07_{round_prefix.lower()}_method_lock.v1",
            "roundId": round_id,
            "lockedAtUtc": registered,
            "repository": repository,
            "refinementConfigSha256": sha256_file(adaptive_path),
            "adaptiveBasisResultSha256": sha256_file(result_path),
            "runnerSha256": sha256_file(REPO / "scripts/e01/run_s19_l07.py"),
            "coreSha256": sha256_file(REPO / "src/e01_s19_occupancy_search/core.py"),
            "registeredSettingCount": len(new_rows),
            "passed": True,
        },
    )


def _exact_frame_match(
    expected: pd.DataFrame,
    observed: pd.DataFrame,
    *,
    sort_columns: list[str],
    excluded_columns: tuple[str, ...] = (),
) -> tuple[bool, str | None]:
    """Compare deterministic scientific tables while excluding runtime telemetry."""

    columns = [name for name in expected.columns if name not in excluded_columns]
    if set(columns) != set(name for name in observed.columns if name not in excluded_columns):
        return False, "column_set_mismatch"
    left = expected[columns].sort_values(sort_columns, kind="stable").reset_index(drop=True)
    right = observed[columns].sort_values(sort_columns, kind="stable").reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(left, right, check_dtype=False, check_exact=True)
    except AssertionError as error:
        return False, str(error)[:1000]
    return True, None


def regenerate_all(workers: int) -> None:
    """Independently rerun every L07 setting and require exact scientific replay."""

    repository = assert_repository_lock()
    rows: list[dict[str, Any]] = []
    round_ids = (
        "R01_BOUNDARY_CLOCK",
        "R02_THRESHOLD_TRANSCRIPTION",
        "R03_EXPOSURE_SIMULATOR",
        "R04_ADAPTIVE_EXPOSURE_REFINEMENT",
        "R05_EXPOSURE_LOCAL_BRACKETING",
        "R06_FRESH_SEED_VALIDATION",
    )
    manifest = pd.read_parquet(TRAJECTORY_MANIFEST_PATH).sort_values(
        ["matrixIndex", "candidateId"], kind="stable"
    )
    started = utc_now()
    wall_start = time.perf_counter()
    for round_id in round_ids:
        if round_id in {"R01_BOUNDARY_CLOCK", "R02_THRESHOLD_TRANSCRIPTION"}:
            regenerated_rows: list[dict[str, Any]] = []
            for record in manifest.to_dict(orient="records"):
                with Path(record["cachePath"]).open("rb") as handle:
                    trajectory = pickle.load(handle)
                for setting in expand_frozen_settings(round_id):
                    regenerated_rows.append(
                        _fingerprint_row(
                            materialize_frozen_setting(trajectory, setting), setting, trajectory
                        )
                    )
            fingerprints = pd.DataFrame(regenerated_rows)
            summaries = None
            seeds = None
        else:
            outputs = []
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(_simulation_worker, matrix, round_id): matrix
                    for matrix in range(100)
                }
                for future in as_completed(futures):
                    outputs.append(future.result())
            failures = [row for output in outputs for row in output["failures"]]
            if failures:
                raise RuntimeError(f"{round_id} regeneration produced {len(failures)} failures")
            fingerprints = pd.DataFrame(
                [row for output in outputs for row in output["fingerprints"]]
            )
            summaries = pd.DataFrame(
                [row for output in outputs for row in output["summaries"]]
            )
            seeds = pd.DataFrame([row for output in outputs for row in output["seeds"]])

        expected_fingerprints = pd.read_parquet(
            LOOP_ROOT / f"{round_id}_trajectory_results.parquet"
        )
        passed, detail = _exact_frame_match(
            expected_fingerprints,
            fingerprints,
            sort_columns=["settingId", "candidateId", "matrixIndex"],
        )
        rows.append(
            {
                "roundId": round_id,
                "component": "TRAJECTORY_FINGERPRINTS",
                "expectedRows": len(expected_fingerprints),
                "observedRows": len(fingerprints),
                "exactScientificReplay": passed,
                "detail": detail,
            }
        )

        aggregate = aggregate_occupancy(fingerprints)
        expected_aggregate = pd.read_parquet(
            LOOP_ROOT / f"{round_id}_occupancy_results.parquet"
        )
        passed, detail = _exact_frame_match(
            expected_aggregate,
            aggregate,
            sort_columns=["settingId", "candidateId"],
        )
        rows.append(
            {
                "roundId": round_id,
                "component": "OCCUPANCY_AGGREGATES_AND_BOOTSTRAPS",
                "expectedRows": len(expected_aggregate),
                "observedRows": len(aggregate),
                "exactScientificReplay": passed,
                "detail": detail,
            }
        )

        if summaries is not None and seeds is not None:
            expected_summaries = pd.read_parquet(
                LOOP_ROOT / f"{round_id}_simulation_summary.parquet"
            )
            passed, detail = _exact_frame_match(
                expected_summaries,
                summaries,
                sort_columns=["simulationId", "matrixIndex"],
                excluded_columns=("wallSeconds", "cpuSeconds"),
            )
            rows.append(
                {
                    "roundId": round_id,
                    "component": "SIMULATION_SCIENTIFIC_SUMMARIES",
                    "expectedRows": len(expected_summaries),
                    "observedRows": len(summaries),
                    "exactScientificReplay": passed,
                    "detail": detail,
                }
            )
            expected_seeds = pd.read_parquet(
                LOOP_ROOT / f"{round_id}_seed_manifest.parquet"
            )
            passed, detail = _exact_frame_match(
                expected_seeds,
                seeds,
                sort_columns=["simulationId", "matrixIndex", "purpose", "derivedSeed"],
            )
            rows.append(
                {
                    "roundId": round_id,
                    "component": "SEED_MANIFEST",
                    "expectedRows": len(expected_seeds),
                    "observedRows": len(seeds),
                    "exactScientificReplay": passed,
                    "detail": detail,
                }
            )

    result = pd.DataFrame(rows)
    write_parquet(LOOP_ROOT / "regeneration_results.parquet", result)
    passed = bool(result["exactScientificReplay"].all())
    write_json(
        LOOP_ROOT / "regeneration_validation.json",
        {
            "schema": "eidosoma.e01.s19_l07_regeneration_validation.v1",
            "startedAtUtc": started,
            "completedAtUtc": utc_now(),
            "wallSeconds": time.perf_counter() - wall_start,
            "repository": repository,
            "roundCount": len(round_ids),
            "componentCount": len(result),
            "passedComponentCount": int(result["exactScientificReplay"].sum()),
            "failedComponentCount": int((~result["exactScientificReplay"]).sum()),
            "exactScientificReplay": passed,
            "passed": passed,
        },
    )
    if not passed:
        raise RuntimeError("L07 exact regeneration failed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=("prepare", "run-r01", "run-r02", "run-r03", "prepare-r04", "run-r04", "prepare-r05", "run-r05", "prepare-r06", "run-r06", "regenerate"),
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be between 1 and 8")
    return args


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "run-r01":
        run_frozen("R01_BOUNDARY_CLOCK", args.workers)
    elif args.command == "run-r02":
        run_frozen("R02_THRESHOLD_TRANSCRIPTION", args.workers)
    elif args.command == "run-r03":
        run_simulation("R03_EXPOSURE_SIMULATOR", args.workers)
    elif args.command == "prepare-r04":
        prepare_adaptive_round("R04_ADAPTIVE_EXPOSURE_REFINEMENT")
    elif args.command == "run-r04":
        run_simulation("R04_ADAPTIVE_EXPOSURE_REFINEMENT", args.workers)
    elif args.command == "prepare-r05":
        prepare_adaptive_round("R05_EXPOSURE_LOCAL_BRACKETING")
    elif args.command == "run-r05":
        run_simulation("R05_EXPOSURE_LOCAL_BRACKETING", args.workers)
    elif args.command == "prepare-r06":
        prepare_adaptive_round("R06_FRESH_SEED_VALIDATION")
    elif args.command == "run-r06":
        run_simulation("R06_FRESH_SEED_VALIDATION", args.workers)
    elif args.command == "regenerate":
        regenerate_all(args.workers)


if __name__ == "__main__":
    main()
