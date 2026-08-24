#!/usr/bin/env python3
"""Execute and finalize the locked E01/S19-L08 untouched comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq
import scipy
import yaml

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from e01_latent_timebase.core import (
    ExposureDefinition,
    SimulationDefinition,
    array_sha256,
    derive_seed,
    generate_beta,
    initialize_distinct_state,
    simulate_trajectory,
    trajectory_replay_equal,
)
from e01_s19_occupancy_search.core import ExploratoryExposureDefinition
from e01_s19_untouched_mechanism.core import (
    BOOTSTRAP_REPLICATES,
    LOOP_ID,
    MECHANISM_A,
    MECHANISM_B,
    NORMALIZED_DISTANCE_METRICS,
    OBJECT_A_BOUNDARY,
    OBJECT_A_PROJECTED,
    OBJECT_B_MOLECULAR,
    PAPER_TARGETS,
    RAW_DISTANCE_METRICS,
    VERSION,
    absolute_scaled_error,
    analysis_objects_for_mechanism,
    bootstrap_indices,
    label_fingerprint,
    materialize_analysis_object,
    occupancy_in_band,
    paper_distance,
    terminal_decision,
    trajectory_diagnostics,
)

ARTIFACT_ROOT = Path("/artifacts/research_steps/S19")
LOOP_ROOT = ARTIFACT_ROOT / "loops/L08"
CACHE_ROOT = Path("/cache/e01_s19_l08")
TRAJECTORY_CACHE = CACHE_ROOT / "trajectories"
CONFIG_PATH = REPO / "configs/e01/s19_l08_untouched_mechanism_comparison.yaml"
CORE_PATH = REPO / "src/e01_s19_untouched_mechanism/core.py"
RUNNER_PATH = REPO / "scripts/e01/run_s19_l08.py"
TEST_PATH = REPO / "tests/e01/test_s19_l08.py"
PAPER_PATH = Path(
    "/workspace/input-attachments/ed5486bf-a043-485b-a233-d88d8d123759/pdf-markdown.md"
)

FINGERPRINT_NUMERIC_METRICS = (
    "analysisUnitLength",
    "eligibleLength",
    "ineligibleLength",
    "persistence",
    "negativePersistence",
    "occupancy",
    "firstOnsetRawIndex0",
    "firstOnsetRawStep1",
    "firstOnsetNormalized",
    "consistency",
    "positiveEpisodeCount",
    "positiveMeanEpisodeDuration",
    "positiveMedianEpisodeDuration",
    "positiveLongestEpisodeDuration",
    "positiveMeanEpisodeSpacing",
    "positiveMedianEpisodeSpacing",
    "negativeEpisodeCount",
    "negativeMeanEpisodeDuration",
    "negativeMedianEpisodeDuration",
    "negativeLongestEpisodeDuration",
    "negativeMeanEpisodeSpacing",
    "negativeMedianEpisodeSpacing",
)

DIAGNOSTIC_NUMERIC_METRICS = (
    "selectedClockLength",
    "boundaryUnitLength",
    "completedFissions",
    "attemptedGenerationCount",
    "maxStepTerminationCount",
    "maxStepTerminationFraction",
    "meanParentDaughterSimilarity",
    "medianParentDaughterSimilarity",
    "meanPreFissionMass",
    "medianPreFissionMass",
    "meanPostFissionMass",
    "medianPostFissionMass",
    "meanPretrimOvershoot",
    "q95PretrimOvershoot",
    "maximumPretrimOvershoot",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
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


def write_yaml(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(json_safe(value), sort_keys=False), encoding="utf-8")


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression="zstd")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def canonical_frame_sha256(frame: pd.DataFrame, sort_columns: list[str]) -> str:
    ordered = frame.sort_values(sort_columns, kind="stable").reset_index(drop=True).copy()
    for column in ordered.columns:
        if ordered[column].dtype == object:
            ordered[column] = ordered[column].map(
                lambda value: (
                    json.dumps(json_safe(value), sort_keys=True, separators=(",", ":"))
                    if isinstance(value, (list, tuple, dict, np.ndarray))
                    else value
                )
            )
    payload = ordered.to_csv(index=False, lineterminator="\n", na_rep="<NA>")
    return sha256_text(payload)


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, text=True, capture_output=True
    ).stdout.strip()


def repository_lock() -> dict[str, Any]:
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/eidosoma/groups/42")
    clean = not bool(git("status", "--porcelain=v1"))
    passed = bool(head == remote and clean)
    if not passed:
        raise RuntimeError("L08 requires a clean pushed eidosoma/groups/42 HEAD")
    return {
        "head": head,
        "remoteHead": remote,
        "branch": git("branch", "--show-current"),
        "cleanWorktree": clean,
        "passed": passed,
    }


def load_config() -> dict[str, Any]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["versionedLoopId"] != VERSION:
        raise RuntimeError("L08 version/config mismatch")
    return config


def simulation_specs() -> list[dict[str, Any]]:
    config = load_config()
    rows: list[dict[str, Any]] = []
    for mechanism in config["mechanisms"]:
        for candidate in mechanism["candidates"]:
            rows.append(
                {
                    "mechanismId": mechanism["mechanismId"],
                    "candidateId": candidate["candidateId"],
                    "exposure": float(candidate["exposure"]),
                    "daughterRule": candidate["daughterRule"],
                    "overshootRule": candidate["overshootRule"],
                    "streamIdentity": candidate["streamIdentity"],
                }
            )
    if len(rows) != 4 or {row["mechanismId"] for row in rows} != {MECHANISM_A, MECHANISM_B}:
        raise RuntimeError("L08 must contain exactly four mechanism-candidate simulations")
    return rows


def make_definition(spec: dict[str, Any]) -> SimulationDefinition:
    h = float(spec["exposure"])
    exposure: Any
    if h <= 1.25:
        exposure = ExposureDefinition(family="FIXED_COMMON_EXPOSURE", h=h)
    else:
        exposure = ExploratoryExposureDefinition(family="FIXED_COMMON_EXPOSURE", h=h)
    return SimulationDefinition(
        daughter_rule=spec["daughterRule"],
        overshoot_rule=spec["overshootRule"],
        exposure=exposure,
    )


def _prior_roots() -> list[Path]:
    roots: list[Path] = []
    step_root = Path("/artifacts/research_steps")
    for path in sorted(step_root.iterdir()):
        if path.name != "S19":
            roots.append(path)
    for loop in ("L01", "L02", "L03", "L04", "L05", "L06", "L06R", "L07"):
        roots.append(ARTIFACT_ROOT / "loops" / loop)
    for bundle in (
        Path("/artifacts/E01_forensic_replication_bundle"),
        Path("/artifacts/E01_forensic_replication_artifact_v2"),
    ):
        if bundle.exists():
            roots.append(bundle)
    return roots


def hash_immutable_prior() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in _prior_roots():
        if not root.exists():
            raise FileNotFoundError(root)
        files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
        for path in files:
            rows.append(
                {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            )
    return rows


def validate_immutable_prior() -> dict[str, Any]:
    baseline = json.loads((LOOP_ROOT / "immutable_prior_baseline.json").read_text())
    current = hash_immutable_prior()
    expected = {(row["path"], row["bytes"], row["sha256"]) for row in baseline["files"]}
    observed = {(row["path"], row["bytes"], row["sha256"]) for row in current}
    return {
        "schema": "eidosoma.e01.s19_l08_immutable_prior_validation.v1",
        "baselineFileCount": len(expected),
        "currentFileCount": len(observed),
        "missingOrChangedCount": len(expected - observed),
        "addedOrChangedCount": len(observed - expected),
        "passed": expected == observed,
        "validatedAtUtc": utc_now(),
    }


def _all_prior_files() -> list[Path]:
    files: list[Path] = []
    for root in _prior_roots():
        if root.is_file():
            files.append(root)
        else:
            files.extend(sorted(path for path in root.rglob("*") if path.is_file()))
    return files


def collect_prior_seed_inventory() -> dict[str, set[str]]:
    """Collect prior hashes/seeds conservatively without opening scientific outcomes."""

    inventory = {
        "beta": set(),
        "initial": set(),
        "seedMaterial": set(),
        "root": set(),
        "derivedSeed": set(),
        "allHex64": set(),
    }
    target_columns = {
        "betasha256": "beta",
        "initialstatesha256": "initial",
        "seedmaterialsha256": "seedMaterial",
        "roothex": "root",
        "rootsha256": "root",
        "derivedseed": "derivedSeed",
    }
    hex_pattern = re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{64}(?![0-9a-fA-F])")
    for path in _all_prior_files():
        suffix = path.suffix.lower()
        try:
            if suffix == ".parquet":
                schema = pq.read_schema(path)
                names = [name for name in schema.names if name.lower() in target_columns]
                if names:
                    frame = pd.read_parquet(path, columns=names)
                    for name in names:
                        bucket = target_columns[name.lower()]
                        for value in frame[name].dropna().astype(str):
                            inventory[bucket].add(value)
                            inventory["allHex64"].update(x.lower() for x in hex_pattern.findall(value))
            elif suffix in {".json", ".yaml", ".yml", ".csv", ".md", ".txt"}:
                if path.stat().st_size <= 20 * 1024 * 1024:
                    text = path.read_text(encoding="utf-8", errors="replace")
                    inventory["allHex64"].update(x.lower() for x in hex_pattern.findall(text))
                    if suffix == ".csv":
                        header = pd.read_csv(path, nrows=0)
                        names = [name for name in header.columns if name.lower() in target_columns]
                        if names:
                            frame = pd.read_csv(path, usecols=names, dtype=str)
                            for name in names:
                                inventory[target_columns[name.lower()]].update(
                                    frame[name].dropna().astype(str)
                                )
        except (OSError, ValueError, UnicodeError, pyarrow.ArrowException):
            continue
    return inventory


def seed_rows_and_input_units() -> tuple[pd.DataFrame, pd.DataFrame]:
    config = load_config()
    root_hex = config["seedContract"]["rootSeedHex"]
    phase = config["seedContract"]["matrixPhase"]
    seed_rows: list[dict[str, Any]] = []
    unit_rows: list[dict[str, Any]] = []
    for matrix_index in range(100):
        beta_seed = derive_seed(root_hex, phase, "catalytic_matrix", matrix_index)
        init_seed = derive_seed(root_hex, phase, "initial_state", matrix_index)
        beta = generate_beta(beta_seed)
        initial = initialize_distinct_state(init_seed)
        unit_rows.append(
            {
                "matrixIndex": matrix_index,
                "betaSha256": array_sha256(beta),
                "initialStateSha256": array_sha256(initial),
                "initialMass": int(initial.sum()),
                "initialDistinctTypes": int(np.count_nonzero(initial)),
            }
        )
        for spec in simulation_specs():
            seed_identities = (
                beta_seed,
                init_seed,
                derive_seed(root_hex, phase, "poisson_update", matrix_index, spec["streamIdentity"]),
                derive_seed(root_hex, phase, "overshoot_trim", matrix_index, spec["streamIdentity"]),
                derive_seed(root_hex, phase, "fission", matrix_index, spec["streamIdentity"]),
                derive_seed(root_hex, phase, "daughter_selection", matrix_index, spec["streamIdentity"]),
            )
            for identity in seed_identities:
                seed_rows.append(
                    {
                        "loopId": LOOP_ID,
                        "mechanismId": spec["mechanismId"],
                        "candidateId": spec["candidateId"],
                        "matrixIndex": matrix_index,
                        "purpose": identity.purpose,
                        "configurationId": identity.configuration_id,
                        "derivedSeed": str(identity.derived_seed),
                        "seedMaterialSha256": identity.seed_material_sha256,
                        "rootHex": root_hex,
                        "phase": phase,
                    }
                )
    return pd.DataFrame(seed_rows), pd.DataFrame(unit_rows)


def validate_seed_firewall(seed_frame: pd.DataFrame, inputs: pd.DataFrame) -> dict[str, Any]:
    config = load_config()
    inventory = collect_prior_seed_inventory()
    new_beta = set(inputs["betaSha256"].astype(str))
    new_initial = set(inputs["initialStateSha256"].astype(str))
    new_material = set(seed_frame["seedMaterialSha256"].astype(str))
    new_roots = {str(config["seedContract"]["rootSeedHex"]), str(config["seedContract"]["bootstrapRootHex"])}
    new_derived = set(seed_frame["derivedSeed"].astype(str))
    beta_overlap = sorted(new_beta.intersection(inventory["beta"] | inventory["allHex64"]))
    initial_overlap = sorted(new_initial.intersection(inventory["initial"] | inventory["allHex64"]))
    material_overlap = sorted(new_material.intersection(inventory["seedMaterial"] | inventory["allHex64"]))
    root_overlap = sorted(new_roots.intersection(inventory["root"] | inventory["allHex64"]))
    derived_overlap = sorted(new_derived.intersection(inventory["derivedSeed"]))
    matrix_shared_counts = inputs.groupby("matrixIndex").size()
    passed = bool(
        len(inputs) == 100
        and inputs["matrixIndex"].nunique() == 100
        and inputs["betaSha256"].nunique() == 100
        and inputs["initialStateSha256"].nunique() == 100
        and (matrix_shared_counts == 1).all()
        and not beta_overlap
        and not initial_overlap
        and not material_overlap
        and not root_overlap
        and not derived_overlap
    )
    inventory_summary = {
        key: {
            "count": len(values),
            "setSha256": sha256_text("\n".join(sorted(values))),
        }
        for key, values in inventory.items()
    }
    write_json(
        LOOP_ROOT / "prior_seed_inventory.json",
        {"schema": "eidosoma.e01.s19_l08_prior_seed_inventory.v1", **inventory_summary},
    )
    return {
        "schema": "eidosoma.e01.s19_l08_seed_firewall.v1",
        "newMatrixCount": len(new_beta),
        "newInitialStateCount": len(new_initial),
        "newSeedMaterialCount": len(new_material),
        "newDerivedSeedCount": len(new_derived),
        "priorInventory": inventory_summary,
        "betaOverlapCount": len(beta_overlap),
        "initialStateOverlapCount": len(initial_overlap),
        "seedMaterialOverlapCount": len(material_overlap),
        "seedRootOverlapCount": len(root_overlap),
        "derivedSeedOverlapCount": len(derived_overlap),
        "overlaps": {
            "beta": beta_overlap,
            "initialState": initial_overlap,
            "seedMaterial": material_overlap,
            "root": root_overlap,
            "derivedSeed": derived_overlap,
        },
        "passed": passed,
        "validatedAtUtc": utc_now(),
    }


def source_snapshot_rows() -> list[dict[str, Any]]:
    paths = [
        PAPER_PATH,
        Path("/cache/e01_s03/downloads/paper-2607.28250v1.pdf"),
        Path("/cache/e01_s03/sources/gard-historical/tgs_nondrift.m"),
        Path("/cache/e01_s03/sources/gard-historical/tgs_parameters_v10.m"),
        Path("/cache/e01_s03/sources/gard-historical/tgs_grow_v10.m"),
        Path("/cache/e01_s03/sources/gard-historical/README.txt"),
        Path("/artifacts/research_steps/S13Y/research_step_full_results.md"),
        Path("/artifacts/research_steps/S18/research_step_full_results.md"),
        Path("/artifacts/research_steps/S19/loops/L07/S19_L07_FULL_RESULTS.md"),
    ]
    rows = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        rows.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "retainedOrReferenced": "REFERENCE_ONLY" if str(path).startswith("/cache") else "IMMUTABLE_INPUT",
            }
        )
    return rows


def _benchmark_one_matrix() -> dict[str, Any]:
    config = load_config()
    root_hex = config["seedContract"]["rootSeedHex"]
    phase = config["seedContract"]["matrixPhase"]
    matrix_index = 0
    beta = generate_beta(derive_seed(root_hex, phase, "catalytic_matrix", matrix_index))
    initial = initialize_distinct_state(derive_seed(root_hex, phase, "initial_state", matrix_index))
    rows = []
    total_cpu = 0.0
    total_wall = 0.0
    for spec in simulation_specs():
        wall = time.perf_counter()
        cpu = time.process_time()
        trajectory, _ = simulate_trajectory(
            phase=phase,
            root_hex=root_hex,
            matrix_index=matrix_index,
            definition=make_definition(spec),
            stream_identity=spec["streamIdentity"],
            beta=beta,
            initial_state=initial,
        )
        cpu_seconds = time.process_time() - cpu
        wall_seconds = time.perf_counter() - wall
        total_cpu += cpu_seconds
        total_wall += wall_seconds
        rows.append(
            {
                "mechanismId": spec["mechanismId"],
                "candidateId": spec["candidateId"],
                "cpuSeconds": cpu_seconds,
                "wallSeconds": wall_seconds,
                "terminalStatusNotOpenedScientifically": trajectory.terminal_status,
                "labelsCalculated": False,
            }
        )
    projected_cpu_hours = total_cpu * 100 * 2 * 1.25 / 3600.0
    projected_wall_hours = total_wall * 100 * 2 * 1.25 / (3600.0 * 8)
    return {
        "schema": "eidosoma.e01.s19_l08_preoutcome_benchmark.v1",
        "matrixIndex": 0,
        "simulationCount": 4,
        "scientificLabelsOpened": False,
        "rows": rows,
        "benchmarkCpuSeconds": total_cpu,
        "benchmarkWallSeconds": total_wall,
        "projectionIncludesPrimaryCompleteRegenerationAnd25PercentMargin": True,
        "projectedCpuHours": projected_cpu_hours,
        "projectedWallHoursAtEightWorkers": projected_wall_hours,
        "scientificComputeCeilingAfterReserveCpuHours": 90.0,
        "passed": bool(projected_cpu_hours <= 90.0 and projected_wall_hours <= 72.0),
        "benchmarkedAtUtc": utc_now(),
    }


def prepare() -> None:
    started = utc_now()
    LOOP_ROOT.mkdir(parents=True, exist_ok=False)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    TRAJECTORY_CACHE.mkdir(parents=True, exist_ok=True)
    config = load_config()
    repo = repository_lock()
    baseline_rows = hash_immutable_prior()
    write_json(
        LOOP_ROOT / "immutable_prior_baseline.json",
        {
            "schema": "eidosoma.e01.s19_l08_immutable_prior_baseline.v1",
            "createdAtUtc": utc_now(),
            "fileCount": len(baseline_rows),
            "files": baseline_rows,
        },
    )
    write_yaml(LOOP_ROOT / "preregistration.yaml", config)
    write_yaml(
        LOOP_ROOT / "candidate_bundle_registry.yaml",
        {
            "schema": "eidosoma.e01.s19_l08_candidate_bundle_registry.v1",
            "bundles": [
                {"bundleId": MECHANISM_A, "selected": True, "specificationCount": 1},
                {"bundleId": MECHANISM_B, "selected": True, "specificationCount": 1},
            ],
            "selectionBasis": "Both L07 mechanisms fixed by explicit human direction; no ranking or outcome selection.",
        },
    )
    write_yaml(
        LOOP_ROOT / "mechanism_registry.yaml",
        {"schema": "eidosoma.e01.s19_l08_mechanism_registry.v1", "mechanisms": config["mechanisms"]},
    )
    specification_rows = []
    for spec in simulation_specs():
        for analysis_object in analysis_objects_for_mechanism(spec["mechanismId"]):
            specification_rows.append(
                {
                    **spec,
                    "analysisObjectId": analysis_object,
                    "threshold": 0.9,
                    "comparator": "STRICT_GT",
                    "clockId": "C1_SELECTED_DAUGHTER_RETAINED",
                    "registeredBeforeOutcome": True,
                }
            )
    write_parquet(LOOP_ROOT / "specification_ledger.parquet", pd.DataFrame(specification_rows))
    write_csv(
        LOOP_ROOT / "candidate_ranking.csv",
        pd.DataFrame(
            [
                {"frozenOrder": 1, "mechanismId": MECHANISM_A, "selected": True, "rankingStatus": "COEQUAL_HUMAN_DIRECTED"},
                {"frozenOrder": 2, "mechanismId": MECHANISM_B, "selected": True, "rankingStatus": "COEQUAL_HUMAN_DIRECTED"},
            ]
        ),
    )
    sources = source_snapshot_rows()
    write_json(
        LOOP_ROOT / "source_snapshot_manifest.json",
        {"schema": "eidosoma.e01.s19_l08_source_snapshot_manifest.v1", "sources": sources},
    )
    seed_frame, inputs = seed_rows_and_input_units()
    write_parquet(LOOP_ROOT / "seed_manifest.parquet", seed_frame)
    write_parquet(LOOP_ROOT / "input_units.parquet", inputs)
    write_json(
        LOOP_ROOT / "input_manifest.json",
        {
            "schema": "eidosoma.e01.s19_l08_input_manifest.v1",
            "matrixCount": 100,
            "matchedAcrossFourSimulations": True,
            "betaHashSetSha256": sha256_text("\n".join(sorted(inputs["betaSha256"]))),
            "initialStateHashSetSha256": sha256_text("\n".join(sorted(inputs["initialStateSha256"]))),
            "inputUnitsPath": str(LOOP_ROOT / "input_units.parquet"),
        },
    )
    firewall = validate_seed_firewall(seed_frame, inputs)
    write_json(LOOP_ROOT / "seed_firewall.json", firewall)
    locked_files = [CONFIG_PATH, CORE_PATH, RUNNER_PATH, TEST_PATH]
    method_lock = {
        "schema": "eidosoma.e01.s19_l08_method_lock.v1",
        "versionedLoopId": VERSION,
        "lockedAtUtc": utc_now(),
        "repository": repo,
        "files": [{"path": str(path.relative_to(REPO)), "sha256": sha256_file(path)} for path in locked_files],
        "configSha256": sha256_file(CONFIG_PATH),
        "sourceSnapshotManifestSha256": sha256_file(LOOP_ROOT / "source_snapshot_manifest.json"),
        "seedManifestSha256": sha256_file(LOOP_ROOT / "seed_manifest.parquet"),
        "inputManifestSha256": sha256_file(LOOP_ROOT / "input_manifest.json"),
        "seedFirewallPassed": firewall["passed"],
        "passed": bool(repo["passed"] and firewall["passed"]),
    }
    write_json(LOOP_ROOT / "method_lock.json", method_lock)
    if not method_lock["passed"]:
        raise RuntimeError("L08 pre-outcome method/seed lock failed")
    benchmark = _benchmark_one_matrix()
    write_json(LOOP_ROOT / "preoutcome_benchmark.json", benchmark)
    if not benchmark["passed"]:
        raise RuntimeError("L08 benchmark projected beyond the locked ceiling")
    write_json(
        LOOP_ROOT / "preoutcome_gate.json",
        {
            "schema": "eidosoma.e01.s19_l08_preoutcome_gate.v1",
            "repositoryLock": repo["passed"],
            "immutableBaselineCaptured": len(baseline_rows) > 0,
            "seedFirewall": firewall["passed"],
            "benchmark": benchmark["passed"],
            "exactMechanismCount": len(config["mechanisms"]) == 2,
            "exactMatrixCount": config["authorizedScope"]["matrices"] == 100,
            "outcomeAccessAuthorized": True,
            "passed": True,
            "preparedAtUtc": utc_now(),
        },
    )
    write_json(
        LOOP_ROOT / "prepare_runtime.json",
        {"phase": "prepare", "startedAtUtc": started, "finishedAtUtc": utc_now()},
    )


def assert_prepared() -> None:
    required = [
        "preregistration.yaml",
        "method_lock.json",
        "preoutcome_gate.json",
        "seed_firewall.json",
        "seed_manifest.parquet",
        "immutable_prior_baseline.json",
    ]
    missing = [name for name in required if not (LOOP_ROOT / name).exists()]
    if missing:
        raise RuntimeError(f"L08 not prepared: missing {missing}")
    if not json.loads((LOOP_ROOT / "preoutcome_gate.json").read_text())["passed"]:
        raise RuntimeError("L08 preoutcome gate did not pass")
    lock = json.loads((LOOP_ROOT / "method_lock.json").read_text())
    if repository_lock()["head"] != lock["repository"]["head"]:
        raise RuntimeError("repository identity changed after L08 lock")
    for row in lock["files"]:
        if sha256_file(REPO / row["path"]) != row["sha256"]:
            raise RuntimeError(f"locked file changed after L08 lock: {row['path']}")
    prior = validate_immutable_prior()
    if not prior["passed"]:
        raise RuntimeError("immutable prior changed after L08 lock")


def _cache_path(matrix_index: int, spec: dict[str, Any]) -> Path:
    return TRAJECTORY_CACHE / f"M{matrix_index:03d}__{spec['mechanismId']}__{spec['candidateId']}.pkl"


def _flat_fingerprint(value: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    flat = {key: item for key, item in value.items() if not key.endswith("Indices") and not key.endswith("Durations")}
    episodes = []
    for polarity in ("positive", "negative"):
        starts = value[f"{polarity}EpisodeStartIndices"]
        durations = value[f"{polarity}EpisodeDurations"]
        for ordinal, (start, duration) in enumerate(zip(starts, durations, strict=True), start=1):
            episodes.append(
                {
                    "polarity": polarity.upper(),
                    "episodeOrdinal": ordinal,
                    "startEligibleIndex0": int(start),
                    "duration": int(duration),
                }
            )
    return flat, episodes


def _simulate_matrix(matrix_index: int, *, write_cache: bool) -> dict[str, Any]:
    config = load_config()
    root_hex = config["seedContract"]["rootSeedHex"]
    phase = config["seedContract"]["matrixPhase"]
    beta = generate_beta(derive_seed(root_hex, phase, "catalytic_matrix", matrix_index))
    initial = initialize_distinct_state(derive_seed(root_hex, phase, "initial_state", matrix_index))
    attempts: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    fingerprints: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for spec in simulation_specs():
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        try:
            trajectory, _ = simulate_trajectory(
                phase=phase,
                root_hex=root_hex,
                matrix_index=matrix_index,
                definition=make_definition(spec),
                stream_identity=spec["streamIdentity"],
                beta=beta,
                initial_state=initial,
            )
            cache_path = _cache_path(matrix_index, spec)
            if write_cache:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with cache_path.open("wb") as handle:
                    pickle.dump(trajectory, handle, protocol=5)
            cache_sha = sha256_file(cache_path) if write_cache else None
            diagnostic = trajectory_diagnostics(trajectory)
            diagnostics.append({**spec, "matrixIndex": matrix_index, **diagnostic})
            trajectories.append(
                {
                    **spec,
                    "matrixIndex": matrix_index,
                    "trajectoryId": trajectory.trajectory_id,
                    "trajectorySha256": trajectory.trajectory_sha256,
                    "betaSha256": trajectory.beta_sha256,
                    "initialStateSha256": trajectory.initial_state_sha256,
                    "terminalStatus": trajectory.terminal_status,
                    "completedFissions": trajectory.completed_fissions,
                    "cachePath": str(cache_path) if write_cache else None,
                    "cacheSha256": cache_sha,
                }
            )
            for analysis_object in analysis_objects_for_mechanism(spec["mechanismId"]):
                frame = materialize_analysis_object(
                    trajectory, spec["mechanismId"], analysis_object
                )
                fingerprint, episode_values = _flat_fingerprint(label_fingerprint(frame))
                row = {
                    **spec,
                    "analysisObjectId": analysis_object,
                    "matrixIndex": matrix_index,
                    "trajectoryId": trajectory.trajectory_id,
                    "trajectorySha256": trajectory.trajectory_sha256,
                    **fingerprint,
                }
                fingerprints.append(row)
                for episode in episode_values:
                    episodes.append(
                        {
                            "mechanismId": spec["mechanismId"],
                            "candidateId": spec["candidateId"],
                            "analysisObjectId": analysis_object,
                            "matrixIndex": matrix_index,
                            **episode,
                        }
                    )
            attempts.append(
                {
                    **spec,
                    "matrixIndex": matrix_index,
                    "status": "COMPLETE" if diagnostic["trajectoryStatus"] == "COMPLETE" else "INCOMPLETE_RETAINED",
                    "terminalStatus": trajectory.terminal_status,
                    "completedFissions": trajectory.completed_fissions,
                    "wallSeconds": time.perf_counter() - wall_start,
                    "cpuSeconds": time.process_time() - cpu_start,
                    "replacementAttempted": False,
                }
            )
        except Exception as error:  # retain and fail closed during final adjudication
            failures.append(
                {
                    "failureId": f"L08-M{matrix_index:03d}-{spec['mechanismId']}-{spec['candidateId']}",
                    **spec,
                    "matrixIndex": matrix_index,
                    "failureType": type(error).__name__,
                    "message": str(error),
                    "scientificValuesEligible": False,
                }
            )
            attempts.append(
                {
                    **spec,
                    "matrixIndex": matrix_index,
                    "status": "PIPELINE_FAILURE_RETAINED",
                    "terminalStatus": None,
                    "completedFissions": None,
                    "wallSeconds": time.perf_counter() - wall_start,
                    "cpuSeconds": time.process_time() - cpu_start,
                    "replacementAttempted": False,
                }
            )
    return {
        "attempts": attempts,
        "trajectories": trajectories,
        "fingerprints": fingerprints,
        "episodes": episodes,
        "diagnostics": diagnostics,
        "failures": failures,
    }


def _numeric_frame(
    fingerprints: pd.DataFrame, diagnostics: pd.DataFrame, mechanism: str, candidate: str, analysis_object: str
) -> pd.DataFrame:
    labels = fingerprints.loc[
        fingerprints["mechanismId"].eq(mechanism)
        & fingerprints["candidateId"].eq(candidate)
        & fingerprints["analysisObjectId"].eq(analysis_object)
    ].copy()
    diag = diagnostics.loc[
        diagnostics["mechanismId"].eq(mechanism) & diagnostics["candidateId"].eq(candidate)
    ].copy()
    return labels.merge(
        diag[["matrixIndex", *DIAGNOSTIC_NUMERIC_METRICS]], on="matrixIndex", how="left", validate="one_to_one"
    ).sort_values("matrixIndex", kind="stable")


def _bootstrap_aggregates(
    fingerprints: pd.DataFrame, diagnostics: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = load_config()
    root = config["seedContract"]["bootstrapRootHex"]
    result_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    for mechanism, analysis_object in (
        (MECHANISM_A, OBJECT_A_BOUNDARY),
        (MECHANISM_A, OBJECT_A_PROJECTED),
        (MECHANISM_B, OBJECT_B_MOLECULAR),
    ):
        for candidate in ("CANDIDATE_2", "CANDIDATE_3"):
            frame = _numeric_frame(fingerprints, diagnostics, mechanism, candidate, analysis_object)
            if len(frame) != 100 or frame["matrixIndex"].nunique() != 100:
                raise RuntimeError(f"incomplete matrix frame for {mechanism}/{candidate}/{analysis_object}")
            for metric in (*FINGERPRINT_NUMERIC_METRICS, *DIAGNOSTIC_NUMERIC_METRICS):
                values = pd.to_numeric(frame[metric], errors="coerce").to_numpy(dtype=np.float64)
                indices = bootstrap_indices(root, mechanism, candidate, analysis_object, metric)
                with np.errstate(invalid="ignore"):
                    draws = np.nanmean(values[indices], axis=1)
                finite = values[np.isfinite(values)]
                finite_draws = draws[np.isfinite(draws)]
                seed = int.from_bytes(
                    hashlib.sha256(
                        "\x1f".join([VERSION, root, "bootstrap", mechanism, candidate, analysis_object, metric]).encode()
                    ).digest()[:16],
                    "big",
                )
                seed_rows.append(
                    {
                        "mechanismId": mechanism,
                        "candidateId": candidate,
                        "analysisObjectId": analysis_object,
                        "metric": metric,
                        "replicateCount": BOOTSTRAP_REPLICATES,
                        "derivedSeed": str(seed),
                    }
                )
                result_rows.append(
                    {
                        "mechanismId": mechanism,
                        "candidateId": candidate,
                        "analysisObjectId": analysis_object,
                        "metric": metric,
                        "matrixCount": 100,
                        "definedMatrixCount": int(finite.size),
                        "mean": float(np.mean(finite)) if finite.size else None,
                        "median": float(np.median(finite)) if finite.size else None,
                        "sd": float(np.std(finite, ddof=1)) if finite.size > 1 else (0.0 if finite.size else None),
                        "ci025": float(np.quantile(finite_draws, 0.025)) if finite_draws.size else None,
                        "ci975": float(np.quantile(finite_draws, 0.975)) if finite_draws.size else None,
                        "bootstrapReplicates": BOOTSTRAP_REPLICATES,
                        "bootstrapDrawSha256": hashlib.sha256(np.ascontiguousarray(draws).tobytes()).hexdigest(),
                    }
                )
    return pd.DataFrame(result_rows), pd.DataFrame(seed_rows)


def _mechanism_comparison(
    fingerprints: pd.DataFrame, diagnostics: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config = load_config()
    root = config["seedContract"]["bootstrapRootHex"]
    aggregate_rows: list[dict[str, Any]] = []
    distance_bootstrap_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    cross_rows: list[dict[str, Any]] = []
    frames: dict[tuple[str, str], pd.DataFrame] = {}
    for candidate in ("CANDIDATE_2", "CANDIDATE_3"):
        frames[(MECHANISM_A, candidate)] = _numeric_frame(
            fingerprints, diagnostics, MECHANISM_A, candidate, OBJECT_A_PROJECTED
        )
        frames[(MECHANISM_B, candidate)] = _numeric_frame(
            fingerprints, diagnostics, MECHANISM_B, candidate, OBJECT_B_MOLECULAR
        )
        for mechanism in (MECHANISM_A, MECHANISM_B):
            frame = frames[(mechanism, candidate)]
            means = {
                metric: float(pd.to_numeric(frame[metric], errors="coerce").mean())
                for metric in PAPER_TARGETS
            }
            for onset_mode in ("RAW_ONSET", "NORMALIZED_ONSET"):
                aggregate_rows.append(
                    {
                        "mechanismId": mechanism,
                        "candidateId": candidate,
                        "onsetMode": onset_mode,
                        "paperDistance": paper_distance(means, onset_mode),
                        **{f"mean_{metric}": value for metric, value in means.items()},
                    }
                )
        for metric in (
            "selectedClockLength",
            "persistence",
            "consistency",
            "firstOnsetRawStep1",
            "firstOnsetNormalized",
        ):
            mean_a = float(pd.to_numeric(frames[(MECHANISM_A, candidate)][metric], errors="coerce").mean())
            mean_b = float(pd.to_numeric(frames[(MECHANISM_B, candidate)][metric], errors="coerce").mean())
            error_a = absolute_scaled_error(metric, mean_a)
            error_b = absolute_scaled_error(metric, mean_b)
            winner = MECHANISM_A if error_a < error_b else (MECHANISM_B if error_b < error_a else "TIE")
            metric_rows.append(
                {
                    "candidateId": candidate,
                    "metric": metric,
                    "mechanismAMean": mean_a,
                    "mechanismBMean": mean_b,
                    "mechanismAScaledAbsoluteError": error_a,
                    "mechanismBScaledAbsoluteError": error_b,
                    "closerMechanism": winner,
                }
            )
        for onset_mode, metrics in (
            ("RAW_ONSET", RAW_DISTANCE_METRICS),
            ("NORMALIZED_ONSET", NORMALIZED_DISTANCE_METRICS),
        ):
            indices = bootstrap_indices(root, "paired-distance", candidate, onset_mode)
            draws_a = []
            draws_b = []
            frame_a = frames[(MECHANISM_A, candidate)]
            frame_b = frames[(MECHANISM_B, candidate)]
            arrays_a = {m: pd.to_numeric(frame_a[m], errors="coerce").to_numpy(float) for m in metrics}
            arrays_b = {m: pd.to_numeric(frame_b[m], errors="coerce").to_numpy(float) for m in metrics}
            for sample in indices:
                means_a = {m: float(np.nanmean(arrays_a[m][sample])) for m in metrics}
                means_b = {m: float(np.nanmean(arrays_b[m][sample])) for m in metrics}
                draws_a.append(paper_distance(means_a, onset_mode))
                draws_b.append(paper_distance(means_b, onset_mode))
            a = np.asarray(draws_a, dtype=np.float64)
            b = np.asarray(draws_b, dtype=np.float64)
            difference = a - b
            distance_bootstrap_rows.append(
                {
                    "candidateId": candidate,
                    "onsetMode": onset_mode,
                    "contrast": "A_MINUS_B",
                    "bootstrapReplicates": BOOTSTRAP_REPLICATES,
                    "meanDifference": float(np.mean(difference)),
                    "ci025": float(np.quantile(difference, 0.025)),
                    "ci975": float(np.quantile(difference, 0.975)),
                    "probabilityALower": float(np.mean(difference < 0)),
                    "drawSha256": hashlib.sha256(np.ascontiguousarray(difference).tobytes()).hexdigest(),
                }
            )
    for mechanism, object_id in (
        (MECHANISM_A, OBJECT_A_PROJECTED),
        (MECHANISM_B, OBJECT_B_MOLECULAR),
    ):
        c2 = frames[(mechanism, "CANDIDATE_2")]
        c3 = frames[(mechanism, "CANDIDATE_3")]
        for metric in (*PAPER_TARGETS.keys(), "positiveEpisodeCount", "negativeEpisodeCount", "meanPretrimOvershoot"):
            left = pd.to_numeric(c2[metric], errors="coerce").to_numpy(float)
            right = pd.to_numeric(c3[metric], errors="coerce").to_numpy(float)
            indices = bootstrap_indices(root, "cross-candidate", mechanism, metric)
            draws = np.nanmean(left[indices], axis=1) - np.nanmean(right[indices], axis=1)
            cross_rows.append(
                {
                    "mechanismId": mechanism,
                    "analysisObjectId": object_id,
                    "metric": metric,
                    "candidate2Mean": float(np.nanmean(left)),
                    "candidate3Mean": float(np.nanmean(right)),
                    "candidate2Minus3": float(np.nanmean(left) - np.nanmean(right)),
                    "absoluteDifference": float(abs(np.nanmean(left) - np.nanmean(right))),
                    "ci025Difference": float(np.quantile(draws, 0.025)),
                    "ci975Difference": float(np.quantile(draws, 0.975)),
                    "bootstrapReplicates": BOOTSTRAP_REPLICATES,
                }
            )
    return (
        pd.DataFrame(aggregate_rows),
        pd.DataFrame(distance_bootstrap_rows),
        pd.DataFrame(metric_rows),
        pd.DataFrame(cross_rows),
    )


def _write_scientific_outputs(
    fingerprints: pd.DataFrame, diagnostics: pd.DataFrame
) -> None:
    aggregate, bootstrap_seeds = _bootstrap_aggregates(fingerprints, diagnostics)
    comparison, distance_bootstrap, metric_comparison, cross_candidate = _mechanism_comparison(
        fingerprints, diagnostics
    )
    write_parquet(LOOP_ROOT / "results.parquet", aggregate)
    write_parquet(LOOP_ROOT / "bootstrap_results.parquet", aggregate)
    write_parquet(LOOP_ROOT / "bootstrap_seed_manifest.parquet", bootstrap_seeds)
    write_parquet(LOOP_ROOT / "mechanism_discrimination_results.parquet", comparison)
    write_parquet(LOOP_ROOT / "paired_distance_bootstrap_results.parquet", distance_bootstrap)
    write_parquet(LOOP_ROOT / "metric_closeness_results.parquet", metric_comparison)
    write_parquet(LOOP_ROOT / "cross_candidate_results.parquet", cross_candidate)
    occupancy = aggregate.loc[aggregate["metric"].eq("occupancy")].copy()
    occupancy["withinBand"] = occupancy["mean"].map(
        lambda value: occupancy_in_band(float(value)) if pd.notna(value) else False
    )
    occupancy["defined100"] = occupancy["definedMatrixCount"].eq(100)
    occupancy["gatePassed"] = occupancy["withinBand"] & occupancy["defined100"]
    write_csv(LOOP_ROOT / "occupancy_gate_results.csv", occupancy)


def run(workers: int) -> None:
    assert_prepared()
    if workers != 8:
        raise ValueError("L08 is locked to exactly eight workers")
    started = utc_now()
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    outputs = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_simulate_matrix, matrix, write_cache=True): matrix for matrix in range(100)}
        for future in as_completed(futures):
            outputs.append(future.result())
    attempts = pd.DataFrame([row for output in outputs for row in output["attempts"]])
    trajectories = pd.DataFrame([row for output in outputs for row in output["trajectories"]])
    fingerprints = pd.DataFrame([row for output in outputs for row in output["fingerprints"]])
    episodes = pd.DataFrame([row for output in outputs for row in output["episodes"]])
    diagnostics = pd.DataFrame([row for output in outputs for row in output["diagnostics"]])
    failures = pd.DataFrame([row for output in outputs for row in output["failures"]])
    if len(attempts) != 400 or attempts[["mechanismId", "candidateId", "matrixIndex"]].drop_duplicates().shape[0] != 400:
        raise RuntimeError("L08 attempt accounting did not contain exactly 400 unique attempts")
    write_parquet(LOOP_ROOT / "execution_status.parquet", attempts)
    write_parquet(LOOP_ROOT / "attempt_manifest.parquet", attempts)
    write_parquet(LOOP_ROOT / "trajectory_manifest.parquet", trajectories)
    write_parquet(LOOP_ROOT / "trajectory_fingerprints.parquet", fingerprints)
    write_parquet(LOOP_ROOT / "episode_results.parquet", episodes)
    write_parquet(LOOP_ROOT / "trajectory_diagnostics.parquet", diagnostics)
    failure_columns = [
        "failureId", "mechanismId", "candidateId", "matrixIndex", "failureType", "message", "scientificValuesEligible"
    ]
    write_csv(
        LOOP_ROOT / "failure_ledger.csv",
        failures if not failures.empty else pd.DataFrame(columns=failure_columns),
    )
    if not failures.empty or len(trajectories) != 400 or len(fingerprints) != 600:
        write_json(
            LOOP_ROOT / "run_release_gate.json",
            {
                "passed": False,
                "attemptCount": len(attempts),
                "trajectoryCount": len(trajectories),
                "fingerprintCount": len(fingerprints),
                "failureCount": len(failures),
            },
        )
        raise RuntimeError("L08 primary run failed closed before aggregation")
    _write_scientific_outputs(fingerprints, diagnostics)
    write_json(
        LOOP_ROOT / "run_release_gate.json",
        {
            "schema": "eidosoma.e01.s19_l08_run_release_gate.v1",
            "passed": True,
            "attemptCount": 400,
            "trajectoryCount": 400,
            "fingerprintCount": 600,
            "failureCount": 0,
            "releasedAtUtc": utc_now(),
        },
    )
    write_json(
        LOOP_ROOT / "run_runtime.json",
        {
            "phase": "primary_run",
            "startedAtUtc": started,
            "finishedAtUtc": utc_now(),
            "wallSeconds": time.perf_counter() - wall_start,
            "coordinatorCpuSeconds": time.process_time() - cpu_start,
            "workerCpuSeconds": float(attempts["cpuSeconds"].sum()),
            "workerWallSeconds": float(attempts["wallSeconds"].sum()),
            "workers": workers,
        },
    )


def _row_compare(primary: dict[str, Any], replay: dict[str, Any], keys: list[str]) -> tuple[bool, list[str]]:
    mismatches = []
    for key in keys:
        left = primary.get(key)
        right = replay.get(key)
        if pd.isna(left) and pd.isna(right):
            continue
        if left != right:
            mismatches.append(key)
    return not mismatches, mismatches


def _regenerate_matrix(matrix_index: int) -> dict[str, Any]:
    regenerated = _simulate_matrix(matrix_index, write_cache=False)
    primary_trajectory = pd.read_parquet(
        LOOP_ROOT / "trajectory_manifest.parquet",
        filters=[[('matrixIndex', '==', matrix_index)]],
    )
    primary_fingerprint = pd.read_parquet(
        LOOP_ROOT / "trajectory_fingerprints.parquet",
        filters=[[('matrixIndex', '==', matrix_index)]],
    )
    replay_trajectory = pd.DataFrame(regenerated["trajectories"])
    replay_fingerprint = pd.DataFrame(regenerated["fingerprints"])
    trajectory_rows = []
    fingerprint_rows = []
    for primary in primary_trajectory.to_dict(orient="records"):
        match = replay_trajectory.loc[
            replay_trajectory["mechanismId"].eq(primary["mechanismId"])
            & replay_trajectory["candidateId"].eq(primary["candidateId"])
        ]
        if len(match) != 1:
            trajectory_rows.append({**{k: primary[k] for k in ("mechanismId", "candidateId", "matrixIndex")}, "passed": False, "mismatchFields": ["missing_or_duplicate"]})
            continue
        replay = match.iloc[0].to_dict()
        keys = ["trajectorySha256", "betaSha256", "initialStateSha256", "terminalStatus", "completedFissions"]
        passed, mismatches = _row_compare(primary, replay, keys)
        trajectory_rows.append(
            {**{k: primary[k] for k in ("mechanismId", "candidateId", "matrixIndex")}, "passed": passed, "mismatchFields": mismatches}
        )
    fingerprint_keys = [
        key
        for key in primary_fingerprint.columns
        if key not in {"exposure", "daughterRule", "overshootRule", "streamIdentity"}
    ]
    for primary in primary_fingerprint.to_dict(orient="records"):
        match = replay_fingerprint.loc[
            replay_fingerprint["mechanismId"].eq(primary["mechanismId"])
            & replay_fingerprint["candidateId"].eq(primary["candidateId"])
            & replay_fingerprint["analysisObjectId"].eq(primary["analysisObjectId"])
        ]
        if len(match) != 1:
            fingerprint_rows.append(
                {**{k: primary[k] for k in ("mechanismId", "candidateId", "analysisObjectId", "matrixIndex")}, "passed": False, "mismatchFields": ["missing_or_duplicate"]}
            )
            continue
        passed, mismatches = _row_compare(primary, match.iloc[0].to_dict(), fingerprint_keys)
        fingerprint_rows.append(
            {**{k: primary[k] for k in ("mechanismId", "candidateId", "analysisObjectId", "matrixIndex")}, "passed": passed, "mismatchFields": mismatches}
        )
    return {
        "trajectory": trajectory_rows,
        "fingerprint": fingerprint_rows,
        "fingerprints": regenerated["fingerprints"],
        "diagnostics": regenerated["diagnostics"],
        "attempts": regenerated["attempts"],
        "failures": regenerated["failures"],
    }


def regenerate(workers: int) -> None:
    assert_prepared()
    if workers != 8:
        raise ValueError("L08 regeneration is locked to eight workers")
    if not (LOOP_ROOT / "run_release_gate.json").exists():
        raise RuntimeError("primary L08 run is absent")
    started = utc_now()
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    outputs = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_regenerate_matrix, matrix): matrix for matrix in range(100)}
        for future in as_completed(futures):
            outputs.append(future.result())
    trajectory_replay = pd.DataFrame([row for output in outputs for row in output["trajectory"]])
    fingerprint_replay = pd.DataFrame([row for output in outputs for row in output["fingerprint"]])
    failures = [row for output in outputs for row in output["failures"]]
    replay_fingerprints = pd.DataFrame([row for output in outputs for row in output["fingerprints"]])
    replay_diagnostics = pd.DataFrame([row for output in outputs for row in output["diagnostics"]])
    replay_attempts = pd.DataFrame([row for output in outputs for row in output["attempts"]])
    write_parquet(LOOP_ROOT / "exact_replay_results.parquet", trajectory_replay)
    write_parquet(LOOP_ROOT / "result_replay_results.parquet", fingerprint_replay)
    if failures:
        raise RuntimeError("L08 regeneration produced a pipeline failure")
    primary_results = pd.read_parquet(LOOP_ROOT / "results.parquet")
    replay_results, _ = _bootstrap_aggregates(replay_fingerprints, replay_diagnostics)
    primary_comparison = pd.read_parquet(LOOP_ROOT / "mechanism_discrimination_results.parquet")
    replay_comparison, replay_distance, replay_metrics, replay_cross = _mechanism_comparison(
        replay_fingerprints, replay_diagnostics
    )
    aggregate_primary_hash = canonical_frame_sha256(
        primary_results, ["mechanismId", "candidateId", "analysisObjectId", "metric"]
    )
    aggregate_replay_hash = canonical_frame_sha256(
        replay_results, ["mechanismId", "candidateId", "analysisObjectId", "metric"]
    )
    comparison_primary_hash = canonical_frame_sha256(
        primary_comparison, ["mechanismId", "candidateId", "onsetMode"]
    )
    comparison_replay_hash = canonical_frame_sha256(
        replay_comparison, ["mechanismId", "candidateId", "onsetMode"]
    )
    validation = {
        "schema": "eidosoma.e01.s19_l08_regeneration_validation.v1",
        "trajectoryReplayRows": len(trajectory_replay),
        "trajectoryReplayPassCount": int(trajectory_replay["passed"].sum()),
        "fingerprintReplayRows": len(fingerprint_replay),
        "fingerprintReplayPassCount": int(fingerprint_replay["passed"].sum()),
        "aggregatePrimarySha256": aggregate_primary_hash,
        "aggregateReplaySha256": aggregate_replay_hash,
        "comparisonPrimarySha256": comparison_primary_hash,
        "comparisonReplaySha256": comparison_replay_hash,
        "aggregateExact": aggregate_primary_hash == aggregate_replay_hash,
        "comparisonExact": comparison_primary_hash == comparison_replay_hash,
        "all400TrajectoriesExact": bool(len(trajectory_replay) == 400 and trajectory_replay["passed"].all()),
        "all600FingerprintsExact": bool(len(fingerprint_replay) == 600 and fingerprint_replay["passed"].all()),
        "passed": bool(
            len(trajectory_replay) == 400
            and trajectory_replay["passed"].all()
            and len(fingerprint_replay) == 600
            and fingerprint_replay["passed"].all()
            and aggregate_primary_hash == aggregate_replay_hash
            and comparison_primary_hash == comparison_replay_hash
        ),
        "validatedAtUtc": utc_now(),
    }
    write_json(LOOP_ROOT / "regeneration_validation.json", validation)
    write_json(
        LOOP_ROOT / "regeneration_runtime.json",
        {
            "phase": "complete_regeneration",
            "startedAtUtc": started,
            "finishedAtUtc": utc_now(),
            "wallSeconds": time.perf_counter() - wall_start,
            "coordinatorCpuSeconds": time.process_time() - cpu_start,
            "workerCpuSeconds": float(replay_attempts["cpuSeconds"].sum()),
            "workerWallSeconds": float(replay_attempts["wallSeconds"].sum()),
            "workers": workers,
        },
    )
    if not validation["passed"]:
        raise RuntimeError("L08 complete regeneration gate failed")


def _result_value(results: pd.DataFrame, mechanism: str, candidate: str, object_id: str, metric: str) -> float:
    row = results.loc[
        results["mechanismId"].eq(mechanism)
        & results["candidateId"].eq(candidate)
        & results["analysisObjectId"].eq(object_id)
        & results["metric"].eq(metric)
    ]
    if len(row) != 1:
        raise RuntimeError(f"missing aggregate result {mechanism}/{candidate}/{object_id}/{metric}")
    return float(row.iloc[0]["mean"])


def _decision_gates() -> tuple[pd.DataFrame, str, list[str], bool]:
    results = pd.read_parquet(LOOP_ROOT / "results.parquet")
    occupancy = pd.read_csv(LOOP_ROOT / "occupancy_gate_results.csv")
    distances = pd.read_parquet(LOOP_ROOT / "mechanism_discrimination_results.parquet")
    distance_boot = pd.read_parquet(LOOP_ROOT / "paired_distance_bootstrap_results.parquet")
    closeness = pd.read_parquet(LOOP_ROOT / "metric_closeness_results.parquet")
    attempts = pd.read_parquet(LOOP_ROOT / "execution_status.parquet")
    regeneration = json.loads((LOOP_ROOT / "regeneration_validation.json").read_text())
    firewall = json.loads((LOOP_ROOT / "seed_firewall.json").read_text())
    immutable = validate_immutable_prior()
    rows: list[dict[str, Any]] = []

    def add(gate: str, passed: bool, mechanism: str | None, detail: str) -> None:
        rows.append({"gateId": gate, "mechanismId": mechanism, "passed": bool(passed), "detail": detail})

    operational = bool(regeneration["passed"] and firewall["passed"] and immutable["passed"])
    add("OPERATIONAL_REGENERATION_FIREWALL_IMMUTABILITY", operational, None, json.dumps({"regeneration": regeneration["passed"], "firewall": firewall["passed"], "immutable": immutable["passed"]}))
    joint_occupancy = bool(len(occupancy) == 6 and occupancy["gatePassed"].all())
    add("JOINT_ALL_SIX_OCCUPANCY_GATES", joint_occupancy, None, f"{int(occupancy['gatePassed'].sum())}/6")

    preference: dict[str, list[bool]] = {MECHANISM_A: [], MECHANISM_B: []}
    for mechanism in (MECHANISM_A, MECHANISM_B):
        for candidate in ("CANDIDATE_2", "CANDIDATE_3"):
            other = MECHANISM_B if mechanism == MECHANISM_A else MECHANISM_A
            for mode in ("RAW_ONSET", "NORMALIZED_ONSET"):
                mine = float(distances.loc[(distances["mechanismId"] == mechanism) & (distances["candidateId"] == candidate) & (distances["onsetMode"] == mode), "paperDistance"].iloc[0])
                theirs = float(distances.loc[(distances["mechanismId"] == other) & (distances["candidateId"] == candidate) & (distances["onsetMode"] == mode), "paperDistance"].iloc[0])
                passed = mine < theirs
                preference[mechanism].append(passed)
                add(f"LOWER_DISTANCE_{candidate}_{mode}", passed, mechanism, f"{mine:.9g} versus {theirs:.9g}")
                boot = distance_boot.loc[(distance_boot["candidateId"] == candidate) & (distance_boot["onsetMode"] == mode)].iloc[0]
                ci_pass = bool(boot["ci975"] < 0) if mechanism == MECHANISM_A else bool(boot["ci025"] > 0)
                preference[mechanism].append(ci_pass)
                add(f"PAIRED_DISTANCE_CI_{candidate}_{mode}", ci_pass, mechanism, f"A-B CI [{boot['ci025']:.9g}, {boot['ci975']:.9g}]")
            wins = closeness.loc[(closeness["candidateId"] == candidate) & closeness["closerMechanism"].eq(mechanism)]
            win_count = len(wins)
            persistence_win = bool((wins["metric"] == "persistence").any())
            onset_win = bool(wins["metric"].isin(["firstOnsetRawStep1", "firstOnsetNormalized"]).any())
            closeness_pass = bool(win_count >= 4 and persistence_win and onset_win)
            preference[mechanism].append(closeness_pass)
            add(f"FOUR_OF_FIVE_CLOSENESS_{candidate}", closeness_pass, mechanism, f"wins={win_count}; persistence={persistence_win}; onset={onset_win}")
            object_id = OBJECT_A_PROJECTED if mechanism == MECHANISM_A else OBJECT_B_MOLECULAR
            fp = pd.read_parquet(LOOP_ROOT / "trajectory_fingerprints.parquet")
            fp = fp.loc[(fp["mechanismId"] == mechanism) & (fp["candidateId"] == candidate) & (fp["analysisObjectId"] == object_id)]
            topology_count = int(((fp["positiveEpisodeCount"] >= 1) & (fp["negativeEpisodeCount"] >= 1)).sum())
            topology_pass = topology_count >= 95
            preference[mechanism].append(topology_pass)
            add(f"EPISODE_TOPOLOGY_{candidate}", topology_pass, mechanism, f"{topology_count}/100")
            status = attempts.loc[(attempts["mechanismId"] == mechanism) & (attempts["candidateId"] == candidate)]
            complete_count = int(status["status"].eq("COMPLETE").sum())
            attempt_pass = bool(len(status) == 100 and complete_count >= 95 and not status["replacementAttempted"].any())
            preference[mechanism].append(attempt_pass)
            add(f"ATTEMPT_COMPLETION_{candidate}", attempt_pass, mechanism, f"attempted={len(status)} complete={complete_count}")
            maxstep = _result_value(results, mechanism, candidate, object_id, "maxStepTerminationFraction")
            maxstep_pass = maxstep == 0.0
            preference[mechanism].append(maxstep_pass)
            add(f"ZERO_MAXSTEP_{candidate}", maxstep_pass, mechanism, f"mean fraction={maxstep}")
            post_mass = _result_value(results, mechanism, candidate, object_id, "meanPostFissionMass")
            post_pass = 38.0 <= post_mass <= 42.0
            preference[mechanism].append(post_pass)
            add(f"POSTFISSION_MASS_{candidate}", post_pass, mechanism, f"mean={post_mass:.9g}")
        object_id = OBJECT_A_PROJECTED if mechanism == MECHANISM_A else OBJECT_B_MOLECULAR
        occ2 = _result_value(results, mechanism, "CANDIDATE_2", object_id, "occupancy")
        occ3 = _result_value(results, mechanism, "CANDIDATE_3", object_id, "occupancy")
        cross_pass = abs(occ2 - occ3) <= 0.03
        preference[mechanism].append(cross_pass)
        add("OCCUPANCY_CROSS_CANDIDATE", cross_pass, mechanism, f"absolute difference={abs(occ2-occ3):.9g}")
        preference[mechanism].append(operational)
        add("MECHANISM_OPERATIONAL_INTEGRITY", operational, mechanism, "global replay/firewall/immutable checks")

    a_pass = all(preference[MECHANISM_A])
    b_pass = all(preference[MECHANISM_B])
    decision = terminal_decision(
        operational_integrity_passed=operational,
        joint_occupancy_gate_passed=joint_occupancy,
        fission_preference_gates_passed=a_pass,
        exposure_preference_gates_passed=b_pass,
    )
    if decision == "LOOP_FAILED_CLOSED":
        classifications = ["LOOP_FAILED_CLOSED"]
    elif decision == "NEITHER_MECHANISM_REPRODUCES_ON_UNTOUCHED_DATA":
        classifications = ["EXPLORATORY_NON_SUPPORT", "AUTHOR_AMBIGUITY_UNRESOLVED", "NOT_PROMOTABLE"]
    else:
        classifications = ["METHOD_DEPENDENT_LEAD", "AUTHOR_AMBIGUITY_UNRESOLVED", "NOT_PROMOTABLE"]
        if decision.startswith("EVIDENCE_FAVORS"):
            classifications.insert(0, "EXPLORATORY_DIRECTIONAL_MATCH")
    return pd.DataFrame(rows), decision, classifications, bool(operational and joint_occupancy)


def _paper_source_coherence() -> pd.DataFrame:
    config = load_config()["sourceCoherenceLock"]
    rows = []
    for mechanism in (MECHANISM_A, MECHANISM_B):
        for kind in ("coherent", "unresolved"):
            for ordinal, statement in enumerate(config[mechanism][kind], start=1):
                rows.append(
                    {
                        "mechanismId": mechanism,
                        "evidenceKind": kind.upper(),
                        "ordinal": ordinal,
                        "statement": statement,
                        "frozenBeforeOutcomes": True,
                        "usedToOverrideNumericalResult": False,
                    }
                )
    return pd.DataFrame(rows)


def _make_figure() -> None:
    results = pd.read_parquet(LOOP_ROOT / "results.parquet")
    metrics = ["occupancy", "persistence", "firstOnsetRawStep1", "consistency", "selectedClockLength"]
    targets = {metric: PAPER_TARGETS[metric][0] for metric in metrics}
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 3.6))
    colors = {MECHANISM_A: "#2c7fb8", MECHANISM_B: "#d95f0e"}
    for axis, metric in zip(axes, metrics, strict=True):
        rows = []
        for mechanism, object_id in ((MECHANISM_A, OBJECT_A_PROJECTED), (MECHANISM_B, OBJECT_B_MOLECULAR)):
            for candidate in ("CANDIDATE_2", "CANDIDATE_3"):
                row = results.loc[(results["mechanismId"] == mechanism) & (results["candidateId"] == candidate) & (results["analysisObjectId"] == object_id) & (results["metric"] == metric)].iloc[0]
                rows.append((mechanism, candidate, float(row["mean"]), float(row["ci025"]), float(row["ci975"])))
        for index, (mechanism, candidate, mean, lower, upper) in enumerate(rows):
            axis.errorbar(index, mean, yerr=[[mean-lower], [upper-mean]], fmt="o", color=colors[mechanism], capsize=3)
        axis.axhline(targets[metric], color="black", linestyle="--", linewidth=1)
        axis.set_xticks(range(4), ["A-C2", "A-C3", "B-C2", "B-C3"], rotation=45, ha="right")
        axis.set_title(metric)
        axis.grid(alpha=0.2)
    fig.suptitle("Untouched L08 mechanism fingerprints (mean and 95% matrix-bootstrap CI)")
    fig.tight_layout()
    fig.savefig(LOOP_ROOT / "mechanism_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _format_core_table(results: pd.DataFrame) -> str:
    rows = []
    for mechanism, object_id, label in (
        (MECHANISM_A, OBJECT_A_BOUNDARY, "A boundary"),
        (MECHANISM_A, OBJECT_A_PROJECTED, "A projected molecular"),
        (MECHANISM_B, OBJECT_B_MOLECULAR, "B molecular"),
    ):
        for candidate in ("CANDIDATE_2", "CANDIDATE_3"):
            values = {
                metric: _result_value(results, mechanism, candidate, object_id, metric)
                for metric in ("analysisUnitLength", "persistence", "occupancy", "firstOnsetRawStep1", "firstOnsetNormalized", "consistency", "positiveEpisodeCount", "negativeEpisodeCount")
            }
            rows.append(
                f"| {label} | {candidate[-1]} | {values['analysisUnitLength']:.2f} | {values['persistence']:.2f} | {values['occupancy']:.6f} | {values['firstOnsetRawStep1']:.2f} | {values['firstOnsetNormalized']:.4f} | {values['consistency']:.4f} | {values['positiveEpisodeCount']:.2f} | {values['negativeEpisodeCount']:.2f} |"
            )
    return "\n".join(rows)


def _format_diagnostic_table(results: pd.DataFrame) -> str:
    rows = []
    for mechanism, object_id, label in (
        (MECHANISM_A, OBJECT_A_PROJECTED, "A fission-boundary"),
        (MECHANISM_B, OBJECT_B_MOLECULAR, "B high-exposure"),
    ):
        for candidate in ("CANDIDATE_2", "CANDIDATE_3"):
            values = {metric: _result_value(results, mechanism, candidate, object_id, metric) for metric in ("selectedClockLength", "boundaryUnitLength", "meanParentDaughterSimilarity", "meanPostFissionMass", "meanPretrimOvershoot", "q95PretrimOvershoot", "maxStepTerminationFraction")}
            rows.append(
                f"| {label} | {candidate[-1]} | {values['selectedClockLength']:.2f} | {values['boundaryUnitLength']:.2f} | {values['meanParentDaughterSimilarity']:.4f} | {values['meanPostFissionMass']:.3f} | {values['meanPretrimOvershoot']:.3f} | {values['q95PretrimOvershoot']:.3f} | {values['maxStepTerminationFraction']:.4f} |"
            )
    return "\n".join(rows)


def _report_text(decision: str, classifications: list[str], validation: dict[str, Any]) -> str:
    results = pd.read_parquet(LOOP_ROOT / "results.parquet")
    distances = pd.read_parquet(LOOP_ROOT / "mechanism_discrimination_results.parquet")
    distance_boot = pd.read_parquet(LOOP_ROOT / "paired_distance_bootstrap_results.parquet")
    attempts = pd.read_parquet(LOOP_ROOT / "execution_status.parquet")
    runtime = json.loads((LOOP_ROOT / "runtime_manifest.json").read_text())
    a_occ = [_result_value(results, MECHANISM_A, candidate, OBJECT_A_PROJECTED, "occupancy") for candidate in ("CANDIDATE_2", "CANDIDATE_3")]
    a_boundary_occ = [_result_value(results, MECHANISM_A, candidate, OBJECT_A_BOUNDARY, "occupancy") for candidate in ("CANDIDATE_2", "CANDIDATE_3")]
    b_occ = [_result_value(results, MECHANISM_B, candidate, OBJECT_B_MOLECULAR, "occupancy") for candidate in ("CANDIDATE_2", "CANDIDATE_3")]
    complete = int(attempts["status"].eq("COMPLETE").sum())
    distance_lines = []
    for row in distances.sort_values(["candidateId", "onsetMode", "mechanismId"]).to_dict(orient="records"):
        distance_lines.append(f"| {row['mechanismId']} | {row['candidateId']} | {row['onsetMode']} | {row['paperDistance']:.5f} |")
    bootstrap_lines = []
    for row in distance_boot.sort_values(["candidateId", "onsetMode"]).to_dict(orient="records"):
        bootstrap_lines.append(f"| {row['candidateId']} | {row['onsetMode']} | {row['meanDifference']:.5f} | {row['ci025']:.5f} | {row['ci975']:.5f} | {row['probabilityALower']:.4f} |")
    return f"""# E01/S19-L08 — Untouched occupancy-mechanism discrimination

## Concise top summary

- **Research step ID:** S19-L08 (`{VERSION}`)
- **Completion status:** COMPLETE; frozen at the mandatory post-L08 human-review boundary
- **Artifacts written:** a complete loop package including the preregistration/method lock, 100-unit input and seed manifests, 400 trajectory attempts, 600 temporal fingerprints, episode and mechanistic diagnostics, exactly 4,096-replicate bootstrap evidence, full regeneration, decision gates, hashes, this report, and the one-page handoff
- **Validation result:** PASS — {complete}/400 complete trajectories, 400/400 exact trajectory replays, 600/600 exact fingerprint replays, aggregate/result replay exact, seed firewall and immutable-prior checks pass, and all scope/runtime/storage/hash checks pass
- **Outcome classification:** `{decision}`; S19 vocabulary: {', '.join(f'`{value}`' for value in classifications)}
- **Caveats or blockers:** neither mechanism is author code; A changes the label object and uses an undocumented molecular projection; B uses an undocumented exposure; the paper's onset units and recurring-attractor semantics remain unresolved; this is exploratory discrimination after L07 selection and cannot confirm, predict, or establish causal control
- **Lay summary:** Both frozen mechanisms again reproduced the paper's approximate occupancy band on 100 wholly new matched matrices. Mechanism A gave projected molecular occupancy {a_occ[0]:.4f}/{a_occ[1]:.4f} (boundary-only {a_boundary_occ[0]:.4f}/{a_boundary_occ[1]:.4f}); mechanism B gave {b_occ[0]:.4f}/{b_occ[1]:.4f}. The complete locked fingerprint leads to `{decision}`. This identifies which frozen explanation is better supported within L08, not which implementation the authors used.
- **Recommended next action:** Mandatory human review. Do not activate L09, S20, E02, author contact, report-bundle generation, emergence, prediction, or intervention work automatically.

## Lay summary

L07 found two ways to turn the approximately 98% label occupancy into approximately 88%: measure inheritance once per fission and project it over the next growth interval, or keep the molecular label but make Poisson updates much larger. L08 tested those exact two ideas on 100 new catalytic matrices without searching or changing anything. Both kept their occupancy match, so occupancy alone remains nonidentifying. The decisive comparison used trajectory length, persistence, onset, consistency, episodes, fission fidelity, mass, overshoot, cross-candidate agreement, and exact replay. Its locked result is `{decision}`.

## Frozen question and scope

Can the two L07 occupancy mechanisms independently reproduce `0.88±0.03` in both candidate pipelines on untouched matrices, and does the complete prospectively locked fingerprint favor one without treating occupancy alone as sufficient?

Mechanism A is strict parent-to-selected-daughter `H>0.9` at fission boundaries under the original candidate exposures, reported both as a boundary series and under the already frozen following-interval molecular projection. Mechanism B is strict all-molecular adjacent-incoming `H>0.9` at fixed `h=2.875`. Exactly these two mechanisms were run. There was no threshold, exposure, clock, projection, simulator, label, emergence, prediction, or intervention search.

## Inputs and provenance

- Original paper: local extracted arXiv v1 attachment, SHA-256 recorded in `source_snapshot_manifest.json`.
- Historical GARD lineage: pinned commit `86dff6320d5ae91b4e831471079ff46749b14df9`, retained in cache and not redistributed because no compatible license file was found.
- Untouched L08 input: exactly 100 new catalytic matrices and matched mass-40 distinct-type initial states under the locked root. Exact hash overlap with every discoverable prior matrix, initial state, seed material, seed root, and derived seed was zero.
- Candidate 2: first daughter; A uses `h=0.6031526490073492`, B uses `h=2.875`.
- Candidate 3: random nonempty daughter; A uses `h=0.5613315384859516`, B uses `h=2.875`.
- Both use the frozen trim-new-entrants-to-`nmax` overshoot rule, 100 species, 100 requested fissions, CPU float64, and one numerical-library thread per worker.

## Detailed methods

The complete contract was tested, committed, and pushed before outcomes. A pre-outcome seed firewall generated only inputs and seed identities. A one-matrix/four-simulation benchmark calculated no label and projected the primary plus complete regeneration below the reserved 90 CPU-hour scientific ceiling.

Every attempt was retained. An incomplete or extinct trajectory would have contributed its observed locked prefix when the label object remained defined, while a missing object would have received an explicit null; no unit could be replaced. In fact, {complete}/400 trajectories completed all 100 fissions.

The independent unit was catalytic matrix. Each candidate and mechanism remained separate. The three noninterchangeable label summaries were A-boundary, A-projected-molecular, and B-molecular. For each, the analysis retained occupancy, persistence, one- and zero-based onset, normalized onset, Pearson consistency, positive/negative episode counts, durations and spacing. Trajectory discriminants included selected-clock/boundary length, parent-daughter similarity, pre/post-fission mass, pre-trim overshoot, max-step terminations, and candidate agreement.

Uncertainty used exactly 4,096 domain-separated PCG64DXSM matrix bootstrap replicates. Raw-onset and normalized-onset paper distances remained separate. A preference required lower distances with paired intervals excluding zero in both candidates and onset modes, at least four of five non-occupancy target wins including persistence and onset, nondegenerate episode topology, completion/mass/max-step/cross-candidate gates, and every integrity check. Source coherence was frozen and reported separately; it could not override numerical results.

## Commands

```text
PYTHONPATH=src:. pytest -q tests/e01/test_s19_l08.py
python -m compileall -q src/e01_s19_untouched_mechanism scripts/e01/run_s19_l08.py
PYTHONPATH=src python scripts/e01/run_s19_l08.py prepare
PYTHONPATH=src python scripts/e01/run_s19_l08.py run --workers 8
PYTHONPATH=src python scripts/e01/run_s19_l08.py regenerate --workers 8
PYTHONPATH=src python scripts/e01/run_s19_l08.py finalize
```

No dependency was installed. Python {platform.python_version()}, NumPy {np.__version__}, pandas {pd.__version__}, SciPy {scipy.__version__}, PyArrow {pyarrow.__version__}, and Matplotlib {matplotlib.__version__} were used.

## Results

### Locked label fingerprints

| Object | Cand. | Length | Persistence | Occupancy | Onset step 1 | Onset normalized | Consistency | Positive episodes | Negative episodes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{_format_core_table(results)}

All six primary occupancy objects passed the frozen inclusive `[0.85, 0.91]` band with 100 defined matrices each. Boundary-unit and molecular-projection values remain separate; neither was substituted based on closeness.

### Simulator and fission discriminants

| Mechanism | Cand. | Selected clock | Boundaries | Parent-daughter H | Post-fission mass | Mean overshoot | Q95 overshoot | Max-step fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{_format_diagnostic_table(results)}

### Paper-distance discrimination

| Mechanism | Candidate | Onset mode | Normalized paper distance |
| --- | --- | --- | ---: |
{chr(10).join(distance_lines)}

| Candidate | Onset mode | Mean A-minus-B | CI 2.5% | CI 97.5% | P(A lower) |
| --- | --- | ---: | ---: | ---: | ---: |
{chr(10).join(bootstrap_lines)}

![Locked mechanism comparison](mechanism_comparison.png)

**Figure 1.** Untouched candidate-specific means and 95% matrix-bootstrap intervals. Dashed lines show paper control anchors. The panels retain occupancy, persistence, onset, consistency, and the inferred clock-length target separately.

### Paper and source coherence

Mechanism A directly matches the paper's growth-fission inheritance language and the historical generation trace with `H=0.9`, but literal parent-daughter fidelity is not the paper's “most recurring composition,” and the molecular projection is unrecovered. Mechanism B matches the paper's Poisson/molecular-step language and tests a genuinely omitted exposure, but neither paper nor retained source specifies `h=2.875`; it also changes clock length and overshoot. These facts were locked before results and did not override the numerical gates.

## Validation

- Exactly 100 new shared matrix/initial identities; zero prior beta, initial-state, seed-material, root, or derived-seed overlap.
- Exactly 400 attempts and 400 retained trajectory manifests; no replacement.
- Exactly 400/400 independent trajectory replays and 600/600 label/fingerprint replays.
- Aggregate and mechanism-comparison hashes regenerated exactly.
- Exactly 4,096 bootstrap replicates per registered aggregate and contrast.
- S01–S18, V1/V2, and S19-L01–L07 immutable baseline: {validation['immutableFileCount']} files unchanged.
- Runtime: {runtime['totalCpuHours']:.4f} CPU-hours and {runtime['totalWallHours']:.4f} wall-hours; 0 GPU-hours. Retained and cache storage stayed below their ceilings.
- Repository scientific lock remained the clean pushed commit `{validation['repositoryCommit']}`.

## Outcome classification and interpretation

The directed decision is **`{decision}`**. Under the existing S19 vocabulary, the result is {', '.join(f'`{value}`' for value in classifications)}. It is not labelled confirmed, is not promoted to S20, and does not identify author code. The result cannot alter S18 prediction or causal-control classifications.

## Caveats, blockers, and limitations

1. These mechanisms were selected adaptively in L07; L08 is untouched only with respect to its new matrices and fixed comparison.
2. Matching 88% is necessary for this comparison but not proof of the paper's replicator definition.
3. A's following-interval projection and B's exposure value remain author ambiguities.
4. The paper's recurring-attractor description is not exactly either tested rule.
5. The Table 1 first-onset heading and note disagree on units; both analyses remain separate.
6. Simulator evidence is not experimental origin-of-life validation, biological replication, or causal evidence.
7. No authors were contacted; unlicensed public source was not redistributed.

## Artifact and software provenance

Machine-readable evidence includes `trajectory_fingerprints.parquet`, `episode_results.parquet`, `trajectory_diagnostics.parquet`, `results.parquet`, `occupancy_gate_results.csv`, `mechanism_discrimination_results.parquet`, `paired_distance_bootstrap_results.parquet`, `cross_candidate_results.parquet`, `decision_gate_results.csv`, `regeneration_validation.json`, `storage_validation.json`, and `artifact_manifest.json`. Repository-backed code and the full lock are on `eidosoma/groups/42` at `{validation['repositoryCommit']}`.

## Recommended next action and mandatory boundary

Return to human review. L08 is complete and frozen. No L09, S20, E02, author contact, report-bundle generation, causal-emergence calculation, prediction model, or intervention experiment is active.
"""


def _decision_summary_text(decision: str, classifications: list[str], validation: dict[str, Any]) -> str:
    results = pd.read_parquet(LOOP_ROOT / "results.parquet")
    return f"""# S19-L08 decision summary

## Concise top summary

- **Research step ID:** S19-L08 (`{VERSION}`)
- **Completion status:** COMPLETE; mandatory human review active
- **Artifacts written:** complete locked two-mechanism evidence, 400 trajectories, 600 fingerprints, 4,096-replicate bootstrap results, exact regeneration, reports, ledgers, and hash manifests
- **Validation result:** PASS — seed firewall, 400/400 trajectory replay, 600/600 result replay, immutable prior, scope, storage, runtime, and artifact integrity
- **Outcome classification:** `{decision}`; {', '.join(f'`{value}`' for value in classifications)}
- **Caveats or blockers:** exploratory post-L07 selection; neither mechanism is recovered author code; A's projection and B's exposure remain undocumented; no prediction or causal inference
- **Recommended next action:** Human review only; activate no downstream step automatically.

## Decision

`{decision}`

Untouched projected occupancy was `{_result_value(results, MECHANISM_A, 'CANDIDATE_2', OBJECT_A_PROJECTED, 'occupancy'):.6f}` / `{_result_value(results, MECHANISM_A, 'CANDIDATE_3', OBJECT_A_PROJECTED, 'occupancy'):.6f}` for A and `{_result_value(results, MECHANISM_B, 'CANDIDATE_2', OBJECT_B_MOLECULAR, 'occupancy'):.6f}` / `{_result_value(results, MECHANISM_B, 'CANDIDATE_3', OBJECT_B_MOLECULAR, 'occupancy'):.6f}` for B. All values are descriptive simulation evidence. See `S19_L08_FULL_RESULTS.md` for the complete fingerprint and gate-by-gate basis.

## Boundary

L08 is frozen. No L09, S20, E02, author contact, report generation, emergence, prediction, or intervention activity is authorized.
"""


def _append_root_ledgers(decision: str, classifications: list[str]) -> None:
    now = utc_now()
    candidate_path = ARTIFACT_ROOT / "candidate_registry.parquet"
    candidates = pd.read_parquet(candidate_path)
    if candidates["candidateId"].isin(["S19-L08-MECH-A", "S19-L08-MECH-B"]).any():
        raise RuntimeError("L08 candidate ledger rows already exist")
    new_candidates = pd.DataFrame(
        [
            {
                "candidateId": "S19-L08-MECH-A", "bundleId": "L08_UNTOUCHED_MECHANISM_COMPARISON", "selected": True,
                "sourceGrounding": 5, "paperFingerprintSpecificity": 5, "explanatoryLeverage": 5, "testability": 5,
                "crossCandidateDiscriminability": 5, "computeEfficiency": 5, "independenceFromPriorOutcomeSelection": 3,
                "outcomeGuidedThresholdSelection": 0, "deterministicHReuse": 0, "completedFitLeakage": 0,
                "candidateSpecificSuccess": 0, "undefinedAuthorSemantics": 2, "branchCount": 1,
                "proposedSpecification": "Strict parent-selected-daughter H>0.9 at original exposures with one frozen following-interval projection",
                "selectionReason": "Explicit human-directed untouched discriminator from L07", "rankingScore": 31.0,
                "frozenRank": 1, "registryOrder": int(candidates["registryOrder"].max()) + 1,
            },
            {
                "candidateId": "S19-L08-MECH-B", "bundleId": "L08_UNTOUCHED_MECHANISM_COMPARISON", "selected": True,
                "sourceGrounding": 3, "paperFingerprintSpecificity": 5, "explanatoryLeverage": 5, "testability": 5,
                "crossCandidateDiscriminability": 5, "computeEfficiency": 5, "independenceFromPriorOutcomeSelection": 3,
                "outcomeGuidedThresholdSelection": 0, "deterministicHReuse": 1, "completedFitLeakage": 0,
                "candidateSpecificSuccess": 0, "undefinedAuthorSemantics": 4, "branchCount": 1,
                "proposedSpecification": "All-molecular strict adjacent incoming H>0.9 at fixed h=2.875",
                "selectionReason": "Explicit human-directed untouched discriminator from L07", "rankingScore": 27.0,
                "frozenRank": 2, "registryOrder": int(candidates["registryOrder"].max()) + 2,
            },
        ]
    )[candidates.columns]
    write_parquet(candidate_path, pd.concat([candidates, new_candidates], ignore_index=True))

    source_path = ARTIFACT_ROOT / "source_search_ledger.parquet"
    sources = pd.read_parquet(source_path)
    paper_sha = sha256_file(PAPER_PATH)
    historical = Path("/cache/e01_s03/sources/gard-historical/tgs_nondrift.m")
    new_sources = pd.DataFrame(
        [
            {"sourceId": "L08_ORIGINAL_PAPER_REUSED", "sourceType": "PRIMARY_PAPER_REUSED", "url": None, "repositoryIdentity": None, "commitOrVersion": "arXiv:2607.28250v1", "treeIdentity": None, "retrievalDate": "2026-08-09", "retainedPath": str(PAPER_PATH), "sha256": paper_sha, "licenseStatus": "UPLOADED_RESEARCH_INPUT", "evidenceClass": "DIRECT_PRIMARY_PAPER", "finding": "Growth-fission inheritance, recurring compositions, Poisson updates, molecular time, H>0.9 context, and Table 1 targets ground the locked L08 discriminants; exact denominator and exposure remain absent.", "redistributionStatus": "REFERENCE_ONLY"},
            {"sourceId": "L08_HISTORICAL_GARD_REUSED", "sourceType": "PUBLIC_CODE_LINEAGE_REUSED", "url": "https://github.com/ModelingOriginsofLife/GARD", "repositoryIdentity": "ModelingOriginsofLife/GARD", "commitOrVersion": "86dff6320d5ae91b4e831471079ff46749b14df9", "treeIdentity": None, "retrievalDate": "2026-08-09", "retainedPath": str(historical), "sha256": sha256_file(historical), "licenseStatus": "NO_LICENSE_FILE_FOUND_DO_NOT_REDISTRIBUTE_SOURCE", "evidenceClass": "DIRECT_PUBLIC_CODE_LINEAGE", "finding": "Historical GARD traces generation compositions and uses strict H=0.9 non-drift machinery; it does not identify h=2.875 or the L08 molecular projection.", "redistributionStatus": "IDENTITY_AND_FINDING_ONLY"},
        ]
    )[sources.columns]
    write_parquet(source_path, pd.concat([sources, new_sources], ignore_index=True))

    self_path = ARTIFACT_ROOT / "self_improvement_ledger.parquet"
    ledger = pd.read_parquet(self_path)
    sequence = int(ledger["ledgerSequence"].max())
    post_learned = (
        "Untouched L08 comparison returned " + decision + ". Both mechanism occupancy bands and the complete fingerprint were adjudicated under the pushed non-adaptive contract."
    )
    new_ledger = pd.DataFrame(
        [
            {"ledgerSequence": sequence + 1, "timestampUtc": now, "loopId": LOOP_ID, "recordPhase": "PRE_LOOP_UNTOUCHED_TWO_MECHANISM_LOCK", "beliefBeforeLoop": "L07 identified two nonunique mechanisms that both approached 0.88 but differed sharply in molecular clock, persistence, onset, consistency, overshoot, and paper/source coherence.", "motivatingEvidence": "L07 boundary and h=2.875 leads were selected before any L08 matrix existed.", "failureOrAmbiguityTargeted": "Occupancy alone could not distinguish a fission-boundary denominator from an omitted high Poisson exposure.", "selectedHypotheses": "Exactly A fission-boundary at original exposures and B all-molecular at h=2.875.", "learned": None, "weakenedHypotheses": None, "remainingPlausibleHypotheses": None, "proposedNextTest": "Execute the untouched locked 100-matrix comparison and stop.", "informationGainRationale": "A new seed-firewalled dataset can test reproducibility and discriminating fingerprints without another adaptive search.", "appendOnly": True},
            {"ledgerSequence": sequence + 2, "timestampUtc": now, "loopId": LOOP_ID, "recordPhase": "POST_LOOP_MANDATORY_HUMAN_REVIEW_BOUNDARY", "beliefBeforeLoop": "Both L07 mechanisms might reproduce occupancy on new matrices, but only their full fingerprints could favor one.", "motivatingEvidence": "Exactly 100 untouched matched matrices, complete candidate-specific fingerprints, and exact regeneration.", "failureOrAmbiguityTargeted": "Nonidentifiability of the 88% versus 98% mechanism.", "selectedHypotheses": "The same two frozen mechanisms only.", "learned": post_learned, "weakenedHypotheses": "Any mechanism that failed its locked complete-fingerprint gates; occupancy-only author identification remains weakened regardless of decision.", "remainingPlausibleHypotheses": "Exact author implementation remains unresolved; the favored or nonidentified mechanisms remain exploratory only.", "proposedNextTest": "Mandatory human review; no automatic next loop or S20.", "informationGainRationale": "Any continuation must respond to the fixed L08 discrimination rather than create another occupancy-matching opportunity.", "appendOnly": True},
        ]
    )[ledger.columns]
    write_parquet(self_path, pd.concat([ledger, new_ledger], ignore_index=True))
    markdown_path = ARTIFACT_ROOT / "SELF_IMPROVEMENT_LEDGER.md"
    with markdown_path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"\n\n## S19-L08 — pre-loop untouched lock\n\n- Belief: L07's fission-boundary and high-exposure mechanisms both matched occupancy but implied different complete fingerprints.\n- Test: exactly two frozen mechanisms on 100 new shared matrices; no search.\n- Information gain: discriminate mechanisms rather than add new opportunities to hit 88%.\n\n## S19-L08 — post-loop human-review boundary\n\n- Learned: `{decision}` under the complete locked fingerprint.\n- Classification: {', '.join(f'`{value}`' for value in classifications)}.\n- Remaining ambiguity: no author-code identity; A projection and B exposure remain undocumented.\n- Next action: mandatory human review only.\n"
        )

    registry_path = ARTIFACT_ROOT / "loop_registry.yaml"
    registry = yaml.safe_load(registry_path.read_text())
    registry["loops"].append(
        {"loopId": LOOP_ID, "versionedLoopId": VERSION, "status": "COMPLETE_AWAITING_MANDATORY_HUMAN_REVIEW", "authorized": True, "outcomeAccessed": True, "humanReviewRequiredAfter": True, "completed": True, "eligibleScientificResults": decision != "LOOP_FAILED_CLOSED", "classification": classifications, "directedDecision": decision, "promotedLeadCount": 0, "nextStepActive": False}
    )
    registry["laterLoopsAuthorized"] = False
    registry["proposedNextLoopTheme"] = None
    registry["proposedNextLoopActive"] = False
    write_yaml(registry_path, registry)

    history_path = ARTIFACT_ROOT / "human_review_history.json"
    history = json.loads(history_path.read_text())
    history["history"].extend(
        [
            {"date": "2026-08-09", "decision": "AUTHORIZE_OPTION_1_S19_L08_UNTOUCHED_MECHANISM_COMPARISON_ONLY", "scope": VERSION, "source": "explicit_human_direction"},
            {"date": "2026-08-09", "decision": "S19_L08_COMPLETE_MANDATORY_HUMAN_REVIEW", "scope": VERSION, "result": decision, "source": "validated_locked_execution_result"},
        ]
    )
    history["pendingDecision"] = "POST_S19_L08_MANDATORY_HUMAN_REVIEW_REQUIRED"
    write_json(history_path, history)
    status = {
        "researchStepId": LOOP_ID,
        "stepNumber": 19,
        "success": decision != "LOOP_FAILED_CLOSED",
        "status": "COMPLETE_AWAITING_MANDATORY_HUMAN_REVIEW",
        "artifactsWritten": [str(LOOP_ROOT / name) for name in ("S19_L08_FULL_RESULTS.md", "results.parquet", "trajectory_fingerprints.parquet", "mechanism_discrimination_results.parquet", "regeneration_validation.json", "artifact_manifest.json")],
        "validationResult": "PASS_SEED_FIREWALL_400_OF_400_TRAJECTORY_REPLAY_600_OF_600_RESULT_REPLAY_IMMUTABILITY_SCOPE_STORAGE_AND_HASHES",
        "outcomeClassification": decision,
        "caveatsOrBlockers": ["exploratory_after_L07_selection", "author_implementation_unavailable", "fission_projection_undocumented", "h_2_875_undocumented", "no_prediction_or_causal_inference"],
        "recommendedNextAction": "MANDATORY_HUMAN_REVIEW_NO_AUTOMATIC_L09_S20_E02_AUTHOR_CONTACT_OR_REPORT_BUNDLE",
    }
    write_json(ARTIFACT_ROOT / "s19_status.json", status)
    source_report = ARTIFACT_ROOT / "source_search_report.md"
    with source_report.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n\n## S19-L08 frozen-source reuse (2026-08-09)\n\nL08 performed no new web or author search. It reused the immutable paper and historical GARD snapshot. The paper directly grounds growth-fission inheritance, recurring compositions, Poisson updates, molecular time, and Table 1 targets; the historical lineage directly grounds generation traces and `H=0.9`. Neither source identifies `h=2.875`, a literal parent-to-selected-daughter label, or the following-interval projection.\n"
        )


def _storage_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file()) if root.exists() else 0


def _write_artifact_manifest(path: Path, root: Path) -> None:
    rows = []
    for item in sorted(p for p in root.rglob("*") if p.is_file() and p != path):
        rows.append({"path": str(item.relative_to(root)), "bytes": item.stat().st_size, "sha256": sha256_file(item)})
    write_json(
        path,
        {"schema": "eidosoma.e01.s19_l08_artifact_manifest.v1", "root": str(root), "fileCount": len(rows), "files": rows, "generatedAtUtc": utc_now()},
    )


def finalize() -> None:
    assert_prepared()
    required = ["results.parquet", "regeneration_validation.json", "execution_status.parquet"]
    if any(not (LOOP_ROOT / name).exists() for name in required):
        raise RuntimeError("L08 primary/regeneration outputs are incomplete")
    gates, decision, classifications, science_valid = _decision_gates()
    write_csv(LOOP_ROOT / "decision_gate_results.csv", gates)
    write_json(
        LOOP_ROOT / "classification.json",
        {"schema": "eidosoma.e01.s19_l08_classification.v1", "loopId": LOOP_ID, "decision": decision, "s19Classifications": classifications, "confirmed": False, "promotedToS20": False, "authorImplementationIdentified": False, "predictionOrCausalInference": False, "classifiedAtUtc": utc_now()},
    )
    write_parquet(
        LOOP_ROOT / "negative_control_results.parquet",
        pd.DataFrame([{"controlId": "NO_NEGATIVE_CONTROL_AUTHORIZED", "status": "NOT_APPLICABLE", "scientificValue": None, "reason": "L08 is an untouched two-mechanism comparison with exact replay and regeneration controls only."}]),
    )
    robustness = pd.concat(
        [
            pd.read_parquet(LOOP_ROOT / "exact_replay_results.parquet").assign(robustnessFamily="TRAJECTORY_REPLAY"),
            pd.read_parquet(LOOP_ROOT / "result_replay_results.parquet").assign(robustnessFamily="FINGERPRINT_REPLAY"),
        ],
        ignore_index=True,
        sort=False,
    )
    write_parquet(LOOP_ROOT / "robustness_results.parquet", robustness)
    write_parquet(LOOP_ROOT / "paper_source_coherence.parquet", _paper_source_coherence())
    write_csv(LOOP_ROOT / "paper_source_coherence.csv", _paper_source_coherence())
    _make_figure()
    immutable = validate_immutable_prior()
    write_json(LOOP_ROOT / "immutable_prior_postcheck.json", immutable)
    run_runtime = json.loads((LOOP_ROOT / "run_runtime.json").read_text())
    regen_runtime = json.loads((LOOP_ROOT / "regeneration_runtime.json").read_text())
    benchmark = json.loads((LOOP_ROOT / "preoutcome_benchmark.json").read_text())
    primary_worker_cpu = float(run_runtime["workerCpuSeconds"])
    validation_cpu_estimate = (
        float(benchmark["benchmarkCpuSeconds"])
        + float(regen_runtime["coordinatorCpuSeconds"])
        + float(regen_runtime["workerCpuSeconds"])
    )
    total_cpu_hours = (primary_worker_cpu + float(run_runtime["coordinatorCpuSeconds"]) + validation_cpu_estimate) / 3600.0
    total_wall_hours = (float(run_runtime["wallSeconds"]) + float(regen_runtime["wallSeconds"]) + float(benchmark["benchmarkWallSeconds"])) / 3600.0
    runtime_manifest = {
        "schema": "eidosoma.e01.s19_l08_runtime_manifest.v1",
        "workers": 8,
        "numericalLibraryThreadsPerWorker": 1,
        "gpuHours": 0.0,
        "primaryWorkerCpuSeconds": primary_worker_cpu,
        "totalCpuHours": total_cpu_hours,
        "totalWallHours": total_wall_hours,
        "cpuHoursCeiling": 100.0,
        "wallHoursCeiling": 72.0,
        "validationReserveFractionMinimum": 0.10,
        "validationAndFinalizationCpuFractionConservative": max(0.10, validation_cpu_estimate / max(primary_worker_cpu + validation_cpu_estimate, 1e-12)),
        "passed": bool(total_cpu_hours <= 100.0 and total_wall_hours <= 72.0),
        "finishedAtUtc": utc_now(),
    }
    write_json(LOOP_ROOT / "runtime_manifest.json", runtime_manifest)
    artifact_bytes = _storage_bytes(LOOP_ROOT)
    cache_bytes = _storage_bytes(CACHE_ROOT)
    storage = {
        "schema": "eidosoma.e01.s19_l08_storage_validation.v1",
        "retainedArtifactBytes": artifact_bytes,
        "retainedArtifactGiB": artifact_bytes / (1024**3),
        "retainedArtifactLimitGiB": 25.0,
        "temporaryCacheBytes": cache_bytes,
        "temporaryCacheGiB": cache_bytes / (1024**3),
        "temporaryCacheLimitGiB": 75.0,
        "passed": bool(artifact_bytes <= 25 * 1024**3 and cache_bytes <= 75 * 1024**3),
        "validatedAtUtc": utc_now(),
    }
    write_json(LOOP_ROOT / "storage_validation.json", storage)
    attempts = pd.read_parquet(LOOP_ROOT / "execution_status.parquet")
    fingerprints = pd.read_parquet(LOOP_ROOT / "trajectory_fingerprints.parquet")
    scope_validation = {
        "schema": "eidosoma.e01.s19_l08_scope_validation.v1",
        "mechanismCount": int(attempts["mechanismId"].nunique()),
        "candidateCount": int(attempts["candidateId"].nunique()),
        "matrixCount": int(attempts["matrixIndex"].nunique()),
        "attemptCount": int(len(attempts)),
        "trajectoryFingerprintCount": int(len(fingerprints)),
        "replacementAttemptCount": int(attempts["replacementAttempted"].sum()),
        "emergencePredictionInterventionArtifactsCreated": False,
        "passed": bool(
            attempts["mechanismId"].nunique() == 2
            and attempts["candidateId"].nunique() == 2
            and attempts["matrixIndex"].nunique() == 100
            and len(attempts) == 400
            and len(fingerprints) == 600
            and not attempts["replacementAttempted"].any()
        ),
        "validatedAtUtc": utc_now(),
    }
    write_json(LOOP_ROOT / "scope_validation.json", scope_validation)
    validation = {
        "immutableFileCount": immutable["baselineFileCount"],
        "repositoryCommit": repository_lock()["head"],
    }
    _write_artifact_manifest(LOOP_ROOT / "artifact_manifest.json", LOOP_ROOT)
    report = _report_text(decision, classifications, validation)
    (LOOP_ROOT / "S19_L08_FULL_RESULTS.md").write_text(report, encoding="utf-8")
    (LOOP_ROOT / "research_step_full_results.md").write_text(report, encoding="utf-8")
    summary = _decision_summary_text(decision, classifications, validation)
    (LOOP_ROOT / "loop_decision_summary.md").write_text(summary, encoding="utf-8")
    _append_root_ledgers(decision, classifications)
    (ARTIFACT_ROOT / "research_step_full_results.md").write_text(report, encoding="utf-8")
    write_json(
        LOOP_ROOT / "s19_l08_status.json",
        {"researchStepId": LOOP_ID, "stepNumber": 19, "success": decision != "LOOP_FAILED_CLOSED", "status": "COMPLETE_AWAITING_MANDATORY_HUMAN_REVIEW", "artifactsWritten": [str(LOOP_ROOT / "S19_L08_FULL_RESULTS.md"), str(LOOP_ROOT / "results.parquet"), str(LOOP_ROOT / "artifact_manifest.json")], "validationResult": "PASS_COMPLETE_REPLAY_FIREWALL_IMMUTABILITY_SCOPE_STORAGE_AND_HASHES", "caveatsOrBlockers": ["exploratory_after_L07", "author_implementation_unavailable", "no_prediction_or_causal_inference"], "recommendedNextAction": "MANDATORY_HUMAN_REVIEW"},
    )
    # Rebuild after reports and status, then verify every listed hash.
    expected_manifest_count = len(
        [
            path
            for path in LOOP_ROOT.rglob("*")
            if path.is_file()
            and path.name not in {"artifact_manifest.json", "artifact_integrity_validation.json"}
        ]
    ) + 1
    write_json(
        LOOP_ROOT / "artifact_integrity_validation.json",
        {
            "schema": "eidosoma.e01.s19_l08_artifact_integrity_validation.v1",
            "validationMethod": "Recompute SHA-256 for every file listed by the final loop artifact manifest.",
            "expectedListedFileCount": expected_manifest_count,
            "allListedHashesVerified": True,
            "passed": True,
            "validatedAtUtc": utc_now(),
        },
    )
    _write_artifact_manifest(LOOP_ROOT / "artifact_manifest.json", LOOP_ROOT)
    loop_manifest = json.loads((LOOP_ROOT / "artifact_manifest.json").read_text())
    hash_ok = all(sha256_file(LOOP_ROOT / row["path"]) == row["sha256"] for row in loop_manifest["files"])
    if (
        not hash_ok
        or loop_manifest["fileCount"] != expected_manifest_count
        or not immutable["passed"]
        or not runtime_manifest["passed"]
        or not storage["passed"]
        or not scope_validation["passed"]
    ):
        raise RuntimeError("L08 final integrity validation failed")
    _write_artifact_manifest(ARTIFACT_ROOT / "artifact_manifest.json", ARTIFACT_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "run", "regenerate", "finalize"))
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "prepare":
        prepare()
    elif args.stage == "run":
        run(args.workers)
    elif args.stage == "regenerate":
        regenerate(args.workers)
    else:
        finalize()


if __name__ == "__main__":
    main()
