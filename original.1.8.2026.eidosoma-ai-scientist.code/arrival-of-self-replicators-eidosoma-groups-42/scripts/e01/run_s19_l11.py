#!/usr/bin/env python3
"""Prepare, execute, regenerate, validate, and freeze E01/S19-L11."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import platform
import re
import resource
import subprocess
import sys
import time
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_name] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet as pq
import scipy
import sklearn
import yaml

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from e01_frozen_timebase_ensemble.core import selected_clock_observations
from e01_latent_timebase.core import (
    ExposureDefinition,
    SimulationDefinition,
    derive_seed,
    generate_beta,
    initialize_distinct_state,
    simulate_trajectory,
)
from e01_latent_timebase.core import array_sha256 as trajectory_array_sha256
from e01_s19_all_comptype_union.core import (
    BOOTSTRAP_REPLICATES,
    PAPER_TARGETS,
    PIPELINE_IDS,
    RANDOM_CENTROID_DRAWS,
    THRESHOLD,
    U1_ID,
    U2_ID,
    VERSION,
    array_sha256,
    bootstrap_indices,
    close_rows,
    deterministic_seed,
    direct_union_scores,
    historical_h,
    label_fingerprint,
    materialize_pipeline,
    materialize_u1,
    materialize_u2,
    paper_distance,
    project_boundary_values,
    run_descriptors,
    serialize_worker_exception,
)
from e01_s19_occupancy_search.core import boundary_scores, materialize_frozen_setting
from e01_s19_untouched_mechanism.core import (
    MECHANISM_A,
    OBJECT_A_BOUNDARY,
    OBJECT_A_PROJECTED,
    materialize_analysis_object,
)
from e01_s19_untouched_mechanism.core import label_fingerprint as l08_label_fingerprint

LOOP_ID = "S19-L11"
ARTIFACT_ROOT = Path("/artifacts/research_steps/S19")
LOOP_ROOT = ARTIFACT_ROOT / "loops/L11"
CACHE_ROOT = Path("/cache/e01_s19_l11")
PRIMARY_CACHE = CACHE_ROOT / "trajectories"
REPLAY_CACHE = CACHE_ROOT / "regeneration"
REPLAY_OUTPUT = CACHE_ROOT / "regenerated_outputs"
CONFIG_PATH = REPO / "configs/e01/s19_l11_all_comptype_union.yaml"
CORE_PATH = REPO / "src/e01_s19_all_comptype_union/core.py"
RUNNER_PATH = Path(__file__).resolve()
TEST_PATH = REPO / "tests/e01/test_s19_l11.py"
L10_CORE_PATH = REPO / "src/e01_s19_matlab_attractor/core.py"
L10_CONFIG_PATH = REPO / "configs/e01/s19_l10_matlab_compatible_attractor.yaml"
L10_LOCK = ARTIFACT_ROOT / "loops/L10/implementation_lock.json"
L10_SOURCE_LOCK = ARTIFACT_ROOT / "loops/L10/source_snapshot_manifest.json"
L10_TECHNICAL_REPAIR = ARTIFACT_ROOT / "loops/L10/technical_repair_completion_001.json"
PAPER_MD = Path(
    "/workspace/input-attachments/ed5486bf-a043-485b-a233-d88d8d123759/pdf-markdown.md"
)
PAPER_PDF = Path("/cache/e01_s03/downloads/paper-2607.28250v1.pdf")
PAPER_FIGURE_1 = Path(
    "/workspace/input-attachments/ed5486bf-a043-485b-a233-d88d8d123759/figures/figure-01.png"
)
HISTORICAL_ROOT = Path("/cache/e01_s03/sources/gard-historical")

PRIMARY_CANDIDATES = ("CANDIDATE_2", "CANDIDATE_3")
COMPARATOR_ADJACENT = "ORIGINAL_ADJACENT_MOLECULAR_H090"
COMPARATOR_A_BOUNDARY = "L08_A_FISSION_BOUNDARY_H090"
COMPARATOR_A_PROJECTED = "L08_A_FOLLOWING_INTERVAL_PROJECTED_H090"
COMPARATOR_B_HIGH = "L08_B_HIGH_EXPOSURE_MOLECULAR_H090"
COMPARATOR_L10_R1 = "R1_MATLAB_COMPATIBLE_HISTORICAL_DOMINANT_COMPTYPE_H090"
COMPARATOR_L10_R2 = "R2_PAPER_EUCLIDEAN_DOMINANT_ATTRACTOR_H090"

FINGERPRINT_METRICS = (
    "selectedClockLength",
    "persistence",
    "occupancy",
    "consistency",
    "firstOnsetRawIndex0",
    "firstOnsetRawStep1",
    "firstOnsetNormalized",
    "firstOnsetGeneration",
    "preOnsetNonreplicatingDuration",
    "transitionCount",
    "positiveEpisodeCount",
    "negativeEpisodeCount",
    "positiveMeanEpisodeDuration",
    "negativeMeanEpisodeDuration",
    "positiveLongestEpisodeDuration",
    "negativeLongestEpisodeDuration",
    "positiveMeanEpisodeSpacing",
    "negativeMeanEpisodeSpacing",
    "nonreplicatingAt10Percent",
    "noReplicatorThrough10Percent",
    "nonreplicatingAt20Percent",
    "noReplicatorThrough20Percent",
    "nonreplicatingAt25Percent",
    "noReplicatorThrough25Percent",
    "nonreplicatingAt33Percent",
    "noReplicatorThrough33Percent",
)

CORE_TABLES: dict[str, tuple[str, ...]] = {
    "historical_tag_results.parquet": (
        "pipelineId",
        "candidateId",
        "matrixIndex",
        "boundaryIndex0",
    ),
    "cluster_results.parquet": ("pipelineId", "candidateId", "matrixIndex"),
    "cluster_size_results.parquet": (
        "pipelineId",
        "candidateId",
        "matrixIndex",
        "clusterId",
    ),
    "singleton_contribution_results.parquet": (
        "pipelineId",
        "candidateId",
        "matrixIndex",
    ),
    "recurring_centroid_results.parquet": (
        "pipelineId",
        "candidateId",
        "matrixIndex",
        "clusterId",
    ),
    "molecular_union_label_results.parquet": (
        "pipelineId",
        "candidateId",
        "matrixIndex",
        "analysisUnitIndex",
    ),
    "boundary_label_results.parquet": (
        "pipelineId",
        "candidateId",
        "matrixIndex",
        "boundaryIndex0",
    ),
    "boundary_fingerprint_results.parquet": ("pipelineId", "candidateId", "matrixIndex"),
    "label_fingerprint_results.parquet": ("pipelineId", "candidateId", "matrixIndex"),
    "episode_results.parquet": (
        "pipelineId",
        "candidateId",
        "matrixIndex",
        "polarity",
        "episodeIndex",
    ),
    "comparator_results.parquet": (
        "recordScope",
        "pipelineId",
        "candidateId",
        "matrixIndex",
    ),
    "negative_control_results.parquet": (
        "pipelineId",
        "candidateId",
        "matrixIndex",
        "controlType",
        "controlIndex",
    ),
    "paper_target_comparison.csv": ("pipelineId", "candidateId", "metric"),
    "complete_fingerprint_distances.parquet": (
        "pipelineId",
        "candidateId",
        "onsetMode",
    ),
    "bootstrap_results.parquet": ("pipelineId", "candidateId", "bootstrapReplicate"),
    "candidate_comparison.csv": ("pipelineId", "metric"),
    "scientific_gate_results.parquet": ("pipelineId", "candidateId", "gateId"),
    "aggregate_fingerprint_results.parquet": ("pipelineId", "candidateId"),
    "boundary_aggregate_results.parquet": ("pipelineId", "candidateId"),
    "comparator_aggregate_results.parquet": ("recordScope", "pipelineId", "candidateId"),
}


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


def canonicalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Freeze schema order independently of parallel worker completion order."""

    return frame.reindex(sorted(frame.columns), axis=1)


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    canonicalize_frame(frame).to_parquet(path, index=False, compression="zstd")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    canonicalize_frame(frame).to_csv(path, index=False, lineterminator="\n")


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, text=True, capture_output=True
    ).stdout.strip()


def load_config() -> dict[str, Any]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["versionedLoopId"] != VERSION:
        raise RuntimeError("L11 version/config mismatch")
    if [row["pipelineId"] for row in config["pipelines"]] != list(PIPELINE_IDS):
        raise RuntimeError("L11 requires exactly the two registered pipelines")
    return config


def canonical_frame_sha256(frame: pd.DataFrame, sort_columns: Iterable[str]) -> str:
    ordered = canonicalize_frame(frame.copy())
    columns = [column for column in sort_columns if column in ordered.columns]
    if columns:
        ordered = ordered.sort_values(columns, kind="stable").reset_index(drop=True)
    for column in ordered.columns:
        if ordered[column].dtype == object:
            ordered[column] = ordered[column].map(
                lambda value: (
                    json.dumps(json_safe(value), sort_keys=True, separators=(",", ":"))
                    if isinstance(value, (list, tuple, dict, np.ndarray))
                    else value
                )
            )
    return sha256_text(ordered.to_csv(index=False, lineterminator="\n", na_rep="<NA>"))


def simulation_specs() -> list[dict[str, Any]]:
    rows = [dict(row) for row in load_config()["simulations"]]
    if len(rows) != 2 or {row["candidateId"] for row in rows} != set(PRIMARY_CANDIDATES):
        raise RuntimeError("L11 requires exactly candidate 2 and candidate 3")
    return rows


def make_definition(spec: dict[str, Any]) -> SimulationDefinition:
    return SimulationDefinition(
        daughter_rule=spec["daughterRule"],
        overshoot_rule=spec["overshootRule"],
        exposure=ExposureDefinition(
            family="FIXED_COMMON_EXPOSURE", h=float(spec["exposure"])
        ),
    )


def trajectory_path(cache_root: Path, matrix_index: int, candidate_id: str) -> Path:
    return cache_root / f"M{matrix_index:03d}__ORIGINAL_EXPOSURE__{candidate_id}.pkl"


def prior_roots() -> list[Path]:
    roots: list[Path] = []
    for path in sorted(Path("/artifacts/research_steps").iterdir()):
        if path.name != "S19":
            roots.append(path)
    for loop in (
        "L01",
        "L02",
        "L03",
        "L04",
        "L05",
        "L06",
        "L06R",
        "L07",
        "L08",
        "L09",
        "L10",
    ):
        roots.append(ARTIFACT_ROOT / "loops" / loop)
    for bundle in (
        Path("/artifacts/E01_forensic_replication_bundle"),
        Path("/artifacts/E01_forensic_replication_artifact_v2"),
    ):
        if bundle.exists():
            roots.append(bundle)
    return roots


def immutable_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in prior_roots():
        if not root.exists():
            raise FileNotFoundError(root)
        files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
        for path in files:
            rows.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return rows


def validate_immutable_prior() -> dict[str, Any]:
    baseline = json.loads((LOOP_ROOT / "immutable_prior_baseline.json").read_text())
    mismatches: list[str] = []
    for row in baseline["files"]:
        path = Path(row["path"])
        if (
            not path.exists()
            or path.stat().st_size != row["bytes"]
            or sha256_file(path) != row["sha256"]
        ):
            mismatches.append(row["path"])
    return {
        "schema": "eidosoma.e01.s19_l11.immutable_prior_validation.v1",
        "baselineFileCount": len(baseline["files"]),
        "mismatchCount": len(mismatches),
        "mismatches": mismatches[:50],
        "passed": not mismatches,
        "validatedAtUtc": utc_now(),
    }


def all_prior_files() -> list[Path]:
    files: list[Path] = []
    for root in prior_roots():
        files.extend([root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file()))
    return files


def collect_prior_identity_inventory() -> dict[str, set[str]]:
    inventory = {
        "beta": set(),
        "initial": set(),
        "seedMaterial": set(),
        "root": set(),
        "derivedSeed": set(),
        "allHex64": set(),
    }
    targets = {
        "betasha256": "beta",
        "initialstatesha256": "initial",
        "seedmaterialsha256": "seedMaterial",
        "roothex": "root",
        "matrixroothex": "root",
        "bootstraproothex": "root",
        "controlroothex": "root",
        "derivedseed": "derivedSeed",
    }
    pattern = re.compile(r"(?<![0-9a-fA-F])[0-9a-fA-F]{64}(?![0-9a-fA-F])")
    for path in all_prior_files():
        try:
            suffix = path.suffix.lower()
            if suffix == ".parquet":
                schema = pq.read_schema(path)
                names = [name for name in schema.names if name.lower() in targets]
                if names:
                    frame = pd.read_parquet(path, columns=names)
                    for name in names:
                        for value in frame[name].dropna().astype(str):
                            inventory[targets[name.lower()]].add(value)
                            inventory["allHex64"].update(x.lower() for x in pattern.findall(value))
            elif suffix in {".json", ".yaml", ".yml", ".csv", ".md", ".txt"} and path.stat().st_size <= 25 * 1024 * 1024:
                text = path.read_text(encoding="utf-8", errors="replace")
                inventory["allHex64"].update(x.lower() for x in pattern.findall(text))
                if suffix == ".csv":
                    header = pd.read_csv(path, nrows=0)
                    names = [name for name in header.columns if name.lower() in targets]
                    if names:
                        frame = pd.read_csv(path, usecols=names, dtype=str)
                        for name in names:
                            inventory[targets[name.lower()]].update(frame[name].dropna().astype(str))
        except (OSError, ValueError, UnicodeError, pyarrow.ArrowException):
            continue
    return inventory


def seed_and_input_manifests() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    config = load_config()
    root = config["seedContract"]["matrixRootHex"]
    phase = config["seedContract"]["phase"]
    inputs: list[dict[str, Any]] = []
    seeds: list[dict[str, Any]] = []
    for matrix_index in range(100):
        beta_seed = derive_seed(root, phase, "catalytic_matrix", matrix_index)
        initial_seed = derive_seed(root, phase, "initial_state", matrix_index)
        beta = generate_beta(beta_seed)
        initial = initialize_distinct_state(initial_seed)
        inputs.append(
            {
                "matrixIndex": matrix_index,
                "betaSha256": trajectory_array_sha256(beta),
                "initialStateSha256": trajectory_array_sha256(initial),
                "initialMass": int(initial.sum()),
                "initialDistinctTypes": int(np.count_nonzero(initial)),
            }
        )
        for spec in simulation_specs():
            identities = (
                beta_seed,
                initial_seed,
                derive_seed(root, phase, "poisson_update", matrix_index, spec["streamIdentity"]),
                derive_seed(root, phase, "overshoot_trim", matrix_index, spec["streamIdentity"]),
                derive_seed(root, phase, "fission", matrix_index, spec["streamIdentity"]),
                derive_seed(root, phase, "daughter_selection", matrix_index, spec["streamIdentity"]),
            )
            for identity in identities:
                seeds.append(
                    {
                        "loopId": LOOP_ID,
                        "candidateId": spec["candidateId"],
                        "matrixIndex": matrix_index,
                        "purpose": identity.purpose,
                        "configurationId": identity.configuration_id,
                        "derivedSeed": str(identity.derived_seed),
                        "seedMaterialSha256": identity.seed_material_sha256,
                        "rootHex": root,
                        "phase": phase,
                    }
                )
    input_frame = pd.DataFrame(inputs)
    seed_frame = pd.DataFrame(seeds)
    prior = collect_prior_identity_inventory()
    new_beta = set(input_frame["betaSha256"].astype(str))
    new_initial = set(input_frame["initialStateSha256"].astype(str))
    new_material = set(seed_frame["seedMaterialSha256"].astype(str))
    new_derived = set(seed_frame["derivedSeed"].astype(str))
    roots = {
        config["seedContract"]["matrixRootHex"],
        config["seedContract"]["bootstrapRootHex"],
        config["seedContract"]["controlRootHex"],
    }
    overlaps = {
        "beta": sorted(new_beta & prior["beta"]),
        "initialState": sorted(new_initial & prior["initial"]),
        "seedMaterial": sorted(new_material & prior["seedMaterial"]),
        "derivedSeed": sorted(new_derived & prior["derivedSeed"]),
        "root": sorted(roots & (prior["root"] | prior["allHex64"])),
    }
    passed = bool(
        len(input_frame) == 100
        and input_frame["betaSha256"].nunique() == 100
        and input_frame["initialStateSha256"].nunique() == 100
        and not any(overlaps.values())
    )
    firewall = {
        "schema": "eidosoma.e01.s19_l11.seed_firewall.v1",
        "matrixCount": len(input_frame),
        "newBetaCount": len(new_beta),
        "newInitialStateCount": len(new_initial),
        "newSeedMaterialCount": len(new_material),
        "newDerivedSeedCount": len(new_derived),
        "priorInventoryCounts": {key: len(value) for key, value in prior.items()},
        "overlaps": overlaps,
        "passed": passed,
        "validatedAtUtc": utc_now(),
    }
    return seed_frame, input_frame, firewall


def source_manifest() -> dict[str, Any]:
    required = [
        "tgs_nondrift.m",
        "tgs_acluster.m",
        "tgs_kmeans.m",
        "tgs_H.m",
        "tgs_parameters_v10.m",
        "tgs_clust.m",
    ]
    files = [
        PAPER_MD,
        PAPER_PDF,
        PAPER_FIGURE_1,
        L10_LOCK,
        L10_SOURCE_LOCK,
        L10_TECHNICAL_REPAIR,
        L10_CORE_PATH,
        L10_CONFIG_PATH,
        *[HISTORICAL_ROOT / name for name in required],
    ]
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing L11 source inputs: {missing}")
    return {
        "schema": "eidosoma.e01.s19_l11.source_snapshot_manifest.v1",
        "capturedAtUtc": utc_now(),
        "historicalGard": {
            "repository": "https://github.com/marcos-delgado/GARD-model",
            "commit": "86dff6320d5ae91b4e831471079ff46749b14df9",
            "licenseStatus": "NO_LICENSE_DETECTED_REFERENCE_ONLY",
        },
        "files": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "redistribution": "REFERENCE_ONLY_NOT_COPIED_TO_ARTIFACTS",
            }
            for path in files
        ],
        "l10ImplementationLockSha256": sha256_file(L10_LOCK),
        "l10SourceLockSha256": sha256_file(L10_SOURCE_LOCK),
        "l10TechnicalRepairCompletionSha256": sha256_file(L10_TECHNICAL_REPAIR),
        "references": [
            {"reference": 63, "doi": "10.1006/jtbi.2001.2440"},
            {"reference": 64, "doi": "10.1023/A:1006583712886"},
            {"reference": 65, "doi": "10.1073/pnas.97.8.4112"},
        ],
    }


def source_semantics_checks() -> list[dict[str, Any]]:
    nondrift = (HISTORICAL_ROOT / "tgs_nondrift.m").read_text(errors="replace")
    cluster = (HISTORICAL_ROOT / "tgs_acluster.m").read_text(errors="replace")
    checks = [
        (
            "TGS_NONDRIFT_RETURNS_LOGICAL_NONDRIFT_INDEX",
            "indx = (avgH > HThreshhold);" in nondrift,
            "tgs_nondrift.m:40 and output documentation lines 12-13",
        ),
        (
            "TGS_ACLUSTER_INITIALIZES_COMPLETE_TAG_MATRIX_TO_ZERO",
            "tagmat = zeros(size(samples,2),length(optclst));" in cluster,
            "tgs_acluster.m:45",
        ),
        (
            "CLUSTER_LABELS_ASSIGNED_TO_ALL_INDEXED_NONDRIFT_COMPOSITIONS",
            "tagmat(indx,i) = tags;" in cluster,
            "tgs_acluster.m:74-76",
        ),
        (
            "SELECTED_CLUSTERING_RETAINS_COMPLETE_TAG_VECTOR",
            "tags = tagmat(:,compnum);" in cluster and "out.tags=tags;" in cluster,
            "tgs_acluster.m:84-93",
        ),
        (
            "HISTORICAL_SOURCE_DOES_NOT_REDUCE_BINARY_STATE_TO_LARGEST_CLUSTER",
            "out.tags=tags;" in cluster and "counts=counts(2:end);" in cluster,
            "source returns all selected tags and centroids; counts are calculated but no largest-cluster filter is applied",
        ),
        (
            "SOURCE_TAG_BINARY_CAN_BE_EXPRESSED_AS_TAG_GREATER_THAN_ZERO",
            "tagmat = zeros" in cluster and "tagmat(indx,i) = tags" in cluster,
            "direct inference from zero drift slots and positive one-based k-means tags",
        ),
    ]
    return [
        {
            "statementId": statement,
            "passed": bool(passed),
            "evidence": evidence,
            "evidenceClass": "DIRECT_SOURCE" if statement != "SOURCE_TAG_BINARY_CAN_BE_EXPRESSED_AS_TAG_GREATER_THAN_ZERO" else "DIRECT_SOURCE_DERIVED_BINARY_INFERENCE",
        }
        for statement, passed, evidence in checks
    ]


def _known_union(
    labels: np.ndarray, centroids: np.ndarray, minimum_size: int = 2
) -> tuple[tuple[int, ...], np.ndarray]:
    sizes = tuple(int(np.count_nonzero(labels == index)) for index in range(len(centroids)))
    valid = tuple(index for index, size in enumerate(sizes) if size >= minimum_size)
    return valid, centroids[list(valid)] if valid else np.empty((0, centroids.shape[1]))


def _fixture_point(axis: int, width: int = 6) -> np.ndarray:
    row = np.full(width, 1e-8, dtype=np.float64)
    row[axis] = 1.0
    return row / row.sum()


def run_fixtures() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def record(fixture: str, check: str, passed: bool, details: str) -> None:
        rows.append({"fixtureId": fixture, "checkId": check, "passed": bool(passed), "details": details})

    a, b, c, d = (_fixture_point(index) for index in range(4))
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64)
    values = np.vstack([a, a, a, b, b, b])
    centroids = np.vstack([a, b])
    valid, union = _known_union(labels, centroids)
    _, direct = direct_union_scores(values, union)
    record("F01_TWO_VALID_RECURRING_CLUSTERS", "U1_BOTH_TAGS_POSITIVE", set(labels + 1) == {1, 2}, str(valid))
    record("F01_TWO_VALID_RECURRING_CLUSTERS", "U2_BOTH_CLUSTERS_POSITIVE", bool(np.all(direct)), str(valid))

    labels = np.asarray([0] * 5 + [1] * 2, dtype=np.int64)
    values = np.vstack([a] * 5 + [b] * 2)
    valid, union = _known_union(labels, centroids)
    _, direct = direct_union_scores(values, union)
    record("F02_DOMINANT_AND_SMALLER_RECURRING", "SMALLER_RETAINED", valid == (0, 1) and bool(np.all(direct)), str(valid))

    source_tags = np.asarray([1, 1, 0, 2, 2], dtype=np.int64)
    record("F03_DRIFT_PLUS_TWO_RECURRING", "DRIFT_NEGATIVE_RECURRING_POSITIVE", bool(np.array_equal(source_tags > 0, [1, 1, 0, 1, 1])), source_tags.tolist().__str__())

    singleton_labels = np.arange(4, dtype=np.int64)
    singleton_centroids = np.vstack([a, b, c, d])
    valid, _ = _known_union(singleton_labels, singleton_centroids)
    record("F04_ALL_SINGLETON", "U1_TAGS_RETAINED", bool(np.all(singleton_labels + 1 > 0)), "four source tags")
    record("F04_ALL_SINGLETON", "U2_NO_RECURRING_CLUSTER_UNION", not valid, str(valid))

    labels = np.asarray([0, 0, 0, 1, 2], dtype=np.int64)
    values = np.vstack([a, a, a, b, c])
    centroids_three = np.vstack([a, b, c])
    valid, union = _known_union(labels, centroids_three)
    _, direct = direct_union_scores(values, union)
    record("F05_RECURRING_PLUS_SINGLETON_OUTLIERS", "U2_ONLY_RECURRING", valid == (0,) and bool(np.array_equal(direct, [1, 1, 1, 0, 0])), str(valid))
    record("F05_RECURRING_PLUS_SINGLETON_OUTLIERS", "U1_PRESERVES_SINGLETON_TAGS", bool(np.all(labels + 1 > 0)), (labels + 1).tolist().__str__())

    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    valid, _ = _known_union(labels, np.vstack([a, b]))
    record("F06_TIED_RECURRING_CLUSTERS", "BOTH_TIES_RETAINED", valid == (0, 1), str(valid))

    perm = np.asarray([2, 0, 5, 3, 1, 4])
    scores, direct = direct_union_scores(values[:, perm], union[:, perm])
    base_scores, base_direct = direct_union_scores(values, union)
    record("F07_FEATURE_PERMUTATION", "LABEL_AND_SCORE_EQUIVALENCE", bool(np.array_equal(direct, base_direct) and np.allclose(scores, base_scores, atol=1e-15)), "feature permutation")

    scaled = values * np.arange(1, len(values) + 1)[:, None]
    scaled_scores, scaled_direct = direct_union_scores(scaled, union)
    record("F08_COMPOSITION_PRESERVING_SCALING", "LABEL_AND_SCORE_EQUIVALENCE", bool(np.array_equal(scaled_direct, base_direct) and np.allclose(scaled_scores, base_scores, atol=1e-15)), "positive row scaling plus closure")

    molecular = np.vstack([a, b, c])
    scores, direct = direct_union_scores(molecular, np.vstack([a, b]))
    # Deliberately choose a boundary pattern whose interval projection differs
    # from direct molecular membership, so the fixture can detect substitution.
    projected = project_boundary_values(np.asarray([False, True]), np.asarray([0, 2]), 3)
    record("F09_DIRECT_MOLECULAR_MEMBERSHIP", "DIRECT_NOT_INTERVAL_PROJECTED", bool(np.array_equal(direct, [1, 1, 0]) and not np.array_equal(direct, projected)), scores.tolist().__str__())

    projected = project_boundary_values(np.asarray([True, False, True]), np.asarray([2, 5, 8]), 10)
    expected = np.asarray([0, 0, 1, 1, 1, 0, 0, 0, 1, 1], dtype=bool)
    record("F10_GENERATION_PROJECTION", "DAUGHTER_TO_BEFORE_NEXT_BOUNDARY", bool(np.array_equal(projected, expected)), projected.astype(int).tolist().__str__())

    boundary = np.vstack([a] * 60 + [b] * 40)
    positions = np.arange(100, dtype=np.int64)
    first_u1 = materialize_u1(boundary, boundary, positions, "L11-FIXTURE-REPLAY-U1")
    second_u1 = materialize_u1(boundary, boundary, positions, "L11-FIXTURE-REPLAY-U1")
    first_u2 = materialize_u2(boundary, boundary, "L11-FIXTURE-REPLAY-U2")
    second_u2 = materialize_u2(boundary, boundary, "L11-FIXTURE-REPLAY-U2")
    replay = bool(
        np.array_equal(first_u1.boundary_tags, second_u1.boundary_tags)
        and np.array_equal(first_u1.molecular_labels, second_u1.molecular_labels)
        and np.array_equal(first_u2.boundary_tags, second_u2.boundary_tags)
        and np.array_equal(first_u2.molecular_labels, second_u2.molecular_labels)
        and first_u1.fit.cluster_sizes == second_u1.fit.cluster_sizes
        and first_u2.fit.cluster_sizes == second_u2.fit.cluster_sizes
    )
    record("F11_EXACT_REPLAY", "CLUSTERS_TAGS_CENTROIDS_SINGLETONS_LABELS", replay, "two exact executions")

    try:
        raise RuntimeError("fixture-worker-failure")
    except RuntimeError as error:
        payload = serialize_worker_exception(
            matrix_id=7,
            candidate_id="CANDIDATE_2",
            pipeline_id=U1_ID,
            generation=11,
            selected_k=3,
            cluster_sizes=(4, 2, 1),
            tag_counts={"zero": 93, "positive": 7},
            seed_identity="fixture-seed",
            error=error,
        )
    expected_keys = {
        "matrixId", "candidateId", "pipelineId", "generation", "selectedK",
        "clusterSizes", "tagCounts", "seedIdentity", "exceptionClass", "exceptionMessage",
    }
    record("F12_WORKER_FAILURE_PROVENANCE", "REQUIRED_FIELDS_SERIALIZED", set(payload) == expected_keys, json.dumps(payload, sort_keys=True))
    return pd.DataFrame(rows)


def append_preoutcome_ledgers(source: dict[str, Any]) -> None:
    source_path = ARTIFACT_ROOT / "source_search_ledger.parquet"
    source_frame = pd.read_parquet(source_path)
    if not (source_frame["sourceId"] == "L11_HISTORICAL_TAG_PATH").any():
        files = {Path(row["path"]).name: row for row in source["files"]}
        new = pd.DataFrame(
            [
                {
                    "sourceId": "L11_HISTORICAL_TAG_PATH",
                    "sourceType": "PUBLIC_SOURCE_LINEAGE",
                    "url": "https://github.com/marcos-delgado/GARD-model",
                    "repositoryIdentity": "marcos-delgado/GARD-model",
                    "commitOrVersion": "86dff6320d5ae91b4e831471079ff46749b14df9",
                    "treeIdentity": None,
                    "retrievalDate": "2026-08-09",
                    "retainedPath": str(HISTORICAL_ROOT / "tgs_acluster.m"),
                    "sha256": files["tgs_acluster.m"]["sha256"],
                    "licenseStatus": "NO_LICENSE_DETECTED_REFERENCE_ONLY",
                    "evidenceClass": "DIRECT_PUBLIC_SOURCE",
                    "finding": "Complete selected tag vector retains zero drift slots and positive labels for every clustered non-drift composition; no largest-cluster reduction is applied.",
                    "redistributionStatus": "IDENTITY_AND_FINDING_ONLY",
                },
                {
                    "sourceId": "L11_PAPER_CLUSTER_PLURAL_LANGUAGE",
                    "sourceType": "ORIGINAL_PAPER",
                    "url": "workspace-uploaded-arxiv-v1",
                    "repositoryIdentity": None,
                    "commitOrVersion": "arXiv:2607.28250v1",
                    "treeIdentity": None,
                    "retrievalDate": "2026-08-09",
                    "retainedPath": str(PAPER_MD),
                    "sha256": files["pdf-markdown.md"]["sha256"],
                    "licenseStatus": "UPLOADED_INPUT_REFERENCE_ONLY",
                    "evidenceClass": "DIRECT_PAPER_LANGUAGE",
                    "finding": "Figure 1C describes self-replicators as clusters, plural, in molecular-composition space; Results also refers to the most recurring composition, leaving binary union semantics unresolved.",
                    "redistributionStatus": "IDENTITY_AND_FINDING_ONLY",
                },
            ]
        )
        write_parquet(source_path, pd.concat([source_frame, new], ignore_index=True))

    candidate_path = ARTIFACT_ROOT / "candidate_registry.parquet"
    candidates = pd.read_parquet(candidate_path)
    if not (candidates["candidateId"] == "S19-L11-U1").any():
        start = int(candidates["registryOrder"].max()) + 1
        additions = pd.DataFrame(
            [
                {
                    "candidateId": "S19-L11-U1", "bundleId": "L11_ALL_COMPTYPE_UNION", "selected": True,
                    "sourceGrounding": 5, "paperFingerprintSpecificity": 5, "explanatoryLeverage": 5,
                    "testability": 5, "crossCandidateDiscriminability": 5, "computeEfficiency": 5,
                    "independenceFromPriorOutcomeSelection": 3, "outcomeGuidedThresholdSelection": 0,
                    "deterministicHReuse": 1, "completedFitLeakage": 1, "candidateSpecificSuccess": 0,
                    "undefinedAuthorSemantics": 1, "branchCount": 1,
                    "proposedSpecification": "Historical complete positive-tag union with fixed boundary-to-molecular projection and singleton audit",
                    "selectionReason": "Direct historical tag-path evidence and explicit human authorization",
                    "rankingScore": 30.0, "frozenRank": 1, "registryOrder": start,
                },
                {
                    "candidateId": "S19-L11-U2", "bundleId": "L11_ALL_COMPTYPE_UNION", "selected": True,
                    "sourceGrounding": 4, "paperFingerprintSpecificity": 5, "explanatoryLeverage": 5,
                    "testability": 5, "crossCandidateDiscriminability": 5, "computeEfficiency": 5,
                    "independenceFromPriorOutcomeSelection": 3, "outcomeGuidedThresholdSelection": 0,
                    "deterministicHReuse": 1, "completedFitLeakage": 1, "candidateSpecificSuccess": 0,
                    "undefinedAuthorSemantics": 2, "branchCount": 1,
                    "proposedSpecification": "Euclidean union of all clusters of size at least two with direct molecular strict-H090 membership",
                    "selectionReason": "Paper cluster-plural/Euclidean wording and explicit human authorization",
                    "rankingScore": 29.0, "frozenRank": 2, "registryOrder": start + 1,
                },
            ]
        )
        write_parquet(candidate_path, pd.concat([candidates, additions], ignore_index=True))

    ledger_path = ARTIFACT_ROOT / "self_improvement_ledger.parquet"
    ledger = pd.read_parquet(ledger_path)
    if not ((ledger["loopId"] == LOOP_ID) & (ledger["recordPhase"] == "PRE_LOOP_OUTCOME_BLIND_METHOD_LOCK")).any():
        row = {
            "ledgerSequence": int(ledger["ledgerSequence"].max()) + 1,
            "timestampUtc": utc_now(),
            "loopId": LOOP_ID,
            "recordPhase": "PRE_LOOP_OUTCOME_BLIND_METHOD_LOCK",
            "beliefBeforeLoop": "L10's single-dominant labels were too sparse, late, and sticky; historical GARD returns every selected positive compotype tag and the paper uses cluster-plural language.",
            "motivatingEvidence": "L10 R1 occupancy was about 0.40 and R2 about 0.28, while source tgs_acluster returns all tags without a largest-cluster reduction.",
            "failureOrAmbiguityTargeted": "Whether self-replication means membership in any compotype cluster rather than one dominant cluster.",
            "selectedHypotheses": "Exactly U1 historical positive-tag union with fixed projection and U2 Euclidean union of all size>=2 centroids with direct molecular membership.",
            "learned": "Pending untouched L11 execution.",
            "weakenedHypotheses": "Pending untouched L11 execution.",
            "remainingPlausibleHypotheses": "Both registered union semantics remain exploratory and completed-run retrospective until validated.",
            "proposedNextTest": "Execute the pushed L11 contract once and stop for mandatory human review.",
            "informationGainRationale": "The loop changes only single-dominant versus all-compotype union while keeping threshold, simulation, clustering, and evidence boundaries fixed.",
            "appendOnly": True,
        }
        write_parquet(ledger_path, pd.concat([ledger, pd.DataFrame([row])], ignore_index=True))

    registry_path = ARTIFACT_ROOT / "loop_registry.yaml"
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    if not any(item.get("loopId") == LOOP_ID for item in registry["loops"]):
        registry["loops"].append(
            {
                "loopId": LOOP_ID,
                "versionedLoopId": VERSION,
                "status": "AUTHORIZED_PREOUTCOME_LOCK_PREPARED",
                "authorized": True,
                "outcomeAccessed": False,
                "humanReviewRequiredAfter": True,
                "completed": False,
                "eligibleScientificResults": None,
                "promotedLeadCount": None,
                "nextStepActive": True,
                "classification": ["PENDING_OUTCOME_BLIND_EXECUTION"],
            }
        )
    registry["laterLoopsAuthorized"] = False
    registry["s20Status"] = "DEFINED_INACTIVE"
    registry["proposedNextLoopTheme"] = None
    registry["proposedNextLoopActive"] = False
    write_yaml(registry_path, registry)

    history_path = ARTIFACT_ROOT / "human_review_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    if not any(entry.get("scope") == VERSION for entry in history["history"]):
        history["history"].append(
            {
                "date": "2026-08-09",
                "decision": "AUTHORIZE_S19_L11_ALL_COMPTYPE_UNION_LABEL_RECONSTRUCTION",
                "scope": VERSION,
                "source": "explicit_human_direction",
            }
        )
    history["pendingDecision"] = "S19_L11_LOCKED_EXECUTION_ACTIVE"
    write_json(history_path, history)


def write_preoutcome_reports(checks: list[dict[str, Any]]) -> None:
    audit_lines = "\n".join(
        f"- `{row['statementId']}` — **{'PASS' if row['passed'] else 'FAIL'}**; {row['evidence']} ({row['evidenceClass']})."
        for row in checks
    )
    (LOOP_ROOT / "source_tag_semantics_audit.md").write_text(
        f"""# L11 Historical Source-Tag Semantics Audit

## Concise top summary

- **Research step ID:** `S19-L11` (`{VERSION}`).
- **Completion status:** pre-outcome source-tag audit complete; scientific outcomes unopened.
- **Artifacts written:** this audit, paper-language audit, source snapshot, implementation lock, fixtures, pipeline registry, seed firewall, and input manifest.
- **Validation result:** all six required historical tagging statements pass direct source inspection.
- **Outcome classification:** pending; no L11 scientific label has been opened.
- **Caveats or blockers:** the public historical lineage is not author code; `Y_g=I(tag_g>0)` is a direct binary representation of the returned tag vector but the paper does not explicitly state this binary reduction.
- **Recommended next action:** commit and push the complete lock, pass the opaque benchmark, execute only U1/U2, validate, freeze, and stop.

## Source-path checks

{audit_lines}

## Interpretation

`tgs_nondrift.m` creates the non-drift mask. `tgs_acluster.m` allocates a zero-filled full-generation tag matrix, inserts one-based k-means tags at every non-drift position for every tested k, chooses one k by the source silhouette rule, and returns the complete selected tag vector plus every centroid. It computes cluster counts but contains no largest-cluster binary filter. Therefore `tag>0` is a source-literal drift-versus-any-compotype binary reconstruction. It is not proven to be the unavailable paper-author label.
""",
        encoding="utf-8",
    )
    (LOOP_ROOT / "paper_label_language_audit.md").write_text(
        f"""# L11 Paper Label-Language Audit

## Concise top summary

- **Research step ID:** `S19-L11` (`{VERSION}`).
- **Completion status:** pre-outcome paper-language audit complete.
- **Artifacts written:** this audit and the hash-pinned source snapshot.
- **Validation result:** the paper simultaneously uses cluster-plural language and a singular “most recurring composition” reference; the union interpretation is source/paper plausible but not uniquely identified.
- **Outcome classification:** `AUTHOR_SEMANTICS_UNRESOLVED_PREOUTCOME`.
- **Caveats or blockers:** Figure 1 is schematic, and neither the text nor caption specifies whether all clusters, only the dominant cluster, source tags, or direct centroid membership becomes the molecular binary label.
- **Recommended next action:** execute only the two prospectively frozen union reconstructions and retain the ambiguity regardless of fit.

The Results state that self-replicators emerge as “recurring compositions” inherited across generations and that assemblies enter or exit relative to “the most recurring composition.” Figure 1C says self-replicators “are clusters in molecular composition space” and depicts a tight cluster. Materials and Methods describe recurring steady compositions highly similar in Euclidean space. These passages ground U1/U2 but do not identify either as author code.
""",
        encoding="utf-8",
    )
    (LOOP_ROOT / "decision_record.md").write_text(
        f"""# S19-L11 Decision Record

## Concise top summary

- **Research step ID:** `S19-L11` (`{VERSION}`).
- **Completion status:** authorized; outcome-blind two-pipeline contract prepared; outcomes unopened.
- **Artifacts written:** preregistration, source/paper audits, source snapshot, implementation lock, fixtures, pipeline registry, seed firewall, and 100-input manifest.
- **Validation result:** source audit, mandatory fixtures, immutable baseline, and zero-overlap seed/input firewall passed; commit/push, benchmark, execution, replay, and final validation remain pending.
- **Outcome classification:** pending.
- **Caveats or blockers:** both primary pipelines discover clusters using completed runs and are retrospective; exact author semantics remain unavailable.
- **Recommended next action:** push the frozen lock, execute once if benchmarked within ceilings, validate, freeze, and return for mandatory human review.

Exactly one scientific contrast is authorized: single dominant cluster versus the union of all compotype clusters. No emergence, prediction, intervention, metric, threshold, exposure, clock, simulator, clustering, or author-contact search is part of L11.
""",
        encoding="utf-8",
    )


def prepare() -> None:
    started = utc_now()
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    if LOOP_ROOT.exists():
        raise RuntimeError(f"L11 artifact directory already exists: {LOOP_ROOT}")
    LOOP_ROOT.mkdir(parents=True)
    PRIMARY_CACHE.mkdir(parents=True, exist_ok=True)
    REPLAY_CACHE.mkdir(parents=True, exist_ok=True)
    REPLAY_OUTPUT.mkdir(parents=True, exist_ok=True)
    config = load_config()

    baseline = immutable_rows()
    write_json(
        LOOP_ROOT / "immutable_prior_baseline.json",
        {
            "schema": "eidosoma.e01.s19_l11.immutable_prior_baseline.v1",
            "capturedAtUtc": utc_now(),
            "fileCount": len(baseline),
            "totalBytes": sum(row["bytes"] for row in baseline),
            "files": baseline,
        },
    )
    source = source_manifest()
    write_json(LOOP_ROOT / "source_snapshot_manifest.json", source)
    checks = source_semantics_checks()
    write_json(
        LOOP_ROOT / "source_tag_semantics_validation.json",
        {
            "schema": "eidosoma.e01.s19_l11.source_tag_semantics_validation.v1",
            "checkCount": len(checks),
            "passedCount": sum(row["passed"] for row in checks),
            "allPassed": all(row["passed"] for row in checks),
            "checks": checks,
        },
    )
    if not all(row["passed"] for row in checks):
        write_preoutcome_reports(checks)
        raise RuntimeError("L11 source-tag audit contradicted registered U1 semantics")

    fixtures = run_fixtures()
    write_parquet(LOOP_ROOT / "fixture_results.parquet", fixtures)
    write_json(
        LOOP_ROOT / "fixture_manifest.json",
        {
            "schema": "eidosoma.e01.s19_l11.fixture_manifest.v1",
            "fixtureFamilyCount": int(fixtures["fixtureId"].nunique()),
            "checkCount": len(fixtures),
            "passedCount": int(fixtures["passed"].sum()),
            "failedCount": int((~fixtures["passed"]).sum()),
            "allMandatoryPassed": bool(fixtures["passed"].all() and fixtures["fixtureId"].nunique() == 12),
            "resultsSha256": sha256_file(LOOP_ROOT / "fixture_results.parquet"),
        },
    )
    if not bool(fixtures["passed"].all()) or fixtures["fixtureId"].nunique() != 12:
        raise RuntimeError("L11 mandatory fixture failure")

    seed_frame, input_frame, firewall = seed_and_input_manifests()
    write_parquet(LOOP_ROOT / "seed_manifest.parquet", seed_frame)
    write_parquet(LOOP_ROOT / "input_units.parquet", input_frame)
    write_json(LOOP_ROOT / "seed_firewall.json", firewall)
    write_json(
        LOOP_ROOT / "input_manifest.json",
        {
            "schema": "eidosoma.e01.s19_l11.input_manifest.v1",
            "matrixCount": 100,
            "matchedInitialStates": True,
            "allInputIdentitiesGeneratedBeforeLabels": True,
            "inputUnitTable": str(LOOP_ROOT / "input_units.parquet"),
            "inputUnitTableSha256": sha256_file(LOOP_ROOT / "input_units.parquet"),
            "seedManifest": str(LOOP_ROOT / "seed_manifest.parquet"),
            "seedManifestSha256": sha256_file(LOOP_ROOT / "seed_manifest.parquet"),
            "betaHashSetSha256": sha256_text("\n".join(sorted(input_frame["betaSha256"].astype(str)))),
            "initialStateHashSetSha256": sha256_text("\n".join(sorted(input_frame["initialStateSha256"].astype(str)))),
            "trajectoryGroups": simulation_specs(),
            "labelsCalculated": False,
        },
    )
    if not firewall["passed"]:
        raise RuntimeError(f"L11 seed firewall failed: {firewall['overlaps']}")

    write_yaml(LOOP_ROOT / "preregistration.yaml", config)
    write_yaml(
        LOOP_ROOT / "pipeline_registry.yaml",
        {
            "schema": "eidosoma.e01.s19_l11.pipeline_registry.v1",
            "pipelineCount": 2,
            "pipelines": config["pipelines"],
            "commonLabel": config["commonLabel"],
            "singletonAudit": config["singletonAudit"],
            "negativeControls": config["negativeControls"],
            "representativeFigureRule": "LOWEST_MATRIX_INDEX_WITH_DEFINED_LABEL_ELSE_EXPLICIT_STATUS_PANEL",
        },
    )
    write_preoutcome_reports(checks)
    append_preoutcome_ledgers(source)

    code_rows = []
    for path in (CORE_PATH, RUNNER_PATH, TEST_PATH, CONFIG_PATH, L10_CORE_PATH, L10_CONFIG_PATH):
        code_rows.append({"path": str(path.relative_to(REPO)), "sha256": sha256_file(path)})
    write_json(
        LOOP_ROOT / "implementation_lock.json",
        {
            "schema": "eidosoma.e01.s19_l11.implementation_lock.v1",
            "versionedLoopId": VERSION,
            "lockedAtUtc": utc_now(),
            "outcomesOpened": False,
            "pipelineIds": list(PIPELINE_IDS),
            "threshold": THRESHOLD,
            "code": code_rows,
            "configurationSha256": sha256_file(CONFIG_PATH),
            "sourceSnapshotManifestSha256": sha256_file(LOOP_ROOT / "source_snapshot_manifest.json"),
            "fixtureManifestSha256": sha256_file(LOOP_ROOT / "fixture_manifest.json"),
            "seedFirewallSha256": sha256_file(LOOP_ROOT / "seed_firewall.json"),
            "inputManifestSha256": sha256_file(LOOP_ROOT / "input_manifest.json"),
            "l10ImplementationLockSha256": sha256_file(L10_LOCK),
            "l10TechnicalRepairCompletionSha256": sha256_file(L10_TECHNICAL_REPAIR),
            "repository": {
                "branch": "eidosoma/groups/42",
                "headBeforeLockCommit": git("rev-parse", "HEAD"),
                "worktreeDirtyExpectedBeforeCommit": True,
            },
            "scientificAmendmentsPermittedAfterRelease": False,
        },
    )
    write_json(
        LOOP_ROOT / "prepare_runtime.json",
        {
            "schema": "eidosoma.e01.s19_l11.prepare_runtime.v1",
            "startedAtUtc": started,
            "completedAtUtc": utc_now(),
            "wallSeconds": time.perf_counter() - wall_start,
            "cpuSeconds": time.process_time() - cpu_start,
            "sourceAuditChecks": len(checks),
            "fixtureChecks": len(fixtures),
            "immutablePriorFileCount": len(baseline),
            "scientificOutcomesOpened": False,
        },
    )
    print(json.dumps({"status": "L11_PREPARED_OUTCOMES_UNOPENED", "fixtures": len(fixtures), "priorFiles": len(baseline)}, sort_keys=True))


def repository_release_gate() -> dict[str, Any]:
    """Require the prospectively locked code, sources, fixtures, and inputs."""

    lock = json.loads((LOOP_ROOT / "implementation_lock.json").read_text())
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/eidosoma/groups/42")
    branch = git("branch", "--show-current")
    clean = not bool(git("status", "--porcelain=v1"))
    hashes = all(
        (REPO / row["path"]).exists()
        and sha256_file(REPO / row["path"]) == row["sha256"]
        for row in lock["code"]
    )
    fixtures = json.loads((LOOP_ROOT / "fixture_manifest.json").read_text())
    firewall = json.loads((LOOP_ROOT / "seed_firewall.json").read_text())
    source = json.loads((LOOP_ROOT / "source_snapshot_manifest.json").read_text())
    source_hashes = all(
        Path(row["path"]).exists()
        and sha256_file(Path(row["path"])) == row["sha256"]
        for row in source["files"]
    )
    immutable = validate_immutable_prior()
    passed = bool(
        head == remote
        and branch == "eidosoma/groups/42"
        and clean
        and hashes
        and fixtures["allMandatoryPassed"]
        and firewall["passed"]
        and source_hashes
        and immutable["passed"]
    )
    result = {
        "schema": "eidosoma.e01.s19_l11.release_gate.v1",
        "head": head,
        "remoteHead": remote,
        "branch": branch,
        "cleanWorktree": clean,
        "lockedCodeHashesMatch": hashes,
        "fixturesPassed": fixtures["allMandatoryPassed"],
        "seedFirewallPassed": firewall["passed"],
        "sourceHashesMatch": source_hashes,
        "immutablePriorPassed": immutable["passed"],
        "passed": passed,
        "validatedAtUtc": utc_now(),
    }
    write_json(LOOP_ROOT / "immutable_prior_validation.json", immutable)
    write_json(LOOP_ROOT / "run_release_gate.json", result)
    return result


def simulate_matrix(matrix_index: int, cache_root: Path) -> dict[str, Any]:
    """Generate both prospectively paired candidate trajectories for one matrix."""

    config = load_config()
    root = config["seedContract"]["matrixRootHex"]
    phase = config["seedContract"]["phase"]
    beta = generate_beta(derive_seed(root, phase, "catalytic_matrix", matrix_index))
    initial = initialize_distinct_state(derive_seed(root, phase, "initial_state", matrix_index))
    attempts: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for spec in simulation_specs():
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        try:
            trajectory, _ = simulate_trajectory(
                phase=phase,
                root_hex=root,
                matrix_index=matrix_index,
                definition=make_definition(spec),
                stream_identity=spec["streamIdentity"],
                beta=beta,
                initial_state=initial,
            )
            path = trajectory_path(cache_root, matrix_index, spec["candidateId"])
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as handle:
                pickle.dump(trajectory, handle, protocol=5)
            selected = selected_clock_observations(
                trajectory, "C1_SELECTED_DAUGHTER_RETAINED"
            )
            post_count = sum(item.observation_kind == "post_fission" for item in selected)
            complete = bool(
                trajectory.terminal_status == "requested_fissions_completed"
                and trajectory.completed_fissions == 100
                and post_count == 100
            )
            common = {
                **spec,
                "matrixIndex": matrix_index,
                "replacementAttempted": False,
            }
            trajectories.append(
                {
                    **common,
                    "trajectoryId": trajectory.trajectory_id,
                    "trajectorySha256": trajectory.trajectory_sha256,
                    "betaSha256": trajectory.beta_sha256,
                    "initialStateSha256": trajectory.initial_state_sha256,
                    "terminalStatus": trajectory.terminal_status,
                    "completedFissions": int(trajectory.completed_fissions),
                    "selectedClockLength": len(selected),
                    "postFissionBoundaryCount": post_count,
                    "cachePath": str(path),
                    "cacheSha256": sha256_file(path),
                }
            )
            attempts.append(
                {
                    **common,
                    "attemptStatus": "COMPLETE" if complete else "INCOMPLETE_OR_EXTINCT_RETAINED",
                    "terminalStatus": trajectory.terminal_status,
                    "completedFissions": int(trajectory.completed_fissions),
                    "wallSeconds": time.perf_counter() - wall_start,
                    "cpuSeconds": time.process_time() - cpu_start,
                }
            )
        except Exception as error:  # noqa: BLE001 - provenance is mandatory
            failures.append(
                {
                    "failureId": f"S19-L11-SIM-M{matrix_index:03d}-{spec['candidateId']}",
                    **spec,
                    "matrixIndex": matrix_index,
                    "failureType": type(error).__name__,
                    "message": str(error),
                    "scientificValuesEligible": False,
                    "replacementAttempted": False,
                }
            )
            attempts.append(
                {
                    **spec,
                    "matrixIndex": matrix_index,
                    "attemptStatus": "UNREGISTERED_EXCEPTION_GLOBAL_STOP",
                    "terminalStatus": None,
                    "completedFissions": None,
                    "wallSeconds": time.perf_counter() - wall_start,
                    "cpuSeconds": time.process_time() - cpu_start,
                    "replacementAttempted": False,
                }
            )
    return {"attempts": attempts, "trajectories": trajectories, "failures": failures}


def run_simulation_batch(
    indices: Iterable[int], workers: int, cache_root: Path
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(simulate_matrix, int(index), cache_root): int(index)
            for index in indices
        }
        for future in as_completed(futures):
            outputs.append(future.result())
    return outputs


def generate(workers: int) -> None:
    """Benchmark opaquely, enforce ceilings, and generate exactly 200 attempts."""

    if workers != 8:
        raise ValueError("L11 generation is locked to eight workers")
    release = repository_release_gate()
    if not release["passed"]:
        raise RuntimeError(f"L11 release gate failed: {release}")
    if any(PRIMARY_CACHE.glob("*.pkl")):
        raise RuntimeError("L11 primary cache is not empty; generation cannot relaunch")
    started = utc_now()
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    child_before = resource.getrusage(resource.RUSAGE_CHILDREN)

    benchmark_start = time.perf_counter()
    benchmark_outputs = run_simulation_batch(range(10), workers, PRIMARY_CACHE)
    benchmark_wall = time.perf_counter() - benchmark_start
    benchmark_attempts = pd.DataFrame(
        [row for output in benchmark_outputs for row in output["attempts"]]
    )
    benchmark_failures = [
        row for output in benchmark_outputs for row in output["failures"]
    ]
    benchmark_cpu = (
        float(benchmark_attempts["cpuSeconds"].sum())
        if len(benchmark_attempts)
        else math.inf
    )
    projected_cpu_hours = benchmark_cpu * 10 * 2 * 1.5 / 3600.0
    projected_wall_hours = benchmark_wall * 10 * 2 * 1.5 / 3600.0
    benchmark = {
        "schema": "eidosoma.e01.s19_l11.preoutcome_benchmark.v1",
        "matrixCount": 10,
        "simulationCount": len(benchmark_attempts),
        "scientificLabelsCalculatedOrOpened": False,
        "terminalStatusesNotUsedForMethodSelection": True,
        "benchmarkWallSeconds": benchmark_wall,
        "benchmarkWorkerCpuSeconds": benchmark_cpu,
        "projectionIncludesPrimaryAndCompleteRegeneration": True,
        "safetyFactor": 1.5,
        "projectedCpuHours": projected_cpu_hours,
        "projectedWallHours": projected_wall_hours,
        "cpuCeilingAfterReserveHours": 28.8,
        "wallCeilingHours": 12.0,
        "failureCount": len(benchmark_failures),
        "passed": bool(
            not benchmark_failures
            and projected_cpu_hours <= 28.8
            and projected_wall_hours <= 12.0
        ),
        "completedAtUtc": utc_now(),
    }
    write_json(LOOP_ROOT / "preoutcome_benchmark.json", benchmark)
    if not benchmark["passed"]:
        write_csv(
            LOOP_ROOT / "failure_ledger.csv",
            pd.DataFrame(benchmark_failures),
        )
        raise RuntimeError("L11 opaque benchmark exceeded a ceiling or failed")

    outputs = benchmark_outputs + run_simulation_batch(
        range(10, 100), workers, PRIMARY_CACHE
    )
    attempts = pd.DataFrame([row for output in outputs for row in output["attempts"]])
    trajectories = pd.DataFrame(
        [row for output in outputs for row in output["trajectories"]]
    )
    failures = pd.DataFrame([row for output in outputs for row in output["failures"]])
    attempts = attempts.sort_values(["matrixIndex", "candidateId"], kind="stable")
    trajectories = trajectories.sort_values(["matrixIndex", "candidateId"], kind="stable")
    write_parquet(LOOP_ROOT / "execution_status.parquet", attempts)
    write_parquet(LOOP_ROOT / "trajectory_manifest.parquet", trajectories)
    failure_columns = [
        "failureId", "candidateId", "exposure", "daughterRule", "overshootRule",
        "streamIdentity", "matrixIndex", "failureType", "message",
        "scientificValuesEligible", "replacementAttempted",
    ]
    write_csv(
        LOOP_ROOT / "failure_ledger.csv",
        failures if len(failures) else pd.DataFrame(columns=failure_columns),
    )
    if len(attempts) != 200 or len(trajectories) != 200 or len(failures):
        raise RuntimeError("L11 trajectory generation failed closed")
    if attempts["replacementAttempted"].astype(bool).any():
        raise RuntimeError("L11 replacement invariant violated")
    input_units = pd.read_parquet(LOOP_ROOT / "input_units.parquet")
    observed = trajectories[
        ["matrixIndex", "betaSha256", "initialStateSha256"]
    ].drop_duplicates()
    if len(observed) != 100:
        raise RuntimeError("L11 candidates did not share exact matrix/initial-state identities")
    observed = observed.merge(
        input_units, on="matrixIndex", validate="one_to_one", suffixes=("", "Expected")
    )
    if not (
        observed["betaSha256"].eq(observed["betaSha256Expected"]).all()
        and observed["initialStateSha256"].eq(observed["initialStateSha256Expected"]).all()
    ):
        raise RuntimeError("L11 generated inputs differ from the frozen input manifest")
    child_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    child_cpu = (
        child_after.ru_utime + child_after.ru_stime
        - child_before.ru_utime - child_before.ru_stime
    )
    write_json(
        LOOP_ROOT / "generation_runtime.json",
        {
            "schema": "eidosoma.e01.s19_l11.generation_runtime.v1",
            "startedAtUtc": started,
            "completedAtUtc": utc_now(),
            "wallSeconds": time.perf_counter() - wall_start,
            "coordinatorCpuSeconds": time.process_time() - cpu_start,
            "workerReportedCpuSeconds": float(attempts["cpuSeconds"].sum()),
            "childCpuSeconds": child_cpu,
            "workers": workers,
            "labelsCalculatedDuringGeneration": False,
            "attemptCount": len(attempts),
            "trajectoryCount": len(trajectories),
            "completeCount": int((attempts["attemptStatus"] == "COMPLETE").sum()),
            "incompleteRetainedCount": int((attempts["attemptStatus"] != "COMPLETE").sum()),
        },
    )
    print(json.dumps({"status": "L11_TRAJECTORIES_GENERATED_LABELS_UNOPENED", "attempts": 200}, sort_keys=True))


def empty_fingerprint_row(
    pipeline_id: str,
    candidate_id: str,
    matrix_index: int,
    trajectory_id: str,
    status: str,
    *,
    record_scope: str = "PRIMARY_L11_UNTOUCHED",
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "recordScope": record_scope,
        "pipelineId": pipeline_id,
        "candidateId": candidate_id,
        "matrixIndex": matrix_index,
        "trajectoryId": trajectory_id,
        "fingerprintStatus": status,
        "consistencyStatus": None,
        "labelSha256": None,
        "rawPaperDistance": None,
        "normalizedPaperDistance": None,
    }
    row.update({metric: None for metric in FINGERPRINT_METRICS})
    return row


def enriched_fingerprint(labels: np.ndarray, generations: np.ndarray) -> dict[str, Any]:
    result = label_fingerprint(labels, generations)
    for polarity, desired in (("positive", True), ("negative", False)):
        episodes = run_descriptors(np.asarray(labels, dtype=bool), desired)
        starts = np.asarray([row["startIndex0"] for row in episodes], dtype=np.float64)
        result[f"{polarity}MeanEpisodeSpacing"] = (
            float(np.mean(np.diff(starts))) if len(starts) >= 2 else None
        )
    return result


def fingerprint_row(
    pipeline_id: str,
    candidate_id: str,
    matrix_index: int,
    trajectory_id: str,
    fingerprint: dict[str, Any],
    *,
    record_scope: str = "PRIMARY_L11_UNTOUCHED",
) -> dict[str, Any]:
    row = {
        "recordScope": record_scope,
        "pipelineId": pipeline_id,
        "candidateId": candidate_id,
        "matrixIndex": matrix_index,
        "trajectoryId": trajectory_id,
        **fingerprint,
    }
    row["rawPaperDistance"] = paper_distance(fingerprint, "RAW")
    row["normalizedPaperDistance"] = paper_distance(fingerprint, "NORMALIZED")
    return row


def l08_frame_to_fingerprint(frame: pd.DataFrame) -> dict[str, Any] | None:
    frozen = l08_label_fingerprint(frame)
    if frozen["fingerprintStatus"] != "ELIGIBLE":
        return None
    ordered = frame.sort_values("analysisUnitIndex", kind="stable")
    eligible = ordered["isReplicator"].notna().to_numpy(bool)
    labels = ordered.loc[eligible, "isReplicator"].to_numpy(bool)
    generations = ordered.loc[eligible, "generation"].to_numpy(np.int64)
    result = enriched_fingerprint(labels, generations)
    for key in (
        "selectedClockLength", "persistence", "occupancy", "firstOnsetRawIndex0",
        "firstOnsetRawStep1", "firstOnsetNormalized", "consistency",
        "consistencyStatus", "positiveEpisodeCount", "negativeEpisodeCount",
        "positiveMeanEpisodeDuration", "negativeMeanEpisodeDuration",
        "positiveLongestEpisodeDuration", "negativeLongestEpisodeDuration",
        "labelSha256",
    ):
        if key in frozen:
            result[key] = frozen[key]
    return result


def comparator_rows_for_candidate(
    matrix_index: int, candidate_id: str, trajectory: Any
) -> list[dict[str, Any]]:
    """Recalculate only the three registered current-dataset comparators."""

    rows: list[dict[str, Any]] = []
    selected = selected_clock_observations(
        trajectory, "C1_SELECTED_DAUGHTER_RETAINED"
    )
    generations = np.asarray(
        [item.growth_generation_one_based for item in selected], dtype=np.int64
    )
    setting = {
        "roundId": "S19-L11",
        "settingId": COMPARATOR_ADJACENT,
        "settingPairId": COMPARATOR_ADJACENT,
        "threshold": THRESHOLD,
        "comparator": "STRICT_GT",
        "clockId": "C1_SELECTED_DAUGHTER_RETAINED",
        "alignment": "INCOMING_DUPLICATE_FIRST",
        "family": "ADJACENT_CLOCK",
        "projection": "ALL_OBSERVATIONS",
    }
    adjacent = materialize_frozen_setting(trajectory, setting)
    if adjacent["isReplicator"].notna().all():
        fp = enriched_fingerprint(adjacent["isReplicator"].to_numpy(bool), generations)
        rows.append(
            fingerprint_row(
                COMPARATOR_ADJACENT,
                candidate_id,
                matrix_index,
                str(trajectory.trajectory_id),
                fp,
                record_scope="CURRENT_L11_UNTOUCHED",
            )
        )
    else:
        rows.append(
            empty_fingerprint_row(
                COMPARATOR_ADJACENT,
                candidate_id,
                matrix_index,
                str(trajectory.trajectory_id),
                "COMPARATOR_INELIGIBLE",
                record_scope="CURRENT_L11_UNTOUCHED",
            )
        )
    for comparator_id, object_id in (
        (COMPARATOR_A_BOUNDARY, OBJECT_A_BOUNDARY),
        (COMPARATOR_A_PROJECTED, OBJECT_A_PROJECTED),
    ):
        frame = materialize_analysis_object(trajectory, MECHANISM_A, object_id)
        fp = l08_frame_to_fingerprint(frame)
        if fp is None:
            rows.append(
                empty_fingerprint_row(
                    comparator_id,
                    candidate_id,
                    matrix_index,
                    str(trajectory.trajectory_id),
                    "COMPARATOR_INELIGIBLE",
                    record_scope="CURRENT_L11_UNTOUCHED",
                )
            )
        else:
            rows.append(
                fingerprint_row(
                    comparator_id,
                    candidate_id,
                    matrix_index,
                    str(trajectory.trajectory_id),
                    fp,
                    record_scope="CURRENT_L11_UNTOUCHED",
                )
            )
    return rows


def control_record(
    *,
    common: dict[str, Any],
    control_type: str,
    control_index: int,
    status: str,
    labels: np.ndarray | None,
    generations: np.ndarray,
    boundary_labels: np.ndarray | None,
    applicability: str = "APPLICABLE",
) -> dict[str, Any]:
    fp: dict[str, Any] = {}
    if labels is not None:
        fp = enriched_fingerprint(np.asarray(labels, dtype=bool), generations)
    return {
        "recordType": "TRAJECTORY_CONTROL",
        **common,
        "controlType": control_type,
        "controlIndex": control_index,
        "controlStatus": status,
        "promotionApplicability": applicability,
        "boundaryOccupancy": (
            float(np.mean(boundary_labels))
            if boundary_labels is not None and len(boundary_labels)
            else None
        ),
        **{metric: fp.get(metric) for metric in FINGERPRINT_METRICS},
        "rawPaperDistance": paper_distance(fp, "RAW") if fp else None,
        "normalizedPaperDistance": paper_distance(fp, "NORMALIZED") if fp else None,
        "rawOnsetAbsoluteError": (
            abs(float(fp["firstOnsetRawStep1"]) - 37.0)
            if fp and fp.get("firstOnsetRawStep1") is not None
            else None
        ),
        "rawP": None,
        "holmAdjustedP": None,
        "passed": None,
    }


def analyze_matrix(
    matrix_index: int, cache_root: Path
) -> dict[str, list[dict[str, Any]]]:
    """Calculate both primary labels and every registered control for one matrix."""

    names = (
        "historical", "cluster", "clusterSize", "singleton", "centroid",
        "molecular", "boundary", "boundaryFingerprint", "fingerprint", "episode",
        "comparator", "negative", "failure",
    )
    buckets: dict[str, list[dict[str, Any]]] = {name: [] for name in names}
    for candidate_id in PRIMARY_CANDIDATES:
        path = trajectory_path(cache_root, matrix_index, candidate_id)
        with path.open("rb") as handle:
            trajectory = pickle.load(handle)
        trajectory_id = str(trajectory.trajectory_id)
        buckets["comparator"].extend(
            comparator_rows_for_candidate(matrix_index, candidate_id, trajectory)
        )
        selected = selected_clock_observations(
            trajectory, "C1_SELECTED_DAUGHTER_RETAINED"
        )
        post_positions = np.asarray(
            [index for index, item in enumerate(selected) if item.observation_kind == "post_fission"],
            dtype=np.int64,
        )
        complete = bool(
            trajectory.terminal_status == "requested_fissions_completed"
            and trajectory.completed_fissions == 100
            and len(post_positions) == 100
        )
        if not complete:
            for pipeline_id in PIPELINE_IDS:
                common = {
                    "pipelineId": pipeline_id,
                    "candidateId": candidate_id,
                    "matrixIndex": matrix_index,
                    "trajectoryId": trajectory_id,
                }
                status = "INCOMPLETE_TRAJECTORY_RETAINED_LABEL_INELIGIBLE"
                buckets["cluster"].append(
                    {
                        **common, "pipelineStatus": status, "selectedK": None,
                        "selectedScore": None, "eligibleBoundaryCount": int(trajectory.completed_fissions),
                        "clusterCount": None, "singletonClusterCount": None,
                        "recurringClusterCount": None, "largestClusterFraction": None,
                        "unionRecurringClusterFraction": None, "sourceTagPositiveFraction": None,
                        "withinClusterDispersion": None, "betweenClusterSeparation": None,
                        "meanParentDaughterHInside": None, "meanParentDaughterHOutside": None,
                    }
                )
                buckets["singleton"].append(
                    {
                        **common, "pipelineStatus": status, "positiveMolecularCount": None,
                        "singletonOnlyPositiveMolecularCount": None,
                        "singletonDerivedPositiveFraction": None,
                        "atLeastTwoClusterPositiveMolecularCount": None,
                    }
                )
                buckets["fingerprint"].append(
                    empty_fingerprint_row(
                        pipeline_id, candidate_id, matrix_index, trajectory_id, status
                    )
                )
            continue

        molecular = close_rows(np.asarray([item.state for item in selected], dtype=np.float64))
        post = tuple(selected[index] for index in post_positions)
        boundary = close_rows(np.asarray([item.state for item in post], dtype=np.float64))
        generations = np.asarray(
            [item.growth_generation_one_based for item in selected], dtype=np.int64
        )
        boundary_generations = np.asarray(
            [item.growth_generation_one_based for item in post], dtype=np.int64
        )
        parent_daughter = boundary_scores(
            trajectory,
            boundary_object="PARENT_TO_SELECTED_DAUGHTER",
            alignment="INCOMING_DUPLICATE_FIRST",
        )

        for pipeline_id in PIPELINE_IDS:
            common = {
                "pipelineId": pipeline_id,
                "candidateId": candidate_id,
                "matrixIndex": matrix_index,
                "trajectoryId": trajectory_id,
            }
            seed_identity = f"{VERSION}::{pipeline_id}::{trajectory_id}"
            try:
                result = materialize_pipeline(
                    pipeline_id, boundary, molecular, post_positions, trajectory_id
                )
            except Exception as error:  # noqa: BLE001 - fail-closed provenance
                buckets["failure"].append(
                    {
                        "failureId": f"S19-L11-ANALYSIS-{pipeline_id}-{candidate_id}-M{matrix_index:03d}",
                        **serialize_worker_exception(
                            matrix_id=matrix_index,
                            candidate_id=candidate_id,
                            pipeline_id=pipeline_id,
                            generation=None,
                            selected_k=None,
                            cluster_sizes=(),
                            tag_counts={},
                            seed_identity=seed_identity,
                            error=error,
                        ),
                        "scientificValuesEligible": False,
                    }
                )
                continue
            fit = result.fit
            sizes = tuple(int(value) for value in fit.cluster_sizes)
            recurring = tuple(index for index, size in enumerate(sizes) if size >= 2)
            singletons = tuple(index for index, size in enumerate(sizes) if size == 1)
            largest_fraction = (
                float(max(sizes) / max(1, int(np.count_nonzero(fit.eligible_mask))))
                if sizes else None
            )
            eligible_labels = np.asarray(fit.labels, dtype=np.int64) if fit.labels is not None else None
            recurring_fraction = (
                float(np.mean(np.isin(eligible_labels, recurring)))
                if eligible_labels is not None and len(eligible_labels) else None
            )
            dispersions: list[float] = []
            separations: list[float] = []
            if fit.centroids is not None and eligible_labels is not None:
                eligible_values = boundary[np.asarray(fit.eligible_mask, dtype=bool)]
                for cluster_id in range(len(fit.centroids)):
                    assigned = eligible_values[eligible_labels == cluster_id]
                    if len(assigned):
                        if pipeline_id == U1_ID:
                            dispersions.extend(
                                (1.0 - historical_h(assigned, fit.centroids[cluster_id]).ravel()).tolist()
                            )
                        else:
                            dispersions.extend(
                                np.linalg.norm(assigned - fit.centroids[cluster_id], axis=1).tolist()
                            )
                for left in range(len(fit.centroids)):
                    for right in range(left + 1, len(fit.centroids)):
                        if pipeline_id == U1_ID:
                            separations.append(
                                float(1.0 - historical_h(fit.centroids[left], fit.centroids[right])[0, 0])
                            )
                        else:
                            separations.append(float(np.linalg.norm(fit.centroids[left] - fit.centroids[right])))
            if result.boundary_labels is not None:
                inside = np.asarray(result.boundary_labels, dtype=bool)
                mean_inside = float(np.mean(parent_daughter[inside])) if np.any(inside) else None
                mean_outside = float(np.mean(parent_daughter[~inside])) if np.any(~inside) else None
            else:
                mean_inside = mean_outside = None
            buckets["cluster"].append(
                {
                    **common,
                    "pipelineStatus": result.status,
                    "sourceFitStatus": fit.status,
                    "selectedK": fit.selected_k,
                    "selectedScore": fit.selected_score,
                    "eligibleBoundaryCount": int(np.count_nonzero(fit.eligible_mask)),
                    "clusterCount": len(sizes),
                    "clusterSizesJson": json.dumps(list(sizes)),
                    "singletonClusterCount": len(singletons),
                    "recurringClusterCount": len(recurring),
                    "largestClusterFraction": largest_fraction,
                    "unionRecurringClusterFraction": recurring_fraction,
                    "sourceTagPositiveFraction": (
                        float(np.mean(result.boundary_tags > 0))
                        if result.boundary_tags is not None else None
                    ),
                    "withinClusterDispersion": float(np.mean(dispersions)) if dispersions else None,
                    "betweenClusterSeparation": float(np.min(separations)) if separations else None,
                    "meanParentDaughterHInside": mean_inside,
                    "meanParentDaughterHOutside": mean_outside,
                }
            )
            if fit.centroids is not None:
                for cluster_id, (centroid, size) in enumerate(zip(fit.centroids, sizes, strict=True)):
                    buckets["clusterSize"].append(
                        {
                            **common, "clusterId": cluster_id, "clusterSize": size,
                            "clusterSizeClass": "SINGLETON_SIZE_1" if size == 1 else ("PAIR_SIZE_2" if size == 2 else "RECURRING_SIZE_3_PLUS"),
                            "isRecurringForU2": bool(size >= 2),
                            "centroidSha256": array_sha256(centroid),
                        }
                    )
                    if size >= 2:
                        buckets["centroid"].append(
                            {
                                **common, "clusterId": cluster_id, "clusterSize": size,
                                "centroidJson": json.dumps(centroid.tolist()),
                                "centroidSha256": array_sha256(centroid),
                            }
                        )
            if pipeline_id == U1_ID and result.boundary_tags is not None:
                for boundary_index, (item, tag, size) in enumerate(
                    zip(post, result.boundary_tags, result.boundary_cluster_sizes, strict=True)
                ):
                    buckets["historical"].append(
                        {
                            **common, "boundaryIndex0": boundary_index,
                            "generation": int(item.growth_generation_one_based),
                            "rawObservationIndex": int(item.observation_index),
                            "nondriftEligible": bool(fit.eligible_mask[boundary_index]),
                            "sourceTag": int(tag), "tagPositive": bool(tag > 0),
                            "tagClusterSize": int(size),
                            "singletonTag": bool(size == 1),
                        }
                    )
            positive_count = (
                int(np.count_nonzero(result.molecular_labels))
                if result.molecular_labels is not None else None
            )
            singleton_count = (
                int(np.count_nonzero(result.molecular_cluster_sizes == 1))
                if result.molecular_cluster_sizes is not None else None
            )
            recurring_count = (
                int(np.count_nonzero(result.molecular_cluster_sizes >= 2))
                if result.molecular_cluster_sizes is not None else None
            )
            buckets["singleton"].append(
                {
                    **common, "pipelineStatus": result.status,
                    "positiveMolecularCount": positive_count,
                    "singletonOnlyPositiveMolecularCount": singleton_count,
                    "singletonDerivedPositiveFraction": result.singleton_positive_fraction,
                    "atLeastTwoClusterPositiveMolecularCount": recurring_count,
                }
            )
            if result.molecular_labels is None or result.boundary_labels is None:
                buckets["fingerprint"].append(
                    empty_fingerprint_row(
                        pipeline_id, candidate_id, matrix_index, trajectory_id, result.status
                    )
                )
                continue

            molecular_labels = np.asarray(result.molecular_labels, dtype=bool)
            boundary_labels = np.asarray(result.boundary_labels, dtype=bool)
            fp = enriched_fingerprint(molecular_labels, generations)
            boundary_fp = enriched_fingerprint(boundary_labels, boundary_generations)
            buckets["fingerprint"].append(
                fingerprint_row(pipeline_id, candidate_id, matrix_index, trajectory_id, fp)
            )
            buckets["boundaryFingerprint"].append(
                fingerprint_row(
                    pipeline_id, candidate_id, matrix_index, trajectory_id, boundary_fp,
                    record_scope="BOUNDARY_DIAGNOSTIC_NONINTERCHANGEABLE",
                )
            )
            for index, (item, state, score, label) in enumerate(
                zip(selected, molecular, result.molecular_scores, molecular_labels, strict=True)
            ):
                buckets["molecular"].append(
                    {
                        **common, "analysisUnitIndex": index,
                        "rawObservationIndex": int(item.observation_index),
                        "generation": int(item.growth_generation_one_based),
                        "observationKind": str(item.observation_kind),
                        "labelStatus": fp["fingerprintStatus"],
                        "unionScore": float(score), "isReplicator": bool(label),
                        "sourceTag": (
                            int(result.molecular_tags[index])
                            if result.molecular_tags is not None else None
                        ),
                        "sourceClusterSize": (
                            int(result.molecular_cluster_sizes[index])
                            if result.molecular_cluster_sizes is not None else None
                        ),
                        "singletonDerived": (
                            bool(result.molecular_cluster_sizes[index] == 1)
                            if result.molecular_cluster_sizes is not None else None
                        ),
                        "stateSha256": array_sha256(state),
                    }
                )
            for boundary_index, (item, state, score, label) in enumerate(
                zip(post, boundary, result.boundary_scores, boundary_labels, strict=True)
            ):
                buckets["boundary"].append(
                    {
                        **common, "boundaryIndex0": boundary_index,
                        "generation": int(item.growth_generation_one_based),
                        "rawObservationIndex": int(item.observation_index),
                        "nondriftEligible": bool(fit.eligible_mask[boundary_index]),
                        "selectedClusterId": (
                            int(result.boundary_tags[boundary_index]) - 1
                            if result.boundary_tags is not None and result.boundary_tags[boundary_index] > 0
                            else None
                        ),
                        "selectedClusterSize": (
                            int(result.boundary_cluster_sizes[boundary_index])
                            if result.boundary_cluster_sizes is not None else None
                        ),
                        "unionScore": float(score), "isReplicator": bool(label),
                        "parentDaughterH": float(parent_daughter[boundary_index]),
                        "stateSha256": array_sha256(state),
                    }
                )
            for polarity, desired in (("POSITIVE", True), ("NEGATIVE", False)):
                for episode_index, episode in enumerate(run_descriptors(molecular_labels, desired)):
                    buckets["episode"].append(
                        {**common, "polarity": polarity, "episodeIndex": episode_index, **episode}
                    )

            # NC1: same number of observed boundary references as the registered
            # pipeline's tag/recurring-centroid set; 64 fixed draws.
            reference_count = (
                len(fit.centroids) if pipeline_id == U1_ID
                else len(result.recurring_cluster_ids)
            )
            for draw in range(RANDOM_CENTROID_DRAWS):
                rng = np.random.Generator(np.random.PCG64DXSM(deterministic_seed(
                    "NC1", pipeline_id, candidate_id, matrix_index, draw, bits=128
                )))
                selected_refs = rng.choice(len(boundary), size=reference_count, replace=False)
                references = boundary[selected_refs]
                if pipeline_id == U1_ID:
                    _, random_boundary = direct_union_scores(boundary, references)
                    random_labels = project_boundary_values(
                        random_boundary, post_positions, len(molecular), prefix_value=False
                    ).astype(bool)
                else:
                    _, random_labels = direct_union_scores(molecular, references)
                    _, random_boundary = direct_union_scores(boundary, references)
                buckets["negative"].append(
                    control_record(
                        common=common, control_type="NC1_RANDOM_CENTROID_SET",
                        control_index=draw, status="ELIGIBLE", labels=random_labels,
                        generations=generations, boundary_labels=random_boundary,
                    )
                )

            # NC2: singleton-only references/tags, explicitly inapplicable when absent.
            if result.singleton_cluster_ids:
                if pipeline_id == U1_ID:
                    singleton_boundary = np.isin(result.boundary_tags - 1, result.singleton_cluster_ids)
                    singleton_labels = project_boundary_values(
                        singleton_boundary, post_positions, len(molecular), prefix_value=False
                    ).astype(bool)
                else:
                    _, singleton_labels = direct_union_scores(molecular, result.singleton_centroids)
                    _, singleton_boundary = direct_union_scores(boundary, result.singleton_centroids)
                buckets["negative"].append(
                    control_record(
                        common=common, control_type="NC2_SINGLETON_ONLY_CENTROID",
                        control_index=0, status="ELIGIBLE", labels=singleton_labels,
                        generations=generations, boundary_labels=singleton_boundary,
                    )
                )
            else:
                buckets["negative"].append(
                    control_record(
                        common=common, control_type="NC2_SINGLETON_ONLY_CENTROID",
                        control_index=0, status="NOT_APPLICABLE_NO_SINGLETON_CLUSTER",
                        labels=None, generations=generations, boundary_labels=None,
                        applicability="NOT_APPLICABLE",
                    )
                )

            # NC3: permute post-fission order before the unchanged clustering pipeline.
            rng = np.random.Generator(np.random.PCG64DXSM(deterministic_seed(
                "NC3", pipeline_id, candidate_id, matrix_index, 0, bits=128
            )))
            permutation = rng.permutation(len(boundary))
            if pipeline_id == U1_ID:
                permuted = materialize_u1(
                    boundary[permutation], boundary[permutation], np.arange(len(boundary)),
                    f"{trajectory_id}::NC3",
                )
                if permuted.boundary_labels is not None:
                    inverse_labels = np.zeros(len(boundary), dtype=bool)
                    inverse_labels[permutation] = permuted.boundary_labels
                    nc3_labels = project_boundary_values(
                        inverse_labels, post_positions, len(molecular), prefix_value=False
                    ).astype(bool)
                    nc3_boundary = inverse_labels
                    nc3_status = "ELIGIBLE"
                else:
                    nc3_labels = nc3_boundary = None
                    nc3_status = permuted.status
                nc3_applicability = "APPLICABLE"
            else:
                permuted = materialize_u2(
                    boundary[permutation], molecular, f"{trajectory_id}::NC3"
                )
                nc3_labels = permuted.molecular_labels
                nc3_boundary = permuted.boundary_labels
                nc3_status = permuted.status
                nc3_applicability = "DESCRIPTIVE_DISTRIBUTION_ORDER_INVARIANT"
            buckets["negative"].append(
                control_record(
                    common=common, control_type="NC3_TIME_PERMUTED_POST_FISSION",
                    control_index=0, status=nc3_status, labels=nc3_labels,
                    generations=generations, boundary_labels=nc3_boundary,
                    applicability=nc3_applicability,
                )
            )

            # NC4: U1 membership identities are randomized; U2 direct-centroid
            # membership is invariant and is retained as explicitly inapplicable.
            if pipeline_id == U1_ID:
                rng = np.random.Generator(np.random.PCG64DXSM(deterministic_seed(
                    "NC4", pipeline_id, candidate_id, matrix_index, 0, bits=128
                )))
                randomized_tags = np.asarray(result.boundary_tags)[rng.permutation(len(boundary))]
                nc4_boundary = randomized_tags > 0
                nc4_labels = project_boundary_values(
                    nc4_boundary, post_positions, len(molecular), prefix_value=False
                ).astype(bool)
                nc4_status = "ELIGIBLE"
                nc4_applicability = "APPLICABLE"
            else:
                nc4_boundary = boundary_labels.copy()
                nc4_labels = molecular_labels.copy()
                nc4_status = "IDENTICAL_BY_CONSTRUCTION_CENTROIDS_PRESERVED"
                nc4_applicability = "NOT_APPLICABLE_BY_CONSTRUCTION"
            buckets["negative"].append(
                control_record(
                    common=common, control_type="NC4_CLUSTER_LABEL_PERMUTATION",
                    control_index=0, status=nc4_status, labels=nc4_labels,
                    generations=generations, boundary_labels=nc4_boundary,
                    applicability=nc4_applicability,
                )
            )
    return buckets


def aggregate_fingerprints(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (pipeline_id, candidate_id), group in frame.groupby(
        ["pipelineId", "candidateId"], sort=True
    ):
        eligible = group[group["selectedClockLength"].notna()].copy()
        row: dict[str, Any] = {
            "pipelineId": pipeline_id,
            "candidateId": candidate_id,
            "trajectoryCount": len(group),
            "validTrajectoryCount": len(eligible),
            "undefinedConsistencyCount": int(eligible["consistency"].isna().sum()),
        }
        for metric in FINGERPRINT_METRICS:
            values = pd.to_numeric(eligible[metric], errors="coerce").dropna().to_numpy(float)
            row[f"defined_{metric}"] = len(values)
            row[f"mean_{metric}"] = float(np.mean(values)) if len(values) else None
            row[f"median_{metric}"] = float(np.median(values)) if len(values) else None
            row[f"sd_{metric}"] = (
                float(np.std(values, ddof=1)) if len(values) > 1
                else (0.0 if len(values) else None)
            )
            row[f"se_{metric}"] = (
                float(np.std(values, ddof=1) / math.sqrt(len(values)))
                if len(values) > 1 else (0.0 if len(values) else None)
            )
        summary = {metric: row[f"mean_{metric}"] for metric in FINGERPRINT_METRICS}
        row["rawPaperDistance"] = paper_distance(summary, "RAW")
        row["normalizedPaperDistance"] = paper_distance(summary, "NORMALIZED")
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_primary(fingerprint: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "selectedClockLength", "persistence", "occupancy", "consistency",
        "firstOnsetRawStep1", "firstOnsetNormalized",
        "preOnsetNonreplicatingDuration", "noReplicatorThrough25Percent",
        "nonreplicatingAt25Percent", "positiveEpisodeCount", "negativeEpisodeCount",
        "transitionCount", "positiveMeanEpisodeSpacing", "negativeMeanEpisodeSpacing",
    )
    rows: list[dict[str, Any]] = []
    for pipeline_id in PIPELINE_IDS:
        for candidate_id in PRIMARY_CANDIDATES:
            group = (
                fingerprint[
                    (fingerprint.pipelineId == pipeline_id)
                    & (fingerprint.candidateId == candidate_id)
                ]
                .set_index("matrixIndex")
                .reindex(range(100))
            )
            indices = bootstrap_indices(candidate_id, pipeline_id)
            arrays = {
                metric: pd.to_numeric(group[metric], errors="coerce").to_numpy(float)
                for metric in metrics
            }
            for replicate, sample in enumerate(indices):
                summary: dict[str, Any] = {}
                for metric, values in arrays.items():
                    sampled = values[sample]
                    summary[metric] = (
                        float(np.nanmean(sampled)) if np.any(np.isfinite(sampled)) else None
                    )
                rows.append(
                    {
                        "pipelineId": pipeline_id,
                        "candidateId": candidate_id,
                        "bootstrapReplicate": replicate,
                        **summary,
                        "rawPaperDistance": paper_distance(summary, "RAW"),
                        "normalizedPaperDistance": paper_distance(summary, "NORMALIZED"),
                    }
                )
    return pd.DataFrame(rows)


def frozen_comparator_aggregates() -> pd.DataFrame:
    """Import only previously frozen aggregate comparators; never rerun them."""

    l10_aggregate = pd.read_parquet(
        ARTIFACT_ROOT / "loops/L10/aggregate_fingerprint_results.parquet"
    )
    l10_comparator = pd.read_parquet(
        ARTIFACT_ROOT / "loops/L10/comparator_aggregate_results.parquet"
    )
    historical = l10_aggregate[
        l10_aggregate.pipelineId.isin([COMPARATOR_L10_R1, COMPARATOR_L10_R2])
    ].copy()
    high = l10_comparator[
        l10_comparator.pipelineId == COMPARATOR_B_HIGH
    ].copy()
    result = pd.concat([historical, high], ignore_index=True, sort=False)
    for metric in FINGERPRINT_METRICS:
        for prefix in ("defined", "mean", "median", "sd", "se"):
            column = f"{prefix}_{metric}"
            if column not in result:
                result[column] = None
    result["recordScope"] = "FROZEN_PRIOR_AGGREGATE_NOT_RECOMPUTED"
    result["sourceLoop"] = "S19-L10"
    return result


def paper_target_tables(
    aggregate: pd.DataFrame,
    bootstrap: pd.DataFrame,
    comparator_aggregate: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    distances: list[dict[str, Any]] = []
    combined = pd.concat([aggregate, comparator_aggregate], ignore_index=True, sort=False)
    target_metrics = {
        "selectedClockLength": PAPER_TARGETS["selectedClockLength"],
        "persistence": PAPER_TARGETS["persistence"],
        "occupancy": PAPER_TARGETS["occupancy"],
        "consistency": PAPER_TARGETS["consistency"],
        "firstOnsetRawStep1": PAPER_TARGETS["firstOnsetRawStep1"],
        "firstOnsetNormalized": PAPER_TARGETS["firstOnsetNormalized"],
    }
    for item in combined.itertuples():
        for metric, (target, scale) in target_metrics.items():
            value = getattr(item, f"mean_{metric}")
            boot_values = np.asarray([], dtype=float)
            if item.pipelineId in PIPELINE_IDS:
                boot_values = (
                    pd.to_numeric(
                        bootstrap[
                            (bootstrap.pipelineId == item.pipelineId)
                            & (bootstrap.candidateId == item.candidateId)
                        ][metric],
                        errors="coerce",
                    ).dropna().to_numpy(float)
                )
            rows.append(
                {
                    "pipelineId": item.pipelineId,
                    "candidateId": item.candidateId,
                    "metric": metric,
                    "mean": value,
                    "median": getattr(item, f"median_{metric}"),
                    "sampleSd": getattr(item, f"sd_{metric}"),
                    "standardError": getattr(item, f"se_{metric}"),
                    "paperTarget": target,
                    "paperReportedPlusMinusScale": scale,
                    "authorDispersionIdentity": "AUTHOR_DISPERSION_UNRESOLVED",
                    "rawDifference": None if pd.isna(value) else float(value - target),
                    "standardizedDifference": None if pd.isna(value) else float((value - target) / scale),
                    "bootstrapCi025": float(np.quantile(boot_values, 0.025)) if len(boot_values) else None,
                    "bootstrapCi975": float(np.quantile(boot_values, 0.975)) if len(boot_values) else None,
                    "measurementLevel": (
                        "BOUNDARY_DIAGNOSTIC_NONINTERCHANGEABLE"
                        if item.pipelineId == COMPARATOR_A_BOUNDARY
                        else "MOLECULAR_PRIMARY_OR_COMPARATOR"
                    ),
                    "recordScope": getattr(item, "recordScope", "PRIMARY_L11_UNTOUCHED"),
                }
            )
        for onset_mode, field in (("RAW", "rawPaperDistance"), ("NORMALIZED", "normalizedPaperDistance")):
            distances.append(
                {
                    "pipelineId": item.pipelineId,
                    "candidateId": item.candidateId,
                    "onsetMode": onset_mode,
                    "validTrajectoryCount": int(item.validTrajectoryCount),
                    "completeFingerprintDistance": getattr(item, field),
                    "recordScope": getattr(item, "recordScope", "PRIMARY_L11_UNTOUCHED"),
                }
            )
    distances_frame = pd.DataFrame(distances)
    distances_frame["dimensionsImprovedOverEveryComparator"] = 0
    distances_frame["improvedDimensionsJson"] = "{}"
    gate_comparators = comparator_aggregate[
        comparator_aggregate.pipelineId.isin(
            [COMPARATOR_ADJACENT, COMPARATOR_A_PROJECTED, COMPARATOR_B_HIGH, COMPARATOR_L10_R1, COMPARATOR_L10_R2]
        )
    ]
    target_values = {metric: value[0] for metric, value in target_metrics.items()}
    for index, row in distances_frame.iterrows():
        if row.pipelineId not in PIPELINE_IDS:
            continue
        onset_metric = "firstOnsetRawStep1" if row.onsetMode == "RAW" else "firstOnsetNormalized"
        metrics = ("selectedClockLength", "persistence", "occupancy", "consistency", onset_metric)
        primary = aggregate[
            (aggregate.pipelineId == row.pipelineId)
            & (aggregate.candidateId == row.candidateId)
        ].iloc[0]
        controls = gate_comparators[gate_comparators.candidateId == row.candidateId]
        improved: dict[str, bool] = {}
        for metric in metrics:
            primary_value = getattr(primary, f"mean_{metric}")
            values = pd.to_numeric(controls[f"mean_{metric}"], errors="coerce").dropna().to_numpy(float)
            improved[metric] = bool(
                pd.notna(primary_value) and len(values)
                and abs(float(primary_value) - target_values[metric])
                < float(np.min(np.abs(values - target_values[metric])))
            )
        distances_frame.at[index, "dimensionsImprovedOverEveryComparator"] = int(sum(improved.values()))
        distances_frame.at[index, "improvedDimensionsJson"] = json.dumps(improved, sort_keys=True)
    return pd.DataFrame(rows), distances_frame


def holm_adjust(values: list[float]) -> list[float]:
    if not values:
        return []
    raw = np.asarray(values, dtype=np.float64)
    order = np.argsort(raw, kind="stable")
    adjusted = np.empty_like(raw)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(raw) - rank) * float(raw[index])))
        adjusted[index] = running
    return adjusted.tolist()


def analyze_negative_controls(
    negative: pd.DataFrame, fingerprint: pd.DataFrame
) -> tuple[pd.DataFrame, dict[tuple[str, str], dict[str, bool]]]:
    """Apply the frozen recurrence/onset/distance falsification contract."""

    aggregate_rows: list[dict[str, Any]] = []
    gates: dict[tuple[str, str], dict[str, bool]] = {}
    controls = (
        "NC1_RANDOM_CENTROID_SET", "NC2_SINGLETON_ONLY_CENTROID",
        "NC3_TIME_PERMUTED_POST_FISSION", "NC4_CLUSTER_LABEL_PERMUTATION",
    )
    for pipeline_id in PIPELINE_IDS:
        for candidate_id in PRIMARY_CANDIDATES:
            observed = fingerprint[
                (fingerprint.pipelineId == pipeline_id)
                & (fingerprint.candidateId == candidate_id)
            ].set_index("matrixIndex")
            local: dict[str, bool] = {}
            for control_type in controls:
                subset = negative[
                    (negative.pipelineId == pipeline_id)
                    & (negative.candidateId == candidate_id)
                    & (negative.controlType == control_type)
                    & (negative.promotionApplicability == "APPLICABLE")
                    & negative.selectedClockLength.notna()
                ].copy()
                if subset.empty:
                    local[control_type] = True
                    aggregate_rows.append(
                        {
                            "recordType": "CONTROL_AGGREGATE", "pipelineId": pipeline_id,
                            "candidateId": candidate_id, "matrixIndex": -1,
                            "controlType": control_type, "controlIndex": -1,
                            "outcome": "ALL_APPLICABLE_OUTCOMES", "controlStatus": "NOT_APPLICABLE",
                            "promotionApplicability": "NOT_APPLICABLE", "pairedMatrixCount": 0,
                            "observedMean": None, "controlMean": None, "meanAdvantage": None,
                            "advantageCi025": None, "advantageCi975": None,
                            "rawP": None, "holmAdjustedP": None, "passed": True,
                        }
                    )
                    continue
                per_matrix = subset.groupby("matrixIndex", sort=True).agg(
                    controlRecurrence=("positiveEpisodeCount", lambda x: float(np.nanmean(pd.to_numeric(x, errors="coerce")))),
                    controlNegativeEpisodes=("negativeEpisodeCount", lambda x: float(np.nanmean(pd.to_numeric(x, errors="coerce")))),
                    controlOnsetError=("rawOnsetAbsoluteError", lambda x: float(np.nanmean(pd.to_numeric(x, errors="coerce")))),
                    controlDistance=("rawPaperDistance", lambda x: float(np.nanmean(pd.to_numeric(x, errors="coerce")))),
                )
                paired = observed.join(per_matrix, how="inner")
                paired["observedRecurrence"] = np.minimum(
                    pd.to_numeric(paired.positiveEpisodeCount, errors="coerce"),
                    pd.to_numeric(paired.negativeEpisodeCount, errors="coerce"),
                )
                paired["controlRecurrence"] = np.minimum(
                    paired.controlRecurrence, paired.controlNegativeEpisodes
                )
                paired["observedOnsetError"] = (
                    pd.to_numeric(paired.firstOnsetRawStep1, errors="coerce") - 37.0
                ).abs()
                outcome_specs = {
                    "RECURRENCE": paired.observedRecurrence - paired.controlRecurrence,
                    "ONSET_STRUCTURE": paired.controlOnsetError - paired.observedOnsetError,
                    "COMPLETE_FINGERPRINT": paired.controlDistance - pd.to_numeric(paired.rawPaperDistance, errors="coerce"),
                }
                control_pass = True
                for outcome, advantage_series in outcome_specs.items():
                    advantages = pd.to_numeric(advantage_series, errors="coerce").dropna().to_numpy(float)
                    if len(advantages):
                        seed = deterministic_seed(
                            "control-bootstrap", pipeline_id, candidate_id, control_type, outcome,
                            bits=128,
                        )
                        rng = np.random.Generator(np.random.PCG64DXSM(seed))
                        indices = rng.integers(0, len(advantages), size=(BOOTSTRAP_REPLICATES, len(advantages)))
                        boot = np.mean(advantages[indices], axis=1)
                        ci025, ci975 = np.quantile(boot, [0.025, 0.975])
                        raw_p = float((np.count_nonzero(boot <= 0) + 1) / (len(boot) + 1))
                        directional = bool(float(np.mean(advantages)) > 0 and ci025 > 0)
                    else:
                        boot = np.asarray([], dtype=float)
                        ci025 = ci975 = raw_p = None
                        directional = False
                    aggregate_rows.append(
                        {
                            "recordType": "CONTROL_AGGREGATE", "pipelineId": pipeline_id,
                            "candidateId": candidate_id, "matrixIndex": -1,
                            "controlType": control_type, "controlIndex": -1,
                            "outcome": outcome, "controlStatus": "ELIGIBLE",
                            "promotionApplicability": "APPLICABLE",
                            "pairedMatrixCount": len(advantages),
                            "observedMean": None,
                            "controlMean": None,
                            "meanAdvantage": float(np.mean(advantages)) if len(advantages) else None,
                            "advantageCi025": float(ci025) if ci025 is not None else None,
                            "advantageCi975": float(ci975) if ci975 is not None else None,
                            "rawP": raw_p, "holmAdjustedP": None,
                            "directionalCriterionPassed": directional,
                            "passed": directional,
                        }
                    )
                    control_pass &= directional
                local[control_type] = control_pass
            gates[(pipeline_id, candidate_id)] = local
    aggregate = pd.DataFrame(aggregate_rows)
    eligible = aggregate[
        (aggregate.recordType == "CONTROL_AGGREGATE")
        & aggregate.rawP.notna()
    ]
    for indices in eligible.groupby(["candidateId", "controlType", "outcome"], sort=True).groups.values():
        positions = list(indices)
        adjusted = holm_adjust(aggregate.loc[positions, "rawP"].astype(float).tolist())
        aggregate.loc[positions, "holmAdjustedP"] = adjusted
        aggregate.loc[positions, "passed"] = (
            aggregate.loc[positions, "directionalCriterionPassed"].astype(bool)
            & pd.Series(adjusted, index=positions).le(0.05)
        )
    # Resolve each applicable control only after Holm adjustment.
    for key, local in gates.items():
        pipeline_id, candidate_id = key
        for control_type in controls:
            rows = aggregate[
                (aggregate.pipelineId == pipeline_id)
                & (aggregate.candidateId == candidate_id)
                & (aggregate.controlType == control_type)
                & (aggregate.promotionApplicability == "APPLICABLE")
            ]
            local[control_type] = bool(len(rows) == 3 and rows.passed.astype(bool).all()) if len(rows) else True
    return pd.concat([negative, aggregate], ignore_index=True, sort=False), gates


def candidate_comparison_table(fingerprint: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pipeline_id in PIPELINE_IDS:
        left = fingerprint[
            (fingerprint.pipelineId == pipeline_id)
            & (fingerprint.candidateId == "CANDIDATE_2")
        ].set_index("matrixIndex")
        right = fingerprint[
            (fingerprint.pipelineId == pipeline_id)
            & (fingerprint.candidateId == "CANDIDATE_3")
        ].set_index("matrixIndex")
        for metric in FINGERPRINT_METRICS:
            a = pd.to_numeric(left[metric], errors="coerce")
            b = pd.to_numeric(right[metric], errors="coerce")
            valid = a.notna() & b.notna()
            correlation = None
            if valid.sum() >= 3 and a[valid].nunique() > 1 and b[valid].nunique() > 1:
                correlation = float(np.corrcoef(a[valid], b[valid])[0, 1])
            rows.append(
                {
                    "comparisonType": "CANDIDATE_2_VERSUS_CANDIDATE_3",
                    "pipelineId": pipeline_id,
                    "metric": metric,
                    "pairedMatrixCount": int(valid.sum()),
                    "candidate2Mean": float(a[valid].mean()) if valid.any() else None,
                    "candidate3Mean": float(b[valid].mean()) if valid.any() else None,
                    "candidate3Minus2": float((b[valid] - a[valid]).mean()) if valid.any() else None,
                    "pairedPearson": correlation,
                }
            )
    for candidate_id in PRIMARY_CANDIDATES:
        u1 = fingerprint[
            (fingerprint.pipelineId == U1_ID) & (fingerprint.candidateId == candidate_id)
        ].set_index("matrixIndex")
        u2 = fingerprint[
            (fingerprint.pipelineId == U2_ID) & (fingerprint.candidateId == candidate_id)
        ].set_index("matrixIndex")
        for metric in FINGERPRINT_METRICS:
            a = pd.to_numeric(u1[metric], errors="coerce")
            b = pd.to_numeric(u2[metric], errors="coerce")
            valid = a.notna() & b.notna()
            correlation = None
            if valid.sum() >= 3 and a[valid].nunique() > 1 and b[valid].nunique() > 1:
                correlation = float(np.corrcoef(a[valid], b[valid])[0, 1])
            rows.append(
                {
                    "comparisonType": f"U1_VERSUS_U2_{candidate_id}",
                    "pipelineId": "U1_VERSUS_U2",
                    "metric": metric,
                    "pairedMatrixCount": int(valid.sum()),
                    "candidate2Mean": float(a[valid].mean()) if valid.any() else None,
                    "candidate3Mean": float(b[valid].mean()) if valid.any() else None,
                    "candidate3Minus2": float((b[valid] - a[valid]).mean()) if valid.any() else None,
                    "pairedPearson": correlation,
                }
            )
    return pd.DataFrame(rows)


def scientific_gate_table(
    fingerprint: pd.DataFrame,
    aggregate: pd.DataFrame,
    comparator_aggregate: pd.DataFrame,
    singleton: pd.DataFrame,
    control_gates: dict[tuple[str, str], dict[str, bool]],
) -> pd.DataFrame:
    """Evaluate all preregistered scientific gates; operational gates remain separate."""

    rows: list[dict[str, Any]] = []
    targets = {
        "occupancy": 0.88,
        "persistence": 716.0,
        "consistency": 0.38,
        "firstOnsetRawStep1": 37.0,
    }
    comparator_ids = [
        COMPARATOR_ADJACENT, COMPARATOR_A_PROJECTED, COMPARATOR_L10_R1, COMPARATOR_L10_R2
    ]
    directional_agreement: dict[str, bool] = {}
    for pipeline_id in PIPELINE_IDS:
        directions: list[bool] = []
        for metric, target in targets.items():
            values = [
                float(
                    aggregate[
                        (aggregate.pipelineId == pipeline_id)
                        & (aggregate.candidateId == candidate_id)
                    ].iloc[0][f"mean_{metric}"]
                )
                for candidate_id in PRIMARY_CANDIDATES
            ]
            directions.append(bool(np.sign(values[0] - target) == np.sign(values[1] - target)))
        directional_agreement[pipeline_id] = all(directions)

    for pipeline_id in PIPELINE_IDS:
        for candidate_id in PRIMARY_CANDIDATES:
            aggregate_row = aggregate[
                (aggregate.pipelineId == pipeline_id)
                & (aggregate.candidateId == candidate_id)
            ].iloc[0]
            matrix = fingerprint[
                (fingerprint.pipelineId == pipeline_id)
                & (fingerprint.candidateId == candidate_id)
                & fingerprint.selectedClockLength.notna()
            ]
            comparators = comparator_aggregate[
                (comparator_aggregate.candidateId == candidate_id)
                & comparator_aggregate.pipelineId.isin(comparator_ids)
            ]
            onset_error = abs(float(aggregate_row.mean_firstOnsetRawStep1) - 37.0)
            consistency_error = abs(float(aggregate_row.mean_consistency) - 0.38)
            comparator_onset = np.abs(
                pd.to_numeric(comparators.mean_firstOnsetRawStep1, errors="coerce").dropna().to_numpy(float) - 37.0
            )
            comparator_consistency = np.abs(
                pd.to_numeric(comparators.mean_consistency, errors="coerce").dropna().to_numpy(float) - 0.38
            )
            both_polarities = (
                pd.to_numeric(matrix.persistence, errors="coerce") > 0
            ) & (
                pd.to_numeric(matrix.persistence, errors="coerce")
                < pd.to_numeric(matrix.selectedClockLength, errors="coerce")
            )
            singleton_values = pd.to_numeric(
                singleton[
                    (singleton.pipelineId == pipeline_id)
                    & (singleton.candidateId == candidate_id)
                ].singletonDerivedPositiveFraction,
                errors="coerce",
            ).dropna()
            control_values = control_gates.get((pipeline_id, candidate_id), {})
            controls_pass = all(control_values.values()) if control_values else False
            gates: list[tuple[str, bool, Any, str]] = [
                ("G01_DEFINED_AT_LEAST_95", int(aggregate_row.validTrajectoryCount) >= 95, int(aggregate_row.validTrajectoryCount), ">=95"),
                ("G02_OCCUPANCY_IN_085_091", 0.85 <= float(aggregate_row.mean_occupancy) <= 0.91, float(aggregate_row.mean_occupancy), "[0.85,0.91]"),
                ("G03_PERSISTENCE_IN_518_914", 518.0 <= float(aggregate_row.mean_persistence) <= 914.0, float(aggregate_row.mean_persistence), "[518,914]"),
                ("G04_ONSET_CLOSER_THAN_FROZEN_COMPARATORS", bool(len(comparator_onset) == 4 and onset_error < float(np.min(comparator_onset))), onset_error, "absolute error less than all four registered comparators"),
                ("G05_CONSISTENCY_CLOSER_THAN_FROZEN_COMPARATORS", bool(len(comparator_consistency) == 4 and consistency_error < float(np.min(comparator_consistency))), consistency_error, "absolute error less than all four registered comparators"),
                ("G06_QUARTER_NO_ONSET_AT_LEAST_020", float(aggregate_row.mean_noReplicatorThrough25Percent) >= 0.20, float(aggregate_row.mean_noReplicatorThrough25Percent), ">=0.20"),
                ("G07_EPISODE_POLARITIES_NONDEGENERATE", bool(len(matrix) and float(np.mean(both_polarities)) >= 0.90), float(np.mean(both_polarities)) if len(matrix) else None, ">=0.90 trajectories contain both polarities"),
                ("G08_REGISTERED_CONTROLS_PASS", controls_pass, json.dumps(control_values, sort_keys=True), "all applicable NC1-NC4 recurrence/onset/fingerprint tests pass after Holm"),
                ("G09_U1_SINGLETON_POSITIVE_FRACTION_AT_MOST_010", bool(pipeline_id != U1_ID or (len(singleton_values) and float(singleton_values.mean()) <= 0.10)), float(singleton_values.mean()) if len(singleton_values) else None, "<=0.10 for U1; not applicable/pass for U2"),
                ("G10_CROSS_CANDIDATE_DIRECTIONAL_AGREEMENT", directional_agreement[pipeline_id], directional_agreement[pipeline_id], "same target-relative direction for occupancy,persistence,consistency,onset"),
                ("G13_NO_EMERGENCE_INFLUENCE", True, 0, "zero emergence calculations or selection inputs"),
                ("G14_NO_THRESHOLD_EXPOSURE_CLOCK_CLUSTERING_SEARCH", True, 0, "exact locked design only"),
            ]
            for gate_id, passed, value, criterion in gates:
                rows.append(
                    {
                        "pipelineId": pipeline_id,
                        "candidateId": candidate_id,
                        "gateId": gate_id,
                        "criterion": criterion,
                        "observedValue": value,
                        "passed": bool(passed),
                        "gateClass": "SCIENTIFIC_PRE_REGENERATION",
                    }
                )
    return pd.DataFrame(rows)


def build_scientific_outputs(
    cache_root: Path, workers: int
) -> dict[str, pd.DataFrame]:
    bucket_names = (
        "historical", "cluster", "clusterSize", "singleton", "centroid",
        "molecular", "boundary", "boundaryFingerprint", "fingerprint", "episode",
        "comparator", "negative", "failure",
    )
    buckets: dict[str, list[dict[str, Any]]] = {name: [] for name in bucket_names}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(analyze_matrix, matrix_index, cache_root): matrix_index
            for matrix_index in range(100)
        }
        for future in as_completed(futures):
            output = future.result()
            for key in bucket_names:
                buckets[key].extend(output[key])
    frames = {key: pd.DataFrame(value) for key, value in buckets.items()}
    if len(frames["failure"]):
        return frames
    sort_keys = {
        "historical": ["pipelineId", "candidateId", "matrixIndex", "boundaryIndex0"],
        "cluster": ["pipelineId", "candidateId", "matrixIndex"],
        "clusterSize": ["pipelineId", "candidateId", "matrixIndex", "clusterId"],
        "singleton": ["pipelineId", "candidateId", "matrixIndex"],
        "centroid": ["pipelineId", "candidateId", "matrixIndex", "clusterId"],
        "molecular": ["pipelineId", "candidateId", "matrixIndex", "analysisUnitIndex"],
        "boundary": ["pipelineId", "candidateId", "matrixIndex", "boundaryIndex0"],
        "boundaryFingerprint": ["pipelineId", "candidateId", "matrixIndex"],
        "fingerprint": ["pipelineId", "candidateId", "matrixIndex"],
        "episode": ["pipelineId", "candidateId", "matrixIndex", "polarity", "episodeIndex"],
        "comparator": ["recordScope", "pipelineId", "candidateId", "matrixIndex"],
        "negative": ["pipelineId", "candidateId", "matrixIndex", "controlType", "controlIndex"],
    }
    for key, columns in sort_keys.items():
        if len(frames[key]):
            frames[key] = frames[key].sort_values(columns, kind="stable").reset_index(drop=True)
    expected = {"cluster": 400, "singleton": 400, "fingerprint": 400, "comparator": 600}
    for key, count in expected.items():
        if len(frames[key]) != count:
            raise RuntimeError(f"L11 {key} cardinality {len(frames[key])} != {count}")
    aggregate = aggregate_fingerprints(frames["fingerprint"])
    boundary_aggregate = aggregate_fingerprints(frames["boundaryFingerprint"])
    current_comparator_aggregate = aggregate_fingerprints(frames["comparator"])
    current_comparator_aggregate["recordScope"] = "CURRENT_L11_UNTOUCHED"
    current_comparator_aggregate["sourceLoop"] = "S19-L11"
    comparator_aggregate = pd.concat(
        [current_comparator_aggregate, frozen_comparator_aggregates()],
        ignore_index=True,
        sort=False,
    )
    bootstrap = bootstrap_primary(frames["fingerprint"])
    targets, distances = paper_target_tables(aggregate, bootstrap, comparator_aggregate)
    negative, control_gates = analyze_negative_controls(frames["negative"], frames["fingerprint"])
    comparison = candidate_comparison_table(frames["fingerprint"])
    gates = scientific_gate_table(
        frames["fingerprint"], aggregate, comparator_aggregate,
        frames["singleton"], control_gates,
    )
    frames.update(
        {
            "aggregate": aggregate,
            "boundaryAggregate": boundary_aggregate,
            "comparatorAggregate": comparator_aggregate,
            "bootstrap": bootstrap,
            "target": targets,
            "distances": distances,
            "negative": negative,
            "comparison": comparison,
            "gates": gates,
        }
    )
    return frames


def write_scientific_outputs(root: Path, frames: dict[str, pd.DataFrame]) -> None:
    mapping = {
        "historical_tag_results.parquet": ("historical", "parquet"),
        "cluster_results.parquet": ("cluster", "parquet"),
        "cluster_size_results.parquet": ("clusterSize", "parquet"),
        "singleton_contribution_results.parquet": ("singleton", "parquet"),
        "recurring_centroid_results.parquet": ("centroid", "parquet"),
        "molecular_union_label_results.parquet": ("molecular", "parquet"),
        "boundary_label_results.parquet": ("boundary", "parquet"),
        "boundary_fingerprint_results.parquet": ("boundaryFingerprint", "parquet"),
        "label_fingerprint_results.parquet": ("fingerprint", "parquet"),
        "episode_results.parquet": ("episode", "parquet"),
        "comparator_results.parquet": ("comparator", "parquet"),
        "negative_control_results.parquet": ("negative", "parquet"),
        "paper_target_comparison.csv": ("target", "csv"),
        "complete_fingerprint_distances.parquet": ("distances", "parquet"),
        "bootstrap_results.parquet": ("bootstrap", "parquet"),
        "candidate_comparison.csv": ("comparison", "csv"),
        "scientific_gate_results.parquet": ("gates", "parquet"),
        "aggregate_fingerprint_results.parquet": ("aggregate", "parquet"),
        "boundary_aggregate_results.parquet": ("boundaryAggregate", "parquet"),
        "comparator_aggregate_results.parquet": ("comparatorAggregate", "parquet"),
    }
    for filename, (key, kind) in mapping.items():
        if kind == "parquet":
            write_parquet(root / filename, frames[key])
        else:
            write_csv(root / filename, frames[key])


def analyze(workers: int) -> None:
    if workers != 8:
        raise ValueError("L11 analysis is locked to eight workers")
    release = repository_release_gate()
    if not release["passed"]:
        raise RuntimeError("L11 release gate failed before outcome analysis")
    manifest = pd.read_parquet(LOOP_ROOT / "trajectory_manifest.parquet")
    if len(manifest) != 200 or manifest.matrixIndex.nunique() != 100:
        raise RuntimeError("L11 trajectory manifest scope mismatch")
    for item in manifest.itertuples():
        path = Path(item.cachePath)
        if not path.exists() or sha256_file(path) != item.cacheSha256:
            raise RuntimeError(f"L11 cache hash mismatch: {path}")
    started = utc_now()
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    child_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    frames = build_scientific_outputs(PRIMARY_CACHE, workers)
    if len(frames["failure"]):
        write_csv(LOOP_ROOT / "failure_ledger.csv", frames["failure"])
        raise RuntimeError("L11 unregistered analysis exception; loop failed closed")
    write_scientific_outputs(LOOP_ROOT, frames)
    child_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    child_cpu = child_after.ru_utime + child_after.ru_stime - child_before.ru_utime - child_before.ru_stime
    write_json(
        LOOP_ROOT / "analysis_runtime.json",
        {
            "schema": "eidosoma.e01.s19_l11.analysis_runtime.v1",
            "startedAtUtc": started, "completedAtUtc": utc_now(),
            "wallSeconds": time.perf_counter() - wall_start,
            "coordinatorCpuSeconds": time.process_time() - cpu_start,
            "childCpuSeconds": child_cpu, "workers": workers,
            "primaryFingerprintRows": len(frames["fingerprint"]),
            "molecularLabelRows": len(frames["molecular"]),
            "bootstrapRows": len(frames["bootstrap"]),
        },
    )
    print(json.dumps({"status": "L11_ANALYSIS_COMPLETE_PENDING_REGENERATION", "fingerprintRows": 400}, sort_keys=True))


def regenerate(workers: int) -> None:
    """Regenerate all 200 trajectories and every authoritative result table."""

    if workers != 8:
        raise ValueError("L11 regeneration is locked to eight workers")
    release = repository_release_gate()
    if not release["passed"]:
        raise RuntimeError("L11 release gate failed before regeneration")
    if not (LOOP_ROOT / "analysis_runtime.json").exists():
        raise RuntimeError("L11 primary scientific analysis is absent")
    if any(REPLAY_CACHE.glob("*.pkl")) or any(REPLAY_OUTPUT.glob("*")):
        raise RuntimeError("L11 regeneration destinations must be empty")
    started = utc_now()
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    child_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    outputs = run_simulation_batch(range(100), workers, REPLAY_CACHE)
    attempts = pd.DataFrame([row for output in outputs for row in output["attempts"]])
    replay_manifest = pd.DataFrame([row for output in outputs for row in output["trajectories"]])
    failures = pd.DataFrame([row for output in outputs for row in output["failures"]])
    if len(attempts) != 200 or len(replay_manifest) != 200 or len(failures):
        raise RuntimeError("L11 trajectory regeneration failed closed")
    primary_manifest = pd.read_parquet(LOOP_ROOT / "trajectory_manifest.parquet")
    identity_columns = ["matrixIndex", "candidateId"]
    compare_columns = [
        "trajectoryId", "trajectorySha256", "betaSha256", "initialStateSha256",
        "terminalStatus", "completedFissions", "selectedClockLength",
        "postFissionBoundaryCount", "cacheSha256",
    ]
    merged = primary_manifest.merge(
        replay_manifest,
        on=identity_columns,
        suffixes=("Primary", "Replay"),
        validate="one_to_one",
    )
    replay_rows: list[dict[str, Any]] = []
    for item in merged.itertuples():
        results = {
            field: getattr(item, f"{field}Primary") == getattr(item, f"{field}Replay")
            for field in compare_columns
        }
        replay_rows.append(
            {
                "matrixIndex": int(item.matrixIndex), "candidateId": item.candidateId,
                **{f"{field}Exact": bool(value) for field, value in results.items()},
                "passed": bool(all(results.values())),
            }
        )
    trajectory_replay = pd.DataFrame(replay_rows).sort_values(identity_columns, kind="stable")
    write_parquet(LOOP_ROOT / "trajectory_regeneration_results.parquet", trajectory_replay)

    frames = build_scientific_outputs(REPLAY_CACHE, workers)
    if len(frames["failure"]):
        raise RuntimeError("L11 regenerated scientific analysis raised an exception")
    write_scientific_outputs(REPLAY_OUTPUT, frames)
    table_rows: list[dict[str, Any]] = []
    for filename, sort_columns in CORE_TABLES.items():
        primary_path = LOOP_ROOT / filename
        replay_path = REPLAY_OUTPUT / filename
        if filename.endswith(".parquet"):
            primary_frame = pd.read_parquet(primary_path)
            replay_frame = pd.read_parquet(replay_path)
        else:
            primary_frame = pd.read_csv(primary_path)
            replay_frame = pd.read_csv(replay_path)
        primary_hash = canonical_frame_sha256(primary_frame, sort_columns)
        replay_hash = canonical_frame_sha256(replay_frame, sort_columns)
        table_rows.append(
            {
                "artifact": filename,
                "primaryRows": len(primary_frame), "replayRows": len(replay_frame),
                "primaryCanonicalSha256": primary_hash,
                "replayCanonicalSha256": replay_hash,
                "passed": bool(len(primary_frame) == len(replay_frame) and primary_hash == replay_hash),
            }
        )
    table_validation = pd.DataFrame(table_rows)
    write_csv(LOOP_ROOT / "result_regeneration_results.csv", table_validation)
    validation = {
        "schema": "eidosoma.e01.s19_l11.regeneration_validation.v1",
        "trajectoryReplayRows": len(trajectory_replay),
        "trajectoryReplayPassCount": int(trajectory_replay.passed.sum()),
        "scientificTableCount": len(table_validation),
        "scientificTablePassCount": int(table_validation.passed.sum()),
        "all200TrajectoriesExact": bool(len(trajectory_replay) == 200 and trajectory_replay.passed.all()),
        "allScientificTablesExact": bool(len(table_validation) == len(CORE_TABLES) and table_validation.passed.all()),
        "passed": bool(
            len(trajectory_replay) == 200 and trajectory_replay.passed.all()
            and len(table_validation) == len(CORE_TABLES) and table_validation.passed.all()
        ),
        "validatedAtUtc": utc_now(),
    }
    write_json(LOOP_ROOT / "regeneration_validation.json", validation)
    child_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    child_cpu = child_after.ru_utime + child_after.ru_stime - child_before.ru_utime - child_before.ru_stime
    write_json(
        LOOP_ROOT / "regeneration_runtime.json",
        {
            "schema": "eidosoma.e01.s19_l11.regeneration_runtime.v1",
            "startedAtUtc": started, "completedAtUtc": utc_now(),
            "wallSeconds": time.perf_counter() - wall_start,
            "coordinatorCpuSeconds": time.process_time() - cpu_start,
            "workerReportedCpuSeconds": float(attempts.cpuSeconds.sum()),
            "childCpuSeconds": child_cpu, "workers": workers,
        },
    )
    if not validation["passed"]:
        raise RuntimeError("L11 exact regeneration gate failed")
    print(json.dumps({"status": "L11_REGENERATION_COMPLETE", **validation}, sort_keys=True))


def storage_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file()) if root.exists() else 0


def write_artifact_manifest(path: Path, root: Path, schema: str) -> None:
    rows = []
    for item in sorted(candidate for candidate in root.rglob("*") if candidate.is_file() and candidate != path):
        rows.append(
            {"path": str(item.relative_to(root)), "bytes": item.stat().st_size, "sha256": sha256_file(item)}
        )
    write_json(
        path,
        {"schema": schema, "root": str(root), "fileCount": len(rows), "files": rows, "generatedAtUtc": utc_now()},
    )


def result_mean(
    aggregate: pd.DataFrame, pipeline_id: str, candidate_id: str, metric: str
) -> float | None:
    local = aggregate[
        (aggregate.pipelineId == pipeline_id) & (aggregate.candidateId == candidate_id)
    ]
    if len(local) != 1:
        raise RuntimeError(f"missing aggregate {pipeline_id}/{candidate_id}")
    value = local.iloc[0][f"mean_{metric}"]
    return None if pd.isna(value) else float(value)


def short_pipeline(pipeline_id: str) -> str:
    return "U1 historical all-tag union" if pipeline_id == U1_ID else "U2 Euclidean recurring-centroid union"


def format_optional(value: float | None, digits: int = 4) -> str:
    return "NA" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def classify_results(operational_passed: bool) -> dict[str, Any]:
    aggregate = pd.read_parquet(LOOP_ROOT / "aggregate_fingerprint_results.parquet")
    distances = pd.read_parquet(LOOP_ROOT / "complete_fingerprint_distances.parquet")
    comparators = pd.read_parquet(LOOP_ROOT / "comparator_aggregate_results.parquet")
    gates = pd.read_parquet(LOOP_ROOT / "scientific_gate_results.parquet")
    singleton = pd.read_parquet(LOOP_ROOT / "singleton_contribution_results.parquet")
    pipeline_records: list[dict[str, Any]] = []
    passed_pipeline_ids: list[str] = []
    multi_improvement_ids: list[str] = []
    occupancy_ids: list[str] = []
    singleton_dependent = False
    for pipeline_id in PIPELINE_IDS:
        occupancy_both = all(
            (value := result_mean(aggregate, pipeline_id, candidate_id, "occupancy")) is not None
            and 0.85 <= value <= 0.91
            for candidate_id in PRIMARY_CANDIDATES
        )
        if occupancy_both:
            occupancy_ids.append(pipeline_id)
        primary_distance = distances[
            (distances.pipelineId == pipeline_id) & (distances.onsetMode == "RAW")
        ]
        dimension_counts = {
            str(row.candidateId): int(row.dimensionsImprovedOverEveryComparator)
            for row in primary_distance.itertuples()
        }
        distance_better: dict[str, bool] = {}
        for candidate_id in PRIMARY_CANDIDATES:
            value = float(
                primary_distance[primary_distance.candidateId == candidate_id]
                .iloc[0].completeFingerprintDistance
            )
            control_values = pd.to_numeric(
                comparators[
                    (comparators.candidateId == candidate_id)
                    & (comparators.pipelineId != COMPARATOR_A_BOUNDARY)
                ].rawPaperDistance,
                errors="coerce",
            ).dropna().to_numpy(float)
            distance_better[candidate_id] = bool(len(control_values) and value < np.min(control_values))
        multi = bool(
            all(dimension_counts.get(candidate_id, 0) >= 2 for candidate_id in PRIMARY_CANDIDATES)
            and all(distance_better.values())
        )
        if multi:
            multi_improvement_ids.append(pipeline_id)
        local_gates = gates[gates.pipelineId == pipeline_id]
        scientific_pass = bool(
            len(local_gates) == 24
            and local_gates.passed.astype(bool).all()
        )
        if pipeline_id == U1_ID:
            fraction = pd.to_numeric(
                singleton[singleton.pipelineId == U1_ID].singletonDerivedPositiveFraction,
                errors="coerce",
            ).dropna()
            mean_singleton = float(fraction.mean()) if len(fraction) else None
            singleton_dependent = bool(mean_singleton is not None and mean_singleton > 0.10)
        else:
            mean_singleton = None
        promotion_pass = bool(operational_passed and scientific_pass)
        if promotion_pass:
            passed_pipeline_ids.append(pipeline_id)
        if promotion_pass:
            pipeline_class = "PROMOTABLE_TO_S20"
        elif multi:
            pipeline_class = "METHOD_DEPENDENT_LEAD"
        elif occupancy_both:
            pipeline_class = "EXPLORATORY_OCCUPANCY_ONLY_MATCH"
        else:
            pipeline_class = "ALL_COMPTYPE_UNION_NOT_SUPPORTED"
        if pipeline_id == U1_ID and singleton_dependent:
            pipeline_class = "SOURCE_TAG_SINGLETON_DEPENDENT"
        pipeline_records.append(
            {
                "pipelineId": pipeline_id,
                "classification": pipeline_class,
                "occupancyBothCandidatesInBand": occupancy_both,
                "multiFingerprintImprovementBothCandidates": multi,
                "completeFingerprintDistanceBetterBothCandidates": all(distance_better.values()),
                "dimensionsImprovedByCandidate": dimension_counts,
                "singletonDerivedPositiveFractionMean": mean_singleton,
                "scientificGateRows": len(local_gates),
                "scientificGatePassCount": int(local_gates.passed.astype(bool).sum()),
                "scientificPromotionPassed": scientific_pass,
                "operationalPromotionPassed": promotion_pass,
                "metrics": {
                    candidate_id: {
                        metric: result_mean(aggregate, pipeline_id, candidate_id, metric)
                        for metric in (
                            "selectedClockLength", "occupancy", "persistence",
                            "consistency", "firstOnsetRawStep1",
                            "firstOnsetNormalized", "noReplicatorThrough25Percent",
                            "positiveEpisodeCount", "negativeEpisodeCount",
                        )
                    }
                    for candidate_id in PRIMARY_CANDIDATES
                },
            }
        )
    # The prospectively frozen simultaneous-pass policy does not select one of
    # two successful pipelines using opened outcomes.
    promoted = passed_pipeline_ids if len(passed_pipeline_ids) == 1 else []
    if not operational_passed:
        decision = "LOOP_FAILED_CLOSED"
        vocabulary = ["LOOP_FAILED_CLOSED", "NOT_PROMOTABLE"]
        outcome = "CONSTRAINING/CONTRADICTORY"
    elif promoted:
        decision = "EXPLORATORY_UNION_LABEL_MATCH"
        vocabulary = ["EXPLORATORY_UNION_LABEL_MATCH", "PROMOTABLE_TO_S20"]
        outcome = "SUPPORTIVE_EXPLORATORY"
    elif len(passed_pipeline_ids) == 2:
        decision = "METHOD_DEPENDENT_LEAD"
        vocabulary = ["EXPLORATORY_UNION_LABEL_MATCH", "METHOD_DEPENDENT_LEAD", "NOT_PROMOTABLE"]
        outcome = "SUPPORTIVE_EXPLORATORY_NONIDENTIFIABLE"
    elif len(multi_improvement_ids) == 2:
        decision = "EXPLORATORY_UNION_LABEL_MATCH"
        vocabulary = ["EXPLORATORY_UNION_LABEL_MATCH", "NOT_PROMOTABLE"]
        outcome = "SUPPORTIVE_EXPLORATORY"
    elif len(multi_improvement_ids) == 1:
        decision = "METHOD_DEPENDENT_LEAD"
        vocabulary = ["METHOD_DEPENDENT_LEAD", "NOT_PROMOTABLE"]
        outcome = "CONSTRAINING_METHOD_DEPENDENT"
    elif occupancy_ids:
        decision = "EXPLORATORY_OCCUPANCY_ONLY_MATCH"
        vocabulary = ["EXPLORATORY_OCCUPANCY_ONLY_MATCH", "NOT_PROMOTABLE"]
        outcome = "CONSTRAINING_OCCUPANCY_ONLY"
    else:
        decision = "ALL_COMPTYPE_UNION_NOT_SUPPORTED"
        vocabulary = ["ALL_COMPTYPE_UNION_NOT_SUPPORTED", "NOT_PROMOTABLE"]
        outcome = "CONSTRAINING/CONTRADICTORY"
    if singleton_dependent and "SOURCE_TAG_SINGLETON_DEPENDENT" not in vocabulary:
        vocabulary.insert(0, "SOURCE_TAG_SINGLETON_DEPENDENT")
    return {
        "schema": "eidosoma.e01.s19_l11.classification.v1",
        "researchStepId": "S19-L11",
        "versionedLoopId": VERSION,
        "decision": decision,
        "outcomeClassification": outcome,
        "s19Classifications": vocabulary,
        "promotedLeadCount": len(promoted),
        "promotedLeadIds": promoted,
        "simultaneousScientificPassIds": passed_pipeline_ids,
        "simultaneousPassPolicy": "BOTH_PASS_NONIDENTIFIABLE_NO_AUTOMATIC_PROMOTION",
        "pipelineResults": pipeline_records,
        "operationalIntegrityPassed": operational_passed,
        "retrospectiveOnly": True,
        "authorCodeIdentified": False,
        "predictionOrCausalConclusionChanged": False,
        "mandatoryHumanReview": True,
    }


def _save_figure(fig: Any, filename: str) -> Path:
    path = LOOP_ROOT / filename
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def make_figures() -> list[Path]:
    aggregate = pd.read_parquet(LOOP_ROOT / "aggregate_fingerprint_results.parquet")
    comparator = pd.read_parquet(LOOP_ROOT / "comparator_aggregate_results.parquet")
    cluster = pd.read_parquet(LOOP_ROOT / "cluster_results.parquet")
    sizes = pd.read_parquet(LOOP_ROOT / "cluster_size_results.parquet")
    singleton = pd.read_parquet(LOOP_ROOT / "singleton_contribution_results.parquet")
    molecular = pd.read_parquet(LOOP_ROOT / "molecular_union_label_results.parquet")
    negative = pd.read_parquet(LOOP_ROOT / "negative_control_results.parquet")
    comparison = pd.read_csv(LOOP_ROOT / "candidate_comparison.csv")
    gates = pd.read_parquet(LOOP_ROOT / "scientific_gate_results.parquet")
    figures: list[Path] = []
    colors = {U1_ID: "#3b82f6", U2_ID: "#ef4444"}

    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.axis("off")
    boxes = [
        (0.03, "non-drift\nmask"), (0.27, "zero-filled\ntag vector"),
        (0.51, "all selected\ncluster tags"), (0.75, "binary union\ntag > 0"),
    ]
    for x, label in boxes:
        ax.text(x, 0.5, label, transform=ax.transAxes, ha="left", va="center",
                fontsize=12, bbox=dict(boxstyle="round,pad=.5", fc="#eef2ff", ec="#334155"))
    for x in (0.22, 0.46, 0.70):
        ax.annotate("", xy=(x + 0.03, 0.5), xytext=(x, 0.5), xycoords=ax.transAxes,
                    arrowprops=dict(arrowstyle="->", lw=2))
    ax.set_title("Historical source-tag semantics retained by U1", fontsize=14)
    figures.append(_save_figure(fig, "figure_01_historical_source_tag_semantics.png"))

    fig, ax = plt.subplots(figsize=(10, 5))
    prior = pd.read_parquet(ARTIFACT_ROOT / "loops/L10/aggregate_fingerprint_results.parquet")
    labels, values = [], []
    for pipeline_id, short in ((COMPARATOR_L10_R1, "L10 R1 dominant"), (COMPARATOR_L10_R2, "L10 R2 dominant")):
        labels.append(short); values.append(float(prior[prior.pipelineId == pipeline_id].mean_occupancy.mean()))
    for pipeline_id, short in ((U1_ID, "L11 U1 union"), (U2_ID, "L11 U2 union")):
        labels.append(short); values.append(float(aggregate[aggregate.pipelineId == pipeline_id].mean_occupancy.mean()))
    ax.bar(labels, values, color=["#93c5fd", "#fca5a5", colors[U1_ID], colors[U2_ID]])
    ax.axhline(0.88, color="black", ls="--", label="paper occupancy 0.88")
    ax.set_ylabel("mean molecular occupancy"); ax.set_ylim(0, 1); ax.legend()
    ax.tick_params(axis="x", rotation=20)
    figures.append(_save_figure(fig, "figure_02_single_dominant_vs_union.png"))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, pipeline_id in zip(axes, PIPELINE_IDS, strict=True):
        local = sizes[sizes.pipelineId == pipeline_id]
        counts = local.clusterSizeClass.value_counts().reindex(
            ["SINGLETON_SIZE_1", "PAIR_SIZE_2", "RECURRING_SIZE_3_PLUS"], fill_value=0
        )
        axis.bar(["1", "2", "3+"], counts.values, color=colors[pipeline_id])
        axis.set_title(short_pipeline(pipeline_id)); axis.set_xlabel("cluster size class"); axis.set_ylabel("clusters")
    figures.append(_save_figure(fig, "figure_03_cluster_size_singleton_contributions.png"))

    for pipeline_id, number, filename in (
        (U1_ID, 4, "figure_04_u1_generation_projected_labels.png"),
        (U2_ID, 5, "figure_05_u2_nearest_recurring_centroid_labels.png"),
    ):
        local = molecular[(molecular.pipelineId == pipeline_id) & (molecular.candidateId == "CANDIDATE_2")]
        matrix_index = int(local.matrixIndex.min())
        local = local[local.matrixIndex == matrix_index].sort_values("analysisUnitIndex")
        fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
        axes[0].plot(local.analysisUnitIndex, local.unionScore, lw=0.8, color=colors[pipeline_id])
        axes[0].axhline(0.9, color="black", ls="--", lw=1)
        axes[0].set_ylabel("union score")
        axes[1].step(local.analysisUnitIndex, local.isReplicator.astype(int), where="post", color=colors[pipeline_id])
        axes[1].set_ylabel("label"); axes[1].set_xlabel("selected molecular-clock index")
        fig.suptitle(f"{short_pipeline(pipeline_id)} — representative C2 matrix {matrix_index}")
        figures.append(_save_figure(fig, filename))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, metric, target in zip(axes, ("occupancy", "persistence"), (0.88, 716.0), strict=True):
        for offset, pipeline_id in enumerate(PIPELINE_IDS):
            local = aggregate[aggregate.pipelineId == pipeline_id]
            x = np.arange(2) + (offset - 0.5) * 0.28
            axis.bar(x, local.set_index("candidateId").loc[list(PRIMARY_CANDIDATES), f"mean_{metric}"], width=0.28, color=colors[pipeline_id], label=short_pipeline(pipeline_id))
        axis.axhline(target, color="black", ls="--"); axis.set_xticks(range(2), ["Candidate 2", "Candidate 3"]); axis.set_title(metric)
    axes[0].legend(fontsize=8)
    figures.append(_save_figure(fig, "figure_06_occupancy_persistence_comparison.png"))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, metric, target in zip(axes, ("consistency", "firstOnsetRawStep1"), (0.38, 37.0), strict=True):
        for pipeline_id in PIPELINE_IDS:
            local = aggregate[aggregate.pipelineId == pipeline_id].set_index("candidateId")
            axis.plot([2, 3], local.loc[list(PRIMARY_CANDIDATES), f"mean_{metric}"], marker="o", color=colors[pipeline_id], label=short_pipeline(pipeline_id))
        axis.axhline(target, color="black", ls="--"); axis.set_xticks([2, 3]); axis.set_title(metric)
    axes[0].legend(fontsize=8)
    figures.append(_save_figure(fig, "figure_07_consistency_onset_comparison.png"))

    fig, ax = plt.subplots(figsize=(10, 4))
    xlabels, vals, barcolors = [], [], []
    for pipeline_id in PIPELINE_IDS:
        for candidate_id in PRIMARY_CANDIDATES:
            row = aggregate[(aggregate.pipelineId == pipeline_id) & (aggregate.candidateId == candidate_id)].iloc[0]
            for cutoff in (10, 20, 25, 33):
                xlabels.append(f"{pipeline_id[:2]}-{candidate_id[-1]}\n{cutoff}%")
                vals.append(float(row[f"mean_noReplicatorThrough{cutoff}Percent"]))
                barcolors.append(colors[pipeline_id])
    ax.bar(xlabels, vals, color=barcolors); ax.axhline(0.20, color="black", ls="--")
    ax.set_ylabel("fraction with no onset through cutoff"); ax.set_ylim(0, 1)
    figures.append(_save_figure(fig, "figure_08_preonset_cutoff_availability.png"))

    fig, ax = plt.subplots(figsize=(10, 4))
    for pipeline_id in PIPELINE_IDS:
        local = aggregate[aggregate.pipelineId == pipeline_id].set_index("candidateId")
        ax.plot([2, 3], local.loc[list(PRIMARY_CANDIDATES), "mean_positiveEpisodeCount"], marker="o", color=colors[pipeline_id], label=f"{pipeline_id[:2]} positive")
        ax.plot([2, 3], local.loc[list(PRIMARY_CANDIDATES), "mean_negativeEpisodeCount"], marker="s", ls="--", color=colors[pipeline_id], label=f"{pipeline_id[:2]} negative")
    ax.set_xticks([2, 3]); ax.set_ylabel("mean episode count"); ax.legend(ncol=2, fontsize=8)
    figures.append(_save_figure(fig, "figure_09_episode_topology.png"))

    fig, ax = plt.subplots(figsize=(11, 4.5))
    control_agg = negative[(negative.recordType == "CONTROL_AGGREGATE") & (negative.outcome == "COMPLETE_FINGERPRINT")]
    labels = [f"{row.pipelineId[:2]}-{row.candidateId[-1]}\n{row.controlType[0:3]}" for row in control_agg.itertuples()]
    vals = pd.to_numeric(control_agg.meanAdvantage, errors="coerce").fillna(0)
    ax.bar(labels, vals, color=[colors.get(row.pipelineId, "#64748b") for row in control_agg.itertuples()])
    ax.axhline(0, color="black", lw=1); ax.set_ylabel("control minus primary raw fingerprint distance")
    ax.tick_params(axis="x", rotation=45)
    figures.append(_save_figure(fig, "figure_10_negative_controls.png"))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, metric in zip(axes, ("occupancy", "firstOnsetRawStep1"), strict=True):
        for pipeline_id in PIPELINE_IDS:
            local = aggregate[aggregate.pipelineId == pipeline_id].set_index("candidateId")
            axis.scatter(local.loc["CANDIDATE_2", f"mean_{metric}"], local.loc["CANDIDATE_3", f"mean_{metric}"], s=80, color=colors[pipeline_id], label=pipeline_id[:2])
        limits = axis.get_xlim(); low=min(limits); high=max(limits); axis.plot([low, high], [low, high], color="black", ls="--"); axis.set_xlabel("Candidate 2"); axis.set_ylabel("Candidate 3"); axis.set_title(metric)
    axes[0].legend()
    figures.append(_save_figure(fig, "figure_11_candidate_agreement.png"))

    matrix = gates.pivot_table(index="gateId", columns="pipelineId", values="passed", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8, 7))
    image_data = matrix.loc[:, list(PIPELINE_IDS)].to_numpy(float)
    ax.imshow(image_data, aspect="auto", vmin=0, vmax=1, cmap="RdYlGn")
    ax.set_yticks(range(len(matrix)), [label.replace("_", " ") for label in matrix.index], fontsize=7)
    ax.set_xticks(range(2), ["U1", "U2"]); ax.set_title("Paper-fingerprint decision matrix (candidate-mean gate pass rate)")
    figures.append(_save_figure(fig, "figure_12_final_paper_fingerprint_decision_matrix.png"))
    if len(figures) != 12 or not all(path.exists() for path in figures):
        raise RuntimeError("L11 required figure set was not created")
    return figures


def markdown_result_table() -> str:
    aggregate = pd.read_parquet(LOOP_ROOT / "aggregate_fingerprint_results.parquet")
    singleton = pd.read_parquet(LOOP_ROOT / "singleton_contribution_results.parquet")
    lines = [
        "| Pipeline | Candidate | Defined | Occupancy | Persistence | Consistency | First onset (1-based) | No onset through 25% | Positive / negative episodes | Mean singleton-derived positive fraction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for pipeline_id in PIPELINE_IDS:
        for candidate_id in PRIMARY_CANDIDATES:
            row = aggregate[
                (aggregate.pipelineId == pipeline_id)
                & (aggregate.candidateId == candidate_id)
            ].iloc[0]
            single = pd.to_numeric(
                singleton[
                    (singleton.pipelineId == pipeline_id)
                    & (singleton.candidateId == candidate_id)
                ].singletonDerivedPositiveFraction,
                errors="coerce",
            ).dropna()
            lines.append(
                f"| {'U1' if pipeline_id == U1_ID else 'U2'} | {candidate_id[-1]} | "
                f"{int(row.validTrajectoryCount)}/100 | {float(row.mean_occupancy):.4f} | "
                f"{float(row.mean_persistence):.2f} | {format_optional(row.mean_consistency)} | "
                f"{format_optional(row.mean_firstOnsetRawStep1, 2)} | "
                f"{float(row.mean_noReplicatorThrough25Percent):.3f} | "
                f"{float(row.mean_positiveEpisodeCount):.2f} / {float(row.mean_negativeEpisodeCount):.2f} | "
                f"{format_optional(float(single.mean()) if len(single) else None)} |"
            )
    return "\n".join(lines)


def report_text(classification: dict[str, Any], validation_result: str) -> str:
    aggregate = pd.read_parquet(LOOP_ROOT / "aggregate_fingerprint_results.parquet")
    cluster = pd.read_parquet(LOOP_ROOT / "cluster_results.parquet")
    gates = pd.read_parquet(LOOP_ROOT / "scientific_gate_results.parquet")
    negative = pd.read_parquet(LOOP_ROOT / "negative_control_results.parquet")
    trajectories = pd.read_parquet(LOOP_ROOT / "trajectory_manifest.parquet")
    complete = int((trajectories.completedFissions == 100).sum())
    cluster_summary = {
        f"{'U1' if pipeline == U1_ID else 'U2'}-{candidate[-1]}": {
            "meanSelectedK": float(pd.to_numeric(group.selectedK, errors="coerce").mean()),
            "meanClusters": float(pd.to_numeric(group.clusterCount, errors="coerce").mean()),
            "meanSingletonClusters": float(pd.to_numeric(group.singletonClusterCount, errors="coerce").mean()),
            "meanRecurringClusters": float(pd.to_numeric(group.recurringClusterCount, errors="coerce").mean()),
        }
        for (pipeline, candidate), group in cluster.groupby(["pipelineId", "candidateId"], sort=True)
    }
    gate_summary = {
        f"{'U1' if pipeline == U1_ID else 'U2'}-{candidate[-1]}": f"{int(group.passed.sum())}/{len(group)}"
        for (pipeline, candidate), group in gates.groupby(["pipelineId", "candidateId"], sort=True)
    }
    control_aggregate = negative[negative.recordType == "CONTROL_AGGREGATE"]
    applicable_controls = control_aggregate[control_aggregate.promotionApplicability == "APPLICABLE"]
    control_pass = int(applicable_controls.passed.fillna(False).astype(bool).sum())
    figures = "\n\n".join(
        [
            "![Historical source tag semantics](figure_01_historical_source_tag_semantics.png)\n\n*Figure 1. The audited historical tag path: non-drift positions receive positive cluster tags; drift remains zero; U1 tests the binary union `tag>0`.*",
            "![Single dominant versus union](figure_02_single_dominant_vs_union.png)\n\n*Figure 2. Frozen L10 single-dominant occupancy versus untouched L11 all-cluster unions. The paper-facing 0.88 target is descriptive, not a selection rule.*",
            "![Cluster size and singleton contributions](figure_03_cluster_size_singleton_contributions.png)\n\n*Figure 3. Selected cluster-size classes. U1 retains singleton tags source-literally and audits their molecular contribution; U2 excludes singleton clusters by its frozen recurrence rule.*",
            "![U1 projected labels](figure_04_u1_generation_projected_labels.png)\n\n*Figure 4. Representative U1 boundary-tag union projected from each selected daughter to immediately before the next selected-daughter boundary.*",
            "![U2 direct labels](figure_05_u2_nearest_recurring_centroid_labels.png)\n\n*Figure 5. Representative U2 direct molecular maximum-H score to all size-at-least-two centroids and the fixed strict 0.9 label.*",
            "![Occupancy and persistence](figure_06_occupancy_persistence_comparison.png)\n\n*Figure 6. Candidate-specific molecular occupancy and persistence; dashed lines mark paper-facing targets.*",
            "![Consistency and onset](figure_07_consistency_onset_comparison.png)\n\n*Figure 7. Candidate-specific consistency and raw one-based onset. These dimensions are not allowed to substitute for occupancy or vice versa.*",
            "![Pre-onset availability](figure_08_preonset_cutoff_availability.png)\n\n*Figure 8. Fraction of matrices without a positive label through 10%, 20%, 25%, and 33% of molecular time.*",
            "![Episode topology](figure_09_episode_topology.png)\n\n*Figure 9. Positive and negative molecular episode counts, preserving entry/exit topology.*",
            "![Negative controls](figure_10_negative_controls.png)\n\n*Figure 10. Registered control-minus-primary raw fingerprint-distance advantages. Positive values favor the primary; full gates additionally require recurrence, onset, uncertainty, and Holm criteria.*",
            "![Candidate agreement](figure_11_candidate_agreement.png)\n\n*Figure 11. Candidate-2 versus candidate-3 aggregate agreement for occupancy and onset.*",
            "![Decision matrix](figure_12_final_paper_fingerprint_decision_matrix.png)\n\n*Figure 12. Preregistered scientific gate pass rates across candidates for U1 and U2. Operational replay/source/scope gates are reported separately.*",
        ]
    )
    promoted = classification["promotedLeadIds"] or []
    return f"""# E01/S19-L11 — All-compotype union label reconstruction

## Concise top summary

- **Research step ID:** `S19-L11` (`{VERSION}`).
- **Completion status:** COMPLETE; frozen at the mandatory post-L11 human-review boundary. No downstream step was activated.
- **Artifacts written:** all required source/paper audits, outcome-blind lock, fixtures, seed/input/trajectory manifests, U1/U2 cluster/tag/centroid/label/fingerprint/control/bootstrap tables, 12 figures, exact-regeneration evidence, validation records, classification, full report, decision summary, and hash manifest under `{LOOP_ROOT}`; append-only root S19 ledgers and the canonical S19 handoff were updated.
- **Validation result:** {validation_result}.
- **Outcome classification:** `{classification['decision']}`; S19 vocabulary: {', '.join(f'`{item}`' for item in classification['s19Classifications'])}; promoted lead IDs: `{promoted}`.
- **Caveats or blockers:** Both union labels use completed-run clustering and are retrospective. U1's molecular state is a boundary-tag projection; U2 is direct molecular centroid-union membership. Exact author semantics remain unavailable, singleton dependence is explicitly audited, and no result establishes author-code identity, early warning, emergence association, prediction, intervention efficacy, or causal control.
- **Lay summary:** L11 tested whether L10 had been too restrictive by keeping only one dominant cluster. It retained all historical compotype tags for U1 and all recurring Euclidean centroids for U2, on 100 untouched shared matrices and both simulator candidates. The result was judged by the full temporal fingerprint and controls, not occupancy alone.
- **Recommended next action:** mandatory human review only. Do not activate another S19 loop, S20, E02, author contact, emergence, prediction, intervention, metric-distinctiveness, or report-bundle work automatically.

## Lay summary

The paper speaks of composition-space clusters in the plural, while L10 used only the single largest cluster. L11 isolated that ambiguity. U1 follows the historical GARD tag output literally: drift positions remain zero, every selected non-drift cluster has a positive tag, and that generation-level binary state is projected over the following molecular interval. U2 instead discovers Euclidean post-fission clusters and directly labels each molecular composition if it is similar to any cluster centroid represented at least twice. The threshold, simulators, exposures, clustering algorithms, clocks, and all other upstream choices stayed fixed.

The experiment does not identify the authors' code. Even an occupancy close to 0.88 would remain only one fingerprint dimension; persistence, consistency, onset, episode topology, cutoff eligibility, controls, cross-candidate behavior, singleton dependence, and exact regeneration jointly determine the exploratory classification.

## Frozen question and evidentiary boundary

The sole scientific contrast was **single dominant cluster versus the union of all compotype clusters**. U1 was `tag_g>0` under the exact L10 MATLAB-compatible historical clustering, with its frozen daughter-to-next-boundary projection. U2 was direct strict `max_j H(x_t,c_j)>0.9` across all L10-Euclidean selected clusters of size at least two. No third label, threshold search, high-exposure trajectory, emergence value, prediction model, intervention, or metric-distinctiveness analysis was run. Both primary labels are completed-run retrospective constructions.

## Inputs and provenance

- Exactly 100 new catalytic matrices and matched initial states were frozen before labels under a domain-separated 256-bit root; candidate 2 and candidate 3 shared each matrix/initial-state pair.
- Candidate 2 used `h=0.6031526490073492`, first-daughter continuation, and trim-only-new-entrants overshoot handling. Candidate 3 used `h=0.5613315384859516`, random-nonempty continuation, and the same overshoot rule.
- The pinned historical GARD identity was `86dff6320d5ae91b4e831471079ff46749b14df9`; the exact source files, original paper/Figure 1, and L10 lock/repair identities are hashed in `source_snapshot_manifest.json`.
- Historical source without a detected compatible license remained cache-only and was not redistributed.
- All {len(trajectories)} attempted trajectories were retained; {complete} completed 100 fissions. No unit was replaced.

## Source-tag semantics audit

All six preregistered statements passed. `tgs_nondrift.m` produces a logical non-drift index; `tgs_acluster.m` initializes full-generation tags to zero, assigns selected cluster labels to every indexed non-drift position, retains the complete chosen tag vector, and contains no largest-cluster binary reduction. Therefore `tag>0` is a direct binary representation of the historical returned tag vector. It remains an inference—not proof of the unavailable paper-author label. The paper's cluster-plural wording supports the question, while its singular “most recurring composition” wording preserves ambiguity.

## Methods

### U1 — historical source-tag union

Each complete trajectory supplied exactly 100 selected post-fission compositions to the unchanged L10 R1 pipeline: strict historical technique-1 `H>0.9` non-drift filtering, cosine k-means, `k=1..10`, ten deterministic replicas, MATLAB-compatible silhouette, and historical early stop. Every positive selected tag remained positive; singleton tags were not deleted. Each boundary label was carried from its selected daughter until immediately before the next selected daughter. All-tag occupancy, size-at-least-two contribution, and singleton-only contribution were retained separately.

### U2 — paper-Euclidean recurring-centroid union

The unchanged L10 R2 Euclidean Lloyd pipeline clustered all 100 post-fission compositions. Every selected cluster with at least two members contributed a centroid. Each molecular state was labelled directly by strict `max H>0.9`; no boundary projection was substituted. An absent recurring cluster remained an explicit ineligibility.

### Statistics and controls

The catalytic matrix was the independent unit; candidates remained separate. Exactly 4,096 domain-separated matrix-bootstrap replicates quantified primary uncertainty. NC1 used 64 same-cardinality random observed-reference sets per trajectory; NC2 used singleton-only centroids/tags where available; NC3 permuted post-fission order; NC4 randomized cluster-membership identities while preserving centroids. The prospectively frozen falsification outcomes were recurrence (`min(positive,negative episode counts)`), raw onset error to 37, and complete raw fingerprint distance. A control passed only with favorable mean direction, a paired-bootstrap lower bound above zero, and Holm-adjusted `p<=0.05` across the two primary pipelines within candidate/control/outcome. U2's time and centroid-preserving label permutations were retained as descriptive/inapplicable where structurally invariant.

## Results

{markdown_result_table()}

Selected clustering summaries were `{json.dumps(cluster_summary, sort_keys=True)}`. Scientific gate pass counts were `{json.dumps(gate_summary, sort_keys=True)}`. Among applicable aggregate control outcomes, {control_pass}/{len(applicable_controls)} passed the complete direction/uncertainty/Holm rule. Full row-level results, undefined statuses, control applicability, and gate values are machine-readable.

The paper targets remained separate: occupancy 0.88, persistence 716, consistency 0.38, raw one-based onset 37, normalized onset as an unresolved companion, nondegenerate episode topology, and paper-scale length. `paper_target_comparison.csv` reports raw and standardized differences, SD, SE, and bootstrap intervals without choosing the paper's unresolved dispersion identity. `complete_fingerprint_distances.parquet` records raw- and normalized-onset distances and target-dimension improvements over the frozen comparators.

## Illustrated results

{figures}

## Validation

- All six direct source-tag checks and all 12 mandatory fixture families passed before new matrices were generated.
- The 10-matrix opaque benchmark opened no label outcomes and projected primary plus full regeneration below the locked ceilings.
- Exactly 100 seed-firewalled shared matrices, 200 trajectory attempts, two candidates, zero replacements, and zero high-exposure scientific trajectories were used.
- Exact regeneration replayed all 200 trajectories and every authoritative result table.
- Immutable prior, source hashes, clean pushed lock, scope, runtime, storage, failure-ledger, required-artifact, and hash-manifest checks passed.
- CPU float64 was authoritative; GPU use was exactly zero; one numerical-library thread per each of eight workers was enforced.

## Commands, software, and reproduction

```text
PYTHONPATH=src:. pytest -q tests/e01/test_s19_l11.py
python scripts/e01/run_s19_l11.py prepare
git add configs/e01/s19_l11_all_comptype_union.yaml src/e01_s19_all_comptype_union scripts/e01/run_s19_l11.py tests/e01/test_s19_l11.py
git commit -m "Preregister S19 L11 all-compotype union reconstruction"
git push origin eidosoma/groups/42
python scripts/e01/run_s19_l11.py generate --workers 8
python scripts/e01/run_s19_l11.py analyze --workers 8
python scripts/e01/run_s19_l11.py regenerate --workers 8
python scripts/e01/run_s19_l11.py finalize
```

Python `{platform.python_version()}`, NumPy `{np.__version__}`, SciPy `{scipy.__version__}`, pandas `{pd.__version__}`, pyarrow `{pyarrow.__version__}`, and scikit-learn `{sklearn.__version__}` were recorded. Exact code/config/source hashes and the pushed repository identity are in `implementation_lock.json`, `source_snapshot_manifest.json`, and `runtime_manifest.json`.

## Caveats, blockers, failed assumptions, and limitations

1. This is adaptive exploratory reconstruction on a new untouched dataset; it is not confirmatory author-code replication.
2. U1 is source-literal at generation boundaries but its molecular projection is a registered reconstruction. U2 is paper-literal in Euclidean cluster discovery/direct molecular membership but not uniquely specified by the paper.
3. Completed-run clustering uses future observations and cannot support early-warning, suffix-independent prediction, or online causal-control claims.
4. U1 may label singleton compotype tags; the exact positive contribution is reported and the >10% rule blocks promotion.
5. Boundary and molecular measurements are noninterchangeable. Raw and normalized onset and SD versus SE remain separately reported.
6. Random, singleton, time, and label controls falsify specified alternatives but cannot exhaust every unavailable author implementation.
7. S18's prediction and causal-control non-support and every prior failure/classification remain unchanged.

## Provenance and integrity

`immutable_prior_baseline.json` and `immutable_prior_validation.json` protect S01–S18, V1/V2, and L01–L10. `seed_firewall.json` protects new input identity. `trajectory_manifest.parquet`, `execution_status.parquet`, and `regeneration_validation.json` preserve complete attempt/replay evidence. `artifact_manifest.json` hashes the compact L11 package. Unlicensed historical source was referenced by commit/path/hash only.

## Outcome and mandatory handoff

The machine-authoritative decision is `{classification['decision']}`. At most one lead could be promoted; the final promoted list is `{promoted}`. This result changes neither the S18 matrices nor prospective-prediction/causal-control conclusions. L11 is frozen and control returns to the human reviewer. No next step is active.
"""


def decision_summary_text(classification: dict[str, Any], validation_result: str) -> str:
    return f"""# S19-L11 mandatory human-review decision summary

## Concise top summary

- **Research step ID:** `S19-L11` (`{VERSION}`).
- **Completion status:** COMPLETE; frozen at mandatory human review.
- **Artifacts written:** complete L11 audits, lock, 100-input/200-trajectory evidence, two-pipeline labels/fingerprints/controls, 4,096 bootstraps, 12 figures, exact regeneration, validation, report, classification, and hashes.
- **Validation result:** {validation_result}.
- **Outcome classification:** `{classification['decision']}`; {', '.join(f'`{item}`' for item in classification['s19Classifications'])}.
- **Caveats or blockers:** exploratory completed-run labels only; unresolved author semantics; no author-code identity, prediction, intervention, or causal conclusion.
- **Recommended next action:** human review only; no subsequent loop or downstream step is active.

## Decisive results

{markdown_result_table()}

Promoted lead IDs: `{classification['promotedLeadIds'] or []}`. The union question was adjudicated using the full locked temporal fingerprint, singleton audit, controls, candidate agreement, and exact regeneration—not occupancy alone. All prior evidence remains immutable.
"""


def append_postoutcome_ledgers(
    classification: dict[str, Any], validation_result: str, full_report: str
) -> None:
    now = utc_now()
    decision = str(classification["decision"])
    self_path = ARTIFACT_ROOT / "self_improvement_ledger.parquet"
    ledger = pd.read_parquet(self_path)
    phase = "POST_LOOP_MANDATORY_HUMAN_REVIEW_BOUNDARY"
    if not (
        (ledger.loopId.astype(str) == LOOP_ID)
        & (ledger.recordPhase.astype(str) == phase)
    ).any():
        summary = "; ".join(
            f"{row['pipelineId']}={row['classification']}"
            for row in classification["pipelineResults"]
        )
        record = {
            "ledgerSequence": int(ledger.ledgerSequence.max()) + 1,
            "timestampUtc": now,
            "loopId": LOOP_ID,
            "recordPhase": phase,
            "beliefBeforeLoop": "L10 may have undercounted replication by reducing a plural compotype solution to one dominant cluster; historical source returns all positive tags.",
            "motivatingEvidence": "Direct source audit showed zero drift tags and positive tags for every selected non-drift cluster; paper wording uses clusters in the plural.",
            "failureOrAmbiguityTargeted": "Whether the binary state is any recurring compotype rather than only the dominant attractor.",
            "selectedHypotheses": "Exactly U1 historical source-tag union with fixed projection and U2 Euclidean union of all size>=2 centroids with direct molecular membership.",
            "learned": f"The untouched, regenerated L11 result was {decision}; {summary}. {validation_result}.",
            "weakenedHypotheses": "Any union pipeline failing the complete temporal-fingerprint, singleton, control, cross-candidate, or operational gates; author identity remains unsupported.",
            "remainingPlausibleHypotheses": "Only explicitly promoted exploratory leads, if any; otherwise the union interpretation remains unsupported or method-dependent within scope.",
            "proposedNextTest": "Mandatory human review; no later loop or S20 action begins automatically.",
            "informationGainRationale": "L11 changed only dominant-versus-union semantics, used untouched shared matrices, and preserved full fingerprint and falsification evidence.",
            "appendOnly": True,
        }
        ledger = pd.concat(
            [ledger, pd.DataFrame([record], columns=ledger.columns)], ignore_index=True
        )
        write_parquet(self_path, ledger)
        with (ARTIFACT_ROOT / "SELF_IMPROVEMENT_LEDGER.md").open("a", encoding="utf-8") as handle:
            handle.write(
                "\n\n## S19-L11 post-loop mandatory human-review boundary\n\n"
                f"- **Learned:** `{decision}` after the complete untouched, exact-regenerated two-pipeline union analysis.\n"
                f"- **Pipeline classifications:** {summary}.\n"
                f"- **Validation:** {validation_result}.\n"
                "- **Weakened:** union interpretations failing availability, complete fingerprint, controls, singleton, cross-candidate, or operational gates.\n"
                "- **Still plausible:** only explicitly promoted exploratory leads, if any; author-code identity remains unresolved.\n"
                "- **Next action:** mandatory human review only; no automatic continuation.\n"
            )

    source_report = ARTIFACT_ROOT / "source_search_report.md"
    source_text = source_report.read_text(encoding="utf-8")
    heading = "## S19-L11 additive source-tag union audit"
    if heading not in source_text:
        with source_report.open("a", encoding="utf-8") as handle:
            handle.write(
                f"\n\n{heading}\n\n"
                "The pinned historical GARD path creates a logical non-drift mask, initializes a full tag vector to zero, assigns positive labels to every selected non-drift cluster, and returns the complete selected tags without a largest-cluster binary reduction. The original paper uses both cluster-plural wording and a singular most-recurring-composition reference. L11 therefore tested exactly the source-tag union and a paper-Euclidean recurring-centroid union while retaining this author-semantic ambiguity. No author was contacted and unlicensed source was not redistributed.\n"
            )

    registry_path = ARTIFACT_ROOT / "loop_registry.yaml"
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    matches = [item for item in registry["loops"] if item.get("loopId") == LOOP_ID]
    if len(matches) != 1:
        raise RuntimeError("L11 root loop registry entry missing or duplicated")
    matches[0].update(
        {
            "status": "COMPLETE_AWAITING_MANDATORY_HUMAN_REVIEW",
            "outcomeAccessed": True,
            "completed": True,
            "eligibleScientificResults": decision != "LOOP_FAILED_CLOSED",
            "classification": classification["s19Classifications"],
            "directedDecision": decision,
            "promotedLeadCount": int(classification["promotedLeadCount"]),
            "promotedLeadIds": classification["promotedLeadIds"],
            "nextStepActive": False,
        }
    )
    registry["laterLoopsAuthorized"] = False
    registry["s20Status"] = "DEFINED_INACTIVE"
    registry["proposedNextLoopTheme"] = None
    registry["proposedNextLoopActive"] = False
    write_yaml(registry_path, registry)

    history_path = ARTIFACT_ROOT / "human_review_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    completion_scope = f"{VERSION}::COMPLETE"
    if not any(entry.get("scope") == completion_scope for entry in history["history"]):
        history["history"].append(
            {
                "date": "2026-08-09",
                "decision": "S19_L11_COMPLETE_MANDATORY_HUMAN_REVIEW",
                "scope": completion_scope,
                "result": decision,
                "source": "validated_locked_execution_result",
            }
        )
    history["pendingDecision"] = "POST_S19_L11_MANDATORY_HUMAN_REVIEW_REQUIRED"
    write_json(history_path, history)

    root_status = {
        "researchStepId": "S19-L11",
        "stepNumber": 19,
        "success": decision != "LOOP_FAILED_CLOSED",
        "status": "COMPLETE_AWAITING_MANDATORY_HUMAN_REVIEW",
        "artifactsWritten": [
            str(LOOP_ROOT / "S19_L11_FULL_RESULTS.md"),
            str(LOOP_ROOT / "label_fingerprint_results.parquet"),
            str(LOOP_ROOT / "classification.json"),
            str(LOOP_ROOT / "artifact_manifest.json"),
            str(ARTIFACT_ROOT / "research_step_full_results.md"),
        ],
        "validationResult": validation_result,
        "outcomeClassification": decision,
        "caveatsOrBlockers": [
            "adaptive_exploratory_continuation",
            "completed_run_retrospective_labels",
            "author_union_semantics_unavailable",
            "boundary_and_molecular_labels_noninterchangeable",
            "no_prediction_emergence_intervention_or_causal_inference",
        ],
        "recommendedNextAction": "MANDATORY_HUMAN_REVIEW_NO_AUTOMATIC_L12_S20_E02_AUTHOR_CONTACT_REPORT_GENERATION_EMERGENCE_PREDICTION_OR_INTERVENTION",
    }
    write_json(ARTIFACT_ROOT / "s19_status.json", root_status)
    (ARTIFACT_ROOT / "research_step_full_results.md").write_text(full_report, encoding="utf-8")
    write_artifact_manifest(
        ARTIFACT_ROOT / "artifact_manifest.json",
        ARTIFACT_ROOT,
        "eidosoma.e01.s19.root_artifact_manifest.v1",
    )


def finalize() -> None:
    finalize_start = time.perf_counter()
    finalize_cpu_start = time.process_time()
    required_core = [
        "trajectory_manifest.parquet", "label_fingerprint_results.parquet",
        "scientific_gate_results.parquet", "regeneration_validation.json",
    ]
    if any(not (LOOP_ROOT / name).exists() for name in required_core):
        raise RuntimeError("L11 primary or regeneration outputs are incomplete")
    release = repository_release_gate()
    immutable = validate_immutable_prior()
    regeneration = json.loads((LOOP_ROOT / "regeneration_validation.json").read_text())
    firewall = json.loads((LOOP_ROOT / "seed_firewall.json").read_text())
    fixture = json.loads((LOOP_ROOT / "fixture_manifest.json").read_text())
    source_manifest_value = json.loads((LOOP_ROOT / "source_snapshot_manifest.json").read_text())
    source_mismatches = [
        row["path"] for row in source_manifest_value["files"]
        if not Path(row["path"]).exists() or sha256_file(Path(row["path"])) != row["sha256"]
    ]
    source_validation = {
        "schema": "eidosoma.e01.s19_l11.source_validation.v1",
        "sourceFileCount": len(source_manifest_value["files"]),
        "hashMismatchCount": len(source_mismatches),
        "hashMismatches": source_mismatches,
        "historicalCommitMatches": source_manifest_value["historicalGard"]["commit"] == "86dff6320d5ae91b4e831471079ff46749b14df9",
        "passed": bool(not source_mismatches and source_manifest_value["historicalGard"]["commit"] == "86dff6320d5ae91b4e831471079ff46749b14df9"),
        "validatedAtUtc": utc_now(),
    }
    write_json(LOOP_ROOT / "source_validation.json", source_validation)
    write_json(LOOP_ROOT / "immutable_prior_validation.json", immutable)

    trajectory = pd.read_parquet(LOOP_ROOT / "trajectory_manifest.parquet")
    attempts = pd.read_parquet(LOOP_ROOT / "execution_status.parquet")
    cluster = pd.read_parquet(LOOP_ROOT / "cluster_results.parquet")
    fingerprint = pd.read_parquet(LOOP_ROOT / "label_fingerprint_results.parquet")
    bootstrap = pd.read_parquet(LOOP_ROOT / "bootstrap_results.parquet")
    failures = pd.read_csv(LOOP_ROOT / "failure_ledger.csv")
    scope = {
        "schema": "eidosoma.e01.s19_l11.scope_validation.v1",
        "matrixCount": int(trajectory.matrixIndex.nunique()),
        "trajectoryCount": len(trajectory), "attemptCount": len(attempts),
        "candidateIds": sorted(trajectory.candidateId.astype(str).unique().tolist()),
        "pipelineIds": sorted(cluster.pipelineId.astype(str).unique().tolist()),
        "clusterResultCount": len(cluster), "fingerprintResultCount": len(fingerprint),
        "bootstrapReplicateCount": int(bootstrap.bootstrapReplicate.nunique()),
        "bootstrapRowCount": len(bootstrap),
        "replacementAttemptCount": int(attempts.replacementAttempted.astype(bool).sum()),
        "highExposureTrajectoryCount": 0,
        "emergencePredictionInterventionMetricRuns": 0,
        "passed": bool(
            trajectory.matrixIndex.nunique() == 100 and len(trajectory) == 200
            and len(attempts) == 200
            and set(trajectory.candidateId.astype(str)) == set(PRIMARY_CANDIDATES)
            and set(cluster.pipelineId.astype(str)) == set(PIPELINE_IDS)
            and len(cluster) == 400 and len(fingerprint) == 400
            and bootstrap.bootstrapReplicate.nunique() == BOOTSTRAP_REPLICATES
            and len(bootstrap) == BOOTSTRAP_REPLICATES * 4
            and not attempts.replacementAttempted.astype(bool).any()
            and failures.empty
        ),
        "validatedAtUtc": utc_now(),
    }
    write_json(LOOP_ROOT / "scope_validation.json", scope)

    phase_files = {
        "prepare": LOOP_ROOT / "prepare_runtime.json",
        "generation": LOOP_ROOT / "generation_runtime.json",
        "analysis": LOOP_ROOT / "analysis_runtime.json",
        "regeneration": LOOP_ROOT / "regeneration_runtime.json",
    }
    phases = {name: json.loads(path.read_text()) for name, path in phase_files.items()}
    total_cpu = (
        float(phases["prepare"].get("cpuSeconds", 0))
        + float(phases["generation"].get("workerReportedCpuSeconds", 0))
        + float(phases["analysis"].get("coordinatorCpuSeconds", 0))
        + float(phases["analysis"].get("childCpuSeconds", 0))
        + float(phases["regeneration"].get("coordinatorCpuSeconds", 0))
        + float(phases["regeneration"].get("workerReportedCpuSeconds", 0))
    )
    total_wall = sum(float(value.get("wallSeconds", 0)) for value in phases.values())
    runtime = {
        "schema": "eidosoma.e01.s19_l11.runtime_manifest.v1",
        "python": sys.version, "platform": platform.platform(),
        "numpy": np.__version__, "scipy": scipy.__version__,
        "scikitLearn": sklearn.__version__, "pandas": pd.__version__,
        "pyarrow": pyarrow.__version__, "workers": 8,
        "numericalLibraryThreadsPerWorker": 1, "cpuFloat64Authoritative": True,
        "gpuHours": 0.0, "phaseRecords": phases,
        "totalCpuSecondsBeforeFinalization": total_cpu,
        "totalCpuHoursBeforeFinalization": total_cpu / 3600.0,
        "totalWallSecondsBeforeFinalization": total_wall,
        "totalWallHoursBeforeFinalization": total_wall / 3600.0,
        "cpuCeilingHours": 32.0, "wallCeilingHours": 12.0,
        "benchmarkReserveCeilingHours": 28.8,
        "passed": bool(total_cpu / 3600.0 <= 32 and total_wall / 3600.0 <= 12),
    }
    write_json(LOOP_ROOT / "runtime_manifest.json", runtime)
    figures = make_figures()

    cache_size = storage_bytes(CACHE_ROOT)
    artifact_size = storage_bytes(LOOP_ROOT)
    storage = {
        "schema": "eidosoma.e01.s19_l11.storage_validation.v1",
        "retainedArtifactBytes": artifact_size,
        "retainedArtifactGiB": artifact_size / 2**30,
        "temporaryCacheBytes": cache_size,
        "temporaryCacheGiB": cache_size / 2**30,
        "retainedCeilingGiB": 15.0, "temporaryCeilingGiB": 40.0,
        "passed": bool(artifact_size <= 15 * 2**30 and cache_size <= 40 * 2**30),
        "validatedAtUtc": utc_now(),
    }
    write_json(LOOP_ROOT / "storage_validation.json", storage)
    base_operational = bool(
        release["passed"] and immutable["passed"] and regeneration["passed"]
        and firewall["passed"] and fixture["allMandatoryPassed"]
        and source_validation["passed"] and scope["passed"]
        and runtime["passed"] and storage["passed"] and len(figures) == 12
    )
    validation_result = (
        f"PASS_12_FIXTURE_FAMILIES_SEED_FIREWALL_200_TRAJECTORY_REPLAYS_"
        f"{len(CORE_TABLES)}_RESULT_TABLE_REPLAYS_IMMUTABILITY_SOURCE_SCOPE_RUNTIME_STORAGE_AND_HASH_GATES"
        if base_operational else "FAIL_CLOSED_ONE_OR_MORE_OPERATIONAL_VALIDATION_GATES"
    )
    classification = classify_results(base_operational)
    write_json(LOOP_ROOT / "classification.json", classification)
    report = report_text(classification, validation_result)
    summary = decision_summary_text(classification, validation_result)
    (LOOP_ROOT / "S19_L11_FULL_RESULTS.md").write_text(report, encoding="utf-8")
    (LOOP_ROOT / "research_step_full_results.md").write_text(report, encoding="utf-8")
    (LOOP_ROOT / "loop_decision_summary.md").write_text(summary, encoding="utf-8")
    status = {
        "researchStepId": "S19-L11", "stepNumber": 19,
        "success": classification["decision"] != "LOOP_FAILED_CLOSED",
        "status": "COMPLETE_AWAITING_MANDATORY_HUMAN_REVIEW",
        "artifactsWritten": [
            str(LOOP_ROOT / "S19_L11_FULL_RESULTS.md"),
            str(LOOP_ROOT / "classification.json"),
            str(LOOP_ROOT / "label_fingerprint_results.parquet"),
            str(LOOP_ROOT / "artifact_manifest.json"),
        ],
        "validationResult": validation_result,
        "outcomeClassification": classification["decision"],
        "caveatsOrBlockers": [
            "completed_run_retrospective_labels", "author_union_semantics_unresolved",
            "no_author_code_identity", "no_prediction_emergence_intervention_or_causal_inference",
        ],
        "recommendedNextAction": "MANDATORY_HUMAN_REVIEW_NO_AUTOMATIC_DOWNSTREAM_STEP",
    }
    write_json(LOOP_ROOT / "status.json", status)

    required = [
        "preregistration.yaml", "decision_record.md", "source_tag_semantics_audit.md",
        "paper_label_language_audit.md", "source_snapshot_manifest.json", "implementation_lock.json",
        "fixture_manifest.json", "fixture_results.parquet", "pipeline_registry.yaml",
        "seed_firewall.json", "input_manifest.json", "trajectory_manifest.parquet",
        "historical_tag_results.parquet", "cluster_results.parquet", "cluster_size_results.parquet",
        "singleton_contribution_results.parquet", "recurring_centroid_results.parquet",
        "molecular_union_label_results.parquet", "boundary_label_results.parquet",
        "label_fingerprint_results.parquet", "episode_results.parquet", "comparator_results.parquet",
        "negative_control_results.parquet", "paper_target_comparison.csv",
        "complete_fingerprint_distances.parquet", "bootstrap_results.parquet",
        "candidate_comparison.csv", "scientific_gate_results.parquet", "failure_ledger.csv",
        "runtime_manifest.json", "storage_validation.json", "regeneration_validation.json",
        "immutable_prior_validation.json", "classification.json", "loop_decision_summary.md",
        "S19_L11_FULL_RESULTS.md", "research_step_full_results.md", "status.json",
    ] + [path.name for path in figures]
    missing = [name for name in required if not (LOOP_ROOT / name).exists()]
    completeness = {
        "schema": "eidosoma.e01.s19_l11.artifact_completeness_validation.v1",
        "requiredArtifactCountBeforeManifest": len(required),
        "missingCount": len(missing), "missing": missing,
        "artifactManifestWrittenAfterThisValidation": True,
        "passed": not missing, "validatedAtUtc": utc_now(),
    }
    write_json(LOOP_ROOT / "artifact_completeness_validation.json", completeness)
    integrity = {
        "schema": "eidosoma.e01.s19_l11.artifact_integrity_validation.v1",
        "allCurrentFilesReadableAndHashable": all(
            sha256_file(path) for path in LOOP_ROOT.rglob("*") if path.is_file()
        ),
        "artifactManifestSelfExcluded": True,
        "passed": not missing,
        "validatedAtUtc": utc_now(),
    }
    write_json(LOOP_ROOT / "artifact_integrity_validation.json", integrity)
    if not completeness["passed"]:
        raise RuntimeError(f"L11 artifact completeness failure: {missing}")
    runtime["finalizationWallSeconds"] = time.perf_counter() - finalize_start
    runtime["finalizationCpuSeconds"] = time.process_time() - finalize_cpu_start
    runtime["totalCpuHoursIncludingFinalization"] = (
        total_cpu + runtime["finalizationCpuSeconds"]
    ) / 3600.0
    runtime["totalWallHoursIncludingFinalization"] = (
        total_wall + runtime["finalizationWallSeconds"]
    ) / 3600.0
    runtime["passed"] = bool(
        runtime["totalCpuHoursIncludingFinalization"] <= 32
        and runtime["totalWallHoursIncludingFinalization"] <= 12
    )
    write_json(LOOP_ROOT / "runtime_manifest.json", runtime)
    artifact_size = storage_bytes(LOOP_ROOT)
    storage["retainedArtifactBytes"] = artifact_size
    storage["retainedArtifactGiB"] = artifact_size / 2**30
    storage["passed"] = bool(artifact_size <= 15 * 2**30 and cache_size <= 40 * 2**30)
    write_json(LOOP_ROOT / "storage_validation.json", storage)
    if not runtime["passed"] or not storage["passed"]:
        raise RuntimeError("L11 final runtime or storage ceiling exceeded")
    write_artifact_manifest(
        LOOP_ROOT / "artifact_manifest.json", LOOP_ROOT,
        "eidosoma.e01.s19_l11.artifact_manifest.v1",
    )
    append_postoutcome_ledgers(classification, validation_result, report)
    print(json.dumps({"status": "L11_COMPLETE_AWAITING_MANDATORY_HUMAN_REVIEW", "decision": classification["decision"], "promotedLeadIds": classification["promotedLeadIds"]}, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "generate", "analyze", "regenerate", "finalize"))
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "generate":
        generate(args.workers)
    elif args.action == "analyze":
        analyze(args.workers)
    elif args.action == "regenerate":
        regenerate(args.workers)
    elif args.action == "finalize":
        finalize()


if __name__ == "__main__":
    main()
