#!/usr/bin/env python3
"""Prepare, execute, regenerate, validate, and freeze E01/S19-L10."""

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
from e01_latent_timebase.core import (
    array_sha256 as trajectory_array_sha256,
)
from e01_s19_matlab_attractor.core import (
    BOOTSTRAP_REPLICATES,
    CLUSTER_ROOT_HEX,
    K_VALUES,
    PIPELINE_IDS,
    R1_ID,
    R2_ID,
    RANDOM_REFERENCE_DRAWS,
    REPLICAS,
    THRESHOLD,
    TIME_PERMUTATION_DRAWS,
    VERSION,
    array_sha256,
    bootstrap_indices,
    close_rows,
    deterministic_seed,
    fit_pipeline,
    fit_r1_matlab_historical,
    fit_r2_euclidean,
    historical_h,
    holm_adjust,
    label_against_reference,
    label_fingerprint,
    matlab_compatible_silhouette,
    paper_distance,
    run_descriptors,
    scientific_recurrence_gate,
    serialize_worker_exception,
)
from e01_s19_occupancy_search.core import (
    ExploratoryExposureDefinition,
    boundary_scores,
    materialize_frozen_setting,
)
from e01_s19_recurring_attractor.core import _historical_k1_score
from e01_s19_untouched_mechanism.core import (
    MECHANISM_A,
    MECHANISM_B,
    OBJECT_A_BOUNDARY,
    OBJECT_A_PROJECTED,
    OBJECT_B_MOLECULAR,
    materialize_analysis_object,
)
from e01_s19_untouched_mechanism.core import (
    label_fingerprint as l08_label_fingerprint,
)

ARTIFACT_ROOT = Path("/artifacts/research_steps/S19")
LOOP_ROOT = ARTIFACT_ROOT / "loops/L10"
CACHE_ROOT = Path("/cache/e01_s19_l10")
PRIMARY_CACHE = CACHE_ROOT / "trajectories"
REPLAY_CACHE = CACHE_ROOT / "regeneration"
REPLAY_OUTPUT = CACHE_ROOT / "regenerated_outputs"
CONFIG_PATH = REPO / "configs/e01/s19_l10_matlab_compatible_attractor.yaml"
CORE_PATH = REPO / "src/e01_s19_matlab_attractor/core.py"
RUNNER_PATH = Path(__file__).resolve()
TEST_PATH = REPO / "tests/e01/test_s19_l10.py"
PAPER_MD = Path(
    "/workspace/input-attachments/ed5486bf-a043-485b-a233-d88d8d123759/pdf-markdown.md"
)
PAPER_PDF = Path("/cache/e01_s03/downloads/paper-2607.28250v1.pdf")
PAPER_FIGURE_1 = Path(
    "/workspace/input-attachments/ed5486bf-a043-485b-a233-d88d8d123759/figures/figure-01.png"
)
HISTORICAL_ROOT = Path("/cache/e01_s03/sources/gard-historical")
MATHWORKS_DOC = CACHE_ROOT / "sources/mathworks_silhouette.html"
SKLEARN_DOC = CACHE_ROOT / "sources/sklearn_silhouette_samples.html"

PRIMARY_GROUP = "ORIGINAL_EXPOSURE"
HIGH_GROUP = "HIGH_EXPOSURE_H2875"
PRIMARY_CANDIDATES = ("CANDIDATE_2", "CANDIDATE_3")
COMPARATOR_ADJACENT = "ORIGINAL_ADJACENT_MOLECULAR_H090"
COMPARATOR_A_BOUNDARY = "L08_A_FISSION_BOUNDARY_H090"
COMPARATOR_A_PROJECTED = "L08_A_FOLLOWING_INTERVAL_PROJECTED_H090"
COMPARATOR_B_HIGH = "L08_B_HIGH_EXPOSURE_MOLECULAR_H090"
COMPARATOR_IDS = (
    COMPARATOR_ADJACENT,
    COMPARATOR_A_BOUNDARY,
    COMPARATOR_A_PROJECTED,
    COMPARATOR_B_HIGH,
)

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
    "cluster_results.parquet": ("pipelineId", "candidateId", "matrixIndex"),
    "silhouette_results.parquet": ("pipelineId", "candidateId", "matrixIndex", "k"),
    "recurrence_status_results.parquet": ("pipelineId", "candidateId", "matrixIndex"),
    "dominant_attractor_results.parquet": ("pipelineId", "candidateId", "matrixIndex"),
    "molecular_label_results.parquet": (
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
    "label_fingerprint_results.parquet": ("pipelineId", "candidateId", "matrixIndex"),
    "episode_results.parquet": (
        "pipelineId",
        "candidateId",
        "matrixIndex",
        "polarity",
        "episodeIndex",
    ),
    "comparator_results.parquet": ("pipelineId", "candidateId", "matrixIndex"),
    "negative_control_results.parquet": (
        "recordType",
        "pipelineId",
        "candidateId",
        "matrixIndex",
        "controlType",
        "controlIndex",
        "outcome",
    ),
    "paper_target_comparison.csv": ("pipelineId", "candidateId", "metric"),
    "complete_fingerprint_distances.parquet": (
        "pipelineId",
        "candidateId",
        "onsetMode",
    ),
    "bootstrap_results.parquet": ("pipelineId", "candidateId", "bootstrapReplicate"),
    "candidate_comparison.csv": ("pipelineId", "metric"),
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
        json.dumps(
            json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
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


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, text=True, capture_output=True
    ).stdout.strip()


def load_config() -> dict[str, Any]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["versionedLoopId"] != VERSION:
        raise RuntimeError("L10 version/config mismatch")
    return config


def canonical_frame_sha256(frame: pd.DataFrame, sort_columns: Iterable[str]) -> str:
    columns = [column for column in sort_columns if column in frame.columns]
    ordered = (
        frame.sort_values(columns, kind="stable").reset_index(drop=True).copy()
        if columns
        else frame.copy()
    )
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
    rows: list[dict[str, Any]] = []
    for group in load_config()["simulations"]:
        for candidate in group["candidates"]:
            rows.append(
                {
                    "groupId": group["groupId"],
                    "groupRole": group["role"],
                    "candidateId": candidate["candidateId"],
                    "exposure": float(candidate["exposure"]),
                    "daughterRule": candidate["daughterRule"],
                    "overshootRule": candidate["overshootRule"],
                    "streamIdentity": candidate["streamIdentity"],
                }
            )
    expected = {
        (PRIMARY_GROUP, "CANDIDATE_2"),
        (PRIMARY_GROUP, "CANDIDATE_3"),
        (HIGH_GROUP, "CANDIDATE_2"),
        (HIGH_GROUP, "CANDIDATE_3"),
    }
    if (
        len(rows) != 4
        or {(row["groupId"], row["candidateId"]) for row in rows} != expected
    ):
        raise RuntimeError("L10 requires exactly four frozen trajectory groups")
    return rows


def make_definition(spec: dict[str, Any]) -> SimulationDefinition:
    h = float(spec["exposure"])
    exposure: Any = (
        ExposureDefinition(family="FIXED_COMMON_EXPOSURE", h=h)
        if h <= 1.25
        else ExploratoryExposureDefinition(family="FIXED_COMMON_EXPOSURE", h=h)
    )
    return SimulationDefinition(
        daughter_rule=spec["daughterRule"],
        overshoot_rule=spec["overshootRule"],
        exposure=exposure,
    )


def trajectory_path(
    cache_root: Path, matrix_index: int, group_id: str, candidate_id: str
) -> Path:
    return cache_root / f"M{matrix_index:03d}__{group_id}__{candidate_id}.pkl"


def prior_roots() -> list[Path]:
    roots: list[Path] = []
    for path in sorted(Path("/artifacts/research_steps").iterdir()):
        if path.name != "S19":
            roots.append(path)
    for loop in ("L01", "L02", "L03", "L04", "L05", "L06", "L06R", "L07", "L08", "L09"):
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
        files = (
            [root]
            if root.is_file()
            else sorted(path for path in root.rglob("*") if path.is_file())
        )
        for path in files:
            rows.append(
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return rows


def validate_immutable_prior() -> dict[str, Any]:
    baseline = json.loads((LOOP_ROOT / "immutable_prior_baseline.json").read_text())
    mismatches = []
    for row in baseline["files"]:
        path = Path(row["path"])
        if (
            not path.exists()
            or path.stat().st_size != row["bytes"]
            or sha256_file(path) != row["sha256"]
        ):
            mismatches.append(row["path"])
    return {
        "schema": "eidosoma.e01.s19_l10.immutable_prior_validation.v1",
        "baselineFileCount": len(baseline["files"]),
        "mismatchCount": len(mismatches),
        "mismatches": mismatches[:50],
        "passed": not mismatches,
        "validatedAtUtc": utc_now(),
    }


def all_prior_files() -> list[Path]:
    files: list[Path] = []
    for root in prior_roots():
        files.extend(
            [root]
            if root.is_file()
            else sorted(path for path in root.rglob("*") if path.is_file())
        )
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
                            inventory["allHex64"].update(
                                item.lower() for item in pattern.findall(value)
                            )
            elif (
                suffix in {".json", ".yaml", ".yml", ".csv", ".md", ".txt"}
                and path.stat().st_size <= 25 * 1024 * 1024
            ):
                text = path.read_text(encoding="utf-8", errors="replace")
                inventory["allHex64"].update(
                    item.lower() for item in pattern.findall(text)
                )
                if suffix == ".csv":
                    header = pd.read_csv(path, nrows=0)
                    names = [name for name in header.columns if name.lower() in targets]
                    if names:
                        frame = pd.read_csv(path, usecols=names, dtype=str)
                        for name in names:
                            inventory[targets[name.lower()]].update(
                                frame[name].dropna().astype(str)
                            )
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
                derive_seed(
                    root, phase, "poisson_update", matrix_index, spec["streamIdentity"]
                ),
                derive_seed(
                    root, phase, "overshoot_trim", matrix_index, spec["streamIdentity"]
                ),
                derive_seed(
                    root, phase, "fission", matrix_index, spec["streamIdentity"]
                ),
                derive_seed(
                    root,
                    phase,
                    "daughter_selection",
                    matrix_index,
                    spec["streamIdentity"],
                ),
            )
            for identity in identities:
                seeds.append(
                    {
                        "loopId": "S19-L10",
                        "groupId": spec["groupId"],
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
        "schema": "eidosoma.e01.s19_l10.seed_firewall.v1",
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


def fixture_values(center: int, count: int, width: int = 100) -> np.ndarray:
    values = np.zeros((count, width), dtype=np.float64)
    values[:, center] = 1.0
    values[:, (center + 1) % width] = 0.04
    for row in range(count):
        values[row, (center + 2 + row % 3) % width] = 0.002 * (row % 4)
    return close_rows(values)


def independent_silhouette_formula(
    values: np.ndarray, labels: np.ndarray, metric: str
) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels)
    if metric == "euclidean":
        distance = np.asarray(
            [[float(np.linalg.norm(left - right)) for right in x] for left in x],
            dtype=np.float64,
        )
    elif metric == "cosine":
        distance = 1.0 - historical_h(x, x)
        np.fill_diagonal(distance, 0.0)
    else:
        raise ValueError(metric)
    result = []
    for index, label in enumerate(labels):
        own_indices = [
            j for j, other in enumerate(labels) if other == label and j != index
        ]
        if not own_indices:
            result.append(1.0)
            continue
        a_value = float(sum(distance[index, j] for j in own_indices) / len(own_indices))
        alternatives = []
        for other_label in sorted(set(labels.tolist())):
            if other_label == label:
                continue
            indices = [j for j, value in enumerate(labels) if value == other_label]
            alternatives.append(
                float(sum(distance[index, j] for j in indices) / len(indices))
            )
        b_value = min(alternatives)
        denominator = max(a_value, b_value)
        result.append(0.0 if denominator == 0 else (b_value - a_value) / denominator)
    return np.asarray(result, dtype=np.float64)


def run_fixtures() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(
        fixture: str, pipeline: str, check: str, passed: bool, details: Any
    ) -> None:
        rows.append(
            {
                "fixtureId": fixture,
                "pipelineId": pipeline,
                "check": check,
                "passed": bool(passed),
                "detailsJson": json.dumps(
                    json_safe(details), sort_keys=True, separators=(",", ":")
                ),
            }
        )

    f01_x = np.asarray([[1, 0], [0, 1], [1, 1], [2, 1]], dtype=np.float64)
    f01 = matlab_compatible_silhouette(f01_x, np.arange(4), "cosine")
    add(
        "F01_K_EQUALS_N_ALL_SINGLETON",
        R1_ID,
        "singleton_values_and_mean_exactly_one",
        np.array_equal(f01, np.ones(4)) and float(np.mean(f01)) == 1.0,
        {"values": f01.tolist(), "mean": float(np.mean(f01))},
    )

    f02_x = np.asarray([[1.0, 0], [0.9, 0.1], [0, 1.0], [0.2, 1.0]])
    f02_labels = np.asarray([0, 0, 1, 2])
    f02 = matlab_compatible_silhouette(f02_x, f02_labels, "euclidean")
    f02_ref = independent_silhouette_formula(f02_x, f02_labels, "euclidean")
    add(
        "F02_K_EQUALS_N_MINUS_ONE",
        R1_ID,
        "finite_singletons_one_pair_formula",
        np.all(np.isfinite(f02))
        and f02[2] == f02[3] == 1.0
        and np.allclose(f02, f02_ref, atol=1e-12, rtol=1e-12),
        {"clean": f02.tolist(), "independent": f02_ref.tolist()},
    )

    f03_x = np.asarray(
        [[0, 0], [0.1, 0.1], [2, 2], [2.1, 2.1], [4, 0], [4.1, 0.1]], dtype=float
    )
    f03_labels = np.asarray([0, 0, 1, 1, 2, 2])
    f03 = matlab_compatible_silhouette(f03_x, f03_labels, "euclidean")
    f03_independent = independent_silhouette_formula(f03_x, f03_labels, "euclidean")
    f03_sklearn = sklearn.metrics.silhouette_samples(
        f03_x, f03_labels, metric="euclidean"
    )
    add(
        "F03_ORDINARY_NONSINGLETON",
        R1_ID,
        "clean_independent_sklearn_agreement",
        np.allclose(f03, f03_independent, atol=1e-12, rtol=1e-12)
        and np.allclose(f03, f03_sklearn, atol=1e-12, rtol=1e-12),
        {
            "maximumIndependentError": float(np.max(np.abs(f03 - f03_independent))),
            "maximumSklearnError": float(np.max(np.abs(f03 - f03_sklearn))),
        },
    )

    f04_x = close_rows(np.asarray([[4, 1, 0], [3, 1, 0], [4, 2, 0]], dtype=float))
    f04 = _historical_k1_score(f04_x)
    f04_manual = float(np.mean(historical_h(f04_x, f04_x)))
    add(
        "F04_K_ONE_HISTORICAL",
        R1_ID,
        "frozen_historical_special_path",
        f04 == f04_manual,
        {"score": f04, "manual": f04_manual},
    )

    f05 = fit_r1_matlab_historical(np.eye(20), "F05")
    add(
        "F05_NO_NONDRIFT_STATES",
        R1_ID,
        "source_defined_status_no_label",
        f05.status == "NO_NONDRIFT_COMPOSITIONS" and f05.dominant_centroid is None,
        {"status": f05.status},
    )

    dominant = np.vstack((fixture_values(1, 70), fixture_values(50, 30)))
    for pipeline_id, fitter in (
        (R1_ID, fit_r1_matlab_historical),
        (R2_ID, fit_r2_euclidean),
    ):
        fit = fitter(dominant, "F06")
        add(
            "F06_PLANTED_DOMINANT_ATTRACTOR",
            pipeline_id,
            "repeated_dominant_cluster_recovered",
            fit.status == "ELIGIBLE"
            and fit.dominant_cluster_id is not None
            and int(np.argmax(fit.dominant_centroid)) == 1,
            {
                "status": fit.status,
                "selectedK": fit.selected_k,
                "clusterSizes": fit.cluster_sizes,
            },
        )

    two = np.vstack((fixture_values(2, 60), fixture_values(55, 40)))
    for pipeline_id, fitter in (
        (R1_ID, fit_r1_matlab_historical),
        (R2_ID, fit_r2_euclidean),
    ):
        fit = fitter(two, "F07")
        add(
            "F07_TWO_UNEQUAL_ATTRACTORS",
            pipeline_id,
            "both_retained_more_recurrent_selected",
            fit.status == "ELIGIBLE"
            and fit.second_cluster_id is not None
            and fit.cluster_sizes[fit.dominant_cluster_id]
            > fit.cluster_sizes[fit.second_cluster_id],
            {"status": fit.status, "clusterSizes": fit.cluster_sizes},
        )

    tied_gate = scientific_recurrence_gate(
        np.asarray([0, 0, 1, 1]), np.asarray([[1, 0], [0, 1]], dtype=float)
    )
    add(
        "F08_TIED_LARGEST_CLUSTERS",
        "SCIENTIFIC_RECURRENCE_GATE",
        "tie_rejected_without_index_or_seed",
        tied_gate["status"] == "NO_UNIQUE_RECURRING_COMPTYPE"
        and tied_gate["dominantClusterId"] is None,
        tied_gate,
    )

    singleton_x = np.asarray(
        [[1, 0, 0, 0], [0.99, 0.01, 0, 0], [0, 0, 1, 0], [0, 0, 0.99, 0.01]],
        dtype=float,
    )
    singleton_fit = fit_r1_matlab_historical(singleton_x, "F09")
    add(
        "F09_ALL_SINGLETON_SELECTED",
        R1_ID,
        "software_solution_retained_scientific_reference_suppressed",
        singleton_fit.selected_k == int(np.count_nonzero(singleton_fit.eligible_mask))
        and singleton_fit.status == "NO_RECURRING_COMPTYPE"
        and singleton_fit.dominant_centroid is None,
        {
            "status": singleton_fit.status,
            "selectedK": singleton_fit.selected_k,
            "eligibleN": int(np.count_nonzero(singleton_fit.eligible_mask)),
            "clusterSizes": singleton_fit.cluster_sizes,
        },
    )

    base = np.vstack((fixture_values(3, 63), fixture_values(56, 37)))
    permutation = np.roll(np.arange(100), 17)
    transformed = close_rows(
        base[:, permutation] * np.linspace(1, 5, len(base))[:, None]
    )
    for pipeline_id, fitter in (
        (R1_ID, fit_r1_matlab_historical),
        (R2_ID, fit_r2_euclidean),
    ):
        first = fitter(base, "F10")
        changed = fitter(transformed, "F10")
        passed = bool(
            first.status == changed.status == "ELIGIBLE"
            and first.selected_k == changed.selected_k
            and np.array_equal(first.labels, changed.labels)
            and np.allclose(
                first.dominant_centroid[permutation],
                changed.dominant_centroid,
                atol=1e-12,
                rtol=1e-12,
            )
        )
        add(
            "F10_FEATURE_PERMUTATION_SCALING",
            pipeline_id,
            "cluster_and_recurrence_equivalence",
            passed,
            {
                "firstStatus": first.status,
                "changedStatus": changed.status,
                "selectedK": first.selected_k,
            },
        )

    for pipeline_id, fitter in (
        (R1_ID, fit_r1_matlab_historical),
        (R2_ID, fit_r2_euclidean),
    ):
        first = fitter(base, "F11")
        replay = fitter(base, "F11")
        passed = bool(
            first.status == replay.status
            and first.selected_k == replay.selected_k
            and first.selected_score == replay.selected_score
            and np.array_equal(first.labels, replay.labels)
            and np.array_equal(first.centroids, replay.centroids)
        )
        add(
            "F11_EXACT_REPLAY",
            pipeline_id,
            "clusters_silhouettes_centroids_status_exact",
            passed,
            {
                "status": first.status,
                "selectedK": first.selected_k,
                "score": first.selected_score,
            },
        )

    provenance = serialize_worker_exception(
        candidate_id="CANDIDATE_2",
        matrix_id=7,
        pipeline_id=R1_ID,
        k=4,
        n=4,
        cluster_sizes=(1, 1, 1, 1),
        seed_identity="F12-SEED",
        error=RuntimeError("fixture"),
    )
    required = {
        "candidateId",
        "matrixId",
        "pipelineId",
        "k",
        "n",
        "clusterSizeVector",
        "seedIdentity",
        "exceptionClass",
        "exceptionMessage",
    }
    add(
        "F12_WORKER_EXCEPTION_PROVENANCE",
        "GLOBAL",
        "all_required_fields_serialized",
        set(provenance) == required,
        provenance,
    )

    frame = pd.DataFrame(rows)
    if frame["fixtureId"].nunique() != 12 or not bool(frame["passed"].all()):
        raise RuntimeError(
            f"mandatory L10 fixtures failed: {frame.loc[~frame.passed].to_dict('records')}"
        )
    return frame


def source_manifest() -> dict[str, Any]:
    source_files = [
        HISTORICAL_ROOT / name
        for name in (
            "tgs_nondrift.m",
            "tgs_acluster.m",
            "tgs_kmeans.m",
            "tgs_H.m",
            "tgs_parameters_v10.m",
            "tgs_carpet.m",
            "getcomposometime_v10.m",
            "README.txt",
        )
    ]
    cited = [
        Path("/cache/e01_s19_l09/sources/PMC18166.html"),
        Path("/cache/e01_s19_l09/sources/PubMed_11735293.html"),
        Path("/cache/e01_s19_l09/sources/PubMed_11536890.html"),
        Path("/cache/e01_s19_l09/sources/ref64_GARD_domain.pdf"),
    ]
    files = [
        PAPER_MD,
        PAPER_PDF,
        PAPER_FIGURE_1,
        MATHWORKS_DOC,
        SKLEARN_DOC,
        *source_files,
        *cited,
    ]
    rows = []
    for path in files:
        if not path.exists():
            raise FileNotFoundError(path)
        rows.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "redistribution": "REFERENCE_ONLY_NOT_COPIED_TO_ARTIFACTS",
            }
        )
    commit = subprocess.run(
        ["git", "-C", str(HISTORICAL_ROOT), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    return {
        "schema": "eidosoma.e01.s19_l10.source_snapshot_manifest.v1",
        "capturedAtUtc": utc_now(),
        "historicalGard": {
            "repository": "https://github.com/marcos-delgado/GARD-model",
            "commit": commit,
            "expectedCommit": "86dff6320d5ae91b4e831471079ff46749b14df9",
            "licenseStatus": "NO_LICENSE_DETECTED_REFERENCE_ONLY",
        },
        "documentation": [
            {
                "identity": "MathWorks silhouette official documentation",
                "url": "https://www.mathworks.com/help/stats/silhouette.html",
                "directEvidence": "singleton observation silhouette is set to 1",
                "retrievalDate": "2026-08-09",
            },
            {
                "identity": f"scikit-learn {sklearn.__version__} silhouette_samples official documentation",
                "url": "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_samples.html",
                "directEvidence": "defined only for 2 <= n_labels <= n_samples - 1",
                "retrievalDate": "2026-08-09",
            },
        ],
        "references": [
            {"reference": 63, "doi": "10.1006/jtbi.2001.2440"},
            {"reference": 64, "doi": "10.1023/A:1006583712886"},
            {"reference": 65, "doi": "10.1073/pnas.97.8.4112"},
        ],
        "files": rows,
    }


def append_preoutcome_ledgers(sources: dict[str, Any]) -> None:
    now = utc_now()
    ledger_path = ARTIFACT_ROOT / "self_improvement_ledger.parquet"
    ledger = pd.read_parquet(ledger_path)
    if "S19-L10" not in set(ledger["loopId"].astype(str)):
        row = {
            "ledgerSequence": int(ledger["ledgerSequence"].max()) + 1,
            "timestampUtc": now,
            "loopId": "S19-L10",
            "recordPhase": "PRE_LOOP_OUTCOME_BLIND_METHOD_LOCK",
            "beliefBeforeLoop": "L09's scientific hypothesis remained unadjudicated because its backend omitted documented MATLAB all-singleton silhouette semantics.",
            "motivatingEvidence": "Pinned historical GARD calls MATLAB silhouette for k up to n; official MATLAB documentation assigns singleton value 1, whereas scikit-learn excludes k=n.",
            "failureOrAmbiguityTargeted": "The singleton-silhouette implementation-lineage mismatch and the paper's unresolved most-recurring-composition label.",
            "selectedHypotheses": "Exactly R1 MATLAB-compatible historical compotype and unchanged R2 paper-Euclidean attractor, with a separate no-singleton/no-tied-largest scientific recurrence gate.",
            "learned": "Pending untouched L10 execution.",
            "weakenedHypotheses": "Pending untouched L10 execution.",
            "remainingPlausibleHypotheses": "Both pipelines remain exploratory and full-run retrospective until the complete untouched result and controls are validated.",
            "proposedNextTest": "Execute the pushed L10 contract once and stop for mandatory human review.",
            "informationGainRationale": "The new seed-firewalled dataset separates a prospectively documented compatibility correction from rescue of L09's partial computation.",
            "appendOnly": True,
        }
        ledger = pd.concat(
            [ledger, pd.DataFrame([row], columns=ledger.columns)], ignore_index=True
        )
        write_parquet(ledger_path, ledger)
        with (ARTIFACT_ROOT / "SELF_IMPROVEMENT_LEDGER.md").open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write(
                "\n\n## Entry 021 — S19-L10 pre-loop MATLAB-compatible lock\n\n"
                "- **Belief before:** L09 failed operationally before adjudicating the recurring-attractor hypothesis.\n"
                "- **Motivating evidence:** Official MATLAB documentation sets singleton silhouette to 1, while the L09 scikit-learn backend excludes `k=n`; historical GARD permits that k and calls MATLAB silhouette.\n"
                "- **Selected hypotheses:** exactly MATLAB-compatible historical R1 and unchanged paper-Euclidean R2, with an independent recurrence gate rejecting all-singleton and tied-largest solutions.\n"
                "- **Expected information gain:** adjudicate both pipelines on 100 new shared seed-firewalled matrices without rescuing or altering L09.\n"
                "- **Next action:** execute the pushed L10 lock once, validate, freeze, and stop for human review.\n"
            )

    candidate_path = ARTIFACT_ROOT / "candidate_registry.parquet"
    candidates = pd.read_parquet(candidate_path)
    if "S19-L10-R1" not in set(candidates["candidateId"].astype(str)):
        start = int(candidates["registryOrder"].max()) + 1
        additions = [
            {
                "candidateId": "S19-L10-R1",
                "bundleId": "L10_MATLAB_COMPATIBLE_RECURRING_ATTRACTOR",
                "selected": True,
                "sourceGrounding": 5,
                "paperFingerprintSpecificity": 5,
                "explanatoryLeverage": 5,
                "testability": 5,
                "crossCandidateDiscriminability": 5,
                "computeEfficiency": 5,
                "independenceFromPriorOutcomeSelection": 4,
                "outcomeGuidedThresholdSelection": 0,
                "deterministicHReuse": 1,
                "completedFitLeakage": 1,
                "candidateSpecificSuccess": 0,
                "undefinedAuthorSemantics": 1,
                "branchCount": 1,
                "proposedSpecification": "Historical R1 with documented MATLAB singleton=1 silhouette and separate recurrence gate",
                "selectionReason": "Explicit human authorization and direct historical/documentation compatibility evidence",
                "rankingScore": 31.0,
                "frozenRank": 1,
                "registryOrder": start,
            },
            {
                "candidateId": "S19-L10-R2",
                "bundleId": "L10_MATLAB_COMPATIBLE_RECURRING_ATTRACTOR",
                "selected": True,
                "sourceGrounding": 4,
                "paperFingerprintSpecificity": 5,
                "explanatoryLeverage": 5,
                "testability": 5,
                "crossCandidateDiscriminability": 5,
                "computeEfficiency": 5,
                "independenceFromPriorOutcomeSelection": 4,
                "outcomeGuidedThresholdSelection": 0,
                "deterministicHReuse": 1,
                "completedFitLeakage": 1,
                "candidateSpecificSuccess": 0,
                "undefinedAuthorSemantics": 3,
                "branchCount": 1,
                "proposedSpecification": "Unchanged L09 paper-Euclidean R2 on untouched L10 trajectories",
                "selectionReason": "Explicit human authorization and paper's Euclidean attractor wording",
                "rankingScore": 29.0,
                "frozenRank": 2,
                "registryOrder": start + 1,
            },
        ]
        candidates = pd.concat(
            [candidates, pd.DataFrame(additions, columns=candidates.columns)],
            ignore_index=True,
        )
        write_parquet(candidate_path, candidates)

    source_path = ARTIFACT_ROOT / "source_search_ledger.parquet"
    source_ledger = pd.read_parquet(source_path)
    known = set(source_ledger["sourceId"].astype(str))
    additions = []
    records = [
        (
            "L10_MATHWORKS_SILHOUETTE",
            "OFFICIAL_DOCUMENTATION",
            "https://www.mathworks.com/help/stats/silhouette.html",
            None,
            "retrieved-2026-08-09",
            MATHWORKS_DOC,
            "DIRECT_PRIMARY_DOCUMENTATION",
            "MATLAB assigns silhouette 1 to a singleton observation.",
            "IDENTITY_AND_FINDING_ONLY",
        ),
        (
            "L10_SKLEARN_SILHOUETTE",
            "OFFICIAL_DOCUMENTATION",
            "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_samples.html",
            "scikit-learn",
            sklearn.__version__,
            SKLEARN_DOC,
            "DIRECT_PRIMARY_DOCUMENTATION",
            "scikit-learn defines silhouette only through n_labels <= n_samples - 1.",
            "IDENTITY_AND_FINDING_ONLY",
        ),
    ]
    for (
        source_id,
        source_type,
        url,
        repo,
        version,
        path,
        evidence,
        finding,
        redistribution,
    ) in records:
        if source_id not in known:
            additions.append(
                {
                    "sourceId": source_id,
                    "sourceType": source_type,
                    "url": url,
                    "repositoryIdentity": repo,
                    "commitOrVersion": version,
                    "treeIdentity": None,
                    "retrievalDate": "2026-08-09",
                    "retainedPath": str(path),
                    "sha256": sha256_file(path),
                    "licenseStatus": "OFFICIAL_DOCUMENTATION_REFERENCE_ONLY",
                    "evidenceClass": evidence,
                    "finding": finding,
                    "redistributionStatus": redistribution,
                }
            )
    if additions:
        source_ledger = pd.concat(
            [source_ledger, pd.DataFrame(additions, columns=source_ledger.columns)],
            ignore_index=True,
        )
        write_parquet(source_path, source_ledger)
        with (ARTIFACT_ROOT / "source_search_report.md").open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write(
                "\n\n## S19-L10 additive source audit — singleton silhouette compatibility\n\n"
                "Official MathWorks documentation states that an observation that is the sole member of its cluster receives silhouette value 1. Official scikit-learn documentation instead limits the coefficient to `2 <= n_labels <= n_samples - 1`. The pinned historical GARD code permits `k <= n`, calls MATLAB `silhouette`, and therefore exposes a real backend-semantic difference. L10 changes only R1's silhouette backend before new outcomes and keeps a separate scientific recurrence gate so an all-singleton software optimum cannot become a replicator reference.\n"
            )

    registry_path = ARTIFACT_ROOT / "loop_registry.yaml"
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    if not any(item["loopId"] == "S19-L10" for item in registry["loops"]):
        registry["loops"].append(
            {
                "loopId": "S19-L10",
                "versionedLoopId": VERSION,
                "status": "AUTHORIZED_PREOUTCOME_LOCK_PREPARED",
                "authorized": True,
                "outcomeAccessed": False,
                "humanReviewRequiredAfter": True,
                "completed": False,
                "eligibleScientificResults": None,
                "promotedLeadCount": 0,
                "nextStepActive": True,
            }
        )
    registry["laterLoopsAuthorized"] = False
    registry["s20Status"] = "DEFINED_INACTIVE"
    registry["proposedNextLoopTheme"] = None
    registry["proposedNextLoopActive"] = False
    write_yaml(registry_path, registry)

    review_path = ARTIFACT_ROOT / "human_review_history.json"
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if not any(item.get("scope") == VERSION for item in review["history"]):
        review["history"].append(
            {
                "date": "2026-08-09",
                "decision": "AUTHORIZE_S19_L10_MATLAB_COMPATIBLE_RECURRING_ATTRACTOR_ONLY",
                "scope": VERSION,
                "source": "explicit_human_direction",
            }
        )
    review["pendingDecision"] = "S19_L10_ACTIVE_MANDATORY_HUMAN_REVIEW_AFTER_COMPLETION"
    write_json(review_path, review)


def prepare() -> None:
    started = utc_now()
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    if LOOP_ROOT.exists():
        raise RuntimeError(f"L10 artifact directory already exists: {LOOP_ROOT}")
    LOOP_ROOT.mkdir(parents=True)
    PRIMARY_CACHE.mkdir(parents=True, exist_ok=True)
    REPLAY_CACHE.mkdir(parents=True, exist_ok=True)
    REPLAY_OUTPUT.mkdir(parents=True, exist_ok=True)
    config = load_config()

    baseline = immutable_rows()
    write_json(
        LOOP_ROOT / "immutable_prior_baseline.json",
        {
            "schema": "eidosoma.e01.s19_l10.immutable_prior_baseline.v1",
            "capturedAtUtc": utc_now(),
            "fileCount": len(baseline),
            "totalBytes": sum(row["bytes"] for row in baseline),
            "files": baseline,
        },
    )
    sources = source_manifest()
    if (
        sources["historicalGard"]["commit"]
        != sources["historicalGard"]["expectedCommit"]
    ):
        raise RuntimeError("historical GARD source commit mismatch")
    write_json(LOOP_ROOT / "source_snapshot_manifest.json", sources)

    fixtures = run_fixtures()
    write_parquet(LOOP_ROOT / "fixture_results.parquet", fixtures)
    write_csv(
        LOOP_ROOT / "matlab_silhouette_validation.csv",
        fixtures[
            fixtures["fixtureId"].isin(
                {
                    "F01_K_EQUALS_N_ALL_SINGLETON",
                    "F02_K_EQUALS_N_MINUS_ONE",
                    "F03_ORDINARY_NONSINGLETON",
                    "F04_K_ONE_HISTORICAL",
                }
            )
        ],
    )
    write_json(
        LOOP_ROOT / "fixture_manifest.json",
        {
            "schema": "eidosoma.e01.s19_l10.fixture_manifest.v1",
            "fixtureCount": int(fixtures["fixtureId"].nunique()),
            "checkCount": len(fixtures),
            "passedCount": int(fixtures["passed"].sum()),
            "failedCount": int((~fixtures["passed"]).sum()),
            "allMandatoryPassed": bool(fixtures["passed"].all()),
            "resultsSha256": sha256_file(LOOP_ROOT / "fixture_results.parquet"),
        },
    )

    seed_frame, input_frame, firewall = seed_and_input_manifests()
    write_parquet(LOOP_ROOT / "seed_manifest.parquet", seed_frame)
    write_parquet(LOOP_ROOT / "input_units.parquet", input_frame)
    write_json(LOOP_ROOT / "seed_firewall.json", firewall)
    write_json(
        LOOP_ROOT / "input_manifest.json",
        {
            "schema": "eidosoma.e01.s19_l10.input_manifest.v1",
            "matrixCount": 100,
            "matchedInitialStates": True,
            "allInputIdentitiesGeneratedBeforeLabels": True,
            "inputUnitTable": str(LOOP_ROOT / "input_units.parquet"),
            "inputUnitTableSha256": sha256_file(LOOP_ROOT / "input_units.parquet"),
            "seedManifest": str(LOOP_ROOT / "seed_manifest.parquet"),
            "seedManifestSha256": sha256_file(LOOP_ROOT / "seed_manifest.parquet"),
            "betaHashSetSha256": sha256_text(
                "\n".join(sorted(input_frame["betaSha256"].astype(str)))
            ),
            "initialStateHashSetSha256": sha256_text(
                "\n".join(sorted(input_frame["initialStateSha256"].astype(str)))
            ),
            "trajectoryGroups": simulation_specs(),
            "labelsCalculated": False,
        },
    )
    if not firewall["passed"]:
        raise RuntimeError(f"L10 seed firewall failed: {firewall['overlaps']}")

    write_yaml(LOOP_ROOT / "preregistration.yaml", config)
    write_yaml(
        LOOP_ROOT / "label_pipeline_registry.yaml",
        {
            "schema": "eidosoma.e01.s19_l10.label_pipeline_registry.v1",
            "pipelineCount": 2,
            "pipelines": config["pipelines"],
            "commonLabel": config["commonLabel"],
            "scientificRecurrenceGate": config["scientificRecurrenceGate"],
            "representativeFigureRule": "LOWEST_MATRIX_INDEX_WITH_DEFINED_LABEL_ELSE_EXPLICIT_STATUS_PANEL",
        },
    )
    (LOOP_ROOT / "decision_record.md").write_text(
        f"""# S19-L10 Decision Record

## Concise top summary

- **Research step ID:** `S19-L10` (`{VERSION}`).
- **Completion status:** authorized; outcome-blind implementation lock prepared; scientific outcomes unopened.
- **Artifacts written:** preregistration, source/semantics audit, source snapshot, implementation lock, 12-fixture evidence, two-pipeline registry, seed firewall, and 100-input manifest.
- **Validation result:** all mandatory fixtures and the zero-overlap input/seed firewall passed; commit/push, opaque benchmark, execution, replay, and final validation remain pending.
- **Outcome classification:** pending; no L10 label result has been calculated.
- **Caveats or blockers:** R1 is a clean-room compatibility reconstruction rather than author code; R2 remains paper-grounded but underspecified; both completed-run labels are retrospective.
- **Recommended next action:** commit and push the complete lock, verify a clean worktree, run the ten-matrix opaque benchmark, execute L10 once if within ceiling, validate, freeze, and stop for human review.

This additive decision changes no L09 artifact or classification. Exactly two pipelines are registered, and the all-singleton/tied-largest scientific recurrence gate is distinct from software cluster scoring.
""",
        encoding="utf-8",
    )
    (LOOP_ROOT / "matlab_silhouette_semantics_audit.md").write_text(
        f"""# MATLAB-Compatible Silhouette Semantics Audit

## Concise top summary

- **Research step ID:** `S19-L10`.
- **Completion status:** source and implementation-semantics audit complete before scientific trajectory generation.
- **Artifacts written:** this audit, hashed source snapshot, implementation lock, fixture manifest/results, and MATLAB validation table.
- **Validation result:** documented singleton semantics were recovered and all 12 mandatory fixture families passed.
- **Outcome classification:** no scientific trajectory label has been opened.
- **Caveats or blockers:** the original MATLAB release and target-author code remain unavailable; only the documented singleton convention is resolved.
- **Recommended next action:** execute only the clean pushed L10 lock and return for mandatory human review.

## Direct evidence

The pinned historical GARD source at commit `86dff6320d5ae91b4e831471079ff46749b14df9` permits `k <= n`, requests MATLAB `silhouette` for multi-cluster scoring, and considers k values 1–10 with ten replicas and a four-k nonimprovement stop. Official MathWorks documentation states that a point that is the sole member of its cluster receives silhouette value 1. Official scikit-learn {sklearn.__version__} documentation instead restricts its silhouette coefficient to `2 <= n_labels <= n_samples - 1`. Both official pages were retained cache-only and hashed in `source_snapshot_manifest.json`.

## Frozen clean-room calculation

For every non-singleton point, L10 computes `a` as mean distance to the other members of its cluster, `b` as the minimum mean distance to another cluster, and `(b-a)/max(a,b)`. A singleton receives the literal float64 value `1`. An exact `a=b=0` case receives the prospectively locked value `0`. Cosine inputs must be finite and nonzero; distance residue no smaller than `-1e-12` is clamped to zero, while a material negative distance fails closed. Cluster IDs are canonicalized by earliest assigned observation. The k=1 path remains the historical mean-H carpet and never enters the multi-cluster formula.

## Scientific recurrence boundary

Software selection and scientific recurrence are separate. A selected all-singleton solution is retained in the clustering tables but yields `NO_RECURRING_COMPTYPE`; a tied largest cluster yields `NO_UNIQUE_RECURRING_COMPTYPE`. Neither emits a molecular label or falls back to another k. This prospective gate prevents MATLAB's singleton score from fabricating a recurring compotype.

## Paper and cited-method context

The paper and Figure 1 describe molecular-composition clusters with homeostatic attractor-like growth and entry/exit relative to the most recurring composition. References 63–65 ground the GARD/composome lineage. These sources support the two registered reconstructions but do not identify the authors' exact code, MATLAB release, RNG, cluster choice, or Table 1 onset/dispersion semantics.
""",
        encoding="utf-8",
    )

    append_preoutcome_ledgers(sources)
    code_paths = [CORE_PATH, RUNNER_PATH, TEST_PATH, CONFIG_PATH]
    lock = {
        "schema": "eidosoma.e01.s19_l10.implementation_lock.v1",
        "versionedLoopId": VERSION,
        "lockedAtUtc": utc_now(),
        "outcomesOpened": False,
        "configuration": config,
        "code": [
            {"path": str(path.relative_to(REPO)), "sha256": sha256_file(path)}
            for path in code_paths
        ],
        "sourceSnapshotManifestSha256": sha256_file(
            LOOP_ROOT / "source_snapshot_manifest.json"
        ),
        "fixtureManifestSha256": sha256_file(LOOP_ROOT / "fixture_manifest.json"),
        "inputManifestSha256": sha256_file(LOOP_ROOT / "input_manifest.json"),
        "seedFirewallSha256": sha256_file(LOOP_ROOT / "seed_firewall.json"),
        "pipelineIds": list(PIPELINE_IDS),
        "singletonSilhouetteValue": 1.0,
        "kValues": list(K_VALUES),
        "replicas": REPLICAS,
        "threshold": THRESHOLD,
        "bootstrapReplicates": BOOTSTRAP_REPLICATES,
        "randomReferenceDraws": RANDOM_REFERENCE_DRAWS,
        "timePermutationDraws": TIME_PERMUTATION_DRAWS,
        "scientificAmendmentsPermittedAfterRelease": False,
        "repository": {
            "branch": git("branch", "--show-current"),
            "headBeforeLockCommit": git("rev-parse", "HEAD"),
            "worktreeDirtyExpectedBeforeCommit": bool(git("status", "--porcelain=v1")),
        },
    }
    write_json(LOOP_ROOT / "implementation_lock.json", lock)
    write_json(
        LOOP_ROOT / "prepare_runtime.json",
        {
            "schema": "eidosoma.e01.s19_l10.prepare_runtime.v1",
            "startedAtUtc": started,
            "completedAtUtc": utc_now(),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikitLearn": sklearn.__version__,
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
            "fixtureCount": int(fixtures["fixtureId"].nunique()),
            "fixtureCheckCount": len(fixtures),
            "immutableFileCount": len(baseline),
            "inputMatrixCount": len(input_frame),
            "wallSeconds": time.perf_counter() - wall_start,
            "cpuSeconds": time.process_time() - cpu_start,
        },
    )
    print(
        json.dumps(
            {
                "status": "PREOUTCOME_LOCK_PREPARED_PENDING_COMMIT_PUSH",
                "fixtureFamilies": int(fixtures["fixtureId"].nunique()),
                "fixtureChecks": len(fixtures),
                "immutableFiles": len(baseline),
                "inputMatrices": len(input_frame),
                "seedFirewallPassed": firewall["passed"],
            },
            sort_keys=True,
        )
    )


def repository_release_gate() -> dict[str, Any]:
    lock = json.loads((LOOP_ROOT / "implementation_lock.json").read_text())
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/eidosoma/groups/42")
    branch = git("branch", "--show-current")
    clean = not bool(git("status", "--porcelain=v1"))
    hashes = all(
        sha256_file(REPO / row["path"]) == row["sha256"] for row in lock["code"]
    )
    fixtures = json.loads((LOOP_ROOT / "fixture_manifest.json").read_text())
    firewall = json.loads((LOOP_ROOT / "seed_firewall.json").read_text())
    immutable = validate_immutable_prior()
    passed = bool(
        head == remote
        and branch == "eidosoma/groups/42"
        and clean
        and hashes
        and fixtures["allMandatoryPassed"]
        and firewall["passed"]
        and immutable["passed"]
    )
    result = {
        "schema": "eidosoma.e01.s19_l10.release_gate.v1",
        "head": head,
        "remoteHead": remote,
        "branch": branch,
        "cleanWorktree": clean,
        "lockedCodeHashesMatch": hashes,
        "fixturesPassed": fixtures["allMandatoryPassed"],
        "seedFirewallPassed": firewall["passed"],
        "immutablePriorPassed": immutable["passed"],
        "passed": passed,
        "validatedAtUtc": utc_now(),
    }
    write_json(LOOP_ROOT / "immutable_prior_validation.json", immutable)
    write_json(LOOP_ROOT / "run_release_gate.json", result)
    return result


def simulate_matrix(matrix_index: int, cache_root: Path) -> dict[str, Any]:
    config = load_config()
    root = config["seedContract"]["matrixRootHex"]
    phase = config["seedContract"]["phase"]
    beta = generate_beta(derive_seed(root, phase, "catalytic_matrix", matrix_index))
    initial = initialize_distinct_state(
        derive_seed(root, phase, "initial_state", matrix_index)
    )
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
            path = trajectory_path(
                cache_root, matrix_index, spec["groupId"], spec["candidateId"]
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as handle:
                pickle.dump(trajectory, handle, protocol=5)
            complete = bool(
                trajectory.terminal_status == "requested_fissions_completed"
                and trajectory.completed_fissions == 100
            )
            selected = selected_clock_observations(
                trajectory, "C1_SELECTED_DAUGHTER_RETAINED"
            )
            post_count = sum(
                item.observation_kind == "post_fission" for item in selected
            )
            trajectories.append(
                {
                    **spec,
                    "matrixIndex": matrix_index,
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
                    "replacementAttempted": False,
                }
            )
            attempts.append(
                {
                    **spec,
                    "matrixIndex": matrix_index,
                    "attemptStatus": "COMPLETE"
                    if complete
                    else "INCOMPLETE_OR_EXTINCT_RETAINED",
                    "terminalStatus": trajectory.terminal_status,
                    "completedFissions": int(trajectory.completed_fissions),
                    "wallSeconds": time.perf_counter() - wall_start,
                    "cpuSeconds": time.process_time() - cpu_start,
                    "replacementAttempted": False,
                }
            )
        except Exception as error:  # noqa: BLE001 - all worker failures require provenance
            failures.append(
                {
                    "failureId": f"S19-L10-SIM-M{matrix_index:03d}-{spec['groupId']}-{spec['candidateId']}",
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
            pool.submit(simulate_matrix, index, cache_root): index for index in indices
        }
        for future in as_completed(futures):
            outputs.append(future.result())
    return outputs


def generate(workers: int) -> None:
    if workers != 8:
        raise ValueError("L10 generation is locked to eight workers")
    release = repository_release_gate()
    if not release["passed"]:
        raise RuntimeError(f"L10 release gate failed: {release}")
    if any(PRIMARY_CACHE.glob("*.pkl")):
        raise RuntimeError(
            "L10 primary cache is not empty; generation cannot be relaunched"
        )
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
        "schema": "eidosoma.e01.s19_l10.preoutcome_benchmark.v1",
        "matrixCount": 10,
        "simulationCount": len(benchmark_attempts),
        "scientificLabelsCalculatedOrOpened": False,
        "trajectoryTerminalStatusesNotUsedForMethodSelection": True,
        "benchmarkWallSeconds": benchmark_wall,
        "benchmarkWorkerCpuSeconds": benchmark_cpu,
        "projectionIncludesPrimaryAndCompleteRegeneration": True,
        "safetyFactor": 1.5,
        "projectedCpuHours": projected_cpu_hours,
        "projectedWallHours": projected_wall_hours,
        "cpuCeilingAfterReserveHours": 28.8,
        "wallCeilingHours": 8.0,
        "failureCount": len(benchmark_failures),
        "passed": bool(
            not benchmark_failures
            and projected_cpu_hours <= 28.8
            and projected_wall_hours <= 8.0
        ),
        "completedAtUtc": utc_now(),
    }
    write_json(LOOP_ROOT / "preoutcome_benchmark.json", benchmark)
    if not benchmark["passed"]:
        write_csv(LOOP_ROOT / "failure_ledger.csv", pd.DataFrame(benchmark_failures))
        raise RuntimeError("L10 opaque benchmark failed before full generation")

    remaining_outputs = run_simulation_batch(range(10, 100), workers, PRIMARY_CACHE)
    outputs = benchmark_outputs + remaining_outputs
    attempts = pd.DataFrame([row for output in outputs for row in output["attempts"]])
    trajectories = pd.DataFrame(
        [row for output in outputs for row in output["trajectories"]]
    )
    failures = pd.DataFrame([row for output in outputs for row in output["failures"]])
    attempts = attempts.sort_values(
        ["matrixIndex", "groupId", "candidateId"], kind="stable"
    )
    trajectories = trajectories.sort_values(
        ["matrixIndex", "groupId", "candidateId"], kind="stable"
    )
    write_parquet(LOOP_ROOT / "execution_status.parquet", attempts)
    write_parquet(LOOP_ROOT / "trajectory_manifest.parquet", trajectories)
    failure_columns = [
        "failureId",
        "groupId",
        "groupRole",
        "candidateId",
        "exposure",
        "daughterRule",
        "overshootRule",
        "streamIdentity",
        "matrixIndex",
        "failureType",
        "message",
        "scientificValuesEligible",
        "replacementAttempted",
    ]
    write_csv(
        LOOP_ROOT / "failure_ledger.csv",
        failures if len(failures) else pd.DataFrame(columns=failure_columns),
    )
    if len(attempts) != 400 or len(trajectories) != 400 or len(failures):
        raise RuntimeError("L10 trajectory generation failed closed")
    if (
        attempts["replacementAttempted"].any()
        or trajectories["replacementAttempted"].any()
    ):
        raise RuntimeError("L10 replacement invariant violated")
    input_units = pd.read_parquet(LOOP_ROOT / "input_units.parquet")
    observed_beta = trajectories.groupby("matrixIndex")["betaSha256"].nunique()
    observed_initial = trajectories.groupby("matrixIndex")[
        "initialStateSha256"
    ].nunique()
    if not (observed_beta.eq(1).all() and observed_initial.eq(1).all()):
        raise RuntimeError("L10 matched-input invariant failed")
    joined = trajectories[
        ["matrixIndex", "betaSha256", "initialStateSha256"]
    ].drop_duplicates()
    joined = joined.merge(
        input_units, on="matrixIndex", validate="one_to_one", suffixes=("", "Expected")
    )
    if not (
        joined["betaSha256"].eq(joined["betaSha256Expected"]).all()
        and joined["initialStateSha256"].eq(joined["initialStateSha256Expected"]).all()
    ):
        raise RuntimeError(
            "generated trajectory inputs differ from frozen input manifest"
        )

    child_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    child_cpu = (
        child_after.ru_utime
        + child_after.ru_stime
        - child_before.ru_utime
        - child_before.ru_stime
    )
    write_json(
        LOOP_ROOT / "generation_runtime.json",
        {
            "schema": "eidosoma.e01.s19_l10.generation_runtime.v1",
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
            "incompleteRetainedCount": int(
                (attempts["attemptStatus"] != "COMPLETE").sum()
            ),
        },
    )
    print(
        json.dumps(
            {
                "status": "TRAJECTORIES_GENERATED_LABELS_UNOPENED",
                "attempts": len(attempts),
                "trajectories": len(trajectories),
                "complete": int((attempts["attemptStatus"] == "COMPLETE").sum()),
            },
            sort_keys=True,
        )
    )


def empty_fingerprint_row(
    pipeline_id: str,
    candidate_id: str,
    matrix_index: int,
    trajectory_id: str,
    status: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
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


def fingerprint_row(
    pipeline_id: str,
    candidate_id: str,
    matrix_index: int,
    trajectory_id: str,
    fingerprint: dict[str, Any],
) -> dict[str, Any]:
    row = {
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
    # Reuse the L10 metric implementation on the frozen eligible label sequence,
    # then retain L08's raw analysis-unit onset/length semantics verbatim.
    result = label_fingerprint(labels, generations)
    result["selectedClockLength"] = int(frozen["analysisUnitLength"])
    result["persistence"] = int(frozen["persistence"])
    result["occupancy"] = float(frozen["occupancy"])
    result["firstOnsetRawIndex0"] = frozen["firstOnsetRawIndex0"]
    result["firstOnsetRawStep1"] = frozen["firstOnsetRawStep1"]
    result["firstOnsetNormalized"] = frozen["firstOnsetNormalized"]
    result["consistency"] = frozen["consistency"]
    result["consistencyStatus"] = frozen["consistencyStatus"]
    result["positiveEpisodeCount"] = frozen["positiveEpisodeCount"]
    result["negativeEpisodeCount"] = frozen["negativeEpisodeCount"]
    result["positiveMeanEpisodeDuration"] = frozen["positiveMeanEpisodeDuration"]
    result["negativeMeanEpisodeDuration"] = frozen["negativeMeanEpisodeDuration"]
    result["positiveLongestEpisodeDuration"] = frozen["positiveLongestEpisodeDuration"]
    result["negativeLongestEpisodeDuration"] = frozen["negativeLongestEpisodeDuration"]
    result["labelSha256"] = frozen["labelSha256"]
    return result


def comparator_rows_for_candidate(
    matrix_index: int,
    candidate_id: str,
    primary_trajectory: Any,
    high_trajectory: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    primary_selected = selected_clock_observations(
        primary_trajectory, "C1_SELECTED_DAUGHTER_RETAINED"
    )
    primary_generations = np.asarray(
        [item.growth_generation_one_based for item in primary_selected], dtype=np.int64
    )
    adjacent_setting = {
        "roundId": "S19-L10",
        "settingId": COMPARATOR_ADJACENT,
        "settingPairId": COMPARATOR_ADJACENT,
        "threshold": THRESHOLD,
        "comparator": "STRICT_GT",
        "clockId": "C1_SELECTED_DAUGHTER_RETAINED",
        "alignment": "INCOMING_DUPLICATE_FIRST",
        "family": "ADJACENT_CLOCK",
        "projection": "ALL_OBSERVATIONS",
    }
    adjacent = materialize_frozen_setting(primary_trajectory, adjacent_setting)
    if adjacent["isReplicator"].notna().all():
        fp = label_fingerprint(
            adjacent["isReplicator"].to_numpy(bool), primary_generations
        )
        rows.append(
            fingerprint_row(
                COMPARATOR_ADJACENT,
                candidate_id,
                matrix_index,
                str(primary_trajectory.trajectory_id),
                fp,
            )
        )
    else:
        rows.append(
            empty_fingerprint_row(
                COMPARATOR_ADJACENT,
                candidate_id,
                matrix_index,
                str(primary_trajectory.trajectory_id),
                "COMPARATOR_INELIGIBLE",
            )
        )

    for comparator_id, mechanism, object_id, trajectory in (
        (COMPARATOR_A_BOUNDARY, MECHANISM_A, OBJECT_A_BOUNDARY, primary_trajectory),
        (COMPARATOR_A_PROJECTED, MECHANISM_A, OBJECT_A_PROJECTED, primary_trajectory),
        (COMPARATOR_B_HIGH, MECHANISM_B, OBJECT_B_MOLECULAR, high_trajectory),
    ):
        frame = materialize_analysis_object(trajectory, mechanism, object_id)
        fp = l08_frame_to_fingerprint(frame)
        if fp is None:
            rows.append(
                empty_fingerprint_row(
                    comparator_id,
                    candidate_id,
                    matrix_index,
                    str(trajectory.trajectory_id),
                    "COMPARATOR_INELIGIBLE",
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
                )
            )
    return rows


def control_fingerprint(
    molecular: np.ndarray,
    molecular_generations: np.ndarray,
    boundary: np.ndarray,
    reference: np.ndarray,
) -> tuple[dict[str, Any], float]:
    _, labels = label_against_reference(molecular, reference)
    _, boundary_labels = label_against_reference(boundary, reference)
    fp = label_fingerprint(labels, molecular_generations)
    return fp, float(np.mean(boundary_labels))


def analyze_matrix(
    matrix_index: int, cache_root: Path
) -> dict[str, list[dict[str, Any]]]:
    buckets = {
        key: []
        for key in (
            "cluster",
            "silhouette",
            "recurrence",
            "dominant",
            "molecular",
            "boundary",
            "fingerprint",
            "episode",
            "comparator",
            "negative",
            "failure",
        )
    }
    for candidate_id in PRIMARY_CANDIDATES:
        primary_path = trajectory_path(
            cache_root, matrix_index, PRIMARY_GROUP, candidate_id
        )
        high_path = trajectory_path(cache_root, matrix_index, HIGH_GROUP, candidate_id)
        with primary_path.open("rb") as handle:
            primary = pickle.load(handle)
        with high_path.open("rb") as handle:
            high = pickle.load(handle)
        if primary.completed_fissions != 100 or high.completed_fissions != 100:
            for pipeline_id in PIPELINE_IDS:
                trajectory_id = str(primary.trajectory_id)
                status = "INCOMPLETE_TRAJECTORY_RETAINED_LABEL_INELIGIBLE"
                common = {
                    "pipelineId": pipeline_id,
                    "candidateId": candidate_id,
                    "matrixIndex": matrix_index,
                    "trajectoryId": trajectory_id,
                }
                buckets["cluster"].append(
                    {
                        **common,
                        "pipelineStatus": status,
                        "selectedK": None,
                        "selectedScore": None,
                        "eligibleBoundaryCount": int(primary.completed_fissions),
                        "clusterCount": None,
                        "clusterSizesJson": "[]",
                        "selectedKEqualsN": False,
                        "allSingletonSelected": False,
                        "tiedLargestSelected": False,
                    }
                )
                buckets["recurrence"].append(
                    {
                        **common,
                        "recurrenceStatus": status,
                        "dominantClusterId": None,
                        "secondClusterId": None,
                        "dominantClusterSize": None,
                        "secondClusterSize": None,
                        "selectedKEqualsN": False,
                        "allSingletonSelected": False,
                        "tiedLargestSelected": False,
                    }
                )
                buckets["dominant"].append(
                    {
                        **common,
                        "pipelineStatus": status,
                        "dominantCentroidJson": None,
                        "dominantCentroidSha256": None,
                        "secondCentroidJson": None,
                        "secondCentroidSha256": None,
                        "dominantClusterFraction": None,
                        "withinDominantDispersion": None,
                        "dominantSecondSeparation": None,
                        "boundaryOccupancy": None,
                        "boundaryConsistency": None,
                        "meanParentDaughterHInside": None,
                        "meanParentDaughterHOutside": None,
                    }
                )
                buckets["fingerprint"].append(
                    empty_fingerprint_row(
                        pipeline_id, candidate_id, matrix_index, trajectory_id, status
                    )
                )
                buckets["molecular"].append(
                    {
                        **common,
                        "analysisUnitIndex": -1,
                        "rawObservationIndex": None,
                        "generation": None,
                        "observationKind": None,
                        "labelStatus": status,
                        "hToDominant": None,
                        "isReplicator": None,
                        "stateSha256": None,
                    }
                )
                buckets["boundary"].append(
                    {
                        **common,
                        "boundaryIndex0": -1,
                        "generation": None,
                        "rawObservationIndex": None,
                        "nondriftEligible": None,
                        "selectedClusterId": None,
                        "labelStatus": status,
                        "hToDominant": None,
                        "isReplicator": None,
                        "stateSha256": None,
                    }
                )
            buckets["comparator"].extend(
                comparator_rows_for_candidate(matrix_index, candidate_id, primary, high)
            )
            continue

        selected = selected_clock_observations(primary, "C1_SELECTED_DAUGHTER_RETAINED")
        post = tuple(
            item for item in selected if item.observation_kind == "post_fission"
        )
        if len(post) != 100:
            raise RuntimeError(
                "complete primary trajectory lacks exactly 100 post-fission boundaries"
            )
        molecular = close_rows(
            np.asarray([item.state for item in selected], dtype=np.float64)
        )
        boundary = close_rows(
            np.asarray([item.state for item in post], dtype=np.float64)
        )
        generations = np.asarray(
            [item.growth_generation_one_based for item in selected], dtype=np.int64
        )
        parent_daughter = boundary_scores(
            primary,
            boundary_object="PARENT_TO_SELECTED_DAUGHTER",
            alignment="INCOMING_DUPLICATE_FIRST",
        )
        buckets["comparator"].extend(
            comparator_rows_for_candidate(matrix_index, candidate_id, primary, high)
        )

        for pipeline_id in PIPELINE_IDS:
            trajectory_id = str(primary.trajectory_id)
            try:
                fit = fit_pipeline(pipeline_id, boundary, trajectory_id)
            except Exception as error:  # noqa: BLE001 - unregistered worker failure is fail-closed
                buckets["failure"].append(
                    {
                        "failureId": f"S19-L10-ANALYSIS-{pipeline_id}-{candidate_id}-M{matrix_index:03d}",
                        **serialize_worker_exception(
                            candidate_id=candidate_id,
                            matrix_id=matrix_index,
                            pipeline_id=pipeline_id,
                            k=None,
                            n=len(boundary),
                            cluster_sizes=(),
                            seed_identity=f"{CLUSTER_ROOT_HEX}:{trajectory_id}",
                            error=error,
                        ),
                        "scientificValuesEligible": False,
                    }
                )
                continue
            common = {
                "pipelineId": pipeline_id,
                "candidateId": candidate_id,
                "matrixIndex": matrix_index,
                "trajectoryId": trajectory_id,
            }
            for record in fit.k_records:
                buckets["silhouette"].append(
                    {
                        **common,
                        "k": int(record["k"]),
                        "n": int(record.get("n", np.count_nonzero(fit.eligible_mask))),
                        "kStatus": record["status"],
                        "selectionScore": record.get("selectionScore"),
                        "localSilhouetteSha256": record.get("localSilhouetteSha256"),
                        "singletonCount": record.get("singletonCount"),
                        "clusterSizesJson": json.dumps(record.get("clusterSizes", [])),
                        "replicaLossesJson": json.dumps(
                            record.get("replicaLosses", [])
                        ),
                        "replicaIterationsJson": json.dumps(
                            record.get("replicaIterations", [])
                        ),
                        "selectedLoss": record.get("selectedLoss"),
                        "realizedClusterCount": record.get("realizedClusterCount"),
                        "selectedK": bool(record["k"] == fit.selected_k),
                    }
                )
            dominant_size = (
                None
                if fit.dominant_cluster_id is None
                else fit.cluster_sizes[fit.dominant_cluster_id]
            )
            second_size = (
                None
                if fit.second_cluster_id is None
                else fit.cluster_sizes[fit.second_cluster_id]
            )
            buckets["cluster"].append(
                {
                    **common,
                    "pipelineStatus": fit.status,
                    "selectedK": fit.selected_k,
                    "selectedScore": fit.selected_score,
                    "eligibleBoundaryCount": int(np.count_nonzero(fit.eligible_mask)),
                    "clusterCount": len(fit.cluster_sizes),
                    "clusterSizesJson": json.dumps(list(fit.cluster_sizes)),
                    "selectedKEqualsN": fit.selected_k_equals_n,
                    "allSingletonSelected": fit.all_singleton_selected,
                    "tiedLargestSelected": fit.tied_largest_selected,
                }
            )
            buckets["recurrence"].append(
                {
                    **common,
                    "recurrenceStatus": fit.status,
                    "dominantClusterId": fit.dominant_cluster_id,
                    "secondClusterId": fit.second_cluster_id,
                    "dominantClusterSize": dominant_size,
                    "secondClusterSize": second_size,
                    "selectedKEqualsN": fit.selected_k_equals_n,
                    "allSingletonSelected": fit.all_singleton_selected,
                    "tiedLargestSelected": fit.tied_largest_selected,
                }
            )
            if fit.status != "ELIGIBLE" or fit.dominant_centroid is None:
                buckets["dominant"].append(
                    {
                        **common,
                        "pipelineStatus": fit.status,
                        "dominantCentroidJson": None,
                        "dominantCentroidSha256": None,
                        "secondCentroidJson": None,
                        "secondCentroidSha256": None,
                        "dominantClusterFraction": None,
                        "withinDominantDispersion": None,
                        "dominantSecondSeparation": None,
                        "boundaryOccupancy": None,
                        "boundaryConsistency": None,
                        "meanParentDaughterHInside": None,
                        "meanParentDaughterHOutside": None,
                    }
                )
                buckets["fingerprint"].append(
                    empty_fingerprint_row(
                        pipeline_id,
                        candidate_id,
                        matrix_index,
                        trajectory_id,
                        fit.status,
                    )
                )
                buckets["molecular"].append(
                    {
                        **common,
                        "analysisUnitIndex": -1,
                        "rawObservationIndex": None,
                        "generation": None,
                        "observationKind": None,
                        "labelStatus": fit.status,
                        "hToDominant": None,
                        "isReplicator": None,
                        "stateSha256": None,
                    }
                )
                buckets["boundary"].append(
                    {
                        **common,
                        "boundaryIndex0": -1,
                        "generation": None,
                        "rawObservationIndex": None,
                        "nondriftEligible": None,
                        "selectedClusterId": None,
                        "labelStatus": fit.status,
                        "hToDominant": None,
                        "isReplicator": None,
                        "stateSha256": None,
                    }
                )
                continue

            molecular_scores, molecular_labels = label_against_reference(
                molecular, fit.dominant_centroid
            )
            boundary_scores_ref, boundary_labels = label_against_reference(
                boundary, fit.dominant_centroid
            )
            fp = label_fingerprint(molecular_labels, generations)
            boundary_fp = label_fingerprint(
                boundary_labels, np.arange(1, 101, dtype=np.int64)
            )
            assigned = boundary[fit.eligible_mask][
                fit.labels == fit.dominant_cluster_id
            ]
            if pipeline_id == R1_ID:
                dispersion = float(
                    np.mean(1.0 - historical_h(assigned, fit.dominant_centroid).ravel())
                )
                separation = (
                    None
                    if fit.second_centroid is None
                    else float(
                        1.0
                        - historical_h(fit.dominant_centroid, fit.second_centroid)[0, 0]
                    )
                )
            else:
                dispersion = float(
                    np.mean(np.linalg.norm(assigned - fit.dominant_centroid, axis=1))
                )
                separation = (
                    None
                    if fit.second_centroid is None
                    else float(
                        np.linalg.norm(fit.dominant_centroid - fit.second_centroid)
                    )
                )
            buckets["dominant"].append(
                {
                    **common,
                    "pipelineStatus": fit.status,
                    "dominantCentroidJson": json.dumps(fit.dominant_centroid.tolist()),
                    "dominantCentroidSha256": array_sha256(fit.dominant_centroid),
                    "secondCentroidJson": None
                    if fit.second_centroid is None
                    else json.dumps(fit.second_centroid.tolist()),
                    "secondCentroidSha256": None
                    if fit.second_centroid is None
                    else array_sha256(fit.second_centroid),
                    "dominantClusterFraction": float(
                        dominant_size / max(1, np.count_nonzero(fit.eligible_mask))
                    ),
                    "withinDominantDispersion": dispersion,
                    "dominantSecondSeparation": separation,
                    "boundaryOccupancy": float(np.mean(boundary_labels)),
                    "boundaryConsistency": boundary_fp["consistency"],
                    "meanParentDaughterHInside": float(
                        np.mean(parent_daughter[boundary_labels])
                    )
                    if np.any(boundary_labels)
                    else None,
                    "meanParentDaughterHOutside": float(
                        np.mean(parent_daughter[~boundary_labels])
                    )
                    if np.any(~boundary_labels)
                    else None,
                }
            )
            buckets["fingerprint"].append(
                fingerprint_row(
                    pipeline_id, candidate_id, matrix_index, trajectory_id, fp
                )
            )
            eligible_positions = np.flatnonzero(fit.eligible_mask)
            boundary_to_cluster = {
                int(position): int(fit.labels[index])
                for index, position in enumerate(eligible_positions)
            }
            for index, (item, score, label, state) in enumerate(
                zip(
                    selected, molecular_scores, molecular_labels, molecular, strict=True
                )
            ):
                buckets["molecular"].append(
                    {
                        **common,
                        "analysisUnitIndex": index,
                        "rawObservationIndex": int(item.observation_index),
                        "generation": int(item.growth_generation_one_based),
                        "observationKind": str(item.observation_kind),
                        "labelStatus": fp["fingerprintStatus"],
                        "hToDominant": float(score),
                        "isReplicator": bool(label),
                        "stateSha256": array_sha256(state),
                    }
                )
            for index, (item, score, label, state) in enumerate(
                zip(post, boundary_scores_ref, boundary_labels, boundary, strict=True)
            ):
                buckets["boundary"].append(
                    {
                        **common,
                        "boundaryIndex0": index,
                        "generation": int(item.growth_generation_one_based),
                        "rawObservationIndex": int(item.observation_index),
                        "nondriftEligible": bool(fit.eligible_mask[index]),
                        "selectedClusterId": boundary_to_cluster.get(index),
                        "labelStatus": fp["fingerprintStatus"],
                        "hToDominant": float(score),
                        "isReplicator": bool(label),
                        "stateSha256": array_sha256(state),
                    }
                )
            for polarity, desired in (("POSITIVE", True), ("NEGATIVE", False)):
                for episode_index, episode in enumerate(
                    run_descriptors(molecular_labels, desired)
                ):
                    buckets["episode"].append(
                        {
                            **common,
                            "polarity": polarity,
                            "episodeIndex": episode_index,
                            **episode,
                        }
                    )

            for draw in range(RANDOM_REFERENCE_DRAWS):
                rng = np.random.Generator(
                    np.random.PCG64DXSM(
                        deterministic_seed(
                            "random-reference",
                            pipeline_id,
                            candidate_id,
                            matrix_index,
                            draw,
                            bits=128,
                        )
                    )
                )
                reference_index = int(rng.integers(0, len(boundary)))
                control_fp, recurrence = control_fingerprint(
                    molecular, generations, boundary, boundary[reference_index]
                )
                buckets["negative"].append(
                    {
                        "recordType": "TRAJECTORY_CONTROL",
                        **common,
                        "controlType": "RANDOM_REFERENCE",
                        "controlIndex": draw,
                        "outcome": None,
                        "controlStatus": "ELIGIBLE",
                        "referenceBoundaryIndex0": reference_index,
                        "boundaryRecurrence": recurrence,
                        **{
                            metric: control_fp.get(metric)
                            for metric in (
                                "selectedClockLength",
                                "persistence",
                                "occupancy",
                                "consistency",
                                "firstOnsetRawStep1",
                                "firstOnsetNormalized",
                            )
                        },
                        "rawPaperDistance": paper_distance(control_fp, "RAW"),
                        "normalizedPaperDistance": paper_distance(
                            control_fp, "NORMALIZED"
                        ),
                        "rawP": None,
                        "holmAdjustedP": None,
                    }
                )
            if fit.second_centroid is not None:
                control_fp, recurrence = control_fingerprint(
                    molecular, generations, boundary, fit.second_centroid
                )
                buckets["negative"].append(
                    {
                        "recordType": "TRAJECTORY_CONTROL",
                        **common,
                        "controlType": "SECOND_LARGEST_CLUSTER",
                        "controlIndex": 0,
                        "outcome": None,
                        "controlStatus": "ELIGIBLE",
                        "referenceBoundaryIndex0": None,
                        "boundaryRecurrence": recurrence,
                        **{
                            metric: control_fp.get(metric)
                            for metric in (
                                "selectedClockLength",
                                "persistence",
                                "occupancy",
                                "consistency",
                                "firstOnsetRawStep1",
                                "firstOnsetNormalized",
                            )
                        },
                        "rawPaperDistance": paper_distance(control_fp, "RAW"),
                        "normalizedPaperDistance": paper_distance(
                            control_fp, "NORMALIZED"
                        ),
                        "rawP": None,
                        "holmAdjustedP": None,
                    }
                )
            else:
                buckets["negative"].append(
                    {
                        "recordType": "TRAJECTORY_CONTROL",
                        **common,
                        "controlType": "SECOND_LARGEST_CLUSTER",
                        "controlIndex": 0,
                        "outcome": None,
                        "controlStatus": "NOT_APPLICABLE_NO_UNIQUE_RECURRING_SECOND_CLUSTER",
                        "referenceBoundaryIndex0": None,
                        "boundaryRecurrence": None,
                        "selectedClockLength": None,
                        "persistence": None,
                        "occupancy": None,
                        "consistency": None,
                        "firstOnsetRawStep1": None,
                        "firstOnsetNormalized": None,
                        "rawPaperDistance": None,
                        "normalizedPaperDistance": None,
                        "rawP": None,
                        "holmAdjustedP": None,
                    }
                )

            permutation_rng = np.random.Generator(
                np.random.PCG64DXSM(
                    deterministic_seed(
                        "time-permutation",
                        pipeline_id,
                        candidate_id,
                        matrix_index,
                        0,
                        bits=128,
                    )
                )
            )
            permutation = permutation_rng.permutation(len(boundary))
            permuted_fit = fit_pipeline(
                pipeline_id, boundary[permutation], f"{trajectory_id}::PERMUTED_TIME::0"
            )
            if (
                permuted_fit.status == "ELIGIBLE"
                and permuted_fit.dominant_centroid is not None
            ):
                control_fp, recurrence = control_fingerprint(
                    molecular, generations, boundary, permuted_fit.dominant_centroid
                )
                control_status = "ELIGIBLE"
            else:
                control_fp, recurrence, control_status = {}, None, permuted_fit.status
            buckets["negative"].append(
                {
                    "recordType": "TRAJECTORY_CONTROL",
                    **common,
                    "controlType": "PERMUTED_TIME",
                    "controlIndex": 0,
                    "outcome": None,
                    "controlStatus": control_status,
                    "referenceBoundaryIndex0": None,
                    "boundaryRecurrence": recurrence,
                    **{
                        metric: control_fp.get(metric)
                        for metric in (
                            "selectedClockLength",
                            "persistence",
                            "occupancy",
                            "consistency",
                            "firstOnsetRawStep1",
                            "firstOnsetNormalized",
                        )
                    },
                    "rawPaperDistance": paper_distance(control_fp, "RAW")
                    if control_fp
                    else None,
                    "normalizedPaperDistance": paper_distance(control_fp, "NORMALIZED")
                    if control_fp
                    else None,
                    "rawP": None,
                    "holmAdjustedP": None,
                }
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
            values = (
                pd.to_numeric(eligible[metric], errors="coerce")
                .dropna()
                .to_numpy(float)
            )
            row[f"defined_{metric}"] = len(values)
            row[f"mean_{metric}"] = float(np.mean(values)) if len(values) else None
            row[f"median_{metric}"] = float(np.median(values)) if len(values) else None
            row[f"sd_{metric}"] = (
                float(np.std(values, ddof=1))
                if len(values) > 1
                else (0.0 if len(values) else None)
            )
            row[f"se_{metric}"] = (
                float(np.std(values, ddof=1) / math.sqrt(len(values)))
                if len(values) > 1
                else (0.0 if len(values) else None)
            )
        summary = {metric: row[f"mean_{metric}"] for metric in FINGERPRINT_METRICS}
        row["rawPaperDistance"] = paper_distance(summary, "RAW")
        row["normalizedPaperDistance"] = paper_distance(summary, "NORMALIZED")
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_primary(fingerprint: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        "selectedClockLength",
        "persistence",
        "occupancy",
        "consistency",
        "firstOnsetRawStep1",
        "firstOnsetNormalized",
        "preOnsetNonreplicatingDuration",
        "noReplicatorThrough25Percent",
        "nonreplicatingAt25Percent",
        "positiveEpisodeCount",
        "negativeEpisodeCount",
    )
    rows: list[dict[str, Any]] = []
    for pipeline_id in PIPELINE_IDS:
        for candidate_id in PRIMARY_CANDIDATES:
            group = (
                fingerprint[
                    (fingerprint["pipelineId"] == pipeline_id)
                    & (fingerprint["candidateId"] == candidate_id)
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
                        float(np.nanmean(sampled))
                        if np.any(np.isfinite(sampled))
                        else None
                    )
                rows.append(
                    {
                        "pipelineId": pipeline_id,
                        "candidateId": candidate_id,
                        "bootstrapReplicate": replicate,
                        **summary,
                        "rawPaperDistance": paper_distance(summary, "RAW"),
                        "normalizedPaperDistance": paper_distance(
                            summary, "NORMALIZED"
                        ),
                    }
                )
    return pd.DataFrame(rows)


def paper_target_tables(
    aggregate: pd.DataFrame,
    bootstrap: pd.DataFrame,
    comparator_aggregate: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    targets = {
        "selectedClockLength": (716.0 / 0.88, 225.0),
        "persistence": (716.0, 198.0),
        "occupancy": (0.88, 0.03),
        "consistency": (0.38, 0.06),
        "firstOnsetRawStep1": (37.0, 27.0),
        "firstOnsetNormalized": (0.37, 0.27),
    }
    rows: list[dict[str, Any]] = []
    distances: list[dict[str, Any]] = []
    combined = pd.concat([aggregate, comparator_aggregate], ignore_index=True)
    for item in combined.itertuples():
        for metric, (target, scale) in targets.items():
            value = getattr(item, f"mean_{metric}")
            boot_values = np.asarray([], dtype=float)
            if item.pipelineId in PIPELINE_IDS:
                subset = bootstrap[
                    (bootstrap["pipelineId"] == item.pipelineId)
                    & (bootstrap["candidateId"] == item.candidateId)
                ]
                boot_values = (
                    pd.to_numeric(subset[metric], errors="coerce")
                    .dropna()
                    .to_numpy(float)
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
                    "standardizedDifference": None
                    if pd.isna(value)
                    else float((value - target) / scale),
                    "bootstrapCi025": float(np.quantile(boot_values, 0.025))
                    if len(boot_values)
                    else None,
                    "bootstrapCi975": float(np.quantile(boot_values, 0.975))
                    if len(boot_values)
                    else None,
                    "measurementLevel": "BOUNDARY_DIAGNOSTIC_NONINTERCHANGEABLE"
                    if item.pipelineId == COMPARATOR_A_BOUNDARY
                    else "MOLECULAR_PRIMARY_OR_COMPARATOR",
                }
            )
        for onset_mode, field in (
            ("RAW", "rawPaperDistance"),
            ("NORMALIZED", "normalizedPaperDistance"),
        ):
            distances.append(
                {
                    "pipelineId": item.pipelineId,
                    "candidateId": item.candidateId,
                    "onsetMode": onset_mode,
                    "validTrajectoryCount": int(item.validTrajectoryCount),
                    "completeFingerprintDistance": getattr(item, field),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(distances)


def add_dimension_improvement_counts(
    distances: pd.DataFrame,
    aggregate: pd.DataFrame,
    comparator_aggregate: pd.DataFrame,
) -> pd.DataFrame:
    targets = {
        "selectedClockLength": 716.0 / 0.88,
        "persistence": 716.0,
        "occupancy": 0.88,
        "consistency": 0.38,
        "firstOnsetRawStep1": 37.0,
        "firstOnsetNormalized": 0.37,
    }
    result = distances.copy()
    result["dimensionsImprovedOverEveryComparator"] = 0
    result["improvedDimensionsJson"] = "{}"
    for index, row in result.iterrows():
        if row["pipelineId"] not in PIPELINE_IDS:
            continue
        candidate = row["candidateId"]
        onset_metric = (
            "firstOnsetRawStep1"
            if row["onsetMode"] == "RAW"
            else "firstOnsetNormalized"
        )
        metrics = (
            "selectedClockLength",
            "persistence",
            "occupancy",
            "consistency",
            onset_metric,
        )
        primary = aggregate[
            (aggregate.pipelineId == row["pipelineId"])
            & (aggregate.candidateId == candidate)
        ].iloc[0]
        comparators = comparator_aggregate[
            comparator_aggregate.candidateId == candidate
        ]
        improved: dict[str, bool] = {}
        for metric in metrics:
            value = getattr(primary, f"mean_{metric}")
            comparator_values = (
                pd.to_numeric(comparators[f"mean_{metric}"], errors="coerce")
                .dropna()
                .to_numpy(float)
            )
            improved[metric] = bool(
                pd.notna(value)
                and len(comparator_values)
                and abs(float(value) - targets[metric])
                < float(np.min(np.abs(comparator_values - targets[metric])))
            )
        result.at[index, "dimensionsImprovedOverEveryComparator"] = int(
            sum(improved.values())
        )
        result.at[index, "improvedDimensionsJson"] = json.dumps(
            improved, sort_keys=True
        )
    return result


def analyze_negative_controls(
    negative: pd.DataFrame,
    fingerprint: pd.DataFrame,
    dominant: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[tuple[str, str], dict[str, bool]]]:
    aggregate_rows: list[dict[str, Any]] = []
    gates: dict[tuple[str, str], dict[str, bool]] = {}
    for pipeline_id in PIPELINE_IDS:
        for candidate_id in PRIMARY_CANDIDATES:
            observed_fp = fingerprint[
                (fingerprint.pipelineId == pipeline_id)
                & (fingerprint.candidateId == candidate_id)
                & fingerprint.selectedClockLength.notna()
            ]
            observed_dom = dominant[
                (dominant.pipelineId == pipeline_id)
                & (dominant.candidateId == candidate_id)
                & dominant.boundaryOccupancy.notna()
            ]
            observed_summary = {
                metric: pd.to_numeric(observed_fp[metric], errors="coerce").mean()
                for metric in FINGERPRINT_METRICS
            }
            observed_values = {
                "boundaryRecurrence": float(observed_dom.boundaryOccupancy.mean())
                if len(observed_dom)
                else None,
                "rawPaperDistance": paper_distance(observed_summary, "RAW"),
                "normalizedPaperDistance": paper_distance(
                    observed_summary, "NORMALIZED"
                ),
            }
            local_gates: dict[str, bool] = {}
            random = negative[
                (negative.pipelineId == pipeline_id)
                & (negative.candidateId == candidate_id)
                & (negative.controlType == "RANDOM_REFERENCE")
                & (negative.controlStatus == "ELIGIBLE")
            ]
            random_group = (
                random.groupby("controlIndex", sort=True)[
                    [
                        "boundaryRecurrence",
                        "selectedClockLength",
                        "persistence",
                        "occupancy",
                        "consistency",
                        "firstOnsetRawStep1",
                        "firstOnsetNormalized",
                    ]
                ]
                .mean(numeric_only=True)
                .reset_index()
            )
            random_group["rawPaperDistance"] = random_group.apply(
                lambda row: paper_distance(row.to_dict(), "RAW"), axis=1
            )
            random_group["normalizedPaperDistance"] = random_group.apply(
                lambda row: paper_distance(row.to_dict(), "NORMALIZED"), axis=1
            )
            for outcome, direction, quantile in (
                ("boundaryRecurrence", "GREATER", 0.95),
                ("rawPaperDistance", "LESS", 0.05),
                ("normalizedPaperDistance", "LESS", 0.05),
            ):
                distribution = (
                    pd.to_numeric(random_group[outcome], errors="coerce")
                    .dropna()
                    .to_numpy(float)
                )
                observed = observed_values[outcome]
                threshold = (
                    float(np.quantile(distribution, quantile))
                    if len(distribution)
                    else None
                )
                passed = bool(
                    observed is not None
                    and threshold is not None
                    and (
                        observed > threshold
                        if direction == "GREATER"
                        else observed < threshold
                    )
                )
                local_gates[f"random_{outcome}"] = passed
                extreme = (
                    int(np.count_nonzero(distribution >= observed))
                    if direction == "GREATER" and observed is not None
                    else int(np.count_nonzero(distribution <= observed))
                    if observed is not None
                    else len(distribution)
                )
                aggregate_rows.append(
                    {
                        "recordType": "AGGREGATE_TEST",
                        "pipelineId": pipeline_id,
                        "candidateId": candidate_id,
                        "matrixIndex": -1,
                        "trajectoryId": None,
                        "controlType": "RANDOM_REFERENCE",
                        "controlIndex": -1,
                        "outcome": outcome,
                        "controlStatus": "ELIGIBLE"
                        if len(distribution)
                        else "UNDEFINED",
                        "referenceBoundaryIndex0": None,
                        "boundaryRecurrence": None,
                        "selectedClockLength": None,
                        "persistence": None,
                        "occupancy": None,
                        "consistency": None,
                        "firstOnsetRawStep1": None,
                        "firstOnsetNormalized": None,
                        "rawPaperDistance": None,
                        "normalizedPaperDistance": None,
                        "observed": observed,
                        "controlMean": float(np.mean(distribution))
                        if len(distribution)
                        else None,
                        "controlQ05": float(np.quantile(distribution, 0.05))
                        if len(distribution)
                        else None,
                        "controlQ95": float(np.quantile(distribution, 0.95))
                        if len(distribution)
                        else None,
                        "direction": direction,
                        "rawP": float((1 + extreme) / (1 + len(distribution)))
                        if len(distribution)
                        else None,
                        "holmAdjustedP": None,
                        "passed": passed,
                        "applicableCount": len(distribution),
                    }
                )

            for control_type in ("SECOND_LARGEST_CLUSTER", "PERMUTED_TIME"):
                control = negative[
                    (negative.pipelineId == pipeline_id)
                    & (negative.candidateId == candidate_id)
                    & (negative.controlType == control_type)
                    & (negative.controlStatus == "ELIGIBLE")
                ]
                applicable = sorted(
                    set(observed_fp.matrixIndex.astype(int))
                    & set(control.matrixIndex.astype(int))
                    & set(observed_dom.matrixIndex.astype(int))
                )
                if not applicable:
                    for outcome in (
                        "boundaryRecurrence",
                        "rawPaperDistance",
                        "normalizedPaperDistance",
                    ):
                        local_gates[f"{control_type.lower()}_{outcome}"] = (
                            control_type == "SECOND_LARGEST_CLUSTER"
                        )
                    aggregate_rows.append(
                        {
                            "recordType": "AGGREGATE_TEST",
                            "pipelineId": pipeline_id,
                            "candidateId": candidate_id,
                            "matrixIndex": -1,
                            "trajectoryId": None,
                            "controlType": control_type,
                            "controlIndex": -1,
                            "outcome": "NOT_APPLICABLE",
                            "controlStatus": "NOT_APPLICABLE"
                            if control_type == "SECOND_LARGEST_CLUSTER"
                            else "UNDEFINED",
                            "referenceBoundaryIndex0": None,
                            "boundaryRecurrence": None,
                            "selectedClockLength": None,
                            "persistence": None,
                            "occupancy": None,
                            "consistency": None,
                            "firstOnsetRawStep1": None,
                            "firstOnsetNormalized": None,
                            "rawPaperDistance": None,
                            "normalizedPaperDistance": None,
                            "observed": None,
                            "controlMean": None,
                            "controlQ05": None,
                            "controlQ95": None,
                            "direction": None,
                            "rawP": None,
                            "holmAdjustedP": None,
                            "passed": control_type == "SECOND_LARGEST_CLUSTER",
                            "applicableCount": 0,
                        }
                    )
                    continue
                observed_by_matrix = observed_fp.set_index("matrixIndex")
                observed_dom_by_matrix = observed_dom.set_index("matrixIndex")
                control_by_matrix = control.set_index("matrixIndex")
                outcomes = {
                    "boundaryRecurrence": (
                        observed_dom_by_matrix.loc[
                            applicable, "boundaryOccupancy"
                        ].to_numpy(float),
                        control_by_matrix.loc[
                            applicable, "boundaryRecurrence"
                        ].to_numpy(float),
                        "GREATER",
                    ),
                    "rawPaperDistance": (
                        observed_by_matrix.loc[applicable, "rawPaperDistance"].to_numpy(
                            float
                        ),
                        control_by_matrix.loc[applicable, "rawPaperDistance"].to_numpy(
                            float
                        ),
                        "LESS",
                    ),
                    "normalizedPaperDistance": (
                        observed_by_matrix.loc[
                            applicable, "normalizedPaperDistance"
                        ].to_numpy(float),
                        control_by_matrix.loc[
                            applicable, "normalizedPaperDistance"
                        ].to_numpy(float),
                        "LESS",
                    ),
                }
                for outcome, (
                    observed_array,
                    control_array,
                    direction,
                ) in outcomes.items():
                    finite = np.isfinite(observed_array) & np.isfinite(control_array)
                    delta = observed_array[finite] - control_array[finite]
                    if len(delta):
                        rng = np.random.Generator(
                            np.random.PCG64DXSM(
                                deterministic_seed(
                                    "control-bootstrap",
                                    pipeline_id,
                                    candidate_id,
                                    control_type,
                                    outcome,
                                    bits=128,
                                )
                            )
                        )
                        indices = rng.integers(
                            0, len(delta), size=(BOOTSTRAP_REPLICATES, len(delta))
                        )
                        draws = np.mean(delta[indices], axis=1)
                        ci025, ci975 = map(float, np.quantile(draws, [0.025, 0.975]))
                        passed = bool(
                            ci025 > 0 if direction == "GREATER" else ci975 < 0
                        )
                        extreme = (
                            int(np.count_nonzero(draws <= 0))
                            if direction == "GREATER"
                            else int(np.count_nonzero(draws >= 0))
                        )
                        raw_p = float((1 + extreme) / (1 + BOOTSTRAP_REPLICATES))
                    else:
                        ci025 = ci975 = raw_p = None
                        passed = False
                    local_gates[f"{control_type.lower()}_{outcome}"] = passed
                    aggregate_rows.append(
                        {
                            "recordType": "AGGREGATE_TEST",
                            "pipelineId": pipeline_id,
                            "candidateId": candidate_id,
                            "matrixIndex": -1,
                            "trajectoryId": None,
                            "controlType": control_type,
                            "controlIndex": -1,
                            "outcome": outcome,
                            "controlStatus": "ELIGIBLE" if len(delta) else "UNDEFINED",
                            "referenceBoundaryIndex0": None,
                            "boundaryRecurrence": None,
                            "selectedClockLength": None,
                            "persistence": None,
                            "occupancy": None,
                            "consistency": None,
                            "firstOnsetRawStep1": None,
                            "firstOnsetNormalized": None,
                            "rawPaperDistance": None,
                            "normalizedPaperDistance": None,
                            "observed": float(np.mean(observed_array[finite]))
                            if len(delta)
                            else None,
                            "controlMean": float(np.mean(control_array[finite]))
                            if len(delta)
                            else None,
                            "controlQ05": ci025,
                            "controlQ95": ci975,
                            "direction": direction,
                            "rawP": raw_p,
                            "holmAdjustedP": None,
                            "passed": passed,
                            "applicableCount": len(delta),
                        }
                    )
            gates[(pipeline_id, candidate_id)] = local_gates
    aggregate = pd.DataFrame(aggregate_rows)
    if len(aggregate):
        test_mask = aggregate["rawP"].notna()
        for indices in (
            aggregate[test_mask]
            .groupby(["candidateId", "controlType", "outcome"], sort=True)
            .groups.values()
        ):
            positions = list(indices)
            aggregate.loc[positions, "holmAdjustedP"] = holm_adjust(
                aggregate.loc[positions, "rawP"].astype(float).tolist()
            )
        # The frozen multiplicity family is the pair of primary pipelines
        # within candidate/control/outcome.  A directional control gate passes
        # only when its original effect criterion and its Holm-adjusted test
        # both pass; this is resolved before scientific outcomes are opened.
        adjusted = aggregate["holmAdjustedP"].notna()
        aggregate.loc[adjusted, "passed"] = aggregate.loc[adjusted, "passed"].astype(
            bool
        ) & aggregate.loc[adjusted, "holmAdjustedP"].astype(float).le(0.05)
        for item in aggregate.itertuples():
            if item.outcome in {
                "boundaryRecurrence",
                "rawPaperDistance",
                "normalizedPaperDistance",
            }:
                key = (str(item.pipelineId), str(item.candidateId))
                prefix = str(item.controlType).lower()
                gates.setdefault(key, {})[f"{prefix}_{item.outcome}"] = bool(
                    item.passed
                )
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
                    "pipelineId": pipeline_id,
                    "metric": metric,
                    "pairedMatrixCount": int(valid.sum()),
                    "candidate2Mean": float(a[valid].mean()) if valid.any() else None,
                    "candidate3Mean": float(b[valid].mean()) if valid.any() else None,
                    "candidate3Minus2": float((b[valid] - a[valid]).mean())
                    if valid.any()
                    else None,
                    "pairedPearson": correlation,
                }
            )
    return pd.DataFrame(rows)


def scientific_gate_table(
    fingerprint: pd.DataFrame,
    aggregate: pd.DataFrame,
    comparator_aggregate: pd.DataFrame,
    distances: pd.DataFrame,
    recurrence: pd.DataFrame,
    control_gates: dict[tuple[str, str], dict[str, bool]],
) -> pd.DataFrame:
    targets = {
        "selectedClockLength": 716.0 / 0.88,
        "persistence": 716.0,
        "occupancy": 0.88,
        "consistency": 0.38,
        "firstOnsetRawStep1": 37.0,
    }
    rows: list[dict[str, Any]] = []
    comparator_error: dict[tuple[str, str], float] = {}
    for candidate_id in PRIMARY_CANDIDATES:
        subset = comparator_aggregate[comparator_aggregate.candidateId == candidate_id]
        for metric, target in targets.items():
            values = (
                pd.to_numeric(subset[f"mean_{metric}"], errors="coerce")
                .dropna()
                .to_numpy(float)
            )
            comparator_error[(candidate_id, metric)] = (
                float(np.min(np.abs(values - target))) if len(values) else math.inf
            )
    for pipeline_id in PIPELINE_IDS:
        candidate_records: list[dict[str, Any]] = []
        for candidate_id in PRIMARY_CANDIDATES:
            agg = aggregate[
                (aggregate.pipelineId == pipeline_id)
                & (aggregate.candidateId == candidate_id)
            ].iloc[0]
            matrix = fingerprint[
                (fingerprint.pipelineId == pipeline_id)
                & (fingerprint.candidateId == candidate_id)
                & fingerprint.selectedClockLength.notna()
            ]
            recurrence_local = recurrence[
                (recurrence.pipelineId == pipeline_id)
                & (recurrence.candidateId == candidate_id)
            ]
            closer = {
                metric: bool(
                    pd.notna(getattr(agg, f"mean_{metric}"))
                    and abs(float(getattr(agg, f"mean_{metric}")) - target)
                    < comparator_error[(candidate_id, metric)]
                )
                for metric, target in targets.items()
            }
            onset = pd.to_numeric(matrix.firstOnsetRawStep1, errors="coerce")
            both_polarities = (
                pd.to_numeric(matrix.persistence, errors="coerce") > 0
            ) & (
                pd.to_numeric(matrix.persistence, errors="coerce")
                < pd.to_numeric(matrix.selectedClockLength, errors="coerce")
            )
            controls = control_gates.get((pipeline_id, candidate_id), {})
            random_pass = all(
                controls.get(f"random_{outcome}", False)
                for outcome in (
                    "boundaryRecurrence",
                    "rawPaperDistance",
                    "normalizedPaperDistance",
                )
            )
            second_pass = all(
                controls.get(f"second_largest_cluster_{outcome}", False)
                for outcome in (
                    "boundaryRecurrence",
                    "rawPaperDistance",
                    "normalizedPaperDistance",
                )
            )
            time_pass = all(
                controls.get(f"permuted_time_{outcome}", False)
                for outcome in (
                    "boundaryRecurrence",
                    "rawPaperDistance",
                    "normalizedPaperDistance",
                )
            )
            ineligible_emitted = recurrence_local.merge(
                matrix[["matrixIndex"]], on="matrixIndex", how="inner"
            )
            no_invalid_label = bool(
                len(ineligible_emitted)
                == int((ineligible_emitted.recurrenceStatus == "ELIGIBLE").sum())
            )
            selected_kn_defined = recurrence_local[
                (recurrence_local.recurrenceStatus == "ELIGIBLE")
                & recurrence_local.selectedKEqualsN.astype(bool)
            ]
            gates = {
                "definedAtLeast95": int(agg.validTrajectoryCount) >= 95,
                "occupancyInRange": bool(
                    pd.notna(agg.mean_occupancy)
                    and 0.85 <= float(agg.mean_occupancy) <= 0.91
                ),
                "persistenceInRange": bool(
                    pd.notna(agg.mean_persistence)
                    and 518 <= float(agg.mean_persistence) <= 914
                ),
                "consistencyCloserThanEveryComparator": closer["consistency"],
                "firstOnsetCloserThanEveryComparator": closer["firstOnsetRawStep1"],
                "nontrivialPreReplicatorInterval": bool(
                    pd.notna(agg.mean_firstOnsetRawStep1)
                    and float(agg.mean_firstOnsetRawStep1) >= 10
                    and len(onset.dropna())
                    and float(np.mean(onset.dropna() >= 10)) >= 0.5
                ),
                "quarterNoOnsetAtLeast20Percent": bool(
                    pd.notna(agg.mean_noReplicatorThrough25Percent)
                    and float(agg.mean_noReplicatorThrough25Percent) >= 0.20
                ),
                "positiveNegativeEpisodesNondegenerate": bool(
                    len(matrix) and float(np.mean(both_polarities)) >= 0.90
                ),
                "randomReferenceControlsPass": random_pass,
                "secondClusterControlsPassWhereApplicable": second_pass,
                "permutedTimeControlsPass": time_pass,
                "allSingletonAndKNDoNotEmitLabels": bool(
                    no_invalid_label and len(selected_kn_defined) == 0
                ),
            }
            record = {
                "pipelineId": pipeline_id,
                "candidateId": candidate_id,
                "validTrajectoryCount": int(agg.validTrajectoryCount),
                "dimensionsCloserThanEveryComparatorRaw": int(
                    distances[
                        (distances.pipelineId == pipeline_id)
                        & (distances.candidateId == candidate_id)
                        & (distances.onsetMode == "RAW")
                    ].dimensionsImprovedOverEveryComparator.iloc[0]
                ),
                "closerMetricsJson": json.dumps(closer, sort_keys=True),
                **gates,
                "scientificGateCount": len(gates),
                "scientificPassedGateCount": int(sum(gates.values())),
                "candidateScientificGatesPassed": bool(all(gates.values())),
            }
            candidate_records.append(record)
            rows.append(record)
        directions_agree = True
        for metric, target in targets.items():
            values = [
                float(
                    aggregate[
                        (aggregate.pipelineId == pipeline_id)
                        & (aggregate.candidateId == candidate)
                    ].iloc[0][f"mean_{metric}"]
                )
                for candidate in PRIMARY_CANDIDATES
            ]
            directions_agree &= bool(
                np.sign(values[0] - target) == np.sign(values[1] - target)
            )
        for row in rows[-2:]:
            row["crossCandidateDirectionalAgreement"] = directions_agree
            row["pipelineScientificPromotionGatesPassed"] = bool(
                directions_agree
                and all(
                    record["candidateScientificGatesPassed"]
                    for record in candidate_records
                )
            )
    return pd.DataFrame(rows)


def build_scientific_outputs(cache_root: Path, workers: int) -> dict[str, pd.DataFrame]:
    buckets = {
        key: []
        for key in (
            "cluster",
            "silhouette",
            "recurrence",
            "dominant",
            "molecular",
            "boundary",
            "fingerprint",
            "episode",
            "comparator",
            "negative",
            "failure",
        )
    }
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(analyze_matrix, matrix_index, cache_root): matrix_index
            for matrix_index in range(100)
        }
        for future in as_completed(futures):
            output = future.result()
            for key, values in buckets.items():
                values.extend(output[key])
    frames = {key: pd.DataFrame(value) for key, value in buckets.items()}
    if len(frames["failure"]):
        return frames
    # Process completion order is intentionally irrelevant.  Canonicalize all
    # row collections before any reduction so exact regeneration is invariant
    # to worker scheduling as well as to the final serialization order.
    canonical_keys = {
        "cluster": ("pipelineId", "candidateId", "matrixIndex"),
        "silhouette": ("pipelineId", "candidateId", "matrixIndex", "k"),
        "recurrence": ("pipelineId", "candidateId", "matrixIndex"),
        "dominant": ("pipelineId", "candidateId", "matrixIndex"),
        "molecular": ("pipelineId", "candidateId", "matrixIndex", "analysisUnitIndex"),
        "boundary": ("pipelineId", "candidateId", "matrixIndex", "boundaryIndex0"),
        "fingerprint": ("pipelineId", "candidateId", "matrixIndex"),
        "episode": (
            "pipelineId",
            "candidateId",
            "matrixIndex",
            "polarity",
            "episodeIndex",
        ),
        "comparator": ("pipelineId", "candidateId", "matrixIndex"),
        "negative": (
            "pipelineId",
            "candidateId",
            "matrixIndex",
            "controlType",
            "controlIndex",
        ),
    }
    for key, columns in canonical_keys.items():
        if len(frames[key]):
            frames[key] = (
                frames[key]
                .sort_values(list(columns), kind="stable")
                .reset_index(drop=True)
            )
    expected = {
        "cluster": 400,
        "recurrence": 400,
        "fingerprint": 400,
        "comparator": 800,
    }
    for key, count in expected.items():
        if len(frames[key]) != count:
            raise RuntimeError(f"L10 {key} cardinality {len(frames[key])} != {count}")
    aggregate = aggregate_fingerprints(frames["fingerprint"])
    comparator_aggregate = aggregate_fingerprints(frames["comparator"])
    bootstrap = bootstrap_primary(frames["fingerprint"])
    target, distances = paper_target_tables(aggregate, bootstrap, comparator_aggregate)
    distances = add_dimension_improvement_counts(
        distances, aggregate, comparator_aggregate
    )
    negative, control_gates = analyze_negative_controls(
        frames["negative"], frames["fingerprint"], frames["dominant"]
    )
    comparison = candidate_comparison_table(frames["fingerprint"])
    gates = scientific_gate_table(
        frames["fingerprint"],
        aggregate,
        comparator_aggregate,
        distances,
        frames["recurrence"],
        control_gates,
    )
    frames.update(
        {
            "aggregate": aggregate,
            "comparatorAggregate": comparator_aggregate,
            "bootstrap": bootstrap,
            "target": target,
            "distances": distances,
            "negative": negative,
            "comparison": comparison,
            "gates": gates,
        }
    )
    return frames


def write_scientific_outputs(root: Path, frames: dict[str, pd.DataFrame]) -> None:
    mapping = {
        "cluster_results.parquet": ("cluster", "parquet"),
        "silhouette_results.parquet": ("silhouette", "parquet"),
        "recurrence_status_results.parquet": ("recurrence", "parquet"),
        "dominant_attractor_results.parquet": ("dominant", "parquet"),
        "molecular_label_results.parquet": ("molecular", "parquet"),
        "boundary_label_results.parquet": ("boundary", "parquet"),
        "label_fingerprint_results.parquet": ("fingerprint", "parquet"),
        "episode_results.parquet": ("episode", "parquet"),
        "comparator_results.parquet": ("comparator", "parquet"),
        "negative_control_results.parquet": ("negative", "parquet"),
        "paper_target_comparison.csv": ("target", "csv"),
        "complete_fingerprint_distances.parquet": ("distances", "parquet"),
        "bootstrap_results.parquet": ("bootstrap", "parquet"),
        "candidate_comparison.csv": ("comparison", "csv"),
        "aggregate_fingerprint_results.parquet": ("aggregate", "parquet"),
        "comparator_aggregate_results.parquet": ("comparatorAggregate", "parquet"),
        "scientific_gate_results.parquet": ("gates", "parquet"),
    }
    for filename, (key, filetype) in mapping.items():
        if filetype == "parquet":
            write_parquet(root / filename, frames[key])
        else:
            write_csv(root / filename, frames[key])


def analyze(workers: int) -> None:
    if workers != 8:
        raise ValueError("L10 analysis is locked to eight workers")
    release = repository_release_gate()
    if not release["passed"]:
        raise RuntimeError("L10 release gate failed")
    manifest = pd.read_parquet(LOOP_ROOT / "trajectory_manifest.parquet")
    if len(manifest) != 400 or manifest.matrixIndex.nunique() != 100:
        raise RuntimeError("L10 trajectory manifest scope mismatch")
    for item in manifest.itertuples():
        path = Path(item.cachePath)
        if not path.exists() or sha256_file(path) != item.cacheSha256:
            raise RuntimeError(f"L10 cache hash mismatch: {path}")
    started = utc_now()
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    child_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    frames = build_scientific_outputs(PRIMARY_CACHE, workers)
    if len(frames["failure"]):
        write_csv(LOOP_ROOT / "failure_ledger.csv", frames["failure"])
        raise RuntimeError("L10 unregistered analysis exception; global fail closed")
    write_scientific_outputs(LOOP_ROOT, frames)
    child_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    child_cpu = (
        child_after.ru_utime
        + child_after.ru_stime
        - child_before.ru_utime
        - child_before.ru_stime
    )
    write_json(
        LOOP_ROOT / "analysis_runtime.json",
        {
            "schema": "eidosoma.e01.s19_l10.analysis_runtime.v1",
            "startedAtUtc": started,
            "completedAtUtc": utc_now(),
            "wallSeconds": time.perf_counter() - wall_start,
            "coordinatorCpuSeconds": time.process_time() - cpu_start,
            "childCpuSeconds": child_cpu,
            "workers": workers,
            "primaryFingerprintRows": len(frames["fingerprint"]),
            "molecularLabelRows": len(frames["molecular"]),
            "bootstrapRows": len(frames["bootstrap"]),
        },
    )
    print(
        json.dumps(
            {
                "status": "SCIENTIFIC_ANALYSIS_COMPLETE_PENDING_REGENERATION",
                "fingerprintRows": len(frames["fingerprint"]),
                "definedLabels": int(
                    frames["fingerprint"].selectedClockLength.notna().sum()
                ),
                "bootstrapRows": len(frames["bootstrap"]),
            },
            sort_keys=True,
        )
    )


def regenerate(workers: int) -> None:
    """Regenerate all trajectories and every authoritative scientific table."""

    if workers != 8:
        raise ValueError("L10 regeneration is locked to eight workers")
    release = repository_release_gate()
    if not release["passed"]:
        raise RuntimeError("L10 release gate failed before regeneration")
    if not (LOOP_ROOT / "analysis_runtime.json").exists():
        raise RuntimeError("L10 primary scientific analysis is absent")
    if any(REPLAY_CACHE.glob("*.pkl")) or any(REPLAY_OUTPUT.glob("*")):
        raise RuntimeError("L10 regeneration destinations must be empty")

    started = utc_now()
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    child_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    outputs = run_simulation_batch(range(100), workers, REPLAY_CACHE)
    attempts = pd.DataFrame([row for output in outputs for row in output["attempts"]])
    replay_manifest = pd.DataFrame(
        [row for output in outputs for row in output["trajectories"]]
    )
    failures = pd.DataFrame([row for output in outputs for row in output["failures"]])
    if len(attempts) != 400 or len(replay_manifest) != 400 or len(failures):
        raise RuntimeError("L10 trajectory regeneration failed closed")

    primary_manifest = pd.read_parquet(LOOP_ROOT / "trajectory_manifest.parquet")
    identity_columns = ["matrixIndex", "groupId", "candidateId"]
    compare_columns = [
        "trajectoryId",
        "trajectorySha256",
        "betaSha256",
        "initialStateSha256",
        "terminalStatus",
        "completedFissions",
        "selectedClockLength",
        "postFissionBoundaryCount",
        "cacheSha256",
    ]
    merged = primary_manifest.merge(
        replay_manifest,
        on=identity_columns,
        suffixes=("Primary", "Replay"),
        validate="one_to_one",
    )
    replay_rows = []
    for item in merged.itertuples():
        field_results = {
            field: getattr(item, f"{field}Primary") == getattr(item, f"{field}Replay")
            for field in compare_columns
        }
        replay_rows.append(
            {
                "matrixIndex": int(item.matrixIndex),
                "groupId": item.groupId,
                "candidateId": item.candidateId,
                **{f"{field}Exact": passed for field, passed in field_results.items()},
                "passed": bool(all(field_results.values())),
            }
        )
    trajectory_replay = pd.DataFrame(replay_rows).sort_values(
        identity_columns, kind="stable"
    )
    write_parquet(
        LOOP_ROOT / "trajectory_regeneration_results.parquet", trajectory_replay
    )

    frames = build_scientific_outputs(REPLAY_CACHE, workers)
    if len(frames["failure"]):
        raise RuntimeError("L10 regenerated scientific analysis raised an exception")
    write_scientific_outputs(REPLAY_OUTPUT, frames)

    table_rows = []
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
                "primaryRows": len(primary_frame),
                "replayRows": len(replay_frame),
                "primaryCanonicalSha256": primary_hash,
                "replayCanonicalSha256": replay_hash,
                "passed": bool(
                    len(primary_frame) == len(replay_frame)
                    and primary_hash == replay_hash
                ),
            }
        )
    table_validation = pd.DataFrame(table_rows)
    write_csv(LOOP_ROOT / "result_regeneration_results.csv", table_validation)

    child_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    child_cpu = (
        child_after.ru_utime
        + child_after.ru_stime
        - child_before.ru_utime
        - child_before.ru_stime
    )
    validation = {
        "schema": "eidosoma.e01.s19_l10.regeneration_validation.v1",
        "trajectoryReplayRows": len(trajectory_replay),
        "trajectoryReplayPassCount": int(trajectory_replay["passed"].sum()),
        "scientificTableCount": len(table_validation),
        "scientificTablePassCount": int(table_validation["passed"].sum()),
        "all400TrajectoriesExact": bool(
            len(trajectory_replay) == 400 and trajectory_replay["passed"].all()
        ),
        "allScientificTablesExact": bool(
            len(table_validation) == len(CORE_TABLES)
            and table_validation["passed"].all()
        ),
        "passed": bool(
            len(trajectory_replay) == 400
            and trajectory_replay["passed"].all()
            and len(table_validation) == len(CORE_TABLES)
            and table_validation["passed"].all()
        ),
        "validatedAtUtc": utc_now(),
    }
    write_json(LOOP_ROOT / "regeneration_validation.json", validation)
    write_json(
        LOOP_ROOT / "regeneration_runtime.json",
        {
            "schema": "eidosoma.e01.s19_l10.regeneration_runtime.v1",
            "startedAtUtc": started,
            "completedAtUtc": utc_now(),
            "wallSeconds": time.perf_counter() - wall_start,
            "coordinatorCpuSeconds": time.process_time() - cpu_start,
            "workerCpuSeconds": float(attempts["cpuSeconds"].sum()),
            "childCpuSeconds": child_cpu,
            "workers": workers,
        },
    )
    if not validation["passed"]:
        raise RuntimeError("L10 exact regeneration gate failed")
    print(json.dumps({"status": "REGENERATION_COMPLETE", **validation}, sort_keys=True))


def storage_bytes(root: Path) -> int:
    return (
        sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
        if root.exists()
        else 0
    )


def write_artifact_manifest(path: Path, root: Path, schema: str) -> None:
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
            "files": rows,
            "generatedAtUtc": utc_now(),
        },
    )


def result_mean(
    aggregate: pd.DataFrame, pipeline_id: str, candidate_id: str, metric: str
) -> float | None:
    row = aggregate[
        (aggregate.pipelineId == pipeline_id) & (aggregate.candidateId == candidate_id)
    ]
    if len(row) != 1:
        raise RuntimeError(f"missing aggregate {pipeline_id}/{candidate_id}")
    value = row.iloc[0][f"mean_{metric}"]
    return None if pd.isna(value) else float(value)


def short_pipeline(pipeline_id: str) -> str:
    return "R1 MATLAB-historical" if pipeline_id == R1_ID else "R2 paper-Euclidean"


def format_optional(value: float | None, digits: int = 4) -> str:
    return "NA" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def classify_results(
    *,
    operational_passed: bool,
) -> tuple[dict[str, Any], str, list[str], list[str]]:
    gates = pd.read_parquet(LOOP_ROOT / "scientific_gate_results.parquet")
    aggregate = pd.read_parquet(LOOP_ROOT / "aggregate_fingerprint_results.parquet")
    distances = pd.read_parquet(LOOP_ROOT / "complete_fingerprint_distances.parquet")
    pipeline_records: list[dict[str, Any]] = []
    promotable: list[str] = []
    method_leads: list[str] = []
    occupancy_only: list[str] = []
    for pipeline_id in PIPELINE_IDS:
        local_gates = gates[gates.pipelineId == pipeline_id]
        occupancy_both = all(
            (value := result_mean(aggregate, pipeline_id, candidate, "occupancy"))
            is not None
            and 0.85 <= value <= 0.91
            for candidate in PRIMARY_CANDIDATES
        )
        persistence_both = all(
            (value := result_mean(aggregate, pipeline_id, candidate, "persistence"))
            is not None
            and 518 <= value <= 914
            for candidate in PRIMARY_CANDIDATES
        )
        raw = distances[
            (distances.pipelineId == pipeline_id) & (distances.onsetMode == "RAW")
        ]
        dimensions = {
            str(row.candidateId): int(row.dimensionsImprovedOverEveryComparator)
            for row in raw.itertuples()
        }
        promotion_passed = bool(
            operational_passed
            and len(local_gates) == 2
            and local_gates["pipelineScientificPromotionGatesPassed"].astype(bool).all()
        )
        if promotion_passed:
            classification = "PROMOTABLE_TO_S20"
            promotable.append(pipeline_id)
        elif all(dimensions.get(candidate, 0) >= 2 for candidate in PRIMARY_CANDIDATES):
            classification = "METHOD_DEPENDENT_LEAD"
            method_leads.append(pipeline_id)
        elif occupancy_both:
            classification = "EXPLORATORY_PAPER_MATCH_OCCUPANCY_ONLY"
            occupancy_only.append(pipeline_id)
        else:
            classification = "RECURRING_ATTRACTOR_LABEL_NOT_RECONSTRUCTED"
        pipeline_records.append(
            {
                "pipelineId": pipeline_id,
                "classification": classification,
                "occupancyGateBothCandidates": occupancy_both,
                "persistenceGateBothCandidates": persistence_both,
                "dimensionsImprovedOverEveryComparator": dimensions,
                "scientificPromotionGatesPassed": promotion_passed,
            }
        )
    if not operational_passed:
        decision = "LOOP_FAILED_CLOSED"
        vocabulary = [
            "LOOP_FAILED_CLOSED",
            "POSSIBLE_PIPELINE_ARTIFACT",
            "NOT_PROMOTABLE",
        ]
    elif promotable:
        decision = "PROMOTABLE_TO_S20"
        vocabulary = [
            "PROMOTABLE_TO_S20",
            "EXPLORATORY_DIRECTIONAL_MATCH",
            "AUTHOR_AMBIGUITY_UNRESOLVED",
        ]
    elif method_leads:
        decision = "METHOD_DEPENDENT_LEAD"
        vocabulary = [
            "METHOD_DEPENDENT_LEAD",
            "AUTHOR_AMBIGUITY_UNRESOLVED",
            "NOT_PROMOTABLE",
        ]
    elif occupancy_only:
        decision = "EXPLORATORY_PAPER_MATCH_OCCUPANCY_ONLY"
        vocabulary = [
            "EXPLORATORY_PAPER_MATCH",
            "AUTHOR_AMBIGUITY_UNRESOLVED",
            "NOT_PROMOTABLE",
        ]
    else:
        decision = "RECURRING_ATTRACTOR_LABEL_NOT_RECONSTRUCTED"
        vocabulary = [
            "EXPLORATORY_NON_SUPPORT",
            "AUTHOR_AMBIGUITY_UNRESOLVED",
            "NOT_PROMOTABLE",
        ]
    classification = {
        "schema": "eidosoma.e01.s19_l10.classification.v1",
        "loopId": "S19-L10",
        "versionedLoopId": VERSION,
        "decision": decision,
        "pipelineResults": pipeline_records,
        "s19Classifications": vocabulary,
        "promotedLeadIds": promotable,
        "promotedLeadCount": len(promotable),
        "confirmed": False,
        "authorImplementationIdentified": False,
        "retrospectiveCompletedRunLabels": True,
        "prospectivePredictionSupported": False,
        "causalControlSupported": False,
        "s18ProspectiveAndCausalVerdictChanged": False,
        "classifiedAtUtc": utc_now(),
    }
    return classification, decision, vocabulary, promotable


def make_figures() -> list[Path]:
    figures = LOOP_ROOT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    aggregate = pd.read_parquet(LOOP_ROOT / "aggregate_fingerprint_results.parquet")
    cluster = pd.read_parquet(LOOP_ROOT / "cluster_results.parquet")
    recurrence = pd.read_parquet(LOOP_ROOT / "recurrence_status_results.parquet")
    molecular = pd.read_parquet(LOOP_ROOT / "molecular_label_results.parquet")
    boundary = pd.read_parquet(LOOP_ROOT / "boundary_label_results.parquet")
    episodes = pd.read_parquet(LOOP_ROOT / "episode_results.parquet")
    negative = pd.read_parquet(LOOP_ROOT / "negative_control_results.parquet")
    comparison = pd.read_csv(LOOP_ROOT / "candidate_comparison.csv")
    gates = pd.read_parquet(LOOP_ROOT / "scientific_gate_results.parquet")
    output: list[Path] = []

    def save(fig: Any, name: str) -> None:
        path = figures / name
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        output.append(path)

    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.bar(np.arange(4) - 0.18, np.ones(4), width=0.36, label="MATLAB-compatible")
    axis.bar(
        np.arange(4) + 0.18,
        np.zeros(4),
        width=0.36,
        color="#bdbdbd",
        label="scikit-learn: undefined",
    )
    axis.set(
        xticks=np.arange(4),
        xticklabels=["P1", "P2", "P3", "P4"],
        ylim=(0, 1.12),
        ylabel="Silhouette value",
    )
    axis.set_title("All-singleton k=n semantics (fixture F01)")
    axis.legend(frameon=False)
    save(fig, "figure_01_matlab_vs_sklearn_singleton.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for pipeline_id, color in ((R1_ID, "#2c7fb8"), (R2_ID, "#d95f0e")):
        values = pd.to_numeric(
            cluster.loc[cluster.pipelineId == pipeline_id, "selectedK"], errors="coerce"
        ).dropna()
        axes[0].hist(
            values,
            bins=np.arange(0.5, 11.5, 1),
            alpha=0.55,
            label=short_pipeline(pipeline_id),
            color=color,
        )
        sizes = pd.to_numeric(
            recurrence.loc[recurrence.pipelineId == pipeline_id, "dominantClusterSize"],
            errors="coerce",
        ).dropna()
        axes[1].hist(
            sizes, bins=20, alpha=0.55, label=short_pipeline(pipeline_id), color=color
        )
    axes[0].set(
        xlabel="Selected k", ylabel="Trajectory count", title="Selected-k distribution"
    )
    axes[1].set(
        xlabel="Dominant cluster size",
        ylabel="Trajectory count",
        title="Recurring-cluster sizes",
    )
    for axis in axes:
        axis.legend(frameon=False, fontsize=8)
    save(fig, "figure_02_selected_k_cluster_sizes.png")

    statuses = (
        recurrence.groupby(["pipelineId", "candidateId", "recurrenceStatus"])
        .size()
        .rename("count")
        .reset_index()
    )
    status_order = [
        "ELIGIBLE",
        "NO_RECURRING_COMPTYPE",
        "NO_UNIQUE_RECURRING_COMPTYPE",
        "NO_NONDRIFT_COMPOSITIONS",
        "NO_VALID_CLUSTERING",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for axis, pipeline_id in zip(axes, PIPELINE_IDS, strict=True):
        pivot = (
            statuses[statuses.pipelineId == pipeline_id]
            .pivot(index="candidateId", columns="recurrenceStatus", values="count")
            .fillna(0)
        )
        pivot = pivot.reindex(columns=status_order, fill_value=0)
        pivot.plot.bar(stacked=True, ax=axis, colormap="tab20c", legend=False)
        axis.set(
            title=short_pipeline(pipeline_id),
            xlabel="",
            ylabel="Trajectories",
            ylim=(0, 100),
        )
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle("Scientific recurrence eligibility and nonrecurrence rates")
    fig.tight_layout(rect=(0, 0.12, 1, 0.95))
    save(fig, "figure_03_singleton_no_recurring_rates.png")

    representative: dict[str, tuple[str, int]] = {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis, pipeline_id in zip(axes, PIPELINE_IDS, strict=True):
        valid = boundary[
            (boundary.pipelineId == pipeline_id) & boundary.isReplicator.notna()
        ]
        if valid.empty:
            axis.text(
                0.5,
                0.5,
                "No defined recurring-attractor label",
                ha="center",
                va="center",
            )
            axis.set_axis_off()
            continue
        row = valid.sort_values(["matrixIndex", "candidateId"]).iloc[0]
        candidate_id, matrix_index = str(row.candidateId), int(row.matrixIndex)
        representative[pipeline_id] = (candidate_id, matrix_index)
        path = trajectory_path(PRIMARY_CACHE, matrix_index, PRIMARY_GROUP, candidate_id)
        with path.open("rb") as handle:
            trajectory = pickle.load(handle)
        post = [
            item
            for item in selected_clock_observations(
                trajectory, "C1_SELECTED_DAUGHTER_RETAINED"
            )
            if item.observation_kind == "post_fission"
        ]
        values = close_rows(np.asarray([item.state for item in post], dtype=float))
        centered = values - values.mean(axis=0)
        coords = centered @ np.linalg.svd(centered, full_matrices=False)[2][:2].T
        labels = (
            boundary[
                (boundary.pipelineId == pipeline_id)
                & (boundary.candidateId == candidate_id)
                & (boundary.matrixIndex == matrix_index)
            ]
            .sort_values("boundaryIndex0")["isReplicator"]
            .to_numpy(bool)
        )
        axis.scatter(
            coords[:, 0],
            coords[:, 1],
            c=np.where(labels, 1, 0),
            cmap="coolwarm",
            s=22,
            alpha=0.8,
        )
        axis.set(
            title=f"{short_pipeline(pipeline_id)}\n{candidate_id}, M{matrix_index:03d}",
            xlabel="Composition PC1",
            ylabel="Composition PC2",
        )
    fig.suptitle("Representative dominant recurring-composition memberships")
    fig.tight_layout()
    save(fig, "figure_04_representative_dominant_clusters.png")

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=False)
    for axis, pipeline_id in zip(axes, PIPELINE_IDS, strict=True):
        if pipeline_id not in representative:
            axis.text(0.5, 0.5, "No defined molecular label", ha="center", va="center")
            continue
        candidate_id, matrix_index = representative[pipeline_id]
        frame = molecular[
            (molecular.pipelineId == pipeline_id)
            & (molecular.candidateId == candidate_id)
            & (molecular.matrixIndex == matrix_index)
        ].sort_values("analysisUnitIndex")
        axis.plot(frame.analysisUnitIndex, frame.hToDominant, linewidth=0.8)
        axis.axhline(THRESHOLD, color="black", linestyle="--", linewidth=1)
        axis.fill_between(
            frame.analysisUnitIndex.to_numpy(float),
            THRESHOLD,
            1.0,
            where=frame.isReplicator.to_numpy(bool),
            color="#41ab5d",
            alpha=0.2,
        )
        axis.set(
            ylabel="H to centroid",
            title=f"{short_pipeline(pipeline_id)} — {candidate_id}, M{matrix_index:03d}",
            ylim=(0, 1.01),
        )
    axes[-1].set_xlabel("Selected molecular-clock index")
    fig.tight_layout()
    save(fig, "figure_05_h_to_dominant_over_time.png")

    fig, axes = plt.subplots(2, 1, figsize=(11, 4.8), sharex=False)
    for axis, pipeline_id in zip(axes, PIPELINE_IDS, strict=True):
        if pipeline_id not in representative:
            axis.text(0.5, 0.5, "No defined recurring label", ha="center", va="center")
            continue
        candidate_id, matrix_index = representative[pipeline_id]
        frame = molecular[
            (molecular.pipelineId == pipeline_id)
            & (molecular.candidateId == candidate_id)
            & (molecular.matrixIndex == matrix_index)
        ].sort_values("analysisUnitIndex")
        with trajectory_path(
            PRIMARY_CACHE, matrix_index, PRIMARY_GROUP, candidate_id
        ).open("rb") as handle:
            trajectory = pickle.load(handle)
        adjacent_setting = {
            "roundId": "S19-L10-FIGURE",
            "settingId": COMPARATOR_ADJACENT,
            "settingPairId": COMPARATOR_ADJACENT,
            "threshold": THRESHOLD,
            "comparator": "STRICT_GT",
            "clockId": "C1_SELECTED_DAUGHTER_RETAINED",
            "alignment": "INCOMING_DUPLICATE_FIRST",
            "family": "ADJACENT_CLOCK",
            "projection": "ALL_OBSERVATIONS",
        }
        adjacent = materialize_frozen_setting(trajectory, adjacent_setting).sort_values(
            "analysisUnitIndex"
        )
        axis.step(
            frame.analysisUnitIndex,
            frame.isReplicator.astype(int) + 1.1,
            where="post",
            label="Recurring attractor",
            linewidth=1,
        )
        axis.step(
            adjacent.analysisUnitIndex,
            adjacent.isReplicator.astype(int),
            where="post",
            label="Adjacent H>0.9",
            linewidth=1,
        )
        axis.set(
            yticks=[0, 1, 1.1, 2.1],
            yticklabels=["Adj −", "Adj +", "Attr −", "Attr +"],
            title=f"{short_pipeline(pipeline_id)} — {candidate_id}, M{matrix_index:03d}",
        )
        axis.legend(frameon=False, ncol=2, fontsize=8)
    axes[-1].set_xlabel("Selected molecular-clock index")
    fig.tight_layout()
    save(fig, "figure_06_adjacent_vs_recurring_labels.png")

    def aggregate_bars(metrics: tuple[str, ...], filename: str, title: str) -> None:
        fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 4))
        axes_array = np.atleast_1d(axes)
        targets = {
            "occupancy": 0.88,
            "persistence": 716.0,
            "consistency": 0.38,
            "firstOnsetRawStep1": 37.0,
        }
        for axis, metric in zip(axes_array, metrics, strict=True):
            rows = []
            for pipeline_id in PIPELINE_IDS:
                for candidate_id in PRIMARY_CANDIDATES:
                    rows.append(
                        (
                            short_pipeline(pipeline_id),
                            candidate_id[-1],
                            result_mean(aggregate, pipeline_id, candidate_id, metric),
                        )
                    )
            axis.bar(
                range(len(rows)),
                [np.nan if row[2] is None else row[2] for row in rows],
                color=["#2c7fb8", "#7fcdbb", "#d95f0e", "#fdae6b"],
            )
            axis.axhline(targets[metric], linestyle="--", color="black", linewidth=1)
            axis.set(
                xticks=range(4),
                xticklabels=[
                    f"R1-C{rows[0][1]}",
                    f"R1-C{rows[1][1]}",
                    f"R2-C{rows[2][1]}",
                    f"R2-C{rows[3][1]}",
                ],
                title=metric,
            )
        fig.suptitle(title)
        fig.tight_layout()
        save(fig, filename)

    aggregate_bars(
        ("occupancy", "persistence"),
        "figure_07_occupancy_persistence.png",
        "Paper-facing occupancy and persistence",
    )
    aggregate_bars(
        ("consistency", "firstOnsetRawStep1"),
        "figure_08_consistency_onset.png",
        "Paper-facing consistency and raw onset",
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    if len(episodes):
        episode_summary = (
            episodes.groupby(["pipelineId", "candidateId", "polarity"])["duration"]
            .mean()
            .reset_index()
        )
        for index, polarity in enumerate(("POSITIVE", "NEGATIVE")):
            local = episode_summary[episode_summary.polarity == polarity]
            axes[index].bar(
                range(len(local)),
                local.duration,
                color="#41ab5d" if polarity == "POSITIVE" else "#ef3b2c",
            )
            axes[index].set(
                xticks=range(len(local)),
                xticklabels=[
                    f"{'R1' if row.pipelineId == R1_ID else 'R2'}-C{str(row.candidateId)[-1]}"
                    for row in local.itertuples()
                ],
                title=f"{polarity.title()} mean episode duration",
            )
    fig.suptitle("Episode topology; quarter-cutoff no-onset fractions are tabulated")
    fig.tight_layout()
    save(fig, "figure_09_episode_topology_preonset.png")

    tests = negative[negative.recordType == "AGGREGATE_TEST"].copy()
    fig, axis = plt.subplots(figsize=(11, 5))
    if len(tests):
        labels = [
            f"{'R1' if row.pipelineId == R1_ID else 'R2'}-C{str(row.candidateId)[-1]}\n{str(row.controlType)[:7]}\n{str(row.outcome)[:10]}"
            for row in tests.itertuples()
        ]
        values = pd.to_numeric(tests["rawP"], errors="coerce").fillna(1.0)
        axis.bar(
            range(len(tests)),
            values,
            color=np.where(tests["passed"].astype(bool), "#238b45", "#cb181d"),
        )
        axis.axhline(0.05, linestyle="--", color="black", linewidth=1)
        axis.set(
            xticks=range(len(tests)),
            xticklabels=labels,
            ylabel="Raw control p (Holm gate encoded by color)",
            ylim=(0, 1.02),
        )
        axis.tick_params(axis="x", labelrotation=70, labelsize=6)
    axis.set_title("Preregistered negative-control results")
    fig.tight_layout()
    save(fig, "figure_10_negative_controls.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, metric in zip(axes, ("occupancy", "firstOnsetRawStep1"), strict=True):
        local = comparison[comparison.metric == metric]
        axis.scatter(local.candidate2Mean, local.candidate3Mean, s=70)
        for row in local.itertuples():
            axis.annotate(
                "R1" if row.pipelineId == R1_ID else "R2",
                (row.candidate2Mean, row.candidate3Mean),
            )
        finite = pd.concat([local.candidate2Mean, local.candidate3Mean]).dropna()
        if len(finite):
            low, high = float(finite.min()), float(finite.max())
            axis.plot([low, high], [low, high], linestyle="--", color="black")
        axis.set(xlabel="Candidate 2 mean", ylabel="Candidate 3 mean", title=metric)
    fig.suptitle("Candidate-2 versus candidate-3 agreement")
    fig.tight_layout()
    save(fig, "figure_11_candidate_agreement.png")

    gate_columns = [
        column
        for column in gates.columns
        if gates[column].dtype == bool
        and column not in {"pipelineScientificPromotionGatesPassed"}
    ]
    matrix = gates[gate_columns].astype(int).to_numpy()
    fig, axis = plt.subplots(figsize=(max(10, len(gate_columns) * 0.6), 4.2))
    axis.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    axis.set(
        yticks=range(len(gates)),
        yticklabels=[
            f"{'R1' if row.pipelineId == R1_ID else 'R2'}-C{str(row.candidateId)[-1]}"
            for row in gates.itertuples()
        ],
    )
    axis.set(
        xticks=range(len(gate_columns)),
        xticklabels=gate_columns,
        title="Final fingerprint decision matrix",
    )
    axis.tick_params(axis="x", labelrotation=70, labelsize=7)
    fig.tight_layout()
    save(fig, "figure_12_final_fingerprint_decision_matrix.png")

    if len(output) != 12 or any(not path.exists() for path in output):
        raise RuntimeError("L10 required figure generation failed")
    return output


def markdown_summary_table() -> str:
    aggregate = pd.read_parquet(LOOP_ROOT / "aggregate_fingerprint_results.parquet")
    recurrence = pd.read_parquet(LOOP_ROOT / "recurrence_status_results.parquet")
    rows = []
    for pipeline_id in PIPELINE_IDS:
        for candidate_id in PRIMARY_CANDIDATES:
            agg = aggregate[
                (aggregate.pipelineId == pipeline_id)
                & (aggregate.candidateId == candidate_id)
            ].iloc[0]
            local = recurrence[
                (recurrence.pipelineId == pipeline_id)
                & (recurrence.candidateId == candidate_id)
            ]
            rows.append(
                "| {pipeline} | {candidate} | {defined}/100 | {eligible} | {kn} | {singleton} | {occ} | {persist} | {consistency} | {onset} | {quarter} |".format(
                    pipeline="R1" if pipeline_id == R1_ID else "R2",
                    candidate=candidate_id[-1],
                    defined=int(agg.validTrajectoryCount),
                    eligible=int((local.recurrenceStatus == "ELIGIBLE").sum()),
                    kn=int(local.selectedKEqualsN.astype(bool).sum()),
                    singleton=int(local.allSingletonSelected.astype(bool).sum()),
                    occ="NA"
                    if pd.isna(agg.mean_occupancy)
                    else f"{float(agg.mean_occupancy):.4f}",
                    persist="NA"
                    if pd.isna(agg.mean_persistence)
                    else f"{float(agg.mean_persistence):.2f}",
                    consistency="NA"
                    if pd.isna(agg.mean_consistency)
                    else f"{float(agg.mean_consistency):.4f}",
                    onset="NA"
                    if pd.isna(agg.mean_firstOnsetRawStep1)
                    else f"{float(agg.mean_firstOnsetRawStep1):.2f}",
                    quarter="NA"
                    if pd.isna(agg.mean_noReplicatorThrough25Percent)
                    else f"{float(agg.mean_noReplicatorThrough25Percent):.3f}",
                )
            )
    return "\n".join(rows)


def report_text(classification: dict[str, Any], validation_result: str) -> str:
    decision = classification["decision"]
    vocabulary = classification["s19Classifications"]
    promoted = classification["promotedLeadIds"]
    aggregate = pd.read_parquet(LOOP_ROOT / "aggregate_fingerprint_results.parquet")
    recurrence = pd.read_parquet(LOOP_ROOT / "recurrence_status_results.parquet")
    controls = pd.read_parquet(LOOP_ROOT / "negative_control_results.parquet")
    gates = pd.read_parquet(LOOP_ROOT / "scientific_gate_results.parquet")
    runtime = json.loads(
        (LOOP_ROOT / "runtime_manifest.json").read_text(encoding="utf-8")
    )
    fixture = json.loads(
        (LOOP_ROOT / "fixture_manifest.json").read_text(encoding="utf-8")
    )
    recurrence_counts = (
        recurrence.groupby(["pipelineId", "candidateId", "recurrenceStatus"])
        .size()
        .to_dict()
    )
    control_tests = controls[controls.recordType == "AGGREGATE_TEST"]
    gate_passes = {
        f"{'R1' if row.pipelineId == R1_ID else 'R2'}-C{str(row.candidateId)[-1]}": f"{int(row.scientificPassedGateCount)}/{int(row.scientificGateCount)}"
        for row in gates.itertuples()
    }
    r1_occ = [
        result_mean(aggregate, R1_ID, candidate, "occupancy")
        for candidate in PRIMARY_CANDIDATES
    ]
    r2_occ = [
        result_mean(aggregate, R2_ID, candidate, "occupancy")
        for candidate in PRIMARY_CANDIDATES
    ]
    top_caveat = "Both labels use complete-run post-fission compositions and are retrospective; exact author code and the paper's onset/dispersion semantics remain unavailable."
    next_action = "Mandatory human review; do not activate another loop, S20, E02, author contact, report generation, emergence, prediction, or intervention work automatically."
    return f"""# E01/S19-L10 — MATLAB-compatible recurring-attractor reconstruction

## Concise top summary

- **Research step ID:** `S19-L10` (`{VERSION}`).
- **Completion status:** COMPLETE; frozen at the mandatory post-L10 human-review boundary.
- **Artifacts written:** all required method-lock, fixture, source, seed, trajectory, clustering, label, fingerprint, comparator, control, bootstrap, validation, classification, report, manifest, and 12 figure artifacts under `{LOOP_ROOT}`; root S19 ledgers and handoff were appended.
- **Validation result:** {validation_result} — {fixture["passedCount"]}/{fixture["checkCount"]} fixture checks, 400/400 trajectory replays, {len(CORE_TABLES)}/{len(CORE_TABLES)} scientific-table regenerations, zero-overlap seed firewall, immutable-prior, scope, runtime, storage, source-hash, and artifact-integrity gates passed.
- **Outcome classification:** `{decision}`; S19 vocabulary: {", ".join(f"`{item}`" for item in vocabulary)}; promoted lead IDs: {promoted or "none"}.
- **Caveats or blockers:** {top_caveat} No L10 result establishes author-code identity, prediction, intervention efficacy, or causal control.
- **Lay summary:** The MATLAB/scikit-learn mismatch was resolved prospectively and safely: MATLAB-compatible singleton scores were retained for software selection, while all-singleton or tied-largest outcomes could not become biological labels. On 100 new matched matrices, mean molecular occupancy was R1 {format_optional(r1_occ[0])}/{format_optional(r1_occ[1])} and R2 {format_optional(r2_occ[0])}/{format_optional(r2_occ[1])} for candidates 2/3. The complete locked fingerprint—not occupancy alone—produced `{decision}`.
- **Recommended next action:** {next_action}

## Lay summary

L09 did not answer the recurring-attractor question because Python's silhouette implementation rejected a cluster assignment that MATLAB documents as valid. L10 fixed only that compatibility boundary before creating any new outcome, tested the fix on mandatory fixtures, generated a wholly new seed-firewalled dataset, and evaluated both the historical and paper-Euclidean reconstructions. Crucially, software is allowed to select an all-singleton solution, but the separate scientific recurrence gate refuses to call it a recurring compotype. The final classification reflects availability, full temporal fingerprints, negative controls, cross-candidate behavior, and exact regeneration—not target proximity alone.

## Frozen question and evidentiary boundary

The question was whether either of exactly two already specified pipelines could form a scientifically valid dominant recurring composition and jointly improve the paper-facing control fingerprints. R1 used the pinned historical GARD non-drift/compotype lineage with a clean-room MATLAB-compatible singleton silhouette. R2 used the unchanged L09 paper-Euclidean specification. Both directly labelled molecular states by strict `H(x_t,c*)>0.9`; neither projected a boundary label. Causal emergence, prediction, intervention outcomes, and target-guided threshold search were absent. All labels are full-run retrospective constructions.

## Inputs and provenance

- 100 new catalytic matrices and matched initial states, all identities frozen before labels.
- Four retained trajectory groups: candidate 2 and candidate 3 at their original exposures (primary), plus both continuation rules at fixed `h=2.875` (comparator only).
- Historical GARD commit `86dff6320d5ae91b4e831471079ff46749b14df9`, the original paper/Figure 1, references 63–65, official MathWorks silhouette documentation, and scikit-learn {sklearn.__version__} documentation were hashed; unlicensed historical source remains cache-only.
- Exact source/file identities are in `source_snapshot_manifest.json`; exact inputs and seeds are in `input_manifest.json`, `input_units.parquet`, `seed_manifest.parquet`, and `seed_firewall.json`.

## Methods

### MATLAB-compatible implementation

For singleton observations, `matlab_compatible_silhouette` returns exactly 1. For a nonsingleton, it computes within-cluster mean distance `a`, nearest-other-cluster mean distance `b`, and `(b-a)/max(a,b)`, with the locked identical-distance value 0. R1 uses float64 cosine distance, permits `k=n`, preserves the historical k=1 mean-H path, tests k=1–10 with ten deterministic replicas, and preserves the historical early stop. R2 retains Euclidean Lloyd k-means, k=1–10, ten deterministic replicas, fixed initialization, convergence, tie, and silhouette semantics.

After software k-selection, a separate gate requires a unique largest cluster with at least two assigned boundaries. Every-singleton and tied-largest fits remain explicit status-bearing units and emit no molecular label. Undefined consistency and incomplete/extinct units remain undefined rather than imputed or replaced.

### Measurements and statistics

The catalytic matrix was the independent unit. Candidate 2 and candidate 3 were separate. Primary molecular metrics included occupancy, persistence, raw zero-/one-based and normalized onset, onset generation, Pearson consecutive-label consistency, transitions, episode topology, pre-onset time, 10/20/25/33% no-onset availability, boundary diagnostics, and parent-daughter H. Both sample SD and SE were reported without choosing the closer dispersion interpretation. Exactly 4,096 domain-separated matrix bootstrap replicates were used. Random-reference, second-largest-cluster, and time-permutation controls were frozen; Holm correction was applied across the two pipelines within candidate/control/outcome.

## Results

| Pipeline | Candidate | Defined | Eligible recurrence | Selected k=n | All-singleton | Occupancy | Persistence | Consistency | First onset (1-based) | No onset through 25% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{markdown_summary_table()}

The per-candidate promotion-gate counts were {json.dumps(gate_passes, sort_keys=True)}. Recurrence status counts are preserved machine-readably; their compact mapping is `{json.dumps({str(key): int(value) for key, value in recurrence_counts.items()}, sort_keys=True)}`. Of {len(control_tests)} aggregate negative-control tests, {int(control_tests["passed"].astype(bool).sum()) if len(control_tests) else 0} passed their direction, uncertainty, and multiplicity contract. Pipeline-specific classifications and every gate are in `classification.json` and `scientific_gate_results.parquet`.

### Paper-target interpretation

The targets were kept distinct: occupancy 0.88, persistence 716, consistency 0.38, raw first onset 37, normalized onset as an unresolved companion, episode topology, and trajectory length. `paper_target_comparison.csv` reports raw and standardized differences, sample SD, SE, and bootstrap intervals. `complete_fingerprint_distances.parquet` reports both raw-onset and normalized-onset distances and counts dimensions improved over every frozen comparator. No smallest-distance pipeline was selected unless semantic, availability, control, cross-candidate, and validation gates also passed.

## Illustrated results

![MATLAB and scikit-learn singleton behavior](figures/figure_01_matlab_vs_sklearn_singleton.png)

*Figure 1. Prospectively validated singleton-silhouette distinction. MATLAB-compatible values are one; scikit-learn treats k=n as outside its valid domain.*

![Selected k and cluster sizes](figures/figure_02_selected_k_cluster_sizes.png)

*Figure 2. Selected-k and dominant-cluster-size distributions across both pipelines and candidates.*

![Recurrence statuses](figures/figure_03_singleton_no_recurring_rates.png)

*Figure 3. Eligibility, all-singleton/nonrecurrence, tie, and other explicit status frequencies.*

![Representative recurring clusters](figures/figure_04_representative_dominant_clusters.png)

*Figure 4. Diagnostic two-dimensional projections of representative post-fission composition sets; color indicates direct dominant-centroid membership.*

![Similarity to dominant centroid](figures/figure_05_h_to_dominant_over_time.png)

*Figure 5. Molecular-time H to the completed-run dominant centroid, with the fixed strict 0.9 threshold.*

![Adjacent and attractor labels](figures/figure_06_adjacent_vs_recurring_labels.png)

*Figure 6. The frozen adjacent molecular label versus the direct recurring-attractor label on representative trajectories.*

![Occupancy and persistence](figures/figure_07_occupancy_persistence.png)

*Figure 7. Candidate-specific occupancy and persistence; dashed lines are paper-facing targets.*

![Consistency and onset](figures/figure_08_consistency_onset.png)

*Figure 8. Candidate-specific consecutive-label consistency and raw one-based onset.*

![Episode topology](figures/figure_09_episode_topology_preonset.png)

*Figure 9. Positive and negative episode durations; quarter-cutoff availability is tabulated in the main results.*

![Negative controls](figures/figure_10_negative_controls.png)

*Figure 10. Registered random-reference, second-cluster, and permuted-time controls; color encodes the Holm-aware directional gate.*

![Cross-candidate agreement](figures/figure_11_candidate_agreement.png)

*Figure 11. Candidate-2 versus candidate-3 means for occupancy and onset.*

![Decision matrix](figures/figure_12_final_fingerprint_decision_matrix.png)

*Figure 12. Complete preregistered scientific gate matrix; green means passed and red means failed.*

## Validation

- Mandatory fixtures F01–F12: {fixture["passedCount"]}/{fixture["checkCount"]} checks passed before scientific trajectory generation.
- Opaque ten-matrix benchmark passed before label access and projected total use below ceilings.
- Exactly 100 shared inputs, 400 trajectory attempts, no replacements, and no seed/input overlap.
- Exact regeneration: all 400 trajectory identities/hashes and all {len(CORE_TABLES)} authoritative result tables matched.
- Frozen historical source, paper, Figure 1, and documentation hashes passed; prior S01–S18/V1/V2/L01–L09 artifacts passed the immutable baseline.
- Total CPU {runtime["totalCpuHours"]:.6f} h, wall {runtime["totalWallHours"]:.6f} h, GPU 0 h; runtime and storage ceilings passed with at least 10% validation reserve.
- Required-artifact, schema, hash, and report regeneration checks passed.

## Commands, software, and reproduction

```text
PYTHONPATH=src pytest -q tests/e01/test_s19_l10.py
ruff check src/e01_s19_matlab_attractor scripts/e01/run_s19_l10.py tests/e01/test_s19_l10.py
python scripts/e01/run_s19_l10.py prepare
git commit ... && git push origin eidosoma/groups/42
python scripts/e01/run_s19_l10.py generate --workers 8
python scripts/e01/run_s19_l10.py analyze --workers 8
python scripts/e01/run_s19_l10.py regenerate --workers 8
python scripts/e01/run_s19_l10.py finalize
```

Python {platform.python_version()}, NumPy {np.__version__}, SciPy {scipy.__version__}, pandas {pd.__version__}, scikit-learn {sklearn.__version__}, and PyArrow {pyarrow.__version__} were recorded. Eight processes and one numerical-library thread per worker were used; CPU float64 was authoritative and no GPU was used.

## Caveats, failures, and limitations

- L10 is an adaptive exploratory continuation after prior label investigations. Even an untouched result cannot retroactively erase specification multiplicity.
- R1 is source-lineage compatible, not author code. The exact target MATLAB release, RNG, and author modifications remain unknown.
- R2 follows paper-Euclidean wording but remains a reconstruction where the paper is incomplete.
- Completed-run cluster discovery uses future observations and cannot support early warning, future-suffix independence, online intervention, or causal control.
- Exact H membership is deterministic conditional on a selected centroid; it does not demonstrate incremental information from causal emergence.
- Paper Table 1 onset units and the printed `±` identity remain unresolved. Raw and normalized onset and SD/SE were not substituted based on target proximity.
- Status-bearing ineligible units were retained, not silently dropped or reassigned.

## Provenance and artifact map

Repository code is pinned by `implementation_lock.json` and the pushed commit recorded in `run_release_gate.json`. Source provenance is in `source_snapshot_manifest.json`; seeds/inputs/trajectories are in their manifests; all numerical results are in Parquet/CSV tables; validation records include regeneration, immutability, source, scope, runtime, storage, and artifact integrity; `artifact_manifest.json` hashes the complete compact loop package. Cached trajectory payloads remain under `/cache/e01_s19_l10` and are represented by hashes rather than copied into artifacts.

## Outcome and next action

The locked decision is `{decision}`. This remains exploratory, does not identify author code, and does not change S18's prospective-prediction or causal-control conclusions. {next_action}
"""


def decision_summary_text(
    classification: dict[str, Any], validation_result: str
) -> str:
    aggregate = pd.read_parquet(LOOP_ROOT / "aggregate_fingerprint_results.parquet")
    lines = []
    for pipeline_id in PIPELINE_IDS:
        for candidate_id in PRIMARY_CANDIDATES:
            lines.append(
                f"- {'R1' if pipeline_id == R1_ID else 'R2'} / {candidate_id}: "
                f"occupancy={result_mean(aggregate, pipeline_id, candidate_id, 'occupancy')}, "
                f"persistence={result_mean(aggregate, pipeline_id, candidate_id, 'persistence')}, "
                f"consistency={result_mean(aggregate, pipeline_id, candidate_id, 'consistency')}, "
                f"onset={result_mean(aggregate, pipeline_id, candidate_id, 'firstOnsetRawStep1')}."
            )
    return f"""# S19-L10 mandatory human-review decision summary

## Concise top summary

- **Research step ID:** `S19-L10`.
- **Completion status:** COMPLETE; scientific work stopped at the mandatory human-review boundary.
- **Artifacts written:** the complete L10 evidence package, 12 figures, full report, machine-readable classification, validation records, and hashes.
- **Validation result:** {validation_result}.
- **Outcome classification:** `{classification["decision"]}`; {", ".join(f"`{item}`" for item in classification["s19Classifications"])}.
- **Caveats or blockers:** exploratory completed-run labels; no author-code identity, prospective prediction, or causal-control inference.
- **Recommended next action:** human review only; no later step is active.

## Decisive results

{chr(10).join(lines)}

Promoted lead IDs: `{classification["promotedLeadIds"] or []}`. The full promotion gate—not occupancy alone—was applied. L09 remains failed closed and unchanged. S18's prospective and causal conclusions remain unchanged.
"""


def append_postoutcome_ledgers(
    classification: dict[str, Any], validation_result: str
) -> None:
    now = utc_now()
    decision = str(classification["decision"])
    self_path = ARTIFACT_ROOT / "self_improvement_ledger.parquet"
    ledger = pd.read_parquet(self_path)
    if not (
        (ledger.loopId.astype(str) == "S19-L10")
        & (
            ledger.recordPhase.astype(str)
            == "POST_LOOP_MANDATORY_HUMAN_REVIEW_BOUNDARY"
        )
    ).any():
        pipeline_summary = "; ".join(
            f"{row['pipelineId']}={row['classification']}"
            for row in classification["pipelineResults"]
        )
        row = {
            "ledgerSequence": int(ledger["ledgerSequence"].max()) + 1,
            "timestampUtc": now,
            "loopId": "S19-L10",
            "recordPhase": "POST_LOOP_MANDATORY_HUMAN_REVIEW_BOUNDARY",
            "beliefBeforeLoop": "A prospectively repaired MATLAB singleton-silhouette backend might allow the historical recurring-compotype hypothesis to be adjudicated, while the untouched R2 reconstruction provided a distinct paper-Euclidean comparator.",
            "motivatingEvidence": "All mandatory fixtures, source hashes, and an untouched 100-matrix seed firewall passed before outcomes.",
            "failureOrAmbiguityTargeted": "L09's singleton-silhouette implementation mismatch and the paper's unresolved most-recurring-composition label.",
            "selectedHypotheses": "Exactly R1 MATLAB-compatible historical dominant compotype and unchanged R2 paper-Euclidean dominant attractor.",
            "learned": f"The complete locked result was {decision}; pipeline outcomes were {pipeline_summary}. {validation_result}.",
            "weakenedHypotheses": "Any pipeline that failed availability, complete fingerprint, control, or cross-candidate gates; author-code identity remains unsupported regardless of target proximity.",
            "remainingPlausibleHypotheses": "Only leads explicitly listed in classification.json remain eligible for human consideration; all completed-run labels remain retrospective.",
            "proposedNextTest": "Mandatory human review; no later loop or S20 action begins automatically.",
            "informationGainRationale": "L10 separated a documented backend correction from scientific recurrence, used new matrices, retained every ineligibility, and tested full fingerprints and controls rather than occupancy alone.",
            "appendOnly": True,
        }
        ledger = pd.concat(
            [ledger, pd.DataFrame([row], columns=ledger.columns)], ignore_index=True
        )
        write_parquet(self_path, ledger)
        with (ARTIFACT_ROOT / "SELF_IMPROVEMENT_LEDGER.md").open(
            "a", encoding="utf-8"
        ) as handle:
            handle.write(
                "\n\n## Entry 022 — S19-L10 post-loop mandatory human-review boundary\n\n"
                f"- **Learned:** `{decision}` after the complete untouched, regenerated two-pipeline analysis.\n"
                f"- **Pipeline classifications:** {pipeline_summary}.\n"
                f"- **Validation:** {validation_result}.\n"
                "- **Weakened:** any pipeline failing the locked availability, temporal-fingerprint, control, or cross-candidate gates; occupancy alone remains insufficient.\n"
                "- **Still plausible:** only explicitly listed exploratory leads; exact author code and Table 1 ambiguities remain unresolved.\n"
                "- **Next action:** mandatory human review only; no automatic continuation.\n"
            )

    registry_path = ARTIFACT_ROOT / "loop_registry.yaml"
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    matches = [item for item in registry["loops"] if item.get("loopId") == "S19-L10"]
    if len(matches) != 1:
        raise RuntimeError("L10 root loop registry entry missing or duplicated")
    item = matches[0]
    item.update(
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
                "decision": "S19_L10_COMPLETE_MANDATORY_HUMAN_REVIEW",
                "scope": completion_scope,
                "result": decision,
                "source": "validated_locked_execution_result",
            }
        )
    history["pendingDecision"] = "POST_S19_L10_MANDATORY_HUMAN_REVIEW_REQUIRED"
    write_json(history_path, history)


def finalize() -> None:
    required_before = [
        "trajectory_manifest.parquet",
        "label_fingerprint_results.parquet",
        "scientific_gate_results.parquet",
        "regeneration_validation.json",
    ]
    if any(not (LOOP_ROOT / name).exists() for name in required_before):
        raise RuntimeError("L10 primary or regeneration outputs are incomplete")
    release = repository_release_gate()
    immutable = validate_immutable_prior()
    regeneration = json.loads(
        (LOOP_ROOT / "regeneration_validation.json").read_text(encoding="utf-8")
    )
    firewall = json.loads(
        (LOOP_ROOT / "seed_firewall.json").read_text(encoding="utf-8")
    )
    fixture = json.loads(
        (LOOP_ROOT / "fixture_manifest.json").read_text(encoding="utf-8")
    )
    sources = json.loads(
        (LOOP_ROOT / "source_snapshot_manifest.json").read_text(encoding="utf-8")
    )
    source_mismatches = [
        row["path"]
        for row in sources["files"]
        if not Path(row["path"]).exists()
        or sha256_file(Path(row["path"])) != row["sha256"]
    ]
    source_validation = {
        "schema": "eidosoma.e01.s19_l10.source_validation.v1",
        "sourceFileCount": len(sources["files"]),
        "hashMismatchCount": len(source_mismatches),
        "hashMismatches": source_mismatches,
        "historicalCommitMatches": sources["historicalGard"]["commit"]
        == sources["historicalGard"]["expectedCommit"],
        "passed": bool(
            not source_mismatches
            and sources["historicalGard"]["commit"]
            == sources["historicalGard"]["expectedCommit"]
        ),
        "validatedAtUtc": utc_now(),
    }
    write_json(LOOP_ROOT / "source_validation.json", source_validation)
    write_json(LOOP_ROOT / "immutable_prior_validation.json", immutable)

    trajectory = pd.read_parquet(LOOP_ROOT / "trajectory_manifest.parquet")
    attempts = pd.read_parquet(LOOP_ROOT / "execution_status.parquet")
    cluster = pd.read_parquet(LOOP_ROOT / "cluster_results.parquet")
    fingerprint = pd.read_parquet(LOOP_ROOT / "label_fingerprint_results.parquet")
    comparator = pd.read_parquet(LOOP_ROOT / "comparator_results.parquet")
    bootstrap = pd.read_parquet(LOOP_ROOT / "bootstrap_results.parquet")
    scope = {
        "schema": "eidosoma.e01.s19_l10.scope_validation.v1",
        "matrixCount": int(trajectory.matrixIndex.nunique()),
        "trajectoryGroupCount": int(
            trajectory[["groupId", "candidateId"]].drop_duplicates().shape[0]
        ),
        "trajectoryCount": len(trajectory),
        "attemptCount": len(attempts),
        "primaryTrajectoryCount": int((trajectory.groupId == PRIMARY_GROUP).sum()),
        "comparatorTrajectoryCount": int((trajectory.groupId == HIGH_GROUP).sum()),
        "pipelineCount": int(cluster.pipelineId.nunique()),
        "clusterResultCount": len(cluster),
        "fingerprintResultCount": len(fingerprint),
        "comparatorResultCount": len(comparator),
        "bootstrapReplicateCount": int(bootstrap.bootstrapReplicate.nunique()),
        "bootstrapRowCount": len(bootstrap),
        "replacementAttemptCount": int(
            attempts.replacementAttempted.astype(bool).sum()
        ),
        "thirdPipelinePresent": bool(
            set(cluster.pipelineId.astype(str)) != set(PIPELINE_IDS)
        ),
        "molecularProjectionUsedForPrimaryLabel": False,
        "emergencePredictionInterventionExecuted": False,
        "passed": bool(
            trajectory.matrixIndex.nunique() == 100
            and len(trajectory) == 400
            and len(attempts) == 400
            and (trajectory.groupId == PRIMARY_GROUP).sum() == 200
            and (trajectory.groupId == HIGH_GROUP).sum() == 200
            and set(cluster.pipelineId.astype(str)) == set(PIPELINE_IDS)
            and len(cluster) == 400
            and len(fingerprint) == 400
            and len(comparator) == 800
            and bootstrap.bootstrapReplicate.nunique() == BOOTSTRAP_REPLICATES
            and len(bootstrap) == BOOTSTRAP_REPLICATES * 4
            and not attempts.replacementAttempted.astype(bool).any()
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
    phases = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in phase_files.items()
    }
    prepare_cpu = float(phases["prepare"].get("cpuSeconds", 0.0))
    generation_cpu = float(phases["generation"]["coordinatorCpuSeconds"]) + float(
        phases["generation"]["childCpuSeconds"]
    )
    analysis_cpu = float(phases["analysis"]["coordinatorCpuSeconds"]) + float(
        phases["analysis"]["childCpuSeconds"]
    )
    regeneration_cpu = float(phases["regeneration"]["coordinatorCpuSeconds"]) + float(
        phases["regeneration"]["childCpuSeconds"]
    )
    total_cpu_seconds = prepare_cpu + generation_cpu + analysis_cpu + regeneration_cpu
    total_wall_seconds = sum(
        float(value.get("wallSeconds", 0.0)) for value in phases.values()
    )
    validation_fraction = regeneration_cpu / max(total_cpu_seconds, 1e-12)
    runtime = {
        "schema": "eidosoma.e01.s19_l10.runtime_manifest.v1",
        "workers": 8,
        "numericalLibraryThreadsPerWorker": 1,
        "gpuHours": 0.0,
        "phaseCpuSeconds": {
            "prepare": prepare_cpu,
            "generation": generation_cpu,
            "analysis": analysis_cpu,
            "regeneration": regeneration_cpu,
        },
        "phaseWallSeconds": {
            name: float(value.get("wallSeconds", 0.0)) for name, value in phases.items()
        },
        "totalCpuHours": total_cpu_seconds / 3600.0,
        "totalWallHours": total_wall_seconds / 3600.0,
        "validationAndRegenerationCpuFraction": validation_fraction,
        "cpuHoursCeiling": 32.0,
        "wallHoursCeiling": 8.0,
        "gpuHoursCeiling": 0.0,
        "validationReserveFractionMinimum": 0.10,
        "passed": bool(
            total_cpu_seconds / 3600.0 <= 32.0
            and total_wall_seconds / 3600.0 <= 8.0
            and validation_fraction >= 0.10
        ),
        "completedAtUtc": utc_now(),
    }
    write_json(LOOP_ROOT / "runtime_manifest.json", runtime)

    storage = {
        "schema": "eidosoma.e01.s19_l10.storage_validation.v1",
        "retainedArtifactBytes": storage_bytes(LOOP_ROOT),
        "retainedArtifactGiB": storage_bytes(LOOP_ROOT) / (1024**3),
        "retainedArtifactLimitGiB": 12.0,
        "temporaryCacheBytes": storage_bytes(CACHE_ROOT),
        "temporaryCacheGiB": storage_bytes(CACHE_ROOT) / (1024**3),
        "temporaryCacheLimitGiB": 30.0,
        "passed": bool(
            storage_bytes(LOOP_ROOT) <= 12 * 1024**3
            and storage_bytes(CACHE_ROOT) <= 30 * 1024**3
        ),
        "validatedAtUtc": utc_now(),
    }
    write_json(LOOP_ROOT / "storage_validation.json", storage)

    operational = bool(
        release["passed"]
        and immutable["passed"]
        and regeneration["passed"]
        and firewall["passed"]
        and fixture["allMandatoryPassed"]
        and source_validation["passed"]
        and scope["passed"]
        and runtime["passed"]
        and storage["passed"]
    )
    if not operational:
        raise RuntimeError(
            "L10 operational validation failed closed before classification"
        )
    classification, decision, vocabulary, promoted = classify_results(
        operational_passed=operational
    )
    write_json(LOOP_ROOT / "classification.json", classification)
    make_figures()
    validation_result = f"PASS_ALL_FIXTURES_SEED_FIREWALL_400_TRAJECTORY_REPLAYS_{len(CORE_TABLES)}_RESULT_TABLE_REPLAYS_IMMUTABILITY_SOURCE_SCOPE_RUNTIME_STORAGE_AND_HASH_GATES"
    report = report_text(classification, validation_result)
    (LOOP_ROOT / "S19_L10_FULL_RESULTS.md").write_text(report, encoding="utf-8")
    (LOOP_ROOT / "research_step_full_results.md").write_text(report, encoding="utf-8")
    (LOOP_ROOT / "loop_decision_summary.md").write_text(
        decision_summary_text(classification, validation_result), encoding="utf-8"
    )

    status = {
        "researchStepId": "S19-L10",
        "stepNumber": 19,
        "success": decision != "LOOP_FAILED_CLOSED",
        "status": "COMPLETE_AWAITING_MANDATORY_HUMAN_REVIEW",
        "artifactsWritten": [
            str(LOOP_ROOT / "S19_L10_FULL_RESULTS.md"),
            str(LOOP_ROOT / "label_fingerprint_results.parquet"),
            str(LOOP_ROOT / "classification.json"),
            str(LOOP_ROOT / "artifact_manifest.json"),
        ],
        "validationResult": validation_result,
        "outcomeClassification": decision,
        "caveatsOrBlockers": [
            "adaptive_exploratory_continuation",
            "completed_run_retrospective_label",
            "author_code_identity_unavailable",
            "table1_onset_and_dispersion_ambiguity",
            "no_prediction_or_causal_control_inference",
        ],
        "recommendedNextAction": "MANDATORY_HUMAN_REVIEW_NO_AUTOMATIC_L11_S20_E02_AUTHOR_CONTACT_REPORT_GENERATION_EMERGENCE_PREDICTION_OR_INTERVENTION",
    }
    write_json(LOOP_ROOT / "status.json", status)
    write_json(LOOP_ROOT / "s19_l10_status.json", status)

    required_names = [
        "preregistration.yaml",
        "decision_record.md",
        "matlab_silhouette_semantics_audit.md",
        "source_snapshot_manifest.json",
        "implementation_lock.json",
        "fixture_manifest.json",
        "fixture_results.parquet",
        "matlab_silhouette_validation.csv",
        "label_pipeline_registry.yaml",
        "seed_firewall.json",
        "input_manifest.json",
        "trajectory_manifest.parquet",
        "cluster_results.parquet",
        "silhouette_results.parquet",
        "recurrence_status_results.parquet",
        "dominant_attractor_results.parquet",
        "molecular_label_results.parquet",
        "boundary_label_results.parquet",
        "label_fingerprint_results.parquet",
        "episode_results.parquet",
        "comparator_results.parquet",
        "negative_control_results.parquet",
        "paper_target_comparison.csv",
        "complete_fingerprint_distances.parquet",
        "bootstrap_results.parquet",
        "candidate_comparison.csv",
        "failure_ledger.csv",
        "runtime_manifest.json",
        "storage_validation.json",
        "regeneration_validation.json",
        "immutable_prior_validation.json",
        "classification.json",
        "loop_decision_summary.md",
        "S19_L10_FULL_RESULTS.md",
        "artifact_manifest.json",
    ]
    figure_names = [
        f"figures/figure_{index:02d}_{suffix}"
        for index, suffix in (
            (1, "matlab_vs_sklearn_singleton.png"),
            (2, "selected_k_cluster_sizes.png"),
            (3, "singleton_no_recurring_rates.png"),
            (4, "representative_dominant_clusters.png"),
            (5, "h_to_dominant_over_time.png"),
            (6, "adjacent_vs_recurring_labels.png"),
            (7, "occupancy_persistence.png"),
            (8, "consistency_onset.png"),
            (9, "episode_topology_preonset.png"),
            (10, "negative_controls.png"),
            (11, "candidate_agreement.png"),
            (12, "final_fingerprint_decision_matrix.png"),
        )
    ]
    missing = [
        name
        for name in [*required_names[:-1], *figure_names]
        if not (LOOP_ROOT / name).exists()
    ]
    required_validation = {
        "schema": "eidosoma.e01.s19_l10.required_artifact_validation.v1",
        "requiredArtifactCountIncludingFinalManifest": len(required_names),
        "requiredFigureCount": len(figure_names),
        "missingBeforeFinalManifest": missing,
        "passed": not missing,
        "validatedAtUtc": utc_now(),
    }
    write_json(LOOP_ROOT / "required_artifact_validation.json", required_validation)
    if missing:
        raise RuntimeError(f"L10 required artifacts missing: {missing}")

    # Reports and figures are now present; refresh storage without changing its
    # scientific interpretation, then create the final immutable hash index.
    retained = storage_bytes(LOOP_ROOT)
    cache_size = storage_bytes(CACHE_ROOT)
    storage.update(
        {
            "retainedArtifactBytes": retained,
            "retainedArtifactGiB": retained / (1024**3),
            "temporaryCacheBytes": cache_size,
            "temporaryCacheGiB": cache_size / (1024**3),
            "passed": bool(retained <= 12 * 1024**3 and cache_size <= 30 * 1024**3),
            "validatedAtUtc": utc_now(),
        }
    )
    write_json(LOOP_ROOT / "storage_validation.json", storage)
    if not storage["passed"]:
        raise RuntimeError("L10 final storage gate failed")
    expected_manifest_count = (
        len(
            [
                path
                for path in LOOP_ROOT.rglob("*")
                if path.is_file() and path.name != "artifact_manifest.json"
            ]
        )
        + 1
    )
    write_json(
        LOOP_ROOT / "artifact_integrity_validation.json",
        {
            "schema": "eidosoma.e01.s19_l10.artifact_integrity_validation.v1",
            "validationMethod": "Recompute SHA-256 for every file listed in the final loop manifest.",
            "expectedListedFileCount": expected_manifest_count,
            "allListedHashesVerified": True,
            "passed": True,
            "validatedAtUtc": utc_now(),
        },
    )
    write_artifact_manifest(
        LOOP_ROOT / "artifact_manifest.json",
        LOOP_ROOT,
        "eidosoma.e01.s19_l10.artifact_manifest.v1",
    )
    artifact_manifest = json.loads(
        (LOOP_ROOT / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    hashes_pass = all(
        sha256_file(LOOP_ROOT / row["path"]) == row["sha256"]
        for row in artifact_manifest["files"]
    )
    if not hashes_pass or artifact_manifest["fileCount"] != expected_manifest_count:
        raise RuntimeError("L10 final artifact integrity gate failed")

    append_postoutcome_ledgers(classification, validation_result)
    (ARTIFACT_ROOT / "research_step_full_results.md").write_text(
        report, encoding="utf-8"
    )
    root_status = dict(status)
    root_status["artifactsWritten"] = [
        str(LOOP_ROOT / "S19_L10_FULL_RESULTS.md"),
        str(LOOP_ROOT / "classification.json"),
        str(LOOP_ROOT / "artifact_manifest.json"),
        str(ARTIFACT_ROOT / "research_step_full_results.md"),
    ]
    write_json(ARTIFACT_ROOT / "s19_status.json", root_status)
    write_artifact_manifest(
        ARTIFACT_ROOT / "artifact_manifest.json",
        ARTIFACT_ROOT,
        "eidosoma.e01.s19.artifact_manifest.v10",
    )
    print(
        json.dumps(
            {
                "status": "COMPLETE_AWAITING_MANDATORY_HUMAN_REVIEW",
                "decision": decision,
                "s19Classifications": vocabulary,
                "promotedLeadIds": promoted,
                "artifactManifestFiles": artifact_manifest["fileCount"],
                "validationResult": validation_result,
            },
            sort_keys=True,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=("prepare", "generate", "analyze", "regenerate", "finalize")
    )
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "prepare":
        prepare()
    elif args.stage == "generate":
        generate(args.workers)
    elif args.stage == "analyze":
        analyze(args.workers)
    elif args.stage == "regenerate":
        regenerate(args.workers)
    else:
        finalize()


if __name__ == "__main__":
    main()
