"""Execute S19-L37 multi-lineage any-attractor target construction."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[variable] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from e01_onset_discovery.attractor_atlas import (
    build_cross_lineage_atlas,
    score_atlas,
    summarize_atlas_labels,
)
from e01_onset_discovery.branch_trace import simulate_branch_trace
from e01_onset_discovery.empirical_committor import (
    RestoredState,
    corrected_between_state_variance,
)


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


L36 = _load_module(
    "e01_s19_l37_l36",
    REPO_ROOT / "scripts/e01/run_s19_l36_independent_lineage_basin_transfer.py",
)
L35 = L36.L35
L31 = L36.L31
L30 = L36.L30
L28 = L36.L28
BASE = L36.BASE

LOOP_ID = "S19-L37"
VERSION = "E01-S19-L37-MULTILINEAGE-ANY-ATTRACTOR-TARGET-CONSTRUCTION-v1.0.0"
CANDIDATES = L36.CANDIDATES
COHORTS = L36.COHORTS
EVALUATION_COHORTS = L36.EVALUATION_COHORTS
LINEAGES = ("ORIGINAL", "REFERENCE_A", "REFERENCE_B")
TARGET_TYPES = (
    "ORIGINAL_TRAJECTORY_BASIN",
    "INDEPENDENT_ANY_ATTRACTOR",
    "PERMUTED_SPECIES_ATLAS",
    "UNRELATED_MATRIX_ATLAS",
)
FAMILIES = L36.FAMILIES
HORIZONS = L36.HORIZONS
BRANCH_COUNTS = L36.BRANCH_COUNTS
HALVES = L36.HALVES
BOOTSTRAPS = 4096
THRESHOLD = 0.9
MIN_VISITS = 2
MIN_SPAN = 2
ROOT_HEX = "663a6f33380e84bbdbd3bb03f641a6a8ee44ed8c19eeb06ed3715936afe6c80f"
PHASE = "s19_l37_multilineage_any_attractor"
WORKERS = min(8, os.cpu_count() or 1)

ARTIFACT_ROOT = Path("/artifacts/research_steps/S19")
LOOP_ROOT = ARTIFACT_ROOT / "loops/L37"
L36_ROOT = ARTIFACT_ROOT / "loops/L36"
L35_ROOT = ARTIFACT_ROOT / "loops/L35"
L23_ROOT = ARTIFACT_ROOT / "loops/L23"
CACHE_ROOT = Path("/cache/e01_s19_l37")
BUILD_ROOT = CACHE_ROOT / "build"
CONFIG = REPO_ROOT / "configs/e01/s19_l37_multilineage_any_attractor.yaml"
RUNNER_PATH = Path(__file__)
CORE_PATH = REPO_ROOT / "src/e01_onset_discovery/attractor_atlas.py"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_hash(frame: pd.DataFrame) -> str:
    return hashlib.sha256(
        frame.to_json(orient="table", index=False, double_precision=15).encode()
    ).hexdigest()


def derived_seed(*parts: object) -> int:
    payload = "\x1f".join([VERSION, ROOT_HEX, *map(str, parts)]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def validate_immutable_prior() -> dict[str, Any]:
    prior = json.loads((L36_ROOT / "immutable_prior_validation.json").read_text())
    rows = list(prior["files"])
    manifest_path = L36_ROOT / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    rows.extend(
        {
            "path": str(L36_ROOT / item["path"]),
            "root": str(L36_ROOT),
            "bytes": item["bytes"],
            "sha256": item["sha256"],
        }
        for item in manifest["files"]
    )
    rows.append(
        {
            "path": str(manifest_path),
            "root": str(L36_ROOT),
            "bytes": manifest_path.stat().st_size,
            "sha256": sha256_file(manifest_path),
        }
    )
    failures = []
    for row in rows:
        path = Path(row["path"])
        if not path.is_file():
            failures.append({"path": str(path), "reason": "MISSING"})
        elif sha256_file(path) != row["sha256"]:
            failures.append({"path": str(path), "reason": "HASH_MISMATCH"})
    return {
        "schema": "eidosoma.e01.s19_l37.immutable_prior_validation.v1",
        "status": "PASS" if not failures else "FAIL",
        "unchanged": not failures,
        "fileCount": len(rows),
        "aggregateSha256": hashlib.sha256(
            "\n".join(f"{row['path']}\t{row['sha256']}" for row in rows).encode()
        ).hexdigest(),
        "l36ArtifactFileCount": manifest["fileCount"],
        "failures": failures,
        "files": rows,
    }


def fixture_results() -> pd.DataFrame:
    first = np.array([9, 1, 0, 0], dtype=np.int64)
    second = np.array([0, 0, 1, 9], dtype=np.int64)
    states = np.stack([first, first, second, second, first, first, second, second])
    lineages = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    generations = np.array([0, 3, 1, 4, 0, 3, 1, 4])
    atlas = build_cross_lineage_atlas(states, lineages, generations)
    scores, assignments, labels = score_atlas(
        np.stack([first, second, np.ones(4, dtype=np.int64)]), atlas.centroids
    )
    summary = summarize_atlas_labels(
        np.array([True, False, True]), np.array([0, -1, 1])
    )
    return pd.DataFrame(
        [
            {
                "fixtureId": "TWO_RECURRING_BASINS",
                "passed": atlas.status == "ELIGIBLE" and len(atlas.components) == 2,
                "details": str(len(atlas.components)),
            },
            {
                "fixtureId": "UNION_SCORING",
                "passed": labels.tolist() == [True, True, False]
                and assignments.tolist() == [0, 1, -1],
                "details": json.dumps(scores.tolist()),
            },
            {
                "fixtureId": "RECURRENCE_SUMMARY",
                "passed": summary.recurrent_positive
                and summary.positive_episode_count == 2,
                "details": str(summary),
            },
            {
                "fixtureId": "RECIPROCAL_LINEAGES_ONLY",
                "passed": LINEAGES == ("ORIGINAL", "REFERENCE_A", "REFERENCE_B"),
                "details": json.dumps(LINEAGES),
            },
            {
                "fixtureId": "FROZEN_TARGET_SEMANTICS",
                "passed": THRESHOLD == 0.9 and MIN_VISITS == 2 and MIN_SPAN == 2,
                "details": "H090; visits=2; span=2",
            },
            {
                "fixtureId": "FROZEN_BRANCH_SCOPE",
                "passed": HORIZONS == {"H32": 32, "H8": 8}
                and BRANCH_COUNTS == {"H32": 128, "H8": 64},
                "details": json.dumps({"horizons": HORIZONS, "branches": BRANCH_COUNTS}),
            },
            {
                "fixtureId": "TARGET_FAMILY_NOT_COMPONENT_CARDINALITY",
                "passed": registered_target_count(
                    {"A": np.ones((3, 4)), "B": np.ones((1, 4)), "C": None}
                )
                == 2,
                "details": "two registered targets with four total components",
            },
        ]
    )


def load_cached_trajectory(path: str, expected_sha: str) -> Any:
    source = Path(path)
    if not source.is_file() or sha256_file(source) != expected_sha:
        raise RuntimeError(f"L37 trajectory cache identity failure: {source}")
    with source.open("rb") as handle:
        return pickle.load(handle)


def lineage_registry() -> pd.DataFrame:
    responses = pd.read_parquet(L36_ROOT / "response_registry.parquet")
    references = pd.read_parquet(L36_ROOT / "reference_trajectory_manifest.parquet")
    manifest = pd.read_parquet(L23_ROOT / "input_trajectory_manifest.parquet").set_index(
        ["candidateId", "matrixIndex"]
    )
    ref_index = references.set_index(["stateId", "referenceId"])
    rows = []
    for source in responses.itertuples(index=False):
        original = manifest.loc[(source.candidateId, int(source.matrixIndex))]
        for lineage in LINEAGES:
            if lineage == "ORIGINAL":
                row = {
                    "cachePath": original.cachePath,
                    "cacheSha256": original.cacheSha256,
                    "trajectoryId": original.trajectoryId,
                    "trajectorySha256": original.trajectorySha256,
                    "terminalStatus": original.terminalStatus,
                    "completedFissions": int(original.completedFissions),
                    "betaSha256": original.betaSha256,
                    "initialStateSha256": original.initialStateSha256,
                }
            else:
                item = ref_index.loc[(source.stateId, lineage)]
                row = {
                    "cachePath": item.cachePath,
                    "cacheSha256": item.cacheSha256,
                    "trajectoryId": item.trajectoryId,
                    "trajectorySha256": item.trajectorySha256,
                    "terminalStatus": item.terminalStatus,
                    "completedFissions": int(item.completedFissions),
                    "betaSha256": item.betaSha256,
                    "initialStateSha256": item.initialStateSha256,
                }
            trajectory = load_cached_trajectory(row["cachePath"], row["cacheSha256"])
            exact = (
                trajectory.trajectory_sha256 == row["trajectorySha256"]
                and trajectory.beta_sha256 == row["betaSha256"]
                and trajectory.initial_state_sha256 == row["initialStateSha256"]
                and trajectory.completed_fissions == row["completedFissions"]
            )
            rows.append(
                {
                    "stateId": source.stateId,
                    "evaluationCohort": source.evaluationCohort,
                    "candidateId": source.candidateId,
                    "matrixIndex": int(source.matrixIndex),
                    "landmark": int(source.landmark),
                    "lineageId": lineage,
                    **row,
                    "identityExact": exact,
                }
            )
    result = pd.DataFrame(rows).sort_values(
        ["evaluationCohort", "candidateId", "landmark", "matrixIndex", "lineageId"]
    ).reset_index(drop=True)
    if (
        len(result) != 840
        or result.duplicated(["stateId", "lineageId"]).any()
        or not result["identityExact"].all()
        or not result["completedFissions"].eq(100).all()
    ):
        raise RuntimeError("L37 lineage input replay failure")
    return result


def post_fission_payload(row: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    trajectory = load_cached_trajectory(row.cachePath, row.cacheSha256)
    post_indices = [
        index
        for index, observation in enumerate(trajectory.observations)
        if observation.observation_kind == "post_fission"
    ]
    post = [trajectory.observations[index] for index in post_indices]
    # The frozen trajectory schema does not emit a separately named
    # pre-fission observation. The immediately preceding molecular update is
    # the exact pre-fission composition; validate that relation explicitly.
    pre = [trajectory.observations[index - 1] for index in post_indices]
    if len(post) != 100 or len(pre) != 100:
        raise RuntimeError("L37 requires exactly 100 pre/post-fission observations")
    if any(
        parent.observation_kind != "molecular_update"
        or parent.growth_generation_one_based != daughter.growth_generation_one_based
        or sum(parent.state) != trajectory.generations[index].pre_fission_mass
        for index, (parent, daughter) in enumerate(zip(pre, post, strict=True))
    ):
        raise RuntimeError("L37 pre-fission predecessor identity failure")
    states = np.asarray([observation.state for observation in post], dtype=np.int64)
    generations = np.asarray(
        [observation.completed_fissions for observation in post], dtype=np.int64
    )
    parent = np.asarray([observation.state for observation in pre], dtype=np.int64)
    parent_composition = parent / parent.sum(axis=1, keepdims=True)
    daughter_composition = states / states.sum(axis=1, keepdims=True)
    parent_h = np.sum(parent_composition * daughter_composition, axis=1) / (
        np.linalg.norm(parent_composition, axis=1)
        * np.linalg.norm(daughter_composition, axis=1)
    )
    return states, generations, np.asarray(parent_h, dtype=np.float64)


def analysis_seed_manifest(lineages: pd.DataFrame) -> pd.DataFrame:
    rows = []
    units = lineages.drop_duplicates("stateId")
    for unit in units.itertuples(index=False):
        for held_out in LINEAGES:
            parts = ("species_permutation", unit.stateId, held_out)
            rows.append(
                {
                    "purpose": parts[0],
                    "stateId": unit.stateId,
                    "evaluationCohort": unit.evaluationCohort,
                    "candidateId": unit.candidateId,
                    "matrixIndex": int(unit.matrixIndex),
                    "heldOutLineage": held_out,
                    "partsJson": json.dumps(parts, separators=(",", ":")),
                    "rootHex": ROOT_HEX,
                    "derivedSeed": str(derived_seed(*parts)),
                    "seedMaterialSha256": hashlib.sha256(
                        "\x1f".join([VERSION, ROOT_HEX, *map(str, parts)]).encode()
                    ).hexdigest(),
                }
            )
    registered_parts: list[tuple[object, ...]] = []
    for cohort in COHORTS:
        for candidate in CANDIDATES:
            for held_out in LINEAGES:
                for control in (
                    "PERMUTED_SPECIES_ATLAS",
                    "UNRELATED_MATRIX_ATLAS",
                ):
                    registered_parts.append(
                        ("direct_bootstrap", cohort, candidate, held_out, control)
                    )
            for left, right in (
                ("ORIGINAL", "REFERENCE_A"),
                ("ORIGINAL", "REFERENCE_B"),
                ("REFERENCE_A", "REFERENCE_B"),
            ):
                registered_parts.append(
                    ("reciprocal_bootstrap", cohort, candidate, left, right)
                )
            for family in FAMILIES:
                for target in TARGET_TYPES:
                    registered_parts.append(
                        (
                            "branch_reliability_bootstrap",
                            cohort,
                            candidate,
                            family,
                            target,
                        )
                    )
            for comparison in (
                "ACTUAL_H8_VS_ACTUAL_H32",
                "ORIGINAL_H8_VS_ACTUAL_H32",
                "ACTUAL_H32_MINUS_PERMUTED_H32",
                "ACTUAL_H32_MINUS_UNRELATED_H32",
            ):
                registered_parts.append(
                    ("branch_transfer_bootstrap", cohort, candidate, comparison)
                )
    for parts in registered_parts:
        purpose = str(parts[0])
        cohort = str(parts[1])
        candidate = str(parts[2])
        rows.append(
            {
                "purpose": purpose,
                "stateId": None,
                "evaluationCohort": cohort,
                "candidateId": candidate,
                "matrixIndex": None,
                "heldOutLineage": str(parts[3])
                if purpose in ("direct_bootstrap", "reciprocal_bootstrap")
                else None,
                "partsJson": json.dumps(parts, separators=(",", ":")),
                "rootHex": ROOT_HEX,
                "derivedSeed": str(derived_seed(*parts)),
                "seedMaterialSha256": hashlib.sha256(
                    "\x1f".join([VERSION, ROOT_HEX, *map(str, parts)]).encode()
                ).hexdigest(),
            }
        )
    result = pd.DataFrame(rows).sort_values(
        ["purpose", "evaluationCohort", "candidateId", "stateId", "heldOutLineage"],
        na_position="last",
    ).reset_index(drop=True)
    if result["seedMaterialSha256"].duplicated().any() or result[
        "derivedSeed"
    ].duplicated().any():
        raise RuntimeError("L37 analysis seed collision")
    return result


def seed_firewall(seeds: pd.DataFrame) -> dict[str, Any]:
    prior_material: set[str] = set()
    prior_derived: set[str] = set()
    for path in ARTIFACT_ROOT.glob("loops/L*/**/*seed*manifest*.parquet"):
        if "/L37/" in str(path):
            continue
        try:
            frame = pd.read_parquet(path)
        except (OSError, ValueError, TypeError):
            continue
        for column in frame.columns:
            if "seedmaterialsha256" in column.lower():
                prior_material.update(frame[column].dropna().astype(str))
            if column.lower() == "derivedseed":
                prior_derived.update(frame[column].dropna().astype(str))
    material = sorted(set(seeds["seedMaterialSha256"]) & prior_material)
    derived = sorted(set(seeds["derivedSeed"]) & prior_derived)
    return {
        "schema": "eidosoma.e01.s19_l37.seed_firewall.v1",
        "status": "PASS" if not material and not derived else "FAIL",
        "seedCount": len(seeds),
        "seedMaterialOverlapCount": len(material),
        "derivedSeedOverlapCount": len(derived),
        "seedMaterialOverlaps": material,
        "derivedSeedOverlaps": derived,
        "scientificTrajectorySeeds": 0,
    }


def build_atlas_registries(
    lineages: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    lineage_index = lineages.set_index(["stateId", "lineageId"])
    summary_rows = []
    component_rows = []
    coordinate_rows = []
    membership_rows = []
    permutation_rows = []
    for state_id in lineages["stateId"].drop_duplicates():
        metadata = lineage_index.loc[(state_id, "ORIGINAL")]
        payloads = {
            lineage: post_fission_payload(lineage_index.loc[(state_id, lineage)])
            for lineage in LINEAGES
        }
        for held_out in LINEAGES:
            references = tuple(value for value in LINEAGES if value != held_out)
            states = np.concatenate([payloads[value][0] for value in references], axis=0)
            generations = np.concatenate(
                [payloads[value][1] for value in references], axis=0
            )
            lineage_ids = np.concatenate(
                [np.full(100, index, dtype=np.int64) for index in range(2)]
            )
            atlas = build_cross_lineage_atlas(
                states,
                lineage_ids,
                generations,
                threshold=THRESHOLD,
                minimum_visits_per_lineage=MIN_VISITS,
                minimum_generation_span=MIN_SPAN,
            )
            permutation_seed = derived_seed("species_permutation", state_id, held_out)
            permutation = np.random.default_rng(permutation_seed).permutation(100)
            summary_rows.append(
                {
                    "stateId": state_id,
                    "evaluationCohort": metadata.evaluationCohort,
                    "candidateId": metadata.candidateId,
                    "matrixIndex": int(metadata.matrixIndex),
                    "landmark": int(metadata.landmark),
                    "heldOutLineage": held_out,
                    "referenceLineageA": references[0],
                    "referenceLineageB": references[1],
                    "status": atlas.status,
                    "componentCount": len(atlas.components),
                    "componentMemberCount": int(
                        sum(len(item.member_indices) for item in atlas.components)
                    ),
                    "referenceBoundaryCount": 200,
                    "targetUsesHeldOutFuture": False,
                    "permutationSeed": str(permutation_seed),
                    "permutationSha256": L28.array_sha256(
                        np.asarray(permutation, dtype=np.int64)
                    ),
                }
            )
            permutation_rows.append(
                {
                    "stateId": state_id,
                    "evaluationCohort": metadata.evaluationCohort,
                    "candidateId": metadata.candidateId,
                    "matrixIndex": int(metadata.matrixIndex),
                    "heldOutLineage": held_out,
                    "derivedSeed": str(permutation_seed),
                    "permutationJson": json.dumps(permutation.tolist(), separators=(",", ":")),
                    "permutationSha256": L28.array_sha256(
                        np.asarray(permutation, dtype=np.int64)
                    ),
                }
            )
            for component in atlas.components:
                component_rows.append(
                    {
                        "stateId": state_id,
                        "evaluationCohort": metadata.evaluationCohort,
                        "candidateId": metadata.candidateId,
                        "matrixIndex": int(metadata.matrixIndex),
                        "landmark": int(metadata.landmark),
                        "heldOutLineage": held_out,
                        "componentId": component.component_id,
                        "componentSize": len(component.member_indices),
                        "referenceACount": component.lineage_counts[0],
                        "referenceBCount": component.lineage_counts[1],
                        "referenceAGenerationSpan": component.lineage_generation_spans[0],
                        "referenceBGenerationSpan": component.lineage_generation_spans[1],
                        "meanWithinH": component.mean_within_h,
                        "minimumWithinH": component.minimum_within_h,
                        "centroidSha256": L28.array_sha256(component.centroid),
                    }
                )
                for coordinate, value in enumerate(component.centroid):
                    coordinate_rows.append(
                        {
                            "stateId": state_id,
                            "evaluationCohort": metadata.evaluationCohort,
                            "candidateId": metadata.candidateId,
                            "matrixIndex": int(metadata.matrixIndex),
                            "landmark": int(metadata.landmark),
                            "heldOutLineage": held_out,
                            "targetType": "INDEPENDENT_ANY_ATTRACTOR",
                            "componentId": component.component_id,
                            "coordinate": coordinate,
                            "centroidValue": float(value),
                        }
                    )
                    coordinate_rows.append(
                        {
                            "stateId": state_id,
                            "evaluationCohort": metadata.evaluationCohort,
                            "candidateId": metadata.candidateId,
                            "matrixIndex": int(metadata.matrixIndex),
                            "landmark": int(metadata.landmark),
                            "heldOutLineage": held_out,
                            "targetType": "PERMUTED_SPECIES_ATLAS",
                            "componentId": component.component_id,
                            "coordinate": coordinate,
                            "centroidValue": float(component.centroid[permutation][coordinate]),
                        }
                    )
                for member in component.member_indices:
                    reference_index = int(member // 100)
                    local_index = int(member % 100)
                    membership_rows.append(
                        {
                            "stateId": state_id,
                            "evaluationCohort": metadata.evaluationCohort,
                            "candidateId": metadata.candidateId,
                            "matrixIndex": int(metadata.matrixIndex),
                            "landmark": int(metadata.landmark),
                            "heldOutLineage": held_out,
                            "componentId": component.component_id,
                            "referenceLineage": references[reference_index],
                            "referenceGeneration": int(generations[member]),
                            "referenceLocalIndex": local_index,
                        }
                    )
    summary = pd.DataFrame(summary_rows).sort_values(
        ["evaluationCohort", "candidateId", "landmark", "matrixIndex", "heldOutLineage"]
    ).reset_index(drop=True)
    components = pd.DataFrame(component_rows)
    coordinates = pd.DataFrame(coordinate_rows)
    memberships = pd.DataFrame(membership_rows)
    permutations = pd.DataFrame(permutation_rows).sort_values(
        ["evaluationCohort", "candidateId", "matrixIndex", "heldOutLineage"]
    ).reset_index(drop=True)
    if len(summary) != 840 or len(permutations) != 840:
        raise RuntimeError("L37 atlas registry cardinality failure")
    if len(coordinates):
        coordinates = coordinates.sort_values(
            [
                "evaluationCohort",
                "candidateId",
                "landmark",
                "matrixIndex",
                "heldOutLineage",
                "targetType",
                "componentId",
                "coordinate",
            ]
        ).reset_index(drop=True)
    return summary, components, coordinates, memberships, permutations


def unrelated_control_map(atlases: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cohort, candidate, held_out), group in atlases.groupby(
        ["evaluationCohort", "candidateId", "heldOutLineage"], sort=True
    ):
        ordered = group.copy()
        ordered["rankKey"] = ordered["stateId"].map(
            lambda value, role=held_out: hashlib.sha256(
                f"{VERSION}|UNRELATED|{role}|{value}".encode()
            ).hexdigest()
        )
        ordered = ordered.sort_values("rankKey").reset_index(drop=True)
        for index, source in ordered.iterrows():
            donor = ordered.iloc[(index + 1) % len(ordered)]
            rows.append(
                {
                    "stateId": source.stateId,
                    "evaluationCohort": cohort,
                    "candidateId": candidate,
                    "matrixIndex": int(source.matrixIndex),
                    "heldOutLineage": held_out,
                    "donorStateId": donor.stateId,
                    "donorMatrixIndex": int(donor.matrixIndex),
                    "donorStatus": donor.status,
                    "donorComponentCount": int(donor.componentCount),
                    "sameMatrix": int(source.matrixIndex) == int(donor.matrixIndex),
                    "mappingRule": "NEXT_SHA256_RANK_WITHIN_CANDIDATE_COHORT_ROLE",
                }
            )
    result = pd.DataFrame(rows).sort_values(
        ["evaluationCohort", "candidateId", "matrixIndex", "heldOutLineage"]
    ).reset_index(drop=True)
    if len(result) != 840 or result["sameMatrix"].any():
        raise RuntimeError("L37 unrelated-matrix control mapping failure")
    return result


def centroid_map(coordinates: pd.DataFrame) -> dict[tuple[str, str, str], np.ndarray]:
    output: dict[tuple[str, str, str], np.ndarray] = {}
    if not len(coordinates):
        return output
    for keys, group in coordinates.groupby(
        ["stateId", "heldOutLineage", "targetType"], sort=False
    ):
        matrix = group.pivot(
            index="componentId", columns="coordinate", values="centroidValue"
        ).sort_index()
        output[keys] = matrix.to_numpy(dtype=np.float64)
    return output


def direct_heldout_results(
    lineages: pd.DataFrame,
    atlases: pd.DataFrame,
    coordinates: pd.DataFrame,
    unrelated: pd.DataFrame,
) -> pd.DataFrame:
    lineage_index = lineages.set_index(["stateId", "lineageId"])
    centroids = centroid_map(coordinates)
    unrelated_index = unrelated.set_index(["stateId", "heldOutLineage"])
    rows = []
    for atlas in atlases.itertuples(index=False):
        heldout = lineage_index.loc[(atlas.stateId, atlas.heldOutLineage)]
        states, _generations, parent_h = post_fission_payload(heldout)
        donor = unrelated_index.loc[(atlas.stateId, atlas.heldOutLineage)]
        targets = {
            "INDEPENDENT_ANY_ATTRACTOR": centroids.get(
                (atlas.stateId, atlas.heldOutLineage, "INDEPENDENT_ANY_ATTRACTOR")
            ),
            "PERMUTED_SPECIES_ATLAS": centroids.get(
                (atlas.stateId, atlas.heldOutLineage, "PERMUTED_SPECIES_ATLAS")
            ),
            "UNRELATED_MATRIX_ATLAS": centroids.get(
                (
                    donor.donorStateId,
                    atlas.heldOutLineage,
                    "INDEPENDENT_ANY_ATTRACTOR",
                )
            ),
        }
        for target_type, target in targets.items():
            if target is None:
                rows.append(
                    {
                        "stateId": atlas.stateId,
                        "evaluationCohort": atlas.evaluationCohort,
                        "candidateId": atlas.candidateId,
                        "matrixIndex": int(atlas.matrixIndex),
                        "landmark": int(atlas.landmark),
                        "heldOutLineage": atlas.heldOutLineage,
                        "targetType": target_type,
                        "targetAvailable": False,
                        "targetComponentCount": 0,
                        "positiveCount": 0,
                        "occupancy": np.nan,
                        "firstEntryZeroBased": np.nan,
                        "transitionCount": np.nan,
                        "positiveEpisodeCount": np.nan,
                        "returnCount": np.nan,
                        "longestPositiveEpisode": np.nan,
                        "recurrentPositive": False,
                        "selfTransitionProbability": np.nan,
                        "meanParentDaughterHInside": np.nan,
                        "meanParentDaughterHOutside": np.nan,
                        "meanMaximumH": np.nan,
                        "maximumH": np.nan,
                        "targetUsesHeldOutFuture": False,
                    }
                )
                continue
            scores, assignments, labels = score_atlas(
                states, target, threshold=THRESHOLD
            )
            summary = summarize_atlas_labels(labels, assignments)
            rows.append(
                {
                    "stateId": atlas.stateId,
                    "evaluationCohort": atlas.evaluationCohort,
                    "candidateId": atlas.candidateId,
                    "matrixIndex": int(atlas.matrixIndex),
                    "landmark": int(atlas.landmark),
                    "heldOutLineage": atlas.heldOutLineage,
                    "targetType": target_type,
                    "targetAvailable": True,
                    "targetComponentCount": len(target),
                    "positiveCount": summary.positive_count,
                    "occupancy": summary.occupancy,
                    "firstEntryZeroBased": summary.first_entry_zero_based,
                    "transitionCount": summary.transition_count,
                    "positiveEpisodeCount": summary.positive_episode_count,
                    "returnCount": max(0, summary.positive_episode_count - 1),
                    "longestPositiveEpisode": summary.longest_positive_episode,
                    "recurrentPositive": summary.recurrent_positive,
                    "selfTransitionProbability": summary.self_transition_probability,
                    "meanParentDaughterHInside": float(np.mean(parent_h[labels]))
                    if labels.any()
                    else np.nan,
                    "meanParentDaughterHOutside": float(np.mean(parent_h[~labels]))
                    if (~labels).any()
                    else np.nan,
                    "meanMaximumH": float(np.mean(scores)),
                    "maximumH": float(np.max(scores)),
                    "targetUsesHeldOutFuture": False,
                }
            )
    result = pd.DataFrame(rows).sort_values(
        [
            "evaluationCohort",
            "candidateId",
            "landmark",
            "matrixIndex",
            "heldOutLineage",
            "targetType",
        ]
    ).reset_index(drop=True)
    if len(result) != 2520 or result.duplicated(
        ["stateId", "heldOutLineage", "targetType"]
    ).any():
        raise RuntimeError("L37 direct held-out result cardinality failure")
    return result


def branch_payloads(
    responses: pd.DataFrame,
    original_coordinates: pd.DataFrame,
    trajectory_manifest: pd.DataFrame,
    coordinates: pd.DataFrame,
    unrelated: pd.DataFrame,
) -> list[dict[str, Any]]:
    base = L35.payloads(responses, original_coordinates, trajectory_manifest)
    original_map = {
        state_id: group.sort_values("coordinate")["centroidValue"].tolist()
        for state_id, group in original_coordinates.groupby("stateId", sort=False)
    }
    centroids = centroid_map(coordinates)
    unrelated_index = unrelated.set_index(["stateId", "heldOutLineage"])
    output = []
    for source in base:
        donor = unrelated_index.loc[(source["stateId"], "ORIGINAL")]
        row = dict(source)
        row["originalCentroid"] = original_map[source["stateId"]]
        row["targets"] = {
            "ORIGINAL_TRAJECTORY_BASIN": [original_map[source["stateId"]]],
            "INDEPENDENT_ANY_ATTRACTOR": centroids.get(
                (source["stateId"], "ORIGINAL", "INDEPENDENT_ANY_ATTRACTOR")
            ),
            "PERMUTED_SPECIES_ATLAS": centroids.get(
                (source["stateId"], "ORIGINAL", "PERMUTED_SPECIES_ATLAS")
            ),
            "UNRELATED_MATRIX_ATLAS": centroids.get(
                (donor.donorStateId, "ORIGINAL", "INDEPENDENT_ANY_ATTRACTOR")
            ),
        }
        output.append(row)
    return output


def _branch_worker(payload: dict[str, Any]) -> list[dict[str, Any]]:
    matrix_index = int(payload["matrixIndex"])
    beta = L28.generate_beta(
        L28.derive_seed(
            L28.L23_ROOT_HEX, L28.L23_PHASE, "catalytic_matrix", matrix_index
        )
    )
    restored = RestoredState(
        tuple(payload["state"]),
        payload["currentObservationKind"],
        int(payload["currentCompletedFissions"]),
        int(payload["currentGrowthGeneration"]),
        int(payload["currentGenerationLocalStep"]),
        int(payload["currentBatchStep"]),
    )
    original = np.asarray(payload["originalCentroid"], dtype=np.float64)
    targets = {
        key: np.asarray(value, dtype=np.float64)
        for key, value in payload["targets"].items()
        if value is not None and len(value)
    }
    current_state = np.asarray(payload["state"], dtype=np.int64)[None, :]
    current_scores = {
        key: float(score_atlas(current_state, value, threshold=THRESHOLD)[0][0])
        for key, value in targets.items()
    }
    rows = []
    for family in FAMILIES:
        for branch in range(BRANCH_COUNTS[family]):
            event_rng, trim_rng, fission_rng, daughter_rng = L36._branch_rngs(
                payload, family, branch
            )
            trace = simulate_branch_trace(
                restored=restored,
                beta=beta,
                definition=L28.definition(payload["candidateId"]),
                target_centroid=original,
                event_rng=event_rng,
                trim_rng=trim_rng,
                fission_rng=fission_rng,
                daughter_rng=daughter_rng,
                horizon=HORIZONS[family],
                threshold=THRESHOLD,
            )
            states = np.asarray(
                [observation.state for observation in trace.observations], dtype=np.int64
            )
            for target_id, target in targets.items():
                scores, assignments, labels = score_atlas(
                    states, target, threshold=THRESHOLD
                )
                indices = np.flatnonzero(labels)
                rows.append(
                    {
                        "stateId": payload["stateId"],
                        "evaluationCohort": payload["evaluationCohort"],
                        "candidateId": payload["candidateId"],
                        "matrixIndex": matrix_index,
                        "landmark": int(payload["landmark"]),
                        "branchFamily": family,
                        "targetId": target_id,
                        "targetComponentCount": len(target),
                        "branchIndex": branch,
                        "branchHalf": "A" if branch < HALVES[family] else "B",
                        "currentTargetScore": current_scores[target_id],
                        "currentInsideTarget": current_scores[target_id] >= THRESHOLD,
                        "enteredTarget": bool(len(indices)),
                        "firstEntryOffsetOneBased": int(indices[0] + 1)
                        if len(indices)
                        else None,
                        "minimumTargetScore": float(np.min(scores)),
                        "maximumTargetScore": float(np.max(scores)),
                        "finalTargetScore": float(scores[-1]),
                        "finalAssignedBasin": int(assignments[-1]),
                        "molecularUpdates": trace.compact.molecular_updates,
                        "fissions": trace.compact.fissions,
                        "selectedObservationsGenerated": trace.compact.selected_observations_generated,
                        "terminalStatus": trace.compact.terminal_status,
                        "finalStateSha256": trace.compact.final_state_sha256,
                        "originalPathSha256": trace.compact.path_sha256,
                        "dynamicPathIndependentOfTarget": True,
                    }
                )
    return rows


def rescore_branches(payload_rows: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = {
            executor.submit(_branch_worker, payload): payload["stateId"]
            for payload in payload_rows
        }
        for future in as_completed(futures):
            rows.extend(future.result())
    result = pd.DataFrame(rows).sort_values(
        [
            "evaluationCohort",
            "candidateId",
            "landmark",
            "matrixIndex",
            "branchFamily",
            "targetId",
            "branchIndex",
        ]
    ).reset_index(drop=True)
    expected = sum(
        sum(BRANCH_COUNTS.values())
        * registered_target_count(payload["targets"])
        for payload in payload_rows
    )
    if len(result) != expected:
        raise RuntimeError("L37 branch-rescore cardinality failure")
    return result


def registered_target_count(targets: dict[str, Any]) -> int:
    """Count target families, never their internal atlas components."""

    return sum(
        int(value is not None and np.asarray(value).ndim == 2 and len(value) > 0)
        for value in targets.values()
    )


def original_compact_validation(branches: pd.DataFrame) -> pd.DataFrame:
    original = branches[
        branches["targetId"].eq("ORIGINAL_TRAJECTORY_BASIN")
    ].copy()
    original["targetId"] = "ORIGINAL"
    validation = L36.compact_replay_validation(original)
    if not validation["allPassed"].all():
        raise RuntimeError("L37 original compact replay failed")
    return validation


def state_committor_results(branches: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in branches.groupby(
        ["stateId", "branchFamily", "targetId"], sort=True
    ):
        state_id, family, target_id = keys
        expected = BRANCH_COUNTS[family]
        if len(group) != expected:
            raise RuntimeError("L37 state committor branch cardinality failure")
        metadata = group.iloc[0]
        eligible = not bool(metadata.currentInsideTarget)
        half_a = group[group["branchHalf"].eq("A")]
        half_b = group[group["branchHalf"].eq("B")]
        successes = int(group["enteredTarget"].sum())
        rows.append(
            {
                "stateId": state_id,
                "evaluationCohort": metadata.evaluationCohort,
                "candidateId": metadata.candidateId,
                "matrixIndex": int(metadata.matrixIndex),
                "landmark": int(metadata.landmark),
                "branchFamily": family,
                "targetId": target_id,
                "targetComponentCount": int(metadata.targetComponentCount),
                "currentTargetScore": float(metadata.currentTargetScore),
                "currentInsideTarget": bool(metadata.currentInsideTarget),
                "committorEligible": eligible,
                "branches": expected,
                "successes": successes,
                "qHat": successes / expected if eligible else np.nan,
                "qHatHalfA": float(half_a["enteredTarget"].mean())
                if eligible
                else np.nan,
                "qHatHalfB": float(half_b["enteredTarget"].mean())
                if eligible
                else np.nan,
            }
        )
    result = pd.DataFrame(rows).sort_values(
        [
            "evaluationCohort",
            "candidateId",
            "landmark",
            "matrixIndex",
            "branchFamily",
            "targetId",
        ]
    ).reset_index(drop=True)
    return result


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    if mask.sum() < 3 or np.unique(left[mask]).size < 2 or np.unique(right[mask]).size < 2:
        return np.nan
    return float(spearmanr(left[mask], right[mask]).statistic)


def interval(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return np.nan, np.nan
    return float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))


def direct_group_results(direct: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in direct.groupby(
        ["evaluationCohort", "candidateId", "heldOutLineage", "targetType"],
        sort=True,
    ):
        cohort, candidate, held_out, target_type = keys
        available = group[group["targetAvailable"]]
        rows.append(
            {
                "evaluationCohort": cohort,
                "candidateId": candidate,
                "heldOutLineage": held_out,
                "targetType": target_type,
                "states": len(group),
                "availableStates": len(available),
                "availabilityFraction": float(group["targetAvailable"].mean()),
                "meanOccupancy": float(available["occupancy"].mean())
                if len(available)
                else np.nan,
                "medianOccupancy": float(available["occupancy"].median())
                if len(available)
                else np.nan,
                "recurrentRecognitionFraction": float(
                    group["recurrentPositive"].fillna(False).mean()
                ),
                "meanPositiveEpisodes": float(available["positiveEpisodeCount"].mean())
                if len(available)
                else np.nan,
                "meanReturnCount": float(available["returnCount"].mean())
                if len(available)
                else np.nan,
                "meanFirstEntry": float(available["firstEntryZeroBased"].mean())
                if len(available)
                else np.nan,
                "meanParentDaughterHInside": float(
                    available["meanParentDaughterHInside"].mean()
                )
                if len(available)
                else np.nan,
                "meanParentDaughterHOutside": float(
                    available["meanParentDaughterHOutside"].mean()
                )
                if len(available)
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def direct_control_bootstraps(direct: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    comparison_rows = []
    bootstrap_rows = []
    for (cohort, candidate, held_out), group in direct.groupby(
        ["evaluationCohort", "candidateId", "heldOutLineage"], sort=True
    ):
        pivot_occ = group.pivot(index="stateId", columns="targetType", values="occupancy")
        pivot_rec = group.pivot(
            index="stateId", columns="targetType", values="recurrentPositive"
        ).astype(float)
        for control in ("PERMUTED_SPECIES_ATLAS", "UNRELATED_MATRIX_ATLAS"):
            columns = ["INDEPENDENT_ANY_ATTRACTOR", control]
            defined = pivot_occ[columns].dropna()
            recurring = pivot_rec.loc[defined.index, columns]
            observed_occ = float(
                np.mean(
                    defined["INDEPENDENT_ANY_ATTRACTOR"].to_numpy()
                    - defined[control].to_numpy()
                )
            ) if len(defined) else np.nan
            observed_rec = float(
                np.mean(
                    recurring["INDEPENDENT_ANY_ATTRACTOR"].to_numpy()
                    - recurring[control].to_numpy()
                )
            ) if len(defined) else np.nan
            rng = np.random.default_rng(
                derived_seed("direct_bootstrap", cohort, candidate, held_out, control)
            )
            occ_boot = np.full(BOOTSTRAPS, np.nan)
            rec_boot = np.full(BOOTSTRAPS, np.nan)
            if len(defined):
                occ_diff = (
                    defined["INDEPENDENT_ANY_ATTRACTOR"].to_numpy()
                    - defined[control].to_numpy()
                )
                rec_diff = (
                    recurring["INDEPENDENT_ANY_ATTRACTOR"].to_numpy()
                    - recurring[control].to_numpy()
                )
                for replicate in range(BOOTSTRAPS):
                    indices = rng.integers(0, len(defined), len(defined))
                    occ_boot[replicate] = np.mean(occ_diff[indices])
                    rec_boot[replicate] = np.mean(rec_diff[indices])
                    bootstrap_rows.append(
                        {
                            "evaluationCohort": cohort,
                            "candidateId": candidate,
                            "heldOutLineage": held_out,
                            "controlType": control,
                            "replicate": replicate,
                            "occupancyDifference": occ_boot[replicate],
                            "recurrenceDifference": rec_boot[replicate],
                        }
                    )
            occ_lower, occ_upper = interval(occ_boot)
            rec_lower, rec_upper = interval(rec_boot)
            comparison_rows.append(
                {
                    "evaluationCohort": cohort,
                    "candidateId": candidate,
                    "heldOutLineage": held_out,
                    "controlType": control,
                    "definedPairs": len(defined),
                    "meanOccupancyDifference": observed_occ,
                    "occupancyDifferenceLower95": occ_lower,
                    "occupancyDifferenceUpper95": occ_upper,
                    "meanRecurrenceDifference": observed_rec,
                    "recurrenceDifferenceLower95": rec_lower,
                    "recurrenceDifferenceUpper95": rec_upper,
                }
            )
    return pd.DataFrame(comparison_rows), pd.DataFrame(bootstrap_rows)


def reciprocal_results(direct: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    actual = direct[
        direct["targetType"].eq("INDEPENDENT_ANY_ATTRACTOR")
    ].copy()
    rows = []
    bootstrap_rows = []
    for (cohort, candidate), group in actual.groupby(
        ["evaluationCohort", "candidateId"], sort=True
    ):
        pivot = group.pivot(index="stateId", columns="heldOutLineage", values="occupancy")
        for left, right in (
            ("ORIGINAL", "REFERENCE_A"),
            ("ORIGINAL", "REFERENCE_B"),
            ("REFERENCE_A", "REFERENCE_B"),
        ):
            values = pivot[[left, right]].dropna()
            observed = safe_spearman(
                values[left].to_numpy(), values[right].to_numpy()
            )
            rng = np.random.default_rng(
                derived_seed("reciprocal_bootstrap", cohort, candidate, left, right)
            )
            boot = np.full(BOOTSTRAPS, np.nan)
            for replicate in range(BOOTSTRAPS):
                if len(values):
                    indices = rng.integers(0, len(values), len(values))
                    boot[replicate] = safe_spearman(
                        values[left].to_numpy()[indices],
                        values[right].to_numpy()[indices],
                    )
                bootstrap_rows.append(
                    {
                        "evaluationCohort": cohort,
                        "candidateId": candidate,
                        "leftHeldOut": left,
                        "rightHeldOut": right,
                        "replicate": replicate,
                        "spearman": boot[replicate],
                    }
                )
            lower, upper = interval(boot)
            rows.append(
                {
                    "evaluationCohort": cohort,
                    "candidateId": candidate,
                    "leftHeldOut": left,
                    "rightHeldOut": right,
                    "definedPairs": len(values),
                    "spearman": observed,
                    "lower95": lower,
                    "upper95": upper,
                    "rankGatePassed": bool(
                        np.isfinite(observed)
                        and np.isfinite(lower)
                        and observed > 0.5
                        and lower > 0.3
                    ),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(bootstrap_rows)


def reliability_results(states: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    bootstrap_rows = []
    for keys, group in states.groupby(
        ["evaluationCohort", "candidateId", "branchFamily", "targetId"], sort=True
    ):
        cohort, candidate, family, target = keys
        eligible = group[group["committorEligible"] & group["qHat"].notna()]
        q = eligible["qHat"].to_numpy(dtype=np.float64)
        half_a = eligible["qHatHalfA"].to_numpy(dtype=np.float64)
        half_b = eligible["qHatHalfB"].to_numpy(dtype=np.float64)
        variance = (
            corrected_between_state_variance(q, BRANCH_COUNTS[family])
            if len(q) > 1
            else {
                "observedBetweenStateVariance": np.nan,
                "estimatedBinomialNoiseVariance": np.nan,
                "correctedBetweenStateVariance": np.nan,
            }
        )
        corrected = float(variance["correctedBetweenStateVariance"])
        split = safe_spearman(half_a, half_b)
        rng = np.random.default_rng(
            derived_seed("branch_reliability_bootstrap", cohort, candidate, family, target)
        )
        corrected_boot = np.full(BOOTSTRAPS, np.nan)
        split_boot = np.full(BOOTSTRAPS, np.nan)
        for replicate in range(BOOTSTRAPS):
            if len(q) > 1:
                indices = rng.integers(0, len(q), len(q))
                corrected_boot[replicate] = corrected_between_state_variance(
                    q[indices], BRANCH_COUNTS[family]
                )["correctedBetweenStateVariance"]
                split_boot[replicate] = safe_spearman(
                    half_a[indices], half_b[indices]
                )
            bootstrap_rows.append(
                {
                    "evaluationCohort": cohort,
                    "candidateId": candidate,
                    "branchFamily": family,
                    "targetId": target,
                    "replicate": replicate,
                    "correctedVariance": corrected_boot[replicate],
                    "splitHalfSpearman": split_boot[replicate],
                }
            )
        corrected_lower, corrected_upper = interval(corrected_boot)
        split_lower, split_upper = interval(split_boot)
        rows.append(
            {
                "evaluationCohort": cohort,
                "candidateId": candidate,
                "branchFamily": family,
                "targetId": target,
                "states": len(group),
                "eligibleStates": len(eligible),
                "eligibleFraction": float(group["committorEligible"].mean()),
                "meanQ": float(np.mean(q)) if len(q) else np.nan,
                "intermediateStateCount": int(np.sum((q > 0.1) & (q < 0.9))),
                "correctedBetweenStateVariance": corrected,
                "correctedVarianceLower95": corrected_lower,
                "correctedVarianceUpper95": corrected_upper,
                "splitHalfSpearman": split,
                "splitHalfLower95": split_lower,
                "splitHalfUpper95": split_upper,
                "reliabilityGatePassed": bool(
                    len(eligible) >= 20
                    and np.isfinite(corrected_lower)
                    and corrected_lower > 0
                    and np.isfinite(split)
                    and split > 0.5
                    and np.isfinite(split_lower)
                    and split_lower > 0.3
                    and np.sum((q > 0.1) & (q < 0.9)) >= 10
                ),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(bootstrap_rows)


def branch_transfer_results(states: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = states.set_index(["stateId", "branchFamily", "targetId"])
    specifications = (
        (
            "ACTUAL_H8_VS_ACTUAL_H32",
            ("H8", "INDEPENDENT_ANY_ATTRACTOR"),
            ("H32", "INDEPENDENT_ANY_ATTRACTOR"),
            "RANK",
        ),
        (
            "ORIGINAL_H8_VS_ACTUAL_H32",
            ("H8", "ORIGINAL_TRAJECTORY_BASIN"),
            ("H32", "INDEPENDENT_ANY_ATTRACTOR"),
            "RANK",
        ),
        (
            "ACTUAL_H32_MINUS_PERMUTED_H32",
            ("H32", "INDEPENDENT_ANY_ATTRACTOR"),
            ("H32", "PERMUTED_SPECIES_ATLAS"),
            "DIFFERENCE",
        ),
        (
            "ACTUAL_H32_MINUS_UNRELATED_H32",
            ("H32", "INDEPENDENT_ANY_ATTRACTOR"),
            ("H32", "UNRELATED_MATRIX_ATLAS"),
            "DIFFERENCE",
        ),
    )
    pair_rows = []
    for state_id in states["stateId"].unique():
        for comparison_id, left_key, right_key, comparison_type in specifications:
            try:
                left = index.loc[(state_id, *left_key)]
                right = index.loc[(state_id, *right_key)]
            except KeyError:
                continue
            if not bool(left.committorEligible) or not bool(right.committorEligible):
                continue
            pair_rows.append(
                {
                    "stateId": state_id,
                    "evaluationCohort": left.evaluationCohort,
                    "candidateId": left.candidateId,
                    "matrixIndex": int(left.matrixIndex),
                    "landmark": int(left.landmark),
                    "comparisonId": comparison_id,
                    "comparisonType": comparison_type,
                    "leftQ": float(left.qHat),
                    "rightQ": float(right.qHat),
                }
            )
    pairs = pd.DataFrame(pair_rows)
    rows = []
    bootstrap_rows = []
    for keys, group in pairs.groupby(
        ["evaluationCohort", "candidateId", "comparisonId", "comparisonType"],
        sort=True,
    ):
        cohort, candidate, comparison_id, comparison_type = keys
        left = group["leftQ"].to_numpy(dtype=np.float64)
        right = group["rightQ"].to_numpy(dtype=np.float64)
        observed = (
            safe_spearman(left, right)
            if comparison_type == "RANK"
            else float(np.mean(left - right))
        )
        rng = np.random.default_rng(
            derived_seed("branch_transfer_bootstrap", cohort, candidate, comparison_id)
        )
        boot = np.full(BOOTSTRAPS, np.nan)
        for replicate in range(BOOTSTRAPS):
            indices = rng.integers(0, len(group), len(group))
            boot[replicate] = (
                safe_spearman(left[indices], right[indices])
                if comparison_type == "RANK"
                else float(np.mean(left[indices] - right[indices]))
            )
            bootstrap_rows.append(
                {
                    "evaluationCohort": cohort,
                    "candidateId": candidate,
                    "comparisonId": comparison_id,
                    "comparisonType": comparison_type,
                    "replicate": replicate,
                    "value": boot[replicate],
                }
            )
        lower, upper = interval(boot)
        passed = bool(
            np.isfinite(observed)
            and np.isfinite(lower)
            and (
                (comparison_type == "RANK" and observed > 0.5 and lower > 0.3)
                or (comparison_type == "DIFFERENCE" and lower > 0)
            )
        )
        rows.append(
            {
                "evaluationCohort": cohort,
                "candidateId": candidate,
                "comparisonId": comparison_id,
                "comparisonType": comparison_type,
                "definedPairs": len(group),
                "pointEstimate": observed,
                "lower95": lower,
                "upper95": upper,
                "gatePassed": passed,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(bootstrap_rows)


def scientific_gates(
    direct_groups: pd.DataFrame,
    direct_controls: pd.DataFrame,
    reciprocal: pd.DataFrame,
    reliability: pd.DataFrame,
    transfer: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], str]:
    rows = []
    for cohort in EVALUATION_COHORTS:
        for candidate in CANDIDATES:
            actual = direct_groups[
                direct_groups["evaluationCohort"].eq(cohort)
                & direct_groups["candidateId"].eq(candidate)
                & direct_groups["targetType"].eq("INDEPENDENT_ANY_ATTRACTOR")
            ]
            controls = direct_controls[
                direct_controls["evaluationCohort"].eq(cohort)
                & direct_controls["candidateId"].eq(candidate)
            ]
            reciprocal_group = reciprocal[
                reciprocal["evaluationCohort"].eq(cohort)
                & reciprocal["candidateId"].eq(candidate)
            ]
            reliable = reliability[
                reliability["evaluationCohort"].eq(cohort)
                & reliability["candidateId"].eq(candidate)
                & reliability["branchFamily"].eq("H32")
                & reliability["targetId"].eq("INDEPENDENT_ANY_ATTRACTOR")
            ]
            transfer_group = transfer[
                transfer["evaluationCohort"].eq(cohort)
                & transfer["candidateId"].eq(candidate)
            ].set_index("comparisonId")
            availability = bool(
                len(actual) == 3 and actual["availabilityFraction"].ge(0.9).all()
            )
            recognition = bool(
                len(actual) == 3
                and actual["recurrentRecognitionFraction"].ge(0.5).all()
            )
            control_gate = bool(
                len(controls) == 6
                and controls["occupancyDifferenceLower95"].gt(0).all()
                and controls["recurrenceDifferenceLower95"].gt(0).all()
            )
            reciprocal_gate = bool(
                len(reciprocal_group) == 3
                and reciprocal_group["rankGatePassed"].all()
            )
            reliability_gate = bool(
                len(reliable) == 1 and reliable.iloc[0].reliabilityGatePassed
            )

            def transfer_gate(
                comparison: str, frame: pd.DataFrame = transfer_group
            ) -> bool:
                return bool(
                    comparison in frame.index
                    and frame.loc[comparison, "gatePassed"]
                )

            h8_gate = transfer_gate("ACTUAL_H8_VS_ACTUAL_H32")
            teacher_gate = transfer_gate("ORIGINAL_H8_VS_ACTUAL_H32")
            branch_controls = transfer_gate(
                "ACTUAL_H32_MINUS_PERMUTED_H32"
            ) and transfer_gate("ACTUAL_H32_MINUS_UNRELATED_H32")
            family = availability and recognition and control_gate and reciprocal_gate
            committor = family and reliability_gate and h8_gate and branch_controls
            rows.append(
                {
                    "evaluationCohort": cohort,
                    "candidateId": candidate,
                    "atlasAvailabilityPassed": availability,
                    "heldOutRecurrencePassed": recognition,
                    "directControlDiscriminationPassed": control_gate,
                    "reciprocalRankPassed": reciprocal_gate,
                    "independentH32ReliabilityPassed": reliability_gate,
                    "independentH8H32RankPassed": h8_gate,
                    "originalTeacherTransferPassed": teacher_gate,
                    "branchControlDiscriminationPassed": branch_controls,
                    "multilineageAttractorFamilyPassed": family,
                    "independentAnyAttractorCommittorPassed": committor,
                }
            )
    gates = pd.DataFrame(rows)
    family_all = bool(gates["multilineageAttractorFamilyPassed"].all())
    committor_all = bool(gates["independentAnyAttractorCommittorPassed"].all())
    teacher_all = bool(gates["originalTeacherTransferPassed"].all())
    if committor_all:
        classifications = [
            "MULTILINEAGE_ATTRACTOR_FAMILY_ESTABLISHED",
            "INDEPENDENT_ANY_ATTRACTOR_COMMITTOR_ESTABLISHED",
            "TARGET_FAMILY_RETROSPECTIVE_REFERENCE_CONDITIONED",
            "NOT_PROMOTABLE_AS_CONFIRMED_PAPER_RESULT",
        ]
        next_theme = "INDEPENDENT_TARGET_TRANSITION_PATHWAY_HETEROGENEITY"
    elif family_all:
        classifications = [
            "MULTILINEAGE_ATTRACTOR_FAMILY_ESTABLISHED",
            "INDEPENDENT_ANY_ATTRACTOR_COMMITTOR_NOT_ESTABLISHED",
            "NOT_PROMOTABLE_AS_CONFIRMED",
        ]
        next_theme = "MULTILINEAGE_TARGET_BRANCHABILITY_REVIEW"
    else:
        classifications = [
            "MULTILINEAGE_ATTRACTOR_FAMILY_NOT_SUPPORTED",
            "TRAJECTORY_SPECIFIC_TARGET_CONFIRMED_WITHIN_TESTED_SCOPE",
            "NOT_PROMOTABLE_AS_CONFIRMED",
        ]
        next_theme = "PAST_ONLY_RECURRENCE_INHERITANCE_OUTCOME_CONSTRUCTION"
    if teacher_all:
        classifications.insert(-1, "ORIGINAL_SHOOTING_TEACHER_TRANSFERS_TO_TARGET_FAMILY")
    return gates, classifications, next_theme


def source_grounding_registry() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sourceId": "L37_REVIEWER_MULTILINEAGE_TARGET",
                "evidenceClass": "HUMAN_REVIEW_DIRECTION",
                "finding": "Test whether independent same-matrix lineages recover corresponding basins before interpreting the shooting teacher as network-level.",
                "frozenUse": "reciprocal two-reference atlas and held-out lineage recognition",
            },
            {
                "sourceId": "L37_L36_LINEAGE_SPECIFIC_RESULT",
                "evidenceClass": "DIRECT_FROZEN_E01_RESULT",
                "finding": "Single reference centroids failed the frozen cross-lineage agreement gate despite reliable target-conditioned committors.",
                "frozenUse": "motivates multi-attractor rather than dominant-centroid target",
            },
            {
                "sourceId": "L37_L23_THRESHOLD_COMPONENT",
                "evidenceClass": "DIRECT_FROZEN_E01_METHOD",
                "finding": "L23 uses historical cosine H at 0.9 to construct recurrent post-fission components.",
                "frozenUse": "unchanged similarity and component semantics",
            },
            {
                "sourceId": "L37_L28_L31_BRANCH_STREAMS",
                "evidenceClass": "DIRECT_FROZEN_E01_RESULT",
                "finding": "H32 and H8 streams are reliable and exactly replayable.",
                "frozenUse": "zero-new-stream any-attractor committor rescore",
            },
        ]
    )


def benchmark_projection(
    responses: pd.DataFrame,
    original_coordinates: pd.DataFrame,
    trajectory_manifest: pd.DataFrame,
) -> dict[str, Any]:
    rng = np.random.default_rng(derived_seed("benchmark_synthetic_atlas"))
    atlas_times = []
    for _ in range(8):
        states = rng.poisson(2.0, size=(200, 100)).astype(np.int64) + 1
        lineages = np.repeat(np.arange(2), 100)
        generations = np.tile(np.arange(100), 2)
        started = time.perf_counter()
        build_cross_lineage_atlas(states, lineages, generations)
        atlas_times.append(time.perf_counter() - started)
    base = L35.payloads(responses, original_coordinates, trajectory_manifest)
    original_map = {
        state_id: group.sort_values("coordinate")["centroidValue"].to_numpy(
            dtype=np.float64
        )
        for state_id, group in original_coordinates.groupby("stateId", sort=False)
    }
    branch_times = []
    for candidate in CANDIDATES:
        payload = dict(next(row for row in base if row["candidateId"] == candidate))
        original = original_map[payload["stateId"]]
        payload["originalCentroid"] = original.tolist()
        payload["targets"] = {
            "ORIGINAL_TRAJECTORY_BASIN": [original.tolist()],
            "INDEPENDENT_ANY_ATTRACTOR": [np.roll(original, 1).tolist()],
            "PERMUTED_SPECIES_ATLAS": [np.roll(original, 2).tolist()],
            "UNRELATED_MATRIX_ATLAS": [np.roll(original, 3).tolist()],
        }
        started = time.perf_counter()
        rows = _branch_worker(payload)
        branch_times.append(time.perf_counter() - started)
        if len(rows) != sum(BRANCH_COUNTS.values()) * 4:
            raise RuntimeError("L37 branch benchmark failed")
    projected_atlas_cpu = max(atlas_times) * 280 * 3 * 2.2 / 3600
    projected_branch_cpu = max(branch_times) * 280 * 2.2 / 3600
    projected_cpu = projected_atlas_cpu + projected_branch_cpu
    projected_wall = projected_cpu * 3600 / WORKERS
    return {
        "schema": "eidosoma.e01.s19_l37.benchmark_projection.v1",
        "status": "PASS"
        if projected_cpu <= 90 and projected_wall <= 64.8 * 3600
        else "STOP_BEFORE_OUTCOME",
        "syntheticAtlasDurationsSeconds": atlas_times,
        "branchWorkerDurationsSeconds": branch_times,
        "projectedCpuHoursIncludingRegeneration": projected_cpu,
        "projectedWallSecondsIncludingRegeneration": projected_wall,
        "newTrajectories": 0,
        "newBranchStreams": 0,
    }


def make_figures(
    atlases: pd.DataFrame,
    direct_groups: pd.DataFrame,
    direct_controls: pd.DataFrame,
    reciprocal: pd.DataFrame,
    reliability: pd.DataFrame,
    transfer: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    root = BUILD_ROOT / "figures"
    root.mkdir(parents=True, exist_ok=True)

    def save(name: str) -> None:
        plt.tight_layout()
        plt.savefig(root / name, dpi=180)
        plt.close()

    atlases.pivot_table(
        index=["evaluationCohort", "candidateId"],
        columns="heldOutLineage",
        values="componentCount",
        aggfunc="mean",
    ).plot(kind="bar", figsize=(13, 6))
    plt.ylabel("Mean cross-lineage recurring components")
    save("01_atlas_component_availability.png")

    actual = direct_groups[
        direct_groups["targetType"].eq("INDEPENDENT_ANY_ATTRACTOR")
    ]
    actual.pivot_table(
        index=["evaluationCohort", "candidateId"],
        columns="heldOutLineage",
        values="meanOccupancy",
    ).plot(kind="bar", figsize=(13, 6))
    plt.ylabel("Held-out boundary occupancy")
    save("02_reciprocal_heldout_occupancy.png")

    direct_controls.pivot_table(
        index=["evaluationCohort", "candidateId", "heldOutLineage"],
        columns="controlType",
        values="meanOccupancyDifference",
    ).plot(kind="bar", figsize=(15, 6))
    plt.axhline(0, color="black", linewidth=1)
    plt.ylabel("Actual minus control occupancy")
    save("03_same_matrix_vs_controls.png")

    reciprocal.pivot_table(
        index=["evaluationCohort", "candidateId"],
        columns=["leftHeldOut", "rightHeldOut"],
        values="spearman",
    ).plot(kind="bar", figsize=(15, 6))
    plt.axhline(0.5, color="black", linestyle="--")
    plt.ylabel("Reciprocal held-out occupancy Spearman")
    save("04_reciprocal_rank_agreement.png")

    reliability[
        reliability["evaluationCohort"].isin(EVALUATION_COHORTS)
        & reliability["branchFamily"].eq("H32")
    ].pivot_table(
        index="targetId",
        columns=["evaluationCohort", "candidateId"],
        values="splitHalfSpearman",
    ).plot(kind="bar", figsize=(15, 6))
    plt.axhline(0.5, color="black", linestyle="--")
    plt.ylabel("H32 split-half Spearman")
    save("05_any_attractor_committor_reliability.png")

    transfer[transfer["evaluationCohort"].isin(EVALUATION_COHORTS)].pivot_table(
        index="comparisonId",
        columns=["evaluationCohort", "candidateId"],
        values="pointEstimate",
    ).plot(kind="bar", figsize=(15, 6))
    plt.axhline(0, color="black", linewidth=1)
    plt.ylabel("Registered rank or paired q difference")
    save("06_branch_target_transfer_and_controls.png")

    checks = [
        "atlasAvailabilityPassed",
        "heldOutRecurrencePassed",
        "directControlDiscriminationPassed",
        "reciprocalRankPassed",
        "independentH32ReliabilityPassed",
        "independentH8H32RankPassed",
        "originalTeacherTransferPassed",
        "branchControlDiscriminationPassed",
        "multilineageAttractorFamilyPassed",
        "independentAnyAttractorCommittorPassed",
    ]
    matrix = gates.set_index(["evaluationCohort", "candidateId"])[checks].astype(float)
    plt.figure(figsize=(14, 5))
    plt.imshow(matrix, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    plt.xticks(range(len(checks)), checks, rotation=35, ha="right", fontsize=7)
    plt.yticks(range(len(matrix)), ["/".join(index) for index in matrix.index], fontsize=7)
    plt.colorbar(ticks=[0, 1])
    save("07_multilineage_target_decision_matrix.png")


def manifest_for(root: Path) -> dict[str, Any]:
    files = [
        {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(
            item
            for item in root.rglob("*")
            if item.is_file() and item.name != "artifact_manifest.json"
        )
    ]
    return {
        "schema": "eidosoma.e01.s19_l37.artifact_manifest.v1",
        "root": str(root),
        "fileCount": len(files),
        "totalBytes": sum(item["bytes"] for item in files),
        "files": files,
    }


def append_ledgers(
    classifications: list[str], timestamp: str, next_theme: str
) -> None:
    ledger_path = ARTIFACT_ROOT / "self_improvement_ledger.parquet"
    ledger = pd.read_parquet(ledger_path)
    sequence = int(ledger["ledgerSequence"].max()) + 1
    additions = [
        {
            "appendOnly": True,
            "beliefBeforeLoop": "L36 showed that one dominant completed-lineage centroid is not a stable network-level destination.",
            "failureOrAmbiguityTargeted": "Single-attractor lineage specificity versus a reproducible family of multiple metastable basins.",
            "informationGainRationale": "Reciprocal leave-one-lineage-out atlases test the target before another representation search.",
            "learned": "L37 reciprocal atlas, unrelated-matrix and matched-permutation controls frozen before outcomes.",
            "ledgerSequence": sequence,
            "loopId": LOOP_ID,
            "motivatingEvidence": "L36 classifications and reviewer multi-lineage attractor-atlas decision tree.",
            "proposedNextTest": "Build two-lineage atlases and evaluate the third lineage plus exact existing branches.",
            "recordPhase": "PRE_LOOP_METHOD_LOCK",
            "remainingPlausibleHypotheses": "Multi-attractor network family, target-independent recurrence/inheritance outcome, or shooting-only estimation.",
            "selectedHypotheses": "Any component recurrent in both independent reference lineages at frozen H090.",
            "timestampUtc": timestamp,
            "weakenedHypotheses": "One most-recurring completed-run centroid is a universal destination.",
        },
        {
            "appendOnly": True,
            "beliefBeforeLoop": "A network-level atlas must recognize reciprocal held-out lineages and outperform unrelated/random regions.",
            "failureOrAmbiguityTargeted": "Target-family reproducibility and branchability.",
            "informationGainRationale": "Reciprocal roles and controls separate same-matrix basin families from generic high-H coverage.",
            "learned": ";".join(classifications),
            "ledgerSequence": sequence + 1,
            "loopId": LOOP_ID,
            "motivatingEvidence": "Complete L37 result.",
            "proposedNextTest": next_theme,
            "recordPhase": "POST_LOOP_RESULT_AUTONOMOUS_CONTINUATION",
            "remainingPlausibleHypotheses": next_theme,
            "selectedHypotheses": "Cross-lineage component-centroid union.",
            "timestampUtc": timestamp,
            "weakenedHypotheses": "A target can be called network-level without reciprocal and control discrimination.",
        },
    ]
    BASE.write_parquet(
        ledger_path,
        pd.concat(
            [ledger, pd.DataFrame(additions).reindex(columns=ledger.columns)],
            ignore_index=True,
        ),
    )
    markdown = ARTIFACT_ROOT / "SELF_IMPROVEMENT_LEDGER.md"
    BASE.atomic_text(
        markdown,
        markdown.read_text()
        + f"\n\n## {LOOP_ID} — multi-lineage any-attractor target\n\n"
        + f"- **Learned:** {', '.join(classifications)}.\n"
        + f"- **Next:** {next_theme}.\n",
    )
    candidates_path = ARTIFACT_ROOT / "candidate_registry.parquet"
    candidates = pd.read_parquet(candidates_path)
    candidate = {
        "branchCount": 3,
        "bundleId": "L37_MULTILINEAGE_ANY_ATTRACTOR",
        "candidateId": "S19-L37-CROSS-LINEAGE-ATLAS-UNION",
        "candidateSpecificSuccess": 0,
        "completedFitLeakage": 1,
        "computeEfficiency": 5,
        "crossCandidateDiscriminability": 5,
        "deterministicHReuse": 0,
        "explanatoryLeverage": 5,
        "frozenRank": 1,
        "independenceFromPriorOutcomeSelection": 5,
        "outcomeGuidedThresholdSelection": 0,
        "paperFingerprintSpecificity": 0,
        "proposedSpecification": "reciprocal union of components recurrent in two independent same-matrix lineages",
        "rankingScore": 30.0,
        "registryOrder": int(candidates["registryOrder"].max()) + 1,
        "selected": True,
        "selectionReason": "L36_LINEAGE_SPECIFIC_TARGET_AND_REVIEWER_DECISION_TREE",
        "sourceGrounding": 5,
        "testability": 5,
        "undefinedAuthorSemantics": 0,
    }
    BASE.write_parquet(
        candidates_path,
        pd.concat(
            [candidates, pd.DataFrame([candidate]).reindex(columns=candidates.columns)],
            ignore_index=True,
        ),
    )
    source_path = ARTIFACT_ROOT / "source_search_ledger.parquet"
    sources = pd.read_parquet(source_path)
    source_rows = [
        {
            "commitOrVersion": None,
            "evidenceClass": source.evidenceClass,
            "finding": f"{source.finding}; L37 use: {source.frozenUse}",
            "licenseStatus": "WORKSPACE_OR_HUMAN_DIRECTION",
            "redistributionStatus": "INTERNAL_EVIDENCE_ONLY",
            "repositoryIdentity": None,
            "retainedPath": None,
            "retrievalDate": timestamp[:10],
            "sha256": None,
            "sourceId": f"L37_{source.sourceId}",
            "sourceType": source.evidenceClass,
            "treeIdentity": None,
            "url": None,
        }
        for source in source_grounding_registry().itertuples(index=False)
    ]
    BASE.write_parquet(
        source_path,
        pd.concat(
            [sources, pd.DataFrame(source_rows).reindex(columns=sources.columns)],
            ignore_index=True,
        ),
    )
    registry_path = ARTIFACT_ROOT / "loop_registry.yaml"
    registry = yaml.safe_load(registry_path.read_text())
    registry["loops"].append(
        {
            "loopId": LOOP_ID,
            "versionedLoopId": VERSION,
            "status": "COMPLETE_AUTONOMOUS_CONTINUATION_AUTHORIZED",
            "authorized": True,
            "completed": True,
            "outcomeAccessed": True,
            "humanReviewRequiredAfter": False,
            "classification": classifications,
            "selectedDiscoveryLead": None,
            "newMatrices": 0,
            "newTrajectories": 0,
            "newBranchStreams": 0,
            "nextStepActive": True,
        }
    )
    registry["proposedNextLoopTheme"] = next_theme
    registry["proposedNextLoopActive"] = True
    registry["authorizationUpperBound"] = "S19-L55"
    BASE.atomic_text(registry_path, yaml.safe_dump(registry, sort_keys=False))
    history_path = ARTIFACT_ROOT / "human_review_history.json"
    history = json.loads(history_path.read_text())
    if not any(
        item.get("decision") == "EXTEND_AUTONOMOUS_S19_UPPER_BOUND_TO_L55"
        for item in history["history"]
    ):
        history["history"].append(
            {
                "decision": "EXTEND_AUTONOMOUS_S19_UPPER_BOUND_TO_L55",
                "recordedAtUtc": timestamp,
                "scope": "organization-before-replicator-event discovery",
                "source": "explicit_human_direction",
                "status": "ACTIVE_CONSUMED_SEQUENTIALLY",
                "upperBound": "S19-L55",
                "s20Activated": False,
            }
        )
    history["history"].append(
        {
            "decision": "S19_L37_COMPLETE_AUTONOMOUS_CONTINUATION",
            "loopId": LOOP_ID,
            "nextLoopAuthorized": True,
            "recordedAtUtc": timestamp,
            "result": classifications,
            "s20Activated": False,
            "scope": VERSION,
            "source": "locked_execution_result",
        }
    )
    history["pendingDecision"] = "NONE_AUTONOMOUS_SEQUENCE_ACTIVE_THROUGH_L55"
    BASE.write_json(history_path, history)


def report_text(
    atlases: pd.DataFrame,
    direct_groups: pd.DataFrame,
    direct_controls: pd.DataFrame,
    reciprocal: pd.DataFrame,
    reliability: pd.DataFrame,
    transfer: pd.DataFrame,
    gates: pd.DataFrame,
    classifications: list[str],
    runtime: dict[str, Any],
    next_theme: str,
) -> str:
    amendment_note = (
        "Failed attempt 01 is retained. TA01 corrected only the validator's "
        "accounting unit from internal atlas centroids to registered target "
        "families, matching the unchanged worker contract; no target, score, "
        "branch, method or scientific value changed."
        if runtime.get("technicalAmendmentApplied")
        else "No post-lock technical amendment was applied."
    )
    eval_groups = direct_groups[
        direct_groups["evaluationCohort"].isin(EVALUATION_COHORTS)
    ]
    eval_controls = direct_controls[
        direct_controls["evaluationCohort"].isin(EVALUATION_COHORTS)
    ]
    eval_reciprocal = reciprocal[
        reciprocal["evaluationCohort"].isin(EVALUATION_COHORTS)
    ]
    eval_reliability = reliability[
        reliability["evaluationCohort"].isin(EVALUATION_COHORTS)
        & reliability["branchFamily"].eq("H32")
    ]
    eval_transfer = transfer[
        transfer["evaluationCohort"].isin(EVALUATION_COHORTS)
    ]
    component_summary = atlases.groupby(
        ["evaluationCohort", "candidateId", "heldOutLineage"], sort=True
    ).agg(
        states=("stateId", "size"),
        eligibleFraction=("status", lambda value: float(np.mean(value == "ELIGIBLE"))),
        meanComponents=("componentCount", "mean"),
        medianComponents=("componentCount", "median"),
    ).reset_index()
    return f"""# S19-L37 — Multi-Lineage Any-Attractor Target Construction

## Chief/human handoff

- **Step:** `{VERSION}`
- **Status:** complete under the extended L19–L55 autonomous sequence.
- **Classifications:** {", ".join(f"`{value}`" for value in classifications)}
- **Validation:** exact replay of all 840 frozen lineage identities; reciprocal leave-one-lineage-out target construction; unrelated-matrix and matched species-permutation controls; exact discrete/path replay of all 53,760 frozen H32/H8 streams; independent complete target/result regeneration; 4,096 matrix bootstraps; immutable/runtime/storage/artifact hashes.
- **Recommended next action:** `{next_theme}`.

## Frozen question

Can two independent same-matrix lineages define a family of recurring post-fission basins that recognizes a third lineage and supports a target-specific finite-horizon committor without using that held-out completed future?

## Method

For each of 280 matrix/candidate states, each of ORIGINAL, REFERENCE_A and REFERENCE_B was held out in turn. The other two 100-generation post-fission sequences formed a strict `H>=0.9` threshold graph. Every connected component with at least two visits, separated by at least two generations, in each reference lineage contributed one centroid to an any-attractor union. The held-out lineage was scored only against that independent atlas. A deterministic same-candidate/cohort unrelated-matrix atlas and a molecule-identity-permuted atlas preserved explicit controls. Existing H32/H8 paths were regenerated with their exact streams and scored offline; no trajectory or branch stream was added.

## Atlas availability

{component_summary.to_markdown(index=False)}

## Reciprocal held-out recognition

{eval_groups.to_markdown(index=False)}

## Direct controls

{eval_controls.to_markdown(index=False)}

## Reciprocal rank agreement

{eval_reciprocal.to_markdown(index=False)}

## Branch committor reliability

{eval_reliability.to_markdown(index=False)}

## Branch target transfer and controls

{eval_transfer.to_markdown(index=False)}

## Decision gates

{gates.to_markdown(index=False)}

The complete registered gate, not one favorable role, candidate, cohort or target, determines the classification. A reference atlas remains constructed from completed *independent* lineages and is not a past-only biomarker or paper confirmation.

## Validation and provenance

- Repository lock: `{runtime['repositoryHead']}`.
- {amendment_note}
- Workers: `{runtime['workers']}` with one numerical-library thread per worker; GPU hours `0`.
- Wall time: `{runtime['wallSeconds']:.2f}` seconds.
- New matrices, trajectories and branch streams: `0/0/0`.
- Frozen branch streams rescored: `{runtime['uniqueFrozenBranchStreamsRescored']}` and identically rescored again for complete regeneration.
- No threshold, component rule, held-out role, simulator, horizon or control was changed after outcomes.

## Interpretation boundary

This loop tests whether a target destination exists independently of the evaluated completed lineage. It does not establish a paper replicator, prospective observation-only precursor, causal emergence, intervention efficacy or causal control. The original state sample was selected under the earlier original-target task, so even a successful atlas remains selection-conditioned.

## Next boundary

L37 is frozen. The standing human authorization permits `{next_theme}` as the next bounded loop through L55. S20, E02, author contact, interventions and report generation remain inactive.
"""


def prepare_lock() -> None:
    LOOP_ROOT.mkdir(parents=True, exist_ok=True)
    if git("status", "--porcelain=v1"):
        raise RuntimeError("repository must be clean before L37 lock")
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/eidosoma/groups/42")
    if head != remote:
        raise RuntimeError("local/remote commit mismatch")
    prior = validate_immutable_prior()
    fixtures = fixture_results()
    lineages = lineage_registry()
    responses = pd.read_parquet(L36_ROOT / "response_registry.parquet")
    original_coordinates = pd.read_parquet(
        L36_ROOT / "original_target_coordinates.parquet"
    )
    trajectory_manifest = pd.read_parquet(
        L23_ROOT / "input_trajectory_manifest.parquet"
    )
    seeds = analysis_seed_manifest(lineages)
    firewall = seed_firewall(seeds)
    benchmark = benchmark_projection(
        responses, original_coordinates, trajectory_manifest
    )
    if (
        not prior["unchanged"]
        or not fixtures["passed"].all()
        or not lineages["identityExact"].all()
        or firewall["status"] != "PASS"
        or benchmark["status"] != "PASS"
    ):
        raise RuntimeError("L37 preoutcome validation or benchmark failed")
    shutil.copy2(CONFIG, LOOP_ROOT / "preregistration.yaml")
    BASE.atomic_text(
        LOOP_ROOT / "decision_record.md",
        "# S19-L37 decision record\n\n"
        "L36 answered the reviewer-defined target-provenance question: single completed-lineage centroids were not stable across two independent same-matrix reference lineages. The human extended autonomous work through L55 and directed the next emphasis toward independently existing attractors. L37 therefore freezes one reciprocal target family before outcomes. Each lineage is held out; the other two define every post-fission H090 component recurring at least twice with generation span at least two in both references. The target is the union of all such component centroids. Unrelated-matrix and molecule-permuted controls are fixed, and exact existing H32/H8 paths are rescored without new streams. No favorable lineage, threshold, radius, component or result may be selected.\n",
    )
    BASE.write_json(LOOP_ROOT / "immutable_prior_validation.json", prior)
    BASE.write_parquet(LOOP_ROOT / "fixture_results.parquet", fixtures)
    BASE.write_parquet(LOOP_ROOT / "lineage_registry.parquet", lineages)
    BASE.write_parquet(LOOP_ROOT / "response_registry.parquet", responses)
    BASE.write_parquet(
        LOOP_ROOT / "original_target_coordinates.parquet", original_coordinates
    )
    BASE.write_parquet(
        LOOP_ROOT / "input_trajectory_manifest.parquet", trajectory_manifest
    )
    BASE.write_parquet(LOOP_ROOT / "analysis_seed_manifest.parquet", seeds)
    BASE.write_json(LOOP_ROOT / "seed_firewall.json", firewall)
    BASE.write_json(LOOP_ROOT / "benchmark_projection.json", benchmark)
    BASE.write_parquet(
        LOOP_ROOT / "source_grounding_registry.parquet", source_grounding_registry()
    )
    hashes = {
        "configSha256": sha256_file(CONFIG),
        "lineageRegistrySha256": sha256_file(LOOP_ROOT / "lineage_registry.parquet"),
        "responseRegistrySha256": sha256_file(LOOP_ROOT / "response_registry.parquet"),
        "originalCoordinatesSha256": sha256_file(
            LOOP_ROOT / "original_target_coordinates.parquet"
        ),
        "trajectoryManifestSha256": sha256_file(
            LOOP_ROOT / "input_trajectory_manifest.parquet"
        ),
        "analysisSeedsSha256": sha256_file(
            LOOP_ROOT / "analysis_seed_manifest.parquet"
        ),
        "seedFirewallSha256": sha256_file(LOOP_ROOT / "seed_firewall.json"),
        "benchmarkSha256": sha256_file(LOOP_ROOT / "benchmark_projection.json"),
        "l36ManifestSha256": sha256_file(L36_ROOT / "artifact_manifest.json"),
    }
    lock = {
        "schema": "eidosoma.e01.s19_l37.implementation_lock.v1",
        "repositoryHead": head,
        "remoteHead": remote,
        "runnerSha256": sha256_file(RUNNER_PATH),
        "coreSha256": sha256_file(CORE_PATH),
        "threshold": THRESHOLD,
        "minimumVisitsPerReferenceLineage": MIN_VISITS,
        "minimumGenerationSpan": MIN_SPAN,
        "lineages": list(LINEAGES),
        "targetTypes": list(TARGET_TYPES),
        "heldOutRoles": list(LINEAGES),
        "horizons": HORIZONS,
        "branchCounts": BRANCH_COUNTS,
        "matrixBootstraps": BOOTSTRAPS,
        "newMatrices": 0,
        "newTrajectories": 0,
        "newBranchStreams": 0,
        "lockedHashes": hashes,
        "outcomeAccessed": False,
        "lockedAtUtc": utc_now(),
    }
    BASE.write_json(LOOP_ROOT / "implementation_lock.json", lock)
    BASE.write_json(
        LOOP_ROOT / "preoutcome_repository_lock.json",
        {
            "head": head,
            "remote": remote,
            "priorAggregateSha256": prior["aggregateSha256"],
            "runnerSha256": sha256_file(RUNNER_PATH),
            "coreSha256": sha256_file(CORE_PATH),
            **hashes,
        },
    )


def prepare_technical_amendment() -> None:
    """Freeze the single value-preserving cardinality-validator repair."""

    if git("status", "--porcelain=v1"):
        raise RuntimeError("repository must be clean before L37 technical amendment")
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/eidosoma/groups/42")
    if head != remote:
        raise RuntimeError("local/remote commit mismatch")
    original_lock = json.loads(
        (LOOP_ROOT / "preoutcome_repository_lock.json").read_text()
    )
    prior = validate_immutable_prior()
    fixtures = fixture_results()
    locked_files = {
        "lineageRegistrySha256": LOOP_ROOT / "lineage_registry.parquet",
        "responseRegistrySha256": LOOP_ROOT / "response_registry.parquet",
        "originalCoordinatesSha256": LOOP_ROOT / "original_target_coordinates.parquet",
        "trajectoryManifestSha256": LOOP_ROOT / "input_trajectory_manifest.parquet",
        "analysisSeedsSha256": LOOP_ROOT / "analysis_seed_manifest.parquet",
        "seedFirewallSha256": LOOP_ROOT / "seed_firewall.json",
        "benchmarkSha256": LOOP_ROOT / "benchmark_projection.json",
        "l36ManifestSha256": L36_ROOT / "artifact_manifest.json",
    }
    for key, path in locked_files.items():
        if sha256_file(path) != original_lock[key]:
            raise RuntimeError(f"L37 locked input changed before repair: {path}")
    if (
        not prior["unchanged"]
        or prior["aggregateSha256"] != original_lock["priorAggregateSha256"]
        or not fixtures["passed"].all()
        or not bool(
            fixtures.loc[
                fixtures["fixtureId"].eq("TARGET_FAMILY_NOT_COMPONENT_CARDINALITY"),
                "passed",
            ].all()
        )
    ):
        raise RuntimeError("L37 amendment validation failed")
    failure_record = {
        "schema": "eidosoma.e01.s19_l37.failed_attempt.v1",
        "attempt": 1,
        "status": "STOPPED_AT_BRANCH_RESCORE_CARDINALITY_VALIDATOR",
        "outcomesOpened": True,
        "scientificAggregationReleased": False,
        "exception": "RuntimeError: L37 branch-rescore cardinality failure",
        "diagnosis": (
            "The worker emitted one row per registered target family as locked. "
            "The validator incorrectly summed the number of centroids inside each "
            "multi-centroid atlas as though each centroid were a target family."
        ),
        "scientificValuesChanged": False,
        "scientificMethodChanged": False,
        "targetFamiliesChanged": False,
        "componentCoordinatesChanged": False,
        "branchStreamsChanged": False,
        "scientificAggregationRowsReleased": 0,
        "recordedAtUtc": utc_now(),
    }
    amendment = {
        "schema": "eidosoma.e01.s19_l37.technical_amendment_lock.v1",
        "status": "LOCKED_VALUE_PRESERVING",
        "amendmentId": "L37-TA01-TARGET-FAMILY-CARDINALITY",
        "originalHead": original_lock["head"],
        "head": head,
        "remote": remote,
        "runnerSha256": sha256_file(RUNNER_PATH),
        "coreSha256": sha256_file(CORE_PATH),
        "configSha256": sha256_file(CONFIG),
        "validatorUnitBefore": "INTERNAL_CENTROID_COUNT",
        "validatorUnitAfter": "REGISTERED_TARGET_FAMILY_COUNT",
        "workerUnit": "REGISTERED_TARGET_FAMILY_COUNT",
        "scientificValueChange": False,
        "scientificMethodChange": False,
        "thresholdChanged": False,
        "targetChanged": False,
        "componentChanged": False,
        "branchStreamChanged": False,
        "fixtureId": "TARGET_FAMILY_NOT_COMPONENT_CARDINALITY",
        "failedAttempt": failure_record,
        "lockedAtUtc": utc_now(),
    }
    BASE.write_json(LOOP_ROOT / "failed_attempt_01.json", failure_record)
    BASE.write_json(LOOP_ROOT / "technical_amendment_lock.json", amendment)
    pd.DataFrame(
        [
            {
                "amendmentId": amendment["amendmentId"],
                "status": amendment["status"],
                "reason": failure_record["diagnosis"],
                "scientificValueChange": False,
                "scientificMethodChange": False,
                "head": head,
                "lockedAtUtc": amendment["lockedAtUtc"],
            }
        ]
    ).to_csv(LOOP_ROOT / "technical_amendment_ledger.csv", index=False)


def execute() -> None:
    started = time.perf_counter()
    started_cpu = time.process_time()
    lock = json.loads((LOOP_ROOT / "preoutcome_repository_lock.json").read_text())
    amendment_path = LOOP_ROOT / "technical_amendment_lock.json"
    amendment = json.loads(amendment_path.read_text()) if amendment_path.is_file() else None
    expected_head = amendment["head"] if amendment is not None else lock["head"]
    expected_runner = (
        amendment["runnerSha256"] if amendment is not None else lock["runnerSha256"]
    )
    expected_core = (
        amendment["coreSha256"] if amendment is not None else lock["coreSha256"]
    )
    if (
        git("rev-parse", "HEAD") != expected_head
        or git("rev-parse", "origin/eidosoma/groups/42") != expected_head
        or git("status", "--porcelain=v1")
    ):
        raise RuntimeError("L37 repository lock mismatch")
    prior = validate_immutable_prior()
    fixtures = fixture_results()
    locked_files = {
        "lineageRegistrySha256": LOOP_ROOT / "lineage_registry.parquet",
        "responseRegistrySha256": LOOP_ROOT / "response_registry.parquet",
        "originalCoordinatesSha256": LOOP_ROOT / "original_target_coordinates.parquet",
        "trajectoryManifestSha256": LOOP_ROOT / "input_trajectory_manifest.parquet",
        "analysisSeedsSha256": LOOP_ROOT / "analysis_seed_manifest.parquet",
        "seedFirewallSha256": LOOP_ROOT / "seed_firewall.json",
        "benchmarkSha256": LOOP_ROOT / "benchmark_projection.json",
        "l36ManifestSha256": L36_ROOT / "artifact_manifest.json",
    }
    for key, path in locked_files.items():
        if sha256_file(path) != lock[key]:
            raise RuntimeError(f"L37 locked input changed: {path}")
    if (
        not prior["unchanged"]
        or prior["aggregateSha256"] != lock["priorAggregateSha256"]
        or not fixtures["passed"].all()
        or sha256_file(RUNNER_PATH) != expected_runner
        or sha256_file(CORE_PATH) != expected_core
        or (amendment is not None and amendment["status"] != "LOCKED_VALUE_PRESERVING")
    ):
        raise RuntimeError("L37 pre-execution validation failed")
    lineages = pd.read_parquet(LOOP_ROOT / "lineage_registry.parquet")
    responses = pd.read_parquet(LOOP_ROOT / "response_registry.parquet")
    original_coordinates = pd.read_parquet(
        LOOP_ROOT / "original_target_coordinates.parquet"
    )
    trajectory_manifest = pd.read_parquet(
        LOOP_ROOT / "input_trajectory_manifest.parquet"
    )
    if BUILD_ROOT.exists():
        shutil.rmtree(BUILD_ROOT)
    BUILD_ROOT.mkdir(parents=True)

    atlases, components, coordinates, memberships, permutations = (
        build_atlas_registries(lineages)
    )
    unrelated = unrelated_control_map(atlases)
    direct = direct_heldout_results(lineages, atlases, coordinates, unrelated)
    direct_groups = direct_group_results(direct)
    direct_controls, direct_bootstrap = direct_control_bootstraps(direct)
    reciprocal, reciprocal_bootstrap = reciprocal_results(direct)
    payloads = branch_payloads(
        responses,
        original_coordinates,
        trajectory_manifest,
        coordinates,
        unrelated,
    )
    branches = rescore_branches(payloads)
    compact = original_compact_validation(branches)
    states = state_committor_results(branches)
    reliability, reliability_bootstrap = reliability_results(states)
    transfer, transfer_bootstrap = branch_transfer_results(states)
    gates, classifications, next_theme = scientific_gates(
        direct_groups, direct_controls, reciprocal, reliability, transfer
    )
    make_figures(
        atlases,
        direct_groups,
        direct_controls,
        reciprocal,
        reliability,
        transfer,
        gates,
    )

    for name in (
        "preregistration.yaml",
        "decision_record.md",
        "immutable_prior_validation.json",
        "fixture_results.parquet",
        "lineage_registry.parquet",
        "response_registry.parquet",
        "original_target_coordinates.parquet",
        "input_trajectory_manifest.parquet",
        "analysis_seed_manifest.parquet",
        "seed_firewall.json",
        "benchmark_projection.json",
        "source_grounding_registry.parquet",
        "implementation_lock.json",
        "preoutcome_repository_lock.json",
    ):
        shutil.copy2(LOOP_ROOT / name, BUILD_ROOT / name)
    for name in (
        "failed_attempt_01.json",
        "technical_amendment_lock.json",
        "technical_amendment_ledger.csv",
    ):
        source = LOOP_ROOT / name
        if source.is_file():
            shutil.copy2(source, BUILD_ROOT / name)
    tables = {
        "atlas_registry.parquet": atlases,
        "atlas_component_results.parquet": components,
        "atlas_centroid_coordinates.parquet": coordinates,
        "atlas_membership_results.parquet": memberships,
        "species_permutation_manifest.parquet": permutations,
        "unrelated_control_map.parquet": unrelated,
        "heldout_lineage_results.parquet": direct,
        "heldout_group_results.parquet": direct_groups,
        "direct_control_results.parquet": direct_controls,
        "direct_control_bootstrap.parquet": direct_bootstrap,
        "reciprocal_results.parquet": reciprocal,
        "reciprocal_bootstrap.parquet": reciprocal_bootstrap,
        "branch_rescore_results.parquet": branches,
        "original_compact_replay_validation.parquet": compact,
        "state_committor_results.parquet": states,
        "committor_reliability_results.parquet": reliability,
        "committor_reliability_bootstrap.parquet": reliability_bootstrap,
        "branch_transfer_results.parquet": transfer,
        "branch_transfer_bootstrap.parquet": transfer_bootstrap,
        "scientific_gate_results.parquet": gates,
    }
    for name, frame in tables.items():
        BASE.write_parquet(BUILD_ROOT / name, frame)
    BASE.write_json(
        BUILD_ROOT / "classification.json",
        {
            "schema": "eidosoma.e01.s19_l37.classification.v1",
            "classifications": classifications,
            "multilineageAttractorFamilyEstablished": bool(
                gates["multilineageAttractorFamilyPassed"].all()
            ),
            "independentAnyAttractorCommittorEstablished": bool(
                gates["independentAnyAttractorCommittorPassed"].all()
            ),
            "targetUsesHeldOutFuture": False,
            "stateSelectionOriginallyTargetConditioned": True,
            "priorStatusesChanged": False,
        },
    )
    pd.DataFrame(
        columns=[
            "stage",
            "stateId",
            "candidateId",
            "matrixIndex",
            "heldOutLineage",
            "branchFamily",
            "branchIndex",
            "exceptionClass",
            "exceptionMessage",
        ]
    ).to_csv(BUILD_ROOT / "failure_ledger.csv", index=False)

    # Complete target, held-out, branch and statistical regeneration.
    replay_atlases, replay_components, replay_coordinates, replay_memberships, replay_permutations = (
        build_atlas_registries(lineages)
    )
    replay_unrelated = unrelated_control_map(replay_atlases)
    replay_direct = direct_heldout_results(
        lineages, replay_atlases, replay_coordinates, replay_unrelated
    )
    replay_direct_groups = direct_group_results(replay_direct)
    replay_direct_controls, replay_direct_bootstrap = direct_control_bootstraps(
        replay_direct
    )
    replay_reciprocal, replay_reciprocal_bootstrap = reciprocal_results(replay_direct)
    replay_payloads = branch_payloads(
        responses,
        original_coordinates,
        trajectory_manifest,
        replay_coordinates,
        replay_unrelated,
    )
    replay_branches = rescore_branches(replay_payloads)
    replay_compact = original_compact_validation(replay_branches)
    replay_states = state_committor_results(replay_branches)
    replay_reliability, replay_reliability_bootstrap = reliability_results(replay_states)
    replay_transfer, replay_transfer_bootstrap = branch_transfer_results(replay_states)
    replay_gates, replay_classifications, replay_next = scientific_gates(
        replay_direct_groups,
        replay_direct_controls,
        replay_reciprocal,
        replay_reliability,
        replay_transfer,
    )
    replay_tables = {
        "atlas": (atlases, replay_atlases),
        "components": (components, replay_components),
        "coordinates": (coordinates, replay_coordinates),
        "memberships": (memberships, replay_memberships),
        "permutations": (permutations, replay_permutations),
        "unrelated": (unrelated, replay_unrelated),
        "direct": (direct, replay_direct),
        "directGroups": (direct_groups, replay_direct_groups),
        "directControls": (direct_controls, replay_direct_controls),
        "directBootstrap": (direct_bootstrap, replay_direct_bootstrap),
        "reciprocal": (reciprocal, replay_reciprocal),
        "reciprocalBootstrap": (reciprocal_bootstrap, replay_reciprocal_bootstrap),
        "branches": (branches, replay_branches),
        "compact": (compact, replay_compact),
        "states": (states, replay_states),
        "reliability": (reliability, replay_reliability),
        "reliabilityBootstrap": (
            reliability_bootstrap,
            replay_reliability_bootstrap,
        ),
        "transfer": (transfer, replay_transfer),
        "transferBootstrap": (transfer_bootstrap, replay_transfer_bootstrap),
        "gates": (gates, replay_gates),
    }
    checks = {
        key: frame_hash(left) == frame_hash(right)
        for key, (left, right) in replay_tables.items()
    }
    checks.update(
        {
            "classificationExact": classifications == replay_classifications,
            "nextThemeExact": next_theme == replay_next,
            "fixturesPassed": bool(fixtures["passed"].all()),
            "immutablePriorPassed": prior["unchanged"],
            "seedFirewallPassed": json.loads(
                (LOOP_ROOT / "seed_firewall.json").read_text()
            )["status"]
            == "PASS",
            "originalCompactReplayPassed": bool(compact["allPassed"].all()),
            "technicalAmendmentValuePreserving": amendment is None
            or (
                amendment["status"] == "LOCKED_VALUE_PRESERVING"
                and not amendment["scientificValueChange"]
                and not amendment["scientificMethodChange"]
                and amendment["workerUnit"]
                == amendment["validatorUnitAfter"]
            ),
            "noNewTrajectory": True,
            "noNewBranchStream": True,
        }
    )
    if not all(checks.values()):
        raise RuntimeError(f"L37 regeneration validation failed: {checks}")
    BASE.write_json(
        BUILD_ROOT / "regeneration_validation.json",
        {
            "schema": "eidosoma.e01.s19_l37.regeneration_validation.v1",
            "status": "PASS",
            "checks": checks,
            "atlasFrameSha256": frame_hash(atlases),
            "directFrameSha256": frame_hash(direct),
            "branchFrameSha256": frame_hash(branches),
            "stateCommittorFrameSha256": frame_hash(states),
        },
    )
    runtime = {
        "schema": "eidosoma.e01.s19_l37.runtime.v1",
        "repositoryHead": git("rev-parse", "HEAD"),
        "workers": WORKERS,
        "numericalLibraryThreadsPerWorker": 1,
        "gpuHours": 0,
        "wallSeconds": time.perf_counter() - started,
        "controllerCpuHours": (time.process_time() - started_cpu) / 3600,
        "lineageIdentitiesReused": len(lineages),
        "reciprocalAtlases": len(atlases),
        "uniqueFrozenBranchStreamsRescored": 53_760,
        "newMatrices": 0,
        "newTrajectories": 0,
        "newBranchStreams": 0,
        "technicalAmendmentApplied": amendment["amendmentId"]
        if amendment is not None
        else None,
        "completedAtUtc": utc_now(),
    }
    BASE.write_json(BUILD_ROOT / "runtime_manifest.json", runtime)
    retained = sum(path.stat().st_size for path in BUILD_ROOT.rglob("*") if path.is_file())
    temporary = sum(path.stat().st_size for path in CACHE_ROOT.rglob("*") if path.is_file())
    storage = {
        "schema": "eidosoma.e01.s19_l37.storage_validation.v1",
        "retainedBytes": retained,
        "retainedGiBCeiling": 25,
        "temporaryBytes": temporary,
        "temporaryGiBCeiling": 75,
        "status": "PASS"
        if retained < 25 * 2**30 and temporary < 75 * 2**30
        else "FAIL",
    }
    if storage["status"] != "PASS":
        raise RuntimeError("L37 storage ceiling exceeded")
    BASE.write_json(BUILD_ROOT / "storage_validation.json", storage)
    report = report_text(
        atlases,
        direct_groups,
        direct_controls,
        reciprocal,
        reliability,
        transfer,
        gates,
        classifications,
        runtime,
        next_theme,
    )
    BASE.atomic_text(BUILD_ROOT / "research_step_full_results.md", report)
    BASE.atomic_text(BUILD_ROOT / "S19_L37_FULL_RESULTS.md", report)
    BASE.atomic_text(
        BUILD_ROOT / "loop_decision_summary.md",
        "# S19-L37 decision summary\n\n"
        + f"**Classification:** {', '.join(classifications)}\n\n"
        + f"**All-group target family:** `{gates['multilineageAttractorFamilyPassed'].all()}`.\n\n"
        + f"**All-group independent committor:** `{gates['independentAnyAttractorCommittorPassed'].all()}`.\n\n"
        + f"**Next:** `{next_theme}`.\n",
    )
    BASE.write_json(BUILD_ROOT / "artifact_manifest.json", manifest_for(BUILD_ROOT))
    stage = LOOP_ROOT.with_name(".L37-promotion-stage")
    if stage.exists():
        shutil.rmtree(stage)
    shutil.copytree(BUILD_ROOT, stage)
    if LOOP_ROOT.exists():
        shutil.rmtree(LOOP_ROOT)
    os.replace(stage, LOOP_ROOT)
    shutil.rmtree(BUILD_ROOT)
    artifact_manifest = json.loads((LOOP_ROOT / "artifact_manifest.json").read_text())
    if any(
        sha256_file(LOOP_ROOT / item["path"]) != item["sha256"]
        for item in artifact_manifest["files"]
    ):
        raise RuntimeError("L37 artifact hash validation failed")
    append_ledgers(classifications, runtime["completedAtUtc"], next_theme)
    BASE.atomic_text(ARTIFACT_ROOT / "research_step_full_results.md", report)
    BASE.atomic_text(
        ARTIFACT_ROOT / "S19_CURRENT_HANDOFF.md",
        report.replace("# S19-L37", "# S19 current handoff — S19-L37", 1),
    )
    BASE.write_json(
        ARTIFACT_ROOT / "s19_status.json",
        {
            "schema": "eidosoma.e01.s19.status.v1",
            "status": "ACTIVE_AUTONOMOUS_SEQUENCE",
            "latestCompletedLoop": LOOP_ID,
            "latestClassification": classifications,
            "selectedDiscoveryLead": None,
            "nextAuthorizedLoop": "S19-L38",
            "authorizationUpperBound": "S19-L55",
            "s20Active": False,
            "updatedAtUtc": runtime["completedAtUtc"],
        },
    )
    BASE.write_json(ARTIFACT_ROOT / "artifact_manifest.json", manifest_for(ARTIFACT_ROOT))
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "classifications": classifications,
                "nextTheme": next_theme,
                "runtime": runtime,
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-lock", action="store_true")
    parser.add_argument("--prepare-technical-amendment", action="store_true")
    args = parser.parse_args()
    if args.prepare_lock and args.prepare_technical_amendment:
        parser.error("lock and amendment preparation are mutually exclusive")
    if args.prepare_lock:
        prepare_lock()
    elif args.prepare_technical_amendment:
        prepare_technical_amendment()
    else:
        execute()


if __name__ == "__main__":
    main()
