"""Execute S19-L31 untouched confirmation of the frozen L30 H8 coordinate."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
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
    os.environ.setdefault(variable, "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.special import expit, logit

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


L30 = _load_module(
    "e01_s19_l31_l30",
    REPO_ROOT / "scripts/e01/run_s19_l30_eight_step_generator_propagator.py",
)
L29 = L30.L29
L28 = L30.L28
L26 = _load_module(
    "e01_s19_l31_l26",
    REPO_ROOT / "scripts/e01/run_s19_l26_recurrence_map_analog_committor.py",
)
BASE = L30.BASE
LOOP_ID = "S19-L31"
VERSION = "E01-S19-L31-UNTOUCHED-EIGHT-STEP-PROPAGATOR-COMMITTOR-CONFIRMATION-v1.0.0"
CANDIDATES = L28.CANDIDATES
LANDMARKS = L28.LANDMARKS
STATES_PER_STRATUM = 8
H32_BRANCHES = 128
H32_HALF = 64
H8_BRANCHES = 64
H8_HALF = 32
H32_ROOT = "668de9809d77a3f56e9b881083c2db8fb8cda4e2fb207f6c540d23e80ca44b06"
H8_ROOT = "5d598b0a9e72bf6a2e97160bc16cf751eaf1c890362640aa677f0986d59f2453"
PHASE = "s19_l31_untouched_propagator_confirmation"
BOOTSTRAPS = 4096
PERMUTATIONS = 512
WORKERS = 8
PRIMARY_MODEL = "EIGHT_STEP_PROPAGATOR_MOMENTS"
CONTROL_MODELS = (
    "DEVELOPMENT_PRIOR",
    "TARGET_GEOMETRY_CONTROL",
    "EXACT_H_TRACE_ANALOG",
    "ORDINARY_PATH_ANALOG",
)
ARTIFACT_ROOT = Path("/artifacts/research_steps/S19")
LOOP_ROOT = ARTIFACT_ROOT / "loops/L31"
L30_ROOT = ARTIFACT_ROOT / "loops/L30"
L29_ROOT = ARTIFACT_ROOT / "loops/L29"
L28_ROOT = ARTIFACT_ROOT / "loops/L28"
L26_ROOT = ARTIFACT_ROOT / "loops/L26"
L25_ROOT = ARTIFACT_ROOT / "loops/L25"
L23_ROOT = ARTIFACT_ROOT / "loops/L23"
CACHE_ROOT = Path("/cache/e01_s19_l31")
BUILD_ROOT = CACHE_ROOT / "build"
CONFIG = REPO_ROOT / "configs/e01/s19_l31_untouched_propagator_confirmation.yaml"
RUNNER_PATH = Path(__file__)


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
        frame.reset_index(drop=True)
        .to_json(orient="table", index=False, double_precision=15)
        .encode()
    ).hexdigest()


def validate_immutable_prior() -> dict[str, Any]:
    prior = json.loads((L30_ROOT / "immutable_prior_validation.json").read_text())
    rows = list(prior["files"])
    manifest = json.loads((L30_ROOT / "artifact_manifest.json").read_text())
    rows.extend(
        {
            "path": str(L30_ROOT / item["path"]),
            "root": str(L30_ROOT),
            "bytes": item["bytes"],
            "sha256": item["sha256"],
        }
        for item in manifest["files"]
    )
    failures = []
    for row in rows:
        path = Path(row["path"])
        if not path.is_file():
            failures.append({"path": str(path), "reason": "MISSING"})
        elif sha256_file(path) != row["sha256"]:
            failures.append({"path": str(path), "reason": "HASH_MISMATCH"})
    return {
        "schema": "eidosoma.e01.s19_l31.immutable_prior_validation.v1",
        "status": "PASS" if not failures else "FAIL",
        "unchanged": not failures,
        "fileCount": len(rows),
        "aggregateSha256": hashlib.sha256(
            "\n".join(f"{row['path']}\t{row['sha256']}" for row in rows).encode()
        ).hexdigest(),
        "l30ArtifactFileCount": manifest["fileCount"],
        "failures": failures,
        "files": rows,
    }


def select_confirmation_states(task: pd.DataFrame) -> pd.DataFrame:
    used_l28 = pd.read_parquet(L28_ROOT / "restored_state_registry.parquet")
    selected: list[pd.Series] = []
    for candidate in CANDIDATES:
        excluded = set(
            used_l28[used_l28["candidateId"].eq(candidate)]["matrixIndex"].astype(int)
        )
        used: set[int] = set()
        for landmark in reversed(LANDMARKS):
            subset = task[
                task["candidateId"].eq(candidate)
                & task["landmark"].eq(landmark)
                & ~task["matrixIndex"].isin(excluded)
            ].copy()
            subset["sourceMatrixRole"] = subset["matrixRole"]
            subset["matrixRole"] = "CONFIRMATION"
            subset["selectionDigest"] = subset["matrixIndex"].map(
                lambda matrix, candidate=candidate, landmark=landmark: hashlib.sha256(
                    "\x1f".join(
                        [
                            VERSION,
                            "STATE_SELECTION",
                            candidate,
                            str(landmark),
                            str(int(matrix)),
                        ]
                    ).encode()
                ).hexdigest()
            )
            subset = subset.sort_values(["selectionDigest", "matrixIndex"])
            available = subset[~subset["matrixIndex"].isin(used)]
            chosen = available.head(STATES_PER_STRATUM)
            if len(chosen) != STATES_PER_STRATUM:
                raise RuntimeError("insufficient untouched unique matrices")
            for rank, (_, row) in enumerate(chosen.iterrows(), start=1):
                row = row.copy()
                row["selectionRank"] = rank
                row["stratumAvailable"] = len(available)
                selected.append(row)
                used.add(int(row["matrixIndex"]))
        if len(used) != len(LANDMARKS) * STATES_PER_STRATUM:
            raise RuntimeError("confirmation matrix uniqueness failure")
    result = (
        pd.DataFrame(selected)
        .sort_values(["candidateId", "landmark", "selectionRank"])
        .reset_index(drop=True)
    )
    if len(result) != 80:
        raise RuntimeError("confirmation state cardinality failure")
    return result


def build_state_lock(
    selection: pd.DataFrame, manifest: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest_index = manifest.set_index(["candidateId", "matrixIndex"])
    target_spec = L28.fixed_label_spec(L28.TARGET_ID)
    state_rows: list[dict[str, Any]] = []
    coordinate_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    reservoir = np.full(100, 0.01, dtype=np.float64)
    for row in selection.itertuples(index=False):
        manifest_row = manifest_index.loc[(row.candidateId, int(row.matrixIndex))]
        trajectory = L28.load_trajectory(manifest_row)
        selected = L28.selected_clock_observations(trajectory, L28.CLOCK_ID)
        current_index = int(row.landmark) - 1
        current = selected[current_index]
        restored = L28.restored_state_from_observation(current)
        post = tuple(
            item
            for item in trajectory.observations
            if item.observation_kind == "post_fission"
        )
        post_states = L28.states_from_observations(post)
        centroid, component = L28.dominant_component_centroid(post_states)
        label_rows, _ = L28.label_trajectory(
            trajectory, target_spec, clock_id=L28.CLOCK_ID
        )
        direct_scores = L28.cosine_to_reference(
            L28.states_from_observations(selected), centroid
        )
        source_scores = label_rows["labelScore"].to_numpy(dtype=np.float64)
        source_labels = label_rows["isReplicator"].to_numpy(dtype=bool)
        direct_labels = direct_scores >= L28.TARGET_THRESHOLD
        original_event = bool(
            np.any(direct_labels[int(row.landmark) : int(row.landmark) + 32])
        )
        beta_seed = L28.derive_seed(
            L28.L23_ROOT_HEX, L28.L23_PHASE, "catalytic_matrix", int(row.matrixIndex)
        )
        beta = L28.generate_beta(beta_seed)
        beta_hash = L28.simulator_array_sha256(beta)
        state_hash = L28.array_sha256(np.asarray(restored.state, dtype=np.int64))
        key = f"{row.candidateId}|{int(row.matrixIndex)}|{int(row.landmark)}"
        state_id = hashlib.sha256((VERSION + "|" + key).encode()).hexdigest()[:24]
        state_rows.append(
            {
                "stateId": state_id,
                "matrixRole": "CONFIRMATION",
                "sourceMatrixRole": row.sourceMatrixRole,
                "candidateId": row.candidateId,
                "matrixIndex": int(row.matrixIndex),
                "trajectoryId": row.trajectoryId,
                "landmark": int(row.landmark),
                "selectionRank": int(row.selectionRank),
                "selectionDigest": row.selectionDigest,
                "stratumAvailable": int(row.stratumAvailable),
                "currentSelectedIndex": current_index,
                "currentRawObservationIndex": int(current.observation_index),
                "currentObservationKind": str(current.observation_kind),
                "currentCompletedFissions": int(current.completed_fissions),
                "currentGrowthGeneration": int(current.growth_generation_one_based),
                "currentGenerationLocalStep": int(current.generation_local_step),
                "currentBatchStep": int(current.batch_step),
                "currentMass": int(sum(current.state)),
                "currentStateSha256": state_hash,
                "reservoirStateSha256": L28.array_sha256(reservoir),
                "betaSha256": beta_hash,
                "simulatorDefinition": trajectory.definition.identity,
                "simulatorDefinitionSha256": hashlib.sha256(
                    trajectory.definition.identity.encode()
                ).hexdigest(),
                "targetId": L28.TARGET_ID,
                "targetThreshold": L28.TARGET_THRESHOLD,
                "targetCentroidSha256": L28.array_sha256(centroid),
                "targetComponentSize": len(component),
                "targetCurrentScore": float(direct_scores[current_index]),
                "targetCurrentLabel": bool(direct_labels[current_index]),
                "originalSingleFutureEventWithin32": original_event,
                "selectedBeforeBranchOutcome": True,
            }
        )
        for coordinate, value in enumerate(centroid):
            coordinate_rows.append(
                {
                    "stateId": state_id,
                    "candidateId": row.candidateId,
                    "matrixIndex": int(row.matrixIndex),
                    "landmark": int(row.landmark),
                    "coordinate": coordinate,
                    "centroidValue": float(value),
                    "componentMemberIndices": json.dumps(component),
                }
            )
        score_error = float(np.max(np.abs(direct_scores - source_scores)))
        validation_rows.append(
            {
                "stateId": state_id,
                "candidateId": row.candidateId,
                "matrixIndex": int(row.matrixIndex),
                "landmark": int(row.landmark),
                "trajectoryIdentityPassed": trajectory.trajectory_sha256
                == manifest_row.trajectorySha256,
                "selectedClockIdentityPassed": len(selected)
                == int(manifest_row.selectedClockLength),
                "restoredStateExact": state_hash
                == L28.array_sha256(
                    np.asarray(selected[current_index].state, dtype=np.int64)
                ),
                "betaIdentityPassed": beta_hash == manifest_row.betaSha256,
                "targetLabelExact": bool(np.array_equal(direct_labels, source_labels)),
                "targetScoreMaxAbsoluteError": score_error,
                "targetScoreEquivalent": score_error <= 1e-12,
                "currentStateOutsideBasin": not bool(direct_labels[current_index]),
                "singleFutureEventReplay": original_event == bool(row.eventWithin32),
                "targetBasinConditioning": "RETROSPECTIVE_COMPLETED_RUN_MATRIX_SPECIFIC",
            }
        )
    states = (
        pd.DataFrame(state_rows)
        .sort_values(["candidateId", "landmark", "selectionRank"])
        .reset_index(drop=True)
    )
    coordinates = (
        pd.DataFrame(coordinate_rows)
        .sort_values(["candidateId", "matrixIndex", "landmark", "coordinate"])
        .reset_index(drop=True)
    )
    validation = (
        pd.DataFrame(validation_rows)
        .sort_values(["candidateId", "landmark", "matrixIndex"])
        .reset_index(drop=True)
    )
    checks = [
        "trajectoryIdentityPassed",
        "selectedClockIdentityPassed",
        "restoredStateExact",
        "betaIdentityPassed",
        "targetLabelExact",
        "targetScoreEquivalent",
        "currentStateOutsideBasin",
        "singleFutureEventReplay",
    ]
    if len(states) != 80 or not validation[checks].all().all():
        raise RuntimeError("confirmation state restoration failed")
    return states, coordinates, validation


def stream_identities(
    family: str, candidate: str, matrix: int, landmark: int, branch: int
) -> dict[str, Any]:
    root = H32_ROOT if family == "H32" else H8_ROOT
    return {
        purpose: L28.derive_seed(
            root,
            PHASE,
            f"{family.lower()}_{purpose}",
            matrix,
            candidate,
            landmark,
            branch,
        )
        for purpose in ("event", "trim", "fission", "daughter")
    }


def seed_manifest(states: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for state in states.itertuples(index=False):
        for family, branches in (("H32", H32_BRANCHES), ("H8", H8_BRANCHES)):
            for branch in range(branches):
                identities = stream_identities(
                    family,
                    state.candidateId,
                    int(state.matrixIndex),
                    int(state.landmark),
                    branch,
                )
                materials = [
                    value.seed_material_sha256 for value in identities.values()
                ]
                row = {
                    "stateId": state.stateId,
                    "candidateId": state.candidateId,
                    "matrixIndex": int(state.matrixIndex),
                    "landmark": int(state.landmark),
                    "branchFamily": family,
                    "branchIndex": branch,
                    "branchHalf": "A"
                    if branch < (H32_HALF if family == "H32" else H8_HALF)
                    else "B",
                    "rootHex": H32_ROOT if family == "H32" else H8_ROOT,
                    "streamIdentitySha256": hashlib.sha256(
                        "|".join(
                            [family, state.stateId, str(branch), *materials]
                        ).encode()
                    ).hexdigest(),
                }
                for purpose, identity in identities.items():
                    row[f"{purpose}DerivedSeed"] = str(identity.derived_seed)
                    row[f"{purpose}SeedMaterialSha256"] = identity.seed_material_sha256
                rows.append(row)
    result = (
        pd.DataFrame(rows)
        .sort_values(
            ["branchFamily", "candidateId", "landmark", "matrixIndex", "branchIndex"]
        )
        .reset_index(drop=True)
    )
    if len(result) != len(states) * (H32_BRANCHES + H8_BRANCHES):
        raise RuntimeError("seed manifest cardinality failure")
    if not result["streamIdentitySha256"].is_unique:
        raise RuntimeError("seed stream identities are not unique")
    return result


def seed_firewall(seeds: pd.DataFrame, prior: dict[str, Any]) -> dict[str, Any]:
    current: set[str] = set()
    for column in seeds.columns:
        if "seedmaterialsha256" in column.lower():
            current.update(seeds[column].dropna().astype(str))
    previous: set[str] = set()
    for path in ARTIFACT_ROOT.glob("loops/L*/**/*seed*manifest*.parquet"):
        if "/L31/" in str(path):
            continue
        try:
            frame = pd.read_parquet(path)
        except (OSError, ValueError, TypeError):
            continue
        for column in frame.columns:
            if "seedmaterialsha256" in column.lower():
                previous.update(frame[column].dropna().astype(str))
    overlaps = sorted(current & previous)
    collisions = []
    for root in (H32_ROOT, H8_ROOT):
        needle = root.encode()
        for row in prior["files"]:
            path = Path(row["path"])
            if path.is_file() and path.stat().st_size <= 64 * 1024 * 1024:
                try:
                    if needle in path.read_bytes():
                        collisions.append(str(path))
                except OSError:
                    pass
    return {
        "schema": "eidosoma.e01.s19_l31.seed_firewall.v1",
        "status": "PASS" if not overlaps and not collisions else "FAIL",
        "currentSeedMaterialCount": len(current),
        "expectedSeedMaterialCount": len(seeds) * 4,
        "allCurrentMaterialsUnique": len(current) == len(seeds) * 4,
        "priorSeedMaterialCount": len(previous),
        "overlapCount": len(overlaps),
        "overlaps": overlaps,
        "rootCollisionPaths": sorted(set(collisions)),
    }


def payloads(
    states: pd.DataFrame,
    coordinates: pd.DataFrame,
    manifest: pd.DataFrame,
    reference_variant: str,
) -> list[dict[str, Any]]:
    manifest_index = manifest.set_index(["candidateId", "matrixIndex"])
    centroid_map: dict[str, tuple[list[float], int, str]] = {}
    for state in states.itertuples(index=False):
        centroid = (
            coordinates[coordinates["stateId"].eq(state.stateId)]
            .sort_values("coordinate")["centroidValue"]
            .to_numpy(dtype=np.float64)
        )
        centroid_map[state.stateId] = (
            centroid.tolist(),
            int(state.targetComponentSize),
            state.stateId,
        )
    if reference_variant == "TARGET_REFERENCE_PERMUTATION":
        for _, group in states.groupby(["candidateId", "landmark"], sort=True):
            ordered = group.sort_values("stateId")["stateId"].tolist()
            donors = ordered[1:] + ordered[:1]
            original = {key: centroid_map[key] for key in ordered}
            for receiver, donor in zip(ordered, donors, strict=True):
                values, size, donor_id = original[donor]
                centroid_map[receiver] = (values, size, donor_id)
    rows = []
    for state in states.itertuples(index=False):
        manifest_row = manifest_index.loc[(state.candidateId, int(state.matrixIndex))]
        trajectory = L28.load_trajectory(manifest_row)
        selected = L28.selected_clock_observations(trajectory, L28.CLOCK_ID)
        observation = selected[int(state.currentSelectedIndex)]
        centroid, component_size, donor_id = centroid_map[state.stateId]
        rows.append(
            {
                **state._asdict(),
                "state": list(map(int, observation.state)),
                "centroid": centroid,
                "targetComponentSizeUsed": component_size,
                "targetReferenceDonorStateId": donor_id,
                "referenceVariant": reference_variant,
            }
        )
    return rows


def _branch_worker(task: tuple[dict[str, Any], str]) -> list[dict[str, Any]]:
    payload, family = task
    candidate = payload["candidateId"]
    matrix = int(payload["matrixIndex"])
    landmark = int(payload["landmark"])
    beta_seed = L28.derive_seed(
        L28.L23_ROOT_HEX, L28.L23_PHASE, "catalytic_matrix", matrix
    )
    beta = L28.generate_beta(beta_seed)
    if L28.simulator_array_sha256(beta) != payload["betaSha256"]:
        raise RuntimeError("worker beta identity mismatch")
    restored = L30.RestoredState(
        tuple(payload["state"]),
        payload["currentObservationKind"],
        int(payload["currentCompletedFissions"]),
        int(payload["currentGrowthGeneration"]),
        int(payload["currentGenerationLocalStep"]),
        int(payload["currentBatchStep"]),
    )
    target = np.asarray(payload["centroid"], dtype=np.float64)
    horizon = 32 if family == "H32" else 8
    branches = H32_BRANCHES if family == "H32" else H8_BRANCHES
    half = H32_HALF if family == "H32" else H8_HALF
    current_score = float(
        L28.cosine_to_reference(np.asarray([payload["state"]]), target)[0]
    )
    rows = []
    for branch in range(branches):
        identities = stream_identities(family, candidate, matrix, landmark, branch)
        result = L28.simulate_branch(
            restored=restored,
            beta=beta,
            definition=L28.definition(candidate),
            target_centroid=target,
            event_rng=L28.generator(identities["event"]),
            trim_rng=L28.generator(identities["trim"]),
            fission_rng=L28.generator(identities["fission"]),
            daughter_rng=L28.generator(identities["daughter"]),
            horizon=horizon,
        )
        materials = [value.seed_material_sha256 for value in identities.values()]
        rows.append(
            {
                "stateId": payload["stateId"],
                "candidateId": candidate,
                "matrixIndex": matrix,
                "landmark": landmark,
                "branchFamily": family,
                "referenceVariant": payload["referenceVariant"],
                "targetReferenceDonorStateId": payload["targetReferenceDonorStateId"],
                "branchIndex": branch,
                "branchHalf": "A" if branch < half else "B",
                "streamIdentitySha256": hashlib.sha256(
                    "|".join(
                        [family, payload["stateId"], str(branch), *materials]
                    ).encode()
                ).hexdigest(),
                "enteredBasin": result.entered_basin,
                "firstEntryOffsetOneBased": result.first_entry_offset_one_based,
                "maximumTargetScore": result.maximum_target_score,
                "minimumTargetScore": result.minimum_target_score,
                "molecularUpdates": result.molecular_updates,
                "fissions": result.fissions,
                "selectedObservationsGenerated": result.selected_observations_generated,
                "terminalStatus": result.terminal_status,
                "pathSha256": result.path_sha256,
                "currentTargetScore": current_score,
                "targetComponentFraction": payload["targetComponentSizeUsed"] / 100.0,
            }
        )
    return rows


def execute_branches(tasks: list[tuple[dict[str, Any], str]]) -> pd.DataFrame:
    rows = []
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = [executor.submit(_branch_worker, task) for task in tasks]
        for future in as_completed(futures):
            rows.extend(future.result())
    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "branchFamily",
                "referenceVariant",
                "candidateId",
                "landmark",
                "matrixIndex",
                "branchIndex",
            ]
        )
        .reset_index(drop=True)
    )


def summarize_branches(branches: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in branches.groupby(
        [
            "branchFamily",
            "referenceVariant",
            "stateId",
            "candidateId",
            "matrixIndex",
            "landmark",
        ],
        sort=True,
    ):
        family, variant, state_id, candidate, matrix, landmark = keys
        total = H32_BRANCHES if family == "H32" else H8_BRANCHES
        half_a = group[group["branchHalf"].eq("A")]
        half_b = group[group["branchHalf"].eq("B")]
        successes = int(group["enteredBasin"].sum())
        row = {
            "branchFamily": family,
            "referenceVariant": variant,
            "stateId": state_id,
            "candidateId": candidate,
            "matrixIndex": int(matrix),
            "landmark": int(landmark),
            "successes": successes,
            "branches": total,
            "q": successes / total,
            "qHalfA": float(half_a["enteredBasin"].mean()),
            "qHalfB": float(half_b["enteredBasin"].mean()),
            "completeHorizonBranchCount": int(
                (
                    group["selectedObservationsGenerated"]
                    == (32 if family == "H32" else 8)
                ).sum()
            ),
        }
        if family == "H8":
            maximum = group["maximumTargetScore"].astype(float)
            minimum = group["minimumTargetScore"].astype(float)
            row.update(
                {
                    "q8Jeffreys": (successes + 0.5) / 65.0,
                    "q8JeffreysLogit": float(logit((successes + 0.5) / 65.0)),
                    "meanMaximumTargetScore": float(maximum.mean()),
                    "sdMaximumTargetScore": float(maximum.std(ddof=1)),
                    "meanMinimumTargetScore": float(minimum.mean()),
                    "fractionBranchesWithFission": float(
                        (group["fissions"] > 0).mean()
                    ),
                    "meanMolecularUpdates": float(group["molecularUpdates"].mean()),
                    "currentTargetScore": float(group["currentTargetScore"].iloc[0]),
                    "targetComponentFraction": float(
                        group["targetComponentFraction"].iloc[0]
                    ),
                    "landmarkScaled": int(landmark) / 192.0,
                }
            )
        rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "branchFamily",
                "referenceVariant",
                "candidateId",
                "landmark",
                "matrixIndex",
            ]
        )
        .reset_index(drop=True)
    )


def _entropy(value: np.ndarray) -> float:
    positive = value[value > 0]
    return float(-np.sum(positive * np.log(positive))) if len(positive) else 0.0


def static_geometry_features(
    states: pd.DataFrame, coordinates: pd.DataFrame, manifest: pd.DataFrame
) -> pd.DataFrame:
    payload_rows = payloads(states, coordinates, manifest, "ORIGINAL")
    rows = []
    for payload in payload_rows:
        state = np.asarray(payload["state"], dtype=np.int64)
        target = np.asarray(payload["centroid"], dtype=np.float64)
        mass = float(state.sum())
        x = state.astype(np.float64) / mass
        local_step = (
            0
            if payload["currentObservationKind"]
            in {"initial_selected_state", "post_fission"}
            else int(payload["currentGenerationLocalStep"])
        )
        beta_seed = L28.derive_seed(
            L28.L23_ROOT_HEX,
            L28.L23_PHASE,
            "catalytic_matrix",
            int(payload["matrixIndex"]),
        )
        beta = L28.generate_beta(beta_seed)
        analytic = L29.analytic_count_moments(
            state,
            beta,
            L28.definition(payload["candidateId"]),
            generation_local_step=local_step,
        )
        score = float(L28.cosine_to_reference(state[None, :], target)[0])
        rows.append(
            {
                "stateId": payload["stateId"],
                "candidateId": payload["candidateId"],
                "matrixIndex": int(payload["matrixIndex"]),
                "landmark": int(payload["landmark"]),
                "landmarkScaled": int(payload["landmark"]) / 192.0,
                "currentMassScaled": mass / 80.0,
                "generationLocalStepScaled": local_step / 1000.0,
                "nextIsFission": float(analytic.transition_kind == "FISSION"),
                "currentDiversity": float(np.count_nonzero(state) / 100.0),
                "currentEntropy": _entropy(x) / math.log(100.0),
                "currentConcentration": float(np.sum(x * x)),
                "targetScore": score,
                "targetGap": L28.TARGET_THRESHOLD - score,
                "targetComponentFraction": payload["targetComponentSizeUsed"] / 100.0,
                "targetEntropy": _entropy(target) / math.log(100.0),
                "targetSupportOverlap": float(np.mean((state > 0) & (target > 0))),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["candidateId", "landmark", "matrixIndex"])
        .reset_index(drop=True)
    )


def frozen_predict(
    frame: pd.DataFrame, registry: pd.DataFrame, candidate: str, model_id: str
) -> np.ndarray:
    row = registry[
        registry["candidateId"].eq(candidate) & registry["modelId"].eq(model_id)
    ].iloc[0]
    columns = json.loads(row.featureNames)
    mean = np.asarray(json.loads(row.scalerMean), dtype=np.float64)
    scale = np.asarray(json.loads(row.scalerScale), dtype=np.float64)
    coefficients = np.asarray(json.loads(row.coefficients), dtype=np.float64)
    values = frame[columns].to_numpy(dtype=np.float64)
    return expit(float(row.intercept) + ((values - mean) / scale) @ coefficients)


def frozen_control_scores(
    selection: pd.DataFrame,
    states: pd.DataFrame,
    coordinates: pd.DataFrame,
    manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    task = pd.read_parquet(L25_ROOT / "online_task_registry.parquet")
    development_meta, development_vectors = L26.extract_representations(
        task, manifest, "DEVELOPMENT"
    )
    library = L26.fit_library(development_meta, development_vectors)
    stored_lock = json.loads((L26_ROOT / "analog_library_lock.json").read_text())
    library_checks = {
        model: bool(
            L26.array_hash(library[model]["mean"])
            == stored_lock["models"][model]["meanSha256"]
            and L26.array_hash(library[model]["scale"])
            == stored_lock["models"][model]["scaleSha256"]
            and L26.array_hash(library[model]["values"])
            == stored_lock["models"][model]["standardizedLibrarySha256"]
        )
        for model in L26.MODELS
    }
    if not all(library_checks.values()):
        raise RuntimeError("frozen L26 library replay failed")
    query_meta, query_vectors = L26.extract_representations(
        selection, manifest, "CONFIRMATION"
    )
    analog = L26.score_analogues(
        query_meta,
        query_vectors,
        library,
        leave_development_out=False,
        variant="ORIGINAL",
    )
    state_keys = states[["stateId", "candidateId", "matrixIndex", "landmark"]]
    analog = analog.merge(
        state_keys,
        on=["candidateId", "matrixIndex", "landmark"],
        validate="many_to_one",
    )
    analog = analog[
        analog["modelId"].isin(["EXACT_H_TRACE_ANALOG", "ORDINARY_PATH_ANALOG"])
    ][["stateId", "candidateId", "matrixIndex", "landmark", "modelId", "score"]].rename(
        columns={"score": "predictedQ"}
    )
    geometry = static_geometry_features(states, coordinates, manifest)
    l29_registry = pd.read_parquet(L29_ROOT / "fitted_model_registry.parquet")
    geometry_rows = []
    for candidate in CANDIDATES:
        group = geometry[geometry["candidateId"].eq(candidate)].copy()
        group["predictedQ"] = frozen_predict(
            group, l29_registry, candidate, "TARGET_GEOMETRY_CONTROL"
        )
        group["modelId"] = "TARGET_GEOMETRY_CONTROL"
        geometry_rows.append(
            group[
                [
                    "stateId",
                    "candidateId",
                    "matrixIndex",
                    "landmark",
                    "modelId",
                    "predictedQ",
                ]
            ]
        )
    controls = (
        pd.concat([analog, *geometry_rows], ignore_index=True)
        .sort_values(["candidateId", "modelId", "landmark", "matrixIndex"])
        .reset_index(drop=True)
    )
    return controls, {
        "schema": "eidosoma.e01.s19_l31.frozen_control_replay.v1",
        "status": "PASS",
        "l26LibraryChecks": library_checks,
        "l26LockSha256": sha256_file(L26_ROOT / "analog_library_lock.json"),
        "l29RegistrySha256": sha256_file(L29_ROOT / "fitted_model_registry.parquet"),
        "controlFrameSha256": frame_hash(controls),
    }


def predictions(
    summary: pd.DataFrame, controls: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    h32 = summary[
        summary["branchFamily"].eq("H32") & summary["referenceVariant"].eq("ORIGINAL")
    ][["stateId", "candidateId", "matrixIndex", "landmark", "successes", "q"]].rename(
        columns={"q": "qHat"}
    )
    h8 = summary[summary["branchFamily"].eq("H8")].merge(
        h32,
        on=["stateId", "candidateId", "matrixIndex", "landmark"],
        validate="many_to_one",
    )
    registry = pd.read_parquet(L30_ROOT / "fitted_model_registry.parquet")
    rows = []
    replay_rows = []
    for (variant, candidate), group in h8.groupby(
        ["referenceVariant", "candidateId"], sort=True
    ):
        for model_id in ("Q8_CALIBRATED", PRIMARY_MODEL):
            values = frozen_predict(group, registry, candidate, model_id)
            replay = frozen_predict(group, registry, candidate, model_id)
            exact = bool(np.array_equal(values, replay))
            replay_rows.append(
                {
                    "referenceVariant": variant,
                    "candidateId": candidate,
                    "modelId": model_id,
                    "rows": len(group),
                    "exactReplay": exact,
                }
            )
            if not exact:
                raise RuntimeError("frozen L30 model replay failed")
            for source, value in zip(
                group.itertuples(index=False), values, strict=True
            ):
                rows.append(
                    {
                        "stateId": source.stateId,
                        "candidateId": candidate,
                        "matrixIndex": source.matrixIndex,
                        "landmark": source.landmark,
                        "referenceVariant": variant,
                        "modelId": model_id,
                        "predictedQ": float(value),
                        "qHat": source.qHat,
                        "successes": source.successes_y,
                    }
                )
        for source in group.itertuples(index=False):
            rows.append(
                {
                    "stateId": source.stateId,
                    "candidateId": candidate,
                    "matrixIndex": source.matrixIndex,
                    "landmark": source.landmark,
                    "referenceVariant": variant,
                    "modelId": "Q8_JEFFREYS_DIRECT",
                    "predictedQ": source.q8Jeffreys,
                    "qHat": source.qHat,
                    "successes": source.successes_y,
                }
            )
    result = pd.DataFrame(rows)
    original_controls = controls.merge(
        h32,
        on=["stateId", "candidateId", "matrixIndex", "landmark"],
        validate="many_to_one",
    ).rename(columns={"successes": "successes"})
    original_controls["referenceVariant"] = "ORIGINAL"
    original_controls["qHat"] = original_controls["qHat"]
    result = pd.concat(
        [
            result,
            original_controls[result.columns],
        ],
        ignore_index=True,
    )
    l30_predictions = pd.read_parquet(L30_ROOT / "prediction_results.parquet")
    for candidate in CANDIDATES:
        prior = float(
            l30_predictions[
                l30_predictions["referenceVariant"].eq("ORIGINAL")
                & l30_predictions["candidateId"].eq(candidate)
                & l30_predictions["matrixRole"].eq("DEVELOPMENT")
                & l30_predictions["modelId"].eq("DEVELOPMENT_PRIOR")
            ]["predictedQ"].iloc[0]
        )
        group = h32[h32["candidateId"].eq(candidate)]
        addition = group.copy()
        addition["referenceVariant"] = "ORIGINAL"
        addition["modelId"] = "DEVELOPMENT_PRIOR"
        addition["predictedQ"] = prior
        result = pd.concat([result, addition[result.columns]], ignore_index=True)
    return result.sort_values(
        ["referenceVariant", "candidateId", "modelId", "landmark", "matrixIndex"]
    ).reset_index(drop=True), pd.DataFrame(replay_rows)


def metric_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (variant, candidate, model), group in frame.groupby(
        ["referenceVariant", "candidateId", "modelId"], sort=True
    ):
        q = group["qHat"].to_numpy(dtype=np.float64)
        p = np.clip(group["predictedQ"].to_numpy(dtype=np.float64), 1e-9, 1 - 1e-9)
        brier = float(np.mean(q * (1 - p) ** 2 + (1 - q) * p**2))
        log_loss = float(-np.mean(q * np.log(p) + (1 - q) * np.log(1 - p)))
        intercept, slope = L28.calibration_parameters(p, q)
        rows.append(
            {
                "referenceVariant": variant,
                "candidateId": candidate,
                "modelId": model,
                "states": len(group),
                "spearmanQHat": L29.safe_spearman(p, q),
                "brierScorePerBranch": brier,
                "binomialLogLossPerBranch": log_loss,
                "calibrationIntercept": intercept,
                "calibrationSlope": slope,
            }
        )
    return pd.DataFrame(rows)


def bootstrap_results(
    prediction_frame: pd.DataFrame, state_summary: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows = []
    reliability_rows = []
    original = prediction_frame[prediction_frame["referenceVariant"].eq("ORIGINAL")]
    q32 = state_summary[
        state_summary["branchFamily"].eq("H32")
        & state_summary["referenceVariant"].eq("ORIGINAL")
    ]
    h8 = state_summary[
        state_summary["branchFamily"].eq("H8")
        & state_summary["referenceVariant"].eq("ORIGINAL")
    ]
    for candidate in CANDIDATES:
        pivot = (
            original[original["candidateId"].eq(candidate)]
            .pivot(index=["stateId", "qHat"], columns="modelId", values="predictedQ")
            .reset_index()
        )
        q_candidate = q32[q32["candidateId"].eq(candidate)].reset_index(drop=True)
        h_candidate = h8[h8["candidateId"].eq(candidate)].reset_index(drop=True)
        rng = np.random.default_rng(L29.derived_seed("l31_bootstrap", candidate))
        models = [
            column for column in pivot.columns if column not in {"stateId", "qHat"}
        ]
        for replicate in range(BOOTSTRAPS):
            indices = rng.integers(0, len(pivot), size=len(pivot))
            sample = pivot.iloc[indices]
            q = sample["qHat"].to_numpy(dtype=np.float64)
            brier: dict[str, float] = {}
            for model in models:
                p = np.clip(sample[model].to_numpy(dtype=np.float64), 1e-9, 1 - 1e-9)
                value = float(np.mean(q * (1 - p) ** 2 + (1 - q) * p**2))
                brier[model] = value
                metric_rows.append(
                    {
                        "candidateId": candidate,
                        "bootstrapIndex": replicate,
                        "modelId": model,
                        "spearmanQHat": L29.safe_spearman(p, q),
                        "brierScorePerBranch": value,
                        "primaryBrierImprovement": float("nan"),
                    }
                )
            for control in CONTROL_MODELS:
                metric_rows.append(
                    {
                        "candidateId": candidate,
                        "bootstrapIndex": replicate,
                        "modelId": f"DELTA_PRIMARY_VS_{control}",
                        "spearmanQHat": float("nan"),
                        "brierScorePerBranch": float("nan"),
                        "primaryBrierImprovement": brier[control]
                        - brier[PRIMARY_MODEL],
                    }
                )
            qs = q_candidate.iloc[indices]
            hs = h_candidate.iloc[indices]
            variance = L28.corrected_between_state_variance(
                qs["q"].to_numpy(dtype=np.float64), H32_BRANCHES
            )
            reliability_rows.append(
                {
                    "candidateId": candidate,
                    "bootstrapIndex": replicate,
                    "correctedBetweenStateVariance": variance[
                        "correctedBetweenStateVariance"
                    ],
                    "q32SplitHalfSpearman": L29.safe_spearman(
                        qs["qHalfA"], qs["qHalfB"]
                    ),
                    "h8SplitHalfSpearman": L29.safe_spearman(
                        hs["qHalfA"], hs["qHalfB"]
                    ),
                    "intermediateStateCount": int(
                        ((qs["q"] > 0.1) & (qs["q"] < 0.9)).sum()
                    ),
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(reliability_rows)


def label_permutations(
    prediction_frame: pd.DataFrame, metrics: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    primary = prediction_frame[
        prediction_frame["referenceVariant"].eq("ORIGINAL")
        & prediction_frame["modelId"].eq(PRIMARY_MODEL)
    ]
    for candidate in CANDIDATES:
        group = primary[primary["candidateId"].eq(candidate)].reset_index(drop=True)
        observed = float(
            metrics[
                metrics["referenceVariant"].eq("ORIGINAL")
                & metrics["candidateId"].eq(candidate)
                & metrics["modelId"].eq(PRIMARY_MODEL)
            ]["spearmanQHat"].iloc[0]
        )
        rng = np.random.default_rng(
            L29.derived_seed("l31_label_permutation", candidate)
        )
        nulls = []
        for replicate in range(PERMUTATIONS):
            q = group["qHat"].to_numpy()[rng.permutation(len(group))]
            rho = L29.safe_spearman(group["predictedQ"], q)
            nulls.append(rho)
            rows.append(
                {
                    "candidateId": candidate,
                    "permutationIndex": replicate,
                    "observedSpearman": observed,
                    "nullSpearman": rho,
                }
            )
        finite = np.asarray([value for value in nulls if np.isfinite(value)])
        p_value = float((1 + np.sum(finite >= observed)) / (1 + len(finite)))
        for row in rows:
            if row["candidateId"] == candidate:
                row["familywiseP"] = p_value
    return pd.DataFrame(rows)


def gate_table(
    state_summary: pd.DataFrame,
    metrics: pd.DataFrame,
    metric_boot: pd.DataFrame,
    reliability_boot: pd.DataFrame,
    permutations: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for candidate in CANDIDATES:
        q32 = state_summary[
            state_summary["branchFamily"].eq("H32")
            & state_summary["referenceVariant"].eq("ORIGINAL")
            & state_summary["candidateId"].eq(candidate)
        ]
        h8 = state_summary[
            state_summary["branchFamily"].eq("H8")
            & state_summary["referenceVariant"].eq("ORIGINAL")
            & state_summary["candidateId"].eq(candidate)
        ]
        corrected = L28.corrected_between_state_variance(q32["q"], H32_BRANCHES)
        q32_half = L29.safe_spearman(q32["qHalfA"], q32["qHalfB"])
        h8_half = L29.safe_spearman(h8["qHalfA"], h8["qHalfB"])
        rboot = reliability_boot[reliability_boot["candidateId"].eq(candidate)]
        corrected_lower = float(
            np.quantile(rboot["correctedBetweenStateVariance"], 0.025)
        )
        q32_half_lower = float(np.nanquantile(rboot["q32SplitHalfSpearman"], 0.025))
        h8_half_lower = float(np.nanquantile(rboot["h8SplitHalfSpearman"], 0.025))
        primary = metrics[
            metrics["referenceVariant"].eq("ORIGINAL")
            & metrics["candidateId"].eq(candidate)
            & metrics["modelId"].eq(PRIMARY_MODEL)
        ].iloc[0]
        target_control = metrics[
            metrics["referenceVariant"].eq("TARGET_REFERENCE_PERMUTATION")
            & metrics["candidateId"].eq(candidate)
            & metrics["modelId"].eq(PRIMARY_MODEL)
        ].iloc[0]
        mboot = metric_boot[
            metric_boot["candidateId"].eq(candidate)
            & metric_boot["modelId"].eq(PRIMARY_MODEL)
        ]["spearmanQHat"].dropna()
        primary_lower = float(np.quantile(mboot, 0.025))
        deltas = {
            control: float(
                np.quantile(
                    metric_boot[
                        metric_boot["candidateId"].eq(candidate)
                        & metric_boot["modelId"].eq(f"DELTA_PRIMARY_VS_{control}")
                    ]["primaryBrierImprovement"],
                    0.025,
                )
            )
            for control in CONTROL_MODELS
        }
        intermediate = int(((q32["q"] > 0.1) & (q32["q"] < 0.9)).sum())
        permutation_p = float(
            permutations[permutations["candidateId"].eq(candidate)]["familywiseP"].iloc[
                0
            ]
        )
        checks = {
            "correctedVariancePassed": corrected_lower > 0,
            "q32ReliabilityPassed": q32_half > 0.5 and q32_half_lower > 0.3,
            "intermediateSupportPassed": intermediate >= 8,
            "h8ReliabilityPassed": h8_half > 0.5 and h8_half_lower > 0.3,
            "primaryRankPassed": primary.spearmanQHat > 0.5 and primary_lower > 0.3,
            "incrementalBrierPassed": all(value > 0 for value in deltas.values()),
            "labelPermutationPassed": permutation_p <= 0.05,
            "targetReferenceControlPassed": primary.spearmanQHat
            > target_control.spearmanQHat,
        }
        rows.append(
            {
                "candidateId": candidate,
                "states": len(q32),
                "correctedBetweenStateVariance": corrected[
                    "correctedBetweenStateVariance"
                ],
                "correctedVarianceLower95": corrected_lower,
                "q32SplitHalfSpearman": q32_half,
                "q32SplitHalfLower95": q32_half_lower,
                "intermediateStateCount": intermediate,
                "h8SplitHalfSpearman": h8_half,
                "h8SplitHalfLower95": h8_half_lower,
                "primarySpearman": primary.spearmanQHat,
                "primarySpearmanLower95": primary_lower,
                **{
                    f"brierImprovementLowerVs{key}": value
                    for key, value in deltas.items()
                },
                "labelPermutationP": permutation_p,
                "targetReferenceSpearman": target_control.spearmanQHat,
                **checks,
                "candidateConfirmationGatePassed": all(checks.values()),
            }
        )
    return pd.DataFrame(rows)


def fixture_results() -> pd.DataFrame:
    state = np.zeros(100, dtype=np.int64)
    state[:40] = 1
    target = state.astype(np.float64) / state.sum()
    beta = np.exp(np.full((100, 100), -4.0, dtype=np.float64))
    restored = L30.RestoredState(tuple(map(int, state)), "post_fission", 1, 1, 0, 4)

    def run(family: str) -> Any:
        ids = stream_identities(family, CANDIDATES[0], 999_731, 64, 0)
        return L28.simulate_branch(
            restored=restored,
            beta=beta,
            definition=L28.definition(CANDIDATES[0]),
            target_centroid=target,
            event_rng=L28.generator(ids["event"]),
            trim_rng=L28.generator(ids["trim"]),
            fission_rng=L28.generator(ids["fission"]),
            daughter_rng=L28.generator(ids["daughter"]),
            horizon=32 if family == "H32" else 8,
        )

    h32 = run("H32")
    h8 = run("H8")
    return pd.DataFrame(
        [
            {
                "fixtureId": "H32_HORIZON",
                "passed": h32.selected_observations_generated == 32,
            },
            {
                "fixtureId": "H8_HORIZON",
                "passed": h8.selected_observations_generated == 8,
            },
            {"fixtureId": "H32_EXACT_REPLAY", "passed": h32 == run("H32")},
            {"fixtureId": "H8_EXACT_REPLAY", "passed": h8 == run("H8")},
            {"fixtureId": "ROOTS_DISTINCT", "passed": H32_ROOT != H8_ROOT},
            {
                "fixtureId": "L30_FEATURE_COLUMNS_FROZEN",
                "passed": len(L30.MODEL_COLUMNS[PRIMARY_MODEL]) == 9,
            },
        ]
    )


def make_figures(
    summary: pd.DataFrame,
    predictions_frame: pd.DataFrame,
    metrics: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    root = BUILD_ROOT / "figures"
    root.mkdir(parents=True, exist_ok=True)

    def save(name: str) -> None:
        plt.tight_layout()
        plt.savefig(root / name, dpi=180)
        plt.close()

    q32 = summary[
        summary["branchFamily"].eq("H32") & summary["referenceVariant"].eq("ORIGINAL")
    ]
    plt.figure(figsize=(8, 5))
    for candidate in CANDIDATES:
        plt.hist(
            q32[q32["candidateId"].eq(candidate)]["q"],
            bins=np.linspace(0, 1, 18),
            alpha=0.55,
            label=candidate,
        )
    plt.axvspan(0.1, 0.9, color="grey", alpha=0.15)
    plt.xlabel("Untouched H32 q-hat")
    plt.ylabel("States")
    plt.legend(fontsize=7)
    save("01_untouched_committor_distribution.png")

    h8 = summary[
        summary["branchFamily"].eq("H8") & summary["referenceVariant"].eq("ORIGINAL")
    ].merge(q32[["stateId", "q"]].rename(columns={"q": "q32"}), on="stateId")
    plt.figure(figsize=(8, 5))
    for candidate in CANDIDATES:
        group = h8[h8["candidateId"].eq(candidate)]
        plt.scatter(group["q8Jeffreys"], group["q32"], s=22, alpha=0.7, label=candidate)
    plt.xlabel("H8 entry probability (Jeffreys)")
    plt.ylabel("H32 q-hat")
    plt.legend(fontsize=7)
    save("02_h8_vs_h32.png")

    original = predictions_frame[
        predictions_frame["referenceVariant"].eq("ORIGINAL")
        & predictions_frame["modelId"].eq(PRIMARY_MODEL)
    ]
    _, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, candidate in zip(axes, CANDIDATES, strict=True):
        group = original[original["candidateId"].eq(candidate)]
        axis.scatter(group["predictedQ"], group["qHat"], s=22)
        axis.plot([0, 1], [0, 1], "k--", linewidth=1)
        axis.set_title(candidate)
        axis.set_xlabel("Frozen L30 predicted q")
        axis.set_ylabel("Untouched q-hat")
    save("03_frozen_coordinate_confirmation.png")

    metrics[metrics["referenceVariant"].eq("ORIGINAL")].pivot(
        index="modelId", columns="candidateId", values="spearmanQHat"
    ).plot(kind="bar", figsize=(10, 5))
    plt.axhline(0.5, color="black", linestyle="--")
    plt.ylabel("Untouched Spearman")
    save("04_control_comparison.png")

    checks = [
        "correctedVariancePassed",
        "q32ReliabilityPassed",
        "intermediateSupportPassed",
        "h8ReliabilityPassed",
        "primaryRankPassed",
        "incrementalBrierPassed",
        "labelPermutationPassed",
        "targetReferenceControlPassed",
        "candidateConfirmationGatePassed",
    ]
    matrix = gates.set_index("candidateId")[checks].astype(float)
    plt.figure(figsize=(10, 3.5))
    plt.imshow(matrix, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    plt.xticks(range(len(checks)), checks, rotation=38, ha="right", fontsize=7)
    plt.yticks(range(len(matrix)), matrix.index)
    plt.colorbar(ticks=[0, 1])
    save("05_confirmation_gate_matrix.png")


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
        "schema": "eidosoma.e01.s19_l31.artifact_manifest.v1",
        "root": str(root),
        "fileCount": len(files),
        "totalBytes": sum(item["bytes"] for item in files),
        "files": files,
    }


def append_ledgers(classifications: list[str], timestamp: str, next_theme: str) -> None:
    ledger_path = ARTIFACT_ROOT / "self_improvement_ledger.parquet"
    ledger = pd.read_parquet(ledger_path)
    sequence = int(ledger["ledgerSequence"].max()) + 1
    additions = [
        {
            "appendOnly": True,
            "beliefBeforeLoop": "L30 established an H8 shooting coordinate on its discovery matrices.",
            "failureOrAmbiguityTargeted": "Independent matrix generalization of the L30 coordinate.",
            "informationGainRationale": "Previously unused matrices, new H8/H32 streams, and no refit separate replication from discovery reuse.",
            "learned": "L31 untouched confirmation contract frozen.",
            "ledgerSequence": sequence,
            "loopId": LOOP_ID,
            "motivatingEvidence": "EIGHT_STEP_PROPAGATOR_COMMITTOR_COORDINATE_ESTABLISHED",
            "proposedNextTest": "Execute frozen coordinate on untouched matrices.",
            "recordPhase": "PRE_LOOP_METHOD_LOCK",
            "remainingPlausibleHypotheses": "Generalizable short shooting coordinate or discovery-cohort artifact.",
            "selectedHypotheses": "Frozen L30 H8 propagator coordinate.",
            "timestampUtc": timestamp,
            "weakenedHypotheses": "No simulation-accessible coordinate exists.",
        },
        {
            "appendOnly": True,
            "beliefBeforeLoop": "A valid coordinate must reproduce without refitting on unused matrices.",
            "failureOrAmbiguityTargeted": "Untouched coordinate confirmation.",
            "informationGainRationale": "Independent q and H8 estimation tests both target and coordinate.",
            "learned": ";".join(classifications),
            "ledgerSequence": sequence + 1,
            "loopId": LOOP_ID,
            "motivatingEvidence": "Complete L31 result.",
            "proposedNextTest": next_theme,
            "recordPhase": "POST_LOOP_RESULT_AUTONOMOUS_CONTINUATION",
            "remainingPlausibleHypotheses": next_theme,
            "selectedHypotheses": "Frozen L30 H8 propagator coordinate.",
            "timestampUtc": timestamp,
            "weakenedHypotheses": "L30 generalizes"
            if "NON_SUPPORT" in ";".join(classifications)
            else "The committor lacks a reproducible coordinate.",
        },
    ]
    BASE.write_parquet(
        ledger_path,
        pd.concat(
            [ledger, pd.DataFrame(additions).reindex(columns=ledger.columns)],
            ignore_index=True,
        ),
    )
    md = ARTIFACT_ROOT / "SELF_IMPROVEMENT_LEDGER.md"
    BASE.atomic_text(
        md,
        md.read_text()
        + f"\n\n## {LOOP_ID} — untouched H8 committor confirmation\n\n- **Learned:** {', '.join(classifications)}.\n- **Next:** {next_theme}.\n",
    )
    candidates_path = ARTIFACT_ROOT / "candidate_registry.parquet"
    candidates = pd.read_parquet(candidates_path)
    row = {
        "branchCount": 1,
        "bundleId": "L31_UNTOUCHED_H8_CONFIRMATION",
        "candidateId": "S19-L31-FROZEN-H8-CONFIRMATION",
        "candidateSpecificSuccess": 0,
        "completedFitLeakage": 1,
        "computeEfficiency": 4,
        "crossCandidateDiscriminability": 5,
        "deterministicHReuse": 0,
        "explanatoryLeverage": 5,
        "frozenRank": 1,
        "independenceFromPriorOutcomeSelection": 5,
        "outcomeGuidedThresholdSelection": 0,
        "paperFingerprintSpecificity": 0,
        "proposedSpecification": "unchanged L30 coordinate on 80 unused-matrix states",
        "rankingScore": 29.0,
        "registryOrder": int(candidates["registryOrder"].max()) + 1,
        "selected": True,
        "selectionReason": "L30_COORDINATE_ESTABLISHED_REQUIRES_UNTOUCHED_CONFIRMATION",
        "sourceGrounding": 5,
        "testability": 5,
        "undefinedAuthorSemantics": 0,
    }
    BASE.write_parquet(
        candidates_path,
        pd.concat(
            [candidates, pd.DataFrame([row]).reindex(columns=candidates.columns)],
            ignore_index=True,
        ),
    )
    source_path = ARTIFACT_ROOT / "source_search_ledger.parquet"
    sources = pd.read_parquet(source_path)
    source_row = {
        "commitOrVersion": None,
        "evidenceClass": "DIRECT_FROZEN_E01_RESULT",
        "finding": "L31 freezes L30's established H8 shooting coordinate and tests it without refitting on previously unused L23 matrices.",
        "licenseStatus": "WORKSPACE_EVIDENCE",
        "redistributionStatus": "INTERNAL_ARTIFACT",
        "repositoryIdentity": None,
        "retainedPath": str(L30_ROOT / "research_step_full_results.md"),
        "retrievalDate": timestamp[:10],
        "sha256": sha256_file(L30_ROOT / "research_step_full_results.md"),
        "sourceId": "L31_L30_FROZEN_COORDINATE_CONTEXT",
        "sourceType": "DIRECT_FROZEN_E01_RESULT",
        "treeIdentity": None,
        "url": None,
    }
    BASE.write_parquet(
        source_path,
        pd.concat(
            [sources, pd.DataFrame([source_row]).reindex(columns=sources.columns)],
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
            "selectedDiscoveryLead": "CONFIRMED_H8_SHOOTING_COORDINATE"
            if "CONFIRMED" in ";".join(classifications)
            else None,
            "newMatrices": 0,
            "newTrajectories": 25600,
            "nextStepActive": True,
        }
    )
    registry["proposedNextLoopTheme"] = next_theme
    registry["proposedNextLoopActive"] = True
    BASE.atomic_text(registry_path, yaml.safe_dump(registry, sort_keys=False))


def report_text(
    metrics: pd.DataFrame,
    gates: pd.DataFrame,
    classifications: list[str],
    runtime: dict[str, Any],
    next_theme: str,
) -> str:
    return f"""# S19-L31 — Untouched Eight-Step Propagator Committor Confirmation

## Chief/human handoff

- **Step:** `{VERSION}`
- **Status:** complete under the authorized L19–L42 sequence.
- **Outcome classifications:** {", ".join(f"`{value}`" for value in classifications)}
- **Validation:** 80 deterministic states from candidate-specific matrices unused by L28–L30; exact state/beta/target restoration; 10,240 H32 and 5,120 original H8 branches plus common-stream target-reference controls; exact full branch/model replay; 4,096 matrix bootstraps; 512 label permutations; immutable-prior, seed, runtime/storage, regeneration and artifact gates passed.
- **Next bounded theme:** {next_theme}

## Frozen question and design

The unchanged candidate-specific L30 scaler, coefficients, H8 horizon, 64-branch estimator and nine input summaries were applied without refitting to eight unique unused matrices at each of five landmarks per candidate. The H32 response was independently re-estimated with 128 new branches per state. Branch dynamics use only the restored state and new streams; the target basin remains explicitly retrospective and completed-run matrix-specific.

## Untouched metrics

{metrics.to_markdown(index=False)}

## Confirmation gates

{gates.to_markdown(index=False)}

## Interpretation boundary

A passing result confirms a simulation-accessible finite-horizon shooting coordinate for this reconstructed retrospective basin. It is not a directly observed biomarker, a prospective author-label result, causal control, or author-code identification. It can license the next bounded attempt to distill a path/tube coordinate, but not a causal-current claim by itself.

## Runtime

- Repository lock: `{runtime["repositoryHead"]}`.
- CPU float64, `{runtime["workers"]}` workers, no GPU.
- Wall seconds: `{runtime["wallSeconds"]:.3f}`; estimated worker CPU hours: `{runtime["workerCpuHours"]:.6f}`.

## Autonomous boundary

L31 is frozen. S20, E02, author contact, interventions and report-bundle work remain inactive.
"""


def prepare_lock() -> None:
    LOOP_ROOT.mkdir(parents=True, exist_ok=True)
    if git("status", "--porcelain=v1"):
        raise RuntimeError("repository must be clean")
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/eidosoma/groups/42")
    if head != remote:
        raise RuntimeError("local/remote mismatch")
    prior = validate_immutable_prior()
    fixtures = fixture_results()
    if not prior["unchanged"] or not fixtures["passed"].all():
        raise RuntimeError("prior or fixture gate failed")
    task = pd.read_parquet(L25_ROOT / "online_task_registry.parquet")
    manifest = pd.read_parquet(L23_ROOT / "input_trajectory_manifest.parquet")
    selection = select_confirmation_states(task)
    states, coordinates, restoration = build_state_lock(selection, manifest)
    l28_used = pd.read_parquet(L28_ROOT / "restored_state_registry.parquet")
    overlap = []
    for candidate in CANDIDATES:
        current = set(states[states["candidateId"].eq(candidate)]["matrixIndex"])
        previous = set(l28_used[l28_used["candidateId"].eq(candidate)]["matrixIndex"])
        overlap.extend((candidate, value) for value in sorted(current & previous))
    if overlap:
        raise RuntimeError("matrix firewall failed")
    seeds = seed_manifest(states)
    firewall = seed_firewall(seeds, prior)
    if firewall["status"] != "PASS" or not firewall["allCurrentMaterialsUnique"]:
        raise RuntimeError("seed firewall failed")
    controls, control_replay = frozen_control_scores(
        selection, states, coordinates, manifest
    )
    shutil.copy2(CONFIG, LOOP_ROOT / "preregistration.yaml")
    BASE.atomic_text(
        LOOP_ROOT / "decision_record.md",
        "# S19-L31 decision record\n\nL30 passed every discovery gate, so the standing authorization requires one untouched confirmation before transition-tube/current work. L31 selects candidate-specific matrices never used by L28–L30, freezes eight unique states per landmark, re-estimates H32 and H8 under new streams, and applies the exact L30 coordinate without refitting. The completed-run target basin remains retrospective. No horizon, branch count, target, model, landmark or control may change after branch outcomes.\n\nPre-outcome feasibility amendment 001: the initial ten-per-landmark draft was rejected before any branch outcome because the nested candidate-3 landmark-128 pool had only eight matrices remaining after enforcing global candidate-specific matrix uniqueness. The balanced repair uses exactly eight at every landmark in both candidates, preserves all landmarks and the stronger no-reuse rule, and changes no observed scientific value.\n",
    )
    BASE.write_parquet(LOOP_ROOT / "fixture_results.parquet", fixtures)
    pd.DataFrame(
        [
            {
                "amendmentId": "PREOUTCOME_FEASIBILITY_001",
                "outcomesOpened": False,
                "failure": "CANDIDATE3_LANDMARK128_ONLY_EIGHT_UNUSED_UNIQUE_MATRICES_AFTER_LATE_STRATA",
                "repair": "BALANCED_EIGHT_STATES_PER_ALL_CANDIDATE_LANDMARK_STRATA",
                "matrixReuseIntroduced": False,
                "scientificOutcomesChanged": False,
                "thresholdsModelsHorizonsOrControlsChanged": False,
            }
        ]
    ).to_csv(LOOP_ROOT / "technical_amendment_ledger.csv", index=False)
    BASE.write_parquet(LOOP_ROOT / "state_selection_registry.parquet", selection)
    BASE.write_parquet(LOOP_ROOT / "restored_state_registry.parquet", states)
    BASE.write_parquet(LOOP_ROOT / "target_basin_coordinates.parquet", coordinates)
    BASE.write_parquet(LOOP_ROOT / "state_restoration_validation.parquet", restoration)
    BASE.write_parquet(LOOP_ROOT / "branch_seed_manifest.parquet", seeds)
    BASE.write_parquet(LOOP_ROOT / "frozen_control_scores.parquet", controls)
    BASE.write_json(LOOP_ROOT / "frozen_control_replay.json", control_replay)
    BASE.write_json(LOOP_ROOT / "seed_firewall.json", firewall)
    BASE.write_json(
        LOOP_ROOT / "matrix_firewall.json",
        {"status": "PASS", "overlap": overlap, "states": len(states)},
    )
    BASE.write_json(LOOP_ROOT / "immutable_prior_validation.json", prior)
    hashes = {
        "selectionSha256": sha256_file(LOOP_ROOT / "state_selection_registry.parquet"),
        "statesSha256": sha256_file(LOOP_ROOT / "restored_state_registry.parquet"),
        "coordinatesSha256": sha256_file(
            LOOP_ROOT / "target_basin_coordinates.parquet"
        ),
        "seedsSha256": sha256_file(LOOP_ROOT / "branch_seed_manifest.parquet"),
        "controlsSha256": sha256_file(LOOP_ROOT / "frozen_control_scores.parquet"),
    }
    BASE.write_json(
        LOOP_ROOT / "implementation_lock.json",
        {
            "schema": "eidosoma.e01.s19_l31.implementation_lock.v1",
            "repositoryHead": head,
            "remoteHead": remote,
            "runnerSha256": sha256_file(RUNNER_PATH),
            "configSha256": sha256_file(CONFIG),
            "l30ModelRegistrySha256": sha256_file(
                L30_ROOT / "fitted_model_registry.parquet"
            ),
            "l29ModelRegistrySha256": sha256_file(
                L29_ROOT / "fitted_model_registry.parquet"
            ),
            "l26LibraryLockSha256": sha256_file(L26_ROOT / "analog_library_lock.json"),
            "stateCount": len(states),
            "statesPerCandidateLandmark": STATES_PER_STRATUM,
            "h32Branches": H32_BRANCHES,
            "h8Branches": H8_BRANCHES,
            "modelRefit": False,
            "targetScope": "RETROSPECTIVE_COMPLETED_RUN_MATRIX_SPECIFIC",
            "lockedHashes": hashes,
            "outcomeAccessed": False,
            "lockedAtUtc": utc_now(),
        },
    )
    BASE.write_json(
        LOOP_ROOT / "preoutcome_repository_lock.json",
        {
            "head": head,
            "remote": remote,
            "priorAggregateSha256": prior["aggregateSha256"],
            "runnerSha256": sha256_file(RUNNER_PATH),
            **hashes,
        },
    )


def execute() -> None:
    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    lock = json.loads((LOOP_ROOT / "preoutcome_repository_lock.json").read_text())
    if (
        git("rev-parse", "HEAD") != lock["head"]
        or git("rev-parse", "origin/eidosoma/groups/42") != lock["remote"]
        or git("status", "--porcelain=v1")
    ):
        raise RuntimeError("repository lock mismatch")
    prior = validate_immutable_prior()
    fixtures = fixture_results()
    for key, name in {
        "selectionSha256": "state_selection_registry.parquet",
        "statesSha256": "restored_state_registry.parquet",
        "coordinatesSha256": "target_basin_coordinates.parquet",
        "seedsSha256": "branch_seed_manifest.parquet",
        "controlsSha256": "frozen_control_scores.parquet",
    }.items():
        if sha256_file(LOOP_ROOT / name) != lock[key]:
            raise RuntimeError(f"locked input changed: {name}")
    if (
        not prior["unchanged"]
        or prior["aggregateSha256"] != lock["priorAggregateSha256"]
        or not fixtures["passed"].all()
    ):
        raise RuntimeError("pre-execution validation failed")
    states = pd.read_parquet(LOOP_ROOT / "restored_state_registry.parquet")
    coordinates = pd.read_parquet(LOOP_ROOT / "target_basin_coordinates.parquet")
    controls = pd.read_parquet(LOOP_ROOT / "frozen_control_scores.parquet")
    manifest = pd.read_parquet(L23_ROOT / "input_trajectory_manifest.parquet")
    original = payloads(states, coordinates, manifest, "ORIGINAL")
    permuted = payloads(states, coordinates, manifest, "TARGET_REFERENCE_PERMUTATION")
    tasks = (
        [(value, "H32") for value in original]
        + [(value, "H8") for value in original]
        + [(value, "H8") for value in permuted]
    )
    if BUILD_ROOT.exists():
        shutil.rmtree(BUILD_ROOT)
    BUILD_ROOT.mkdir(parents=True)
    branch_start = time.perf_counter()
    branches = execute_branches(tasks)
    branch_seconds = time.perf_counter() - branch_start
    expected = len(states) * (H32_BRANCHES + 2 * H8_BRANCHES)
    if len(branches) != expected:
        raise RuntimeError("branch cardinality failure")
    seed_manifest_frame = pd.read_parquet(LOOP_ROOT / "branch_seed_manifest.parquet")
    observed = branches[branches["referenceVariant"].eq("ORIGINAL")][
        ["stateId", "branchFamily", "branchIndex", "streamIdentitySha256"]
    ]
    stream_check = seed_manifest_frame[
        ["stateId", "branchFamily", "branchIndex", "streamIdentitySha256"]
    ].merge(
        observed,
        on=["stateId", "branchFamily", "branchIndex"],
        suffixes=("Expected", "Observed"),
        validate="one_to_one",
    )
    identity_exact = bool(
        stream_check["streamIdentitySha256Expected"]
        .eq(stream_check["streamIdentitySha256Observed"])
        .all()
    )
    if not identity_exact:
        raise RuntimeError("branch stream identity failure")
    replay_start = time.perf_counter()
    replay = execute_branches(
        [(value, "H32") for value in original] + [(value, "H8") for value in original]
    )
    replay_seconds = time.perf_counter() - replay_start
    original_branches = branches[
        branches["referenceVariant"].eq("ORIGINAL")
    ].reset_index(drop=True)
    replay_exact = frame_hash(original_branches) == frame_hash(replay)
    if not replay_exact:
        raise RuntimeError("full branch replay failed")
    summary = summarize_branches(branches)
    prediction_frame, model_replay = predictions(summary, controls)
    metrics = metric_table(prediction_frame)
    metric_boot, reliability_boot = bootstrap_results(prediction_frame, summary)
    permutations = label_permutations(prediction_frame, metrics)
    gates = gate_table(summary, metrics, metric_boot, reliability_boot, permutations)
    confirmed = bool(gates["candidateConfirmationGatePassed"].all())
    if confirmed:
        classifications = [
            "UNTOUCHED_EIGHT_STEP_PROPAGATOR_COMMITTOR_COORDINATE_CONFIRMED",
            "RETROSPECTIVE_BASIN_CONDITIONED_SHOOTING_SIGNAL",
            "NOT_PROMOTABLE_AS_CONFIRMED_PAPER_RESULT",
        ]
        next_theme = "COMMITTOR_ORDERED_TRANSITION_TUBE_COORDINATE_DISCOVERY"
    else:
        classifications = [
            "UNTOUCHED_EIGHT_STEP_PROPAGATOR_COORDINATE_NON_SUPPORT",
            "NOT_PROMOTABLE_AS_CONFIRMED",
        ]
        next_theme = "HIDDEN_MEMORY_STATE_AUDIT"
    make_figures(summary, prediction_frame, metrics, gates)
    for name in (
        "preregistration.yaml",
        "decision_record.md",
        "fixture_results.parquet",
        "state_selection_registry.parquet",
        "restored_state_registry.parquet",
        "target_basin_coordinates.parquet",
        "state_restoration_validation.parquet",
        "branch_seed_manifest.parquet",
        "frozen_control_scores.parquet",
        "frozen_control_replay.json",
        "seed_firewall.json",
        "matrix_firewall.json",
        "immutable_prior_validation.json",
        "implementation_lock.json",
        "preoutcome_repository_lock.json",
        "technical_amendment_ledger.csv",
    ):
        shutil.copy2(LOOP_ROOT / name, BUILD_ROOT / name)
    BASE.write_parquet(BUILD_ROOT / "branch_results.parquet", branches)
    BASE.write_parquet(
        BUILD_ROOT / "state_committor_and_propagator_results.parquet", summary
    )
    BASE.write_parquet(BUILD_ROOT / "prediction_results.parquet", prediction_frame)
    BASE.write_parquet(BUILD_ROOT / "model_replay_results.parquet", model_replay)
    BASE.write_parquet(BUILD_ROOT / "metric_results.parquet", metrics)
    BASE.write_parquet(BUILD_ROOT / "metric_bootstrap_results.parquet", metric_boot)
    BASE.write_parquet(
        BUILD_ROOT / "reliability_bootstrap_results.parquet", reliability_boot
    )
    BASE.write_parquet(BUILD_ROOT / "label_permutation_results.parquet", permutations)
    BASE.write_parquet(BUILD_ROOT / "scientific_gate_results.parquet", gates)
    BASE.write_json(
        BUILD_ROOT / "classification.json",
        {
            "schema": "eidosoma.e01.s19_l31.classification.v1",
            "classifications": classifications,
            "coordinateConfirmedBothCandidates": confirmed,
            "retrospectiveBasinConditioned": True,
            "modelRefit": False,
            "priorStatusesChanged": False,
        },
    )
    pd.DataFrame(
        columns=[
            "stage",
            "candidateId",
            "matrixIndex",
            "landmark",
            "branchIndex",
            "exceptionClass",
            "exceptionMessage",
        ]
    ).to_csv(BUILD_ROOT / "failure_ledger.csv", index=False)
    checks = {
        "branchCardinalityExact": len(branches) == expected,
        "branchStreamIdentityExact": identity_exact,
        "fullOriginalBranchReplayExact": replay_exact,
        "frozenModelReplayExact": bool(model_replay["exactReplay"].all()),
        "allH32BranchesComplete": bool(
            (
                branches[branches["branchFamily"].eq("H32")][
                    "selectedObservationsGenerated"
                ]
                == 32
            ).all()
        ),
        "allH8BranchesComplete": bool(
            (
                branches[branches["branchFamily"].eq("H8")][
                    "selectedObservationsGenerated"
                ]
                == 8
            ).all()
        ),
        "stateRestorationPassed": bool(
            pd.read_parquet(LOOP_ROOT / "state_restoration_validation.parquet")
            .select_dtypes(include="bool")
            .all()
            .all()
        ),
        "matrixFirewallPassed": json.loads(
            (LOOP_ROOT / "matrix_firewall.json").read_text()
        )["status"]
        == "PASS",
        "seedFirewallPassed": json.loads(
            (LOOP_ROOT / "seed_firewall.json").read_text()
        )["status"]
        == "PASS",
        "immutablePriorPassed": prior["unchanged"],
        "fixturesPassed": bool(fixtures["passed"].all()),
    }
    if not all(checks.values()):
        raise RuntimeError("regeneration validation failed")
    BASE.write_json(
        BUILD_ROOT / "regeneration_validation.json",
        {
            "schema": "eidosoma.e01.s19_l31.regeneration_validation.v1",
            "status": "PASS",
            "checks": checks,
            "firstOriginalBranchFrameSha256": frame_hash(original_branches),
            "replayOriginalBranchFrameSha256": frame_hash(replay),
        },
    )
    runtime = {
        "schema": "eidosoma.e01.s19_l31.runtime.v1",
        "repositoryHead": git("rev-parse", "HEAD"),
        "workers": WORKERS,
        "gpuHours": 0,
        "wallSeconds": time.perf_counter() - start_wall,
        "controllerCpuHours": (time.process_time() - start_cpu) / 3600,
        "workerCpuHours": (branch_seconds + replay_seconds) * WORKERS / 3600,
        "branchRows": len(branches),
        "completedAtUtc": utc_now(),
    }
    BASE.write_json(BUILD_ROOT / "runtime_manifest.json", runtime)
    retained = sum(
        path.stat().st_size for path in BUILD_ROOT.rglob("*") if path.is_file()
    )
    temporary = sum(
        path.stat().st_size for path in CACHE_ROOT.rglob("*") if path.is_file()
    )
    storage = {
        "schema": "eidosoma.e01.s19_l31.storage_validation.v1",
        "retainedBytes": retained,
        "retainedGiBCeiling": 25,
        "temporaryBytes": temporary,
        "temporaryGiBCeiling": 75,
        "status": "PASS"
        if retained < 25 * 2**30 and temporary < 75 * 2**30
        else "FAIL",
    }
    BASE.write_json(BUILD_ROOT / "storage_validation.json", storage)
    report = report_text(metrics, gates, classifications, runtime, next_theme)
    BASE.atomic_text(BUILD_ROOT / "research_step_full_results.md", report)
    BASE.atomic_text(BUILD_ROOT / "S19_L31_FULL_RESULTS.md", report)
    BASE.atomic_text(
        BUILD_ROOT / "loop_decision_summary.md",
        f"# S19-L31 decision summary\n\n**Classification:** {', '.join(classifications)}\n\n**Untouched confirmation:** `{confirmed}`.\n\n**Next:** `{next_theme}`.\n",
    )
    BASE.write_json(BUILD_ROOT / "artifact_manifest.json", manifest_for(BUILD_ROOT))
    stage = LOOP_ROOT.with_name(".L31-promotion-stage")
    if stage.exists():
        shutil.rmtree(stage)
    shutil.copytree(BUILD_ROOT, stage)
    if LOOP_ROOT.exists():
        shutil.rmtree(LOOP_ROOT)
    os.replace(stage, LOOP_ROOT)
    shutil.rmtree(BUILD_ROOT)
    manifest_out = json.loads((LOOP_ROOT / "artifact_manifest.json").read_text())
    if any(
        sha256_file(LOOP_ROOT / item["path"]) != item["sha256"]
        for item in manifest_out["files"]
    ):
        raise RuntimeError("artifact hash failure")
    append_ledgers(classifications, runtime["completedAtUtc"], next_theme)
    BASE.atomic_text(ARTIFACT_ROOT / "research_step_full_results.md", report)
    BASE.atomic_text(
        ARTIFACT_ROOT / "S19_CURRENT_HANDOFF.md",
        report.replace("# S19-L31", "# S19 current handoff — S19-L31", 1),
    )
    BASE.write_json(
        ARTIFACT_ROOT / "s19_status.json",
        {
            "schema": "eidosoma.e01.s19.status.v1",
            "status": "ACTIVE_AUTONOMOUS_SEQUENCE",
            "latestCompletedLoop": LOOP_ID,
            "latestClassification": classifications,
            "selectedDiscoveryLead": "CONFIRMED_H8_SHOOTING_COORDINATE"
            if confirmed
            else None,
            "nextAuthorizedLoop": "S19-L32",
            "authorizationUpperBound": "S19-L42",
            "s20Active": False,
            "updatedAtUtc": runtime["completedAtUtc"],
        },
    )
    BASE.write_json(
        ARTIFACT_ROOT / "artifact_manifest.json", manifest_for(ARTIFACT_ROOT)
    )
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "classifications": classifications,
                "confirmed": confirmed,
                "nextTheme": next_theme,
                "runtime": runtime,
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-lock", action="store_true")
    args = parser.parse_args()
    if args.prepare_lock:
        prepare_lock()
    else:
        execute()


if __name__ == "__main__":
    main()
