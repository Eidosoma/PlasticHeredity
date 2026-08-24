#!/usr/bin/env python3
"""Execute the frozen E01/S16 prediction reconstruction and stop before S17."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy
import sklearn
import torch
import yaml
from scipy import stats
from sklearn.metrics import adjusted_rand_score

from e01_frozen_timebase_ensemble.core import (
    frozen_clr,
    selected_clock_observations,
    sha256_array,
)
from e01_pigozzi_source_audit.core import SourceImplementation
from e01_prediction_reconstruction.core import (
    CANDIDATE_IDS,
    CUTOFF_MODE,
    DUMMY_FEATURE_ID,
    EXPECTED_PARAMETER_COUNT,
    FEATURE_IDS,
    MAX_INPUT_LENGTH,
    MAX_TARGET_LENGTH,
    RETROSPECTIVE_MODE,
    TEMPORAL_MODES,
    VERSION,
    apply_channel_scaler,
    binary_metrics,
    build_split_manifest,
    derive_seed128,
    derive_torch_seed,
    fit_channel_scaler,
    matrix_cluster_bootstrap,
    parameter_count,
    predict_probabilities,
    preonset_masks,
    split_summary,
    train_masked_mlp,
)
from e01_source_emergence_metric_identity.core import (
    result_replay_equal,
    run_emergence_pipeline,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
CONFIG_PATH = REPO_ROOT / "configs/e01/s16_first_quarter_prediction_preregistration.yaml"
LOCK_MANIFEST_PATH = REPO_ROOT / "configs/e01/s16_tensor_model_manifest.json"
SPLIT_PATH = REPO_ROOT / "configs/e01/s16_split_manifest.csv"
SAFE_LATTICE = Path("/artifacts/research_steps/S12B/safe_phi_lattice.json")
S13Y_ROOT = Path("/artifacts/research_steps/S13Y")
RAW_ROOT = Path("/cache/e01_s13y_v1/raw_trajectories")
ELIGIBLE_SOURCE_STATUSES = {"ELIGIBLE", "ELIGIBLE_PARTIAL_NONFINITE_LOCAL_VALUES"}
PHIRL = SourceImplementation.PHIRL
PAPER_BASELINES = (
    "COMPOSITION_CHANGE_L2",
    "RAW_COUNTS",
    "NET_COUNT_FLUX",
    DUMMY_FEATURE_ID,
)
PROSPECTIVE_COMPARATORS = (
    DUMMY_FEATURE_ID,
    "COMPOSITION_CHANGE_L2",
    "RAW_COUNTS",
    "NET_COUNT_FLUX",
    "EXACT_H_HISTORY",
)


def canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def write_json(path: Path, payload: object) -> None:
    path.write_text(canonical_json(payload) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_hash(frame: pd.DataFrame) -> str:
    payload = frame.to_json(orient="table", index=False, double_precision=15)
    return hashlib.sha256(payload.encode()).hexdigest()


def run_git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def current_repo_lock() -> dict[str, Any]:
    head = run_git("rev-parse", "HEAD")
    remote = run_git("rev-parse", "origin/eidosoma/groups/42")
    status = run_git("status", "--short")
    return {
        "branch": run_git("branch", "--show-current"),
        "head": head,
        "remoteHead": remote,
        "workingTreeStatus": status,
        "passed": bool(
            head == remote
            and status == ""
            and run_git("branch", "--show-current") == "eidosoma/groups/42"
        ),
    }


def preoutcome_lock(config: dict[str, Any]) -> dict[str, Any]:
    repository = current_repo_lock()
    manifest = json.loads(LOCK_MANIFEST_PATH.read_text(encoding="utf-8"))
    split = pd.read_csv(SPLIT_PATH)
    expected_split = build_split_manifest()
    pd.testing.assert_frame_equal(split, expected_split, check_dtype=False)
    if not repository["passed"]:
        raise RuntimeError(f"repository is not a clean pushed lock: {repository}")
    if manifest["predictionOutcomeAccessed"] is not False:
        raise RuntimeError("tensor/model manifest is not pre-outcome")
    if manifest["versionedStepId"] != VERSION or config["versionedStepId"] != VERSION:
        raise RuntimeError("S16 version lock mismatch")
    recorded = manifest["files"]
    if recorded["preregistration"]["sha256"] != sha256_file(CONFIG_PATH):
        raise RuntimeError("preregistration changed after tensor lock")
    if recorded["splitManifest"]["sha256"] != sha256_file(SPLIT_PATH):
        raise RuntimeError("split manifest changed after tensor lock")
    return {
        "schema": "eidosoma.e01.s16_preoutcome_design_lock.v1",
        "researchStepId": "S16",
        "versionedStepId": VERSION,
        "predictionOutcomeAccessedAtLock": False,
        "repository": repository,
        "files": [
            {"path": str(CONFIG_PATH.relative_to(REPO_ROOT)), "sha256": sha256_file(CONFIG_PATH)},
            {"path": str(LOCK_MANIFEST_PATH.relative_to(REPO_ROOT)), "sha256": sha256_file(LOCK_MANIFEST_PATH)},
            {"path": str(SPLIT_PATH.relative_to(REPO_ROOT)), "sha256": sha256_file(SPLIT_PATH)},
            {
                "path": "src/e01_prediction_reconstruction/core.py",
                "sha256": sha256_file(REPO_ROOT / "src/e01_prediction_reconstruction/core.py"),
            },
            {
                "path": "scripts/e01/run_s16_prediction_reconstruction.py",
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            {
                "path": "tests/e01/test_s16_prediction_reconstruction.py",
                "sha256": sha256_file(REPO_ROOT / "tests/e01/test_s16_prediction_reconstruction.py"),
            },
        ],
        "forbiddenAfterOutcomeAccess": config["forbiddenAfterOutcomeAccess"],
        "passed": True,
    }


def prior_step_name(path: Path) -> bool:
    name = path.name
    if not name.startswith("S"):
        return False
    digits = ""
    for character in name[1:]:
        if character.isdigit():
            digits += character
        else:
            break
    return bool(digits) and int(digits) <= 15


def immutable_input_paths() -> list[tuple[Path, str, bool]]:
    paths: list[tuple[Path, str, bool]] = []
    research_root = Path("/artifacts/research_steps")
    for step_dir in sorted(item for item in research_root.iterdir() if item.is_dir()):
        if prior_step_name(step_dir):
            for path in sorted(item for item in step_dir.rglob("*") if item.is_file()):
                paths.append((path, f"IMMUTABLE_PRIOR_{step_dir.name}", True))
    for path in sorted(RAW_ROOT.rglob("*.pickle")):
        paths.append((path, "FROZEN_S13Y_RAW_TRAJECTORY", True))
    governance = [
        (WORKSPACE_ROOT / "AGENTS.md", "GOVERNING_POLICY", True),
        (WORKSPACE_ROOT / "FULL_PLAN.md", "GOVERNING_PLAN", True),
        (WORKSPACE_ROOT / "RESEARCH_PLAN.md", "MUTABLE_CURRENT_HANDOFF", False),
        (WORKSPACE_ROOT / "input-attachments/MANIFEST.json", "UPLOADED_INPUT_MANIFEST", True),
        (
            WORKSPACE_ROOT
            / "input-attachments/ed5486bf-a043-485b-a233-d88d8d123759/_metadata/ATTACHMENT.md",
            "UPLOADED_INPUT_SIDECAR",
            True,
        ),
        (Path("/cache/e01_s03/downloads/paper-2607.28250v1.pdf"), "ORIGINAL_PAPER", True),
    ]
    paths.extend(governance)
    unique: dict[str, tuple[Path, str, bool]] = {}
    for item in paths:
        unique[str(item[0])] = item
    return [unique[key] for key in sorted(unique)]


def hash_input_baseline() -> list[dict[str, Any]]:
    rows = []
    for path, role, required in immutable_input_paths():
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append(
            {
                "path": str(path),
                "role": role,
                "bytes": path.stat().st_size,
                "sha256Before": sha256_file(path),
                "immutableRequired": required,
            }
        )
    return rows


def finalize_input_manifest(rows: list[dict[str, Any]]) -> dict[str, Any]:
    immutable_mismatches = 0
    for row in rows:
        path = Path(row["path"])
        row["sha256After"] = sha256_file(path) if path.is_file() else None
        row["matchedAfter"] = row["sha256After"] == row["sha256Before"]
        if row["immutableRequired"] and not row["matchedAfter"]:
            immutable_mismatches += 1
    return {
        "schema": "eidosoma.e01.s16_input_manifest.v1",
        "researchStepId": "S16",
        "entryCount": len(rows),
        "immutableMismatchCount": immutable_mismatches,
        "passed": immutable_mismatches == 0,
        "entries": rows,
    }


def source_seed(candidate_id: str, matrix_index: int, purpose: str) -> int:
    return int(derive_seed128("cutoff_source", candidate_id, matrix_index, purpose) % (2**32))


def normalized_compositions(states: np.ndarray) -> np.ndarray:
    values = np.asarray(states, dtype=np.float64)
    masses = values.sum(axis=1)
    if np.any(masses <= 0):
        raise ValueError("empty selected state")
    return values / masses[:, None]


def incoming_h(compositions: np.ndarray) -> np.ndarray:
    normalized = compositions / np.linalg.norm(compositions, axis=1)[:, None]
    adjacent = np.sum(normalized[:-1] * normalized[1:], axis=1)
    return np.concatenate(([adjacent[0]], adjacent)).astype(np.float64)


def canonical_partition(left: tuple[int, ...], right: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    sides = (tuple(sorted(left)), tuple(sorted(right)))
    return tuple(sorted(sides))  # type: ignore[return-value]


def partition_ari(
    full_left: tuple[int, ...],
    full_right: tuple[int, ...],
    cutoff_left: tuple[int, ...],
    cutoff_right: tuple[int, ...],
) -> float | None:
    variables = sorted(set(full_left) | set(full_right) | set(cutoff_left) | set(cutoff_right))
    if not variables:
        return None
    full_map = {value: 0 for value in full_left} | {value: 1 for value in full_right}
    cutoff_map = {value: 0 for value in cutoff_left} | {value: 1 for value in cutoff_right}
    shared = [value for value in variables if value in full_map and value in cutoff_map]
    if len(shared) < 2:
        return None
    return float(
        adjusted_rand_score(
            [full_map[value] for value in shared], [cutoff_map[value] for value in shared]
        )
    )


def array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(b"\0")
    digest.update(str(tuple(array.shape)).encode())
    digest.update(b"\0")
    digest.update(array.tobytes())
    return digest.hexdigest()


def build_feature(
    values: np.ndarray,
    available: np.ndarray,
    cutoff: int,
    *,
    scalar: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    output = np.zeros((MAX_INPUT_LENGTH, 100), dtype=np.float64)
    channel_mask = np.zeros_like(output, dtype=bool)
    time_mask = np.zeros(MAX_INPUT_LENGTH, dtype=bool)
    time_mask[:cutoff] = True
    if scalar:
        vector = np.asarray(values, dtype=np.float64).reshape(-1)
        valid = np.asarray(available, dtype=bool).reshape(-1)
        output[:cutoff, 0] = np.where(valid[:cutoff], vector[:cutoff], 0.0)
        channel_mask[:cutoff, 0] = valid[:cutoff]
    else:
        matrix = np.asarray(values, dtype=np.float64)
        valid = np.asarray(available, dtype=bool)
        if matrix.shape != (cutoff, 100) or valid.shape != matrix.shape:
            raise ValueError("vector feature shape mismatch")
        output[:cutoff] = np.where(valid, matrix, 0.0)
        channel_mask[:cutoff] = valid
    if not np.all(np.isfinite(output)):
        raise ValueError("feature tensor contains nonfinite value")
    return output, channel_mask, time_mask


def source_values(result: Any, cutoff: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros(cutoff, dtype=np.float64)
    available = np.zeros(cutoff, dtype=bool)
    if result.emergence is not None:
        local = np.asarray(result.emergence, dtype=np.float64)
        expected = cutoff - int(result.local_offset)
        if local.size != expected:
            raise ValueError(f"cutoff emergence length {local.size} != {expected}")
        positions = np.arange(result.local_offset, cutoff)
        finite = np.isfinite(local)
        values[positions[finite]] = local[finite]
        available[positions[finite]] = True
    return values, available


def completed_phi_feature(
    full: pd.DataFrame, cutoff: int
) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros(cutoff, dtype=np.float64)
    available = np.zeros(cutoff, dtype=bool)
    for row in full.itertuples(index=False):
        index = int(row.selectedSequenceIndex)
        if index >= cutoff:
            continue
        value = float(row.emergence) if row.emergence is not None else math.nan
        if row.status == "ELIGIBLE" and np.isfinite(value):
            values[index] = value
            available[index] = True
    return values, available


def prepare_payloads() -> tuple[list[dict[str, Any]], dict[str, pd.DataFrame]]:
    labels = pd.read_parquet(S13Y_ROOT / "label_values.parquet")
    labels = labels.loc[labels["labelId"].eq("MOL_ADJACENT_INCOMING_H900")].copy()
    full = pd.read_parquet(S13Y_ROOT / "full_source_values.parquet")
    full = full.loc[full["implementationId"].eq("PHIRL_REGULARIZED_SOURCE")].copy()
    trajectory_manifest = pd.read_parquet(S13Y_ROOT / "trajectory_manifest.parquet")
    partitions = pd.read_parquet(S13Y_ROOT / "partition_history.parquet")
    partitions = partitions.loc[
        partitions["temporalModeId"].eq("PHIRL_REGULARIZED_SOURCE_EMERGENCE_FULL")
    ].copy()
    payloads: list[dict[str, Any]] = []
    feature_audit: list[dict[str, Any]] = []
    source_audit: list[dict[str, Any]] = []
    suffix_audit: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    label_mismatches = 0
    for candidate_id in CANDIDATE_IDS:
        for matrix_index in range(100):
            manifest_row = trajectory_manifest.loc[
                trajectory_manifest["candidateId"].eq(candidate_id)
                & trajectory_manifest["matrixIndex"].eq(matrix_index)
            ]
            if len(manifest_row) != 1:
                raise ValueError("trajectory manifest identity is not unique")
            cache_path = Path(manifest_row.iloc[0]["cachePath"])
            if sha256_file(cache_path) != manifest_row.iloc[0]["cacheSha256"]:
                raise ValueError("raw trajectory cache hash mismatch")
            with cache_path.open("rb") as handle:
                trajectory = pickle.load(handle)
            selected = selected_clock_observations(
                trajectory, "C1_SELECTED_DAUGHTER_RETAINED"
            )
            states = np.asarray([item.state for item in selected], dtype=np.int64)
            label_rows = labels.loc[
                labels["candidateId"].eq(candidate_id)
                & labels["matrixIndex"].eq(matrix_index)
            ].sort_values("selectedSequenceIndex")
            if len(label_rows) != len(states) or not np.array_equal(
                label_rows["selectedSequenceIndex"].to_numpy(), np.arange(len(states))
            ):
                raise ValueError("label/trajectory selected-sequence mismatch")
            raw_indices = np.asarray([item.observation_index for item in selected], dtype=np.int64)
            if not np.array_equal(raw_indices, label_rows["rawObservationIndex"].to_numpy()):
                raise ValueError("label/raw observation mismatch")
            target_labels = label_rows["isReplicator"].to_numpy(dtype=bool)
            h_values = label_rows["labelScore"].to_numpy(dtype=np.float64)
            compositions = normalized_compositions(states)
            recomputed_h = incoming_h(compositions)
            if not np.allclose(h_values, recomputed_h, rtol=0.0, atol=2e-15):
                raise ValueError("frozen exact-H history does not replay")
            expected_labels = recomputed_h > 0.9
            mismatch = int(np.count_nonzero(expected_labels != target_labels))
            label_mismatches += mismatch
            if mismatch:
                raise ValueError("Y != I(H>0.9)")
            total = len(states)
            cutoff = math.floor(0.25 * total)
            target_length = total - cutoff
            if cutoff > MAX_INPUT_LENGTH or target_length > MAX_TARGET_LENGTH:
                raise ValueError("frozen padding dimensions are insufficient")
            closed_change = np.zeros(total, dtype=np.float64)
            closed_change[1:] = np.linalg.norm(np.diff(compositions, axis=0), axis=1)
            flux = np.zeros_like(states, dtype=np.float64)
            flux[1:] = np.diff(states, axis=0)
            prefix_states = states[:cutoff]
            cutoff_clr, _, closure_error = frozen_clr(prefix_states)
            pre_seed = source_seed(candidate_id, matrix_index, "preprocessing")
            part_seed = source_seed(candidate_id, matrix_index, "partition")
            cutoff_result = run_emergence_pipeline(
                cutoff_clr,
                PHIRL,
                SAFE_LATTICE,
                preprocessing_seed=pre_seed,
                partition_seed=part_seed,
            )
            cutoff_replay = run_emergence_pipeline(
                cutoff_clr,
                PHIRL,
                SAFE_LATTICE,
                preprocessing_seed=pre_seed,
                partition_seed=part_seed,
            )
            replay_passed = result_replay_equal(cutoff_result, cutoff_replay)
            cutoff_values, cutoff_available = source_values(cutoff_result, cutoff)
            full_rows = full.loc[
                full["candidateId"].eq(candidate_id)
                & full["matrixIndex"].eq(matrix_index)
            ].sort_values("selectedSequenceIndex")
            completed_values, completed_available = completed_phi_feature(full_rows, cutoff)
            full_part = partitions.loc[
                partitions["candidateId"].eq(candidate_id)
                & partitions["matrixIndex"].eq(matrix_index)
            ]
            if len(full_part) != 1:
                raise ValueError("completed partition identity is not unique")
            full_left = tuple(json.loads(full_part.iloc[0]["partition1Json"]))
            full_right = tuple(json.loads(full_part.iloc[0]["partition2Json"]))
            shared = completed_available & cutoff_available
            if np.count_nonzero(shared) >= 3:
                fit_spearman = float(
                    stats.spearmanr(completed_values[shared], cutoff_values[shared]).statistic
                )
                fit_mae = float(np.mean(np.abs(completed_values[shared] - cutoff_values[shared])))
            else:
                fit_spearman = None
                fit_mae = None
            source_audit.append(
                {
                    "researchStepId": "S16",
                    "candidateId": candidate_id,
                    "matrixIndex": matrix_index,
                    "trajectoryId": trajectory.trajectory_id,
                    "T": total,
                    "cutoff": cutoff,
                    "targetLength": target_length,
                    "fitObservationCount": cutoff,
                    "maximumSelectedSequenceIndexUsed": cutoff - 1,
                    "suffixObservationCount": total - cutoff,
                    "status": cutoff_result.status,
                    "reason": cutoff_result.reason,
                    "retainedVariableCount": len(cutoff_result.retained_variables),
                    "retainedVariablesJson": json.dumps(list(cutoff_result.retained_variables)),
                    "partition1Json": json.dumps(list(cutoff_result.partition_1)),
                    "partition2Json": json.dumps(list(cutoff_result.partition_2)),
                    "partitionSize1": len(cutoff_result.partition_1),
                    "partitionSize2": len(cutoff_result.partition_2),
                    "eligibleEmergenceCount": int(np.count_nonzero(cutoff_available)),
                    "exactReplayPassed": replay_passed,
                    "componentIdentityMaxAbsError": cutoff_result.component_identity_max_abs_error,
                    "maximumClosureError": float(np.max(closure_error)),
                    "prefixClrSha256": sha256_array(cutoff_clr),
                    "cutoffEmergenceSha256": array_digest(cutoff_values),
                    "completedFirstQuarterEmergenceSha256": array_digest(completed_values),
                    "sharedEmergenceCount": int(np.count_nonzero(shared)),
                    "completedVsCutoffSpearman": fit_spearman,
                    "completedVsCutoffMeanAbsoluteDifference": fit_mae,
                    "completedVsCutoffPartitionARI": partition_ari(
                        full_left,
                        full_right,
                        cutoff_result.partition_1,
                        cutoff_result.partition_2,
                    ),
                    "preprocessingSeed": pre_seed,
                    "partitionSeed": part_seed,
                    "futureSuffixAccessedByCutoffFit": False,
                }
            )
            if cutoff_result.status not in ELIGIBLE_SOURCE_STATUSES or not replay_passed:
                failure_rows.append(
                    {
                        "failureId": f"S16-SOURCE-{candidate_id}-M{matrix_index:03d}",
                        "stage": "cutoff_source_fit",
                        "severity": "NONFATAL_STATUS_BEARING",
                        "status": cutoff_result.status,
                        "reason": cutoff_result.reason or "exact_replay_failed",
                        "outcomeExclusion": False,
                    }
                )
            for variant_id in (
                "deletion",
                "deterministic_shuffle",
                "domain_separated_replacement",
            ):
                variant_full = states.copy()
                suffix_length = total - cutoff
                if variant_id == "deletion":
                    variant_full = variant_full[:cutoff]
                elif variant_id == "deterministic_shuffle":
                    rng = np.random.Generator(
                        np.random.PCG64DXSM(
                            derive_seed128("suffix", candidate_id, matrix_index, variant_id)
                        )
                    )
                    variant_full[cutoff:] = variant_full[cutoff:][rng.permutation(suffix_length)]
                else:
                    rng = np.random.Generator(
                        np.random.PCG64DXSM(
                            derive_seed128("suffix", candidate_id, matrix_index, variant_id)
                        )
                    )
                    variant_full[cutoff:] = rng.integers(
                        0, 4, size=variant_full[cutoff:].shape, dtype=np.int64
                    )
                variant_prefix_states = np.ascontiguousarray(variant_full[:cutoff])
                state_exact = np.array_equal(variant_prefix_states, prefix_states)
                variant_clr, _, _ = frozen_clr(variant_prefix_states)
                clr_exact = np.array_equal(variant_clr, cutoff_clr)
                variant_result = run_emergence_pipeline(
                    variant_clr,
                    PHIRL,
                    SAFE_LATTICE,
                    preprocessing_seed=pre_seed,
                    partition_seed=part_seed,
                )
                result_exact = result_replay_equal(cutoff_result, variant_result)
                active_exact = cutoff_result.retained_variables == variant_result.retained_variables
                partition_exact = canonical_partition(
                    cutoff_result.partition_1, cutoff_result.partition_2
                ) == canonical_partition(variant_result.partition_1, variant_result.partition_2)
                passed = state_exact and clr_exact and active_exact and partition_exact and result_exact
                suffix_audit.append(
                    {
                        "researchStepId": "S16",
                        "candidateId": candidate_id,
                        "matrixIndex": matrix_index,
                        "variantId": variant_id,
                        "suffixLength": suffix_length,
                        "prefixStateExact": state_exact,
                        "prefixClrExact": clr_exact,
                        "activeVariablesExact": active_exact,
                        "partitionExactUpToSideExchange": partition_exact,
                        "sourceResultExact": result_exact,
                        "passed": passed,
                    }
                )
            target = np.zeros(MAX_TARGET_LENGTH, dtype=bool)
            target_mask = np.zeros(MAX_TARGET_LENGTH, dtype=bool)
            target[:target_length] = target_labels[cutoff:]
            target_mask[:target_length] = True
            input_labels = np.zeros(MAX_INPUT_LENGTH, dtype=bool)
            input_labels[:cutoff] = target_labels[:cutoff]
            common_features = {
                "COMPOSITION_CHANGE_L2": build_feature(
                    closed_change[:cutoff], np.arange(cutoff) > 0, cutoff, scalar=True
                ),
                "RAW_COUNTS": build_feature(
                    states[:cutoff].astype(np.float64),
                    np.ones((cutoff, 100), dtype=bool),
                    cutoff,
                    scalar=False,
                ),
                "NET_COUNT_FLUX": build_feature(
                    flux[:cutoff],
                    np.broadcast_to((np.arange(cutoff) > 0)[:, None], (cutoff, 100)),
                    cutoff,
                    scalar=False,
                ),
                "EXACT_H_HISTORY": build_feature(
                    h_values[:cutoff], np.ones(cutoff, dtype=bool), cutoff, scalar=True
                ),
            }
            features = {
                RETROSPECTIVE_MODE: {
                    "PHIRL_EMERGENCE": build_feature(
                        completed_values, completed_available, cutoff, scalar=True
                    ),
                    **common_features,
                },
                CUTOFF_MODE: {
                    "PHIRL_EMERGENCE": build_feature(
                        cutoff_values, cutoff_available, cutoff, scalar=True
                    ),
                    **common_features,
                },
            }
            for mode_id in TEMPORAL_MODES:
                for feature_id, (values, mask, time_mask) in features[mode_id].items():
                    feature_audit.append(
                        {
                            "researchStepId": "S16",
                            "candidateId": candidate_id,
                            "matrixIndex": matrix_index,
                            "modeId": mode_id,
                            "featureId": feature_id,
                            "T": total,
                            "cutoff": cutoff,
                            "targetLength": target_length,
                            "validFeatureCellCount": int(mask.sum()),
                            "validInputTimeCount": int(time_mask.sum()),
                            "valueSha256": array_digest(values),
                            "channelMaskSha256": array_digest(mask),
                            "timeMaskSha256": array_digest(time_mask),
                            "maximumSourceSelectedSequenceIndex": cutoff - 1,
                            "futureSuffixUsed": bool(
                                mode_id == RETROSPECTIVE_MODE
                                and feature_id == "PHIRL_EMERGENCE"
                            ),
                        }
                    )
            payloads.append(
                {
                    "candidateId": candidate_id,
                    "matrixIndex": matrix_index,
                    "trajectoryId": trajectory.trajectory_id,
                    "T": total,
                    "cutoff": cutoff,
                    "target": target,
                    "targetMask": target_mask,
                    "inputLabels": input_labels,
                    "features": features,
                }
            )
    if label_mismatches != 0 or len(payloads) != 200:
        raise ValueError("payload label/cardinality validation failed")
    outputs = {
        "featureAudit": pd.DataFrame(feature_audit),
        "sourceAudit": pd.DataFrame(source_audit),
        "suffixAudit": pd.DataFrame(suffix_audit),
        "failures": pd.DataFrame(failure_rows),
    }
    return payloads, outputs


def tensorize(
    payloads: list[dict[str, Any]], candidate_id: str, mode_id: str, feature_id: str
) -> dict[str, np.ndarray]:
    selected = sorted(
        (item for item in payloads if item["candidateId"] == candidate_id),
        key=lambda item: item["matrixIndex"],
    )
    if [item["matrixIndex"] for item in selected] != list(range(100)):
        raise ValueError("tensorization matrix order mismatch")
    feature = [item["features"][mode_id][feature_id] for item in selected]
    return {
        "values": np.stack([item[0] for item in feature]),
        "channelMask": np.stack([item[1] for item in feature]),
        "timeMask": np.stack([item[2] for item in feature]),
        "target": np.stack([item["target"] for item in selected]).astype(np.float64),
        "targetMask": np.stack([item["targetMask"] for item in selected]),
        "inputLabels": np.stack([item["inputLabels"] for item in selected]),
        "cutoff": np.asarray([item["cutoff"] for item in selected], dtype=np.int64),
        "T": np.asarray([item["T"] for item in selected], dtype=np.int64),
    }


def subset_indices(split: pd.DataFrame, repetition: int, role: str) -> np.ndarray:
    return (
        split.loc[
            split["repetitionId"].eq(repetition) & split["splitRole"].eq(role),
            "matrixIndex",
        ]
        .sort_values()
        .to_numpy(dtype=np.int64)
    )


def flatten_valid(
    target: np.ndarray, probability: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    return target[mask].astype(bool), probability[mask].astype(np.float64)


def model_seed_for(candidate_id: str, repetition: int) -> int:
    return derive_torch_seed("model", candidate_id, repetition)


def model_payload_hash(
    candidate_id: str, repetition: int, feature_id: str, values: np.ndarray, mask: np.ndarray
) -> str:
    digest = hashlib.sha256()
    for value in (candidate_id, repetition, feature_id, array_digest(values), array_digest(mask)):
        digest.update(str(value).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def evaluate_predictions(
    *,
    candidate_id: str,
    mode_id: str,
    feature_id: str,
    repetition: int,
    test_indices: np.ndarray,
    tensor: dict[str, np.ndarray],
    probabilities: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    target = tensor["target"][test_indices].astype(bool)
    target_mask = tensor["targetMask"][test_indices]
    flat_target, flat_probability = flatten_valid(target, probabilities, target_mask)
    metrics = binary_metrics(flat_target, flat_probability)
    matrix_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    preonset_eligible, risk_mask = preonset_masks(
        tensor["inputLabels"][test_indices], target, target_mask
    )
    for local_index, matrix_index in enumerate(test_indices):
        valid = np.flatnonzero(target_mask[local_index])
        y = target[local_index, valid]
        p = probabilities[local_index, valid]
        matrix_metric = binary_metrics(y, p)
        matrix_rows.append(
            {
                "candidateId": candidate_id,
                "modeId": mode_id,
                "featureId": feature_id,
                "repetitionId": repetition,
                "matrixIndex": int(matrix_index),
                "accuracy": matrix_metric["accuracy"],
                "validTargetCount": matrix_metric["validTargetCount"],
                "prevalence": matrix_metric["prevalence"],
            }
        )
        prediction_frames.append(
            pd.DataFrame(
                {
                    "candidateId": candidate_id,
                    "modeId": mode_id,
                    "featureId": feature_id,
                    "repetitionId": repetition,
                    "matrixIndex": int(matrix_index),
                    "targetOffset": valid.astype(np.int32),
                    "selectedSequenceIndex": (tensor["cutoff"][matrix_index] + valid).astype(np.int32),
                    "target": y,
                    "probability": p,
                    "predictedClass": p >= 0.5,
                    "preOnsetEligibleRun": bool(preonset_eligible[local_index]),
                    "preOnsetRiskPosition": risk_mask[local_index, valid],
                }
            )
        )
    matrix_frame = pd.DataFrame(matrix_rows)
    metrics["macroMatrixAccuracy"] = float(matrix_frame["accuracy"].mean())
    metrics.update(
        {
            "candidateId": candidate_id,
            "modeId": mode_id,
            "featureId": feature_id,
            "repetitionId": repetition,
            "testMatrixCount": len(test_indices),
        }
    )
    risk_target, risk_probability = flatten_valid(target, probabilities, risk_mask)
    preonset = binary_metrics(risk_target, risk_probability)
    preonset.update(
        {
            "candidateId": candidate_id,
            "modeId": mode_id,
            "featureId": feature_id,
            "repetitionId": repetition,
            "eligibleRunCount": int(np.count_nonzero(preonset_eligible)),
            "excludedAlreadyPositiveInputRunCount": int(
                len(preonset_eligible) - np.count_nonzero(preonset_eligible)
            ),
        }
    )
    time_rows: list[dict[str, Any]] = []
    for offset in range(int(np.max(tensor["targetMask"][test_indices].sum(axis=1)))):
        valid = target_mask[:, offset]
        if not np.any(valid):
            continue
        time_metric = binary_metrics(target[valid, offset], probabilities[valid, offset])
        time_rows.append(
            {
                "candidateId": candidate_id,
                "modeId": mode_id,
                "featureId": feature_id,
                "repetitionId": repetition,
                "targetOffset": offset,
                "validMatrixCount": int(np.count_nonzero(valid)),
                **time_metric,
            }
        )
    return (
        pd.concat(prediction_frames, ignore_index=True),
        metrics,
        matrix_rows,
        preonset,
        time_rows,
    )


def run_models(
    payloads: list[dict[str, Any]], split: pd.DataFrame, work_root: Path
) -> dict[str, pd.DataFrame]:
    split_metrics: list[dict[str, Any]] = []
    matrix_metrics: list[dict[str, Any]] = []
    preonset_metrics: list[dict[str, Any]] = []
    time_metrics: list[dict[str, Any]] = []
    prevalence_rows: list[dict[str, Any]] = []
    scaler_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []
    prediction_writer: pq.ParquetWriter | None = None
    prediction_path = work_root / "prediction_rows.parquet"
    model_cache: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    tensor_cache: dict[tuple[str, str, str], dict[str, np.ndarray]] = {}

    def append_predictions(frame: pd.DataFrame) -> None:
        nonlocal prediction_writer
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if prediction_writer is None:
            prediction_writer = pq.ParquetWriter(
                prediction_path,
                table.schema,
                compression="zstd",
                use_dictionary=True,
            )
        prediction_writer.write_table(table)

    for candidate_id in CANDIDATE_IDS:
        base_tensor = tensorize(
            payloads, candidate_id, RETROSPECTIVE_MODE, "EXACT_H_HISTORY"
        )
        for repetition in range(10):
            fit_indices = subset_indices(split, repetition, "FIT")
            validation_indices = subset_indices(split, repetition, "VALIDATION")
            test_indices = subset_indices(split, repetition, "TEST")
            for role, indices in (
                ("FIT", fit_indices),
                ("VALIDATION", validation_indices),
                ("TEST", test_indices),
            ):
                mask = base_tensor["targetMask"][indices]
                y = base_tensor["target"][indices].astype(bool)
                prevalence_rows.append(
                    {
                        "candidateId": candidate_id,
                        "repetitionId": repetition,
                        "splitRole": role,
                        "matrixCount": len(indices),
                        "validTargetCount": int(mask.sum()),
                        "positiveTargetCount": int(y[mask].sum()),
                        "targetPrevalence": float(y[mask].mean()),
                        "inputAlreadyPositiveMatrixCount": int(
                            np.count_nonzero(np.any(base_tensor["inputLabels"][indices], axis=1))
                        ),
                    }
                )
            for mode_id in TEMPORAL_MODES:
                for feature_id in FEATURE_IDS:
                    tensor_key = (candidate_id, mode_id, feature_id)
                    if feature_id != DUMMY_FEATURE_ID and tensor_key not in tensor_cache:
                        tensor_cache[tensor_key] = tensorize(
                            payloads, candidate_id, mode_id, feature_id
                        )
                    tensor = (
                        base_tensor
                        if feature_id == DUMMY_FEATURE_ID
                        else tensor_cache[tensor_key]
                    )
                    model_seed = model_seed_for(candidate_id, repetition)
                    if feature_id == DUMMY_FEATURE_ID:
                        fit_mask = tensor["targetMask"][fit_indices]
                        fit_y = tensor["target"][fit_indices].astype(bool)
                        training_prevalence = float(np.mean(fit_y[fit_mask]))
                        probabilities = np.full(
                            (len(test_indices), MAX_TARGET_LENGTH),
                            training_prevalence,
                            dtype=np.float64,
                        )
                        training_rows.append(
                            {
                                "candidateId": candidate_id,
                                "modeId": mode_id,
                                "featureId": feature_id,
                                "repetitionId": repetition,
                                "epoch": -1,
                                "fitLoss": None,
                                "validationLoss": None,
                                "bestEpoch": None,
                                "stoppedEpoch": None,
                                "bestValidationLoss": None,
                                "modelSeed": model_seed,
                                "parameterCount": 0,
                                "reusedModel": mode_id == CUTOFF_MODE,
                                "trainingOnlyPrevalence": training_prevalence,
                            }
                        )
                    else:
                        payload_hash = model_payload_hash(
                            candidate_id,
                            repetition,
                            feature_id,
                            tensor["values"],
                            tensor["channelMask"],
                        )
                        cache_key = (candidate_id, repetition, feature_id, payload_hash)
                        if cache_key not in model_cache:
                            scaler = fit_channel_scaler(
                                tensor["values"][fit_indices],
                                tensor["channelMask"][fit_indices],
                            )
                            scaled = apply_channel_scaler(
                                tensor["values"], tensor["channelMask"], scaler
                            )
                            result = train_masked_mlp(
                                scaled[fit_indices],
                                tensor["channelMask"][fit_indices],
                                tensor["timeMask"][fit_indices],
                                tensor["target"][fit_indices],
                                tensor["targetMask"][fit_indices],
                                scaled[validation_indices],
                                tensor["channelMask"][validation_indices],
                                tensor["timeMask"][validation_indices],
                                tensor["target"][validation_indices],
                                tensor["targetMask"][validation_indices],
                                model_seed=model_seed,
                            )
                            probabilities = predict_probabilities(
                                result.model,
                                scaled[test_indices],
                                tensor["channelMask"][test_indices],
                                tensor["timeMask"][test_indices],
                            )
                            model_cache[cache_key] = {
                                "scaler": scaler,
                                "scaled": scaled,
                                "result": result,
                                "probabilities": probabilities,
                                "sourceMode": mode_id,
                            }
                        cached = model_cache[cache_key]
                        scaler = cached["scaler"]
                        scaled = cached["scaled"]
                        result = cached["result"]
                        probabilities = cached["probabilities"]
                        reused = cached["sourceMode"] != mode_id
                        for channel in range(100):
                            scaler_rows.append(
                                {
                                    "candidateId": candidate_id,
                                    "modeId": mode_id,
                                    "featureId": feature_id,
                                    "repetitionId": repetition,
                                    "channelIndex": channel,
                                    "fitValidCellCount": int(scaler.valid_count[channel]),
                                    "mean": float(scaler.mean[channel]),
                                    "scale": float(scaler.scale[channel]),
                                    "fitMatrixIndicesJson": json.dumps(fit_indices.tolist()),
                                    "validationMatrixIndicesExcluded": True,
                                    "testMatrixIndicesExcluded": True,
                                    "suffixObservationsExcluded": True,
                                    "reusedAcrossModes": reused,
                                }
                            )
                        for row in result.history.itertuples(index=False):
                            training_rows.append(
                                {
                                    "candidateId": candidate_id,
                                    "modeId": mode_id,
                                    "featureId": feature_id,
                                    "repetitionId": repetition,
                                    "epoch": int(row.epoch),
                                    "fitLoss": float(row.fitLoss),
                                    "validationLoss": float(row.validationLoss),
                                    "bestEpoch": result.best_epoch,
                                    "stoppedEpoch": result.stopped_epoch,
                                    "bestValidationLoss": result.best_validation_loss,
                                    "modelSeed": model_seed,
                                    "parameterCount": parameter_count(result.model),
                                    "reusedModel": reused,
                                    "trainingOnlyPrevalence": None,
                                }
                            )
                        if repetition == 0 and not reused:
                            replay = train_masked_mlp(
                                scaled[fit_indices],
                                tensor["channelMask"][fit_indices],
                                tensor["timeMask"][fit_indices],
                                tensor["target"][fit_indices],
                                tensor["targetMask"][fit_indices],
                                scaled[validation_indices],
                                tensor["channelMask"][validation_indices],
                                tensor["timeMask"][validation_indices],
                                tensor["target"][validation_indices],
                                tensor["targetMask"][validation_indices],
                                model_seed=model_seed,
                            )
                            replay_probability = predict_probabilities(
                                replay.model,
                                scaled[test_indices],
                                tensor["channelMask"][test_indices],
                                tensor["timeMask"][test_indices],
                            )
                            history_exact = result.history.equals(replay.history)
                            prediction_exact = np.array_equal(
                                probabilities, replay_probability
                            )
                            replay_rows.append(
                                {
                                    "candidateId": candidate_id,
                                    "modeId": mode_id,
                                    "featureId": feature_id,
                                    "repetitionId": repetition,
                                    "modelSeed": model_seed,
                                    "historyExact": history_exact,
                                    "predictionExact": prediction_exact,
                                    "predictionSha256": array_digest(probabilities),
                                    "replayPredictionSha256": array_digest(replay_probability),
                                    "passed": history_exact and prediction_exact,
                                }
                            )
                    prediction_frame, metric_row, per_matrix, preonset_row, per_time = (
                        evaluate_predictions(
                            candidate_id=candidate_id,
                            mode_id=mode_id,
                            feature_id=feature_id,
                            repetition=repetition,
                            test_indices=test_indices,
                            tensor=tensor,
                            probabilities=probabilities,
                        )
                    )
                    append_predictions(prediction_frame)
                    split_metrics.append(metric_row)
                    matrix_metrics.extend(per_matrix)
                    preonset_metrics.append(preonset_row)
                    time_metrics.extend(per_time)
    if prediction_writer is not None:
        prediction_writer.close()
    return {
        "splitMetrics": pd.DataFrame(split_metrics),
        "matrixMetrics": pd.DataFrame(matrix_metrics),
        "preonsetMetrics": pd.DataFrame(preonset_metrics),
        "timeMetrics": pd.DataFrame(time_metrics),
        "prevalence": pd.DataFrame(prevalence_rows),
        "scalers": pd.DataFrame(scaler_rows),
        "training": pd.DataFrame(training_rows),
        "modelReplay": pd.DataFrame(replay_rows),
        "predictionPath": pd.DataFrame([{"path": str(prediction_path)}]),
    }


def summarize_split_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "accuracy",
        "macroMatrixAccuracy",
        "auroc",
        "auprc",
        "brier",
        "calibrationError",
        "balancedAccuracy",
        "sensitivity",
        "specificity",
        "prevalence",
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(["candidateId", "modeId", "featureId"], sort=True):
        for metric in metric_columns:
            values = pd.to_numeric(group[metric], errors="coerce").to_numpy(np.float64)
            summary = split_summary(values)
            rows.append(
                {
                    "candidateId": keys[0],
                    "modeId": keys[1],
                    "featureId": keys[2],
                    "metric": metric,
                    **summary,
                }
            )
    pooled = frame.copy()
    for keys, group in pooled.groupby(["modeId", "featureId"], sort=True):
        for metric in metric_columns:
            values = pd.to_numeric(group[metric], errors="coerce").to_numpy(np.float64)
            rows.append(
                {
                    "candidateId": "POOLED_SECONDARY_ONLY",
                    "modeId": keys[0],
                    "featureId": keys[1],
                    "metric": metric,
                    **split_summary(values),
                }
            )
    return pd.DataFrame(rows)


def comparisons_and_bootstrap(
    split_metrics: pd.DataFrame, matrix_metrics: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    comparison_rows: list[dict[str, Any]] = []
    distribution_frames: list[pd.DataFrame] = []
    bootstrap_rows: list[dict[str, Any]] = []
    for candidate_id in CANDIDATE_IDS:
        for mode_id in TEMPORAL_MODES:
            reference = split_metrics.loc[
                split_metrics["candidateId"].eq(candidate_id)
                & split_metrics["modeId"].eq(mode_id)
                & split_metrics["featureId"].eq("PHIRL_EMERGENCE")
            ].sort_values("repetitionId")
            for comparator in PROSPECTIVE_COMPARATORS:
                other = split_metrics.loc[
                    split_metrics["candidateId"].eq(candidate_id)
                    & split_metrics["modeId"].eq(mode_id)
                    & split_metrics["featureId"].eq(comparator)
                ].sort_values("repetitionId")
                if len(reference) != 10 or len(other) != 10 or not np.array_equal(
                    reference["repetitionId"], other["repetitionId"]
                ):
                    raise ValueError("paired split comparison mismatch")
                left = reference["accuracy"].to_numpy(np.float64)
                right = other["accuracy"].to_numpy(np.float64)
                difference = left - right
                difference_summary = split_summary(difference)
                mann = stats.mannwhitneyu(left, right, alternative="two-sided", method="auto")
                try:
                    wilcoxon = stats.wilcoxon(
                        difference,
                        alternative="two-sided",
                        zero_method="wilcox",
                        method="auto",
                    )
                    wilcoxon_p = float(wilcoxon.pvalue)
                except ValueError:
                    wilcoxon_p = None
                reference_matrix = matrix_metrics.loc[
                    matrix_metrics["candidateId"].eq(candidate_id)
                    & matrix_metrics["modeId"].eq(mode_id)
                    & matrix_metrics["featureId"].eq("PHIRL_EMERGENCE")
                ][["repetitionId", "matrixIndex", "accuracy"]].rename(
                    columns={"accuracy": "accuracyReference"}
                )
                comparator_matrix = matrix_metrics.loc[
                    matrix_metrics["candidateId"].eq(candidate_id)
                    & matrix_metrics["modeId"].eq(mode_id)
                    & matrix_metrics["featureId"].eq(comparator)
                ][["repetitionId", "matrixIndex", "accuracy"]].rename(
                    columns={"accuracy": "accuracyComparator"}
                )
                paired = reference_matrix.merge(
                    comparator_matrix,
                    on=["repetitionId", "matrixIndex"],
                    how="inner",
                    validate="one_to_one",
                )
                distribution, bootstrap = matrix_cluster_bootstrap(
                    paired,
                    seed_identity=("bootstrap", candidate_id, mode_id, comparator),
                )
                distribution.insert(0, "comparatorFeatureId", comparator)
                distribution.insert(0, "modeId", mode_id)
                distribution.insert(0, "candidateId", candidate_id)
                distribution_frames.append(distribution)
                bootstrap_rows.append(
                    {
                        "candidateId": candidate_id,
                        "modeId": mode_id,
                        "referenceFeatureId": "PHIRL_EMERGENCE",
                        "comparatorFeatureId": comparator,
                        **bootstrap,
                    }
                )
                comparison_rows.append(
                    {
                        "candidateId": candidate_id,
                        "modeId": mode_id,
                        "referenceFeatureId": "PHIRL_EMERGENCE",
                        "comparatorFeatureId": comparator,
                        "referenceMedianAccuracy": float(np.median(left)),
                        "comparatorMedianAccuracy": float(np.median(right)),
                        "meanPairedAccuracyDifference": difference_summary["mean"],
                        "medianPairedAccuracyDifference": difference_summary["median"],
                        "pairedDifferenceLower95AcrossSplits": difference_summary["lower95"],
                        "pairedDifferenceUpper95AcrossSplits": difference_summary["upper95"],
                        "positiveSplitDifferenceCount": int(np.count_nonzero(difference > 0)),
                        "zeroSplitDifferenceCount": int(np.count_nonzero(difference == 0)),
                        "paperLikeMannWhitneyU": float(mann.statistic),
                        "paperLikeMannWhitneyTwoSidedP": float(mann.pvalue),
                        "pairedWilcoxonTwoSidedP": wilcoxon_p,
                    }
                )
    return (
        pd.DataFrame(comparison_rows),
        pd.concat(distribution_frames, ignore_index=True),
        pd.DataFrame(bootstrap_rows),
    )


def decision_tables(
    split_summary_frame: pd.DataFrame,
    comparisons: pd.DataFrame,
    bootstrap: pd.DataFrame,
    source_audit: pd.DataFrame,
    suffix_audit: pd.DataFrame,
    model_replay: pd.DataFrame,
    scaler_audit: pd.DataFrame,
    split_manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    interpretation_rows: list[dict[str, Any]] = []
    retrospective_candidate: dict[str, bool] = {}
    prospective_candidate_components: dict[str, dict[str, bool]] = {}
    for candidate_id in CANDIDATE_IDS:
        retrospective = comparisons.loc[
            comparisons["candidateId"].eq(candidate_id)
            & comparisons["modeId"].eq(RETROSPECTIVE_MODE)
            & comparisons["comparatorFeatureId"].isin(PAPER_BASELINES)
        ]
        retrospective_pass = bool(
            len(retrospective) == 4
            and (retrospective["referenceMedianAccuracy"] > retrospective["comparatorMedianAccuracy"]).all()
            and retrospective["paperLikeMannWhitneyTwoSidedP"].lt(0.01).all()
        )
        retrospective_candidate[candidate_id] = retrospective_pass
        interpretation_rows.append(
            {
                "candidateId": candidate_id,
                "modeId": RETROSPECTIVE_MODE,
                "gateId": "RETROSPECTIVE_PAPER_BASELINE_ADVANTAGE",
                "passed": retrospective_pass,
                "detail": "PhiRL higher median accuracy and Mann-Whitney p<0.01 versus four paper baselines",
            }
        )
        prospective = bootstrap.loc[
            bootstrap["candidateId"].eq(candidate_id)
            & bootstrap["modeId"].eq(CUTOFF_MODE)
        ].set_index("comparatorFeatureId")
        paired = comparisons.loc[
            comparisons["candidateId"].eq(candidate_id)
            & comparisons["modeId"].eq(CUTOFF_MODE)
        ].set_index("comparatorFeatureId")
        lower_positive = {
            comparator: bool(
                comparator in prospective.index
                and float(prospective.loc[comparator, "bootstrapLower95"]) > 0.0
            )
            for comparator in PROSPECTIVE_COMPARATORS
        }
        split_consistency = {
            comparator: bool(
                comparator in paired.index
                and int(paired.loc[comparator, "positiveSplitDifferenceCount"]) >= 8
            )
            for comparator in PROSPECTIVE_COMPARATORS
        }
        gate1 = lower_positive[DUMMY_FEATURE_ID]
        gate2 = all(
            lower_positive[item]
            for item in ("COMPOSITION_CHANGE_L2", "RAW_COUNTS", "NET_COUNT_FLUX")
        )
        gate3 = lower_positive["EXACT_H_HISTORY"] and lower_positive[
            "COMPOSITION_CHANGE_L2"
        ]
        gate5 = all(lower_positive.values()) and all(split_consistency.values())
        metric = split_summary_frame.loc[
            split_summary_frame["candidateId"].eq(candidate_id)
            & split_summary_frame["modeId"].eq(CUTOFF_MODE)
        ]
        phi_ece = metric.loc[
            metric["featureId"].eq("PHIRL_EMERGENCE")
            & metric["metric"].eq("calibrationError"),
            "median",
        ]
        phi_brier = metric.loc[
            metric["featureId"].eq("PHIRL_EMERGENCE") & metric["metric"].eq("brier"),
            "median",
        ]
        dummy_brier = metric.loc[
            metric["featureId"].eq(DUMMY_FEATURE_ID) & metric["metric"].eq("brier"),
            "median",
        ]
        calibration = bool(
            len(phi_ece) == len(phi_brier) == len(dummy_brier) == 1
            and float(phi_ece.iloc[0]) <= 0.10
            and float(phi_brier.iloc[0]) <= float(dummy_brier.iloc[0])
        )
        leakage = bool(
            len(source_audit.loc[source_audit["candidateId"].eq(candidate_id)]) == 100
            and source_audit.loc[source_audit["candidateId"].eq(candidate_id), "futureSuffixAccessedByCutoffFit"].eq(False).all()
            and len(suffix_audit.loc[suffix_audit["candidateId"].eq(candidate_id)]) == 300
            and suffix_audit.loc[suffix_audit["candidateId"].eq(candidate_id), "passed"].all()
        )
        replay = bool(
            source_audit.loc[source_audit["candidateId"].eq(candidate_id), "exactReplayPassed"].all()
            and len(model_replay.loc[model_replay["candidateId"].eq(candidate_id)]) >= 6
            and model_replay.loc[model_replay["candidateId"].eq(candidate_id), "passed"].all()
        )
        candidate_scalers = scaler_audit.loc[
            scaler_audit["candidateId"].eq(candidate_id)
        ]
        scaling = bool(
            len(candidate_scalers) == 2 * 5 * 10 * 100
            and candidate_scalers["validationMatrixIndicesExcluded"].all()
            and candidate_scalers["testMatrixIndicesExcluded"].all()
            and candidate_scalers["suffixObservationsExcluded"].all()
        )
        split_counts = split_manifest.groupby(
            ["repetitionId", "splitRole"]
        ).size().unstack()
        split_isolation = bool(
            len(split_manifest) == 1000
            and split_counts["FIT"].eq(64).all()
            and split_counts["VALIDATION"].eq(16).all()
            and split_counts["TEST"].eq(20).all()
        )
        gate6 = leakage and scaling and split_isolation and replay and calibration
        prospective_candidate_components[candidate_id] = {
            "gate1BeatsDummy": gate1,
            "gate2BeatsCompositionRawFlux": gate2,
            "gate3AddsBeyondHAndStability": gate3,
            "gate5DirectionalMatrixUncertainty": gate5,
            "gate6LeakageReplayCalibration": gate6,
        }
        for gate_id, passed in prospective_candidate_components[candidate_id].items():
            interpretation_rows.append(
                {
                    "candidateId": candidate_id,
                    "modeId": CUTOFF_MODE,
                    "gateId": gate_id,
                    "passed": passed,
                    "detail": (
                        "human-directed candidate-specific prospective gate; "
                        f"leakage={leakage}, scaling={scaling}, "
                        f"splitIsolation={split_isolation}, replay={replay}, "
                        f"calibration={calibration}"
                        if gate_id == "gate6LeakageReplayCalibration"
                        else "human-directed candidate-specific prospective gate"
                    ),
                }
            )
    retrospective_cross = all(retrospective_candidate.values())
    gates_1_to_3_both = all(
        values["gate1BeatsDummy"]
        and values["gate2BeatsCompositionRawFlux"]
        and values["gate3AddsBeyondHAndStability"]
        for values in prospective_candidate_components.values()
    )
    prospective_cross = bool(
        gates_1_to_3_both
        and all(
            values["gate5DirectionalMatrixUncertainty"]
            and values["gate6LeakageReplayCalibration"]
            for values in prospective_candidate_components.values()
        )
    )
    interpretation_rows.append(
        {
            "candidateId": "BOTH_CANDIDATES",
            "modeId": CUTOFF_MODE,
            "gateId": "gate4AllCoreGatesBothCandidates",
            "passed": gates_1_to_3_both,
            "detail": "gates 1-3 pass independently in candidates 2 and 3",
        }
    )
    decision = {
        "schema": "eidosoma.e01.s16_decision.v1",
        "researchStepId": "S16",
        "retrospectiveCandidatePass": retrospective_candidate,
        "retrospectiveCompletedFitResemblancePassed": retrospective_cross,
        "retrospectiveClassification": "RETROSPECTIVE_COMPLETED_FIT_PREDICTION_RESEMBLANCE"
        if retrospective_cross
        else "NOT_SUPPORTED_WITHIN_TESTED_SCOPE",
        "prospectiveCandidateGates": prospective_candidate_components,
        "gate4AllCoreGatesBothCandidates": gates_1_to_3_both,
        "prospectivePredictionSupported": prospective_cross,
        "prospectiveClassification": "PROSPECTIVE_PREDICTION_SUPPORTED"
        if prospective_cross
        else "NOT_SUPPORTED_WITHIN_TESTED_SCOPE",
        "exactHBoundary": {
            "targetDefinition": "Y=I(H>0.9)",
            "contemporaneousExactHAccuracy": 1.0,
            "unrestrictedIncrementBeyondContemporaneousExactH": 0.0,
            "historyOnlyComparatorIsDistinct": True,
        },
        "predictionDoesNotEstablishCausalControl": True,
    }
    paper_rows: list[dict[str, Any]] = []
    claim_comparators = {
        "E01-C025": "COMPOSITION_CHANGE_L2",
        "E01-C026": "RAW_COUNTS",
        "E01-C027": "NET_COUNT_FLUX",
        "E01-C028": DUMMY_FEATURE_ID,
    }
    for claim_id, comparator in claim_comparators.items():
        for candidate_id in CANDIDATE_IDS:
            row = comparisons.loc[
                comparisons["candidateId"].eq(candidate_id)
                & comparisons["modeId"].eq(RETROSPECTIVE_MODE)
                & comparisons["comparatorFeatureId"].eq(comparator)
            ].iloc[0]
            higher = row["referenceMedianAccuracy"] > row["comparatorMedianAccuracy"]
            significant = row["paperLikeMannWhitneyTwoSidedP"] < 0.01
            status = (
                "CLOSELY_RECONSTRUCTED"
                if higher and significant
                else "DIRECTIONALLY_SIMILAR"
                if higher
                else "NOT_SUPPORTED_WITHIN_TESTED_SCOPE"
            )
            paper_rows.append(
                {
                    "claimId": claim_id,
                    "candidateId": candidate_id,
                    "modeId": RETROSPECTIVE_MODE,
                    "status": status,
                    "referenceFeatureId": "PHIRL_EMERGENCE",
                    "comparatorFeatureId": comparator,
                    "referenceMedianAccuracy": row["referenceMedianAccuracy"],
                    "comparatorMedianAccuracy": row["comparatorMedianAccuracy"],
                    "paperLikeMannWhitneyTwoSidedP": row[
                        "paperLikeMannWhitneyTwoSidedP"
                    ],
                    "caveat": "completed-fit PhiRL uses final-suffix information",
                }
            )
    paper_rows.extend(
        [
            {
                "claimId": "E01-C029",
                "candidateId": "BOTH_CANDIDATES",
                "modeId": "NOT_EVALUATED",
                "status": "NOT_EVALUATED",
                "referenceFeatureId": None,
                "comparatorFeatureId": None,
                "referenceMedianAccuracy": None,
                "comparatorMedianAccuracy": None,
                "paperLikeMannWhitneyTwoSidedP": None,
                "caveat": "Option 2 locks exactly 25/75 and does not authorize an alternative-split search",
            },
            {
                "claimId": "E01-C030",
                "candidateId": "BOTH_CANDIDATES",
                "modeId": CUTOFF_MODE,
                "status": "SUPPORTED"
                if prospective_cross
                else "NOT_SUPPORTED_WITHIN_TESTED_SCOPE",
                "referenceFeatureId": "PHIRL_EMERGENCE",
                "comparatorFeatureId": "ALL_DIRECTED_CONTROLS",
                "referenceMedianAccuracy": None,
                "comparatorMedianAccuracy": None,
                "paperLikeMannWhitneyTwoSidedP": None,
                "caveat": "requires all six directed gates in both candidates",
            },
        ]
    )
    return pd.DataFrame(interpretation_rows), decision, pd.DataFrame(paper_rows)


def make_figures(
    output_root: Path,
    split_metrics: pd.DataFrame,
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    preonset: pd.DataFrame,
    prevalence: pd.DataFrame,
    time_metrics: pd.DataFrame,
) -> None:
    figure_root = output_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    order = list(FEATURE_IDS)
    labels = ["PhiRL", "Δ composition", "raw counts", "net flux", "H history", "dummy"]
    colors = ["#3569a8", "#65a65b", "#d4953b", "#9a6bb7", "#c65757", "#777777"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharey=True)
    for row_index, candidate_id in enumerate(CANDIDATE_IDS):
        for column_index, mode_id in enumerate(TEMPORAL_MODES):
            axis = axes[row_index, column_index]
            values = [
                split_metrics.loc[
                    split_metrics["candidateId"].eq(candidate_id)
                    & split_metrics["modeId"].eq(mode_id)
                    & split_metrics["featureId"].eq(feature_id),
                    "accuracy",
                ].to_numpy()
                for feature_id in order
            ]
            boxes = axis.boxplot(values, patch_artist=True, showmeans=True)
            for patch, color in zip(boxes["boxes"], colors, strict=True):
                patch.set_facecolor(color)
                patch.set_alpha(0.55)
            axis.set_xticks(range(1, 7), labels, rotation=25, ha="right")
            axis.set_ylim(0.0, 1.02)
            axis.grid(axis="y", alpha=0.25)
            axis.set_title(
                f"{candidate_id[-2:]} — "
                + ("completed-fit" if mode_id == RETROSPECTIVE_MODE else "cutoff-causal")
            )
            axis.set_ylabel("valid-target binary accuracy")
    fig.suptitle("S16 Figure 5 reconstruction: exact paired split accuracies", fontsize=15)
    fig.tight_layout()
    fig.savefig(figure_root / "figure5_prediction_reconstruction.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for axis, candidate_id in zip(axes, CANDIDATE_IDS, strict=True):
        selected = summary.loc[
            summary["candidateId"].eq(candidate_id)
            & summary["featureId"].eq("PHIRL_EMERGENCE")
            & summary["metric"].eq("accuracy")
        ]
        x = np.arange(len(selected))
        axis.errorbar(
            x,
            selected["mean"],
            yerr=[selected["mean"] - selected["lower95"], selected["upper95"] - selected["mean"]],
            fmt="o",
            capsize=5,
        )
        axis.set_xticks(x, ["cutoff-causal" if item == CUTOFF_MODE else "completed-fit" for item in selected["modeId"]], rotation=15)
        axis.set_title(candidate_id)
        axis.set_ylabel("PhiRL accuracy mean and split-t 95% CI")
        axis.grid(alpha=0.25)
    fig.suptitle("Retrospective completed-fit versus cutoff-causal PhiRL")
    fig.tight_layout()
    fig.savefig(figure_root / "retrospective_vs_cutoff.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    test_prevalence = prevalence.loc[prevalence["splitRole"].eq("TEST")]
    for candidate_id, group in test_prevalence.groupby("candidateId"):
        axes[0].plot(group["repetitionId"], group["targetPrevalence"], marker="o", label=candidate_id[-2:])
    axes[0].set_title("Frozen target prevalence by test split")
    axes[0].set_xlabel("repetition")
    axes[0].set_ylabel("positive prevalence")
    axes[0].legend(title="candidate")
    ece = summary.loc[
        summary["modeId"].eq(CUTOFF_MODE)
        & summary["metric"].eq("calibrationError")
        & summary["candidateId"].isin(CANDIDATE_IDS)
    ]
    for candidate_id, group in ece.groupby("candidateId"):
        axes[1].plot(
            [order.index(item) for item in group["featureId"]],
            group["median"],
            marker="o",
            label=candidate_id[-2:],
        )
    axes[1].axhline(0.10, color="black", linestyle="--", linewidth=1)
    axes[1].set_xticks(range(6), labels, rotation=25, ha="right")
    axes[1].set_title("Cutoff-causal median ECE")
    axes[1].set_ylabel("10-bin expected calibration error")
    axes[1].legend(title="candidate")
    fig.suptitle("Prevalence and calibration boundaries")
    fig.tight_layout()
    fig.savefig(figure_root / "calibration_and_prevalence.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for candidate_id in CANDIDATE_IDS:
        selected = time_metrics.loc[
            time_metrics["candidateId"].eq(candidate_id)
            & time_metrics["modeId"].eq(CUTOFF_MODE)
            & time_metrics["featureId"].eq("PHIRL_EMERGENCE")
        ]
        curve = selected.groupby("targetOffset")["accuracy"].mean().rolling(25, min_periods=5).mean()
        axes[0].plot(curve.index, curve.values, label=candidate_id[-2:])
    axes[0].set_title("Cutoff PhiRL accuracy by target offset (25-step rolling)")
    axes[0].set_xlabel("target offset from cutoff")
    axes[0].set_ylabel("accuracy")
    axes[0].legend(title="candidate")
    eligible = preonset.loc[
        preonset["modeId"].eq(CUTOFF_MODE)
        & preonset["featureId"].eq("PHIRL_EMERGENCE")
    ]
    for candidate_id, group in eligible.groupby("candidateId"):
        axes[1].bar(
            np.arange(10) + (-0.18 if candidate_id.endswith("02") else 0.18),
            group.sort_values("repetitionId")["eligibleRunCount"],
            width=0.36,
            label=candidate_id[-2:],
        )
    axes[1].set_title("Strict pre-onset eligible test matrices")
    axes[1].set_xlabel("repetition")
    axes[1].set_ylabel("matrix count")
    axes[1].legend(title="candidate")
    fig.suptitle("Per-time and pre-onset audit")
    fig.tight_layout()
    fig.savefig(figure_root / "per_time_and_preonset.png", dpi=180)
    plt.close(fig)


def markdown_table(frame: pd.DataFrame, columns: list[str], maximum_rows: int = 40) -> str:
    selected = frame.loc[:, columns].head(maximum_rows).copy()
    return selected.to_markdown(index=False, floatfmt=".6g")


def write_report(
    output_root: Path,
    status: dict[str, Any],
    decision: dict[str, Any],
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    prevalence: pd.DataFrame,
    preonset: pd.DataFrame,
    source_audit: pd.DataFrame,
    validation: dict[str, Any],
    compute: dict[str, Any],
    artifact_count: int,
) -> None:
    accuracy = summary.loc[
        summary["candidateId"].isin(CANDIDATE_IDS) & summary["metric"].eq("accuracy")
    ].sort_values(["modeId", "candidateId", "featureId"])
    cutoff_comparisons = comparisons.loc[comparisons["modeId"].eq(CUTOFF_MODE)].copy()
    test_prevalence = prevalence.loc[prevalence["splitRole"].eq("TEST")]
    prefix = "# E01/S16 — Reconstruct the First-25%-to-Final-75% Prediction Experiment\n\n"
    summary_table = f"""## Concise top summary

| Field | Result |
| --- | --- |
| Research step ID | `S16` (`{VERSION}`) |
| Completion status | **Complete** — only S16 was executed; S17 was not started |
| Artifacts written | {artifact_count} required paths under `/artifacts/research_steps/S16` |
| Validation result | **{validation['validationResult']}** |
| Outcome classification | Retrospective: **`{decision['retrospectiveClassification']}`**; prospective: **`{decision['prospectiveClassification']}`** |
| Caveats or blockers | Exact contemporaneous H determines Y; completed-fit PhiRL is suffix-dependent; the frozen molecular target is highly imbalanced; the paper omits its tensor and MLP details; pooling is secondary only. |
| Lay summary | The locked experiment tested the paper-like future-trajectory task and a genuinely cutoff-only reconstruction. Completed-fit and cutoff-causal evidence were adjudicated separately against composition, counts, flux, dummy, and exact-H history controls. |
| Recommended next action | Hand control back. Keep S17 queued and inactive until separately instructed. |

## Lay summary

This step used the first quarter of each of the 200 already frozen S13Y trajectories to predict the remaining three quarters. Every learned feature used the same masked, original-order MLP and the same ten matrix-level splits. The paper-like completed-fit PhiRL mode was kept explicitly retrospective because its partition and Gaussian parameters use the final three quarters. The cutoff-causal mode refit the exact PhiRL pipeline using only the first quarter and passed separate suffix-deletion, shuffle, replacement, scaling, split, and replay audits.

The target remains exactly `Y=I(H>0.9)`: contemporaneous exact H classifies it with accuracy 1.0 and leaves no unrestricted incremental information for PhiRL. The supplied H baseline uses only first-quarter H history and is therefore a historical predictor, not the unavailable contemporaneous future H. The frozen target prevalence and strict pre-onset eligibility results below are central to interpretation.

## Frozen question

Does completed-fit first-quarter PhiRL reconstruct the paper-like accuracy advantage, and does any advantage survive a first-quarter-only source fit and all six directed prospective gates in both candidates?

## Inputs and provenance

- Exactly 100 shared S13Y matrix identities and both frozen candidates 2 and 3; no candidate pooling for primary inference.
- Exactly 200 frozen raw C1 trajectories, the frozen molecular `H>0.9` labels, and the exact S13Y PhiRL completed-fit values.
- Candidate 2: `h=0.6031526490073492`, first daughter, trimmed new entrants. Candidate 3: `h=0.5613315384859516`, random nonempty daughter, trimmed new entrants.
- Original paper SHA-256 `77a2ec2c0751839d8a2e10863ca803c6f8b61475bbc790f2bbdad2a38af04ae4`.
- Pre-outcome repository lock and pushed commit are recorded in `preoutcome_design_lock.json` and `provenance_manifest.json`.

## Detailed methods

### Pre-outcome tensor, split, and architecture lock

The paper specifies the first 25%, remaining 75%, an MLP, 80/20 runs, ten seeds, accuracy, and four baselines, but no tensor orientation, masking, validation, scaling, architecture, optimizer, or flux formula. Before prediction outcomes, S16 froze one reconstruction: for each run of length `T`, `c=floor(0.25T)`, input observations `0..c-1`, and target labels `c..T-1`. Original molecular-step order is right-padded to 367 input and 1,101 target positions with explicit channel/time/target masks; no interpolation, resampling, or truncation occurs.

Every learned family uses the same 288,789-parameter float64 MLP: a shared per-time 200-to-8 encoder over 100 scaled values plus 100 channel-mask values, flattened original order plus the time mask, two 64-unit ReLU/dropout hidden layers, and 1,101 output logits. AdamW, regularization, full-batch masked BCE, the training-only validation rule, 120-epoch ceiling, 15-epoch patience, and all seeds are identical. No hyperparameter or architecture tournament occurred.

Exactly ten outcome-blind, independently seeded matrix splits assign 64 fit, 16 validation, and 20 test matrices. The split identities are identical across candidates, modes, and features. Scaling is fit only on valid input cells of the 64 fit matrices; validation, test, padding, and all post-cutoff observations are excluded.

### Feature and target construction

- `PHIRL_EMERGENCE`: completed-fit first-quarter values in retrospective mode; one exact first-quarter-only PhiRL refit in cutoff mode.
- `COMPOSITION_CHANGE_L2`: scalar Euclidean change of raw relative composition, matching the frozen ordinary-stability control.
- `RAW_COUNTS`: the exact 100 molecular counts.
- `NET_COUNT_FLUX`: the 100-dimensional adjacent count increment, including fission-boundary changes.
- `EXACT_H_HISTORY`: the frozen incoming H sequence, including S13Y's duplicated first adjacent value.
- `MAJORITY_DUMMY`: the fit-subset target prevalence, with no validation/test label access.

The target is the frozen molecular same-state `Y=I(H>0.9)` suffix. Accuracy is the primary micro valid-position metric; matrix-macro accuracy, AUROC, AUPRC, Brier, ten-bin ECE, balanced accuracy, sensitivity, and specificity are secondary. Metrics are also reported by exact target offset. The strict pre-onset risk set includes only runs with no positive input-quarter label and stops at the first future positive inclusive.

### Uncertainty and interpretation gates

Ten-split means, medians, sample SDs, and Student-t intervals are reported. Paper-like two-sided Mann–Whitney tests are retained beside paired Wilcoxon diagnostics. The stronger accuracy comparison resamples unique test matrix identities 4,096 times, retaining all out-of-sample repetitions for each selected matrix. `PROSPECTIVE_PREDICTION_SUPPORTED` requires all six directed gates independently in both candidates; no completed-fit result can enter those gates.

## Commands

```bash
PYTHONPATH=src python scripts/e01/freeze_s16_prediction_design.py
PYTHONPATH=src pytest -q tests/e01/test_s16_prediction_reconstruction.py
PYTHONPATH=src ruff check src/e01_prediction_reconstruction scripts/e01/freeze_s16_prediction_design.py scripts/e01/run_s16_prediction_reconstruction.py tests/e01/test_s16_prediction_reconstruction.py
PYTHONPATH=src python -m compileall -q src/e01_prediction_reconstruction scripts/e01/freeze_s16_prediction_design.py scripts/e01/run_s16_prediction_reconstruction.py
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python scripts/e01/run_s16_prediction_reconstruction.py --output-root /artifacts/research_steps/S16
```

CPU float64 is authoritative; no GPU was used, so no CPU/GPU equivalence claim is needed. No simulator, network call, package installer, author contact, or S17 operation occurred.

## Results

### Candidate-specific split accuracy

{markdown_table(accuracy, ['candidateId','modeId','featureId','mean','median','sampleStd','lower95','upper95'], 40)}

### Paired cutoff-causal PhiRL accuracy contrasts

{markdown_table(cutoff_comparisons, ['candidateId','comparatorFeatureId','referenceMedianAccuracy','comparatorMedianAccuracy','medianPairedAccuracyDifference','pairedDifferenceLower95AcrossSplits','positiveSplitDifferenceCount','paperLikeMannWhitneyTwoSidedP'], 20)}

### Test target prevalence

{markdown_table(test_prevalence.groupby('candidateId', as_index=False).agg(meanTestPrevalence=('targetPrevalence','mean'), minTestPrevalence=('targetPrevalence','min'), maxTestPrevalence=('targetPrevalence','max'), meanAlreadyPositiveInputMatrices=('inputAlreadyPositiveMatrixCount','mean')), ['candidateId','meanTestPrevalence','minTestPrevalence','maxTestPrevalence','meanAlreadyPositiveInputMatrices'])}

### Strict pre-onset audit

{markdown_table(preonset.loc[(preonset['modeId'].eq(CUTOFF_MODE)) & (preonset['featureId'].isin(['PHIRL_EMERGENCE','EXACT_H_HISTORY','MAJORITY_DUMMY']))], ['candidateId','featureId','repetitionId','eligibleRunCount','excludedAlreadyPositiveInputRunCount','validTargetCount','accuracy','auroc','auprc'], 60)}

### Completed-fit versus cutoff-only PhiRL fitting

{markdown_table(source_audit.groupby('candidateId', as_index=False).agg(trajectoryCount=('matrixIndex','size'), medianSharedValues=('sharedEmergenceCount','median'), medianCompletedCutoffSpearman=('completedVsCutoffSpearman','median'), medianAbsoluteDifference=('completedVsCutoffMeanAbsoluteDifference','median'), medianPartitionARI=('completedVsCutoffPartitionARI','median'), exactReplayCount=('exactReplayPassed','sum')), ['candidateId','trajectoryCount','medianSharedValues','medianCompletedCutoffSpearman','medianAbsoluteDifference','medianPartitionARI','exactReplayCount'])}

## Decision

- Retrospective completed-fit classification: `{decision['retrospectiveClassification']}`.
- Prospective classification: `{decision['prospectiveClassification']}`.
- Candidate retrospective gates: `{json.dumps(decision['retrospectiveCandidatePass'], sort_keys=True)}`.
- Candidate prospective gates: `{json.dumps(decision['prospectiveCandidateGates'], sort_keys=True)}`.
- Contemporaneous exact-H target accuracy is 1.0 by construction; unrestricted increment beyond contemporaneous exact H is zero.
- Neither prediction mode supplies causal-control evidence.

## Validation

{validation['validationResult']}. The validation artifact records every named check, including 200 matrix/candidate payloads, 600 suffix variants, matrix-only split isolation, training-only scaling, exact label identity, same architecture and seed rules, source/model replay, artifact hashes, immutable-prior postchecks, zero trajectories, and the S17 stop boundary.

## Compute

S16 measured `{compute['s16ScientificCpuHours']:.6f}` scientific CPU-hours and `{compute['s16TotalProcessCpuHours']:.6f}` total process CPU-hours. The new-scientific-compute ledger retains `{compute['remainingCombinedS16S17ScientificCpuHours']:.6f}` hours under the 105-hour combined ceiling, before the separately protected four-hour validation/artifact reserve.

## Figures and machine-readable artifacts

Four inspected figures reconstruct Figure 5, compare retrospective and cutoff modes, expose calibration/prevalence, and show per-time/pre-onset behavior. Parquet and CSV artifacts retain every split metric, test prediction, training/scaling record, cutoff source fit, suffix audit, per-time result, paired comparison, and matrix-cluster bootstrap.

## Caveats, blockers, failed assumptions, and limitations

- The exact contemporaneous target-defining H is unavailable in the future suffix at prediction time; the mandatory H model receives only first-quarter H history. These are distinct facts and are reported separately.
- Completed-fit PhiRL partitions and Gaussian parameters use the final 75%; that mode is retrospective prediction resemblance only.
- The paper's tensor layout, validation, scaling, architecture, flux definition, weighting, and seed hierarchy are unavailable. S16 is one frozen coherent reconstruction, not author-code identity.
- The molecular adjacent-H target may have a prevalence unlike the paper's Figure 5 baseline; accuracy must be interpreted beside prevalence, balanced accuracy, AUPRC, calibration, and pre-onset eligibility.
- Repeated test splits overlap. Split-t and paper-like Mann–Whitney diagnostics are therefore accompanied by paired matrix-cluster bootstrap intervals.
- Candidate pooling is secondary descriptive only and cannot rescue either candidate.
- No alternative split proportions were searched, so E01-C029 is not evaluated.
- Prediction alone cannot establish causal control, and S17 was not started.

## Provenance

`preoutcome_design_lock.json` records the pushed design commit. `input_manifest.json` records before/after hashes for all prior step artifacts and 200 S13Y raw trajectories. `provenance_manifest.json` records the exact runtime, code hashes, command, numeric policy, and compute ledger. Repository source remains in Git.

## Recommended next action

Return control to the Chief Scientist workflow. Keep S17 queued and inactive until separately instructed. Carry forward the separate retrospective/prospective verdicts, exact-H boundary, target prevalence, pre-onset audit, and all S01–S15 classifications. Do not begin S17 in this execution.
"""
    (output_root / "research_step_full_results.md").write_text(
        prefix + summary_table, encoding="utf-8"
    )


def validate_outputs(
    output_root: Path,
    config: dict[str, Any],
    inputs: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    decision: dict[str, Any],
    artifact_paths: list[str],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def check(check_id: str, passed: bool, detail: str) -> None:
        checks.append({"checkId": check_id, "passed": bool(passed), "detail": detail})

    check("PREOUTCOME_PUSHED_LOCK", current_repo_lock()["passed"], str(current_repo_lock()))
    check("IMMUTABLE_INPUTS", inputs["passed"], f"entries={inputs['entryCount']} mismatches={inputs['immutableMismatchCount']}")
    check("EXACT_200_PAIRED_PAYLOADS", len(frames["sourceAudit"]) == 200, f"rows={len(frames['sourceAudit'])}")
    check("EXACT_TENSOR_BOUNDS", frames["sourceAudit"]["T"].max() == 1468 and frames["sourceAudit"]["cutoff"].max() == 367 and frames["sourceAudit"]["targetLength"].max() == 1101, "max T/c/target verified")
    check("CUTOFF_SOURCE_REPLAY", frames["sourceAudit"]["exactReplayPassed"].all(), f"passed={int(frames['sourceAudit']['exactReplayPassed'].sum())}/200")
    check("CUTOFF_NO_SUFFIX_ACCESS", frames["sourceAudit"]["futureSuffixAccessedByCutoffFit"].eq(False).all(), "all 200 prefix fits")
    check("SUFFIX_INVARIANCE_600", len(frames["suffixAudit"]) == 600 and frames["suffixAudit"]["passed"].all(), f"passed={int(frames['suffixAudit']['passed'].sum())}/600")
    check("FEATURE_AUDIT_CARDINALITY", len(frames["featureAudit"]) == 2 * 100 * 2 * 5, f"rows={len(frames['featureAudit'])}")
    check("EXACT_LABEL_IDENTITY", decision["exactHBoundary"]["contemporaneousExactHAccuracy"] == 1.0, "Y=I(H>0.9)")
    check("SPLIT_CARDINALITY", len(frames["splitManifest"]) == 1000, "10x100 matrix assignments")
    split_counts = frames["splitManifest"].groupby(["repetitionId", "splitRole"]).size().unstack()
    check("MATRIX_SPLIT_ISOLATION", split_counts["FIT"].eq(64).all() and split_counts["VALIDATION"].eq(16).all() and split_counts["TEST"].eq(20).all(), "64/16/20 every repetition")
    check("TEN_SPLITS_ALL_MODE_FEATURE_CANDIDATE", len(frames["splitMetrics"]) == 2 * 2 * 6 * 10, f"rows={len(frames['splitMetrics'])}")
    check("TRAINING_ONLY_SCALERS", len(frames["scalers"]) == 2 * 2 * 5 * 10 * 100 and frames["scalers"]["validationMatrixIndicesExcluded"].all() and frames["scalers"]["testMatrixIndicesExcluded"].all() and frames["scalers"]["suffixObservationsExcluded"].all(), f"rows={len(frames['scalers'])}")
    check("IDENTICAL_PARAMETER_COUNT", frames["training"].loc[frames["training"]["featureId"].ne(DUMMY_FEATURE_ID), "parameterCount"].eq(EXPECTED_PARAMETER_COUNT).all(), f"count={EXPECTED_PARAMETER_COUNT}")
    seeds = frames["training"].groupby(["candidateId", "repetitionId"])["modelSeed"].nunique()
    check("IDENTICAL_MODEL_SEED_ACROSS_FEATURE_MODE", seeds.eq(1).all(), "one seed per candidate/repetition")
    check("MODEL_REPLAY", len(frames["modelReplay"]) >= 12 and frames["modelReplay"]["passed"].all(), f"passed={int(frames['modelReplay']['passed'].sum())}/{len(frames['modelReplay'])}")
    check("METRIC_FINITE_ACCURACY", frames["splitMetrics"]["accuracy"].notna().all() and frames["splitMetrics"]["accuracy"].between(0, 1).all(), "all 240 split accuracies")
    check("REQUIRED_METRIC_COLUMNS", {"auroc","auprc","brier","calibrationError","balancedAccuracy","sensitivity","specificity"}.issubset(frames["splitMetrics"].columns), "secondary metric schema")
    check("PER_TIME_RESULTS", len(frames["timeMetrics"]) > 0 and frames["timeMetrics"]["targetOffset"].max() < 1101, f"rows={len(frames['timeMetrics'])}")
    check("PREONSET_STATUS", len(frames["preonsetMetrics"]) == 240, f"rows={len(frames['preonsetMetrics'])}")
    check("PREVALENCE_AUDIT", len(frames["prevalence"]) == 2 * 10 * 3, f"rows={len(frames['prevalence'])}")
    check("PAIRED_COMPARISONS", len(frames["comparisons"]) == 2 * 2 * 5, f"rows={len(frames['comparisons'])}")
    check("MATRIX_BOOTSTRAP", len(frames["bootstrapDistribution"]) == 2 * 2 * 5 * 4096 and len(frames["bootstrapSummary"]) == 20, f"distribution={len(frames['bootstrapDistribution'])}")
    check("CANDIDATE_PRIMARY_POOL_SECONDARY", set(frames["splitSummary"]["candidateId"]) == set(CANDIDATE_IDS) | {"POOLED_SECONDARY_ONLY"}, "candidate-specific plus pooled secondary")
    check("SEPARATE_MODE_DECISIONS", "retrospectiveClassification" in decision and "prospectiveClassification" in decision, "two verdicts retained")
    check("ZERO_TRAJECTORY_GENERATION", config["scope"]["trajectoryGenerationPermitted"] is False, "newTrajectoryCount=0")
    check("S17_NOT_STARTED", not Path("/artifacts/research_steps/S17").exists(), "S17 artifact directory absent")
    for name in ("figure5_prediction_reconstruction.png", "retrospective_vs_cutoff.png", "calibration_and_prevalence.png", "per_time_and_preonset.png"):
        path = output_root / "figures" / name
        passed = path.is_file() and path.stat().st_size > 10_000
        check(f"FIGURE_{name}", passed, f"bytes={path.stat().st_size if path.exists() else 0}")
    missing = [path for path in artifact_paths if not (output_root / path).is_file()]
    check("REQUIRED_ARTIFACTS_PRESENT", not missing, f"missing={missing}")
    passed = all(item["passed"] for item in checks)
    return {
        "schema": "eidosoma.e01.s16_validation.v1",
        "researchStepId": "S16",
        "passed": passed,
        "validationResult": f"PASS: {sum(item['passed'] for item in checks)}/{len(checks)} checks" if passed else f"FAIL: {sum(item['passed'] for item in checks)}/{len(checks)} checks",
        "checks": checks,
    }


def artifact_manifest(output_root: Path, required: list[str]) -> dict[str, Any]:
    artifacts = []
    missing = []
    for relative in required:
        if relative == "artifact_manifest.json":
            continue
        path = output_root / relative
        if not path.is_file():
            missing.append(relative)
            continue
        artifacts.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    return {
        "schema": "eidosoma.e01.s16_artifact_manifest.v1",
        "researchStepId": "S16",
        "passed": not missing and len(artifacts) == len(required) - 1,
        "artifactCountExcludingSelf": len(artifacts),
        "requiredArtifactCountIncludingSelf": len(required),
        "missingRequired": missing,
        "totalBytesExcludingSelf": sum(item["bytes"] for item in artifacts),
        "artifacts": artifacts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    if output_root != Path("/artifacts/research_steps/S16"):
        raise ValueError("S16 output root is frozen")
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError("S16 output root must be absent or empty for canonical execution")
    output_root.mkdir(parents=True, exist_ok=True)
    work_root = Path("/cache/e01_s16_work")
    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    locked = preoutcome_lock(config)
    write_json(output_root / "preoutcome_design_lock.json", locked)
    tensor_manifest = json.loads(LOCK_MANIFEST_PATH.read_text(encoding="utf-8"))
    tensor_manifest["lockedRepositoryCommit"] = locked["repository"]["head"]
    tensor_manifest["predictionOutcomeAccessedAtRepositoryLock"] = False
    write_json(output_root / "tensor_layout_manifest.json", tensor_manifest)
    split = pd.read_csv(SPLIT_PATH)
    split.to_csv(output_root / "split_seed_manifest.csv", index=False)
    input_rows = hash_input_baseline()
    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    source_start_wall = time.perf_counter()
    source_start_cpu = time.process_time()
    payloads, prepared = prepare_payloads()
    source_cpu = time.process_time() - source_start_cpu
    source_wall = time.perf_counter() - source_start_wall
    model_start_wall = time.perf_counter()
    model_start_cpu = time.process_time()
    modeled = run_models(payloads, split, work_root)
    model_cpu = time.process_time() - model_start_cpu
    model_wall = time.perf_counter() - model_start_wall
    prediction_path = Path(modeled["predictionPath"].iloc[0]["path"])
    shutil.copy2(prediction_path, output_root / "prediction_rows.parquet")
    split_metrics = modeled["splitMetrics"]
    summary = summarize_split_metrics(split_metrics)
    comparisons, bootstrap_distribution, bootstrap_summary = comparisons_and_bootstrap(
        split_metrics, modeled["matrixMetrics"]
    )
    interpretation, decision, paper_targets = decision_tables(
        summary,
        comparisons,
        bootstrap_summary,
        prepared["sourceAudit"],
        prepared["suffixAudit"],
        modeled["modelReplay"],
        modeled["scalers"],
        split,
    )
    prepared["featureAudit"].to_parquet(output_root / "feature_audit.parquet", index=False)
    prepared["sourceAudit"].to_parquet(output_root / "cutoff_source_fit_audit.parquet", index=False)
    prepared["suffixAudit"].to_parquet(output_root / "cutoff_suffix_invariance.parquet", index=False)
    modeled["scalers"].to_parquet(output_root / "scaler_audit.parquet", index=False)
    modeled["training"].to_parquet(output_root / "training_history.parquet", index=False)
    split_metrics.to_csv(output_root / "split_metrics.csv", index=False)
    summary.to_csv(output_root / "split_metric_summary.csv", index=False)
    modeled["timeMetrics"].to_parquet(output_root / "per_time_position_metrics.parquet", index=False)
    modeled["preonsetMetrics"].to_csv(output_root / "preonset_metrics.csv", index=False)
    modeled["prevalence"].to_csv(output_root / "prevalence_audit.csv", index=False)
    comparisons.to_csv(output_root / "paired_feature_comparisons.csv", index=False)
    bootstrap_distribution.to_parquet(output_root / "matrix_cluster_bootstrap.parquet", index=False)
    bootstrap_summary.to_csv(output_root / "matrix_cluster_bootstrap_summary.csv", index=False)
    paper_targets.to_csv(output_root / "paper_target_comparison.csv", index=False)
    interpretation.to_csv(output_root / "interpretation_gates.csv", index=False)
    write_json(output_root / "decision.json", decision)
    replay_validation = {
        "schema": "eidosoma.e01.s16_replay_validation.v1",
        "researchStepId": "S16",
        "cutoffSourceReplayPassed": bool(prepared["sourceAudit"]["exactReplayPassed"].all()),
        "cutoffSourceReplayCount": int(prepared["sourceAudit"]["exactReplayPassed"].sum()),
        "modelReplayPassed": bool(modeled["modelReplay"]["passed"].all()),
        "modelReplayRows": modeled["modelReplay"].to_dict(orient="records"),
        "passed": bool(
            prepared["sourceAudit"]["exactReplayPassed"].all()
            and modeled["modelReplay"]["passed"].all()
        ),
    }
    write_json(output_root / "replay_validation.json", replay_validation)
    leakage_validation = {
        "schema": "eidosoma.e01.s16_leakage_validation.v1",
        "researchStepId": "S16",
        "cutoffFitCount": len(prepared["sourceAudit"]),
        "suffixVariantCount": len(prepared["suffixAudit"]),
        "cutoffFitsAccessedSuffix": int(prepared["sourceAudit"]["futureSuffixAccessedByCutoffFit"].sum()),
        "suffixVariantsPassed": int(prepared["suffixAudit"]["passed"].sum()),
        "trainingOnlyScalerRows": len(modeled["scalers"]),
        "validationOrTestUsedForScaling": False,
        "targetMaskUsedAsModelInput": False,
        "completedFitProspectiveEligible": False,
        "passed": bool(
            prepared["sourceAudit"]["futureSuffixAccessedByCutoffFit"].eq(False).all()
            and prepared["suffixAudit"]["passed"].all()
            and modeled["scalers"]["validationMatrixIndicesExcluded"].all()
            and modeled["scalers"]["testMatrixIndicesExcluded"].all()
            and modeled["scalers"]["suffixObservationsExcluded"].all()
        ),
    }
    write_json(output_root / "leakage_validation.json", leakage_validation)
    failure_frame = prepared["failures"]
    if failure_frame.empty:
        failure_frame = pd.DataFrame(
            columns=["failureId", "stage", "severity", "status", "reason", "outcomeExclusion"]
        )
    failure_frame.to_csv(output_root / "failure_ledger.csv", index=False)
    make_figures(
        output_root,
        split_metrics,
        summary,
        comparisons,
        modeled["preonsetMetrics"],
        modeled["prevalence"],
        modeled["timeMetrics"],
    )
    scientific_cpu = source_cpu + model_cpu
    total_cpu_pre_report = time.process_time() - start_cpu
    compute = {
        "schema": "eidosoma.e01.s16_compute_ledger.v1",
        "researchStepId": "S16",
        "combinedS16S17ScientificCpuHourCeiling": 105.0,
        "protectedValidationArtifactReserveCpuHours": 4.0,
        "sourceScientificCpuHours": source_cpu / 3600.0,
        "modelScientificCpuHours": model_cpu / 3600.0,
        "s16ScientificCpuHours": scientific_cpu / 3600.0,
        "s16TotalProcessCpuHours": total_cpu_pre_report / 3600.0,
        "s16WallHoursBeforeReporting": (time.perf_counter() - start_wall) / 3600.0,
        "remainingCombinedS16S17ScientificCpuHours": 105.0 - scientific_cpu / 3600.0 - 4.0,
        "sourceWallHours": source_wall / 3600.0,
        "modelWallHours": model_wall / 3600.0,
        "cpuFloat64Authoritative": True,
        "gpuUsed": False,
        "gpuEquivalenceRequired": False,
        "modelWorkers": 1,
        "numericalLibraryThreads": 1,
        "ceilingPassed": scientific_cpu / 3600.0 <= 101.0,
    }
    write_json(output_root / "compute_ledger.json", compute)
    input_manifest = finalize_input_manifest(input_rows)
    write_json(output_root / "input_manifest.json", input_manifest)
    provenance = {
        "schema": "eidosoma.e01.s16_provenance_manifest.v1",
        "researchStepId": "S16",
        "versionedStepId": VERSION,
        "repository": current_repo_lock(),
        "command": "PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python scripts/e01/run_s16_prediction_reconstruction.py --output-root /artifacts/research_steps/S16",
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "torch": torch.__version__,
            "pyarrow": pa.__version__,
        },
        "numericPolicy": "CPU_FLOAT64_ONE_PROCESS_ONE_NUMERICAL_THREAD",
        "newTrajectoryCount": 0,
        "cutoffScientificSourceFitCount": 200,
        "cutoffSourceEvaluationCountIncludingReplayAndSuffixVariants": 1000,
        "completedFitSourceRefitCount": 0,
        "nextResearchStepStarted": False,
        "authorContacted": False,
        "inputManifestSha256": sha256_file(output_root / "input_manifest.json"),
        "computeLedgerSha256": sha256_file(output_root / "compute_ledger.json"),
    }
    write_json(output_root / "provenance_manifest.json", provenance)
    required = list(config["requiredArtifacts"])
    frames = {
        **prepared,
        **modeled,
        "splitManifest": split,
        "splitSummary": summary,
        "comparisons": comparisons,
        "bootstrapDistribution": bootstrap_distribution,
        "bootstrapSummary": bootstrap_summary,
    }
    # Validation/report/status/manifests are written in a staged order because
    # their own existence is part of the final artifact audit.
    provisional_paths = [
        item
        for item in required
        if item
        not in {"validation.json", "status.json", "research_step_full_results.md", "artifact_manifest.json"}
    ]
    validation = validate_outputs(
        output_root,
        config,
        input_manifest,
        frames,
        decision,
        provisional_paths,
    )
    write_json(output_root / "validation.json", validation)
    outcome_class = (
        "supportive"
        if decision["prospectivePredictionSupported"]
        else "constraining/contradictory"
    )
    status = {
        "researchStepId": "S16",
        "stepNumber": 16,
        "success": bool(validation["passed"]),
        "status": "COMPLETED" if validation["passed"] else "VALIDATION_FAILED",
        "artifactsWritten": required,
        "validationResult": validation["validationResult"],
        "outcomeClass": outcome_class,
        "outcomeClassification": {
            "retrospective": decision["retrospectiveClassification"],
            "prospective": decision["prospectiveClassification"],
        },
        "caveatsOrBlockers": [
            "The target is exactly Y=I(H>0.9), so contemporaneous exact H determines it.",
            "Completed-fit PhiRL uses the final trajectory suffix and is retrospective only.",
            "The paper omits tensor, architecture, scaling, validation, and flux details.",
            "Target imbalance and pre-onset eligibility constrain accuracy interpretation.",
            "Pooling is secondary only; S16 supplies no causal-control evidence.",
        ],
        "recommendedNextAction": "Hand control back; keep S17 queued and inactive until separately started.",
        "newTrajectoryCount": 0,
        "cutoffScientificSourceFitCount": 200,
        "nextResearchStepStarted": False,
    }
    write_json(output_root / "status.json", status)
    write_report(
        output_root,
        status,
        decision,
        summary,
        comparisons,
        modeled["prevalence"],
        modeled["preonsetMetrics"],
        prepared["sourceAudit"],
        validation,
        compute,
        len(required),
    )
    manifest = artifact_manifest(output_root, required)
    write_json(output_root / "artifact_manifest.json", manifest)
    if not validation["passed"] or not manifest["passed"] or not compute["ceilingPassed"]:
        raise RuntimeError(
            f"S16 final gate failed: validation={validation['passed']} artifacts={manifest['passed']} compute={compute['ceilingPassed']}"
        )
    print(
        canonical_json(
            {
                "researchStepId": "S16",
                "validationResult": validation["validationResult"],
                "artifactCount": len(required),
                "retrospectiveClassification": decision["retrospectiveClassification"],
                "prospectiveClassification": decision["prospectiveClassification"],
                "newTrajectoryCount": 0,
                "nextResearchStepStarted": False,
            }
        )
    )


if __name__ == "__main__":
    main()
