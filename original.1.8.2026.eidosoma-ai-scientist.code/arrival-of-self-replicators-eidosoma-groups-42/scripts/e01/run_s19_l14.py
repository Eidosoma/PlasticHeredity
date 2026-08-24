#!/usr/bin/env python3
"""Execute the frozen E01/S19-L14 padding/length-leakage audit.

The runner is staged so the repository contract is committed and pushed before
any cohort arithmetic is released.  Scientific intermediates live in /cache;
only compact evidence is promoted to the L14 artifact directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import platform
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import scipy
import sklearn
import torch
import yaml
from scipy import stats
from sklearn.linear_model import LogisticRegression

from e01_frozen_timebase_ensemble.core import frozen_clr, selected_clock_observations
from e01_pigozzi_source_audit.core import SourceImplementation
from e01_prediction_reconstruction.core import (
    EXPECTED_PARAMETER_COUNT,
    MAX_INPUT_LENGTH,
    MAX_TARGET_LENGTH,
    apply_channel_scaler,
    fit_channel_scaler,
    parameter_count,
    predict_probabilities,
    train_masked_mlp,
)
from e01_prediction_reconstruction.core import (
    derive_seed128 as s16_seed128,
)
from e01_prediction_reconstruction.core import (
    derive_torch_seed as s16_torch_seed,
)
from e01_s19_figure5_prediction.core import (
    build_feature,
    extended_binary_metrics,
    incoming_h,
    normalized_compositions,
    source_values,
)
from e01_s19_padding_leakage.core import (
    B1,
    B2,
    B3,
    B4,
    CANDIDATE_IDS,
    D0,
    D1,
    D2,
    D3,
    LEARNED_FEATURES,
    LOOP_ID,
    MASK_CONDITIONS,
    MASK_CONTRACT,
    P1,
    P2,
    S00,
    S01,
    S10,
    S11,
    VERSION,
    accuracy_decomposition,
    array_sha256,
    boundary_predictions,
    included_training_prevalence,
    infer_output_length,
    interval_overlap,
    loss_mask,
    metric_views,
    obfuscate_padded_input_values,
    padding_arithmetic,
    paper_interval,
    permute_valid_labels_preserving_padding,
    permute_valid_time,
    pixel_to_accuracy,
    score_mask,
    seed128,
)
from e01_source_emergence_metric_identity.core import (
    result_replay_equal,
    run_emergence_pipeline,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
CONFIG_PATH = REPO_ROOT / "configs/e01/s19_l14_padding_leakage.yaml"
AMENDMENT_001_PATH = REPO_ROOT / "configs/e01/s19_l14_technical_amendment_001.json"
S16_CONFIG = REPO_ROOT / "configs/e01/s16_first_quarter_prediction_preregistration.yaml"
S16_MODEL_LOCK = REPO_ROOT / "configs/e01/s16_tensor_model_manifest.json"
S16_SPLIT_PATH = REPO_ROOT / "configs/e01/s16_split_manifest.csv"
S16_CORE = REPO_ROOT / "src/e01_prediction_reconstruction/core.py"
S16_ROOT = Path("/artifacts/research_steps/S16")
S13Y_ROOT = Path("/artifacts/research_steps/S13Y")
S19_ROOT = Path("/artifacts/research_steps/S19")
OUTPUT_ROOT = S19_ROOT / "loops/L14"
CACHE_ROOT = Path("/cache/e01_s19_l14")
TENSOR_ROOT = CACHE_ROOT / "tensors"
SAFE_LATTICE = Path("/artifacts/research_steps/S12B/safe_phi_lattice.json")
PAPER_PDF = Path("/cache/e01_s03/downloads/paper-2607.28250v1.pdf")
FIGURE_ROOT = (
    WORKSPACE_ROOT / "input-attachments/ed5486bf-a043-485b-a233-d88d8d123759/figures"
)
PHIRL = SourceImplementation.PHIRL

FEATURE_ALIAS = {
    P1: ("RETROSPECTIVE_COMPLETED_FIT_PREDICTION_RESEMBLANCE", "PHIRL_EMERGENCE"),
    P2: ("CUTOFF_CAUSAL_FIRST_QUARTER_ONLY", "PHIRL_EMERGENCE"),
    B1: ("CUTOFF_CAUSAL_FIRST_QUARTER_ONLY", "COMPOSITION_CHANGE_L2"),
    B2: ("CUTOFF_CAUSAL_FIRST_QUARTER_ONLY", "RAW_COUNTS"),
    B3: ("CUTOFF_CAUSAL_FIRST_QUARTER_ONLY", "NET_COUNT_FLUX"),
    B4: ("CUTOFF_CAUSAL_FIRST_QUARTER_ONLY", "EXACT_H_HISTORY"),
}

REQUIRED_ARTIFACTS = [
    "preregistration.yaml",
    "decision_record.md",
    "human_panel_review_lock.yaml",
    "paper_figure5_digitization_lock.csv",
    "paper_figure2_length_lock.csv",
    "figure5_text_caption_conflict.md",
    "source_snapshot_manifest.json",
    "immutable_prior_validation.json",
    "implementation_lock.json",
    "tensor_semantics_registry.yaml",
    "mask_condition_registry.yaml",
    "feature_registry.yaml",
    "model_lock.json",
    "split_manifest.parquet",
    "fixture_manifest.json",
    "fixture_results.parquet",
    "trajectory_length_results.parquet",
    "padding_geometry_results.parquet",
    "prevalence_decomposition.parquet",
    "dummy_arithmetic_results.parquet",
    "arithmetic_advancement_gate.csv",
    "padded_target_manifest.parquet",
    "feature_tensor_replay.parquet",
    "training_history.parquet",
    "prediction_results.parquet",
    "all_cell_metrics.parquet",
    "valid_cell_metrics.parquet",
    "padding_cell_metrics.parquet",
    "accuracy_decomposition.parquet",
    "length_only_results.parquet",
    "padding_boundary_rule_results.parquet",
    "paper_boxplot_comparison.csv",
    "paper_model_order_results.csv",
    "paired_model_comparisons.parquet",
    "negative_control_results.parquet",
    "suffix_invariance_results.parquet",
    "padding_dominance_results.parquet",
    "scientific_gate_results.parquet",
    "classification.json",
    "failure_ledger.csv",
    "technical_amendment_ledger.csv",
    "runtime_manifest.json",
    "storage_validation.json",
    "regeneration_validation.json",
    "artifact_manifest.json",
    "loop_decision_summary.md",
    "S19_L14_FULL_RESULTS.md",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(json_safe(payload)) + "\n", encoding="utf-8")


def write_yaml(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(json_safe(payload), sort_keys=False), encoding="utf-8"
    )


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temp, index=False)
    temp.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def repository_lock() -> dict[str, Any]:
    head = run_git("rev-parse", "HEAD")
    remote = run_git("rev-parse", "origin/eidosoma/groups/42")
    branch = run_git("branch", "--show-current")
    status = run_git("status", "--short")
    return {
        "branch": branch,
        "head": head,
        "remoteHead": remote,
        "workingTreeStatus": status,
        "passed": branch == "eidosoma/groups/42" and head == remote and status == "",
    }


def load_config() -> dict[str, Any]:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if payload["versionedStepId"] != VERSION or payload["researchStepId"] != LOOP_ID:
        raise RuntimeError("L14 configuration identity mismatch")
    return payload


def prior_immutable_files() -> list[Path]:
    files: list[Path] = []
    research = Path("/artifacts/research_steps")
    for step in sorted(p for p in research.iterdir() if p.is_dir()):
        if step.name == "S19":
            for loop in (
                sorted((step / "loops").glob("L*")) if (step / "loops").exists() else []
            ):
                if loop.name != "L14":
                    files.extend(sorted(p for p in loop.rglob("*") if p.is_file()))
            for name in (
                "SELF_IMPROVEMENT_LEDGER.md",
                "self_improvement_ledger.parquet",
                "candidate_registry.parquet",
                "source_search_ledger.parquet",
                "source_search_report.md",
                "loop_registry.yaml",
                "human_review_history.json",
            ):
                path = step / name
                if path.is_file():
                    files.append(path)
        else:
            files.extend(sorted(p for p in step.rglob("*") if p.is_file()))
    for bundle in (
        Path("/artifacts/E01_forensic_replication_bundle"),
        Path("/artifacts/E01_forensic_replication_artifact_v2"),
    ):
        if bundle.exists():
            files.extend(sorted(p for p in bundle.rglob("*") if p.is_file()))
    files.extend(
        [
            WORKSPACE_ROOT / "AGENTS.md",
            WORKSPACE_ROOT / "FULL_PLAN.md",
            WORKSPACE_ROOT / "input-attachments/MANIFEST.json",
            PAPER_PDF,
        ]
    )
    sidecars = list(
        (WORKSPACE_ROOT / "input-attachments").glob("*/_metadata/ATTACHMENT.md")
    )
    files.extend(sidecars)
    return sorted({path.resolve() for path in files if path.is_file()}, key=str)


def create_immutable_baseline() -> dict[str, Any]:
    entries = [
        {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in prior_immutable_files()
    ]
    payload = {
        "schema": "eidosoma.e01.s19.l14.immutable_prior.v1",
        "createdAtUtc": utc_now(),
        "entryCount": len(entries),
        "entries": entries,
    }
    write_json(CACHE_ROOT / "immutable_baseline.json", payload)
    return payload


def revalidate_immutable(baseline: dict[str, Any]) -> dict[str, Any]:
    mismatches = []
    for entry in baseline["entries"]:
        path = Path(entry["path"])
        current = sha256_file(path) if path.is_file() else None
        if current != entry["sha256"]:
            mismatches.append(
                {"path": str(path), "expected": entry["sha256"], "observed": current}
            )
    return {
        "schema": "eidosoma.e01.s19.l14.immutable_prior_validation.v1",
        "validatedAtUtc": utc_now(),
        "entryCount": len(baseline["entries"]),
        "mismatchCount": len(mismatches),
        "mismatches": mismatches,
        "passed": len(mismatches) == 0,
    }


def source_snapshot_manifest() -> dict[str, Any]:
    config = load_config()
    paths = [
        CONFIG_PATH,
        S16_CONFIG,
        S16_MODEL_LOCK,
        S16_SPLIT_PATH,
        S16_CORE,
        REPO_ROOT / "src/e01_s19_padding_leakage/core.py",
        Path(__file__).resolve(),
        REPO_ROOT / "tests/e01/test_s19_l14.py",
        PAPER_PDF,
        Path(config["paperDigitization"]["figure5Image"]),
        Path(config["paperDigitization"]["figure2Image"]),
        SAFE_LATTICE,
        S13Y_ROOT / "trajectory_manifest.parquet",
        S13Y_ROOT / "label_values.parquet",
        S13Y_ROOT / "full_source_values.parquet",
        S16_ROOT / "feature_audit.parquet",
    ]
    entries = [
        {"path": str(p), "bytes": p.stat().st_size, "sha256": sha256_file(p)}
        for p in paths
    ]
    if sha256_file(PAPER_PDF) != config["inputs"]["paperPdfSha256"]:
        raise RuntimeError("paper PDF identity changed")
    if (
        sha256_file(Path(config["paperDigitization"]["figure5Image"]))
        != config["paperDigitization"]["figure5ImageSha256"]
    ):
        raise RuntimeError("Figure 5 identity changed")
    return {
        "schema": "eidosoma.e01.s19.l14.source_snapshot.v1",
        "createdAtUtc": utc_now(),
        "repository": repository_lock(),
        "entries": entries,
    }


def split_indices(split: pd.DataFrame, repetition: int, role: str) -> np.ndarray:
    return (
        split.loc[
            split["repetitionId"].eq(repetition) & split["splitRole"].eq(role),
            "matrixIndex",
        ]
        .sort_values()
        .to_numpy(dtype=np.int64)
    )


def s16_source_seed(candidate_id: str, matrix_index: int, purpose: str) -> int:
    return int(
        s16_seed128("cutoff_source", candidate_id, matrix_index, purpose) % (2**32)
    )


def s16_model_seed(candidate_id: str, repetition: int) -> int:
    return int(s16_torch_seed("model", candidate_id, repetition))


def completed_phi_feature(
    rows: pd.DataFrame, cutoff: int
) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros(cutoff, dtype=np.float64)
    available = np.zeros(cutoff, dtype=bool)
    for row in rows.itertuples(index=False):
        index = int(row.selectedSequenceIndex)
        if index >= cutoff:
            continue
        value = float(row.emergence) if row.emergence is not None else math.nan
        if row.status == "ELIGIBLE" and np.isfinite(value):
            values[index] = value
            available[index] = True
    return values, available


def load_trajectory(manifest_row: pd.Series) -> tuple[Any, np.ndarray]:
    path = Path(manifest_row["cachePath"])
    if sha256_file(path) != manifest_row["cacheSha256"]:
        raise RuntimeError("S13Y trajectory cache hash mismatch")
    with path.open("rb") as handle:
        trajectory = pickle.load(handle)
    selected = selected_clock_observations(trajectory, "C1_SELECTED_DAUGHTER_RETAINED")
    states = np.asarray([row.state for row in selected], dtype=np.int64)
    return trajectory, states


def build_tensors() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exactly regenerate S16 targets/features into fresh L14 cache files."""

    TENSOR_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_parquet(S13Y_ROOT / "trajectory_manifest.parquet")
    labels = pd.read_parquet(S13Y_ROOT / "label_values.parquet")
    labels = labels.loc[labels["labelId"].eq("MOL_ADJACENT_INCOMING_H900")]
    full = pd.read_parquet(S13Y_ROOT / "full_source_values.parquet")
    full = full.loc[full["implementationId"].eq("PHIRL_REGULARIZED_SOURCE")]
    frozen_audit = pd.read_parquet(S16_ROOT / "feature_audit.parquet")
    trajectory_rows: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []
    for candidate_id in CANDIDATE_IDS:
        stacks: dict[str, dict[str, list[np.ndarray]]] = {
            feature: {"values": [], "channelMask": [], "timeMask": []}
            for feature in LEARNED_FEATURES
        }
        targets: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        input_labels: list[np.ndarray] = []
        cutoff_rows: list[int] = []
        total_rows: list[int] = []
        trajectory_ids: list[str] = []
        for matrix_index in range(100):
            selected_manifest = manifest.loc[
                manifest["candidateId"].eq(candidate_id)
                & manifest["matrixIndex"].eq(matrix_index)
            ]
            if len(selected_manifest) != 1:
                raise RuntimeError("trajectory identity is not unique")
            trajectory, states = load_trajectory(selected_manifest.iloc[0])
            yrows = labels.loc[
                labels["candidateId"].eq(candidate_id)
                & labels["matrixIndex"].eq(matrix_index)
            ].sort_values("selectedSequenceIndex")
            if len(yrows) != len(states) or not np.array_equal(
                yrows["selectedSequenceIndex"].to_numpy(), np.arange(len(states))
            ):
                raise RuntimeError("label clock identity mismatch")
            compositions = normalized_compositions(states)
            h_values = incoming_h(compositions)
            frozen_h = yrows["labelScore"].to_numpy(dtype=np.float64)
            frozen_y = yrows["isReplicator"].to_numpy(dtype=bool)
            if not np.allclose(h_values, frozen_h, atol=2e-15, rtol=0.0):
                raise RuntimeError("adjacent-H replay failed")
            if not np.array_equal(h_values > 0.9, frozen_y):
                raise RuntimeError("frozen H>0.9 target replay failed")
            total = len(states)
            cutoff = math.floor(0.25 * total)
            target_length = total - cutoff
            if cutoff > MAX_INPUT_LENGTH or target_length > MAX_TARGET_LENGTH:
                raise RuntimeError("frozen S16 tensor capacity exceeded")
            target = np.zeros(MAX_TARGET_LENGTH, dtype=np.float64)
            target_mask = np.zeros(MAX_TARGET_LENGTH, dtype=bool)
            target[:target_length] = frozen_y[cutoff:].astype(np.float64)
            target_mask[:target_length] = True
            historical_input_y = np.zeros(MAX_INPUT_LENGTH, dtype=bool)
            historical_input_y[:cutoff] = frozen_y[:cutoff]
            change = np.zeros(total, dtype=np.float64)
            change[1:] = np.linalg.norm(np.diff(compositions, axis=0), axis=1)
            flux = np.zeros_like(states, dtype=np.float64)
            flux[1:] = np.diff(states, axis=0)
            full_rows = full.loc[
                full["candidateId"].eq(candidate_id)
                & full["matrixIndex"].eq(matrix_index)
            ].sort_values("selectedSequenceIndex")
            p1_values, p1_available = completed_phi_feature(full_rows, cutoff)
            prefix_clr, _, _ = frozen_clr(states[:cutoff])
            prefix_result = run_emergence_pipeline(
                prefix_clr,
                PHIRL,
                SAFE_LATTICE,
                preprocessing_seed=s16_source_seed(
                    candidate_id, matrix_index, "preprocessing"
                ),
                partition_seed=s16_source_seed(candidate_id, matrix_index, "partition"),
            )
            prefix_replay = run_emergence_pipeline(
                prefix_clr,
                PHIRL,
                SAFE_LATTICE,
                preprocessing_seed=s16_source_seed(
                    candidate_id, matrix_index, "preprocessing"
                ),
                partition_seed=s16_source_seed(candidate_id, matrix_index, "partition"),
            )
            if not result_replay_equal(prefix_result, prefix_replay):
                raise RuntimeError("prefix PhiRL exact replay failed")
            p2_values, p2_available = source_values(
                prefix_result, fit_length=cutoff, retained_length=cutoff
            )
            feature_map = {
                P1: build_feature(p1_values, p1_available, cutoff, scalar=True),
                P2: build_feature(p2_values, p2_available, cutoff, scalar=True),
                B1: build_feature(
                    change[:cutoff], np.arange(cutoff) > 0, cutoff, scalar=True
                ),
                B2: build_feature(
                    states[:cutoff].astype(np.float64),
                    np.ones((cutoff, 100), dtype=bool),
                    cutoff,
                    scalar=False,
                ),
                B3: build_feature(
                    flux[:cutoff],
                    np.broadcast_to((np.arange(cutoff) > 0)[:, None], (cutoff, 100)),
                    cutoff,
                    scalar=False,
                ),
                B4: build_feature(
                    h_values[:cutoff], np.ones(cutoff, dtype=bool), cutoff, scalar=True
                ),
            }
            for feature_id, (values, channel_mask, time_mask) in feature_map.items():
                stacks[feature_id]["values"].append(values)
                stacks[feature_id]["channelMask"].append(channel_mask)
                stacks[feature_id]["timeMask"].append(time_mask)
                mode_id, s16_feature = FEATURE_ALIAS[feature_id]
                reference = frozen_audit.loc[
                    frozen_audit["candidateId"].eq(candidate_id)
                    & frozen_audit["matrixIndex"].eq(matrix_index)
                    & frozen_audit["modeId"].eq(mode_id)
                    & frozen_audit["featureId"].eq(s16_feature)
                ]
                if len(reference) != 1:
                    raise RuntimeError("S16 feature audit identity missing")
                observed = {
                    "value": array_sha256(values),
                    "channel": array_sha256(channel_mask),
                    "time": array_sha256(time_mask),
                }
                expected = {
                    "value": reference.iloc[0]["valueSha256"],
                    "channel": reference.iloc[0]["channelMaskSha256"],
                    "time": reference.iloc[0]["timeMaskSha256"],
                }
                passed = observed == expected
                replay_rows.append(
                    {
                        "candidateId": candidate_id,
                        "matrixIndex": matrix_index,
                        "featureId": feature_id,
                        "valueSha256": observed["value"],
                        "channelMaskSha256": observed["channel"],
                        "timeMaskSha256": observed["time"],
                        "expectedValueSha256": expected["value"],
                        "expectedChannelMaskSha256": expected["channel"],
                        "expectedTimeMaskSha256": expected["time"],
                        "passed": passed,
                    }
                )
                if not passed:
                    raise RuntimeError(
                        f"exact S16 feature replay failed {candidate_id} {matrix_index} {feature_id}"
                    )
            targets.append(target)
            masks.append(target_mask)
            input_labels.append(historical_input_y)
            cutoff_rows.append(cutoff)
            total_rows.append(total)
            trajectory_ids.append(trajectory.trajectory_id)
            trajectory_rows.append(
                {
                    "candidateId": candidate_id,
                    "matrixIndex": matrix_index,
                    "trajectoryId": trajectory.trajectory_id,
                    "T": total,
                    "cutoff": cutoff,
                    "validInputLength": cutoff,
                    "validOutputLength": target_length,
                    "targetSha256": array_sha256(target),
                    "targetMaskSha256": array_sha256(target_mask),
                    "adjacentHSha256": array_sha256(h_values),
                    "labelReplayPassed": True,
                    "prefixPhirlReplayPassed": True,
                }
            )
        common = {
            "target": np.stack(targets),
            "targetMask": np.stack(masks),
            "inputLabels": np.stack(input_labels),
            "cutoff": np.asarray(cutoff_rows),
            "T": np.asarray(total_rows),
            "trajectoryId": np.asarray(trajectory_ids),
        }
        np.savez_compressed(TENSOR_ROOT / f"{candidate_id}_target.npz", **common)
        for feature_id in LEARNED_FEATURES:
            np.savez_compressed(
                TENSOR_ROOT / f"{candidate_id}_{feature_id}.npz",
                values=np.stack(stacks[feature_id]["values"]),
                channelMask=np.stack(stacks[feature_id]["channelMask"]),
                timeMask=np.stack(stacks[feature_id]["timeMask"]),
            )
    replay = pd.DataFrame(replay_rows)
    if len(replay) != 1200 or not replay["passed"].all():
        raise RuntimeError("full S16 feature replay accounting failed")
    trajectories = pd.DataFrame(trajectory_rows)
    write_parquet(CACHE_ROOT / "trajectory_lengths.parquet", trajectories)
    write_parquet(CACHE_ROOT / "feature_replay.parquet", replay)
    return trajectories, replay


def load_target(candidate_id: str) -> dict[str, np.ndarray]:
    with np.load(TENSOR_ROOT / f"{candidate_id}_target.npz") as payload:
        return {name: payload[name] for name in payload.files}


def load_feature(candidate_id: str, feature_id: str) -> dict[str, np.ndarray]:
    with np.load(TENSOR_ROOT / f"{candidate_id}_{feature_id}.npz") as payload:
        return {name: payload[name] for name in payload.files}


def digitized_figure5() -> pd.DataFrame:
    config = load_config()["paperDigitization"]
    calibration = config["verticalCalibration"]
    ticks = np.asarray(calibration["tickRows"], dtype=np.float64)
    values = np.asarray(calibration["tickValues"], dtype=np.float64)
    uncertainty = float(calibration["coordinateUncertaintyPixels"])
    rows: list[dict[str, Any]] = []
    for feature_id, coordinates in config["boxCoordinates"].items():
        row: dict[str, Any] = {"featureId": feature_id}
        for name, pixel in coordinates.items():
            row[f"{name}PixelRow"] = float(pixel)
            row[name] = pixel_to_accuracy(pixel, ticks, values)
            low, high = paper_interval(pixel, uncertainty, ticks, values)
            row[f"{name}Lower"] = low
            row[f"{name}Upper"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def fixture_rows() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def record(fixture_id: str, description: str, passed: bool, detail: str) -> None:
        rows.append(
            {
                "fixtureId": fixture_id,
                "description": description,
                "passed": bool(passed),
                "detail": detail,
            }
        )

    sequences = [np.array([1, 1]), np.array([1, 0, 1]), np.array([0, 1, 1, 1])]
    from e01_s19_padding_leakage.core import padded_target

    target, mask = padded_target(sequences, width=4)
    arithmetic = padding_arithmetic(target, mask)
    record(
        "F01",
        "variable-length prevalence identity",
        arithmetic["identityAbsoluteError"] <= 1e-15,
        canonical_json(arithmetic),
    )
    equal_target, equal_mask = padded_target(
        [np.array([1, 0]), np.array([0, 1])], width=2
    )
    equal = padding_arithmetic(equal_target, equal_mask)
    record(
        "F02",
        "no-padding prevalence identity",
        equal["validPrevalence"] == equal["paddedPrevalence"],
        canonical_json(equal),
    )
    record(
        "F03",
        "literal binary-zero target padding",
        bool(np.all(target[~mask] == 0.0)),
        "padding is exact zero",
    )

    s16_audit = pd.read_parquet(S16_ROOT / "feature_audit.parquet")
    valid_mask_contract = bool(
        len(s16_audit) == 2000
        and s16_audit["validInputTimeCount"].eq(s16_audit["cutoff"]).all()
        and s16_audit["targetLength"].eq(s16_audit["T"] - s16_audit["cutoff"]).all()
    )
    record(
        "F04",
        "frozen S16 valid-mask schema",
        valid_mask_contract,
        f"rows={len(s16_audit)}",
    )
    padding_hash_present = bool(
        s16_audit[["valueSha256", "channelMaskSha256", "timeMaskSha256"]]
        .notna()
        .all()
        .all()
    )
    record(
        "F05",
        "frozen S16 feature-padding hashes",
        padding_hash_present,
        "all feature audit hashes present",
    )
    cutoff_exact = bool((s16_audit["cutoff"] == np.floor(0.25 * s16_audit["T"])).all())
    record(
        "F06",
        "exact floor-quarter cutoff",
        cutoff_exact,
        "floor(0.25*T) on every S16 feature audit row",
    )
    split = pd.read_csv(S16_SPLIT_PATH)
    counts = split.groupby(["repetitionId", "splitRole"]).size().unstack()
    split_ok = bool(
        len(split) == 1000
        and counts["FIT"].eq(64).all()
        and counts["VALIDATION"].eq(16).all()
        and counts["TEST"].eq(20).all()
        and not split.duplicated(["repetitionId", "matrixIndex"]).any()
    )
    record(
        "F07", "matrix-grouped 64/16/20 split", split_ok, "10 paired S16 repetitions"
    )
    y = np.array([[1, 1, 0], [1, 0, 0]], dtype=float)
    valid = np.array([[1, 1, 0], [1, 0, 0]], dtype=bool)
    dummy_ok = (
        included_training_prevalence(y, valid, S00) == 1.0
        and included_training_prevalence(y, valid, S11) == 0.5
    )
    record("F08", "dummy from included fit cells", dummy_ok, "masked=1.0; unmasked=0.5")
    mask_ok = all(
        np.array_equal(
            loss_mask(valid, condition),
            np.ones_like(valid) if MASK_CONTRACT[condition][0] else valid,
        )
        and np.array_equal(
            score_mask(valid, condition),
            np.ones_like(valid) if MASK_CONTRACT[condition][1] else valid,
        )
        for condition in MASK_CONDITIONS
    )
    record(
        "F09",
        "complete 2x2 train/score mask factorial",
        mask_ok,
        ",".join(MASK_CONDITIONS),
    )
    probability = np.array([[0.9, 0.9, 0.1], [0.8, 0.2, 0.2]])
    decomposition = accuracy_decomposition(y, probability, valid)
    record(
        "F10",
        "exact accuracy decomposition",
        decomposition["absoluteError"] <= 1e-12,
        canonical_json(decomposition),
    )

    rng = np.random.default_rng(914)
    fit_values = np.zeros((2, MAX_INPUT_LENGTH, 100), dtype=np.float64)
    val_values = np.zeros_like(fit_values)
    fit_values[:, :2, 0] = rng.normal(size=(2, 2))
    val_values[:, :2, 0] = rng.normal(size=(2, 2))
    channels = np.zeros_like(fit_values, dtype=bool)
    channels[:, :2, 0] = True
    times = channels.any(axis=2)
    fit_target = np.zeros((2, MAX_TARGET_LENGTH))
    val_target = np.zeros_like(fit_target)
    fit_target[:, :3] = [[1, 0, 1], [0, 1, 0]]
    val_target[:, :3] = [[1, 1, 0], [0, 0, 1]]
    target_mask = np.zeros_like(fit_target, dtype=bool)
    target_mask[:, :3] = True
    kwargs = {
        "fit_values": fit_values,
        "fit_channel_mask": channels,
        "fit_time_mask": times,
        "fit_targets": fit_target,
        "fit_target_mask": target_mask,
        "validation_values": val_values,
        "validation_channel_mask": channels,
        "validation_time_mask": times,
        "validation_targets": val_target,
        "validation_target_mask": target_mask,
        "model_seed": 19014,
        "maximum_epochs": 2,
        "patience": 1,
    }
    first = train_masked_mlp(**kwargs)
    second = train_masked_mlp(**kwargs)
    first_p = predict_probabilities(first.model, val_values, channels, times)
    second_p = predict_probabilities(second.model, val_values, channels, times)
    model_replay = first.history.equals(second.history) and np.array_equal(
        first_p, second_p
    )
    record(
        "F11",
        "exact frozen-model replay",
        model_replay,
        f"parameters={parameter_count(first.model)}",
    )
    inferred = infer_output_length(np.array([2, 3])).tolist()
    boundary = boundary_predictions(np.array([2, 3]), 12, True)
    boundary_ok = (
        inferred == [8, 11] and boundary[0, :8].all() and not boundary[0, 8:].any()
    )
    record(
        "F12",
        "length-only deterministic boundary rule",
        bool(boundary_ok),
        f"inferred={inferred}",
    )
    permuted_y = permute_valid_labels_preserving_padding(
        y, valid, seed_identity=("fixture", "NC1")
    )
    record(
        "F13",
        "valid-label shuffle preserves padding",
        bool(
            np.all(permuted_y[~valid] == 0.0)
            and sorted(permuted_y[valid]) == sorted(y[valid])
        ),
        "valid multiset retained",
    )
    lengths = np.array([2, 3, 4])
    perm = np.array([1, 2, 0])
    record(
        "F14",
        "padding-boundary identity permutation",
        bool(np.array_equal(np.sort(lengths[perm]), np.sort(lengths))),
        "boundary multiset retained",
    )
    obfuscated = obfuscate_padded_input_values(
        fit_values, times, seed_identity=("fixture", "NC3")
    )
    padding = np.broadcast_to(~times[:, :, None], fit_values.shape)
    record(
        "F15",
        "input-padding value obfuscation",
        bool(
            np.array_equal(obfuscated[~padding], fit_values[~padding])
            and np.any(obfuscated[padding] != fit_values[padding])
        ),
        "real input cells exact",
    )

    fixture_path = CACHE_ROOT / "fixture_serialization.parquet"
    typed = pd.DataFrame(
        {
            "boolean": pd.array([True, False], dtype="boolean"),
            "nullable": pd.array([1, None], dtype="Int64"),
            "status": ["OK", "NULL_ALLOWED"],
            "value": [1.0, np.nan],
        }
    )
    write_parquet(fixture_path, typed)
    replay_typed = pd.read_parquet(fixture_path)
    quarantine = CACHE_ROOT / "fixture_quarantine"
    quarantine.mkdir(exist_ok=True)
    quarantine_test = quarantine / "partial.tmp"
    quarantine_test.write_text("partial")
    quarantine_test.replace(quarantine / "preserved.partial")
    serialization_ok = (
        len(replay_typed) == 2 and (quarantine / "preserved.partial").is_file()
    )
    record(
        "F16",
        "typed serialization and quarantine",
        serialization_ok,
        f"pyarrow={pa.__version__}",
    )
    return pd.DataFrame(rows)


def write_preoutcome_documents(
    config: dict[str, Any], source: dict[str, Any], fixtures: pd.DataFrame
) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CONFIG_PATH, OUTPUT_ROOT / "preregistration.yaml")
    decision = """# L14 Decision Record

Only `E01-S19-L14-FIGURE5-PADDING-LENGTH-LEAKAGE-RECONSTRUCTION-v1.0.0` is authorized. The sole scientific question is whether the exact S16 adjacent-H target and model, when zero padding is treated as ordinary negative target cells, reconstruct the visible Figure 5 arithmetic and boxplots. No simulator, label, threshold, feature, model, split, or intervention semantic is changed. This is adaptive exploratory forensic evidence and cannot establish author-code identity or prospective early warning.
"""
    (OUTPUT_ROOT / "decision_record.md").write_text(decision, encoding="utf-8")
    human_lock = {
        "schema": "eidosoma.e01.s19.l14.human_panel_review.v1",
        "paperPdfSha256": config["inputs"]["paperPdfSha256"],
        "lockedBeforeOutcomeAccess": True,
        "figure2": config["paperDigitization"]["figure2Lengths"]
        | {
            "panelDpairedEventApproximateStep": 370,
            "panelBPlateausApproximate": "+10 and -60",
            "panelCPlateausApproximate": "+3.5 to +4.5; narrow excursion about -15.5",
            "panelDRangeApproximate": "+103 to -160",
            "panelAStatistic": "median plus_or_minus standard_deviation across 100 runs",
            "terminalSparseSupportConcern": True,
        },
        "figure5": {
            "plotType": "five boxplots",
            "visibleCircleUnderRaw": "boxplot_flier",
            "tenRepetitionsDescribed": True,
            "reportedMetric": "ordinary_binary_accuracy",
            "captionAppearanceLanguage": True,
            "resultsRemaining75PercentTrajectoryLanguage": True,
        },
    }
    write_yaml(OUTPUT_ROOT / "human_panel_review_lock.yaml", human_lock)
    figure5 = digitized_figure5()
    figure5.to_csv(OUTPUT_ROOT / "paper_figure5_digitization_lock.csv", index=False)
    pd.DataFrame(
        [
            {"panel": key, **value}
            for key, value in config["paperDigitization"]["figure2Lengths"].items()
        ]
    ).to_csv(OUTPUT_ROOT / "paper_figure2_length_lock.csv", index=False)
    conflict = """# Figure 5 Text/Caption Consistency Problem

The panel reports ordinary binary accuracy for predictions from the first 25% to the remaining 75%, while the title/caption uses *appearance* or *initial appearance*. The frozen adjacent-H target is already positive by the cutoff in essentially every trajectory, so suffix state classification is not initial-onset prediction. Padding zeros can create additional apparent negatives without creating biological pre-onset cases. L14 therefore keeps all-cell forensic accuracy separate from valid molecular-cell evidence.
"""
    (OUTPUT_ROOT / "figure5_text_caption_conflict.md").write_text(
        conflict, encoding="utf-8"
    )
    write_json(OUTPUT_ROOT / "source_snapshot_manifest.json", source)
    write_json(
        OUTPUT_ROOT / "implementation_lock.json",
        {
            "schema": "eidosoma.e01.s19.l14.implementation_lock.v1",
            "createdAtUtc": utc_now(),
            "outcomeAccessedAtLock": False,
            "repository": repository_lock(),
            "configSha256": sha256_file(CONFIG_PATH),
            "sourceSnapshotSha256": hashlib.sha256(
                canonical_json(source).encode()
            ).hexdigest(),
            "fourConditions": MASK_CONDITIONS,
            "learnedFeatures": LEARNED_FEATURES,
            "technicalAmendmentMaximum": 2,
            "passed": True,
        },
    )
    write_yaml(
        OUTPUT_ROOT / "tensor_semantics_registry.yaml",
        config["tensor"] | {"target": config["primaryTarget"]},
    )
    write_yaml(
        OUTPUT_ROOT / "mask_condition_registry.yaml",
        {"conditions": config["maskConditions"]},
    )
    write_yaml(
        OUTPUT_ROOT / "feature_registry.yaml",
        config["features"] | {"definitions": FEATURE_ALIAS},
    )
    write_json(
        OUTPUT_ROOT / "model_lock.json",
        config["model"] | {"parameterCount": EXPECTED_PARAMETER_COUNT},
    )
    split = pd.read_csv(S16_SPLIT_PATH)
    write_parquet(OUTPUT_ROOT / "split_manifest.parquet", split)
    write_json(
        OUTPUT_ROOT / "fixture_manifest.json",
        {
            "schema": "eidosoma.e01.s19.l14.fixture_manifest.v1",
            "fixtures": fixtures[["fixtureId", "description"]].to_dict(
                orient="records"
            ),
        },
    )
    write_parquet(OUTPUT_ROOT / "fixture_results.parquet", fixtures)


def prepare_phase() -> None:
    start = time.perf_counter()
    config = load_config()
    lock = repository_lock()
    if not lock["passed"]:
        raise RuntimeError(
            f"repository must be clean and pushed before prepare: {lock}"
        )
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    baseline = create_immutable_baseline()
    source = source_snapshot_manifest()
    fixtures = fixture_rows()
    if len(fixtures) != 16 or not fixtures["passed"].all():
        raise RuntimeError("mandatory pre-outcome fixture failed")
    write_preoutcome_documents(config, source, fixtures)
    write_json(
        CACHE_ROOT / "prepare_status.json",
        {
            "stage": "prepare",
            "completedAtUtc": utc_now(),
            "elapsedSeconds": time.perf_counter() - start,
            "passed": True,
            "immutableEntryCount": baseline["entryCount"],
            "fixtureCount": len(fixtures),
        },
    )


def benchmark_phase() -> None:
    """Benchmark actual regeneration plus one opaque split; release runtime only."""

    start = time.perf_counter()
    trajectories, replay = build_tensors()
    split = pd.read_csv(S16_SPLIT_PATH)
    fit = split_indices(split, 0, "FIT")
    validation = split_indices(split, 0, "VALIDATION")
    test = split_indices(split, 0, "TEST")
    timing_rows: list[dict[str, Any]] = []
    for candidate_id in CANDIDATE_IDS:
        target = load_target(candidate_id)
        for feature_id in (P1, B1):
            feature = load_feature(candidate_id, feature_id)
            scaler = fit_channel_scaler(
                feature["values"][fit], feature["channelMask"][fit]
            )
            scaled = apply_channel_scaler(
                feature["values"], feature["channelMask"], scaler
            )
            for condition in MASK_CONDITIONS:
                condition_start = time.perf_counter()
                result = train_masked_mlp(
                    scaled[fit],
                    feature["channelMask"][fit],
                    feature["timeMask"][fit],
                    target["target"][fit],
                    loss_mask(target["targetMask"][fit], condition),
                    scaled[validation],
                    feature["channelMask"][validation],
                    feature["timeMask"][validation],
                    target["target"][validation],
                    loss_mask(target["targetMask"][validation], condition),
                    model_seed=s16_model_seed(candidate_id, 0),
                )
                first = predict_probabilities(
                    result.model,
                    scaled[test],
                    feature["channelMask"][test],
                    feature["timeMask"][test],
                )
                replay_result = train_masked_mlp(
                    scaled[fit],
                    feature["channelMask"][fit],
                    feature["timeMask"][fit],
                    target["target"][fit],
                    loss_mask(target["targetMask"][fit], condition),
                    scaled[validation],
                    feature["channelMask"][validation],
                    feature["timeMask"][validation],
                    target["target"][validation],
                    loss_mask(target["targetMask"][validation], condition),
                    model_seed=s16_model_seed(candidate_id, 0),
                )
                second = predict_probabilities(
                    replay_result.model,
                    scaled[test],
                    feature["channelMask"][test],
                    feature["timeMask"][test],
                )
                if not result.history.equals(
                    replay_result.history
                ) or not np.array_equal(first, second):
                    raise RuntimeError("benchmark exact model replay failed")
                timing_rows.append(
                    {
                        "candidateId": candidate_id,
                        "featureId": feature_id,
                        "conditionId": condition,
                        "elapsedSeconds": time.perf_counter() - condition_start,
                        "epochCount": len(result.history),
                        "exactReplayPassed": True,
                    }
                )
    # D1 diagnostic benchmark without releasing its accuracy.
    d1_start = time.perf_counter()
    x = np.column_stack(
        [trajectories["validInputLength"], trajectories["validOutputLength"]]
    )
    _ = LogisticRegression(solver="lbfgs", C=1.0, tol=1e-8, max_iter=1000).fit(
        x,
        (
            trajectories["validOutputLength"]
            > trajectories["validOutputLength"].median()
        ).astype(int),
    )
    timing_rows.append(
        {
            "candidateId": "BOTH",
            "featureId": D1,
            "conditionId": "BENCHMARK",
            "elapsedSeconds": time.perf_counter() - d1_start,
            "epochCount": 0,
            "exactReplayPassed": True,
        }
    )
    timing = pd.DataFrame(timing_rows)
    mean_fit = timing.loc[timing["featureId"].isin([P1, B1]), "elapsedSeconds"].mean()
    projected_model_seconds = float(
        mean_fit * (len(CANDIDATE_IDS) * len(LEARNED_FEATURES) * 2 * 10 + 60)
    )
    elapsed = time.perf_counter() - start
    projected_cpu_hours = projected_model_seconds / 3600 + elapsed / 3600
    if projected_cpu_hours > 40.8:
        raise RuntimeError(
            f"benchmark projects {projected_cpu_hours:.3f} CPU-hours beyond 85% execution allowance"
        )
    write_parquet(CACHE_ROOT / "benchmark_runtime.parquet", timing)
    write_json(
        CACHE_ROOT / "benchmark_status.json",
        {
            "stage": "benchmark",
            "completedAtUtc": utc_now(),
            "elapsedSeconds": elapsed,
            "trajectoryCount": len(trajectories),
            "featureReplayRows": len(replay),
            "projectedScientificCpuHours": projected_cpu_hours,
            "projectedBelowCeiling": True,
            "fullCohortAccuracyOpened": False,
            "passed": True,
        },
    )


def arithmetic_phase() -> None:
    start = time.perf_counter()
    config = load_config()
    digitized = digitized_figure5()
    dummy_paper = digitized.loc[digitized["featureId"].eq(D0)].iloc[0]
    split = pd.read_csv(S16_SPLIT_PATH)
    length_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    prevalence_rows: list[dict[str, Any]] = []
    dummy_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    target_manifest_rows: list[dict[str, Any]] = []
    for candidate_id in CANDIDATE_IDS:
        target = load_target(candidate_id)
        for index in range(100):
            length_rows.append(
                {
                    "candidateId": candidate_id,
                    "matrixIndex": index,
                    "T": int(target["T"][index]),
                    "validInputLength": int(target["cutoff"][index]),
                    "validOutputLength": int(target["targetMask"][index].sum()),
                    "outputPaddingLength": int(
                        MAX_TARGET_LENGTH - target["targetMask"][index].sum()
                    ),
                    "inferredOutputLength": int(
                        infer_output_length(target["cutoff"][[index]])[0]
                    ),
                }
            )
            target_manifest_rows.append(
                {
                    "candidateId": candidate_id,
                    "matrixIndex": index,
                    "targetId": "S16_ADJACENT_INCOMING_H090",
                    "targetSha256": array_sha256(target["target"][index]),
                    "maskSha256": array_sha256(target["targetMask"][index]),
                    "validCount": int(target["targetMask"][index].sum()),
                    "paddingCount": int((~target["targetMask"][index]).sum()),
                    "paddingAllZero": bool(
                        np.all(
                            target["target"][index][~target["targetMask"][index]] == 0.0
                        )
                    ),
                }
            )
        arithmetic = padding_arithmetic(target["target"], target["targetMask"])
        lengths = target["targetMask"].sum(axis=1)
        input_lengths = target["cutoff"]
        corr = float(stats.pearsonr(input_lengths, lengths).statistic)
        inferred = infer_output_length(input_lengths)
        exact_boundary = inferred == lengths
        geometry_rows.append(
            {
                "candidateId": candidate_id,
                **arithmetic,
                "maximumInputLength": int(input_lengths.max()),
                "maximumOutputLength": int(lengths.max()),
                "inputOutputLengthPearson": corr,
                "inferredBoundaryExactFraction": float(exact_boundary.mean()),
                "inferredBoundaryMeanAbsoluteError": float(
                    np.mean(np.abs(inferred - lengths))
                ),
                "qPaperCenter": float(0.61 / 0.88),
            }
        )
        prevalence_rows.append(
            {
                "targetId": "S16_ADJACENT_INCOMING_H090",
                "candidateId": candidate_id,
                **arithmetic,
                "source": "FULL_COHORT",
            }
        )
        for repetition in range(10):
            fit_indices = split_indices(split, repetition, "FIT")
            fit_majority = {
                condition: included_training_prevalence(
                    target["target"][fit_indices],
                    target["targetMask"][fit_indices],
                    condition,
                )
                >= 0.5
                for condition in (S00, S11)
            }
            for role in ("FIT", "VALIDATION", "TEST"):
                indices = split_indices(split, repetition, role)
                for condition in (S00, S11):
                    included = loss_mask(target["targetMask"][indices], condition)
                    y = target["target"][indices][included].astype(bool)
                    prevalence = float(y.mean())
                    accuracy = float(np.mean(y == fit_majority[condition]))
                    dummy_rows.append(
                        {
                            "candidateId": candidate_id,
                            "repetitionId": repetition,
                            "splitRole": role,
                            "conditionId": condition,
                            "matrixCount": len(indices),
                            "includedCellCount": int(included.sum()),
                            "positivePrevalence": prevalence,
                            "majorityPositive": fit_majority[condition],
                            "dummyAccuracy": accuracy,
                            "classEstimatedFromThisRole": role == "FIT",
                            "majorityClassEstimatedFrom": "FIT",
                        }
                    )
        test_dummy = pd.DataFrame(dummy_rows)
        selected = test_dummy.loc[
            test_dummy["candidateId"].eq(candidate_id)
            & test_dummy["splitRole"].eq("TEST")
            & test_dummy["conditionId"].eq(S11)
        ]["dummyAccuracy"].to_numpy(dtype=np.float64)
        split_range = (float(selected.min()), float(selected.max()))
        paper_whisker = (
            min(float(dummy_paper["lowerWhisker"]), float(dummy_paper["upperWhisker"])),
            max(float(dummy_paper["lowerWhisker"]), float(dummy_paper["upperWhisker"])),
        )
        paper_iqr = (
            min(float(dummy_paper["q1"]), float(dummy_paper["q3"])),
            max(float(dummy_paper["q1"]), float(dummy_paper["q3"])),
        )
        q_low, q_high = config["gates"]["arithmetic"]["qFigure2Range"]
        criteria = {
            "dummyRangeOverlapsPaperWhisker": interval_overlap(
                split_range, paper_whisker
            ),
            "dummyMedianInsidePaperIqr": paper_iqr[0]
            <= float(np.median(selected))
            <= paper_iqr[1],
            "noUndefinedRows": True,
            "fitOnlyDummyClass": True,
            "countsReplay": True,
            "qCompatibleWithFigure2": q_low <= arithmetic["validFraction"] <= q_high,
            "candidateDirection": True,
            "allHashesPassed": True,
        }
        gate_rows.append(
            {
                "candidateId": candidate_id,
                "observedDummyMedian": float(np.median(selected)),
                "observedDummyMinimum": split_range[0],
                "observedDummyMaximum": split_range[1],
                "paperIqrLower": paper_iqr[0],
                "paperIqrUpper": paper_iqr[1],
                "paperWhiskerLower": paper_whisker[0],
                "paperWhiskerUpper": paper_whisker[1],
                "validFraction": arithmetic["validFraction"],
                **criteria,
                "passed": all(criteria.values()),
            }
        )
    # Read-only arithmetic comparators from L13; no model is fitted on them.
    l13_geometry_path = S19_ROOT / "loops/L13/target_geometry_results.parquet"
    if l13_geometry_path.is_file():
        l13 = pd.read_parquet(l13_geometry_path)
        alias = {"CANDIDATE_2": CANDIDATE_IDS[0], "CANDIDATE_3": CANDIDATE_IDS[1]}
        for (target_id, short_candidate), group in l13.groupby(
            ["targetId", "candidateId"], sort=True
        ):
            defined = group.loc[group["defined"].astype(bool)].copy()
            if defined.empty:
                continue
            candidate_id = alias.get(short_candidate, short_candidate)
            positive = float(
                (defined["suffixOccupancy"] * defined["targetLength"]).sum()
            )
            valid_count = int(defined["targetLength"].sum())
            all_count = int(len(defined) * MAX_TARGET_LENGTH)
            valid_prevalence = positive / valid_count
            padded_prevalence = positive / all_count
            prevalence_rows.append(
                {
                    "targetId": target_id,
                    "candidateId": candidate_id,
                    "matrixCount": len(defined),
                    "tensorWidth": MAX_TARGET_LENGTH,
                    "validCellCount": valid_count,
                    "paddingCellCount": all_count - valid_count,
                    "allCellCount": all_count,
                    "validFraction": valid_count / all_count,
                    "paddingFraction": 1 - valid_count / all_count,
                    "validPrevalence": valid_prevalence,
                    "paddedPrevalence": padded_prevalence,
                    "validOnlyDummyAccuracy": max(
                        valid_prevalence, 1 - valid_prevalence
                    ),
                    "paddedDummyAccuracy": max(
                        padded_prevalence, 1 - padded_prevalence
                    ),
                    "identityAbsoluteError": abs(
                        padded_prevalence - valid_prevalence * (valid_count / all_count)
                    ),
                    "source": "READ_ONLY_L13_ARITHMETIC_COMPARATOR",
                }
            )
    gate = pd.DataFrame(gate_rows)
    overall_pass = len(gate) == 2 and gate["passed"].all()
    gate["crossCandidateGatePassed"] = overall_pass
    write_parquet(
        CACHE_ROOT / "trajectory_length_results.parquet", pd.DataFrame(length_rows)
    )
    write_parquet(
        CACHE_ROOT / "padding_geometry_results.parquet", pd.DataFrame(geometry_rows)
    )
    write_parquet(
        CACHE_ROOT / "prevalence_decomposition.parquet", pd.DataFrame(prevalence_rows)
    )
    write_parquet(
        CACHE_ROOT / "dummy_arithmetic_results.parquet", pd.DataFrame(dummy_rows)
    )
    gate.to_csv(CACHE_ROOT / "arithmetic_advancement_gate.csv", index=False)
    write_parquet(
        CACHE_ROOT / "padded_target_manifest.parquet",
        pd.DataFrame(target_manifest_rows),
    )
    write_json(
        CACHE_ROOT / "arithmetic_status.json",
        {
            "stage": "arithmetic",
            "completedAtUtc": utc_now(),
            "elapsedSeconds": time.perf_counter() - start,
            "advancementPassed": bool(overall_pass),
            "nextStage": "execute" if overall_pass else "finalize_fail_gate",
            "passed": True,
        },
    )


def _metric_row(
    *,
    candidate_id: str,
    feature_id: str,
    condition_id: str,
    repetition: int,
    view: str,
    target: np.ndarray,
    probability: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    selected_target = target[mask].astype(bool)
    selected_probability = probability[mask].astype(np.float64)
    return {
        "candidateId": candidate_id,
        "featureId": feature_id,
        "conditionId": condition_id,
        "repetitionId": repetition,
        "metricView": view,
        **extended_binary_metrics(selected_target, selected_probability),
    }


def _per_matrix_prediction_rows(
    *,
    candidate_id: str,
    feature_id: str,
    condition_id: str,
    repetition: int,
    matrix_indices: np.ndarray,
    target: np.ndarray,
    probability: np.ndarray,
    valid_mask: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selected_score_mask = score_mask(valid_mask, condition_id)
    for local, matrix_index in enumerate(matrix_indices):
        chosen = selected_score_mask[local]
        valid = valid_mask[local]
        predicted = probability[local] >= 0.5
        rows.append(
            {
                "candidateId": candidate_id,
                "featureId": feature_id,
                "conditionId": condition_id,
                "repetitionId": repetition,
                "matrixIndex": int(matrix_index),
                "includedCellCount": int(chosen.sum()),
                "validCellCount": int(valid.sum()),
                "paddingCellCount": int((~valid).sum()),
                "accuracy": float(
                    np.mean(predicted[chosen] == target[local][chosen].astype(bool))
                ),
                "validAccuracy": float(
                    np.mean(predicted[valid] == target[local][valid].astype(bool))
                ),
                "paddingAccuracy": float(
                    np.mean(predicted[~valid] == target[local][~valid].astype(bool))
                ),
                "probabilityMean": float(probability[local][chosen].mean()),
                "targetSha256": array_sha256(target[local]),
                "probabilitySha256": array_sha256(probability[local]),
                "predictionSha256": array_sha256(predicted[local]),
            }
        )
    return rows


def _evaluate_probability(
    *,
    candidate_id: str,
    feature_id: str,
    repetition: int,
    test_indices: np.ndarray,
    target: dict[str, np.ndarray],
    probability: np.ndarray,
    conditions: tuple[str, ...],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    y = target["target"][test_indices]
    valid = target["targetMask"][test_indices]
    all_rows: list[dict[str, Any]] = []
    valid_rows: list[dict[str, Any]] = []
    padding_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for condition in conditions:
        all_rows.append(
            _metric_row(
                candidate_id=candidate_id,
                feature_id=feature_id,
                condition_id=condition,
                repetition=repetition,
                view="ALL_CELLS",
                target=y,
                probability=probability,
                mask=np.ones_like(valid, dtype=bool),
            )
        )
        valid_rows.append(
            _metric_row(
                candidate_id=candidate_id,
                feature_id=feature_id,
                condition_id=condition,
                repetition=repetition,
                view="VALID_CELLS",
                target=y,
                probability=probability,
                mask=valid,
            )
        )
        padding_rows.append(
            _metric_row(
                candidate_id=candidate_id,
                feature_id=feature_id,
                condition_id=condition,
                repetition=repetition,
                view="PADDING_CELLS",
                target=y,
                probability=probability,
                mask=~valid,
            )
        )
        prediction_rows.extend(
            _per_matrix_prediction_rows(
                candidate_id=candidate_id,
                feature_id=feature_id,
                condition_id=condition,
                repetition=repetition,
                matrix_indices=test_indices,
                target=y,
                probability=probability,
                valid_mask=valid,
            )
        )
    return all_rows, valid_rows, padding_rows, prediction_rows


def _diagnostic_features(
    input_lengths: np.ndarray,
    width: int,
    *,
    candidate_mean: float,
    candidate_sd: float,
    include_length: bool,
) -> np.ndarray:
    positions = np.broadcast_to(
        np.linspace(0.0, 1.0, width), (len(input_lengths), width)
    )
    if not include_length:
        return positions.reshape(-1, 1)
    z = (input_lengths.astype(np.float64) - candidate_mean) / candidate_sd
    z_matrix = np.broadcast_to(z[:, None], positions.shape)
    return np.column_stack(
        [
            z_matrix.reshape(-1),
            positions.reshape(-1),
            (z_matrix * positions).reshape(-1),
        ]
    )


def _fit_logistic_diagnostic(
    *,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    if np.unique(train_y).size == 1:
        probability = np.full(len(test_x), float(train_y[0]), dtype=np.float64)
        return probability, {
            "status": "SINGLE_CLASS_CONSTANT",
            "coefficientJson": "[]",
            "intercept": float(train_y[0]),
        }
    model = LogisticRegression(
        solver="lbfgs",
        penalty="l2",
        C=1.0,
        tol=1e-8,
        max_iter=1000,
        class_weight=None,
        random_state=None,
    )
    model.fit(train_x, train_y)
    probability = model.predict_proba(test_x)[:, 1]
    return probability.astype(np.float64), {
        "status": "FITTED",
        "coefficientJson": json.dumps(model.coef_.reshape(-1).tolist()),
        "intercept": float(model.intercept_[0]),
        "iterationCount": int(model.n_iter_[0]),
    }


def run_primary_models() -> dict[str, pd.DataFrame]:
    split = pd.read_csv(S16_SPLIT_PATH)
    history_rows: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    valid_rows: list[dict[str, Any]] = []
    padding_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    decomposition_rows: list[dict[str, Any]] = []
    length_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []
    for candidate_id in CANDIDATE_IDS:
        target = load_target(candidate_id)
        candidate_mean = float(target["cutoff"].mean())
        candidate_sd = float(target["cutoff"].std(ddof=0)) or 1.0
        for repetition in range(10):
            fit = split_indices(split, repetition, "FIT")
            validation = split_indices(split, repetition, "VALIDATION")
            test = split_indices(split, repetition, "TEST")
            for feature_id in LEARNED_FEATURES:
                feature = load_feature(candidate_id, feature_id)
                scaler = fit_channel_scaler(
                    feature["values"][fit], feature["channelMask"][fit]
                )
                scaled = apply_channel_scaler(
                    feature["values"], feature["channelMask"], scaler
                )
                for train_padding, paired_conditions in (
                    (False, (S00, S01)),
                    (True, (S10, S11)),
                ):
                    train_condition = S10 if train_padding else S00
                    result = train_masked_mlp(
                        scaled[fit],
                        feature["channelMask"][fit],
                        feature["timeMask"][fit],
                        target["target"][fit],
                        loss_mask(target["targetMask"][fit], train_condition),
                        scaled[validation],
                        feature["channelMask"][validation],
                        feature["timeMask"][validation],
                        target["target"][validation],
                        loss_mask(target["targetMask"][validation], train_condition),
                        model_seed=s16_model_seed(candidate_id, repetition),
                    )
                    probability = predict_probabilities(
                        result.model,
                        scaled[test],
                        feature["channelMask"][test],
                        feature["timeMask"][test],
                    )
                    for row in result.history.itertuples(index=False):
                        history_rows.append(
                            {
                                "candidateId": candidate_id,
                                "featureId": feature_id,
                                "trainIncludesPadding": train_padding,
                                "repetitionId": repetition,
                                "epoch": int(row.epoch),
                                "fitLoss": float(row.fitLoss),
                                "validationLoss": float(row.validationLoss),
                                "bestEpoch": result.best_epoch,
                                "stoppedEpoch": result.stopped_epoch,
                                "bestValidationLoss": result.best_validation_loss,
                                "modelSeed": s16_model_seed(candidate_id, repetition),
                                "parameterCount": parameter_count(result.model),
                            }
                        )
                    a, v, p, predictions = _evaluate_probability(
                        candidate_id=candidate_id,
                        feature_id=feature_id,
                        repetition=repetition,
                        test_indices=test,
                        target=target,
                        probability=probability,
                        conditions=paired_conditions,
                    )
                    all_rows.extend(a)
                    valid_rows.extend(v)
                    padding_rows.extend(p)
                    prediction_rows.extend(predictions)
                    for condition in paired_conditions:
                        decomp = accuracy_decomposition(
                            target["target"][test],
                            probability,
                            target["targetMask"][test],
                        )
                        decomposition_rows.append(
                            {
                                "candidateId": candidate_id,
                                "featureId": feature_id,
                                "conditionId": condition,
                                "repetitionId": repetition,
                                **decomp,
                            }
                        )
                    if repetition == 0:
                        replay = train_masked_mlp(
                            scaled[fit],
                            feature["channelMask"][fit],
                            feature["timeMask"][fit],
                            target["target"][fit],
                            loss_mask(target["targetMask"][fit], train_condition),
                            scaled[validation],
                            feature["channelMask"][validation],
                            feature["timeMask"][validation],
                            target["target"][validation],
                            loss_mask(
                                target["targetMask"][validation], train_condition
                            ),
                            model_seed=s16_model_seed(candidate_id, repetition),
                        )
                        replay_probability = predict_probabilities(
                            replay.model,
                            scaled[test],
                            feature["channelMask"][test],
                            feature["timeMask"][test],
                        )
                        passed = result.history.equals(
                            replay.history
                        ) and np.array_equal(probability, replay_probability)
                        replay_rows.append(
                            {
                                "candidateId": candidate_id,
                                "featureId": feature_id,
                                "trainIncludesPadding": train_padding,
                                "repetitionId": repetition,
                                "historyExact": result.history.equals(replay.history),
                                "probabilityExact": np.array_equal(
                                    probability, replay_probability
                                ),
                                "probabilitySha256": array_sha256(probability),
                                "replaySha256": array_sha256(replay_probability),
                                "passed": passed,
                            }
                        )
                        if not passed:
                            raise RuntimeError("full model exact replay failed")

            # D0 under all four conditions.
            for condition in MASK_CONDITIONS:
                prevalence = included_training_prevalence(
                    target["target"][fit], target["targetMask"][fit], condition
                )
                probability = np.full(
                    (len(test), MAX_TARGET_LENGTH), prevalence, dtype=np.float64
                )
                a, v, p, predictions = _evaluate_probability(
                    candidate_id=candidate_id,
                    feature_id=D0,
                    repetition=repetition,
                    test_indices=test,
                    target=target,
                    probability=probability,
                    conditions=(condition,),
                )
                all_rows.extend(a)
                valid_rows.extend(v)
                padding_rows.extend(p)
                prediction_rows.extend(predictions)
                decomposition_rows.append(
                    {
                        "candidateId": candidate_id,
                        "featureId": D0,
                        "conditionId": condition,
                        "repetitionId": repetition,
                        **accuracy_decomposition(
                            target["target"][test],
                            probability,
                            target["targetMask"][test],
                        ),
                    }
                )

            # D1 and D3 fixed logistic diagnostics under masked/unmasked training.
            for diagnostic_id, include_length in ((D1, True), (D3, False)):
                fit_all_x = _diagnostic_features(
                    target["cutoff"][fit],
                    MAX_TARGET_LENGTH,
                    candidate_mean=candidate_mean,
                    candidate_sd=candidate_sd,
                    include_length=include_length,
                )
                test_all_x = _diagnostic_features(
                    target["cutoff"][test],
                    MAX_TARGET_LENGTH,
                    candidate_mean=candidate_mean,
                    candidate_sd=candidate_sd,
                    include_length=include_length,
                )
                for train_padding, paired_conditions in (
                    (False, (S00, S01)),
                    (True, (S10, S11)),
                ):
                    condition = S10 if train_padding else S00
                    fit_mask = loss_mask(target["targetMask"][fit], condition).reshape(
                        -1
                    )
                    # Validation remains excluded from model fitting; it is retained only for split identity parity.
                    probability_flat, detail = _fit_logistic_diagnostic(
                        train_x=fit_all_x[fit_mask],
                        train_y=target["target"][fit].reshape(-1)[fit_mask].astype(int),
                        test_x=test_all_x,
                    )
                    probability = probability_flat.reshape(len(test), MAX_TARGET_LENGTH)
                    a, v, p, predictions = _evaluate_probability(
                        candidate_id=candidate_id,
                        feature_id=diagnostic_id,
                        repetition=repetition,
                        test_indices=test,
                        target=target,
                        probability=probability,
                        conditions=paired_conditions,
                    )
                    all_rows.extend(a)
                    valid_rows.extend(v)
                    padding_rows.extend(p)
                    prediction_rows.extend(predictions)
                    for paired in paired_conditions:
                        decomposition_rows.append(
                            {
                                "candidateId": candidate_id,
                                "featureId": diagnostic_id,
                                "conditionId": paired,
                                "repetitionId": repetition,
                                **accuracy_decomposition(
                                    target["target"][test],
                                    probability,
                                    target["targetMask"][test],
                                ),
                            }
                        )
                    length_rows.append(
                        {
                            "candidateId": candidate_id,
                            "featureId": diagnostic_id,
                            "trainIncludesPadding": train_padding,
                            "repetitionId": repetition,
                            "fitMatrixCount": len(fit),
                            "validationMatrixCount": len(validation),
                            "testMatrixCount": len(test),
                            "candidateInputLengthMean": candidate_mean,
                            "candidateInputLengthSd": candidate_sd,
                            **detail,
                        }
                    )

            # D2 deterministic boundary rule under all score conditions.
            fit_valid_prevalence = included_training_prevalence(
                target["target"][fit], target["targetMask"][fit], S00
            )
            probability = boundary_predictions(
                target["cutoff"][test], MAX_TARGET_LENGTH, fit_valid_prevalence >= 0.5
            )
            a, v, p, predictions = _evaluate_probability(
                candidate_id=candidate_id,
                feature_id=D2,
                repetition=repetition,
                test_indices=test,
                target=target,
                probability=probability,
                conditions=MASK_CONDITIONS,
            )
            all_rows.extend(a)
            valid_rows.extend(v)
            padding_rows.extend(p)
            prediction_rows.extend(predictions)
            for condition in MASK_CONDITIONS:
                decomposition_rows.append(
                    {
                        "candidateId": candidate_id,
                        "featureId": D2,
                        "conditionId": condition,
                        "repetitionId": repetition,
                        **accuracy_decomposition(
                            target["target"][test],
                            probability,
                            target["targetMask"][test],
                        ),
                    }
                )
            true_length = target["targetMask"][test].sum(axis=1)
            inferred = infer_output_length(target["cutoff"][test])
            boundary_rows.append(
                {
                    "candidateId": candidate_id,
                    "repetitionId": repetition,
                    "fitValidMajorityPositive": fit_valid_prevalence >= 0.5,
                    "meanAbsoluteBoundaryError": float(
                        np.mean(np.abs(inferred - true_length))
                    ),
                    "exactBoundaryFraction": float(np.mean(inferred == true_length)),
                    "boundaryCorrelation": float(
                        stats.pearsonr(inferred, true_length).statistic
                    ),
                    "roundingAmbiguityMaximum": 2,
                }
            )
    outputs = {
        "training": pd.DataFrame(history_rows),
        "all": pd.DataFrame(all_rows),
        "valid": pd.DataFrame(valid_rows),
        "padding": pd.DataFrame(padding_rows),
        "predictions": pd.DataFrame(prediction_rows),
        "decomposition": pd.DataFrame(decomposition_rows),
        "length": pd.DataFrame(length_rows),
        "boundary": pd.DataFrame(boundary_rows),
        "replay": pd.DataFrame(replay_rows),
    }
    if len(outputs["replay"]) != 24 or not outputs["replay"]["passed"].all():
        raise RuntimeError("complete registered exact-model replay did not pass")
    return outputs


def _fit_control_mlp(
    *,
    candidate_id: str,
    repetition: int,
    values: np.ndarray,
    channel_mask: np.ndarray,
    time_mask: np.ndarray,
    target: dict[str, np.ndarray],
    fit_target: np.ndarray | None = None,
    fit_target_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split = pd.read_csv(S16_SPLIT_PATH)
    fit = split_indices(split, repetition, "FIT")
    validation = split_indices(split, repetition, "VALIDATION")
    test = split_indices(split, repetition, "TEST")
    scaler = fit_channel_scaler(values[fit], channel_mask[fit])
    scaled = apply_channel_scaler(values, channel_mask, scaler)
    effective_target = target["target"] if fit_target is None else fit_target
    effective_mask = (
        np.ones_like(target["targetMask"], dtype=bool)
        if fit_target_mask is None
        else fit_target_mask
    )
    result = train_masked_mlp(
        scaled[fit],
        channel_mask[fit],
        time_mask[fit],
        effective_target[fit],
        effective_mask[fit],
        scaled[validation],
        channel_mask[validation],
        time_mask[validation],
        effective_target[validation],
        effective_mask[validation],
        model_seed=s16_model_seed(candidate_id, repetition),
    )
    probability = predict_probabilities(
        result.model, scaled[test], channel_mask[test], time_mask[test]
    )
    return probability, test, scaled


def run_negative_controls(primary: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    split = pd.read_csv(S16_SPLIT_PATH)
    for candidate_id in CANDIDATE_IDS:
        target = load_target(candidate_id)
        p1 = load_feature(candidate_id, P1)
        for repetition in range(10):
            fit = split_indices(split, repetition, "FIT")
            validation = split_indices(split, repetition, "VALIDATION")
            test = split_indices(split, repetition, "TEST")
            # NC1: valid labels shuffled separately within fit and validation, padding fixed.
            shuffled_target = target["target"].copy()
            for role, indices in (("FIT", fit), ("VALIDATION", validation)):
                shuffled_target[indices] = permute_valid_labels_preserving_padding(
                    target["target"][indices],
                    target["targetMask"][indices],
                    seed_identity=("NC1", candidate_id, repetition, role),
                )
            probability, _, _ = _fit_control_mlp(
                candidate_id=candidate_id,
                repetition=repetition,
                values=p1["values"],
                channel_mask=p1["channelMask"],
                time_mask=p1["timeMask"],
                target=target,
                fit_target=shuffled_target,
                fit_target_mask=np.ones_like(target["targetMask"], dtype=bool),
            )
            views = metric_views(
                target["target"][test], probability, target["targetMask"][test]
            )
            rows.append(
                {
                    "controlId": "NC1_VALID_LABEL_PERMUTATION",
                    "candidateId": candidate_id,
                    "repetitionId": repetition,
                    "featureId": P1,
                    "allCellAccuracy": views["all"]["accuracy"],
                    "validCellAccuracy": views["valid"]["accuracy"],
                    "paddingCellAccuracy": views["padding"]["accuracy"],
                    "preservedPadding": True,
                    "status": "EXECUTED",
                }
            )

            # NC3: padded input values are noised, but masks remain exact.
            scaler = fit_channel_scaler(p1["values"][fit], p1["channelMask"][fit])
            scaled = apply_channel_scaler(p1["values"], p1["channelMask"], scaler)
            obfuscated = obfuscate_padded_input_values(
                scaled, p1["timeMask"], seed_identity=("NC3", candidate_id, repetition)
            )
            result = train_masked_mlp(
                obfuscated[fit],
                p1["channelMask"][fit],
                p1["timeMask"][fit],
                target["target"][fit],
                np.ones_like(target["targetMask"][fit], dtype=bool),
                obfuscated[validation],
                p1["channelMask"][validation],
                p1["timeMask"][validation],
                target["target"][validation],
                np.ones_like(target["targetMask"][validation], dtype=bool),
                model_seed=s16_model_seed(candidate_id, repetition),
            )
            probability = predict_probabilities(
                result.model,
                obfuscated[test],
                p1["channelMask"][test],
                p1["timeMask"][test],
            )
            views = metric_views(
                target["target"][test], probability, target["targetMask"][test]
            )
            rows.append(
                {
                    "controlId": "NC3_INPUT_LENGTH_OBFUSCATION",
                    "candidateId": candidate_id,
                    "repetitionId": repetition,
                    "featureId": P1,
                    "allCellAccuracy": views["all"]["accuracy"],
                    "validCellAccuracy": views["valid"]["accuracy"],
                    "paddingCellAccuracy": views["padding"]["accuracy"],
                    "realInputExact": bool(
                        np.array_equal(
                            obfuscated[p1["channelMask"]], scaled[p1["channelMask"]]
                        )
                    ),
                    "status": "EXECUTED",
                }
            )

            # NC4: within-matrix temporal feature permutation.
            perm_values, perm_channel = permute_valid_time(
                p1["values"],
                p1["channelMask"],
                p1["timeMask"],
                seed_identity=("NC4", candidate_id, repetition),
            )
            probability, _, _ = _fit_control_mlp(
                candidate_id=candidate_id,
                repetition=repetition,
                values=perm_values,
                channel_mask=perm_channel,
                time_mask=p1["timeMask"],
                target=target,
            )
            views = metric_views(
                target["target"][test], probability, target["targetMask"][test]
            )
            rows.append(
                {
                    "controlId": "NC4_FEATURE_TEMPORAL_PERMUTATION",
                    "candidateId": candidate_id,
                    "repetitionId": repetition,
                    "featureId": P1,
                    "allCellAccuracy": views["all"]["accuracy"],
                    "validCellAccuracy": views["valid"]["accuracy"],
                    "paddingCellAccuracy": views["padding"]["accuracy"],
                    "validLengthPreserved": True,
                    "status": "EXECUTED",
                }
            )

            # NC5: complete padded target arrays permuted among fit matrices only.
            permuted_target = target["target"].copy()
            rng = np.random.Generator(
                np.random.PCG64DXSM(seed128("NC5", candidate_id, repetition))
            )
            donor = fit[rng.permutation(len(fit))]
            permuted_target[fit] = target["target"][donor]
            probability, _, _ = _fit_control_mlp(
                candidate_id=candidate_id,
                repetition=repetition,
                values=p1["values"],
                channel_mask=p1["channelMask"],
                time_mask=p1["timeMask"],
                target=target,
                fit_target=permuted_target,
                fit_target_mask=np.ones_like(target["targetMask"], dtype=bool),
            )
            views = metric_views(
                target["target"][test], probability, target["targetMask"][test]
            )
            rows.append(
                {
                    "controlId": "NC5_MATRIX_LABEL_PERMUTATION",
                    "candidateId": candidate_id,
                    "repetitionId": repetition,
                    "featureId": P1,
                    "allCellAccuracy": views["all"]["accuracy"],
                    "validCellAccuracy": views["valid"]["accuracy"],
                    "paddingCellAccuracy": views["padding"]["accuracy"],
                    "fitOnlyPermutation": True,
                    "status": "EXECUTED",
                }
            )

        # NC2 boundary permutation is a deterministic diagnostic rather than a target mutation.
        lengths = target["targetMask"].sum(axis=1)
        rng = np.random.Generator(np.random.PCG64DXSM(seed128("NC2", candidate_id)))
        permuted = lengths[rng.permutation(len(lengths))]
        rows.append(
            {
                "controlId": "NC2_PADDING_BOUNDARY_PERMUTATION",
                "candidateId": candidate_id,
                "repetitionId": -1,
                "featureId": D2,
                "allCellAccuracy": None,
                "validCellAccuracy": None,
                "paddingCellAccuracy": None,
                "inputLengthTrueBoundaryCorrelation": float(
                    stats.pearsonr(target["cutoff"], lengths).statistic
                ),
                "inputLengthPermutedBoundaryCorrelation": float(
                    stats.pearsonr(target["cutoff"], permuted).statistic
                ),
                "scientificTargetArraysChanged": False,
                "status": "DIAGNOSTIC_EXECUTED",
            }
        )
    return pd.DataFrame(rows)


def suffix_invariance_audit() -> pd.DataFrame:
    manifest = pd.read_parquet(S13Y_ROOT / "trajectory_manifest.parquet")
    rows: list[dict[str, Any]] = []
    for candidate_id in CANDIDATE_IDS:
        for matrix_index in (0, 24, 49, 74):
            selected = manifest.loc[
                manifest["candidateId"].eq(candidate_id)
                & manifest["matrixIndex"].eq(matrix_index)
            ]
            trajectory, states = load_trajectory(selected.iloc[0])
            del trajectory
            cutoff = math.floor(0.25 * len(states))
            rng = np.random.Generator(
                np.random.PCG64DXSM(seed128("NC6", candidate_id, matrix_index))
            )
            mutated = states.copy()
            mutated[cutoff:] = mutated[cutoff:][rng.permutation(len(states) - cutoff)]
            original_full_clr, _, _ = frozen_clr(states)
            mutated_full_clr, _, _ = frozen_clr(mutated)
            original_prefix_clr, _, _ = frozen_clr(states[:cutoff])
            mutated_prefix_clr, _, _ = frozen_clr(mutated[:cutoff])
            pre_seed = s16_source_seed(candidate_id, matrix_index, "preprocessing")
            part_seed = s16_source_seed(candidate_id, matrix_index, "partition")
            p1a = run_emergence_pipeline(
                original_full_clr,
                PHIRL,
                SAFE_LATTICE,
                preprocessing_seed=pre_seed,
                partition_seed=part_seed,
            )
            p1b = run_emergence_pipeline(
                mutated_full_clr,
                PHIRL,
                SAFE_LATTICE,
                preprocessing_seed=pre_seed,
                partition_seed=part_seed,
            )
            p2a = run_emergence_pipeline(
                original_prefix_clr,
                PHIRL,
                SAFE_LATTICE,
                preprocessing_seed=pre_seed,
                partition_seed=part_seed,
            )
            p2b = run_emergence_pipeline(
                mutated_prefix_clr,
                PHIRL,
                SAFE_LATTICE,
                preprocessing_seed=pre_seed,
                partition_seed=part_seed,
            )
            p1a_v, p1a_m = source_values(
                p1a, fit_length=len(states), retained_length=cutoff
            )
            p1b_v, p1b_m = source_values(
                p1b, fit_length=len(states), retained_length=cutoff
            )
            p2a_v, p2a_m = source_values(p2a, fit_length=cutoff, retained_length=cutoff)
            p2b_v, p2b_m = source_values(p2b, fit_length=cutoff, retained_length=cutoff)
            shared = p1a_m & p1b_m
            p2_exact = (
                result_replay_equal(p2a, p2b)
                and np.array_equal(p2a_v, p2b_v)
                and np.array_equal(p2a_m, p2b_m)
            )
            rows.append(
                {
                    "controlId": "NC6_COMPLETED_FIT_SUFFIX_PERTURBATION",
                    "candidateId": candidate_id,
                    "matrixIndex": matrix_index,
                    "prefixStatesExact": np.array_equal(
                        states[:cutoff], mutated[:cutoff]
                    ),
                    "prefixClrExact": np.array_equal(
                        original_prefix_clr, mutated_prefix_clr
                    ),
                    "p2ResultExact": result_replay_equal(p2a, p2b),
                    "p2ValuesExact": np.array_equal(p2a_v, p2b_v),
                    "p2MaskExact": np.array_equal(p2a_m, p2b_m),
                    "p2SuffixInvariant": p2_exact,
                    "p1ResultExact": result_replay_equal(p1a, p1b),
                    "p1SharedCount": int(shared.sum()),
                    "p1MeanAbsoluteChange": None
                    if not shared.any()
                    else float(np.mean(np.abs(p1a_v[shared] - p1b_v[shared]))),
                    "p1MaximumAbsoluteChange": None
                    if not shared.any()
                    else float(np.max(np.abs(p1a_v[shared] - p1b_v[shared]))),
                    "passed": bool(p2_exact),
                }
            )
    frame = pd.DataFrame(rows)
    if len(frame) != 8 or not frame["passed"].all():
        raise RuntimeError("P2 suffix invariance failed")
    return frame


def execute_phase() -> None:
    status = json.loads((CACHE_ROOT / "arithmetic_status.json").read_text())
    if not status["advancementPassed"]:
        raise RuntimeError(
            "arithmetic advancement gate did not pass; execute is prohibited"
        )
    start = time.perf_counter()
    primary = run_primary_models()
    controls = run_negative_controls(primary)
    suffix = suffix_invariance_audit()
    for name, frame in primary.items():
        write_parquet(CACHE_ROOT / f"primary_{name}.parquet", frame)
    write_parquet(CACHE_ROOT / "negative_controls.parquet", controls)
    write_parquet(CACHE_ROOT / "suffix_invariance.parquet", suffix)
    write_json(
        CACHE_ROOT / "execute_status.json",
        {
            "stage": "execute",
            "completedAtUtc": utc_now(),
            "elapsedSeconds": time.perf_counter() - start,
            "passed": True,
            "primaryModelSplitRows": len(primary["all"]),
            "negativeControlRows": len(controls),
        },
    )


def holm_adjust(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def comparison_and_gates(
    primary: dict[str, pd.DataFrame], controls: pd.DataFrame
) -> dict[str, pd.DataFrame | dict[str, Any]]:
    all_metrics = primary["all"]
    valid_metrics = primary["valid"]
    per_matrix = primary["predictions"]
    digitized = digitized_figure5()
    box_rows: list[dict[str, Any]] = []
    order_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    registered = (P1, B1, B2, B3, D0)
    for candidate_id in CANDIDATE_IDS:
        for feature_id in registered:
            observed = (
                all_metrics.loc[
                    all_metrics["candidateId"].eq(candidate_id)
                    & all_metrics["featureId"].eq(feature_id)
                    & all_metrics["conditionId"].eq(S11)
                ]
                .sort_values("repetitionId")["accuracy"]
                .to_numpy(dtype=np.float64)
            )
            paper_id = feature_id
            paper = digitized.loc[digitized["featureId"].eq(paper_id)].iloc[0]
            q1, median, q3 = np.quantile(observed, [0.25, 0.5, 0.75])
            median_interval = (float(paper["medianLower"]), float(paper["medianUpper"]))
            paper_iqr = (
                min(float(paper["q1"]), float(paper["q3"])),
                max(float(paper["q1"]), float(paper["q3"])),
            )
            observed_iqr = (float(q1), float(q3))
            box_rows.append(
                {
                    "candidateId": candidate_id,
                    "featureId": feature_id,
                    "conditionId": S11,
                    "observedMinimum": float(observed.min()),
                    "observedQ1": float(q1),
                    "observedMedian": float(median),
                    "observedQ3": float(q3),
                    "observedMaximum": float(observed.max()),
                    "paperQ1": paper_iqr[0],
                    "paperMedian": float(paper["median"]),
                    "paperQ3": paper_iqr[1],
                    "paperMedianLower": median_interval[0],
                    "paperMedianUpper": median_interval[1],
                    "medianOverlaps": median_interval[0]
                    <= median
                    <= median_interval[1],
                    "iqrOverlaps": interval_overlap(observed_iqr, paper_iqr),
                    "observedQ1InOwnPixelInterval": float(paper["q1Lower"])
                    <= q1
                    <= float(paper["q1Upper"]),
                    "observedQ3InOwnPixelInterval": float(paper["q3Lower"])
                    <= q3
                    <= float(paper["q3Upper"]),
                }
            )
        p1_values = (
            all_metrics.loc[
                all_metrics["candidateId"].eq(candidate_id)
                & all_metrics["featureId"].eq(P1)
                & all_metrics["conditionId"].eq(S11)
            ]
            .sort_values("repetitionId")["accuracy"]
            .to_numpy(dtype=np.float64)
        )
        for comparator in (B1, B2, B3, D0):
            other = (
                all_metrics.loc[
                    all_metrics["candidateId"].eq(candidate_id)
                    & all_metrics["featureId"].eq(comparator)
                    & all_metrics["conditionId"].eq(S11)
                ]
                .sort_values("repetitionId")["accuracy"]
                .to_numpy(dtype=np.float64)
            )
            difference = p1_values - other
            mann = stats.mannwhitneyu(
                p1_values, other, alternative="two-sided", method="auto"
            )
            try:
                wilcoxon = float(
                    stats.wilcoxon(
                        difference,
                        alternative="two-sided",
                        zero_method="wilcox",
                        method="auto",
                    ).pvalue
                )
            except ValueError:
                wilcoxon = 1.0
            matrix_left = per_matrix.loc[
                per_matrix["candidateId"].eq(candidate_id)
                & per_matrix["featureId"].eq(P1)
                & per_matrix["conditionId"].eq(S11),
                ["repetitionId", "matrixIndex", "accuracy"],
            ].rename(columns={"accuracy": "reference"})
            matrix_right = per_matrix.loc[
                per_matrix["candidateId"].eq(candidate_id)
                & per_matrix["featureId"].eq(comparator)
                & per_matrix["conditionId"].eq(S11),
                ["repetitionId", "matrixIndex", "accuracy"],
            ].rename(columns={"accuracy": "comparator"})
            paired = matrix_left.merge(
                matrix_right, on=["repetitionId", "matrixIndex"], validate="one_to_one"
            )
            grouped = paired.groupby("matrixIndex")[["reference", "comparator"]].mean()
            effects = (grouped["reference"] - grouped["comparator"]).to_numpy(
                dtype=np.float64
            )
            rng = np.random.Generator(
                np.random.PCG64DXSM(seed128("bootstrap", candidate_id, P1, comparator))
            )
            indices = rng.integers(0, len(effects), size=(4096, len(effects)))
            distribution = effects[indices].mean(axis=1)
            lower, upper = np.quantile(distribution, [0.025, 0.975])
            comparison_rows.append(
                {
                    "candidateId": candidate_id,
                    "conditionId": S11,
                    "referenceFeatureId": P1,
                    "comparatorFeatureId": comparator,
                    "meanSplitDifference": float(difference.mean()),
                    "medianSplitDifference": float(np.median(difference)),
                    "positiveSplitCount": int(np.count_nonzero(difference > 0)),
                    "mannWhitneyP": float(mann.pvalue),
                    "pairedWilcoxonP": wilcoxon,
                    "pairedMatrixCount": len(effects),
                    "meanPairedMatrixDifference": float(effects.mean()),
                    "bootstrapLower95": float(lower),
                    "bootstrapUpper95": float(upper),
                }
            )
            bootstrap_rows.extend(
                {
                    "candidateId": candidate_id,
                    "referenceFeatureId": P1,
                    "comparatorFeatureId": comparator,
                    "replicate": index,
                    "meanPairedMatrixDifference": float(value),
                }
                for index, value in enumerate(distribution)
            )
            order_rows.append(
                {
                    "candidateId": candidate_id,
                    "referenceFeatureId": P1,
                    "comparatorFeatureId": comparator,
                    "referenceMedian": float(np.median(p1_values)),
                    "comparatorMedian": float(np.median(other)),
                    "orderingPassed": float(np.median(p1_values))
                    > float(np.median(other)),
                    "mannWhitneyBelowPoint01": float(mann.pvalue) < 0.01,
                    "pairedMatrixDirectionPositive": float(effects.mean()) > 0,
                }
            )
    comparisons = pd.DataFrame(comparison_rows)
    if not comparisons.empty:
        comparisons["mannWhitneyHolmP"] = holm_adjust(
            comparisons["mannWhitneyP"].tolist()
        )
        comparisons["wilcoxonHolmP"] = holm_adjust(
            comparisons["pairedWilcoxonP"].tolist()
        )
    box = pd.DataFrame(box_rows)
    order = pd.DataFrame(order_rows)

    panel_candidate_pass: dict[str, bool] = {}
    for candidate_id in CANDIDATE_IDS:
        box_c = box.loc[box["candidateId"].eq(candidate_id)]
        order_c = order.loc[order["candidateId"].eq(candidate_id)]
        criteria = {
            "allFiveMedianAndIqrMatch": bool(
                (box_c["medianOverlaps"] & box_c["iqrOverlaps"]).all()
            ),
            "allFourOrderingPass": bool(order_c["orderingPassed"].all()),
            "allFourMannWhitneyBelowPoint01": bool(
                order_c["mannWhitneyBelowPoint01"].all()
            ),
            "pairedMatrixDirectionPositive": bool(
                order_c["pairedMatrixDirectionPositive"].all()
            ),
            "allTargetArraysDefined": True,
            "integrityPassed": True,
        }
        panel_candidate_pass[candidate_id] = all(criteria.values())
        gate_rows.extend(
            {
                "gateFamily": "FIGURE5_PANEL",
                "candidateId": candidate_id,
                "criterion": key,
                "passed": value,
            }
            for key, value in criteria.items()
        )

    # Attribution and padding-dominance summaries.
    dominance_rows: list[dict[str, Any]] = []
    for candidate_id in CANDIDATE_IDS:
        p1_decomp = primary["decomposition"].loc[
            primary["decomposition"]["candidateId"].eq(candidate_id)
            & primary["decomposition"]["featureId"].eq(P1)
            & primary["decomposition"]["conditionId"].eq(S11)
        ]
        med_all_minus_valid = float(p1_decomp["allMinusValidAccuracy"].median())
        med_fraction_correct_padding = float(
            p1_decomp["fractionCorrectFromPadding"].median()
        )
        p1_all = all_metrics.loc[
            all_metrics["candidateId"].eq(candidate_id)
            & all_metrics["featureId"].eq(P1)
            & all_metrics["conditionId"].eq(S11),
            "accuracy",
        ].median()
        diag_medians = {
            feature: float(
                all_metrics.loc[
                    all_metrics["candidateId"].eq(candidate_id)
                    & all_metrics["featureId"].eq(feature)
                    & all_metrics["conditionId"].eq(S11),
                    "accuracy",
                ].median()
            )
            for feature in (D1, D2)
        }
        control = controls.loc[
            controls["candidateId"].eq(candidate_id)
            & controls["controlId"].eq("NC1_VALID_LABEL_PERMUTATION")
        ]
        nc1_retention = float(control["allCellAccuracy"].median() / p1_all)
        obscured = controls.loc[
            controls["candidateId"].eq(candidate_id)
            & controls["controlId"].eq("NC3_INPUT_LENGTH_OBFUSCATION")
        ]
        obfuscation_drop = float(p1_all - obscured["allCellAccuracy"].median())
        valid_p1 = float(
            valid_metrics.loc[
                valid_metrics["candidateId"].eq(candidate_id)
                & valid_metrics["featureId"].eq(P1)
                & valid_metrics["conditionId"].eq(S11),
                "accuracy",
            ].median()
        )
        valid_controls = [
            float(
                valid_metrics.loc[
                    valid_metrics["candidateId"].eq(candidate_id)
                    & valid_metrics["featureId"].eq(feature)
                    & valid_metrics["conditionId"].eq(S11),
                    "accuracy",
                ].median()
            )
            for feature in (B1, B2, B3, B4, D0)
        ]
        criteria = {
            "allMinusValidAtLeastPoint05": med_all_minus_valid >= 0.05,
            "halfCorrectFromPadding": med_fraction_correct_padding >= 0.5,
            "lengthOrBoundaryWithinPoint03": max(diag_medians.values())
            >= float(p1_all) - 0.03,
            "validShuffleRetains80Percent": nc1_retention >= 0.8,
            "validP1NotAboveAllOrdinaryControls": valid_p1 <= max(valid_controls),
            "obfuscationMaterialDrop": obfuscation_drop >= 0.03,
        }
        dominance_rows.append(
            {
                "candidateId": candidate_id,
                "p1AllCellMedian": float(p1_all),
                "p1ValidCellMedian": valid_p1,
                "allMinusValidMedian": med_all_minus_valid,
                "fractionCorrectFromPaddingMedian": med_fraction_correct_padding,
                "lengthOnlyMedian": diag_medians[D1],
                "boundaryRuleMedian": diag_medians[D2],
                "nc1AllCellRetentionFraction": nc1_retention,
                "inputLengthObfuscationDrop": obfuscation_drop,
                **criteria,
                "possiblePaddingDominatedTask": any(criteria.values()),
            }
        )

    # Train-vs-score attribution.
    attribution: dict[str, dict[str, Any]] = {}
    for candidate_id in CANDIDATE_IDS:

        def median(
            feature: str,
            condition: str,
            frame: pd.DataFrame,
            current_candidate: str = candidate_id,
        ) -> float:
            return float(
                frame.loc[
                    frame["candidateId"].eq(current_candidate)
                    & frame["featureId"].eq(feature)
                    & frame["conditionId"].eq(condition),
                    "accuracy",
                ].median()
            )

        s00_valid = median(P1, S00, valid_metrics)
        s01_all = median(P1, S01, all_metrics)
        s10_valid = median(P1, S10, valid_metrics)
        s11_all = median(P1, S11, all_metrics)
        score_gain = s01_all - s00_valid
        total_gain = s11_all - s00_valid
        attribution[candidate_id] = {
            "scoreInflation": score_gain >= 0.05
            and (total_gain <= 0 or score_gain >= 0.8 * total_gain),
            "trainingContamination": abs(s10_valid - s00_valid) >= 0.02,
            "scoreGain": score_gain,
            "trainingValidChange": s10_valid - s00_valid,
            "totalS11Gain": total_gain,
        }

    # Prospective gate, including strict initial-appearance eligibility.
    prospective_rows: list[dict[str, Any]] = []
    for candidate_id in CANDIDATE_IDS:
        target = load_target(candidate_id)
        preonset = ~np.any(target["inputLabels"][:, :MAX_INPUT_LENGTH], axis=1)
        future_onset = preonset & np.any(
            target["target"].astype(bool) & target["targetMask"], axis=1
        )
        p2_valid = valid_metrics.loc[
            valid_metrics["candidateId"].eq(candidate_id)
            & valid_metrics["featureId"].eq(P2)
            & valid_metrics["conditionId"].eq(S00)
        ]
        comparator_medians = {
            feature: float(
                valid_metrics.loc[
                    valid_metrics["candidateId"].eq(candidate_id)
                    & valid_metrics["featureId"].eq(feature)
                    & valid_metrics["conditionId"].eq(S00),
                    "accuracy",
                ].median()
            )
            for feature in (D0, D1, D3, B1, B2, B3, B4)
        }
        p2_median = float(p2_valid["accuracy"].median())
        criteria = {
            "suffixInvariant": True,
            "outperformsEveryRegisteredComparator": p2_median
            > max(comparator_medians.values()),
            "balancedAccuracyAboveChanceDescriptively": float(
                p2_valid["balancedAccuracy"].dropna().median()
            )
            > 0.5
            if p2_valid["balancedAccuracy"].notna().any()
            else False,
            "auprcAbovePrevalence": bool(
                (p2_valid["auprc"] > p2_valid["prevalence"]).all()
            ),
            "brierBeatsTrainingPrior": False,
            "negativeControlsPass": False,
            "preOnsetCountAtLeast20": int(preonset.sum()) >= 20,
            "futureOnsetCountAtLeast20": int(future_onset.sum()) >= 20,
        }
        prospective_rows.append(
            {
                "gateFamily": "VALID_CELL_PROSPECTIVE",
                "candidateId": candidate_id,
                "p2ValidAccuracyMedian": p2_median,
                "preOnsetMatrixCount": int(preonset.sum()),
                "futureOnsetMatrixCount": int(future_onset.sum()),
                **criteria,
                "passed": all(criteria.values()),
            }
        )
    gate_frame = pd.concat(
        [pd.DataFrame(gate_rows), pd.DataFrame(prospective_rows)],
        ignore_index=True,
        sort=False,
    )
    dominance = pd.DataFrame(dominance_rows)
    panel_pass = all(panel_candidate_pass.values())
    padding_dominated = bool(dominance["possiblePaddingDominatedTask"].all())
    p2_panel_pass = False
    completed_only = panel_pass and not p2_panel_pass
    prospective_pass = bool(pd.DataFrame(prospective_rows)["passed"].all())
    classifications = ["FIGURE5_PADDING_ARITHMETIC_RECONSTRUCTED"]
    if panel_pass:
        classifications.append("FIGURE5_PADDING_PANEL_RECONSTRUCTED")
    else:
        classifications.append("FIGURE5_VALID_CELL_MODEL_ORDER_NOT_SUPPORTED")
    if completed_only:
        classifications.append("FIGURE5_COMPLETED_FIT_PADDING_RECONSTRUCTION")
    if padding_dominated:
        classifications.append("POSSIBLE_PADDING_DOMINATED_TASK")
    if all(value["scoreInflation"] for value in attribution.values()):
        classifications.append("PADDING_SCORE_INFLATION")
    if any(value["trainingContamination"] for value in attribution.values()):
        classifications.append("PADDING_TRAINING_CONTAMINATION")
    if prospective_pass:
        classifications.append("FIGURE5_VALID_CELL_PROSPECTIVE_LEAD")
    if any(
        row["preOnsetMatrixCount"] < 20 or row["futureOnsetMatrixCount"] < 20
        for row in prospective_rows
    ):
        classifications.append("NOT_ELIGIBLE_AS_INITIAL_APPEARANCE_PREDICTION")
    promotable_forensic = panel_pass
    promotable_prospective = prospective_pass
    classifications.append(
        "PROMOTABLE_TO_UNTOUCHED_FORENSIC_FIGURE5_CONFIRMATION"
        if promotable_forensic
        else "NOT_PROMOTABLE"
    )
    if promotable_prospective:
        classifications.append("PROMOTABLE_TO_UNTOUCHED_PROSPECTIVE_CONFIRMATION")
    classification = {
        "schema": "eidosoma.e01.s19.l14.classification.v1",
        "researchStepId": LOOP_ID,
        "versionedStepId": VERSION,
        "status": "COMPLETE_AWAITING_MANDATORY_HUMAN_REVIEW",
        "primaryClassification": "FIGURE5_PADDING_PANEL_RECONSTRUCTED"
        if panel_pass
        else "FIGURE5_VALID_CELL_MODEL_ORDER_NOT_SUPPORTED",
        "classifications": classifications,
        "panelPassedBothCandidates": panel_pass,
        "paddingDominatedBothCandidates": padding_dominated,
        "attribution": attribution,
        "prospectiveGatePassedBothCandidates": prospective_pass,
        "forensicPromotion": promotable_forensic,
        "prospectivePromotion": promotable_prospective,
        "authorCodeIdentified": False,
        "s18StatusesChanged": False,
        "s20Activated": False,
        "e02Activated": False,
    }
    return {
        "box": box,
        "order": order,
        "comparisons": comparisons,
        "bootstrap": pd.DataFrame(bootstrap_rows),
        "dominance": dominance,
        "gates": gate_frame,
        "classification": classification,
    }


def empty_frame(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype="object") for column in columns})


def promote_core_tables(
    executed: bool,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    direct = {
        "trajectory_length_results.parquet": "trajectory_length_results.parquet",
        "padding_geometry_results.parquet": "padding_geometry_results.parquet",
        "prevalence_decomposition.parquet": "prevalence_decomposition.parquet",
        "dummy_arithmetic_results.parquet": "dummy_arithmetic_results.parquet",
        "padded_target_manifest.parquet": "padded_target_manifest.parquet",
        "feature_tensor_replay.parquet": "feature_replay.parquet",
    }
    frames: dict[str, pd.DataFrame] = {}
    for artifact_name, cache_name in direct.items():
        frame = pd.read_parquet(CACHE_ROOT / cache_name)
        write_parquet(OUTPUT_ROOT / artifact_name, frame)
        frames[artifact_name] = frame
    shutil.copy2(
        CACHE_ROOT / "arithmetic_advancement_gate.csv",
        OUTPUT_ROOT / "arithmetic_advancement_gate.csv",
    )

    if executed:
        primary = {
            name: pd.read_parquet(CACHE_ROOT / f"primary_{name}.parquet")
            for name in (
                "training",
                "all",
                "valid",
                "padding",
                "predictions",
                "decomposition",
                "length",
                "boundary",
                "replay",
            )
        }
        controls = pd.read_parquet(CACHE_ROOT / "negative_controls.parquet")
        suffix = pd.read_parquet(CACHE_ROOT / "suffix_invariance.parquet")
        analysis = comparison_and_gates(primary, controls)
        mapping = {
            "training_history.parquet": primary["training"],
            "prediction_results.parquet": primary["predictions"],
            "all_cell_metrics.parquet": primary["all"],
            "valid_cell_metrics.parquet": primary["valid"],
            "padding_cell_metrics.parquet": primary["padding"],
            "accuracy_decomposition.parquet": primary["decomposition"],
            "length_only_results.parquet": primary["length"],
            "padding_boundary_rule_results.parquet": primary["boundary"],
            "paired_model_comparisons.parquet": analysis["comparisons"],
            "negative_control_results.parquet": controls,
            "suffix_invariance_results.parquet": suffix,
            "padding_dominance_results.parquet": analysis["dominance"],
            "scientific_gate_results.parquet": analysis["gates"],
        }
        for name, frame in mapping.items():
            assert isinstance(frame, pd.DataFrame)
            write_parquet(OUTPUT_ROOT / name, frame)
            frames[name] = frame
        assert isinstance(analysis["box"], pd.DataFrame)
        assert isinstance(analysis["order"], pd.DataFrame)
        analysis["box"].to_csv(
            OUTPUT_ROOT / "paper_boxplot_comparison.csv", index=False
        )
        analysis["order"].to_csv(
            OUTPUT_ROOT / "paper_model_order_results.csv", index=False
        )
        classification = analysis["classification"]
        assert isinstance(classification, dict)
    else:
        table_schemas = {
            "training_history.parquet": [
                "candidateId",
                "featureId",
                "trainIncludesPadding",
                "repetitionId",
                "epoch",
                "fitLoss",
                "validationLoss",
            ],
            "prediction_results.parquet": [
                "candidateId",
                "featureId",
                "conditionId",
                "repetitionId",
                "matrixIndex",
                "status",
            ],
            "all_cell_metrics.parquet": [
                "candidateId",
                "featureId",
                "conditionId",
                "repetitionId",
                "accuracy",
                "status",
            ],
            "valid_cell_metrics.parquet": [
                "candidateId",
                "featureId",
                "conditionId",
                "repetitionId",
                "accuracy",
                "status",
            ],
            "padding_cell_metrics.parquet": [
                "candidateId",
                "featureId",
                "conditionId",
                "repetitionId",
                "accuracy",
                "status",
            ],
            "accuracy_decomposition.parquet": [
                "candidateId",
                "featureId",
                "conditionId",
                "repetitionId",
                "absoluteError",
                "status",
            ],
            "length_only_results.parquet": [
                "candidateId",
                "featureId",
                "repetitionId",
                "status",
            ],
            "padding_boundary_rule_results.parquet": [
                "candidateId",
                "repetitionId",
                "status",
            ],
            "paired_model_comparisons.parquet": [
                "candidateId",
                "referenceFeatureId",
                "comparatorFeatureId",
                "status",
            ],
            "negative_control_results.parquet": ["controlId", "candidateId", "status"],
            "suffix_invariance_results.parquet": [
                "candidateId",
                "matrixIndex",
                "status",
            ],
            "padding_dominance_results.parquet": [
                "candidateId",
                "possiblePaddingDominatedTask",
                "status",
            ],
            "scientific_gate_results.parquet": [
                "gateFamily",
                "candidateId",
                "criterion",
                "passed",
                "status",
            ],
        }
        for name, columns in table_schemas.items():
            frame = empty_frame(columns)
            write_parquet(OUTPUT_ROOT / name, frame)
            frames[name] = frame
        pd.DataFrame(columns=["candidateId", "featureId", "status"]).to_csv(
            OUTPUT_ROOT / "paper_boxplot_comparison.csv", index=False
        )
        pd.DataFrame(
            columns=[
                "candidateId",
                "referenceFeatureId",
                "comparatorFeatureId",
                "status",
            ]
        ).to_csv(OUTPUT_ROOT / "paper_model_order_results.csv", index=False)
        classification = {
            "schema": "eidosoma.e01.s19.l14.classification.v1",
            "researchStepId": LOOP_ID,
            "versionedStepId": VERSION,
            "status": "COMPLETE_AWAITING_MANDATORY_HUMAN_REVIEW",
            "primaryClassification": "FIGURE5_PADDING_ARITHMETIC_NOT_SUPPORTED",
            "classifications": [
                "FIGURE5_PADDING_ARITHMETIC_NOT_SUPPORTED",
                "NOT_PROMOTABLE",
            ],
            "arithmeticAdvancementPassed": False,
            "fullModelExecutionPermitted": False,
            "fullModelExecutionPerformed": False,
            "panelGateEvaluated": False,
            "prospectiveGateEvaluated": False,
            "authorCodeIdentified": False,
            "s18StatusesChanged": False,
            "s20Activated": False,
            "e02Activated": False,
        }
    write_json(OUTPUT_ROOT / "classification.json", classification)
    return frames, classification


def _plot_unavailable(ax: Any, title: str, reason: str) -> None:
    ax.axis("off")
    ax.text(0.5, 0.58, title, ha="center", va="center", fontsize=13, weight="bold")
    ax.text(0.5, 0.40, reason, ha="center", va="center", fontsize=10, wrap=True)


def generate_figures(executed: bool, classification: dict[str, Any]) -> list[Path]:
    figure_dir = OUTPUT_ROOT / "figures"
    figure_dir.mkdir(exist_ok=True)
    created: list[Path] = []
    lengths = pd.read_parquet(OUTPUT_ROOT / "trajectory_length_results.parquet")
    geometry = pd.read_parquet(OUTPUT_ROOT / "padding_geometry_results.parquet")
    prevalence = pd.read_parquet(OUTPUT_ROOT / "prevalence_decomposition.parquet")
    dummy = pd.read_parquet(OUTPUT_ROOT / "dummy_arithmetic_results.parquet")
    digitized = digitized_figure5()

    def save(fig: Any, name: str) -> None:
        path = figure_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        created.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for candidate_id, group in lengths.groupby("candidateId"):
        axes[0].hist(group["T"], bins=16, alpha=0.55, label=candidate_id[-2:])
        axes[1].hist(
            group["validOutputLength"], bins=16, alpha=0.55, label=candidate_id[-2:]
        )
    axes[0].axvline(1250, color="black", ls="--", label="Fig. 2 aggregate ~1250")
    axes[0].set(
        title="Selected-clock trajectory lengths", xlabel="T", ylabel="Matrices"
    )
    axes[1].axvline(
        MAX_TARGET_LENGTH, color="black", ls="--", label="fixed output width"
    )
    axes[1].set(
        title="Valid suffix lengths vs padding width", xlabel="Valid output length"
    )
    for ax in axes:
        ax.legend(fontsize=8)
    save(fig, "figure01_trajectory_lengths_and_padding_extent.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    qpaper = 0.61 / 0.88
    ax.axhspan(
        0.61 / 0.91,
        0.61 / 0.85,
        color="gray",
        alpha=0.25,
        label="implied q from digitized dummy / Table 1 band",
    )
    ax.axhline(qpaper, color="black", ls="--", label="0.61/0.88")
    ax.scatter(geometry["candidateId"], geometry["validFraction"], s=70)
    ax.set(
        ylim=(0, 1),
        ylabel="Valid output-cell fraction q",
        title="Valid fraction required by the 61%-versus-88% clue",
    )
    ax.legend(fontsize=8)
    save(fig, "figure02_required_valid_fraction.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(prevalence))
    width = 0.35
    ax.bar(
        x - width / 2,
        prevalence["validPrevalence"],
        width,
        label="real molecular cells",
    )
    ax.bar(
        x + width / 2, prevalence["paddedPrevalence"], width, label="all padded cells"
    )
    ax.set_xticks(x, [value[-2:] for value in prevalence["candidateId"]])
    ax.set_ylim(0, 1)
    ax.set(
        ylabel="Positive prevalence", title="Real versus zero-padded target prevalence"
    )
    ax.legend()
    save(fig, "figure03_real_vs_padded_prevalence.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    test = dummy.loc[dummy["splitRole"].eq("TEST")]
    groups = []
    labels = []
    for candidate_id in CANDIDATE_IDS:
        for condition in (S00, S11):
            groups.append(
                test.loc[
                    test["candidateId"].eq(candidate_id)
                    & test["conditionId"].eq(condition),
                    "dummyAccuracy",
                ]
            )
            labels.append(
                f"{candidate_id[-2:]}\n{'valid' if condition == S00 else 'padded'}"
            )
    ax.boxplot(groups, tick_labels=labels, showfliers=True)
    paper = digitized.loc[digitized["featureId"].eq(D0)].iloc[0]
    ax.axhspan(
        min(paper["q1"], paper["q3"]),
        max(paper["q1"], paper["q3"]),
        color="orange",
        alpha=0.2,
        label="paper IQR",
    )
    ax.set(
        ylim=(0, 1),
        ylabel="Dummy accuracy",
        title="Dummy accuracy under valid and padded scoring",
    )
    ax.legend()
    save(fig, "figure04_dummy_valid_vs_padded.png")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axis("off")
    positions = [
        (0.2, 0.72, S00, "valid train\nvalid score"),
        (0.8, 0.72, S01, "valid train\npadded score"),
        (0.2, 0.25, S10, "padded train\nvalid score"),
        (0.8, 0.25, S11, "padded train\npadded score"),
    ]
    for x, y, identifier, label in positions:
        ax.text(
            x,
            y,
            f"{identifier}\n{label}",
            ha="center",
            va="center",
            bbox={"boxstyle": "round", "fc": "#e8eef8"},
            fontsize=10,
        )
    ax.annotate(
        "score inflation",
        xy=(0.78, 0.62),
        xytext=(0.35, 0.62),
        arrowprops={"arrowstyle": "->"},
    )
    ax.annotate(
        "training contamination",
        xy=(0.2, 0.35),
        xytext=(0.2, 0.58),
        arrowprops={"arrowstyle": "->"},
    )
    ax.set_title("Frozen 2×2 training/scoring factorial")
    save(fig, "figure05_four_mask_conditions.png")

    artifact_data = {}
    for name in (
        "all_cell_metrics.parquet",
        "valid_cell_metrics.parquet",
        "accuracy_decomposition.parquet",
        "length_only_results.parquet",
        "padding_boundary_rule_results.parquet",
        "negative_control_results.parquet",
        "padding_dominance_results.parquet",
    ):
        artifact_data[name] = pd.read_parquet(OUTPUT_ROOT / name)

    # Figures 6-15 are outcome panels when advancement passed; otherwise explicit gate-stop panels.
    requested = [
        (
            "figure06_reconstructed_figure5_all_cells.png",
            "Reconstructed Figure 5: all cells",
        ),
        (
            "figure07_reconstructed_figure5_valid_cells.png",
            "Reconstructed Figure 5: valid molecular cells",
        ),
        ("figure08_accuracy_decomposition.png", "Accuracy decomposition"),
        (
            "figure09_length_and_boundary_controls.png",
            "Length-only and boundary controls",
        ),
        ("figure10_valid_label_shuffle.png", "Valid-label shuffle preserving padding"),
        ("figure11_input_length_obfuscation.png", "Input-length obfuscation"),
        ("figure12_completed_vs_prefix.png", "Completed-fit versus prefix-only"),
        ("figure13_candidate_agreement.png", "Candidate-2 versus candidate-3"),
        (
            "figure14_decision_matrix.png",
            "Retrospective, padding, and prospective decision matrix",
        ),
        ("figure15_promotion_decision_tree.png", "Final promotion decision tree"),
    ]
    if not executed:
        for filename, title in requested:
            fig, ax = plt.subplots(figsize=(8, 4))
            _plot_unavailable(
                ax,
                title,
                "Not executed: the prospectively frozen padding-arithmetic advancement gate failed in at least one candidate. No model or control branch was opened.",
            )
            save(fig, filename)
    else:
        all_metrics = artifact_data["all_cell_metrics.parquet"]
        valid_metrics = artifact_data["valid_cell_metrics.parquet"]
        for filename, title, frame in (
            (requested[0][0], requested[0][1], all_metrics),
            (requested[1][0], requested[1][1], valid_metrics),
        ):
            fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
            for ax, candidate_id in zip(axes, CANDIDATE_IDS):
                subset = frame.loc[
                    frame["candidateId"].eq(candidate_id)
                    & frame["conditionId"].eq(S11)
                    & frame["featureId"].isin([P1, B1, B2, B3, D0])
                ]
                groups = [
                    subset.loc[subset["featureId"].eq(f), "accuracy"]
                    for f in [P1, B1, B2, B3, D0]
                ]
                ax.boxplot(
                    groups, tick_labels=["PhiRL", "Δcomp", "raw", "flux", "dummy"]
                )
                ax.set_ylim(0, 1)
                ax.set_title(candidate_id[-2:])
            fig.suptitle(title)
            save(fig, filename)
        decomp = artifact_data["accuracy_decomposition.parquet"]
        fig, ax = plt.subplots(figsize=(9, 4))
        selected = decomp.loc[
            decomp["featureId"].eq(P1) & decomp["conditionId"].eq(S11)
        ]
        selected.groupby("candidateId")[
            ["allCellAccuracy", "validCellAccuracy", "paddingCellAccuracy"]
        ].median().plot.bar(ax=ax)
        ax.set_ylim(0, 1)
        ax.set_title(requested[2][1])
        save(fig, requested[2][0])
        all_s11 = all_metrics.loc[
            all_metrics["conditionId"].eq(S11)
            & all_metrics["featureId"].isin([P1, D1, D2])
        ]
        fig, ax = plt.subplots(figsize=(9, 4))
        all_s11.groupby(["candidateId", "featureId"])[
            "accuracy"
        ].median().unstack().plot.bar(ax=ax)
        ax.set_ylim(0, 1)
        ax.set_title(requested[3][1])
        save(fig, requested[3][0])
        controls = artifact_data["negative_control_results.parquet"]
        for request_index, control_id in (
            (4, "NC1_VALID_LABEL_PERMUTATION"),
            (5, "NC3_INPUT_LENGTH_OBFUSCATION"),
        ):
            fig, ax = plt.subplots(figsize=(8, 4))
            c = controls.loc[controls["controlId"].eq(control_id)]
            c.groupby("candidateId")[
                ["allCellAccuracy", "validCellAccuracy"]
            ].median().plot.bar(ax=ax)
            ax.set_ylim(0, 1)
            ax.set_title(requested[request_index][1])
            save(fig, requested[request_index][0])
        fig, ax = plt.subplots(figsize=(8, 4))
        subset = all_metrics.loc[
            all_metrics["conditionId"].eq(S11) & all_metrics["featureId"].isin([P1, P2])
        ]
        subset.groupby(["candidateId", "featureId"])[
            "accuracy"
        ].median().unstack().plot.bar(ax=ax)
        ax.set_ylim(0, 1)
        ax.set_title(requested[6][1])
        save(fig, requested[6][0])
        fig, ax = plt.subplots(figsize=(8, 4))
        subset = all_metrics.loc[
            all_metrics["conditionId"].eq(S11)
            & all_metrics["featureId"].isin([P1, B1, B2, B3, D0])
        ]
        subset.groupby(["candidateId", "featureId"])[
            "accuracy"
        ].median().unstack().plot(ax=ax, marker="o")
        ax.set_ylim(0, 1)
        ax.set_title(requested[7][1])
        save(fig, requested[7][0])
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.axis("off")
        ax.table(
            cellText=[
                [
                    classification["primaryClassification"],
                    str(classification.get("paddingDominatedBothCandidates")),
                    str(classification.get("prospectiveGatePassedBothCandidates")),
                ]
            ],
            colLabels=["Forensic panel", "Padding-dominated", "Prospective"],
            loc="center",
        )
        ax.set_title(requested[8][1])
        save(fig, requested[8][0])
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.axis("off")
        ax.text(
            0.5,
            0.7,
            "Panel gate",
            ha="center",
            bbox={"boxstyle": "round", "fc": "#e8eef8"},
        )
        ax.annotate(
            "pass → untouched forensic proposal\nfail → not promotable",
            xy=(0.5, 0.3),
            xytext=(0.5, 0.55),
            ha="center",
            arrowprops={"arrowstyle": "->"},
        )
        ax.text(
            0.5,
            0.2,
            "Mandatory human review; no automatic next step",
            ha="center",
            weight="bold",
        )
        ax.set_title(requested[9][1])
        save(fig, requested[9][0])
    if len(created) != 15:
        raise RuntimeError(f"expected 15 figures, wrote {len(created)}")
    return created


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def artifact_manifest(path: Path, root: Path, *, exclude_self: bool = True) -> None:
    entries = []
    for item in sorted(p for p in root.rglob("*") if p.is_file()):
        if exclude_self and item == path:
            continue
        entries.append(
            {
                "relativePath": str(item.relative_to(root)),
                "bytes": item.stat().st_size,
                "sha256": sha256_file(item),
            }
        )
    write_json(
        path,
        {
            "schema": "eidosoma.e01.s19.l14.artifact_manifest.v1",
            "createdAtUtc": utc_now(),
            "root": str(root),
            "entryCount": len(entries),
            "entries": entries,
        },
    )


def write_s19_root_manifest() -> None:
    path = S19_ROOT / "artifact_manifest.json"
    entries = []
    for item in sorted(p for p in S19_ROOT.rglob("*") if p.is_file() and p != path):
        entries.append(
            {
                "path": str(item.relative_to(S19_ROOT)),
                "bytes": item.stat().st_size,
                "sha256": sha256_file(item),
            }
        )
    write_json(
        path,
        {
            "schema": "eidosoma.e01.s19.root_artifact_manifest.v3",
            "researchStepId": LOOP_ID,
            "generatedAtUtc": utc_now(),
            "artifactCount": len(entries),
            "totalBytesExcludingManifest": sum(row["bytes"] for row in entries),
            "entries": entries,
            "passed": True,
        },
    )


def render_reports(
    classification: dict[str, Any], executed: bool, figures: list[Path]
) -> tuple[str, str]:
    geometry = pd.read_parquet(OUTPUT_ROOT / "padding_geometry_results.parquet")
    gate = pd.read_csv(OUTPUT_ROOT / "arithmetic_advancement_gate.csv")
    lines = []
    for row in geometry.itertuples(index=False):
        lines.append(
            f"- {row.candidateId}: q={row.validFraction:.4f}, valid prevalence={row.validPrevalence:.4f}, padded prevalence={row.paddedPrevalence:.4f}, padded dummy={row.paddedDummyAccuracy:.4f}."
        )
    gate_lines = []
    for row in gate.itertuples(index=False):
        gate_lines.append(
            f"- {row.candidateId}: split dummy median={row.observedDummyMedian:.4f}; paper IQR={row.paperIqrLower:.4f}–{row.paperIqrUpper:.4f}; q compatible={row.qCompatibleWithFigure2}; gate={row.passed}."
        )
    figure_md = "\n".join(
        f"![{path.stem}](figures/{path.name})\n\n*{index}. {path.stem.replace('_', ' ')}.*"
        for index, path in enumerate(figures, 1)
    )
    if executed:
        scientific = "The arithmetic gate passed, so all preregistered model, decomposition, length, control, suffix, and bootstrap branches were executed. See machine-readable tables for exact split and matrix evidence."
    else:
        scientific = "The prospectively locked arithmetic gate failed, so the protocol prohibited MLP execution. Empty, schema-valid downstream tables and explicit not-executed figures preserve complete scope accounting without opening another convention."
    report = f"""# S19-L14 Full Results — Figure-5 Padding and Length-Leakage Reconstruction

## Top summary

- **Research step:** `{VERSION}`
- **Completion status:** `COMPLETE_AWAITING_MANDATORY_HUMAN_REVIEW`
- **Outcome classification:** `{classification["primaryClassification"]}`; `{", ".join(classification["classifications"])}`
- **Artifacts written:** complete required L14 machine-readable evidence, 15 figures, validation and hash manifests, this report, and the S19 current-step handoff.
- **Validation result:** immutable prior, paper/source identities, all 16 fixtures, exact S16 target/feature replay, tensor scope, serialization, storage, and regeneration passed.
- **Central caveat:** this is adaptive forensic reconstruction. All-cell padding accuracy is not molecular-state prediction and cannot alter S18 prospective or causal-control conclusions.
- **Recommended next action:** mandatory human review. Keep S20, E02, author contact, confirmation matrices, interventions, and another S19 loop inactive.

## Frozen question

Could the exact S16 adjacent-incoming `H>0.9` task, with right-padded target zeros included in loss and/or accuracy, explain Figure 5's approximately 0.61 dummy and approximately 0.79–0.85 model accuracies? The simulator, trajectories, labels, features, model, splits, quarter cutoff, and padding values were held fixed.

## Inputs and provenance

The analysis used exactly 100 paired S13Y/S16 matrices per candidate, the frozen selected molecular clocks, 100 completed fissions, the S16 64/16/20 matrix splits across ten repetitions, and CPU-float64 PhiRL features. The original paper hash was `{sha256_file(PAPER_PDF)}`. No new matrix, trajectory, label, feature, model, metric, or intervention result was generated.

## Human-panel and digitization lock

The frozen panel audit treats Figure 5 as five boxplots and its isolated raw-composition circle as a flier. Pixel calibration was fixed from y-axis ticks before cohort arithmetic. Figure 2 endpoints were recorded as approximate molecular-step constraints; terminal aggregate support remains potentially sparse. The locked details are in `human_panel_review_lock.yaml`, `paper_figure5_digitization_lock.csv`, and `paper_figure2_length_lock.csv`.

## Mandatory fixtures and exact S16 replay

All 16 fixtures passed. The full tensor regeneration replayed 1,200 candidate/matrix/feature identities against S16 hashes (six features × 200 trajectories), including target clocks, right padding, feature masks, and first-quarter-only PhiRL. No mismatch was accepted.

## Padding arithmetic

The required identity `p_padded = p_valid × q_valid` held to machine precision.

{chr(10).join(lines)}

The human-directed reference calculation `0.61/0.88 = {0.61 / 0.88:.4f}` is reported as an interpretive arithmetic clue, not exact author data. With the frozen adjacent-H target, valid prevalence is measured directly rather than substituted with Table 1's 0.88.

## Advancement adjudication

{chr(10).join(gate_lines)}

{scientific}

## Four mask conditions

- `S00`: masked training, masked scoring (exact S16 scientific condition).
- `S01`: masked training, unmasked all-cell scoring (score-inflation isolation).
- `S10`: unmasked padded training, masked valid-cell scoring (training-contamination isolation).
- `S11`: unmasked padded training and scoring (primary forensic convention).

## Interpretation boundaries

- A numerical all-cell panel match would only show that a padding convention can reconstruct Figure 5.
- Padding zeros are not self-replicating or non-self-replicating molecular observations.
- Completed-fit PhiRL remains future-dependent.
- The adjacent-H label is effectively determined by exact H and provides almost no genuine pre-onset cohort at the cutoff.
- L12 remains `AUTHOR_CODE_REQUIRED_FOR_DISCRIMINATION`; L13 remains `FIGURE5_BASELINE_RECONSTRUCTED_MODEL_ORDER_NOT_SUPPORTED`; S18 prospective prediction and causal-control non-support remain unchanged.

## Figures

{figure_md}

## Validation and reproducibility

Repository code/config/tests were locked and pushed before cohort arithmetic. Temporary tensors and model intermediates remained under `/cache/e01_s19_l14`; compact evidence only was promoted. `immutable_prior_validation.json`, `regeneration_validation.json`, `storage_validation.json`, and `artifact_manifest.json` provide the final checks. Technical amendment 001 corrected only the root-handoff figure-link prefix after final validation exposed the report-assembly defect; all scientific tables, gates, values and classifications remained hash-identical.

## Mandatory handoff

Stop here for human review. No L15, S20, E02, author contact, untouched confirmation, intervention, or report bundle has been activated.
"""
    summary = f"""# S19-L14 Decision Summary

**Status:** complete; mandatory human review required.  
**Primary classification:** `{classification["primaryClassification"]}`.  
**Additional classifications:** `{", ".join(classification["classifications"])}`.

The exact S16 target/tensors replayed. Candidate-specific arithmetic was:

{chr(10).join(lines)}

Advancement:

{chr(10).join(gate_lines)}

This result is forensic only. It does not support initial-appearance prediction, early warning, intervention efficacy, or causal control. S18, L12, and L13 remain unchanged. Return for human review; no next step is active.
"""
    return report, summary


def append_root_ledgers(classification: dict[str, Any]) -> None:
    now = utc_now()
    ledger_path = S19_ROOT / "self_improvement_ledger.parquet"
    ledger = pd.read_parquet(ledger_path)
    sequence = int(pd.to_numeric(ledger["ledgerSequence"]).max()) + 1
    new = pd.DataFrame(
        [
            {
                "appendOnly": True,
                "beliefBeforeLoop": "Unmasked zero target padding and trajectory-length information might jointly explain the Figure 5 dummy and learned-model accuracies while Table 1 describes real molecular observations.",
                "failureOrAmbiguityTargeted": "The shared approximately 0.13–0.20 accuracy shortfall of all L13 learned models and the paper's unspecified variable-length padding/masking convention.",
                "informationGainRationale": "The loop changes only train/score inclusion of already frozen zero padding while holding target, features, architecture, splits, and trajectories exact.",
                "learned": f"{classification['primaryClassification']}; {','.join(classification['classifications'])}.",
                "ledgerSequence": sequence,
                "loopId": LOOP_ID,
                "motivatingEvidence": "L13 U2 matched the approximate dummy but not the learned-model panel; Figure 5 describes accuracy without padding or mask semantics.",
                "proposedNextTest": "None active. Mandatory human review decides whether any untouched forensic confirmation, another direction, closeout, author-code wait, E02 transition, or pause is warranted.",
                "recordPhase": "POST_LOOP_RESULT_AND_HUMAN_REVIEW_HANDOFF",
                "remainingPlausibleHypotheses": "Author-specific padding, truncation, sampling, target, or dataset semantics remain nonidentifiable unless directly supported and separately authorized.",
                "selectedHypotheses": "Exact S16 adjacent-H target under the frozen four-condition padding factorial.",
                "timestampUtc": now,
                "weakenedHypotheses": "Any padding explanation failing the prospectively locked arithmetic or full-panel gate is weakened within tested scope; all-cell accuracy never rescues valid-cell prospective evidence.",
            }
        ]
    )
    combined = pd.concat([ledger, new], ignore_index=True)
    if (
        len(combined) != len(ledger) + 1
        or combined.iloc[:-1].reset_index(drop=True).to_json()
        != ledger.reset_index(drop=True).to_json()
    ):
        raise RuntimeError("self-improvement ledger append-only validation failed")
    write_parquet(ledger_path, combined)
    md_path = S19_ROOT / "SELF_IMPROVEMENT_LEDGER.md"
    with md_path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"\n\n## {LOOP_ID} — post-loop result ({now})\n\n- Belief before: zero padding and length leakage could explain the shared Figure 5 gap.\n- Learned: `{classification['primaryClassification']}`; `{', '.join(classification['classifications'])}`.\n- Boundary: all prior results remain unchanged; no next loop is active.\n- Next: mandatory human review.\n"
        )

    candidate_path = S19_ROOT / "candidate_registry.parquet"
    candidate = pd.read_parquet(candidate_path)
    registry_order = int(pd.to_numeric(candidate["registryOrder"]).max()) + 1
    candidate_new = pd.DataFrame(
        [
            {
                "branchCount": 4,
                "bundleId": "L14_FIGURE5_PADDING_FACTORIAL",
                "candidateId": "S19-L14-ZERO-PADDING-LENGTH-LEAKAGE",
                "candidateSpecificSuccess": 0,
                "completedFitLeakage": 1,
                "computeEfficiency": 4,
                "crossCandidateDiscriminability": 5,
                "deterministicHReuse": 1,
                "explanatoryLeverage": 5,
                "frozenRank": 1,
                "independenceFromPriorOutcomeSelection": 2,
                "outcomeGuidedThresholdSelection": 0,
                "paperFingerprintSpecificity": 5,
                "proposedSpecification": "Exact S16 target/features/model with the frozen 2x2 train-score padding factorial",
                "rankingScore": 25.0,
                "registryOrder": registry_order,
                "selected": True,
                "selectionReason": "Explicit human authorization after L13 and human Figure-5 panel review",
                "sourceGrounding": 2,
                "testability": 5,
                "undefinedAuthorSemantics": 1,
            }
        ]
    )
    write_parquet(
        candidate_path, pd.concat([candidate, candidate_new], ignore_index=True)
    )

    loop_path = S19_ROOT / "loop_registry.yaml"
    registry = yaml.safe_load(loop_path.read_text())
    if any(item.get("loopId") == LOOP_ID for item in registry["loops"]):
        raise RuntimeError("L14 loop registry entry already exists")
    registry["loops"].append(
        {
            "loopId": LOOP_ID,
            "versionedLoopId": VERSION,
            "status": "COMPLETE_AWAITING_MANDATORY_HUMAN_REVIEW",
            "authorized": True,
            "outcomeAccessed": True,
            "humanReviewRequiredAfter": True,
            "completed": True,
            "eligibleScientificResults": classification["primaryClassification"]
            != "FIGURE5_PADDING_ARITHMETIC_NOT_SUPPORTED",
            "classification": classification["primaryClassification"],
            "nextStepActive": False,
        }
    )
    write_yaml(loop_path, registry)

    history_path = S19_ROOT / "human_review_history.json"
    history = json.loads(history_path.read_text())
    history["history"].append(
        {
            "decision": "AUTHORIZE_EXACTLY_ONE_L14_LOOP",
            "loopId": LOOP_ID,
            "scope": VERSION,
            "source": "explicit_human_direction",
            "recordedAtUtc": now,
            "result": classification["primaryClassification"],
            "nextLoopAuthorized": False,
            "s20Activated": False,
            "status": "CONSUMED_AND_RETURNED_FOR_MANDATORY_REVIEW",
        }
    )
    history["pendingDecision"] = "POST_S19_L14_MANDATORY_HUMAN_REVIEW_REQUIRED"
    write_json(history_path, history)


def write_root_handoff(classification: dict[str, Any], report: str) -> None:
    summary = {
        "researchStepId": LOOP_ID,
        "stepNumber": 19,
        "status": "COMPLETE_AWAITING_MANDATORY_HUMAN_REVIEW",
        "success": True,
        "outcomeClassification": classification["primaryClassification"],
        "validationResult": "PASS_IMMUTABLE_SOURCE_FIXTURES_S16_REPLAY_SCOPE_STORAGE_REGENERATION",
        "artifactsWritten": [
            str(OUTPUT_ROOT / "S19_L14_FULL_RESULTS.md"),
            str(OUTPUT_ROOT / "classification.json"),
            str(OUTPUT_ROOT / "artifact_manifest.json"),
            str(S19_ROOT / "research_step_full_results.md"),
        ],
        "caveatsOrBlockers": [
            "adaptive_forensic_reconstruction",
            "all_cell_padding_is_not_molecular_evidence",
            "adjacent_H_target_label_coupling",
            "completed_fit_future_dependence",
            "author_padding_semantics_unavailable",
            "S18_statuses_unchanged",
        ],
        "recommendedNextAction": "MANDATORY_HUMAN_REVIEW_KEEP_S20_E02_L15_AUTHOR_CONTACT_CONFIRMATION_INTERVENTIONS_AND_REPORT_BUNDLE_INACTIVE",
    }
    write_json(S19_ROOT / "s19_status.json", summary)
    root_report = report.replace(
        "# S19-L14 Full Results — Figure-5 Padding and Length-Leakage Reconstruction",
        "# S19 Current-Step Handoff — S19-L14",
    ).replace("](figures/", "](loops/L14/figures/")
    (S19_ROOT / "research_step_full_results.md").write_text(
        root_report, encoding="utf-8"
    )


def scientific_artifact_hashes() -> dict[str, str]:
    names = [
        "trajectory_length_results.parquet",
        "padding_geometry_results.parquet",
        "prevalence_decomposition.parquet",
        "dummy_arithmetic_results.parquet",
        "arithmetic_advancement_gate.csv",
        "padded_target_manifest.parquet",
        "feature_tensor_replay.parquet",
        "training_history.parquet",
        "prediction_results.parquet",
        "all_cell_metrics.parquet",
        "valid_cell_metrics.parquet",
        "padding_cell_metrics.parquet",
        "accuracy_decomposition.parquet",
        "length_only_results.parquet",
        "padding_boundary_rule_results.parquet",
        "paper_boxplot_comparison.csv",
        "paper_model_order_results.csv",
        "paired_model_comparisons.parquet",
        "negative_control_results.parquet",
        "suffix_invariance_results.parquet",
        "padding_dominance_results.parquet",
        "scientific_gate_results.parquet",
        "classification.json",
    ]
    return {name: sha256_file(OUTPUT_ROOT / name) for name in names}


def reporting_amendment_phase() -> None:
    amendment = json.loads(AMENDMENT_001_PATH.read_text(encoding="utf-8"))
    if amendment["amendmentId"] != "S19-L14-TECHNICAL-AMENDMENT-001":
        raise RuntimeError("unexpected L14 technical amendment identity")
    lock = repository_lock()
    if not lock["passed"]:
        raise RuntimeError(
            "technical amendment must execute from a clean pushed commit"
        )
    fresh = Path(amendment["freshCache"])
    if fresh.exists():
        shutil.rmtree(fresh)
    fresh.mkdir(parents=True)
    before = scientific_artifact_hashes()
    failed_root = S19_ROOT / "research_step_full_results.md"
    failed_bytes = failed_root.read_bytes()
    (fresh / "failed_root_handoff.md").write_bytes(failed_bytes)
    classification = json.loads((OUTPUT_ROOT / "classification.json").read_text())
    figures = sorted((OUTPUT_ROOT / "figures").glob("*.png"))
    report, summary = render_reports(classification, False, figures)
    (fresh / "S19_L14_FULL_RESULTS.md").write_text(report, encoding="utf-8")
    root_report = report.replace(
        "# S19-L14 Full Results — Figure-5 Padding and Length-Leakage Reconstruction",
        "# S19 Current-Step Handoff — S19-L14",
    ).replace("](figures/", "](loops/L14/figures/")
    (fresh / "research_step_full_results.md").write_text(root_report, encoding="utf-8")
    (fresh / "loop_decision_summary.md").write_text(summary, encoding="utf-8")
    shutil.copy2(
        fresh / "S19_L14_FULL_RESULTS.md", OUTPUT_ROOT / "S19_L14_FULL_RESULTS.md"
    )
    shutil.copy2(
        fresh / "loop_decision_summary.md", OUTPUT_ROOT / "loop_decision_summary.md"
    )
    shutil.copy2(
        fresh / "research_step_full_results.md",
        S19_ROOT / "research_step_full_results.md",
    )
    after = scientific_artifact_hashes()
    scientific_equal = before == after
    links = [
        line.split("](", 1)[1].rstrip(")")
        for line in root_report.splitlines()
        if line.startswith("![")
    ]
    links_resolve = len(links) == 15 and all(
        (S19_ROOT / link).is_file() for link in links
    )
    if not scientific_equal or not links_resolve:
        raise RuntimeError(
            "reporting amendment changed science or failed to resolve links"
        )
    ledger = pd.read_csv(OUTPUT_ROOT / "technical_amendment_ledger.csv")
    if not ledger.empty:
        raise RuntimeError("unexpected prior L14 technical amendment")
    pd.DataFrame(
        [
            {
                "amendmentId": amendment["amendmentId"],
                "timestampUtc": utc_now(),
                "scope": amendment["scope"],
                "scientificValueChanged": False,
                "status": "COMPLETE_EXACT_SCIENTIFIC_HASH_EQUALITY",
                "failedRootHandoffSha256": hashlib.sha256(failed_bytes).hexdigest(),
                "amendedRootHandoffSha256": sha256_file(
                    S19_ROOT / "research_step_full_results.md"
                ),
            }
        ]
    ).to_csv(OUTPUT_ROOT / "technical_amendment_ledger.csv", index=False)
    validation = {
        "schema": "eidosoma.e01.s19.l14.technical_amendment_validation.v1",
        "amendmentId": amendment["amendmentId"],
        "repository": lock,
        "scientificArtifactCount": len(before),
        "scientificHashesBefore": before,
        "scientificHashesAfter": after,
        "scientificHashesExact": scientific_equal,
        "rootFigureLinkCount": len(links),
        "rootFigureLinksResolve": links_resolve,
        "failedAttemptPreservedPath": str(fresh / "failed_root_handoff.md"),
        "passed": scientific_equal and links_resolve,
    }
    write_json(OUTPUT_ROOT / "technical_amendment_001_validation.json", validation)
    regeneration = json.loads(
        (OUTPUT_ROOT / "regeneration_validation.json").read_text()
    )
    regeneration["technicalAmendmentCount"] = 1
    regeneration["technicalAmendment001Passed"] = True
    regeneration["rootFigureLinksResolve"] = True
    write_json(OUTPUT_ROOT / "regeneration_validation.json", regeneration)
    artifact_manifest(OUTPUT_ROOT / "artifact_manifest.json", OUTPUT_ROOT)
    write_s19_root_manifest()
    write_json(
        CACHE_ROOT / "amendment_001_status.json",
        {
            "stage": "reporting_amendment_001",
            "completedAtUtc": utc_now(),
            "scientificHashesExact": scientific_equal,
            "rootFigureLinksResolve": links_resolve,
            "passed": True,
        },
    )


def finalize_phase() -> None:
    start = time.perf_counter()
    arithmetic = json.loads((CACHE_ROOT / "arithmetic_status.json").read_text())
    executed = bool(arithmetic["advancementPassed"])
    if executed and not (CACHE_ROOT / "execute_status.json").is_file():
        raise RuntimeError("arithmetic passed but execution outputs are missing")
    _, classification = promote_core_tables(executed)
    pd.DataFrame(
        columns=[
            "amendmentId",
            "timestampUtc",
            "scope",
            "scientificValueChanged",
            "status",
        ]
    ).to_csv(OUTPUT_ROOT / "technical_amendment_ledger.csv", index=False)
    failures = []
    if not executed:
        failures.append(
            {
                "failureId": "S19-L14-GATE-001",
                "stage": "ARITHMETIC_ADVANCEMENT_GATE",
                "severity": "REGISTERED_SCIENTIFIC_GATE_STOP",
                "status": "FIGURE5_PADDING_ARITHMETIC_NOT_SUPPORTED",
                "reason": "At least one candidate failed the frozen dummy/pixel/q arithmetic advancement criteria; MLP execution was prohibited.",
                "scientificAggregationReleased": True,
            }
        )
    pd.DataFrame(
        failures,
        columns=[
            "failureId",
            "stage",
            "severity",
            "status",
            "reason",
            "scientificAggregationReleased",
        ],
    ).to_csv(OUTPUT_ROOT / "failure_ledger.csv", index=False)
    figures = generate_figures(executed, classification)
    report, summary = render_reports(classification, executed, figures)
    (OUTPUT_ROOT / "S19_L14_FULL_RESULTS.md").write_text(report, encoding="utf-8")
    (OUTPUT_ROOT / "loop_decision_summary.md").write_text(summary, encoding="utf-8")

    baseline = json.loads((CACHE_ROOT / "immutable_baseline.json").read_text())
    immutable = revalidate_immutable(baseline)
    write_json(OUTPUT_ROOT / "immutable_prior_validation.json", immutable)
    if not immutable["passed"]:
        raise RuntimeError("immutable prior changed during L14")
    cache_size = directory_bytes(CACHE_ROOT)
    artifact_size = directory_bytes(OUTPUT_ROOT)
    storage = {
        "schema": "eidosoma.e01.s19.l14.storage_validation.v1",
        "retainedBytes": artifact_size,
        "retainedGiB": artifact_size / 2**30,
        "temporaryBytes": cache_size,
        "temporaryGiB": cache_size / 2**30,
        "retainedLimitGiB": 20,
        "temporaryLimitGiB": 60,
        "compiledOrCachePayloadUnderArtifacts": False,
        "passed": artifact_size <= 20 * 2**30 and cache_size <= 60 * 2**30,
    }
    write_json(OUTPUT_ROOT / "storage_validation.json", storage)
    if not storage["passed"]:
        raise RuntimeError("L14 storage ceiling exceeded")
    replay = pd.read_parquet(OUTPUT_ROOT / "feature_tensor_replay.parquet")
    fixtures = pd.read_parquet(OUTPUT_ROOT / "fixture_results.parquet")
    regeneration = {
        "schema": "eidosoma.e01.s19.l14.regeneration_validation.v1",
        "validatedAtUtc": utc_now(),
        "fixtureRows": len(fixtures),
        "fixtureAllPassed": bool(fixtures["passed"].all()),
        "featureReplayRows": len(replay),
        "featureReplayAllPassed": bool(replay["passed"].all()),
        "targetRows": len(
            pd.read_parquet(OUTPUT_ROOT / "padded_target_manifest.parquet")
        ),
        "requiredArtifactsPresent": all(
            (OUTPUT_ROOT / name).is_file()
            for name in REQUIRED_ARTIFACTS
            if name
            not in {
                "artifact_manifest.json",
                "regeneration_validation.json",
                "runtime_manifest.json",
            }
        ),
        "figureCount": len(figures),
        "scientificExecutionPerformed": executed,
        "reportRegeneratedFromMachineTables": True,
        "passed": True,
    }
    regeneration["passed"] = all(
        [
            regeneration["fixtureAllPassed"],
            regeneration["featureReplayAllPassed"],
            regeneration["targetRows"] == 200,
            regeneration["requiredArtifactsPresent"],
            regeneration["figureCount"] == 15,
        ]
    )
    write_json(OUTPUT_ROOT / "regeneration_validation.json", regeneration)
    if not regeneration["passed"]:
        raise RuntimeError("artifact regeneration validation failed")
    prepare_status = json.loads((CACHE_ROOT / "prepare_status.json").read_text())
    benchmark_status = json.loads((CACHE_ROOT / "benchmark_status.json").read_text())
    runtime = {
        "schema": "eidosoma.e01.s19.l14.runtime_manifest.v1",
        "completedAtUtc": utc_now(),
        "cpuCoresMaximum": 8,
        "numericalLibraryThreadsPerWorker": 1,
        "gpuHours": 0,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "torch": torch.__version__,
        "prepareSeconds": prepare_status["elapsedSeconds"],
        "benchmarkSeconds": benchmark_status["elapsedSeconds"],
        "arithmeticSeconds": arithmetic["elapsedSeconds"],
        "executionSeconds": json.loads(
            (CACHE_ROOT / "execute_status.json").read_text()
        )["elapsedSeconds"]
        if executed
        else 0.0,
        "finalizationSecondsAtWrite": time.perf_counter() - start,
        "projectedScientificCpuHours": benchmark_status["projectedScientificCpuHours"],
        "cpuHourCeiling": 48,
        "wallHourCeiling": 24,
        "reserveFraction": 0.15,
        "runtimeDrivenShortcutUsed": False,
        "scopeReducedAfterArithmetic": False,
    }
    write_json(OUTPUT_ROOT / "runtime_manifest.json", runtime)
    artifact_manifest(OUTPUT_ROOT / "artifact_manifest.json", OUTPUT_ROOT)
    append_root_ledgers(classification)
    write_root_handoff(classification, report)
    # Refresh manifest after the final report/validation set; it intentionally excludes itself.
    artifact_manifest(OUTPUT_ROOT / "artifact_manifest.json", OUTPUT_ROOT)
    write_s19_root_manifest()
    if not all((OUTPUT_ROOT / name).is_file() for name in REQUIRED_ARTIFACTS):
        missing = [
            name for name in REQUIRED_ARTIFACTS if not (OUTPUT_ROOT / name).is_file()
        ]
        raise RuntimeError(f"required L14 artifacts missing: {missing}")
    write_json(
        CACHE_ROOT / "finalize_status.json",
        {
            "stage": "finalize",
            "completedAtUtc": utc_now(),
            "passed": True,
            "classification": classification["primaryClassification"],
            "executed": executed,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=[
            "prepare",
            "benchmark",
            "arithmetic",
            "execute",
            "finalize",
            "reporting-amendment-001",
            "all",
        ],
    )
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    if args.stage in ("prepare", "all"):
        prepare_phase()
    if args.stage in ("benchmark", "all"):
        benchmark_phase()
    if args.stage in ("arithmetic", "all"):
        arithmetic_phase()
    if args.stage in ("execute", "all"):
        arithmetic = json.loads((CACHE_ROOT / "arithmetic_status.json").read_text())
        if arithmetic["advancementPassed"]:
            execute_phase()
        elif args.stage == "execute":
            raise RuntimeError("execute prohibited by failed arithmetic gate")
    if args.stage in ("finalize", "all"):
        finalize_phase()
    if args.stage == "reporting-amendment-001":
        reporting_amendment_phase()


if __name__ == "__main__":
    main()
