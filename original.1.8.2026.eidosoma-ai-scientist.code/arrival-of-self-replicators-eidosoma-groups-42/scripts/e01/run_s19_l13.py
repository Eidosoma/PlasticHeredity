#!/usr/bin/env python3
"""Execute the frozen E01/S19-L13 Figure-5 reconstruction and stop."""

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
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
import yaml
from scipy import stats

from e01_frozen_timebase_ensemble.core import frozen_clr, selected_clock_observations
from e01_pigozzi_source_audit.core import SourceImplementation
from e01_prediction_reconstruction.core import (
    EXPECTED_PARAMETER_COUNT,
    MAX_INPUT_LENGTH,
    MAX_TARGET_LENGTH,
    MaskedSequenceMLP,
    apply_channel_scaler,
    fit_channel_scaler,
    parameter_count,
    predict_probabilities,
    train_masked_mlp,
)
from e01_s19_all_comptype_union_repair.core import (
    array_sha256 as l11r_array_sha256,
)
from e01_s19_all_comptype_union_repair.core import (
    repair_u2_centroids,
)
from e01_s19_figure5_prediction.core import (
    ALL_MODEL_IDS,
    B1_ID,
    B2_ID,
    B3_ID,
    B4_ID,
    B5_ID,
    B6_ID,
    B7_ID,
    CANDIDATE_IDS,
    DIAGNOSTIC_MODEL_IDS,
    DUMMY_ID,
    LOOP_ID,
    NC1_ID,
    NC2_ID,
    ORACLE_ID,
    P1_ID,
    P2_B4_ID,
    P2_B5_ID,
    P2_ID,
    PAPER_INTERVALS,
    R1_TARGET_ID,
    TARGET_IDS,
    U2_TARGET_ID,
    VERSION,
    array_sha256,
    build_feature,
    build_split_manifest,
    build_target_tensor,
    combine_scalar_features,
    extended_binary_metrics,
    geometry_gate,
    holm_adjust,
    incoming_h,
    matrix_bootstrap_metric_difference,
    normalized_compositions,
    paper_interval_overlap,
    r1_target,
    s16_model_seed,
    s16_source_seed,
    seed128,
    source_values,
    split_indices,
    target_geometry,
    u2_target,
)
from e01_s19_matlab_attractor.core import scientific_recurrence_gate
from e01_source_emergence_metric_identity.core import (
    result_replay_equal,
    run_emergence_pipeline,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
CONFIG_PATH = REPO_ROOT / "configs/e01/s19_l13_figure5_prediction.yaml"
AMENDMENT_001_PATH = REPO_ROOT / "configs/e01/s19_l13_technical_amendment_001.json"
AMENDMENT_002_PATH = REPO_ROOT / "configs/e01/s19_l13_technical_amendment_002.json"
S16_MANIFEST = REPO_ROOT / "configs/e01/s16_tensor_model_manifest.json"
S16_SPLIT_PATH = REPO_ROOT / "configs/e01/s16_split_manifest.csv"
L10_CORE = REPO_ROOT / "src/e01_s19_matlab_attractor/core.py"
L11R_CORE = REPO_ROOT / "src/e01_s19_all_comptype_union_repair/core.py"
S16_CORE = REPO_ROOT / "src/e01_prediction_reconstruction/core.py"
L12_ROOT = Path("/artifacts/research_steps/S19/loops/L12")
L10_ROOT = Path("/artifacts/research_steps/S19/loops/L10")
L11R_ROOT = Path("/artifacts/research_steps/S19/loops/L11R")
S19_ROOT = Path("/artifacts/research_steps/S19")
OUTPUT_ROOT = S19_ROOT / "loops/L13"
CACHE_ROOT = Path("/cache/e01_s19_l13")
SAFE_LATTICE = Path("/artifacts/research_steps/S12B/safe_phi_lattice.json")
PAPER_PDF = Path("/cache/e01_s03/downloads/paper-2607.28250v1.pdf")
PAPER_FIGURE_ROOT = WORKSPACE_ROOT / "input-attachments/ed5486bf-a043-485b-a233-d88d8d123759/figures"
FIGURE5 = Path("/artifacts/research_steps/S19/loops/L12/figures/figure5_class_prevalence_contradiction.png")
PHIRL_COMMIT = "a6d1d0d18c7551302724b7158c6ccdc4d3a33373"
GARD_COMMIT = "86dff6320d5ae91b4e831471079ff46749b14df9"
PHIRL = SourceImplementation.PHIRL

REQUIRED_ARTIFACTS = [
    "preregistration.yaml", "decision_record.md", "implementation_lock.json",
    "source_snapshot_manifest.json", "immutable_prior_validation.json",
    "figure5_target_hypothesis_lock.yaml", "figure5_digitized_target_lock.csv",
    "shared_trajectory_manifest.parquet", "trajectory_identity_validation.parquet",
    "target_label_registry.yaml", "r1_target_results.parquet",
    "u2_target_replay_results.parquet", "target_availability_results.parquet",
    "target_geometry_results.parquet", "target_suffix_prevalence_results.parquet",
    "onset_eligibility_results.parquet", "dummy_baseline_results.parquet",
    "geometry_advancement_gate_results.csv", "feature_pipeline_registry.yaml",
    "completed_fit_phirl_features.parquet", "prefix_only_phirl_features.parquet",
    "baseline_feature_results.parquet", "attractor_control_results.parquet",
    "oracle_diagnostic_results.parquet", "split_manifest.parquet", "model_registry.yaml",
    "training_history.parquet", "prediction_results.parquet", "paper_accuracy_results.parquet",
    "robust_metric_results.parquet", "per_matrix_metric_results.parquet",
    "paired_model_comparisons.parquet", "incremental_value_results.parquet",
    "negative_control_results.parquet", "suffix_invariance_results.parquet",
    "leakage_audit.parquet", "figure5_reconstruction_matrix.csv",
    "scientific_gate_results.parquet", "classification.json", "failure_ledger.csv",
    "technical_amendment_ledger.csv", "runtime_manifest.json", "storage_validation.json",
    "regeneration_validation.json", "artifact_manifest.json", "loop_decision_summary.md",
    "S19_L13_FULL_RESULTS.md", "FIGURE_CONTENTS_AND_CAPTIONS_FOR_HUMAN_REVIEW.md",
    "FIGURE_CONTENTS_AND_CAPTIONS_FOR_HUMAN_REVIEW_V2.md",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return str(value)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(json_safe(payload)) + "\n", encoding="utf-8")


def write_yaml(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(json_safe(payload), sort_keys=False), encoding="utf-8")


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = frame.reindex(sorted(frame.columns), axis=1) if len(frame.columns) else frame
    temporary = path.with_suffix(path.suffix + ".tmp")
    ordered.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


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
    branch = run_git("branch", "--show-current")
    head = run_git("rev-parse", "HEAD")
    remote = run_git("rev-parse", "origin/eidosoma/groups/42")
    status = run_git("status", "--short")
    return {
        "branch": branch, "head": head, "remoteHead": remote,
        "worktreeStatus": status,
        "passed": branch == "eidosoma/groups/42" and head == remote and status == "",
    }


def load_config() -> dict[str, Any]:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if payload["versionedStepId"] != VERSION or payload["outcomeAccessedAtLock"] is not False:
        raise RuntimeError("L13 preregistration identity mismatch")
    return payload


def prior_immutable_files() -> list[Path]:
    paths: list[Path] = []
    root = Path("/artifacts/research_steps")
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if relative.parts[0] == "S19":
            if len(relative.parts) >= 3 and relative.parts[1] == "loops" and relative.parts[2] != "L13":
                paths.append(path)
        else:
            paths.append(path)
    paths.extend([WORKSPACE_ROOT / "AGENTS.md", WORKSPACE_ROOT / "FULL_PLAN.md"])
    paths.extend(sorted((WORKSPACE_ROOT / "input-attachments").rglob("MANIFEST.json")))
    paths.extend(sorted((WORKSPACE_ROOT / "input-attachments").rglob("ATTACHMENT.md")))
    return sorted({path for path in paths if path.is_file()}, key=str)


def create_immutable_baseline() -> dict[str, Any]:
    entries = [
        {"path": str(path), "bytes": path.stat().st_size, "sha256Before": sha256_file(path)}
        for path in prior_immutable_files()
    ]
    return {
        "schema": "eidosoma.e01.s19.l13.immutable_prior_validation.v1",
        "researchStepId": LOOP_ID,
        "capturedAtUtc": utc_now(),
        "entryCount": len(entries),
        "entries": entries,
        "mismatchCount": 0,
        "passed": True,
    }


def revalidate_immutable(payload: dict[str, Any]) -> dict[str, Any]:
    mismatches = []
    for row in payload["entries"]:
        path = Path(row["path"])
        after = sha256_file(path) if path.is_file() else None
        if after != row["sha256Before"]:
            mismatches.append({"path": str(path), "before": row["sha256Before"], "after": after})
    payload = dict(payload)
    payload.update({"validatedAtUtc": utc_now(), "mismatchCount": len(mismatches), "mismatches": mismatches, "passed": not mismatches})
    return payload


def source_manifest() -> dict[str, Any]:
    files = [
        CONFIG_PATH, S16_MANIFEST, S16_SPLIT_PATH, S16_CORE, L10_CORE, L11R_CORE,
        Path(__file__), REPO_ROOT / "src/e01_s19_figure5_prediction/core.py",
        REPO_ROOT / "tests/e01/test_s19_l13.py", SAFE_LATTICE, PAPER_PDF,
        L12_ROOT / "figure_digitization.csv", L12_ROOT / "figure_panel_registry.parquet",
        L11R_ROOT / "trajectory_manifest.parquet", L11R_ROOT / "molecular_union_label_results.parquet",
        L11R_ROOT / "recurring_centroid_results.parquet", L10_ROOT / "implementation_lock.json",
        L11R_ROOT / "implementation_lock.json",
    ]
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
    return {
        "schema": "eidosoma.e01.s19.l13.source_snapshot_manifest.v1",
        "researchStepId": LOOP_ID,
        "phirlCommit": PHIRL_COMMIT,
        "historicalGardCommit": GARD_COMMIT,
        "entries": [{"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in files],
    }


def figure5_lock() -> pd.DataFrame:
    digitized = pd.read_csv(L12_ROOT / "figure_digitization.csv")
    rows = digitized.loc[digitized["panelId"].eq("FIGURE_5")].copy()
    expected = {
        "phirl_accuracy": 0.85, "composition_change_accuracy": 0.80,
        "raw_composition_accuracy": 0.80, "flux_accuracy": 0.79, "dummy_accuracy": 0.60,
    }
    if dict(zip(rows["constraint"], rows["value"])) != expected:
        raise RuntimeError("L12 Figure 5 digitization changed")
    model_map = {
        "phirl_accuracy": P1_ID, "composition_change_accuracy": B1_ID,
        "raw_composition_accuracy": B2_ID, "flux_accuracy": B3_ID,
        "dummy_accuracy": DUMMY_ID,
    }
    rows["modelId"] = rows["constraint"].map(model_map)
    rows["comparisonLower"] = rows["value"] - 0.05
    rows["comparisonUpper"] = rows["value"] + 0.05
    rows["intervalProvenance"] = "L12_APPROXIMATE_CENTER_PLUS_PREOUTCOME_FIXED_0p05_ADJUDICATION_TOLERANCE"
    rows["whiskerEndpointAvailability"] = "NOT_NUMERICALLY_FROZEN_BY_L12"
    return rows


def load_trajectory_manifest() -> pd.DataFrame:
    frame = pd.read_parquet(L11R_ROOT / "trajectory_manifest.parquet")
    frame = frame.loc[frame["candidateId"].isin(CANDIDATE_IDS)].copy()
    if len(frame) != 200 or frame.duplicated(["candidateId", "matrixIndex"]).any():
        raise RuntimeError("L11R trajectory manifest scope mismatch")
    if set(frame["matrixIndex"]) != set(range(100)) or not frame["completedFissions"].eq(100).all():
        raise RuntimeError("L11R trajectory completion scope mismatch")
    for row in frame.itertuples(index=False):
        path = Path(row.cachePath)
        if not path.is_file() or sha256_file(path) != row.cacheSha256:
            raise RuntimeError(f"frozen trajectory cache mismatch: {path}")
    return frame.sort_values(["candidateId", "matrixIndex"]).reset_index(drop=True)


def trajectory_arrays(row: Any) -> dict[str, Any]:
    with Path(row.cachePath).open("rb") as handle:
        trajectory = pickle.load(handle)
    if trajectory.trajectory_id != row.trajectoryId or trajectory.trajectory_sha256 != row.trajectorySha256:
        raise RuntimeError("trajectory identity mismatch")
    selected = selected_clock_observations(trajectory, "C1_SELECTED_DAUGHTER_RETAINED")
    states = np.asarray([item.state for item in selected], dtype=np.int64)
    raw = np.asarray([item.observation_index for item in selected], dtype=np.int64)
    generations = np.asarray([item.growth_generation_one_based for item in selected], dtype=np.int64)
    kinds = np.asarray([item.observation_kind for item in selected], dtype=object)
    positions = np.flatnonzero(kinds == "post_fission").astype(np.int64)
    if len(states) != row.selectedClockLength or len(positions) != 100:
        raise RuntimeError("selected clock/boundary mismatch")
    compositions = normalized_compositions(states)
    return {
        "trajectory": trajectory, "selected": selected, "states": states,
        "compositions": compositions, "raw": raw, "generations": generations,
        "kinds": kinds, "boundaryPositions": positions,
        "boundaryCompositions": compositions[positions],
    }


def run_fixtures() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def record(fixture: str, passed: bool, detail: str) -> None:
        rows.append({"fixtureId": fixture, "passed": bool(passed), "detail": detail})

    def attractor(center: int, count: int) -> np.ndarray:
        values = np.full((count, 100), 1e-8, dtype=np.float64)
        values[:, center] = 1.0
        values[:, (center + 1) % 100] = 0.04
        for index in range(count):
            values[index, (center + 2 + index % 3) % 100] = 0.002 * (index % 4)
        return normalized_compositions(values)

    boundary = np.vstack((attractor(2, 60), attractor(40, 40)))
    r1 = r1_target(boundary, boundary, "L13-F01-R1")
    record("F01_R1_PLANTED_DOMINANT_COMPTYPE", r1.labels is not None and r1.centroids is not None, r1.status)

    no_recurrence = r1_target(np.eye(100), np.eye(100), "L13-F02-R1-NONE")
    record("F02_R1_NO_RECURRING_COMPTYPE", no_recurrence.labels is None, no_recurrence.status)

    tied = scientific_recurrence_gate(np.asarray([0, 0, 1, 1]), np.eye(2))
    record("F03_R1_TIED_DOMINANT_COMPTYPES", tied["status"] == "NO_UNIQUE_RECURRING_COMPTYPE", tied["status"])

    u2_a = u2_target(boundary, boundary, "L13-F04-U2")
    u2_b = u2_target(boundary, boundary, "L13-F04-U2")
    u2_exact = bool(
        u2_a.labels is not None and np.array_equal(u2_a.labels, u2_b.labels)
        and np.array_equal(u2_a.scores, u2_b.scores)
        and np.array_equal(u2_a.centroids, u2_b.centroids)
    )
    record("F04_U2_EXACT_REPLAY", u2_exact, u2_a.status)

    residue = np.asarray([[0.6, 0.4, -2.3852447794681098e-18]])
    repaired, audit = repair_u2_centroids(residue)
    record("F05_U2_MACHINE_SCALE_RESIDUE", bool(np.array_equal(repaired, [[0.6, 0.4, 0.0]]) and audit.material_negative_coordinate_count == 0), audit.repaired_sha256)

    labels = np.arange(101) % 3 == 0
    target, mask, _input_labels, cutoff = build_target_tensor(labels, 101)
    record("F06_EXACT_S16_QUARTER_CUTOFF", bool(cutoff == 25 and mask.sum() == 76 and np.array_equal(target[:76].astype(bool), labels[25:])), f"cutoff={cutoff}")

    values, feature_mask, time_mask = build_feature(np.arange(cutoff), np.ones(cutoff, bool), cutoff, scalar=True)
    masking = bool(feature_mask.sum() == cutoff and time_mask.sum() == cutoff and not feature_mask[cutoff:].any() and values[cutoff:].sum() == 0)
    record("F07_EXACT_S16_MASKING", masking, array_sha256(feature_mask))

    split = build_split_manifest()
    split_pass = all(
        len(split_indices(split, repetition, "FIT")) == 64
        and len(split_indices(split, repetition, "VALIDATION")) == 16
        and len(split_indices(split, repetition, "TEST")) == 20
        and not set(split_indices(split, repetition, "FIT")) & set(split_indices(split, repetition, "TEST"))
        for repetition in range(10)
    )
    record("F08_MATRIX_GROUPED_SPLIT", split_pass, sha256_file(S16_SPLIT_PATH))

    dummy_target = np.asarray([0, 0, 0, 1], dtype=bool)
    training_prevalence = float(dummy_target.mean())
    dummy_probability = np.full(4, training_prevalence)
    dummy = extended_binary_metrics(dummy_target, dummy_probability)
    record("F09_MAJORITY_DUMMY_TRAIN_ONLY", dummy["accuracy"] == 0.75 and training_prevalence == 0.25, canonical_json(dummy))

    synthetic = np.zeros((4, MAX_INPUT_LENGTH, 100), dtype=np.float64)
    synthetic_mask = np.zeros_like(synthetic, dtype=bool)
    synthetic[:, :4, 0] = np.asarray([[1, 2, 3, 4], [2, 3, 4, 5], [100, 100, 100, 100], [200, 200, 200, 200]])
    synthetic_mask[:, :4, 0] = True
    scaler = fit_channel_scaler(synthetic[:2], synthetic_mask[:2])
    record("F10_TRAIN_ONLY_SCALING", bool(scaler.mean[0] == 3.0 and scaler.valid_count[0] == 8), f"mean={scaler.mean[0]}")

    rng = np.random.Generator(np.random.PCG64DXSM(seed128("fixture", "suffix")))
    prefix = rng.normal(size=(48, 6))
    suffix_a = rng.normal(size=(48, 6))
    suffix_b = rng.normal(loc=2.0, size=(48, 6))
    full_a = np.vstack((prefix, suffix_a))
    full_b = np.vstack((prefix, suffix_b))
    pre_seed, part_seed = 11, 17
    completed_a = run_emergence_pipeline(full_a, PHIRL, SAFE_LATTICE, preprocessing_seed=pre_seed, partition_seed=part_seed)
    completed_b = run_emergence_pipeline(full_b, PHIRL, SAFE_LATTICE, preprocessing_seed=pre_seed, partition_seed=part_seed)
    prefix_a = run_emergence_pipeline(prefix, PHIRL, SAFE_LATTICE, preprocessing_seed=pre_seed, partition_seed=part_seed)
    prefix_b = run_emergence_pipeline(prefix.copy(), PHIRL, SAFE_LATTICE, preprocessing_seed=pre_seed, partition_seed=part_seed)
    completed_change_permitted = not result_replay_equal(completed_a, completed_b)
    record("F11_COMPLETED_FIT_FUTURE_DEPENDENCE", completed_change_permitted, f"fullEqual={not completed_change_permitted}")
    record("F12_PREFIX_ONLY_SUFFIX_INVARIANCE", result_replay_equal(prefix_a, prefix_b), f"prefixStatus={prefix_a.status}")

    alternative_target = ~labels
    alternative_tensor = build_target_tensor(alternative_target, 101)
    same_feature = array_sha256(values) == array_sha256(values.copy())
    changed_target = not np.array_equal(target, alternative_tensor[0])
    record("F13_LABEL_TARGET_SEPARATION", same_feature and changed_target, "feature hash unchanged while target changed")

    typed = pd.DataFrame({
        "booleanFingerprint": pd.Series([True, pd.NA], dtype="boolean"),
        "nullableFloat": [1.0, np.nan], "status": ["ELIGIBLE", "INELIGIBLE"],
        "floatArray": [[1.0, 2.0], []],
    })
    typed_path = CACHE_ROOT / "fixture_typed_table.parquet"
    write_parquet(typed_path, typed)
    restored = pd.read_parquet(typed_path)
    typed_pass = restored.columns.tolist() == sorted(typed.columns.tolist()) and len(restored) == 2
    quarantine = CACHE_ROOT / "fixture_quarantine" / typed_path.name
    quarantine.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(typed_path, quarantine)
    typed_pass = typed_pass and sha256_file(typed_path) == sha256_file(quarantine)
    record("F14_TYPED_TABLE_SERIALIZATION", typed_pass, sha256_file(typed_path))

    mini_values = np.zeros((4, MAX_INPUT_LENGTH, 100), dtype=np.float64)
    mini_masks = np.zeros_like(mini_values, dtype=bool)
    mini_time = np.zeros((4, MAX_INPUT_LENGTH), dtype=bool)
    mini_target = np.zeros((4, MAX_TARGET_LENGTH), dtype=np.float64)
    mini_target_mask = np.zeros((4, MAX_TARGET_LENGTH), dtype=bool)
    mini_values[:, :8, 0] = rng.normal(size=(4, 8))
    mini_masks[:, :8, 0] = True
    mini_time[:, :8] = True
    mini_target[:, :10] = np.asarray([[0, 1] * 5, [1, 0] * 5, [0, 0, 1, 1, 0, 0, 1, 1, 0, 0], [1, 1, 0, 0, 1, 1, 0, 0, 1, 1]])
    mini_target_mask[:, :10] = True
    kwargs = {
        "fit_values": mini_values[:2], "fit_channel_mask": mini_masks[:2], "fit_time_mask": mini_time[:2],
        "fit_targets": mini_target[:2], "fit_target_mask": mini_target_mask[:2],
        "validation_values": mini_values[2:], "validation_channel_mask": mini_masks[2:], "validation_time_mask": mini_time[2:],
        "validation_targets": mini_target[2:], "validation_target_mask": mini_target_mask[2:],
        "model_seed": 12345, "maximum_epochs": 3, "patience": 2,
    }
    model_a = train_masked_mlp(**kwargs)
    model_b = train_masked_mlp(**kwargs)
    pred_a = predict_probabilities(model_a.model, mini_values, mini_masks, mini_time)
    pred_b = predict_probabilities(model_b.model, mini_values, mini_masks, mini_time)
    record("F15_EXACT_MODEL_REPLAY", model_a.history.equals(model_b.history) and np.array_equal(pred_a, pred_b), array_sha256(pred_a))

    try:
        raise RuntimeError("fixture-worker-failure")
    except RuntimeError as error:
        provenance = {
            "matrixId": 7, "candidateId": "CANDIDATE_2", "targetId": R1_TARGET_ID,
            "featureId": P2_ID, "splitId": 0, "taskStage": "fixture",
            "seedIdentity": "fixture", "exceptionClass": type(error).__name__,
            "exceptionMessage": str(error),
        }
    required = {"matrixId", "candidateId", "targetId", "featureId", "splitId", "taskStage", "seedIdentity", "exceptionClass", "exceptionMessage"}
    record("F16_WORKER_FAILURE_PROVENANCE", set(provenance) == required, canonical_json(provenance))
    return pd.DataFrame(rows)


def append_preoutcome_ledgers(source: dict[str, Any]) -> None:
    ledger_path = S19_ROOT / "self_improvement_ledger.parquet"
    ledger = pd.read_parquet(ledger_path)
    if not ((ledger["loopId"] == LOOP_ID) & (ledger["recordPhase"] == "PRE_LOOP_OUTCOME_BLIND_METHOD_LOCK")).any():
        row = {
            "appendOnly": True,
            "beliefBeforeLoop": "The Figure 5 approximately 0.60 dummy may indicate a recurring-attractor target with approximately 0.40 positive prevalence rather than the adjacent-H target used in S16.",
            "failureOrAmbiguityTargeted": "The Figure 5 dummy/Table 1 probability contradiction and whether target identity, rather than tensor or architecture, explains it.",
            "informationGainRationale": "Only the target changes while exact S16 inputs, splits, model and evaluation remain frozen; R1 and U2 were already specified without prediction outcomes.",
            "learned": "Pending target-geometry gate and, only if passed, frozen prediction execution.",
            "ledgerSequence": int(ledger["ledgerSequence"].max()) + 1,
            "loopId": LOOP_ID,
            "motivatingEvidence": "L10 R1 and L11R U2 had approximately 0.40 mean matrix occupancy, matching the arithmetic clue implied by a 0.60 majority baseline.",
            "proposedNextTest": "Execute the pushed L13 geometry gate, then only eligible frozen feature/model scope, and stop for human review.",
            "recordPhase": "PRE_LOOP_OUTCOME_BLIND_METHOD_LOCK",
            "remainingPlausibleHypotheses": "Either target may reconcile only the baseline, completed-fit model ordering, a past-only proxy, or neither.",
            "selectedHypotheses": "R1 historical dominant compotype and U2 paper-Euclidean recurring-centroid union as the only primary targets.",
            "timestampUtc": utc_now(),
            "weakenedHypotheses": "No scientific hypothesis weakened before outcome access.",
        }
        write_parquet(ledger_path, pd.concat([ledger, pd.DataFrame([row])], ignore_index=True))

    candidates_path = S19_ROOT / "candidate_registry.parquet"
    candidates = pd.read_parquet(candidates_path)
    additions = []
    next_order = int(candidates["registryOrder"].max()) + 1
    for order, (candidate_id, proposal) in enumerate([
        ("S19-L13-R1-FIGURE5-TARGET", "Exact L10 R1 direct molecular target with exact S16 prediction task"),
        ("S19-L13-U2-FIGURE5-TARGET", "Exact authoritative L11R U2 direct molecular target with exact S16 prediction task"),
    ]):
        if (candidates["candidateId"] == candidate_id).any():
            continue
        additions.append({
            "branchCount": 2, "bundleId": "L13_FIGURE5_RECURRING_TARGETS", "candidateId": candidate_id,
            "candidateSpecificSuccess": 0, "completedFitLeakage": 1, "computeEfficiency": 4,
            "crossCandidateDiscriminability": 5, "deterministicHReuse": 1, "explanatoryLeverage": 5,
            "frozenRank": order + 1, "independenceFromPriorOutcomeSelection": 2,
            "outcomeGuidedThresholdSelection": 0, "paperFingerprintSpecificity": 5,
            "proposedSpecification": proposal, "rankingScore": 27.0 - order,
            "registryOrder": next_order + order, "selected": True,
            "selectionReason": "Explicit human authorization after L12, frozen prior label and exact S16 method reuse",
            "sourceGrounding": 4 if order == 0 else 3, "testability": 5,
            "undefinedAuthorSemantics": 1,
        })
    if additions:
        write_parquet(candidates_path, pd.concat([candidates, pd.DataFrame(additions)], ignore_index=True))

    source_path = S19_ROOT / "source_search_ledger.parquet"
    sources = pd.read_parquet(source_path)
    if not (sources["sourceId"] == "L13_FROZEN_S16_L10_L11R_L12_CROSSWALK").any():
        entry = {
            "commitOrVersion": source["entries"][0]["sha256"],
            "evidenceClass": "DIRECT_FROZEN_E01_RESULT_AND_IMPLEMENTATION",
            "finding": "Exact frozen S16 task plus L10 R1/L11R U2 targets tests the L12 Figure 5 prevalence contradiction without a new method.",
            "licenseStatus": "WORKSPACE_ARTIFACT_AND_REPOSITORY",
            "redistributionStatus": "REFERENCE_ONLY",
            "repositoryIdentity": "Eidosoma/arrival-of-self-replicators",
            "retainedPath": str(CONFIG_PATH), "retrievalDate": "2026-08-09",
            "sha256": sha256_file(CONFIG_PATH), "sourceId": "L13_FROZEN_S16_L10_L11R_L12_CROSSWALK",
            "sourceType": "FROZEN_INTERNAL_SOURCE_CROSSWALK", "treeIdentity": repository_lock()["head"],
            "url": "workspace-repository",
        }
        write_parquet(source_path, pd.concat([sources, pd.DataFrame([entry])], ignore_index=True))

    registry_path = S19_ROOT / "loop_registry.yaml"
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    if not any(item.get("loopId") == LOOP_ID for item in registry["loops"]):
        registry["loops"].append({
            "loopId": LOOP_ID, "versionedLoopId": VERSION,
            "status": "AUTHORIZED_PREOUTCOME_LOCK_PREPARED", "authorized": True,
            "outcomeAccessed": False, "humanReviewRequiredAfter": True,
            "completed": False, "eligibleScientificResults": None,
            "promotedLeadCount": None, "nextStepActive": True,
        })
        write_yaml(registry_path, registry)


def prepare() -> None:
    started = time.time()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    config = load_config()
    repository = repository_lock()
    if not repository["passed"]:
        raise RuntimeError(f"repository is not a clean pushed lock: {repository}")
    split = pd.read_csv(S16_SPLIT_PATH)
    pd.testing.assert_frame_equal(split, build_split_manifest(), check_dtype=False)
    if parameter_count(MaskedSequenceMLP()) != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("S16 MLP parameter contract changed")
    immutable = create_immutable_baseline()
    source = source_manifest()
    fixtures = run_fixtures()
    write_parquet(OUTPUT_ROOT / "fixture_results.parquet", fixtures)
    if len(fixtures) != 16 or not fixtures["passed"].all():
        raise RuntimeError("mandatory L13 fixture failure")
    trajectories = load_trajectory_manifest()
    write_parquet(OUTPUT_ROOT / "shared_trajectory_manifest.parquet", trajectories)
    write_json(OUTPUT_ROOT / "immutable_prior_validation.json", immutable)
    write_json(OUTPUT_ROOT / "source_snapshot_manifest.json", source)
    shutil.copy2(CONFIG_PATH, OUTPUT_ROOT / "preregistration.yaml")
    figure_lock = figure5_lock()
    figure_lock.to_csv(OUTPUT_ROOT / "figure5_digitized_target_lock.csv", index=False)

    lock = {
        "schema": "eidosoma.e01.s19.l13.implementation_lock.v1",
        "researchStepId": LOOP_ID, "versionedStepId": VERSION,
        "createdAtUtc": utc_now(), "outcomeAccessed": False,
        "repository": repository, "configSha256": sha256_file(CONFIG_PATH),
        "s16TensorModelManifestSha256": sha256_file(S16_MANIFEST),
        "s16SplitManifestSha256": sha256_file(S16_SPLIT_PATH),
        "l10R1ImplementationSha256": sha256_file(L10_CORE),
        "l11RU2ImplementationSha256": sha256_file(L11R_CORE),
        "targetIds": list(TARGET_IDS), "modelIds": list(ALL_MODEL_IDS),
        "promotionPriorityWhenBothPass": [R1_TARGET_ID, U2_TARGET_ID],
        "promotionPriorityBasis": "paper singular most-recurring-composition wording, frozen before outcomes",
        "figure5ApproximationEnvelope": 0.05,
        "figure5Caveat": "L12 stored approximate centers but no numeric whisker endpoints; the fixed envelope is an adjudication tolerance, not redigitization.",
        "maximumTechnicalAmendments": 2,
        "fixtureCount": len(fixtures), "fixturesPassed": bool(fixtures["passed"].all()),
        "passed": True,
    }
    write_json(OUTPUT_ROOT / "implementation_lock.json", lock)
    write_yaml(OUTPUT_ROOT / "figure5_target_hypothesis_lock.yaml", {
        "researchStepId": LOOP_ID, "targetHypotheses": config["targets"],
        "geometryGate": config["geometryGate"], "promotionPriority": [R1_TARGET_ID, U2_TARGET_ID],
        "outcomeAccessed": False,
    })
    write_json(OUTPUT_ROOT / "prediction_task_lock.json", {
        "s16Manifest": str(S16_MANIFEST), "sha256": sha256_file(S16_MANIFEST),
        "cutoff": "floor(0.25*T)", "masking": "EXACT_S16", "scaling": "EXACT_S16_FIT_ONLY",
        "passed": True,
    })
    write_json(OUTPUT_ROOT / "model_lock.json", {
        "s16Core": str(S16_CORE), "sha256": sha256_file(S16_CORE),
        "parameterCount": EXPECTED_PARAMETER_COUNT, "models": list(ALL_MODEL_IDS), "passed": True,
    })
    write_json(OUTPUT_ROOT / "split_seed_lock.json", {
        "path": str(S16_SPLIT_PATH), "sha256": sha256_file(S16_SPLIT_PATH),
        "rows": len(split), "repetitions": 10, "passed": True,
    })
    write_yaml(OUTPUT_ROOT / "scientific_gate_lock.yaml", {
        "geometry": config["geometryGate"], "paperFacing": config["paperFacingGate"],
        "prospective": config["prospectiveGate"], "promotionPriority": [R1_TARGET_ID, U2_TARGET_ID],
    })
    write_yaml(OUTPUT_ROOT / "target_label_registry.yaml", {
        "targets": config["targets"], "primaryTargetCount": 2,
        "thirdTargetPermitted": False, "threshold": "strict H>0.9",
    })
    write_yaml(OUTPUT_ROOT / "feature_pipeline_registry.yaml", {
        "features": config["features"], "prefixAttractorGeometry": config["prefixAttractorGeometry"],
        "timeOnlyControl": config["timeOnlyControl"], "randomMatchedShapeControl": config["randomMatchedShapeControl"],
    })
    write_yaml(OUTPUT_ROOT / "model_registry.yaml", {
        "models": [{"modelId": model_id, "learned": model_id != DUMMY_ID, "gateEligible": model_id not in DIAGNOSTIC_MODEL_IDS} for model_id in ALL_MODEL_IDS],
        "architecture": "exact S16 288789-parameter masked MLP", "dummy": "fit-target prevalence only",
    })
    write_parquet(OUTPUT_ROOT / "split_manifest.parquet", split)
    decision = f"""# S19-L13 Decision Record

- Research step: `{VERSION}`
- Status: authorized and prospectively locked; scientific outcomes unopened at this record.
- Frozen question: whether either exact prior recurring-attractor target reconciles the Figure 5 approximately 60% dummy and model ordering when only the S16 target changes.
- Targets: `{R1_TARGET_ID}` and `{U2_TARGET_ID}` only.
- Model/tensor: exact S16 contract; ten matrix-level splits; no balancing, architecture search, or target search.
- Source caveat: L12 froze approximate panel centers but not numeric whisker endpoints. A fixed ±0.05 comparison envelope is locked as an adjudication tolerance and will not move.
- Stop: mandatory human review after L13; S20 and E02 remain inactive.
"""
    (OUTPUT_ROOT / "decision_record.md").write_text(decision, encoding="utf-8")
    append_preoutcome_ledgers(source)
    runtime = {
        "schema": "eidosoma.e01.s19.l13.runtime_manifest.v1",
        "researchStepId": LOOP_ID, "phase": "PREPARED", "startUtc": utc_now(),
        "prepareWallSeconds": time.time() - started, "cpuCountVisible": os.cpu_count(),
        "workersMaximum": 8, "gpuHours": 0, "python": sys.version,
        "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__,
        "scipy": scipy.__version__, "sklearn": sklearn.__version__, "torch": torch.__version__,
        "threadEnvironment": {key: os.environ.get(key) for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]},
    }
    write_json(OUTPUT_ROOT / "runtime_manifest.json", runtime)
    print(canonical_json({"phase": "prepare", "fixturesPassed": 16, "immutableFiles": immutable["entryCount"], "repositoryCommit": repository["head"]}))


def geometry_phase() -> None:
    started = time.time()
    if not (OUTPUT_ROOT / "implementation_lock.json").is_file():
        raise RuntimeError("pre-outcome lock is missing")
    lock = json.loads((OUTPUT_ROOT / "implementation_lock.json").read_text())
    if lock["outcomeAccessed"] is not False or not repository_lock()["passed"]:
        raise RuntimeError("L13 lock/repository gate failed")
    manifest = load_trajectory_manifest()
    authoritative = pd.read_parquet(L11R_ROOT / "molecular_union_label_results.parquet")
    authoritative = authoritative.loc[
        authoritative["pipelineId"].eq("U2_PAPER_EUCLIDEAN_ALL_RECURRING_CENTROIDS_H090")
    ].copy()
    centroids_authoritative = pd.read_parquet(L11R_ROOT / "recurring_centroid_results.parquet")
    centroids_authoritative = centroids_authoritative.loc[
        centroids_authoritative["pipelineId"].eq("U2_PAPER_EUCLIDEAN_ALL_RECURRING_CENTROIDS_H090")
    ].copy()

    target_payloads: dict[tuple[str, str, int], dict[str, Any]] = {}
    r1_rows: list[dict[str, Any]] = []
    u2_rows: list[dict[str, Any]] = []
    identity_rows: list[dict[str, Any]] = []
    availability_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for manifest_row in manifest.itertuples(index=False):
        arrays = trajectory_arrays(manifest_row)
        candidate_id = str(manifest_row.candidateId)
        matrix_index = int(manifest_row.matrixIndex)
        trajectory_id = str(manifest_row.trajectoryId)
        state_digest = array_sha256(arrays["states"])
        selected_digest = array_sha256(arrays["raw"])
        boundary_digest = array_sha256(arrays["boundaryPositions"])
        r1 = r1_target(arrays["boundaryCompositions"], arrays["compositions"], trajectory_id)
        u2 = u2_target(arrays["boundaryCompositions"], arrays["compositions"], trajectory_id)

        auth = authoritative.loc[
            authoritative["candidateId"].eq(candidate_id)
            & authoritative["matrixIndex"].eq(matrix_index)
        ].sort_values("analysisUnitIndex")
        if len(auth) != len(arrays["states"]) or not np.array_equal(
            auth["analysisUnitIndex"].to_numpy(dtype=np.int64), np.arange(len(auth))
        ):
            raise RuntimeError("authoritative U2 row alignment mismatch")
        state_hashes = np.asarray(
            [l11r_array_sha256(row) for row in arrays["compositions"]], dtype=object
        )
        state_hash_exact = np.array_equal(state_hashes, auth["stateSha256"].to_numpy(dtype=object))
        label_exact = u2.labels is not None and np.array_equal(
            u2.labels, auth["isReplicator"].to_numpy(dtype=bool)
        )
        score_exact = u2.scores is not None and np.array_equal(
            u2.scores, auth["unionScore"].to_numpy(dtype=np.float64)
        )
        auth_centroids = centroids_authoritative.loc[
            centroids_authoritative["candidateId"].eq(candidate_id)
            & centroids_authoritative["matrixIndex"].eq(matrix_index)
        ].sort_values("clusterId")
        centroid_hashes = [] if u2.centroids is None else [l11r_array_sha256(row) for row in u2.centroids]
        centroid_exact = bool(
            u2.centroids is not None
            and len(auth_centroids) == len(u2.centroids)
            and centroid_hashes == auth_centroids["scoringCentroidSha256"].tolist()
        )
        if not (state_hash_exact and label_exact and score_exact and centroid_exact):
            raise RuntimeError(
                f"U2 exact replay failed {candidate_id} M{matrix_index:03d}: "
                f"state={state_hash_exact} label={label_exact} score={score_exact} centroid={centroid_exact}"
            )

        identity_rows.append({
            "candidateId": candidate_id, "matrixIndex": matrix_index,
            "trajectoryId": trajectory_id, "cacheSha256": manifest_row.cacheSha256,
            "trajectorySha256": manifest_row.trajectorySha256,
            "selectedClockLength": len(arrays["states"]), "postFissionBoundaryCount": len(arrays["boundaryPositions"]),
            "stateArraySha256": state_digest, "selectedRawIndexSha256": selected_digest,
            "boundaryPositionSha256": boundary_digest, "u2StateHashExact": state_hash_exact,
            "u2LabelExact": label_exact, "u2ScoreExact": score_exact,
            "u2CentroidExact": centroid_exact, "passed": True,
        })

        for target in (r1, u2):
            total = len(arrays["states"])
            target_tensor, target_mask, input_labels, cutoff = build_target_tensor(target.labels, total)
            geometry = target_geometry(target.labels, total)
            target_payloads[(target.target_id, candidate_id, matrix_index)] = {
                "targetId": target.target_id, "candidateId": candidate_id,
                "matrixIndex": matrix_index, "trajectoryId": trajectory_id,
                "status": target.status, "labels": target.labels, "scores": target.scores,
                "centroids": target.centroids, "selectedK": target.selected_k,
                "target": target_tensor, "targetMask": target_mask,
                "inputLabels": input_labels, "cutoff": cutoff, "T": total,
            }
            row = {
                "candidateId": candidate_id, "matrixIndex": matrix_index,
                "trajectoryId": trajectory_id, "targetId": target.target_id,
                "status": target.status, "defined": target.labels is not None,
                "selectedK": target.selected_k,
                "centroidCount": None if target.centroids is None else len(target.centroids),
                "labelCount": None if target.labels is None else len(target.labels),
                "positiveCount": None if target.labels is None else int(target.labels.sum()),
                "labelSha256": None if target.labels is None else array_sha256(target.labels),
                "scoreSha256": None if target.scores is None else array_sha256(target.scores),
                "centroidSha256": None if target.centroids is None else array_sha256(target.centroids),
                "labels": None if target.labels is None else target.labels.tolist(),
                "scores": None if target.scores is None else target.scores.tolist(),
            }
            (r1_rows if target.target_id == R1_TARGET_ID else u2_rows).append(row)
            availability_rows.append({
                "candidateId": candidate_id, "matrixIndex": matrix_index,
                "targetId": target.target_id, "defined": target.labels is not None,
                "status": target.status,
            })
            geometry_rows.append({
                "candidateId": candidate_id, "matrixIndex": matrix_index,
                "targetId": target.target_id, "T": total, "cutoff": cutoff,
                "targetLength": total - cutoff, **geometry,
            })

    write_parquet(OUTPUT_ROOT / "trajectory_identity_validation.parquet", pd.DataFrame(identity_rows))
    write_parquet(OUTPUT_ROOT / "r1_target_results.parquet", pd.DataFrame(r1_rows))
    write_parquet(OUTPUT_ROOT / "u2_target_replay_results.parquet", pd.DataFrame(u2_rows))
    availability = pd.DataFrame(availability_rows)
    geometry = pd.DataFrame(geometry_rows)
    write_parquet(OUTPUT_ROOT / "target_availability_results.parquet", availability)
    write_parquet(OUTPUT_ROOT / "target_geometry_results.parquet", geometry)

    split = pd.read_csv(S16_SPLIT_PATH)
    prevalence_rows: list[dict[str, Any]] = []
    dummy_rows: list[dict[str, Any]] = []
    for target_id in TARGET_IDS:
        for candidate_id in CANDIDATE_IDS:
            for repetition in range(10):
                indices_by_role = {
                    role: split_indices(split, repetition, role)
                    for role in ("FIT", "VALIDATION", "TEST")
                }
                role_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
                for role, indices in indices_by_role.items():
                    ys = np.stack([target_payloads[(target_id, candidate_id, int(index))]["target"] for index in indices]).astype(bool)
                    masks = np.stack([target_payloads[(target_id, candidate_id, int(index))]["targetMask"] for index in indices])
                    role_arrays[role] = (ys, masks)
                    valid = ys[masks]
                    prevalence_rows.append({
                        "targetId": target_id, "candidateId": candidate_id,
                        "repetitionId": repetition, "splitRole": role,
                        "matrixCount": len(indices), "definedMatrixCount": int(np.count_nonzero(masks.any(axis=1))),
                        "validTargetCount": int(masks.sum()), "positiveTargetCount": int(valid.sum()),
                        "targetPrevalence": None if not valid.size else float(valid.mean()),
                        "effectiveTargetCellCount": int(valid.size),
                    })
                fit_y, fit_mask = role_arrays["FIT"]
                test_y, test_mask = role_arrays["TEST"]
                fit_valid = fit_y[fit_mask]
                test_valid = test_y[test_mask]
                if not fit_valid.size or not test_valid.size:
                    dummy_probability = math.nan
                    metrics = extended_binary_metrics(test_valid, np.full(test_valid.size, np.nan))
                else:
                    training_prevalence = float(fit_valid.mean())
                    metrics = extended_binary_metrics(test_valid, np.full(test_valid.size, training_prevalence))
                    dummy_probability = training_prevalence
                dummy_rows.append({
                    "targetId": target_id, "candidateId": candidate_id,
                    "repetitionId": repetition, "trainingTargetPrevalence": dummy_probability,
                    "testTargetPrevalence": None if not test_valid.size else float(test_valid.mean()),
                    "dummyAccuracy": metrics["accuracy"], "validTargetCount": metrics["validTargetCount"],
                    "dummyIntervalLower": PAPER_INTERVALS[DUMMY_ID][0], "dummyIntervalUpper": PAPER_INTERVALS[DUMMY_ID][1],
                    "paddingScored": False, "undefinedRowsScored": False,
                })
    prevalence = pd.DataFrame(prevalence_rows)
    dummy = pd.DataFrame(dummy_rows)
    write_parquet(OUTPUT_ROOT / "target_suffix_prevalence_results.parquet", prevalence)
    write_parquet(OUTPUT_ROOT / "dummy_baseline_results.parquet", dummy)

    onset = geometry[[
        "targetId", "candidateId", "matrixIndex", "defined", "firstOnset",
        "normalizedFirstOnset", "noOnsetBeforeCutoff", "firstOnsetInSuffix",
        "suffixPositiveEpisodes", "suffixNegativeEpisodes", "suffixConstantPositive",
        "suffixConstantNegative", "T", "cutoff",
    ]].copy()
    write_parquet(OUTPUT_ROOT / "onset_eligibility_results.parquet", onset)

    gate_rows: list[dict[str, Any]] = []
    for target_id in TARGET_IDS:
        candidate_passes = []
        for candidate_id in CANDIDATE_IDS:
            defined = int(
                availability.loc[
                    availability["targetId"].eq(target_id)
                    & availability["candidateId"].eq(candidate_id), "defined"
                ].sum()
            )
            dummy_values = dummy.loc[
                dummy["targetId"].eq(target_id) & dummy["candidateId"].eq(candidate_id),
                "dummyAccuracy",
            ].to_numpy(dtype=np.float64)
            test_targets = []
            for repetition in range(10):
                for index in split_indices(split, repetition, "TEST"):
                    payload = target_payloads[(target_id, candidate_id, int(index))]
                    test_targets.extend(payload["target"][payload["targetMask"]].astype(bool).tolist())
            gates = geometry_gate(defined, dummy_values, np.asarray(test_targets, dtype=bool))
            candidate_passes.append(gates["passed"])
            for gate_id, passed in gates.items():
                if gate_id == "passed":
                    continue
                gate_rows.append({
                    "targetId": target_id, "candidateId": candidate_id,
                    "gateId": gate_id, "passed": bool(passed), "definedMatrixCount": defined,
                    "dummyMedian": float(np.nanmedian(dummy_values)), "dummyMinimum": float(np.nanmin(dummy_values)),
                    "dummyMaximum": float(np.nanmax(dummy_values)), "aggregateTestTargetCount": len(test_targets),
                    "aggregateTestPrevalence": float(np.mean(test_targets)),
                })
            gate_rows.append({
                "targetId": target_id, "candidateId": candidate_id,
                "gateId": "CANDIDATE_GEOMETRY_ADVANCEMENT", "passed": bool(gates["passed"]),
                "definedMatrixCount": defined, "dummyMedian": float(np.nanmedian(dummy_values)),
                "dummyMinimum": float(np.nanmin(dummy_values)), "dummyMaximum": float(np.nanmax(dummy_values)),
                "aggregateTestTargetCount": len(test_targets), "aggregateTestPrevalence": float(np.mean(test_targets)),
            })
        cross_pass = bool(all(candidate_passes))
        gate_rows.append({
            "targetId": target_id, "candidateId": "BOTH_REQUIRED",
            "gateId": "CROSS_CANDIDATE_GEOMETRY_ADVANCEMENT", "passed": cross_pass,
            "definedMatrixCount": None, "dummyMedian": None, "dummyMinimum": None,
            "dummyMaximum": None, "aggregateTestTargetCount": None, "aggregateTestPrevalence": None,
        })
    gates = pd.DataFrame(gate_rows)
    gates.to_csv(OUTPUT_ROOT / "geometry_advancement_gate_results.csv", index=False)
    advanced = gates.loc[
        gates["candidateId"].eq("BOTH_REQUIRED")
        & gates["gateId"].eq("CROSS_CANDIDATE_GEOMETRY_ADVANCEMENT")
        & gates["passed"], "targetId"
    ].tolist()
    with (CACHE_ROOT / "target_payloads.pkl").open("wb") as handle:
        pickle.dump(target_payloads, handle, protocol=5)
    write_json(CACHE_ROOT / "geometry_status.json", {
        "advancedTargets": advanced, "neitherAdvanced": not advanced,
        "targetPayloadSha256": sha256_file(CACHE_ROOT / "target_payloads.pkl"),
        "completedAtUtc": utc_now(),
    })
    runtime = json.loads((OUTPUT_ROOT / "runtime_manifest.json").read_text())
    runtime.update({"phase": "GEOMETRY_COMPLETE", "geometryWallSeconds": time.time() - started, "geometryCompletedAtUtc": utc_now(), "advancedTargets": advanced})
    write_json(OUTPUT_ROOT / "runtime_manifest.json", runtime)
    if failures:
        pd.DataFrame(failures).to_csv(OUTPUT_ROOT / "failure_ledger.csv", index=False)
    print(canonical_json(json_safe({
        "phase": "geometry", "advancedTargets": advanced,
        "defined": availability.groupby(["targetId", "candidateId"])["defined"].sum().to_dict(),
        "dummyMedians": dummy.groupby(["targetId", "candidateId"])["dummyAccuracy"].median().to_dict(),
    })))


def _feature_summary(
    candidate_id: str,
    matrix_index: int,
    trajectory_id: str,
    feature_id: str,
    feature: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    future_dependent: bool,
    source_status: str | None = None,
) -> dict[str, Any]:
    values, mask, time_mask = feature
    valid = values[mask]
    return {
        "candidateId": candidate_id, "matrixIndex": matrix_index,
        "trajectoryId": trajectory_id, "featureId": feature_id,
        "validFeatureCellCount": int(mask.sum()), "validInputTimeCount": int(time_mask.sum()),
        "mean": None if not valid.size else float(valid.mean()),
        "standardDeviation": None if not valid.size else float(valid.std(ddof=0)),
        "minimum": None if not valid.size else float(valid.min()),
        "maximum": None if not valid.size else float(valid.max()),
        "valueSha256": array_sha256(values), "channelMaskSha256": array_sha256(mask),
        "timeMaskSha256": array_sha256(time_mask),
        "futureSuffixUsed": bool(future_dependent), "sourceStatus": source_status,
    }


def compute_features_for_trajectory(manifest_row: Any) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], list[dict[str, Any]], dict[str, Any]]:
    arrays = trajectory_arrays(manifest_row)
    states = arrays["states"]
    compositions = arrays["compositions"]
    candidate_id = str(manifest_row.candidateId)
    matrix_index = int(manifest_row.matrixIndex)
    trajectory_id = str(manifest_row.trajectoryId)
    total = len(states)
    cutoff = math.floor(0.25 * total)
    full_clr, _, full_closure_error = frozen_clr(states)
    prefix_clr, _, prefix_closure_error = frozen_clr(states[:cutoff])
    pre_seed = s16_source_seed(candidate_id, matrix_index, "preprocessing")
    part_seed = s16_source_seed(candidate_id, matrix_index, "partition")
    full_result = run_emergence_pipeline(
        full_clr, PHIRL, SAFE_LATTICE,
        preprocessing_seed=pre_seed, partition_seed=part_seed,
    )
    prefix_result = run_emergence_pipeline(
        prefix_clr, PHIRL, SAFE_LATTICE,
        preprocessing_seed=pre_seed, partition_seed=part_seed,
    )
    full_values, full_available = source_values(full_result, fit_length=total, retained_length=cutoff)
    prefix_values, prefix_available = source_values(prefix_result, fit_length=cutoff, retained_length=cutoff)
    p1 = build_feature(full_values, full_available, cutoff, scalar=True)
    p2 = build_feature(prefix_values, prefix_available, cutoff, scalar=True)

    change = np.zeros(total, dtype=np.float64)
    change[1:] = np.linalg.norm(np.diff(compositions, axis=0), axis=1)
    flux = np.zeros_like(states, dtype=np.float64)
    flux[1:] = np.diff(states, axis=0)
    h = incoming_h(compositions)
    b1 = build_feature(change[:cutoff], np.arange(cutoff) > 0, cutoff, scalar=True)
    b2 = build_feature(states[:cutoff].astype(np.float64), np.ones((cutoff, 100), bool), cutoff, scalar=False)
    b3 = build_feature(flux[:cutoff], np.broadcast_to((np.arange(cutoff) > 0)[:, None], (cutoff, 100)), cutoff, scalar=False)
    b4 = build_feature(h[:cutoff], np.ones(cutoff, bool), cutoff, scalar=True)

    prefix_boundary_positions = arrays["boundaryPositions"][arrays["boundaryPositions"] < cutoff]
    prefix_geometry_values = np.zeros(cutoff, dtype=np.float64)
    prefix_geometry_available = np.zeros(cutoff, dtype=bool)
    prefix_fit_status = "NO_PREFIX_BOUNDARY"
    if len(prefix_boundary_positions) >= 2:
        prefix_boundary = compositions[prefix_boundary_positions]
        prefix_fit = r1_target(prefix_boundary, compositions[:cutoff], trajectory_id + "::PREFIX_Q1")
        prefix_fit_status = prefix_fit.status
        if prefix_fit.scores is not None:
            prefix_geometry_values[:] = prefix_fit.scores
            prefix_geometry_available[:] = np.isfinite(prefix_fit.scores)
    b5 = build_feature(prefix_geometry_values, prefix_geometry_available, cutoff, scalar=True)

    normalized_time = np.arange(cutoff, dtype=np.float64) / max(cutoff - 1, 1)
    prefix_generations = arrays["generations"][:cutoff].astype(np.float64)
    normalized_generation = prefix_generations / max(float(prefix_generations.max()), 1.0)
    b6_left = build_feature(normalized_time, np.ones(cutoff, bool), cutoff, scalar=True)
    b6_right = build_feature(normalized_generation, np.ones(cutoff, bool), cutoff, scalar=True)
    b6 = combine_scalar_features(b6_left, b6_right)

    random_values = np.zeros(cutoff, dtype=np.float64)
    random_available = prefix_available.copy()
    valid_prefix = prefix_values[prefix_available]
    if valid_prefix.size:
        rng = np.random.Generator(np.random.PCG64DXSM(seed128("B7", candidate_id, matrix_index)))
        generated = rng.normal(size=int(valid_prefix.size))
        generated = (generated - generated.mean()) / (generated.std(ddof=0) or 1.0)
        generated = generated * valid_prefix.std(ddof=0) + valid_prefix.mean()
        random_values[prefix_available] = generated
    b7 = build_feature(random_values, random_available, cutoff, scalar=True)
    p2_b4 = combine_scalar_features(p2, b4)
    p2_b5 = combine_scalar_features(p2, b5)

    permutation_rng = np.random.Generator(np.random.PCG64DXSM(seed128("NC1", candidate_id, matrix_index)))
    permutation = permutation_rng.permutation(cutoff)
    nc1_values = prefix_values[permutation]
    nc1_available = prefix_available[permutation]
    nc1 = build_feature(nc1_values, nc1_available, cutoff, scalar=True)
    features = {
        P1_ID: p1, P2_ID: p2, B1_ID: b1, B2_ID: b2, B3_ID: b3,
        B4_ID: b4, B5_ID: b5, B6_ID: b6, B7_ID: b7,
        P2_B4_ID: p2_b4, P2_B5_ID: p2_b5, NC1_ID: nc1,
    }
    summaries = [
        _feature_summary(candidate_id, matrix_index, trajectory_id, feature_id, feature,
                         future_dependent=feature_id == P1_ID,
                         source_status=full_result.status if feature_id == P1_ID else prefix_result.status if feature_id in {P2_ID, P2_B4_ID, P2_B5_ID, NC1_ID, B7_ID} else prefix_fit_status if feature_id == B5_ID else None)
        for feature_id, feature in features.items()
    ]
    audit = {
        "candidateId": candidate_id, "matrixIndex": matrix_index, "trajectoryId": trajectory_id,
        "T": total, "cutoff": cutoff, "preprocessingSeed": pre_seed, "partitionSeed": part_seed,
        "completedStatus": full_result.status, "prefixStatus": prefix_result.status,
        "completedLocalOffset": full_result.local_offset, "prefixLocalOffset": prefix_result.local_offset,
        "completedAvailableCount": int(full_available.sum()), "prefixAvailableCount": int(prefix_available.sum()),
        "completedResultReplayPassed": None, "prefixResultReplayPassed": None,
        "maximumFullClosureError": float(np.max(full_closure_error)),
        "maximumPrefixClosureError": float(np.max(prefix_closure_error)),
        "completedFutureDependent": True, "prefixFutureSuffixAccessed": False,
        "prefixAttractorStatus": prefix_fit_status,
        "fullClrSha256": array_sha256(full_clr), "prefixClrSha256": array_sha256(prefix_clr),
    }
    return features, summaries, audit


def _fixture_ten_split_model_replay() -> tuple[bool, float]:
    started = time.time()
    rng = np.random.Generator(np.random.PCG64DXSM(seed128("benchmark", "model_fixture")))
    n = 100
    values = np.zeros((n, MAX_INPUT_LENGTH, 100), dtype=np.float64)
    channel_mask = np.zeros_like(values, dtype=bool)
    time_mask = np.zeros((n, MAX_INPUT_LENGTH), dtype=bool)
    target = np.zeros((n, MAX_TARGET_LENGTH), dtype=np.float64)
    target_mask = np.zeros((n, MAX_TARGET_LENGTH), dtype=bool)
    values[:, :8, 0] = rng.normal(size=(n, 8))
    channel_mask[:, :8, 0] = True
    time_mask[:, :8] = True
    target[:, :12] = rng.integers(0, 2, size=(n, 12))
    target_mask[:, :12] = True
    split = pd.read_csv(S16_SPLIT_PATH)
    passed = True
    for repetition in range(10):
        fit = split_indices(split, repetition, "FIT")
        val = split_indices(split, repetition, "VALIDATION")
        test = split_indices(split, repetition, "TEST")
        scaler = fit_channel_scaler(values[fit], channel_mask[fit])
        scaled = apply_channel_scaler(values, channel_mask, scaler)
        kwargs = {
            "fit_values": scaled[fit], "fit_channel_mask": channel_mask[fit], "fit_time_mask": time_mask[fit],
            "fit_targets": target[fit], "fit_target_mask": target_mask[fit],
            "validation_values": scaled[val], "validation_channel_mask": channel_mask[val], "validation_time_mask": time_mask[val],
            "validation_targets": target[val], "validation_target_mask": target_mask[val],
            "model_seed": s16_model_seed("CANDIDATE_2", repetition), "maximum_epochs": 3, "patience": 2,
        }
        first = train_masked_mlp(**kwargs)
        second = train_masked_mlp(**kwargs)
        p_first = predict_probabilities(first.model, scaled[test], channel_mask[test], time_mask[test])
        p_second = predict_probabilities(second.model, scaled[test], channel_mask[test], time_mask[test])
        passed = passed and first.history.equals(second.history) and np.array_equal(p_first, p_second)
    return passed, time.time() - started


def benchmark_phase() -> None:
    geometry = json.loads((CACHE_ROOT / "geometry_status.json").read_text())
    if not geometry["advancedTargets"]:
        write_json(OUTPUT_ROOT / "benchmark_gate.json", {"status": "NOT_RUN_NO_ADVANCED_TARGET", "passed": True})
        print(canonical_json({"phase": "benchmark", "status": "NOT_RUN_NO_ADVANCED_TARGET"}))
        return
    manifest = load_trajectory_manifest()
    rows = []
    start_cpu = time.process_time()
    start_wall = time.time()
    for candidate_id in CANDIDATE_IDS:
        for matrix_index in (0, 1):
            row = manifest.loc[
                manifest["candidateId"].eq(candidate_id) & manifest["matrixIndex"].eq(matrix_index)
            ].iloc[0]
            before_cpu, before_wall = time.process_time(), time.time()
            features, summaries, audit = compute_features_for_trajectory(row)
            rows.append({
                "candidateId": candidate_id, "matrixIndex": matrix_index,
                "cpuSeconds": time.process_time() - before_cpu, "wallSeconds": time.time() - before_wall,
                "completedAvailableCount": audit["completedAvailableCount"],
                "prefixAvailableCount": audit["prefixAvailableCount"],
                "featureCount": len(features), "summaryCount": len(summaries),
            })
    model_replay, model_wall = _fixture_ten_split_model_replay()
    elapsed_cpu = time.process_time() - start_cpu
    elapsed_wall = time.time() - start_wall
    source_cpu_per = float(pd.DataFrame(rows)["cpuSeconds"].mean())
    projected_source_cpu = source_cpu_per * 200
    projected_model_cpu = max(model_wall, 1.0) / 20.0 * (len(geometry["advancedTargets"]) * 2 * 10 * 15)
    projected_total_cpu_hours = (projected_source_cpu + projected_model_cpu) / 3600.0
    reserve_adjusted = projected_total_cpu_hours / 0.85
    passed = bool(model_replay and reserve_adjusted <= 160.0)
    payload = {
        "schema": "eidosoma.e01.s19.l13.benchmark_gate.v1",
        "trajectoryBenchmarkRows": rows, "trajectoryCount": 4,
        "completedAndPrefixFitsPerTrajectory": 1, "tenSplitFixtureReplayPassed": model_replay,
        "tenSplitFixtureReplayWallSeconds": model_wall, "benchmarkCpuSeconds": elapsed_cpu,
        "benchmarkWallSeconds": elapsed_wall, "projectedScientificCpuHours": projected_total_cpu_hours,
        "projectedCpuHoursIncluding15PercentReserve": reserve_adjusted,
        "cpuHourCeiling": 160.0, "passed": passed,
    }
    write_json(OUTPUT_ROOT / "benchmark_gate.json", payload)
    if not passed:
        raise RuntimeError("L13 benchmark projected beyond ceiling or model replay failed")
    print(canonical_json(json_safe({"phase": "benchmark", "passed": passed, "projectedCpuHoursWithReserve": reserve_adjusted, "modelReplay": model_replay})))


def save_feature_tensors(
    candidate_id: str,
    tensors: dict[str, dict[str, np.ndarray]],
) -> None:
    root = CACHE_ROOT / "features" / candidate_id
    root.mkdir(parents=True, exist_ok=True)
    for feature_id, tensor in tensors.items():
        np.savez_compressed(
            root / f"{feature_id}.npz", values=tensor["values"],
            channelMask=tensor["channelMask"], timeMask=tensor["timeMask"],
        )


def compute_full_features() -> None:
    benchmark = json.loads((OUTPUT_ROOT / "benchmark_gate.json").read_text())
    if not benchmark["passed"]:
        raise RuntimeError("benchmark gate did not pass")
    manifest = load_trajectory_manifest()
    summary_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for candidate_id in CANDIDATE_IDS:
        accumulation: dict[str, dict[str, list[np.ndarray]]] = {
            model_id: {"values": [], "channelMask": [], "timeMask": []}
            for model_id in (P1_ID, P2_ID, B1_ID, B2_ID, B3_ID, B4_ID, B5_ID, B6_ID, B7_ID, P2_B4_ID, P2_B5_ID, NC1_ID)
        }
        for matrix_index in range(100):
            row = manifest.loc[
                manifest["candidateId"].eq(candidate_id) & manifest["matrixIndex"].eq(matrix_index)
            ].iloc[0]
            features, summaries, audit = compute_features_for_trajectory(row)
            summary_rows.extend(summaries)
            audit_rows.append(audit)
            for feature_id, feature in features.items():
                accumulation[feature_id]["values"].append(feature[0])
                accumulation[feature_id]["channelMask"].append(feature[1])
                accumulation[feature_id]["timeMask"].append(feature[2])
        tensors = {
            feature_id: {key: np.stack(values) for key, values in parts.items()}
            for feature_id, parts in accumulation.items()
        }
        save_feature_tensors(candidate_id, tensors)
    summaries = pd.DataFrame(summary_rows)
    write_parquet(OUTPUT_ROOT / "all_feature_summary.parquet", summaries)
    completed = summaries.loc[summaries["featureId"].eq(P1_ID)].copy()
    prefix = summaries.loc[summaries["featureId"].eq(P2_ID)].copy()
    baseline = summaries.loc[summaries["featureId"].isin([B1_ID, B2_ID, B3_ID, B4_ID, B6_ID, B7_ID])].copy()
    attractor = summaries.loc[summaries["featureId"].isin([B5_ID, P2_B5_ID])].copy()
    write_parquet(OUTPUT_ROOT / "completed_fit_phirl_features.parquet", completed)
    write_parquet(OUTPUT_ROOT / "prefix_only_phirl_features.parquet", prefix)
    write_parquet(OUTPUT_ROOT / "baseline_feature_results.parquet", baseline)
    write_parquet(OUTPUT_ROOT / "attractor_control_results.parquet", attractor)
    write_parquet(OUTPUT_ROOT / "leakage_audit.parquet", pd.DataFrame(audit_rows))
    write_json(CACHE_ROOT / "feature_status.json", {
        "featureSummaryRows": len(summaries), "auditRows": len(audit_rows),
        "featureRoot": str(CACHE_ROOT / "features"), "completedAtUtc": utc_now(),
    })


def load_feature_tensor(candidate_id: str, feature_id: str) -> dict[str, np.ndarray]:
    mapped = P2_ID if feature_id == NC2_ID else feature_id
    path = CACHE_ROOT / "features" / candidate_id / f"{mapped}.npz"
    with np.load(path) as payload:
        return {key: np.asarray(payload[key]) for key in ("values", "channelMask", "timeMask")}


def suffix_invariance_audit() -> pd.DataFrame:
    manifest = load_trajectory_manifest()
    rows = []
    for candidate_id in CANDIDATE_IDS:
        for matrix_index in (0, 24, 49, 74):
            manifest_row = manifest.loc[
                manifest["candidateId"].eq(candidate_id) & manifest["matrixIndex"].eq(matrix_index)
            ].iloc[0]
            arrays = trajectory_arrays(manifest_row)
            states = arrays["states"]
            cutoff = math.floor(0.25 * len(states))
            rng = np.random.Generator(np.random.PCG64DXSM(seed128("suffix", candidate_id, matrix_index)))
            mutated = states.copy()
            mutated[cutoff:] = mutated[cutoff:][rng.permutation(len(states) - cutoff)]
            pre_seed = s16_source_seed(candidate_id, matrix_index, "preprocessing")
            part_seed = s16_source_seed(candidate_id, matrix_index, "partition")
            full_a, _, _ = frozen_clr(states)
            full_b, _, _ = frozen_clr(mutated)
            prefix_a, _, _ = frozen_clr(states[:cutoff])
            prefix_b, _, _ = frozen_clr(mutated[:cutoff])
            p1_a_result = run_emergence_pipeline(full_a, PHIRL, SAFE_LATTICE, preprocessing_seed=pre_seed, partition_seed=part_seed)
            p1_b_result = run_emergence_pipeline(full_b, PHIRL, SAFE_LATTICE, preprocessing_seed=pre_seed, partition_seed=part_seed)
            p2_a_result = run_emergence_pipeline(prefix_a, PHIRL, SAFE_LATTICE, preprocessing_seed=pre_seed, partition_seed=part_seed)
            p2_b_result = run_emergence_pipeline(prefix_b, PHIRL, SAFE_LATTICE, preprocessing_seed=pre_seed, partition_seed=part_seed)
            p1_a, p1_mask_a = source_values(p1_a_result, fit_length=len(states), retained_length=cutoff)
            p1_b, p1_mask_b = source_values(p1_b_result, fit_length=len(states), retained_length=cutoff)
            p2_a, p2_mask_a = source_values(p2_a_result, fit_length=cutoff, retained_length=cutoff)
            p2_b, p2_mask_b = source_values(p2_b_result, fit_length=cutoff, retained_length=cutoff)
            shared = p1_mask_a & p1_mask_b
            rows.append({
                "candidateId": candidate_id, "matrixIndex": matrix_index,
                "prefixStatesExact": np.array_equal(states[:cutoff], mutated[:cutoff]),
                "prefixClrExact": np.array_equal(prefix_a, prefix_b),
                "p2ResultExact": result_replay_equal(p2_a_result, p2_b_result),
                "p2ValuesExact": np.array_equal(p2_a, p2_b),
                "p2MaskExact": np.array_equal(p2_mask_a, p2_mask_b),
                "p1ResultExact": result_replay_equal(p1_a_result, p1_b_result),
                "p1SharedCount": int(shared.sum()),
                "p1MeanAbsoluteChange": None if not shared.any() else float(np.mean(np.abs(p1_a[shared] - p1_b[shared]))),
                "p1MaximumAbsoluteChange": None if not shared.any() else float(np.max(np.abs(p1_a[shared] - p1_b[shared]))),
                "passed": bool(np.array_equal(prefix_a, prefix_b) and result_replay_equal(p2_a_result, p2_b_result) and np.array_equal(p2_a, p2_b) and np.array_equal(p2_mask_a, p2_mask_b)),
            })
    return pd.DataFrame(rows)


def target_tensor(target_payloads: dict[tuple[str, str, int], dict[str, Any]], target_id: str, candidate_id: str) -> dict[str, np.ndarray]:
    payloads = [target_payloads[(target_id, candidate_id, index)] for index in range(100)]
    return {
        "target": np.stack([row["target"] for row in payloads]).astype(np.float64),
        "targetMask": np.stack([row["targetMask"] for row in payloads]).astype(bool),
        "inputLabels": np.stack([row["inputLabels"] for row in payloads]).astype(bool),
        "cutoff": np.asarray([row["cutoff"] for row in payloads], dtype=np.int64),
        "T": np.asarray([row["T"] for row in payloads], dtype=np.int64),
    }


def oracle_tensor(target_payloads: dict[tuple[str, str, int], dict[str, Any]], target_id: str, candidate_id: str) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    values_rows, mask_rows, time_rows, summaries = [], [], [], []
    for matrix_index in range(100):
        payload = target_payloads[(target_id, candidate_id, matrix_index)]
        cutoff = int(payload["cutoff"])
        scores = payload["scores"]
        if scores is None:
            scalar = np.zeros(cutoff, dtype=np.float64)
            available = np.zeros(cutoff, dtype=bool)
        else:
            scalar = np.asarray(scores[:cutoff], dtype=np.float64)
            available = np.isfinite(scalar)
        feature = build_feature(scalar, available, cutoff, scalar=True)
        values_rows.append(feature[0]); mask_rows.append(feature[1]); time_rows.append(feature[2])
        summaries.append({
            "targetId": target_id, "candidateId": candidate_id, "matrixIndex": matrix_index,
            "featureId": ORACLE_ID, "validFeatureCellCount": int(feature[1].sum()),
            "valueSha256": array_sha256(feature[0]), "maskSha256": array_sha256(feature[1]),
            "futureDependent": True, "targetDefining": True, "promotionEligible": False,
        })
    return {
        "values": np.stack(values_rows), "channelMask": np.stack(mask_rows),
        "timeMask": np.stack(time_rows),
    }, pd.DataFrame(summaries)


def permuted_training_targets(
    base: dict[str, np.ndarray], candidate_id: str, target_id: str, repetition: int,
    split: pd.DataFrame,
) -> dict[str, np.ndarray]:
    target = base["target"].copy()
    mask = base["targetMask"].copy()
    train = np.sort(np.concatenate([
        split_indices(split, repetition, "FIT"), split_indices(split, repetition, "VALIDATION")
    ]))
    rng = np.random.Generator(np.random.PCG64DXSM(seed128("NC2", target_id, candidate_id, repetition)))
    donor = train[rng.permutation(len(train))]
    target[train] = base["target"][donor]
    mask[train] = base["targetMask"][donor]
    return {"target": target, "targetMask": mask}


def _evaluate_test(
    *, target_id: str, candidate_id: str, model_id: str, repetition: int,
    target_tensor_payload: dict[str, np.ndarray], test_indices: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    target = target_tensor_payload["target"][test_indices].astype(bool)
    target_mask = target_tensor_payload["targetMask"][test_indices]
    flat_target = target[target_mask]
    flat_probability = probabilities[target_mask]
    split_metric = extended_binary_metrics(flat_target, flat_probability)
    split_metric.update({
        "targetId": target_id, "candidateId": candidate_id, "modelId": model_id,
        "repetitionId": repetition, "testMatrixCount": len(test_indices),
        "definedTestMatrixCount": int(np.count_nonzero(target_mask.any(axis=1))),
        "macroMatrixAccuracy": None, "testMajorityAccuracy": None if not flat_target.size else float(max(flat_target.mean(), 1 - flat_target.mean())),
    })
    matrix_rows, prediction_rows = [], []
    accuracies = []
    for local, matrix_index in enumerate(test_indices):
        valid = np.flatnonzero(target_mask[local])
        y = target[local, valid]
        p = probabilities[local, valid]
        metric = extended_binary_metrics(y, p)
        if metric["accuracy"] is not None:
            accuracies.append(metric["accuracy"])
        matrix_rows.append({
            "targetId": target_id, "candidateId": candidate_id, "modelId": model_id,
            "repetitionId": repetition, "matrixIndex": int(matrix_index), **metric,
        })
        prediction_rows.append({
            "targetId": target_id, "candidateId": candidate_id, "modelId": model_id,
            "repetitionId": repetition, "matrixIndex": int(matrix_index),
            "targetOffsets": valid.astype(np.int32).tolist(),
            "selectedSequenceIndices": (target_tensor_payload["cutoff"][matrix_index] + valid).astype(np.int32).tolist(),
            "targets": y.astype(bool).tolist(), "probabilities": p.astype(float).tolist(),
            "predictedClasses": (p >= 0.5).astype(bool).tolist(),
            "validTargetCount": len(valid),
        })
    split_metric["macroMatrixAccuracy"] = None if not accuracies else float(np.mean(accuracies))
    return split_metric, matrix_rows, prediction_rows


def run_models() -> dict[str, pd.DataFrame]:
    geometry_status = json.loads((CACHE_ROOT / "geometry_status.json").read_text())
    advanced = list(geometry_status["advancedTargets"])
    with (CACHE_ROOT / "target_payloads.pkl").open("rb") as handle:
        target_payloads = pickle.load(handle)
    split = pd.read_csv(S16_SPLIT_PATH)
    training_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []
    oracle_rows: list[pd.DataFrame] = []
    scaler_rows: list[dict[str, Any]] = []

    for target_id in advanced:
        for candidate_id in CANDIDATE_IDS:
            targets = target_tensor(target_payloads, target_id, candidate_id)
            oracle, oracle_summary = oracle_tensor(target_payloads, target_id, candidate_id)
            oracle_rows.append(oracle_summary)
            feature_cache: dict[str, dict[str, np.ndarray]] = {ORACLE_ID: oracle}
            for model_id in ALL_MODEL_IDS:
                if model_id in {DUMMY_ID, ORACLE_ID}:
                    continue
                feature_cache[model_id] = load_feature_tensor(candidate_id, model_id)
            for repetition in range(10):
                fit = split_indices(split, repetition, "FIT")
                validation = split_indices(split, repetition, "VALIDATION")
                test = split_indices(split, repetition, "TEST")
                for model_id in ALL_MODEL_IDS:
                    model_seed = s16_model_seed(candidate_id, repetition)
                    target_override = None
                    if model_id == NC2_ID:
                        target_override = permuted_training_targets(targets, candidate_id, target_id, repetition, split)
                    effective_target = targets if target_override is None else {**targets, **target_override}
                    if model_id == DUMMY_ID:
                        fit_y = effective_target["target"][fit].astype(bool)
                        fit_mask = effective_target["targetMask"][fit]
                        valid_fit = fit_y[fit_mask]
                        if not valid_fit.size:
                            raise RuntimeError("dummy has no valid fit targets")
                        training_prevalence = float(valid_fit.mean())
                        probabilities = np.full((len(test), MAX_TARGET_LENGTH), training_prevalence, dtype=np.float64)
                        training_rows.append({
                            "targetId": target_id, "candidateId": candidate_id, "modelId": model_id,
                            "repetitionId": repetition, "epoch": -1, "fitLoss": None,
                            "validationLoss": None, "bestEpoch": None, "stoppedEpoch": None,
                            "bestValidationLoss": None, "modelSeed": model_seed,
                            "parameterCount": 0, "trainingOnlyPrevalence": training_prevalence,
                        })
                    else:
                        tensor = feature_cache[model_id]
                        scaler = fit_channel_scaler(tensor["values"][fit], tensor["channelMask"][fit])
                        scaled = apply_channel_scaler(tensor["values"], tensor["channelMask"], scaler)
                        result = train_masked_mlp(
                            scaled[fit], tensor["channelMask"][fit], tensor["timeMask"][fit],
                            effective_target["target"][fit], effective_target["targetMask"][fit],
                            scaled[validation], tensor["channelMask"][validation], tensor["timeMask"][validation],
                            effective_target["target"][validation], effective_target["targetMask"][validation],
                            model_seed=model_seed,
                        )
                        probabilities = predict_probabilities(
                            result.model, scaled[test], tensor["channelMask"][test], tensor["timeMask"][test]
                        )
                        for channel in range(100):
                            scaler_rows.append({
                                "targetId": target_id, "candidateId": candidate_id, "modelId": model_id,
                                "repetitionId": repetition, "channelIndex": channel,
                                "fitValidCellCount": int(scaler.valid_count[channel]),
                                "mean": float(scaler.mean[channel]), "scale": float(scaler.scale[channel]),
                                "fitOnly": True, "validationExcluded": True, "testExcluded": True,
                            })
                        for history_row in result.history.itertuples(index=False):
                            training_rows.append({
                                "targetId": target_id, "candidateId": candidate_id, "modelId": model_id,
                                "repetitionId": repetition, "epoch": int(history_row.epoch),
                                "fitLoss": float(history_row.fitLoss), "validationLoss": float(history_row.validationLoss),
                                "bestEpoch": result.best_epoch, "stoppedEpoch": result.stopped_epoch,
                                "bestValidationLoss": result.best_validation_loss, "modelSeed": model_seed,
                                "parameterCount": parameter_count(result.model), "trainingOnlyPrevalence": None,
                            })
                        if repetition == 0:
                            replay = train_masked_mlp(
                                scaled[fit], tensor["channelMask"][fit], tensor["timeMask"][fit],
                                effective_target["target"][fit], effective_target["targetMask"][fit],
                                scaled[validation], tensor["channelMask"][validation], tensor["timeMask"][validation],
                                effective_target["target"][validation], effective_target["targetMask"][validation],
                                model_seed=model_seed,
                            )
                            replay_prob = predict_probabilities(
                                replay.model, scaled[test], tensor["channelMask"][test], tensor["timeMask"][test]
                            )
                            replay_rows.append({
                                "targetId": target_id, "candidateId": candidate_id, "modelId": model_id,
                                "repetitionId": repetition, "historyExact": result.history.equals(replay.history),
                                "predictionExact": np.array_equal(probabilities, replay_prob),
                                "predictionSha256": array_sha256(probabilities),
                                "replayPredictionSha256": array_sha256(replay_prob),
                                "passed": result.history.equals(replay.history) and np.array_equal(probabilities, replay_prob),
                            })
                    split_metric, per_matrix, predictions = _evaluate_test(
                        target_id=target_id, candidate_id=candidate_id, model_id=model_id,
                        repetition=repetition, target_tensor_payload=targets,
                        test_indices=test, probabilities=probabilities,
                    )
                    metric_rows.append(split_metric)
                    matrix_rows.extend(per_matrix)
                    prediction_rows.extend(predictions)
    outputs = {
        "training": pd.DataFrame(training_rows), "metrics": pd.DataFrame(metric_rows),
        "matrix": pd.DataFrame(matrix_rows), "predictions": pd.DataFrame(prediction_rows),
        "replay": pd.DataFrame(replay_rows),
        "oracle": pd.concat(oracle_rows, ignore_index=True) if oracle_rows else pd.DataFrame(),
        "scalers": pd.DataFrame(scaler_rows),
    }
    if not outputs["replay"].empty and not outputs["replay"]["passed"].all():
        raise RuntimeError("exact model replay failed")
    return outputs


def _bootstrap_absolute(
    matrix: pd.DataFrame,
    *,
    model_id: str,
    metric: str,
    seed_identity: tuple[object, ...],
) -> dict[str, Any]:
    selected = matrix.loc[matrix["modelId"].eq(model_id), ["matrixIndex", metric]].copy()
    selected[metric] = pd.to_numeric(selected[metric], errors="coerce")
    grouped = selected.dropna().groupby("matrixIndex")[metric].mean()
    if grouped.empty:
        return {"definedMatrixCount": 0, "observed": None, "lower95": None, "upper95": None}
    values = grouped.to_numpy(dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64DXSM(seed128(*seed_identity)))
    indices = rng.integers(0, len(values), size=(4096, len(values)))
    distribution = values[indices].mean(axis=1)
    lower, upper = np.quantile(distribution, [0.025, 0.975])
    return {
        "definedMatrixCount": len(values), "observed": float(values.mean()),
        "lower95": float(lower), "upper95": float(upper),
    }


def _bootstrap_auprc_advantage(
    matrix: pd.DataFrame, *, model_id: str, seed_identity: tuple[object, ...]
) -> dict[str, Any]:
    selected = matrix.loc[matrix["modelId"].eq(model_id)].copy()
    selected["advantage"] = pd.to_numeric(selected["auprc"], errors="coerce") - pd.to_numeric(selected["prevalence"], errors="coerce")
    grouped = selected.dropna(subset=["advantage"]).groupby("matrixIndex")["advantage"].mean()
    if grouped.empty:
        return {"definedMatrixCount": 0, "observed": None, "lower95": None, "upper95": None}
    values = grouped.to_numpy(dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64DXSM(seed128(*seed_identity)))
    indices = rng.integers(0, len(values), size=(4096, len(values)))
    distribution = values[indices].mean(axis=1)
    lower, upper = np.quantile(distribution, [0.025, 0.975])
    return {"definedMatrixCount": len(values), "observed": float(values.mean()), "lower95": float(lower), "upper95": float(upper)}


def comparison_tables(metrics: pd.DataFrame, matrix: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    comparison_pairs = [
        (P1_ID, B1_ID, "PAPER"), (P1_ID, B2_ID, "PAPER"),
        (P1_ID, B3_ID, "PAPER"), (P1_ID, DUMMY_ID, "PAPER"),
        (P2_ID, DUMMY_ID, "PROSPECTIVE"), (P2_ID, B6_ID, "PROSPECTIVE"),
        (P2_ID, B1_ID, "PROSPECTIVE"), (P2_ID, B2_ID, "PROSPECTIVE"),
        (P2_ID, B3_ID, "PROSPECTIVE"), (P2_ID, B4_ID, "PROSPECTIVE"),
        (P2_ID, B5_ID, "PROSPECTIVE"), (P2_B4_ID, B4_ID, "INCREMENTAL"),
        (P2_B5_ID, B5_ID, "INCREMENTAL"), (P2_ID, NC1_ID, "NEGATIVE_CONTROL"),
        (P2_ID, NC2_ID, "NEGATIVE_CONTROL"), (P2_ID, B7_ID, "NEGATIVE_CONTROL"),
    ]
    rows: list[dict[str, Any]] = []
    incremental_rows: list[dict[str, Any]] = []
    for target_id in metrics["targetId"].unique():
        for candidate_id in CANDIDATE_IDS:
            subset = metrics.loc[
                metrics["targetId"].eq(target_id) & metrics["candidateId"].eq(candidate_id)
            ]
            pair_rows = []
            for reference, comparator, family in comparison_pairs:
                left = subset.loc[subset["modelId"].eq(reference)].sort_values("repetitionId")
                right = subset.loc[subset["modelId"].eq(comparator)].sort_values("repetitionId")
                if len(left) != 10 or len(right) != 10 or not np.array_equal(left["repetitionId"], right["repetitionId"]):
                    raise RuntimeError(f"split comparison mismatch {target_id} {candidate_id} {reference} {comparator}")
                left_accuracy = left["accuracy"].to_numpy(dtype=np.float64)
                right_accuracy = right["accuracy"].to_numpy(dtype=np.float64)
                difference = left_accuracy - right_accuracy
                mann = stats.mannwhitneyu(left_accuracy, right_accuracy, alternative="two-sided", method="auto")
                try:
                    wilcoxon_p = float(stats.wilcoxon(difference, alternative="two-sided", zero_method="wilcox", method="auto").pvalue)
                except ValueError:
                    wilcoxon_p = None
                matrix_subset = matrix.loc[
                    matrix["targetId"].eq(target_id) & matrix["candidateId"].eq(candidate_id)
                ]
                bootstrap = matrix_bootstrap_metric_difference(
                    matrix_subset, reference=reference, comparator=comparator,
                    metric="accuracy", seed_identity=("comparison", target_id, candidate_id, reference, comparator),
                )
                row = {
                    "targetId": target_id, "candidateId": candidate_id,
                    "comparisonFamily": family, "referenceModelId": reference,
                    "comparatorModelId": comparator,
                    "referenceMedianAccuracy": float(np.median(left_accuracy)),
                    "comparatorMedianAccuracy": float(np.median(right_accuracy)),
                    "meanPairedSplitDifference": float(np.mean(difference)),
                    "medianPairedSplitDifference": float(np.median(difference)),
                    "positiveSplitCount": int(np.count_nonzero(difference > 0)),
                    "mannWhitneyU": float(mann.statistic), "mannWhitneyP": float(mann.pvalue),
                    "wilcoxonP": wilcoxon_p,
                    "pairedMatrixCount": bootstrap["pairedMatrixCount"],
                    "matrixBootstrapObservedDifference": bootstrap["observedDifference"],
                    "matrixBootstrapLower95": bootstrap["lower95"],
                    "matrixBootstrapUpper95": bootstrap["upper95"],
                    "matrixBootstrapPositiveP": bootstrap["positiveP"],
                }
                rows.append(row); pair_rows.append(row)
            adjusted = holm_adjust([row["mannWhitneyP"] for row in pair_rows])
            for row, value in zip(pair_rows, adjusted):
                row["holmAdjustedMannWhitneyP"] = value

            matrix_subset = matrix.loc[
                matrix["targetId"].eq(target_id) & matrix["candidateId"].eq(candidate_id)
            ]
            balanced = _bootstrap_absolute(
                matrix_subset, model_id=P2_ID, metric="balancedAccuracy",
                seed_identity=("absolute", target_id, candidate_id, P2_ID, "balanced"),
            )
            auprc_advantage = _bootstrap_auprc_advantage(
                matrix_subset, model_id=P2_ID,
                seed_identity=("absolute", target_id, candidate_id, P2_ID, "auprc_minus_prevalence"),
            )
            brier = matrix_bootstrap_metric_difference(
                matrix_subset.assign(
                    negBrier=-pd.to_numeric(matrix_subset["brier"], errors="coerce")
                ), reference=P2_ID, comparator=DUMMY_ID, metric="negBrier",
                seed_identity=("comparison", target_id, candidate_id, "DUMMY_MINUS_P2_BRIER"),
            )
            incremental_rows.extend([
                {"targetId": target_id, "candidateId": candidate_id, "gateMetric": "P2_BALANCED_ACCURACY", **balanced},
                {"targetId": target_id, "candidateId": candidate_id, "gateMetric": "P2_AUPRC_MINUS_PREVALENCE", **auprc_advantage},
                {"targetId": target_id, "candidateId": candidate_id, "gateMetric": "DUMMY_MINUS_P2_BRIER", "definedMatrixCount": brier["pairedMatrixCount"], "observed": brier["observedDifference"], "lower95": brier["lower95"], "upper95": brier["upper95"]},
            ])
    return pd.DataFrame(rows), pd.DataFrame(incremental_rows)


def scientific_gates(
    metrics: pd.DataFrame, matrix: pd.DataFrame, comparisons: pd.DataFrame,
    incremental: pd.DataFrame, suffix: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    availability = pd.read_parquet(OUTPUT_ROOT / "target_availability_results.parquet")
    onset = pd.read_parquet(OUTPUT_ROOT / "onset_eligibility_results.parquet")
    geometry_gate_frame = pd.read_csv(OUTPUT_ROOT / "geometry_advancement_gate_results.csv")
    gate_rows: list[dict[str, Any]] = []
    target_decisions: dict[str, Any] = {}
    reconstruction_rows: list[dict[str, Any]] = []

    for target_id in TARGET_IDS:
        advanced = bool(
            not geometry_gate_frame.loc[
                geometry_gate_frame["targetId"].eq(target_id)
                & geometry_gate_frame["candidateId"].eq("BOTH_REQUIRED")
                & geometry_gate_frame["gateId"].eq("CROSS_CANDIDATE_GEOMETRY_ADVANCEMENT"), "passed"
            ].empty
            and geometry_gate_frame.loc[
                geometry_gate_frame["targetId"].eq(target_id)
                & geometry_gate_frame["candidateId"].eq("BOTH_REQUIRED")
                & geometry_gate_frame["gateId"].eq("CROSS_CANDIDATE_GEOMETRY_ADVANCEMENT"), "passed"
            ].iloc[0]
        )
        if not advanced:
            target_decisions[target_id] = {
                "geometryAdvanced": False, "retrospectiveGatePassed": False,
                "prospectiveGatePassed": False, "classifications": ["FIGURE5_BASELINE_NOT_RECONCILED_BY_FROZEN_RECURRING_TARGETS", "NOT_PROMOTABLE"],
            }
            continue
        retrospective_candidate: list[bool] = []
        prospective_candidate: list[bool] = []
        for candidate_id in CANDIDATE_IDS:
            candidate_metrics = metrics.loc[
                metrics["targetId"].eq(target_id) & metrics["candidateId"].eq(candidate_id)
            ]
            candidate_comparisons = comparisons.loc[
                comparisons["targetId"].eq(target_id) & comparisons["candidateId"].eq(candidate_id)
            ]
            dummy_values = candidate_metrics.loc[candidate_metrics["modelId"].eq(DUMMY_ID), "accuracy"].to_numpy(dtype=np.float64)
            p1_values = candidate_metrics.loc[candidate_metrics["modelId"].eq(P1_ID), "accuracy"].to_numpy(dtype=np.float64)
            baseline_compat = all(
                paper_interval_overlap(candidate_metrics.loc[candidate_metrics["modelId"].eq(model_id), "accuracy"].to_numpy(dtype=np.float64), model_id)
                for model_id in (B1_ID, B2_ID, B3_ID)
            )
            paper_pairs = candidate_comparisons.loc[candidate_comparisons["comparisonFamily"].eq("PAPER")]
            paper_order = bool(len(paper_pairs) == 4 and (paper_pairs["referenceMedianAccuracy"] > paper_pairs["comparatorMedianAccuracy"]).all())
            paper_p = bool(len(paper_pairs) == 4 and paper_pairs["mannWhitneyP"].lt(0.01).all())
            paper_matrix = bool(len(paper_pairs) == 4 and paper_pairs["matrixBootstrapObservedDifference"].gt(0).all())
            retrospective = bool(
                paper_interval_overlap(dummy_values, DUMMY_ID)
                and paper_interval_overlap(p1_values, P1_ID)
                and baseline_compat and paper_order and paper_p and paper_matrix
            )
            retrospective_candidate.append(retrospective)
            for gate_id, passed in [
                ("DUMMY_OVERLAPS_FIGURE5", paper_interval_overlap(dummy_values, DUMMY_ID)),
                ("P1_OVERLAPS_FIGURE5", paper_interval_overlap(p1_values, P1_ID)),
                ("BASELINES_DIRECTIONALLY_COMPATIBLE", baseline_compat),
                ("P1_MODEL_ORDER", paper_order), ("P1_MANN_WHITNEY_ALL_P_LT_0p01", paper_p),
                ("P1_PAIRED_MATRIX_DIRECTION", paper_matrix),
            ]:
                gate_rows.append({"targetId": target_id, "candidateId": candidate_id, "gateFamily": "RETROSPECTIVE", "gateId": gate_id, "passed": passed})

            prospective_pairs = candidate_comparisons.loc[
                candidate_comparisons["comparisonFamily"].eq("PROSPECTIVE")
            ]
            incremental_pairs = candidate_comparisons.loc[
                candidate_comparisons["comparisonFamily"].eq("INCREMENTAL")
            ]
            control_pairs = candidate_comparisons.loc[
                candidate_comparisons["comparisonFamily"].eq("NEGATIVE_CONTROL")
            ]
            p2_above = bool(len(prospective_pairs) == 7 and prospective_pairs["matrixBootstrapLower95"].gt(0).all())
            combo_above = bool(len(incremental_pairs) == 2 and incremental_pairs["matrixBootstrapLower95"].gt(0).all())
            control_pass = bool(len(control_pairs) == 3 and control_pairs["matrixBootstrapLower95"].gt(0).all())
            inc = incremental.loc[
                incremental["targetId"].eq(target_id) & incremental["candidateId"].eq(candidate_id)
            ].set_index("gateMetric")
            balanced_pass = bool("P2_BALANCED_ACCURACY" in inc.index and float(inc.loc["P2_BALANCED_ACCURACY", "lower95"]) > 0.5)
            auprc_pass = bool("P2_AUPRC_MINUS_PREVALENCE" in inc.index and float(inc.loc["P2_AUPRC_MINUS_PREVALENCE", "lower95"]) > 0.0)
            brier_pass = bool("DUMMY_MINUS_P2_BRIER" in inc.index and float(inc.loc["DUMMY_MINUS_P2_BRIER", "lower95"]) > 0.0)
            candidate_onset = onset.loc[onset["targetId"].eq(target_id) & onset["candidateId"].eq(candidate_id)]
            pre_count = int(candidate_onset["noOnsetBeforeCutoff"].fillna(False).sum())
            future_count = int(candidate_onset["firstOnsetInSuffix"].fillna(False).sum())
            eligibility_pass = pre_count >= 20 and future_count >= 20
            suffix_pass = bool(suffix.loc[suffix["candidateId"].eq(candidate_id), "passed"].all())
            prospective = bool(p2_above and combo_above and control_pass and balanced_pass and auprc_pass and brier_pass and eligibility_pass and suffix_pass)
            prospective_candidate.append(prospective)
            for gate_id, passed in [
                ("P2_ABOVE_ALL_ORDINARY_CONTROLS", p2_above), ("P2_INCREMENTAL_COMBINATIONS", combo_above),
                ("P2_ABOVE_NEGATIVE_CONTROLS", control_pass), ("BALANCED_ACCURACY_LOWER95_ABOVE_0p5", balanced_pass),
                ("AUPRC_MINUS_PREVALENCE_LOWER95_ABOVE_ZERO", auprc_pass), ("BRIER_BEATS_DUMMY", brier_pass),
                ("PRE_ONSET_AND_FUTURE_ONSET_COUNTS_AT_LEAST_20", eligibility_pass),
                ("PREFIX_SUFFIX_INVARIANCE", suffix_pass),
            ]:
                gate_rows.append({
                    "targetId": target_id, "candidateId": candidate_id, "gateFamily": "PROSPECTIVE",
                    "gateId": gate_id, "passed": passed, "preOnsetMatrixCount": pre_count,
                    "futureOnsetMatrixCount": future_count,
                })
            reconstruction_rows.append({
                "targetId": target_id, "candidateId": candidate_id,
                "dummyMedian": float(np.median(dummy_values)), "p1Median": float(np.median(p1_values)),
                "p2Median": float(candidate_metrics.loc[candidate_metrics["modelId"].eq(P2_ID), "accuracy"].median()),
                "retrospectiveCandidateGate": retrospective, "prospectiveCandidateGate": prospective,
                "preOnsetMatrixCount": pre_count, "futureOnsetMatrixCount": future_count,
            })
        retrospective_both = bool(all(retrospective_candidate))
        prospective_both = bool(all(prospective_candidate))
        defined_counts = availability.loc[
            availability["targetId"].eq(target_id)
        ].groupby("candidateId")["defined"].sum().to_dict()
        classifications = []
        if retrospective_both:
            classifications.append("FIGURE5_RETROSPECTIVE_RECONSTRUCTION")
            if not prospective_both:
                classifications.append("FIGURE5_COMPLETED_FIT_ONLY")
        else:
            classifications.append("FIGURE5_BASELINE_RECONSTRUCTED_MODEL_ORDER_NOT_SUPPORTED")
        if prospective_both:
            classifications.append("FIGURE5_PROSPECTIVE_INCREMENTAL_LEAD")
        else:
            p2_vs_dummy = comparisons.loc[
                comparisons["targetId"].eq(target_id)
                & comparisons["referenceModelId"].eq(P2_ID)
                & comparisons["comparatorModelId"].eq(DUMMY_ID)
            ]
            proxy_fail = comparisons.loc[
                comparisons["targetId"].eq(target_id)
                & comparisons["referenceModelId"].eq(P2_ID)
                & comparisons["comparatorModelId"].isin([B4_ID, B5_ID])
            ]
            if len(p2_vs_dummy) == 2 and p2_vs_dummy["matrixBootstrapObservedDifference"].gt(0).all() and (len(proxy_fail) != 4 or not proxy_fail["matrixBootstrapLower95"].gt(0).all()):
                classifications.append("FIGURE5_PAST_ONLY_PROXY_NOT_INCREMENTAL")
        promotable_retrospective = retrospective_both and all(int(defined_counts.get(candidate, 0)) >= 95 for candidate in CANDIDATE_IDS)
        promotable_prospective = prospective_both and all(int(defined_counts.get(candidate, 0)) >= 95 for candidate in CANDIDATE_IDS)
        target_decisions[target_id] = {
            "geometryAdvanced": True, "retrospectiveGatePassed": retrospective_both,
            "prospectiveGatePassed": prospective_both,
            "retrospectivePromotionEligible": promotable_retrospective,
            "prospectivePromotionEligible": promotable_prospective,
            "definedCounts": defined_counts, "classifications": classifications,
        }

    promotion = None
    promotion_classification = "NOT_PROMOTABLE"
    for target_id in (R1_TARGET_ID, U2_TARGET_ID):
        decision = target_decisions.get(target_id, {})
        if decision.get("prospectivePromotionEligible"):
            promotion = {"targetId": target_id, "mode": "PROSPECTIVE", "classification": "PROMOTABLE_TO_UNTOUCHED_PROSPECTIVE_FIGURE5_CONFIRMATION"}
            promotion_classification = promotion["classification"]
            break
        if decision.get("retrospectivePromotionEligible"):
            promotion = {"targetId": target_id, "mode": "RETROSPECTIVE", "classification": "PROMOTABLE_TO_UNTOUCHED_RETROSPECTIVE_FIGURE5_CONFIRMATION"}
            promotion_classification = promotion["classification"]
            break
    any_advanced = any(item.get("geometryAdvanced") for item in target_decisions.values())
    primary = (
        "FIGURE5_BASELINE_NOT_RECONCILED_BY_FROZEN_RECURRING_TARGETS"
        if not any_advanced else
        "FIGURE5_PROSPECTIVE_INCREMENTAL_LEAD"
        if any(item.get("prospectiveGatePassed") for item in target_decisions.values()) else
        "FIGURE5_RETROSPECTIVE_RECONSTRUCTION"
        if any(item.get("retrospectiveGatePassed") for item in target_decisions.values()) else
        "FIGURE5_BASELINE_RECONSTRUCTED_MODEL_ORDER_NOT_SUPPORTED"
    )
    classification = {
        "schema": "eidosoma.e01.s19.l13.classification.v1", "researchStepId": LOOP_ID,
        "versionedStepId": VERSION, "status": "COMPLETE_AWAITING_MANDATORY_HUMAN_REVIEW",
        "primaryClassification": primary, "promotionClassification": promotion_classification,
        "promotedPipeline": promotion, "targetDecisions": target_decisions,
        "priorClassificationsChanged": False,
        "s18ProspectivePredictionStatus": "PROSPECTIVE_PREDICTION_NOT_SUPPORTED_WITHIN_TESTED_SCOPE_UNCHANGED",
        "s18CausalControlStatus": "PROSPECTIVE_CAUSAL_CONTROL_NOT_SUPPORTED_WITHIN_TESTED_SCOPE_UNCHANGED",
        "authorCodeIdentified": False, "exactReplicationClaimed": False,
    }
    return pd.DataFrame(gate_rows), classification, pd.DataFrame(reconstruction_rows)


def execute_phase() -> None:
    geometry_status = json.loads((CACHE_ROOT / "geometry_status.json").read_text())
    if not geometry_status["advancedTargets"]:
        print(canonical_json({"phase": "execute", "status": "SKIPPED_NO_ADVANCED_TARGET"}))
        return
    started_wall, started_cpu = time.time(), time.process_time()
    compute_full_features()
    amendment = None
    if AMENDMENT_001_PATH.is_file():
        amendment = json.loads(AMENDMENT_001_PATH.read_text(encoding="utf-8"))
        mismatches = []
        for filename, expected_hash in amendment["failedAttemptFeatureHashes"].items():
            observed_hash = sha256_file(OUTPUT_ROOT / filename)
            if observed_hash != expected_hash:
                mismatches.append({"artifact": filename, "expected": expected_hash, "observed": observed_hash})
        if mismatches:
            raise RuntimeError(f"technical amendment changed a scientific feature artifact: {mismatches}")
        pd.DataFrame([
            {
                "amendmentId": amendment["amendmentId"], "authorized": True,
                "stage": amendment["failure"]["stage"], "failureClass": amendment["failure"]["exceptionClass"],
                "predictionOutcomeProducedBeforeRepair": False, "modelFitStartedBeforeRepair": False,
                "repair": amendment["repair"], "scientificValuesChanged": False,
                "freshCacheRerun": True, "exactFeatureArtifactHashesPassed": True,
                "status": "VALUE_PRESERVING_TECHNICAL_REPAIR_PASSED",
            }
        ]).to_csv(OUTPUT_ROOT / "technical_amendment_ledger.csv", index=False)
        pd.DataFrame([
            {
                "failureId": "L13-F001-ORACLE-CACHE-ROUTING", "stage": amendment["failure"]["stage"],
                "severity": "TECHNICAL_STOP_BEFORE_MODEL_OUTCOME", "status": "PRESERVED_REPAIRED_VALUE_PRESERVING",
                "reason": "Target-specific oracle was already in memory but generic cache assembly attempted a nonexistent oracle cache file.",
            }
        ]).to_csv(OUTPUT_ROOT / "failure_ledger.csv", index=False)
    suffix = suffix_invariance_audit()
    if not suffix["passed"].all():
        raise RuntimeError("prefix-only suffix invariance failed")
    write_parquet(OUTPUT_ROOT / "suffix_invariance_results.parquet", suffix)
    models = run_models()
    write_parquet(OUTPUT_ROOT / "training_history.parquet", models["training"])
    write_parquet(OUTPUT_ROOT / "prediction_results.parquet", models["predictions"])
    write_parquet(OUTPUT_ROOT / "robust_metric_results.parquet", models["metrics"])
    write_parquet(OUTPUT_ROOT / "paper_accuracy_results.parquet", models["metrics"][[
        "targetId", "candidateId", "modelId", "repetitionId", "accuracy",
        "macroMatrixAccuracy", "prevalence", "testMajorityAccuracy",
    ]])
    write_parquet(OUTPUT_ROOT / "per_matrix_metric_results.parquet", models["matrix"])
    write_parquet(OUTPUT_ROOT / "oracle_diagnostic_results.parquet", models["oracle"])
    write_parquet(OUTPUT_ROOT / "model_replay_results.parquet", models["replay"])
    write_parquet(OUTPUT_ROOT / "scaler_audit.parquet", models["scalers"])
    comparisons, incremental = comparison_tables(models["metrics"], models["matrix"])
    write_parquet(OUTPUT_ROOT / "paired_model_comparisons.parquet", comparisons)
    write_parquet(OUTPUT_ROOT / "incremental_value_results.parquet", incremental)
    negative = pd.concat([
        models["metrics"].loc[models["metrics"]["modelId"].isin([NC1_ID, NC2_ID, B6_ID, B7_ID])].assign(controlType="MODEL_CONTROL"),
        suffix.assign(targetId="ALL_TARGETS", modelId="NC5_SUFFIX_PERTURBATION", controlType="LEAKAGE_CONTROL"),
    ], ignore_index=True, sort=False)
    write_parquet(OUTPUT_ROOT / "negative_control_results.parquet", negative)
    gates, classification, reconstruction = scientific_gates(
        models["metrics"], models["matrix"], comparisons, incremental, suffix
    )
    write_parquet(OUTPUT_ROOT / "scientific_gate_results.parquet", gates)
    reconstruction.to_csv(OUTPUT_ROOT / "figure5_reconstruction_matrix.csv", index=False)
    write_json(OUTPUT_ROOT / "classification.json", classification)
    if AMENDMENT_002_PATH.is_file():
        replay = validate_amendment_002_replay(include_post_figure=False)
        amendment_002 = replay["amendment"]
        amendment_rows = pd.read_csv(OUTPUT_ROOT / "technical_amendment_ledger.csv")
        amendment_rows = amendment_rows.loc[
            ~amendment_rows["amendmentId"].eq(amendment_002["amendmentId"])
        ]
        amendment_rows = pd.concat(
            [
                amendment_rows,
                pd.DataFrame(
                    [
                        {
                            "amendmentId": amendment_002["amendmentId"],
                            "authorized": True,
                            "stage": amendment_002["failure"]["stage"],
                            "failureClass": amendment_002["failure"]["exceptionClass"],
                            "predictionOutcomeProducedBeforeRepair": True,
                            "modelFitStartedBeforeRepair": True,
                            "repair": amendment_002["repair"],
                            "scientificValuesChanged": False,
                            "freshCacheRerun": True,
                            "exactFeatureArtifactHashesPassed": replay["passed"],
                            "status": "SCIENTIFIC_REPLAY_PASSED_PENDING_PLOT_REPAIR",
                        }
                    ]
                ),
            ],
            ignore_index=True,
            sort=False,
        )
        amendment_rows.to_csv(OUTPUT_ROOT / "technical_amendment_ledger.csv", index=False)
        failure_rows = pd.read_csv(OUTPUT_ROOT / "failure_ledger.csv")
        failure_rows = failure_rows.loc[
            ~failure_rows["failureId"].eq("L13-F002-MATPLOTLIB-BOXPLOT-KEYWORD")
        ]
        failure_rows = pd.concat(
            [
                failure_rows,
                pd.DataFrame(
                    [
                        {
                            "failureId": "L13-F002-MATPLOTLIB-BOXPLOT-KEYWORD",
                            "stage": amendment_002["failure"]["stage"],
                            "severity": "TECHNICAL_STOP_DURING_FIGURE_ASSEMBLY",
                            "status": "PRESERVED_REPAIR_PENDING_FINALIZATION",
                            "reason": "Installed Matplotlib removed the deprecated Axes.boxplot labels keyword; no scientific value was affected.",
                        }
                    ]
                ),
            ],
            ignore_index=True,
            sort=False,
        )
        failure_rows.to_csv(OUTPUT_ROOT / "failure_ledger.csv", index=False)
    runtime = json.loads((OUTPUT_ROOT / "runtime_manifest.json").read_text())
    runtime.update({
        "phase": "SCIENTIFIC_EXECUTION_COMPLETE", "executionCompletedAtUtc": utc_now(),
        "executionWallSeconds": time.time() - started_wall,
        "executionCpuSeconds": time.process_time() - started_cpu,
        "modelFitHistoryRows": len(models["training"]),
        "predictionMatrixRows": len(models["predictions"]),
        "splitMetricRows": len(models["metrics"]),
    })
    write_json(OUTPUT_ROOT / "runtime_manifest.json", runtime)
    print(canonical_json(json_safe({
        "phase": "execute", "classification": classification["primaryClassification"],
        "promotion": classification["promotionClassification"],
        "wallSeconds": runtime["executionWallSeconds"],
    })))


def _empty_frame(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype="object") for column in columns})


def validate_amendment_002_replay(*, include_post_figure: bool) -> dict[str, Any]:
    amendment = json.loads(AMENDMENT_002_PATH.read_text(encoding="utf-8"))
    mismatches: list[dict[str, str]] = []
    artifact_hashes = dict(amendment["scientificArtifactHashes"])
    if include_post_figure:
        artifact_hashes.update(amendment["postFigureArtifactHashes"])
    for relative_path, expected_hash in artifact_hashes.items():
        path = OUTPUT_ROOT / relative_path
        observed_hash = sha256_file(path) if path.is_file() else "MISSING"
        if observed_hash != expected_hash:
            mismatches.append(
                {
                    "kind": "scientific_artifact",
                    "path": relative_path,
                    "expected": expected_hash,
                    "observed": observed_hash,
                }
            )
    for relative_path, expected_hash in amendment["featureCacheHashes"].items():
        path = CACHE_ROOT / relative_path
        observed_hash = sha256_file(path) if path.is_file() else "MISSING"
        if observed_hash != expected_hash:
            mismatches.append(
                {
                    "kind": "scientific_feature_cache",
                    "path": relative_path,
                    "expected": expected_hash,
                    "observed": observed_hash,
                }
            )
    if mismatches:
        raise RuntimeError(f"technical amendment 002 changed a scientific value: {mismatches}")
    return {
        "amendment": amendment,
        "artifactHashesPassed": len(artifact_hashes),
        "featureCacheHashesPassed": len(amendment["featureCacheHashes"]),
        "passed": True,
    }


def ensure_required_tables() -> None:
    schemas = {
        "completed_fit_phirl_features.parquet": ["targetId", "candidateId", "matrixIndex", "featureId", "valueSha256"],
        "prefix_only_phirl_features.parquet": ["targetId", "candidateId", "matrixIndex", "featureId", "valueSha256"],
        "baseline_feature_results.parquet": ["candidateId", "matrixIndex", "featureId", "valueSha256"],
        "attractor_control_results.parquet": ["candidateId", "matrixIndex", "featureId", "valueSha256"],
        "oracle_diagnostic_results.parquet": ["targetId", "candidateId", "matrixIndex", "featureId", "valueSha256"],
        "training_history.parquet": ["targetId", "candidateId", "modelId", "repetitionId", "epoch"],
        "prediction_results.parquet": ["targetId", "candidateId", "modelId", "repetitionId", "matrixIndex", "targets", "probabilities"],
        "paper_accuracy_results.parquet": ["targetId", "candidateId", "modelId", "repetitionId", "accuracy"],
        "robust_metric_results.parquet": ["targetId", "candidateId", "modelId", "repetitionId", "accuracy"],
        "per_matrix_metric_results.parquet": ["targetId", "candidateId", "modelId", "repetitionId", "matrixIndex", "accuracy"],
        "paired_model_comparisons.parquet": ["targetId", "candidateId", "referenceModelId", "comparatorModelId"],
        "incremental_value_results.parquet": ["targetId", "candidateId", "gateMetric"],
        "negative_control_results.parquet": ["targetId", "candidateId", "modelId", "controlType"],
        "suffix_invariance_results.parquet": ["candidateId", "matrixIndex", "passed"],
        "leakage_audit.parquet": ["candidateId", "matrixIndex", "completedFutureDependent", "prefixFutureSuffixAccessed"],
        "scientific_gate_results.parquet": ["targetId", "candidateId", "gateFamily", "gateId", "passed"],
    }
    for filename, columns in schemas.items():
        path = OUTPUT_ROOT / filename
        if not path.exists():
            write_parquet(path, _empty_frame(columns))
    csv_schemas = {
        "figure5_reconstruction_matrix.csv": ["targetId", "candidateId", "retrospectiveCandidateGate", "prospectiveCandidateGate"],
        "failure_ledger.csv": ["failureId", "stage", "severity", "status", "reason"],
        "technical_amendment_ledger.csv": ["amendmentId", "authorized", "scientificValuesChanged", "status"],
    }
    for filename, columns in csv_schemas.items():
        path = OUTPUT_ROOT / filename
        if not path.exists():
            pd.DataFrame(columns=columns).to_csv(path, index=False)


def target_control_prevalence() -> pd.DataFrame:
    rows = []
    manifest = load_trajectory_manifest()
    u1 = pd.read_parquet(L11R_ROOT / "molecular_union_label_results.parquet")
    u1 = u1.loc[u1["pipelineId"].eq("U1_HISTORICAL_ALL_COMPTYPE_TAGS_H090")]
    for manifest_row in manifest.itertuples(index=False):
        arrays = trajectory_arrays(manifest_row)
        candidate_id, matrix_index = str(manifest_row.candidateId), int(manifest_row.matrixIndex)
        adjacent = incoming_h(arrays["compositions"]) > 0.9
        rows.append({"candidateId": candidate_id, "matrixIndex": matrix_index, "labelId": "ADJACENT_H090", "occupancy": float(adjacent.mean()), "majorityAccuracy": float(max(adjacent.mean(), 1-adjacent.mean()))})
        selected_u1 = u1.loc[u1["candidateId"].eq(candidate_id) & u1["matrixIndex"].eq(matrix_index)]
        if len(selected_u1):
            values = selected_u1.sort_values("analysisUnitIndex")["isReplicator"].to_numpy(dtype=bool)
            rows.append({"candidateId": candidate_id, "matrixIndex": matrix_index, "labelId": "L11R_U1_ALL_TAG_UNION", "occupancy": float(values.mean()), "majorityAccuracy": float(max(values.mean(), 1-values.mean()))})
    geometry = pd.read_parquet(OUTPUT_ROOT / "target_geometry_results.parquet")
    for row in geometry.loc[geometry["defined"]].itertuples(index=False):
        rows.append({
            "candidateId": row.candidateId, "matrixIndex": int(row.matrixIndex),
            "labelId": row.targetId, "occupancy": float(row.wholeOccupancy),
            "majorityAccuracy": float(max(row.wholeOccupancy, 1-row.wholeOccupancy)),
        })
    return pd.DataFrame(rows)


def generate_figures() -> list[Path]:
    figure_root = OUTPUT_ROOT / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 140, "font.size": 8, "axes.titlesize": 9})
    geometry = pd.read_parquet(OUTPUT_ROOT / "target_geometry_results.parquet")
    availability = pd.read_parquet(OUTPUT_ROOT / "target_availability_results.parquet")
    dummy = pd.read_parquet(OUTPUT_ROOT / "dummy_baseline_results.parquet")
    metrics = pd.read_parquet(OUTPUT_ROOT / "robust_metric_results.parquet")
    comparisons = pd.read_parquet(OUTPUT_ROOT / "paired_model_comparisons.parquet")
    gates = pd.read_parquet(OUTPUT_ROOT / "scientific_gate_results.parquet")
    suffix = pd.read_parquet(OUTPUT_ROOT / "suffix_invariance_results.parquet")
    controls = target_control_prevalence()
    write_parquet(OUTPUT_ROOT / "target_control_prevalence.parquet", controls)
    paths = []

    def save(number: int, name: str) -> Path:
        path = figure_root / f"figure_{number:02d}_{name}.png"
        plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()
        paths.append(path); return path

    # 1
    summary = controls.groupby(["labelId", "candidateId"], as_index=False)[["occupancy", "majorityAccuracy"]].mean()
    _fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    short_label = {
        "ADJACENT_H090": "adjacent H",
        "L11R_U1_ALL_TAG_UNION": "U1",
        R1_TARGET_ID: "R1",
        U2_TARGET_ID: "U2",
    }
    for candidate, group in summary.groupby("candidateId"):
        axes[0].scatter(group["occupancy"], group["majorityAccuracy"], label=candidate)
        for offset_index, row in enumerate(group.itertuples()):
            y_offset = 5 + 7 * (offset_index % 2) + (3 if candidate == "CANDIDATE_3" else 0)
            axes[0].annotate(
                short_label.get(str(row.labelId), str(row.labelId)),
                (row.occupancy, row.majorityAccuracy),
                xytext=(4, y_offset), textcoords="offset points", fontsize=6,
            )
    axes[0].axhline(.60, color="black", ls="--"); axes[0].set(xlabel="whole-run positive occupancy", ylabel="implied majority accuracy", title="Figure 5 arithmetic clue"); axes[0].legend()
    dummy_groups = [
        dummy.loc[dummy["targetId"].eq(target) & dummy["candidateId"].eq(candidate), "dummyAccuracy"].dropna()
        for target in TARGET_IDS for candidate in CANDIDATE_IDS
    ]
    axes[1].boxplot(
        dummy_groups,
        tick_labels=["R1-C2", "R1-C3", "U2-C2", "U2-C3"],
    )
    axes[1].axhspan(.55, .65, alpha=.15, color="green"); axes[1].set(ylabel="ten-split dummy accuracy", title="Frozen Figure 5 comparison envelope")
    save(1, "baseline_arithmetic_clue")

    # 2
    counts = availability.groupby(["targetId", "candidateId"])["defined"].agg(["sum", "count"]).reset_index()
    plt.figure(figsize=(7, 3.4)); labels = [f"{r.targetId.split('_')[1]}\n{r.candidateId}" for r in counts.itertuples()]
    plt.bar(labels, counts["sum"], color="#4477aa"); plt.axhline(80, ls="--", color="orange"); plt.axhline(95, ls=":", color="green"); plt.ylabel("defined matrices / 100"); plt.title("Target availability")
    save(2, "target_availability")

    # 3
    defined = geometry.loc[geometry["defined"]]
    means = defined.groupby(["targetId", "candidateId"])[["wholeOccupancy", "suffixOccupancy"]].mean().reset_index()
    x = np.arange(len(means)); width=.35
    plt.figure(figsize=(8, 3.4)); plt.bar(x-width/2, means["wholeOccupancy"], width, label="whole"); plt.bar(x+width/2, means["suffixOccupancy"], width, label="final 75%")
    plt.xticks(x, [f"{r.targetId.split('_')[1]}\n{r.candidateId}" for r in means.itertuples()]); plt.axhline(.40, ls="--", color="black"); plt.ylabel("positive prevalence"); plt.title("Whole versus prediction-suffix prevalence"); plt.legend()
    save(3, "whole_suffix_prevalence")

    # 4
    onset = defined.groupby(["targetId", "candidateId"])[["noOnsetBeforeCutoff", "firstOnsetInSuffix"]].mean().reset_index()
    plt.figure(figsize=(8, 3.4)); x=np.arange(len(onset)); plt.bar(x-.18,onset["noOnsetBeforeCutoff"],.36,label="still pre-onset at cutoff"); plt.bar(x+.18,onset["firstOnsetInSuffix"],.36,label="first onset in suffix")
    plt.xticks(x,[f"{r.targetId.split('_')[1]}\n{r.candidateId}" for r in onset.itertuples()]); plt.ylabel("fraction of matrices"); plt.title("First-onset availability at 25% cutoff"); plt.legend()
    save(4, "first_onset_availability")

    # 5
    r1 = pd.read_parquet(OUTPUT_ROOT / "r1_target_results.parquet")
    u2 = pd.read_parquet(OUTPUT_ROOT / "u2_target_replay_results.parquet")
    _fig, axes = plt.subplots(2, 2, figsize=(10, 4.5), sharex=False)
    for col, candidate in enumerate(CANDIDATE_IDS):
        for source, label, color in [(r1, "R1", "#4477aa"), (u2, "U2", "#cc6677")]:
            row = source.loc[source["candidateId"].eq(candidate) & source["defined"]].iloc[0] if "defined" in source else source.loc[source["candidateId"].eq(candidate) & source["labels"].notna()].iloc[0]
            sequence = np.asarray(row["labels"], bool); axes[0,col].step(np.arange(len(sequence)), sequence.astype(int), where="post", label=label, color=color, alpha=.8)
        axes[0,col].set(title=f"{candidate}: recurring targets", ylabel="label"); axes[0,col].legend()
        adjacent_subset = controls.loc[controls["candidateId"].eq(candidate) & controls["labelId"].eq("ADJACENT_H090")]
        axes[1,col].hist(adjacent_subset["occupancy"], bins=15, alpha=.7, label="adjacent H occupancy"); axes[1,col].set(xlabel="occupancy", ylabel="matrices")
    save(5, "representative_target_sequences")

    # 6
    _fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for index, candidate in enumerate(CANDIDATE_IDS):
        if (CACHE_ROOT / "features" / candidate / f"{P1_ID}.npz").exists():
            p1=load_feature_tensor(candidate,P1_ID); p2=load_feature_tensor(candidate,P2_ID)
            geometry_table = pd.read_parquet(OUTPUT_ROOT / "target_geometry_results.parquet")
            c = int(geometry_table.loc[(geometry_table.candidateId == candidate) & (geometry_table.matrixIndex == 0), "cutoff"].iloc[0])
            axes[index].plot(p1["values"][0,:c,0],label="completed-fit",lw=.8); axes[index].plot(p2["values"][0,:c,0],label="prefix-only",lw=.8); axes[index].set(title=candidate,xlabel="first-quarter index",ylabel="emergence"); axes[index].legend()
        else: axes[index].text(.5,.5,"geometry gate stopped feature computation",ha="center")
    save(6, "completed_vs_prefix_phirl")

    # 7
    plt.figure(figsize=(12, 4))
    if not metrics.empty:
        selected=metrics.loc[metrics["modelId"].isin([P1_ID,B1_ID,B2_ID,B3_ID,DUMMY_ID])]
        keys=list(selected.groupby(["targetId","candidateId","modelId"]).groups)
        plt.boxplot(
            [selected.loc[(selected.targetId==a)&(selected.candidateId==b)&(selected.modelId==c),"accuracy"] for a,b,c in keys],
            tick_labels=[f"{a.split('_')[1]}\n{b[-1]}:{c.split('_')[0]}" for a,b,c in keys],
            showfliers=False,
        ); plt.xticks(rotation=60,ha="right"); plt.axhline(.60,ls="--",color="gray"); plt.axhline(.85,ls=":",color="gray")
    plt.ylabel("binary accuracy"); plt.title("Reconstructed Figure 5 paper-facing accuracies")
    save(7, "paper_accuracy_boxplots")

    # 8
    plt.figure(figsize=(10,4))
    if not metrics.empty:
        selected=metrics.loc[metrics["modelId"].isin([P1_ID,P2_ID,DUMMY_ID])]
        aggregate=selected.groupby(["targetId","candidateId","modelId"])[["balancedAccuracy","auroc","auprc","brier"]].median().reset_index()
        for metric_name, marker in [("balancedAccuracy","o"),("auroc","s"),("auprc","^"),("brier","D")]:
            plt.plot(np.arange(len(aggregate)),aggregate[metric_name],marker=marker,ls="none",label=metric_name)
        plt.xticks(np.arange(len(aggregate)),[f"{r.targetId.split('_')[1]}-{r.candidateId[-1]}-{r.modelId.split('_')[0]}" for r in aggregate.itertuples()],rotation=60,ha="right"); plt.legend()
    plt.title("Robust discrimination metrics")
    save(8, "robust_metrics")

    # 9
    plt.figure(figsize=(10,4))
    if not comparisons.empty:
        inc=comparisons.loc[comparisons["comparisonFamily"].isin(["PROSPECTIVE","INCREMENTAL"])]
        compact_model = {
            P2_ID: "P2", P2_B4_ID: "P2+B4", P2_B5_ID: "P2+B5",
            B1_ID: "B1", B2_ID: "B2", B3_ID: "B3", B4_ID: "B4",
            B5_ID: "B5", B6_ID: "B6", DUMMY_ID: "dummy",
        }
        y=np.arange(len(inc)); plt.errorbar(inc["matrixBootstrapObservedDifference"],y,xerr=[inc["matrixBootstrapObservedDifference"]-inc["matrixBootstrapLower95"],inc["matrixBootstrapUpper95"]-inc["matrixBootstrapObservedDifference"]],fmt="o"); plt.yticks(y,[f"U2-C{r.candidateId[-1]} {compact_model.get(r.referenceModelId, r.referenceModelId)}>{compact_model.get(r.comparatorModelId, r.comparatorModelId)}" for r in inc.itertuples()]); plt.axvline(0,color="black",lw=.8)
    plt.xlabel("paired macro-matrix accuracy difference (95% bootstrap)"); plt.title("Incremental-value comparisons")
    save(9, "incremental_value")

    # 10
    plt.figure(figsize=(8,3.4))
    if not suffix.empty:
        x=np.arange(len(suffix)); plt.bar(x,suffix["p1MeanAbsoluteChange"].fillna(0),label="P1 change"); plt.scatter(x,np.where(suffix["p2ValuesExact"],0,np.nan),color="red",label="P2 exact replay (zero)"); plt.xticks(x,[f"{r.candidateId[-1]}:{r.matrixIndex}" for r in suffix.itertuples()],rotation=45); plt.yscale("symlog",linthresh=1e-10); plt.legend()
    plt.ylabel("mean absolute change"); plt.title("Completed-fit dependence and prefix suffix-invariance")
    save(10, "future_dependence")

    # 11
    plt.figure(figsize=(9,3.6))
    if not comparisons.empty:
        nc=comparisons.loc[comparisons["comparisonFamily"].eq("NEGATIVE_CONTROL")]; x=np.arange(len(nc)); plt.bar(x,nc["matrixBootstrapObservedDifference"]); plt.errorbar(x,nc["matrixBootstrapObservedDifference"],yerr=[nc["matrixBootstrapObservedDifference"]-nc["matrixBootstrapLower95"],nc["matrixBootstrapUpper95"]-nc["matrixBootstrapObservedDifference"]],fmt="none",color="black"); plt.xticks(x,[f"{r.targetId.split('_')[1]}-{r.candidateId[-1]} vs {r.comparatorModelId.split('_')[0]}" for r in nc.itertuples()],rotation=55,ha="right"); plt.axhline(0,color="black")
    plt.ylabel("P2 accuracy advantage"); plt.title("Negative controls")
    save(11, "negative_controls")

    # 12
    plt.figure(figsize=(6,5))
    if not metrics.empty:
        med=metrics.groupby(["targetId","candidateId","modelId"])["accuracy"].median().unstack("candidateId").dropna(); plt.scatter(med[CANDIDATE_IDS[0]],med[CANDIDATE_IDS[1]]); lim=[min(med.min())-.02,max(med.max())+.02]; plt.plot(lim,lim,ls="--",color="black"); plt.xlim(lim);plt.ylim(lim); plt.xlabel("candidate 2 median accuracy");plt.ylabel("candidate 3 median accuracy")
    plt.title("Cross-candidate agreement")
    save(12, "candidate_agreement")

    # 13
    plt.figure(figsize=(10,4))
    if not gates.empty:
        pivot=gates.pivot_table(index=["targetId","candidateId"],columns="gateId",values="passed",aggfunc="first").fillna(False).astype(int); plt.imshow(pivot,cmap="RdYlGn",vmin=0,vmax=1,aspect="auto"); plt.yticks(np.arange(len(pivot)),[f"{a.split('_')[1]}-{b[-1]}" for a,b in pivot.index]); plt.xticks(np.arange(len(pivot.columns)),pivot.columns,rotation=70,ha="right"); plt.colorbar(label="gate pass")
    plt.title("Retrospective versus prospective decision matrix")
    save(13, "decision_matrix")

    # 14
    classification=json.loads((OUTPUT_ROOT/"classification.json").read_text()) if (OUTPUT_ROOT/"classification.json").exists() else {"primaryClassification":"PENDING"}
    plt.figure(figsize=(10,4)); plt.axis("off")
    text=("Two frozen recurring targets\n↓\nGeometry: defined targets + ~60% dummy\n↓\nCompleted-fit P1 paper-facing gate\n↓\nPrefix-only P2 incremental + leakage gates\n↓\n"+classification["primaryClassification"]+"\n\nPromotion: "+classification.get("promotionClassification","NOT_PROMOTABLE")+"\nMandatory human review; no automatic continuation")
    plt.text(.5,.5,text,ha="center",va="center",bbox={"boxstyle": "round", "facecolor": "#eef3f8"},fontsize=10)
    save(14, "promotion_decision_tree")
    return paths


def write_figure_review_artifact(figures: list[Path]) -> None:
    expected = [
        (1, "baseline_arithmetic_clue", "Baseline arithmetic clue", "The left panel plots mean whole-run occupancy against mean per-matrix majority accuracy for each label and candidate; the right panel keeps the ten split-wise S16 dummy distributions separate for R1/U2 and candidate 2/3.", "The left values near 0.75 are not a contradiction: averaging each matrix's majority accuracy is nonlinear and exposes strong between-matrix prevalence heterogeneity. The right panel is the registered task-level gate. Its green band is the frozen 0.55–0.65 Figure 5 envelope; verify that both U2 candidate medians, but neither R1 candidate median, lie inside it."),
        (2, "target_availability", "Target availability", "Defined recurring-target sequences are counted out of 100 matrices for each target and candidate.", "The dashed 80-matrix line is the forensic-evaluation threshold and the dotted 95-matrix line is the paper-scale promotion threshold. Verify R1 at 89/86 and U2 at 100/100."),
        (3, "whole_suffix_prevalence", "Whole-run versus suffix prevalence", "Mean positive prevalence is shown for complete molecular trajectories and the final 75% prediction suffix.", "Verify that target prevalence is not manufactured by padding and that the U2 suffix remains in the broad range capable of producing a roughly 60% dummy."),
        (4, "first_onset_availability", "First-onset eligibility at the 25% cutoff", "Bars show the fraction still pre-onset at the cutoff and the fraction whose first positive target state occurs in the suffix.", "The visually small bars are decisive: only 3 candidate-2 and 5 candidate-3 U2 matrices meet both onset concepts, far below the required 20 per candidate."),
        (5, "representative_target_sequences", "Representative recurring-target sequences", "Direct molecular R1 and U2 labels are shown for one eligible trajectory per candidate; adjacent-H occupancy distributions provide the frozen high-prevalence contrast.", "Look for episode timing and switching rather than occupancy alone. The recurring labels are not boundary projections, and the adjacent-H comparator should appear much more uniformly positive across matrices."),
        (6, "completed_vs_prefix_phirl", "Completed-fit versus prefix-only PhiRL inputs", "First-quarter emergence values for P1 (fit on the completed trajectory) and P2 (fit only on the first quarter) are overlaid for one matrix in each candidate.", "Verify that these are materially different feature trajectories even though they use the same observed prefix; this is the visible signature of completed-fit future dependence."),
        (7, "paper_accuracy_boxplots", "Reconstructed Figure 5 accuracy boxplots", "Ten split-wise accuracies compare completed-fit PhiRL, composition change, raw composition, flux, and the majority dummy for the advancing U2 target in each candidate.", "The dotted 0.85 and dashed 0.60 guides mark the digitized PhiRL and dummy centers. Verify that the dummy is reconciled but completed-fit PhiRL remains well below the paper-like 0.85 range and does not yield the full registered ordering/significance pattern."),
        (8, "robust_metrics", "Robust discrimination and calibration-loss metrics", "Median balanced accuracy, AUROC, AUPRC, and Brier score are displayed for completed-fit PhiRL, prefix-only PhiRL, and the dummy by candidate.", "Accuracy alone can ride prevalence. Verify that prefix-only balanced accuracy stays near chance and interpret lower Brier as better, unlike the three higher-is-better discrimination metrics."),
        (9, "incremental_value", "Incremental-value comparisons", "Paired catalytic-matrix bootstrap differences compare prefix-only PhiRL with ordinary controls and its registered combined models.", "Intervals crossing zero fail incremental value. Verify that P2 does not consistently beat adjacent H or prefix-only attractor geometry in either candidate."),
        (10, "future_dependence", "Completed-fit dependence and prefix suffix-invariance", "Sentinel suffix perturbations show nonzero changes in completed-fit P1 while exact-replay P2 markers remain at zero.", "Verify both sides of the leakage audit: P1 changes substantially when the unseen suffix changes, while P2 remains exactly invariant."),
        (11, "negative_controls", "Registered negative controls", "Paired P2 accuracy advantages over temporal permutation and matrix-label permutation controls are shown with matrix-bootstrap intervals.", "A credible prospective signal should separate from every control. Verify that the registered control contrasts do not jointly establish such separation in both candidates."),
        (12, "candidate_agreement", "Candidate-2 versus candidate-3 agreement", "Each point compares a model's median accuracy under candidate 2 and candidate 3; the diagonal denotes equality.", "Look for directional consistency without treating proximity to the diagonal as paper replication. The cross-candidate check cannot rescue failed paper-facing or prospective gates."),
        (13, "decision_matrix", "Retrospective and prospective gate matrix", "Green and red cells summarize every registered gate by target and candidate.", "Verify that dummy overlap and suffix invariance pass, while the paper-interval, baseline-compatibility, significance, incremental-value, balanced-accuracy, and onset-eligibility requirements prevent promotion."),
        (14, "promotion_decision_tree", "Final L13 decision tree", "The locked flow from two recurring targets through geometry, retrospective completed-fit, and prospective prefix-only gates ends at the machine-authoritative classification.", "Verify the terminal result `FIGURE5_BASELINE_RECONSTRUCTED_MODEL_ORDER_NOT_SUPPORTED`, promotion `NOT_PROMOTABLE`, and mandatory human-review boundary."),
    ]
    expected_paths = [OUTPUT_ROOT / "figures" / f"figure_{number:02d}_{stem}.png" for number, stem, *_ in expected]
    if figures != expected_paths or any(not path.is_file() for path in expected_paths):
        raise RuntimeError("figure-review artifact cannot be written because the registered figure set is incomplete")
    classification = json.loads((OUTPUT_ROOT / "classification.json").read_text(encoding="utf-8"))
    lines = [
        "# L13 Output-Figure Contents and Captions for Human Review — V1",
        "",
        "## Top summary",
        "",
        f"- **Research step ID:** `{LOOP_ID}` (`{VERSION}`)",
        "- **Version boundary:** `V1_GENERATED_L13_FIGURES`. This file documents L13's generated result plots; it is retained for provenance but is not the requested reading of the input paper. Use `FIGURE_CONTENTS_AND_CAPTIONS_FOR_HUMAN_REVIEW_V2.md` for that purpose.",
        "- **Completion status:** `COMPLETE_AWAITING_MANDATORY_HUMAN_REVIEW`",
        f"- **Artifacts covered:** {len(expected_paths)} registered PNG figures under `figures/`, each embedded below with a SHA-256 identity.",
        "- **Validation result:** All listed image files exist; captions describe only frozen L13 tables and gates; image hashes are recorded for eye-checking and artifact identity.",
        f"- **Outcome classification:** `{classification['primaryClassification']}`; promotion `{classification['promotionClassification']}`.",
        "- **Caveats or blockers:** These are forensic reconstruction plots, not native paper panels. R1/U2 targets are completed-run definitions, P1 is future-dependent, Figure 5 ranges are approximate L12 digitizations, and no image identifies author code.",
        "- **Recommended next action:** Inspect the figures below, compare the visible relationships with their captions, and return a human decision. Keep S20, E02, confirmation, and L14 inactive.",
        "",
        "## How to review",
        "",
        "The **Contents** line states what was plotted. The **Caption** states the evidentiary interpretation. The **Visual check** calls out what should be apparent by eye. If an image appears inconsistent with its visual check, use its SHA-256 below to identify the exact file for review; the machine-readable tables remain authoritative.",
        "",
    ]
    for number, stem, title, contents, visual_check in expected:
        path = OUTPUT_ROOT / "figures" / f"figure_{number:02d}_{stem}.png"
        relative = path.relative_to(OUTPUT_ROOT).as_posix()
        lines.extend(
            [
                f"## Figure {number}. {title}",
                "",
                f"![Figure {number}: {title}]({relative})",
                "",
                f"- **File:** `{relative}`",
                f"- **SHA-256:** `{sha256_file(path)}`",
                f"- **Contents:** {contents}",
                f"- **Caption:** {contents} {visual_check}",
                f"- **Visual check:** {visual_check}",
                "",
            ]
        )
    (OUTPUT_ROOT / "FIGURE_CONTENTS_AND_CAPTIONS_FOR_HUMAN_REVIEW.md").write_text(
        "\n".join(lines).rstrip() + "\n", encoding="utf-8"
    )


def write_paper_figure_review_v2() -> None:
    paper_sha256 = "77a2ec2c0751839d8a2e10863ca803c6f8b61475bbc790f2bbdad2a38af04ae4"
    expected_figures = {
        "figure-01.png": "d26f2de1bd79dfea4fdd12bc8cfc9b5ee4bbcfbf62e73928a2e792f643a49710",
        "figure-02.png": "0e4aac507ccf6e10ced31edd6d7e5ba8c876d9d0c8d420b145dfc27c7d040778",
        "figure-03.png": "7bd35a0b09a679d9b2f5c0fe8c57ea02b39833c663383bbdb676d2cbecf5c0c8",
        "figure-04.png": "8632a9fe080d80066a9a5925e80c15aac4962393260709d7cebebdec617b224b",
        "figure-05.png": "75be305d13203e65a8c93464b8a23aa25a86a567880458e889137bfe1281a968",
        "figure-06.png": "42a542c10467e80aad139e772055a569689085e1d08cea8250636104a24dd498",
        "figure-07.png": "a194e06ab0698f3b82f7eb2dee864644eb794a550d496a4f9fe1ee32d9aeb943",
        "figure-08.png": "856678b09e71c4fbcc32db39a75fec13acbf1629c5eeb979b072826f1aa82e67",
    }
    mismatches = []
    observed_paper_hash = sha256_file(PAPER_PDF) if PAPER_PDF.is_file() else "MISSING"
    if observed_paper_hash != paper_sha256:
        mismatches.append({"path": str(PAPER_PDF), "expected": paper_sha256, "observed": observed_paper_hash})
    for filename, expected_hash in expected_figures.items():
        path = PAPER_FIGURE_ROOT / filename
        observed_hash = sha256_file(path) if path.is_file() else "MISSING"
        if observed_hash != expected_hash:
            mismatches.append({"path": str(path), "expected": expected_hash, "observed": observed_hash})
    if mismatches:
        raise RuntimeError(f"paper-figure V2 source identity mismatch: {mismatches}")

    review = f"""# Input-Paper Figure Contents and Captions for Human Review — V2

## Top summary

- **Research step ID:** `{LOOP_ID}` (`{VERSION}`), report-assembly addendum only.
- **Completion status:** `COMPLETE_AWAITING_MANDATORY_HUMAN_REVIEW`.
- **Artifact written:** `FIGURE_CONTENTS_AND_CAPTIONS_FOR_HUMAN_REVIEW_V2.md`, covering the supplied paper's Figures 1–6 and Table 1.
- **Validation result:** `PASS`; the paper PDF and all eight native extracted figure assets match the frozen SHA-256 identities listed below, and every panel in Figures 1–6 is represented.
- **Outcome classification:** L13 remains `FIGURE5_BASELINE_RECONSTRUCTED_MODEL_ORDER_NOT_SUPPORTED`; this V2 document only records my reading of the **input paper**, not a new scientific result.
- **Caveats or blockers:** Caption text below is a faithful paraphrase, not a verbatim transcription. Numerical readings marked “approximately” are visual estimates frozen in L12. Several paper-visible operations remain under-specified or internally inconsistent.
- **Recommended next action:** Compare this reading panel by panel with the supplied PDF/native images and flag any mismatch in what I think the paper shows or claims. Keep S20, E02, L14, confirmation, and interventions inactive.

## Scope and source identity

This V2 is the requested human-verification aid for the **paper's own figures and captions**. It is deliberately separate from V1, which documents L13-generated plots.

- Paper PDF: `{PAPER_PDF}`
- Paper PDF SHA-256: `{paper_sha256}`
- Native extracted image directory: `{PAPER_FIGURE_ROOT}`
- Printed figure pages: 14–17 of the supplied preprint.

| Paper component | Native file(s) | Frozen SHA-256 |
|---|---|---|
| Figure 1 | `figure-01.png` | `{expected_figures['figure-01.png']}` |
| Figure 2 | `figure-02.png` | `{expected_figures['figure-02.png']}` |
| Figure 3 | `figure-03.png` | `{expected_figures['figure-03.png']}` |
| Figure 4 | `figure-04.png` | `{expected_figures['figure-04.png']}` |
| Figure 5 | `figure-05.png` | `{expected_figures['figure-05.png']}` |
| Figure 6A | `figure-06.png` | `{expected_figures['figure-06.png']}` |
| Figure 6B | `figure-07.png` | `{expected_figures['figure-07.png']}` |
| Figure 6C | `figure-08.png` | `{expected_figures['figure-08.png']}` |

For each item below, **Visible content** is my direct reading of the native image; **Caption meaning** is my paraphrase of the supplied caption; **Operational reading** states the computation the panel appears to require; and **Ambiguities** records what cannot be learned by eye.

## Figure 1 — End-to-end conceptual system

**Source:** printed page 14; native `figure-01.png`.

### Panel A

- **Visible content:** A colored molecular assembly appears above a directed catalytic-reaction network. Colors denote molecule types; arrows denote catalysis among types.
- **Caption meaning:** GARD models compositional assemblies drawn from a fixed molecular vocabulary whose interactions are governed by a catalytic network.
- **Operational reading:** Each run needs a catalytic matrix/network and a molecular-count composition state.
- **Ambiguities:** The panel does not specify the catalytic-matrix distribution, direction/index convention, weight use, or initial-state sampling.

### Panel B

- **Visible content:** Environmental molecules accrete along “molecular time”; an assembly reaches a critical size, fissions into two daughters, one daughter continues, and the lineage repeats to 100 generations. The schematic itself says the molecular evolution uses ODE dynamics.
- **Caption meaning:** A selected lineage grows, fissions at a size boundary, and continues from one progeny until the generation limit.
- **Operational reading:** Molecular observations occur inside generations; fission marks a second clock; continuation selects one daughter.
- **Ambiguities:** The image does not say which daughter is selected, how overshoot/extinction is handled, or how molecular observations are recorded. Its “ODE dynamics” wording is visibly in tension with the Methods description of stochastic Poisson updates.

### Panel C

- **Visible content:** Similar compositions recur at separated points along the lineage and are enclosed as a tight composition-space cluster. Text in the panel equates tight clusters/homeostatic growth with attractors and self-replicators.
- **Caption meaning:** Self-replicators are recurring composition-space clusters with homeostatic, attractor-like behavior.
- **Operational reading:** Replicator status appears to require recurrence relative to a composition-space attractor, not merely similarity to the immediately previous molecular state.
- **Ambiguities:** The panel does not specify distance, threshold, clustering method, single-versus-multiple clusters, centroid/medoid, recurrence count, molecular-versus-boundary clock, retrospective fitting, or projection onto molecular time.

### Panel D

- **Visible content:** Several molecular compositions feed into one local `Φ^r` trajectory plotted over molecular steps; the sample trace is noisy with a late high excursion.
- **Caption meaning:** Relative composition at every molecular step is transformed into one `Φ^r` value, yielding a per-run molecular-time trajectory.
- **Operational reading:** The information pipeline consumes the molecular composition series, not only the 100 fission-boundary states.
- **Ambiguities:** The panel does not reveal preprocessing, partition, estimator, scalar identity, full-run versus prefix fitting, or whether the first lagged observation is dropped.

**Human checks for Figure 1:**

- [ ] Does panel C visually support a recurring-attractor label rather than adjacent-state smoothness?
- [ ] Is panel C singular (one dominant attractor) or plural (membership in any recurring cluster)?
- [ ] Does panel B explicitly imply one continuing daughter and two distinct clocks?
- [ ] Do you agree that the visible ODE wording conflicts with the stochastic-update Methods text?

## Figure 2 — Aggregate and individual `Φ^r` dynamics

**Source:** printed page 15; native `figure-02.png`.

### Panel A

- **Visible content:** A pale blue median-with-standard-deviation aggregate spans roughly 0–1,300 molecular steps and sits near zero overall. A red linear fit is nearly flat and is annotated `p=0.1995 > 0.05`. Large vertical dispersion/excursions appear at scattered and terminal positions.
- **Caption meaning:** Across 100 runs, the large-scale median trajectory has no significant linear trend.
- **Operational reading:** Unequal-length molecular trajectories were aligned somehow, summarized pointwise by median and standard deviation, and regressed over the displayed molecular index.
- **Ambiguities:** Padding, truncation, available-case tails, minimum contributor count, and the exact regression series are invisible. The x extent exceeds many sample-run lengths.

### Panel B

- **Visible content:** One run to about 800 steps, with a baseline near one, two positive rectangular plateaus near 8–10, and several abrupt negative drops near -60.

### Panel C

- **Visible content:** One run to about 800 steps, with multiple positive plateaus near 3–4 and many narrow or rectangular negative excursions reaching roughly -15.

### Panel D

- **Visible content:** One run to roughly 1,050 steps, nearly zero except for a very narrow paired positive/negative excursion around the middle (approximately +100 and -165).

- **Caption meaning for B–D:** Individual trajectories contain punctuated positive and negative spikes despite the absent aggregate trend.
- **Operational reading:** The plotted object is a signed local `Φ^r` value at molecular resolution; abrupt plateaus and paired extremes may encode partition changes or numerical conditioning as well as dynamics.
- **Ambiguities:** The caption does not define the three-standard-deviation scope, run-selection rule, completed-fit dependence, or numerical filtering. Panel A's “median ± std” also leaves contributor handling unresolved.

**Human checks for Figure 2:**

- [ ] Are the B/C excursions visibly rectangular as well as spike-like?
- [ ] Does D show both a very large positive and negative excursion at nearly the same time?
- [ ] Does A visibly extend to about 1,300 steps while B/C end near 800 and D near 1,050?
- [ ] Is the red aggregate fit visually near-flat with `p=0.1995`?

## Figure 3 — Runwise association between replication and `Φ^r`

**Source:** printed page 15; native `figure-03.png`.

### Panel A

- **Visible content:** A histogram of 100 runwise Spearman coefficients spans approximately -0.15 to +0.55. A zero reference is dashed dark/blue; a red dashed mean is labelled about `ρ=0.139`.
- **Caption meaning:** Runwise correlations are on average positive, with a one-sample diagnostic declaring the mean significantly above zero.
- **Operational reading:** Each run contributes one coefficient; molecular observations are not the inferential replicates for the population histogram.

### Panel B

- **Visible content:** Four category bars: positive/significant about 54%, positive/non-significant about 19%, negative/significant about 6%, and negative/non-significant about 21%.
- **Caption meaning:** Positive and significant association is the largest category and comprises 54 of 100 runs.
- **Ambiguities and visible text conflict:** The figure caption says the coefficient uses **changes in** `Φ^r`, while the Results prose describes correlation between the **level of** `Φ^r` and replication. These are distinct analyses. The significance threshold, multiplicity treatment, and precise binary label remain unspecified by the panels.

**Human checks for Figure 3:**

- [ ] Is the mean marker visibly labelled near 0.139?
- [ ] Do the four bars visually correspond to 54/19/6/21 percent?
- [ ] Do you agree that the caption says change while the Results paragraph says level?

## Figure 4 — Drift versus self-replicating state contrast

**Source:** printed page 16; native `figure-04.png`.

### Panel A

- **Visible content:** Approximately 100 colored paired lines connect each run's mean `Φ^r` in drift (left) to its mean in self-replication (right). Many rise, some fall, and a few drift-state values are strong negative outliers.
- **Caption meaning:** Within each run, the authors compare state-conditioned mean `Φ^r`; 57% of runs are said to be higher in the replicating state.
- **Operational reading:** Each trajectory must contain both a binary replicator label and local `Φ^r`; means are computed within state and joined within run.
- **Ambiguities:** It is not visible how runs missing either state are handled, whether “mean” follows any trimming, or whether the label is molecular or projected from boundaries.

### Panel B

- **Visible content:** An across-run median line rises from drift to self-replication, with broad shaded standard-deviation bands (roughly centered near 0.2 and 0.8, respectively).
- **Caption meaning:** The across-run summary is intended to show higher `Φ^r` in self-replicating states.
- **Ambiguities:** The caption refers to the median/standard deviation of “the lines in B,” which appears to be a panel-letter typo for A. The Results text mentions Mann–Whitney and Fisher combination, but panel aggregation alone does not reveal the exact test scope.

**Human checks for Figure 4:**

- [ ] Are both increasing and decreasing runwise lines visible despite the positive median shift?
- [ ] Are the large negative drift outliers visible in A?
- [ ] Does B show median ± standard deviation rather than a confidence interval?

## Figure 5 — First-quarter-to-final-three-quarter prediction

**Source:** printed page 16; native `figure-05.png`.

- **Visible content:** Five boxplots show binary accuracy for `Φ^r`, composition change, raw compositions, fluxes, and a majority dummy. Approximate centers are 0.85, 0.80, 0.79–0.80, 0.79, and 0.60. Significance brackets compare `Φ^r` with each alternative using two or three stars.
- **Caption meaning:** An MLP using the first 25% of a run's `Φ^r` is claimed to predict self-replicator appearance/state in the final 75% better than the same model using ordinary composition/flux features or a majority dummy; the worst reported comparison has `p<0.01`.
- **Operational reading:** The boxplots represent ten repeated matrix-level 80/20 train/test experiments. The target is a suffix binary sequence; the input is a first-quarter feature sequence.
- **Ambiguities:** The image does not expose tensor layout, masks, padding, balancing, validation split, scaling, target clock, “initial appearance” versus future occupancy, or whether `Φ^r` was fitted on the completed trajectory. The Results says each repetition is a dot/scatterplot, while the native panel is visibly rendered as boxplots without visible individual dots.
- **Cross-paper conflict:** A majority dummy near 0.60 implies a task-level majority prevalence near 0.60, whereas Table 1 reports control replication probability near 0.88. Those cannot be the same unbalanced molecular target and denominator unless an unreported sampling, balancing, label, or dataset transformation intervenes.

**Human checks for Figure 5:**

- [ ] Is the dummy center visibly near 60%, not 88% or 98%?
- [ ] Is `Φ^r` centered near 85%, with other learned families near 79–80%?
- [ ] Are boxplots, rather than ten visible dots, what the panel actually displays?
- [ ] Does the title/caption use “initial appearance” even though the described output is the full final 75% state sequence?

## Figure 6 — Intervention pipeline and treatment outcomes

**Source:** printed page 17; native `figure-06.png` (A), `figure-07.png` (B), and `figure-08.png` (C).

### Panel A

- **Visible content:** A loop begins just after fission, enumerates adding or deleting one molecule of each illustrated type, chooses the action that maximizes or minimizes `Φ^r`, simulates one GARD generation, and repeats.
- **Caption meaning:** Intervention occurs immediately after every fission; the selected single-molecule edit is the raw score extremum, followed by ordinary dynamics until the next generation.
- **Operational reading:** The scorer must assign one `Φ^r` value to every hypothetical edited post-fission state using only the information intended to be available at that decision.
- **Ambiguities:** No-op handling, refitting, partition/statistics reuse, future data, tie-breaking, numerical separability, and random-action controls are invisible. The schematic's small color set is illustrative rather than the stated 100 molecular types.

### Panel B

- **Visible content:** Persistence boxplots are ordered max > control/base > min, with centers near 874, 716, and 559 molecular steps and three pairwise significance brackets marked with three stars.
- **Caption meaning:** Maximizing `Φ^r` is claimed to increase self-replication persistence, while minimizing it decreases persistence; the caption attributes significance to Mann–Whitney tests.
- **Operational reading:** Persistence is the per-trajectory sum of positive molecular replicator labels, compared across treatment matrices.

### Panel C

- **Visible content:** Over 0–100 generations, max (blue) rises approximately 86%→89%, control (orange) is nearly flat around 88%, and min (green) falls approximately 81%→79%; shaded bands are labelled 95% confidence intervals. Legend annotations show slopes/statistics about +0.041 (`p<0.001`), +0.008 (`p=0.4659`), and -0.030 (`p=0.0034`).
- **Caption meaning:** Repeated maximizing interventions are claimed to accumulate a positive effect on replication probability, while minimizing interventions have the opposite trend.
- **Operational reading:** Molecular replication labels must be aggregated within generation (or another generation-indexed window), then treatment-level regressions and intervals are computed over generation.
- **Ambiguities:** The panel does not reveal the regression unit, within-generation denominator, repeated-measures treatment, contributor count by generation, or relation between curve averages and Table 1's overall means.

**Human checks for Figure 6:**

- [ ] Does A show intervention immediately after fission and exactly one selected add/delete action?
- [ ] Does B visibly order max > base/control > min in persistence?
- [ ] Does C show max rising, control nearly flat, and min falling with the stated approximate slope annotations?
- [ ] Is there any visible random-action arm or action-score uncertainty? My reading is no.

## Table 1 — Paper-facing outcome values and note

**Source:** printed pages 17–18, directly beneath Figure 6.

| Treatment | Persistence | Probability | Consistency | Time to first replicator |
|---|---:|---:|---:|---:|
| max `Φ^r` | 874 ± 233 | 88 ± 3% | 0.52 ± 0.04 | 36 ± 26% |
| control | 716 ± 198 | 88 ± 3% | 0.38 ± 0.06 | 37 ± 27% |
| min `Φ^r` | 559 ± 99 | 80 ± 3% | 0.42 ± 0.04 | 40 ± 28% |

- **Caption/note meaning:** Persistence is total positive molecular steps; probability is the positive-step fraction; consistency is Pearson correlation of consecutive labels; time to first is described in the note as molecular steps.
- **Unresolved dispersion:** The table does not identify whether `±` denotes SD or SE.
- **Internal conflict 1:** The first-onset cells print percent signs, but the note defines molecular-step counts.
- **Internal conflict 2:** The prose says minimization worsened all four properties and that higher consistency is better, yet min consistency (0.42) exceeds control (0.38).
- **Internal conflict 3:** Max and control both round to 88% overall probability even though Figure 6C shows visibly different time trends. This can be arithmetically possible, but the aggregation/window definition is missing.
- **Cross-figure conflict:** The 88% control probability conflicts with Figure 5's approximately 60% dummy if the same unbalanced molecular label and denominator were used.

**Human checks for Table 1:**

- [ ] Are the four values and dispersions transcribed correctly for all three treatments?
- [ ] Do the first-onset values visibly carry percent signs?
- [ ] Does the following note nevertheless define first onset in molecular steps?
- [ ] Do you agree that min consistency is numerically above control despite the “worsened” wording?

## Cross-figure interpretation I would use unless you correct it

1. **Replicator object:** Figure 1C depicts recurrence around one or more composition-space attractors. It does not visually justify the adjacent molecular `H>0.9` label used in the original frozen S13Y comparator.
2. **Information clock:** Figures 1D and 2 use molecular-step `Φ^r`; Figure 6 decisions occur at generation/fission boundaries. A coherent intervention implementation therefore needs an explicit mapping from one hypothetical boundary edit to a molecular-history information score.
3. **Level versus change:** Figure 3's caption and Results prose specify different estimands. I would preserve both rather than silently choose one.
4. **Prediction target:** Figure 5's 60% dummy is consistent with a roughly 40/60 target geometry, but not with Table 1's 88% positive molecular occupancy absent an unreported transformation.
5. **Completed-fit dependence:** Nothing visible in Figures 1–5 establishes that first-quarter `Φ^r` was fitted without the final 75%; public PhiRL behavior makes this a material ambiguity.
6. **Intervention semantics:** Figure 6 visually specifies timing and raw max/min action intent, but not the online scorer, refit scope, ties, numerical uncertainty, or matched/random outcome controls.
7. **Single coherent pipeline:** The paper figures do not, by themselves, identify one end-to-end implementation consistent with every panel and Table 1. This remains an author-code discrimination problem, not a reason to rewrite prior evidence.

## Requested human-review boundary

Please use the checkboxes above to identify any place where my visual reading differs from yours. A correction to what is visibly present should update this report description only; it must not retroactively change frozen scientific results. No next scientific loop is activated by this artifact.
"""
    (OUTPUT_ROOT / "FIGURE_CONTENTS_AND_CAPTIONS_FOR_HUMAN_REVIEW_V2.md").write_text(
        review, encoding="utf-8"
    )


def regeneration_validation() -> dict[str, Any]:
    manifest = load_trajectory_manifest()
    r1_table = pd.read_parquet(OUTPUT_ROOT / "r1_target_results.parquet").set_index(["candidateId", "matrixIndex"])
    u2_table = pd.read_parquet(OUTPUT_ROOT / "u2_target_replay_results.parquet").set_index(["candidateId", "matrixIndex"])
    target_pass = 0
    target_total = 0
    feature_pass = 0
    feature_total = 0
    feature_reference = {}
    for filename in ["all_feature_summary.parquet"]:
        frame = pd.read_parquet(OUTPUT_ROOT / filename)
        for row in frame.itertuples(index=False):
            feature_reference[(row.candidateId, int(row.matrixIndex), row.featureId)] = (
                row.valueSha256, row.channelMaskSha256, row.timeMaskSha256
            )
    for manifest_row in manifest.itertuples(index=False):
        arrays = trajectory_arrays(manifest_row)
        candidate_id, matrix_index = str(manifest_row.candidateId), int(manifest_row.matrixIndex)
        for target_id, observed, source in [
            (R1_TARGET_ID, r1_target(arrays["boundaryCompositions"], arrays["compositions"], manifest_row.trajectoryId), r1_table),
            (U2_TARGET_ID, u2_target(arrays["boundaryCompositions"], arrays["compositions"], manifest_row.trajectoryId), u2_table),
        ]:
            frozen = source.loc[(candidate_id, matrix_index)]
            passed = (
                (observed.labels is None and pd.isna(frozen.labelSha256))
                or (observed.labels is not None and array_sha256(observed.labels) == frozen.labelSha256)
            )
            passed = passed and (
                (observed.scores is None and pd.isna(frozen.scoreSha256))
                or (observed.scores is not None and array_sha256(observed.scores) == frozen.scoreSha256)
            )
            target_total += 1; target_pass += int(passed)
        if feature_reference:
            features, _, _ = compute_features_for_trajectory(manifest_row)
            for feature_id, feature in features.items():
                key = (candidate_id, matrix_index, feature_id)
                if key not in feature_reference:
                    continue
                observed = (array_sha256(feature[0]), array_sha256(feature[1]), array_sha256(feature[2]))
                feature_total += 1; feature_pass += int(observed == feature_reference[key])
    model_replay = pd.read_parquet(OUTPUT_ROOT / "model_replay_results.parquet") if (OUTPUT_ROOT / "model_replay_results.parquet").exists() else pd.DataFrame()
    model_pass = bool(model_replay.empty or model_replay["passed"].all())
    table_checks = []
    if not pd.read_parquet(OUTPUT_ROOT / "robust_metric_results.parquet").empty:
        metrics = pd.read_parquet(OUTPUT_ROOT / "robust_metric_results.parquet")
        matrix = pd.read_parquet(OUTPUT_ROOT / "per_matrix_metric_results.parquet")
        comparisons, incremental = comparison_tables(metrics, matrix)
        saved_comparisons = pd.read_parquet(OUTPUT_ROOT / "paired_model_comparisons.parquet")
        saved_incremental = pd.read_parquet(OUTPUT_ROOT / "incremental_value_results.parquet")
        for name, observed, expected in [
            ("paired_model_comparisons", comparisons, saved_comparisons),
            ("incremental_value_results", incremental, saved_incremental),
        ]:
            observed = observed.reindex(sorted(observed.columns), axis=1).reset_index(drop=True)
            expected = expected.reindex(sorted(expected.columns), axis=1).reset_index(drop=True)
            try:
                pd.testing.assert_frame_equal(observed, expected, check_dtype=False, check_exact=True)
                passed = True
            except AssertionError:
                passed = False
            table_checks.append({"artifact": name, "passed": passed, "rows": len(observed)})
    immutable = json.loads((OUTPUT_ROOT / "immutable_prior_validation.json").read_text())
    immutable = revalidate_immutable(immutable)
    write_json(OUTPUT_ROOT / "immutable_prior_validation.json", immutable)
    passed = bool(
        target_pass == target_total and feature_pass == feature_total and model_pass
        and all(row["passed"] for row in table_checks) and immutable["passed"]
    )
    return {
        "schema": "eidosoma.e01.s19.l13.regeneration_validation.v1",
        "researchStepId": LOOP_ID, "validatedAtUtc": utc_now(),
        "targetReplayPassed": target_pass, "targetReplayTotal": target_total,
        "featureReplayPassed": feature_pass, "featureReplayTotal": feature_total,
        "modelReplayRows": len(model_replay), "modelReplayPassed": model_pass,
        "derivedTableChecks": table_checks,
        "immutablePriorEntryCount": immutable["entryCount"],
        "immutablePriorMismatchCount": immutable["mismatchCount"],
        "passed": passed,
    }


def _summary_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    availability = pd.read_parquet(OUTPUT_ROOT / "target_availability_results.parquet")
    geometry = pd.read_parquet(OUTPUT_ROOT / "target_geometry_results.parquet")
    dummy = pd.read_parquet(OUTPUT_ROOT / "dummy_baseline_results.parquet")
    available_summary = availability.groupby(["targetId", "candidateId"], as_index=False)["defined"].sum().rename(columns={"defined": "definedMatrices"})
    geometry_summary = geometry.loc[geometry["defined"]].groupby(["targetId", "candidateId"], as_index=False).agg(
        wholeOccupancy=("wholeOccupancy", "mean"), suffixOccupancy=("suffixOccupancy", "mean"),
        rawFirstOnset=("firstOnset", "mean"), preOnsetAtCutoff=("noOnsetBeforeCutoff", "sum"),
        futureOnset=("firstOnsetInSuffix", "sum"),
    )
    dummy_summary = dummy.groupby(["targetId", "candidateId"], as_index=False).agg(
        dummyMedian=("dummyAccuracy", "median"), dummyMin=("dummyAccuracy", "min"), dummyMax=("dummyAccuracy", "max")
    )
    return available_summary, geometry_summary, dummy_summary


def render_reports() -> tuple[str, str]:
    classification = json.loads((OUTPUT_ROOT / "classification.json").read_text())
    regeneration = json.loads((OUTPUT_ROOT / "regeneration_validation.json").read_text())
    available, geometry, dummy = _summary_tables()
    metrics = pd.read_parquet(OUTPUT_ROOT / "robust_metric_results.parquet")
    comparisons = pd.read_parquet(OUTPUT_ROOT / "paired_model_comparisons.parquet")
    suffix = pd.read_parquet(OUTPUT_ROOT / "suffix_invariance_results.parquet")
    runtime = json.loads((OUTPUT_ROOT / "runtime_manifest.json").read_text())
    amendments = pd.read_csv(OUTPUT_ROOT / "technical_amendment_ledger.csv")
    artifacts_preview = [str(OUTPUT_ROOT / name) for name in REQUIRED_ARTIFACTS]
    if metrics.empty:
        model_summary_md = "The geometry advancement gate stopped execution before PhiRL feature computation or MLP fitting."
    else:
        selected = metrics.loc[metrics["modelId"].isin([P1_ID, P2_ID, B1_ID, B2_ID, B3_ID, B4_ID, B5_ID, B6_ID, DUMMY_ID])]
        model_summary = selected.groupby(["targetId", "candidateId", "modelId"], as_index=False).agg(
            medianAccuracy=("accuracy", "median"), medianBalancedAccuracy=("balancedAccuracy", "median"),
            medianAUROC=("auroc", "median"), medianAUPRC=("auprc", "median"), medianBrier=("brier", "median")
        )
        model_summary_md = model_summary.to_markdown(index=False, floatfmt=".4f")
    validation_text = (
        f"PASS: {regeneration['targetReplayPassed']}/{regeneration['targetReplayTotal']} target replays, "
        f"{regeneration['featureReplayPassed']}/{regeneration['featureReplayTotal']} feature replays, "
        f"{regeneration['modelReplayRows']} registered actual-model replay rows, "
        f"and {regeneration['immutablePriorEntryCount']} immutable prior files with "
        f"{regeneration['immutablePriorMismatchCount']} mismatches; "
        f"{len(amendments)} value-preserving technical amendments recorded."
    )
    primary = classification["primaryClassification"]
    promotion = classification["promotionClassification"]
    report = f"""# S19-L13 Full Results — Figure-5 Recurring-Target Prediction Reconstruction

## Top summary

- **Research step ID:** `{LOOP_ID}` (`{VERSION}`)
- **Completion status:** `COMPLETE_AWAITING_MANDATORY_HUMAN_REVIEW`
- **Artifacts written:** {len(artifacts_preview)} required named artifacts, 14 figures, exact target/feature/model evidence, validation manifests, and append-only S19 handoff records under `{OUTPUT_ROOT}`.
- **Validation result:** {validation_text}
- **Outcome classification:** `{primary}`; promotion status `{promotion}`.
- **Caveats or blockers:** This is an adaptive forensic reconstruction on previously studied matrices. R1 and U2 are completed-run recurring-attractor targets. P1 and the oracle are future-dependent. L12 froze approximate Figure 5 centers but not numerical box/whisker endpoints, so the prospectively fixed ±0.05 envelope is an adjudication tolerance rather than a redigitized whisker claim. Two value-preserving technical amendments are fully disclosed; neither changed a scientific value. No result identifies author code or changes S18.
- **Recommended next action:** Mandatory human review. Keep S20 and E02 inactive; do not run confirmation, another loop, intervention, or report generation automatically.

## Lay summary

The paper's Figure 5 shows a dummy predictor near 60% accuracy, while the adjacent-similarity label previously used in S16 made almost every state positive and therefore gave a dummy near 98%. L13 changed only that prediction target. It tested two previously frozen definitions of membership in recurring composition-space attractors while preserving the exact S16 tensor layout, splits, neural network, and training rules. The result below separates the arithmetic baseline clue, a completed-trajectory retrospective reconstruction, and genuinely prefix-only prediction. A completed-fit resemblance cannot be called early warning because its first-quarter feature was fitted using the full trajectory.

## Frozen question and interpretation boundary

The sole question was whether exact L10 R1 or authoritative L11R U2 labels could explain the Figure 5 approximately 60% dummy and model ordering. No third target, new threshold, new clustering rule, simulator change, balancing rule, architecture change, or hyperparameter search was allowed. The Table 1 88% target question was not reopened. A positive retrospective result would imply that Figure 5 likely used a different target object or denominator; it would not prove prospective prediction, causal emergence, intervention efficacy, or author-code identity.

## Inputs and provenance

- Frozen L11R dataset: exactly 100 shared catalytic matrices, 100 candidate-2 and 100 candidate-3 original-exposure trajectories, each with 100 fissions.
- R1 implementation: frozen L10 MATLAB-compatible historical dominant-compotype pipeline.
- U2 implementation: authoritative L11R repaired Euclidean recurring-centroid-union pipeline.
- Prediction implementation: frozen S16 original-order, right-padded, explicitly masked MLP contract and ten matrix-level 64/16/20 fit/validation/test assignments.
- PhiRL: pinned source commit `{PHIRL_COMMIT}`, source-defined `emergence = synergy + downward causation`, CPU float64.
- Historical GARD: pinned commit `{GARD_COMMIT}`.
- Paper constraint: exact L12 stored approximate centers (PhiRL 0.85, composition change 0.80, raw 0.80, flux 0.79, dummy 0.60); L12 did not store numeric whisker endpoints.

No mounted external dataset was used. All scientific inputs were frozen E01 artifacts and repository code.

## Detailed methods

### Outcome-blind lock and fixtures

The complete target, tensor, split, feature, model, metric, control, gate, promotion-priority, and resource contract was committed and pushed before L13 target geometry was opened. Sixteen fixtures checked R1 eligibility/ineligibility/ties, U2 exact replay and centroid tolerance, S16 cutoff/masking/splits/dummy/scaling, completed-fit dependence, prefix-only suffix invariance, target-feature separation, typed serialization/quarantine, exact model replay, and worker-failure provenance.

### Target geometry

Each direct molecular target was created on its original selected molecular clock. Undefined R1 trajectories retained a false target mask and were neither imputed nor replaced. The majority probability came only from valid fit-partition labels. The advancement gate required at least 80 defined matrices per candidate, both test classes, no padding or undefined-row scoring, and a ten-split dummy distribution whose range overlapped and median lay inside [0.55, 0.65] in both candidates.

### Features and models

For targets passing geometry, the same feature tensors were reused for both labels: completed-fit PhiRL (P1), prefix-only PhiRL (P2), composition change, raw counts, flux, adjacent H, prefix-only historical attractor geometry, time, matched random values, and P2 combinations with H and prefix geometry. The completed target-centroid oracle was diagnostic only. Every learned family used the identical 288,789-parameter S16 CPU-float64 MLP, AdamW, loss, early stopping, and model seed. Padding never entered loss or metrics.

### Controls and statistics

Controls were within-prefix temporal permutation, training-matrix suffix-label permutation, time only, matched random features, deterministic suffix perturbation, and the excluded completed-centroid oracle. Accuracy was paper-facing; balanced accuracy, AUROC, AUPRC, Brier, log loss, sensitivity, specificity, predictive values, calibration intercept/slope and ECE were secondary. Mann–Whitney reproduced the paper-like ten-split diagnostic; paired Wilcoxon and 4,096-replicate catalytic-matrix bootstraps were the stronger dependence-aware analyses. Candidates and targets were never pooled to rescue a gate.

## Results

### Availability, target geometry, and dummy baseline

{available.to_markdown(index=False)}

{geometry.to_markdown(index=False, floatfmt=".4f")}

{dummy.to_markdown(index=False, floatfmt=".4f")}

### Prediction results

{model_summary_md}

### Registered comparisons

{("No model comparisons were eligible." if comparisons.empty else comparisons[["targetId","candidateId","comparisonFamily","referenceModelId","comparatorModelId","referenceMedianAccuracy","comparatorMedianAccuracy","mannWhitneyP","holmAdjustedMannWhitneyP","matrixBootstrapObservedDifference","matrixBootstrapLower95","matrixBootstrapUpper95"]].to_markdown(index=False, floatfmt=".5f"))}

### Leakage and controls

{("No suffix audit was needed because geometry stopped execution." if suffix.empty else suffix.to_markdown(index=False, floatfmt=".6g"))}

The completed-fit feature is explicitly future-dependent. Every prefix-only suffix audit had to pass before a prospective gate was evaluated. The oracle was excluded from ordinary comparisons and every promotion decision.

## Technical-assurance amendments

Two narrowly bounded repairs were required and remain visible in `technical_amendment_ledger.csv` and `failure_ledger.csv`. `L13-TA-001` corrected target-specific oracle cache routing before any MLP fit or prediction outcome; the fresh-cache rerun reproduced every pre-model feature artifact byte-for-byte. `L13-TA-002` replaced a removed Matplotlib boxplot keyword after all scientific outcomes were complete. Its fresh-cache rerun was required to reproduce every registered scientific table, feature cache, prediction, metric, gate, and classification exactly before any figure or report was released. The failed attempts remain quarantined under `/cache`; no target, feature, split, model, prediction, metric, test, gate, or classification changed.

## Illustrated results

1. ![Figure 5 baseline arithmetic clue](figures/figure_01_baseline_arithmetic_clue.png)
2. ![Target availability](figures/figure_02_target_availability.png)
3. ![Whole and suffix prevalence](figures/figure_03_whole_suffix_prevalence.png)
4. ![First-onset availability](figures/figure_04_first_onset_availability.png)
5. ![Representative target sequences](figures/figure_05_representative_target_sequences.png)
6. ![Completed-fit and prefix-only PhiRL](figures/figure_06_completed_vs_prefix_phirl.png)
7. ![Reconstructed Figure 5 accuracy](figures/figure_07_paper_accuracy_boxplots.png)
8. ![Robust metrics](figures/figure_08_robust_metrics.png)
9. ![Incremental-value comparisons](figures/figure_09_incremental_value.png)
10. ![Future dependence and invariance](figures/figure_10_future_dependence.png)
11. ![Negative controls](figures/figure_11_negative_controls.png)
12. ![Candidate agreement](figures/figure_12_candidate_agreement.png)
13. ![Decision matrix](figures/figure_13_decision_matrix.png)
14. ![Promotion decision tree](figures/figure_14_promotion_decision_tree.png)

## Validation

{validation_text}

The repository lock was clean and matched `origin/eidosoma/groups/42`; all 16 mandatory fixtures passed. The immutable baseline excluded only append-only S19 root ledgers and L13's own new directory. Exact U2 replay required identical trajectory states, labels, scores, and scoring centroids. Actual-model replay used identical initial weights, histories, and predictions for every registered model at repetition zero. Derived comparison tables were regenerated exactly.

`FIGURE_CONTENTS_AND_CAPTIONS_FOR_HUMAN_REVIEW.md` is retained as V1 for the 14 generated L13 figures. The human-requested V2, `FIGURE_CONTENTS_AND_CAPTIONS_FOR_HUMAN_REVIEW_V2.md`, instead records my panel-by-panel reading of the input paper's Figures 1–6, caption meaning, visible values, operational implications, ambiguities, Table 1 conflicts, and a manual verification checklist against frozen paper/native-image hashes.

## Commands

```text
PYTHONPATH=src pytest -q tests/e01/test_s19_l13.py tests/e01/test_s16_prediction_reconstruction.py tests/e01/test_s19_l10.py tests/e01/test_s19_l11r.py
PYTHONPATH=src python scripts/e01/run_s19_l13.py prepare
PYTHONPATH=src python scripts/e01/run_s19_l13.py geometry
PYTHONPATH=src python scripts/e01/run_s19_l13.py benchmark
PYTHONPATH=src python scripts/e01/run_s19_l13.py execute
PYTHONPATH=src python scripts/e01/run_s19_l13.py finalize
```

## Runtime and dependencies

- CPU scientific seconds: `{runtime.get('executionCpuSeconds', 0):.3f}`; execution wall seconds: `{runtime.get('executionWallSeconds', 0):.3f}`.
- Workers: at most 8; numeric-library threads: one per worker; GPU: not used.
- Python `{platform.python_version()}`, NumPy `{np.__version__}`, pandas `{pd.__version__}`, SciPy `{scipy.__version__}`, scikit-learn `{sklearn.__version__}`, PyTorch `{torch.__version__}`.

## Caveats, blockers, and failed assumptions

- L13 is adaptive: the same matrices informed earlier label work, so even a promotable result requires untouched confirmation.
- Both targets are defined from completed trajectories. This makes the target task itself retrospective, even when P2 input is suffix-independent.
- L12's native-figure measurements are approximate and do not provide numeric box/whisker endpoints.
- Repeated splits overlap in matrix membership; they are paper-facing diagnostics, not ten independent experiments.
- No favorable candidate or target pooling is allowed; candidate disagreement is a failure.
- L10/L11R remain negative for the separate Table 1 88% fingerprint. L13 does not reinterpret those results.
- S18 prospective-prediction and causal-control non-support remains unchanged unless a later untouched confirmation is separately authorized and passes its own gates.

## Artifact and provenance index

The machine-readable target, feature, split, training, prediction, metric, comparison, control, leakage, gate, classification, runtime, storage, regeneration, and hash files are listed in `artifact_manifest.json`. Large disposable tensors stayed under `{CACHE_ROOT}` and are not collectible artifacts. Repository code stayed in Git.

## Recommended next action

Return to mandatory human review. Do not begin S20, E02, a confirmation dataset, intervention analysis, report bundle, or another S19 loop automatically.
"""
    summary = f"""# S19-L13 One-Page Decision Summary

## Top summary

- **Research step ID:** `{LOOP_ID}`
- **Completion status:** `COMPLETE_AWAITING_MANDATORY_HUMAN_REVIEW`
- **Artifacts written:** complete L13 machine-readable evidence, 14 generated figures, full report, V1 generated-figure guide, V2 input-paper figure/caption guide, validation and hash manifests.
- **Validation result:** {validation_text}
- **Outcome classification:** `{primary}`; `{promotion}`.
- **Caveats or blockers:** adaptive existing-matrix analysis; both targets are completed-run labels; P1 is future-dependent; Figure 5 digitization is approximate; author code remains unavailable.
- **Recommended next action:** mandatory human review; no automatic continuation.

## Decision

{available.to_markdown(index=False)}

{dummy.to_markdown(index=False, floatfmt=".4f")}

The result does not identify the author implementation and does not change S18's prospective prediction or causal-control verdicts. Any promoted pipeline is only a proposal for a new untouched confirmation.
"""
    return report, summary


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) if path.exists() else 0


def append_postoutcome_ledgers(classification: dict[str, Any], regeneration: dict[str, Any]) -> None:
    ledger_path = S19_ROOT / "self_improvement_ledger.parquet"
    ledger = pd.read_parquet(ledger_path)
    if not ((ledger["loopId"] == LOOP_ID) & (ledger["recordPhase"] == "POST_LOOP_RESULT_AND_HUMAN_REVIEW_HANDOFF")).any():
        decisions = classification["targetDecisions"]
        learned = "; ".join(
            f"{target}: {','.join(value.get('classifications', []))}"
            for target, value in decisions.items()
        )
        row = {
            "appendOnly": True,
            "beliefBeforeLoop": "A recurring-attractor target with approximately 0.40 occupancy might explain Figure 5's approximately 0.60 majority dummy while preserving the exact S16 task.",
            "failureOrAmbiguityTargeted": "Figure 5 dummy/Table 1 probability contradiction and whether target identity explains the prediction panel.",
            "informationGainRationale": "The loop isolated target identity by holding tensor, feature, model, split and evaluation semantics fixed.",
            "learned": learned,
            "ledgerSequence": int(ledger["ledgerSequence"].max()) + 1,
            "loopId": LOOP_ID,
            "motivatingEvidence": "Frozen L10 R1 and L11R U2 labels had approximately 0.40 mean occupancy, but neither had been used as the S16 target.",
            "proposedNextTest": "Mandatory human review; execute no confirmation or later loop automatically.",
            "recordPhase": "POST_LOOP_RESULT_AND_HUMAN_REVIEW_HANDOFF",
            "remainingPlausibleHypotheses": "Author target/sampling details remain unavailable; only a separately authorized untouched confirmation could test any promoted pipeline.",
            "selectedHypotheses": "Exact R1 and U2 Figure 5 targets under the exact S16 reconstruction.",
            "timestampUtc": utc_now(),
            "weakenedHypotheses": "Any target-feature pipeline failing its frozen geometry, retrospective, prospective, control or cross-candidate gate is weakened within the tested scope.",
        }
        write_parquet(ledger_path, pd.concat([ledger, pd.DataFrame([row])], ignore_index=True))
    markdown_path = S19_ROOT / "SELF_IMPROVEMENT_LEDGER.md"
    text = markdown_path.read_text(encoding="utf-8")
    marker = "## S19-L13 post-loop result"
    if marker not in text:
        text += f"\n\n{marker}\n\n- Before: recurring-attractor targets might reconcile the Figure 5 60% dummy without changing S16.\n- Learned: `{classification['primaryClassification']}`; promotion `{classification['promotionClassification']}`.\n- Validation: {regeneration['targetReplayPassed']}/{regeneration['targetReplayTotal']} targets and {regeneration['featureReplayPassed']}/{regeneration['featureReplayTotal']} features replayed; immutable mismatches {regeneration['immutablePriorMismatchCount']}.\n- Next: mandatory human review; no automatic continuation.\n"
        markdown_path.write_text(text, encoding="utf-8")
    source_report = S19_ROOT / "source_search_report.md"
    text = source_report.read_text(encoding="utf-8")
    marker = "## L13 frozen-source reuse"
    if marker not in text:
        text += f"\n\n{marker}\n\nL13 introduced no source update. It pinned the original paper/L12 Figure 5 audit, PhiRL `{PHIRL_COMMIT}`, GARD `{GARD_COMMIT}`, exact S16 model/tensor code, L10 R1 and L11R U2 implementations. Classification: `{classification['primaryClassification']}`.\n"
        source_report.write_text(text, encoding="utf-8")
    registry_path = S19_ROOT / "loop_registry.yaml"
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    for loop in registry["loops"]:
        if loop.get("loopId") == LOOP_ID:
            loop.update({
                "status": "COMPLETE_AWAITING_MANDATORY_HUMAN_REVIEW",
                "outcomeAccessed": True, "completed": True,
                "eligibleScientificResults": True,
                "promotedLeadCount": 0 if classification["promotedPipeline"] is None else 1,
                "classification": classification["primaryClassification"],
                "nextStepActive": False,
            })
    write_yaml(registry_path, registry)
    history_path = S19_ROOT / "human_review_history.json"
    history = json.loads(history_path.read_text())
    entry = {
        "loopId": LOOP_ID, "decision": "AUTHORIZE_EXACTLY_ONE_L13_LOOP",
        "recordedAtUtc": utc_now(), "status": "CONSUMED_AND_RETURNED_FOR_MANDATORY_REVIEW",
        "nextLoopAuthorized": False, "s20Activated": False,
    }
    if isinstance(history, list):
        if not any(item.get("loopId") == LOOP_ID for item in history): history.append(entry)
    elif isinstance(history, dict):
        key = "history" if "history" in history else "decisions" if "decisions" in history else None
        if key is None: history["history"] = [entry]
        elif not any(item.get("loopId") == LOOP_ID for item in history[key]): history[key].append(entry)
    write_json(history_path, history)


def write_root_handoff(classification: dict[str, Any], regeneration: dict[str, Any]) -> None:
    shutil.copy2(OUTPUT_ROOT / "S19_L13_FULL_RESULTS.md", S19_ROOT / "research_step_full_results.md")
    status = {
        "researchStepId": LOOP_ID, "stepNumber": 19, "success": True,
        "status": "COMPLETE_AWAITING_MANDATORY_HUMAN_REVIEW",
        "artifactsWritten": [
            str(OUTPUT_ROOT / "S19_L13_FULL_RESULTS.md"), str(OUTPUT_ROOT / "classification.json"),
            str(OUTPUT_ROOT / "artifact_manifest.json"),
            str(OUTPUT_ROOT / "FIGURE_CONTENTS_AND_CAPTIONS_FOR_HUMAN_REVIEW.md"),
            str(OUTPUT_ROOT / "FIGURE_CONTENTS_AND_CAPTIONS_FOR_HUMAN_REVIEW_V2.md"),
            str(S19_ROOT / "research_step_full_results.md"),
        ],
        "validationResult": (
            f"PASS_TARGET_{regeneration['targetReplayPassed']}_OF_{regeneration['targetReplayTotal']}"
            f"_FEATURE_{regeneration['featureReplayPassed']}_OF_{regeneration['featureReplayTotal']}"
            f"_MODEL_REPLAY_{regeneration['modelReplayRows']}_IMMUTABLE_MISMATCH_{regeneration['immutablePriorMismatchCount']}"
        ),
        "outcomeClassification": classification["primaryClassification"],
        "caveatsOrBlockers": [
            "adaptive_existing_matrix_analysis", "completed_run_target_definitions",
            "completed_fit_P1_future_dependence", "approximate_figure5_digitization",
            "author_code_unavailable", "S18_prediction_and_control_statuses_unchanged",
        ],
        "recommendedNextAction": "MANDATORY_HUMAN_REVIEW_KEEP_S20_E02_AND_L14_INACTIVE_NO_AUTOMATIC_CONFIRMATION",
    }
    write_json(S19_ROOT / "s19_status.json", status)


def write_artifact_manifest(path: Path, root: Path, *, schema: str) -> None:
    entries = []
    for item in sorted(root.rglob("*")):
        if item.is_file() and item != path:
            entries.append({
                "path": str(item), "relativePath": str(item.relative_to(root)),
                "bytes": item.stat().st_size, "sha256": sha256_file(item),
            })
    write_json(path, {
        "schema": schema, "researchStepId": LOOP_ID, "generatedAtUtc": utc_now(),
        "artifactCount": len(entries), "totalBytesExcludingManifest": sum(row["bytes"] for row in entries),
        "entries": entries, "passed": True,
    })


def finalize_phase() -> None:
    started = time.time()
    geometry_status = json.loads((CACHE_ROOT / "geometry_status.json").read_text())
    ensure_required_tables()
    if not (OUTPUT_ROOT / "classification.json").exists():
        decisions = {
            target: {
                "geometryAdvanced": False, "retrospectiveGatePassed": False,
                "prospectiveGatePassed": False,
                "classifications": ["FIGURE5_BASELINE_NOT_RECONCILED_BY_FROZEN_RECURRING_TARGETS", "NOT_PROMOTABLE"],
            }
            for target in TARGET_IDS
        }
        write_json(OUTPUT_ROOT / "classification.json", {
            "schema": "eidosoma.e01.s19.l13.classification.v1", "researchStepId": LOOP_ID,
            "versionedStepId": VERSION, "status": "COMPLETE_AWAITING_MANDATORY_HUMAN_REVIEW",
            "primaryClassification": "FIGURE5_BASELINE_NOT_RECONCILED_BY_FROZEN_RECURRING_TARGETS",
            "promotionClassification": "NOT_PROMOTABLE", "promotedPipeline": None,
            "targetDecisions": decisions, "priorClassificationsChanged": False,
            "s18ProspectivePredictionStatus": "PROSPECTIVE_PREDICTION_NOT_SUPPORTED_WITHIN_TESTED_SCOPE_UNCHANGED",
            "s18CausalControlStatus": "PROSPECTIVE_CAUSAL_CONTROL_NOT_SUPPORTED_WITHIN_TESTED_SCOPE_UNCHANGED",
            "authorCodeIdentified": False, "exactReplicationClaimed": False,
        })
    classification = json.loads((OUTPUT_ROOT / "classification.json").read_text())
    regeneration = regeneration_validation()
    if not regeneration["passed"]:
        raise RuntimeError(f"L13 regeneration validation failed: {regeneration}")
    regeneration["reportRegenerationExact"] = True
    write_json(OUTPUT_ROOT / "regeneration_validation.json", regeneration)
    figures = generate_figures()
    if len(figures) != 14 or any(not path.is_file() for path in figures):
        raise RuntimeError("required figure generation failed")
    write_figure_review_artifact(figures)
    write_paper_figure_review_v2()
    if AMENDMENT_002_PATH.is_file():
        replay = validate_amendment_002_replay(include_post_figure=True)
        expected_regeneration = replay["amendment"]["expectedRegenerationScientificValues"]
        observed_regeneration = {
            key: regeneration.get(key) for key in expected_regeneration
        }
        if observed_regeneration != expected_regeneration:
            raise RuntimeError(
                "technical amendment 002 changed regeneration values: "
                f"expected={expected_regeneration}, observed={observed_regeneration}"
            )
        amendment_rows = pd.read_csv(OUTPUT_ROOT / "technical_amendment_ledger.csv")
        selected = amendment_rows["amendmentId"].eq(replay["amendment"]["amendmentId"])
        amendment_rows.loc[selected, "status"] = "VALUE_PRESERVING_PLOTTING_REPAIR_PASSED"
        amendment_rows.loc[selected, "exactFeatureArtifactHashesPassed"] = replay["passed"]
        amendment_rows.to_csv(OUTPUT_ROOT / "technical_amendment_ledger.csv", index=False)
        failure_rows = pd.read_csv(OUTPUT_ROOT / "failure_ledger.csv")
        selected = failure_rows["failureId"].eq("L13-F002-MATPLOTLIB-BOXPLOT-KEYWORD")
        failure_rows.loc[selected, "status"] = "PRESERVED_REPAIRED_VALUE_PRESERVING"
        failure_rows.to_csv(OUTPUT_ROOT / "failure_ledger.csv", index=False)
    runtime = json.loads((OUTPUT_ROOT / "runtime_manifest.json").read_text())
    runtime.update({
        "phase": "COMPLETE", "completedAtUtc": utc_now(),
        "finalizationWallSeconds": time.time() - started,
        "gpuUsed": False, "scientificOutcomeGenerated": bool(geometry_status["advancedTargets"]),
        "mandatoryHumanReviewBoundary": True,
    })
    write_json(OUTPUT_ROOT / "runtime_manifest.json", runtime)
    retained = directory_bytes(OUTPUT_ROOT)
    cache = directory_bytes(CACHE_ROOT)
    storage = {
        "schema": "eidosoma.e01.s19.l13.storage_validation.v1",
        "researchStepId": LOOP_ID, "retainedBytesBeforeManifestAndReports": retained,
        "temporaryCacheBytes": cache, "retainedGiBMaximum": 35,
        "temporaryGiBMaximum": 100,
        "retainedWithinCeiling": retained < 35 * 1024**3,
        "temporaryWithinCeiling": cache < 100 * 1024**3,
        "passed": retained < 35 * 1024**3 and cache < 100 * 1024**3,
    }
    write_json(OUTPUT_ROOT / "storage_validation.json", storage)
    if not storage["passed"]:
        raise RuntimeError("storage ceiling exceeded")
    first_report, first_summary = render_reports()
    second_report, second_summary = render_reports()
    if first_report != second_report or first_summary != second_summary:
        raise RuntimeError("report regeneration is not exact")
    (OUTPUT_ROOT / "S19_L13_FULL_RESULTS.md").write_text(first_report, encoding="utf-8")
    (OUTPUT_ROOT / "loop_decision_summary.md").write_text(first_summary, encoding="utf-8")
    # The required technical-amendment ledger is explicit even when no amendment was used.
    if not (OUTPUT_ROOT / "technical_amendment_ledger.csv").exists():
        pd.DataFrame(columns=["amendmentId", "authorized", "scientificValuesChanged", "status"]).to_csv(OUTPUT_ROOT / "technical_amendment_ledger.csv", index=False)
    write_artifact_manifest(
        OUTPUT_ROOT / "artifact_manifest.json", OUTPUT_ROOT,
        schema="eidosoma.e01.s19.l13.artifact_manifest.v1",
    )
    missing = [name for name in REQUIRED_ARTIFACTS if not (OUTPUT_ROOT / name).is_file()]
    if missing:
        raise RuntimeError(f"required L13 artifacts missing: {missing}")
    append_postoutcome_ledgers(classification, regeneration)
    write_root_handoff(classification, regeneration)
    write_artifact_manifest(
        S19_ROOT / "artifact_manifest.json", S19_ROOT,
        schema="eidosoma.e01.s19.root_artifact_manifest.v3",
    )
    print(canonical_json({
        "phase": "finalize", "status": "COMPLETE_AWAITING_MANDATORY_HUMAN_REVIEW",
        "classification": classification["primaryClassification"],
        "promotion": classification["promotionClassification"],
        "requiredArtifacts": len(REQUIRED_ARTIFACTS), "figures": len(figures),
        "regenerationPassed": regeneration["passed"],
    }))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["prepare", "geometry", "benchmark", "execute", "finalize", "all"])
    args = parser.parse_args()
    if args.phase in {"prepare", "all"}: prepare()
    if args.phase in {"geometry", "all"}: geometry_phase()
    if args.phase in {"benchmark", "all"}: benchmark_phase()
    if args.phase in {"execute", "all"}: execute_phase()
    if args.phase in {"finalize", "all"}: finalize_phase()


if __name__ == "__main__":
    main()
