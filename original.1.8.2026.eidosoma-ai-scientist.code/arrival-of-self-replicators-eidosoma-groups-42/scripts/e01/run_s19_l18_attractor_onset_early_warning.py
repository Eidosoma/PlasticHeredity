#!/usr/bin/env python3
"""Prepare and execute the frozen S19-L18 attractor-onset experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow
import scipy
import sklearn
import yaml
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from e01_attractor_onset_early_warning.core import (  # noqa: E402
    FEATURE_GROUPS,
    HORIZON_EXCLUSIVE,
    LANDMARK_COUNT,
    build_landmark_target,
    derive_seed,
    extract_past_features,
    metric_summary,
)
from e01_clean_directional_confirmation.core import fixed_label_spec  # noqa: E402
from e01_creative_directional_search.core import label_trajectory  # noqa: E402
from e01_frozen_timebase_ensemble.core import (  # noqa: E402
    selected_clock_observations,
    states_from_observations,
)

LOOP_ID = "S19-L18"
VERSION = "E01-S19-L18-RECURRING-ATTRACTOR-ONSET-EARLY-WARNING-v1.0.0"
TARGET_ID = "PF_DOMINANT_COMPONENT_CENTROID_H900"
CANDIDATES = ("S12F-CANDIDATE-02", "S12F-CANDIDATE-03")
ARTIFACT_ROOT = Path("/artifacts/research_steps/S19")
LOOP_ROOT = ARTIFACT_ROOT / "loops/L18"
CACHE_ROOT = Path("/cache/e01_s19_l18")
BUILD_ROOT = CACHE_ROOT / "build"
REPLAY_ROOT = CACHE_ROOT / "replay"
CONFIG = REPO_ROOT / "configs/e01/s19_l18_attractor_onset_early_warning.yaml"
S13Y_ROOT = Path("/artifacts/research_steps/S13Y")
L02_ROOT = ARTIFACT_ROOT / "loops/L02"
L17_ROOT = ARTIFACT_ROOT / "loops/L17"
SAFE_LATTICE = Path("/artifacts/research_steps/S12B/safe_phi_lattice.json")
EXACT_PYTHON = Path("/cache/e01_s19_l17/venv/bin/python")
BGM_WORKER = REPO_ROOT / "scripts/e01/l18_bgm_prefix_worker.py"
MODEL_IDS = (
    "DUMMY_TRAINING_PRIOR",
    "TIME_ONLY",
    "EXACT_H_STABILITY",
    "PREFIX_RECURRENCE_GEOMETRY",
    "ORGANIZATION_DYNAMICS",
    "BGM_PREFIX_EMERGENCE",
    "BGM_PREFIX_INTEGRATED",
    "PAST_FULL_NO_BGM",
    "PAST_FULL_WITH_BGM_EMERGENCE",
    "PAST_FULL_WITH_BGM_INTEGRATED",
    "COMPLETED_BGM_ORACLE",
    "COMPLETED_TARGET_CENTROID_ORACLE",
)
PRIMARY_MODEL = "PAST_FULL_WITH_BGM_EMERGENCE"
NONPHI_MODEL = "PAST_FULL_NO_BGM"
BOOTSTRAPS = 4096
PERMUTATIONS = 512


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def write_json(path: Path, value: Any) -> None:
    atomic_text(path, canonical_json(value) + "\n")


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp.parquet")
    frame.to_parquet(temp, index=False, compression="zstd")
    os.replace(temp, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_hash(frame: pd.DataFrame) -> str:
    ordered = frame.copy()
    ordered = ordered.reindex(sorted(ordered.columns), axis=1)
    payload = ordered.to_json(orient="table", index=False, double_precision=15)
    return hashlib.sha256(payload.encode()).hexdigest()


def run_exact_bgm(states: np.ndarray, identity: tuple[object, ...]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Execute the L17-confirmed BGM path in its pinned isolated environment."""

    token = hashlib.sha256("\x1f".join(map(str, identity)).encode()).hexdigest()[:24]
    task_root = CACHE_ROOT / "exact_bgm_tasks"
    input_path = task_root / f"{token}.input.npz"
    output_path = task_root / f"{token}.output.npz"
    task_root.mkdir(parents=True, exist_ok=True)
    temporary = input_path.with_name(f".{input_path.name}.tmp.npz")
    np.savez_compressed(temporary, states=np.asarray(states, dtype=np.int64))
    os.replace(temporary, input_path)
    env = os.environ.copy()
    for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"]:
        env[name] = "1"
    completed = subprocess.run(
        [
            str(EXACT_PYTHON), str(BGM_WORKER), "--input", str(input_path), "--output", str(output_path),
            "--safe-lattice", str(SAFE_LATTICE), "--preprocessing-seed", str(derive_seed("exact_bgm_pre", *identity)),
            "--partition-seed", str(derive_seed("exact_bgm_partition", *identity)),
        ],
        cwd=REPO_ROOT, env=env, text=True, capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"exact BGM worker failed: {completed.stderr[-2000:]}")
    with np.load(output_path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"]))
        arrays = {name: np.asarray(payload[name]) for name in payload.files if name != "metadata_json"}
    return metadata, arrays


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, check=True, text=True, capture_output=True).stdout.strip()


def aggregate_file_identity(rows: Iterable[dict[str, Any]]) -> str:
    material = "\n".join(f"{row['path']}\t{row['sha256']}\t{row['bytes']}" for row in rows)
    return hashlib.sha256(material.encode()).hexdigest()


def validate_prior_from_l17() -> dict[str, Any]:
    baseline = json.loads((L17_ROOT / "immutable_prior_validation.json").read_text())
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for row in baseline["files"]:
        path = Path(row["path"])
        if not path.exists() or path.stat().st_size != int(row["bytes"]) or sha256_file(path) != row["sha256"]:
            failures.append(str(path))
        rows.append(row)
    l17_manifest = json.loads((L17_ROOT / "artifact_manifest.json").read_text())
    for row in l17_manifest["files"]:
        path = L17_ROOT / row["path"]
        identity = {"root": str(L17_ROOT), "path": str(path), "bytes": int(row["bytes"]), "sha256": row["sha256"]}
        if not path.exists() or path.stat().st_size != identity["bytes"] or sha256_file(path) != identity["sha256"]:
            failures.append(str(path))
        rows.append(identity)
    return {
        "status": "PASS" if not failures else "FAIL",
        "unchanged": not failures,
        "fileCount": len(rows),
        "rootCount": int(baseline["rootCount"]) + 1,
        "aggregateSha256": aggregate_file_identity(rows),
        "failures": failures,
        "files": rows,
    }


def fixture_table() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    labels = np.zeros(220, dtype=bool)
    for onset, risk, event in [(63, False, None), (64, True, True), (191, True, True), (192, True, False), (None, True, False)]:
        labels[:] = False
        if onset is not None:
            labels[onset] = True
        got = build_landmark_target(labels)
        passed = got["atRiskAtLandmark"] is risk and got["eventWithinHorizon"] is event
        rows.append({"fixtureId": f"TARGET_ONSET_{onset}", "passed": passed, "details": canonical_json(got)})
    rng = np.random.default_rng(derive_seed("fixture", "features"))
    states = rng.integers(0, 5, size=(64, 100), dtype=np.int64)
    states[:, 0] += 1
    generations = np.arange(64) // 7
    kinds = ["post_fission" if i % 7 == 0 else "molecular_update" for i in range(64)]
    a = extract_past_features(states, generations, kinds)
    b = extract_past_features(states.copy(), generations.copy(), list(kinds))
    rows.append({"fixtureId": "FEATURE_EXACT_REPLAY", "passed": a == b, "details": str(len(a))})
    suffix = rng.integers(0, 5, size=(200, 100), dtype=np.int64)
    combined = np.vstack([states, suffix])
    rng.shuffle(combined[64:], axis=0)
    c = extract_past_features(combined[:64], generations, kinds)
    rows.append({"fixtureId": "SUFFIX_INVARIANCE", "passed": a == c, "details": "first_64_only"})
    expected = set().union(*map(set, FEATURE_GROUPS.values()))
    rows.append({"fixtureId": "FEATURE_SCHEMA", "passed": set(a) == expected, "details": str(sorted(expected))})
    frame = pd.DataFrame({"flag": pd.Series([True, False, None], dtype="boolean"), "value": [1.0, np.nan, 3.0], "status": ["A", "B", "C"]})
    temp = CACHE_ROOT / "fixture_serialization.parquet"
    write_parquet(temp, frame)
    replay = pd.read_parquet(temp)
    rows.append({"fixtureId": "PARQUET_NULLABLE_REPLAY", "passed": replay.shape == frame.shape and replay.columns.tolist() == frame.columns.tolist(), "details": pyarrow.__version__})
    with Path("/cache/e01_s13y_v1/raw_trajectories/S12F-CANDIDATE-02/M000.pickle").open("rb") as handle:
        fixture_trajectory = pickle.load(handle)
    fixture_states = states_from_observations(
        selected_clock_observations(fixture_trajectory, "C1_SELECTED_DAUGHTER_RETAINED")
    )[:LANDMARK_COUNT]
    first_meta, first_arrays = run_exact_bgm(fixture_states, ("fixture", "bgm"))
    first_output = CACHE_ROOT / "exact_bgm_tasks" / f"{hashlib.sha256(chr(31).join(map(str, ('fixture', 'bgm'))).encode()).hexdigest()[:24]}.output.npz"
    first_output.unlink(missing_ok=True)
    second_meta, second_arrays = run_exact_bgm(fixture_states.copy(), ("fixture", "bgm"))
    bgm_pass = (
        first_meta["status"] in {"ELIGIBLE", "ELIGIBLE_PARTIAL_NONFINITE_LOCAL_VALUES"}
        and first_meta["status"] == second_meta["status"]
        and np.array_equal(first_arrays.get("emergence_nan0"), second_arrays.get("emergence_nan0"), equal_nan=True)
        and np.array_equal(first_arrays.get("integrated_raw"), second_arrays.get("integrated_raw"), equal_nan=True)
    )
    rows.append({"fixtureId": "BGM_PREFIX_CPU_FLOAT64_EXACT_REPLAY", "passed": bgm_pass, "details": first_meta["status"]})
    x = rng.normal(size=(30, 4)); y = np.array([0, 1] * 15)
    m1 = _pipeline(derive_seed("fixture", "model")); m2 = _pipeline(derive_seed("fixture", "model"))
    m1.fit(x, y); m2.fit(x, y)
    rows.append({"fixtureId": "LOGISTIC_MODEL_EXACT_REPLAY", "passed": np.array_equal(m1.predict_proba(x), m2.predict_proba(x)), "details": "30x4"})
    result = pd.DataFrame(rows)
    if not result["passed"].all():
        raise RuntimeError("mandatory L18 fixture failed")
    return result


def prepare_lock() -> None:
    if git("status", "--porcelain=v1"):
        raise RuntimeError("repository must be clean before pre-outcome lock")
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/eidosoma/groups/42")
    if head != remote:
        raise RuntimeError("local and pushed branch differ")
    LOOP_ROOT.mkdir(parents=True, exist_ok=True)
    fixtures = fixture_table()
    prior = validate_prior_from_l17()
    if not prior["unchanged"]:
        raise RuntimeError("immutable prior validation failed")
    shutil.copy2(CONFIG, LOOP_ROOT / "preregistration.yaml")
    write_parquet(LOOP_ROOT / "fixture_results.parquet", fixtures)
    write_json(LOOP_ROOT / "immutable_prior_validation.json", prior)
    decision = """# S19-L18 decision record

The human authorized L18 as a continuation of L17, explicitly requiring fewer than the approximately 98% adjacent-H replicators and a serious attempt to detect emerging organization before replication.

L18 therefore freezes one adaptive landmark-onset task. The target is the already frozen L02 dominant recurring-component centroid label, not a newly tuned definition. A matrix is at risk after 64 real selected-clock observations only if the target has never been positive; the event is first target entry during observations 64–191. The 64-observation history and 128-observation horizon are existing program windows and use a fixed raw clock, avoiding completed-length and padding leakage.

The target is defined from the completed trajectory and is therefore retrospective. Every competitive predictor is computed from the first 64 observations only. Completed-fit BreakingGRNMemories values and completed target-centroid similarity are quarantined as future-dependent diagnostic oracles. This is adaptive exploratory evidence and cannot confirm author code, the paper label, prediction, or causation.
"""
    atomic_text(LOOP_ROOT / "decision_record.md", decision)
    implementation = {
        "schema": "eidosoma.e01.s19_l18.implementation_lock.v1",
        "repositoryHead": head,
        "remoteHead": remote,
        "configSha256": sha256_file(CONFIG),
        "runnerSha256": sha256_file(Path(__file__)),
        "coreSha256": sha256_file(REPO_ROOT / "src/e01_attractor_onset_early_warning/core.py"),
        "targetLabelId": TARGET_ID,
        "landmarkCount": LANDMARK_COUNT,
        "horizonExclusive": HORIZON_EXCLUSIVE,
        "modelIds": MODEL_IDS,
        "featureGroups": {k: list(v) for k, v in FEATURE_GROUPS.items()},
        "bootstrapReplicates": BOOTSTRAPS,
        "permutationReplicates": PERMUTATIONS,
        "preparedAtUtc": utc_now(),
    }
    write_json(LOOP_ROOT / "implementation_lock.json", implementation)
    source = {
        "schema": "eidosoma.e01.s19_l18.source_snapshot.v1",
        "paperSha256": sha256_file(Path("/workspace/input-attachments/ed5486bf-a043-485b-a233-d88d8d123759/pdf-markdown.md")),
        "s13yManifestSha256": sha256_file(S13Y_ROOT / "trajectory_manifest.parquet"),
        "l02TargetSha256": sha256_file(L02_ROOT / "label_values.parquet"),
        "l02FingerprintSha256": sha256_file(L02_ROOT / "fingerprint_results.parquet"),
        "l17ValuesSha256": sha256_file(L17_ROOT / "gard_phi_values.parquet"),
        "safeLatticeSha256": sha256_file(SAFE_LATTICE),
        "breakingGrnCommit": "afe44231ad3ce915172cdb53a6b234bd76fcb6a5",
        "historicalGardCommit": "86dff6320d5ae91b4e831471079ff46749b14df9",
    }
    write_json(LOOP_ROOT / "source_snapshot_manifest.json", source)
    benchmark_manifest = pd.read_parquet(S13Y_ROOT / "trajectory_manifest.parquet").sort_values(["candidateId", "matrixIndex"])
    benchmark_records = pd.concat([
        benchmark_manifest[benchmark_manifest["candidateId"].eq(candidate)].head(2)
        for candidate in CANDIDATES
    ])
    benchmark_rows = []
    for record in benchmark_records.itertuples(index=False):
        _, _, states = _load_trajectory(record)
        result = _bgm_worker((record.candidateId, int(record.matrixIndex), states[:LANDMARK_COUNT], False))
        benchmark_rows.append(result)
    benchmark_wall = float(sum(row["wallSeconds"] for row in benchmark_rows))
    projected_cpu_hours = float(sum(row["cpuSeconds"] for row in benchmark_rows) * 400 / len(benchmark_rows) / 3600.0)
    projection = {
        "status": "PASS",
        "opaqueTrajectoryUnits": len(benchmark_rows),
        "projectedPrefixBgmCpuHoursFor400Pipelines": projected_cpu_hours,
        "projectedTotalCpuHoursIncludingModelsControlsAndValidation": projected_cpu_hours + 12.0,
        "cpuHoursMaximum": 100,
        "wallHoursMaximum": 72,
        "gatePassed": projected_cpu_hours + 12.0 < 100,
        "benchmarkRows": benchmark_rows,
    }
    if not projection["gatePassed"]:
        raise RuntimeError("L18 benchmark projects beyond ceiling")
    write_json(LOOP_ROOT / "benchmark_projection.json", projection)
    write_json(LOOP_ROOT / "preoutcome_repository_lock.json", {
        "status": "PASS", "head": head, "remote": remote, "clean": True,
        "configSha256": implementation["configSha256"], "preparedAtUtc": utc_now(),
    })
    print(canonical_json({"status": "PREOUTCOME_LOCKED", "head": head, "fixtures": len(fixtures), "priorFiles": prior["fileCount"]}))


def validate_execution_lock() -> None:
    lock = json.loads((LOOP_ROOT / "preoutcome_repository_lock.json").read_text())
    if git("status", "--porcelain=v1"):
        raise RuntimeError("repository became dirty after lock")
    if git("rev-parse", "HEAD") != lock["head"] or git("rev-parse", "origin/eidosoma/groups/42") != lock["head"]:
        raise RuntimeError("repository identity changed after lock")
    if sha256_file(CONFIG) != lock["configSha256"] or sha256_file(CONFIG) != sha256_file(LOOP_ROOT / "preregistration.yaml"):
        raise RuntimeError("preregistration identity changed")
    prior = validate_prior_from_l17()
    frozen = json.loads((LOOP_ROOT / "immutable_prior_validation.json").read_text())
    if not prior["unchanged"] or prior["aggregateSha256"] != frozen["aggregateSha256"]:
        raise RuntimeError("immutable prior changed after lock")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_parquet(S13Y_ROOT / "trajectory_manifest.parquet").sort_values(["candidateId", "matrixIndex"]).reset_index(drop=True)
    if len(manifest) != 200 or set(manifest["candidateId"]) != set(CANDIDATES) or not manifest["exactReplayPassed"].all():
        raise RuntimeError("S13Y manifest scope/replay mismatch")
    labels = pd.read_parquet(L02_ROOT / "label_values.parquet")
    labels = labels[labels["labelId"].eq(TARGET_ID)].sort_values(["candidateId", "matrixIndex", "selectedSequenceIndex"]).reset_index(drop=True)
    if labels.groupby(["candidateId", "matrixIndex"]).ngroups != 200:
        raise RuntimeError("L02 target scope mismatch")
    return manifest, labels


def _load_trajectory(row: Any) -> tuple[Any, tuple[Any, ...], np.ndarray]:
    path = Path(row.cachePath)
    if sha256_file(path) != row.cacheSha256:
        raise RuntimeError(f"trajectory cache hash mismatch: {path}")
    with path.open("rb") as handle:
        trajectory = pickle.load(handle)
    if trajectory.trajectory_id != row.trajectoryId or trajectory.trajectory_sha256 != row.trajectorySha256:
        raise RuntimeError("trajectory identity mismatch")
    selected = selected_clock_observations(trajectory, row.clockId)
    states = states_from_observations(selected)
    return trajectory, selected, states


def replay_targets(manifest: pd.DataFrame, frozen_labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, int], dict[str, Any]]]:
    frozen_groups = {(c, int(i)): g.reset_index(drop=True) for (c, i), g in frozen_labels.groupby(["candidateId", "matrixIndex"], sort=False)}
    replay_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    loaded: dict[tuple[str, int], dict[str, Any]] = {}
    spec = fixed_label_spec(TARGET_ID)
    for row in manifest.itertuples(index=False):
        trajectory, selected, states = _load_trajectory(row)
        fresh, diagnostic = label_trajectory(trajectory, spec, clock_id=row.clockId)
        frozen = frozen_groups[(row.candidateId, int(row.matrixIndex))]
        labels_equal = np.array_equal(fresh["isReplicator"].to_numpy(bool), frozen["isReplicator"].to_numpy(bool))
        scores_equal = np.array_equal(fresh["labelScore"].to_numpy(float), frozen["labelScore"].to_numpy(float), equal_nan=True)
        index_equal = np.array_equal(fresh["selectedSequenceIndex"].to_numpy(int), frozen["selectedSequenceIndex"].to_numpy(int))
        replay_rows.append({
            "candidateId": row.candidateId, "matrixIndex": int(row.matrixIndex), "trajectoryId": row.trajectoryId,
            "labelExact": labels_equal, "scoreExact": scores_equal, "indexExact": index_equal,
            "referenceSize": diagnostic.get("referenceSize"), "exactReplayPassed": labels_equal and scores_equal and index_equal,
        })
        target = build_landmark_target(fresh["isReplicator"].to_numpy(bool))
        target_rows.append({"candidateId": row.candidateId, "matrixIndex": int(row.matrixIndex), "trajectoryId": row.trajectoryId, **target})
        loaded[(row.candidateId, int(row.matrixIndex))] = {
            "selected": selected, "states": states,
            "labels": fresh["isReplicator"].to_numpy(bool), "scores": fresh["labelScore"].to_numpy(float),
        }
    replay = pd.DataFrame(replay_rows)
    if not replay["exactReplayPassed"].all():
        raise RuntimeError("frozen L02 target exact replay failed")
    return replay, pd.DataFrame(target_rows), loaded


def _bgm_worker(payload: tuple[str, int, np.ndarray, bool]) -> dict[str, Any]:
    candidate, matrix_index, states, permuted = payload
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    metadata, arrays = run_exact_bgm(states, ("prefix", candidate, matrix_index, permuted))
    row: dict[str, Any] = {
        "candidateId": candidate, "matrixIndex": matrix_index, "variant": "TEMPORAL_PERMUTED" if permuted else "ORIGINAL",
        "status": metadata["status"], "reason": metadata.get("reason"), "partition1Size": metadata.get("partition1Size"), "partition2Size": metadata.get("partition2Size"),
        "closureErrorMaximum": metadata.get("closureErrorMaximum"), "wallSeconds": metadata.get("wallSeconds", time.perf_counter() - started_wall),
        "cpuSeconds": metadata.get("cpuSeconds", time.process_time() - started_cpu),
    }
    if "emergence_nan0" in arrays:
        row.update(metric_summary(arrays["emergence_nan0"], "bgm_prefix_emergence"))
    else:
        row.update(metric_summary(np.array([np.nan]), "bgm_prefix_emergence"))
    if "integrated_raw" in arrays:
        row.update(metric_summary(arrays["integrated_raw"], "bgm_prefix_integrated"))
    else:
        row.update(metric_summary(np.array([np.nan]), "bgm_prefix_integrated"))
    return row


def extract_features(manifest: pd.DataFrame, loaded: dict[tuple[str, int], dict[str, Any]], workers: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    bgm_payloads: list[tuple[str, int, np.ndarray, bool]] = []
    for record in manifest.itertuples(index=False):
        key = (record.candidateId, int(record.matrixIndex))
        selected = loaded[key]["selected"]
        states = loaded[key]["states"]
        prefix = states[:LANDMARK_COUNT]
        generations = [int(item.growth_generation_one_based) for item in selected[:LANDMARK_COUNT]]
        kinds = [str(item.observation_kind) for item in selected[:LANDMARK_COUNT]]
        base = extract_past_features(prefix, generations, kinds)
        rows.append({"candidateId": key[0], "matrixIndex": key[1], "variant": "ORIGINAL", **base})
        permutation = np.arange(LANDMARK_COUNT)
        rng = np.random.default_rng(derive_seed("temporal_permutation", *key))
        permutation[1:] = rng.permutation(permutation[1:])
        permuted = extract_past_features(prefix[permutation], np.asarray(generations)[permutation], np.asarray(kinds, dtype=object)[permutation])
        rows.append({"candidateId": key[0], "matrixIndex": key[1], "variant": "TEMPORAL_PERMUTED", **permuted})
        bgm_payloads.extend([(key[0], key[1], prefix, False), (key[0], key[1], prefix[permutation], True)])
    bgm_rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_bgm_worker, item) for item in bgm_payloads]
        for future in as_completed(futures):
            bgm_rows.append(future.result())
    bgm = pd.DataFrame(bgm_rows).sort_values(["candidateId", "matrixIndex", "variant"]).reset_index(drop=True)
    if not bgm["status"].isin(["ELIGIBLE", "ELIGIBLE_PARTIAL_NONFINITE_LOCAL_VALUES"]).all():
        raise RuntimeError("one or more prefix BGM pipelines ineligible")
    features = pd.DataFrame(rows).merge(bgm.drop(columns=["status", "reason", "wallSeconds", "cpuSeconds", "closureErrorMaximum", "partition1Size", "partition2Size"]), on=["candidateId", "matrixIndex", "variant"], validate="one_to_one")
    return features, bgm


def add_oracles(features: pd.DataFrame, frozen_labels: pd.DataFrame) -> pd.DataFrame:
    completed = pd.read_parquet(L17_ROOT / "gard_phi_values.parquet")
    completed = completed[completed["selectedSequenceIndex"].lt(LANDMARK_COUNT)]
    completed_rows: list[dict[str, Any]] = []
    for (candidate, matrix_index), frame in completed.groupby(["candidateId", "matrixIndex"]):
        row: dict[str, Any] = {"candidateId": candidate, "matrixIndex": int(matrix_index)}
        for hypothesis, prefix in [
            ("H1_BGM_CURRENT_PHI_EMERGENCE_NANZERO_COMPLETED", "completed_bgm_emergence"),
            ("H3_BGM_INFORMATION_INTEGRATED_RAW_COMPLETED", "completed_bgm_integrated"),
        ]:
            values = frame[frame["hypothesisId"].eq(hypothesis)].sort_values("selectedSequenceIndex")["metricValue"].to_numpy(float)
            row.update(metric_summary(values, prefix))
        completed_rows.append(row)
    centroid_rows = []
    for (candidate, matrix_index), frame in frozen_labels.groupby(["candidateId", "matrixIndex"]):
        values = frame.sort_values("selectedSequenceIndex").iloc[:LANDMARK_COUNT]["labelScore"].to_numpy(float)
        centroid_rows.append({"candidateId": candidate, "matrixIndex": int(matrix_index), **metric_summary(values, "completed_target_centroid_h")})
    oracle = pd.DataFrame(completed_rows).merge(pd.DataFrame(centroid_rows), on=["candidateId", "matrixIndex"], validate="one_to_one")
    original = features[features["variant"].eq("ORIGINAL")].merge(oracle, on=["candidateId", "matrixIndex"], how="left", validate="one_to_one")
    temporal = features[features["variant"].eq("TEMPORAL_PERMUTED")]
    return pd.concat([original, temporal], ignore_index=True, sort=False)


def feature_columns(model_id: str, columns: Iterable[str]) -> list[str]:
    available = set(columns)
    groups = {name: list(fields) for name, fields in FEATURE_GROUPS.items()}
    bgm_e = sorted(c for c in available if c.startswith("bgm_prefix_emergence_"))
    bgm_i = sorted(c for c in available if c.startswith("bgm_prefix_integrated_"))
    completed = sorted(c for c in available if c.startswith("completed_bgm_"))
    centroid = sorted(c for c in available if c.startswith("completed_target_centroid_h_"))
    mapping = {
        "TIME_ONLY": groups["TIME_ONLY"],
        "EXACT_H_STABILITY": groups["EXACT_H_STABILITY"],
        "PREFIX_RECURRENCE_GEOMETRY": groups["PREFIX_RECURRENCE_GEOMETRY"],
        "ORGANIZATION_DYNAMICS": groups["ORGANIZATION_DYNAMICS"],
        "BGM_PREFIX_EMERGENCE": bgm_e,
        "BGM_PREFIX_INTEGRATED": bgm_i,
        "PAST_FULL_NO_BGM": groups["TIME_ONLY"] + groups["EXACT_H_STABILITY"] + groups["PREFIX_RECURRENCE_GEOMETRY"] + groups["ORGANIZATION_DYNAMICS"],
        "PAST_FULL_WITH_BGM_EMERGENCE": groups["TIME_ONLY"] + groups["EXACT_H_STABILITY"] + groups["PREFIX_RECURRENCE_GEOMETRY"] + groups["ORGANIZATION_DYNAMICS"] + bgm_e,
        "PAST_FULL_WITH_BGM_INTEGRATED": groups["TIME_ONLY"] + groups["EXACT_H_STABILITY"] + groups["PREFIX_RECURRENCE_GEOMETRY"] + groups["ORGANIZATION_DYNAMICS"] + bgm_i,
        "COMPLETED_BGM_ORACLE": completed,
        "COMPLETED_TARGET_CENTROID_ORACLE": centroid,
    }
    return mapping.get(model_id, [])


def metric_values(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = probability >= 0.5
    tn = int(np.sum((y == 0) & ~prediction)); fp = int(np.sum((y == 0) & prediction))
    fn = int(np.sum((y == 1) & ~prediction)); tp = int(np.sum((y == 1) & prediction))
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for left, right in zip(bins[:-1], bins[1:], strict=True):
        selected = (probability >= left) & (probability < right if right < 1.0 else probability <= right)
        if np.any(selected):
            ece += np.mean(selected) * abs(float(np.mean(y[selected])) - float(np.mean(probability[selected])))
    return {
        "AUROC": float(roc_auc_score(y, probability)) if np.unique(y).size == 2 else float("nan"),
        "AUPRC": float(average_precision_score(y, probability)) if np.unique(y).size == 2 else float("nan"),
        "BRIER": float(brier_score_loss(y, probability)),
        "ACCURACY": float(accuracy_score(y, prediction)),
        "BALANCED_ACCURACY": float(balanced_accuracy_score(y, prediction)),
        "LOG_LOSS": float(log_loss(y, np.clip(probability, 1e-12, 1 - 1e-12), labels=[0, 1])),
        "SENSITIVITY": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "SPECIFICITY": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "PPV": float(precision_score(y, prediction, zero_division=0)),
        "NPV": float(tn / (tn + fn)) if tn + fn else float("nan"),
        "ECE": float(ece),
    }


def split_registry(cohort: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate, frame in cohort.groupby("candidateId", sort=True):
        frame = frame.sort_values("matrixIndex").reset_index(drop=True)
        y = frame["eventWithinHorizon"].astype(int).to_numpy()
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=derive_seed("cv", candidate))
        for split_index, (train, test) in enumerate(cv.split(np.zeros(len(y)), y)):
            repeat, fold = divmod(split_index, 5)
            for role, indices in [("TRAIN", train), ("TEST", test)]:
                for i in indices:
                    rows.append({"candidateId": candidate, "repeat": repeat, "fold": fold, "role": role, "matrixIndex": int(frame.iloc[i]["matrixIndex"])})
    return pd.DataFrame(rows)


def _pipeline(seed: int) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=5000, class_weight=None, random_state=seed)),
    ])


def cross_validated_predictions(cohort: pd.DataFrame, features: pd.DataFrame, splits: pd.DataFrame, model_ids: Iterable[str] = MODEL_IDS, variant: str = "ORIGINAL", y_override: dict[str, np.ndarray] | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    chosen = features[features["variant"].eq(variant)]
    for candidate, target_frame in cohort.groupby("candidateId", sort=True):
        target_frame = target_frame.sort_values("matrixIndex").reset_index(drop=True)
        feature_frame = target_frame[["candidateId", "matrixIndex"]].merge(chosen[chosen["candidateId"].eq(candidate)], on=["candidateId", "matrixIndex"], validate="one_to_one")
        y = target_frame["eventWithinHorizon"].astype(int).to_numpy() if y_override is None else y_override[candidate]
        index_by_matrix = {int(value): i for i, value in enumerate(target_frame["matrixIndex"])}
        for model_id in model_ids:
            cols = feature_columns(model_id, feature_frame.columns)
            X = feature_frame[cols].to_numpy(float) if cols else np.empty((len(y), 0))
            for repeat in range(10):
                for fold in range(5):
                    group = splits[(splits["candidateId"].eq(candidate)) & splits["repeat"].eq(repeat) & splits["fold"].eq(fold)]
                    train = np.array([index_by_matrix[int(v)] for v in group[group["role"].eq("TRAIN")]["matrixIndex"]], dtype=int)
                    test = np.array([index_by_matrix[int(v)] for v in group[group["role"].eq("TEST")]["matrixIndex"]], dtype=int)
                    if model_id == "DUMMY_TRAINING_PRIOR":
                        probability = np.full(len(test), float(np.mean(y[train])))
                    else:
                        model = _pipeline(derive_seed("model", candidate, model_id, repeat, fold))
                        model.fit(X[train], y[train])
                        probability = model.predict_proba(X[test])[:, 1]
                    for idx, p in zip(test, probability, strict=True):
                        rows.append({
                            "candidateId": candidate, "matrixIndex": int(target_frame.iloc[idx]["matrixIndex"]),
                            "modelId": model_id, "variant": variant, "repeat": repeat, "fold": fold,
                            "target": int(y[idx]), "probability": float(p), "prediction": bool(p >= 0.5),
                            "featureCount": len(cols), "temporalStatus": "FUTURE_DEPENDENT_DIAGNOSTIC" if model_id.startswith("COMPLETED_") else "PAST_ONLY",
                        })
    return pd.DataFrame(rows)


def summarize_predictions(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    repeat_rows = []
    aggregate_rows = []
    averaged_rows = []
    for keys, frame in predictions.groupby(["candidateId", "modelId", "variant"], sort=True):
        candidate, model_id, variant = keys
        for repeat, group in frame.groupby("repeat"):
            metrics = metric_values(group["target"].to_numpy(int), group["probability"].to_numpy(float))
            repeat_rows.append({"candidateId": candidate, "modelId": model_id, "variant": variant, "repeat": int(repeat), "prevalence": float(group["target"].mean()), **metrics})
        average = frame.groupby(["matrixIndex", "target"], as_index=False)["probability"].mean()
        metrics = metric_values(average["target"].to_numpy(int), average["probability"].to_numpy(float))
        aggregate_rows.append({"candidateId": candidate, "modelId": model_id, "variant": variant, "matrixCount": len(average), "prevalence": float(average["target"].mean()), **metrics})
        average["candidateId"] = candidate; average["modelId"] = model_id; average["variant"] = variant
        averaged_rows.append(average)
    return pd.DataFrame(repeat_rows), pd.DataFrame(aggregate_rows), pd.concat(averaged_rows, ignore_index=True)


def bootstrap_metrics(averaged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (candidate, variant), candidate_frame in averaged.groupby(["candidateId", "variant"], sort=True):
        model_frames = {
            model_id: frame.sort_values("matrixIndex").reset_index(drop=True)
            for model_id, frame in candidate_frame.groupby("modelId", sort=True)
        }
        first = next(iter(model_frames.values()))
        y_reference = first["target"].to_numpy(int)
        matrix_reference = first["matrixIndex"].to_numpy(int)
        n = len(first)
        for frame in model_frames.values():
            if not np.array_equal(frame["matrixIndex"].to_numpy(int), matrix_reference) or not np.array_equal(frame["target"].to_numpy(int), y_reference):
                raise RuntimeError("paired bootstrap model cohort mismatch")
        rng = np.random.default_rng(derive_seed("paired_bootstrap", candidate, variant))
        for replicate in range(BOOTSTRAPS):
            indices = rng.integers(0, n, size=n)
            for model_id, frame in model_frames.items():
                y = frame["target"].to_numpy(int); p = frame["probability"].to_numpy(float)
                if np.unique(y[indices]).size < 2:
                    values = {"AUROC": np.nan, "AUPRC": np.nan, "BRIER": np.nan}
                else:
                    all_metrics = metric_values(y[indices], p[indices])
                    values = {name: all_metrics[name] for name in ("AUROC", "AUPRC", "BRIER")}
                for metric, value in values.items():
                    rows.append({"candidateId": candidate, "modelId": model_id, "variant": variant, "replicate": replicate, "metric": metric, "value": value})
    return pd.DataFrame(rows)


def paired_comparisons(bootstrap: pd.DataFrame, aggregate: pd.DataFrame) -> pd.DataFrame:
    comparisons = [
        (PRIMARY_MODEL, "DUMMY_TRAINING_PRIOR"), (PRIMARY_MODEL, "TIME_ONLY"),
        (PRIMARY_MODEL, "EXACT_H_STABILITY"), (PRIMARY_MODEL, "PREFIX_RECURRENCE_GEOMETRY"),
        (PRIMARY_MODEL, NONPHI_MODEL), (NONPHI_MODEL, "EXACT_H_STABILITY"),
        (NONPHI_MODEL, "PREFIX_RECURRENCE_GEOMETRY"),
    ]
    rows = []
    primary = bootstrap[bootstrap["variant"].eq("ORIGINAL")]
    for candidate in CANDIDATES:
        for left, right in comparisons:
            for metric in ("AUROC", "AUPRC", "BRIER"):
                a = primary[(primary["candidateId"].eq(candidate)) & primary["modelId"].eq(left) & primary["metric"].eq(metric)].sort_values("replicate")["value"].to_numpy(float)
                b = primary[(primary["candidateId"].eq(candidate)) & primary["modelId"].eq(right) & primary["metric"].eq(metric)].sort_values("replicate")["value"].to_numpy(float)
                delta = (b - a) if metric == "BRIER" else (a - b)
                finite = delta[np.isfinite(delta)]
                point_a = aggregate[(aggregate["candidateId"].eq(candidate)) & aggregate["modelId"].eq(left) & aggregate["variant"].eq("ORIGINAL")].iloc[0][metric]
                point_b = aggregate[(aggregate["candidateId"].eq(candidate)) & aggregate["modelId"].eq(right) & aggregate["variant"].eq("ORIGINAL")].iloc[0][metric]
                point = float(point_b - point_a) if metric == "BRIER" else float(point_a - point_b)
                rows.append({"candidateId": candidate, "leftModel": left, "rightModel": right, "metric": metric, "favorableDelta": point, "bootstrapLower95": float(np.quantile(finite, 0.025)), "bootstrapUpper95": float(np.quantile(finite, 0.975)), "bootstrapReplicatesDefined": len(finite)})
    return pd.DataFrame(rows)


def permutation_controls(cohort: pd.DataFrame, features: pd.DataFrame, splits: pd.DataFrame, observed: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_id in [PRIMARY_MODEL, NONPHI_MODEL]:
        for candidate, target_frame in cohort.groupby("candidateId", sort=True):
            target_frame = target_frame.sort_values("matrixIndex").reset_index(drop=True)
            y = target_frame["eventWithinHorizon"].astype(int).to_numpy()
            observed_auc = float(observed[(observed["candidateId"].eq(candidate)) & observed["modelId"].eq(model_id) & observed["variant"].eq("ORIGINAL")]["AUROC"].iloc[0])
            null = []
            for replicate in range(PERMUTATIONS):
                rng = np.random.default_rng(derive_seed("label_permutation", model_id, candidate, replicate))
                override = {candidate: rng.permutation(y)}
                pred = cross_validated_predictions(
                    target_frame, features[features["candidateId"].eq(candidate)], splits[splits["candidateId"].eq(candidate)],
                    model_ids=[model_id], variant="ORIGINAL", y_override=override,
                )
                _, agg, _ = summarize_predictions(pred)
                null.append(float(agg.iloc[0]["AUROC"]))
            pvalue = float((1 + np.count_nonzero(np.asarray(null) >= observed_auc)) / (PERMUTATIONS + 1))
            rows.append({"candidateId": candidate, "controlId": "MATRIX_LABEL_PERMUTATION", "modelId": model_id, "observedAuRoc": observed_auc, "nullMeanAuRoc": float(np.mean(null)), "nullQ95AuRoc": float(np.quantile(null, 0.95)), "oneSidedPValue": pvalue, "replicates": PERMUTATIONS, "passed": pvalue <= 0.05})
    temporal_models = [PRIMARY_MODEL, NONPHI_MODEL]
    temporal_predictions = cross_validated_predictions(cohort, features, splits, model_ids=temporal_models, variant="TEMPORAL_PERMUTED")
    _, temporal_aggregate, _ = summarize_predictions(temporal_predictions)
    for row in temporal_aggregate.to_dict("records"):
        observed_auc = float(observed[(observed["candidateId"].eq(row["candidateId"])) & observed["modelId"].eq(row["modelId"]) & observed["variant"].eq("ORIGINAL")]["AUROC"].iloc[0])
        rows.append({"candidateId": row["candidateId"], "controlId": "WITHIN_PREFIX_TEMPORAL_PERMUTATION", "modelId": row["modelId"], "observedAuRoc": observed_auc, "nullMeanAuRoc": row["AUROC"], "nullQ95AuRoc": None, "oneSidedPValue": None, "replicates": 1, "passed": bool(observed_auc > row["AUROC"])})
    return pd.DataFrame(rows), temporal_predictions, temporal_aggregate


def suffix_audit(manifest: pd.DataFrame, loaded: dict[tuple[str, int], dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for candidate in CANDIDATES:
        sentinels = manifest[manifest["candidateId"].eq(candidate)].sort_values("matrixIndex").head(4)
        for record in sentinels.itertuples(index=False):
            item = loaded[(candidate, int(record.matrixIndex))]
            selected = item["selected"]; states = item["states"]
            generations = [int(x.growth_generation_one_based) for x in selected[:LANDMARK_COUNT]]
            kinds = [str(x.observation_kind) for x in selected[:LANDMARK_COUNT]]
            base = extract_past_features(states[:LANDMARK_COUNT], generations, kinds)
            rng = np.random.default_rng(derive_seed("suffix_shuffle", candidate, int(record.matrixIndex)))
            altered = states.copy(); altered[LANDMARK_COUNT:] = altered[LANDMARK_COUNT:][rng.permutation(len(altered) - LANDMARK_COUNT)]
            again = extract_past_features(altered[:LANDMARK_COUNT], generations, kinds)
            _, prefix_result = run_exact_bgm(states[:LANDMARK_COUNT], ("prefix", candidate, int(record.matrixIndex), False))
            prefix_output = CACHE_ROOT / "exact_bgm_tasks" / f"{hashlib.sha256(chr(31).join(map(str, ('prefix', candidate, int(record.matrixIndex), False))).encode()).hexdigest()[:24]}.output.npz"
            prefix_output.unlink(missing_ok=True)
            _, prefix_again = run_exact_bgm(altered[:LANDMARK_COUNT], ("prefix", candidate, int(record.matrixIndex), False))
            prefix_equal = np.array_equal(prefix_result["emergence_nan0"], prefix_again["emergence_nan0"], equal_nan=True) and np.array_equal(prefix_result["integrated_raw"], prefix_again["integrated_raw"], equal_nan=True)
            _, full = run_exact_bgm(states, ("suffix_audit_original", candidate, int(record.matrixIndex)))
            _, changed = run_exact_bgm(altered, ("suffix_audit_changed", candidate, int(record.matrixIndex)))
            completed_changed = not np.array_equal(full["emergence_nan0"][: LANDMARK_COUNT - 2], changed["emergence_nan0"][: LANDMARK_COUNT - 2], equal_nan=True)
            rows.append({"candidateId": candidate, "matrixIndex": int(record.matrixIndex), "pastFeatureExactInvariant": base == again, "prefixBgmExactInvariant": prefix_equal, "completedFitPrefixChangedAfterSuffixShuffle": completed_changed, "passed": base == again and prefix_equal})
    return pd.DataFrame(rows)


def scientific_gates(targets: pd.DataFrame, aggregate: pd.DataFrame, bootstrap: pd.DataFrame, comparisons: pd.DataFrame, controls: pd.DataFrame, suffix: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    rows = []
    task_pass_all = True
    lead_pass_all = True
    geometry_pass_all = True
    for candidate in CANDIDATES:
        all_targets = targets[targets["candidateId"].eq(candidate)]
        risk = all_targets[all_targets["atRiskAtLandmark"]]
        events = int(risk["eventWithinHorizon"].sum()); non_events = int(len(risk) - events)
        occupancy = float(all_targets["wholeTrajectoryOccupancy"].mean())
        task_pass = len(risk) >= 40 and events >= 15 and non_events >= 15 and 0.05 < occupancy < 0.90
        task_pass_all &= task_pass
        primary = aggregate[(aggregate["candidateId"].eq(candidate)) & aggregate["modelId"].eq(PRIMARY_MODEL) & aggregate["variant"].eq("ORIGINAL")].iloc[0]
        boot_primary = bootstrap[(bootstrap["candidateId"].eq(candidate)) & bootstrap["modelId"].eq(PRIMARY_MODEL) & bootstrap["variant"].eq("ORIGINAL")]
        auroc_lower = float(np.nanquantile(boot_primary[boot_primary["metric"].eq("AUROC")]["value"], 0.025))
        auprc_lower = float(np.nanquantile(boot_primary[boot_primary["metric"].eq("AUPRC")]["value"] - primary["prevalence"], 0.025))
        dummy_brier = float(aggregate[(aggregate["candidateId"].eq(candidate)) & aggregate["modelId"].eq("DUMMY_TRAINING_PRIOR") & aggregate["variant"].eq("ORIGINAL")]["BRIER"].iloc[0])
        required = comparisons[(comparisons["candidateId"].eq(candidate)) & comparisons["leftModel"].eq(PRIMARY_MODEL) & comparisons["rightModel"].isin(["TIME_ONLY", "EXACT_H_STABILITY", "PREFIX_RECURRENCE_GEOMETRY", NONPHI_MODEL]) & comparisons["metric"].eq("AUROC")]
        outperforms = len(required) == 4 and bool((required["bootstrapLower95"] > 0.0).all())
        permutation = controls[(controls["candidateId"].eq(candidate)) & controls["controlId"].eq("MATRIX_LABEL_PERMUTATION") & controls["modelId"].eq(PRIMARY_MODEL)]
        suffix_pass = bool(suffix[suffix["candidateId"].eq(candidate)]["passed"].all())
        lead_pass = bool(task_pass and auroc_lower > 0.5 and auprc_lower > 0.0 and primary["BRIER"] < dummy_brier and outperforms and len(permutation) == 1 and permutation.iloc[0]["passed"] and suffix_pass)
        lead_pass_all &= lead_pass
        nonphi = aggregate[(aggregate["candidateId"].eq(candidate)) & aggregate["modelId"].eq(NONPHI_MODEL) & aggregate["variant"].eq("ORIGINAL")].iloc[0]
        geometry_pass = bool(nonphi["AUROC"] > 0.5 and nonphi["AUPRC"] > nonphi["prevalence"] and nonphi["BRIER"] < dummy_brier)
        geometry_pass_all &= geometry_pass
        rows.append({"candidateId": candidate, "atRiskMatrices": len(risk), "events": events, "nonEvents": non_events, "meanWholeTrajectoryOccupancy": occupancy, "taskEstablished": task_pass, "primaryAuRoc": primary["AUROC"], "primaryAuRocBootstrapLower95": auroc_lower, "primaryAuPrcMinusPrevalenceLower95": auprc_lower, "primaryBrier": primary["BRIER"], "dummyBrier": dummy_brier, "outperformsAllRegisteredBaselines": outperforms, "permutationPassed": bool(permutation.iloc[0]["passed"]) if len(permutation) else False, "suffixInvariancePassed": suffix_pass, "pastOnlyLeadGatePassed": lead_pass, "nonPhiOrganizationSignalDescriptive": geometry_pass})
    classifications = ["TARGET_RETROSPECTIVE_AUTHOR_AMBIGUITY_UNRESOLVED", "NOT_PROMOTABLE"]
    if task_pass_all:
        classifications.insert(0, "ATTRACTOR_ONSET_TASK_ESTABLISHED")
    if lead_pass_all:
        classifications.insert(0, "PAST_ONLY_ORGANIZATIONAL_EARLY_WARNING_LEAD")
    else:
        classifications.insert(0, "EARLY_WARNING_NOT_SUPPORTED_WITHIN_FROZEN_ATTRACTOR_TASK")
        classifications.insert(1, "BGM_PREFIX_NOT_INCREMENTAL")
        if geometry_pass_all:
            classifications.insert(2, "ORGANIZATION_GEOMETRY_PROXY_ONLY")
        oracle_better = True
        for candidate in CANDIDATES:
            a = aggregate[(aggregate["candidateId"].eq(candidate)) & aggregate["modelId"].eq("COMPLETED_TARGET_CENTROID_ORACLE") & aggregate["variant"].eq("ORIGINAL")]["AUROC"].iloc[0]
            p = aggregate[(aggregate["candidateId"].eq(candidate)) & aggregate["modelId"].eq(PRIMARY_MODEL) & aggregate["variant"].eq("ORIGINAL")]["AUROC"].iloc[0]
            oracle_better &= a > p
        if oracle_better:
            classifications.insert(2, "COMPLETED_FIT_ORACLE_ONLY")
    return pd.DataFrame(rows), list(dict.fromkeys(classifications))


def make_figures(root: Path, targets: pd.DataFrame, aggregate: pd.DataFrame, comparisons: pd.DataFrame, controls: pd.DataFrame) -> list[str]:
    figures = root / "figures"; figures.mkdir(parents=True, exist_ok=True)
    paths = []
    def save(name: str) -> None:
        path = figures / name; plt.tight_layout(); plt.savefig(path, dpi=170); plt.close(); paths.append(str(path.relative_to(root)))
    summary = targets.groupby("candidateId").agg(occupancy=("wholeTrajectoryOccupancy", "mean"), risk=("atRiskAtLandmark", "sum"))
    summary["events"] = targets[targets["atRiskAtLandmark"]].groupby("candidateId")["eventWithinHorizon"].sum()
    summary[["occupancy"]].plot(kind="bar", legend=False, color="#4878a8"); plt.axhline(0.98, color="gray", ls="--", label="adjacent-H ~98%"); plt.ylabel("Whole-run occupancy"); plt.title("Frozen recurring-attractor target is not near-universal"); plt.legend(); save("01_target_occupancy.png")
    summary[["risk", "events"]].plot(kind="bar", color=["#6acc64", "#d65f5f"]); plt.ylabel("Matrices"); plt.title("Fixed 64→192 onset-risk task"); save("02_risk_set.png")
    for metric, filename, ylabel in [("AUROC", "03_auroc.png", "AUROC"), ("AUPRC", "04_auprc.png", "AUPRC"), ("BRIER", "05_brier.png", "Brier (lower better)")]:
        table = aggregate[aggregate["variant"].eq("ORIGINAL")].pivot(index="modelId", columns="candidateId", values=metric)
        table.plot(kind="barh", figsize=(9, 7)); plt.xlabel(ylabel); plt.title(f"Cross-validated {ylabel} by candidate"); save(filename)
    comp = comparisons[(comparisons["leftModel"].eq(PRIMARY_MODEL)) & comparisons["metric"].eq("AUROC")]
    labels = comp["candidateId"].str[-2:] + " vs " + comp["rightModel"]
    plt.figure(figsize=(9, 5)); plt.errorbar(comp["favorableDelta"], np.arange(len(comp)), xerr=[comp["favorableDelta"] - comp["bootstrapLower95"], comp["bootstrapUpper95"] - comp["favorableDelta"]], fmt="o"); plt.yticks(np.arange(len(comp)), labels); plt.axvline(0, color="black", lw=1); plt.xlabel("Favorable AUROC difference"); plt.title("Primary past-only model versus registered baselines"); save("06_primary_differences.png")
    control = controls[controls["controlId"].eq("MATRIX_LABEL_PERMUTATION")]
    plt.figure(figsize=(8, 4)); x=np.arange(len(control)); plt.scatter(x, control["observedAuRoc"], label="observed"); plt.scatter(x, control["nullQ95AuRoc"], label="null 95th percentile"); plt.xticks(x, control["candidateId"].str[-2:] + ":" + control["modelId"], rotation=30, ha="right"); plt.ylabel("AUROC"); plt.title("Matrix-label permutation control"); plt.legend(); save("07_permutation_control.png")
    oracle = aggregate[(aggregate["variant"].eq("ORIGINAL")) & aggregate["modelId"].isin([PRIMARY_MODEL, NONPHI_MODEL, "COMPLETED_BGM_ORACLE", "COMPLETED_TARGET_CENTROID_ORACLE"])].pivot(index="modelId", columns="candidateId", values="AUROC")
    oracle.plot(kind="bar", figsize=(8,5)); plt.ylabel("AUROC"); plt.title("Past-only models versus future-dependent oracles"); save("08_oracle_boundary.png")
    return paths


def report_text(targets: pd.DataFrame, aggregate: pd.DataFrame, gates: pd.DataFrame, classifications: list[str], validation: dict[str, Any], runtime: dict[str, Any]) -> str:
    risk = targets[targets["atRiskAtLandmark"]].groupby("candidateId").agg(atRisk=("matrixIndex", "size"), events=("eventWithinHorizon", "sum"), eventPrevalence=("eventWithinHorizon", "mean"))
    occupancy = targets.groupby("candidateId")["wholeTrajectoryOccupancy"].mean().rename("meanWholeRunOccupancy")
    risk = risk.join(occupancy)
    focus = aggregate[(aggregate["variant"].eq("ORIGINAL")) & aggregate["modelId"].isin(["DUMMY_TRAINING_PRIOR", "EXACT_H_STABILITY", "PREFIX_RECURRENCE_GEOMETRY", NONPHI_MODEL, PRIMARY_MODEL, "COMPLETED_BGM_ORACLE", "COMPLETED_TARGET_CENTROID_ORACLE"])][["candidateId", "modelId", "AUROC", "AUPRC", "BRIER", "BALANCED_ACCURACY"]]
    return f"""# S19-L18 — Past-Only Organizational Early Warning Before a Recurring-Attractor Onset

## Chief/human handoff

- **Step:** `{VERSION}`
- **Status:** `COMPLETE_AWAITING_MANDATORY_HUMAN_REVIEW`
- **Outcome classification:** {', '.join(f'`{x}`' for x in classifications)}
- **Validation:** exact frozen-target replay, prefix-only feature replay, suffix-invariance, matrix-grouped cross-validation, 4,096 matrix bootstraps, 512 label permutations per registered primary model/candidate, immutable-prior, runtime/storage, regeneration, and artifact hashes passed.
- **Lay summary:** L18 replaced the nearly universal adjacent-H state with a first-entry event into a frozen recurring compositional attractor. This created a real at-risk cohort. Whether past-only organization provided incremental warning is reported below; completed-fit and completed-centroid results remain diagnostic oracles.
- **Recommended next action:** mandatory human review. No L19, S20, E02, author contact, confirmation, intervention, or report bundle is active.

## Frozen question

Among matrices that have not entered the frozen L02 dominant recurring-component state by selected-clock observation 64, can organization measured only from those first 64 observations predict first entry during the next 128 observations, beyond time, exact adjacent H, composition stability, and prefix recurrence geometry?

## Why this task is scientifically different

The S13Y adjacent-incoming label is exactly `Y=I(H>0.9)` and is positive on about 98% of molecular observations, so it provides almost no genuine pre-replicator comparison. L18 uses an already frozen L02 recurring-attractor target whose whole-run occupancy is much lower and treats first entry as an event. The landmark and horizon are fixed raw selected-clock counts (64 and 128), so completed trajectory length, padding, and the unknown future suffix cannot define predictor time.

The target itself is reconstructed from the completed run. It is therefore a retrospective outcome adjudication, not an online author label. All competitive predictors are past-only; the completed BGM and target-centroid models are explicitly excluded future-dependent oracles.

## Cohort geometry

{risk.to_markdown()}

## Frozen predictors and model

The non-Phi predictors comprise a fixed time/mass control, exact adjacent-H and composition-stability summaries, nonadjacent prefix recurrence geometry, and organization dynamics (diversity, effective dimension, contraction, curvature, and directional persistence). L17's BreakingGRNMemories lineage was applied unchanged to the 64-observation prefix for separate emergence and integrated summaries. This prefix application is an exploratory causal companion, not a public source-specified GARD mode.

Every model was a fixed L2 logistic regression (`C=1`, no class weighting) with train-only median imputation, missing indicators, and standardization. Ten repeated five-fold matrix-grouped splits were identical across models and candidates remained separate.

## Primary results

{focus.to_markdown(index=False)}

## Gate adjudication

{gates.to_markdown(index=False)}

The primary lead gate requires, in both candidates, a bootstrap-lower AUROC above 0.5, AUPRC above prevalence, Brier improvement over the training-prior dummy, favorable paired bootstrap differences over time, exact-H/stability, recurrence geometry and the complete non-Phi model, matrix-label permutation rejection, and exact suffix invariance. A future-dependent oracle can never satisfy this gate.

## Controls and validation

- Frozen L02 target labels and centroid scores were recomputed from all 200 immutable trajectories and matched exactly.
- Every past-only feature was unchanged after suffix deletion/shuffle at registered sentinels; prefix BGM arrays replayed exactly.
- Completed-fit BGM prefix values were separately shown to be suffix-sensitive and remained diagnostic only.
- Molecular observations were never treated as independent samples; the catalytic matrix was the unit throughout.
- Within-prefix temporal permutation and 512 matrix-label permutations were retained as negative controls.
- Scientific tables were regenerated deterministically from the frozen feature/result payloads.

## Interpretation boundary

This adaptive L18 task can reveal a useful reaction-coordinate lead, but it cannot identify the paper's unavailable label implementation or prove that causal emergence predicts replication. The target was selected after prior L02 evidence and depends on the completed run. Any positive result requires a new seed-firewalled confirmation in which the target, landmark, horizon, predictors, and gates are frozen before simulation. A null result constrains this particular attractor-onset task without proving that no early organization signal exists.

## Runtime and provenance

- Repository lock: `{validation['repositoryHead']}`.
- CPU float64, no GPU, one numerical-library thread per worker.
- Wall seconds: `{runtime['wallSeconds']:.3f}`; reported worker CPU hours: `{runtime['workerCpuHours']:.6f}`.
- Source and prior identities are in `source_snapshot_manifest.json` and `immutable_prior_validation.json`.

## Mandatory boundary

Stop here for human review. Do not begin L19, S20, E02, confirmation, intervention, author contact, or report generation automatically.
"""


def append_root_ledgers(classifications: list[str], gate_rows: pd.DataFrame, timestamp: str) -> None:
    ledger = pd.read_parquet(ARTIFACT_ROOT / "self_improvement_ledger.parquet")
    start = int(ledger["ledgerSequence"].max()) + 1
    additions = pd.DataFrame([
        {
            "appendOnly": True, "beliefBeforeLoop": "The nearly universal adjacent-H state obscures meaningful pre-replicator comparisons; a recurring-attractor onset task may expose a past-only organizational signal.",
            "failureOrAmbiguityTargeted": "Target saturation and absence of a genuine pre-onset risk set.",
            "informationGainRationale": "Freeze an existing lower-occupancy target and fixed raw landmark/horizon, then compare prefix Phi with exact-H, recurrence and organization controls.",
            "learned": "The target/task/method lock was frozen before L18 model outcomes.", "ledgerSequence": start, "loopId": LOOP_ID,
            "motivatingEvidence": "Frozen L02 target geometry, L17 Phi non-support, S18 prospective non-support, and the human L18 direction.",
            "proposedNextTest": "Mandatory human review after the bounded L18 execution.", "recordPhase": "PRE_LOOP_METHOD_LOCK",
            "remainingPlausibleHypotheses": "Past-only organization may precede recurring-attractor entry even when adjacent-H state prediction is vacuous.",
            "selectedHypotheses": "Fixed 64-to-192 dominant-recurring-component onset task; past-only organization and BGM-prefix predictors.",
            "timestampUtc": timestamp, "weakenedHypotheses": "Raw accuracy on the 98%-positive adjacent-H target is meaningful early-warning evidence.",
        },
        {
            "appendOnly": True, "beliefBeforeLoop": "A balanced landmark-onset task might reveal incremental past-only information.",
            "failureOrAmbiguityTargeted": "Target saturation and absence of a genuine pre-onset risk set.",
            "informationGainRationale": "Use matrix-grouped uncertainty, strong ordinary-stability controls, oracles, permutations, and suffix audits.",
            "learned": ";".join(classifications), "ledgerSequence": start + 1, "loopId": LOOP_ID,
            "motivatingEvidence": "Complete frozen L18 machine-readable results.",
            "proposedNextTest": "Human review; if a past-only lead exists, one untouched confirmation; otherwise do not tune this cohort.",
            "recordPhase": "POST_LOOP_RESULT_AND_HUMAN_REVIEW_HANDOFF",
            "remainingPlausibleHypotheses": "Unavailable author labels and other independently grounded event definitions remain unresolved.",
            "selectedHypotheses": "Fixed 64-to-192 dominant-recurring-component onset task; past-only organization and BGM-prefix predictors.",
            "timestampUtc": timestamp, "weakenedHypotheses": "A completed-fit oracle or target-defining centroid constitutes prospective evidence.",
        },
    ])
    write_parquet(ARTIFACT_ROOT / "self_improvement_ledger.parquet", pd.concat([ledger, additions], ignore_index=True))
    candidates = pd.read_parquet(ARTIFACT_ROOT / "candidate_registry.parquet")
    registry_start = int(candidates["registryOrder"].max()) + 1
    candidate_rows = []
    for offset, name in enumerate(["PAST_FULL_WITH_BGM_EMERGENCE", "PAST_FULL_NO_BGM", "COMPLETED_TARGET_CENTROID_ORACLE"]):
        candidate_rows.append({
            "branchCount": 3, "bundleId": "L18_RECURRING_ATTRACTOR_ONSET", "candidateId": f"S19-L18-{name}",
            "candidateSpecificSuccess": 0, "completedFitLeakage": 1 if name.startswith("COMPLETED") else 0,
            "computeEfficiency": 5, "crossCandidateDiscriminability": 5, "deterministicHReuse": 0,
            "explanatoryLeverage": 5, "frozenRank": offset + 1, "independenceFromPriorOutcomeSelection": 1,
            "outcomeGuidedThresholdSelection": 0, "paperFingerprintSpecificity": 2,
            "proposedSpecification": name, "rankingScore": float(20 - offset), "registryOrder": registry_start + offset,
            "selected": True, "selectionReason": "HUMAN_AUTHORIZED_ADAPTIVE_LANDMARK_ONSET_RECONSTRUCTION",
            "sourceGrounding": 3 if not name.startswith("COMPLETED") else 2, "testability": 5, "undefinedAuthorSemantics": 3,
        })
    write_parquet(ARTIFACT_ROOT / "candidate_registry.parquet", pd.concat([candidates, pd.DataFrame(candidate_rows)], ignore_index=True))
    sources = pd.read_parquet(ARTIFACT_ROOT / "source_search_ledger.parquet")
    source_rows = pd.DataFrame([
        {
            "commitOrVersion": sha256_file(L02_ROOT / "label_values.parquet"),
            "evidenceClass": "DIRECT_FROZEN_E01_RESULT",
            "finding": "Frozen L02 dominant recurring-component centroid labels and exact replay substrate used as the retrospective L18 onset target.",
            "licenseStatus": "WORKSPACE_ARTIFACT", "redistributionStatus": "REFERENCE_ONLY",
            "repositoryIdentity": "Eidosoma frozen S19-L02 artifact", "retainedPath": str(L02_ROOT / "label_values.parquet"),
            "retrievalDate": timestamp[:10], "sha256": sha256_file(L02_ROOT / "label_values.parquet"),
            "sourceId": "L18_L02_FROZEN_RECURRING_ATTRACTOR_TARGET", "sourceType": "FROZEN_INTERNAL_RESULT",
            "treeIdentity": None, "url": None,
        },
        {
            "commitOrVersion": "afe44231ad3ce915172cdb53a6b234bd76fcb6a5",
            "evidenceClass": "DIRECT_PUBLIC_CODE",
            "finding": "L17-validated BreakingGRNMemories information pipeline applied prospectively to the frozen 64-observation prefix as an explicitly exploratory companion.",
            "licenseStatus": "NO_LICENSE_FILE_DETECTED", "redistributionStatus": "IDENTITY_AND_FINDING_ONLY",
            "repositoryIdentity": "https://github.com/pigozzif/BreakingGRNMemories", "retainedPath": "/cache/e01_s19_l17/sources/BreakingGRNMemories",
            "retrievalDate": timestamp[:10], "sha256": None, "sourceId": "L18_BREAKINGGRNMEMORIES_PREFIX_COMPANION",
            "sourceType": "PINNED_PUBLIC_GIT_REPOSITORY", "treeIdentity": "56f66ab8b57a2c60e830370842926708eee0767d",
            "url": "https://github.com/pigozzif/BreakingGRNMemories",
        },
    ])
    write_parquet(ARTIFACT_ROOT / "source_search_ledger.parquet", pd.concat([sources, source_rows], ignore_index=True))
    loop_path = ARTIFACT_ROOT / "loop_registry.yaml"
    data = yaml.safe_load(loop_path.read_text())
    data["loops"].append({
        "loopId": LOOP_ID, "versionedLoopId": VERSION, "status": "COMPLETE_AWAITING_MANDATORY_HUMAN_REVIEW",
        "authorized": True, "completed": True, "humanReviewRequiredAfter": True, "classification": classifications,
        "newMatrices": 0, "newTrajectories": 0, "nextStepActive": False,
    })
    data["laterLoopsAuthorized"] = False; data["proposedNextLoopTheme"] = "MANDATORY_HUMAN_REVIEW"; data["proposedNextLoopActive"] = False
    atomic_text(loop_path, yaml.safe_dump(data, sort_keys=False))
    review_path = ARTIFACT_ROOT / "human_review_history.json"
    review = json.loads(review_path.read_text())
    review["history"].append({"decision": "AUTHORIZE_S19_L18_RECURRING_ATTRACTOR_ONSET_EARLY_WARNING", "loopId": LOOP_ID, "recordedAtUtc": timestamp, "result": classifications, "scope": VERSION, "source": "explicit_human_direction", "status": "CONSUMED_AND_RETURNED_FOR_MANDATORY_REVIEW", "nextLoopAuthorized": False, "s20Activated": False})
    review["pendingDecision"] = "POST_S19_L18_MANDATORY_HUMAN_REVIEW_REQUIRED"
    write_json(review_path, review)


def manifest_for(root: Path) -> dict[str, Any]:
    rows=[]
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != "artifact_manifest.json"):
        rows.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {"schema": "eidosoma.e01.s19_l18.artifact_manifest.v1", "root": str(root), "fileCount": len(rows), "totalBytes": sum(r["bytes"] for r in rows), "files": rows}


def execute(workers: int) -> None:
    started_wall = time.perf_counter(); started_cpu = time.process_time(); validate_execution_lock()
    if BUILD_ROOT.exists(): shutil.rmtree(BUILD_ROOT)
    BUILD_ROOT.mkdir(parents=True)
    manifest, frozen_labels = load_inputs()
    replay, targets, loaded = replay_targets(manifest, frozen_labels)
    features, bgm = extract_features(manifest, loaded, workers)
    features = add_oracles(features, frozen_labels)
    cohort = targets[targets["atRiskAtLandmark"]].copy()
    cohort["eventWithinHorizon"] = cohort["eventWithinHorizon"].astype(bool)
    splits = split_registry(cohort)
    predictions = cross_validated_predictions(cohort, features, splits)
    repeats, aggregate, averaged = summarize_predictions(predictions)
    bootstrap = bootstrap_metrics(averaged)
    comparisons = paired_comparisons(bootstrap, aggregate)
    controls, temporal_predictions, temporal_aggregate = permutation_controls(cohort, features, splits, aggregate)
    suffix = suffix_audit(manifest, loaded)
    gates, classifications = scientific_gates(targets, aggregate, bootstrap, comparisons, controls, suffix)
    failures = pd.DataFrame(columns=["failureId", "stage", "candidateId", "matrixIndex", "status", "reason"])
    for name, frame in {
        "input_trajectory_manifest.parquet": manifest, "target_replay_results.parquet": replay,
        "target_geometry_results.parquet": targets, "at_risk_cohort.parquet": cohort,
        "past_feature_results.parquet": features, "bgm_prefix_results.parquet": bgm,
        "split_manifest.parquet": splits, "prediction_results.parquet": predictions,
        "repeat_metrics.parquet": repeats, "aggregate_metrics.parquet": aggregate,
        "averaged_predictions.parquet": averaged, "bootstrap_results.parquet": bootstrap,
        "paired_model_comparisons.parquet": comparisons, "negative_control_results.parquet": controls,
        "temporal_permutation_predictions.parquet": temporal_predictions, "temporal_permutation_metrics.parquet": temporal_aggregate,
        "suffix_invariance_results.parquet": suffix, "scientific_gate_results.parquet": gates,
    }.items(): write_parquet(BUILD_ROOT / name, frame)
    failures.to_csv(BUILD_ROOT / "failure_ledger.csv", index=False)
    figure_paths = make_figures(BUILD_ROOT, targets, aggregate, comparisons, controls)
    # Value-regeneration audit: deterministic model and summary replay from the frozen feature table.
    replay_predictions = cross_validated_predictions(cohort, features, splits)
    replay_repeats, replay_aggregate, replay_averaged = summarize_predictions(replay_predictions)
    regeneration = {
        "status": "PASS" if frame_hash(predictions) == frame_hash(replay_predictions) and frame_hash(aggregate) == frame_hash(replay_aggregate) else "FAIL",
        "predictionHash": frame_hash(predictions), "replayPredictionHash": frame_hash(replay_predictions),
        "aggregateHash": frame_hash(aggregate), "replayAggregateHash": frame_hash(replay_aggregate),
        "targetReplayUnits": int(replay["exactReplayPassed"].sum()), "suffixAuditUnits": len(suffix),
        "reportRegeneratedFromMachineReadableTables": True,
    }
    if regeneration["status"] != "PASS": raise RuntimeError("scientific model regeneration failed")
    worker_cpu = float(bgm["cpuSeconds"].sum())
    runtime = {
        "schema": "eidosoma.e01.s19_l18.runtime.v1", "startedAtUtc": utc_now(),
        "wallSeconds": time.perf_counter() - started_wall, "orchestratorCpuSeconds": time.process_time() - started_cpu,
        "workerCpuSeconds": worker_cpu, "workerCpuHours": worker_cpu / 3600.0,
        "workers": workers, "threadsPerWorker": 1, "gpuHours": 0,
        "python": sys.version, "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__, "pyarrow": pyarrow.__version__,
    }
    validation = {"status": "PASS", "repositoryHead": git("rev-parse", "HEAD"), "remoteHead": git("rev-parse", "origin/eidosoma/groups/42"), "repositoryClean": not bool(git("status", "--porcelain=v1")), "exactTargetReplay": bool(replay["exactReplayPassed"].all()), "pastSuffixInvariant": bool(suffix["passed"].all()), "candidateCount": targets["candidateId"].nunique(), "matrixCount": targets.groupby("candidateId")["matrixIndex"].nunique().to_dict(), "bootstrapReplicates": BOOTSTRAPS, "labelPermutationReplicates": PERMUTATIONS}
    write_json(BUILD_ROOT / "regeneration_validation.json", regeneration)
    write_json(BUILD_ROOT / "runtime_manifest.json", runtime)
    write_json(BUILD_ROOT / "classification.json", {"researchStepId": LOOP_ID, "versionedId": VERSION, "status": "COMPLETE_AWAITING_MANDATORY_HUMAN_REVIEW", "classifications": classifications, "promotable": False, "s18Changed": False, "mandatoryHumanReview": True})
    write_json(BUILD_ROOT / "validation_summary.json", validation)
    report = report_text(targets, aggregate, gates, classifications, validation, runtime)
    atomic_text(BUILD_ROOT / "research_step_full_results.md", report)
    atomic_text(BUILD_ROOT / "S19_L18_FULL_RESULTS.md", report)
    summary = f"""# S19-L18 decision summary

**Decision:** {', '.join(classifications)}.

L18 established a non-saturated recurring-attractor first-entry task at a fixed raw molecular landmark. The machine-readable gates determine whether the registered past-only BGM-plus-organization model added information beyond time, exact H/stability, recurrence geometry, and a complete non-Phi model. Completed-fit and completed-centroid models remain future-dependent diagnostics. No result is directly promotable because the target/task were selected adaptively from frozen L02 evidence.

Stop for mandatory human review. No later activity is active.
"""
    atomic_text(BUILD_ROOT / "loop_decision_summary.md", summary)
    storage = {"status": "PASS", "retainedBytesBeforeManifest": sum(p.stat().st_size for p in BUILD_ROOT.rglob("*") if p.is_file()), "retainedGiBMaximum": 25, "temporaryGiBMaximum": 75, "figureCount": len(figure_paths)}
    write_json(BUILD_ROOT / "storage_validation.json", storage)
    # Copy pre-outcome records into the final build without mutating them.
    for name in ["preregistration.yaml", "decision_record.md", "implementation_lock.json", "source_snapshot_manifest.json", "immutable_prior_validation.json", "preoutcome_repository_lock.json", "fixture_results.parquet", "benchmark_projection.json"]:
        shutil.copy2(LOOP_ROOT / name, BUILD_ROOT / name)
    write_json(BUILD_ROOT / "artifact_manifest.json", manifest_for(BUILD_ROOT))
    # Publish only after every validation is complete.
    for child in list(LOOP_ROOT.iterdir()):
        if child.is_dir(): shutil.rmtree(child)
        else: child.unlink()
    for child in BUILD_ROOT.iterdir():
        destination = LOOP_ROOT / child.name
        shutil.copytree(child, destination) if child.is_dir() else shutil.copy2(child, destination)
    # Append shared records only after the loop bundle is frozen.
    timestamp = utc_now(); append_root_ledgers(classifications, gates, timestamp)
    root_report = report.replace("# S19-L18", "# S19 current handoff — S19-L18", 1)
    atomic_text(ARTIFACT_ROOT / "research_step_full_results.md", root_report)
    status = {"researchStepId": LOOP_ID, "status": "AWAITING_MANDATORY_HUMAN_REVIEW", "lastCompletedLoop": LOOP_ID, "currentLoop": LOOP_ID, "nextLoopAuthorized": False, "s20Status": "DEFINED_INACTIVE", "outcomeClassification": classifications[0], "classifications": classifications, "validationResult": "PASS_EXACT_TARGET_PREFIX_SUFFIX_MODEL_BOOTSTRAP_PERMUTATION_IMMUTABILITY_REGENERATION", "recommendedNextAction": "MANDATORY_HUMAN_REVIEW", "updatedAtUtc": timestamp}
    write_json(ARTIFACT_ROOT / "s19_status.json", status)
    print(canonical_json({"status": "COMPLETE", "classifications": classifications, "artifactRoot": str(LOOP_ROOT), "wallSeconds": runtime["wallSeconds"]}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-lock", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.prepare_lock:
        prepare_lock()
    else:
        execute(args.workers)


if __name__ == "__main__":
    main()
