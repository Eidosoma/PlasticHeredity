#!/usr/bin/env python3
"""Run S19-L45 PhiID incremental-value audit for hereditary episodes."""

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
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ[variable] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from e01_breakinggrn_transfer_audit.core import array_sha256
from e01_frozen_timebase_ensemble.core import (
    selected_clock_observations,
    states_from_observations,
)
from e01_onset_discovery.empirical_committor import restored_state_from_observation
from e01_onset_discovery.heredity_phi_incremental import (
    composition_controls,
    fit_binomial_ridge,
    metric_summary,
    predict_probability,
    probability_metrics,
)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


L44 = load_module(
    "e01_l44_runner",
    ROOT / "scripts/e01/run_s19_l44_plastic_heredity_process_family.py",
)
BASE = L44.BASE
ARTIFACT_ROOT = Path("/artifacts/research_steps/S19")
LOOP_ROOT = ARTIFACT_ROOT / "loops/L45"
L17_ROOT = ARTIFACT_ROOT / "loops/L17"
L18_ROOT = ARTIFACT_ROOT / "loops/L18"
L23_ROOT = ARTIFACT_ROOT / "loops/L23"
L44_ROOT = ARTIFACT_ROOT / "loops/L44"
BUILD_ROOT = Path("/cache/e01_s19_l45/build")
CACHE_ROOT = Path("/cache/e01_s19_l45")
CONFIG = ROOT / "configs/e01/s19_l45_phi_incremental_hereditary_episode.yaml"
RUNNER_PATH = Path(__file__).resolve()
CORE_PATH = ROOT / "src/e01_onset_discovery/heredity_phi_incremental.py"
WORKER_PATH = ROOT / "scripts/e01/l45_phi_process_worker.py"
EXACT_PYTHON = Path("/cache/e01_s19_l17/venv/bin/python")
SAFE_LATTICE = Path("/artifacts/research_steps/S12B/safe_phi_lattice.json")
LOOP_ID = "S19-L45"
VERSION = "E01-S19-L45-PHI-INCREMENTAL-VALUE-FOR-HEREDITARY-EPISODE-v1.0.0"
TARGET = "NEW_HEREDITARY_EPISODE_RUN3"
MODES = (
    "PAST_ONLY_PREFIX_FIT",
    "RETROSPECTIVE_COMPLETED_TRAJECTORY_FIT",
)
METRICS = (
    "emergence_nan0",
    "integrated_raw",
    "synergy_raw",
    "downward_causation_raw",
)
MODEL_IDS = (
    "TRAINING_PRIOR",
    "INHERITANCE_BASELINE",
    "DIRECT_CONTROLS",
    "PAST_PHI_ONLY",
    "DIRECT_PLUS_PAST_PHI",
    "COMPLETED_PHI_ONLY",
    "DIRECT_PLUS_COMPLETED_PHI",
)
EVALUATION_COHORTS = ("L28_VALIDATION", "L31_CONFIRMATION")
BOOTSTRAPS = 4096
PERMUTATIONS = 512
WORKERS = 8
SEED_ROOT = bytes.fromhex(
    "442f012fb97ae59c5d6822748218b2715fb89b5fa4174f07c037244076736cb1"
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_hash(frame: pd.DataFrame) -> str:
    ordered = frame.reindex(sorted(frame.columns), axis=1).reset_index(drop=True)
    return hashlib.sha256(
        ordered.to_json(orient="table", index=False, double_precision=15).encode()
    ).hexdigest()


def seed_material(*parts: object) -> bytes:
    return hashlib.sha256(
        SEED_ROOT + b"\x00" + json.dumps(parts, separators=(",", ":")).encode()
    ).digest()


def derived_seed(*parts: object, bits: int = 32) -> int:
    return int.from_bytes(seed_material(*parts)[: bits // 8], "big")


def validate_immutable_prior() -> dict[str, Any]:
    prior = L44.validate_immutable_prior()
    manifest = json.loads((L44_ROOT / "artifact_manifest.json").read_text())
    rows = []
    for row in manifest["files"]:
        path = L44_ROOT / row["path"]
        actual = sha256_file(path) if path.exists() else None
        rows.append(
            {
                "path": str(path),
                "expectedSha256": row["sha256"],
                "actualSha256": actual,
                "unchanged": actual == row["sha256"],
            }
        )
    passed = bool(prior["unchanged"] and rows and all(row["unchanged"] for row in rows))
    return {
        "schema": "eidosoma.e01.s19_l45.immutable_prior_validation.v1",
        "status": "PASS" if passed else "FAIL",
        "unchanged": passed,
        "priorThroughL43Unchanged": bool(prior["unchanged"]),
        "validatedL44ArtifactCount": len(rows),
        "aggregateSha256": hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "rows": rows,
    }


def source_registry() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sourceId": "L45_L17_BGM_PIPELINE",
                "evidenceClass": "DIRECT_PUBLIC_CODE_LINEAGE",
                "finding": "BreakingGRNMemories defines a complete PhiID path whose emergence is synergy plus downward causation and also exposes integrated Phi-r.",
                "frozenUse": "exact L17/L18-validated source pipeline",
                "url": "https://github.com/pigozzif/BreakingGRNMemories",
            },
            {
                "sourceId": "L45_PHIID_TAXONOMY",
                "evidenceClass": "PRIMARY_METHOD_PAPER",
                "finding": "PhiID separates heterogeneous information-dynamic atoms rather than treating integrated information as one mechanism.",
                "frozenUse": "retain emergence, integrated, synergy and downward-causation identities separately",
                "url": "https://arxiv.org/abs/1909.02297",
            },
            {
                "sourceId": "L45_CAUSAL_EMERGENCE_MULTIVARIATE",
                "evidenceClass": "PRIMARY_METHOD_PAPER",
                "finding": "PhiID-based causal-emergence quantities are multivariate information decompositions, not causal intervention proof.",
                "frozenUse": "interpretation boundary and component identities",
                "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC7833221/",
            },
            {
                "sourceId": "L45_L44_PROCESS_TARGET",
                "evidenceClass": "DIRECT_FROZEN_E01_RESULT",
                "finding": "The run-3 process was selected prospectively by the L44 hierarchy after passing reliability and count-matched order gates in both candidates and cohorts.",
                "frozenUse": "sole L45 committor target",
                "url": None,
            },
        ]
    )


def build_task_registry() -> pd.DataFrame:
    target = pd.read_parquet(L44_ROOT / "state_process_results.parquet")
    target = target[target["processId"].eq(TARGET)].copy()
    prefix = pd.read_parquet(L44_ROOT / "prefix_control_registry.parquet")
    summary = pd.read_parquet(L44.L42_ROOT / "prefix_state_summary.parquet")
    response = pd.read_parquet(L44.L42_ROOT / "response_registry.parquet")
    manifest = pd.read_parquet(L23_ROOT / "input_trajectory_manifest.parquet")
    manifest = manifest[
        [
            "candidateId",
            "matrixIndex",
            "trajectoryId",
            "trajectorySha256",
            "betaSha256",
            "initialStateSha256",
            "selectedClockLength",
            "clockId",
            "cachePath",
            "cacheSha256",
        ]
    ]
    task = target.merge(
        prefix,
        on=["stateId", "evaluationCohort", "candidateId", "matrixIndex", "landmark"],
        validate="one_to_one",
    )
    task = task.merge(
        summary[["stateId", "latestParentDaughterH"]],
        on="stateId",
        validate="one_to_one",
    )
    task = task.merge(
        response[["stateId", "currentStateSha256"]],
        on="stateId",
        validate="one_to_one",
    )
    task = task.merge(
        manifest, on=["candidateId", "matrixIndex"], validate="one_to_one"
    )
    task = task.sort_values(
        ["evaluationCohort", "candidateId", "matrixIndex"]
    ).reset_index(drop=True)
    if (
        len(task) != 280
        or task.duplicated(["candidateId", "matrixIndex"]).any()
        or not task["clockId"].eq("C1_SELECTED_DAUGHTER_RETAINED").all()
        or (task["selectedClockLength"] < task["landmark"]).any()
    ):
        raise RuntimeError("L45 task registry scope failure")
    return task


def seed_manifest(task: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in task.itertuples(index=False):
        for mode in MODES:
            for purpose in ("preprocess", "partition"):
                parts = (purpose, mode, row.candidateId, int(row.matrixIndex))
                material = seed_material(*parts)
                rows.append(
                    {
                        "purpose": purpose,
                        "temporalMode": mode,
                        "candidateId": row.candidateId,
                        "matrixIndex": int(row.matrixIndex),
                        "derivedSeed": str(int.from_bytes(material[:4], "big")),
                        "seedMaterialSha256": material.hex(),
                        "partsJson": json.dumps(parts),
                    }
                )
    for purpose, count in (
        ("bootstrap", BOOTSTRAPS),
        ("feature_permutation", PERMUTATIONS),
        ("target_permutation", PERMUTATIONS),
    ):
        for candidate in ("S12F-CANDIDATE-02", "S12F-CANDIDATE-03"):
            for cohort in EVALUATION_COHORTS:
                parts = (purpose, candidate, cohort, count)
                material = seed_material(*parts)
                rows.append(
                    {
                        "purpose": purpose,
                        "temporalMode": None,
                        "candidateId": candidate,
                        "matrixIndex": None,
                        "derivedSeed": str(int.from_bytes(material[:16], "big")),
                        "seedMaterialSha256": material.hex(),
                        "partsJson": json.dumps(parts),
                    }
                )
    frame = pd.DataFrame(rows)
    if frame["seedMaterialSha256"].duplicated().any():
        raise RuntimeError("L45 seed-material collision")
    return frame


def seed_firewall(seeds: pd.DataFrame) -> dict[str, Any]:
    prior_material: set[str] = set()
    for path in ARTIFACT_ROOT.glob("loops/L*/**/*seed*manifest*.parquet"):
        if "/L45/" in str(path):
            continue
        try:
            frame = pd.read_parquet(path)
        except (OSError, ValueError, TypeError):
            continue
        for column in frame.columns:
            if "seedmaterialsha256" in column.lower():
                prior_material.update(frame[column].dropna().astype(str))
    overlap = sorted(set(seeds["seedMaterialSha256"]) & prior_material)
    return {
        "schema": "eidosoma.e01.s19_l45.seed_firewall.v1",
        "status": "PASS" if not overlap else "FAIL",
        "analysisSeedCount": len(seeds),
        "newScientificSimulationStreams": 0,
        "seedMaterialOverlapCount": len(overlap),
        "overlap": overlap,
    }


def fixture_results() -> pd.DataFrame:
    summary = metric_summary(np.arange(10, dtype=float))
    controls = composition_controls(np.array([[1, 1], [2, 2]], dtype=np.int64))
    x = np.arange(12, dtype=float)[:, None]
    k = np.array([1, 1, 1, 1, 2, 3, 5, 7, 8, 9, 9, 9])
    n = np.full(12, 10)
    first = fit_binomial_ridge(x, k, n, seed=42)
    second = fit_binomial_ridge(x, k, n, seed=42)
    p1 = predict_probability(first, x)
    p2 = predict_probability(second, x)
    l17 = pd.read_parquet(L17_ROOT / "fixture_results.parquet")
    l18 = pd.read_parquet(L18_ROOT / "fixture_results.parquet")
    rows = [
        ("METRIC_CURRENT", summary.current == 9),
        ("METRIC_RECENT_SLOPE", abs(summary.recent_slope - 1) <= 1e-12),
        (
            "COMPOSITION_SCALING_CLOSURE",
            abs(controls["currentAdjacentMolecularH"] - 1) <= 1e-12,
        ),
        ("RIDGE_MODEL_EXACT_REPLAY", np.array_equal(p1, p2)),
        ("RIDGE_MODEL_DIRECTION", p1[-1] > p1[0]),
        ("L17_SOURCE_FIXTURES", bool(l17["fixturePassed"].all())),
        ("L18_PREFIX_FIXTURES", bool(l18["passed"].all())),
        ("MODE_SEPARATION", MODES[0] != MODES[1]),
    ]
    return pd.DataFrame(
        [{"fixtureId": name, "passed": bool(passed)} for name, passed in rows]
    )


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def run_exact_bgm(
    states: np.ndarray,
    *,
    preprocessing_seed: int,
    partition_seed: int,
    output_root: Path,
    identity: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    token = hashlib.sha256(identity.encode()).hexdigest()[:24]
    input_path = output_root / f"{token}.input.npz"
    output_path = output_root / f"{token}.output.npz"
    _atomic_npz(input_path, states=np.asarray(states, dtype=np.int64))
    env = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        env[name] = "1"
    completed = subprocess.run(
        [
            str(EXACT_PYTHON),
            str(WORKER_PATH),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--safe-lattice",
            str(SAFE_LATTICE),
            "--preprocessing-seed",
            str(preprocessing_seed),
            "--partition-seed",
            str(partition_seed),
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"L45 exact Phi worker failed: {completed.stderr[-2000:]}")
    with np.load(output_path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata_json"]))
        arrays = {
            name: np.asarray(payload[name])
            for name in payload.files
            if name != "metadata_json"
        }
    return metadata, arrays


def _load_trajectory(row: Any) -> tuple[Any, tuple[Any, ...], np.ndarray]:
    path = Path(row.cachePath)
    if not path.is_file() or sha256_file(path) != row.cacheSha256:
        raise RuntimeError(f"L45 trajectory cache identity failure: {path}")
    with path.open("rb") as handle:
        trajectory = pickle.load(handle)
    if (
        trajectory.trajectory_id != row.trajectoryId
        or trajectory.trajectory_sha256 != row.trajectorySha256
        or trajectory.beta_sha256 != row.betaSha256
        or trajectory.initial_state_sha256 != row.initialStateSha256
        or trajectory.configuration_id != row.candidateId
        or int(trajectory.matrix_index) != int(row.matrixIndex)
    ):
        raise RuntimeError("L45 trajectory payload identity mismatch")
    selected = selected_clock_observations(trajectory, row.clockId)
    states = states_from_observations(selected)
    return trajectory, selected, states


def _mode_seeds(row: Any, mode: str) -> tuple[int, int]:
    return (
        derived_seed("preprocess", mode, row.candidateId, int(row.matrixIndex)),
        derived_seed("partition", mode, row.candidateId, int(row.matrixIndex)),
    )


def _metric_feature_row(
    arrays: dict[str, np.ndarray], *, mode: str, landmark: int
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    expected = landmark - 2
    for metric in METRICS:
        values = arrays.get(metric)
        if values is None:
            selected = np.array([], dtype=np.float64)
        elif mode == "PAST_ONLY_PREFIX_FIT":
            selected = np.asarray(values, dtype=np.float64)
        else:
            selected = np.asarray(values, dtype=np.float64)[:expected]
        summary = metric_summary(selected, recent=8)
        output[f"{metric}__current"] = summary.current
        output[f"{metric}__recentSlope"] = summary.recent_slope
        output[f"{metric}__finiteFraction"] = summary.finite_fraction
        output[f"{metric}__observations"] = summary.observations
        output[f"{metric}__selectedSha256"] = array_sha256(selected)
        if values is not None and summary.observations != expected:
            raise RuntimeError(
                f"L45 local value alignment mismatch {mode} {metric}: "
                f"{summary.observations} != {expected}"
            )
    return output


def state_feature_worker(row_dict: dict[str, Any], pass_root: Path) -> dict[str, Any]:
    row = type("Task", (), row_dict)()
    started = time.perf_counter()
    _, selected, states = _load_trajectory(row)
    landmark = int(row.landmark)
    current = selected[landmark - 1]
    restored = restored_state_from_observation(current)
    restored_hash = array_sha256(np.asarray(restored.state, dtype=np.int64))
    if restored_hash != row.currentStateSha256:
        raise RuntimeError("L45 current restored-state identity mismatch")
    direct = {
        "stateId": row.stateId,
        "evaluationCohort": row.evaluationCohort,
        "candidateId": row.candidateId,
        "matrixIndex": int(row.matrixIndex),
        "landmark": landmark,
        **composition_controls(states[:landmark]),
    }
    phi_rows: list[dict[str, Any]] = []
    for mode in MODES:
        input_states = states[:landmark] if mode == MODES[0] else states
        pre_seed, part_seed = _mode_seeds(row, mode)
        metadata, arrays = run_exact_bgm(
            input_states,
            preprocessing_seed=pre_seed,
            partition_seed=part_seed,
            output_root=pass_root,
            identity=f"{row.stateId}|{mode}",
        )
        allowed = metadata["status"] in {
            "ELIGIBLE",
            "ELIGIBLE_PARTIAL_NONFINITE_LOCAL_VALUES",
        }
        feature = _metric_feature_row(arrays, mode=mode, landmark=landmark)
        phi_rows.append(
            {
                "stateId": row.stateId,
                "evaluationCohort": row.evaluationCohort,
                "candidateId": row.candidateId,
                "matrixIndex": int(row.matrixIndex),
                "landmark": landmark,
                "temporalMode": mode,
                "status": metadata["status"],
                "reason": metadata.get("reason"),
                "defined": bool(allowed and all(name in arrays for name in METRICS)),
                "partition1Size": metadata.get("partition1Size"),
                "partition2Size": metadata.get("partition2Size"),
                "inputObservations": len(input_states),
                "preprocessingSeed": pre_seed,
                "partitionSeed": part_seed,
                "completedTrajectoryFutureDependent": mode == MODES[1],
                "targetUsesCompletedTestTrajectory": False,
                **feature,
            }
        )
    return {
        "stateId": row.stateId,
        "direct": direct,
        "phi": phi_rows,
        "identity": {
            "stateId": row.stateId,
            "candidateId": row.candidateId,
            "matrixIndex": int(row.matrixIndex),
            "trajectoryId": row.trajectoryId,
            "trajectorySha256": row.trajectorySha256,
            "cacheSha256": row.cacheSha256,
            "currentStateSha256": restored_hash,
            "selectedClockLength": len(states),
            "landmark": landmark,
            "identityPassed": True,
        },
        "wallSeconds": time.perf_counter() - started,
    }


def execute_feature_pass(
    task: pd.DataFrame, pass_root: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if pass_root.exists():
        shutil.rmtree(pass_root)
    pass_root.mkdir(parents=True)
    results: list[dict[str, Any]] = []
    records = task.to_dict(orient="records")
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = [
            executor.submit(state_feature_worker, row, pass_root) for row in records
        ]
        for future in as_completed(futures):
            results.append(future.result())
    phi = (
        pd.DataFrame([row for result in results for row in result["phi"]])
        .sort_values(["evaluationCohort", "candidateId", "matrixIndex", "temporalMode"])
        .reset_index(drop=True)
    )
    direct = (
        pd.DataFrame([result["direct"] for result in results])
        .sort_values(["evaluationCohort", "candidateId", "matrixIndex"])
        .reset_index(drop=True)
    )
    identity = (
        pd.DataFrame([result["identity"] for result in results])
        .sort_values(["candidateId", "matrixIndex"])
        .reset_index(drop=True)
    )
    runtime = (
        pd.DataFrame(
            [
                {"stateId": result["stateId"], "wallSeconds": result["wallSeconds"]}
                for result in results
            ]
        )
        .sort_values("stateId")
        .reset_index(drop=True)
    )
    if len(phi) != 560 or len(direct) != 280 or len(identity) != 280:
        raise RuntimeError("L45 feature-pass cardinality failure")
    return phi, direct, identity, runtime


INHERITANCE_COLUMNS = [
    "latestParentDaughterH",
    "prefixInheritanceFraction",
    "prefixTrailingInheritanceRun",
    "prefixBoundaryCount",
]
DIRECT_COLUMNS = [
    *INHERITANCE_COLUMNS,
    "currentMass",
    "currentGenerationLocalStep",
    "landmark",
    "currentAdjacentMolecularH",
    "currentCompositionChange",
]


def feature_table(
    task: pd.DataFrame, phi: pd.DataFrame, direct: pd.DataFrame
) -> tuple[pd.DataFrame, list[str], list[str]]:
    summary_columns = [
        f"{metric}__{summary}"
        for metric in METRICS
        for summary in ("current", "recentSlope")
    ]
    wide_parts = []
    mode_prefix = {MODES[0]: "past", MODES[1]: "completed"}
    for mode, prefix in mode_prefix.items():
        subset = phi[phi["temporalMode"].eq(mode)][
            ["stateId", "defined", *summary_columns]
        ].copy()
        subset = subset.rename(
            columns={
                "defined": f"{prefix}PhiDefined",
                **{column: f"{prefix}__{column}" for column in summary_columns},
            }
        )
        wide_parts.append(subset)
    wide = wide_parts[0].merge(wide_parts[1], on="stateId", validate="one_to_one")
    base_columns = [
        "stateId",
        "evaluationCohort",
        "candidateId",
        "matrixIndex",
        "landmark",
        "successes",
        "trials",
        "qHat",
        "qHatHalfA",
        "qHatHalfB",
        "eligible",
        "latestParentDaughterH",
        "prefixInheritanceFraction",
        "prefixTrailingInheritanceRun",
        "prefixMaximumInheritanceRun",
        "prefixBoundaryCount",
        "currentMass",
        "currentGenerationLocalStep",
        "currentCompletedFissions",
    ]
    result = (
        task[base_columns]
        .merge(
            direct,
            on=[
                "stateId",
                "evaluationCohort",
                "candidateId",
                "matrixIndex",
                "landmark",
            ],
            validate="one_to_one",
        )
        .merge(wide, on="stateId", validate="one_to_one")
        .sort_values(["evaluationCohort", "candidateId", "matrixIndex"])
        .reset_index(drop=True)
    )
    past = [f"past__{column}" for column in summary_columns]
    completed = [f"completed__{column}" for column in summary_columns]
    return result, past, completed


def model_columns(
    model_id: str, past_columns: list[str], completed_columns: list[str]
) -> list[str]:
    mapping = {
        "TRAINING_PRIOR": [],
        "INHERITANCE_BASELINE": INHERITANCE_COLUMNS,
        "DIRECT_CONTROLS": DIRECT_COLUMNS,
        "PAST_PHI_ONLY": past_columns,
        "DIRECT_PLUS_PAST_PHI": [*DIRECT_COLUMNS, *past_columns],
        "COMPLETED_PHI_ONLY": completed_columns,
        "DIRECT_PLUS_COMPLETED_PHI": [*DIRECT_COLUMNS, *completed_columns],
    }
    return list(mapping[model_id])


def fit_models_and_predict(
    features: pd.DataFrame,
    past_columns: list[str],
    completed_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_rows: list[dict[str, Any]] = []
    registry_rows: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []
    eligible = features[features["eligible"]].copy()
    for candidate, candidate_frame in eligible.groupby("candidateId", sort=True):
        train = candidate_frame[
            candidate_frame["evaluationCohort"].eq("L28_DEVELOPMENT")
        ].sort_values("matrixIndex")
        if len(train) < 32:
            raise RuntimeError("L45 insufficient development states")
        for model_id in MODEL_IDS:
            columns = model_columns(model_id, past_columns, completed_columns)
            seed = derived_seed("model", candidate, model_id)
            if model_id == "TRAINING_PRIOR":
                prior = float(train["successes"].sum() / train["trials"].sum())
                model = None
                replay_exact = True
            else:
                model = fit_binomial_ridge(
                    train[columns].to_numpy(float),
                    train["successes"].to_numpy(int),
                    train["trials"].to_numpy(int),
                    seed=seed,
                    c=1.0,
                )
                replay = fit_binomial_ridge(
                    train[columns].to_numpy(float),
                    train["successes"].to_numpy(int),
                    train["trials"].to_numpy(int),
                    seed=seed,
                    c=1.0,
                )
                first = predict_probability(
                    model, candidate_frame[columns].to_numpy(float)
                )
                second = predict_probability(
                    replay, candidate_frame[columns].to_numpy(float)
                )
                replay_exact = bool(np.array_equal(first, second))
                prior = float("nan")
            registry_rows.append(
                {
                    "candidateId": candidate,
                    "modelId": model_id,
                    "featureCount": len(columns),
                    "featuresJson": json.dumps(columns),
                    "trainingStates": len(train),
                    "seed": seed,
                    "ridgeC": 1.0,
                    "fitCohort": "L28_DEVELOPMENT",
                    "completedFitFutureDependent": "COMPLETED" in model_id,
                    "exactReplayPassed": replay_exact,
                }
            )
            for cohort, group in candidate_frame.groupby(
                "evaluationCohort", sort=False
            ):
                ordered = group.sort_values("matrixIndex")
                probability = (
                    np.full(len(ordered), prior, dtype=np.float64)
                    if model is None
                    else predict_probability(model, ordered[columns].to_numpy(float))
                )
                for row, value in zip(
                    ordered.itertuples(index=False), probability, strict=True
                ):
                    prediction_rows.append(
                        {
                            "stateId": row.stateId,
                            "evaluationCohort": cohort,
                            "candidateId": candidate,
                            "matrixIndex": int(row.matrixIndex),
                            "modelId": model_id,
                            "probability": float(value),
                            "qHat": float(row.qHat),
                            "successes": int(row.successes),
                            "trials": int(row.trials),
                            "completedFitFutureDependent": "COMPLETED" in model_id,
                        }
                    )
            replay_rows.append(
                {
                    "candidateId": candidate,
                    "modelId": model_id,
                    "exactReplayPassed": replay_exact,
                }
            )
    predictions = (
        pd.DataFrame(prediction_rows)
        .sort_values(["evaluationCohort", "candidateId", "modelId", "matrixIndex"])
        .reset_index(drop=True)
    )
    registry = pd.DataFrame(registry_rows)
    replay = pd.DataFrame(replay_rows)
    if len(predictions) != int(eligible.shape[0] * len(MODEL_IDS)):
        raise RuntimeError("L45 prediction cardinality mismatch")
    if not replay["exactReplayPassed"].all():
        raise RuntimeError("L45 model exact replay failure")
    return predictions, registry, replay


def metric_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in predictions.groupby(
        ["evaluationCohort", "candidateId", "modelId"], sort=False
    ):
        rows.append(
            {
                "evaluationCohort": keys[0],
                "candidateId": keys[1],
                "modelId": keys[2],
                "states": len(group),
                **probability_metrics(
                    group["qHat"].to_numpy(float),
                    group["probability"].to_numpy(float),
                    group["successes"].to_numpy(int),
                    group["trials"].to_numpy(int),
                ),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_model_results(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    bootstrap_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    comparisons = {
        "PAST_INCREMENTAL_OVER_DIRECT": (
            "DIRECT_CONTROLS",
            "DIRECT_PLUS_PAST_PHI",
        ),
        "COMPLETED_INCREMENTAL_OVER_DIRECT": (
            "DIRECT_CONTROLS",
            "DIRECT_PLUS_COMPLETED_PHI",
        ),
        "PAST_PHI_OVER_PRIOR": ("TRAINING_PRIOR", "PAST_PHI_ONLY"),
    }
    evaluation = predictions[predictions["evaluationCohort"].isin(EVALUATION_COHORTS)]
    for keys, raw in evaluation.groupby(
        ["evaluationCohort", "candidateId"], sort=False
    ):
        matrices = sorted(raw["matrixIndex"].unique())
        by_model = {
            model: group.set_index("matrixIndex").loc[matrices]
            for model, group in raw.groupby("modelId", sort=False)
        }
        rng = np.random.default_rng(
            derived_seed("bootstrap", *keys, BOOTSTRAPS, bits=128)
        )
        for replicate in range(BOOTSTRAPS):
            chosen = rng.integers(0, len(matrices), size=len(matrices))
            for model_id, frame in by_model.items():
                sample = frame.iloc[chosen]
                metrics = probability_metrics(
                    sample["qHat"].to_numpy(float),
                    sample["probability"].to_numpy(float),
                    sample["successes"].to_numpy(int),
                    sample["trials"].to_numpy(int),
                )
                bootstrap_rows.append(
                    {
                        "evaluationCohort": keys[0],
                        "candidateId": keys[1],
                        "modelId": model_id,
                        "replicate": replicate,
                        **metrics,
                    }
                )
    boot = pd.DataFrame(bootstrap_rows)
    for keys, group in boot.groupby(["evaluationCohort", "candidateId"], sort=False):
        for comparison_id, (baseline, augmented) in comparisons.items():
            left = group[group["modelId"].eq(baseline)].sort_values("replicate")
            right = group[group["modelId"].eq(augmented)].sort_values("replicate")
            brier_gain = left["qBrier"].to_numpy() - right["qBrier"].to_numpy()
            log_gain = (
                left["matrixBinomialLogLoss"].to_numpy()
                - right["matrixBinomialLogLoss"].to_numpy()
            )
            rank_gain = right["spearman"].to_numpy() - left["spearman"].to_numpy()
            point = metric_table(
                predictions[
                    predictions["evaluationCohort"].eq(keys[0])
                    & predictions["candidateId"].eq(keys[1])
                    & predictions["modelId"].isin([baseline, augmented])
                ]
            ).set_index("modelId")
            comparison_rows.append(
                {
                    "evaluationCohort": keys[0],
                    "candidateId": keys[1],
                    "comparisonId": comparison_id,
                    "baselineModelId": baseline,
                    "augmentedModelId": augmented,
                    "qBrierGain": float(
                        point.loc[baseline, "qBrier"] - point.loc[augmented, "qBrier"]
                    ),
                    "qBrierGainLower95": float(np.quantile(brier_gain, 0.025)),
                    "qBrierGainUpper95": float(np.quantile(brier_gain, 0.975)),
                    "logLossGain": float(
                        point.loc[baseline, "matrixBinomialLogLoss"]
                        - point.loc[augmented, "matrixBinomialLogLoss"]
                    ),
                    "logLossGainLower95": float(np.quantile(log_gain, 0.025)),
                    "logLossGainUpper95": float(np.quantile(log_gain, 0.975)),
                    "spearmanGain": float(
                        point.loc[augmented, "spearman"]
                        - point.loc[baseline, "spearman"]
                    ),
                    "spearmanGainLower95": float(np.nanquantile(rank_gain, 0.025)),
                    "spearmanGainUpper95": float(np.nanquantile(rank_gain, 0.975)),
                }
            )
    return boot, pd.DataFrame(comparison_rows)


def _fit_augmented_prediction(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    columns: list[str],
    *,
    seed: int,
    train_successes: np.ndarray | None = None,
) -> np.ndarray:
    successes = (
        train["successes"].to_numpy(int)
        if train_successes is None
        else np.asarray(train_successes, dtype=int)
    )
    model = fit_binomial_ridge(
        train[columns].to_numpy(float),
        successes,
        train["trials"].to_numpy(int),
        seed=seed,
        c=1.0,
    )
    return predict_probability(model, evaluation[columns].to_numpy(float))


def permutation_controls(
    features: pd.DataFrame,
    predictions: pd.DataFrame,
    past_columns: list[str],
    comparisons: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible = features[features["eligible"]].copy()
    direct_prediction = predictions[
        predictions["modelId"].eq("DIRECT_CONTROLS")
    ].set_index("stateId")["probability"]
    rows: list[dict[str, Any]] = []
    columns = [*DIRECT_COLUMNS, *past_columns]
    for candidate, candidate_frame in eligible.groupby("candidateId", sort=True):
        train = (
            candidate_frame[candidate_frame["evaluationCohort"].eq("L28_DEVELOPMENT")]
            .sort_values("matrixIndex")
            .reset_index(drop=True)
        )
        evaluation_groups = {
            cohort: candidate_frame[candidate_frame["evaluationCohort"].eq(cohort)]
            .sort_values("matrixIndex")
            .reset_index(drop=True)
            for cohort in EVALUATION_COHORTS
        }
        for replicate in range(PERMUTATIONS):
            feature_train = train.copy()
            train_rng = np.random.default_rng(
                derived_seed(
                    "feature_permutation", candidate, "TRAIN", replicate, bits=128
                )
            )
            feature_train[past_columns] = (
                train[past_columns].iloc[train_rng.permutation(len(train))].to_numpy()
            )
            target_rng = np.random.default_rng(
                derived_seed("target_permutation", candidate, replicate, bits=128)
            )
            target_order = target_rng.permutation(len(train))
            target_successes = train["successes"].to_numpy(int)[target_order]
            target_trials = train["trials"].to_numpy(int)[target_order]
            # Preserve successes/trials as a pair; assign both to the permuted
            # training outcome rows for the target-permutation null.
            target_train = train.copy()
            target_train["trials"] = target_trials
            for cohort, evaluation in evaluation_groups.items():
                evaluation_permuted = evaluation.copy()
                eval_rng = np.random.default_rng(
                    derived_seed(
                        "feature_permutation",
                        candidate,
                        cohort,
                        replicate,
                        bits=128,
                    )
                )
                evaluation_permuted[past_columns] = (
                    evaluation[past_columns]
                    .iloc[eval_rng.permutation(len(evaluation))]
                    .to_numpy()
                )
                feature_probability = _fit_augmented_prediction(
                    feature_train,
                    evaluation_permuted,
                    columns,
                    seed=derived_seed(
                        "feature_permutation_model", candidate, replicate
                    ),
                )
                target_probability = _fit_augmented_prediction(
                    target_train,
                    evaluation,
                    columns,
                    seed=derived_seed("target_permutation_model", candidate, replicate),
                    train_successes=target_successes,
                )
                baseline_probability = direct_prediction.loc[
                    evaluation["stateId"]
                ].to_numpy(float)
                baseline = probability_metrics(
                    evaluation["qHat"].to_numpy(float),
                    baseline_probability,
                    evaluation["successes"].to_numpy(int),
                    evaluation["trials"].to_numpy(int),
                )
                for control_id, probability in (
                    ("PAST_PHI_FEATURE_PERMUTATION", feature_probability),
                    ("DEVELOPMENT_TARGET_PERMUTATION", target_probability),
                ):
                    metric = probability_metrics(
                        evaluation["qHat"].to_numpy(float),
                        probability,
                        evaluation["successes"].to_numpy(int),
                        evaluation["trials"].to_numpy(int),
                    )
                    rows.append(
                        {
                            "evaluationCohort": cohort,
                            "candidateId": candidate,
                            "controlId": control_id,
                            "replicate": replicate,
                            "qBrierGainOverDirect": baseline["qBrier"]
                            - metric["qBrier"],
                            "logLossGainOverDirect": baseline["matrixBinomialLogLoss"]
                            - metric["matrixBinomialLogLoss"],
                        }
                    )
    nulls = pd.DataFrame(rows)
    observed = comparisons[
        comparisons["comparisonId"].eq("PAST_INCREMENTAL_OVER_DIRECT")
    ]
    summary_rows: list[dict[str, Any]] = []
    for row in observed.itertuples(index=False):
        for control_id, group in nulls[
            nulls["evaluationCohort"].eq(row.evaluationCohort)
            & nulls["candidateId"].eq(row.candidateId)
        ].groupby("controlId"):
            summary_rows.append(
                {
                    "evaluationCohort": row.evaluationCohort,
                    "candidateId": row.candidateId,
                    "controlId": control_id,
                    "observedQBrierGain": row.qBrierGain,
                    "observedLogLossGain": row.logLossGain,
                    "qBrierPermutationP": (
                        1 + int((group["qBrierGainOverDirect"] >= row.qBrierGain).sum())
                    )
                    / (len(group) + 1),
                    "logLossPermutationP": (
                        1
                        + int((group["logLossGainOverDirect"] >= row.logLossGain).sum())
                    )
                    / (len(group) + 1),
                }
            )
    summary = pd.DataFrame(summary_rows)
    for control_id in summary["controlId"].unique():
        for column in ("qBrierPermutationP", "logLossPermutationP"):
            index = summary[summary["controlId"].eq(control_id)].index
            p = summary.loc[index, column].to_numpy(float)
            order = np.argsort(p)
            adjusted = np.empty_like(p)
            running = 0.0
            for rank, position in enumerate(order):
                running = max(running, (len(p) - rank) * p[position])
                adjusted[position] = min(1.0, running)
            summary.loc[index, f"{column}Holm"] = adjusted
    return nulls, summary


def suffix_invariance_results(
    task: pd.DataFrame, original_phi: pd.DataFrame, cache_root: Path
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    sentinels = (
        task.sort_values(["evaluationCohort", "candidateId", "matrixIndex"])
        .groupby(["evaluationCohort", "candidateId"], sort=False)
        .head(4)
    )
    summary_columns = [
        f"{metric}__{summary}"
        for metric in METRICS
        for summary in ("current", "recentSlope")
    ]
    original_index = original_phi.set_index(["stateId", "temporalMode"])
    for row in sentinels.itertuples(index=False):
        _, _, states = _load_trajectory(row)
        landmark = int(row.landmark)
        rng = np.random.default_rng(
            derived_seed("suffix_perturbation", row.stateId, bits=128)
        )
        perturbed = states.copy()
        suffix = perturbed[landmark:].copy()
        if len(suffix) > 1:
            perturbed[landmark:] = suffix[rng.permutation(len(suffix))]
        for mode in MODES:
            input_states = perturbed[:landmark] if mode == MODES[0] else perturbed
            pre_seed, part_seed = _mode_seeds(row, mode)
            metadata, arrays = run_exact_bgm(
                input_states,
                preprocessing_seed=pre_seed,
                partition_seed=part_seed,
                output_root=cache_root,
                identity=f"suffix|{row.stateId}|{mode}",
            )
            if metadata["status"] not in {
                "ELIGIBLE",
                "ELIGIBLE_PARTIAL_NONFINITE_LOCAL_VALUES",
            }:
                raise RuntimeError("L45 suffix sentinel Phi pipeline ineligible")
            fresh = _metric_feature_row(arrays, mode=mode, landmark=landmark)
            original = original_index.loc[(row.stateId, mode)]
            differences = []
            for column in summary_columns:
                left = float(original[column])
                right = float(fresh[column])
                if np.isnan(left) and np.isnan(right):
                    differences.append(0.0)
                elif np.isfinite(left) and np.isfinite(right):
                    differences.append(abs(left - right))
                else:
                    differences.append(float("inf"))
            rows.append(
                {
                    "stateId": row.stateId,
                    "evaluationCohort": row.evaluationCohort,
                    "candidateId": row.candidateId,
                    "matrixIndex": int(row.matrixIndex),
                    "temporalMode": mode,
                    "maximumFeatureDifference": max(differences),
                    "featureArrayExact": max(differences) == 0,
                    "suffixWasPermuted": len(suffix) > 1,
                    "eligible": True,
                }
            )
    return pd.DataFrame(rows)


def scientific_gates(
    comparisons: pd.DataFrame,
    permutation_summary: pd.DataFrame,
    suffix: pd.DataFrame,
    phi: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], str]:
    past = comparisons[
        comparisons["comparisonId"].eq("PAST_INCREMENTAL_OVER_DIRECT")
        & comparisons["evaluationCohort"].isin(EVALUATION_COHORTS)
    ]
    completed = comparisons[
        comparisons["comparisonId"].eq("COMPLETED_INCREMENTAL_OVER_DIRECT")
        & comparisons["evaluationCohort"].isin(EVALUATION_COHORTS)
    ]
    past_prefix_exact = bool(
        suffix[suffix["temporalMode"].eq(MODES[0])]["featureArrayExact"].all()
    )
    completed_future_dependent = bool(
        (
            suffix[suffix["temporalMode"].eq(MODES[1])]["maximumFeatureDifference"] > 0
        ).any()
    )
    availability = bool(
        len(phi) == 560
        and phi["defined"].all()
        and phi.groupby("temporalMode")["stateId"].nunique().eq(280).all()
    )
    incremental = bool(
        len(past) == 4
        and (past["qBrierGain"] > 0).all()
        and (past["qBrierGainLower95"] > 0).all()
        and (past["logLossGain"] > 0).all()
        and (past["logLossGainLower95"] > 0).all()
    )
    permutation_pass = bool(
        len(permutation_summary) == 8
        and (permutation_summary["qBrierPermutationPHolm"] < 0.05).all()
        and (permutation_summary["logLossPermutationPHolm"] < 0.05).all()
    )
    completed_alignment = bool(
        len(completed) == 4
        and (completed["qBrierGainLower95"] > 0).all()
        and (completed["logLossGainLower95"] > 0).all()
    )
    gate_rows = [
        {
            "gateId": "SOURCE_FEATURE_AVAILABILITY",
            "passed": availability,
            "prospectiveEligible": True,
        },
        {
            "gateId": "PAST_ONLY_SUFFIX_INVARIANCE",
            "passed": past_prefix_exact,
            "prospectiveEligible": True,
        },
        {
            "gateId": "PAST_PHI_INCREMENTAL_BRIER_AND_LOGLOSS_ALL_GROUPS",
            "passed": incremental,
            "prospectiveEligible": True,
        },
        {
            "gateId": "PAST_PHI_PERMUTATION_CONTROLS_ALL_GROUPS",
            "passed": permutation_pass,
            "prospectiveEligible": True,
        },
        {
            "gateId": "COMPLETED_FIT_FUTURE_DEPENDENCE_DEMONSTRATED",
            "passed": completed_future_dependent,
            "prospectiveEligible": False,
        },
        {
            "gateId": "COMPLETED_FIT_RETROSPECTIVE_INCREMENTAL_ALIGNMENT",
            "passed": completed_alignment,
            "prospectiveEligible": False,
        },
    ]
    prospective = (
        availability and past_prefix_exact and incremental and permutation_pass
    )
    classifications = [
        "PAST_ONLY_PHI_INCREMENTAL_PROCESS_LEAD"
        if prospective
        else "PAST_ONLY_PHI_NOT_INCREMENTAL_FOR_HEREDITARY_EPISODE"
    ]
    if completed_alignment:
        classifications.append("COMPLETED_FIT_PHI_RETROSPECTIVE_PROCESS_ALIGNMENT")
    elif not prospective:
        classifications.append("PHI_PROCESS_NON_SUPPORT")
    classifications.append("NOT_PROMOTABLE_AS_CONFIRMED")
    next_theme = (
        "L46_UNTOUCHED_PAST_ONLY_PHI_PROCESS_CONFIRMATION_DESIGN"
        if prospective
        else "L46_FUNCTIONAL_HEREDITARY_REGIME_TRANSITION_AUDIT"
    )
    return pd.DataFrame(gate_rows), classifications, next_theme


def make_figures(
    metrics: pd.DataFrame,
    comparisons: pd.DataFrame,
    permutation_summary: pd.DataFrame,
    suffix: pd.DataFrame,
    phi: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    root = BUILD_ROOT / "figures"
    root.mkdir(parents=True, exist_ok=True)
    evaluation = metrics[metrics["evaluationCohort"].isin(EVALUATION_COHORTS)]

    fig, ax = plt.subplots(figsize=(11, 5))
    chosen = evaluation[
        evaluation["modelId"].isin(
            [
                "TRAINING_PRIOR",
                "INHERITANCE_BASELINE",
                "DIRECT_CONTROLS",
                "DIRECT_PLUS_PAST_PHI",
                "DIRECT_PLUS_COMPLETED_PHI",
            ]
        )
    ]
    labels = [
        f"{r.evaluationCohort.split('_')[-1][:4]}-{r.candidateId[-2:]}\n{r.modelId.replace('DIRECT_PLUS_', '').replace('_PHI', '').lower()}"
        for r in chosen.itertuples(index=False)
    ]
    ax.bar(np.arange(len(chosen)), chosen["qBrier"])
    ax.set_xticks(np.arange(len(chosen)), labels, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Q-hat Brier (lower is better)")
    ax.set_title("Hereditary-episode committor prediction")
    fig.tight_layout()
    fig.savefig(root / "01_committor_brier_models.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for index, metric in enumerate(("qBrierGain", "logLossGain")):
        chosen = comparisons[
            comparisons["evaluationCohort"].isin(EVALUATION_COHORTS)
            & comparisons["comparisonId"].isin(
                [
                    "PAST_INCREMENTAL_OVER_DIRECT",
                    "COMPLETED_INCREMENTAL_OVER_DIRECT",
                ]
            )
        ]
        x = np.arange(len(chosen))
        axes[index].bar(x, chosen[metric])
        axes[index].vlines(
            x,
            chosen[f"{metric}Lower95"],
            chosen[f"{metric}Upper95"],
            color="black",
        )
        axes[index].axhline(0, color="black", lw=1)
        axes[index].set_xticks(
            x,
            [
                f"{r.evaluationCohort.split('_')[-1][:4]}-{r.candidateId[-2:]}\n{'past' if r.comparisonId.startswith('PAST') else 'completed'}"
                for r in chosen.itertuples(index=False)
            ],
            rotation=45,
            ha="right",
            fontsize=7,
        )
        axes[index].set_title(metric)
    fig.tight_layout()
    fig.savefig(root / "02_incremental_effects.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.scatter(
        permutation_summary["qBrierPermutationPHolm"],
        permutation_summary["logLossPermutationPHolm"],
    )
    ax.axvline(0.05, color="grey", ls="--")
    ax.axhline(0.05, color="grey", ls="--")
    ax.set_xlabel("Holm-adjusted feature/target-null Brier p")
    ax.set_ylabel("Holm-adjusted feature/target-null log-loss p")
    ax.set_title("Registered permutation controls")
    fig.tight_layout()
    fig.savefig(root / "03_permutation_controls.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    grouped = suffix.groupby("temporalMode")["maximumFeatureDifference"].max()
    ax.bar(grouped.index, grouped.values)
    ax.set_yscale("symlog", linthresh=1e-15)
    ax.set_ylabel("Maximum feature change after suffix permutation")
    ax.set_title("Past-only invariance versus completed-fit dependence")
    fig.tight_layout()
    fig.savefig(root / "04_suffix_invariance.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    columns = [f"{metric}__current" for metric in METRICS]
    values = phi[phi["temporalMode"].eq(MODES[0])][columns].corr().to_numpy()
    image = ax.imshow(values, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(METRICS)), METRICS, rotation=45, ha="right")
    ax.set_yticks(range(len(METRICS)), METRICS)
    ax.set_title("Past-only PhiID component correlations")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(root / "05_phiid_component_map.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    matrix = gates.set_index("gateId")[["passed"]].astype(int)
    image = ax.imshow(matrix.to_numpy(), vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks([0], ["passed"])
    ax.set_yticks(range(len(matrix)), matrix.index, fontsize=7)
    ax.set_title("L45 scientific gate matrix")
    fig.colorbar(image, ax=ax, ticks=[0, 1])
    fig.tight_layout()
    fig.savefig(root / "06_decision_matrix.png", dpi=160)
    plt.close(fig)


def manifest_for(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            rows.append(
                {
                    "path": str(path.relative_to(root)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return {
        "schema": "eidosoma.e01.s19_l45.artifact_manifest.v1",
        "loopId": LOOP_ID,
        "files": rows,
        "fileCount": len(rows),
        "aggregateSha256": hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def report_text(
    metrics: pd.DataFrame,
    comparisons: pd.DataFrame,
    permutation_summary: pd.DataFrame,
    suffix: pd.DataFrame,
    gates: pd.DataFrame,
    classifications: list[str],
    runtime: dict[str, Any],
    next_theme: str,
) -> str:
    evaluation_metrics = metrics[
        metrics["evaluationCohort"].isin(EVALUATION_COHORTS)
        & metrics["modelId"].isin(
            [
                "TRAINING_PRIOR",
                "INHERITANCE_BASELINE",
                "DIRECT_CONTROLS",
                "PAST_PHI_ONLY",
                "DIRECT_PLUS_PAST_PHI",
                "DIRECT_PLUS_COMPLETED_PHI",
            ]
        )
    ][
        [
            "evaluationCohort",
            "candidateId",
            "modelId",
            "states",
            "qBrier",
            "matrixBinomialLogLoss",
            "spearman",
            "calibrationIntercept",
            "calibrationSlope",
        ]
    ]
    evaluation_comparisons = comparisons[
        comparisons["evaluationCohort"].isin(EVALUATION_COHORTS)
    ]
    return f"""# S19-L45 — PhiID Incremental Value for a Hereditary Episode

## Chief/human handoff

- **Step:** `{VERSION}`
- **Status:** complete under the extended L19–L65 autonomous sequence.
- **Classifications:** {", ".join(f"`{item}`" for item in classifications)}
- **Validation:** immutable L44-and-earlier baseline; L17/L18 source-equivalence fixtures; exact 280-state/trajectory replay; two prospectively separated temporal modes; exact prefix suffix-invariance sentinels; source-defined component identities; train-only preprocessing; 4,096 catalytic-matrix bootstraps; 512 feature and 512 target permutations; exact feature/model/table regeneration; storage and artifact hashes.
- **Recommended next action:** `{next_theme}`.

## Frozen question

Does source-defined information dynamics add held-out probability information about the frozen `NEW_HEREDITARY_EPISODE_RUN3` empirical committor beyond direct inheritance frequency, current streak, fission opportunities, adjacent parent–daughter and molecular H, composition change, mass, phase and elapsed selected-clock time?

The target, candidates, 280 matrix/state identities, F12 horizon, three-inheritance certification, metric identities, two feature summaries, ridge-binomial model, training cohort, evaluation cohorts, metrics, bootstraps and null controls were frozen before Phi outcomes. `PAST_ONLY_PREFIX_FIT` uses observations available at the landmark only. `RETROSPECTIVE_COMPLETED_TRAJECTORY_FIT` is explicitly future-dependent and cannot support prospective language.

## Anchor results

### Held-out model metrics

{evaluation_metrics.to_markdown(index=False)}

### Incremental comparisons

{evaluation_comparisons.to_markdown(index=False)}

### Permutation controls

{permutation_summary.to_markdown(index=False)}

### Suffix invariance/dependence

{suffix.groupby("temporalMode").agg(sentinels=("stateId", "size"), maximumFeatureDifference=("maximumFeatureDifference", "max"), exact=("featureArrayExact", "all")).reset_index().to_markdown(index=False)}

### Scientific gates

{gates.to_markdown(index=False)}

## Interpretation boundary

PhiID emergence, integrated Phi-r, synergy and downward causation are retained as separate public-source identities. They are computational information-dynamic summaries, not intervention evidence or proof of physical downward causation. Incremental value is required over direct heredity and phase controls; a raw Phi correlation is not sufficient. Completed-fit alignment, if present, is a retrospective description because suffix information changes the fitted prefix values.

The L44 target is an operational three-fission heredity episode, not exact return to a privileged composition and not an author-code replicator label. This loop changes no S18, paper-replication, intervention or causal-control classification.

## Provenance and validation

- Repository lock: `{runtime["repositoryHead"]}`.
- Workers: `{runtime["workers"]}`; one numerical-library thread per worker; GPU hours `0`.
- New matrices/trajectories/branch streams: `0/0/0`.
- Frozen states and trajectories: `{runtime["states"]}`.
- Exact Phi pipeline evaluations across both full passes: `{runtime["phiPipelineEvaluations"]}`.
- Wall time: `{runtime["wallSeconds"]:.2f}` seconds.
- S01–S18, V1/V2 and S19-L01–L44 remain unchanged.

## Reproduction

```bash
PYTHONPATH=src pytest -q tests/e01/test_s19_l45.py
python -m ruff check src/e01_onset_discovery/heredity_phi_incremental.py scripts/e01/l45_phi_process_worker.py scripts/e01/run_s19_l45_phi_incremental_hereditary_episode.py tests/e01/test_s19_l45.py
python scripts/e01/run_s19_l45_phi_incremental_hereditary_episode.py --prepare-lock
python scripts/e01/run_s19_l45_phi_incremental_hereditary_episode.py
```
"""


def append_ledgers(classifications: list[str], timestamp: str, next_theme: str) -> None:
    ledger_path = ARTIFACT_ROOT / "self_improvement_ledger.parquet"
    ledger = pd.read_parquet(ledger_path)
    sequence = int(ledger["ledgerSequence"].max()) + 1
    additions = [
        {
            "appendOnly": True,
            "beliefBeforeLoop": "L44 identified a modest but reliable and count-matched order-enriched three-fission hereditary-episode committor, while direct inheritance variables remained strong baselines.",
            "failureOrAmbiguityTargeted": "Whether source-defined past-only PhiID features contain incremental process information beyond direct heredity, stability, opportunity, mass and phase.",
            "informationGainRationale": "The L44 hierarchy selected one target before Phi access; L45 freezes four public-source metric identities, two summaries, one model and two temporal modes.",
            "learned": "L45 Phi/process contract locked before outcomes.",
            "ledgerSequence": sequence,
            "loopId": LOOP_ID,
            "motivatingEvidence": "L44 selected run-3 process; L17/L18 validated public BGM/IIGR Phi pipeline; PhiID source literature.",
            "proposedNextTest": "Apply prefix-only and completed-fit PhiID features to the frozen L44 process committor.",
            "recordPhase": "PRE_LOOP_METHOD_LOCK",
            "remainingPlausibleHypotheses": "Past-only Phi incremental signal, retrospective-only alignment, direct heredity proxy, or Phi non-support.",
            "selectedHypotheses": "PhiID information dynamics may track the transition into an ordered hereditary episode beyond direct process controls.",
            "timestampUtc": timestamp,
            "weakenedHypotheses": "A raw Phi association alone is evidence of an independent precursor.",
        },
        {
            "appendOnly": True,
            "beliefBeforeLoop": "A prospective information lead must improve held-out probability scoring over all direct controls in both candidates and both independent evaluation cohorts.",
            "failureOrAmbiguityTargeted": "Past-only incremental value versus retrospective completed-fit resemblance.",
            "informationGainRationale": "Suffix invariance and train-only preprocessing prevent completed future information from entering the past-only gate.",
            "learned": ";".join(classifications),
            "ledgerSequence": sequence + 1,
            "loopId": LOOP_ID,
            "motivatingEvidence": "Complete L45 result.",
            "proposedNextTest": next_theme,
            "recordPhase": "POST_LOOP_RESULT_AUTONOMOUS_CONTINUATION",
            "remainingPlausibleHypotheses": next_theme,
            "selectedHypotheses": "Source-defined PhiID versus hereditary-process committor.",
            "timestampUtc": timestamp,
            "weakenedHypotheses": "Completed-fit information can support an online organizational-warning claim.",
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
        + f"\n\n## {LOOP_ID} — PhiID incremental value for hereditary episodes\n\n"
        + f"- **Learned:** {', '.join(classifications)}.\n"
        + f"- **Next:** `{next_theme}`.\n",
    )

    candidates_path = ARTIFACT_ROOT / "candidate_registry.parquet"
    candidates = pd.read_parquet(candidates_path)
    selected = "PAST_ONLY_PHI_INCREMENTAL_PROCESS_LEAD" in classifications
    candidate = {
        "branchCount": 4,
        "bundleId": "L45_PHIID_HEREDITARY_EPISODE",
        "candidateId": "S19-L45-PAST-ONLY-PHIID-HEREDITARY-EPISODE",
        "candidateSpecificSuccess": 0,
        "completedFitLeakage": 0,
        "computeEfficiency": 4,
        "crossCandidateDiscriminability": 5,
        "deterministicHReuse": 0,
        "explanatoryLeverage": 5,
        "frozenRank": 1,
        "independenceFromPriorOutcomeSelection": 4,
        "outcomeGuidedThresholdSelection": 0,
        "paperFingerprintSpecificity": 2,
        "proposedSpecification": "source-defined past-only PhiID current and recent-slope features for L44 run-3 committor beyond direct hereditary controls",
        "rankingScore": 29.0,
        "registryOrder": int(candidates["registryOrder"].max()) + 1,
        "selected": selected,
        "selectionReason": "L44_PROSPECTIVE_HIERARCHY_AND_HUMAN_DIRECTED_L45_SIDE_QUEST",
        "sourceGrounding": 5,
        "testability": 5,
        "undefinedAuthorSemantics": 0,
    }
    BASE.write_parquet(
        candidates_path,
        pd.concat(
            [
                candidates,
                pd.DataFrame([candidate]).reindex(columns=candidates.columns),
            ],
            ignore_index=True,
        ),
    )

    source_path = ARTIFACT_ROOT / "source_search_ledger.parquet"
    sources = pd.read_parquet(source_path)
    additions_source = [
        {
            "commitOrVersion": None,
            "evidenceClass": row.evidenceClass,
            "finding": f"{row.finding}; L45 use: {row.frozenUse}",
            "licenseStatus": "PUBLIC_METADATA_OR_WORKSPACE_EVIDENCE",
            "redistributionStatus": "REFERENCE_ONLY",
            "repositoryIdentity": None,
            "retainedPath": None,
            "retrievalDate": timestamp[:10],
            "sha256": None,
            "sourceId": row.sourceId,
            "sourceType": row.evidenceClass,
            "treeIdentity": None,
            "url": row.url,
        }
        for row in source_registry().itertuples(index=False)
    ]
    BASE.write_parquet(
        source_path,
        pd.concat(
            [sources, pd.DataFrame(additions_source).reindex(columns=sources.columns)],
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
            "selectedDiscoveryLead": "PAST_ONLY_PHIID_HEREDITARY_EPISODE"
            if selected
            else None,
            "newMatrices": 0,
            "newTrajectories": 0,
            "newBranchStreams": 0,
            "nextStepActive": True,
        }
    )
    registry["proposedNextLoopTheme"] = next_theme
    registry["proposedNextLoopActive"] = True
    registry["authorizationUpperBound"] = "S19-L65"
    BASE.atomic_text(registry_path, yaml.safe_dump(registry, sort_keys=False))

    history_path = ARTIFACT_ROOT / "human_review_history.json"
    history = json.loads(history_path.read_text())
    history["history"].append(
        {
            "decision": "S19_L45_COMPLETE_AUTONOMOUS_CONTINUATION",
            "loopId": LOOP_ID,
            "nextLoopAuthorized": True,
            "recordedAtUtc": timestamp,
            "result": classifications,
            "s20Activated": False,
            "scope": VERSION,
            "source": "locked_execution_result",
        }
    )
    history["pendingDecision"] = "NONE_AUTONOMOUS_SEQUENCE_ACTIVE_THROUGH_L65"
    BASE.write_json(history_path, history)


def benchmark_projection() -> dict[str, Any]:
    root = CACHE_ROOT / "benchmark"
    if root.exists():
        shutil.rmtree(root)
    rng = np.random.default_rng(derived_seed("synthetic_benchmark"))
    states = rng.poisson(0.8, size=(64, 100)).astype(np.int64)
    states[:, 0] += 1
    start = time.perf_counter()
    metadata, arrays = run_exact_bgm(
        states,
        preprocessing_seed=derived_seed("benchmark_pre"),
        partition_seed=derived_seed("benchmark_partition"),
        output_root=root,
        identity="SYNTHETIC_BENCHMARK",
    )
    seconds = time.perf_counter() - start
    status = metadata["status"] in {
        "ELIGIBLE",
        "ELIGIBLE_PARTIAL_NONFINITE_LOCAL_VALUES",
    } and all(metric in arrays for metric in METRICS)
    projected_evaluations = 280 * 2 * 2 + 24 * 2
    projected_wall = seconds * projected_evaluations / WORKERS * 1.5
    projected_cpu = seconds * projected_evaluations / 3600 * 1.5
    return {
        "schema": "eidosoma.e01.s19_l45.benchmark_projection.v1",
        "syntheticOnly": True,
        "singlePipelineWallSeconds": seconds,
        "projectedPipelineEvaluations": projected_evaluations,
        "projectedWallHoursUpper": projected_wall / 3600,
        "projectedCpuHoursUpper": projected_cpu,
        "cpuHoursCeiling": 100,
        "wallHoursCeiling": 72,
        "workers": WORKERS,
        "status": "PASS"
        if status and projected_wall < 72 * 3600 * 0.85 and projected_cpu < 85
        else "FAIL",
    }


def prepare_lock() -> None:
    LOOP_ROOT.mkdir(parents=True, exist_ok=True)
    if git("status", "--porcelain=v1"):
        raise RuntimeError("repository must be clean before L45 lock")
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/eidosoma/groups/42")
    if head != remote:
        raise RuntimeError("L45 local/remote commit mismatch")
    prior = validate_immutable_prior()
    fixtures = fixture_results()
    task = build_task_registry()
    seeds = seed_manifest(task)
    firewall = seed_firewall(seeds)
    benchmark = benchmark_projection()
    if (
        not prior["unchanged"]
        or not fixtures["passed"].all()
        or firewall["status"] != "PASS"
        or benchmark["status"] != "PASS"
    ):
        raise RuntimeError("L45 preoutcome validation failed")

    shutil.copy2(CONFIG, LOOP_ROOT / "preregistration.yaml")
    BASE.atomic_text(
        LOOP_ROOT / "decision_record.md",
        "# S19-L45 decision record\n\n"
        "The human directed L45 as a bounded side quest after L44, then authorized continued process/regime work through L65. L44's prospectively frozen hierarchy selected only `NEW_HEREDITARY_EPISODE_RUN3`: it was committor-reliable and exceeded the exact fixed-count temporal-order null in both candidates and both evaluation cohorts. L45 therefore does not select a process using Phi. Before opening a Phi outcome, this lock freezes the L17/L18-validated BreakingGRNMemories/IIGR pipeline, emergence, integrated Phi-r, synergy and downward-causation identities, current and final-eight slope summaries, fixed ridge-binomial models, development-only fitting, direct heredity/stability/opportunity/mass/phase controls, two evaluation cohorts, 4,096 matrix bootstraps and two 512-permutation families. Prefix-only and completed-fit features remain separate; completed-fit values are never prospectively eligible.\n",
    )
    BASE.write_json(LOOP_ROOT / "immutable_prior_validation.json", prior)
    BASE.write_parquet(LOOP_ROOT / "fixture_results.parquet", fixtures)
    BASE.write_parquet(LOOP_ROOT / "task_registry.parquet", task)
    BASE.write_parquet(LOOP_ROOT / "analysis_seed_manifest.parquet", seeds)
    BASE.write_json(LOOP_ROOT / "seed_firewall.json", firewall)
    BASE.write_json(LOOP_ROOT / "benchmark_projection.json", benchmark)
    BASE.write_parquet(LOOP_ROOT / "source_registry.parquet", source_registry())

    source_snapshot = json.loads(
        (L17_ROOT / "source_snapshot_manifest.json").read_text()
    )
    source_lock = {
        "schema": "eidosoma.e01.s19_l45.source_snapshot_manifest.v1",
        "breakingGrnMemories": source_snapshot["breakingGrnMemories"],
        "IIGR": source_snapshot["IIGR"],
        "PhiRL": source_snapshot["PhiRL"],
        "safeLattice": source_snapshot["safeLattice"],
        "l17SourceSnapshotSha256": sha256_file(
            L17_ROOT / "source_snapshot_manifest.json"
        ),
        "l17ImplementationLockSha256": sha256_file(
            L17_ROOT / "implementation_lock.json"
        ),
        "l18PrefixWorkerSha256": sha256_file(
            ROOT / "scripts/e01/l18_bgm_prefix_worker.py"
        ),
        "l45WorkerSha256": sha256_file(WORKER_PATH),
        "safeLatticeSha256": sha256_file(SAFE_LATTICE),
    }
    BASE.write_json(LOOP_ROOT / "source_snapshot_manifest.json", source_lock)
    locked_inputs = {
        "taskRegistry": LOOP_ROOT / "task_registry.parquet",
        "analysisSeeds": LOOP_ROOT / "analysis_seed_manifest.parquet",
        "firewall": LOOP_ROOT / "seed_firewall.json",
        "benchmark": LOOP_ROOT / "benchmark_projection.json",
        "sourceSnapshot": LOOP_ROOT / "source_snapshot_manifest.json",
        "l44StateProcess": L44_ROOT / "state_process_results.parquet",
        "l44PrefixControls": L44_ROOT / "prefix_control_registry.parquet",
        "l44ArtifactManifest": L44_ROOT / "artifact_manifest.json",
        "l23TrajectoryManifest": L23_ROOT / "input_trajectory_manifest.parquet",
        "safeLattice": SAFE_LATTICE,
    }
    hashes = {name: sha256_file(path) for name, path in locked_inputs.items()}
    lock = {
        "schema": "eidosoma.e01.s19_l45.implementation_lock.v1",
        "repositoryHead": head,
        "remoteHead": remote,
        "runnerSha256": sha256_file(RUNNER_PATH),
        "coreSha256": sha256_file(CORE_PATH),
        "workerSha256": sha256_file(WORKER_PATH),
        "target": TARGET,
        "temporalModes": list(MODES),
        "metricIdentities": list(METRICS),
        "metricSummaries": ["current", "recentSlopeFinal8"],
        "modelIds": list(MODEL_IDS),
        "ridgeC": 1.0,
        "trainingCohort": "L28_DEVELOPMENT",
        "evaluationCohorts": list(EVALUATION_COHORTS),
        "matrixBootstraps": BOOTSTRAPS,
        "permutationReplicates": PERMUTATIONS,
        "workers": WORKERS,
        "newMatrices": 0,
        "newTrajectories": 0,
        "newBranchStreams": 0,
        "completedTestTrajectoryUsedByTarget": False,
        "lockedInputHashes": hashes,
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
            "workerSha256": sha256_file(WORKER_PATH),
            "lockedInputHashes": hashes,
        },
    )


def compute_downstream(
    task: pd.DataFrame, phi: pd.DataFrame, direct: pd.DataFrame
) -> tuple[dict[str, pd.DataFrame], list[str], list[str]]:
    features, past_columns, completed_columns = feature_table(task, phi, direct)
    predictions, model_registry, model_replay = fit_models_and_predict(
        features, past_columns, completed_columns
    )
    metrics = metric_table(predictions)
    bootstrap, comparisons = bootstrap_model_results(predictions)
    permutation_nulls, permutation_summary = permutation_controls(
        features, predictions, past_columns, comparisons
    )
    tables = {
        "state_feature_results.parquet": features,
        "model_registry.parquet": model_registry,
        "model_replay_validation.parquet": model_replay,
        "prediction_results.parquet": predictions,
        "model_metric_results.parquet": metrics,
        "bootstrap_results.parquet": bootstrap,
        "incremental_comparison_results.parquet": comparisons,
        "permutation_null_results.parquet": permutation_nulls,
        "permutation_control_summary.parquet": permutation_summary,
    }
    return tables, past_columns, completed_columns


def execute() -> None:
    started = time.perf_counter()
    started_cpu = time.process_time()
    lock = json.loads((LOOP_ROOT / "preoutcome_repository_lock.json").read_text())
    if (
        git("rev-parse", "HEAD") != lock["head"]
        or git("rev-parse", "origin/eidosoma/groups/42") != lock["remote"]
        or git("status", "--porcelain=v1")
    ):
        raise RuntimeError("L45 repository lock mismatch")
    prior = validate_immutable_prior()
    fixtures = fixture_results()
    locked_inputs = {
        "taskRegistry": LOOP_ROOT / "task_registry.parquet",
        "analysisSeeds": LOOP_ROOT / "analysis_seed_manifest.parquet",
        "firewall": LOOP_ROOT / "seed_firewall.json",
        "benchmark": LOOP_ROOT / "benchmark_projection.json",
        "sourceSnapshot": LOOP_ROOT / "source_snapshot_manifest.json",
        "l44StateProcess": L44_ROOT / "state_process_results.parquet",
        "l44PrefixControls": L44_ROOT / "prefix_control_registry.parquet",
        "l44ArtifactManifest": L44_ROOT / "artifact_manifest.json",
        "l23TrajectoryManifest": L23_ROOT / "input_trajectory_manifest.parquet",
        "safeLattice": SAFE_LATTICE,
    }
    if any(
        sha256_file(path) != lock["lockedInputHashes"][name]
        for name, path in locked_inputs.items()
    ):
        raise RuntimeError("L45 locked input changed")
    if (
        not prior["unchanged"]
        or prior["aggregateSha256"] != lock["priorAggregateSha256"]
        or not fixtures["passed"].all()
        or sha256_file(RUNNER_PATH) != lock["runnerSha256"]
        or sha256_file(CORE_PATH) != lock["coreSha256"]
        or sha256_file(WORKER_PATH) != lock["workerSha256"]
    ):
        raise RuntimeError("L45 pre-execution validation failed")
    task = pd.read_parquet(LOOP_ROOT / "task_registry.parquet")
    if frame_hash(task) != frame_hash(build_task_registry()):
        raise RuntimeError("L45 task registry regeneration mismatch")

    if BUILD_ROOT.exists():
        shutil.rmtree(BUILD_ROOT)
    BUILD_ROOT.mkdir(parents=True)
    first_phi, first_direct, first_identity, first_runtime = execute_feature_pass(
        task, CACHE_ROOT / "feature_pass_1"
    )
    first_tables, _, _ = compute_downstream(task, first_phi, first_direct)
    first_suffix = suffix_invariance_results(
        task, first_phi, CACHE_ROOT / "suffix_pass_1"
    )
    gates, classifications, next_theme = scientific_gates(
        first_tables["incremental_comparison_results.parquet"],
        first_tables["permutation_control_summary.parquet"],
        first_suffix,
        first_phi,
    )
    make_figures(
        first_tables["model_metric_results.parquet"],
        first_tables["incremental_comparison_results.parquet"],
        first_tables["permutation_control_summary.parquet"],
        first_suffix,
        first_phi,
        gates,
    )

    # Full independent feature and downstream regeneration from the immutable
    # trajectories, source lock, state indices and seeds.
    second_phi, second_direct, second_identity, second_runtime = execute_feature_pass(
        task, CACHE_ROOT / "feature_pass_2"
    )
    second_tables, _, _ = compute_downstream(task, second_phi, second_direct)
    second_suffix = suffix_invariance_results(
        task, second_phi, CACHE_ROOT / "suffix_pass_2"
    )
    second_gates, second_classes, second_next = scientific_gates(
        second_tables["incremental_comparison_results.parquet"],
        second_tables["permutation_control_summary.parquet"],
        second_suffix,
        second_phi,
    )
    comparisons = {
        "phiFeatures": (first_phi, second_phi),
        "directFeatures": (first_direct, second_direct),
        "trajectoryIdentity": (first_identity, second_identity),
        "suffixInvariance": (first_suffix, second_suffix),
        "scientificGates": (gates, second_gates),
        **{name: (frame, second_tables[name]) for name, frame in first_tables.items()},
    }
    exact = {
        name: frame_hash(left) == frame_hash(right)
        for name, (left, right) in comparisons.items()
    }
    regeneration = {
        "schema": "eidosoma.e01.s19_l45.regeneration_validation.v1",
        "status": "PASS"
        if all(exact.values())
        and classifications == second_classes
        and next_theme == second_next
        else "FAIL",
        "tableExact": exact,
        "classificationExact": classifications == second_classes,
        "nextThemeExact": next_theme == second_next,
        "stateCount": len(second_identity),
        "phiFeatureRows": len(second_phi),
    }
    if regeneration["status"] != "PASS":
        raise RuntimeError("L45 exact regeneration failure")

    tables = {
        "trajectory_identity_validation.parquet": first_identity,
        "direct_control_features.parquet": first_direct,
        "phi_feature_results.parquet": first_phi,
        "feature_runtime_results.parquet": first_runtime,
        "suffix_invariance_results.parquet": first_suffix,
        "scientific_gate_results.parquet": gates,
        **first_tables,
    }
    for name, frame in tables.items():
        BASE.write_parquet(BUILD_ROOT / name, frame)
    BASE.write_json(
        BUILD_ROOT / "classification.json",
        {
            "schema": "eidosoma.e01.s19_l45.classification.v1",
            "classifications": classifications,
            "nextTheme": next_theme,
            "target": TARGET,
            "pastOnlyProspectiveLead": "PAST_ONLY_PHI_INCREMENTAL_PROCESS_LEAD"
            in classifications,
            "completedFitProspectiveEligible": False,
            "priorStatusesChanged": False,
        },
    )
    pd.DataFrame(
        columns=[
            "failureId",
            "stage",
            "status",
            "reason",
            "scientificValuesReleased",
        ]
    ).to_csv(BUILD_ROOT / "failure_ledger.csv", index=False)
    runtime = {
        "schema": "eidosoma.e01.s19_l45.runtime.v1",
        "repositoryHead": lock["head"],
        "workers": WORKERS,
        "numericalLibraryThreadsPerWorker": 1,
        "gpuHours": 0,
        "wallSeconds": time.perf_counter() - started,
        "controllerCpuHours": (time.process_time() - started_cpu) / 3600,
        "states": len(task),
        "phiPipelineEvaluations": 280 * 2 * 2 + len(first_suffix) + len(second_suffix),
        "firstFeaturePassWorkerWallSeconds": float(first_runtime["wallSeconds"].sum()),
        "secondFeaturePassWorkerWallSeconds": float(
            second_runtime["wallSeconds"].sum()
        ),
        "newMatrices": 0,
        "newTrajectories": 0,
        "newBranchStreams": 0,
        "completedAtUtc": utc_now(),
    }
    BASE.write_json(BUILD_ROOT / "runtime_manifest.json", runtime)
    BASE.write_json(BUILD_ROOT / "regeneration_validation.json", regeneration)
    retained_bytes = sum(
        path.stat().st_size for path in BUILD_ROOT.rglob("*") if path.is_file()
    )
    storage = {
        "schema": "eidosoma.e01.s19_l45.storage_validation.v1",
        "status": "PASS" if retained_bytes <= 25 * 1024**3 else "FAIL",
        "retainedBytes": retained_bytes,
        "retainedGiBCeiling": 25,
        "temporaryBytes": retained_bytes,
        "temporaryGiBCeiling": 75,
    }
    BASE.write_json(BUILD_ROOT / "storage_validation.json", storage)
    report = report_text(
        first_tables["model_metric_results.parquet"],
        first_tables["incremental_comparison_results.parquet"],
        first_tables["permutation_control_summary.parquet"],
        first_suffix,
        gates,
        classifications,
        runtime,
        next_theme,
    )
    BASE.atomic_text(BUILD_ROOT / "S19_L45_FULL_RESULTS.md", report)
    BASE.atomic_text(BUILD_ROOT / "research_step_full_results.md", report)
    BASE.atomic_text(
        BUILD_ROOT / "loop_decision_summary.md",
        f"# S19-L45 decision summary\n\n**Classification:** {', '.join(classifications)}\n\n**Next:** `{next_theme}`.\n",
    )
    for path in (BUILD_ROOT / "figures").glob("*.png"):
        if not path.stat().st_size:
            raise RuntimeError(f"empty L45 figure: {path}")
    if storage["status"] != "PASS":
        raise RuntimeError("L45 storage ceiling exceeded")

    for path in BUILD_ROOT.iterdir():
        destination = LOOP_ROOT / path.name
        if path.is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(path, destination)
        else:
            shutil.copy2(path, destination)
    BASE.write_json(LOOP_ROOT / "artifact_manifest.json", manifest_for(LOOP_ROOT))
    if manifest_for(LOOP_ROOT) != json.loads(
        (LOOP_ROOT / "artifact_manifest.json").read_text()
    ):
        raise RuntimeError("L45 artifact manifest regeneration failed")

    append_ledgers(classifications, runtime["completedAtUtc"], next_theme)
    root_report = (
        f"# S19 current-step report\n\nLatest completed loop: `{LOOP_ID}`.\n\n"
        f"Classification: {', '.join(classifications)}.\n\n"
        f"Next autonomous theme: `{next_theme}`.\n"
    )
    BASE.atomic_text(ARTIFACT_ROOT / "S19_CURRENT_STEP_REPORT.md", root_report)
    BASE.atomic_text(ARTIFACT_ROOT / "CURRENT_STEP_HANDOFF.md", root_report)
    BASE.write_json(
        ARTIFACT_ROOT / "s19_status.json",
        {
            "schema": "eidosoma.e01.s19.status.v1",
            "programStatus": "ACTIVE_AUTONOMOUS_SEQUENCE",
            "latestCompletedLoop": LOOP_ID,
            "latestClassification": classifications,
            "selectedDiscoveryLead": "PAST_ONLY_PHIID_HEREDITARY_EPISODE"
            if "PAST_ONLY_PHI_INCREMENTAL_PROCESS_LEAD" in classifications
            else None,
            "nextAuthorizedLoop": "S19-L46",
            "authorizationUpperBound": "S19-L65",
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
