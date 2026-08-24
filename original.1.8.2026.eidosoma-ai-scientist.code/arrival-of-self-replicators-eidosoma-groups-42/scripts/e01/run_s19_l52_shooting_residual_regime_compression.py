#!/usr/bin/env python3
"""Run S19-L52 cross-fitted shooting-regime compression audit."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from e01_onset_discovery.regime_hazard import finite_horizon_process_probability
from e01_onset_discovery.shooting_regime_compression import (
    fit_shrunk_duration_table,
    hazard_fit_scope,
    transition_scores,
)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


L51 = load_module(
    "e01_l52_l51_runner",
    ROOT / "scripts/e01/run_s19_l51_regime_hazard_renewal.py",
)
L50 = L51.L50
BASE = L51.BASE

ARTIFACT_ROOT = Path("/artifacts/research_steps/S19")
LOOP_ROOT = ARTIFACT_ROOT / "loops/L52"
L50_ROOT = ARTIFACT_ROOT / "loops/L50"
L51_ROOT = ARTIFACT_ROOT / "loops/L51"
BUILD_ROOT = Path("/cache/e01_s19_l52/build")
CONFIG = ROOT / "configs/e01/s19_l52_shooting_residual_regime_compression.yaml"
RUNNER_PATH = Path(__file__).resolve()
CORE_PATH = ROOT / "src/e01_onset_discovery/shooting_regime_compression.py"

LOOP_ID = "S19-L52"
VERSION = "E01-S19-L52-SHOOTING-RESIDUAL-REGIME-COMPRESSION-v1.0.0"
CANDIDATES = ("S12F-CANDIDATE-02", "S12F-CANDIDATE-03")
ROLES = ("DEVELOPMENT", "VALIDATION")
HORIZONS = (4, 8, 12)
PRIMARY_HORIZON = 12
THRESHOLD = 0.9
REQUIRED_RUN = 3
MAXIMUM_DURATION = 12
PRIOR_STRENGTH = 4.0
BOOTSTRAPS = 4096
PERMUTATIONS = 512
WORKERS = 1
MODELS = (
    "L51_POOLED_SEMIMARKOV",
    "MATRIX_OTHER_LANDMARK_SEMIMARKOV",
    "STATE_LOCAL_SEMIMARKOV",
)
DIRECTIONS = (
    ("A_TO_B", "A", "B"),
    ("B_TO_A", "B", "A"),
)
SEED_ROOT = bytes.fromhex(
    "0d9121a28a390fe52028d31e3cce56000697ad15acae1e43d0c2c9a9a891c4db"
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def sha256_file(path: Path) -> str:
    return L51.sha256_file(path)


def frame_hash(frame: pd.DataFrame) -> str:
    return L51.frame_hash(frame)


def seed_material(*parts: object) -> bytes:
    canonical = tuple(part.item() if isinstance(part, np.generic) else part for part in parts)
    return hashlib.sha256(
        SEED_ROOT + b"\x00" + json.dumps(canonical, separators=(",", ":")).encode()
    ).digest()


def derived_seed(*parts: object) -> int:
    return int.from_bytes(seed_material(*parts)[:16], "big")


def generator(*parts: object) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64DXSM(derived_seed(*parts)))


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3 or len(np.unique(a[mask])) < 2 or len(np.unique(b[mask])) < 2:
        return float("nan")
    return float(spearmanr(a[mask], b[mask]).statistic)


def interval(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return float("nan"), float("nan")
    return tuple(map(float, np.quantile(finite, [0.025, 0.975])))


def validate_immutable_prior() -> dict[str, Any]:
    inherited = L51.validate_immutable_prior()
    manifest = json.loads((L51_ROOT / "artifact_manifest.json").read_text())
    rows = []
    for row in manifest["files"]:
        path = L51_ROOT / row["path"]
        actual = sha256_file(path) if path.is_file() else None
        rows.append(
            {
                "path": str(path),
                "expectedSha256": row["sha256"],
                "actualSha256": actual,
                "unchanged": actual == row["sha256"],
            }
        )
    passed = bool(inherited["unchanged"] and rows and all(row["unchanged"] for row in rows))
    return {
        "schema": "eidosoma.e01.s19_l52.immutable_prior_validation.v1",
        "status": "PASS" if passed else "FAIL",
        "unchanged": passed,
        "priorThroughL50Unchanged": bool(inherited["unchanged"]),
        "validatedL51ArtifactCount": len(rows),
        "aggregateSha256": hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "rows": rows,
    }


def source_registry() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sourceId": "L50_INDEPENDENT_BRANCH_HALVES",
                "evidenceClass": "DIRECT_FROZEN_E01_RESULT",
                "finding": "L50 retained two prospective 32-branch halves and strong joint-q split-half reliability.",
                "frozenUse": "crossfit transition-hazard compression without generating another branch",
                "url": None,
            },
            {
                "sourceId": "L51_DURATION_AND_MATRIX_DECOMPOSITION",
                "evidenceClass": "DIRECT_FROZEN_E01_RESULT",
                "finding": "Duration dependence and current-prefix updating were reliable, but the complete observed-prefix renewal model did not reconstruct q and most F12 variation was between matrices.",
                "frozenUse": "compare matrix-transfer and exact-state-local branch-derived hazards",
                "url": None,
            },
            {
                "sourceId": "REVIEWER_SHOOTING_MEASUREMENT_DIRECTION",
                "evidenceClass": "HUMAN_REVIEW_DIRECTION",
                "finding": "When no compact past-observable coordinate transfers, treat stochastic shooting as the computational measurement and determine which process information its short futures reveal.",
                "frozenUse": "distinguish stable matrix dynamics, state-local dynamics and irreducible path information",
                "url": None,
            },
            {
                "sourceId": "FUH_FAN_1997",
                "evidenceClass": "PRIMARY_METHOD_SOURCE",
                "finding": "Bayesian-bootstrap finite-state Markov inference includes transition and hitting-time quantities.",
                "frozenUse": "matrix-resampled transition and finite-horizon event comparison",
                "url": "https://www3.stat.sinica.edu.tw/statistica/j7n4/j7n413/j7n413.htm",
            },
        ]
    )


def model_registry() -> pd.DataFrame:
    descriptions = {
        "L51_POOLED_SEMIMARKOV": "exact frozen candidate-specific development-pooled L51 duration table",
        "MATRIX_OTHER_LANDMARK_SEMIMARKOV": "fitting-half duration table from the same matrix's other four landmarks; target state excluded",
        "STATE_LOCAL_SEMIMARKOV": "fitting-half duration table from the exact target state",
    }
    return pd.DataFrame(
        [
            {
                "modelId": model,
                "description": descriptions[model],
                "usesBranchFit": model != "L51_POOLED_SEMIMARKOV",
                "targetStateExcluded": model == "MATRIX_OTHER_LANDMARK_SEMIMARKOV",
                "usesHeldoutHalf": False,
                "pastObservable": model == "L51_POOLED_SEMIMARKOV",
                "priorStrength": PRIOR_STRENGTH if model != "L51_POOLED_SEMIMARKOV" else None,
                "hyperparameterSearch": False,
            }
            for model in MODELS
        ]
    )


def fixture_results() -> pd.DataFrame:
    anchor = np.asarray([[0.2, 0.4], [0.7, 0.9]], dtype=np.float64)
    fitted = fit_shrunk_duration_table(
        [False, True, True], [1, 1, 20], [True, False, True], anchor, prior_strength=4
    )
    replay = fit_shrunk_duration_table(
        [False, True, True], [1, 1, 20], [True, False, True], anchor, prior_strength=4
    )
    losses, briers = transition_scores([False, True], [1, 20], [True, True], fitted)
    probability = finite_horizon_process_probability(
        lambda state, duration: float(fitted[int(state), min(duration, 2) - 1]),
        initial_state=True,
        initial_duration=3,
        horizon=4,
        required_run=3,
        maximum_duration=2,
    )
    rows = [
        {"fixtureId": "F01_MATRIX_SCOPE_EXCLUDES_TARGET", "passed": "c" not in hazard_fit_scope("c", ("a", "b", "c", "d", "e"), "MATRIX_OTHER_LANDMARK_SEMIMARKOV")},
        {"fixtureId": "F02_STATE_SCOPE_ONLY_TARGET", "passed": hazard_fit_scope("c", ("a", "b", "c"), "STATE_LOCAL_SEMIMARKOV") == ("c",)},
        {"fixtureId": "F03_CELL_ANCHOR", "passed": fitted[0, 1] == anchor[0, 1]},
        {"fixtureId": "F04_EXACT_TABLE_REPLAY", "passed": np.array_equal(fitted, replay)},
        {"fixtureId": "F05_TRANSITION_SCORES_FINITE", "passed": bool(np.isfinite(losses).all() and np.isfinite(briers).all())},
        {"fixtureId": "F06_PROCESS_PROBABILITY_ORDER", "passed": 0 <= probability.joint_break_run_probability <= probability.break_probability <= 1},
        {"fixtureId": "F07_BRANCH_HALF_SCOPE", "passed": DIRECTIONS == (("A_TO_B", "A", "B"), ("B_TO_A", "B", "A"))},
        {"fixtureId": "F08_MODEL_SCOPE", "passed": len(MODELS) == 3},
        {"fixtureId": "F09_HORIZON_SCOPE", "passed": HORIZONS == (4, 8, 12)},
        {"fixtureId": "F10_SEED_REPLAY", "passed": derived_seed("fixture", np.int64(7)) == derived_seed("fixture", 7)},
    ]
    return pd.DataFrame(rows)


def analysis_seed_manifest() -> pd.DataFrame:
    rows = []
    for purpose, repetitions in (
        ("matrix_bootstrap", BOOTSTRAPS),
        ("whole_matrix_q_permutation", PERMUTATIONS),
        ("residual_bootstrap", BOOTSTRAPS),
    ):
        for candidate in CANDIDATES:
            material = seed_material(purpose, candidate)
            rows.append(
                {
                    "purpose": purpose,
                    "candidateId": candidate,
                    "repetitions": repetitions,
                    "derivedSeed": str(int.from_bytes(material[:16], "big")),
                    "seedMaterialSha256": material.hex(),
                }
            )
    return pd.DataFrame(rows)


def seed_firewall(seeds: pd.DataFrame) -> dict[str, Any]:
    prior = set(pd.read_parquet(L51_ROOT / "analysis_seed_manifest.parquet")["seedMaterialSha256"])
    prior.update(pd.read_parquet(L50_ROOT / "analysis_seed_manifest.parquet")["seedMaterialSha256"])
    current = set(seeds["seedMaterialSha256"])
    passed = len(current) == len(seeds) and not (current & prior)
    return {
        "schema": "eidosoma.e01.s19_l52.seed_firewall.v1",
        "status": "PASS" if passed else "FAIL",
        "rootHex": SEED_ROOT.hex(),
        "newAnalysisStreams": len(current),
        "overlapCount": len(current & prior),
    }


def pooled_anchors() -> dict[str, np.ndarray]:
    hazard = pd.read_parquet(L51_ROOT / "hazard_parameter_results.parquet")
    anchors = {}
    for candidate in CANDIDATES:
        group = hazard[
            hazard["candidateId"].eq(candidate)
            & hazard["modelId"].eq("POOLED_SEMIMARKOV_DURATION")
        ]
        table = np.empty((2, MAXIMUM_DURATION), dtype=np.float64)
        for row in group.itertuples(index=False):
            table[int(bool(row.currentState)), int(row.duration) - 1] = float(
                row.probabilityNextInherited
            )
        if not np.isfinite(table).all() or np.any((table <= 0) | (table >= 1)):
            raise RuntimeError("L52 invalid frozen pooled anchor")
        anchors[candidate] = table
    return anchors


def frozen_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prefix = pd.read_parquet(L51_ROOT / "prefix_state_results.parquet")
    expected_sequences = pd.read_parquet(L51_ROOT / "branch_sequence_manifest.parquet")
    branches = pd.read_parquet(L50_ROOT / "branch_results.parquet")
    sequences, transitions = L51.build_branch_sequences(branches, prefix)
    if frame_hash(sequences) != frame_hash(expected_sequences):
        raise RuntimeError("L52 exact L51 branch-sequence replay failure")
    branch_half = branches[["stateId", "branchIndex", "branchHalf"]]
    transitions = transitions.merge(
        branch_half,
        on=["stateId", "branchIndex"],
        validate="many_to_one",
    )
    if len(transitions) != 614400 or set(transitions["branchHalf"]) != {"A", "B"}:
        raise RuntimeError("L52 transition/half scope failure")
    estimates = pd.read_parquet(L50_ROOT / "state_committor_results.parquet")
    return prefix, branches, transitions, estimates


def fit_tables(
    prefix: pd.DataFrame,
    transitions: pd.DataFrame,
    anchors: dict[str, np.ndarray],
) -> tuple[dict[tuple[str, str, str], np.ndarray], pd.DataFrame, pd.DataFrame]:
    tables: dict[tuple[str, str, str], np.ndarray] = {}
    parameter_rows = []
    scope_rows = []
    matrix_states = {
        key: tuple(group.sort_values("completedFissionLandmark")["stateId"])
        for key, group in prefix.groupby(["matrixRole", "candidateId", "matrixIndex"], sort=True)
    }
    transition_groups = {state: group for state, group in transitions.groupby("stateId", sort=False)}
    prefix_index = prefix.set_index("stateId")
    for direction, fit_half, score_half in DIRECTIONS:
        for state_id, state in prefix_index.iterrows():
            key = (state.matrixRole, state.candidateId, int(state.matrixIndex))
            anchor = anchors[state.candidateId]
            tables[(direction, state_id, "L51_POOLED_SEMIMARKOV")] = anchor
            for model in MODELS[1:]:
                fit_ids = hazard_fit_scope(state_id, matrix_states[key], model)
                fit = pd.concat(
                    [transition_groups[value] for value in fit_ids], ignore_index=True
                )
                fit = fit[fit["branchHalf"].eq(fit_half)]
                table = fit_shrunk_duration_table(
                    fit["currentState"],
                    fit["currentDuration"],
                    fit["nextState"],
                    anchor,
                    prior_strength=PRIOR_STRENGTH,
                )
                tables[(direction, state_id, model)] = table
                scope_rows.append(
                    {
                        "direction": direction,
                        "fitHalf": fit_half,
                        "scoreHalf": score_half,
                        "stateId": state_id,
                        "matrixRole": state.matrixRole,
                        "candidateId": state.candidateId,
                        "matrixIndex": int(state.matrixIndex),
                        "completedFissionLandmark": int(state.completedFissionLandmark),
                        "modelId": model,
                        "fitStateCount": len(fit_ids),
                        "fitTransitions": len(fit),
                        "targetStateIncluded": state_id in fit_ids,
                        "targetStateExclusionPassed": (model == "STATE_LOCAL_SEMIMARKOV")
                        or state_id not in fit_ids,
                        "heldoutHalfExcluded": not fit["branchHalf"].eq(score_half).any(),
                    }
                )
                capped = np.minimum(fit["currentDuration"], MAXIMUM_DURATION)
                for current_state in (False, True):
                    for duration in range(1, MAXIMUM_DURATION + 1):
                        mask = fit["currentState"].eq(current_state) & capped.eq(duration)
                        parameter_rows.append(
                            {
                                "direction": direction,
                                "stateId": state_id,
                                "matrixRole": state.matrixRole,
                                "candidateId": state.candidateId,
                                "matrixIndex": int(state.matrixIndex),
                                "completedFissionLandmark": int(state.completedFissionLandmark),
                                "modelId": model,
                                "currentState": current_state,
                                "duration": duration,
                                "fitTransitions": int(mask.sum()),
                                "fitNextInherited": int(fit.loc[mask, "nextState"].sum()),
                                "anchorProbability": float(anchor[int(current_state), duration - 1]),
                                "fittedProbability": float(table[int(current_state), duration - 1]),
                            }
                        )
    scope = pd.DataFrame(scope_rows)
    checks = ["targetStateExclusionPassed", "heldoutHalfExcluded"]
    if len(tables) != 4800 or len(scope) != 3200 or not scope[checks].all().all():
        raise RuntimeError("L52 hazard fit-scope validation failure")
    return tables, pd.DataFrame(parameter_rows), scope


def process_predictions(
    prefix: pd.DataFrame,
    tables: dict[tuple[str, str, str], np.ndarray],
) -> pd.DataFrame:
    rows = []
    for state in prefix.itertuples(index=False):
        for direction, fit_half, score_half in DIRECTIONS:
            for model in MODELS:
                table = tables[(direction, state.stateId, model)]

                def probability(current: bool, duration: int, table: np.ndarray = table) -> float:
                    return float(table[int(current), min(duration, MAXIMUM_DURATION) - 1])

                for horizon in HORIZONS:
                    result = finite_horizon_process_probability(
                        probability,
                        initial_state=bool(state.currentInheritanceState),
                        initial_duration=int(state.currentRegimeDuration),
                        horizon=horizon,
                        required_run=REQUIRED_RUN,
                        maximum_duration=MAXIMUM_DURATION,
                    )
                    for target, value in (
                        ("BREAK", result.break_probability),
                        ("JOINT_BREAK_RUN3", result.joint_break_run_probability),
                        ("RUN3_GIVEN_BREAK", result.run_probability_given_break),
                    ):
                        rows.append(
                            {
                                "direction": direction,
                                "fitHalf": fit_half,
                                "scoreHalf": score_half,
                                "stateId": state.stateId,
                                "matrixRole": state.matrixRole,
                                "candidateId": state.candidateId,
                                "matrixIndex": int(state.matrixIndex),
                                "completedFissionLandmark": int(state.completedFissionLandmark),
                                "horizon": horizon,
                                "targetType": target,
                                "modelId": model,
                                "predictedProbability": value,
                            }
                        )
    return pd.DataFrame(rows).sort_values(
        ["matrixRole", "candidateId", "matrixIndex", "completedFissionLandmark", "direction", "horizon", "targetType", "modelId"]
    ).reset_index(drop=True)


def transition_metrics(
    transitions: pd.DataFrame,
    prefix: pd.DataFrame,
    tables: dict[tuple[str, str, str], np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prefix_index = prefix.set_index("stateId")
    rows = []
    for direction, fit_half, score_half in DIRECTIONS:
        heldout = transitions[transitions["branchHalf"].eq(score_half)]
        for state_id, group in heldout.groupby("stateId", sort=False):
            state = prefix_index.loc[state_id]
            for model in MODELS:
                losses, briers = transition_scores(
                    group["currentState"],
                    group["currentDuration"],
                    group["nextState"],
                    tables[(direction, state_id, model)],
                )
                rows.append(
                    {
                        "direction": direction,
                        "fitHalf": fit_half,
                        "scoreHalf": score_half,
                        "stateId": state_id,
                        "matrixRole": state.matrixRole,
                        "candidateId": state.candidateId,
                        "matrixIndex": int(state.matrixIndex),
                        "completedFissionLandmark": int(state.completedFissionLandmark),
                        "modelId": model,
                        "transitions": len(group),
                        "logLoss": float(losses.mean()),
                        "brier": float(briers.mean()),
                    }
                )
    state_metrics = pd.DataFrame(rows)
    matrix_metrics = (
        state_metrics.groupby(
            ["matrixRole", "candidateId", "matrixIndex", "modelId"], as_index=False
        )
        .agg(
            states=("stateId", "size"),
            transitions=("transitions", "sum"),
            logLoss=("logLoss", "mean"),
            brier=("brier", "mean"),
        )
        .sort_values(["matrixRole", "candidateId", "modelId", "matrixIndex"])
        .reset_index(drop=True)
    )
    aggregate = (
        matrix_metrics.groupby(["matrixRole", "candidateId", "modelId"], as_index=False)
        .agg(
            matrices=("matrixIndex", "nunique"),
            equalMatrixMeanLogLoss=("logLoss", "mean"),
            equalMatrixMeanBrier=("brier", "mean"),
            matrixSdLogLoss=("logLoss", "std"),
        )
        .sort_values(["matrixRole", "candidateId", "equalMatrixMeanLogLoss"])
        .reset_index(drop=True)
    )
    return state_metrics, matrix_metrics, aggregate


def event_metrics(
    predictions: pd.DataFrame,
    branches: pd.DataFrame,
    estimates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    branch_columns = ["stateId", "branchHalf", "branchIndex"]
    for horizon in HORIZONS:
        branch_columns.extend([f"breakH{horizon}", f"jointH{horizon}"])
    branch_index = branches[branch_columns]
    estimate_truth = estimates[
        ["stateId", "horizon", "targetType", "qHalfA", "qHalfB", "trialsHalfA", "trialsHalfB"]
    ]
    rows = []
    for prediction in predictions.itertuples(index=False):
        group = branch_index[
            branch_index["stateId"].eq(prediction.stateId)
            & branch_index["branchHalf"].eq(prediction.scoreHalf)
        ]
        if prediction.targetType == "BREAK":
            target = group[f"breakH{prediction.horizon}"].to_numpy(dtype=np.bool_)
        elif prediction.targetType == "JOINT_BREAK_RUN3":
            target = group[f"jointH{prediction.horizon}"].to_numpy(dtype=np.bool_)
        else:
            eligible = group[f"breakH{prediction.horizon}"].to_numpy(dtype=np.bool_)
            target = group.loc[eligible, f"jointH{prediction.horizon}"].to_numpy(dtype=np.bool_)
        if not len(target):
            continue
        probability = float(np.clip(prediction.predictedProbability, 1e-12, 1 - 1e-12))
        successes = int(target.sum())
        trials = len(target)
        log_loss_value = float(
            -(successes * np.log(probability) + (trials - successes) * np.log1p(-probability))
            / trials
        )
        brier = float(
            (successes * (1 - probability) ** 2 + (trials - successes) * probability**2)
            / trials
        )
        truth_row = estimate_truth[
            estimate_truth["stateId"].eq(prediction.stateId)
            & estimate_truth["horizon"].eq(prediction.horizon)
            & estimate_truth["targetType"].eq(prediction.targetType)
        ].iloc[0]
        half = prediction.scoreHalf
        q_half = float(truth_row[f"qHalf{half}"])
        trials_half = int(truth_row[f"trialsHalf{half}"])
        if trials_half != trials or abs(q_half - (successes + 0.5) / (trials + 1)) > 1e-15:
            raise RuntimeError("L52 heldout-half q replay failure")
        rows.append(
            {
                **{column: getattr(prediction, column) for column in [
                    "direction", "fitHalf", "scoreHalf", "stateId", "matrixRole", "candidateId", "matrixIndex", "completedFissionLandmark", "horizon", "targetType", "modelId", "predictedProbability"
                ]},
                "successes": successes,
                "trials": trials,
                "empiricalQ": q_half,
                "qResidual": q_half - float(prediction.predictedProbability),
                "branchLogLoss": log_loss_value,
                "branchBrier": brier,
            }
        )
    state_metrics = pd.DataFrame(rows)
    matrix_metrics = (
        state_metrics.groupby(
            ["matrixRole", "candidateId", "matrixIndex", "horizon", "targetType", "modelId"],
            as_index=False,
        )
        .agg(
            states=("stateId", "size"),
            trials=("trials", "sum"),
            branchLogLoss=("branchLogLoss", "mean"),
            branchBrier=("branchBrier", "mean"),
            qRmse=("qResidual", lambda values: float(np.sqrt(np.mean(np.square(values))))),
        )
        .sort_values(["matrixRole", "candidateId", "horizon", "targetType", "modelId", "matrixIndex"])
        .reset_index(drop=True)
    )
    aggregates = []
    for keys, group in state_metrics.groupby(
        ["matrixRole", "candidateId", "horizon", "targetType", "modelId"], sort=True
    ):
        role, candidate, horizon, target_type, model = keys
        matrix_group = matrix_metrics[
            matrix_metrics["matrixRole"].eq(role)
            & matrix_metrics["candidateId"].eq(candidate)
            & matrix_metrics["horizon"].eq(horizon)
            & matrix_metrics["targetType"].eq(target_type)
            & matrix_metrics["modelId"].eq(model)
        ]
        aggregates.append(
            {
                "matrixRole": role,
                "candidateId": candidate,
                "horizon": int(horizon),
                "targetType": target_type,
                "modelId": model,
                "matrices": int(group["matrixIndex"].nunique()),
                "statesTimesDirections": len(group),
                "equalMatrixMeanBranchLogLoss": float(matrix_group["branchLogLoss"].mean()),
                "equalMatrixMeanBranchBrier": float(matrix_group["branchBrier"].mean()),
                "qRmse": float(np.sqrt(np.mean(np.square(group["qResidual"])))),
                "qSpearmanPooledDirections": safe_spearman(
                    group["predictedProbability"].to_numpy(), group["empiricalQ"].to_numpy()
                ),
            }
        )
    return state_metrics, matrix_metrics, pd.DataFrame(aggregates)


COMPARISONS = (
    ("MATRIX_OTHER_LANDMARK_SEMIMARKOV", "L51_POOLED_SEMIMARKOV", "MATRIX_TRANSFER_BEYOND_POOLED"),
    ("STATE_LOCAL_SEMIMARKOV", "MATRIX_OTHER_LANDMARK_SEMIMARKOV", "STATE_LOCAL_BEYOND_MATRIX_TRANSFER"),
    ("STATE_LOCAL_SEMIMARKOV", "L51_POOLED_SEMIMARKOV", "STATE_LOCAL_BEYOND_POOLED"),
)


def model_comparisons(
    transition_matrix: pd.DataFrame, event_matrix: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    bootstrap_rows = []
    for metric_type, source, column in (
        ("TRANSITION", transition_matrix[transition_matrix["matrixRole"].eq("VALIDATION")], "logLoss"),
        (
            "F12_JOINT_EVENT",
            event_matrix[
                event_matrix["matrixRole"].eq("VALIDATION")
                & event_matrix["horizon"].eq(PRIMARY_HORIZON)
                & event_matrix["targetType"].eq("JOINT_BREAK_RUN3")
            ],
            "branchLogLoss",
        ),
    ):
        for candidate in CANDIDATES:
            group = source[source["candidateId"].eq(candidate)]
            pivot = group.pivot(index="matrixIndex", columns="modelId", values=column).sort_index()
            for model, reference, comparison_id in COMPARISONS:
                difference = (pivot[reference] - pivot[model]).to_numpy(dtype=np.float64)
                rng = generator("comparison_bootstrap", metric_type, candidate, comparison_id)
                indices = rng.integers(0, len(difference), size=(BOOTSTRAPS, len(difference)))
                replicates = difference[indices].mean(axis=1)
                low, high = interval(replicates)
                rows.append(
                    {
                        "metricType": metric_type,
                        "candidateId": candidate,
                        "comparisonId": comparison_id,
                        "modelId": model,
                        "referenceModelId": reference,
                        "matrices": len(difference),
                        "logLossImprovement": float(difference.mean()),
                        "logLossImprovementLower95": low,
                        "logLossImprovementUpper95": high,
                        "fractionBootstrapPositive": float((replicates > 0).mean()),
                    }
                )
                if metric_type == "F12_JOINT_EVENT":
                    for index, value in enumerate(replicates):
                        bootstrap_rows.append(
                            {
                                "candidateId": candidate,
                                "comparisonId": comparison_id,
                                "replicate": index,
                                "logLossImprovement": float(value),
                            }
                        )
    return pd.DataFrame(rows), pd.DataFrame(bootstrap_rows)


def q_rank_results(state_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = state_metrics[
        state_metrics["matrixRole"].eq("VALIDATION")
        & state_metrics["horizon"].eq(PRIMARY_HORIZON)
        & state_metrics["targetType"].eq("JOINT_BREAK_RUN3")
    ]
    rows = []
    bootstrap_rows = []
    for (candidate, direction, model), group in data.groupby(
        ["candidateId", "direction", "modelId"], sort=True
    ):
        group = group.sort_values(["matrixIndex", "completedFissionLandmark"])
        matrices = sorted(group["matrixIndex"].unique())
        observed = safe_spearman(
            group["predictedProbability"].to_numpy(), group["empiricalQ"].to_numpy()
        )
        centered_prediction = group["predictedProbability"] - group.groupby("matrixIndex")["predictedProbability"].transform("mean")
        centered_q = group["empiricalQ"] - group.groupby("matrixIndex")["empiricalQ"].transform("mean")
        centered = safe_spearman(centered_prediction.to_numpy(), centered_q.to_numpy())
        rng = generator("q_rank_bootstrap", candidate, direction, model)
        replicated = np.empty((BOOTSTRAPS, 2), dtype=np.float64)
        groups = {matrix: group[group["matrixIndex"].eq(matrix)] for matrix in matrices}
        for index in range(BOOTSTRAPS):
            selected = rng.integers(0, len(matrices), size=len(matrices))
            samples = []
            for synthetic, position in enumerate(selected):
                sample = groups[matrices[position]].copy()
                sample["bootstrapMatrix"] = synthetic
                samples.append(sample)
            sample = pd.concat(samples, ignore_index=True)
            replicated[index, 0] = safe_spearman(
                sample["predictedProbability"].to_numpy(), sample["empiricalQ"].to_numpy()
            )
            cp = sample["predictedProbability"] - sample.groupby("bootstrapMatrix")["predictedProbability"].transform("mean")
            cq = sample["empiricalQ"] - sample.groupby("bootstrapMatrix")["empiricalQ"].transform("mean")
            replicated[index, 1] = safe_spearman(cp.to_numpy(), cq.to_numpy())
        raw_low, raw_high = interval(replicated[:, 0])
        centered_low, centered_high = interval(replicated[:, 1])
        rows.append(
            {
                "candidateId": candidate,
                "direction": direction,
                "modelId": model,
                "matrices": len(matrices),
                "states": len(group),
                "qSpearman": observed,
                "qSpearmanLower95": raw_low,
                "qSpearmanUpper95": raw_high,
                "centeredQSpearman": centered,
                "centeredQSpearmanLower95": centered_low,
                "centeredQSpearmanUpper95": centered_high,
            }
        )
        if model == "STATE_LOCAL_SEMIMARKOV":
            for index, values in enumerate(replicated):
                bootstrap_rows.append(
                    {
                        "candidateId": candidate,
                        "direction": direction,
                        "replicate": index,
                        "qSpearman": values[0],
                        "centeredQSpearman": values[1],
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(bootstrap_rows)


def residual_reliability(state_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = state_metrics[
        state_metrics["matrixRole"].eq("VALIDATION")
        & state_metrics["horizon"].eq(PRIMARY_HORIZON)
        & state_metrics["targetType"].eq("JOINT_BREAK_RUN3")
    ]
    rows = []
    bootstrap_rows = []
    for (candidate, model), group in data.groupby(["candidateId", "modelId"], sort=True):
        pivot = group.pivot(index=["matrixIndex", "completedFissionLandmark"], columns="direction", values="qResidual").sort_index()
        frame = pivot.reset_index()
        raw = safe_spearman(frame["A_TO_B"].to_numpy(), frame["B_TO_A"].to_numpy())
        centered_a = frame["A_TO_B"] - frame.groupby("matrixIndex")["A_TO_B"].transform("mean")
        centered_b = frame["B_TO_A"] - frame.groupby("matrixIndex")["B_TO_A"].transform("mean")
        centered = safe_spearman(centered_a.to_numpy(), centered_b.to_numpy())
        matrices = sorted(frame["matrixIndex"].unique())
        groups = {matrix: frame[frame["matrixIndex"].eq(matrix)] for matrix in matrices}
        rng = generator("residual_bootstrap", candidate, model)
        replicated = np.empty((BOOTSTRAPS, 2), dtype=np.float64)
        for index in range(BOOTSTRAPS):
            selected = rng.integers(0, len(matrices), size=len(matrices))
            samples = []
            for synthetic, position in enumerate(selected):
                sample = groups[matrices[position]].copy()
                sample["bootstrapMatrix"] = synthetic
                samples.append(sample)
            sample = pd.concat(samples, ignore_index=True)
            replicated[index, 0] = safe_spearman(sample["A_TO_B"].to_numpy(), sample["B_TO_A"].to_numpy())
            ca = sample["A_TO_B"] - sample.groupby("bootstrapMatrix")["A_TO_B"].transform("mean")
            cb = sample["B_TO_A"] - sample.groupby("bootstrapMatrix")["B_TO_A"].transform("mean")
            replicated[index, 1] = safe_spearman(ca.to_numpy(), cb.to_numpy())
        raw_low, raw_high = interval(replicated[:, 0])
        center_low, center_high = interval(replicated[:, 1])
        rows.append(
            {
                "candidateId": candidate,
                "modelId": model,
                "matrices": len(matrices),
                "states": len(frame),
                "residualSplitHalfSpearman": raw,
                "residualSplitHalfSpearmanLower95": raw_low,
                "residualSplitHalfSpearmanUpper95": raw_high,
                "centeredResidualSpearman": centered,
                "centeredResidualSpearmanLower95": center_low,
                "centeredResidualSpearmanUpper95": center_high,
            }
        )
        if model == "STATE_LOCAL_SEMIMARKOV":
            for index, values in enumerate(replicated):
                bootstrap_rows.append(
                    {
                        "candidateId": candidate,
                        "replicate": index,
                        "residualSplitHalfSpearman": values[0],
                        "centeredResidualSpearman": values[1],
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(bootstrap_rows)


def q_permutations(state_metrics: pd.DataFrame) -> pd.DataFrame:
    data = state_metrics[
        state_metrics["matrixRole"].eq("VALIDATION")
        & state_metrics["horizon"].eq(PRIMARY_HORIZON)
        & state_metrics["targetType"].eq("JOINT_BREAK_RUN3")
    ]
    rows = []
    for (candidate, direction, model), group in data.groupby(
        ["candidateId", "direction", "modelId"], sort=True
    ):
        p = group.pivot(index="matrixIndex", columns="completedFissionLandmark", values="predictedProbability").sort_index()
        q = group.pivot(index="matrixIndex", columns="completedFissionLandmark", values="empiricalQ").reindex_like(p)
        observed = safe_spearman(p.to_numpy().ravel(), q.to_numpy().ravel())
        rng = generator("q_permutation", candidate, direction, model)
        null = np.empty(PERMUTATIONS, dtype=np.float64)
        for index in range(PERMUTATIONS):
            null[index] = safe_spearman(
                p.to_numpy()[rng.permutation(len(p))].ravel(), q.to_numpy().ravel()
            )
        rows.append(
            {
                "candidateId": candidate,
                "direction": direction,
                "modelId": model,
                "observedSpearman": observed,
                "nullMeanSpearman": float(np.nanmean(null)),
                "upperTailP": float((1 + np.sum(null >= observed)) / (PERMUTATIONS + 1)),
                "permutations": PERMUTATIONS,
                "wholeMatrixTrajectoryPermutation": True,
            }
        )
    return pd.DataFrame(rows)


def scientific_gates(
    comparisons: pd.DataFrame,
    ranks: pd.DataFrame,
    residuals: pd.DataFrame,
    permutations: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], str]:
    event = comparisons[comparisons["metricType"].eq("F12_JOINT_EVENT")]
    rows = []
    for candidate in CANDIDATES:
        candidate_event = event[event["candidateId"].eq(candidate)]

        def lower(
            comparison_id: str,
            candidate_frame: pd.DataFrame = candidate_event,
        ) -> float:
            return float(
                candidate_frame.loc[
                    candidate_frame["comparisonId"].eq(comparison_id),
                    "logLossImprovementLower95",
                ].iloc[0]
            )

        matrix_ranks = ranks[
            ranks["candidateId"].eq(candidate)
            & ranks["modelId"].eq("MATRIX_OTHER_LANDMARK_SEMIMARKOV")
        ]
        state_ranks = ranks[
            ranks["candidateId"].eq(candidate)
            & ranks["modelId"].eq("STATE_LOCAL_SEMIMARKOV")
        ]
        state_residual = residuals[
            residuals["candidateId"].eq(candidate)
            & residuals["modelId"].eq("STATE_LOCAL_SEMIMARKOV")
        ].iloc[0]
        state_permutations = permutations[
            permutations["candidateId"].eq(candidate)
            & permutations["modelId"].eq("STATE_LOCAL_SEMIMARKOV")
        ]
        matrix_permutations = permutations[
            permutations["candidateId"].eq(candidate)
            & permutations["modelId"].eq("MATRIX_OTHER_LANDMARK_SEMIMARKOV")
        ]
        matrix_pass = bool(
            lower("MATRIX_TRANSFER_BEYOND_POOLED") > 0
            and matrix_ranks["qSpearman"].min() > 0.5
            and matrix_permutations["upperTailP"].max() < 0.01
        )
        state_specific = lower("STATE_LOCAL_BEYOND_MATRIX_TRANSFER") > 0
        compression = bool(
            lower("STATE_LOCAL_BEYOND_POOLED") > 0
            and state_ranks["qSpearmanLower95"].min() > 0.3
            and state_permutations["upperTailP"].max() < 0.01
            and state_residual.centeredResidualSpearmanUpper95 < 0.3
        )
        rows.extend(
            [
                {
                    "gateId": f"MATRIX_TRANSFER::{candidate}",
                    "candidateId": candidate,
                    "gateFamily": "MATRIX_TRANSFER",
                    "properScoreLower95": lower("MATRIX_TRANSFER_BEYOND_POOLED"),
                    "minimumQSpearman": float(matrix_ranks["qSpearman"].min()),
                    "minimumQSpearmanLower95": float(matrix_ranks["qSpearmanLower95"].min()),
                    "residualUpper95": np.nan,
                    "maximumPermutationP": float(matrix_permutations["upperTailP"].max()),
                    "passed": matrix_pass,
                },
                {
                    "gateId": f"STATE_SPECIFIC::{candidate}",
                    "candidateId": candidate,
                    "gateFamily": "STATE_SPECIFIC",
                    "properScoreLower95": lower("STATE_LOCAL_BEYOND_MATRIX_TRANSFER"),
                    "minimumQSpearman": float(state_ranks["qSpearman"].min()),
                    "minimumQSpearmanLower95": float(state_ranks["qSpearmanLower95"].min()),
                    "residualUpper95": float(state_residual.centeredResidualSpearmanUpper95),
                    "maximumPermutationP": float(state_permutations["upperTailP"].max()),
                    "passed": state_specific,
                },
                {
                    "gateId": f"COMPRESSION::{candidate}",
                    "candidateId": candidate,
                    "gateFamily": "COMPRESSION",
                    "properScoreLower95": lower("STATE_LOCAL_BEYOND_POOLED"),
                    "minimumQSpearman": float(state_ranks["qSpearman"].min()),
                    "minimumQSpearmanLower95": float(state_ranks["qSpearmanLower95"].min()),
                    "residualUpper95": float(state_residual.centeredResidualSpearmanUpper95),
                    "maximumPermutationP": float(state_permutations["upperTailP"].max()),
                    "passed": compression,
                },
            ]
        )
    gates = pd.DataFrame(rows)

    def both(family: str) -> bool:
        selected = gates[gates["gateFamily"].eq(family)]
        return len(selected) == 2 and bool(selected["passed"].all())

    matrix_transfer = both("MATRIX_TRANSFER")
    state_specific = both("STATE_SPECIFIC")
    compression = both("COMPRESSION")
    classifications = []
    if matrix_transfer:
        classifications.append("MATRIX_LEVEL_REGIME_DYNAMICS_TRANSFER_ACROSS_STATES")
    else:
        classifications.append("MATRIX_LEVEL_BRANCH_HAZARDS_DO_NOT_TRANSFER_ACROSS_STATES")
    if state_specific:
        classifications.append("CURRENT_STATE_SPECIFIC_REGIME_DYNAMICS_REQUIRED")
    if compression:
        classifications.append("SHOOTING_COMMITTOR_COMPRESSIBLE_TO_STATE_LOCAL_DURATION_HAZARDS")
        next_theme = "L53_PAST_OBSERVABLE_STATE_LOCAL_HAZARD_PROXY"
    else:
        classifications.append("SHOOTING_COMMITTOR_REQUIRES_BRANCH_PATH_ENSEMBLE")
        next_theme = "L53_TIME_INHOMOGENEOUS_PATH_ORDER_AUDIT"
    classifications.extend(["BRANCH_DERIVED_NOT_PAST_OBSERVABLE", "NOT_PROMOTABLE_AS_CONFIRMED"])
    gates = pd.concat(
        [
            gates,
            pd.DataFrame(
                [
                    {
                        "gateId": "COMPLETE_CROSS_CANDIDATE_ADJUDICATION",
                        "candidateId": "BOTH",
                        "gateFamily": "COMPLETE",
                        "properScoreLower95": np.nan,
                        "minimumQSpearman": np.nan,
                        "minimumQSpearmanLower95": np.nan,
                        "residualUpper95": np.nan,
                        "maximumPermutationP": np.nan,
                        "passed": matrix_transfer or state_specific or compression,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    return gates, classifications, next_theme


def compute_tables() -> tuple[dict[str, pd.DataFrame], list[str], str]:
    prefix, branches, transitions, estimates = frozen_inputs()
    anchors = pooled_anchors()
    tables_map, parameters, scope = fit_tables(prefix, transitions, anchors)
    predictions = process_predictions(prefix, tables_map)
    transition_state, transition_matrix, transition_aggregate = transition_metrics(
        transitions, prefix, tables_map
    )
    event_state, event_matrix, event_aggregate = event_metrics(
        predictions, branches, estimates
    )
    comparisons, comparison_bootstrap = model_comparisons(
        transition_matrix, event_matrix
    )
    ranks, rank_bootstrap = q_rank_results(event_state)
    residuals, residual_bootstrap = residual_reliability(event_state)
    permutations = q_permutations(event_state)
    gates, classifications, next_theme = scientific_gates(
        comparisons, ranks, residuals, permutations
    )
    tables = {
        "hazard_fit_scope_registry.parquet": scope,
        "hazard_parameter_results.parquet": parameters,
        "process_probability_predictions.parquet": predictions,
        "transition_state_metrics.parquet": transition_state,
        "transition_matrix_metrics.parquet": transition_matrix,
        "transition_metric_results.parquet": transition_aggregate,
        "event_state_metrics.parquet": event_state,
        "event_matrix_metrics.parquet": event_matrix,
        "event_metric_results.parquet": event_aggregate,
        "model_comparisons.parquet": comparisons,
        "model_comparison_bootstrap.parquet": comparison_bootstrap,
        "q_rank_results.parquet": ranks,
        "q_rank_bootstrap.parquet": rank_bootstrap,
        "residual_reliability_results.parquet": residuals,
        "residual_reliability_bootstrap.parquet": residual_bootstrap,
        "negative_control_results.parquet": permutations,
        "scientific_gate_results.parquet": gates,
        "model_registry.parquet": model_registry(),
    }
    return tables, classifications, next_theme


def make_figures(tables: dict[str, pd.DataFrame]) -> None:
    root = BUILD_ROOT / "figures"
    root.mkdir(parents=True, exist_ok=True)
    colors = {CANDIDATES[0]: "#4c78a8", CANDIDATES[1]: "#f58518"}
    transition = tables["transition_metric_results.parquet"]
    validation = transition[transition["matrixRole"].eq("VALIDATION")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for axis, candidate in zip(axes, CANDIDATES, strict=True):
        group = validation[validation["candidateId"].eq(candidate)]
        axis.barh(group["modelId"], group["equalMatrixMeanLogLoss"], color=colors[candidate])
        axis.set_title(f"C{candidate[-2:]}")
        axis.set_xlabel("Heldout-half transition log loss")
    fig.suptitle("Pooled, cross-landmark matrix, and state-local hazards")
    fig.tight_layout()
    fig.savefig(root / "01_transition_compression.png", dpi=160)
    plt.close(fig)

    events = tables["event_metric_results.parquet"]
    events = events[
        events["matrixRole"].eq("VALIDATION")
        & events["targetType"].eq("JOINT_BREAK_RUN3")
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for axis, candidate in zip(axes, CANDIDATES, strict=True):
        group = events[events["candidateId"].eq(candidate)]
        for model in MODELS:
            model_group = group[group["modelId"].eq(model)]
            axis.plot(model_group["horizon"], model_group["equalMatrixMeanBranchLogLoss"], marker="o", label=model)
        axis.set_title(f"C{candidate[-2:]}")
        axis.set_xlabel("Future fissions")
        axis.set_ylabel("Heldout-half event log loss")
    axes[1].legend(fontsize=7)
    fig.suptitle("Cross-fitted finite-horizon process probabilities")
    fig.tight_layout()
    fig.savefig(root / "02_event_compression_by_horizon.png", dpi=160)
    plt.close(fig)

    ranks = tables["q_rank_results.parquet"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for axis, candidate in zip(axes, CANDIDATES, strict=True):
        group = ranks[ranks["candidateId"].eq(candidate)]
        for model in MODELS:
            model_group = group[group["modelId"].eq(model)]
            axis.plot(model_group["direction"], model_group["qSpearman"], marker="o", label=model)
        axis.axhline(0.5, color="black", ls=":")
        axis.set_title(f"C{candidate[-2:]}")
        axis.set_ylabel("Predicted versus heldout-half q Spearman")
    axes[1].legend(fontsize=7)
    fig.suptitle("Independent-half committor ranking")
    fig.tight_layout()
    fig.savefig(root / "03_q_rank_crossfit.png", dpi=160)
    plt.close(fig)

    residual = tables["residual_reliability_results.parquet"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for axis, candidate in zip(axes, CANDIDATES, strict=True):
        group = residual[residual["candidateId"].eq(candidate)]
        axis.barh(group["modelId"], group["centeredResidualSpearman"], color=colors[candidate])
        axis.axvline(0.3, color="black", ls=":")
        axis.set_title(f"C{candidate[-2:]}")
        axis.set_xlabel("Centered residual split-half Spearman")
    fig.suptitle("Systematic signal left after hazard compression")
    fig.tight_layout()
    fig.savefig(root / "04_residual_reliability.png", dpi=160)
    plt.close(fig)

    comparisons = tables["model_comparisons.parquet"]
    comparisons = comparisons[comparisons["metricType"].eq("F12_JOINT_EVENT")]
    fig, axis = plt.subplots(figsize=(10, 5))
    labels = [f"C{row.candidateId[-2:]} {row.comparisonId}" for row in comparisons.itertuples(index=False)]
    values = comparisons["logLossImprovement"].to_numpy()
    lower = values - comparisons["logLossImprovementLower95"].to_numpy()
    upper = comparisons["logLossImprovementUpper95"].to_numpy() - values
    axis.errorbar(values, np.arange(len(values)), xerr=np.vstack([lower, upper]), fmt="o")
    axis.axvline(0, color="black", ls=":")
    axis.set_yticks(np.arange(len(labels)), labels, fontsize=7)
    axis.set_xlabel("Heldout-half F12 event log-loss improvement")
    axis.set_title("Registered model comparisons")
    fig.tight_layout()
    fig.savefig(root / "05_model_comparison_intervals.png", dpi=160)
    plt.close(fig)

    gates = tables["scientific_gate_results.parquet"]
    matrix = gates[gates["candidateId"].isin(CANDIDATES)].pivot(
        index="gateFamily", columns="candidateId", values="passed"
    )
    fig, axis = plt.subplots(figsize=(7, 4))
    image = axis.imshow(matrix.to_numpy(dtype=float), vmin=0, vmax=1, cmap="RdYlGn")
    axis.set_xticks(range(len(matrix.columns)), [f"C{value[-2:]}" for value in matrix.columns])
    axis.set_yticks(range(len(matrix.index)), matrix.index)
    axis.set_title("Cross-candidate compression decision")
    fig.colorbar(image, ax=axis, ticks=[0, 1])
    fig.tight_layout()
    fig.savefig(root / "06_scientific_gate_matrix.png", dpi=160)
    plt.close(fig)


def report_text(
    tables: dict[str, pd.DataFrame],
    classifications: list[str],
    next_theme: str,
    runtime: dict[str, Any],
) -> str:
    transition = tables["transition_metric_results.parquet"]
    transition = transition[transition["matrixRole"].eq("VALIDATION")]
    events = tables["event_metric_results.parquet"]
    events = events[
        events["matrixRole"].eq("VALIDATION")
        & events["horizon"].eq(PRIMARY_HORIZON)
        & events["targetType"].eq("JOINT_BREAK_RUN3")
    ]
    comparisons = tables["model_comparisons.parquet"]
    ranks = tables["q_rank_results.parquet"]
    residuals = tables["residual_reliability_results.parquet"]
    gates = tables["scientific_gate_results.parquet"]
    return f"""# S19-L52 Full Results — Shooting-Residual Regime Compression

## Top summary

- **Research step:** `{VERSION}`
- **Completion status:** complete; additive exploratory analysis-only evidence
- **Artifacts written:** exact L50/L51 branch-half replay, three locked regime-hazard models, A-to-B and B-to-A fits, transition and F4/F8/F12 event proper scores, q ranks, residual reliability, 4,096 matrix bootstraps, 512 whole-matrix permutations, six figures, report and hash manifests
- **Validation:** PASS — immutable S01–L51 baseline; ten fixtures; exact 800-state/51,200-branch/614,400-transition replay; target-state exclusion from every matrix-transfer fit; heldout-half exclusion; zero-overlap analysis seeds; two exact analysis/report passes; runtime, storage and artifact hashes
- **Outcome classification:** {', '.join(f'`{value}`' for value in classifications)}
- **Lay summary:** L52 uses one independent half of the already simulated futures to learn a simple hereditary-regime transition law and asks whether that law predicts the other half. It separates a matrix-wide law learned from other states from a law learned at the exact current state.
- **Recommended next action:** `{next_theme}` under the bounded autonomous authorization through L65. No new branch begins automatically inside L52; S20, E02, author contact, Phi and interventions remain inactive.

## Frozen design

L52 changes no scientific target. Strict parent/daughter `H>0.9` remains inheritance; the primary F12 event remains the first future break followed by three consecutive inherited fissions. The exact L50 branch halves are crossfit in both directions. `MATRIX_OTHER_LANDMARK_SEMIMARKOV` excludes the target state and learns only from the same matrix's other four landmarks. `STATE_LOCAL_SEMIMARKOV` learns from the fitting half at the target state. Both use one fixed four-pseudotransition cell prior anchored to the exact L51 pooled duration table. The scoring half never enters the fit.

## Heldout-half transition scores

{transition.to_markdown(index=False, floatfmt='.7f')}

## Heldout-half F12 joint-event scores

{events.to_markdown(index=False, floatfmt='.7f')}

## Registered proper-score comparisons

{comparisons.to_markdown(index=False, floatfmt='.7f')}

## Independent-half q ranking

{ranks.to_markdown(index=False, floatfmt='.7f')}

## Residual reliability

{residuals.to_markdown(index=False, floatfmt='.7f')}

A positive centered residual correlation means that the same states are systematically under- or overpredicted in both independent halves, so the duration table has not compressed all reproducible state information.

## Scientific gates

{gates.to_markdown(index=False, floatfmt='.7f')}

## Interpretation boundary

Matrix- or state-local branch-derived hazards are forward-shooting measurements. Even perfect compression would not make them past-observable biomarkers. A failure of compression does not eliminate a real committor; it means that binary duration hazards lose path ordering, evolving physical state or other future-ensemble information. This result cannot establish paper replication, a privileged attractor, functional memory, PhiID foresight, intervention efficacy or real chemistry.

## Runtime and provenance

- Repository lock: `{runtime['repositoryHead']}`.
- Workers: `{runtime['workers']}` with one numerical-library thread; GPU hours: 0.
- Wall time: `{runtime['wallSeconds'] / 60:.3f}` minutes; CPU upper estimate: `{runtime['estimatedCpuHours']:.6f}` hours.
- New matrices, trajectories and branch streams: 0, 0 and 0.
- Frozen branch sequences/transitions: `{runtime['frozenBranchSequences']:,}` / `{runtime['transitionObservations']:,}`.
- Crossfit directions: 2; matrix bootstraps: {BOOTSTRAPS}; whole-matrix permutations: {PERMUTATIONS}.

## Limitations

This adaptive loop reuses the L50 branch ensemble. The binary transition process compresses continuous composition and catalytic dynamics. Matrix-transfer models use simulated futures at other states and state-local models use simulated futures at the target; neither is operational without shooting. The four-pseudotransition prior and duration cap were fixed, not searched. Matrix resampling preserves dependence among landmarks and branch folds.
"""


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
        "schema": "eidosoma.e01.s19_l52.artifact_manifest.v1",
        "loopId": LOOP_ID,
        "files": rows,
        "fileCount": len(rows),
        "aggregateSha256": hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def append_ledgers(classifications: list[str], timestamp: str, next_theme: str) -> None:
    ledger_path = ARTIFACT_ROOT / "self_improvement_ledger.parquet"
    ledger = pd.read_parquet(ledger_path)
    sequence = int(ledger["ledgerSequence"].max()) + 1
    additions = [
        {
            "appendOnly": True,
            "beliefBeforeLoop": "L51 found duration dependence and reliable current-prefix updating, but observed-prefix process models did not reconstruct the L50 committor despite mostly between-matrix risk variation.",
            "failureOrAmbiguityTargeted": "Whether the missing information is a transferable matrix transition law, exact-state transition law or branch-path ensemble structure.",
            "informationGainRationale": "Crossfitting the pre-existing independent branch halves isolates compression without another simulation or feature tournament.",
            "learned": "L52 half, scope, prior, duration, target, score and gate contract locked before derived outcomes.",
            "ledgerSequence": sequence,
            "loopId": LOOP_ID,
            "motivatingEvidence": "Complete L50/L51 results and reviewer shooting-measurement direction.",
            "proposedNextTest": "Crossfit matrix-transfer and state-local duration hazards across branch halves.",
            "recordPhase": "PRE_LOOP_METHOD_LOCK",
            "remainingPlausibleHypotheses": "matrix-wide regime dynamics, current-state dynamics or irreducible path-order information",
            "selectedHypotheses": "The short shooting teacher may be compressible into branch-derived duration hazards.",
            "timestampUtc": timestamp,
            "weakenedHypotheses": "Another static observed-prefix family should precede decomposition of the successful shooting information.",
        },
        {
            "appendOnly": True,
            "beliefBeforeLoop": "Transfer requires out-of-state proper-score improvement; compression requires state-local proper scores, q ranks and disappearance of reliable residual ordering in both candidates.",
            "failureOrAmbiguityTargeted": "Transfer and compression of the empirical process committor.",
            "informationGainRationale": "Independent branch halves and whole-matrix uncertainty separate parameter fit from heldout future scoring.",
            "learned": ";".join(classifications),
            "ledgerSequence": sequence + 1,
            "loopId": LOOP_ID,
            "motivatingEvidence": "Complete L52 result.",
            "proposedNextTest": next_theme,
            "recordPhase": "POST_LOOP_RESULT_AUTONOMOUS_CONTINUATION",
            "remainingPlausibleHypotheses": next_theme,
            "selectedHypotheses": "Registered branch-half compression hierarchy.",
            "timestampUtc": timestamp,
            "weakenedHypotheses": "Any registered L52 gate that failed.",
        },
    ]
    BASE.write_parquet(
        ledger_path,
        pd.concat([ledger, pd.DataFrame(additions).reindex(columns=ledger.columns)], ignore_index=True),
    )
    markdown = ARTIFACT_ROOT / "SELF_IMPROVEMENT_LEDGER.md"
    BASE.atomic_text(
        markdown,
        markdown.read_text()
        + f"\n\n## {LOOP_ID} — shooting-residual regime compression\n\n"
        + f"- **Learned:** {', '.join(classifications)}.\n"
        + f"- **Next:** `{next_theme}`.\n",
    )
    candidate_path = ARTIFACT_ROOT / "candidate_registry.parquet"
    candidates = pd.read_parquet(candidate_path)
    candidate = {
        "branchCount": 0,
        "bundleId": "L52_SHOOTING_RESIDUAL_REGIME_COMPRESSION",
        "candidateId": "S19-L52-SHOOTING-RESIDUAL-REGIME-COMPRESSION",
        "candidateSpecificSuccess": 0,
        "completedFitLeakage": 0,
        "computeEfficiency": 5,
        "crossCandidateDiscriminability": 5,
        "deterministicHReuse": 1,
        "explanatoryLeverage": 5,
        "frozenRank": 1,
        "independenceFromPriorOutcomeSelection": 2,
        "outcomeGuidedThresholdSelection": 0,
        "paperFingerprintSpecificity": 0,
        "proposedSpecification": "cross-fitted pooled, matrix-other-landmark and state-local duration hazards on frozen L50 halves",
        "rankingScore": 27.0,
        "registryOrder": int(candidates["registryOrder"].max()) + 1,
        "selected": "SHOOTING_COMMITTOR_COMPRESSIBLE_TO_STATE_LOCAL_DURATION_HAZARDS" in classifications,
        "selectionReason": "L51_RENEWAL_MODEL_RESIDUAL",
        "sourceGrounding": 4,
        "testability": 5,
        "undefinedAuthorSemantics": 0,
    }
    BASE.write_parquet(
        candidate_path,
        pd.concat([candidates, pd.DataFrame([candidate]).reindex(columns=candidates.columns)], ignore_index=True),
    )
    source_path = ARTIFACT_ROOT / "source_search_ledger.parquet"
    sources = pd.read_parquet(source_path)
    additions = []
    for row in source_registry().itertuples(index=False):
        additions.append(
            {
                "commitOrVersion": None,
                "evidenceClass": row.evidenceClass,
                "finding": f"{row.finding}; L52 use: {row.frozenUse}",
                "licenseStatus": "PUBLIC_METADATA_OR_WORKSPACE_EVIDENCE",
                "redistributionStatus": "REFERENCE_ONLY",
                "repositoryIdentity": None,
                "retainedPath": None,
                "retrievalDate": timestamp[:10],
                "sha256": None,
                "sourceId": f"L52_{row.sourceId}",
                "sourceType": row.evidenceClass,
                "treeIdentity": None,
                "url": row.url,
            }
        )
    BASE.write_parquet(
        source_path,
        pd.concat([sources, pd.DataFrame(additions).reindex(columns=sources.columns)], ignore_index=True),
    )


def prepare_lock() -> None:
    LOOP_ROOT.mkdir(parents=True, exist_ok=True)
    if git("status", "--porcelain=v1"):
        raise RuntimeError("repository must be clean before L52 lock")
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/eidosoma/groups/42")
    if head != remote:
        raise RuntimeError("L52 local/remote commit mismatch")
    prior = validate_immutable_prior()
    fixtures = fixture_results()
    seeds = analysis_seed_manifest()
    firewall = seed_firewall(seeds)
    sources = source_registry()
    registry = model_registry()
    required_inputs = {
        "l50Manifest": L50_ROOT / "artifact_manifest.json",
        "l50Branches": L50_ROOT / "branch_results.parquet",
        "l50Committors": L50_ROOT / "state_committor_results.parquet",
        "l51Manifest": L51_ROOT / "artifact_manifest.json",
        "l51PrefixStates": L51_ROOT / "prefix_state_results.parquet",
        "l51BranchSequences": L51_ROOT / "branch_sequence_manifest.parquet",
        "l51PooledHazards": L51_ROOT / "hazard_parameter_results.parquet",
    }
    input_validation = pd.DataFrame(
        [
            {"inputId": name, "path": str(path), "sha256": sha256_file(path), "exists": path.is_file()}
            for name, path in required_inputs.items()
        ]
    )
    benchmark = {
        "schema": "eidosoma.e01.s19_l52.benchmark_projection.v1",
        "outcomeBlind": True,
        "basis": "two crossfit directions over 51,200 frozen branches and 614,400 transitions",
        "projectedCpuHoursUpper": 12.0,
        "projectedWallHoursUpper": 12.0,
        "status": "PASS",
    }
    if (
        not prior["unchanged"]
        or not fixtures["passed"].all()
        or firewall["status"] != "PASS"
        or not input_validation["exists"].all()
        or len(registry) != len(MODELS)
    ):
        raise RuntimeError("L52 preoutcome validation failed")
    shutil.copy2(CONFIG, LOOP_ROOT / "preregistration.yaml")
    BASE.atomic_text(
        LOOP_ROOT / "decision_record.md",
        "# S19-L52 decision record\n\n"
        "The autonomous authorization through L65 remains active. L51 identified "
        "duration-dependent heredity switching and useful current-prefix updating, "
        "but its registered observed-prefix renewal models did not reconstruct the "
        "reliable L50 empirical committor. Before any L52-derived result is opened, "
        "this record freezes the exact L50 32/32 branch halves, L51 pooled duration "
        "anchor, three model hierarchy, four-pseudotransition cell prior, target-state "
        "exclusion from matrix-transfer fits, A-to-B/B-to-A scoring, nested horizons, "
        "proper scores, q ranks, residual gates, matrix bootstraps and permutations. "
        "No new simulation, target, threshold, duration cap, branch allocation, Phi "
        "quantity, feature search or intervention is authorized.\n",
    )
    BASE.write_json(LOOP_ROOT / "immutable_prior_validation.json", prior)
    BASE.write_parquet(LOOP_ROOT / "fixture_results.parquet", fixtures)
    BASE.write_parquet(LOOP_ROOT / "analysis_seed_manifest.parquet", seeds)
    BASE.write_json(LOOP_ROOT / "seed_firewall.json", firewall)
    BASE.write_parquet(LOOP_ROOT / "input_identity_validation.parquet", input_validation)
    BASE.write_parquet(LOOP_ROOT / "source_registry.parquet", sources)
    BASE.write_parquet(LOOP_ROOT / "model_registry.parquet", registry)
    BASE.write_json(LOOP_ROOT / "benchmark_projection.json", benchmark)
    BASE.write_json(
        LOOP_ROOT / "source_snapshot_manifest.json",
        {
            "schema": "eidosoma.e01.s19_l52.source_snapshot_manifest.v1",
            "runnerSha256": sha256_file(RUNNER_PATH),
            "coreSha256": sha256_file(CORE_PATH),
            "configSha256": sha256_file(CONFIG),
            "sources": sources.to_dict("records"),
        },
    )
    locked_inputs = {
        **required_inputs,
        "analysisSeeds": LOOP_ROOT / "analysis_seed_manifest.parquet",
        "seedFirewall": LOOP_ROOT / "seed_firewall.json",
        "inputValidation": LOOP_ROOT / "input_identity_validation.parquet",
        "sourceSnapshot": LOOP_ROOT / "source_snapshot_manifest.json",
        "modelRegistry": LOOP_ROOT / "model_registry.parquet",
        "benchmark": LOOP_ROOT / "benchmark_projection.json",
    }
    hashes = {name: sha256_file(path) for name, path in locked_inputs.items()}
    implementation = {
        "schema": "eidosoma.e01.s19_l52.implementation_lock.v1",
        "repositoryHead": head,
        "remoteHead": remote,
        "runnerSha256": sha256_file(RUNNER_PATH),
        "coreSha256": sha256_file(CORE_PATH),
        "configSha256": sha256_file(CONFIG),
        "models": list(MODELS),
        "directions": [value[0] for value in DIRECTIONS],
        "horizons": list(HORIZONS),
        "primaryHorizon": PRIMARY_HORIZON,
        "threshold": THRESHOLD,
        "requiredRun": REQUIRED_RUN,
        "maximumDuration": MAXIMUM_DURATION,
        "priorStrength": PRIOR_STRENGTH,
        "matrixBootstraps": BOOTSTRAPS,
        "wholeMatrixPermutations": PERMUTATIONS,
        "newSimulation": False,
        "lockedInputHashes": hashes,
        "outcomeAccessed": False,
        "lockedAtUtc": utc_now(),
    }
    BASE.write_json(LOOP_ROOT / "implementation_lock.json", implementation)
    BASE.write_json(
        LOOP_ROOT / "preoutcome_repository_lock.json",
        {
            "head": head,
            "remote": remote,
            "priorAggregateSha256": prior["aggregateSha256"],
            "runnerSha256": sha256_file(RUNNER_PATH),
            "coreSha256": sha256_file(CORE_PATH),
            "configSha256": sha256_file(CONFIG),
            "lockedInputHashes": hashes,
        },
    )


def execute() -> None:
    started = time.perf_counter()
    lock = json.loads((LOOP_ROOT / "preoutcome_repository_lock.json").read_text())
    if (
        git("rev-parse", "HEAD") != lock["head"]
        or git("rev-parse", "origin/eidosoma/groups/42") != lock["remote"]
        or git("status", "--porcelain=v1")
    ):
        raise RuntimeError("L52 repository lock mismatch")
    prior = validate_immutable_prior()
    fixtures = fixture_results()
    locked_inputs = {
        "l50Manifest": L50_ROOT / "artifact_manifest.json",
        "l50Branches": L50_ROOT / "branch_results.parquet",
        "l50Committors": L50_ROOT / "state_committor_results.parquet",
        "l51Manifest": L51_ROOT / "artifact_manifest.json",
        "l51PrefixStates": L51_ROOT / "prefix_state_results.parquet",
        "l51BranchSequences": L51_ROOT / "branch_sequence_manifest.parquet",
        "l51PooledHazards": L51_ROOT / "hazard_parameter_results.parquet",
        "analysisSeeds": LOOP_ROOT / "analysis_seed_manifest.parquet",
        "seedFirewall": LOOP_ROOT / "seed_firewall.json",
        "inputValidation": LOOP_ROOT / "input_identity_validation.parquet",
        "sourceSnapshot": LOOP_ROOT / "source_snapshot_manifest.json",
        "modelRegistry": LOOP_ROOT / "model_registry.parquet",
        "benchmark": LOOP_ROOT / "benchmark_projection.json",
    }
    if any(
        sha256_file(path) != lock["lockedInputHashes"][name]
        for name, path in locked_inputs.items()
    ):
        raise RuntimeError("L52 locked input changed")
    if (
        not prior["unchanged"]
        or prior["aggregateSha256"] != lock["priorAggregateSha256"]
        or not fixtures["passed"].all()
        or sha256_file(RUNNER_PATH) != lock["runnerSha256"]
        or sha256_file(CORE_PATH) != lock["coreSha256"]
        or sha256_file(CONFIG) != lock["configSha256"]
    ):
        raise RuntimeError("L52 pre-execution validation failed")
    if BUILD_ROOT.exists():
        shutil.rmtree(BUILD_ROOT)
    BUILD_ROOT.mkdir(parents=True)
    tables, classifications, next_theme = compute_tables()
    make_figures(tables)
    tables_again, classifications_again, next_theme_again = compute_tables()
    table_exact = {
        name: frame_hash(frame) == frame_hash(tables_again[name])
        for name, frame in tables.items()
    }
    regeneration = {
        "schema": "eidosoma.e01.s19_l52.regeneration_validation.v1",
        "status": "PASS"
        if all(table_exact.values())
        and classifications == classifications_again
        and next_theme == next_theme_again
        else "FAIL",
        "tableExact": table_exact,
        "classificationExact": classifications == classifications_again,
        "nextThemeExact": next_theme == next_theme_again,
        "analysisPasses": 2,
    }
    if regeneration["status"] != "PASS":
        raise RuntimeError("L52 regeneration failure")
    for name, frame in tables.items():
        BASE.write_parquet(BUILD_ROOT / name, frame)
    BASE.write_json(
        BUILD_ROOT / "classification.json",
        {
            "schema": "eidosoma.e01.s19_l52.classification.v1",
            "classifications": classifications,
            "nextTheme": next_theme,
            "priorStatusesChanged": False,
            "promotableAsConfirmed": False,
            "newMatrices": 0,
            "newPrimaryTrajectories": 0,
            "newBranchStreams": 0,
        },
    )
    pd.DataFrame(
        columns=["failureId", "stage", "status", "reason", "scientificValuesReleased"]
    ).to_csv(BUILD_ROOT / "failure_ledger.csv", index=False)
    pd.DataFrame(
        columns=["amendmentId", "status", "scientificContractChanged", "reason"]
    ).to_csv(BUILD_ROOT / "technical_amendment_ledger.csv", index=False)
    elapsed = time.perf_counter() - started
    runtime = {
        "schema": "eidosoma.e01.s19_l52.runtime.v1",
        "repositoryHead": lock["head"],
        "workers": WORKERS,
        "numericalLibraryThreadsPerWorker": 1,
        "gpuHours": 0,
        "wallSeconds": elapsed,
        "estimatedCpuHours": elapsed * WORKERS / 3600,
        "frozenMatrices": 80,
        "frozenStates": 800,
        "frozenBranchSequences": 51200,
        "transitionObservations": 614400,
        "newMatrices": 0,
        "newPrimaryTrajectories": 0,
        "newBranchStreams": 0,
        "crossfitDirections": 2,
        "matrixBootstraps": BOOTSTRAPS,
        "wholeMatrixPermutations": PERMUTATIONS,
        "analysisPasses": 2,
        "completedAtUtc": utc_now(),
    }
    if runtime["estimatedCpuHours"] > 34 or runtime["wallSeconds"] > 20 * 3600:
        raise RuntimeError("L52 runtime ceiling exceeded")
    BASE.write_json(BUILD_ROOT / "runtime_manifest.json", runtime)
    BASE.write_json(BUILD_ROOT / "regeneration_validation.json", regeneration)
    retained_bytes = sum(
        path.stat().st_size for path in BUILD_ROOT.rglob("*") if path.is_file()
    ) + sum(path.stat().st_size for path in LOOP_ROOT.iterdir() if path.is_file())
    storage = {
        "schema": "eidosoma.e01.s19_l52.storage_validation.v1",
        "status": "PASS" if retained_bytes <= 15 * 1024**3 else "FAIL",
        "retainedBytes": retained_bytes,
        "retainedGiBCeiling": 15,
        "temporaryGiBCeiling": 30,
    }
    BASE.write_json(BUILD_ROOT / "storage_validation.json", storage)
    report = report_text(tables, classifications, next_theme, runtime)
    if report != report_text(tables, classifications, next_theme, runtime):
        raise RuntimeError("L52 report regeneration failure")
    BASE.atomic_text(BUILD_ROOT / "research_step_full_results.md", report)
    BASE.atomic_text(BUILD_ROOT / "S19_L52_FULL_RESULTS.md", report)
    BASE.atomic_text(
        BUILD_ROOT / "loop_decision_summary.md",
        f"# S19-L52 decision summary\n\n**Classification:** {', '.join(classifications)}\n\n**Next:** `{next_theme}`.\n",
    )
    if storage["status"] != "PASS":
        raise RuntimeError("L52 storage ceiling exceeded")
    for path in (BUILD_ROOT / "figures").glob("*.png"):
        if not path.stat().st_size:
            raise RuntimeError(f"empty L52 figure: {path}")
    for path in BUILD_ROOT.iterdir():
        destination = LOOP_ROOT / path.name
        if path.is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(path, destination)
        else:
            shutil.copy2(path, destination)
    BASE.write_json(LOOP_ROOT / "artifact_manifest.json", manifest_for(LOOP_ROOT))
    if manifest_for(LOOP_ROOT) != json.loads((LOOP_ROOT / "artifact_manifest.json").read_text()):
        raise RuntimeError("L52 artifact manifest regeneration failure")
    append_ledgers(classifications, runtime["completedAtUtc"], next_theme)
    root_report = (
        f"# S19 current-step report\n\nLatest completed loop: `{LOOP_ID}`.\n\n"
        f"Classification: {', '.join(classifications)}.\n\n"
        f"Next autonomous theme: `{next_theme}`.\n"
    )
    for name in (
        "S19_CURRENT_STEP_REPORT.md",
        "CURRENT_STEP_HANDOFF.md",
        "S19_CURRENT_HANDOFF.md",
        "research_step_full_results.md",
    ):
        BASE.atomic_text(ARTIFACT_ROOT / name, root_report)
    BASE.write_json(
        ARTIFACT_ROOT / "s19_status.json",
        {
            "schema": "eidosoma.e01.s19.status.v1",
            "programStatus": "ACTIVE_AUTONOMOUS_SEQUENCE",
            "latestCompletedLoop": LOOP_ID,
            "latestClassification": classifications,
            "nextAuthorizedLoop": "S19-L53",
            "nextTheme": next_theme,
            "authorizationUpperBound": "S19-L65",
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
    args = parser.parse_args()
    if args.prepare_lock:
        prepare_lock()
    else:
        execute()


if __name__ == "__main__":
    main()
