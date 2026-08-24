#!/usr/bin/env python3
"""Run S19-L51 heredity-regime hazard and renewal decomposition."""

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
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from e01_onset_discovery.regime_hazard import (
    finite_horizon_process_probability,
    fit_iid,
    fit_markov,
    fit_semimarkov,
    posterior_matrix_markov,
    trailing_run_length,
    transition_rows,
    transport_duration_effect,
)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


L50 = load_module(
    "e01_l51_l50_runner",
    ROOT / "scripts/e01/run_s19_l50_fission_aligned_process_committor.py",
)
BASE = L50.BASE

ARTIFACT_ROOT = Path("/artifacts/research_steps/S19")
LOOP_ROOT = ARTIFACT_ROOT / "loops/L51"
L23_ROOT = ARTIFACT_ROOT / "loops/L23"
L44_ROOT = ARTIFACT_ROOT / "loops/L44"
L50_ROOT = ARTIFACT_ROOT / "loops/L50"
BUILD_ROOT = Path("/cache/e01_s19_l51/build")
CONFIG = ROOT / "configs/e01/s19_l51_regime_hazard_renewal.yaml"
RUNNER_PATH = Path(__file__).resolve()
CORE_PATH = ROOT / "src/e01_onset_discovery/regime_hazard.py"

LOOP_ID = "S19-L51"
VERSION = "E01-S19-L51-HEREDITY-REGIME-HAZARD-RENEWAL-DECOMPOSITION-v1.0.0"
CANDIDATES = ("S12F-CANDIDATE-02", "S12F-CANDIDATE-03")
ROLES = ("DEVELOPMENT", "VALIDATION")
HORIZONS = (4, 8, 12)
PRIMARY_HORIZON = 12
LANDMARKS = (20, 35, 50, 65, 80)
THRESHOLD = 0.9
REQUIRED_RUN = 3
MAXIMUM_DURATION = 12
PRIOR_STRENGTH = 1.0
BOOTSTRAPS = 4096
PERMUTATIONS = 512
WORKERS = 1
MODELS = (
    "POOLED_IID",
    "POOLED_MARKOV",
    "POOLED_SEMIMARKOV_DURATION",
    "EARLY_PREFIX_MATRIX_MARKOV",
    "EARLY_PREFIX_MATRIX_SEMIMARKOV",
    "CURRENT_PREFIX_MATRIX_SEMIMARKOV",
)
SEED_ROOT = bytes.fromhex(
    "419ea7d9e245aec34463677124295d50a90b783cfed52f650264f88f9349cf18"
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def sha256_file(path: Path) -> str:
    return L50.sha256_file(path)


def frame_hash(frame: pd.DataFrame) -> str:
    return L50.frame_hash(frame)


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
    inherited = L50.validate_immutable_prior()
    manifest = json.loads((L50_ROOT / "artifact_manifest.json").read_text())
    rows = []
    for row in manifest["files"]:
        path = L50_ROOT / row["path"]
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
        "schema": "eidosoma.e01.s19_l51.immutable_prior_validation.v1",
        "status": "PASS" if passed else "FAIL",
        "unchanged": passed,
        "priorThroughL49RUnchanged": bool(inherited["unchanged"]),
        "validatedL50ArtifactCount": len(rows),
        "aggregateSha256": hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "rows": rows,
    }


def source_registry() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sourceId": "L50_FISSION_ALIGNED_RESULT",
                "evidenceClass": "DIRECT_FROZEN_E01_RESULT",
                "finding": "Fission-aligned joint process probability was reproducible, but shooting did not improve heldout Brier beyond direct history in both candidates.",
                "frozenUse": "decompose the same frozen process probability without generating new outcomes",
                "url": None,
            },
            {
                "sourceId": "L44_DWELL_HAZARD_RESULT",
                "evidenceClass": "DIRECT_FROZEN_E01_RESULT",
                "finding": "Inherited-episode break hazard declined with elapsed dwell, while first-order Markov behavior improved over IID in every prior cohort.",
                "frozenUse": "prospectively distinguish first-order state dependence from duration dependence",
                "url": None,
            },
            {
                "sourceId": "REVIEWER_REGIME_SWITCHING_DIRECTION",
                "evidenceClass": "HUMAN_REVIEW_DIRECTION",
                "finding": "Treat plastic heredity as switching between hereditary and nonhereditary regimes; separate break, conditional resumption and joint risk, and test IID, Markov, semi-Markov, renewal and matrix propensity.",
                "frozenUse": "registered L51 question and interpretation boundary",
                "url": None,
            },
            {
                "sourceId": "COLE_LEE_WHITMORE_ZASLAVSKY_1995",
                "evidenceClass": "PRIMARY_METHOD_SOURCE",
                "finding": "Empirical-Bayes analysis of heterogeneous two-state Markov chains models individual transition probabilities with beta marginals.",
                "frozenUse": "shrink matrix-prefix transition probabilities to pooled candidate-specific probabilities",
                "url": "https://doi.org/10.1080/01621459.1995.10476641",
            },
            {
                "sourceId": "CARAVENNA_GIACOMIN_ZAMBOTTI_2022",
                "evidenceClass": "PRIMARY_METHOD_SOURCE",
                "finding": "Renewal binary sequences are governed by waiting-time distributions; Markov behavior is a restricted special case.",
                "frozenUse": "interpret dwell-dependent transition probability as a renewal/semi-Markov alternative",
                "url": "https://doi.org/10.1007/s10955-022-02893-8",
            },
            {
                "sourceId": "FUH_FAN_1997",
                "evidenceClass": "PRIMARY_METHOD_SOURCE",
                "finding": "Matrix-beta Bayesian bootstrap methods support uncertainty analysis for finite-state Markov transition and hitting-time quantities.",
                "frozenUse": "matrix-level resampling of transition and finite-horizon probability comparisons",
                "url": "https://www3.stat.sinica.edu.tw/statistica/j7n4/j7n413/j7n413.htm",
            },
        ]
    )


def fixture_results() -> pd.DataFrame:
    constant = finite_horizon_process_probability(
        lambda _state, _duration: 0.75,
        initial_state=True,
        initial_duration=5,
        horizon=8,
    )
    constant_again = finite_horizon_process_probability(
        lambda _state, _duration: 0.75,
        initial_state=True,
        initial_duration=5,
        horizon=8,
    )
    current = [False, False, True, True]
    following = [False, True, True, False]
    markov = fit_markov(current, following)
    semi = fit_semimarkov(current, [1, 2, 1, 2], following, markov)
    matrix = posterior_matrix_markov([False, True, True, True, False], markov)
    transported = transport_duration_effect(matrix, markov, semi)
    rows = [
        {"fixtureId": "F01_TRAILING_RUN", "passed": trailing_run_length([True, False, False]) == 2},
        {"fixtureId": "F02_TRANSITION_ROWS", "passed": len(transition_rows(True, 3, [True, False])) == 2},
        {"fixtureId": "F03_IID_SMOOTHING", "passed": fit_iid([True, False, True]) == 0.625},
        {"fixtureId": "F04_MARKOV_SHAPE", "passed": markov.shape == (2,)},
        {"fixtureId": "F05_SEMIMARKOV_SHAPE", "passed": semi.shape == (2, MAXIMUM_DURATION)},
        {"fixtureId": "F06_MATRIX_POSTERIOR_FINITE", "passed": bool(np.isfinite(matrix).all())},
        {"fixtureId": "F07_DURATION_TRANSPORT_FINITE", "passed": bool(np.isfinite(transported).all())},
        {"fixtureId": "F08_PROCESS_ORDER", "passed": 0 <= constant.joint_break_run_probability <= constant.break_probability <= 1},
        {"fixtureId": "F09_EXACT_PROCESS_REPLAY", "passed": constant == constant_again},
        {"fixtureId": "F10_SCOPE", "passed": HORIZONS == (4, 8, 12) and MODELS[-1] == "CURRENT_PREFIX_MATRIX_SEMIMARKOV"},
        {"fixtureId": "F11_SEED_REPLAY", "passed": derived_seed("fixture", np.int64(2)) == derived_seed("fixture", 2)},
    ]
    return pd.DataFrame(rows)


def analysis_seed_manifest() -> pd.DataFrame:
    rows = []
    for purpose, repetitions in (
        ("matrix_bootstrap", BOOTSTRAPS),
        ("whole_matrix_probability_permutation", PERMUTATIONS),
        ("duration_alignment_permutation", PERMUTATIONS),
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
    prior_material = set(
        pd.read_parquet(L50_ROOT / "analysis_seed_manifest.parquet")["seedMaterialSha256"]
    )
    branches = pd.read_parquet(L50_ROOT / "branch_seed_manifest.parquet")
    for column in branches.columns:
        if column.endswith("SeedMaterialSha256"):
            prior_material.update(branches[column].dropna().astype(str))
    current = set(seeds["seedMaterialSha256"].astype(str))
    passed = len(current) == len(seeds) and not (current & prior_material)
    return {
        "schema": "eidosoma.e01.s19_l51.seed_firewall.v1",
        "status": "PASS" if passed else "FAIL",
        "newAnalysisStreams": len(current),
        "overlapCount": len(current & prior_material),
        "rootHex": SEED_ROOT.hex(),
    }


def model_registry() -> pd.DataFrame:
    descriptions = {
        "POOLED_IID": "one development-pooled inheritance probability",
        "POOLED_MARKOV": "development-pooled next-inheritance probability conditional on current binary regime",
        "POOLED_SEMIMARKOV_DURATION": "development-pooled current-regime and capped-dwell hazard with one pooled-prior pseudo-transition",
        "EARLY_PREFIX_MATRIX_MARKOV": "matrix-specific transition probabilities estimated only through fission 20 and shrunk to pooled Markov values",
        "EARLY_PREFIX_MATRIX_SEMIMARKOV": "fixed fission-20 matrix propensity plus pooled duration odds ratios",
        "CURRENT_PREFIX_MATRIX_SEMIMARKOV": "current-landmark matrix propensity plus pooled duration odds ratios",
    }
    return pd.DataFrame(
        [
            {
                "modelId": model,
                "description": descriptions[model],
                "developmentFitOnly": model.startswith("POOLED"),
                "usesValidationOutcome": False,
                "usesMatrixPrefix": "MATRIX" in model,
                "usesDuration": "SEMIMARKOV" in model,
                "earlyPrefixFrozenAtFission": 20 if model.startswith("EARLY") else None,
                "hyperparameterSearch": False,
            }
            for model in MODELS
        ]
    )


def _boundary_sequence(candidate: str, matrix: int, manifest_index: pd.DataFrame) -> tuple[np.ndarray, Any]:
    source = manifest_index.loc[(candidate, matrix)]
    trajectory = L50.L28.load_trajectory(source)
    selected = tuple(L50.L28.selected_clock_observations(trajectory, L50.L28.CLOCK_ID))
    indices = [i for i, obs in enumerate(selected) if obs.observation_kind == "post_fission"]
    scores = np.asarray([L50._boundary_h(selected, i) for i in indices], dtype=np.float64)
    if len(scores) != 100 or trajectory.trajectory_sha256 != source.trajectorySha256:
        raise RuntimeError("L51 frozen trajectory or boundary identity failure")
    return scores > THRESHOLD, source


def build_prefix_states(states: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_parquet(L23_ROOT / "input_trajectory_manifest.parquet")
    manifest_index = manifest.set_index(["candidateId", "matrixIndex"])
    rows = []
    validations = []
    for (candidate, matrix), group in states.groupby(["candidateId", "matrixIndex"], sort=True):
        inherited, source = _boundary_sequence(candidate, int(matrix), manifest_index)
        early = inherited[:20]
        for state in group.sort_values("completedFissionLandmark").itertuples(index=False):
            generation = int(state.completedFissionLandmark)
            prefix = inherited[:generation]
            current_state = bool(prefix[-1])
            current_duration = trailing_run_length(prefix)
            row = {
                "stateId": state.stateId,
                "matrixRole": state.matrixRole,
                "candidateId": candidate,
                "matrixIndex": int(matrix),
                "completedFissionLandmark": generation,
                "earlyPrefixSequence": "".join("1" if value else "0" for value in early),
                "currentPrefixSequence": "".join("1" if value else "0" for value in prefix),
                "currentInheritanceState": current_state,
                "currentRegimeDuration": current_duration,
                "earlyInheritanceFraction": float(early.mean()),
                "currentInheritanceFraction": float(prefix.mean()),
                "targetUsesCompletedTestTrajectory": False,
            }
            rows.append(row)
            validations.append(
                {
                    "stateId": state.stateId,
                    "trajectoryIdentityPassed": source.trajectorySha256 == state.trajectorySha256,
                    "cacheIdentityPassed": sha256_file(Path(source.cachePath)) == source.cacheSha256,
                    "prefixLengthPassed": len(prefix) == generation,
                    "currentStatePassed": current_state == (float(state.latestParentDaughterH) > THRESHOLD),
                    "trailingDurationPassed": current_duration == (int(state.prefixTrailingInheritanceRun) if current_state else 0 or current_duration),
                    "prefixFractionPassed": abs(float(prefix.mean()) - float(state.prefixInheritanceFraction)) <= 1e-15,
                }
            )
    prefix = pd.DataFrame(rows).sort_values(
        ["matrixRole", "candidateId", "matrixIndex", "completedFissionLandmark"]
    ).reset_index(drop=True)
    validation = pd.DataFrame(validations).sort_values("stateId").reset_index(drop=True)
    # L50 stores only the positive trailing run.  For noninherited states L51
    # independently reconstructs the negative dwell from the frozen prefix.
    validation.loc[
        ~prefix.set_index("stateId").loc[validation.stateId, "currentInheritanceState"].to_numpy(),
        "trailingDurationPassed",
    ] = True
    checks = [column for column in validation if column.endswith("Passed")]
    if len(prefix) != 800 or not validation[checks].all().all():
        raise RuntimeError("L51 prefix-state replay failure")
    return prefix, validation


def build_branch_sequences(
    branches: pd.DataFrame, prefix: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prefix_index = prefix.set_index("stateId")
    sequence_rows = []
    transition_records = []
    h_columns = [f"parentDaughterH{index:02d}" for index in range(1, 13)]
    for branch in branches.itertuples(index=False):
        initial = prefix_index.loc[branch.stateId]
        future = tuple(float(getattr(branch, column)) > THRESHOLD for column in h_columns)
        sequence_rows.append(
            {
                "stateId": branch.stateId,
                "matrixRole": branch.matrixRole,
                "candidateId": branch.candidateId,
                "matrixIndex": int(branch.matrixIndex),
                "completedFissionLandmark": int(branch.completedFissionLandmark),
                "branchIndex": int(branch.branchIndex),
                "branchIdentitySha256": branch.branchIdentitySha256,
                "inheritanceSequence": "".join("1" if value else "0" for value in future),
                "strictThreshold": THRESHOLD,
            }
        )
        for step, (current, duration, following) in enumerate(
            transition_rows(
                bool(initial.currentInheritanceState),
                int(initial.currentRegimeDuration),
                future,
            ),
            start=1,
        ):
            transition_records.append(
                {
                    "stateId": branch.stateId,
                    "matrixRole": branch.matrixRole,
                    "candidateId": branch.candidateId,
                    "matrixIndex": int(branch.matrixIndex),
                    "completedFissionLandmark": int(branch.completedFissionLandmark),
                    "branchIndex": int(branch.branchIndex),
                    "futureStep": step,
                    "currentState": current,
                    "currentDuration": duration,
                    "nextState": following,
                }
            )
    sequences = pd.DataFrame(sequence_rows).sort_values(
        ["matrixRole", "candidateId", "matrixIndex", "completedFissionLandmark", "branchIndex"]
    ).reset_index(drop=True)
    transitions = pd.DataFrame(transition_records).sort_values(
        ["matrixRole", "candidateId", "matrixIndex", "completedFissionLandmark", "branchIndex", "futureStep"]
    ).reset_index(drop=True)
    if len(sequences) != 51200 or len(transitions) != 614400:
        raise RuntimeError("L51 branch sequence scope failure")
    return sequences, transitions


def fit_pooled_models(transitions: pd.DataFrame) -> tuple[dict[str, dict[str, np.ndarray | float]], pd.DataFrame]:
    fitted: dict[str, dict[str, np.ndarray | float]] = {}
    rows = []
    development = transitions[transitions["matrixRole"].eq("DEVELOPMENT")]
    for candidate in CANDIDATES:
        group = development[development["candidateId"].eq(candidate)]
        iid = fit_iid(group["nextState"])
        markov = fit_markov(group["currentState"], group["nextState"])
        semi = fit_semimarkov(
            group["currentState"],
            group["currentDuration"],
            group["nextState"],
            markov,
            maximum_duration=MAXIMUM_DURATION,
            prior_strength=PRIOR_STRENGTH,
        )
        fitted[candidate] = {"iid": iid, "markov": markov, "semi": semi}
        rows.append(
            {
                "candidateId": candidate,
                "modelId": "POOLED_IID",
                "currentState": None,
                "duration": None,
                "probabilityNextInherited": iid,
                "fitTransitions": len(group),
            }
        )
        for state in (0, 1):
            rows.append(
                {
                    "candidateId": candidate,
                    "modelId": "POOLED_MARKOV",
                    "currentState": bool(state),
                    "duration": None,
                    "probabilityNextInherited": float(markov[state]),
                    "fitTransitions": int((group["currentState"] == bool(state)).sum()),
                }
            )
            for duration in range(1, MAXIMUM_DURATION + 1):
                rows.append(
                    {
                        "candidateId": candidate,
                        "modelId": "POOLED_SEMIMARKOV_DURATION",
                        "currentState": bool(state),
                        "duration": duration,
                        "probabilityNextInherited": float(semi[state, duration - 1]),
                        "fitTransitions": int(
                            (
                                (group["currentState"] == bool(state))
                                & np.minimum(group["currentDuration"], MAXIMUM_DURATION).eq(duration)
                            ).sum()
                        ),
                    }
                )
    return fitted, pd.DataFrame(rows)


def _decode(sequence: str) -> np.ndarray:
    return np.asarray([value == "1" for value in sequence], dtype=np.bool_)


def state_model_tables(
    prefix: pd.DataFrame, fitted: dict[str, dict[str, np.ndarray | float]]
) -> tuple[dict[str, dict[str, np.ndarray]], pd.DataFrame]:
    output: dict[str, dict[str, np.ndarray]] = {}
    rows = []
    for row in prefix.itertuples(index=False):
        pooled_markov = np.asarray(fitted[row.candidateId]["markov"], dtype=np.float64)
        pooled_semi = np.asarray(fitted[row.candidateId]["semi"], dtype=np.float64)
        early_markov = posterior_matrix_markov(
            _decode(row.earlyPrefixSequence), pooled_markov, prior_strength=PRIOR_STRENGTH
        )
        current_markov = posterior_matrix_markov(
            _decode(row.currentPrefixSequence), pooled_markov, prior_strength=PRIOR_STRENGTH
        )
        early_semi = transport_duration_effect(early_markov, pooled_markov, pooled_semi)
        current_semi = transport_duration_effect(current_markov, pooled_markov, pooled_semi)
        iid = float(fitted[row.candidateId]["iid"])
        tables = {
            "POOLED_IID": np.full((2, MAXIMUM_DURATION), iid),
            "POOLED_MARKOV": np.tile(pooled_markov[:, None], (1, MAXIMUM_DURATION)),
            "POOLED_SEMIMARKOV_DURATION": pooled_semi,
            "EARLY_PREFIX_MATRIX_MARKOV": np.tile(early_markov[:, None], (1, MAXIMUM_DURATION)),
            "EARLY_PREFIX_MATRIX_SEMIMARKOV": early_semi,
            "CURRENT_PREFIX_MATRIX_SEMIMARKOV": current_semi,
        }
        output[row.stateId] = tables
        equilibrium_early = float(
            early_markov[0] / (early_markov[0] + 1.0 - early_markov[1])
        )
        equilibrium_current = float(
            current_markov[0] / (current_markov[0] + 1.0 - current_markov[1])
        )
        rows.append(
            {
                "stateId": row.stateId,
                "matrixRole": row.matrixRole,
                "candidateId": row.candidateId,
                "matrixIndex": int(row.matrixIndex),
                "completedFissionLandmark": int(row.completedFissionLandmark),
                "earlyPInheritedAfterNoninheritance": float(early_markov[0]),
                "earlyPInheritedAfterInheritance": float(early_markov[1]),
                "currentPInheritedAfterNoninheritance": float(current_markov[0]),
                "currentPInheritedAfterInheritance": float(current_markov[1]),
                "earlyEquilibriumInheritance": equilibrium_early,
                "currentEquilibriumInheritance": equilibrium_current,
                "currentState": bool(row.currentInheritanceState),
                "currentRegimeDuration": int(row.currentRegimeDuration),
                "usesOnlyObservedPrefix": True,
            }
        )
    return output, pd.DataFrame(rows).sort_values("stateId").reset_index(drop=True)


def transition_metrics(
    transitions: pd.DataFrame, model_tables: dict[str, dict[str, np.ndarray]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    accumulators: dict[tuple[str, str, int, str], list[float]] = {}
    for state_id, group in transitions.groupby("stateId", sort=False):
        tables = model_tables[state_id]
        current = group["currentState"].to_numpy(dtype=np.int64)
        duration = np.minimum(
            group["currentDuration"].to_numpy(dtype=np.int64), MAXIMUM_DURATION
        ) - 1
        target = group["nextState"].to_numpy(dtype=np.float64)
        base = group.iloc[0]
        for model in MODELS:
            probability = np.clip(tables[model][current, duration], 1e-12, 1 - 1e-12)
            losses = -(target * np.log(probability) + (1 - target) * np.log1p(-probability))
            briers = (target - probability) ** 2
            key = (
                str(base.matrixRole),
                str(base.candidateId),
                int(base.matrixIndex),
                model,
            )
            values = accumulators.setdefault(key, [0.0, 0.0, 0.0])
            values[0] += float(losses.sum())
            values[1] += float(briers.sum())
            values[2] += len(target)
    rows = []
    for (role, candidate, matrix, model), (loss_sum, brier_sum, count) in accumulators.items():
        rows.append(
            {
                "matrixRole": role,
                "candidateId": candidate,
                "matrixIndex": matrix,
                "modelId": model,
                "transitions": int(count),
                "logLoss": loss_sum / count,
                "brier": brier_sum / count,
            }
        )
    per_matrix = pd.DataFrame(rows).sort_values(
        ["matrixRole", "candidateId", "modelId", "matrixIndex"]
    ).reset_index(drop=True)
    aggregate = (
        per_matrix.groupby(["matrixRole", "candidateId", "modelId"], as_index=False)
        .agg(
            matrices=("matrixIndex", "nunique"),
            transitions=("transitions", "sum"),
            equalMatrixMeanLogLoss=("logLoss", "mean"),
            equalMatrixMeanBrier=("brier", "mean"),
            matrixSdLogLoss=("logLoss", "std"),
        )
        .sort_values(["matrixRole", "candidateId", "equalMatrixMeanLogLoss"])
        .reset_index(drop=True)
    )
    return per_matrix, aggregate


TRANSITION_COMPARISONS = (
    ("POOLED_SEMIMARKOV_DURATION", "POOLED_MARKOV", "DURATION_BEYOND_MARKOV"),
    ("EARLY_PREFIX_MATRIX_MARKOV", "POOLED_MARKOV", "EARLY_MATRIX_BEYOND_POOLED"),
    ("EARLY_PREFIX_MATRIX_SEMIMARKOV", "POOLED_SEMIMARKOV_DURATION", "STABLE_MATRIX_BEYOND_DURATION"),
    ("CURRENT_PREFIX_MATRIX_SEMIMARKOV", "EARLY_PREFIX_MATRIX_SEMIMARKOV", "CURRENT_UPDATE_BEYOND_EARLY"),
)


def transition_comparisons(per_matrix: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    validation = per_matrix[per_matrix["matrixRole"].eq("VALIDATION")]
    rows = []
    bootstrap_rows = []
    for candidate in CANDIDATES:
        group = validation[validation["candidateId"].eq(candidate)]
        pivot = group.pivot(index="matrixIndex", columns="modelId", values="logLoss").sort_index()
        for model, reference, comparison_id in TRANSITION_COMPARISONS:
            difference = (pivot[reference] - pivot[model]).to_numpy(dtype=np.float64)
            rng = generator("transition_bootstrap", candidate, comparison_id)
            indices = rng.integers(0, len(difference), size=(BOOTSTRAPS, len(difference)))
            replicates = difference[indices].mean(axis=1)
            low, high = interval(replicates)
            rows.append(
                {
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


def process_predictions(
    prefix: pd.DataFrame, model_tables: dict[str, dict[str, np.ndarray]]
) -> pd.DataFrame:
    rows = []
    for state in prefix.itertuples(index=False):
        for model in MODELS:
            table = model_tables[state.stateId][model]

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
                            "stateId": state.stateId,
                            "matrixRole": state.matrixRole,
                            "candidateId": state.candidateId,
                            "matrixIndex": int(state.matrixIndex),
                            "completedFissionLandmark": int(state.completedFissionLandmark),
                            "horizon": horizon,
                            "targetType": target,
                            "modelId": model,
                            "predictedProbability": value,
                            "targetUsesCompletedTestTrajectory": False,
                        }
                    )
    return pd.DataFrame(rows).sort_values(
        ["matrixRole", "candidateId", "matrixIndex", "completedFissionLandmark", "horizon", "targetType", "modelId"]
    ).reset_index(drop=True)


def process_metrics(
    predictions: pd.DataFrame, estimates: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    truth = estimates[
        [
            "stateId",
            "horizon",
            "targetType",
            "successes",
            "trials",
            "q",
            "observedTarget",
        ]
    ]
    merged = predictions.merge(
        truth, on=["stateId", "horizon", "targetType"], validate="many_to_one"
    )
    merged = merged[merged["trials"].gt(0) & merged["predictedProbability"].notna()].copy()
    probability = np.clip(merged["predictedProbability"].to_numpy(dtype=np.float64), 1e-12, 1 - 1e-12)
    successes = merged["successes"].to_numpy(dtype=np.float64)
    trials = merged["trials"].to_numpy(dtype=np.float64)
    merged["branchLogLoss"] = -(
        successes * np.log(probability) + (trials - successes) * np.log1p(-probability)
    ) / trials
    merged["branchBrier"] = (
        successes * (1 - probability) ** 2 + (trials - successes) * probability**2
    ) / trials
    merged["qSquaredError"] = (merged["q"] - merged["predictedProbability"]) ** 2
    per_matrix = (
        merged.groupby(
            ["matrixRole", "candidateId", "matrixIndex", "horizon", "targetType", "modelId"],
            as_index=False,
        )
        .agg(
            states=("stateId", "size"),
            trials=("trials", "sum"),
            branchLogLoss=("branchLogLoss", "mean"),
            branchBrier=("branchBrier", "mean"),
            qRmse=("qSquaredError", lambda values: float(np.sqrt(np.mean(values)))),
        )
        .sort_values(["matrixRole", "candidateId", "horizon", "targetType", "modelId", "matrixIndex"])
        .reset_index(drop=True)
    )
    rows = []
    for keys, group in merged.groupby(
        ["matrixRole", "candidateId", "horizon", "targetType", "modelId"], sort=True
    ):
        role, candidate, horizon, target, model = keys
        matrix_group = per_matrix[
            per_matrix["matrixRole"].eq(role)
            & per_matrix["candidateId"].eq(candidate)
            & per_matrix["horizon"].eq(horizon)
            & per_matrix["targetType"].eq(target)
            & per_matrix["modelId"].eq(model)
        ]
        rows.append(
            {
                "matrixRole": role,
                "candidateId": candidate,
                "horizon": int(horizon),
                "targetType": target,
                "modelId": model,
                "matrices": int(group["matrixIndex"].nunique()),
                "states": len(group),
                "equalMatrixMeanBranchLogLoss": float(matrix_group["branchLogLoss"].mean()),
                "equalMatrixMeanBranchBrier": float(matrix_group["branchBrier"].mean()),
                "qRmse": float(np.sqrt(group["qSquaredError"].mean())),
                "qSpearman": safe_spearman(
                    group["predictedProbability"].to_numpy(), group["q"].to_numpy()
                ),
            }
        )
    return merged, per_matrix, pd.DataFrame(rows)


PROCESS_COMPARISONS = (
    ("POOLED_SEMIMARKOV_DURATION", "POOLED_MARKOV", "PROCESS_DURATION_BEYOND_MARKOV"),
    ("EARLY_PREFIX_MATRIX_SEMIMARKOV", "POOLED_SEMIMARKOV_DURATION", "PROCESS_STABLE_MATRIX_BEYOND_DURATION"),
    ("CURRENT_PREFIX_MATRIX_SEMIMARKOV", "EARLY_PREFIX_MATRIX_SEMIMARKOV", "PROCESS_CURRENT_BEYOND_EARLY"),
    ("CURRENT_PREFIX_MATRIX_SEMIMARKOV", "POOLED_MARKOV", "PROCESS_FULL_BEYOND_MARKOV"),
)


def process_comparisons(per_matrix: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    validation = per_matrix[per_matrix["matrixRole"].eq("VALIDATION")]
    rows = []
    bootstrap_rows = []
    for candidate in CANDIDATES:
        for horizon in HORIZONS:
            for target in ("BREAK", "JOINT_BREAK_RUN3", "RUN3_GIVEN_BREAK"):
                group = validation[
                    validation["candidateId"].eq(candidate)
                    & validation["horizon"].eq(horizon)
                    & validation["targetType"].eq(target)
                ]
                pivot = group.pivot(index="matrixIndex", columns="modelId", values="branchLogLoss").dropna().sort_index()
                for model, reference, comparison_id in PROCESS_COMPARISONS:
                    difference = (pivot[reference] - pivot[model]).to_numpy(dtype=np.float64)
                    rng = generator("process_bootstrap", candidate, horizon, target, comparison_id)
                    indices = rng.integers(0, len(difference), size=(BOOTSTRAPS, len(difference)))
                    replicates = difference[indices].mean(axis=1)
                    low, high = interval(replicates)
                    rows.append(
                        {
                            "candidateId": candidate,
                            "horizon": horizon,
                            "targetType": target,
                            "comparisonId": comparison_id,
                            "modelId": model,
                            "referenceModelId": reference,
                            "matrices": len(difference),
                            "branchLogLossImprovement": float(difference.mean()),
                            "branchLogLossImprovementLower95": low,
                            "branchLogLossImprovementUpper95": high,
                            "fractionBootstrapPositive": float((replicates > 0).mean()),
                        }
                    )
                    if horizon == PRIMARY_HORIZON and target == "JOINT_BREAK_RUN3":
                        for index, value in enumerate(replicates):
                            bootstrap_rows.append(
                                {
                                    "candidateId": candidate,
                                    "comparisonId": comparison_id,
                                    "replicate": index,
                                    "branchLogLossImprovement": float(value),
                                }
                            )
    return pd.DataFrame(rows), pd.DataFrame(bootstrap_rows)


def realized_future_metrics(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    validation = merged[
        merged["matrixRole"].eq("VALIDATION")
        & merged["targetType"].eq("JOINT_BREAK_RUN3")
        & merged["observedTarget"].notna()
    ]
    for (candidate, horizon, model), group in validation.groupby(
        ["candidateId", "horizon", "modelId"], sort=True
    ):
        y = group["observedTarget"].astype(bool).to_numpy(dtype=np.bool_)
        p = np.clip(group["predictedProbability"].to_numpy(dtype=np.float64), 1e-12, 1 - 1e-12)
        rows.append(
            {
                "candidateId": candidate,
                "horizon": int(horizon),
                "modelId": model,
                "states": len(group),
                "positiveRate": float(y.mean()),
                "brier": float(brier_score_loss(y, p)),
                "logLoss": float(log_loss(y, p, labels=[False, True])),
                "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan"),
                "auprc": float(average_precision_score(y, p)) if y.any() else float("nan"),
                "interpretation": "ONE_REALIZED_FUTURE_DIAGNOSTIC_NOT_COMMITTOR_TRUTH",
            }
        )
    return pd.DataFrame(rows)


def q_alignment_permutations(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    validation = merged[
        merged["matrixRole"].eq("VALIDATION")
        & merged["targetType"].eq("JOINT_BREAK_RUN3")
    ]
    for candidate in CANDIDATES:
        for horizon in HORIZONS:
            for model in (
                "POOLED_SEMIMARKOV_DURATION",
                "EARLY_PREFIX_MATRIX_SEMIMARKOV",
                "CURRENT_PREFIX_MATRIX_SEMIMARKOV",
            ):
                group = validation[
                    validation["candidateId"].eq(candidate)
                    & validation["horizon"].eq(horizon)
                    & validation["modelId"].eq(model)
                ]
                q = group.pivot(index="matrixIndex", columns="completedFissionLandmark", values="q").sort_index()
                p = group.pivot(index="matrixIndex", columns="completedFissionLandmark", values="predictedProbability").reindex_like(q)
                observed = safe_spearman(p.to_numpy().ravel(), q.to_numpy().ravel())
                rng = generator("q_alignment_permutation", candidate, horizon, model)
                null = np.empty(PERMUTATIONS, dtype=np.float64)
                for replicate in range(PERMUTATIONS):
                    order = rng.permutation(len(p))
                    null[replicate] = safe_spearman(p.to_numpy()[order].ravel(), q.to_numpy().ravel())
                rows.append(
                    {
                        "candidateId": candidate,
                        "horizon": horizon,
                        "modelId": model,
                        "observedSpearman": observed,
                        "nullMeanSpearman": float(np.nanmean(null)),
                        "upperTailP": float((1 + np.sum(null >= observed)) / (PERMUTATIONS + 1)),
                        "permutations": PERMUTATIONS,
                        "wholeMatrixTrajectoryPermutation": True,
                    }
                )
    return pd.DataFrame(rows)


def _variance_components(values: np.ndarray, noise: np.ndarray) -> tuple[float, float, float, float, float]:
    matrix_count, state_count = values.shape
    matrix_means = values.mean(axis=1)
    ms_between = state_count * float(np.var(matrix_means, ddof=1))
    ms_within = float(np.sum((values - matrix_means[:, None]) ** 2) / (matrix_count * (state_count - 1)))
    mean_noise = float(noise.mean())
    between = max((ms_between - ms_within) / state_count, 0.0)
    within = max(ms_within - mean_noise, 0.0)
    total = between + within
    fraction = between / total if total > 0 else float("nan")
    landmark_variance = float(np.var(values.mean(axis=0), ddof=0))
    return between, within, total, fraction, landmark_variance


def variance_decomposition(estimates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    validation = estimates[
        estimates["matrixRole"].eq("VALIDATION")
        & estimates["targetType"].isin(["BREAK", "JOINT_BREAK_RUN3"])
    ]
    rows = []
    bootstrap_rows = []
    for (candidate, horizon, target), group in validation.groupby(
        ["candidateId", "horizon", "targetType"], sort=True
    ):
        q = group.pivot(index="matrixIndex", columns="completedFissionLandmark", values="q").sort_index()
        noise = group.pivot(index="matrixIndex", columns="completedFissionLandmark", values="binomialNoise").reindex_like(q)
        if q.shape != (40, 5):
            raise RuntimeError("L51 variance decomposition requires balanced 40x5 states")
        components = _variance_components(q.to_numpy(), noise.to_numpy())
        rng = generator("variance_bootstrap", candidate, horizon, target)
        replicated = np.empty((BOOTSTRAPS, 5), dtype=np.float64)
        for index in range(BOOTSTRAPS):
            sample = rng.integers(0, len(q), size=len(q))
            replicated[index] = _variance_components(
                q.to_numpy()[sample], noise.to_numpy()[sample]
            )
        lows = np.nanquantile(replicated, 0.025, axis=0)
        highs = np.nanquantile(replicated, 0.975, axis=0)
        rows.append(
            {
                "candidateId": candidate,
                "horizon": int(horizon),
                "targetType": target,
                "matrices": len(q),
                "statesPerMatrix": q.shape[1],
                "betweenMatrixVariance": components[0],
                "betweenMatrixVarianceLower95": lows[0],
                "betweenMatrixVarianceUpper95": highs[0],
                "withinMatrixVariance": components[1],
                "withinMatrixVarianceLower95": lows[1],
                "withinMatrixVarianceUpper95": highs[1],
                "correctedTotalVariance": components[2],
                "betweenMatrixFraction": components[3],
                "betweenMatrixFractionLower95": lows[3],
                "betweenMatrixFractionUpper95": highs[3],
                "landmarkMeanVariance": components[4],
                "withinMatrixGenerationSpearman": safe_spearman(
                    np.tile(np.asarray(LANDMARKS, dtype=np.float64), len(q)),
                    (q.to_numpy() - q.to_numpy().mean(axis=1, keepdims=True)).ravel(),
                ),
            }
        )
        if horizon == PRIMARY_HORIZON:
            for index, values in enumerate(replicated):
                bootstrap_rows.append(
                    {
                        "candidateId": candidate,
                        "targetType": target,
                        "replicate": index,
                        "betweenMatrixVariance": values[0],
                        "withinMatrixVariance": values[1],
                        "correctedTotalVariance": values[2],
                        "betweenMatrixFraction": values[3],
                        "landmarkMeanVariance": values[4],
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(bootstrap_rows)


def matrix_propensity_summary(propensity: pd.DataFrame) -> pd.DataFrame:
    rows = []
    validation = propensity[propensity["matrixRole"].eq("VALIDATION")]
    for candidate in CANDIDATES:
        group = validation[validation["candidateId"].eq(candidate)]
        pivot = group.pivot(index="matrixIndex", columns="completedFissionLandmark", values="currentEquilibriumInheritance").sort_index()
        early = group.groupby("matrixIndex")["earlyEquilibriumInheritance"].first().reindex(pivot.index)
        rows.append(
            {
                "candidateId": candidate,
                "matrices": len(pivot),
                "earlyPropensityMean": float(early.mean()),
                "earlyPropensitySdBetweenMatrices": float(early.std(ddof=1)),
                "currentPropensitySdAllStates": float(group["currentEquilibriumInheritance"].std(ddof=1)),
                "earlyVsCurrentSpearmanAtF80": safe_spearman(
                    early.to_numpy(), pivot[80].to_numpy()
                ),
                "meanAbsoluteWithinMatrixUpdate": float(
                    np.abs(pivot.to_numpy() - early.to_numpy()[:, None]).mean()
                ),
                "meanWithinMatrixRange": float(
                    (pivot.max(axis=1) - pivot.min(axis=1)).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def duration_alignment_control(
    transitions: pd.DataFrame,
    fitted: dict[str, dict[str, np.ndarray | float]],
) -> pd.DataFrame:
    rows = []
    validation = transitions[transitions["matrixRole"].eq("VALIDATION")]
    for candidate in CANDIDATES:
        group = validation[validation["candidateId"].eq(candidate)].copy()
        current = group["currentState"].to_numpy(dtype=np.int64)
        duration = np.minimum(group["currentDuration"].to_numpy(dtype=np.int64), MAXIMUM_DURATION) - 1
        target = group["nextState"].to_numpy(dtype=np.float64)
        markov = np.asarray(fitted[candidate]["markov"], dtype=np.float64)
        semi = np.asarray(fitted[candidate]["semi"], dtype=np.float64)
        p_markov = np.clip(markov[current], 1e-12, 1 - 1e-12)
        p_semi = np.clip(semi[current, duration], 1e-12, 1 - 1e-12)
        markov_loss = -(target * np.log(p_markov) + (1 - target) * np.log1p(-p_markov))
        observed = float(
            np.mean(markov_loss + target * np.log(p_semi) + (1 - target) * np.log1p(-p_semi))
        )
        rng = generator("duration_alignment_permutation", candidate)
        null = np.empty(PERMUTATIONS, dtype=np.float64)
        for replicate in range(PERMUTATIONS):
            permuted = duration.copy()
            for state in (0, 1):
                positions = np.flatnonzero(current == state)
                permuted[positions] = permuted[rng.permutation(positions)]
            p_null = np.clip(semi[current, permuted], 1e-12, 1 - 1e-12)
            null[replicate] = float(
                np.mean(markov_loss + target * np.log(p_null) + (1 - target) * np.log1p(-p_null))
            )
        rows.append(
            {
                "candidateId": candidate,
                "controlId": "DURATION_PERMUTED_WITHIN_CURRENT_STATE",
                "observedLogLossImprovement": observed,
                "nullMeanImprovement": float(null.mean()),
                "upperTailP": float((1 + np.sum(null >= observed)) / (PERMUTATIONS + 1)),
                "permutations": PERMUTATIONS,
            }
        )
    return pd.DataFrame(rows)


def scientific_gates(
    transition_compare: pd.DataFrame,
    process_compare: pd.DataFrame,
    process_metric: pd.DataFrame,
    q_permutations: pd.DataFrame,
    variance: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], str]:
    rows = []
    for candidate in CANDIDATES:
        transition = transition_compare[transition_compare["candidateId"].eq(candidate)]
        duration_lower = float(
            transition.loc[transition["comparisonId"].eq("DURATION_BEYOND_MARKOV"), "logLossImprovementLower95"].iloc[0]
        )
        stable_lower = float(
            transition.loc[transition["comparisonId"].eq("STABLE_MATRIX_BEYOND_DURATION"), "logLossImprovementLower95"].iloc[0]
        )
        update_lower = float(
            transition.loc[transition["comparisonId"].eq("CURRENT_UPDATE_BEYOND_EARLY"), "logLossImprovementLower95"].iloc[0]
        )
        rows.extend(
            [
                {"gateId": f"DURATION::{candidate}", "candidateId": candidate, "gateFamily": "TRANSITION_DURATION", "lower95": duration_lower, "spearman": np.nan, "permutationP": np.nan, "passed": duration_lower > 0},
                {"gateId": f"STABLE_MATRIX::{candidate}", "candidateId": candidate, "gateFamily": "EARLY_MATRIX_PROPENSITY", "lower95": stable_lower, "spearman": np.nan, "permutationP": np.nan, "passed": stable_lower > 0},
                {"gateId": f"CURRENT_UPDATE::{candidate}", "candidateId": candidate, "gateFamily": "LONGITUDINAL_UPDATE", "lower95": update_lower, "spearman": np.nan, "permutationP": np.nan, "passed": update_lower > 0},
            ]
        )
        comparison = process_compare[
            process_compare["candidateId"].eq(candidate)
            & process_compare["horizon"].eq(PRIMARY_HORIZON)
            & process_compare["targetType"].eq("JOINT_BREAK_RUN3")
            & process_compare["comparisonId"].eq("PROCESS_FULL_BEYOND_MARKOV")
        ].iloc[0]
        metric = process_metric[
            process_metric["matrixRole"].eq("VALIDATION")
            & process_metric["candidateId"].eq(candidate)
            & process_metric["horizon"].eq(PRIMARY_HORIZON)
            & process_metric["targetType"].eq("JOINT_BREAK_RUN3")
            & process_metric["modelId"].eq("CURRENT_PREFIX_MATRIX_SEMIMARKOV")
        ].iloc[0]
        permutation = q_permutations[
            q_permutations["candidateId"].eq(candidate)
            & q_permutations["horizon"].eq(PRIMARY_HORIZON)
            & q_permutations["modelId"].eq("CURRENT_PREFIX_MATRIX_SEMIMARKOV")
        ].iloc[0]
        reconstruct = bool(
            comparison.branchLogLossImprovementLower95 > 0
            and metric.qSpearman >= 0.5
            and permutation.upperTailP < 0.01
        )
        rows.append(
            {
                "gateId": f"COMMITTOR_RECONSTRUCTION::{candidate}",
                "candidateId": candidate,
                "gateFamily": "F12_JOINT_PROCESS",
                "lower95": float(comparison.branchLogLossImprovementLower95),
                "spearman": float(metric.qSpearman),
                "permutationP": float(permutation.upperTailP),
                "passed": reconstruct,
            }
        )
        variance_row = variance[
            variance["candidateId"].eq(candidate)
            & variance["horizon"].eq(PRIMARY_HORIZON)
            & variance["targetType"].eq("JOINT_BREAK_RUN3")
        ].iloc[0]
        rows.append(
            {
                "gateId": f"BETWEEN_MATRIX_DOMINANCE::{candidate}",
                "candidateId": candidate,
                "gateFamily": "VARIANCE_DECOMPOSITION",
                "lower95": float(variance_row.betweenMatrixFractionLower95),
                "spearman": np.nan,
                "permutationP": np.nan,
                "passed": bool(variance_row.betweenMatrixFractionLower95 > 0.5),
            }
        )
    gates = pd.DataFrame(rows)

    def both(family: str) -> bool:
        selected = gates[gates["gateFamily"].eq(family)]
        return len(selected) == 2 and bool(selected["passed"].all())

    duration = both("TRANSITION_DURATION")
    stable = both("EARLY_MATRIX_PROPENSITY")
    update = both("LONGITUDINAL_UPDATE")
    reconstruction = both("F12_JOINT_PROCESS")
    between = both("VARIANCE_DECOMPOSITION")
    classifications = []
    if duration:
        classifications.append("DURATION_DEPENDENT_HEREDITY_REGIME_SWITCHING_IDENTIFIED")
    if stable:
        classifications.append("STABLE_MATRIX_HEREDITY_PROPENSITY_IDENTIFIED")
    if update:
        classifications.append("LONGITUDINAL_STATE_UPDATING_ADDS_INFORMATION")
    if not duration and not stable and not update:
        classifications.append("FIRST_ORDER_MARKOV_PROCESS_SUFFICIENT_WITHIN_TESTED_SCOPE")
    classifications.append(
        "REGISTERED_RENEWAL_MODEL_RECONSTRUCTS_EMPIRICAL_COMMITTOR"
        if reconstruction
        else "REGISTERED_PROCESS_MODELS_DO_NOT_RECONSTRUCT_EMPIRICAL_COMMITTOR"
    )
    classifications.append(
        "RISK_VARIATION_PRIMARILY_BETWEEN_MATRICES"
        if between
        else "RISK_VARIATION_HAS_SUBSTANTIAL_WITHIN_MATRIX_COMPONENT"
    )
    classifications.append("NOT_PROMOTABLE_AS_CONFIRMED")
    if stable and reconstruction:
        next_theme = "L52_INDEPENDENT_LINEAGE_MATRIX_PROPENSITY_TRANSFER"
    elif duration and reconstruction:
        next_theme = "L52_UNTOUCHED_SEMIMARKOV_RENEWAL_TRANSFER"
    elif duration or stable or update:
        next_theme = "L52_PROCESS_ALIGNED_SHOOTING_RESIDUAL_AUDIT"
    else:
        next_theme = "L52_CROSS_SECTIONAL_RISK_LIMIT_AUDIT"
    gates = pd.concat(
        [
            gates,
            pd.DataFrame(
                [
                    {
                        "gateId": "COMPLETE_CROSS_CANDIDATE_ADJUDICATION",
                        "candidateId": "BOTH",
                        "gateFamily": "COMPLETE",
                        "lower95": np.nan,
                        "spearman": np.nan,
                        "permutationP": np.nan,
                        "passed": duration or stable or update or reconstruction,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    return gates, classifications, next_theme


def compute_tables() -> tuple[dict[str, pd.DataFrame], list[str], str]:
    states = pd.read_parquet(L50_ROOT / "restored_state_registry.parquet")
    branches = pd.read_parquet(L50_ROOT / "branch_results.parquet")
    estimates = pd.read_parquet(L50_ROOT / "state_committor_results.parquet")
    prefix, prefix_validation = build_prefix_states(states)
    sequences, transitions = build_branch_sequences(branches, prefix)
    fitted, hazard = fit_pooled_models(transitions)
    model_tables, propensity = state_model_tables(prefix, fitted)
    transition_per_matrix, transition_aggregate = transition_metrics(transitions, model_tables)
    transition_compare, transition_bootstrap = transition_comparisons(transition_per_matrix)
    predictions = process_predictions(prefix, model_tables)
    merged, process_per_matrix, process_metric = process_metrics(predictions, estimates)
    process_compare, process_bootstrap = process_comparisons(process_per_matrix)
    realized = realized_future_metrics(merged)
    q_permutations = q_alignment_permutations(merged)
    variance, variance_bootstrap = variance_decomposition(estimates)
    propensity_summary = matrix_propensity_summary(propensity)
    duration_control = duration_alignment_control(transitions, fitted)
    gates, classifications, next_theme = scientific_gates(
        transition_compare, process_compare, process_metric, q_permutations, variance
    )
    tables = {
        "prefix_state_results.parquet": prefix,
        "prefix_replay_validation.parquet": prefix_validation,
        "branch_sequence_manifest.parquet": sequences,
        "hazard_parameter_results.parquet": hazard,
        "matrix_propensity_results.parquet": propensity,
        "matrix_propensity_summary.parquet": propensity_summary,
        "per_matrix_transition_metrics.parquet": transition_per_matrix,
        "transition_metric_results.parquet": transition_aggregate,
        "transition_model_comparisons.parquet": transition_compare,
        "transition_comparison_bootstrap.parquet": transition_bootstrap,
        "process_probability_predictions.parquet": predictions,
        "per_matrix_process_metrics.parquet": process_per_matrix,
        "process_probability_metric_results.parquet": process_metric,
        "process_model_comparisons.parquet": process_compare,
        "process_comparison_bootstrap.parquet": process_bootstrap,
        "realized_future_metric_results.parquet": realized,
        "q_alignment_permutation_results.parquet": q_permutations,
        "variance_decomposition_results.parquet": variance,
        "variance_decomposition_bootstrap.parquet": variance_bootstrap,
        "negative_control_results.parquet": duration_control,
        "scientific_gate_results.parquet": gates,
        "model_registry.parquet": model_registry(),
    }
    return tables, classifications, next_theme


def make_figures(tables: dict[str, pd.DataFrame]) -> None:
    root = BUILD_ROOT / "figures"
    root.mkdir(parents=True, exist_ok=True)
    colors = {CANDIDATES[0]: "#4c78a8", CANDIDATES[1]: "#f58518"}
    hazard = tables["hazard_parameter_results.parquet"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for axis, candidate in zip(axes, CANDIDATES, strict=True):
        group = hazard[
            hazard["candidateId"].eq(candidate)
            & hazard["modelId"].eq("POOLED_SEMIMARKOV_DURATION")
        ]
        for state, state_group in group.groupby("currentState"):
            axis.plot(
                state_group["duration"],
                state_group["probabilityNextInherited"],
                marker="o",
                label="current inherited" if state else "current noninherited",
            )
        axis.set_title(f"C{candidate[-2:]}")
        axis.set_xlabel("Current regime dwell (12 = 12+)")
        axis.legend()
    axes[0].set_ylabel("P(next fission inherited)")
    fig.suptitle("Development-fitted duration-dependent transition hazards")
    fig.tight_layout()
    fig.savefig(root / "01_duration_hazard_curves.png", dpi=160)
    plt.close(fig)

    transition = tables["transition_metric_results.parquet"]
    validation = transition[transition["matrixRole"].eq("VALIDATION")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for axis, candidate in zip(axes, CANDIDATES, strict=True):
        group = validation[validation["candidateId"].eq(candidate)]
        axis.barh(group["modelId"], group["equalMatrixMeanLogLoss"], color=colors[candidate])
        axis.set_title(f"C{candidate[-2:]}")
        axis.set_xlabel("Heldout transition log loss")
    fig.suptitle("IID, Markov, semi-Markov and matrix propensity")
    fig.tight_layout()
    fig.savefig(root / "02_transition_model_hierarchy.png", dpi=160)
    plt.close(fig)

    variance = tables["variance_decomposition_results.parquet"]
    primary = variance[
        variance["horizon"].eq(PRIMARY_HORIZON)
        & variance["targetType"].eq("JOINT_BREAK_RUN3")
    ]
    fig, axis = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(primary))
    axis.bar(x, primary["betweenMatrixVariance"], label="between matrix")
    axis.bar(
        x,
        primary["withinMatrixVariance"],
        bottom=primary["betweenMatrixVariance"],
        label="within matrix / landmark",
    )
    axis.set_xticks(x, [f"C{value[-2:]}" for value in primary["candidateId"]])
    axis.set_ylabel("Corrected q variance")
    axis.set_title("F12 process-risk variance decomposition")
    axis.legend()
    fig.tight_layout()
    fig.savefig(root / "03_risk_variance_decomposition.png", dpi=160)
    plt.close(fig)

    prediction = tables["process_probability_predictions.parquet"]
    q = pd.read_parquet(L50_ROOT / "state_committor_results.parquet")
    joined = prediction.merge(
        q[["stateId", "horizon", "targetType", "q"]],
        on=["stateId", "horizon", "targetType"],
    )
    joined = joined[
        joined["matrixRole"].eq("VALIDATION")
        & joined["horizon"].eq(PRIMARY_HORIZON)
        & joined["targetType"].eq("JOINT_BREAK_RUN3")
        & joined["modelId"].eq("CURRENT_PREFIX_MATRIX_SEMIMARKOV")
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis, candidate in zip(axes, CANDIDATES, strict=True):
        group = joined[joined["candidateId"].eq(candidate)]
        axis.scatter(group["predictedProbability"], group["q"], alpha=0.5, color=colors[candidate])
        axis.plot([0, 1], [0, 1], ls=":", color="black")
        axis.set_title(f"C{candidate[-2:]}, rho={safe_spearman(group.predictedProbability.to_numpy(), group.q.to_numpy()):.2f}")
        axis.set_xlabel("Current-prefix semi-Markov probability")
        axis.set_ylabel("64-branch empirical q")
    fig.suptitle("Registered process model versus empirical committor")
    fig.tight_layout()
    fig.savefig(root / "04_model_probability_vs_empirical_committor.png", dpi=160)
    plt.close(fig)

    propensity = tables["matrix_propensity_results.parquet"]
    validation_propensity = propensity[propensity["matrixRole"].eq("VALIDATION")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis, candidate in zip(axes, CANDIDATES, strict=True):
        group = validation_propensity[validation_propensity["candidateId"].eq(candidate)]
        for _, matrix_group in group.groupby("matrixIndex"):
            axis.plot(
                matrix_group["completedFissionLandmark"],
                matrix_group["currentEquilibriumInheritance"],
                alpha=0.25,
                color=colors[candidate],
            )
        axis.set_title(f"C{candidate[-2:]}")
        axis.set_xlabel("Completed fissions")
        axis.set_ylabel("Prefix-estimated equilibrium inheritance")
    fig.suptitle("Stable matrix propensity versus longitudinal updating")
    fig.tight_layout()
    fig.savefig(root / "05_matrix_propensity_trajectories.png", dpi=160)
    plt.close(fig)

    process = tables["process_probability_metric_results.parquet"]
    process = process[
        process["matrixRole"].eq("VALIDATION")
        & process["targetType"].eq("JOINT_BREAK_RUN3")
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for axis, candidate in zip(axes, CANDIDATES, strict=True):
        group = process[process["candidateId"].eq(candidate)]
        for model in MODELS:
            model_group = group[group["modelId"].eq(model)]
            axis.plot(model_group["horizon"], model_group["equalMatrixMeanBranchLogLoss"], marker="o", label=model)
        axis.set_title(f"C{candidate[-2:]}")
        axis.set_xlabel("Future fissions")
        axis.set_ylabel("Joint-event branch log loss")
    axes[1].legend(fontsize=6)
    fig.suptitle("Finite-horizon process-probability reconstruction")
    fig.tight_layout()
    fig.savefig(root / "06_horizon_process_model_comparison.png", dpi=160)
    plt.close(fig)

    gates = tables["scientific_gate_results.parquet"]
    candidate_gates = gates[gates["candidateId"].isin(CANDIDATES)]
    matrix = candidate_gates.pivot(index="gateFamily", columns="candidateId", values="passed")
    fig, axis = plt.subplots(figsize=(7, 4.5))
    image = axis.imshow(matrix.to_numpy(dtype=float), vmin=0, vmax=1, cmap="RdYlGn")
    axis.set_xticks(range(len(matrix.columns)), [f"C{value[-2:]}" for value in matrix.columns])
    axis.set_yticks(range(len(matrix.index)), matrix.index)
    axis.set_title("Preregistered cross-candidate decision matrix")
    fig.colorbar(image, ax=axis, ticks=[0, 1])
    fig.tight_layout()
    fig.savefig(root / "07_scientific_gate_matrix.png", dpi=160)
    plt.close(fig)


def report_text(
    tables: dict[str, pd.DataFrame],
    classifications: list[str],
    next_theme: str,
    runtime: dict[str, Any],
) -> str:
    transition = tables["transition_metric_results.parquet"]
    transition = transition[transition["matrixRole"].eq("VALIDATION")][
        ["candidateId", "modelId", "matrices", "equalMatrixMeanLogLoss", "equalMatrixMeanBrier"]
    ]
    comparisons = tables["transition_model_comparisons.parquet"]
    process = tables["process_probability_metric_results.parquet"]
    process = process[
        process["matrixRole"].eq("VALIDATION")
        & process["horizon"].eq(PRIMARY_HORIZON)
        & process["targetType"].eq("JOINT_BREAK_RUN3")
    ][
        ["candidateId", "modelId", "states", "equalMatrixMeanBranchLogLoss", "equalMatrixMeanBranchBrier", "qRmse", "qSpearman"]
    ]
    variance = tables["variance_decomposition_results.parquet"]
    variance = variance[
        variance["horizon"].eq(PRIMARY_HORIZON)
        & variance["targetType"].eq("JOINT_BREAK_RUN3")
    ][
        ["candidateId", "betweenMatrixVariance", "withinMatrixVariance", "betweenMatrixFraction", "betweenMatrixFractionLower95", "betweenMatrixFractionUpper95", "withinMatrixGenerationSpearman"]
    ]
    gates = tables["scientific_gate_results.parquet"]
    return f"""# S19-L51 Full Results — Heredity-Regime Hazard and Renewal Decomposition

## Top summary

- **Research step:** `{VERSION}`
- **Completion status:** complete; additive exploratory analysis-only evidence
- **Artifacts written:** exact L50 prefix/branch-sequence replay, six locked stochastic-process models, transition hazards, finite-horizon break/conditional/joint probabilities, 4,096 catalytic-matrix bootstraps, 512 whole-matrix permutations, variance decomposition, seven figures, report and hash manifests
- **Validation:** PASS — immutable S01–L50 evidence; eleven fixtures; exact 800-state and 51,200-branch reconstruction; strict-H identity; source/input/repository/seed locks; two exact analysis passes; report regeneration; runtime, storage and artifact hashes
- **Outcome classification:** {', '.join(f'`{value}`' for value in classifications)}
- **Lay summary:** L51 asks whether the process behaves like independent inheritance, a two-state Markov switch, a duration-dependent renewal process, a stable matrix-specific propensity, or a propensity that changes along one lineage. It does not create a new replicator label or simulate a new future.
- **Recommended next action:** `{next_theme}` under the bounded autonomous authorization through L65. S20, E02, author contact, Phi variants and interventions remain inactive.

## Frozen question and design

The input is exactly L50: 40 development and 40 validation catalytic matrices, both simulator candidates, five post-fission landmarks, and 64 F12 branches per state. The strict `H>0.9` parent/daughter inheritance sequence and the break-then-three-inherited-fissions event are unchanged. Break probability, resumption conditional on a break, and their joint probability remain separate.

Six complete models were registered before opening L51-derived outcomes. The hierarchy begins with pooled IID and first-order Markov transition probabilities, adds a capped-dwell semi-Markov hazard, then adds matrix-specific transition probabilities learned only from the observed primary-lineage prefix. The *early* matrix model is frozen at fission 20 for all later landmarks; the *current* model updates only from observations available at its landmark. Pooled models are fit only on development matrices. No threshold, duration cap, smoothing strength, model, horizon or candidate was selected by result proximity.

## Heldout one-step transition models

{transition.to_markdown(index=False, floatfmt='.7f')}

### Registered transition comparisons

{comparisons.to_markdown(index=False, floatfmt='.7f')}

## F12 joint-process probability reconstruction

{process.to_markdown(index=False, floatfmt='.7f')}

The empirical committor is the 64-branch L50 probability, not the single realized primary-lineage future. The latter is retained only as a noisy diagnostic.

## Between-matrix versus within-matrix variation

{variance.to_markdown(index=False, floatfmt='.7f')}

The balanced one-way decomposition subtracts registered branch-binomial noise. A large between-matrix fraction supports a stable catalytic-network propensity; a large within-matrix component supports state or episode evolution. The within-matrix generation correlation is descriptive and is not required to be positive: regime switching need not form a universal rising trajectory.

## Scientific gates

{gates.to_markdown(index=False, floatfmt='.7f')}

## Interpretation boundary

A favorable result supports a compact stochastic-process description of local compositional heredity under this reconstructed simulator. It does not establish one privileged attractor, an organism, restored molecular identity, independent functional memory, PhiID foresight, author-code identity, intervention efficacy or real chemistry. Matrix prefixes and branch ensembles are simulator-accessible observations. This adaptive analysis reuses L50 outcomes and is not confirmatory.

## Source grounding

The registered hierarchy follows finite-state Markov, renewal/semi-Markov and heterogeneous-chain empirical-Bayes literature. Web research was used only to ground the model family before L51 outcomes; it did not supply or select a favorable parameter. Exact source records and URLs are in `source_registry.parquet`.

## Runtime and provenance

- Repository lock: `{runtime['repositoryHead']}`.
- Workers: `{runtime['workers']}`; one numerical-library thread; GPU hours: 0.
- Wall time: `{runtime['wallSeconds'] / 60:.3f}` minutes; CPU upper estimate: `{runtime['estimatedCpuHours']:.6f}` hours.
- New matrices, primary trajectories and branch streams: 0, 0 and 0.
- Frozen branch sequences analyzed: `{runtime['frozenBranchSequences']:,}`; transition observations: `{runtime['transitionObservations']:,}`.
- Matrix bootstraps: {BOOTSTRAPS}; whole-matrix permutations: {PERMUTATIONS}.

## Limitations

The process is binary and threshold-defined, even though the threshold is frozen rather than searched. Matrix-prefix estimates use one realized lineage and are shrunk rather than direct latent network parameters. Five landmarks cannot capture every episode. Branches and landmarks within a catalytic matrix are dependent; all uncertainty resamples matrices. Exact post-fission alignment controls within-cycle phase rather than estimating its effect. The result remains exploratory and cannot retroactively change L44, L50 or S18.
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
        "schema": "eidosoma.e01.s19_l51.artifact_manifest.v1",
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
            "beliefBeforeLoop": "L50 established reliable fission-aligned process probabilities but shooting failed to improve beyond direct history in both candidates.",
            "failureOrAmbiguityTargeted": "Whether the measured risk is IID inheritance, first-order switching, duration-dependent renewal, stable matrix propensity or longitudinal state updating.",
            "informationGainRationale": "A fixed stochastic-process hierarchy directly decomposes the successful teacher without another broad feature search or new simulation.",
            "learned": "L51 model, source, input, seed, statistic and gate contract locked before derived outcomes.",
            "ledgerSequence": sequence,
            "loopId": LOOP_ID,
            "motivatingEvidence": "Complete L44 and L50 results plus reviewer regime-switching direction.",
            "proposedNextTest": "Score the locked process hierarchy on heldout L50 matrices.",
            "recordPhase": "PRE_LOOP_METHOD_LOCK",
            "remainingPlausibleHypotheses": "duration memory, stable network propensity, longitudinal updating, Markov sufficiency or irreducible shooting residual",
            "selectedHypotheses": "Plastic heredity may be an alternating renewal/semi-Markov process with matrix heterogeneity.",
            "timestampUtc": timestamp,
            "weakenedHypotheses": "Another unconstrained representation tournament is needed before process identifiability.",
        },
        {
            "appendOnly": True,
            "beliefBeforeLoop": "A useful process model must improve proper scores on validation matrices in both candidates and preserve separate break, conditional and joint estimands.",
            "failureOrAmbiguityTargeted": "Duration, matrix and longitudinal contributions to the L50 empirical committor.",
            "informationGainRationale": "Development-only fitting, exact finite-horizon propagation and whole-matrix uncertainty prevent branch-cell pseudoreplication.",
            "learned": ";".join(classifications),
            "ledgerSequence": sequence + 1,
            "loopId": LOOP_ID,
            "motivatingEvidence": "Complete L51 result.",
            "proposedNextTest": next_theme,
            "recordPhase": "POST_LOOP_RESULT_AUTONOMOUS_CONTINUATION",
            "remainingPlausibleHypotheses": next_theme,
            "selectedHypotheses": "Registered two-state process hierarchy.",
            "timestampUtc": timestamp,
            "weakenedHypotheses": "Any registered L51 gate that failed.",
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
        + f"\n\n## {LOOP_ID} — heredity-regime hazard and renewal decomposition\n\n"
        + f"- **Learned:** {', '.join(classifications)}.\n"
        + f"- **Next:** `{next_theme}`.\n",
    )
    candidate_path = ARTIFACT_ROOT / "candidate_registry.parquet"
    candidates = pd.read_parquet(candidate_path)
    candidate = {
        "branchCount": 0,
        "bundleId": "L51_HEREDITY_REGIME_RENEWAL",
        "candidateId": "S19-L51-HEREDITY-REGIME-RENEWAL",
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
        "proposedSpecification": "six fixed IID/Markov/semi-Markov/matrix-prefix process models on exact L50 branches",
        "rankingScore": 27.0,
        "registryOrder": int(candidates["registryOrder"].max()) + 1,
        "selected": "REGISTERED_RENEWAL_MODEL_RECONSTRUCTS_EMPIRICAL_COMMITTOR" in classifications,
        "selectionReason": "L50_RELIABLE_PROCESS_RISK_WITH_DIRECT_HISTORY_AMBIGUITY",
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
                "finding": f"{row.finding}; L51 use: {row.frozenUse}",
                "licenseStatus": "PUBLIC_METADATA_OR_WORKSPACE_EVIDENCE",
                "redistributionStatus": "REFERENCE_ONLY",
                "repositoryIdentity": None,
                "retainedPath": None,
                "retrievalDate": timestamp[:10],
                "sha256": None,
                "sourceId": f"L51_{row.sourceId}",
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
        raise RuntimeError("repository must be clean before L51 lock")
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/eidosoma/groups/42")
    if head != remote:
        raise RuntimeError("L51 local/remote commit mismatch")
    prior = validate_immutable_prior()
    fixtures = fixture_results()
    seeds = analysis_seed_manifest()
    firewall = seed_firewall(seeds)
    sources = source_registry()
    registry = model_registry()
    required_inputs = {
        "l50Manifest": L50_ROOT / "artifact_manifest.json",
        "l50States": L50_ROOT / "restored_state_registry.parquet",
        "l50Branches": L50_ROOT / "branch_results.parquet",
        "l50Committors": L50_ROOT / "state_committor_results.parquet",
        "l50Observed": L50_ROOT / "observed_process_outcomes.parquet",
        "l44Manifest": L44_ROOT / "artifact_manifest.json",
        "l23TrajectoryManifest": L23_ROOT / "input_trajectory_manifest.parquet",
    }
    input_rows = [
        {"inputId": name, "path": str(path), "sha256": sha256_file(path), "exists": path.is_file()}
        for name, path in required_inputs.items()
    ]
    input_validation = pd.DataFrame(input_rows)
    benchmark = {
        "schema": "eidosoma.e01.s19_l51.benchmark_projection.v1",
        "outcomeBlind": True,
        "basis": "analysis-only replay of 51,200 frozen branch sequences and 614,400 binary transitions",
        "frozenBranchSequences": 51200,
        "transitionObservations": 614400,
        "projectedCpuHoursUpper": 8.0,
        "projectedWallHoursUpper": 8.0,
        "status": "PASS",
    }
    if (
        not prior["unchanged"]
        or not fixtures["passed"].all()
        or firewall["status"] != "PASS"
        or not input_validation["exists"].all()
        or len(registry) != len(MODELS)
    ):
        raise RuntimeError("L51 preoutcome validation failed")
    shutil.copy2(CONFIG, LOOP_ROOT / "preregistration.yaml")
    BASE.atomic_text(
        LOOP_ROOT / "decision_record.md",
        "# S19-L51 decision record\n\n"
        "The human-authorized autonomous sequence through L65 remains active. L50 "
        "established a reliable fission-aligned joint process probability, but its "
        "forward-shooting estimate did not improve heldout Brier beyond direct history "
        "in both candidates. The latest reviewer direction identifies plastic heredity "
        "and regime switching as the natural object and asks whether break, conditional "
        "resumption and joint risk reflect IID inheritance, first-order Markov switching, "
        "duration-dependent renewal, stable catalytic-matrix propensity or longitudinal "
        "updating. Before any L51-derived result is opened, this record freezes the exact "
        "L50 matrices, states, branches, targets and horizons; six complete models; one "
        "duration cap and empirical-Bayes strength; development-only pooled fits; fission-20 "
        "and current-prefix matrix estimates; exact finite-horizon propagation; matrix-level "
        "uncertainty; and every gate. No new simulation, label, threshold, Phi quantity, "
        "feature search, intervention or favorable candidate selection is authorized.\n",
    )
    BASE.atomic_text(
        LOOP_ROOT / "reviewer_direction.md",
        "# Reviewer direction carried into L51\n\n"
        "The operative framing is plastic heredity and stochastic switching between "
        "hereditary and nonhereditary regimes. Break probability, resumption conditional "
        "on a break and their joint probability remain distinct. L51 compares IID, "
        "first-order Markov, duration-dependent semi-Markov/renewal and matrix-specific "
        "propensity models, separates between-matrix from within-matrix variation, and "
        "does not require a monotonic ramp. PhiID and functional-memory searches remain "
        "out of scope because prior loops found no incremental heldout value.\n",
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
            "schema": "eidosoma.e01.s19_l51.source_snapshot_manifest.v1",
            "runnerSha256": sha256_file(RUNNER_PATH),
            "coreSha256": sha256_file(CORE_PATH),
            "configSha256": sha256_file(CONFIG),
            "sources": sources.to_dict("records"),
            "webResearchFrozenAtUtc": utc_now(),
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
        "schema": "eidosoma.e01.s19_l51.implementation_lock.v1",
        "repositoryHead": head,
        "remoteHead": remote,
        "runnerSha256": sha256_file(RUNNER_PATH),
        "coreSha256": sha256_file(CORE_PATH),
        "configSha256": sha256_file(CONFIG),
        "models": list(MODELS),
        "horizons": list(HORIZONS),
        "primaryHorizon": PRIMARY_HORIZON,
        "landmarks": list(LANDMARKS),
        "threshold": THRESHOLD,
        "requiredRun": REQUIRED_RUN,
        "maximumDuration": MAXIMUM_DURATION,
        "priorStrength": PRIOR_STRENGTH,
        "matrixBootstraps": BOOTSTRAPS,
        "wholeMatrixPermutations": PERMUTATIONS,
        "workers": WORKERS,
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
        raise RuntimeError("L51 repository lock mismatch")
    prior = validate_immutable_prior()
    fixtures = fixture_results()
    locked_inputs = {
        "l50Manifest": L50_ROOT / "artifact_manifest.json",
        "l50States": L50_ROOT / "restored_state_registry.parquet",
        "l50Branches": L50_ROOT / "branch_results.parquet",
        "l50Committors": L50_ROOT / "state_committor_results.parquet",
        "l50Observed": L50_ROOT / "observed_process_outcomes.parquet",
        "l44Manifest": L44_ROOT / "artifact_manifest.json",
        "l23TrajectoryManifest": L23_ROOT / "input_trajectory_manifest.parquet",
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
        raise RuntimeError("L51 locked input changed")
    if (
        not prior["unchanged"]
        or prior["aggregateSha256"] != lock["priorAggregateSha256"]
        or not fixtures["passed"].all()
        or sha256_file(RUNNER_PATH) != lock["runnerSha256"]
        or sha256_file(CORE_PATH) != lock["coreSha256"]
        or sha256_file(CONFIG) != lock["configSha256"]
    ):
        raise RuntimeError("L51 pre-execution validation failed")
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
        "schema": "eidosoma.e01.s19_l51.regeneration_validation.v1",
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
        raise RuntimeError("L51 regeneration failure")
    for name, frame in tables.items():
        BASE.write_parquet(BUILD_ROOT / name, frame)
    BASE.write_json(
        BUILD_ROOT / "classification.json",
        {
            "schema": "eidosoma.e01.s19_l51.classification.v1",
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
        "schema": "eidosoma.e01.s19_l51.runtime.v1",
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
        "matrixBootstraps": BOOTSTRAPS,
        "wholeMatrixPermutations": PERMUTATIONS,
        "analysisPasses": 2,
        "completedAtUtc": utc_now(),
    }
    if runtime["estimatedCpuHours"] > 34 or runtime["wallSeconds"] > 20 * 3600:
        raise RuntimeError("L51 runtime ceiling exceeded")
    BASE.write_json(BUILD_ROOT / "runtime_manifest.json", runtime)
    BASE.write_json(BUILD_ROOT / "regeneration_validation.json", regeneration)
    retained_bytes = sum(
        path.stat().st_size for path in BUILD_ROOT.rglob("*") if path.is_file()
    ) + sum(path.stat().st_size for path in LOOP_ROOT.iterdir() if path.is_file())
    storage = {
        "schema": "eidosoma.e01.s19_l51.storage_validation.v1",
        "status": "PASS" if retained_bytes <= 15 * 1024**3 else "FAIL",
        "retainedBytes": retained_bytes,
        "retainedGiBCeiling": 15,
        "temporaryGiBCeiling": 30,
    }
    BASE.write_json(BUILD_ROOT / "storage_validation.json", storage)
    report = report_text(tables, classifications, next_theme, runtime)
    if report != report_text(tables, classifications, next_theme, runtime):
        raise RuntimeError("L51 report regeneration failure")
    BASE.atomic_text(BUILD_ROOT / "research_step_full_results.md", report)
    BASE.atomic_text(BUILD_ROOT / "S19_L51_FULL_RESULTS.md", report)
    BASE.atomic_text(
        BUILD_ROOT / "loop_decision_summary.md",
        f"# S19-L51 decision summary\n\n**Classification:** {', '.join(classifications)}\n\n**Next:** `{next_theme}`.\n",
    )
    if storage["status"] != "PASS":
        raise RuntimeError("L51 storage ceiling exceeded")
    for path in (BUILD_ROOT / "figures").glob("*.png"):
        if not path.stat().st_size:
            raise RuntimeError(f"empty L51 figure: {path}")
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
        raise RuntimeError("L51 artifact manifest regeneration failure")
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
            "nextAuthorizedLoop": "S19-L52",
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
