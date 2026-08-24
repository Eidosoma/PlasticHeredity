#!/usr/bin/env python3
"""Run S19-L50 fission-aligned process-committor horizon audit."""

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
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from e01_onset_discovery.empirical_committor import RestoredState
from e01_onset_discovery.fission_aligned_process import (
    future_post_fission_count,
    nested_process_scores,
    post_fission_index,
)
from e01_onset_discovery.fission_clock_recurrence import simulate_fission_clock
from e01_onset_discovery.longitudinal_process_risk import (
    jeffreys_mean,
    trailing_true_run,
)
from e01_onset_discovery.recurrence_inheritance import cosine_h


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


L49R = load_module(
    "e01_l50_l49r_runner",
    ROOT / "scripts/e01/run_s19_l49r_longitudinal_process_committor_repair.py",
)
L49 = L49R.L49
L28 = L49.L28
BASE = L49.BASE

ARTIFACT_ROOT = Path("/artifacts/research_steps/S19")
LOOP_ROOT = ARTIFACT_ROOT / "loops/L50"
L23_ROOT = ARTIFACT_ROOT / "loops/L23"
L24_ROOT = ARTIFACT_ROOT / "loops/L24"
L49R_ROOT = ARTIFACT_ROOT / "loops/L49R"
BUILD_ROOT = Path("/cache/e01_s19_l50/build")
CONFIG = ROOT / "configs/e01/s19_l50_fission_aligned_process_committor.yaml"
RUNNER_PATH = Path(__file__).resolve()
CORE_PATH = ROOT / "src/e01_onset_discovery/fission_aligned_process.py"

LOOP_ID = "S19-L50"
VERSION = "E01-S19-L50-FISSION-ALIGNED-PROCESS-COMMITTOR-HORIZON-v1.0.0"
CANDIDATES = ("S12F-CANDIDATE-02", "S12F-CANDIDATE-03")
ROLES = ("DEVELOPMENT", "VALIDATION")
GENERATIONS = (20, 35, 50, 65, 80)
HORIZONS = (4, 8, 12)
PRIMARY_HORIZON = 12
MATRICES_PER_ROLE = 40
BRANCHES = 64
HALF = 32
THRESHOLD = 0.9
REQUIRED_RUN = 3
BOOTSTRAPS = 4096
PERMUTATIONS = 512
WORKERS = min(8, os.cpu_count() or 1)
SEED_ROOT = bytes.fromhex(
    "7125b9850a4bf944a0f9d5ffc9d99fb41aca78b160d688afb963b1dc060e343a"
)
CONTROL_COLUMNS = (
    "normalizedGeneration",
    "prefixInheritanceFraction",
    "recentFiveInheritanceFraction",
    "prefixTrailingInheritanceRun",
    "latestParentDaughterH",
    "fissionsSinceLatestBreak",
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def sha256_file(path: Path) -> str:
    return L49.sha256_file(path)


def frame_hash(frame: pd.DataFrame) -> str:
    return L49.frame_hash(frame)


def seed_material(*parts: object) -> bytes:
    canonical = tuple(
        part.item() if isinstance(part, np.generic) else part for part in parts
    )
    return hashlib.sha256(
        SEED_ROOT + b"\x00" + json.dumps(canonical, separators=(",", ":")).encode()
    ).digest()


def derived_seed(*parts: object) -> int:
    return int.from_bytes(seed_material(*parts)[:16], "big")


def generator(*parts: object) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64DXSM(derived_seed(*parts)))


def interval(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return float("nan"), float("nan")
    return tuple(map(float, np.quantile(finite, [0.025, 0.975])))


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3 or len(np.unique(a[mask])) < 2 or len(np.unique(b[mask])) < 2:
        return float("nan")
    return float(spearmanr(a[mask], b[mask]).statistic)


def validate_immutable_prior() -> dict[str, Any]:
    prior = L49R.validate_immutable_prior()
    manifest = json.loads((L49R_ROOT / "artifact_manifest.json").read_text())
    rows = []
    for row in manifest["files"]:
        path = L49R_ROOT / row["path"]
        actual = sha256_file(path) if path.is_file() else None
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
        "schema": "eidosoma.e01.s19_l50.immutable_prior_validation.v1",
        "status": "PASS" if passed else "FAIL",
        "unchanged": passed,
        "priorThroughL49Unchanged": bool(prior["unchanged"]),
        "validatedL49RArtifactCount": len(rows),
        "aggregateSha256": hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "rows": rows,
    }


def source_registry() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sourceId": "L49R_LONGITUDINAL_RESULT",
                "evidenceClass": "DIRECT_FROZEN_E01_RESULT",
                "finding": "Molecular-clock landmarks mixed growth phase; joint F12 probability was highly split-half reliable, while conditional renewal availability and incremental forecast gates failed.",
                "frozenUse": "align observation with the fission-defined event and keep joint, break and conditional components separate",
                "url": None,
            },
            {
                "sourceId": "REVIEWER_CLOCK_MATCHING",
                "evidenceClass": "HUMAN_REVIEW_DIRECTION",
                "finding": "Match the observation clock to a process defined over fission opportunities and report generational and molecular clocks separately.",
                "frozenUse": "restore exact post-fission states at five fixed completed-generation landmarks",
                "url": None,
            },
            {
                "sourceId": "TRANSITION_PATH_COMMITTOR",
                "evidenceClass": "PRIMARY_METHOD_SOURCE",
                "finding": "A committor is a state-conditioned probability of reaching a specified future event.",
                "frozenUse": "evaluate nested fixed event horizons without choosing one by outcome proximity",
                "url": "https://doi.org/10.1007/978-3-540-79537-1_13",
            },
        ]
    )


def fixture_results() -> pd.DataFrame:
    early = nested_process_scores(
        [0.2, 0.91, 0.92, 0.93, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95]
    )
    late = nested_process_scores(
        [0.95, 0.95, 0.2, 0.91, 0.92, 0.93, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95]
    )
    state = np.zeros(100, dtype=np.int64)
    state[:40] = 1
    beta = np.exp(np.full((100, 100), -4.0, dtype=np.float64))
    restored = RestoredState(tuple(map(int, state)), "post_fission", 20, 21, 0, 0)

    def simulate() -> Any:
        streams = [np.random.Generator(np.random.PCG64DXSM(5000 + i)) for i in range(4)]
        return simulate_fission_clock(
            restored=restored,
            beta=beta,
            definition=L28.definition(CANDIDATES[0]),
            event_rng=streams[0],
            trim_rng=streams[1],
            fission_rng=streams[2],
            daughter_rng=streams[3],
            future_fissions=PRIMARY_HORIZON,
        )

    first = simulate()
    replay = simulate()
    return pd.DataFrame(
        [
            {"fixtureId": "F01_EARLY_EVENT_NESTING", "passed": all(early[h].event for h in HORIZONS)},
            {"fixtureId": "F02_LATE_EVENT_NESTING", "passed": not late[4].event and late[8].event and late[12].event},
            {"fixtureId": "F03_STRICT_H090", "passed": nested_process_scores([0.9, 0.91, 0.92, 0.93] + [0.95] * 8)[4].event},
            {"fixtureId": "F04_POST_FISSION_RESTORE", "passed": restored.observation_kind == "post_fission" and restored.completed_fissions == 20},
            {"fixtureId": "F05_EXACT_BRANCH_REPLAY", "passed": first == replay and first.fissions == PRIMARY_HORIZON},
            {"fixtureId": "F06_HORIZON_SCOPE", "passed": HORIZONS == (4, 8, 12) and PRIMARY_HORIZON == 12},
            {"fixtureId": "F07_GENERATION_SCOPE", "passed": GENERATIONS == (20, 35, 50, 65, 80)},
            {"fixtureId": "F08_SEED_REPLAY", "passed": derived_seed("fixture", np.int64(5)) == derived_seed("fixture", 5)},
            {"fixtureId": "F09_JEFFREYS", "passed": jeffreys_mean(3, 4) == 0.7},
        ]
    )


def select_matrices() -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_parquet(L23_ROOT / "input_trajectory_manifest.parquet")
    firewall = pd.read_parquet(L24_ROOT / "matrix_firewall.parquet")
    excluded = set(
        pd.read_parquet(L49R_ROOT / "matrix_selection_registry.parquet")["matrixIndex"].astype(int)
    )
    eligible = manifest[
        manifest["terminalStatus"].eq("requested_fissions_completed")
        & manifest["completedFissions"].ge(100)
    ]
    shared = (
        eligible.groupby("matrixIndex")["candidateId"]
        .nunique()
        .loc[lambda values: values.eq(len(CANDIDATES))]
        .index
    )
    rows: list[dict[str, Any]] = []
    for role in ROLES:
        pool = firewall[
            firewall["matrixRole"].eq(role)
            & firewall["matrixIndex"].isin(shared)
            & ~firewall["matrixIndex"].isin(excluded)
        ].copy()
        pool["selectionDigest"] = pool["matrixIndex"].map(
            lambda matrix, role=role: hashlib.sha256(
                f"{VERSION}|MATRIX_SELECTION|{role}|{int(matrix)}".encode()
            ).hexdigest()
        )
        pool = pool.sort_values(["selectionDigest", "matrixIndex"])
        if len(pool) < MATRICES_PER_ROLE:
            raise RuntimeError("insufficient unused shared matrices for L50")
        for rank, row in enumerate(pool.head(MATRICES_PER_ROLE).itertuples(), start=1):
            rows.append(
                {
                    "matrixRole": role,
                    "matrixIndex": int(row.matrixIndex),
                    "selectionRank": rank,
                    "selectionDigest": row.selectionDigest,
                    "eligibleUnusedSharedPool": len(pool),
                    "excludedFromL49R": False,
                    "selectedBeforeOutcome": True,
                }
            )
    selected = pd.DataFrame(rows).sort_values(["matrixRole", "selectionRank"]).reset_index(drop=True)
    expanded = pd.DataFrame(
        [
            {**row._asdict(), "candidateId": candidate, "completedFissionLandmark": generation}
            for row in selected.itertuples(index=False)
            for candidate in CANDIDATES
            for generation in GENERATIONS
        ]
    ).sort_values(
        ["matrixRole", "candidateId", "matrixIndex", "completedFissionLandmark"]
    ).reset_index(drop=True)
    expanded["stateId"] = expanded.apply(
        lambda row: hashlib.sha256(
            f"{VERSION}|{row.matrixRole}|{row.candidateId}|{int(row.matrixIndex)}|{int(row.completedFissionLandmark)}".encode()
        ).hexdigest()[:24],
        axis=1,
    )
    if len(selected) != 80 or len(expanded) != 800 or set(selected.matrixIndex) & excluded:
        raise RuntimeError("L50 selection scope failure")
    return selected, expanded


def _boundary_h(selected: tuple[Any, ...], boundary_index: int) -> float:
    if boundary_index == 0 or selected[boundary_index - 1].observation_kind != "molecular_update":
        raise RuntimeError("post-fission boundary lacks selected pre-fission parent")
    return cosine_h(
        np.asarray(selected[boundary_index - 1].state, dtype=np.int64),
        np.asarray(selected[boundary_index].state, dtype=np.int64),
    )


def build_states(
    expanded: pd.DataFrame,
) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_parquet(L23_ROOT / "input_trajectory_manifest.parquet")
    manifest_index = manifest.set_index(["candidateId", "matrixIndex"])
    payloads: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    observed_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    for (role, candidate, matrix), group in expanded.groupby(
        ["matrixRole", "candidateId", "matrixIndex"], sort=True
    ):
        source = manifest_index.loc[(candidate, int(matrix))]
        trajectory = L28.load_trajectory(source)
        selected = tuple(L28.selected_clock_observations(trajectory, L28.CLOCK_ID))
        beta = L28.generate_beta(
            L28.derive_seed(L28.L23_ROOT_HEX, L28.L23_PHASE, "catalytic_matrix", int(matrix))
        )
        beta_hash = L28.simulator_array_sha256(beta)
        if beta_hash != source.betaSha256:
            raise RuntimeError("L50 beta identity mismatch")
        boundary_indices = [
            index
            for index, observation in enumerate(selected)
            if observation.observation_kind == "post_fission"
        ]
        boundary_h = [_boundary_h(selected, index) for index in boundary_indices]
        for row in group.sort_values("completedFissionLandmark").itertuples(index=False):
            generation = int(row.completedFissionLandmark)
            current_index = post_fission_index(selected, generation)
            current = selected[current_index]
            restored = L28.restored_state_from_observation(current)
            future_indices = [index for index in boundary_indices if index > current_index][
                :PRIMARY_HORIZON
            ]
            if (
                future_post_fission_count(selected, current_index) < PRIMARY_HORIZON
                or len(future_indices) != PRIMARY_HORIZON
            ):
                raise RuntimeError("L50 F12 future availability failure")
            future_h = np.asarray(
                [_boundary_h(selected, index) for index in future_indices],
                dtype=np.float64,
            )
            observed = nested_process_scores(
                future_h,
                HORIZONS,
                threshold=THRESHOLD,
                required_run=REQUIRED_RUN,
            )
            prefix_positions = [
                position
                for position, index in enumerate(boundary_indices)
                if index <= current_index
            ]
            prefix_h = np.asarray(
                [boundary_h[position] for position in prefix_positions], dtype=np.float64
            )
            inherited = prefix_h > THRESHOLD
            latest_break_positions = np.flatnonzero(~inherited)
            fissions_since_break = (
                len(inherited) - 1 - int(latest_break_positions[-1])
                if len(latest_break_positions)
                else len(inherited)
            )
            state = np.asarray(restored.state, dtype=np.int64)
            state_id = row.stateId
            base = {
                "stateId": state_id,
                "matrixRole": role,
                "candidateId": candidate,
                "matrixIndex": int(matrix),
                "selectionRank": int(row.selectionRank),
                "completedFissionLandmark": generation,
                "normalizedGeneration": generation / 100.0,
                "trajectoryId": source.trajectoryId,
                "currentSelectedIndex": current_index,
                "currentObservationKind": current.observation_kind,
                "currentCompletedFissions": int(current.completed_fissions),
                "currentGrowthGeneration": int(current.growth_generation_one_based),
                "currentGenerationLocalStep": int(current.generation_local_step),
                "currentBatchStep": int(current.batch_step),
                "currentMass": int(state.sum()),
                "prefixBoundaryCount": len(prefix_h),
                "prefixInheritanceFraction": float(inherited.mean()),
                "recentFiveInheritanceFraction": float(inherited[-5:].mean()),
                "prefixTrailingInheritanceRun": trailing_true_run(inherited),
                "latestParentDaughterH": float(prefix_h[-1]),
                "fissionsSinceLatestBreak": int(fissions_since_break),
                "futureFissionsAvailable": future_post_fission_count(selected, current_index),
                "currentStateSha256": L28.array_sha256(state),
                "betaSha256": beta_hash,
                "trajectorySha256": source.trajectorySha256,
                "selectedClockLength": len(selected),
                "targetUsesCompletedTestTrajectory": False,
            }
            state_rows.append(base)
            for horizon in HORIZONS:
                score = observed[horizon]
                observed_rows.append(
                    {
                        "stateId": state_id,
                        "matrixRole": role,
                        "candidateId": candidate,
                        "matrixIndex": int(matrix),
                        "completedFissionLandmark": generation,
                        "horizon": horizon,
                        "observedBreak": score.break_observed,
                        "observedJointEvent": score.event,
                        "observedConditionalEvent": score.event if score.break_observed else None,
                        "observedBreakBoundaryOneBased": score.break_boundary_one_based,
                        "observedCertificationBoundaryOneBased": score.certification_boundary_one_based,
                        "observedFutureInheritanceFraction": float((future_h[:horizon] > THRESHOLD).mean()),
                        "targetUsesCompletedTestTrajectory": False,
                    }
                )
            validation_rows.append(
                {
                    "stateId": state_id,
                    "trajectoryIdentityPassed": trajectory.trajectory_sha256 == source.trajectorySha256,
                    "trajectoryCacheIdentityPassed": sha256_file(Path(source.cachePath)) == source.cacheSha256,
                    "betaIdentityPassed": beta_hash == source.betaSha256,
                    "postFissionIdentityPassed": current.observation_kind == "post_fission" and int(current.completed_fissions) == generation,
                    "restoredStateExact": L28.array_sha256(state) == L28.array_sha256(np.asarray(current.state, dtype=np.int64)),
                    "futureF12Available": len(future_indices) == PRIMARY_HORIZON,
                    "selectedBeforeOutcome": True,
                }
            )
            payloads.append({**base, "state": list(map(int, restored.state))})
    states = pd.DataFrame(state_rows).sort_values(
        ["matrixRole", "candidateId", "matrixIndex", "completedFissionLandmark"]
    ).reset_index(drop=True)
    observed_frame = pd.DataFrame(observed_rows).sort_values(
        ["matrixRole", "candidateId", "matrixIndex", "completedFissionLandmark", "horizon"]
    ).reset_index(drop=True)
    validation = pd.DataFrame(validation_rows).sort_values("stateId").reset_index(drop=True)
    checks = [
        column
        for column in validation
        if column.endswith("Passed")
        or column in ("restoredStateExact", "futureF12Available", "selectedBeforeOutcome")
    ]
    if (
        len(payloads) != 800
        or len(states) != 800
        or len(observed_frame) != 2400
        or not validation[checks].all().all()
        or states["futureFissionsAvailable"].min() < PRIMARY_HORIZON
    ):
        raise RuntimeError("L50 restored-state validation failure")
    return payloads, states, observed_frame, validation


def branch_seed_manifest(payloads: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for payload in payloads:
        for branch in range(BRANCHES):
            row = {
                "stateId": payload["stateId"],
                "matrixRole": payload["matrixRole"],
                "candidateId": payload["candidateId"],
                "matrixIndex": int(payload["matrixIndex"]),
                "completedFissionLandmark": int(payload["completedFissionLandmark"]),
                "branchIndex": branch,
                "branchHalf": "A" if branch < HALF else "B",
                "rootHex": SEED_ROOT.hex(),
            }
            materials = []
            for purpose in ("event", "trim", "fission", "daughter"):
                parts = ("branch", payload["stateId"], branch, purpose)
                row[f"{purpose}DerivedSeed"] = str(derived_seed(*parts))
                row[f"{purpose}SeedMaterialSha256"] = seed_material(*parts).hex()
                materials.append(row[f"{purpose}SeedMaterialSha256"])
            row["branchIdentitySha256"] = hashlib.sha256(
                "|".join([payload["stateId"], str(branch), *materials]).encode()
            ).hexdigest()
            rows.append(row)
    frame = pd.DataFrame(rows).sort_values(
        ["matrixRole", "candidateId", "matrixIndex", "completedFissionLandmark", "branchIndex"]
    ).reset_index(drop=True)
    if len(frame) != 800 * BRANCHES or frame["branchIdentitySha256"].duplicated().any():
        raise RuntimeError("L50 branch seed scope failure")
    return frame


def analysis_seed_manifest() -> pd.DataFrame:
    rows = []
    for candidate in CANDIDATES:
        for horizon in HORIZONS:
            for target in ("BREAK", "JOINT_BREAK_RUN3", "RUN3_GIVEN_BREAK"):
                parts = ("matrix_bootstrap", "VALIDATION", candidate, horizon, target)
                rows.append(
                    {
                        "purpose": "matrix_bootstrap_reliability",
                        "matrixRole": "VALIDATION",
                        "candidateId": candidate,
                        "horizon": horizon,
                        "targetType": target,
                        "repetitions": BOOTSTRAPS,
                        "derivedSeed": str(derived_seed(*parts)),
                        "seedMaterialSha256": seed_material(*parts).hex(),
                    }
                )
            parts = ("predictive_bootstrap", candidate, horizon)
            rows.append(
                {
                    "purpose": "matrix_bootstrap_prediction",
                    "matrixRole": "VALIDATION",
                    "candidateId": candidate,
                    "horizon": horizon,
                    "targetType": "JOINT_BREAK_RUN3",
                    "repetitions": BOOTSTRAPS,
                    "derivedSeed": str(derived_seed(*parts)),
                    "seedMaterialSha256": seed_material(*parts).hex(),
                }
            )
            parts = (
                "matrix_permutation",
                "VALIDATION",
                candidate,
                horizon,
                PERMUTATIONS,
            )
            rows.append(
                {
                    "purpose": "matrix_trajectory_permutation",
                    "matrixRole": "VALIDATION",
                    "candidateId": candidate,
                    "horizon": horizon,
                    "targetType": "JOINT_BREAK_RUN3",
                    "repetitions": PERMUTATIONS,
                    "derivedSeed": str(derived_seed(*parts)),
                    "seedMaterialSha256": seed_material(*parts).hex(),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["purpose", "candidateId", "horizon", "targetType"]
    ).reset_index(drop=True)


def seed_firewall(branches: pd.DataFrame, analysis: pd.DataFrame) -> dict[str, Any]:
    prior_material: set[str] = set()
    prior_derived: set[str] = set()
    for path in ARTIFACT_ROOT.glob("loops/L*/**/*seed*manifest*.parquet"):
        if "/L50/" in str(path):
            continue
        try:
            frame = pd.read_parquet(path)
        except (OSError, ValueError, TypeError):
            continue
        for column in frame.columns:
            lower = column.lower()
            if "seedmaterialsha256" in lower:
                prior_material.update(frame[column].dropna().astype(str))
            if lower == "derivedseed" or lower.endswith("derivedseed"):
                prior_derived.update(frame[column].dropna().astype(str))
    current_material = set(analysis["seedMaterialSha256"].astype(str))
    current_derived = set(analysis["derivedSeed"].astype(str))
    for column in branches.columns:
        lower = column.lower()
        if "seedmaterialsha256" in lower:
            current_material.update(branches[column].dropna().astype(str))
        if lower.endswith("derivedseed"):
            current_derived.update(branches[column].dropna().astype(str))
    overlap_m = sorted(current_material & prior_material)
    overlap_d = sorted(current_derived & prior_derived)
    passed = not overlap_m and not overlap_d
    return {
        "schema": "eidosoma.e01.s19_l50.seed_firewall.v1",
        "status": "PASS" if passed else "FAIL",
        "newBranchStreams": len(branches),
        "analysisStreams": len(analysis),
        "seedMaterialUnique": len(current_material) == len(branches) * 4 + len(analysis),
        "seedMaterialOverlapCount": len(overlap_m),
        "derivedSeedOverlapCount": len(overlap_d),
        "seedMaterialOverlaps": overlap_m,
        "derivedSeedOverlaps": overlap_d,
    }


def _branch_worker(payload: dict[str, Any]) -> list[dict[str, Any]]:
    beta = L28.generate_beta(
        L28.derive_seed(
            L28.L23_ROOT_HEX,
            L28.L23_PHASE,
            "catalytic_matrix",
            int(payload["matrixIndex"]),
        )
    )
    if L28.simulator_array_sha256(beta) != payload["betaSha256"]:
        raise RuntimeError(f"L50 worker beta mismatch: {payload['stateId']}")
    restored = RestoredState(
        tuple(payload["state"]),
        payload["currentObservationKind"],
        int(payload["currentCompletedFissions"]),
        int(payload["currentGrowthGeneration"]),
        int(payload["currentGenerationLocalStep"]),
        int(payload["currentBatchStep"]),
    )
    rows = []
    for branch in range(BRANCHES):
        trace = simulate_fission_clock(
            restored=restored,
            beta=beta,
            definition=L28.definition(payload["candidateId"]),
            event_rng=generator("branch", payload["stateId"], branch, "event"),
            trim_rng=generator("branch", payload["stateId"], branch, "trim"),
            fission_rng=generator("branch", payload["stateId"], branch, "fission"),
            daughter_rng=generator("branch", payload["stateId"], branch, "daughter"),
            future_fissions=PRIMARY_HORIZON,
        )
        scores = nested_process_scores(
            trace.parent_daughter_h,
            HORIZONS,
            threshold=THRESHOLD,
            required_run=REQUIRED_RUN,
        )
        materials = [
            seed_material("branch", payload["stateId"], branch, purpose).hex()
            for purpose in ("event", "trim", "fission", "daughter")
        ]
        base = {
            "stateId": payload["stateId"],
            "matrixRole": payload["matrixRole"],
            "candidateId": payload["candidateId"],
            "matrixIndex": int(payload["matrixIndex"]),
            "completedFissionLandmark": int(payload["completedFissionLandmark"]),
            "branchIndex": branch,
            "branchHalf": "A" if branch < HALF else "B",
            "branchIdentitySha256": hashlib.sha256(
                "|".join([payload["stateId"], str(branch), *materials]).encode()
            ).hexdigest(),
            "molecularUpdates": trace.molecular_updates,
            "fissions": trace.fissions,
            "terminalStatus": trace.terminal_status,
            "pathSha256": trace.path_sha256,
            "targetUsesCompletedTestTrajectory": False,
        }
        for index, value in enumerate(trace.parent_daughter_h, start=1):
            base[f"parentDaughterH{index:02d}"] = float(value)
        for horizon in HORIZONS:
            score = scores[horizon]
            base[f"breakH{horizon}"] = score.break_observed
            base[f"jointH{horizon}"] = score.event
            base[f"conditionalH{horizon}"] = score.event if score.break_observed else None
            base[f"breakBoundaryH{horizon}"] = score.break_boundary_one_based
            base[f"certificationBoundaryH{horizon}"] = score.certification_boundary_one_based
        rows.append(base)
    return rows


def execute_branches(payloads: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(_branch_worker, payload): payload["stateId"] for payload in payloads}
        for future in as_completed(futures):
            rows.extend(future.result())
    frame = pd.DataFrame(rows).sort_values(
        ["matrixRole", "candidateId", "matrixIndex", "completedFissionLandmark", "branchIndex"]
    ).reset_index(drop=True)
    if (
        len(frame) != len(payloads) * BRANCHES
        or frame.duplicated(["stateId", "branchIndex"]).any()
        or frame.groupby("stateId").size().ne(BRANCHES).any()
        or frame["fissions"].ne(PRIMARY_HORIZON).any()
        or frame["targetUsesCompletedTestTrajectory"].any()
    ):
        raise RuntimeError("L50 branch output scope failure")
    return frame


def state_estimates(
    branches: pd.DataFrame, states: pd.DataFrame, observed: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for state_id, group in branches.groupby("stateId", sort=False):
        first = group.iloc[0]
        for horizon in HORIZONS:
            for target_type, column in (
                ("BREAK", f"breakH{horizon}"),
                ("JOINT_BREAK_RUN3", f"jointH{horizon}"),
                ("RUN3_GIVEN_BREAK", f"conditionalH{horizon}"),
            ):
                if target_type == "RUN3_GIVEN_BREAK":
                    eligible = group[group[f"breakH{horizon}"]]
                else:
                    eligible = group
                trials = len(eligible)
                successes = int(eligible[column].fillna(False).sum())
                raw = successes / trials if trials else float("nan")
                noise = raw * (1 - raw) / trials if trials else float("nan")
                halves: dict[str, tuple[float, int, int]] = {}
                for half in ("A", "B"):
                    part = group[group["branchHalf"].eq(half)]
                    if target_type == "RUN3_GIVEN_BREAK":
                        part = part[part[f"breakH{horizon}"]]
                    half_trials = len(part)
                    half_successes = int(part[column].fillna(False).sum())
                    halves[half] = (
                        jeffreys_mean(half_successes, half_trials),
                        half_successes,
                        half_trials,
                    )
                rows.append(
                    {
                        "stateId": state_id,
                        "matrixRole": first.matrixRole,
                        "candidateId": first.candidateId,
                        "matrixIndex": int(first.matrixIndex),
                        "completedFissionLandmark": int(first.completedFissionLandmark),
                        "horizon": horizon,
                        "targetType": target_type,
                        "successes": successes,
                        "trials": trials,
                        "dataInformed": trials > 0,
                        "q": jeffreys_mean(successes, trials),
                        "qHalfA": halves["A"][0],
                        "qHalfB": halves["B"][0],
                        "successesHalfA": halves["A"][1],
                        "successesHalfB": halves["B"][1],
                        "trialsHalfA": halves["A"][2],
                        "trialsHalfB": halves["B"][2],
                        "binomialNoise": noise,
                    }
                )
    estimates = pd.DataFrame(rows)
    long = (
        estimates.merge(
            states,
            on=[
                "stateId",
                "matrixRole",
                "candidateId",
                "matrixIndex",
                "completedFissionLandmark",
            ],
            validate="many_to_one",
        )
        .sort_values(
            [
                "matrixRole",
                "candidateId",
                "matrixIndex",
                "completedFissionLandmark",
                "horizon",
                "targetType",
            ]
        )
        .reset_index(drop=True)
    )
    observed_long = observed.copy()
    observed_long["targetType"] = "JOINT_BREAK_RUN3"
    observed_long = observed_long.rename(columns={"observedJointEvent": "observedTarget"})
    long = long.merge(
        observed_long[
            [
                "stateId",
                "horizon",
                "targetType",
                "observedTarget",
                "observedBreak",
            ]
        ],
        on=["stateId", "horizon", "targetType"],
        how="left",
        validate="many_to_one",
    )
    if len(long) != 800 * len(HORIZONS) * 3:
        raise RuntimeError("L50 state-estimate scope failure")
    return long


def _center(frame: pd.DataFrame, column: str) -> np.ndarray:
    return (
        frame[column]
        - frame.groupby("_matrixUnit", sort=False)[column].transform("mean")
    ).to_numpy(float)


def _reliability_metrics(frame: pd.DataFrame) -> dict[str, float]:
    data = frame.copy()
    if "_matrixUnit" not in data:
        data["_matrixUnit"] = data["matrixIndex"].astype(str)
    eligible = data[
        data["dataInformed"]
        & data["trialsHalfA"].gt(0)
        & data["trialsHalfB"].gt(0)
        & np.isfinite(data["binomialNoise"])
    ].copy()
    if len(eligible) < 3:
        return {
            "splitHalfSpearman": float("nan"),
            "centeredSplitHalfSpearman": float("nan"),
            "correctedBetweenStateVariance": float("nan"),
            "correctedWithinMatrixVariance": float("nan"),
            "meanAbsoluteSuccessiveDelta": float("nan"),
            "intermediateStates": 0.0,
            "dataInformedFraction": float(len(eligible) / len(data)) if len(data) else 0.0,
        }
    observed_variance = float(np.var(eligible["q"], ddof=1))
    mean_noise = float(eligible["binomialNoise"].mean())
    within = []
    for _, group in eligible.groupby("_matrixUnit", sort=False):
        if len(group) >= 2:
            within.append(
                float(np.var(group["q"], ddof=1) - group["binomialNoise"].mean())
            )
    ordered = eligible.sort_values(["_matrixUnit", "completedFissionLandmark"])
    return {
        "splitHalfSpearman": safe_spearman(eligible["qHalfA"], eligible["qHalfB"]),
        "centeredSplitHalfSpearman": safe_spearman(
            _center(eligible, "qHalfA"), _center(eligible, "qHalfB")
        ),
        "correctedBetweenStateVariance": observed_variance - mean_noise,
        "correctedWithinMatrixVariance": float(np.mean(within)) if within else float("nan"),
        "meanAbsoluteSuccessiveDelta": float(
            ordered.groupby("_matrixUnit")["q"].diff().abs().mean()
        ),
        "intermediateStates": float(
            eligible["q"].between(0.1, 0.9, inclusive="neither").sum()
        ),
        "dataInformedFraction": float(len(eligible) / len(data)),
    }


def reliability_results(
    estimates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    point_rows = []
    bootstrap_rows = []
    for (role, candidate, horizon, target), group in estimates.groupby(
        ["matrixRole", "candidateId", "horizon", "targetType"], sort=True
    ):
        point = _reliability_metrics(group)
        row = {
            "matrixRole": role,
            "candidateId": candidate,
            "horizon": int(horizon),
            "targetType": target,
            "matrices": group["matrixIndex"].nunique(),
            "states": len(group),
            **point,
        }
        if role == "VALIDATION":
            matrices = sorted(group["matrixIndex"].unique())
            rng = generator("matrix_bootstrap", role, candidate, int(horizon), target)
            draws: dict[str, list[float]] = {key: [] for key in point}
            for replicate in range(BOOTSTRAPS):
                sampled = rng.choice(matrices, size=len(matrices), replace=True)
                pieces = []
                for unit, matrix in enumerate(sampled):
                    piece = group[group["matrixIndex"].eq(matrix)].copy()
                    piece["_matrixUnit"] = f"{unit}:{matrix}"
                    pieces.append(piece)
                metrics = _reliability_metrics(pd.concat(pieces, ignore_index=True))
                for key, value in metrics.items():
                    draws[key].append(value)
                bootstrap_rows.append(
                    {
                        "candidateId": candidate,
                        "horizon": int(horizon),
                        "targetType": target,
                        "replicate": replicate,
                        **metrics,
                    }
                )
            for key, values in draws.items():
                low, high = interval(np.asarray(values))
                row[f"{key}Lower95"] = low
                row[f"{key}Upper95"] = high
        else:
            for key in point:
                row[f"{key}Lower95"] = float("nan")
                row[f"{key}Upper95"] = float("nan")
        point_rows.append(row)
    return (
        pd.DataFrame(point_rows).sort_values(
            ["matrixRole", "candidateId", "horizon", "targetType"]
        ).reset_index(drop=True),
        pd.DataFrame(bootstrap_rows).sort_values(
            ["candidateId", "horizon", "targetType", "replicate"]
        ).reset_index(drop=True),
    )


def landmark_results(estimates: pd.DataFrame) -> pd.DataFrame:
    return (
        estimates.groupby(
            [
                "matrixRole",
                "candidateId",
                "horizon",
                "targetType",
                "completedFissionLandmark",
            ],
            as_index=False,
        )
        .agg(
            matrices=("matrixIndex", "nunique"),
            meanQ=("q", "mean"),
            medianQ=("q", "median"),
            minimumQ=("q", "min"),
            maximumQ=("q", "max"),
            dataInformedFraction=("dataInformed", "mean"),
            meanTrials=("trials", "mean"),
        )
        .sort_values(
            ["matrixRole", "candidateId", "targetType", "horizon", "completedFissionLandmark"]
        )
        .reset_index(drop=True)
    )


def _fit_logistic(
    train_x: np.ndarray, train_y: np.ndarray
) -> tuple[StandardScaler, LogisticRegression] | None:
    if len(np.unique(train_y)) < 2:
        return None
    scaler = StandardScaler().fit(train_x)
    model = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=5000,
        random_state=derived_seed("logistic") % (2**32 - 1),
    ).fit(scaler.transform(train_x), train_y)
    return scaler, model


def _predict(
    fitted: tuple[StandardScaler, LogisticRegression] | None,
    values: np.ndarray,
    prior: float,
) -> np.ndarray:
    if fitted is None:
        return np.full(len(values), prior)
    scaler, model = fitted
    return model.predict_proba(scaler.transform(values))[:, 1]


def prediction_results(
    estimates: pd.DataFrame, observed: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    joint = estimates[estimates["targetType"].eq("JOINT_BREAK_RUN3")].copy()
    joint = joint.merge(
        observed[
            ["stateId", "horizon", "observedJointEvent"]
        ],
        on=["stateId", "horizon"],
        validate="one_to_one",
    )
    rows = []
    models = []
    for candidate in CANDIDATES:
        for horizon in HORIZONS:
            train = joint[
                joint["candidateId"].eq(candidate)
                & joint["matrixRole"].eq("DEVELOPMENT")
                & joint["horizon"].eq(horizon)
            ].copy()
            validation = joint[
                joint["candidateId"].eq(candidate)
                & joint["matrixRole"].eq("VALIDATION")
                & joint["horizon"].eq(horizon)
            ].copy()
            y_train = train["observedJointEvent"].astype(int).to_numpy()
            prior = float((y_train.sum() + 0.5) / (len(y_train) + 1))
            control_fit = _fit_logistic(train[list(CONTROL_COLUMNS)].to_numpy(float), y_train)
            time_fit = _fit_logistic(train[["normalizedGeneration"]].to_numpy(float), y_train)
            q_train = np.clip(train["q"].to_numpy(float), 1e-6, 1 - 1e-6)
            combined_train = np.c_[
                train[list(CONTROL_COLUMNS)].to_numpy(float),
                np.log(q_train / (1 - q_train)),
            ]
            combined_fit = _fit_logistic(combined_train, y_train)
            models.extend(
                [
                    {
                        "candidateId": candidate,
                        "horizon": horizon,
                        "modelId": "PAST_CONTROLS",
                        "trainingMatrices": train["matrixIndex"].nunique(),
                        "trainingStates": len(train),
                        "trainingPositiveRate": float(y_train.mean()),
                        "featureCount": len(CONTROL_COLUMNS),
                    },
                    {
                        "candidateId": candidate,
                        "horizon": horizon,
                        "modelId": "PAST_PLUS_SHOOTING",
                        "trainingMatrices": train["matrixIndex"].nunique(),
                        "trainingStates": len(train),
                        "trainingPositiveRate": float(y_train.mean()),
                        "featureCount": len(CONTROL_COLUMNS) + 1,
                    },
                ]
            )
            for role, frame in (("DEVELOPMENT", train), ("VALIDATION", validation)):
                q = np.clip(frame["q"].to_numpy(float), 1e-6, 1 - 1e-6)
                controls = frame[list(CONTROL_COLUMNS)].to_numpy(float)
                combined = np.c_[controls, np.log(q / (1 - q))]
                probabilities = {
                    "DEVELOPMENT_PRIOR": np.full(len(frame), prior),
                    "TIME_ONLY": _predict(time_fit, frame[["normalizedGeneration"]].to_numpy(float), prior),
                    "PAST_CONTROLS": _predict(control_fit, controls, prior),
                    "SHOOTING_Q_JOINT": q,
                    "PAST_PLUS_SHOOTING": _predict(combined_fit, combined, prior),
                }
                for model_id, probability in probabilities.items():
                    for position, source in enumerate(frame.itertuples(index=False)):
                        rows.append(
                            {
                                "stateId": source.stateId,
                                "matrixRole": role,
                                "candidateId": candidate,
                                "matrixIndex": int(source.matrixIndex),
                                "completedFissionLandmark": int(source.completedFissionLandmark),
                                "horizon": horizon,
                                "modelId": model_id,
                                "probability": float(probability[position]),
                                "observedJointEvent": bool(source.observedJointEvent),
                                "fitOnDevelopmentOnly": model_id != "SHOOTING_Q_JOINT",
                                "usesForwardShooting": model_id in ("SHOOTING_Q_JOINT", "PAST_PLUS_SHOOTING"),
                            }
                        )
    return (
        pd.DataFrame(rows).sort_values(
            ["matrixRole", "candidateId", "horizon", "modelId", "matrixIndex", "completedFissionLandmark"]
        ).reset_index(drop=True),
        pd.DataFrame(models).sort_values(["candidateId", "horizon", "modelId"]).reset_index(drop=True),
    )


def _binary_metrics(frame: pd.DataFrame) -> dict[str, float]:
    y = frame["observedJointEvent"].astype(int).to_numpy()
    p = np.clip(frame["probability"].to_numpy(float), 1e-9, 1 - 1e-9)
    predicted = p >= 0.5
    return {
        "brier": float(brier_score_loss(y, p)),
        "logLoss": float(log_loss(y, p, labels=[0, 1])),
        "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "auprc": float(average_precision_score(y, p)) if y.sum() else float("nan"),
        "balancedAccuracy": float(balanced_accuracy_score(y, predicted)) if len(np.unique(y)) > 1 else float("nan"),
        "positiveRate": float(y.mean()),
        "meanPredictedProbability": float(p.mean()),
    }


def predictive_metrics(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    point_rows = []
    for (role, candidate, horizon, model), group in predictions.groupby(
        ["matrixRole", "candidateId", "horizon", "modelId"], sort=True
    ):
        point_rows.append(
            {
                "matrixRole": role,
                "candidateId": candidate,
                "horizon": int(horizon),
                "modelId": model,
                "matrices": group["matrixIndex"].nunique(),
                "states": len(group),
                **_binary_metrics(group),
            }
        )
    point = pd.DataFrame(point_rows).sort_values(
        ["matrixRole", "candidateId", "horizon", "modelId"]
    ).reset_index(drop=True)
    bootstrap_rows = []
    comparison_rows = []
    pairs = (
        ("SHOOTING_Q_JOINT", "DEVELOPMENT_PRIOR"),
        ("SHOOTING_Q_JOINT", "PAST_CONTROLS"),
        ("PAST_PLUS_SHOOTING", "PAST_CONTROLS"),
    )
    for candidate in CANDIDATES:
        for horizon in HORIZONS:
            group = predictions[
                predictions["matrixRole"].eq("VALIDATION")
                & predictions["candidateId"].eq(candidate)
                & predictions["horizon"].eq(horizon)
            ]
            matrices = sorted(group["matrixIndex"].unique())
            model_ids = sorted(group["modelId"].unique())
            per_matrix = {
                model: {
                    matrix: _binary_metrics(
                        group[group["modelId"].eq(model) & group["matrixIndex"].eq(matrix)]
                    )["brier"]
                    for matrix in matrices
                }
                for model in model_ids
            }
            rng = generator("predictive_bootstrap", candidate, horizon)
            effects: dict[tuple[str, str], list[float]] = {pair: [] for pair in pairs}
            for replicate in range(BOOTSTRAPS):
                sampled = rng.choice(matrices, size=len(matrices), replace=True)
                briers = {
                    model: float(np.mean([per_matrix[model][int(matrix)] for matrix in sampled]))
                    for model in model_ids
                }
                for model, brier in briers.items():
                    bootstrap_rows.append(
                        {
                            "candidateId": candidate,
                            "horizon": horizon,
                            "replicate": replicate,
                            "modelId": model,
                            "brier": brier,
                        }
                    )
                for pair in pairs:
                    effects[pair].append(briers[pair[1]] - briers[pair[0]])
            p = point[
                point["matrixRole"].eq("VALIDATION")
                & point["candidateId"].eq(candidate)
                & point["horizon"].eq(horizon)
            ].set_index("modelId")
            for model, reference in pairs:
                values = np.asarray(effects[(model, reference)])
                low, high = interval(values)
                comparison_rows.append(
                    {
                        "candidateId": candidate,
                        "horizon": horizon,
                        "modelId": model,
                        "referenceModelId": reference,
                        "brierImprovement": float(p.loc[reference, "brier"] - p.loc[model, "brier"]),
                        "brierImprovementLower95": low,
                        "brierImprovementUpper95": high,
                        "fractionBootstrapPositive": float((values > 0).mean()),
                    }
                )
    return (
        point,
        pd.DataFrame(bootstrap_rows).sort_values(
            ["candidateId", "horizon", "replicate", "modelId"]
        ).reset_index(drop=True),
        pd.DataFrame(comparison_rows).sort_values(
            ["candidateId", "horizon", "modelId", "referenceModelId"]
        ).reset_index(drop=True),
    )


def negative_control_results(
    predictions: pd.DataFrame, comparisons: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for candidate in CANDIDATES:
        for horizon in HORIZONS:
            group = predictions[
                predictions["matrixRole"].eq("VALIDATION")
                & predictions["candidateId"].eq(candidate)
                & predictions["horizon"].eq(horizon)
            ]
            shooting = group[group["modelId"].eq("SHOOTING_Q_JOINT")].copy()
            controls = group[group["modelId"].eq("PAST_CONTROLS")]
            control_brier = _binary_metrics(controls)["brier"]
            observed = float(
                comparisons[
                    comparisons["candidateId"].eq(candidate)
                    & comparisons["horizon"].eq(horizon)
                    & comparisons["modelId"].eq("SHOOTING_Q_JOINT")
                    & comparisons["referenceModelId"].eq("PAST_CONTROLS")
                ]["brierImprovement"].iloc[0]
            )
            matrix_ids = sorted(shooting["matrixIndex"].unique())
            trajectories = {
                matrix: shooting[shooting["matrixIndex"].eq(matrix)]
                .sort_values("completedFissionLandmark")["probability"]
                .to_numpy(float)
                for matrix in matrix_ids
            }
            rng = generator("matrix_permutation", "VALIDATION", candidate, horizon, PERMUTATIONS)
            null = []
            for replicate in range(PERMUTATIONS):
                donors = rng.permutation(matrix_ids)
                permuted = shooting.sort_values(
                    ["matrixIndex", "completedFissionLandmark"]
                ).copy()
                permuted["probability"] = np.concatenate(
                    [trajectories[int(donor)] for donor in donors]
                )
                effect = control_brier - _binary_metrics(permuted)["brier"]
                null.append(effect)
                rows.append(
                    {
                        "candidateId": candidate,
                        "horizon": horizon,
                        "controlId": "WHOLE_MATRIX_Q_TRAJECTORY_PERMUTATION",
                        "replicate": replicate,
                        "brierImprovementOverPastControls": effect,
                        "observedImprovement": observed,
                        "permutationP": np.nan,
                    }
                )
            p_value = float((1 + np.sum(np.asarray(null) >= observed)) / (PERMUTATIONS + 1))
            for row in rows:
                if row["candidateId"] == candidate and row["horizon"] == horizon:
                    row["permutationP"] = p_value
    return pd.DataFrame(rows).sort_values(
        ["candidateId", "horizon", "replicate"]
    ).reset_index(drop=True)


def clock_horizon_comparison(
    reliability: pd.DataFrame, metrics: pd.DataFrame
) -> pd.DataFrame:
    l49_rel = pd.read_parquet(L49R_ROOT / "committor_reliability_results.parquet")
    l49_met = pd.read_parquet(L49R_ROOT / "predictive_metric_results.parquet")
    rows = []
    for candidate in CANDIDATES:
        previous_rel = l49_rel[
            l49_rel["matrixRole"].eq("VALIDATION") & l49_rel["candidateId"].eq(candidate)
        ].iloc[0]
        previous_met = l49_met[
            l49_met["matrixRole"].eq("VALIDATION")
            & l49_met["candidateId"].eq(candidate)
            & l49_met["modelId"].eq("SHOOTING_Q_JOINT")
        ].iloc[0]
        rows.append(
            {
                "clockId": "L49R_MIXED_SELECTED_MOLECULAR",
                "candidateId": candidate,
                "horizon": 12,
                "targetType": "RUN3_GIVEN_BREAK",
                "matrices": int(previous_rel.matrices),
                "states": int(previous_rel.states),
                "splitHalfSpearman": previous_rel.stateSplitHalfSpearmanConditional,
                "centeredSplitHalfSpearman": previous_rel.centeredSplitHalfSpearmanConditional,
                "dataInformedFraction": previous_rel.dataInformedFraction,
                "shootingAuroc": previous_met.auroc,
                "descriptiveOnlyAcrossDifferentCohorts": True,
            }
        )
        for horizon in HORIZONS:
            current_rel = reliability[
                reliability["matrixRole"].eq("VALIDATION")
                & reliability["candidateId"].eq(candidate)
                & reliability["horizon"].eq(horizon)
                & reliability["targetType"].eq("JOINT_BREAK_RUN3")
            ].iloc[0]
            current_met = metrics[
                metrics["matrixRole"].eq("VALIDATION")
                & metrics["candidateId"].eq(candidate)
                & metrics["horizon"].eq(horizon)
                & metrics["modelId"].eq("SHOOTING_Q_JOINT")
            ].iloc[0]
            rows.append(
                {
                    "clockId": "L50_POST_FISSION_BOUNDARY",
                    "candidateId": candidate,
                    "horizon": horizon,
                    "targetType": "JOINT_BREAK_RUN3",
                    "matrices": int(current_rel.matrices),
                    "states": int(current_rel.states),
                    "splitHalfSpearman": current_rel.splitHalfSpearman,
                    "centeredSplitHalfSpearman": current_rel.centeredSplitHalfSpearman,
                    "dataInformedFraction": current_rel.dataInformedFraction,
                    "shootingAuroc": current_met.auroc,
                    "descriptiveOnlyAcrossDifferentCohorts": True,
                }
            )
    return pd.DataFrame(rows)


def scientific_gates(
    estimates: pd.DataFrame,
    reliability: pd.DataFrame,
    comparisons: pd.DataFrame,
    controls: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], str]:
    rows = []
    for candidate in CANDIDATES:
        for horizon in HORIZONS:
            row = reliability[
                reliability["matrixRole"].eq("VALIDATION")
                & reliability["candidateId"].eq(candidate)
                & reliability["horizon"].eq(horizon)
                & reliability["targetType"].eq("JOINT_BREAK_RUN3")
            ].iloc[0]
            passed = bool(
                row.dataInformedFraction == 1
                and row.intermediateStates >= 20
                and row.centeredSplitHalfSpearmanLower95 > 0.3
                and row.correctedWithinMatrixVarianceLower95 > 0
            )
            rows.append(
                {
                    "gateId": f"MEASUREMENT_H{horizon}::{candidate}",
                    "candidateId": candidate,
                    "horizon": horizon,
                    "gateFamily": "JOINT_PROCESS_MEASUREMENT",
                    "centeredSplitHalfLower95": row.centeredSplitHalfSpearmanLower95,
                    "correctedWithinVarianceLower95": row.correctedWithinMatrixVarianceLower95,
                    "intermediateStates": int(row.intermediateStates),
                    "observedJointEvents": int(
                        estimates[
                            estimates["matrixRole"].eq("VALIDATION")
                            & estimates["candidateId"].eq(candidate)
                            & estimates["horizon"].eq(horizon)
                            & estimates["targetType"].eq("JOINT_BREAK_RUN3")
                        ]["observedTarget"].fillna(False).sum()
                    ),
                    "brierImprovementOverPriorLower95": np.nan,
                    "brierImprovementOverControlsLower95": np.nan,
                    "permutationP": np.nan,
                    "passed": passed,
                }
            )
            candidate_comparisons = comparisons[
                comparisons["candidateId"].eq(candidate)
                & comparisons["horizon"].eq(horizon)
            ]
            prior = candidate_comparisons[
                candidate_comparisons["modelId"].eq("SHOOTING_Q_JOINT")
                & candidate_comparisons["referenceModelId"].eq("DEVELOPMENT_PRIOR")
            ].iloc[0]
            past = candidate_comparisons[
                candidate_comparisons["modelId"].eq("SHOOTING_Q_JOINT")
                & candidate_comparisons["referenceModelId"].eq("PAST_CONTROLS")
            ].iloc[0]
            permutation_p = float(
                controls[
                    controls["candidateId"].eq(candidate)
                    & controls["horizon"].eq(horizon)
                ]["permutationP"].iloc[0]
            )
            events = rows[-1]["observedJointEvents"]
            forecast_pass = bool(
                events >= 20
                and prior.brierImprovementLower95 > 0
                and past.brierImprovementLower95 > 0
                and permutation_p <= 0.05
            )
            rows.append(
                {
                    "gateId": f"FORECAST_H{horizon}::{candidate}",
                    "candidateId": candidate,
                    "horizon": horizon,
                    "gateFamily": "INDEPENDENT_OBSERVED_FUTURE",
                    "centeredSplitHalfLower95": np.nan,
                    "correctedWithinVarianceLower95": np.nan,
                    "intermediateStates": np.nan,
                    "observedJointEvents": events,
                    "brierImprovementOverPriorLower95": prior.brierImprovementLower95,
                    "brierImprovementOverControlsLower95": past.brierImprovementLower95,
                    "permutationP": permutation_p,
                    "passed": forecast_pass,
                }
            )
    gates = pd.DataFrame(rows)
    primary = gates[gates["horizon"].eq(PRIMARY_HORIZON)]
    measurement_pass = bool(
        primary[primary["gateFamily"].eq("JOINT_PROCESS_MEASUREMENT")]["passed"].all()
    )
    forecast_pass = bool(
        primary[primary["gateFamily"].eq("INDEPENDENT_OBSERVED_FUTURE")]["passed"].all()
    )
    conditional = reliability[
        reliability["matrixRole"].eq("VALIDATION")
        & reliability["horizon"].eq(PRIMARY_HORIZON)
        & reliability["targetType"].eq("RUN3_GIVEN_BREAK")
    ]
    conditional_pass = bool(
        conditional["dataInformedFraction"].ge(0.95).all()
        and conditional["centeredSplitHalfSpearmanLower95"].gt(0.3).all()
        and conditional["correctedWithinMatrixVarianceLower95"].gt(0).all()
    )
    if measurement_pass and forecast_pass:
        classifications = [
            "FISSION_ALIGNED_JOINT_PROCESS_COMMITTOR_ESTABLISHED",
            "SHOOTING_INCREMENTAL_FOR_ONLINE_HEREDITY_RENEWAL",
        ]
        classifications.append(
            "CONDITIONAL_RENEWAL_COMPONENT_IDENTIFIED"
            if conditional_pass
            else "JOINT_PROCESS_IDENTIFIABLE_CONDITIONAL_RENEWAL_NOT_UNIVERSAL"
        )
        classifications.append("PROMOTABLE_TO_UNTOUCHED_PROCESS_CONFIRMATION")
        next_theme = "L51_UNTOUCHED_FISSION_ALIGNED_PROCESS_COMMITTOR_CONFIRMATION"
    elif measurement_pass:
        classifications = [
            "FISSION_ALIGNED_JOINT_PROCESS_RISK_IDENTIFIED",
            "SHOOTING_NOT_INCREMENTAL_BEYOND_DIRECT_HISTORY",
            "NOT_PROMOTABLE_AS_CONFIRMED",
        ]
        next_theme = "L51_PROCESS_HAZARD_RENEWAL_BASELINE_AUDIT"
    else:
        classifications = [
            "PROCESS_EVENT_HORIZON_AND_PHASE_NOT_IDENTIFIED",
            "SHOOTING_REMAINS_CROSS_SECTIONAL_ENSEMBLE_MEASUREMENT",
            "NOT_PROMOTABLE_AS_CONFIRMED",
        ]
        next_theme = "L51_PROCESS_HAZARD_RENEWAL_BASELINE_AUDIT"
    gates = pd.concat(
        [
            gates,
            pd.DataFrame(
                [
                    {
                        "gateId": "COMPLETE_PRIMARY_F12",
                        "candidateId": "BOTH",
                        "horizon": PRIMARY_HORIZON,
                        "gateFamily": "COMPLETE",
                        "centeredSplitHalfLower95": np.nan,
                        "correctedWithinVarianceLower95": np.nan,
                        "intermediateStates": np.nan,
                        "observedJointEvents": int(
                            estimates[
                                estimates["matrixRole"].eq("VALIDATION")
                                & estimates["horizon"].eq(PRIMARY_HORIZON)
                                & estimates["targetType"].eq("JOINT_BREAK_RUN3")
                            ]["observedTarget"].fillna(False).sum()
                        ),
                        "brierImprovementOverPriorLower95": np.nan,
                        "brierImprovementOverControlsLower95": np.nan,
                        "permutationP": np.nan,
                        "passed": measurement_pass and forecast_pass,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    return gates, classifications, next_theme


def compute_tables(
    branches: pd.DataFrame,
    states: pd.DataFrame,
    observed: pd.DataFrame,
    validation: pd.DataFrame,
    selection: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], list[str], str]:
    estimates = state_estimates(branches, states, observed)
    reliability, reliability_bootstrap = reliability_results(estimates)
    landmarks = landmark_results(estimates)
    predictions, model_registry = prediction_results(estimates, observed)
    metrics, metric_bootstrap, comparisons = predictive_metrics(predictions)
    controls = negative_control_results(predictions, comparisons)
    clock_comparison = clock_horizon_comparison(reliability, metrics)
    gates, classifications, next_theme = scientific_gates(
        estimates, reliability, comparisons, controls
    )
    tables = {
        "matrix_selection_registry.parquet": selection,
        "restored_state_registry.parquet": states,
        "state_restoration_validation.parquet": validation,
        "observed_process_outcomes.parquet": observed,
        "branch_results.parquet": branches,
        "state_committor_results.parquet": estimates,
        "committor_reliability_results.parquet": reliability,
        "committor_reliability_bootstrap.parquet": reliability_bootstrap,
        "landmark_horizon_results.parquet": landmarks,
        "model_registry.parquet": model_registry,
        "prediction_results.parquet": predictions,
        "predictive_metric_results.parquet": metrics,
        "predictive_metric_bootstrap.parquet": metric_bootstrap,
        "paired_predictive_comparisons.parquet": comparisons,
        "negative_control_results.parquet": controls,
        "clock_horizon_comparison.parquet": clock_comparison,
        "scientific_gate_results.parquet": gates,
    }
    return tables, classifications, next_theme


def make_figures(tables: dict[str, pd.DataFrame]) -> None:
    root = BUILD_ROOT / "figures"
    root.mkdir(parents=True, exist_ok=True)
    landmarks = tables["landmark_horizon_results.parquet"]
    reliability = tables["committor_reliability_results.parquet"]
    estimates = tables["state_committor_results.parquet"]
    metrics = tables["predictive_metric_results.parquet"]
    comparisons = tables["paired_predictive_comparisons.parquet"]
    gates = tables["scientific_gate_results.parquet"]
    colors = {CANDIDATES[0]: "#4c78a8", CANDIDATES[1]: "#f58518"}

    validation = landmarks[
        landmarks["matrixRole"].eq("VALIDATION")
        & landmarks["targetType"].eq("JOINT_BREAK_RUN3")
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for axis, horizon in zip(axes, HORIZONS, strict=True):
        for candidate, group in validation[validation["horizon"].eq(horizon)].groupby("candidateId"):
            axis.plot(
                group["completedFissionLandmark"],
                group["meanQ"],
                marker="o",
                color=colors[candidate],
                label=f"C{candidate[-2:]}",
            )
        axis.set_title(f"F{horizon}")
        axis.set_xlabel("Completed fissions")
    axes[0].set_ylabel("Joint break + new run-3 probability")
    axes[0].legend()
    fig.suptitle("Fission-aligned process risk at fixed landmarks")
    fig.tight_layout()
    fig.savefig(root / "01_fission_aligned_risk_trajectories.png", dpi=160)
    plt.close(fig)

    primary = estimates[
        estimates["matrixRole"].eq("VALIDATION")
        & estimates["horizon"].eq(PRIMARY_HORIZON)
        & estimates["targetType"].eq("JOINT_BREAK_RUN3")
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis, candidate in zip(axes, CANDIDATES, strict=True):
        group = primary[primary["candidateId"].eq(candidate)]
        axis.scatter(group["qHalfA"], group["qHalfB"], alpha=0.5)
        axis.plot([0, 1], [0, 1], ls=":", color="black")
        row = reliability[
            reliability["matrixRole"].eq("VALIDATION")
            & reliability["candidateId"].eq(candidate)
            & reliability["horizon"].eq(PRIMARY_HORIZON)
            & reliability["targetType"].eq("JOINT_BREAK_RUN3")
        ].iloc[0]
        axis.set_title(f"C{candidate[-2:]} centered rho={row.centeredSplitHalfSpearman:.2f}")
        axis.set_xlabel("32-branch half A")
        axis.set_ylabel("32-branch half B")
    fig.suptitle("F12 joint-process split-half reliability")
    fig.tight_layout()
    fig.savefig(root / "02_joint_split_half_reliability.png", dpi=160)
    plt.close(fig)

    rel_plot = reliability[
        reliability["matrixRole"].eq("VALIDATION")
        & reliability["targetType"].eq("JOINT_BREAK_RUN3")
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for candidate, group in rel_plot.groupby("candidateId"):
        ax.plot(group["horizon"], group["centeredSplitHalfSpearman"], marker="o", color=colors[candidate], label=f"C{candidate[-2:]}")
    ax.axhline(0.5, ls=":", color="black")
    ax.set_xticks(HORIZONS)
    ax.set_xlabel("Future fission horizon")
    ax.set_ylabel("Centered split-half Spearman")
    ax.set_title("Risk-coordinate reliability across nested horizons")
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "03_horizon_reliability.png", dpi=160)
    plt.close(fig)

    heldout = metrics[
        metrics["matrixRole"].eq("VALIDATION") & metrics["horizon"].eq(PRIMARY_HORIZON)
    ]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    pivot = heldout.pivot(index="modelId", columns="candidateId", values="brier")
    pivot.plot(kind="bar", ax=ax, color=[colors[candidate] for candidate in pivot.columns])
    ax.set_ylabel("Brier score (lower is better)")
    ax.set_title("F12 held-out realized-process forecast")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(root / "04_f12_validation_brier.png", dpi=160)
    plt.close(fig)

    plot = comparisons[
        comparisons["modelId"].eq("SHOOTING_Q_JOINT")
        & comparisons["referenceModelId"].eq("PAST_CONTROLS")
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for candidate, group in plot.groupby("candidateId"):
        ax.errorbar(
            group["horizon"],
            group["brierImprovement"],
            yerr=np.vstack(
                [
                    group["brierImprovement"] - group["brierImprovementLower95"],
                    group["brierImprovementUpper95"] - group["brierImprovement"],
                ]
            ),
            marker="o",
            color=colors[candidate],
            label=f"C{candidate[-2:]}",
        )
    ax.axhline(0, ls=":", color="black")
    ax.set_xticks(HORIZONS)
    ax.set_xlabel("Future fission horizon")
    ax.set_ylabel("Brier improvement over past controls")
    ax.set_title("Incremental shooting value across fixed horizons")
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / "05_horizon_brier_increment.png", dpi=160)
    plt.close(fig)

    matrix = gates[gates["candidateId"].isin(CANDIDATES)].pivot(
        index=["gateFamily", "horizon"], columns="candidateId", values="passed"
    )
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(matrix.astype(float), vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(matrix.columns)), [f"C{x[-2:]}" for x in matrix.columns])
    ax.set_yticks(range(len(matrix.index)), [f"{family} F{h}" for family, h in matrix.index], fontsize=8)
    ax.set_title("Process-clock and horizon decision matrix")
    fig.colorbar(image, ax=ax, ticks=[0, 1])
    fig.tight_layout()
    fig.savefig(root / "06_decision_matrix.png", dpi=160)
    plt.close(fig)


def report_text(
    tables: dict[str, pd.DataFrame],
    classifications: list[str],
    next_theme: str,
    runtime: dict[str, Any],
) -> str:
    reliability = tables["committor_reliability_results.parquet"]
    reliability = reliability[
        reliability["matrixRole"].eq("VALIDATION")
        & reliability["targetType"].isin(["JOINT_BREAK_RUN3", "RUN3_GIVEN_BREAK"])
    ][
        [
            "candidateId",
            "horizon",
            "targetType",
            "states",
            "dataInformedFraction",
            "intermediateStates",
            "splitHalfSpearman",
            "centeredSplitHalfSpearman",
            "centeredSplitHalfSpearmanLower95",
            "correctedWithinMatrixVariance",
            "correctedWithinMatrixVarianceLower95",
        ]
    ]
    landmarks = tables["landmark_horizon_results.parquet"]
    landmarks = landmarks[
        landmarks["matrixRole"].eq("VALIDATION")
        & landmarks["targetType"].eq("JOINT_BREAK_RUN3")
    ][
        [
            "candidateId",
            "horizon",
            "completedFissionLandmark",
            "meanQ",
            "medianQ",
            "minimumQ",
            "maximumQ",
        ]
    ]
    metrics = tables["predictive_metric_results.parquet"]
    metrics = metrics[
        metrics["matrixRole"].eq("VALIDATION") & metrics["horizon"].eq(PRIMARY_HORIZON)
    ][
        [
            "candidateId",
            "modelId",
            "states",
            "brier",
            "logLoss",
            "auroc",
            "auprc",
            "balancedAccuracy",
            "positiveRate",
        ]
    ]
    comparisons = tables["paired_predictive_comparisons.parquet"]
    gates = tables["scientific_gate_results.parquet"]
    return f"""# S19-L50 Full Results — Fission-Aligned Process-Committor Horizon and Phase Identifiability

## Top summary

- **Research step:** `{VERSION}`
- **Completion status:** complete; additive exploratory simulator evidence
- **Artifacts written:** immutable/source/input/seed locks, 80 shared matrices, 800 exact post-fission states, {runtime['newBranchStreams']:,} F12 branches with nested F4/F8/F12 outcomes, exact branch replay, 4,096 matrix bootstraps, 512 whole-risk-trajectory permutations per candidate/horizon, six figures, report and hash manifests
- **Validation:** PASS — immutable S01–L49R baseline; nine fixtures; zero matrix overlap with L49R; exact post-fission generation and state restoration; seed firewall; two exact branch campaigns; exact analysis/report regeneration; runtime, storage and artifact hashes
- **Outcome classification:** {', '.join(f'`{value}`' for value in classifications)}
- **Lay summary:** L50 asks the same online process question at exact post-fission generations, so the observation clock now matches the event clock. It measures the probability of a future inheritance break followed by a new three-fission hereditary episode within four, eight and twelve future fissions, without choosing a horizon after seeing the results.
- **Recommended next action:** `{next_theme}` under the autonomous authorization through L65. S20, E02, author contact and intervention work remain inactive.

## Frozen question and design

L49R mixed molecular-update and post-fission states. Its joint process probability was reliable across branch halves, but its conditional renewal component was undefined in some highly stable matrices and its full gate failed. L50 prospectively aligns every state to completed fissions 20, 35, 50, 65 and 80. It excludes all L49R-selected matrices and uses 40 new development plus 40 new validation identities from the frozen L23 cohort. The fixed F4, F8 and F12 horizons are nested prefixes of the same branch and are never selected against outcome proximity.

The outcome remains online and destination-free: observe the first future strict `H<=0.9` parent/daughter break, then certify a new episode only after three consecutive strict `H>0.9` inherited fissions. The primary probability is the joint chance of break plus certification. Break probability and certification conditional on a break are retained separately.

## Measurement reliability

{reliability.to_markdown(index=False, floatfmt='.6f')}

The joint probability is defined for every state. The conditional component is shown separately because a state with no break in 64 branches has no empirical conditional-renewal trials; this is physical stability information, not missing data to impute.

## Fixed-landmark risk geometry

{landmarks.to_markdown(index=False, floatfmt='.6f')}

## F12 independent realized-future forecast

{metrics.to_markdown(index=False, floatfmt='.6f')}

## Registered Brier comparisons across horizons

{comparisons.to_markdown(index=False, floatfmt='.6f')}

Past controls are fitted only on development matrices and contain generation, full and recent inheritance frequency, trailing streak, latest parent/daughter H and fissions since the latest break. Whole five-state q trajectories are permuted among validation matrices as the registered alignment null.

## Scientific gates

{gates.to_markdown(index=False, floatfmt='.6f')}

## Interpretation boundary

A passing result supports only an online-defined, simulator-accessible process propensity. It does not establish a static biomarker, paper replication, causal-emergence mechanism, intervention effect or real-chemistry claim. F4/F8 results cannot replace the registered F12 primary gate. Any favorable result remains adaptive and requires a later untouched matrix confirmation.

## Runtime and provenance

- Repository lock: `{runtime['repositoryHead']}`.
- Workers: `{runtime['workers']}` with one numerical-library thread per worker; GPU hours: 0.
- Wall time: `{runtime['wallSeconds'] / 60:.3f}` minutes; worker CPU upper estimate: `{runtime['estimatedWorkerCpuHours']:.6f}` hours.
- New primary matrices/trajectories: 0/0; new branch streams: `{runtime['newBranchStreams']:,}`; exact branch campaigns: 2.
- Matrix bootstraps: {BOOTSTRAPS}; matrix-trajectory permutations: {PERMUTATIONS} per candidate/horizon.

## Limitations

The 80 matrices come from a previously generated but L50-outcome-unopened L23 cohort, so the loop is exploratory rather than a new-matrix confirmation. Five fixed generations cannot describe every regime switch. Branch estimates remain Monte Carlo measurements. The process detects renewed local compositional heredity, not restoration of an old composition or function. Repeated states within a catalytic matrix are dependent and are therefore resampled and permuted only as whole matrix trajectories.
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
        "schema": "eidosoma.e01.s19_l50.artifact_manifest.v1",
        "loopId": LOOP_ID,
        "files": rows,
        "fileCount": len(rows),
        "aggregateSha256": hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
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
            "beliefBeforeLoop": "L49R showed reliable joint branch-half probabilities but mixed molecular and post-fission phases, incomplete conditional-renewal trials, and no robust increment over direct history controls.",
            "failureOrAmbiguityTargeted": "Whether process-clock alignment and a fixed nested horizon distinguish a longitudinal committor from cross-sectional matrix propensity.",
            "informationGainRationale": "Exact post-fission states remove growth-phase mixing; nested F4/F8/F12 prefixes reveal horizon effects without choosing one after outcomes.",
            "learned": "L50 fission-clock, horizon, state, matrix, branch, control and gate contract locked before outcomes.",
            "ledgerSequence": sequence,
            "loopId": LOOP_ID,
            "motivatingEvidence": "Complete L49R result and reviewer clock-matching direction.",
            "proposedNextTest": "Measure joint, break and conditional renewal probabilities at five fixed post-fission generations.",
            "recordPhase": "PRE_LOOP_METHOD_LOCK",
            "remainingPlausibleHypotheses": "Fission-aligned state signal, horizon-specific signal, direct-history sufficiency or cross-sectional-only shooting.",
            "selectedHypotheses": "The process committor may be longitudinally identifiable only on its native fission clock.",
            "timestampUtc": timestamp,
            "weakenedHypotheses": "Another broad static feature family should precede clock alignment.",
        },
        {
            "appendOnly": True,
            "beliefBeforeLoop": "A fission-aligned process lead requires reliable F12 within-matrix q trajectories and independent forecast value beyond direct heredity history in both candidates.",
            "failureOrAmbiguityTargeted": "Clock, horizon, joint-versus-conditional target and independent realized-future value.",
            "informationGainRationale": "New branch streams on L49R-unselected matrices and whole-matrix uncertainty provide an independent exploratory adjudication.",
            "learned": ";".join(classifications),
            "ledgerSequence": sequence + 1,
            "loopId": LOOP_ID,
            "motivatingEvidence": "Complete L50 result.",
            "proposedNextTest": next_theme,
            "recordPhase": "POST_LOOP_RESULT_AUTONOMOUS_CONTINUATION",
            "remainingPlausibleHypotheses": next_theme,
            "selectedHypotheses": "Fission-aligned online process committor.",
            "timestampUtc": timestamp,
            "weakenedHypotheses": "Any registered L50 gate that failed.",
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
        + f"\n\n## {LOOP_ID} — fission-aligned process-committor horizons\n\n"
        + f"- **Learned:** {', '.join(classifications)}.\n"
        + f"- **Next:** `{next_theme}`.\n",
    )

    candidate_path = ARTIFACT_ROOT / "candidate_registry.parquet"
    candidates = pd.read_parquet(candidate_path)
    candidate = {
        "branchCount": BRANCHES,
        "bundleId": "L50_FISSION_ALIGNED_PROCESS_COMMITTOR",
        "candidateId": "S19-L50-FISSION-ALIGNED-PROCESS-COMMITTOR",
        "candidateSpecificSuccess": 0,
        "completedFitLeakage": 0,
        "computeEfficiency": 3,
        "crossCandidateDiscriminability": 5,
        "deterministicHReuse": 1,
        "explanatoryLeverage": 5,
        "frozenRank": 1,
        "independenceFromPriorOutcomeSelection": 4,
        "outcomeGuidedThresholdSelection": 0,
        "paperFingerprintSpecificity": 0,
        "proposedSpecification": "five fixed post-fission generations, nested F4/F8/F12 break-plus-run3 shooting probabilities and development-only history controls",
        "rankingScore": 28.0,
        "registryOrder": int(candidates["registryOrder"].max()) + 1,
        "selected": "PROMOTABLE_TO_UNTOUCHED_PROCESS_CONFIRMATION" in classifications,
        "selectionReason": "L49R_MIXED_PHASE_AND_CONDITIONAL_AVAILABILITY",
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
    source_additions = []
    for row in source_registry().itertuples(index=False):
        source_additions.append(
            {
                "commitOrVersion": None,
                "evidenceClass": row.evidenceClass,
                "finding": f"{row.finding}; L50 use: {row.frozenUse}",
                "licenseStatus": "PUBLIC_METADATA_OR_WORKSPACE_EVIDENCE",
                "redistributionStatus": "REFERENCE_ONLY",
                "repositoryIdentity": None,
                "retainedPath": None,
                "retrievalDate": timestamp[:10],
                "sha256": None,
                "sourceId": f"L50_{row.sourceId}",
                "sourceType": row.evidenceClass,
                "treeIdentity": None,
                "url": row.url,
            }
        )
    BASE.write_parquet(
        source_path,
        pd.concat([sources, pd.DataFrame(source_additions).reindex(columns=sources.columns)], ignore_index=True),
    )


def prepare_lock() -> None:
    LOOP_ROOT.mkdir(parents=True, exist_ok=True)
    if git("status", "--porcelain=v1"):
        raise RuntimeError("repository must be clean before L50 lock")
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/eidosoma/groups/42")
    if head != remote:
        raise RuntimeError("L50 local/remote commit mismatch")
    prior = validate_immutable_prior()
    fixtures = fixture_results()
    selection, expanded = select_matrices()
    branch_seeds = branch_seed_manifest(expanded.to_dict("records"))
    analysis_seeds = analysis_seed_manifest()
    firewall = seed_firewall(branch_seeds, analysis_seeds)
    manifest = pd.read_parquet(L23_ROOT / "input_trajectory_manifest.parquet")
    selected_matrices = set(selection["matrixIndex"].astype(int))
    input_rows = []
    for row in manifest[manifest["matrixIndex"].isin(selected_matrices)].itertuples(index=False):
        path = Path(row.cachePath)
        actual = sha256_file(path) if path.is_file() else None
        input_rows.append(
            {
                "candidateId": row.candidateId,
                "matrixIndex": int(row.matrixIndex),
                "cachePath": str(path),
                "expectedSha256": row.cacheSha256,
                "actualSha256": actual,
                "passed": actual == row.cacheSha256,
            }
        )
    input_validation = pd.DataFrame(input_rows)
    prior_runtime = json.loads((L49R_ROOT / "runtime_manifest.json").read_text())
    scale = len(branch_seeds) / float(prior_runtime["newBranchStreams"])
    projected_branch_seconds = (
        float(prior_runtime["branchSeconds"] + prior_runtime["replaySeconds"]) * scale
    )
    benchmark = {
        "schema": "eidosoma.e01.s19_l50.benchmark_projection.v1",
        "outcomeBlind": True,
        "basis": "frozen L49R exact F12 branch-campaign runtime scaled by registered branch count",
        "newBranchesPerPass": len(branch_seeds),
        "exactBranchPasses": 2,
        "projectedBranchWallHours": projected_branch_seconds / 3600,
        "projectedWorkerCpuHoursUpper": projected_branch_seconds * WORKERS / 3600,
        "workers": WORKERS,
        "status": "PASS" if projected_branch_seconds / 3600 < 60 else "FAIL",
    }
    if (
        not prior["unchanged"]
        or not fixtures["passed"].all()
        or not input_validation["passed"].all()
        or firewall["status"] != "PASS"
        or benchmark["status"] != "PASS"
        or len(selection) != 80
        or len(expanded) != 800
    ):
        raise RuntimeError("L50 preoutcome validation failed")
    shutil.copy2(CONFIG, LOOP_ROOT / "preregistration.yaml")
    BASE.atomic_text(
        LOOP_ROOT / "decision_record.md",
        "# S19-L50 decision record\n\n"
        "L49R showed that its joint branch probability was reproducible while its "
        "conditional-renewal availability and increment beyond direct controls failed; "
        "its fixed molecular landmarks also mixed growth phases. Before any L50 state "
        "or branch outcome is opened, this record freezes exact post-fission states at "
        "completed generations 20, 35, 50, 65 and 80, exactly 40 development and 40 "
        "validation matrices unused by L49R, 64 F12 branches per state, and nested F4, "
        "F8 and F12 scoring of the unchanged strict-H090 break-plus-run3 process. Joint, "
        "break and conditional probabilities remain separate. F12 is primary; no horizon "
        "may rescue another. Development-only direct-history controls, matrix bootstraps, "
        "whole-q-trajectory permutations and every gate are fixed. No centroid, Phi metric, "
        "new primary trajectory, architecture search or intervention is authorized.\n",
    )
    BASE.write_json(LOOP_ROOT / "immutable_prior_validation.json", prior)
    BASE.write_parquet(LOOP_ROOT / "fixture_results.parquet", fixtures)
    BASE.write_parquet(LOOP_ROOT / "matrix_selection_registry.parquet", selection)
    BASE.write_parquet(LOOP_ROOT / "state_selection_registry.parquet", expanded)
    BASE.write_parquet(LOOP_ROOT / "input_identity_validation.parquet", input_validation)
    BASE.write_parquet(LOOP_ROOT / "branch_seed_manifest.parquet", branch_seeds)
    BASE.write_parquet(LOOP_ROOT / "analysis_seed_manifest.parquet", analysis_seeds)
    BASE.write_json(LOOP_ROOT / "seed_firewall.json", firewall)
    BASE.write_json(LOOP_ROOT / "benchmark_projection.json", benchmark)
    sources = source_registry()
    BASE.write_parquet(LOOP_ROOT / "source_registry.parquet", sources)
    BASE.write_json(
        LOOP_ROOT / "source_snapshot_manifest.json",
        {
            "schema": "eidosoma.e01.s19_l50.source_snapshot_manifest.v1",
            "runnerSha256": sha256_file(RUNNER_PATH),
            "coreSha256": sha256_file(CORE_PATH),
            "configSha256": sha256_file(CONFIG),
            "paperSha256": "77a2ec2c0751839d8a2e10863ca803c6f8b61475bbc790f2bbdad2a38af04ae4",
            "sources": sources.to_dict("records"),
        },
    )
    locked_inputs = {
        "matrixSelection": LOOP_ROOT / "matrix_selection_registry.parquet",
        "stateSelection": LOOP_ROOT / "state_selection_registry.parquet",
        "inputValidation": LOOP_ROOT / "input_identity_validation.parquet",
        "branchSeeds": LOOP_ROOT / "branch_seed_manifest.parquet",
        "analysisSeeds": LOOP_ROOT / "analysis_seed_manifest.parquet",
        "seedFirewall": LOOP_ROOT / "seed_firewall.json",
        "benchmark": LOOP_ROOT / "benchmark_projection.json",
        "sourceSnapshot": LOOP_ROOT / "source_snapshot_manifest.json",
        "trajectoryManifest": L23_ROOT / "input_trajectory_manifest.parquet",
        "matrixFirewall": L24_ROOT / "matrix_firewall.parquet",
        "l49rManifest": L49R_ROOT / "artifact_manifest.json",
    }
    hashes = {name: sha256_file(path) for name, path in locked_inputs.items()}
    implementation = {
        "schema": "eidosoma.e01.s19_l50.implementation_lock.v1",
        "repositoryHead": head,
        "remoteHead": remote,
        "runnerSha256": sha256_file(RUNNER_PATH),
        "coreSha256": sha256_file(CORE_PATH),
        "configSha256": sha256_file(CONFIG),
        "sharedMatrices": 80,
        "states": 800,
        "branchesPerState": BRANCHES,
        "newBranchStreams": len(branch_seeds),
        "workers": WORKERS,
        "completedFissionLandmarks": list(GENERATIONS),
        "horizons": list(HORIZONS),
        "primaryHorizon": PRIMARY_HORIZON,
        "threshold": THRESHOLD,
        "requiredRun": REQUIRED_RUN,
        "matrixBootstraps": BOOTSTRAPS,
        "matrixPermutations": PERMUTATIONS,
        "controlColumns": list(CONTROL_COLUMNS),
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
        raise RuntimeError("L50 repository lock mismatch")
    prior = validate_immutable_prior()
    fixtures = fixture_results()
    locked_inputs = {
        "matrixSelection": LOOP_ROOT / "matrix_selection_registry.parquet",
        "stateSelection": LOOP_ROOT / "state_selection_registry.parquet",
        "inputValidation": LOOP_ROOT / "input_identity_validation.parquet",
        "branchSeeds": LOOP_ROOT / "branch_seed_manifest.parquet",
        "analysisSeeds": LOOP_ROOT / "analysis_seed_manifest.parquet",
        "seedFirewall": LOOP_ROOT / "seed_firewall.json",
        "benchmark": LOOP_ROOT / "benchmark_projection.json",
        "sourceSnapshot": LOOP_ROOT / "source_snapshot_manifest.json",
        "trajectoryManifest": L23_ROOT / "input_trajectory_manifest.parquet",
        "matrixFirewall": L24_ROOT / "matrix_firewall.parquet",
        "l49rManifest": L49R_ROOT / "artifact_manifest.json",
    }
    if any(
        sha256_file(path) != lock["lockedInputHashes"][name]
        for name, path in locked_inputs.items()
    ):
        raise RuntimeError("L50 locked input changed")
    if (
        not prior["unchanged"]
        or prior["aggregateSha256"] != lock["priorAggregateSha256"]
        or not fixtures["passed"].all()
        or sha256_file(RUNNER_PATH) != lock["runnerSha256"]
        or sha256_file(CORE_PATH) != lock["coreSha256"]
        or sha256_file(CONFIG) != lock["configSha256"]
    ):
        raise RuntimeError("L50 pre-execution validation failed")
    selection, expanded = select_matrices()
    if frame_hash(selection) != frame_hash(
        pd.read_parquet(LOOP_ROOT / "matrix_selection_registry.parquet")
    ):
        raise RuntimeError("L50 matrix selection regeneration failure")
    if frame_hash(expanded) != frame_hash(
        pd.read_parquet(LOOP_ROOT / "state_selection_registry.parquet")
    ):
        raise RuntimeError("L50 state selection regeneration failure")
    payloads, states, observed, state_validation = build_states(expanded)
    if BUILD_ROOT.exists():
        shutil.rmtree(BUILD_ROOT)
    BUILD_ROOT.mkdir(parents=True)

    branch_started = time.perf_counter()
    branches = execute_branches(payloads)
    branch_seconds = time.perf_counter() - branch_started
    replay_started = time.perf_counter()
    branch_replay = execute_branches(payloads)
    replay_seconds = time.perf_counter() - replay_started
    branch_exact = frame_hash(branches) == frame_hash(branch_replay)
    if not branch_exact:
        raise RuntimeError("L50 exact branch replay failure")
    seed_manifest = pd.read_parquet(LOOP_ROOT / "branch_seed_manifest.parquet")
    identity = branches.merge(
        seed_manifest[["stateId", "branchIndex", "branchIdentitySha256"]],
        on=["stateId", "branchIndex"],
        suffixes=("", "Expected"),
        validate="one_to_one",
    )
    if not identity["branchIdentitySha256"].eq(
        identity["branchIdentitySha256Expected"]
    ).all():
        raise RuntimeError("L50 branch identity mismatch")

    tables, classifications, next_theme = compute_tables(
        branches, states, observed, state_validation, selection
    )
    make_figures(tables)
    tables_again, classifications_again, next_theme_again = compute_tables(
        branches.copy(),
        states.copy(),
        observed.copy(),
        state_validation.copy(),
        selection.copy(),
    )
    table_exact = {
        name: frame_hash(frame) == frame_hash(tables_again[name])
        for name, frame in tables.items()
    }
    regeneration = {
        "schema": "eidosoma.e01.s19_l50.regeneration_validation.v1",
        "status": "PASS"
        if branch_exact
        and all(table_exact.values())
        and classifications == classifications_again
        and next_theme == next_theme_again
        else "FAIL",
        "branchCampaignExact": branch_exact,
        "tableExact": table_exact,
        "classificationExact": classifications == classifications_again,
        "nextThemeExact": next_theme == next_theme_again,
        "analysisPasses": 2,
    }
    if regeneration["status"] != "PASS":
        raise RuntimeError("L50 regeneration failure")
    for name, frame in tables.items():
        BASE.write_parquet(BUILD_ROOT / name, frame)
    BASE.write_json(
        BUILD_ROOT / "classification.json",
        {
            "schema": "eidosoma.e01.s19_l50.classification.v1",
            "classifications": classifications,
            "nextTheme": next_theme,
            "priorStatusesChanged": False,
            "promotableAsConfirmed": False,
            "newMatrices": 0,
            "newPrimaryTrajectories": 0,
            "newBranchStreams": len(branches),
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
        "schema": "eidosoma.e01.s19_l50.runtime.v1",
        "repositoryHead": lock["head"],
        "workers": WORKERS,
        "numericalLibraryThreadsPerWorker": 1,
        "gpuHours": 0,
        "wallSeconds": elapsed,
        "branchSeconds": branch_seconds,
        "replaySeconds": replay_seconds,
        "estimatedWorkerCpuHours": (branch_seconds + replay_seconds) * WORKERS / 3600,
        "sharedMatrices": len(selection),
        "restoredStates": len(states),
        "newBranchStreams": len(branches),
        "newMatrices": 0,
        "newPrimaryTrajectories": 0,
        "matrixBootstraps": BOOTSTRAPS,
        "matrixPermutationsPerCandidateHorizon": PERMUTATIONS,
        "exactBranchCampaigns": 2,
        "completedAtUtc": utc_now(),
    }
    if runtime["estimatedWorkerCpuHours"] > 90 or runtime["wallSeconds"] > 60 * 3600:
        raise RuntimeError("L50 runtime ceiling exceeded")
    BASE.write_json(BUILD_ROOT / "runtime_manifest.json", runtime)
    BASE.write_json(BUILD_ROOT / "regeneration_validation.json", regeneration)
    retained_bytes = sum(
        path.stat().st_size for path in BUILD_ROOT.rglob("*") if path.is_file()
    ) + sum(path.stat().st_size for path in LOOP_ROOT.iterdir() if path.is_file())
    storage = {
        "schema": "eidosoma.e01.s19_l50.storage_validation.v1",
        "status": "PASS" if retained_bytes <= 25 * 1024**3 else "FAIL",
        "retainedBytes": retained_bytes,
        "retainedGiBCeiling": 25,
        "temporaryGiBCeiling": 75,
    }
    BASE.write_json(BUILD_ROOT / "storage_validation.json", storage)
    report = report_text(tables, classifications, next_theme, runtime)
    if report != report_text(tables, classifications, next_theme, runtime):
        raise RuntimeError("L50 report regeneration failure")
    BASE.atomic_text(BUILD_ROOT / "research_step_full_results.md", report)
    BASE.atomic_text(BUILD_ROOT / "S19_L50_FULL_RESULTS.md", report)
    BASE.atomic_text(
        BUILD_ROOT / "loop_decision_summary.md",
        f"# S19-L50 decision summary\n\n**Classification:** {', '.join(classifications)}\n\n**Next:** `{next_theme}`.\n",
    )
    if storage["status"] != "PASS":
        raise RuntimeError("L50 storage ceiling exceeded")
    for path in (BUILD_ROOT / "figures").glob("*.png"):
        if not path.stat().st_size:
            raise RuntimeError(f"empty L50 figure: {path}")
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
        raise RuntimeError("L50 artifact manifest regeneration failure")

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
            "nextAuthorizedLoop": "S19-L51",
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
