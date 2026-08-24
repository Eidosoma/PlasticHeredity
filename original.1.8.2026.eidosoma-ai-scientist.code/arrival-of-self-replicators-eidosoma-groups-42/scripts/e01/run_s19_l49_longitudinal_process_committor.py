#!/usr/bin/env python3
"""Run S19-L49 longitudinal process-committor risk audit."""

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
from e01_onset_discovery.fission_clock_recurrence import simulate_fission_clock
from e01_onset_discovery.longitudinal_process_risk import (
    jeffreys_mean,
    score_new_hereditary_episode,
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


L48 = load_module(
    "e01_l49_l48_runner",
    ROOT / "scripts/e01/run_s19_l48_process_committor_shooting_efficiency.py",
)
L41 = L48.L41
L28 = L41.L28
BASE = L48.BASE

ARTIFACT_ROOT = Path("/artifacts/research_steps/S19")
LOOP_ROOT = ARTIFACT_ROOT / "loops/L49"
L23_ROOT = ARTIFACT_ROOT / "loops/L23"
L24_ROOT = ARTIFACT_ROOT / "loops/L24"
L48_ROOT = ARTIFACT_ROOT / "loops/L48"
BUILD_ROOT = Path("/cache/e01_s19_l49/build")
CONFIG = ROOT / "configs/e01/s19_l49_longitudinal_process_committor.yaml"
RUNNER_PATH = Path(__file__).resolve()
CORE_PATH = ROOT / "src/e01_onset_discovery/longitudinal_process_risk.py"

LOOP_ID = "S19-L49"
VERSION = "E01-S19-L49-LONGITUDINAL-PROCESS-COMMITTOR-RISK-TRAJECTORY-v1.0.0"
CANDIDATES = ("S12F-CANDIDATE-02", "S12F-CANDIDATE-03")
ROLES = ("DEVELOPMENT", "VALIDATION")
LANDMARKS = (64, 96, 128, 160, 192)
MATRICES_PER_ROLE = 20
BRANCHES = 64
HALF = 32
FISSION_HORIZON = 12
THRESHOLD = 0.9
REQUIRED_RUN = 3
BOOTSTRAPS = 4096
PERMUTATIONS = 512
WORKERS = min(8, os.cpu_count() or 1)
SEED_ROOT = bytes.fromhex(
    "7ce4281f7613ee8b97ea89df3c60a0f924342523125a211826ff7ea1cd515c4f"
)
CONTROL_COLUMNS = (
    "normalizedLandmark",
    "currentMass",
    "distanceToFissionMass",
    "currentGenerationLocalStep",
    "currentCompletedFissions",
    "prefixInheritanceFraction",
    "prefixTrailingInheritanceRun",
    "latestParentDaughterH",
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
    prior = L48.validate_immutable_prior()
    manifest = json.loads((L48_ROOT / "artifact_manifest.json").read_text())
    rows = []
    for row in manifest["files"]:
        path = L48_ROOT / row["path"]
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
        "schema": "eidosoma.e01.s19_l49.immutable_prior_validation.v1",
        "status": "PASS" if passed else "FAIL",
        "unchanged": passed,
        "priorThroughL47Unchanged": bool(prior["unchanged"]),
        "validatedL48ArtifactCount": len(rows),
        "aggregateSha256": hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "rows": rows,
    }


def source_registry() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sourceId": "E_VANDEN_EIJNDEN_TRANSITION_PATH_THEORY",
                "evidenceClass": "PRIMARY_METHOD_SOURCE",
                "finding": "A forward committor is the probability that a trajectory initiated at a state reaches a defined future set before the competing set.",
                "frozenUse": "interpret branch shooting as a state-conditioned probability, not a static physical biomarker",
                "url": "https://doi.org/10.1007/978-3-540-79537-1_13",
            },
            {
                "sourceId": "LOUWERSE_SIVAK_2022",
                "evidenceClass": "PRIMARY_METHOD_SOURCE",
                "finding": "The committor is the transition-path probability and a quantitative reaction-coordinate benchmark.",
                "frozenUse": "test whether the process probability changes longitudinally and predicts an independent realized future",
                "url": "https://doi.org/10.1103/PhysRevLett.128.170602",
            },
            {
                "sourceId": "L48_BRANCH_BUDGET",
                "evidenceClass": "DIRECT_FROZEN_E01_RESULT",
                "finding": "All 64 estimator-half branches were required by the conservative cross-candidate probability-scoring contract.",
                "frozenUse": "exactly 64 new branches per longitudinal state; no reduced budget or adaptive allocation",
                "url": None,
            },
            {
                "sourceId": "REVIEWER_PROCESS_FIRST",
                "evidenceClass": "HUMAN_REVIEW_DIRECTION",
                "finding": "Separate inheritance, break, resumption and new hereditary regimes and treat stochastic shooting as a legitimate computational measurement when static proxies fail.",
                "frozenUse": "break, conditional run-three and joint probabilities are retained separately",
                "url": None,
            },
        ]
    )


def fixture_results() -> pd.DataFrame:
    certified = score_new_hereditary_episode([0.95, 0.2, 0.91, 0.92, 0.93])
    uninterrupted = score_new_hereditary_episode(np.full(12, 0.95))
    strict = score_new_hereditary_episode([0.9, 0.91, 0.92, 0.93])
    ordered = score_new_hereditary_episode([0.2, 0.91, 0.92, 0.93])
    reversed_order = score_new_hereditary_episode([0.91, 0.92, 0.93, 0.2])

    state = np.zeros(100, dtype=np.int64)
    state[:40] = 1
    beta = np.exp(np.full((100, 100), -4.0, dtype=np.float64))
    restored = RestoredState(tuple(map(int, state)), "initial_selected_state", 0, 1, 0, 0)

    def simulate() -> Any:
        streams = [np.random.Generator(np.random.PCG64DXSM(4900 + i)) for i in range(4)]
        return simulate_fission_clock(
            restored=restored,
            beta=beta,
            definition=L28.definition(CANDIDATES[0]),
            event_rng=streams[0],
            trim_rng=streams[1],
            fission_rng=streams[2],
            daughter_rng=streams[3],
            future_fissions=FISSION_HORIZON,
        )

    first = simulate()
    replay = simulate()
    return pd.DataFrame(
        [
            {"fixtureId": "F01_BREAK_RUN3_CERTIFIES", "passed": certified.event and certified.certification_boundary_one_based == 5},
            {"fixtureId": "F02_UNINTERRUPTED_EXCLUDED", "passed": not uninterrupted.break_observed and not uninterrupted.event},
            {"fixtureId": "F03_STRICT_H090", "passed": strict.break_observed and strict.event},
            {"fixtureId": "F04_TEMPORAL_ORDER", "passed": ordered.event and not reversed_order.event},
            {"fixtureId": "F05_TRAILING_RUN", "passed": trailing_true_run([False, True, True]) == 2},
            {"fixtureId": "F06_JEFFREYS", "passed": jeffreys_mean(3, 4) == 0.7 and jeffreys_mean(0, 0) == 0.5},
            {"fixtureId": "F07_SYNTHETIC_BRANCH_REPLAY", "passed": first == replay and first.fissions == FISSION_HORIZON},
            {"fixtureId": "F08_SEED_REPLAY", "passed": derived_seed("fixture", np.int64(3)) == derived_seed("fixture", 3)},
            {"fixtureId": "F09_SCOPE", "passed": LANDMARKS == (64, 96, 128, 160, 192) and BRANCHES == 64 and FISSION_HORIZON == 12},
        ]
    )


def select_matrices() -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_parquet(L23_ROOT / "input_trajectory_manifest.parquet")
    firewall = pd.read_parquet(L24_ROOT / "matrix_firewall.parquet")
    eligible = manifest[
        manifest["terminalStatus"].eq("requested_fissions_completed")
        & manifest["completedFissions"].ge(100)
        & manifest["selectedClockLength"].gt(max(LANDMARKS) + 1)
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
            firewall["matrixRole"].eq(role) & firewall["matrixIndex"].isin(shared)
        ].copy()
        pool["selectionDigest"] = pool["matrixIndex"].map(
            lambda matrix, role=role: hashlib.sha256(
                f"{VERSION}|STATE_SELECTION|{role}|{int(matrix)}".encode()
            ).hexdigest()
        )
        pool = pool.sort_values(["selectionDigest", "matrixIndex"])
        if len(pool) < MATRICES_PER_ROLE:
            raise RuntimeError("insufficient shared longitudinal matrices")
        for rank, row in enumerate(pool.head(MATRICES_PER_ROLE).itertuples(), start=1):
            rows.append(
                {
                    "matrixRole": role,
                    "matrixIndex": int(row.matrixIndex),
                    "selectionRank": rank,
                    "selectionDigest": row.selectionDigest,
                    "eligibleSharedPool": len(pool),
                    "selectedBeforeBranchOutcome": True,
                }
            )
    selected = pd.DataFrame(rows).sort_values(["matrixRole", "selectionRank"]).reset_index(drop=True)
    expanded = pd.DataFrame(
        [
            {
                **row._asdict(),
                "candidateId": candidate,
                "landmark": landmark,
            }
            for row in selected.itertuples(index=False)
            for candidate in CANDIDATES
            for landmark in LANDMARKS
        ]
    ).sort_values(["matrixRole", "candidateId", "matrixIndex", "landmark"]).reset_index(drop=True)
    expanded["stateId"] = expanded.apply(
        lambda row: hashlib.sha256(
            f"{VERSION}|{row.matrixRole}|{row.candidateId}|{int(row.matrixIndex)}|{int(row.landmark)}".encode()
        ).hexdigest()[:24],
        axis=1,
    )
    if len(selected) != 40 or len(expanded) != 400:
        raise RuntimeError("L49 state-selection cardinality failure")
    return selected, expanded


def _future_boundaries(selected: tuple[Any, ...], current_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    states: list[tuple[int, ...]] = []
    scores: list[float] = []
    generations: list[int] = []
    for index in range(current_index + 1, len(selected)):
        observation = selected[index]
        if observation.observation_kind != "post_fission":
            continue
        if index == 0 or selected[index - 1].observation_kind != "molecular_update":
            raise RuntimeError("future boundary predecessor mismatch")
        parent = np.asarray(selected[index - 1].state, dtype=np.int64)
        daughter = np.asarray(observation.state, dtype=np.int64)
        states.append(tuple(map(int, daughter)))
        scores.append(cosine_h(parent, daughter))
        generations.append(int(observation.completed_fissions))
        if len(states) == FISSION_HORIZON:
            break
    if len(states) != FISSION_HORIZON:
        raise RuntimeError("fewer than 12 frozen future fissions")
    return (
        np.asarray(states, dtype=np.int64),
        np.asarray(scores, dtype=np.float64),
        np.asarray(generations, dtype=np.int64),
    )


def build_states(expanded: pd.DataFrame) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
            raise RuntimeError("L49 beta identity mismatch")
        for row in group.sort_values("landmark").itertuples(index=False):
            current_index = int(row.landmark) - 1
            current = selected[current_index]
            restored = L28.restored_state_from_observation(current)
            boundary_indices = [
                index
                for index, item in enumerate(selected[: current_index + 1])
                if item.observation_kind == "post_fission"
            ]
            if not boundary_indices:
                raise RuntimeError("L49 prefix has no fission boundary")
            prefix_h = []
            for index in boundary_indices:
                if index == 0 or selected[index - 1].observation_kind != "molecular_update":
                    raise RuntimeError("L49 prefix boundary predecessor mismatch")
                prefix_h.append(
                    cosine_h(
                        np.asarray(selected[index - 1].state, dtype=np.int64),
                        np.asarray(selected[index].state, dtype=np.int64),
                    )
                )
            inherited = np.asarray(prefix_h, dtype=np.float64) > THRESHOLD
            latest_prefix = np.asarray(selected[boundary_indices[-1]].state, dtype=np.int64)
            future_states, future_h, future_generations = _future_boundaries(
                selected, current_index
            )
            observed = score_new_hereditary_episode(
                future_h, threshold=THRESHOLD, required_run=REQUIRED_RUN
            )
            state_id = row.stateId
            restored_state = np.asarray(restored.state, dtype=np.int64)
            base = {
                "stateId": state_id,
                "matrixRole": role,
                "candidateId": candidate,
                "matrixIndex": int(matrix),
                "landmark": int(row.landmark),
                "selectionRank": int(row.selectionRank),
                "trajectoryId": source.trajectoryId,
                "currentSelectedIndex": current_index,
                "currentObservationKind": current.observation_kind,
                "currentCompletedFissions": int(current.completed_fissions),
                "currentGrowthGeneration": int(current.growth_generation_one_based),
                "currentGenerationLocalStep": int(current.generation_local_step),
                "currentBatchStep": int(current.batch_step),
                "currentMass": int(restored_state.sum()),
                "distanceToFissionMass": int(max(0, 80 - restored_state.sum())),
                "normalizedLandmark": float(row.landmark / max(LANDMARKS)),
                "prefixBoundaryCount": len(boundary_indices),
                "prefixInheritanceFraction": float(inherited.mean()),
                "prefixTrailingInheritanceRun": trailing_true_run(inherited),
                "latestParentDaughterH": float(prefix_h[-1]),
                "currentStateSha256": L28.array_sha256(restored_state),
                "latestPrefixDaughterSha256": L28.array_sha256(latest_prefix),
                "betaSha256": beta_hash,
                "trajectorySha256": source.trajectorySha256,
                "selectedClockLength": len(selected),
                "targetUsesCompletedTestTrajectory": False,
            }
            state_rows.append(base)
            observed_rows.append(
                {
                    "stateId": state_id,
                    "matrixRole": role,
                    "candidateId": candidate,
                    "matrixIndex": int(matrix),
                    "landmark": int(row.landmark),
                    "observedBreak": observed.break_observed,
                    "observedJointEvent": observed.event,
                    "observedConditionalEvent": observed.event if observed.break_observed else None,
                    "observedBreakBoundaryOneBased": observed.break_boundary_one_based,
                    "observedCertificationBoundaryOneBased": observed.certification_boundary_one_based,
                    "observedPostbreakOpportunities": observed.postbreak_opportunities,
                    "observedFutureInheritanceFraction": float((future_h > THRESHOLD).mean()),
                    "futureFissionHorizon": FISSION_HORIZON,
                    "targetUsesCompletedTestTrajectory": False,
                }
            )
            validation_rows.append(
                {
                    "stateId": state_id,
                    "trajectoryIdentityPassed": trajectory.trajectory_sha256 == source.trajectorySha256,
                    "trajectoryCacheIdentityPassed": sha256_file(Path(source.cachePath)) == source.cacheSha256,
                    "betaIdentityPassed": beta_hash == source.betaSha256,
                    "selectedClockIdentityPassed": len(selected) == int(source.selectedClockLength),
                    "restoredStateExact": L28.array_sha256(restored_state) == L28.array_sha256(np.asarray(current.state, dtype=np.int64)),
                    "futureFissionCardinalityPassed": len(future_states) == FISSION_HORIZON,
                    "futureGenerationOrderPassed": bool(np.all(np.diff(future_generations) > 0)),
                    "selectedBeforeBranchOutcome": True,
                }
            )
            payloads.append(
                {
                    **base,
                    "state": list(map(int, restored.state)),
                    "latestPrefixDaughter": list(map(int, latest_prefix)),
                }
            )
    states = pd.DataFrame(state_rows).sort_values(
        ["matrixRole", "candidateId", "matrixIndex", "landmark"]
    ).reset_index(drop=True)
    observed = pd.DataFrame(observed_rows).sort_values(
        ["matrixRole", "candidateId", "matrixIndex", "landmark"]
    ).reset_index(drop=True)
    validation = pd.DataFrame(validation_rows).sort_values("stateId").reset_index(drop=True)
    checks = [column for column in validation if column.endswith("Passed") or column in ("restoredStateExact", "selectedBeforeBranchOutcome")]
    if len(payloads) != 400 or len(states) != 400 or not validation[checks].all().all():
        raise RuntimeError("L49 state restoration validation failure")
    return payloads, states, observed, validation


def branch_seed_manifest(payloads: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for payload in payloads:
        for branch in range(BRANCHES):
            row = {
                "stateId": payload["stateId"],
                "matrixRole": payload["matrixRole"],
                "candidateId": payload["candidateId"],
                "matrixIndex": int(payload["matrixIndex"]),
                "landmark": int(payload["landmark"]),
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
    result = pd.DataFrame(rows).sort_values(
        ["matrixRole", "candidateId", "matrixIndex", "landmark", "branchIndex"]
    ).reset_index(drop=True)
    if len(result) != 400 * BRANCHES or result["branchIdentitySha256"].duplicated().any():
        raise RuntimeError("L49 branch seed scope failure")
    return result


def analysis_seed_manifest() -> pd.DataFrame:
    rows = []
    for purpose in ("matrix_bootstrap", "matrix_permutation"):
        repetitions = BOOTSTRAPS if purpose == "matrix_bootstrap" else PERMUTATIONS
        for role in ROLES:
            for candidate in CANDIDATES:
                parts = (purpose, role, candidate, repetitions)
                rows.append(
                    {
                        "purpose": purpose,
                        "matrixRole": role,
                        "candidateId": candidate,
                        "repetitions": repetitions,
                        "rootHex": SEED_ROOT.hex(),
                        "derivedSeed": str(derived_seed(*parts)),
                        "seedMaterialSha256": seed_material(*parts).hex(),
                    }
                )
    return pd.DataFrame(rows).sort_values(["purpose", "matrixRole", "candidateId"]).reset_index(drop=True)


def seed_firewall(branches: pd.DataFrame, analysis: pd.DataFrame) -> dict[str, Any]:
    prior_material: set[str] = set()
    prior_derived: set[str] = set()
    for path in ARTIFACT_ROOT.glob("loops/L*/**/*seed*manifest*.parquet"):
        if "/L49/" in str(path):
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
        "schema": "eidosoma.e01.s19_l49.seed_firewall.v1",
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
        raise RuntimeError(f"L49 worker beta mismatch: {payload['stateId']}")
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
            future_fissions=FISSION_HORIZON,
        )
        process = score_new_hereditary_episode(
            trace.parent_daughter_h,
            threshold=THRESHOLD,
            required_run=REQUIRED_RUN,
        )
        materials = [
            seed_material("branch", payload["stateId"], branch, purpose).hex()
            for purpose in ("event", "trim", "fission", "daughter")
        ]
        rows.append(
            {
                "stateId": payload["stateId"],
                "matrixRole": payload["matrixRole"],
                "candidateId": payload["candidateId"],
                "matrixIndex": int(payload["matrixIndex"]),
                "landmark": int(payload["landmark"]),
                "branchIndex": branch,
                "branchHalf": "A" if branch < HALF else "B",
                "branchIdentitySha256": hashlib.sha256(
                    "|".join([payload["stateId"], str(branch), *materials]).encode()
                ).hexdigest(),
                "breakObserved": process.break_observed,
                "jointEvent": process.event,
                "conditionalEvent": process.event if process.break_observed else None,
                "breakBoundaryOneBased": process.break_boundary_one_based,
                "certificationBoundaryOneBased": process.certification_boundary_one_based,
                "postbreakOpportunities": process.postbreak_opportunities,
                "postbreakInheritedCount": process.postbreak_inherited_count,
                "maximumPostbreakRun": process.maximum_postbreak_run,
                "futureInheritanceFraction": float(np.mean(np.asarray(trace.parent_daughter_h) > THRESHOLD)),
                "molecularUpdates": trace.molecular_updates,
                "fissions": trace.fissions,
                "terminalStatus": trace.terminal_status,
                "pathSha256": trace.path_sha256,
                "targetUsesCompletedTestTrajectory": False,
            }
        )
    return rows


def execute_branches(payloads: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(_branch_worker, payload): payload["stateId"] for payload in payloads}
        for future in as_completed(futures):
            rows.extend(future.result())
    result = pd.DataFrame(rows).sort_values(
        ["matrixRole", "candidateId", "matrixIndex", "landmark", "branchIndex"]
    ).reset_index(drop=True)
    if (
        len(result) != len(payloads) * BRANCHES
        or result.duplicated(["stateId", "branchIndex"]).any()
        or result.groupby("stateId").size().ne(BRANCHES).any()
        or result["targetUsesCompletedTestTrajectory"].any()
    ):
        raise RuntimeError("L49 branch output scope failure")
    return result


def state_estimates(
    branches: pd.DataFrame, states: pd.DataFrame, observed: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for state_id, group in branches.groupby("stateId", sort=False):
        first = group.iloc[0]
        halves: dict[str, dict[str, float]] = {}
        for half in ("A", "B"):
            part = group[group["branchHalf"].eq(half)]
            conditional = part[part["breakObserved"]]
            halves[half] = {
                "qBreak": jeffreys_mean(int(part["breakObserved"].sum()), len(part)),
                "qJoint": jeffreys_mean(int(part["jointEvent"].sum()), len(part)),
                "qConditional": jeffreys_mean(
                    int(conditional["jointEvent"].sum()), len(conditional)
                ),
                "conditionalTrials": len(conditional),
            }
        conditional = group[group["breakObserved"]]
        conditional_trials = len(conditional)
        conditional_successes = int(conditional["jointEvent"].sum())
        raw_conditional = (
            conditional_successes / conditional_trials
            if conditional_trials
            else float("nan")
        )
        conditional_noise = (
            raw_conditional * (1 - raw_conditional) / conditional_trials
            if conditional_trials
            else float("nan")
        )
        rows.append(
            {
                "stateId": state_id,
                "matrixRole": first.matrixRole,
                "candidateId": first.candidateId,
                "matrixIndex": int(first.matrixIndex),
                "landmark": int(first.landmark),
                "branches": len(group),
                "breakSuccesses": int(group["breakObserved"].sum()),
                "jointSuccesses": int(group["jointEvent"].sum()),
                "conditionalTrials": conditional_trials,
                "conditionalSuccesses": conditional_successes,
                "conditionalDataInformed": conditional_trials > 0,
                "qBreak": jeffreys_mean(int(group["breakObserved"].sum()), len(group)),
                "qJoint": jeffreys_mean(int(group["jointEvent"].sum()), len(group)),
                "qConditional": jeffreys_mean(
                    conditional_successes, conditional_trials
                ),
                "qBreakHalfA": halves["A"]["qBreak"],
                "qBreakHalfB": halves["B"]["qBreak"],
                "qJointHalfA": halves["A"]["qJoint"],
                "qJointHalfB": halves["B"]["qJoint"],
                "qConditionalHalfA": halves["A"]["qConditional"],
                "qConditionalHalfB": halves["B"]["qConditional"],
                "conditionalTrialsHalfA": int(halves["A"]["conditionalTrials"]),
                "conditionalTrialsHalfB": int(halves["B"]["conditionalTrials"]),
                "conditionalBinomialNoise": conditional_noise,
                "meanMolecularUpdates": float(group["molecularUpdates"].mean()),
                "meanFutureInheritanceFraction": float(
                    group["futureInheritanceFraction"].mean()
                ),
                "completionFraction": float(group["fissions"].eq(FISSION_HORIZON).mean()),
                "targetUsesCompletedTestTrajectory": False,
            }
        )
    estimates = pd.DataFrame(rows)
    result = (
        states.merge(estimates, on=["stateId", "matrixRole", "candidateId", "matrixIndex", "landmark"], validate="one_to_one")
        .merge(observed, on=["stateId", "matrixRole", "candidateId", "matrixIndex", "landmark"], validate="one_to_one")
        .sort_values(["matrixRole", "candidateId", "matrixIndex", "landmark"])
        .reset_index(drop=True)
    )
    if len(result) != 400 or result["branches"].ne(BRANCHES).any():
        raise RuntimeError("L49 state estimate scope failure")
    return result


def _center_within_matrix(frame: pd.DataFrame, column: str) -> np.ndarray:
    return (
        frame[column]
        - frame.groupby("_matrixUnit", sort=False)[column].transform("mean")
    ).to_numpy(dtype=np.float64)


def _reliability_metrics(frame: pd.DataFrame) -> dict[str, float]:
    data = frame.copy()
    if "_matrixUnit" not in data:
        data["_matrixUnit"] = data["matrixIndex"].astype(str)
    q = data["qConditional"].to_numpy(dtype=np.float64)
    noise = data["conditionalBinomialNoise"].to_numpy(dtype=np.float64)
    finite = np.isfinite(q) & np.isfinite(noise)
    observed_variance = float(np.var(q[finite], ddof=1)) if finite.sum() > 1 else float("nan")
    mean_noise = float(np.mean(noise[finite])) if finite.any() else float("nan")
    within_values = []
    for _, group in data.groupby("_matrixUnit", sort=False):
        valid = group[np.isfinite(group["qConditional"]) & np.isfinite(group["conditionalBinomialNoise"])]
        if len(valid) >= 2:
            within_values.append(
                float(np.var(valid["qConditional"], ddof=1) - valid["conditionalBinomialNoise"].mean())
            )
    return {
        "stateSplitHalfSpearmanConditional": safe_spearman(
            data["qConditionalHalfA"].to_numpy(float),
            data["qConditionalHalfB"].to_numpy(float),
        ),
        "stateSplitHalfSpearmanJoint": safe_spearman(
            data["qJointHalfA"].to_numpy(float),
            data["qJointHalfB"].to_numpy(float),
        ),
        "centeredSplitHalfSpearmanConditional": safe_spearman(
            _center_within_matrix(data, "qConditionalHalfA"),
            _center_within_matrix(data, "qConditionalHalfB"),
        ),
        "correctedBetweenStateVariance": observed_variance - mean_noise,
        "correctedWithinMatrixVariance": float(np.mean(within_values)) if within_values else float("nan"),
        "meanAbsoluteSuccessiveDelta": float(
            data.sort_values(["_matrixUnit", "landmark"])
            .groupby("_matrixUnit")["qConditional"]
            .diff()
            .abs()
            .mean()
        ),
        "intermediateStates": float(data["qConditional"].between(0.1, 0.9, inclusive="neither").sum()),
        "dataInformedFraction": float(data["conditionalDataInformed"].mean()),
    }


def reliability_results(estimates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    bootstrap_rows = []
    for (role, candidate), group in estimates.groupby(["matrixRole", "candidateId"], sort=True):
        point = _reliability_metrics(group)
        matrices = sorted(group["matrixIndex"].unique())
        rng = generator("matrix_bootstrap", role, candidate, BOOTSTRAPS)
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
                    "matrixRole": role,
                    "candidateId": candidate,
                    "replicate": replicate,
                    **metrics,
                }
            )
        row = {
            "matrixRole": role,
            "candidateId": candidate,
            "matrices": len(matrices),
            "states": len(group),
            **point,
        }
        for key, values in draws.items():
            low, high = interval(np.asarray(values))
            row[f"{key}Lower95"] = low
            row[f"{key}Upper95"] = high
        rows.append(row)
    return (
        pd.DataFrame(rows).sort_values(["matrixRole", "candidateId"]).reset_index(drop=True),
        pd.DataFrame(bootstrap_rows).sort_values(["matrixRole", "candidateId", "replicate"]).reset_index(drop=True),
    )


def longitudinal_results(estimates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix_rows = []
    for keys, group in estimates.groupby(["matrixRole", "candidateId", "matrixIndex"], sort=True):
        ordered = group.sort_values("landmark")
        x = ordered["normalizedLandmark"].to_numpy(float)
        q = ordered["qConditional"].to_numpy(float)
        joint = ordered["qJoint"].to_numpy(float)
        matrix_rows.append(
            {
                "matrixRole": keys[0],
                "candidateId": keys[1],
                "matrixIndex": int(keys[2]),
                "conditionalSlope": float(np.polyfit(x, q, 1)[0]),
                "jointSlope": float(np.polyfit(x, joint, 1)[0]),
                "conditionalRange": float(np.max(q) - np.min(q)),
                "jointRange": float(np.max(joint) - np.min(joint)),
                "conditionalSuccessiveIncreases": int((np.diff(q) > 0).sum()),
                "conditionalSuccessiveDecreases": int((np.diff(q) < 0).sum()),
                "observedJointEvents": int(ordered["observedJointEvent"].sum()),
            }
        )
    matrices = pd.DataFrame(matrix_rows)
    summary_rows = []
    for (role, candidate, landmark), group in estimates.groupby(
        ["matrixRole", "candidateId", "landmark"], sort=True
    ):
        rng = generator("landmark_bootstrap", role, candidate, int(landmark))
        values = group["qConditional"].to_numpy(float)
        joint = group["qJoint"].to_numpy(float)
        boot_c = np.empty(BOOTSTRAPS)
        boot_j = np.empty(BOOTSTRAPS)
        for replicate in range(BOOTSTRAPS):
            indices = rng.integers(0, len(group), size=len(group))
            boot_c[replicate] = np.mean(values[indices])
            boot_j[replicate] = np.mean(joint[indices])
        c_low, c_high = interval(boot_c)
        j_low, j_high = interval(boot_j)
        summary_rows.append(
            {
                "matrixRole": role,
                "candidateId": candidate,
                "landmark": int(landmark),
                "matrices": len(group),
                "meanQConditional": float(np.mean(values)),
                "qConditionalLower95": c_low,
                "qConditionalUpper95": c_high,
                "meanQJoint": float(np.mean(joint)),
                "qJointLower95": j_low,
                "qJointUpper95": j_high,
                "observedBreakRate": float(group["observedBreak"].mean()),
                "observedJointEventRate": float(group["observedJointEvent"].mean()),
                "meanCurrentMass": float(group["currentMass"].mean()),
                "meanPrefixInheritanceFraction": float(group["prefixInheritanceFraction"].mean()),
            }
        )
    return matrices.sort_values(["matrixRole", "candidateId", "matrixIndex"]).reset_index(drop=True), pd.DataFrame(summary_rows)


def _fit_logistic(train_x: np.ndarray, train_y: np.ndarray) -> tuple[StandardScaler, LogisticRegression] | None:
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


def _predict_fitted(
    fitted: tuple[StandardScaler, LogisticRegression] | None,
    values: np.ndarray,
    prior: float,
) -> np.ndarray:
    if fitted is None:
        return np.full(len(values), prior, dtype=np.float64)
    scaler, model = fitted
    return model.predict_proba(scaler.transform(values))[:, 1]


def prediction_results(estimates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    model_rows = []
    for candidate in CANDIDATES:
        train = estimates[
            estimates["candidateId"].eq(candidate)
            & estimates["matrixRole"].eq("DEVELOPMENT")
        ].copy()
        validation = estimates[
            estimates["candidateId"].eq(candidate)
            & estimates["matrixRole"].eq("VALIDATION")
        ].copy()
        y_train = train["observedJointEvent"].astype(int).to_numpy()
        prior = float((y_train.sum() + 0.5) / (len(y_train) + 1))
        control_fit = _fit_logistic(train[list(CONTROL_COLUMNS)].to_numpy(float), y_train)
        time_fit = _fit_logistic(train[["normalizedLandmark"]].to_numpy(float), y_train)
        q_train = np.clip(train["qJoint"].to_numpy(float), 1e-6, 1 - 1e-6)
        combined_columns = list(CONTROL_COLUMNS)
        combined_train = np.c_[
            train[combined_columns].to_numpy(float), np.log(q_train / (1 - q_train))
        ]
        combined_fit = _fit_logistic(combined_train, y_train)
        model_rows.extend(
            [
                {
                    "candidateId": candidate,
                    "modelId": "PAST_CONTROLS",
                    "trainingMatrices": train["matrixIndex"].nunique(),
                    "trainingStates": len(train),
                    "trainingPositiveRate": float(y_train.mean()),
                    "featureCount": len(CONTROL_COLUMNS),
                    "fitOnDevelopmentOnly": True,
                },
                {
                    "candidateId": candidate,
                    "modelId": "PAST_PLUS_SHOOTING",
                    "trainingMatrices": train["matrixIndex"].nunique(),
                    "trainingStates": len(train),
                    "trainingPositiveRate": float(y_train.mean()),
                    "featureCount": len(CONTROL_COLUMNS) + 1,
                    "fitOnDevelopmentOnly": True,
                },
            ]
        )
        for role, frame in (("DEVELOPMENT", train), ("VALIDATION", validation)):
            q = np.clip(frame["qJoint"].to_numpy(float), 1e-6, 1 - 1e-6)
            inputs = frame[list(CONTROL_COLUMNS)].to_numpy(float)
            combined = np.c_[inputs, np.log(q / (1 - q))]
            probabilities = {
                "DEVELOPMENT_PRIOR": np.full(len(frame), prior),
                "TIME_ONLY": _predict_fitted(time_fit, frame[["normalizedLandmark"]].to_numpy(float), prior),
                "PAST_CONTROLS": _predict_fitted(control_fit, inputs, prior),
                "SHOOTING_Q_JOINT": q,
                "PAST_PLUS_SHOOTING": _predict_fitted(combined_fit, combined, prior),
            }
            for model, probability in probabilities.items():
                for position, source in enumerate(frame.itertuples(index=False)):
                    rows.append(
                        {
                            "stateId": source.stateId,
                            "matrixRole": role,
                            "candidateId": candidate,
                            "matrixIndex": int(source.matrixIndex),
                            "landmark": int(source.landmark),
                            "modelId": model,
                            "probability": float(probability[position]),
                            "observedJointEvent": bool(source.observedJointEvent),
                            "observedBreak": bool(source.observedBreak),
                            "developmentOnlyFit": model != "SHOOTING_Q_JOINT",
                            "usesForwardShooting": model in ("SHOOTING_Q_JOINT", "PAST_PLUS_SHOOTING"),
                        }
                    )
    predictions = pd.DataFrame(rows).sort_values(
        ["matrixRole", "candidateId", "modelId", "matrixIndex", "landmark"]
    ).reset_index(drop=True)
    return predictions, pd.DataFrame(model_rows)


def _binary_metrics(frame: pd.DataFrame) -> dict[str, float]:
    y = frame["observedJointEvent"].astype(int).to_numpy()
    p = np.clip(frame["probability"].to_numpy(float), 1e-9, 1 - 1e-9)
    prediction = p >= 0.5
    return {
        "brier": float(brier_score_loss(y, p)),
        "logLoss": float(log_loss(y, p, labels=[0, 1])),
        "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "auprc": float(average_precision_score(y, p)) if y.sum() else float("nan"),
        "balancedAccuracy": float(balanced_accuracy_score(y, prediction)) if len(np.unique(y)) > 1 else float("nan"),
        "positiveRate": float(y.mean()),
        "meanPredictedProbability": float(p.mean()),
    }


def predictive_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    point_rows = []
    bootstrap_rows = []
    for (role, candidate, model), group in predictions.groupby(
        ["matrixRole", "candidateId", "modelId"], sort=True
    ):
        point_rows.append(
            {
                "matrixRole": role,
                "candidateId": candidate,
                "modelId": model,
                "matrices": group["matrixIndex"].nunique(),
                "states": len(group),
                **_binary_metrics(group),
            }
        )
    for candidate in CANDIDATES:
        group = predictions[
            predictions["matrixRole"].eq("VALIDATION")
            & predictions["candidateId"].eq(candidate)
        ]
        matrices = sorted(group["matrixIndex"].unique())
        rng = generator("predictive_bootstrap", "VALIDATION", candidate, BOOTSTRAPS)
        for replicate in range(BOOTSTRAPS):
            sampled = rng.choice(matrices, size=len(matrices), replace=True)
            pieces = []
            for unit, matrix in enumerate(sampled):
                piece = group[group["matrixIndex"].eq(matrix)].copy()
                piece["_matrixUnit"] = unit
                pieces.append(piece)
            sample = pd.concat(pieces, ignore_index=True)
            for model, model_group in sample.groupby("modelId", sort=True):
                bootstrap_rows.append(
                    {
                        "candidateId": candidate,
                        "replicate": replicate,
                        "modelId": model,
                        **_binary_metrics(model_group),
                    }
                )
    point = pd.DataFrame(point_rows).sort_values(["matrixRole", "candidateId", "modelId"]).reset_index(drop=True)
    bootstrap = pd.DataFrame(bootstrap_rows).sort_values(["candidateId", "replicate", "modelId"]).reset_index(drop=True)
    comparisons = []
    pairs = (
        ("SHOOTING_Q_JOINT", "DEVELOPMENT_PRIOR"),
        ("SHOOTING_Q_JOINT", "PAST_CONTROLS"),
        ("PAST_PLUS_SHOOTING", "PAST_CONTROLS"),
    )
    for candidate in CANDIDATES:
        p = point[point["matrixRole"].eq("VALIDATION") & point["candidateId"].eq(candidate)].set_index("modelId")
        b = bootstrap[bootstrap["candidateId"].eq(candidate)]
        for model, reference in pairs:
            pivot = b[b["modelId"].isin([model, reference])].pivot(index="replicate", columns="modelId", values="brier")
            effects = pivot[reference] - pivot[model]
            low, high = interval(effects.to_numpy())
            comparisons.append(
                {
                    "candidateId": candidate,
                    "modelId": model,
                    "referenceModelId": reference,
                    "brierImprovement": float(p.loc[reference, "brier"] - p.loc[model, "brier"]),
                    "brierImprovementLower95": low,
                    "brierImprovementUpper95": high,
                    "fractionBootstrapPositive": float((effects > 0).mean()),
                }
            )
    return point, bootstrap, pd.DataFrame(comparisons)


def conditional_forecast_results(estimates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (role, candidate), group in estimates.groupby(["matrixRole", "candidateId"], sort=True):
        eligible = group[group["observedBreak"]].copy()
        if not len(eligible):
            continue
        y = eligible["observedJointEvent"].astype(int).to_numpy()
        q = np.clip(eligible["qConditional"].to_numpy(float), 1e-9, 1 - 1e-9)
        prior_source = estimates[
            estimates["matrixRole"].eq("DEVELOPMENT")
            & estimates["candidateId"].eq(candidate)
            & estimates["observedBreak"]
        ]
        prior = float((prior_source["observedJointEvent"].sum() + 0.5) / (len(prior_source) + 1))
        rows.append(
            {
                "matrixRole": role,
                "candidateId": candidate,
                "eligibleObservedBreakStates": len(eligible),
                "observedConditionalEventRate": float(y.mean()),
                "conditionalQBrier": float(brier_score_loss(y, q)),
                "conditionalPriorBrier": float(brier_score_loss(y, np.full(len(y), prior))),
                "conditionalBrierImprovementOverPrior": float(brier_score_loss(y, np.full(len(y), prior)) - brier_score_loss(y, q)),
                "conditionalQAuroc": float(roc_auc_score(y, q)) if len(np.unique(y)) > 1 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def negative_control_results(
    predictions: pd.DataFrame, comparisons: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for candidate in CANDIDATES:
        group = predictions[
            predictions["matrixRole"].eq("VALIDATION")
            & predictions["candidateId"].eq(candidate)
        ]
        shooting = group[group["modelId"].eq("SHOOTING_Q_JOINT")].copy()
        controls = group[group["modelId"].eq("PAST_CONTROLS")]
        control_brier = _binary_metrics(controls)["brier"]
        observed = comparisons[
            comparisons["candidateId"].eq(candidate)
            & comparisons["modelId"].eq("SHOOTING_Q_JOINT")
            & comparisons["referenceModelId"].eq("PAST_CONTROLS")
        ]["brierImprovement"].iloc[0]
        matrix_ids = sorted(shooting["matrixIndex"].unique())
        trajectories = {
            matrix: shooting[shooting["matrixIndex"].eq(matrix)]
            .sort_values("landmark")["probability"]
            .to_numpy(float)
            for matrix in matrix_ids
        }
        rng = generator("matrix_permutation", "VALIDATION", candidate, PERMUTATIONS)
        null = []
        for replicate in range(PERMUTATIONS):
            donors = rng.permutation(matrix_ids)
            permuted = shooting.sort_values(["matrixIndex", "landmark"]).copy()
            permuted["probability"] = np.concatenate(
                [trajectories[int(donor)] for donor in donors]
            )
            effect = control_brier - _binary_metrics(permuted)["brier"]
            null.append(effect)
            rows.append(
                {
                    "candidateId": candidate,
                    "controlId": "WHOLE_MATRIX_Q_TRAJECTORY_PERMUTATION",
                    "replicate": replicate,
                    "brierImprovementOverPastControls": effect,
                    "observedImprovement": observed,
                    "permutationP": np.nan,
                }
            )
        p_value = float((1 + np.sum(np.asarray(null) >= observed)) / (PERMUTATIONS + 1))
        for row in rows:
            if row["candidateId"] == candidate:
                row["permutationP"] = p_value
    return pd.DataFrame(rows)


def scientific_gates(
    estimates: pd.DataFrame,
    reliability: pd.DataFrame,
    comparisons: pd.DataFrame,
    controls: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], str]:
    rows = []
    validation_reliability = reliability[reliability["matrixRole"].eq("VALIDATION")]
    for row in validation_reliability.itertuples(index=False):
        passed = bool(
            row.dataInformedFraction >= 0.95
            and row.intermediateStates >= 20
            and row.centeredSplitHalfSpearmanConditionalLower95 > 0.3
            and row.correctedWithinMatrixVarianceLower95 > 0
        )
        rows.append(
            {
                "gateId": f"MEASUREMENT::{row.candidateId}",
                "candidateId": row.candidateId,
                "gateFamily": "LONGITUDINAL_MEASUREMENT",
                "dataInformedFraction": row.dataInformedFraction,
                "intermediateStates": int(row.intermediateStates),
                "centeredSplitHalfLower95": row.centeredSplitHalfSpearmanConditionalLower95,
                "correctedWithinVarianceLower95": row.correctedWithinMatrixVarianceLower95,
                "brierImprovementOverPriorLower95": np.nan,
                "brierImprovementOverControlsLower95": np.nan,
                "permutationP": np.nan,
                "observedJointEvents": int(
                    estimates[
                        estimates["matrixRole"].eq("VALIDATION")
                        & estimates["candidateId"].eq(row.candidateId)
                    ]["observedJointEvent"].sum()
                ),
                "passed": passed,
            }
        )
    for candidate in CANDIDATES:
        candidate_comparisons = comparisons[comparisons["candidateId"].eq(candidate)]
        versus_prior = candidate_comparisons[
            candidate_comparisons["modelId"].eq("SHOOTING_Q_JOINT")
            & candidate_comparisons["referenceModelId"].eq("DEVELOPMENT_PRIOR")
        ].iloc[0]
        versus_controls = candidate_comparisons[
            candidate_comparisons["modelId"].eq("SHOOTING_Q_JOINT")
            & candidate_comparisons["referenceModelId"].eq("PAST_CONTROLS")
        ].iloc[0]
        permutation_p = float(
            controls[controls["candidateId"].eq(candidate)]["permutationP"].iloc[0]
        )
        events = int(
            estimates[
                estimates["matrixRole"].eq("VALIDATION")
                & estimates["candidateId"].eq(candidate)
            ]["observedJointEvent"].sum()
        )
        passed = bool(
            events >= 20
            and versus_prior.brierImprovementLower95 > 0
            and versus_controls.brierImprovementLower95 > 0
            and permutation_p <= 0.05
        )
        rows.append(
            {
                "gateId": f"OBSERVED_FORECAST::{candidate}",
                "candidateId": candidate,
                "gateFamily": "INDEPENDENT_OBSERVED_FUTURE",
                "dataInformedFraction": np.nan,
                "intermediateStates": np.nan,
                "centeredSplitHalfLower95": np.nan,
                "correctedWithinVarianceLower95": np.nan,
                "brierImprovementOverPriorLower95": versus_prior.brierImprovementLower95,
                "brierImprovementOverControlsLower95": versus_controls.brierImprovementLower95,
                "permutationP": permutation_p,
                "observedJointEvents": events,
                "passed": passed,
            }
        )
    gate_frame = pd.DataFrame(rows)
    measurement_pass = bool(
        gate_frame[gate_frame["gateFamily"].eq("LONGITUDINAL_MEASUREMENT")]["passed"].all()
    )
    forecast_pass = bool(
        gate_frame[gate_frame["gateFamily"].eq("INDEPENDENT_OBSERVED_FUTURE")]["passed"].all()
    )
    if measurement_pass and forecast_pass:
        classifications = [
            "LONGITUDINAL_PROCESS_COMMITTOR_STATE_DEPENDENT",
            "SHOOTING_FORECASTS_NEXT_HEREDITARY_EPISODE",
            "SIMULATION_ACCESSIBLE_PROCESS_PRECURSOR_LEAD",
            "PROMOTABLE_TO_UNTOUCHED_PROCESS_COMMITTOR_CONFIRMATION",
        ]
        next_theme = "L50_UNTOUCHED_PROCESS_COMMITTOR_PRECURSOR_CONFIRMATION"
    elif measurement_pass:
        classifications = [
            "LONGITUDINAL_PROCESS_COMMITTOR_STATE_DEPENDENT",
            "PROCESS_COMMITTOR_NOT_INCREMENTAL_FOR_REALIZED_EPISODE",
            "SHOOTING_REMAINS_ENSEMBLE_MEASUREMENT",
            "NOT_PROMOTABLE_AS_CONFIRMED",
        ]
        next_theme = "L50_PROCESS_EVENT_HORIZON_AND_PHASE_IDENTIFIABILITY"
    else:
        classifications = [
            "NO_RELIABLE_WITHIN_LINEAGE_PROCESS_RISK_TRAJECTORY",
            "SHOOTING_REMAINS_CROSS_SECTIONAL_ENSEMBLE_MEASUREMENT",
            "NOT_PROMOTABLE_AS_CONFIRMED",
        ]
        next_theme = "L50_PROCESS_EVENT_HORIZON_AND_PHASE_IDENTIFIABILITY"
    gate_frame = pd.concat(
        [
            gate_frame,
            pd.DataFrame(
                [
                    {
                        "gateId": "COMPLETE_LOOP",
                        "candidateId": "BOTH",
                        "gateFamily": "COMPLETE",
                        "dataInformedFraction": np.nan,
                        "intermediateStates": np.nan,
                        "centeredSplitHalfLower95": np.nan,
                        "correctedWithinVarianceLower95": np.nan,
                        "brierImprovementOverPriorLower95": np.nan,
                        "brierImprovementOverControlsLower95": np.nan,
                        "permutationP": np.nan,
                        "observedJointEvents": int(
                            estimates[estimates["matrixRole"].eq("VALIDATION")]["observedJointEvent"].sum()
                        ),
                        "passed": measurement_pass and forecast_pass,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    return gate_frame, classifications, next_theme


def compute_tables(
    branches: pd.DataFrame,
    states: pd.DataFrame,
    observed: pd.DataFrame,
    validation: pd.DataFrame,
    selection: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], list[str], str]:
    estimates = state_estimates(branches, states, observed)
    reliability, reliability_bootstrap = reliability_results(estimates)
    matrix_trajectories, landmark_summary = longitudinal_results(estimates)
    predictions, model_registry = prediction_results(estimates)
    metrics, metric_bootstrap, comparisons = predictive_metrics(predictions)
    conditional = conditional_forecast_results(estimates)
    controls = negative_control_results(predictions, comparisons)
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
        "matrix_risk_trajectory_results.parquet": matrix_trajectories,
        "landmark_risk_summary.parquet": landmark_summary,
        "model_registry.parquet": model_registry,
        "prediction_results.parquet": predictions,
        "predictive_metric_results.parquet": metrics,
        "predictive_metric_bootstrap.parquet": metric_bootstrap,
        "paired_predictive_comparisons.parquet": comparisons,
        "conditional_forecast_results.parquet": conditional,
        "negative_control_results.parquet": controls,
        "scientific_gate_results.parquet": gates,
    }
    return tables, classifications, next_theme


def make_figures(tables: dict[str, pd.DataFrame]) -> None:
    figure_root = BUILD_ROOT / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    summary = tables["landmark_risk_summary.parquet"]
    estimates = tables["state_committor_results.parquet"]
    reliability = tables["committor_reliability_results.parquet"]
    metrics = tables["predictive_metric_results.parquet"]
    comparisons = tables["paired_predictive_comparisons.parquet"]
    colors = {CANDIDATES[0]: "#4c78a8", CANDIDATES[1]: "#f58518"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for axis, role in zip(axes, ROLES, strict=True):
        for candidate, group in summary[summary["matrixRole"].eq(role)].groupby("candidateId"):
            group = group.sort_values("landmark")
            axis.plot(group["landmark"], group["meanQConditional"], marker="o", color=colors[candidate], label=f"C{candidate[-2:]}")
            axis.fill_between(group["landmark"], group["qConditionalLower95"], group["qConditionalUpper95"], color=colors[candidate], alpha=0.15)
        axis.set_title(role)
        axis.set_xlabel("Selected-clock landmark")
    axes[0].set_ylabel("Conditional run-3 committor")
    axes[0].legend()
    fig.suptitle("Longitudinal process risk on fixed matrix-shared landmarks")
    fig.tight_layout()
    fig.savefig(figure_root / "01_longitudinal_conditional_committor.png", dpi=160)
    plt.close(fig)

    validation = estimates[estimates["matrixRole"].eq("VALIDATION")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for axis, candidate in zip(axes, CANDIDATES, strict=True):
        group = validation[validation["candidateId"].eq(candidate)]
        axis.scatter(group["qConditionalHalfA"], group["qConditionalHalfB"], alpha=0.65)
        axis.plot([0, 1], [0, 1], color="black", ls=":")
        row = reliability[reliability["matrixRole"].eq("VALIDATION") & reliability["candidateId"].eq(candidate)].iloc[0]
        axis.set_title(f"C{candidate[-2:]} centered rho={row.centeredSplitHalfSpearmanConditional:.2f}")
        axis.set_xlabel("32-branch half A")
        axis.set_ylabel("32-branch half B")
    fig.suptitle("Independent-half longitudinal measurement reliability")
    fig.tight_layout()
    fig.savefig(figure_root / "02_split_half_reliability.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for axis, candidate in zip(axes, CANDIDATES, strict=True):
        group = validation[validation["candidateId"].eq(candidate)]
        jitter = generator("figure_jitter", candidate).normal(0, 0.015, len(group))
        axis.scatter(group["qJoint"], group["observedJointEvent"].astype(float) + jitter, alpha=0.5)
        bins = pd.qcut(group["qJoint"], q=5, duplicates="drop")
        calibrated = group.assign(_bin=bins).groupby("_bin", observed=False).agg(q=("qJoint", "mean"), y=("observedJointEvent", "mean"))
        axis.plot(calibrated["q"], calibrated["y"], marker="o", color="black")
        axis.plot([0, 1], [0, 1], ls=":", color="gray")
        axis.set_title(f"C{candidate[-2:]}")
        axis.set_xlabel("64-branch joint process probability")
    axes[0].set_ylabel("Independent observed F12 event")
    fig.suptitle("Shooting probability versus frozen realized future")
    fig.tight_layout()
    fig.savefig(figure_root / "03_observed_event_calibration.png", dpi=160)
    plt.close(fig)

    heldout = metrics[metrics["matrixRole"].eq("VALIDATION")]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    pivot = heldout.pivot(index="modelId", columns="candidateId", values="brier")
    pivot.plot(kind="bar", ax=ax, color=[colors[c] for c in pivot.columns])
    ax.set_ylabel("Brier score (lower is better)")
    ax.set_title("Held-out realized-process forecasting")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(figure_root / "04_validation_brier_scores.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    plot = comparisons[comparisons["modelId"].eq("SHOOTING_Q_JOINT")].copy()
    labels = [f"C{row.candidateId[-2:]} vs {row.referenceModelId}" for row in plot.itertuples()]
    ax.errorbar(
        np.arange(len(plot)),
        plot["brierImprovement"],
        yerr=np.vstack([
            plot["brierImprovement"] - plot["brierImprovementLower95"],
            plot["brierImprovementUpper95"] - plot["brierImprovement"],
        ]),
        fmt="o",
    )
    ax.axhline(0, color="black", ls=":")
    ax.set_xticks(np.arange(len(plot)), labels, rotation=25, ha="right")
    ax.set_ylabel("Brier improvement")
    ax.set_title("Shooting increment over registered baselines")
    fig.tight_layout()
    fig.savefig(figure_root / "05_brier_improvement.png", dpi=160)
    plt.close(fig)

    gates = tables["scientific_gate_results.parquet"]
    candidate_gates = gates[gates["candidateId"].isin(CANDIDATES)].pivot(
        index="gateFamily", columns="candidateId", values="passed"
    )
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    image = ax.imshow(candidate_gates.astype(float), vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(candidate_gates.columns)), [f"C{x[-2:]}" for x in candidate_gates.columns])
    ax.set_yticks(range(len(candidate_gates.index)), candidate_gates.index, fontsize=8)
    ax.set_title("Longitudinal process-risk decision matrix")
    fig.colorbar(image, ax=ax, ticks=[0, 1])
    fig.tight_layout()
    fig.savefig(figure_root / "06_decision_matrix.png", dpi=160)
    plt.close(fig)


def report_text(
    tables: dict[str, pd.DataFrame],
    classifications: list[str],
    next_theme: str,
    runtime: dict[str, Any],
) -> str:
    reliability = tables["committor_reliability_results.parquet"]
    reliability = reliability[
        [
            "matrixRole",
            "candidateId",
            "matrices",
            "states",
            "dataInformedFraction",
            "intermediateStates",
            "stateSplitHalfSpearmanConditional",
            "centeredSplitHalfSpearmanConditional",
            "centeredSplitHalfSpearmanConditionalLower95",
            "correctedWithinMatrixVariance",
            "correctedWithinMatrixVarianceLower95",
            "meanAbsoluteSuccessiveDelta",
        ]
    ]
    landmark = tables["landmark_risk_summary.parquet"][
        [
            "matrixRole",
            "candidateId",
            "landmark",
            "meanQConditional",
            "qConditionalLower95",
            "qConditionalUpper95",
            "meanQJoint",
            "observedBreakRate",
            "observedJointEventRate",
        ]
    ]
    metrics = tables["predictive_metric_results.parquet"]
    metrics = metrics[metrics["matrixRole"].eq("VALIDATION")][
        ["candidateId", "modelId", "states", "brier", "logLoss", "auroc", "auprc", "balancedAccuracy", "positiveRate"]
    ]
    comparisons = tables["paired_predictive_comparisons.parquet"]
    gates = tables["scientific_gate_results.parquet"]
    return f"""# S19-L49 Full Results — Longitudinal Process-Committor Risk Trajectory

## Top summary

- **Research step:** `{VERSION}`
- **Completion status:** complete; additive exploratory simulator evidence
- **Artifacts written:** immutable/source/input/seed locks, 400 restored states, {runtime['newBranchStreams']:,} new F12 branch identities and exact replays, state and longitudinal committors, independent observed-future forecasts, 4,096 matrix bootstraps, 512 matrix-trajectory permutations, six figures, regeneration/storage/hash manifests
- **Validation:** PASS — immutable S01–L48 baseline; nine fixtures; exact 40-matrix shared selection; 400/400 state restorations; seed firewall; two exact branch campaigns; exact analysis/report regeneration; runtime, storage and artifact hashes
- **Outcome classification:** {', '.join(f'`{value}`' for value in classifications)}
- **Lay summary:** This loop measures whether the chance of breaking and then rebuilding a three-fission hereditary episode changes along the same simulated lineage. It separately estimates the chance of a break, the chance of renewed heredity after a break, and their joint chance. It then asks whether those simulated probabilities forecast what happened in the already frozen, independently realized continuation.
- **Recommended next action:** `{next_theme}` under the autonomous authorization through L65. S20, E02, author contact and intervention work remain inactive.

## Frozen question and boundary

L48 showed that the conservative process-probability contract required all 64 estimator-half futures. L49 therefore does not reduce or adapt the branch budget. It chooses 20 development and 20 validation catalytic matrices by outcome-blind SHA-256 rank, shares those matrix identities across candidate 2 and candidate 3, and restores five fixed molecular-clock landmarks per trajectory. Every state receives exactly 64 new domain-separated continuations through 12 future fissions.

The process is defined without a completed trajectory: the first strict `H<=0.9` parent/daughter break must be followed by three strict `H>0.9` inherited fissions. Uninterrupted inheritance is not a new episode. No centroid, attractor inferred from the completed run, Phi quantity, paper label, intervention or target search enters the analysis.

## Longitudinal measurement reliability

{reliability.to_markdown(index=False, floatfmt='.6f')}

The centered split-half statistic removes each matrix's mean and asks whether independent 32-branch halves agree about the *within-matrix* ordering of its five states. Corrected within-matrix variance subtracts the expected conditional-binomial measurement noise from the five-state sample variance and bootstraps whole catalytic matrices.

## Risk trajectory by fixed landmark

{landmark.to_markdown(index=False, floatfmt='.6f')}

These are fixed-clock summaries, not event-aligned curves. A lack of monotonic increase does not erase state dependence; it means the process risk behaves as regime switching rather than as a universal clock-like ramp.

## Independent realized-future forecast

{metrics.to_markdown(index=False, floatfmt='.6f')}

## Registered Brier comparisons

{comparisons.to_markdown(index=False, floatfmt='.6f')}

The past-control model is fitted only on development matrices and uses normalized time, mass, distance to fission, generation-local step, completed fissions, prefix inheritance frequency, trailing inheritance run and latest parent/daughter H. The direct shooting score is not fitted to the observed outcome. Whole q-trajectories are permuted among validation matrices as the registered alignment null.

## Scientific gates

{gates.to_markdown(index=False, floatfmt='.6f')}

## Interpretation

A passing result supports only a simulator-accessible precursor to a newly certified hereditary episode: at the current state, an ensemble of matched stochastic continuations estimates a probability that can be checked against another realized future. It is not a compact observed-state biomarker, author-code reconstruction, paper replication, causal-emergence result, intervention effect, or claim about real chemistry. A nonpassing observed-future gate would not invalidate the branch probability; it would show that the estimated committor does not add robust forecast value over direct state history at this scope.

## Runtime, validation and provenance

- Repository lock: `{runtime['repositoryHead']}`.
- Workers: `{runtime['workers']}` with one numerical-library thread each; GPU hours: 0.
- Wall time: `{runtime['wallSeconds'] / 60:.3f}` minutes; worker CPU upper estimate: `{runtime['estimatedWorkerCpuHours']:.6f}` hours.
- Matrices: 40 shared across candidates; restored states: 400; new primary matrices/trajectories: 0/0.
- New branch streams: `{runtime['newBranchStreams']:,}`; exact branch campaigns: 2.
- Matrix bootstraps: {BOOTSTRAPS}; matrix-trajectory permutations: {PERMUTATIONS}.
- Web research supplied method context only; exact URLs and frozen uses are in `source_registry.parquet`.

## Limitations

The five landmarks are early fixed molecular-clock states, not fission-aligned transition interfaces. The 64-branch estimates remain Monte Carlo quantities and their conditional sample sizes vary with break frequency. The observed target is one frozen future per state and overlapping 12-fission windows from the same matrix are dependent; all uncertainty therefore resamples catalytic matrices, not state rows. The event captures renewed compositional inheritance, not restoration of an old composition or function. Any favorable result remains adaptive, exploratory and simulator-specific until an untouched matrix campaign confirms it.
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
        "schema": "eidosoma.e01.s19_l49.artifact_manifest.v1",
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
            "beliefBeforeLoop": "L48 established that the online run-three process probability is measurable but required all 64 estimator-half futures under the conservative cross-candidate contract.",
            "failureOrAmbiguityTargeted": "Whether that probability changes coherently along the same lineage and forecasts a separately realized new hereditary episode.",
            "informationGainRationale": "Repeated fixed landmarks within shared matrices separate true longitudinal state variation from cross-matrix heterogeneity; a development/validation firewall tests an independently realized future.",
            "learned": "L49 longitudinal state, branch, control and validation contract locked before scientific outcomes.",
            "ledgerSequence": sequence,
            "loopId": LOOP_ID,
            "motivatingEvidence": "L44 process identifiability, L48 full-half requirement, reviewer process-first and shooting-as-measurement framing.",
            "proposedNextTest": "Measure break, conditional renewal and joint process probabilities at five fixed states per matrix.",
            "recordPhase": "PRE_LOOP_METHOD_LOCK",
            "remainingPlausibleHypotheses": "Within-lineage process risk, stationary matrix-specific propensity, direct-history sufficiency or simulation-only forecasting.",
            "selectedHypotheses": "A process committor may be a longitudinal simulator-accessible precursor even when no compact static representation transfers.",
            "timestampUtc": timestamp,
            "weakenedHypotheses": "Another broad static-feature tournament should precede testing the teacher probability itself over time.",
        },
        {
            "appendOnly": True,
            "beliefBeforeLoop": "A useful longitudinal precursor requires independent-half within-matrix reliability and independent realized-future forecast value beyond direct past controls in both candidates.",
            "failureOrAmbiguityTargeted": "State dependence, temporal behavior and independent-event calibration of the process committor.",
            "informationGainRationale": "Whole-matrix bootstraps and trajectory permutations preserve the repeated-landmark dependence structure.",
            "learned": ";".join(classifications),
            "ledgerSequence": sequence + 1,
            "loopId": LOOP_ID,
            "motivatingEvidence": "Complete L49 result.",
            "proposedNextTest": next_theme,
            "recordPhase": "POST_LOOP_RESULT_AUTONOMOUS_CONTINUATION",
            "remainingPlausibleHypotheses": next_theme,
            "selectedHypotheses": "Longitudinal online process-committor measurement.",
            "timestampUtc": timestamp,
            "weakenedHypotheses": "Any registered L49 measurement or forecast gate that failed.",
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
        + f"\n\n## {LOOP_ID} — longitudinal process-committor risk\n\n"
        + f"- **Learned:** {', '.join(classifications)}.\n"
        + f"- **Next:** `{next_theme}`.\n",
    )

    candidate_path = ARTIFACT_ROOT / "candidate_registry.parquet"
    candidates = pd.read_parquet(candidate_path)
    candidate = {
        "branchCount": BRANCHES,
        "bundleId": "L49_LONGITUDINAL_PROCESS_COMMITTOR",
        "candidateId": "S19-L49-LONGITUDINAL-PROCESS-COMMITTOR",
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
        "proposedSpecification": "five fixed longitudinal states, 64 F12 shoots, online break-plus-run3 joint probability, independent frozen-future scoring",
        "rankingScore": 28.0,
        "registryOrder": int(candidates["registryOrder"].max()) + 1,
        "selected": "SIMULATION_ACCESSIBLE_PROCESS_PRECURSOR_LEAD" in classifications,
        "selectionReason": "L48_FULL_HALF_REQUIREMENT_AND_PROCESS_FIRST_REVIEW",
        "sourceGrounding": 4,
        "testability": 5,
        "undefinedAuthorSemantics": 0,
    }
    BASE.write_parquet(
        candidate_path,
        pd.concat(
            [candidates, pd.DataFrame([candidate]).reindex(columns=candidates.columns)],
            ignore_index=True,
        ),
    )

    source_path = ARTIFACT_ROOT / "source_search_ledger.parquet"
    sources = pd.read_parquet(source_path)
    additions = []
    for row in source_registry().itertuples(index=False):
        additions.append(
            {
                "commitOrVersion": None,
                "evidenceClass": row.evidenceClass,
                "finding": f"{row.finding}; L49 use: {row.frozenUse}",
                "licenseStatus": "PUBLIC_METADATA_OR_WORKSPACE_EVIDENCE",
                "redistributionStatus": "REFERENCE_ONLY",
                "repositoryIdentity": None,
                "retainedPath": None,
                "retrievalDate": timestamp[:10],
                "sha256": None,
                "sourceId": f"L49_{row.sourceId}",
                "sourceType": row.evidenceClass,
                "treeIdentity": None,
                "url": row.url,
            }
        )
    BASE.write_parquet(
        source_path,
        pd.concat(
            [sources, pd.DataFrame(additions).reindex(columns=sources.columns)],
            ignore_index=True,
        ),
    )


def prepare_lock() -> None:
    LOOP_ROOT.mkdir(parents=True, exist_ok=True)
    if git("status", "--porcelain=v1"):
        raise RuntimeError("repository must be clean before L49 lock")
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/eidosoma/groups/42")
    if head != remote:
        raise RuntimeError("L49 local/remote commit mismatch")
    prior = validate_immutable_prior()
    fixtures = fixture_results()
    selection, expanded = select_matrices()
    branch_seeds = branch_seed_manifest(expanded.to_dict("records"))
    analysis_seeds = analysis_seed_manifest()
    firewall = seed_firewall(branch_seeds, analysis_seeds)
    manifest = pd.read_parquet(L23_ROOT / "input_trajectory_manifest.parquet")
    if len(manifest) != 800:
        raise RuntimeError("L49 frozen trajectory manifest cardinality changed")
    input_checks = []
    selected_matrices = set(selection["matrixIndex"].astype(int))
    for row in manifest[manifest["matrixIndex"].isin(selected_matrices)].itertuples(index=False):
        path = Path(row.cachePath)
        input_checks.append(
            {
                "candidateId": row.candidateId,
                "matrixIndex": int(row.matrixIndex),
                "cachePath": str(path),
                "expectedSha256": row.cacheSha256,
                "actualSha256": sha256_file(path) if path.is_file() else None,
                "passed": path.is_file() and sha256_file(path) == row.cacheSha256,
            }
        )
    input_validation = pd.DataFrame(input_checks)
    l41_runtime = json.loads((ARTIFACT_ROOT / "loops/L41/runtime_manifest.json").read_text())
    prior_branch_seconds = float(l41_runtime.get("branchSeconds", l41_runtime.get("wallSeconds", 3600)))
    prior_branch_count = 53_760
    projected_one_pass = prior_branch_seconds / prior_branch_count * len(branch_seeds)
    benchmark = {
        "schema": "eidosoma.e01.s19_l49.benchmark_projection.v1",
        "outcomeBlind": True,
        "basis": "frozen L41 measured F12/F4 branch execution cost; no L49 scientific branch executed before lock",
        "priorMeasuredBranchRows": prior_branch_count,
        "newBranchesPerPass": len(branch_seeds),
        "exactBranchPasses": 2,
        "projectedWallHoursUpper": projected_one_pass * 2.5 / 3600,
        "projectedWorkerCpuHoursUpper": projected_one_pass * 2.5 * WORKERS / 3600,
        "workers": WORKERS,
        "status": "PASS" if projected_one_pass * 2.5 / 3600 < 60 else "FAIL",
    }
    if (
        not prior["unchanged"]
        or not fixtures["passed"].all()
        or not input_validation["passed"].all()
        or firewall["status"] != "PASS"
        or benchmark["status"] != "PASS"
        or len(selection) != 40
        or len(expanded) != 400
    ):
        raise RuntimeError("L49 preoutcome validation failed")
    shutil.copy2(CONFIG, LOOP_ROOT / "preregistration.yaml")
    BASE.atomic_text(
        LOOP_ROOT / "decision_record.md",
        "# S19-L49 decision record\n\n"
        "The human-authorized autonomous sequence through L65 and the explicit allowance of up to eight CPUs support one bounded new branch campaign where parallelism materially shortens execution. L48 found that the conservative process measurement required the complete 64-branch estimator half and that uncertainty-based allocation did not improve it. Before any L49 scientific branch or frozen observed-future process result is opened, this record locks forty shared catalytic matrices—twenty development and twenty validation—by SHA-256 rank, both simulator candidates, five fixed C1 landmarks, exactly 64 new F12 continuations per state, strict H>0.9 inheritance, first future break followed by a three-inherited-fission episode, and separate break, conditional and joint probabilities. The frozen realized F12 continuation is an independent scoring outcome and never selects a state, method or threshold. Past controls, logistic regularization, matrix bootstraps, matrix-trajectory permutations and every success gate are fixed. No completed-run centroid, Phi quantity, new primary trajectory, architecture search or intervention is authorized.\n",
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
            "schema": "eidosoma.e01.s19_l49.source_snapshot_manifest.v1",
            "runnerSha256": sha256_file(RUNNER_PATH),
            "coreSha256": sha256_file(CORE_PATH),
            "configSha256": sha256_file(CONFIG),
            "fissionClockCoreSha256": sha256_file(ROOT / "src/e01_onset_discovery/fission_clock_recurrence.py"),
            "gardPaperSha256": "77a2ec2c0751839d8a2e10863ca803c6f8b61475bbc790f2bbdad2a38af04ae4",
            "webResearchDate": utc_now()[:10],
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
        "l48Manifest": L48_ROOT / "artifact_manifest.json",
    }
    hashes = {name: sha256_file(path) for name, path in locked_inputs.items()}
    lock = {
        "schema": "eidosoma.e01.s19_l49.implementation_lock.v1",
        "repositoryHead": head,
        "remoteHead": remote,
        "runnerSha256": sha256_file(RUNNER_PATH),
        "coreSha256": sha256_file(CORE_PATH),
        "configSha256": sha256_file(CONFIG),
        "matrixCount": 40,
        "states": 400,
        "branchesPerState": BRANCHES,
        "newBranchStreams": len(branch_seeds),
        "workers": WORKERS,
        "landmarks": list(LANDMARKS),
        "futureFissionHorizon": FISSION_HORIZON,
        "threshold": THRESHOLD,
        "requiredRun": REQUIRED_RUN,
        "matrixBootstraps": BOOTSTRAPS,
        "matrixPermutations": PERMUTATIONS,
        "controlColumns": list(CONTROL_COLUMNS),
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
        raise RuntimeError("L49 repository lock mismatch")
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
        "l48Manifest": L48_ROOT / "artifact_manifest.json",
    }
    if any(
        sha256_file(path) != lock["lockedInputHashes"][name]
        for name, path in locked_inputs.items()
    ):
        raise RuntimeError("L49 locked input changed")
    if (
        not prior["unchanged"]
        or prior["aggregateSha256"] != lock["priorAggregateSha256"]
        or not fixtures["passed"].all()
        or sha256_file(RUNNER_PATH) != lock["runnerSha256"]
        or sha256_file(CORE_PATH) != lock["coreSha256"]
        or sha256_file(CONFIG) != lock["configSha256"]
    ):
        raise RuntimeError("L49 pre-execution validation failed")
    selection, expanded = select_matrices()
    if frame_hash(selection) != frame_hash(pd.read_parquet(LOOP_ROOT / "matrix_selection_registry.parquet")):
        raise RuntimeError("L49 matrix selection regeneration failure")
    if frame_hash(expanded) != frame_hash(pd.read_parquet(LOOP_ROOT / "state_selection_registry.parquet")):
        raise RuntimeError("L49 state selection regeneration failure")
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
        raise RuntimeError("L49 exact branch replay failure")
    seed_manifest = pd.read_parquet(LOOP_ROOT / "branch_seed_manifest.parquet")
    identity = branches.merge(
        seed_manifest[["stateId", "branchIndex", "branchIdentitySha256"]],
        on=["stateId", "branchIndex"],
        suffixes=("", "Expected"),
        validate="one_to_one",
    )
    if not identity["branchIdentitySha256"].eq(identity["branchIdentitySha256Expected"]).all():
        raise RuntimeError("L49 branch identity manifest mismatch")

    tables, classifications, next_theme = compute_tables(
        branches, states, observed, state_validation, selection
    )
    make_figures(tables)
    tables_again, classifications_again, next_theme_again = compute_tables(
        branches.copy(), states.copy(), observed.copy(), state_validation.copy(), selection.copy()
    )
    table_exact = {
        name: frame_hash(frame) == frame_hash(tables_again[name])
        for name, frame in tables.items()
    }
    regeneration = {
        "schema": "eidosoma.e01.s19_l49.regeneration_validation.v1",
        "status": "PASS" if branch_exact and all(table_exact.values()) and classifications == classifications_again and next_theme == next_theme_again else "FAIL",
        "branchCampaignExact": branch_exact,
        "tableExact": table_exact,
        "classificationExact": classifications == classifications_again,
        "nextThemeExact": next_theme == next_theme_again,
        "analysisPasses": 2,
    }
    if regeneration["status"] != "PASS":
        raise RuntimeError("L49 regeneration failure")
    for name, frame in tables.items():
        BASE.write_parquet(BUILD_ROOT / name, frame)
    BASE.write_json(
        BUILD_ROOT / "classification.json",
        {
            "schema": "eidosoma.e01.s19_l49.classification.v1",
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
        "schema": "eidosoma.e01.s19_l49.runtime.v1",
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
        "matrixPermutations": PERMUTATIONS,
        "exactBranchCampaigns": 2,
        "completedAtUtc": utc_now(),
    }
    if runtime["estimatedWorkerCpuHours"] > 90 or runtime["wallSeconds"] > 60 * 3600:
        raise RuntimeError("L49 runtime ceiling exceeded")
    BASE.write_json(BUILD_ROOT / "runtime_manifest.json", runtime)
    BASE.write_json(BUILD_ROOT / "regeneration_validation.json", regeneration)
    retained_bytes = sum(path.stat().st_size for path in BUILD_ROOT.rglob("*") if path.is_file()) + sum(path.stat().st_size for path in LOOP_ROOT.iterdir() if path.is_file())
    storage = {
        "schema": "eidosoma.e01.s19_l49.storage_validation.v1",
        "status": "PASS" if retained_bytes <= 25 * 1024**3 else "FAIL",
        "retainedBytes": retained_bytes,
        "retainedGiBCeiling": 25,
        "temporaryGiBCeiling": 75,
    }
    BASE.write_json(BUILD_ROOT / "storage_validation.json", storage)
    report = report_text(tables, classifications, next_theme, runtime)
    if report != report_text(tables, classifications, next_theme, runtime):
        raise RuntimeError("L49 report regeneration failure")
    BASE.atomic_text(BUILD_ROOT / "S19_L49_FULL_RESULTS.md", report)
    BASE.atomic_text(BUILD_ROOT / "research_step_full_results.md", report)
    BASE.atomic_text(
        BUILD_ROOT / "loop_decision_summary.md",
        f"# S19-L49 decision summary\n\n**Classification:** {', '.join(classifications)}\n\n**Next:** `{next_theme}`.\n",
    )
    if storage["status"] != "PASS":
        raise RuntimeError("L49 storage ceiling exceeded")
    for path in (BUILD_ROOT / "figures").glob("*.png"):
        if not path.stat().st_size:
            raise RuntimeError(f"empty L49 figure: {path}")

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
        raise RuntimeError("L49 artifact manifest regeneration failure")

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
            "nextAuthorizedLoop": "S19-L50",
            "nextTheme": next_theme,
            "authorizationUpperBound": "S19-L65",
            "s20Active": False,
            "updatedAtUtc": runtime["completedAtUtc"],
        },
    )
    BASE.write_json(ARTIFACT_ROOT / "artifact_manifest.json", manifest_for(ARTIFACT_ROOT))
    print(json.dumps({"status": "COMPLETE", "classifications": classifications, "nextTheme": next_theme, "runtime": runtime}, indent=2))


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
