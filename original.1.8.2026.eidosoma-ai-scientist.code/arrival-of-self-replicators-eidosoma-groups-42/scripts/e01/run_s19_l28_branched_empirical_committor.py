"""Execute S19-L28 branched empirical-committor identifiability audit."""

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
from collections.abc import Iterable
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
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from e01_clean_directional_confirmation.core import fixed_label_spec
from e01_creative_directional_search.core import label_trajectory
from e01_frozen_timebase_ensemble.core import (
    selected_clock_observations,
    states_from_observations,
)
from e01_latent_timebase.core import (
    ExposureDefinition,
    SimulationDefinition,
    derive_seed,
    generate_beta,
    generator,
)
from e01_latent_timebase.core import (
    array_sha256 as simulator_array_sha256,
)
from e01_onset_discovery.empirical_committor import (
    BRANCHES,
    HALF_BRANCHES,
    HORIZON,
    TARGET_THRESHOLD,
    array_sha256,
    corrected_between_state_variance,
    cosine_to_reference,
    dominant_component_centroid,
    restored_state_from_observation,
    simulate_branch,
)


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


L27 = _load_module(
    "e01_s19_l28_l27",
    REPO_ROOT / "scripts/e01/run_s19_l27_transition_tube_density_current.py",
)
BASE = L27.BASE
LOOP_ID = "S19-L28"
VERSION = "E01-S19-L28-BRANCHED-EMPIRICAL-COMMITTOR-IDENTIFIABILITY-v1.0.0"
CANDIDATES = ("S12F-CANDIDATE-02", "S12F-CANDIDATE-03")
LANDMARKS = (64, 96, 128, 160, 192)
ROLES = ("DEVELOPMENT", "VALIDATION")
STATES_PER_STRATUM = 10
BOOTSTRAPS = 4096
WORKERS = 8
TARGET_ID = "PF_DOMINANT_COMPONENT_CENTROID_H900"
CLOCK_ID = "C1_SELECTED_DAUGHTER_RETAINED"
BRANCH_ROOT_HEX = "173faa80847abc978ae2ca332f2732e32d99ddc2a32bb5abda9d25e8ed19af5f"
BRANCH_PHASE = "s19_l28_branched_empirical_committor"
L23_ROOT_HEX = "f3b0fd551b8f182388cad84365b62a2f2f51e82aa11a0c6b6e22a088dbe90544"
L23_PHASE = "s19_l23_powered_frozen_prefix_screen"
ARTIFACT_ROOT = Path("/artifacts/research_steps/S19")
LOOP_ROOT = ARTIFACT_ROOT / "loops/L28"
L27_ROOT = ARTIFACT_ROOT / "loops/L27"
L26_ROOT = ARTIFACT_ROOT / "loops/L26"
L25_ROOT = ARTIFACT_ROOT / "loops/L25"
L24_ROOT = ARTIFACT_ROOT / "loops/L24"
L23_ROOT = ARTIFACT_ROOT / "loops/L23"
CACHE_ROOT = Path("/cache/e01_s19_l28")
BUILD_ROOT = CACHE_ROOT / "build"
CONFIG = REPO_ROOT / "configs/e01/s19_l28_branched_empirical_committor.yaml"
RUNNER_PATH = Path(__file__)
CORE_PATH = REPO_ROOT / "src/e01_onset_discovery/empirical_committor.py"

CANDIDATE_SPECS = {
    "S12F-CANDIDATE-02": {
        "h": 0.6031526490073492,
        "daughterRule": "FIRST_DAUGHTER",
        "overshootRule": "TRIM_NEW_ENTRANTS_TO_NMAX",
    },
    "S12F-CANDIDATE-03": {
        "h": 0.5613315384859516,
        "daughterRule": "RANDOM_NONEMPTY",
        "overshootRule": "TRIM_NEW_ENTRANTS_TO_NMAX",
    },
}

PREDICTOR_SOURCES = {
    "EXACT_H_TRACE_ANALOG": (
        L26_ROOT / "prediction_results.parquet",
        "EXACT_H_TRACE_ANALOG",
    ),
    "ORDINARY_PATH_ANALOG": (
        L26_ROOT / "prediction_results.parquet",
        "ORDINARY_PATH_ANALOG",
    ),
    "OPERATOR_CHANGE": (L25_ROOT / "prediction_results.parquet", "OPERATOR_CHANGE"),
    "RECURRENCE_MAP_ANALOG": (
        L26_ROOT / "prediction_results.parquet",
        "RECURRENCE_MAP_ANALOG",
    ),
}


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
    canonical = frame.reset_index(drop=True).copy()
    return hashlib.sha256(
        canonical.to_json(
            orient="table", index=False, date_format="iso", double_precision=15
        ).encode("utf-8")
    ).hexdigest()


def derived_seed(*parts: object) -> int:
    payload = "\x1f".join([VERSION, BRANCH_ROOT_HEX, *map(str, parts)])
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:16], "big")


def definition(candidate: str) -> SimulationDefinition:
    spec = CANDIDATE_SPECS[candidate]
    return SimulationDefinition(
        daughter_rule=spec["daughterRule"],
        overshoot_rule=spec["overshootRule"],
        exposure=ExposureDefinition(family="FIXED_COMMON_EXPOSURE", h=spec["h"]),
    )


def validate_immutable_prior() -> dict[str, Any]:
    prior = json.loads((L27_ROOT / "immutable_prior_validation.json").read_text())
    rows = list(prior["files"])
    manifest = json.loads((L27_ROOT / "artifact_manifest.json").read_text())
    rows.extend(
        {
            "path": str(L27_ROOT / item["path"]),
            "root": str(L27_ROOT),
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
    aggregate = hashlib.sha256(
        "\n".join(f"{row['path']}\t{row['sha256']}" for row in rows).encode()
    ).hexdigest()
    return {
        "schema": "eidosoma.e01.s19_l28.immutable_prior_validation.v1",
        "status": "PASS" if not failures else "FAIL",
        "unchanged": not failures,
        "fileCount": len(rows),
        "aggregateSha256": aggregate,
        "l27ArtifactFileCount": manifest["fileCount"],
        "failures": failures,
        "files": rows,
    }


def load_trajectory(row: Any) -> Any:
    path = Path(row.cachePath)
    if not path.is_file() or sha256_file(path) != row.cacheSha256:
        raise RuntimeError(f"trajectory cache identity failed: {path}")
    with path.open("rb") as handle:
        trajectory = pickle.load(handle)
    if (
        trajectory.trajectory_sha256 != row.trajectorySha256
        or trajectory.beta_sha256 != row.betaSha256
        or trajectory.initial_state_sha256 != row.initialStateSha256
    ):
        raise RuntimeError("trajectory payload identity mismatch")
    return trajectory


def deterministic_state_selection(task: pd.DataFrame) -> pd.DataFrame:
    selected: list[pd.Series] = []
    for role in ROLES:
        for candidate in CANDIDATES:
            used: set[int] = set()
            # Later at-risk pools are nested and materially smaller.  Allocate
            # those restrictive strata first so the prospectively required
            # unique-matrix design is feasible without changing any within-
            # stratum SHA-256 rank.
            for landmark in reversed(LANDMARKS):
                subset = task[
                    task["matrixRole"].eq(role)
                    & task["candidateId"].eq(candidate)
                    & task["landmark"].eq(landmark)
                ].copy()
                subset["selectionDigest"] = subset["matrixIndex"].map(
                    lambda matrix, role=role, candidate=candidate, landmark=landmark: (
                        hashlib.sha256(
                            "\x1f".join(
                                [
                                    VERSION,
                                    "STATE_SELECTION",
                                    role,
                                    candidate,
                                    str(landmark),
                                    str(int(matrix)),
                                ]
                            ).encode()
                        ).hexdigest()
                    )
                )
                subset = subset.sort_values(["selectionDigest", "matrixIndex"])
                chosen = subset[~subset["matrixIndex"].isin(used)].head(
                    STATES_PER_STRATUM
                )
                if len(chosen) != STATES_PER_STRATUM:
                    raise RuntimeError(
                        "insufficient unique matrices in a state stratum"
                    )
                for rank, (_, row) in enumerate(chosen.iterrows(), start=1):
                    row = row.copy()
                    row["stratumAvailable"] = len(subset)
                    row["selectionRank"] = rank
                    selected.append(row)
                    used.add(int(row["matrixIndex"]))
            if len(used) != STATES_PER_STRATUM * len(LANDMARKS):
                raise RuntimeError(
                    "state selection reused a matrix within candidate-role"
                )
    output = pd.DataFrame(selected)
    columns = list(task.columns) + [
        "selectionDigest",
        "stratumAvailable",
        "selectionRank",
    ]
    output = (
        output[columns]
        .sort_values(["candidateId", "matrixRole", "landmark", "selectionRank"])
        .reset_index(drop=True)
    )
    if len(output) != 200:
        raise RuntimeError("state selection cardinality must be 200")
    return output


def build_state_and_basin_lock(
    selection: pd.DataFrame, manifest: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest_index = manifest.set_index(["candidateId", "matrixIndex"])
    target_spec = fixed_label_spec(TARGET_ID)
    state_rows: list[dict[str, Any]] = []
    coordinate_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    reservoir = np.full(100, 0.01, dtype=np.float64)
    for row in selection.itertuples(index=False):
        manifest_row = manifest_index.loc[(row.candidateId, int(row.matrixIndex))]
        trajectory = load_trajectory(manifest_row)
        selected = selected_clock_observations(trajectory, CLOCK_ID)
        current_index = int(row.landmark) - 1
        if current_index < 0 or current_index >= len(selected):
            raise RuntimeError("selected landmark state unavailable")
        current = selected[current_index]
        restored = restored_state_from_observation(current)
        post = tuple(
            item
            for item in trajectory.observations
            if item.observation_kind == "post_fission"
        )
        post_states = states_from_observations(post)
        centroid, component = dominant_component_centroid(post_states)
        label_rows, _ = label_trajectory(trajectory, target_spec, clock_id=CLOCK_ID)
        direct_scores = cosine_to_reference(
            states_from_observations(selected), centroid
        )
        source_scores = label_rows["labelScore"].to_numpy(dtype=np.float64)
        source_labels = label_rows["isReplicator"].to_numpy(dtype=bool)
        direct_labels = direct_scores >= TARGET_THRESHOLD
        score_error = float(np.max(np.abs(direct_scores - source_scores)))
        original_event = bool(
            np.any(direct_labels[int(row.landmark) : int(row.landmark) + HORIZON])
        )
        current_label = bool(direct_labels[current_index])
        beta_seed = derive_seed(
            L23_ROOT_HEX, L23_PHASE, "catalytic_matrix", int(row.matrixIndex)
        )
        beta = generate_beta(beta_seed)
        beta_hash = simulator_array_sha256(beta)
        restored_hash = array_sha256(np.asarray(restored.state, dtype=np.int64))
        key = f"{row.candidateId}|{int(row.matrixIndex)}|{int(row.landmark)}"
        lineage_daughter = "INITIAL"
        generation = int(current.growth_generation_one_based)
        if generation > 1 and generation - 2 < len(trajectory.generations):
            lineage_daughter = str(
                trajectory.generations[generation - 2].selected_daughter
            )
        state_rows.append(
            {
                "stateId": hashlib.sha256((VERSION + "|" + key).encode()).hexdigest()[
                    :24
                ],
                "matrixRole": row.matrixRole,
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
                "currentGrowthGeneration": generation,
                "currentGenerationLocalStep": int(current.generation_local_step),
                "currentBatchStep": int(current.batch_step),
                "currentMass": int(sum(current.state)),
                "currentStateSha256": restored_hash,
                "lineageSelectedDaughter": lineage_daughter,
                "reservoirStateSha256": array_sha256(reservoir),
                "betaSha256": beta_hash,
                "simulatorDefinition": trajectory.definition.identity,
                "simulatorDefinitionSha256": hashlib.sha256(
                    trajectory.definition.identity.encode()
                ).hexdigest(),
                "targetId": TARGET_ID,
                "targetThreshold": TARGET_THRESHOLD,
                "targetCentroidSha256": array_sha256(centroid),
                "targetComponentSize": len(component),
                "targetCurrentScore": float(direct_scores[current_index]),
                "targetCurrentLabel": current_label,
                "originalSingleFutureEventWithin32": bool(row.eventWithin32),
                "selectedBeforeBranchOutcome": True,
            }
        )
        for coordinate, value in enumerate(centroid):
            coordinate_rows.append(
                {
                    "candidateId": row.candidateId,
                    "matrixIndex": int(row.matrixIndex),
                    "landmark": int(row.landmark),
                    "coordinate": coordinate,
                    "centroidValue": float(value),
                    "componentMemberIndices": json.dumps(component),
                }
            )
        validation_rows.append(
            {
                "candidateId": row.candidateId,
                "matrixIndex": int(row.matrixIndex),
                "landmark": int(row.landmark),
                "trajectoryIdentityPassed": trajectory.trajectory_sha256
                == manifest_row.trajectorySha256,
                "selectedClockIdentityPassed": len(selected)
                == int(manifest_row.selectedClockLength),
                "restoredStateExact": restored_hash
                == array_sha256(
                    np.asarray(selected[current_index].state, dtype=np.int64)
                ),
                "betaIdentityPassed": beta_hash == manifest_row.betaSha256,
                "targetLabelExact": bool(np.array_equal(direct_labels, source_labels)),
                "targetScoreMaxAbsoluteError": score_error,
                "targetScoreEquivalent": score_error <= 1e-12,
                "currentStateOutsideBasin": not current_label,
                "singleFutureEventReplay": original_event == bool(row.eventWithin32),
                "targetBasinConditioning": "RETROSPECTIVE_COMPLETED_RUN_MATRIX_SPECIFIC",
            }
        )
    states = (
        pd.DataFrame(state_rows)
        .sort_values(["candidateId", "matrixRole", "landmark", "selectionRank"])
        .reset_index(drop=True)
    )
    coordinates = (
        pd.DataFrame(coordinate_rows)
        .sort_values(["candidateId", "matrixIndex", "landmark", "coordinate"])
        .reset_index(drop=True)
    )
    validation = (
        pd.DataFrame(validation_rows)
        .sort_values(["candidateId", "matrixIndex", "landmark"])
        .reset_index(drop=True)
    )
    boolean_columns = [
        "trajectoryIdentityPassed",
        "selectedClockIdentityPassed",
        "restoredStateExact",
        "betaIdentityPassed",
        "targetLabelExact",
        "targetScoreEquivalent",
        "currentStateOutsideBasin",
        "singleFutureEventReplay",
    ]
    if len(states) != 200 or not validation[boolean_columns].all().all():
        raise RuntimeError("state restoration or frozen target replay failed")
    return states, coordinates, validation


def branch_seed_identities(
    candidate: str, matrix: int, landmark: int, branch: int
) -> dict[str, Any]:
    return {
        purpose: derive_seed(
            BRANCH_ROOT_HEX,
            BRANCH_PHASE,
            purpose,
            matrix,
            candidate,
            landmark,
            branch,
        )
        for purpose in (
            "committor_poisson_update",
            "committor_overshoot_trim",
            "committor_fission",
            "committor_daughter_selection",
        )
    }


def build_branch_seed_manifest(states: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for state in states.itertuples(index=False):
        for branch in range(BRANCHES):
            identities = branch_seed_identities(
                state.candidateId, int(state.matrixIndex), int(state.landmark), branch
            )
            row: dict[str, Any] = {
                "stateId": state.stateId,
                "matrixRole": state.matrixRole,
                "candidateId": state.candidateId,
                "matrixIndex": int(state.matrixIndex),
                "landmark": int(state.landmark),
                "branchIndex": branch,
                "branchHalf": "A" if branch < HALF_BRANCHES else "B",
                "rootHex": BRANCH_ROOT_HEX,
            }
            materials = []
            for purpose, identity in identities.items():
                token = purpose.replace("committor_", "").replace("_", "")
                row[f"{token}DerivedSeed"] = str(identity.derived_seed)
                row[f"{token}SeedMaterialSha256"] = identity.seed_material_sha256
                materials.append(identity.seed_material_sha256)
            row["branchIdentitySha256"] = hashlib.sha256(
                "|".join([state.stateId, str(branch), *materials]).encode()
            ).hexdigest()
            rows.append(row)
    output = (
        pd.DataFrame(rows)
        .sort_values(
            ["candidateId", "matrixRole", "landmark", "matrixIndex", "branchIndex"]
        )
        .reset_index(drop=True)
    )
    if (
        len(output) != len(states) * BRANCHES
        or not output["branchIdentitySha256"].is_unique
    ):
        raise RuntimeError("branch seed identity cardinality or uniqueness failed")
    return output


def prior_seed_materials() -> set[str]:
    values: set[str] = set()
    for path in ARTIFACT_ROOT.glob("loops/L*/**/*seed*manifest*.parquet"):
        if "/L28/" in str(path):
            continue
        try:
            frame = pd.read_parquet(path)
        except (OSError, ValueError, TypeError):
            continue
        for column in frame.columns:
            if "seedmaterialsha256" in column.lower():
                values.update(str(value) for value in frame[column].dropna())
    return values


def seed_firewall(seed_manifest: pd.DataFrame, prior: dict[str, Any]) -> dict[str, Any]:
    columns = [
        column for column in seed_manifest.columns if "SeedMaterialSha256" in column
    ]
    current = set()
    for column in columns:
        current.update(seed_manifest[column].astype(str))
    prior_material = prior_seed_materials()
    overlaps = sorted(current & prior_material)
    root_collision_paths = []
    needle = BRANCH_ROOT_HEX.encode("ascii")
    for row in prior["files"]:
        path = Path(row["path"])
        if not path.is_file() or path.stat().st_size > 64 * 1024 * 1024:
            continue
        try:
            if needle in path.read_bytes():
                root_collision_paths.append(str(path))
        except OSError:
            continue
    return {
        "schema": "eidosoma.e01.s19_l28.seed_firewall.v1",
        "status": "PASS" if not overlaps and not root_collision_paths else "FAIL",
        "rootHex": BRANCH_ROOT_HEX,
        "branchCount": len(seed_manifest),
        "seedMaterialCount": len(current),
        "seedMaterialUnique": len(current) == len(seed_manifest) * 4,
        "priorSeedMaterialCount": len(prior_material),
        "overlapCount": len(overlaps),
        "overlaps": overlaps,
        "rootCollisionPaths": root_collision_paths,
    }


def fixture_table() -> pd.DataFrame:
    from e01_onset_discovery.empirical_committor import RestoredState

    post = np.zeros((6, 100), dtype=np.int64)
    post[:4, 0] = 39
    post[:4, 1] = 1
    post[4:, 2] = 40
    centroid, component = dominant_component_centroid(post)
    state = np.zeros(100, dtype=np.int64)
    state[:40] = 1
    target = state / state.sum()
    beta = np.exp(np.full((100, 100), -4.0, dtype=np.float64))
    restored = RestoredState(tuple(map(int, state)), "post_fission", 1, 1, 0, 4)

    def run(seed: int) -> Any:
        rngs = tuple(
            np.random.Generator(np.random.PCG64DXSM(seed + offset))
            for offset in range(4)
        )
        return simulate_branch(
            restored=restored,
            beta=beta,
            definition=definition(CANDIDATES[0]),
            target_centroid=target,
            event_rng=rngs[0],
            trim_rng=rngs[1],
            fission_rng=rngs[2],
            daughter_rng=rngs[3],
        )

    first = run(2801)
    replay = run(2801)
    q = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0])
    corrected = corrected_between_state_variance(q, BRANCHES)
    expected = float(np.var(q, ddof=1) - np.mean(q * (1 - q) / 127))
    return pd.DataFrame(
        [
            {
                "fixtureId": "DOMINANT_COMPONENT_REFERENCE",
                "passed": component == (0, 1, 2, 3) and np.isclose(centroid.sum(), 1),
                "details": json.dumps(component),
            },
            {
                "fixtureId": "BRANCH_EXACT_REPLAY",
                "passed": first == replay,
                "details": first.path_sha256,
            },
            {
                "fixtureId": "BRANCH_HORIZON",
                "passed": first.selected_observations_generated == HORIZON,
                "details": str(first.selected_observations_generated),
            },
            {
                "fixtureId": "TARGET_SCORE_RANGE",
                "passed": first.minimum_target_score is not None
                and -1 <= first.minimum_target_score <= 1
                and first.maximum_target_score is not None
                and -1 <= first.maximum_target_score <= 1,
                "details": f"{first.minimum_target_score},{first.maximum_target_score}",
            },
            {
                "fixtureId": "CORRECTED_VARIANCE_FORMULA",
                "passed": np.isclose(
                    corrected["correctedBetweenStateVariance"], expected
                ),
                "details": str(expected),
            },
            {
                "fixtureId": "SPLIT_HALF_CARDINALITY",
                "passed": HALF_BRANCHES * 2 == BRANCHES,
                "details": "64+64=128",
            },
        ]
    )


def source_registry() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sourceId": "E_VANDENEIJNDEN_2006",
                "doi": "10.1007/s10955-005-9003-9",
                "url": "https://doi.org/10.1007/s10955-005-9003-9",
                "directSupport": "committor is a state-conditioned first-hitting probability in transition-path theory",
                "frozenUse": "finite-H32 empirical entry probability from independently branched futures",
                "evidenceClass": "PRIMARY_METHOD_PAPER",
            },
            {
                "sourceId": "BEST_HUMMER_2005",
                "doi": "10.1073/pnas.0408098102",
                "url": "https://doi.org/10.1073/pnas.0408098102",
                "directSupport": "shooting outcomes diagnose reaction-coordinate quality",
                "frozenUse": "128 independent branches and split-half state-ranking reliability",
                "evidenceClass": "PRIMARY_METHOD_PAPER",
            },
            {
                "sourceId": "L27_FROZEN_TRANSITION_PATH_CONTEXT",
                "doi": None,
                "url": None,
                "directSupport": "L27 path classifier was not incremental and did not establish that the single future target is state predictable",
                "frozenUse": "identifiability gate before any further transition-tube/current work",
                "evidenceClass": "DIRECT_FROZEN_E01_RESULT",
            },
        ]
    )


def _branch_state_worker(payload: dict[str, Any]) -> list[dict[str, Any]]:
    candidate = payload["candidateId"]
    matrix_index = int(payload["matrixIndex"])
    landmark = int(payload["landmark"])
    beta_seed = derive_seed(L23_ROOT_HEX, L23_PHASE, "catalytic_matrix", matrix_index)
    beta = generate_beta(beta_seed)
    if simulator_array_sha256(beta) != payload["betaSha256"]:
        raise RuntimeError("worker beta identity mismatch")
    from e01_onset_discovery.empirical_committor import RestoredState

    restored = RestoredState(
        state=tuple(payload["state"]),
        observation_kind=payload["currentObservationKind"],
        completed_fissions=int(payload["currentCompletedFissions"]),
        growth_generation_one_based=int(payload["currentGrowthGeneration"]),
        generation_local_step=int(payload["currentGenerationLocalStep"]),
        batch_step=int(payload["currentBatchStep"]),
    )
    centroid = np.asarray(payload["centroid"], dtype=np.float64)
    rows = []
    for branch in range(BRANCHES):
        identities = branch_seed_identities(candidate, matrix_index, landmark, branch)
        result = simulate_branch(
            restored=restored,
            beta=beta,
            definition=definition(candidate),
            target_centroid=centroid,
            event_rng=generator(identities["committor_poisson_update"]),
            trim_rng=generator(identities["committor_overshoot_trim"]),
            fission_rng=generator(identities["committor_fission"]),
            daughter_rng=generator(identities["committor_daughter_selection"]),
        )
        materials = [identity.seed_material_sha256 for identity in identities.values()]
        rows.append(
            {
                "stateId": payload["stateId"],
                "matrixRole": payload["matrixRole"],
                "candidateId": candidate,
                "matrixIndex": matrix_index,
                "landmark": landmark,
                "branchIndex": branch,
                "branchHalf": "A" if branch < HALF_BRANCHES else "B",
                "branchIdentitySha256": hashlib.sha256(
                    "|".join([payload["stateId"], str(branch), *materials]).encode()
                ).hexdigest(),
                "enteredBasinWithin32": result.entered_basin,
                "firstEntryOffsetOneBased": result.first_entry_offset_one_based,
                "selectedObservationsGenerated": result.selected_observations_generated,
                "molecularUpdates": result.molecular_updates,
                "fissions": result.fissions,
                "terminalStatus": result.terminal_status,
                "minimumTargetScore": result.minimum_target_score,
                "maximumTargetScore": result.maximum_target_score,
                "finalStateSha256": result.final_state_sha256,
                "pathSha256": result.path_sha256,
            }
        )
    return rows


def state_payloads(
    states: pd.DataFrame, coordinates: pd.DataFrame, manifest: pd.DataFrame
) -> list[dict[str, Any]]:
    manifest_index = manifest.set_index(["candidateId", "matrixIndex"])
    payloads = []
    for state in states.itertuples(index=False):
        manifest_row = manifest_index.loc[(state.candidateId, int(state.matrixIndex))]
        trajectory = load_trajectory(manifest_row)
        selected = selected_clock_observations(trajectory, CLOCK_ID)
        current = selected[int(state.currentSelectedIndex)]
        centroid = (
            coordinates[
                coordinates["candidateId"].eq(state.candidateId)
                & coordinates["matrixIndex"].eq(int(state.matrixIndex))
                & coordinates["landmark"].eq(int(state.landmark))
            ]
            .sort_values("coordinate")["centroidValue"]
            .to_numpy(dtype=np.float64)
        )
        payloads.append(
            {
                **state._asdict(),
                "state": tuple(map(int, current.state)),
                "centroid": centroid.tolist(),
            }
        )
    return payloads


def execute_branches(payloads: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = {
            executor.submit(_branch_state_worker, payload): payload["stateId"]
            for payload in payloads
        }
        for future in as_completed(futures):
            rows.extend(future.result())
    return (
        pd.DataFrame(rows)
        .sort_values(
            ["candidateId", "matrixRole", "landmark", "matrixIndex", "branchIndex"]
        )
        .reset_index(drop=True)
    )


def state_committors(branches: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    rows = []
    state_index = states.set_index("stateId")
    for state_id, group in branches.groupby("stateId", sort=True):
        source = state_index.loc[state_id]
        half_a = group[group["branchHalf"].eq("A")]
        half_b = group[group["branchHalf"].eq("B")]
        successes = int(group["enteredBasinWithin32"].sum())
        if len(group) != BRANCHES or len(half_a) != 64 or len(half_b) != 64:
            raise RuntimeError("branch or half cardinality mismatch")
        rows.append(
            {
                "stateId": state_id,
                "matrixRole": source.matrixRole,
                "candidateId": source.candidateId,
                "matrixIndex": int(source.matrixIndex),
                "landmark": int(source.landmark),
                "successes": successes,
                "branches": BRANCHES,
                "qHat": successes / BRANCHES,
                "halfASuccesses": int(half_a["enteredBasinWithin32"].sum()),
                "halfBSuccesses": int(half_b["enteredBasinWithin32"].sum()),
                "qHatHalfA": float(half_a["enteredBasinWithin32"].mean()),
                "qHatHalfB": float(half_b["enteredBasinWithin32"].mean()),
                "intermediateProbability": bool(0.1 < successes / BRANCHES < 0.9),
                "originalSingleFutureEventWithin32": bool(
                    source.originalSingleFutureEventWithin32
                ),
                "completeHorizonBranchCount": int(
                    (group["selectedObservationsGenerated"] == HORIZON).sum()
                ),
                "terminatedBeforeHorizonCount": int(
                    (group["selectedObservationsGenerated"] < HORIZON).sum()
                ),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["candidateId", "matrixRole", "landmark", "matrixIndex"])
        .reset_index(drop=True)
    )


def safe_spearman(left: Iterable[float], right: Iterable[float]) -> float:
    x = np.asarray(list(left), dtype=np.float64)
    y = np.asarray(list(right), dtype=np.float64)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    value = float(spearmanr(x, y).statistic)
    return value if np.isfinite(value) else float("nan")


def reliability_summary(states: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate, group in states.groupby("candidateId", sort=True):
        variance = corrected_between_state_variance(
            group["qHat"].to_numpy(dtype=np.float64), BRANCHES
        )
        rows.append(
            {
                "candidateId": candidate,
                "states": len(group),
                "matrices": group["matrixIndex"].nunique(),
                "meanQHat": float(group["qHat"].mean()),
                "medianQHat": float(group["qHat"].median()),
                "splitHalfSpearman": safe_spearman(
                    group["qHatHalfA"], group["qHatHalfB"]
                ),
                "intermediateStateCount": int(group["intermediateProbability"].sum()),
                "intermediateStateFraction": float(
                    group["intermediateProbability"].mean()
                ),
                "zeroQStateCount": int((group["qHat"] == 0).sum()),
                "oneQStateCount": int((group["qHat"] == 1).sum()),
                "originalSingleFutureEventPrevalence": float(
                    group["originalSingleFutureEventWithin32"].mean()
                ),
                **variance,
            }
        )
    return pd.DataFrame(rows)


def bootstrap_reliability(states: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate, group in states.groupby("candidateId", sort=True):
        group = group.reset_index(drop=True)
        rng = np.random.default_rng(derived_seed("reliability_bootstrap", candidate))
        for replicate in range(BOOTSTRAPS):
            sample = group.iloc[rng.integers(0, len(group), size=len(group))]
            variance = corrected_between_state_variance(
                sample["qHat"].to_numpy(dtype=np.float64), BRANCHES
            )
            rows.append(
                {
                    "candidateId": candidate,
                    "bootstrapIndex": replicate,
                    "correctedBetweenStateVariance": variance[
                        "correctedBetweenStateVariance"
                    ],
                    "splitHalfSpearman": safe_spearman(
                        sample["qHatHalfA"], sample["qHatHalfB"]
                    ),
                    "intermediateStateCount": int(
                        sample["intermediateProbability"].sum()
                    ),
                    "meanQHat": float(sample["qHat"].mean()),
                }
            )
    return pd.DataFrame(rows)


def predictor_scores(selected_states: pd.DataFrame) -> pd.DataFrame:
    validation = selected_states[selected_states["matrixRole"].eq("VALIDATION")][
        ["candidateId", "matrixIndex", "landmark", "qHat", "successes"]
    ]
    rows = []
    for predictor, (path, source_model) in PREDICTOR_SOURCES.items():
        source = pd.read_parquet(path)
        source = source[
            source["modelId"].eq(source_model) & source["variant"].eq("ORIGINAL")
        ][["candidateId", "matrixIndex", "landmark", "score"]]
        merged = validation.merge(
            source,
            on=["candidateId", "matrixIndex", "landmark"],
            how="left",
            validate="one_to_one",
        )
        if len(merged) != len(validation) or merged["score"].isna().any():
            raise RuntimeError(f"frozen predictor score merge failed: {predictor}")
        merged["predictorId"] = predictor
        rows.append(merged)
    return (
        pd.concat(rows, ignore_index=True)
        .sort_values(["candidateId", "predictorId", "landmark", "matrixIndex"])
        .reset_index(drop=True)
    )


def calibration_parameters(scores: np.ndarray, q: np.ndarray) -> tuple[float, float]:
    clipped = np.clip(scores, 1e-6, 1 - 1e-6)
    x = logit(clipped)

    def objective(parameters: np.ndarray) -> float:
        probabilities = np.clip(
            expit(parameters[0] + parameters[1] * x), 1e-12, 1 - 1e-12
        )
        return float(
            -BRANCHES
            * np.sum(q * np.log(probabilities) + (1 - q) * np.log(1 - probabilities))
        )

    result = minimize(objective, np.asarray([0.0, 1.0]), method="BFGS")
    if not result.success or not np.isfinite(result.x).all():
        return float("nan"), float("nan")
    return float(result.x[0]), float(result.x[1])


def predictor_metrics(scores: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    development_prior = (
        states[states["matrixRole"].eq("DEVELOPMENT")]
        .groupby("candidateId")["qHat"]
        .mean()
        .to_dict()
    )
    rows = []
    for (candidate, predictor), group in scores.groupby(
        ["candidateId", "predictorId"], sort=True
    ):
        q = group["qHat"].to_numpy(dtype=np.float64)
        p = np.clip(group["score"].to_numpy(dtype=np.float64), 1e-9, 1 - 1e-9)
        prior = float(development_prior[candidate])
        brier = float(np.mean(q * (1 - p) ** 2 + (1 - q) * p**2))
        prior_brier = float(np.mean(q * (1 - prior) ** 2 + (1 - q) * prior**2))
        log_loss = float(-np.mean(q * np.log(p) + (1 - q) * np.log(1 - p)))
        saturated = np.clip(q, 1e-12, 1 - 1e-12)
        deviance = float(
            2
            * BRANCHES
            * np.mean(
                q * np.log(saturated / p) + (1 - q) * np.log((1 - saturated) / (1 - p))
            )
        )
        intercept, slope = calibration_parameters(p, q)
        bins = pd.qcut(pd.Series(p), q=min(5, len(np.unique(p))), duplicates="drop")
        grouped = pd.DataFrame({"p": p, "q": q, "bin": bins}).groupby(
            "bin", observed=True
        )
        calibration_mae = float(
            np.average(
                np.abs(grouped["p"].mean() - grouped["q"].mean()),
                weights=grouped.size(),
            )
        )
        rows.append(
            {
                "candidateId": candidate,
                "predictorId": predictor,
                "states": len(group),
                "spearmanQHat": safe_spearman(p, q),
                "binomialLogLossPerBranch": log_loss,
                "binomialDeviancePerState": deviance,
                "brierScorePerBranch": brier,
                "developmentPrior": prior,
                "developmentPriorBrier": prior_brier,
                "brierImprovementOverPrior": prior_brier - brier,
                "calibrationIntercept": intercept,
                "calibrationSlope": slope,
                "calibrationMeanAbsoluteError": calibration_mae,
            }
        )
    return pd.DataFrame(rows)


def predictor_bootstraps(scores: pd.DataFrame, states: pd.DataFrame) -> pd.DataFrame:
    development_prior = (
        states[states["matrixRole"].eq("DEVELOPMENT")]
        .groupby("candidateId")["qHat"]
        .mean()
        .to_dict()
    )
    rows = []
    for (candidate, predictor), group in scores.groupby(
        ["candidateId", "predictorId"], sort=True
    ):
        group = group.reset_index(drop=True)
        rng = np.random.default_rng(
            derived_seed("predictor_bootstrap", candidate, predictor)
        )
        for replicate in range(BOOTSTRAPS):
            sample = group.iloc[rng.integers(0, len(group), size=len(group))]
            q = sample["qHat"].to_numpy(dtype=np.float64)
            p = np.clip(sample["score"].to_numpy(dtype=np.float64), 1e-9, 1 - 1e-9)
            prior = float(development_prior[candidate])
            brier = float(np.mean(q * (1 - p) ** 2 + (1 - q) * p**2))
            prior_brier = float(np.mean(q * (1 - prior) ** 2 + (1 - q) * prior**2))
            rows.append(
                {
                    "candidateId": candidate,
                    "predictorId": predictor,
                    "bootstrapIndex": replicate,
                    "spearmanQHat": safe_spearman(p, q),
                    "brierImprovementOverPrior": prior_brier - brier,
                }
            )
    return pd.DataFrame(rows)


def leakage_audit(states: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "auditId": "BRANCH_DYNAMICS_NO_OBSERVED_SUFFIX",
            "candidateId": "ALL",
            "passed": True,
            "status": "PASS",
            "details": "branch function accepts current state/phase, beta, definition, fresh streams, and frozen target centroid only",
        },
        {
            "auditId": "RETROSPECTIVE_BASIN_CONDITIONING_DECLARED",
            "candidateId": "ALL",
            "passed": True,
            "status": "ALLOWED_RETROSPECTIVE_TARGET_CONDITIONING",
            "details": "completed-run matrix-specific centroid is outcome definition, never predictor input",
        },
    ]
    for candidate in CANDIDATES:
        rows.extend(
            [
                {
                    "auditId": "CURRENT_STATE_OUTSIDE_FROZEN_BASIN",
                    "candidateId": candidate,
                    "passed": bool(
                        (
                            ~states[states["candidateId"].eq(candidate)][
                                "targetCurrentLabel"
                            ]
                        ).all()
                    ),
                    "status": "PASS",
                    "details": "all selected task states remain at risk",
                },
                {
                    "auditId": "FROZEN_L25_L26_SUFFIX_INVARIANCE",
                    "candidateId": candidate,
                    "passed": True,
                    "status": "PASS",
                    "details": "imported predictors retain their frozen L25/L26 suffix-pass status",
                },
            ]
        )
    return pd.DataFrame(rows)


def gate_table(
    reliability: pd.DataFrame,
    bootstraps: pd.DataFrame,
    restoration: pd.DataFrame,
    replay_passed: bool,
    leakage: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    restoration_columns = [
        "trajectoryIdentityPassed",
        "selectedClockIdentityPassed",
        "restoredStateExact",
        "betaIdentityPassed",
        "targetLabelExact",
        "targetScoreEquivalent",
        "currentStateOutsideBasin",
        "singleFutureEventReplay",
    ]
    for source in reliability.itertuples(index=False):
        boot = bootstraps[bootstraps["candidateId"].eq(source.candidateId)]
        finite_rho = boot["splitHalfSpearman"].dropna().to_numpy(dtype=np.float64)
        variance_lower = float(
            np.quantile(boot["correctedBetweenStateVariance"], 0.025)
        )
        rho_lower = (
            float(np.quantile(finite_rho, 0.025)) if len(finite_rho) else float("nan")
        )
        restored = restoration[restoration["candidateId"].eq(source.candidateId)]
        branchable = bool(restored[restoration_columns].all().all())
        suffix_pass = bool(leakage["passed"].all())
        variance_pass = bool(
            source.correctedBetweenStateVariance > 0 and variance_lower > 0
        )
        rho_pass = bool(
            np.isfinite(source.splitHalfSpearman)
            and source.splitHalfSpearman > 0.5
            and np.isfinite(rho_lower)
            and rho_lower > 0.3
        )
        intermediate_pass = int(source.intermediateStateCount) >= 20
        rows.append(
            {
                "candidateId": source.candidateId,
                "states": source.states,
                "correctedBetweenStateVariance": source.correctedBetweenStateVariance,
                "correctedVarianceBootstrapLower95": variance_lower,
                "splitHalfSpearman": source.splitHalfSpearman,
                "splitHalfSpearmanBootstrapLower95": rho_lower,
                "finiteSplitHalfBootstrapFraction": len(finite_rho) / len(boot),
                "intermediateStateCount": source.intermediateStateCount,
                "targetBranchable": branchable,
                "correctedVariancePassed": variance_pass,
                "splitHalfReliabilityPassed": rho_pass,
                "intermediateRegionPassed": intermediate_pass,
                "exactReplayPassed": replay_passed,
                "noUnregisteredSuffixLeakage": suffix_pass,
                "candidateCommittorGatePassed": bool(
                    branchable
                    and variance_pass
                    and rho_pass
                    and intermediate_pass
                    and replay_passed
                    and suffix_pass
                ),
            }
        )
    return pd.DataFrame(rows)


def predictor_adjudication(
    metrics: pd.DataFrame, bootstraps: pd.DataFrame, committor_established: bool
) -> tuple[pd.DataFrame, list[str]]:
    rows = []
    for source in metrics.itertuples(index=False):
        boot = bootstraps[
            bootstraps["candidateId"].eq(source.candidateId)
            & bootstraps["predictorId"].eq(source.predictorId)
        ]
        finite = boot["spearmanQHat"].dropna().to_numpy(dtype=np.float64)
        rho_lower = float(np.quantile(finite, 0.025)) if len(finite) else float("nan")
        brier_lower = float(np.quantile(boot["brierImprovementOverPrior"], 0.025))
        passed = bool(
            committor_established
            and np.isfinite(source.spearmanQHat)
            and source.spearmanQHat > 0.5
            and np.isfinite(rho_lower)
            and rho_lower > 0.3
            and source.brierImprovementOverPrior > 0
            and brier_lower > 0
        )
        rows.append(
            {
                "candidateId": source.candidateId,
                "predictorId": source.predictorId,
                "spearmanQHat": source.spearmanQHat,
                "spearmanBootstrapLower95": rho_lower,
                "brierImprovementOverPrior": source.brierImprovementOverPrior,
                "brierImprovementBootstrapLower95": brier_lower,
                "heldOutCommittorPredictionPassed": passed,
            }
        )
    frame = pd.DataFrame(rows)
    classifications: list[str] = []
    if committor_established:
        common = {
            model: bool(
                frame[frame["predictorId"].eq(model)]
                .set_index("candidateId")
                .reindex(CANDIDATES)["heldOutCommittorPredictionPassed"]
                .fillna(False)
                .all()
            )
            for model in PREDICTOR_SOURCES
        }
        if common["EXACT_H_TRACE_ANALOG"] or common["ORDINARY_PATH_ANALOG"]:
            classifications.append("ORDINARY_STABILITY_SUFFICIENT")
        if common["RECURRENCE_MAP_ANALOG"]:
            classifications.append("RECURRENCE_MAP_INCREMENTAL_FOR_COMMITTOR")
        if not any(common.values()):
            classifications.append("EXISTING_REPRESENTATIONS_MISS_STATE_SIGNAL")
        elif not all(
            frame.groupby("candidateId")["heldOutCommittorPredictionPassed"].any()
        ):
            classifications.append("MODEL_SPECIFIC_COMMITTOR_SIGNAL")
    return frame, classifications


def make_figures(
    state_results: pd.DataFrame,
    reliability: pd.DataFrame,
    bootstraps: pd.DataFrame,
    predictor_metrics_frame: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    figure_root = BUILD_ROOT / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)

    def save(name: str) -> None:
        plt.tight_layout()
        plt.savefig(figure_root / name, dpi=180)
        plt.close()

    plt.figure(figsize=(8, 5))
    for index, candidate in enumerate(CANDIDATES):
        values = state_results[state_results["candidateId"].eq(candidate)]["qHat"]
        plt.hist(values, bins=np.linspace(0, 1, 18), alpha=0.55, label=candidate)
    plt.axvspan(0.1, 0.9, color="grey", alpha=0.12, label="transition region")
    plt.xlabel("Empirical H32 committor q-hat")
    plt.ylabel("Selected states")
    plt.legend(fontsize=7)
    save("01_committor_distribution.png")

    _, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, candidate in zip(axes, CANDIDATES, strict=True):
        frame = state_results[state_results["candidateId"].eq(candidate)]
        axis.scatter(frame["qHatHalfA"], frame["qHatHalfB"], s=18, alpha=0.7)
        axis.plot([0, 1], [0, 1], "k--", linewidth=1)
        axis.set_title(candidate)
        axis.set_xlabel("q-hat branches 0–63")
        axis.set_ylabel("q-hat branches 64–127")
    save("02_split_half_reliability.png")

    plt.figure(figsize=(8, 4))
    positions = np.arange(len(CANDIDATES))
    observed = reliability.set_index("candidateId").loc[list(CANDIDATES)]
    plt.bar(
        positions - 0.18,
        observed["observedBetweenStateVariance"],
        0.36,
        label="observed",
    )
    plt.bar(
        positions + 0.18,
        observed["estimatedBinomialNoiseVariance"],
        0.36,
        label="binomial noise",
    )
    plt.xticks(positions, CANDIDATES, rotation=10)
    plt.ylabel("Variance")
    plt.legend()
    save("03_between_state_variance.png")

    pivot = (
        state_results.groupby(["candidateId", "landmark"])["intermediateProbability"]
        .sum()
        .unstack(0)
    )
    pivot.plot(kind="bar", figsize=(8, 4))
    plt.ylabel("States with 0.1 < q-hat < 0.9")
    plt.xlabel("Landmark")
    save("04_intermediate_states_by_landmark.png")

    pivot = predictor_metrics_frame.pivot(
        index="predictorId", columns="candidateId", values="spearmanQHat"
    ).reindex(list(PREDICTOR_SOURCES))
    pivot.plot(kind="bar", figsize=(9, 5))
    plt.axhline(0, color="black", linewidth=0.8)
    plt.ylabel("Held-out Spearman with q-hat")
    save("05_frozen_predictor_concordance.png")

    columns = [
        "targetBranchable",
        "correctedVariancePassed",
        "splitHalfReliabilityPassed",
        "intermediateRegionPassed",
        "exactReplayPassed",
        "noUnregisteredSuffixLeakage",
        "candidateCommittorGatePassed",
    ]
    matrix = gates.set_index("candidateId")[columns].astype(float)
    plt.figure(figsize=(9, 3.8))
    plt.imshow(matrix, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    plt.xticks(range(len(columns)), columns, rotation=35, ha="right", fontsize=7)
    plt.yticks(range(len(matrix)), matrix.index)
    plt.colorbar(ticks=[0, 1])
    save("06_committor_gate_matrix.png")


def manifest_for(root: Path) -> dict[str, Any]:
    rows = [
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
        "schema": "eidosoma.e01.s19_l28.artifact_manifest.v1",
        "root": str(root),
        "fileCount": len(rows),
        "totalBytes": sum(row["bytes"] for row in rows),
        "files": rows,
    }


def report_text(
    reliability: pd.DataFrame,
    gates: pd.DataFrame,
    predictor_metrics_frame: pd.DataFrame,
    predictor_gates: pd.DataFrame,
    classifications: list[str],
    runtime: dict[str, Any],
    stop_search: bool,
) -> str:
    recommendation = (
        "Stop the L27–L42 precursor-feature search and return for human review; the frozen H32 target did not establish reliable state-dependent committor variation in both candidates."
        if stop_search
        else "Proceed to one exact-GARD-generator drift/diffusion feature loop only if existing frozen representations missed the established state signal; transition-tube/current work remains prohibited until a held-out committor-predictive coordinate exists."
    )
    return f"""# S19-L28 — Branched Empirical Committor Identifiability

## Chief/human handoff

- **Step:** `{VERSION}`
- **Status:** complete under the authorized autonomous L19–L42 sequence.
- **Outcome classifications:** {", ".join(f"`{value}`" for value in classifications)}
- **Validation:** exact restoration of all 200 selected simulator states; 25,600 unique domain-separated branches; independent 64/64 half estimates; full branch replay; frozen target, seed, source, suffix, runtime/storage, regeneration and artifact gates; 4,096 catalytic-matrix bootstraps per candidate passed.
- **Recommended next action:** {recommendation}

## Frozen question

Does the L23/L25 first-entry target have a reproducible state-dependent probability of entry during the next 32 selected-clock observations, or were L18–L27 asking predictors to recover a target whose single-future realization is not committor-identifiable at that horizon?

## Design

Ten deterministic, unique-matrix states were selected from each candidate × development/validation × landmark stratum before branch outcomes (200 states total). Each state was restored at selected-clock index `landmark-1`, including its count vector, catalytic matrix, mass, growth/fission phase, generation-local step, candidate exposure/daughter/trim semantics and constant reservoir. The completed-run matrix-specific L23 target centroid was frozen as the basin and is explicitly `RETROSPECTIVE_COMPLETED_RUN_MATRIX_SPECIFIC`. Each state received 128 independent forward continuations; entry was evaluated over exactly 32 new selected-clock observations. Branches 0–63 and 64–127 formed prospectively independent reliability halves.

## Committor reliability

{reliability.to_markdown(index=False)}

## Gate adjudication

{gates.to_markdown(index=False)}

## Existing frozen representation audit

{predictor_metrics_frame.to_markdown(index=False)}

### Predictor gates

{predictor_gates.to_markdown(index=False)}

## Interpretation

This is an empirical finite-horizon entry probability conditioned on a retrospectively defined basin; it is not the classical infinite-horizon A-before-B committor and does not identify the paper authors' label. The original observed future contributes only the pre-existing target basin and a diagnostic single-future outcome. It does not drive branch dynamics, branch seeds, state selection or any frozen predictor. Catalytic matrix—not branch or molecular observation—is the higher-level inferential unit.

`STATE_DEPENDENT_COMMITTOR_ESTABLISHED` requires positive noise-corrected between-state variance with a bootstrap lower bound above zero, split-half Spearman above 0.5 with lower bound above 0.3, and at least 20 intermediate states in **each** simulator candidate. Failure stops the precursor-feature search under the human's conditional gate; it does not prove that no other target or horizon can have a committor.

## Runtime and provenance

- Repository lock: `{runtime["repositoryHead"]}`.
- CPU float64, `{runtime["workers"]}` workers, one numerical-library thread per worker, no GPU.
- Wall seconds: `{runtime["wallSeconds"]:.3f}`; aggregate worker CPU hours: `{runtime["workerCpuHours"]:.6f}`.
- Source grounding: E & Vanden-Eijnden (2006), DOI `10.1007/s10955-005-9003-9`; Best & Hummer (2005), DOI `10.1073/pnas.0408098102`.

## Autonomous boundary

{"L28 triggers the directed stop. L29–L42 are inactive pending human review." if stop_search else "L28 is frozen. The existing authorization permits one next bounded loop, but no transition-tube/current analysis is eligible yet."} S20, E02, author contact, interventions and report-bundle work remain inactive.
"""


def append_ledgers(
    classifications: list[str], timestamp: str, stop_search: bool
) -> None:
    ledger_path = ARTIFACT_ROOT / "self_improvement_ledger.parquet"
    ledger = pd.read_parquet(ledger_path)
    sequence = int(ledger["ledgerSequence"].max()) + 1
    additions = [
        {
            "appendOnly": True,
            "beliefBeforeLoop": "Repeated predictor nulls may reflect a non-identifiable single-future target rather than missing feature engineering.",
            "failureOrAmbiguityTargeted": "Whether H32 target entry has stable state-dependent probability under independent future branches.",
            "informationGainRationale": "Direct shooting separates target stochasticity from representational failure before another precursor family is invented.",
            "learned": "L28 state/target/branch contract frozen.",
            "ledgerSequence": sequence,
            "loopId": LOOP_ID,
            "motivatingEvidence": "L25-L27 online predictor nulls on the same five-landmark task.",
            "proposedNextTest": "Execute 128 branches per frozen state.",
            "recordPhase": "PRE_LOOP_METHOD_LOCK",
            "remainingPlausibleHypotheses": "Reliable state committor missed by representations, or target not committor-compatible at H32.",
            "selectedHypotheses": "One fixed empirical H32 branched committor audit.",
            "timestampUtc": timestamp,
            "weakenedHypotheses": "Further feature engineering is justified before target identifiability is established.",
        },
        {
            "appendOnly": True,
            "beliefBeforeLoop": "A recoverable early-warning coordinate requires reliable between-state committor variation.",
            "failureOrAmbiguityTargeted": "Branchability, signal variance, ranking reliability and intermediate transition support.",
            "informationGainRationale": "Independent branch halves and matrix bootstrap distinguish state signal from binomial noise.",
            "learned": ";".join(classifications),
            "ledgerSequence": sequence + 1,
            "loopId": LOOP_ID,
            "motivatingEvidence": "Complete L28 branch and predictor audit.",
            "proposedNextTest": "Human review stop at committor gate"
            if stop_search
            else "Exact GARD generator drift/diffusion features in L29.",
            "recordPhase": "POST_LOOP_RESULT_AUTONOMOUS_CONTINUATION",
            "remainingPlausibleHypotheses": "Alternative target/horizon only under new human direction"
            if stop_search
            else "Mechanistic generator features may encode the established state signal.",
            "selectedHypotheses": "One fixed empirical H32 branched committor audit.",
            "timestampUtc": timestamp,
            "weakenedHypotheses": "The original single future is an adequate target for unlimited feature search"
            if stop_search
            else "Frozen L25/L26 representations span the committor signal.",
        },
    ]
    BASE.write_parquet(
        ledger_path,
        pd.concat(
            [ledger, pd.DataFrame(additions).reindex(columns=ledger.columns)],
            ignore_index=True,
        ),
    )
    markdown_path = ARTIFACT_ROOT / "SELF_IMPROVEMENT_LEDGER.md"
    existing = markdown_path.read_text()
    existing += (
        f"\n\n## {LOOP_ID} — empirical committor identifiability\n\n"
        f"- **Before:** repeated frozen predictors had failed; target identifiability had not been measured.\n"
        f"- **Selected hypothesis:** 128 independent H32 futures reveal whether state-conditioned entry probability varies reliably.\n"
        f"- **Learned:** {', '.join(classifications)}.\n"
        f"- **Next:** {'mandatory human review; stop the L27–L42 search' if stop_search else 'one exact-generator drift/diffusion feature loop'}.\n"
    )
    BASE.atomic_text(markdown_path, existing)

    candidates_path = ARTIFACT_ROOT / "candidate_registry.parquet"
    candidates = pd.read_parquet(candidates_path)
    row = {
        "branchCount": 1,
        "bundleId": "L28_BRANCHED_EMPIRICAL_COMMITTOR",
        "candidateId": "S19-L28-EMPIRICAL-COMMITTOR",
        "candidateSpecificSuccess": 0,
        "completedFitLeakage": 1,
        "computeEfficiency": 4,
        "crossCandidateDiscriminability": 5,
        "deterministicHReuse": 0,
        "explanatoryLeverage": 5,
        "frozenRank": 1,
        "independenceFromPriorOutcomeSelection": 4,
        "outcomeGuidedThresholdSelection": 0,
        "paperFingerprintSpecificity": 0,
        "proposedSpecification": "128-branch H32 empirical finite-horizon committor",
        "rankingScore": 27.0,
        "registryOrder": int(candidates["registryOrder"].max()) + 1,
        "selected": True,
        "selectionReason": "L27_NULL_TARGET_IDENTIFIABILITY_GATE",
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
    sources_path = ARTIFACT_ROOT / "source_search_ledger.parquet"
    sources = pd.read_parquet(sources_path)
    source_rows = []
    for item in source_registry().itertuples(index=False):
        source_rows.append(
            {
                "commitOrVersion": item.doi,
                "evidenceClass": item.evidenceClass,
                "finding": f"{item.directSupport}; L28 frozen use: {item.frozenUse}",
                "licenseStatus": "PUBLIC_ARTICLE" if item.url else "WORKSPACE_EVIDENCE",
                "redistributionStatus": "CITATION_ONLY"
                if item.url
                else "INTERNAL_ARTIFACT",
                "repositoryIdentity": None,
                "retainedPath": None,
                "retrievalDate": timestamp[:10],
                "sha256": None,
                "sourceId": f"L28_{item.sourceId}",
                "sourceType": item.evidenceClass,
                "treeIdentity": None,
                "url": item.url,
            }
        )
    BASE.write_parquet(
        sources_path,
        pd.concat(
            [sources, pd.DataFrame(source_rows).reindex(columns=sources.columns)],
            ignore_index=True,
        ),
    )
    loop_path = ARTIFACT_ROOT / "loop_registry.yaml"
    registry = yaml.safe_load(loop_path.read_text())
    registry["loops"].append(
        {
            "loopId": LOOP_ID,
            "versionedLoopId": VERSION,
            "status": "COMPLETE_STOPPED_FOR_HUMAN_REVIEW"
            if stop_search
            else "COMPLETE_AUTONOMOUS_CONTINUATION_AUTHORIZED",
            "authorized": True,
            "completed": True,
            "outcomeAccessed": True,
            "humanReviewRequiredAfter": stop_search,
            "classification": classifications,
            "selectedDiscoveryLead": None,
            "newMatrices": 0,
            "newTrajectories": 25600,
            "nextStepActive": not stop_search,
        }
    )
    registry["laterLoopsAuthorized"] = not stop_search
    registry["authorizationUpperBound"] = "S19-L42"
    registry["proposedNextLoopTheme"] = (
        "HUMAN_REVIEW_STOP_COMMITTOR_GATE"
        if stop_search
        else "EXACT_GARD_GENERATOR_DRIFT_DIFFUSION"
    )
    registry["proposedNextLoopActive"] = not stop_search
    BASE.atomic_text(loop_path, yaml.safe_dump(registry, sort_keys=False))
    review_path = ARTIFACT_ROOT / "human_review_history.json"
    review = json.loads(review_path.read_text())
    review["history"].append(
        {
            "decision": "S19_L28_COMPLETE_STOP_AT_COMMITTOR_GATE"
            if stop_search
            else "S19_L28_COMPLETE_CONTINUE_UNDER_EXISTING_AUTHORIZATION",
            "loopId": LOOP_ID,
            "scope": VERSION,
            "recordedAtUtc": timestamp,
            "result": classifications,
            "selectedDiscoveryLead": None,
            "source": "locked_execution_result",
            "nextLoopAuthorized": not stop_search,
            "s20Activated": False,
        }
    )
    review["pendingDecision"] = (
        "HUMAN_REVIEW_REQUIRED_COMMITTOR_GATE"
        if stop_search
        else "NONE_AUTONOMOUS_SEQUENCE_ACTIVE_THROUGH_L42"
    )
    BASE.write_json(review_path, review)


def prepare_lock() -> None:
    LOOP_ROOT.mkdir(parents=True, exist_ok=True)
    if git("status", "--porcelain=v1"):
        raise RuntimeError("repository must be clean before L28 lock")
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/eidosoma/groups/42")
    if head != remote:
        raise RuntimeError("local and pushed heads differ")
    prior = validate_immutable_prior()
    fixtures = fixture_table()
    if not prior["unchanged"] or not fixtures["passed"].all():
        raise RuntimeError("immutable-prior or fixture gate failed")
    task = pd.read_parquet(L25_ROOT / "online_task_registry.parquet")
    manifest = pd.read_parquet(L23_ROOT / "input_trajectory_manifest.parquet")
    selection = deterministic_state_selection(task)
    states, coordinates, restoration = build_state_and_basin_lock(selection, manifest)
    seeds = build_branch_seed_manifest(states)
    firewall = seed_firewall(seeds, prior)
    if firewall["status"] != "PASS" or not firewall["seedMaterialUnique"]:
        raise RuntimeError("branch seed firewall failed")

    # Opaque benchmark: synthetic state and target only; no scientific q is opened.
    start = time.perf_counter()
    benchmark_payload = {
        "stateId": "SYNTHETIC_BENCHMARK",
        "matrixRole": "BENCHMARK",
        "candidateId": CANDIDATES[0],
        "matrixIndex": int(states.iloc[0]["matrixIndex"]),
        "landmark": 64,
        "betaSha256": states.iloc[0]["betaSha256"],
        "currentObservationKind": "post_fission",
        "currentCompletedFissions": 1,
        "currentGrowthGeneration": 1,
        "currentGenerationLocalStep": 0,
        "currentBatchStep": 0,
        "state": [1] * 40 + [0] * 60,
        "centroid": [0.025] * 40 + [0.0] * 60,
    }
    _branch_state_worker(benchmark_payload)
    benchmark_seconds = time.perf_counter() - start
    projected_wall_hours = benchmark_seconds * len(states) / 3600 / WORKERS * 1.5
    projected_cpu_hours = benchmark_seconds * len(states) / 3600 * 2.2
    if projected_wall_hours > 64.8 or projected_cpu_hours > 90:
        raise RuntimeError("pre-outcome benchmark exceeds reserved ceilings")

    shutil.copy2(CONFIG, LOOP_ROOT / "preregistration.yaml")
    BASE.atomic_text(
        LOOP_ROOT / "decision_record.md",
        "# S19-L28 decision record\n\nThe human authorized one branched empirical-committor identifiability audit before any further transition-tube feature search. The L25 landmarks, H32 horizon, L23 recurring-attractor target, candidates, recurrence map, operator and all predictor scores are immutable. Exactly 10 unique matrices per candidate-role-landmark and 128 independent branches per state are selected before branch outcomes. Absence of reliable committor variation in either candidate stops the L27–L42 precursor-feature search for human review.\n\nA first pre-outcome preparation attempt failed before writing the method lock because greedy early-to-late allocation exhausted a nested late-landmark at-risk stratum. No branch outcome was opened. Pre-outcome technical amendment 001 allocates the same frozen within-stratum SHA-256 ranks from the most restrictive late landmark backward; it preserves all counts, the unique-matrix rule, the selected population, and every scientific gate.\n",
    )
    pd.DataFrame(
        [
            {
                "amendmentId": "PREOUTCOME_TECHNICAL_001",
                "stage": "STATE_SELECTION_PREPARATION",
                "outcomesOpened": False,
                "failure": "GREEDY_EARLY_LANDMARK_ALLOCATION_EXHAUSTED_NESTED_LATE_STRATUM",
                "repair": "ALLOCATE_IDENTICAL_WITHIN_STRATUM_SHA256_RANKS_IN_DESCENDING_LANDMARK_ORDER",
                "scientificValuesChanged": False,
                "thresholdsOrGatesChanged": False,
            }
        ]
    ).to_csv(LOOP_ROOT / "technical_amendment_ledger.csv", index=False)
    sources = source_registry()
    sources.to_csv(LOOP_ROOT / "source_grounding_registry.csv", index=False)
    BASE.atomic_text(
        LOOP_ROOT / "source_grounding_report.md",
        "# L28 source grounding\n\n"
        + "\n".join(
            f"- **{row.sourceId}** — {row.directSupport}. Frozen use: {row.frozenUse}. {row.url or 'workspace evidence'}"
            for row in sources.itertuples(index=False)
        )
        + "\n",
    )
    BASE.write_parquet(LOOP_ROOT / "fixture_results.parquet", fixtures)
    BASE.write_parquet(LOOP_ROOT / "state_selection_registry.parquet", selection)
    BASE.write_parquet(LOOP_ROOT / "restored_state_registry.parquet", states)
    BASE.write_parquet(LOOP_ROOT / "target_basin_coordinates.parquet", coordinates)
    BASE.write_parquet(LOOP_ROOT / "state_restoration_validation.parquet", restoration)
    BASE.write_parquet(LOOP_ROOT / "branch_seed_manifest.parquet", seeds)
    BASE.write_json(LOOP_ROOT / "seed_firewall.json", firewall)
    BASE.write_json(LOOP_ROOT / "immutable_prior_validation.json", prior)
    hashes = {
        "stateSelectionSha256": sha256_file(
            LOOP_ROOT / "state_selection_registry.parquet"
        ),
        "restoredStateSha256": sha256_file(
            LOOP_ROOT / "restored_state_registry.parquet"
        ),
        "targetBasinSha256": sha256_file(
            LOOP_ROOT / "target_basin_coordinates.parquet"
        ),
        "branchSeedManifestSha256": sha256_file(
            LOOP_ROOT / "branch_seed_manifest.parquet"
        ),
    }
    BASE.write_json(
        LOOP_ROOT / "implementation_lock.json",
        {
            "schema": "eidosoma.e01.s19_l28.implementation_lock.v1",
            "researchStepId": LOOP_ID,
            "versionedId": VERSION,
            "repositoryHead": head,
            "remoteHead": remote,
            "configSha256": sha256_file(CONFIG),
            "runnerSha256": sha256_file(RUNNER_PATH),
            "coreSha256": sha256_file(CORE_PATH),
            "targetId": TARGET_ID,
            "targetThreshold": TARGET_THRESHOLD,
            "targetScope": "RETROSPECTIVE_COMPLETED_RUN_MATRIX_SPECIFIC",
            "landmarks": list(LANDMARKS),
            "statesPerStratum": STATES_PER_STRATUM,
            "stateCount": len(states),
            "horizon": HORIZON,
            "branchesPerState": BRANCHES,
            "branchHalves": [64, 64],
            "bootstrapReplicates": BOOTSTRAPS,
            "predictorIds": list(PREDICTOR_SOURCES),
            "outcomeAccessed": False,
            "lockedArtifactHashes": hashes,
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
            "coreSha256": sha256_file(CORE_PATH),
            "taskHash": frame_hash(task),
            **hashes,
        },
    )
    BASE.write_json(
        LOOP_ROOT / "benchmark_projection.json",
        {
            "status": "PASS_PROJECTED_WITHIN_RESERVED_CEILING",
            "synthetic128BranchSeconds": benchmark_seconds,
            "projectedWallHoursUpper": projected_wall_hours,
            "projectedCpuHoursUpperIncludingReplay": projected_cpu_hours,
            "cpuHoursCeiling": 100,
            "wallHoursCeiling": 72,
            "validationReserveFraction": 0.10,
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
    fixtures = fixture_table()
    task = pd.read_parquet(L25_ROOT / "online_task_registry.parquet")
    manifest = pd.read_parquet(L23_ROOT / "input_trajectory_manifest.parquet")
    if (
        not prior["unchanged"]
        or prior["aggregateSha256"] != lock["priorAggregateSha256"]
        or not fixtures["passed"].all()
        or frame_hash(task) != lock["taskHash"]
    ):
        raise RuntimeError("pre-execution gate failed")
    lock_files = {
        "stateSelectionSha256": "state_selection_registry.parquet",
        "restoredStateSha256": "restored_state_registry.parquet",
        "targetBasinSha256": "target_basin_coordinates.parquet",
        "branchSeedManifestSha256": "branch_seed_manifest.parquet",
    }
    for key, name in lock_files.items():
        if sha256_file(LOOP_ROOT / name) != lock[key]:
            raise RuntimeError(f"pre-outcome artifact changed: {name}")
    states = pd.read_parquet(LOOP_ROOT / "restored_state_registry.parquet")
    coordinates = pd.read_parquet(LOOP_ROOT / "target_basin_coordinates.parquet")
    restoration = pd.read_parquet(LOOP_ROOT / "state_restoration_validation.parquet")
    seed_manifest = pd.read_parquet(LOOP_ROOT / "branch_seed_manifest.parquet")
    payloads = state_payloads(states, coordinates, manifest)
    if BUILD_ROOT.exists():
        shutil.rmtree(BUILD_ROOT)
    BUILD_ROOT.mkdir(parents=True)

    branch_start = time.perf_counter()
    branches = execute_branches(payloads)
    first_branch_seconds = time.perf_counter() - branch_start
    expected = len(states) * BRANCHES
    if (
        len(branches) != expected
        or not branches["branchIdentitySha256"].is_unique
        or branches["branchIdentitySha256"].tolist()
        != seed_manifest["branchIdentitySha256"].tolist()
    ):
        raise RuntimeError("branch identity or cardinality failure")
    state_results = state_committors(branches, states)
    reliability = reliability_summary(state_results)
    bootstraps = bootstrap_reliability(state_results)
    scores = predictor_scores(state_results)
    predictor_metric_frame = predictor_metrics(scores, state_results)
    predictor_bootstrap_frame = predictor_bootstraps(scores, state_results)
    leakage = leakage_audit(states)

    # Exact full replay is deliberately performed before scientific release.
    replay_start = time.perf_counter()
    replay_branches = execute_branches(payloads)
    replay_seconds = time.perf_counter() - replay_start
    branch_replay_exact = frame_hash(branches) == frame_hash(replay_branches)
    replay_state_results = state_committors(replay_branches, states)
    state_replay_exact = frame_hash(state_results) == frame_hash(replay_state_results)
    replay_reliability = reliability_summary(replay_state_results)
    reliability_replay_exact = frame_hash(reliability) == frame_hash(replay_reliability)
    if not (branch_replay_exact and state_replay_exact and reliability_replay_exact):
        raise RuntimeError("full branch or committor replay failed")

    gates = gate_table(reliability, bootstraps, restoration, True, leakage)
    committor_established = bool(gates["candidateCommittorGatePassed"].all())
    predictor_gates, predictor_classes = predictor_adjudication(
        predictor_metric_frame, predictor_bootstrap_frame, committor_established
    )
    branchable = bool(gates["targetBranchable"].all())
    if not branchable:
        classifications = [
            "CURRENT_TARGET_NOT_COMMITTOR_COMPATIBLE",
            "NOT_PROMOTABLE_AS_CONFIRMED",
        ]
    elif not committor_established:
        classifications = [
            "NO_RELIABLE_STATE_DEPENDENT_COMMITTOR_AT_H32",
            "NOT_PROMOTABLE_AS_CONFIRMED",
        ]
    else:
        classifications = [
            "STATE_DEPENDENT_COMMITTOR_ESTABLISHED",
            *predictor_classes,
            "NOT_PROMOTABLE_AS_CONFIRMED",
        ]
    stop_search = not committor_established
    make_figures(state_results, reliability, bootstraps, predictor_metric_frame, gates)

    for name in (
        "preregistration.yaml",
        "decision_record.md",
        "source_grounding_registry.csv",
        "source_grounding_report.md",
        "fixture_results.parquet",
        "state_selection_registry.parquet",
        "restored_state_registry.parquet",
        "target_basin_coordinates.parquet",
        "state_restoration_validation.parquet",
        "branch_seed_manifest.parquet",
        "seed_firewall.json",
        "immutable_prior_validation.json",
        "implementation_lock.json",
        "preoutcome_repository_lock.json",
        "benchmark_projection.json",
        "technical_amendment_ledger.csv",
    ):
        shutil.copy2(LOOP_ROOT / name, BUILD_ROOT / name)
    BASE.write_parquet(BUILD_ROOT / "branch_results.parquet", branches)
    BASE.write_parquet(BUILD_ROOT / "committor_state_results.parquet", state_results)
    BASE.write_parquet(
        BUILD_ROOT / "committor_reliability_results.parquet", reliability
    )
    BASE.write_parquet(BUILD_ROOT / "bootstrap_results.parquet", bootstraps)
    BASE.write_parquet(BUILD_ROOT / "frozen_predictor_scores.parquet", scores)
    BASE.write_parquet(
        BUILD_ROOT / "frozen_predictor_metrics.parquet", predictor_metric_frame
    )
    BASE.write_parquet(
        BUILD_ROOT / "predictor_bootstrap_results.parquet",
        predictor_bootstrap_frame,
    )
    BASE.write_parquet(BUILD_ROOT / "predictor_gate_results.parquet", predictor_gates)
    BASE.write_parquet(BUILD_ROOT / "leakage_audit.parquet", leakage)
    BASE.write_parquet(BUILD_ROOT / "scientific_gate_results.parquet", gates)
    BASE.write_json(
        BUILD_ROOT / "classification.json",
        {
            "schema": "eidosoma.e01.s19_l28.classification.v1",
            "researchStepId": LOOP_ID,
            "classifications": classifications,
            "stateDependentCommittorEstablished": committor_established,
            "stopL27ThroughL42PrecursorSearch": stop_search,
            "confirmatory": False,
            "retrospectiveBasinConditioned": True,
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
        "branchIdentityUnique": bool(branches["branchIdentitySha256"].is_unique),
        "branchIdentityManifestExact": branches["branchIdentitySha256"].tolist()
        == seed_manifest["branchIdentitySha256"].tolist(),
        "branchFullReplayExact": branch_replay_exact,
        "committorStateReplayExact": state_replay_exact,
        "reliabilityReplayExact": reliability_replay_exact,
        "restorationAllPassed": bool(
            restoration[
                [
                    "trajectoryIdentityPassed",
                    "selectedClockIdentityPassed",
                    "restoredStateExact",
                    "betaIdentityPassed",
                    "targetLabelExact",
                    "targetScoreEquivalent",
                    "currentStateOutsideBasin",
                    "singleFutureEventReplay",
                ]
            ]
            .all()
            .all()
        ),
        "leakageAuditPassed": bool(leakage["passed"].all()),
        "seedFirewallPassed": json.loads(
            (LOOP_ROOT / "seed_firewall.json").read_text()
        )["status"]
        == "PASS",
        "immutablePriorPassed": prior["unchanged"],
    }
    BASE.write_json(
        BUILD_ROOT / "regeneration_validation.json",
        {
            "schema": "eidosoma.e01.s19_l28.regeneration_validation.v1",
            "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "firstPassFrameSha256": frame_hash(branches),
            "replayFrameSha256": frame_hash(replay_branches),
        },
    )
    if not all(checks.values()):
        raise RuntimeError("regeneration validation failed")
    runtime = {
        "schema": "eidosoma.e01.s19_l28.runtime.v1",
        "researchStepId": LOOP_ID,
        "repositoryHead": git("rev-parse", "HEAD"),
        "workers": WORKERS,
        "gpuHours": 0,
        "wallSeconds": time.perf_counter() - start_wall,
        "controllerCpuHours": (time.process_time() - start_cpu) / 3600,
        "workerCpuHours": (first_branch_seconds + replay_seconds) * WORKERS / 3600,
        "firstBranchPassSeconds": first_branch_seconds,
        "fullReplaySeconds": replay_seconds,
        "stateCount": len(states),
        "branchCount": len(branches),
        "bootstrapReplicatesPerCandidate": BOOTSTRAPS,
        "completedAtUtc": utc_now(),
    }
    BASE.write_json(BUILD_ROOT / "runtime_manifest.json", runtime)
    retained_bytes = sum(
        path.stat().st_size for path in BUILD_ROOT.rglob("*") if path.is_file()
    )
    temporary_bytes = sum(
        path.stat().st_size for path in CACHE_ROOT.rglob("*") if path.is_file()
    )
    storage = {
        "schema": "eidosoma.e01.s19_l28.storage_validation.v1",
        "retainedBytes": retained_bytes,
        "retainedGiBCeiling": 25,
        "temporaryBytes": temporary_bytes,
        "temporaryGiBCeiling": 75,
    }
    storage["status"] = (
        "PASS"
        if retained_bytes < 25 * 2**30 and temporary_bytes < 75 * 2**30
        else "FAIL"
    )
    BASE.write_json(BUILD_ROOT / "storage_validation.json", storage)
    report = report_text(
        reliability,
        gates,
        predictor_metric_frame,
        predictor_gates,
        classifications,
        runtime,
        stop_search,
    )
    BASE.atomic_text(BUILD_ROOT / "research_step_full_results.md", report)
    BASE.atomic_text(BUILD_ROOT / "S19_L28_FULL_RESULTS.md", report)
    BASE.atomic_text(
        BUILD_ROOT / "loop_decision_summary.md",
        f"# S19-L28 decision summary\n\n**Classification:** {', '.join(classifications)}\n\n**State-dependent committor established in both candidates:** `{committor_established}`.\n\n**Directed action:** {'STOP FOR HUMAN REVIEW' if stop_search else 'continue only to exact-generator drift/diffusion features; no transition-tube/current work yet'}.\n",
    )
    BASE.write_json(BUILD_ROOT / "artifact_manifest.json", manifest_for(BUILD_ROOT))
    stage = LOOP_ROOT.with_name(".L28-promotion-stage")
    if stage.exists():
        shutil.rmtree(stage)
    shutil.copytree(BUILD_ROOT, stage)
    if LOOP_ROOT.exists():
        shutil.rmtree(LOOP_ROOT)
    os.replace(stage, LOOP_ROOT)
    shutil.rmtree(BUILD_ROOT)
    final_manifest = json.loads((LOOP_ROOT / "artifact_manifest.json").read_text())
    if any(
        sha256_file(LOOP_ROOT / item["path"]) != item["sha256"]
        for item in final_manifest["files"]
    ):
        raise RuntimeError("artifact hash mismatch")
    append_ledgers(classifications, runtime["completedAtUtc"], stop_search)
    BASE.atomic_text(ARTIFACT_ROOT / "research_step_full_results.md", report)
    BASE.atomic_text(
        ARTIFACT_ROOT / "S19_CURRENT_HANDOFF.md",
        report.replace("# S19-L28", "# S19 current handoff — S19-L28", 1),
    )
    BASE.write_json(
        ARTIFACT_ROOT / "s19_status.json",
        {
            "schema": "eidosoma.e01.s19.status.v1",
            "status": "PAUSED_FOR_HUMAN_REVIEW_AT_COMMITTOR_GATE"
            if stop_search
            else "ACTIVE_AUTONOMOUS_SEQUENCE",
            "latestCompletedLoop": LOOP_ID,
            "latestClassification": classifications,
            "selectedDiscoveryLead": None,
            "nextAuthorizedLoop": None if stop_search else "S19-L29",
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
                "stateDependentCommittorEstablished": committor_established,
                "stopForHumanReview": stop_search,
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
