"""Execute S19-L36 independent-lineage target-basin transfer audit."""

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
    os.environ[variable] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import rankdata, spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from e01_latent_timebase.core import initialize_distinct_state, simulate_trajectory
from e01_onset_discovery.basin_transfer import (
    centroid_similarity,
    cosine_scores,
    numerical_equivalence,
    summarize_scores,
)
from e01_onset_discovery.branch_trace import simulate_branch_trace
from e01_onset_discovery.empirical_committor import (
    RestoredState,
    corrected_between_state_variance,
)


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


L35 = _load_module(
    "e01_s19_l36_l35",
    REPO_ROOT / "scripts/e01/run_s19_l35_short_branch_mechanism_attribution.py",
)
L34 = L35.L34
L33 = L35.L33
L31 = L35.L31
L30 = L35.L30
L29 = L35.L29
L28 = L35.L28
BASE = L35.BASE

LOOP_ID = "S19-L36"
VERSION = "E01-S19-L36-INDEPENDENT-LINEAGE-BASIN-TRANSFER-AUDIT-v1.0.0"
CANDIDATES = L28.CANDIDATES
COHORTS = L35.COHORTS
EVALUATION_COHORTS = L35.EVALUATION_COHORTS
REFERENCES = ("REFERENCE_A", "REFERENCE_B")
TARGETS = ("ORIGINAL", *REFERENCES)
FAMILIES = ("H32", "H8")
HORIZONS = {"H32": 32, "H8": 8}
BRANCH_COUNTS = {"H32": 128, "H8": 64}
HALVES = {"H32": 64, "H8": 32}
BOOTSTRAPS = 4096
ROOT_HEX = "3f49c76b5e3b44d027d6d7f8b449531df70470884828c2b8466e4854e7a86426"
PHASE = "s19_l36_independent_lineage_basin"
WORKERS = min(8, os.cpu_count() or 1)
REPLAY_ABSOLUTE_TOLERANCE = 1e-12
REPLAY_RELATIVE_TOLERANCE = 1e-12
REPLAY_MAXIMUM_ULP_ERROR = 16

ARTIFACT_ROOT = Path("/artifacts/research_steps/S19")
LOOP_ROOT = ARTIFACT_ROOT / "loops/L36"
L35_ROOT = ARTIFACT_ROOT / "loops/L35"
L31_ROOT = ARTIFACT_ROOT / "loops/L31"
L30_ROOT = ARTIFACT_ROOT / "loops/L30"
L28_ROOT = ARTIFACT_ROOT / "loops/L28"
L23_ROOT = ARTIFACT_ROOT / "loops/L23"
CACHE_ROOT = Path("/cache/e01_s19_l36")
REFERENCE_CACHE = CACHE_ROOT / "reference_lineages"
REGEN_CACHE = CACHE_ROOT / "reference_lineages_regeneration"
BUILD_ROOT = CACHE_ROOT / "build"
CONFIG = REPO_ROOT / "configs/e01/s19_l36_independent_lineage_basin_transfer.yaml"
RUNNER_PATH = Path(__file__)
CORE_PATH = REPO_ROOT / "src/e01_onset_discovery/basin_transfer.py"


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
    return hashlib.sha256(
        frame.reset_index(drop=True)
        .to_json(orient="table", index=False, double_precision=15)
        .encode()
    ).hexdigest()


def derived_seed(*parts: object) -> int:
    payload = "\x1f".join([VERSION, ROOT_HEX, *map(str, parts)])
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:16], "big")


def validate_immutable_prior() -> dict[str, Any]:
    prior = json.loads((L35_ROOT / "immutable_prior_validation.json").read_text())
    rows = list(prior["files"])
    manifest = json.loads((L35_ROOT / "artifact_manifest.json").read_text())
    rows.extend(
        {
            "path": str(L35_ROOT / item["path"]),
            "root": str(L35_ROOT),
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
    return {
        "schema": "eidosoma.e01.s19_l36.immutable_prior_validation.v1",
        "status": "PASS" if not failures else "FAIL",
        "unchanged": not failures,
        "fileCount": len(rows),
        "aggregateSha256": hashlib.sha256(
            "\n".join(f"{row['path']}\t{row['sha256']}" for row in rows).encode()
        ).hexdigest(),
        "l35ArtifactFileCount": manifest["fileCount"],
        "failures": failures,
        "files": rows,
    }


def reference_unit_registry(
    responses: pd.DataFrame, manifest: pd.DataFrame
) -> pd.DataFrame:
    manifest_index = manifest.set_index(["candidateId", "matrixIndex"])
    rows = []
    for source in responses.itertuples(index=False):
        item = manifest_index.loc[(source.candidateId, int(source.matrixIndex))]
        # Reconstruct the frozen L23 initial state from its registered seed.
        # The first selected-clock observation is not necessarily the initial
        # state, so it is not an admissible substitute for this identity gate.
        initial = initialize_distinct_state(
            L28.derive_seed(
                L28.L23_ROOT_HEX,
                L28.L23_PHASE,
                "initial_state",
                int(source.matrixIndex),
            )
        )
        beta = L28.generate_beta(
            L28.derive_seed(
                L28.L23_ROOT_HEX,
                L28.L23_PHASE,
                "catalytic_matrix",
                int(source.matrixIndex),
            )
        )
        rows.append(
            {
                "stateId": source.stateId,
                "evaluationCohort": source.evaluationCohort,
                "candidateId": source.candidateId,
                "matrixIndex": int(source.matrixIndex),
                "landmark": int(source.landmark),
                "initialState": initial.tolist(),
                "initialStateSha256": L28.simulator_array_sha256(initial),
                "expectedInitialStateSha256": item.initialStateSha256,
                "betaSha256": L28.simulator_array_sha256(beta),
                "expectedBetaSha256": item.betaSha256,
                "sourceTrajectorySha256": item.trajectorySha256,
                "sourceCacheSha256": item.cacheSha256,
                "simulatorDefinition": source.simulatorDefinition,
                "initialStateExact": L28.simulator_array_sha256(initial)
                == item.initialStateSha256,
                "betaExact": L28.simulator_array_sha256(beta) == item.betaSha256,
                "oneStatePerMatrixCandidate": True,
            }
        )
    result = pd.DataFrame(rows).sort_values(
        ["evaluationCohort", "candidateId", "landmark", "matrixIndex"]
    ).reset_index(drop=True)
    if (
        len(result) != 280
        or result.duplicated(["candidateId", "matrixIndex"]).any()
        or not result[["initialStateExact", "betaExact"]].all().all()
    ):
        raise RuntimeError("L36 reference input registry failure")
    return result


def reference_seed_manifest(units: pd.DataFrame) -> pd.DataFrame:
    rows = []
    purposes = (
        "catalytic_matrix",
        "initial_state",
        "poisson_update",
        "overshoot_trim",
        "fission",
        "daughter_selection",
    )
    for unit in units.itertuples(index=False):
        for reference in REFERENCES:
            for purpose in purposes:
                identity = L28.derive_seed(
                    ROOT_HEX,
                    PHASE,
                    purpose,
                    int(unit.matrixIndex),
                    f"{unit.candidateId}__{reference}",
                ) if purpose not in ("catalytic_matrix", "initial_state") else L28.derive_seed(
                    ROOT_HEX, PHASE, purpose, int(unit.matrixIndex)
                )
                rows.append(
                    {
                        "stateId": unit.stateId,
                        "evaluationCohort": unit.evaluationCohort,
                        "candidateId": unit.candidateId,
                        "matrixIndex": int(unit.matrixIndex),
                        "referenceId": reference,
                        "purpose": purpose,
                        "rootHex": ROOT_HEX,
                        "phase": PHASE,
                        "streamIdentity": f"{unit.candidateId}__{reference}",
                        "derivedSeed": str(identity.derived_seed),
                        "seedMaterialSha256": identity.seed_material_sha256,
                        "inputOverriddenWithFrozenValue": purpose
                        in ("catalytic_matrix", "initial_state"),
                    }
                )
    result = pd.DataFrame(rows).sort_values(
        ["candidateId", "matrixIndex", "referenceId", "purpose"]
    ).reset_index(drop=True)
    # beta/initial seed identities are intentionally shared by A/B for a matrix;
    # event/fission streams must remain unique.
    stochastic = result[~result["inputOverriddenWithFrozenValue"]]
    if (
        len(result) != 280 * 2 * 6
        or stochastic["seedMaterialSha256"].duplicated().any()
    ):
        raise RuntimeError("L36 reference seed manifest failure")
    return result


def seed_firewall(seeds: pd.DataFrame) -> dict[str, Any]:
    stochastic = set(
        seeds.loc[~seeds["inputOverriddenWithFrozenValue"], "seedMaterialSha256"]
    )
    prior: set[str] = set()
    for path in ARTIFACT_ROOT.glob("loops/L*/**/*seed*manifest*.parquet"):
        if "/L36/" in str(path):
            continue
        try:
            frame = pd.read_parquet(path)
        except (OSError, ValueError, TypeError):
            continue
        for column in frame.columns:
            if "seedmaterialsha256" in column.lower():
                prior.update(frame[column].dropna().astype(str))
    overlaps = sorted(stochastic & prior)
    return {
        "schema": "eidosoma.e01.s19_l36.seed_firewall.v1",
        "status": "PASS" if not overlaps else "FAIL",
        "newReferenceTrajectories": 560,
        "newMatrices": 0,
        "newInitialStates": 0,
        "newStochasticSeedMaterials": len(stochastic),
        "overlapCount": len(overlaps),
        "overlaps": overlaps,
    }


def _reference_cache_path(root: Path, state_id: str, reference: str) -> Path:
    return root / f"{state_id}__{reference}.pkl"


def _reference_worker(task: tuple[dict[str, Any], str, str]) -> list[dict[str, Any]]:
    unit, root_string, reference = task
    root = Path(root_string)
    matrix_index = int(unit["matrixIndex"])
    beta = L28.generate_beta(
        L28.derive_seed(
            L28.L23_ROOT_HEX, L28.L23_PHASE, "catalytic_matrix", matrix_index
        )
    )
    initial = np.asarray(unit["initialState"], dtype=np.int64)
    started = time.perf_counter()
    trajectory, _ = simulate_trajectory(
        phase=PHASE,
        root_hex=ROOT_HEX,
        matrix_index=matrix_index,
        definition=L28.definition(unit["candidateId"]),
        stream_identity=f"{unit['candidateId']}__{reference}",
        beta=beta,
        initial_state=initial,
    )
    path = _reference_cache_path(root, unit["stateId"], reference)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(trajectory, handle, protocol=5)
    post = tuple(
        row for row in trajectory.observations if row.observation_kind == "post_fission"
    )
    status = "ELIGIBLE" if trajectory.completed_fissions == 100 and len(post) == 100 else "INCOMPLETE_REFERENCE_LINEAGE"
    centroid = None
    component: list[int] = []
    if status == "ELIGIBLE":
        values = L28.states_from_observations(post)
        centroid_value, members = L28.dominant_component_centroid(values)
        centroid = centroid_value.tolist()
        component = list(map(int, members))
    return [
        {
            "stateId": unit["stateId"],
            "evaluationCohort": unit["evaluationCohort"],
            "candidateId": unit["candidateId"],
            "matrixIndex": matrix_index,
            "landmark": int(unit["landmark"]),
            "referenceId": reference,
            "status": status,
            "terminalStatus": trajectory.terminal_status,
            "completedFissions": int(trajectory.completed_fissions),
            "selectedClockLength": len(
                L28.selected_clock_observations(trajectory, L28.CLOCK_ID)
            ),
            "trajectoryId": trajectory.trajectory_id,
            "trajectorySha256": trajectory.trajectory_sha256,
            "betaSha256": trajectory.beta_sha256,
            "initialStateSha256": trajectory.initial_state_sha256,
            "cachePath": str(path),
            "cacheSha256": sha256_file(path),
            "componentSize": len(component),
            "componentMemberIndices": json.dumps(component),
            "centroid": centroid,
            "centroidSha256": L28.array_sha256(np.asarray(centroid, dtype=np.float64))
            if centroid is not None
            else None,
            "wallSeconds": time.perf_counter() - started,
            "replacementAttempted": False,
        }
    ]


def generate_references(units: pd.DataFrame, root: Path) -> pd.DataFrame:
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    tasks = [
        (unit._asdict(), str(root), reference)
        for unit in units.itertuples(index=False)
        for reference in REFERENCES
    ]
    rows = []
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(_reference_worker, task): task for task in tasks}
        for future in as_completed(futures):
            rows.extend(future.result())
    result = pd.DataFrame(rows).sort_values(
        ["evaluationCohort", "candidateId", "landmark", "matrixIndex", "referenceId"]
    ).reset_index(drop=True)
    if len(result) != 560 or result.duplicated(["stateId", "referenceId"]).any():
        raise RuntimeError("L36 reference trajectory scope failure")
    return result


def target_registries(
    responses: pd.DataFrame,
    original_coordinates: pd.DataFrame,
    references: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    original_map = {
        state_id: group.sort_values("coordinate")["centroidValue"].to_numpy(
            dtype=np.float64
        )
        for state_id, group in original_coordinates.groupby("stateId", sort=False)
    }
    reference_map = {
        (row.stateId, row.referenceId): np.asarray(row.centroid, dtype=np.float64)
        for row in references[references["status"].eq("ELIGIBLE")].itertuples(
            index=False
        )
    }
    coordinate_rows = []
    summary_rows = []
    comparison_rows = []
    response_index = responses.set_index("stateId")
    reference_index = references.set_index(["stateId", "referenceId"])
    for state_id, source in response_index.iterrows():
        current = None
        for target_id in TARGETS:
            if target_id == "ORIGINAL":
                centroid = original_map[state_id]
                status = "ELIGIBLE"
                component_size = int(source.targetComponentSize)
                trajectory_sha = source.trajectoryId
            else:
                ref = reference_index.loc[(state_id, target_id)]
                status = ref.status
                component_size = int(ref.componentSize)
                trajectory_sha = ref.trajectorySha256
                centroid = reference_map.get((state_id, target_id))
            if current is None:
                # Exact state is loaded later into worker payloads; source score is frozen.
                current_score = float(source.targetCurrentScore) if target_id == "ORIGINAL" else float("nan")
            else:
                current_score = float(cosine_scores(current[None, :], centroid)[0, 0])
            summary_rows.append(
                {
                    "stateId": state_id,
                    "evaluationCohort": source.evaluationCohort,
                    "candidateId": source.candidateId,
                    "matrixIndex": int(source.matrixIndex),
                    "landmark": int(source.landmark),
                    "targetId": target_id,
                    "status": status,
                    "componentSize": component_size,
                    "targetSourceTrajectory": trajectory_sha,
                    "centroidSha256": L28.array_sha256(centroid)
                    if centroid is not None
                    else None,
                    "currentScoreDeferred": target_id != "ORIGINAL",
                    "currentScore": current_score,
                }
            )
            if centroid is not None:
                for coordinate, value in enumerate(centroid):
                    coordinate_rows.append(
                        {
                            "stateId": state_id,
                            "evaluationCohort": source.evaluationCohort,
                            "candidateId": source.candidateId,
                            "matrixIndex": int(source.matrixIndex),
                            "landmark": int(source.landmark),
                            "targetId": target_id,
                            "coordinate": coordinate,
                            "centroidValue": float(value),
                        }
                    )
        centroids = {
            target: original_map[state_id]
            if target == "ORIGINAL"
            else reference_map.get((state_id, target))
            for target in TARGETS
        }
        for left, right in (
            ("REFERENCE_A", "REFERENCE_B"),
            ("ORIGINAL", "REFERENCE_A"),
            ("ORIGINAL", "REFERENCE_B"),
        ):
            defined = centroids[left] is not None and centroids[right] is not None
            value = (
                centroid_similarity(centroids[left], centroids[right])
                if defined
                else float("nan")
            )
            comparison_rows.append(
                {
                    "stateId": state_id,
                    "evaluationCohort": source.evaluationCohort,
                    "candidateId": source.candidateId,
                    "matrixIndex": int(source.matrixIndex),
                    "landmark": int(source.landmark),
                    "leftTarget": left,
                    "rightTarget": right,
                    "defined": defined,
                    "centroidH": value,
                    "strictH090Agreement": bool(defined and value > 0.9),
                }
            )
    summary = pd.DataFrame(summary_rows).sort_values(
        ["evaluationCohort", "candidateId", "landmark", "matrixIndex", "targetId"]
    ).reset_index(drop=True)
    coordinates = pd.DataFrame(coordinate_rows).sort_values(
        [
            "evaluationCohort",
            "candidateId",
            "landmark",
            "matrixIndex",
            "targetId",
            "coordinate",
        ]
    ).reset_index(drop=True)
    comparisons = pd.DataFrame(comparison_rows).sort_values(
        [
            "evaluationCohort",
            "candidateId",
            "landmark",
            "matrixIndex",
            "leftTarget",
            "rightTarget",
        ]
    ).reset_index(drop=True)
    if len(summary) != 840 or len(comparisons) != 840:
        raise RuntimeError("L36 target registry scope failure")
    return summary, coordinates, comparisons


def add_current_scores(
    summary: pd.DataFrame,
    coordinates: pd.DataFrame,
    payload_rows: list[dict[str, Any]],
) -> pd.DataFrame:
    payload_index = {row["stateId"]: row for row in payload_rows}
    coordinate_map = {
        (state_id, target): group.sort_values("coordinate")[
            "centroidValue"
        ].to_numpy(dtype=np.float64)
        for (state_id, target), group in coordinates.groupby(
            ["stateId", "targetId"], sort=False
        )
    }
    result = summary.copy()
    scores = []
    for row in result.itertuples(index=False):
        target = coordinate_map.get((row.stateId, row.targetId))
        if target is None:
            scores.append(float("nan"))
            continue
        state = np.asarray(payload_index[row.stateId]["state"], dtype=np.int64)
        scores.append(float(cosine_scores(state[None, :], target)[0, 0]))
    result["currentScore"] = scores
    result["currentScoreDeferred"] = False
    result["currentInsideTarget"] = result["currentScore"] >= 0.9
    result["committorEligible"] = result["status"].eq("ELIGIBLE") & ~result[
        "currentInsideTarget"
    ]
    original = result[result["targetId"].eq("ORIGINAL")]
    frozen = pd.read_parquet(L35_ROOT / "response_registry.parquet").set_index("stateId")
    errors = [
        abs(row.currentScore - float(frozen.loc[row.stateId, "targetCurrentScore"]))
        for row in original.itertuples(index=False)
    ]
    if max(errors) > 1e-12 or original["currentInsideTarget"].any():
        raise RuntimeError("L36 original current-target replay failure")
    return result


def _branch_identities(
    payload: dict[str, Any], family: str, branch: int
) -> dict[str, Any]:
    candidate = payload["candidateId"]
    matrix = int(payload["matrixIndex"])
    landmark = int(payload["landmark"])
    if payload["evaluationCohort"] == "L31_CONFIRMATION":
        return L31.stream_identities(family, candidate, matrix, landmark, branch)
    if family == "H32":
        return L28.branch_seed_identities(candidate, matrix, landmark, branch)
    return L30.branch_seeds(candidate, matrix, landmark, branch)


def _branch_rngs(
    payload: dict[str, Any], family: str, branch: int
) -> tuple[Any, Any, Any, Any]:
    identities = _branch_identities(payload, family, branch)
    if payload["evaluationCohort"] == "L31_CONFIRMATION":
        keys = ("event", "trim", "fission", "daughter")
    elif family == "H32":
        keys = (
            "committor_poisson_update",
            "committor_overshoot_trim",
            "committor_fission",
            "committor_daughter_selection",
        )
    else:
        keys = (
            "propagator_event",
            "propagator_trim",
            "propagator_fission",
            "propagator_daughter",
        )
    return tuple(L28.generator(identities[key]) for key in keys)


def _branch_worker(payload: dict[str, Any]) -> list[dict[str, Any]]:
    matrix_index = int(payload["matrixIndex"])
    beta = L28.generate_beta(
        L28.derive_seed(
            L28.L23_ROOT_HEX, L28.L23_PHASE, "catalytic_matrix", matrix_index
        )
    )
    restored = RestoredState(
        tuple(payload["state"]),
        payload["currentObservationKind"],
        int(payload["currentCompletedFissions"]),
        int(payload["currentGrowthGeneration"]),
        int(payload["currentGenerationLocalStep"]),
        int(payload["currentBatchStep"]),
    )
    targets = {
        key: np.asarray(value, dtype=np.float64)
        for key, value in payload["targets"].items()
        if value is not None
    }
    if "ORIGINAL" not in targets:
        raise RuntimeError("original target missing")
    current_state = np.asarray(payload["state"], dtype=np.int64)
    current_scores = {
        target: float(cosine_scores(current_state[None, :], centroid)[0, 0])
        for target, centroid in targets.items()
    }
    rows = []
    for family in FAMILIES:
        for branch in range(BRANCH_COUNTS[family]):
            event_rng, trim_rng, fission_rng, daughter_rng = _branch_rngs(
                payload, family, branch
            )
            trace = simulate_branch_trace(
                restored=restored,
                beta=beta,
                definition=L28.definition(payload["candidateId"]),
                target_centroid=targets["ORIGINAL"],
                event_rng=event_rng,
                trim_rng=trim_rng,
                fission_rng=fission_rng,
                daughter_rng=daughter_rng,
                horizon=HORIZONS[family],
                threshold=0.9,
            )
            states = np.asarray([row.state for row in trace.observations], dtype=np.int64)
            target_names = list(targets)
            values = cosine_scores(states, np.stack([targets[name] for name in target_names]))
            for target_index, target in enumerate(target_names):
                summary = summarize_scores(values[:, target_index], threshold=0.9)
                rows.append(
                    {
                        "stateId": payload["stateId"],
                        "evaluationCohort": payload["evaluationCohort"],
                        "candidateId": payload["candidateId"],
                        "matrixIndex": matrix_index,
                        "landmark": int(payload["landmark"]),
                        "branchFamily": family,
                        "targetId": target,
                        "branchIndex": branch,
                        "branchHalf": "A"
                        if branch < HALVES[family]
                        else "B",
                        "currentTargetScore": current_scores[target],
                        "currentInsideTarget": current_scores[target] >= 0.9,
                        "enteredTarget": summary.entered,
                        "firstEntryOffsetOneBased": summary.first_entry_offset_one_based,
                        "minimumTargetScore": summary.minimum_score,
                        "maximumTargetScore": summary.maximum_score,
                        "finalTargetScore": summary.final_score,
                        "molecularUpdates": trace.compact.molecular_updates,
                        "fissions": trace.compact.fissions,
                        "selectedObservationsGenerated": trace.compact.selected_observations_generated,
                        "terminalStatus": trace.compact.terminal_status,
                        "finalStateSha256": trace.compact.final_state_sha256,
                        "originalPathSha256": trace.compact.path_sha256,
                        "dynamicPathIndependentOfTarget": True,
                    }
                )
    return rows


def branch_payloads(
    base_payloads: list[dict[str, Any]], coordinates: pd.DataFrame
) -> list[dict[str, Any]]:
    coordinate_map = {
        (state_id, target): group.sort_values("coordinate")[
            "centroidValue"
        ].tolist()
        for (state_id, target), group in coordinates.groupby(
            ["stateId", "targetId"], sort=False
        )
    }
    output = []
    for source in base_payloads:
        row = dict(source)
        row["targets"] = {
            target: coordinate_map.get((source["stateId"], target)) for target in TARGETS
        }
        output.append(row)
    return output


def rescore_branches(payload_rows: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = {
            executor.submit(_branch_worker, payload): payload["stateId"]
            for payload in payload_rows
        }
        for future in as_completed(futures):
            rows.extend(future.result())
    result = pd.DataFrame(rows).sort_values(
        [
            "evaluationCohort",
            "candidateId",
            "landmark",
            "matrixIndex",
            "branchFamily",
            "targetId",
            "branchIndex",
        ]
    ).reset_index(drop=True)
    expected = sum(
        sum(BRANCH_COUNTS.values())
        * sum(target is not None for target in payload["targets"].values())
        for payload in payload_rows
    )
    if len(result) != expected:
        raise RuntimeError("L36 branch-rescore cardinality failure")
    return result


def compact_replay_validation(branches: pd.DataFrame) -> pd.DataFrame:
    generated = branches[branches["targetId"].eq("ORIGINAL")].copy()
    sources = []
    l28_h32 = pd.read_parquet(L28_ROOT / "branch_results.parquet").rename(
        columns={
            "enteredBasinWithin32": "sourceEnteredTarget",
            "branchIdentitySha256": "sourceStreamIdentity",
        }
    )
    l28_h32["evaluationCohort"] = np.where(
        l28_h32["matrixRole"].eq("DEVELOPMENT"),
        "L28_DEVELOPMENT",
        "L28_VALIDATION",
    )
    l28_h32["branchFamily"] = "H32"
    sources.append(l28_h32)
    l30_h8 = pd.read_parquet(L30_ROOT / "short_branch_results.parquet")
    l30_h8 = l30_h8[l30_h8["referenceVariant"].eq("ORIGINAL")].rename(
        columns={
            "enteredBasinWithin8": "sourceEnteredTarget",
            "streamIdentitySha256": "sourceStreamIdentity",
        }
    )
    l30_h8["evaluationCohort"] = np.where(
        l30_h8["matrixRole"].eq("DEVELOPMENT"),
        "L28_DEVELOPMENT",
        "L28_VALIDATION",
    )
    l30_h8["branchFamily"] = "H8"
    sources.append(l30_h8)
    l31 = pd.read_parquet(L31_ROOT / "branch_results.parquet")
    l31 = l31[l31["referenceVariant"].eq("ORIGINAL")].rename(
        columns={
            "enteredBasin": "sourceEnteredTarget",
            "streamIdentitySha256": "sourceStreamIdentity",
        }
    )
    l31["evaluationCohort"] = "L31_CONFIRMATION"
    sources.append(l31)
    columns = [
        "stateId",
        "evaluationCohort",
        "branchFamily",
        "branchIndex",
        "sourceEnteredTarget",
        "firstEntryOffsetOneBased",
        "maximumTargetScore",
        "minimumTargetScore",
        "molecularUpdates",
        "fissions",
        "selectedObservationsGenerated",
        "terminalStatus",
        "pathSha256",
    ]
    source = pd.concat([frame[columns] for frame in sources], ignore_index=True)
    merged = generated.merge(
        source,
        on=["stateId", "evaluationCohort", "branchFamily", "branchIndex"],
        suffixes=("", "Source"),
        validate="one_to_one",
    )

    def equal(left: Any, right: Any) -> bool:
        if left is None or (isinstance(left, float) and np.isnan(left)):
            return right is None or (isinstance(right, float) and np.isnan(right))
        return left == right

    maximum_equivalence = numerical_equivalence(
        merged["maximumTargetScore"].to_numpy(dtype=np.float64),
        merged["maximumTargetScoreSource"].to_numpy(dtype=np.float64),
        absolute_tolerance=REPLAY_ABSOLUTE_TOLERANCE,
        relative_tolerance=REPLAY_RELATIVE_TOLERANCE,
        maximum_ulp_error=REPLAY_MAXIMUM_ULP_ERROR,
    )
    minimum_equivalence = numerical_equivalence(
        merged["minimumTargetScore"].to_numpy(dtype=np.float64),
        merged["minimumTargetScoreSource"].to_numpy(dtype=np.float64),
        absolute_tolerance=REPLAY_ABSOLUTE_TOLERANCE,
        relative_tolerance=REPLAY_RELATIVE_TOLERANCE,
        maximum_ulp_error=REPLAY_MAXIMUM_ULP_ERROR,
    )
    rows = []
    for row in merged.itertuples(index=False):
        maximum_pair = numerical_equivalence(
            np.asarray([row.maximumTargetScore], dtype=np.float64),
            np.asarray([row.maximumTargetScoreSource], dtype=np.float64),
            absolute_tolerance=REPLAY_ABSOLUTE_TOLERANCE,
            relative_tolerance=REPLAY_RELATIVE_TOLERANCE,
            maximum_ulp_error=REPLAY_MAXIMUM_ULP_ERROR,
        )
        minimum_pair = numerical_equivalence(
            np.asarray([row.minimumTargetScore], dtype=np.float64),
            np.asarray([row.minimumTargetScoreSource], dtype=np.float64),
            absolute_tolerance=REPLAY_ABSOLUTE_TOLERANCE,
            relative_tolerance=REPLAY_RELATIVE_TOLERANCE,
            maximum_ulp_error=REPLAY_MAXIMUM_ULP_ERROR,
        )
        checks = {
            "entryExact": bool(row.enteredTarget) == bool(row.sourceEnteredTarget),
            "firstEntryExact": equal(
                row.firstEntryOffsetOneBased, row.firstEntryOffsetOneBasedSource
            ),
            "maximumScoreNumericallyEquivalent": maximum_pair.passed,
            "minimumScoreNumericallyEquivalent": minimum_pair.passed,
            "molecularUpdatesExact": row.molecularUpdates
            == row.molecularUpdatesSource,
            "fissionsExact": row.fissions == row.fissionsSource,
            "selectedObservationsExact": row.selectedObservationsGenerated
            == row.selectedObservationsGeneratedSource,
            "terminalExact": row.terminalStatus == row.terminalStatusSource,
            "pathExact": row.originalPathSha256 == row.pathSha256,
        }
        rows.append(
            {
                "stateId": row.stateId,
                "evaluationCohort": row.evaluationCohort,
                "candidateId": row.candidateId,
                "matrixIndex": int(row.matrixIndex),
                "landmark": int(row.landmark),
                "branchFamily": row.branchFamily,
                "branchIndex": int(row.branchIndex),
                "maximumScoreBitExact": equal(
                    row.maximumTargetScore, row.maximumTargetScoreSource
                ),
                "minimumScoreBitExact": equal(
                    row.minimumTargetScore, row.minimumTargetScoreSource
                ),
                "maximumScoreAbsoluteError": maximum_pair.max_absolute_error,
                "maximumScoreRelativeError": maximum_pair.max_relative_error,
                "maximumScoreUlpError": maximum_pair.max_ulp_error,
                "minimumScoreAbsoluteError": minimum_pair.max_absolute_error,
                "minimumScoreRelativeError": minimum_pair.max_relative_error,
                "minimumScoreUlpError": minimum_pair.max_ulp_error,
                **checks,
                "allPassed": all(checks.values()),
            }
        )
    result = pd.DataFrame(rows)
    if (
        len(result) != 53_760
        or not maximum_equivalence.passed
        or not minimum_equivalence.passed
        or not result["allPassed"].all()
    ):
        raise RuntimeError("L36 original-target compact replay failed")
    return result


def state_committor_results(
    branches: pd.DataFrame, targets: pd.DataFrame
) -> pd.DataFrame:
    target_index = targets.set_index(["stateId", "targetId"])
    rows = []
    for keys, group in branches.groupby(
        ["stateId", "branchFamily", "targetId"], sort=True
    ):
        state_id, family, target = keys
        target_row = target_index.loc[(state_id, target)]
        expected = BRANCH_COUNTS[family]
        half = HALVES[family]
        if len(group) != expected:
            raise RuntimeError("L36 state branch cardinality mismatch")
        half_a = group[group["branchHalf"].eq("A")]
        half_b = group[group["branchHalf"].eq("B")]
        eligible = bool(target_row.committorEligible)
        successes = int(group["enteredTarget"].sum())
        rows.append(
            {
                "stateId": state_id,
                "evaluationCohort": target_row.evaluationCohort,
                "candidateId": target_row.candidateId,
                "matrixIndex": int(target_row.matrixIndex),
                "landmark": int(target_row.landmark),
                "branchFamily": family,
                "targetId": target,
                "targetStatus": target_row.status,
                "currentTargetScore": float(target_row.currentScore),
                "currentInsideTarget": bool(target_row.currentInsideTarget),
                "committorEligible": eligible,
                "branches": expected,
                "successes": successes,
                "q": successes / expected if eligible else float("nan"),
                "qHalfA": float(half_a["enteredTarget"].mean())
                if eligible
                else float("nan"),
                "qHalfB": float(half_b["enteredTarget"].mean())
                if eligible
                else float("nan"),
                "intermediateProbability": bool(
                    eligible and 0.1 < successes / expected < 0.9
                ),
                "completeHorizonBranches": int(
                    (group["selectedObservationsGenerated"] == HORIZONS[family]).sum()
                ),
                "branchHalfCardinalityPassed": len(half_a) == half
                and len(half_b) == half,
            }
        )
    result = pd.DataFrame(rows).sort_values(
        [
            "evaluationCohort",
            "candidateId",
            "landmark",
            "matrixIndex",
            "branchFamily",
            "targetId",
        ]
    ).reset_index(drop=True)
    if not result["branchHalfCardinalityPassed"].all():
        raise RuntimeError("L36 branch half failure")
    return result


def safe_spearman(left: Any, right: Any) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3 or np.unique(x[valid]).size < 2 or np.unique(y[valid]).size < 2:
        return float("nan")
    return float(spearmanr(x[valid], y[valid]).statistic)


def target_availability_results(
    targets: pd.DataFrame, comparisons: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    comparison = comparisons[
        comparisons["leftTarget"].eq("REFERENCE_A")
        & comparisons["rightTarget"].eq("REFERENCE_B")
    ]
    for (cohort, candidate), group in targets.groupby(
        ["evaluationCohort", "candidateId"], sort=True
    ):
        comp = comparison[
            comparison["evaluationCohort"].eq(cohort)
            & comparison["candidateId"].eq(candidate)
        ]
        row = {
            "evaluationCohort": cohort,
            "candidateId": candidate,
            "states": int(group["stateId"].nunique()),
            "referenceAEligibleFraction": float(
                group[group["targetId"].eq("REFERENCE_A")]["status"].eq("ELIGIBLE").mean()
            ),
            "referenceBEligibleFraction": float(
                group[group["targetId"].eq("REFERENCE_B")]["status"].eq("ELIGIBLE").mean()
            ),
            "referenceAAtRiskFraction": float(
                group[group["targetId"].eq("REFERENCE_A")]["committorEligible"].mean()
            ),
            "referenceBAtRiskFraction": float(
                group[group["targetId"].eq("REFERENCE_B")]["committorEligible"].mean()
            ),
            "referenceCentroidAgreementFraction": float(
                comp["strictH090Agreement"].mean()
            ),
            "referenceCentroidHMean": float(comp["centroidH"].mean()),
            "referenceCentroidHMedian": float(comp["centroidH"].median()),
        }
        row["availabilityGatePassed"] = bool(
            row["referenceAEligibleFraction"] >= 0.9
            and row["referenceBEligibleFraction"] >= 0.9
            and row["referenceAAtRiskFraction"] >= 0.8
            and row["referenceBAtRiskFraction"] >= 0.8
        )
        row["centroidAgreementGatePassed"] = bool(
            row["referenceCentroidAgreementFraction"] >= 0.8
        )
        rows.append(row)
    return pd.DataFrame(rows)


def reliability_results(states: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in states.groupby(
        ["evaluationCohort", "candidateId", "branchFamily", "targetId"], sort=True
    ):
        cohort, candidate, family, target = keys
        eligible = group[group["committorEligible"]].copy()
        variance = corrected_between_state_variance(
            eligible["q"].to_numpy(dtype=np.float64), BRANCH_COUNTS[family]
        ) if len(eligible) > 1 else {
            "observedBetweenStateVariance": float("nan"),
            "expectedBinomialNoiseVariance": float("nan"),
            "correctedBetweenStateVariance": float("nan"),
        }
        rows.append(
            {
                "evaluationCohort": cohort,
                "candidateId": candidate,
                "branchFamily": family,
                "targetId": target,
                "states": len(group),
                "eligibleStates": len(eligible),
                "eligibleFraction": len(eligible) / len(group),
                "meanQ": float(eligible["q"].mean()) if len(eligible) else float("nan"),
                "splitHalfSpearman": safe_spearman(
                    eligible["qHalfA"], eligible["qHalfB"]
                ),
                "intermediateStateCount": int(eligible["intermediateProbability"].sum()),
                **variance,
            }
        )
    return pd.DataFrame(rows)


def transfer_pairs(states: pd.DataFrame) -> pd.DataFrame:
    index = states.set_index(["stateId", "branchFamily", "targetId"])
    rows = []
    for state_id in states["stateId"].unique():
        try:
            original_h32 = index.loc[(state_id, "H32", "ORIGINAL")]
            original_h8 = index.loc[(state_id, "H8", "ORIGINAL")]
        except KeyError:
            continue
        refs = {}
        for reference in REFERENCES:
            for family in FAMILIES:
                key = (state_id, family, reference)
                refs[(family, reference)] = index.loc[key] if key in index.index else None
        base = {
            "stateId": state_id,
            "evaluationCohort": original_h32.evaluationCohort,
            "candidateId": original_h32.candidateId,
            "matrixIndex": int(original_h32.matrixIndex),
            "landmark": int(original_h32.landmark),
            "originalQ32": float(original_h32.q),
            "originalQ8": float(original_h8.q),
        }
        for family in FAMILIES:
            for reference in REFERENCES:
                row = refs[(family, reference)]
                base[f"{reference}_{family}"] = (
                    float(row.q) if row is not None else float("nan")
                )
        base["referenceMeanH32"] = float(
            np.nanmean([base["REFERENCE_A_H32"], base["REFERENCE_B_H32"]])
        ) if np.isfinite([base["REFERENCE_A_H32"], base["REFERENCE_B_H32"]]).any() else float("nan")
        base["referenceMeanH8"] = float(
            np.nanmean([base["REFERENCE_A_H8"], base["REFERENCE_B_H8"]])
        ) if np.isfinite([base["REFERENCE_A_H8"], base["REFERENCE_B_H8"]]).any() else float("nan")
        rows.append(base)
    return pd.DataFrame(rows).sort_values(
        ["evaluationCohort", "candidateId", "landmark", "matrixIndex"]
    ).reset_index(drop=True)


COMPARISON_SPECS = {
    "REFERENCE_A_VS_REFERENCE_B_H32": ("REFERENCE_A_H32", "REFERENCE_B_H32"),
    "REFERENCE_A_H8_VS_H32": ("REFERENCE_A_H8", "REFERENCE_A_H32"),
    "REFERENCE_B_H8_VS_H32": ("REFERENCE_B_H8", "REFERENCE_B_H32"),
    "ORIGINAL_H8_VS_REFERENCE_MEAN_H32": ("originalQ8", "referenceMeanH32"),
    "ORIGINAL_H32_VS_REFERENCE_MEAN_H32": ("originalQ32", "referenceMeanH32"),
    "REFERENCE_MEAN_H8_VS_H32": ("referenceMeanH8", "referenceMeanH32"),
}


def transfer_results(pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    bootstrap_rows = []
    for (cohort, candidate), group in pairs.groupby(
        ["evaluationCohort", "candidateId"], sort=True
    ):
        group = group.reset_index(drop=True)
        for metric, (left_name, right_name) in COMPARISON_SPECS.items():
            left = group[left_name].to_numpy(dtype=np.float64)
            right = group[right_name].to_numpy(dtype=np.float64)
            valid = np.isfinite(left) & np.isfinite(right)
            point = safe_spearman(left, right)
            rng = np.random.default_rng(
                derived_seed("transfer_bootstrap", cohort, candidate, metric)
            )
            distribution = np.full(BOOTSTRAPS, np.nan)
            if valid.sum() >= 3:
                x = left[valid]
                y = right[valid]
                indices = rng.integers(0, len(x), size=(BOOTSTRAPS, len(x)))
                for replicate, sample in enumerate(indices):
                    xs = x[sample]
                    ys = y[sample]
                    if np.unique(xs).size < 2 or np.unique(ys).size < 2:
                        continue
                    distribution[replicate] = np.corrcoef(
                        rankdata(xs), rankdata(ys)
                    )[0, 1]
            finite = distribution[np.isfinite(distribution)]
            summary_rows.append(
                {
                    "evaluationCohort": cohort,
                    "candidateId": candidate,
                    "comparisonId": metric,
                    "states": len(group),
                    "definedPairs": int(valid.sum()),
                    "spearman": point,
                    "lower95": float(np.quantile(finite, 0.025))
                    if len(finite)
                    else float("nan"),
                    "upper95": float(np.quantile(finite, 0.975))
                    if len(finite)
                    else float("nan"),
                    "rankGatePassed": bool(
                        np.isfinite(point)
                        and point > 0.5
                        and len(finite)
                        and np.quantile(finite, 0.025) > 0.3
                    ),
                }
            )
            bootstrap_rows.append(
                {
                    "evaluationCohort": cohort,
                    "candidateId": candidate,
                    "comparisonId": metric,
                    "replicates": BOOTSTRAPS,
                    "definedReplicates": len(finite),
                    "distributionSha256": hashlib.sha256(
                        np.asarray(distribution, dtype="<f8").tobytes()
                    ).hexdigest(),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(bootstrap_rows)


def reliability_bootstrap_intervals(states: pd.DataFrame) -> pd.DataFrame:
    rows = []
    subset = states[
        states["evaluationCohort"].isin(EVALUATION_COHORTS)
        & states["branchFamily"].eq("H32")
        & states["targetId"].isin(REFERENCES)
        & states["committorEligible"]
    ]
    for keys, group in subset.groupby(
        ["evaluationCohort", "candidateId", "targetId"], sort=True
    ):
        cohort, candidate, target = keys
        group = group.reset_index(drop=True)
        rng = np.random.default_rng(
            derived_seed("reliability_bootstrap", cohort, candidate, target)
        )
        corrected = np.full(BOOTSTRAPS, np.nan)
        halves = np.full(BOOTSTRAPS, np.nan)
        for replicate in range(BOOTSTRAPS):
            sample = group.iloc[rng.integers(0, len(group), size=len(group))]
            corrected[replicate] = corrected_between_state_variance(
                sample["q"].to_numpy(dtype=np.float64), 128
            )["correctedBetweenStateVariance"]
            halves[replicate] = safe_spearman(sample["qHalfA"], sample["qHalfB"])
        rows.append(
            {
                "evaluationCohort": cohort,
                "candidateId": candidate,
                "targetId": target,
                "replicates": BOOTSTRAPS,
                "correctedVarianceLower95": float(np.nanquantile(corrected, 0.025)),
                "correctedVarianceUpper95": float(np.nanquantile(corrected, 0.975)),
                "splitHalfSpearmanLower95": float(np.nanquantile(halves, 0.025)),
                "splitHalfSpearmanUpper95": float(np.nanquantile(halves, 0.975)),
                "correctedDistributionSha256": hashlib.sha256(
                    np.asarray(corrected, dtype="<f8").tobytes()
                ).hexdigest(),
                "halfDistributionSha256": hashlib.sha256(
                    np.asarray(halves, dtype="<f8").tobytes()
                ).hexdigest(),
            }
        )
    return pd.DataFrame(rows)


def centroid_agreement_bootstrap(
    comparisons: pd.DataFrame,
) -> pd.DataFrame:
    subset = comparisons[
        comparisons["leftTarget"].eq("REFERENCE_A")
        & comparisons["rightTarget"].eq("REFERENCE_B")
    ]
    rows = []
    for (cohort, candidate), group in subset.groupby(
        ["evaluationCohort", "candidateId"], sort=True
    ):
        values = group["strictH090Agreement"].to_numpy(dtype=np.float64)
        rng = np.random.default_rng(
            derived_seed("centroid_bootstrap", cohort, candidate)
        )
        distribution = values[
            rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))
        ].mean(axis=1)
        rows.append(
            {
                "evaluationCohort": cohort,
                "candidateId": candidate,
                "replicates": BOOTSTRAPS,
                "agreementFraction": float(values.mean()),
                "agreementLower95": float(np.quantile(distribution, 0.025)),
                "agreementUpper95": float(np.quantile(distribution, 0.975)),
                "distributionSha256": hashlib.sha256(
                    np.asarray(distribution, dtype="<f8").tobytes()
                ).hexdigest(),
            }
        )
    return pd.DataFrame(rows)


def scientific_gates(
    availability: pd.DataFrame,
    transfer: pd.DataFrame,
    reliability: pd.DataFrame,
    reliability_bootstrap: pd.DataFrame,
    centroid_bootstrap: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], str]:
    rows = []
    for cohort in EVALUATION_COHORTS:
        for candidate in CANDIDATES:
            available = availability[
                availability["evaluationCohort"].eq(cohort)
                & availability["candidateId"].eq(candidate)
            ].iloc[0]
            centroid = centroid_bootstrap[
                centroid_bootstrap["evaluationCohort"].eq(cohort)
                & centroid_bootstrap["candidateId"].eq(candidate)
            ].iloc[0]
            transfer_group = transfer[
                transfer["evaluationCohort"].eq(cohort)
                & transfer["candidateId"].eq(candidate)
            ].set_index("comparisonId")
            reliability_group = reliability[
                reliability["evaluationCohort"].eq(cohort)
                & reliability["candidateId"].eq(candidate)
                & reliability["branchFamily"].eq("H32")
                & reliability["targetId"].isin(REFERENCES)
            ].set_index("targetId")
            boot_group = reliability_bootstrap[
                reliability_bootstrap["evaluationCohort"].eq(cohort)
                & reliability_bootstrap["candidateId"].eq(candidate)
            ].set_index("targetId")
            reliability_pass = all(
                reliability_group.loc[target, "correctedBetweenStateVariance"] > 0
                and boot_group.loc[target, "correctedVarianceLower95"] > 0
                and reliability_group.loc[target, "splitHalfSpearman"] > 0.5
                and boot_group.loc[target, "splitHalfSpearmanLower95"] > 0.3
                and reliability_group.loc[target, "intermediateStateCount"] >= 5
                for target in REFERENCES
            )
            corresponding_h8_pass = all(
                bool(
                    transfer_group.loc[
                        f"{target}_H8_VS_H32", "rankGatePassed"
                    ]
                )
                for target in REFERENCES
            )
            rows.append(
                {
                    "evaluationCohort": cohort,
                    "candidateId": candidate,
                    "availabilityPassed": bool(available.availabilityGatePassed),
                    "centroidAgreementPointPassed": bool(
                        available.centroidAgreementGatePassed
                    ),
                    "centroidAgreementLowerPassed": bool(
                        centroid.agreementLower95 >= 0.7
                    ),
                    "referenceH32ReliabilityPassed": reliability_pass,
                    "referenceAReferenceBH32RankPassed": bool(
                        transfer_group.loc[
                            "REFERENCE_A_VS_REFERENCE_B_H32", "rankGatePassed"
                        ]
                    ),
                    "correspondingReferenceH8Passed": corresponding_h8_pass,
                    "originalH8TransferPassed": bool(
                        transfer_group.loc[
                            "ORIGINAL_H8_VS_REFERENCE_MEAN_H32", "rankGatePassed"
                        ]
                    ),
                }
            )
    gates = pd.DataFrame(rows)
    gates["independentBasinStable"] = gates[
        [
            "availabilityPassed",
            "centroidAgreementPointPassed",
            "centroidAgreementLowerPassed",
            "referenceH32ReliabilityPassed",
            "referenceAReferenceBH32RankPassed",
            "correspondingReferenceH8Passed",
        ]
    ].all(axis=1)
    gates["originalShootingTransfers"] = gates[
        ["independentBasinStable", "originalH8TransferPassed"]
    ].all(axis=1)
    stable = bool(gates["independentBasinStable"].all())
    transfer_pass = bool(gates["originalShootingTransfers"].all())
    if stable and transfer_pass:
        classifications = [
            "INDEPENDENT_LINEAGE_BASIN_COMMITTOR_STABLE",
            "SHOOTING_COORDINATE_TRANSFERS_ACROSS_LINEAGE_DEFINED_BASINS",
            "NOT_PAST_ONLY_OR_PAPER_CONFIRMED",
        ]
        next_theme = "INDEPENDENT_BASIN_TEACHER_STUDENT_CURRENT_STATE_AUDIT"
    elif stable:
        classifications = [
            "INDEPENDENT_LINEAGE_BASIN_COMMITTOR_STABLE",
            "ORIGINAL_TARGET_SHOOTING_COORDINATE_NONTRANSFER",
            "NOT_PROMOTABLE_AS_CONFIRMED",
        ]
        next_theme = "INDEPENDENT_BASIN_TEACHER_STUDENT_CURRENT_STATE_AUDIT"
    else:
        classifications = [
            "TARGET_BASIN_LINEAGE_SPECIFIC",
            "CURRENT_COMMITTOR_TARGET_NOT_NETWORK_STABLE",
            "NOT_PROMOTABLE_AS_CONFIRMED",
        ]
        next_theme = "MULTILINEAGE_ANY_ATTRACTOR_ENTRY_TARGET_CONSTRUCTION"
    return gates, classifications, next_theme


def fixture_results() -> pd.DataFrame:
    states = np.array([[4, 0, 0], [3, 1, 0], [0, 4, 0]], dtype=np.int64)
    centroids = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
    scores = cosine_scores(states, centroids)
    summary = summarize_scores(scores[:, 0], threshold=0.9)
    return pd.DataFrame(
        [
            {
                "fixtureId": "MULTITARGET_SCORE_SHAPE",
                "passed": scores.shape == (3, 2),
                "details": str(scores.shape),
            },
            {
                "fixtureId": "FIRST_ENTRY_ONE_BASED",
                "passed": summary.first_entry_offset_one_based == 1,
                "details": str(summary.first_entry_offset_one_based),
            },
            {
                "fixtureId": "CENTROID_SCALE_INVARIANCE",
                "passed": np.isclose(
                    centroid_similarity(np.array([1.0, 2.0]), np.array([2.0, 4.0])),
                    1,
                ),
                "details": "H=1",
            },
            {
                "fixtureId": "THREE_TARGETS_ONLY",
                "passed": TARGETS == ("ORIGINAL", "REFERENCE_A", "REFERENCE_B"),
                "details": json.dumps(TARGETS),
            },
            {
                "fixtureId": "FROZEN_HORIZONS_BRANCHES",
                "passed": HORIZONS == {"H32": 32, "H8": 8}
                and BRANCH_COUNTS == {"H32": 128, "H8": 64},
                "details": json.dumps({"horizons": HORIZONS, "branches": BRANCH_COUNTS}),
            },
            {
                "fixtureId": "BOOTSTRAP_SCOPE",
                "passed": BOOTSTRAPS == 4096,
                "details": "matrix bootstrap",
            },
        ]
    )


def benchmark_projection(
    units: pd.DataFrame,
    base_payloads: list[dict[str, Any]],
    original_coordinates: pd.DataFrame,
) -> dict[str, Any]:
    benchmark_hex = hashlib.sha256(b"E01-S19-L36-BENCHMARK-ONLY").hexdigest()
    chosen_units = [
        units[units["candidateId"].eq(candidate)].iloc[0].to_dict()
        for candidate in CANDIDATES
    ]
    reference_durations = []
    for unit in chosen_units:
        for reference in REFERENCES:
            started = time.perf_counter()
            beta = L28.generate_beta(
                L28.derive_seed(
                    L28.L23_ROOT_HEX,
                    L28.L23_PHASE,
                    "catalytic_matrix",
                    int(unit["matrixIndex"]),
                )
            )
            simulate_trajectory(
                phase="s19_l36_benchmark_only",
                root_hex=benchmark_hex,
                matrix_index=int(unit["matrixIndex"]),
                definition=L28.definition(unit["candidateId"]),
                stream_identity=f"{unit['candidateId']}__{reference}",
                beta=beta,
                initial_state=np.asarray(unit["initialState"], dtype=np.int64),
            )
            reference_durations.append(time.perf_counter() - started)
    payload_map = {row["stateId"]: row for row in base_payloads}
    coordinate_map = {
        state_id: group.sort_values("coordinate")["centroidValue"].tolist()
        for state_id, group in original_coordinates.groupby("stateId", sort=False)
    }
    branch_durations = []
    for unit in chosen_units:
        payload = dict(payload_map[unit["stateId"]])
        original = coordinate_map[unit["stateId"]]
        payload["targets"] = {
            "ORIGINAL": original,
            "REFERENCE_A": np.roll(original, 1).tolist(),
            "REFERENCE_B": np.roll(original, 2).tolist(),
        }
        started = time.perf_counter()
        rows = _branch_worker(payload)
        branch_durations.append(time.perf_counter() - started)
        if not rows:
            raise RuntimeError("L36 branch benchmark failed")
    projected_reference_cpu = max(reference_durations) * 560 * 2.2 / 3600
    projected_branch_cpu = max(branch_durations) * 280 * 2.2 / 3600
    projected_cpu = projected_reference_cpu + projected_branch_cpu
    projected_wall = projected_cpu * 3600 / WORKERS
    return {
        "schema": "eidosoma.e01.s19_l36.benchmark_projection.v1",
        "status": "PASS"
        if projected_cpu <= 90 and projected_wall <= 64.8 * 3600
        else "STOP_BEFORE_OUTCOME",
        "referenceDurationsSeconds": reference_durations,
        "branchDurationsSeconds": branch_durations,
        "projectedCpuHoursIncludingRegeneration": projected_cpu,
        "projectedWallSecondsIncludingRegeneration": projected_wall,
        "referenceTrajectories": 560,
        "replayedBranchStreams": 53_760,
    }


def analysis_seed_manifest() -> pd.DataFrame:
    parts_list: list[tuple[object, ...]] = []
    for cohort in COHORTS:
        for candidate in CANDIDATES:
            parts_list.append(("centroid_bootstrap", cohort, candidate))
            for comparison in COMPARISON_SPECS:
                parts_list.append(
                    ("transfer_bootstrap", cohort, candidate, comparison)
                )
    for cohort in EVALUATION_COHORTS:
        for candidate in CANDIDATES:
            for target in REFERENCES:
                parts_list.append(
                    ("reliability_bootstrap", cohort, candidate, target)
                )
    rows = []
    for parts in parts_list:
        payload = "\x1f".join([VERSION, ROOT_HEX, *map(str, parts)])
        rows.append(
            {
                "purpose": str(parts[0]),
                "partsJson": json.dumps(parts, separators=(",", ":")),
                "rootHex": ROOT_HEX,
                "derivedSeed": str(derived_seed(*parts)),
                "seedMaterialSha256": hashlib.sha256(payload.encode()).hexdigest(),
                "scientificTrajectorySeed": False,
            }
        )
    result = pd.DataFrame(rows).sort_values(["purpose", "partsJson"]).reset_index(
        drop=True
    )
    if result["seedMaterialSha256"].duplicated().any() or result[
        "derivedSeed"
    ].duplicated().any():
        raise RuntimeError("L36 analysis seed collision")
    return result


def analysis_seed_firewall(seeds: pd.DataFrame) -> dict[str, Any]:
    prior_material: set[str] = set()
    prior_derived: set[str] = set()
    for path in ARTIFACT_ROOT.glob("loops/L*/**/*seed*manifest*.parquet"):
        if "/L36/" in str(path):
            continue
        try:
            frame = pd.read_parquet(path)
        except (OSError, ValueError, TypeError):
            continue
        for column in frame.columns:
            if "seedmaterialsha256" in column.lower():
                prior_material.update(frame[column].dropna().astype(str))
            if column.lower() == "derivedseed":
                prior_derived.update(frame[column].dropna().astype(str))
    material = sorted(set(seeds["seedMaterialSha256"]) & prior_material)
    derived = sorted(set(seeds["derivedSeed"]) & prior_derived)
    return {
        "schema": "eidosoma.e01.s19_l36.analysis_seed_firewall.v1",
        "status": "PASS" if not material and not derived else "FAIL",
        "seedCount": len(seeds),
        "materialOverlapCount": len(material),
        "derivedOverlapCount": len(derived),
        "materialOverlaps": material,
        "derivedOverlaps": derived,
    }


def source_grounding_registry() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sourceId": "REVIEWER_INDEPENDENT_BASIN_DIRECTION",
                "evidenceClass": "HUMAN_REVIEW_DIRECTION",
                "finding": "Compare the completed-trajectory basin with a basin identified from an independent lineage under the same catalytic matrix.",
                "frozenUse": "two independently seeded reference lineages per state/matrix/candidate",
            },
            {
                "sourceId": "L23_FROZEN_TARGET",
                "evidenceClass": "DIRECT_FROZEN_E01_METHOD",
                "finding": "Dominant strict-H090 post-fission connected-component centroid defines the current target basin.",
                "frozenUse": "unchanged target construction for ORIGINAL and both references",
            },
            {
                "sourceId": "L28_RELIABLE_ORIGINAL_COMMITTOR",
                "evidenceClass": "DIRECT_FROZEN_E01_RESULT",
                "finding": "Original completed-lineage target produced reliable state-dependent H32 committor variation.",
                "frozenUse": "original-target comparator and exact H32 path replay",
            },
            {
                "sourceId": "L30_L31_H8_TEACHER",
                "evidenceClass": "DIRECT_FROZEN_E01_RESULT",
                "finding": "H8 shooting against the original target predicts original H32 committor across two cohorts.",
                "frozenUse": "cross-target transfer test without refitting or new branch streams",
            },
        ]
    )


def make_figures(
    references: pd.DataFrame,
    comparisons: pd.DataFrame,
    availability: pd.DataFrame,
    states: pd.DataFrame,
    reliability: pd.DataFrame,
    transfer: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    root = BUILD_ROOT / "figures"
    root.mkdir(parents=True, exist_ok=True)

    def save(name: str) -> None:
        plt.tight_layout()
        plt.savefig(root / name, dpi=180)
        plt.close()

    references.pivot_table(
        index="evaluationCohort",
        columns=["candidateId", "referenceId"],
        values="completedFissions",
        aggfunc="mean",
    ).plot(kind="bar", figsize=(12, 5))
    plt.axhline(100, color="black", linestyle="--", linewidth=1)
    plt.ylabel("Mean completed fissions")
    save("01_reference_lineage_completion.png")

    comp = comparisons[
        comparisons["leftTarget"].eq("REFERENCE_A")
        & comparisons["rightTarget"].eq("REFERENCE_B")
    ]
    _, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    for axis, ((cohort, candidate), group) in zip(
        axes.flat,
        comp.groupby(["evaluationCohort", "candidateId"], sort=True),
        strict=True,
    ):
        axis.hist(group["centroidH"].dropna(), bins=15)
        axis.axvline(0.9, color="red", linestyle="--")
        axis.set_title(f"{cohort}/{candidate[-2:]}", fontsize=8)
        axis.set_xlabel("H(reference A, reference B)")
    save("02_reference_centroid_agreement.png")

    availability.set_index(["evaluationCohort", "candidateId"])[
        [
            "referenceAEligibleFraction",
            "referenceBEligibleFraction",
            "referenceAAtRiskFraction",
            "referenceBAtRiskFraction",
            "referenceCentroidAgreementFraction",
        ]
    ].plot(kind="bar", figsize=(14, 6))
    plt.ylim(0, 1)
    plt.ylabel("Fraction")
    save("03_target_availability_and_at_risk.png")

    q = states[
        states["branchFamily"].eq("H32")
        & states["targetId"].isin(REFERENCES)
    ].pivot(index="stateId", columns="targetId", values="q").dropna()
    plt.figure(figsize=(6, 6))
    plt.scatter(q["REFERENCE_A"], q["REFERENCE_B"], s=14, alpha=0.65)
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("H32 q to reference A")
    plt.ylabel("H32 q to reference B")
    save("04_reference_committor_agreement.png")

    reliability[
        reliability["evaluationCohort"].isin(EVALUATION_COHORTS)
        & reliability["branchFamily"].eq("H32")
    ].pivot_table(
        index="targetId",
        columns=["evaluationCohort", "candidateId"],
        values="splitHalfSpearman",
    ).plot(kind="bar", figsize=(13, 6))
    plt.axhline(0.5, color="black", linestyle="--")
    plt.ylabel("H32 independent-half Spearman")
    save("05_reference_committor_reliability.png")

    transfer[
        transfer["evaluationCohort"].isin(EVALUATION_COHORTS)
    ].pivot_table(
        index="comparisonId",
        columns=["evaluationCohort", "candidateId"],
        values="spearman",
    ).plot(kind="bar", figsize=(15, 6))
    plt.axhline(0.5, color="black", linestyle="--")
    plt.ylabel("Spearman")
    save("06_cross_target_transfer.png")

    checks = [
        "availabilityPassed",
        "centroidAgreementPointPassed",
        "centroidAgreementLowerPassed",
        "referenceH32ReliabilityPassed",
        "referenceAReferenceBH32RankPassed",
        "correspondingReferenceH8Passed",
        "originalH8TransferPassed",
        "independentBasinStable",
        "originalShootingTransfers",
    ]
    matrix = gates.set_index(["evaluationCohort", "candidateId"])[checks].astype(float)
    plt.figure(figsize=(13, 5))
    plt.imshow(matrix, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    plt.xticks(range(len(checks)), checks, rotation=35, ha="right", fontsize=7)
    plt.yticks(
        range(len(matrix)), ["/".join(index) for index in matrix.index], fontsize=7
    )
    plt.colorbar(ticks=[0, 1])
    save("07_basin_transfer_gate_matrix.png")


def manifest_for(root: Path) -> dict[str, Any]:
    files = [
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
        "schema": "eidosoma.e01.s19_l36.artifact_manifest.v1",
        "root": str(root),
        "fileCount": len(files),
        "totalBytes": sum(item["bytes"] for item in files),
        "files": files,
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
            "beliefBeforeLoop": "The original committor may be hard to transfer because its target basin is reconstructed from the same completed lineage.",
            "failureOrAmbiguityTargeted": "Trajectory-specific target geometry versus beta-conditioned reproducible attractor.",
            "informationGainRationale": "Two independent same-beta/same-initial reference lineages directly test target stability before another student model.",
            "learned": "L36 two-reference target and exact existing-branch rescore contract frozen before new reference outcomes.",
            "ledgerSequence": sequence,
            "loopId": LOOP_ID,
            "motivatingEvidence": "Reviewer basin-transfer proposal and L35 target/entry-dominated teacher result.",
            "proposedNextTest": "Generate exactly two reference lineages and audit centroid/committor/H8 transfer.",
            "recordPhase": "PRE_LOOP_METHOD_LOCK",
            "remainingPlausibleHypotheses": "Network-stable attractor, multi-attractor lineage specificity, or shooting-only target-conditioned information.",
            "selectedHypotheses": "Frozen original versus reference A/reference B basin transfer.",
            "timestampUtc": timestamp,
            "weakenedHypotheses": "The current completed-run basin can be assumed network-stable without direct evidence.",
        },
        {
            "appendOnly": True,
            "beliefBeforeLoop": "A stable target must reproduce across both independent lineages, candidates and evaluation cohorts.",
            "failureOrAmbiguityTargeted": "Basin availability, centroid identity and response transfer.",
            "informationGainRationale": "Exact common-stream H32/H8 rescoring isolates target identity from simulator-path differences.",
            "learned": ";".join(classifications),
            "ledgerSequence": sequence + 1,
            "loopId": LOOP_ID,
            "motivatingEvidence": "Complete L36 result.",
            "proposedNextTest": next_theme,
            "recordPhase": "POST_LOOP_RESULT_AUTONOMOUS_CONTINUATION",
            "remainingPlausibleHypotheses": next_theme,
            "selectedHypotheses": "Two independent completed-lineage basins under fixed beta/initial state.",
            "timestampUtc": timestamp,
            "weakenedHypotheses": "Original completed-lineage basin is automatically the unique catalytic-network attractor.",
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
        + f"\n\n## {LOOP_ID} — independent-lineage basin transfer\n\n"
        + f"- **Learned:** {', '.join(classifications)}.\n"
        + f"- **Next:** {next_theme}.\n",
    )
    candidates_path = ARTIFACT_ROOT / "candidate_registry.parquet"
    candidates = pd.read_parquet(candidates_path)
    row = {
        "branchCount": 3,
        "bundleId": "L36_INDEPENDENT_LINEAGE_BASIN_TRANSFER",
        "candidateId": "S19-L36-TWO-INDEPENDENT-REFERENCE-BASINS",
        "candidateSpecificSuccess": 0,
        "completedFitLeakage": 1,
        "computeEfficiency": 4,
        "crossCandidateDiscriminability": 5,
        "deterministicHReuse": 0,
        "explanatoryLeverage": 5,
        "frozenRank": 1,
        "independenceFromPriorOutcomeSelection": 5,
        "outcomeGuidedThresholdSelection": 0,
        "paperFingerprintSpecificity": 0,
        "proposedSpecification": "original and two same-beta/same-initial independent L23 target basins; exact H32/H8 path rescore",
        "rankingScore": 29.0,
        "registryOrder": int(candidates["registryOrder"].max()) + 1,
        "selected": True,
        "selectionReason": "REVIEWER_TARGET_BASIN_TRANSFER_HYPOTHESIS",
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
    source_path = ARTIFACT_ROOT / "source_search_ledger.parquet"
    sources = pd.read_parquet(source_path)
    source_rows = [
        {
            "commitOrVersion": None,
            "evidenceClass": source.evidenceClass,
            "finding": f"{source.finding}; L36 use: {source.frozenUse}",
            "licenseStatus": "WORKSPACE_OR_HUMAN_DIRECTION",
            "redistributionStatus": "INTERNAL_EVIDENCE_ONLY",
            "repositoryIdentity": None,
            "retainedPath": None,
            "retrievalDate": timestamp[:10],
            "sha256": None,
            "sourceId": f"L36_{source.sourceId}",
            "sourceType": source.evidenceClass,
            "treeIdentity": None,
            "url": None,
        }
        for source in source_grounding_registry().itertuples(index=False)
    ]
    BASE.write_parquet(
        source_path,
        pd.concat(
            [sources, pd.DataFrame(source_rows).reindex(columns=sources.columns)],
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
            "selectedDiscoveryLead": None,
            "newMatrices": 0,
            "newTrajectories": 560,
            "newBranchStreams": 0,
            "nextStepActive": True,
        }
    )
    registry["proposedNextLoopTheme"] = next_theme
    registry["proposedNextLoopActive"] = True
    BASE.atomic_text(registry_path, yaml.safe_dump(registry, sort_keys=False))
    history_path = ARTIFACT_ROOT / "human_review_history.json"
    history = json.loads(history_path.read_text())
    history["history"].append(
        {
            "decision": "S19_L36_COMPLETE_AUTONOMOUS_CONTINUATION",
            "loopId": LOOP_ID,
            "nextLoopAuthorized": True,
            "recordedAtUtc": timestamp,
            "result": classifications,
            "s20Activated": False,
            "scope": VERSION,
            "source": "locked_execution_result",
        }
    )
    history["pendingDecision"] = "NONE_AUTONOMOUS_SEQUENCE_ACTIVE_THROUGH_L42"
    BASE.write_json(history_path, history)


def report_text(
    availability: pd.DataFrame,
    comparisons: pd.DataFrame,
    reliability: pd.DataFrame,
    transfer: pd.DataFrame,
    gates: pd.DataFrame,
    classifications: list[str],
    runtime: dict[str, Any],
    next_theme: str,
) -> str:
    eval_availability = availability[
        availability["evaluationCohort"].isin(EVALUATION_COHORTS)
    ]
    eval_reliability = reliability[
        reliability["evaluationCohort"].isin(EVALUATION_COHORTS)
        & reliability["branchFamily"].eq("H32")
    ]
    eval_transfer = transfer[
        transfer["evaluationCohort"].isin(EVALUATION_COHORTS)
    ]
    centroid = comparisons[
        comparisons["leftTarget"].eq("REFERENCE_A")
        & comparisons["rightTarget"].eq("REFERENCE_B")
        & comparisons["evaluationCohort"].isin(EVALUATION_COHORTS)
    ]
    centroid_summary = centroid.groupby(["evaluationCohort", "candidateId"]).agg(
        states=("stateId", "size"),
        meanH=("centroidH", "mean"),
        medianH=("centroidH", "median"),
        strictH090Agreement=("strictH090Agreement", "mean"),
    ).reset_index()
    return f"""# S19-L36 — Independent-Lineage Basin Transfer Audit

## Chief/human handoff

- **Step:** `{VERSION}`
- **Status:** complete under the authorized L19–L42 sequence.
- **Classifications:** {", ".join(f"`{value}`" for value in classifications)}
- **Validation:** two independently seeded reference lineages per 280 frozen state/matrix/candidate units; exact beta and initial-state reuse; exact discrete/path original-target replay of all 53,760 H32/H8 branch streams with the recorded TA01 float64 numerical-equivalence contract; independent full lineage and rescore regeneration; 4,096 matrix bootstraps; immutable/runtime/storage/artifact hashes.
- **Recommended next action:** `{next_theme}`.

## Question

Is the basin used by the empirical committor a reproducible property of the catalytic network, or a trajectory-specific object reconstructed from the same completed lineage being explained? L36 changes only target provenance. For every frozen state it generates two independent 100-fission reference lineages under the same beta, initial state and candidate semantics, applies the unchanged L23 target construction, and rescored the exact existing H32/H8 stochastic paths against ORIGINAL, REFERENCE_A and REFERENCE_B targets.

## Availability and basin agreement

{eval_availability.to_markdown(index=False)}

Reference-centroid agreement:

{centroid_summary.to_markdown(index=False)}

Incomplete reference lineages and states already inside an independent target were retained as explicit ineligible units and were never replaced.

## Independent-target committor reliability

{eval_reliability[['evaluationCohort','candidateId','targetId','eligibleStates','meanQ','splitHalfSpearman','intermediateStateCount','correctedBetweenStateVariance']].to_markdown(index=False)}

## Cross-lineage response and teacher transfer

{eval_transfer[['evaluationCohort','candidateId','comparisonId','definedPairs','spearman','lower95','upper95','rankGatePassed']].to_markdown(index=False)}

## Frozen decision gates

{gates.to_markdown(index=False)}

The classifications are {', '.join(classifications)}. A stable independent target would still be completed-lineage-conditioned; a transferring H8 coordinate would still use forward stochastic shooting. Neither result is a past-observable early-warning marker.

## Validation and provenance

- Repository lock: `{runtime['repositoryHead']}`.
- Failed attempt 01 remains recorded. TA01 changed only the score-extrema replay comparison from bit equality to finite-mask equality plus absolute and relative error <= `1e-12` and ULP distance <= `16`; labels, entry times, clocks, branch paths and all later scientific calculations were unchanged.
- Workers: `{runtime['workers']}` with one numerical-library thread per worker; GPU hours `0`.
- Wall time: `{runtime['wallSeconds']:.2f}` seconds.
- Reference trajectories generated: `{runtime['referenceTrajectoriesGenerated']}` plus the same full regeneration scope.
- Unique frozen H32/H8 branch streams rescored: `{runtime['uniqueFrozenBranchStreamsRescored']}`; new branch streams: `0`.
- No matrix, initial state, target threshold, target construction, branch horizon, simulator or incomplete unit was changed or replaced.

## Limitations

REFERENCE_A and REFERENCE_B are full-trajectory retrospective constructions. Two references can reveal lineage dependence but cannot exhaust a genuinely multimodal attractor landscape. The same original state is evaluated against target basins that may not be reachable from it, which is why current-inside and availability status are explicit. One state per matrix still prevents within-matrix ordering. This audit does not test PhiRL, the paper's prediction claim, intervention, or causal control.

## Next boundary

L36 is frozen. The standing authorization permits `{next_theme}` as the only next loop. S20, E02, author contact, interventions, reactive-current work and report generation remain inactive.
"""


def prepare_lock() -> None:
    LOOP_ROOT.mkdir(parents=True, exist_ok=True)
    if git("status", "--porcelain=v1"):
        raise RuntimeError("repository must be clean before L36 lock")
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/eidosoma/groups/42")
    if head != remote:
        raise RuntimeError("local/remote commit mismatch")
    prior = validate_immutable_prior()
    fixtures = fixture_results()
    responses = L35.response_registry()
    coordinates = L33.target_coordinates(responses)
    trajectory_manifest = pd.read_parquet(
        L23_ROOT / "input_trajectory_manifest.parquet"
    )
    base_payloads = L35.payloads(responses, coordinates, trajectory_manifest)
    units = reference_unit_registry(responses, trajectory_manifest)
    reference_seeds = reference_seed_manifest(units)
    firewall = seed_firewall(reference_seeds)
    analysis_seeds = analysis_seed_manifest()
    analysis_firewall = analysis_seed_firewall(analysis_seeds)
    benchmark = benchmark_projection(units, base_payloads, coordinates)
    if (
        not prior["unchanged"]
        or not fixtures["passed"].all()
        or not units[["initialStateExact", "betaExact"]].all().all()
        or firewall["status"] != "PASS"
        or analysis_firewall["status"] != "PASS"
        or benchmark["status"] != "PASS"
    ):
        raise RuntimeError("L36 preoutcome validation or benchmark gate failed")
    shutil.copy2(CONFIG, LOOP_ROOT / "preregistration.yaml")
    BASE.atomic_text(
        LOOP_ROOT / "decision_record.md",
        "# S19-L36 decision record\n\n"
        "L35 showed that realized entry and completed-run basin geometry dominate the successful shooting teacher while no frozen physical branch mechanism transferred across both candidates and cohorts. The reviewer identified target provenance as the next ambiguity: the basin itself was reconstructed from the completed trajectory being explained. L36 freezes two independent same-beta, same-initial-state reference lineages per state before their outcomes. ORIGINAL, REFERENCE_A and REFERENCE_B use exactly the same L23 dominant post-fission strict-H090 component-centroid construction. Existing H32/H8 paths are replayed once and rescored without new branch streams. Incomplete lineages and current-inside-target states remain explicit; references cannot be selected by outcome.\n",
    )
    BASE.write_parquet(LOOP_ROOT / "fixture_results.parquet", fixtures)
    BASE.write_json(LOOP_ROOT / "benchmark_projection.json", benchmark)
    BASE.write_parquet(LOOP_ROOT / "response_registry.parquet", responses)
    BASE.write_parquet(LOOP_ROOT / "original_target_coordinates.parquet", coordinates)
    BASE.write_parquet(LOOP_ROOT / "reference_unit_registry.parquet", units)
    BASE.write_parquet(LOOP_ROOT / "reference_seed_manifest.parquet", reference_seeds)
    BASE.write_json(LOOP_ROOT / "seed_firewall.json", firewall)
    BASE.write_parquet(LOOP_ROOT / "analysis_seed_manifest.parquet", analysis_seeds)
    BASE.write_json(LOOP_ROOT / "analysis_seed_firewall.json", analysis_firewall)
    BASE.write_parquet(
        LOOP_ROOT / "source_grounding_registry.parquet", source_grounding_registry()
    )
    BASE.write_json(LOOP_ROOT / "immutable_prior_validation.json", prior)
    hashes = {
        "responsesSha256": sha256_file(LOOP_ROOT / "response_registry.parquet"),
        "originalCoordinatesSha256": sha256_file(
            LOOP_ROOT / "original_target_coordinates.parquet"
        ),
        "referenceUnitsSha256": sha256_file(
            LOOP_ROOT / "reference_unit_registry.parquet"
        ),
        "referenceSeedsSha256": sha256_file(
            LOOP_ROOT / "reference_seed_manifest.parquet"
        ),
        "analysisSeedsSha256": sha256_file(
            LOOP_ROOT / "analysis_seed_manifest.parquet"
        ),
        "seedFirewallSha256": sha256_file(LOOP_ROOT / "seed_firewall.json"),
        "analysisFirewallSha256": sha256_file(
            LOOP_ROOT / "analysis_seed_firewall.json"
        ),
        "benchmarkSha256": sha256_file(LOOP_ROOT / "benchmark_projection.json"),
        "l28BranchesSha256": sha256_file(L28_ROOT / "branch_results.parquet"),
        "l30BranchesSha256": sha256_file(
            L30_ROOT / "short_branch_results.parquet"
        ),
        "l31BranchesSha256": sha256_file(L31_ROOT / "branch_results.parquet"),
        "l23ManifestSha256": sha256_file(
            L23_ROOT / "input_trajectory_manifest.parquet"
        ),
    }
    BASE.write_json(
        LOOP_ROOT / "implementation_lock.json",
        {
            "schema": "eidosoma.e01.s19_l36.implementation_lock.v1",
            "repositoryHead": head,
            "remoteHead": remote,
            "runnerSha256": sha256_file(RUNNER_PATH),
            "coreSha256": sha256_file(CORE_PATH),
            "configSha256": sha256_file(CONFIG),
            "referenceRootHex": ROOT_HEX,
            "referencePhase": PHASE,
            "referenceIds": list(REFERENCES),
            "targets": list(TARGETS),
            "referenceTrajectories": 560,
            "newMatrices": 0,
            "newInitialStates": 0,
            "newBranchStreams": 0,
            "horizons": HORIZONS,
            "branchCounts": BRANCH_COUNTS,
            "targetThreshold": 0.9,
            "targetConstruction": "FROZEN_L23_DOMINANT_POST_FISSION_H090_COMPONENT_CENTROID",
            "matrixBootstraps": BOOTSTRAPS,
            "lockedHashes": hashes,
            "outcomeAccessed": False,
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
            **hashes,
        },
    )


def prepare_technical_amendment() -> None:
    """Freeze the single value-preserving replay repair after failed attempt 01."""

    if git("status", "--porcelain=v1"):
        raise RuntimeError("repository must be clean before L36 technical amendment")
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/eidosoma/groups/42")
    if head != remote:
        raise RuntimeError("local/remote commit mismatch")
    original_lock = json.loads(
        (LOOP_ROOT / "preoutcome_repository_lock.json").read_text()
    )
    prior = validate_immutable_prior()
    locked_files = {
        "responsesSha256": LOOP_ROOT / "response_registry.parquet",
        "originalCoordinatesSha256": LOOP_ROOT / "original_target_coordinates.parquet",
        "referenceUnitsSha256": LOOP_ROOT / "reference_unit_registry.parquet",
        "referenceSeedsSha256": LOOP_ROOT / "reference_seed_manifest.parquet",
        "analysisSeedsSha256": LOOP_ROOT / "analysis_seed_manifest.parquet",
        "seedFirewallSha256": LOOP_ROOT / "seed_firewall.json",
        "analysisFirewallSha256": LOOP_ROOT / "analysis_seed_firewall.json",
        "benchmarkSha256": LOOP_ROOT / "benchmark_projection.json",
        "l28BranchesSha256": L28_ROOT / "branch_results.parquet",
        "l30BranchesSha256": L30_ROOT / "short_branch_results.parquet",
        "l31BranchesSha256": L31_ROOT / "branch_results.parquet",
        "l23ManifestSha256": L23_ROOT / "input_trajectory_manifest.parquet",
    }
    for key, path in locked_files.items():
        if sha256_file(path) != original_lock[key]:
            raise RuntimeError(f"L36 locked input changed before repair: {path}")
    diagnostic_path = CACHE_ROOT / "diagnostic_branches.parquet"
    if not diagnostic_path.is_file():
        raise RuntimeError("L36 failed-attempt diagnostic table is missing")
    diagnostic = pd.read_parquet(diagnostic_path)
    validation = compact_replay_validation(diagnostic)
    if not prior["unchanged"] or prior["aggregateSha256"] != original_lock[
        "priorAggregateSha256"
    ]:
        raise RuntimeError("L36 immutable prior changed before repair")
    maximum_absolute = float(validation["maximumScoreAbsoluteError"].max())
    minimum_absolute = float(validation["minimumScoreAbsoluteError"].max())
    maximum_relative = float(validation["maximumScoreRelativeError"].max())
    minimum_relative = float(validation["minimumScoreRelativeError"].max())
    maximum_ulp = int(validation["maximumScoreUlpError"].max())
    minimum_ulp = int(validation["minimumScoreUlpError"].max())
    exact_discrete = bool(
        validation[
            [
                "entryExact",
                "firstEntryExact",
                "molecularUpdatesExact",
                "fissionsExact",
                "selectedObservationsExact",
                "terminalExact",
                "pathExact",
            ]
        ]
        .all()
        .all()
    )
    failure_record = {
        "schema": "eidosoma.e01.s19_l36.failed_attempt.v1",
        "attempt": 1,
        "status": "STOPPED_AT_ORIGINAL_COMPACT_REPLAY_GATE",
        "outcomesOpened": True,
        "scientificAggregationReleased": False,
        "exception": "RuntimeError: L36 original-target compact replay failed",
        "diagnosis": "All discrete outcomes, clocks, paths and entry times agreed exactly; score extrema differed only through floating-point evaluation order.",
        "diagnosticBranchesSha256": sha256_file(diagnostic_path),
        "rowsCompared": len(validation),
        "maximumScoreBitExactRows": int(validation["maximumScoreBitExact"].sum()),
        "minimumScoreBitExactRows": int(validation["minimumScoreBitExact"].sum()),
        "maximumAbsoluteError": maximum_absolute,
        "minimumAbsoluteError": minimum_absolute,
        "maximumRelativeError": maximum_relative,
        "minimumRelativeError": minimum_relative,
        "maximumUlpError": maximum_ulp,
        "minimumUlpError": minimum_ulp,
        "allDiscreteValuesExact": exact_discrete,
        "recordedAtUtc": utc_now(),
    }
    amendment = {
        "schema": "eidosoma.e01.s19_l36.technical_amendment_lock.v1",
        "status": "LOCKED_VALUE_PRESERVING",
        "amendmentId": "L36-TA01-NUMERICAL-REPLAY-EQUIVALENCE",
        "originalHead": original_lock["head"],
        "head": head,
        "remote": remote,
        "runnerSha256": sha256_file(RUNNER_PATH),
        "coreSha256": sha256_file(CORE_PATH),
        "configSha256": sha256_file(CONFIG),
        "absoluteTolerance": REPLAY_ABSOLUTE_TOLERANCE,
        "relativeTolerance": REPLAY_RELATIVE_TOLERANCE,
        "maximumUlpError": REPLAY_MAXIMUM_ULP_ERROR,
        "finiteMasksMustMatch": True,
        "discreteValuesMustMatchExactly": True,
        "scientificValueChange": False,
        "scientificMethodChange": False,
        "thresholdChanged": False,
        "targetChanged": False,
        "branchStreamChanged": False,
        "diagnosticBranchesSha256": sha256_file(diagnostic_path),
        "observedDiagnostics": failure_record,
        "lockedAtUtc": utc_now(),
    }
    if (
        not exact_discrete
        or maximum_absolute > REPLAY_ABSOLUTE_TOLERANCE
        or minimum_absolute > REPLAY_ABSOLUTE_TOLERANCE
        or maximum_relative > REPLAY_RELATIVE_TOLERANCE
        or minimum_relative > REPLAY_RELATIVE_TOLERANCE
        or maximum_ulp > REPLAY_MAXIMUM_ULP_ERROR
        or minimum_ulp > REPLAY_MAXIMUM_ULP_ERROR
    ):
        raise RuntimeError("L36 failed attempt is not value-preserving under TA01")
    BASE.write_json(LOOP_ROOT / "failed_attempt_01.json", failure_record)
    BASE.write_json(LOOP_ROOT / "technical_amendment_lock.json", amendment)
    pd.DataFrame(
        [
            {
                "amendmentId": amendment["amendmentId"],
                "status": amendment["status"],
                "reason": failure_record["diagnosis"],
                "scientificValueChange": False,
                "scientificMethodChange": False,
                "head": head,
                "lockedAtUtc": amendment["lockedAtUtc"],
            }
        ]
    ).to_csv(LOOP_ROOT / "technical_amendment_ledger.csv", index=False)


def execute() -> None:
    started = time.perf_counter()
    started_cpu = time.process_time()
    lock = json.loads((LOOP_ROOT / "preoutcome_repository_lock.json").read_text())
    amendment_path = LOOP_ROOT / "technical_amendment_lock.json"
    amendment = json.loads(amendment_path.read_text()) if amendment_path.is_file() else None
    expected_head = amendment["head"] if amendment is not None else lock["head"]
    expected_runner = (
        amendment["runnerSha256"] if amendment is not None else lock["runnerSha256"]
    )
    expected_core = (
        amendment["coreSha256"] if amendment is not None else lock["coreSha256"]
    )
    if (
        git("rev-parse", "HEAD") != expected_head
        or git("rev-parse", "origin/eidosoma/groups/42") != expected_head
        or git("status", "--porcelain=v1")
    ):
        raise RuntimeError("L36 repository lock mismatch")
    prior = validate_immutable_prior()
    fixtures = fixture_results()
    locked_files = {
        "responsesSha256": LOOP_ROOT / "response_registry.parquet",
        "originalCoordinatesSha256": LOOP_ROOT / "original_target_coordinates.parquet",
        "referenceUnitsSha256": LOOP_ROOT / "reference_unit_registry.parquet",
        "referenceSeedsSha256": LOOP_ROOT / "reference_seed_manifest.parquet",
        "analysisSeedsSha256": LOOP_ROOT / "analysis_seed_manifest.parquet",
        "seedFirewallSha256": LOOP_ROOT / "seed_firewall.json",
        "analysisFirewallSha256": LOOP_ROOT / "analysis_seed_firewall.json",
        "benchmarkSha256": LOOP_ROOT / "benchmark_projection.json",
        "l28BranchesSha256": L28_ROOT / "branch_results.parquet",
        "l30BranchesSha256": L30_ROOT / "short_branch_results.parquet",
        "l31BranchesSha256": L31_ROOT / "branch_results.parquet",
        "l23ManifestSha256": L23_ROOT / "input_trajectory_manifest.parquet",
    }
    for key, path in locked_files.items():
        if sha256_file(path) != lock[key]:
            raise RuntimeError(f"L36 locked input changed: {path}")
    if (
        not prior["unchanged"]
        or prior["aggregateSha256"] != lock["priorAggregateSha256"]
        or not fixtures["passed"].all()
        or sha256_file(RUNNER_PATH) != expected_runner
        or sha256_file(CORE_PATH) != expected_core
        or (amendment is not None and amendment["status"] != "LOCKED_VALUE_PRESERVING")
    ):
        raise RuntimeError("L36 pre-execution validation failed")
    responses = pd.read_parquet(LOOP_ROOT / "response_registry.parquet")
    original_coordinates = pd.read_parquet(
        LOOP_ROOT / "original_target_coordinates.parquet"
    )
    units = pd.read_parquet(LOOP_ROOT / "reference_unit_registry.parquet")
    trajectory_manifest = pd.read_parquet(
        L23_ROOT / "input_trajectory_manifest.parquet"
    )
    base_payloads = L35.payloads(responses, original_coordinates, trajectory_manifest)
    if BUILD_ROOT.exists():
        shutil.rmtree(BUILD_ROOT)
    BUILD_ROOT.mkdir(parents=True)
    references = generate_references(units, REFERENCE_CACHE)
    target_summary, target_coordinates, target_comparisons = target_registries(
        responses, original_coordinates, references
    )
    target_summary = add_current_scores(
        target_summary, target_coordinates, base_payloads
    )
    scored_payloads = branch_payloads(base_payloads, target_coordinates)
    branches = rescore_branches(scored_payloads)
    compact_validation = compact_replay_validation(branches)
    states = state_committor_results(branches, target_summary)
    availability = target_availability_results(target_summary, target_comparisons)
    reliability = reliability_results(states)
    pairs = transfer_pairs(states)
    transfer, transfer_bootstrap = transfer_results(pairs)
    reliability_bootstrap = reliability_bootstrap_intervals(states)
    centroid_bootstrap = centroid_agreement_bootstrap(target_comparisons)
    gates, classifications, next_theme = scientific_gates(
        availability,
        transfer,
        reliability,
        reliability_bootstrap,
        centroid_bootstrap,
    )
    make_figures(
        references,
        target_comparisons,
        availability,
        states,
        reliability,
        transfer,
        gates,
    )
    for name in (
        "preregistration.yaml",
        "decision_record.md",
        "fixture_results.parquet",
        "benchmark_projection.json",
        "response_registry.parquet",
        "original_target_coordinates.parquet",
        "reference_unit_registry.parquet",
        "reference_seed_manifest.parquet",
        "seed_firewall.json",
        "analysis_seed_manifest.parquet",
        "analysis_seed_firewall.json",
        "source_grounding_registry.parquet",
        "immutable_prior_validation.json",
        "implementation_lock.json",
        "preoutcome_repository_lock.json",
    ):
        shutil.copy2(LOOP_ROOT / name, BUILD_ROOT / name)
    for name in (
        "failed_attempt_01.json",
        "technical_amendment_lock.json",
        "technical_amendment_ledger.csv",
    ):
        source = LOOP_ROOT / name
        if source.is_file():
            shutil.copy2(source, BUILD_ROOT / name)
    reference_artifact = references.drop(columns=["centroid"])
    BASE.write_parquet(
        BUILD_ROOT / "reference_trajectory_manifest.parquet", reference_artifact
    )
    BASE.write_parquet(BUILD_ROOT / "target_registry.parquet", target_summary)
    BASE.write_parquet(
        BUILD_ROOT / "target_coordinate_registry.parquet", target_coordinates
    )
    BASE.write_parquet(
        BUILD_ROOT / "target_centroid_comparison.parquet", target_comparisons
    )
    BASE.write_parquet(BUILD_ROOT / "branch_rescore_results.parquet", branches)
    BASE.write_parquet(
        BUILD_ROOT / "original_compact_replay_validation.parquet",
        compact_validation,
    )
    BASE.write_parquet(BUILD_ROOT / "state_committor_results.parquet", states)
    BASE.write_parquet(
        BUILD_ROOT / "target_availability_results.parquet", availability
    )
    BASE.write_parquet(BUILD_ROOT / "committor_reliability_results.parquet", reliability)
    BASE.write_parquet(BUILD_ROOT / "transfer_pair_results.parquet", pairs)
    BASE.write_parquet(BUILD_ROOT / "transfer_results.parquet", transfer)
    BASE.write_parquet(
        BUILD_ROOT / "transfer_bootstrap_manifest.parquet", transfer_bootstrap
    )
    BASE.write_parquet(
        BUILD_ROOT / "reliability_bootstrap_intervals.parquet",
        reliability_bootstrap,
    )
    BASE.write_parquet(
        BUILD_ROOT / "centroid_agreement_bootstrap.parquet", centroid_bootstrap
    )
    BASE.write_parquet(BUILD_ROOT / "scientific_gate_results.parquet", gates)
    BASE.write_json(
        BUILD_ROOT / "classification.json",
        {
            "schema": "eidosoma.e01.s19_l36.classification.v1",
            "classifications": classifications,
            "independentBasinStableAllGroups": bool(
                gates["independentBasinStable"].all()
            ),
            "originalShootingTransfersAllGroups": bool(
                gates["originalShootingTransfers"].all()
            ),
            "pastOnlySignalEstablished": False,
            "targetsCompletedLineageConditioned": True,
            "priorStatusesChanged": False,
        },
    )
    pd.DataFrame(
        columns=[
            "stage",
            "stateId",
            "candidateId",
            "matrixIndex",
            "referenceId",
            "branchFamily",
            "branchIndex",
            "exceptionClass",
            "exceptionMessage",
        ]
    ).to_csv(BUILD_ROOT / "failure_ledger.csv", index=False)

    # Full independent regeneration of both new reference lineages and every
    # rescored frozen branch stream.
    replay_references = generate_references(units, REGEN_CACHE)
    replay_summary, replay_coordinates, replay_comparisons = target_registries(
        responses, original_coordinates, replay_references
    )
    replay_summary = add_current_scores(
        replay_summary, replay_coordinates, base_payloads
    )
    replay_payloads = branch_payloads(base_payloads, replay_coordinates)
    replay_branches = rescore_branches(replay_payloads)
    reference_columns = [
        column
        for column in references.columns
        if column not in ("cachePath", "wallSeconds", "centroid")
    ]
    checks = {
        "referenceTrajectoryExact": frame_hash(references[reference_columns])
        == frame_hash(replay_references[reference_columns]),
        "targetSummaryExact": frame_hash(target_summary)
        == frame_hash(replay_summary),
        "targetCoordinateExact": frame_hash(target_coordinates)
        == frame_hash(replay_coordinates),
        "targetComparisonExact": frame_hash(target_comparisons)
        == frame_hash(replay_comparisons),
        "branchRescoreExact": frame_hash(branches) == frame_hash(replay_branches),
        "originalCompactExact": bool(compact_validation["allPassed"].all()),
        "referenceSeedFirewallPassed": json.loads(
            (LOOP_ROOT / "seed_firewall.json").read_text()
        )["status"]
        == "PASS",
        "analysisSeedFirewallPassed": json.loads(
            (LOOP_ROOT / "analysis_seed_firewall.json").read_text()
        )["status"]
        == "PASS",
        "fixturesPassed": bool(fixtures["passed"].all()),
        "immutablePriorPassed": prior["unchanged"],
        "noNewBranchStreams": True,
        "noReplacement": bool(~references["replacementAttempted"].any()),
        "technicalAmendmentValuePreserving": amendment is None
        or amendment["scientificValueChange"] is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"L36 regeneration validation failed: {checks}")
    BASE.write_json(
        BUILD_ROOT / "regeneration_validation.json",
        {
            "schema": "eidosoma.e01.s19_l36.regeneration_validation.v1",
            "status": "PASS",
            "checks": checks,
            "referenceFrameSha256": frame_hash(references[reference_columns]),
            "targetFrameSha256": frame_hash(target_summary),
            "branchFrameSha256": frame_hash(branches),
            "stateCommittorFrameSha256": frame_hash(states),
        },
    )
    runtime = {
        "schema": "eidosoma.e01.s19_l36.runtime.v1",
        "repositoryHead": git("rev-parse", "HEAD"),
        "workers": WORKERS,
        "numericalLibraryThreadsPerWorker": 1,
        "gpuHours": 0,
        "wallSeconds": time.perf_counter() - started,
        "controllerCpuHours": (time.process_time() - started_cpu) / 3600,
        "referenceTrajectoriesGenerated": len(references),
        "referenceTrajectoriesRegenerated": len(replay_references),
        "uniqueFrozenBranchStreamsRescored": 53_760,
        "newBranchStreams": 0,
        "technicalAmendmentApplied": amendment is not None,
        "replayAbsoluteTolerance": REPLAY_ABSOLUTE_TOLERANCE,
        "replayRelativeTolerance": REPLAY_RELATIVE_TOLERANCE,
        "replayMaximumUlpError": REPLAY_MAXIMUM_ULP_ERROR,
        "completedAtUtc": utc_now(),
    }
    BASE.write_json(BUILD_ROOT / "runtime_manifest.json", runtime)
    retained = sum(path.stat().st_size for path in BUILD_ROOT.rglob("*") if path.is_file())
    temporary = sum(path.stat().st_size for path in CACHE_ROOT.rglob("*") if path.is_file())
    storage = {
        "schema": "eidosoma.e01.s19_l36.storage_validation.v1",
        "retainedBytes": retained,
        "retainedGiBCeiling": 25,
        "temporaryBytes": temporary,
        "temporaryGiBCeiling": 75,
        "status": "PASS"
        if retained < 25 * 2**30 and temporary < 75 * 2**30
        else "FAIL",
    }
    if storage["status"] != "PASS":
        raise RuntimeError("L36 storage ceiling exceeded")
    BASE.write_json(BUILD_ROOT / "storage_validation.json", storage)
    report = report_text(
        availability,
        target_comparisons,
        reliability,
        transfer,
        gates,
        classifications,
        runtime,
        next_theme,
    )
    BASE.atomic_text(BUILD_ROOT / "research_step_full_results.md", report)
    BASE.atomic_text(BUILD_ROOT / "S19_L36_FULL_RESULTS.md", report)
    BASE.atomic_text(
        BUILD_ROOT / "loop_decision_summary.md",
        "# S19-L36 decision summary\n\n"
        + f"**Classification:** {', '.join(classifications)}\n\n"
        + f"**Independent basin stable in every group:** `{gates['independentBasinStable'].all()}`.\n\n"
        + f"**Next:** `{next_theme}`.\n",
    )
    BASE.write_json(BUILD_ROOT / "artifact_manifest.json", manifest_for(BUILD_ROOT))
    stage = LOOP_ROOT.with_name(".L36-promotion-stage")
    if stage.exists():
        shutil.rmtree(stage)
    shutil.copytree(BUILD_ROOT, stage)
    if LOOP_ROOT.exists():
        shutil.rmtree(LOOP_ROOT)
    os.replace(stage, LOOP_ROOT)
    shutil.rmtree(BUILD_ROOT)
    artifact_manifest = json.loads((LOOP_ROOT / "artifact_manifest.json").read_text())
    if any(
        sha256_file(LOOP_ROOT / item["path"]) != item["sha256"]
        for item in artifact_manifest["files"]
    ):
        raise RuntimeError("L36 artifact hash validation failed")
    append_ledgers(classifications, runtime["completedAtUtc"], next_theme)
    BASE.atomic_text(ARTIFACT_ROOT / "research_step_full_results.md", report)
    BASE.atomic_text(
        ARTIFACT_ROOT / "S19_CURRENT_HANDOFF.md",
        report.replace("# S19-L36", "# S19 current handoff — S19-L36", 1),
    )
    BASE.write_json(
        ARTIFACT_ROOT / "s19_status.json",
        {
            "schema": "eidosoma.e01.s19.status.v1",
            "status": "ACTIVE_AUTONOMOUS_SEQUENCE",
            "latestCompletedLoop": LOOP_ID,
            "latestClassification": classifications,
            "selectedDiscoveryLead": None,
            "nextAuthorizedLoop": "S19-L37",
            "authorizationUpperBound": "S19-L42",
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
    parser.add_argument("--prepare-technical-amendment", action="store_true")
    args = parser.parse_args()
    if args.prepare_lock and args.prepare_technical_amendment:
        parser.error("lock and amendment preparation are mutually exclusive")
    if args.prepare_lock:
        prepare_lock()
    elif args.prepare_technical_amendment:
        prepare_technical_amendment()
    else:
        execute()


if __name__ == "__main__":
    main()
