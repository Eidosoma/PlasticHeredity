"""Execute S19-L35 short-branch ensemble mechanism attribution."""

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

from e01_onset_discovery.branch_trace import simulate_branch_trace
from e01_onset_discovery.empirical_committor import RestoredState


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


L34 = _load_module(
    "e01_s19_l35_l34",
    REPO_ROOT / "scripts/e01/run_s19_l34_full_state_graph_committor.py",
)
L33 = L34.L33
L32 = L33.L32
L31 = L33.L31
L30 = L33.L30
L29 = L33.L29
L28 = L33.L28
BASE = L34.BASE

LOOP_ID = "S19-L35"
VERSION = "E01-S19-L35-SHORT-BRANCH-ENSEMBLE-MECHANISM-ATTRIBUTION-v1.0.0"
CANDIDATES = L28.CANDIDATES
COHORTS = ("L28_DEVELOPMENT", "L28_VALIDATION", "L31_CONFIRMATION")
EVALUATION_COHORTS = ("L28_VALIDATION", "L31_CONFIRMATION")
HORIZON = 8
BRANCHES = 64
HALF = 32
BOOTSTRAPS = 4096
ROOT_HEX = "f2313acc049372b13f340fbfca9bce50c7bb357f3063840626b628332e43bd20"
WORKERS = min(8, os.cpu_count() or 1)

PRIMARY_METRICS = (
    "cumulativeEntryFraction",
    "atRiskMeanTargetScore",
    "atRiskMeanTargetScoreChangeFromCurrent",
    "allBranchTargetScoreSd",
    "atRiskCompositionDispersion",
    "fissionFractionAtOffset",
    "atRiskMeanMass",
    "atRiskMeanJoinShareMaximum",
    "atRiskMeanGrossSampledEvents",
)
MECHANICAL_METRICS = (
    "atRiskCompositionDispersion",
    "fissionFractionAtOffset",
    "atRiskMeanMass",
    "atRiskMeanJoinShareMaximum",
    "atRiskMeanGrossSampledEvents",
)
BASIN_CONDITIONED_METRICS = (
    "atRiskMeanTargetScore",
    "atRiskMeanTargetScoreChangeFromCurrent",
    "allBranchTargetScoreSd",
)

ARTIFACT_ROOT = Path("/artifacts/research_steps/S19")
LOOP_ROOT = ARTIFACT_ROOT / "loops/L35"
L34_ROOT = ARTIFACT_ROOT / "loops/L34"
L31_ROOT = ARTIFACT_ROOT / "loops/L31"
L30_ROOT = ARTIFACT_ROOT / "loops/L30"
L23_ROOT = ARTIFACT_ROOT / "loops/L23"
CACHE_ROOT = Path("/cache/e01_s19_l35")
BUILD_ROOT = CACHE_ROOT / "build"
CONFIG = REPO_ROOT / "configs/e01/s19_l35_short_branch_mechanism_attribution.yaml"
RUNNER_PATH = Path(__file__)
CORE_PATH = REPO_ROOT / "src/e01_onset_discovery/branch_trace.py"


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
    prior = json.loads((L34_ROOT / "immutable_prior_validation.json").read_text())
    rows = list(prior["files"])
    manifest = json.loads((L34_ROOT / "artifact_manifest.json").read_text())
    rows.extend(
        {
            "path": str(L34_ROOT / item["path"]),
            "root": str(L34_ROOT),
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
        "schema": "eidosoma.e01.s19_l35.immutable_prior_validation.v1",
        "status": "PASS" if not failures else "FAIL",
        "unchanged": not failures,
        "fileCount": len(rows),
        "aggregateSha256": hashlib.sha256(
            "\n".join(f"{row['path']}\t{row['sha256']}" for row in rows).encode()
        ).hexdigest(),
        "l34ArtifactFileCount": manifest["fileCount"],
        "failures": failures,
        "files": rows,
    }


def response_registry() -> pd.DataFrame:
    result = L33.response_registry()
    if len(result) != 280 or result["stateId"].duplicated().any():
        raise RuntimeError("L35 response registry scope mismatch")
    return result


def replay_seed_manifest(responses: pd.DataFrame) -> pd.DataFrame:
    cohort = responses[["stateId", "evaluationCohort"]]
    l30 = pd.read_parquet(L30_ROOT / "short_branch_seed_manifest.parquet").merge(
        cohort, on="stateId", validate="many_to_one"
    )
    l30["sourceLoop"] = "L30"
    l31 = pd.read_parquet(L31_ROOT / "branch_seed_manifest.parquet")
    l31 = l31[l31["branchFamily"].eq("H8")].merge(
        cohort, on="stateId", validate="many_to_one"
    )
    l31["sourceLoop"] = "L31"
    shared = [
        "stateId",
        "candidateId",
        "matrixIndex",
        "landmark",
        "branchIndex",
        "branchHalf",
        "rootHex",
        "streamIdentitySha256",
        "eventDerivedSeed",
        "eventSeedMaterialSha256",
        "trimDerivedSeed",
        "trimSeedMaterialSha256",
        "fissionDerivedSeed",
        "fissionSeedMaterialSha256",
        "daughterDerivedSeed",
        "daughterSeedMaterialSha256",
        "evaluationCohort",
        "sourceLoop",
    ]
    result = pd.concat([l30[shared], l31[shared]], ignore_index=True).sort_values(
        ["evaluationCohort", "candidateId", "landmark", "matrixIndex", "branchIndex"]
    ).reset_index(drop=True)
    if (
        len(result) != 17_920
        or result.duplicated(["stateId", "branchIndex"]).any()
        or not (result.groupby("stateId").size() == BRANCHES).all()
        or not result["streamIdentitySha256"].is_unique
    ):
        raise RuntimeError("L35 replay seed scope mismatch")
    result["newScientificSeed"] = False
    result["replayOnly"] = True
    return result


def payloads(
    responses: pd.DataFrame, coordinates: pd.DataFrame, manifest: pd.DataFrame
) -> list[dict[str, Any]]:
    output = L29.state_payloads(
        responses, coordinates, manifest, reference_variant="ORIGINAL"
    )
    loop_map = {
        row.stateId: "L31"
        if row.evaluationCohort == "L31_CONFIRMATION"
        else "L30"
        for row in responses.itertuples(index=False)
    }
    for item in output:
        item["sourceLoop"] = loop_map[item["stateId"]]
        item["evaluationCohort"] = responses.set_index("stateId").loc[
            item["stateId"], "evaluationCohort"
        ]
    if len(output) != 280:
        raise RuntimeError("L35 payload scope mismatch")
    return output


def _stream_identities(payload: dict[str, Any], branch: int) -> dict[str, Any]:
    candidate = payload["candidateId"]
    matrix = int(payload["matrixIndex"])
    landmark = int(payload["landmark"])
    if payload["sourceLoop"] == "L30":
        return L30.branch_seeds(candidate, matrix, landmark, branch)
    return L31.stream_identities("H8", candidate, matrix, landmark, branch)


def _rngs(payload: dict[str, Any], branch: int) -> tuple[Any, Any, Any, Any]:
    identities = _stream_identities(payload, branch)
    if payload["sourceLoop"] == "L30":
        return tuple(
            L28.generator(identities[name])
            for name in (
                "propagator_event",
                "propagator_trim",
                "propagator_fission",
                "propagator_daughter",
            )
        )
    return tuple(
        L28.generator(identities[name])
        for name in ("event", "trim", "fission", "daughter")
    )


def _stream_hash(payload: dict[str, Any], branch: int) -> str:
    identities = _stream_identities(payload, branch)
    materials = [identity.seed_material_sha256 for identity in identities.values()]
    parts = [payload["stateId"], str(branch), *materials]
    if payload["sourceLoop"] == "L31":
        parts.insert(0, "H8")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def _composition_dispersion(states: list[tuple[int, ...]]) -> float:
    if len(states) < 2:
        return float("nan")
    values = np.asarray(states, dtype=np.float64)
    masses = values.sum(axis=1)
    valid = masses > 0
    values = values[valid] / masses[valid, None]
    if len(values) < 2:
        return float("nan")
    center = values.mean(axis=0)
    denominator = np.linalg.norm(values, axis=1) * np.linalg.norm(center)
    similarities = np.divide(
        values @ center,
        denominator,
        out=np.full(len(values), np.nan),
        where=denominator > 0,
    )
    return float(np.nanmean(1.0 - similarities))


def _safe_mean(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(array)) if np.isfinite(array).any() else float("nan")


def _safe_sd(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.nanstd(array, ddof=1)) if np.isfinite(array).sum() > 1 else float("nan")


def _safe_quantile(values: list[float], q: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.nanquantile(array, q)) if np.isfinite(array).any() else float("nan")


def _worker(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    matrix_index = int(payload["matrixIndex"])
    beta = L28.generate_beta(
        L28.derive_seed(
            L28.L23_ROOT_HEX, L28.L23_PHASE, "catalytic_matrix", matrix_index
        )
    )
    if L28.simulator_array_sha256(beta) != payload["betaSha256"]:
        raise RuntimeError(f"beta replay failure: {payload['stateId']}")
    restored = RestoredState(
        tuple(payload["state"]),
        payload["currentObservationKind"],
        int(payload["currentCompletedFissions"]),
        int(payload["currentGrowthGeneration"]),
        int(payload["currentGenerationLocalStep"]),
        int(payload["currentBatchStep"]),
    )
    target = np.asarray(payload["centroid"], dtype=np.float64)
    branch_records: list[dict[str, Any]] = []
    observations_by_branch: dict[int, tuple[Any, ...]] = {}
    eventual_entry: dict[int, bool] = {}
    for branch in range(BRANCHES):
        event_rng, trim_rng, fission_rng, daughter_rng = _rngs(payload, branch)
        result = simulate_branch_trace(
            restored=restored,
            beta=beta,
            definition=L28.definition(payload["candidateId"]),
            target_centroid=target,
            event_rng=event_rng,
            trim_rng=trim_rng,
            fission_rng=fission_rng,
            daughter_rng=daughter_rng,
            horizon=HORIZON,
            threshold=float(payload["targetThreshold"]),
        )
        compact = result.compact
        eventual_entry[branch] = compact.entered_basin
        observations_by_branch[branch] = result.observations
        branch_records.append(
            {
                "stateId": payload["stateId"],
                "evaluationCohort": payload["evaluationCohort"],
                "sourceLoop": payload["sourceLoop"],
                "candidateId": payload["candidateId"],
                "matrixIndex": matrix_index,
                "landmark": int(payload["landmark"]),
                "branchIndex": branch,
                "branchHalf": "A" if branch < HALF else "B",
                "streamIdentitySha256": _stream_hash(payload, branch),
                "enteredBasin": compact.entered_basin,
                "firstEntryOffsetOneBased": compact.first_entry_offset_one_based,
                "maximumTargetScore": compact.maximum_target_score,
                "minimumTargetScore": compact.minimum_target_score,
                "molecularUpdates": compact.molecular_updates,
                "fissions": compact.fissions,
                "selectedObservationsGenerated": compact.selected_observations_generated,
                "terminalStatus": compact.terminal_status,
                "pathSha256": compact.path_sha256,
                "finalStateSha256": compact.final_state_sha256,
            }
        )
    observation_records: list[dict[str, Any]] = []
    state_offset_records: list[dict[str, Any]] = []
    previous_scores = {
        branch: float(payload["targetCurrentScore"]) for branch in range(BRANCHES)
    }
    prefix_min_distance = {branch: float("inf") for branch in range(BRANCHES)}
    cumulative_fission = {branch: False for branch in range(BRANCHES)}
    for offset in range(1, HORIZON + 1):
        offset_rows = []
        for branch, observations in observations_by_branch.items():
            if len(observations) < offset:
                continue
            obs = observations[offset - 1]
            prior_entered = bool(obs.entered_at_or_before and not obs.entered_at_offset)
            at_risk_before = not prior_entered
            cumulative_fission[branch] = cumulative_fission[branch] or (
                obs.observation_kind == "post_fission"
            )
            distance = 1.0 - obs.target_score
            prefix_min_distance[branch] = min(prefix_min_distance[branch], distance)
            row = {
                "stateId": payload["stateId"],
                "evaluationCohort": payload["evaluationCohort"],
                "candidateId": payload["candidateId"],
                "matrixIndex": matrix_index,
                "landmark": int(payload["landmark"]),
                "branchIndex": branch,
                "branchHalf": "A" if branch < HALF else "B",
                "offset": offset,
                "observationKind": obs.observation_kind,
                "generation": obs.generation,
                "generationLocalStep": obs.generation_local_step,
                "targetScore": obs.target_score,
                "targetDistance": distance,
                "targetScoreChangeFromCurrent": obs.target_score
                - float(payload["targetCurrentScore"]),
                "targetScoreChangeFromPrevious": obs.target_score
                - previous_scores[branch],
                "minimumTargetDistanceThroughOffset": prefix_min_distance[branch],
                "ordinaryAdjacentH": obs.ordinary_adjacent_h,
                "enteredAtOffset": obs.entered_at_offset,
                "enteredAtOrBefore": obs.entered_at_or_before,
                "atRiskBeforeOffset": at_risk_before,
                "eventualEntryByH8": eventual_entry[branch],
                "mass": obs.mass,
                "diversity": obs.diversity,
                "entropy": obs.entropy,
                "concentration": obs.concentration,
                "joinShareMaximum": obs.join_share_maximum,
                "joinShareEntropy": obs.join_share_entropy,
                "lossShareMaximum": obs.loss_share_maximum,
                "lossShareEntropy": obs.loss_share_entropy,
                "boostMaximum": obs.boost_maximum,
                "boostSd": obs.boost_sd,
                "nonzeroReactionTypes": obs.nonzero_reaction_types,
                "grossSampledEvents": obs.gross_sampled_events,
                "overshoot": obs.overshoot,
                "exposure": obs.exposure,
                "cumulativeFission": cumulative_fission[branch],
                "state": obs.state,
            }
            offset_rows.append(row)
            observation_records.append({key: value for key, value in row.items() if key != "state"})
            previous_scores[branch] = obs.target_score
        at_risk = [row for row in offset_rows if row["atRiskBeforeOffset"]]
        entrants = [row for row in at_risk if row["eventualEntryByH8"]]
        nonentrants = [row for row in at_risk if not row["eventualEntryByH8"]]
        state_records = []
        for half_name, half_rows in (
            ("ALL", offset_rows),
            ("A", [row for row in offset_rows if row["branchHalf"] == "A"]),
            ("B", [row for row in offset_rows if row["branchHalf"] == "B"]),
        ):
            half_at_risk = [row for row in half_rows if row["atRiskBeforeOffset"]]
            denominator = BRANCHES if half_name == "ALL" else HALF
            values = lambda name, rows=half_at_risk: [float(row[name]) for row in rows]
            all_values = lambda name, rows=half_rows: [float(row[name]) for row in rows]
            state_records.append(
                {
                    "stateId": payload["stateId"],
                    "evaluationCohort": payload["evaluationCohort"],
                    "candidateId": payload["candidateId"],
                    "matrixIndex": matrix_index,
                    "landmark": int(payload["landmark"]),
                    "branchHalf": half_name,
                    "offset": offset,
                    "branchesObserved": len(half_rows),
                    "cumulativeEntryFraction": sum(
                        row["enteredAtOrBefore"] for row in half_rows
                    )
                    / denominator,
                    "newEntryFraction": sum(row["enteredAtOffset"] for row in half_rows)
                    / denominator,
                    "atRiskBeforeCount": len(half_at_risk),
                    "atRiskMeanTargetScore": _safe_mean(values("targetScore")),
                    "atRiskMeanTargetScoreChangeFromCurrent": _safe_mean(
                        values("targetScoreChangeFromCurrent")
                    ),
                    "atRiskMeanTargetScoreChangeFromPrevious": _safe_mean(
                        values("targetScoreChangeFromPrevious")
                    ),
                    "allBranchTargetScoreMean": _safe_mean(all_values("targetScore")),
                    "allBranchTargetScoreSd": _safe_sd(all_values("targetScore")),
                    "allBranchTargetScoreP10": _safe_quantile(all_values("targetScore"), 0.1),
                    "allBranchTargetScoreP50": _safe_quantile(all_values("targetScore"), 0.5),
                    "allBranchTargetScoreP90": _safe_quantile(all_values("targetScore"), 0.9),
                    "atRiskTargetScoreSd": _safe_sd(values("targetScore")),
                    "atRiskMeanMinimumDistanceReached": _safe_mean(
                        values("minimumTargetDistanceThroughOffset")
                    ),
                    "atRiskCompositionDispersion": _composition_dispersion(
                        [row["state"] for row in half_at_risk]
                    ),
                    "fissionFractionAtOffset": _safe_mean(
                        [row["observationKind"] == "post_fission" for row in half_rows]
                    ),
                    "cumulativeFissionFraction": _safe_mean(
                        all_values("cumulativeFission")
                    ),
                    "atRiskMeanMass": _safe_mean(values("mass")),
                    "atRiskMeanDiversity": _safe_mean(values("diversity")),
                    "atRiskMeanEntropy": _safe_mean(values("entropy")),
                    "atRiskMeanConcentration": _safe_mean(values("concentration")),
                    "atRiskMeanOrdinaryAdjacentH": _safe_mean(values("ordinaryAdjacentH")),
                    "atRiskMeanJoinShareMaximum": _safe_mean(values("joinShareMaximum")),
                    "atRiskMeanJoinShareEntropy": _safe_mean(values("joinShareEntropy")),
                    "atRiskMeanLossShareMaximum": _safe_mean(values("lossShareMaximum")),
                    "atRiskMeanBoostMaximum": _safe_mean(values("boostMaximum")),
                    "atRiskMeanBoostSd": _safe_mean(values("boostSd")),
                    "atRiskMeanNonzeroReactionTypes": _safe_mean(
                        values("nonzeroReactionTypes")
                    ),
                    "atRiskMeanGrossSampledEvents": _safe_mean(
                        values("grossSampledEvents")
                    ),
                }
            )
        if offset == 1:
            first_passage = [
                row["firstEntryOffsetOneBased"]
                for row in branch_records
                if row["firstEntryOffsetOneBased"] is not None
            ]
            for row in state_records:
                row["eventualEntryFraction"] = _safe_mean(
                    [eventual_entry[index] for index in range(BRANCHES) if row["branchHalf"] == "ALL" or ("A" if index < HALF else "B") == row["branchHalf"]]
                )
                row["conditionalFirstPassageMean"] = _safe_mean(first_passage)
        for row in state_records:
            row.setdefault("eventualEntryFraction", float("nan"))
            row.setdefault("conditionalFirstPassageMean", float("nan"))
        if len(offset_rows) != BRANCHES:
            raise RuntimeError(f"incomplete H8 trace: {payload['stateId']} offset {offset}")
        state_offset_records.extend(state_records)
    contrasts = []
    contrast_features = (
        "targetScore",
        "targetScoreChangeFromCurrent",
        "minimumTargetDistanceThroughOffset",
        "mass",
        "entropy",
        "concentration",
        "ordinaryAdjacentH",
        "joinShareMaximum",
        "joinShareEntropy",
        "boostMaximum",
        "boostSd",
        "grossSampledEvents",
    )
    for offset in range(1, HORIZON + 1):
        rows = [
            row
            for row in observation_records
            if row["offset"] == offset and row["atRiskBeforeOffset"]
        ]
        entrants = [row for row in rows if row["eventualEntryByH8"]]
        nonentrants = [row for row in rows if not row["eventualEntryByH8"]]
        for feature in contrast_features:
            entrant_mean = _safe_mean([float(row[feature]) for row in entrants])
            nonentrant_mean = _safe_mean([float(row[feature]) for row in nonentrants])
            contrasts.append(
                {
                    "stateId": payload["stateId"],
                    "evaluationCohort": payload["evaluationCohort"],
                    "candidateId": payload["candidateId"],
                    "matrixIndex": matrix_index,
                    "landmark": int(payload["landmark"]),
                    "offset": offset,
                    "featureId": feature,
                    "entrantBranches": len(entrants),
                    "nonentrantBranches": len(nonentrants),
                    "entrantMean": entrant_mean,
                    "nonentrantMean": nonentrant_mean,
                    "entrantMinusNonentrant": entrant_mean - nonentrant_mean
                    if np.isfinite(entrant_mean) and np.isfinite(nonentrant_mean)
                    else float("nan"),
                }
            )
    return {
        "branches": branch_records,
        "observations": observation_records,
        "stateOffsets": state_offset_records,
        "contrasts": contrasts,
    }


def execute_traces(payload_rows: list[dict[str, Any]]) -> tuple[pd.DataFrame, ...]:
    branches: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    offsets: list[dict[str, Any]] = []
    contrasts: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = {
            executor.submit(_worker, payload): payload["stateId"]
            for payload in payload_rows
        }
        for future in as_completed(futures):
            result = future.result()
            branches.extend(result["branches"])
            observations.extend(result["observations"])
            offsets.extend(result["stateOffsets"])
            contrasts.extend(result["contrasts"])
    branch_frame = pd.DataFrame(branches).sort_values(
        ["evaluationCohort", "candidateId", "landmark", "matrixIndex", "branchIndex"]
    ).reset_index(drop=True)
    observation_frame = pd.DataFrame(observations).sort_values(
        [
            "evaluationCohort",
            "candidateId",
            "landmark",
            "matrixIndex",
            "branchIndex",
            "offset",
        ]
    ).reset_index(drop=True)
    offset_frame = pd.DataFrame(offsets).sort_values(
        [
            "evaluationCohort",
            "candidateId",
            "landmark",
            "matrixIndex",
            "branchHalf",
            "offset",
        ]
    ).reset_index(drop=True)
    contrast_frame = pd.DataFrame(contrasts).sort_values(
        [
            "evaluationCohort",
            "candidateId",
            "landmark",
            "matrixIndex",
            "offset",
            "featureId",
        ]
    ).reset_index(drop=True)
    if (
        len(branch_frame) != 17_920
        or len(observation_frame) != 17_920 * HORIZON
        or len(offset_frame) != 280 * HORIZON * 3
        or len(contrast_frame) != 280 * HORIZON * 12
    ):
        raise RuntimeError("L35 trace output cardinality mismatch")
    return branch_frame, observation_frame, offset_frame, contrast_frame


def _normalize_nullable(value: Any) -> Any:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return value


def compact_replay_validation(branches: pd.DataFrame) -> pd.DataFrame:
    source_l30 = pd.read_parquet(L30_ROOT / "short_branch_results.parquet")
    source_l30 = source_l30[source_l30["referenceVariant"].eq("ORIGINAL")].copy()
    source_l30["sourceLoop"] = "L30"
    source_l30 = source_l30.rename(
        columns={"enteredBasinWithin8": "sourceEnteredBasin"}
    )
    source_l31 = pd.read_parquet(L31_ROOT / "branch_results.parquet")
    source_l31 = source_l31[
        source_l31["branchFamily"].eq("H8")
        & source_l31["referenceVariant"].eq("ORIGINAL")
    ].copy()
    source_l31["sourceLoop"] = "L31"
    source_l31 = source_l31.rename(columns={"enteredBasin": "sourceEnteredBasin"})
    columns = [
        "stateId",
        "branchIndex",
        "sourceLoop",
        "streamIdentitySha256",
        "sourceEnteredBasin",
        "firstEntryOffsetOneBased",
        "maximumTargetScore",
        "minimumTargetScore",
        "molecularUpdates",
        "fissions",
        "selectedObservationsGenerated",
        "terminalStatus",
        "pathSha256",
    ]
    source = pd.concat([source_l30[columns], source_l31[columns]], ignore_index=True)
    merged = branches.merge(
        source,
        on=["stateId", "branchIndex", "sourceLoop"],
        suffixes=("", "Source"),
        validate="one_to_one",
    )
    rows = []
    for row in merged.itertuples(index=False):
        scalar_fields = (
            "streamIdentitySha256",
            "firstEntryOffsetOneBased",
            "maximumTargetScore",
            "minimumTargetScore",
            "molecularUpdates",
            "fissions",
            "selectedObservationsGenerated",
            "terminalStatus",
            "pathSha256",
        )
        checks = {
            name: _normalize_nullable(getattr(row, name))
            == _normalize_nullable(getattr(row, f"{name}Source"))
            for name in scalar_fields
        }
        checks["enteredBasin"] = bool(row.enteredBasin) == bool(
            row.sourceEnteredBasin
        )
        rows.append(
            {
                "stateId": row.stateId,
                "sourceLoop": row.sourceLoop,
                "candidateId": row.candidateId,
                "matrixIndex": int(row.matrixIndex),
                "landmark": int(row.landmark),
                "branchIndex": int(row.branchIndex),
                "allCompactFieldsExact": all(checks.values()),
                "failedFields": json.dumps(
                    [name for name, passed in checks.items() if not passed]
                ),
            }
        )
    result = pd.DataFrame(rows)
    if len(result) != 17_920 or not result["allCompactFieldsExact"].all():
        raise RuntimeError("L35 exact compact branch replay failed")
    return result


def add_responses(offsets: pd.DataFrame, responses: pd.DataFrame) -> pd.DataFrame:
    response_columns = [
        "stateId",
        "successes",
        "qHat",
        "qHatHalfA",
        "qHatHalfB",
        "shortSuccesses",
        "q8",
        "q8HalfA",
        "q8HalfB",
        "targetCurrentScore",
        "currentMass",
        "targetComponentSize",
    ]
    return offsets.merge(
        responses[response_columns], on="stateId", validate="many_to_one"
    )


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3 or np.unique(left[valid]).size < 2 or np.unique(right[valid]).size < 2:
        return float("nan")
    return float(spearmanr(left[valid], right[valid]).statistic)


def correlation_results(offsets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    all_rows = offsets[offsets["branchHalf"].eq("ALL")]
    for (cohort, candidate, offset), group in all_rows.groupby(
        ["evaluationCohort", "candidateId", "offset"], sort=True
    ):
        for metric in PRIMARY_METRICS:
            values = group[metric].to_numpy(dtype=np.float64)
            rows.append(
                {
                    "evaluationCohort": cohort,
                    "candidateId": candidate,
                    "offset": int(offset),
                    "metricId": metric,
                    "metricClass": "TEACHER_REALIZED_ENTRY"
                    if metric == "cumulativeEntryFraction"
                    else (
                        "RETROSPECTIVE_BASIN_CONDITIONED"
                        if metric in BASIN_CONDITIONED_METRICS
                        else "BRANCH_PHYSICAL_MECHANISM"
                    ),
                    "states": len(group),
                    "definedStates": int(np.isfinite(values).sum()),
                    "spearmanH32": _spearman(
                        values, group["qHat"].to_numpy(dtype=np.float64)
                    ),
                    "spearmanH8": _spearman(
                        values, group["q8"].to_numpy(dtype=np.float64)
                    ),
                }
            )
    return pd.DataFrame(rows)


def _bootstrap_spearman(
    x: np.ndarray, y: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return np.full(BOOTSTRAPS, np.nan)
    indices = rng.integers(0, len(x), size=(BOOTSTRAPS, len(x)))
    output = np.full(BOOTSTRAPS, np.nan)
    for index, sample in enumerate(indices):
        left = x[sample]
        right = y[sample]
        if np.unique(left).size < 2 or np.unique(right).size < 2:
            continue
        left_rank = rankdata(left)
        right_rank = rankdata(right)
        output[index] = np.corrcoef(left_rank, right_rank)[0, 1]
    return output


def bootstrap_intervals(offsets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    all_rows = offsets[
        offsets["branchHalf"].eq("ALL")
        & offsets["evaluationCohort"].isin(EVALUATION_COHORTS)
    ]
    for (cohort, candidate, offset), group in all_rows.groupby(
        ["evaluationCohort", "candidateId", "offset"], sort=True
    ):
        for metric in PRIMARY_METRICS:
            values = group[metric].to_numpy(dtype=np.float64)
            response = group["qHat"].to_numpy(dtype=np.float64)
            distribution = _bootstrap_spearman(
                values,
                response,
                np.random.default_rng(
                    derived_seed("bootstrap", cohort, candidate, offset, metric)
                ),
            )
            finite = distribution[np.isfinite(distribution)]
            rows.append(
                {
                    "evaluationCohort": cohort,
                    "candidateId": candidate,
                    "offset": int(offset),
                    "metricId": metric,
                    "replicates": BOOTSTRAPS,
                    "definedReplicates": len(finite),
                    "spearmanLower95": float(np.quantile(finite, 0.025))
                    if len(finite)
                    else float("nan"),
                    "spearmanMedian": float(np.quantile(finite, 0.5))
                    if len(finite)
                    else float("nan"),
                    "spearmanUpper95": float(np.quantile(finite, 0.975))
                    if len(finite)
                    else float("nan"),
                    "distributionSha256": hashlib.sha256(
                        np.asarray(distribution, dtype="<f8").tobytes()
                    ).hexdigest(),
                }
            )
    return pd.DataFrame(rows)


def half_reliability(offsets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cohort, candidate, offset), group in offsets.groupby(
        ["evaluationCohort", "candidateId", "offset"], sort=True
    ):
        left = group[group["branchHalf"].eq("A")].set_index("stateId")
        right = group[group["branchHalf"].eq("B")].set_index("stateId")
        if set(left.index) != set(right.index):
            raise RuntimeError("L35 split-half state mismatch")
        right = right.loc[left.index]
        for metric in PRIMARY_METRICS:
            x = left[metric].to_numpy(dtype=np.float64)
            y = right[metric].to_numpy(dtype=np.float64)
            rows.append(
                {
                    "evaluationCohort": cohort,
                    "candidateId": candidate,
                    "offset": int(offset),
                    "metricId": metric,
                    "states": len(left),
                    "splitHalfSpearman": _spearman(x, y),
                    "halfAMean": _safe_mean(x.tolist()),
                    "halfBMean": _safe_mean(y.tolist()),
                }
            )
    return pd.DataFrame(rows)


def aggregate_contrasts(contrasts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in contrasts.groupby(
        ["evaluationCohort", "candidateId", "offset", "featureId"], sort=True
    ):
        cohort, candidate, offset, feature = keys
        values = group["entrantMinusNonentrant"].to_numpy(dtype=np.float64)
        finite = values[np.isfinite(values)]
        rng = np.random.default_rng(
            derived_seed("contrast_bootstrap", cohort, candidate, offset, feature)
        )
        if len(finite):
            samples = finite[rng.integers(0, len(finite), size=(BOOTSTRAPS, len(finite)))]
            distribution = np.mean(samples, axis=1)
        else:
            distribution = np.full(BOOTSTRAPS, np.nan)
        rows.append(
            {
                "evaluationCohort": cohort,
                "candidateId": candidate,
                "offset": int(offset),
                "featureId": feature,
                "definedStates": len(finite),
                "meanEntrantMinusNonentrant": float(np.mean(finite))
                if len(finite)
                else float("nan"),
                "lower95": float(np.nanquantile(distribution, 0.025))
                if len(finite)
                else float("nan"),
                "upper95": float(np.nanquantile(distribution, 0.975))
                if len(finite)
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def teacher_feature_attribution() -> pd.DataFrame:
    models = pd.read_parquet(L30_ROOT / "fitted_model_registry.parquet")
    models = models[
        models["referenceVariant"].eq("ORIGINAL")
        & models["modelId"].eq("EIGHT_STEP_PROPAGATOR_MOMENTS")
    ]
    rows = []
    for model in models.itertuples(index=False):
        names = json.loads(model.featureNames)
        coefficients = json.loads(model.coefficients)
        for name, coefficient in zip(names, coefficients, strict=True):
            rows.append(
                {
                    "candidateId": model.candidateId,
                    "modelId": model.modelId,
                    "featureName": name,
                    "standardizedCoefficient": float(coefficient),
                    "absoluteCoefficient": abs(float(coefficient)),
                    "sourceLoop": "L30",
                    "confirmationModelRefit": False,
                    "interpretation": "FROZEN_TEACHER_ATTRIBUTION_NOT_PRESENT_STATE_PROXY",
                }
            )
    result = pd.DataFrame(rows)
    result["absoluteRankWithinCandidate"] = result.groupby("candidateId")[
        "absoluteCoefficient"
    ].rank(method="first", ascending=False).astype(int)
    return result.sort_values(
        ["candidateId", "absoluteRankWithinCandidate"]
    ).reset_index(drop=True)


def mechanism_gate(
    correlations: pd.DataFrame, intervals: pd.DataFrame
) -> tuple[pd.DataFrame, list[str], str]:
    merged = correlations.merge(
        intervals,
        on=["evaluationCohort", "candidateId", "offset", "metricId"],
        validate="one_to_one",
    )
    rows = []
    for metric in PRIMARY_METRICS:
        for offset in range(1, HORIZON + 1):
            subset = merged[
                merged["metricId"].eq(metric)
                & merged["offset"].eq(offset)
                & merged["evaluationCohort"].isin(EVALUATION_COHORTS)
            ]
            complete = len(subset) == 4
            point = complete and bool((subset["spearmanH32"] > 0.5).all())
            lower = complete and bool((subset["spearmanLower95"] > 0.3).all())
            rows.append(
                {
                    "metricId": metric,
                    "metricClass": "TEACHER_REALIZED_ENTRY"
                    if metric == "cumulativeEntryFraction"
                    else (
                        "RETROSPECTIVE_BASIN_CONDITIONED"
                        if metric in BASIN_CONDITIONED_METRICS
                        else "BRANCH_PHYSICAL_MECHANISM"
                    ),
                    "offset": offset,
                    "fourGroupComplete": complete,
                    "allPointRanksAboveHalf": point,
                    "allLower95AbovePointThree": lower,
                    "beforeEntryLocalizationEligible": offset <= 4,
                    "commonGatePassed": bool(point and lower and offset <= 4),
                    "minimumSpearmanH32": float(subset["spearmanH32"].min())
                    if complete
                    else float("nan"),
                    "minimumLower95": float(subset["spearmanLower95"].min())
                    if complete
                    else float("nan"),
                }
            )
    gates = pd.DataFrame(rows)
    physical = gates[
        gates["metricId"].isin(MECHANICAL_METRICS) & gates["commonGatePassed"]
    ]
    teacher = gates[
        ~gates["metricId"].isin(MECHANICAL_METRICS) & gates["commonGatePassed"]
    ]
    if len(physical):
        classifications = [
            "BRANCH_SIGNAL_LOCALIZES_BEFORE_ENTRY",
            "MECHANICAL_BRANCH_FEATURE_PROXY_CANDIDATE_IDENTIFIED",
            "NOT_YET_PAST_ONLY",
        ]
        next_theme = "PRESENT_STATE_PROXY_FOR_LOCKED_BRANCH_MECHANISM"
    elif len(teacher):
        classifications = [
            "BRANCH_SIGNAL_EMERGES_AS_TARGET_APPROACH_OR_ENTRY",
            "SHOOTING_TEACHER_NOT_DISTILLED_TO_PREENTRY_MECHANISM",
            "NOT_PROMOTABLE_AS_CONFIRMED",
        ]
        next_theme = "TARGET_BASIN_TRANSFER_INDEPENDENT_REFERENCE_AUDIT"
    else:
        classifications = [
            "NO_UNIVERSAL_SHORT_BRANCH_MECHANISM",
            "SHOOTING_TEACHER_NOT_DISTILLED_TO_PREENTRY_MECHANISM",
            "NOT_PROMOTABLE_AS_CONFIRMED",
        ]
        next_theme = "TARGET_BASIN_TRANSFER_INDEPENDENT_REFERENCE_AUDIT"
    return gates, classifications, next_theme


def input_replay_validation(
    payload_rows: list[dict[str, Any]], responses: pd.DataFrame
) -> pd.DataFrame:
    response_index = responses.set_index("stateId")
    rows = []
    for payload in payload_rows:
        source = response_index.loc[payload["stateId"]]
        state = np.asarray(payload["state"], dtype=np.int64)
        target = np.asarray(payload["centroid"], dtype=np.float64)
        beta = L28.generate_beta(
            L28.derive_seed(
                L28.L23_ROOT_HEX,
                L28.L23_PHASE,
                "catalytic_matrix",
                int(payload["matrixIndex"]),
            )
        )
        checks = {
            "stateIdentityPassed": L28.array_sha256(state)
            == source.currentStateSha256,
            "massPassed": int(state.sum()) == int(source.currentMass),
            "targetIdentityPassed": L28.array_sha256(target)
            == source.targetCentroidSha256,
            "betaIdentityPassed": L28.simulator_array_sha256(beta)
            == source.betaSha256,
            "definitionIdentityPassed": L28.definition(
                payload["candidateId"]
            ).identity
            == source.simulatorDefinition,
            "clockIdentityPassed": int(payload["currentSelectedIndex"])
            == int(source.currentSelectedIndex)
            and int(payload["currentRawObservationIndex"])
            == int(source.currentRawObservationIndex),
            "responseIdentityPassed": float(payload["qHat"])
            == float(source.qHat)
            and float(payload["q8"]) == float(source.q8),
            "atRiskPassed": not bool(source.targetCurrentLabel),
        }
        rows.append(
            {
                "stateId": payload["stateId"],
                "evaluationCohort": payload["evaluationCohort"],
                "candidateId": payload["candidateId"],
                "matrixIndex": int(payload["matrixIndex"]),
                "landmark": int(payload["landmark"]),
                **checks,
                "allPassed": all(checks.values()),
            }
        )
    result = pd.DataFrame(rows)
    if len(result) != 280 or not result["allPassed"].all():
        raise RuntimeError("L35 input replay validation failed")
    return result


def replay_seed_validation(
    manifest: pd.DataFrame, payload_rows: list[dict[str, Any]]
) -> pd.DataFrame:
    payload_index = {payload["stateId"]: payload for payload in payload_rows}
    rows = []
    for row in manifest.itertuples(index=False):
        payload = payload_index[row.stateId]
        identities = _stream_identities(payload, int(row.branchIndex))
        tokens = (
            ("propagator_event", "propagator_trim", "propagator_fission", "propagator_daughter")
            if payload["sourceLoop"] == "L30"
            else ("event", "trim", "fission", "daughter")
        )
        generated = list(identities.values())
        checks = {
            "streamIdentityPassed": _stream_hash(payload, int(row.branchIndex))
            == row.streamIdentitySha256,
            "derivedSeedPassed": all(
                str(identity.derived_seed) == str(getattr(row, f"{name.replace('propagator_', '')}DerivedSeed"))
                for name, identity in zip(tokens, generated, strict=True)
            ),
            "seedMaterialPassed": all(
                identity.seed_material_sha256
                == getattr(row, f"{name.replace('propagator_', '')}SeedMaterialSha256")
                for name, identity in zip(tokens, generated, strict=True)
            ),
            "replayOnly": bool(row.replayOnly and not row.newScientificSeed),
        }
        rows.append(
            {
                "stateId": row.stateId,
                "sourceLoop": row.sourceLoop,
                "branchIndex": int(row.branchIndex),
                **checks,
                "allPassed": all(checks.values()),
            }
        )
    result = pd.DataFrame(rows)
    if len(result) != 17_920 or not result["allPassed"].all():
        raise RuntimeError("L35 replay seed validation failed")
    return result


def fixture_results() -> pd.DataFrame:
    rng = np.random.default_rng(derived_seed("fixture"))
    state = rng.poisson(1.1, size=100).astype(np.int64)
    state[0] += 1
    beta = np.exp(rng.normal(-3.0, 0.7, size=(100, 100)))
    target = rng.random(100)
    target /= target.sum()
    restored = RestoredState(tuple(map(int, state)), "molecular_update", 4, 5, 6, 23)
    seeds = [derived_seed("fixture", index) for index in range(4)]

    def traced():
        return simulate_branch_trace(
            restored=restored,
            beta=beta,
            definition=L28.definition(CANDIDATES[0]),
            target_centroid=target,
            event_rng=np.random.default_rng(seeds[0]),
            trim_rng=np.random.default_rng(seeds[1]),
            fission_rng=np.random.default_rng(seeds[2]),
            daughter_rng=np.random.default_rng(seeds[3]),
            horizon=HORIZON,
        )

    first = traced()
    second = traced()
    trace_replay = first.compact == second.compact and len(first.observations) == len(
        second.observations
    )
    if trace_replay:
        for left, right in zip(first.observations, second.observations, strict=True):
            for field in left.__dataclass_fields__:
                left_value = getattr(left, field)
                right_value = getattr(right, field)
                if isinstance(left_value, float) and np.isnan(left_value):
                    trace_replay &= isinstance(right_value, float) and np.isnan(
                        right_value
                    )
                else:
                    trace_replay &= left_value == right_value
    compact = L28.simulate_branch(
        restored=restored,
        beta=beta,
        definition=L28.definition(CANDIDATES[0]),
        target_centroid=target,
        event_rng=np.random.default_rng(seeds[0]),
        trim_rng=np.random.default_rng(seeds[1]),
        fission_rng=np.random.default_rng(seeds[2]),
        daughter_rng=np.random.default_rng(seeds[3]),
        horizon=HORIZON,
    )
    dispersion = _composition_dispersion([row.state for row in first.observations])
    rows = [
        {
            "fixtureId": "TRACE_COMPACT_EXACT_EQUIVALENCE",
            "passed": first.compact == compact,
            "details": first.compact.path_sha256,
        },
        {
            "fixtureId": "TRACE_EXACT_REPLAY",
            "passed": trace_replay,
            "details": str(len(first.observations)),
        },
        {
            "fixtureId": "EIGHT_OFFSETS",
            "passed": len(first.observations) == HORIZON
            and [row.offset for row in first.observations] == list(range(1, 9)),
            "details": "offsets 1..8",
        },
        {
            "fixtureId": "FINITE_PHYSICAL_SUMMARIES",
            "passed": all(
                np.isfinite(row.mass)
                and np.isfinite(row.entropy)
                and np.isfinite(row.join_share_maximum)
                for row in first.observations
            ),
            "details": "mass, entropy, propensity concentration",
        },
        {
            "fixtureId": "COMPOSITION_DISPERSION_NONNEGATIVE",
            "passed": np.isfinite(dispersion) and dispersion >= -1e-15,
            "details": str(dispersion),
        },
        {
            "fixtureId": "PRIMARY_METRICS_FIXED",
            "passed": len(PRIMARY_METRICS) == 9
            and len(MECHANICAL_METRICS) == 5,
            "details": json.dumps(PRIMARY_METRICS),
        },
        {
            "fixtureId": "BOOTSTRAP_SCOPE_FIXED",
            "passed": BOOTSTRAPS == 4096,
            "details": "catalytic-matrix bootstrap",
        },
    ]
    return pd.DataFrame(rows)


def benchmark_projection(payload_rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [
        next(row for row in payload_rows if row["candidateId"] == candidate)
        for candidate in CANDIDATES
    ]
    durations = []
    for payload in selected:
        started = time.perf_counter()
        result = _worker(payload)
        durations.append(time.perf_counter() - started)
        if len(result["branches"]) != BRANCHES:
            raise RuntimeError("L35 benchmark branch cardinality failure")
    projected_wall_seconds = max(durations) * len(payload_rows) / WORKERS * 2.25
    projected_cpu_hours = max(durations) * len(payload_rows) * 2.25 / 3600
    return {
        "schema": "eidosoma.e01.s19_l35.benchmark_projection.v1",
        "status": "PASS"
        if projected_wall_seconds <= 64.8 * 3600 and projected_cpu_hours <= 90
        else "STOP_BEFORE_FULL_REPLAY",
        "benchmarkStates": len(selected),
        "branchesPerState": BRANCHES,
        "durationsSeconds": durations,
        "projectedWallSecondsIncludingFullRegenerationAndFinalization": projected_wall_seconds,
        "projectedCpuHoursIncludingFullRegenerationAndFinalization": projected_cpu_hours,
        "newScientificOutcomes": 0,
    }


def analysis_seed_manifest() -> pd.DataFrame:
    rows = []
    parts_list: list[tuple[object, ...]] = [("fixture",)]
    parts_list.extend(("fixture", index) for index in range(4))
    for cohort in EVALUATION_COHORTS:
        for candidate in CANDIDATES:
            for offset in range(1, HORIZON + 1):
                for metric in PRIMARY_METRICS:
                    parts_list.append(("bootstrap", cohort, candidate, offset, metric))
    for cohort in COHORTS:
        for candidate in CANDIDATES:
            for offset in range(1, HORIZON + 1):
                for feature in (
                    "targetScore",
                    "targetScoreChangeFromCurrent",
                    "minimumTargetDistanceThroughOffset",
                    "mass",
                    "entropy",
                    "concentration",
                    "ordinaryAdjacentH",
                    "joinShareMaximum",
                    "joinShareEntropy",
                    "boostMaximum",
                    "boostSd",
                    "grossSampledEvents",
                ):
                    parts_list.append(
                        ("contrast_bootstrap", cohort, candidate, offset, feature)
                    )
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
    if result["derivedSeed"].duplicated().any() or result[
        "seedMaterialSha256"
    ].duplicated().any():
        raise RuntimeError("L35 analysis seed collision")
    return result


def analysis_seed_firewall(seeds: pd.DataFrame) -> dict[str, Any]:
    prior_material: set[str] = set()
    prior_derived: set[str] = set()
    for path in ARTIFACT_ROOT.glob("loops/L*/**/*seed*manifest*.parquet"):
        if "/L35/" in str(path):
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
    material_overlap = sorted(set(seeds["seedMaterialSha256"]) & prior_material)
    derived_overlap = sorted(set(seeds["derivedSeed"]) & prior_derived)
    return {
        "schema": "eidosoma.e01.s19_l35.analysis_seed_firewall.v1",
        "status": "PASS" if not material_overlap and not derived_overlap else "FAIL",
        "analysisSeedCount": len(seeds),
        "scientificTrajectorySeedCount": 0,
        "materialOverlapCount": len(material_overlap),
        "derivedOverlapCount": len(derived_overlap),
        "materialOverlaps": material_overlap,
        "derivedOverlaps": derived_overlap,
    }


def source_grounding_registry() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sourceId": "REVIEWER_TEACHER_SIGNAL_DIRECTION",
                "evidenceClass": "HUMAN_REVIEW_DIRECTION",
                "finding": "Locate what changes inside successful short branches before starting another unrelated representation family.",
                "frozenUse": "offset-resolved entrant/nonentrant and ensemble-mechanism audit",
            },
            {
                "sourceId": "L30_ESTABLISHED_H8_COORDINATE",
                "evidenceClass": "DIRECT_FROZEN_E01_RESULT",
                "finding": "64-branch H8 propagator predicted H32 committor on held-out L28 matrices in both candidates.",
                "frozenUse": "teacher ensemble and frozen coefficient attribution",
            },
            {
                "sourceId": "L31_UNTOUCHED_H8_CONFIRMATION",
                "evidenceClass": "DIRECT_FROZEN_E01_RESULT",
                "finding": "The fixed L30 H8 coordinate independently transferred to 80 untouched matrix states.",
                "frozenUse": "second evaluation cohort and exact stream replay",
            },
            {
                "sourceId": "L34_FULL_STATE_GRAPH_NON_SUPPORT",
                "evidenceClass": "DIRECT_FROZEN_E01_RESULT",
                "finding": "One fixed compact current-state/beta graph signature did not recover the teacher across both cohorts and candidates.",
                "frozenUse": "motivation to inspect teacher mechanics rather than add a broad feature family",
            },
        ]
    )


def make_figures(
    offsets: pd.DataFrame,
    correlations: pd.DataFrame,
    reliability: pd.DataFrame,
    contrasts: pd.DataFrame,
    attribution: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    root = BUILD_ROOT / "figures"
    root.mkdir(parents=True, exist_ok=True)

    def save(name: str) -> None:
        plt.tight_layout()
        plt.savefig(root / name, dpi=180)
        plt.close()

    all_rows = offsets[offsets["branchHalf"].eq("ALL")]
    plt.figure(figsize=(10, 6))
    for (candidate, cohort), group in all_rows.groupby(
        ["candidateId", "evaluationCohort"], sort=True
    ):
        means = group.groupby("offset")["cumulativeEntryFraction"].mean()
        plt.plot(means.index, means.values, marker="o", label=f"{candidate[-2:]} {cohort}")
    plt.xlabel("Short-branch selected-clock offset")
    plt.ylabel("Mean cumulative basin-entry fraction")
    plt.legend(fontsize=7)
    save("01_cumulative_entry_by_offset.png")

    evaluation = correlations[
        correlations["evaluationCohort"].isin(EVALUATION_COHORTS)
    ]
    physical = evaluation[evaluation["metricId"].isin(MECHANICAL_METRICS)]
    _, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    for axis, ((cohort, candidate), group) in zip(
        axes.flat,
        physical.groupby(["evaluationCohort", "candidateId"], sort=True),
        strict=True,
    ):
        for metric, rows in group.groupby("metricId", sort=True):
            axis.plot(rows["offset"], rows["spearmanH32"], marker=".", label=metric)
        axis.axhline(0.5, color="black", linestyle="--", linewidth=1)
        axis.set_title(f"{cohort} / {candidate}", fontsize=8)
        axis.set_xlabel("offset")
        axis.set_ylabel("Spearman with H32 q-hat")
    axes[0, 0].legend(fontsize=5)
    save("02_physical_mechanism_rank_trajectories.png")

    subset = reliability[
        reliability["evaluationCohort"].isin(EVALUATION_COHORTS)
        & reliability["metricId"].isin(PRIMARY_METRICS)
    ]
    subset.pivot_table(
        index=["metricId", "offset"],
        columns=["evaluationCohort", "candidateId"],
        values="splitHalfSpearman",
    ).plot(figsize=(13, 6))
    plt.axhline(0.5, color="black", linestyle="--", linewidth=1)
    plt.ylabel("Independent 32/32 branch-half Spearman")
    save("03_branch_half_reliability.png")

    selected_contrasts = contrasts[
        contrasts["evaluationCohort"].isin(EVALUATION_COHORTS)
        & contrasts["featureId"].isin(
            ["targetScore", "mass", "joinShareMaximum", "grossSampledEvents"]
        )
    ]
    _, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    for axis, (feature, group) in zip(
        axes.flat, selected_contrasts.groupby("featureId", sort=True), strict=True
    ):
        for (cohort, candidate), rows in group.groupby(
            ["evaluationCohort", "candidateId"], sort=True
        ):
            axis.plot(
                rows["offset"],
                rows["meanEntrantMinusNonentrant"],
                marker="o",
                label=f"{cohort}/{candidate[-2:]}",
            )
        axis.axhline(0, color="black", linewidth=1)
        axis.set_title(feature)
        axis.set_xlabel("offset")
    axes[0, 0].legend(fontsize=5)
    save("04_entrant_nonentrant_divergence.png")

    _, axes = plt.subplots(1, 2, figsize=(13, 6))
    for axis, (candidate, group) in zip(
        axes, attribution.groupby("candidateId", sort=True), strict=True
    ):
        group = group.sort_values("absoluteCoefficient")
        axis.barh(group["featureName"], group["absoluteCoefficient"])
        axis.set_title(candidate)
        axis.tick_params(axis="y", labelsize=7)
    save("05_frozen_teacher_feature_attribution.png")

    matrix = gates.pivot(index="metricId", columns="offset", values="commonGatePassed")
    plt.figure(figsize=(11, 6))
    plt.imshow(matrix.astype(float), vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    plt.xticks(range(HORIZON), range(1, HORIZON + 1))
    plt.yticks(range(len(matrix)), matrix.index, fontsize=7)
    plt.xlabel("offset")
    plt.colorbar(ticks=[0, 1])
    save("06_common_mechanism_gate_matrix.png")


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
        "schema": "eidosoma.e01.s19_l35.artifact_manifest.v1",
        "root": str(root),
        "fileCount": len(files),
        "totalBytes": sum(item["bytes"] for item in files),
        "files": files,
    }


def append_ledgers(
    classifications: list[str], timestamp: str, next_theme: str, solution: bool
) -> None:
    ledger_path = ARTIFACT_ROOT / "self_improvement_ledger.parquet"
    ledger = pd.read_parquet(ledger_path)
    sequence = int(ledger["ledgerSequence"].max()) + 1
    additions = [
        {
            "appendOnly": True,
            "beliefBeforeLoop": "The confirmed H8 propagator contains information absent from tested current/prefix summaries.",
            "failureOrAmbiguityTargeted": "Which branch-scale physical mechanism first carries that information.",
            "informationGainRationale": "Replay existing short branches observation by observation before adding another representation family.",
            "learned": "L35 exact-trace and fixed offset/mechanism contract frozen before full aggregation.",
            "ledgerSequence": sequence,
            "loopId": LOOP_ID,
            "motivatingEvidence": "Reviewer teacher-signal proposal plus L30/L31 support and L32-L34 non-support.",
            "proposedNextTest": "Exact H8 replay and offset-resolved physical attribution.",
            "recordPhase": "PRE_LOOP_METHOD_LOCK",
            "remainingPlausibleHypotheses": "Hidden phase/flux mechanism, target-conditioned approach, or irreducible stochastic shooting information.",
            "selectedHypotheses": "Five locked physical branch mechanisms plus separated target/entry teacher diagnostics.",
            "timestampUtc": timestamp,
            "weakenedHypotheses": "Another broad static feature tournament is the next informative action.",
        },
        {
            "appendOnly": True,
            "beliefBeforeLoop": "A transferable mechanism should separate before realized entry in both candidates and independent cohorts.",
            "failureOrAmbiguityTargeted": "Physical mechanism versus target-entry tautology.",
            "informationGainRationale": "Common offset/metric gates prevent favorable post hoc localization.",
            "learned": ";".join(classifications),
            "ledgerSequence": sequence + 1,
            "loopId": LOOP_ID,
            "motivatingEvidence": "Complete L35 exact replay result.",
            "proposedNextTest": next_theme,
            "recordPhase": "POST_LOOP_SOLUTION_HUMAN_REVIEW"
            if solution
            else "POST_LOOP_RESULT_AUTONOMOUS_CONTINUATION",
            "remainingPlausibleHypotheses": next_theme,
            "selectedHypotheses": "Fixed H8 teacher-mechanism audit.",
            "timestampUtc": timestamp,
            "weakenedHypotheses": "A universal early physical mechanism is visible in the registered H8 summaries"
            if not solution
            else "The branch signal appears only at entry.",
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
        + f"\n\n## {LOOP_ID} — short-branch mechanism attribution\n\n"
        + f"- **Learned:** {', '.join(classifications)}.\n"
        + f"- **Next:** {next_theme}.\n",
    )
    candidates_path = ARTIFACT_ROOT / "candidate_registry.parquet"
    candidates = pd.read_parquet(candidates_path)
    row = {
        "branchCount": len(PRIMARY_METRICS),
        "bundleId": "L35_SHORT_BRANCH_MECHANISM_ATTRIBUTION",
        "candidateId": "S19-L35-FROZEN-H8-TEACHER-MECHANISMS",
        "candidateSpecificSuccess": 0,
        "completedFitLeakage": 1,
        "computeEfficiency": 5,
        "crossCandidateDiscriminability": 5,
        "deterministicHReuse": 0,
        "explanatoryLeverage": 5,
        "frozenRank": 1,
        "independenceFromPriorOutcomeSelection": 4,
        "outcomeGuidedThresholdSelection": 0,
        "paperFingerprintSpecificity": 0,
        "proposedSpecification": "exact frozen H8 replay at offsets 1-8; separate entry/target teacher from five physical mechanisms",
        "rankingScore": 29.0,
        "registryOrder": int(candidates["registryOrder"].max()) + 1,
        "selected": True,
        "selectionReason": "REVIEWER_TEACHER_SIGNAL_MECHANISM_ATTRIBUTION",
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
            "finding": f"{source.finding}; L35 use: {source.frozenUse}",
            "licenseStatus": "WORKSPACE_OR_HUMAN_DIRECTION",
            "redistributionStatus": "INTERNAL_EVIDENCE_ONLY",
            "repositoryIdentity": None,
            "retainedPath": None,
            "retrievalDate": timestamp[:10],
            "sha256": None,
            "sourceId": f"L35_{source.sourceId}",
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
            "status": "COMPLETE_SOLUTION_BOUNDARY"
            if solution
            else "COMPLETE_AUTONOMOUS_CONTINUATION_AUTHORIZED",
            "authorized": True,
            "completed": True,
            "outcomeAccessed": True,
            "humanReviewRequiredAfter": solution,
            "classification": classifications,
            "selectedDiscoveryLead": "LOCKED_BRANCH_MECHANISM"
            if solution
            else None,
            "newMatrices": 0,
            "newTrajectories": 0,
            "newBranchStreams": 0,
            "replayedBranchStreams": 17_920,
            "nextStepActive": not solution,
        }
    )
    registry["proposedNextLoopTheme"] = next_theme
    registry["proposedNextLoopActive"] = not solution
    BASE.atomic_text(registry_path, yaml.safe_dump(registry, sort_keys=False))
    history_path = ARTIFACT_ROOT / "human_review_history.json"
    history = json.loads(history_path.read_text())
    history["history"].append(
        {
            "decision": "S19_L35_SOLUTION_HUMAN_REVIEW"
            if solution
            else "S19_L35_COMPLETE_AUTONOMOUS_CONTINUATION",
            "loopId": LOOP_ID,
            "nextLoopAuthorized": not solution,
            "recordedAtUtc": timestamp,
            "result": classifications,
            "s20Activated": False,
            "scope": VERSION,
            "source": "locked_execution_result",
        }
    )
    history["pendingDecision"] = (
        "HUMAN_REVIEW_REQUIRED_AFTER_EARLY_SOLUTION"
        if solution
        else "NONE_AUTONOMOUS_SEQUENCE_ACTIVE_THROUGH_L42"
    )
    BASE.write_json(history_path, history)


def report_text(
    correlations: pd.DataFrame,
    intervals: pd.DataFrame,
    reliability: pd.DataFrame,
    contrasts: pd.DataFrame,
    attribution: pd.DataFrame,
    gates: pd.DataFrame,
    classifications: list[str],
    runtime: dict[str, Any],
    next_theme: str,
) -> str:
    evaluation = correlations[
        correlations["evaluationCohort"].isin(EVALUATION_COHORTS)
    ].merge(
        intervals,
        on=["evaluationCohort", "candidateId", "offset", "metricId"],
        how="left",
    )
    best_rows = (
        evaluation.sort_values("spearmanH32", ascending=False)
        .groupby(["evaluationCohort", "candidateId"], as_index=False)
        .head(5)
    )
    physical_best = (
        evaluation[evaluation["metricId"].isin(MECHANICAL_METRICS)]
        .sort_values("spearmanH32", ascending=False)
        .groupby(["evaluationCohort", "candidateId"], as_index=False)
        .head(1)
    )
    reliable = reliability[
        reliability["evaluationCohort"].isin(EVALUATION_COHORTS)
    ].sort_values("splitHalfSpearman", ascending=False).head(12)
    early_contrasts = contrasts[
        contrasts["evaluationCohort"].isin(EVALUATION_COHORTS)
        & contrasts["offset"].le(4)
    ].sort_values("meanEntrantMinusNonentrant", key=lambda x: x.abs(), ascending=False).head(16)
    attr = attribution[
        attribution["absoluteRankWithinCandidate"].le(5)
    ][
        [
            "candidateId",
            "absoluteRankWithinCandidate",
            "featureName",
            "standardizedCoefficient",
        ]
    ]
    return f"""# S19-L35 — Short-Branch Ensemble Mechanism Attribution

## Chief/human handoff

- **Step:** `{VERSION}`
- **Status:** complete under the authorized L19–L42 sequence.
- **Classifications:** {", ".join(f"`{value}`" for value in classifications)}
- **Validation:** exact replay of 17,920 frozen H8 streams and 143,360 generated observations; exact compact path, entry, terminal and stream identities; exact 280 state/beta/clock/target/q inputs; deterministic full regeneration; independent branch halves; 4,096 catalytic-matrix bootstraps; immutable, runtime, storage and artifact hashes.
- **Recommended next action:** `{next_theme}`.

## Frozen question

What information appears during the successful eight-observation stochastic propagation that the tested present-state and observed-prefix summaries discarded? L35 does not create another predictor. It replays the already established L30/L31 teacher branches and records every intermediate physical update. Cumulative entry and similarity to the completed-run basin are explicitly separated from five branch-physical summaries. A future-branch measurement is not a past-observable biomarker.

## Inputs and method

- 200 L28 states (development and validation) and 80 untouched L31 confirmation states.
- Exactly 64 existing H8 continuations per state, split prospectively into two 32-branch halves.
- Exactly eight selected-clock observations per continuation; zero new branch streams, matrices, trajectories, targets, thresholds or simulator settings.
- Candidate 2 and candidate 3 and all three cohorts remain separate.
- Nine registered state-offset metrics; the physical solution gate uses exactly five metrics, the same metric and offset no later than offset four, and all four candidate/evaluation-cohort groups.
- H32 empirical committor is the response; catalytic matrix/state is the independent higher-level unit.

## Main result

The common physical-mechanism gate was `{'PASS' if gates[gates['metricId'].isin(MECHANICAL_METRICS)]['commonGatePassed'].any() else 'FAIL'}`. The strongest registered row in each evaluation group was:

{best_rows[['evaluationCohort','candidateId','offset','metricId','spearmanH32','spearmanLower95']].to_markdown(index=False)}

The strongest *physical* row in each group was:

{physical_best[['evaluationCohort','candidateId','offset','metricId','spearmanH32','spearmanLower95']].to_markdown(index=False)}

The locked common-gate matrix is:

{gates[gates['offset'].le(4)][['metricId','metricClass','offset','minimumSpearmanH32','minimumLower95','commonGatePassed']].to_markdown(index=False)}

## When branches separate

Entrant-versus-nonentrant branches were compared only while still at risk before each offset. The largest early contrasts are:

{early_contrasts[['evaluationCohort','candidateId','offset','featureId','definedStates','meanEntrantMinusNonentrant','lower95','upper95']].to_markdown(index=False)}

These contrasts describe what becomes visible *after stochastic propagation begins*. They cannot show that the same information is measurable at the starting state.

## Teacher attribution and reliability

The frozen L30 teacher's largest standardized coefficients were:

{attr.to_markdown(index=False)}

Independent 32/32 branch-half reliability was highest for:

{reliable[['evaluationCohort','candidateId','offset','metricId','splitHalfSpearman']].to_markdown(index=False)}

Reliability of a future-ensemble summary is distinct from past-observable transferability.

## Interpretation

{', '.join(classifications)}. The audit distinguishes three layers: realized basin entry (teacher), completed-run basin-distance summaries (retrospective and target-conditioned), and physical branch evolution. No L35 result changes S18, supports the paper's PhiRL claim, establishes early warning, or licenses intervention or reactive-current analysis.

## Validation and reproducibility

- Repository lock: `{runtime['repositoryHead']}` on `eidosoma/groups/42`.
- Workers: `{runtime['workers']}`; one numerical-library thread each; GPU hours `0`.
- Wall time: `{runtime['wallSeconds']:.2f}` seconds; controller CPU does not include worker CPU and is reported separately.
- The full scientific scope was independently regenerated from the same frozen states and streams and every table hash matched.
- Compact regenerated results were compared field by field with both L30 and L31 authoritative branch artifacts.

## Caveats

The target basin remains matrix-specific and reconstructed from each completed trajectory. Target-score quantities therefore contain retrospective basin information. The branches themselves are forward stochastic samples. There is one selected state per catalytic matrix in these cohorts, so within-matrix ordering remains unidentifiable. A physical branch feature that separates early would still require a new loop testing an outcome-blind present-state or observed-history proxy.

## Next boundary

L35 is frozen. The standing autonomous authorization permits only the narrowly named `{next_theme}` continuation if no solution boundary was reached. S20, E02, author contact, interventions, reactive-current claims and report generation remain inactive.
"""


def prepare_lock() -> None:
    LOOP_ROOT.mkdir(parents=True, exist_ok=True)
    if git("status", "--porcelain=v1"):
        raise RuntimeError("repository must be clean before L35 lock")
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/eidosoma/groups/42")
    if head != remote:
        raise RuntimeError("local/remote commit mismatch")
    prior = validate_immutable_prior()
    fixtures = fixture_results()
    responses = response_registry()
    coordinates = L33.target_coordinates(responses)
    manifest = pd.read_parquet(L23_ROOT / "input_trajectory_manifest.parquet")
    payload_rows = payloads(responses, coordinates, manifest)
    input_validation = input_replay_validation(payload_rows, responses)
    replay_seeds = replay_seed_manifest(responses)
    replay_validation = replay_seed_validation(replay_seeds, payload_rows)
    analysis_seeds = analysis_seed_manifest()
    firewall = analysis_seed_firewall(analysis_seeds)
    benchmark = benchmark_projection(payload_rows)
    if (
        not prior["unchanged"]
        or not fixtures["passed"].all()
        or not input_validation["allPassed"].all()
        or not replay_validation["allPassed"].all()
        or firewall["status"] != "PASS"
        or benchmark["status"] != "PASS"
    ):
        raise RuntimeError("L35 preoutcome validation or benchmark gate failed")
    shutil.copy2(CONFIG, LOOP_ROOT / "preregistration.yaml")
    BASE.atomic_text(
        LOOP_ROOT / "decision_record.md",
        "# S19-L35 decision record\n\n"
        "The reviewer proposed using the confirmed short propagator as a teacher and asking what it reveals that current/prefix representations discard. L35 freezes that question before full trace aggregation. It replays exactly the existing L30/L31 H8 streams, retains each intermediate observation, and separates realized entry and target-basin geometry from five physical mechanisms: composition dispersion, fission, mass, propensity concentration and sampled-event volume. The same physical metric and offset no later than four must satisfy the rank gate in both candidates and both evaluation cohorts. No branch-derived quantity is a past-only predictor, no new stochastic outcome is generated, and no favorable offset may be promoted post hoc.\n",
    )
    BASE.write_parquet(LOOP_ROOT / "fixture_results.parquet", fixtures)
    BASE.write_json(LOOP_ROOT / "benchmark_projection.json", benchmark)
    BASE.write_parquet(LOOP_ROOT / "response_registry.parquet", responses)
    BASE.write_parquet(LOOP_ROOT / "target_coordinate_registry.parquet", coordinates)
    BASE.write_parquet(LOOP_ROOT / "replay_seed_manifest.parquet", replay_seeds)
    BASE.write_parquet(LOOP_ROOT / "analysis_seed_manifest.parquet", analysis_seeds)
    BASE.write_json(LOOP_ROOT / "analysis_seed_firewall.json", firewall)
    BASE.write_parquet(LOOP_ROOT / "input_replay_validation.parquet", input_validation)
    BASE.write_parquet(LOOP_ROOT / "replay_seed_validation.parquet", replay_validation)
    BASE.write_parquet(
        LOOP_ROOT / "source_grounding_registry.parquet", source_grounding_registry()
    )
    BASE.write_json(LOOP_ROOT / "immutable_prior_validation.json", prior)
    hashes = {
        "responsesSha256": sha256_file(LOOP_ROOT / "response_registry.parquet"),
        "coordinatesSha256": sha256_file(
            LOOP_ROOT / "target_coordinate_registry.parquet"
        ),
        "replaySeedsSha256": sha256_file(LOOP_ROOT / "replay_seed_manifest.parquet"),
        "analysisSeedsSha256": sha256_file(
            LOOP_ROOT / "analysis_seed_manifest.parquet"
        ),
        "inputReplaySha256": sha256_file(
            LOOP_ROOT / "input_replay_validation.parquet"
        ),
        "replaySeedValidationSha256": sha256_file(
            LOOP_ROOT / "replay_seed_validation.parquet"
        ),
        "benchmarkSha256": sha256_file(LOOP_ROOT / "benchmark_projection.json"),
        "l30BranchSha256": sha256_file(L30_ROOT / "short_branch_results.parquet"),
        "l31BranchSha256": sha256_file(L31_ROOT / "branch_results.parquet"),
        "l23ManifestSha256": sha256_file(
            L23_ROOT / "input_trajectory_manifest.parquet"
        ),
    }
    BASE.write_json(
        LOOP_ROOT / "implementation_lock.json",
        {
            "schema": "eidosoma.e01.s19_l35.implementation_lock.v1",
            "repositoryHead": head,
            "remoteHead": remote,
            "runnerSha256": sha256_file(RUNNER_PATH),
            "coreSha256": sha256_file(CORE_PATH),
            "configSha256": sha256_file(CONFIG),
            "horizon": HORIZON,
            "branchesPerState": BRANCHES,
            "replayedBranches": 17_920,
            "newScientificBranches": 0,
            "primaryMetrics": list(PRIMARY_METRICS),
            "physicalMechanismMetrics": list(MECHANICAL_METRICS),
            "commonOffsetMaximum": 4,
            "matrixBootstraps": BOOTSTRAPS,
            "branchDerivedIsPastOnly": False,
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


def execute() -> None:
    started = time.perf_counter()
    started_cpu = time.process_time()
    lock = json.loads((LOOP_ROOT / "preoutcome_repository_lock.json").read_text())
    if (
        git("rev-parse", "HEAD") != lock["head"]
        or git("rev-parse", "origin/eidosoma/groups/42") != lock["remote"]
        or git("status", "--porcelain=v1")
    ):
        raise RuntimeError("L35 repository lock mismatch")
    prior = validate_immutable_prior()
    locked_files = {
        "responsesSha256": LOOP_ROOT / "response_registry.parquet",
        "coordinatesSha256": LOOP_ROOT / "target_coordinate_registry.parquet",
        "replaySeedsSha256": LOOP_ROOT / "replay_seed_manifest.parquet",
        "analysisSeedsSha256": LOOP_ROOT / "analysis_seed_manifest.parquet",
        "inputReplaySha256": LOOP_ROOT / "input_replay_validation.parquet",
        "replaySeedValidationSha256": LOOP_ROOT / "replay_seed_validation.parquet",
        "benchmarkSha256": LOOP_ROOT / "benchmark_projection.json",
        "l30BranchSha256": L30_ROOT / "short_branch_results.parquet",
        "l31BranchSha256": L31_ROOT / "branch_results.parquet",
        "l23ManifestSha256": L23_ROOT / "input_trajectory_manifest.parquet",
    }
    for key, path in locked_files.items():
        if sha256_file(path) != lock[key]:
            raise RuntimeError(f"L35 locked input changed: {path}")
    fixtures = fixture_results()
    if (
        not prior["unchanged"]
        or prior["aggregateSha256"] != lock["priorAggregateSha256"]
        or not fixtures["passed"].all()
        or sha256_file(RUNNER_PATH) != lock["runnerSha256"]
        or sha256_file(CORE_PATH) != lock["coreSha256"]
    ):
        raise RuntimeError("L35 pre-execution validation failed")
    responses = pd.read_parquet(LOOP_ROOT / "response_registry.parquet")
    coordinates = pd.read_parquet(LOOP_ROOT / "target_coordinate_registry.parquet")
    trajectory_manifest = pd.read_parquet(
        L23_ROOT / "input_trajectory_manifest.parquet"
    )
    payload_rows = payloads(responses, coordinates, trajectory_manifest)
    if BUILD_ROOT.exists():
        shutil.rmtree(BUILD_ROOT)
    BUILD_ROOT.mkdir(parents=True)
    branches, observations, offsets, state_contrasts = execute_traces(payload_rows)
    compact_validation = compact_replay_validation(branches)
    offsets = add_responses(offsets, responses)
    correlations = correlation_results(offsets)
    intervals = bootstrap_intervals(offsets)
    reliability = half_reliability(offsets)
    contrasts = aggregate_contrasts(state_contrasts)
    attribution = teacher_feature_attribution()
    gates, classifications, next_theme = mechanism_gate(correlations, intervals)
    solution = classifications[0] == "BRANCH_SIGNAL_LOCALIZES_BEFORE_ENTRY"
    make_figures(offsets, correlations, reliability, contrasts, attribution, gates)
    for name in (
        "preregistration.yaml",
        "decision_record.md",
        "fixture_results.parquet",
        "benchmark_projection.json",
        "response_registry.parquet",
        "target_coordinate_registry.parquet",
        "replay_seed_manifest.parquet",
        "analysis_seed_manifest.parquet",
        "analysis_seed_firewall.json",
        "input_replay_validation.parquet",
        "replay_seed_validation.parquet",
        "source_grounding_registry.parquet",
        "immutable_prior_validation.json",
        "implementation_lock.json",
        "preoutcome_repository_lock.json",
    ):
        shutil.copy2(LOOP_ROOT / name, BUILD_ROOT / name)
    BASE.write_parquet(BUILD_ROOT / "branch_compact_replay_results.parquet", branches)
    BASE.write_parquet(BUILD_ROOT / "compact_replay_validation.parquet", compact_validation)
    BASE.write_parquet(BUILD_ROOT / "branch_trace_observations.parquet", observations)
    BASE.write_parquet(BUILD_ROOT / "state_offset_ensemble_results.parquet", offsets)
    BASE.write_parquet(BUILD_ROOT / "state_entrant_nonentrant_contrasts.parquet", state_contrasts)
    BASE.write_parquet(BUILD_ROOT / "entrant_nonentrant_contrast_results.parquet", contrasts)
    BASE.write_parquet(BUILD_ROOT / "offset_correlation_results.parquet", correlations)
    BASE.write_parquet(BUILD_ROOT / "bootstrap_interval_results.parquet", intervals)
    BASE.write_parquet(BUILD_ROOT / "branch_half_reliability_results.parquet", reliability)
    BASE.write_parquet(BUILD_ROOT / "teacher_feature_attribution.parquet", attribution)
    BASE.write_parquet(BUILD_ROOT / "mechanism_gate_results.parquet", gates)
    BASE.write_json(
        BUILD_ROOT / "classification.json",
        {
            "schema": "eidosoma.e01.s19_l35.classification.v1",
            "classifications": classifications,
            "physicalMechanismGatePassed": solution,
            "branchDerivedFutureInformation": True,
            "pastOnlySignalEstablished": False,
            "targetBasinRetrospective": True,
            "newScientificOutcomes": 0,
            "priorStatusesChanged": False,
        },
    )
    pd.DataFrame(
        columns=[
            "stage",
            "stateId",
            "candidateId",
            "matrixIndex",
            "branchIndex",
            "exceptionClass",
            "exceptionMessage",
        ]
    ).to_csv(BUILD_ROOT / "failure_ledger.csv", index=False)

    # Exact full regeneration from frozen state and branch streams.
    replay_branches, replay_observations, replay_offsets, replay_contrasts = execute_traces(
        payload_rows
    )
    replay_offsets = add_responses(replay_offsets, responses)
    checks = {
        "branchCompactExact": frame_hash(replay_branches) == frame_hash(branches),
        "traceObservationExact": frame_hash(replay_observations)
        == frame_hash(observations),
        "stateOffsetExact": frame_hash(replay_offsets) == frame_hash(offsets),
        "stateContrastExact": frame_hash(replay_contrasts)
        == frame_hash(state_contrasts),
        "authoritativeCompactReplayPassed": bool(
            compact_validation["allCompactFieldsExact"].all()
        ),
        "inputReplayPassed": bool(
            pd.read_parquet(LOOP_ROOT / "input_replay_validation.parquet")[
                "allPassed"
            ].all()
        ),
        "seedReplayPassed": bool(
            pd.read_parquet(LOOP_ROOT / "replay_seed_validation.parquet")[
                "allPassed"
            ].all()
        ),
        "fixturesPassed": bool(fixtures["passed"].all()),
        "immutablePriorPassed": prior["unchanged"],
        "noNewScientificBranches": True,
        "candidateCohortSeparationPassed": len(
            correlations.groupby(["evaluationCohort", "candidateId"])
        )
        == 6,
    }
    if not all(checks.values()):
        raise RuntimeError(f"L35 regeneration validation failed: {checks}")
    BASE.write_json(
        BUILD_ROOT / "regeneration_validation.json",
        {
            "schema": "eidosoma.e01.s19_l35.regeneration_validation.v1",
            "status": "PASS",
            "checks": checks,
            "branchFrameSha256": frame_hash(branches),
            "observationFrameSha256": frame_hash(observations),
            "offsetFrameSha256": frame_hash(offsets),
            "correlationFrameSha256": frame_hash(correlations),
        },
    )
    runtime = {
        "schema": "eidosoma.e01.s19_l35.runtime.v1",
        "repositoryHead": git("rev-parse", "HEAD"),
        "workers": WORKERS,
        "numericalLibraryThreadsPerWorker": 1,
        "gpuHours": 0,
        "wallSeconds": time.perf_counter() - started,
        "controllerCpuHours": (time.process_time() - started_cpu) / 3600,
        "replayedBranches": len(branches) + len(replay_branches),
        "uniqueFrozenBranchStreams": len(branches),
        "newScientificBranchStreams": 0,
        "completedAtUtc": utc_now(),
    }
    BASE.write_json(BUILD_ROOT / "runtime_manifest.json", runtime)
    retained = sum(path.stat().st_size for path in BUILD_ROOT.rglob("*") if path.is_file())
    temporary = sum(path.stat().st_size for path in CACHE_ROOT.rglob("*") if path.is_file())
    storage = {
        "schema": "eidosoma.e01.s19_l35.storage_validation.v1",
        "retainedBytes": retained,
        "retainedGiBCeiling": 25,
        "temporaryBytes": temporary,
        "temporaryGiBCeiling": 75,
        "status": "PASS"
        if retained < 25 * 2**30 and temporary < 75 * 2**30
        else "FAIL",
    }
    if storage["status"] != "PASS":
        raise RuntimeError("L35 storage ceiling exceeded")
    BASE.write_json(BUILD_ROOT / "storage_validation.json", storage)
    report = report_text(
        correlations,
        intervals,
        reliability,
        contrasts,
        attribution,
        gates,
        classifications,
        runtime,
        next_theme,
    )
    BASE.atomic_text(BUILD_ROOT / "research_step_full_results.md", report)
    BASE.atomic_text(BUILD_ROOT / "S19_L35_FULL_RESULTS.md", report)
    BASE.atomic_text(
        BUILD_ROOT / "loop_decision_summary.md",
        "# S19-L35 decision summary\n\n"
        + f"**Classification:** {', '.join(classifications)}\n\n"
        + f"**Common physical mechanism localized:** `{solution}`.\n\n"
        + f"**Next:** `{next_theme}`.\n",
    )
    BASE.write_json(BUILD_ROOT / "artifact_manifest.json", manifest_for(BUILD_ROOT))
    stage = LOOP_ROOT.with_name(".L35-promotion-stage")
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
        raise RuntimeError("L35 artifact hash validation failed")
    append_ledgers(classifications, runtime["completedAtUtc"], next_theme, solution)
    BASE.atomic_text(ARTIFACT_ROOT / "research_step_full_results.md", report)
    BASE.atomic_text(
        ARTIFACT_ROOT / "S19_CURRENT_HANDOFF.md",
        report.replace("# S19-L35", "# S19 current handoff — S19-L35", 1),
    )
    BASE.write_json(
        ARTIFACT_ROOT / "s19_status.json",
        {
            "schema": "eidosoma.e01.s19.status.v1",
            "status": "HUMAN_REVIEW_REQUIRED_SOLUTION"
            if solution
            else "ACTIVE_AUTONOMOUS_SEQUENCE",
            "latestCompletedLoop": LOOP_ID,
            "latestClassification": classifications,
            "selectedDiscoveryLead": "LOCKED_BRANCH_MECHANISM"
            if solution
            else None,
            "nextAuthorizedLoop": None if solution else "S19-L36",
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
                "solution": solution,
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
