"""Execute S19-L40 online recurrence-after-departure committor audit."""

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
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from e01_onset_discovery.recurrence_after_departure import (
    exact_departure_return_order_probability,
    score_recurrence_after_departure,
)


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


L39 = _load_module(
    "e01_s19_l40_l39",
    REPO_ROOT / "scripts/e01/run_s19_l39_sustained_inheritance_committor.py",
)
L38 = L39.L38
L37 = L39.L37
L36 = L39.L36
L28 = L39.L28
BASE = L39.BASE
RestoredState = L39.RestoredState
corrected_between_state_variance = L39.corrected_between_state_variance

LOOP_ID = "S19-L40"
VERSION = "E01-S19-L40-ONLINE-RECURRENCE-AFTER-DEPARTURE-COMMITTOR-v1.0.0"
CANDIDATES = L39.CANDIDATES
COHORTS = L39.COHORTS
EVALUATION_COHORTS = L39.EVALUATION_COHORTS
FAMILIES = L39.FAMILIES
HORIZONS = L39.HORIZONS
BRANCH_COUNTS = L39.BRANCH_COUNTS
HALVES = L39.HALVES
TARGETS = (
    "PREFIX_ANCHOR",
    "SPECIES_PERMUTED_PREFIX_ANCHOR",
    "UNRELATED_MATRIX_PREFIX_ANCHOR",
)
PRIMARY_TARGET = TARGETS[0]
THRESHOLD = 0.9
BOOTSTRAPS = 4096
ROOT_HEX = "8f0342ed1b0fa4922a6a1ad6f3e4a66f0a0ca49a70572142b3527960918408fb"
PHASE = "s19_l40_recurrence_after_departure"
WORKERS = min(8, os.cpu_count() or 1)

ARTIFACT_ROOT = Path("/artifacts/research_steps/S19")
LOOP_ROOT = ARTIFACT_ROOT / "loops/L40"
L39_ROOT = ARTIFACT_ROOT / "loops/L39"
L38_ROOT = ARTIFACT_ROOT / "loops/L38"
CACHE_ROOT = Path("/cache/e01_s19_l40")
BUILD_ROOT = CACHE_ROOT / "build"
CONFIG = REPO_ROOT / "configs/e01/s19_l40_recurrence_after_departure.yaml"
RUNNER_PATH = Path(__file__)
CORE_PATH = REPO_ROOT / "src/e01_onset_discovery/recurrence_after_departure.py"


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
        frame.to_json(orient="table", index=False, double_precision=15).encode()
    ).hexdigest()


def derived_seed(*parts: object) -> int:
    material = "\x1f".join([VERSION, ROOT_HEX, *map(str, parts)]).encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "little")


def seed_material_sha256(*parts: object) -> str:
    return hashlib.sha256(
        "\x1f".join([VERSION, ROOT_HEX, *map(str, parts)]).encode()
    ).hexdigest()


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.unique(x[mask]).size < 2 or np.unique(y[mask]).size < 2:
        return float("nan")
    return float(spearmanr(x[mask], y[mask]).statistic)


def interval(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return float("nan"), float("nan")
    low, high = np.quantile(finite, [0.025, 0.975])
    return float(low), float(high)


def validate_immutable_prior() -> dict[str, Any]:
    inherited = json.loads((L39_ROOT / "immutable_prior_validation.json").read_text())
    rows = list(inherited["files"])
    manifest = json.loads((L39_ROOT / "artifact_manifest.json").read_text())
    rows.extend(
        {
            "path": str(L39_ROOT / item["path"]),
            "root": str(L39_ROOT),
            "bytes": item["bytes"],
            "sha256": item["sha256"],
        }
        for item in manifest["files"]
    )
    checked = []
    for row in rows:
        path = Path(row["path"])
        actual = sha256_file(path) if path.is_file() else None
        checked.append({**row, "actualSha256": actual, "unchanged": actual == row["sha256"]})
    aggregate = hashlib.sha256(
        "\n".join(f"{row['path']}|{row['sha256']}" for row in checked).encode()
    ).hexdigest()
    passed = all(row["unchanged"] for row in checked)
    return {
        "schema": "eidosoma.e01.s19_l40.immutable_prior_validation.v1",
        "status": "PASS" if passed else "FAIL",
        "unchanged": passed,
        "fileCount": len(checked),
        "aggregateSha256": aggregate,
        "l39ManifestSha256": sha256_file(L39_ROOT / "artifact_manifest.json"),
        "files": checked,
    }


def fixture_results() -> pd.DataFrame:
    anchor = np.asarray([10, 0], dtype=np.int64)
    near = np.asarray([9, 1], dtype=np.int64)
    far = np.asarray([0, 10], dtype=np.int64)
    event = score_recurrence_after_departure(
        anchor=anchor,
        future_states=np.asarray([near, far, near]),
        generations=np.asarray([3, 4, 5]),
        offsets_one_based=np.asarray([2, 4, 6]),
        threshold=THRESHOLD,
    )
    no_departure = score_recurrence_after_departure(
        anchor=anchor,
        future_states=np.asarray([near, near, near]),
        generations=np.asarray([3, 4, 5]),
        offsets_one_based=np.asarray([2, 4, 6]),
        threshold=THRESHOLD,
    )
    no_return = score_recurrence_after_departure(
        anchor=anchor,
        future_states=np.asarray([far, far]),
        generations=np.asarray([3, 4]),
        offsets_one_based=np.asarray([2, 4]),
        threshold=THRESHOLD,
    )
    replay = score_recurrence_after_departure(
        anchor=anchor.copy(),
        future_states=np.asarray([near, far, near]),
        generations=np.asarray([3, 4, 5]),
        offsets_one_based=np.asarray([2, 4, 6]),
        threshold=THRESHOLD,
    )
    return pd.DataFrame(
        [
            {
                "fixtureId": "DEPARTURE_THEN_RETURN_CERTIFIED",
                "passed": event.event
                and event.departure_boundary_one_based == 2
                and event.certification_boundary_one_based == 3,
                "details": "certification occurs only at the post-departure return",
            },
            {
                "fixtureId": "ADJACENT_SMOOTHNESS_NOT_RECURRENCE",
                "passed": not no_departure.event and not no_departure.departure_observed,
                "details": "remaining near the anchor never establishes recurrence",
            },
            {
                "fixtureId": "DEPARTURE_WITHOUT_RETURN_REJECTED",
                "passed": not no_return.event and no_return.departure_observed,
                "details": "departure alone is not a certified return",
            },
            {
                "fixtureId": "ORDER_NULL_EXACT",
                "passed": abs(exact_departure_return_order_probability(3, 2) - 2 / 3) <= 1e-15,
                "details": "two of three fixed-count orders have a departure before a near state",
            },
            {
                "fixtureId": "H8_PROGRESS_TARGET_MECHANICS",
                "passed": event.return_progress > THRESHOLD and no_departure.return_progress == 0,
                "details": "soft progress is target-specific and zero without departure",
            },
            {
                "fixtureId": "EXACT_REPLAY",
                "passed": event == replay,
                "details": "all continuous and discrete values replay exactly",
            },
            {
                "fixtureId": "FROZEN_SCOPE",
                "passed": FAMILIES == ("H32", "H8")
                and BRANCH_COUNTS == {"H32": 128, "H8": 64}
                and TARGETS
                == (
                    "PREFIX_ANCHOR",
                    "SPECIES_PERMUTED_PREFIX_ANCHOR",
                    "UNRELATED_MATRIX_PREFIX_ANCHOR",
                ),
                "details": json.dumps(
                    {"families": FAMILIES, "branchCounts": BRANCH_COUNTS, "targets": TARGETS}
                ),
            },
        ]
    )


def anchor_registry() -> pd.DataFrame:
    boundaries = pd.read_parquet(L38_ROOT / "prefix_boundary_registry.parquet")
    donors = pd.read_parquet(L38_ROOT / "unrelated_control_map.parquet")
    permutations = pd.read_parquet(L38_ROOT / "species_permutation_manifest.parquet")
    latest = (
        boundaries.sort_values(["stateId", "generation"])
        .groupby("stateId", as_index=False)
        .tail(1)
        .set_index("stateId")
    )
    donor_map = donors.set_index("stateId")["donorStateId"].to_dict()
    permutation_map = permutations.set_index("stateId")["permutation"].to_dict()
    rows = []
    for state_id, row in latest.iterrows():
        anchor = np.asarray(row.state, dtype=np.int64)
        donor_id = donor_map[state_id]
        unrelated = np.asarray(latest.loc[donor_id, "state"], dtype=np.int64)
        permutation = np.asarray(permutation_map[state_id], dtype=np.int64)
        for target_id, value in (
            ("PREFIX_ANCHOR", anchor),
            ("SPECIES_PERMUTED_PREFIX_ANCHOR", anchor[permutation]),
            ("UNRELATED_MATRIX_PREFIX_ANCHOR", unrelated),
        ):
            rows.append(
                {
                    "stateId": state_id,
                    "evaluationCohort": row.evaluationCohort,
                    "candidateId": row.candidateId,
                    "matrixIndex": int(row.matrixIndex),
                    "landmark": int(row.landmark),
                    "targetId": target_id,
                    "anchorGeneration": int(row.generation),
                    "anchorSelectedClockIndex": int(row.selectedClockIndex),
                    "anchor": value.tolist(),
                    "anchorSha256": L28.simulator_array_sha256(value),
                    "donorStateId": donor_id if target_id == "UNRELATED_MATRIX_PREFIX_ANCHOR" else None,
                    "targetUsesCompletedTestTrajectory": False,
                }
            )
    result = pd.DataFrame(rows).sort_values(
        ["evaluationCohort", "candidateId", "landmark", "matrixIndex", "targetId"]
    ).reset_index(drop=True)
    if len(result) != 280 * len(TARGETS):
        raise RuntimeError("L40 anchor registry cardinality failure")
    return result


def build_payloads(anchors: pd.DataFrame) -> list[dict[str, Any]]:
    payloads = L39.build_payloads()
    anchor_map = {
        state_id: {
            row.targetId: row.anchor
            for row in group.itertuples(index=False)
        }
        for state_id, group in anchors.groupby("stateId", sort=False)
    }
    output = []
    for payload in payloads:
        row = dict(payload)
        row["l40Anchors"] = anchor_map[payload["stateId"]]
        output.append(row)
    if len(output) != 280:
        raise RuntimeError("L40 payload cardinality failure")
    return output


def analysis_seed_manifest() -> pd.DataFrame:
    comparisons = (
        "H8_RETURN_PROGRESS_VS_H32_PRIMARY",
        "H8_EVENT_Q_VS_H32_PRIMARY",
        "L39_H8_INHERITANCE_PROPENSITY_VS_H32_PRIMARY",
        "H32_PRIMARY_MINUS_PERMUTED_ANCHOR",
        "H32_PRIMARY_MINUS_UNRELATED_ANCHOR",
        "H32_PRIMARY_MINUS_ORDER_NULL",
        "H32_MIXED_OPPORTUNITY_VS_PRIMARY",
        "PREFIX_INHERITANCE_FRACTION_VS_H32_PRIMARY",
        "CURRENT_MASS_VS_H32_PRIMARY",
        "GENERATION_PHASE_VS_H32_PRIMARY",
    )
    rows = []
    for cohort in COHORTS:
        for candidate in CANDIDATES:
            for family in FAMILIES:
                for target in TARGETS:
                    parts = ("reliability_bootstrap", cohort, candidate, family, target)
                    rows.append(
                        {
                            "purpose": parts[0],
                            "partsJson": json.dumps(parts),
                            "evaluationCohort": cohort,
                            "candidateId": candidate,
                            "branchFamily": family,
                            "targetId": target,
                            "comparisonId": None,
                            "rootHex": ROOT_HEX,
                            "derivedSeed": str(derived_seed(*parts)),
                            "seedMaterialSha256": seed_material_sha256(*parts),
                        }
                    )
            for comparison in comparisons:
                parts = ("transfer_bootstrap", cohort, candidate, comparison)
                rows.append(
                    {
                        "purpose": parts[0],
                        "partsJson": json.dumps(parts),
                        "evaluationCohort": cohort,
                        "candidateId": candidate,
                        "branchFamily": None,
                        "targetId": None,
                        "comparisonId": comparison,
                        "rootHex": ROOT_HEX,
                        "derivedSeed": str(derived_seed(*parts)),
                        "seedMaterialSha256": seed_material_sha256(*parts),
                    }
                )
    result = pd.DataFrame(rows).sort_values(
        ["purpose", "evaluationCohort", "candidateId", "branchFamily", "targetId", "comparisonId"],
        na_position="last",
    ).reset_index(drop=True)
    if result["derivedSeed"].duplicated().any() or result["seedMaterialSha256"].duplicated().any():
        raise RuntimeError("L40 analysis seed collision")
    return result


def seed_firewall(seeds: pd.DataFrame) -> dict[str, Any]:
    prior_material: set[str] = set()
    prior_derived: set[str] = set()
    for path in ARTIFACT_ROOT.glob("loops/L*/**/*seed*manifest*.parquet"):
        if "/L40/" in str(path):
            continue
        try:
            frame = pd.read_parquet(path)
        except (OSError, ValueError, TypeError):
            continue
        for column in frame.columns:
            lowered = column.lower()
            if "seedmaterialsha256" in lowered:
                prior_material.update(frame[column].dropna().astype(str))
            if lowered == "derivedseed":
                prior_derived.update(frame[column].dropna().astype(str))
    material = sorted(set(seeds["seedMaterialSha256"]) & prior_material)
    derived = sorted(set(seeds["derivedSeed"]) & prior_derived)
    return {
        "schema": "eidosoma.e01.s19_l40.seed_firewall.v1",
        "status": "PASS" if not material and not derived else "FAIL",
        "analysisSeedCount": len(seeds),
        "seedMaterialOverlapCount": len(material),
        "derivedSeedOverlapCount": len(derived),
        "seedMaterialOverlaps": material,
        "derivedSeedOverlaps": derived,
        "newBranchSeeds": 0,
    }


def source_grounding_registry() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sourceId": "L40_REVIEWER_RECURRENCE_AFTER_DEPARTURE",
                "evidenceClass": "HUMAN_REVIEW_DIRECTION",
                "finding": "A clean recurrence event should require temporal separation and an intervening departure.",
                "frozenUse": "online return to the latest observed post-fission anchor after a future departure",
            },
            {
                "sourceId": "L40_REVIEWER_CLOCK_SEPARATION",
                "evidenceClass": "HUMAN_REVIEW_DIRECTION",
                "finding": "Generational and molecular clocks must remain separate.",
                "frozenUse": "primary event only at post-fission boundaries; molecular offsets diagnostic only",
            },
            {
                "sourceId": "L40_L39_ORDER_NULL",
                "evidenceClass": "DIRECT_FROZEN_E01_RESULT",
                "finding": "L39 sustained inheritance was explained by high marginal inheritance frequency under a fixed-count order null.",
                "frozenUse": "requires an exact order-conditioned control for return dynamics",
            },
            {
                "sourceId": "L40_L28_L31_EXACT_BRANCH_STREAMS",
                "evidenceClass": "DIRECT_FROZEN_E01_RESULT",
                "finding": "The exact H32/H8 branch streams are replayable and candidate separated.",
                "frozenUse": "zero-new-stream process committor audit",
            },
        ]
    )


def _branch_worker(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    matrix_index = int(payload["matrixIndex"])
    beta = L28.generate_beta(
        L28.derive_seed(L28.L23_ROOT_HEX, L28.L23_PHASE, "catalytic_matrix", matrix_index)
    )
    if L28.simulator_array_sha256(beta) != payload["betaSha256"]:
        raise RuntimeError(f"L40 beta replay failure: {payload['stateId']}")
    restored = RestoredState(
        tuple(payload["state"]),
        payload["currentObservationKind"],
        int(payload["currentCompletedFissions"]),
        int(payload["currentGrowthGeneration"]),
        int(payload["currentGenerationLocalStep"]),
        int(payload["currentBatchStep"]),
    )
    original = np.asarray(payload["centroid"], dtype=np.float64)
    outcomes: list[dict[str, Any]] = []
    compact_rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        for branch in range(BRANCH_COUNTS[family]):
            event_rng, trim_rng, fission_rng, daughter_rng = L36._branch_rngs(payload, family, branch)
            trace = L37.simulate_branch_trace(
                restored=restored,
                beta=beta,
                definition=L28.definition(payload["candidateId"]),
                target_centroid=original,
                event_rng=event_rng,
                trim_rng=trim_rng,
                fission_rng=fission_rng,
                daughter_rng=daughter_rng,
                horizon=HORIZONS[family],
                threshold=THRESHOLD,
            )
            compact = trace.compact
            boundary_observations = [
                observation for observation in trace.observations if observation.observation_kind == "post_fission"
            ]
            future_states = np.asarray(
                [observation.state for observation in boundary_observations], dtype=np.int64
            ).reshape((-1, 100))
            generations = np.asarray(
                [observation.generation for observation in boundary_observations], dtype=np.int64
            )
            offsets = np.asarray(
                [observation.offset for observation in boundary_observations], dtype=np.int64
            )
            compact_rows.append(
                {
                    "stateId": payload["stateId"],
                    "evaluationCohort": payload["evaluationCohort"],
                    "candidateId": payload["candidateId"],
                    "matrixIndex": matrix_index,
                    "landmark": int(payload["landmark"]),
                    "branchFamily": family,
                    "targetId": "ORIGINAL",
                    "branchIndex": branch,
                    "branchHalf": "A" if branch < HALVES[family] else "B",
                    "enteredTarget": compact.entered_basin,
                    "firstEntryOffsetOneBased": compact.first_entry_offset_one_based,
                    "maximumTargetScore": compact.maximum_target_score,
                    "minimumTargetScore": compact.minimum_target_score,
                    "molecularUpdates": compact.molecular_updates,
                    "fissions": compact.fissions,
                    "selectedObservationsGenerated": compact.selected_observations_generated,
                    "terminalStatus": compact.terminal_status,
                    "originalPathSha256": compact.path_sha256,
                }
            )
            for target_id in TARGETS:
                anchor = np.asarray(payload["l40Anchors"][target_id], dtype=np.int64)
                scored = score_recurrence_after_departure(
                    anchor=anchor,
                    future_states=future_states,
                    generations=generations,
                    offsets_one_based=offsets,
                    threshold=THRESHOLD,
                )
                outcomes.append(
                    {
                        "stateId": payload["stateId"],
                        "evaluationCohort": payload["evaluationCohort"],
                        "candidateId": payload["candidateId"],
                        "matrixIndex": matrix_index,
                        "landmark": int(payload["landmark"]),
                        "branchFamily": family,
                        "targetId": target_id,
                        "branchIndex": branch,
                        "branchHalf": "A" if branch < HALVES[family] else "B",
                        "event": scored.event,
                        "departureBoundaryOneBased": scored.departure_boundary_one_based,
                        "departureGeneration": scored.departure_generation,
                        "departureOffsetOneBased": scored.departure_offset_one_based,
                        "certificationBoundaryOneBased": scored.certification_boundary_one_based,
                        "certificationGeneration": scored.certification_generation,
                        "certificationOffsetOneBased": scored.certification_offset_one_based,
                        "futureBoundaryCount": scored.future_boundary_count,
                        "nearAnchorCount": scored.near_anchor_count,
                        "departedCount": scored.departed_count,
                        "departureObserved": scored.departure_observed,
                        "mixedMembershipOpportunity": scored.mixed_membership_opportunity,
                        "maximumPostdepartureH": scored.maximum_postdeparture_h,
                        "returnProgress": scored.return_progress,
                        "exactOrderNullEventProbability": scored.exact_order_null_event_probability,
                        "anchorScoreSequence": json.dumps(scored.scores, separators=(",", ":")),
                        "selectedObservationsGenerated": compact.selected_observations_generated,
                        "terminalStatus": compact.terminal_status,
                        "originalPathSha256": compact.path_sha256,
                        "targetUsesCompletedTestTrajectory": False,
                        "originalCentroidUsedOnlyForFrozenPathDigest": True,
                    }
                )
    return {"outcomes": outcomes, "compact": compact_rows}


def execute_branches(payloads: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    outcomes: list[dict[str, Any]] = []
    compact: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(_branch_worker, payload): payload["stateId"] for payload in payloads}
        for future in as_completed(futures):
            result = future.result()
            outcomes.extend(result["outcomes"])
            compact.extend(result["compact"])
    outcome_frame = pd.DataFrame(outcomes).sort_values(
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
    compact_frame = pd.DataFrame(compact).sort_values(
        ["evaluationCohort", "candidateId", "landmark", "matrixIndex", "branchFamily", "branchIndex"]
    ).reset_index(drop=True)
    if (
        len(outcome_frame) != 53_760 * len(TARGETS)
        or len(compact_frame) != 53_760
        or outcome_frame.duplicated(["stateId", "branchFamily", "targetId", "branchIndex"]).any()
        or compact_frame.duplicated(["stateId", "branchFamily", "branchIndex"]).any()
    ):
        raise RuntimeError("L40 branch result cardinality failure")
    return outcome_frame, compact_frame


def state_committor_results(outcomes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (state_id, family, target), group in outcomes.groupby(
        ["stateId", "branchFamily", "targetId"], sort=True
    ):
        expected = BRANCH_COUNTS[family]
        if len(group) != expected:
            raise RuntimeError("L40 per-state branch cardinality failure")
        half_a = group[group["branchHalf"].eq("A")]
        half_b = group[group["branchHalf"].eq("B")]
        events = group["event"].to_numpy(dtype=np.bool_)
        rows.append(
            {
                "stateId": state_id,
                "evaluationCohort": group["evaluationCohort"].iloc[0],
                "candidateId": group["candidateId"].iloc[0],
                "matrixIndex": int(group["matrixIndex"].iloc[0]),
                "landmark": int(group["landmark"].iloc[0]),
                "branchFamily": family,
                "targetId": target,
                "branches": len(group),
                "successes": int(events.sum()),
                "qHat": float(events.mean()),
                "qHatHalfA": float(half_a["event"].mean()),
                "qHatHalfB": float(half_b["event"].mean()),
                "qDeparture": float(group["departureObserved"].mean()),
                "qMixedMembershipOpportunity": float(group["mixedMembershipOpportunity"].mean()),
                "meanReturnProgress": float(group["returnProgress"].mean()),
                "meanExactOrderNullProbability": float(group["exactOrderNullEventProbability"].mean()),
                "meanFutureBoundaryCount": float(group["futureBoundaryCount"].mean()),
                "meanNearAnchorCount": float(group["nearAnchorCount"].mean()),
                "meanDepartedCount": float(group["departedCount"].mean()),
                "meanDepartureOffset": float(group["departureOffsetOneBased"].mean()),
                "meanCertificationOffset": float(group["certificationOffsetOneBased"].mean()),
                "opportunityWithoutCertificationFraction": float(
                    np.mean(group["mixedMembershipOpportunity"] & ~group["event"])
                ),
                "committorEligible": bool(
                    len(group) == expected
                    and group["selectedObservationsGenerated"].eq(HORIZONS[family]).all()
                ),
                "targetUsesCompletedTestTrajectory": False,
            }
        )
    result = pd.DataFrame(rows).sort_values(
        ["evaluationCohort", "candidateId", "landmark", "matrixIndex", "branchFamily", "targetId"]
    ).reset_index(drop=True)
    if len(result) != 280 * len(FAMILIES) * len(TARGETS):
        raise RuntimeError("L40 state committor cardinality failure")
    return result


def reliability_results(states: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    bootstrap_rows = []
    for (cohort, candidate, family, target), group in states.groupby(
        ["evaluationCohort", "candidateId", "branchFamily", "targetId"], sort=True
    ):
        eligible = group[group["committorEligible"]]
        q = eligible["qHat"].to_numpy(dtype=np.float64)
        half_a = eligible["qHatHalfA"].to_numpy(dtype=np.float64)
        half_b = eligible["qHatHalfB"].to_numpy(dtype=np.float64)
        variance = corrected_between_state_variance(q, BRANCH_COUNTS[family])
        split = safe_spearman(half_a, half_b)
        rng = np.random.default_rng(
            derived_seed("reliability_bootstrap", cohort, candidate, family, target)
        )
        corrected_boot = np.full(BOOTSTRAPS, np.nan)
        split_boot = np.full(BOOTSTRAPS, np.nan)
        for replicate in range(BOOTSTRAPS):
            indices = rng.integers(0, len(eligible), len(eligible))
            corrected_boot[replicate] = corrected_between_state_variance(
                q[indices], BRANCH_COUNTS[family]
            )["correctedBetweenStateVariance"]
            split_boot[replicate] = safe_spearman(half_a[indices], half_b[indices])
            bootstrap_rows.append(
                {
                    "evaluationCohort": cohort,
                    "candidateId": candidate,
                    "branchFamily": family,
                    "targetId": target,
                    "replicate": replicate,
                    "correctedVariance": corrected_boot[replicate],
                    "splitHalfSpearman": split_boot[replicate],
                }
            )
        corrected_lower, corrected_upper = interval(corrected_boot)
        split_lower, split_upper = interval(split_boot)
        intermediate = int(np.sum((q > 0.1) & (q < 0.9)))
        rows.append(
            {
                "evaluationCohort": cohort,
                "candidateId": candidate,
                "branchFamily": family,
                "targetId": target,
                "states": len(group),
                "eligibleStates": len(eligible),
                "meanQ": float(np.mean(q)),
                "minimumQ": float(np.min(q)),
                "maximumQ": float(np.max(q)),
                "intermediateStateCount": intermediate,
                "observedBetweenStateVariance": variance["observedBetweenStateVariance"],
                "estimatedBinomialNoiseVariance": variance["estimatedBinomialNoiseVariance"],
                "correctedBetweenStateVariance": variance["correctedBetweenStateVariance"],
                "correctedVarianceLower95": corrected_lower,
                "correctedVarianceUpper95": corrected_upper,
                "splitHalfSpearman": split,
                "splitHalfLower95": split_lower,
                "splitHalfUpper95": split_upper,
                "reliabilityGatePassed": bool(
                    len(eligible) >= 20
                    and corrected_lower > 0
                    and split > 0.5
                    and split_lower > 0.3
                    and intermediate >= 20
                ),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(bootstrap_rows)


def transfer_pairs(
    states: pd.DataFrame,
    responses: pd.DataFrame,
    prefixes: pd.DataFrame,
    l39_states: pd.DataFrame,
) -> pd.DataFrame:
    index = states.set_index(["stateId", "branchFamily", "targetId"])
    response_index = responses.set_index("stateId")
    prefix_index = prefixes.set_index("stateId")
    l39_index = l39_states.set_index(["stateId", "branchFamily"])
    specifications = (
        ("H8_RETURN_PROGRESS_VS_H32_PRIMARY", "RANK"),
        ("H8_EVENT_Q_VS_H32_PRIMARY", "RANK"),
        ("L39_H8_INHERITANCE_PROPENSITY_VS_H32_PRIMARY", "RANK"),
        ("H32_PRIMARY_MINUS_PERMUTED_ANCHOR", "DIFFERENCE"),
        ("H32_PRIMARY_MINUS_UNRELATED_ANCHOR", "DIFFERENCE"),
        ("H32_PRIMARY_MINUS_ORDER_NULL", "DIFFERENCE"),
        ("H32_MIXED_OPPORTUNITY_VS_PRIMARY", "RANK"),
        ("PREFIX_INHERITANCE_FRACTION_VS_H32_PRIMARY", "RANK"),
        ("CURRENT_MASS_VS_H32_PRIMARY", "RANK"),
        ("GENERATION_PHASE_VS_H32_PRIMARY", "RANK"),
    )
    rows = []
    for state_id in states["stateId"].drop_duplicates():
        h32 = index.loc[(state_id, "H32", PRIMARY_TARGET)]
        h8 = index.loc[(state_id, "H8", PRIMARY_TARGET)]
        permuted = index.loc[(state_id, "H32", "SPECIES_PERMUTED_PREFIX_ANCHOR")]
        unrelated = index.loc[(state_id, "H32", "UNRELATED_MATRIX_PREFIX_ANCHOR")]
        response = response_index.loc[state_id]
        prefix = prefix_index.loc[state_id]
        old_h8 = l39_index.loc[(state_id, "H8")]
        values = {
            "H8_RETURN_PROGRESS_VS_H32_PRIMARY": (h8.meanReturnProgress, h32.qHat),
            "H8_EVENT_Q_VS_H32_PRIMARY": (h8.qHat, h32.qHat),
            "L39_H8_INHERITANCE_PROPENSITY_VS_H32_PRIMARY": (
                old_h8.qAnyInheritance,
                h32.qHat,
            ),
            "H32_PRIMARY_MINUS_PERMUTED_ANCHOR": (h32.qHat, permuted.qHat),
            "H32_PRIMARY_MINUS_UNRELATED_ANCHOR": (h32.qHat, unrelated.qHat),
            "H32_PRIMARY_MINUS_ORDER_NULL": (
                h32.qHat,
                h32.meanExactOrderNullProbability,
            ),
            "H32_MIXED_OPPORTUNITY_VS_PRIMARY": (
                h32.qMixedMembershipOpportunity,
                h32.qHat,
            ),
            "PREFIX_INHERITANCE_FRACTION_VS_H32_PRIMARY": (
                prefix.prefixInheritanceFraction,
                h32.qHat,
            ),
            "CURRENT_MASS_VS_H32_PRIMARY": (response.currentMass, h32.qHat),
            "GENERATION_PHASE_VS_H32_PRIMARY": (
                response.currentGenerationLocalStep,
                h32.qHat,
            ),
        }
        for comparison, comparison_type in specifications:
            left, right = values[comparison]
            rows.append(
                {
                    "stateId": state_id,
                    "evaluationCohort": h32.evaluationCohort,
                    "candidateId": h32.candidateId,
                    "matrixIndex": int(h32.matrixIndex),
                    "comparisonId": comparison,
                    "comparisonType": comparison_type,
                    "leftValue": float(left),
                    "rightValue": float(right),
                }
            )
    result = pd.DataFrame(rows).sort_values(
        ["evaluationCohort", "candidateId", "comparisonId", "stateId"]
    ).reset_index(drop=True)
    if len(result) != 280 * len(specifications):
        raise RuntimeError("L40 transfer-pair cardinality failure")
    return result


def transfer_results(pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    bootstrap_rows = []
    for (cohort, candidate, comparison, comparison_type), group in pairs.groupby(
        ["evaluationCohort", "candidateId", "comparisonId", "comparisonType"], sort=True
    ):
        left = group["leftValue"].to_numpy(dtype=np.float64)
        right = group["rightValue"].to_numpy(dtype=np.float64)
        observed = (
            safe_spearman(left, right)
            if comparison_type == "RANK"
            else float(np.mean(left - right))
        )
        rng = np.random.default_rng(derived_seed("transfer_bootstrap", cohort, candidate, comparison))
        boot = np.full(BOOTSTRAPS, np.nan)
        for replicate in range(BOOTSTRAPS):
            indices = rng.integers(0, len(group), len(group))
            boot[replicate] = (
                safe_spearman(left[indices], right[indices])
                if comparison_type == "RANK"
                else float(np.mean(left[indices] - right[indices]))
            )
            bootstrap_rows.append(
                {
                    "evaluationCohort": cohort,
                    "candidateId": candidate,
                    "comparisonId": comparison,
                    "comparisonType": comparison_type,
                    "replicate": replicate,
                    "value": boot[replicate],
                }
            )
        lower, upper = interval(boot)
        rows.append(
            {
                "evaluationCohort": cohort,
                "candidateId": candidate,
                "comparisonId": comparison,
                "comparisonType": comparison_type,
                "definedPairs": len(group),
                "pointEstimate": observed,
                "lower95": lower,
                "upper95": upper,
                "gatePassed": bool(
                    np.isfinite(observed)
                    and np.isfinite(lower)
                    and (
                        (comparison_type == "RANK" and observed > 0.5 and lower > 0.3)
                        or (comparison_type == "DIFFERENCE" and lower > 0)
                    )
                ),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(bootstrap_rows)


def boundary_hazard_results(outcomes: pd.DataFrame) -> pd.DataFrame:
    primary = outcomes[outcomes["targetId"].eq(PRIMARY_TARGET)]
    rows = []
    for (cohort, candidate, family), group in primary.groupby(
        ["evaluationCohort", "candidateId", "branchFamily"], sort=True
    ):
        maximum = int(group["futureBoundaryCount"].max())
        survival = 1.0
        for boundary in range(1, maximum + 1):
            at_risk = group[
                group["futureBoundaryCount"].ge(boundary)
                & (
                    group["certificationBoundaryOneBased"].isna()
                    | group["certificationBoundaryOneBased"].ge(boundary)
                )
            ]
            events = int(group["certificationBoundaryOneBased"].eq(boundary).sum())
            hazard = events / len(at_risk) if len(at_risk) else float("nan")
            if np.isfinite(hazard):
                survival *= 1.0 - hazard
            rows.append(
                {
                    "evaluationCohort": cohort,
                    "candidateId": candidate,
                    "branchFamily": family,
                    "futureBoundaryOneBased": boundary,
                    "branchesAtRisk": len(at_risk),
                    "certifications": events,
                    "discreteHazard": hazard,
                    "survivalWithoutCertification": survival,
                    "cumulativeCertificationIncidence": 1.0 - survival,
                }
            )
    return pd.DataFrame(rows)


def scientific_gates(
    reliability: pd.DataFrame,
    transfers: pd.DataFrame,
    states: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], str]:
    rows = []
    for cohort in EVALUATION_COHORTS:
        for candidate in CANDIDATES:
            rel = reliability[
                reliability["evaluationCohort"].eq(cohort)
                & reliability["candidateId"].eq(candidate)
            ].set_index(["branchFamily", "targetId"])
            comparison = transfers[
                transfers["evaluationCohort"].eq(cohort)
                & transfers["candidateId"].eq(candidate)
            ].set_index("comparisonId")
            h32_reliable = bool(rel.loc[("H32", PRIMARY_TARGET), "reliabilityGatePassed"])
            h8_progress = bool(comparison.loc["H8_RETURN_PROGRESS_VS_H32_PRIMARY", "gatePassed"])
            permuted = bool(comparison.loc["H32_PRIMARY_MINUS_PERMUTED_ANCHOR", "gatePassed"])
            unrelated = bool(comparison.loc["H32_PRIMARY_MINUS_UNRELATED_ANCHOR", "gatePassed"])
            order = bool(comparison.loc["H32_PRIMARY_MINUS_ORDER_NULL", "gatePassed"])
            selected_states = states[
                states["evaluationCohort"].eq(cohort)
                & states["candidateId"].eq(candidate)
                & states["branchFamily"].eq("H32")
                & states["targetId"].eq(PRIMARY_TARGET)
            ]
            opportunity = bool(
                selected_states["opportunityWithoutCertificationFraction"].mean() >= 0.1
            )
            target_gate = h32_reliable and permuted and unrelated and order and opportunity
            short_gate = target_gate and h8_progress
            rows.append(
                {
                    "evaluationCohort": cohort,
                    "candidateId": candidate,
                    "primaryH32Reliable": h32_reliable,
                    "h8ReturnProgressTransferPassed": h8_progress,
                    "permutedAnchorControlPassed": permuted,
                    "unrelatedAnchorControlPassed": unrelated,
                    "sequenceOrderControlPassed": order,
                    "opportunityNondegeneracyPassed": opportunity,
                    "recurrenceAfterDepartureTargetPassed": target_gate,
                    "shortShootingCoordinatePassed": short_gate,
                    "l39InheritancePropensityPassed": bool(
                        comparison.loc[
                            "L39_H8_INHERITANCE_PROPENSITY_VS_H32_PRIMARY", "gatePassed"
                        ]
                    ),
                    "prefixInheritanceControlPassed": bool(
                        comparison.loc[
                            "PREFIX_INHERITANCE_FRACTION_VS_H32_PRIMARY", "gatePassed"
                        ]
                    ),
                    "massControlPassed": bool(
                        comparison.loc["CURRENT_MASS_VS_H32_PRIMARY", "gatePassed"]
                    ),
                    "phaseControlPassed": bool(
                        comparison.loc["GENERATION_PHASE_VS_H32_PRIMARY", "gatePassed"]
                    ),
                }
            )
    gates = pd.DataFrame(rows)
    target_all = bool(gates["recurrenceAfterDepartureTargetPassed"].all())
    short_all = bool(gates["shortShootingCoordinatePassed"].all())
    anchor_all = bool(
        gates[["permutedAnchorControlPassed", "unrelatedAnchorControlPassed"]].all(axis=None)
    )
    order_all = bool(gates["sequenceOrderControlPassed"].all())
    if short_all:
        classifications = [
            "ONLINE_RECURRENCE_AFTER_DEPARTURE_COMMITTOR_ESTABLISHED",
            "SHORT_SHOOTING_RETURN_COORDINATE_ESTABLISHED",
            "PROMOTABLE_TO_UNTOUCHED_PROCESS_CONFIRMATION",
        ]
        next_theme = "UNTOUCHED_RECURRENCE_AFTER_DEPARTURE_CONFIRMATION"
    elif target_all:
        classifications = [
            "ONLINE_RECURRENCE_AFTER_DEPARTURE_COMMITTOR_ESTABLISHED",
            "SHOOTING_ONLY_RETURN_ESTIMATOR_REQUIRED",
            "NOT_PROMOTABLE_AS_CONFIRMED",
        ]
        next_theme = "RECURRENCE_SHOOTING_BUDGET_EFFICIENCY"
    elif not anchor_all:
        classifications = [
            "RECURRENCE_AFTER_DEPARTURE_NOT_ANCHOR_SPECIFIC",
            "GENERIC_COMPOSITIONAL_RETURN_SUFFICIENT",
            "NOT_PROMOTABLE_AS_CONFIRMED",
        ]
        next_theme = "REPEATED_CROSS_GENERATION_RECURRENCE_COMMITTOR"
    elif not order_all:
        classifications = [
            "RECURRENCE_AFTER_DEPARTURE_ORDER_NOT_SUPPORTED",
            "MEMBERSHIP_FREQUENCY_SUFFICIENT",
            "NOT_PROMOTABLE_AS_CONFIRMED",
        ]
        next_theme = "REPEATED_CROSS_GENERATION_RECURRENCE_COMMITTOR"
    else:
        classifications = [
            "RECURRENCE_AFTER_DEPARTURE_NOT_COMMITTOR_COMPATIBLE",
            "PROCESS_TARGET_REQUIRES_REDEFINITION",
            "NOT_PROMOTABLE_AS_CONFIRMED",
        ]
        next_theme = "REPEATED_CROSS_GENERATION_RECURRENCE_COMMITTOR"
    return gates, classifications, next_theme


def benchmark_projection() -> dict[str, Any]:
    prior_runtime = json.loads((L39_ROOT / "runtime_manifest.json").read_text())
    source_wall = float(prior_runtime["wallSeconds"])
    projected_wall = source_wall * 1.25
    projected_cpu = max(source_wall * WORKERS / 3600 * 1.25, 0.0)
    return {
        "schema": "eidosoma.e01.s19_l40.benchmark_projection.v1",
        "status": "PASS"
        if projected_cpu <= 90 and projected_wall <= 64.8 * 3600
        else "STOP_BEFORE_OUTCOME",
        "sourceLoop": "S19-L39",
        "sourceWallSeconds": source_wall,
        "projectedWallSecondsIncludingRegeneration": projected_wall,
        "projectedCpuHoursIncludingRegeneration": projected_cpu,
        "newMatrices": 0,
        "newTrajectories": 0,
        "newBranchStreams": 0,
        "scientificOutcomeRetained": False,
    }


def make_figures(
    anchors: pd.DataFrame,
    states: pd.DataFrame,
    reliability: pd.DataFrame,
    transfers: pd.DataFrame,
    hazards: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    root = BUILD_ROOT / "figures"
    root.mkdir(parents=True, exist_ok=True)

    def save(name: str) -> None:
        plt.tight_layout()
        plt.savefig(root / name, dpi=180)
        plt.close()

    anchors[anchors["targetId"].eq(PRIMARY_TARGET)].groupby(
        ["evaluationCohort", "candidateId"], as_index=False
    ).agg(
        anchorGenerationMean=("anchorGeneration", "mean"),
        anchorClockMean=("anchorSelectedClockIndex", "mean"),
    ).set_index(["evaluationCohort", "candidateId"]).plot(kind="bar", figsize=(13, 6))
    plt.ylabel("Frozen observed-prefix anchor location")
    save("01_prefix_anchor_geometry.png")

    h32 = states[
        states["branchFamily"].eq("H32") & states["targetId"].eq(PRIMARY_TARGET)
    ]
    for (cohort, candidate), group in h32.groupby(["evaluationCohort", "candidateId"], sort=True):
        plt.hist(group["qHat"], bins=np.linspace(0, 1, 21), alpha=0.45, label=f"{cohort}/{candidate}")
    plt.xlabel("H32 recurrence-after-departure probability")
    plt.ylabel("States")
    plt.legend(fontsize=6)
    save("02_return_committor_distributions.png")

    reliability[
        reliability["evaluationCohort"].isin(EVALUATION_COHORTS)
        & reliability["targetId"].eq(PRIMARY_TARGET)
    ].pivot_table(
        index="branchFamily",
        columns=["evaluationCohort", "candidateId"],
        values="splitHalfSpearman",
    ).plot(kind="bar", figsize=(14, 6))
    plt.axhline(0.5, color="black", linestyle="--")
    plt.ylabel("Split-half Spearman")
    save("03_return_committor_reliability.png")

    transfers[
        transfers["evaluationCohort"].isin(EVALUATION_COHORTS)
        & transfers["comparisonId"].isin(
            [
                "H8_RETURN_PROGRESS_VS_H32_PRIMARY",
                "L39_H8_INHERITANCE_PROPENSITY_VS_H32_PRIMARY",
                "H32_PRIMARY_MINUS_PERMUTED_ANCHOR",
                "H32_PRIMARY_MINUS_UNRELATED_ANCHOR",
                "H32_PRIMARY_MINUS_ORDER_NULL",
                "CURRENT_MASS_VS_H32_PRIMARY",
            ]
        )
    ].pivot_table(
        index="comparisonId",
        columns=["evaluationCohort", "candidateId"],
        values="pointEstimate",
    ).plot(kind="bar", figsize=(15, 7))
    plt.axhline(0, color="black", linewidth=1)
    plt.ylabel("Registered rank or paired probability difference")
    save("04_short_shooting_and_anchor_controls.png")

    hazard_view = hazards[
        hazards["evaluationCohort"].isin(EVALUATION_COHORTS)
        & hazards["branchFamily"].eq("H32")
    ]
    for (cohort, candidate), group in hazard_view.groupby(["evaluationCohort", "candidateId"]):
        plt.plot(
            group["futureBoundaryOneBased"],
            group["cumulativeCertificationIncidence"],
            marker="o",
            label=f"{cohort}/{candidate}",
        )
    plt.xlabel("Future post-fission boundary")
    plt.ylabel("Cumulative certified-return incidence")
    plt.legend(fontsize=7)
    save("05_online_return_hazard.png")

    checks = [
        "primaryH32Reliable",
        "h8ReturnProgressTransferPassed",
        "permutedAnchorControlPassed",
        "unrelatedAnchorControlPassed",
        "sequenceOrderControlPassed",
        "opportunityNondegeneracyPassed",
        "recurrenceAfterDepartureTargetPassed",
        "shortShootingCoordinatePassed",
    ]
    matrix = gates.set_index(["evaluationCohort", "candidateId"])[checks].astype(float)
    plt.figure(figsize=(14, 5))
    plt.imshow(matrix, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    plt.xticks(range(len(checks)), checks, rotation=35, ha="right", fontsize=7)
    plt.yticks(range(len(matrix)), ["/".join(index) for index in matrix.index], fontsize=7)
    plt.colorbar(ticks=[0, 1])
    save("06_decision_matrix.png")


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
        "schema": "eidosoma.e01.s19_l40.artifact_manifest.v1",
        "root": str(root),
        "fileCount": len(files),
        "totalBytes": sum(item["bytes"] for item in files),
        "files": files,
    }


def append_ledgers(classifications: list[str], timestamp: str, next_theme: str) -> None:
    ledger_path = ARTIFACT_ROOT / "self_improvement_ledger.parquet"
    ledger = pd.read_parquet(ledger_path)
    sequence = int(ledger["ledgerSequence"].max()) + 1
    additions = [
        {
            "appendOnly": True,
            "beliefBeforeLoop": "L39 showed that inheritance streaks reflect marginal inheritance frequency rather than distinct temporal ordering.",
            "failureOrAmbiguityTargeted": "Whether the system has a genuine capacity to return after leaving a past-defined compositional neighborhood.",
            "informationGainRationale": "A future departure followed by return excludes ordinary adjacent smoothness and requires no completed trajectory.",
            "learned": "L40 latest-prefix-anchor departure/return event locked before outcomes.",
            "ledgerSequence": sequence,
            "loopId": LOOP_ID,
            "motivatingEvidence": "Complete L39 result and reviewer process-first recommendation.",
            "proposedNextTest": "Rescore exact H32/H8 paths against one past-defined return event.",
            "recordPhase": "PRE_LOOP_METHOD_LOCK",
            "remainingPlausibleHypotheses": "Return after departure, repeated cross-generation recurrence, fission-conditioned homeostasis, or shooting-only estimation.",
            "selectedHypotheses": "The latest observed post-fission state supplies a past-defined homeostatic return anchor.",
            "timestampUtc": timestamp,
            "weakenedHypotheses": "High parent/daughter inheritance frequency alone demonstrates sustained organization.",
        },
        {
            "appendOnly": True,
            "beliefBeforeLoop": "A clean return process must be committor-compatible, anchor-specific, sequence-specific and recoverable by a short target-mechanics coordinate.",
            "failureOrAmbiguityTargeted": "Committor compatibility and H8 recoverability of recurrence after departure.",
            "informationGainRationale": "Exact existing streams and frozen anchor controls isolate return dynamics without future target construction.",
            "learned": ";".join(classifications),
            "ledgerSequence": sequence + 1,
            "loopId": LOOP_ID,
            "motivatingEvidence": "Complete L40 result.",
            "proposedNextTest": next_theme,
            "recordPhase": "POST_LOOP_RESULT_AUTONOMOUS_CONTINUATION",
            "remainingPlausibleHypotheses": next_theme,
            "selectedHypotheses": "Online recurrence after departure from a past-defined anchor.",
            "timestampUtc": timestamp,
            "weakenedHypotheses": "Any high-H return pattern is necessarily anchor-specific and temporally organized.",
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
        + f"\n\n## {LOOP_ID} — online recurrence after departure\n\n"
        + f"- **Learned:** {', '.join(classifications)}.\n"
        + f"- **Next:** {next_theme}.\n",
    )

    candidates_path = ARTIFACT_ROOT / "candidate_registry.parquet"
    candidates = pd.read_parquet(candidates_path)
    candidate = {
        "branchCount": 1,
        "bundleId": "L40_RECURRENCE_AFTER_DEPARTURE",
        "candidateId": "S19-L40-LATEST-PREFIX-ANCHOR-RETURN-H090",
        "candidateSpecificSuccess": 0,
        "completedFitLeakage": 0,
        "computeEfficiency": 5,
        "crossCandidateDiscriminability": 5,
        "deterministicHReuse": 0,
        "explanatoryLeverage": 5,
        "frozenRank": 1,
        "independenceFromPriorOutcomeSelection": 5,
        "outcomeGuidedThresholdSelection": 0,
        "paperFingerprintSpecificity": 0,
        "proposedSpecification": "future post-fission departure then strict-H090 return to latest observed post-fission anchor",
        "rankingScore": 30.0,
        "registryOrder": int(candidates["registryOrder"].max()) + 1,
        "selected": True,
        "selectionReason": "L39_ORDER_NULL_AND_REVIEWER_RECURRENCE_AFTER_DEPARTURE",
        "sourceGrounding": 5,
        "testability": 5,
        "undefinedAuthorSemantics": 0,
    }
    BASE.write_parquet(
        candidates_path,
        pd.concat(
            [candidates, pd.DataFrame([candidate]).reindex(columns=candidates.columns)],
            ignore_index=True,
        ),
    )

    source_path = ARTIFACT_ROOT / "source_search_ledger.parquet"
    sources = pd.read_parquet(source_path)
    source_rows = [
        {
            "commitOrVersion": None,
            "evidenceClass": source.evidenceClass,
            "finding": f"{source.finding}; L40 use: {source.frozenUse}",
            "licenseStatus": "WORKSPACE_OR_HUMAN_DIRECTION",
            "redistributionStatus": "INTERNAL_EVIDENCE_ONLY",
            "repositoryIdentity": None,
            "retainedPath": None,
            "retrievalDate": timestamp[:10],
            "sha256": None,
            "sourceId": f"L40_{source.sourceId}",
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
            "selectedDiscoveryLead": (
                "RECURRENCE_AFTER_DEPARTURE_PROCESS"
                if "PROMOTABLE_TO_UNTOUCHED_PROCESS_CONFIRMATION" in classifications
                else None
            ),
            "newMatrices": 0,
            "newTrajectories": 0,
            "newBranchStreams": 0,
            "nextStepActive": True,
        }
    )
    registry["proposedNextLoopTheme"] = next_theme
    registry["proposedNextLoopActive"] = True
    registry["authorizationUpperBound"] = "S19-L55"
    BASE.atomic_text(registry_path, yaml.safe_dump(registry, sort_keys=False))

    history_path = ARTIFACT_ROOT / "human_review_history.json"
    history = json.loads(history_path.read_text())
    history["history"].append(
        {
            "decision": "S19_L40_COMPLETE_AUTONOMOUS_CONTINUATION",
            "loopId": LOOP_ID,
            "nextLoopAuthorized": True,
            "recordedAtUtc": timestamp,
            "result": classifications,
            "s20Activated": False,
            "scope": VERSION,
            "source": "locked_execution_result",
        }
    )
    history["pendingDecision"] = "NONE_AUTONOMOUS_SEQUENCE_ACTIVE_THROUGH_L55"
    BASE.write_json(history_path, history)


def report_text(
    anchors: pd.DataFrame,
    states: pd.DataFrame,
    reliability: pd.DataFrame,
    transfers: pd.DataFrame,
    hazards: pd.DataFrame,
    gates: pd.DataFrame,
    classifications: list[str],
    runtime: dict[str, Any],
    next_theme: str,
) -> str:
    evaluation_reliability = reliability[
        reliability["evaluationCohort"].isin(EVALUATION_COHORTS)
    ]
    key_transfers = transfers[
        transfers["evaluationCohort"].isin(EVALUATION_COHORTS)
        & transfers["comparisonId"].isin(
            [
                "H8_RETURN_PROGRESS_VS_H32_PRIMARY",
                "H8_EVENT_Q_VS_H32_PRIMARY",
                "L39_H8_INHERITANCE_PROPENSITY_VS_H32_PRIMARY",
                "H32_PRIMARY_MINUS_PERMUTED_ANCHOR",
                "H32_PRIMARY_MINUS_UNRELATED_ANCHOR",
                "H32_PRIMARY_MINUS_ORDER_NULL",
                "PREFIX_INHERITANCE_FRACTION_VS_H32_PRIMARY",
                "CURRENT_MASS_VS_H32_PRIMARY",
                "GENERATION_PHASE_VS_H32_PRIMARY",
            ]
        )
    ]
    state_summary = states.groupby(
        ["evaluationCohort", "candidateId", "branchFamily", "targetId"], as_index=False
    ).agg(
        meanQ=("qHat", "mean"),
        meanDeparture=("qDeparture", "mean"),
        meanMixedOpportunity=("qMixedMembershipOpportunity", "mean"),
        meanReturnProgress=("meanReturnProgress", "mean"),
        meanOrderNull=("meanExactOrderNullProbability", "mean"),
        meanOpportunityWithoutCertification=("opportunityWithoutCertificationFraction", "mean"),
    )
    anchor_summary = anchors.groupby(
        ["evaluationCohort", "candidateId", "targetId"], as_index=False
    ).agg(
        anchors=("stateId", "size"),
        meanAnchorGeneration=("anchorGeneration", "mean"),
        meanAnchorClock=("anchorSelectedClockIndex", "mean"),
    )
    hazard_summary = hazards[
        hazards["evaluationCohort"].isin(EVALUATION_COHORTS)
        & hazards["branchFamily"].eq("H32")
    ]
    return f"""# S19-L40 — Online Recurrence After a Certified Departure

## Chief/human handoff

- **Step:** `{VERSION}`
- **Status:** complete under the extended L19–L55 autonomous sequence.
- **Classifications:** {", ".join(f"`{value}`" for value in classifications)}
- **Validation:** immutable-prior and anchor replay; exact numerical/discrete/path replay of all 53,760 frozen H32/H8 streams; seven fixtures; candidate-separated split-half reliability; three fixed anchor/order controls; 4,096 catalytic-matrix bootstraps; independent full regeneration; runtime/storage/artifact hashes.
- **Recommended next action:** `{next_theme}`.

## Frozen question

Does the simulator have a reliable state-dependent probability of leaving and then returning to a compositional neighborhood fixed entirely from the observed past? The sole primary anchor is the latest selected post-fission composition before the restored state. Departure is the first future post-fission boundary with `H<=0.9`; online certification is the first later future boundary with strict `H>0.9` to that same anchor.

This cannot be satisfied by ordinary adjacent smoothness: a trajectory that remains near the anchor is never positive. No completed trajectory, completed-run centroid, future-defined basin, threshold variant, anchor search, or horizon search is used.

## Frozen anchors

{anchor_summary.to_markdown(index=False)}

## Process probability and opportunity

{state_summary.to_markdown(index=False)}

## Committor reliability

{evaluation_reliability.to_markdown(index=False)}

## H8 coordinate and controls

{key_transfers.to_markdown(index=False)}

The primary short coordinate is mean maximum post-departure H over the frozen H8 branches, with zero assigned when no departure occurs. Molecule-permuted and unrelated-matrix anchors test specificity. The exact order null fixes each branch's near/departed counts and randomizes only their ordering.

## Online return hazard

{hazard_summary.to_markdown(index=False)}

Post-fission certification is never projected onto intervening molecular observations. Molecular offsets remain named diagnostics only.

## Scientific gates

{gates.to_markdown(index=False)}

## Validation and provenance

- Repository commit: `{runtime['repositoryHead']}`.
- Workers: `{runtime['workers']}`; one numerical-library thread per worker.
- Wall time: `{runtime['wallSeconds']:.3f}` seconds.
- New matrices/trajectories/branch streams: `0/0/0`.
- All scientific frames and paths were independently regenerated from the lock.
- Every S01–S18 and S19-L01–L39 artifact remains immutable.

## Interpretation boundary

A positive result would establish only a simulator-defined return process and a conditional stochastic-shooting coordinate. It would not identify the paper label, an author implementation, a static observed biomarker, causal emergence, intervention efficacy, biological replication, or causal control.

## Next boundary

L40 is frozen. The standing human authorization permits `{next_theme}` as the next bounded loop through L55. S20, E02, author contact, interventions and report-bundle generation remain inactive.
"""


def prepare_lock() -> None:
    LOOP_ROOT.mkdir(parents=True, exist_ok=True)
    if git("status", "--porcelain=v1"):
        raise RuntimeError("repository must be clean before L40 lock")
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/eidosoma/groups/42")
    if head != remote:
        raise RuntimeError("local/remote commit mismatch")
    prior = validate_immutable_prior()
    fixtures = fixture_results()
    anchors = anchor_registry()
    seeds = analysis_seed_manifest()
    firewall = seed_firewall(seeds)
    benchmark = benchmark_projection()
    if (
        not prior["unchanged"]
        or not fixtures["passed"].all()
        or firewall["status"] != "PASS"
        or benchmark["status"] != "PASS"
    ):
        raise RuntimeError("L40 preoutcome validation or benchmark failed")
    shutil.copy2(CONFIG, LOOP_ROOT / "preregistration.yaml")
    BASE.atomic_text(
        LOOP_ROOT / "decision_record.md",
        "# S19-L40 decision record\n\n"
        "L39 showed that a reliable run-of-three inheritance probability was almost perfectly explained by the marginal inherited/non-inherited counts under an exact temporal-order null. The reviewer recommended defining process outcomes that cannot reduce to adjacent smoothness, separating molecular and generational clocks, and establishing a branch-half-reliable committor before predictor search. L40 freezes the latest selected post-fission composition already visible in each state prefix as the sole primary anchor. A future post-fission boundary must first leave its strict-H>0.9 neighborhood, and a later post-fission boundary must return before the event is certified. The anchor, threshold, H32/H8 horizons, candidates and stochastic paths are fixed. Molecule-permuted and unrelated-matrix anchors, exact fixed-count order, mixed-state opportunity, prefix heredity, mass and phase are controls. No completed test trajectory or future-defined target enters the event.\n",
    )
    BASE.write_json(LOOP_ROOT / "immutable_prior_validation.json", prior)
    BASE.write_parquet(LOOP_ROOT / "fixture_results.parquet", fixtures)
    BASE.write_parquet(LOOP_ROOT / "anchor_registry.parquet", anchors)
    for name in (
        "response_registry.parquet",
        "original_target_coordinates.parquet",
        "input_trajectory_manifest.parquet",
        "prefix_boundary_registry.parquet",
        "prefix_state_summary.parquet",
    ):
        shutil.copy2(L39_ROOT / name, LOOP_ROOT / name)
    for name in ("unrelated_control_map.parquet", "species_permutation_manifest.parquet"):
        shutil.copy2(L38_ROOT / name, LOOP_ROOT / name)
    BASE.write_parquet(LOOP_ROOT / "analysis_seed_manifest.parquet", seeds)
    BASE.write_json(LOOP_ROOT / "seed_firewall.json", firewall)
    BASE.write_json(LOOP_ROOT / "benchmark_projection.json", benchmark)
    BASE.write_parquet(
        LOOP_ROOT / "source_grounding_registry.parquet", source_grounding_registry()
    )
    hashes = {
        "configSha256": sha256_file(CONFIG),
        "responsesSha256": sha256_file(LOOP_ROOT / "response_registry.parquet"),
        "coordinatesSha256": sha256_file(LOOP_ROOT / "original_target_coordinates.parquet"),
        "manifestSha256": sha256_file(LOOP_ROOT / "input_trajectory_manifest.parquet"),
        "boundariesSha256": sha256_file(LOOP_ROOT / "prefix_boundary_registry.parquet"),
        "summariesSha256": sha256_file(LOOP_ROOT / "prefix_state_summary.parquet"),
        "anchorsSha256": sha256_file(LOOP_ROOT / "anchor_registry.parquet"),
        "donorsSha256": sha256_file(LOOP_ROOT / "unrelated_control_map.parquet"),
        "permutationsSha256": sha256_file(LOOP_ROOT / "species_permutation_manifest.parquet"),
        "seedsSha256": sha256_file(LOOP_ROOT / "analysis_seed_manifest.parquet"),
        "firewallSha256": sha256_file(LOOP_ROOT / "seed_firewall.json"),
        "benchmarkSha256": sha256_file(LOOP_ROOT / "benchmark_projection.json"),
        "l39ManifestSha256": sha256_file(L39_ROOT / "artifact_manifest.json"),
    }
    BASE.write_json(
        LOOP_ROOT / "implementation_lock.json",
        {
            "schema": "eidosoma.e01.s19_l40.implementation_lock.v1",
            "repositoryHead": head,
            "remoteHead": remote,
            "runnerSha256": sha256_file(RUNNER_PATH),
            "coreSha256": sha256_file(CORE_PATH),
            "threshold": THRESHOLD,
            "thresholdComparison": "RETURN_STRICT_GREATER_DEPARTURE_COMPLEMENT",
            "anchor": "LATEST_OBSERVED_PREFIX_POST_FISSION_COMPOSITION",
            "targetIds": list(TARGETS),
            "shortCoordinate": "H8_MEAN_MAXIMUM_POSTDEPARTURE_H_ZERO_WITHOUT_DEPARTURE",
            "branchFamilies": list(FAMILIES),
            "horizons": HORIZONS,
            "branchCounts": BRANCH_COUNTS,
            "matrixBootstraps": BOOTSTRAPS,
            "newMatrices": 0,
            "newTrajectories": 0,
            "newBranchStreams": 0,
            "completedTestTrajectoryUsedByScientificTarget": False,
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
        raise RuntimeError("L40 repository lock mismatch")
    prior = validate_immutable_prior()
    fixtures = fixture_results()
    locked_files = {
        "responsesSha256": LOOP_ROOT / "response_registry.parquet",
        "coordinatesSha256": LOOP_ROOT / "original_target_coordinates.parquet",
        "manifestSha256": LOOP_ROOT / "input_trajectory_manifest.parquet",
        "boundariesSha256": LOOP_ROOT / "prefix_boundary_registry.parquet",
        "summariesSha256": LOOP_ROOT / "prefix_state_summary.parquet",
        "anchorsSha256": LOOP_ROOT / "anchor_registry.parquet",
        "donorsSha256": LOOP_ROOT / "unrelated_control_map.parquet",
        "permutationsSha256": LOOP_ROOT / "species_permutation_manifest.parquet",
        "seedsSha256": LOOP_ROOT / "analysis_seed_manifest.parquet",
        "firewallSha256": LOOP_ROOT / "seed_firewall.json",
        "benchmarkSha256": LOOP_ROOT / "benchmark_projection.json",
        "l39ManifestSha256": L39_ROOT / "artifact_manifest.json",
    }
    for key, path in locked_files.items():
        if sha256_file(path) != lock[key]:
            raise RuntimeError(f"L40 locked input changed: {path}")
    if (
        not prior["unchanged"]
        or prior["aggregateSha256"] != lock["priorAggregateSha256"]
        or not fixtures["passed"].all()
        or sha256_file(RUNNER_PATH) != lock["runnerSha256"]
        or sha256_file(CORE_PATH) != lock["coreSha256"]
    ):
        raise RuntimeError("L40 pre-execution validation failed")
    responses = pd.read_parquet(LOOP_ROOT / "response_registry.parquet")
    boundaries = pd.read_parquet(LOOP_ROOT / "prefix_boundary_registry.parquet")
    summaries = pd.read_parquet(LOOP_ROOT / "prefix_state_summary.parquet")
    anchors = pd.read_parquet(LOOP_ROOT / "anchor_registry.parquet")
    prefixes = L39.prefix_controls(boundaries, summaries)
    l39_states = pd.read_parquet(L39_ROOT / "state_committor_results.parquet")
    payloads = build_payloads(anchors)

    if BUILD_ROOT.exists():
        shutil.rmtree(BUILD_ROOT)
    BUILD_ROOT.mkdir(parents=True)
    outcomes, compact = execute_branches(payloads)
    compact_validation = L36.compact_replay_validation(compact)
    states = state_committor_results(outcomes)
    reliability, reliability_bootstrap = reliability_results(states)
    pairs = transfer_pairs(states, responses, prefixes, l39_states)
    transfers, transfer_bootstrap = transfer_results(pairs)
    hazards = boundary_hazard_results(outcomes)
    gates, classifications, next_theme = scientific_gates(reliability, transfers, states)
    make_figures(anchors, states, reliability, transfers, hazards, gates)

    for name in (
        "preregistration.yaml",
        "decision_record.md",
        "immutable_prior_validation.json",
        "fixture_results.parquet",
        "anchor_registry.parquet",
        "response_registry.parquet",
        "original_target_coordinates.parquet",
        "input_trajectory_manifest.parquet",
        "prefix_boundary_registry.parquet",
        "prefix_state_summary.parquet",
        "unrelated_control_map.parquet",
        "species_permutation_manifest.parquet",
        "analysis_seed_manifest.parquet",
        "seed_firewall.json",
        "benchmark_projection.json",
        "source_grounding_registry.parquet",
        "implementation_lock.json",
        "preoutcome_repository_lock.json",
    ):
        shutil.copy2(LOOP_ROOT / name, BUILD_ROOT / name)
    tables = {
        "prefix_process_control_results.parquet": prefixes,
        "branch_outcome_results.parquet": outcomes,
        "compact_branch_replay.parquet": compact,
        "compact_replay_validation.parquet": compact_validation,
        "state_committor_results.parquet": states,
        "committor_reliability_results.parquet": reliability,
        "committor_reliability_bootstrap.parquet": reliability_bootstrap,
        "transfer_pairs.parquet": pairs,
        "transfer_results.parquet": transfers,
        "transfer_bootstrap.parquet": transfer_bootstrap,
        "boundary_hazard_results.parquet": hazards,
        "scientific_gate_results.parquet": gates,
    }
    for name, frame in tables.items():
        BASE.write_parquet(BUILD_ROOT / name, frame)
    BASE.write_json(
        BUILD_ROOT / "classification.json",
        {
            "schema": "eidosoma.e01.s19_l40.classification.v1",
            "classifications": classifications,
            "recurrenceAfterDepartureCommittorEstablished": bool(
                gates["recurrenceAfterDepartureTargetPassed"].all()
            ),
            "shortShootingCoordinateEstablished": bool(
                gates["shortShootingCoordinatePassed"].all()
            ),
            "onlineCertificationPrimary": True,
            "targetUsesCompletedTestTrajectory": False,
            "priorStatusesChanged": False,
        },
    )
    pd.DataFrame(
        columns=[
            "stage",
            "stateId",
            "candidateId",
            "matrixIndex",
            "branchFamily",
            "targetId",
            "branchIndex",
            "exceptionClass",
            "exceptionMessage",
        ]
    ).to_csv(BUILD_ROOT / "failure_ledger.csv", index=False)

    replay_outcomes, replay_compact = execute_branches(payloads)
    replay_compact_validation = L36.compact_replay_validation(replay_compact)
    replay_states = state_committor_results(replay_outcomes)
    replay_reliability, replay_reliability_bootstrap = reliability_results(replay_states)
    replay_pairs = transfer_pairs(replay_states, responses, prefixes, l39_states)
    replay_transfers, replay_transfer_bootstrap = transfer_results(replay_pairs)
    replay_hazards = boundary_hazard_results(replay_outcomes)
    replay_gates, replay_classifications, replay_next = scientific_gates(
        replay_reliability, replay_transfers, replay_states
    )
    replay_tables = {
        "outcomes": (outcomes, replay_outcomes),
        "compact": (compact, replay_compact),
        "compactValidation": (compact_validation, replay_compact_validation),
        "states": (states, replay_states),
        "reliability": (reliability, replay_reliability),
        "reliabilityBootstrap": (reliability_bootstrap, replay_reliability_bootstrap),
        "pairs": (pairs, replay_pairs),
        "transfers": (transfers, replay_transfers),
        "transferBootstrap": (transfer_bootstrap, replay_transfer_bootstrap),
        "hazards": (hazards, replay_hazards),
        "gates": (gates, replay_gates),
    }
    checks = {
        name: frame_hash(left) == frame_hash(right)
        for name, (left, right) in replay_tables.items()
    }
    checks.update(
        {
            "classificationExact": classifications == replay_classifications,
            "nextThemeExact": next_theme == replay_next,
            "fixturesPassed": bool(fixtures["passed"].all()),
            "immutablePriorPassed": prior["unchanged"],
            "seedFirewallPassed": json.loads((LOOP_ROOT / "seed_firewall.json").read_text())[
                "status"
            ]
            == "PASS",
            "compactReplayPassed": bool(compact_validation["allPassed"].all()),
            "targetUsesNoCompletedTestTrajectory": bool(
                (~outcomes["targetUsesCompletedTestTrajectory"]).all()
            ),
            "returnAfterDepartureInvariant": bool(
                (
                    outcomes.loc[outcomes["event"], "certificationBoundaryOneBased"]
                    > outcomes.loc[outcomes["event"], "departureBoundaryOneBased"]
                ).all()
            ),
            "noNewTrajectory": True,
            "noNewBranchStream": True,
        }
    )
    if not all(checks.values()):
        raise RuntimeError(f"L40 regeneration validation failed: {checks}")
    BASE.write_json(
        BUILD_ROOT / "regeneration_validation.json",
        {
            "schema": "eidosoma.e01.s19_l40.regeneration_validation.v1",
            "status": "PASS",
            "checks": checks,
            "outcomeFrameSha256": frame_hash(outcomes),
            "stateFrameSha256": frame_hash(states),
            "gateFrameSha256": frame_hash(gates),
        },
    )
    runtime = {
        "schema": "eidosoma.e01.s19_l40.runtime.v1",
        "repositoryHead": git("rev-parse", "HEAD"),
        "workers": WORKERS,
        "numericalLibraryThreadsPerWorker": 1,
        "gpuHours": 0,
        "wallSeconds": time.perf_counter() - started,
        "controllerCpuHours": (time.process_time() - started_cpu) / 3600,
        "states": 280,
        "uniqueFrozenBranchStreamsScored": 53_760,
        "targetScoresPerBranch": len(TARGETS),
        "newMatrices": 0,
        "newTrajectories": 0,
        "newBranchStreams": 0,
        "completedAtUtc": utc_now(),
    }
    BASE.write_json(BUILD_ROOT / "runtime_manifest.json", runtime)
    retained = sum(path.stat().st_size for path in BUILD_ROOT.rglob("*") if path.is_file())
    temporary = sum(path.stat().st_size for path in CACHE_ROOT.rglob("*") if path.is_file())
    storage = {
        "schema": "eidosoma.e01.s19_l40.storage_validation.v1",
        "retainedBytes": retained,
        "retainedGiBCeiling": 25,
        "temporaryBytes": temporary,
        "temporaryGiBCeiling": 75,
        "status": "PASS"
        if retained < 25 * 2**30 and temporary < 75 * 2**30
        else "FAIL",
    }
    if storage["status"] != "PASS":
        raise RuntimeError("L40 storage ceiling exceeded")
    BASE.write_json(BUILD_ROOT / "storage_validation.json", storage)
    report = report_text(
        anchors,
        states,
        reliability,
        transfers,
        hazards,
        gates,
        classifications,
        runtime,
        next_theme,
    )
    BASE.atomic_text(BUILD_ROOT / "research_step_full_results.md", report)
    BASE.atomic_text(BUILD_ROOT / "S19_L40_FULL_RESULTS.md", report)
    BASE.atomic_text(
        BUILD_ROOT / "loop_decision_summary.md",
        "# S19-L40 decision summary\n\n"
        + f"**Classification:** {', '.join(classifications)}\n\n"
        + f"**All-group return target:** `{gates['recurrenceAfterDepartureTargetPassed'].all()}`.\n\n"
        + f"**All-group H8 progress coordinate:** `{gates['shortShootingCoordinatePassed'].all()}`.\n\n"
        + f"**Next:** `{next_theme}`.\n",
    )
    BASE.write_json(BUILD_ROOT / "artifact_manifest.json", manifest_for(BUILD_ROOT))
    stage = LOOP_ROOT.with_name(".L40-promotion-stage")
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
        raise RuntimeError("L40 artifact hash validation failed")
    append_ledgers(classifications, runtime["completedAtUtc"], next_theme)
    BASE.atomic_text(ARTIFACT_ROOT / "research_step_full_results.md", report)
    BASE.atomic_text(
        ARTIFACT_ROOT / "S19_CURRENT_HANDOFF.md",
        report.replace("# S19-L40", "# S19 current handoff — S19-L40", 1),
    )
    BASE.write_json(
        ARTIFACT_ROOT / "s19_status.json",
        {
            "schema": "eidosoma.e01.s19.status.v1",
            "status": "ACTIVE_AUTONOMOUS_SEQUENCE",
            "latestCompletedLoop": LOOP_ID,
            "latestClassification": classifications,
            "selectedDiscoveryLead": (
                "RECURRENCE_AFTER_DEPARTURE_PROCESS"
                if "PROMOTABLE_TO_UNTOUCHED_PROCESS_CONFIRMATION" in classifications
                else None
            ),
            "nextAuthorizedLoop": "S19-L41",
            "authorizationUpperBound": "S19-L55",
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
