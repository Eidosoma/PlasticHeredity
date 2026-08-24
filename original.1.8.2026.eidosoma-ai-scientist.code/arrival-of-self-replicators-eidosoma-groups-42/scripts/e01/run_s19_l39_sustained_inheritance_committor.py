"""Execute S19-L39 sustained parent/daughter inheritance committor audit."""

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

from e01_onset_discovery.sustained_inheritance import (
    exact_order_null_probability,
    maximum_true_run,
    score_sustained_inheritance,
)


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


L38 = _load_module(
    "e01_s19_l39_l38",
    REPO_ROOT / "scripts/e01/run_s19_l38_recurrence_inheritance_outcome.py",
)
L37 = L38.L37
L36 = L38.L36
L28 = L38.L28
BASE = L38.BASE
RestoredState = L38.RestoredState
corrected_between_state_variance = L38.corrected_between_state_variance

LOOP_ID = "S19-L39"
VERSION = "E01-S19-L39-SUSTAINED-HOMEOSTATIC-INHERITANCE-COMMITTOR-v1.0.0"
CANDIDATES = L38.CANDIDATES
COHORTS = L38.COHORTS
EVALUATION_COHORTS = L38.EVALUATION_COHORTS
FAMILIES = L38.FAMILIES
HORIZONS = L38.HORIZONS
BRANCH_COUNTS = L38.BRANCH_COUNTS
HALVES = L38.HALVES
THRESHOLD = 0.9
REQUIRED_RUN = 3
BOOTSTRAPS = 4096
ROOT_HEX = "c9a0c3d4d6aa540c5eeae53f1a827b3564fa7836765c56b823a3d2ae8b672a0a"
PHASE = "s19_l39_sustained_inheritance"
WORKERS = min(8, os.cpu_count() or 1)

ARTIFACT_ROOT = Path("/artifacts/research_steps/S19")
LOOP_ROOT = ARTIFACT_ROOT / "loops/L39"
L38_ROOT = ARTIFACT_ROOT / "loops/L38"
CACHE_ROOT = Path("/cache/e01_s19_l39")
BUILD_ROOT = CACHE_ROOT / "build"
CONFIG = REPO_ROOT / "configs/e01/s19_l39_sustained_inheritance.yaml"
RUNNER_PATH = Path(__file__)
CORE_PATH = REPO_ROOT / "src/e01_onset_discovery/sustained_inheritance.py"


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
    material = "\x1f".join([VERSION, ROOT_HEX, *map(str, parts)]).encode()
    return hashlib.sha256(material).hexdigest()


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
    inherited = json.loads((L38_ROOT / "immutable_prior_validation.json").read_text())
    rows = list(inherited["files"])
    manifest = json.loads((L38_ROOT / "artifact_manifest.json").read_text())
    rows.extend(
        {
            "path": str(L38_ROOT / item["path"]),
            "root": str(L38_ROOT),
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
        "schema": "eidosoma.e01.s19_l39.immutable_prior_validation.v1",
        "status": "PASS" if passed else "FAIL",
        "unchanged": passed,
        "fileCount": len(checked),
        "aggregateSha256": aggregate,
        "l38ManifestSha256": sha256_file(L38_ROOT / "artifact_manifest.json"),
        "files": checked,
    }


def fixture_results() -> pd.DataFrame:
    values = np.asarray([False, True, True, True, False], dtype=np.bool_)
    result = score_sustained_inheritance(
        inherited=values,
        generations=np.arange(10, 15, dtype=np.int64),
        offsets_one_based=np.asarray([2, 4, 6, 8, 10], dtype=np.int64),
        required_run=REQUIRED_RUN,
    )
    broken = score_sustained_inheritance(
        inherited=np.asarray([True, True, False, True, True], dtype=np.bool_),
        generations=np.arange(1, 6, dtype=np.int64),
        offsets_one_based=np.arange(1, 6, dtype=np.int64),
        required_run=REQUIRED_RUN,
    )
    replay = score_sustained_inheritance(
        inherited=values.copy(),
        generations=np.arange(10, 15, dtype=np.int64),
        offsets_one_based=np.asarray([2, 4, 6, 8, 10], dtype=np.int64),
        required_run=REQUIRED_RUN,
    )
    return pd.DataFrame(
        [
            {
                "fixtureId": "ONLINE_CERTIFICATION_AFTER_THIRD_FISSION",
                "passed": result.event and result.certification_boundary_one_based == 4,
                "details": "certification occurs only when the third consecutive inherited fission is observed",
            },
            {
                "fixtureId": "RETROSPECTIVE_PHYSICAL_ONSET_SEPARATE",
                "passed": result.retrospective_onset_boundary_one_based == 2
                and result.certification_offset_one_based == 8
                and result.retrospective_onset_offset_one_based == 4,
                "details": "physical run start is descriptive and never used for online certification",
            },
            {
                "fixtureId": "BROKEN_STREAK_REJECTED",
                "passed": not broken.event and broken.maximum_consecutive_inherited == 2,
                "details": "two pairs separated by failure do not establish sustained heredity",
            },
            {
                "fixtureId": "ORDER_NULL_EXACT",
                "passed": abs(exact_order_null_probability(4, 3, 3) - 0.5) <= 1e-15,
                "details": "two of four fixed-count orders contain a run of three",
            },
            {
                "fixtureId": "NO_PREFIX_CARRY_IN",
                "passed": not score_sustained_inheritance(
                    inherited=np.asarray([True, True], dtype=np.bool_),
                    generations=np.asarray([1, 2], dtype=np.int64),
                    offsets_one_based=np.asarray([1, 2], dtype=np.int64),
                    required_run=REQUIRED_RUN,
                ).event,
                "details": "only future fissions can form the registered run",
            },
            {
                "fixtureId": "EXACT_REPLAY",
                "passed": result == replay,
                "details": "all discrete outputs replay exactly",
            },
            {
                "fixtureId": "FROZEN_SCOPE",
                "passed": FAMILIES == ("H32", "H8")
                and BRANCH_COUNTS == {"H32": 128, "H8": 64}
                and REQUIRED_RUN == 3
                and THRESHOLD == 0.9,
                "details": json.dumps(
                    {
                        "families": FAMILIES,
                        "branchCounts": BRANCH_COUNTS,
                        "requiredRun": REQUIRED_RUN,
                        "threshold": THRESHOLD,
                    }
                ),
            },
        ]
    )


def analysis_seed_manifest() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    comparisons = (
        "H8_INHERITANCE_PROPENSITY_VS_H32_SUSTAINED",
        "H8_EXPECTED_INHERITED_FRACTION_VS_H32_SUSTAINED",
        "H8_SUSTAINED_VS_H32_SUSTAINED",
        "PREFIX_INHERITANCE_FRACTION_VS_H32_SUSTAINED",
        "PREFIX_TRAILING_RUN_VS_H32_SUSTAINED",
        "CURRENT_MASS_VS_H32_SUSTAINED",
        "GENERATION_PHASE_VS_H32_SUSTAINED",
        "H32_FISSION_OPPORTUNITY_VS_H32_SUSTAINED",
        "H32_ORDER_NULL_VS_H32_SUSTAINED",
        "H32_SUSTAINED_MINUS_ORDER_NULL",
    )
    for cohort in COHORTS:
        for candidate in CANDIDATES:
            for family in FAMILIES:
                parts = ("reliability_bootstrap", cohort, candidate, family)
                rows.append(
                    {
                        "purpose": parts[0],
                        "partsJson": json.dumps(parts),
                        "evaluationCohort": cohort,
                        "candidateId": candidate,
                        "branchFamily": family,
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
                        "comparisonId": comparison,
                        "rootHex": ROOT_HEX,
                        "derivedSeed": str(derived_seed(*parts)),
                        "seedMaterialSha256": seed_material_sha256(*parts),
                    }
                )
    result = pd.DataFrame(rows).sort_values(
        ["purpose", "evaluationCohort", "candidateId", "branchFamily", "comparisonId"],
        na_position="last",
    ).reset_index(drop=True)
    if result["derivedSeed"].duplicated().any() or result["seedMaterialSha256"].duplicated().any():
        raise RuntimeError("L39 analysis seed collision")
    return result


def seed_firewall(seeds: pd.DataFrame) -> dict[str, Any]:
    prior_material: set[str] = set()
    prior_derived: set[str] = set()
    for path in ARTIFACT_ROOT.glob("loops/L*/**/*seed*manifest*.parquet"):
        if "/L39/" in str(path):
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
        "schema": "eidosoma.e01.s19_l39.seed_firewall.v1",
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
                "sourceId": "L39_REVIEWER_PROCESS_NOT_DESTINATION",
                "evidenceClass": "HUMAN_REVIEW_DIRECTION",
                "finding": "Replace a completed-run destination with online recurrence, inheritance and homeostasis processes.",
                "frozenUse": "sustained parent/daughter heredity as a distinct process outcome",
            },
            {
                "sourceId": "L39_PAPER_HOMEOSTATIC_GROWTH_FISSION",
                "evidenceClass": "DIRECT_PAPER_LANGUAGE",
                "finding": "The manuscript describes compositional heredity and homeostatic growth across growth-fission generations.",
                "frozenUse": "strict H090 parent/daughter inheritance without claiming author-label identity",
            },
            {
                "sourceId": "L39_L38_INHERITANCE_DIAGNOSTIC",
                "evidenceClass": "DIRECT_FROZEN_E01_RESULT",
                "finding": "L38 inheritance-only H8 committors were reliable across both candidates and both evaluation cohorts.",
                "frozenUse": "predeclared H8 inheritance-propensity predictor",
            },
            {
                "sourceId": "L39_L28_L31_EXACT_BRANCH_STREAMS",
                "evidenceClass": "DIRECT_FROZEN_E01_RESULT",
                "finding": "The exact H32/H8 branches are replayable and candidate separated.",
                "frozenUse": "zero-new-stream process committor audit",
            },
        ]
    )


def build_payloads() -> list[dict[str, Any]]:
    responses = pd.read_parquet(L38_ROOT / "response_registry.parquet")
    coordinates = pd.read_parquet(L38_ROOT / "original_target_coordinates.parquet")
    manifest = pd.read_parquet(L38_ROOT / "input_trajectory_manifest.parquet")
    payloads = L38.L35.payloads(responses, coordinates, manifest)
    if len(payloads) != 280:
        raise RuntimeError("L39 branch payload cardinality failure")
    return payloads


def _branch_worker(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    matrix_index = int(payload["matrixIndex"])
    beta = L28.generate_beta(
        L28.derive_seed(L28.L23_ROOT_HEX, L28.L23_PHASE, "catalytic_matrix", matrix_index)
    )
    if L28.simulator_array_sha256(beta) != payload["betaSha256"]:
        raise RuntimeError(f"L39 beta replay failure: {payload['stateId']}")
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
            # The frozen original centroid is passed only to reproduce the exact
            # historical path digest. No target score enters an L39 event,
            # feature, comparison, gate, or report.
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
            inherited = np.asarray(
                [observation.ordinary_adjacent_h > THRESHOLD for observation in boundary_observations],
                dtype=np.bool_,
            )
            generations = np.asarray(
                [observation.generation for observation in boundary_observations], dtype=np.int64
            )
            offsets = np.asarray(
                [observation.offset for observation in boundary_observations], dtype=np.int64
            )
            scored = score_sustained_inheritance(
                inherited=inherited,
                generations=generations,
                offsets_one_based=offsets,
                required_run=REQUIRED_RUN,
            )
            parent_h = np.asarray(
                [observation.ordinary_adjacent_h for observation in boundary_observations],
                dtype=np.float64,
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
            outcomes.append(
                {
                    "stateId": payload["stateId"],
                    "evaluationCohort": payload["evaluationCohort"],
                    "candidateId": payload["candidateId"],
                    "matrixIndex": matrix_index,
                    "landmark": int(payload["landmark"]),
                    "branchFamily": family,
                    "branchIndex": branch,
                    "branchHalf": "A" if branch < HALVES[family] else "B",
                    "event": scored.event,
                    "certificationOffsetOneBased": scored.certification_offset_one_based,
                    "certificationBoundaryOneBased": scored.certification_boundary_one_based,
                    "certificationGeneration": scored.certification_generation,
                    "retrospectiveOnsetOffsetOneBased": scored.retrospective_onset_offset_one_based,
                    "retrospectiveOnsetBoundaryOneBased": scored.retrospective_onset_boundary_one_based,
                    "retrospectiveOnsetGeneration": scored.retrospective_onset_generation,
                    "futureBoundaryCount": scored.future_boundary_count,
                    "inheritedFutureBoundaryCount": scored.inherited_future_boundary_count,
                    "maximumConsecutiveInherited": scored.maximum_consecutive_inherited,
                    "fissionOpportunity": scored.fission_opportunity,
                    "exactOrderNullEventProbability": scored.exact_order_null_event_probability,
                    "anyInheritedFutureBoundary": bool(np.any(inherited)),
                    "inheritedFutureBoundaryFraction": (
                        float(np.mean(inherited)) if len(inherited) else float("nan")
                    ),
                    "meanParentDaughterH": float(np.mean(parent_h)) if len(parent_h) else float("nan"),
                    "minimumParentDaughterH": float(np.min(parent_h)) if len(parent_h) else float("nan"),
                    "maximumParentDaughterH": float(np.max(parent_h)) if len(parent_h) else float("nan"),
                    "inheritanceSequence": "".join("1" if value else "0" for value in inherited),
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
        ["evaluationCohort", "candidateId", "landmark", "matrixIndex", "branchFamily", "branchIndex"]
    ).reset_index(drop=True)
    compact_frame = pd.DataFrame(compact).sort_values(
        ["evaluationCohort", "candidateId", "landmark", "matrixIndex", "branchFamily", "branchIndex"]
    ).reset_index(drop=True)
    keys = ["stateId", "branchFamily", "branchIndex"]
    if (
        len(outcome_frame) != 53_760
        or len(compact_frame) != 53_760
        or outcome_frame.duplicated(keys).any()
        or compact_frame.duplicated(keys).any()
    ):
        raise RuntimeError("L39 branch result cardinality failure")
    return outcome_frame, compact_frame


def prefix_controls(boundaries: pd.DataFrame, summaries: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for state_id, group in boundaries.groupby("stateId", sort=True):
        ordered = group.sort_values("generation")
        inherited = ordered["inherited"].to_numpy(dtype=np.bool_)
        trailing = 0
        for value in inherited[::-1]:
            if not value:
                break
            trailing += 1
        summary = summaries[summaries["stateId"].eq(state_id)].iloc[0]
        rows.append(
            {
                "stateId": state_id,
                "evaluationCohort": summary.evaluationCohort,
                "candidateId": summary.candidateId,
                "matrixIndex": int(summary.matrixIndex),
                "landmark": int(summary.landmark),
                "prefixInheritanceFraction": float(summary.prefixInheritanceFraction),
                "prefixTrailingInheritanceRun": trailing,
                "prefixMaximumInheritanceRun": maximum_true_run(inherited),
                "prefixBoundaryCount": len(inherited),
            }
        )
    result = pd.DataFrame(rows).sort_values(
        ["evaluationCohort", "candidateId", "landmark", "matrixIndex"]
    ).reset_index(drop=True)
    if len(result) != 280:
        raise RuntimeError("L39 prefix-control cardinality failure")
    return result


def state_committor_results(outcomes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (state_id, family), group in outcomes.groupby(["stateId", "branchFamily"], sort=True):
        expected = BRANCH_COUNTS[family]
        if len(group) != expected:
            raise RuntimeError("L39 per-state branch cardinality failure")
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
                "branches": len(group),
                "successes": int(events.sum()),
                "qHat": float(events.mean()),
                "qHatHalfA": float(half_a["event"].mean()),
                "qHatHalfB": float(half_b["event"].mean()),
                "qFissionOpportunity": float(group["fissionOpportunity"].mean()),
                "qAnyInheritance": float(group["anyInheritedFutureBoundary"].mean()),
                "meanInheritedBoundaryFraction": float(group["inheritedFutureBoundaryFraction"].mean()),
                "meanInheritedBoundaryCount": float(group["inheritedFutureBoundaryCount"].mean()),
                "meanFutureBoundaryCount": float(group["futureBoundaryCount"].mean()),
                "meanMaximumConsecutiveInherited": float(group["maximumConsecutiveInherited"].mean()),
                "meanExactOrderNullProbability": float(group["exactOrderNullEventProbability"].mean()),
                "meanCertificationOffset": float(group["certificationOffsetOneBased"].mean()),
                "meanRetrospectiveOnsetOffset": float(group["retrospectiveOnsetOffsetOneBased"].mean()),
                "opportunityWithoutCertificationFraction": float(
                    np.mean(group["fissionOpportunity"] & ~group["event"])
                ),
                "committorEligible": bool(
                    len(group) == expected
                    and group["selectedObservationsGenerated"].eq(HORIZONS[family]).all()
                ),
                "targetUsesCompletedTestTrajectory": False,
            }
        )
    result = pd.DataFrame(rows).sort_values(
        ["evaluationCohort", "candidateId", "landmark", "matrixIndex", "branchFamily"]
    ).reset_index(drop=True)
    if len(result) != 280 * len(FAMILIES):
        raise RuntimeError("L39 state committor cardinality failure")
    return result


def reliability_results(states: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    bootstrap_rows = []
    for (cohort, candidate, family), group in states.groupby(
        ["evaluationCohort", "candidateId", "branchFamily"], sort=True
    ):
        eligible = group[group["committorEligible"]]
        q = eligible["qHat"].to_numpy(dtype=np.float64)
        half_a = eligible["qHatHalfA"].to_numpy(dtype=np.float64)
        half_b = eligible["qHatHalfB"].to_numpy(dtype=np.float64)
        variance = corrected_between_state_variance(q, BRANCH_COUNTS[family])
        split = safe_spearman(half_a, half_b)
        rng = np.random.default_rng(derived_seed("reliability_bootstrap", cohort, candidate, family))
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
) -> pd.DataFrame:
    state_index = states.set_index(["stateId", "branchFamily"])
    response_index = responses.set_index("stateId")
    prefix_index = prefixes.set_index("stateId")
    specifications = (
        (
            "H8_INHERITANCE_PROPENSITY_VS_H32_SUSTAINED",
            lambda h8, h32, response, prefix: h8.qAnyInheritance,
            "RANK",
        ),
        (
            "H8_EXPECTED_INHERITED_FRACTION_VS_H32_SUSTAINED",
            lambda h8, h32, response, prefix: h8.meanInheritedBoundaryFraction,
            "RANK",
        ),
        (
            "H8_SUSTAINED_VS_H32_SUSTAINED",
            lambda h8, h32, response, prefix: h8.qHat,
            "RANK",
        ),
        (
            "PREFIX_INHERITANCE_FRACTION_VS_H32_SUSTAINED",
            lambda h8, h32, response, prefix: prefix.prefixInheritanceFraction,
            "RANK",
        ),
        (
            "PREFIX_TRAILING_RUN_VS_H32_SUSTAINED",
            lambda h8, h32, response, prefix: prefix.prefixTrailingInheritanceRun,
            "RANK",
        ),
        (
            "CURRENT_MASS_VS_H32_SUSTAINED",
            lambda h8, h32, response, prefix: response.currentMass,
            "RANK",
        ),
        (
            "GENERATION_PHASE_VS_H32_SUSTAINED",
            lambda h8, h32, response, prefix: response.currentGenerationLocalStep,
            "RANK",
        ),
        (
            "H32_FISSION_OPPORTUNITY_VS_H32_SUSTAINED",
            lambda h8, h32, response, prefix: h32.qFissionOpportunity,
            "RANK",
        ),
        (
            "H32_ORDER_NULL_VS_H32_SUSTAINED",
            lambda h8, h32, response, prefix: h32.meanExactOrderNullProbability,
            "RANK",
        ),
        (
            "H32_SUSTAINED_MINUS_ORDER_NULL",
            lambda h8, h32, response, prefix: h32.qHat,
            "DIFFERENCE",
        ),
    )
    rows = []
    for state_id in states["stateId"].drop_duplicates():
        h8 = state_index.loc[(state_id, "H8")]
        h32 = state_index.loc[(state_id, "H32")]
        response = response_index.loc[state_id]
        prefix = prefix_index.loc[state_id]
        for comparison, left_function, comparison_type in specifications:
            left = float(left_function(h8, h32, response, prefix))
            right = (
                float(h32.meanExactOrderNullProbability)
                if comparison == "H32_SUSTAINED_MINUS_ORDER_NULL"
                else float(h32.qHat)
            )
            rows.append(
                {
                    "stateId": state_id,
                    "evaluationCohort": h32.evaluationCohort,
                    "candidateId": h32.candidateId,
                    "matrixIndex": int(h32.matrixIndex),
                    "comparisonId": comparison,
                    "comparisonType": comparison_type,
                    "leftValue": left,
                    "rightValue": right,
                }
            )
    result = pd.DataFrame(rows).sort_values(
        ["evaluationCohort", "candidateId", "comparisonId", "stateId"]
    ).reset_index(drop=True)
    if len(result) != 280 * len(specifications):
        raise RuntimeError("L39 transfer-pair cardinality failure")
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
        passed = bool(
            np.isfinite(observed)
            and np.isfinite(lower)
            and (
                (comparison_type == "RANK" and observed > 0.5 and lower > 0.3)
                or (comparison_type == "DIFFERENCE" and lower > 0)
            )
        )
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
                "gatePassed": passed,
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(bootstrap_rows)


def boundary_hazard_results(outcomes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cohort, candidate, family), group in outcomes.groupby(
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
            ].set_index("branchFamily")
            comparison = transfers[
                transfers["evaluationCohort"].eq(cohort)
                & transfers["candidateId"].eq(candidate)
            ].set_index("comparisonId")
            h32_reliable = bool(rel.loc["H32", "reliabilityGatePassed"])
            h8_reliable = bool(rel.loc["H8", "reliabilityGatePassed"])
            h8_transfer = bool(
                comparison.loc[
                    "H8_INHERITANCE_PROPENSITY_VS_H32_SUSTAINED", "gatePassed"
                ]
            )
            sequence = bool(
                comparison.loc["H32_SUSTAINED_MINUS_ORDER_NULL", "gatePassed"]
            )
            selected_states = states[
                states["evaluationCohort"].eq(cohort)
                & states["candidateId"].eq(candidate)
                & states["branchFamily"].eq("H32")
            ]
            opportunity = bool(
                selected_states["opportunityWithoutCertificationFraction"].mean() >= 0.1
            )
            target_gate = h32_reliable and sequence and opportunity
            short_gate = target_gate and h8_transfer
            rows.append(
                {
                    "evaluationCohort": cohort,
                    "candidateId": candidate,
                    "h32CommittorReliable": h32_reliable,
                    "h8SustainedCommittorReliable": h8_reliable,
                    "h8InheritancePropensityTransferPassed": h8_transfer,
                    "sequenceOrderControlPassed": sequence,
                    "opportunityNondegeneracyPassed": opportunity,
                    "sustainedInheritanceTargetPassed": target_gate,
                    "shortShootingCoordinatePassed": short_gate,
                    "staticPrefixInheritancePassed": bool(
                        comparison.loc[
                            "PREFIX_INHERITANCE_FRACTION_VS_H32_SUSTAINED", "gatePassed"
                        ]
                    ),
                    "massControlPassed": bool(
                        comparison.loc["CURRENT_MASS_VS_H32_SUSTAINED", "gatePassed"]
                    ),
                    "phaseControlPassed": bool(
                        comparison.loc["GENERATION_PHASE_VS_H32_SUSTAINED", "gatePassed"]
                    ),
                }
            )
    gates = pd.DataFrame(rows)
    target_all = bool(gates["sustainedInheritanceTargetPassed"].all())
    short_all = bool(gates["shortShootingCoordinatePassed"].all())
    sequence_all = bool(gates["sequenceOrderControlPassed"].all())
    if short_all:
        classifications = [
            "SUSTAINED_INHERITANCE_COMMITTOR_ESTABLISHED",
            "SHORT_SHOOTING_HEREDITY_COORDINATE_ESTABLISHED",
            "PROMOTABLE_TO_UNTOUCHED_PROCESS_CONFIRMATION",
        ]
        next_theme = "UNTOUCHED_SUSTAINED_INHERITANCE_CONFIRMATION"
    elif target_all:
        classifications = [
            "SUSTAINED_INHERITANCE_COMMITTOR_ESTABLISHED",
            "SHOOTING_ONLY_PROCESS_ESTIMATOR_REQUIRED",
            "NOT_PROMOTABLE_AS_CONFIRMED",
        ]
        next_theme = "SUSTAINED_INHERITANCE_SHOOTING_BUDGET_EFFICIENCY"
    elif not sequence_all:
        classifications = [
            "SUSTAINED_INHERITANCE_ORDER_NOT_SUPPORTED",
            "ORDINARY_INHERITANCE_FREQUENCY_SUFFICIENT",
            "NOT_PROMOTABLE_AS_CONFIRMED",
        ]
        next_theme = "RECURRENCE_AFTER_DEPARTURE_COMMITTOR"
    else:
        classifications = [
            "SUSTAINED_INHERITANCE_OUTCOME_NOT_COMMITTOR_COMPATIBLE",
            "PROCESS_TARGET_REQUIRES_REDEFINITION",
            "NOT_PROMOTABLE_AS_CONFIRMED",
        ]
        next_theme = "RECURRENCE_AFTER_DEPARTURE_COMMITTOR"
    return gates, classifications, next_theme


def benchmark_projection() -> dict[str, Any]:
    prior_runtime = json.loads((L38_ROOT / "runtime_manifest.json").read_text())
    prior_wall = float(prior_runtime["wallSeconds"])
    # L39 replays the same paths twice and performs fewer target calculations
    # than L38. The measured L38 full-execution wall time is therefore a direct
    # conservative opaque benchmark for one pass.
    projected_wall = prior_wall * 1.20
    projected_cpu = max(
        float(prior_runtime.get("controllerCpuHours", 0.0)) * 2.0,
        projected_wall * WORKERS / 3600,
    )
    return {
        "schema": "eidosoma.e01.s19_l39.benchmark_projection.v1",
        "status": "PASS"
        if projected_cpu <= 90 and projected_wall <= 64.8 * 3600
        else "STOP_BEFORE_OUTCOME",
        "sourceLoop": "S19-L38",
        "sourceWallSeconds": prior_wall,
        "projectedWallSecondsIncludingRegeneration": projected_wall,
        "projectedCpuHoursIncludingRegeneration": projected_cpu,
        "newMatrices": 0,
        "newTrajectories": 0,
        "newBranchStreams": 0,
        "scientificOutcomeRetained": False,
    }


def make_figures(
    prefixes: pd.DataFrame,
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

    prefixes.pivot_table(
        index=["evaluationCohort", "candidateId"],
        values=["prefixInheritanceFraction", "prefixTrailingInheritanceRun"],
        aggfunc="mean",
    ).plot(kind="bar", figsize=(13, 6))
    plt.ylabel("Observed-prefix inheritance summary")
    save("01_prefix_inheritance_geometry.png")

    h32 = states[states["branchFamily"].eq("H32")]
    for (cohort, candidate), group in h32.groupby(["evaluationCohort", "candidateId"], sort=True):
        plt.hist(group["qHat"], bins=np.linspace(0, 1, 21), alpha=0.45, label=f"{cohort}/{candidate}")
    plt.xlabel("H32 sustained-inheritance probability")
    plt.ylabel("States")
    plt.legend(fontsize=6)
    save("02_sustained_committor_distributions.png")

    reliability[reliability["evaluationCohort"].isin(EVALUATION_COHORTS)].pivot_table(
        index="branchFamily",
        columns=["evaluationCohort", "candidateId"],
        values="splitHalfSpearman",
    ).plot(kind="bar", figsize=(14, 6))
    plt.axhline(0.5, color="black", linestyle="--")
    plt.ylabel("Split-half Spearman")
    save("03_committor_reliability.png")

    transfers[
        transfers["evaluationCohort"].isin(EVALUATION_COHORTS)
        & transfers["comparisonId"].isin(
            [
                "H8_INHERITANCE_PROPENSITY_VS_H32_SUSTAINED",
                "H8_EXPECTED_INHERITED_FRACTION_VS_H32_SUSTAINED",
                "PREFIX_INHERITANCE_FRACTION_VS_H32_SUSTAINED",
                "CURRENT_MASS_VS_H32_SUSTAINED",
                "GENERATION_PHASE_VS_H32_SUSTAINED",
                "H32_SUSTAINED_MINUS_ORDER_NULL",
            ]
        )
    ].pivot_table(
        index="comparisonId",
        columns=["evaluationCohort", "candidateId"],
        values="pointEstimate",
    ).plot(kind="bar", figsize=(15, 7))
    plt.axhline(0, color="black", linewidth=1)
    plt.ylabel("Registered rank or paired probability difference")
    save("04_short_shooting_and_controls.png")

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
    plt.xlabel("Future fission boundary")
    plt.ylabel("Cumulative certification incidence")
    plt.legend(fontsize=7)
    save("05_online_certification_hazard.png")

    checks = [
        "h32CommittorReliable",
        "h8InheritancePropensityTransferPassed",
        "sequenceOrderControlPassed",
        "opportunityNondegeneracyPassed",
        "sustainedInheritanceTargetPassed",
        "shortShootingCoordinatePassed",
        "staticPrefixInheritancePassed",
        "massControlPassed",
        "phaseControlPassed",
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
        "schema": "eidosoma.e01.s19_l39.artifact_manifest.v1",
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
            "beliefBeforeLoop": "L38 showed that recurrence plus inheritance was not confirmation-stable, while H8 inheritance alone had reproducible state variation.",
            "failureOrAmbiguityTargeted": "Whether compositional heredity is a sustained process rather than a destination or single inherited fission.",
            "informationGainRationale": "Three consecutive inherited fissions are the minimal unambiguously sustained run and can be certified online.",
            "learned": "L39 strict-H090 run-of-three inheritance event locked before outcomes.",
            "ledgerSequence": sequence,
            "loopId": LOOP_ID,
            "motivatingEvidence": "Complete L38 result and reviewer process-first recommendation.",
            "proposedNextTest": "Rescore exact H32/H8 paths and separate certification from retrospective onset.",
            "recordPhase": "PRE_LOOP_METHOD_LOCK",
            "remainingPlausibleHypotheses": "Sustained inheritance, recurrence after departure, homeostatic return, or shooting-only process estimation.",
            "selectedHypotheses": "Three consecutive inherited future fissions define sustained compositional heredity.",
            "timestampUtc": timestamp,
            "weakenedHypotheses": "One exact completed-run destination or one recurrence event is required to define organization.",
        },
        {
            "appendOnly": True,
            "beliefBeforeLoop": "A useful process target must have reliable H32 state variation, temporal ordering beyond opportunity, and a transferable short-shooting coordinate.",
            "failureOrAmbiguityTargeted": "Committor compatibility and short-horizon recoverability of sustained heredity.",
            "informationGainRationale": "Exact existing streams isolate target semantics and temporal ordering without new stochastic adaptation.",
            "learned": ";".join(classifications),
            "ledgerSequence": sequence + 1,
            "loopId": LOOP_ID,
            "motivatingEvidence": "Complete L39 result.",
            "proposedNextTest": next_theme,
            "recordPhase": "POST_LOOP_RESULT_AUTONOMOUS_CONTINUATION",
            "remainingPlausibleHypotheses": next_theme,
            "selectedHypotheses": "Online-certified sustained parent/daughter heredity.",
            "timestampUtc": timestamp,
            "weakenedHypotheses": "Any high marginal parent/daughter similarity automatically supplies a useful sustained-process committor.",
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
        + f"\n\n## {LOOP_ID} — sustained heredity with online certification\n\n"
        + f"- **Learned:** {', '.join(classifications)}.\n"
        + f"- **Next:** {next_theme}.\n",
    )

    candidates_path = ARTIFACT_ROOT / "candidate_registry.parquet"
    candidates = pd.read_parquet(candidates_path)
    candidate = {
        "branchCount": 1,
        "bundleId": "L39_SUSTAINED_INHERITANCE_COMMITTOR",
        "candidateId": "S19-L39-STRICT-H090-THREE-FISSION-HEREDITY",
        "candidateSpecificSuccess": 0,
        "completedFitLeakage": 0,
        "computeEfficiency": 5,
        "crossCandidateDiscriminability": 5,
        "deterministicHReuse": 0,
        "explanatoryLeverage": 5,
        "frozenRank": 1,
        "independenceFromPriorOutcomeSelection": 4,
        "outcomeGuidedThresholdSelection": 0,
        "paperFingerprintSpecificity": 0,
        "proposedSpecification": "online certification of three consecutive strict-H090 parent-to-daughter inheritance events",
        "rankingScore": 29.0,
        "registryOrder": int(candidates["registryOrder"].max()) + 1,
        "selected": True,
        "selectionReason": "L38_INHERITANCE_SIGNAL_AND_REVIEWER_PROCESS_FIRST_DIRECTION",
        "sourceGrounding": 4,
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
            "finding": f"{source.finding}; L39 use: {source.frozenUse}",
            "licenseStatus": "WORKSPACE_OR_HUMAN_DIRECTION",
            "redistributionStatus": "INTERNAL_EVIDENCE_ONLY",
            "repositoryIdentity": None,
            "retainedPath": None,
            "retrievalDate": timestamp[:10],
            "sha256": None,
            "sourceId": f"L39_{source.sourceId}",
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
                "SUSTAINED_INHERITANCE_PROCESS"
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
            "decision": "S19_L39_COMPLETE_AUTONOMOUS_CONTINUATION",
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
    prefixes: pd.DataFrame,
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
                "H8_INHERITANCE_PROPENSITY_VS_H32_SUSTAINED",
                "H8_SUSTAINED_VS_H32_SUSTAINED",
                "PREFIX_INHERITANCE_FRACTION_VS_H32_SUSTAINED",
                "CURRENT_MASS_VS_H32_SUSTAINED",
                "GENERATION_PHASE_VS_H32_SUSTAINED",
                "H32_SUSTAINED_MINUS_ORDER_NULL",
            ]
        )
    ]
    state_summary = states.groupby(
        ["evaluationCohort", "candidateId", "branchFamily"], as_index=False
    ).agg(
        meanQ=("qHat", "mean"),
        meanOpportunity=("qFissionOpportunity", "mean"),
        meanOrderNull=("meanExactOrderNullProbability", "mean"),
        meanAnyInheritance=("qAnyInheritance", "mean"),
        meanInheritedFraction=("meanInheritedBoundaryFraction", "mean"),
        meanOpportunityWithoutCertification=("opportunityWithoutCertificationFraction", "mean"),
    )
    prefix_summary = prefixes.groupby(
        ["evaluationCohort", "candidateId"], as_index=False
    ).agg(
        meanPrefixInheritance=("prefixInheritanceFraction", "mean"),
        meanTrailingRun=("prefixTrailingInheritanceRun", "mean"),
        meanMaximumRun=("prefixMaximumInheritanceRun", "mean"),
    )
    hazard_summary = hazards[
        hazards["evaluationCohort"].isin(EVALUATION_COHORTS)
        & hazards["branchFamily"].eq("H32")
    ].copy()
    return f"""# S19-L39 — Sustained Parent–Daughter Heredity with Online Certification

## Chief/human handoff

- **Step:** `{VERSION}`
- **Status:** complete under the extended L19–L55 autonomous sequence.
- **Classifications:** {", ".join(f"`{value}`" for value in classifications)}
- **Validation:** exact immutable-input validation; exact numerical/discrete/path replay of all 53,760 frozen H32/H8 streams; online/retrospective-onset fixtures; split-half reliability; exact order-conditioned control; 4,096 catalytic-matrix bootstraps; independent full regeneration; runtime/storage/artifact hashes.
- **Recommended next action:** `{next_theme}`.

## Frozen question

Does the exact existing stochastic ensemble assign a reliable state-dependent probability to sustained compositional heredity? The registered event is three consecutive future selected fissions whose parent-to-selected-daughter cosine similarity is strictly greater than `0.9`. It is certified online only at the third qualifying fission. The first fission in the eventually certified run is reported separately as retrospective physical onset and is never available to the online detector before certification.

The run length was not searched: three is the minimal run longer than a pair and therefore the minimal unambiguously sustained event. No completed test trajectory, target centroid, recurrence atlas, threshold variant, horizon variant, new simulation stream, or paper-outcome proximity enters the scientific target.

## Inputs and method

- 280 frozen unique-matrix L28/L31 restored states.
- 35,840 exact H32 continuations and 17,920 exact H8 continuations; no new branch stream.
- Candidate 2 and candidate 3 remain separate.
- Catalytic matrix is the independent higher-level unit.
- The original completed-run centroid is touched only by unchanged path-digest instrumentation required to prove equality with the frozen branch paths; its scores are excluded from every L39 event, feature, comparison, gate and interpretation.
- Exact order-conditioned null holds each branch's number of fissions and inherited fissions fixed and randomizes only their order.

## Observed-prefix heredity

{prefix_summary.to_markdown(index=False)}

## Process probability and opportunity

{state_summary.to_markdown(index=False)}

## Committor reliability

{evaluation_reliability.to_markdown(index=False)}

The primary H32 target gate requires a corrected between-state variance lower bound above zero, split-half rho above `0.5` with lower bound above `0.3`, and at least 20 states with `0.1 < q < 0.9` in each candidate/evaluation cohort. H8 sustained-event reliability is reported but is not required for the registered short coordinate; the primary short coordinate is the already motivated probability of observing any inherited fission within H8.

## Short-shooting and ordinary controls

{key_transfers.to_markdown(index=False)}

The target must exceed the exact order-conditioned run null and leave at least 10% of branches with enough fissions but without certification. Static prefix inheritance, current mass, and generation-local phase are controls, not rescue branches.

## Online certification hazard

{hazard_summary.to_markdown(index=False)}

Boundary-level certification hazards are not projected over intervening molecular observations. Molecular offsets for certification and retrospective run start remain separately preserved in `branch_outcome_results.parquet`.

## Scientific gates

{gates.to_markdown(index=False)}

## Validation and provenance

- Repository commit: `{runtime['repositoryHead']}`.
- Workers: `{runtime['workers']}` with one numerical-library thread each.
- Wall time: `{runtime['wallSeconds']:.3f}` seconds.
- New matrices/trajectories/branch streams: `0/0/0`.
- Every scientific table and all 53,760 paths were independently regenerated from the locked inputs.
- Every S01–S18 and S19-L01–L38 artifact remains immutable.

## Interpretation boundary

This loop tests a simulator-defined process, not the manuscript's unavailable author label. Even a positive result would establish only a reproducible probability of sustained parent/daughter compositional heredity and, conditionally, a short stochastic-shooting estimator. It would not establish the paper pipeline, a static observed biomarker, causal emergence, intervention efficacy, biological replication, or causal control.

## Next boundary

L39 is frozen. The standing human authorization permits `{next_theme}` as the next bounded loop through L55. S20, E02, author contact, interventions and report-bundle generation remain inactive.
"""


def prepare_lock() -> None:
    LOOP_ROOT.mkdir(parents=True, exist_ok=True)
    if git("status", "--porcelain=v1"):
        raise RuntimeError("repository must be clean before L39 lock")
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/eidosoma/groups/42")
    if head != remote:
        raise RuntimeError("local/remote commit mismatch")
    prior = validate_immutable_prior()
    fixtures = fixture_results()
    seeds = analysis_seed_manifest()
    firewall = seed_firewall(seeds)
    benchmark = benchmark_projection()
    if (
        not prior["unchanged"]
        or not fixtures["passed"].all()
        or firewall["status"] != "PASS"
        or benchmark["status"] != "PASS"
    ):
        raise RuntimeError("L39 preoutcome validation or benchmark failed")
    payloads = build_payloads()
    if len(payloads) != 280:
        raise RuntimeError("L39 input payload mismatch")
    shutil.copy2(CONFIG, LOOP_ROOT / "preregistration.yaml")
    BASE.atomic_text(
        LOOP_ROOT / "decision_record.md",
        "# S19-L39 decision record\n\n"
        "L38's completed-run-independent recurrence/inheritance event failed the full confirmation-cohort committor gate, while its inheritance-only H8 diagnostic was reliable in both candidates and both evaluation cohorts. The reviewer recommended shifting from destination prediction to separately defined recurrence, inheritance and homeostasis processes, distinguishing retrospective physical onset from online certification, and establishing a branch-half-reliable process committor before predictor search. L39 therefore freezes one event before outcomes: three consecutive future selected fissions must each preserve the parent composition in the selected daughter at strict cosine H>0.9. Three is the minimal unambiguously sustained run and is not searched. Certification is the third qualifying fission; the first fission of that run is retrospective only. Existing H32/H8 states, candidates, horizons and stochastic streams are unchanged. Fission opportunity, exact order-conditioned probability, prefix heredity, mass and phase are frozen controls. No completed test trajectory or target geometry enters the event.\n",
    )
    BASE.write_json(LOOP_ROOT / "immutable_prior_validation.json", prior)
    BASE.write_parquet(LOOP_ROOT / "fixture_results.parquet", fixtures)
    for name in (
        "response_registry.parquet",
        "original_target_coordinates.parquet",
        "input_trajectory_manifest.parquet",
        "prefix_boundary_registry.parquet",
        "prefix_state_summary.parquet",
    ):
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
        "seedsSha256": sha256_file(LOOP_ROOT / "analysis_seed_manifest.parquet"),
        "firewallSha256": sha256_file(LOOP_ROOT / "seed_firewall.json"),
        "benchmarkSha256": sha256_file(LOOP_ROOT / "benchmark_projection.json"),
        "l38ManifestSha256": sha256_file(L38_ROOT / "artifact_manifest.json"),
    }
    implementation = {
        "schema": "eidosoma.e01.s19_l39.implementation_lock.v1",
        "repositoryHead": head,
        "remoteHead": remote,
        "runnerSha256": sha256_file(RUNNER_PATH),
        "coreSha256": sha256_file(CORE_PATH),
        "threshold": THRESHOLD,
        "thresholdComparison": "STRICT_GREATER_THAN",
        "requiredConsecutiveFissions": REQUIRED_RUN,
        "certificationOnset": "THIRD_QUALIFYING_FISSION",
        "retrospectivePhysicalOnset": "FIRST_FISSION_IN_CERTIFIED_RUN",
        "prefixCarryIn": False,
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
        raise RuntimeError("L39 repository lock mismatch")
    prior = validate_immutable_prior()
    fixtures = fixture_results()
    locked_files = {
        "responsesSha256": LOOP_ROOT / "response_registry.parquet",
        "coordinatesSha256": LOOP_ROOT / "original_target_coordinates.parquet",
        "manifestSha256": LOOP_ROOT / "input_trajectory_manifest.parquet",
        "boundariesSha256": LOOP_ROOT / "prefix_boundary_registry.parquet",
        "summariesSha256": LOOP_ROOT / "prefix_state_summary.parquet",
        "seedsSha256": LOOP_ROOT / "analysis_seed_manifest.parquet",
        "firewallSha256": LOOP_ROOT / "seed_firewall.json",
        "benchmarkSha256": LOOP_ROOT / "benchmark_projection.json",
        "l38ManifestSha256": L38_ROOT / "artifact_manifest.json",
    }
    for key, path in locked_files.items():
        if sha256_file(path) != lock[key]:
            raise RuntimeError(f"L39 locked input changed: {path}")
    if (
        not prior["unchanged"]
        or prior["aggregateSha256"] != lock["priorAggregateSha256"]
        or not fixtures["passed"].all()
        or sha256_file(RUNNER_PATH) != lock["runnerSha256"]
        or sha256_file(CORE_PATH) != lock["coreSha256"]
    ):
        raise RuntimeError("L39 pre-execution validation failed")
    responses = pd.read_parquet(LOOP_ROOT / "response_registry.parquet")
    boundaries = pd.read_parquet(LOOP_ROOT / "prefix_boundary_registry.parquet")
    summaries = pd.read_parquet(LOOP_ROOT / "prefix_state_summary.parquet")
    payloads = build_payloads()
    prefixes = prefix_controls(boundaries, summaries)

    if BUILD_ROOT.exists():
        shutil.rmtree(BUILD_ROOT)
    BUILD_ROOT.mkdir(parents=True)
    outcomes, compact = execute_branches(payloads)
    compact_validation = L36.compact_replay_validation(compact)
    states = state_committor_results(outcomes)
    reliability, reliability_bootstrap = reliability_results(states)
    pairs = transfer_pairs(states, responses, prefixes)
    transfers, transfer_bootstrap = transfer_results(pairs)
    hazards = boundary_hazard_results(outcomes)
    gates, classifications, next_theme = scientific_gates(reliability, transfers, states)
    make_figures(prefixes, states, reliability, transfers, hazards, gates)

    for name in (
        "preregistration.yaml",
        "decision_record.md",
        "immutable_prior_validation.json",
        "fixture_results.parquet",
        "response_registry.parquet",
        "original_target_coordinates.parquet",
        "input_trajectory_manifest.parquet",
        "prefix_boundary_registry.parquet",
        "prefix_state_summary.parquet",
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
            "schema": "eidosoma.e01.s19_l39.classification.v1",
            "classifications": classifications,
            "sustainedInheritanceCommittorEstablished": bool(
                gates["sustainedInheritanceTargetPassed"].all()
            ),
            "shortShootingCoordinateEstablished": bool(
                gates["shortShootingCoordinatePassed"].all()
            ),
            "onlineCertificationPrimary": True,
            "retrospectivePhysicalOnsetDescriptiveOnly": True,
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
            "branchIndex",
            "exceptionClass",
            "exceptionMessage",
        ]
    ).to_csv(BUILD_ROOT / "failure_ledger.csv", index=False)

    # Full independent regeneration of every scientific value and frozen path.
    replay_outcomes, replay_compact = execute_branches(payloads)
    replay_compact_validation = L36.compact_replay_validation(replay_compact)
    replay_prefixes = prefix_controls(boundaries, summaries)
    replay_states = state_committor_results(replay_outcomes)
    replay_reliability, replay_reliability_bootstrap = reliability_results(replay_states)
    replay_pairs = transfer_pairs(replay_states, responses, replay_prefixes)
    replay_transfers, replay_transfer_bootstrap = transfer_results(replay_pairs)
    replay_hazards = boundary_hazard_results(replay_outcomes)
    replay_gates, replay_classifications, replay_next = scientific_gates(
        replay_reliability, replay_transfers, replay_states
    )
    replay_tables = {
        "prefixes": (prefixes, replay_prefixes),
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
            "onlineCertificationAfterRetrospectiveOnset": bool(
                (
                    outcomes.loc[outcomes["event"], "certificationOffsetOneBased"]
                    >= outcomes.loc[
                        outcomes["event"], "retrospectiveOnsetOffsetOneBased"
                    ]
                ).all()
            ),
            "noNewTrajectory": True,
            "noNewBranchStream": True,
        }
    )
    if not all(checks.values()):
        raise RuntimeError(f"L39 regeneration validation failed: {checks}")
    BASE.write_json(
        BUILD_ROOT / "regeneration_validation.json",
        {
            "schema": "eidosoma.e01.s19_l39.regeneration_validation.v1",
            "status": "PASS",
            "checks": checks,
            "outcomeFrameSha256": frame_hash(outcomes),
            "stateFrameSha256": frame_hash(states),
            "gateFrameSha256": frame_hash(gates),
        },
    )
    runtime = {
        "schema": "eidosoma.e01.s19_l39.runtime.v1",
        "repositoryHead": git("rev-parse", "HEAD"),
        "workers": WORKERS,
        "numericalLibraryThreadsPerWorker": 1,
        "gpuHours": 0,
        "wallSeconds": time.perf_counter() - started,
        "controllerCpuHours": (time.process_time() - started_cpu) / 3600,
        "states": 280,
        "uniqueFrozenBranchStreamsScored": 53_760,
        "newMatrices": 0,
        "newTrajectories": 0,
        "newBranchStreams": 0,
        "completedAtUtc": utc_now(),
    }
    BASE.write_json(BUILD_ROOT / "runtime_manifest.json", runtime)
    retained = sum(path.stat().st_size for path in BUILD_ROOT.rglob("*") if path.is_file())
    temporary = sum(path.stat().st_size for path in CACHE_ROOT.rglob("*") if path.is_file())
    storage = {
        "schema": "eidosoma.e01.s19_l39.storage_validation.v1",
        "retainedBytes": retained,
        "retainedGiBCeiling": 25,
        "temporaryBytes": temporary,
        "temporaryGiBCeiling": 75,
        "status": "PASS"
        if retained < 25 * 2**30 and temporary < 75 * 2**30
        else "FAIL",
    }
    if storage["status"] != "PASS":
        raise RuntimeError("L39 storage ceiling exceeded")
    BASE.write_json(BUILD_ROOT / "storage_validation.json", storage)
    report = report_text(
        prefixes,
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
    BASE.atomic_text(BUILD_ROOT / "S19_L39_FULL_RESULTS.md", report)
    BASE.atomic_text(
        BUILD_ROOT / "loop_decision_summary.md",
        "# S19-L39 decision summary\n\n"
        + f"**Classification:** {', '.join(classifications)}\n\n"
        + f"**All-group sustained target:** `{gates['sustainedInheritanceTargetPassed'].all()}`.\n\n"
        + f"**All-group H8 shooting coordinate:** `{gates['shortShootingCoordinatePassed'].all()}`.\n\n"
        + f"**Next:** `{next_theme}`.\n",
    )
    BASE.write_json(BUILD_ROOT / "artifact_manifest.json", manifest_for(BUILD_ROOT))
    stage = LOOP_ROOT.with_name(".L39-promotion-stage")
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
        raise RuntimeError("L39 artifact hash validation failed")

    append_ledgers(classifications, runtime["completedAtUtc"], next_theme)
    BASE.atomic_text(ARTIFACT_ROOT / "research_step_full_results.md", report)
    BASE.atomic_text(
        ARTIFACT_ROOT / "S19_CURRENT_HANDOFF.md",
        report.replace("# S19-L39", "# S19 current handoff — S19-L39", 1),
    )
    BASE.write_json(
        ARTIFACT_ROOT / "s19_status.json",
        {
            "schema": "eidosoma.e01.s19.status.v1",
            "status": "ACTIVE_AUTONOMOUS_SEQUENCE",
            "latestCompletedLoop": LOOP_ID,
            "latestClassification": classifications,
            "selectedDiscoveryLead": (
                "SUSTAINED_INHERITANCE_PROCESS"
                if "PROMOTABLE_TO_UNTOUCHED_PROCESS_CONFIRMATION" in classifications
                else None
            ),
            "nextAuthorizedLoop": "S19-L40",
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
