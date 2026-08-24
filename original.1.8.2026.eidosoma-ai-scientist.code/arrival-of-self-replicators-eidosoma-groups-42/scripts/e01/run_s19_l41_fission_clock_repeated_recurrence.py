"""Execute S19-L41 fission-clock repeated-recurrence committor audit."""

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

from e01_onset_discovery.empirical_committor import RestoredState
from e01_onset_discovery.fission_clock_recurrence import (
    score_repeated_recurrence,
    simulate_fission_clock,
)
from e01_onset_discovery.sustained_inheritance import maximum_true_run


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


L40 = _load_module(
    "e01_s19_l41_l40",
    REPO_ROOT / "scripts/e01/run_s19_l40_recurrence_after_departure.py",
)
L38 = L40.L38
L28 = L40.L28
BASE = L40.BASE

LOOP_ID = "S19-L41"
VERSION = "E01-S19-L41-FISSION-CLOCK-REPEATED-CROSS-GENERATION-RECURRENCE-v1.0.0"
CANDIDATES = L40.CANDIDATES
COHORTS = L40.COHORTS
EVALUATION_COHORTS = L40.EVALUATION_COHORTS
FAMILIES = ("F12", "F4")
FISSION_HORIZONS = {"F12": 12, "F4": 4}
BRANCH_COUNTS = {"F12": 128, "F4": 64}
HALVES = {"F12": 64, "F4": 32}
TARGETS = (
    "PRIMARY_PREFIX_HISTORY",
    "SPECIES_PERMUTED_PREFIX",
    "UNRELATED_MATRIX_PREFIX",
    "BRANCH_ONLY_HISTORY",
    "ORDER_PERMUTED_FUTURE",
)
PRIMARY_TARGET = TARGETS[0]
THRESHOLD = 0.9
MINIMUM_GENERATION_GAP = 2
REQUIRED_RETURN_BOUNDARIES = 2
BOOTSTRAPS = 4096
ROOT_HEX = "270ba09b24080c98b30d3a77881f69aedbb19b522aa1225102a163d1631870fd"
PHASE = "s19_l41_fission_clock_repeated_recurrence"
WORKERS = min(8, os.cpu_count() or 1)

ARTIFACT_ROOT = Path("/artifacts/research_steps/S19")
LOOP_ROOT = ARTIFACT_ROOT / "loops/L41"
L40_ROOT = ARTIFACT_ROOT / "loops/L40"
CACHE_ROOT = Path("/cache/e01_s19_l41")
BUILD_ROOT = CACHE_ROOT / "build"
CONFIG = REPO_ROOT / "configs/e01/s19_l41_fission_clock_repeated_recurrence.yaml"
RUNNER_PATH = Path(__file__)
CORE_PATH = REPO_ROOT / "src/e01_onset_discovery/fission_clock_recurrence.py"


def utc_now() -> str:
    return L40.utc_now()


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def sha256_file(path: Path) -> str:
    return L40.sha256_file(path)


def frame_hash(frame: pd.DataFrame) -> str:
    return L40.frame_hash(frame)


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


def seed_material(*parts: object) -> bytes:
    return "\x1f".join([VERSION, ROOT_HEX, PHASE, *map(str, parts)]).encode()


def derived_seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256(seed_material(*parts)).digest()[:16], "big")


def seed_sha256(*parts: object) -> str:
    return hashlib.sha256(seed_material(*parts)).hexdigest()


def generator(*parts: object) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64DXSM(derived_seed(*parts)))


def validate_immutable_prior() -> dict[str, Any]:
    inherited = json.loads((L40_ROOT / "immutable_prior_validation.json").read_text())
    rows = list(inherited["files"])
    manifest = json.loads((L40_ROOT / "artifact_manifest.json").read_text())
    rows.extend(
        {
            "path": str(L40_ROOT / item["path"]),
            "root": str(L40_ROOT),
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
        "schema": "eidosoma.e01.s19_l41.immutable_prior_validation.v1",
        "status": "PASS" if passed else "FAIL",
        "unchanged": passed,
        "fileCount": len(checked),
        "aggregateSha256": aggregate,
        "l40ManifestSha256": sha256_file(L40_ROOT / "artifact_manifest.json"),
        "files": checked,
    }


def fixture_results() -> pd.DataFrame:
    a = np.asarray([10, 0], dtype=np.int64)
    b = np.asarray([0, 10], dtype=np.int64)

    def score(prefix: list[np.ndarray], future: list[np.ndarray]) -> Any:
        return score_repeated_recurrence(
            prefix_states=np.asarray(prefix, dtype=np.int64).reshape((-1, 2)),
            prefix_generations=np.arange(1, len(prefix) + 1),
            future_states=np.asarray(future, dtype=np.int64).reshape((-1, 2)),
            future_generations=np.arange(
                len(prefix) + 1, len(prefix) + len(future) + 1
            ),
            future_offsets_one_based=np.arange(2, 2 + 2 * len(future), 2),
            threshold=THRESHOLD,
            minimum_generation_gap=MINIMUM_GENERATION_GAP,
            required_return_boundaries=REQUIRED_RETURN_BOUNDARIES,
        )

    repeated = score([a, b], [a, b, a])
    residence = score([a, b], [a, a, a])
    ordered = score([a, b], [a, b, a])
    permuted = score([a, b], [b, a, a])
    empty = score([a, b], [])
    replay = score([a.copy(), b.copy()], [a.copy(), b.copy(), a.copy()])
    return pd.DataFrame(
        [
            {
                "fixtureId": "TWO_RETURN_BOUNDARIES_CERTIFY",
                "passed": repeated.event
                and repeated.certification_boundary_one_based == 2
                and repeated.return_boundary_count == 3,
                "details": json.dumps(
                    {
                        "returns": repeated.return_boundary_count,
                        "certification": repeated.certification_boundary_one_based,
                    }
                ),
            },
            {
                "fixtureId": "CONTINUOUS_RESIDENCE_COUNTS_ONCE",
                "passed": not residence.event
                and residence.return_boundary_count == 1
                and residence.membership_only_event,
                "details": "continuous strict-H residence is membership, not repeated recovery",
            },
            {
                "fixtureId": "ORDER_CONTROL_CHANGES_EVENT",
                "passed": ordered.event and not permuted.event,
                "details": "same future state multiset, different boundary order",
            },
            {
                "fixtureId": "EMPTY_FUTURE_STATUS",
                "passed": not empty.event and empty.future_boundary_count == 0,
                "details": "incomplete/extinct future remains a nonreplaced status-bearing unit",
            },
            {
                "fixtureId": "EXACT_SCORER_REPLAY",
                "passed": repeated == replay,
                "details": "all discrete and numerical fields exact",
            },
            {
                "fixtureId": "FISSION_CLOCKS_FROZEN",
                "passed": FISSION_HORIZONS == {"F12": 12, "F4": 4}
                and BRANCH_COUNTS == {"F12": 128, "F4": 64},
                "details": json.dumps(
                    {"horizons": FISSION_HORIZONS, "branches": BRANCH_COUNTS}
                ),
            },
            {
                "fixtureId": "TARGET_SCOPE_FROZEN",
                "passed": len(TARGETS) == 5
                and THRESHOLD == 0.9
                and REQUIRED_RETURN_BOUNDARIES == 2,
                "details": json.dumps(
                    {
                        "targets": TARGETS,
                        "threshold": THRESHOLD,
                        "requiredReturns": REQUIRED_RETURN_BOUNDARIES,
                    }
                ),
            },
        ]
    )


def load_inputs() -> tuple[pd.DataFrame, ...]:
    return tuple(
        pd.read_parquet(L40_ROOT / name)
        for name in (
            "response_registry.parquet",
            "original_target_coordinates.parquet",
            "input_trajectory_manifest.parquet",
            "prefix_boundary_registry.parquet",
            "prefix_state_summary.parquet",
            "unrelated_control_map.parquet",
            "species_permutation_manifest.parquet",
        )
    )


def build_payloads() -> list[dict[str, Any]]:
    responses, coordinates, manifest, boundaries, summaries, donors, permutations = load_inputs()
    payloads = L38.branch_payloads(
        responses,
        coordinates,
        manifest,
        boundaries,
        summaries,
        donors,
        permutations,
    )
    if len(payloads) != 280:
        raise RuntimeError("L41 payload cardinality failure")
    return payloads


def branch_seed_manifest(payloads: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for payload in payloads:
        for family in FAMILIES:
            for branch in range(BRANCH_COUNTS[family]):
                row = {
                    "stateId": payload["stateId"],
                    "evaluationCohort": payload["evaluationCohort"],
                    "candidateId": payload["candidateId"],
                    "matrixIndex": int(payload["matrixIndex"]),
                    "landmark": int(payload["landmark"]),
                    "branchFamily": family,
                    "futureFissions": FISSION_HORIZONS[family],
                    "branchIndex": branch,
                    "branchHalf": "A" if branch < HALVES[family] else "B",
                    "rootHex": ROOT_HEX,
                }
                materials = []
                for purpose in ("event", "trim", "fission", "daughter", "order"):
                    parts = ("branch", payload["stateId"], family, branch, purpose)
                    row[f"{purpose}DerivedSeed"] = str(derived_seed(*parts))
                    row[f"{purpose}SeedMaterialSha256"] = seed_sha256(*parts)
                    materials.append(row[f"{purpose}SeedMaterialSha256"])
                row["branchIdentitySha256"] = hashlib.sha256(
                    "|".join(
                        [payload["stateId"], family, str(branch), *materials]
                    ).encode()
                ).hexdigest()
                rows.append(row)
    result = pd.DataFrame(rows).sort_values(
        [
            "evaluationCohort",
            "candidateId",
            "landmark",
            "matrixIndex",
            "branchFamily",
            "branchIndex",
        ]
    ).reset_index(drop=True)
    if len(result) != 53_760 or result["branchIdentitySha256"].duplicated().any():
        raise RuntimeError("L41 branch seed scope failure")
    return result


def analysis_seed_manifest() -> pd.DataFrame:
    comparisons = (
        "F4_RETURN_COUNT_VS_F12_PRIMARY",
        "F4_ANY_RETURN_Q_VS_F12_PRIMARY",
        "F4_INHERITANCE_FRACTION_VS_F12_PRIMARY",
        "F12_PRIMARY_MINUS_SPECIES_PERMUTED",
        "F12_PRIMARY_MINUS_UNRELATED",
        "F12_PRIMARY_MINUS_BRANCH_ONLY",
        "F12_PRIMARY_MINUS_ORDER_PERMUTED",
        "F12_MEMBERSHIP_Q_VS_PRIMARY",
        "F12_INHERITANCE_FRACTION_VS_PRIMARY",
        "PREFIX_INHERITANCE_FRACTION_VS_PRIMARY",
        "CURRENT_MASS_VS_PRIMARY",
        "GENERATION_PHASE_VS_PRIMARY",
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
                            "evaluationCohort": cohort,
                            "candidateId": candidate,
                            "branchFamily": family,
                            "targetId": target,
                            "comparisonId": None,
                            "partsJson": json.dumps(parts),
                            "rootHex": ROOT_HEX,
                            "derivedSeed": str(derived_seed(*parts)),
                            "seedMaterialSha256": seed_sha256(*parts),
                        }
                    )
            for comparison in comparisons:
                parts = ("transfer_bootstrap", cohort, candidate, comparison)
                rows.append(
                    {
                        "purpose": parts[0],
                        "evaluationCohort": cohort,
                        "candidateId": candidate,
                        "branchFamily": None,
                        "targetId": None,
                        "comparisonId": comparison,
                        "partsJson": json.dumps(parts),
                        "rootHex": ROOT_HEX,
                        "derivedSeed": str(derived_seed(*parts)),
                        "seedMaterialSha256": seed_sha256(*parts),
                    }
                )
    result = pd.DataFrame(rows).sort_values(
        [
            "purpose",
            "evaluationCohort",
            "candidateId",
            "branchFamily",
            "targetId",
            "comparisonId",
        ],
        na_position="last",
    ).reset_index(drop=True)
    if result["derivedSeed"].duplicated().any() or result["seedMaterialSha256"].duplicated().any():
        raise RuntimeError("L41 analysis seed collision")
    return result


def seed_firewall(branches: pd.DataFrame, analysis: pd.DataFrame) -> dict[str, Any]:
    current_material = set(analysis["seedMaterialSha256"].astype(str))
    current_derived = set(analysis["derivedSeed"].astype(str))
    for column in branches.columns:
        if column.endswith("SeedMaterialSha256"):
            current_material.update(branches[column].astype(str))
        elif column.endswith("DerivedSeed"):
            current_derived.update(branches[column].astype(str))
    prior_material: set[str] = set()
    prior_derived: set[str] = set()
    for path in ARTIFACT_ROOT.glob("loops/L*/**/*seed*manifest*.parquet"):
        if "/L41/" in str(path):
            continue
        try:
            frame = pd.read_parquet(path)
        except (OSError, ValueError, TypeError):
            continue
        for column in frame.columns:
            lowered = column.lower()
            if "seedmaterialsha256" in lowered:
                prior_material.update(frame[column].dropna().astype(str))
            if lowered == "derivedseed" or lowered.endswith("derivedseed"):
                prior_derived.update(frame[column].dropna().astype(str))
    material_overlap = sorted(current_material & prior_material)
    derived_overlap = sorted(current_derived & prior_derived)
    passed = not material_overlap and not derived_overlap
    return {
        "schema": "eidosoma.e01.s19_l41.seed_firewall.v1",
        "status": "PASS" if passed else "FAIL",
        "branchStreamCount": len(branches),
        "scientificStreamSeedCount": len(branches) * 4,
        "orderPermutationSeedCount": len(branches),
        "analysisSeedCount": len(analysis),
        "seedMaterialOverlapCount": len(material_overlap),
        "derivedSeedOverlapCount": len(derived_overlap),
        "seedMaterialOverlaps": material_overlap,
        "derivedSeedOverlaps": derived_overlap,
    }


def source_grounding_registry() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sourceId": "L41_REVIEWER_PROCESS_CLOCK",
                "evidenceClass": "HUMAN_REVIEW_DIRECTION",
                "finding": "An event defined over fissions should be measured in fission opportunities rather than a short molecular window.",
                "frozenUse": "independent F12 primary and F4 short fission-clock ensembles",
            },
            {
                "sourceId": "L41_REVIEWER_RECOVERY",
                "evidenceClass": "HUMAN_REVIEW_DIRECTION",
                "finding": "Return after genuine departure is stronger evidence of memory than uninterrupted inheritance.",
                "frozenUse": "far-to-near return transitions and two online certifications",
            },
            {
                "sourceId": "L41_L40_FIXED_ANCHOR_NULL",
                "evidenceClass": "DIRECT_FROZEN_E01_RESULT",
                "finding": "Return to the single latest prefix state was rare and no better than fixed-membership order expectation.",
                "frozenUse": "multiple eligible past references with order-permuted and reference controls",
            },
            {
                "sourceId": "L41_L39_INHERITANCE_BASELINE",
                "evidenceClass": "DIRECT_FROZEN_E01_RESULT",
                "finding": "Sustained inheritance streaks were explained by marginal inheritance frequency.",
                "frozenUse": "parent-daughter frequency and streak diagnostics remain explicit controls",
            },
        ]
    )


def _stream_parts(payload: dict[str, Any], family: str, branch: int, purpose: str) -> tuple[object, ...]:
    return ("branch", payload["stateId"], family, branch, purpose)


def _simulate(payload: dict[str, Any], family: str, branch: int) -> Any:
    beta = L28.generate_beta(
        L28.derive_seed(
            L28.L23_ROOT_HEX,
            L28.L23_PHASE,
            "catalytic_matrix",
            int(payload["matrixIndex"]),
        )
    )
    if L28.simulator_array_sha256(beta) != payload["betaSha256"]:
        raise RuntimeError(f"L41 beta replay failure: {payload['stateId']}")
    restored = RestoredState(
        tuple(payload["state"]),
        payload["currentObservationKind"],
        int(payload["currentCompletedFissions"]),
        int(payload["currentGrowthGeneration"]),
        int(payload["currentGenerationLocalStep"]),
        int(payload["currentBatchStep"]),
    )
    return simulate_fission_clock(
        restored=restored,
        beta=beta,
        definition=L28.definition(payload["candidateId"]),
        event_rng=generator(*_stream_parts(payload, family, branch, "event")),
        trim_rng=generator(*_stream_parts(payload, family, branch, "trim")),
        fission_rng=generator(*_stream_parts(payload, family, branch, "fission")),
        daughter_rng=generator(*_stream_parts(payload, family, branch, "daughter")),
        future_fissions=FISSION_HORIZONS[family],
    )


def _target_prefix(payload: dict[str, Any], target: str) -> tuple[np.ndarray, np.ndarray]:
    if target in ("PRIMARY_PREFIX_HISTORY", "ORDER_PERMUTED_FUTURE"):
        states = np.asarray(payload["prefixStates"], dtype=np.int64)
        generations = np.asarray(payload["prefixGenerations"], dtype=np.int64)
    elif target == "SPECIES_PERMUTED_PREFIX":
        states = np.asarray(payload["permutedPrefixStates"], dtype=np.int64)
        generations = np.asarray(payload["prefixGenerations"], dtype=np.int64)
    elif target == "UNRELATED_MATRIX_PREFIX":
        states = np.asarray(payload["unrelatedPrefixStates"], dtype=np.int64)
        generations = np.asarray(payload["unrelatedPrefixGenerations"], dtype=np.int64)
    elif target == "BRANCH_ONLY_HISTORY":
        states = np.empty((0, 100), dtype=np.int64)
        generations = np.empty(0, dtype=np.int64)
    else:
        raise ValueError(f"unknown L41 target: {target}")
    return states, generations


def _worker(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    traces: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []
    for family in FAMILIES:
        for branch in range(BRANCH_COUNTS[family]):
            trace = _simulate(payload, family, branch)
            future_states = np.asarray(trace.future_states, dtype=np.int64).reshape((-1, 100))
            future_generations = np.asarray(trace.future_generations, dtype=np.int64)
            offsets = np.asarray(trace.future_offsets_one_based, dtype=np.int64)
            parent_h = np.asarray(trace.parent_daughter_h, dtype=np.float64)
            inherited = parent_h > THRESHOLD
            branch_identity = hashlib.sha256(
                "|".join(
                    [
                        payload["stateId"],
                        family,
                        str(branch),
                        *[
                            seed_sha256(*_stream_parts(payload, family, branch, purpose))
                            for purpose in ("event", "trim", "fission", "daughter", "order")
                        ],
                    ]
                ).encode()
            ).hexdigest()
            traces.append(
                {
                    "stateId": payload["stateId"],
                    "evaluationCohort": payload["evaluationCohort"],
                    "candidateId": payload["candidateId"],
                    "matrixIndex": int(payload["matrixIndex"]),
                    "landmark": int(payload["landmark"]),
                    "branchFamily": family,
                    "futureFissionHorizon": FISSION_HORIZONS[family],
                    "branchIndex": branch,
                    "branchHalf": "A" if branch < HALVES[family] else "B",
                    "branchIdentitySha256": branch_identity,
                    "selectedObservationsGenerated": trace.selected_observations_generated,
                    "molecularUpdates": trace.molecular_updates,
                    "fissions": trace.fissions,
                    "terminalStatus": trace.terminal_status,
                    "completedFissionHorizon": trace.fissions == FISSION_HORIZONS[family],
                    "inheritedFutureBoundaryCount": int(inherited.sum()),
                    "inheritanceFraction": float(inherited.mean()) if len(inherited) else 0.0,
                    "maximumInheritanceRun": maximum_true_run(inherited),
                    "anyInheritedFission": bool(np.any(inherited)),
                    "parentDaughterHMean": float(np.mean(parent_h)) if len(parent_h) else float("nan"),
                    "parentDaughterHMinimum": float(np.min(parent_h)) if len(parent_h) else float("nan"),
                    "parentDaughterHMaximum": float(np.max(parent_h)) if len(parent_h) else float("nan"),
                    "finalStateSha256": trace.final_state_sha256,
                    "pathSha256": trace.path_sha256,
                }
            )
            order_rng = generator(*_stream_parts(payload, family, branch, "order"))
            permutation = order_rng.permutation(len(future_states))
            permutation_digest = hashlib.sha256(
                np.asarray(permutation, dtype="<i8").tobytes()
            ).hexdigest()
            for target in TARGETS:
                prefix_states, prefix_generations = _target_prefix(payload, target)
                states_for_score = (
                    future_states[permutation]
                    if target == "ORDER_PERMUTED_FUTURE"
                    else future_states
                )
                scored = score_repeated_recurrence(
                    prefix_states=prefix_states,
                    prefix_generations=prefix_generations,
                    future_states=states_for_score,
                    future_generations=future_generations,
                    future_offsets_one_based=offsets,
                    threshold=THRESHOLD,
                    minimum_generation_gap=MINIMUM_GENERATION_GAP,
                    required_return_boundaries=REQUIRED_RETURN_BOUNDARIES,
                )
                outcomes.append(
                    {
                        "stateId": payload["stateId"],
                        "evaluationCohort": payload["evaluationCohort"],
                        "candidateId": payload["candidateId"],
                        "matrixIndex": int(payload["matrixIndex"]),
                        "landmark": int(payload["landmark"]),
                        "branchFamily": family,
                        "targetId": target,
                        "branchIndex": branch,
                        "branchHalf": "A" if branch < HALVES[family] else "B",
                        "event": scored.event,
                        "certificationBoundaryOneBased": scored.certification_boundary_one_based,
                        "certificationGeneration": scored.certification_generation,
                        "certificationOffsetOneBased": scored.certification_offset_one_based,
                        "firstReturnBoundaryOneBased": scored.first_return_boundary_one_based,
                        "firstReturnGeneration": scored.first_return_generation,
                        "firstReturnOffsetOneBased": scored.first_return_offset_one_based,
                        "futureBoundaryCount": scored.future_boundary_count,
                        "returnBoundaryCount": scored.return_boundary_count,
                        "qualifyingPairCount": scored.qualifying_pair_count,
                        "distinctReferenceGenerationCount": scored.distinct_reference_generation_count,
                        "membershipBoundaryCount": scored.membership_boundary_count,
                        "membershipPairCount": scored.membership_pair_count,
                        "membershipOnlyEvent": scored.membership_only_event,
                        "maximumReturnH": scored.maximum_return_h,
                        "maximumMembershipH": scored.maximum_membership_h,
                        "returnBoundaryFlags": json.dumps(scored.return_boundary_flags),
                        "membershipBoundaryFlags": json.dumps(scored.membership_boundary_flags),
                        "futureOrderPermuted": target == "ORDER_PERMUTED_FUTURE",
                        "futureOrderPermutationSha256": permutation_digest
                        if target == "ORDER_PERMUTED_FUTURE"
                        else None,
                        "pathSha256": trace.path_sha256,
                        "targetUsesCompletedTestTrajectory": False,
                    }
                )
    return {"traces": traces, "outcomes": outcomes}


def execute_branches(payloads: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    traces: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(_worker, payload): payload["stateId"] for payload in payloads}
        for future in as_completed(futures):
            result = future.result()
            traces.extend(result["traces"])
            outcomes.extend(result["outcomes"])
    trace_frame = pd.DataFrame(traces).sort_values(
        [
            "evaluationCohort",
            "candidateId",
            "landmark",
            "matrixIndex",
            "branchFamily",
            "branchIndex",
        ]
    ).reset_index(drop=True)
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
    if (
        len(trace_frame) != 53_760
        or len(outcome_frame) != 53_760 * len(TARGETS)
        or trace_frame.duplicated(["stateId", "branchFamily", "branchIndex"]).any()
        or outcome_frame.duplicated(
            ["stateId", "branchFamily", "targetId", "branchIndex"]
        ).any()
    ):
        raise RuntimeError("L41 branch output cardinality failure")
    return trace_frame, outcome_frame


def state_trace_results(traces: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (state_id, family), group in traces.groupby(["stateId", "branchFamily"], sort=True):
        rows.append(
            {
                "stateId": state_id,
                "evaluationCohort": group["evaluationCohort"].iloc[0],
                "candidateId": group["candidateId"].iloc[0],
                "matrixIndex": int(group["matrixIndex"].iloc[0]),
                "landmark": int(group["landmark"].iloc[0]),
                "branchFamily": family,
                "branches": len(group),
                "completionFraction": float(group["completedFissionHorizon"].mean()),
                "meanFissions": float(group["fissions"].mean()),
                "meanSelectedObservations": float(group["selectedObservationsGenerated"].mean()),
                "meanMolecularUpdates": float(group["molecularUpdates"].mean()),
                "meanInheritanceFraction": float(group["inheritanceFraction"].mean()),
                "qAnyInheritedFission": float(group["anyInheritedFission"].mean()),
                "meanMaximumInheritanceRun": float(group["maximumInheritanceRun"].mean()),
                "meanParentDaughterH": float(group["parentDaughterHMean"].mean()),
            }
        )
    result = pd.DataFrame(rows).sort_values(
        ["evaluationCohort", "candidateId", "landmark", "matrixIndex", "branchFamily"]
    ).reset_index(drop=True)
    if len(result) != 280 * len(FAMILIES):
        raise RuntimeError("L41 state trace cardinality failure")
    return result


def state_committor_results(outcomes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (state_id, family, target), group in outcomes.groupby(
        ["stateId", "branchFamily", "targetId"], sort=True
    ):
        expected = BRANCH_COUNTS[family]
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
                "qAnyReturn": float(group["returnBoundaryCount"].gt(0).mean()),
                "qMembershipOnlyEvent": float(group["membershipOnlyEvent"].mean()),
                "meanReturnBoundaryCount": float(group["returnBoundaryCount"].mean()),
                "meanMembershipBoundaryCount": float(group["membershipBoundaryCount"].mean()),
                "meanQualifyingPairCount": float(group["qualifyingPairCount"].mean()),
                "meanDistinctReferenceGenerationCount": float(
                    group["distinctReferenceGenerationCount"].mean()
                ),
                "meanFirstReturnBoundary": float(group["firstReturnBoundaryOneBased"].mean()),
                "meanCertificationBoundary": float(
                    group["certificationBoundaryOneBased"].mean()
                ),
                "committorEligible": len(group) == expected,
                "targetUsesCompletedTestTrajectory": False,
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
    if len(result) != 280 * len(FAMILIES) * len(TARGETS):
        raise RuntimeError("L41 state committor cardinality failure")
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
        variance = L40.corrected_between_state_variance(q, BRANCH_COUNTS[family])
        split = safe_spearman(half_a, half_b)
        rng = generator("reliability_bootstrap", cohort, candidate, family, target)
        corrected_boot = np.full(BOOTSTRAPS, np.nan)
        split_boot = np.full(BOOTSTRAPS, np.nan)
        for replicate in range(BOOTSTRAPS):
            indices = rng.integers(0, len(eligible), len(eligible))
            corrected_boot[replicate] = L40.corrected_between_state_variance(
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
                **variance,
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
    traces: pd.DataFrame,
    responses: pd.DataFrame,
    prefixes: pd.DataFrame,
) -> pd.DataFrame:
    index = states.set_index(["stateId", "branchFamily", "targetId"])
    trace_index = traces.set_index(["stateId", "branchFamily"])
    response_index = responses.set_index("stateId")
    prefix_index = prefixes.set_index("stateId")
    definitions = (
        ("F4_RETURN_COUNT_VS_F12_PRIMARY", "RANK"),
        ("F4_ANY_RETURN_Q_VS_F12_PRIMARY", "RANK"),
        ("F4_INHERITANCE_FRACTION_VS_F12_PRIMARY", "RANK"),
        ("F12_PRIMARY_MINUS_SPECIES_PERMUTED", "DIFFERENCE"),
        ("F12_PRIMARY_MINUS_UNRELATED", "DIFFERENCE"),
        ("F12_PRIMARY_MINUS_BRANCH_ONLY", "DIFFERENCE"),
        ("F12_PRIMARY_MINUS_ORDER_PERMUTED", "DIFFERENCE"),
        ("F12_MEMBERSHIP_Q_VS_PRIMARY", "RANK"),
        ("F12_INHERITANCE_FRACTION_VS_PRIMARY", "RANK"),
        ("PREFIX_INHERITANCE_FRACTION_VS_PRIMARY", "RANK"),
        ("CURRENT_MASS_VS_PRIMARY", "RANK"),
        ("GENERATION_PHASE_VS_PRIMARY", "RANK"),
    )
    rows = []
    for state_id in states["stateId"].drop_duplicates():
        primary = index.loc[(state_id, "F12", PRIMARY_TARGET)]
        f4 = index.loc[(state_id, "F4", PRIMARY_TARGET)]
        f12_trace = trace_index.loc[(state_id, "F12")]
        f4_trace = trace_index.loc[(state_id, "F4")]
        controls = {
            "SPECIES": index.loc[(state_id, "F12", "SPECIES_PERMUTED_PREFIX")],
            "UNRELATED": index.loc[(state_id, "F12", "UNRELATED_MATRIX_PREFIX")],
            "BRANCH": index.loc[(state_id, "F12", "BRANCH_ONLY_HISTORY")],
            "ORDER": index.loc[(state_id, "F12", "ORDER_PERMUTED_FUTURE")],
        }
        response = response_index.loc[state_id]
        prefix = prefix_index.loc[state_id]
        values = {
            "F4_RETURN_COUNT_VS_F12_PRIMARY": f4.meanReturnBoundaryCount,
            "F4_ANY_RETURN_Q_VS_F12_PRIMARY": f4.qAnyReturn,
            "F4_INHERITANCE_FRACTION_VS_F12_PRIMARY": f4_trace.meanInheritanceFraction,
            "F12_PRIMARY_MINUS_SPECIES_PERMUTED": primary.qHat - controls["SPECIES"].qHat,
            "F12_PRIMARY_MINUS_UNRELATED": primary.qHat - controls["UNRELATED"].qHat,
            "F12_PRIMARY_MINUS_BRANCH_ONLY": primary.qHat - controls["BRANCH"].qHat,
            "F12_PRIMARY_MINUS_ORDER_PERMUTED": primary.qHat - controls["ORDER"].qHat,
            "F12_MEMBERSHIP_Q_VS_PRIMARY": primary.qMembershipOnlyEvent,
            "F12_INHERITANCE_FRACTION_VS_PRIMARY": f12_trace.meanInheritanceFraction,
            "PREFIX_INHERITANCE_FRACTION_VS_PRIMARY": prefix.prefixInheritanceFraction,
            "CURRENT_MASS_VS_PRIMARY": float(response.currentMass),
            "GENERATION_PHASE_VS_PRIMARY": float(
                response.currentGenerationLocalStep
            ),
        }
        for comparison, comparison_type in definitions:
            rows.append(
                {
                    "stateId": state_id,
                    "evaluationCohort": primary.evaluationCohort,
                    "candidateId": primary.candidateId,
                    "matrixIndex": int(primary.matrixIndex),
                    "landmark": int(primary.landmark),
                    "comparisonId": comparison,
                    "comparisonType": comparison_type,
                    "predictor": float(values[comparison]),
                    "response": float(primary.qHat),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["evaluationCohort", "candidateId", "comparisonId", "landmark", "matrixIndex"]
    ).reset_index(drop=True)


def transfer_results(pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    bootstrap_rows = []
    for (cohort, candidate, comparison, comparison_type), group in pairs.groupby(
        ["evaluationCohort", "candidateId", "comparisonId", "comparisonType"], sort=True
    ):
        predictor = group["predictor"].to_numpy(dtype=np.float64)
        response = group["response"].to_numpy(dtype=np.float64)
        observed = (
            safe_spearman(predictor, response)
            if comparison_type == "RANK"
            else float(np.mean(predictor))
        )
        rng = generator("transfer_bootstrap", cohort, candidate, comparison)
        boot = np.full(BOOTSTRAPS, np.nan)
        for replicate in range(BOOTSTRAPS):
            indices = rng.integers(0, len(group), len(group))
            boot[replicate] = (
                safe_spearman(predictor[indices], response[indices])
                if comparison_type == "RANK"
                else float(np.mean(predictor[indices]))
            )
            bootstrap_rows.append(
                {
                    "evaluationCohort": cohort,
                    "candidateId": candidate,
                    "comparisonId": comparison,
                    "replicate": replicate,
                    "value": boot[replicate],
                }
            )
        low, high = interval(boot)
        rows.append(
            {
                "evaluationCohort": cohort,
                "candidateId": candidate,
                "comparisonId": comparison,
                "comparisonType": comparison_type,
                "definedPairs": int(np.isfinite(predictor).sum()),
                "pointEstimate": observed,
                "lower95": low,
                "upper95": high,
                "gatePassed": bool(
                    (comparison_type == "RANK" and observed > 0.5 and low > 0.3)
                    or (comparison_type == "DIFFERENCE" and low > 0)
                ),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(bootstrap_rows)


def hazard_and_renewal(
    outcomes: pd.DataFrame, traces: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = outcomes[
        outcomes["targetId"].eq(PRIMARY_TARGET)
        & outcomes["branchFamily"].eq("F12")
    ]
    hazards = []
    for (cohort, candidate), group in primary.groupby(
        ["evaluationCohort", "candidateId"], sort=True
    ):
        survival = 1.0
        for boundary in range(1, FISSION_HORIZONS["F12"] + 1):
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
            hazards.append(
                {
                    "evaluationCohort": cohort,
                    "candidateId": candidate,
                    "futureFissionOneBased": boundary,
                    "branchesAtRisk": len(at_risk),
                    "certifications": events,
                    "discreteHazard": hazard,
                    "survivalWithoutCertification": survival,
                    "cumulativeCertificationIncidence": 1.0 - survival,
                }
            )
    renewal = []
    primary_index = primary.set_index(
        ["stateId", "branchFamily", "branchIndex"]
    )
    trace_primary = traces[traces["branchFamily"].eq("F12")]
    for (cohort, candidate), group in trace_primary.groupby(
        ["evaluationCohort", "candidateId"], sort=True
    ):
        event_rows = primary[
            primary["evaluationCohort"].eq(cohort)
            & primary["candidateId"].eq(candidate)
        ]
        gap = (
            event_rows["certificationBoundaryOneBased"]
            - event_rows["firstReturnBoundaryOneBased"]
        )
        renewal.append(
            {
                "evaluationCohort": cohort,
                "candidateId": candidate,
                "branches": len(event_rows),
                "meanAnyReturnProbability": float(event_rows["returnBoundaryCount"].gt(0).mean()),
                "meanRepeatedReturnProbability": float(event_rows["event"].mean()),
                "meanMembershipOnlyProbability": float(event_rows["membershipOnlyEvent"].mean()),
                "meanFirstReturnBoundary": float(event_rows["firstReturnBoundaryOneBased"].mean()),
                "meanCertificationBoundary": float(event_rows["certificationBoundaryOneBased"].mean()),
                "meanInterReturnGap": float(gap.mean()),
                "meanInheritanceFraction": float(group["inheritanceFraction"].mean()),
                "meanMaximumInheritanceRun": float(group["maximumInheritanceRun"].mean()),
                "completedHorizonFraction": float(group["completedFissionHorizon"].mean()),
                "indexLookupExact": bool(
                    all(
                        (row.stateId, "F12", int(row.branchIndex)) in primary_index.index
                        for row in group.itertuples(index=False)
                    )
                ),
            }
        )
    return pd.DataFrame(hazards), pd.DataFrame(renewal)


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
            selected = states[
                states["evaluationCohort"].eq(cohort)
                & states["candidateId"].eq(candidate)
                & states["branchFamily"].eq("F12")
                & states["targetId"].eq(PRIMARY_TARGET)
            ]
            reliability_passed = bool(rel.loc[("F12", PRIMARY_TARGET), "reliabilityGatePassed"])
            species = bool(
                comparison.loc["F12_PRIMARY_MINUS_SPECIES_PERMUTED", "gatePassed"]
            )
            unrelated = bool(
                comparison.loc["F12_PRIMARY_MINUS_UNRELATED", "gatePassed"]
            )
            order = bool(
                comparison.loc["F12_PRIMARY_MINUS_ORDER_PERMUTED", "gatePassed"]
            )
            opportunity_gap = float(
                np.mean(selected["qMembershipOnlyEvent"] - selected["qHat"])
            )
            opportunity = opportunity_gap >= 0.1
            f4 = bool(comparison.loc["F4_RETURN_COUNT_VS_F12_PRIMARY", "gatePassed"])
            target_passed = reliability_passed and species and unrelated and order and opportunity
            rows.append(
                {
                    "evaluationCohort": cohort,
                    "candidateId": candidate,
                    "primaryF12Reliable": reliability_passed,
                    "speciesPermutationControlPassed": species,
                    "unrelatedMatrixControlPassed": unrelated,
                    "futureOrderControlPassed": order,
                    "membershipOpportunityMinusPrimary": opportunity_gap,
                    "membershipOpportunityGatePassed": opportunity,
                    "repeatedRecurrenceTargetPassed": target_passed,
                    "f4ReturnCountRankPassed": f4,
                    "shortShootingCoordinatePassed": target_passed and f4,
                }
            )
    gates = pd.DataFrame(rows)
    target_all = bool(gates["repeatedRecurrenceTargetPassed"].all())
    short_all = bool(gates["shortShootingCoordinatePassed"].all())
    reliable_all = bool(gates["primaryF12Reliable"].all())
    reference_all = bool(
        gates[
            ["speciesPermutationControlPassed", "unrelatedMatrixControlPassed"]
        ].all(axis=None)
    )
    order_all = bool(gates["futureOrderControlPassed"].all())
    if target_all and short_all:
        classifications = [
            "STATE_DEPENDENT_REPEATED_RECURRENCE_COMMITTOR_ESTABLISHED",
            "FISSION_CLOCK_SHOOTING_COORDINATE_ESTABLISHED",
            "PROMOTABLE_TO_UNTOUCHED_PROCESS_CONFIRMATION",
        ]
        next_theme = "UNTOUCHED_FISSION_CLOCK_RECURRENCE_CONFIRMATION"
    elif target_all:
        classifications = [
            "STATE_DEPENDENT_REPEATED_RECURRENCE_COMMITTOR_ESTABLISHED",
            "SHOOTING_REQUIRED_AT_LONGER_PROCESS_CLOCK",
            "NOT_PROMOTABLE_AS_CONFIRMED",
        ]
        next_theme = "ADAPTIVE_SHOOTING_EFFICIENCY_FOR_PROCESS_COMMITTOR"
    elif reliable_all and reference_all and not order_all:
        classifications = [
            "REPEATED_RECURRENCE_ORDER_NOT_SUPPORTED",
            "MEMBERSHIP_FREQUENCY_SUFFICIENT",
            "NOT_PROMOTABLE_AS_CONFIRMED",
        ]
        next_theme = "FISSION_CONDITIONED_HEREDITY_RECOVERY_HAZARD"
    elif reliable_all and not reference_all:
        classifications = [
            "RECURRENCE_REFERENCE_NONSPECIFIC",
            "PROCESS_TARGET_REQUIRES_REDEFINITION",
            "NOT_PROMOTABLE_AS_CONFIRMED",
        ]
        next_theme = "FISSION_CONDITIONED_HEREDITY_RECOVERY_HAZARD"
    else:
        classifications = [
            "NO_RELIABLE_REPEATED_RECURRENCE_COMMITTOR_AT_F12",
            "PROCESS_TARGET_REQUIRES_REDEFINITION",
            "NOT_PROMOTABLE_AS_CONFIRMED",
        ]
        next_theme = "FISSION_CONDITIONED_HEREDITY_RECOVERY_HAZARD"
    return gates, classifications, next_theme


def benchmark_projection(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    start = time.perf_counter()
    count = 0
    digests = []
    for payload in payloads[:2]:
        for family in FAMILIES:
            for branch in range(2):
                trace = _simulate(payload, family, branch)
                digests.append(trace.path_sha256)
                count += 1
    elapsed = time.perf_counter() - start
    serial_seconds_per_branch = elapsed / count
    projected_cpu = serial_seconds_per_branch * 53_760 * 2.2 / 3600
    projected_wall = serial_seconds_per_branch * 53_760 * 2.2 / WORKERS
    status = "PASS" if projected_cpu <= 90 and projected_wall <= 64.8 * 3600 else "STOP_BEFORE_OUTCOME"
    return {
        "schema": "eidosoma.e01.s19_l41.benchmark_projection.v1",
        "status": status,
        "opaqueStates": 2,
        "opaqueBranches": count,
        "opaquePathDigestAggregate": hashlib.sha256("|".join(digests).encode()).hexdigest(),
        "outcomeScoringPerformed": False,
        "serialSecondsPerBranch": serial_seconds_per_branch,
        "projectedCpuHoursIncludingFullRegeneration": projected_cpu,
        "projectedWallSecondsIncludingFullRegeneration": projected_wall,
        "newMatrices": 0,
        "newTrajectories": 0,
        "newBranchStreams": 53_760,
    }


def make_figures(
    states: pd.DataFrame,
    reliability: pd.DataFrame,
    transfers: pd.DataFrame,
    hazards: pd.DataFrame,
    traces: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    root = BUILD_ROOT / "figures"
    root.mkdir(parents=True, exist_ok=True)

    def save(name: str) -> None:
        plt.tight_layout()
        plt.savefig(root / name, dpi=180)
        plt.close()

    primary = states[
        states["branchFamily"].eq("F12") & states["targetId"].eq(PRIMARY_TARGET)
    ]
    for candidate, group in primary.groupby("candidateId"):
        plt.hist(group["qHat"], bins=np.linspace(0, 1, 21), alpha=0.55, label=candidate)
    plt.xlabel("F12 repeated-recurrence committor")
    plt.ylabel("states")
    plt.legend(fontsize=7)
    save("01_f12_committor_distributions.png")

    rel = reliability[
        reliability["branchFamily"].eq("F12")
        & reliability["targetId"].eq(PRIMARY_TARGET)
    ]
    labels = [f"{r.evaluationCohort}\n{r.candidateId[-2:]}" for r in rel.itertuples()]
    plt.bar(np.arange(len(rel)), rel["splitHalfSpearman"], color="#4c78a8")
    plt.axhline(0.5, color="black", linestyle="--", linewidth=1)
    plt.xticks(np.arange(len(rel)), labels, rotation=25, ha="right", fontsize=7)
    plt.ylabel("split-half Spearman")
    save("02_committor_reliability.png")

    selected = transfers[
        transfers["comparisonId"].isin(
            [
                "F4_RETURN_COUNT_VS_F12_PRIMARY",
                "F12_PRIMARY_MINUS_SPECIES_PERMUTED",
                "F12_PRIMARY_MINUS_UNRELATED",
                "F12_PRIMARY_MINUS_ORDER_PERMUTED",
            ]
        )
    ]
    labels = [f"{r.candidateId[-2:]} {r.comparisonId[:12]}" for r in selected.itertuples()]
    plt.bar(np.arange(len(selected)), selected["pointEstimate"], color="#f58518")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xticks(np.arange(len(selected)), labels, rotation=65, ha="right", fontsize=6)
    plt.ylabel("rank or q difference")
    save("03_short_shooting_and_controls.png")

    for candidate, group in hazards[
        hazards["evaluationCohort"].isin(EVALUATION_COHORTS)
    ].groupby("candidateId"):
        summary = group.groupby("futureFissionOneBased")["cumulativeCertificationIncidence"].mean()
        plt.plot(summary.index, summary.values, marker="o", label=candidate)
    plt.xlabel("future fission")
    plt.ylabel("cumulative certification incidence")
    plt.legend(fontsize=7)
    save("04_online_certification_hazard.png")

    trace = traces[traces["branchFamily"].eq("F12")]
    for candidate, group in trace.groupby("candidateId"):
        plt.hist(group["inheritanceFraction"], bins=np.linspace(0, 1, 21), alpha=0.55, label=candidate)
    plt.xlabel("ordinary parent-daughter inheritance fraction")
    plt.ylabel("branches")
    plt.legend(fontsize=7)
    save("05_inheritance_frequency_baseline.png")

    checks = [
        "primaryF12Reliable",
        "speciesPermutationControlPassed",
        "unrelatedMatrixControlPassed",
        "futureOrderControlPassed",
        "membershipOpportunityGatePassed",
        "f4ReturnCountRankPassed",
    ]
    matrix = gates.set_index(["evaluationCohort", "candidateId"])[checks].astype(int)
    plt.imshow(matrix.to_numpy(), aspect="auto", vmin=0, vmax=1, cmap="RdYlGn")
    plt.xticks(range(len(checks)), checks, rotation=35, ha="right", fontsize=6)
    plt.yticks(range(len(matrix)), ["/".join(item) for item in matrix.index], fontsize=7)
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
        "schema": "eidosoma.e01.s19_l41.artifact_manifest.v1",
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
            "beliefBeforeLoop": "L40 showed that return to one exact prefix anchor was rare and no better than its membership-order control.",
            "failureOrAmbiguityTargeted": "Whether exact-anchor sparsity and the short molecular clock hid a transferable repeated-recurrence process.",
            "informationGainRationale": "Two online far-to-near returns over fixed fission opportunities test recovery without a completed destination or uninterrupted residence.",
            "learned": "L41 fission-clock repeated-recurrence contract locked before outcomes.",
            "ledgerSequence": sequence,
            "loopId": LOOP_ID,
            "motivatingEvidence": "Complete L39/L40 results and reviewer recovery/process-clock direction.",
            "proposedNextTest": "Generate independent F12/F4 branch ensembles from the frozen states.",
            "recordPhase": "PRE_LOOP_METHOD_LOCK",
            "remainingPlausibleHypotheses": "Repeated recurrence, fission-conditioned heredity recovery, or shooting-only process estimation.",
            "selectedHypotheses": "Multiple past references and a fission-matched clock reveal repeated return capacity.",
            "timestampUtc": timestamp,
            "weakenedHypotheses": "Return to the single latest composition is the unique useful homeostasis target.",
        },
        {
            "appendOnly": True,
            "beliefBeforeLoop": "A genuine recurrence capacity must be reliable, reference-specific, order-specific and distinguishable from membership and inheritance frequency.",
            "failureOrAmbiguityTargeted": "Committor compatibility and short-shooting recoverability of repeated cross-generation recurrence.",
            "informationGainRationale": "Independent F12/F4 ensembles align opportunity counts while control permutations preserve the sampled future states.",
            "learned": ";".join(classifications),
            "ledgerSequence": sequence + 1,
            "loopId": LOOP_ID,
            "motivatingEvidence": "Complete L41 result.",
            "proposedNextTest": next_theme,
            "recordPhase": "POST_LOOP_RESULT_AUTONOMOUS_CONTINUATION",
            "remainingPlausibleHypotheses": next_theme,
            "selectedHypotheses": "Online two-return process at a fixed fission clock.",
            "timestampUtc": timestamp,
            "weakenedHypotheses": "Any repeated high-H pattern necessarily demonstrates temporal memory beyond recurrence opportunities.",
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
        + f"\n\n## {LOOP_ID} — fission-clock repeated recurrence\n\n"
        + f"- **Learned:** {', '.join(classifications)}.\n"
        + f"- **Next:** {next_theme}.\n",
    )

    candidates_path = ARTIFACT_ROOT / "candidate_registry.parquet"
    candidates = pd.read_parquet(candidates_path)
    candidate = {
        "branchCount": 2,
        "bundleId": "L41_FISSION_CLOCK_REPEATED_RECURRENCE",
        "candidateId": "S19-L41-TWO-ONLINE-RETURN-BOUNDARIES-H090",
        "candidateSpecificSuccess": 0,
        "completedFitLeakage": 0,
        "computeEfficiency": 4,
        "crossCandidateDiscriminability": 5,
        "deterministicHReuse": 0,
        "explanatoryLeverage": 5,
        "frozenRank": 1,
        "independenceFromPriorOutcomeSelection": 5,
        "outcomeGuidedThresholdSelection": 0,
        "paperFingerprintSpecificity": 0,
        "proposedSpecification": "two far-to-near nonadjacent-generation returns within exactly 12 future fissions",
        "rankingScore": 29.0,
        "registryOrder": int(candidates["registryOrder"].max()) + 1,
        "selected": True,
        "selectionReason": "L40_FIXED_ANCHOR_NULL_AND_REVIEWER_PROCESS_CLOCK",
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
    additions_source = [
        {
            "commitOrVersion": None,
            "evidenceClass": source.evidenceClass,
            "finding": f"{source.finding}; L41 use: {source.frozenUse}",
            "licenseStatus": "WORKSPACE_OR_HUMAN_DIRECTION",
            "redistributionStatus": "INTERNAL_EVIDENCE_ONLY",
            "repositoryIdentity": None,
            "retainedPath": None,
            "retrievalDate": timestamp[:10],
            "sha256": None,
            "sourceId": f"L41_{source.sourceId}",
            "sourceType": source.evidenceClass,
            "treeIdentity": None,
            "url": None,
        }
        for source in source_grounding_registry().itertuples(index=False)
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
            "selectedDiscoveryLead": (
                "FISSION_CLOCK_REPEATED_RECURRENCE_PROCESS"
                if "PROMOTABLE_TO_UNTOUCHED_PROCESS_CONFIRMATION" in classifications
                else None
            ),
            "newMatrices": 0,
            "newTrajectories": 0,
            "newBranchStreams": 53_760,
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
            "decision": "S19_L41_COMPLETE_AUTONOMOUS_CONTINUATION",
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
    states: pd.DataFrame,
    reliability: pd.DataFrame,
    transfers: pd.DataFrame,
    renewal: pd.DataFrame,
    gates: pd.DataFrame,
    classifications: list[str],
    runtime: dict[str, Any],
    next_theme: str,
) -> str:
    rel = reliability[
        reliability["branchFamily"].eq("F12")
        & reliability["targetId"].eq(PRIMARY_TARGET)
    ]
    selected_transfers = transfers[
        transfers["comparisonId"].isin(
            [
                "F4_RETURN_COUNT_VS_F12_PRIMARY",
                "F12_PRIMARY_MINUS_SPECIES_PERMUTED",
                "F12_PRIMARY_MINUS_UNRELATED",
                "F12_PRIMARY_MINUS_ORDER_PERMUTED",
                "F12_INHERITANCE_FRACTION_VS_PRIMARY",
            ]
        )
    ]
    primary = states[
        states["branchFamily"].eq("F12") & states["targetId"].eq(PRIMARY_TARGET)
    ]
    summary = primary.groupby(["evaluationCohort", "candidateId"], as_index=False).agg(
        states=("stateId", "size"),
        meanQ=("qHat", "mean"),
        meanAnyReturn=("qAnyReturn", "mean"),
        meanMembershipOpportunity=("qMembershipOnlyEvent", "mean"),
        meanReturnBoundaries=("meanReturnBoundaryCount", "mean"),
    )
    return f"""# S19-L41 — Fission-Clock Repeated Cross-Generation Recurrence

## Chief/human handoff

- **Step:** `{VERSION}`
- **Status:** complete under the extended L19–L55 autonomous sequence.
- **Classifications:** {", ".join(f"`{item}`" for item in classifications)}
- **Validation:** immutable L40-and-earlier baseline; seven fixtures; zero-overlap branch and analysis seeds; 53,760 independent F12/F4 branch streams; candidate-separated 4,096 matrix bootstraps; exact full branch, score, statistic and report regeneration; storage and artifact hashes.
- **Recommended next action:** `{next_theme}`.

## Frozen question

Does a clock matched to the process reveal a reliable probability of repeated recurrence? A return is certified only when a future post-fission state has strict `H>0.9` to an eligible boundary at least two generations earlier and the immediately preceding boundary was `H<=0.9` to that same reference. At most one return is counted per future boundary. The primary event is online certification of a second return within exactly 12 future fissions. Continuous residence near a reference does not count as repeated recovery.

The short coordinate is calculated from an independent 64-branch, four-fission ensemble. It is not a static biomarker. No completed trajectory, completed-run centroid, threshold search, recurrence-count search, horizon search, paper label, emergence value, intervention or new catalytic matrix enters this loop.

## Anchor results

### F12 process probabilities

{summary.to_markdown(index=False)}

### Empirical committor reliability

{rel.to_markdown(index=False)}

### Short shooting, reference, order and inheritance controls

{selected_transfers.to_markdown(index=False)}

### Renewal and ordinary-inheritance context

{renewal.to_markdown(index=False)}

### Locked scientific gates

{gates.to_markdown(index=False)}

## Interpretation

The event separates three objects that previous loops partially conflated: ordinary parent-to-daughter inheritance frequency, temporal ordering of compositional membership, and recovery after a genuine far-to-near transition. The ordinary inheritance fraction and run length remain controls; neither defines the target. Online certification remains distinct from the first return and from any retrospective physical onset.

The one-per-branch future-order control preserves the sampled future boundary compositions and fission count while changing their temporal order. Species-permuted and unrelated prefixes test whether apparent recovery is specific to the observed past. The membership-only event quantifies how often the same compositions would look recurrent without requiring departure. An event must pass all of these gates in both candidates and both held-out evaluation cohorts before it can become a confirmation lead.

## Clock and statistical units

The primary horizon is 12 future fission opportunities, not 32 molecular observations. F4 and F12 use independent domain-separated stochastic streams. Catalytic matrix is the independent higher-level unit; candidates and cohorts remain separate. Incomplete or extinct branches are retained as nonreplaced status-bearing units. Hazard and renewal outputs use the post-fission clock, while molecular-update counts are diagnostics only.

## Validation and provenance

- Repository lock: `{runtime['repositoryHead']}`.
- Workers: `{runtime['workers']}` with one numerical-library thread each.
- New matrices/trajectories: `0/0`; new branch streams: `{runtime['newBranchStreams']}`.
- Wall time: `{runtime['wallSeconds']:.2f}` seconds; GPU hours: `0`.
- Exact full regeneration reran every stochastic stream from its frozen seed identity and reproduced every scientific frame and classification.
- S01–S18, V1/V2 and S19-L01–L40 remain unchanged.

## Caveats and boundaries

This is exploratory simulation evidence. A positive result would establish only a reproducible process committor and possibly a simulation-based short-shooting coordinate. It would not identify author code, reproduce the paper, establish Phi-r as an independent precursor, prove causal control, or establish a biological claim. A negative ordering result would constrain this precise two-return process, not all robustness, error correction, recovery or homeostasis.

## Reproduction

```bash
PYTHONPATH=src pytest -q tests/e01/test_s19_l41.py
python -m ruff check src/e01_onset_discovery/fission_clock_recurrence.py scripts/e01/run_s19_l41_fission_clock_repeated_recurrence.py tests/e01/test_s19_l41.py
python scripts/e01/run_s19_l41_fission_clock_repeated_recurrence.py --prepare-lock
python scripts/e01/run_s19_l41_fission_clock_repeated_recurrence.py
```
"""


def prepare_lock() -> None:
    LOOP_ROOT.mkdir(parents=True, exist_ok=True)
    if git("status", "--porcelain=v1"):
        raise RuntimeError("repository must be clean before L41 lock")
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/eidosoma/groups/42")
    if head != remote:
        raise RuntimeError("L41 local/remote commit mismatch")
    prior = validate_immutable_prior()
    fixtures = fixture_results()
    payloads = build_payloads()
    branch_seeds = branch_seed_manifest(payloads)
    analysis_seeds = analysis_seed_manifest()
    firewall = seed_firewall(branch_seeds, analysis_seeds)
    benchmark = benchmark_projection(payloads)
    if (
        not prior["unchanged"]
        or not fixtures["passed"].all()
        or firewall["status"] != "PASS"
        or benchmark["status"] != "PASS"
    ):
        raise RuntimeError("L41 preoutcome validation or benchmark failed")
    shutil.copy2(CONFIG, LOOP_ROOT / "preregistration.yaml")
    BASE.atomic_text(
        LOOP_ROOT / "decision_record.md",
        "# S19-L41 decision record\n\n"
        "L40 found that exact-anchor recovery was rare and was not enriched over its order-conditioned membership expectation. The reviewer emphasized separating ordinary inheritance frequency, temporal ordering and homeostatic recovery, restricting claims to at-risk process clocks, and not interpreting a short molecular horizon as sufficient for a multi-fission event. L41 therefore freezes one nonduplicative process before outcomes: at post-fission boundaries, a far-to-near transition relative to any eligible nonadjacent past boundary is one online return, and the second future return certifies the event. The primary ensemble follows exactly 12 future fissions; an independent short ensemble follows exactly four. Continuous residence does not generate repeated returns. Species-permuted, unrelated-history, branch-only, order-permuted, membership-only and ordinary-inheritance controls are fixed. No completed destination or author label is used.\n",
    )
    BASE.write_json(LOOP_ROOT / "immutable_prior_validation.json", prior)
    BASE.write_parquet(LOOP_ROOT / "fixture_results.parquet", fixtures)
    for name in (
        "response_registry.parquet",
        "original_target_coordinates.parquet",
        "input_trajectory_manifest.parquet",
        "prefix_boundary_registry.parquet",
        "prefix_state_summary.parquet",
        "unrelated_control_map.parquet",
        "species_permutation_manifest.parquet",
    ):
        shutil.copy2(L40_ROOT / name, LOOP_ROOT / name)
    BASE.write_parquet(LOOP_ROOT / "branch_seed_manifest.parquet", branch_seeds)
    BASE.write_parquet(LOOP_ROOT / "analysis_seed_manifest.parquet", analysis_seeds)
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
        "donorsSha256": sha256_file(LOOP_ROOT / "unrelated_control_map.parquet"),
        "permutationsSha256": sha256_file(LOOP_ROOT / "species_permutation_manifest.parquet"),
        "branchSeedsSha256": sha256_file(LOOP_ROOT / "branch_seed_manifest.parquet"),
        "analysisSeedsSha256": sha256_file(LOOP_ROOT / "analysis_seed_manifest.parquet"),
        "firewallSha256": sha256_file(LOOP_ROOT / "seed_firewall.json"),
        "benchmarkSha256": sha256_file(LOOP_ROOT / "benchmark_projection.json"),
        "l40ManifestSha256": sha256_file(L40_ROOT / "artifact_manifest.json"),
    }
    lock = {
        "schema": "eidosoma.e01.s19_l41.implementation_lock.v1",
        "repositoryHead": head,
        "remoteHead": remote,
        "runnerSha256": sha256_file(RUNNER_PATH),
        "coreSha256": sha256_file(CORE_PATH),
        "threshold": THRESHOLD,
        "minimumGenerationGap": MINIMUM_GENERATION_GAP,
        "requiredReturnBoundaries": REQUIRED_RETURN_BOUNDARIES,
        "futureFissionHorizons": FISSION_HORIZONS,
        "branchCounts": BRANCH_COUNTS,
        "targets": list(TARGETS),
        "matrixBootstraps": BOOTSTRAPS,
        "newMatrices": 0,
        "newTrajectories": 0,
        "newBranchStreams": 53_760,
        "completedTestTrajectoryUsed": False,
        "lockedHashes": hashes,
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
        raise RuntimeError("L41 repository lock mismatch")
    prior = validate_immutable_prior()
    fixtures = fixture_results()
    locked_files = {
        "responsesSha256": LOOP_ROOT / "response_registry.parquet",
        "coordinatesSha256": LOOP_ROOT / "original_target_coordinates.parquet",
        "manifestSha256": LOOP_ROOT / "input_trajectory_manifest.parquet",
        "boundariesSha256": LOOP_ROOT / "prefix_boundary_registry.parquet",
        "summariesSha256": LOOP_ROOT / "prefix_state_summary.parquet",
        "donorsSha256": LOOP_ROOT / "unrelated_control_map.parquet",
        "permutationsSha256": LOOP_ROOT / "species_permutation_manifest.parquet",
        "branchSeedsSha256": LOOP_ROOT / "branch_seed_manifest.parquet",
        "analysisSeedsSha256": LOOP_ROOT / "analysis_seed_manifest.parquet",
        "firewallSha256": LOOP_ROOT / "seed_firewall.json",
        "benchmarkSha256": LOOP_ROOT / "benchmark_projection.json",
        "l40ManifestSha256": L40_ROOT / "artifact_manifest.json",
    }
    for key, path in locked_files.items():
        if sha256_file(path) != lock[key]:
            raise RuntimeError(f"L41 locked input changed: {path}")
    if (
        not prior["unchanged"]
        or prior["aggregateSha256"] != lock["priorAggregateSha256"]
        or not fixtures["passed"].all()
        or sha256_file(RUNNER_PATH) != lock["runnerSha256"]
        or sha256_file(CORE_PATH) != lock["coreSha256"]
    ):
        raise RuntimeError("L41 pre-execution validation failed")
    payloads = build_payloads()
    responses = pd.read_parquet(LOOP_ROOT / "response_registry.parquet")
    boundaries = pd.read_parquet(LOOP_ROOT / "prefix_boundary_registry.parquet")
    summaries = pd.read_parquet(LOOP_ROOT / "prefix_state_summary.parquet")
    prefixes = L40.L39.prefix_controls(boundaries, summaries)

    if BUILD_ROOT.exists():
        shutil.rmtree(BUILD_ROOT)
    BUILD_ROOT.mkdir(parents=True)
    traces, outcomes = execute_branches(payloads)
    state_traces = state_trace_results(traces)
    states = state_committor_results(outcomes)
    reliability, reliability_bootstrap = reliability_results(states)
    pairs = transfer_pairs(states, state_traces, responses, prefixes)
    transfers, transfer_bootstrap = transfer_results(pairs)
    hazards, renewal = hazard_and_renewal(outcomes, traces)
    gates, classifications, next_theme = scientific_gates(reliability, transfers, states)
    make_figures(states, reliability, transfers, hazards, traces, gates)

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
        "unrelated_control_map.parquet",
        "species_permutation_manifest.parquet",
        "branch_seed_manifest.parquet",
        "analysis_seed_manifest.parquet",
        "seed_firewall.json",
        "benchmark_projection.json",
        "source_grounding_registry.parquet",
        "implementation_lock.json",
        "preoutcome_repository_lock.json",
    ):
        shutil.copy2(LOOP_ROOT / name, BUILD_ROOT / name)
    tables = {
        "branch_trace_results.parquet": traces,
        "branch_outcome_results.parquet": outcomes,
        "state_process_control_results.parquet": state_traces,
        "state_committor_results.parquet": states,
        "committor_reliability_results.parquet": reliability,
        "committor_reliability_bootstrap.parquet": reliability_bootstrap,
        "transfer_pairs.parquet": pairs,
        "transfer_results.parquet": transfers,
        "transfer_bootstrap.parquet": transfer_bootstrap,
        "boundary_hazard_results.parquet": hazards,
        "renewal_results.parquet": renewal,
        "scientific_gate_results.parquet": gates,
    }
    for name, frame in tables.items():
        BASE.write_parquet(BUILD_ROOT / name, frame)
    BASE.write_json(
        BUILD_ROOT / "classification.json",
        {
            "schema": "eidosoma.e01.s19_l41.classification.v1",
            "classifications": classifications,
            "repeatedRecurrenceCommittorEstablished": bool(
                gates["repeatedRecurrenceTargetPassed"].all()
            ),
            "shortFissionClockCoordinateEstablished": bool(
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
            "branchIndex",
            "exceptionClass",
            "exceptionMessage",
        ]
    ).to_csv(BUILD_ROOT / "failure_ledger.csv", index=False)

    replay_traces, replay_outcomes = execute_branches(payloads)
    replay_state_traces = state_trace_results(replay_traces)
    replay_states = state_committor_results(replay_outcomes)
    replay_reliability, replay_reliability_bootstrap = reliability_results(replay_states)
    replay_pairs = transfer_pairs(replay_states, replay_state_traces, responses, prefixes)
    replay_transfers, replay_transfer_bootstrap = transfer_results(replay_pairs)
    replay_hazards, replay_renewal = hazard_and_renewal(replay_outcomes, replay_traces)
    replay_gates, replay_classifications, replay_next = scientific_gates(
        replay_reliability, replay_transfers, replay_states
    )
    replay_tables = {
        "traces": (traces, replay_traces),
        "outcomes": (outcomes, replay_outcomes),
        "stateTraces": (state_traces, replay_state_traces),
        "states": (states, replay_states),
        "reliability": (reliability, replay_reliability),
        "reliabilityBootstrap": (
            reliability_bootstrap,
            replay_reliability_bootstrap,
        ),
        "pairs": (pairs, replay_pairs),
        "transfers": (transfers, replay_transfers),
        "transferBootstrap": (transfer_bootstrap, replay_transfer_bootstrap),
        "hazards": (hazards, replay_hazards),
        "renewal": (renewal, replay_renewal),
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
            "seedFirewallPassed": json.loads(
                (LOOP_ROOT / "seed_firewall.json").read_text()
            )["status"]
            == "PASS",
            "branchIdentitiesExact": traces["branchIdentitySha256"].equals(
                replay_traces["branchIdentitySha256"]
            ),
            "pathDigestsExact": traces["pathSha256"].equals(
                replay_traces["pathSha256"]
            ),
            "noCompletedTarget": bool(
                (~outcomes["targetUsesCompletedTestTrajectory"]).all()
            ),
            "onlineReturnCountInvariant": bool(
                outcomes.loc[outcomes["event"], "returnBoundaryCount"].ge(2).all()
            ),
            "membershipSupersetInvariant": bool(
                outcomes["membershipBoundaryCount"].ge(
                    outcomes["returnBoundaryCount"]
                ).all()
            ),
            "noNewMatrix": True,
            "noNewTrajectory": True,
        }
    )
    if not all(checks.values()):
        raise RuntimeError(f"L41 regeneration validation failed: {checks}")
    BASE.write_json(
        BUILD_ROOT / "regeneration_validation.json",
        {
            "schema": "eidosoma.e01.s19_l41.regeneration_validation.v1",
            "status": "PASS",
            "checks": checks,
            "traceFrameSha256": frame_hash(traces),
            "outcomeFrameSha256": frame_hash(outcomes),
            "stateFrameSha256": frame_hash(states),
            "gateFrameSha256": frame_hash(gates),
        },
    )
    runtime = {
        "schema": "eidosoma.e01.s19_l41.runtime.v1",
        "repositoryHead": git("rev-parse", "HEAD"),
        "workers": WORKERS,
        "numericalLibraryThreadsPerWorker": 1,
        "gpuHours": 0,
        "wallSeconds": time.perf_counter() - started,
        "controllerCpuHours": (time.process_time() - started_cpu) / 3600,
        "states": 280,
        "newBranchStreams": 53_760,
        "f12BranchStreams": 280 * BRANCH_COUNTS["F12"],
        "f4BranchStreams": 280 * BRANCH_COUNTS["F4"],
        "targetScoresPerBranch": len(TARGETS),
        "newMatrices": 0,
        "newTrajectories": 0,
        "completedAtUtc": utc_now(),
    }
    BASE.write_json(BUILD_ROOT / "runtime_manifest.json", runtime)
    retained = sum(path.stat().st_size for path in BUILD_ROOT.rglob("*") if path.is_file())
    temporary = sum(path.stat().st_size for path in CACHE_ROOT.rglob("*") if path.is_file())
    storage = {
        "schema": "eidosoma.e01.s19_l41.storage_validation.v1",
        "retainedBytes": retained,
        "retainedGiBCeiling": 25,
        "temporaryBytes": temporary,
        "temporaryGiBCeiling": 75,
        "status": "PASS"
        if retained < 25 * 2**30 and temporary < 75 * 2**30
        else "FAIL",
    }
    if storage["status"] != "PASS":
        raise RuntimeError("L41 storage ceiling exceeded")
    BASE.write_json(BUILD_ROOT / "storage_validation.json", storage)
    report = report_text(
        states,
        reliability,
        transfers,
        renewal,
        gates,
        classifications,
        runtime,
        next_theme,
    )
    BASE.atomic_text(BUILD_ROOT / "research_step_full_results.md", report)
    BASE.atomic_text(BUILD_ROOT / "S19_L41_FULL_RESULTS.md", report)
    BASE.atomic_text(
        BUILD_ROOT / "loop_decision_summary.md",
        "# S19-L41 decision summary\n\n"
        + f"**Classification:** {', '.join(classifications)}\n\n"
        + f"**All-group repeated-recurrence target:** `{gates['repeatedRecurrenceTargetPassed'].all()}`.\n\n"
        + f"**All-group F4 shooting coordinate:** `{gates['shortShootingCoordinatePassed'].all()}`.\n\n"
        + f"**Next:** `{next_theme}`.\n",
    )
    BASE.write_json(BUILD_ROOT / "artifact_manifest.json", manifest_for(BUILD_ROOT))
    stage = LOOP_ROOT.with_name(".L41-promotion-stage")
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
        raise RuntimeError("L41 artifact hash validation failed")
    append_ledgers(classifications, runtime["completedAtUtc"], next_theme)
    BASE.atomic_text(ARTIFACT_ROOT / "research_step_full_results.md", report)
    BASE.atomic_text(
        ARTIFACT_ROOT / "S19_CURRENT_HANDOFF.md",
        report.replace("# S19-L41", "# S19 current handoff — S19-L41", 1),
    )
    BASE.write_json(
        ARTIFACT_ROOT / "s19_status.json",
        {
            "schema": "eidosoma.e01.s19.status.v1",
            "status": "ACTIVE_AUTONOMOUS_SEQUENCE",
            "latestCompletedLoop": LOOP_ID,
            "latestClassification": classifications,
            "selectedDiscoveryLead": (
                "FISSION_CLOCK_REPEATED_RECURRENCE_PROCESS"
                if "PROMOTABLE_TO_UNTOUCHED_PROCESS_CONFIRMATION" in classifications
                else None
            ),
            "nextAuthorizedLoop": "S19-L42",
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
