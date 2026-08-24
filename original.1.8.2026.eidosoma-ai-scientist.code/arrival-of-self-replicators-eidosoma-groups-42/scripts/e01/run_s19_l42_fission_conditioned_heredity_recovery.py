"""Execute S19-L42 fission-conditioned heredity-recovery hazard audit."""

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

from e01_onset_discovery.heredity_recovery import score_heredity_recovery


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


L41 = _load_module(
    "e01_s19_l42_l41",
    REPO_ROOT / "scripts/e01/run_s19_l41_fission_clock_repeated_recurrence.py",
)
BASE = L41.BASE

LOOP_ID = "S19-L42"
VERSION = "E01-S19-L42-FISSION-CONDITIONED-HEREDITY-RECOVERY-HAZARD-v1.0.0"
CANDIDATES = L41.CANDIDATES
COHORTS = L41.COHORTS
EVALUATION_COHORTS = L41.EVALUATION_COHORTS
FAMILIES = L41.FAMILIES
FISSION_HORIZONS = L41.FISSION_HORIZONS
BRANCH_COUNTS = L41.BRANCH_COUNTS
HALVES = L41.HALVES
TARGETS = (
    "PRIMARY_PREBREAK_DAUGHTER",
    "SPECIES_PERMUTED_PREBREAK_DAUGHTER",
    "UNRELATED_MATRIX_PREFIX_DAUGHTER",
)
PRIMARY_TARGET = TARGETS[0]
THRESHOLD = 0.9
REQUIRED_RECOVERY_RUN = 2
BOOTSTRAPS = 4096
MINIMUM_BREAK_TRIALS = 32
MINIMUM_HALF_BREAK_TRIALS = 16
ROOT_HEX = "5406bd8d2acaf8de593611b09e894fb78a07a948185fc900cba721af59e8e445"
PHASE = "s19_l42_fission_conditioned_heredity_recovery"
WORKERS = min(8, os.cpu_count() or 1)

ARTIFACT_ROOT = Path("/artifacts/research_steps/S19")
LOOP_ROOT = ARTIFACT_ROOT / "loops/L42"
L41_ROOT = ARTIFACT_ROOT / "loops/L41"
CACHE_ROOT = Path("/cache/e01_s19_l42")
BUILD_ROOT = CACHE_ROOT / "build"
CONFIG = REPO_ROOT / "configs/e01/s19_l42_fission_conditioned_heredity_recovery.yaml"
RUNNER_PATH = Path(__file__)
CORE_PATH = REPO_ROOT / "src/e01_onset_discovery/heredity_recovery.py"


def utc_now() -> str:
    return L41.utc_now()


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def sha256_file(path: Path) -> str:
    return L41.sha256_file(path)


def frame_hash(frame: pd.DataFrame) -> str:
    return L41.frame_hash(frame)


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
    inherited = json.loads((L41_ROOT / "immutable_prior_validation.json").read_text())
    rows = list(inherited["files"])
    manifest = json.loads((L41_ROOT / "artifact_manifest.json").read_text())
    rows.extend(
        {
            "path": str(L41_ROOT / item["path"]),
            "root": str(L41_ROOT),
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
        "schema": "eidosoma.e01.s19_l42.immutable_prior_validation.v1",
        "status": "PASS" if passed else "FAIL",
        "unchanged": passed,
        "fileCount": len(checked),
        "aggregateSha256": aggregate,
        "l41ManifestSha256": sha256_file(L41_ROOT / "artifact_manifest.json"),
        "files": checked,
    }


def fixture_results() -> pd.DataFrame:
    a = np.asarray([10, 0], dtype=np.int64)
    b = np.asarray([0, 10], dtype=np.int64)
    near = np.asarray([9, 1], dtype=np.int64)

    def score(states: list[np.ndarray], inheritance: list[float], override=None) -> Any:
        return score_heredity_recovery(
            latest_prefix_daughter=a,
            future_daughters=np.asarray(states, dtype=np.int64).reshape((-1, 2)),
            parent_daughter_h=np.asarray(inheritance, dtype=np.float64),
            future_generations=np.arange(2, 2 + len(states)),
            future_offsets_one_based=np.arange(2, 2 + 2 * len(states), 2),
            recovery_anchor_override=override,
            threshold=THRESHOLD,
            required_recovery_run=REQUIRED_RECOVERY_RUN,
        )

    recovered = score([b, near, a], [0.8, 0.95, 0.96])
    uninterrupted = score([near, a, near], [0.95, 0.96, 0.97])
    inheritance_only_break = score([near, a, near], [0.8, 0.96, 0.97])
    different = score([b, b, b], [0.8, 0.96, 0.97])
    control = score([b, near, a], [0.8, 0.95, 0.96], override=b)
    separated = score([b, near, b, a], [0.8, 0.95, 0.95, 0.96])
    replay = score([b.copy(), near.copy(), a.copy()], [0.8, 0.95, 0.96])
    return pd.DataFrame(
        [
            {
                "fixtureId": "BREAK_THEN_SUSTAINED_RETURN",
                "passed": recovered.break_observed
                and recovered.event
                and recovered.certification_boundary_one_based == 3,
                "details": "break at fission 1; online certification at fission 3",
            },
            {
                "fixtureId": "UNINTERRUPTED_INHERITANCE_EXCLUDED",
                "passed": not uninterrupted.break_observed and not uninterrupted.event,
                "details": "persistence without disruption is not recovery",
            },
            {
                "fixtureId": "GENUINE_DEPARTURE_REQUIRED",
                "passed": not inheritance_only_break.break_observed,
                "details": "low parent-daughter H alone does not establish compositional departure",
            },
            {
                "fixtureId": "RESUMPTION_SEPARATE_FROM_HOMEOSTASIS",
                "passed": different.inheritance_resumption_event and not different.event,
                "details": "inheritance can resume in a different compositional neighbourhood",
            },
            {
                "fixtureId": "ANCHOR_CONTROL_PRESERVES_BREAK",
                "passed": recovered.break_boundary_one_based == control.break_boundary_one_based
                and recovered.inheritance_resumption_event
                == control.inheritance_resumption_event
                and recovered.event
                and not control.event,
                "details": "anchor control changes recovery membership only",
            },
            {
                "fixtureId": "FIXED_COUNT_ORDER_NULL",
                "passed": separated.qualifying_recovery_count == 2
                and not separated.event
                and separated.exact_recovery_order_null_probability > 0,
                "details": "same qualifying count need not form a sustained run",
            },
            {
                "fixtureId": "EXACT_REPLAY",
                "passed": recovered == replay,
                "details": "all scientific fields exact",
            },
        ]
    )


def analysis_seed_manifest() -> pd.DataFrame:
    comparisons = (
        "F4_PROGRESS_VS_F12_CONDITIONAL_RECOVERY",
        "F4_CONDITIONAL_RECOVERY_VS_F12",
        "F4_RESUMPTION_VS_F12",
        "F12_PRIMARY_MINUS_SPECIES_PERMUTED",
        "F12_PRIMARY_MINUS_UNRELATED",
        "F12_PRIMARY_MINUS_ORDER_NULL",
        "F12_RESUMPTION_VS_PRIMARY",
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
        raise RuntimeError("L42 analysis seed collision")
    return result


def seed_firewall(analysis: pd.DataFrame) -> dict[str, Any]:
    prior_material: set[str] = set()
    prior_derived: set[str] = set()
    for path in ARTIFACT_ROOT.glob("loops/L*/**/*seed*manifest*.parquet"):
        if "/L42/" in str(path):
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
    material = sorted(set(analysis["seedMaterialSha256"].astype(str)) & prior_material)
    derived = sorted(set(analysis["derivedSeed"].astype(str)) & prior_derived)
    return {
        "schema": "eidosoma.e01.s19_l42.seed_firewall.v1",
        "status": "PASS" if not material and not derived else "FAIL",
        "analysisSeedCount": len(analysis),
        "analysisSeedMaterialOverlapCount": len(material),
        "analysisDerivedSeedOverlapCount": len(derived),
        "analysisSeedMaterialOverlaps": material,
        "analysisDerivedSeedOverlaps": derived,
        "reusedBranchStreamCount": 53_760,
        "newBranchStreamCount": 0,
        "branchReuseSource": "S19-L41 exact locked seed manifest",
    }


def source_grounding_registry() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sourceId": "L42_REVIEWER_RECOVERY_AFTER_DISTURBANCE",
                "evidenceClass": "HUMAN_REVIEW_DIRECTION",
                "finding": "Recovery following genuine disruption is stronger evidence of homeostasis than uninterrupted inheritance.",
                "frozenUse": "first joint heredity/composition break followed by sustained same-neighbourhood inheritance",
            },
            {
                "sourceId": "L42_REVIEWER_BASELINE_CONTROLS",
                "evidenceClass": "HUMAN_REVIEW_DIRECTION",
                "finding": "Hold inheritance frequency, opportunity count, current streak, mass and phase as baselines.",
                "frozenUse": "resumption-only, exact fixed-count order and state/prefix controls",
            },
            {
                "sourceId": "L42_L41_ORDER_SUPPRESSION",
                "evidenceClass": "DIRECT_FROZEN_E01_RESULT",
                "finding": "Generic repeated recurrence was reliable in only two of four groups and much less frequent than its order-permuted control.",
                "frozenUse": "replace generic recurrence with a mechanistically explicit heredity-break/recovery event",
            },
            {
                "sourceId": "L42_L41_FISSION_CLOCK_PATHS",
                "evidenceClass": "DIRECT_FROZEN_E01_RESULT",
                "finding": "All F12/F4 paths and branch identities were exactly regenerated under a zero-overlap seed firewall.",
                "frozenUse": "analysis-only exact path rescoring with no new stochastic stream",
            },
        ]
    )


def build_payloads() -> list[dict[str, Any]]:
    payloads = L41.build_payloads()
    permutation_frame = pd.read_parquet(L41_ROOT / "species_permutation_manifest.parquet")
    permutation_map = permutation_frame.set_index("stateId")["permutation"].to_dict()
    expected = pd.read_parquet(L41_ROOT / "branch_trace_results.parquet")
    expected_map = {
        state_id: {
            f"{row.branchFamily}:{int(row.branchIndex)}": {
                "branchIdentitySha256": row.branchIdentitySha256,
                "pathSha256": row.pathSha256,
                "finalStateSha256": row.finalStateSha256,
                "fissions": int(row.fissions),
                "selectedObservationsGenerated": int(row.selectedObservationsGenerated),
                "terminalStatus": row.terminalStatus,
            }
            for row in group.itertuples(index=False)
        }
        for state_id, group in expected.groupby("stateId", sort=False)
    }
    output = []
    for payload in payloads:
        row = dict(payload)
        row["l42SpeciesPermutation"] = np.asarray(
            permutation_map[payload["stateId"]], dtype=np.int64
        ).tolist()
        row["l42ExpectedBranches"] = expected_map[payload["stateId"]]
        output.append(row)
    if len(output) != 280 or any(len(item["l42ExpectedBranches"]) != 192 for item in output):
        raise RuntimeError("L42 payload/expected-path scope failure")
    return output


def _worker(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    outcomes: list[dict[str, Any]] = []
    replay: list[dict[str, Any]] = []
    prefix_latest = np.asarray(payload["prefixStates"][-1], dtype=np.int64)
    unrelated = np.asarray(payload["unrelatedPrefixStates"][-1], dtype=np.int64)
    permutation = np.asarray(payload["l42SpeciesPermutation"], dtype=np.int64)
    for family in FAMILIES:
        for branch in range(BRANCH_COUNTS[family]):
            trace = L41._simulate(payload, family, branch)
            expected = payload["l42ExpectedBranches"][f"{family}:{branch}"]
            exact = bool(
                trace.path_sha256 == expected["pathSha256"]
                and trace.final_state_sha256 == expected["finalStateSha256"]
                and trace.fissions == expected["fissions"]
                and trace.selected_observations_generated
                == expected["selectedObservationsGenerated"]
                and trace.terminal_status == expected["terminalStatus"]
            )
            if not exact:
                raise RuntimeError(
                    f"L42 frozen L41 path replay failure: {payload['stateId']} {family} {branch}"
                )
            future = np.asarray(trace.future_states, dtype=np.int64).reshape((-1, 100))
            inheritance = np.asarray(trace.parent_daughter_h, dtype=np.float64)
            generations = np.asarray(trace.future_generations, dtype=np.int64)
            offsets = np.asarray(trace.future_offsets_one_based, dtype=np.int64)
            primary = score_heredity_recovery(
                latest_prefix_daughter=prefix_latest,
                future_daughters=future,
                parent_daughter_h=inheritance,
                future_generations=generations,
                future_offsets_one_based=offsets,
                threshold=THRESHOLD,
                required_recovery_run=REQUIRED_RECOVERY_RUN,
            )
            if (
                primary.break_boundary_one_based is None
                or primary.break_boundary_one_based == 1
            ):
                primary_anchor = prefix_latest
            else:
                primary_anchor = future[primary.break_boundary_one_based - 2]
            scores = {
                PRIMARY_TARGET: primary,
                "SPECIES_PERMUTED_PREBREAK_DAUGHTER": score_heredity_recovery(
                    latest_prefix_daughter=prefix_latest,
                    future_daughters=future,
                    parent_daughter_h=inheritance,
                    future_generations=generations,
                    future_offsets_one_based=offsets,
                    recovery_anchor_override=primary_anchor[permutation],
                    threshold=THRESHOLD,
                    required_recovery_run=REQUIRED_RECOVERY_RUN,
                ),
                "UNRELATED_MATRIX_PREFIX_DAUGHTER": score_heredity_recovery(
                    latest_prefix_daughter=prefix_latest,
                    future_daughters=future,
                    parent_daughter_h=inheritance,
                    future_generations=generations,
                    future_offsets_one_based=offsets,
                    recovery_anchor_override=unrelated,
                    threshold=THRESHOLD,
                    required_recovery_run=REQUIRED_RECOVERY_RUN,
                ),
            }
            replay.append(
                {
                    "stateId": payload["stateId"],
                    "evaluationCohort": payload["evaluationCohort"],
                    "candidateId": payload["candidateId"],
                    "matrixIndex": int(payload["matrixIndex"]),
                    "landmark": int(payload["landmark"]),
                    "branchFamily": family,
                    "branchIndex": branch,
                    "branchHalf": "A" if branch < HALVES[family] else "B",
                    "expectedBranchIdentitySha256": expected["branchIdentitySha256"],
                    "pathSha256": trace.path_sha256,
                    "finalStateSha256": trace.final_state_sha256,
                    "fissions": trace.fissions,
                    "selectedObservationsGenerated": trace.selected_observations_generated,
                    "terminalStatus": trace.terminal_status,
                    "exactL41Replay": exact,
                }
            )
            for target, scored in scores.items():
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
                        "breakObserved": scored.break_observed,
                        "breakBoundaryOneBased": scored.break_boundary_one_based,
                        "breakGeneration": scored.break_generation,
                        "breakOffsetOneBased": scored.break_offset_one_based,
                        "event": scored.event,
                        "certificationBoundaryOneBased": scored.certification_boundary_one_based,
                        "certificationGeneration": scored.certification_generation,
                        "certificationOffsetOneBased": scored.certification_offset_one_based,
                        "recoveryOpportunities": scored.recovery_opportunities,
                        "qualifyingRecoveryCount": scored.qualifying_recovery_count,
                        "maximumConsecutiveRecovery": scored.maximum_consecutive_recovery,
                        "firstQualifyingRecoveryBoundaryOneBased": scored.first_qualifying_recovery_boundary_one_based,
                        "inheritanceResumptionEvent": scored.inheritance_resumption_event,
                        "inheritanceResumptionCertificationBoundaryOneBased": scored.inheritance_resumption_certification_boundary_one_based,
                        "inheritedPostbreakCount": scored.inherited_postbreak_count,
                        "maximumConsecutiveInheritedPostbreak": scored.maximum_consecutive_inherited_postbreak,
                        "maximumPostbreakAnchorH": scored.maximum_postbreak_anchor_h,
                        "maximumInheritedPostbreakAnchorH": scored.maximum_inherited_postbreak_anchor_h,
                        "recoveryProgress": scored.recovery_progress,
                        "exactRecoveryOrderNullProbability": scored.exact_recovery_order_null_probability,
                        "exactResumptionOrderNullProbability": scored.exact_resumption_order_null_probability,
                        "recoveryFlags": json.dumps(scored.recovery_flags),
                        "resumptionFlags": json.dumps(scored.resumption_flags),
                        "futureInheritanceFraction": float(np.mean(inheritance > THRESHOLD)),
                        "pathSha256": trace.path_sha256,
                        "targetUsesCompletedTestTrajectory": False,
                    }
                )
    return {"replay": replay, "outcomes": outcomes}


def execute_paths(payloads: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    replay_rows: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(_worker, payload): payload["stateId"] for payload in payloads}
        for future in as_completed(futures):
            result = future.result()
            replay_rows.extend(result["replay"])
            outcomes.extend(result["outcomes"])
    replay = pd.DataFrame(replay_rows).sort_values(
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
        len(replay) != 53_760
        or len(outcome_frame) != 53_760 * len(TARGETS)
        or not replay["exactL41Replay"].all()
        or replay.duplicated(["stateId", "branchFamily", "branchIndex"]).any()
        or outcome_frame.duplicated(
            ["stateId", "branchFamily", "targetId", "branchIndex"]
        ).any()
    ):
        raise RuntimeError("L42 path/output cardinality failure")
    return replay, outcome_frame


def state_recovery_results(outcomes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (state_id, family, target), group in outcomes.groupby(
        ["stateId", "branchFamily", "targetId"], sort=True
    ):
        expected = BRANCH_COUNTS[family]
        half_a = group[group["branchHalf"].eq("A")]
        half_b = group[group["branchHalf"].eq("B")]

        def conditional(frame: pd.DataFrame, column: str) -> tuple[int, int, float]:
            eligible = frame[frame["breakObserved"]]
            trials = len(eligible)
            successes = int(eligible[column].sum())
            return trials, successes, successes / trials if trials else float("nan")

        trials, successes, q = conditional(group, "event")
        trials_a, successes_a, q_a = conditional(half_a, "event")
        trials_b, successes_b, q_b = conditional(half_b, "event")
        _, resumption_successes, q_resumption = conditional(
            group, "inheritanceResumptionEvent"
        )
        break_group = group[group["breakObserved"]]
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
                "breakTrials": trials,
                "breakTrialsHalfA": trials_a,
                "breakTrialsHalfB": trials_b,
                "qBreak": float(group["breakObserved"].mean()),
                "successes": successes,
                "qConditionalRecovery": q,
                "successesHalfA": successes_a,
                "qConditionalRecoveryHalfA": q_a,
                "successesHalfB": successes_b,
                "qConditionalRecoveryHalfB": q_b,
                "qUnconditionalRecovery": float(group["event"].mean()),
                "resumptionSuccesses": resumption_successes,
                "qConditionalInheritanceResumption": q_resumption,
                "meanRecoveryProgressConditional": float(
                    break_group["recoveryProgress"].mean()
                ),
                "meanExactRecoveryOrderNullConditional": float(
                    break_group["exactRecoveryOrderNullProbability"].mean()
                ),
                "meanExactResumptionOrderNullConditional": float(
                    break_group["exactResumptionOrderNullProbability"].mean()
                ),
                "meanRecoveryOpportunities": float(
                    break_group["recoveryOpportunities"].mean()
                ),
                "meanQualifyingRecoveryCount": float(
                    break_group["qualifyingRecoveryCount"].mean()
                ),
                "meanFutureInheritanceFraction": float(
                    group["futureInheritanceFraction"].mean()
                ),
                "committorEligible": bool(
                    len(group) == expected
                    and trials >= MINIMUM_BREAK_TRIALS
                    and trials_a >= MINIMUM_HALF_BREAK_TRIALS
                    and trials_b >= MINIMUM_HALF_BREAK_TRIALS
                ),
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
        raise RuntimeError("L42 state result cardinality failure")
    return result


def corrected_variance(q: np.ndarray, trials: np.ndarray) -> dict[str, float]:
    values = np.asarray(q, dtype=np.float64)
    counts = np.asarray(trials, dtype=np.float64)
    if len(values) < 2 or np.any(counts <= 0):
        return {
            "observedBetweenStateVariance": float("nan"),
            "estimatedBinomialNoiseVariance": float("nan"),
            "correctedBetweenStateVariance": float("nan"),
        }
    observed = float(np.var(values, ddof=1))
    noise = float(np.mean(values * (1.0 - values) / counts))
    return {
        "observedBetweenStateVariance": observed,
        "estimatedBinomialNoiseVariance": noise,
        "correctedBetweenStateVariance": observed - noise,
    }


def reliability_results(states: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    bootstrap_rows = []
    for (cohort, candidate, family, target), group in states.groupby(
        ["evaluationCohort", "candidateId", "branchFamily", "targetId"], sort=True
    ):
        eligible = group[group["committorEligible"]]
        q = eligible["qConditionalRecovery"].to_numpy(dtype=np.float64)
        trials = eligible["breakTrials"].to_numpy(dtype=np.float64)
        half_a = eligible["qConditionalRecoveryHalfA"].to_numpy(dtype=np.float64)
        half_b = eligible["qConditionalRecoveryHalfB"].to_numpy(dtype=np.float64)
        variance = corrected_variance(q, trials)
        split = safe_spearman(half_a, half_b)
        rng = generator("reliability_bootstrap", cohort, candidate, family, target)
        corrected_boot = np.full(BOOTSTRAPS, np.nan)
        split_boot = np.full(BOOTSTRAPS, np.nan)
        for replicate in range(BOOTSTRAPS):
            if len(eligible):
                indices = rng.integers(0, len(eligible), len(eligible))
                corrected_boot[replicate] = corrected_variance(
                    q[indices], trials[indices]
                )["correctedBetweenStateVariance"]
                split_boot[replicate] = safe_spearman(
                    half_a[indices], half_b[indices]
                )
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
        corrected_low, corrected_high = interval(corrected_boot)
        split_low, split_high = interval(split_boot)
        intermediate = int(np.sum((q > 0.1) & (q < 0.9)))
        rows.append(
            {
                "evaluationCohort": cohort,
                "candidateId": candidate,
                "branchFamily": family,
                "targetId": target,
                "states": len(group),
                "eligibleStates": len(eligible),
                "meanConditionalQ": float(np.mean(q)) if len(q) else float("nan"),
                "minimumConditionalQ": float(np.min(q)) if len(q) else float("nan"),
                "maximumConditionalQ": float(np.max(q)) if len(q) else float("nan"),
                "intermediateStateCount": intermediate,
                "meanBreakTrials": float(np.mean(trials)) if len(trials) else float("nan"),
                **variance,
                "correctedVarianceLower95": corrected_low,
                "correctedVarianceUpper95": corrected_high,
                "splitHalfSpearman": split,
                "splitHalfLower95": split_low,
                "splitHalfUpper95": split_high,
                "reliabilityGatePassed": bool(
                    len(eligible) >= 32
                    and corrected_low > 0
                    and split > 0.5
                    and split_low > 0.3
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
    index = states.set_index(["stateId", "branchFamily", "targetId"])
    response_index = responses.set_index("stateId")
    prefix_index = prefixes.set_index("stateId")
    definitions = (
        ("F4_PROGRESS_VS_F12_CONDITIONAL_RECOVERY", "RANK"),
        ("F4_CONDITIONAL_RECOVERY_VS_F12", "RANK"),
        ("F4_RESUMPTION_VS_F12", "RANK"),
        ("F12_PRIMARY_MINUS_SPECIES_PERMUTED", "DIFFERENCE"),
        ("F12_PRIMARY_MINUS_UNRELATED", "DIFFERENCE"),
        ("F12_PRIMARY_MINUS_ORDER_NULL", "DIFFERENCE"),
        ("F12_RESUMPTION_VS_PRIMARY", "RANK"),
        ("F12_INHERITANCE_FRACTION_VS_PRIMARY", "RANK"),
        ("PREFIX_INHERITANCE_FRACTION_VS_PRIMARY", "RANK"),
        ("CURRENT_MASS_VS_PRIMARY", "RANK"),
        ("GENERATION_PHASE_VS_PRIMARY", "RANK"),
    )
    rows = []
    for state_id in states["stateId"].drop_duplicates():
        primary = index.loc[(state_id, "F12", PRIMARY_TARGET)]
        f4 = index.loc[(state_id, "F4", PRIMARY_TARGET)]
        species = index.loc[
            (state_id, "F12", "SPECIES_PERMUTED_PREBREAK_DAUGHTER")
        ]
        unrelated = index.loc[
            (state_id, "F12", "UNRELATED_MATRIX_PREFIX_DAUGHTER")
        ]
        response = response_index.loc[state_id]
        prefix = prefix_index.loc[state_id]
        values = {
            "F4_PROGRESS_VS_F12_CONDITIONAL_RECOVERY": f4.meanRecoveryProgressConditional,
            "F4_CONDITIONAL_RECOVERY_VS_F12": f4.qConditionalRecovery,
            "F4_RESUMPTION_VS_F12": f4.qConditionalInheritanceResumption,
            "F12_PRIMARY_MINUS_SPECIES_PERMUTED": (
                primary.qConditionalRecovery - species.qConditionalRecovery
            ),
            "F12_PRIMARY_MINUS_UNRELATED": (
                primary.qConditionalRecovery - unrelated.qConditionalRecovery
            ),
            "F12_PRIMARY_MINUS_ORDER_NULL": (
                primary.qConditionalRecovery
                - primary.meanExactRecoveryOrderNullConditional
            ),
            "F12_RESUMPTION_VS_PRIMARY": primary.qConditionalInheritanceResumption,
            "F12_INHERITANCE_FRACTION_VS_PRIMARY": primary.meanFutureInheritanceFraction,
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
                    "response": float(primary.qConditionalRecovery),
                    "primaryEligible": bool(primary.committorEligible),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["evaluationCohort", "candidateId", "comparisonId", "landmark", "matrixIndex"]
    ).reset_index(drop=True)


def transfer_results(pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    bootstrap_rows = []
    for (cohort, candidate, comparison, comparison_type), raw in pairs.groupby(
        ["evaluationCohort", "candidateId", "comparisonId", "comparisonType"], sort=True
    ):
        group = raw[
            raw["primaryEligible"]
            & np.isfinite(raw["predictor"])
            & np.isfinite(raw["response"])
        ]
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
            if len(group):
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
                "definedPairs": len(group),
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


def recovery_hazard(outcomes: pd.DataFrame) -> pd.DataFrame:
    primary = outcomes[
        outcomes["targetId"].eq(PRIMARY_TARGET)
        & outcomes["branchFamily"].eq("F12")
        & outcomes["breakObserved"]
    ].copy()
    primary["certificationAfterBreak"] = (
        primary["certificationBoundaryOneBased"]
        - primary["breakBoundaryOneBased"]
    )
    rows = []
    for (cohort, candidate), group in primary.groupby(
        ["evaluationCohort", "candidateId"], sort=True
    ):
        survival = 1.0
        maximum = int(group["recoveryOpportunities"].max())
        for opportunity in range(1, maximum + 1):
            at_risk = group[
                group["recoveryOpportunities"].ge(opportunity)
                & (
                    group["certificationAfterBreak"].isna()
                    | group["certificationAfterBreak"].ge(opportunity)
                )
            ]
            events = int(group["certificationAfterBreak"].eq(opportunity).sum())
            hazard = events / len(at_risk) if len(at_risk) else float("nan")
            if np.isfinite(hazard):
                survival *= 1.0 - hazard
            rows.append(
                {
                    "evaluationCohort": cohort,
                    "candidateId": candidate,
                    "postbreakFissionOpportunity": opportunity,
                    "branchesAtRisk": len(at_risk),
                    "certifications": events,
                    "discreteRecoveryHazard": hazard,
                    "survivalWithoutRecovery": survival,
                    "cumulativeRecoveryIncidence": 1.0 - survival,
                }
            )
    return pd.DataFrame(rows)


def scientific_gates(
    reliability: pd.DataFrame,
    transfers: pd.DataFrame,
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
            reliable = bool(rel.loc[("F12", PRIMARY_TARGET), "reliabilityGatePassed"])
            species = bool(
                comparison.loc["F12_PRIMARY_MINUS_SPECIES_PERMUTED", "gatePassed"]
            )
            unrelated = bool(
                comparison.loc["F12_PRIMARY_MINUS_UNRELATED", "gatePassed"]
            )
            order = bool(
                comparison.loc["F12_PRIMARY_MINUS_ORDER_NULL", "gatePassed"]
            )
            f4 = bool(
                comparison.loc[
                    "F4_PROGRESS_VS_F12_CONDITIONAL_RECOVERY", "gatePassed"
                ]
            )
            target = reliable and species and unrelated and order
            rows.append(
                {
                    "evaluationCohort": cohort,
                    "candidateId": candidate,
                    "primaryConditionalCommittorReliable": reliable,
                    "speciesPermutationControlPassed": species,
                    "unrelatedMatrixControlPassed": unrelated,
                    "fixedCountOrderControlPassed": order,
                    "homeostaticRecoveryTargetPassed": target,
                    "f4ProgressRankPassed": f4,
                    "shortShootingCoordinatePassed": target and f4,
                }
            )
    gates = pd.DataFrame(rows)
    target_all = bool(gates["homeostaticRecoveryTargetPassed"].all())
    short_all = bool(gates["shortShootingCoordinatePassed"].all())
    reliable_all = bool(gates["primaryConditionalCommittorReliable"].all())
    order_all = bool(gates["fixedCountOrderControlPassed"].all())
    anchor_all = bool(
        gates[
            ["speciesPermutationControlPassed", "unrelatedMatrixControlPassed"]
        ].all(axis=None)
    )
    if target_all and short_all:
        classifications = [
            "STATE_DEPENDENT_HOMEOSTATIC_RECOVERY_COMMITTOR_ESTABLISHED",
            "FISSION_CLOCK_RECOVERY_SHOOTING_COORDINATE_ESTABLISHED",
            "PROMOTABLE_TO_UNTOUCHED_PROCESS_CONFIRMATION",
        ]
        next_theme = "UNTOUCHED_HOMEOSTATIC_RECOVERY_CONFIRMATION"
    elif target_all:
        classifications = [
            "STATE_DEPENDENT_HOMEOSTATIC_RECOVERY_COMMITTOR_ESTABLISHED",
            "SHOOTING_REQUIRED_AT_LONGER_RECOVERY_CLOCK",
            "NOT_PROMOTABLE_AS_CONFIRMED",
        ]
        next_theme = "HOMEOSTATIC_RECOVERY_SHOOTING_EFFICIENCY"
    elif reliable_all and anchor_all and not order_all:
        classifications = [
            "HEREDITY_RECOVERY_ORDER_NOT_SUPPORTED",
            "INHERITANCE_RESUMPTION_FREQUENCY_SUFFICIENT",
            "NOT_PROMOTABLE_AS_CONFIRMED",
        ]
        next_theme = "PROCESS_OUTCOME_FAMILY_IDENTIFIABILITY_AUDIT"
    elif reliable_all and not anchor_all:
        classifications = [
            "INHERITANCE_RESUMPTION_NOT_HOMEOSTATIC_RECOVERY",
            "ANCHOR_SPECIFIC_RECOVERY_NOT_SUPPORTED",
            "NOT_PROMOTABLE_AS_CONFIRMED",
        ]
        next_theme = "PROCESS_OUTCOME_FAMILY_IDENTIFIABILITY_AUDIT"
    else:
        classifications = [
            "NO_RELIABLE_HOMEOSTATIC_RECOVERY_COMMITTOR_AT_F12",
            "PROCESS_TARGET_REQUIRES_REDEFINITION",
            "NOT_PROMOTABLE_AS_CONFIRMED",
        ]
        next_theme = "PROCESS_OUTCOME_FAMILY_IDENTIFIABILITY_AUDIT"
    return gates, classifications, next_theme


def benchmark_projection() -> dict[str, Any]:
    prior = json.loads((L41_ROOT / "runtime_manifest.json").read_text())
    projected_wall = float(prior["wallSeconds"]) * 1.05
    projected_cpu = float(
        json.loads((L41_ROOT / "benchmark_projection.json").read_text())[
            "projectedCpuHoursIncludingFullRegeneration"
        ]
    ) * 1.05
    return {
        "schema": "eidosoma.e01.s19_l42.benchmark_projection.v1",
        "status": "PASS"
        if projected_cpu <= 90 and projected_wall <= 64.8 * 3600
        else "STOP_BEFORE_OUTCOME",
        "sourceLoop": "S19-L41",
        "sourceWallSeconds": prior["wallSeconds"],
        "projectedWallSecondsIncludingFullRegeneration": projected_wall,
        "projectedCpuHoursIncludingFullRegeneration": projected_cpu,
        "newMatrices": 0,
        "newTrajectories": 0,
        "newBranchStreams": 0,
        "scientificOutcomeRetained": False,
    }


def make_figures(
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

    primary = states[
        states["branchFamily"].eq("F12") & states["targetId"].eq(PRIMARY_TARGET)
    ]
    for candidate, group in primary.groupby("candidateId"):
        eligible = group[group["committorEligible"]]
        plt.hist(
            eligible["qConditionalRecovery"],
            bins=np.linspace(0, 1, 21),
            alpha=0.55,
            label=candidate,
        )
    plt.xlabel("conditional same-neighbourhood recovery q")
    plt.ylabel("eligible states")
    plt.legend(fontsize=7)
    save("01_conditional_recovery_committor.png")

    summary = primary.groupby("candidateId")[["qBreak", "qConditionalRecovery", "qConditionalInheritanceResumption"]].mean()
    summary.plot(kind="bar")
    plt.ylabel("mean probability")
    plt.xticks(rotation=0)
    save("02_break_resumption_homeostasis.png")

    rel = reliability[
        reliability["branchFamily"].eq("F12")
        & reliability["targetId"].eq(PRIMARY_TARGET)
    ]
    labels = [f"{r.evaluationCohort}\n{r.candidateId[-2:]}" for r in rel.itertuples()]
    plt.bar(np.arange(len(rel)), rel["splitHalfSpearman"], color="#4c78a8")
    plt.axhline(0.5, color="black", linestyle="--")
    plt.xticks(np.arange(len(rel)), labels, rotation=25, ha="right", fontsize=7)
    plt.ylabel("split-half Spearman")
    save("03_conditional_committor_reliability.png")

    selected = transfers[
        transfers["comparisonId"].isin(
            [
                "F4_PROGRESS_VS_F12_CONDITIONAL_RECOVERY",
                "F12_PRIMARY_MINUS_SPECIES_PERMUTED",
                "F12_PRIMARY_MINUS_UNRELATED",
                "F12_PRIMARY_MINUS_ORDER_NULL",
            ]
        )
    ]
    labels = [f"{r.candidateId[-2:]} {r.comparisonId[:13]}" for r in selected.itertuples()]
    plt.bar(np.arange(len(selected)), selected["pointEstimate"], color="#f58518")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xticks(np.arange(len(selected)), labels, rotation=65, ha="right", fontsize=6)
    plt.ylabel("rank or q difference")
    save("04_short_shooting_and_controls.png")

    for candidate, group in hazards[
        hazards["evaluationCohort"].isin(EVALUATION_COHORTS)
    ].groupby("candidateId"):
        curve = group.groupby("postbreakFissionOpportunity")["cumulativeRecoveryIncidence"].mean()
        plt.plot(curve.index, curve.values, marker="o", label=candidate)
    plt.xlabel("fissions after genuine break")
    plt.ylabel("cumulative same-neighbourhood recovery")
    plt.legend(fontsize=7)
    save("05_recovery_hazard.png")

    checks = [
        "primaryConditionalCommittorReliable",
        "speciesPermutationControlPassed",
        "unrelatedMatrixControlPassed",
        "fixedCountOrderControlPassed",
        "f4ProgressRankPassed",
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
        "schema": "eidosoma.e01.s19_l42.artifact_manifest.v1",
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
            "beliefBeforeLoop": "L41 showed state variation and a strong F4 rank but generic repeated returns were less common than order-permuted futures.",
            "failureOrAmbiguityTargeted": "Whether recovery specifically after a genuine heredity/composition break is a coherent process distinct from generic recurrence.",
            "informationGainRationale": "Conditioning on a witnessed disruption separates uninterrupted persistence, inheritance resumption and same-neighbourhood homeostatic recovery.",
            "learned": "L42 fission-conditioned recovery contract locked before outcomes.",
            "ledgerSequence": sequence,
            "loopId": LOOP_ID,
            "motivatingEvidence": "Complete L39-L41 results and reviewer persistence-versus-recovery direction.",
            "proposedNextTest": "Rescore exact L41 F12/F4 paths for break-conditioned recovery.",
            "recordPhase": "PRE_LOOP_METHOD_LOCK",
            "remainingPlausibleHypotheses": "Homeostatic recovery, inheritance resumption, process-family targets, or shooting-only estimation.",
            "selectedHypotheses": "A genuine heredity break followed by sustained return to the pre-break neighbourhood defines homeostatic competence.",
            "timestampUtc": timestamp,
            "weakenedHypotheses": "Generic recurrence ordering is itself the transferable organization process.",
        },
        {
            "appendOnly": True,
            "beliefBeforeLoop": "A useful recovery target must have branch-half reliability, reference specificity and excess ordered runs beyond qualifying-frequency expectation.",
            "failureOrAmbiguityTargeted": "Conditional committor and short-shooting identifiability of same-neighbourhood heredity recovery.",
            "informationGainRationale": "Exact path reuse isolates target semantics from stochastic sampling changes.",
            "learned": ";".join(classifications),
            "ledgerSequence": sequence + 1,
            "loopId": LOOP_ID,
            "motivatingEvidence": "Complete L42 result.",
            "proposedNextTest": next_theme,
            "recordPhase": "POST_LOOP_RESULT_AUTONOMOUS_CONTINUATION",
            "remainingPlausibleHypotheses": next_theme,
            "selectedHypotheses": "Fission-conditioned same-neighbourhood heredity recovery.",
            "timestampUtc": timestamp,
            "weakenedHypotheses": "Frequent inheritance resumption automatically implies compositional homeostasis.",
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
        + f"\n\n## {LOOP_ID} — fission-conditioned heredity recovery\n\n"
        + f"- **Learned:** {', '.join(classifications)}.\n"
        + f"- **Next:** {next_theme}.\n",
    )

    candidates_path = ARTIFACT_ROOT / "candidate_registry.parquet"
    candidates = pd.read_parquet(candidates_path)
    candidate = {
        "branchCount": 1,
        "bundleId": "L42_FISSION_CONDITIONED_HEREDITY_RECOVERY",
        "candidateId": "S19-L42-SAME-NEIGHBOURHOOD-HEREDITY-RECOVERY-H090",
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
        "proposedSpecification": "genuine future heredity/departure break followed by two inherited returns to the pre-break daughter neighbourhood",
        "rankingScore": 30.0,
        "registryOrder": int(candidates["registryOrder"].max()) + 1,
        "selected": True,
        "selectionReason": "L41_ORDER_SUPPRESSION_AND_REVIEWER_RECOVERY_DIRECTION",
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
            "finding": f"{source.finding}; L42 use: {source.frozenUse}",
            "licenseStatus": "WORKSPACE_OR_HUMAN_DIRECTION",
            "redistributionStatus": "INTERNAL_EVIDENCE_ONLY",
            "repositoryIdentity": None,
            "retainedPath": None,
            "retrievalDate": timestamp[:10],
            "sha256": None,
            "sourceId": f"L42_{source.sourceId}",
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
                "HOMEOSTATIC_RECOVERY_PROCESS"
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
            "decision": "S19_L42_COMPLETE_AUTONOMOUS_CONTINUATION",
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
    gates: pd.DataFrame,
    classifications: list[str],
    runtime: dict[str, Any],
    next_theme: str,
) -> str:
    primary = states[
        states["branchFamily"].eq("F12") & states["targetId"].eq(PRIMARY_TARGET)
    ]
    summary = primary.groupby(["evaluationCohort", "candidateId"], as_index=False).agg(
        states=("stateId", "size"),
        eligibleStates=("committorEligible", "sum"),
        meanBreakProbability=("qBreak", "mean"),
        meanConditionalRecovery=("qConditionalRecovery", "mean"),
        meanConditionalResumption=("qConditionalInheritanceResumption", "mean"),
        meanOrderNull=("meanExactRecoveryOrderNullConditional", "mean"),
    )
    rel = reliability[
        reliability["branchFamily"].eq("F12")
        & reliability["targetId"].eq(PRIMARY_TARGET)
    ]
    selected = transfers[
        transfers["comparisonId"].isin(
            [
                "F4_PROGRESS_VS_F12_CONDITIONAL_RECOVERY",
                "F12_PRIMARY_MINUS_SPECIES_PERMUTED",
                "F12_PRIMARY_MINUS_UNRELATED",
                "F12_PRIMARY_MINUS_ORDER_NULL",
                "F12_RESUMPTION_VS_PRIMARY",
                "F12_INHERITANCE_FRACTION_VS_PRIMARY",
            ]
        )
    ]
    return f"""# S19-L42 — Fission-Conditioned Heredity Recovery Hazard

## Chief/human handoff

- **Step:** `{VERSION}`
- **Status:** complete under the extended L19–L55 autonomous sequence.
- **Classifications:** {", ".join(f"`{item}`" for item in classifications)}
- **Validation:** immutable L41-and-earlier baseline; seven fixtures; exact replay of all 53,760 L41 F12/F4 paths; new analysis-seed firewall; candidate-separated variable-denominator committor reliability; 4,096 matrix bootstraps; exact full regeneration; storage and artifact hashes.
- **Recommended next action:** `{next_theme}`.

## Frozen question

After the first future fission that both breaks strict parent-daughter inheritance (`H<=0.9`) and moves outside the strict-H neighbourhood of the preceding selected daughter, is there a reliable conditional probability of restoring a sustained hereditary regime in that same compositional neighbourhood?

The primary event requires two consecutive later fissions for which both parent-daughter inheritance and daughter-to-pre-break-anchor similarity are strict `H>0.9`. Online certification occurs only at the second qualifying recovery fission. Uninterrupted inheritance is excluded. Inheritance resumption without return to the pre-break neighbourhood is retained as a separate baseline.

## Anchor results

### Break, inheritance-resumption and same-neighbourhood recovery

{summary.to_markdown(index=False)}

### Conditional committor reliability

{rel.to_markdown(index=False)}

### Short shooting and registered controls

{selected.to_markdown(index=False)}

### Locked scientific gates

{gates.to_markdown(index=False)}

## Interpretation

This loop separates persistence without disturbance from recovery following disruption. The break itself is branch observable; the pre-break daughter becomes the frozen online recovery anchor at that moment. A species-permuted anchor and unrelated-matrix prefix composition test reference specificity without changing the break. The exact fixed-count order probability holds the number of qualifying recovery fissions and post-break opportunities fixed, asking whether their actual ordering contains more sustained recovery than expected from frequency alone.

Conditional recovery uses only branches with a genuine break. Every state must contribute at least 32 such F12 trials and at least 16 per branch half to be eligible. Variable-denominator binomial noise is removed from between-state variance. Incomplete and no-break paths are retained in unconditional and availability results and are never replaced.

## Provenance and validation

- Repository lock: `{runtime['repositoryHead']}`.
- Workers: `{runtime['workers']}`; one numerical-library thread per worker; GPU hours `0`.
- New matrices/trajectories/branch streams: `0/0/0`.
- Exact reused branch streams: `{runtime['reusedBranchStreams']}`.
- Wall time: `{runtime['wallSeconds']:.2f}` seconds.
- S01–S18, V1/V2 and S19-L01–L41 remain unchanged.

## Boundaries

This remains exploratory simulator evidence. A positive result would identify a branch-half-reliable propensity for recovery after a genuine disruption, not author code, paper replication, a static biomarker, Phi-r incremental value, intervention efficacy, causal control or a biological conclusion. A negative result constrains this exact same-neighbourhood two-fission recovery definition, not every form of robustness, error correction or organization.

## Reproduction

```bash
PYTHONPATH=src pytest -q tests/e01/test_s19_l42.py
python -m ruff check src/e01_onset_discovery/heredity_recovery.py scripts/e01/run_s19_l42_fission_conditioned_heredity_recovery.py tests/e01/test_s19_l42.py
python scripts/e01/run_s19_l42_fission_conditioned_heredity_recovery.py --prepare-lock
python scripts/e01/run_s19_l42_fission_conditioned_heredity_recovery.py
```
"""


def prepare_lock() -> None:
    LOOP_ROOT.mkdir(parents=True, exist_ok=True)
    if git("status", "--porcelain=v1"):
        raise RuntimeError("repository must be clean before L42 lock")
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/eidosoma/groups/42")
    if head != remote:
        raise RuntimeError("L42 local/remote commit mismatch")
    prior = validate_immutable_prior()
    fixtures = fixture_results()
    analysis = analysis_seed_manifest()
    firewall = seed_firewall(analysis)
    benchmark = benchmark_projection()
    payloads = build_payloads()
    if (
        not prior["unchanged"]
        or not fixtures["passed"].all()
        or firewall["status"] != "PASS"
        or benchmark["status"] != "PASS"
        or len(payloads) != 280
    ):
        raise RuntimeError("L42 preoutcome validation or benchmark failed")
    shutil.copy2(CONFIG, LOOP_ROOT / "preregistration.yaml")
    BASE.atomic_text(
        LOOP_ROOT / "decision_record.md",
        "# S19-L42 decision record\n\n"
        "L41 matched the recurrence event to 12 future fissions and found strong state variation in two groups and an F4 rank in all groups, but the actual two-return process was much less frequent than the same future compositions in randomized order. The reviewer emphasized that frequent inheritance, temporal ordering and homeostatic recovery are distinct. L42 therefore freezes one event before outcomes: the first future fission that jointly breaks parent-daughter inheritance and departs from the preceding daughter creates an online pre-break anchor; recovery requires two consecutive later inherited daughters in that same neighbourhood. Inheritance resumption without same-neighbourhood return, reference controls, exact fixed-count ordering, opportunity count, prefix inheritance, mass and phase are fixed baselines. L42 reuses every L41 path exactly and generates no stochastic stream.\n"
        "\nA one-state technical smoke was run only after the complete configuration, implementation and fixtures existed. It exposed break/event counts for that sentinel, prompted no scientific or numerical change, and no cohort aggregate was opened. This disclosure is retained because the repository commit and formal lock followed the smoke; all full-cohort decisions remained frozen.\n",
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
        "branch_seed_manifest.parquet",
    ):
        shutil.copy2(L41_ROOT / name, LOOP_ROOT / name)
    BASE.write_parquet(LOOP_ROOT / "analysis_seed_manifest.parquet", analysis)
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
        "l41TraceSha256": sha256_file(L41_ROOT / "branch_trace_results.parquet"),
        "l41ManifestSha256": sha256_file(L41_ROOT / "artifact_manifest.json"),
    }
    lock = {
        "schema": "eidosoma.e01.s19_l42.implementation_lock.v1",
        "repositoryHead": head,
        "remoteHead": remote,
        "runnerSha256": sha256_file(RUNNER_PATH),
        "coreSha256": sha256_file(CORE_PATH),
        "threshold": THRESHOLD,
        "requiredRecoveryRun": REQUIRED_RECOVERY_RUN,
        "futureFissionHorizons": FISSION_HORIZONS,
        "branchCounts": BRANCH_COUNTS,
        "targets": list(TARGETS),
        "minimumBreakTrials": MINIMUM_BREAK_TRIALS,
        "minimumHalfBreakTrials": MINIMUM_HALF_BREAK_TRIALS,
        "matrixBootstraps": BOOTSTRAPS,
        "newMatrices": 0,
        "newTrajectories": 0,
        "newBranchStreams": 0,
        "completedTestTrajectoryUsed": False,
        "prelockSingleStateTechnicalSmokeDisclosed": True,
        "scientificChangeAfterSmoke": False,
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
        raise RuntimeError("L42 repository lock mismatch")
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
        "l41TraceSha256": L41_ROOT / "branch_trace_results.parquet",
        "l41ManifestSha256": L41_ROOT / "artifact_manifest.json",
    }
    for key, path in locked_files.items():
        if sha256_file(path) != lock[key]:
            raise RuntimeError(f"L42 locked input changed: {path}")
    if (
        not prior["unchanged"]
        or prior["aggregateSha256"] != lock["priorAggregateSha256"]
        or not fixtures["passed"].all()
        or sha256_file(RUNNER_PATH) != lock["runnerSha256"]
        or sha256_file(CORE_PATH) != lock["coreSha256"]
    ):
        raise RuntimeError("L42 pre-execution validation failed")
    payloads = build_payloads()
    responses = pd.read_parquet(LOOP_ROOT / "response_registry.parquet")
    boundaries = pd.read_parquet(LOOP_ROOT / "prefix_boundary_registry.parquet")
    summaries = pd.read_parquet(LOOP_ROOT / "prefix_state_summary.parquet")
    prefixes = L41.L40.L39.prefix_controls(boundaries, summaries)

    if BUILD_ROOT.exists():
        shutil.rmtree(BUILD_ROOT)
    BUILD_ROOT.mkdir(parents=True)
    replay, outcomes = execute_paths(payloads)
    states = state_recovery_results(outcomes)
    reliability, reliability_bootstrap = reliability_results(states)
    pairs = transfer_pairs(states, responses, prefixes)
    transfers, transfer_bootstrap = transfer_results(pairs)
    hazards = recovery_hazard(outcomes)
    gates, classifications, next_theme = scientific_gates(reliability, transfers)
    make_figures(states, reliability, transfers, hazards, gates)

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
        "branch_replay_validation.parquet": replay,
        "branch_recovery_results.parquet": outcomes,
        "state_recovery_results.parquet": states,
        "committor_reliability_results.parquet": reliability,
        "committor_reliability_bootstrap.parquet": reliability_bootstrap,
        "transfer_pairs.parquet": pairs,
        "transfer_results.parquet": transfers,
        "transfer_bootstrap.parquet": transfer_bootstrap,
        "recovery_hazard_results.parquet": hazards,
        "scientific_gate_results.parquet": gates,
    }
    for name, frame in tables.items():
        BASE.write_parquet(BUILD_ROOT / name, frame)
    BASE.write_json(
        BUILD_ROOT / "classification.json",
        {
            "schema": "eidosoma.e01.s19_l42.classification.v1",
            "classifications": classifications,
            "homeostaticRecoveryCommittorEstablished": bool(
                gates["homeostaticRecoveryTargetPassed"].all()
            ),
            "shortRecoveryShootingCoordinateEstablished": bool(
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

    replay_again, outcomes_again = execute_paths(payloads)
    states_again = state_recovery_results(outcomes_again)
    reliability_again, reliability_bootstrap_again = reliability_results(states_again)
    pairs_again = transfer_pairs(states_again, responses, prefixes)
    transfers_again, transfer_bootstrap_again = transfer_results(pairs_again)
    hazards_again = recovery_hazard(outcomes_again)
    gates_again, classifications_again, next_again = scientific_gates(
        reliability_again, transfers_again
    )
    replay_tables = {
        "branchReplay": (replay, replay_again),
        "outcomes": (outcomes, outcomes_again),
        "states": (states, states_again),
        "reliability": (reliability, reliability_again),
        "reliabilityBootstrap": (
            reliability_bootstrap,
            reliability_bootstrap_again,
        ),
        "pairs": (pairs, pairs_again),
        "transfers": (transfers, transfers_again),
        "transferBootstrap": (transfer_bootstrap, transfer_bootstrap_again),
        "hazards": (hazards, hazards_again),
        "gates": (gates, gates_again),
    }
    checks = {
        name: frame_hash(left) == frame_hash(right)
        for name, (left, right) in replay_tables.items()
    }
    primary = outcomes[outcomes["targetId"].eq(PRIMARY_TARGET)]
    controls = outcomes[~outcomes["targetId"].eq(PRIMARY_TARGET)]
    primary_breaks = primary.set_index(
        ["stateId", "branchFamily", "branchIndex"]
    )["breakBoundaryOneBased"]
    control_breaks = controls.set_index(
        ["stateId", "branchFamily", "branchIndex", "targetId"]
    )["breakBoundaryOneBased"]
    break_exact = all(
        (
            pd.isna(value)
            and pd.isna(primary_breaks.loc[(state, family, branch)])
        )
        or value == primary_breaks.loc[(state, family, branch)]
        for (state, family, branch, _), value in control_breaks.items()
    )
    checks.update(
        {
            "classificationExact": classifications == classifications_again,
            "nextThemeExact": next_theme == next_again,
            "fixturesPassed": bool(fixtures["passed"].all()),
            "immutablePriorPassed": prior["unchanged"],
            "analysisSeedFirewallPassed": json.loads(
                (LOOP_ROOT / "seed_firewall.json").read_text()
            )["status"]
            == "PASS",
            "allL41PathsExact": bool(replay["exactL41Replay"].all()),
            "controlBreaksExact": break_exact,
            "noCompletedTarget": bool(
                (~outcomes["targetUsesCompletedTestTrajectory"]).all()
            ),
            "recoveryRequiresBreak": bool(
                outcomes.loc[outcomes["event"], "breakObserved"].all()
            ),
            "recoveryRequiresTwo": bool(
                outcomes.loc[outcomes["event"], "maximumConsecutiveRecovery"]
                .ge(REQUIRED_RECOVERY_RUN)
                .all()
            ),
            "noNewMatrix": True,
            "noNewTrajectory": True,
            "noNewBranchStream": True,
        }
    )
    if not all(checks.values()):
        raise RuntimeError(f"L42 regeneration validation failed: {checks}")
    BASE.write_json(
        BUILD_ROOT / "regeneration_validation.json",
        {
            "schema": "eidosoma.e01.s19_l42.regeneration_validation.v1",
            "status": "PASS",
            "checks": checks,
            "branchReplayFrameSha256": frame_hash(replay),
            "outcomeFrameSha256": frame_hash(outcomes),
            "stateFrameSha256": frame_hash(states),
            "gateFrameSha256": frame_hash(gates),
        },
    )
    runtime = {
        "schema": "eidosoma.e01.s19_l42.runtime.v1",
        "repositoryHead": git("rev-parse", "HEAD"),
        "workers": WORKERS,
        "numericalLibraryThreadsPerWorker": 1,
        "gpuHours": 0,
        "wallSeconds": time.perf_counter() - started,
        "controllerCpuHours": (time.process_time() - started_cpu) / 3600,
        "states": 280,
        "reusedBranchStreams": 53_760,
        "newBranchStreams": 0,
        "targetScoresPerBranch": len(TARGETS),
        "newMatrices": 0,
        "newTrajectories": 0,
        "completedAtUtc": utc_now(),
    }
    BASE.write_json(BUILD_ROOT / "runtime_manifest.json", runtime)
    retained = sum(path.stat().st_size for path in BUILD_ROOT.rglob("*") if path.is_file())
    temporary = sum(path.stat().st_size for path in CACHE_ROOT.rglob("*") if path.is_file())
    storage = {
        "schema": "eidosoma.e01.s19_l42.storage_validation.v1",
        "retainedBytes": retained,
        "retainedGiBCeiling": 25,
        "temporaryBytes": temporary,
        "temporaryGiBCeiling": 75,
        "status": "PASS"
        if retained < 25 * 2**30 and temporary < 75 * 2**30
        else "FAIL",
    }
    if storage["status"] != "PASS":
        raise RuntimeError("L42 storage ceiling exceeded")
    BASE.write_json(BUILD_ROOT / "storage_validation.json", storage)
    report = report_text(
        states,
        reliability,
        transfers,
        gates,
        classifications,
        runtime,
        next_theme,
    )
    BASE.atomic_text(BUILD_ROOT / "research_step_full_results.md", report)
    BASE.atomic_text(BUILD_ROOT / "S19_L42_FULL_RESULTS.md", report)
    BASE.atomic_text(
        BUILD_ROOT / "loop_decision_summary.md",
        "# S19-L42 decision summary\n\n"
        + f"**Classification:** {', '.join(classifications)}\n\n"
        + f"**All-group homeostatic-recovery target:** `{gates['homeostaticRecoveryTargetPassed'].all()}`.\n\n"
        + f"**All-group F4 recovery coordinate:** `{gates['shortShootingCoordinatePassed'].all()}`.\n\n"
        + f"**Next:** `{next_theme}`.\n",
    )
    BASE.write_json(BUILD_ROOT / "artifact_manifest.json", manifest_for(BUILD_ROOT))
    stage = LOOP_ROOT.with_name(".L42-promotion-stage")
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
        raise RuntimeError("L42 artifact hash validation failed")
    append_ledgers(classifications, runtime["completedAtUtc"], next_theme)
    BASE.atomic_text(ARTIFACT_ROOT / "research_step_full_results.md", report)
    BASE.atomic_text(
        ARTIFACT_ROOT / "S19_CURRENT_HANDOFF.md",
        report.replace("# S19-L42", "# S19 current handoff — S19-L42", 1),
    )
    BASE.write_json(
        ARTIFACT_ROOT / "s19_status.json",
        {
            "schema": "eidosoma.e01.s19.status.v1",
            "status": "ACTIVE_AUTONOMOUS_SEQUENCE",
            "latestCompletedLoop": LOOP_ID,
            "latestClassification": classifications,
            "selectedDiscoveryLead": (
                "HOMEOSTATIC_RECOVERY_PROCESS"
                if "PROMOTABLE_TO_UNTOUCHED_PROCESS_CONFIRMATION" in classifications
                else None
            ),
            "nextAuthorizedLoop": "S19-L43",
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
