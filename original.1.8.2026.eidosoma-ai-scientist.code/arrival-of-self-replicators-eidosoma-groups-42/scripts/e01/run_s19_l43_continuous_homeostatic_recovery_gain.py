#!/usr/bin/env python3
"""Run S19-L43 continuous homeostatic-recovery gain audit."""

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

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from e01_onset_discovery.heredity_recovery_gain import (
    score_heredity_recovery_gain,
)

ROOT = REPO_ROOT
ARTIFACT_ROOT = Path("/artifacts/research_steps/S19")
LOOP_ROOT = ARTIFACT_ROOT / "loops/L43"
L42_ROOT = ARTIFACT_ROOT / "loops/L42"
BUILD_ROOT = Path("/cache/e01_s19_l43")
CONFIG = ROOT / "configs/e01/s19_l43_continuous_homeostatic_recovery_gain.yaml"
RUNNER_PATH = Path(__file__).resolve()
CORE_PATH = ROOT / "src/e01_onset_discovery/heredity_recovery_gain.py"
LOOP_ID = "S19-L43"
VERSION = "E01-S19-L43-CONTINUOUS-HOMEOSTATIC-RECOVERY-GAIN-v1.0.0"
THRESHOLD = 0.9
REQUIRED_RUN = 2
FAMILIES = ("F12", "F4")
BRANCH_COUNTS = {"F12": 128, "F4": 64}
HALVES = {"F12": 64, "F4": 32}
MIN_TRIALS = {"F12": 32, "F4": 12}
MIN_HALF_TRIALS = {"F12": 16, "F4": 6}
BOOTSTRAPS = 4096
TARGETS = (
    "PRIMARY_PREBREAK_DAUGHTER",
    "SPECIES_PERMUTED_PREBREAK_DAUGHTER",
    "UNRELATED_MATRIX_PREFIX_DAUGHTER",
)
PRIMARY = TARGETS[0]
EVALUATION_COHORTS = ("L28_VALIDATION", "L31_CONFIRMATION")
SOURCE_URLS = (
    "https://pubmed.ncbi.nlm.nih.gov/16010993/",
    "https://pubmed.ncbi.nlm.nih.gov/30045888/",
    "https://pubmed.ncbi.nlm.nih.gov/11536890/",
)
SEED_ROOT = bytes.fromhex(
    "baaee2386d93dc3bce6cb99eb4acdb51eb7db8ca56e6048665649f73ba7a3a09"
)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


L42 = load_module(
    "e01_l42_runner",
    ROOT / "scripts/e01/run_s19_l42_fission_conditioned_heredity_recovery.py",
)
BASE = L42.BASE


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frame_hash(frame: pd.DataFrame) -> str:
    return hashlib.sha256(
        frame.sort_index(axis=1).sort_values(list(frame.columns), na_position="last")
        .reset_index(drop=True)
        .to_json(orient="records", double_precision=15)
        .encode()
    ).hexdigest()


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    if mask.sum() < 3 or len(np.unique(left[mask])) < 2 or len(np.unique(right[mask])) < 2:
        return float("nan")
    return float(spearmanr(left[mask], right[mask]).statistic)


def interval(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return float("nan"), float("nan")
    return tuple(map(float, np.quantile(finite, [0.025, 0.975])))


def seed_material(*parts: object) -> bytes:
    return hashlib.sha256(
        SEED_ROOT + b"\x00" + json.dumps(parts, separators=(",", ":")).encode()
    ).digest()


def derived_seed(*parts: object) -> int:
    return int.from_bytes(seed_material(*parts)[:16], "big")


def validate_immutable_prior() -> dict[str, Any]:
    prior = L42.validate_immutable_prior()
    manifest = json.loads((L42_ROOT / "artifact_manifest.json").read_text())
    rows = []
    for row in manifest["files"]:
        path = L42_ROOT / row["path"]
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
    aggregate = hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema": "eidosoma.e01.s19_l43.immutable_prior_validation.v1",
        "status": "PASS" if passed else "FAIL",
        "unchanged": passed,
        "priorThroughL41Unchanged": bool(prior["unchanged"]),
        "l42ArtifactCount": len(rows),
        "aggregateSha256": aggregate,
        "l42ManifestSha256": sha256_file(L42_ROOT / "artifact_manifest.json"),
        "rows": rows,
    }


def fixture_results() -> pd.DataFrame:
    def score(future: list[list[int]], inheritance: list[float], anchor=None):
        return score_heredity_recovery_gain(
            latest_prefix_daughter=np.asarray([10, 0]),
            future_daughters=np.asarray(future),
            parent_daughter_h=np.asarray(inheritance),
            future_generations=np.arange(1, len(future) + 1),
            future_offsets_one_based=np.arange(1, len(future) + 1),
            recovery_anchor_override=None if anchor is None else np.asarray(anchor),
        )

    primary = score([[0, 10], [5, 5], [8, 2]], [0.2, 0.95, 0.95])
    elsewhere = score(
        [[0, 10], [5, 5], [8, 2]], [0.2, 0.95, 0.95], anchor=[0, 10]
    )
    uninterrupted = score([[9, 1], [8, 2], [9, 1]], [0.95, 0.95, 0.95])
    no_resumption = score([[0, 10], [5, 5], [8, 2]], [0.2, 0.95, 0.2])
    strict = score([[0, 10], [5, 5], [8, 2]], [0.2, 0.9, 0.95])
    rows = [
        ("BREAK_THEN_POSITIVE_GAIN", primary.recovery_gain is not None and primary.recovery_gain > 0),
        ("UNINTERRUPTED_INHERITANCE_EXCLUDED", not uninterrupted.break_observed),
        ("ANCHOR_SPECIFIC_DIRECTION", elsewhere.recovery_gain is not None and elsewhere.recovery_gain < 0),
        ("ANCHOR_CONTROL_PRESERVES_CLOCK", primary.resumption_certification_boundary_one_based == elsewhere.resumption_certification_boundary_one_based),
        ("NO_RESUMPTION_UNDEFINED", no_resumption.recovery_gain is None),
        ("STRICT_H090", not strict.resumption_observed),
        ("SECOND_INHERITED_DIAGNOSTIC", primary.second_inherited_gain == primary.recovery_gain),
        ("EXACT_REPLAY", primary == score([[0, 10], [5, 5], [8, 2]], [0.2, 0.95, 0.95])),
    ]
    return pd.DataFrame(
        [{"fixtureId": name, "passed": bool(passed)} for name, passed in rows]
    )


def analysis_seed_manifest() -> pd.DataFrame:
    rows = []
    purposes = []
    for cohort in ("L28_DEVELOPMENT", *EVALUATION_COHORTS):
        for candidate in ("S12F-CANDIDATE-02", "S12F-CANDIDATE-03"):
            for family in FAMILIES:
                for target in TARGETS:
                    purposes.append(("reliability_bootstrap", cohort, candidate, family, target))
                for comparison in (
                    "PRIMARY_MINUS_SPECIES_PERMUTED",
                    "PRIMARY_MINUS_UNRELATED",
                    "F4_TO_F12_GAIN",
                    "ORDINARY_BASELINE_RANKS",
                ):
                    purposes.append(("matrix_bootstrap", cohort, candidate, family, comparison))
    for parts in purposes:
        material = seed_material(*parts)
        rows.append(
            {
                "purpose": parts[0],
                "evaluationCohort": parts[1],
                "candidateId": parts[2],
                "branchFamily": parts[3],
                "targetOrComparisonId": parts[4],
                "partsJson": json.dumps(parts),
                "rootHex": SEED_ROOT.hex(),
                "derivedSeed": str(int.from_bytes(material[:16], "big")),
                "seedMaterialSha256": material.hex(),
            }
        )
    frame = pd.DataFrame(rows).sort_values(list(pd.DataFrame(rows).columns)).reset_index(drop=True)
    if frame["derivedSeed"].duplicated().any() or frame["seedMaterialSha256"].duplicated().any():
        raise RuntimeError("L43 analysis seed collision")
    return frame


def seed_firewall(analysis: pd.DataFrame) -> dict[str, Any]:
    prior_material: set[str] = set()
    prior_derived: set[str] = set()
    for path in ARTIFACT_ROOT.glob("loops/L*/**/*seed*manifest*.parquet"):
        if "/L43/" in str(path):
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
    material = set(analysis["seedMaterialSha256"].astype(str))
    derived = set(analysis["derivedSeed"].astype(str))
    overlap_m = sorted(material & prior_material)
    overlap_d = sorted(derived & prior_derived)
    return {
        "schema": "eidosoma.e01.s19_l43.seed_firewall.v1",
        "status": "PASS" if not overlap_m and not overlap_d else "FAIL",
        "analysisSeedCount": len(analysis),
        "newBranchStreams": 0,
        "seedMaterialOverlapCount": len(overlap_m),
        "derivedSeedOverlapCount": len(overlap_d),
        "seedMaterialOverlaps": overlap_m,
        "derivedSeedOverlaps": overlap_d,
    }


def source_grounding_registry() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sourceId": "L43_REVIEWER_CONTINUOUS_PROCESS",
                "evidenceClass": "HUMAN_REVIEW_DIRECTION",
                "finding": "Frequency, temporal ordering, and recovery after disruption are distinct; continuous and hazard outcomes may be more informative than one binary label.",
                "frozenUse": "continuous gain at first two-fission inheritance resumption after a genuine break",
                "url": None,
            },
            {
                "sourceId": "L43_L42_RARE_BINARY_RECOVERY",
                "evidenceClass": "DIRECT_FROZEN_E01_RESULT",
                "finding": "Strict same-neighbourhood two-fission recovery occurred in only 0.24%-0.94% of break-conditioned branches while generic inheritance resumption occurred in about 88%-90%.",
                "frozenUse": "replace no threshold; retain H continuously at the same break and resumption times",
                "url": None,
            },
            {
                "sourceId": "L43_GARD_HOMEOSTATIC_GROWTH",
                "evidenceClass": "PRIMARY_LITERATURE",
                "finding": "GARD reproduction is described as fission followed by composition-governed homeostatic growth.",
                "frozenUse": "interpret return toward the pre-break composition as a bounded homeostatic-gain proxy",
                "url": SOURCE_URLS[0],
            },
            {
                "sourceId": "L43_GARD_COMPOSITIONAL_INHERITANCE",
                "evidenceClass": "PRIMARY_LITERATURE",
                "finding": "Composition-preserving growth and transfer to fission progeny define compositional reproduction in the GARD lineage.",
                "frozenUse": "retain parent-daughter inheritance resumption as the process clock",
                "url": SOURCE_URLS[1],
            },
        ]
    )


def build_payloads() -> list[dict[str, Any]]:
    payloads = L42.build_payloads()
    if len(payloads) != 280:
        raise RuntimeError("L43 payload scope failure")
    return payloads


def worker(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    outcomes: list[dict[str, Any]] = []
    replay: list[dict[str, Any]] = []
    prefix_latest = np.asarray(payload["prefixStates"][-1], dtype=np.int64)
    unrelated = np.asarray(payload["unrelatedPrefixStates"][-1], dtype=np.int64)
    permutation = np.asarray(payload["l42SpeciesPermutation"], dtype=np.int64)
    for family in FAMILIES:
        for branch in range(BRANCH_COUNTS[family]):
            trace = L42.L41._simulate(payload, family, branch)
            expected = payload["l42ExpectedBranches"][f"{family}:{branch}"]
            exact = bool(
                trace.path_sha256 == expected["pathSha256"]
                and trace.final_state_sha256 == expected["finalStateSha256"]
                and trace.fissions == expected["fissions"]
                and trace.selected_observations_generated == expected["selectedObservationsGenerated"]
                and trace.terminal_status == expected["terminalStatus"]
            )
            if not exact:
                raise RuntimeError(
                    f"L43 frozen L41 path replay failure: {payload['stateId']} {family} {branch}"
                )
            future = np.asarray(trace.future_states, dtype=np.int64).reshape((-1, 100))
            inheritance = np.asarray(trace.parent_daughter_h, dtype=np.float64)
            generations = np.asarray(trace.future_generations, dtype=np.int64)
            offsets = np.asarray(trace.future_offsets_one_based, dtype=np.int64)
            primary = score_heredity_recovery_gain(
                latest_prefix_daughter=prefix_latest,
                future_daughters=future,
                parent_daughter_h=inheritance,
                future_generations=generations,
                future_offsets_one_based=offsets,
                threshold=THRESHOLD,
                required_resumption_run=REQUIRED_RUN,
            )
            if primary.break_boundary_one_based in (None, 1):
                primary_anchor = prefix_latest
            else:
                primary_anchor = future[primary.break_boundary_one_based - 2]
            scores = {
                PRIMARY: primary,
                "SPECIES_PERMUTED_PREBREAK_DAUGHTER": score_heredity_recovery_gain(
                    latest_prefix_daughter=prefix_latest,
                    future_daughters=future,
                    parent_daughter_h=inheritance,
                    future_generations=generations,
                    future_offsets_one_based=offsets,
                    recovery_anchor_override=primary_anchor[permutation],
                    threshold=THRESHOLD,
                    required_resumption_run=REQUIRED_RUN,
                ),
                "UNRELATED_MATRIX_PREFIX_DAUGHTER": score_heredity_recovery_gain(
                    latest_prefix_daughter=prefix_latest,
                    future_daughters=future,
                    parent_daughter_h=inheritance,
                    future_generations=generations,
                    future_offsets_one_based=offsets,
                    recovery_anchor_override=unrelated,
                    threshold=THRESHOLD,
                    required_resumption_run=REQUIRED_RUN,
                ),
            }
            common = {
                "stateId": payload["stateId"],
                "evaluationCohort": payload["evaluationCohort"],
                "candidateId": payload["candidateId"],
                "matrixIndex": int(payload["matrixIndex"]),
                "landmark": int(payload["landmark"]),
                "branchFamily": family,
                "branchIndex": branch,
                "branchHalf": "A" if branch < HALVES[family] else "B",
            }
            replay.append(
                {
                    **common,
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
                        **common,
                        "targetId": target,
                        "breakObserved": scored.break_observed,
                        "breakBoundaryOneBased": scored.break_boundary_one_based,
                        "breakGeneration": scored.break_generation,
                        "breakOffsetOneBased": scored.break_offset_one_based,
                        "resumptionObserved": scored.resumption_observed,
                        "resumptionCertificationBoundaryOneBased": scored.resumption_certification_boundary_one_based,
                        "resumptionCertificationGeneration": scored.resumption_certification_generation,
                        "resumptionCertificationOffsetOneBased": scored.resumption_certification_offset_one_based,
                        "postbreakOpportunities": scored.postbreak_opportunities,
                        "inheritedPostbreakCount": scored.inherited_postbreak_count,
                        "maximumConsecutiveInheritedPostbreak": scored.maximum_consecutive_inherited_postbreak,
                        "breakAnchorH": scored.break_anchor_h,
                        "certificationAnchorH": scored.certification_anchor_h,
                        "recoveryGain": scored.recovery_gain,
                        "maximumPostbreakAnchorH": scored.maximum_postbreak_anchor_h,
                        "maximumInheritedPostbreakAnchorH": scored.maximum_inherited_postbreak_anchor_h,
                        "maximumRecoveryGain": scored.maximum_recovery_gain,
                        "breakToCertificationH": scored.break_to_certification_h,
                        "firstInheritedAnchorH": scored.first_inherited_anchor_h,
                        "secondInheritedAnchorH": scored.second_inherited_anchor_h,
                        "secondInheritedGain": scored.second_inherited_gain,
                        "resumptionLagFissions": scored.resumption_lag_fissions,
                        "futureInheritanceFraction": float(np.mean(inheritance > THRESHOLD)),
                        "inheritedFlags": json.dumps(scored.inherited_flags),
                        "pathSha256": trace.path_sha256,
                        "targetUsesCompletedTestTrajectory": False,
                    }
                )
    return {"replay": replay, "outcomes": outcomes}


def execute_paths(payloads: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    replay_rows: list[dict[str, Any]] = []
    outcome_rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(worker, payload) for payload in payloads]
        for future in as_completed(futures):
            result = future.result()
            replay_rows.extend(result["replay"])
            outcome_rows.extend(result["outcomes"])
    keys = ["candidateId", "matrixIndex", "landmark", "branchFamily", "branchIndex"]
    replay = pd.DataFrame(replay_rows).sort_values(keys).reset_index(drop=True)
    outcomes = (
        pd.DataFrame(outcome_rows)
        .sort_values([*keys, "targetId"])
        .reset_index(drop=True)
    )
    if len(replay) != 53_760 or len(outcomes) != 161_280 or not replay["exactL41Replay"].all():
        raise RuntimeError("L43 replay/output scope failure")
    return replay, outcomes


def state_gain_results(outcomes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = [
        "stateId",
        "evaluationCohort",
        "candidateId",
        "matrixIndex",
        "landmark",
        "branchFamily",
        "targetId",
    ]
    for keys, group in outcomes.groupby(group_columns, sort=False):
        valid = group[group["resumptionObserved"] & group["recoveryGain"].notna()]
        halves = valid.groupby("branchHalf")["recoveryGain"].agg(["count", "mean"])
        family = str(keys[5])
        trials_a = int(halves.loc["A", "count"]) if "A" in halves.index else 0
        trials_b = int(halves.loc["B", "count"]) if "B" in halves.index else 0
        eligible = bool(
            len(valid) >= MIN_TRIALS[family]
            and trials_a >= MIN_HALF_TRIALS[family]
            and trials_b >= MIN_HALF_TRIALS[family]
        )
        values = valid["recoveryGain"].to_numpy(dtype=np.float64)
        primary_group = group[group["targetId"] == PRIMARY]
        rows.append(
            {
                **dict(zip(group_columns, keys, strict=True)),
                "branches": len(group),
                "breakTrials": int(group["breakObserved"].sum()),
                "resumptionTrials": len(valid),
                "resumptionTrialsHalfA": trials_a,
                "resumptionTrialsHalfB": trials_b,
                "qBreak": float(group["breakObserved"].mean()),
                "qResumptionUnconditional": float(group["resumptionObserved"].mean()),
                "qResumptionGivenBreak": (
                    float(group.loc[group["breakObserved"], "resumptionObserved"].mean())
                    if group["breakObserved"].any()
                    else float("nan")
                ),
                "meanRecoveryGain": float(np.mean(values)) if len(values) else float("nan"),
                "recoveryGainVariance": float(np.var(values, ddof=1)) if len(values) > 1 else float("nan"),
                "meanRecoveryGainHalfA": float(halves.loc["A", "mean"]) if "A" in halves.index else float("nan"),
                "meanRecoveryGainHalfB": float(halves.loc["B", "mean"]) if "B" in halves.index else float("nan"),
                "meanBreakAnchorH": float(valid["breakAnchorH"].mean()) if len(valid) else float("nan"),
                "meanCertificationAnchorH": float(valid["certificationAnchorH"].mean()) if len(valid) else float("nan"),
                "meanMaximumRecoveryGain": float(valid["maximumRecoveryGain"].mean()) if len(valid) else float("nan"),
                "meanSecondInheritedGain": float(valid["secondInheritedGain"].mean()) if len(valid) else float("nan"),
                "meanResumptionLagFissions": float(valid["resumptionLagFissions"].mean()) if len(valid) else float("nan"),
                "meanPostbreakOpportunities": float(group.loc[group["breakObserved"], "postbreakOpportunities"].mean()) if group["breakObserved"].any() else float("nan"),
                "meanInheritedPostbreakCount": float(group.loc[group["breakObserved"], "inheritedPostbreakCount"].mean()) if group["breakObserved"].any() else float("nan"),
                "meanFutureInheritanceFraction": float(group["futureInheritanceFraction"].mean()),
                "committorEligible": eligible,
                "targetUsesCompletedTestTrajectory": bool(primary_group["targetUsesCompletedTestTrajectory"].any()) if len(primary_group) else False,
            }
        )
    result = pd.DataFrame(rows).sort_values(group_columns).reset_index(drop=True)
    if len(result) != 1_680:
        raise RuntimeError("L43 state result scope failure")
    return result


def corrected_continuous_variance(group: pd.DataFrame) -> dict[str, float]:
    means = group["meanRecoveryGain"].to_numpy(dtype=np.float64)
    within = group["recoveryGainVariance"].to_numpy(dtype=np.float64)
    trials = group["resumptionTrials"].to_numpy(dtype=np.float64)
    observed = float(np.var(means, ddof=1)) if len(means) > 1 else float("nan")
    sampling = float(np.nanmean(within / trials)) if len(means) else float("nan")
    return {
        "observedBetweenStateVariance": observed,
        "estimatedBranchSamplingVariance": sampling,
        "correctedBetweenStateVariance": observed - sampling,
    }


def matrix_resample(group: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    matrices = group["matrixIndex"].drop_duplicates().to_numpy()
    chosen = rng.choice(matrices, size=len(matrices), replace=True)
    parts = []
    for replicate, matrix in enumerate(chosen):
        part = group[group["matrixIndex"] == matrix].copy()
        part["bootstrapMatrix"] = replicate
        part["stateId"] = part["stateId"].astype(str) + f"::b{replicate}"
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def reliability_results(states: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    bootstrap_rows = []
    group_columns = ["evaluationCohort", "candidateId", "branchFamily", "targetId"]
    for keys, raw_group in states.groupby(group_columns, sort=False):
        group = raw_group[raw_group["committorEligible"]].copy()
        variance = corrected_continuous_variance(group)
        rho = safe_spearman(
            group["meanRecoveryGainHalfA"].to_numpy(float),
            group["meanRecoveryGainHalfB"].to_numpy(float),
        )
        parts = ("reliability_bootstrap", *keys)
        rng = np.random.default_rng(derived_seed(*parts))
        for replicate in range(BOOTSTRAPS):
            sampled = matrix_resample(group, rng)
            sampled_variance = corrected_continuous_variance(sampled)
            bootstrap_rows.append(
                {
                    **dict(zip(group_columns, keys, strict=True)),
                    "replicate": replicate,
                    "meanRecoveryGain": float(sampled["meanRecoveryGain"].mean()),
                    "correctedBetweenStateVariance": sampled_variance["correctedBetweenStateVariance"],
                    "splitHalfSpearman": safe_spearman(
                        sampled["meanRecoveryGainHalfA"].to_numpy(float),
                        sampled["meanRecoveryGainHalfB"].to_numpy(float),
                    ),
                }
            )
        boot = pd.DataFrame(bootstrap_rows[-BOOTSTRAPS:])
        mean_ci = interval(boot["meanRecoveryGain"].to_numpy(float))
        variance_ci = interval(boot["correctedBetweenStateVariance"].to_numpy(float))
        rho_ci = interval(boot["splitHalfSpearman"].to_numpy(float))
        rows.append(
            {
                **dict(zip(group_columns, keys, strict=True)),
                "states": len(raw_group),
                "eligibleStates": len(group),
                "meanRecoveryGain": float(group["meanRecoveryGain"].mean()),
                "meanRecoveryGainLower95": mean_ci[0],
                "meanRecoveryGainUpper95": mean_ci[1],
                **variance,
                "correctedVarianceLower95": variance_ci[0],
                "correctedVarianceUpper95": variance_ci[1],
                "splitHalfSpearman": rho,
                "splitHalfLower95": rho_ci[0],
                "splitHalfUpper95": rho_ci[1],
                "reliabilityGatePassed": bool(
                    len(group) >= 32
                    and mean_ci[0] > 0
                    and variance_ci[0] > 0
                    and rho > 0.5
                    and rho_ci[0] > 0.3
                ),
            }
        )
    return (
        pd.DataFrame(rows).sort_values(group_columns).reset_index(drop=True),
        pd.DataFrame(bootstrap_rows).sort_values([*group_columns, "replicate"]).reset_index(drop=True),
    )


def paired_control_results(states: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible = states[states["committorEligible"]].copy()
    wide = eligible.pivot_table(
        index=["stateId", "evaluationCohort", "candidateId", "matrixIndex", "landmark", "branchFamily"],
        columns="targetId",
        values="meanRecoveryGain",
    ).reset_index()
    rows = []
    bootstrap_rows = []
    for keys, group in wide.groupby(
        ["evaluationCohort", "candidateId", "branchFamily"], sort=False
    ):
        for control in TARGETS[1:]:
            usable = group.dropna(subset=[PRIMARY, control]).copy()
            usable["difference"] = usable[PRIMARY] - usable[control]
            comparison = (
                "PRIMARY_MINUS_SPECIES_PERMUTED"
                if control.startswith("SPECIES")
                else "PRIMARY_MINUS_UNRELATED"
            )
            rng = np.random.default_rng(derived_seed("matrix_bootstrap", *keys, comparison))
            values = []
            for replicate in range(BOOTSTRAPS):
                sampled = matrix_resample(usable, rng)
                value = float(sampled["difference"].mean())
                values.append(value)
                bootstrap_rows.append(
                    {
                        "evaluationCohort": keys[0],
                        "candidateId": keys[1],
                        "branchFamily": keys[2],
                        "comparisonId": comparison,
                        "replicate": replicate,
                        "difference": value,
                    }
                )
            ci = interval(np.asarray(values))
            rows.append(
                {
                    "evaluationCohort": keys[0],
                    "candidateId": keys[1],
                    "branchFamily": keys[2],
                    "comparisonId": comparison,
                    "definedPairs": len(usable),
                    "meanDifference": float(usable["difference"].mean()),
                    "lower95": ci[0],
                    "upper95": ci[1],
                    "gatePassed": bool(len(usable) >= 32 and ci[0] > 0),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(bootstrap_rows)


def transfer_results(
    states: pd.DataFrame, prefixes: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    primary = states[(states["targetId"] == PRIMARY) & states["committorEligible"]].copy()
    f12 = primary[primary["branchFamily"] == "F12"].drop(columns="branchFamily")
    f4 = primary[primary["branchFamily"] == "F4"].drop(columns="branchFamily")
    pairs = f12.merge(
        f4[
            [
                "stateId",
                "meanRecoveryGain",
                "meanFutureInheritanceFraction",
                "qResumptionGivenBreak",
            ]
        ].rename(
            columns={
                "meanRecoveryGain": "f4MeanRecoveryGain",
                "meanFutureInheritanceFraction": "f4FutureInheritanceFraction",
                "qResumptionGivenBreak": "f4QResumptionGivenBreak",
            }
        ),
        on="stateId",
        how="inner",
    ).rename(columns={"meanRecoveryGain": "f12MeanRecoveryGain"})
    prefix_columns = [
        column
        for column in (
            "stateId",
            "currentMass",
            "generationPhase",
            "prefixInheritanceFraction",
        )
        if column in prefixes.columns
    ]
    pairs = pairs.merge(prefixes[prefix_columns], on="stateId", how="left")
    comparisons = {
        "F4_GAIN_TO_F12_GAIN": "f4MeanRecoveryGain",
        "F12_INHERITANCE_FREQUENCY_TO_GAIN": "meanFutureInheritanceFraction",
        "F4_INHERITANCE_FREQUENCY_TO_F12_GAIN": "f4FutureInheritanceFraction",
        "F12_RESUMPTION_PROBABILITY_TO_GAIN": "qResumptionGivenBreak",
        "CURRENT_MASS_TO_GAIN": "currentMass",
        "GENERATION_PHASE_TO_GAIN": "generationPhase",
    }
    rows = []
    bootstrap_rows = []
    for keys, group in pairs.groupby(["evaluationCohort", "candidateId"], sort=False):
        for comparison, predictor in comparisons.items():
            if predictor not in group.columns:
                continue
            usable = group[np.isfinite(group[predictor]) & np.isfinite(group["f12MeanRecoveryGain"])].copy()
            point = safe_spearman(
                usable[predictor].to_numpy(float), usable["f12MeanRecoveryGain"].to_numpy(float)
            )
            rng = np.random.default_rng(
                derived_seed("matrix_bootstrap", keys[0], keys[1], "F4", comparison)
            )
            values = []
            for replicate in range(BOOTSTRAPS):
                sampled = matrix_resample(usable, rng)
                value = safe_spearman(
                    sampled[predictor].to_numpy(float),
                    sampled["f12MeanRecoveryGain"].to_numpy(float),
                )
                values.append(value)
                bootstrap_rows.append(
                    {
                        "evaluationCohort": keys[0],
                        "candidateId": keys[1],
                        "comparisonId": comparison,
                        "replicate": replicate,
                        "spearman": value,
                    }
                )
            ci = interval(np.asarray(values))
            rows.append(
                {
                    "evaluationCohort": keys[0],
                    "candidateId": keys[1],
                    "comparisonId": comparison,
                    "definedPairs": len(usable),
                    "spearman": point,
                    "lower95": ci[0],
                    "upper95": ci[1],
                    "gatePassed": bool(
                        comparison == "F4_GAIN_TO_F12_GAIN"
                        and len(usable) >= 32
                        and point > 0.5
                        and ci[0] > 0.3
                    ),
                }
            )
    return pairs, pd.DataFrame(rows), pd.DataFrame(bootstrap_rows)


def gain_hazard(outcomes: pd.DataFrame) -> pd.DataFrame:
    primary = outcomes[
        (outcomes["targetId"] == PRIMARY)
        & (outcomes["branchFamily"] == "F12")
        & outcomes["resumptionObserved"]
    ].copy()
    return (
        primary.groupby(
            ["evaluationCohort", "candidateId", "resumptionLagFissions"],
            dropna=False,
        )
        .agg(
            branches=("stateId", "size"),
            meanRecoveryGain=("recoveryGain", "mean"),
            medianRecoveryGain=("recoveryGain", "median"),
            positiveGainFraction=("recoveryGain", lambda values: float((values > 0).mean())),
            meanBreakAnchorH=("breakAnchorH", "mean"),
            meanCertificationAnchorH=("certificationAnchorH", "mean"),
        )
        .reset_index()
    )


def scientific_gates(
    reliability: pd.DataFrame,
    controls: pd.DataFrame,
    transfers: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], str]:
    rows = []
    for cohort in EVALUATION_COHORTS:
        for candidate in ("S12F-CANDIDATE-02", "S12F-CANDIDATE-03"):
            rel = reliability[
                (reliability["evaluationCohort"] == cohort)
                & (reliability["candidateId"] == candidate)
                & (reliability["branchFamily"] == "F12")
                & (reliability["targetId"] == PRIMARY)
            ].iloc[0]
            con = controls[
                (controls["evaluationCohort"] == cohort)
                & (controls["candidateId"] == candidate)
                & (controls["branchFamily"] == "F12")
            ]
            tr = transfers[
                (transfers["evaluationCohort"] == cohort)
                & (transfers["candidateId"] == candidate)
                & (transfers["comparisonId"] == "F4_GAIN_TO_F12_GAIN")
            ].iloc[0]
            species = bool(
                con.loc[
                    con["comparisonId"] == "PRIMARY_MINUS_SPECIES_PERMUTED",
                    "gatePassed",
                ].iloc[0]
            )
            unrelated = bool(
                con.loc[
                    con["comparisonId"] == "PRIMARY_MINUS_UNRELATED", "gatePassed"
                ].iloc[0]
            )
            target = bool(rel["reliabilityGatePassed"] and species and unrelated)
            short = bool(target and tr["gatePassed"])
            rows.append(
                {
                    "evaluationCohort": cohort,
                    "candidateId": candidate,
                    "continuousGainReliable": bool(rel["reliabilityGatePassed"]),
                    "speciesPermutationControlPassed": species,
                    "unrelatedMatrixControlPassed": unrelated,
                    "anchorSpecificContinuousTargetPassed": target,
                    "f4GainRankPassed": bool(tr["gatePassed"]),
                    "shortShootingCoordinatePassed": short,
                }
            )
    gates = pd.DataFrame(rows)
    target_all = bool(gates["anchorSpecificContinuousTargetPassed"].all())
    short_all = bool(gates["shortShootingCoordinatePassed"].all())
    anchor_all = bool(
        gates[["speciesPermutationControlPassed", "unrelatedMatrixControlPassed"]]
        .all(axis=None)
    )
    if target_all and short_all:
        classifications = [
            "STATE_DEPENDENT_CONTINUOUS_HOMEOSTATIC_RECOVERY_ESTABLISHED",
            "FISSION_CLOCK_RECOVERY_GAIN_COORDINATE_ESTABLISHED",
            "PROMOTABLE_TO_UNTOUCHED_PROCESS_CONFIRMATION",
        ]
        next_theme = "UNTOUCHED_CONTINUOUS_HOMEOSTATIC_RECOVERY_CONFIRMATION"
    elif target_all:
        classifications = [
            "CONTINUOUS_HOMEOSTATIC_RECOVERY_PROPENSITY_ESTABLISHED",
            "SHORT_SHOOTING_COORDINATE_NOT_ESTABLISHED",
            "NOT_PROMOTABLE_AS_CONFIRMED",
        ]
        next_theme = "CONTINUOUS_RECOVERY_SHOOTING_EFFICIENCY"
    elif anchor_all:
        classifications = [
            "ANCHOR_SPECIFIC_RECOVERY_TENDENCY_WITHOUT_RELIABLE_COMMITTOR",
            "NOT_PROMOTABLE_AS_CONFIRMED",
        ]
        next_theme = "HOMEOSTATIC_RECOVERY_HAZARD_SURVIVAL_AUDIT"
    else:
        classifications = [
            "CONTINUOUS_HOMEOSTATIC_RECOVERY_NOT_SUPPORTED",
            "PROCESS_OUTCOME_FAMILY_IDENTIFIABILITY_REQUIRED",
            "NOT_PROMOTABLE_AS_CONFIRMED",
        ]
        next_theme = "PROCESS_OUTCOME_FAMILY_IDENTIFIABILITY_AUDIT"
    return gates, classifications, next_theme


def benchmark_projection() -> dict[str, Any]:
    l42 = json.loads((L42_ROOT / "runtime_manifest.json").read_text())
    projected_wall = float(l42["wallSeconds"]) * 2.25
    projected_cpu = float(l42["controllerCpuHours"]) * 2.25 * 8
    return {
        "schema": "eidosoma.e01.s19_l43.benchmark_projection.v1",
        "basis": "L42 full exact path replay, doubled for independent full regeneration",
        "projectedWallSeconds": projected_wall,
        "projectedCpuHoursConservative": projected_cpu,
        "wallHoursCeiling": 72,
        "cpuHoursCeiling": 100,
        "status": "PASS"
        if projected_wall < 72 * 3600 and projected_cpu < 90
        else "FAIL",
    }


def make_figures(
    states: pd.DataFrame,
    reliability: pd.DataFrame,
    controls: pd.DataFrame,
    transfers: pd.DataFrame,
    hazards: pd.DataFrame,
    gates: pd.DataFrame,
) -> None:
    figure_root = BUILD_ROOT / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    primary = states[(states["targetId"] == PRIMARY) & (states["branchFamily"] == "F12")]

    fig, ax = plt.subplots(figsize=(9, 5))
    labels = []
    data = []
    for (cohort, candidate), group in primary.groupby(["evaluationCohort", "candidateId"]):
        labels.append(f"{cohort.replace('L28_', '').replace('L31_', '')}\n{candidate[-2:]}")
        data.append(group.loc[group["committorEligible"], "meanRecoveryGain"].dropna())
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.axhline(0, color="black", lw=1)
    ax.set_ylabel("Mean H gain at heredity resumption")
    ax.set_title("Continuous recovery gain by cohort and candidate")
    fig.tight_layout()
    fig.savefig(figure_root / "01_continuous_recovery_gain.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    rel = reliability[
        (reliability["targetId"] == PRIMARY) & (reliability["branchFamily"] == "F12")
    ]
    ax.scatter(rel["splitHalfSpearman"], rel["correctedBetweenStateVariance"])
    for row in rel.itertuples(index=False):
        ax.annotate(f"{row.evaluationCohort[-4:]}-{row.candidateId[-2:]}", (row.splitHalfSpearman, row.correctedBetweenStateVariance), fontsize=7)
    ax.axvline(0.5, color="grey", ls="--")
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("Split-half state-rank Spearman")
    ax.set_ylabel("Corrected between-state variance")
    ax.set_title("Continuous target reliability")
    fig.tight_layout()
    fig.savefig(figure_root / "02_gain_reliability.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    c12 = controls[controls["branchFamily"] == "F12"].copy()
    x = np.arange(len(c12))
    ax.bar(x, c12["meanDifference"])
    ax.vlines(x, c12["lower95"], c12["upper95"], color="black")
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x, [f"{r.evaluationCohort[-4:]}-{r.candidateId[-2:]}\n{r.comparisonId.split('_MINUS_')[-1][:5]}" for r in c12.itertuples(index=False)], rotation=35, ha="right")
    ax.set_ylabel("Primary minus control gain")
    ax.set_title("Anchor specificity with identical branch opportunities")
    fig.tight_layout()
    fig.savefig(figure_root / "03_anchor_controls.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    transfer = transfers[transfers["comparisonId"] == "F4_GAIN_TO_F12_GAIN"]
    x = np.arange(len(transfer))
    ax.bar(x, transfer["spearman"])
    ax.vlines(x, transfer["lower95"], transfer["upper95"], color="black")
    ax.axhline(0.5, color="grey", ls="--")
    ax.set_xticks(x, [f"{r.evaluationCohort[-4:]}-{r.candidateId[-2:]}" for r in transfer.itertuples(index=False)])
    ax.set_ylabel("F4-to-F12 Spearman")
    ax.set_title("Independent short shooting coordinate")
    fig.tight_layout()
    fig.savefig(figure_root / "04_short_shooting_transfer.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for (cohort, candidate), group in hazards.groupby(["evaluationCohort", "candidateId"]):
        ax.plot(group["resumptionLagFissions"], group["meanRecoveryGain"], marker="o", label=f"{cohort[-4:]}-{candidate[-2:]}")
    ax.axhline(0, color="black", lw=1)
    ax.set_xlabel("Fissions from break to resumption certification")
    ax.set_ylabel("Mean H gain")
    ax.set_title("Recovery gain versus resumption time")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figure_root / "05_gain_hazard_clock.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    checks = [
        "continuousGainReliable",
        "speciesPermutationControlPassed",
        "unrelatedMatrixControlPassed",
        "f4GainRankPassed",
    ]
    matrix = gates.set_index(["evaluationCohort", "candidateId"])[checks].astype(int)
    image = ax.imshow(matrix.to_numpy(), vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(checks)), checks, rotation=35, ha="right")
    ax.set_yticks(range(len(matrix)), [f"{a[-4:]}-{b[-2:]}" for a, b in matrix.index])
    ax.set_title("L43 scientific gate matrix")
    fig.colorbar(image, ax=ax, ticks=[0, 1])
    fig.tight_layout()
    fig.savefig(figure_root / "06_decision_matrix.png", dpi=160)
    plt.close(fig)


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
        "schema": "eidosoma.e01.s19_l43.artifact_manifest.v1",
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
            "beliefBeforeLoop": "L42 showed generic inheritance resumption after a genuine break was common, but strict same-neighbourhood sustained recovery was too rare for a reliable binary committor.",
            "failureOrAmbiguityTargeted": "Whether thresholding concealed a continuous, anchor-specific homeostatic recovery propensity at the first sustained inheritance resumption.",
            "informationGainRationale": "The continuous paired gain preserves the same break, resumption clock, fissions and inheritance opportunities while comparing actual, molecule-permuted and unrelated anchors.",
            "learned": "L43 continuous-gain contract locked before cohort outcomes.",
            "ledgerSequence": sequence,
            "loopId": LOOP_ID,
            "motivatingEvidence": "Complete L42 result, reviewer process/frequency/recovery distinction, and primary GARD homeostatic-growth descriptions.",
            "proposedNextTest": "Rescore exact L41 paths for continuous anchor gain at resumption.",
            "recordPhase": "PRE_LOOP_METHOD_LOCK",
            "remainingPlausibleHypotheses": "Continuous homeostatic gain, hazard-specific recovery, generic inheritance frequency, or shooting-only estimation.",
            "selectedHypotheses": "A reliable organization signal may exist as continuous return toward the pre-break state even when strict-H recovery is rare.",
            "timestampUtc": timestamp,
            "weakenedHypotheses": "A strict binary return threshold is required to detect homeostatic tendency.",
        },
        {
            "appendOnly": True,
            "beliefBeforeLoop": "A continuous signal must be state-reliable, positive, anchor-specific, and ranked by an independent short ensemble in both candidates and cohorts.",
            "failureOrAmbiguityTargeted": "Continuous recovery identifiability and short-shooting transfer.",
            "informationGainRationale": "Exact path reuse and within-branch anchor controls isolate target semantics from simulation variance and inheritance opportunity count.",
            "learned": ";".join(classifications),
            "ledgerSequence": sequence + 1,
            "loopId": LOOP_ID,
            "motivatingEvidence": "Complete L43 result.",
            "proposedNextTest": next_theme,
            "recordPhase": "POST_LOOP_RESULT_AUTONOMOUS_CONTINUATION",
            "remainingPlausibleHypotheses": next_theme,
            "selectedHypotheses": "Continuous gain at heredity resumption.",
            "timestampUtc": timestamp,
            "weakenedHypotheses": "Frequent inheritance resumption by itself establishes return to a prior hereditary regime.",
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
        + f"\n\n## {LOOP_ID} — continuous homeostatic recovery gain\n\n"
        + f"- **Learned:** {', '.join(classifications)}.\n"
        + f"- **Next:** {next_theme}.\n",
    )

    candidates_path = ARTIFACT_ROOT / "candidate_registry.parquet"
    candidates = pd.read_parquet(candidates_path)
    candidate = {
        "branchCount": 1,
        "bundleId": "L43_CONTINUOUS_HOMEOSTATIC_RECOVERY_GAIN",
        "candidateId": "S19-L43-CONTINUOUS-GAIN-AT-INHERITANCE-RESUMPTION",
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
        "proposedSpecification": "continuous H gain from genuine break to first two-fission inheritance resumption relative to online pre-break anchor",
        "rankingScore": 29.0,
        "registryOrder": int(candidates["registryOrder"].max()) + 1,
        "selected": True,
        "selectionReason": "L42_BINARY_RECOVERY_TOO_RARE_AND_REVIEWER_CONTINUOUS_PROCESS_DIRECTION",
        "sourceGrounding": 5,
        "testability": 5,
        "undefinedAuthorSemantics": 0,
    }
    BASE.write_parquet(
        candidates_path,
        pd.concat([candidates, pd.DataFrame([candidate]).reindex(columns=candidates.columns)], ignore_index=True),
    )

    source_path = ARTIFACT_ROOT / "source_search_ledger.parquet"
    sources = pd.read_parquet(source_path)
    source_rows = [
        {
            "commitOrVersion": None,
            "evidenceClass": row.evidenceClass,
            "finding": f"{row.finding}; L43 use: {row.frozenUse}",
            "licenseStatus": "PUBLIC_METADATA_OR_WORKSPACE_EVIDENCE",
            "redistributionStatus": "REFERENCE_ONLY",
            "repositoryIdentity": None,
            "retainedPath": None,
            "retrievalDate": timestamp[:10],
            "sha256": None,
            "sourceId": row.sourceId,
            "sourceType": row.evidenceClass,
            "treeIdentity": None,
            "url": row.url,
        }
        for row in source_grounding_registry().itertuples(index=False)
    ]
    BASE.write_parquet(
        source_path,
        pd.concat([sources, pd.DataFrame(source_rows).reindex(columns=sources.columns)], ignore_index=True),
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
            "selectedDiscoveryLead": "CONTINUOUS_HOMEOSTATIC_RECOVERY" if "PROMOTABLE_TO_UNTOUCHED_PROCESS_CONFIRMATION" in classifications else None,
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
            "decision": "S19_L43_COMPLETE_AUTONOMOUS_CONTINUATION",
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
    reliability: pd.DataFrame,
    controls: pd.DataFrame,
    transfers: pd.DataFrame,
    gates: pd.DataFrame,
    classifications: list[str],
    runtime: dict[str, Any],
    next_theme: str,
) -> str:
    primary = reliability[
        (reliability["targetId"] == PRIMARY)
        & (reliability["branchFamily"] == "F12")
    ][
        [
            "evaluationCohort",
            "candidateId",
            "eligibleStates",
            "meanRecoveryGain",
            "meanRecoveryGainLower95",
            "meanRecoveryGainUpper95",
            "correctedBetweenStateVariance",
            "correctedVarianceLower95",
            "splitHalfSpearman",
            "splitHalfLower95",
            "reliabilityGatePassed",
        ]
    ]
    controls12 = controls[controls["branchFamily"] == "F12"]
    transfer = transfers[transfers["comparisonId"] == "F4_GAIN_TO_F12_GAIN"]
    return f"""# S19-L43 — Continuous Homeostatic Recovery Gain

## Chief/human handoff

- **Step:** `{VERSION}`
- **Status:** complete under the extended L19–L55 autonomous sequence.
- **Classifications:** {", ".join(f"`{item}`" for item in classifications)}
- **Validation:** immutable L42-and-earlier baseline; eight fixtures; exact replay of all 53,760 L41 F12/F4 paths; candidate-separated continuous-outcome reliability; paired anchor controls; 4,096 catalytic-matrix bootstraps; exact full regeneration; storage and artifact hashes.
- **Recommended next action:** `{next_theme}`.

## Frozen question

After a future fission jointly breaks parent–daughter inheritance and departs from the preceding daughter, does the first subsequent run of two inherited fissions move the composition back toward the online pre-break daughter by a reliable amount? The primary continuous value is `H(certifying daughter, pre-break daughter) - H(break daughter, pre-break daughter)`. No new threshold, run length, horizon, trajectory, matrix or branch stream was searched or generated.

The same physical break and resumption certification are scored against the actual pre-break daughter, its frozen species permutation, and an unrelated-matrix prefix daughter. These paired controls therefore hold the number of fissions, inherited fissions, resumption order, opportunities, mass and phase fixed.

## Anchor results

### F12 primary continuous-gain reliability

{primary.to_markdown(index=False)}

### Paired anchor controls

{controls12.to_markdown(index=False)}

### Independent F4-to-F12 transfer

{transfer.to_markdown(index=False)}

### Scientific gates

{gates.to_markdown(index=False)}

## Interpretation

This analysis separates ordinary inheritance frequency from continuous restoration toward a specific pre-break composition. A positive paired anchor effect cannot be created by different branch opportunities or inherited-fission counts because every anchor is evaluated on the identical branch and at the identical certification boundary. State-dependent reliability and F4-to-F12 transfer are separately required before the result can be treated as a shooting coordinate.

The outcome remains a simulator proxy. It does not establish an author implementation, paper replication, a static biomarker, Phi-r incremental value, intervention efficacy, causal control or a biological conclusion.

## Provenance and validation

- Repository lock: `{runtime['repositoryHead']}`.
- Workers: `{runtime['workers']}`; one numerical-library thread per worker; GPU hours `0`.
- New matrices/trajectories/branch streams: `0/0/0`.
- Exact reused branch streams: `{runtime['reusedBranchStreams']}`.
- Wall time: `{runtime['wallSeconds']:.2f}` seconds.
- S01–S18, V1/V2 and S19-L01–L42 remain unchanged.

## Reproduction

```bash
PYTHONPATH=src pytest -q tests/e01/test_s19_l43.py
python -m ruff check src/e01_onset_discovery/heredity_recovery_gain.py scripts/e01/run_s19_l43_continuous_homeostatic_recovery_gain.py tests/e01/test_s19_l43.py
python scripts/e01/run_s19_l43_continuous_homeostatic_recovery_gain.py --prepare-lock
python scripts/e01/run_s19_l43_continuous_homeostatic_recovery_gain.py
```
"""


def prepare_lock() -> None:
    LOOP_ROOT.mkdir(parents=True, exist_ok=True)
    if git("status", "--porcelain=v1"):
        raise RuntimeError("repository must be clean before L43 lock")
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/eidosoma/groups/42")
    if head != remote:
        raise RuntimeError("L43 local/remote commit mismatch")
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
        raise RuntimeError("L43 preoutcome validation failed")
    shutil.copy2(CONFIG, LOOP_ROOT / "preregistration.yaml")
    BASE.atomic_text(
        LOOP_ROOT / "decision_record.md",
        "# S19-L43 decision record\n\n"
        "L42 retained the reviewer-requested distinction between common inheritance and recovery after genuine disruption. Strict-H sustained return to the exact pre-break neighbourhood was reference-specific but occurred in less than one percent of eligible branches, so it could not define a reliable binary committor. Before opening any L43 cohort result, L43 freezes one continuous alternative without changing the break, H threshold, two-fission resumption clock, horizons, candidates, states or branches: the change in H to the online pre-break daughter from the break state to the certifying resumption daughter. Molecule-permuted and unrelated anchors use the identical physical break and certification, exactly holding frequency and opportunities fixed.\n",
    )
    BASE.write_json(LOOP_ROOT / "immutable_prior_validation.json", prior)
    BASE.write_parquet(LOOP_ROOT / "fixture_results.parquet", fixtures)
    pd.DataFrame(
        [
            {
                "attemptId": "PRELOCK-001",
                "stage": "standalone_import_bootstrap",
                "status": "TECHNICAL_ONLY_RESOLVED_BEFORE_OUTCOME",
                "reason": "repository src path was not inserted for direct script execution",
                "scientificContractChanged": False,
                "scientificOutcomeAccessed": False,
            },
            {
                "attemptId": "PRELOCK-002",
                "stage": "analysis_seed_manifest_serialization",
                "status": "TECHNICAL_ONLY_RESOLVED_BEFORE_OUTCOME",
                "reason": "unsigned 128-bit seed identity exceeded Arrow signed integer inference; stored as canonical decimal string",
                "scientificContractChanged": False,
                "scientificOutcomeAccessed": False,
            },
        ]
    ).to_csv(LOOP_ROOT / "preoutcome_technical_attempts.csv", index=False)
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
        shutil.copy2(L42_ROOT / name, LOOP_ROOT / name)
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
        "l42ManifestSha256": sha256_file(L42_ROOT / "artifact_manifest.json"),
    }
    lock = {
        "schema": "eidosoma.e01.s19_l43.implementation_lock.v1",
        "repositoryHead": head,
        "remoteHead": remote,
        "runnerSha256": sha256_file(RUNNER_PATH),
        "coreSha256": sha256_file(CORE_PATH),
        "threshold": THRESHOLD,
        "requiredResumptionRun": REQUIRED_RUN,
        "futureFissionHorizons": {"F12": 12, "F4": 4},
        "branchCounts": BRANCH_COUNTS,
        "targets": list(TARGETS),
        "primaryOutcome": "certificationAnchorH-minus-breakAnchorH",
        "matrixBootstraps": BOOTSTRAPS,
        "newMatrices": 0,
        "newTrajectories": 0,
        "newBranchStreams": 0,
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
        raise RuntimeError("L43 repository lock mismatch")
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
        "l42ManifestSha256": L42_ROOT / "artifact_manifest.json",
    }
    for key, path in locked_files.items():
        if sha256_file(path) != lock[key]:
            raise RuntimeError(f"L43 locked input changed: {path}")
    if (
        not prior["unchanged"]
        or prior["aggregateSha256"] != lock["priorAggregateSha256"]
        or not fixtures["passed"].all()
        or sha256_file(RUNNER_PATH) != lock["runnerSha256"]
        or sha256_file(CORE_PATH) != lock["coreSha256"]
    ):
        raise RuntimeError("L43 pre-execution validation failed")

    payloads = build_payloads()
    prefixes = L42.L41.L40.L39.prefix_controls(
        pd.read_parquet(LOOP_ROOT / "prefix_boundary_registry.parquet"),
        pd.read_parquet(LOOP_ROOT / "prefix_state_summary.parquet"),
    )
    response_controls = pd.read_parquet(LOOP_ROOT / "response_registry.parquet")[
        ["stateId", "currentMass", "currentGenerationLocalStep"]
    ]
    prefixes = prefixes.merge(response_controls, on="stateId", how="left")
    prefixes["generationPhase"] = prefixes["currentGenerationLocalStep"]
    if BUILD_ROOT.exists():
        shutil.rmtree(BUILD_ROOT)
    BUILD_ROOT.mkdir(parents=True)
    replay, outcomes = execute_paths(payloads)
    states = state_gain_results(outcomes)
    reliability, reliability_bootstrap = reliability_results(states)
    controls, control_bootstrap = paired_control_results(states)
    pairs, transfers, transfer_bootstrap = transfer_results(states, prefixes)
    hazards = gain_hazard(outcomes)
    gates, classifications, next_theme = scientific_gates(
        reliability, controls, transfers
    )
    make_figures(states, reliability, controls, transfers, hazards, gates)

    tables = {
        "branch_replay_validation.parquet": replay,
        "branch_gain_results.parquet": outcomes,
        "state_gain_results.parquet": states,
        "continuous_gain_reliability_results.parquet": reliability,
        "continuous_gain_reliability_bootstrap.parquet": reliability_bootstrap,
        "paired_anchor_control_results.parquet": controls,
        "paired_anchor_control_bootstrap.parquet": control_bootstrap,
        "short_long_transfer_pairs.parquet": pairs,
        "short_long_transfer_results.parquet": transfers,
        "short_long_transfer_bootstrap.parquet": transfer_bootstrap,
        "gain_hazard_results.parquet": hazards,
        "scientific_gate_results.parquet": gates,
    }
    for name, frame in tables.items():
        BASE.write_parquet(BUILD_ROOT / name, frame)
    BASE.write_json(
        BUILD_ROOT / "classification.json",
        {
            "schema": "eidosoma.e01.s19_l43.classification.v1",
            "classifications": classifications,
            "continuousHomeostaticRecoveryEstablished": "STATE_DEPENDENT_CONTINUOUS_HOMEOSTATIC_RECOVERY_ESTABLISHED" in classifications,
            "shortRecoveryCoordinateEstablished": "FISSION_CLOCK_RECOVERY_GAIN_COORDINATE_ESTABLISHED" in classifications,
            "targetUsesCompletedTestTrajectory": False,
            "priorStatusesChanged": False,
        },
    )
    pd.DataFrame(
        columns=["failureId", "stage", "status", "reason", "scientificValuesReleased"]
    ).to_csv(BUILD_ROOT / "failure_ledger.csv", index=False)

    # Independent full regeneration from frozen states and streams.
    replay_again, outcomes_again = execute_paths(payloads)
    states_again = state_gain_results(outcomes_again)
    reliability_again, reliability_bootstrap_again = reliability_results(states_again)
    controls_again, control_bootstrap_again = paired_control_results(states_again)
    pairs_again, transfers_again, transfer_bootstrap_again = transfer_results(
        states_again, prefixes
    )
    hazards_again = gain_hazard(outcomes_again)
    gates_again, classifications_again, next_again = scientific_gates(
        reliability_again, controls_again, transfers_again
    )
    comparisons = {
        "replay": (replay, replay_again),
        "outcomes": (outcomes, outcomes_again),
        "states": (states, states_again),
        "reliability": (reliability, reliability_again),
        "reliabilityBootstrap": (reliability_bootstrap, reliability_bootstrap_again),
        "controls": (controls, controls_again),
        "controlBootstrap": (control_bootstrap, control_bootstrap_again),
        "pairs": (pairs, pairs_again),
        "transfers": (transfers, transfers_again),
        "transferBootstrap": (transfer_bootstrap, transfer_bootstrap_again),
        "hazards": (hazards, hazards_again),
        "gates": (gates, gates_again),
    }
    exact = {name: frame_hash(a) == frame_hash(b) for name, (a, b) in comparisons.items()}
    regeneration = {
        "schema": "eidosoma.e01.s19_l43.regeneration_validation.v1",
        "status": "PASS" if all(exact.values()) and classifications == classifications_again and next_theme == next_again else "FAIL",
        "tableExact": exact,
        "classificationExact": classifications == classifications_again,
        "nextThemeExact": next_theme == next_again,
        "branchReplayRows": len(replay_again),
        "scientificOutcomeRows": len(outcomes_again),
    }
    if regeneration["status"] != "PASS":
        raise RuntimeError("L43 exact regeneration failure")

    runtime = {
        "schema": "eidosoma.e01.s19_l43.runtime.v1",
        "repositoryHead": lock["head"],
        "workers": 8,
        "numericalLibraryThreadsPerWorker": 1,
        "gpuHours": 0,
        "wallSeconds": time.perf_counter() - started,
        "controllerCpuHours": (time.process_time() - started_cpu) / 3600,
        "states": 280,
        "reusedBranchStreams": len(replay),
        "newBranchStreams": 0,
        "targetScoresPerBranch": 3,
        "newMatrices": 0,
        "newTrajectories": 0,
        "completedAtUtc": utc_now(),
    }
    BASE.write_json(BUILD_ROOT / "runtime_manifest.json", runtime)
    BASE.write_json(BUILD_ROOT / "regeneration_validation.json", regeneration)
    retained_bytes = sum(path.stat().st_size for path in BUILD_ROOT.rglob("*") if path.is_file())
    storage = {
        "schema": "eidosoma.e01.s19_l43.storage_validation.v1",
        "status": "PASS" if retained_bytes <= 25 * 1024**3 else "FAIL",
        "retainedBytes": retained_bytes,
        "retainedGiBCeiling": 25,
        "temporaryBytes": retained_bytes,
        "temporaryGiBCeiling": 75,
    }
    BASE.write_json(BUILD_ROOT / "storage_validation.json", storage)
    report = report_text(
        reliability, controls, transfers, gates, classifications, runtime, next_theme
    )
    BASE.atomic_text(BUILD_ROOT / "S19_L43_FULL_RESULTS.md", report)
    BASE.atomic_text(BUILD_ROOT / "research_step_full_results.md", report)
    BASE.atomic_text(
        BUILD_ROOT / "loop_decision_summary.md",
        f"# S19-L43 decision summary\n\n**Classification:** {', '.join(classifications)}\n\n**Next:** `{next_theme}`.\n",
    )
    for path in (BUILD_ROOT / "figures").glob("*.png"):
        if not path.stat().st_size:
            raise RuntimeError(f"empty L43 figure: {path}")
    if storage["status"] != "PASS":
        raise RuntimeError("L43 storage ceiling exceeded")

    # Promote compact build atomically only after scientific and regeneration gates pass.
    for path in BUILD_ROOT.iterdir():
        destination = LOOP_ROOT / path.name
        if path.is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(path, destination)
        else:
            shutil.copy2(path, destination)
    BASE.write_json(LOOP_ROOT / "artifact_manifest.json", manifest_for(LOOP_ROOT))
    manifest_check = manifest_for(LOOP_ROOT)
    current_manifest = json.loads((LOOP_ROOT / "artifact_manifest.json").read_text())
    if manifest_check != current_manifest:
        raise RuntimeError("L43 artifact manifest regeneration failed")

    append_ledgers(classifications, runtime["completedAtUtc"], next_theme)
    root_report = (
        f"# S19 current-step report\n\nLatest completed loop: `{LOOP_ID}`.\n\n"
        f"Classification: {', '.join(classifications)}.\n\nNext autonomous theme: `{next_theme}`.\n"
    )
    BASE.atomic_text(ARTIFACT_ROOT / "S19_CURRENT_STEP_REPORT.md", root_report)
    BASE.atomic_text(ARTIFACT_ROOT / "CURRENT_STEP_HANDOFF.md", root_report)
    BASE.write_json(
        ARTIFACT_ROOT / "s19_status.json",
        {
            "schema": "eidosoma.e01.s19.status.v1",
            "programStatus": "ACTIVE_AUTONOMOUS_SEQUENCE",
            "latestCompletedLoop": LOOP_ID,
            "latestClassification": classifications,
            "selectedDiscoveryLead": "CONTINUOUS_HOMEOSTATIC_RECOVERY" if "PROMOTABLE_TO_UNTOUCHED_PROCESS_CONFIRMATION" in classifications else None,
            "nextAuthorizedLoop": "S19-L44",
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
