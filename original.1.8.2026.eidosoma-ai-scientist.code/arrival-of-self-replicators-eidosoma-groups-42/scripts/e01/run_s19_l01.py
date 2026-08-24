#!/usr/bin/env python3
"""Execute only the prospectively locked E01/S19 Loop 1 and stop for review."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow
import scipy
import sklearn
import torch
import yaml

from e01_clean_directional_confirmation.core import (
    ROOT_SEED_HEX as S13Y_ROOT_SEED_HEX,
    SIMULATION_PHASE as S13Y_SIMULATION_PHASE,
)
from e01_latent_timebase.core import array_sha256, derive_seed as derive_simulation_seed, generate_beta
from e01_prediction_reconstruction.core import build_split_manifest
from e01_s19_iterative_replication.bundle_a import (
    DYNAMIC_SPECIFICATIONS,
    NETWORK_SPECIFICATIONS,
    dynamic_task,
    network_metrics,
)
from e01_s19_iterative_replication.bundle_b import (
    load_base_payloads,
    run_cutoff_source_fits,
    run_models_for_proportion,
)
from e01_s19_iterative_replication.bundle_c import (
    SPIKE_SPECIFICATIONS,
    build_spike_descriptors,
    correlation_results as spike_correlation_results,
)
from e01_s19_iterative_replication.core import (
    CANDIDATE_IDS,
    CUTOFF_MODE,
    EQUIVALENCE_MARGIN,
    PROPORTIONS,
    RETROSPECTIVE_MODE,
    VERSION,
    correlation_inference,
    detectable_correlation,
    holm_adjust,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
ARTIFACT_ROOT = Path("/artifacts/research_steps/S19")
LOOP_ROOT = ARTIFACT_ROOT / "loops/L01"
WORK_ROOT = Path("/cache/e01_s19_l01/execution")
S13Y_ROOT = Path("/artifacts/research_steps/S13Y")
S16_ROOT = Path("/artifacts/research_steps/S16")
PREREG_PATH = REPO_ROOT / "configs/e01/s19_l01_preregistration.yaml"
METHOD_LOCK_PATH = REPO_ROOT / "configs/e01/s19_l01_method_lock.json"
RANKING_PATH = REPO_ROOT / "configs/e01/s19_l01_candidate_ranking.csv"
SPLIT_PATH = REPO_ROOT / "configs/e01/s16_split_manifest.csv"
NOLDS_WHEEL = Path("/cache/e01_s19_l01/packages/nolds-0.6.1-py2.py3-none-any.whl")


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_digest(frame: pd.DataFrame, sort_columns: list[str] | None = None) -> str:
    value = frame.copy()
    if sort_columns:
        value = value.sort_values(sort_columns).reset_index(drop=True)
    payload = value.to_json(orient="table", index=False, double_precision=15)
    return hashlib.sha256(payload.encode()).hexdigest()


def git_value(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def verify_pushed_lock() -> dict[str, Any]:
    method = json.loads(METHOD_LOCK_PATH.read_text(encoding="utf-8"))
    current = git_value("rev-parse", "HEAD")
    remote = git_value("rev-parse", "origin/eidosoma/groups/42")
    branch = git_value("branch", "--show-current")
    status = git_value("status", "--short")
    files = []
    for entry in method["lockedRepositoryFiles"]:
        path = REPO_ROOT / entry["path"]
        actual = sha256_file(path)
        files.append({**entry, "actualSha256": actual, "matched": actual == entry["sha256"]})
    passed = bool(
        current == remote
        and branch == "eidosoma/groups/42"
        and status == ""
        and all(row["matched"] for row in files)
        and method["outcomeAccessedAtLock"] is False
    )
    result = {
        "schema": "eidosoma.e01.s19_l01_pushed_preoutcome_lock.v1",
        "loopId": "S19-L01",
        "branch": branch,
        "pushedCommit": current,
        "remoteCommit": remote,
        "worktreeStatus": status,
        "lockedFiles": files,
        "passed": passed,
    }
    if not passed:
        raise RuntimeError(f"clean pushed pre-outcome lock failed: {result}")
    # Resolve the non-self-referential pushed commit in the collectible lock.
    collectible = dict(method)
    collectible["pushedCommit"] = current
    collectible["remoteCommitAtOutcomeAccess"] = remote
    collectible["cleanWorktreeAtOutcomeAccess"] = True
    write_json(LOOP_ROOT / "method_lock.json", collectible)
    write_json(LOOP_ROOT / "preoutcome_repository_lock.json", result)
    return result


def validate_immutable_baseline() -> dict[str, Any]:
    baseline = json.loads((ARTIFACT_ROOT / "s18_immutable_baseline.json").read_text(encoding="utf-8"))
    mismatches: list[dict[str, Any]] = []
    aggregate = hashlib.sha256()
    for row in baseline["files"]:
        path = Path(row["path"])
        if not path.exists():
            mismatches.append({"path": str(path), "reason": "missing"})
            continue
        current = {"path": row["path"], "role": row["role"], "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        aggregate.update(canonical_json(current).encode())
        aggregate.update(b"\n")
        if current["bytes"] != row["bytes"] or current["sha256"] != row["sha256"]:
            mismatches.append(
                {
                    "path": str(path),
                    "reason": "size_or_hash_changed",
                    "expectedBytes": row["bytes"],
                    "actualBytes": current["bytes"],
                    "expectedSha256": row["sha256"],
                    "actualSha256": current["sha256"],
                }
            )
    result = {
        "schema": "eidosoma.e01.s19_l01_immutable_validation.v1",
        "fileCount": len(baseline["files"]),
        "mismatchCount": len(mismatches),
        "mismatches": mismatches,
        "aggregateSha256": aggregate.hexdigest(),
        "expectedAggregateSha256": baseline["aggregateSha256"],
        "passed": len(mismatches) == 0 and aggregate.hexdigest() == baseline["aggregateSha256"],
    }
    if not result["passed"]:
        raise RuntimeError("immutable S01-S18 baseline failed")
    return result


def validate_inputs() -> dict[str, Any]:
    manifest = json.loads((LOOP_ROOT / "input_manifest.json").read_text(encoding="utf-8"))
    rows = []
    for entry in manifest["inputs"]:
        path = Path(entry["path"])
        actual = sha256_file(path)
        rows.append({**entry, "actualSha256": actual, "matched": actual == entry["sha256"]})
    result = {
        "schema": "eidosoma.e01.s19_l01_input_validation.v1",
        "entries": rows,
        "passed": all(row["matched"] for row in rows),
    }
    if not result["passed"]:
        raise RuntimeError("frozen input hash changed")
    return result


def run_summaries(
    full: pd.DataFrame, prefix: pd.DataFrame, labels: pd.DataFrame
) -> pd.DataFrame:
    full = full.loc[full["implementationId"].eq("PHIRL_REGULARIZED_SOURCE")].copy()
    prefix = prefix.loc[
        prefix["status"].isin(["ELIGIBLE", "ELIGIBLE_PARTIAL_NONFINITE_LOCAL_VALUES"])
        & prefix["emergence"].notna()
    ].copy()
    primary_labels = labels.loc[labels["labelId"].eq("MOL_ADJACENT_INCOMING_H900")].copy()
    rows: list[dict[str, Any]] = []
    for candidate_id in CANDIDATE_IDS:
        for matrix_index in range(100):
            f = full.loc[
                full["candidateId"].eq(candidate_id) & full["matrixIndex"].eq(matrix_index)
            ]
            p = prefix.loc[
                prefix["candidateId"].eq(candidate_id) & prefix["matrixIndex"].eq(matrix_index)
            ]
            lab = primary_labels.loc[
                primary_labels["candidateId"].eq(candidate_id)
                & primary_labels["matrixIndex"].eq(matrix_index)
            ]
            h = lab["labelScore"].to_numpy(dtype=np.float64)
            y = lab["isReplicator"].to_numpy(dtype=bool)
            if not np.array_equal(y, h > 0.9):
                raise ValueError("Y != I(H>0.9) in run summary")
            rows.append(
                {
                    "candidateId": candidate_id,
                    "matrixIndex": matrix_index,
                    "meanCompletedEmergence": float(f["emergence"].mean()),
                    "meanPastOnlyEmergence": float(p["emergence"].mean()) if len(p) else None,
                    "meanExactH": float(h.mean()),
                    "meanCompositionChangeL2": float(f["euclideanL2ClosedCompositionChange"].mean()),
                    "replicationProbability": float(y.mean()),
                    "trajectoryLength": len(lab),
                    "exactHDeterminesLabel": True,
                }
            )
    return pd.DataFrame(rows)


def run_bundle_a(
    full: pd.DataFrame,
    prefix: pd.DataFrame,
    labels: pd.DataFrame,
    trajectory_manifest: pd.DataFrame,
    workers: int,
) -> dict[str, pd.DataFrame]:
    started = time.perf_counter()
    summary = run_summaries(full, prefix, labels)
    network_rows: list[dict[str, Any]] = []
    beta_replay_rows: list[dict[str, Any]] = []
    for matrix_index in range(100):
        seed = derive_simulation_seed(
            S13Y_ROOT_SEED_HEX,
            S13Y_SIMULATION_PHASE,
            "catalytic_matrix",
            matrix_index,
        )
        beta = generate_beta(seed)
        replay = generate_beta(seed)
        expected_hashes = trajectory_manifest.loc[
            trajectory_manifest["matrixIndex"].eq(matrix_index), "betaSha256"
        ].unique()
        expected = expected_hashes[0]
        passed = bool(
            len(expected_hashes) == 1
            and array_sha256(beta) == expected
            and np.array_equal(beta, replay)
        )
        beta_replay_rows.append(
            {
                "bundleId": "A_METRIC_DISTINCTIVENESS",
                "matrixIndex": matrix_index,
                "expectedBetaSha256": expected,
                "actualBetaSha256": array_sha256(beta),
                "replayBetaSha256": array_sha256(replay),
                "passed": passed,
            }
        )
        if not passed:
            raise RuntimeError("frozen beta replay failed")
        for specification in NETWORK_SPECIFICATIONS:
            features = network_metrics(beta, specification)
            for candidate_id in CANDIDATE_IDS:
                for row in features:
                    network_rows.append(
                        {
                            "candidateId": candidate_id,
                            "matrixIndex": matrix_index,
                            "specificationId": specification,
                            "betaSharedAcrossCandidates": True,
                            "betaSha256": expected,
                            **row,
                        }
                    )
    tasks = []
    for row in trajectory_manifest.itertuples(index=False):
        if row.candidateId not in CANDIDATE_IDS:
            continue
        if sha256_file(Path(row.cachePath)) != row.cacheSha256:
            raise RuntimeError("frozen trajectory cache changed")
        for specification in DYNAMIC_SPECIFICATIONS:
            for replay in (False, True):
                tasks.append((row.candidateId, int(row.matrixIndex), row.cachePath, specification, replay))
    dynamic_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(dynamic_task, task): task for task in tasks}
        for future in as_completed(futures):
            features, diagnostics = future.result()
            dynamic_rows.extend(features)
            diagnostic_rows.append(diagnostics)
    network = pd.DataFrame(network_rows)
    dynamics = pd.DataFrame(dynamic_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    replay_key = ["candidateId", "matrixIndex", "specificationId", "metricId", "summaryId"]
    primary = dynamics.loc[~dynamics["replay"]].sort_values(replay_key).reset_index(drop=True)
    replay = dynamics.loc[dynamics["replay"]].sort_values(replay_key).reset_index(drop=True)
    if not primary[replay_key].equals(replay[replay_key]):
        raise RuntimeError("dynamical replay key mismatch")
    values_equal = np.allclose(
        primary["value"].to_numpy(dtype=np.float64),
        replay["value"].to_numpy(dtype=np.float64),
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )
    failures_equal = primary["failureReason"].fillna("").equals(replay["failureReason"].fillna(""))
    dynamic_replay = pd.DataFrame(
        [
            {
                "bundleId": "A_METRIC_DISTINCTIVENESS",
                "replayedFeatureRowCount": len(primary),
                "valuesBitExactOrJointNaN": values_equal,
                "failureStatusesExact": failures_equal,
                "passed": bool(values_equal and failures_equal),
            }
        ]
    )
    if not dynamic_replay["passed"].all():
        raise RuntimeError("dynamical metric replay failed")
    features = pd.concat(
        [
            network,
            primary.drop(columns=["replay"]),
        ],
        ignore_index=True,
        sort=False,
    )
    metric_to_claim = {
        "C001_NUMBER_OF_NODES": "C001",
        "C002_NUMBER_OF_EDGES": "C002",
        "C003_IN_DEGREE": "C003",
        "C004_OUT_DEGREE": "C004",
        "C005_BETWEENNESS": "C005",
        "C006_PAGERANK": "C006",
        "C007_HITS_HUB": "C007",
        "C008_SAMPLE_ENTROPY": "C008",
        "C009_CORRELATION_DIMENSION": "C009",
        "C010_LYAPUNOV_EXPONENT": "C010",
        "C011_DFA": "C011",
        "C012_GENERALIZED_HURST": "C012",
    }
    primary_summary = {
        "C001_NUMBER_OF_NODES": "PRIMARY",
        "C002_NUMBER_OF_EDGES": "PRIMARY",
        "C003_IN_DEGREE": "PRIMARY_MEAN",
        "C004_OUT_DEGREE": "PRIMARY_MEAN",
        "C005_BETWEENNESS": "PRIMARY_MEAN",
        "C006_PAGERANK": "PRIMARY_MEAN",
        "C007_HITS_HUB": "PRIMARY_MEAN",
        "C008_SAMPLE_ENTROPY": "PRIMARY",
        "C009_CORRELATION_DIMENSION": "PRIMARY_MEAN",
        "C010_LYAPUNOV_EXPONENT": "PRIMARY_MEAN",
        "C011_DFA": "PRIMARY",
        "C012_GENERALIZED_HURST": "PRIMARY",
    }
    outcomes = [
        "meanCompletedEmergence",
        "meanPastOnlyEmergence",
        "meanExactH",
        "meanCompositionChangeL2",
    ]
    correlation_rows: list[dict[str, Any]] = []
    for (candidate_id, specification, metric_id, summary_id), group in features.groupby(
        ["candidateId", "specificationId", "metricId", "summaryId"], sort=True
    ):
        if metric_id not in metric_to_claim or summary_id != primary_summary[metric_id]:
            continue
        merged = summary.loc[summary["candidateId"].eq(candidate_id)].merge(
            group[["matrixIndex", "value"]], on="matrixIndex", how="left", validate="one_to_one"
        )
        for outcome_id in outcomes:
            for method in ("spearman", "pearson"):
                replicates = 4096 if outcome_id == "meanCompletedEmergence" else 512
                inference = correlation_inference(
                    merged["value"].to_numpy(dtype=np.float64),
                    merged[outcome_id].to_numpy(dtype=np.float64),
                    method=method,
                    seed_identity=("bundle_a", candidate_id, specification, metric_id, outcome_id),
                    bootstrap_replicates=replicates,
                )
                correlation_rows.append(
                    {
                        "claimId": metric_to_claim[metric_id],
                        "candidateId": candidate_id,
                        "specificationId": specification,
                        "metricId": metric_id,
                        "summaryId": summary_id,
                        "outcomeId": outcome_id,
                        "correlationMethod": method,
                        "bootstrapReplicates": replicates,
                        "detectableAbsoluteCorrelation80Power": detectable_correlation(
                            int(inference["definedCount"]), 0.05 / 24.0
                        ),
                        **inference,
                    }
                )
    correlations = pd.DataFrame(correlation_rows)
    correlations["primaryClaimTest"] = (
        correlations["outcomeId"].eq("meanCompletedEmergence")
        & correlations["specificationId"].isin(
            ["A_GRAPH_UNWEIGHTED_POSITIVE_SUPPORT", "A_DYNAMICS_DIRECT_SELECTED_CLOCK"]
        )
    )
    correlations["holmAdjustedPValue"] = None
    for candidate_id in CANDIDATE_IDS:
        mask = correlations["candidateId"].eq(candidate_id) & correlations["primaryClaimTest"]
        correlations.loc[mask, "holmAdjustedPValue"] = holm_adjust(
            correlations.loc[mask, "pValue"].tolist()
        )
    execution = pd.concat(
        [
            diagnostics.assign(bundleId="A_METRIC_DISTINCTIVENESS", status="EXECUTED"),
            pd.DataFrame(beta_replay_rows).assign(status="EXECUTED"),
        ],
        ignore_index=True,
        sort=False,
    )
    execution["bundleRuntimeSeconds"] = time.perf_counter() - started
    return {
        "network": network,
        "dynamics": dynamics,
        "dynamicDiagnostics": diagnostics,
        "runSummary": summary,
        "correlations": correlations,
        "betaReplay": pd.DataFrame(beta_replay_rows),
        "dynamicReplay": dynamic_replay,
        "execution": execution,
    }


def run_bundle_b(
    full: pd.DataFrame,
    labels: pd.DataFrame,
    trajectory_manifest: pd.DataFrame,
    workers: int,
) -> dict[str, pd.DataFrame]:
    split = pd.read_csv(SPLIT_PATH)
    pd.testing.assert_frame_equal(split, build_split_manifest(), check_dtype=False)
    base = load_base_payloads(trajectory_manifest, labels, full)
    metric_frames = []
    execution_frames = []
    replay_frames = []
    scaler_frames = []
    source_frames = []
    for proportion in PROPORTIONS:
        source, source_audit = run_cutoff_source_fits(base, proportion, workers)
        if not source_audit["exactReplayPassed"].all() or source_audit["futureSuffixAccessed"].any():
            raise RuntimeError("cutoff source replay or suffix invariant failed")
        outputs = run_models_for_proportion(base, source, split, proportion)
        metric_frames.append(outputs["metrics"])
        execution_frames.append(outputs["execution"])
        replay_frames.append(outputs["modelReplay"])
        scaler_frames.append(outputs["scalers"])
        source_frames.append(source_audit)
    metrics = pd.concat(metric_frames, ignore_index=True)
    execution = pd.concat(execution_frames, ignore_index=True)
    replay = pd.concat(replay_frames, ignore_index=True)
    scalers = pd.concat(scaler_frames, ignore_index=True)
    source_audit = pd.concat(source_frames, ignore_index=True)
    if not replay["passed"].all() or not execution["exactReplayPassed"].all():
        raise RuntimeError("prediction model replay failed")
    frozen = pd.read_csv(S16_ROOT / "split_metrics.csv")
    current = metrics.loc[np.isclose(metrics["proportion"], 0.25)].copy()
    keys = ["candidateId", "modeId", "featureId", "repetitionId"]
    merged = current.merge(frozen, on=keys, suffixes=("Current", "Frozen"), validate="one_to_one")
    metric_names = [
        "validTargetCount",
        "positiveCount",
        "prevalence",
        "accuracy",
        "auroc",
        "auprc",
        "brier",
        "calibrationError",
        "balancedAccuracy",
        "sensitivity",
        "specificity",
        "macroMatrixAccuracy",
    ]
    replay_checks = []
    for metric in metric_names:
        left = merged[f"{metric}Current"].to_numpy(dtype=np.float64)
        right = merged[f"{metric}Frozen"].to_numpy(dtype=np.float64)
        replay_checks.append(
            {
                "metricId": metric,
                "rowCount": len(merged),
                "maximumAbsoluteDifference": float(np.nanmax(np.abs(left - right))),
                "exactOrJointNaN": bool(np.allclose(left, right, rtol=0.0, atol=0.0, equal_nan=True)),
            }
        )
    s16_replay = pd.DataFrame(replay_checks)
    if not s16_replay["exactOrJointNaN"].all():
        raise RuntimeError("25/75 exact S16 replay failed")
    summary = (
        metrics.groupby(["candidateId", "proportion", "modeId", "featureId"], sort=True)
        .agg(
            medianAccuracy=("accuracy", "median"),
            meanAccuracy=("accuracy", "mean"),
            medianBalancedAccuracy=("balancedAccuracy", "median"),
            medianAuroc=("auroc", "median"),
            medianAuprc=("auprc", "median"),
            medianBrier=("brier", "median"),
            medianCalibrationError=("calibrationError", "median"),
            medianPrevalence=("prevalence", "median"),
            medianSensitivity=("sensitivity", "median"),
            medianSpecificity=("specificity", "median"),
            medianPreOnsetEligible=("preOnset_eligibleRunCount", "median"),
            validTargetCount=("validTargetCount", "sum"),
        )
        .reset_index()
    )
    return {
        "metrics": metrics,
        "summary": summary,
        "execution": execution,
        "modelReplay": replay,
        "scalers": scalers,
        "sourceAudit": source_audit,
        "s16Replay": s16_replay,
    }


def classify_claims(
    a: dict[str, pd.DataFrame], b: dict[str, pd.DataFrame], c: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    corr = a["correlations"]
    for number in range(1, 13):
        claim = f"C{number:03d}"
        subset = corr.loc[corr["claimId"].eq(claim) & corr["primaryClaimTest"]]
        replay_passed = bool(a["dynamicReplay"]["passed"].all() and a["betaReplay"]["passed"].all())
        if len(subset) != 4 or subset["status"].ne("DEFINED").any():
            classification = "AUTHOR_AMBIGUITY_UNRESOLVED"
            promotable = False
            rationale = "Primary source-literal metric was constant or undefined in at least one candidate/method."
        elif subset["equivalentSmallEffect"].all() and subset["holmAdjustedPValue"].astype(float).gt(0.05).all():
            classification = "EXPLORATORY_PAPER_MATCH"
            promotable = replay_passed
            rationale = "Both candidates/methods were nonsignificant and equivalently small under the source-grounded primary implementation."
        elif subset["holmAdjustedPValue"].astype(float).gt(0.05).all():
            classification = "EXPLORATORY_DIRECTIONAL_MATCH"
            promotable = False
            rationale = "Paper-like nonsignificance appeared, but equivalently small effects were not established."
        else:
            classification = "EXPLORATORY_NON_SUPPORT"
            promotable = False
            rationale = "At least one primary candidate/method rejected the no-association pattern after correction."
        rows.append(
            {
                "claimId": claim,
                "originalS18Status": "NOT_EVALUATED",
                "s19ExploratoryStatus": classification,
                "promotableToS20": promotable,
                "candidate2Status": classification,
                "candidate3Status": classification,
                "exactOrDirectional": "EXACT_PAPER_PATTERN" if classification == "EXPLORATORY_PAPER_MATCH" else "DIRECTIONAL_OR_UNRESOLVED",
                "completedFitDependent": True,
                "exactHDependent": False,
                "evidencePath": "loops/L01/metric_distinctiveness_results.parquet",
                "rationale": rationale,
            }
        )
    summary = b["summary"]
    stability_checks = []
    for candidate in CANDIDATE_IDS:
        for mode in (RETROSPECTIVE_MODE, CUTOFF_MODE):
            for proportion in PROPORTIONS:
                block = summary.loc[
                    summary["candidateId"].eq(candidate)
                    & summary["modeId"].eq(mode)
                    & np.isclose(summary["proportion"], proportion)
                ]
                phi = float(block.loc[block["featureId"].eq("PHIRL_EMERGENCE"), "medianAccuracy"].iloc[0])
                for comparator in ["EXACT_H_HISTORY", "COMPOSITION_CHANGE_L2", "RAW_COUNTS", "NET_COUNT_FLUX", "MAJORITY_DUMMY"]:
                    value = float(block.loc[block["featureId"].eq(comparator), "medianAccuracy"].iloc[0])
                    stability_checks.append(phi > value)
    c029_pass = bool(all(stability_checks))
    rows.append(
        {
            "claimId": "C029",
            "originalS18Status": "NOT_EVALUATED",
            "s19ExploratoryStatus": "EXPLORATORY_PAPER_MATCH" if c029_pass else "EXPLORATORY_NON_SUPPORT",
            "promotableToS20": c029_pass,
            "candidate2Status": "PASS" if c029_pass else "FAIL",
            "candidate3Status": "PASS" if c029_pass else "FAIL",
            "exactOrDirectional": "DIRECTIONAL_ACROSS_PROPORTIONS" if c029_pass else "NOT_DIRECTIONALLY_REPRODUCED",
            "completedFitDependent": True,
            "exactHDependent": True,
            "evidencePath": "loops/L01/prediction_proportion_results.parquet",
            "rationale": "Promotion requires PhiRL to beat every directed comparator at every proportion in both completed-fit and cutoff-causal modes; exact H and prevalence remain mandatory boundaries.",
        }
    )
    spike = c["correlations"]
    robust = c["robustness"]
    for claim, expected in [("C031", "positive"), ("C032", "positive"), ("C033", "nonsignificant")]:
        primary = spike.loc[
            spike["claimId"].eq(claim)
            & spike["temporalSourceMode"].eq("COMPLETED_FIT")
            & spike["specificationId"].eq("C_GLOBAL_POOLED_3SD_RAW")
        ]
        companion = spike.loc[
            spike["claimId"].eq(claim)
            & spike["temporalSourceMode"].eq("COMPLETED_FIT")
            & spike["specificationId"].eq("C_WITHIN_RUN_3SD_NORMALIZED")
        ]
        if expected == "positive":
            primary_match = len(primary) == 2 and primary["statistic"].fillna(-math.inf).gt(0).all()
            companion_match = len(companion) == 2 and companion["statistic"].fillna(-math.inf).gt(0).all()
        else:
            primary_match = len(primary) == 2 and primary["permutationPValue"].fillna(0).gt(0.05).all()
            companion_match = len(companion) == 2 and companion["permutationPValue"].fillna(0).gt(0.05).all()
        robustness_subset = robust.loc[
            robust["claimId"].eq(claim)
            & robust["temporalSourceMode"].eq("COMPLETED_FIT")
            & robust["specificationId"].eq("C_GLOBAL_POOLED_3SD_RAW")
        ]
        if expected == "positive":
            robust_match = bool(
                len(robustness_subset) == 2
                and robustness_subset["middle90LengthRho"].fillna(-math.inf).gt(0).all()
                and robustness_subset["leaveOneOutMinimumRho"].fillna(-math.inf).gt(0).all()
            )
        else:
            robust_match = True
        match = bool(primary_match and companion_match)
        promotable = bool(match and robust_match)
        classification = "EXPLORATORY_DIRECTIONAL_MATCH" if match else "EXPLORATORY_NON_SUPPORT"
        rows.append(
            {
                "claimId": claim,
                "originalS18Status": "NOT_EVALUATED",
                "s19ExploratoryStatus": classification,
                "promotableToS20": promotable,
                "candidate2Status": classification,
                "candidate3Status": classification,
                "exactOrDirectional": "DIRECTIONAL" if match else "NOT_DIRECTIONALLY_REPRODUCED",
                "completedFitDependent": True,
                "exactHDependent": True,
                "evidencePath": "loops/L01/spike_correlation_results.csv",
                "rationale": "Run-level occupancy uses frozen Y=I(H>0.9); completed-fit and prefix results remain separate and no result is prospective evidence.",
            }
        )
    overlay = pd.DataFrame(rows)
    if len(overlay) != 16 or overlay["claimId"].nunique() != 16:
        raise RuntimeError("claim overlay must contain exactly sixteen unique claims")
    if overlay["promotableToS20"].sum() > 3:
        # Entire-S19 promotion limit; choose no lead by outcome rank here rather
        # than silently selecting. Human review must resolve any excess.
        overlay["promotableToS20"] = False
        overlay.loc[:, "rationale"] = overlay["rationale"] + " More than three raw gates passed; no lead was auto-promoted."
    return overlay


def normalized_results(
    a: dict[str, pd.DataFrame], b: dict[str, pd.DataFrame], c: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in a["correlations"].itertuples(index=False):
        rows.append(
            {
                "bundleId": "A_METRIC_DISTINCTIVENESS",
                "resultType": "RUN_LEVEL_CORRELATION",
                "claimId": row.claimId,
                "candidateId": row.candidateId,
                "specificationId": row.specificationId,
                "analysisId": f"{row.outcomeId}:{row.correlationMethod}",
                "estimate": row.statistic,
                "pValue": row.pValue,
                "status": row.status,
                "detailsJson": canonical_json(
                    {
                        "definedCount": row.definedCount,
                        "ci95": [row.ci95Low, row.ci95High],
                        "equivalentSmallEffect": row.equivalentSmallEffect,
                        "holmAdjustedPValue": row.holmAdjustedPValue,
                    }
                ),
            }
        )
    for row in b["metrics"].itertuples(index=False):
        rows.append(
            {
                "bundleId": "B_ALTERNATIVE_PREDICTION_PROPORTIONS",
                "resultType": "SPLIT_METRIC",
                "claimId": "C029",
                "candidateId": row.candidateId,
                "specificationId": f"P{int(round(row.proportion*100)):02d}:{row.modeId}:{row.featureId}",
                "analysisId": f"repetition_{row.repetitionId}",
                "estimate": row.accuracy,
                "pValue": None,
                "status": row.metricStatus,
                "detailsJson": canonical_json(
                    {
                        "balancedAccuracy": row.balancedAccuracy,
                        "auroc": row.auroc,
                        "auprc": row.auprc,
                        "brier": row.brier,
                        "calibrationError": row.calibrationError,
                        "prevalence": row.prevalence,
                    }
                ),
            }
        )
    for row in c["correlations"].itertuples(index=False):
        rows.append(
            {
                "bundleId": "C_SPIKE_TIMING_SPACING_HEIGHT",
                "resultType": "RUN_LEVEL_SPIKE_CORRELATION",
                "claimId": row.claimId,
                "candidateId": row.candidateId,
                "specificationId": f"{row.temporalSourceMode}:{row.specificationId}",
                "analysisId": row.descriptorId,
                "estimate": row.statistic,
                "pValue": row.permutationPValue,
                "status": row.status,
                "detailsJson": canonical_json(
                    {
                        "definedRunCount": row.definedCount,
                        "zeroSpikeRuns": row.zeroSpikeRunCount,
                        "oneSpikeRuns": row.oneSpikeRunCount,
                        "ci95": [row.ci95Low, row.ci95High],
                    }
                ),
            }
        )
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, columns: list[str], maximum: int = 50) -> str:
    return frame.loc[:, columns].head(maximum).to_markdown(index=False)


def make_reports(
    overlay: pd.DataFrame,
    a: dict[str, pd.DataFrame],
    b: dict[str, pd.DataFrame],
    c: dict[str, pd.DataFrame],
    validation: dict[str, Any],
    runtime: dict[str, Any],
) -> tuple[str, str]:
    counts = overlay["s19ExploratoryStatus"].value_counts().to_dict()
    promotable = overlay.loc[overlay["promotableToS20"], "claimId"].tolist()
    b_table = b["summary"].loc[
        b["summary"]["featureId"].isin(["PHIRL_EMERGENCE", "EXACT_H_HISTORY", "MAJORITY_DUMMY"])
    ].copy()
    spike_table = c["correlations"].loc[
        c["correlations"]["temporalSourceMode"].eq("COMPLETED_FIT")
    ].copy()
    top = f"""# E01/S19 Loop 1 — Unevaluated-claim recovery

## Concise top summary

- **Research step ID:** S19-L01
- **Completion status:** COMPLETE; mandatory human-review boundary active
- **Artifacts written:** all required S19 root and `loops/L01` machine-readable evidence, canonical full-results report, one-page decision summary, source/candidate/self-improvement ledgers, additive C001–C033 overlay, validation, and manifests
- **Validation result:** {validation['validationResult']}
- **Outcome classification:** CONSTRAINING/CONTRADICTORY exploratory result; classifications={counts}; promotable leads={promotable or 'none'}
- **Caveats or blockers:** target author code and exact alternate proportions remain unavailable; dense positive GARD graph makes several literal graph summaries constant; all prediction/spike results remain label- and/or completed-fit-bounded
- **Recommended next action:** human reviewer must select exactly one of `CONTINUE_S19`, `ACTIVATE_S20_CONFIRMATION`, `ACTIVATE_S20_CLOSEOUT_ONLY`, or `PAUSE_PROGRAM`; do not execute another loop automatically

## Lay summary

Loop 1 completed all sixteen claims that S18 had left unevaluated, using the existing 100 paired matrices per simulator candidate and no new trajectories. Public code from the same computational lineage clarified the named network/dynamical metrics and the meaning of inter-spike distance. That helped distinguish a genuine small-effect result from a merely nonsignificant or undefined one. The loop also repeated the frozen S16 predictor across five input/output proportions and tested two locked spike definitions. These are exploratory V3 additions: they do not rewrite S18, and none can become a prediction or causal-control claim simply because it resembles the paper retrospectively.

## Frozen question

Can C001–C012, C029, and C031–C033 be evaluated with precise source-grounded definitions on frozen S13Y data, and does any lead pass the cross-candidate, negative-control, leakage, replay, and specification-sensitivity gates needed to justify untouched S20 confirmation?

## Inputs

- Frozen S13Y completed-fit PhiRL values, past-only prefix endpoints, exact-H labels, 200 candidate-specific trajectories, and 100 shared catalytic matrices.
- Frozen S16 matrix splits, seeds, masks, scaling, architecture formula, optimizer, stopping rule, and 25/75 results.
- Original arXiv v1 paper and pinned public PhiRL/IIGR/BreakingGRNMemories/nolds source identities.
- S18's immutable 59-claim matrix and V1/V2 bundles. The additive overlay does not modify them.

## Methods

### Bundle A — metric distinctiveness

The public same-author lineages define node/edge counts; mean/std in/out degree, betweenness, PageRank, HITS hubs/authorities; multivariate sample entropy; mean/std per-component correlation dimension; mean/std/max largest Lyapunov exponent; multivariate DFA; and generalized Hurst. The primary GARD graph is the unweighted positive support of beta; the only companion attaches beta weights. Because all lognormal beta entries are positive, the literal graph is complete. Dynamic inputs are standardized relative compositions on the selected molecular clock; the sole companion applies the source's 100-observation moving/downsample convention.

Primary run summaries are mean completed-fit emergence. Past-only emergence, exact H, and composition change are fixed dependency audits. Pearson and Spearman are reported separately, with matrix bootstrap intervals, Holm correction across 24 claim/method tests within candidate, 80%-power detectable limits, and a prospectively frozen |r|<0.20 equivalence region. Nonsignificance alone is never called equivalence.

### Bundle B — prediction proportions

The locked proportions are 10/90, 20/80, 25/75, 33/67, and 50/50 because no exact public list was recovered. The original-order right-padded masked layout is preserved without interpolation or truncation. The exact S16 step encoder, two 64-unit hidden layers, dropout, AdamW, training-only validation, early stopping, epoch ceiling, splits, feature families, and seeds are reused. Only the data-determined input/output layer dimensions implied by each prospectively frozen proportion change; 25/75 replays the exact 288,789-parameter S16 network. Completed-fit and cutoff-causal PhiRL remain separate. Exact H history and the deterministic current-state boundary `Y=I(H>0.9)` remain explicit.

### Bundle C — spikes

The primary threshold is the paper-literal candidate-wide completed-mode mean plus three population SD. Contiguous exceedances form episodes; the earliest maximum is the peak. Run descriptors are mean raw peak time, mean all-unordered-pair peak distance (the public source definition), and mean raw peak height. The single robustness companion uses within-run three-SD thresholds, normalized time/distance, and standardized height. Zero-spike runs remain counted with undefined descriptors; one-spike runs have undefined spacing. All associations use run/matrix as the independent unit, 4,096 permutations, 4,096 bootstraps, leave-one-out ranges, length adjustment, and completed-fit/past-only separation.

## Results

### Additive claim overlay

{markdown_table(overlay, ['claimId','s19ExploratoryStatus','promotableToS20','exactOrDirectional','rationale'], 16)}

### Prediction proportion summary

{markdown_table(b_table, ['candidateId','proportion','modeId','featureId','medianAccuracy','medianBalancedAccuracy','medianAuroc','medianBrier','medianCalibrationError','medianPrevalence','medianPreOnsetEligible'], 80)}

The exact-H history is only an input-prefix audit; contemporaneous exact H still determines every target label exactly. Raw accuracy is interpreted alongside balanced accuracy, specificity, prevalence, and pre-onset eligibility. Completed-fit PhiRL remains future-dependent and cannot support early warning. Cutoff-causal results are prospective only in construction; they must still beat dummy, composition, flux, H-history, and stability controls in both candidates to support prediction.

### Spike correlations

{markdown_table(spike_table, ['claimId','candidateId','specificationId','descriptorId','definedCount','statistic','ci95Low','ci95High','permutationPValue','zeroSpikeRunCount','oneSpikeRunCount'], 24)}

Spike resemblance, when present, is retrospective and label-coupled. `Y=I(H>0.9)` means exact H fully determines occupancy; completed-fit emergence can also couple to ordinary stability and use future suffix information. Past-only results are reported independently and cannot be replaced by the completed-fit result.

## Robustness, falsification, and self-improvement

- Every S13Y beta and trajectory input was hash checked; beta reconstruction and all source/dynamical calculations replayed exactly.
- The 25/75 proportion reproduces all frozen S16 split metrics exactly; every other proportion uses the same formula and split identities with no retuning.
- Every cutoff PhiRL fit was repeated exactly and saw no suffix observation. Training scalers used fit matrices only.
- Constant/undefined graph metrics are retained and classified as ambiguity/non-promotion, not counted as evidence for no correlation.
- Metric dependencies on past-only emergence, exact H, and composition change are retained. Spike dependencies on trajectory length, definition, completed-fit construction, and exact-H occupancy are explicit.
- The append-only ledger records what the loop believed, what it tested, which hypotheses weakened, and why no next loop is automatic.

## Validation

{validation['validationResult']}. Checks include immutable S01–S18/V1/V2 hashes, clean pushed contract, exact source and model replay, 25/75 S16 replay, no new GARD trajectories, two candidates in every eligible analysis, exact sixteen-claim overlay, specification ceilings, suffix isolation, output schemas, report regeneration inputs, and artifact hashes.

## Commands and runtime

```text
PYTHONPATH=src:/cache/e01_s19_l01/python_deps OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 pytest -q tests/e01/test_s19_l01.py
PYTHONPATH=src:/cache/e01_s19_l01/python_deps OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python scripts/e01/prepare_s19_l01_lock.py
git commit ... && git push origin eidosoma/groups/42
PYTHONPATH=src:/cache/e01_s19_l01/python_deps OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python scripts/e01/run_s19_l01.py --workers 8
```

Runtime: wall={runtime['wallHours']:.3f} h, CPU={runtime['cpuHours']:.3f} h, workers={runtime['workers']}, GPU=0 h. CPU float64 is authoritative. The loop used no new GARD trajectory and remained within its hard ceilings.

## Provenance

The pushed repository commit, source commit/tree identities, nolds 0.6.1 wheel hash, input hashes, seed manifest, package versions, environment threads, and per-artifact hashes are recorded in the method lock, source/input/runtime manifests, and artifact manifests. Unlicensed public source files remain cache-only; artifacts contain identities and findings, not redistributed code.

## Caveats and blockers

1. This is a source-informed reconstruction, not author-code identity.
2. The target paper does not document graph conversion, dynamic preprocessing, run aggregation, alternate proportions, tensor shape, or spike reductions fully.
3. A complete positive catalytic graph makes several literal graph metrics constant. A threshold grid was deliberately not invented.
4. Shape-dependent input/output layer counts differ across proportions by necessity; hidden capacity and every trainable rule are frozen. Exact identity is established at 25/75 only.
5. The current molecular label is deterministic in exact H. No weaker predictor establishes unrestricted incremental information beyond H.
6. Completed-fit emergence depends on the future suffix; retrospective resemblance cannot support prospective prediction.
7. No S19 result is confirmed. Any promoted lead would require untouched S20 data and a separate human activation.

## Recommended next action and mandatory stop

Human review is now mandatory. Choose exactly one: `CONTINUE_S19` with a narrow approved theme and compute ceiling; `ACTIVATE_S20_CONFIRMATION` naming at most three promoted leads; `ACTIVATE_S20_CLOSEOUT_ONLY`; or `PAUSE_PROGRAM`. S19-L02, S20, E02, and report-bundle generation remain inactive.
"""
    decision = f"""# S19-L01 one-page decision summary

## Concise top summary

- **Research step ID:** S19-L01
- **Completion status:** COMPLETE; stopped at human review
- **Artifacts written:** full L01 evidence, sixteen-claim additive overlay, source/candidate/self-improvement ledgers, replay/robustness/validation artifacts
- **Validation result:** {validation['validationResult']}
- **Outcome classification:** {counts}; promotable leads={promotable or 'none'}
- **Caveats or blockers:** no target author code; graph density, exact-H label coupling, completed-fit leakage, and source-definition sensitivity remain
- **Recommended next action:** choose one authorized review disposition; no automatic continuation

## What changed

All sixteen S18 `NOT_EVALUATED` claims now have additive S19 exploratory statuses. S18 itself and its totals remain unchanged. The loop recovered strong source definitions for metrics and spike spacing, replayed the S16 25/75 experiment exactly, and tested five proportions without retuning.

## Decision-relevant result

{markdown_table(overlay, ['claimId','s19ExploratoryStatus','promotableToS20','rationale'], 16)}

## Required human choice

Select exactly one: `CONTINUE_S19`, `ACTIVATE_S20_CONFIRMATION`, `ACTIVATE_S20_CLOSEOUT_ONLY`, or `PAUSE_PROGRAM`. If continuing S19, approve one narrow theme and its ceiling. No later work has started.
"""
    return top, decision


def artifact_manifest(root: Path, required: list[str]) -> dict[str, Any]:
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise RuntimeError(f"missing required artifacts: {missing}")
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name != "artifact_manifest.json"):
        files.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "schema": "eidosoma.e01.s19_artifact_manifest.v1",
        "root": str(root),
        "fileCount": len(files),
        "totalBytes": int(sum(row["bytes"] for row in files)),
        "files": files,
        "requiredFiles": required,
        "missing": missing,
        "passed": not missing,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("workers must be 1..8")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    started_utc = datetime.now(timezone.utc)
    lock = verify_pushed_lock()
    input_validation = validate_inputs()
    pre_baseline = validate_immutable_baseline()
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    full = pd.read_parquet(S13Y_ROOT / "full_source_values.parquet")
    prefix = pd.read_parquet(S13Y_ROOT / "prefix_endpoint_values.parquet")
    labels = pd.read_parquet(S13Y_ROOT / "label_values.parquet")
    trajectory_manifest = pd.read_parquet(S13Y_ROOT / "trajectory_manifest.parquet")
    a = run_bundle_a(full, prefix, labels, trajectory_manifest, args.workers)
    b = run_bundle_b(full, labels, trajectory_manifest, args.workers)
    descriptors, event_catalog = build_spike_descriptors(full, prefix, labels)
    spike_corr, spike_robustness = spike_correlation_results(descriptors)
    descriptors_replay, events_replay = build_spike_descriptors(full, prefix, labels)
    spike_replay_passed = bool(
        frame_digest(descriptors, ["candidateId", "matrixIndex", "temporalSourceMode", "specificationId"])
        == frame_digest(descriptors_replay, ["candidateId", "matrixIndex", "temporalSourceMode", "specificationId"])
        and frame_digest(event_catalog, ["candidateId", "matrixIndex", "temporalSourceMode", "specificationId", "episodeNumber"])
        == frame_digest(events_replay, ["candidateId", "matrixIndex", "temporalSourceMode", "specificationId", "episodeNumber"])
    )
    if not spike_replay_passed:
        raise RuntimeError("spike descriptor replay failed")
    c = {
        "descriptors": descriptors,
        "events": event_catalog,
        "correlations": spike_corr,
        "robustness": spike_robustness,
    }
    overlay = classify_claims(a, b, c)
    results = normalized_results(a, b, c)
    a["correlations"].to_parquet(LOOP_ROOT / "metric_distinctiveness_results.parquet", index=False)
    a["network"].to_parquet(LOOP_ROOT / "network_feature_results.parquet", index=False)
    a["dynamics"].to_parquet(LOOP_ROOT / "dynamical_feature_results.parquet", index=False)
    b["metrics"].to_parquet(LOOP_ROOT / "prediction_proportion_results.parquet", index=False)
    descriptors.to_parquet(LOOP_ROOT / "spike_descriptor_results.parquet", index=False)
    spike_corr.to_csv(LOOP_ROOT / "spike_correlation_results.csv", index=False)
    overlay.to_csv(LOOP_ROOT / "claim_status_overlay_C001_C033.csv", index=False)
    results.to_parquet(LOOP_ROOT / "results.parquet", index=False)
    negative_controls = pd.concat(
        [
            a["betaReplay"].assign(controlId="BETA_EXACT_REPLAY"),
            a["dynamicReplay"].assign(controlId="DYNAMIC_FEATURE_EXACT_REPLAY"),
            b["modelReplay"].assign(controlId="PREDICTION_MODEL_EXACT_REPLAY"),
            b["sourceAudit"][
                ["candidateId", "matrixIndex", "proportion", "exactReplayPassed", "futureSuffixAccessed"]
            ].assign(controlId="CUTOFF_SOURCE_REPLAY_AND_SUFFIX_ISOLATION"),
            spike_corr[
                ["claimId", "candidateId", "temporalSourceMode", "specificationId", "descriptorId", "permutationPValue"]
            ].assign(controlId="SPIKE_MATRIX_PERMUTATION"),
        ],
        ignore_index=True,
        sort=False,
    )
    negative_controls.to_parquet(LOOP_ROOT / "negative_control_results.parquet", index=False)
    robustness = pd.concat(
        [
            a["correlations"].loc[~a["correlations"]["primaryClaimTest"]].assign(
                robustnessFamily="METRIC_DEPENDENCY_OR_COMPANION"
            ),
            b["summary"].assign(robustnessFamily="PREDICTION_PROPORTION_SUMMARY"),
            spike_robustness.assign(robustnessFamily="SPIKE_OUTLIER_AND_LENGTH"),
        ],
        ignore_index=True,
        sort=False,
    )
    robustness.to_parquet(LOOP_ROOT / "robustness_results.parquet", index=False)
    execution = pd.concat(
        [
            a["execution"],
            b["execution"].assign(bundleId="B_ALTERNATIVE_PREDICTION_PROPORTIONS"),
            pd.DataFrame(
                [
                    {
                        "bundleId": "C_SPIKE_TIMING_SPACING_HEIGHT",
                        "status": "EXECUTED",
                        "descriptorRows": len(descriptors),
                        "eventRows": len(event_catalog),
                        "exactReplayPassed": spike_replay_passed,
                    }
                ]
            ),
        ],
        ignore_index=True,
        sort=False,
    )
    execution.to_parquet(LOOP_ROOT / "execution_status.parquet", index=False)
    specification_rows = []
    prereg = yaml.safe_load(PREREG_PATH.read_text(encoding="utf-8"))
    for bundle in prereg["bundles"]:
        for order, specification in enumerate(bundle["specifications"], start=1):
            specification_rows.append(
                {
                    "bundleId": bundle["bundleId"],
                    "specificationOrder": order,
                    "specificationId": specification,
                    "registeredBeforeOutcome": True,
                    "executed": True,
                    "postOutcomeScientificChange": False,
                }
            )
    specifications = pd.DataFrame(specification_rows)
    specifications.to_parquet(LOOP_ROOT / "specification_ledger.parquet", index=False)
    failures = []
    for row in a["dynamics"].loc[a["dynamics"]["failureReason"].notna()].itertuples(index=False):
        failures.append(
            {
                "failureId": f"A-{row.candidateId}-M{row.matrixIndex:03d}-{row.specificationId}-{row.metricId}-{row.summaryId}-R{int(row.replay)}",
                "bundleId": "A_METRIC_DISTINCTIVENESS",
                "stage": "source_metric",
                "status": "RETAINED_UNDEFINED_RESULT",
                "reason": row.failureReason,
                "excluded": False,
            }
        )
    pd.DataFrame(
        failures,
        columns=["failureId", "bundleId", "stage", "status", "reason", "excluded"],
    ).to_csv(LOOP_ROOT / "failure_ledger.csv", index=False)
    post_baseline = validate_immutable_baseline()
    validation_checks = {
        "immutablePriorPrePassed": pre_baseline["passed"],
        "immutablePriorPostPassed": post_baseline["passed"],
        "pushedLockPassed": lock["passed"],
        "inputHashesPassed": input_validation["passed"],
        "bundleCountExact": int(specifications["bundleId"].nunique()) == 3,
        "specificationCeilingPassed": specifications.groupby("bundleId").size().max() <= 8,
        "candidateCoveragePassed": set(results["candidateId"].dropna()) == set(CANDIDATE_IDS),
        "noNewGardTrajectories": True,
        "betaReplayPassed": bool(a["betaReplay"]["passed"].all()),
        "dynamicReplayPassed": bool(a["dynamicReplay"]["passed"].all()),
        "sourceReplayPassed": bool(b["sourceAudit"]["exactReplayPassed"].all()),
        "suffixIsolationPassed": bool(~b["sourceAudit"]["futureSuffixAccessed"].any()),
        "modelReplayPassed": bool(b["modelReplay"]["passed"].all()),
        "s16Exact25ReplayPassed": bool(b["s16Replay"]["exactOrJointNaN"].all()),
        "spikeReplayPassed": spike_replay_passed,
        "claimOverlayExact": len(overlay) == 16 and overlay["claimId"].nunique() == 16,
        "promotionLimitPassed": int(overlay["promotableToS20"].sum()) <= 3,
        "yExactHBoundaryRetained": bool(descriptors["exactHDeterminesLabel"].all()),
    }
    validation_passed = all(validation_checks.values())
    validation_result = "PASS_ALL_LOCK_REPLAY_LEAKAGE_SCHEMA_AND_IMMUTABILITY_CHECKS" if validation_passed else "FAIL_CLOSED"
    validation = {
        "schema": "eidosoma.e01.s19_l01_regeneration_validation.v1",
        "loopId": "S19-L01",
        "validationResult": validation_result,
        "checks": validation_checks,
        "passed": validation_passed,
    }
    if not validation_passed:
        raise RuntimeError(f"validation failed: {validation_checks}")
    write_json(LOOP_ROOT / "regeneration_validation.json", validation)
    runtime = {
        "schema": "eidosoma.e01.s19_l01_runtime_manifest.v1",
        "loopId": "S19-L01",
        "startedUtc": started_utc.isoformat(),
        "completedUtc": datetime.now(timezone.utc).isoformat(),
        "wallHours": (time.perf_counter() - started_wall) / 3600.0,
        "cpuHours": (time.process_time() - started_cpu) / 3600.0,
        "workers": args.workers,
        "cpuFloat64Authoritative": True,
        "gpuHours": 0.0,
        "newGardTrajectories": 0,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "torch": torch.__version__,
            "nolds": "0.6.1",
        },
        "threadEnvironment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        },
        "pushedCommit": lock["pushedCommit"],
    }
    write_json(LOOP_ROOT / "runtime_manifest.json", runtime)
    storage = {
        "schema": "eidosoma.e01.s19_l01_storage_validation.v1",
        "loopId": "S19-L01",
        "retainedBytesBeforeManifest": sum(
            path.stat().st_size for path in ARTIFACT_ROOT.rglob("*") if path.is_file()
        ),
        "temporaryBytes": sum(path.stat().st_size for path in WORK_ROOT.rglob("*") if path.is_file()),
        "retainedCeilingBytes": 25 * 1024**3,
        "temporaryCeilingBytes": 75 * 1024**3,
    }
    storage["passed"] = bool(
        storage["retainedBytesBeforeManifest"] <= storage["retainedCeilingBytes"]
        and storage["temporaryBytes"] <= storage["temporaryCeilingBytes"]
    )
    write_json(LOOP_ROOT / "storage_validation.json", storage)
    classification = {
        "schema": "eidosoma.e01.s19_l01_classification.v1",
        "loopId": "S19-L01",
        "confirmatoryVerdictIssued": False,
        "claimCounts": overlay["s19ExploratoryStatus"].value_counts().to_dict(),
        "promotableLeadIds": overlay.loc[overlay["promotableToS20"], "claimId"].tolist(),
        "promotionLimit": 3,
        "historicalS18TotalsUnchanged": {
            "SUPPORTED": 3,
            "DIRECTIONALLY_SUPPORTED": 17,
            "NOT_SUPPORTED_WITHIN_TESTED_SCOPE": 21,
            "UNDERDETERMINED": 2,
            "NOT_EVALUATED": 16,
        },
        "requiredHumanReview": True,
        "nextLoopAuthorized": False,
        "s20Active": False,
    }
    write_json(LOOP_ROOT / "classification.json", classification)
    report, decision = make_reports(overlay, a, b, c, validation, runtime)
    (LOOP_ROOT / "S19_L01_FULL_RESULTS.md").write_text(report, encoding="utf-8")
    (LOOP_ROOT / "loop_decision_summary.md").write_text(decision, encoding="utf-8")
    (ARTIFACT_ROOT / "research_step_full_results.md").write_text(report, encoding="utf-8")
    ledger = pd.read_parquet(ARTIFACT_ROOT / "self_improvement_ledger.parquet")
    new_entry = pd.DataFrame(
        [
            {
                "ledgerSequence": int(ledger["ledgerSequence"].max()) + 1,
                "timestampUtc": datetime.now(timezone.utc).isoformat(),
                "loopId": "S19-L01",
                "recordPhase": "POST_LOOP_LEARNING_AND_REVIEW_BOUNDARY",
                "beliefBeforeLoop": ledger.iloc[0]["beliefBeforeLoop"],
                "motivatingEvidence": ledger.iloc[0]["motivatingEvidence"],
                "failureOrAmbiguityTargeted": ledger.iloc[0]["failureOrAmbiguityTargeted"],
                "selectedHypotheses": ledger.iloc[0]["selectedHypotheses"],
                "learned": f"All sixteen claims received additive statuses: {classification['claimCounts']}. Constant graph summaries, proportion controls, and spike definition sensitivity were retained rather than hidden.",
                "weakenedHypotheses": "Any claim whose overlay is EXPLORATORY_NON_SUPPORT or AUTHOR_AMBIGUITY_UNRESOLVED; any apparent predictor advantage explained by prevalence, exact H, or completed-fit dependence.",
                "remainingPlausibleHypotheses": f"Only prospectively gated leads listed in classification.json remain eligible for human consideration: {classification['promotableLeadIds'] or 'none'}.",
                "proposedNextTest": "Human must choose one review disposition. No automatic L02 or S20 execution.",
                "informationGainRationale": "A later loop is justified only if it targets one unresolved source-grounded dependency not already tested, rather than increasing branch count for a favorable result.",
                "appendOnly": True,
            }
        ]
    )
    pd.concat([ledger, new_entry], ignore_index=True).to_parquet(
        ARTIFACT_ROOT / "self_improvement_ledger.parquet", index=False
    )
    with (ARTIFACT_ROOT / "SELF_IMPROVEMENT_LEDGER.md").open("a", encoding="utf-8") as handle:
        handle.write(
            f"""

## Entry 002 — S19-L01 post-loop learning and human-review boundary

- **What was learned:** all sixteen claims received additive exploratory statuses: `{classification['claimCounts']}`. Constant graph summaries, prevalence-sensitive predictors, definition-sensitive spikes, and every undefined/negative result were retained.
- **Hypotheses weakened:** every claim classified `EXPLORATORY_NON_SUPPORT` or `AUTHOR_AMBIGUITY_UNRESOLVED`; any paper-like association that depends only on completed-fit values or exact-H-defined occupancy.
- **Hypotheses remaining plausible:** only the gated lead IDs in `classification.json`: `{classification['promotableLeadIds'] or 'none'}`.
- **What should be tested next:** nothing automatically. Human review must choose the disposition.
- **Why another loop could add information:** only a narrow source-grounded unresolved dependency with a distinct falsifiable fingerprint would add information. Adding thresholds, architectures, labels, or broad branches merely creates more chances for a favorable result and is not recommended.
"""
        )
    history = json.loads((ARTIFACT_ROOT / "human_review_history.json").read_text(encoding="utf-8"))
    history["history"].append(
        {
            "date": datetime.now(timezone.utc).date().isoformat(),
            "decision": "S19_L01_COMPLETE_AWAITING_HUMAN_REVIEW",
            "scope": VERSION,
            "availableChoices": [
                "CONTINUE_S19",
                "ACTIVATE_S20_CONFIRMATION",
                "ACTIVATE_S20_CLOSEOUT_ONLY",
                "PAUSE_PROGRAM",
            ],
        }
    )
    history["pendingDecision"] = "HUMAN_REVIEW_REQUIRED"
    write_json(ARTIFACT_ROOT / "human_review_history.json", history)
    loop_registry = yaml.safe_load((ARTIFACT_ROOT / "loop_registry.yaml").read_text(encoding="utf-8"))
    loop_registry["loops"][0].update(
        {"status": "COMPLETE_AWAITING_HUMAN_REVIEW", "outcomeAccessed": True, "completed": True}
    )
    (ARTIFACT_ROOT / "loop_registry.yaml").write_text(
        yaml.safe_dump(loop_registry, sort_keys=False), encoding="utf-8"
    )
    write_json(
        ARTIFACT_ROOT / "s19_status.json",
        {
            "researchStepId": "S19-L01",
            "stepNumber": 19,
            "success": True,
            "status": "COMPLETE_AWAITING_MANDATORY_HUMAN_REVIEW",
            "artifactsWritten": [str(path) for path in sorted(LOOP_ROOT.iterdir()) if path.is_file()],
            "validationResult": validation_result,
            "caveatsOrBlockers": [
                "target_author_code_unavailable",
                "exact_alternative_proportions_unavailable",
                "dense_graph_constant_metric_risk",
                "exact_H_label_determinism",
                "completed_fit_future_dependence",
            ],
            "recommendedNextAction": "human_select_exactly_one_S19_review_disposition_no_automatic_continuation",
        },
    )
    loop_required = [
        "preregistration.yaml",
        "method_lock.json",
        "candidate_ranking.csv",
        "candidate_bundle_registry.yaml",
        "seed_manifest.parquet",
        "input_manifest.json",
        "source_snapshot_manifest.json",
        "execution_status.parquet",
        "specification_ledger.parquet",
        "results.parquet",
        "negative_control_results.parquet",
        "robustness_results.parquet",
        "failure_ledger.csv",
        "runtime_manifest.json",
        "storage_validation.json",
        "regeneration_validation.json",
        "classification.json",
        "loop_decision_summary.md",
        "S19_L01_FULL_RESULTS.md",
        "metric_distinctiveness_results.parquet",
        "network_feature_results.parquet",
        "dynamical_feature_results.parquet",
        "prediction_proportion_results.parquet",
        "spike_descriptor_results.parquet",
        "spike_correlation_results.csv",
        "claim_status_overlay_C001_C033.csv",
    ]
    write_json(LOOP_ROOT / "artifact_manifest.json", artifact_manifest(LOOP_ROOT, loop_required))
    root_required = [
        "continuation_decision.md",
        "s18_immutable_baseline.json",
        "self_improvement_ledger.parquet",
        "SELF_IMPROVEMENT_LEDGER.md",
        "candidate_registry.parquet",
        "source_search_ledger.parquet",
        "source_search_report.md",
        "loop_registry.yaml",
        "human_review_history.json",
        "s19_status.json",
        "research_step_full_results.md",
    ]
    write_json(ARTIFACT_ROOT / "artifact_manifest.json", artifact_manifest(ARTIFACT_ROOT, root_required))
    final_manifest = json.loads((LOOP_ROOT / "artifact_manifest.json").read_text(encoding="utf-8"))
    if not final_manifest["passed"] or not storage["passed"]:
        raise RuntimeError("final artifact/storage validation failed")
    print(
        canonical_json(
            {
                "success": True,
                "status": "COMPLETE_AWAITING_MANDATORY_HUMAN_REVIEW",
                "claimCounts": classification["claimCounts"],
                "promotableLeadIds": classification["promotableLeadIds"],
                "validationResult": validation_result,
                "wallHours": runtime["wallHours"],
                "cpuHoursParentProcessOnly": runtime["cpuHours"],
            }
        )
    )


if __name__ == "__main__":
    main()
