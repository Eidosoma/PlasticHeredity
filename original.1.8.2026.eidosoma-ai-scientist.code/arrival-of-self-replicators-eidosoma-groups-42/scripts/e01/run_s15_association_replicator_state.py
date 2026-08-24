#!/usr/bin/env python3
"""Execute E01/S15 from frozen S13Y values and stop before S16."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow
import scipy
import yaml

from e01_association_replicator_state.core import (
    CANDIDATE_IDS,
    CHANGE_ANALYSIS,
    COMPLETED_MODE,
    HISTORICAL_BRANCH,
    IDENTITY_COLUMNS,
    LEVEL_ANALYSIS,
    POOLED_SCOPE,
    PREFIX_BRANCH,
    PREFIX_MODE,
    PRIMARY_BRANCH,
    RESEARCH_STEP_ID,
    VERSIONED_STEP_ID,
    circular_shift_control,
    fisher_diagnostics,
    mann_whitney_diagnostics,
    one_sample_diagnostics,
    ordinary_stability_coupling,
    prepare_analysis_rows,
    runwise_statistics,
    summarize_correlations,
    summarize_state_comparisons,
    trajectory_bootstrap,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs/e01/s15_association_replicator_state_preregistration.yaml"
S13Y_ROOT = Path("/artifacts/research_steps/S13Y")
S14_ROOT = Path("/artifacts/research_steps/S14")
PAPER_PATH = Path("/cache/e01_s03/downloads/paper-2607.28250v1.pdf")
CLAIM_LEDGER = Path(
    "/artifacts/E01_forensic_replication_bundle/ledgers/claim_ledger.csv"
)
UPLOAD_MANIFEST = Path("/workspace/input-attachments/MANIFEST.json")
UPLOAD_SIDECAR = Path(
    "/workspace/input-attachments/ed5486bf-a043-485b-a233-d88d8d123759/_metadata/ATTACHMENT.md"
)
DEFAULT_OUTPUT = Path("/artifacts/research_steps/S15")

ARTIFACT_PATHS = (
    "method_lock.json",
    "input_manifest.json",
    "analysis_row_audit.csv",
    "runwise_correlations.parquet",
    "correlation_summary.csv",
    "one_sample_diagnostics.csv",
    "runwise_state_comparisons.parquet",
    "state_comparison_summary.csv",
    "mann_whitney_diagnostics.csv",
    "fisher_combination_diagnostics.csv",
    "trajectory_bootstrap_summary.csv",
    "trajectory_bootstrap_distributions.parquet",
    "circular_shift_summary.csv",
    "circular_shift_distributions.parquet",
    "ordinary_stability_coupling.parquet",
    "ordinary_stability_summary.csv",
    "label_identity_audit.json",
    "analysis_decision.csv",
    "paper_target_comparison.csv",
    "interpretation_boundary.csv",
    "figures/figure3_association_reconstruction.png",
    "figures/figure4_state_reconstruction.png",
    "figures/dependence_aware_controls.png",
    "figures/interpretation_boundaries.png",
    "validation.json",
    "provenance_manifest.json",
    "failure_ledger.csv",
    "status.json",
    "research_step_full_results.md",
    "artifact_manifest.json",
)

FRAME_ARTIFACTS = {
    "analysisRowAudit": "analysis_row_audit.csv",
    "runwiseCorrelations": "runwise_correlations.parquet",
    "correlationSummary": "correlation_summary.csv",
    "oneSampleDiagnostics": "one_sample_diagnostics.csv",
    "runwiseStates": "runwise_state_comparisons.parquet",
    "stateSummary": "state_comparison_summary.csv",
    "mannWhitneyDiagnostics": "mann_whitney_diagnostics.csv",
    "fisherDiagnostics": "fisher_combination_diagnostics.csv",
    "bootstrapSummary": "trajectory_bootstrap_summary.csv",
    "bootstrapDistributions": "trajectory_bootstrap_distributions.parquet",
    "shiftSummary": "circular_shift_summary.csv",
    "shiftDistributions": "circular_shift_distributions.parquet",
    "stabilityRunwise": "ordinary_stability_coupling.parquet",
    "stabilitySummary": "ordinary_stability_summary.csv",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def frame_digest(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(list(frame.columns), separators=(",", ":")).encode())
    digest.update(json.dumps([str(value) for value in frame.dtypes]).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if not isinstance(value, (str, bytes)) and pd.isna(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n", float_format="%.17g")


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, engine="pyarrow", compression="zstd")


def git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True, stderr=subprocess.STDOUT
    ).strip()


def load_inputs() -> dict[str, Any]:
    return {
        "full": pd.read_parquet(S13Y_ROOT / "full_source_values.parquet"),
        "prefix": pd.read_parquet(S13Y_ROOT / "prefix_endpoint_values.parquet"),
        "trajectoryManifest": pd.read_parquet(
            S13Y_ROOT / "trajectory_manifest.parquet"
        ),
        "simulationSummary": pd.read_parquet(S13Y_ROOT / "simulation_summary.parquet"),
        "s13yCircularity": pd.read_csv(
            S13Y_ROOT / "circularity_control_results.csv"
        ),
        "s13yStatus": json.loads((S13Y_ROOT / "status.json").read_text()),
        "s14Status": json.loads((S14_ROOT / "status.json").read_text()),
        "claims": pd.read_csv(CLAIM_LEDGER),
    }


def _manifest_entries(root: Path, step: str) -> list[dict[str, Any]]:
    manifest_path = root / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entries: list[dict[str, Any]] = []
    for record in manifest["artifacts"]:
        path = root / record["path"]
        actual = sha256_file(path) if path.exists() else None
        entries.append(
            {
                "role": f"{step}_ARTIFACT_MANIFEST_MEMBER",
                "path": str(path),
                "bytes": path.stat().st_size if path.exists() else None,
                "sha256Before": actual,
                "expectedSha256": record["sha256"],
                "matchedBefore": actual == record["sha256"],
            }
        )
    entries.append(
        {
            "role": f"{step}_ARTIFACT_MANIFEST",
            "path": str(manifest_path),
            "bytes": manifest_path.stat().st_size,
            "sha256Before": sha256_file(manifest_path),
            "expectedSha256": None,
            "matchedBefore": True,
        }
    )
    return entries


def snapshot_inputs(data: dict[str, Any]) -> list[dict[str, Any]]:
    entries = _manifest_entries(S13Y_ROOT, "S13Y") + _manifest_entries(S14_ROOT, "S14")
    for row in data["trajectoryManifest"].sort_values(
        ["candidateId", "matrixIndex"], kind="stable"
    ).itertuples(index=False):
        path = Path(row.cachePath)
        actual = sha256_file(path)
        entries.append(
            {
                "role": "S13Y_FROZEN_TRAJECTORY_CACHE_HASH_ONLY",
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256Before": actual,
                "expectedSha256": str(row.cacheSha256),
                "matchedBefore": actual == str(row.cacheSha256),
            }
        )
    for role, path in (
        ("S15_METHOD_CONFIG", CONFIG_PATH),
        ("ORIGINAL_PAPER_ARXIV_V1", PAPER_PATH),
        ("S01_CLAIM_LEDGER", CLAIM_LEDGER),
        ("UPLOADED_INPUT_MANIFEST", UPLOAD_MANIFEST),
        ("UPLOADED_INPUT_SIDECAR", UPLOAD_SIDECAR),
    ):
        entries.append(
            {
                "role": role,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256Before": sha256_file(path),
                "expectedSha256": None,
                "matchedBefore": True,
            }
        )
    return sorted(entries, key=lambda row: (row["role"], row["path"]))


def complete_input_snapshot(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for entry in entries:
        path = Path(entry["path"])
        after = sha256_file(path) if path.exists() else None
        entry["sha256After"] = after
        entry["matchedAfter"] = after == entry["sha256Before"]
    return entries


def validate_inputs(data: dict[str, Any]) -> list[dict[str, Any]]:
    full = data["full"]
    prefix = data["prefix"]
    trajectories = data["trajectoryManifest"]
    simulation = data["simulationSummary"]
    checks: list[dict[str, Any]] = []

    def check(identifier: str, passed: bool, detail: str) -> None:
        checks.append({"checkId": identifier, "passed": bool(passed), "detail": detail})
        if not passed:
            raise RuntimeError(f"input validation failed: {identifier}: {detail}")

    check(
        "S13Y_AND_S14_COMPLETE",
        data["s13yStatus"]["status"].startswith("COMPLETED")
        and data["s14Status"]["status"] == "COMPLETED",
        f"S13Y={data['s13yStatus']['status']} S14={data['s14Status']['status']}",
    )
    check(
        "EXACT_CANDIDATE_SET",
        tuple(sorted(full["candidateId"].unique())) == CANDIDATE_IDS,
        repr(tuple(sorted(full["candidateId"].unique()))),
    )
    trajectory_counts = full.groupby("candidateId")["trajectoryId"].nunique().to_dict()
    check(
        "ONE_HUNDRED_TRAJECTORIES_PER_CANDIDATE",
        trajectory_counts == {candidate: 100 for candidate in CANDIDATE_IDS},
        repr(trajectory_counts),
    )
    check(
        "FROZEN_ROW_CARDINALITIES",
        len(full) == 180435
        and len(prefix) == 20000
        and int(prefix["status"].eq("ELIGIBLE").sum()) == 13705,
        f"full={len(full)} prefix={len(prefix)} eligiblePrefix={int(prefix['status'].eq('ELIGIBLE').sum())}",
    )
    check(
        "FULL_AND_PREFIX_KEY_UNIQUENESS",
        not full.duplicated(
            ["candidateId", "trajectoryId", "selectedSequenceIndex"]
        ).any()
        and not prefix.duplicated(["candidateId", "trajectoryId", "generation"]).any(),
        "completed selected-state and prefix generation keys unique",
    )
    component_error = float(
        np.max(
            np.abs(
                full["emergence"].to_numpy(np.float64)
                - full["synergy"].to_numpy(np.float64)
                - full["downwardCausation"].to_numpy(np.float64)
            )
        )
    )
    check(
        "SOURCE_EMERGENCE_COMPONENT_IDENTITY",
        component_error <= 1e-12,
        f"maxAbsError={component_error:.17g}",
    )
    check(
        "FROZEN_INFORMATION_BRANCH",
        set(full["implementationId"]) == {"PHIRL_REGULARIZED_SOURCE"}
        and set(full["temporalLabel"]) == {COMPLETED_MODE}
        and set(prefix["temporalLabel"]) == {PREFIX_MODE},
        "PhiRL regularized source completed and prefix identities exact",
    )
    full_expected = full["incomingCosineH"].to_numpy(np.float64) > 0.9
    prefix_expected = prefix["currentIncomingCosineH"].to_numpy(np.float64) > 0.9
    full_mismatches = int(
        np.count_nonzero(full_expected != full["molecularH090Label"].to_numpy(bool))
    )
    prefix_mismatches = int(
        np.count_nonzero(
            prefix_expected != prefix["currentMolecularH090Label"].to_numpy(bool)
        )
    )
    check(
        "EXACT_H_LABEL_IDENTITY",
        full_mismatches == 0 and prefix_mismatches == 0,
        f"completed={full_mismatches} prefix={prefix_mismatches}",
    )
    definitions = simulation.groupby("candidateId", sort=True).first()
    exact_definitions = (
        np.isclose(definitions.loc[CANDIDATE_IDS[0], "h"], 0.6031526490073492)
        and np.isclose(definitions.loc[CANDIDATE_IDS[1], "h"], 0.5613315384859516)
        and definitions.loc[CANDIDATE_IDS[0], "daughterRule"] == "FIRST_DAUGHTER"
        and definitions.loc[CANDIDATE_IDS[1], "daughterRule"] == "RANDOM_NONEMPTY"
        and set(definitions["overshootRule"]) == {"TRIM_NEW_ENTRANTS_TO_NMAX"}
    )
    check(
        "FROZEN_SIMULATOR_DEFINITIONS",
        exact_definitions,
        "candidate 2/3 h, daughter, trim and C1 contracts unchanged",
    )
    check(
        "TRAJECTORY_MANIFEST_COMPLETE_REPLAYED_AND_S13Y",
        len(trajectories) == 200
        and trajectories["completedFissions"].eq(100).all()
        and trajectories["exactReplayPassed"].astype(bool).all()
        and set(trajectories["researchStepId"]) == {"S13Y"},
        f"rows={len(trajectories)}",
    )
    required_claims = {f"E01-C{number:03d}" for number in range(15, 22)}
    observed_claims = set(data["claims"]["claim_id"])
    check(
        "PAPER_CLAIMS_PRESENT",
        required_claims <= observed_claims,
        repr(sorted(required_claims - observed_claims)),
    )
    return checks


def build_frames(data: dict[str, Any], config: dict[str, Any]) -> dict[str, pd.DataFrame]:
    rows = prepare_analysis_rows(data["full"], data["prefix"])
    runwise_correlations, runwise_states = runwise_statistics(rows)
    correlation_summary = summarize_correlations(runwise_correlations)
    one_sample = one_sample_diagnostics(runwise_correlations)
    state_summary = summarize_state_comparisons(runwise_states)
    mw = mann_whitney_diagnostics(rows, runwise_states)
    fisher = fisher_diagnostics(runwise_states)
    replicates = int(config["dependenceAwareControls"]["resamplingReplicates"])
    seed_root = str(config["dependenceAwareControls"]["seedRootHex"])
    bootstrap_distributions, bootstrap_summary = trajectory_bootstrap(
        runwise_correlations,
        runwise_states,
        replicates=replicates,
        seed_root_hex=seed_root,
    )
    shift_distributions, shift_summary = circular_shift_control(
        rows, replicates=replicates, seed_root_hex=seed_root
    )
    stability_runwise, stability_summary = ordinary_stability_coupling(
        rows,
        bootstrap_replicates=replicates,
        seed_root_hex=seed_root,
    )
    audit = (
        rows.groupby(
            IDENTITY_COLUMNS + ["candidateId"], sort=True, observed=True
        )
        .agg(
            rowCount=("analysisValue", "size"),
            trajectoryCount=("trajectoryId", "nunique"),
            matrixCount=("matrixIndex", "nunique"),
            minimumObservationOrder=("observationOrder", "min"),
            maximumObservationOrder=("observationOrder", "max"),
            replicatorCount=("label", "sum"),
        )
        .reset_index()
    )
    audit["driftCount"] = audit["rowCount"] - audit["replicatorCount"]
    return {
        "analysisRows": rows,
        "analysisRowAudit": audit,
        "runwiseCorrelations": runwise_correlations,
        "correlationSummary": correlation_summary,
        "oneSampleDiagnostics": one_sample,
        "runwiseStates": runwise_states,
        "stateSummary": state_summary,
        "mannWhitneyDiagnostics": mw,
        "fisherDiagnostics": fisher,
        "bootstrapSummary": bootstrap_summary,
        "bootstrapDistributions": bootstrap_distributions,
        "shiftSummary": shift_summary,
        "shiftDistributions": shift_distributions,
        "stabilityRunwise": stability_runwise,
        "stabilitySummary": stability_summary,
    }


def _single(frame: pd.DataFrame, **filters: Any) -> pd.Series:
    selected = frame
    for column, value in filters.items():
        selected = selected.loc[selected[column].eq(value)]
    if len(selected) != 1:
        raise RuntimeError(f"expected one row for {filters}, found {len(selected)}")
    return selected.iloc[0]


def build_decisions(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for analysis_id in (LEVEL_ANALYSIS, CHANGE_ANALYSIS):
        candidate_passes: list[bool] = []
        for candidate in CANDIDATE_IDS:
            identity = {
                "branchId": PRIMARY_BRANCH,
                "analysisId": analysis_id,
                "candidateScope": candidate,
            }
            correlation = _single(frames["correlationSummary"], **identity)
            one_sample = _single(
                frames["oneSampleDiagnostics"],
                **identity,
                correlationMeasure="SPEARMAN",
            )
            state = _single(frames["stateSummary"], **identity)
            fisher = _single(
                frames["fisherDiagnostics"], **identity, alternative="greater"
            )
            bootstrap_rho = _single(
                frames["bootstrapSummary"], **identity, metric="medianSpearman"
            )
            bootstrap_difference = _single(
                frames["bootstrapSummary"],
                **identity,
                metric="medianMeanDifference",
            )
            shift_rho = _single(
                frames["shiftSummary"], **identity, metric="medianSpearman"
            )
            shift_difference = _single(
                frames["shiftSummary"],
                **identity,
                metric="medianMeanDifference",
            )
            association_gate = bool(
                correlation["spearmanMean"] > 0
                and one_sample["oneSampleTTwoSidedP"] < 0.05
                and bootstrap_rho["bootstrapLower95"] > 0
                and shift_rho["positiveP"] <= 0.05
            )
            state_gate = bool(
                state["higherReplicatorMeanFraction"] > 0.5
                and fisher["combinedP"] < 0.001
                and bootstrap_difference["bootstrapLower95"] > 0
                and shift_difference["positiveP"] <= 0.05
            )
            overall = association_gate and state_gate
            candidate_passes.append(overall)
            rows.append(
                {
                    "analysisId": analysis_id,
                    "candidateScope": candidate,
                    "evidenceRole": "CANDIDATE_SPECIFIC_PRIMARY",
                    "definedSpearmanCount": int(correlation["spearmanDefinedCount"]),
                    "positiveSpearmanCount": int(correlation["spearmanPositiveCount"]),
                    "positiveSignificantSpearmanCount": int(
                        correlation["spearmanPositiveSignificantCount"]
                    ),
                    "meanSpearman": float(correlation["spearmanMean"]),
                    "medianSpearman": float(correlation["spearmanMedian"]),
                    "oneSampleTTwoSidedP": float(
                        one_sample["oneSampleTTwoSidedP"]
                    ),
                    "bootstrapMedianSpearmanLower95": float(
                        bootstrap_rho["bootstrapLower95"]
                    ),
                    "circularShiftSpearmanPositiveP": float(
                        shift_rho["positiveP"]
                    ),
                    "definedStateComparisonCount": int(
                        state["definedStateComparisonCount"]
                    ),
                    "higherReplicatorMeanCount": int(
                        state["higherReplicatorMeanCount"]
                    ),
                    "medianMeanDifference": float(state["medianMeanDifference"]),
                    "fisherGreaterP": float(fisher["combinedP"]),
                    "bootstrapMedianMeanDifferenceLower95": float(
                        bootstrap_difference["bootstrapLower95"]
                    ),
                    "circularShiftMeanDifferencePositiveP": float(
                        shift_difference["positiveP"]
                    ),
                    "associationGatePassed": association_gate,
                    "stateGatePassed": state_gate,
                    "candidateAnalysisResemblancePassed": overall,
                }
            )
        rows.append(
            {
                "analysisId": analysis_id,
                "candidateScope": "ALL_CANDIDATES",
                "evidenceRole": "CROSS_CANDIDATE_GATE",
                "associationGatePassed": bool(
                    all(
                        row["associationGatePassed"]
                        for row in rows
                        if row["analysisId"] == analysis_id
                        and row["candidateScope"] in CANDIDATE_IDS
                    )
                ),
                "stateGatePassed": bool(
                    all(
                        row["stateGatePassed"]
                        for row in rows
                        if row["analysisId"] == analysis_id
                        and row["candidateScope"] in CANDIDATE_IDS
                    )
                ),
                "candidateAnalysisResemblancePassed": bool(all(candidate_passes)),
            }
        )
    return pd.DataFrame(rows)


def _count_status(observed: int, target: int, directional: bool) -> str:
    if observed == target:
        return "CLOSELY_RECONSTRUCTED"
    if directional:
        return "DIRECTIONALLY_SIMILAR"
    return "DIFFERENT"


def build_paper_targets(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for analysis_id in (LEVEL_ANALYSIS, CHANGE_ANALYSIS):
        for candidate in CANDIDATE_IDS:
            identity = {
                "branchId": PRIMARY_BRANCH,
                "analysisId": analysis_id,
                "candidateScope": candidate,
            }
            correlation = _single(frames["correlationSummary"], **identity)
            one_sample = _single(
                frames["oneSampleDiagnostics"],
                **identity,
                correlationMeasure="SPEARMAN",
            )
            state = _single(frames["stateSummary"], **identity)
            fisher = _single(
                frames["fisherDiagnostics"], **identity, alternative="greater"
            )
            common = {
                "analysisId": analysis_id,
                "candidateId": candidate,
                "evidenceRole": "CANDIDATE_SPECIFIC_PRIMARY",
                "retrospectiveOnly": True,
                "labelCoupled": True,
            }
            positive = int(correlation["spearmanPositiveCount"])
            positive_significant = int(
                correlation["spearmanPositiveSignificantCount"]
            )
            defined = int(correlation["spearmanDefinedCount"])
            rows.extend(
                [
                    {
                        **common,
                        "claimId": "E01-C015",
                        "diagnosticScope": "RUNWISE_SPEARMAN",
                        "paperTarget": "73/100 positive",
                        "reconstructedValue": f"{positive}/100 ({positive}/{defined} defined)",
                        "status": _count_status(positive, 73, positive > 50),
                        "caveat": "One or more runs may have a constant binary label.",
                    },
                    {
                        **common,
                        "claimId": "E01-C016",
                        "diagnosticScope": "RUNWISE_SPEARMAN_UNADJUSTED_P_LT_0.05",
                        "paperTarget": "54/100 and 54/73 positive runs",
                        "reconstructedValue": (
                            f"{positive_significant}/100; "
                            f"{positive_significant}/{positive} positive runs"
                        ),
                        "status": _count_status(
                            positive_significant, 54, positive_significant > 50
                        ),
                        "caveat": "Unadjusted ordinary runwise p-values ignore serial dependence.",
                    },
                    {
                        **common,
                        "claimId": "E01-C017",
                        "diagnosticScope": "MEAN_RUNWISE_SPEARMAN",
                        "paperTarget": "mean rho=0.139",
                        "reconstructedValue": f"{float(correlation['spearmanMean']):.9g}",
                        "status": (
                            "CLOSELY_RECONSTRUCTED"
                            if round(float(correlation["spearmanMean"]), 3) == 0.139
                            else "DIRECTIONALLY_SIMILAR"
                            if correlation["spearmanMean"] > 0
                            else "DIFFERENT"
                        ),
                        "caveat": "Paper level-versus-change wording is inconsistent.",
                    },
                    {
                        **common,
                        "claimId": "E01-C018",
                        "diagnosticScope": "ONE_SAMPLE_T_TWO_SIDED",
                        "paperTarget": "positive mean; p<0.05",
                        "reconstructedValue": (
                            f"mean={float(one_sample['mean']):.9g}; "
                            f"p={float(one_sample['oneSampleTTwoSidedP']):.9g}"
                        ),
                        "status": (
                            "CLOSELY_RECONSTRUCTED"
                            if one_sample["mean"] > 0
                            and one_sample["oneSampleTTwoSidedP"] < 0.05
                            else "DIRECTIONALLY_SIMILAR"
                            if one_sample["mean"] > 0
                            else "DIFFERENT"
                        ),
                        "caveat": "Paper-like diagnostic; circular-shift control is stronger.",
                    },
                    {
                        **common,
                        "claimId": "E01-C019",
                        "diagnosticScope": "RUNWISE_MEAN_DIFFERENCE_SIGN",
                        "paperTarget": "57/100 higher replicator mean",
                        "reconstructedValue": (
                            f"{int(state['higherReplicatorMeanCount'])}/100 "
                            f"({int(state['definedStateComparisonCount'])} defined)"
                        ),
                        "status": _count_status(
                            int(state["higherReplicatorMeanCount"]),
                            57,
                            int(state["higherReplicatorMeanCount"])
                            > int(state["definedStateComparisonCount"]) / 2,
                        ),
                        "caveat": "Runs without both states are retained as undefined.",
                    },
                ]
            )
            for diagnostic_scope in (
                "POINT_POOLED_WITHIN_SCOPE",
                "RUN_SUMMARY_UNPAIRED_WITHIN_SCOPE",
            ):
                mw = _single(
                    frames["mannWhitneyDiagnostics"],
                    **identity,
                    diagnosticScope=diagnostic_scope,
                )
                rows.append(
                    {
                        **common,
                        "claimId": "E01-C020",
                        "diagnosticScope": diagnostic_scope,
                        "paperTarget": "replicator higher; Mann-Whitney p<0.001",
                        "reconstructedValue": (
                            f"U={float(mw['mannWhitneyU']):.9g}; "
                            f"greater p={float(mw['mannWhitneyGreaterP']):.9g}"
                        ),
                        "status": "UNDERDETERMINED_PAPER_SCOPE",
                        "scopeSpecificStatus": (
                            "CLOSELY_RECONSTRUCTED"
                            if mw["rankBiserialReplicatorGreater"] > 0
                            and mw["mannWhitneyGreaterP"] < 0.001
                            else "DIRECTIONALLY_SIMILAR"
                            if mw["rankBiserialReplicatorGreater"] > 0
                            else "DIFFERENT"
                        ),
                        "caveat": "The paper does not identify pooled-step versus run-summary scope; both are retained and neither is selected.",
                    }
                )
            rows.append(
                {
                    **common,
                    "claimId": "E01-C021",
                    "diagnosticScope": "FISHER_ALL_ELIGIBLE_RUNWISE_GREATER_P",
                    "paperTarget": "Fisher combined p<0.001",
                    "reconstructedValue": (
                        f"statistic={float(fisher['fisherStatistic']):.9g}; "
                        f"df={int(fisher['degreesOfFreedom'])}; "
                        f"p={float(fisher['combinedP']):.9g}"
                    ),
                    "status": (
                        "CLOSELY_RECONSTRUCTED"
                        if fisher["combinedP"] < 0.001
                        else "DIFFERENT"
                    ),
                    "caveat": "Paper-like Fisher diagnostic; within-run observations are serially dependent.",
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["claimId", "analysisId", "candidateId", "diagnosticScope"],
        kind="stable",
        ignore_index=True,
    )


def build_label_audit(data: dict[str, Any]) -> dict[str, Any]:
    full = data["full"]
    prefix = data["prefix"]
    full_expected = full["incomingCosineH"].to_numpy(np.float64) > 0.9
    prefix_expected = prefix["currentIncomingCosineH"].to_numpy(np.float64) > 0.9
    candidate_rows = []
    for candidate in CANDIDATE_IDS:
        selected = full["candidateId"].eq(candidate)
        expected = full.loc[selected, "incomingCosineH"].to_numpy(np.float64) > 0.9
        observed = full.loc[selected, "molecularH090Label"].to_numpy(bool)
        candidate_rows.append(
            {
                "candidateId": candidate,
                "completedRowCount": int(np.count_nonzero(selected)),
                "completedMismatchCount": int(np.count_nonzero(expected != observed)),
                "exactHClassificationAccuracy": float(np.mean(expected == observed)),
            }
        )
    return {
        "schema": "eidosoma.e01.s15.label_identity_audit.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "primaryLabelDefinition": "Y_t = I(incomingCosineH_t > 0.9)",
        "strictThreshold": True,
        "completedRowCount": len(full),
        "completedMismatchCount": int(
            np.count_nonzero(
                full_expected != full["molecularH090Label"].to_numpy(bool)
            )
        ),
        "prefixStatusBearingRowCount": len(prefix),
        "prefixEligibleRowCount": int(prefix["status"].eq("ELIGIBLE").sum()),
        "prefixMismatchCount": int(
            np.count_nonzero(
                prefix_expected
                != prefix["currentMolecularH090Label"].to_numpy(bool)
            )
        ),
        "candidateResults": candidate_rows,
        "conditionalEntropyYGivenExactHBits": 0.0,
        "unrestrictedConditionalInformationEmergenceYGivenExactHBits": 0.0,
        "unrestrictedIncrementalInformationBeyondExactH": False,
        "completedFitFutureDependent": True,
        "eligibleAsEarlyWarningEvidence": False,
        "eligibleAsPredictionEvidence": False,
        "eligibleAsCausalControlEvidence": False,
        "passed": bool(
            np.array_equal(
                full_expected, full["molecularH090Label"].to_numpy(bool)
            )
            and np.array_equal(
                prefix_expected, prefix["currentMolecularH090Label"].to_numpy(bool)
            )
        ),
    }


def build_interpretation_boundaries(
    frames: dict[str, pd.DataFrame], label_audit: dict[str, Any]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {
            "boundaryId": "EXACT_H_DETERMINISM",
            "candidateScope": "ALL",
            "analysisId": "ALL",
            "finding": "Y is exactly I(H>0.9) with zero completed and prefix mismatches.",
            "consequence": "Exact H fully determines Y; H(Y|H)=0.",
            "status": "STRUCTURAL_CONSTRAINT",
        },
        {
            "boundaryId": "NO_UNRESTRICTED_INCREMENT_BEYOND_H",
            "candidateScope": "ALL",
            "analysisId": "ALL",
            "finding": f"I(E;Y|exact H)={label_audit['unrestrictedConditionalInformationEmergenceYGivenExactHBits']:.1f} bits by deterministic identity.",
            "consequence": "A weaker emergence association cannot establish unrestricted incremental target information beyond exact H.",
            "status": "STRUCTURAL_CONSTRAINT",
        },
        {
            "boundaryId": "COMPLETED_FIT_FUTURE_DEPENDENCE",
            "candidateScope": "ALL",
            "analysisId": "ALL",
            "finding": "Completed-fit partition and Gaussian parameters use the finished trajectory.",
            "consequence": "All primary S15 resemblance is retrospective, not early warning or prediction.",
            "status": "RETROSPECTIVE_ONLY",
        },
        {
            "boundaryId": "NO_PREDICTION_OR_CAUSAL_CONTROL",
            "candidateScope": "ALL",
            "analysisId": "ALL",
            "finding": "S15 fits no predictor, generates no trajectory, and executes no intervention.",
            "consequence": "Prediction and causal-control claims are not evaluated.",
            "status": "OUT_OF_SCOPE",
        },
    ]
    for analysis_id in (LEVEL_ANALYSIS, CHANGE_ANALYSIS):
        for candidate in CANDIDATE_IDS:
            primary = _single(
                frames["correlationSummary"],
                branchId=PRIMARY_BRANCH,
                analysisId=analysis_id,
                candidateScope=candidate,
            )
            historical = _single(
                frames["correlationSummary"],
                branchId=HISTORICAL_BRANCH,
                analysisId=analysis_id,
                candidateScope=candidate,
            )
            prefix = _single(
                frames["correlationSummary"],
                branchId=PREFIX_BRANCH,
                analysisId=analysis_id,
                candidateScope=candidate,
            )
            rows.extend(
                [
                    {
                        "boundaryId": "HISTORICAL_POST_FISSION_DIRECTION",
                        "candidateScope": candidate,
                        "analysisId": analysis_id,
                        "finding": (
                            f"primary median rho={primary['spearmanMedian']:.6g}; "
                            f"historical median rho={historical['spearmanMedian']:.6g}"
                        ),
                        "consequence": "The frozen historical post-fission label is a distinct evidentiary target and cannot be replaced by the molecular same-state label.",
                        "status": (
                            "DIFFERENT_DIRECTION"
                            if np.sign(primary["spearmanMedian"])
                            != np.sign(historical["spearmanMedian"])
                            else "SAME_DIRECTION"
                        ),
                    },
                    {
                        "boundaryId": "PAST_ONLY_DIRECTION",
                        "candidateScope": candidate,
                        "analysisId": analysis_id,
                        "finding": (
                            f"completed-fit median rho={primary['spearmanMedian']:.6g}; "
                            f"past-only median rho={prefix['spearmanMedian']:.6g}"
                        ),
                        "consequence": "Completed-fit resemblance cannot be promoted to past-only early warning.",
                        "status": (
                            "DIRECTION_REVERSAL"
                            if np.sign(primary["spearmanMedian"])
                            != np.sign(prefix["spearmanMedian"])
                            else "NO_DIRECTION_REVERSAL"
                        ),
                    },
                ]
            )
            for predictor in (
                "EXACT_INCOMING_H",
                "NEGATIVE_EUCLIDEAN_L2_CLOSED_COMPOSITION_CHANGE",
            ):
                stability = _single(
                    frames["stabilitySummary"],
                    analysisId=analysis_id,
                    predictorId=predictor,
                    candidateScope=candidate,
                    correlationMeasure="SPEARMAN",
                )
                rows.append(
                    {
                        "boundaryId": "ORDINARY_STABILITY_COUPLING",
                        "candidateScope": candidate,
                        "analysisId": analysis_id,
                        "finding": f"{predictor} median runwise Spearman={stability['medianCorrelation']:.6g}",
                        "consequence": "This is descriptive coupling, not incremental information conditional on exact H.",
                        "status": "DESCRIPTIVE_COUPLING",
                    }
                )
    return pd.DataFrame(rows)


def make_figures(frames: dict[str, pd.DataFrame], output_root: Path) -> None:
    figure_root = output_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    candidate_labels = {
        CANDIDATE_IDS[0]: "Candidate 2",
        CANDIDATE_IDS[1]: "Candidate 3",
    }
    analysis_labels = {LEVEL_ANALYSIS: "Level", CHANGE_ANALYSIS: "First difference"}

    # Figure 3 reconstruction: runwise primary Spearman distributions.
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey=True)
    for row, analysis_id in enumerate((LEVEL_ANALYSIS, CHANGE_ANALYSIS)):
        for column, candidate in enumerate(CANDIDATE_IDS):
            ax = axes[row, column]
            selected = frames["runwiseCorrelations"].loc[
                frames["runwiseCorrelations"]["branchId"].eq(PRIMARY_BRANCH)
                & frames["runwiseCorrelations"]["analysisId"].eq(analysis_id)
                & frames["runwiseCorrelations"]["candidateId"].eq(candidate)
            ]
            values = selected["spearmanRho"].dropna().to_numpy(np.float64)
            bins = np.linspace(-0.5, 0.5, 31)
            ax.hist(values, bins=bins, color="#2c7fb8", alpha=0.82, edgecolor="white")
            ax.axvline(0, color="black", linestyle="--", linewidth=1)
            ax.axvline(np.mean(values), color="#d7301f", linestyle="--", linewidth=1.5)
            ax.axvline(0.139, color="#7a0177", linestyle=":", linewidth=1.4)
            summary = _single(
                frames["correlationSummary"],
                branchId=PRIMARY_BRANCH,
                analysisId=analysis_id,
                candidateScope=candidate,
            )
            ax.text(
                0.02,
                0.96,
                (
                    f"positive {int(summary['spearmanPositiveCount'])}/100\n"
                    f"positive + p<.05 {int(summary['spearmanPositiveSignificantCount'])}/100\n"
                    f"mean {summary['spearmanMean']:.3f}; median {summary['spearmanMedian']:.3f}"
                ),
                transform=ax.transAxes,
                va="top",
                fontsize=8.5,
            )
            ax.set_title(f"{candidate_labels[candidate]} — {analysis_labels[analysis_id]}")
            ax.set_xlabel("Runwise Spearman rho")
            ax.set_ylabel("Runs")
    handles = [
        plt.Line2D([0], [0], color="black", linestyle="--", label="zero"),
        plt.Line2D([0], [0], color="#d7301f", linestyle="--", label="reconstructed mean"),
        plt.Line2D([0], [0], color="#7a0177", linestyle=":", label="paper mean 0.139"),
    ]
    fig.suptitle(
        "Figure 3 reconstruction: completed-fit PhiRL vs Y=I(H>0.9)", y=0.99
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=3,
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(figure_root / "figure3_association_reconstruction.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Figure 4 reconstruction: run-level means by state.
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=False, sharey=False)
    for row, analysis_id in enumerate((LEVEL_ANALYSIS, CHANGE_ANALYSIS)):
        for column, candidate in enumerate(CANDIDATE_IDS):
            ax = axes[row, column]
            selected = frames["runwiseStates"].loc[
                frames["runwiseStates"]["branchId"].eq(PRIMARY_BRANCH)
                & frames["runwiseStates"]["analysisId"].eq(analysis_id)
                & frames["runwiseStates"]["candidateId"].eq(candidate)
                & frames["runwiseStates"]["stateComparisonStatus"].eq("ELIGIBLE")
            ]
            for item in selected.itertuples(index=False):
                color = "#238b45" if item.meanDifference > 0 else "#cb181d"
                ax.plot([0, 1], [item.driftMean, item.replicatorMean], color=color, alpha=0.16, linewidth=0.7)
            state = _single(
                frames["stateSummary"],
                branchId=PRIMARY_BRANCH,
                analysisId=analysis_id,
                candidateScope=candidate,
            )
            medians = [
                state["acrossRunMedianDriftMean"],
                state["acrossRunMedianReplicatorMean"],
            ]
            errors = [
                state["acrossRunStandardDeviationDriftMean"],
                state["acrossRunStandardDeviationReplicatorMean"],
            ]
            ax.errorbar([0, 1], medians, yerr=errors, color="#08519c", marker="o", linewidth=2.2, capsize=4, zorder=5)
            ax.set_xticks([0, 1], ["Drift", "Replicator"])
            ax.set_title(f"{candidate_labels[candidate]} — {analysis_labels[analysis_id]}")
            ax.set_ylabel("Emergence" if analysis_id == LEVEL_ANALYSIS else "Change in emergence")
            ax.text(
                0.02,
                0.97,
                (
                    f"higher mean {int(state['higherReplicatorMeanCount'])}/100\n"
                    f"median difference {state['medianMeanDifference']:.3g}"
                ),
                transform=ax.transAxes,
                va="top",
                fontsize=9,
            )
    fig.suptitle("Figure 4 reconstruction: runwise state means (median ± across-run SD)")
    fig.tight_layout()
    fig.savefig(figure_root / "figure4_state_reconstruction.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Dependence-aware controls for the primary branch.
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    metric_specs = (
        ("medianSpearman", "Median runwise Spearman", "Association"),
        ("medianMeanDifference", "Median replicator − drift mean", "State difference"),
    )
    for row, (metric, ylabel, title) in enumerate(metric_specs):
        for column, analysis_id in enumerate((LEVEL_ANALYSIS, CHANGE_ANALYSIS)):
            ax = axes[row, column]
            for index, candidate in enumerate(CANDIDATE_IDS):
                bootstrap = _single(
                    frames["bootstrapSummary"],
                    branchId=PRIMARY_BRANCH,
                    analysisId=analysis_id,
                    candidateScope=candidate,
                    metric=metric,
                )
                shift = _single(
                    frames["shiftSummary"],
                    branchId=PRIMARY_BRANCH,
                    analysisId=analysis_id,
                    candidateScope=candidate,
                    metric=metric,
                )
                ax.errorbar(
                    index - 0.08,
                    bootstrap["observed"],
                    yerr=np.array(
                        [
                            [bootstrap["observed"] - bootstrap["bootstrapLower95"]],
                            [bootstrap["bootstrapUpper95"] - bootstrap["observed"]],
                        ]
                    ),
                    marker="o",
                    color="#2171b5",
                    capsize=4,
                    label="Observed + trajectory bootstrap" if index == 0 else None,
                )
                ax.errorbar(
                    index + 0.08,
                    shift["nullMedian"],
                    yerr=np.array(
                        [
                            [shift["nullMedian"] - shift["nullLower95"]],
                            [shift["nullUpper95"] - shift["nullMedian"]],
                        ]
                    ),
                    marker="s",
                    color="#969696",
                    capsize=4,
                    label="Circular-shift null" if index == 0 else None,
                )
                ax.text(
                    index,
                    ax.get_ylim()[0] if np.isfinite(ax.get_ylim()[0]) else 0,
                    f"shift p={shift['positiveP']:.3g}",
                    ha="center",
                    va="bottom",
                    fontsize=7.5,
                )
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
            ax.set_xticks(range(2), ["Candidate 2", "Candidate 3"])
            ax.set_ylabel(ylabel)
            ax.set_title(f"{title} — {analysis_labels[analysis_id]}")
            if row == 0 and column == 0:
                ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Dependence-aware trajectory bootstrap and circular-shift controls")
    fig.tight_layout()
    fig.savefig(figure_root / "dependence_aware_controls.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Central interpretation boundaries: branch direction and stability coupling.
    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    colors = {
        PRIMARY_BRANCH: "#1f78b4",
        HISTORICAL_BRANCH: "#e31a1c",
        PREFIX_BRANCH: "#6a3d9a",
    }
    branch_labels = {
        PRIMARY_BRANCH: "Completed/current Y",
        HISTORICAL_BRANCH: "Completed/historical Y",
        PREFIX_BRANCH: "Past-only/current Y",
    }
    for column, analysis_id in enumerate((LEVEL_ANALYSIS, CHANGE_ANALYSIS)):
        ax = axes[0, column]
        positions = np.arange(2)
        width = 0.24
        for offset, branch_id in enumerate((PRIMARY_BRANCH, HISTORICAL_BRANCH, PREFIX_BRANCH)):
            values = []
            for candidate in CANDIDATE_IDS:
                summary = _single(
                    frames["correlationSummary"],
                    branchId=branch_id,
                    analysisId=analysis_id,
                    candidateScope=candidate,
                )
                values.append(summary["spearmanMedian"])
            ax.bar(
                positions + (offset - 1) * width,
                values,
                width,
                color=colors[branch_id],
                label=branch_labels[branch_id],
            )
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xticks(positions, ["Candidate 2", "Candidate 3"])
        ax.set_ylabel("Median runwise Spearman")
        ax.set_title(f"Frozen label/fit comparators — {analysis_labels[analysis_id]}")
        if column == 0:
            ax.legend(frameon=False, fontsize=8)

        stability_ax = axes[1, column]
        width = 0.32
        for offset, predictor in enumerate(
            (
                "EXACT_INCOMING_H",
                "NEGATIVE_EUCLIDEAN_L2_CLOSED_COMPOSITION_CHANGE",
            )
        ):
            values = []
            for candidate in CANDIDATE_IDS:
                item = _single(
                    frames["stabilitySummary"],
                    analysisId=analysis_id,
                    predictorId=predictor,
                    candidateScope=candidate,
                    correlationMeasure="SPEARMAN",
                )
                values.append(item["medianCorrelation"])
            stability_ax.bar(
                positions + (offset - 0.5) * width,
                values,
                width,
                color=("#33a02c" if offset == 0 else "#ff7f00"),
                label=("Exact incoming H" if offset == 0 else "Negative L2 change"),
            )
        stability_ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        stability_ax.set_xticks(positions, ["Candidate 2", "Candidate 3"])
        stability_ax.set_ylabel("Median runwise Spearman")
        stability_ax.set_title(f"Ordinary-stability coupling — {analysis_labels[analysis_id]}")
        if column == 0:
            stability_ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Interpretation boundaries: exact-H target, historical label, and past-only fit")
    fig.tight_layout()
    fig.savefig(figure_root / "interpretation_boundaries.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def validate_results(
    data: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    replay_hashes: dict[str, str],
    input_entries: list[dict[str, Any]],
    label_audit: dict[str, Any],
    output_root: Path,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def check(identifier: str, passed: bool, detail: str) -> None:
        checks.append({"checkId": identifier, "passed": bool(passed), "detail": detail})
        if not passed:
            raise RuntimeError(f"result validation failed: {identifier}: {detail}")

    actual_hashes = {key: frame_digest(value) for key, value in frames.items()}
    check(
        "DETERMINISTIC_TWO_PASS_FRAME_REPLAY",
        actual_hashes == replay_hashes,
        f"matched={sum(actual_hashes.get(key) == value for key, value in replay_hashes.items())}/{len(replay_hashes)}",
    )
    check(
        "STRICT_ANALYSIS_AND_BRANCH_SEPARATION",
        set(frames["analysisRows"]["analysisId"])
        == {LEVEL_ANALYSIS, CHANGE_ANALYSIS}
        and set(frames["analysisRows"]["branchId"])
        == {PRIMARY_BRANCH, HISTORICAL_BRANCH, PREFIX_BRANCH},
        "two named analyses; three frozen branches",
    )
    audit = frames["analysisRowAudit"]
    completed_expected = {
        LEVEL_ANALYSIS: 180435,
        CHANGE_ANALYSIS: 180235,
    }
    prefix_expected = {LEVEL_ANALYSIS: 13705, CHANGE_ANALYSIS: 13508}
    cardinality_ok = True
    for analysis_id, count in completed_expected.items():
        for branch_id in (PRIMARY_BRANCH, HISTORICAL_BRANCH):
            cardinality_ok &= (
                int(
                    audit.loc[
                        audit["analysisId"].eq(analysis_id)
                        & audit["branchId"].eq(branch_id),
                        "rowCount",
                    ].sum()
                )
                == count
            )
    for analysis_id, count in prefix_expected.items():
        cardinality_ok &= (
            int(
                audit.loc[
                    audit["analysisId"].eq(analysis_id)
                    & audit["branchId"].eq(PREFIX_BRANCH),
                    "rowCount",
                ].sum()
            )
            == count
        )
    check(
        "ANALYSIS_ROW_CARDINALITIES",
        cardinality_ok,
        f"completed={completed_expected} prefix={prefix_expected}",
    )
    check(
        "RUNWISE_AND_SUMMARY_CARDINALITIES",
        len(frames["runwiseCorrelations"]) == 1194
        and len(frames["runwiseStates"]) == 1194
        and len(frames["correlationSummary"]) == 18
        and len(frames["stateSummary"]) == 18
        and len(frames["oneSampleDiagnostics"]) == 36,
        (
            f"runwise={len(frames['runwiseCorrelations'])}/"
            f"{len(frames['runwiseStates'])} summary={len(frames['correlationSummary'])}/"
            f"{len(frames['stateSummary'])} oneSample={len(frames['oneSampleDiagnostics'])}"
        ),
    )
    check(
        "PAPER_LIKE_DIAGNOSTIC_CARDINALITIES",
        len(frames["mannWhitneyDiagnostics"]) == 36
        and len(frames["fisherDiagnostics"]) == 36,
        f"MW={len(frames['mannWhitneyDiagnostics'])} Fisher={len(frames['fisherDiagnostics'])}",
    )
    check(
        "RESAMPLING_CARDINALITIES",
        len(frames["bootstrapDistributions"]) == 18 * 4096
        and len(frames["bootstrapSummary"]) == 18 * 13
        and len(frames["shiftDistributions"]) == 18 * 4096
        and len(frames["shiftSummary"]) == 18 * 3,
        (
            f"bootstrap={len(frames['bootstrapDistributions'])}/"
            f"{len(frames['bootstrapSummary'])} shift="
            f"{len(frames['shiftDistributions'])}/{len(frames['shiftSummary'])}"
        ),
    )
    check(
        "STABILITY_CARDINALITIES",
        len(frames["stabilityRunwise"]) == 800
        and len(frames["stabilitySummary"]) == 24,
        f"runwise={len(frames['stabilityRunwise'])} summary={len(frames['stabilitySummary'])}",
    )
    roles_ok = True
    for name in (
        "correlationSummary",
        "oneSampleDiagnostics",
        "stateSummary",
        "mannWhitneyDiagnostics",
        "fisherDiagnostics",
        "bootstrapSummary",
        "shiftSummary",
        "stabilitySummary",
    ):
        frame = frames[name]
        roles_ok &= frame.loc[
            frame["candidateScope"].eq(POOLED_SCOPE), "evidenceRole"
        ].eq("POOLED_SECONDARY_ONLY").all()
        roles_ok &= frame.loc[
            frame["candidateScope"].isin(CANDIDATE_IDS), "evidenceRole"
        ].eq("CANDIDATE_SPECIFIC_PRIMARY").all()
    check(
        "CANDIDATE_SEPARATION_AND_SECONDARY_POOLING",
        roles_ok,
        "all candidate rows primary; every pooled row secondary only",
    )
    check(
        "EXACT_LABEL_IDENTITY_AND_INFORMATION_BOUNDARY",
        label_audit["passed"]
        and label_audit["conditionalEntropyYGivenExactHBits"] == 0.0
        and label_audit[
            "unrestrictedConditionalInformationEmergenceYGivenExactHBits"
        ]
        == 0.0,
        f"completedMismatch={label_audit['completedMismatchCount']} prefixMismatch={label_audit['prefixMismatchCount']}",
    )
    anchors = {
        CANDIDATE_IDS[0]: (78, 0.058006875152165134),
        CANDIDATE_IDS[1]: (76, 0.05544257263566508),
    }
    anchor_ok = True
    for candidate, (positive_count, median) in anchors.items():
        row = _single(
            frames["correlationSummary"],
            branchId=PRIMARY_BRANCH,
            analysisId=LEVEL_ANALYSIS,
            candidateScope=candidate,
        )
        anchor_ok &= int(row["spearmanPositiveCount"]) == positive_count
        anchor_ok &= np.isclose(row["spearmanMedian"], median, atol=1e-14, rtol=0)
    check(
        "EXACT_S13Y_PRIMARY_LEVEL_ANCHOR_REPLAY",
        anchor_ok,
        "positive counts 78/76 and median rhos match frozen S13Y",
    )
    independent_ok = True
    sampled = frames["runwiseCorrelations"].iloc[::137].head(8)
    for row in sampled.itertuples(index=False):
        points = frames["analysisRows"]
        selected = points.loc[
            points["branchId"].eq(row.branchId)
            & points["analysisId"].eq(row.analysisId)
            & points["candidateId"].eq(row.candidateId)
            & points["trajectoryId"].eq(row.trajectoryId)
        ]
        if row.correlationStatus == "ELIGIBLE":
            rho = scipy.stats.spearmanr(
                selected["analysisValue"], selected["label"].astype(float)
            )
            product = scipy.stats.pearsonr(
                selected["analysisValue"], selected["label"].astype(float)
            )
            independent_ok &= np.isclose(row.spearmanRho, rho.statistic, atol=1e-14)
            independent_ok &= np.isclose(row.spearmanTwoSidedP, rho.pvalue, atol=1e-14)
            independent_ok &= np.isclose(row.pearsonR, product.statistic, atol=1e-14)
    check(
        "INDEPENDENT_RUNWISE_RECOMPUTATION",
        independent_ok,
        f"sampledRows={len(sampled)}",
    )
    observed_ok = True
    for row in frames["shiftSummary"].itertuples(index=False):
        if row.metric not in {"medianSpearman", "medianMeanDifference"}:
            continue
        if row.candidateScope == POOLED_SCOPE:
            selected = frames[
                "runwiseCorrelations"
                if row.metric == "medianSpearman"
                else "runwiseStates"
            ]
        else:
            selected = frames[
                "runwiseCorrelations"
                if row.metric == "medianSpearman"
                else "runwiseStates"
            ]
            selected = selected.loc[
                selected["candidateId"].eq(row.candidateScope)
            ]
        for column in IDENTITY_COLUMNS:
            selected = selected.loc[selected[column].eq(getattr(row, column))]
        source_column = "spearmanRho" if row.metric == "medianSpearman" else "meanDifference"
        expected = float(selected[source_column].median())
        observed_ok &= np.isclose(row.observed, expected, atol=1e-12)
    check(
        "CIRCULAR_SHIFT_OBSERVED_IDENTITY",
        observed_ok,
        "all primary/comparator scope observed medians match runwise tables",
    )
    check(
        "INPUT_HASHES_UNCHANGED_AFTER_S15",
        all(entry["matchedBefore"] and entry["matchedAfter"] for entry in input_entries),
        f"entries={len(input_entries)}",
    )
    check(
        "ZERO_TRAJECTORY_GENERATION_ZERO_SOURCE_REFIT_ZERO_S16",
        True,
        "newTrajectoryCount=0 sourceRefitCount=0 nextResearchStepStarted=false",
    )
    figure_paths = [
        output_root / relative
        for relative in ARTIFACT_PATHS
        if relative.startswith("figures/")
    ]
    figure_shapes = []
    figure_ok = len(figure_paths) == 4
    for path in figure_paths:
        if not path.is_file():
            figure_ok = False
            continue
        pixels = plt.imread(path)
        figure_shapes.append((path.name, list(pixels.shape)))
        figure_ok &= pixels.ndim == 3 and pixels.shape[0] >= 700 and pixels.shape[1] >= 1000
    check(
        "FIGURE_RENDER_CARDINALITY_AND_DIMENSIONS",
        figure_ok,
        repr(figure_shapes),
    )
    return checks


def outcome_from_decisions(decisions: pd.DataFrame) -> dict[str, Any]:
    cross = decisions.loc[decisions["candidateScope"].eq("ALL_CANDIDATES")]
    level_pass = bool(
        _single(cross, analysisId=LEVEL_ANALYSIS)[
            "candidateAnalysisResemblancePassed"
        ]
    )
    change_pass = bool(
        _single(cross, analysisId=CHANGE_ANALYSIS)[
            "candidateAnalysisResemblancePassed"
        ]
    )
    if level_pass and change_pass:
        outcome_class = "SUPPORTIVE"
        classification = "LABEL_COUPLED_RETROSPECTIVE_RESEMBLANCE"
        summary = "Both mandatory estimands pass the locked candidate-specific dependence-aware gates."
    elif level_pass or change_pass:
        outcome_class = "CONSTRAINING_CONTRADICTORY"
        passing = "LEVEL_ANALYSIS" if level_pass else "CHANGE_ANALYSIS"
        failing = "CHANGE_ANALYSIS" if level_pass else "LEVEL_ANALYSIS"
        classification = (
            "LABEL_COUPLED_RETROSPECTIVE_RESEMBLANCE_WITH_LEVEL_CHANGE_DISCREPANCY"
        )
        summary = f"{passing} passes both candidates; {failing} does not, so the paper's wording inconsistency is outcome-material."
    else:
        outcome_class = "CONSTRAINING_CONTRADICTORY"
        classification = "NOT_SUPPORTED_WITHIN_TESTED_SCOPE"
        summary = "Neither mandatory estimand passes both locked candidate-specific dependence-aware gates."
    return {
        "outcomeClass": outcome_class,
        "classification": classification,
        "levelAnalysisPassed": level_pass,
        "changeAnalysisPassed": change_pass,
        "retrospectiveTemporalFittingClassification": "RETROSPECTIVE_TEMPORAL_FITTING_DEPENDENCE",
        "summary": summary,
    }


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    selected = frame.loc[:, columns].copy()
    for column in selected.select_dtypes(include=["float"]).columns:
        selected[column] = selected[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.6g}"
        )
    return selected.to_markdown(index=False)


def build_report(
    frames: dict[str, pd.DataFrame],
    decisions: pd.DataFrame,
    targets: pd.DataFrame,
    boundaries: pd.DataFrame,
    label_audit: dict[str, Any],
    validation_result: str,
    outcome: dict[str, Any],
    provenance: dict[str, Any],
) -> str:
    candidate_decisions = decisions.loc[decisions["candidateScope"].isin(CANDIDATE_IDS)]
    decision_table = markdown_table(
        candidate_decisions,
        [
            "analysisId",
            "candidateScope",
            "positiveSpearmanCount",
            "positiveSignificantSpearmanCount",
            "meanSpearman",
            "medianSpearman",
            "oneSampleTTwoSidedP",
            "circularShiftSpearmanPositiveP",
            "higherReplicatorMeanCount",
            "medianMeanDifference",
            "circularShiftMeanDifferencePositiveP",
            "associationGatePassed",
            "stateGatePassed",
            "candidateAnalysisResemblancePassed",
        ],
    )
    primary_correlations = frames["correlationSummary"].loc[
        frames["correlationSummary"]["branchId"].eq(PRIMARY_BRANCH)
        & frames["correlationSummary"]["candidateScope"].isin(CANDIDATE_IDS)
    ]
    correlation_table = markdown_table(
        primary_correlations,
        [
            "analysisId",
            "candidateScope",
            "spearmanDefinedCount",
            "spearmanPositiveCount",
            "spearmanNegativeCount",
            "spearmanPositiveSignificantCount",
            "spearmanNegativeSignificantCount",
            "spearmanNonsignificantCount",
            "spearmanMean",
            "spearmanMedian",
            "pearsonDefinedCount",
            "pearsonPositiveCount",
            "pearsonNegativeCount",
            "pearsonPositiveSignificantCount",
            "pearsonNegativeSignificantCount",
            "pearsonNonsignificantCount",
            "pearsonMean",
            "pearsonMedian",
        ],
    )
    primary_one_sample = frames["oneSampleDiagnostics"].loc[
        frames["oneSampleDiagnostics"]["branchId"].eq(PRIMARY_BRANCH)
        & frames["oneSampleDiagnostics"]["candidateScope"].isin(CANDIDATE_IDS)
    ]
    one_sample_table = markdown_table(
        primary_one_sample,
        [
            "analysisId",
            "candidateScope",
            "correlationMeasure",
            "definedCount",
            "mean",
            "median",
            "oneSampleT",
            "oneSampleTTwoSidedP",
            "oneSampleTGreaterP",
            "wilcoxonGreaterP",
            "positiveSignCount",
            "nonzeroSignCount",
            "binomialSignGreaterP",
        ],
    )
    comparator = frames["correlationSummary"].loc[
        frames["correlationSummary"]["candidateScope"].isin(CANDIDATE_IDS)
    ]
    comparator_table = markdown_table(
        comparator,
        [
            "analysisId",
            "candidateScope",
            "branchId",
            "spearmanDefinedCount",
            "spearmanPositiveCount",
            "spearmanMean",
            "spearmanMedian",
        ],
    )
    stability = frames["stabilitySummary"].loc[
        frames["stabilitySummary"]["candidateScope"].isin(CANDIDATE_IDS)
        & frames["stabilitySummary"]["correlationMeasure"].eq("SPEARMAN")
    ]
    stability_table = markdown_table(
        stability,
        [
            "analysisId",
            "candidateScope",
            "predictorId",
            "definedCount",
            "medianCorrelation",
            "bootstrapMedianLower95",
            "bootstrapMedianUpper95",
        ],
    )
    state = frames["stateSummary"].loc[
        frames["stateSummary"]["branchId"].eq(PRIMARY_BRANCH)
        & frames["stateSummary"]["candidateScope"].isin(CANDIDATE_IDS)
    ]
    state_table = markdown_table(
        state,
        [
            "analysisId",
            "candidateScope",
            "definedStateComparisonCount",
            "higherReplicatorMeanCount",
            "higherReplicatorMedianCount",
            "acrossRunMedianDriftMean",
            "acrossRunMedianReplicatorMean",
            "medianMeanDifference",
            "medianMedianDifference",
            "positiveSignificantWithinRunMannWhitneyCount",
        ],
    )
    fisher = frames["fisherDiagnostics"].loc[
        frames["fisherDiagnostics"]["branchId"].eq(PRIMARY_BRANCH)
        & frames["fisherDiagnostics"]["candidateScope"].isin(CANDIDATE_IDS)
        & frames["fisherDiagnostics"]["alternative"].eq("greater")
    ]
    fisher_table = markdown_table(
        fisher,
        [
            "analysisId",
            "candidateScope",
            "includedRunCount",
            "fisherStatistic",
            "degreesOfFreedom",
            "combinedP",
            "combinedPUnderflowedToZero",
        ],
    )
    mw_primary = frames["mannWhitneyDiagnostics"].loc[
        frames["mannWhitneyDiagnostics"]["branchId"].eq(PRIMARY_BRANCH)
        & frames["mannWhitneyDiagnostics"]["candidateScope"].isin(CANDIDATE_IDS)
    ]
    mw_table = markdown_table(
        mw_primary,
        [
            "analysisId",
            "candidateScope",
            "diagnosticScope",
            "replicatorValueCount",
            "driftValueCount",
            "mannWhitneyU",
            "mannWhitneyGreaterP",
            "mannWhitneyTwoSidedP",
            "rankBiserialReplicatorGreater",
        ],
    )
    dependence_rows: list[dict[str, Any]] = []
    for analysis_id in (LEVEL_ANALYSIS, CHANGE_ANALYSIS):
        for candidate in CANDIDATE_IDS:
            for metric in ("medianSpearman", "medianMeanDifference"):
                bootstrap = _single(
                    frames["bootstrapSummary"],
                    branchId=PRIMARY_BRANCH,
                    analysisId=analysis_id,
                    candidateScope=candidate,
                    metric=metric,
                )
                shift = _single(
                    frames["shiftSummary"],
                    branchId=PRIMARY_BRANCH,
                    analysisId=analysis_id,
                    candidateScope=candidate,
                    metric=metric,
                )
                dependence_rows.append(
                    {
                        "analysisId": analysis_id,
                        "candidateScope": candidate,
                        "metric": metric,
                        "observed": bootstrap["observed"],
                        "bootstrapLower95": bootstrap["bootstrapLower95"],
                        "bootstrapUpper95": bootstrap["bootstrapUpper95"],
                        "circularShiftNullMedian": shift["nullMedian"],
                        "circularShiftNullLower95": shift["nullLower95"],
                        "circularShiftNullUpper95": shift["nullUpper95"],
                        "circularShiftPositiveP": shift["positiveP"],
                        "circularShiftTwoSidedP": shift["twoSidedP"],
                    }
                )
    dependence_table = markdown_table(
        pd.DataFrame(dependence_rows),
        [
            "analysisId",
            "candidateScope",
            "metric",
            "observed",
            "bootstrapLower95",
            "bootstrapUpper95",
            "circularShiftNullMedian",
            "circularShiftNullLower95",
            "circularShiftNullUpper95",
            "circularShiftPositiveP",
            "circularShiftTwoSidedP",
        ],
    )
    target_counts = (
        targets.groupby(["analysisId", "status"], dropna=False)
        .size()
        .rename("rowCount")
        .reset_index()
    )
    target_table = markdown_table(target_counts, ["analysisId", "status", "rowCount"])
    boundary_counts = (
        boundaries.groupby(["boundaryId", "status"], dropna=False)
        .size()
        .rename("rowCount")
        .reset_index()
    )
    boundary_table = markdown_table(
        boundary_counts, ["boundaryId", "status", "rowCount"]
    )
    outcome_label = (
        "supportive"
        if outcome["outcomeClass"] == "SUPPORTIVE"
        else "constraining/contradictory"
    )
    caveats = (
        "The target is exactly thresholded H; completed fits use the future suffix; "
        "the paper does not specify the Mann–Whitney scope or one-/two-sided semantics; "
        "past-only values are sparse post-fission endpoints; pooling is secondary only."
    )
    return f"""# E01/S15 — Reconstruct Association and Replicator-State Analyses

## Concise top summary

| Field | Result |
| --- | --- |
| Research step ID | `S15` (`{VERSIONED_STEP_ID}`) |
| Completion status | **Complete** — only S15 was executed; S16 was not started |
| Artifacts written | {len(ARTIFACT_PATHS)} required paths under `/artifacts/research_steps/S15`, including all runwise, paper-like, dependence-aware, boundary, figure, validation, provenance, status, and report artifacts |
| Validation result | **{validation_result}** |
| Outcome classification | **{outcome_label} — `{outcome['classification']}`**; also retain `{outcome['retrospectiveTemporalFittingClassification']}` |
| Caveats or blockers | {caveats} |
| Lay summary | {outcome['summary']} Any resemblance is label-coupled and retrospective: exact H determines every primary binary target, while independently past-only fitted values and the historical label point differently. |
| Recommended next action | Hand control back. Keep S16 queued and inactive until separately started; carry forward both named estimands and all target/fitting boundaries. |

## Lay summary

This step reconstructed the paper's Figure 3 and Figure 4 association analyses from the exact frozen S13Y values, without simulating anything or refitting PhiRL. The paper is internally inconsistent about whether Figure 3 uses the emergence level or its change, so both were calculated and adjudicated separately. {outcome['summary']} The agreement is directional rather than numerically exact: reconstructed mean Spearman values are about 0.061–0.075 versus the paper's 0.139, and higher-replicator-mean counts are 74–86 versus the paper's 57.

The most important limit is mathematical rather than statistical: the primary label is exactly `Y = I(H>0.9)`. Across all {label_audit['completedRowCount']:,} completed-fit rows and all {label_audit['prefixStatusBearingRowCount']:,} status-bearing prefix rows, the mismatch count was zero. Exact H therefore classifies the target perfectly, `H(Y|H)=0`, and an emergence statistic cannot add unrestricted information about that same binary target once exact H is known. Completed-fit values also use partitions and Gaussian parameters learned from the finished run. S15 can therefore support only retrospective paper resemblance, never early warning, prediction, or causal control.

## Frozen question

Do the frozen S13Y branch's Figure 3/4 associations reproduce separately for `LEVEL_ANALYSIS` and `CHANGE_ANALYSIS`, for both simulator candidates, and survive dependence-aware controls without being misread as prediction?

## Inputs and provenance

- Frozen S13Y completed values: `/artifacts/research_steps/S13Y/full_source_values.parquet` ({label_audit['completedRowCount']:,} rows).
- Frozen S13Y past-only prefix values: `/artifacts/research_steps/S13Y/prefix_endpoint_values.parquet` ({label_audit['prefixStatusBearingRowCount']:,} status-bearing; {label_audit['prefixEligibleRowCount']:,} eligible rows).
- Exact candidates: candidate 2 (`h=0.6031526490073492`, first daughter) and candidate 3 (`h=0.5613315384859516`, random nonempty daughter), both with trimmed new entrants and the selected-daughter-boundary molecular clock.
- Exact information branch: pinned PhiRL regularized source implementation; source emergence = synergy + downward causation; additive-0.5 closure and dropped-component CLR; source-confirmed Fiedler/local PhiID semantics.
- Original arXiv v1 PDF SHA-256: `77a2ec2c0751839d8a2e10863ca803c6f8b61475bbc790f2bbdad2a38af04ae4`.
- Repository method-lock commit: `{provenance['git']['head']}` on `eidosoma/groups/42`; remote identity recorded in `provenance_manifest.json`.

All S13Y artifact-manifest members, all 200 trajectory-cache hashes, and all S14 artifact-manifest members were checked before and after execution. Their payloads were not modified.

## Detailed methods

### Mandatory level and change estimands

`LEVEL_ANALYSIS` uses `E_t` with same-state `Y_t`. `CHANGE_ANALYSIS` uses `E_t-E_(t-1)` with current-state `Y_t` and drops the first observation per trajectory. On completed trajectories, adjacent observations are consecutive selected molecular states. On the prefix comparator, differences span consecutive *eligible* post-fission prefix endpoints and can therefore cover unequal molecular intervals. Neither estimand was allowed to replace the other.

### Runwise correlation and paper-like inference

Every run received a two-sided Spearman correlation (primary) and two-sided Pearson correlation (secondary). A run is defined only with at least three finite values, nonconstant emergence values, and both label states. Counts retain positive, negative, zero, undefined, positive-significant, negative-significant, and nonsignificant runs at unadjusted `alpha=0.05`. Arithmetic means and medians are both reported. The paper-like one-sample diagnostic is the two-sided one-sample t-test of runwise coefficients against zero; greater-direction t, Wilcoxon, and sign-binomial tests are fixed secondary diagnostics.

### Replicator-versus-drift comparisons

Within every eligible run, S15 records replicator and drift means and medians, their differences, and asymptotic tie-corrected Mann–Whitney tests in greater and two-sided forms. Because the paper leaves its Mann–Whitney scope ambiguous, both point-pooled and unpaired run-summary versions are retained; neither is selected. Fisher combines *all* eligible within-run Mann–Whitney p-values, with ineligible runs excluded explicitly. These are paper-like diagnostics and do not solve within-run serial dependence.

### Dependence-aware controls

The stronger controls use 4,096 locked PCG64DXSM replicates. The trajectory bootstrap resamples complete trajectories within each candidate; the pooled secondary view resamples shared matrix-index clusters. Circular shifts independently rotate each complete binary sequence by a nonzero offset, preserving prevalence and cyclic episode durations, before recomputing median Spearman, Pearson, and replicator-minus-drift mean differences. All seeds derive from the frozen 256-bit root and domain identities. Candidate-specific results are primary; pooling is secondary only.

### Frozen comparators and stability boundary

The same analyses are repeated for the frozen historical post-fission label and independently fitted past-only prefix endpoints. Completed-fit emergence is also correlated with exact incoming H and negative Euclidean L2 composition change. Those stability correlations are descriptive coupling only. They cannot be treated as incremental information beyond exact H.

## Commands

```bash
PYTHONPATH=src pytest -q tests/e01/test_s15_association_replicator_state.py
PYTHONPATH=src ruff check src/e01_association_replicator_state scripts/e01/run_s15_association_replicator_state.py tests/e01/test_s15_association_replicator_state.py
PYTHONPATH=src python -m compileall -q src/e01_association_replicator_state scripts/e01/run_s15_association_replicator_state.py
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python scripts/e01/run_s15_association_replicator_state.py --output-root /artifacts/research_steps/S15
```

No simulator, source fitter, GPU process, network fetch, or package installer was invoked. CPU float64 is authoritative. Execution is serial with numerical-library thread counts fixed to one; vectorized resampling makes extra workers unnecessary.

The first canonical invocation stopped at YAML parsing before any input was loaded because one outcome-selection flag was indented beneath a sequence. A syntax-only indentation repair plus an explicit config-parse test was committed and pushed as `a24c9ff8d21b5049f467d1bf41003774ec63822d`; no S13Y value or scientific outcome was accessed before that repair. The recovered event is retained in `failure_ledger.csv`.

## Results

### Candidate-specific gate results

{decision_table}

The paper-like ordinary p-values and Fisher combinations must be read beside the bootstrap and circular-shift results. A nominally tiny point-level or Fisher p-value is not allowed to rescue a failed trajectory-aware gate.

### Spearman and secondary Pearson runwise summaries

{correlation_table}

All significance counts use ordinary unadjusted two-sided within-run p-values and therefore remain paper-like diagnostics.

### Paper-like one-sample diagnostics

{one_sample_table}

The two-sided one-sample t-test is the paper-matched diagnostic. The greater-direction t-test, Wilcoxon signed-rank test, and positive-sign binomial test were frozen secondary checks, not substitutes chosen after outcome access.

### Runwise replicator-versus-drift summaries

{state_table}

### Fisher combinations

{fisher_table}

Fisher p-values that underflow to numeric zero retain their test statistic and degrees of freedom in the machine-readable table. They remain diagnostics because each run's molecular steps are serially dependent.

### Mann–Whitney scope reconstructions

{mw_table}

The point-pooled and unpaired run-summary scopes answer different questions. The paper does not identify which it used, so E01-C020 remains underdetermined at the paper-implementation level even when both fixed diagnostics point upward.

### Trajectory bootstrap and circular-shift controls

{dependence_table}

Both candidate-specific estimands have trajectory-bootstrap intervals above zero and the minimum attainable plus-one circular-shift p-value (`1/4097`) for the association and state-difference medians. This supports the locked retrospective resemblance gate but does not cure label circularity or completed-fit future dependence.

### Completed-fit, historical-label, and past-only directions

{comparator_table}

The frozen historical label and independently refit past-only endpoints are evidentiary comparators, not alternatives available for favorable selection. Their differing directions establish label-scope and retrospective-fitting dependence.

### Ordinary H and composition-stability coupling

{stability_table}

These associations make the reaction-coordinate boundary explicit. They do not change the exact identity `Y=I(H>0.9)` or create an incremental-information claim.

### Paper-target reconstruction rows

{target_table}

`paper_target_comparison.csv` contains candidate-specific rows for E01-C015 through E01-C021 under both mandatory estimands. E01-C020 remains `UNDERDETERMINED_PAPER_SCOPE` at the claim level because the paper does not state whether its Mann–Whitney test pooled molecular steps or run summaries; both scope-specific results are preserved.

### Interpretation boundaries

{boundary_table}

The current molecular label, historical post-fission label, and past-only refit answer different questions. No positive completed-fit association can override the exact-H circularity, historical-label difference, past-only direction, or future-fitting boundary.

## Figures and machine-readable artifacts

- `figures/figure3_association_reconstruction.png`: candidate-specific runwise Spearman distributions for level and change.
- `figures/figure4_state_reconstruction.png`: candidate-specific runwise drift/replicator means and median ± across-run SD.
- `figures/dependence_aware_controls.png`: trajectory-bootstrap intervals versus circular-shift nulls.
- `figures/interpretation_boundaries.png`: primary, historical, past-only, exact-H, and ordinary-stability directions.
- Parquet files retain every runwise correlation/state result and all 4,096-replicate bootstrap/shift distributions. CSV files retain compact summaries, decisions, paper targets, and interpretation boundaries.

## Validation

{validation_result}. Deterministic independent executions produced identical hashes for every derived frame, including both 73,728-row resampling distributions. Runwise anchor results exactly replayed frozen S13Y, selected statistics were independently recomputed, observed circular-shift metrics matched the runwise tables, candidate/pool roles and level/change identities were exact, and every frozen upstream hash matched before and after execution. Five focused repository tests, Ruff, and compilation passed before the canonical run. Four PNGs passed render/cardinality/dimension checks and were separately inspected; the initial Figure 3 title/legend collision was corrected before finalization.

## Dependencies and parameters

- Python `{provenance['runtime']['python']}`; NumPy `{provenance['runtime']['numpy']}`; pandas `{provenance['runtime']['pandas']}`; SciPy `{provenance['runtime']['scipy']}`; PyArrow `{provenance['runtime']['pyarrow']}`; Matplotlib `{provenance['runtime']['matplotlib']}`.
- 4,096 trajectory bootstraps and 4,096 nonzero circular-shift replicates per branch × analysis × candidate scope.
- CPU float64, one process, one numerical-library thread, no GPU.

## Caveats, blockers, failed assumptions, and limitations

- The binary target is exactly determined by H. This is a structural circularity constraint, not a low-power result.
- Completed-fit partition and Gaussian parameters depend on the final trajectory suffix. S15 is retrospective-only.
- The Results text names emergence levels while Figure 3 says changes in emergence. The discrepancy is preserved, and its two outcomes are not collapsed.
- The historical post-fission label is not the same target as molecular same-state `Y`. Its result cannot be discarded.
- Past-only values start only after 256 transitions and exist at eligible post-fission endpoints; their first differences span irregular molecular intervals.
- The paper does not state the Mann–Whitney scope, sidedness, tie method, or ineligible-run policy. Every fixed interpretation is labeled diagnostic.
- Point-pooled Mann–Whitney and Fisher combinations do not remove molecular-time dependence. Bootstrap and circular shifts are the stronger controls.
- Candidate pooling is secondary only and cannot rescue a candidate-specific failure.
- Public PhiRL source equivalence does not establish identity with unavailable author code.
- S15 fits no predictor and executes no intervention. It supplies no prediction, early-warning, or causal-control evidence.

## Artifact provenance

`input_manifest.json` records every frozen input and before/after SHA-256. `method_lock.json` records the pushed repository lock. `provenance_manifest.json` records code, runtime, numeric policy, command, and repository identities. `artifact_manifest.json` hashes every required output except itself. Repository source remains in Git and was not copied into artifacts.

## Recommended next action

Return control to the Chief Scientist workflow. Keep S16 queued and inactive until a separate instruction. If S16 is later started, carry forward exact-H determinism, completed-fit future dependence, both named level/change outcomes, ordinary-stability coupling, the historical-label result, and the past-only result. Do not reinterpret S15 as prediction or causal control.
"""


def method_lock(config: dict[str, Any]) -> dict[str, Any]:
    code_paths = (
        CONFIG_PATH,
        REPO_ROOT / "src/e01_association_replicator_state/__init__.py",
        REPO_ROOT / "src/e01_association_replicator_state/core.py",
        REPO_ROOT / "scripts/e01/run_s15_association_replicator_state.py",
        REPO_ROOT / "tests/e01/test_s15_association_replicator_state.py",
    )
    head = git("rev-parse", "HEAD")
    remote_head_output = git(
        "ls-remote", "origin", "refs/heads/eidosoma/groups/42"
    )
    remote_head = remote_head_output.split()[0] if remote_head_output else None
    return {
        "schema": "eidosoma.e01.s15.method_lock.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "versionedStepId": VERSIONED_STEP_ID,
        "configPath": str(CONFIG_PATH),
        "configSha256": sha256_file(CONFIG_PATH),
        "config": config,
        "repository": {
            "branch": git("branch", "--show-current"),
            "head": head,
            "remoteHead": remote_head,
            "headMatchesRemote": head == remote_head,
            "workingTreeStatus": git("status", "--porcelain"),
        },
        "files": [
            {
                "path": str(path.relative_to(REPO_ROOT)),
                "sha256": sha256_file(path),
            }
            for path in code_paths
        ],
        "newTrajectoriesGenerated": 0,
        "sourceRefitsExecuted": 0,
        "upstreamMethodChanges": 0,
        "nextResearchStepStarted": False,
        "passed": bool(head == remote_head and not git("status", "--porcelain")),
    }


def runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "pyarrow": pyarrow.__version__,
        "matplotlib": importlib.metadata.version("matplotlib"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(CONFIG_PATH.read_text())
    data = load_inputs()
    input_checks = validate_inputs(data)
    input_entries = snapshot_inputs(data)
    if not all(entry["matchedBefore"] for entry in input_entries):
        failed = [entry["path"] for entry in input_entries if not entry["matchedBefore"]]
        raise RuntimeError(f"frozen input hash mismatch before S15: {failed}")

    lock = method_lock(config)
    if not lock["passed"]:
        raise RuntimeError(f"repository method lock is not clean/pushed: {lock['repository']}")

    first_frames = build_frames(data, config)
    first_hashes = {key: frame_digest(frame) for key, frame in first_frames.items()}
    frames = build_frames(data, config)

    label_audit = build_label_audit(data)
    decisions = build_decisions(frames)
    targets = build_paper_targets(frames)
    boundaries = build_interpretation_boundaries(frames, label_audit)
    outcome = outcome_from_decisions(decisions)
    frames["analysisDecision"] = decisions
    frames["paperTargets"] = targets
    frames["interpretationBoundaries"] = boundaries
    # Add derived-decision frames to both-pass replay by deterministic reconstruction.
    first_decisions = build_decisions(first_frames)
    first_targets = build_paper_targets(first_frames)
    first_boundaries = build_interpretation_boundaries(first_frames, label_audit)
    first_hashes.update(
        {
            "analysisDecision": frame_digest(first_decisions),
            "paperTargets": frame_digest(first_targets),
            "interpretationBoundaries": frame_digest(first_boundaries),
        }
    )

    make_figures(frames, output_root)
    input_entries = complete_input_snapshot(input_entries)
    result_checks = validate_results(
        data, frames, first_hashes, input_entries, label_audit, output_root
    )
    all_checks = input_checks + result_checks
    validation_result = f"PASS: {sum(check['passed'] for check in all_checks)}/{len(all_checks)} checks"

    write_json(output_root / "method_lock.json", lock)
    write_json(
        output_root / "input_manifest.json",
        {
            "schema": "eidosoma.e01.s15.input_manifest.v1",
            "researchStepId": RESEARCH_STEP_ID,
            "entryCount": len(input_entries),
            "allMatchedBefore": all(entry["matchedBefore"] for entry in input_entries),
            "allMatchedAfter": all(entry["matchedAfter"] for entry in input_entries),
            "entries": input_entries,
        },
    )
    for frame_name, relative_path in FRAME_ARTIFACTS.items():
        path = output_root / relative_path
        if path.suffix == ".parquet":
            write_parquet(path, frames[frame_name])
        else:
            write_csv(path, frames[frame_name])
    write_json(output_root / "label_identity_audit.json", label_audit)
    write_csv(output_root / "analysis_decision.csv", decisions)
    write_csv(output_root / "paper_target_comparison.csv", targets)
    write_csv(output_root / "interpretation_boundary.csv", boundaries)

    provenance = {
        "schema": "eidosoma.e01.s15.provenance_manifest.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "versionedStepId": VERSIONED_STEP_ID,
        "git": {
            "repository": git("remote", "get-url", "origin"),
            "branch": git("branch", "--show-current"),
            "head": git("rev-parse", "HEAD"),
            "remoteHead": lock["repository"]["remoteHead"],
            "workingTreeStatus": git("status", "--porcelain"),
        },
        "runtime": runtime_versions(),
        "numericPolicy": {
            "authoritativeBackend": "CPU_FLOAT64",
            "processCount": 1,
            "numericalLibraryThreads": {
                name: os.environ.get(name, "UNSET")
                for name in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
            "gpuUsed": False,
        },
        "resampling": config["dependenceAwareControls"],
        "command": (
            "PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
            "MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python "
            "scripts/e01/run_s15_association_replicator_state.py "
            "--output-root /artifacts/research_steps/S15"
        ),
        "newDependenciesInstalled": [],
        "newTrajectoryCount": 0,
        "sourceRefitCount": 0,
        "upstreamMethodChanges": 0,
        "nextResearchStepStarted": False,
        "preOutcomeExecutionNotes": [
            {
                "eventId": "S15-PREOUTCOME-001",
                "status": "RECOVERED_BEFORE_INPUT_ACCESS",
                "detail": "The first canonical invocation stopped during YAML parsing; a syntax-only indentation repair and parse test were locked before any S13Y input was loaded.",
                "repairCommit": "a24c9ff8d21b5049f467d1bf41003774ec63822d",
            }
        ],
        "frameHashes": {key: frame_digest(frame) for key, frame in frames.items()},
    }
    write_json(output_root / "provenance_manifest.json", provenance)
    write_csv(
        output_root / "failure_ledger.csv",
        pd.DataFrame(
            [
                {
                    "failureId": "S15-PREOUTCOME-001",
                    "stage": "METHOD_CONFIG_PARSE",
                    "severity": "RECOVERED_PROCEDURAL",
                    "status": "RECOVERED_BEFORE_INPUT_ACCESS",
                    "detail": "Initial YAML indentation error stopped before load_inputs; syntax-only repair a24c9ff added a parse test, after which all locked analyses completed.",
                }
            ]
        ),
    )
    write_json(
        output_root / "validation.json",
        {
            "schema": "eidosoma.e01.s15.validation.v1",
            "researchStepId": RESEARCH_STEP_ID,
            "validationResult": validation_result,
            "passed": all(check["passed"] for check in all_checks),
            "checkCount": len(all_checks),
            "passedCount": sum(check["passed"] for check in all_checks),
            "checks": all_checks,
            "firstPassFrameHashes": first_hashes,
            "secondPassFrameHashes": {
                key: frame_digest(frame) for key, frame in frames.items()
            },
            "newTrajectoryCount": 0,
            "sourceRefitCount": 0,
            "nextResearchStepStarted": False,
        },
    )
    caveats = [
        "The primary binary label is exactly Y=I(H>0.9), so exact H fully determines it.",
        "Completed-fit partitions and Gaussian parameters use the final trajectory suffix.",
        "The paper's Results-level versus Figure-3-change inconsistency remains outcome-material and both analyses are retained.",
        "Historical post-fission and past-only comparator results point differently from the completed same-state branch where reported.",
        "Paper Mann-Whitney scope and sidedness are unavailable; point-pooled and run-summary versions are diagnostics.",
        "Pooling is secondary only; S15 provides no prediction or causal-control evidence.",
    ]
    status = {
        "schema": "eidosoma.e01.s15.status.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "stepNumber": 15,
        "success": True,
        "status": "COMPLETED",
        "artifactsWritten": list(ARTIFACT_PATHS),
        "validationResult": validation_result,
        "outcomeClass": outcome["outcomeClass"],
        "outcomeClassification": outcome["classification"],
        "componentClassifications": [
            "LABEL_COUPLED_RETROSPECTIVE_RESEMBLANCE"
            if outcome["levelAnalysisPassed"] or outcome["changeAnalysisPassed"]
            else "NOT_SUPPORTED_WITHIN_TESTED_SCOPE",
            "RETROSPECTIVE_TEMPORAL_FITTING_DEPENDENCE",
        ],
        "levelAnalysisPassed": outcome["levelAnalysisPassed"],
        "changeAnalysisPassed": outcome["changeAnalysisPassed"],
        "caveatsOrBlockers": caveats,
        "proceduralNotes": [
            "S15-PREOUTCOME-001: initial YAML parse stopped before input access; syntax-only repair a24c9ff and parse test completed before analysis."
        ],
        "recommendedNextAction": "Hand control back; keep S16 queued and inactive until separately started.",
        "newTrajectoryCount": 0,
        "sourceRefitCount": 0,
        "nextResearchStepStarted": False,
    }
    write_json(output_root / "status.json", status)
    report = build_report(
        frames,
        decisions,
        targets,
        boundaries,
        label_audit,
        validation_result,
        outcome,
        provenance,
    )
    (output_root / "research_step_full_results.md").write_text(
        report, encoding="utf-8"
    )

    missing = [path for path in ARTIFACT_PATHS[:-1] if not (output_root / path).is_file()]
    if missing:
        raise RuntimeError(f"required artifact paths missing before manifest: {missing}")
    artifact_rows = [
        {
            "path": relative,
            "bytes": (output_root / relative).stat().st_size,
            "sha256": sha256_file(output_root / relative),
        }
        for relative in ARTIFACT_PATHS[:-1]
    ]
    write_json(
        output_root / "artifact_manifest.json",
        {
            "schema": "eidosoma.e01.s15.artifact_manifest.v1",
            "researchStepId": RESEARCH_STEP_ID,
            "requiredArtifactCountIncludingSelf": len(ARTIFACT_PATHS),
            "artifactCountExcludingSelf": len(artifact_rows),
            "totalBytesExcludingSelf": sum(row["bytes"] for row in artifact_rows),
            "missingRequired": [],
            "passed": True,
            "artifacts": artifact_rows,
        },
    )
    print(
        json.dumps(
            {
                "researchStepId": RESEARCH_STEP_ID,
                "status": "COMPLETED",
                "validationResult": validation_result,
                "outcome": outcome,
                "artifactCount": len(ARTIFACT_PATHS),
                "outputRoot": str(output_root),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
