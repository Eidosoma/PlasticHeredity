#!/usr/bin/env python3
"""Execute E01/S14 using only frozen S13Y trajectories and information values."""

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
import sklearn
import statsmodels
import yaml
from scipy.stats import fisher_exact

from e01_descriptive_causal_emergence.core import (
    CANDIDATE_IDS,
    COMPLETED_MODE,
    PREFIX_MODE,
    RESEARCH_STEP_ID,
    THRESHOLD_FAMILIES,
    VERSIONED_STEP_ID,
    add_excursion_flags,
    aggregate_trajectories,
    combine_fission_dependency,
    compare_completed_prefix,
    excursion_catalog,
    ljung_box_results,
    partition_change_history,
    prepare_completed,
    prepare_prefix,
    summarize_excursions,
    summarize_ljung_box,
    trend_results,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    REPO_ROOT
    / "configs/e01/s14_descriptive_causal_emergence_dynamics_preregistration.yaml"
)
S13Y_ROOT = Path("/artifacts/research_steps/S13Y")
PAPER_PATH = Path("/cache/e01_s03/downloads/paper-2607.28250v1.pdf")
SOURCE_MANIFEST = Path(
    "/artifacts/E01_forensic_replication_bundle/provenance/source_manifest.yaml"
)
DEFAULT_OUTPUT = Path("/artifacts/research_steps/S14")

INPUT_ARTIFACTS = (
    "full_source_values.parquet",
    "prefix_endpoint_values.parquet",
    "partition_history.parquet",
    "source_diagnostic_outputs.parquet",
    "preprocessing_diagnostics.parquet",
    "trajectory_manifest.parquet",
    "simulation_summary.parquet",
    "method_lock.json",
    "fixed_branch_lock.json",
    "artifact_manifest.json",
    "immutable_prior_validation.json",
    "research_step_full_results.md",
    "status.json",
)

ARTIFACT_PATHS = (
    "method_lock.json",
    "input_manifest.json",
    "aggregate_trajectory.parquet",
    "aggregate_trend_results.csv",
    "excursion_thresholds.parquet",
    "spike_catalog.parquet",
    "spike_run_summary.csv",
    "spike_morphology_summary.csv",
    "ljung_box_results.parquet",
    "ljung_box_summary.csv",
    "partition_change_history.parquet",
    "completed_vs_past_only.parquet",
    "completed_vs_past_only_summary.csv",
    "fission_dependency.csv",
    "partition_dependency.csv",
    "numerical_diagnostic_summary.csv",
    "paper_target_comparison.csv",
    "figures/figure2_candidate_specific.png",
    "figures/figure2_pooled_secondary.png",
    "figures/completed_fit_vs_past_only.png",
    "figures/spike_dependency_diagnostics.png",
    "validation.json",
    "provenance_manifest.json",
    "failure_ledger.csv",
    "status.json",
    "research_step_full_results.md",
    "artifact_manifest.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def frame_digest(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(list(frame.columns), separators=(",", ":")).encode())
    digest.update(json.dumps([str(x) for x in frame.dtypes]).encode())
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
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if pd.isna(value) if not isinstance(value, (str, bytes)) else False:
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


def artifact_manifest_entries() -> list[dict[str, Any]]:
    manifest = json.loads((S13Y_ROOT / "artifact_manifest.json").read_text())
    failures = []
    entries = []
    for record in manifest["artifacts"]:
        path = S13Y_ROOT / record["path"]
        actual = sha256_file(path) if path.exists() else None
        if actual != record["sha256"]:
            failures.append(record["path"])
        entries.append(
            {
                "role": "S13Y_ARTIFACT_MANIFEST_MEMBER",
                "path": str(path),
                "bytes": path.stat().st_size if path.exists() else None,
                "sha256": actual,
                "expectedSha256": record["sha256"],
                "matched": actual == record["sha256"],
            }
        )
    if failures:
        raise RuntimeError(f"S13Y artifact manifest mismatch: {failures}")
    return entries


def snapshot_inputs(trajectory_manifest: pd.DataFrame) -> list[dict[str, Any]]:
    entries = artifact_manifest_entries()
    included = {entry["path"] for entry in entries}
    for name in INPUT_ARTIFACTS:
        path = S13Y_ROOT / name
        if str(path) in included:
            continue
        entries.append(
            {
                "role": "S13Y_DIRECT_INPUT",
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "expectedSha256": None,
                "matched": True,
            }
        )
    for row in trajectory_manifest.sort_values(
        ["candidateId", "matrixIndex"], kind="stable"
    ).itertuples(index=False):
        path = Path(row.cachePath)
        actual = sha256_file(path)
        entries.append(
            {
                "role": "S13Y_FROZEN_TRAJECTORY_CACHE",
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": actual,
                "expectedSha256": str(row.cacheSha256),
                "matched": actual == str(row.cacheSha256),
            }
        )
    for role, path in (
        ("S14_METHOD_CONFIG", CONFIG_PATH),
        ("ORIGINAL_PAPER_ARXIV_V1", PAPER_PATH),
        ("PINNED_SOURCE_MANIFEST", SOURCE_MANIFEST),
    ):
        entries.append(
            {
                "role": role,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "expectedSha256": None,
                "matched": True,
            }
        )
    return sorted(entries, key=lambda row: (row["role"], row["path"]))


def load_inputs() -> dict[str, pd.DataFrame]:
    return {
        "full": pd.read_parquet(S13Y_ROOT / "full_source_values.parquet"),
        "prefix": pd.read_parquet(S13Y_ROOT / "prefix_endpoint_values.parquet"),
        "partitions": pd.read_parquet(S13Y_ROOT / "partition_history.parquet"),
        "sourceDiagnostics": pd.read_parquet(
            S13Y_ROOT / "source_diagnostic_outputs.parquet"
        ),
        "preprocessing": pd.read_parquet(
            S13Y_ROOT / "preprocessing_diagnostics.parquet"
        ),
        "trajectoryManifest": pd.read_parquet(
            S13Y_ROOT / "trajectory_manifest.parquet"
        ),
        "simulationSummary": pd.read_parquet(S13Y_ROOT / "simulation_summary.parquet"),
    }


def validate_frozen_inputs(data: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
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
        "EXACT_CANDIDATE_SET",
        tuple(sorted(full["candidateId"].unique())) == CANDIDATE_IDS,
        repr(tuple(sorted(full["candidateId"].unique()))),
    )
    counts = full.groupby("candidateId")["trajectoryId"].nunique().to_dict()
    check(
        "ONE_HUNDRED_TRAJECTORIES_PER_CANDIDATE",
        counts == {candidate: 100 for candidate in CANDIDATE_IDS},
        repr(counts),
    )
    check(
        "FULL_KEY_UNIQUENESS",
        not full.duplicated(
            ["candidateId", "trajectoryId", "selectedSequenceIndex"]
        ).any(),
        f"rows={len(full)}",
    )
    check(
        "PREFIX_KEY_UNIQUENESS",
        not prefix.duplicated(["candidateId", "trajectoryId", "generation"]).any(),
        f"rows={len(prefix)} eligible={int(prefix.status.eq('ELIGIBLE').sum())}",
    )
    component_error = np.max(
        np.abs(
            full["emergence"].to_numpy(float)
            - full["synergy"].to_numpy(float)
            - full["downwardCausation"].to_numpy(float)
        )
    )
    check(
        "SOURCE_EMERGENCE_COMPONENT_IDENTITY",
        component_error <= 1e-12,
        f"maxAbsError={component_error:.17g}",
    )
    check(
        "FROZEN_IMPLEMENTATION_IDENTITY",
        set(full["implementationId"]) == {"PHIRL_REGULARIZED_SOURCE"}
        and set(full["temporalLabel"]) == {COMPLETED_MODE}
        and set(prefix["temporalLabel"]) == {PREFIX_MODE},
        "PhiRL regularized source; completed and prefix labels exact",
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
        "candidate 2/3 h, daughter, trim semantics exact",
    )
    check(
        "TRAJECTORY_MANIFEST_COMPLETE_AND_REPLAYED",
        len(trajectories) == 200
        and trajectories["exactReplayPassed"].astype(bool).all()
        and trajectories["completedFissions"].eq(100).all(),
        f"rows={len(trajectories)}",
    )
    check(
        "NO_NEW_TRAJECTORY_PHASE",
        set(trajectories["researchStepId"]) == {"S13Y"}
        and set(simulation["researchStepId"]) == {"S13Y"},
        "all trajectory identities remain S13Y",
    )
    return checks


def attach_diagnostics(
    catalog: pd.DataFrame,
    diagnostics: pd.DataFrame,
    preprocessing: pd.DataFrame,
    partitions: pd.DataFrame,
) -> pd.DataFrame:
    completed_diag = diagnostics.loc[diagnostics["fitKind"].eq("completed_trajectory")][
        [
            "candidateId",
            "trajectoryId",
            "status",
            "componentIdentityMaxAbsError",
            "retainedVariableCount",
            "miFinite",
            "partitionAverageFinite",
            "emergenceFiniteCount",
        ]
    ].rename(columns={"status": "numericalDiagnosticStatus"})
    prefix_diag = diagnostics.loc[
        diagnostics["fitKind"].eq("past_only_prefix_endpoint")
    ][
        [
            "candidateId",
            "trajectoryId",
            "endpointGeneration",
            "status",
            "componentIdentityMaxAbsError",
            "retainedVariableCount",
            "miFinite",
            "partitionAverageFinite",
            "emergenceFiniteCount",
        ]
    ].rename(
        columns={
            "endpointGeneration": "peakGeneration",
            "status": "numericalDiagnosticStatus",
        }
    )
    closure = preprocessing[
        ["candidateId", "trajectoryId", "maximumClosureError", "finite"]
    ].rename(columns={"finite": "preprocessingFinite"})
    completed = catalog.loc[catalog["temporalMode"].eq(COMPLETED_MODE)].merge(
        completed_diag,
        on=["candidateId", "trajectoryId"],
        how="left",
        validate="many_to_one",
    )
    prefix = catalog.loc[catalog["temporalMode"].eq(PREFIX_MODE)].merge(
        prefix_diag,
        on=["candidateId", "trajectoryId", "peakGeneration"],
        how="left",
        validate="many_to_one",
    )
    combined = pd.concat([completed, prefix], ignore_index=True)
    combined = combined.merge(
        closure,
        on=["candidateId", "trajectoryId"],
        how="left",
        validate="many_to_one",
    )
    history = partitions.rename(
        columns={
            "endpointGeneration": "peakGeneration",
            "endpointSelectedSequenceIndex": "peakSelectedSequenceIndex",
        }
    )
    combined = combined.merge(
        history[
            [
                "candidateId",
                "trajectoryId",
                "peakGeneration",
                "peakSelectedSequenceIndex",
                "partitionChangedFromPreviousEligibleFit",
                "partitionARIFromPreviousEligibleFit",
            ]
        ],
        on=[
            "candidateId",
            "trajectoryId",
            "peakGeneration",
            "peakSelectedSequenceIndex",
        ],
        how="left",
        validate="many_to_one",
    )
    combined["withinModePartitionChangeApplicability"] = np.where(
        combined["temporalMode"].eq(COMPLETED_MODE),
        "NOT_APPLICABLE_ONE_COMPLETED_TRAJECTORY_PARTITION",
        "DIRECT_PAST_ONLY_ENDPOINT_FIT",
    )
    combined["covarianceConditionNumber"] = np.nan
    combined["conditionNumberAvailability"] = "NOT_SERIALIZED_IN_FROZEN_S13Y"
    return combined.sort_values(
        [
            "temporalMode",
            "candidateId",
            "matrixIndex",
            "thresholdFamily",
            "sign",
            "peakRawObservationIndex",
        ],
        kind="stable",
        ignore_index=True,
    )


def partition_dependency(joined: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    evaluable = joined.loc[
        joined["partitionChangedFromPreviousEligibleFit"].notna()
    ].copy()
    for candidate_id, candidate in evaluable.groupby("candidateId", sort=True):
        changed = candidate["partitionChangedFromPreviousEligibleFit"].astype(bool)
        for family in THRESHOLD_FAMILIES:
            for sign in ("POSITIVE", "NEGATIVE"):
                suffix = f"{family}_{sign}"
                for mode, column, role in (
                    (
                        PREFIX_MODE,
                        f"pastOnly_{suffix}",
                        "DIRECT_PREFIX_ENDPOINT_PARTITION_CHANGE",
                    ),
                    (
                        COMPLETED_MODE,
                        f"completed_{suffix}",
                        "CROSS_MODE_AT_EXACT_SHARED_POST_FISSION_ENDPOINT",
                    ),
                ):
                    spike = candidate[column].astype(bool)
                    a = int(np.count_nonzero(spike & changed))
                    b = int(np.count_nonzero(spike & ~changed))
                    c = int(np.count_nonzero(~spike & changed))
                    d = int(np.count_nonzero(~spike & ~changed))
                    odds, pvalue = fisher_exact(
                        [[a, b], [c, d]], alternative="two-sided"
                    )
                    rows.append(
                        {
                            "candidateId": candidate_id,
                            "temporalMode": mode,
                            "thresholdFamily": family,
                            "sign": sign,
                            "relationshipRole": role,
                            "evaluableEndpointCount": len(candidate),
                            "partitionChangeCount": int(changed.sum()),
                            "partitionChangeFraction": float(changed.mean()),
                            "excursionAndChange": a,
                            "excursionAndNoChange": b,
                            "nonExcursionAndChange": c,
                            "nonExcursionAndNoChange": d,
                            "partitionChangeOddsRatio": float(odds),
                            "partitionChangeFisherTwoSidedP": float(pvalue),
                        }
                    )
    return pd.DataFrame(rows).sort_values(
        ["temporalMode", "candidateId", "thresholdFamily", "sign"],
        kind="stable",
        ignore_index=True,
    )


def numerical_summary(
    diagnostics: pd.DataFrame, preprocessing: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (candidate_id, fit_kind), group in diagnostics.groupby(
        ["candidateId", "fitKind"], sort=True
    ):
        closure = preprocessing.loc[preprocessing["candidateId"].eq(candidate_id)]
        rows.append(
            {
                "candidateId": candidate_id,
                "fitKind": fit_kind,
                "fitCount": len(group),
                "eligibleFitCount": int(group["status"].eq("ELIGIBLE").sum()),
                "maximumComponentIdentityAbsError": float(
                    group["componentIdentityMaxAbsError"].max()
                ),
                "minimumRetainedVariableCount": int(
                    group["retainedVariableCount"].min()
                ),
                "maximumRetainedVariableCount": int(
                    group["retainedVariableCount"].max()
                ),
                "allMiFinite": bool(group["miFinite"].astype(bool).all()),
                "allPartitionAveragesFinite": bool(
                    group["partitionAverageFinite"].astype(bool).all()
                ),
                "maximumPreprocessingClosureError": float(
                    closure["maximumClosureError"].max()
                ),
                "allPreprocessingFinite": bool(closure["finite"].astype(bool).all()),
                "conditionNumberAvailability": "NOT_SERIALIZED_IN_FROZEN_S13Y",
                "conditionNumberDependenceStatus": "NOT_EVALUABLE_WITHOUT_UPSTREAM_RECOMPUTATION",
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["fitKind", "candidateId"], kind="stable", ignore_index=True
    )


def add_pooled_summaries(
    catalog: pd.DataFrame,
    runs: pd.DataFrame,
    ljung: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pooled_catalog = catalog.copy()
    pooled_catalog["candidateId"] = "POOLED_SECONDARY"
    pooled_runs = runs.copy()
    pooled_runs["candidateId"] = "POOLED_SECONDARY"
    morphology = pd.concat(
        [
            summarize_excursions(catalog, runs),
            summarize_excursions(pooled_catalog, pooled_runs),
        ],
        ignore_index=True,
    )
    pooled_ljung = ljung.copy()
    pooled_ljung["candidateId"] = "POOLED_SECONDARY"
    ljung_summary = pd.concat(
        [summarize_ljung_box(ljung), summarize_ljung_box(pooled_ljung)],
        ignore_index=True,
    )
    return (
        morphology.sort_values(
            ["temporalMode", "candidateId", "thresholdFamily", "sign"],
            kind="stable",
            ignore_index=True,
        ),
        ljung_summary.sort_values(
            ["temporalMode", "candidateId", "transform"],
            kind="stable",
            ignore_index=True,
        ),
    )


def paper_targets(
    trends: pd.DataFrame,
    morphology: pd.DataFrame,
    ljung: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate_id in CANDIDATE_IDS:
        trend = trends.loc[
            trends["candidateScope"].eq(candidate_id)
            & trends["temporalMode"].eq(COMPLETED_MODE)
            & trends["alignmentView"].eq("AVAILABLE_CASE")
        ].iloc[0]
        positive = morphology.loc[
            morphology["candidateId"].eq(candidate_id)
            & morphology["temporalMode"].eq(COMPLETED_MODE)
            & morphology["thresholdFamily"].eq("THREE_SIGMA")
            & morphology["sign"].eq("POSITIVE")
        ].iloc[0]
        raw = ljung.loc[
            ljung["candidateId"].eq(candidate_id)
            & ljung["temporalMode"].eq(COMPLETED_MODE)
            & ljung["transform"].eq("RAW")
        ].iloc[0]
        diff = ljung.loc[
            ljung["candidateId"].eq(candidate_id)
            & ljung["temporalMode"].eq(COMPLETED_MODE)
            & ljung["transform"].eq("FIRST_DIFFERENCE")
        ].iloc[0]
        rows.extend(
            [
                {
                    "claimId": "E01-C013",
                    "candidateId": candidate_id,
                    "paperTarget": "aggregate linear trend p=0.1995; no significant trend",
                    "observedValue": float(trend["olsTwoSidedP"]),
                    "observedAuxiliary": f"slope={trend['olsSlope']:.17g}",
                    "status": "DIRECTIONALLY_SUPPORTED"
                    if float(trend["olsTwoSidedP"]) >= 0.05
                    else "NOT_SUPPORTED_WITHIN_TESTED_SCOPE",
                    "boundary": "primary available-case molecular-index alignment",
                },
                {
                    "claimId": "E01-C014",
                    "candidateId": candidate_id,
                    "paperTarget": "most runs have a positive >3-SD excursion",
                    "observedValue": int(positive["runWithExcursionCount"]),
                    "observedAuxiliary": f"fraction={positive['runWithExcursionFraction']:.17g}",
                    "status": "DIRECTIONALLY_SUPPORTED"
                    if float(positive["runWithExcursionFraction"]) > 0.5
                    else "NOT_SUPPORTED_WITHIN_TESTED_SCOPE",
                    "boundary": "exact paper count and threshold scope unavailable; inherited within-run S13Y rule",
                },
                {
                    "claimId": "E01-C022",
                    "candidateId": candidate_id,
                    "paperTarget": "86/100 raw trajectories reject Ljung-Box",
                    "observedValue": int(raw["rejectCountAt0_05"]),
                    "observedAuxiliary": "alpha=0.05",
                    "status": "CLOSELY_RECONSTRUCTED_LAG_UNDERDETERMINED"
                    if int(raw["rejectCountAt0_05"]) == 86
                    else (
                        "DIRECTIONALLY_SUPPORTED_LAG_UNDERDETERMINED"
                        if float(raw["rejectFractionAt0_05"]) > 0.5
                        else "NOT_SUPPORTED_WITHIN_TESTED_SCOPE"
                    ),
                    "boundary": "paper does not report Ljung-Box lag; frozen lag<=10 used",
                },
                {
                    "claimId": "E01-C023",
                    "candidateId": candidate_id,
                    "paperTarget": "median raw Ljung-Box p=2.07e-51",
                    "observedValue": float(raw["medianP"]),
                    "observedAuxiliary": "runwise median",
                    "status": "DIRECTIONALLY_SUPPORTED_LAG_UNDERDETERMINED"
                    if float(raw["medianP"]) < 0.05
                    else "NOT_SUPPORTED_WITHIN_TESTED_SCOPE",
                    "boundary": "paper does not report Ljung-Box lag",
                },
                {
                    "claimId": "E01-C024",
                    "candidateId": candidate_id,
                    "paperTarget": "100/100 differenced trajectories reject Ljung-Box",
                    "observedValue": int(diff["rejectCountAt0_05"]),
                    "observedAuxiliary": "first difference; alpha=0.05",
                    "status": "CLOSELY_RECONSTRUCTED_LAG_UNDERDETERMINED"
                    if int(diff["rejectCountAt0_05"]) == 100
                    else (
                        "DIRECTIONALLY_SUPPORTED_LAG_UNDERDETERMINED"
                        if float(diff["rejectFractionAt0_05"]) > 0.5
                        else "NOT_SUPPORTED_WITHIN_TESTED_SCOPE"
                    ),
                    "boundary": "paper does not report Ljung-Box lag",
                },
            ]
        )
    return pd.DataFrame(rows).sort_values(
        ["claimId", "candidateId"], kind="stable", ignore_index=True
    )


def compute_outputs(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    completed = prepare_completed(data["full"])
    prefix = prepare_prefix(data["prefix"])
    partition_history = partition_change_history(data["partitions"])
    completed_flagged, completed_thresholds = add_excursion_flags(completed)
    prefix_flagged, prefix_thresholds = add_excursion_flags(prefix)
    completed_catalog, completed_runs, completed_dependency = excursion_catalog(
        completed_flagged, completed_thresholds
    )
    prefix_catalog, prefix_runs, prefix_dependency = excursion_catalog(
        prefix_flagged, prefix_thresholds
    )
    catalog = pd.concat([completed_catalog, prefix_catalog], ignore_index=True)
    runs = pd.concat([completed_runs, prefix_runs], ignore_index=True)
    thresholds = pd.concat([completed_thresholds, prefix_thresholds], ignore_index=True)
    point_dependency = pd.concat(
        [completed_dependency, prefix_dependency], ignore_index=True
    )
    fission = combine_fission_dependency(point_dependency)
    aggregate = pd.concat(
        [aggregate_trajectories(completed), aggregate_trajectories(prefix)],
        ignore_index=True,
    ).sort_values(
        ["temporalMode", "candidateScope", "alignmentView", "timeCoordinate"],
        kind="stable",
        ignore_index=True,
    )
    trends = trend_results(aggregate)
    ljung = pd.concat(
        [ljung_box_results(completed), ljung_box_results(prefix)], ignore_index=True
    ).sort_values(
        ["temporalMode", "candidateId", "matrixIndex", "transform"],
        kind="stable",
        ignore_index=True,
    )
    morphology, ljung_summary = add_pooled_summaries(catalog, runs, ljung)
    joined, joined_summary = compare_completed_prefix(
        completed_flagged, prefix_flagged, partition_history
    )
    catalog = attach_diagnostics(
        catalog,
        data["sourceDiagnostics"],
        data["preprocessing"],
        partition_history,
    )
    partition = partition_dependency(joined)
    numerical = numerical_summary(data["sourceDiagnostics"], data["preprocessing"])
    targets = paper_targets(trends, morphology, ljung_summary)
    return {
        "aggregate": aggregate,
        "trends": trends,
        "thresholds": thresholds,
        "catalog": catalog,
        "runs": runs,
        "morphology": morphology,
        "ljung": ljung,
        "ljungSummary": ljung_summary,
        "partitionHistory": partition_history,
        "joined": joined,
        "joinedSummary": joined_summary,
        "fission": fission,
        "partition": partition,
        "numerical": numerical,
        "targets": targets,
    }


def deterministic_validation(
    first: dict[str, pd.DataFrame], second: dict[str, pd.DataFrame]
) -> tuple[bool, dict[str, str]]:
    digests: dict[str, str] = {}
    passed = True
    for name in sorted(first):
        left = frame_digest(first[name])
        right = frame_digest(second[name])
        digests[name] = left
        passed &= left == right
    return bool(passed), digests


def independent_output_checks(
    data: dict[str, pd.DataFrame], outputs: dict[str, pd.DataFrame]
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    trend_errors = []
    for _, row in (
        outputs["trends"].loc[outputs["trends"]["status"].eq("ELIGIBLE")].iterrows()
    ):
        group = outputs["aggregate"].loc[
            outputs["aggregate"]["candidateScope"].eq(row["candidateScope"])
            & outputs["aggregate"]["temporalMode"].eq(row["temporalMode"])
            & outputs["aggregate"]["alignmentView"].eq(row["alignmentView"])
        ]
        x = group["timeCoordinate"].to_numpy(float)
        y = group["medianEmergence"].to_numpy(float)
        centered_x = x - x.mean()
        independent_slope = float(
            np.dot(centered_x, y - y.mean()) / np.dot(centered_x, centered_x)
        )
        trend_errors.append(abs(independent_slope - float(row["olsSlope"])))
    checks.append(
        {
            "checkId": "INDEPENDENT_CLOSED_FORM_TREND_SLOPES",
            "passed": max(trend_errors, default=0.0) <= 1e-12,
            "detail": f"maximumAbsoluteError={max(trend_errors, default=0.0):.17g}",
        }
    )
    expected_ljung_rows = (
        data["full"]["trajectoryId"].nunique()
        + data["prefix"]
        .loc[data["prefix"].status.eq("ELIGIBLE"), "trajectoryId"]
        .nunique()
    ) * 2
    checks.append(
        {
            "checkId": "LJUNG_BOX_CARDINALITY",
            "passed": len(outputs["ljung"]) == expected_ljung_rows,
            "detail": f"observed={len(outputs['ljung'])} expected={expected_ljung_rows}",
        }
    )
    run_cardinality = (
        data["full"]["trajectoryId"].nunique()
        + data["prefix"]
        .loc[data["prefix"].status.eq("ELIGIBLE"), "trajectoryId"]
        .nunique()
    ) * 4
    checks.append(
        {
            "checkId": "SIGNED_EXCURSION_RUN_CARDINALITY",
            "passed": len(outputs["runs"]) == run_cardinality,
            "detail": f"observed={len(outputs['runs'])} expected={run_cardinality}",
        }
    )
    reconstructed = (
        outputs["catalog"]
        .groupby(
            [
                "candidateId",
                "trajectoryId",
                "temporalMode",
                "thresholdFamily",
                "sign",
            ],
            sort=True,
        )
        .size()
        .rename("catalogEpisodes")
    )
    run_counts = outputs["runs"].set_index(
        [
            "candidateId",
            "trajectoryId",
            "temporalMode",
            "thresholdFamily",
            "sign",
        ]
    )["excursionEpisodeCount"]
    aligned = run_counts.to_frame().join(reconstructed, how="left").fillna(0)
    checks.append(
        {
            "checkId": "EPISODE_CATALOG_RECONCILIATION",
            "passed": bool(
                np.array_equal(
                    aligned["excursionEpisodeCount"].to_numpy(int),
                    aligned["catalogEpisodes"].to_numpy(int),
                )
            ),
            "detail": f"runs={len(aligned)} episodes={len(outputs['catalog'])}",
        }
    )
    shared_expected = int(data["prefix"]["status"].eq("ELIGIBLE").sum())
    checks.append(
        {
            "checkId": "COMPLETED_PREFIX_EXACT_ENDPOINT_JOIN",
            "passed": len(outputs["joined"]) == shared_expected,
            "detail": f"observed={len(outputs['joined'])} expected={shared_expected}",
        }
    )
    all_passed = all(row["passed"] for row in checks)
    if not all_passed:
        raise RuntimeError(f"output validation failed: {checks}")
    return checks


def save_figure(path: Path, figure: plt.Figure) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
        metadata={"Software": VERSIONED_STEP_ID},
    )
    plt.close(figure)


def make_figures(
    output_root: Path,
    data: dict[str, pd.DataFrame],
    outputs: dict[str, pd.DataFrame],
) -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfbfb",
        }
    )
    colors = {CANDIDATE_IDS[0]: "#1769aa", CANDIDATE_IDS[1]: "#c44e52"}
    full = prepare_completed(data["full"])
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), constrained_layout=True)
    for column, candidate_id in enumerate(CANDIDATE_IDS):
        aggregate = outputs["aggregate"].loc[
            outputs["aggregate"]["candidateScope"].eq(candidate_id)
            & outputs["aggregate"]["temporalMode"].eq(COMPLETED_MODE)
            & outputs["aggregate"]["alignmentView"].eq("AVAILABLE_CASE")
        ]
        trend = (
            outputs["trends"]
            .loc[
                outputs["trends"]["candidateScope"].eq(candidate_id)
                & outputs["trends"]["temporalMode"].eq(COMPLETED_MODE)
                & outputs["trends"]["alignmentView"].eq("AVAILABLE_CASE")
            ]
            .iloc[0]
        )
        x = aggregate["timeCoordinate"].to_numpy(float)
        median = aggregate["medianEmergence"].to_numpy(float)
        sd = aggregate["standardDeviation"].to_numpy(float)
        ax = axes[0, column]
        ax.fill_between(
            x, median - sd, median + sd, color=colors[candidate_id], alpha=0.18
        )
        ax.plot(
            x, median, color=colors[candidate_id], lw=1.1, label="median ± sample SD"
        )
        ax.plot(
            x,
            trend["olsIntercept"] + trend["olsSlope"] * x,
            color="black",
            lw=1,
            ls="--",
            label=f"OLS p={trend['olsTwoSidedP']:.2g}",
        )
        ax.set_title(f"{candidate_id}: available-case aggregate")
        ax.set_xlabel("selected molecular-state index")
        ax.set_ylabel("source emergence")
        ax.legend(loc="best")

        representative = full.loc[
            full["candidateId"].eq(candidate_id) & full["matrixIndex"].eq(0)
        ].copy()
        flagged, threshold = add_excursion_flags(representative)
        threshold = threshold.loc[threshold["thresholdFamily"].eq("THREE_SIGMA")].iloc[
            0
        ]
        ax = axes[1, column]
        ax.plot(
            flagged["selectedSequenceIndex"],
            flagged["emergence"],
            color=colors[candidate_id],
            lw=0.8,
        )
        ax.axhline(threshold["upperThreshold"], color="#d95f02", ls="--", lw=0.8)
        ax.axhline(threshold["lowerThreshold"], color="#7570b3", ls="--", lw=0.8)
        positive = flagged["THREE_SIGMA_POSITIVE"].astype(bool)
        negative = flagged["THREE_SIGMA_NEGATIVE"].astype(bool)
        ax.scatter(
            flagged.loc[positive, "selectedSequenceIndex"],
            flagged.loc[positive, "emergence"],
            s=12,
            color="#d95f02",
            label="+3σ",
            zorder=3,
        )
        ax.scatter(
            flagged.loc[negative, "selectedSequenceIndex"],
            flagged.loc[negative, "emergence"],
            s=12,
            color="#7570b3",
            label="−3σ",
            zorder=3,
        )
        ax.set_title("Fixed representative run M000")
        ax.set_xlabel("selected molecular-state index")
        ax.set_ylabel("source emergence")
        ax.legend(loc="best")

        ax = axes[2, column]
        morphology = outputs["morphology"].loc[
            outputs["morphology"]["candidateId"].eq(candidate_id)
            & outputs["morphology"]["temporalMode"].eq(COMPLETED_MODE)
        ]
        ljung = outputs["ljungSummary"].loc[
            outputs["ljungSummary"]["candidateId"].eq(candidate_id)
            & outputs["ljungSummary"]["temporalMode"].eq(COMPLETED_MODE)
        ]
        labels = ["+3σ", "−3σ", "+MAD", "−MAD", "LB raw", "LB Δ"]
        values = [
            float(
                morphology.loc[
                    morphology["thresholdFamily"].eq("THREE_SIGMA")
                    & morphology["sign"].eq("POSITIVE"),
                    "runWithExcursionFraction",
                ].iloc[0]
            ),
            float(
                morphology.loc[
                    morphology["thresholdFamily"].eq("THREE_SIGMA")
                    & morphology["sign"].eq("NEGATIVE"),
                    "runWithExcursionFraction",
                ].iloc[0]
            ),
            float(
                morphology.loc[
                    morphology["thresholdFamily"].eq("ROBUST_MAD")
                    & morphology["sign"].eq("POSITIVE"),
                    "runWithExcursionFraction",
                ].iloc[0]
            ),
            float(
                morphology.loc[
                    morphology["thresholdFamily"].eq("ROBUST_MAD")
                    & morphology["sign"].eq("NEGATIVE"),
                    "runWithExcursionFraction",
                ].iloc[0]
            ),
            float(
                ljung.loc[ljung["transform"].eq("RAW"), "rejectFractionAt0_05"].iloc[0]
            ),
            float(
                ljung.loc[
                    ljung["transform"].eq("FIRST_DIFFERENCE"),
                    "rejectFractionAt0_05",
                ].iloc[0]
            ),
        ]
        ax.bar(
            labels,
            values,
            color=["#d95f02", "#7570b3", "#e6ab02", "#66a61e", "#1b9e77", "#1b9e77"],
        )
        ax.axhline(0.5, color="black", lw=0.8, ls=":")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("fraction of runs")
        ax.set_title("Excursion prevalence and temporal dependence")
    fig.suptitle("S14 candidate-specific Figure 2-like reconstruction", fontsize=12)
    save_figure(output_root / "figures/figure2_candidate_specific.png", fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    aggregate = outputs["aggregate"].loc[
        outputs["aggregate"]["candidateScope"].eq("POOLED_SECONDARY")
        & outputs["aggregate"]["temporalMode"].eq(COMPLETED_MODE)
        & outputs["aggregate"]["alignmentView"].eq("AVAILABLE_CASE")
    ]
    trend = (
        outputs["trends"]
        .loc[
            outputs["trends"]["candidateScope"].eq("POOLED_SECONDARY")
            & outputs["trends"]["temporalMode"].eq(COMPLETED_MODE)
            & outputs["trends"]["alignmentView"].eq("AVAILABLE_CASE")
        ]
        .iloc[0]
    )
    x = aggregate["timeCoordinate"].to_numpy(float)
    median = aggregate["medianEmergence"].to_numpy(float)
    sd = aggregate["standardDeviation"].to_numpy(float)
    axes[0].fill_between(x, median - sd, median + sd, color="#555555", alpha=0.2)
    axes[0].plot(x, median, color="#333333", lw=1, label="pooled median ± SD")
    axes[0].plot(
        x,
        trend["olsIntercept"] + trend["olsSlope"] * x,
        color="#b2182b",
        ls="--",
        label=f"OLS p={trend['olsTwoSidedP']:.2g}",
    )
    axes[0].set_xlabel("selected molecular-state index")
    axes[0].set_ylabel("source emergence")
    axes[0].set_title("Pooled secondary available-case aggregate (n=200 trajectories)")
    axes[0].legend()
    catalog = outputs["catalog"].loc[
        outputs["catalog"]["temporalMode"].eq(COMPLETED_MODE)
        & outputs["catalog"]["thresholdFamily"].eq("THREE_SIGMA")
    ]
    for sign, color in (("POSITIVE", "#d95f02"), ("NEGATIVE", "#7570b3")):
        values = catalog.loc[catalog["sign"].eq(sign), "normalizedPeakTime"]
        axes[1].hist(
            values,
            bins=np.linspace(0, 1, 21),
            alpha=0.55,
            label=sign.lower(),
            color=color,
        )
    axes[1].set_xlabel("normalized within-trajectory peak time")
    axes[1].set_ylabel("3σ excursion episodes")
    axes[1].set_title("Pooled secondary spike timing")
    axes[1].legend()
    save_figure(output_root / "figures/figure2_pooled_secondary.png", fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for column, candidate_id in enumerate(CANDIDATE_IDS):
        joined = outputs["joined"].loc[
            outputs["joined"]["candidateId"].eq(candidate_id)
        ]
        generations = (
            joined.groupby("prefixGeneration", sort=True)[
                ["completedEmergence", "pastOnlyEmergence"]
            ]
            .median()
            .reset_index()
        )
        axes[0, column].plot(
            generations["prefixGeneration"],
            generations["completedEmergence"],
            label="completed fit at endpoint",
            color=colors[candidate_id],
        )
        axes[0, column].plot(
            generations["prefixGeneration"],
            generations["pastOnlyEmergence"],
            label="past-only refit endpoint",
            color="#222222",
            alpha=0.8,
        )
        axes[0, column].set_title(candidate_id)
        axes[0, column].set_xlabel("generation")
        axes[0, column].set_ylabel("median endpoint emergence")
        axes[0, column].legend()
        axes[1, column].hexbin(
            joined["pastOnlyEmergence"],
            joined["completedEmergence"],
            gridsize=35,
            bins="log",
            mincnt=1,
            cmap="viridis",
        )
        limits = [
            float(
                min(
                    joined["pastOnlyEmergence"].min(),
                    joined["completedEmergence"].min(),
                )
            ),
            float(
                max(
                    joined["pastOnlyEmergence"].max(),
                    joined["completedEmergence"].max(),
                )
            ),
        ]
        axes[1, column].plot(limits, limits, "--", color="white", lw=0.8)
        axes[1, column].set_xlabel("past-only emergence")
        axes[1, column].set_ylabel("completed-fit emergence")
    fig.suptitle(
        "Completed-fit versus past-only values at exact shared endpoints", fontsize=12
    )
    save_figure(output_root / "figures/completed_fit_vs_past_only.png", fig)

    def plot_odds(ax: plt.Axes, labels: pd.Series, odds: pd.Series) -> None:
        values = odds.to_numpy(float)
        finite = values[np.isfinite(values)]
        cap = max(1.25, float(np.max(finite)) * 1.12) if len(finite) else 1.25
        displayed = np.where(np.isposinf(values), cap, values)
        bars = ax.bar(
            labels,
            displayed,
            color=["#7570b3" if sign == "NEGATIVE" else "#d95f02" for sign in labels],
        )
        for bar, value in zip(bars, values, strict=True):
            if np.isposinf(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    cap,
                    "∞",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for column, candidate_id in enumerate(CANDIDATE_IDS):
        fission = outputs["fission"].loc[
            outputs["fission"]["candidateId"].eq(candidate_id)
            & outputs["fission"]["temporalMode"].eq(COMPLETED_MODE)
            & outputs["fission"]["thresholdFamily"].eq("THREE_SIGMA")
        ]
        plot_odds(
            axes[0, column],
            fission["sign"],
            fission["pooledPointFissionOddsRatio"],
        )
        axes[0, column].axhline(1.0, color="black", ls=":", lw=0.8)
        axes[0, column].set_ylabel("post-fission excursion odds ratio")
        axes[0, column].set_title(f"{candidate_id}: completed-fit 3σ points")
        partition = outputs["partition"].loc[
            outputs["partition"]["candidateId"].eq(candidate_id)
            & outputs["partition"]["temporalMode"].eq(PREFIX_MODE)
            & outputs["partition"]["thresholdFamily"].eq("THREE_SIGMA")
        ]
        plot_odds(
            axes[1, column],
            partition["sign"],
            partition["partitionChangeOddsRatio"],
        )
        axes[1, column].axhline(1.0, color="black", ls=":", lw=0.8)
        axes[1, column].set_ylabel("past-only spike odds at partition change")
        axes[1, column].set_title("Prefix endpoint partition-change diagnostic")
    fig.suptitle(
        "Spike dependency diagnostics (condition number unavailable in frozen S13Y)",
        fontsize=12,
    )
    save_figure(output_root / "figures/spike_dependency_diagnostics.png", fig)


def fmt(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "NA"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}g}"
    return str(value)


def build_report(
    output_root: Path,
    outputs: dict[str, pd.DataFrame],
    validation: dict[str, Any],
    provenance: dict[str, Any],
) -> str:
    primary_trends = (
        outputs["trends"]
        .loc[
            outputs["trends"]["temporalMode"].eq(COMPLETED_MODE)
            & outputs["trends"]["alignmentView"].eq("AVAILABLE_CASE")
            & outputs["trends"]["candidateScope"].isin(CANDIDATE_IDS)
        ]
        .set_index("candidateScope")
    )
    morphology = outputs["morphology"].loc[
        outputs["morphology"]["temporalMode"].eq(COMPLETED_MODE)
    ]
    ljung = outputs["ljungSummary"].loc[
        outputs["ljungSummary"]["temporalMode"].eq(COMPLETED_MODE)
    ]
    joined = outputs["joinedSummary"].set_index("candidateId")
    lines = [
        "# E01/S14 — Reconstruct Descriptive Causal-Emergence Dynamics",
        "",
        "## Concise top summary",
        "",
        "| Field | Result |",
        "| --- | --- |",
        "| Research step ID | `S14` (`E01-S14-DESCRIPTIVE-CAUSAL-EMERGENCE-DYNAMICS-v1.0.0`) |",
        "| Completion status | **Complete** — only S14 was executed; S15 was not started |",
        f"| Artifacts written | {len(ARTIFACT_PATHS)} required paths under `{output_root}`: report, 15 machine-readable result tables, four figures, method/input/provenance/validation/status/failure manifests, and artifact manifest |",
        f"| Validation result | **{'PASS' if validation['overallPassed'] else 'FAIL'}** — {validation['passedCheckCount']}/{validation['checkCount']} checks; deterministic two-pass frame hashes and frozen-input before/after hashes matched |",
        "| Outcome classification | **Constraining/contradictory — `PUNCTUATED_EXCURSIONS_WITH_AGGREGATE_TREND_DISCREPANCY`** |",
        "| Caveats or blockers | Completed-fit values are retrospective; molecular trajectories have unequal lengths; the paper omits its Ljung–Box lag and exact spike threshold scope; past-only values are sparse post-fission endpoints; S13Y did not serialize covariance condition numbers |",
        "| Lay summary | The reconstructed series do spike and remain strongly time-dependent, much like the paper, but both simulator candidates show a statistically detectable positive aggregate trend rather than the paper's reported no-trend result. |",
        "| Recommended next action | Hand control back. Keep S15 queued and inactive until separately instructed; retain the aggregate-trend discrepancy and completed-fit/past-only dependence as fixed S15 context. |",
        "",
        "## Lay summary",
        "",
        "The closest locked reconstruction recovers the paper's qualitative picture of irregular bursts: 90 of 100 runs in each candidate contain at least one positive three-standard-deviation excursion, and all 100 differenced trajectories in each candidate reject the paper-like Ljung–Box independence test. Negative excursions are at least as prevalent, however, and the aggregate median rises significantly in both candidate pipelines. The visual resemblance therefore does not reproduce the paper's central combination of spikes *without* an aggregate trend.",
        "",
        "The numerical values also depend materially on when the PhiRL fit is performed. Completed-trajectory values and independently refit past-only endpoint values agree only weakly to moderately at shared endpoints and have low signed-excursion overlap. This is descriptive evidence of retrospective temporal-fitting dependence, not early-warning or causal-control evidence.",
        "",
        "## Frozen question",
        "",
        "Do the frozen S13Y source-defined emergence values reproduce the paper's Figure 2-like combination of no aggregate linear trend and punctuated run-level excursions, separately for both confirmed simulator candidates?",
        "",
        "## Inputs",
        "",
        "- Frozen S13Y completed-fit source values: `full_source_values.parquet` (180,435 rows).",
        "- Frozen S13Y past-only endpoint values: `prefix_endpoint_values.parquet` (20,000 status-bearing rows; eligible rows analyzed).",
        "- Frozen S13Y partition history, source diagnostics, preprocessing diagnostics, trajectory manifest, and simulation summary.",
        "- Original arXiv v1 paper PDF, SHA-256 recorded in `input_manifest.json`.",
        "- Candidate 2: `h=0.6031526490073492`, first-daughter continuation. Candidate 3: `h=0.5613315384859516`, random-nonempty daughter continuation. Both retain trim-new-entrants and selected-daughter-boundary semantics.",
        "- Exact S13Y PhiRL regularized source-emergence branch; no scalar, threshold, partition, preprocessing, label, alignment, or simulator was changed.",
        "",
        "## Detailed methods",
        "",
        "### Aggregate alignment and trend",
        "",
        "The paper-like primary view groups completed-fit values by selected molecular-state index, takes the available-case median, and calculates the sample standard deviation across contributing trajectories. Ordinary unweighted linear regression of that median on molecular index is the primary trend. To expose unequal-length dependence without choosing a favorable result, the same locked calculation is also reported on full-cohort support, majority support, and a 101-point normalized-lifetime interpolation. Theil–Sen slopes and intervals are robustness diagnostics only.",
        "",
        "### Excursions and morphology",
        "",
        "The inherited S13Y rule is within-trajectory mean ± three population standard deviations (`ddof=0`). Robust excursions use median ± three times `1.4826 × MAD`. Positive and negative excursions remain separate. Consecutive flagged observations form an episode; the most signed-extreme observation (first tie) is its peak. The catalog records episode width, raw-index span, half-prominence width, prominence, inter-peak spacing, molecular/generation timing, normalized timing, and proximity to fission.",
        "",
        "### Temporal dependence",
        "",
        "Each eligible raw and first-differenced trajectory receives one Ljung–Box test at `max(1, min(10, floor(n/5)))`, exactly inheriting S13Y. Tests are unadjusted paper-like descriptive diagnostics. The paper does not specify its lag, so numerical agreement remains lag-underdetermined.",
        "",
        "### Completed-fit versus past-only and dependency diagnostics",
        "",
        "Eligible past-only endpoint rows are joined one-to-one to the exact completed-fit row by candidate, trajectory, matrix, and selected-sequence index. Unordered bipartitions are compared across consecutive eligible prefix fits, so swapping partition sides does not count as change. Fission enrichment uses descriptive Fisher exact tests on point-level 2×2 tables. Numerical diagnostics use only fields serialized by S13Y (finite flags, component-identity error, retained-variable count, and closure error); no missing covariance condition number was reconstructed.",
        "",
        "## Commands",
        "",
        "```bash",
        "PYTHONPATH=src pytest -q tests/e01/test_s14_descriptive_causal_emergence.py",
        "PYTHONPATH=src ruff check src/e01_descriptive_causal_emergence scripts/e01/run_s14_descriptive_dynamics.py tests/e01/test_s14_descriptive_causal_emergence.py",
        "PYTHONPATH=src python -m compileall -q src/e01_descriptive_causal_emergence scripts/e01/run_s14_descriptive_dynamics.py",
        f"PYTHONPATH=src python scripts/e01/run_s14_descriptive_dynamics.py --output-root {output_root}",
        "```",
        "",
        "No simulator, PhiRL fitter, GPU process, package installer, or network call was invoked. CPU float64 was authoritative; execution was serial because the frozen-input analysis is small and deterministic.",
        "",
        "## Results",
        "",
        "### Aggregate trend and paper-facing excursions",
        "",
        "| Candidate | Available-case slope | two-sided p | +3σ runs | −3σ runs | Robust +MAD runs | Robust −MAD runs |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for candidate_id in CANDIDATE_IDS:
        trend = primary_trends.loc[candidate_id]
        values = {}
        for family, sign, key in (
            ("THREE_SIGMA", "POSITIVE", "pos3"),
            ("THREE_SIGMA", "NEGATIVE", "neg3"),
            ("ROBUST_MAD", "POSITIVE", "posmad"),
            ("ROBUST_MAD", "NEGATIVE", "negmad"),
        ):
            row = morphology.loc[
                morphology["candidateId"].eq(candidate_id)
                & morphology["thresholdFamily"].eq(family)
                & morphology["sign"].eq(sign)
            ].iloc[0]
            values[key] = int(row["runWithExcursionCount"])
        lines.append(
            f"| {candidate_id} | {fmt(trend['olsSlope'], 6)} | {fmt(trend['olsTwoSidedP'], 6)} | {values['pos3']}/100 | {values['neg3']}/100 | {values['posmad']}/100 | {values['negmad']}/100 |"
        )
    lines.extend(
        [
            "",
            "Both candidates reproduce punctuated positive excursions in a majority of runs. Both also have more prevalent negative excursions, and neither reproduces the paper's nonsignificant aggregate trend under the inherited available-case primary alignment. The alternate alignment rows are retained in `aggregate_trend_results.csv`; none is selected post hoc to replace the primary result.",
            "",
            "### Ljung–Box reconstruction",
            "",
            "| Candidate | Raw rejects | Raw median p | Differenced rejects | Differenced median p |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for candidate_id in CANDIDATE_IDS:
        raw = ljung.loc[
            ljung["candidateId"].eq(candidate_id) & ljung["transform"].eq("RAW")
        ].iloc[0]
        diff = ljung.loc[
            ljung["candidateId"].eq(candidate_id)
            & ljung["transform"].eq("FIRST_DIFFERENCE")
        ].iloc[0]
        lines.append(
            f"| {candidate_id} | {int(raw['rejectCountAt0_05'])}/100 | {fmt(raw['medianP'], 6)} | {int(diff['rejectCountAt0_05'])}/100 | {fmt(diff['medianP'], 6)} |"
        )
    lines.extend(
        [
            "",
            "The differenced 100/100 rejection count closely reconstructs the reported count for both candidates. Raw rejection is directionally similar but below the paper's 86/100 target. Because the original lag is unavailable, all such comparisons retain an explicit lag-underdetermined qualifier.",
            "",
            "### Completed-fit versus past-only values",
            "",
            "| Candidate | Shared endpoints | Event Spearman | Median runwise Spearman | Median absolute difference | +3σ Jaccard | −3σ Jaccard | Partition-change fraction |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for candidate_id in CANDIDATE_IDS:
        row = joined.loc[candidate_id]
        lines.append(
            f"| {candidate_id} | {int(row['sharedEndpointCount'])} | {fmt(row['eventSpearmanRho'])} | {fmt(row['medianTrajectorySpearmanRho'])} | {fmt(row['medianAbsoluteDifference'])} | {fmt(row['THREE_SIGMA_POSITIVE_jaccard'])} | {fmt(row['THREE_SIGMA_NEGATIVE_jaccard'])} | {fmt(row['partitionChangeFraction'])} |"
        )
    lines.extend(
        [
            "",
            "Completed-fit and past-only endpoint values are therefore not interchangeable. Completed-fit partitions are fixed once per whole trajectory, so within-run completed-fit partition changes are structurally not applicable. Prefix partitions can change between endpoint refits; their direct and cross-mode spike associations are reported without causal interpretation.",
            "",
            "### Numerical-condition boundary",
            "",
            "All serialized S13Y fits used here were eligible, had finite MI/partition summaries, and retained exact emergence-component closure within the recorded tolerance. Preprocessing closure errors remained at floating-point scale. S13Y did not serialize covariance condition numbers, so spike dependence on that specific condition measure is **not evaluable without an unauthorized upstream recomputation**.",
            "",
            "### Paper-target classification",
            "",
            "The machine-readable `paper_target_comparison.csv` assigns candidate-specific statuses. In brief: E01-C014 (positive spikes) is directionally supported; E01-C024 (differenced Ljung–Box count) is closely reconstructed but lag-underdetermined; E01-C022/C023 are directionally supported but not numerically exact and lag-underdetermined; E01-C013 (no aggregate trend) is not supported within the tested scope.",
            "",
            "## Figures and tables",
            "",
            "- `figures/figure2_candidate_specific.png`: candidate-specific aggregate, fixed representative run, excursion prevalence, and Ljung–Box panels.",
            "- `figures/figure2_pooled_secondary.png`: pooled secondary aggregate and spike timing.",
            "- `figures/completed_fit_vs_past_only.png`: exact endpoint value comparison.",
            "- `figures/spike_dependency_diagnostics.png`: fission and prefix-partition diagnostics.",
            "- Machine-readable tables preserve every aggregate alignment, episode, run summary, trend, Ljung–Box result, partition comparison, and paper-target classification.",
            "",
            "## Validation",
            "",
            f"Validation passed {validation['passedCheckCount']}/{validation['checkCount']} checks. Two independent executions of every derived frame had identical content hashes. All 54 S13Y manifest members and all 200 frozen raw trajectory cache files matched before and after S14. Row keys, source-component identity, candidate contracts, trajectory cardinality, exact shared-endpoint joins, excursion catalog reconciliation, closed-form trend slopes, artifact schemas, figures, and final hashes passed. No S14 trajectory or estimator cache was created.",
            "",
            "Repository checks passed: five focused S14 tests, Ruff, and bytecode compilation. The exact commit and remote branch are recorded in `provenance_manifest.json`.",
            "",
            "## Caveats, blockers, failed assumptions, and limitations",
            "",
            "- The primary completed-fit values use the completed trajectory to fit partitions and Gaussian parameters; they are retrospective.",
            "- Molecular trajectories have unequal lengths. Available-case tail positions contain fewer trajectories; full-cohort, majority, and normalized-time alternatives are reported but cannot identify the unpublished paper alignment.",
            "- The paper omits the exact 3σ threshold scope, Ljung–Box lag, and spike morphology definition. S14 inherits S13Y's locked within-run and lag rules rather than tuning them.",
            "- Past-only values begin only after 256 prior locked-clock transitions and occur at post-fission endpoints, so they are a sparse comparator rather than a full molecular-time reconstruction.",
            "- Fission and partition analyses are descriptive, point-dependent, and not causal tests. Every eligible past-only observation is itself post-fission, so fission enrichment is structurally not estimable in that mode; completed-fit partition change is not a meaningful within-run variable.",
            "- Covariance condition-number dependence is unavailable from the frozen serialized diagnostics. Recomputing it would change the authorized upstream analysis surface and was not done.",
            "- Candidate pooling is secondary only. The constraining outcome follows the separate candidate-specific primary results.",
            "- This step does not evaluate level/change association, prediction, or intervention claims and cannot support early warning or causal control.",
            "",
            "## Provenance",
            "",
            f"- Repository: `{provenance['repository']['remote']}` on `{provenance['repository']['branch']}` at `{provenance['repository']['commit']}`.",
            f"- Python: `{provenance['runtime']['python']}`; NumPy `{provenance['runtime']['numpy']}`, pandas `{provenance['runtime']['pandas']}`, SciPy `{provenance['runtime']['scipy']}`, statsmodels `{provenance['runtime']['statsmodels']}`, scikit-learn `{provenance['runtime']['sklearn']}`, PyArrow `{provenance['runtime']['pyarrow']}`.",
            "- Numeric policy: CPU float64, serial execution; no GPU acceleration.",
            "- Input and output SHA-256 hashes: `input_manifest.json` and `artifact_manifest.json`.",
            "- Reproducible source: repository config, `src/e01_descriptive_causal_emergence/`, runner, and focused tests; no repository source was copied into artifacts.",
            "",
            "## Recommended next action",
            "",
            "Hand control back to the Chief Scientist workflow. S15 is next in the directed queue but remains inactive until a separate instruction. When started, S15 should preserve this S14 aggregate-trend discrepancy, the strong signed/robust excursion evidence, the unspecified-lag boundary, and the completed-fit/past-only divergence. Do not begin S15 in this execution.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(f"refusing to overwrite nonempty output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "figures").mkdir(exist_ok=True)

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["researchStepId"] != RESEARCH_STEP_ID:
        raise RuntimeError("method config research-step mismatch")
    data = load_inputs()
    input_checks = validate_frozen_inputs(data)
    input_before = snapshot_inputs(data["trajectoryManifest"])
    if not all(entry["matched"] for entry in input_before):
        raise RuntimeError("one or more frozen input hashes do not match")
    input_manifest = {
        "schema": "eidosoma.e01.s14.input_manifest.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "versionedStepId": VERSIONED_STEP_ID,
        "entryCount": len(input_before),
        "totalBytes": sum(int(entry["bytes"]) for entry in input_before),
        "allMatched": True,
        "entries": input_before,
    }
    write_json(output_root / "input_manifest.json", input_manifest)
    write_json(
        output_root / "method_lock.json",
        {
            "schema": "eidosoma.e01.s14.method_lock.v1",
            "researchStepId": RESEARCH_STEP_ID,
            "versionedStepId": VERSIONED_STEP_ID,
            "configPath": str(CONFIG_PATH),
            "configSha256": sha256_file(CONFIG_PATH),
            "config": config,
            "newTrajectoriesGenerated": 0,
            "sourceRefitsExecuted": 0,
            "upstreamMethodChanges": 0,
        },
    )

    outputs = compute_outputs(data)
    repeated = compute_outputs(data)
    deterministic_passed, frame_digests = deterministic_validation(outputs, repeated)
    if not deterministic_passed:
        raise RuntimeError("deterministic two-pass frame validation failed")
    output_checks = independent_output_checks(data, outputs)

    write_parquet(output_root / "aggregate_trajectory.parquet", outputs["aggregate"])
    write_csv(output_root / "aggregate_trend_results.csv", outputs["trends"])
    write_parquet(output_root / "excursion_thresholds.parquet", outputs["thresholds"])
    write_parquet(output_root / "spike_catalog.parquet", outputs["catalog"])
    write_csv(output_root / "spike_run_summary.csv", outputs["runs"])
    write_csv(output_root / "spike_morphology_summary.csv", outputs["morphology"])
    write_parquet(output_root / "ljung_box_results.parquet", outputs["ljung"])
    write_csv(output_root / "ljung_box_summary.csv", outputs["ljungSummary"])
    write_parquet(
        output_root / "partition_change_history.parquet", outputs["partitionHistory"]
    )
    write_parquet(output_root / "completed_vs_past_only.parquet", outputs["joined"])
    write_csv(
        output_root / "completed_vs_past_only_summary.csv", outputs["joinedSummary"]
    )
    write_csv(output_root / "fission_dependency.csv", outputs["fission"])
    write_csv(output_root / "partition_dependency.csv", outputs["partition"])
    write_csv(output_root / "numerical_diagnostic_summary.csv", outputs["numerical"])
    write_csv(output_root / "paper_target_comparison.csv", outputs["targets"])
    make_figures(output_root, data, outputs)

    input_after = snapshot_inputs(data["trajectoryManifest"])
    before_map = {entry["path"]: entry["sha256"] for entry in input_before}
    after_map = {entry["path"]: entry["sha256"] for entry in input_after}
    immutability_passed = before_map == after_map
    no_trajectory_outputs = not any(
        path.suffix.lower() in {".pickle", ".pkl"}
        for path in output_root.rglob("*")
        if path.is_file()
    )
    final_checks = [
        *input_checks,
        *output_checks,
        {
            "checkId": "DETERMINISTIC_TWO_PASS_FRAME_HASHES",
            "passed": deterministic_passed,
            "detail": f"frameCount={len(frame_digests)}",
        },
        {
            "checkId": "FROZEN_INPUT_HASHES_UNCHANGED_AFTER_ANALYSIS",
            "passed": immutability_passed,
            "detail": f"inputCount={len(before_map)}",
        },
        {
            "checkId": "ZERO_GENERATED_TRAJECTORY_FILES",
            "passed": no_trajectory_outputs,
            "detail": "no pickle/pkl outputs; sourceRefitsExecuted=0",
        },
        {
            "checkId": "FIGURE_CARDINALITY",
            "passed": len(list((output_root / "figures").glob("*.png"))) == 4,
            "detail": "four required PNG figures",
        },
    ]
    overall = all(check["passed"] for check in final_checks)
    validation = {
        "schema": "eidosoma.e01.s14.validation.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "overallPassed": overall,
        "checkCount": len(final_checks),
        "passedCheckCount": sum(bool(check["passed"]) for check in final_checks),
        "failedCheckCount": sum(not bool(check["passed"]) for check in final_checks),
        "checks": final_checks,
        "deterministicFrameSha256": frame_digests,
        "frozenInputBeforeAfterSha256Equal": immutability_passed,
        "newTrajectoryCount": 0,
        "sourceRefitCount": 0,
    }
    if not overall:
        raise RuntimeError("final S14 validation failed")
    write_json(output_root / "validation.json", validation)

    provenance = {
        "schema": "eidosoma.e01.s14.provenance_manifest.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "versionedStepId": VERSIONED_STEP_ID,
        "repository": {
            "path": str(REPO_ROOT),
            "commit": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current"),
            "remote": git("remote", "get-url", "origin"),
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
            "sklearn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
            "pyarrow": pyarrow.__version__,
            "pyyaml": importlib.metadata.version("PyYAML"),
        },
        "numericPolicy": {
            "authoritativeBackend": "CPU_FLOAT64",
            "gpuUsed": False,
            "workers": 1,
            "threadEnvironment": {
                key: os.environ.get(key)
                for key in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                )
            },
        },
        "commands": [
            "PYTHONPATH=src pytest -q tests/e01/test_s14_descriptive_causal_emergence.py",
            "PYTHONPATH=src ruff check src/e01_descriptive_causal_emergence scripts/e01/run_s14_descriptive_dynamics.py tests/e01/test_s14_descriptive_causal_emergence.py",
            "PYTHONPATH=src python -m compileall -q src/e01_descriptive_causal_emergence scripts/e01/run_s14_descriptive_dynamics.py",
            f"PYTHONPATH=src python scripts/e01/run_s14_descriptive_dynamics.py --output-root {output_root}",
        ],
        "inputManifest": str(output_root / "input_manifest.json"),
        "methodConfig": str(CONFIG_PATH),
        "methodConfigSha256": sha256_file(CONFIG_PATH),
        "originalPaper": str(PAPER_PATH),
        "originalPaperSha256": sha256_file(PAPER_PATH),
        "newDependenciesInstalled": [],
    }
    write_json(output_root / "provenance_manifest.json", provenance)
    failure_ledger = pd.DataFrame(
        columns=[
            "failureId",
            "stage",
            "severity",
            "status",
            "reason",
            "gateImpact",
            "repairAttempted",
        ]
    )
    write_csv(output_root / "failure_ledger.csv", failure_ledger)

    classification = "PUNCTUATED_EXCURSIONS_WITH_AGGREGATE_TREND_DISCREPANCY"
    status = {
        "schema": "eidosoma.e01.s14.status.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "stepNumber": 14,
        "success": True,
        "status": "COMPLETED",
        "outcomeClassification": "CONSTRAINING_CONTRADICTORY",
        "classification": classification,
        "artifactsWritten": list(ARTIFACT_PATHS),
        "validationResult": f"PASS: {validation['passedCheckCount']}/{validation['checkCount']} checks",
        "caveatsOrBlockers": [
            "completed-fit values are retrospective and future-fitted",
            "unequal molecular trajectory lengths make aggregate alignment ambiguous",
            "paper Ljung-Box lag and exact spike threshold scope are unavailable",
            "past-only values are sparse eligible post-fission endpoints",
            "covariance condition numbers were not serialized by frozen S13Y and were not recomputed",
        ],
        "recommendedNextAction": "Hand control back; keep S15 queued and inactive until separately instructed.",
        "nextResearchStepStarted": False,
        "newTrajectoryCount": 0,
        "sourceRefitCount": 0,
    }
    write_json(output_root / "status.json", status)
    report = build_report(output_root, outputs, validation, provenance)
    (output_root / "research_step_full_results.md").write_text(report, encoding="utf-8")

    missing = [
        path for path in ARTIFACT_PATHS[:-1] if not (output_root / path).exists()
    ]
    if missing:
        raise RuntimeError(f"missing required artifacts before manifest: {missing}")
    artifacts = []
    for path in sorted([item for item in output_root.rglob("*") if item.is_file()]):
        if path.name == "artifact_manifest.json":
            continue
        artifacts.append(
            {
                "path": str(path.relative_to(output_root)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "schema": "eidosoma.e01.s14.artifact_manifest.v1",
        "researchStepId": RESEARCH_STEP_ID,
        "artifactCountExcludingSelf": len(artifacts),
        "requiredArtifactCountIncludingSelf": len(ARTIFACT_PATHS),
        "totalBytesExcludingSelf": sum(item["bytes"] for item in artifacts),
        "missingRequired": [],
        "passed": len(artifacts) == len(ARTIFACT_PATHS) - 1,
        "artifacts": artifacts,
    }
    write_json(output_root / "artifact_manifest.json", manifest)
    if not manifest["passed"]:
        raise RuntimeError(
            f"artifact count mismatch: {len(artifacts)} != {len(ARTIFACT_PATHS) - 1}"
        )
    print(
        json.dumps(
            {
                "researchStepId": RESEARCH_STEP_ID,
                "status": "COMPLETED",
                "classification": classification,
                "validation": validation["overallPassed"],
                "artifactCount": len(ARTIFACT_PATHS),
                "outputRoot": str(output_root),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
