#!/usr/bin/env python3
"""Execute the frozen E01-S12-STRICT-MRR-v1.0.0 computation."""

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
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import yaml
from scipy.stats import linregress, spearmanr
from statsmodels.stats.diagnostic import acorr_ljungbox

from e01_gard_reproducibility import StreamPurpose
from e01_strict_mrr.analysis import (
    BaselineAnalysis,
    analysis_seed_bundle,
    analyze_baseline,
)
from e01_strict_mrr.core import (
    NUMERIC_TOLERANCE,
    PREPROCESSING_IDS,
    REDUNDANCY_IDS,
    action_null_envelope,
    expanding_estimates,
    lineage_event_rows,
    preprocess_states,
    score_action_candidates,
    simulate_baseline,
)
from e01_strict_mrr.intervention import (
    InterventionTrajectory,
    intervention_event_rows,
    simulate_intervention,
)

REPO = Path(__file__).resolve().parents[2]
STEP_ROOT = Path("/artifacts/research_steps/S12")
BUNDLE_ROOT = Path("/artifacts/E01_forensic_replication_bundle/data/s12_strict_mrr")
CACHE_ROOT = Path("/cache/e01_s12")
PREREG = REPO / "configs/e01/s12_strict_mrr_preregistration.yaml"
AMENDMENT = REPO / "configs/e01/s12_strict_mrr_preregistration_amendment_1.yaml"
AMENDMENT_2 = REPO / "configs/e01/s12_strict_mrr_preregistration_amendment_2.yaml"
CLAIM_LEDGER = Path(
    "/artifacts/E01_forensic_replication_bundle/ledgers/claim_ledger.csv"
)
FIGURE_ROOT = STEP_ROOT / "figures"


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_parquet(path, index=False, compression="zstd")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, lineterminator="\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()


def _simulate_baseline_worker(index: int) -> tuple[Any, float]:
    started = time.perf_counter()
    result = simulate_baseline(index)
    return result, time.perf_counter() - started


def _simulate_triplet_worker(
    matrix_index: int, first_authorized_generation: int
) -> list[InterventionTrajectory]:
    return [
        simulate_intervention(
            matrix_index,
            condition,
            first_authorized_generation=first_authorized_generation,
        )
        for condition in ("max", "control", "min")
    ]


def flatten_seed_payload(
    payload: dict[str, Any], *, seed_role: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for purpose, stream in payload["streams"].items():
        rows.append(
            {
                "seedRole": seed_role,
                "experimentId": payload["experimentId"],
                "specificationId": payload["specificationId"],
                "trajectoryId": payload["trajectoryId"],
                "replicateIndex": payload["replicateIndex"],
                "couplingPolicy": payload["couplingPolicy"],
                "couplingReason": payload["couplingReason"],
                "purpose": purpose,
                **stream,
            }
        )
    return rows


def immutable_preflight() -> dict[str, Any]:
    sys.path.insert(0, str(REPO / "scripts/e01"))
    import freeze_s12_preregistration as freezer

    base = freezer.validate_preregistration(require_no_outcomes=False)
    amendment = freezer.validate_amendment()
    amendment_2 = freezer.validate_amendment_2()
    if not base["success"] or not amendment["success"] or not amendment_2["success"]:
        raise RuntimeError(
            "S12 frozen-input preflight failed: "
            f"{base['errors']} {amendment['errors']} {amendment_2['errors']}"
        )
    return {"base": base, "amendment": amendment, "amendment2": amendment_2}


def baseline_rows(
    analyses: list[BaselineAnalysis],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    observation_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    preprocessing_rows: list[dict[str, Any]] = []
    expanding_rows: list[dict[str, Any]] = []
    post_fission_rows: list[dict[str, Any]] = []
    partition_rows: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    whole_rows: list[dict[str, Any]] = []
    whole_local: list[dict[str, Any]] = []
    for analysis in analyses:
        trajectory = analysis.trajectory
        for index, state in enumerate(trajectory.states):
            observation_rows.append(
                {
                    "trajectoryId": trajectory.trajectory_id,
                    "matrixIndex": trajectory.matrix_index,
                    "observationIndex": index,
                    "observationKind": trajectory.observation_kinds[index],
                    "generation": int(trajectory.generations[index]),
                    "growthGenerationOneBased": int(
                        trajectory.growth_generations_one_based[index]
                    ),
                    "molecularStep": int(trajectory.molecular_steps[index]),
                    "generationLocalStep": int(
                        trajectory.generation_local_steps[index]
                    ),
                    "mass": int(np.sum(state)),
                    "state": state.tolist(),
                }
            )
            for preprocessing_id in PREPROCESSING_IDS:
                preprocessing_rows.append(
                    {
                        "trajectoryId": trajectory.trajectory_id,
                        "matrixIndex": trajectory.matrix_index,
                        "observationIndex": index,
                        "preprocessingId": preprocessing_id,
                        "status": "ELIGIBLE",
                        "reason": None,
                        "inputMass": int(analysis.preprocessing.masses[index]),
                        "zeroCount": int(analysis.preprocessing.zero_counts[index]),
                        "coordinateDimension": 99,
                        "finite": bool(
                            np.all(
                                np.isfinite(
                                    analysis.preprocessing.coordinates[
                                        preprocessing_id
                                    ][index]
                                )
                            )
                        ),
                        "maximumAbsoluteInverseError": float(
                            analysis.preprocessing.maximum_inverse_errors[
                                preprocessing_id
                            ][index]
                        ),
                        "closureError": float(
                            analysis.preprocessing.maximum_closure_errors[
                                preprocessing_id
                            ][index]
                        ),
                    }
                )
        for payload in lineage_event_rows(trajectory):
            event_rows.append(
                {
                    "trajectoryId": payload["trajectoryId"],
                    "matrixIndex": payload["matrixIndex"],
                    "recordType": payload["recordType"],
                    "generationIndexOneBased": payload["generation_index_one_based"],
                    "globalEventIndexOneBased": payload["globalEventIndexOneBased"],
                    "recordPayloadJson": json.dumps(
                        _jsonable(
                            {
                                key: value
                                for key, value in payload.items()
                                if key
                                not in {
                                    "trajectoryId",
                                    "matrixIndex",
                                    "recordType",
                                    "globalEventIndexOneBased",
                                }
                            }
                        ),
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                }
            )
        labels.extend(analysis.label_rows)
        whole_rows.extend(analysis.whole_rows)
        whole_local.extend(analysis.whole_local_rows)
        for preprocessing_id in PREPROCESSING_IDS:
            lock = analysis.locks[preprocessing_id]
            partition_rows.extend(
                {
                    **row,
                    "trajectoryId": trajectory.trajectory_id,
                    "matrixIndex": trajectory.matrix_index,
                    "partitionId": (
                        lock.partition_id
                        if row["status"] == "ELIGIBLE_LOCKED"
                        else None
                    ),
                    "scope": "PROSPECTIVE_PAST_ONLY_EXPANDING",
                }
                for row in lock.history
            )
            for index, estimate in enumerate(analysis.estimates[preprocessing_id]):
                generation = int(trajectory.generations[index])
                base = {
                    "specificationId": "E01-S12-STRICT-MRR-v1.0.0",
                    "trajectoryId": trajectory.trajectory_id,
                    "matrixIndex": trajectory.matrix_index,
                    "observationIndex": index,
                    "observationKind": trajectory.observation_kinds[index],
                    "molecularStep": int(trajectory.molecular_steps[index]),
                    "generation": generation,
                    "preprocessingId": preprocessing_id,
                    "partitionBranchId": "E01-S12-PARTITION-PASTONLY-FIRST-PASS-LOCK-v1.0.0",
                    "partitionId": (
                        lock.partition_id
                        if lock.observation_index is not None
                        and index >= lock.observation_index
                        else None
                    ),
                    "estimatorId": "E01-S10-PHYID-GAUSSIAN-STRICT-v1.0.0",
                    "aggregateId": "E01-S10-AGG-PAPER-EQUATION-v1.0.0",
                    "tau": 1,
                    "nEff": estimate.n_eff,
                    "status": estimate.status,
                    "reason": estimate.reason,
                    "value": estimate.value,
                    "numericalRank": estimate.numerical_rank,
                    "rankTolerance": estimate.rank_tolerance,
                    "conditionNumber": estimate.condition_number,
                    "minimumEigenvalue": estimate.minimum_eigenvalue,
                    "latticeClosureError": estimate.lattice_closure_error,
                    "paperEquationClosureError": estimate.paper_equation_closure_error,
                    "totalMutualInformation": estimate.total_mutual_information,
                    "historicalReplicator": analysis.historical_labels.get(generation),
                    "onlineReplicator": analysis.online_labels.get(generation),
                    "scopeLabel": "PROSPECTIVE_PAST_ONLY_EXPANDING",
                }
                for redundancy_id in REDUNDANCY_IDS:
                    row = {
                        **base,
                        "redundancyId": redundancy_id,
                        "atomEvaluationPolicy": (
                            "ANALYTIC_MMI_MEAN_ATOMS_EVERY_STEP"
                            if redundancy_id.endswith("MMI-v1.0.0")
                            else "CCS_SOURCE_AT_FROZEN_CHECKPOINTS_NO_ATOM_IMPUTATION"
                        ),
                    }
                    expanding_rows.append(row)
                    if trajectory.observation_kinds[index] == "post_fission":
                        post_fission_rows.append(row.copy())
    return (
        observation_rows,
        event_rows,
        preprocessing_rows,
        expanding_rows,
        post_fission_rows,
        partition_rows,
        labels,
        whole_rows,
        whole_local,
    )


def coverage_tables(
    analyses: list[BaselineAnalysis],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    molecular_rows: list[dict[str, Any]] = []
    generation_rows: list[dict[str, Any]] = []
    first_rows: list[dict[str, Any]] = []
    for preprocessing_id in PREPROCESSING_IDS:
        for analysis in analyses:
            trajectory = analysis.trajectory
            estimates = analysis.estimates[preprocessing_id]
            eligible = [
                index
                for index, estimate in enumerate(estimates)
                if estimate.status == "ELIGIBLE_NUMERIC_STRICT_EXPANDING"
            ]
            post = [
                index
                for index in eligible
                if trajectory.observation_kinds[index] == "post_fission"
            ]
            first_rows.append(
                {
                    "trajectoryId": trajectory.trajectory_id,
                    "matrixIndex": trajectory.matrix_index,
                    "preprocessingId": preprocessing_id,
                    "firstEligibleObservationIndex": eligible[0] if eligible else None,
                    "firstEligibleMolecularStep": (
                        int(trajectory.molecular_steps[eligible[0]])
                        if eligible
                        else None
                    ),
                    "firstEligibleGeneration": (
                        int(trajectory.generations[eligible[0]]) if eligible else None
                    ),
                    "firstEligiblePostFissionObservationIndex": post[0]
                    if post
                    else None,
                    "firstEligiblePostFissionMolecularStep": (
                        int(trajectory.molecular_steps[post[0]]) if post else None
                    ),
                    "firstEligiblePostFissionGeneration": (
                        int(trajectory.generations[post[0]]) if post else None
                    ),
                    "eligibleObservationCount": len(eligible),
                    "eligiblePostFissionCount": len(post),
                    "totalObservationCount": len(estimates),
                    "totalPostFissionCount": sum(
                        kind == "post_fission" for kind in trajectory.observation_kinds
                    ),
                }
            )
        max_step = max(
            int(analysis.trajectory.molecular_steps[-1]) for analysis in analyses
        )
        for step in range(max_step + 1):
            indicators: list[bool] = []
            observed = 0
            for analysis in analyses:
                trajectory = analysis.trajectory
                if step == 0:
                    indices = [0]
                else:
                    indices = [
                        index
                        for index, value in enumerate(trajectory.molecular_steps)
                        if int(value) == step
                        and trajectory.observation_kinds[index] == "molecular_event"
                    ]
                if indices:
                    observed += 1
                    indicators.append(
                        analysis.estimates[preprocessing_id][indices[0]].status
                        == "ELIGIBLE_NUMERIC_STRICT_EXPANDING"
                    )
                else:
                    indicators.append(False)
            molecular_rows.append(
                {
                    "preprocessingId": preprocessing_id,
                    "molecularStep": step,
                    "eligibleTrajectoryCount": sum(indicators),
                    "observedTrajectoryCount": observed,
                    "trajectoryCountR": 12,
                    "coverageC": sum(indicators) / 12,
                }
            )
        for generation in range(1, 101):
            indicators = []
            for analysis in analyses:
                trajectory = analysis.trajectory
                indices = [
                    index
                    for index, kind in enumerate(trajectory.observation_kinds)
                    if kind == "post_fission"
                    and int(trajectory.generations[index]) == generation
                ]
                indicators.append(
                    bool(indices)
                    and analysis.estimates[preprocessing_id][indices[0]].status
                    == "ELIGIBLE_NUMERIC_STRICT_EXPANDING"
                )
            generation_rows.append(
                {
                    "preprocessingId": preprocessing_id,
                    "generation": generation,
                    "eligibleTrajectoryCount": sum(indicators),
                    "trajectoryCountR": 12,
                    "coverageC": sum(indicators) / 12,
                }
            )
    return molecular_rows, generation_rows, first_rows


def suppression_rows(expanding_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(expanding_rows)
    grouped = (
        frame.assign(reason=frame["reason"].fillna("NONE_ELIGIBLE"))
        .groupby(["preprocessingId", "redundancyId", "status", "reason"], dropna=False)
        .size()
        .reset_index(name="rowCount")
    )
    return grouped.to_dict("records")


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float | None, str | None]:
    if x.size < 2:
        return None, "FEWER_THAN_TWO_PAIRS"
    if np.unique(x).size < 2:
        return None, "CONSTANT_PHI_R"
    if np.unique(y).size < 2:
        return None, "CONSTANT_LABEL"
    value = float(spearmanr(x, y).statistic)
    if not np.isfinite(value):
        return None, "NONFINITE_SPEARMAN"
    return value, None


def association_analysis(
    analyses: list[BaselineAnalysis],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    aggregate_summary: dict[str, Any] = {}
    estimands = (
        ("continuing_replication", "historical", 0),
        ("later_replication_one_generation", "historical", 1),
        ("past_only_companion_continuing", "online", 0),
    )
    for preprocessing_index, preprocessing_id in enumerate(PREPROCESSING_IDS):
        for redundancy_index, redundancy_id in enumerate(REDUNDANCY_IDS):
            for estimand_index, (estimand, label_family, lead) in enumerate(estimands):
                branch_values: list[float] = []
                paired_arrays: list[tuple[np.ndarray, np.ndarray]] = []
                for analysis in analyses:
                    trajectory = analysis.trajectory
                    labels = (
                        analysis.historical_labels
                        if label_family == "historical"
                        else analysis.online_labels
                    )
                    x: list[float] = []
                    y: list[float] = []
                    for index, kind in enumerate(trajectory.observation_kinds):
                        if kind != "post_fission":
                            continue
                        estimate = analysis.estimates[preprocessing_id][index]
                        if estimate.status != "ELIGIBLE_NUMERIC_STRICT_EXPANDING":
                            continue
                        generation = int(trajectory.generations[index])
                        outcome = labels.get(generation + lead)
                        if outcome is None:
                            continue
                        assert estimate.value is not None
                        x.append(float(estimate.value))
                        y.append(float(bool(outcome)))
                    x_array = np.asarray(x, dtype=np.float64)
                    y_array = np.asarray(y, dtype=np.float64)
                    coefficient, reason = _safe_spearman(x_array, y_array)
                    status = (
                        "ELIGIBLE"
                        if coefficient is not None and x_array.size >= 24
                        else "INELIGIBLE"
                    )
                    if coefficient is not None and x_array.size < 24:
                        reason = "FEWER_THAN_24_ELIGIBLE_POST_FISSION_PAIRS"
                        coefficient = None
                    rows.append(
                        {
                            "rowType": "trajectory",
                            "estimandId": estimand,
                            "labelFamily": label_family,
                            "preprocessingId": preprocessing_id,
                            "redundancyId": redundancy_id,
                            "trajectoryId": trajectory.trajectory_id,
                            "matrixIndex": trajectory.matrix_index,
                            "pairCount": int(x_array.size),
                            "status": status,
                            "reason": reason,
                            "spearmanRho": coefficient,
                        }
                    )
                    if coefficient is not None:
                        branch_values.append(coefficient)
                        paired_arrays.append((x_array, y_array))

                bundle = analysis_seed_bundle(
                    trajectory_id="E01-S12-ASSOCIATION",
                    replicate_index=(
                        preprocessing_index * 100
                        + redundancy_index * 10
                        + estimand_index
                    ),
                    namespace_tag="bootstrap-null",
                )
                rng = bundle.fresh_generators()[StreamPurpose.ESTIMATOR]
                if len(branch_values) >= 1:
                    coefficients = np.asarray(branch_values, dtype=np.float64)
                    observed = float(np.median(coefficients))
                    bootstrap = np.median(
                        rng.choice(
                            coefficients,
                            size=(4096, coefficients.size),
                            replace=True,
                        ),
                        axis=1,
                    )
                    lower, upper = np.quantile(bootstrap, [0.025, 0.975])
                    null_values = np.empty(4096, dtype=np.float64)
                    for permutation_index in range(4096):
                        permuted_coefficients: list[float] = []
                        for x_array, y_array in paired_arrays:
                            shift = int(rng.integers(1, y_array.size))
                            candidate, _ = _safe_spearman(
                                x_array, np.roll(y_array, shift)
                            )
                            if candidate is not None:
                                permuted_coefficients.append(candidate)
                        null_values[permutation_index] = (
                            float(np.median(permuted_coefficients))
                            if permuted_coefficients
                            else np.nan
                        )
                    valid_null = null_values[np.isfinite(null_values)]
                    permutation_p = (
                        (1 + int(np.sum(valid_null >= observed)))
                        / (1 + valid_null.size)
                        if valid_null.size
                        else None
                    )
                    status = "ELIGIBLE" if len(branch_values) >= 9 else "INELIGIBLE"
                    reason = (
                        None
                        if status == "ELIGIBLE"
                        else "FEWER_THAN_9_DEFINED_TRAJECTORIES"
                    )
                else:
                    observed = lower = upper = permutation_p = None
                    status = "INELIGIBLE"
                    reason = "NO_DEFINED_TRAJECTORY_STATISTIC"
                summary = {
                    "rowType": "summary",
                    "estimandId": estimand,
                    "labelFamily": label_family,
                    "preprocessingId": preprocessing_id,
                    "redundancyId": redundancy_id,
                    "trajectoryId": None,
                    "matrixIndex": None,
                    "pairCount": None,
                    "definedTrajectoryCount": len(branch_values),
                    "positiveTrajectoryCount": sum(
                        value > 0 for value in branch_values
                    ),
                    "status": status,
                    "reason": reason,
                    "spearmanRho": observed,
                    "bootstrapLower95": lower,
                    "bootstrapUpper95": upper,
                    "circularShiftPermutationPPositive": permutation_p,
                    "bootstrapReplicates": 4096,
                    "permutationReplicates": 4096,
                    "analysisSeedPayloadJson": json.dumps(
                        bundle.to_payload(), sort_keys=True, separators=(",", ":")
                    ),
                }
                rows.append(summary)
                aggregate_summary[
                    f"{estimand}::{preprocessing_id}::{redundancy_id}"
                ] = summary
    return rows, aggregate_summary


def whole_descriptive_analysis(
    whole_local_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    frame = pd.DataFrame(whole_local_rows)
    rows: list[dict[str, Any]] = []
    for (preprocessing_id, redundancy_id), branch in frame.groupby(
        ["preprocessingId", "redundancyId"]
    ):
        run_coefficients: list[float] = []
        for trajectory_id, trace in branch.groupby("trajectoryId"):
            valid = trace[
                (trace["status"] == "ELIGIBLE")
                & trace["historicalReplicator"].notna()
                & trace["value"].notna()
            ]
            coefficient, reason = _safe_spearman(
                valid["value"].to_numpy(dtype=np.float64),
                valid["historicalReplicator"].astype(float).to_numpy(),
            )
            if coefficient is not None:
                run_coefficients.append(coefficient)
            rows.append(
                {
                    "analysisType": "whole_local_label_association",
                    "scopeLabel": "DESCRIPTIVE_NONPROSPECTIVE",
                    "preprocessingId": preprocessing_id,
                    "redundancyId": redundancy_id,
                    "trajectoryId": trajectory_id,
                    "status": "ELIGIBLE" if coefficient is not None else "INELIGIBLE",
                    "reason": reason,
                    "value": coefficient,
                    "sampleCount": len(valid),
                }
            )
        progress_grid = np.linspace(0.0, 1.0, 1001, dtype=np.float64)
        interpolated: list[np.ndarray] = []
        complete_local_source = True
        for _, trace in branch.groupby("trajectoryId"):
            valid = trace[
                (trace["status"] == "ELIGIBLE") & trace["value"].notna()
            ].sort_values("transitionTargetObservationIndex")
            if len(valid) != len(trace):
                complete_local_source = False
                continue
            series = valid["value"].to_numpy(dtype=np.float64)
            source_grid = np.linspace(0.0, 1.0, series.size, dtype=np.float64)
            interpolated.append(np.interp(progress_grid, source_grid, series))
        complete_local_source &= len(interpolated) == 12
        if complete_local_source:
            aggregate_trace = np.mean(np.stack(interpolated), axis=0)
            regression = linregress(progress_grid, aggregate_trace)
            eligible_values = branch["value"].to_numpy(dtype=np.float64)
            overall_mean = float(np.mean(eligible_values))
            overall_sd = float(np.std(eligible_values, ddof=1))
            threshold = overall_mean + 3.0 * overall_sd
            spike_runs = int(
                sum(
                    bool(np.any(trace["value"].to_numpy(dtype=np.float64) > threshold))
                    for _, trace in branch.groupby("trajectoryId")
                )
            )
        else:
            regression = None
            threshold = None
            spike_runs = None
        rows.extend(
            [
                {
                    "analysisType": "whole_aggregate_linear_trend",
                    "scopeLabel": "DESCRIPTIVE_NONPROSPECTIVE",
                    "preprocessingId": preprocessing_id,
                    "redundancyId": redundancy_id,
                    "trajectoryId": None,
                    "status": "ELIGIBLE" if complete_local_source else "INELIGIBLE",
                    "reason": None
                    if complete_local_source
                    else "INCOMPLETE_PINNED_SOURCE_LOCAL_DECOMPOSITIONS",
                    "value": float(regression.slope)
                    if regression is not None
                    else None,
                    "pValue": float(regression.pvalue)
                    if regression is not None
                    else None,
                    "sampleCount": int(progress_grid.size)
                    if complete_local_source
                    else 0,
                    "trajectoryCount": len(interpolated),
                    "alignment": "normalized_transition_progress_linear_interpolation",
                },
                {
                    "analysisType": "whole_positive_three_sd_spike_runs",
                    "scopeLabel": "DESCRIPTIVE_NONPROSPECTIVE",
                    "preprocessingId": preprocessing_id,
                    "redundancyId": redundancy_id,
                    "trajectoryId": None,
                    "status": "ELIGIBLE" if complete_local_source else "INELIGIBLE",
                    "reason": None
                    if complete_local_source
                    else "INCOMPLETE_PINNED_SOURCE_LOCAL_DECOMPOSITIONS",
                    "value": spike_runs,
                    "threshold": threshold,
                    "sampleCount": 12 if complete_local_source else len(interpolated),
                },
                {
                    "analysisType": "whole_median_run_label_association",
                    "scopeLabel": "DESCRIPTIVE_NONPROSPECTIVE",
                    "preprocessingId": preprocessing_id,
                    "redundancyId": redundancy_id,
                    "trajectoryId": None,
                    "status": "ELIGIBLE" if run_coefficients else "INELIGIBLE",
                    "reason": None if run_coefficients else "NO_DEFINED_RUNS",
                    "value": (
                        float(np.median(run_coefficients)) if run_coefficients else None
                    ),
                    "positiveRunCount": sum(value > 0 for value in run_coefficients),
                    "sampleCount": len(run_coefficients),
                },
            ]
        )
        raw_p_values: list[float] = []
        differenced_p_values: list[float] = []
        for trajectory_id, trace in branch.groupby("trajectoryId"):
            valid = trace[
                (trace["status"] == "ELIGIBLE") & trace["value"].notna()
            ].sort_values("transitionTargetObservationIndex")
            series = valid["value"].to_numpy(dtype=np.float64)
            if series.size > 20:
                lag = min(20, series.size // 5)
                p_value = float(
                    acorr_ljungbox(series, lags=[lag], return_df=True)[
                        "lb_pvalue"
                    ].iloc[0]
                )
                rows.append(
                    {
                        "analysisType": "whole_ljung_box_raw",
                        "scopeLabel": "DESCRIPTIVE_NONPROSPECTIVE",
                        "preprocessingId": preprocessing_id,
                        "redundancyId": redundancy_id,
                        "trajectoryId": trajectory_id,
                        "status": "ELIGIBLE",
                        "reason": None,
                        "value": p_value,
                        "lag": lag,
                        "sampleCount": series.size,
                    }
                )
                raw_p_values.append(p_value)
            else:
                rows.append(
                    {
                        "analysisType": "whole_ljung_box_raw",
                        "scopeLabel": "DESCRIPTIVE_NONPROSPECTIVE",
                        "preprocessingId": preprocessing_id,
                        "redundancyId": redundancy_id,
                        "trajectoryId": trajectory_id,
                        "status": "INELIGIBLE",
                        "reason": "PINNED_SOURCE_LOCAL_SERIES_UNAVAILABLE_OR_TOO_SHORT",
                        "value": None,
                        "lag": None,
                        "sampleCount": series.size,
                    }
                )
            differenced = np.diff(series)
            if differenced.size > 20:
                differenced_lag = min(20, differenced.size // 5)
                differenced_p = float(
                    acorr_ljungbox(
                        differenced,
                        lags=[differenced_lag],
                        return_df=True,
                    )["lb_pvalue"].iloc[0]
                )
                rows.append(
                    {
                        "analysisType": "whole_ljung_box_differenced",
                        "scopeLabel": "DESCRIPTIVE_NONPROSPECTIVE",
                        "preprocessingId": preprocessing_id,
                        "redundancyId": redundancy_id,
                        "trajectoryId": trajectory_id,
                        "status": "ELIGIBLE",
                        "reason": None,
                        "value": differenced_p,
                        "lag": differenced_lag,
                        "sampleCount": differenced.size,
                    }
                )
                differenced_p_values.append(differenced_p)
            else:
                rows.append(
                    {
                        "analysisType": "whole_ljung_box_differenced",
                        "scopeLabel": "DESCRIPTIVE_NONPROSPECTIVE",
                        "preprocessingId": preprocessing_id,
                        "redundancyId": redundancy_id,
                        "trajectoryId": trajectory_id,
                        "status": "INELIGIBLE",
                        "reason": "PINNED_SOURCE_LOCAL_SERIES_UNAVAILABLE_OR_TOO_SHORT",
                        "value": None,
                        "lag": None,
                        "sampleCount": differenced.size,
                    }
                )
        for analysis_type, values in (
            ("whole_ljung_box_raw_summary", raw_p_values),
            ("whole_ljung_box_differenced_summary", differenced_p_values),
        ):
            rows.append(
                {
                    "analysisType": analysis_type,
                    "scopeLabel": "DESCRIPTIVE_NONPROSPECTIVE",
                    "preprocessingId": preprocessing_id,
                    "redundancyId": redundancy_id,
                    "trajectoryId": None,
                    "status": "ELIGIBLE" if len(values) == 12 else "INELIGIBLE",
                    "reason": None
                    if len(values) == 12
                    else "INCOMPLETE_12_TRAJECTORIES",
                    "value": float(np.median(values)) if values else None,
                    "rejectionCountAtAlpha0p05": sum(value < 0.05 for value in values),
                    "sampleCount": len(values),
                }
            )
    return rows


def baseline_feasibility_gate(
    analyses: list[BaselineAnalysis],
    first_rows: list[dict[str, Any]],
    *,
    baseline_cpu_seconds: float,
    baseline_wall_seconds: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply the frozen all-or-none gate and benchmark every qualifying pilot."""

    first = pd.DataFrame(first_rows)
    qualifying: list[int] = []
    coverage_details: list[dict[str, Any]] = []
    for analysis in analyses:
        matrix_index = analysis.trajectory.matrix_index
        rows = first[first["matrixIndex"] == matrix_index]
        by_preprocessing = {
            row["preprocessingId"]: row for row in rows.to_dict("records")
        }
        requirements = []
        for preprocessing_id in PREPROCESSING_IDS:
            row = by_preprocessing[preprocessing_id]
            final = analysis.estimates[preprocessing_id][-1]
            requirements.append(
                row["eligiblePostFissionCount"] >= 24
                and analysis.locks[preprocessing_id].status == "ELIGIBLE_LOCKED"
                and final.status == "ELIGIBLE_NUMERIC_STRICT_EXPANDING"
            )
        passed = all(requirements)
        coverage_details.append(
            {
                "matrixIndex": matrix_index,
                "passedCoverageRequirements": passed,
                "eligiblePostFissionCounts": {
                    preprocessing_id: int(
                        by_preprocessing[preprocessing_id]["eligiblePostFissionCount"]
                    )
                    for preprocessing_id in PREPROCESSING_IDS
                },
                "firstEligibleGenerations": {
                    preprocessing_id: by_preprocessing[preprocessing_id][
                        "firstEligiblePostFissionGeneration"
                    ]
                    for preprocessing_id in PREPROCESSING_IDS
                },
            }
        )
        if passed:
            qualifying.append(matrix_index)

    generation_coverage: dict[str, float] = {}
    first_generations: list[float] = []
    for preprocessing_id in PREPROCESSING_IDS:
        final_count = sum(
            analysis.estimates[preprocessing_id][-1].status
            == "ELIGIBLE_NUMERIC_STRICT_EXPANDING"
            for analysis in analyses
        )
        generation_coverage[preprocessing_id] = final_count / 12
        values = first.loc[
            first["preprocessingId"] == preprocessing_id,
            "firstEligiblePostFissionGeneration",
        ].dropna()
        first_generations.extend(float(value) for value in values)
    median_first_generation = (
        float(np.median(first_generations)) if first_generations else math.inf
    )

    pilot_rows: list[dict[str, Any]] = []
    pilot_seed_rows: list[dict[str, Any]] = []
    pilot_times: list[float] = []
    pilot_pass_indices: list[int] = []
    pilot_first_generations: dict[int, int] = {}
    for matrix_index in sorted(qualifying):
        analysis = analyses[matrix_index]
        common_indices = []
        for index, kind in enumerate(analysis.trajectory.observation_kinds):
            if kind != "post_fission":
                continue
            if all(
                analysis.estimates[preprocessing_id][index].status
                == "ELIGIBLE_NUMERIC_STRICT_EXPANDING"
                for preprocessing_id in PREPROCESSING_IDS
            ):
                common_indices.append(index)
        if not common_indices:
            pilot_rows.append(
                {
                    "matrixIndex": matrix_index,
                    "status": "INELIGIBLE",
                    "reason": "NO_COMMON_ELIGIBLE_POST_FISSION_POINT",
                }
            )
            continue
        index = common_indices[0]
        bundle = analysis_seed_bundle(
            trajectory_id=f"E01-S12-PILOT-{matrix_index:02d}",
            replicate_index=matrix_index,
            namespace_tag="candidate-replay-null",
        )
        pilot_seed_rows.extend(
            flatten_seed_payload(bundle.to_payload(), seed_role="candidate_pilot")
        )
        rng = bundle.fresh_generators()[StreamPurpose.INTERVENTION]
        started = time.perf_counter()
        prefix_coordinates = {
            preprocessing_id: analysis.preprocessing.coordinates[preprocessing_id][
                : index + 1
            ]
            for preprocessing_id in PREPROCESSING_IDS
        }
        scores = score_action_candidates(
            analysis.trajectory.states[index],
            preprocessing_coordinates=prefix_coordinates,
            locks=analysis.locks,
        )
        replay = score_action_candidates(
            analysis.trajectory.states[index],
            preprocessing_coordinates=prefix_coordinates,
            locks=analysis.locks,
        )
        replay_success = len(scores) == len(replay) and all(
            left["candidateId"] == right["candidateId"]
            and left["preprocessingId"] == right["preprocessingId"]
            and left["status"] == right["status"]
            and left["score"] == right["score"]
            for left, right in zip(scores, replay, strict=True)
        )
        candidate_success = all(
            row["status"] == "ELIGIBLE_NUMERIC_STRICT_EXPANDING" for row in scores
        )
        envelopes: dict[str, Any] = {}
        envelope_success = True
        for preprocessing_id in PREPROCESSING_IDS:
            subset = [
                row for row in scores if row["preprocessingId"] == preprocessing_id
            ]
            for direction in ("max", "min"):
                envelope = action_null_envelope(
                    subset, direction=direction, rng=rng, families=4096
                )
                envelopes[f"{preprocessing_id}::{direction}"] = envelope
                envelope_success &= envelope["status"] == "ELIGIBLE"
        elapsed = time.perf_counter() - started
        pilot_times.append(elapsed)
        passed = replay_success and candidate_success and envelope_success
        if passed:
            pilot_pass_indices.append(matrix_index)
            pilot_first_generations[matrix_index] = int(
                analysis.trajectory.generations[index]
            )
        pilot_rows.append(
            {
                "matrixIndex": matrix_index,
                "status": "PASS" if passed else "FAIL",
                "reason": None if passed else "CANDIDATE_PILOT_GATE_FAILURE",
                "observationIndex": index,
                "generation": int(analysis.trajectory.generations[index]),
                "molecularStep": int(analysis.trajectory.molecular_steps[index]),
                "candidateCount": len(scores) // 2,
                "candidateRows": len(scores),
                "allCandidatesStrictEligible": candidate_success,
                "exactReplay": replay_success,
                "allNullEnvelopesEligible": envelope_success,
                "runtimeSeconds": elapsed,
                "envelopes": envelopes,
            }
        )

    selected = sorted(pilot_pass_indices)[:6]
    remaining_decisions = sum(
        2 * (100 - pilot_first_generations[index] + 1) for index in selected
    )
    median_pilot = float(np.median(pilot_times)) if pilot_times else math.inf
    projected_intervention_cpu = median_pilot * remaining_decisions
    projected_cpu_hours = (baseline_cpu_seconds + projected_intervention_cpu) / 3600.0
    projected_wall_hours = (
        baseline_wall_seconds + projected_intervention_cpu / 8.0
    ) / 3600.0
    current_bytes = directory_bytes(STEP_ROOT) + directory_bytes(CACHE_ROOT)
    projected_bytes = max(current_bytes * 4, current_bytes + 512 * 1024 * 1024)
    free_bytes = shutil.disk_usage(STEP_ROOT).free

    checks = {
        "exactTwelveCompleteBaselines": len(analyses) == 12
        and all(
            analysis.trajectory.lineage.completed_fissions == 100
            for analysis in analyses
        ),
        "minimumSixQualifyingTrajectories": len(qualifying) >= 6,
        "finalCoverageAtLeast0p75Both": all(
            value >= 0.75 for value in generation_coverage.values()
        ),
        "medianFirstEligibleGenerationAtMost25": median_first_generation <= 25,
        "minimumSixCompleteCandidatePilots": len(pilot_pass_indices) >= 6,
        "observedBaselineCpuHoursAtMost20": baseline_cpu_seconds / 3600.0 <= 20,
        "projectedCompleteS12CpuHoursAtMost200": projected_cpu_hours <= 200,
        "projectedWallHoursAtMost48": projected_wall_hours <= 48,
        "projectedStorageAtMost20GiB": projected_bytes <= 20 * 1024**3,
        "freeSpaceAtLeast1p25Projection": free_bytes >= 1.25 * projected_bytes,
    }
    passed = all(checks.values())
    if passed and len(selected) != 6:
        raise RuntimeError("passing intervention gate did not select exactly six")
    if not passed:
        selected = []
    result = {
        "schema": "eidosoma.e01.s12_intervention_feasibility_gate.v1",
        "researchStepId": "S12",
        "gateId": "E01-S12-BASELINE-FEASIBILITY-INTERVENTION-v1.0.0",
        "status": "PASS_RUN_EXACTLY_SIX_TRIPLETS"
        if passed
        else "FAIL_RUN_ZERO_TRIPLETS",
        "success": passed,
        "checks": checks,
        "qualifyingMatrixIndices": sorted(qualifying),
        "candidatePilotPassingMatrixIndices": sorted(pilot_pass_indices),
        "selectedMatrixIndices": selected,
        "selectionRule": "first_six_ascending_from_candidate_pilot_passing_qualifiers",
        "coverageDetails": coverage_details,
        "finalCoverage": generation_coverage,
        "medianFirstEligibleGeneration": median_first_generation,
        "observedBaselineCpuHours": baseline_cpu_seconds / 3600.0,
        "observedBaselineWallHours": baseline_wall_seconds / 3600.0,
        "medianCompletePilotSeconds": median_pilot,
        "projectedRemainingTreatedDecisionPoints": remaining_decisions,
        "projectedCompleteS12CpuHours": projected_cpu_hours,
        "projectedCompleteS12WallHours": projected_wall_hours,
        "currentBytes": current_bytes,
        "projectedBytes": projected_bytes,
        "freeBytes": free_bytes,
        "pilotResults": pilot_rows,
        "firstEligiblePilotGenerationByMatrix": pilot_first_generations,
    }
    return result, pilot_rows, pilot_seed_rows


def intervention_outputs(
    interventions: list[InterventionTrajectory],
    analyses: list[BaselineAnalysis],
    gate: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    observation_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    partition_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    label_rows: list[dict[str, Any]] = []
    expanding_rows: list[dict[str, Any]] = []
    phi_summary_rows: list[dict[str, Any]] = []

    by_pair: dict[tuple[int, str], InterventionTrajectory] = {
        (item.matrix_index, item.condition): item for item in interventions
    }
    for trajectory in interventions:
        seed_rows.extend(
            flatten_seed_payload(
                trajectory.seed_payload, seed_role="intervention_trajectory"
            )
        )
        action_by_generation = {
            row["generation"]: row for row in trajectory.action_rows
        }
        for index, state in enumerate(trajectory.states):
            generation = int(trajectory.generations[index])
            action = (
                action_by_generation.get(generation)
                if trajectory.observation_kinds[index] == "post_fission"
                else None
            )
            observation_rows.append(
                {
                    "trajectoryId": trajectory.trajectory_id,
                    "matrixIndex": trajectory.matrix_index,
                    "condition": trajectory.condition,
                    "observationIndex": index,
                    "observationKind": trajectory.observation_kinds[index],
                    "generation": generation,
                    "molecularStep": int(trajectory.molecular_steps[index]),
                    "mass": int(np.sum(state)),
                    "state": state.tolist(),
                    "actionStatus": action["status"] if action else None,
                    "selectedCandidateId": (
                        action.get("selectedCandidateId") if action else None
                    ),
                }
            )
        for event in intervention_event_rows(trajectory):
            event_rows.append(
                {
                    "trajectoryId": event["trajectoryId"],
                    "matrixIndex": event["matrixIndex"],
                    "condition": event["condition"],
                    "recordType": event["recordType"],
                    "recordPayloadJson": json.dumps(
                        _jsonable(event["recordPayload"]),
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                }
            )
        for row in trajectory.candidate_rows:
            candidate_rows.append(
                {
                    **row,
                    "reason": row.get("reason"),
                }
            )
        for row in trajectory.action_rows:
            action_rows.append(
                {
                    **{
                        key: value for key, value in row.items() if key != "diagnostics"
                    },
                    "diagnosticsJson": json.dumps(
                        _jsonable(row.get("diagnostics", [])),
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                }
            )
        partition_rows.extend(trajectory.partition_rows)

        preprocessing = preprocess_states(trajectory.states)
        for preprocessing_id in PREPROCESSING_IDS:
            lock = trajectory.partition_locks.get(preprocessing_id)
            if lock is None:
                lock = analyses[trajectory.matrix_index].locks[preprocessing_id]
                # A control trajectory is an exact baseline replay. No treated
                # trajectory may borrow the baseline partition.
                if trajectory.condition != "control":
                    lock = type(lock)(
                        status="INELIGIBLE",
                        reason="PARTITION_NOT_LOCKED_ON_TREATED_TRAJECTORY",
                        preprocessing_id=preprocessing_id,
                        observation_index=None,
                        generation=None,
                        molecular_step=None,
                        part_a=None,
                        part_b=None,
                        partition_id=None,
                        objective=None,
                        relative_eigengap=None,
                        minimum_side_fraction=None,
                        replay_maximum_objective_error=None,
                        replay_minimum_ari=None,
                        history=(),
                    )
            estimates = expanding_estimates(
                preprocessing.coordinates[preprocessing_id], lock
            )
            for index, estimate in enumerate(estimates):
                base = {
                    "specificationId": "E01-S12-STRICT-MRR-v1.0.0",
                    "trajectoryId": trajectory.trajectory_id,
                    "matrixIndex": trajectory.matrix_index,
                    "condition": trajectory.condition,
                    "observationIndex": index,
                    "observationKind": trajectory.observation_kinds[index],
                    "molecularStep": int(trajectory.molecular_steps[index]),
                    "generation": int(trajectory.generations[index]),
                    "preprocessingId": preprocessing_id,
                    "partitionBranchId": "E01-S12-PARTITION-PASTONLY-FIRST-PASS-LOCK-v1.0.0",
                    "partitionId": (
                        lock.partition_id
                        if lock.observation_index is not None
                        and index >= lock.observation_index
                        else None
                    ),
                    "estimatorId": "E01-S10-PHYID-GAUSSIAN-STRICT-v1.0.0",
                    "aggregateId": "E01-S10-AGG-PAPER-EQUATION-v1.0.0",
                    "tau": 1,
                    "nEff": estimate.n_eff,
                    "status": estimate.status,
                    "reason": estimate.reason,
                    "value": estimate.value,
                    "numericalRank": estimate.numerical_rank,
                    "rankTolerance": estimate.rank_tolerance,
                    "conditionNumber": estimate.condition_number,
                    "minimumEigenvalue": estimate.minimum_eigenvalue,
                    "latticeClosureError": estimate.lattice_closure_error,
                    "paperEquationClosureError": estimate.paper_equation_closure_error,
                    "totalMutualInformation": estimate.total_mutual_information,
                    "scopeLabel": "PROSPECTIVE_PAST_ONLY_EXPANDING",
                }
                for redundancy_id in REDUNDANCY_IDS:
                    expanding_rows.append(
                        {
                            **base,
                            "redundancyId": redundancy_id,
                            "atomEvaluationPolicy": (
                                "ANALYTIC_MMI_MEAN_ATOMS_EVERY_STEP"
                                if redundancy_id.endswith("MMI-v1.0.0")
                                else "CCS_SOURCE_AT_FROZEN_CHECKPOINTS_NO_ATOM_IMPUTATION"
                            ),
                        }
                    )

        post_indices = [
            index
            for index, kind in enumerate(trajectory.observation_kinds)
            if kind == "post_fission"
        ]
        from e01_replicator_labels import historical_technique1_labels

        labels = historical_technique1_labels(
            trajectory.states[post_indices],
            trajectory_id=trajectory.trajectory_id,
            observation_ids=tuple(f"generation-{g:03d}" for g in range(1, 101)),
            configuration_id="E01-S08-YH-T1-HGT090-v1.0.0",
            threshold=0.9,
            evidence_class="PINNED_PUBLIC_HISTORICAL_SOURCE_BEHAVIOR",
        )
        for generation, record in enumerate(labels.rows, start=1):
            payload = record.as_dict()
            payload.update(
                {
                    "matrixIndex": trajectory.matrix_index,
                    "condition": trajectory.condition,
                    "generation": generation,
                    "labelBranch": "historical",
                }
            )
            label_rows.append(payload)

    labels_frame = pd.DataFrame(label_rows)
    g0_by_matrix = {
        int(key): int(value)
        for key, value in gate["firstEligiblePilotGenerationByMatrix"].items()
        if int(key) in gate["selectedMatrixIndices"]
    }
    for matrix_index in gate["selectedMatrixIndices"]:
        g0 = g0_by_matrix[matrix_index]
        for condition in ("max", "control", "min"):
            subset = labels_frame[
                (labels_frame["matrixIndex"] == matrix_index)
                & (labels_frame["condition"] == condition)
                & (labels_frame["generation"] >= g0 + 1)
                & (labels_frame["generation"] <= 100)
            ]
            valid = subset[subset["isReplicator"].notna()]
            trajectory = by_pair[(matrix_index, condition)]
            applied = sum(
                row["status"] == "ELIGIBLE_ACTION_APPLIED"
                for row in trajectory.action_rows
            )
            suppressed = sum(
                row["status"] == "INELIGIBLE_ACTION_NOT_SEPARABLE"
                for row in trajectory.action_rows
            )
            metric_rows.append(
                {
                    "matrixIndex": matrix_index,
                    "condition": condition,
                    "commonRiskOriginGenerationG0": g0,
                    "commonRiskGenerationCount": 100 - g0,
                    "validHistoricalLabelCount": len(valid),
                    "historicalReplicatorCount": int(
                        valid["isReplicator"].astype(bool).sum()
                    ),
                    "historicalReplicatorFraction": (
                        float(valid["isReplicator"].astype(bool).mean())
                        if len(valid)
                        else None
                    ),
                    "separableActionsApplied": applied,
                    "notSeparableSuppressions": suppressed,
                    "completedFissions": trajectory.completed_fissions,
                    "status": (
                        "ELIGIBLE"
                        if len(valid) == 100 - g0
                        and trajectory.completed_fissions == 100
                        else "INELIGIBLE"
                    ),
                }
            )

            trajectory_phi = [
                row
                for row in expanding_rows
                if row["matrixIndex"] == matrix_index
                and row["condition"] == condition
                and row["observationKind"] == "post_fission"
                and row["generation"] >= g0 + 1
                and row["generation"] <= 100
            ]
            for preprocessing_id in PREPROCESSING_IDS:
                for redundancy_id in REDUNDANCY_IDS:
                    branch = [
                        row
                        for row in trajectory_phi
                        if row["preprocessingId"] == preprocessing_id
                        and row["redundancyId"] == redundancy_id
                    ]
                    eligible_values = [
                        float(row["value"])
                        for row in branch
                        if row["status"] == "ELIGIBLE_NUMERIC_STRICT_EXPANDING"
                        and row["value"] is not None
                    ]
                    expected = 100 - g0
                    phi_summary_rows.append(
                        {
                            "matrixIndex": matrix_index,
                            "condition": condition,
                            "commonRiskOriginGenerationG0": g0,
                            "preprocessingId": preprocessing_id,
                            "redundancyId": redundancy_id,
                            "expectedPostFissionCount": expected,
                            "eligiblePostFissionCount": len(eligible_values),
                            "meanStrictExpandingPhiR": (
                                float(np.mean(eligible_values))
                                if eligible_values
                                else None
                            ),
                            "status": (
                                "ELIGIBLE"
                                if len(branch) == expected
                                and len(eligible_values) == expected
                                else "INELIGIBLE_INCOMPLETE_COMMON_RISK_ESTIMATES"
                            ),
                            "scopeLabel": "PROSPECTIVE_POST_ELIGIBILITY_COMMON_RISK",
                        }
                    )

    metric_frame = pd.DataFrame(metric_rows)
    contrast_rows: list[dict[str, Any]] = []
    contrast_seed_rows: list[dict[str, Any]] = []
    for contrast_index, (left, right) in enumerate(
        (("max", "control"), ("control", "min"))
    ):
        left_values = (
            metric_frame[metric_frame["condition"] == left]
            .sort_values("matrixIndex")["historicalReplicatorCount"]
            .to_numpy(dtype=np.float64)
        )
        right_values = (
            metric_frame[metric_frame["condition"] == right]
            .sort_values("matrixIndex")["historicalReplicatorCount"]
            .to_numpy(dtype=np.float64)
        )
        differences = left_values - right_values
        bundle = analysis_seed_bundle(
            trajectory_id="E01-S12-INTERVENTION-CONTRAST",
            replicate_index=contrast_index,
            namespace_tag=f"{left}-vs-{right}",
        )
        contrast_seed_rows.extend(
            flatten_seed_payload(bundle.to_payload(), seed_role="intervention_contrast")
        )
        rng = bundle.fresh_generators()[StreamPurpose.ESTIMATOR]
        bootstrap = np.mean(
            rng.choice(differences, size=(4096, differences.size), replace=True),
            axis=1,
        )
        lower, upper = np.quantile(bootstrap, [0.025, 0.975])
        sign_means = []
        for mask in range(2**differences.size):
            signs = np.asarray(
                [
                    1.0 if mask & (1 << index) else -1.0
                    for index in range(differences.size)
                ]
            )
            sign_means.append(float(np.mean(differences * signs)))
        observed = float(np.mean(differences))
        exact_p = sum(value >= observed for value in sign_means) / len(sign_means)
        contrast_rows.append(
            {
                "contrast": f"{left}_minus_{right}",
                "pairCount": differences.size,
                "meanDifference": observed,
                "medianDifference": float(np.median(differences)),
                "positivePairCount": int(np.sum(differences > 0)),
                "zeroPairCount": int(np.sum(differences == 0)),
                "negativePairCount": int(np.sum(differences < 0)),
                "bootstrapLower95": float(lower),
                "bootstrapUpper95": float(upper),
                "exactOneSidedSignFlipP": float(exact_p),
                "bootstrapReplicates": 4096,
                "exactSignFlipCount": 2**differences.size,
            }
        )
    metric_rows.extend(contrast_rows)

    pairing: dict[str, Any] = {
        "schema": "eidosoma.e01.s12_intervention_pairing_audit.v1",
        "triplets": [],
        "success": True,
    }
    for matrix_index in gate["selectedMatrixIndices"]:
        max_run = by_pair[(matrix_index, "max")]
        control = by_pair[(matrix_index, "control")]
        min_run = by_pair[(matrix_index, "min")]
        baseline = analyses[matrix_index].trajectory
        g0 = int(gate["firstEligiblePilotGenerationByMatrix"][matrix_index])
        g0_indices = [
            index
            for index, kind in enumerate(control.observation_kinds)
            if kind == "post_fission" and int(control.generations[index]) == g0
        ]
        if len(g0_indices) != 1:
            raise RuntimeError("common-risk origin did not map to one decision point")
        g0_observation_index = g0_indices[0]
        common_beta = np.array_equal(max_run.beta, control.beta) and np.array_equal(
            min_run.beta, control.beta
        )
        common_initial = (
            max_run.initial_state == control.initial_state == min_run.initial_state
        )
        control_replay = np.array_equal(control.states, baseline.states)
        common_prefix = np.array_equal(
            max_run.states[: g0_observation_index + 1],
            control.states[: g0_observation_index + 1],
        ) and np.array_equal(
            min_run.states[: g0_observation_index + 1],
            control.states[: g0_observation_index + 1],
        )
        treatments: dict[str, Any] = {}
        for candidate in (max_run, min_run):
            no_applied_before_g0 = not any(
                row["generation"] < g0 and row["status"] == "ELIGIBLE_ACTION_APPLIED"
                for row in candidate.action_rows
            )
            first_action = next(
                (
                    row
                    for row in candidate.action_rows
                    if row["status"] == "ELIGIBLE_ACTION_APPLIED"
                    and row.get("selectedCandidateId") != "noop"
                ),
                None,
            )
            minimum = min(candidate.states.shape[0], control.states.shape[0])
            differences = np.flatnonzero(
                np.any(candidate.states[:minimum] != control.states[:minimum], axis=1)
            )
            first_state = int(differences[0]) if differences.size else None
            prefix_identical = (
                True
                if first_action is None
                else not differences.size
                or first_state > int(first_action["observationIndex"])
            )
            treatments[candidate.condition] = {
                "firstAppliedActionGeneration": (
                    first_action["generation"] if first_action else None
                ),
                "firstAppliedActionObservationIndex": (
                    first_action["observationIndex"] if first_action else None
                ),
                "noAppliedActionBeforeCommonRiskOrigin": no_applied_before_g0,
                "firstStateDivergenceObservationIndex": first_state,
                "firstRawDrawNoLongerSemanticallyAligned": (
                    {
                        "generation": first_action["generation"],
                        "nextGrowthEventDrawOneBased": 1,
                    }
                    if first_action
                    else None
                ),
                "stateIdenticalThroughDecisionObservation": prefix_identical,
            }
        success = (
            common_beta
            and common_initial
            and control_replay
            and common_prefix
            and all(
                item["stateIdenticalThroughDecisionObservation"]
                and item["noAppliedActionBeforeCommonRiskOrigin"]
                for item in treatments.values()
            )
        )
        pairing["success"] &= success
        pairing["triplets"].append(
            {
                "matrixIndex": matrix_index,
                "commonBeta": common_beta,
                "commonInitialState": common_initial,
                "controlExactlyReplaysBaselineObservations": control_replay,
                "commonRiskOriginGenerationG0": g0,
                "commonRiskOriginObservationIndex": g0_observation_index,
                "allConditionsStateIdenticalThroughCommonRiskOrigin": common_prefix,
                "treatments": treatments,
                "success": success,
            }
        )
    return (
        observation_rows,
        event_rows,
        candidate_rows,
        action_rows,
        partition_rows,
        pairing,
        metric_rows,
        seed_rows + contrast_seed_rows,
        label_rows,
        expanding_rows,
        phi_summary_rows,
    )


def classify_claims(
    association_summary: dict[str, Any],
    whole_descriptive: list[dict[str, Any]],
    gate: dict[str, Any],
    intervention_metrics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Assign every S01 claim exactly one frozen S12 vocabulary status."""

    ledger = pd.read_csv(CLAIM_LEDGER)
    allowed = {
        "SUPPORTED",
        "DIRECTIONALLY_SUPPORTED",
        "NOT_SUPPORTED_WITHIN_STRICT_SCOPE",
        "UNDERDETERMINED",
        "NOT_EVALUATED",
    }

    def association_status(estimand: str) -> tuple[str, str]:
        summaries = [
            row
            for key, row in association_summary.items()
            if key.startswith(f"{estimand}::")
        ]
        # MMI and CCS scalar rows are intentionally identical; require both
        # preprocessing representations, not duplicate redundancy identities.
        by_preprocessing: dict[str, dict[str, Any]] = {}
        for row in summaries:
            by_preprocessing.setdefault(row["preprocessingId"], row)
        if set(by_preprocessing) != set(PREPROCESSING_IDS):
            return "UNDERDETERMINED", "one or both preprocessing branches absent"
        rows = list(by_preprocessing.values())
        adequate = all(
            row["definedTrajectoryCount"] >= 9 and row["status"] == "ELIGIBLE"
            for row in rows
        )
        directional = adequate and all(
            row["positiveTrajectoryCount"] >= 9
            and row["spearmanRho"] is not None
            and row["spearmanRho"] > 0
            and row["bootstrapLower95"] is not None
            and row["bootstrapLower95"] > 0
            and row["circularShiftPermutationPPositive"] is not None
            and row["circularShiftPermutationPPositive"] <= 0.05
            for row in rows
        )
        if directional:
            return (
                "DIRECTIONALLY_SUPPORTED",
                "both frozen representations passed the restricted positive association rule",
            )
        not_supported = adequate and all(
            row["bootstrapUpper95"] is not None and row["bootstrapUpper95"] <= 0
            for row in rows
        )
        if not_supported:
            return (
                "NOT_SUPPORTED_WITHIN_STRICT_SCOPE",
                "both adequate post-eligibility intervals had upper bounds at or below zero",
            )
        return (
            "UNDERDETERMINED",
            "restricted association did not meet either frozen directional or precise non-support rule",
        )

    continuing_status, continuing_reason = association_status("continuing_replication")

    whole_frame = pd.DataFrame(whole_descriptive)
    trend = whole_frame[whole_frame["analysisType"] == "whole_aggregate_linear_trend"]
    trend_status = (
        "DIRECTIONALLY_SUPPORTED"
        if len(trend) == 4
        and bool((trend["status"] == "ELIGIBLE").all())
        and bool((trend["pValue"] > 0.05).all())
        else (
            "NOT_SUPPORTED_WITHIN_STRICT_SCOPE"
            if len(trend) == 4 and bool((trend["status"] == "ELIGIBLE").all())
            else "UNDERDETERMINED"
        )
    )
    raw_temporal = whole_frame[
        whole_frame["analysisType"] == "whole_ljung_box_raw_summary"
    ]
    differenced_temporal = whole_frame[
        whole_frame["analysisType"] == "whole_ljung_box_differenced_summary"
    ]
    raw_majority_status = (
        "DIRECTIONALLY_SUPPORTED"
        if len(raw_temporal) == 4
        and bool((raw_temporal["status"] == "ELIGIBLE").all())
        and bool((raw_temporal["rejectionCountAtAlpha0p05"] >= 7).all())
        else (
            "NOT_SUPPORTED_WITHIN_STRICT_SCOPE"
            if len(raw_temporal) == 4
            and bool((raw_temporal["status"] == "ELIGIBLE").all())
            else "UNDERDETERMINED"
        )
    )
    raw_median_status = (
        "DIRECTIONALLY_SUPPORTED"
        if len(raw_temporal) == 4
        and bool((raw_temporal["status"] == "ELIGIBLE").all())
        and bool((raw_temporal["value"] < 0.05).all())
        else (
            "NOT_SUPPORTED_WITHIN_STRICT_SCOPE"
            if len(raw_temporal) == 4
            and bool((raw_temporal["status"] == "ELIGIBLE").all())
            else "UNDERDETERMINED"
        )
    )
    differenced_status = (
        "DIRECTIONALLY_SUPPORTED"
        if len(differenced_temporal) == 4
        and bool((differenced_temporal["status"] == "ELIGIBLE").all())
        and bool((differenced_temporal["rejectionCountAtAlpha0p05"] == 12).all())
        else (
            "NOT_SUPPORTED_WITHIN_STRICT_SCOPE"
            if len(differenced_temporal) == 4
            and bool((differenced_temporal["status"] == "ELIGIBLE").all())
            else "UNDERDETERMINED"
        )
    )

    intervention_status = "UNDERDETERMINED"
    intervention_reason = "frozen baseline feasibility gate did not authorize triplets"
    if gate["success"]:
        metrics = pd.DataFrame(intervention_metrics)
        contrasts = metrics[metrics.get("contrast", pd.Series(dtype=str)).notna()]
        conditions = metrics[metrics.get("condition", pd.Series(dtype=str)).notna()]
        if len(contrasts) == 2 and len(conditions) == 18:
            enough_actions = (
                sum(
                    conditions[
                        (conditions["matrixIndex"] == matrix_index)
                        & (conditions["condition"].isin(["max", "min"]))
                    ]["separableActionsApplied"].min()
                    >= 3
                    for matrix_index in gate["selectedMatrixIndices"]
                )
                >= 4
            )
            enough_triplets = (
                contrasts["positivePairCount"].min() >= 5
                and (contrasts["bootstrapLower95"] > 0).all()
            )
            if enough_actions and enough_triplets:
                intervention_status = "DIRECTIONALLY_SUPPORTED"
                intervention_reason = "both restricted paired contrasts passed the frozen direction and interval rule"
            elif (
                enough_actions
                and (contrasts["bootstrapUpper95"] <= 0).all()
                and (conditions["status"] == "ELIGIBLE").all()
            ):
                intervention_status = "NOT_SUPPORTED_WITHIN_STRICT_SCOPE"
                intervention_reason = "valid restricted paired intervals excluded the expected positive contrasts"
            else:
                intervention_reason = "triplets ran but action density or paired evidence did not meet a decisive frozen rule"

    always_underdetermined = {
        "E01-C014",
        "E01-C025",
        "E01-C026",
        "E01-C027",
        "E01-C028",
        "E01-C029",
        "E01-C030",
        "E01-C031",
        "E01-C032",
        "E01-C033",
        "E01-C034",
        "E01-C035",
        "E01-C036",
        "E01-C037",
        "E01-C038",
        "E01-C039",
        "E01-C040",
        "E01-C041",
        "E01-C042",
        "E01-C043",
        "E01-C044",
        "E01-C045",
        "E01-C047",
        "E01-C048",
        "E01-C049",
        "E01-C050",
        "E01-C051",
        "E01-C052",
        "E01-C053",
        "E01-C055",
        "E01-C056",
        "E01-C057",
    }
    intervention_direction_claims = {
        "E01-C046",
        "E01-C054",
        "E01-C058",
        "E01-C059",
    }
    association_claims = {
        "E01-C015",
        "E01-C016",
        "E01-C017",
        "E01-C018",
        "E01-C019",
        "E01-C020",
        "E01-C021",
    }
    rows: list[dict[str, Any]] = []
    for claim in ledger.to_dict("records"):
        claim_id = claim["claim_id"]
        if claim_id == "E01-C013":
            status = trend_status
            reason = "12-matrix completed-trajectory directional trend comparison only"
            evidence = "whole_descriptive_analysis.csv"
        elif claim_id == "E01-C022":
            status = raw_majority_status
            reason = (
                "12-matrix completed-trajectory raw Ljung-Box majority comparison only"
            )
            evidence = "whole_descriptive_analysis.csv"
        elif claim_id == "E01-C023":
            status = raw_median_status
            reason = "12-matrix completed-trajectory raw median-p directional comparison; the published exact value is not reproduced"
            evidence = "whole_descriptive_analysis.csv"
        elif claim_id == "E01-C024":
            status = differenced_status
            reason = "12-matrix completed-trajectory differenced Ljung-Box all-run comparison only"
            evidence = "whole_descriptive_analysis.csv"
        elif claim_id in association_claims:
            status = continuing_status
            reason = continuing_reason
            evidence = "association_results.csv"
        elif claim_id in intervention_direction_claims:
            status = intervention_status
            reason = intervention_reason
            evidence = "intervention_results.csv"
        elif claim_id in always_underdetermined:
            status = "UNDERDETERMINED"
            reason = (
                "claim requires unavailable fixed-window, pre-eligibility, early-warning, "
                "early/every-fission intervention, or exact Figure 6/Table 1 scope"
            )
            evidence = "preregistration.yaml"
        else:
            status = "NOT_EVALUATED"
            reason = "claim family is outside the bounded S12 strict MRR"
            evidence = None
        if status not in allowed:
            raise RuntimeError(f"invalid claim status {status}")
        rows.append(
            {
                "claimId": claim_id,
                "claimFamily": claim["claim_family"],
                "claimText": claim["claim_text"],
                "status": status,
                "restrictedEstimand": (
                    "E01-S12 strict post-eligibility or descriptive scope"
                    if status != "NOT_EVALUATED"
                    else None
                ),
                "reason": reason,
                "evidenceArtifact": evidence,
                "authorImplementationIdentity": "UNAVAILABLE::NO_AUTHOR_CODE_RELEASE_FOUND",
                "fixedWindowRecovered": False,
            }
        )
    if len(rows) != 59 or {row["status"] for row in rows} - allowed:
        raise RuntimeError("claim status matrix failed completeness/vocabulary")
    return rows


def plot_coverage(
    molecular_rows: list[dict[str, Any]], generation_rows: list[dict[str, Any]]
) -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    for rows, x_field, filename, xlabel in (
        (
            molecular_rows,
            "molecularStep",
            "eligibility_coverage_molecular_step.png",
            "Molecular step",
        ),
        (
            generation_rows,
            "generation",
            "eligibility_coverage_generation.png",
            "Generation",
        ),
    ):
        frame = pd.DataFrame(rows)
        fig, axis = plt.subplots(figsize=(8, 4.5))
        for preprocessing_id, branch in frame.groupby("preprocessingId"):
            label = "dropped CLR" if "DROPCLR" in preprocessing_id else "Helmert ILR"
            axis.plot(branch[x_field], branch["coverageC"], label=label, linewidth=1.8)
        axis.axhline(
            0.75, color="black", linestyle="--", linewidth=1, label="gate 0.75"
        )
        axis.set(xlabel=xlabel, ylabel="Eligibility coverage C(t)", ylim=(-0.02, 1.02))
        axis.legend(frameon=False)
        axis.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(FIGURE_ROOT / filename, dpi=180)
        plt.close(fig)


def plot_expanding_examples(
    analyses: list[BaselineAnalysis],
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=False)
    for axis, matrix_index in zip(axes, (0, 5, 11), strict=True):
        analysis = analyses[matrix_index]
        trajectory = analysis.trajectory
        for preprocessing_id in PREPROCESSING_IDS:
            values = np.asarray(
                [
                    estimate.value
                    if estimate.status == "ELIGIBLE_NUMERIC_STRICT_EXPANDING"
                    else np.nan
                    for estimate in analysis.estimates[preprocessing_id]
                ],
                dtype=np.float64,
            )
            label = "dropped CLR" if "DROPCLR" in preprocessing_id else "Helmert ILR"
            axis.plot(trajectory.molecular_steps, values, linewidth=0.8, label=label)
        axis.set_title(f"Matrix {matrix_index:02d}: strict expanding, past-only")
        axis.set_ylabel("Phi-r (nats)")
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("Molecular step")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / "strict_expanding_examples.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("--workers must be in 1..8")
    required_thread_environment = {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    actual_threads = {key: os.environ.get(key) for key in required_thread_environment}
    if actual_threads != required_thread_environment:
        raise RuntimeError(
            f"numeric thread environment is not frozen: {actual_threads}"
        )
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 0")

    import cupy as cp

    gpu_properties = cp.cuda.runtime.getDeviceProperties(0)
    gpu_name = gpu_properties["name"]
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode("utf-8")
    gpu_query = subprocess.run(
        [
            "nvidia-smi",
            "--id=0",
            "--query-gpu=uuid,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    gpu_uuid, gpu_driver = [value.strip() for value in gpu_query.split(",")]
    expected_gpu_uuid = prereg_gpu_uuid = yaml.safe_load(
        PREREG.read_text(encoding="utf-8")
    )["runtimeAndPrecision"]["gpu"]["uuid"]
    if gpu_uuid != expected_gpu_uuid:
        raise RuntimeError(
            f"GPU UUID {gpu_uuid} differs from preregistered {prereg_gpu_uuid}"
        )

    overall_started = time.perf_counter()
    STEP_ROOT.mkdir(parents=True, exist_ok=True)
    BUNDLE_ROOT.mkdir(parents=True, exist_ok=True)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    preflight = immutable_preflight()
    prereg = yaml.safe_load(PREREG.read_text(encoding="utf-8"))
    implementation_commit = git_output("rev-parse", "HEAD")
    remote_commit = git_output(
        "ls-remote", "origin", "refs/heads/eidosoma/groups/42"
    ).split()[0]
    if implementation_commit != remote_commit:
        raise RuntimeError("implementation HEAD is not the pushed branch head")
    if git_output("status", "--short"):
        raise RuntimeError("S12 full run requires a clean committed worktree")

    baseline_started = time.perf_counter()
    baselines: list[Any] = []
    simulation_cpu_seconds = 0.0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_simulate_baseline_worker, index): index
            for index in range(12)
        }
        for future in as_completed(futures):
            trajectory, runtime = future.result()
            baselines.append(trajectory)
            simulation_cpu_seconds += runtime
    baselines.sort(key=lambda item: item.matrix_index)
    if [item.matrix_index for item in baselines] != list(range(12)):
        raise RuntimeError("baseline cardinality or identities are incomplete")
    if any(
        item.lineage.completed_fissions != 100
        or item.lineage.terminal_status != "requested_generations_completed"
        for item in baselines
    ):
        raise RuntimeError("one or more baselines did not complete 100 fissions")

    analyses: list[BaselineAnalysis] = []
    analysis_cpu_seconds = 0.0
    for trajectory in baselines:
        analysis = analyze_baseline(trajectory)
        analyses.append(analysis)
        analysis_cpu_seconds += analysis.runtime_seconds
    baseline_wall_seconds = time.perf_counter() - baseline_started

    replay_started = time.perf_counter()
    replay_rows: list[dict[str, Any]] = []
    for matrix_index in (0, 5, 11):
        regenerated = simulate_baseline(matrix_index)
        original = baselines[matrix_index]
        success = (
            regenerated.trajectory_sha256 == original.trajectory_sha256
            and np.array_equal(regenerated.beta, original.beta)
            and np.array_equal(regenerated.states, original.states)
        )
        replay_rows.append(
            {
                "matrixIndex": matrix_index,
                "trajectoryId": original.trajectory_id,
                "expectedTrajectorySha256": original.trajectory_sha256,
                "regeneratedTrajectorySha256": regenerated.trajectory_sha256,
                "matrixExact": bool(np.array_equal(regenerated.beta, original.beta)),
                "statesExact": bool(
                    np.array_equal(regenerated.states, original.states)
                ),
                "success": success,
            }
        )
    replay_seconds = time.perf_counter() - replay_started
    if not all(row["success"] for row in replay_rows):
        raise RuntimeError("baseline same-engine regeneration failed")

    (
        observation_rows,
        event_rows,
        preprocessing_rows,
        expanding_rows,
        post_fission_rows,
        partition_rows,
        label_rows,
        whole_rows,
        whole_local_rows,
    ) = baseline_rows(analyses)
    molecular_coverage, generation_coverage, first_rows = coverage_tables(analyses)
    suppressions = suppression_rows(expanding_rows)
    association_rows, association_summary = association_analysis(analyses)
    whole_descriptive = whole_descriptive_analysis(whole_local_rows)
    numerical_rows = [row for analysis in analyses for row in analysis.numerical_rows]

    np.savez_compressed(
        STEP_ROOT / "baseline_matrices.npz",
        **{f"matrix_{item.matrix_index:02d}": item.beta for item in baselines},
    )
    write_parquet(STEP_ROOT / "baseline_trajectory_events.parquet", event_rows)
    write_parquet(STEP_ROOT / "baseline_observations.parquet", observation_rows)
    write_parquet(STEP_ROOT / "preprocessing_status.parquet", preprocessing_rows)
    write_parquet(STEP_ROOT / "replicator_labels.parquet", label_rows)
    write_parquet(STEP_ROOT / "expanding_estimates.parquet", expanding_rows)
    write_parquet(STEP_ROOT / "post_fission_estimates.parquet", post_fission_rows)
    write_parquet(STEP_ROOT / "partition_history.parquet", partition_rows)
    write_csv(STEP_ROOT / "first_eligibility.csv", first_rows)
    write_csv(
        STEP_ROOT / "eligibility_coverage_by_molecular_step.csv",
        molecular_coverage,
    )
    write_csv(
        STEP_ROOT / "eligibility_coverage_by_generation.csv",
        generation_coverage,
    )
    write_csv(STEP_ROOT / "suppression_summary.csv", suppressions)
    write_parquet(STEP_ROOT / "whole_trajectory_estimates.parquet", whole_rows)
    write_parquet(STEP_ROOT / "whole_trajectory_local_values.parquet", whole_local_rows)
    write_csv(STEP_ROOT / "association_results.csv", association_rows)
    write_csv(STEP_ROOT / "whole_descriptive_analysis.csv", whole_descriptive)

    trajectory_manifest = {
        "schema": "eidosoma.e01.s12_trajectory_manifest.v1",
        "researchStepId": "S12",
        "preregistrationVersion": prereg["preregistrationVersion"],
        "implementationCommit": implementation_commit,
        "baselineCount": 12,
        "trajectories": [
            {
                "trajectoryId": item.trajectory_id,
                "matrixIndex": item.matrix_index,
                "trajectorySha256": item.trajectory_sha256,
                "matrixSha256": hashlib.sha256(
                    item.beta.astype("<f8", copy=False).tobytes()
                ).hexdigest(),
                "observationCount": int(item.states.shape[0]),
                "molecularEventCount": len(item.lineage.events),
                "fissionCount": len(item.lineage.fissions),
                "completedFissions": item.lineage.completed_fissions,
                "terminalStatus": item.lineage.terminal_status,
                "initialMass": int(sum(item.lineage.initial_state)),
                "finalMass": int(sum(item.lineage.final_state)),
            }
            for item in baselines
        ],
    }
    write_json(STEP_ROOT / "trajectory_manifest.json", trajectory_manifest)
    seed_rows: list[dict[str, Any]] = []
    for analysis in analyses:
        seed_rows.extend(
            flatten_seed_payload(
                analysis.trajectory.seed_payload, seed_role="baseline_trajectory"
            )
        )
        for payload in analysis.analysis_seed_payloads:
            seed_rows.extend(
                flatten_seed_payload(payload, seed_role="baseline_analysis")
            )

    baseline_cpu_seconds = (
        simulation_cpu_seconds + analysis_cpu_seconds + replay_seconds
    )
    gate, pilot_rows, pilot_seed_rows = baseline_feasibility_gate(
        analyses,
        first_rows,
        baseline_cpu_seconds=baseline_cpu_seconds,
        baseline_wall_seconds=baseline_wall_seconds,
    )
    seed_rows.extend(pilot_seed_rows)
    write_json(STEP_ROOT / "intervention_feasibility_gate.json", gate)
    write_json(STEP_ROOT / "candidate_pilot_results.json", pilot_rows)

    interventions: list[InterventionTrajectory] = []
    intervention_started = time.perf_counter()
    if gate["success"]:
        with ProcessPoolExecutor(max_workers=min(args.workers, 6)) as executor:
            futures = {
                executor.submit(
                    _simulate_triplet_worker,
                    matrix_index,
                    gate["firstEligiblePilotGenerationByMatrix"][matrix_index],
                ): matrix_index
                for matrix_index in gate["selectedMatrixIndices"]
            }
            for future in as_completed(futures):
                interventions.extend(future.result())
        interventions.sort(key=lambda item: (item.matrix_index, item.condition))
        if len(interventions) != 18:
            raise RuntimeError("passing gate must yield exactly six complete triplets")
        if any(item.completed_fissions != 100 for item in interventions):
            raise RuntimeError(
                "an intervention condition did not complete 100 fissions"
            )
    intervention_wall_seconds = time.perf_counter() - intervention_started
    intervention_cpu_seconds = sum(item.runtime_seconds for item in interventions)

    if interventions:
        (
            intervention_observations,
            intervention_events,
            intervention_candidates,
            intervention_actions,
            intervention_partitions,
            pairing_audit,
            intervention_metrics,
            intervention_seed_rows,
            intervention_labels,
            intervention_expanding,
            intervention_phi_summary,
        ) = intervention_outputs(interventions, analyses, gate)
        seed_rows.extend(intervention_seed_rows)
    else:
        intervention_observations = []
        intervention_events = []
        intervention_candidates = []
        intervention_actions = []
        intervention_partitions = []
        intervention_metrics = []
        intervention_labels = []
        intervention_expanding = []
        intervention_phi_summary = []
        pairing_audit = {
            "schema": "eidosoma.e01.s12_intervention_pairing_audit.v1",
            "status": "NOT_RUN_BASELINE_FEASIBILITY_GATE_FAILED",
            "success": True,
            "triplets": [],
        }
    write_parquet(
        STEP_ROOT / "intervention_trajectories.parquet", intervention_observations
    )
    write_parquet(STEP_ROOT / "intervention_event_logs.parquet", intervention_events)
    write_parquet(
        STEP_ROOT / "intervention_candidate_scores.parquet",
        intervention_candidates,
    )
    write_parquet(STEP_ROOT / "intervention_action_log.parquet", intervention_actions)
    write_parquet(
        STEP_ROOT / "intervention_partition_history.parquet",
        intervention_partitions,
    )
    write_parquet(STEP_ROOT / "intervention_labels.parquet", intervention_labels)
    write_parquet(
        STEP_ROOT / "intervention_expanding_estimates.parquet",
        intervention_expanding,
    )
    write_csv(STEP_ROOT / "intervention_results.csv", intervention_metrics)
    write_csv(STEP_ROOT / "intervention_phi_summary.csv", intervention_phi_summary)
    write_json(STEP_ROOT / "intervention_pairing_audit.json", pairing_audit)
    write_json(
        STEP_ROOT / "intervention_branch_status.json",
        {
            "schema": "eidosoma.e01.s12_intervention_branch_status.v1",
            "researchStepId": "S12",
            "authorizedMaximumTriplets": 6,
            "gateStatus": gate["status"],
            "gatePassed": gate["success"],
            "selectedMatrixIndices": gate["selectedMatrixIndices"],
            "tripletsRun": len(interventions) // 3,
            "conditionsRun": len(interventions),
            "status": (
                "COMPLETED_EXACTLY_SIX_TRIPLETS"
                if interventions
                else "NOT_RUN_BASELINE_FEASIBILITY_GATE_FAILED"
            ),
            "reason": (
                None
                if interventions
                else "FROZEN_BASELINE_FEASIBILITY_GATE_DID_NOT_PASS"
            ),
        },
    )

    claim_rows = classify_claims(
        association_summary, whole_descriptive, gate, intervention_metrics
    )
    write_csv(STEP_ROOT / "claim_status_matrix.csv", claim_rows)
    write_parquet(STEP_ROOT / "seed_manifest.parquet", seed_rows)

    plot_coverage(molecular_coverage, generation_coverage)
    plot_expanding_examples(analyses)

    numerical_validation = {
        "schema": "eidosoma.e01.s12_numerical_validation.v1",
        "researchStepId": "S12",
        "checkpointCount": len(numerical_rows),
        "expectedCheckpointCount": 36,
        "allPassed": len(numerical_rows) == 36
        and all(row["status"] == "PASS" for row in numerical_rows),
        "maximumRunningVsPhyidAbsoluteError": max(
            (
                row["runningVsPhyidAbsoluteError"]
                for row in numerical_rows
                if row["runningVsPhyidAbsoluteError"] is not None
            ),
            default=None,
        ),
        "maximumPhyidVsOmegaCpuAbsoluteError": max(
            (
                row["phyidVsOmegaCpuMaximumAbsoluteError"]
                for row in numerical_rows
                if row["phyidVsOmegaCpuMaximumAbsoluteError"] is not None
            ),
            default=None,
        ),
        "maximumOmegaCpuVsGpuAbsoluteError": max(
            (
                row["omegaCpuVsGpuMaximumAbsoluteError"]
                for row in numerical_rows
                if row["omegaCpuVsGpuMaximumAbsoluteError"] is not None
            ),
            default=None,
        ),
        "absoluteTolerance": NUMERIC_TOLERANCE,
        "rows": numerical_rows,
    }
    write_json(STEP_ROOT / "numerical_validation.json", numerical_validation)
    regeneration_validation = {
        "schema": "eidosoma.e01.s12_regeneration_validation.v1",
        "researchStepId": "S12",
        "selectionRule": "matrix_indices_0_5_11_frozen_in_preregistration",
        "success": all(row["success"] for row in replay_rows),
        "rows": replay_rows,
    }
    write_json(STEP_ROOT / "regeneration_validation.json", regeneration_validation)

    complete_runtime_seconds = time.perf_counter() - overall_started
    runtime_manifest = {
        "schema": "eidosoma.e01.s12_runtime_manifest.v1",
        "researchStepId": "S12",
        "implementationCommit": implementation_commit,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "workerProcesses": args.workers,
        "numericalLibraryThreads": 1,
        "threadEnvironment": actual_threads,
        "cudaVisibleDevices": os.environ["CUDA_VISIBLE_DEVICES"],
        "cupy": cp.__version__,
        "gpu": {
            "name": gpu_name,
            "uuid": gpu_uuid,
            "driver": gpu_driver,
            "visibleDeviceCount": cp.cuda.runtime.getDeviceCount(),
            "cudaRuntimeVersion": cp.cuda.runtime.runtimeGetVersion(),
            "precision": "float64",
            "tf32": False,
            "mixedPrecision": False,
        },
        "simulationCpuSeconds": simulation_cpu_seconds,
        "analysisCpuSeconds": analysis_cpu_seconds,
        "regenerationWallAsCpuSeconds": replay_seconds,
        "interventionCpuSeconds": intervention_cpu_seconds,
        "totalMeasuredTaskCpuHours": (baseline_cpu_seconds + intervention_cpu_seconds)
        / 3600.0,
        "baselineWallHours": baseline_wall_seconds / 3600.0,
        "interventionWallHours": intervention_wall_seconds / 3600.0,
        "completeWallHours": complete_runtime_seconds / 3600.0,
        "cpuCeilingHours": 200.0,
        "overallE01CpuCeilingHours": 250.0,
        "wallCeilingHours": 48.0,
        "runtimeCeilingsPassed": (baseline_cpu_seconds + intervention_cpu_seconds)
        / 3600.0
        <= 200.0
        and complete_runtime_seconds / 3600.0 <= 48.0,
    }
    write_json(STEP_ROOT / "runtime_manifest.json", runtime_manifest)

    storage_validation = {
        "schema": "eidosoma.e01.s12_storage_validation.v1",
        "researchStepId": "S12",
        "stepArtifactBytes": directory_bytes(STEP_ROOT),
        "cacheBytes": directory_bytes(CACHE_ROOT),
        "combinedBytes": directory_bytes(STEP_ROOT) + directory_bytes(CACHE_ROOT),
        "byteCeiling": 20 * 1024**3,
        "success": directory_bytes(STEP_ROOT) + directory_bytes(CACHE_ROOT)
        <= 20 * 1024**3,
        "forbiddenArtifactCacheEntries": [
            path.relative_to(STEP_ROOT).as_posix()
            for path in STEP_ROOT.rglob("*")
            if path.is_file()
            and (
                "__pycache__" in path.parts
                or path.suffix in {".pyc", ".pyo", ".so", ".o"}
            )
        ],
    }
    storage_validation["success"] &= not storage_validation[
        "forbiddenArtifactCacheEntries"
    ]
    write_json(STEP_ROOT / "storage_validation.json", storage_validation)

    postflight = immutable_preflight()
    immutable_input_audit = {
        "schema": "eidosoma.e01.s12_immutable_input_audit.v1",
        "researchStepId": "S12",
        "preRun": preflight,
        "postRun": postflight,
        "success": all(
            branch["success"]
            for phase in (preflight, postflight)
            for branch in phase.values()
        ),
        "scope": "S10_S11_S11R_artifacts_and_source_contracts_plus_all_frozen_S12_inputs",
    }
    write_json(STEP_ROOT / "immutable_input_audit.json", immutable_input_audit)

    failure_rows: list[dict[str, Any]] = []
    attempt_failure = STEP_ROOT / "execution_attempt_001_failure.json"
    if attempt_failure.is_file():
        failure_rows.append(
            {
                "failureFamily": "execution_attempt",
                "scopeId": "E01-S12-STRICT-MRR-EXECUTION-ATTEMPT-001",
                "status": "FAILED_CLOSED_PRESERVED",
                "reason": "PINNED_PHYID_NONFINITE_DECOMPOSITION_ABORTED_INITIAL_ORCHESTRATION",
                "count": 1,
                "retained": True,
            }
        )
    for row in suppressions:
        if row["status"] != "ELIGIBLE_NUMERIC_STRICT_EXPANDING":
            failure_rows.append(
                {
                    "failureFamily": "baseline_estimate_suppression",
                    "scopeId": f"{row['preprocessingId']}::{row['redundancyId']}",
                    "status": row["status"],
                    "reason": row["reason"],
                    "count": row["rowCount"],
                    "retained": True,
                }
            )
    if not gate["success"]:
        for check, passed in gate["checks"].items():
            if not passed:
                failure_rows.append(
                    {
                        "failureFamily": "intervention_feasibility_gate",
                        "scopeId": gate["gateId"],
                        "status": "FAIL",
                        "reason": check,
                        "count": 1,
                        "retained": True,
                    }
                )
    for row in intervention_actions:
        if str(row.get("status", "")).startswith("INELIGIBLE"):
            failure_rows.append(
                {
                    "failureFamily": "intervention_action_suppression",
                    "scopeId": row["trajectoryId"],
                    "status": row["status"],
                    "reason": row.get("reason"),
                    "count": 1,
                    "retained": True,
                }
            )
    whole_source_failures = [
        row for row in whole_rows if row.get("sourceAtomStatus") != "ELIGIBLE"
    ]
    if whole_source_failures:
        normalized_source_failures = [
            {
                **row,
                "sourceAtomReason": row.get("sourceAtomReason") or row.get("reason"),
            }
            for row in whole_source_failures
        ]
        for (preprocessing_id, redundancy_id, reason), group in pd.DataFrame(
            normalized_source_failures
        ).groupby(
            ["preprocessingId", "redundancyId", "sourceAtomReason"], dropna=False
        ):
            failure_rows.append(
                {
                    "failureFamily": "whole_trajectory_source_atom_crosscheck",
                    "scopeId": f"{preprocessing_id}::{redundancy_id}",
                    "status": "INELIGIBLE",
                    "reason": reason,
                    "count": len(group),
                    "retained": True,
                }
            )
    for row in numerical_rows:
        if row["status"] != "PASS":
            failure_rows.append(
                {
                    "failureFamily": "source_numerical_checkpoint",
                    "scopeId": (
                        f"{row['trajectoryId']}::{row['preprocessingId']}::"
                        f"{row['redundancyId']}::{row['checkpoint']}"
                    ),
                    "status": row["status"],
                    "reason": row.get("reason"),
                    "count": 1,
                    "retained": True,
                }
            )
    write_csv(STEP_ROOT / "failure_ledger.csv", failure_rows)

    baseline_complete = len(observation_rows) == sum(
        item.states.shape[0] for item in baselines
    )
    estimate_expected = len(observation_rows) * 4
    estimate_complete = len(expanding_rows) == estimate_expected
    validation_checks = {
        "preregistrationAndAmendment": preflight["base"]["success"]
        and preflight["amendment"]["success"]
        and preflight["amendment2"]["success"],
        "immutableInputsStillExactAfterRun": immutable_input_audit["success"],
        "exactTwelveBaselines": len(baselines) == 12,
        "allBaselinesComplete100Fissions": all(
            item.lineage.completed_fissions == 100 for item in baselines
        ),
        "baselineObservationCompleteness": baseline_complete,
        "everyObservationFourStatusRows": estimate_complete,
        "sameEngineRegeneration": regeneration_validation["success"],
        "numericalCrosschecks": numerical_validation["allPassed"],
        "wholeStrictScalarCompleteness": len(whole_rows) == 48
        and all(row["status"] == "ELIGIBLE_NUMERIC_STRICT_WHOLE" for row in whole_rows),
        "wholePinnedSourceAtomCompleteness": len(whole_rows) == 48
        and all(row.get("sourceAtomStatus") == "ELIGIBLE" for row in whole_rows),
        "preprocessingFiniteInverse": all(
            row["finite"] and row["maximumAbsoluteInverseError"] <= NUMERIC_TOLERANCE
            for row in preprocessing_rows
        ),
        "claimMatrixCompleteVocabulary": len(claim_rows) == 59,
        "interventionCountRule": len(interventions) in {0, 18}
        and (
            (not gate["success"] and len(interventions) == 0)
            or len(interventions) == 18
        ),
        "interventionPairing": pairing_audit["success"],
        "interventionEstimateCompleteness": (
            len(intervention_expanding)
            == sum(item.states.shape[0] for item in interventions) * 4
        ),
        "runtimeCeilings": runtime_manifest["runtimeCeilingsPassed"],
        "storage": storage_validation["success"],
    }
    validation_summary = {
        "schema": "eidosoma.e01.s12_validation_summary.v1",
        "researchStepId": "S12",
        "stepNumber": 12,
        "success": all(validation_checks.values()),
        "status": (
            "COMPUTATION_VALIDATION_PASS"
            if all(validation_checks.values())
            else "COMPUTATION_VALIDATION_FAIL"
        ),
        "checks": validation_checks,
        "counts": {
            "baselineMatrices": len(baselines),
            "baselineObservations": len(observation_rows),
            "baselineEventAndFissionRows": len(event_rows),
            "expandingStatusRows": len(expanding_rows),
            "postFissionStatusRows": len(post_fission_rows),
            "interventionTriplets": len(interventions) // 3,
            "interventionConditions": len(interventions),
            "interventionCandidateRows": len(intervention_candidates),
            "interventionExpandingStatusRows": len(intervention_expanding),
            "claimRows": len(claim_rows),
            "seedRows": len(seed_rows),
        },
        "frozenScopeStatement": (
            "E01-S12-STRICT-MRR-v1.0.0 cannot recover the failed fixed-window "
            "scope and is not an exact author-implementation, Figure 6, or Table 1 replication."
        ),
    }
    write_json(STEP_ROOT / "validation_summary.json", validation_summary)
    write_json(
        BUNDLE_ROOT / "bundle_pointer.json",
        {
            "schema": "eidosoma.e01.s12_bundle_pointer.v1",
            "researchStepId": "S12",
            "canonicalArtifactRoot": str(STEP_ROOT),
            "trajectoryManifest": str(STEP_ROOT / "trajectory_manifest.json"),
            "implementationCommit": implementation_commit,
            "preregistrationSha256": sha256_file(PREREG),
            "amendmentSha256": sha256_file(AMENDMENT),
            "amendment2Sha256": sha256_file(AMENDMENT_2),
        },
    )
    run_summary = {
        "researchStepId": "S12",
        "stepNumber": 12,
        "implementationCommit": implementation_commit,
        "validationSuccess": validation_summary["success"],
        "validationChecks": validation_checks,
        "baselineGate": gate,
        "firstEligibility": first_rows,
        "associationSummaries": [
            row for row in association_rows if row["rowType"] == "summary"
        ],
        "wholeDescriptive": whole_descriptive,
        "interventionMetrics": intervention_metrics,
        "claimStatusCounts": pd.Series([row["status"] for row in claim_rows])
        .value_counts()
        .to_dict(),
        "runtimeManifest": runtime_manifest,
        "storageValidation": storage_validation,
    }
    write_json(STEP_ROOT / "run_summary.json", run_summary)
    print(json.dumps(_jsonable(run_summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
