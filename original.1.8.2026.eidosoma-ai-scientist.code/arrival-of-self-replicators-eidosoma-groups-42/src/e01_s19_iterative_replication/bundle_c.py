"""Locked Bundle C spike timing, spacing, and height reconstruction."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from e01_s19_iterative_replication.core import (
    CANDIDATE_IDS,
    PERMUTATION_REPLICATES,
    all_pair_mean_distance,
    correlation_inference,
    excursion_episodes,
    partial_spearman,
)

SPIKE_SPECIFICATIONS = (
    "C_GLOBAL_POOLED_3SD_RAW",
    "C_WITHIN_RUN_3SD_NORMALIZED",
)
TEMPORAL_SOURCE_MODES = ("COMPLETED_FIT", "PAST_ONLY_PREFIX")


def _series_tables(
    full_source_values: pd.DataFrame, prefix_endpoint_values: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    full = full_source_values.loc[
        full_source_values["implementationId"].eq("PHIRL_REGULARIZED_SOURCE")
        & full_source_values["status"].eq("ELIGIBLE")
        & full_source_values["emergence"].notna()
    ].copy()
    full = full.rename(
        columns={
            "selectedSequenceIndex": "sequenceIndex",
            "rawObservationIndex": "rawIndex",
        }
    )
    full["temporalSourceMode"] = "COMPLETED_FIT"
    prefix = prefix_endpoint_values.loc[
        prefix_endpoint_values["status"].isin(
            ["ELIGIBLE", "ELIGIBLE_PARTIAL_NONFINITE_LOCAL_VALUES"]
        )
        & prefix_endpoint_values["emergence"].notna()
    ].copy()
    prefix = prefix.rename(
        columns={
            "endpointSelectedSequenceIndex": "sequenceIndex",
            "endpointRawObservationIndex": "rawIndex",
        }
    )
    prefix["temporalSourceMode"] = "PAST_ONLY_PREFIX"
    columns = [
        "candidateId",
        "trajectoryId",
        "matrixIndex",
        "sequenceIndex",
        "rawIndex",
        "emergence",
        "temporalSourceMode",
    ]
    return {"COMPLETED_FIT": full[columns], "PAST_ONLY_PREFIX": prefix[columns]}


def run_outcomes(label_values: pd.DataFrame) -> pd.DataFrame:
    labels = label_values.loc[label_values["labelId"].eq("MOL_ADJACENT_INCOMING_H900")].copy()
    rows: list[dict[str, Any]] = []
    for (candidate_id, matrix_index), group in labels.groupby(["candidateId", "matrixIndex"], sort=True):
        group = group.sort_values("selectedSequenceIndex")
        target = group["isReplicator"].to_numpy(dtype=bool)
        h = group["labelScore"].to_numpy(dtype=np.float64)
        if not np.array_equal(target, h > 0.9):
            raise ValueError("Y != I(H>0.9) in Bundle C")
        rows.append(
            {
                "candidateId": candidate_id,
                "matrixIndex": int(matrix_index),
                "trajectoryId": group.iloc[0]["trajectoryId"],
                "replicationProbability": float(target.mean()),
                "positiveLabelCount": int(target.sum()),
                "trajectoryLength": len(target),
                "lastRawIndex": int(group["rawObservationIndex"].max()),
                "meanExactH": float(h.mean()),
                "exactHDeterminesLabel": True,
            }
        )
    return pd.DataFrame(rows)


def build_spike_descriptors(
    full_source_values: pd.DataFrame,
    prefix_endpoint_values: pd.DataFrame,
    label_values: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return run descriptors and event catalog under both locked definitions."""

    tables = _series_tables(full_source_values, prefix_endpoint_values)
    outcomes = run_outcomes(label_values)
    descriptor_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for temporal_mode, table in tables.items():
        for candidate_id in CANDIDATE_IDS:
            candidate = table.loc[table["candidateId"].eq(candidate_id)].copy()
            global_center = float(candidate["emergence"].mean()) if len(candidate) else math.nan
            global_scale = float(candidate["emergence"].std(ddof=0)) if len(candidate) else math.nan
            global_threshold = global_center + 3.0 * global_scale
            for matrix_index in range(100):
                outcome = outcomes.loc[
                    outcomes["candidateId"].eq(candidate_id)
                    & outcomes["matrixIndex"].eq(matrix_index)
                ].iloc[0]
                series = candidate.loc[candidate["matrixIndex"].eq(matrix_index)].sort_values("sequenceIndex")
                values = series["emergence"].to_numpy(dtype=np.float64)
                raw_indices = series["rawIndex"].to_numpy(dtype=np.float64)
                for specification in SPIKE_SPECIFICATIONS:
                    if specification == "C_GLOBAL_POOLED_3SD_RAW":
                        center, scale, threshold = global_center, global_scale, global_threshold
                    else:
                        center = float(np.mean(values)) if len(values) else math.nan
                        scale = float(np.std(values, ddof=0)) if len(values) else math.nan
                        threshold = center + 3.0 * scale
                    episodes = excursion_episodes(values, threshold) if len(values) and np.isfinite(threshold) else []
                    peak_local = np.asarray([episode.peak_position for episode in episodes], dtype=np.int64)
                    peak_raw = raw_indices[peak_local] if len(peak_local) else np.empty(0)
                    peak_values = values[peak_local] if len(peak_local) else np.empty(0)
                    if specification == "C_GLOBAL_POOLED_3SD_RAW":
                        time_values = peak_raw
                        height_values = peak_values
                    else:
                        denominator = max(1.0, float(outcome["lastRawIndex"]))
                        time_values = peak_raw / denominator
                        height_values = (
                            (peak_values - center) / scale
                            if len(peak_values) and np.isfinite(scale) and scale > 0
                            else np.full(len(peak_values), np.nan)
                        )
                    time_descriptor = float(np.mean(time_values)) if len(time_values) else None
                    spacing_descriptor = all_pair_mean_distance(time_values)
                    height_descriptor = float(np.mean(height_values)) if len(height_values) else None
                    descriptor_rows.append(
                        {
                            "candidateId": candidate_id,
                            "matrixIndex": matrix_index,
                            "trajectoryId": outcome["trajectoryId"],
                            "temporalSourceMode": temporal_mode,
                            "specificationId": specification,
                            "thresholdCenter": center if np.isfinite(center) else None,
                            "thresholdScale": scale if np.isfinite(scale) else None,
                            "threshold": threshold if np.isfinite(threshold) else None,
                            "sourceObservationCount": len(values),
                            "spikeCount": len(episodes),
                            "zeroSpike": len(episodes) == 0,
                            "oneSpike": len(episodes) == 1,
                            "spikeTime": time_descriptor,
                            "interSpikeDistance": spacing_descriptor,
                            "spikeHeight": height_descriptor,
                            "replicationProbability": float(outcome["replicationProbability"]),
                            "trajectoryLength": int(outcome["trajectoryLength"]),
                            "lastRawIndex": int(outcome["lastRawIndex"]),
                            "meanExactH": float(outcome["meanExactH"]),
                            "exactHDeterminesLabel": True,
                        }
                    )
                    for episode_number, (episode, local, raw, value) in enumerate(
                        zip(episodes, peak_local, peak_raw, peak_values), start=1
                    ):
                        event_rows.append(
                            {
                                "candidateId": candidate_id,
                                "matrixIndex": matrix_index,
                                "temporalSourceMode": temporal_mode,
                                "specificationId": specification,
                                "episodeNumber": episode_number,
                                "episodeStartSeriesPosition": episode.start,
                                "episodeEndSeriesPosition": episode.end,
                                "peakSeriesPosition": int(local),
                                "peakRawIndex": float(raw),
                                "peakEmergence": float(value),
                                "excessOverThreshold": float(value - threshold),
                            }
                        )
    return pd.DataFrame(descriptor_rows), pd.DataFrame(event_rows)


def correlation_results(descriptors: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    results: list[dict[str, Any]] = []
    robustness: list[dict[str, Any]] = []
    for (candidate_id, temporal_mode, specification), group in descriptors.groupby(
        ["candidateId", "temporalSourceMode", "specificationId"], sort=True
    ):
        group = group.sort_values("matrixIndex")
        zero_count = int(group["zeroSpike"].sum())
        one_count = int(group["oneSpike"].sum())
        for descriptor, claim_id in [
            ("spikeTime", "C031"),
            ("interSpikeDistance", "C032"),
            ("spikeHeight", "C033"),
        ]:
            x = group[descriptor].to_numpy(dtype=np.float64)
            y = group["replicationProbability"].to_numpy(dtype=np.float64)
            length = group["trajectoryLength"].to_numpy(dtype=np.float64)
            inference = correlation_inference(
                x,
                y,
                method="spearman",
                seed_identity=("bundle_c", candidate_id, temporal_mode, specification, descriptor),
                permutation_replicates=PERMUTATION_REPLICATES,
            )
            row = {
                "claimId": claim_id,
                "candidateId": candidate_id,
                "temporalSourceMode": temporal_mode,
                "specificationId": specification,
                "descriptorId": descriptor,
                "totalRunCount": len(group),
                "zeroSpikeRunCount": zero_count,
                "oneSpikeRunCount": one_count,
                **inference,
            }
            results.append(row)
            keep = np.isfinite(x) & np.isfinite(y)
            defined_x, defined_y, defined_length = x[keep], y[keep], length[keep]
            loo_values: list[float] = []
            if len(defined_x) >= 6 and np.ptp(defined_x) > 0 and np.ptp(defined_y) > 0:
                for excluded in range(len(defined_x)):
                    take = np.arange(len(defined_x)) != excluded
                    if np.ptp(defined_x[take]) > 0 and np.ptp(defined_y[take]) > 0:
                        loo_values.append(float(stats.spearmanr(defined_x[take], defined_y[take]).statistic))
            length_rho = (
                float(stats.spearmanr(defined_x, defined_length).statistic)
                if len(defined_x) >= 4 and np.ptp(defined_x) > 0 and np.ptp(defined_length) > 0
                else None
            )
            partial = partial_spearman(defined_x, defined_y, defined_length)
            low, high = np.quantile(defined_length, [0.05, 0.95]) if len(defined_length) else (math.nan, math.nan)
            middle = (defined_length >= low) & (defined_length <= high)
            middle_rho = (
                float(stats.spearmanr(defined_x[middle], defined_y[middle]).statistic)
                if middle.sum() >= 4 and np.ptp(defined_x[middle]) > 0 and np.ptp(defined_y[middle]) > 0
                else None
            )
            robustness.append(
                {
                    "claimId": claim_id,
                    "candidateId": candidate_id,
                    "temporalSourceMode": temporal_mode,
                    "specificationId": specification,
                    "descriptorId": descriptor,
                    "definedRunCount": int(keep.sum()),
                    "leaveOneOutMinimumRho": float(np.min(loo_values)) if loo_values else None,
                    "leaveOneOutMaximumRho": float(np.max(loo_values)) if loo_values else None,
                    "descriptorVsTrajectoryLengthRho": length_rho,
                    "partialSpearmanControllingLength": partial,
                    "middle90LengthRho": middle_rho,
                    "middle90DefinedRunCount": int(middle.sum()),
                }
            )
    return pd.DataFrame(results), pd.DataFrame(robustness)


__all__ = [
    "SPIKE_SPECIFICATIONS",
    "build_spike_descriptors",
    "correlation_results",
    "run_outcomes",
]
