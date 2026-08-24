"""Pure frozen-value analyses for E01/S15.

This module consumes only already-materialized S13Y information values and
labels.  It does not import a simulator or a PhiRL fitter.
"""

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import (
    binomtest,
    combine_pvalues,
    mannwhitneyu,
    pearsonr,
    rankdata,
    spearmanr,
    ttest_1samp,
    wilcoxon,
)

RESEARCH_STEP_ID = "S15"
VERSIONED_STEP_ID = "E01-S15-ASSOCIATION-REPLICATOR-STATE-ANALYSES-v1.0.0"
CANDIDATE_IDS = ("S12F-CANDIDATE-02", "S12F-CANDIDATE-03")
COMPLETED_MODE = "RETROSPECTIVE_FULL_TRAJECTORY_LOCAL"
PREFIX_MODE = "PAST_ONLY_PREFIX_ENDPOINT"
LEVEL_ANALYSIS = "LEVEL_ANALYSIS"
CHANGE_ANALYSIS = "CHANGE_ANALYSIS"
PRIMARY_BRANCH = "COMPLETED_FIT_PRIMARY"
HISTORICAL_BRANCH = "COMPLETED_FIT_HISTORICAL_COMPARATOR"
PREFIX_BRANCH = "PAST_ONLY_PREFIX_COMPARATOR"
PRIMARY_LABEL = "MOL_ADJACENT_INCOMING_H900"
HISTORICAL_LABEL = "HISTORICAL_H090_REPLICATOR"
POOLED_SCOPE = "POOLED_SECONDARY"
ALPHA = 0.05

IDENTITY_COLUMNS = [
    "branchId",
    "comparisonRole",
    "temporalMode",
    "labelId",
    "analysisId",
]
RUN_COLUMNS = ["candidateId", "matrixIndex", "trajectoryId"]


def derive_seed(root_hex: str, *tokens: str) -> int:
    """Derive a domain-separated 128-bit integer seed."""

    digest = hashlib.sha256()
    digest.update(bytes.fromhex(root_hex))
    for token in tokens:
        digest.update(b"\x00")
        digest.update(str(token).encode("utf-8"))
    return int.from_bytes(digest.digest()[:16], "big")


def _analysis_variants(base: pd.DataFrame) -> list[pd.DataFrame]:
    """Materialize the locked level and current-label first-difference rows."""

    levels = base.copy()
    levels["analysisId"] = LEVEL_ANALYSIS
    levels["analysisValue"] = levels["emergence"].astype(np.float64)
    levels["previousObservationOrder"] = np.nan

    changes = base.copy()
    groups = changes.groupby(RUN_COLUMNS, sort=False, observed=True)
    changes["analysisValue"] = groups["emergence"].diff()
    changes["previousObservationOrder"] = groups["observationOrder"].shift(1)
    changes = changes.loc[changes["analysisValue"].notna()].copy()
    changes["analysisId"] = CHANGE_ANALYSIS
    return [levels, changes]


def prepare_analysis_rows(full: pd.DataFrame, prefix: pd.DataFrame) -> pd.DataFrame:
    """Create the three frozen branches and both mandatory estimands."""

    full_required = {
        "candidateId",
        "trajectoryId",
        "matrixIndex",
        "selectedSequenceIndex",
        "rawObservationIndex",
        "status",
        "emergence",
        "incomingCosineH",
        "euclideanL2ClosedCompositionChange",
        "molecularH090Label",
        "historicalH090Label",
    }
    prefix_required = {
        "candidateId",
        "trajectoryId",
        "matrixIndex",
        "endpointSelectedSequenceIndex",
        "endpointRawObservationIndex",
        "status",
        "emergence",
        "currentIncomingCosineH",
        "currentMolecularH090Label",
    }
    missing_full = full_required - set(full.columns)
    missing_prefix = prefix_required - set(prefix.columns)
    if missing_full or missing_prefix:
        raise ValueError(
            f"missing completed={sorted(missing_full)} prefix={sorted(missing_prefix)}"
        )

    completed = full.loc[full["status"].eq("ELIGIBLE")].copy()
    completed = completed.sort_values(
        ["candidateId", "matrixIndex", "selectedSequenceIndex"], kind="stable"
    )
    common = pd.DataFrame(
        {
            "candidateId": completed["candidateId"],
            "matrixIndex": completed["matrixIndex"].astype(int),
            "trajectoryId": completed["trajectoryId"],
            "observationOrder": completed["selectedSequenceIndex"].astype(int),
            "rawObservationIndex": completed["rawObservationIndex"].astype(int),
            "emergence": completed["emergence"].astype(np.float64),
            "incomingCosineH": completed["incomingCosineH"].astype(np.float64),
            "negativeEuclideanL2ClosedCompositionChange": -completed[
                "euclideanL2ClosedCompositionChange"
            ].astype(np.float64),
        }
    )

    primary = common.copy()
    primary["branchId"] = PRIMARY_BRANCH
    primary["comparisonRole"] = "PRIMARY_PAPER_FACING_RETROSPECTIVE"
    primary["temporalMode"] = COMPLETED_MODE
    primary["labelId"] = PRIMARY_LABEL
    primary["label"] = completed["molecularH090Label"].astype(bool).to_numpy()

    historical = common.copy()
    historical["branchId"] = HISTORICAL_BRANCH
    historical["comparisonRole"] = "FROZEN_HISTORICAL_POST_FISSION_COMPARATOR"
    historical["temporalMode"] = COMPLETED_MODE
    historical["labelId"] = HISTORICAL_LABEL
    historical["label"] = completed["historicalH090Label"].astype(bool).to_numpy()

    eligible_prefix = prefix.loc[prefix["status"].eq("ELIGIBLE")].copy()
    eligible_prefix = eligible_prefix.sort_values(
        ["candidateId", "matrixIndex", "endpointSelectedSequenceIndex"],
        kind="stable",
    )
    past_only = pd.DataFrame(
        {
            "candidateId": eligible_prefix["candidateId"],
            "matrixIndex": eligible_prefix["matrixIndex"].astype(int),
            "trajectoryId": eligible_prefix["trajectoryId"],
            "observationOrder": eligible_prefix[
                "endpointSelectedSequenceIndex"
            ].astype(int),
            "rawObservationIndex": eligible_prefix[
                "endpointRawObservationIndex"
            ].astype(int),
            "emergence": eligible_prefix["emergence"].astype(np.float64),
            "incomingCosineH": eligible_prefix["currentIncomingCosineH"].astype(
                np.float64
            ),
            "negativeEuclideanL2ClosedCompositionChange": np.nan,
            "branchId": PREFIX_BRANCH,
            "comparisonRole": "PAST_ONLY_CURRENT_ENDPOINT_COMPARATOR",
            "temporalMode": PREFIX_MODE,
            "labelId": PRIMARY_LABEL,
            "label": eligible_prefix["currentMolecularH090Label"]
            .astype(bool)
            .to_numpy(),
        }
    )

    pieces: list[pd.DataFrame] = []
    for branch in (primary, historical, past_only):
        pieces.extend(_analysis_variants(branch))
    result = pd.concat(pieces, ignore_index=True)
    result["label"] = result["label"].astype(bool)
    result["analysisValue"] = result["analysisValue"].astype(np.float64)
    result.sort_values(
        IDENTITY_COLUMNS + RUN_COLUMNS + ["observationOrder"],
        kind="stable",
        inplace=True,
        ignore_index=True,
    )
    return result


def _safe_correlations(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    if len(x) < 3:
        return {"status": "INELIGIBLE_TOO_FEW_VALUES"}
    if np.unique(x).size < 2:
        return {"status": "INELIGIBLE_CONSTANT_ANALYSIS_VALUE"}
    if np.unique(y).size < 2:
        return {"status": "INELIGIBLE_CONSTANT_LABEL"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho = spearmanr(x, y)
        product = pearsonr(x, y)
    return {
        "status": "ELIGIBLE",
        "spearmanRho": float(rho.statistic),
        "spearmanTwoSidedP": float(rho.pvalue),
        "pearsonR": float(product.statistic),
        "pearsonTwoSidedP": float(product.pvalue),
    }


def _safe_mann_whitney(replication: np.ndarray, drift: np.ndarray) -> dict[str, Any]:
    if not len(replication) or not len(drift):
        return {"status": "INELIGIBLE_MISSING_STATE"}
    greater = mannwhitneyu(
        replication, drift, alternative="greater", method="asymptotic"
    )
    two_sided = mannwhitneyu(
        replication, drift, alternative="two-sided", method="asymptotic"
    )
    denominator = float(len(replication) * len(drift))
    return {
        "status": "ELIGIBLE",
        "mannWhitneyU": float(greater.statistic),
        "mannWhitneyGreaterP": float(greater.pvalue),
        "mannWhitneyTwoSidedP": float(two_sided.pvalue),
        "rankBiserialReplicatorGreater": float(
            2.0 * float(greater.statistic) / denominator - 1.0
        ),
    }


def runwise_statistics(
    analysis_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate frozen runwise correlations and state comparisons."""

    correlation_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    keys = IDENTITY_COLUMNS + RUN_COLUMNS
    for identity, group in analysis_rows.groupby(keys, sort=True, observed=True):
        metadata = dict(zip(keys, identity, strict=True))
        ordered = group.sort_values("observationOrder", kind="stable")
        x = ordered["analysisValue"].to_numpy(np.float64)
        y = ordered["label"].to_numpy(bool)
        correlations = _safe_correlations(x, y.astype(np.float64))
        correlation_rows.append(
            {
                **metadata,
                "n": len(x),
                "replicatorCount": int(np.count_nonzero(y)),
                "driftCount": int(np.count_nonzero(~y)),
                "replicatorPrevalence": float(np.mean(y)),
                "correlationStatus": correlations.pop("status"),
                **correlations,
            }
        )

        replication = x[y]
        drift = x[~y]
        mw = _safe_mann_whitney(replication, drift)
        if len(replication) and len(drift):
            rep_mean = float(np.mean(replication))
            drift_mean = float(np.mean(drift))
            rep_median = float(np.median(replication))
            drift_median = float(np.median(drift))
            state_values = {
                "replicatorMean": rep_mean,
                "driftMean": drift_mean,
                "meanDifference": rep_mean - drift_mean,
                "replicatorMedian": rep_median,
                "driftMedian": drift_median,
                "medianDifference": rep_median - drift_median,
            }
        else:
            state_values = {}
        state_rows.append(
            {
                **metadata,
                "n": len(x),
                "replicatorCount": len(replication),
                "driftCount": len(drift),
                "stateComparisonStatus": mw.pop("status"),
                **state_values,
                **mw,
            }
        )
    correlations = pd.DataFrame(correlation_rows).sort_values(
        keys, kind="stable", ignore_index=True
    )
    states = pd.DataFrame(state_rows).sort_values(
        keys, kind="stable", ignore_index=True
    )
    return correlations, states


def _scope_frames(frame: pd.DataFrame) -> Iterable[tuple[str, str, pd.DataFrame]]:
    for candidate in CANDIDATE_IDS:
        yield (
            candidate,
            "CANDIDATE_SPECIFIC_PRIMARY",
            frame.loc[frame["candidateId"].eq(candidate)],
        )
    yield POOLED_SCOPE, "POOLED_SECONDARY_ONLY", frame


def summarize_correlations(runwise: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for identity, branch in runwise.groupby(
        IDENTITY_COLUMNS, sort=True, observed=True
    ):
        metadata = dict(zip(IDENTITY_COLUMNS, identity, strict=True))
        for scope, role, scoped in _scope_frames(branch):
            record: dict[str, Any] = {
                **metadata,
                "candidateScope": scope,
                "evidenceRole": role,
                "trajectoryCount": len(scoped),
            }
            for prefix, value_col, p_col in (
                ("spearman", "spearmanRho", "spearmanTwoSidedP"),
                ("pearson", "pearsonR", "pearsonTwoSidedP"),
            ):
                values = scoped[value_col].to_numpy(np.float64)
                pvalues = scoped[p_col].to_numpy(np.float64)
                finite = np.isfinite(values) & np.isfinite(pvalues)
                selected = values[finite]
                selected_p = pvalues[finite]
                positive = selected > 0
                negative = selected < 0
                significant = selected_p < ALPHA
                record.update(
                    {
                        f"{prefix}DefinedCount": len(selected),
                        f"{prefix}UndefinedCount": int(len(values) - len(selected)),
                        f"{prefix}PositiveCount": int(np.count_nonzero(positive)),
                        f"{prefix}NegativeCount": int(np.count_nonzero(negative)),
                        f"{prefix}ZeroCount": int(
                            np.count_nonzero(selected == 0.0)
                        ),
                        f"{prefix}SignificantCount": int(
                            np.count_nonzero(significant)
                        ),
                        f"{prefix}PositiveSignificantCount": int(
                            np.count_nonzero(positive & significant)
                        ),
                        f"{prefix}NegativeSignificantCount": int(
                            np.count_nonzero(negative & significant)
                        ),
                        f"{prefix}NonsignificantCount": int(
                            np.count_nonzero(~significant)
                        ),
                        f"{prefix}Mean": float(np.mean(selected))
                        if len(selected)
                        else np.nan,
                        f"{prefix}Median": float(np.median(selected))
                        if len(selected)
                        else np.nan,
                        f"{prefix}PositiveFractionOfDefined": float(
                            np.mean(positive)
                        )
                        if len(selected)
                        else np.nan,
                        f"{prefix}PositiveSignificantFractionOfDefined": float(
                            np.mean(positive & significant)
                        )
                        if len(selected)
                        else np.nan,
                    }
                )
            rows.append(record)
    return pd.DataFrame(rows).sort_values(
        IDENTITY_COLUMNS + ["candidateScope"], kind="stable", ignore_index=True
    )


def one_sample_diagnostics(runwise: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for identity, branch in runwise.groupby(
        IDENTITY_COLUMNS, sort=True, observed=True
    ):
        metadata = dict(zip(IDENTITY_COLUMNS, identity, strict=True))
        for scope, role, scoped in _scope_frames(branch):
            for correlation, column in (
                ("SPEARMAN", "spearmanRho"),
                ("PEARSON", "pearsonR"),
            ):
                values = scoped[column].to_numpy(np.float64)
                values = values[np.isfinite(values)]
                row: dict[str, Any] = {
                    **metadata,
                    "candidateScope": scope,
                    "evidenceRole": role,
                    "correlationMeasure": correlation,
                    "definedCount": len(values),
                    "mean": float(np.mean(values)) if len(values) else np.nan,
                    "median": float(np.median(values)) if len(values) else np.nan,
                    "sampleStandardDeviation": float(np.std(values, ddof=1))
                    if len(values) > 1
                    else np.nan,
                }
                if len(values) > 1 and np.unique(values).size > 1:
                    two = ttest_1samp(values, popmean=0.0, alternative="two-sided")
                    greater = ttest_1samp(values, popmean=0.0, alternative="greater")
                    row.update(
                        {
                            "oneSampleT": float(two.statistic),
                            "oneSampleTTwoSidedP": float(two.pvalue),
                            "oneSampleTGreaterP": float(greater.pvalue),
                        }
                    )
                    nonzero = values[values != 0]
                    if len(nonzero):
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            signed = wilcoxon(
                                nonzero,
                                alternative="greater",
                                zero_method="wilcox",
                                method="approx",
                            )
                        row["wilcoxonGreaterStatistic"] = float(signed.statistic)
                        row["wilcoxonGreaterP"] = float(signed.pvalue)
                        signs = binomtest(
                            int(np.count_nonzero(nonzero > 0)),
                            len(nonzero),
                            p=0.5,
                            alternative="greater",
                        )
                        row["positiveSignCount"] = int(np.count_nonzero(nonzero > 0))
                        row["nonzeroSignCount"] = len(nonzero)
                        row["binomialSignGreaterP"] = float(signs.pvalue)
                    row["status"] = "ELIGIBLE"
                else:
                    row["status"] = "INELIGIBLE_TOO_FEW_OR_CONSTANT"
                rows.append(row)
    return pd.DataFrame(rows).sort_values(
        IDENTITY_COLUMNS + ["candidateScope", "correlationMeasure"],
        kind="stable",
        ignore_index=True,
    )


def summarize_state_comparisons(runwise: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for identity, branch in runwise.groupby(
        IDENTITY_COLUMNS, sort=True, observed=True
    ):
        metadata = dict(zip(IDENTITY_COLUMNS, identity, strict=True))
        for scope, role, scoped in _scope_frames(branch):
            eligible = scoped.loc[scoped["stateComparisonStatus"].eq("ELIGIBLE")]
            mean_diff = eligible["meanDifference"].to_numpy(np.float64)
            median_diff = eligible["medianDifference"].to_numpy(np.float64)
            mw_p = eligible["mannWhitneyGreaterP"].to_numpy(np.float64)
            rows.append(
                {
                    **metadata,
                    "candidateScope": scope,
                    "evidenceRole": role,
                    "trajectoryCount": len(scoped),
                    "definedStateComparisonCount": len(eligible),
                    "undefinedStateComparisonCount": int(len(scoped) - len(eligible)),
                    "higherReplicatorMeanCount": int(np.count_nonzero(mean_diff > 0)),
                    "lowerReplicatorMeanCount": int(np.count_nonzero(mean_diff < 0)),
                    "equalReplicatorMeanCount": int(np.count_nonzero(mean_diff == 0)),
                    "higherReplicatorMeanFraction": float(np.mean(mean_diff > 0))
                    if len(mean_diff)
                    else np.nan,
                    "higherReplicatorMedianCount": int(
                        np.count_nonzero(median_diff > 0)
                    ),
                    "lowerReplicatorMedianCount": int(
                        np.count_nonzero(median_diff < 0)
                    ),
                    "equalReplicatorMedianCount": int(
                        np.count_nonzero(median_diff == 0)
                    ),
                    "higherReplicatorMedianFraction": float(
                        np.mean(median_diff > 0)
                    )
                    if len(median_diff)
                    else np.nan,
                    "positiveSignificantWithinRunMannWhitneyCount": int(
                        np.count_nonzero((mean_diff > 0) & (mw_p < ALPHA))
                    ),
                    "nonsignificantWithinRunMannWhitneyCount": int(
                        np.count_nonzero(mw_p >= ALPHA)
                    ),
                    "acrossRunMedianReplicatorMean": float(
                        eligible["replicatorMean"].median()
                    )
                    if len(eligible)
                    else np.nan,
                    "acrossRunStandardDeviationReplicatorMean": float(
                        eligible["replicatorMean"].std(ddof=1)
                    )
                    if len(eligible) > 1
                    else np.nan,
                    "acrossRunMedianDriftMean": float(eligible["driftMean"].median())
                    if len(eligible)
                    else np.nan,
                    "acrossRunStandardDeviationDriftMean": float(
                        eligible["driftMean"].std(ddof=1)
                    )
                    if len(eligible) > 1
                    else np.nan,
                    "meanMeanDifference": float(np.mean(mean_diff))
                    if len(mean_diff)
                    else np.nan,
                    "medianMeanDifference": float(np.median(mean_diff))
                    if len(mean_diff)
                    else np.nan,
                    "meanMedianDifference": float(np.mean(median_diff))
                    if len(median_diff)
                    else np.nan,
                    "medianMedianDifference": float(np.median(median_diff))
                    if len(median_diff)
                    else np.nan,
                }
            )
    return pd.DataFrame(rows).sort_values(
        IDENTITY_COLUMNS + ["candidateScope"], kind="stable", ignore_index=True
    )


def mann_whitney_diagnostics(
    analysis_rows: pd.DataFrame, runwise_states: pd.DataFrame
) -> pd.DataFrame:
    """Retain every predeclared interpretation of the paper's ambiguous scope."""

    rows: list[dict[str, Any]] = []
    grouped_points = analysis_rows.groupby(IDENTITY_COLUMNS, sort=True, observed=True)
    for identity, branch_points in grouped_points:
        metadata = dict(zip(IDENTITY_COLUMNS, identity, strict=True))
        branch_states = runwise_states
        for column, value in metadata.items():
            branch_states = branch_states.loc[branch_states[column].eq(value)]
        for scope, role, scoped_points in _scope_frames(branch_points):
            if scope == POOLED_SCOPE:
                scoped_states = branch_states
            else:
                scoped_states = branch_states.loc[
                    branch_states["candidateId"].eq(scope)
                ]
            point_rep = scoped_points.loc[
                scoped_points["label"], "analysisValue"
            ].to_numpy(np.float64)
            point_drift = scoped_points.loc[
                ~scoped_points["label"], "analysisValue"
            ].to_numpy(np.float64)
            for diagnostic_scope, replication, drift in (
                ("POINT_POOLED_WITHIN_SCOPE", point_rep, point_drift),
                (
                    "RUN_SUMMARY_UNPAIRED_WITHIN_SCOPE",
                    scoped_states.loc[
                        scoped_states["stateComparisonStatus"].eq("ELIGIBLE"),
                        "replicatorMean",
                    ].to_numpy(np.float64),
                    scoped_states.loc[
                        scoped_states["stateComparisonStatus"].eq("ELIGIBLE"),
                        "driftMean",
                    ].to_numpy(np.float64),
                ),
            ):
                result = _safe_mann_whitney(replication, drift)
                rows.append(
                    {
                        **metadata,
                        "candidateScope": scope,
                        "evidenceRole": role,
                        "diagnosticScope": diagnostic_scope,
                        "replicatorValueCount": len(replication),
                        "driftValueCount": len(drift),
                        "status": result.pop("status"),
                        **result,
                    }
                )
    return pd.DataFrame(rows).sort_values(
        IDENTITY_COLUMNS + ["candidateScope", "diagnosticScope"],
        kind="stable",
        ignore_index=True,
    )


def fisher_diagnostics(runwise_states: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for identity, branch in runwise_states.groupby(
        IDENTITY_COLUMNS, sort=True, observed=True
    ):
        metadata = dict(zip(IDENTITY_COLUMNS, identity, strict=True))
        for scope, role, scoped in _scope_frames(branch):
            eligible = scoped.loc[scoped["stateComparisonStatus"].eq("ELIGIBLE")]
            for alternative, column in (
                ("greater", "mannWhitneyGreaterP"),
                ("two-sided", "mannWhitneyTwoSidedP"),
            ):
                pvalues = eligible[column].to_numpy(np.float64)
                pvalues = pvalues[np.isfinite(pvalues)]
                if len(pvalues):
                    clipped = np.clip(pvalues, np.nextafter(0.0, 1.0), 1.0)
                    statistic, combined = combine_pvalues(clipped, method="fisher")
                    status = "ELIGIBLE"
                else:
                    statistic, combined, status = np.nan, np.nan, "INELIGIBLE"
                rows.append(
                    {
                        **metadata,
                        "candidateScope": scope,
                        "evidenceRole": role,
                        "alternative": alternative,
                        "includedRunCount": len(pvalues),
                        "excludedRunCount": int(len(scoped) - len(pvalues)),
                        "fisherStatistic": float(statistic),
                        "degreesOfFreedom": int(2 * len(pvalues)),
                        "combinedP": float(combined),
                        "combinedPUnderflowedToZero": bool(combined == 0.0)
                        if np.isfinite(combined)
                        else False,
                        "status": status,
                    }
                )
    return pd.DataFrame(rows).sort_values(
        IDENTITY_COLUMNS + ["candidateScope", "alternative"],
        kind="stable",
        ignore_index=True,
    )


BOOTSTRAP_METRICS = (
    "medianSpearman",
    "meanSpearman",
    "positiveSpearmanFraction",
    "positiveSignificantSpearmanFraction",
    "medianPearson",
    "meanPearson",
    "medianMeanDifference",
    "meanMeanDifference",
    "positiveMeanDifferenceFraction",
    "medianMedianDifference",
    "positiveMedianDifferenceFraction",
    "medianReplicatorMean",
    "medianDriftMean",
)


def _metric_arrays(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    rho = frame["spearmanRho"].to_numpy(np.float64)
    rho_p = frame["spearmanTwoSidedP"].to_numpy(np.float64)
    pearson = frame["pearsonR"].to_numpy(np.float64)
    mean_diff = frame["meanDifference"].to_numpy(np.float64)
    median_diff = frame["medianDifference"].to_numpy(np.float64)
    return {
        "spearman": rho,
        "spearmanPositive": np.where(np.isfinite(rho), rho > 0, np.nan),
        "spearmanPositiveSignificant": np.where(
            np.isfinite(rho) & np.isfinite(rho_p),
            (rho > 0) & (rho_p < ALPHA),
            np.nan,
        ),
        "pearson": pearson,
        "meanDifference": mean_diff,
        "meanDifferencePositive": np.where(
            np.isfinite(mean_diff), mean_diff > 0, np.nan
        ),
        "medianDifference": median_diff,
        "medianDifferencePositive": np.where(
            np.isfinite(median_diff), median_diff > 0, np.nan
        ),
        "replicatorMean": frame["replicatorMean"].to_numpy(np.float64),
        "driftMean": frame["driftMean"].to_numpy(np.float64),
    }


def _summarize_sampled_arrays(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return {
            "medianSpearman": np.nanmedian(arrays["spearman"], axis=1),
            "meanSpearman": np.nanmean(arrays["spearman"], axis=1),
            "positiveSpearmanFraction": np.nanmean(
                arrays["spearmanPositive"], axis=1
            ),
            "positiveSignificantSpearmanFraction": np.nanmean(
                arrays["spearmanPositiveSignificant"], axis=1
            ),
            "medianPearson": np.nanmedian(arrays["pearson"], axis=1),
            "meanPearson": np.nanmean(arrays["pearson"], axis=1),
            "medianMeanDifference": np.nanmedian(
                arrays["meanDifference"], axis=1
            ),
            "meanMeanDifference": np.nanmean(arrays["meanDifference"], axis=1),
            "positiveMeanDifferenceFraction": np.nanmean(
                arrays["meanDifferencePositive"], axis=1
            ),
            "medianMedianDifference": np.nanmedian(
                arrays["medianDifference"], axis=1
            ),
            "positiveMedianDifferenceFraction": np.nanmean(
                arrays["medianDifferencePositive"], axis=1
            ),
            "medianReplicatorMean": np.nanmedian(
                arrays["replicatorMean"], axis=1
            ),
            "medianDriftMean": np.nanmedian(arrays["driftMean"], axis=1),
        }


def _observed_metrics(frame: pd.DataFrame) -> dict[str, float]:
    arrays = _metric_arrays(frame)
    sampled = {key: value.reshape(1, -1) for key, value in arrays.items()}
    return {key: float(value[0]) for key, value in _summarize_sampled_arrays(sampled).items()}


def trajectory_bootstrap(
    runwise_correlations: pd.DataFrame,
    runwise_states: pd.DataFrame,
    *,
    replicates: int,
    seed_root_hex: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resample trajectories, or paired matrix clusters for pooled summaries."""

    merge_keys = IDENTITY_COLUMNS + RUN_COLUMNS
    combined = runwise_correlations.merge(
        runwise_states,
        on=merge_keys,
        how="outer",
        validate="one_to_one",
        suffixes=("Correlation", "State"),
    )
    distributions: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for identity, branch in combined.groupby(
        IDENTITY_COLUMNS, sort=True, observed=True
    ):
        metadata = dict(zip(IDENTITY_COLUMNS, identity, strict=True))
        for scope, role, scoped in _scope_frames(branch):
            scoped = scoped.sort_values(
                ["matrixIndex", "candidateId"], kind="stable", ignore_index=True
            )
            seed = derive_seed(
                seed_root_hex,
                "trajectory_bootstrap",
                *[str(value) for value in identity],
                scope,
            )
            rng = np.random.Generator(np.random.PCG64DXSM(seed))
            arrays = _metric_arrays(scoped)
            if scope == POOLED_SCOPE:
                clusters = sorted(scoped["matrixIndex"].unique().tolist())
                max_width = max(
                    int(np.count_nonzero(scoped["matrixIndex"].eq(cluster)))
                    for cluster in clusters
                )
                sampled_clusters = rng.integers(
                    0, len(clusters), size=(replicates, len(clusters))
                )
                sampled_arrays: dict[str, np.ndarray] = {}
                for name, values in arrays.items():
                    matrix = np.full((len(clusters), max_width), np.nan, np.float64)
                    for row, cluster in enumerate(clusters):
                        selected = values[
                            scoped["matrixIndex"].to_numpy() == cluster
                        ]
                        matrix[row, : len(selected)] = selected
                    sampled_arrays[name] = matrix[sampled_clusters].reshape(
                        replicates, -1
                    )
                unit_count = len(clusters)
                resampling_unit = "SHARED_MATRIX_INDEX_CLUSTER"
            else:
                indices = rng.integers(
                    0, len(scoped), size=(replicates, len(scoped))
                )
                sampled_arrays = {name: values[indices] for name, values in arrays.items()}
                unit_count = len(scoped)
                resampling_unit = "TRAJECTORY"
            metrics = _summarize_sampled_arrays(sampled_arrays)
            distribution = pd.DataFrame(
                {
                    **{key: [value] * replicates for key, value in metadata.items()},
                    "candidateScope": scope,
                    "evidenceRole": role,
                    "resamplingUnit": resampling_unit,
                    "unitCount": unit_count,
                    "replicate": np.arange(replicates, dtype=np.int64),
                    "seed": str(seed),
                    **metrics,
                }
            )
            distributions.append(distribution)
            observed = _observed_metrics(scoped)
            for metric in BOOTSTRAP_METRICS:
                values = metrics[metric]
                summaries.append(
                    {
                        **metadata,
                        "candidateScope": scope,
                        "evidenceRole": role,
                        "resamplingUnit": resampling_unit,
                        "unitCount": unit_count,
                        "replicates": replicates,
                        "seed": str(seed),
                        "metric": metric,
                        "observed": observed[metric],
                        "bootstrapMedian": float(np.nanmedian(values)),
                        "bootstrapLower95": float(np.nanquantile(values, 0.025)),
                        "bootstrapUpper95": float(np.nanquantile(values, 0.975)),
                    }
                )
    distribution_frame = pd.concat(distributions, ignore_index=True)
    summary_frame = pd.DataFrame(summaries).sort_values(
        IDENTITY_COLUMNS + ["candidateScope", "metric"],
        kind="stable",
        ignore_index=True,
    )
    return distribution_frame, summary_frame


def all_cyclic_shift_metrics(
    values: np.ndarray, labels: np.ndarray
) -> dict[str, np.ndarray]:
    """Return each metric for ``labels`` rolled by every possible offset."""

    x = np.asarray(values, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    n = len(x)
    if n < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        empty = np.full(n, np.nan, dtype=np.float64)
        return {name: empty.copy() for name in ("spearman", "pearson", "meanDifference")}

    def roll_dot(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        transformed = np.fft.ifft(
            np.fft.fft(left) * np.conj(np.fft.fft(right))
        ).real
        transformed[np.abs(transformed) < 1e-14] = 0.0
        return transformed

    centered_x = x - np.mean(x)
    centered_y = y - np.mean(y)
    pearson = roll_dot(centered_x, centered_y) / np.sqrt(
        np.dot(centered_x, centered_x) * np.dot(centered_y, centered_y)
    )
    ranked_x = rankdata(x, method="average")
    ranked_x -= np.mean(ranked_x)
    spearman = roll_dot(ranked_x, centered_y) / np.sqrt(
        np.dot(ranked_x, ranked_x) * np.dot(centered_y, centered_y)
    )
    positive_count = int(np.sum(y))
    negative_count = n - positive_count
    state_sum = roll_dot(x, y)
    mean_difference = state_sum / positive_count - (
        np.sum(x) - state_sum
    ) / negative_count

    # Make the observed element algebraically identical to direct calculations.
    pearson[0] = float(pearsonr(x, y).statistic)
    spearman[0] = float(spearmanr(x, y).statistic)
    mean_difference[0] = float(np.mean(x[y.astype(bool)]) - np.mean(x[~y.astype(bool)]))
    return {
        "spearman": np.clip(spearman, -1.0, 1.0),
        "pearson": np.clip(pearson, -1.0, 1.0),
        "meanDifference": mean_difference,
    }


def circular_shift_control(
    analysis_rows: pd.DataFrame,
    *,
    replicates: int,
    seed_root_hex: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    distributions: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for identity, branch in analysis_rows.groupby(
        IDENTITY_COLUMNS, sort=True, observed=True
    ):
        metadata = dict(zip(IDENTITY_COLUMNS, identity, strict=True))
        for scope, role, scoped in _scope_frames(branch):
            trajectory_metrics: list[dict[str, np.ndarray]] = []
            trajectory_seed_records: list[str] = []
            for run_identity, trajectory in scoped.groupby(
                RUN_COLUMNS, sort=True, observed=True
            ):
                ordered = trajectory.sort_values("observationOrder", kind="stable")
                x = ordered["analysisValue"].to_numpy(np.float64)
                y = ordered["label"].to_numpy(bool)
                metrics = all_cyclic_shift_metrics(x, y)
                if not np.isfinite(metrics["spearman"][0]):
                    continue
                candidate, matrix_index, trajectory_id = run_identity
                seed = derive_seed(
                    seed_root_hex,
                    "circular_shift",
                    *[str(value) for value in identity],
                    str(candidate),
                    str(matrix_index),
                    str(trajectory_id),
                )
                rng = np.random.Generator(np.random.PCG64DXSM(seed))
                shifts = rng.integers(1, len(x), size=replicates)
                trajectory_metrics.append(
                    {
                        "spearman": metrics["spearman"][shifts],
                        "pearson": metrics["pearson"][shifts],
                        "meanDifference": metrics["meanDifference"][shifts],
                        "observedSpearman": np.array(metrics["spearman"][0]),
                        "observedPearson": np.array(metrics["pearson"][0]),
                        "observedMeanDifference": np.array(
                            metrics["meanDifference"][0]
                        ),
                    }
                )
                trajectory_seed_records.append(
                    f"{candidate}|{matrix_index}|{trajectory_id}|{seed}"
                )
            seed_digest = hashlib.sha256(
                "\n".join(trajectory_seed_records).encode("utf-8")
            ).hexdigest()
            if not trajectory_metrics:
                continue
            null = {
                "medianSpearman": np.median(
                    np.vstack([item["spearman"] for item in trajectory_metrics]),
                    axis=0,
                ),
                "medianPearson": np.median(
                    np.vstack([item["pearson"] for item in trajectory_metrics]),
                    axis=0,
                ),
                "medianMeanDifference": np.median(
                    np.vstack(
                        [item["meanDifference"] for item in trajectory_metrics]
                    ),
                    axis=0,
                ),
            }
            observed = {
                "medianSpearman": float(
                    np.median(
                        [item["observedSpearman"] for item in trajectory_metrics]
                    )
                ),
                "medianPearson": float(
                    np.median(
                        [item["observedPearson"] for item in trajectory_metrics]
                    )
                ),
                "medianMeanDifference": float(
                    np.median(
                        [
                            item["observedMeanDifference"]
                            for item in trajectory_metrics
                        ]
                    )
                ),
            }
            distributions.append(
                pd.DataFrame(
                    {
                        **{key: [value] * replicates for key, value in metadata.items()},
                        "candidateScope": scope,
                        "evidenceRole": role,
                        "trajectoryCount": len(trajectory_metrics),
                        "replicate": np.arange(replicates, dtype=np.int64),
                        "trajectorySeedDigest": seed_digest,
                        **null,
                    }
                )
            )
            for metric, values in null.items():
                observed_value = observed[metric]
                summaries.append(
                    {
                        **metadata,
                        "candidateScope": scope,
                        "evidenceRole": role,
                        "trajectoryCount": len(trajectory_metrics),
                        "replicates": replicates,
                        "trajectorySeedDigest": seed_digest,
                        "metric": metric,
                        "observed": observed_value,
                        "nullMedian": float(np.median(values)),
                        "nullLower95": float(np.quantile(values, 0.025)),
                        "nullUpper95": float(np.quantile(values, 0.975)),
                        "positiveP": float(
                            (1 + np.count_nonzero(values >= observed_value))
                            / (replicates + 1)
                        ),
                        "twoSidedP": float(
                            (
                                1
                                + np.count_nonzero(
                                    np.abs(values) >= abs(observed_value)
                                )
                            )
                            / (replicates + 1)
                        ),
                    }
                )
    return (
        pd.concat(distributions, ignore_index=True),
        pd.DataFrame(summaries).sort_values(
            IDENTITY_COLUMNS + ["candidateScope", "metric"],
            kind="stable",
            ignore_index=True,
        ),
    )


def ordinary_stability_coupling(
    analysis_rows: pd.DataFrame,
    *,
    bootstrap_replicates: int,
    seed_root_hex: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Describe completed-fit coupling to H and ordinary composition stability."""

    primary = analysis_rows.loc[analysis_rows["branchId"].eq(PRIMARY_BRANCH)]
    rows: list[dict[str, Any]] = []
    for identity, group in primary.groupby(
        ["candidateId", "matrixIndex", "trajectoryId", "analysisId"],
        sort=True,
        observed=True,
    ):
        metadata = dict(
            zip(
                ["candidateId", "matrixIndex", "trajectoryId", "analysisId"],
                identity,
                strict=True,
            )
        )
        x = group["analysisValue"].to_numpy(np.float64)
        for predictor_id, column in (
            ("EXACT_INCOMING_H", "incomingCosineH"),
            (
                "NEGATIVE_EUCLIDEAN_L2_CLOSED_COMPOSITION_CHANGE",
                "negativeEuclideanL2ClosedCompositionChange",
            ),
        ):
            predictor = group[column].to_numpy(np.float64)
            finite = np.isfinite(x) & np.isfinite(predictor)
            result = _safe_correlations(x[finite], predictor[finite])
            rows.append(
                {
                    **metadata,
                    "predictorId": predictor_id,
                    "n": int(np.count_nonzero(finite)),
                    "status": result.pop("status"),
                    **result,
                }
            )
    runwise = pd.DataFrame(rows).sort_values(
        ["analysisId", "predictorId", "candidateId", "matrixIndex"],
        kind="stable",
        ignore_index=True,
    )
    summaries: list[dict[str, Any]] = []
    for (analysis_id, predictor_id), branch in runwise.groupby(
        ["analysisId", "predictorId"], sort=True, observed=True
    ):
        for scope, role, scoped in _scope_frames(branch):
            for measure, column in (
                ("SPEARMAN", "spearmanRho"),
                ("PEARSON", "pearsonR"),
            ):
                values = scoped[column].to_numpy(np.float64)
                values = values[np.isfinite(values)]
                seed = derive_seed(
                    seed_root_hex,
                    "ordinary_stability_bootstrap",
                    analysis_id,
                    predictor_id,
                    scope,
                    measure,
                )
                rng = np.random.Generator(np.random.PCG64DXSM(seed))
                if len(values):
                    sampled = values[
                        rng.integers(
                            0,
                            len(values),
                            size=(bootstrap_replicates, len(values)),
                        )
                    ]
                    medians = np.median(sampled, axis=1)
                else:
                    medians = np.full(bootstrap_replicates, np.nan)
                summaries.append(
                    {
                        "analysisId": analysis_id,
                        "predictorId": predictor_id,
                        "candidateScope": scope,
                        "evidenceRole": role,
                        "correlationMeasure": measure,
                        "definedCount": len(values),
                        "meanCorrelation": float(np.mean(values))
                        if len(values)
                        else np.nan,
                        "medianCorrelation": float(np.median(values))
                        if len(values)
                        else np.nan,
                        "bootstrapMedianLower95": float(
                            np.nanquantile(medians, 0.025)
                        ),
                        "bootstrapMedianUpper95": float(
                            np.nanquantile(medians, 0.975)
                        ),
                        "bootstrapReplicates": bootstrap_replicates,
                        "seed": str(seed),
                    }
                )
    return runwise, pd.DataFrame(summaries).sort_values(
        ["analysisId", "predictorId", "candidateScope", "correlationMeasure"],
        kind="stable",
        ignore_index=True,
    )
