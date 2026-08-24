"""Pure analysis contracts for E01/S14.

The functions in this module operate only on already-materialized S13Y values.
They do not import or invoke a simulator or a PhiRL estimator.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import peak_prominences, peak_widths
from scipy.stats import fisher_exact, linregress, pearsonr, spearmanr, theilslopes
from sklearn.metrics import adjusted_rand_score
from statsmodels.stats.diagnostic import acorr_ljungbox

RESEARCH_STEP_ID = "S14"
VERSIONED_STEP_ID = "E01-S14-DESCRIPTIVE-CAUSAL-EMERGENCE-DYNAMICS-v1.0.0"
CANDIDATE_IDS = ("S12F-CANDIDATE-02", "S12F-CANDIDATE-03")
COMPLETED_MODE = "RETROSPECTIVE_FULL_TRAJECTORY_LOCAL"
PREFIX_MODE = "PAST_ONLY_PREFIX_ENDPOINT"
THRESHOLD_FAMILIES = ("THREE_SIGMA", "ROBUST_MAD")
SIGNS = ("POSITIVE", "NEGATIVE")


def lag_rule(n: int) -> int:
    """Return the inherited S13Y Ljung–Box lag for a series length."""

    return max(1, min(10, int(n) // 5))


def prepare_completed(full: pd.DataFrame) -> pd.DataFrame:
    """Canonicalize frozen completed-fit values without changing values."""

    required = {
        "candidateId",
        "trajectoryId",
        "matrixIndex",
        "temporalLabel",
        "selectedSequenceIndex",
        "rawObservationIndex",
        "observationKind",
        "generation",
        "molecularStep",
        "status",
        "emergence",
    }
    missing = required - set(full.columns)
    if missing:
        raise ValueError(f"completed values missing columns: {sorted(missing)}")
    frame = full.loc[full["status"].eq("ELIGIBLE")].copy()
    frame = frame.rename(columns={"temporalLabel": "temporalMode"})
    frame["modeObservationIndex"] = frame["selectedSequenceIndex"].astype(int)
    frame["timeCoordinate"] = frame["selectedSequenceIndex"].astype(float)
    frame["emergence"] = pd.to_numeric(frame["emergence"], errors="raise")
    frame["isPostFission"] = frame["observationKind"].eq("post_fission")
    frame.sort_values(
        ["candidateId", "matrixIndex", "selectedSequenceIndex"],
        kind="stable",
        inplace=True,
        ignore_index=True,
    )
    return frame


def prepare_prefix(prefix: pd.DataFrame) -> pd.DataFrame:
    """Canonicalize eligible frozen past-only endpoint values."""

    required = {
        "candidateId",
        "trajectoryId",
        "matrixIndex",
        "temporalLabel",
        "generation",
        "endpointSelectedSequenceIndex",
        "endpointRawObservationIndex",
        "endpointObservationKind",
        "status",
        "emergence",
    }
    missing = required - set(prefix.columns)
    if missing:
        raise ValueError(f"prefix values missing columns: {sorted(missing)}")
    frame = prefix.loc[prefix["status"].eq("ELIGIBLE")].copy()
    frame = frame.rename(
        columns={
            "temporalLabel": "temporalMode",
            "endpointSelectedSequenceIndex": "selectedSequenceIndex",
            "endpointRawObservationIndex": "rawObservationIndex",
            "endpointObservationKind": "observationKind",
        }
    )
    frame["molecularStep"] = np.nan
    frame["timeCoordinate"] = frame["generation"].astype(float)
    frame["emergence"] = pd.to_numeric(frame["emergence"], errors="raise")
    frame["isPostFission"] = frame["observationKind"].eq("post_fission")
    frame.sort_values(
        ["candidateId", "matrixIndex", "generation"],
        kind="stable",
        inplace=True,
        ignore_index=True,
    )
    frame["modeObservationIndex"] = frame.groupby(
        ["candidateId", "trajectoryId"], sort=False
    ).cumcount()
    return frame


def _scope_frames(frame: pd.DataFrame) -> Iterable[tuple[str, pd.DataFrame]]:
    for candidate_id in CANDIDATE_IDS:
        yield candidate_id, frame.loc[frame["candidateId"].eq(candidate_id)]
    yield "POOLED_SECONDARY", frame


def aggregate_trajectories(
    frame: pd.DataFrame,
    *,
    normalized_grid_points: int = 101,
    majority_fraction: float = 0.5,
) -> pd.DataFrame:
    """Construct all frozen aggregate alignment views.

    Molecular-index views use the completed selected-sequence index and prefix
    generation, respectively. The normalized view linearly interpolates every
    trajectory onto a fixed 0–1 grid and therefore always uses the full cohort.
    """

    rows: list[dict[str, Any]] = []
    mode = str(frame["temporalMode"].iloc[0]) if len(frame) else "UNKNOWN"
    for scope, scoped in _scope_frames(frame):
        trajectory_count = int(scoped["trajectoryId"].nunique())
        if trajectory_count == 0:
            continue
        grouped = scoped.groupby("timeCoordinate", sort=True)["emergence"]
        raw = grouped.agg(
            contributingTrajectoryCount="count",
            medianEmergence="median",
            meanEmergence="mean",
            standardDeviation=lambda x: (
                float(np.std(x.to_numpy(float), ddof=1)) if len(x) > 1 else np.nan
            ),
            q25=lambda x: float(np.quantile(x.to_numpy(float), 0.25)),
            q75=lambda x: float(np.quantile(x.to_numpy(float), 0.75)),
            minimum="min",
            maximum="max",
        ).reset_index()
        view_masks = {
            "AVAILABLE_CASE": raw["contributingTrajectoryCount"].ge(1),
            "FULL_COHORT_SUPPORT": raw["contributingTrajectoryCount"].eq(
                trajectory_count
            ),
            "MAJORITY_SUPPORT": raw["contributingTrajectoryCount"].ge(
                int(np.ceil(trajectory_count * majority_fraction))
            ),
        }
        for view, mask in view_masks.items():
            for record in raw.loc[mask].to_dict("records"):
                rows.append(
                    {
                        "candidateScope": scope,
                        "temporalMode": mode,
                        "alignmentView": view,
                        "timeCoordinate": float(record["timeCoordinate"]),
                        "normalizedTime": np.nan,
                        "totalTrajectoryCount": trajectory_count,
                        **{
                            key: record[key]
                            for key in (
                                "contributingTrajectoryCount",
                                "medianEmergence",
                                "meanEmergence",
                                "standardDeviation",
                                "q25",
                                "q75",
                                "minimum",
                                "maximum",
                            )
                        },
                    }
                )

        grid = np.linspace(0.0, 1.0, normalized_grid_points, dtype=np.float64)
        interpolated: list[np.ndarray] = []
        for _, trajectory in scoped.groupby("trajectoryId", sort=True):
            ordered = trajectory.sort_values("modeObservationIndex", kind="stable")
            values = ordered["emergence"].to_numpy(np.float64)
            if len(values) == 1:
                interpolated.append(np.repeat(values[0], len(grid)))
            elif len(values) > 1:
                source_grid = np.linspace(0.0, 1.0, len(values), dtype=np.float64)
                interpolated.append(np.interp(grid, source_grid, values))
        matrix = np.vstack(interpolated)
        for index, coordinate in enumerate(grid):
            values = matrix[:, index]
            rows.append(
                {
                    "candidateScope": scope,
                    "temporalMode": mode,
                    "alignmentView": "NORMALIZED_TIME_101",
                    "timeCoordinate": float(coordinate),
                    "normalizedTime": float(coordinate),
                    "totalTrajectoryCount": trajectory_count,
                    "contributingTrajectoryCount": len(values),
                    "medianEmergence": float(np.median(values)),
                    "meanEmergence": float(np.mean(values)),
                    "standardDeviation": float(np.std(values, ddof=1)),
                    "q25": float(np.quantile(values, 0.25)),
                    "q75": float(np.quantile(values, 0.75)),
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                }
            )
    result = pd.DataFrame(rows)
    result["contributingTrajectoryCount"] = result[
        "contributingTrajectoryCount"
    ].astype(int)
    result["totalTrajectoryCount"] = result["totalTrajectoryCount"].astype(int)
    return result.sort_values(
        ["temporalMode", "candidateScope", "alignmentView", "timeCoordinate"],
        kind="stable",
        ignore_index=True,
    )


def trend_results(aggregate: pd.DataFrame) -> pd.DataFrame:
    """Fit the frozen OLS trend and a Theil–Sen robustness diagnostic."""

    rows: list[dict[str, Any]] = []
    keys = ["candidateScope", "temporalMode", "alignmentView"]
    for identity, group in aggregate.groupby(keys, sort=True):
        clean = group.dropna(subset=["timeCoordinate", "medianEmergence"])
        x = clean["timeCoordinate"].to_numpy(np.float64)
        y = clean["medianEmergence"].to_numpy(np.float64)
        if len(x) < 3 or len(np.unique(x)) < 2:
            rows.append(
                {
                    **dict(zip(keys, identity, strict=True)),
                    "positionCount": len(x),
                    "status": "INELIGIBLE_TOO_FEW_POSITIONS",
                }
            )
            continue
        fit = linregress(x, y)
        robust = theilslopes(y, x, alpha=0.95, method="separate")
        rows.append(
            {
                **dict(zip(keys, identity, strict=True)),
                "positionCount": len(x),
                "minimumTime": float(np.min(x)),
                "maximumTime": float(np.max(x)),
                "minimumContributingTrajectories": int(
                    clean["contributingTrajectoryCount"].min()
                ),
                "maximumContributingTrajectories": int(
                    clean["contributingTrajectoryCount"].max()
                ),
                "olsSlope": float(fit.slope),
                "olsIntercept": float(fit.intercept),
                "olsR": float(fit.rvalue),
                "olsR2": float(fit.rvalue**2),
                "olsTwoSidedP": float(fit.pvalue),
                "olsSlopeStandardError": float(fit.stderr),
                "theilSenSlope": float(robust.slope),
                "theilSenIntercept": float(robust.intercept),
                "theilSenSlopeLower95": float(robust.low_slope),
                "theilSenSlopeUpper95": float(robust.high_slope),
                "status": "ELIGIBLE",
            }
        )
    return pd.DataFrame(rows).sort_values(keys, kind="stable", ignore_index=True)


def add_excursion_flags(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add the two frozen signed excursion families to each trajectory."""

    flagged: list[pd.DataFrame] = []
    threshold_rows: list[dict[str, Any]] = []
    for (candidate_id, trajectory_id), group in frame.groupby(
        ["candidateId", "trajectoryId"], sort=True
    ):
        ordered = group.sort_values("modeObservationIndex", kind="stable").copy()
        values = ordered["emergence"].to_numpy(np.float64)
        mean = float(np.mean(values))
        sd = float(np.std(values, ddof=0))
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        robust_scale = float(1.4826 * mad)
        thresholds = {
            "THREE_SIGMA": (mean - 3.0 * sd, mean + 3.0 * sd, mean, sd),
            "ROBUST_MAD": (
                median - 3.0 * robust_scale,
                median + 3.0 * robust_scale,
                median,
                robust_scale,
            ),
        }
        for family, (lower, upper, center, scale) in thresholds.items():
            positive = values > upper if scale > 0 else np.zeros(len(values), bool)
            negative = values < lower if scale > 0 else np.zeros(len(values), bool)
            ordered[f"{family}_POSITIVE"] = positive
            ordered[f"{family}_NEGATIVE"] = negative
            threshold_rows.append(
                {
                    "candidateId": candidate_id,
                    "trajectoryId": trajectory_id,
                    "matrixIndex": int(ordered["matrixIndex"].iloc[0]),
                    "temporalMode": str(ordered["temporalMode"].iloc[0]),
                    "thresholdFamily": family,
                    "center": center,
                    "scale": scale,
                    "lowerThreshold": lower,
                    "upperThreshold": upper,
                    "observationCount": len(values),
                    "positivePointCount": int(np.count_nonzero(positive)),
                    "negativePointCount": int(np.count_nonzero(negative)),
                }
            )
        flagged.append(ordered)
    return (
        pd.concat(flagged, ignore_index=True),
        pd.DataFrame(threshold_rows).sort_values(
            ["candidateId", "matrixIndex", "thresholdFamily"],
            kind="stable",
            ignore_index=True,
        ),
    )


def _true_segments(flags: np.ndarray) -> list[tuple[int, int]]:
    positions = np.flatnonzero(flags)
    if not len(positions):
        return []
    breaks = np.flatnonzero(np.diff(positions) > 1)
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, len(positions) - 1]
    return [(int(positions[a]), int(positions[b])) for a, b in zip(starts, ends)]


def excursion_catalog(
    flagged: pd.DataFrame, thresholds: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create episode-level morphology, run summaries, and fission enrichment."""

    catalog_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    dependency_rows: list[dict[str, Any]] = []
    threshold_index = thresholds.set_index(
        ["candidateId", "trajectoryId", "thresholdFamily"]
    )
    for (candidate_id, trajectory_id), group in flagged.groupby(
        ["candidateId", "trajectoryId"], sort=True
    ):
        ordered = group.sort_values("modeObservationIndex", kind="stable")
        values = ordered["emergence"].to_numpy(np.float64)
        raw = ordered["rawObservationIndex"].to_numpy(np.int64)
        fission = ordered["isPostFission"].to_numpy(bool)
        fission_raw = raw[fission]
        mode = str(ordered["temporalMode"].iloc[0])
        for family in THRESHOLD_FAMILIES:
            threshold = threshold_index.loc[(candidate_id, trajectory_id, family)]
            for sign in SIGNS:
                flags = ordered[f"{family}_{sign}"].to_numpy(bool)
                segments = _true_segments(flags)
                prior_peak: int | None = None
                for episode_number, (start, end) in enumerate(segments, start=1):
                    signed = values if sign == "POSITIVE" else -values
                    relative_peak = int(np.argmax(signed[start : end + 1]))
                    peak_position = start + relative_peak
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        prominence = float(
                            peak_prominences(signed, np.array([peak_position]))[0][0]
                        )
                        half_width = float(
                            peak_widths(
                                signed,
                                np.array([peak_position]),
                                rel_height=0.5,
                                prominence_data=peak_prominences(
                                    signed, np.array([peak_position])
                                ),
                            )[0][0]
                        )
                    peak_raw = int(raw[peak_position])
                    distance = (
                        int(np.min(np.abs(fission_raw - peak_raw)))
                        if len(fission_raw)
                        else None
                    )
                    peak = ordered.iloc[peak_position]
                    catalog_rows.append(
                        {
                            "candidateId": candidate_id,
                            "trajectoryId": trajectory_id,
                            "matrixIndex": int(peak["matrixIndex"]),
                            "temporalMode": mode,
                            "thresholdFamily": family,
                            "sign": sign,
                            "episodeNumber": episode_number,
                            "center": float(threshold["center"]),
                            "scale": float(threshold["scale"]),
                            "lowerThreshold": float(threshold["lowerThreshold"]),
                            "upperThreshold": float(threshold["upperThreshold"]),
                            "episodeStartRawObservationIndex": int(raw[start]),
                            "episodeEndRawObservationIndex": int(raw[end]),
                            "episodeWidthObservations": int(end - start + 1),
                            "episodeRawMolecularSpan": int(raw[end] - raw[start] + 1),
                            "peakRawObservationIndex": peak_raw,
                            "peakSelectedSequenceIndex": int(
                                peak["selectedSequenceIndex"]
                            ),
                            "peakGeneration": int(peak["generation"]),
                            "peakMolecularStep": float(peak["molecularStep"])
                            if pd.notna(peak["molecularStep"])
                            else np.nan,
                            "peakEmergence": float(peak["emergence"]),
                            "peakSignedStandardizedHeight": float(
                                (
                                    signed[peak_position]
                                    - (
                                        float(threshold["center"])
                                        if sign == "POSITIVE"
                                        else -float(threshold["center"])
                                    )
                                )
                                / float(threshold["scale"])
                            )
                            if float(threshold["scale"]) > 0
                            else np.nan,
                            "peakProminence": prominence,
                            "halfProminenceWidthObservations": half_width,
                            "spacingFromPreviousPeakRawSteps": (
                                peak_raw - prior_peak
                                if prior_peak is not None
                                else np.nan
                            ),
                            "normalizedPeakTime": float(
                                peak_position / max(1, len(ordered) - 1)
                            ),
                            "peakIsPostFission": bool(fission[peak_position]),
                            "episodeContainsPostFission": bool(
                                np.any(fission[start : end + 1])
                            ),
                            "distanceToNearestPostFissionRawSteps": distance,
                        }
                    )
                    prior_peak = peak_raw
                run_rows.append(
                    {
                        "candidateId": candidate_id,
                        "trajectoryId": trajectory_id,
                        "matrixIndex": int(ordered["matrixIndex"].iloc[0]),
                        "temporalMode": mode,
                        "thresholdFamily": family,
                        "sign": sign,
                        "observationCount": len(ordered),
                        "excursionPointCount": int(np.count_nonzero(flags)),
                        "excursionEpisodeCount": len(segments),
                        "hasExcursion": bool(len(segments) > 0),
                    }
                )
                a = int(np.count_nonzero(flags & fission))
                b = int(np.count_nonzero(flags & ~fission))
                c = int(np.count_nonzero(~flags & fission))
                d = int(np.count_nonzero(~flags & ~fission))
                odds, pvalue = fisher_exact([[a, b], [c, d]], alternative="two-sided")
                dependency_rows.append(
                    {
                        "candidateId": candidate_id,
                        "trajectoryId": trajectory_id,
                        "matrixIndex": int(ordered["matrixIndex"].iloc[0]),
                        "temporalMode": mode,
                        "thresholdFamily": family,
                        "sign": sign,
                        "excursionPostFissionPoints": a,
                        "excursionOtherPoints": b,
                        "nonExcursionPostFissionPoints": c,
                        "nonExcursionOtherPoints": d,
                        "fissionOddsRatio": float(odds),
                        "fissionFisherTwoSidedP": float(pvalue),
                    }
                )
    catalog = pd.DataFrame(catalog_rows)
    if len(catalog):
        catalog.sort_values(
            [
                "temporalMode",
                "candidateId",
                "matrixIndex",
                "thresholdFamily",
                "sign",
                "peakRawObservationIndex",
            ],
            kind="stable",
            inplace=True,
            ignore_index=True,
        )
    run_summary = pd.DataFrame(run_rows).sort_values(
        ["temporalMode", "candidateId", "matrixIndex", "thresholdFamily", "sign"],
        kind="stable",
        ignore_index=True,
    )
    dependency = pd.DataFrame(dependency_rows).sort_values(
        ["temporalMode", "candidateId", "matrixIndex", "thresholdFamily", "sign"],
        kind="stable",
        ignore_index=True,
    )
    return catalog, run_summary, dependency


def summarize_excursions(
    catalog: pd.DataFrame, run_summary: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["candidateId", "temporalMode", "thresholdFamily", "sign"]
    for identity, runs in run_summary.groupby(keys, sort=True):
        episodes = catalog
        for key, value in zip(keys, identity, strict=True):
            episodes = episodes.loc[episodes[key].eq(value)]
        defined_spacing = episodes["spacingFromPreviousPeakRawSteps"].dropna()
        rows.append(
            {
                **dict(zip(keys, identity, strict=True)),
                "trajectoryCount": len(runs),
                "runWithExcursionCount": int(runs["hasExcursion"].sum()),
                "runWithExcursionFraction": float(runs["hasExcursion"].mean()),
                "totalExcursionPointCount": int(runs["excursionPointCount"].sum()),
                "totalEpisodeCount": int(runs["excursionEpisodeCount"].sum()),
                "medianEpisodeCountAmongRuns": float(
                    runs["excursionEpisodeCount"].median()
                ),
                "medianEpisodeWidthObservations": float(
                    episodes["episodeWidthObservations"].median()
                )
                if len(episodes)
                else np.nan,
                "medianPeakProminence": float(episodes["peakProminence"].median())
                if len(episodes)
                else np.nan,
                "medianPeakNormalizedTime": float(
                    episodes["normalizedPeakTime"].median()
                )
                if len(episodes)
                else np.nan,
                "medianInterPeakSpacingRawSteps": float(defined_spacing.median())
                if len(defined_spacing)
                else np.nan,
                "peakPostFissionFraction": float(episodes["peakIsPostFission"].mean())
                if len(episodes)
                else np.nan,
                "episodeContainsPostFissionFraction": float(
                    episodes["episodeContainsPostFission"].mean()
                )
                if len(episodes)
                else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(keys, kind="stable", ignore_index=True)


def ljung_box_results(frame: pd.DataFrame) -> pd.DataFrame:
    """Run the inherited lag rule on raw and first-differenced trajectories."""

    rows: list[dict[str, Any]] = []
    for (candidate_id, trajectory_id), group in frame.groupby(
        ["candidateId", "trajectoryId"], sort=True
    ):
        ordered = group.sort_values("modeObservationIndex", kind="stable")
        raw = ordered["emergence"].to_numpy(np.float64)
        for transform, values in (("RAW", raw), ("FIRST_DIFFERENCE", np.diff(raw))):
            if len(values) < 5 or not np.all(np.isfinite(values)):
                rows.append(
                    {
                        "candidateId": candidate_id,
                        "trajectoryId": trajectory_id,
                        "matrixIndex": int(ordered["matrixIndex"].iloc[0]),
                        "temporalMode": str(ordered["temporalMode"].iloc[0]),
                        "transform": transform,
                        "observationCount": len(values),
                        "lag": np.nan,
                        "ljungBoxStatistic": np.nan,
                        "ljungBoxP": np.nan,
                        "rejectAt0_05": False,
                        "status": "INELIGIBLE_TOO_SHORT_OR_NONFINITE",
                    }
                )
                continue
            lag = lag_rule(len(values))
            fit = acorr_ljungbox(values, lags=[lag], return_df=True)
            statistic = float(fit["lb_stat"].iloc[0])
            pvalue = float(fit["lb_pvalue"].iloc[0])
            rows.append(
                {
                    "candidateId": candidate_id,
                    "trajectoryId": trajectory_id,
                    "matrixIndex": int(ordered["matrixIndex"].iloc[0]),
                    "temporalMode": str(ordered["temporalMode"].iloc[0]),
                    "transform": transform,
                    "observationCount": len(values),
                    "lag": lag,
                    "ljungBoxStatistic": statistic,
                    "ljungBoxP": pvalue,
                    "rejectAt0_05": bool(pvalue < 0.05),
                    "status": "ELIGIBLE",
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["temporalMode", "candidateId", "matrixIndex", "transform"],
        kind="stable",
        ignore_index=True,
    )


def summarize_ljung_box(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["candidateId", "temporalMode", "transform"]
    for identity, group in results.groupby(keys, sort=True):
        eligible = group.loc[group["status"].eq("ELIGIBLE")]
        rows.append(
            {
                **dict(zip(keys, identity, strict=True)),
                "trajectoryCount": len(group),
                "eligibleCount": len(eligible),
                "rejectCountAt0_05": int(eligible["rejectAt0_05"].sum()),
                "rejectFractionAt0_05": float(eligible["rejectAt0_05"].mean())
                if len(eligible)
                else np.nan,
                "medianP": float(eligible["ljungBoxP"].median())
                if len(eligible)
                else np.nan,
                "minimumP": float(eligible["ljungBoxP"].min())
                if len(eligible)
                else np.nan,
                "maximumP": float(eligible["ljungBoxP"].max())
                if len(eligible)
                else np.nan,
                "lagMinimum": int(eligible["lag"].min()) if len(eligible) else None,
                "lagMaximum": int(eligible["lag"].max()) if len(eligible) else None,
            }
        )
    return pd.DataFrame(rows).sort_values(keys, kind="stable", ignore_index=True)


def _partition_labels(row: pd.Series) -> tuple[tuple[int, ...], tuple[int, ...]]:
    left = tuple(sorted(int(x) for x in json.loads(row["partition1Json"])))
    right = tuple(sorted(int(x) for x in json.loads(row["partition2Json"])))
    return tuple(sorted((left, right)))  # type: ignore[return-value]


def _membership(
    pair: tuple[tuple[int, ...], tuple[int, ...]],
) -> tuple[list[int], list[int]]:
    variables = sorted(set(pair[0]) | set(pair[1]))
    left = set(pair[0])
    return variables, [0 if variable in left else 1 for variable in variables]


def partition_change_history(partitions: pd.DataFrame) -> pd.DataFrame:
    """Compare unordered consecutive eligible past-only bipartitions."""

    prefix = partitions.loc[
        partitions["fitKind"].eq("past_only_prefix_endpoint")
        & partitions["status"].eq("ELIGIBLE")
    ].copy()
    rows: list[dict[str, Any]] = []
    for (candidate_id, trajectory_id), group in prefix.groupby(
        ["candidateId", "trajectoryId"], sort=True
    ):
        previous: tuple[tuple[int, ...], tuple[int, ...]] | None = None
        for _, row in group.sort_values("endpointGeneration", kind="stable").iterrows():
            current = _partition_labels(row)
            changed: bool | None = None
            ari = np.nan
            if previous is not None:
                variables0, labels0 = _membership(previous)
                variables1, labels1 = _membership(current)
                if variables0 != variables1:
                    raise ValueError(
                        "retained-variable identity changed across prefix fits"
                    )
                changed = current != previous
                ari = float(adjusted_rand_score(labels0, labels1))
            rows.append(
                {
                    "candidateId": candidate_id,
                    "trajectoryId": trajectory_id,
                    "matrixIndex": int(row["matrixIndex"]),
                    "endpointGeneration": int(row["endpointGeneration"]),
                    "endpointSelectedSequenceIndex": int(
                        row["endpointSelectedSequenceIndex"]
                    ),
                    "partitionChangedFromPreviousEligibleFit": changed,
                    "partitionARIFromPreviousEligibleFit": ari,
                    "partitionSize1": int(row["partitionSize1"]),
                    "partitionSize2": int(row["partitionSize2"]),
                    "partitionCanonicalJson": json.dumps(
                        current, separators=(",", ":")
                    ),
                }
            )
            previous = current
    return pd.DataFrame(rows).sort_values(
        ["candidateId", "matrixIndex", "endpointGeneration"],
        kind="stable",
        ignore_index=True,
    )


def compare_completed_prefix(
    completed_flagged: pd.DataFrame,
    prefix_flagged: pd.DataFrame,
    partition_history: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare exact shared endpoint rows under completed versus past-only fits."""

    keys = ["candidateId", "trajectoryId", "matrixIndex", "selectedSequenceIndex"]
    flag_columns = [
        f"{family}_{sign}" for family in THRESHOLD_FAMILIES for sign in SIGNS
    ]
    left = completed_flagged[
        keys + ["rawObservationIndex", "generation", "emergence"] + flag_columns
    ].rename(
        columns={
            "rawObservationIndex": "completedRawObservationIndex",
            "generation": "completedGeneration",
            "emergence": "completedEmergence",
            **{column: f"completed_{column}" for column in flag_columns},
        }
    )
    right = prefix_flagged[
        keys + ["rawObservationIndex", "generation", "emergence"] + flag_columns
    ].rename(
        columns={
            "rawObservationIndex": "prefixRawObservationIndex",
            "generation": "prefixGeneration",
            "emergence": "pastOnlyEmergence",
            **{column: f"pastOnly_{column}" for column in flag_columns},
        }
    )
    joined = right.merge(left, on=keys, how="inner", validate="one_to_one")
    joined["emergenceDifferenceCompletedMinusPastOnly"] = (
        joined["completedEmergence"] - joined["pastOnlyEmergence"]
    )
    history = partition_history.rename(
        columns={
            "endpointGeneration": "prefixGeneration",
            "endpointSelectedSequenceIndex": "selectedSequenceIndex",
        }
    )
    joined = joined.merge(
        history[
            [
                "candidateId",
                "trajectoryId",
                "matrixIndex",
                "prefixGeneration",
                "selectedSequenceIndex",
                "partitionChangedFromPreviousEligibleFit",
                "partitionARIFromPreviousEligibleFit",
            ]
        ],
        on=[
            "candidateId",
            "trajectoryId",
            "matrixIndex",
            "prefixGeneration",
            "selectedSequenceIndex",
        ],
        how="left",
        validate="one_to_one",
    )
    summary_rows: list[dict[str, Any]] = []
    for candidate_id, group in joined.groupby("candidateId", sort=True):
        trajectory_rhos: list[float] = []
        for _, trajectory in group.groupby("trajectoryId", sort=True):
            if len(trajectory) > 2:
                trajectory_rhos.append(
                    float(
                        spearmanr(
                            trajectory["completedEmergence"],
                            trajectory["pastOnlyEmergence"],
                        ).statistic
                    )
                )
        event_spearman = spearmanr(
            group["completedEmergence"], group["pastOnlyEmergence"]
        )
        event_pearson = pearsonr(
            group["completedEmergence"], group["pastOnlyEmergence"]
        )
        row: dict[str, Any] = {
            "candidateId": candidate_id,
            "sharedEndpointCount": len(group),
            "trajectoryCount": int(group["trajectoryId"].nunique()),
            "eventSpearmanRho": float(event_spearman.statistic),
            "eventSpearmanP": float(event_spearman.pvalue),
            "eventPearsonR": float(event_pearson.statistic),
            "eventPearsonP": float(event_pearson.pvalue),
            "medianTrajectorySpearmanRho": float(np.nanmedian(trajectory_rhos)),
            "medianCompletedMinusPastOnly": float(
                group["emergenceDifferenceCompletedMinusPastOnly"].median()
            ),
            "medianAbsoluteDifference": float(
                group["emergenceDifferenceCompletedMinusPastOnly"].abs().median()
            ),
            "rootMeanSquaredDifference": float(
                np.sqrt(
                    np.mean(
                        group["emergenceDifferenceCompletedMinusPastOnly"].to_numpy()
                        ** 2
                    )
                )
            ),
            "partitionChangeEvaluableEndpointCount": int(
                group["partitionChangedFromPreviousEligibleFit"].notna().sum()
            ),
            "partitionChangeFraction": float(
                group["partitionChangedFromPreviousEligibleFit"].dropna().mean()
            ),
        }
        for column in flag_columns:
            a = group[f"completed_{column}"].to_numpy(bool)
            b = group[f"pastOnly_{column}"].to_numpy(bool)
            union = int(np.count_nonzero(a | b))
            row[f"{column}_jaccard"] = (
                float(np.count_nonzero(a & b) / union) if union else 1.0
            )
        summary_rows.append(row)
    return (
        joined.sort_values(keys, kind="stable", ignore_index=True),
        pd.DataFrame(summary_rows).sort_values(
            "candidateId", kind="stable", ignore_index=True
        ),
    )


def combine_fission_dependency(dependency: pd.DataFrame) -> pd.DataFrame:
    """Pool per-trajectory 2x2 tables within candidates for descriptive tests."""

    rows: list[dict[str, Any]] = []
    keys = ["candidateId", "temporalMode", "thresholdFamily", "sign"]
    count_columns = [
        "excursionPostFissionPoints",
        "excursionOtherPoints",
        "nonExcursionPostFissionPoints",
        "nonExcursionOtherPoints",
    ]
    for identity, group in dependency.groupby(keys, sort=True):
        counts = [int(group[column].sum()) for column in count_columns]
        odds, pvalue = fisher_exact(
            [[counts[0], counts[1]], [counts[2], counts[3]]], alternative="two-sided"
        )
        rows.append(
            {
                **dict(zip(keys, identity, strict=True)),
                **dict(zip(count_columns, counts, strict=True)),
                "pooledPointFissionOddsRatio": float(odds),
                "pooledPointFissionFisherTwoSidedP": float(pvalue),
                "interpretationBoundary": "DESCRIPTIVE_POINT_LEVEL_ENRICHMENT_NOT_INDEPENDENT_CAUSAL_TEST",
            }
        )
    return pd.DataFrame(rows).sort_values(keys, kind="stable", ignore_index=True)
