"""Discrete-time survival utilities for the frozen S19-L21 onset task."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

LANDMARK = 64
INTERVAL_ENDS = (128, 192, 256, 320)
INTERVAL_STARTS = (LANDMARK, *INTERVAL_ENDS[:-1])


def build_survival_targets(target_geometry: pd.DataFrame) -> pd.DataFrame:
    """Create one right-censored endpoint per landmark-at-risk matrix."""
    required = {
        "candidateId",
        "matrixIndex",
        "observationCount",
        "firstOnsetIndex0",
        "atRiskAtLandmark",
    }
    if not required.issubset(target_geometry.columns):
        raise ValueError("target geometry is missing required fields")
    rows: list[dict[str, Any]] = []
    for item in target_geometry[target_geometry["atRiskAtLandmark"]].itertuples(
        index=False
    ):
        count = int(item.observationCount)
        onset = float(item.firstOnsetIndex0)
        onset_defined = np.isfinite(onset) and onset >= LANDMARK
        administrative = min(count, INTERVAL_ENDS[-1])
        event = bool(onset_defined and onset < administrative)
        observed = int(onset) if event else administrative
        rows.append(
            {
                "candidateId": item.candidateId,
                "matrixIndex": int(item.matrixIndex),
                "observationCount": count,
                "firstOnsetIndex0": int(onset) if onset_defined else np.nan,
                "eventObservedBy320": event,
                "observedTime": observed,
                "administrativeEnd": administrative,
                "fullyObservedThrough320": count >= INTERVAL_ENDS[-1],
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["candidateId", "matrixIndex"])
        .reset_index(drop=True)
    )


def build_risk_rows(targets: pd.DataFrame, include_post_event_grid: bool) -> pd.DataFrame:
    """Create training risk rows or a complete prediction grid.

    Training rows stop after the first event and exclude intervals censored before
    their endpoint.  Prediction-grid rows retain every fully observed interval so
    cumulative risks can be evaluated at all registered horizons.
    """
    rows: list[dict[str, Any]] = []
    for item in targets.itertuples(index=False):
        onset = float(item.firstOnsetIndex0)
        for interval_id, (start, end) in enumerate(
            zip(INTERVAL_STARTS, INTERVAL_ENDS, strict=True)
        ):
            fully_observed = int(item.observationCount) >= end
            event_in_interval = bool(
                np.isfinite(onset) and onset >= start and onset < end
            )
            if include_post_event_grid:
                # A landmark model can issue a hazard for every registered future
                # interval even when follow-up is later censored.  Scoring, not
                # prediction, excludes endpoints with insufficient follow-up.
                pass
            else:
                if np.isfinite(onset) and onset < start:
                    break
                if not fully_observed and not event_in_interval:
                    break
            row = {
                "candidateId": item.candidateId,
                "matrixIndex": int(item.matrixIndex),
                "intervalId": interval_id,
                "intervalStart": start,
                "intervalEnd": end,
                "eventInInterval": event_in_interval,
                "interval1": float(interval_id == 1),
                "interval2": float(interval_id == 2),
                "interval3": float(interval_id == 3),
            }
            rows.append(row)
            if event_in_interval and not include_post_event_grid:
                break
    return (
        pd.DataFrame(rows)
        .sort_values(["candidateId", "matrixIndex", "intervalId"])
        .reset_index(drop=True)
    )


def cumulative_risk_from_hazards(
    hazards: NDArray[np.float64],
) -> NDArray[np.float64]:
    values = np.clip(np.asarray(hazards, dtype=np.float64), 0.0, 1.0)
    if values.ndim != 2 or values.shape[1] != len(INTERVAL_ENDS):
        raise ValueError("hazards must be n-by-4")
    return 1.0 - np.cumprod(1.0 - values, axis=1)


def concordance_index(
    observed_time: NDArray[np.integer[Any]],
    event: NDArray[np.bool_],
    risk_score: NDArray[np.float64],
) -> float:
    time = np.asarray(observed_time, dtype=np.float64)
    status = np.asarray(event, dtype=bool)
    risk = np.asarray(risk_score, dtype=np.float64)
    earlier = (time[:, None] < time[None, :]) & status[:, None]
    comparable = earlier & (time[None, :] >= time[:, None])
    i, j = np.nonzero(comparable)
    if len(i) == 0:
        return float("nan")
    wins = (risk[i] > risk[j]).astype(float)
    wins += 0.5 * (risk[i] == risk[j])
    return float(np.mean(wins))


def survival_metrics(
    targets: pd.DataFrame, hazards: NDArray[np.float64]
) -> dict[str, float]:
    if len(targets) != len(hazards):
        raise ValueError("target and hazard row counts differ")
    cumulative = cumulative_risk_from_hazards(hazards)
    onset = targets["firstOnsetIndex0"].to_numpy(float)
    observation_count = targets["observationCount"].to_numpy(int)
    briers: list[float] = []
    result: dict[str, float] = {}
    for column, horizon in enumerate(INTERVAL_ENDS):
        positive = np.isfinite(onset) & (onset < horizon)
        eligible = positive | (observation_count >= horizon)
        y = positive[eligible].astype(int)
        p = cumulative[eligible, column]
        result[f"AUROC_{horizon}"] = (
            float(roc_auc_score(y, p)) if np.unique(y).size == 2 else float("nan")
        )
        result[f"AUPRC_{horizon}"] = (
            float(average_precision_score(y, p))
            if np.unique(y).size == 2
            else float("nan")
        )
        result[f"BRIER_{horizon}"] = float(brier_score_loss(y, p))
        result[f"PREVALENCE_{horizon}"] = float(np.mean(y))
        result[f"ELIGIBLE_{horizon}"] = float(len(y))
        briers.append(result[f"BRIER_{horizon}"])
    risk_score = np.sum(cumulative, axis=1)
    result["CINDEX"] = concordance_index(
        targets["observedTime"].to_numpy(int),
        targets["eventObservedBy320"].to_numpy(bool),
        risk_score,
    )
    result["INTEGRATED_BRIER"] = float(np.mean(briers))
    return result
