from __future__ import annotations

import numpy as np
import pandas as pd

from e01_onset_discovery.survival import (
    INTERVAL_ENDS,
    build_risk_rows,
    build_survival_targets,
    concordance_index,
    cumulative_risk_from_hazards,
    survival_metrics,
)


def _geometry() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidateId": "C2",
                "matrixIndex": 0,
                "observationCount": 400,
                "firstOnsetIndex0": 100.0,
                "atRiskAtLandmark": True,
            },
            {
                "candidateId": "C2",
                "matrixIndex": 1,
                "observationCount": 400,
                "firstOnsetIndex0": 210.0,
                "atRiskAtLandmark": True,
            },
            {
                "candidateId": "C2",
                "matrixIndex": 2,
                "observationCount": 400,
                "firstOnsetIndex0": np.nan,
                "atRiskAtLandmark": True,
            },
            {
                "candidateId": "C2",
                "matrixIndex": 3,
                "observationCount": 150,
                "firstOnsetIndex0": np.nan,
                "atRiskAtLandmark": True,
            },
        ]
    )


def test_endpoint_and_risk_set_construction() -> None:
    targets = build_survival_targets(_geometry())
    assert targets["eventObservedBy320"].tolist() == [True, True, False, False]
    training = build_risk_rows(targets, include_post_event_grid=False)
    assert training.groupby("matrixIndex").size().to_dict() == {0: 1, 1: 3, 2: 4, 3: 1}
    assert training[training.matrixIndex.eq(0)]["eventInInterval"].tolist() == [True]
    assert training[training.matrixIndex.eq(1)]["eventInInterval"].tolist() == [False, False, True]


def test_cumulative_risk_identity() -> None:
    hazards = np.full((2, 4), 0.2)
    risk = cumulative_risk_from_hazards(hazards)
    expected = 1.0 - np.power(0.8, np.arange(1, 5))
    assert np.allclose(risk[0], expected)
    assert np.all(np.diff(risk, axis=1) >= 0.0)


def test_concordance_perfect_and_reversed() -> None:
    time = np.array([100, 150, 320])
    event = np.array([True, True, False])
    assert concordance_index(time, event, np.array([3.0, 2.0, 1.0])) == 1.0
    assert concordance_index(time, event, np.array([1.0, 2.0, 3.0])) == 0.0


def test_survival_metrics_schema() -> None:
    targets = build_survival_targets(_geometry().iloc[:3])
    hazards = np.array(
        [
            [0.9, 0.2, 0.1, 0.1],
            [0.1, 0.2, 0.8, 0.2],
            [0.1, 0.1, 0.1, 0.1],
        ]
    )
    metrics = survival_metrics(targets, hazards)
    assert {f"AUROC_{h}" for h in INTERVAL_ENDS}.issubset(metrics)
    assert 0.0 <= metrics["CINDEX"] <= 1.0
    assert 0.0 <= metrics["INTEGRATED_BRIER"] <= 1.0
