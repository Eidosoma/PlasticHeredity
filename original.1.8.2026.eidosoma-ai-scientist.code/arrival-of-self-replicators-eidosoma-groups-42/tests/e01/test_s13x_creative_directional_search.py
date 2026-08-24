from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from e01_creative_directional_search.core import (
    LabelSpec,
    align_labels,
    association_summary,
    binary_spearman,
    derive_seed,
    label_specs,
    resemblance_score,
    transform_values,
)


def test_seed_replay_and_registry_identity() -> None:
    assert derive_seed("x", 1) == derive_seed("x", 1)
    assert derive_seed("x", 1) != derive_seed("x", 2)
    specs = label_specs()
    assert len(specs) == len({item.label_id for item in specs}) == 22
    assert any(item.evidence_tier == "SOURCE_GROUNDED" for item in specs)
    assert any(item.evidence_tier == "TABLE1_DIRECTED_SPECULATIVE" for item in specs)


def test_binary_spearman_and_association_direction() -> None:
    values = np.arange(8, dtype=float)
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)
    rho, p = binary_spearman(values, labels)
    assert rho is not None and rho > 0.8
    assert p is not None and p < 0.05
    result = association_summary(values, labels)
    assert result["meanDifference"] == 4.0
    assert result["standardizedMeanDifference"] is not None


def test_transforms_and_label_alignment_are_explicit() -> None:
    frame = pd.DataFrame(
        {
            "selectedSequenceIndex": [1, 2, 3, 4],
            "rawObservationIndex": [10, 11, 12, 13],
            "generation": [1, 1, 2, 2],
            "emergence": [1.0, 3.0, 2.0, 6.0],
        }
    )
    labels = pd.DataFrame(
        {
            "selectedSequenceIndex": [0, 1, 2, 3, 4, 5],
            "generation": [0, 1, 1, 2, 2, 3],
            "isReplicator": [False, False, True, True, True, False],
        }
    )
    diff = transform_values(frame, "emergence", "BACKWARD_DIFFERENCE")
    assert np.isnan(diff.iloc[0]["value"])
    assert diff.iloc[1]["value"] == 2.0
    same = align_labels(diff, labels, alignment="SAME_STATE", transform="BACKWARD_DIFFERENCE")
    nxt = align_labels(diff, labels, alignment="NEXT_STATE", transform="BACKWARD_DIFFERENCE")
    assert same.tolist() == [0.0, 1.0, 1.0, 1.0]
    assert nxt.tolist() == [1.0, 1.0, 1.0, 0.0]
    generation = transform_values(frame, "emergence", "GENERATION_MEAN")
    assert generation["value"].tolist() == [2.0, 4.0]


def test_resemblance_is_continuous_and_bounded() -> None:
    ideal = {
        "positiveCorrelationFraction": 0.73,
        "positiveSignificantFraction": 0.54,
        "medianCorrelation": 0.1,
        "higherDuringReplicationFraction": 0.57,
        "medianStandardizedMeanDifference": 0.2,
        "positiveThreeSigmaRunFraction": 0.8,
        "robustSpikeRunFraction": 0.8,
        "aggregateTrendP": 0.2,
        "rawLjungBoxFraction": 0.86,
        "differencedLjungBoxFraction": 1.0,
        "labelProbabilityMean": 0.88,
        "labelPersistenceMean": 716.0,
        "labelConsistencyMean": 0.38,
        "labelTimeToFirstMean": 37.0,
    }
    low = {key: 0.0 for key in ideal}
    high_score = resemblance_score(ideal)["directionalResemblanceScore"]
    low_score = resemblance_score(low)["directionalResemblanceScore"]
    assert 0.0 <= low_score < high_score <= 1.0


def test_label_spec_is_immutable() -> None:
    spec = LabelSpec("x", "f", 0.9, "tier", "why")
    assert spec.threshold == 0.9
    assert isinstance(SimpleNamespace(spec=spec).spec, LabelSpec)
