from __future__ import annotations

import pandas as pd
import pytest

from e01_aggregation_interface_repair.core import (
    ADAPTER_ID,
    CANDIDATE_IDS,
    DERIVED_FIELD,
    SOURCE_FIELD,
    adapt_prefix_statistical_view,
    validate_prefix_adapter,
)


def frozen_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    prefix_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    for candidate_ordinal, candidate_id in enumerate(CANDIDATE_IDS):
        for matrix_index in range(32):
            trajectory_id = f"{candidate_id}-M{matrix_index:02d}"
            for generation in range(1, 101):
                raw_index = candidate_ordinal * 100_000 + matrix_index * 1_000 + generation * 2
                label_rows.append(
                    {
                        "candidateId": candidate_id,
                        "trajectoryId": trajectory_id,
                        "matrixIndex": matrix_index,
                        "generation": generation,
                        "labelId": "HISTORICAL_H090_REPLICATOR",
                        "postFissionRawObservationIndex": raw_index,
                    }
                )
                for implementation_id in (
                    "IIGR_CORRECTED_SOURCE",
                    "PHIRL_REGULARIZED_SOURCE",
                ):
                    prefix_rows.append(
                        {
                            "researchStepId": "S12I",
                            "candidateId": candidate_id,
                            "trajectoryId": trajectory_id,
                            "matrixIndex": matrix_index,
                            "implementationId": implementation_id,
                            "generation": generation,
                            SOURCE_FIELD: raw_index,
                            "emergence": float(generation),
                            "status": "ELIGIBLE",
                        }
                    )
    return pd.DataFrame(prefix_rows), pd.DataFrame(label_rows)


def test_adapter_passes_complete_frozen_shape_without_mutating_source() -> None:
    prefix, labels = frozen_frames()
    before = prefix.copy(deep=True)
    adapted = adapt_prefix_statistical_view(prefix)
    validation, audit = validate_prefix_adapter(prefix, adapted, labels)
    pd.testing.assert_frame_equal(prefix, before, check_exact=True)
    pd.testing.assert_frame_equal(prefix, adapted[prefix.columns], check_exact=True)
    assert list(adapted.columns) == [*prefix.columns, DERIVED_FIELD]
    assert adapted[DERIVED_FIELD].equals(adapted[SOURCE_FIELD])
    assert validation["adapterId"] == ADAPTER_ID
    assert validation["inputRowCount"] == 19_200
    assert validation["monotonicGroupCount"] == 192
    assert validation["passed"] is True
    assert len(audit) == 19_200
    assert audit["rowOrdinal"].tolist() == list(range(19_200))


def test_adapter_rejects_missing_noninteger_null_or_preexisting_alias() -> None:
    prefix, _labels = frozen_frames()
    with pytest.raises(ValueError, match="missing required source field"):
        adapt_prefix_statistical_view(prefix.drop(columns=SOURCE_FIELD))
    with pytest.raises(ValueError, match="derived field already exists"):
        adapt_prefix_statistical_view(prefix.assign(**{DERIVED_FIELD: 1}))
    noninteger = prefix.copy()
    noninteger[SOURCE_FIELD] = noninteger[SOURCE_FIELD].astype(float)
    with pytest.raises(TypeError, match="not integer typed"):
        adapt_prefix_statistical_view(noninteger)
    null = prefix.copy()
    null.loc[0, SOURCE_FIELD] = None
    with pytest.raises(ValueError, match="contains nulls"):
        adapt_prefix_statistical_view(null)


def test_validation_fails_duplicate_or_endpoint_mismatch() -> None:
    prefix, labels = frozen_frames()
    duplicate = prefix.copy()
    duplicate.loc[1, "implementationId"] = duplicate.loc[0, "implementationId"]
    adapted = adapt_prefix_statistical_view(duplicate)
    validation, _audit = validate_prefix_adapter(duplicate, adapted, labels)
    assert validation["gates"]["frozenKeyUniqueness"] is False
    assert validation["passed"] is False

    adapted = adapt_prefix_statistical_view(prefix)
    mismatched_labels = labels.copy()
    mismatched_labels.loc[0, "postFissionRawObservationIndex"] += 1
    validation, _audit = validate_prefix_adapter(prefix, adapted, mismatched_labels)
    assert validation["gates"]["exactEndpointIdentityAgreement"] is False
    assert validation["passed"] is False
