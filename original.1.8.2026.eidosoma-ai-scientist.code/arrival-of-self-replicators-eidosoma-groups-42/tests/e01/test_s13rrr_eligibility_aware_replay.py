from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from e01_s13rrr_eligibility_aware_replay.core import (
    ENSEMBLE_REPORTING_ORDER,
    ENSEMBLE_SOURCE_ORDER,
    EXACT_UNAVAILABLE_TASKS,
    PREFIX_REPORTING_ORDER,
    RESEARCH_STEP_ID,
    VERSION,
    expected_slots,
    reorder_columns_exact,
    sentinel_availability,
)

REPO = Path(__file__).resolve().parents[2]


def _prefix(generations: list[int]) -> pd.DataFrame:
    rows = []
    for implementation in ("IIGR_CORRECTED_SOURCE", "PHIRL_REGULARIZED_SOURCE"):
        for generation in generations:
            rows.append(
                {
                    "implementationId": implementation,
                    "generation": generation,
                    "priorLockedClockTransitions": 256,
                }
            )
    return pd.DataFrame(
        rows, columns=["implementationId", "generation", "priorLockedClockTransitions"]
    )


def test_preregistration_freezes_exact_post_outcome_exception() -> None:
    config = yaml.safe_load(
        (
            REPO
            / "configs/e01/s13rrr_eligibility_aware_replay_finalization_preregistration.yaml"
        ).read_text()
    )
    assert config["researchStepId"] == RESEARCH_STEP_ID
    assert config["versionedStepId"] == VERSION
    assert config["replayRuleOverride"]["oldFixedRequirement"] == 3600
    assert config["replayRuleOverride"]["newExactApplicableRequirement"] == 3552
    assert config["replayRuleOverride"]["notApplicableSlotCount"] == 48
    assert set(config["replayRuleOverride"]["exactUnavailableTasks"]) == set(
        EXACT_UNAVAILABLE_TASKS
    )
    assert (
        tuple(
            config["reportingOrderContract"]["exactTables"][
                "prefix_endpoint_values.parquet"
            ]
        )
        == PREFIX_REPORTING_ORDER
    )
    assert (
        tuple(
            config["reportingOrderContract"]["exactTables"]["ensemble_adjudication.csv"]
        )
        == ENSEMBLE_REPORTING_ORDER
    )
    assert set(ENSEMBLE_SOURCE_ORDER) == set(ENSEMBLE_REPORTING_ORDER)
    assert config["statisticsContract"]["executions"] == 2
    assert config["permanentStopRule"]["replayRuleChangePermitted"] is False


@pytest.mark.parametrize(
    ("generations", "applicable", "unavailable"),
    [([30, 50, 100], 3, 0), ([100], 1, 2), ([], 0, 3)],
)
def test_sentinel_availability_matches_source_precedence(
    generations: list[int], applicable: int, unavailable: int
) -> None:
    rows = sentinel_availability(generations)
    assert len(rows) == 3
    assert sum(row["applicable"] for row in rows) == applicable
    assert sum(not row["applicable"] for row in rows) == unavailable
    if len(generations) == 1:
        assert rows[0]["sentinel"] == "first" and rows[0]["applicable"]
        assert {row["sentinel"] for row in rows[1:]} == {"middle", "last"}


def test_expected_slot_counts_are_18_6_or_0() -> None:
    for generations, applicable_count, unavailable_count in (
        ([30, 50, 100], 18, 0),
        ([100], 6, 12),
        ([], 0, 18),
    ):
        applicable, unavailable = expected_slots(
            candidate_id="S12F-CANDIDATE-02",
            matrix_index=68,
            trajectory_id="T",
            prefix=_prefix(generations),
        )
        assert len(applicable) == applicable_count
        assert len(unavailable) == unavailable_count
        assert len(applicable) + len(unavailable) == 18


def test_reporting_reorder_preserves_all_values_and_rows() -> None:
    frame = pd.DataFrame({"b": [2, None], "a": [1, 3], "c": [True, False]})
    ordered = reorder_columns_exact(frame, ("a", "b", "c"))
    assert list(ordered.columns) == ["a", "b", "c"]
    pd.testing.assert_frame_equal(ordered[["b", "a", "c"]], frame, check_exact=True)


def test_reporting_reorder_rejects_field_addition_or_removal() -> None:
    frame = pd.DataFrame({"a": [1], "b": [2]})
    with pytest.raises(ValueError, match="field set differs"):
        reorder_columns_exact(frame, ("a",))
    with pytest.raises(ValueError, match="field set differs"):
        reorder_columns_exact(frame, ("a", "b", "c"))
