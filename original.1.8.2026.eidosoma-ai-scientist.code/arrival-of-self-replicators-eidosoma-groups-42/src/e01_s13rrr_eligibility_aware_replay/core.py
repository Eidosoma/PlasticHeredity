"""Pure contracts for the final S13RRR availability and reporting override."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

VERSION = "E01-S13RRR-ELIGIBILITY-AWARE-REPLAY-FINALIZATION-v1.0.0"
RESEARCH_STEP_ID = "S13RRR"
REPLAY_RULE_ID = "S13RRR_FROZEN_DATA_AVAILABILITY_AWARE_SUFFIX_GATE-v1.0.0"
REPORTING_ORDER_ID = "S13RRR_TWO_TABLE_VALUE_PRESERVING_REPORTING_ORDER-v1.0.0"

IMPLEMENTATIONS = ("IIGR_CORRECTED_SOURCE", "PHIRL_REGULARIZED_SOURCE")
VALIDATION_KINDS = (
    "suffix_deletion",
    "suffix_deterministic_shuffle",
    "suffix_domain_separated_replacement",
)
SENTINEL_NAMES = ("first", "middle", "last")

EXACT_UNAVAILABLE_TASKS = {
    "S12F-CANDIDATE-02/M68": 12,
    "S12F-CANDIDATE-02/M72": 18,
    "S12F-CANDIDATE-03/M72": 18,
}

PREFIX_REPORTING_ORDER = (
    "researchStepId",
    "candidateId",
    "trajectoryId",
    "matrixIndex",
    "clockId",
    "implementationId",
    "temporalModeId",
    "temporalLabel",
    "generation",
    "endpointSelectedSequenceIndex",
    "endpointRawObservationIndex",
    "endpointObservationKind",
    "priorLockedClockTransitions",
    "fitObservationCount",
    "status",
    "reason",
    "synergy",
    "downwardCausation",
    "emergence",
    "localPhiR",
    "historicalLabel",
    "nextHistoricalLabel",
    "pastOnlyCosineLabel",
    "exactReplayPassed",
    "futureSuffixStructuralGatePassed",
    "futureSuffixExecutedSentinelPassed",
)

ENSEMBLE_SOURCE_ORDER = (
    "candidateId",
    "candidateEvidenceStatus",
    "primaryFullAssociationGate",
    "primaryFullDriftGate",
    "primaryFullCoherent",
    "primaryPrefixGate",
    "combinedRetrospectiveAndProspectiveGate",
    "punctuatedGate",
    "phirlOppositeFull",
    "phirlOppositePrefix",
    "operationalCoverageGate",
    "candidateClassification",
)

ENSEMBLE_REPORTING_ORDER = (
    "candidateId",
    "primaryFullAssociationGate",
    "primaryFullDriftGate",
    "primaryFullCoherent",
    "primaryPrefixGate",
    "punctuatedGate",
    "phirlOppositeFull",
    "phirlOppositePrefix",
    "operationalCoverageGate",
    "candidateClassification",
    "candidateEvidenceStatus",
    "combinedRetrospectiveAndProspectiveGate",
)


def sentinel_availability(generations: Sequence[int]) -> list[dict[str, Any]]:
    """Return the frozen first/middle/last availability plan.

    The source stores unique sentinel generations and assigns labels with
    first-before-middle-before-last precedence. Thus one eligible endpoint
    yields one applicable ``first`` sentinel and two nominal duplicate slots.
    """

    ordered = [int(value) for value in generations]
    if not ordered:
        return [
            {
                "sentinel": name,
                "generation": None,
                "applicable": False,
                "reason": "NO_ENDPOINT_AT_OR_ABOVE_256_TRANSITIONS",
            }
            for name in SENTINEL_NAMES
        ]
    nominal = {
        "first": ordered[0],
        "middle": ordered[len(ordered) // 2],
        "last": ordered[-1],
    }
    seen: set[int] = set()
    rows: list[dict[str, Any]] = []
    for name in SENTINEL_NAMES:
        generation = nominal[name]
        applicable = generation not in seen
        rows.append(
            {
                "sentinel": name,
                "generation": generation,
                "applicable": applicable,
                "reason": (
                    None
                    if applicable
                    else "DUPLICATE_SENTINEL_GENERATION_COLLAPSED_BY_SOURCE_PRECEDENCE"
                ),
            }
        )
        seen.add(generation)
    return rows


def expected_slots(
    *, candidate_id: str, matrix_index: int, trajectory_id: str, prefix: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Materialize applicable and unavailable nominal suffix identities."""

    applicable: list[dict[str, Any]] = []
    unavailable: list[dict[str, Any]] = []
    for implementation_id in IMPLEMENTATIONS:
        subset = prefix[
            (prefix["implementationId"] == implementation_id)
            & (prefix["priorLockedClockTransitions"] >= 256)
        ].sort_values("generation")
        plan = sentinel_availability(subset["generation"].astype(int).tolist())
        for item in plan:
            for validation_kind in VALIDATION_KINDS:
                row = {
                    "candidateId": candidate_id,
                    "matrixIndex": int(matrix_index),
                    "trajectoryId": trajectory_id,
                    "implementationId": implementation_id,
                    "eligibleEndpointCount": len(subset),
                    "nominalSentinel": item["sentinel"],
                    "endpointGeneration": item["generation"],
                    "validationKind": validation_kind,
                }
                if item["applicable"]:
                    applicable.append(
                        {
                            **row,
                            "availabilityStatus": "APPLICABLE_EXECUTION_REQUIRED",
                            "availabilityReason": None,
                        }
                    )
                else:
                    unavailable.append(
                        {
                            **row,
                            "availabilityStatus": "NOT_APPLICABLE_FROZEN_ENDPOINT_AVAILABILITY",
                            "availabilityReason": item["reason"],
                        }
                    )
    return pd.DataFrame(applicable), pd.DataFrame(unavailable)


def reorder_columns_exact(frame: pd.DataFrame, order: Sequence[str]) -> pd.DataFrame:
    """Reorder only when the declared and observed field sets are identical."""

    expected = list(order)
    observed = list(frame.columns)
    if len(expected) != len(set(expected)):
        raise ValueError("declared reporting order contains duplicate fields")
    if set(observed) != set(expected) or len(observed) != len(expected):
        raise ValueError(
            f"reporting field set differs: missing={sorted(set(expected) - set(observed))}, "
            f"extra={sorted(set(observed) - set(expected))}"
        )
    return frame.loc[:, expected].copy()
