"""Frozen identities and adjudication rules for S12I.

S12I does not reinterpret S12H's failed upstream gate.  It permits the
boundary-inclusive candidate-1 derivative only as a human-waived,
non-confirmed sensitivity case and carries candidates 2 and 3 unchanged.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

VERSION = "E01-S12I-AGGREGATE-SUPPORT-WAIVER-SENSITIVITY-v1.0.0"
RESEARCH_STEP_ID = "S12I"
EVIDENCE_CLASS = "HUMAN_WAIVED_SOURCE_INFORMED_SENSITIVITY_ANALYSIS"
DERIVED_CANDIDATE_ID = "S12H-CANDIDATE-01-C1-DERIVATIVE-v1.0.0"
CANDIDATE_IDS = (
    "S12F-CANDIDATE-01",
    "S12F-CANDIDATE-02",
    "S12F-CANDIDATE-03",
)


def sensitivity_candidate_registry() -> list[dict[str, Any]]:
    """Return the exact three S12I analysis identities."""

    return [
        {
            "candidateId": "S12F-CANDIDATE-01",
            "analysisIdentity": DERIVED_CANDIDATE_ID,
            "h": 0.5081160391061118,
            "daughterRule": "FIRST_DAUGHTER",
            "overshootRule": "RETAIN_OVERSHOOT",
            "clockId": "C1_SELECTED_DAUGHTER_RETAINED",
            "evidenceStatus": "HUMAN_WAIVED_NEAR_ENVELOPE_NONCONFIRMED",
            "aggregateSupportGateWaived": True,
            "upstreamConfirmed": False,
        },
        {
            "candidateId": "S12F-CANDIDATE-02",
            "analysisIdentity": "S12F-CANDIDATE-02",
            "h": 0.6031526490073492,
            "daughterRule": "FIRST_DAUGHTER",
            "overshootRule": "TRIM_NEW_ENTRANTS_TO_NMAX",
            "clockId": "C1_SELECTED_DAUGHTER_RETAINED",
            "evidenceStatus": "S12FR_UPSTREAM_CONFIRMED",
            "aggregateSupportGateWaived": False,
            "upstreamConfirmed": True,
        },
        {
            "candidateId": "S12F-CANDIDATE-03",
            "analysisIdentity": "S12F-CANDIDATE-03",
            "h": 0.5613315384859516,
            "daughterRule": "RANDOM_NONEMPTY",
            "overshootRule": "TRIM_NEW_ENTRANTS_TO_NMAX",
            "clockId": "C1_SELECTED_DAUGHTER_RETAINED",
            "evidenceStatus": "S12FR_UPSTREAM_CONFIRMED",
            "aggregateSupportGateWaived": False,
            "upstreamConfirmed": True,
        },
    ]


def validate_exact_waiver(s12h_confirmation: Mapping[str, Any]) -> dict[str, Any]:
    """Validate that S12I waives exactly the one observed S12H failure."""

    gates = dict(s12h_confirmation.get("gates", {}))
    failed = sorted(name for name, passed in gates.items() if not bool(passed))
    exact = (
        s12h_confirmation.get("confirmationGatePassed") is False
        and s12h_confirmation.get("aggregateCompatible") is False
        and failed == ["aggregateSupportCompatible"]
        and int(s12h_confirmation.get("sampleEndpointsInsideQ05Q95", -1)) == 3
        and float(s12h_confirmation.get("fractionBeyondAxisUpper1314", -1.0))
        == 0.0625
    )
    return {
        "originalConfirmationGatePassed": s12h_confirmation.get(
            "confirmationGatePassed"
        ),
        "originalAggregateSupportGatePassed": gates.get(
            "aggregateSupportCompatible"
        ),
        "failedGateNames": failed,
        "waivedGateName": "aggregateSupportCompatible",
        "allNonwaivedGatesPassed": all(
            bool(value)
            for name, value in gates.items()
            if name != "aggregateSupportCompatible"
        ),
        "originalFailureRetained": True,
        "waiverTreatedAsPass": False,
        "candidate1UpstreamConfirmed": False,
        "passed": bool(exact),
    }


def sensitivity_classification(
    candidate_rows: Iterable[Mapping[str, Any]],
) -> str:
    """Map unchanged S12G candidate gates to waiver-labeled conclusions."""

    rows = list(candidate_rows)
    if len(rows) != 3 or {row.get("candidateId") for row in rows} != set(
        CANDIDATE_IDS
    ):
        return "UNDERDETERMINED"
    prefix = [bool(row.get("primaryPrefixGate")) for row in rows]
    full = [bool(row.get("primaryFullCoherent")) for row in rows]
    if all(prefix):
        return "EXPLORATORY_SENSITIVITY_SET_PROSPECTIVE_POSITIVE_CONSISTENCY"
    if all(full):
        return "EXPLORATORY_SENSITIVITY_SET_RETROSPECTIVE_POSITIVE_CONSISTENCY"
    if any(prefix) or any(full):
        return "CANDIDATE_SENSITIVE_UNDERDETERMINED"
    return "SENSITIVITY_SET_WIDE_NON_SUPPORT_WITHIN_SOURCE_INFORMED_SCOPE"


def outcome_class(classification: str) -> str:
    """Return the workspace-level evidence class without overstating waiver data."""

    if classification.startswith("EXPLORATORY_SENSITIVITY_SET_"):
        return "supportive (exploratory sensitivity only)"
    if classification == "UNDERDETERMINED":
        return "null"
    return "constraining/contradictory"
