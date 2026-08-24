"""Frozen identities, scaled gates, and adjudication for E01 S13.

The simulator remains the exact S12FR-confirmed S12F implementation.  This
module adds only a new held-out seed domain and the pre-outcome, sample-size
scaled form of the S12G/S12J decision gates.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping
from typing import Any

VERSION = "E01-S13-CONFIRMED-TIMEBASE-BASELINE-SCALEUP-v1.0.0"
RESEARCH_STEP_ID = "S13"
EVIDENCE_CLASS = "HELD_OUT_SOURCE_INFORMED_TWO_CANDIDATE_BASELINE_SCALEUP"
ANALYSIS_ROOT_SEED_HEX = (
    "f1748908fc1eddb34e2767fb5ca88d045a47003af29caeef4c7ffcb94466e21d"
)
CANDIDATE_IDS = ("S12F-CANDIDATE-02", "S12F-CANDIDATE-03")

N_UNITS = 100
DEFINED_AT_LEAST = 80
POSITIVE_ASSOCIATIONS_AT_LEAST = 75
POSITIVE_DRIFT_AT_LEAST = math.ceil((14 / 24) * N_UNITS)
SPIKED_RUNS_AT_LEAST = 75
RAW_LJUNG_BOX_AT_LEAST = math.ceil((21 / 24) * N_UNITS)


def candidate_registry() -> list[dict[str, Any]]:
    """Return the only two candidates authorized for S13."""

    return [
        {
            "candidateId": "S12F-CANDIDATE-02",
            "heldoutAnalysisId": "S13-HELDOUT-CANDIDATE-02-v1.0.0",
            "h": 0.6031526490073492,
            "exposureFamily": "FIXED_COMMON_EXPOSURE",
            "daughterRule": "FIRST_DAUGHTER",
            "overshootRule": "TRIM_NEW_ENTRANTS_TO_NMAX",
            "clockId": "C1_SELECTED_DAUGHTER_RETAINED",
            "evidenceStatus": "S12FR_UPSTREAM_CONFIRMED",
        },
        {
            "candidateId": "S12F-CANDIDATE-03",
            "heldoutAnalysisId": "S13-HELDOUT-CANDIDATE-03-v1.0.0",
            "h": 0.5613315384859516,
            "exposureFamily": "FIXED_COMMON_EXPOSURE",
            "daughterRule": "RANDOM_NONEMPTY",
            "overshootRule": "TRIM_NEW_ENTRANTS_TO_NMAX",
            "clockId": "C1_SELECTED_DAUGHTER_RETAINED",
            "evidenceStatus": "S12FR_UPSTREAM_CONFIRMED",
        },
    ]


def derive_analysis_seed(*identity: object) -> int:
    """Derive a deterministic 32-bit source/statistics seed in the S13 domain."""

    material = analysis_seed_material(*identity)
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


def analysis_seed_material(*identity: object) -> bytes:
    """Return the canonical domain-separated material behind an analysis seed."""

    return "\x1f".join(
        [VERSION, ANALYSIS_ROOT_SEED_HEX, "analysis", *map(str, identity)]
    ).encode("utf-8")


def association_gate(summary: Any, coverage: float) -> bool:
    """Apply the frozen S12J association gate scaled to 100 units."""

    return bool(
        coverage >= 0.80
        and summary.defined_count >= DEFINED_AT_LEAST
        and summary.positive_count >= POSITIVE_ASSOCIATIONS_AT_LEAST
        and summary.median is not None
        and summary.median > 0
        and summary.bootstrap_lower_95 is not None
        and summary.bootstrap_lower_95 > 0
        and summary.circular_shift_positive_p is not None
        and summary.circular_shift_positive_p <= 0.05
    )


def drift_gate(summary: Any) -> bool:
    """Apply the frozen S12J drift gate scaled to 100 units."""

    return bool(
        summary.defined_count >= DEFINED_AT_LEAST
        and summary.positive_count >= POSITIVE_DRIFT_AT_LEAST
        and summary.median_mean_difference is not None
        and summary.median_mean_difference > 0
        and summary.bootstrap_lower_95 is not None
        and summary.bootstrap_lower_95 > 0
        and summary.block_aware_positive_p is not None
        and summary.block_aware_positive_p <= 0.05
    )


def combined_classification(candidate_rows: Iterable[Mapping[str, Any]]) -> str:
    """Apply the confirmatory two-candidate all-pass rule."""

    rows = list(candidate_rows)
    if len(rows) != 2 or {row.get("candidateId") for row in rows} != set(CANDIDATE_IDS):
        return "S13_VALIDATION_FAILED_CLOSED"
    combined = [
        bool(row.get("primaryFullCoherent")) and bool(row.get("primaryPrefixGate"))
        for row in rows
    ]
    if all(combined):
        return "HELD_OUT_TWO_CANDIDATE_RETROSPECTIVE_AND_PROSPECTIVE_SUPPORT"
    if any(combined):
        return "HELD_OUT_TWO_CANDIDATE_CANDIDATE_SENSITIVE_UNDERDETERMINATION"
    return "HELD_OUT_TWO_CANDIDATE_SCALEUP_NON_SUPPORT_WITHIN_SOURCE_INFORMED_SCOPE"


def outcome_class(classification: str) -> str:
    if classification == "HELD_OUT_TWO_CANDIDATE_RETROSPECTIVE_AND_PROSPECTIVE_SUPPORT":
        return "supportive_with_conflicting_prior_evidence"
    if classification.endswith("UNDERDETERMINATION"):
        return "null"
    return "constraining/contradictory"
