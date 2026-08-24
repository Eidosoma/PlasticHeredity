"""Frozen helpers for E01/S19 Loop 2 replicator-definition analysis."""

from .core import (
    CANDIDATE_IDS,
    COMPARATOR_LABEL_ID,
    LABEL_DEFINITIONS,
    LOOP_ID,
    VERSION,
    LabelDefinition,
    derive_seed128,
    fingerprint_from_labels,
    paper_fingerprint_distance,
)

__all__ = [
    "CANDIDATE_IDS",
    "COMPARATOR_LABEL_ID",
    "LABEL_DEFINITIONS",
    "LOOP_ID",
    "VERSION",
    "LabelDefinition",
    "derive_seed128",
    "fingerprint_from_labels",
    "paper_fingerprint_distance",
]
