"""Source-isolated cellular-automaton plastic-heredity campaign."""

from .core import (
    BREAK_HORIZON,
    MAX_FUTURE_HORIZON,
    RENEWAL_RUN,
    FutureOutcome,
    canonical_similarity,
    score_break_renewal,
)

__all__ = [
    "BREAK_HORIZON",
    "MAX_FUTURE_HORIZON",
    "RENEWAL_RUN",
    "FutureOutcome",
    "canonical_similarity",
    "score_break_renewal",
]
