"""Deterministic source-grounding helpers for E01/S19-L16.

L16 is intentionally gated before scientific model execution.  A candidate
prediction convention is executable only when every identity-changing tensor,
loss, score, aggregation, and architecture field is directly specified by the
paper or by public code explicitly linked to the paper's GARD Figure-5 task.
Partial plotting or generic reinforcement-learning code is retained as a clue,
but it cannot silently become an author implementation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

VERSION = "E01-S19-L16-FIGURE5-TENSOR-ARCHITECTURE-DISCRIMINATION-v1.0.0"

REQUIRED_GROUNDING_FIELDS = (
    "input_sequence_representation",
    "variable_length_normalization",
    "input_padding_or_truncation",
    "target_sequence_representation",
    "target_padding_or_truncation",
    "training_loss_and_mask",
    "scoring_mask",
    "output_aggregation",
    "architecture_topology",
    "architecture_capacity",
)

ACCEPTABLE_EVIDENCE = {
    "DIRECT_PAPER_SPECIFICATION",
    "DIRECT_PUBLIC_CODE_EXPLICITLY_LINKED_TO_GARD_FIGURE5",
}


def canonical_json(value: object) -> str:
    """Return stable compact JSON, rejecting non-finite numbers."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_file(path: Path) -> str:
    """Hash a file without loading it in memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(values: NDArray[Any]) -> str:
    """Replay the frozen E01 dtype/shape/bytes array identity."""

    value = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(b"\0")
    digest.update(str(tuple(value.shape)).encode())
    digest.update(b"\0")
    digest.update(value.tobytes())
    return digest.hexdigest()


def assess_hypothesis(
    support_by_field: Mapping[str, str],
) -> dict[str, object]:
    """Apply the prospectively locked complete-source-grounding gate."""

    unknown = sorted(set(support_by_field) - set(REQUIRED_GROUNDING_FIELDS))
    if unknown:
        raise ValueError(f"unregistered grounding fields: {unknown}")
    missing = [field for field in REQUIRED_GROUNDING_FIELDS if field not in support_by_field]
    unsupported = [
        field
        for field in REQUIRED_GROUNDING_FIELDS
        if support_by_field.get(field) not in ACCEPTABLE_EVIDENCE
    ]
    passed = not missing and not unsupported
    return {
        "requiredFieldCount": len(REQUIRED_GROUNDING_FIELDS),
        "specifiedFieldCount": len(support_by_field),
        "directlyGroundedFieldCount": sum(
            support_by_field.get(field) in ACCEPTABLE_EVIDENCE
            for field in REQUIRED_GROUNDING_FIELDS
        ),
        "missingFields": missing,
        "unsupportedFields": unsupported,
        "completeSourceGroundingPassed": passed,
        "registeredForExecution": passed,
    }
