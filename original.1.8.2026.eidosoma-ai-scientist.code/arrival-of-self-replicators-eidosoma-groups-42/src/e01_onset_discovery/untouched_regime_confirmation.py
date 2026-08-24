"""Small contracts for untouched plastic-heredity risk confirmation."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


def confirmation_state_id(
    version: str, candidate_id: str, matrix_index: int, landmark: int
) -> str:
    """Return a deterministic, outcome-independent confirmation-state identity."""

    if matrix_index < 0 or landmark <= 0:
        raise ValueError("invalid confirmation state coordinates")
    payload = f"{version}|{candidate_id}|{matrix_index}|{landmark}".encode()
    return hashlib.sha256(payload).hexdigest()[:24]


def scientific_manifest_equal(
    left: Sequence[Mapping[str, object]],
    right: Sequence[Mapping[str, object]],
    fields: Sequence[str],
) -> bool:
    """Compare ordered scientific manifest fields while ignoring cache paths/timing."""

    if len(left) != len(right) or not fields:
        return False
    return all(
        tuple(a.get(field) for field in fields)
        == tuple(b.get(field) for field in fields)
        for a, b in zip(left, right, strict=True)
    )


def exact_probability_replay(
    expected: NDArray[np.floating], actual: NDArray[np.floating]
) -> bool:
    """Require bit-exact finite predictions under the frozen CPU-float64 path."""

    left = np.asarray(expected, dtype=np.float64)
    right = np.asarray(actual, dtype=np.float64)
    return bool(
        left.shape == right.shape
        and np.isfinite(left).all()
        and np.isfinite(right).all()
        and np.array_equal(left, right)
    )


def confirmation_gate(
    *,
    availability: bool,
    reliability: bool,
    proper_score: bool,
    overall_rank: bool,
    within_matrix_rank: bool,
    permutation: bool,
    replay: bool,
) -> bool:
    """Conjunctive untouched-confirmation gate; no component can rescue another."""

    return all(
        (
            availability,
            reliability,
            proper_score,
            overall_rank,
            within_matrix_rank,
            permutation,
            replay,
        )
    )
