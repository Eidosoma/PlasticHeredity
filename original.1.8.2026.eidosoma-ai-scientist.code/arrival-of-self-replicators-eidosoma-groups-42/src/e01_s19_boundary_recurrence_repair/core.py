"""One-repair-only numerical replay policy for E01/S19-L06R.

This module changes no L06 scientific calculation.  It compares the frozen
canonical and independent CPU-float64 score paths using the already documented
S06 cross-platform policy while retaining exact discrete outputs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

VERSION = "E01-S19-L06R-NUMERICAL-EQUIVALENCE-CONFIRMATION-v1.0.0"
LOOP_ID = "S19-L06R"
ABSOLUTE_TOLERANCE = 1.0e-12
RELATIVE_TOLERANCE = 1.0e-12
MAXIMUM_ULP_DISTANCE = 8


def _ordered_uint64(values: NDArray[np.float64]) -> NDArray[np.uint64]:
    """Map binary64 values to monotonically ordered unsigned integers."""

    bits = np.ascontiguousarray(values, dtype=np.float64).view(np.uint64)
    sign = np.uint64(1 << 63)
    return np.where((bits & sign) != 0, ~bits, bits | sign).astype(np.uint64)


def ulp_distances(
    left: NDArray[np.float64], right: NDArray[np.float64]
) -> NDArray[np.uint64]:
    """Return binary64 ULP distances for aligned finite arrays."""

    a = _ordered_uint64(np.asarray(left, dtype=np.float64))
    b = _ordered_uint64(np.asarray(right, dtype=np.float64))
    return np.where(a >= b, a - b, b - a).astype(np.uint64)


def compare_float64_scores(
    canonical: NDArray[np.float64], independent: NDArray[np.float64]
) -> dict[str, Any]:
    """Apply the locked all-three-bounds numerical-equivalence contract."""

    left = np.asarray(canonical, dtype=np.float64)
    right = np.asarray(independent, dtype=np.float64)
    shape_equal = left.shape == right.shape
    if not shape_equal:
        return {
            "shapeExact": False,
            "finiteMaskExact": False,
            "nonfiniteClassExact": False,
            "finitePairCount": 0,
            "bitExactFinitePairCount": 0,
            "nonBitExactFinitePairCount": 0,
            "maximumAbsoluteError": None,
            "maximumRelativeError": None,
            "maximumUlpDistance": None,
            "absoluteTolerancePassed": False,
            "relativeTolerancePassed": False,
            "ulpTolerancePassed": False,
            "passed": False,
        }
    left_finite = np.isfinite(left)
    right_finite = np.isfinite(right)
    mask_exact = bool(np.array_equal(left_finite, right_finite))
    left_nonfinite = np.where(
        np.isnan(left),
        1,
        np.where(np.isposinf(left), 2, np.where(np.isneginf(left), 3, 0)),
    )
    right_nonfinite = np.where(
        np.isnan(right),
        1,
        np.where(np.isposinf(right), 2, np.where(np.isneginf(right), 3, 0)),
    )
    class_exact = bool(np.array_equal(left_nonfinite, right_nonfinite))
    if not mask_exact:
        finite = np.zeros(left.shape, dtype=bool)
    else:
        finite = left_finite
    a = left[finite]
    b = right[finite]
    absolute = np.abs(a - b)
    scale = np.maximum(np.abs(a), np.abs(b))
    relative = np.divide(absolute, scale, out=np.zeros_like(absolute), where=scale > 0)
    ulp = ulp_distances(a, b)
    bit_exact = a.view(np.uint64) == b.view(np.uint64)
    max_abs = float(np.max(absolute)) if len(absolute) else 0.0
    max_rel = float(np.max(relative)) if len(relative) else 0.0
    max_ulp = int(np.max(ulp)) if len(ulp) else 0
    abs_pass = bool(np.all(absolute <= ABSOLUTE_TOLERANCE))
    rel_pass = bool(np.all(relative <= RELATIVE_TOLERANCE))
    ulp_pass = bool(np.all(ulp <= MAXIMUM_ULP_DISTANCE))
    passed = bool(
        shape_equal
        and mask_exact
        and class_exact
        and abs_pass
        and rel_pass
        and ulp_pass
    )
    return {
        "shapeExact": shape_equal,
        "finiteMaskExact": mask_exact,
        "nonfiniteClassExact": class_exact,
        "finitePairCount": len(a),
        "bitExactFinitePairCount": int(np.count_nonzero(bit_exact)),
        "nonBitExactFinitePairCount": int(len(a) - np.count_nonzero(bit_exact)),
        "maximumAbsoluteError": max_abs,
        "maximumRelativeError": max_rel,
        "maximumUlpDistance": max_ulp,
        "absoluteTolerancePassed": abs_pass,
        "relativeTolerancePassed": rel_pass,
        "ulpTolerancePassed": ulp_pass,
        "passed": passed,
    }


def compare_discrete_recurrence(
    canonical: dict[str, Any], independent: dict[str, Any]
) -> dict[str, bool]:
    """Require exact labels, recurrence counts, alignments, and match identities."""

    array_fields = (
        "labels",
        "distinctPriorBoundaryCount",
        "qualifyingPriorBoundaryCount",
        "firstMatchingBoundaryGeneration",
        "lastMatchingBoundaryGeneration",
        "sourceBoundaryGeneration",
    )
    checks = {
        field: bool(np.array_equal(canonical[field], independent[field]))
        for field in array_fields
    }
    checks["matchingBoundaryGenerations"] = bool(
        canonical["matchingBoundaryGenerations"]
        == independent["matchingBoundaryGenerations"]
    )
    return checks
