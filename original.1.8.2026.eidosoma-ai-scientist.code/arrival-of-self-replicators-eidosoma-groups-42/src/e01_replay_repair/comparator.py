"""Narrow schema-causal exact comparator for the separately versioned S12FR step.

The only normalization is paired IEEE NaN at the two exposure-extrema fields
of matching zero-update generation summaries.  Finite values are compared by
their binary64 bit patterns; all discrete structure remains exact.
"""

from __future__ import annotations

import math
import struct
from dataclasses import asdict, dataclass, fields, is_dataclass
from typing import Any

import numpy as np

from e01_latent_timebase.core import (
    SeedIdentity,
    TimebaseTrajectory,
    trajectory_replay_equal,
)

COMPARATOR_VERSION = "S12FR_SCHEMA_CAUSAL_EXACT_COMPARATOR_v1.0.0"
PERMITTED_NAN_FIELDS = frozenset({"maximum_exposure", "minimum_exposure"})


@dataclass(frozen=True, slots=True)
class FieldDifference:
    path: str
    category: str
    left: str
    right: str
    permitted: bool
    deterministic_cause: str | None

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TrajectoryComparison:
    old_comparator_passed: bool
    repaired_comparator_passed: bool
    discrete_divergence_count: int
    finite_numeric_divergence_count: int
    permitted_paired_nan_count: int
    forbidden_nonfinite_difference_count: int
    differences: tuple[FieldDifference, ...]


def float_bits(value: float) -> str:
    """Return the exact IEEE-754 binary64 representation."""

    return struct.pack(">d", float(value)).hex()


def _value_repr(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number):
            return "NaN"
        if math.isinf(number):
            return "+Infinity" if number > 0 else "-Infinity"
        return f"{number.hex()}|bits={float_bits(number)}"
    if isinstance(value, np.ndarray):
        return f"array(dtype={value.dtype},shape={value.shape})"
    text = repr(value)
    return text if len(text) <= 256 else text[:253] + "..."


def _paired_nan_is_permitted(path: str, field_name: str | None, left_parent: Any, right_parent: Any) -> bool:
    if field_name not in PERMITTED_NAN_FIELDS:
        return False
    if ".generations[" not in f".{path}":
        return False
    return bool(
        getattr(left_parent, "update_count", None) == 0
        and getattr(right_parent, "update_count", None) == 0
    )


def _compare_values(
    left: Any,
    right: Any,
    path: str,
    differences: list[FieldDifference],
    *,
    field_name: str | None = None,
    left_parent: Any = None,
    right_parent: Any = None,
) -> None:
    if type(left) is not type(right):
        differences.append(
            FieldDifference(
                path,
                "DISCRETE_TYPE_DIVERGENCE",
                type(left).__name__,
                type(right).__name__,
                False,
                None,
            )
        )
        return

    if isinstance(left, (float, np.floating)):
        left_value = float(left)
        right_value = float(right)
        left_nan = math.isnan(left_value)
        right_nan = math.isnan(right_value)
        if left_nan or right_nan:
            permitted = bool(
                left_nan
                and right_nan
                and _paired_nan_is_permitted(
                    path, field_name, left_parent, right_parent
                )
            )
            differences.append(
                FieldDifference(
                    path,
                    "PAIRED_SCHEMA_UNDEFINED_NAN_ZERO_UPDATE"
                    if permitted
                    else "FORBIDDEN_NONFINITE_DIVERGENCE",
                    _value_repr(left_value),
                    _value_repr(right_value),
                    permitted,
                    "NO_GROWTH_UPDATE_EXPOSURE_ACCUMULATOR_EMPTY"
                    if permitted
                    else None,
                )
            )
            return
        if math.isinf(left_value) or math.isinf(right_value):
            differences.append(
                FieldDifference(
                    path,
                    "FORBIDDEN_NONFINITE_DIVERGENCE",
                    _value_repr(left_value),
                    _value_repr(right_value),
                    False,
                    None,
                )
            )
            return
        if float_bits(left_value) != float_bits(right_value):
            differences.append(
                FieldDifference(
                    path,
                    "FINITE_NUMERIC_BIT_DIVERGENCE",
                    _value_repr(left_value),
                    _value_repr(right_value),
                    False,
                    None,
                )
            )
        return

    if isinstance(left, np.ndarray):
        if left.dtype != right.dtype or left.shape != right.shape:
            differences.append(
                FieldDifference(
                    path,
                    "DISCRETE_ARRAY_SCHEMA_DIVERGENCE",
                    _value_repr(left),
                    _value_repr(right),
                    False,
                    None,
                )
            )
        elif left.tobytes(order="C") != right.tobytes(order="C"):
            category = (
                "FINITE_NUMERIC_BIT_DIVERGENCE"
                if np.issubdtype(left.dtype, np.floating)
                else "DISCRETE_ARRAY_VALUE_DIVERGENCE"
            )
            differences.append(
                FieldDifference(path, category, _value_repr(left), _value_repr(right), False, None)
            )
        return

    if is_dataclass(left) and not isinstance(left, type):
        for field in fields(left):
            child_path = f"{path}.{field.name}" if path else field.name
            _compare_values(
                getattr(left, field.name),
                getattr(right, field.name),
                child_path,
                differences,
                field_name=field.name,
                left_parent=left,
                right_parent=right,
            )
        return

    if isinstance(left, (tuple, list)):
        if len(left) != len(right):
            differences.append(
                FieldDifference(
                    path,
                    "DISCRETE_SEQUENCE_LENGTH_DIVERGENCE",
                    str(len(left)),
                    str(len(right)),
                    False,
                    None,
                )
            )
            return
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            _compare_values(
                left_item,
                right_item,
                f"{path}[{index}]",
                differences,
                left_parent=left,
                right_parent=right,
            )
        return

    if isinstance(left, dict):
        if list(left) != list(right):
            differences.append(
                FieldDifference(
                    path,
                    "DISCRETE_MAPPING_KEY_DIVERGENCE",
                    repr(list(left)),
                    repr(list(right)),
                    False,
                    None,
                )
            )
            return
        for key in left:
            _compare_values(
                left[key], right[key], f"{path}[{key!r}]", differences
            )
        return

    if left != right:
        differences.append(
            FieldDifference(
                path,
                "DISCRETE_VALUE_DIVERGENCE",
                _value_repr(left),
                _value_repr(right),
                False,
                None,
            )
        )


def _counts(differences: list[FieldDifference]) -> tuple[int, int, int, int]:
    discrete = sum(row.category.startswith("DISCRETE_") for row in differences)
    finite = sum(row.category == "FINITE_NUMERIC_BIT_DIVERGENCE" for row in differences)
    permitted = sum(row.permitted for row in differences)
    forbidden = sum(row.category == "FORBIDDEN_NONFINITE_DIVERGENCE" for row in differences)
    return discrete, finite, permitted, forbidden


def compare_trajectories(
    left: TimebaseTrajectory, right: TimebaseTrajectory
) -> TrajectoryComparison:
    differences: list[FieldDifference] = []
    _compare_values(left, right, "trajectory", differences)
    discrete, finite, permitted, forbidden = _counts(differences)
    repaired = bool(discrete == 0 and finite == 0 and forbidden == 0)
    return TrajectoryComparison(
        old_comparator_passed=trajectory_replay_equal(left, right),
        repaired_comparator_passed=repaired,
        discrete_divergence_count=discrete,
        finite_numeric_divergence_count=finite,
        permitted_paired_nan_count=permitted,
        forbidden_nonfinite_difference_count=forbidden,
        differences=tuple(differences),
    )


def compare_seed_tuples(
    left: tuple[SeedIdentity, ...], right: tuple[SeedIdentity, ...]
) -> tuple[bool, tuple[FieldDifference, ...]]:
    differences: list[FieldDifference] = []
    _compare_values(left, right, "seeds", differences)
    return not differences, tuple(differences)
