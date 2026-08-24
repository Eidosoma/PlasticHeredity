import numpy as np

from e01_s19_boundary_recurrence.core import (
    boundary_recurrence,
    boundary_recurrence_reference,
)
from e01_s19_boundary_recurrence_repair.core import (
    ABSOLUTE_TOLERANCE,
    MAXIMUM_ULP_DISTANCE,
    RELATIVE_TOLERANCE,
    compare_discrete_recurrence,
    compare_float64_scores,
)


def test_contract_constants_are_frozen() -> None:
    assert ABSOLUTE_TOLERANCE == 1e-12
    assert RELATIVE_TOLERANCE == 1e-12
    assert MAXIMUM_ULP_DISTANCE == 8


def test_eight_ulps_pass_and_nine_ulps_fail() -> None:
    base = np.asarray([0.91], dtype=np.float64)
    eight = base.copy()
    nine = base.copy()
    for _ in range(8):
        eight = np.nextafter(eight, np.inf)
    for _ in range(9):
        nine = np.nextafter(nine, np.inf)
    assert compare_float64_scores(base, eight)["passed"]
    result = compare_float64_scores(base, nine)
    assert result["maximumUlpDistance"] == 9
    assert not result["passed"]


def test_nonfinite_masks_and_classes_are_exact() -> None:
    assert compare_float64_scores(
        np.asarray([np.nan, np.inf]), np.asarray([np.nan, np.inf])
    )["passed"]
    assert not compare_float64_scores(np.asarray([np.nan]), np.asarray([0.0]))["passed"]
    assert not compare_float64_scores(np.asarray([np.inf]), np.asarray([-np.inf]))[
        "passed"
    ]


def _state(a: int, b: int, c: int = 0) -> np.ndarray:
    value = np.zeros(100, dtype=np.int64)
    value[:3] = (a, b, c)
    return value


def test_frozen_l06_fixture_passes_numerically_and_discretely() -> None:
    states = np.stack(
        [
            _state(0, 0, 10),
            _state(10, 0),
            _state(10, 0),
            _state(0, 10),
            _state(0, 10),
            _state(10, 0),
            _state(10, 0),
            _state(10, 0),
            _state(10, 0),
        ]
    )
    generations = np.asarray([0, 1, 1, 2, 2, 3, 3, 4, 4], dtype=np.int64)
    kinds = np.asarray(
        [
            "initial_selected_state",
            "molecular_update",
            "post_fission",
            "molecular_update",
            "post_fission",
            "molecular_update",
            "post_fission",
            "molecular_update",
            "post_fission",
        ]
    )
    indices = np.arange(len(states), dtype=np.int64)
    primary = boundary_recurrence(states, generations, kinds, indices)
    independent = boundary_recurrence_reference(states, generations, kinds, indices)
    assert all(compare_discrete_recurrence(primary, independent).values())
    assert compare_float64_scores(primary["scores"], independent["scores"])["passed"]
