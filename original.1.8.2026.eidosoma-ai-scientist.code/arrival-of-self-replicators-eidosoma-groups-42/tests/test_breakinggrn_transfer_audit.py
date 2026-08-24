from __future__ import annotations

import numpy as np

from e01_breakinggrn_transfer_audit.core import (
    array_sha256,
    derive_seed,
    run_breaking_transfer,
)


SAFE_LATTICE = "/artifacts/research_steps/S12B/safe_phi_lattice.json"


def fixture() -> np.ndarray:
    rng = np.random.RandomState(derive_seed("unit", "coupled"))
    x = rng.normal(size=(384, 10))
    x[:, 5:] += 0.3 * x[:, :5]
    return x


def test_deterministic_exact_replay() -> None:
    x = fixture()
    kwargs = {
        "preprocessing_seed": derive_seed("unit", "preprocess"),
        "partition_seed": derive_seed("unit", "partition"),
    }
    first = run_breaking_transfer(x, SAFE_LATTICE, **kwargs)
    second = run_breaking_transfer(x, SAFE_LATTICE, **kwargs)
    assert first.status == second.status
    assert first.partition_1 == second.partition_1
    assert first.partition_2 == second.partition_2
    if first.emergence_nan0 is not None:
        assert array_sha256(first.emergence_nan0) == array_sha256(
            second.emergence_nan0
        )
        assert array_sha256(first.integrated_raw) == array_sha256(
            second.integrated_raw
        )


def test_output_alignment_is_two() -> None:
    result = run_breaking_transfer(
        fixture(),
        SAFE_LATTICE,
        preprocessing_seed=derive_seed("unit", "preprocess"),
        partition_seed=derive_seed("unit", "partition"),
    )
    assert result.local_offset == 2
    if result.emergence_nan0 is not None:
        assert len(result.emergence_nan0) == 382


def test_invalid_input_is_status_bearing() -> None:
    result = run_breaking_transfer(
        np.ones((3, 1)),
        SAFE_LATTICE,
        preprocessing_seed=1,
        partition_seed=2,
    )
    assert result.status == "INELIGIBLE_INPUT_SHAPE"
