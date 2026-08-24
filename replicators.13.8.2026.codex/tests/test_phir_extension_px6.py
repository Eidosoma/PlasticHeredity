from __future__ import annotations

import numpy as np

from plastic_heredity.phir_extension_px6 import (
    CPU_SECONDS,
    DATASETS,
    LAMBDA_GRID,
    analyze_cell,
    input_schema_checks,
    protocol,
    validation_checks,
)


def test_px6_continuum_is_fixed_and_complete() -> None:
    assert LAMBDA_GRID == (0.0, 0.25, 0.5, 0.75, 1.0)
    assert CPU_SECONDS == 2 * 3600
    assert tuple(item.phase for item in DATASETS) == ("PX1", "PX2", "PX3", "PX4", "PX5")


def test_affine_grid_matches_base_plus_lambda_redundancy() -> None:
    base = np.asarray([1.0, 2.0, 3.0, 4.0])
    correction = np.asarray([0.5, -0.5, 1.0, -1.0])
    result, _arrays = analyze_cell(
        {"phase": "TEST", "cell": "affine"}, base, correction
    )
    observed = {row["lambda"]: row["effect"] for row in result["grid"]}
    for weight in LAMBDA_GRID:
        assert observed[weight] == float(np.mean(base + weight * correction))
    endpoints = (float(base.mean()), float((base + correction).mean()))
    assert result["analytic_point_envelope"] == [min(endpoints), max(endpoints)]


def test_uniform_positive_and_definition_sensitive_fixtures() -> None:
    positive, _ = analyze_cell(
        {"phase": "TEST", "cell": "positive"},
        np.asarray([1.0, 1.1, 0.9, 1.2]),
        np.asarray([-0.1, 0.1, 0.0, -0.05]),
    )
    crossing, _ = analyze_cell(
        {"phase": "TEST", "cell": "crossing"},
        np.asarray([-0.5, -0.4, -0.6, -0.5]),
        np.asarray([1.2, 1.1, 1.3, 1.2]),
    )
    assert positive["classification"] == "uniform_positive"
    assert crossing["classification"] == "definition_sensitive"
    assert 0 < crossing["zero_crossing_lambda"] < 1


def test_protocol_forbids_favorable_definition_selection() -> None:
    frozen = protocol()
    assert frozen["prohibitions"]["no_lambda_selection"]
    assert frozen["prohibitions"]["no_outcome_driven_grid_change"]
    assert frozen["prohibitions"]["cannot_rescue_failed_public_nine_atom_result"]
    assert frozen["prohibitions"]["no_48_matrix_campaign"]
    assert "strict-eight is excluded" in frozen["claim_boundary"]


def test_real_archived_input_headers_match_every_contract() -> None:
    checks = input_schema_checks()
    assert checks
    assert all(checks.values()), [name for name, passed in checks.items() if not passed]
    assert DATASETS[1].phase == "PX2"
    assert DATASETS[1].within_columns == ("break_step",)


def test_complete_px6_validation_suite() -> None:
    checks = validation_checks()
    assert len(checks) >= 12
    assert all(checks.values()), [name for name, passed in checks.items() if not passed]
