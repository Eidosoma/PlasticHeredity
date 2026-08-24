from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from e01_s19_all_comptype_union_repair.core import (
    MATERIAL_NEGATIVE_TOLERANCE,
    THRESHOLD,
    U1_ID,
    U2_ID,
    close_rows,
    direct_union_scores,
    historical_h,
    materialize_u1,
    materialize_u2,
    project_boundary_values,
    repair_u2_centroids,
)
from scripts.e01.run_s19_l11r import (
    numeric_comparison_series,
    serialize_gate_observed_value,
    sha256_file,
    verified_quarantine_copy,
)


def _two_cluster_boundaries() -> np.ndarray:
    rows = []
    for index in range(100):
        row = np.full(6, 1e-6, dtype=np.float64)
        row[0 if index < 60 else 3] = 0.85
        row[1 if index < 60 else 4] = 0.15
        rows.append(row / row.sum())
    return np.asarray(rows)


def test_projection_begins_at_boundary_and_preserves_negative_prefix() -> None:
    labels = np.asarray([True, False, True])
    positions = np.asarray([2, 5, 8])
    observed = project_boundary_values(labels, positions, 10, prefix_value=False)
    expected = np.asarray(
        [False, False, True, True, True, False, False, False, True, True]
    )
    np.testing.assert_array_equal(observed, expected)


def test_direct_union_uses_every_centroid_and_strict_threshold() -> None:
    centroids = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    values = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    scores, labels = direct_union_scores(values, centroids)
    np.testing.assert_allclose(scores, [1.0, 1.0, 2**-0.5], atol=1e-15)
    np.testing.assert_array_equal(labels, [True, True, False])


def test_diagnosed_machine_scale_residue_is_clamped_and_reclosed() -> None:
    raw = np.asarray([[0.6, 0.4, -2.3852447794681098e-18]])
    repaired, audit = repair_u2_centroids(raw)
    np.testing.assert_array_equal(repaired, [[0.6, 0.4, 0.0]])
    assert audit.negative_coordinate_count == 1
    assert audit.material_negative_coordinate_count == 0
    assert audit.maximum_unit_sum_error_after_reclosure == 0.0


def test_exact_tolerance_boundary_is_permitted_and_clamped() -> None:
    raw = np.asarray([[0.75, 0.25, -MATERIAL_NEGATIVE_TOLERANCE]])
    repaired, audit = repair_u2_centroids(raw)
    np.testing.assert_array_equal(repaired, [[0.75, 0.25, 0.0]])
    assert audit.minimum_raw_coordinate == -MATERIAL_NEGATIVE_TOLERANCE


def test_coordinate_beyond_tolerance_fails_materially() -> None:
    beyond = np.nextafter(-MATERIAL_NEGATIVE_TOLERANCE, -np.inf)
    with pytest.raises(ValueError, match="material negative"):
        repair_u2_centroids(np.asarray([[1.0, beyond]]))


def test_zero_sum_after_clamp_is_rejected() -> None:
    with pytest.raises(ValueError, match="zero-sum"):
        repair_u2_centroids(np.asarray([[0.0, -1e-18, 0.0]]))


def test_centroid_repair_is_feature_permutation_equivariant() -> None:
    raw = np.asarray([[0.7, -1e-18, 0.3], [0.1, 0.2, 0.7]])
    permutation = np.asarray([2, 0, 1])
    repaired, _ = repair_u2_centroids(raw)
    permuted, _ = repair_u2_centroids(raw[:, permutation])
    np.testing.assert_array_equal(permuted, repaired[:, permutation])


def test_centroid_repair_is_positive_scaling_invariant_after_closure() -> None:
    raw = np.asarray([[0.7, -1e-18, 0.3], [0.1, 0.2, 0.7]])
    repaired, _ = repair_u2_centroids(raw)
    scaled, _ = repair_u2_centroids(raw * np.asarray([[2.0], [0.5]]))
    np.testing.assert_allclose(scaled, repaired, rtol=0.0, atol=1e-15)


def test_independent_validator_replays_scoring_boundary_reclosure_exactly() -> None:
    raw = np.asarray(
        [
            [0.6, 0.4, -3.469446951953614e-18],
            [0.2, 0.3, 0.5],
        ]
    )
    values = np.asarray([[0.6, 0.4, 0.0], [0.2, 0.3, 0.5], [1.0, 1.0, 1.0]])
    first_reclosure, _ = repair_u2_centroids(raw)
    analyze_input_values = close_rows(values)
    canonical_scores, canonical_labels = direct_union_scores(
        close_rows(analyze_input_values), first_reclosure
    )
    independent_scoring_reclosure, _ = repair_u2_centroids(first_reclosure)
    independent_scoring_values = close_rows(close_rows(analyze_input_values))
    independent_scores = np.max(
        historical_h(independent_scoring_values, independent_scoring_reclosure),
        axis=1,
    )
    np.testing.assert_array_equal(canonical_scores, independent_scores)
    np.testing.assert_array_equal(canonical_labels, independent_scores > THRESHOLD)


def test_boolean_reporting_series_is_numeric_without_value_change() -> None:
    observed = numeric_comparison_series(
        pd.Series([True, False, pd.NA], dtype="boolean", index=[4, 7, 9])
    )
    assert observed.dtype == np.dtype("float64")
    assert observed.index.tolist() == [4, 7, 9]
    np.testing.assert_array_equal(observed.iloc[:2].to_numpy(), [1.0, 0.0])
    assert np.isnan(observed.iloc[2])


def test_gate_observed_value_serialization_is_typed_and_lossless() -> None:
    assert serialize_gate_observed_value(True) == ("true", "BOOLEAN", 1.0)
    assert serialize_gate_observed_value(3) == ("3", "INTEGER", 3.0)
    assert serialize_gate_observed_value(0.25) == ("0.25", "FLOAT", 0.25)
    control_map = '{"NC1": true, "NC2": false}'
    assert serialize_gate_observed_value(control_map) == (
        '"{\\"NC1\\": true, \\"NC2\\": false}"',
        "STRING",
        None,
    )


def test_verified_quarantine_copy_preserves_source_and_exact_bytes(tmp_path) -> None:
    source = tmp_path / "source.bin"
    destination = tmp_path / "other" / "destination.bin"
    source.write_bytes(b"L11R-invalidated-partial-output")
    digest = sha256_file(source)
    size = source.stat().st_size
    verified_quarantine_copy(source, destination, digest, size)
    assert source.exists()
    assert destination.read_bytes() == source.read_bytes()
    assert sha256_file(destination) == digest


def test_u1_retains_all_positive_source_tags_and_projects_them() -> None:
    boundary = _two_cluster_boundaries()
    molecular = np.vstack((np.ones((5, 6)), boundary))
    positions = np.arange(5, 105, dtype=np.int64)
    result = materialize_u1(boundary, molecular, positions, "TEST-U1-TWO-CLUSTER")
    assert result.pipeline_id == U1_ID
    assert result.boundary_tags is not None
    assert result.molecular_tags is not None
    assert result.boundary_labels is not None
    assert result.molecular_labels is not None
    assert np.all(result.boundary_tags[result.fit.eligible_mask] > 0)
    assert np.all(result.boundary_tags[~result.fit.eligible_mask] == 0)
    assert not np.any(result.molecular_labels[:5])
    assert len(set(result.boundary_tags[result.boundary_tags > 0])) >= 2


def test_u2_uses_all_clusters_with_at_least_two_members() -> None:
    boundary = _two_cluster_boundaries()
    molecular = np.vstack((boundary[:3], boundary[-3:]))
    result = materialize_u2(boundary, molecular, "TEST-U2-TWO-CLUSTER")
    assert result.pipeline_id == U2_ID
    assert result.recurring_centroids is not None
    assert len(result.recurring_centroids) >= 2
    assert result.molecular_labels is not None
    assert np.all(result.molecular_labels)


def test_u1_exact_replay() -> None:
    boundary = _two_cluster_boundaries()
    positions = np.arange(100, dtype=np.int64)
    first = materialize_u1(boundary, boundary, positions, "TEST-U1-REPLAY")
    second = materialize_u1(boundary, boundary, positions, "TEST-U1-REPLAY")
    np.testing.assert_array_equal(first.boundary_tags, second.boundary_tags)
    np.testing.assert_array_equal(first.molecular_labels, second.molecular_labels)
    assert first.fit.cluster_sizes == second.fit.cluster_sizes


def test_u2_exact_replay() -> None:
    boundary = _two_cluster_boundaries()
    positions = np.arange(100, dtype=np.int64)
    first = materialize_u2(boundary, boundary, "TEST-U2-REPLAY")
    second = materialize_u2(boundary, boundary, "TEST-U2-REPLAY")
    np.testing.assert_array_equal(first.boundary_tags, second.boundary_tags)
    np.testing.assert_array_equal(first.molecular_labels, second.molecular_labels)
    assert first.fit.cluster_sizes == second.fit.cluster_sizes
    assert positions.size == 100
