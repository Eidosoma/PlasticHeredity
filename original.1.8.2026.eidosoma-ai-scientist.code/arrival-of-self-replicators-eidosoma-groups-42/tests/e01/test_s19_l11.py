from __future__ import annotations

import numpy as np

from e01_s19_all_comptype_union.core import (
    U1_ID,
    U2_ID,
    direct_union_scores,
    materialize_u1,
    materialize_u2,
    project_boundary_values,
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
