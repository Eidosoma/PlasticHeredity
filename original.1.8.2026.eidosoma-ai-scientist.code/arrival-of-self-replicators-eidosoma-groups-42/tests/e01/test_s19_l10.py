from __future__ import annotations

import numpy as np
from sklearn.metrics import silhouette_samples

from e01_s19_matlab_attractor.core import (
    BOOTSTRAP_REPLICATES,
    R1_ID,
    R2_ID,
    bootstrap_indices,
    close_rows,
    fit_r1_matlab_historical,
    fit_r2_euclidean,
    label_against_reference,
    label_fingerprint,
    matlab_compatible_silhouette,
    scientific_recurrence_gate,
    serialize_worker_exception,
)


def _attractor(center: int, count: int, *, width: int = 20) -> np.ndarray:
    values = np.zeros((count, width), dtype=float)
    values[:, center] = 1.0
    values[:, (center + 1) % width] = 0.04
    for row in range(count):
        values[row, (center + 2 + row % 3) % width] = 0.002 * (row % 4)
    return close_rows(values)


def test_matlab_singletons_are_exactly_one() -> None:
    values = np.asarray([[1, 0], [0, 1], [1, 1], [2, 1]], dtype=float)
    scores = matlab_compatible_silhouette(values, np.arange(4), "cosine")
    assert np.array_equal(scores, np.ones(4))


def test_k_n_minus_one_and_ordinary_silhouette() -> None:
    values = np.asarray([[1.0, 0], [0.9, 0.1], [0, 1.0], [0.2, 1.0]])
    labels = np.asarray([0, 0, 1, 2])
    scores = matlab_compatible_silhouette(values, labels, "euclidean")
    assert scores[2] == scores[3] == 1.0
    pair_distance = np.linalg.norm(values[0] - values[1])
    b0 = min(
        np.linalg.norm(values[0] - values[2]), np.linalg.norm(values[0] - values[3])
    )
    assert abs(scores[0] - ((b0 - pair_distance) / max(b0, pair_distance))) <= 1e-12

    ordinary_labels = np.asarray([0, 0, 1, 1])
    clean = matlab_compatible_silhouette(values, ordinary_labels, "euclidean")
    reference = silhouette_samples(values, ordinary_labels, metric="euclidean")
    assert np.allclose(clean, reference, atol=1e-12, rtol=1e-12)


def test_recurrence_gate_rejects_singletons_and_ties() -> None:
    centroids = np.eye(4)
    singleton = scientific_recurrence_gate(np.arange(4), centroids)
    assert singleton["status"] == "NO_RECURRING_COMPTYPE"
    tied = scientific_recurrence_gate(np.asarray([0, 0, 1, 1]), centroids[:2])
    assert tied["status"] == "NO_UNIQUE_RECURRING_COMPTYPE"
    unique = scientific_recurrence_gate(np.asarray([0, 0, 0, 1, 1]), centroids[:2])
    assert unique["status"] == "ELIGIBLE"
    assert unique["dominantClusterId"] == 0
    assert unique["secondClusterId"] == 1


def test_planted_dominant_and_two_attractors() -> None:
    values = np.vstack((_attractor(1, 60), _attractor(10, 40)))
    for fit in (
        fit_r1_matlab_historical(values, "fixture-two-attractor"),
        fit_r2_euclidean(values, "fixture-two-attractor"),
    ):
        assert fit.status == "ELIGIBLE"
        assert fit.dominant_cluster_id is not None
        assert fit.second_cluster_id is not None
        assert (
            fit.cluster_sizes[fit.dominant_cluster_id]
            > fit.cluster_sizes[fit.second_cluster_id]
        )
        assert int(np.argmax(fit.dominant_centroid)) == 1


def test_no_nondrift_and_all_singleton_statuses() -> None:
    no_nondrift = fit_r1_matlab_historical(np.eye(20), "fixture-no-nondrift")
    assert no_nondrift.status == "NO_NONDRIFT_COMPOSITIONS"

    # Four isolated but locally repeated points permit the historical filter,
    # then documented singleton=1 makes k=n software-optimal.  The separate
    # recurrence gate must suppress the scientific reference.
    isolated = np.asarray(
        [[1, 0, 0, 0], [0.99, 0.01, 0, 0], [0, 0, 1, 0], [0, 0, 0.99, 0.01]],
        dtype=float,
    )
    fit = fit_r1_matlab_historical(isolated, "fixture-all-singleton")
    assert fit.selected_k == int(np.count_nonzero(fit.eligible_mask))
    assert fit.all_singleton_selected
    assert fit.status == "NO_RECURRING_COMPTYPE"
    assert fit.dominant_centroid is None


def test_feature_permutation_scaling_and_exact_replay() -> None:
    values = np.vstack((_attractor(2, 63), _attractor(12, 37)))
    permutation = np.asarray(
        [7, 3, 19, 0, 12, 1, 9, 14, 2, 8, 6, 16, 4, 18, 11, 5, 10, 13, 15, 17]
    )
    transformed = close_rows(
        values[:, permutation] * np.linspace(1.0, 4.0, len(values))[:, None]
    )
    for fitter, pipeline_id in (
        (fit_r1_matlab_historical, R1_ID),
        (fit_r2_euclidean, R2_ID),
    ):
        first = fitter(values, "fixture-invariance")
        replay = fitter(values, "fixture-invariance")
        permuted = fitter(transformed, "fixture-invariance")
        assert first.pipeline_id == pipeline_id
        assert first.status == replay.status == permuted.status == "ELIGIBLE"
        assert first.selected_k == replay.selected_k == permuted.selected_k
        assert np.array_equal(first.labels, replay.labels)
        assert np.array_equal(first.labels, permuted.labels)
        assert np.array_equal(first.centroids, replay.centroids)
        assert np.allclose(
            first.dominant_centroid[permutation], permuted.dominant_centroid
        )


def test_direct_molecular_labels_and_temporal_fingerprint() -> None:
    molecular = np.asarray([[1, 0], [0, 1], [1, 0]], dtype=float)
    scores, labels = label_against_reference(molecular, np.asarray([1, 0], dtype=float))
    assert np.array_equal(scores, [1.0, 0.0, 1.0])
    assert np.array_equal(labels, [True, False, True])
    result = label_fingerprint(labels, np.asarray([1, 1, 1]))
    assert result["persistence"] == 2
    assert result["positiveEpisodeCount"] == 2
    assert result["transitionCount"] == 2


def test_exception_provenance_and_bootstrap_contract() -> None:
    error = RuntimeError("fixture")
    row = serialize_worker_exception(
        candidate_id="CANDIDATE_2",
        matrix_id=4,
        pipeline_id=R1_ID,
        k=4,
        n=4,
        cluster_sizes=(1, 1, 1, 1),
        seed_identity="fixture-seed",
        error=error,
    )
    assert set(row) == {
        "candidateId",
        "matrixId",
        "pipelineId",
        "k",
        "n",
        "clusterSizeVector",
        "seedIdentity",
        "exceptionClass",
        "exceptionMessage",
    }
    a = bootstrap_indices("CANDIDATE_2", R1_ID)
    b = bootstrap_indices("CANDIDATE_2", R1_ID)
    c = bootstrap_indices("CANDIDATE_3", R1_ID)
    assert a.shape == (BOOTSTRAP_REPLICATES, 100)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
