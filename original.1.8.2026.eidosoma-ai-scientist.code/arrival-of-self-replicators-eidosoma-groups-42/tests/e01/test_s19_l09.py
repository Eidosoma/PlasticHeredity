from __future__ import annotations

import numpy as np

from e01_s19_recurring_attractor.core import (
    BOOTSTRAP_REPLICATES,
    R1_ID,
    R2_ID,
    bootstrap_indices,
    close_rows,
    fit_r1_historical,
    fit_r2_euclidean,
    historical_h,
    historical_nondrift_technique1,
    label_against_reference,
    label_fingerprint,
)


def _attractor(center: int, count: int, *, width: int = 20) -> np.ndarray:
    values = np.zeros((count, width), dtype=float)
    values[:, center] = 1.0
    values[:, (center + 1) % width] = 0.04
    for row in range(count):
        values[row, (center + 2 + row % 3) % width] = 0.002 * (row % 4)
    return close_rows(values)


def test_h_and_historical_nondrift_exact_fixture() -> None:
    values = np.asarray([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0]], dtype=float)
    h = historical_h(values, values)
    assert np.array_equal(np.diag(h), np.ones(4))
    mask, angles, local = historical_nondrift_technique1(values)
    assert np.array_equal(angles, np.asarray([1.0, 1.0, 0.0, 1.0]))
    assert np.array_equal(local, np.asarray([1.0, 0.5, 0.5, 1.0]))
    assert np.array_equal(mask, np.asarray([True, False, False, True]))


def test_planted_dominant_and_two_attractors_are_recovered() -> None:
    values = np.vstack((_attractor(1, 60), _attractor(10, 40)))
    r1 = fit_r1_historical(values, "fixture-two-attractor")
    r2 = fit_r2_euclidean(values, "fixture-two-attractor")
    assert r1.status == r2.status == "ELIGIBLE"
    assert r1.dominant_cluster_id is not None and r2.dominant_cluster_id is not None
    assert r1.second_cluster_id is not None and r2.second_cluster_id is not None
    # Historical local smoothing excludes one boundary on each side of the
    # transition, whereas R2 intentionally uses all 100 boundaries.
    assert r1.cluster_sizes[r1.dominant_cluster_id] == 59
    assert r1.cluster_sizes[r1.second_cluster_id] == 39
    assert r2.cluster_sizes[r2.dominant_cluster_id] == 60
    assert r2.cluster_sizes[r2.second_cluster_id] == 40
    for fit in (r1, r2):
        assert int(np.argmax(fit.dominant_centroid)) == 1


def test_drifting_fixture_has_no_valid_dominant_reference() -> None:
    # Every adjacent state is orthogonal; R1 has no non-drift substrate.  R2
    # groups states geometrically, but no fitted centroid has two strict-H090
    # visits, so it cannot claim a recurring attractor.
    values = np.eye(100, dtype=float)
    r1 = fit_r1_historical(values, "fixture-drift")
    r2 = fit_r2_euclidean(values, "fixture-drift")
    assert r1.status == "INELIGIBLE_NO_NONDRIFT_BOUNDARIES"
    assert r2.status == "INELIGIBLE_NO_VALID_RECURRING_CLUSTER"


def test_feature_permutation_scaling_closure_and_seed_replay() -> None:
    values = np.vstack((_attractor(2, 63), _attractor(12, 37)))
    permutation = np.asarray([7, 3, 19, 0, 12, 1, 9, 14, 2, 8, 6, 16, 4, 18, 11, 5, 10, 13, 15, 17])
    transformed = close_rows(values[:, permutation] * np.linspace(1.0, 4.0, len(values))[:, None])
    for fitter, pipeline_id in ((fit_r1_historical, R1_ID), (fit_r2_euclidean, R2_ID)):
        first = fitter(values, "fixture-invariance")
        replay = fitter(values, "fixture-invariance")
        permuted = fitter(transformed, "fixture-invariance")
        assert first.pipeline_id == pipeline_id
        assert first.status == replay.status == permuted.status == "ELIGIBLE"
        assert first.selected_k == replay.selected_k == permuted.selected_k
        assert np.array_equal(first.labels, replay.labels)
        assert np.array_equal(first.labels, permuted.labels)
        assert np.allclose(first.dominant_centroid[permutation], permuted.dominant_centroid)


def test_molecular_labels_are_direct_not_projected() -> None:
    reference = np.asarray([1.0, 0.0, 0.0])
    molecular = np.asarray([[1, 0, 0], [0, 1, 0], [1, 0, 0]], dtype=float)
    scores, labels = label_against_reference(molecular, reference)
    assert np.array_equal(labels, np.asarray([True, False, True]))
    assert np.allclose(scores, [1.0, 0.0, 1.0])
    fingerprint = label_fingerprint(labels, np.asarray([1, 1, 1]))
    assert fingerprint["persistence"] == 2
    assert fingerprint["positiveEpisodeCount"] == 2
    assert fingerprint["transitionCount"] == 2


def test_machine_scale_negative_centroid_residue_is_value_preserving() -> None:
    molecular = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    clean_scores, clean_labels = label_against_reference(molecular, [1.0, 0.0])
    residue_scores, residue_labels = label_against_reference(molecular, [1.0, -1e-16])
    assert np.array_equal(clean_labels, residue_labels)
    assert np.array_equal(clean_scores, residue_scores)


def test_bootstrap_contract_is_exact_and_domain_separated() -> None:
    a = bootstrap_indices("CANDIDATE_2", R1_ID)
    b = bootstrap_indices("CANDIDATE_2", R1_ID)
    c = bootstrap_indices("CANDIDATE_3", R1_ID)
    assert a.shape == (BOOTSTRAP_REPLICATES, 100)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert int(a.min()) == 0 and int(a.max()) == 99
