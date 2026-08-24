import numpy as np

from aor_replication.composition import clr_transform, cosine_similarity, relative_composition


def test_relative_composition_closes_rows() -> None:
    counts = np.array([[1, 2, 3], [0, 5, 0]])
    result = relative_composition(counts)
    np.testing.assert_allclose(result.sum(axis=1), 1.0)


def test_clr_has_zero_row_sum_before_coordinate_drop() -> None:
    counts = np.array([[1, 2, 3], [0, 5, 0]])
    result = clr_transform(counts, pseudocount=0.5, drop_last=False)
    np.testing.assert_allclose(result.sum(axis=1), 0.0, atol=1e-12)
    assert np.isfinite(result).all()


def test_cosine_similarity_identity_and_orthogonality() -> None:
    reference = np.array([1.0, 0.0])
    values = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    result = cosine_similarity(reference, values)
    np.testing.assert_allclose(result, [1.0, 0.0, 1 / np.sqrt(2)])

