from __future__ import annotations

import pickle
import subprocess
import sys

import numpy as np
import pandas as pd

from plastic_heredity.phir_instruments import (
    local_phi_id_atoms,
    revised_phi_from_partition,
)
from plastic_heredity.phir_rescue_instruments import (
    _cached_local_phi_id_atoms,
    active_partition,
    beta_physical_partition,
    calibrate_numit,
    close_all_clr,
    full_block_revised,
    generate_numit_library,
    macro_phi_score,
    matched_partition_null,
    rank_gaussianize,
)
from plastic_heredity import phir_rescue as rescue
from plastic_heredity.phir_rescue import _transition_bucket, validation_checks


def _counts(seed: int = 17, observations: int = 180, types: int = 12) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.poisson(2.0, size=(observations, types))
    values[:, 0] += np.arange(observations) % 5
    return values


def test_cached_public_atoms_are_exact() -> None:
    rng = np.random.default_rng(1)
    past = rng.normal(size=(2, 200))
    future = 0.35 * past + rng.normal(size=(2, 200))
    expected = local_phi_id_atoms(past, future)
    observed = _cached_local_phi_id_atoms(past, future)
    assert expected.keys() == observed.keys()
    for atom in expected:
        np.testing.assert_allclose(observed[atom], expected[atom], atol=0.0, rtol=0.0)


def test_rank_gaussianization_is_monotone_invariant() -> None:
    values = np.random.default_rng(81).normal(size=(12, 180))
    first, active_first = rank_gaussianize(values)
    second, active_second = rank_gaussianize(np.sinh(values))
    np.testing.assert_array_equal(active_first, active_second)
    np.testing.assert_allclose(first, second, atol=1e-13, rtol=0.0)


def test_full_block_formula_and_partition_swap_are_exact() -> None:
    rng = np.random.default_rng(3)
    data = rng.normal(size=(8, 300))
    first = np.arange(3, dtype=np.int64)
    second = np.arange(3, 8, dtype=np.int64)
    left = full_block_revised(data, first, second)
    right = full_block_revised(data, second, first)
    assert left.revised == right.revised
    assert left.double_redundancy == min(
        left.aa_mi, left.ab_mi, left.ba_mi, left.bb_mi
    )
    np.testing.assert_allclose(
        left.revised,
        left.whole_mi - left.aa_mi - left.bb_mi + left.double_redundancy,
    )


def test_macro_score_matches_public_estimator() -> None:
    rng = np.random.default_rng(4)
    data = rng.normal(size=(10, 240))
    first = np.arange(5, dtype=np.int64)
    second = np.arange(5, 10, dtype=np.int64)
    expected = revised_phi_from_partition(data, first, second)
    observed = macro_phi_score(data, first, second)
    np.testing.assert_allclose(observed.revised, expected[0], atol=1e-12)
    np.testing.assert_allclose(observed.causation, expected[1], atol=1e-12)
    np.testing.assert_allclose(observed.emergence, expected[2], atol=1e-12)
    np.testing.assert_allclose(observed.synergy_persistence, expected[3], atol=1e-12)
    np.testing.assert_allclose(observed.atoms, expected[4], atol=1e-12)


def test_simultaneous_label_permutation_is_invariant() -> None:
    counts = _counts(types=14)
    rng = np.random.default_rng(9)
    beta = np.exp(rng.normal(-4.0, 2.0, size=(14, 14)))
    data, active = rank_gaussianize(close_all_clr(counts))
    pa, pb = beta_physical_partition(beta)
    aa, ab = active_partition(active, pa, pb)
    original_macro = macro_phi_score(data, aa, ab).revised
    original_full = full_block_revised(data, aa, ab).revised

    permutation = rng.permutation(14)
    permuted_counts = counts[:, permutation]
    permuted_beta = beta[np.ix_(permutation, permutation)]
    pdata, pactive = rank_gaussianize(close_all_clr(permuted_counts))
    ppa, ppb = beta_physical_partition(permuted_beta)
    paa, pab = active_partition(pactive, ppa, ppb)
    np.testing.assert_allclose(
        macro_phi_score(pdata, paa, pab).revised, original_macro, atol=1e-9
    )
    np.testing.assert_allclose(
        full_block_revised(pdata, paa, pab).revised, original_full, atol=1e-9
    )


def test_matched_partition_null_is_deterministic() -> None:
    rng = np.random.default_rng(11)
    data = rng.normal(size=(10, 160))
    observed = full_block_revised(data, np.arange(4), np.arange(4, 10)).revised
    first, first_values = matched_partition_null(
        data, 4, 16, np.random.default_rng(88), observed
    )
    second, second_values = matched_partition_null(
        data, 4, 16, np.random.default_rng(88), observed
    )
    assert first == second
    np.testing.assert_array_equal(first_values, second_values)


def test_small_numit_library_serialization_and_calibration() -> None:
    library = generate_numit_library(24, 32, np.random.default_rng(21), burn=16)
    assert library["whole_mi"].shape == (32,)
    assert library["revised"].shape == (32,)
    target = 10
    result = calibrate_numit(
        float(library["revised"][target]),
        float(library["whole_mi"][target]),
        library,
        neighbors=16,
    )
    assert result.valid
    assert 0.0 < result.percentile < 1.0
    assert np.isfinite(result.probit)


def test_transition_bucket_is_nearest_multiple_with_half_up() -> None:
    assert _transition_bucket(23) == 16
    assert _transition_bucket(24) == 32
    assert _transition_bucket(479) == 480


def test_amended_score_fields_do_not_overwrite_legacy_archive(
    monkeypatch,
) -> None:
    counts = _counts(observations=48, types=12)
    beta = np.exp(np.random.default_rng(33).normal(-4.0, 1.0, size=(12, 12)))
    library = generate_numit_library(48, 32, np.random.default_rng(34), burn=16)
    monkeypatch.setattr(rescue, "_library", lambda _bucket: library)
    scores = rescue._score_representation(
        counts,
        beta,
        rescue.smoke_spec(),
        matrix_id=0,
        candidate="02",
        replicate=0,
        representation="fable_style",
    )
    new_columns = {f"fable_style_{name}" for name in scores}
    archived_columns = set(
        pd.read_csv(rescue.PAB24_OUTPUT / "pab24_lineages.csv.gz", nrows=0).columns
    )
    assert new_columns.isdisjoint(archived_columns)


def test_module_checkpoint_loads_in_a_fresh_interpreter(tmp_path) -> None:
    checkpoint = tmp_path / "fixture.pkl"
    written = subprocess.run(
        [
            sys.executable,
            "-m",
            "plastic_heredity.phir_rescue",
            "_checkpoint-fixture",
            "--output",
            str(checkpoint),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    with checkpoint.open("rb") as handle:
        batch = pickle.load(handle)
    assert isinstance(batch, rescue.RescueBatch)
    assert batch.scientific_digest == written
    assert batch.scientific_digest == rescue._batch_digest(batch)
    assert type(batch).__module__ == "plastic_heredity.phir_rescue"


def test_complete_rescue_validation_suite() -> None:
    checks = validation_checks()
    assert len(checks) >= 22
    assert all(checks.values())
