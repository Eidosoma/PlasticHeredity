from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

from aor_replication.bridge_information import fit_bridge_estimators
from aor_replication.covariance_support import (
    DEVELOPMENT_SEEDS,
    POOL_PAIRS,
    REPEATS,
    SUPPORT_LADDER,
    _stability_summary,
    _nested_indices,
    frozen_support_config,
)
from aor_replication.information import fit_causal_trajectory
from aor_replication.support_information import (
    ALL_SUPPORT_INSTRUMENTS,
    PCA_COMPONENTS_PER_MODULE,
    _deterministic_pca,
    gaussian_mi_reading,
    pca_full_revised_from_pairs,
    prepare_support_window,
    score_prepared_pairs,
)


def _counts(
    seed: int = 811, observations: int = 150, types: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    counts = rng.poisson(1.5, size=(observations, types)).astype(np.int64)
    counts[:, 0] += 7 + np.arange(observations) % 5
    counts[:, 1] += (np.arange(observations) // 4) % 3
    beta = np.exp(rng.normal(-4.0, 1.7, size=(types, types)))
    return counts, beta


def _pca_score(past: np.ndarray, future: np.ndarray):
    return pca_full_revised_from_pairs(
        past,
        future,
        np.arange(8, dtype=np.int64),
        np.arange(8, 16, dtype=np.int64),
    )


def test_pca8_independent_block_null_is_near_zero() -> None:
    rng = np.random.default_rng(1)
    pairs = 4096
    past = rng.normal(size=(16, pairs))
    future = np.empty_like(past)
    future[:8] = 0.65 * past[:8] + rng.normal(size=(8, pairs))
    future[8:] = 0.50 * past[8:] + rng.normal(size=(8, pairs))
    reading = _pca_score(past, future)
    assert abs(reading.score) < 0.08


def test_pca8_cross_coupled_var_is_positive_and_above_null() -> None:
    rng = np.random.default_rng(2)
    pairs = 4096
    null_past = rng.normal(size=(16, pairs))
    null_future = np.vstack(
        (
            0.60 * null_past[:8] + rng.normal(size=(8, pairs)),
            0.60 * null_past[8:] + rng.normal(size=(8, pairs)),
        )
    )
    coupled_past = rng.normal(size=(16, pairs))
    coupled_future = np.vstack(
        (
            0.70 * coupled_past[8:] + rng.normal(size=(8, pairs)),
            0.70 * coupled_past[:8] + rng.normal(size=(8, pairs)),
        )
    )
    null = _pca_score(null_past, null_future)
    coupled = _pca_score(coupled_past, coupled_future)
    assert coupled.score > 1.0
    assert coupled.score > null.score + 1.0


def test_pca8_redundant_copy_is_not_spuriously_integrated() -> None:
    rng = np.random.default_rng(3)
    pairs = 4096
    latent = rng.normal(size=(8, pairs))
    past = np.vstack(
        (
            latent + 0.03 * rng.normal(size=(8, pairs)),
            latent + 0.03 * rng.normal(size=(8, pairs)),
        )
    )
    future_latent = 0.70 * latent + rng.normal(size=(8, pairs))
    future = np.vstack(
        (
            future_latent + 0.03 * rng.normal(size=(8, pairs)),
            future_latent + 0.03 * rng.normal(size=(8, pairs)),
        )
    )
    reading = _pca_score(past, future)
    assert abs(reading.score) < 0.10


def test_pca8_gaussian_suppressor_synergy_is_positive() -> None:
    rng = np.random.default_rng(4)
    pairs = 4096
    common = rng.normal(size=(8, pairs))
    suppressed = rng.normal(size=(8, pairs))
    part_a = common + 0.08 * suppressed
    part_b = common - 0.08 * suppressed
    past = np.vstack((part_a, part_b))
    signal = 5.0 * (part_a - part_b)
    future = np.vstack(
        (
            signal + rng.normal(scale=0.7, size=(8, pairs)),
            -signal + rng.normal(scale=0.7, size=(8, pairs)),
        )
    )
    reading = _pca_score(past, future)
    assert reading.score > 1.0
    assert reading.components["whole_mi"] > reading.components["aa_mi"]
    assert reading.components["whole_mi"] > reading.components["bb_mi"]


def test_pca8_is_stable_to_exact_pair_duplication() -> None:
    rng = np.random.default_rng(5)
    past = rng.normal(size=(16, 128))
    future = np.vstack(
        (
            0.8 * past[8:] + 0.8 * rng.normal(size=(8, 128)),
            0.8 * past[:8] + 0.8 * rng.normal(size=(8, 128)),
        )
    )
    original = _pca_score(past, future)
    duplicated = _pca_score(
        np.repeat(past, 2, axis=1), np.repeat(future, 2, axis=1)
    )
    relative_change = abs(duplicated.score - original.score) / abs(original.score)
    assert relative_change < 0.02


def test_pca8_coupled_process_is_stable_as_support_increases() -> None:
    rng = np.random.default_rng(22)
    past = rng.normal(size=(16, 512))
    future = np.vstack(
        (
            0.8 * past[8:] + 0.8 * rng.normal(size=(8, 512)),
            0.8 * past[:8] + 0.8 * rng.normal(size=(8, 512)),
        )
    )
    scores = np.asarray(
        [_pca_score(past[:, :pairs], future[:, :pairs]).score for pairs in (128, 256, 512)]
    )
    assert np.all(scores > 0.0)
    assert np.max(np.abs(scores - scores[-1]) / abs(scores[-1])) < 0.25


def test_all_coordinate_support_instruments_are_molecule_label_invariant() -> None:
    counts, beta = _counts(observations=130)
    original = prepare_support_window(counts, beta)
    indices = np.arange(128, dtype=np.int64)
    observed = score_prepared_pairs(original, indices)
    permutation = np.random.default_rng(33).permutation(counts.shape[1])
    permuted = prepare_support_window(
        counts[:, permutation], beta[np.ix_(permutation, permutation)]
    )
    repeated = score_prepared_pairs(permuted, indices)
    assert tuple(observed) == ALL_SUPPORT_INSTRUMENTS
    # The legacy typeset/macro path deliberately drops the final CLR
    # coordinate, so a molecule permutation changes its representation.  The
    # new all-coordinate instruments must remain label invariant.
    for instrument in (
        "public_nine_atom",
        "pca8_full_revised",
        "raw100_full_revised",
    ):
        np.testing.assert_allclose(
            observed[instrument].score,
            repeated[instrument].score,
            atol=2e-7,
            rtol=0,
        )


def test_existing_macro_wms_replays_exactly_on_contiguous_window() -> None:
    counts, beta = _counts(observations=130)
    prepared = prepare_support_window(counts, beta)
    observed = score_prepared_pairs(prepared, np.arange(129, dtype=np.int64))
    expected = fit_causal_trajectory(counts)
    np.testing.assert_allclose(
        observed["macro_wms"].score,
        expected.values.mean(),
        atol=0.0,
        rtol=0.0,
    )


def test_covariance_rank_and_support_are_reported_before_ridge() -> None:
    rng = np.random.default_rng(41)
    left = rng.normal(size=(100, 64))
    right = rng.normal(size=(100, 64))
    reading = gaussian_mi_reading(left, right)
    assert reading.joint.dimension == 200
    assert reading.joint.samples == 64
    assert reading.joint.rank == 63
    assert reading.joint.ridge > 0.0
    assert np.isfinite(reading.value)


def test_pca_projection_is_fitted_from_past_only() -> None:
    rng = np.random.default_rng(55)
    past = rng.normal(size=(20, 128))
    first = _deterministic_pca(past, PCA_COMPONENTS_PER_MODULE)
    future_a = rng.normal(size=(20, 128))
    future_b = rng.normal(loc=50.0, scale=20.0, size=(20, 128))
    np.testing.assert_array_equal(first.transform(past), first.transform(past))
    assert first.digest == _deterministic_pca(past, PCA_COMPONENTS_PER_MODULE).digest
    assert not np.allclose(first.transform(future_a), first.transform(future_b))


def test_primary_explicit_pairs_do_not_invent_cross_pair_transitions() -> None:
    counts, beta = _counts(observations=150)
    prepared = prepare_support_window(counts, beta)
    selected = np.asarray([0, 3, 7, 14, 31, 63, 95, 127], dtype=np.int64)
    readings = score_prepared_pairs(prepared, selected)
    assert all(reading.pairs == selected.size for reading in readings.values())
    for reading in readings.values():
        assert reading.whole_joint_rank <= selected.size - 1


def test_support_module_has_no_replicator_detector_import() -> None:
    path = Path("src/aor_replication/support_information.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert not any("replicator" in name for name in imported)


def test_prior_full_bridge_remains_available_without_source_changes() -> None:
    counts, beta = _counts(observations=130)
    observed = fit_bridge_estimators(counts, beta)
    assert set(observed) == {
        "macro_wms",
        "macro_mmi",
        "public_nine_atom",
        "full_revised",
    }


def test_support_ladder_is_frozen_nested_and_endpoint_matched() -> None:
    config = frozen_support_config()
    config.validate(require_frozen=True)
    assert config.seeds == DEVELOPMENT_SEEDS
    assert config.supports == SUPPORT_LADDER
    assert config.pool_pairs == POOL_PAIRS == 512
    assert config.repeats == REPEATS == 12
    permutation = np.random.default_rng(91).permutation(POOL_PAIRS - 1)
    previous: set[int] = set()
    for support in SUPPORT_LADDER:
        indices = _nested_indices(POOL_PAIRS, support, permutation)
        assert indices.size == support
        assert indices[-1] == POOL_PAIRS - 1
        assert previous.issubset(set(indices))
        previous = set(indices)


def test_support_runner_has_no_replicator_detector_import() -> None:
    path = Path("src/aor_replication/covariance_support.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert not any("replicator" in name for name in imported)


def test_stability_summary_uses_explicit_mode_column() -> None:
    rows = []
    for instrument_index, instrument in enumerate(ALL_SUPPORT_INSTRUMENTS):
        for run_index in range(3):
            for support in SUPPORT_LADDER:
                for repeat in range(2):
                    rows.append(
                        {
                            "mode": "paired_subsample",
                            "instrument": instrument,
                            "run_index": run_index,
                            "repeat": repeat,
                            "support": support,
                            "ordinary_score": (
                                instrument_index + run_index + support / 10000
                            ),
                        }
                    )
                rows.append(
                    {
                        "mode": "end_anchored",
                        "instrument": instrument,
                        "run_index": run_index,
                        "repeat": -1,
                        "support": support,
                        "ordinary_score": (
                            instrument_index + run_index + support / 10000
                        ),
                    }
                )
    result = _stability_summary(pd.DataFrame(rows))
    assert result.shape[0] == len(ALL_SUPPORT_INSTRUMENTS) * len(SUPPORT_LADDER)
    assert (result["ordering_agreement"] == 1.0).all()
