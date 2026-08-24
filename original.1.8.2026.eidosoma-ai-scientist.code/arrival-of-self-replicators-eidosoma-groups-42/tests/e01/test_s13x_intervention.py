from pathlib import Path

import numpy as np

from e01_creative_directional_search.intervention import (
    _fit_gaussian,
    build_frozen_phirl_scorer,
    source_replay_max_abs,
)
from e01_source_emergence_metric_identity.core import run_emergence_pipeline

SAFE_LATTICE = Path("/artifacts/research_steps/S12B/safe_phi_lattice.json")


def test_gaussian_model_matches_closed_form_univariate_entropy() -> None:
    reference = np.asarray([[0.0, 1.0, 2.0, 3.0]], dtype=float)
    model = _fit_gaussian(reference)
    query = np.asarray([1.25])
    mean = reference.mean()
    variance = reference.var()
    expected = 0.5 * (
        np.log(2.0 * np.pi * variance) + (query[0] - mean) ** 2 / variance
    )
    assert abs(model.entropy(query) - expected) <= 1e-14


def test_frozen_scorer_replays_source_and_enumerates_actions() -> None:
    assert SAFE_LATTICE.is_file()
    rng = np.random.default_rng(20260806)
    counts = rng.poisson(1.4, size=(160, 100)).astype(np.int64)
    counts[:, 0] += 1
    masses = counts.sum(axis=1)
    closed = (counts + 0.5) / (masses[:, None] + 50.0)
    logs = np.log(closed)
    clr = (logs - logs.mean(axis=1, keepdims=True))[:, :99]
    result = run_emergence_pipeline(
        clr,
        "PHIRL_REGULARIZED_SOURCE",
        SAFE_LATTICE,
        preprocessing_seed=17,
        partition_seed=23,
    )
    assert result.status == "ELIGIBLE"
    scorer = build_frozen_phirl_scorer(clr, result, SAFE_LATTICE)
    assert scorer.preprocessing_max_abs_error <= 1e-12
    assert source_replay_max_abs(scorer, result) <= 1e-10
    actions = scorer.score_count_actions(counts[-1])
    assert len(actions) == 100 + np.count_nonzero(counts[-1])
    assert [row["actionOrder"] for row in actions] == list(range(len(actions)))
    assert all(np.isfinite(row["emergence"]) for row in actions)
