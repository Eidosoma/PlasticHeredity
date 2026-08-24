from __future__ import annotations

import numpy as np
import pytest

from e01_onset_discovery.full_state_graph import (
    PRIMARY_VIEW,
    feature_names,
    graph_views,
)
from e01_onset_discovery.regime_capacity_proxy import (
    beta_structure_indices,
    binomial_cell_scores,
    center_within_groups,
)


def test_beta_structure_contract_has_twenty_coordinates() -> None:
    names = feature_names()[PRIMARY_VIEW]
    indices = beta_structure_indices(names)
    selected = tuple(names[index] for index in indices)
    assert len(selected) == 20
    assert all(name.startswith(("beta_log_", "beta_raw_", "beta_singular_")) for name in selected)


def test_beta_structure_is_state_independent() -> None:
    rng = np.random.default_rng(11)
    beta = np.exp(rng.normal(-3, 0.5, size=(100, 100)))
    left = rng.poisson(1.3, size=100).astype(np.int64)
    right = rng.poisson(1.7, size=100).astype(np.int64)
    left[0] += 1
    right[1] += 1
    target = np.ones(100) / 100
    kwargs = {
        "generation_local_step": 1,
        "observation_kind": "post_fission",
        "completed_fissions": 20,
        "batch_step": 10,
        "landmark": 20,
        "target_component_fraction": 0.0,
    }
    first = graph_views(left, beta, target, **kwargs)[PRIMARY_VIEW]
    second = graph_views(right, beta, target, **kwargs)[PRIMARY_VIEW]
    indices = beta_structure_indices(feature_names()[PRIMARY_VIEW])
    assert np.array_equal(first[list(indices)], second[list(indices)])


def test_primary_graph_is_target_invariant() -> None:
    rng = np.random.default_rng(12)
    beta = np.exp(rng.normal(-3, 0.5, size=(100, 100)))
    state = rng.poisson(1.5, size=100).astype(np.int64)
    state[0] += 1
    kwargs = {
        "generation_local_step": 2,
        "observation_kind": "post_fission",
        "completed_fissions": 35,
        "batch_step": 20,
        "landmark": 35,
        "target_component_fraction": 0.0,
    }
    uniform = np.ones(100) / 100
    random = rng.random(100)
    random /= random.sum()
    first = graph_views(state, beta, uniform, **kwargs)[PRIMARY_VIEW]
    second = graph_views(state, beta, random, **kwargs)[PRIMARY_VIEW]
    assert np.array_equal(first, second)


def test_group_centering() -> None:
    centered = center_within_groups([1.0, 3.0, 2.0, 8.0], ["a", "a", "b", "b"])
    assert centered.tolist() == pytest.approx([-1.0, 1.0, -3.0, 3.0])


def test_binomial_scores_match_manual_values() -> None:
    log_loss, brier = binomial_cell_scores([0.25, 0.75], [1, 3], [4, 4])
    assert log_loss.tolist() == pytest.approx([
        -(np.log(0.25) + 3 * np.log(0.75)) / 4,
        -(3 * np.log(0.75) + np.log(0.25)) / 4,
    ])
    q = 1.5 / 5
    assert brier.tolist() == pytest.approx([(q - 0.25) ** 2, ((3.5 / 5) - 0.75) ** 2])
