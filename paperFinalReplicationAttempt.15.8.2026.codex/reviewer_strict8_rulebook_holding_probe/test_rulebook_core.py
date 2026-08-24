from __future__ import annotations

import numpy as np
import pytest
from types import SimpleNamespace

from rulebook_core import (
    EDIT_ARMS,
    RULEBOOK_FEATURE_NAMES,
    aggregate_transitions,
    apply_rulebook_edit,
    cosine,
    flow_jacobian,
    mean_field_flow,
    nearest_form,
    normalized,
    rulebook_features,
    solve_rulebook,
    tangent_stability_margin,
)


K_JOIN = 1e-2
K_LEAVE = 1e-4


def test_normalized_contract() -> None:
    assert np.allclose(normalized(np.array([1, 3])), [0.25, 0.75])
    with pytest.raises(ValueError):
        normalized(np.zeros(3))
    with pytest.raises(ValueError):
        normalized(np.array([1, -1]))


def test_mean_field_flow_is_tangent_to_simplex() -> None:
    beta = np.arange(16, dtype=float).reshape(4, 4) / 10.0 + 0.1
    flow = mean_field_flow(np.array([8, 6, 4, 2]), beta, K_JOIN, K_LEAVE)
    assert flow.shape == (4,)
    assert abs(float(flow.sum())) < 1e-15


def test_uniform_rulebook_for_exchangeable_beta() -> None:
    beta = np.full((8, 8), 2.0)
    solution = solve_rulebook(beta, K_JOIN, K_LEAVE, starts=8, seed=17)
    assert solution.forms.shape == (1, 8)
    assert np.allclose(solution.forms[0], np.full(8, 1 / 8), atol=1e-9)
    assert solution.maximum_flow_residual < 1e-10


def test_rulebook_solver_is_deterministic() -> None:
    rng = np.random.default_rng(4)
    beta = np.exp(rng.normal(-4.0, 2.0, size=(10, 10)))
    left = solve_rulebook(beta, K_JOIN, K_LEAVE, starts=6, seed=123)
    right = solve_rulebook(beta, K_JOIN, K_LEAVE, starts=6, seed=123)
    assert np.array_equal(left.forms, right.forms)
    assert left.iterations == right.iterations


def test_nearest_form_selects_by_cosine() -> None:
    forms = np.asarray([[0.8, 0.2], [0.1, 0.9]])
    index, form, score = nearest_form(np.array([1, 8]), forms)
    assert index == 1
    assert np.array_equal(form, forms[1])
    assert score > 0.99


def test_jacobian_and_stability_are_finite() -> None:
    beta = np.full((6, 6), 1.5)
    form = np.full(6, 1 / 6)
    jacobian = flow_jacobian(form, beta, K_JOIN, K_LEAVE)
    margin = tangent_stability_margin(form, beta, K_JOIN, K_LEAVE)
    assert jacobian.shape == (6, 6)
    assert np.all(np.isfinite(jacobian))
    assert np.isfinite(margin)


def test_flow_jacobian_matches_tangent_finite_difference() -> None:
    rng = np.random.default_rng(61)
    beta = np.exp(rng.normal(-2.0, 0.8, size=(6, 6)))
    form = rng.dirichlet(np.ones(6))
    direction = rng.normal(size=6)
    direction -= direction.mean()
    direction /= np.linalg.norm(direction)
    epsilon = 1e-7
    numerical = (
        mean_field_flow(form + epsilon * direction, beta, K_JOIN, K_LEAVE)
        - mean_field_flow(form - epsilon * direction, beta, K_JOIN, K_LEAVE)
    ) / (2 * epsilon)
    analytic = flow_jacobian(form, beta, K_JOIN, K_LEAVE) @ direction
    assert np.allclose(analytic, numerical, rtol=2e-6, atol=2e-9)


def test_rulebook_feature_contract() -> None:
    rng = np.random.default_rng(9)
    beta = np.exp(rng.normal(-4.0, 1.5, size=(6, 6)))
    solution = solve_rulebook(beta, K_JOIN, K_LEAVE, starts=5, seed=8)
    features = rulebook_features(
        np.array([8, 6, 4, 2, 1, 1]),
        beta,
        solution.forms,
        K_JOIN,
        K_LEAVE,
    )
    assert features.shape == (len(RULEBOOK_FEATURE_NAMES),)
    assert np.all(np.isfinite(features))


def test_directed_edits_preserve_mass_and_occupancy_and_move_as_named() -> None:
    composition = np.array([8, 6, 4, 2])
    target = np.array([0.1, 0.2, 0.3, 0.4])
    toward = apply_rulebook_edit(composition, target, "TOWARD_BOOK_D4", 10)
    away = apply_rulebook_edit(composition, target, "AWAY_BOOK_D4", 10)
    for result in (toward, away):
        assert result.composition.sum() == composition.sum()
        assert result.occupied_before == result.occupied_after == 4
        assert result.achieved_dose == 4
    assert toward.cosine_after > toward.cosine_before
    assert away.cosine_after < away.cosine_before


def test_doses_are_nested_and_random_is_deterministic() -> None:
    composition = np.array([9, 7, 3, 1])
    target = np.array([0.1, 0.2, 0.3, 0.4])
    for stem in ("TOWARD_BOOK", "AWAY_BOOK", "RANDOM_MATCHED"):
        one = apply_rulebook_edit(composition, target, f"{stem}_D1", 44)
        four = apply_rulebook_edit(composition, target, f"{stem}_D4", 44)
        assert one.transfers == four.transfers[:1]
    first = apply_rulebook_edit(composition, target, "RANDOM_MATCHED_D4", 91)
    second = apply_rulebook_edit(composition, target, "RANDOM_MATCHED_D4", 91)
    assert first.transfers == second.transfers
    assert np.array_equal(first.composition, second.composition)


def test_all_edit_arms_are_accepted() -> None:
    composition = np.array([8, 6, 4, 2])
    target = np.array([0.1, 0.2, 0.3, 0.4])
    for arm in EDIT_ARMS:
        result = apply_rulebook_edit(composition, target, arm, 1)
        assert result.composition.sum() == composition.sum()


def test_transition_aggregation() -> None:
    gates = np.asarray([[0, 1, 2, 3, 4], [4, 4, 2, 0, 0]], dtype=np.int8)
    success, trials = aggregate_transitions(gates)
    assert np.array_equal(success[0], [4, 3, 2, 1])
    assert np.array_equal(trials[0], [5, 4, 3, 2])
    assert np.array_equal(success[1], [3, 3, 2, 2])
    assert np.array_equal(trials[1], [5, 3, 3, 2])


def test_cosine_handles_zero_vector() -> None:
    assert cosine(np.zeros(3), np.ones(3)) == 0.0


def _synthetic_replay() -> dict[str, np.ndarray]:
    candidates = []
    matrix_ids = []
    landmarks = []
    for matrix_id in range(2):
        for candidate in ("02", "03"):
            for landmark in (20, 35, 50, 65, 80):
                candidates.append(candidate)
                matrix_ids.append(matrix_id)
                landmarks.append(landmark)
    shape = (len(candidates), 128, 3)
    return {
        "candidates": np.asarray(candidates),
        "matrix_ids": np.asarray(matrix_ids),
        "landmarks": np.asarray(landmarks),
        "deepest_gate": np.zeros(shape, dtype=np.int8),
        "labels": np.zeros(shape, dtype=np.int8),
    }


def test_cross_candidate_holding_uses_other_candidate_and_opposite_half() -> None:
    import run_analysis

    replay = _synthetic_replay()
    c03_m0 = (replay["candidates"] == "03") & (replay["matrix_ids"] == 0)
    replay["deepest_gate"][c03_m0, 64:, :] = 4
    replay["labels"][c03_m0, 64:, :] = 1
    features = run_analysis._holding_features_for_half(
        replay, "A", "cross_candidate"
    )
    c02_m0 = (replay["candidates"] == "02") & (replay["matrix_ids"] == 0)
    # c02 reads the all-positive c03 donor half; c03 reads all-negative c02.
    assert np.all(features[c02_m0, :12] > 0.99)
    assert np.all(features[c02_m0, 12:] > 0.99)
    assert np.all(features[c03_m0][:, [0, 4, 8]] < 0.01)
    # Conditional transitions with no eligible donor trials retain the neutral
    # Jeffreys value rather than being mislabelled as observed failures.
    assert np.all(features[c03_m0][:, [1, 2, 3, 5, 6, 7, 9, 10, 11]] == 0.5)
    assert np.all(features[c03_m0, 12:] < 0.01)
    # Target-half outcomes are excluded from the donor feature.
    replay["deepest_gate"][:, :64, :] = 4
    replay["labels"][:, :64, :] = 1
    repeated = run_analysis._holding_features_for_half(
        replay, "A", "cross_candidate"
    )
    assert np.array_equal(features, repeated)


def test_first8_outcomes_separate_break_hold_and_coherence() -> None:
    import run_analysis

    record = lambda parent, daughter: SimpleNamespace(
        parent=np.asarray(parent), daughter=np.asarray(daughter), h=cosine(parent, daughter)
    )
    vector = np.array([9, 1, 0])
    records = [record(vector, vector) for _ in range(8)]
    spec = SimpleNamespace(
        metric="cosine", inheritance_cutoff=0.9, coherence_cutoff=0.9
    )
    broken, held, coherent, complete = run_analysis._first8_outcomes(
        records, (spec,)
    )
    assert np.array_equal((broken, held, coherent, complete), np.asarray([[0], [1], [1], [1]]))
    shifted = np.array([1, 9, 0])
    records[3] = record(vector, shifted)
    broken, held, coherent, complete = run_analysis._first8_outcomes(
        records, (spec,)
    )
    assert broken[0] == 1
    assert held[0] == 0
    assert coherent[0] == 0
    assert complete[0] == 1


def test_figure_pipeline_accepts_current_pandas_and_matplotlib(tmp_path, monkeypatch) -> None:
    import pandas as pd
    import run_analysis

    monkeypatch.setattr(run_analysis, "OUTPUT_ROOT", tmp_path)
    prediction = []
    for contrast in run_analysis.PREDICTION_CONTRASTS:
        for target in run_analysis.PREDICTION_TARGETS:
            prediction.append(
                {
                    "spec": run_analysis.INTERVENTION_PRIMARY_SPEC,
                    "target": target,
                    "candidate": "02",
                    "half": "A",
                    "contrast": contrast,
                    "log_loss_gain": 0.001,
                }
            )
    pd.DataFrame(prediction).to_csv(tmp_path / "prediction_effects.csv", index=False)
    pd.DataFrame(
        [
            {
                "endpoint": endpoint,
                "candidate": "02",
                "half": "A",
                "rate_effect": 0.01,
                "ci95_lower": -0.01,
                "ci95_upper": 0.03,
                "passes_gate": False,
            }
            for endpoint in run_analysis.INTERVENTION_PRIMARY_ENDPOINTS
        ]
    ).to_csv(tmp_path / "intervention_primary_effects.csv", index=False)
    pd.DataFrame(
        [
            {
                "contrast": "toward_minus_away_d4",
                "generation": generation,
                "candidate": "02",
                "half": "A",
                "effect": 0.01 / generation,
            }
            for generation in (1, 4, 8, 32)
        ]
    ).to_csv(tmp_path / "rulebook_restoration_effects.csv", index=False)
    run_analysis._make_figures()
    assert {
        path.name for path in (tmp_path / "figures").iterdir()
    } == {
        "prediction_gains.png",
        "prediction_gains.pdf",
        "intervention_primary.png",
        "intervention_primary.pdf",
        "rulebook_restoration.png",
        "rulebook_restoration.pdf",
    }
