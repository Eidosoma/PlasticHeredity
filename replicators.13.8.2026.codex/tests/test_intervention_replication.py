from __future__ import annotations

import numpy as np
import pytest

from plastic_heredity.experiment import StateCase
from plastic_heredity.intervention_metrics import (
    compute_one_shot_inference,
    generate_inference_draws,
)
from plastic_heredity.intervention_replication import (
    _fixture_snapshot,
    _protocol,
    validation_checks,
)


REQUIRED_CHECKS = tuple(f"{index:02d}" for index in range(1, 25))


@pytest.fixture(scope="session")
def validation() -> dict:
    return validation_checks()


@pytest.mark.parametrize("prefix", REQUIRED_CHECKS)
def test_each_mandatory_validation_contract_passes(
    validation: dict, prefix: str
) -> None:
    matching = [
        value
        for name, value in validation["checks"].items()
        if name.startswith(prefix + "_")
    ]
    assert len(matching) == 1
    assert matching[0]["passed"] is True


def test_additional_numerical_and_orientation_audits_pass(validation: dict) -> None:
    assert validation["checks"][
        "25_batched_exhaustive_scoring_is_not_an_approximation"
    ]["passed"]
    assert validation["checks"][
        "26_beta_orientation_matches_codex_propensity_equation"
    ]["passed"]


def _metric_fixture() -> tuple[list[StateCase], np.ndarray, np.ndarray]:
    snapshot = _fixture_snapshot()
    beta = np.eye(4, dtype=np.float64)
    cases: list[StateCase] = []
    for matrix_id in range(4):
        for candidate in ("02", "03"):
            for landmark in (20, 35):
                cases.append(
                    StateCase(
                        state_id=(
                            f"metric-c{candidate}-m{matrix_id:03d}-g{landmark:03d}"
                        ),
                        cohort="METRIC",
                        candidate=candidate,
                        matrix_id=matrix_id,
                        landmark=landmark,
                        beta=beta,
                        snapshot=snapshot,
                    )
                )
    arms = 4
    branches = 4
    targets = np.zeros((len(cases), arms, branches), dtype=np.int8)
    # Up is always positive, down is always negative, and random=no-op.  This
    # makes each matrix's paired contrast exactly one after landmark averaging.
    targets[:, 0, :] = 1
    predictions = np.tile(
        np.asarray([0.8, 0.2, 0.5, 0.5], dtype=np.float64),
        (len(cases), 1),
    )
    return cases, targets, predictions


def test_matrix_inference_keeps_landmarks_in_one_matrix_block() -> None:
    cases, targets, predictions = _metric_fixture()
    draws = generate_inference_draws(
        4, 64, 64, np.random.default_rng(1), np.random.default_rng(2)
    )
    metrics, matrix_rows = compute_one_shot_inference(
        cases,
        ("UP", "DOWN", "RANDOM", "NOOP"),
        targets,
        predictions,
        draws,
        up_arm="UP",
        down_arm="DOWN",
    )
    assert len(matrix_rows) == 4 * 4
    assert all(row["up_minus_down"] == 1.0 for row in matrix_rows)
    assert all(
        cell["contrasts"]["up_minus_down"]["estimate"] == 1.0
        for cell in metrics["cells"]
    )
    assert metrics["shared_bootstrap_draws_across_cells"] is True
    assert metrics["shared_randomization_signs_across_cells"] is True


def test_branch_scores_are_ordinary_bernoulli_log_loss_and_brier() -> None:
    cases, targets, predictions = _metric_fixture()
    draws = generate_inference_draws(
        4, 64, 64, np.random.default_rng(3), np.random.default_rng(4)
    )
    metrics, _ = compute_one_shot_inference(
        cases,
        ("UP", "DOWN", "RANDOM", "NOOP"),
        targets,
        predictions,
        draws,
        up_arm="UP",
        down_arm="DOWN",
    )
    scores = metrics["cells"][0]["arms"]["UP"]["branch_scores"]
    assert scores["log_loss"] == pytest.approx(-np.log(0.8))
    assert scores["brier"] == pytest.approx((1.0 - 0.8) ** 2)


def test_protocol_freezes_serial_stops_and_budgeted_sizes() -> None:
    protocol = _protocol()
    budget = protocol["budgeted_serial_program"]
    assert budget["p1"]["matrices"] == 40
    assert budget["p1"]["branches_per_arm"] == 32
    assert budget["chosen_mechanism_confirmation"]["matrices"] == 160
    assert budget["chosen_mechanism_confirmation"]["branches_per_arm"] == 32
    assert budget["p1"]["mandatory_stop_after_seal"] is True

