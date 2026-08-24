import numpy as np

from plastic_heredity.config import CANDIDATES, CohortConfig, ExperimentConfig
from plastic_heredity.experiment import BranchBatch, PROCESS_COLUMNS, StateCase
from plastic_heredity.mechanistic_metrics import (
    compute_mechanistic_metrics,
    holm_adjust,
    paired_matrix_randomization_p,
)
from plastic_heredity.mechanistic_models import MODEL_DEFINITIONS
from plastic_heredity.simulator import Snapshot


def test_holm_adjustment_is_monotone_in_sorted_order():
    raw = [0.03, 0.001, 0.02, 0.2]
    adjusted = holm_adjust(raw)
    assert adjusted == [0.06, 0.004, 0.06, 0.2]


def test_paired_matrix_randomization_detects_consistent_gain():
    matrix_ids = np.repeat(np.arange(20), 5)
    q = np.tile(np.linspace(0.1, 0.9, 5), 20)
    enhanced = np.clip(q, 0.02, 0.98)
    baseline = np.full_like(q, 0.5)
    p_value = paired_matrix_randomization_p(
        q,
        baseline,
        enhanced,
        matrix_ids,
        4096,
        np.random.default_rng(400),
    )
    assert p_value == 1 / 4097


def test_complete_metric_family_has_twelve_primary_tests():
    rng = np.random.default_rng(401)
    cases = []
    batches = []
    prediction_lists = {
        candidate: {
            **{name: [] for name in MODEL_DEFINITIONS},
            **{
                name: []
                for name in (
                    "legacy_prior",
                    "legacy_h9",
                    "legacy_beta",
                    "legacy_full",
                )
            },
        }
        for candidate in CANDIDATES
    }
    for matrix_id in range(4):
        for candidate in CANDIDATES:
            for landmark_index, landmark in enumerate((20, 35, 50, 65, 80)):
                probability = 0.12 + 0.17 * landmark_index + 0.02 * matrix_id
                target = rng.binomial(1, min(probability, 0.92), size=8).astype(np.int8)
                cases.append(
                    StateCase(
                        state_id=f"x-{candidate}-{matrix_id}-{landmark}",
                        cohort="MECHCONF",
                        candidate=candidate,
                        matrix_id=matrix_id,
                        landmark=landmark,
                        beta=np.ones((2, 2)),
                        snapshot=Snapshot(np.ones(2, dtype=np.int64), landmark, (), ()),
                    )
                )
                batches.append(
                    BranchBatch(
                        target=target,
                        process=np.full((8, len(PROCESS_COLUMNS)), np.nan),
                        completed_horizon=np.ones(8, dtype=np.int8),
                    )
                )
                for model, values in prediction_lists[candidate].items():
                    offset = 0.0 if "interaction" in model else 0.03
                    values.append(float(np.clip(probability + offset, 0.01, 0.99)))
    predictions = {
        candidate: {
            model: np.asarray(values, dtype=float) for model, values in models.items()
        }
        for candidate, models in prediction_lists.items()
    }
    experiment = ExperimentConfig(
        development=CohortConfig(4, 8),
        confirmation=CohortConfig(4, 8),
        bootstrap_repetitions=32,
        permutation_repetitions=64,
    )
    metrics = compute_mechanistic_metrics(cases, batches, predictions, experiment)
    assert len(metrics["primary_tests"]) == 12
    assert metrics["family_size"] == 12
    assert set(metrics["support"]) == {"state", "network", "interaction"}
    assert all("randomization_p_holm" in row for row in metrics["primary_tests"])
