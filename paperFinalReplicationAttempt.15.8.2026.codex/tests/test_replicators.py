import numpy as np

from aor_replication.config import ReplicatorConfig
from aor_replication.gard import PHASE_EXCHANGE, PHASE_FISSION, RunTrace
from aor_replication.replicators import detect_replicators, replicator_metrics


def synthetic_trace() -> RunTrace:
    counts = np.array(
        [
            [5, 5, 0],
            [8, 2, 0],
            [4, 4, 0],
            [8, 2, 0],
            [4, 4, 0],
            [8, 2, 0],
            [4, 4, 0],
            [8, 2, 0],
        ],
        dtype=np.int64,
    )
    steps = len(counts)
    zero = np.zeros_like(counts)
    return RunTrace(
        counts=counts,
        generations=np.array([0, 0, 0, 1, 1, 2, 2, 3]),
        phases=np.array(
            [PHASE_EXCHANGE, PHASE_EXCHANGE, PHASE_FISSION, PHASE_EXCHANGE, PHASE_FISSION, PHASE_EXCHANGE, PHASE_FISSION, PHASE_EXCHANGE]
        ),
        joins=zero,
        leaves=zero,
        intervention_species=np.full(steps, -1),
        intervention_delta=np.zeros(steps, dtype=np.int64),
        intervention_score=np.full(steps, np.nan),
        beta=np.eye(3),
        seed=1,
    )


def test_dominant_recurring_composition_is_detected() -> None:
    trace = synthetic_trace()
    result = detect_replicators(
        trace,
        ReplicatorConfig(similarity_threshold=0.999, min_recurrences=3),
    )
    assert result.support == 4
    assert result.labels[[1, 3, 5, 7]].all()
    assert not result.labels[[0, 2, 4, 6]].any()


def test_replicator_metrics_have_reported_units() -> None:
    labels = np.array([False, False, True, True, False, True])
    metrics = replicator_metrics(labels)
    assert metrics.persistence == 3
    assert metrics.probability == 0.5
    assert metrics.time_to_first == 2 / 5
    assert np.isfinite(metrics.consistency)


def test_euclidean_centroid_detector_is_supported() -> None:
    result = detect_replicators(
        synthetic_trace(),
        ReplicatorConfig(
            similarity_threshold=0.999,
            min_recurrences=3,
            similarity_metric="euclidean",
            reference_method="neighbor_centroid",
        ),
    )
    assert result.support == 4
    assert result.labels[[1, 3, 5, 7]].all()
