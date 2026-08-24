import numpy as np

from plastic_heredity.config import ExperimentConfig
from plastic_heredity.experiment import StateCase
from plastic_heredity.intervention_p4 import (
    ARMS,
    SEEDS,
    _two_sided_sign_p,
    acquire_natural_break,
    compute_inference,
    protocol,
)
from plastic_heredity.simulator import FissionRecord, Snapshot


def test_p4_protocol_separates_strength_and_topology() -> None:
    frozen = protocol()
    assert frozen["inference"]["strength_and_topology_classified_separately"]
    assert frozen["p3c_unchanged_and_not_rescued"]
    assert frozen["not_original_cr5"]
    assert tuple(frozen["arms"]) == ARMS


def test_p4_seed_domains_are_unique() -> None:
    assert len(SEEDS) == len(set(SEEDS.values()))


def test_two_sided_sign_test_is_symmetric() -> None:
    values = np.asarray([0.2, 0.1, 0.3])
    signs = np.asarray([[1, 1, 1], [-1, -1, -1], [1, -1, 1], [-1, 1, -1]])
    left, left_null = _two_sided_sign_p(values, signs)
    right, right_null = _two_sided_sign_p(-values, signs)
    assert left == right
    assert np.array_equal(left_null, -right_null)


def test_acquisition_saves_exact_natural_break_daughter(monkeypatch) -> None:
    snapshot = Snapshot(
        composition=np.asarray([2, 1], dtype=np.int64),
        generation=7,
        inheritance=(True,),
        boundary_h=(0.95,),
        previous_growth_steps=4,
        cumulative_growth_steps=9,
    )
    case = StateCase("fixture", "FIX", "02", 0, 20, np.ones((2, 2)), snapshot)
    daughter = np.asarray([1, 1], dtype=np.int64)

    def fixed_advance(*_args, **_kwargs):
        return FissionRecord(
            parent=np.asarray([3, 1], dtype=np.int64),
            daughter=daughter,
            h=0.9,
            growth_steps=6,
        )

    monkeypatch.setattr("plastic_heredity.intervention_p4.advance_fission", fixed_advance)
    broken, anchor, audit = acquire_natural_break(case, ExperimentConfig.quick())
    assert audit["eligible"] is True
    assert np.array_equal(broken.snapshot.composition, daughter)
    assert np.array_equal(anchor, np.asarray([3, 1]))
    assert broken.snapshot.inheritance[-1] is False
    assert broken.snapshot.cumulative_growth_steps == 15


def test_p4_inference_keeps_strength_and_topology_separate() -> None:
    cases = []
    snapshot = Snapshot(
        composition=np.asarray([2, 1], dtype=np.int64),
        generation=8,
        inheritance=(False,),
        boundary_h=(0.8,),
        previous_growth_steps=4,
        cumulative_growth_steps=10,
    )
    for candidate in ("02", "03"):
        for matrix_id in range(20):
            cases.append(
                StateCase(
                    f"c{candidate}-m{matrix_id}",
                    "FIX",
                    candidate,
                    matrix_id,
                    20,
                    np.ones((2, 2)),
                    snapshot,
                )
            )
    targets = np.zeros((len(cases), len(ARMS), 32), dtype=np.int8)
    targets[:, ARMS.index("TIGHTEN"), :] = 1
    geometry = np.tile(
        np.asarray([-np.log(1.5), np.log(1.5), 0.1, 0.0, 0.0]),
        (len(cases), 1),
    )
    metrics, rows, stored = compute_inference(
        cases,
        {"targets": targets},
        {"log_throughput_ratio": geometry},
    )
    assert metrics["strength_all_statistical_cells_pass"]
    assert metrics["topology_classification"] == "negligible_within_0.025"
    assert len(rows) == 4 * 20
    assert stored
