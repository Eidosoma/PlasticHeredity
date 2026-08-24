from __future__ import annotations

import inspect

import numpy as np

from plastic_heredity.generative_nulls import (
    INTERVENTION_ARMS,
    MECHANISMS,
    _future_seed as gn_future_seed,
)
from plastic_heredity.phir_extension_px5 import (
    CPU_SECONDS,
    GN_CASES_PER_MATRIX,
    HORIZON,
    INFORMATION_METRICS,
    INTERVENTION_BRANCHES,
    LANDMARKS,
    MATRICES,
    REPRESENTATIONS,
    UNTREATED_BRANCHES,
    UNTREATED_HALVES,
    PX5Batch,
    _batch_digest,
    _case_archive_path,
    _load_gn_pickle,
    _rule_effect,
    _records_to_pairs,
    protocol,
    scientific_spec,
    validation_checks,
)
from plastic_heredity.simulator import FissionRecord


def test_px5_design_is_fixed_and_bounded() -> None:
    spec = scientific_spec()
    assert MATRICES == 24 and spec.matrices == 24
    assert LANDMARKS == (20, 35, 50, 65, 80)
    assert UNTREATED_BRANCHES == 16
    assert UNTREATED_HALVES == {"A": (0, 8), "B": (8, 16)}
    assert INTERVENTION_BRANCHES == 8
    assert HORIZON == 12 and CPU_SECONDS == 12 * 3600
    assert len(MECHANISMS) == 4
    assert GN_CASES_PER_MATRIX == 40
    assert REPRESENTATIONS == ("material", "functional_flux")
    assert INFORMATION_METRICS == ("full_revised", "public_revised")


def test_explicit_pairs_never_join_independent_branches() -> None:
    launch = np.asarray([2, 1, 0], dtype=np.int64)
    first = FissionRecord(
        np.asarray([4, 2, 0]), np.asarray([2, 1, 0]), 0.95, 3
    )
    second = FissionRecord(
        np.asarray([3, 3, 0]), np.asarray([1, 2, 0]), 0.92, 2
    )
    past, future = _records_to_pairs(launch, (first, second))
    assert np.array_equal(past[0], launch)
    assert np.array_equal(future[0], first.daughter)
    assert np.array_equal(past[1], first.daughter)
    assert np.array_equal(future[1], second.daughter)


def test_batch_digest_excludes_cpu_but_not_scores() -> None:
    provisional = PX5Batch(0, ({"score": 1.0},), ({"ok": 1},), 2.0, "")
    first = PX5Batch(
        0,
        provisional.score_rows,
        provisional.audit_rows,
        provisional.cpu_seconds,
        _batch_digest(provisional),
    )
    changed_time = PX5Batch(0, first.score_rows, first.audit_rows, 999.0, first.scientific_digest)
    changed_score = PX5Batch(0, ({"score": 2.0},), first.audit_rows, 2.0, "")
    assert _batch_digest(first) == _batch_digest(changed_time)
    assert _batch_digest(first) != _batch_digest(changed_score)


def test_gn_future_stream_is_mechanism_and_arm_free() -> None:
    parameters = inspect.signature(gn_future_seed).parameters
    assert "arm" not in parameters and "mechanism" not in parameters


def test_sealed_gn_checkpoint_loads_outside_main_module() -> None:
    batch = _load_gn_pickle(_case_archive_path(0))
    assert batch.__class__.__name__ == "NullBatch"
    assert batch.__class__.__module__ == "plastic_heredity.generative_nulls"


def test_protocol_is_remeasurement_not_outcome_selection() -> None:
    frozen = protocol()
    assert frozen["selection"].startswith("first 24")
    assert frozen["remeasurement"]["explicit_transition_pairs"]
    assert frozen["remeasurement"]["cross_branch_transitions"] is False
    assert frozen["remeasurement"]["intervention_arms"] == list(INTERVENTION_ARMS)
    assert frozen["classification"]["no_omnibus_gate"]
    assert frozen["no_new_matrix_or_branch_selection"]
    assert frozen["no_48_matrix_campaign"]


def test_rule_effect_uses_the_sealed_gn1_arm_namespace() -> None:
    import pandas as pd

    rows = []
    for matrix_id in range(3):
        for landmark in (20, 35):
            rows.extend(
                [
                    {
                        "matrix_id": matrix_id,
                        "landmark": landmark,
                        "candidate": "02",
                        "mechanism": "NATURAL_GARD",
                        "context": "INTERVENTION",
                        "arm": "SOURCE_RULE_DOWN",
                        "score": 3.0,
                    },
                    {
                        "matrix_id": matrix_id,
                        "landmark": landmark,
                        "candidate": "02",
                        "mechanism": "NATURAL_GARD",
                        "context": "INTERVENTION",
                        "arm": "SOURCE_RULE_UP",
                        "score": 1.0,
                    },
                ]
            )
    effect = _rule_effect(pd.DataFrame(rows), "02", "NATURAL_GARD", "score")
    assert np.array_equal(effect.to_numpy(), np.full(3, 2.0))


def test_complete_px5_validation_suite() -> None:
    checks = validation_checks()
    assert len(checks) >= 17
    assert all(checks.values()), [name for name, passed in checks.items() if not passed]
