from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from plastic_heredity.episode_coherence import (
    _cluster_mean_interval,
    _validate_replayed_row,
    episode_geometry,
    run_audit,
)
from plastic_heredity.processes import evaluate_process
from plastic_heredity.simulator import FissionRecord


def _record(
    h: float, parent: tuple[int, ...], daughter: tuple[int, ...]
) -> FissionRecord:
    return FissionRecord(
        parent=np.asarray(parent, dtype=np.int64),
        daughter=np.asarray(daughter, dtype=np.int64),
        h=h,
        growth_steps=1,
    )


def _records(inherited: str) -> list[FissionRecord]:
    basis = (
        (9, 1, 0, 0),
        (7, 3, 0, 0),
        (3, 7, 0, 0),
        (0, 9, 1, 0),
        (0, 6, 4, 0),
        (0, 2, 8, 0),
        (0, 0, 8, 2),
        (0, 0, 4, 6),
        (0, 0, 1, 9),
        (4, 0, 0, 6),
        (8, 0, 0, 2),
        (9, 1, 0, 0),
    )
    rows = []
    for index, value in enumerate(inherited):
        rows.append(
            _record(
                0.95 if value == "T" else 0.80,
                basis[max(0, index - 1)],
                basis[index],
            )
        )
    return rows


def test_drifting_inherited_chain_passes_target_but_fails_episode_coherence():
    records = _records("FTTT")
    assert evaluate_process(records).joint_break_run3
    geometry = episode_geometry(records)
    assert geometry.episode_start_index == 1
    assert geometry.minimum_pairwise_daughter_similarity < 0.9
    assert geometry.first_last_daughter_similarity < 0.9


def test_first_qualifying_run_is_selected_after_interrupted_prefix():
    geometry = episode_geometry(_records("FTTFTTT"))
    assert geometry.first_break_index == 0
    assert geometry.episode_start_index == 4
    assert geometry.episode_end_index == 6


def test_distinctness_uses_every_episode_daughter_and_prebreak_parent():
    records = [
        _record(0.80, (10, 0, 0), (0, 10, 0)),
        _record(0.95, (0, 10, 0), (0, 9, 1)),
        _record(0.95, (0, 9, 1), (0, 8, 2)),
        _record(0.95, (0, 8, 2), (0, 7, 3)),
    ]
    geometry = episode_geometry(records)
    assert geometry.maximum_anchor_similarity == 0.0
    assert geometry.mean_anchor_similarity == 0.0


def test_persistence_is_attached_to_first_episode_not_any_later_run():
    records = _records("FTTTFTTTTT")
    assert evaluate_process(records).persist_5 == 1.0
    geometry = episode_geometry(records)
    assert geometry.observed_inherited_run_length == 3
    assert geometry.persistence_5_status == "failed"


@pytest.mark.parametrize(
    ("inheritance", "status"),
    (("FTTT", "right_censored"), ("FTTTF", "failed"), ("FTTTTT", "observed")),
)
def test_persistence_status_distinguishes_censoring(inheritance: str, status: str):
    assert episode_geometry(_records(inheritance)).persistence_5_status == status


def test_second_renewal_requires_later_break_and_second_run_of_three():
    assert episode_geometry(
        _records("FTTTFTTT")
    ).second_renewal_after_later_break_observed
    assert not episode_geometry(
        _records("FTTTTTTT")
    ).second_renewal_after_later_break_observed


def test_geometry_is_invariant_to_common_molecule_relabelling():
    records = _records("FTTTFTTT")
    permutation = np.asarray((2, 0, 3, 1))
    relabelled = [
        FissionRecord(
            parent=record.parent[permutation],
            daughter=record.daughter[permutation],
            h=record.h,
            growth_steps=record.growth_steps,
        )
        for record in records
    ]
    left = episode_geometry(records).to_dict()
    right = episode_geometry(relabelled).to_dict()
    assert left.keys() == right.keys()
    for key in left:
        if isinstance(left[key], float):
            assert left[key] == pytest.approx(right[key], abs=1e-15)
        else:
            assert left[key] == right[key]


def test_matrix_bootstrap_is_deterministic():
    values = np.asarray((0.0, 1.0, 0.5, 0.25, 0.75))
    matrix_ids = np.asarray((0, 0, 1, 2, 2))
    left = _cluster_mean_interval(values, matrix_ids, ("test",), repetitions=128)
    right = _cluster_mean_interval(values, matrix_ids, ("test",), repetitions=128)
    assert left == right


def test_matrix_bootstrap_includes_clusters_without_qualifying_events():
    values = np.asarray((0.0, 1.0))
    matrix_ids = np.asarray((0, 0))
    result = _cluster_mean_interval(
        values,
        matrix_ids,
        ("zero-event-cluster",),
        repetitions=128,
        matrix_universe=np.arange(3),
    )
    assert result[3] == 2
    assert result[4] == 3


def test_replay_validation_rejects_archived_mismatch():
    expected = pd.Series(
        {
            "joint_break_run3": 1,
            "completed_horizon": 1,
            "break_event": 1.0,
            "resume_2": 1.0,
            "episode_3": 1.0,
            "persist_5": 0.0,
            "old_return": 0.0,
            "positive_gain": 0.0,
            "repeat_return": np.nan,
            "old_anchor_gain": -0.2,
        }
    )
    observed = {
        "joint_break_run3_regenerated": 1,
        "completed_horizon_regenerated": 1,
        **{
            f"process_{name}_regenerated": expected[name]
            for name in (
                "break_event",
                "resume_2",
                "episode_3",
                "persist_5",
                "old_return",
                "positive_gain",
                "repeat_return",
                "old_anchor_gain",
            )
        },
    }
    assert _validate_replayed_row(expected, observed) == 0.0
    observed["process_old_anchor_gain_regenerated"] += 5e-15
    assert _validate_replayed_row(expected, observed) == pytest.approx(5e-15)
    observed["process_old_anchor_gain_regenerated"] = expected["old_anchor_gain"]
    observed["process_episode_3_regenerated"] = 0.0
    with pytest.raises(ValueError, match="episode_3"):
        _validate_replayed_row(expected, observed)


def test_audit_refuses_existing_output_before_replaying(tmp_path):
    destination = tmp_path / "existing"
    destination.mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_audit((), destination, workers=1)
