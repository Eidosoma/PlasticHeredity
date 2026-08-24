import numpy as np

from plastic_heredity.processes import evaluate_process
from plastic_heredity.simulator import FissionRecord


def _record(h: float, daughter: tuple[int, int]) -> FissionRecord:
    return FissionRecord(
        parent=np.asarray((8, 2), dtype=np.int64),
        daughter=np.asarray(daughter, dtype=np.int64),
        h=h,
        growth_steps=1,
    )


def test_joint_target_requires_break_before_run_of_three():
    records = [
        _record(0.95, (4, 1)),
        _record(0.80, (1, 4)),
        _record(0.95, (4, 1)),
        _record(0.96, (4, 1)),
        _record(0.97, (4, 1)),
    ]
    outcome = evaluate_process(records)
    assert outcome.break_event
    assert outcome.resume_2 == 1.0
    assert outcome.episode_3 == 1.0
    assert outcome.joint_break_run3


def test_uninterrupted_run_is_not_a_new_episode():
    outcome = evaluate_process([_record(0.95, (4, 1)) for _ in range(12)])
    assert not outcome.break_event
    assert not outcome.joint_break_run3
    assert np.isnan(outcome.episode_3)


def test_break_without_certified_run_is_negative():
    sequence = (0.8, 0.95, 0.95, 0.7, 0.95, 0.95)
    outcome = evaluate_process([_record(h, (1, 4)) for h in sequence])
    assert outcome.break_event
    assert outcome.resume_2 == 1.0
    assert outcome.episode_3 == 0.0
    assert not outcome.joint_break_run3


def test_old_return_is_scored_at_first_resumption_not_later_suffix():
    records = [
        _record(0.80, (1, 4)),
        _record(0.95, (1, 4)),  # first resumption remains far from old anchor
        _record(0.95, (8, 2)),  # later visit must not redefine the resumption
    ]
    outcome = evaluate_process(records)
    assert outcome.old_return == 0.0
