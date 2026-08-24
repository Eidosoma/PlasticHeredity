from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .simulator import FissionRecord, cosine_similarity

JOINT_BREAK_RUN3 = "JOINT_BREAK_RUN3"


@dataclass(frozen=True)
class ProcessOutcome:
    joint_break_run3: bool
    break_event: bool
    resume_2: float
    episode_3: float
    persist_5: float
    old_return: float
    positive_gain: float
    repeat_return: float
    old_anchor_gain: float

    def to_dict(self) -> dict[str, float | bool]:
        return asdict(self)


def _has_run(values: np.ndarray, length: int) -> bool:
    if values.size < length:
        return False
    return any(bool(values[start : start + length].all()) for start in range(values.size - length + 1))


def evaluate_process(
    records: list[FissionRecord], inheritance_threshold: float = 0.9
) -> ProcessOutcome:
    """Evaluate the destination-free break-and-renewal process.

    Conditional quantities are NaN when their conditioning event is absent. An
    old-neighbourhood return means H > threshold with the pre-break parent after
    the breaking fission. Repeat-return is conditioned on at least one such return.
    """

    inherited = np.asarray(
        [record.h > inheritance_threshold for record in records], dtype=bool
    )
    break_locations = np.flatnonzero(~inherited)
    if break_locations.size == 0:
        return ProcessOutcome(
            joint_break_run3=False,
            break_event=False,
            resume_2=np.nan,
            episode_3=np.nan,
            persist_5=np.nan,
            old_return=np.nan,
            positive_gain=np.nan,
            repeat_return=np.nan,
            old_anchor_gain=np.nan,
        )

    first_break = int(break_locations[0])
    after_break = inherited[first_break + 1 :]
    resume_2 = _has_run(after_break, 2)
    episode_3 = _has_run(after_break, 3)
    persist_5 = _has_run(after_break, 5)

    anchor = records[first_break].parent
    break_daughter = records[first_break].daughter
    baseline_h = cosine_similarity(anchor, break_daughter)
    later_records = records[first_break + 1 :]
    old_h = np.asarray(
        [cosine_similarity(anchor, record.daughter) for record in later_records],
        dtype=np.float64,
    )
    resumption_locations = np.flatnonzero(after_break)
    if resumption_locations.size:
        resumption_offset = int(resumption_locations[0])
        resumption_h = float(old_h[resumption_offset])
        gain = resumption_h - baseline_h
        positive_gain = float(gain > 0.0)
        returned = old_h > inheritance_threshold
        # "Return" is evaluated when heredity first resumes, rather than by
        # searching the full suffix for an unrelated later visit.
        old_return = bool(returned[resumption_offset])
        repeat_return = (
            float(returned[resumption_offset:].sum() >= 2)
            if old_return
            else np.nan
        )
    else:
        gain = np.nan
        positive_gain = np.nan
        old_return = False
        repeat_return = np.nan

    return ProcessOutcome(
        joint_break_run3=bool(episode_3),
        break_event=True,
        resume_2=float(resume_2),
        episode_3=float(episode_3),
        persist_5=float(persist_5),
        old_return=float(old_return),
        positive_gain=positive_gain,
        repeat_return=repeat_return,
        old_anchor_gain=float(gain),
    )
