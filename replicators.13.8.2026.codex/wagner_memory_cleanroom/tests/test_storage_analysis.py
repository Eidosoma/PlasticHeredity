from __future__ import annotations

from pathlib import Path

from wagner_memory_cleanroom.analysis import analyze_carrier, analyze_state
from wagner_memory_cleanroom.config import load_registration
from wagner_memory_cleanroom.storage import read_records, records_digest, write_records


def _row(source, half, history, arm, correct, wrong=0, *, stage="state", checkpoint=None):
    return {
        "stage": stage,
        "source_id": source,
        "history": history,
        "writer": "hard-theta-0" if stage == "state" else "natural_latch",
        "arm": arm,
        "challenge": "neutral_damage" if stage == "state" else "release",
        "age": 0 if stage == "state" else None,
        "checkpoint": checkpoint,
        "theta": 0.0 if stage == "state" else None,
        "half_life": None,
        "coupling": None,
        "half": half,
        "n": 10,
        "correct": correct,
        "wrong": wrong,
        "both": 0,
        "unresolved": 10 - correct - wrong,
        "acquired": 1.0,
    }


def test_compressed_records_round_trip_and_digest_is_order_invariant(tmp_path: Path):
    rows = [_row(0, 0, "A", "state_transplant", 9), _row(0, 0, "A", "reset", 1)]
    first = tmp_path / "first.jsonl.gz"
    second = tmp_path / "second.jsonl.gz"
    write_records(first, rows)
    write_records(second, reversed(rows))
    assert read_records([first]) == rows
    assert records_digest([first]) == records_digest([second])


def test_clustered_state_analysis_does_not_count_futures_as_units():
    registration = load_registration("smoke")
    rows = []
    for source in range(8):
        for half in (0, 1):
            for history in ("A", "B"):
                for arm, correct in (("self", 9), ("state_transplant", 9), ("reset", 1), ("pattern_shuffle", 2)):
                    rows.append(_row(source, half, history, arm, correct))
                age = [_row(source, half, history, arm, correct) for arm, correct in (("state_transplant", 8), ("reset", 1))]
                for row in age:
                    row["age"] = 1
                    rows.append(row)
    # The soft writer must exist for the fixed analysis interface.
    soft = [dict(row, writer="soft-theta-0") for row in rows]
    result = analyze_state(registration, rows + soft)
    assert result["writers"]["hard-theta-0"]["risk_gain"]["n"] == 8
    assert result["writers"]["hard-theta-0"]["self_transplant_pathwise_identity"]

