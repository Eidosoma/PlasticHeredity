from __future__ import annotations

from pathlib import Path

import numpy as np

from wagner_cleanroom.dynamics import sample_rulebook
from wagner_cleanroom.experiment import expected_rows
from wagner_cleanroom.protocol import digest, load_protocol, registration
from wagner_cleanroom.storage import load_rulebook, save_rulebook, seal_directory, verify_checksums
from wagner_cleanroom.verification import cleanroom_violations, format_primary_future_id, parse_primary_future_id


def accepted_rulebook():
    for proposal in range(100):
        result = sample_rulebook("storage-test", proposal)
        if result is not None:
            return result
    raise AssertionError("no eligible fixture rulebook")


def test_registered_counts_and_digests() -> None:
    primary = load_protocol("primary")
    assert expected_rows(primary) == 3_194_880
    assert registration(primary)["protocol_digest"] == digest(primary)
    smoke = load_protocol("primary", "smoke")
    assert not smoke["scientific"]
    assert expected_rows(smoke) < expected_rows(primary)


def test_rulebook_round_trip(tmp_path: Path) -> None:
    original = accepted_rulebook()
    path = tmp_path / "source.npz"
    save_rulebook(path, original)
    restored = load_rulebook(path)
    assert restored.uid == original.uid
    assert np.array_equal(restored.weights, original.weights)
    assert np.array_equal(restored.landscape.successor, original.landscape.successor)
    assert np.array_equal(restored.targets, original.targets)


def test_checksum_seal(tmp_path: Path) -> None:
    (tmp_path / "one.txt").write_text("one\n", encoding="utf-8")
    seal_directory(tmp_path)
    assert verify_checksums(tmp_path) == []
    (tmp_path / "one.txt").write_text("changed\n", encoding="utf-8")
    assert verify_checksums(tmp_path) == ["checksum mismatch one.txt"]


def test_future_id_is_reversible() -> None:
    coordinates = (12, 2, 5, 1, 0, 2, 8, 63)
    assert parse_primary_future_id(format_primary_future_id(*coordinates)) == coordinates


def test_cleanroom_scan_passes() -> None:
    assert cleanroom_violations() == []

