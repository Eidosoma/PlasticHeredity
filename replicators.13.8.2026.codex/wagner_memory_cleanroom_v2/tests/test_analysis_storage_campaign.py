from __future__ import annotations

import gzip
import json

import numpy as np

from wagner_memory_cleanroom_v2.analysis import (
    _direct_crossover,
    _history_logloss_gain_values,
    _mean_strict_correct,
    _reliability,
    load_stage_records,
)
from wagner_memory_cleanroom_v2.campaign import (
    _worker_environment,
    freeze_registration,
    verify_source_snapshot,
)
from wagner_memory_cleanroom_v2.config import load_registration, load_run_registration
from wagner_memory_cleanroom_v2.storage import records_digest, write_records


def _row(
    source: int, half: int, history: str, arm: str, a: int, b: int,
    other: int = 0, hold_a: int | None = None, hold_b: int | None = None,
) -> dict:
    n = a + b + other
    return {
        "cell_id": f"{source}-{half}-{history}-{arm}",
        "source_id": source,
        "half": half,
        "history": history,
        "arm": arm,
        "condition": "test",
        "dest_a": a,
        "dest_b": b,
        "dest_other": other,
        "hold_a": a if hold_a is None else hold_a,
        "hold_b": b if hold_b is None else hold_b,
        "hold_both": 0,
        "n": n,
    }


def test_direct_crossover_is_within_treatment_not_treatment_minus_control() -> None:
    a = _row(0, 0, "A", "treatment", 9, 1)
    b = _row(0, 0, "B", "treatment", 2, 8)
    assert np.isclose(_direct_crossover(a), .8)
    assert np.isclose(_direct_crossover(b), .6)


def test_absolute_hold_uses_strict_eight_cycle_endpoint_not_prediction_destination() -> None:
    row = _row(0, 0, "A", "treatment", 9, 1, hold_a=2, hold_b=0)
    assert np.isclose(_mean_strict_correct([row]), .2)


def test_history_committor_crossfits_halves_against_source_pooled_baseline() -> None:
    rows = []
    for source in range(3):
        for half in (0, 1):
            rows.extend([
                _row(source, half, "A", "carrier", 9, 1),
                _row(source, half, "B", "carrier", 1, 9),
            ])
    sources, gains = _history_logloss_gain_values(rows, "carrier", condition="test")
    assert sources.tolist() == [0, 1, 2]
    assert np.all(gains > .2)


def test_identical_split_halves_have_reliability_one() -> None:
    values = np.asarray([.1, .2, .3])
    assert _reliability(values, values.copy()) == 1.0


def test_record_digest_is_order_independent_and_gzip_is_deterministic(tmp_path) -> None:
    records = [
        {"cell_id": "b", "value": 2},
        {"cell_id": "a", "value": 1},
    ]
    first = tmp_path / "first.jsonl.gz"
    second = tmp_path / "second.jsonl.gz"
    write_records(first, records)
    write_records(second, reversed(records))
    assert records_digest([first]) == records_digest([second])
    with gzip.open(first, "rt") as handle:
        assert len([json.loads(line) for line in handle]) == 2


def test_run_workers_load_only_the_frozen_registration(tmp_path) -> None:
    registration = load_registration("smoke")
    run = tmp_path / "run"
    payload = freeze_registration(run, registration)
    loaded = load_run_registration(run)
    assert loaded.protocol_digest == registration.protocol_digest
    assert loaded.profile == registration.profile
    assert payload["source_snapshot_sha256"]
    assert (run / "provenance" / "scripts" / "run-campaign-detached.sh").is_file()
    assert verify_source_snapshot(run)["valid"]
    environment = _worker_environment(0, False, run)
    assert environment["PYTHONPATH"] == str((run / "provenance" / "src").resolve())
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"


def test_gpu_requested_diagnostic_workers_are_physically_isolated(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setenv("JAX_PLATFORMS", "cuda")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    environment = _worker_environment(1, False, tmp_path)
    assert environment["CUDA_VISIBLE_DEVICES"] == "1"
    assert environment["JAX_PLATFORMS"] == "cuda"


def test_analysis_loader_excludes_source_provenance_stream(tmp_path) -> None:
    stage = tmp_path / "stages" / "state"
    write_records(stage / "worker-0.jsonl.gz", [{"cell_id": "cell", "n": 1}])
    write_records(stage / "worker-0.sources.jsonl.gz", [{"source_id": 0}])
    assert load_stage_records(tmp_path, "state") == [{"cell_id": "cell", "n": 1}]
