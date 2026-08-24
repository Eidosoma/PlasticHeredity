from __future__ import annotations

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import pytest

from reviewer_motif_channel_replication import campaign, cohorts, contract, engine, inference
from reviewer_motif_channel_replication.campaign import _run_checkpoint_tasks
from reviewer_motif_channel_replication.cohorts import construct_fresh_pair_pool
from reviewer_motif_channel_replication.contract import (
    checkpoint_envelope,
    hash_order,
    read_checkpoint,
    seal_registration,
    semantic_seed,
    verify_registration,
    write_checkpoint,
)
from reviewer_motif_channel_replication.snapshot import SOURCE_FILES


def _worker_seed_value(index: int) -> tuple[int, int]:
    return index, semantic_seed("worker-invariant", index, "reader")


def _checkpoint_worker(argument: dict) -> dict:
    return {"pair_id": argument["pair"]["pair_id"], "value": argument["value"]}


def _interrupting_worker(argument: dict) -> dict:
    if argument["value"] == 1:
        raise RuntimeError("synthetic interruption")
    return _checkpoint_worker(argument)


def test_hash_order_and_seeds_ignore_worker_and_input_order() -> None:
    items = ["gamma", "alpha", "beta"]
    assert hash_order(items, "namespace") == hash_order(reversed(items), "namespace")
    assert semantic_seed("pair", 7, "reader") == semantic_seed("pair", 7, "reader")
    assert semantic_seed("pair", 7, "reader") != semantic_seed("pair", 7, "noise")


def test_registration_and_checkpoint_tamper_detection(tmp_path: Path) -> None:
    registration = seal_registration({"stage": 1, "profile": {"pairs": 4}})
    verify_registration(registration)
    altered = {**registration, "stage": 2}
    with pytest.raises(ValueError, match="digest"):
        verify_registration(altered)
    checkpoint = tmp_path / "checkpoint.json"
    write_checkpoint(checkpoint, registration["design_digest"], {"value": 3})
    assert read_checkpoint(checkpoint, registration["design_digest"])["value"] == 3
    text = checkpoint.read_text().replace('"value": 3', '"value": 4')
    checkpoint.write_text(text)
    with pytest.raises(ValueError, match="checksum"):
        read_checkpoint(checkpoint, registration["design_digest"])


def test_fresh_pool_excludes_historical_donors_and_never_reuses() -> None:
    donors = []
    for label, start in (("A", 1), ("B", 101)):
        for offset in range(8):
            donors.append(
                {
                    "donor_id": f"life-31649-0-{start + offset}",
                    "prototype_label": label,
                    "launch_index": 0,
                    "density": 0.5 + (offset % 2) / 256,
                }
            )
    excluded_pair = "narrow-0000-life-31649-0-1-life-31649-0-101"
    pool = construct_fresh_pair_pool(donors, [excluded_pair])
    used = [
        donor
        for pair in pool
        for donor in (pair["a_donor_id"], pair["b_donor_id"])
    ]
    assert "life-31649-0-1" not in used
    assert "life-31649-0-101" not in used
    assert len(used) == len(set(used))
    assert all(pair["density_difference"] <= 0.02 for pair in pool)


def test_source_firewall_is_static_and_data_document_only() -> None:
    assert SOURCE_FILES
    assert all(Path(path).suffix in {".json", ".md"} for path in SOURCE_FILES)
    scientific_modules = (contract, cohorts, engine, inference, campaign)
    for module in scientific_modules:
        source = Path(module.__file__).read_text()
        assert "NewIdeas" not in source
        assert "codex.reconstructionsAndStressTesting" not in source


def test_checkpoint_envelope_is_deterministic() -> None:
    first = checkpoint_envelope("binding", {"b": 2, "a": 1})
    second = checkpoint_envelope("binding", {"a": 1, "b": 2})
    assert first == second


def test_semantic_results_are_worker_count_invariant() -> None:
    tasks = list(range(12))
    sequential = dict(_worker_seed_value(index) for index in tasks)
    with ProcessPoolExecutor(max_workers=3) as executor:
        parallel = dict(executor.map(_worker_seed_value, reversed(tasks)))
    assert sequential == parallel


def test_interruption_keeps_atomic_checkpoints_and_resume_skips_them(
    tmp_path: Path,
) -> None:
    arguments = [
        {"pair": {"pair_id": f"pair-{index}"}, "value": index}
        for index in range(3)
    ]
    checkpoint_dir = tmp_path / "stage" / "checkpoints"
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        _run_checkpoint_tasks(
            arguments,
            _interrupting_worker,
            checkpoint_dir,
            "binding",
            1,
            tmp_path,
        )
    assert (checkpoint_dir / "pair-0.json").exists()
    assert not (checkpoint_dir / "pair-1.json").exists()
    assert contract.load_json(tmp_path / "stage" / "PROGRESS.json")["state"] == "interrupted"
    completed = _run_checkpoint_tasks(
        arguments,
        _checkpoint_worker,
        checkpoint_dir,
        "binding",
        1,
        tmp_path,
    )
    assert completed == 2
    assert contract.load_json(tmp_path / "stage" / "PROGRESS.json")["state"] == "complete"
