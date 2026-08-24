from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import time
from typing import Any

from .config import Registration, load_registration
from .experiment import iter_replay_subset, run_stage_shard
from .storage import atomic_json, write_records


def _device_info() -> list[dict[str, Any]]:
    import jax

    return [{"id": int(item.id), "platform": str(item.platform), "device_kind": str(item.device_kind)} for item in jax.devices()]


def _benchmark_registration(registration: Registration) -> Registration:
    profile = dict(registration.profile)
    profile.update({
        "state_sources": 2,
        "boundary_sources": 2,
        "mark_sources": 2,
        "carrier_sources": 2,
        "futures_scale": 1.0,
        "bootstrap_repetitions": 128,
        "scientific": False,
    })
    return replace(registration, profile_name=registration.profile_name, profile=profile)


def run_worker(
    run_dir: Path,
    profile: str,
    stage: str,
    worker_index: int,
    worker_count: int,
    benchmark: bool,
) -> dict[str, Any]:
    registration = load_registration(profile)
    devices = _device_info()
    if registration.scientific and not benchmark and not any(row["platform"] == "gpu" for row in devices):
        raise RuntimeError("scientific worker has no CUDA device")
    if benchmark:
        benchmark_registration = _benchmark_registration(registration)
        stages: dict[str, Any] = {}
        for benchmark_stage in ("state", "boundary", "slow_mark", "carrier"):
            result = run_stage_shard(benchmark_registration, benchmark_stage, worker_index, worker_count)
            stages[benchmark_stage] = {
                "elapsed_seconds": result.elapsed_seconds,
                "simulated_futures": result.simulated_futures,
                "sources": len(result.sources),
                "records": len(result.records),
            }
        manifest = {
            "format": "wagner-memory-benchmark-worker-v1",
            "worker": worker_index,
            "devices": devices,
            "stages": stages,
        }
        atomic_json(run_dir / "benchmark" / f"worker-{worker_index}.json", manifest)
        return manifest

    result = run_stage_shard(registration, stage, worker_index, worker_count)
    stage_dir = run_dir / "stages" / stage
    count, digest = write_records(stage_dir / f"worker-{worker_index}.jsonl.gz", result.records)
    replay_limit = int(registration.operations["replay_futures_per_stage"])
    replay_first = iter_replay_subset(result.records, replay_limit)
    replay_second = iter_replay_subset(result.records, replay_limit)
    replay_verified = replay_first == replay_second
    manifest = {
        "format": "wagner-memory-worker-manifest-v1",
        "stage": stage,
        "worker": worker_index,
        "worker_count": worker_count,
        "profile": profile,
        "scientific": registration.scientific,
        "devices": devices,
        "records": count,
        "record_digest": digest,
        "simulated_futures": result.simulated_futures,
        "sources": result.sources,
        "elapsed_seconds": result.elapsed_seconds,
        "replay_records": replay_first,
        "replay_verified": replay_verified,
    }
    atomic_json(stage_dir / f"worker-{worker_index}.manifest.json", manifest)
    return manifest

