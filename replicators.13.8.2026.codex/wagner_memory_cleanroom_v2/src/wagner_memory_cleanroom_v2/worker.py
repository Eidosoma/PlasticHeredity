from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import time
from typing import Any

from .config import Registration, canonical_json, load_run_registration
from .contracts import BASE_STAGES, CAMPAIGN_STAGES, assigned_source_ids, validate_stage_records
from .experiment import run_stage_shard
from .storage import atomic_json, write_records


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _device_info() -> list[dict[str, Any]]:
    import jax

    return [
        {
            "id": int(item.id),
            "platform": str(item.platform),
            "device_kind": str(item.device_kind),
            "physical_gpu": os.environ.get("WAGNER_PHYSICAL_GPU"),
        }
        for item in jax.devices()
    ]


def _benchmark_registration(registration: Registration, workers: int) -> Registration:
    per_gpu = int(registration.operations["benchmark_sources_per_gpu"])
    profile = dict(registration.profile)
    profile.update({
        "state_sources": per_gpu * workers,
        "boundary_sources": per_gpu * workers,
        "mark_sources": per_gpu * workers,
        "carrier_sources": per_gpu * workers,
        "futures_scale": 1.0,
        "bootstrap_repetitions": 128,
        "scientific": False,
    })
    return replace(registration, profile=profile)


def _require_worker_device(registration: Registration, devices: list[dict[str, Any]], workers: int) -> None:
    required = int(registration.operations["required_gpu_count"])
    if registration.scientific and workers != required:
        raise RuntimeError(f"scientific worker count {workers} != registered GPU count {required}")
    gpu_devices = [row for row in devices if row["platform"] in {"gpu", "cuda"}]
    if registration.scientific and len(gpu_devices) != 1:
        raise RuntimeError(f"each scientific worker requires exactly one visible GPU; found {len(gpu_devices)}")


def _replay_digest(replay: list[dict[str, Any]]) -> str:
    digest = sha256()
    for row in replay:
        digest.update(canonical_json(row))
    return digest.hexdigest()


def _progress_writer(path: Path, stage: str, worker: int, started: float):
    def write(values: dict[str, Any]) -> None:
        payload = {
            "format": "wagner-memory-worker-progress-v2",
            "state": "running",
            "stage": stage,
            "worker": worker,
            "wall_elapsed_seconds": time.monotonic() - started,
            "updated_at": utc_now(),
            **values,
        }
        atomic_json(path, payload)
        print(json.dumps(payload, sort_keys=True), flush=True)

    return write


def run_worker(
    run_dir: Path,
    stage: str,
    worker_index: int,
    worker_count: int,
    benchmark: bool = False,
) -> dict[str, Any]:
    expected_source_root = (run_dir / "provenance" / "src").resolve()
    running_source_root = Path(__file__).resolve().parents[1]
    if running_source_root != expected_source_root:
        raise RuntimeError(
            f"worker source is not the frozen run snapshot: {running_source_root} != {expected_source_root}"
        )
    registration_payload = json.loads((run_dir / "registration.json").read_text())
    source_snapshot_sha256 = str(registration_payload["source_snapshot_sha256"])
    registration = load_run_registration(run_dir)
    if not 0 <= worker_index < worker_count:
        raise ValueError("worker index is outside its partition")
    devices = _device_info()
    _require_worker_device(registration, devices, worker_count)
    started = time.monotonic()

    if benchmark:
        if stage != "all":
            raise ValueError("benchmark worker stage must be 'all'")
        benchmark_dir = run_dir / "benchmark"
        progress_path = benchmark_dir / f"worker-{worker_index}.progress.json"
        progress = _progress_writer(progress_path, "benchmark", worker_index, started)
        benchmark_registration = _benchmark_registration(registration, worker_count)
        stages: dict[str, Any] = {}
        try:
            for benchmark_stage in BASE_STAGES:
                progress({"benchmark_stage": benchmark_stage, "sources_complete": 0})
                result = run_stage_shard(
                    benchmark_registration,
                    benchmark_stage,
                    worker_index,
                    worker_count,
                    progress=progress,
                    source_domain=f"benchmark:{registration.profile_name}:{benchmark_stage}",
                )
                contract = validate_stage_records(
                    benchmark_stage,
                    benchmark_registration,
                    result.records,
                    result.sources,
                    worker_index,
                    worker_count,
                )
                if not contract["valid"]:
                    raise RuntimeError(f"benchmark {benchmark_stage} count contract failed: {contract['checks']}")
                source_count = len(result.sources)
                stages[benchmark_stage] = {
                    "elapsed_seconds": result.elapsed_seconds,
                    "seconds_per_source": result.elapsed_seconds / max(1, source_count),
                    "simulated_futures": result.simulated_futures,
                    "sources": source_count,
                    "records": len(result.records),
                    "contract": contract,
                }
            manifest = {
                "format": "wagner-memory-benchmark-worker-v2",
                "worker": worker_index,
                "worker_count": worker_count,
                "devices": devices,
                "running_source_root": str(running_source_root),
                "source_snapshot_sha256": source_snapshot_sha256,
                "stages": stages,
                "elapsed_seconds": time.monotonic() - started,
                "completed_at": utc_now(),
            }
            atomic_json(benchmark_dir / f"worker-{worker_index}.json", manifest)
            atomic_json(progress_path, {
                "format": "wagner-memory-worker-progress-v2",
                "state": "complete",
                "stage": "benchmark",
                "worker": worker_index,
                "elapsed_seconds": time.monotonic() - started,
                "updated_at": utc_now(),
            })
            return manifest
        except Exception as exc:
            atomic_json(progress_path, {
                "format": "wagner-memory-worker-progress-v2",
                "state": "failed",
                "stage": "benchmark",
                "worker": worker_index,
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": time.monotonic() - started,
                "updated_at": utc_now(),
            })
            raise

    if stage not in CAMPAIGN_STAGES:
        raise ValueError(f"unsupported campaign stage: {stage}")
    stage_dir = run_dir / "stages" / stage
    progress_path = stage_dir / f"worker-{worker_index}.progress.json"
    progress = _progress_writer(progress_path, stage, worker_index, started)
    expected_sources = assigned_source_ids(stage, registration, worker_index, worker_count)
    progress({"sources_complete": 0, "sources_total": len(expected_sources)})
    try:
        result = run_stage_shard(
            registration, stage, worker_index, worker_count, progress=progress
        )
        contract = validate_stage_records(
            stage,
            registration,
            result.records,
            result.sources,
            worker_index,
            worker_count,
        )
        if not contract["valid"]:
            raise RuntimeError(f"stage count contract failed: {contract['checks']}")
        record_count, record_digest = write_records(
            stage_dir / f"worker-{worker_index}.jsonl.gz", result.records
        )
        source_count, source_digest = write_records(
            stage_dir / f"worker-{worker_index}.sources.jsonl.gz", result.sources
        )
        replay_ids = [str(row["future_id"]) for row in result.replay]
        if len(replay_ids) != len(set(replay_ids)):
            raise RuntimeError("replay future IDs are not unique")
        replay_limit = int(registration.operations["replay_futures_per_stage"])
        if len(result.replay) != min(replay_limit, result.simulated_futures):
            raise RuntimeError("replay sample has the wrong size")
        manifest = {
            "format": "wagner-memory-worker-manifest-v2",
            "stage": stage,
            "base_stage": stage.removesuffix("_audit"),
            "worker": worker_index,
            "worker_count": worker_count,
            "profile": registration.profile_name,
            "scientific": registration.scientific,
            "protocol_digest": registration.protocol_digest,
            "devices": devices,
            "running_source_root": str(running_source_root),
            "source_snapshot_sha256": source_snapshot_sha256,
            "records": record_count,
            "record_stream_sha256": record_digest,
            "source_records": source_count,
            "source_stream_sha256": source_digest,
            "simulated_futures": result.simulated_futures,
            "elapsed_seconds": result.elapsed_seconds,
            "contract": contract,
            "replay": result.replay,
            "replay_sha256": _replay_digest(result.replay),
            "completed_at": utc_now(),
        }
        atomic_json(stage_dir / f"worker-{worker_index}.manifest.json", manifest)
        atomic_json(progress_path, {
            "format": "wagner-memory-worker-progress-v2",
            "state": "complete",
            "stage": stage,
            "worker": worker_index,
            "sources_complete": len(expected_sources),
            "sources_total": len(expected_sources),
            "elapsed_seconds": time.monotonic() - started,
            "updated_at": utc_now(),
        })
        return manifest
    except Exception as exc:
        atomic_json(progress_path, {
            "format": "wagner-memory-worker-progress-v2",
            "state": "failed",
            "stage": stage,
            "worker": worker_index,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": time.monotonic() - started,
            "updated_at": utc_now(),
        })
        raise
