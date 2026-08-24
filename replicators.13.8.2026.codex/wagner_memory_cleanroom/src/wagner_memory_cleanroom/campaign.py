from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

from .analysis import analyze_all
from .config import Registration, canonical_json
from .reporting import write_reports
from .storage import atomic_json, records_digest, sha256_manifest


STAGES = ("state", "boundary", "slow_mark", "carrier", "state_audit", "carrier_audit")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _registration_payload(registration: Registration) -> dict[str, Any]:
    return {
        "format": "wagner-memory-registration-v1",
        "protocol_digest": registration.protocol_digest,
        "profile": registration.profile_name,
        "scientific": registration.scientific,
        "protocol": registration.protocol,
    }


def ensure_registration(run_dir: Path, registration: Registration) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "registration.json"
    payload = _registration_payload(registration)
    if path.exists():
        existing = json.loads(path.read_text())
        if canonical_json(existing) != canonical_json(payload):
            raise RuntimeError("run registration differs from the frozen protocol/profile")
    else:
        atomic_json(path, payload)


def _status(run_dir: Path, **updates: Any) -> None:
    path = run_dir / "STATUS.json"
    current = json.loads(path.read_text()) if path.exists() else {
        "format": "wagner-memory-status-v1",
        "created_at": utc_now(),
    }
    current.update(updates)
    current["updated_at"] = utc_now()
    atomic_json(path, current)


def available_devices() -> list[dict[str, Any]]:
    import jax

    return [
        {"id": int(device.id), "platform": str(device.platform), "device_kind": str(device.device_kind)}
        for device in jax.devices()
    ]


def require_devices(registration: Registration) -> list[dict[str, Any]]:
    devices = available_devices()
    gpu_devices = [device for device in devices if device["platform"] == "gpu"]
    required = int(registration.operations["required_gpu_count"])
    if registration.scientific and len(gpu_devices) != required:
        raise RuntimeError(f"scientific profile requires exactly {required} visible GPUs; found {len(gpu_devices)}")
    return devices


def _worker_command(run_dir: Path, registration: Registration, stage: str, worker: int, benchmark: bool = False) -> list[str]:
    command = [
        sys.executable, "-m", "wagner_memory_cleanroom", "worker",
        "--run", str(run_dir), "--profile", registration.profile_name,
        "--stage", stage, "--worker-index", str(worker), "--worker-count", "2",
    ]
    if benchmark:
        command.append("--benchmark")
    return command


def _worker_environment(worker: int, scientific: bool) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update({
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    })
    if scientific or environment.get("WAGNER_FORCE_GPU") == "1":
        environment["CUDA_VISIBLE_DEVICES"] = str(worker)
        environment["JAX_PLATFORMS"] = "cuda"
    else:
        environment.setdefault("JAX_PLATFORMS", "cpu")
    return environment


def _run_parallel_workers(
    run_dir: Path,
    registration: Registration,
    stage: str,
    *,
    benchmark: bool = False,
    timeout_seconds: float | None = None,
) -> None:
    processes: list[tuple[int, subprocess.Popen[bytes], Any]] = []
    log_dir = run_dir / ("benchmark" if benchmark else "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    for worker in range(2):
        log_path = log_dir / f"{stage}-gpu{worker}.log"
        handle = log_path.open("wb")
        process = subprocess.Popen(
            _worker_command(run_dir, registration, stage, worker, benchmark),
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=_worker_environment(worker, registration.scientific),
            start_new_session=True,
        )
        processes.append((worker, process, handle))
    started = time.monotonic()
    failures: list[str] = []
    try:
        for worker, process, handle in processes:
            remaining = None if timeout_seconds is None else max(1.0, timeout_seconds - (time.monotonic() - started))
            try:
                return_code = process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                for _, other, _ in processes:
                    if other.poll() is None:
                        os.killpg(other.pid, 15)
                failures.append(f"worker {worker} exceeded stage timeout")
                break
            if return_code != 0:
                failures.append(f"worker {worker} exited {return_code}")
            handle.close()
    finally:
        for _, process, handle in processes:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
            if not handle.closed:
                handle.close()
    if failures:
        raise RuntimeError(f"{stage} failed: {'; '.join(failures)}")


def run_benchmark(run_dir: Path, registration: Registration) -> dict[str, Any]:
    ensure_registration(run_dir, registration)
    require_devices(registration)
    benchmark_dir = run_dir / "benchmark"
    if benchmark_dir.exists():
        for path in benchmark_dir.glob("worker-*.json"):
            path.unlink()
    _status(run_dir, phase="benchmark", state="running")
    _run_parallel_workers(run_dir, registration, "all", benchmark=True, timeout_seconds=3600)
    manifests = [json.loads((benchmark_dir / f"worker-{worker}.json").read_text()) for worker in range(2)]
    stage_seconds: dict[str, float] = {}
    stage_futures: dict[str, int] = {}
    full_counts = {
        "state": int(registration.profile["state_sources"]),
        "boundary": int(registration.profile["boundary_sources"]),
        "slow_mark": int(registration.profile["mark_sources"]),
        "carrier": int(registration.profile["carrier_sources"]),
    }
    projected = 0.0
    for stage in ("state", "boundary", "slow_mark", "carrier"):
        seconds_per_source = max(float(row["stages"][stage]["elapsed_seconds"]) for row in manifests)
        stage_seconds[stage] = seconds_per_source
        stage_futures[stage] = sum(int(row["stages"][stage]["simulated_futures"]) for row in manifests)
        multiplier = 2 if stage in {"state", "carrier"} else 1
        projected += seconds_per_source * (full_counts[stage] / 2.0) * multiplier
    projected *= float(registration.operations["benchmark_margin"])
    projected_hours = projected / 3600.0
    admitted = (not registration.scientific) or projected_hours <= float(registration.operations["admission_hours_with_margin"])
    result = {
        "format": "wagner-memory-admission-v1",
        "profile": registration.profile_name,
        "scientific": registration.scientific,
        "workers": manifests,
        "stage_seconds_per_source_slowest_gpu": stage_seconds,
        "benchmark_futures": stage_futures,
        "margin": float(registration.operations["benchmark_margin"]),
        "projected_hours_with_margin": projected_hours,
        "limit_hours": float(registration.operations["admission_hours_with_margin"]),
        "admitted": admitted,
        "created_at": utc_now(),
    }
    atomic_json(run_dir / "benchmark.json", result)
    _status(run_dir, phase="benchmark", state="complete", admitted=admitted, projected_hours=projected_hours)
    return result


def verify_run(run_dir: Path) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    all_equal = True
    for stage in ("state", "carrier"):
        original_paths = sorted((run_dir / "stages" / stage).glob("worker-*.jsonl.gz"))
        audit_paths = sorted((run_dir / "stages" / f"{stage}_audit").glob("worker-*.jsonl.gz"))
        original_count, original_digest = records_digest(original_paths)
        audit_count, audit_digest = records_digest(audit_paths)
        equal = original_count == audit_count and original_digest == audit_digest
        all_equal &= equal
        comparisons[stage] = {
            "original_records": original_count,
            "audit_records": audit_count,
            "original_digest": original_digest,
            "audit_digest": audit_digest,
            "equal": equal,
        }
    worker_manifests = [
        json.loads(path.read_text())
        for path in sorted((run_dir / "stages").glob("*/worker-*.manifest.json"))
    ]
    replay_verified = bool(worker_manifests) and all(row.get("replay_verified") for row in worker_manifests)
    result = {
        "format": "wagner-memory-verification-v1",
        "comparisons": comparisons,
        "independent_regeneration_verified": all_equal,
        "replay_verified": replay_verified,
        "verified": all_equal and replay_verified,
        "created_at": utc_now(),
    }
    atomic_json(run_dir / "verification.json", result)
    return result


def seal_run(run_dir: Path) -> None:
    entries = sha256_manifest(run_dir)
    atomic_json(run_dir / "manifest.json", {"format": "wagner-memory-manifest-v1", "files": entries})
    lines = [f"{entry['sha256']}  {entry['path']}" for entry in entries]
    (run_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n")


def run_campaign(run_dir: Path, registration: Registration) -> dict[str, Any]:
    ensure_registration(run_dir, registration)
    devices = require_devices(registration)
    free_gib = shutil.disk_usage(run_dir).free / 1024**3
    if registration.scientific and free_gib < float(registration.operations["minimum_free_gib"]):
        raise RuntimeError(f"free disk {free_gib:.1f} GiB is below registered guard")
    started = time.monotonic()
    _status(run_dir, phase="admission", state="running", devices=devices, free_gib=free_gib)
    benchmark = run_benchmark(run_dir, registration)
    if not benchmark["admitted"]:
        raise RuntimeError(f"admission rejected projected {benchmark['projected_hours_with_margin']:.2f} h")
    hard_deadline = float(registration.operations["hard_deadline_hours"]) * 3600
    completed: list[str] = []
    for stage in STAGES:
        elapsed = time.monotonic() - started
        if registration.scientific and elapsed >= float(registration.operations["stop_new_shards_hours"]) * 3600:
            raise RuntimeError("registered stop-new-shards boundary reached")
        remaining = hard_deadline - elapsed
        _status(run_dir, phase=stage, state="running", completed_stages=completed, elapsed_seconds=elapsed)
        _run_parallel_workers(run_dir, registration, stage, timeout_seconds=max(60.0, remaining))
        completed.append(stage)
        _status(run_dir, phase=stage, state="complete", completed_stages=completed, elapsed_seconds=time.monotonic() - started)
    verification = verify_run(run_dir)
    if not verification["verified"]:
        raise RuntimeError("independent regeneration or replay verification failed")
    analysis = analyze_all(registration, run_dir)
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    for key in ("state", "boundary", "slow_mark", "carrier"):
        atomic_json(analysis_dir / f"{key}.json", analysis[key])
    atomic_json(analysis_dir / "summary.json", analysis)
    write_reports(run_dir, analysis, verification)
    runtime = {
        "format": "wagner-memory-runtime-v1",
        "profile": registration.profile_name,
        "scientific": registration.scientific,
        "devices": devices,
        "elapsed_seconds": time.monotonic() - started,
        "python": sys.version,
        "completed_at": utc_now(),
    }
    atomic_json(run_dir / "runtime_manifest.json", runtime)
    _status(run_dir, phase="sealed", state="complete", completed_stages=completed, elapsed_seconds=runtime["elapsed_seconds"], verdict=analysis["overall_verdict"])
    seal_run(run_dir)
    return analysis

