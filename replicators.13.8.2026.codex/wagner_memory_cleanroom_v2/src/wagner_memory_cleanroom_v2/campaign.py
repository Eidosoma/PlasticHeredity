from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time
from typing import Any

from .analysis import analyze_all
from .config import PROJECT_DIR, Registration, canonical_json, load_run_registration
from .contracts import BASE_STAGES, CAMPAIGN_STAGES, stage_expectation, validate_protocol_counts
from .reporting import write_reports
from .storage import (
    atomic_json,
    objects_digest,
    records_digest,
    sha256_file,
    sha256_manifest,
)
from .validation import validate


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _status(run_dir: Path, **updates: Any) -> dict[str, Any]:
    path = run_dir / "STATUS.json"
    current = json.loads(path.read_text()) if path.exists() else {
        "format": "wagner-memory-status-v2",
        "created_at": utc_now(),
    }
    current.update(updates)
    current["updated_at"] = utc_now()
    atomic_json(path, current)
    return current


def _copy_frozen(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256_file(source) != sha256_file(destination):
            raise RuntimeError(f"frozen provenance collision: {destination}")
        return
    shutil.copy2(source, destination)


def freeze_registration(run_dir: Path, registration: Registration) -> dict[str, Any]:
    run = run_dir.resolve()
    if run == PROJECT_DIR.resolve():
        raise ValueError("the run directory cannot be the project directory")
    run.mkdir(parents=True, exist_ok=True)
    frozen_protocol = run / "provenance" / "protocols" / registration.protocol_path.name
    _copy_frozen(registration.protocol_path, frozen_protocol)

    source_root = PROJECT_DIR / "src"
    frozen_source = run / "provenance" / "src"
    copied: list[dict[str, Any]] = []
    for source in sorted(source_root.rglob("*.py")):
        relative = source.relative_to(source_root)
        destination = frozen_source / relative
        _copy_frozen(source, destination)
        copied.append({
            "path": f"src/{relative.as_posix()}",
            "sha256": sha256_file(source),
            "bytes": source.stat().st_size,
        })
    for name in (
        "CLEANROOM.md", "SOURCE_BOUNDARY.md", "PREREGISTRATION.md", "README.md",
        "pyproject.toml", "requirements-lock.txt",
        "scripts/run-campaign-detached.sh", "scripts/campaign-status.sh",
    ):
        source = PROJECT_DIR / name
        _copy_frozen(source, run / "provenance" / name)
        copied.append({
            "path": name,
            "sha256": sha256_file(source),
            "bytes": source.stat().st_size,
        })
    copied.append({
        "path": f"protocols/{registration.protocol_path.name}",
        "sha256": sha256_file(registration.protocol_path),
        "bytes": registration.protocol_path.stat().st_size,
    })
    source_digest = sha256()
    for row in sorted(copied, key=lambda value: value["path"]):
        source_digest.update(canonical_json(row))
    snapshot = {
        "format": "wagner-memory-source-snapshot-v2",
        "files": sorted(copied, key=lambda value: value["path"]),
        "snapshot_sha256": source_digest.hexdigest(),
        "frozen_at": utc_now(),
    }
    snapshot_path = run / "provenance" / "source-snapshot.json"
    if snapshot_path.exists():
        existing = json.loads(snapshot_path.read_text())
        # Creation timestamps do not define the source identity.
        if existing.get("files") != snapshot["files"] or existing.get("snapshot_sha256") != snapshot["snapshot_sha256"]:
            raise RuntimeError("run source snapshot differs from current v2 source")
        snapshot = existing
    else:
        atomic_json(snapshot_path, snapshot)

    payload = {
        "format": "wagner-memory-registration-v2",
        "campaign_id": registration.protocol["campaign_id"],
        "profile": registration.profile_name,
        "profile_values": registration.profile,
        "scientific": registration.scientific,
        "protocol_digest": registration.protocol_digest,
        "frozen_protocol_path": frozen_protocol.relative_to(run).as_posix(),
        "protocol": registration.protocol,
        "source_snapshot_sha256": snapshot["snapshot_sha256"],
        "registered_at": utc_now(),
    }
    registration_path = run / "registration.json"
    if registration_path.exists():
        existing = json.loads(registration_path.read_text())
        comparison_keys = set(payload) - {"registered_at"}
        if any(existing.get(key) != payload.get(key) for key in comparison_keys):
            raise RuntimeError("run registration differs from the frozen v2 contract")
        payload = existing
    else:
        atomic_json(registration_path, payload)
    # A read-back through the worker-only loader detects default-protocol drift.
    loaded = load_run_registration(run)
    if loaded.protocol_digest != registration.protocol_digest:
        raise RuntimeError("frozen registration read-back failed")
    return payload


def verify_source_snapshot(run_dir: Path) -> dict[str, Any]:
    provenance = (run_dir / "provenance").resolve()
    snapshot = json.loads((provenance / "source-snapshot.json").read_text())
    registration_payload = json.loads((run_dir / "registration.json").read_text())
    rows = list(snapshot.get("files", []))
    expected_paths = [str(row.get("path", "")) for row in rows]
    actual_paths = sorted(
        path.relative_to(provenance).as_posix()
        for path in provenance.rglob("*")
        if path.is_file() and path.name != "source-snapshot.json"
    )
    files_valid = True
    for row in rows:
        path = (provenance / str(row.get("path", ""))).resolve()
        files_valid &= (
            path.is_relative_to(provenance)
            and path.is_file()
            and path.stat().st_size == int(row.get("bytes", -1))
            and sha256_file(path) == str(row.get("sha256", ""))
        )
    digest = sha256()
    for row in sorted(rows, key=lambda value: str(value.get("path", ""))):
        digest.update(canonical_json(row))
    checks = {
        "format": snapshot.get("format") == "wagner-memory-source-snapshot-v2",
        "paths_unique": len(expected_paths) == len(set(expected_paths)),
        "exact_files": actual_paths == sorted(expected_paths),
        "file_hashes_and_sizes": files_valid,
        "snapshot_digest": digest.hexdigest() == snapshot.get("snapshot_sha256"),
        "registration_digest": (
            snapshot.get("snapshot_sha256")
            == registration_payload.get("source_snapshot_sha256")
        ),
    }
    return {
        "checks": checks,
        "snapshot_sha256": snapshot.get("snapshot_sha256"),
        "files": len(rows),
        "valid": all(checks.values()),
    }


def available_devices() -> list[dict[str, Any]]:
    import jax

    return [
        {
            "id": int(device.id),
            "platform": str(device.platform),
            "device_kind": str(device.device_kind),
        }
        for device in jax.devices()
    ]


def require_devices(registration: Registration) -> list[dict[str, Any]]:
    devices = available_devices()
    gpu_devices = [row for row in devices if row["platform"] in {"gpu", "cuda"}]
    required = int(registration.operations["required_gpu_count"])
    if registration.scientific and len(gpu_devices) != required:
        raise RuntimeError(f"scientific profile requires exactly {required} visible GPUs; found {len(gpu_devices)}")
    return devices


def _worker_environment(worker: int, scientific: bool, run_dir: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update({
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "JAX_ENABLE_X64": "true",
        "PYTHONDONTWRITEBYTECODE": "1",
        "WAGNER_PHYSICAL_GPU": str(worker),
        # Every newly spawned benchmark, science, and audit worker imports the
        # immutable source snapshot made at registration.  This prevents edits
        # to the live project tree from mixing revisions inside a long run.
        "PYTHONPATH": str((run_dir / "provenance" / "src").resolve()),
    })
    gpu_requested = (
        scientific
        or environment.get("WAGNER_FORCE_GPU") == "1"
        or environment.get("JAX_PLATFORMS", "").lower() in {"cuda", "gpu"}
    )
    if gpu_requested:
        environment["CUDA_VISIBLE_DEVICES"] = str(worker)
        environment["JAX_PLATFORMS"] = "cuda"
    else:
        environment.setdefault("JAX_PLATFORMS", "cpu")
    return environment


def _worker_command(run_dir: Path, stage: str, worker: int, workers: int, benchmark: bool) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "wagner_memory_cleanroom_v2",
        "worker",
        "--run",
        str(run_dir),
        "--stage",
        stage,
        "--worker-index",
        str(worker),
        "--worker-count",
        str(workers),
    ]
    if benchmark:
        command.append("--benchmark")
    return command


def _terminate_processes(processes: list[tuple[int, subprocess.Popen[bytes], Any]]) -> None:
    for _, process, _ in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline and any(process.poll() is None for _, process, _ in processes):
        time.sleep(0.1)
    for _, process, _ in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def _run_parallel_workers(
    run_dir: Path,
    registration: Registration,
    stage: str,
    *,
    benchmark: bool,
    timeout_seconds: float,
) -> None:
    workers = int(registration.operations["required_gpu_count"])
    log_dir = run_dir / ("benchmark" if benchmark else "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    processes: list[tuple[int, subprocess.Popen[bytes], Any]] = []
    try:
        for worker in range(workers):
            log_path = log_dir / f"{stage}-gpu{worker}.log"
            handle = log_path.open("wb")
            process = subprocess.Popen(
                _worker_command(run_dir, stage, worker, workers, benchmark),
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=_worker_environment(worker, registration.scientific, run_dir),
                start_new_session=True,
            )
            processes.append((worker, process, handle))
        deadline = time.monotonic() + timeout_seconds
        while True:
            failures = [
                f"worker {worker} exited {process.returncode}"
                for worker, process, _ in processes
                if process.poll() is not None and process.returncode != 0
            ]
            if failures:
                raise RuntimeError(f"{stage} failed: {'; '.join(failures)}")
            if all(process.poll() == 0 for _, process, _ in processes):
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(f"{stage} exceeded its registered time allowance")
            time.sleep(1.0)
    finally:
        _terminate_processes(processes)
        for _, _, handle in processes:
            if not handle.closed:
                handle.close()


def run_benchmark(
    run_dir: Path,
    registration: Registration,
    *,
    timeout_seconds: float = 3600.0,
) -> dict[str, Any]:
    registration_payload = freeze_registration(run_dir, registration)
    devices = require_devices(registration)
    benchmark_dir = run_dir / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    existing = list(benchmark_dir.glob("worker-*.json"))
    if existing:
        raise RuntimeError("benchmark outputs already exist; use a fresh run directory")
    _status(run_dir, phase="benchmark", state="running", devices=devices)
    started = time.monotonic()
    _run_parallel_workers(
        run_dir, registration, "all", benchmark=True, timeout_seconds=timeout_seconds
    )
    workers = int(registration.operations["required_gpu_count"])
    manifests = [
        json.loads((benchmark_dir / f"worker-{worker}.json").read_text())
        for worker in range(workers)
    ]
    if not all(
        row.get("source_snapshot_sha256") == registration_payload["source_snapshot_sha256"]
        and Path(str(row.get("running_source_root", ""))).resolve()
        == (run_dir / "provenance" / "src").resolve()
        for row in manifests
    ):
        raise RuntimeError("benchmark worker did not execute the frozen source snapshot")
    snapshot_verification = verify_source_snapshot(run_dir)
    if not snapshot_verification["valid"]:
        raise RuntimeError(
            f"frozen source snapshot changed during benchmark: {snapshot_verification['checks']}"
        )
    stage_rates: dict[str, float] = {}
    raw_science_seconds = 0.0
    for stage in BASE_STAGES:
        slowest = max(float(row["stages"][stage]["seconds_per_source"]) for row in manifests)
        stage_rates[stage] = slowest
        waves = math.ceil(stage_expectation(stage, registration).sources / workers)
        repeat = 2 if stage in {"state", "carrier"} else 1
        raw_science_seconds += slowest * waves * repeat
    benchmark_elapsed = time.monotonic() - started
    margin = float(registration.operations["benchmark_margin"])
    overhead = float(registration.operations["admission_fixed_overhead_minutes"]) * 60.0
    projected_seconds = benchmark_elapsed + raw_science_seconds * margin + overhead
    projected_hours = projected_seconds / 3600.0
    limit = float(registration.operations["admission_hours_with_margin"])
    admitted = (not registration.scientific) or projected_hours <= limit
    result = {
        "format": "wagner-memory-admission-v2",
        "profile": registration.profile_name,
        "scientific": registration.scientific,
        "protocol_digest": registration.protocol_digest,
        "devices": devices,
        "workers": manifests,
        "source_snapshot_verification": snapshot_verification,
        "stage_seconds_per_source_slowest_gpu": stage_rates,
        "audit_multipliers": {"state": 2, "boundary": 1, "slow_mark": 1, "carrier": 2},
        "raw_science_seconds": raw_science_seconds,
        "benchmark_elapsed_seconds": benchmark_elapsed,
        "margin": margin,
        "fixed_overhead_seconds": overhead,
        "projected_hours_with_margin": projected_hours,
        "limit_hours": limit,
        "admitted": admitted,
        "completed_at": utc_now(),
    }
    atomic_json(run_dir / "benchmark.json", result)
    _status(
        run_dir,
        phase="benchmark",
        state="complete",
        admitted=admitted,
        projected_hours=projected_hours,
    )
    return result


def _stage_paths(run_dir: Path, stage: str, suffix: str) -> list[Path]:
    return sorted((run_dir / "stages" / stage).glob(f"worker-*.{suffix}"))


def verify_run(run_dir: Path) -> dict[str, Any]:
    registration = load_run_registration(run_dir)
    registration_payload = json.loads((run_dir / "registration.json").read_text())
    snapshot_verification = verify_source_snapshot(run_dir)
    expected_source_root = (run_dir / "provenance" / "src").resolve()
    workers = int(registration.operations["required_gpu_count"])
    stage_checks: dict[str, Any] = {}
    aggregate_valid = True
    manifests_by_stage: dict[str, list[dict[str, Any]]] = {}
    for stage in CAMPAIGN_STAGES:
        manifest_paths = _stage_paths(run_dir, stage, "manifest.json")
        record_paths = _stage_paths(run_dir, stage, "jsonl.gz")
        record_paths = [path for path in record_paths if ".sources." not in path.name]
        source_paths = _stage_paths(run_dir, stage, "sources.jsonl.gz")
        exact_files = len(manifest_paths) == len(record_paths) == len(source_paths) == workers
        manifests = [json.loads(path.read_text()) for path in manifest_paths]
        manifests_by_stage[stage] = manifests
        expectation = stage_expectation(stage, registration)
        observed_cells = sum(int(row.get("records", 0)) for row in manifests)
        observed_futures = sum(int(row.get("simulated_futures", 0)) for row in manifests)
        observed_sources = sorted(
            source
            for row in manifests
            for source in row.get("contract", {}).get("observed_sources", [])
        )
        checks = {
            "exact_worker_files": exact_files,
            "worker_contracts": len(manifests) == workers and all(row.get("contract", {}).get("valid") for row in manifests),
            "profile": len(manifests) == workers and all(row.get("profile") == registration.profile_name for row in manifests),
            "protocol": len(manifests) == workers and all(row.get("protocol_digest") == registration.protocol_digest for row in manifests),
            "frozen_source": len(manifests) == workers and all(
                row.get("source_snapshot_sha256") == registration_payload["source_snapshot_sha256"]
                and Path(str(row.get("running_source_root", ""))).resolve() == expected_source_root
                for row in manifests
            ),
            "cells": observed_cells == expectation.cells,
            "futures": observed_futures == expectation.futures,
            "source_partition": observed_sources == list(range(expectation.sources)),
        }
        aggregate_valid &= all(checks.values())
        stage_checks[stage] = {
            "checks": checks,
            "expected_cells": expectation.cells,
            "observed_cells": observed_cells,
            "expected_futures": expectation.futures,
            "observed_futures": observed_futures,
            "expected_sources": expectation.sources,
            "observed_sources": len(observed_sources),
        }

    audit_comparisons: dict[str, Any] = {}
    replay_valid = True
    regeneration_valid = True
    for stage in ("state", "carrier"):
        audit = f"{stage}_audit"
        original_records = [
            path for path in _stage_paths(run_dir, stage, "jsonl.gz")
            if ".sources." not in path.name
        ]
        audit_records = [
            path for path in _stage_paths(run_dir, audit, "jsonl.gz")
            if ".sources." not in path.name
        ]
        original_sources = _stage_paths(run_dir, stage, "sources.jsonl.gz")
        audit_sources = _stage_paths(run_dir, audit, "sources.jsonl.gz")
        record_count, record_digest = records_digest(original_records)
        audit_record_count, audit_record_digest = records_digest(audit_records)
        source_count, source_digest = objects_digest(original_sources, "source_id")
        audit_source_count, audit_source_digest = objects_digest(audit_sources, "source_id")
        record_equal = record_count == audit_record_count and record_digest == audit_record_digest
        source_equal = source_count == audit_source_count and source_digest == audit_source_digest
        original_manifests = {int(row["worker"]): row for row in manifests_by_stage[stage]}
        audit_manifests = {int(row["worker"]): row for row in manifests_by_stage[audit]}
        replay_equal = set(original_manifests) == set(audit_manifests) and all(
            original_manifests[worker].get("replay") == audit_manifests[worker].get("replay")
            and original_manifests[worker].get("replay_sha256") == audit_manifests[worker].get("replay_sha256")
            for worker in original_manifests
        )
        regeneration_valid &= record_equal and source_equal
        replay_valid &= replay_equal
        audit_comparisons[stage] = {
            "original_records": record_count,
            "audit_records": audit_record_count,
            "original_record_sha256": record_digest,
            "audit_record_sha256": audit_record_digest,
            "cell_and_ordered_future_digests_equal": record_equal,
            "original_sources": source_count,
            "audit_sources": audit_source_count,
            "original_source_sha256": source_digest,
            "audit_source_sha256": audit_source_digest,
            "source_records_equal": source_equal,
            "registered_future_replay_equal": replay_equal,
        }
    verified = (
        aggregate_valid
        and regeneration_valid
        and replay_valid
        and snapshot_verification["valid"]
    )
    result = {
        "format": "wagner-memory-verification-v2",
        "protocol_digest": registration.protocol_digest,
        "stage_count_checks": stage_checks,
        "audit_comparisons": audit_comparisons,
        "all_stage_counts_valid": aggregate_valid,
        "independent_regeneration_verified": regeneration_valid,
        "registered_future_replay_verified": replay_valid,
        "source_snapshot_verification": snapshot_verification,
        "verified": verified,
        "completed_at": utc_now(),
    }
    atomic_json(run_dir / "verification.json", result)
    return result


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in ("jax", "jaxlib", "numpy", "scipy"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def runtime_manifest(
    registration: Registration,
    devices: list[dict[str, Any]],
    started_at: str,
    elapsed_seconds: float,
    outcome: str,
) -> dict[str, Any]:
    return {
        "format": "wagner-memory-runtime-v2",
        "campaign_id": registration.protocol["campaign_id"],
        "profile": registration.profile_name,
        "scientific": registration.scientific,
        "protocol_digest": registration.protocol_digest,
        "outcome": outcome,
        "started_at": started_at,
        "completed_at": utc_now(),
        "elapsed_seconds": elapsed_seconds,
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": _package_versions(),
        "devices": devices,
        "runtime_controls": {
            name: os.environ.get(name)
            for name in (
                "CUDA_VISIBLE_DEVICES", "JAX_PLATFORMS", "JAX_ENABLE_X64",
                "XLA_PYTHON_CLIENT_PREALLOCATE", "WAGNER_FORCE_GPU",
                "PYTHONDONTWRITEBYTECODE",
                "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            )
        },
    }


def seal_run(run_dir: Path, outcome: str) -> dict[str, Any]:
    entries = sha256_manifest(run_dir)
    root = sha256()
    for entry in entries:
        root.update(canonical_json(entry))
    manifest = {
        "format": "wagner-memory-seal-v2",
        "outcome": outcome,
        "files": entries,
        "manifest_root_sha256": root.hexdigest(),
        "sealed_at": utc_now(),
    }
    atomic_json(run_dir / "manifest.json", manifest)
    lines = [f"{entry['sha256']}  {entry['path']}" for entry in entries]
    (run_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    return manifest


def run_campaign(run_dir: Path, registration: Registration) -> dict[str, Any]:
    started = time.monotonic()
    started_at = utc_now()
    devices: list[dict[str, Any]] = []
    completed: list[str] = []
    hard_seconds = float(registration.operations["hard_deadline_hours"]) * 3600.0
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_term = signal.getsignal(signal.SIGTERM)
    previous_int = signal.getsignal(signal.SIGINT)

    def hard_timeout(_signum, _frame):
        raise TimeoutError("registered 12-hour campaign deadline reached")

    def interrupted(signum, _frame):
        raise InterruptedError(f"campaign received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    if registration.scientific:
        signal.signal(signal.SIGALRM, hard_timeout)
        signal.setitimer(signal.ITIMER_REAL, hard_seconds)
    try:
        freeze_registration(run_dir, registration)
        validation = validate(registration)
        atomic_json(run_dir / "validation.json", validation)
        if not validation["valid"]:
            raise RuntimeError(f"preflight validation failed: {validation['checks']}")
        # Detect a live-tree change during validation before any worker starts.
        freeze_registration(run_dir, registration)
        count_contract = validate_protocol_counts(registration)
        if not count_contract["valid"]:
            raise RuntimeError(f"protocol count contract failed: {count_contract['checks']}")
        atomic_json(run_dir / "count-contract.json", count_contract)
        devices = require_devices(registration)
        free_gib = shutil.disk_usage(run_dir).free / 1024**3
        if registration.scientific and free_gib < float(registration.operations["minimum_free_gib"]):
            raise RuntimeError(f"free disk {free_gib:.1f} GiB is below the registered guard")
        _status(
            run_dir,
            phase="admission",
            state="running",
            profile=registration.profile_name,
            protocol_digest=registration.protocol_digest,
            devices=devices,
            free_gib=free_gib,
            completed_stages=completed,
        )
        benchmark_timeout = min(3600.0, max(60.0, hard_seconds - (time.monotonic() - started)))
        benchmark = run_benchmark(run_dir, registration, timeout_seconds=benchmark_timeout)
        if not benchmark["admitted"]:
            raise RuntimeError(
                f"admission rejected projected {benchmark['projected_hours_with_margin']:.2f} h"
            )
        stop_new = float(registration.operations["stop_new_shards_hours"]) * 3600.0
        seal_start = float(registration.operations["seal_hours"]) * 3600.0
        for stage in CAMPAIGN_STAGES:
            elapsed = time.monotonic() - started
            if registration.scientific and elapsed >= stop_new:
                raise TimeoutError("registered stop-new-stage boundary reached")
            allowance = seal_start - elapsed if registration.scientific else hard_seconds - elapsed
            if allowance <= 0:
                raise TimeoutError("registered sealing-start boundary reached")
            _status(
                run_dir,
                phase=stage,
                state="running",
                completed_stages=completed,
                elapsed_seconds=elapsed,
            )
            _run_parallel_workers(
                run_dir,
                registration,
                stage,
                benchmark=False,
                timeout_seconds=max(1.0, allowance),
            )
            completed.append(stage)
            _status(
                run_dir,
                phase=stage,
                state="complete",
                completed_stages=completed,
                elapsed_seconds=time.monotonic() - started,
            )
        elapsed = time.monotonic() - started
        if registration.scientific and elapsed > seal_start:
            raise TimeoutError("scientific stages crossed the registered sealing-start boundary")
        _status(
            run_dir,
            phase="sealing",
            state="running",
            completed_stages=completed,
            elapsed_seconds=elapsed,
        )
        verification = verify_run(run_dir)
        if not verification["verified"]:
            raise RuntimeError("independent regeneration, replay, or count verification failed")
        analysis = analyze_all(registration, run_dir)
        analysis_dir = run_dir / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        for key in ("state", "boundary", "slow_mark", "carrier"):
            atomic_json(analysis_dir / f"{key}.json", analysis[key])
        atomic_json(analysis_dir / "summary.json", analysis)
        write_reports(run_dir, analysis, verification)
        elapsed = time.monotonic() - started
        atomic_json(
            run_dir / "runtime_manifest.json",
            runtime_manifest(registration, devices, started_at, elapsed, "complete"),
        )
        _status(
            run_dir,
            phase="sealed",
            state="complete",
            completed_stages=completed,
            elapsed_seconds=elapsed,
            verdict=analysis["overall_verdict"],
            verified=True,
        )
        seal_run(run_dir, "complete")
        return analysis
    except BaseException as exc:
        elapsed = time.monotonic() - started
        try:
            _status(
                run_dir,
                phase="failed",
                state="failed",
                completed_stages=completed,
                elapsed_seconds=elapsed,
                error=f"{type(exc).__name__}: {exc}",
            )
            if (run_dir / "registration.json").exists():
                atomic_json(
                    run_dir / "runtime_manifest.json",
                    runtime_manifest(registration, devices, started_at, elapsed, "failed"),
                )
                seal_run(run_dir, "failed-incomplete")
        finally:
            raise
    finally:
        if registration.scientific:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, previous_handler)
        signal.signal(signal.SIGTERM, previous_term)
        signal.signal(signal.SIGINT, previous_int)
