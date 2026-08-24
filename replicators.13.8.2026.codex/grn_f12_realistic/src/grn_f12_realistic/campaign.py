from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .analysis import analyze_campaign
from .benchmark import run_benchmark
from .cohort import combine_calibration
from .config import PROJECT_ROOT, cohort_size
from .controls import combine_controls
from .runtime import environment_manifest, require_gpu
from .storage import (
    ensure_registration, free_gib, json_digest, load_npz, seal_run, sha256_file,
    source_manifest, update_status, verify_run, write_json_atomic,
)
from .taskqueue import prepare_queue, queue_complete, queue_status


STAGES = ("calibration", "development", "training", "confirmation", "audit", "prediction", "controls")


def registration(protocol: dict[str, Any], profile: str) -> dict[str, Any]:
    sources = source_manifest(PROJECT_ROOT)
    return {
        "format": "grn-f12-registration-v1", "profile": profile,
        "scientific": bool(protocol.get("scientific", False) and profile == "full"),
        "protocol": protocol, "protocol_sha256": json_digest(protocol),
        "source_manifest_sha256": json_digest(sources), "source_manifest": sources,
        "created_at": time.time(),
    }


def _run_queue(run_dir: Path, stage: str, protocol: dict[str, Any], stop_new_epoch: float, hard_epoch: float) -> None:
    prepare_queue(run_dir, stage, protocol)
    if queue_complete(run_dir, stage):
        return
    processes: list[subprocess.Popen] = []
    handles = []
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    for worker_id in range(2):
        handle = (log_dir / f"{stage}-gpu{worker_id}.log").open("a", encoding="utf-8")
        handles.append(handle)
        environment = os.environ.copy()
        environment.update(
            CUDA_VISIBLE_DEVICES=str(worker_id), JAX_PLATFORMS="cuda",
            XLA_PYTHON_CLIENT_PREALLOCATE="false", OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
        )
        command = [
            sys.executable, "-m", "grn_f12_realistic", "worker", "--run", str(run_dir),
            "--stage", stage, "--worker-id", str(worker_id), "--stop-new-epoch", str(stop_new_epoch),
        ]
        processes.append(subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, env=environment))
    sealed = False
    try:
        last_update = 0.0
        while any(process.poll() is None for process in processes):
            now = time.time()
            if now >= hard_epoch:
                for process in processes:
                    if process.poll() is None:
                        process.terminate()
                raise TimeoutError("12-hour hard campaign limit reached")
            if now - last_update >= 15.0:
                update_status(run_dir, phase=stage, queue=queue_status(run_dir, stage), updated_at=now)
                last_update = now
            time.sleep(2.0)
        codes = [int(process.returncode or 0) for process in processes]
        if not queue_complete(run_dir, stage):
            raise RuntimeError(f"{stage} queue incomplete; worker codes {codes}")
        if any(code not in (0, 3) for code in codes):
            raise RuntimeError(f"{stage} workers failed with codes {codes}")
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
            process.wait()
        for handle in handles:
            handle.close()


def _collate_calibration(run_dir: Path, protocol: dict[str, Any]) -> None:
    for tier in ("continuous", "molecular"):
        paths = sorted((run_dir / "data" / "calibration" / tier).glob("network_*.npz"))
        expected = cohort_size(protocol, tier, "calibration")
        if len(paths) != expected:
            raise RuntimeError(f"calibration incomplete for {tier}: {len(paths)}/{expected}")
        result = combine_calibration([load_npz(path) for path in paths])
        result.update(format="grn-f12-calibration-v1", tier=tier)
        write_json_atomic(run_dir / "calibration" / f"{tier}.json", result)


def _collate_audit(run_dir: Path, protocol: dict[str, Any]) -> None:
    for tier in ("continuous", "molecular"):
        paths = sorted((run_dir / "audit" / tier).glob("network_*.json"))
        expected = cohort_size(protocol, tier, "confirmation")
        reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        result = {
            "format": "grn-f12-independent-regeneration-v1", "tier": tier,
            "networks": len(reports), "expected_networks": expected,
            "complete": len(reports) == expected,
            "failures": [report for report in reports if not report.get("pass")],
        }
        result["pass"] = bool(result["complete"] and not result["failures"])
        write_json_atomic(run_dir / "audit" / f"{tier}.json", result)


def replay_analysis(run_dir: str | Path, protocol: dict[str, Any]) -> dict[str, Any]:
    root = Path(run_dir)
    targets = [
        root / "analysis" / "continuous.json", root / "analysis" / "molecular.json",
        root / "analysis" / "summary.json", root / "REPORT.md", root / "LAY_SUMMARY.md",
    ]
    before = {str(path.relative_to(root)): sha256_file(path) for path in targets}
    analyze_campaign(root, protocol)
    after = {str(path.relative_to(root)): sha256_file(path) for path in targets}
    return {"format": "grn-f12-analysis-replay-v1", "identical": before == after, "before": before, "after": after}


def run_campaign(run_dir: str | Path, protocol: dict[str, Any], profile: str) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    require_gpu(expected_visible=int(protocol["operations"]["required_gpus"]))
    if free_gib(root) < float(protocol["operations"]["disk_admission_gib"]):
        raise RuntimeError("admission disk guard triggered")
    registration_value = registration(protocol, profile)
    existing_registration = root / "registration.json"
    if existing_registration.exists():
        frozen = json.loads(existing_registration.read_text(encoding="utf-8"))
        # created_at is part of the first registration and is retained on resume.
        comparison = dict(registration_value)
        comparison["created_at"] = frozen.get("created_at")
        if frozen != comparison:
            raise RuntimeError("cannot resume: protocol or source registration changed")
        registration_value = frozen
    else:
        ensure_registration(root, registration_value)
    started_at = float(registration_value["created_at"])
    stop_new_epoch = started_at + float(protocol["operations"]["stop_new_hours"]) * 3600.0
    hard_epoch = started_at + float(protocol["operations"]["hard_limit_hours"]) * 3600.0
    if time.time() >= hard_epoch:
        raise TimeoutError("registered campaign has exhausted its 12-hour wall-time budget")
    update_status(root, phase="validation", profile=profile, scientific=registration_value["scientific"], started_at=started_at)
    runtime_path = root / "runtime_manifest.json"
    if not runtime_path.exists():
        write_json_atomic(runtime_path, environment_manifest(PROJECT_ROOT))
    benchmark_path = root / "benchmark.json"
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8")) if benchmark_path.exists() else run_benchmark(root, protocol)
    if not benchmark["admitted"]:
        update_status(root, phase="not_admitted", benchmark=benchmark, updated_at=time.time())
        return {"status": "NOT_ADMITTED", "benchmark": benchmark}

    try:
        for stage in STAGES:
            update_status(root, phase=stage, updated_at=time.time())
            _run_queue(root, stage, protocol, stop_new_epoch, hard_epoch)
            if stage == "calibration":
                _collate_calibration(root, protocol)
            elif stage == "audit":
                _collate_audit(root, protocol)
            elif stage == "controls":
                combine_controls(root)
        update_status(root, phase="analysis", updated_at=time.time())
        result = analyze_campaign(root, protocol)
        replay = replay_analysis(root, protocol)
        write_json_atomic(root / "replay_summary.json", replay)
        result["replay_verified"] = bool(replay["identical"])
        if not replay["identical"]:
            result["prediction_verdict"] = "INCOMPLETE"
            result["replay_failure"] = True
        write_json_atomic(root / "FINAL_VERDICT.json", {
            **result,
            "checksum_requirement": "verification.json must report verified=true",
        })
        update_status(
            root, phase="sealing", completed=False, prediction_verdict=result["prediction_verdict"],
            mechanistic_verdict=result["mechanistic_verdict"], ended_at=time.time(),
        )
        seal_run(root)
        sealed = True
        verification = verify_run(root)
        write_json_atomic(root / "verification.json", {
            "format": "grn-f12-verification-v1", **verification,
            "prediction_verdict": result["prediction_verdict"],
            "replay_verified": result["replay_verified"],
        })
        if not verification["verified"]:
            update_status(root, phase="verification_failed", completed=False, error=str(verification), ended_at=time.time())
            raise RuntimeError(f"post-seal verification failed: {verification}")
        result["checksum_verified"] = True
        update_status(
            root, phase="complete", completed=True, prediction_verdict=result["prediction_verdict"],
            mechanistic_verdict=result["mechanistic_verdict"], ended_at=time.time(),
        )
        return result
    except Exception as error:
        phase = "incomplete_deadline" if time.time() >= stop_new_epoch else ("verification_failed" if sealed else "failed")
        update_status(
            root, phase=phase, completed=False, error=repr(error), ended_at=time.time(),
        )
        raise
