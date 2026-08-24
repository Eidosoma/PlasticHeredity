from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path
from typing import Any

from .cohort import audit_one, calibrate_one, simulate_one
from .config import cohort_size
from .controls import run_controls_tier
from .dataset import network_paths
from .predictor import predict_confirmation, train_models
from .runtime import require_gpu
from .storage import free_gib, load_npz, write_json_atomic, write_npz_atomic


def task_specifications(stage: str, protocol: dict[str, Any]) -> list[dict[str, Any]]:
    shard_size = int(protocol["operations"]["shard_networks"])
    tasks: list[dict[str, Any]] = []
    if stage in {"calibration", "development", "confirmation", "audit"}:
        cohort = "confirmation" if stage == "audit" else stage
        for tier in ("continuous", "molecular"):
            count = cohort_size(protocol, tier, cohort)
            for start in range(0, count, shard_size):
                stop = min(start + shard_size, count)
                tasks.append({
                    "id": f"{stage}-{tier}-{start:04d}-{stop:04d}", "stage": stage,
                    "tier": tier, "start": start, "stop": stop,
                })
    elif stage in {"training", "prediction", "controls"}:
        tasks = [{"id": f"{stage}-{tier}", "stage": stage, "tier": tier} for tier in ("continuous", "molecular")]
    else:
        raise ValueError(stage)
    return tasks


def prepare_queue(run_dir: str | Path, stage: str, protocol: dict[str, Any]) -> Path:
    queue = Path(run_dir) / "queues" / stage
    queue.mkdir(parents=True, exist_ok=True)
    expected = task_specifications(stage, protocol)
    manifest = queue / "queue.json"
    value = {"format": "grn-f12-task-queue-v1", "stage": stage, "tasks": expected}
    if manifest.exists():
        if json.loads(manifest.read_text(encoding="utf-8")) != value:
            raise RuntimeError(f"existing {stage} queue differs from registration")
    else:
        write_json_atomic(manifest, value)
    for task in expected:
        path = queue / f"{task['id']}.task.json"
        if not path.exists():
            write_json_atomic(path, task)
    return queue


def queue_status(run_dir: str | Path, stage: str) -> dict[str, int]:
    queue = Path(run_dir) / "queues" / stage
    tasks = list(queue.glob("*.task.json"))
    done = list(queue.glob("*.done.json"))
    failed = list(queue.glob("*.failed.json"))
    locked = list(queue.glob("*.lock"))
    return {"tasks": len(tasks), "done": len(done), "failed": len(failed), "locked": len(locked)}


def queue_complete(run_dir: str | Path, stage: str) -> bool:
    status = queue_status(run_dir, stage)
    return status["tasks"] > 0 and status["done"] == status["tasks"]


def _owner_alive(lock: Path) -> bool:
    try:
        owner = json.loads(lock.read_text(encoding="utf-8"))
        if owner.get("host") != socket.gethostname():
            return True
        os.kill(int(owner["pid"]), 0)
        return True
    except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError, ProcessLookupError):
        return False
    except PermissionError:
        return True


def _claim(queue: Path, worker_id: int) -> tuple[dict[str, Any], Path] | None:
    preferred_tier = "continuous" if worker_id % 2 == 0 else "molecular"
    candidates = []
    for task_path in queue.glob("*.task.json"):
        task_value = json.loads(task_path.read_text(encoding="utf-8"))
        candidates.append((task_path, task_value))
    candidates.sort(key=lambda item: (
        item[1].get("tier") != preferred_tier,
        int(item[1].get("start", 0)),
        str(item[1]["id"]),
    ))
    for task_path, task in candidates:
        done = queue / f"{task['id']}.done.json"
        if done.exists():
            continue
        lock = queue / f"{task['id']}.lock"
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            if not _owner_alive(lock):
                try:
                    lock.unlink()
                except FileNotFoundError:
                    pass
            continue
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid(), "host": socket.gethostname(), "worker": worker_id, "claimed_at": time.time()}, handle)
        return task, lock
    return None


def _threshold(run_dir: Path, tier: str) -> float:
    value = json.loads((run_dir / "calibration" / f"{tier}.json").read_text(encoding="utf-8"))
    return float(value["thresholds"]["q05"])


def _thresholds(run_dir: Path, tier: str) -> dict[str, float]:
    value = json.loads((run_dir / "calibration" / f"{tier}.json").read_text(encoding="utf-8"))
    return {name: float(threshold) for name, threshold in value["thresholds"].items()}


def execute_task(run_dir: Path, task: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    stage, tier = task["stage"], task["tier"]
    started = time.time()
    if stage == "calibration":
        target = run_dir / "data" / "calibration" / tier
        for index in range(int(task["start"]), int(task["stop"])):
            path = target / f"network_{index:04d}.npz"
            if not path.exists():
                write_npz_atomic(path, **calibrate_one(protocol, tier, index))
    elif stage in {"development", "confirmation"}:
        target = run_dir / "data" / stage / tier
        thresholds = _thresholds(run_dir, tier)
        threshold = thresholds["q05"]
        for index in range(int(task["start"]), int(task["stop"])):
            path = target / f"network_{index:04d}.npz"
            if not path.exists():
                write_npz_atomic(
                    path, **simulate_one(
                        protocol, tier, stage, index, threshold,
                        sensitivity_thresholds={"q025": thresholds["q025"], "q10": thresholds["q10"]},
                    )
                )
    elif stage == "audit":
        thresholds = _thresholds(run_dir, tier)
        threshold = thresholds["q05"]
        target = run_dir / "audit" / tier
        for index in range(int(task["start"]), int(task["stop"])):
            report_path = target / f"network_{index:04d}.json"
            if not report_path.exists():
                stored = load_npz(run_dir / "data" / "confirmation" / tier / f"network_{index:04d}.npz")
                write_json_atomic(report_path, audit_one(
                    protocol, tier, index, threshold, stored,
                    {"q025": thresholds["q025"], "q10": thresholds["q10"]},
                ))
    elif stage == "training":
        train_models(run_dir, tier, protocol)
    elif stage == "prediction":
        predict_confirmation(run_dir, tier, protocol)
    elif stage == "controls":
        run_controls_tier(run_dir, tier, protocol)
    else:
        raise ValueError(stage)
    return {"task": task["id"], "seconds": time.time() - started, "completed_at": time.time()}


def worker_loop(
    run_dir: str | Path,
    stage: str,
    worker_id: int,
    protocol: dict[str, Any],
    stop_new_epoch: float | None = None,
) -> int:
    require_gpu(expected_visible=1)
    root = Path(run_dir)
    queue = prepare_queue(root, stage, protocol)
    while True:
        if queue_complete(root, stage):
            return 0
        if stop_new_epoch is not None and time.time() >= stop_new_epoch:
            return 3
        if free_gib(root) < float(protocol["operations"]["disk_running_gib"]):
            raise RuntimeError("running disk guard triggered")
        claimed = _claim(queue, worker_id)
        if claimed is None:
            time.sleep(1.0)
            continue
        task, lock = claimed
        try:
            result = execute_task(root, task, protocol)
            write_json_atomic(queue / f"{task['id']}.done.json", result)
            failed = queue / f"{task['id']}.failed.json"
            if failed.exists():
                failed.unlink()
        except Exception as error:
            write_json_atomic(queue / f"{task['id']}.failed.json", {
                "task": task["id"], "worker": worker_id, "error": repr(error), "time": time.time(),
            })
            raise
        finally:
            try:
                lock.unlink()
            except FileNotFoundError:
                pass
