from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .cohort import simulate_one
from .runtime import require_gpu
from .storage import write_json_atomic


def benchmark_worker(output: str | Path, tier: str, protocol: dict[str, Any]) -> dict[str, Any]:
    devices = require_gpu(expected_visible=1)
    count = int(protocol["operations"]["benchmark_networks_per_tier"])
    threshold = 0.95 if tier == "continuous" else 0.65
    timings: list[float] = []
    digests: list[str] = []
    retained: list[dict[str, np.ndarray]] = []
    for index in range(count):
        started = time.perf_counter()
        result = simulate_one(protocol, tier, "benchmark", index, threshold)
        # Force host materialization and retain a compact proof that the work ran.
        timings.append(time.perf_counter() - started)
        digests.append(str(result["trajectory_digest"][0]))
        if index < 2:
            retained.append(result)
    replay_timings: list[float] = []
    for index, primary in enumerate(retained):
        started = time.perf_counter()
        replay = simulate_one(protocol, tier, "benchmark", index, threshold, executor="loop")
        replay_timings.append(time.perf_counter() - started)
        if not np.array_equal(primary["events"], replay["events"]):
            raise RuntimeError(f"{tier} benchmark replay changed registered events")
        if not np.allclose(primary["endpoints"], replay["endpoints"], rtol=1e-6, atol=1e-6):
            raise RuntimeError(f"{tier} benchmark replay changed endpoints")
        if not np.array_equal(primary["trajectory_digest"], replay["trajectory_digest"]):
            raise RuntimeError(f"{tier} benchmark replay changed trace digests")
    report = {
        "format": "grn-f12-benchmark-worker-v1", "tier": tier, "devices": devices,
        "networks": count, "seconds": timings, "total_seconds": float(sum(timings)),
        "median_steady_seconds": float(np.median(timings[1:] if len(timings) > 1 else timings)),
        "replay_networks": len(replay_timings), "replay_seconds": replay_timings,
        "median_replay_seconds": float(np.median(replay_timings[1:] if len(replay_timings) > 1 else replay_timings)),
        "proof_digests": digests,
    }
    write_json_atomic(output, report)
    return report


def project_campaign(continuous: dict[str, Any], molecular: dict[str, Any], protocol: dict[str, Any], benchmark_elapsed: float) -> dict[str, Any]:
    gpu_seconds = 0.0
    compilation_seconds = 0.0
    details: dict[str, Any] = {}
    for tier, report in (("continuous", continuous), ("molecular", molecular)):
        cfg = protocol["tiers"][tier]
        seconds_per_observational_network = float(report["median_steady_seconds"])
        seconds_per_replay_network = float(report["median_replay_seconds"])
        calibration_equivalent = (
            int(cfg["calibration_networks"]) * int(cfg["calibration_futures"])
            / (int(cfg["futures"]) * 20.0)
        )
        observational_equivalent = int(cfg["development_networks"]) + int(cfg["confirmation_networks"])
        control_equivalent = 0.5 * int(cfg["control_networks"])
        equivalents = calibration_equivalent + observational_equivalent + control_equivalent
        replay_networks = int(cfg["confirmation_networks"])
        tier_seconds = (
            equivalents * seconds_per_observational_network
            + replay_networks * seconds_per_replay_network
        )
        scan_compile = max(float(report["seconds"][0]) - seconds_per_observational_network, 0.0)
        replay_compile = max(float(report["replay_seconds"][0]) - seconds_per_replay_network, 0.0)
        # Fresh processes are used at each scientific stage. Four scan-shaped
        # stages plus the independent replay receive an explicit compile budget.
        tier_compilation = 4.0 * scan_compile + replay_compile
        compilation_seconds += tier_compilation
        gpu_seconds += tier_seconds
        details[tier] = {
            "seconds_per_observational_network": seconds_per_observational_network,
            "seconds_per_replay_network": seconds_per_replay_network,
            "equivalent_observational_networks": equivalents,
            "replay_networks": replay_networks,
            "projected_gpu_seconds": tier_seconds,
            "projected_compilation_seconds": tier_compilation,
        }
    fixed_cpu_and_training_seconds = 1200.0
    projected_wall = benchmark_elapsed + (gpu_seconds + compilation_seconds) / 2.0 + fixed_cpu_and_training_seconds
    margin = float(protocol["operations"]["benchmark_margin"])
    guarded = projected_wall * margin
    limit = float(protocol["operations"]["admission_hours"]) * 3600.0
    return {
        "format": "grn-f12-benchmark-v1", "tiers": details,
        "benchmark_elapsed_seconds": benchmark_elapsed,
        "fixed_cpu_and_training_seconds": fixed_cpu_and_training_seconds,
        "projected_compilation_gpu_seconds": compilation_seconds,
        "projected_wall_seconds": projected_wall, "margin": margin,
        "guarded_wall_seconds": guarded, "admission_limit_seconds": limit,
        "admitted": bool(guarded <= limit),
    }


def run_benchmark(run_dir: str | Path, protocol: dict[str, Any]) -> dict[str, Any]:
    root = Path(run_dir)
    target = root / "benchmark"
    target.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    processes = []
    handles = []
    for device, tier in enumerate(("continuous", "molecular")):
        output = target / f"{tier}.json"
        log_path = target / f"{tier}.log"
        handle = log_path.open("a", encoding="utf-8")
        handles.append(handle)
        environment = os.environ.copy()
        environment.update(
            CUDA_VISIBLE_DEVICES=str(device), JAX_PLATFORMS="cuda",
            XLA_PYTHON_CLIENT_PREALLOCATE="false", OMP_NUM_THREADS="1",
        )
        processes.append(subprocess.Popen(
            [sys.executable, "-m", "grn_f12_realistic", "benchmark-worker", "--run", str(root), "--tier", tier],
            stdout=handle, stderr=subprocess.STDOUT, env=environment,
        ))
    codes = [process.wait() for process in processes]
    for handle in handles:
        handle.close()
    if any(code != 0 for code in codes):
        raise RuntimeError(f"benchmark workers failed with codes {codes}")
    import json

    reports = {
        tier: json.loads((target / f"{tier}.json").read_text(encoding="utf-8"))
        for tier in ("continuous", "molecular")
    }
    result = project_campaign(reports["continuous"], reports["molecular"], protocol, time.perf_counter() - started)
    write_json_atomic(root / "benchmark.json", result)
    return result
