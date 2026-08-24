from __future__ import annotations

import json
import os
import platform
import shutil
import sys
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import sklearn

from .analysis import analyze_predictor, analyze_primary
from .experiment import run_primary, sample_sources
from .predictor import run_predictor_cohort, sample_predictor_sources
from .protocol import digest, load_protocol, write_json_atomic
from .storage import seal_directory, update_status
from .storage import sha256_file
from .verification import format_primary_future_id, replay_primary, validate_environment, verify_run


WALL_LIMIT_SECONDS = 12 * 60 * 60
SOFT_LIMIT_SECONDS = int(11.5 * 60 * 60)
ADMISSION_LIMIT_SECONDS = int(11.25 * 60 * 60)
GIB = 1024**3


def _new_attempt(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    existing = [int(path.name.split("-")[-1]) for path in root.glob("attempt-*") if path.name.split("-")[-1].isdigit()]
    attempt = max(existing, default=0) + 1
    path = root / f"attempt-{attempt:03d}"
    path.mkdir()
    return path


def benchmark(output: Path, workers: int, profile: str = "full") -> dict[str, Any]:
    attempt = _new_attempt(output)
    primary = deepcopy(load_protocol("primary", profile))
    primary.update({
        "scientific": False, "profile": "discarded-benchmark", "source_count": 4,
        "master_seed_label": str(primary["master_seed_label"]) + "|discarded-benchmark",
        "bootstrap_repetitions": 64,
    })
    primary_dir = attempt / "primary"
    sample_started = time.monotonic()
    sample_sources(primary_dir, primary)
    primary_sampling_seconds = time.monotonic() - sample_started
    simulation_started = time.monotonic()
    primary_result = run_primary(primary_dir, primary, workers=min(workers, 4))
    primary_simulation_seconds = time.monotonic() - simulation_started

    predictor = deepcopy(load_protocol("predictor", profile))
    predictor.update({
        "scientific": False, "profile": "discarded-benchmark",
        "development_sources": 4, "histories_per_source": 5, "futures_per_state": 128,
        "development_seed_label": str(predictor["development_seed_label"]) + "|discarded-benchmark",
        "bootstrap_repetitions": 64,
    })
    predictor_dir = attempt / "predictor"
    pred_sample_started = time.monotonic()
    sample_predictor_sources(predictor_dir, predictor, "development")
    predictor_sampling_seconds = time.monotonic() - pred_sample_started
    pred_sim_started = time.monotonic()
    predictor_result = run_predictor_cohort(predictor_dir, predictor, "development", workers=min(workers, 4))
    predictor_simulation_seconds = time.monotonic() - pred_sim_started

    primary_source_seconds = float(primary_result.get("median_source_seconds") or primary_simulation_seconds)
    predictor_source_seconds = float(predictor_result.get("median_source_seconds") or predictor_simulation_seconds)
    projected_components = {
        "primary_sampling": primary_sampling_seconds / 4 * 240,
        "primary_simulation": primary_source_seconds * np.ceil(240 / workers),
        "predictor_sampling": predictor_sampling_seconds / 4 * (96 + 128),
        "predictor_simulation": predictor_source_seconds * np.ceil((96 + 128) / workers),
        "analysis_replay_sealing_allowance": 2 * 60 * 60,
    }
    projected = float(sum(projected_components.values()))
    adjusted = projected * 1.2
    admitted = profile == "smoke" or adjusted <= ADMISSION_LIMIT_SECONDS
    result = {
        "format": "wagner-cleanroom-benchmark-v1",
        "attempt": attempt.name,
        "workers": workers,
        "primary_sampling_seconds": primary_sampling_seconds,
        "primary_simulation_seconds": primary_simulation_seconds,
        "predictor_sampling_seconds": predictor_sampling_seconds,
        "predictor_simulation_seconds": predictor_simulation_seconds,
        "projected_components_seconds": projected_components,
        "projected_seconds": projected,
        "projected_with_20_percent_margin_seconds": adjusted,
        "admission_limit_seconds": ADMISSION_LIMIT_SECONDS,
        "admitted": admitted,
    }
    write_json_atomic(attempt / "benchmark.json", result)
    return result


def _runtime_manifest(started: str, elapsed: float, workers: int, profile: str) -> dict[str, Any]:
    usage = shutil.disk_usage(Path.cwd())
    return {
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
        "wall_limit_seconds": WALL_LIMIT_SECONDS,
        "workers": workers,
        "profile": profile,
        "python": sys.version,
        "platform": platform.platform(),
        "logical_cpus": os.cpu_count(),
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
        "disk_free_bytes_at_finish": usage.free,
    }


def _source_tree_manifest() -> dict[str, str]:
    project_root = Path(__file__).resolve().parents[2]
    paths = [project_root / "pyproject.toml", project_root / "PREREGISTRATION.md"]
    paths.extend(sorted((project_root / "src").rglob("*.py")))
    paths.extend(sorted((project_root / "protocols").glob("*.json")))
    paths.extend(sorted((project_root / "scripts").glob("*.sh")))
    return {str(path.relative_to(project_root)): sha256_file(path) for path in paths}


def _require_resources(root: Path, workers: int, minimum_free_gib: int) -> None:
    if workers < 1 or workers > 12:
        raise RuntimeError("workers must be between 1 and the 12 physical cores")
    free = shutil.disk_usage(root).free
    if free < minimum_free_gib * GIB:
        raise RuntimeError(f"only {free / GIB:.1f} GiB free; {minimum_free_gib} GiB required")


def _lay_summary(primary: dict[str, Any], predictor: dict[str, Any]) -> str:
    pm = primary["metrics"]
    return "\n\n".join([
        "This clean-room experiment asked whether a Wagner gene-regulation network can carry useful information from its present expression state into its descendants. We rebuilt the simulator independently, sampled fresh regulatory networks, and fixed all tests before examining their results. The exact-state replication finished with a **%s** verdict across %d independently sampled networks." % (primary["verdict"], primary["sources"]),
        "Writing one of two possible adult states into a network changed what its descendants later became. Compared with resetting the expression state, the written state changed matching-destination risk by %.3f, while the history crossover was %.3f. The current state also improved probabilistic destination prediction by %.3f nats of log loss. These effects were judged together with persistence, shuffled-state controls, first-generation effects, split-half reliability, and exact replay—not from a single favorable number." % (pm["risk_gain"]["estimate"], pm["crossover"]["estimate"], pm["log_loss_gain"]["estimate"]),
        "A separate extension asked a closer version of the preprint's prediction question: given the recent lineage history, present gene-expression pattern, and regulatory matrix, can we predict a future break followed by three faithfully inherited boundaries? Its sealed exploratory verdict was **%s**. The richer model's log-loss advantage over history alone was %.3f nats, and over a strong basin-and-sensitivity baseline it was %.3f nats." % (predictor["verdict"], predictor["history_log_loss_gain"]["estimate"], predictor["structural_log_loss_gain"]["estimate"]),
        "The result is evidence about an exact expression-state channel in Wagner networks. It does not establish that such a state arises naturally, survives a complete expression reset, or constitutes a hidden molecular carrier. The PH predictor extension is deliberately labelled exploratory and cannot rescue or alter the primary replication verdict.",
    ]) + "\n"


def run_campaign(root: Path, workers: int = 12, profile: str = "full") -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    _require_resources(root, workers, 40)
    validation = validate_environment()
    if not validation["ok"]:
        raise RuntimeError("environment validation failed: " + "; ".join(validation["failures"]))
    primary_protocol = load_protocol("primary", profile)
    predictor_protocol = load_protocol("predictor", profile)
    campaign_registration = {
        "format": "wagner-cleanroom-campaign-v1",
        "profile": profile,
        "primary_digest": digest(primary_protocol),
        "predictor_digest": digest(predictor_protocol),
        "source_tree": _source_tree_manifest(),
    }
    registration_path = root / "campaign_registration.json"
    if registration_path.exists():
        existing = json.loads(registration_path.read_text(encoding="utf-8"))
        if existing != campaign_registration:
            raise RuntimeError("campaign registration mismatch")
    else:
        write_json_atomic(registration_path, campaign_registration)
    started_wall = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    soft_deadline = started_wall + SOFT_LIMIT_SECONDS
    hard_deadline = started_wall + WALL_LIMIT_SECONDS
    update_status(root, phase="benchmark", complete=False, started_utc=started_utc)
    bench = benchmark(root / "benchmark", workers, profile)
    write_json_atomic(root / "benchmark.json", bench)
    if not bench["admitted"]:
        result = {"complete": False, "phase": "admission-blocked", "benchmark": bench}
        update_status(root, **result)
        return result
    if time.monotonic() >= hard_deadline:
        raise TimeoutError("campaign exhausted its wall limit during benchmark")

    update_status(root, phase="primary")
    primary_summary = run_primary(root / "primary", primary_protocol, workers, soft_deadline)
    _require_resources(root, workers, 30)
    if not primary_summary["complete"]:
        result = {"complete": False, "phase": "primary-checkpointed", "primary": primary_summary}
        update_status(root, **result)
        return result
    update_status(root, phase="predictor-development")
    development = run_predictor_cohort(root / "predictor" / "development", predictor_protocol, "development", workers, soft_deadline)
    _require_resources(root, workers, 30)
    if not development["complete"]:
        result = {"complete": False, "phase": "predictor-development-checkpointed", "development": development}
        update_status(root, **result)
        return result
    update_status(root, phase="predictor-evaluation")
    evaluation = run_predictor_cohort(root / "predictor" / "evaluation", predictor_protocol, "evaluation", workers, soft_deadline)
    _require_resources(root, workers, 30)
    if not evaluation["complete"]:
        result = {"complete": False, "phase": "predictor-evaluation-checkpointed", "evaluation": evaluation}
        update_status(root, **result)
        return result
    if time.monotonic() >= hard_deadline:
        raise TimeoutError("campaign exhausted its wall limit before analysis")

    update_status(root, phase="analysis")
    primary_analysis = analyze_primary(root / "primary", primary_protocol)
    predictor_analysis = analyze_predictor(root / "predictor", predictor_protocol)
    (root / "LAY_SUMMARY.md").write_text(_lay_summary(primary_analysis, predictor_analysis), encoding="utf-8")
    primary_futures = int(primary_protocol["conditions"][0]["futures"])
    recurrence_futures = int(primary_protocol["conditions"][1]["futures"])
    persistence_futures = int(primary_protocol["conditions"][2]["futures"])
    source_count = int(primary_protocol["source_count"])
    replay_ids = [
        format_primary_future_id(0, 0, 1, 0, 0, 1, 0, 0),
        format_primary_future_id(source_count // 2, 1, 1, 1, 1, 2, 0, recurrence_futures - 1),
        format_primary_future_id(source_count - 1, 2, 5, 1, 1, 0, 8, persistence_futures - 1),
    ]
    replays = [replay_primary(root, future_id) for future_id in replay_ids]
    replay_summary = {"all_exact": all(item["exact"] for item in replays), "replays": replays}
    write_json_atomic(root / "replay_summary.json", replay_summary)
    verification = verify_run(root)
    elapsed = time.monotonic() - started_wall
    complete = bool(verification["ok"] and replay_summary["all_exact"] and elapsed <= WALL_LIMIT_SECONDS)
    result = {
        "complete": complete,
        "phase": "complete" if complete else "verification-failed",
        "elapsed_seconds": elapsed,
        "primary_verdict": primary_analysis["verdict"],
        "predictor_verdict": predictor_analysis["verdict"],
        "verification": verification,
    }
    update_status(root, **result)
    write_json_atomic(root / "runtime_manifest.json", _runtime_manifest(started_utc, elapsed, workers, profile))
    seal_directory(root)
    return result
