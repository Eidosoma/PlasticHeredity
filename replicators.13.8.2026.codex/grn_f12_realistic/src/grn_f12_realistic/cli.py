from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .analysis import analyze_campaign
from .benchmark import benchmark_worker, run_benchmark
from .campaign import _collate_audit, _collate_calibration, _run_queue, registration, replay_analysis, run_campaign
from .config import PROJECT_ROOT, load_protocol, protocol_for_profile
from .controls import combine_controls
from .runtime import environment_manifest, require_gpu
from .storage import ensure_registration, free_gib, json_digest, source_manifest, verify_run
from .taskqueue import queue_status, worker_loop


QUEUE_STAGES = ("calibration", "development", "training", "confirmation", "audit", "prediction", "controls")


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def _registered_protocol(run_dir: Path, profile: str | None = None) -> tuple[dict[str, Any], str]:
    path = run_dir / "registration.json"
    if path.exists():
        value = json.loads(path.read_text(encoding="utf-8"))
        return value["protocol"], str(value["profile"])
    selected = profile or "full"
    protocol = protocol_for_profile(load_protocol(), selected)
    ensure_registration(run_dir, registration(protocol, selected))
    return protocol, selected


def _verify_registered_source(run_dir: Path) -> None:
    frozen = json.loads((run_dir / "registration.json").read_text(encoding="utf-8"))
    current = json_digest(source_manifest(PROJECT_ROOT))
    if current != frozen["source_manifest_sha256"]:
        raise RuntimeError("source tree changed after run registration")


def _run_selected_stages(run_dir: Path, protocol: dict[str, Any], stages: tuple[str, ...]) -> None:
    require_gpu(expected_visible=2)
    started = time.time()
    stop_new = started + float(protocol["operations"]["stop_new_hours"]) * 3600.0
    hard = started + float(protocol["operations"]["hard_limit_hours"]) * 3600.0
    for stage in stages:
        _run_queue(run_dir, stage, protocol, stop_new, hard)
        if stage == "calibration":
            _collate_calibration(run_dir, protocol)
        elif stage == "audit":
            _collate_audit(run_dir, protocol)
        elif stage == "controls":
            combine_controls(run_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="grn-f12")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate protocol, dependencies, disk, and two GPUs")
    validate.add_argument("--profile", choices=("full", "quick", "smoke"), default="full")

    for command in ("benchmark", "calibrate", "develop", "confirm", "controls", "analyze", "verify", "replay", "status", "campaign"):
        child = subparsers.add_parser(command)
        child.add_argument("--run", type=Path, required=True)
        if command in {"benchmark", "campaign"}:
            child.add_argument("--profile", choices=("full", "quick", "smoke"), default="full")

    worker = subparsers.add_parser("worker", help=argparse.SUPPRESS)
    worker.add_argument("--run", type=Path, required=True)
    worker.add_argument("--stage", choices=QUEUE_STAGES, required=True)
    worker.add_argument("--worker-id", type=int, required=True)
    worker.add_argument("--stop-new-epoch", type=float)

    benchmark_child = subparsers.add_parser("benchmark-worker", help=argparse.SUPPRESS)
    benchmark_child.add_argument("--run", type=Path, required=True)
    benchmark_child.add_argument("--tier", choices=("continuous", "molecular"), required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "validate":
        protocol = protocol_for_profile(load_protocol(), args.profile)
        devices = require_gpu(expected_visible=int(protocol["operations"]["required_gpus"]))
        result = environment_manifest(PROJECT_ROOT)
        result.update(protocol_valid=True, profile=args.profile, devices=devices, free_gib=free_gib(PROJECT_ROOT))
        _print(result)
        return

    run_dir: Path = args.run.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.command == "status":
        registration_path = run_dir / "registration.json"
        status_path = run_dir / "STATUS.json"
        status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {"phase": "not_started"}
        status["queues"] = {
            stage: queue_status(run_dir, stage)
            for stage in QUEUE_STAGES if (run_dir / "queues" / stage).exists()
        }
        status["profile"] = (
            json.loads(registration_path.read_text(encoding="utf-8")).get("profile")
            if registration_path.exists() else None
        )
        _print(status)
        return
    if args.command == "campaign":
        protocol = protocol_for_profile(load_protocol(), args.profile)
        _print(run_campaign(run_dir, protocol, args.profile))
        return

    protocol, profile = _registered_protocol(run_dir, getattr(args, "profile", None))
    if args.command != "verify":
        _verify_registered_source(run_dir)
    if args.command == "benchmark-worker":
        _print(benchmark_worker(run_dir / "benchmark" / f"{args.tier}.json", args.tier, protocol))
    elif args.command == "worker":
        raise SystemExit(worker_loop(run_dir, args.stage, args.worker_id, protocol, args.stop_new_epoch))
    elif args.command == "benchmark":
        require_gpu(expected_visible=2)
        _print(run_benchmark(run_dir, protocol))
    elif args.command == "calibrate":
        _run_selected_stages(run_dir, protocol, ("calibration",))
        _print({"status": "complete", "stage": "calibration"})
    elif args.command == "develop":
        _run_selected_stages(run_dir, protocol, ("development", "training"))
        _print({"status": "complete", "stages": ["development", "training"]})
    elif args.command == "confirm":
        _run_selected_stages(run_dir, protocol, ("confirmation", "audit", "prediction"))
        _print({"status": "complete", "stages": ["confirmation", "audit", "prediction"]})
    elif args.command == "controls":
        _run_selected_stages(run_dir, protocol, ("controls",))
        _print({"status": "complete", "stage": "controls"})
    elif args.command == "analyze":
        _print(analyze_campaign(run_dir, protocol))
    elif args.command == "verify":
        result = verify_run(run_dir)
        _print(result)
        if not result["verified"]:
            raise SystemExit(1)
    elif args.command == "replay":
        result = replay_analysis(run_dir, protocol)
        _print(result)
        if not result["identical"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
