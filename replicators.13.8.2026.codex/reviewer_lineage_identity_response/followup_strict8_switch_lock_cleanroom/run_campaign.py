"""Run the prospective strict-eight matched-donor and switch-lock campaign."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence


TASK_ROOT = Path(__file__).resolve().parent
CODEX_ROOT = TASK_ROOT.parents[1]
if str(CODEX_ROOT) not in sys.path:
    sys.path.insert(0, str(CODEX_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(TASK_ROOT / "artifacts" / "matplotlib"))

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from threadpoolctl import threadpool_limits

from plastic_heredity.config import CANDIDATES, GardConfig
from plastic_heredity.mechanistic import sha256_file
from plastic_heredity.seeds import derive_seed
from plastic_heredity.simulator import advance_fission, generate_beta, generate_initial_composition
from reviewer_lineage_identity_response.followup_dynamic_regime_cleanroom.regime_core import (
    scaled_beta,
    scaled_config,
)
from reviewer_lineage_identity_response.followup_strict8_switch_lock_cleanroom.switch_lock_core import (
    ARMS,
    CHECKPOINTS,
    fork_pair_metrics,
    simulate_donor_lineage,
    simulate_lock_future,
    target_maximum_residence,
    target_occupancy,
    terminal_target_capture,
)


FORMAT = "gard-strict8-switch-lock-cleanroom-v1"
MASTER_SEED = "93215f519746560e614232582321142186621022b13f1e46cbcb1b7d139ff075"
BASE_GARD = GardConfig()
CANDIDATES_USED = ("02", "03")
ANCHORS = (
    ("b1p0_l1p0", 1.0, 1.0),
    ("b2p0_l2p0", 2.0, 2.0),
)
PRIMARY_ANCHOR = "b1p0_l1p0"
KINDS = ("strict_extension", "f12_only")
FUTURE_HORIZON = 32
DONOR_HORIZON = 32
BOOTSTRAPS = 4_096
SOFT_LIMIT_SECONDS = 27_000.0
HARD_LIMIT_SECONDS = 28_800.0
PROJECTION_BUDGET_SECONDS = 23_400.0
MATCH_SCALES = np.asarray([4.0, 4.0, 0.10, 0.10, 10.0], dtype=np.float64)

TIERS: dict[str, dict[str, int]] = {
    "A": {"matrices": 64, "donor_lineages": 192, "future_replicates": 16},
    "B": {"matrices": 48, "donor_lineages": 160, "future_replicates": 12},
    "C": {"matrices": 36, "donor_lineages": 128, "future_replicates": 10},
}

ARTIFACT_ROOT = TASK_ROOT / "artifacts"
PROTOCOL_ROOT = ARTIFACT_ROOT / "protocol"
WORK_ROOT = ARTIFACT_ROOT / "work"
DONOR_ROOT = WORK_ROOT / "donors"
FUTURE_ROOT = WORK_ROOT / "futures"
OUTPUT_ROOT = ARTIFACT_ROOT / "output"
VERIFICATION_ROOT = ARTIFACT_ROOT / "verification"
RECEIPT_ROOT = VERIFICATION_ROOT / "receipts"
STATUS_PATH = ARTIFACT_ROOT / "STATUS.json"
LEDGER_PATH = ARTIFACT_ROOT / "runtime_ledger.json"
PROTOCOL_PATH = PROTOCOL_ROOT / "protocol.json"
REGISTRATION_PATH = PROTOCOL_ROOT / "registration.json"
MANIFEST_PATH = PROTOCOL_ROOT / "scientific_source_manifest.json"
SEED_PATH = PROTOCOL_ROOT / "seed_registry.json"
BENCHMARK_PATH = PROTOCOL_ROOT / "benchmark.json"
RUNTIME_PATH = PROTOCOL_ROOT / "runtime_selection.json"
SELECTION_PATH = PROTOCOL_ROOT / "donor_selection.json"
REGISTRY_PATH = PROTOCOL_ROOT / "donor_registry.npz"

SOURCE_PATHS = {
    "gard_config": CODEX_ROOT / "plastic_heredity" / "config.py",
    "gard_simulator": CODEX_ROOT / "plastic_heredity" / "simulator.py",
    "gard_strict8": CODEX_ROOT / "plastic_heredity" / "regime_confirmation.py",
    "seed_derivation": CODEX_ROOT / "plastic_heredity" / "seeds.py",
    "carrier_adapter": TASK_ROOT.parent / "followup_carrier_cleanroom" / "carrier_core.py",
    "scaling_core": TASK_ROOT.parent / "followup_dynamic_regime_cleanroom" / "regime_core.py",
    "switch_lock_core": TASK_ROOT / "switch_lock_core.py",
    "runner": TASK_ROOT / "run_campaign.py",
    "tests": TASK_ROOT / "test_switch_lock.py",
    "protocol_text": TASK_ROOT / "PROTOCOL.md",
    "reporting_boundary": TASK_ROOT / "REPORTING_BOUNDARY.md",
    "next_steps": TASK_ROOT / "NEXT_STEPS.md",
    "detached_runner": TASK_ROOT / "run_detached_pipeline.sh",
}


class SoftStop(RuntimeError):
    """Raised only after a durable atomic checkpoint."""


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(_json_ready(value), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _assert_local(path: Path) -> None:
    resolved = path.resolve()
    if TASK_ROOT.resolve() not in (resolved, *resolved.parents):
        raise ValueError(f"write outside clean-room folder: {resolved}")


def _write_json(path: Path, value: Any) -> None:
    _assert_local(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(_json_ready(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_npz(path: Path, **values: NDArray) -> None:
    _assert_local(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **values)
    os.replace(temporary, path)


def _load_npz(path: Path) -> dict[str, NDArray]:
    with np.load(path, allow_pickle=False) as bundle:
        return {name: bundle[name] for name in bundle.files}


def _write_dataframe(path: Path, table: pd.DataFrame) -> None:
    _assert_local(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    table.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _write_checksums(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    files = sorted(path for path in directory.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    lines = [f"{sha256_file(path)}  {path.relative_to(directory)}" for path in files]
    (directory / "SHA256SUMS").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _verify_checksums(directory: Path) -> bool:
    checksum = directory / "SHA256SUMS"
    if not checksum.is_file():
        return False
    for line in checksum.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        path = directory / relative
        if not path.is_file() or sha256_file(path) != digest:
            return False
    return True


def _runtime() -> dict[str, str]:
    result = {"python": platform.python_version()}
    for package in ("numpy", "pandas", "scipy", "threadpoolctl"):
        result[package] = importlib.metadata.version(package)
    return result


def _manifest() -> dict[str, Any]:
    entries: dict[str, Any] = {}
    for name, path in sorted(SOURCE_PATHS.items()):
        if not path.is_file():
            raise FileNotFoundError(path)
        entries[name] = {"path": str(path.resolve()), "sha256": sha256_file(path), "bytes": path.stat().st_size}
    result = {"classification": "scientific_input_and_implementation", "entries": entries}
    result["manifest_id"] = _canonical_digest(result)
    return result


def _firewall_audit() -> dict[str, Any]:
    source = (TASK_ROOT / "run_campaign.py").read_text(encoding="utf-8") + (TASK_ROOT / "switch_lock_core.py").read_text(encoding="utf-8")
    forbidden = ("from " + "NewIdeas", "import " + "NewIdeas")
    hits = [token for token in forbidden if token in source]
    return {
        "passed": not hits,
        "forbidden_import_hits": hits,
        "wagner_code_read_imported_copied_or_executed": False,
        "prior_outcomes_role": "hypothesis_generation_only_not_pooled",
    }


def _initial_ledger() -> dict[str, Any]:
    return {
        "format": FORMAT,
        "created_at_epoch": time.time(),
        "cumulative_seconds": 0.0,
        "active_started_epoch": None,
        "active_cumulative_at_start": None,
        "runs_started": 0,
        "soft_limit_seconds": SOFT_LIMIT_SECONDS,
        "hard_limit_seconds": HARD_LIMIT_SECONDS,
    }


def _recover_ledger() -> dict[str, Any]:
    ledger = _read_json(LEDGER_PATH) if LEDGER_PATH.is_file() else _initial_ledger()
    if ledger.get("active_started_epoch") is not None:
        prior = float(ledger.get("active_cumulative_at_start") or ledger.get("cumulative_seconds", 0.0))
        ledger["cumulative_seconds"] = min(
            HARD_LIMIT_SECONDS,
            prior + max(0.0, time.time() - float(ledger["active_started_epoch"])),
        )
        ledger["active_started_epoch"] = None
        ledger["active_cumulative_at_start"] = None
        ledger["recovered_interrupted_run_at_epoch"] = time.time()
        _write_json(LEDGER_PATH, ledger)
    return ledger


def _start_meter() -> None:
    ledger = _recover_ledger()
    if float(ledger["cumulative_seconds"]) >= HARD_LIMIT_SECONDS:
        raise SoftStop("cumulative eight-hour hard limit already exhausted")
    ledger["active_started_epoch"] = time.time()
    ledger["active_cumulative_at_start"] = float(ledger["cumulative_seconds"])
    ledger["runs_started"] = int(ledger.get("runs_started", 0)) + 1
    _write_json(LEDGER_PATH, ledger)


def _ledger_elapsed() -> float:
    ledger = _read_json(LEDGER_PATH) if LEDGER_PATH.is_file() else _initial_ledger()
    elapsed = float(ledger["cumulative_seconds"])
    if ledger.get("active_started_epoch") is not None:
        elapsed = float(ledger["active_cumulative_at_start"]) + time.time() - float(ledger["active_started_epoch"])
    return min(HARD_LIMIT_SECONDS, elapsed)


def _stop_meter() -> None:
    if not LEDGER_PATH.is_file():
        return
    ledger = _read_json(LEDGER_PATH)
    if ledger.get("active_started_epoch") is not None:
        prior = float(ledger.get("active_cumulative_at_start") or 0.0)
        ledger["cumulative_seconds"] = min(
            HARD_LIMIT_SECONDS,
            prior + max(0.0, time.time() - float(ledger["active_started_epoch"])),
        )
        ledger["last_stopped_at_epoch"] = time.time()
        ledger["active_started_epoch"] = None
        ledger["active_cumulative_at_start"] = None
        _write_json(LEDGER_PATH, ledger)


def _check_soft_stop() -> None:
    if _ledger_elapsed() >= SOFT_LIMIT_SECONDS:
        raise SoftStop("cumulative 7.5-hour soft limit reached after durable checkpoint")


def _update_status(
    *,
    state: str,
    stage: str,
    completed: int = 0,
    total: int = 0,
    started_at: float | None = None,
    message: str = "",
    error: str | None = None,
) -> None:
    _write_json(
        STATUS_PATH,
        {
            "format": FORMAT,
            "protocol_id": _read_json(PROTOCOL_PATH).get("protocol_id") if PROTOCOL_PATH.is_file() else None,
            "state": state,
            "stage": stage,
            "completed": completed,
            "total": total,
            "message": message,
            "error": error,
            "pid": os.getpid(),
            "stage_elapsed_seconds": None if started_at is None else time.time() - started_at,
            "cumulative_elapsed_seconds": _ledger_elapsed() if LEDGER_PATH.is_file() else 0.0,
            "updated_at_epoch": time.time(),
        },
    )


def _protocol_payload() -> dict[str, Any]:
    value: dict[str, Any] = {
        "format": FORMAT,
        "status": "frozen_before_donor_or_future_production",
        "classification": "next_preprint_exploratory_followup",
        "questions": [
            "does same-event strict8 extension enrich lineage identity over identically aged matched F12-only B?",
            "can an engineered target wave plus turnover quench create persistence that survives cue removal?",
        ],
        "candidates": list(CANDIDATES_USED),
        "anchors": [
            {"grid_id": name, "beta_multiplier": beta, "leave_multiplier": leave}
            for name, beta, leave in ANCHORS
        ],
        "primary_anchor": PRIMARY_ANCHOR,
        "donors": {
            "horizon": DONOR_HORIZON,
            "b_definition": "third daughter of first registered strict8 window or first F12 run3",
            "strict_extension": "first registered all8 window, captured at its third daughter rather than eighth",
            "f12_only": "F12 event and no registered strict8 anywhere in F32",
            "required_per_kind": 2,
            "strict_selection": "semantic hash",
            "matching_features": ["break_time", "run3_start", "first3_min_pairwise", "first3_max_anchor_h", "B_mass"],
            "matching_scales": MATCH_SCALES.tolist(),
            "composition_identity_not_used_for_matching": True,
        },
        "arms": [arm.__dict__ for arm in ARMS],
        "wave": {
            "coordinates": 32,
            "mask": "beta incoming-plus-outgoing influence rank",
            "period": 4,
            "coupling": 2.0,
            "amplitude": "(1+cos(2*pi*(g-1)/4+phase))/2",
            "quench_absolute_leave_multiplier": 0.5,
            "release_after_generation": 8,
        },
        "endpoints": {
            "primary": ["same_parent_static", "cross_parent_static", "terminal_target_capture", "target_residence"],
            "secondary": ["phase_aligned_same", "phase_aligned_cross", "occupancy"],
            "checkpoints": list(CHECKPOINTS),
        },
        "gates": {
            "strict_enrichment": "primary-anchor strict control same-cross F8 lower>.10, F32 lower>.05, strict-minus-F12 DID lower>.05 in both candidates",
            "strong_static_identity": "strict same-parent F8 lower>.90 in addition to strict enrichment",
            "autonomous_switch_lock": "supported specificity plus pulse-release F32 capture lower>.20 and gain-control lower>.10 in both candidates",
            "phase_coding_secondary": "phase-minus-static same-cross lower>.05 in both candidates; cannot rescue static",
        },
        "tiers": TIERS,
        "inference": {"unit": "whole catalytic matrix", "bootstraps": BOOTSTRAPS, "candidates_separate": True},
        "runtime": {
            "projection_budget_seconds": PROJECTION_BUDGET_SECONDS,
            "soft_limit_seconds": SOFT_LIMIT_SECONDS,
            "hard_limit_seconds": HARD_LIMIT_SECONDS,
            "replay_receipts_are_durable": True,
        },
        "reporting_boundary": {
            "not_current_preprint_evidence": True,
            "not_replication": True,
            "engineered_wave_not_native_heredity": True,
            "prior_results_not_pooled": True,
            "hybrid_contract_and_matrix45_excluded_current_round": True,
        },
    }
    value["protocol_id"] = _canonical_digest(value)
    return value


def prepare() -> None:
    if PROTOCOL_PATH.is_file():
        verify_protocol()
        print(f"Existing protocol verified: {_read_json(PROTOCOL_PATH)['protocol_id']}")
        return
    firewall = _firewall_audit()
    if not firewall["passed"]:
        raise ValueError(f"clean-room firewall failed: {firewall}")
    manifest = _manifest()
    protocol = _protocol_payload()
    _write_json(MANIFEST_PATH, manifest)
    _write_json(PROTOCOL_PATH, protocol)
    _write_json(
        SEED_PATH,
        {
            "master_seed": MASTER_SEED,
            "production_domains": ["donor.beta", "donor.lineage", "selection", "future.dynamics", "future.shuffle", "bootstrap", "replay"],
            "nonproduction_domains": ["test", "smoke", "benchmark"],
        },
    )
    registration = {
        "format": FORMAT,
        "protocol_id": protocol["protocol_id"],
        "source_manifest_id": manifest["manifest_id"],
        "registered_at_epoch": time.time(),
        "runtime": _runtime(),
        "firewall": firewall,
    }
    registration["registration_id"] = _canonical_digest(registration)
    _write_json(REGISTRATION_PATH, registration)
    _write_checksums(PROTOCOL_ROOT)
    _update_status(state="prepared", stage="prepare", message="sealed before donor production")
    print(json.dumps(registration, indent=2))


def verify_protocol() -> dict[str, Any]:
    protocol = _read_json(PROTOCOL_PATH)
    payload = {key: value for key, value in protocol.items() if key != "protocol_id"}
    if protocol.get("protocol_id") != _canonical_digest(payload):
        raise ValueError("protocol digest mismatch")
    manifest = _read_json(MANIFEST_PATH)
    for entry in manifest["entries"].values():
        if sha256_file(Path(entry["path"])) != entry["sha256"]:
            raise ValueError(f"sealed source changed: {entry['path']}")
    if not _verify_checksums(PROTOCOL_ROOT):
        raise ValueError("protocol checksums failed")
    if not _firewall_audit()["passed"]:
        raise ValueError("clean-room firewall failed")
    return protocol


def _matrix_beta(domain: str, matrix_id: int) -> NDArray[np.float64]:
    rng = np.random.default_rng(derive_seed(MASTER_SEED, f"{domain}.beta", matrix_id))
    return generate_beta(BASE_GARD, rng)


def _runtime_selection() -> dict[str, Any]:
    value = _read_json(RUNTIME_PATH)
    payload = {key: item for key, item in value.items() if key != "selection_id"}
    if value["selection_id"] != _canonical_digest(payload):
        raise ValueError("runtime selection digest mismatch")
    return value


def _design() -> dict[str, int]:
    return {key: int(value) for key, value in _runtime_selection()["design"].items()}


def _benchmark_worker(index: int) -> dict[str, float]:
    beta0 = _matrix_beta("benchmark", index)
    donor_started = time.time()
    donor_count = 0
    for candidate in CANDIDATES_USED:
        for anchor_id, beta_multiplier, leave_multiplier in ANCHORS:
            beta = scaled_beta(beta0, beta_multiplier)
            config = scaled_config(BASE_GARD, leave_multiplier)
            for lineage in range(4):
                simulate_donor_lineage(
                    beta,
                    config,
                    CANDIDATES[candidate],
                    seed=derive_seed(MASTER_SEED, "benchmark.donor", index, candidate, anchor_id, lineage),
                )
                donor_count += 1
    donor_rate = (time.time() - donor_started) / donor_count

    future_started = time.time()
    future_count = 0
    state = generate_initial_composition(BASE_GARD, np.random.default_rng(derive_seed(MASTER_SEED, "benchmark.state", index)))
    for candidate in CANDIDATES_USED:
        for anchor_id, beta_multiplier, leave_multiplier in ANCHORS:
            beta = scaled_beta(beta0, beta_multiplier)
            anchor_config = scaled_config(BASE_GARD, leave_multiplier)
            lock_config = scaled_config(BASE_GARD, 0.5)
            for arm in ARMS:
                simulate_lock_future(
                    state,
                    state,
                    beta,
                    anchor_config,
                    lock_config,
                    CANDIDATES[candidate],
                    arm,
                    dynamics_seed=derive_seed(MASTER_SEED, "benchmark.future", index, candidate, anchor_id, arm.name),
                    shuffle_seed=derive_seed(MASTER_SEED, "benchmark.shuffle", index, candidate, anchor_id, arm.name),
                    horizon=8,
                )
                future_count += 1
    future_rate_f32 = (time.time() - future_started) / future_count * (32.0 / 8.0)
    return {"donor_seconds": donor_rate, "future_f32_seconds": future_rate_f32}


def _tier_projection(design: Mapping[str, int], donor_rate: float, future_rate: float, workers: int) -> float:
    donor_units = len(CANDIDATES_USED) * design["matrices"] * len(ANCHORS) * design["donor_lineages"]
    future_units = (
        len(CANDIDATES_USED)
        * design["matrices"]
        * len(ANCHORS)
        * len(KINDS)
        * len(ARMS)
        * design["future_replicates"]
        * 4
    )
    effective_workers = max(1.0, min(float(workers), 16.0) * 0.65)
    one_pass = (donor_units * donor_rate + future_units * future_rate) / effective_workers
    return 2.0 * one_pass * 1.75 + 1_200.0


def benchmark(workers: int) -> dict[str, Any]:
    verify_protocol()
    if RUNTIME_PATH.is_file():
        return _runtime_selection()
    sample_workers = min(max(1, workers), 4)
    started = time.time()
    if sample_workers == 1:
        samples = [_benchmark_worker(0)]
    else:
        with ProcessPoolExecutor(max_workers=sample_workers) as executor:
            samples = list(executor.map(_benchmark_worker, range(sample_workers)))
    donor_rate = float(np.mean([row["donor_seconds"] for row in samples]))
    future_rate = float(np.mean([row["future_f32_seconds"] for row in samples]))
    projections = []
    selected = None
    for tier_name, design in TIERS.items():
        seconds = _tier_projection(design, donor_rate, future_rate, workers)
        row = {"tier": tier_name, "projected_seconds": seconds, "fits": seconds <= PROJECTION_BUDGET_SECONDS}
        projections.append(row)
        if selected is None and row["fits"]:
            selected = tier_name
    if selected is None:
        selected = "C"
    payload = {
        "format": FORMAT,
        "workers": workers,
        "sample_workers": sample_workers,
        "wall_seconds": time.time() - started,
        "donor_seconds_per_lineage": donor_rate,
        "future_seconds_per_f32": future_rate,
        "samples": samples,
        "projections": projections,
    }
    _write_json(BENCHMARK_PATH, payload)
    selection = {
        "format": FORMAT,
        "protocol_id": _read_json(PROTOCOL_PATH)["protocol_id"],
        "benchmark_sha256": sha256_file(BENCHMARK_PATH),
        "tier": selected,
        "design": TIERS[selected],
        "selected_before_donor_production": True,
    }
    selection["selection_id"] = _canonical_digest(selection)
    _write_json(RUNTIME_PATH, selection)
    _write_checksums(PROTOCOL_ROOT)
    return selection


def _donor_path(candidate: str, matrix_id: int) -> Path:
    return DONOR_ROOT / f"c{candidate}_m{matrix_id:03d}.npz"


def _future_path(candidate: str, matrix_id: int) -> Path:
    return FUTURE_ROOT / f"c{candidate}_m{matrix_id:03d}.npz"


def _donor_task(args: tuple[str, int, int]) -> dict[str, NDArray]:
    candidate, matrix_id, lineages = args
    labels = np.zeros((len(ANCHORS), lineages), dtype=np.int8)
    states = np.zeros((len(ANCHORS), lineages, BASE_GARD.n_types), dtype=np.int16)
    features = np.full((len(ANCHORS), lineages, 5), np.nan, dtype=np.float64)
    completed = np.zeros((len(ANCHORS), lineages), dtype=np.int8)
    beta0 = _matrix_beta("donor", matrix_id)
    with threadpool_limits(limits=1):
        for anchor_index, (anchor_id, beta_multiplier, leave_multiplier) in enumerate(ANCHORS):
            beta = scaled_beta(beta0, beta_multiplier)
            config = scaled_config(BASE_GARD, leave_multiplier)
            for lineage in range(lineages):
                event, observed = simulate_donor_lineage(
                    beta,
                    config,
                    CANDIDATES[candidate],
                    seed=derive_seed(MASTER_SEED, "donor.lineage", candidate, matrix_id, anchor_id, lineage),
                    horizon=DONOR_HORIZON,
                )
                completed[anchor_index, lineage] = int(observed == DONOR_HORIZON)
                if observed != DONOR_HORIZON:
                    continue
                label = 2 if event.strict_extension else 1 if event.f12_only else 0
                labels[anchor_index, lineage] = label
                if label:
                    states[anchor_index, lineage] = event.b_state.astype(np.int16)
                    features[anchor_index, lineage] = event.match_features
    return {
        "anchor_ids": np.asarray([row[0] for row in ANCHORS]),
        "labels": labels,
        "states": states,
        "features": features,
        "completed": completed,
    }


def _save_donor_task(args: tuple[str, int, int]) -> str:
    candidate, matrix_id, _ = args
    path = _donor_path(candidate, matrix_id)
    if not path.is_file():
        _atomic_npz(path, **_donor_task(args))
    return str(path)


def _run_tasks(
    function: Callable[[Any], str],
    tasks: Sequence[Any],
    *,
    workers: int,
    stage: str,
) -> None:
    started = time.time()
    existing = sum(Path(_donor_path(task[0], task[1]) if stage == "donors" else _future_path(task[0], task[1])).is_file() for task in tasks)
    completed = existing
    pending = [task for task in tasks if not (_donor_path(task[0], task[1]) if stage == "donors" else _future_path(task[0], task[1])).is_file()]
    _update_status(state="running", stage=stage, completed=completed, total=len(tasks), started_at=started)
    if not pending:
        return
    with ProcessPoolExecutor(max_workers=min(workers, len(pending))) as executor:
        futures = {executor.submit(function, task): task for task in pending}
        for future in as_completed(futures):
            path = future.result()
            completed += 1
            _update_status(
                state="running",
                stage=stage,
                completed=completed,
                total=len(tasks),
                started_at=started,
                message=Path(path).name,
            )
            _check_soft_stop()


def run_donors(workers: int) -> None:
    design = _design()
    tasks = [
        (candidate, matrix_id, design["donor_lineages"])
        for candidate in CANDIDATES_USED
        for matrix_id in range(design["matrices"])
    ]
    _run_tasks(_save_donor_task, tasks, workers=workers, stage="donors")


def _semantic_order(candidate: str, matrix_id: int, anchor_id: str, lineage: int, label: str) -> int:
    return derive_seed(MASTER_SEED, "selection", label, candidate, matrix_id, anchor_id, lineage)


def _match_controls(
    strict_indices: Sequence[int],
    f12_indices: Sequence[int],
    features: NDArray,
    *,
    candidate: str,
    matrix_id: int,
    anchor_id: str,
) -> list[int]:
    available = set(int(value) for value in f12_indices)
    selected: list[int] = []
    for strict_index in strict_indices:
        ranked = []
        for control in available:
            delta = (features[control] - features[strict_index]) / MATCH_SCALES
            distance = float(np.dot(delta, delta))
            ranked.append((distance, _semantic_order(candidate, matrix_id, anchor_id, control, "control_tie"), control))
        if not ranked:
            return []
        control = min(ranked)[2]
        selected.append(control)
        available.remove(control)
    return selected


def select_donors() -> dict[str, Any]:
    verify_protocol()
    if SELECTION_PATH.is_file() and REGISTRY_PATH.is_file():
        return _read_json(SELECTION_PATH)
    design = _design()
    shape = (len(CANDIDATES_USED), design["matrices"], len(ANCHORS), len(KINDS), 2)
    selected_states = np.zeros(shape + (BASE_GARD.n_types,), dtype=np.int16)
    selected_features = np.full(shape + (5,), np.nan, dtype=np.float64)
    selected_lineages = np.full(shape, -1, dtype=np.int32)
    eligibility = np.zeros(shape[:3], dtype=np.int8)
    counts: dict[str, dict[str, int]] = {candidate: {row[0]: 0 for row in ANCHORS} for candidate in CANDIDATES_USED}
    for candidate_index, candidate in enumerate(CANDIDATES_USED):
        for matrix_id in range(design["matrices"]):
            data = _load_npz(_donor_path(candidate, matrix_id))
            for anchor_index, (anchor_id, _, _) in enumerate(ANCHORS):
                labels = data["labels"][anchor_index]
                strict_available = np.flatnonzero(labels == 2).tolist()
                f12_available = np.flatnonzero(labels == 1).tolist()
                if len(strict_available) < 2 or len(f12_available) < 2:
                    continue
                strict = sorted(
                    strict_available,
                    key=lambda lineage: _semantic_order(candidate, matrix_id, anchor_id, lineage, "strict"),
                )[:2]
                controls = _match_controls(
                    strict,
                    f12_available,
                    data["features"][anchor_index],
                    candidate=candidate,
                    matrix_id=matrix_id,
                    anchor_id=anchor_id,
                )
                if len(controls) != 2:
                    continue
                eligibility[candidate_index, matrix_id, anchor_index] = 1
                counts[candidate][anchor_id] += 1
                for kind_index, indices in enumerate((strict, controls)):
                    for parent_index, lineage in enumerate(indices):
                        selected_states[candidate_index, matrix_id, anchor_index, kind_index, parent_index] = data["states"][anchor_index, lineage]
                        selected_features[candidate_index, matrix_id, anchor_index, kind_index, parent_index] = data["features"][anchor_index, lineage]
                        selected_lineages[candidate_index, matrix_id, anchor_index, kind_index, parent_index] = lineage
    _atomic_npz(
        REGISTRY_PATH,
        candidate_ids=np.asarray(CANDIDATES_USED),
        anchor_ids=np.asarray([row[0] for row in ANCHORS]),
        kind_ids=np.asarray(KINDS),
        states=selected_states,
        features=selected_features,
        lineages=selected_lineages,
        eligibility=eligibility,
    )
    payload = {
        "format": FORMAT,
        "protocol_id": _read_json(PROTOCOL_PATH)["protocol_id"],
        "runtime_selection_id": _runtime_selection()["selection_id"],
        "registry_sha256": sha256_file(REGISTRY_PATH),
        "eligible_matrix_counts": counts,
        "required_two_strict_and_two_f12": True,
        "selected_before_fresh_futures": True,
    }
    payload["selection_id"] = _canonical_digest(payload)
    _write_json(SELECTION_PATH, payload)
    _write_checksums(PROTOCOL_ROOT)
    return payload


def _future_task(args: tuple[str, int, int]) -> dict[str, NDArray]:
    candidate, matrix_id, replicates = args
    registry = _load_npz(REGISTRY_PATH)
    candidate_index = CANDIDATES_USED.index(candidate)
    metric_shape = (len(ANCHORS), len(KINDS), len(ARMS), replicates, len(CHECKPOINTS))
    same_static = np.full(metric_shape, np.nan, dtype=np.float64)
    cross_static = np.full(metric_shape, np.nan, dtype=np.float64)
    same_phase = np.full(metric_shape, np.nan, dtype=np.float64)
    cross_phase = np.full(metric_shape, np.nan, dtype=np.float64)
    capture = np.full(metric_shape, np.nan, dtype=np.float64)
    scalar_shape = metric_shape[:-1]
    occupancy = np.full(scalar_shape, np.nan, dtype=np.float64)
    residence = np.full(scalar_shape, np.nan, dtype=np.float64)
    completion = np.full(scalar_shape, np.nan, dtype=np.float64)
    digests = np.full(scalar_shape + (4,), "", dtype="U64")
    beta0 = _matrix_beta("donor", matrix_id)
    with threadpool_limits(limits=1):
        for anchor_index, (anchor_id, beta_multiplier, leave_multiplier) in enumerate(ANCHORS):
            if not registry["eligibility"][candidate_index, matrix_id, anchor_index]:
                continue
            beta = scaled_beta(beta0, beta_multiplier)
            anchor_config = scaled_config(BASE_GARD, leave_multiplier)
            lock_config = scaled_config(BASE_GARD, 0.5)
            for kind_index, kind in enumerate(KINDS):
                targets = registry["states"][candidate_index, matrix_id, anchor_index, kind_index].astype(np.int64)
                for arm_index, arm in enumerate(ARMS):
                    for replicate in range(replicates):
                        trajectories = []
                        target_list = []
                        for parent in range(2):
                            for fork in range(2):
                                trajectory = simulate_lock_future(
                                    targets[parent],
                                    targets[parent],
                                    beta,
                                    anchor_config,
                                    lock_config,
                                    CANDIDATES[candidate],
                                    arm,
                                    dynamics_seed=derive_seed(
                                        MASTER_SEED,
                                        "future.dynamics",
                                        candidate,
                                        matrix_id,
                                        anchor_id,
                                        kind,
                                        arm.name,
                                        replicate,
                                        parent,
                                        fork,
                                    ),
                                    shuffle_seed=derive_seed(
                                        MASTER_SEED,
                                        "future.shuffle",
                                        candidate,
                                        matrix_id,
                                        anchor_id,
                                        kind,
                                        arm.name,
                                        parent,
                                    ),
                                    horizon=FUTURE_HORIZON,
                                )
                                trajectories.append(trajectory)
                                target_list.append(targets[parent])
                        a0, a1, b0, b1 = trajectories
                        metrics = fork_pair_metrics(a0, a1, b0, b1)
                        index = (anchor_index, kind_index, arm_index, replicate)
                        same_static[index] = metrics[0]
                        cross_static[index] = metrics[1]
                        same_phase[index] = metrics[2]
                        cross_phase[index] = metrics[3]
                        capture[index] = np.nanmean(
                            [
                                [terminal_target_capture(trajectory, target, checkpoint) for checkpoint in CHECKPOINTS]
                                for trajectory, target in zip(trajectories, target_list, strict=True)
                            ],
                            axis=0,
                        )
                        occupancy[index] = np.nanmean(
                            [target_occupancy(trajectory, target) for trajectory, target in zip(trajectories, target_list, strict=True)]
                        )
                        residence[index] = np.nanmean(
                            [target_maximum_residence(trajectory, target) for trajectory, target in zip(trajectories, target_list, strict=True)]
                        )
                        completion[index] = np.mean([trajectory.observed == FUTURE_HORIZON for trajectory in trajectories])
                        digests[index] = np.asarray([trajectory.digest for trajectory in trajectories])
    return {
        "anchor_ids": np.asarray([row[0] for row in ANCHORS]),
        "kind_ids": np.asarray(KINDS),
        "arm_ids": np.asarray([arm.name for arm in ARMS]),
        "same_static": same_static,
        "cross_static": cross_static,
        "same_phase": same_phase,
        "cross_phase": cross_phase,
        "capture": capture,
        "occupancy": occupancy,
        "residence": residence,
        "completion": completion,
        "digests": digests,
    }


def _save_future_task(args: tuple[str, int, int]) -> str:
    candidate, matrix_id, _ = args
    path = _future_path(candidate, matrix_id)
    if not path.is_file():
        _atomic_npz(path, **_future_task(args))
    return str(path)


def run_futures(workers: int) -> None:
    design = _design()
    tasks = [
        (candidate, matrix_id, design["future_replicates"])
        for candidate in CANDIDATES_USED
        for matrix_id in range(design["matrices"])
    ]
    _run_tasks(_save_future_task, tasks, workers=workers, stage="futures")


def _bootstrap(values: NDArray, seed: int, confidence: float = 0.95) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return {"point": np.nan, "lower": np.nan, "upper": np.nan, "n_matrices": 0}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, array.size, size=(BOOTSTRAPS, array.size))
    means = array[draws].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "point": float(array.mean()),
        "lower": float(np.quantile(means, alpha)),
        "upper": float(np.quantile(means, 1.0 - alpha)),
        "n_matrices": int(array.size),
    }


def _future_matrix_table() -> pd.DataFrame:
    design = _design()
    rows: list[dict[str, Any]] = []
    for candidate in CANDIDATES_USED:
        for matrix_id in range(design["matrices"]):
            data = _load_npz(_future_path(candidate, matrix_id))
            for anchor_index, (anchor_id, beta, leave) in enumerate(ANCHORS):
                for kind_index, kind in enumerate(KINDS):
                    for arm_index, arm in enumerate(ARMS):
                        if not np.any(np.isfinite(data["same_static"][anchor_index, kind_index, arm_index])):
                            continue
                        row: dict[str, Any] = {
                            "candidate": candidate,
                            "matrix_id": matrix_id,
                            "grid_id": anchor_id,
                            "beta_multiplier": beta,
                            "leave_multiplier": leave,
                            "kind": kind,
                            "arm": arm.name,
                            "occupancy": float(np.nanmean(data["occupancy"][anchor_index, kind_index, arm_index])),
                            "maximum_residence": float(np.nanmean(data["residence"][anchor_index, kind_index, arm_index])),
                            "completion": float(np.nanmean(data["completion"][anchor_index, kind_index, arm_index])),
                        }
                        for checkpoint_index, checkpoint in enumerate(CHECKPOINTS):
                            for field in ("same_static", "cross_static", "same_phase", "cross_phase", "capture"):
                                row[f"{field}_f{checkpoint}"] = float(
                                    np.nanmean(data[field][anchor_index, kind_index, arm_index, :, checkpoint_index])
                                )
                            row[f"static_signal_f{checkpoint}"] = row[f"same_static_f{checkpoint}"] - row[f"cross_static_f{checkpoint}"]
                            row[f"phase_signal_f{checkpoint}"] = row[f"same_phase_f{checkpoint}"] - row[f"cross_phase_f{checkpoint}"]
                        rows.append(row)
    return pd.DataFrame(rows)


def _vector(
    table: pd.DataFrame,
    candidate: str,
    anchor: str,
    kind: str,
    arm: str,
    metric: str,
) -> pd.Series:
    cell = table[
        (table["candidate"] == candidate)
        & (table["grid_id"] == anchor)
        & (table["kind"] == kind)
        & (table["arm"] == arm)
    ]
    return cell.set_index("matrix_id")[metric].sort_index()


def _paired_difference(left: pd.Series, right: pd.Series) -> NDArray:
    paired = pd.concat([left.rename("left"), right.rename("right")], axis=1).dropna()
    return (paired["left"] - paired["right"]).to_numpy(dtype=np.float64)


def _ci(values: NDArray | pd.Series, label: str) -> dict[str, Any]:
    return _bootstrap(np.asarray(values, dtype=np.float64), derive_seed(MASTER_SEED, "bootstrap", label))


def _analyze_summary(table: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {"candidates": {}}
    for candidate in CANDIDATES_USED:
        candidate_result: dict[str, Any] = {"anchors": {}}
        for anchor, _, _ in ANCHORS:
            strict_control = {
                metric: _vector(table, candidate, anchor, "strict_extension", "control", metric)
                for metric in ("same_static_f8", "static_signal_f8", "static_signal_f32")
            }
            f12_signal = _vector(table, candidate, anchor, "f12_only", "control", "static_signal_f8")
            did = _paired_difference(strict_control["static_signal_f8"], f12_signal)
            strict_same_ci = _ci(strict_control["same_static_f8"], f"{candidate}.{anchor}.strict.same_f8")
            strict_signal_f8_ci = _ci(strict_control["static_signal_f8"], f"{candidate}.{anchor}.strict.signal_f8")
            strict_signal_f32_ci = _ci(strict_control["static_signal_f32"], f"{candidate}.{anchor}.strict.signal_f32")
            did_ci = _ci(did, f"{candidate}.{anchor}.strict_f12.did")
            strict_enrichment = bool(
                strict_signal_f8_ci["lower"] > 0.10
                and strict_signal_f32_ci["lower"] > 0.05
                and did_ci["lower"] > 0.05
            )
            strong_static = bool(strict_enrichment and strict_same_ci["lower"] > 0.90)

            qwave_capture16 = _vector(table, candidate, anchor, "strict_extension", "quench_wave", "capture_f16")
            qwave_capture32 = _vector(table, candidate, anchor, "strict_extension", "quench_wave", "capture_f32")
            control_capture16 = _vector(table, candidate, anchor, "strict_extension", "control", "capture_f16")
            control_capture32 = _vector(table, candidate, anchor, "strict_extension", "control", "capture_f32")
            comparisons: dict[str, Any] = {}
            for control_arm in ("control", "quench", "wave", "quench_static", "quench_shuffled", "quench_phase_pi"):
                other = _vector(table, candidate, anchor, "strict_extension", control_arm, "capture_f16")
                comparisons[control_arm] = _ci(
                    _paired_difference(qwave_capture16, other),
                    f"{candidate}.{anchor}.qwave-{control_arm}.capture16",
                )
            qwave16_ci = _ci(qwave_capture16, f"{candidate}.{anchor}.qwave.capture16")
            qwave32_ci = _ci(qwave_capture32, f"{candidate}.{anchor}.qwave.capture32")
            supported = bool(
                qwave16_ci["lower"] > 0.30
                and comparisons["control"]["lower"] > 0.15
                and all(comparisons[name]["lower"] > 0.0 for name in ("quench", "wave", "quench_static", "quench_shuffled", "quench_phase_pi"))
            )
            release_capture32 = _vector(table, candidate, anchor, "strict_extension", "pulse_release", "capture_f32")
            release_ci = _ci(release_capture32, f"{candidate}.{anchor}.release.capture32")
            release_gain = _ci(
                _paired_difference(release_capture32, control_capture32),
                f"{candidate}.{anchor}.release-control.capture32",
            )
            qwave_same = _ci(
                _vector(table, candidate, anchor, "strict_extension", "quench_wave", "same_static_f8"),
                f"{candidate}.{anchor}.qwave.same_f8",
            )
            qwave_signal = _ci(
                _vector(table, candidate, anchor, "strict_extension", "quench_wave", "static_signal_f8"),
                f"{candidate}.{anchor}.qwave.signal_f8",
            )
            autonomous = bool(
                supported
                and release_ci["lower"] > 0.20
                and release_gain["lower"] > 0.10
                and qwave_same["lower"] > 0.90
                and qwave_signal["lower"] > 0.10
            )
            phase = _vector(table, candidate, anchor, "strict_extension", "quench_wave", "phase_signal_f8")
            static = _vector(table, candidate, anchor, "strict_extension", "quench_wave", "static_signal_f8")
            phase_gain = _ci(_paired_difference(phase, static), f"{candidate}.{anchor}.phase-static")
            phase_pass = bool(phase_gain["lower"] > 0.05)
            candidate_result["anchors"][anchor] = {
                "eligible_matrices": int(strict_control["same_static_f8"].notna().sum()),
                "strict_same_parent_f8": strict_same_ci,
                "strict_static_signal_f8": strict_signal_f8_ci,
                "strict_static_signal_f32": strict_signal_f32_ci,
                "strict_minus_f12_signal_f8": did_ci,
                "strict_enrichment_pass": strict_enrichment,
                "strong_static_identity_pass": strong_static,
                "quench_wave_capture_f16": qwave16_ci,
                "quench_wave_capture_f32": qwave32_ci,
                "quench_wave_comparisons_f16": comparisons,
                "supported_specificity_pass": supported,
                "pulse_release_capture_f32": release_ci,
                "pulse_release_minus_control_f32": release_gain,
                "autonomous_switch_lock_pass": autonomous,
                "phase_minus_static_signal_f8": phase_gain,
                "phase_coding_secondary_pass": phase_pass,
            }
        candidate_result["primary_strict_enrichment"] = candidate_result["anchors"][PRIMARY_ANCHOR]["strict_enrichment_pass"]
        candidate_result["primary_strong_static_identity"] = candidate_result["anchors"][PRIMARY_ANCHOR]["strong_static_identity_pass"]
        candidate_result["primary_autonomous_switch_lock"] = candidate_result["anchors"][PRIMARY_ANCHOR]["autonomous_switch_lock_pass"]
        candidate_result["primary_supported_specificity"] = candidate_result["anchors"][PRIMARY_ANCHOR]["supported_specificity_pass"]
        candidate_result["primary_phase_coding_secondary"] = candidate_result["anchors"][PRIMARY_ANCHOR]["phase_coding_secondary_pass"]
        summary["candidates"][candidate] = candidate_result
    summary["strict_enrichment_both"] = all(value["primary_strict_enrichment"] for value in summary["candidates"].values())
    summary["strong_static_identity_both"] = all(value["primary_strong_static_identity"] for value in summary["candidates"].values())
    summary["autonomous_switch_lock_both"] = all(value["primary_autonomous_switch_lock"] for value in summary["candidates"].values())
    summary["supported_specificity_both"] = all(value["primary_supported_specificity"] for value in summary["candidates"].values())
    summary["phase_coding_secondary_both"] = all(value["primary_phase_coding_secondary"] for value in summary["candidates"].values())
    if summary["autonomous_switch_lock_both"]:
        classification = "autonomous_switch_lock_pass"
    elif summary["supported_specificity_both"]:
        classification = "external_supported_stabilization_only"
    elif summary["strong_static_identity_both"]:
        classification = "strong_static_identity_without_switch_lock"
    elif summary["strict_enrichment_both"]:
        classification = "strict8_enrichment_without_strong_identity"
    elif summary["phase_coding_secondary_both"]:
        classification = "phase_aligned_signal_only"
    else:
        classification = "no_registered_primary_signal"
    summary["classification"] = classification
    summary["minimum_completion"] = float(table["completion"].min())
    return summary


def _plots(table: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    primary = table[(table["grid_id"] == PRIMARY_ANCHOR) & (table["arm"] == "control")]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for axis, candidate in zip(axes, CANDIDATES_USED, strict=True):
        cell = primary[primary["candidate"] == candidate]
        means = cell.groupby("kind")[[f"static_signal_f{x}" for x in CHECKPOINTS]].mean().reindex(KINDS)
        for kind in KINDS:
            axis.plot(CHECKPOINTS, means.loc[kind], marker="o", label=kind)
        axis.set(title=f"Candidate {candidate}", xlabel="generation", ylabel="same-minus-cross static H", xticks=CHECKPOINTS)
        axis.legend()
    figure.savefig(OUTPUT_ROOT / "figure_1_strict_vs_f12.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for axis, candidate in zip(axes, CANDIDATES_USED, strict=True):
        cell = table[(table["candidate"] == candidate) & (table["grid_id"] == PRIMARY_ANCHOR) & (table["kind"] == "strict_extension")]
        means = cell.groupby("arm")["capture_f32"].mean().reindex([arm.name for arm in ARMS])
        axis.bar(np.arange(len(means)), means)
        axis.set(title=f"Candidate {candidate}", ylabel="F32 terminal target capture", xticks=np.arange(len(means)), xticklabels=means.index)
        axis.tick_params(axis="x", rotation=55)
    figure.savefig(OUTPUT_ROOT / "figure_2_switch_lock_arms.png", dpi=180)
    plt.close(figure)


def analyze() -> dict[str, Any]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    table = _future_matrix_table()
    summary = _analyze_summary(table)
    _write_dataframe(OUTPUT_ROOT / "matrix_arm_metrics.csv", table)
    selection = _read_json(SELECTION_PATH)
    payload = {
        "format": FORMAT,
        "complete_analysis": True,
        "protocol_id": _read_json(PROTOCOL_PATH)["protocol_id"],
        "runtime_selection_id": _runtime_selection()["selection_id"],
        "donor_selection_id": selection["selection_id"],
        "eligible_matrix_counts": selection["eligible_matrix_counts"],
        "results": summary,
        "reporting_boundary": "next-preprint exploratory; not current-preprint evidence or a replication; exact replay still required",
    }
    _write_json(OUTPUT_ROOT / "primary_summary.json", payload)
    _plots(table)
    _write_checksums(OUTPUT_ROOT)
    return payload


def report() -> None:
    payload = _read_json(OUTPUT_ROOT / "primary_summary.json")
    result = payload["results"]
    technical = [
        "# Strict-eight matched-donor and switch-lock results",
        "",
        "## Registered outcome",
        "",
        f"Classification: `{result['classification']}`.",
        f"Strict-extension enrichment passed in both candidates: **{result['strict_enrichment_both']}**.",
        f"Strong static lineage identity passed in both candidates: **{result['strong_static_identity_both']}**.",
        f"Autonomous switch-lock passed in both candidates: **{result['autonomous_switch_lock_both']}**.",
        f"Phase-aligned secondary signal passed in both candidates: **{result['phase_coding_secondary_both']}**.",
        "",
        "## Boundary",
        "",
        "Strict and F12 donors were selected prospectively from fresh matrices and have identical B age. The wave is an engineered target cue. Static identity is primary; phase alignment cannot rescue it. This is exploratory future-preprint work, not evidence for the current preprint and not a GARD or Wagner replication. A final verdict additionally requires the exact replay audit.",
    ]
    (OUTPUT_ROOT / "TECHNICAL_REPORT.md").write_text("\n".join(technical) + "\n", encoding="utf-8")
    lay = [
        "# Lay summary",
        "",
        f"Rare strict-eight forms carried more parent-specific information than matched short F12 forms: **{result['strict_enrichment_both']}**.",
        f"They met the demanding static heredity standard: **{result['strong_static_identity_both']}**.",
        f"The switch-then-lock wave created a form that remained after the external cue was removed: **{result['autonomous_switch_lock_both']}**.",
        f"A phase-sensitive dynamic trace appeared as a separate exploratory result: **{result['phase_coding_secondary_both']}**.",
        "These findings remain outside the current preprint and provisional until exact replay completes.",
    ]
    (OUTPUT_ROOT / "LAY_SUMMARY.md").write_text("\n".join(lay) + "\n", encoding="utf-8")
    _write_checksums(OUTPUT_ROOT)


def _arrays_equal(left: NDArray, right: NDArray) -> bool:
    if left.dtype.kind in "fc" or right.dtype.kind in "fc":
        return bool(np.array_equal(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


def _expected_jobs() -> list[tuple[str, tuple[Any, ...], Path]]:
    design = _design()
    jobs = []
    for candidate in CANDIDATES_USED:
        for matrix_id in range(design["matrices"]):
            jobs.append(("donor", (candidate, matrix_id, design["donor_lineages"]), _donor_path(candidate, matrix_id)))
            jobs.append(("future", (candidate, matrix_id, design["future_replicates"]), _future_path(candidate, matrix_id)))
    return jobs


def _receipt_path(kind: str, candidate: str, matrix_id: int) -> Path:
    return RECEIPT_ROOT / kind / f"c{candidate}_m{matrix_id:03d}.json"


def _replay_job(job: tuple[str, tuple[Any, ...], Path]) -> dict[str, Any]:
    kind, args, path = job
    expected = _load_npz(path)
    observed = _donor_task(args) if kind == "donor" else _future_task(args)
    exact = set(expected) == set(observed) and all(_arrays_equal(expected[key], observed[key]) for key in expected)
    return {
        "format": FORMAT,
        "kind": kind,
        "candidate": str(args[0]),
        "matrix_id": int(args[1]),
        "checkpoint": str(path),
        "checkpoint_sha256": sha256_file(path),
        "all_arrays_exact": bool(exact),
        "replayed_at_epoch": time.time(),
    }


def _valid_receipt(job: tuple[str, tuple[Any, ...], Path]) -> bool:
    kind, args, checkpoint = job
    path = _receipt_path(kind, str(args[0]), int(args[1]))
    if not path.is_file():
        return False
    value = _read_json(path)
    return bool(value.get("all_arrays_exact") and value.get("checkpoint_sha256") == sha256_file(checkpoint))


def verify(*, full_replay: bool, workers: int) -> dict[str, Any]:
    verify_protocol()
    jobs = _expected_jobs()
    missing = [str(path) for _, _, path in jobs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} checkpoints")
    started = time.time()
    completed = sum(_valid_receipt(job) for job in jobs)
    if full_replay:
        pending = [job for job in jobs if not _valid_receipt(job)]
        _update_status(state="running", stage="verify-replay", completed=completed, total=len(jobs), started_at=started)
        if pending:
            with ProcessPoolExecutor(max_workers=min(workers, len(pending))) as executor:
                futures = {executor.submit(_replay_job, job): job for job in pending}
                for future in as_completed(futures):
                    receipt = future.result()
                    path = _receipt_path(receipt["kind"], receipt["candidate"], receipt["matrix_id"])
                    _write_json(path, receipt)
                    completed += 1
                    _update_status(
                        state="running",
                        stage="verify-replay",
                        completed=completed,
                        total=len(jobs),
                        started_at=started,
                        message=path.name,
                    )
                    if not receipt["all_arrays_exact"]:
                        raise ValueError(f"replay mismatch: {receipt['checkpoint']}")
                    _check_soft_stop()
    exact = sum(_valid_receipt(job) for job in jobs)
    complete = bool(
        full_replay
        and exact == len(jobs)
        and _verify_checksums(PROTOCOL_ROOT)
        and _verify_checksums(OUTPUT_ROOT)
        and _firewall_audit()["passed"]
        and sha256_file(REGISTRY_PATH) == _read_json(SELECTION_PATH)["registry_sha256"]
    )
    audit = {
        "format": FORMAT,
        "protocol_id": _read_json(PROTOCOL_PATH)["protocol_id"],
        "full_replay": full_replay,
        "checkpoint_count": len(jobs),
        "exact_receipt_count": exact,
        "durable_receipts": True,
        "protocol_checksums": _verify_checksums(PROTOCOL_ROOT),
        "output_checksums": _verify_checksums(OUTPUT_ROOT),
        "cleanroom_firewall": _firewall_audit(),
        "complete": complete,
    }
    _write_json(VERIFICATION_ROOT / "verification_audit.json", audit)
    _write_checksums(VERIFICATION_ROOT)
    if full_replay and not complete:
        raise ValueError("full replay verification failed")
    return audit


def smoke() -> None:
    beta = _matrix_beta("smoke", 0)
    initial = generate_initial_composition(BASE_GARD, np.random.default_rng(derive_seed(MASTER_SEED, "smoke.state")))
    for candidate in CANDIDATES_USED:
        seed = derive_seed(MASTER_SEED, "smoke.control", candidate)
        observed = simulate_lock_future(
            initial,
            initial,
            beta,
            BASE_GARD,
            scaled_config(BASE_GARD, 0.5),
            CANDIDATES[candidate],
            ARMS[0],
            dynamics_seed=seed,
            shuffle_seed=derive_seed(MASTER_SEED, "smoke.shuffle", candidate),
            horizon=4,
        )
        rng = np.random.default_rng(seed)
        current = initial.copy()
        expected = []
        for _ in range(4):
            record = advance_fission(current, beta, BASE_GARD, CANDIDATES[candidate], rng)
            expected.append(record.daughter.copy())
            current = record.daughter
        if not np.array_equal(observed.states[:4], np.asarray(expected)):
            raise AssertionError(f"control parity failed for candidate {candidate}")
    print("Two-candidate control parity smoke passed; production seeds were not consumed.")


def run_tests() -> None:
    subprocess.run([sys.executable, "-m", "pytest", str(TASK_ROOT / "test_switch_lock.py"), "-q"], cwd=CODEX_ROOT, check=True)


def status() -> None:
    payload = _read_json(STATUS_PATH) if STATUS_PATH.is_file() else {"format": FORMAT, "state": "not_started", "stage": "none"}
    elapsed = _ledger_elapsed() if LEDGER_PATH.is_file() else 0.0
    payload["cumulative_elapsed_seconds"] = elapsed
    payload["soft_remaining_seconds"] = max(0.0, SOFT_LIMIT_SECONDS - elapsed)
    payload["hard_remaining_seconds"] = max(0.0, HARD_LIMIT_SECONDS - elapsed)
    if RUNTIME_PATH.is_file():
        payload["tier"] = _runtime_selection()["tier"]
    print(json.dumps(payload, indent=2, sort_keys=True))


def remaining_hard() -> None:
    _recover_ledger()
    print(max(0, int(HARD_LIMIT_SECONDS - _ledger_elapsed())))


def mark_hard_stop() -> None:
    _recover_ledger()
    _update_status(state="incomplete_walltime_hard_stop", stage="hard-stop", message="external eight-hour watchdog stopped the pipeline")


def run_all(workers: int) -> None:
    complete = False
    _start_meter()
    try:
        if not PROTOCOL_PATH.is_file():
            prepare()
        verify_protocol()
        run_tests()
        smoke()
        benchmark(workers)
        run_donors(workers)
        select_donors()
        run_futures(workers)
        analyze()
        report()
        audit = verify(full_replay=True, workers=workers)
        complete = bool(audit["complete"])
    except SoftStop as error:
        _stop_meter()
        _update_status(state="incomplete_walltime_soft_stop", stage="soft-stop", message=str(error))
        print(f"Campaign checkpointed without a verdict: {error}", flush=True)
        return
    except Exception as error:
        _stop_meter()
        _update_status(state="failed", stage="failed", message="pipeline stopped", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        _stop_meter()
    if complete:
        _update_status(state="complete", stage="complete", completed=len(_expected_jobs()), total=len(_expected_jobs()), message="analysis and durable exact replay complete")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "test", "smoke", "select", "analyze", "report", "status", "remaining-hard", "mark-hard-stop"):
        commands.add_parser(name)
    for name in ("benchmark", "donors", "futures", "all"):
        child = commands.add_parser(name)
        child.add_argument("--workers", type=int, default=16)
    verification = commands.add_parser("verify")
    verification.add_argument("--workers", type=int, default=16)
    verification.add_argument("--full-replay", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        prepare()
    elif args.command == "test":
        run_tests()
    elif args.command == "smoke":
        smoke()
    elif args.command == "benchmark":
        print(json.dumps(benchmark(args.workers), indent=2))
    elif args.command == "donors":
        run_donors(args.workers)
    elif args.command == "select":
        print(json.dumps(select_donors(), indent=2))
    elif args.command == "futures":
        run_futures(args.workers)
    elif args.command == "analyze":
        print(json.dumps(analyze(), indent=2))
    elif args.command == "report":
        report()
    elif args.command == "verify":
        print(json.dumps(verify(full_replay=args.full_replay, workers=args.workers), indent=2))
    elif args.command == "status":
        status()
    elif args.command == "remaining-hard":
        remaining_hard()
    elif args.command == "mark-hard-stop":
        mark_hard_stop()
    elif args.command == "all":
        run_all(args.workers)


if __name__ == "__main__":
    main()
