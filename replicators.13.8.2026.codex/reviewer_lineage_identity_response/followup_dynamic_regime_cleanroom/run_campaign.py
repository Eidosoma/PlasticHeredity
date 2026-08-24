"""Run the exploratory, clean-room GARD boundary-of-order campaign."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Callable, Iterable, Sequence


TASK_ROOT = Path(__file__).resolve().parent
CODEX_ROOT = TASK_ROOT.parents[1]
WORKSPACE_ROOT = CODEX_ROOT.parent
if str(CODEX_ROOT) not in sys.path:
    sys.path.insert(0, str(CODEX_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(TASK_ROOT / "artifacts" / "matplotlib"))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

from plastic_heredity.config import CANDIDATES, GardConfig
from plastic_heredity.mechanistic import sha256_file
from plastic_heredity.regime_confirmation import CONFIRMATION_MASTER_SEED
from plastic_heredity.seeds import derive_seed
from plastic_heredity.simulator import SimulationError, cosine_similarity, generate_beta
from reviewer_lineage_identity_response.followup_carrier_cleanroom.carrier_core import (
    ArmPolicy,
    CarrierSetting,
    influence_mask,
    simulate_carrier_future,
    writer_signal,
)
from reviewer_lineage_identity_response.followup_dynamic_regime_cleanroom.regime_core import (
    burn_in_state,
    one_molecule_substitution,
    relax_mean_field,
    scaled_beta,
    scaled_config,
    simulate_ph_lineage,
    simulate_twins,
    tangent_stability_margin,
)


FORMAT = "gard-dynamic-regime-cleanroom-v1"
MASTER_SEED = "c5bd56033592044060bb9a68f2ee727f1dba1397ed112877484cc8d3f7dc1dc"
BASE_GARD = GardConfig()
CANDIDATE_NAMES = ("02", "03")
BETA_MULTIPLIERS = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
LEAVE_MULTIPLIERS = (0.5, 1.0, 2.0)
GRID = tuple(
    (f"b{str(beta).replace('.', 'p')}_l{str(leave).replace('.', 'p')}", beta, leave)
    for leave in LEAVE_MULTIPLIERS
    for beta in BETA_MULTIPLIERS
)
CURRENT_GRID_ID = "b1p0_l1p0"
TWIN_HORIZON = 32
BURN_IN = 64
PH_HORIZON = 32
CARRIER_HORIZON = 32
BOOTSTRAPS = 4_096
SOFT_LIMIT_SECONDS = 27_000.0
HARD_LIMIT_SECONDS = 28_800.0
PROJECTION_BUDGET_SECONDS = 21_600.0

TIERS: dict[str, dict[str, int]] = {
    "A": {
        "development_matrices": 24,
        "confirmation_matrices": 64,
        "development_starts": 8,
        "confirmation_starts": 16,
        "ph_lineages": 48,
        "carrier_futures": 24,
    },
    "B": {
        "development_matrices": 18,
        "confirmation_matrices": 48,
        "development_starts": 6,
        "confirmation_starts": 12,
        "ph_lineages": 32,
        "carrier_futures": 16,
    },
    "C": {
        "development_matrices": 12,
        "confirmation_matrices": 32,
        "development_starts": 4,
        "confirmation_starts": 8,
        "ph_lineages": 24,
        "carrier_futures": 12,
    },
}

ARTIFACT_ROOT = TASK_ROOT / "artifacts"
PROTOCOL_ROOT = ARTIFACT_ROOT / "protocol"
WORK_ROOT = ARTIFACT_ROOT / "work"
REGIME_ROOT = WORK_ROOT / "regime"
PH_ROOT = WORK_ROOT / "ph"
CARRIER_ROOT = WORK_ROOT / "carrier"
OUTPUT_ROOT = ARTIFACT_ROOT / "output"
VERIFICATION_ROOT = ARTIFACT_ROOT / "verification"
STATUS_PATH = ARTIFACT_ROOT / "STATUS.json"
LEDGER_PATH = ARTIFACT_ROOT / "runtime_ledger.json"
PROTOCOL_PATH = PROTOCOL_ROOT / "protocol.json"
REGISTRATION_PATH = PROTOCOL_ROOT / "registration.json"
SOURCE_MANIFEST_PATH = PROTOCOL_ROOT / "scientific_source_manifest.json"
SEED_REGISTRY_PATH = PROTOCOL_ROOT / "seed_registry.json"
BENCHMARK_PATH = PROTOCOL_ROOT / "benchmark.json"
RUNTIME_SELECTION_PATH = PROTOCOL_ROOT / "runtime_selection.json"
REGIME_SELECTION_PATH = PROTOCOL_ROOT / "regime_selection.json"

CARRIER_TASK = TASK_ROOT.parent / "followup_carrier_cleanroom"
CARRIER_SELECTION = CARRIER_TASK / "artifacts" / "protocol" / "confirmation_selection.json"
PARENT_TASK = TASK_ROOT.parent
PARENT_PROTOCOL = PARENT_TASK / "artifacts" / "protocol" / "protocol.json"
PARENT_MATRIX_SELECTION = PARENT_TASK / "artifacts" / "protocol" / "matrix_selection.csv"
PARENT_B_BANK = PARENT_TASK / "artifacts" / "output" / "b_bank.csv"

SOURCE_PATHS = {
    "gard_config": CODEX_ROOT / "plastic_heredity" / "config.py",
    "gard_simulator": CODEX_ROOT / "plastic_heredity" / "simulator.py",
    "gard_process": CODEX_ROOT / "plastic_heredity" / "processes.py",
    "seed_derivation": CODEX_ROOT / "plastic_heredity" / "seeds.py",
    "core": TASK_ROOT / "regime_core.py",
    "runner": TASK_ROOT / "run_campaign.py",
    "tests": TASK_ROOT / "test_regime.py",
    "detached_runner": TASK_ROOT / "run_detached_pipeline.sh",
    "protocol_text": TASK_ROOT / "PROTOCOL.md",
    "reporting_boundary": TASK_ROOT / "REPORTING_BOUNDARY.md",
    "carrier_core": CARRIER_TASK / "carrier_core.py",
    "carrier_selection": CARRIER_SELECTION,
    "parent_protocol": PARENT_PROTOCOL,
    "parent_matrix_selection": PARENT_MATRIX_SELECTION,
    "parent_b_bank": PARENT_B_BANK,
}


class SoftStop(RuntimeError):
    """Raised only after a complete atomic checkpoint."""


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
    encoded = json.dumps(_json_ready(value), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _assert_local(path: Path) -> None:
    path.resolve().relative_to(TASK_ROOT.resolve())


def _write_json(path: Path, value: Any) -> None:
    _assert_local(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(_json_ready(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    _assert_local(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    if not rows:
        temporary.write_text("", encoding="utf-8")
    else:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    os.replace(temporary, path)


def _atomic_npz(path: Path, **values: NDArray) -> None:
    _assert_local(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **values)
    os.replace(temporary, path)


def _load_npz(path: Path) -> dict[str, NDArray]:
    with np.load(path, allow_pickle=False) as bundle:
        return {key: bundle[key] for key in bundle.files}


def _manifest() -> dict[str, Any]:
    entries: dict[str, Any] = {}
    for name, path in sorted(SOURCE_PATHS.items()):
        if not path.is_file():
            raise FileNotFoundError(path)
        entries[name] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    result = {"classification": "scientific_input_and_implementation", "entries": entries}
    result["manifest_id"] = _canonical_digest(result)
    return result


def _write_checksums(directory: Path) -> None:
    files = sorted(path for path in directory.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    lines = [f"{sha256_file(path)}  {path.relative_to(directory)}" for path in files]
    (directory / "SHA256SUMS").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _verify_checksums(directory: Path) -> bool:
    lines = (directory / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    for line in lines:
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


def _firewall_audit() -> dict[str, Any]:
    source = (TASK_ROOT / "run_campaign.py").read_text(encoding="utf-8") + (TASK_ROOT / "regime_core.py").read_text(encoding="utf-8")
    forbidden = ("from " + "NewIdeas", "import " + "NewIdeas")
    hits = [token for token in forbidden if token in source]
    return {
        "passed": not hits,
        "forbidden_import_hits": hits,
        "wagner_code_read_imported_or_executed": False,
        "newideas_role": "hypothesis_only_non_evidentiary",
    }


def _initial_ledger() -> dict[str, Any]:
    return {
        "format": FORMAT,
        "created_at_epoch": time.time(),
        "cumulative_seconds": 0.0,
        "active_started_epoch": None,
        "active_cumulative_at_start": None,
        "soft_limit_seconds": SOFT_LIMIT_SECONDS,
        "hard_limit_seconds": HARD_LIMIT_SECONDS,
        "runs_started": 0,
    }


def _recover_ledger() -> dict[str, Any]:
    ledger = _read_json(LEDGER_PATH) if LEDGER_PATH.is_file() else _initial_ledger()
    active = ledger.get("active_started_epoch")
    if active is not None:
        prior = float(ledger.get("active_cumulative_at_start") or ledger.get("cumulative_seconds", 0.0))
        ledger["cumulative_seconds"] = min(HARD_LIMIT_SECONDS, prior + max(0.0, time.time() - float(active)))
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
    # Status readers must never interpret another live process as an
    # interrupted restart.  Recovery is performed only by _start_meter (or the
    # pre-launch remaining-hard command), while ordinary reads project elapsed
    # time from the active timestamp without mutating the ledger.
    ledger = _read_json(LEDGER_PATH) if LEDGER_PATH.is_file() else _initial_ledger()
    elapsed = float(ledger["cumulative_seconds"])
    if ledger.get("active_started_epoch") is not None:
        elapsed = float(ledger["active_cumulative_at_start"]) + time.time() - float(ledger["active_started_epoch"])
    return min(HARD_LIMIT_SECONDS, elapsed)


def _stop_meter() -> None:
    if not LEDGER_PATH.is_file():
        return
    ledger = _read_json(LEDGER_PATH)
    active = ledger.get("active_started_epoch")
    if active is not None:
        prior = float(ledger.get("active_cumulative_at_start") or 0.0)
        ledger["cumulative_seconds"] = min(HARD_LIMIT_SECONDS, prior + max(0.0, time.time() - float(active)))
        ledger["last_stopped_at_epoch"] = time.time()
        ledger["active_started_epoch"] = None
        ledger["active_cumulative_at_start"] = None
        _write_json(LEDGER_PATH, ledger)


def _check_soft_stop() -> None:
    if _ledger_elapsed() >= SOFT_LIMIT_SECONDS:
        raise SoftStop("cumulative 7.5-hour soft limit reached after checkpoint")


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
    payload = {
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
    }
    _write_json(STATUS_PATH, payload)


def _protocol_payload() -> dict[str, Any]:
    value: dict[str, Any] = {
        "format": FORMAT,
        "status": "frozen_before_any_dynamic_regime_scientific_future",
        "classification": "exploratory_followup_not_current_preprint_evidence",
        "question": "does GARD show a reproducible contraction-to-expansion boundary, and is PH or frozen-carrier efficacy enriched nearby?",
        "grid": [
            {"grid_id": grid_id, "beta_multiplier": beta, "leave_multiplier": leave}
            for grid_id, beta, leave in GRID
        ],
        "original_point": CURRENT_GRID_ID,
        "cohorts": {
            "fresh_matrices_not_selected_for_ph": True,
            "tiers": TIERS,
            "development_and_confirmation_seed_domains_disjoint": True,
            "candidates": list(CANDIDATE_NAMES),
        },
        "twins": {
            "burn_in_fissions": BURN_IN,
            "horizon": TWIN_HORIZON,
            "perturbation": "one outcome-blind mass-preserving molecule substitution",
            "coupling": "shared Poisson minimum plus independent residuals; shared molecule-token priorities",
            "primary_damage": "total variation between normalized compositions",
            "corroboration": "one minus cosine similarity",
            "exponent": "log((D8+1/160)/(D0+1/160))/8",
        },
        "selection": {
            "ordered": "minimum pooled development median exponent",
            "expansive": "maximum pooled development median exponent",
            "boundary": "minimum absolute exponent among above-median integrated-damage cells, excluding selected flanks",
            "confirmation_evaluates_all_21_cells": True,
        },
        "ph_overlay": {
            "points": "ordered, boundary, expansive, and original if distinct",
            "lineage_horizon": PH_HORIZON,
            "endpoints": ["F12 break-then-run3", "strict8", "break time", "terminal diversity", "within versus cross similarity"],
            "f12_and_strict8_never_pool_or_rescue_each_other": True,
        },
        "carrier_overlay": {
            "cohort": "previously frozen 47-rule confirmation bank; multiform decoding where eligible",
            "settings": "two settings frozen by prior carrier campaign without retuning",
            "points": "ordered, boundary, expansive",
            "arms": ["correct", "zero", "shuffled"],
            "horizon": CARRIER_HORIZON,
            "not_a_rescue_of_prior_failure": True,
        },
        "decision_rules": {
            "boundary": "ordered 95% CI upper<0, expansive lower>0, connected adjacent sign crossing, both candidates",
            "boundary_susceptibility": "peak integrated damage lies within one Manhattan grid edge of sign crossing",
            "mean_field": "stability margin and damage exponent have negative Spearman association in both candidates",
            "original_adjacent": "original within one edge of zero contour and 90% exponent CI within +/-0.02 in both candidates",
            "ph_enrichment": "boundary minus mean ordered/expansive whole-matrix bootstrap lower>0, separately by endpoint and candidate",
            "carrier_enrichment": "boundary correct-minus-zero positive and exceeds both flank contrasts in both candidates",
        },
        "inference": {
            "unit": "whole catalytic matrix",
            "bootstrap_repetitions": BOOTSTRAPS,
            "candidate_results_separate": True,
            "development_cannot_supply_confirmation_evidence": True,
        },
        "runtime": {
            "benchmark_selects_largest_fitting_registered_tier": True,
            "projection_budget_seconds": PROJECTION_BUDGET_SECONDS,
            "soft_limit_seconds": SOFT_LIMIT_SECONDS,
            "hard_limit_seconds": HARD_LIMIT_SECONDS,
            "restart_does_not_reset_ledger": True,
            "full_exact_checkpoint_replay_required": True,
        },
        "reporting_boundary": {
            "primary_term": "boundary of order",
            "edge_of_chaos_requires_both_stochastic_and_mean_field_gates": True,
            "not_current_preprint_evidence": True,
            "not_independent_gard_replication": True,
            "not_wagner_replication": True,
            "newideas_used_as_data": False,
            "wagner_code_used": False,
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
    _write_json(SOURCE_MANIFEST_PATH, manifest)
    _write_json(
        SEED_REGISTRY_PATH,
        {
            "master_seed": MASTER_SEED,
            "production_domains": ["development", "confirmation", "ph", "carrier", "bootstrap", "replay"],
            "nonproduction_domains": ["test", "smoke", "benchmark"],
        },
    )
    protocol = _protocol_payload()
    _write_json(PROTOCOL_PATH, protocol)
    registration = {
        "format": FORMAT,
        "protocol_id": protocol["protocol_id"],
        "registered_at_epoch": time.time(),
        "source_manifest_id": manifest["manifest_id"],
        "runtime": _runtime(),
        "firewall": firewall,
    }
    registration["registration_id"] = _canonical_digest(registration)
    _write_json(REGISTRATION_PATH, registration)
    _write_checksums(PROTOCOL_ROOT)
    print(json.dumps(registration, indent=2))


def verify_protocol() -> dict[str, Any]:
    protocol = _read_json(PROTOCOL_PATH)
    payload = {key: value for key, value in protocol.items() if key != "protocol_id"}
    if protocol.get("protocol_id") != _canonical_digest(payload):
        raise ValueError("protocol digest mismatch")
    manifest = _read_json(SOURCE_MANIFEST_PATH)
    for entry in manifest["entries"].values():
        if sha256_file(Path(entry["path"])) != entry["sha256"]:
            raise ValueError(f"sealed source changed: {entry['path']}")
    if not _verify_checksums(PROTOCOL_ROOT):
        raise ValueError("protocol checksum verification failed")
    if not _firewall_audit()["passed"]:
        raise ValueError("clean-room firewall failed")
    return protocol


def _matrix_beta(cohort: str, matrix_id: int) -> NDArray:
    rng = np.random.default_rng(derive_seed(MASTER_SEED, f"{cohort}.beta", matrix_id))
    return generate_beta(BASE_GARD, rng)


def _old_beta(matrix_id: int) -> NDArray:
    rng = np.random.default_rng(derive_seed(CONFIRMATION_MASTER_SEED, "REGCONF.beta", matrix_id))
    return generate_beta(BASE_GARD, rng)


def _runtime_selection() -> dict[str, Any]:
    value = _read_json(RUNTIME_SELECTION_PATH)
    payload = {key: item for key, item in value.items() if key != "selection_id"}
    if value["selection_id"] != _canonical_digest(payload):
        raise ValueError("runtime selection digest mismatch")
    return value


def _tier() -> dict[str, int]:
    selection = _runtime_selection()
    return {key: int(value) for key, value in selection["design"].items()}


def _regime_path(cohort: str, candidate: str, matrix_id: int) -> Path:
    return REGIME_ROOT / cohort / f"c{candidate}_m{matrix_id:03d}.npz"


def _regime_task(args: tuple[str, str, int, int]) -> dict[str, NDArray]:
    cohort, candidate, matrix_id, starts = args
    with threadpool_limits(limits=1):
        beta_base = _matrix_beta(cohort, matrix_id)
        shape = (len(GRID), starts)
        exponent = np.full(shape, np.nan, dtype=np.float64)
        cosine_exponent = np.full(shape, np.nan, dtype=np.float64)
        integrated = np.full(shape, np.nan, dtype=np.float64)
        maximum = np.full(shape, np.nan, dtype=np.float64)
        survived = np.zeros(shape, dtype=np.int8)
        coalesced8 = np.zeros(shape, dtype=np.int8)
        coalesced32 = np.zeros(shape, dtype=np.int8)
        saturated = np.zeros(shape, dtype=np.int8)
        completed = np.zeros(shape, dtype=np.int8)
        d_tv = np.full((len(GRID), starts, TWIN_HORIZON + 1), np.nan, dtype=np.float64)
        d_cosine = np.full_like(d_tv, np.nan)
        left_digest = np.full(shape, "", dtype="U64")
        right_digest = np.full(shape, "", dtype="U64")
        stability = np.full(len(GRID), np.nan, dtype=np.float64)
        residual = np.full(len(GRID), np.nan, dtype=np.float64)
        solver_iterations = np.zeros(len(GRID), dtype=np.int32)
        for grid_index, (grid_id, beta_multiplier, leave_multiplier) in enumerate(GRID):
            beta = scaled_beta(beta_base, beta_multiplier)
            config = scaled_config(BASE_GARD, leave_multiplier)
            burn_states: list[NDArray] = []
            for start_index in range(starts):
                try:
                    state = burn_in_state(
                        config,
                        CANDIDATES[candidate],
                        beta,
                        seed=derive_seed(MASTER_SEED, f"{cohort}.burn", candidate, matrix_id, grid_id, start_index),
                        generations=BURN_IN,
                    )
                    perturb_rng = np.random.default_rng(
                        derive_seed(MASTER_SEED, f"{cohort}.perturb", candidate, matrix_id, grid_id, start_index)
                    )
                    stranger = one_molecule_substitution(state, perturb_rng)
                    readout = simulate_twins(
                        state,
                        stranger,
                        beta,
                        config,
                        CANDIDATES[candidate],
                        seed=derive_seed(MASTER_SEED, f"{cohort}.twins", candidate, matrix_id, grid_id, start_index),
                        horizon=TWIN_HORIZON,
                    )
                except SimulationError:
                    continue
                burn_states.append(state)
                exponent[grid_index, start_index] = readout.exponent_1_8
                cosine_exponent[grid_index, start_index] = readout.cosine_exponent_1_8
                integrated[grid_index, start_index] = readout.integrated_damage
                maximum[grid_index, start_index] = readout.maximum_damage
                survived[grid_index, start_index] = int(readout.survival_f32)
                coalesced8[grid_index, start_index] = int(readout.coalesced_by_8)
                coalesced32[grid_index, start_index] = int(readout.coalesced_by_32)
                saturated[grid_index, start_index] = int(readout.saturated_f32)
                completed[grid_index, start_index] = 1
                d_tv[grid_index, start_index] = readout.damage_tv
                d_cosine[grid_index, start_index] = readout.damage_cosine
                left_digest[grid_index, start_index] = readout.left_digest
                right_digest[grid_index, start_index] = readout.right_digest
            if burn_states:
                mean_start = np.mean(np.asarray(burn_states, dtype=np.float64), axis=0)
                form, iterations, flow_residual = relax_mean_field(
                    mean_start, beta, config.k_join, config.k_leave
                )
                solver_iterations[grid_index] = iterations
                residual[grid_index] = flow_residual
                stability[grid_index] = tangent_stability_margin(
                    form, beta, config.k_join, config.k_leave
                )
        return {
            "grid_ids": np.asarray([row[0] for row in GRID]),
            "exponent": exponent,
            "cosine_exponent": cosine_exponent,
            "integrated_damage": integrated,
            "maximum_damage": maximum,
            "survived": survived,
            "coalesced8": coalesced8,
            "coalesced32": coalesced32,
            "saturated": saturated,
            "completed": completed,
            "damage_tv": d_tv,
            "damage_cosine": d_cosine,
            "left_digest": left_digest,
            "right_digest": right_digest,
            "stability_margin": stability,
            "flow_residual": residual,
            "solver_iterations": solver_iterations,
        }


def _save_regime_task(args: tuple[str, str, int, int]) -> str:
    cohort, candidate, matrix_id, _ = args
    path = _regime_path(cohort, candidate, matrix_id)
    if not path.is_file():
        _atomic_npz(path, **_regime_task(args))
    return str(path)


def _run_tasks(
    function: Callable[[Any], str],
    tasks: Sequence[Any],
    *,
    workers: int,
    stage: str,
) -> None:
    pending = list(tasks)
    started = time.time()
    _update_status(state="running", stage=stage, total=len(pending), started_at=started)
    completed = 0
    if workers <= 1:
        iterator: Iterable[str] = map(function, pending)
        for result in iterator:
            completed += 1
            _update_status(state="running", stage=stage, completed=completed, total=len(pending), started_at=started, message=Path(result).name)
            _check_soft_stop()
        return
    for offset in range(0, len(pending), workers):
        batch = pending[offset : offset + workers]
        with ProcessPoolExecutor(max_workers=min(workers, len(batch))) as executor:
            results = list(executor.map(function, batch, chunksize=1))
        for result in results:
            completed += 1
            _update_status(state="running", stage=stage, completed=completed, total=len(pending), started_at=started, message=Path(result).name)
        _check_soft_stop()


def _benchmark_worker(index: int) -> float:
    started = time.time()
    beta = _matrix_beta("benchmark", index)
    config = BASE_GARD
    for candidate in CANDIDATE_NAMES:
        for start_index in range(2):
            state = burn_in_state(
                config,
                CANDIDATES[candidate],
                beta,
                seed=derive_seed(MASTER_SEED, "benchmark.burn", index, candidate, start_index),
                generations=16,
            )
            stranger = one_molecule_substitution(
                state, np.random.default_rng(derive_seed(MASTER_SEED, "benchmark.perturb", index, candidate, start_index))
            )
            simulate_twins(
                state,
                stranger,
                beta,
                config,
                CANDIDATES[candidate],
                seed=derive_seed(MASTER_SEED, "benchmark.twins", index, candidate, start_index),
                horizon=8,
            )
    return time.time() - started


def _tier_units(design: dict[str, int]) -> float:
    regime_pairs = (
        2
        * len(GRID)
        * (
            design["development_matrices"] * design["development_starts"]
            + design["confirmation_matrices"] * design["confirmation_starts"]
        )
    )
    regime_fissions = regime_pairs * (BURN_IN + TWIN_HORIZON)
    points = 4
    ph_fissions = 2 * design["confirmation_matrices"] * points * design["ph_lineages"] * PH_HORIZON
    carrier_fissions = 2 * 47 * 3 * 2 * 3 * design["carrier_futures"] * CARRIER_HORIZON
    # Full replay repeats every stochastic checkpoint.  Carrier is cheaper per
    # fission than the coupled twin path, so count it at half weight.
    return 2.0 * (regime_fissions + ph_fissions + 0.5 * carrier_fissions)


def benchmark(workers: int) -> dict[str, Any]:
    verify_protocol()
    if RUNTIME_SELECTION_PATH.is_file():
        return _runtime_selection()
    sample_workers = min(max(1, workers), 4)
    started = time.time()
    if sample_workers == 1:
        durations = [_benchmark_worker(0)]
    else:
        with ProcessPoolExecutor(max_workers=sample_workers) as executor:
            durations = list(executor.map(_benchmark_worker, range(sample_workers)))
    wall = time.time() - started
    benchmark_units = sample_workers * 2 * 2 * (16 + 8)
    seconds_per_unit_per_worker = sum(durations) / benchmark_units
    effective_workers = max(1.0, min(float(workers), 16.0) * 0.70)
    projections: list[dict[str, Any]] = []
    chosen: str | None = None
    for tier_name, design in TIERS.items():
        projected = seconds_per_unit_per_worker * _tier_units(design) / effective_workers * 1.5 + 900.0
        row = {"tier": tier_name, "projected_seconds": projected, "fits": projected <= PROJECTION_BUDGET_SECONDS}
        projections.append(row)
        if chosen is None and row["fits"]:
            chosen = tier_name
    if chosen is None:
        chosen = "C"
    benchmark_payload = {
        "format": FORMAT,
        "workers": workers,
        "sample_workers": sample_workers,
        "wall_seconds": wall,
        "worker_durations": durations,
        "seconds_per_weighted_fission": seconds_per_unit_per_worker,
        "projections": projections,
    }
    _write_json(BENCHMARK_PATH, benchmark_payload)
    selection = {
        "format": FORMAT,
        "protocol_id": _read_json(PROTOCOL_PATH)["protocol_id"],
        "benchmark_sha256": sha256_file(BENCHMARK_PATH),
        "tier": chosen,
        "design": TIERS[chosen],
        "selection_rule": "largest preregistered tier projected within six hours; C is minimum authorized tier",
        "selected_before_scientific_futures": True,
    }
    selection["selection_id"] = _canonical_digest(selection)
    _write_json(RUNTIME_SELECTION_PATH, selection)
    _write_checksums(PROTOCOL_ROOT)
    return selection


def run_regime(cohort: str, workers: int) -> None:
    verify_protocol()
    design = _tier()
    matrices = design[f"{cohort}_matrices"]
    starts = design[f"{cohort}_starts"]
    tasks = [
        (cohort, candidate, matrix_id, starts)
        for candidate in CANDIDATE_NAMES
        for matrix_id in range(matrices)
    ]
    _run_tasks(_save_regime_task, tasks, workers=workers, stage=f"regime-{cohort}")


def _development_grid_rows() -> list[dict[str, Any]]:
    design = _tier()
    rows: list[dict[str, Any]] = []
    for candidate in CANDIDATE_NAMES:
        for matrix_id in range(design["development_matrices"]):
            data = _load_npz(_regime_path("development", candidate, matrix_id))
            for grid_index, (grid_id, beta, leave) in enumerate(GRID):
                values = data["exponent"][grid_index]
                integrated = data["integrated_damage"][grid_index]
                rows.append(
                    {
                        "candidate": candidate,
                        "matrix_id": matrix_id,
                        "grid_id": grid_id,
                        "beta_multiplier": beta,
                        "leave_multiplier": leave,
                        "exponent": float(np.nanmean(values)),
                        "integrated_damage": float(np.nanmean(integrated)),
                    }
                )
    return rows


def select_regimes() -> dict[str, Any]:
    if REGIME_SELECTION_PATH.is_file():
        return _read_json(REGIME_SELECTION_PATH)
    rows = _development_grid_rows()
    table = pd.DataFrame(rows)
    pooled = table.groupby("grid_id", sort=False)[["exponent", "integrated_damage"]].median().reset_index()
    order_lookup = {grid_id: index for index, (grid_id, _, _) in enumerate(GRID)}
    pooled["order"] = pooled["grid_id"].map(order_lookup)
    ordered = pooled.sort_values(["exponent", "order"]).iloc[0]
    expansive = pooled.sort_values(["exponent", "order"], ascending=[False, True]).iloc[0]
    threshold = float(pooled["integrated_damage"].median())
    candidates = pooled[
        (pooled["integrated_damage"] >= threshold)
        & (~pooled["grid_id"].isin([ordered["grid_id"], expansive["grid_id"]]))
    ].copy()
    candidates["absolute_exponent"] = candidates["exponent"].abs()
    boundary = candidates.sort_values(["absolute_exponent", "order"]).iloc[0]
    selected = {
        "ordered": str(ordered["grid_id"]),
        "boundary": str(boundary["grid_id"]),
        "expansive": str(expansive["grid_id"]),
    }
    overlay = list(dict.fromkeys([*selected.values(), CURRENT_GRID_ID]))
    value = {
        "format": FORMAT,
        "protocol_id": _read_json(PROTOCOL_PATH)["protocol_id"],
        "runtime_selection_id": _runtime_selection()["selection_id"],
        "selected": selected,
        "ph_overlay_grid_ids": overlay,
        "carrier_overlay_grid_ids": [selected["ordered"], selected["boundary"], selected["expansive"]],
        "development_rows": len(rows),
        "integrated_damage_median_threshold": threshold,
        "selected_before_confirmation_futures": True,
    }
    value["selection_id"] = _canonical_digest(value)
    _write_json(REGIME_SELECTION_PATH, value)
    _write_csv(PROTOCOL_ROOT / "development_grid_summary.csv", pooled.to_dict(orient="records"))
    _write_checksums(PROTOCOL_ROOT)
    return value


def _grid_row(grid_id: str) -> tuple[str, float, float]:
    matches = [row for row in GRID if row[0] == grid_id]
    if len(matches) != 1:
        raise ValueError(grid_id)
    return matches[0]


def _ph_path(candidate: str, matrix_id: int) -> Path:
    return PH_ROOT / f"c{candidate}_m{matrix_id:03d}.npz"


def _ph_task(args: tuple[str, int, int]) -> dict[str, NDArray]:
    candidate, matrix_id, lineages = args
    selection = _read_json(REGIME_SELECTION_PATH)
    points = [str(value) for value in selection["ph_overlay_grid_ids"]]
    beta_base = _matrix_beta("confirmation", matrix_id)
    shape = (len(points), lineages)
    f12 = np.zeros(shape, dtype=np.int8)
    strict8 = np.zeros(shape, dtype=np.int8)
    breaks = np.zeros(shape, dtype=np.int8)
    break_time = np.full(shape, -1, dtype=np.int16)
    inherited = np.zeros(shape, dtype=np.int16)
    within = np.full(shape, np.nan, dtype=np.float64)
    terminals = np.zeros((len(points), lineages, BASE_GARD.n_types), dtype=np.int16)
    completed = np.zeros(shape, dtype=np.int8)
    with threadpool_limits(limits=1):
        for point_index, grid_id in enumerate(points):
            _, beta_multiplier, leave_multiplier = _grid_row(grid_id)
            beta = scaled_beta(beta_base, beta_multiplier)
            config = scaled_config(BASE_GARD, leave_multiplier)
            for lineage in range(lineages):
                try:
                    result, terminal = simulate_ph_lineage(
                        beta,
                        config,
                        CANDIDATES[candidate],
                        seed=derive_seed(MASTER_SEED, "ph.lineage", candidate, matrix_id, grid_id, lineage),
                        horizon=PH_HORIZON,
                    )
                except SimulationError:
                    continue
                f12[point_index, lineage] = int(result.f12)
                strict8[point_index, lineage] = int(result.strict8)
                breaks[point_index, lineage] = int(result.break_event)
                break_time[point_index, lineage] = result.break_time
                inherited[point_index, lineage] = result.inherited_boundaries
                within[point_index, lineage] = result.terminal_within8_h
                terminals[point_index, lineage] = terminal
                completed[point_index, lineage] = 1
    return {
        "grid_ids": np.asarray(points),
        "f12": f12,
        "strict8": strict8,
        "break_event": breaks,
        "break_time": break_time,
        "inherited_boundaries": inherited,
        "within8_h": within,
        "terminals": terminals,
        "completed": completed,
    }


def _save_ph_task(args: tuple[str, int, int]) -> str:
    candidate, matrix_id, _ = args
    path = _ph_path(candidate, matrix_id)
    if not path.is_file():
        _atomic_npz(path, **_ph_task(args))
    return str(path)


def run_ph(workers: int) -> None:
    design = _tier()
    tasks = [
        (candidate, matrix_id, design["ph_lineages"])
        for candidate in CANDIDATE_NAMES
        for matrix_id in range(design["confirmation_matrices"])
    ]
    _run_tasks(_save_ph_task, tasks, workers=workers, stage="ph-overlay")


def _carrier_settings() -> list[CarrierSetting]:
    selection = _read_json(CARRIER_SELECTION)
    return [
        CarrierSetting(
            k=int(row["k"]),
            half_life=int(row["half_life"]),
            coupling=float(row["coupling"]),
            copy_mode=str(row["copy_mode"]),
        )
        for row in selection["selected_settings"]
    ]


def _carrier_rules() -> list[int]:
    table = pd.read_csv(PARENT_MATRIX_SELECTION)
    return [int(value) for value in table["matrix_id"] if int(value) not in (11, 54, 63)]


def _carrier_forms(candidate: str, matrix_id: int) -> tuple[NDArray, NDArray | None]:
    table = pd.read_csv(PARENT_B_BANK)
    cell = table[
        (table["candidate"].astype(str).str.zfill(2) == candidate)
        & (table["matrix_id"] == matrix_id)
        & (table["kind"] == "strict")
    ].sort_values("bank_index")
    if cell.empty:
        raise ValueError(f"missing carrier target c{candidate} m{matrix_id}")
    forms = [np.asarray(json.loads(value), dtype=np.int64) for value in cell["final_B"]]
    target = forms[0]
    eligible = [(cosine_similarity(target, form), form) for form in forms[1:]]
    eligible = [row for row in eligible if row[0] <= 0.85]
    other = min(eligible, key=lambda row: row[0])[1] if eligible else None
    return target, other


def _carrier_path(candidate: str, matrix_id: int) -> Path:
    return CARRIER_ROOT / f"c{candidate}_m{matrix_id:03d}.npz"


def _carrier_task(args: tuple[str, int, int]) -> dict[str, NDArray]:
    candidate, matrix_id, futures = args
    points = [str(value) for value in _read_json(REGIME_SELECTION_PATH)["carrier_overlay_grid_ids"]]
    settings = _carrier_settings()
    arms = ("correct", "zero", "shuffled")
    shape = (len(settings), len(points), len(arms), futures)
    terminal8 = np.full(shape, -1, dtype=np.int8)
    origin_correct = np.full(shape, -1, dtype=np.int8)
    carrier_origin_correct = np.full(shape, -1, dtype=np.int8)
    final_h = np.full(shape, np.nan, dtype=np.float64)
    completed = np.zeros(shape, dtype=np.int8)
    state_digest = np.full(shape, "", dtype="U64")
    target, other = _carrier_forms(candidate, matrix_id)
    beta_base = _old_beta(matrix_id)
    with threadpool_limits(limits=1):
        for setting_index, setting in enumerate(settings):
            mask = influence_mask(beta_base, setting.k)
            correct_signal = writer_signal(target, mask)
            active = np.flatnonzero(mask)
            permutation_rng = np.random.default_rng(
                derive_seed(MASTER_SEED, "carrier.shuffle", candidate, matrix_id, setting.setting_id)
            )
            shuffled_signal = correct_signal.copy()
            shuffled_signal[active] = shuffled_signal[np.asarray(permutation_rng.permutation(active), dtype=int)]
            for point_index, grid_id in enumerate(points):
                _, beta_multiplier, leave_multiplier = _grid_row(grid_id)
                beta = scaled_beta(beta_base, beta_multiplier)
                config = scaled_config(BASE_GARD, leave_multiplier)
                for future in range(futures):
                    dynamics_seed = derive_seed(MASTER_SEED, "carrier.dynamics", candidate, matrix_id, setting.setting_id, grid_id, future)
                    carrier_seed = derive_seed(MASTER_SEED, "carrier.copy", candidate, matrix_id, setting.setting_id, grid_id, future)
                    for arm_index, arm in enumerate(arms):
                        policy = ArmPolicy(arm, initial="correct" if arm == "correct" else "zero")
                        override = shuffled_signal if arm == "shuffled" else None
                        readout, _, _, _ = simulate_carrier_future(
                            target,
                            target,
                            other,
                            beta,
                            config,
                            CANDIDATES[candidate],
                            setting,
                            mask,
                            policy,
                            dynamics_seed=dynamics_seed,
                            carrier_seed=carrier_seed,
                            horizon=CARRIER_HORIZON,
                            initial_override=override,
                        )
                        terminal8[setting_index, point_index, arm_index, future] = readout.terminal8_f32
                        origin_correct[setting_index, point_index, arm_index, future] = readout.origin_correct
                        carrier_origin_correct[setting_index, point_index, arm_index, future] = readout.carrier_origin_correct
                        final_h[setting_index, point_index, arm_index, future] = readout.final_target_h
                        completed[setting_index, point_index, arm_index, future] = int(readout.completed)
                        state_digest[setting_index, point_index, arm_index, future] = readout.state_digest
    return {
        "setting_ids": np.asarray([setting.setting_id for setting in settings]),
        "grid_ids": np.asarray(points),
        "arms": np.asarray(arms),
        "terminal8": terminal8,
        "origin_correct": origin_correct,
        "carrier_origin_correct": carrier_origin_correct,
        "final_target_h": final_h,
        "completed": completed,
        "state_digest": state_digest,
        "has_other": np.asarray([int(other is not None)], dtype=np.int8),
    }


def _save_carrier_task(args: tuple[str, int, int]) -> str:
    candidate, matrix_id, _ = args
    path = _carrier_path(candidate, matrix_id)
    if not path.is_file():
        _atomic_npz(path, **_carrier_task(args))
    return str(path)


def run_carrier(workers: int) -> None:
    design = _tier()
    tasks = [
        (candidate, matrix_id, design["carrier_futures"])
        for candidate in CANDIDATE_NAMES
        for matrix_id in _carrier_rules()
    ]
    _run_tasks(_save_carrier_task, tasks, workers=workers, stage="carrier-overlay")


def _bootstrap(values: NDArray, seed: int, confidence: float = 0.95) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return {"point": float("nan"), "lower": float("nan"), "upper": float("nan"), "n_matrices": 0}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, array.size, size=(BOOTSTRAPS, array.size))
    statistics = array[draws].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "point": float(array.mean()),
        "lower": float(np.quantile(statistics, alpha)),
        "upper": float(np.quantile(statistics, 1.0 - alpha)),
        "n_matrices": int(array.size),
    }


def _adjacent(left_id: str, right_id: str) -> bool:
    lookup = {grid_id: (BETA_MULTIPLIERS.index(beta), LEAVE_MULTIPLIERS.index(leave)) for grid_id, beta, leave in GRID}
    left = lookup[left_id]
    right = lookup[right_id]
    return abs(left[0] - right[0]) + abs(left[1] - right[1]) <= 1


def _regime_analysis() -> tuple[pd.DataFrame, dict[str, Any]]:
    design = _tier()
    rows: list[dict[str, Any]] = []
    for candidate in CANDIDATE_NAMES:
        for matrix_id in range(design["confirmation_matrices"]):
            data = _load_npz(_regime_path("confirmation", candidate, matrix_id))
            for index, (grid_id, beta, leave) in enumerate(GRID):
                rows.append(
                    {
                        "candidate": candidate,
                        "matrix_id": matrix_id,
                        "grid_id": grid_id,
                        "beta_multiplier": beta,
                        "leave_multiplier": leave,
                        "exponent": float(np.nanmean(data["exponent"][index])),
                        "cosine_exponent": float(np.nanmean(data["cosine_exponent"][index])),
                        "integrated_damage": float(np.nanmean(data["integrated_damage"][index])),
                        "coalesced8": float(np.mean(data["coalesced8"][index])),
                        "survival_f32": float(np.mean(data["survived"][index])),
                        "stability_margin": float(data["stability_margin"][index]),
                        "flow_residual": float(data["flow_residual"][index]),
                        "completion": float(np.mean(data["completed"][index])),
                    }
                )
    table = pd.DataFrame(rows)
    selection = _read_json(REGIME_SELECTION_PATH)["selected"]
    cells: list[dict[str, Any]] = []
    verdicts: dict[str, Any] = {}
    for candidate in CANDIDATE_NAMES:
        candidate_table = table[table["candidate"] == candidate]
        point_summary: dict[str, Any] = {}
        for grid_id, _, _ in GRID:
            cell = candidate_table[candidate_table["grid_id"] == grid_id]
            exponent_ci = _bootstrap(cell["exponent"].to_numpy(), derive_seed(MASTER_SEED, "bootstrap.regime", candidate, grid_id))
            integrated_ci = _bootstrap(cell["integrated_damage"].to_numpy(), derive_seed(MASTER_SEED, "bootstrap.integrated", candidate, grid_id))
            row = {"candidate": candidate, "grid_id": grid_id, **{f"exponent_{key}": value for key, value in exponent_ci.items()}, **{f"integrated_{key}": value for key, value in integrated_ci.items()}}
            cells.append(row)
            point_summary[grid_id] = row
        ordered = point_summary[selection["ordered"]]
        expansive = point_summary[selection["expansive"]]
        signs = {grid_id: np.sign(point_summary[grid_id]["exponent_point"]) for grid_id, _, _ in GRID}
        crossing_cells = sorted(
            {
                left
                for left, _, _ in GRID
                for right, _, _ in GRID
                if _adjacent(left, right) and signs[left] * signs[right] <= 0
            }
        )
        peak = max(point_summary, key=lambda key: point_summary[key]["integrated_point"])
        current = point_summary[CURRENT_GRID_ID]
        current90 = _bootstrap(
            candidate_table[candidate_table["grid_id"] == CURRENT_GRID_ID]["exponent"].to_numpy(),
            derive_seed(MASTER_SEED, "bootstrap.current90", candidate),
            confidence=0.90,
        )
        association = spearmanr(candidate_table["exponent"], candidate_table["stability_margin"], nan_policy="omit")
        verdicts[candidate] = {
            "ordered_ci_below_zero": bool(ordered["exponent_upper"] < 0.0),
            "expansive_ci_above_zero": bool(expansive["exponent_lower"] > 0.0),
            "crossing_cells": crossing_cells,
            "integrated_damage_peak": peak,
            "susceptibility_near_crossing": bool(any(_adjacent(peak, cell) for cell in crossing_cells)),
            "current_90_ci": current90,
            "current_near_crossing": bool(any(_adjacent(CURRENT_GRID_ID, cell) for cell in crossing_cells)),
            "current_equivalent_zero": bool(current90["lower"] >= -0.02 and current90["upper"] <= 0.02),
            "stability_exponent_spearman": float(association.statistic),
            "stability_association_negative": bool(association.statistic < 0.0),
            "completion_minimum": float(candidate_table["completion"].min()),
            "current_point": current,
        }
    summary = {
        "selection": selection,
        "candidates": verdicts,
        "boundary_detected_both": all(
            value["ordered_ci_below_zero"] and value["expansive_ci_above_zero"] and bool(value["crossing_cells"]) and value["susceptibility_near_crossing"]
            for value in verdicts.values()
        ),
        "mean_field_corroborates_both": all(value["stability_association_negative"] for value in verdicts.values()),
        "original_boundary_adjacent_both": all(value["current_near_crossing"] and value["current_equivalent_zero"] for value in verdicts.values()),
    }
    summary["edge_of_chaos_language_authorized"] = bool(summary["boundary_detected_both"] and summary["mean_field_corroborates_both"])
    return table, {"cells": cells, "summary": summary}


def _ph_analysis() -> tuple[pd.DataFrame, dict[str, Any]]:
    design = _tier()
    selection = _read_json(REGIME_SELECTION_PATH)
    points = [str(value) for value in selection["ph_overlay_grid_ids"]]
    rows: list[dict[str, Any]] = []
    for candidate in CANDIDATE_NAMES:
        for matrix_id in range(design["confirmation_matrices"]):
            data = _load_npz(_ph_path(candidate, matrix_id))
            for index, grid_id in enumerate(points):
                valid = data["completed"][index].astype(bool)
                terminals = data["terminals"][index][valid]
                cross_values = [
                    cosine_similarity(terminals[left], terminals[right])
                    for left in range(len(terminals))
                    for right in range(left + 1, len(terminals))
                ]
                rows.append(
                    {
                        "candidate": candidate,
                        "matrix_id": matrix_id,
                        "grid_id": grid_id,
                        "f12_rate": float(np.mean(data["f12"][index][valid])) if np.any(valid) else np.nan,
                        "strict8_rate": float(np.mean(data["strict8"][index][valid])) if np.any(valid) else np.nan,
                        "break_rate": float(np.mean(data["break_event"][index][valid])) if np.any(valid) else np.nan,
                        "mean_break_time": float(np.mean(data["break_time"][index][valid & (data["break_time"][index] > 0)])) if np.any(valid & (data["break_time"][index] > 0)) else np.nan,
                        "within8_h": float(np.nanmean(data["within8_h"][index][valid])) if np.any(valid) else np.nan,
                        "cross_lineage_h": float(np.mean(cross_values)) if cross_values else np.nan,
                        "terminal_clusters_h90": _cluster_count(terminals, 0.90),
                        "completion": float(np.mean(valid)),
                    }
                )
    table = pd.DataFrame(rows)
    selected = selection["selected"]
    summary: dict[str, Any] = {"candidates": {}}
    for candidate in CANDIDATE_NAMES:
        cell = table[table["candidate"] == candidate]
        candidate_result: dict[str, Any] = {}
        for endpoint in ("f12_rate", "strict8_rate"):
            by_point = {
                grid_id: _bootstrap(
                    cell[cell["grid_id"] == grid_id][endpoint].to_numpy(),
                    derive_seed(MASTER_SEED, "bootstrap.ph", candidate, endpoint, grid_id),
                )
                for grid_id in points
            }
            pivot = cell.pivot(index="matrix_id", columns="grid_id", values=endpoint)
            contrast = pivot[selected["boundary"]] - 0.5 * (
                pivot[selected["ordered"]] + pivot[selected["expansive"]]
            )
            contrast_ci = _bootstrap(
                contrast.to_numpy(), derive_seed(MASTER_SEED, "bootstrap.ph.contrast", candidate, endpoint)
            )
            candidate_result[endpoint] = {
                "points": by_point,
                "boundary_minus_flanks": contrast_ci,
                "enrichment_pass": bool(contrast_ci["lower"] > 0.0),
            }
        summary["candidates"][candidate] = candidate_result
    summary["f12_enrichment_both"] = all(value["f12_rate"]["enrichment_pass"] for value in summary["candidates"].values())
    summary["strict8_enrichment_both"] = all(value["strict8_rate"]["enrichment_pass"] for value in summary["candidates"].values())
    return table, summary


def _cluster_count(states: NDArray, threshold: float) -> int:
    representatives: list[NDArray] = []
    for state in np.asarray(states):
        if not any(cosine_similarity(state, representative) > threshold for representative in representatives):
            representatives.append(state)
    return len(representatives)


def _carrier_analysis() -> tuple[pd.DataFrame, dict[str, Any]]:
    points = [str(value) for value in _read_json(REGIME_SELECTION_PATH)["carrier_overlay_grid_ids"]]
    settings = _carrier_settings()
    arms = ("correct", "zero", "shuffled")
    rows: list[dict[str, Any]] = []
    for candidate in CANDIDATE_NAMES:
        for matrix_id in _carrier_rules():
            data = _load_npz(_carrier_path(candidate, matrix_id))
            for setting_index, setting in enumerate(settings):
                for point_index, grid_id in enumerate(points):
                    for arm_index, arm in enumerate(arms):
                        valid = data["completed"][setting_index, point_index, arm_index].astype(bool)
                        terminal = data["terminal8"][setting_index, point_index, arm_index]
                        origin = data["carrier_origin_correct"][setting_index, point_index, arm_index]
                        rows.append(
                            {
                                "candidate": candidate,
                                "matrix_id": matrix_id,
                                "setting_id": setting.setting_id,
                                "grid_id": grid_id,
                                "arm": arm,
                                "terminal8_rate": float(np.mean(terminal[valid] == 1)) if np.any(valid) else np.nan,
                                "carrier_origin_accuracy": float(np.mean(origin[valid & (origin >= 0)])) if np.any(valid & (origin >= 0)) else np.nan,
                                "has_other": int(data["has_other"][0]),
                                "completion": float(np.mean(valid)),
                            }
                        )
    table = pd.DataFrame(rows)
    selected = _read_json(REGIME_SELECTION_PATH)["selected"]
    summary: dict[str, Any] = {"settings": {}}
    for setting in settings:
        setting_result: dict[str, Any] = {}
        for candidate in CANDIDATE_NAMES:
            cell = table[(table["setting_id"] == setting.setting_id) & (table["candidate"] == candidate)]
            pivot = cell.pivot(index="matrix_id", columns=["grid_id", "arm"], values="terminal8_rate")
            effects = {
                grid_id: pivot[(grid_id, "correct")] - pivot[(grid_id, "zero")]
                for grid_id in points
            }
            effect_ci = {
                grid_id: _bootstrap(values.to_numpy(), derive_seed(MASTER_SEED, "bootstrap.carrier", setting.setting_id, candidate, grid_id))
                for grid_id, values in effects.items()
            }
            boundary = effects[selected["boundary"]]
            vs_ordered = _bootstrap(
                (boundary - effects[selected["ordered"]]).to_numpy(),
                derive_seed(MASTER_SEED, "bootstrap.carrier.did.ordered", setting.setting_id, candidate),
            )
            vs_expansive = _bootstrap(
                (boundary - effects[selected["expansive"]]).to_numpy(),
                derive_seed(MASTER_SEED, "bootstrap.carrier.did.expansive", setting.setting_id, candidate),
            )
            eligible = cell[(cell["grid_id"] == selected["boundary"]) & (cell["arm"] == "correct") & (cell["has_other"] == 1)]
            decoding = _bootstrap(
                eligible["carrier_origin_accuracy"].to_numpy(),
                derive_seed(MASTER_SEED, "bootstrap.carrier.decoding", setting.setting_id, candidate),
            )
            setting_result[candidate] = {
                "correct_minus_zero": effect_ci,
                "boundary_minus_ordered_effect": vs_ordered,
                "boundary_minus_expansive_effect": vs_expansive,
                "boundary_decoding": decoding,
                "enrichment_pass": bool(effect_ci[selected["boundary"]]["lower"] > 0 and vs_ordered["lower"] > 0 and vs_expansive["lower"] > 0 and decoding["lower"] > 0.5),
            }
        setting_result["enrichment_both_candidates"] = all(setting_result[candidate]["enrichment_pass"] for candidate in CANDIDATE_NAMES)
        summary["settings"][setting.setting_id] = setting_result
    summary["any_setting_enriched_both"] = any(value["enrichment_both_candidates"] for value in summary["settings"].values())
    return table, summary


def analyze() -> dict[str, Any]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    regime_table, regime = _regime_analysis()
    ph_table, ph = _ph_analysis()
    carrier_table, carrier = _carrier_analysis()
    regime_table.to_csv(OUTPUT_ROOT / "regime_matrix_cells.csv", index=False)
    pd.DataFrame(regime["cells"]).to_csv(OUTPUT_ROOT / "regime_cell_intervals.csv", index=False)
    ph_table.to_csv(OUTPUT_ROOT / "ph_matrix_cells.csv", index=False)
    carrier_table.to_csv(OUTPUT_ROOT / "carrier_matrix_cells.csv", index=False)
    summary = {
        "format": FORMAT,
        "protocol_id": _read_json(PROTOCOL_PATH)["protocol_id"],
        "runtime_selection_id": _runtime_selection()["selection_id"],
        "regime_selection_id": _read_json(REGIME_SELECTION_PATH)["selection_id"],
        "complete": True,
        "regime": regime["summary"],
        "ph": ph,
        "carrier": carrier,
        "reporting_boundary": "exploratory; not current-preprint evidence; not an independent GARD or Wagner replication",
    }
    _write_json(OUTPUT_ROOT / "primary_summary.json", summary)
    _plot_regime(pd.DataFrame(regime["cells"]))
    _write_checksums(OUTPUT_ROOT)
    return summary


def _plot_regime(cells: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for axis, candidate in zip(axes, CANDIDATE_NAMES, strict=True):
        cell = cells[cells["candidate"] == candidate]
        matrix = cell.pivot(index="leave_multiplier" if "leave_multiplier" in cell else "grid_id", columns="grid_id", values="exponent_point") if False else None
        values = np.full((len(LEAVE_MULTIPLIERS), len(BETA_MULTIPLIERS)), np.nan)
        for _, row in cell.iterrows():
            _, beta, leave = _grid_row(str(row["grid_id"]))
            values[LEAVE_MULTIPLIERS.index(leave), BETA_MULTIPLIERS.index(beta)] = row["exponent_point"]
        image = axis.imshow(values, aspect="auto", cmap="coolwarm", vmin=-np.nanmax(np.abs(values)), vmax=np.nanmax(np.abs(values)))
        axis.set_title(f"Candidate {candidate}: damage exponent")
        axis.set_xticks(range(len(BETA_MULTIPLIERS)), labels=BETA_MULTIPLIERS, rotation=45)
        axis.set_yticks(range(len(LEAVE_MULTIPLIERS)), labels=LEAVE_MULTIPLIERS)
        axis.set_xlabel("catalytic multiplier")
        axis.set_ylabel("leave-rate multiplier")
    figure.colorbar(image, ax=axes, label="per fission")
    figure.savefig(OUTPUT_ROOT / "dynamic_regime_heatmap.png", dpi=180)
    plt.close(figure)


def report() -> None:
    summary = _read_json(OUTPUT_ROOT / "primary_summary.json")
    regime = summary["regime"]
    ph = summary["ph"]
    carrier = summary["carrier"]
    if regime["edge_of_chaos_language_authorized"]:
        regime_sentence = "The stochastic boundary and mean-field diagnostic both passed; edge-of-chaos language is authorized for this reconstructed model, with the stated finite-size definition."
    elif regime["boundary_detected_both"]:
        regime_sentence = "A stochastic boundary of order was detected, but the independent mean-field requirement did not pass, so edge-of-chaos language is not authorized."
    else:
        regime_sentence = "The preregistered two-diagnostic boundary gate did not pass, so this campaign does not establish an edge of chaos."
    lines = [
        "# Exploratory GARD dynamic-regime results",
        "",
        "## Outcome",
        "",
        regime_sentence,
        f"The original point was boundary-adjacent in both candidates: **{regime['original_boundary_adjacent_both']}**.",
        f"F12 enrichment near the selected boundary passed in both candidates: **{ph['f12_enrichment_both']}**.",
        f"Strict-8 enrichment passed independently in both candidates: **{ph['strict8_enrichment_both']}**.",
        f"Any frozen carrier setting showed boundary-specific enrichment in both candidates: **{carrier['any_setting_enriched_both']}**.",
        "",
        "## Candidate diagnostics",
        "",
        "| Candidate | Ordered < 0 | Expansive > 0 | Susceptibility near crossing | Stability agrees | Original near/equivalent |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for candidate, value in regime["candidates"].items():
        lines.append(
            f"| {candidate} | {value['ordered_ci_below_zero']} | {value['expansive_ci_above_zero']} | {value['susceptibility_near_crossing']} | {value['stability_association_negative']} | {value['current_near_crossing'] and value['current_equivalent_zero']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This is an outcome-informed exploratory follow-up in the reconstructed GARD simulator. It is not evidence for the current preprint, not an independent GARD replication, and not a Wagner replication. The weaker F12 endpoint and rare strict-8 endpoint are reported separately. The carrier overlay reused frozen settings and cannot rescue the earlier failed constructive-memory gate.",
        ]
    )
    (OUTPUT_ROOT / "TECHNICAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    lay = [
        "# Lay summary",
        "",
        regime_sentence,
        "We nudged otherwise identical molecular assemblies by one molecule and watched whether that tiny difference disappeared or grew under matched randomness. We then checked whether the forms of plastic heredity used in earlier work became more common around the transition.",
        f"The present operating point qualified as close to that transition in both simulator versions: **{regime['original_boundary_adjacent_both']}**.",
        f"The weaker F12 pattern peaked there in both versions: **{ph['f12_enrichment_both']}**. The rarer strict-eight pattern did so independently: **{ph['strict8_enrichment_both']}**.",
        f"The previously frozen carrier worked especially well there under the demanding registered comparison: **{carrier['any_setting_enriched_both']}**.",
        "These are exploratory simulator results and do not change what the current preprint claims.",
    ]
    (OUTPUT_ROOT / "LAY_SUMMARY.md").write_text("\n".join(lay) + "\n", encoding="utf-8")
    _write_checksums(OUTPUT_ROOT)


def _expected_jobs() -> list[tuple[str, Any, Path]]:
    design = _tier()
    jobs: list[tuple[str, Any, Path]] = []
    for cohort in ("development", "confirmation"):
        for candidate in CANDIDATE_NAMES:
            for matrix_id in range(design[f"{cohort}_matrices"]):
                args = (cohort, candidate, matrix_id, design[f"{cohort}_starts"])
                jobs.append(("regime", args, _regime_path(cohort, candidate, matrix_id)))
    for candidate in CANDIDATE_NAMES:
        for matrix_id in range(design["confirmation_matrices"]):
            args = (candidate, matrix_id, design["ph_lineages"])
            jobs.append(("ph", args, _ph_path(candidate, matrix_id)))
    for candidate in CANDIDATE_NAMES:
        for matrix_id in _carrier_rules():
            args = (candidate, matrix_id, design["carrier_futures"])
            jobs.append(("carrier", args, _carrier_path(candidate, matrix_id)))
    return jobs


def _replay_job(job: tuple[str, Any, Path]) -> dict[str, Any]:
    kind, args, path = job
    expected = _load_npz(path)
    observed = _regime_task(args) if kind == "regime" else _ph_task(args) if kind == "ph" else _carrier_task(args)
    def arrays_equal(left: NDArray, right: NDArray) -> bool:
        if left.dtype.kind in "fc" or right.dtype.kind in "fc":
            return bool(np.array_equal(left, right, equal_nan=True))
        return bool(np.array_equal(left, right))

    exact = set(expected) == set(observed) and all(
        arrays_equal(expected[key], observed[key]) for key in expected
    )
    return {"kind": kind, "path": str(path), "sha256": sha256_file(path), "exact": bool(exact)}


def verify(*, full_replay: bool, workers: int) -> dict[str, Any]:
    verify_protocol()
    jobs = _expected_jobs()
    missing = [str(path) for _, _, path in jobs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} checkpoints")
    rows: list[dict[str, Any]] = []
    if full_replay:
        started = time.time()
        _update_status(state="running", stage="verify-replay", total=len(jobs), started_at=started)
        for offset in range(0, len(jobs), max(1, workers)):
            batch = jobs[offset : offset + max(1, workers)]
            if workers <= 1:
                results = [_replay_job(job) for job in batch]
            else:
                with ProcessPoolExecutor(max_workers=min(workers, len(batch))) as executor:
                    results = list(executor.map(_replay_job, batch, chunksize=1))
            rows.extend(results)
            _update_status(state="running", stage="verify-replay", completed=len(rows), total=len(jobs), started_at=started, message=Path(results[-1]["path"]).name)
            _check_soft_stop()
    complete = bool(
        full_replay
        and all(row["exact"] for row in rows)
        and _verify_checksums(PROTOCOL_ROOT)
        and _verify_checksums(OUTPUT_ROOT)
        and _firewall_audit()["passed"]
    )
    audit = {
        "format": FORMAT,
        "protocol_id": _read_json(PROTOCOL_PATH)["protocol_id"],
        "full_replay": full_replay,
        "checkpoint_count": len(jobs),
        "replayed_checkpoint_count": len(rows),
        "all_arrays_exact": bool(rows and all(row["exact"] for row in rows)),
        "cleanroom_firewall": _firewall_audit(),
        "protocol_checksums": _verify_checksums(PROTOCOL_ROOT),
        "output_checksums": _verify_checksums(OUTPUT_ROOT),
        "complete": complete,
        "checkpoints": rows,
    }
    _write_json(VERIFICATION_ROOT / "verification_audit.json", audit)
    _write_checksums(VERIFICATION_ROOT)
    if full_replay and not complete:
        raise ValueError("full replay verification failed")
    return audit


def smoke() -> None:
    beta = _matrix_beta("smoke", 0)
    for candidate in CANDIDATE_NAMES:
        state = burn_in_state(
            BASE_GARD,
            CANDIDATES[candidate],
            beta,
            seed=derive_seed(MASTER_SEED, "smoke.burn", candidate),
            generations=4,
        )
        identical = simulate_twins(
            state,
            state,
            beta,
            BASE_GARD,
            CANDIDATES[candidate],
            seed=derive_seed(MASTER_SEED, "smoke.identical", candidate),
            horizon=8,
        )
        if not identical.identical_path or identical.left_digest != identical.right_digest or np.any(identical.damage_tv != 0.0):
            raise AssertionError(f"identical twin coupling failed for {candidate}")
        perturb = one_molecule_substitution(state, np.random.default_rng(derive_seed(MASTER_SEED, "smoke.perturb", candidate)))
        first = simulate_twins(state, perturb, beta, BASE_GARD, CANDIDATES[candidate], seed=31, horizon=4)
        second = simulate_twins(state, perturb, beta, BASE_GARD, CANDIDATES[candidate], seed=31, horizon=4)
        if first.left_digest != second.left_digest or not np.array_equal(first.damage_tv, second.damage_tv):
            raise AssertionError("twin replay smoke failed")
    print("Two-candidate coupled-twin smoke passed; production seeds were not consumed.")


def run_tests() -> None:
    subprocess.run([sys.executable, "-m", "pytest", str(TASK_ROOT / "test_regime.py"), "-q"], cwd=CODEX_ROOT, check=True)


def status() -> None:
    if STATUS_PATH.is_file():
        value = _read_json(STATUS_PATH)
    else:
        value = {"format": FORMAT, "state": "not_started", "stage": "none"}
    value["cumulative_elapsed_seconds"] = _ledger_elapsed() if LEDGER_PATH.is_file() else 0.0
    value["soft_remaining_seconds"] = max(0.0, SOFT_LIMIT_SECONDS - value["cumulative_elapsed_seconds"])
    value["hard_remaining_seconds"] = max(0.0, HARD_LIMIT_SECONDS - value["cumulative_elapsed_seconds"])
    print(json.dumps(value, indent=2, sort_keys=True))


def remaining_hard() -> None:
    _recover_ledger()
    print(max(0, int(HARD_LIMIT_SECONDS - _ledger_elapsed())))


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
        run_regime("development", workers)
        select_regimes()
        run_regime("confirmation", workers)
        run_ph(workers)
        run_carrier(workers)
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
        _update_status(state="complete", stage="complete", completed=len(_expected_jobs()), total=len(_expected_jobs()), message="full exact replay and reports complete")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    commands.add_parser("test")
    commands.add_parser("smoke")
    for name in ("benchmark", "development", "confirmation", "ph", "carrier", "all"):
        child = commands.add_parser(name)
        child.add_argument("--workers", type=int, default=16)
    commands.add_parser("select")
    commands.add_parser("analyze")
    commands.add_parser("report")
    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("--workers", type=int, default=16)
    verify_parser.add_argument("--full-replay", action="store_true")
    commands.add_parser("status")
    commands.add_parser("remaining-hard", help=argparse.SUPPRESS)
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
    elif args.command == "development":
        run_regime("development", args.workers)
    elif args.command == "select":
        print(json.dumps(select_regimes(), indent=2))
    elif args.command == "confirmation":
        run_regime("confirmation", args.workers)
    elif args.command == "ph":
        run_ph(args.workers)
    elif args.command == "carrier":
        run_carrier(args.workers)
    elif args.command == "analyze":
        print(json.dumps(analyze(), indent=2))
    elif args.command == "report":
        report()
    elif args.command == "verify":
        print(json.dumps(verify(full_replay=args.full_replay, workers=args.workers), indent=2))
    elif args.command == "all":
        run_all(args.workers)
    elif args.command == "status":
        status()
    elif args.command == "remaining-hard":
        remaining_hard()


if __name__ == "__main__":
    main()
