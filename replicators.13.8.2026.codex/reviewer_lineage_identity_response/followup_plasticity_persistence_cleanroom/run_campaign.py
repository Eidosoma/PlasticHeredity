"""Run the fresh GARD plasticity-persistence and lineage-identity campaign."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
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
from typing import Any, Callable, Iterable, Mapping, Sequence


TASK_ROOT = Path(__file__).resolve().parent
CODEX_ROOT = TASK_ROOT.parents[1]
if str(CODEX_ROOT) not in sys.path:
    sys.path.insert(0, str(CODEX_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(TASK_ROOT / "artifacts" / "matplotlib"))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

from plastic_heredity.config import CANDIDATES, GardConfig, SimulationContract
from plastic_heredity.mechanistic import sha256_file
from plastic_heredity.seeds import derive_seed
from plastic_heredity.simulator import SimulationError, cosine_similarity, generate_beta
from reviewer_lineage_identity_response.followup_dynamic_regime_cleanroom.regime_core import (
    burn_in_state,
    one_molecule_substitution,
    scaled_beta,
    scaled_config,
    simulate_twins,
)
from reviewer_lineage_identity_response.followup_plasticity_persistence_cleanroom.campaign_core import (
    factorial_shapley,
    f12_decomposition,
    last8_coherence,
    reproducible_multiform,
    simulate_detailed_lineage,
    simulate_future_scores,
)


FORMAT = "gard-plasticity-persistence-cleanroom-v1"
MASTER_SEED = "e9b9dce4efce4238c357239abf0aa0d70fa3e65c694c1f23ea661bedc3da47e7"
BASE_GARD = GardConfig()
CANDIDATE_NAMES = ("02", "03")
BETA_MULTIPLIERS = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
LEAVE_MULTIPLIERS = (0.5, 1.0, 2.0)
GRID = tuple(
    (f"b{str(beta).replace('.', 'p')}_l{str(leave).replace('.', 'p')}", beta, leave)
    for leave in LEAVE_MULTIPLIERS
    for beta in BETA_MULTIPLIERS
)
GRID_LOOKUP = {grid_id: (index, beta, leave) for index, (grid_id, beta, leave) in enumerate(GRID)}
ANCHORS = ("b1p0_l0p5", "b1p0_l1p0", "b2p0_l2p0", "b4p0_l2p0")
ARMS = ("native_a", "native_b", "permuted", "natural")
LINEAGE_HORIZON = 32
FUTURE_HORIZON = 32
BOOTSTRAPS = 4_096
PERMUTATION_PROPOSALS = 4_096
SOFT_LIMIT_SECONDS = 27_000.0
HARD_LIMIT_SECONDS = 28_800.0
PROJECTION_BUDGET_SECONDS = 24_300.0

FACTOR_NAMES = ("exposure", "overshoot", "fission", "daughter")
CONTRACTS: tuple[tuple[str, tuple[int, int, int, int], SimulationContract], ...] = tuple(
    (
        f"e{exposure}_o{overshoot}_f{fission}_d{daughter}",
        (exposure, overshoot, fission, daughter),
        SimulationContract(
            name=f"factorial-{exposure}{overshoot}{fission}{daughter}",
            poisson_exposure=(0.10, 0.125)[exposure],
            overshoot_rule=("trim_whole_assembly", "admit_joiners_to_capacity")[overshoot],
            fission_rule=("fixed_size", "binomial")[fission],
            daughter_rule=("first", "second")[daughter],
        ),
    )
    for exposure in (0, 1)
    for overshoot in (0, 1)
    for fission in (0, 1)
    for daughter in (0, 1)
)

TIERS: dict[str, dict[str, int]] = {
    "A": {
        "surface_matrices": 72,
        "surface_lineages": 64,
        "f12_donors": 2,
        "identity_futures": 24,
        "factorial_matrices": 12,
        "factorial_lineages": 12,
        "factorial_twin_starts": 3,
    },
    "B": {
        "surface_matrices": 56,
        "surface_lineages": 48,
        "f12_donors": 2,
        "identity_futures": 20,
        "factorial_matrices": 10,
        "factorial_lineages": 10,
        "factorial_twin_starts": 3,
    },
    "C": {
        "surface_matrices": 40,
        "surface_lineages": 32,
        "f12_donors": 1,
        "identity_futures": 16,
        "factorial_matrices": 8,
        "factorial_lineages": 8,
        "factorial_twin_starts": 2,
    },
}

ARTIFACT_ROOT = TASK_ROOT / "artifacts"
PROTOCOL_ROOT = ARTIFACT_ROOT / "protocol"
WORK_ROOT = ARTIFACT_ROOT / "work"
SURFACE_ROOT = WORK_ROOT / "surface"
IDENTITY_ROOT = WORK_ROOT / "identity"
FACTORIAL_ROOT = WORK_ROOT / "factorial"
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
DONOR_REGISTRY_PATH = PROTOCOL_ROOT / "donor_registry.npz"
DONOR_SELECTION_PATH = PROTOCOL_ROOT / "donor_selection.json"

SOURCE_PATHS = {
    "gard_config": CODEX_ROOT / "plastic_heredity" / "config.py",
    "gard_simulator": CODEX_ROOT / "plastic_heredity" / "simulator.py",
    "gard_process": CODEX_ROOT / "plastic_heredity" / "processes.py",
    "strict8_evaluator": CODEX_ROOT / "plastic_heredity" / "regime_confirmation.py",
    "seed_derivation": CODEX_ROOT / "plastic_heredity" / "seeds.py",
    "dynamic_twin_core": TASK_ROOT.parent / "followup_dynamic_regime_cleanroom" / "regime_core.py",
    "transplant_core": TASK_ROOT.parent / "followup_transplant_arrival_residence" / "transplant_core.py",
    "campaign_core": TASK_ROOT / "campaign_core.py",
    "runner": TASK_ROOT / "run_campaign.py",
    "tests": TASK_ROOT / "test_campaign.py",
    "protocol_text": TASK_ROOT / "PROTOCOL.md",
    "reporting_boundary": TASK_ROOT / "REPORTING_BOUNDARY.md",
    "detached_runner": TASK_ROOT / "run_detached_pipeline.sh",
}


class SoftStop(RuntimeError):
    """Raised after a complete atomic checkpoint at the soft wall-time."""


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


def _write_checksums(directory: Path) -> None:
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


def _write_dataframe(path: Path, table: pd.DataFrame) -> None:
    _assert_local(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    table.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _manifest() -> dict[str, Any]:
    entries: dict[str, Any] = {}
    for name, path in sorted(SOURCE_PATHS.items()):
        if not path.is_file():
            raise FileNotFoundError(path)
        entries[name] = {"path": str(path.resolve()), "sha256": sha256_file(path), "bytes": path.stat().st_size}
    result = {"classification": "scientific_input_and_implementation", "entries": entries}
    result["manifest_id"] = _canonical_digest(result)
    return result


def _runtime() -> dict[str, str]:
    result = {"python": platform.python_version()}
    for package in ("numpy", "pandas", "scipy", "threadpoolctl"):
        result[package] = importlib.metadata.version(package)
    return result


def _firewall_audit() -> dict[str, Any]:
    source = (TASK_ROOT / "run_campaign.py").read_text(encoding="utf-8") + (TASK_ROOT / "campaign_core.py").read_text(encoding="utf-8")
    forbidden = ("from " + "NewIdeas", "import " + "NewIdeas")
    hits = [token for token in forbidden if token in source]
    return {
        "passed": not hits,
        "forbidden_import_hits": hits,
        "wagner_code_read_imported_or_executed": False,
        "prior_outcomes_role": "hypothesis_generation_only_not_pooled",
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
        ledger["cumulative_seconds"] = min(HARD_LIMIT_SECONDS, prior + max(0.0, time.time() - float(ledger["active_started_epoch"])))
        ledger["last_stopped_at_epoch"] = time.time()
        ledger["active_started_epoch"] = None
        ledger["active_cumulative_at_start"] = None
        _write_json(LEDGER_PATH, ledger)


def _check_soft_stop() -> None:
    if _ledger_elapsed() >= SOFT_LIMIT_SECONDS:
        raise SoftStop("cumulative 7.5-hour soft limit reached after checkpoint")


def _update_status(
    *, state: str, stage: str, completed: int = 0, total: int = 0,
    started_at: float | None = None, message: str = "", error: str | None = None,
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
        "status": "frozen_before_any_scientific_future",
        "classification": "next_preprint_exploratory_followup",
        "question": "does weak F12 plasticity trade against strict persistent form, and does high F12 transmit lineage identity or only increase churn?",
        "surface": {
            "grid": [{"grid_id": row[0], "beta_multiplier": row[1], "leave_multiplier": row[2]} for row in GRID],
            "fresh_unselected_matrices": True,
            "candidates": list(CANDIDATE_NAMES),
            "lineage_horizon": LINEAGE_HORIZON,
            "endpoints": ["break_by_f12", "run3_given_break", "joint_f12", "run8", "coherent8", "distinct8", "registered_strict8", "last8_coherence", "cross_lineage_similarity", "stable_multiform"],
        },
        "fixed_contrast": {
            "high_f12": "b2p0_l2p0",
            "flanks": ["b1p0_l0p5", "b4p0_l2p0"],
            "original": "b1p0_l1p0",
            "prior_result_role": "hypothesis_generation_only",
        },
        "identity": {
            "anchors": list(ANCHORS),
            "f12_donor_selection": "first N outcome-eligible lineage indices by semantic hash",
            "strict8_donors": "one separately labelled donor per available matrix-anchor cell",
            "b_definition_f12": "third daughter completing earliest post-break run3 within F12",
            "b_definition_strict8": "eighth daughter completing earliest registered strict8 window",
            "permuted_stranger": "first of 4096 seeded permutations with H<=0.85, otherwise minimum-H proposal",
            "natural_stranger": "different eligible lineage with closest mass, hash tie break",
            "arms": list(ARMS),
            "future_horizon": FUTURE_HORIZON,
        },
        "attractor_census": {
            "stable": "strict minimum pairwise last8 H>0.90",
            "cluster_edge": "terminal H>0.90",
            "multiform": "two clusters each >=5% starts, centroids H<=0.85, reproduced across even/odd halves",
        },
        "factorial": {
            "factors": {
                "exposure": [0.10, 0.125],
                "overshoot": ["trim_whole_assembly", "admit_joiners_to_capacity"],
                "fission": ["fixed_size", "binomial"],
                "daughter": ["first", "second"],
            },
            "contracts": [{"contract_id": row[0], "bits": row[1]} for row in CONTRACTS],
            "full_21_cell_surface": True,
            "endpoints": ["joint_f12", "registered_strict8", "terminal_diversity", "coupled_twin_damage_exponent"],
            "candidate_corner_parity_required": True,
            "attribution": "exact four-factor Shapley decomposition",
        },
        "gates": {
            "tradeoff": "within-matrix 21-cell F12-versus-strict8 Spearman 95% upper<0 in both candidates, plus high-F12 fixed contrast F12 lower>0 and strict8 upper<0",
            "lineage_identity": "native residence exceeds both strangers; same-minus-cross F8 terminal similarity lower>0.10; same-parent similarity lower>0.90, both candidates",
            "shared_destination": "permuted-stranger F16 capture lower>0.40 and native-minus-permuted 90% interval within +/-0.10",
            "transient_churn": "native and stranger F32 capture upper<0.25",
            "dominant_contract_source": "absolute Shapley share point>0.50 and lower>0.35 with replicate-half direction agreement",
        },
        "tiers": TIERS,
        "inference": {
            "unit": "whole catalytic matrix",
            "bootstrap_repetitions": BOOTSTRAPS,
            "candidates_separate": True,
            "f12_and_strict8_never_pool_or_rescue_each_other": True,
        },
        "runtime": {
            "benchmark_selects_largest_fitting_tier": True,
            "projection_budget_seconds": PROJECTION_BUDGET_SECONDS,
            "soft_limit_seconds": SOFT_LIMIT_SECONDS,
            "hard_limit_seconds": HARD_LIMIT_SECONDS,
            "restart_does_not_reset_ledger": True,
            "full_exact_checkpoint_replay_required": True,
        },
        "reporting_boundary": {
            "not_current_preprint_evidence": True,
            "not_independent_gard_replication": True,
            "not_wagner_replication": True,
            "edge_of_chaos_not_a_success_criterion": True,
            "prior_results_not_pooled": True,
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
            "production_domains": ["surface.beta", "surface.lineage", "identity.selection", "identity.future", "factorial.beta", "factorial.lineage", "factorial.twins", "bootstrap", "replay"],
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
        raise ValueError("protocol checksums failed")
    if not _firewall_audit()["passed"]:
        raise ValueError("clean-room firewall failed")
    return protocol


def _matrix_beta(domain: str, matrix_id: int) -> NDArray:
    rng = np.random.default_rng(derive_seed(MASTER_SEED, f"{domain}.beta", matrix_id))
    return generate_beta(BASE_GARD, rng)


def _runtime_selection() -> dict[str, Any]:
    value = _read_json(RUNTIME_SELECTION_PATH)
    payload = {key: item for key, item in value.items() if key != "selection_id"}
    if value["selection_id"] != _canonical_digest(payload):
        raise ValueError("runtime selection digest mismatch")
    return value


def _design() -> dict[str, int]:
    return {key: int(value) for key, value in _runtime_selection()["design"].items()}


def _surface_path(candidate: str, matrix_id: int) -> Path:
    return SURFACE_ROOT / f"c{candidate}_m{matrix_id:03d}.npz"


def _empty_surface(lineages: int) -> dict[str, NDArray]:
    shape = (len(GRID), lineages)
    return {
        "f12": np.zeros(shape, dtype=np.int8),
        "break12": np.zeros(shape, dtype=np.int8),
        "recovery3": np.zeros(shape, dtype=np.int8),
        "first_break": np.full(shape, -1, dtype=np.int8),
        "first_run3_end": np.full(shape, -1, dtype=np.int8),
        "run8": np.zeros(shape, dtype=np.int8),
        "coherent8": np.zeros(shape, dtype=np.int8),
        "distinct8": np.zeros(shape, dtype=np.int8),
        "strict8": np.zeros(shape, dtype=np.int8),
        "strict8_onset": np.full(shape, -1, dtype=np.int8),
        "completed": np.zeros(shape, dtype=np.int8),
        "boundary_h": np.full((len(GRID), lineages, LINEAGE_HORIZON), np.nan, dtype=np.float32),
        "last8": np.zeros((len(GRID), lineages, 8, BASE_GARD.n_types), dtype=np.uint8),
        "b_state": np.zeros((len(GRID), lineages, BASE_GARD.n_types), dtype=np.uint8),
        "strict_b_state": np.zeros((len(GRID), lineages, BASE_GARD.n_types), dtype=np.uint8),
    }


def _surface_task(args: tuple[str, int, int]) -> dict[str, NDArray]:
    candidate, matrix_id, lineages = args
    arrays = _empty_surface(lineages)
    beta_base = _matrix_beta("surface", matrix_id)
    with threadpool_limits(limits=1):
        for grid_index, (grid_id, beta_multiplier, leave_multiplier) in enumerate(GRID):
            beta = scaled_beta(beta_base, beta_multiplier)
            config = scaled_config(BASE_GARD, leave_multiplier)
            for lineage in range(lineages):
                outcome = simulate_detailed_lineage(
                    beta,
                    config,
                    CANDIDATES[candidate],
                    seed=derive_seed(MASTER_SEED, "surface.lineage", candidate, matrix_id, grid_id, lineage),
                )
                for name in (
                    "f12", "break12", "recovery3_given_break", "first_break",
                    "first_run3_end", "run8", "coherent8", "distinct8", "strict8",
                    "strict8_onset", "completed",
                ):
                    target = "recovery3" if name == "recovery3_given_break" else name
                    arrays[target][grid_index, lineage] = int(getattr(outcome, name))
                observed = min(LINEAGE_HORIZON, outcome.boundary_h.size)
                arrays["boundary_h"][grid_index, lineage, :observed] = outcome.boundary_h[:observed]
                arrays["last8"][grid_index, lineage] = outcome.last8.astype(np.uint8)
                arrays["b_state"][grid_index, lineage] = outcome.b_state.astype(np.uint8)
                arrays["strict_b_state"][grid_index, lineage] = outcome.strict_b_state.astype(np.uint8)
    arrays["grid_ids"] = np.asarray([row[0] for row in GRID])
    return arrays


def _save_surface_task(args: tuple[str, int, int]) -> str:
    candidate, matrix_id, _ = args
    path = _surface_path(candidate, matrix_id)
    if not path.is_file():
        _atomic_npz(path, **_surface_task(args))
    return str(path)


def _run_tasks(
    function: Callable[[Any], str], tasks: Sequence[Any], *, workers: int, stage: str
) -> None:
    pending = list(tasks)
    started = time.time()
    existing = 0
    _update_status(state="running", stage=stage, completed=0, total=len(pending), started_at=started)
    if workers <= 1:
        for task in pending:
            result = function(task)
            existing += 1
            _update_status(state="running", stage=stage, completed=existing, total=len(pending), started_at=started, message=Path(result).name)
            _check_soft_stop()
        return
    with ProcessPoolExecutor(max_workers=min(workers, len(pending))) as executor:
        futures = {executor.submit(function, task): task for task in pending}
        for future in as_completed(futures):
            result = future.result()
            existing += 1
            _update_status(state="running", stage=stage, completed=existing, total=len(pending), started_at=started, message=Path(result).name)
            _check_soft_stop()


def run_surface(workers: int) -> None:
    verify_protocol()
    design = _design()
    tasks = [
        (candidate, matrix_id, design["surface_lineages"])
        for candidate in CANDIDATE_NAMES
        for matrix_id in range(design["surface_matrices"])
    ]
    _run_tasks(_save_surface_task, tasks, workers=workers, stage="surface")


def _proposal_permutation(state: NDArray, *, seed: int) -> tuple[NDArray, float, bool]:
    values = np.asarray(state)
    rng = np.random.default_rng(seed)
    best: NDArray | None = None
    best_h = float("inf")
    for _ in range(PERMUTATION_PROPOSALS):
        proposal = np.asarray(rng.permutation(values.size), dtype=np.int16)
        h = cosine_similarity(values, values[proposal])
        if h < best_h:
            best = proposal
            best_h = h
        if h <= 0.85:
            return proposal, float(h), True
    assert best is not None
    return best, float(best_h), False


def _selection_rank(candidate: str, matrix_id: int, grid_id: str, kind: str, lineage: int) -> str:
    return hashlib.sha256(
        f"{MASTER_SEED}|identity.selection|{candidate}|{matrix_id}|{grid_id}|{kind}|{lineage}".encode()
    ).hexdigest()


def build_donor_registry() -> dict[str, Any]:
    if DONOR_REGISTRY_PATH.is_file() and DONOR_SELECTION_PATH.is_file():
        return _read_json(DONOR_SELECTION_PATH)
    design = _design()
    rows: list[dict[str, Any]] = []
    for candidate in CANDIDATE_NAMES:
        for matrix_id in range(design["surface_matrices"]):
            surface = _load_npz(_surface_path(candidate, matrix_id))
            for grid_id in ANCHORS:
                grid_index = GRID_LOOKUP[grid_id][0]
                specifications = (
                    ("f12", np.flatnonzero(surface["f12"][grid_index] == 1), design["f12_donors"], surface["b_state"][grid_index]),
                    ("strict8", np.flatnonzero(surface["strict8"][grid_index] == 1), 1, surface["strict_b_state"][grid_index]),
                )
                for kind, eligible, maximum, state_bank in specifications:
                    ranked = sorted(
                        (int(index) for index in eligible),
                        key=lambda lineage: _selection_rank(candidate, matrix_id, grid_id, kind, lineage),
                    )
                    for donor_order, lineage in enumerate(ranked[:maximum]):
                        target = state_bank[lineage].astype(np.uint8)
                        other_candidates = [int(index) for index in eligible if int(index) != lineage]
                        if other_candidates:
                            mass = int(target.sum())
                            other_candidates.sort(
                                key=lambda index: (
                                    abs(int(state_bank[index].sum()) - mass),
                                    _selection_rank(candidate, matrix_id, grid_id, f"{kind}.natural", index),
                                )
                            )
                            natural_lineage = other_candidates[0]
                            natural = state_bank[natural_lineage].astype(np.uint8)
                            natural_active = True
                        else:
                            natural_lineage = -1
                            natural = np.zeros(BASE_GARD.n_types, dtype=np.uint8)
                            natural_active = False
                        permutation, permutation_h, reached = _proposal_permutation(
                            target,
                            seed=derive_seed(MASTER_SEED, "identity.permutation", candidate, matrix_id, grid_id, kind, lineage),
                        )
                        rows.append(
                            {
                                "candidate": candidate,
                                "matrix_id": matrix_id,
                                "grid_id": grid_id,
                                "grid_index": grid_index,
                                "kind": kind,
                                "donor_order": donor_order,
                                "lineage": lineage,
                                "natural_lineage": natural_lineage,
                                "natural_active": natural_active,
                                "target": target,
                                "natural": natural,
                                "permutation": permutation,
                                "permutation_h": permutation_h,
                                "permutation_reached_0p85": reached,
                            }
                        )
    if not rows:
        raise ValueError("no identity donors selected")
    _atomic_npz(
        DONOR_REGISTRY_PATH,
        candidate=np.asarray([row["candidate"] for row in rows]),
        matrix_id=np.asarray([row["matrix_id"] for row in rows], dtype=np.int16),
        grid_id=np.asarray([row["grid_id"] for row in rows]),
        grid_index=np.asarray([row["grid_index"] for row in rows], dtype=np.int8),
        kind=np.asarray([row["kind"] for row in rows]),
        donor_order=np.asarray([row["donor_order"] for row in rows], dtype=np.int8),
        lineage=np.asarray([row["lineage"] for row in rows], dtype=np.int16),
        natural_lineage=np.asarray([row["natural_lineage"] for row in rows], dtype=np.int16),
        natural_active=np.asarray([row["natural_active"] for row in rows], dtype=np.int8),
        target=np.asarray([row["target"] for row in rows], dtype=np.uint8),
        natural=np.asarray([row["natural"] for row in rows], dtype=np.uint8),
        permutation=np.asarray([row["permutation"] for row in rows], dtype=np.int16),
        permutation_h=np.asarray([row["permutation_h"] for row in rows], dtype=np.float64),
        permutation_reached_0p85=np.asarray([row["permutation_reached_0p85"] for row in rows], dtype=np.int8),
    )
    summary = {
        "format": FORMAT,
        "protocol_id": _read_json(PROTOCOL_PATH)["protocol_id"],
        "runtime_selection_id": _runtime_selection()["selection_id"],
        "status": "frozen_after_surface_before_identity_futures",
        "rows": len(rows),
        "f12_rows": sum(row["kind"] == "f12" for row in rows),
        "strict8_rows": sum(row["kind"] == "strict8" for row in rows),
        "natural_active_rows": sum(bool(row["natural_active"]) for row in rows),
        "permutation_reached_fraction": float(np.mean([row["permutation_reached_0p85"] for row in rows])),
        "registry_sha256": sha256_file(DONOR_REGISTRY_PATH),
    }
    summary["selection_id"] = _canonical_digest(summary)
    _write_json(DONOR_SELECTION_PATH, summary)
    _write_checksums(PROTOCOL_ROOT)
    return summary


def _donor_rows(candidate: str, matrix_id: int) -> list[dict[str, Any]]:
    data = _load_npz(DONOR_REGISTRY_PATH)
    selected = np.flatnonzero((data["candidate"] == candidate) & (data["matrix_id"] == matrix_id))
    return [
        {
            "grid_id": str(data["grid_id"][index]),
            "grid_index": int(data["grid_index"][index]),
            "kind": str(data["kind"][index]),
            "donor_order": int(data["donor_order"][index]),
            "lineage": int(data["lineage"][index]),
            "natural_lineage": int(data["natural_lineage"][index]),
            "natural_active": bool(data["natural_active"][index]),
            "target": data["target"][index].astype(np.int64),
            "natural": data["natural"][index].astype(np.int64),
            "permutation": data["permutation"][index].astype(np.int64),
        }
        for index in selected
    ]


def _identity_path(candidate: str, matrix_id: int) -> Path:
    return IDENTITY_ROOT / f"c{candidate}_m{matrix_id:03d}.npz"


def _identity_task(args: tuple[str, int, int]) -> dict[str, NDArray]:
    candidate, matrix_id, futures = args
    donors = _donor_rows(candidate, matrix_id)
    beta_base = _matrix_beta("surface", matrix_id)
    donor_count = len(donors)
    shape = (donor_count, len(ARMS), futures)
    active = np.ones((donor_count, len(ARMS)), dtype=np.int8)
    capture16 = np.zeros(shape, dtype=np.int8)
    capture32 = np.zeros(shape, dtype=np.int8)
    arrival8 = np.zeros(shape, dtype=np.int8)
    occupancy = np.full(shape, np.nan, dtype=np.float32)
    residence = np.zeros(shape, dtype=np.int8)
    departed = np.zeros(shape, dtype=np.int8)
    reentered = np.zeros(shape, dtype=np.int8)
    observed = np.zeros(shape, dtype=np.int8)
    checkpoint_state = np.zeros((*shape, 3, BASE_GARD.n_types), dtype=np.uint8)
    terminal_target_h = np.full(shape, np.nan, dtype=np.float32)
    target = np.zeros((donor_count, BASE_GARD.n_types), dtype=np.uint8)
    start = np.zeros((donor_count, len(ARMS), BASE_GARD.n_types), dtype=np.uint8)
    with threadpool_limits(limits=1):
        for donor_index, donor in enumerate(donors):
            grid_id = donor["grid_id"]
            _, beta_multiplier, leave_multiplier = GRID[donor["grid_index"]]
            beta = scaled_beta(beta_base, beta_multiplier)
            config = scaled_config(BASE_GARD, leave_multiplier)
            target[donor_index] = donor["target"].astype(np.uint8)
            starts = {
                "native_a": donor["target"],
                "native_b": donor["target"],
                "permuted": donor["target"][donor["permutation"]],
                "natural": donor["natural"],
            }
            if not donor["natural_active"]:
                active[donor_index, ARMS.index("natural")] = 0
            for arm_index, arm in enumerate(ARMS):
                start[donor_index, arm_index] = starts[arm].astype(np.uint8)
                if not active[donor_index, arm_index]:
                    continue
                for future in range(futures):
                    seed_domain = "identity.native_b" if arm == "native_b" else "identity.common"
                    seed = derive_seed(
                        MASTER_SEED,
                        seed_domain,
                        candidate,
                        matrix_id,
                        grid_id,
                        donor["kind"],
                        donor["lineage"],
                        future,
                    )
                    score, checkpoints, _, count = simulate_future_scores(
                        starts[arm],
                        donor["target"],
                        beta,
                        config,
                        CANDIDATES[candidate],
                        seed=seed,
                        horizon=FUTURE_HORIZON,
                    )
                    capture16[donor_index, arm_index, future] = int(score.capture_f16)
                    capture32[donor_index, arm_index, future] = int(score.capture_f32)
                    arrival8[donor_index, arm_index, future] = int(score.arrival_f8)
                    occupancy[donor_index, arm_index, future] = score.occupancy
                    residence[donor_index, arm_index, future] = score.maximum_residence
                    departed[donor_index, arm_index, future] = int(score.departed)
                    reentered[donor_index, arm_index, future] = int(score.reentered)
                    observed[donor_index, arm_index, future] = count
                    checkpoint_state[donor_index, arm_index, future] = checkpoints.astype(np.uint8)
                    terminal_target_h[donor_index, arm_index, future] = cosine_similarity(checkpoints[-1], donor["target"])
    return {
        "arms": np.asarray(ARMS),
        "grid_id": np.asarray([row["grid_id"] for row in donors]),
        "kind": np.asarray([row["kind"] for row in donors]),
        "donor_order": np.asarray([row["donor_order"] for row in donors], dtype=np.int8),
        "lineage": np.asarray([row["lineage"] for row in donors], dtype=np.int16),
        "natural_lineage": np.asarray([row["natural_lineage"] for row in donors], dtype=np.int16),
        "active": active,
        "target": target,
        "start": start,
        "capture16": capture16,
        "capture32": capture32,
        "arrival8": arrival8,
        "occupancy": occupancy,
        "maximum_residence": residence,
        "departed": departed,
        "reentered": reentered,
        "observed": observed,
        "checkpoint_state": checkpoint_state,
        "terminal_target_h": terminal_target_h,
    }


def _save_identity_task(args: tuple[str, int, int]) -> str:
    candidate, matrix_id, _ = args
    path = _identity_path(candidate, matrix_id)
    if not path.is_file():
        _atomic_npz(path, **_identity_task(args))
    return str(path)


def run_identity(workers: int) -> None:
    if not DONOR_SELECTION_PATH.is_file():
        raise FileNotFoundError("donors have not been frozen")
    design = _design()
    tasks = [
        (candidate, matrix_id, design["identity_futures"])
        for candidate in CANDIDATE_NAMES
        for matrix_id in range(design["surface_matrices"])
    ]
    _run_tasks(_save_identity_task, tasks, workers=workers, stage="identity")


def _factorial_path(matrix_id: int) -> Path:
    return FACTORIAL_ROOT / f"m{matrix_id:03d}.npz"


def _factorial_task(args: tuple[int, int, int]) -> dict[str, NDArray]:
    matrix_id, lineages, twin_starts = args
    beta_base = _matrix_beta("factorial", matrix_id)
    shape = (len(CONTRACTS), len(GRID), lineages)
    f12 = np.zeros(shape, dtype=np.int8)
    strict8 = np.zeros(shape, dtype=np.int8)
    break12 = np.zeros(shape, dtype=np.int8)
    recovery3 = np.zeros(shape, dtype=np.int8)
    completion = np.zeros(shape, dtype=np.int8)
    terminal = np.zeros((*shape, BASE_GARD.n_types), dtype=np.uint8)
    exponent = np.full((len(CONTRACTS), len(GRID), twin_starts), np.nan, dtype=np.float32)
    twin_completed = np.zeros_like(exponent, dtype=np.int8)
    with threadpool_limits(limits=1):
        for contract_index, (contract_id, _, contract) in enumerate(CONTRACTS):
            for grid_index, (grid_id, beta_multiplier, leave_multiplier) in enumerate(GRID):
                beta = scaled_beta(beta_base, beta_multiplier)
                config = scaled_config(BASE_GARD, leave_multiplier)
                for lineage in range(lineages):
                    outcome = simulate_detailed_lineage(
                        beta,
                        config,
                        contract,
                        seed=derive_seed(MASTER_SEED, "factorial.lineage", matrix_id, grid_id, contract_id, lineage),
                    )
                    f12[contract_index, grid_index, lineage] = int(outcome.f12)
                    strict8[contract_index, grid_index, lineage] = int(outcome.strict8)
                    break12[contract_index, grid_index, lineage] = int(outcome.break12)
                    recovery3[contract_index, grid_index, lineage] = int(outcome.recovery3_given_break)
                    completion[contract_index, grid_index, lineage] = int(outcome.completed)
                    terminal[contract_index, grid_index, lineage] = outcome.last8[-1].astype(np.uint8)
                for start_index in range(twin_starts):
                    try:
                        state = burn_in_state(
                            config,
                            contract,
                            beta,
                            seed=derive_seed(MASTER_SEED, "factorial.burn", matrix_id, grid_id, contract_id, start_index),
                            generations=64,
                        )
                        stranger = one_molecule_substitution(
                            state,
                            np.random.default_rng(derive_seed(MASTER_SEED, "factorial.perturb", matrix_id, grid_id, contract_id, start_index)),
                        )
                        twins = simulate_twins(
                            state,
                            stranger,
                            beta,
                            config,
                            contract,
                            seed=derive_seed(MASTER_SEED, "factorial.twins", matrix_id, grid_id, contract_id, start_index),
                            horizon=32,
                        )
                    except SimulationError:
                        continue
                    exponent[contract_index, grid_index, start_index] = twins.exponent_1_8
                    twin_completed[contract_index, grid_index, start_index] = 1
    return {
        "contract_ids": np.asarray([row[0] for row in CONTRACTS]),
        "contract_bits": np.asarray([row[1] for row in CONTRACTS], dtype=np.int8),
        "grid_ids": np.asarray([row[0] for row in GRID]),
        "f12": f12,
        "strict8": strict8,
        "break12": break12,
        "recovery3": recovery3,
        "completed": completion,
        "terminal": terminal,
        "damage_exponent": exponent,
        "twin_completed": twin_completed,
    }


def _save_factorial_task(args: tuple[int, int, int]) -> str:
    matrix_id, _, _ = args
    path = _factorial_path(matrix_id)
    if not path.is_file():
        _atomic_npz(path, **_factorial_task(args))
    return str(path)


def run_factorial(workers: int) -> None:
    design = _design()
    tasks = [
        (matrix_id, design["factorial_lineages"], design["factorial_twin_starts"])
        for matrix_id in range(design["factorial_matrices"])
    ]
    _run_tasks(_save_factorial_task, tasks, workers=min(workers, len(tasks)), stage="factorial")


def _benchmark_worker(index: int) -> dict[str, float]:
    beta = _matrix_beta("benchmark", index)
    standard_started = time.time()
    for candidate in CANDIDATE_NAMES:
        for lineage in range(4):
            simulate_detailed_lineage(
                beta,
                BASE_GARD,
                CANDIDATES[candidate],
                seed=derive_seed(MASTER_SEED, "benchmark.lineage", index, candidate, lineage),
            )
    standard_seconds = time.time() - standard_started
    twin_started = time.time()
    for candidate in CANDIDATE_NAMES:
        state = burn_in_state(
            BASE_GARD,
            CANDIDATES[candidate],
            beta,
            seed=derive_seed(MASTER_SEED, "benchmark.burn", index, candidate),
            generations=16,
        )
        stranger = one_molecule_substitution(state, np.random.default_rng(derive_seed(MASTER_SEED, "benchmark.perturb", index, candidate)))
        simulate_twins(state, stranger, beta, BASE_GARD, CANDIDATES[candidate], seed=derive_seed(MASTER_SEED, "benchmark.twins", index, candidate), horizon=8)
    twin_seconds = time.time() - twin_started
    return {
        "standard_seconds_per_fission": standard_seconds / (2 * 4 * LINEAGE_HORIZON),
        "twin_seconds_per_weighted_fission": twin_seconds / (2 * (16 + 2 * 8)),
    }


def _tier_projection(design: Mapping[str, int], standard_rate: float, twin_rate: float, workers: int) -> float:
    surface = 2 * design["surface_matrices"] * len(GRID) * design["surface_lineages"] * LINEAGE_HORIZON
    maximum_donors = 2 * design["surface_matrices"] * len(ANCHORS) * (design["f12_donors"] + 1)
    identity = maximum_donors * len(ARMS) * design["identity_futures"] * FUTURE_HORIZON
    factorial = design["factorial_matrices"] * len(CONTRACTS) * len(GRID) * design["factorial_lineages"] * LINEAGE_HORIZON
    twins = design["factorial_matrices"] * len(CONTRACTS) * len(GRID) * design["factorial_twin_starts"] * (64 + 2 * 32)
    effective_workers = max(1.0, min(float(workers), 16.0) * 0.70)
    production = (standard_rate * (surface + identity + factorial) + twin_rate * twins) / effective_workers
    return 2.0 * production * 1.5 + 1_200.0


def benchmark(workers: int) -> dict[str, Any]:
    verify_protocol()
    if RUNTIME_SELECTION_PATH.is_file():
        return _runtime_selection()
    sample_workers = min(max(1, workers), 4)
    started = time.time()
    if sample_workers == 1:
        samples = [_benchmark_worker(0)]
    else:
        with ProcessPoolExecutor(max_workers=sample_workers) as executor:
            samples = list(executor.map(_benchmark_worker, range(sample_workers)))
    standard_rate = float(np.mean([row["standard_seconds_per_fission"] for row in samples]))
    twin_rate = float(np.mean([row["twin_seconds_per_weighted_fission"] for row in samples]))
    projections: list[dict[str, Any]] = []
    selected: str | None = None
    for tier_name, design in TIERS.items():
        seconds = _tier_projection(design, standard_rate, twin_rate, workers)
        row = {"tier": tier_name, "projected_seconds": seconds, "fits": seconds <= PROJECTION_BUDGET_SECONDS}
        projections.append(row)
        if selected is None and row["fits"]:
            selected = tier_name
    if selected is None:
        selected = "C"
    benchmark_payload = {
        "format": FORMAT,
        "workers": workers,
        "sample_workers": sample_workers,
        "wall_seconds": time.time() - started,
        "standard_seconds_per_fission": standard_rate,
        "twin_seconds_per_weighted_fission": twin_rate,
        "samples": samples,
        "projections": projections,
    }
    _write_json(BENCHMARK_PATH, benchmark_payload)
    selection = {
        "format": FORMAT,
        "protocol_id": _read_json(PROTOCOL_PATH)["protocol_id"],
        "benchmark_sha256": sha256_file(BENCHMARK_PATH),
        "tier": selected,
        "design": TIERS[selected],
        "selected_before_scientific_futures": True,
    }
    selection["selection_id"] = _canonical_digest(selection)
    _write_json(RUNTIME_SELECTION_PATH, selection)
    _write_checksums(PROTOCOL_ROOT)
    return selection


def _bootstrap(values: NDArray, *, seed: int, confidence: float = 0.95) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return {"point": np.nan, "lower": np.nan, "upper": np.nan, "n_matrices": 0}
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


def _mean_cross_similarity(states: NDArray) -> float:
    values = np.asarray(states, dtype=np.float64)
    if values.shape[0] < 2:
        return np.nan
    norms = np.linalg.norm(values, axis=1)
    denominator = np.outer(norms, norms)
    similarities = np.zeros((values.shape[0], values.shape[0]), dtype=np.float64)
    np.divide(values @ values.T, denominator, out=similarities, where=denominator > 0)
    return float(similarities[np.triu_indices(values.shape[0], 1)].mean())


def _surface_matrix_table() -> pd.DataFrame:
    design = _design()
    rows: list[dict[str, Any]] = []
    for candidate in CANDIDATE_NAMES:
        for matrix_id in range(design["surface_matrices"]):
            data = _load_npz(_surface_path(candidate, matrix_id))
            for grid_index, (grid_id, beta, leave) in enumerate(GRID):
                valid = data["completed"][grid_index].astype(bool)
                breaks = data["break12"][grid_index].astype(bool) & valid
                terminals = data["last8"][grid_index, valid, -1]
                multiform, half_a, half_b = reproducible_multiform(data["last8"][grid_index, valid])
                row = {
                    "candidate": candidate,
                    "matrix_id": matrix_id,
                    "grid_id": grid_id,
                    "beta_multiplier": beta,
                    "leave_multiplier": leave,
                    "f12_rate": float(np.mean(data["f12"][grid_index, valid])),
                    "break_rate": float(np.mean(data["break12"][grid_index, valid])),
                    "recovery_given_break": float(np.mean(data["recovery3"][grid_index, breaks])) if np.any(breaks) else np.nan,
                    "run8_rate": float(np.mean(data["run8"][grid_index, valid])),
                    "coherent8_rate": float(np.mean(data["coherent8"][grid_index, valid])),
                    "distinct8_rate": float(np.mean(data["distinct8"][grid_index, valid])),
                    "strict8_rate": float(np.mean(data["strict8"][grid_index, valid])),
                    "last8_coherence": float(np.mean([last8_coherence(block) for block in data["last8"][grid_index, valid]])),
                    "cross_lineage_similarity": _mean_cross_similarity(terminals),
                    "reproducible_multiform": int(multiform),
                    "stable_clusters_half_a": half_a,
                    "stable_clusters_half_b": half_b,
                    "completion": float(np.mean(valid)),
                }
                rows.append(row)
    return pd.DataFrame(rows)


def _surface_summary(table: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {"candidates": {}}
    high = "b2p0_l2p0"
    flanks = ("b1p0_l0p5", "b4p0_l2p0")
    for candidate in CANDIDATE_NAMES:
        cell = table[table["candidate"] == candidate]
        correlations: list[float] = []
        contrasts: dict[str, list[float]] = {"f12": [], "strict8": [], "break_supply": [], "recovery_propensity": [], "decomposition_total": []}
        for matrix_id, matrix in cell.groupby("matrix_id"):
            correlation = spearmanr(matrix["f12_rate"], matrix["strict8_rate"]).statistic
            correlations.append(float(correlation))
            indexed = matrix.set_index("grid_id")
            high_f12 = float(indexed.loc[high, "f12_rate"])
            flank_f12 = float(indexed.loc[list(flanks), "f12_rate"].mean())
            contrasts["f12"].append(high_f12 - flank_f12)
            contrasts["strict8"].append(float(indexed.loc[high, "strict8_rate"] - indexed.loc[list(flanks), "strict8_rate"].mean()))
            break_high = float(indexed.loc[high, "break_rate"])
            break_flank = float(indexed.loc[list(flanks), "break_rate"].mean())
            recovery_high = high_f12 / break_high if break_high > 0 else 0.0
            recovery_flank = flank_f12 / break_flank if break_flank > 0 else 0.0
            total, supply, propensity = f12_decomposition(break_high, recovery_high, break_flank, recovery_flank)
            contrasts["decomposition_total"].append(total)
            contrasts["break_supply"].append(supply)
            contrasts["recovery_propensity"].append(propensity)
        correlation_ci = _bootstrap(np.asarray(correlations), seed=derive_seed(MASTER_SEED, "bootstrap.tradeoff", candidate))
        f12_ci = _bootstrap(np.asarray(contrasts["f12"]), seed=derive_seed(MASTER_SEED, "bootstrap.fixed.f12", candidate))
        strict_ci = _bootstrap(np.asarray(contrasts["strict8"]), seed=derive_seed(MASTER_SEED, "bootstrap.fixed.strict8", candidate))
        candidate_result = {
            "f12_strict8_spearman": correlation_ci,
            "high_minus_flanks_f12": f12_ci,
            "high_minus_flanks_strict8": strict_ci,
            "break_supply_component": _bootstrap(np.asarray(contrasts["break_supply"]), seed=derive_seed(MASTER_SEED, "bootstrap.decomposition.break", candidate)),
            "recovery_propensity_component": _bootstrap(np.asarray(contrasts["recovery_propensity"]), seed=derive_seed(MASTER_SEED, "bootstrap.decomposition.recovery", candidate)),
            "decomposition_total": _bootstrap(np.asarray(contrasts["decomposition_total"]), seed=derive_seed(MASTER_SEED, "bootstrap.decomposition.total", candidate)),
            "tradeoff_pass": bool(correlation_ci["upper"] < 0 and f12_ci["lower"] > 0 and strict_ci["upper"] < 0),
            "multiform_matrix_fraction": {
                grid_id: float(cell[cell["grid_id"] == grid_id]["reproducible_multiform"].mean())
                for grid_id in ANCHORS
            },
            "minimum_completion": float(cell["completion"].min()),
        }
        result["candidates"][candidate] = candidate_result
    result["tradeoff_pass_both"] = all(value["tradeoff_pass"] for value in result["candidates"].values())
    return result


def _identity_matrix_table() -> pd.DataFrame:
    design = _design()
    rows: list[dict[str, Any]] = []
    for candidate in CANDIDATE_NAMES:
        for matrix_id in range(design["surface_matrices"]):
            data = _load_npz(_identity_path(candidate, matrix_id))
            arms = [str(value) for value in data["arms"]]
            for donor in range(data["kind"].size):
                donor_rows = np.flatnonzero(
                    (data["grid_id"] == data["grid_id"][donor])
                    & (data["kind"] == data["kind"][donor])
                    & (np.arange(data["kind"].size) != donor)
                )
                other = int(donor_rows[0]) if donor_rows.size else -1
                native_a = arms.index("native_a")
                native_b = arms.index("native_b")
                natural = arms.index("natural")
                for arm_index, arm in enumerate(arms):
                    if not bool(data["active"][donor, arm_index]):
                        continue
                    rows.append(
                        {
                            "candidate": candidate,
                            "matrix_id": matrix_id,
                            "grid_id": str(data["grid_id"][donor]),
                            "kind": str(data["kind"][donor]),
                            "donor_order": int(data["donor_order"][donor]),
                            "arm": arm,
                            "capture16": float(data["capture16"][donor, arm_index].mean()),
                            "capture32": float(data["capture32"][donor, arm_index].mean()),
                            "arrival8": float(data["arrival8"][donor, arm_index].mean()),
                            "occupancy": float(data["occupancy"][donor, arm_index].mean()),
                            "maximum_residence": float(data["maximum_residence"][donor, arm_index].mean()),
                            "completion": float(np.mean(data["observed"][donor, arm_index] == FUTURE_HORIZON)),
                        }
                    )
                same_values: dict[str, float] = {}
                cross_values: dict[str, float] = {}
                for checkpoint_index, checkpoint in enumerate((8, 16, 32)):
                    same = [
                        cosine_similarity(
                            data["checkpoint_state"][donor, native_a, future, checkpoint_index],
                            data["checkpoint_state"][donor, native_b, future, checkpoint_index],
                        )
                        for future in range(design["identity_futures"])
                    ]
                    if other >= 0:
                        comparison = data["checkpoint_state"][other, native_b, :, checkpoint_index]
                    elif bool(data["active"][donor, natural]):
                        comparison = data["checkpoint_state"][donor, natural, :, checkpoint_index]
                    else:
                        comparison = np.zeros_like(data["checkpoint_state"][donor, native_b, :, checkpoint_index])
                    cross = [
                        cosine_similarity(
                            data["checkpoint_state"][donor, native_a, future, checkpoint_index],
                            comparison[future],
                        )
                        for future in range(design["identity_futures"])
                    ]
                    same_values[f"same_h_f{checkpoint}"] = float(np.mean(same))
                    cross_values[f"cross_h_f{checkpoint}"] = float(np.mean(cross))
                rows.append(
                    {
                        "candidate": candidate,
                        "matrix_id": matrix_id,
                        "grid_id": str(data["grid_id"][donor]),
                        "kind": str(data["kind"][donor]),
                        "donor_order": int(data["donor_order"][donor]),
                        "arm": "fork_comparison",
                        "capture16": np.nan,
                        "capture32": np.nan,
                        "arrival8": np.nan,
                        "occupancy": np.nan,
                        "maximum_residence": np.nan,
                        "completion": np.nan,
                        **same_values,
                        **cross_values,
                    }
                )
    return pd.DataFrame(rows)


def _paired_matrix_effect(table: pd.DataFrame, left: str, right: str, metric: str) -> NDArray:
    pivot = table[table["arm"].isin([left, right])].pivot_table(
        index="matrix_id", columns="arm", values=metric, aggfunc="mean"
    )
    if left not in pivot or right not in pivot:
        return np.empty(0)
    return (pivot[left] - pivot[right]).to_numpy(dtype=np.float64)


def _identity_summary(table: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {"candidates": {}}
    primary_grid = "b2p0_l2p0"
    for candidate in CANDIDATE_NAMES:
        candidate_result: dict[str, Any] = {"anchors": {}}
        for grid_id in ANCHORS:
            cell = table[
                (table["candidate"] == candidate)
                & (table["grid_id"] == grid_id)
                & (table["kind"] == "f12")
            ]
            native_vs_permuted = _bootstrap(
                _paired_matrix_effect(cell, "native_a", "permuted", "maximum_residence"),
                seed=derive_seed(MASTER_SEED, "bootstrap.identity.residence.permuted", candidate, grid_id),
            )
            native_vs_natural = _bootstrap(
                _paired_matrix_effect(cell, "native_a", "natural", "maximum_residence"),
                seed=derive_seed(MASTER_SEED, "bootstrap.identity.residence.natural", candidate, grid_id),
            )
            forks = cell[cell["arm"] == "fork_comparison"].groupby("matrix_id", as_index=False)[["same_h_f8", "cross_h_f8", "same_h_f16", "cross_h_f16", "same_h_f32", "cross_h_f32"]].mean()
            same8 = _bootstrap(forks["same_h_f8"].to_numpy(), seed=derive_seed(MASTER_SEED, "bootstrap.identity.same8", candidate, grid_id))
            difference8 = _bootstrap((forks["same_h_f8"] - forks["cross_h_f8"]).to_numpy(), seed=derive_seed(MASTER_SEED, "bootstrap.identity.diff8", candidate, grid_id))
            permuted = cell[cell["arm"] == "permuted"].groupby("matrix_id")["capture16"].mean().to_numpy()
            native = cell[cell["arm"] == "native_a"].groupby("matrix_id")["capture16"].mean().to_numpy()
            stranger16 = _bootstrap(permuted, seed=derive_seed(MASTER_SEED, "bootstrap.identity.stranger16", candidate, grid_id))
            common = min(native.size, permuted.size)
            equivalence = _bootstrap(native[:common] - permuted[:common], seed=derive_seed(MASTER_SEED, "bootstrap.identity.equivalence", candidate, grid_id), confidence=0.90)
            native32 = cell[cell["arm"] == "native_a"].groupby("matrix_id")["capture32"].mean().to_numpy()
            permuted32 = cell[cell["arm"] == "permuted"].groupby("matrix_id")["capture32"].mean().to_numpy()
            native32_ci = _bootstrap(native32, seed=derive_seed(MASTER_SEED, "bootstrap.identity.native32", candidate, grid_id))
            stranger32_ci = _bootstrap(permuted32, seed=derive_seed(MASTER_SEED, "bootstrap.identity.stranger32", candidate, grid_id))
            anchor_result = {
                "native_minus_permuted_residence": native_vs_permuted,
                "native_minus_natural_residence": native_vs_natural,
                "same_parent_f8": same8,
                "same_minus_cross_f8": difference8,
                "permuted_capture_f16": stranger16,
                "native_minus_permuted_capture_f16_90": equivalence,
                "native_capture_f32": native32_ci,
                "permuted_capture_f32": stranger32_ci,
                "lineage_identity_pass": bool(native_vs_permuted["lower"] > 0 and native_vs_natural["lower"] > 0 and difference8["lower"] > 0.10 and same8["lower"] > 0.90),
                "shared_destination_pass": bool(stranger16["lower"] > 0.40 and equivalence["lower"] >= -0.10 and equivalence["upper"] <= 0.10),
                "transient_churn_pass": bool(native32_ci["upper"] < 0.25 and stranger32_ci["upper"] < 0.25),
            }
            candidate_result["anchors"][grid_id] = anchor_result
        candidate_result["high_f12_lineage_identity"] = candidate_result["anchors"][primary_grid]["lineage_identity_pass"]
        result["candidates"][candidate] = candidate_result
    result["high_f12_lineage_identity_both"] = all(value["high_f12_lineage_identity"] for value in result["candidates"].values())
    return result


def _factorial_matrix_table() -> pd.DataFrame:
    design = _design()
    rows: list[dict[str, Any]] = []
    for matrix_id in range(design["factorial_matrices"]):
        data = _load_npz(_factorial_path(matrix_id))
        for contract_index, (contract_id, bits, _) in enumerate(CONTRACTS):
            for grid_index, (grid_id, beta, leave) in enumerate(GRID):
                valid = data["completed"][contract_index, grid_index].astype(bool)
                terminals = data["terminal"][contract_index, grid_index, valid]
                rows.append(
                    {
                        "matrix_id": matrix_id,
                        "contract_id": contract_id,
                        **{name: bits[index] for index, name in enumerate(FACTOR_NAMES)},
                        "grid_id": grid_id,
                        "beta_multiplier": beta,
                        "leave_multiplier": leave,
                        "f12_rate": float(np.mean(data["f12"][contract_index, grid_index, valid])),
                        "strict8_rate": float(np.mean(data["strict8"][contract_index, grid_index, valid])),
                        "break_rate": float(np.mean(data["break12"][contract_index, grid_index, valid])),
                        "recovery_given_break": float(np.mean(data["recovery3"][contract_index, grid_index, valid & (data["break12"][contract_index, grid_index] == 1)])) if np.any(valid & (data["break12"][contract_index, grid_index] == 1)) else np.nan,
                        "terminal_cross_similarity": _mean_cross_similarity(terminals),
                        "damage_exponent": float(np.nanmean(data["damage_exponent"][contract_index, grid_index])),
                        "completion": float(np.mean(valid)),
                        "twin_completion": float(np.mean(data["twin_completed"][contract_index, grid_index])),
                    }
                )
    return pd.DataFrame(rows)


def _factorial_summary(table: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    shapley_rows: list[dict[str, Any]] = []
    outcomes = ("f12_rate", "strict8_rate", "terminal_cross_similarity", "damage_exponent")
    for matrix_id, matrix in table.groupby("matrix_id"):
        for outcome in outcomes:
            values = {
                tuple(int(row[name]) for name in FACTOR_NAMES): float(row[outcome])
                for _, row in matrix.groupby("contract_id", as_index=False)[[*FACTOR_NAMES, outcome]].mean().iterrows()
            }
            contributions = factorial_shapley(values)
            denominator = float(np.abs(contributions).sum())
            for factor_index, factor in enumerate(FACTOR_NAMES):
                shapley_rows.append(
                    {
                        "matrix_id": int(matrix_id),
                        "outcome": outcome,
                        "factor": factor,
                        "shapley": float(contributions[factor_index]),
                        "absolute_share": float(abs(contributions[factor_index]) / denominator) if denominator > 0 else 0.0,
                    }
                )
    shapley_table = pd.DataFrame(shapley_rows)
    summary: dict[str, Any] = {"outcomes": {}, "candidate_corner_parity": _contract_corner_parity()}
    for outcome in outcomes:
        outcome_result: dict[str, Any] = {"factors": {}}
        subset = shapley_table[shapley_table["outcome"] == outcome]
        for factor in FACTOR_NAMES:
            cell = subset[subset["factor"] == factor].sort_values("matrix_id")
            share = _bootstrap(cell["absolute_share"].to_numpy(), seed=derive_seed(MASTER_SEED, "bootstrap.factorial.share", outcome, factor))
            shapley_ci = _bootstrap(cell["shapley"].to_numpy(), seed=derive_seed(MASTER_SEED, "bootstrap.factorial.shapley", outcome, factor))
            even = float(cell[cell["matrix_id"] % 2 == 0]["shapley"].mean())
            odd = float(cell[cell["matrix_id"] % 2 == 1]["shapley"].mean())
            outcome_result["factors"][factor] = {
                "absolute_share": share,
                "shapley": shapley_ci,
                "replicate_half_direction_agreement": bool(np.sign(even) == np.sign(odd) and np.sign(even) != 0),
                "dominant": bool(share["point"] > 0.50 and share["lower"] > 0.35 and np.sign(even) == np.sign(odd) and np.sign(even) != 0),
            }
        dominant = [factor for factor, value in outcome_result["factors"].items() if value["dominant"]]
        outcome_result["dominant_factors"] = dominant
        summary["outcomes"][outcome] = outcome_result
    summary["distributed_implementation_sensitivity"] = not any(
        value["dominant_factors"] for value in summary["outcomes"].values()
    )
    summary["minimum_completion"] = float(table["completion"].min())
    summary["minimum_twin_completion"] = float(table["twin_completion"].min())
    return shapley_table, summary


def _factorial_effect_table(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    outcomes = ("f12_rate", "strict8_rate", "terminal_cross_similarity", "damage_exponent")
    for matrix_id, matrix in table.groupby("matrix_id"):
        for outcome in outcomes:
            for factor in FACTOR_NAMES:
                effect = float(matrix[matrix[factor] == 1][outcome].mean() - matrix[matrix[factor] == 0][outcome].mean())
                rows.append({"matrix_id": matrix_id, "outcome": outcome, "term": factor, "order": "main", "effect": effect})
            for left_index, left in enumerate(FACTOR_NAMES):
                for right in FACTOR_NAMES[left_index + 1 :]:
                    means = {
                        (a, b): float(matrix[(matrix[left] == a) & (matrix[right] == b)][outcome].mean())
                        for a in (0, 1)
                        for b in (0, 1)
                    }
                    interaction = means[(1, 1)] - means[(1, 0)] - means[(0, 1)] + means[(0, 0)]
                    rows.append({"matrix_id": matrix_id, "outcome": outcome, "term": f"{left}:{right}", "order": "two_way", "effect": interaction})
    return pd.DataFrame(rows)


def analyze() -> dict[str, Any]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    surface_table = _surface_matrix_table()
    identity_table = _identity_matrix_table()
    factorial_table = _factorial_matrix_table()
    factorial_effects = _factorial_effect_table(factorial_table)
    surface = _surface_summary(surface_table)
    identity = _identity_summary(identity_table)
    shapley_table, factorial = _factorial_summary(factorial_table)
    _write_dataframe(OUTPUT_ROOT / "surface_matrix_cells.csv", surface_table)
    _write_dataframe(OUTPUT_ROOT / "identity_donor_cells.csv", identity_table)
    _write_dataframe(OUTPUT_ROOT / "factorial_matrix_cells.csv", factorial_table)
    _write_dataframe(OUTPUT_ROOT / "factorial_effects.csv", factorial_effects)
    _write_dataframe(OUTPUT_ROOT / "factorial_shapley.csv", shapley_table)
    summary = {
        "format": FORMAT,
        "protocol_id": _read_json(PROTOCOL_PATH)["protocol_id"],
        "runtime_selection_id": _runtime_selection()["selection_id"],
        "donor_selection_id": _read_json(DONOR_SELECTION_PATH)["selection_id"],
        "complete": True,
        "surface": surface,
        "identity": identity,
        "factorial": factorial,
        "reporting_boundary": "next-preprint exploratory follow-up; not current-preprint evidence or independent GARD/Wagner replication",
    }
    _write_json(OUTPUT_ROOT / "primary_summary.json", summary)
    _plots(surface_table, identity_table, shapley_table)
    _write_checksums(OUTPUT_ROOT)
    return summary


def _plots(surface: pd.DataFrame, identity: pd.DataFrame, shapley: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for axis, candidate in zip(axes, CANDIDATE_NAMES, strict=True):
        cell = surface[surface["candidate"] == candidate].groupby("grid_id")[["f12_rate", "strict8_rate"]].mean()
        axis.scatter(cell["f12_rate"], cell["strict8_rate"], c=np.arange(len(cell)), cmap="viridis")
        axis.set(title=f"Candidate {candidate}", xlabel="F12 rate", ylabel="strict-8 rate")
    figure.suptitle("Plasticity-persistence surface")
    figure.savefig(OUTPUT_ROOT / "figure_1_tradeoff.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for axis, candidate in zip(axes, CANDIDATE_NAMES, strict=True):
        cell = identity[(identity["candidate"] == candidate) & (identity["kind"] == "f12") & (identity["arm"] == "fork_comparison")]
        means = cell.groupby("grid_id")[["same_h_f8", "cross_h_f8"]].mean().reindex(ANCHORS)
        x = np.arange(len(ANCHORS))
        axis.plot(x, means["same_h_f8"], marker="o", label="same parent")
        axis.plot(x, means["cross_h_f8"], marker="o", label="different parent")
        axis.set(title=f"Candidate {candidate}", xticks=x, xticklabels=ANCHORS, ylim=(0, 1), ylabel="F8 similarity")
        axis.tick_params(axis="x", rotation=35)
        axis.legend()
    figure.savefig(OUTPUT_ROOT / "figure_2_forks.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for axis, outcome in zip(axes.flat, sorted(shapley["outcome"].unique()), strict=True):
        cell = shapley[shapley["outcome"] == outcome].groupby("factor")["absolute_share"].mean().reindex(FACTOR_NAMES)
        axis.bar(cell.index, cell.values)
        axis.set(title=outcome, ylim=(0, 1), ylabel="absolute Shapley share")
        axis.tick_params(axis="x", rotation=30)
    figure.savefig(OUTPUT_ROOT / "figure_3_contract_shapley.png", dpi=180)
    plt.close(figure)


def report() -> None:
    summary = _read_json(OUTPUT_ROOT / "primary_summary.json")
    tradeoff = summary["surface"]["tradeoff_pass_both"]
    identity = summary["identity"]["high_f12_lineage_identity_both"]
    distributed = summary["factorial"]["distributed_implementation_sensitivity"]
    technical = [
        "# Plasticity-persistence and lineage-identity results",
        "",
        "## Outcome",
        "",
        f"The prospectively defined plasticity-persistence trade-off passed in both simulator candidates: **{tradeoff}**.",
        f"The high-F12 regime passed the strong lineage-identity gate in both candidates: **{identity}**.",
        f"Contract sensitivity was distributed rather than attributable to one dominant unresolved choice: **{distributed}**.",
        "",
        "## Candidate trade-off tests",
        "",
        "| Candidate | F12/strict-8 correlation | High-regime F12 contrast | High-regime strict-8 contrast | Pass |",
        "|---|---:|---:|---:|---:|",
    ]
    for candidate, value in summary["surface"]["candidates"].items():
        technical.append(
            f"| {candidate} | {value['f12_strict8_spearman']['point']:.4f} | {value['high_minus_flanks_f12']['point']:.4f} | {value['high_minus_flanks_strict8']['point']:.4f} | {value['tradeoff_pass']} |"
        )
    technical.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "F12 is the weaker operational break-and-recovery endpoint; strict eight is rarer and stronger. Neither endpoint may rescue the other. This is fresh next-preprint exploratory work in reconstructed simulators, not evidence for the current preprint and not an independent GARD or Wagner replication. Damage spreading is used only as a simulator-contract sensitivity diagnostic, not as an edge-of-chaos claim.",
        ]
    )
    (OUTPUT_ROOT / "TECHNICAL_REPORT.md").write_text("\n".join(technical) + "\n", encoding="utf-8")
    lay = [
        "# Lay summary",
        "",
        f"A trade-off between easy breaking/recovery and long-lived coherent form was confirmed in both simulator versions: **{tradeoff}**.",
        f"The more plastic setting also preserved information about which lineage produced a form under the demanding fork-and-stranger test: **{identity}**.",
        "The experiment separates frequent short recovery from the much stronger strict-eight event, so more F12 does not automatically mean stronger heredity.",
        f"The difference between simulator versions appeared distributed across implementation choices rather than attributable to one dominant choice: **{distributed}**.",
        "These are exploratory simulator results for later work and do not alter the current preprint.",
    ]
    (OUTPUT_ROOT / "LAY_SUMMARY.md").write_text("\n".join(lay) + "\n", encoding="utf-8")
    _write_checksums(OUTPUT_ROOT)


def _expected_jobs() -> list[tuple[str, Any, Path]]:
    design = _design()
    jobs: list[tuple[str, Any, Path]] = []
    for candidate in CANDIDATE_NAMES:
        for matrix_id in range(design["surface_matrices"]):
            jobs.append(("surface", (candidate, matrix_id, design["surface_lineages"]), _surface_path(candidate, matrix_id)))
            jobs.append(("identity", (candidate, matrix_id, design["identity_futures"]), _identity_path(candidate, matrix_id)))
    for matrix_id in range(design["factorial_matrices"]):
        jobs.append(
            (
                "factorial",
                (matrix_id, design["factorial_lineages"], design["factorial_twin_starts"]),
                _factorial_path(matrix_id),
            )
        )
    return jobs


def _arrays_equal(left: NDArray, right: NDArray) -> bool:
    if left.dtype.kind in "fc" or right.dtype.kind in "fc":
        return bool(np.array_equal(left, right, equal_nan=True))
    return bool(np.array_equal(left, right))


def _replay_job(job: tuple[str, Any, Path]) -> dict[str, Any]:
    kind, args, path = job
    expected = _load_npz(path)
    observed = _surface_task(args) if kind == "surface" else _identity_task(args) if kind == "identity" else _factorial_task(args)
    exact = set(expected) == set(observed) and all(_arrays_equal(expected[key], observed[key]) for key in expected)
    return {"kind": kind, "path": str(path), "sha256": sha256_file(path), "all_arrays_exact": bool(exact)}


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
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
            futures = {executor.submit(_replay_job, job): job for job in jobs}
            for future in as_completed(futures):
                row = future.result()
                rows.append(row)
                _update_status(state="running", stage="verify-replay", completed=len(rows), total=len(jobs), started_at=started, message=Path(row["path"]).name)
                _check_soft_stop()
    complete = bool(
        full_replay
        and len(rows) == len(jobs)
        and all(row["all_arrays_exact"] for row in rows)
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
        "all_arrays_exact": bool(rows and all(row["all_arrays_exact"] for row in rows)),
        "protocol_checksums": _verify_checksums(PROTOCOL_ROOT),
        "output_checksums": _verify_checksums(OUTPUT_ROOT),
        "cleanroom_firewall": _firewall_audit(),
        "donor_registry_sha256_verified": sha256_file(DONOR_REGISTRY_PATH) == _read_json(DONOR_SELECTION_PATH)["registry_sha256"],
        "complete": complete,
        "checkpoints": rows,
    }
    _write_json(VERIFICATION_ROOT / "verification_audit.json", audit)
    _write_checksums(VERIFICATION_ROOT)
    if full_replay and not complete:
        raise ValueError("full replay verification failed")
    return audit


def _contract_corner_parity() -> bool:
    def mechanics(contract: SimulationContract) -> tuple[Any, ...]:
        return (contract.poisson_exposure, contract.overshoot_rule, contract.fission_rule, contract.daughter_rule)

    lookup = {bits: contract for _, bits, contract in CONTRACTS}
    return mechanics(lookup[(0, 0, 0, 0)]) == mechanics(CANDIDATES["02"]) and mechanics(lookup[(1, 1, 1, 1)]) == mechanics(CANDIDATES["03"])


def smoke() -> None:
    if not _contract_corner_parity():
        raise AssertionError("factorial corners do not reproduce registered candidates")
    beta = _matrix_beta("smoke", 0)
    for candidate, bits in (("02", (0, 0, 0, 0)), ("03", (1, 1, 1, 1))):
        contract = next(contract for _, contract_bits, contract in CONTRACTS if contract_bits == bits)
        seed = derive_seed(MASTER_SEED, "smoke.corner", candidate)
        reference = simulate_detailed_lineage(beta, BASE_GARD, CANDIDATES[candidate], seed=seed)
        observed = simulate_detailed_lineage(beta, BASE_GARD, contract, seed=seed)
        if reference.scalars() != observed.scalars() or not np.array_equal(reference.boundary_h, observed.boundary_h):
            raise AssertionError(f"factorial corner parity failed for candidate {candidate}")
    values = {tuple(bits): float(sum(bits)) for bits in np.ndindex(2, 2, 2, 2)}
    if not np.allclose(factorial_shapley(values), np.ones(4)):
        raise AssertionError("factorial Shapley smoke failed")
    print("Two-candidate corner parity and factorial smoke passed; production seeds were not consumed.")


def run_tests() -> None:
    subprocess.run([sys.executable, "-m", "pytest", str(TASK_ROOT / "test_campaign.py"), "-q"], cwd=CODEX_ROOT, check=True)


def status() -> None:
    payload = _read_json(STATUS_PATH) if STATUS_PATH.is_file() else {"format": FORMAT, "state": "not_started", "stage": "none"}
    payload["cumulative_elapsed_seconds"] = _ledger_elapsed() if LEDGER_PATH.is_file() else 0.0
    payload["soft_remaining_seconds"] = max(0.0, SOFT_LIMIT_SECONDS - payload["cumulative_elapsed_seconds"])
    payload["hard_remaining_seconds"] = max(0.0, HARD_LIMIT_SECONDS - payload["cumulative_elapsed_seconds"])
    print(json.dumps(payload, indent=2, sort_keys=True))


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
        run_surface(workers)
        build_donor_registry()
        run_identity(workers)
        run_factorial(workers)
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
    for name in ("benchmark", "surface", "identity", "factorial", "all"):
        child = commands.add_parser(name)
        child.add_argument("--workers", type=int, default=16)
    commands.add_parser("donors")
    commands.add_parser("analyze")
    commands.add_parser("report")
    verification = commands.add_parser("verify")
    verification.add_argument("--workers", type=int, default=16)
    verification.add_argument("--full-replay", action="store_true")
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
    elif args.command == "surface":
        run_surface(args.workers)
    elif args.command == "donors":
        print(json.dumps(build_donor_registry(), indent=2))
    elif args.command == "identity":
        run_identity(args.workers)
    elif args.command == "factorial":
        run_factorial(args.workers)
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
