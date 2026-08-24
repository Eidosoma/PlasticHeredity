"""Runner for the exploratory clean-room GARD lineage-carrier campaign."""

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
PARENT_TASK = TASK_ROOT.parent
CODEX_ROOT = PARENT_TASK.parent
WORKSPACE_ROOT = CODEX_ROOT.parent
if str(CODEX_ROOT) not in sys.path:
    sys.path.insert(0, str(CODEX_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(TASK_ROOT / "artifacts" / "matplotlib"))

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from threadpoolctl import threadpool_limits

from plastic_heredity.config import CANDIDATES, GardConfig
from plastic_heredity.mechanistic import sha256_file
from plastic_heredity.regime_confirmation import CONFIRMATION_MASTER_SEED
from plastic_heredity.seeds import derive_seed
from plastic_heredity.simulator import advance_fission, generate_beta
from reviewer_lineage_identity_response.followup_carrier_cleanroom.carrier_core import (
    ArmPolicy,
    CarrierSetting,
    bootstrap_mean_ci,
    choose_low_similarity_permutation,
    influence_mask,
    paired_bootstrap_ci,
    permutation_equivariance,
    random_mask,
    raw_cosine,
    reservoir_field,
    simulate_carrier_future,
    update_carrier,
    writer_signal,
)


FORMAT = "gard-lineage-carrier-cleanroom-v1"
MASTER_SEED = "df243aec2474d575012dfa1263e1cfeb1b92c62f962f45fe7cb87df85b713510"
GARD = GardConfig()
ENGINEERING_RULES = (11, 54, 63)
EXPECTED_SHARED_MULTIFORM_RULES = (2, 10, 23, 25, 45, 55, 72, 112, 148, 159, 161, 174, 178, 196)
CANDIDATE_NAMES = ("02", "03")
K_VALUES = (8, 16, 32, 100)
HALF_LIVES = (1, 4, 8)
COUPLINGS = (0.5, 1.0, 2.0)
COPY_MODES = ("ideal", "nominal")
CALIBRATION_FUTURES = 8
CALIBRATION_HORIZON = 32
PERMUTATION_PROPOSALS = 4_096
BOOTSTRAPS = 10_000
SOFT_LIMIT_SECONDS = 27_000.0
HARD_LIMIT_SECONDS = 28_800.0
TIER_BUDGET_SECONDS = 24_300.0
BENCHMARK_MAX_SECONDS = 300.0

CALIBRATION_ARMS = (
    "native_correct",
    "native_zero",
    "native_opposite",
    "native_shuffled",
    "stranger_correct",
    "stranger_zero",
)
PRIMARY_ARMS = (
    "native_correct",
    "stranger_correct",
    "stranger_zero",
    "stranger_shuffled",
    "native_zero",
    "native_shuffled",
    "native_reader_off",
    "native_founder_writer_off",
    "native_renewal_off",
    "native_erase_after_g2",
    "native_erase_rescue_g3",
    "native_no_carrier",
    "native_random_mask",
    "joint_relabel_correct",
)
MULTIFORM_ARMS = ("state_a_carrier_a", "state_a_carrier_b", "state_b_carrier_a", "state_b_carrier_b")
READOUT_FIELDS = (
    "observed",
    "completed",
    "extinct",
    "first_arrival",
    "arrival_f4",
    "arrival_f8",
    "arrival_f16",
    "capture_any_f16",
    "capture_any_f32",
    "capture_any_f64",
    "terminal8_f16",
    "terminal8_f32",
    "terminal8_f64",
    "occupancy",
    "maximum_residence",
    "departed",
    "reentered",
    "final_target_h",
    "final_other_h",
    "origin_correct",
    "carrier_target_h",
    "carrier_other_h",
    "carrier_origin_correct",
    "state_digest",
)
BOOLEAN_FIELDS = {
    "completed",
    "extinct",
    "arrival_f4",
    "arrival_f8",
    "arrival_f16",
    "capture_any_f16",
    "capture_any_f32",
    "capture_any_f64",
    "departed",
    "reentered",
}
INTEGER_FIELDS = {
    "observed",
    "first_arrival",
    "terminal8_f16",
    "terminal8_f32",
    "terminal8_f64",
    "maximum_residence",
    "origin_correct",
    "carrier_origin_correct",
}

ARTIFACT_ROOT = TASK_ROOT / "artifacts"
PROTOCOL_ROOT = ARTIFACT_ROOT / "protocol"
WORK_ROOT = ARTIFACT_ROOT / "work"
CALIBRATION_ROOT = WORK_ROOT / "calibration"
PRIMARY_ROOT = WORK_ROOT / "confirmation_primary"
MULTIFORM_ROOT = WORK_ROOT / "confirmation_multiform"
OUTPUT_ROOT = ARTIFACT_ROOT / "output"
VERIFICATION_ROOT = ARTIFACT_ROOT / "verification"
STATUS_PATH = ARTIFACT_ROOT / "STATUS.json"
LEDGER_PATH = ARTIFACT_ROOT / "runtime_ledger.json"
PROTOCOL_PATH = PROTOCOL_ROOT / "protocol.json"
REGISTRATION_PATH = PROTOCOL_ROOT / "registration.json"
TARGET_REGISTRY_PATH = PROTOCOL_ROOT / "target_registry.csv"
MASK_REGISTRY_PATH = PROTOCOL_ROOT / "mask_registry.json"
SOURCE_MANIFEST_PATH = PROTOCOL_ROOT / "scientific_source_manifest.json"
HYPOTHESIS_MANIFEST_PATH = PROTOCOL_ROOT / "hypothesis_only_manifest.json"
SEED_REGISTRY_PATH = PROTOCOL_ROOT / "seed_registry.json"
BENCHMARK_PATH = PROTOCOL_ROOT / "benchmark.json"
SELECTION_PATH = PROTOCOL_ROOT / "confirmation_selection.json"

PARENT_PROTOCOL = PARENT_TASK / "artifacts" / "protocol" / "protocol.json"
PARENT_SELECTION = PARENT_TASK / "artifacts" / "protocol" / "matrix_selection.csv"
PARENT_B_BANK = PARENT_TASK / "artifacts" / "output" / "b_bank.csv"
PARENT_AUDIT = PARENT_TASK / "artifacts" / "verification" / "verification_audit.json"

HYPOTHESIS_PATHS = {
    "gre_readme": WORKSPACE_ROOT / "NewIdeas/preprints/gre-ideas/README.md",
    "carrier_protocol": WORKSPACE_ROOT / "NewIdeas/preprints/gre-ideas/WAGNER_LINEAGE_CARRIER_V1_PROTOCOL.md",
    "local_ph_protocol": WORKSPACE_ROOT / "NewIdeas/preprints/gre-ideas/WAGNER_LOCAL_PH_V1_PROTOCOL.md",
    "heredity_preregistration": WORKSPACE_ROOT / "NewIdeas/preprints/gre-ideas/WAGNER_HEREDITY_EXPERIMENT_PREREGISTRATION.md",
    "induction_rescue_protocol": WORKSPACE_ROOT / "NewIdeas/preprints/gre-ideas/WAGNER_INDUCTION_RESCUE_PROTOCOL.md",
    "plastic_memory_preregistration": WORKSPACE_ROOT / "NewIdeas/preprints/gre-ideas/PLASTIC_MEMORY_EXPERIMENT_PREREGISTRATION.md",
    "carrier_discovery_report": WORKSPACE_ROOT / "NewIdeas/preprints/gre-ideas/runs/wagner-lineage-carrier-v1/discovery/report.md",
    "carrier_calibration_report": WORKSPACE_ROOT / "NewIdeas/preprints/gre-ideas/runs/wagner-lineage-carrier-v1/carrier-calibration/report.md",
    "carrier_selection": WORKSPACE_ROOT / "NewIdeas/preprints/gre-ideas/runs/wagner-lineage-carrier-v1/carrier-selection.json",
    "carrier_campaign_registration": WORKSPACE_ROOT / "NewIdeas/preprints/gre-ideas/runs/wagner-lineage-carrier-v1/campaign-registration.json",
    "induction_report": WORKSPACE_ROOT / "NewIdeas/preprints/gre-ideas/runs/wagner-induction-rescue-v1/report.md",
    "induction_validation_report": WORKSPACE_ROOT / "NewIdeas/preprints/gre-ideas/runs/wagner-induction-rescue-v1/validation/report.md",
    "induction_candidate": WORKSPACE_ROOT / "NewIdeas/preprints/gre-ideas/runs/wagner-induction-rescue-v1/candidate.json",
    "heredity_report": WORKSPACE_ROOT / "NewIdeas/preprints/gre-ideas/runs/wagner-heredity-v1/report.md",
    "heredity_confirmation_report": WORKSPACE_ROOT / "NewIdeas/preprints/gre-ideas/runs/wagner-heredity-v1/confirmation/report.md",
    "slow_mark_report": WORKSPACE_ROOT / "NewIdeas/preprints/gre-ideas/runs/wagner-heredity-v1/mark-discovery/report.md",
    "gard_mirror_registry": WORKSPACE_ROOT / "NewIdeas/preprints/gre-ideas/runs/wagner-heredity-v1/gard-mirror/calibration-registry.json",
}

BASE_SCIENTIFIC_PATHS = {
    "parent_protocol": PARENT_PROTOCOL,
    "parent_selection": PARENT_SELECTION,
    "parent_b_bank": PARENT_B_BANK,
    "parent_verification_audit": PARENT_AUDIT,
    "gard_config": CODEX_ROOT / "plastic_heredity" / "config.py",
    "gard_simulator": CODEX_ROOT / "plastic_heredity" / "simulator.py",
    "seed_derivation": CODEX_ROOT / "plastic_heredity" / "seeds.py",
    "beta_seed_source": CODEX_ROOT / "plastic_heredity" / "regime_confirmation.py",
    "carrier_core": TASK_ROOT / "carrier_core.py",
    "campaign_runner": TASK_ROOT / "run_campaign.py",
    "test_suite": TASK_ROOT / "test_carrier.py",
    "detached_guard": TASK_ROOT / "run_detached_pipeline.sh",
    "registered_protocol_text": TASK_ROOT / "PROTOCOL.md",
    "reporting_boundary": TASK_ROOT / "REPORTING_BOUNDARY.md",
    "hypothesis_boundary": TASK_ROOT / "HYPOTHESIS_BOUNDARY.md",
}


class SoftStop(RuntimeError):
    """Raised only at a checkpoint boundary after the cumulative soft limit."""


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
    encoded = json.dumps(_json_ready(value), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _assert_local(path: Path) -> None:
    resolved = path.resolve()
    if TASK_ROOT.resolve() not in (resolved, *resolved.parents):
        raise ValueError(f"refusing write outside clean-room task root: {resolved}")


def _write_json(path: Path, value: Any) -> None:
    _assert_local(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(_json_ready(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    _assert_local(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _atomic_npz(path: Path, **values: Any) -> None:
    _assert_local(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **values)
    os.replace(temporary, path)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as bundle:
        return {key: bundle[key] for key in bundle.files}


def _runtime() -> dict[str, str]:
    packages = ("numpy", "pandas", "threadpoolctl")
    result = {"python": platform.python_version()}
    for package in packages:
        result[package] = importlib.metadata.version(package)
    return result


def _manifest(paths: dict[str, Path], classification: str) -> dict[str, Any]:
    entries: dict[str, Any] = {}
    for label, path in sorted(paths.items()):
        if not path.is_file():
            raise FileNotFoundError(f"missing {classification} input: {path}")
        entries[label] = {"path": str(path.resolve()), "sha256": sha256_file(path), "bytes": path.stat().st_size}
    result = {"classification": classification, "entries": entries}
    result["manifest_id"] = _canonical_digest(result)
    return result


def _write_checksums(directory: Path) -> None:
    _assert_local(directory)
    files = sorted(path for path in directory.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    lines = [f"{sha256_file(path)}  {path.relative_to(directory)}" for path in files]
    (directory / "SHA256SUMS").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _verify_checksums(directory: Path) -> dict[str, bool]:
    checksum_path = directory / "SHA256SUMS"
    if not checksum_path.is_file():
        raise FileNotFoundError(checksum_path)
    result: dict[str, bool] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        path = directory / relative
        result[relative] = path.is_file() and sha256_file(path) == digest
    failed = [path for path, passed in result.items() if not passed]
    if failed:
        raise ValueError(f"checksum failures under {directory}: {failed}")
    return result


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
        accrued = min(max(0.0, time.time() - float(active)), HARD_LIMIT_SECONDS - prior)
        ledger["cumulative_seconds"] = min(HARD_LIMIT_SECONDS, prior + accrued)
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
    if not LEDGER_PATH.is_file():
        return 0.0
    ledger = _read_json(LEDGER_PATH)
    elapsed = float(ledger.get("cumulative_seconds", 0.0))
    if ledger.get("active_started_epoch") is not None:
        elapsed = float(ledger.get("active_cumulative_at_start") or elapsed) + max(
            0.0, time.time() - float(ledger["active_started_epoch"])
        )
    return min(HARD_LIMIT_SECONDS, elapsed)


def _stop_meter() -> None:
    if not LEDGER_PATH.is_file():
        return
    ledger = _read_json(LEDGER_PATH)
    active = ledger.get("active_started_epoch")
    if active is not None:
        prior = float(ledger.get("active_cumulative_at_start") or ledger.get("cumulative_seconds", 0.0))
        ledger["cumulative_seconds"] = min(HARD_LIMIT_SECONDS, prior + max(0.0, time.time() - float(active)))
        ledger["active_started_epoch"] = None
        ledger["active_cumulative_at_start"] = None
        ledger["last_stopped_at_epoch"] = time.time()
        _write_json(LEDGER_PATH, ledger)


def _check_soft_stop() -> None:
    if _ledger_elapsed() >= SOFT_LIMIT_SECONDS:
        raise SoftStop("cumulative 27,000-second soft limit reached at a checkpoint boundary")


def _update_status(
    *,
    state: str,
    stage: str,
    completed: int = 0,
    total: int = 0,
    started_at: float | None = None,
    message: str = "",
    eta_seconds: float | None = None,
    error: str | None = None,
) -> None:
    elapsed = 0.0 if started_at is None else max(0.0, time.time() - started_at)
    if eta_seconds is None and completed and total and elapsed:
        eta_seconds = max(0.0, elapsed * (total - completed) / completed)
    payload = {
        "format": FORMAT,
        "state": state,
        "stage": stage,
        "completed": completed,
        "total": total,
        "stage_elapsed_seconds": elapsed,
        "eta_seconds": eta_seconds,
        "message": message,
        "error": error,
        "pid": os.getpid(),
        "cumulative_elapsed_seconds": _ledger_elapsed(),
        "soft_remaining_seconds": max(0.0, SOFT_LIMIT_SECONDS - _ledger_elapsed()),
        "hard_remaining_seconds": max(0.0, HARD_LIMIT_SECONDS - _ledger_elapsed()),
        "updated_at_epoch": time.time(),
    }
    if PROTOCOL_PATH.is_file():
        payload["protocol_id"] = _read_json(PROTOCOL_PATH).get("protocol_id")
    if SELECTION_PATH.is_file():
        payload["selection_id"] = _read_json(SELECTION_PATH).get("selection_id")
    _write_json(STATUS_PATH, payload)


def _firewall_audit() -> dict[str, Any]:
    forbidden_parts = {"src", "tests", "scripts", "notebooks", "code"}
    forbidden_suffixes = {".py", ".pyc", ".ipynb", ".sh", ".toml", ".yaml", ".yml"}
    failures: list[str] = []
    for label, path in HYPOTHESIS_PATHS.items():
        relative_parts = {part.lower() for part in path.parts}
        if relative_parts & forbidden_parts or path.suffix.lower() in forbidden_suffixes:
            failures.append(f"{label}:{path}")
    imported = []
    ideas_root = (WORKSPACE_ROOT / "NewIdeas").resolve()
    for name, module in tuple(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if ideas_root in path.parents:
                imported.append(f"{name}:{path}")
    passed = not failures and not imported
    if not passed:
        raise ValueError(f"clean-room firewall failed: paths={failures}, imports={imported}")
    return {
        "passed": True,
        "policy": "NewIdeas documents and result JSON only; no source/tests/scripts/notebooks/config modules",
        "hypothesis_files": len(HYPOTHESIS_PATHS),
        "newideas_modules_imported": imported,
    }


def _parent_integrity() -> dict[str, Any]:
    audit = _read_json(PARENT_AUDIT)
    required = {"complete": True, "discrete_replay_exact": True, "output_checksums_verified": True}
    failed = [key for key, value in required.items() if audit.get(key) is not value]
    if failed or float(audit.get("maximum_h_error", float("inf"))) != 0.0:
        raise ValueError(f"parent verification audit is not complete: {failed}")
    return {
        "complete": True,
        "format": audit.get("format"),
        "protocol_id": audit.get("protocol_id"),
        "registration_id": audit.get("registration_id"),
        "replayed_lineages": audit.get("replayed_lineages"),
    }


def _selected_rules() -> list[int]:
    table = pd.read_csv(PARENT_SELECTION)
    if len(table) != 50 or "matrix_id" not in table:
        raise ValueError("parent selection is not the verified 50-rule cohort")
    return [int(value) for value in table["matrix_id"]]


def _strict_bank() -> pd.DataFrame:
    table = pd.read_csv(PARENT_B_BANK, dtype={"candidate": str})
    table["candidate"] = table["candidate"].str.zfill(2)
    table = table[table["kind"] == "strict"].copy()
    required = {"candidate", "matrix_id", "bank_index", "lineage", "final_B"}
    if not required.issubset(table.columns):
        raise ValueError("strict-B bank lacks required fields")
    table["matrix_id"] = table["matrix_id"].astype(int)
    table["bank_index"] = table["bank_index"].astype(int)
    table["lineage"] = table["lineage"].astype(int)
    return table.sort_values(["candidate", "matrix_id", "bank_index"])


def _beta(matrix_id: int) -> NDArray[np.float64]:
    rng = np.random.default_rng(derive_seed(CONFIRMATION_MASTER_SEED, "REGCONF.beta", matrix_id))
    return generate_beta(GARD, rng)


def _pair_for_cell(cell: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any], float] | None:
    records = list(cell.sort_values("bank_index").to_dict("records"))
    best: tuple[float, int, int, dict[str, Any], dict[str, Any]] | None = None
    for left_index, left in enumerate(records):
        left_state = np.asarray(json.loads(left["final_B"]), dtype=np.uint8)
        for right in records[left_index + 1 :]:
            if int(left["lineage"]) == int(right["lineage"]):
                continue
            right_state = np.asarray(json.loads(right["final_B"]), dtype=np.uint8)
            left_float = left_state.astype(np.float64)
            right_float = right_state.astype(np.float64)
            denominator = float(np.linalg.norm(left_float) * np.linalg.norm(right_float))
            h = 0.0 if denominator == 0.0 else float(np.dot(left_float, right_float) / denominator)
            key = (h, int(left["bank_index"]), int(right["bank_index"]), left, right)
            if best is None or key[:3] < best[:3]:
                best = key
    if best is None or best[0] > 0.85:
        return None
    return best[3], best[4], float(best[0])


def _registry_rows(bank: pd.DataFrame) -> tuple[list[dict[str, Any]], list[int]]:
    rules = _selected_rules()
    pair_eligibility: dict[str, set[int]] = {candidate: set() for candidate in CANDIDATE_NAMES}
    pairs: dict[tuple[str, int], tuple[dict[str, Any], dict[str, Any], float]] = {}
    for candidate in CANDIDATE_NAMES:
        for matrix_id in rules:
            cell = bank[(bank["candidate"] == candidate) & (bank["matrix_id"] == matrix_id)]
            if cell.empty:
                raise ValueError(f"strict bank missing c{candidate} m{matrix_id:03d}")
            pair = _pair_for_cell(cell)
            if pair is not None:
                pair_eligibility[candidate].add(matrix_id)
                pairs[(candidate, matrix_id)] = pair
    shared = sorted((pair_eligibility["02"] & pair_eligibility["03"]) - set(ENGINEERING_RULES))
    if tuple(shared) != EXPECTED_SHARED_MULTIFORM_RULES:
        raise ValueError(f"shared multiform cohort changed: {shared}")
    rows: list[dict[str, Any]] = []
    for candidate in CANDIDATE_NAMES:
        for matrix_id in rules:
            cell = bank[(bank["candidate"] == candidate) & (bank["matrix_id"] == matrix_id)].sort_values("bank_index")
            target_record = cell.iloc[0].to_dict()
            target = np.asarray(json.loads(target_record["final_B"]), dtype=np.uint8)
            stranger_permutation, stranger_h = choose_low_similarity_permutation(
                target,
                seed=derive_seed(MASTER_SEED, "registry.stranger", candidate, matrix_id),
                proposals=PERMUTATION_PROPOSALS,
            )
            iso_rng = np.random.default_rng(derive_seed(MASTER_SEED, "registry.isomorphism", candidate, matrix_id))
            iso_permutation = np.asarray(iso_rng.permutation(GARD.n_types), dtype=np.int16)
            pair = pairs.get((candidate, matrix_id)) if matrix_id in shared else None
            row = {
                "candidate": candidate,
                "matrix_id": matrix_id,
                "cohort": "engineering" if matrix_id in ENGINEERING_RULES else "primary",
                "target_bank_index": int(target_record["bank_index"]),
                "target_lineage": int(target_record["lineage"]),
                "target": json.dumps(target.tolist(), separators=(",", ":")),
                "stranger_permutation": json.dumps(stranger_permutation.tolist(), separators=(",", ":")),
                "stranger_h": stranger_h,
                "isomorphism_permutation": json.dumps(iso_permutation.tolist(), separators=(",", ":")),
                "multiform": pair is not None,
                "form_a_bank_index": "" if pair is None else int(pair[0]["bank_index"]),
                "form_a_lineage": "" if pair is None else int(pair[0]["lineage"]),
                "form_a": "" if pair is None else json.dumps(json.loads(pair[0]["final_B"]), separators=(",", ":")),
                "form_b_bank_index": "" if pair is None else int(pair[1]["bank_index"]),
                "form_b_lineage": "" if pair is None else int(pair[1]["lineage"]),
                "form_b": "" if pair is None else json.dumps(json.loads(pair[1]["final_B"]), separators=(",", ":")),
                "form_h": "" if pair is None else pair[2],
            }
            rows.append(row)
    return rows, shared


def _mask_registry(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    registry: dict[str, Any] = {"method": "beta_only_total_incoming_plus_outgoing", "cells": {}}
    for row in rows:
        candidate = str(row["candidate"])
        matrix_id = int(row["matrix_id"])
        key = f"c{candidate}_m{matrix_id:03d}"
        beta = _beta(matrix_id)
        cell: dict[str, Any] = {}
        for k in K_VALUES:
            selected = influence_mask(beta, k)
            matched_random = random_mask(
                GARD.n_types,
                k,
                derive_seed(MASTER_SEED, "registry.random_mask", candidate, matrix_id, k),
            )
            shuffle_rng = np.random.default_rng(
                derive_seed(MASTER_SEED, "registry.carrier_shuffle", candidate, matrix_id, k)
            )
            active = np.flatnonzero(selected)
            cell[str(k)] = {
                "influence_indices": active.tolist(),
                "random_indices": np.flatnonzero(matched_random).tolist(),
                "active_shuffle": np.asarray(shuffle_rng.permutation(active), dtype=int).tolist(),
            }
        registry["cells"][key] = cell
    registry["registry_id"] = _canonical_digest(registry)
    return registry


def _all_settings() -> list[CarrierSetting]:
    return [
        CarrierSetting(k=k, half_life=half_life, coupling=coupling, copy_mode=copy_mode)
        for k in K_VALUES
        for half_life in HALF_LIVES
        for coupling in COUPLINGS
        for copy_mode in COPY_MODES
    ]


def prepare() -> None:
    if PROTOCOL_PATH.is_file():
        protocol = verify_protocol()
        print(f"Existing clean-room protocol verified: {protocol['protocol_id']}")
        return
    firewall = _firewall_audit()
    parent = _parent_integrity()
    rows, shared = _registry_rows(_strict_bank())
    masks = _mask_registry(rows)
    scientific_manifest = _manifest(BASE_SCIENTIFIC_PATHS, "scientific_input_and_implementation")
    hypothesis_manifest = _manifest(HYPOTHESIS_PATHS, "hypothesis_only_non_evidentiary")
    _write_json(SOURCE_MANIFEST_PATH, scientific_manifest)
    _write_json(HYPOTHESIS_MANIFEST_PATH, hypothesis_manifest)
    _write_csv(TARGET_REGISTRY_PATH, rows)
    _write_json(MASK_REGISTRY_PATH, masks)
    seed_registry = {
        "master_seed": MASTER_SEED,
        "production_domains": [
            "calibration.dynamics",
            "calibration.carrier",
            "confirmation.dynamics",
            "confirmation.carrier",
            "multiform.dynamics",
            "multiform.carrier",
            "bootstrap",
        ],
        "nonproduction_domains": ["registry.stranger", "registry.isomorphism", "registry.random_mask", "registry.carrier_shuffle", "benchmark", "smoke"],
    }
    _write_json(SEED_REGISTRY_PATH, seed_registry)
    registration = {
        "format": FORMAT,
        "classification": "exploratory_reviewer_prompted_cleanroom_frozen_before_carrier_futures",
        "engineering_rules": list(ENGINEERING_RULES),
        "primary_rules": [rule for rule in _selected_rules() if rule not in ENGINEERING_RULES],
        "shared_multiform_rules": shared,
        "candidates_separate": True,
        "calibration_grid": [setting.to_dict() for setting in _all_settings()],
        "calibration": {"futures": CALIBRATION_FUTURES, "horizon": CALIBRATION_HORIZON, "arms": list(CALIBRATION_ARMS)},
        "calibration_pass_each_candidate": {
            "native_correct_terminal8_f32_point_ge": 0.50,
            "correct_minus_zero_ge": 0.20,
            "correct_minus_opposite_ge": 0.20,
            "correct_minus_shuffle_ge": 0.10,
            "stranger_correct_minus_zero_ge": 0.20,
        },
        "confirmation_arms": list(PRIMARY_ARMS),
        "multiform_arms": list(MULTIFORM_ARMS),
        "tiers": {
            "A": {"futures": 64, "horizon": 64},
            "B": {"futures": 48, "horizon": 64},
            "C": {"futures": 64, "horizon": 32},
        },
        "tier_selection": {"budget_seconds": TIER_BUDGET_SECONDS, "safety_factor": 1.5, "analysis_reserve_seconds": 600.0},
        "confirmation_gates": {
            "native_correct_f32_lower_gt": 0.30,
            "correct_minus_zero_point_ge": 0.20,
            "correct_minus_zero_lower_gt": 0.10,
            "control_difference_lower_gt": 0.0,
            "erasure_removal_fraction_ge": 0.70,
            "rescue_restoration_fraction_ge": 0.70,
            "multiform_each_cross_capture_lower_gt": 0.25,
            "multiform_crossover_point_ge": 0.20,
            "multiform_crossover_lower_gt": 0.10,
            "multiform_origin_accuracy_lower_gt": 0.75,
            "isomorphism_absolute_rate_difference_le": 0.03,
        },
        "bootstrap_repetitions": BOOTSTRAPS,
        "walltime": {"soft_seconds": SOFT_LIMIT_SECONDS, "hard_seconds": HARD_LIMIT_SECONDS, "cumulative_across_restarts": True},
        "reporting_boundary": "not preprint evidence; not a Wagner replication; candidates are contracts, not independent replications",
    }
    registration["registration_id"] = _canonical_digest(registration)
    _write_json(REGISTRATION_PATH, registration)
    protocol = {
        "format": FORMAT,
        "status": "sealed_before_new_carrier_futures",
        "scope": "clean-room molecule-indexed inherited-register stress test",
        "parent_integrity": parent,
        "firewall_audit": firewall,
        "rules": _selected_rules(),
        "engineering_rules": list(ENGINEERING_RULES),
        "primary_rule_count": 47,
        "shared_multiform_rules": shared,
        "candidates": list(CANDIDATE_NAMES),
        "settings": [setting.to_dict() for setting in _all_settings()],
        "carrier_definition": {
            "bounds": [-1.0, 1.0],
            "writer": "centered adult-parent abundance scaled by maximum absolute coordinate",
            "reader": "softmax(coupling * carrier)",
            "half_life_decay": "2**(-1/L)",
            "nominal_copy": {"retention": 0.95, "dropout": 0.02, "gaussian_sigma": 0.05},
        },
        "registration_id": registration["registration_id"],
        "target_registry_sha256": sha256_file(TARGET_REGISTRY_PATH),
        "mask_registry_sha256": sha256_file(MASK_REGISTRY_PATH),
        "scientific_source_manifest_sha256": sha256_file(SOURCE_MANIFEST_PATH),
        "hypothesis_only_manifest_sha256": sha256_file(HYPOTHESIS_MANIFEST_PATH),
        "seed_registry_sha256": sha256_file(SEED_REGISTRY_PATH),
        "runtime": _runtime(),
        "all_writes_below": str(TASK_ROOT.resolve()),
        "manuscript_modified": False,
        "wagner_code_used": False,
    }
    protocol["protocol_id"] = _canonical_digest(protocol)
    _write_json(PROTOCOL_PATH, protocol)
    _write_checksums(PROTOCOL_ROOT)
    _update_status(state="prepared", stage="prepare", message="protocol sealed; no carrier futures generated")
    print(json.dumps({"protocol_id": protocol["protocol_id"], "shared_multiform_rules": shared}, indent=2))


def verify_protocol() -> dict[str, Any]:
    if not PROTOCOL_PATH.is_file():
        raise FileNotFoundError("run prepare before simulation")
    # Once benchmark/selection are appended, rewrite_checksums expands the seal.
    _verify_checksums(PROTOCOL_ROOT)
    protocol = _read_json(PROTOCOL_PATH)
    payload = {key: value for key, value in protocol.items() if key != "protocol_id"}
    if protocol.get("format") != FORMAT or protocol.get("protocol_id") != _canonical_digest(payload):
        raise ValueError("protocol digest mismatch")
    if _manifest(BASE_SCIENTIFIC_PATHS, "scientific_input_and_implementation") != _read_json(SOURCE_MANIFEST_PATH):
        raise ValueError("scientific inputs or clean-room implementation changed after sealing")
    if _manifest(HYPOTHESIS_PATHS, "hypothesis_only_non_evidentiary") != _read_json(HYPOTHESIS_MANIFEST_PATH):
        raise ValueError("hypothesis-only context changed after sealing")
    if _firewall_audit().get("passed") is not True:
        raise ValueError("clean-room firewall no longer passes")
    _parent_integrity()
    if sha256_file(TARGET_REGISTRY_PATH) != protocol["target_registry_sha256"]:
        raise ValueError("target registry changed")
    if sha256_file(MASK_REGISTRY_PATH) != protocol["mask_registry_sha256"]:
        raise ValueError("mask registry changed")
    return protocol


def _load_registry() -> list[dict[str, Any]]:
    table = pd.read_csv(TARGET_REGISTRY_PATH, dtype={"candidate": str})
    rows: list[dict[str, Any]] = []
    for record in table.to_dict("records"):
        candidate = str(record["candidate"]).zfill(2)
        multiform = str(record["multiform"]).lower() in {"true", "1"}
        rows.append(
            {
                **record,
                "candidate": candidate,
                "matrix_id": int(record["matrix_id"]),
                "target": np.asarray(json.loads(record["target"]), dtype=np.uint8),
                "stranger_permutation": np.asarray(json.loads(record["stranger_permutation"]), dtype=np.int16),
                "isomorphism_permutation": np.asarray(json.loads(record["isomorphism_permutation"]), dtype=np.int16),
                "multiform": multiform,
                "form_a": None if not multiform else np.asarray(json.loads(record["form_a"]), dtype=np.uint8),
                "form_b": None if not multiform else np.asarray(json.loads(record["form_b"]), dtype=np.uint8),
            }
        )
    return rows


def _registry_cell(candidate: str, matrix_id: int) -> dict[str, Any]:
    matches = [row for row in _load_registry() if row["candidate"] == candidate and row["matrix_id"] == matrix_id]
    if len(matches) != 1:
        raise ValueError(f"registry cell count for c{candidate} m{matrix_id:03d}: {len(matches)}")
    return matches[0]


def _mask_values(candidate: str, matrix_id: int, k: int, kind: str = "influence") -> NDArray[np.bool_]:
    registry = _read_json(MASK_REGISTRY_PATH)
    values = registry["cells"][f"c{candidate}_m{matrix_id:03d}"][str(k)]
    key = "influence_indices" if kind == "influence" else "random_indices"
    mask = np.zeros(GARD.n_types, dtype=bool)
    mask[np.asarray(values[key], dtype=int)] = True
    return mask


def _shuffled_initial(candidate: str, matrix_id: int, setting: CarrierSetting, target: NDArray, mask: NDArray) -> NDArray[np.float64]:
    values = _read_json(MASK_REGISTRY_PATH)["cells"][f"c{candidate}_m{matrix_id:03d}"][str(setting.k)]
    active = np.asarray(values["influence_indices"], dtype=int)
    shuffled = np.asarray(values["active_shuffle"], dtype=int)
    base = writer_signal(target, mask)
    result = np.zeros_like(base)
    result[active] = base[shuffled]
    return result


def _policy(name: str) -> ArmPolicy:
    policies = {
        "native_correct": ArmPolicy(name),
        "stranger_correct": ArmPolicy(name),
        "native_zero": ArmPolicy(name, initial="zero", renewal=False, no_carrier=True),
        "stranger_zero": ArmPolicy(name, initial="zero", renewal=False, no_carrier=True),
        "native_opposite": ArmPolicy(name, initial="opposite", renewal=False),
        "native_shuffled": ArmPolicy(name, initial="shuffled", renewal=False),
        "stranger_shuffled": ArmPolicy(name, initial="shuffled", renewal=False),
        "native_reader_off": ArmPolicy(name, reader=False),
        "native_founder_writer_off": ArmPolicy(name, initial="zero"),
        "native_renewal_off": ArmPolicy(name, renewal=False),
        "native_erase_after_g2": ArmPolicy(name, erase_after_generation=2),
        "native_erase_rescue_g3": ArmPolicy(name, erase_after_generation=2, rescue_generation=3),
        "native_no_carrier": ArmPolicy(name, initial="zero", renewal=False, no_carrier=True),
        "native_random_mask": ArmPolicy(name),
        "joint_relabel_correct": ArmPolicy(name),
        "state_a_carrier_a": ArmPolicy(name),
        "state_a_carrier_b": ArmPolicy(name),
        "state_b_carrier_a": ArmPolicy(name),
        "state_b_carrier_b": ArmPolicy(name),
    }
    if name not in policies:
        raise ValueError(f"unknown arm: {name}")
    return policies[name]


def _empty_result_arrays(shape: tuple[int, ...], horizon: int) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for field in READOUT_FIELDS:
        if field == "state_digest":
            arrays[field] = np.empty(shape, dtype="<U64")
        elif field in BOOLEAN_FIELDS:
            arrays[field] = np.zeros(shape, dtype=np.bool_)
        elif field in INTEGER_FIELDS:
            arrays[field] = np.full(shape, -1, dtype=np.int16)
        else:
            arrays[field] = np.full(shape, np.nan, dtype=np.float64)
    arrays["boundary_h"] = np.full(shape + (horizon,), np.nan, dtype=np.float64)
    arrays["target_h"] = np.full(shape + (horizon,), np.nan, dtype=np.float64)
    arrays["carrier_target_h_trace"] = np.full(shape + (horizon,), np.nan, dtype=np.float64)
    return arrays


def _store_readout(
    arrays: dict[str, np.ndarray],
    index: tuple[int, ...],
    readout: Any,
    boundary_h: NDArray,
    target_h: NDArray,
    carrier_trace: NDArray,
) -> None:
    values = readout.to_dict()
    for field in READOUT_FIELDS:
        arrays[field][index] = values[field]
    arrays["boundary_h"][index] = boundary_h
    arrays["target_h"][index] = target_h
    arrays["carrier_target_h_trace"][index] = carrier_trace


def _benchmark_batch(batch: int) -> tuple[int, int]:
    limiter = threadpool_limits(limits=1)
    try:
        candidate = CANDIDATE_NAMES[batch % len(CANDIDATE_NAMES)]
        matrix_id = ENGINEERING_RULES[batch % len(ENGINEERING_RULES)]
        row = _registry_cell(candidate, matrix_id)
        beta = _beta(matrix_id)
        setting = CarrierSetting(k=100, half_life=4, coupling=2.0, copy_mode="nominal")
        mask = _mask_values(candidate, matrix_id, setting.k)
        observed_total = 0
        futures = 16
        for future in range(futures):
            readout, _, _, _ = simulate_carrier_future(
                row["target"],
                row["target"],
                None,
                beta,
                GARD,
                CANDIDATES[candidate],
                setting,
                mask,
                _policy("native_correct"),
                dynamics_seed=derive_seed(MASTER_SEED, "benchmark", "dynamics", batch, future),
                carrier_seed=derive_seed(MASTER_SEED, "benchmark", "carrier", batch, future),
                horizon=32,
            )
            observed_total += readout.observed
        return futures, observed_total
    finally:
        limiter.restore_original_limits()


def benchmark(workers: int = 16) -> dict[str, Any]:
    verify_protocol()
    if BENCHMARK_PATH.is_file():
        result = _read_json(BENCHMARK_PATH)
        print(json.dumps(result, indent=2))
        return result
    started = time.time()
    batches = list(range(max(1, workers)))
    _update_status(state="running", stage="benchmark", total=len(batches), started_at=started)
    results: list[tuple[int, int]] = []
    if workers <= 1:
        results = [_benchmark_batch(batch) for batch in batches]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for index, item in enumerate(executor.map(_benchmark_batch, batches, chunksize=1), start=1):
                results.append(item)
                _update_status(state="running", stage="benchmark", completed=index, total=len(batches), started_at=started)
                if time.time() - started > BENCHMARK_MAX_SECONDS:
                    raise TimeoutError("production-seed-safe benchmark exceeded its registered five-minute ceiling")
    wall = time.time() - started
    futures = sum(item[0] for item in results)
    generations = sum(item[1] for item in results)
    result = {
        "format": FORMAT,
        "classification": "production_seed_safe_benchmark",
        "workers": workers,
        "batches": len(batches),
        "futures": futures,
        "observed_generations": generations,
        "wall_seconds": wall,
        "effective_generations_per_wall_second": generations / wall,
        "benchmark_seed_domain": "benchmark",
        "production_seed_consumed": False,
        "within_five_minutes": wall <= BENCHMARK_MAX_SECONDS,
    }
    result["benchmark_id"] = _canonical_digest(result)
    _write_json(BENCHMARK_PATH, result)
    _write_checksums(PROTOCOL_ROOT)
    _update_status(state="benchmarked", stage="benchmark", completed=len(batches), total=len(batches), started_at=started, message=f"throughput={result['effective_generations_per_wall_second']:.1f} generations/s")
    print(json.dumps(result, indent=2))
    return result


def _calibration_path(candidate: str, matrix_id: int) -> Path:
    return CALIBRATION_ROOT / f"c{candidate}_m{matrix_id:03d}.npz"


def _calibration_tasks() -> list[tuple[str, int]]:
    return [(candidate, matrix_id) for candidate in CANDIDATE_NAMES for matrix_id in ENGINEERING_RULES]


def _simulate_calibration_cell(candidate: str, matrix_id: int) -> dict[str, Any]:
    limiter = threadpool_limits(limits=1)
    try:
        protocol = _read_json(PROTOCOL_PATH)
        row = _registry_cell(candidate, matrix_id)
        target = row["target"]
        stranger = target[row["stranger_permutation"]].copy()
        beta = _beta(matrix_id)
        settings = _all_settings()
        shape = (len(settings), len(CALIBRATION_ARMS), CALIBRATION_FUTURES)
        arrays = _empty_result_arrays(shape, CALIBRATION_HORIZON)
        for setting_index, setting in enumerate(settings):
            mask = _mask_values(candidate, matrix_id, setting.k)
            shuffled = _shuffled_initial(candidate, matrix_id, setting, target, mask)
            opposite = writer_signal(stranger, mask)
            for arm_index, arm in enumerate(CALIBRATION_ARMS):
                start = stranger if arm.startswith("stranger") else target
                override = None
                if arm == "native_opposite":
                    override = opposite
                elif arm == "native_shuffled":
                    override = shuffled
                start_group = "stranger" if arm.startswith("stranger") else "native"
                for future in range(CALIBRATION_FUTURES):
                    readout, h, target_h, carrier_h = simulate_carrier_future(
                        start,
                        target,
                        None,
                        beta,
                        GARD,
                        CANDIDATES[candidate],
                        setting,
                        mask,
                        _policy(arm),
                        dynamics_seed=derive_seed(MASTER_SEED, "calibration.dynamics", candidate, matrix_id, start_group, future),
                        carrier_seed=derive_seed(MASTER_SEED, "calibration.carrier", candidate, matrix_id, setting.setting_id, start_group, future),
                        horizon=CALIBRATION_HORIZON,
                        initial_override=override,
                    )
                    _store_readout(arrays, (setting_index, arm_index, future), readout, h, target_h, carrier_h)
        return {
            "format": np.asarray(FORMAT),
            "protocol_id": np.asarray(protocol["protocol_id"]),
            "candidate": np.asarray(candidate),
            "matrix_id": np.asarray(matrix_id, dtype=np.int16),
            "setting_ids": np.asarray([setting.setting_id for setting in settings]),
            "arms": np.asarray(CALIBRATION_ARMS),
            "futures": np.asarray(CALIBRATION_FUTURES, dtype=np.int16),
            "horizon": np.asarray(CALIBRATION_HORIZON, dtype=np.int16),
            **arrays,
        }
    finally:
        limiter.restore_original_limits()


def _save_calibration_task(task: tuple[str, int]) -> tuple[str, int, str, int]:
    candidate, matrix_id = task
    path = _calibration_path(candidate, matrix_id)
    if path.is_file():
        payload = _load_npz(path)
        if str(payload["protocol_id"].item()) != _read_json(PROTOCOL_PATH)["protocol_id"]:
            raise ValueError(f"stale calibration checkpoint: {path}")
        return candidate, matrix_id, "existing", int(payload["state_digest"].size)
    payload = _simulate_calibration_cell(candidate, matrix_id)
    _atomic_npz(path, **payload)
    return candidate, matrix_id, "generated", int(payload["state_digest"].size)


def _run_tasks(
    function: Callable[[Any], tuple[str, int, str, int]],
    tasks: Sequence[Any],
    *,
    workers: int,
    stage: str,
) -> None:
    started = time.time()
    total = len(tasks)
    _update_status(state="running", stage=stage, total=total, started_at=started)
    if workers <= 1:
        iterator: Iterable[tuple[str, int, str, int]] = map(function, tasks)
        for index, result in enumerate(iterator, start=1):
            message = f"c{result[0]} m{result[1]:03d} {result[2]} futures={result[3]}"
            print(f"[{stage}] {index}/{total} {message}", flush=True)
            _update_status(state="running", stage=stage, completed=index, total=total, started_at=started, message=message)
            _check_soft_stop()
        return
    completed = 0
    for offset in range(0, total, workers):
        batch = tasks[offset : offset + workers]
        with ProcessPoolExecutor(max_workers=min(workers, len(batch))) as executor:
            results = list(executor.map(function, batch, chunksize=1))
        for result in results:
            completed += 1
            message = f"c{result[0]} m{result[1]:03d} {result[2]} futures={result[3]}"
            print(f"[{stage}] {completed}/{total} {message}", flush=True)
            _update_status(state="running", stage=stage, completed=completed, total=total, started_at=started, message=message)
        _check_soft_stop()


def _calibration_rule_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate, matrix_id in _calibration_tasks():
        path = _calibration_path(candidate, matrix_id)
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = _load_npz(path)
        settings = [str(value) for value in payload["setting_ids"]]
        arms = [str(value) for value in payload["arms"]]
        for setting_index, setting_id in enumerate(settings):
            for arm_index, arm in enumerate(arms):
                terminal = np.asarray(payload["terminal8_f32"][setting_index, arm_index], dtype=np.int16)
                rows.append(
                    {
                        "candidate": candidate,
                        "matrix_id": matrix_id,
                        "setting_id": setting_id,
                        "arm": arm,
                        "terminal8_f32": float(np.mean(terminal == 1)),
                        "capture_any_f32": float(np.mean(payload["capture_any_f32"][setting_index, arm_index])),
                        "arrival_f16": float(np.mean(payload["arrival_f16"][setting_index, arm_index])),
                        "occupancy": float(np.mean(payload["occupancy"][setting_index, arm_index])),
                        "extinction": float(np.mean(payload["extinct"][setting_index, arm_index])),
                    }
                )
    return rows


def _calibration_selection(metrics: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    setting_lookup = {setting.setting_id: setting for setting in _all_settings()}
    summaries: list[dict[str, Any]] = []
    for setting_id, setting in setting_lookup.items():
        candidate_summaries: dict[str, Any] = {}
        candidate_passes = []
        candidate_scores = []
        for candidate in CANDIDATE_NAMES:
            cell = metrics[(metrics["setting_id"] == setting_id) & (metrics["candidate"] == candidate)]
            by_arm = cell.groupby("arm")["terminal8_f32"].mean().to_dict()
            native = float(by_arm["native_correct"])
            zero = float(by_arm["native_zero"])
            opposite = float(by_arm["native_opposite"])
            shuffled = float(by_arm["native_shuffled"])
            stranger = float(by_arm["stranger_correct"])
            stranger_zero = float(by_arm["stranger_zero"])
            values = {
                "native_correct": native,
                "correct_minus_zero": native - zero,
                "correct_minus_opposite": native - opposite,
                "correct_minus_shuffled": native - shuffled,
                "stranger_correct_minus_zero": stranger - stranger_zero,
            }
            passed = bool(
                native >= 0.50
                and native - zero >= 0.20
                and native - opposite >= 0.20
                and native - shuffled >= 0.10
                and stranger - stranger_zero >= 0.20
            )
            score = min(
                native - 0.50,
                native - zero - 0.20,
                native - opposite - 0.20,
                native - shuffled - 0.10,
                stranger - stranger_zero - 0.20,
            )
            candidate_summaries[candidate] = {**values, "passed": passed, "margin_score": score}
            candidate_passes.append(passed)
            candidate_scores.append(score)
        summaries.append(
            {
                **setting.to_dict(),
                "setting_id": setting_id,
                "candidate_02": candidate_summaries["02"],
                "candidate_03": candidate_summaries["03"],
                "passed_both_candidates": all(candidate_passes),
                "minimum_candidate_margin_score": min(candidate_scores),
            }
        )
    passing = [row for row in summaries if row["passed_both_candidates"]]
    selected: list[dict[str, Any]] = []
    nominal = sorted(
        (row for row in passing if row["copy_mode"] == "nominal"),
        key=lambda row: (row["k"], row["coupling"], row["half_life"], row["setting_id"]),
    )
    if nominal:
        selected.append({**{key: nominal[0][key] for key in ("setting_id", "k", "half_life", "coupling", "copy_mode")}, "selection_role": "smallest_nominal_pass"})
    if passing:
        strongest = sorted(
            passing,
            key=lambda row: (-row["minimum_candidate_margin_score"], row["k"], row["coupling"], row["half_life"], row["setting_id"]),
        )[0]
        if not any(row["setting_id"] == strongest["setting_id"] for row in selected):
            selected.append({**{key: strongest[key] for key in ("setting_id", "k", "half_life", "coupling", "copy_mode")}, "selection_role": "strongest_pass"})
    return summaries, selected


def _tier_projection(selected_count: int, benchmark_result: dict[str, Any]) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    throughput = float(benchmark_result["effective_generations_per_wall_second"])
    primary_cells = len(CANDIDATE_NAMES) * 47
    multiform_cells = len(CANDIDATE_NAMES) * len(EXPECTED_SHARED_MULTIFORM_RULES)
    calibration_replay_generations = (
        len(CANDIDATE_NAMES)
        * len(ENGINEERING_RULES)
        * len(_all_settings())
        * len(CALIBRATION_ARMS)
        * CALIBRATION_FUTURES
        * CALIBRATION_HORIZON
    )
    projections: list[dict[str, Any]] = []
    for tier, futures, horizon in (("A", 64, 64), ("B", 48, 64), ("C", 64, 32)):
        confirmation_generations = selected_count * futures * horizon * (
            primary_cells * len(PRIMARY_ARMS) + multiform_cells * len(MULTIFORM_ARMS)
        )
        # Confirmation is generated and replayed; calibration has already been generated but must be replayed.
        raw_remaining = (2 * confirmation_generations + calibration_replay_generations) / throughput + 600.0
        safe_remaining = 1.5 * raw_remaining
        projected_total = _ledger_elapsed() + safe_remaining
        projections.append(
            {
                "tier": tier,
                "futures": futures,
                "horizon": horizon,
                "selected_settings": selected_count,
                "confirmation_future_generations": confirmation_generations,
                "raw_remaining_seconds": raw_remaining,
                "safety_adjusted_remaining_seconds": safe_remaining,
                "projected_cumulative_seconds": projected_total,
                "fits_6p75_hour_budget": projected_total <= TIER_BUDGET_SECONDS,
            }
        )
    chosen = next((row for row in projections if row["fits_6p75_hour_budget"]), None)
    return chosen, projections


def analyze_calibration() -> dict[str, Any]:
    if SELECTION_PATH.is_file():
        selection = _load_selection()
        print(json.dumps(selection, indent=2))
        return selection
    metrics = pd.DataFrame(_calibration_rule_rows())
    summaries, selected = _calibration_selection(metrics)
    if not BENCHMARK_PATH.is_file():
        raise FileNotFoundError("run benchmark before freezing confirmation selection")
    chosen, projections = _tier_projection(len(selected), _read_json(BENCHMARK_PATH))
    calibration_summary = {
        "format": FORMAT,
        "protocol_id": _read_json(PROTOCOL_PATH)["protocol_id"],
        "settings_examined": len(summaries),
        "settings_passing_both_candidates": sum(bool(row["passed_both_candidates"]) for row in summaries),
        "settings": summaries,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT_ROOT / "calibration_rule_metrics.csv", index=False)
    _write_json(OUTPUT_ROOT / "calibration_summary.json", calibration_summary)
    selection = {
        "format": FORMAT,
        "status": "frozen_after_engineering_before_confirmation_futures",
        "protocol_id": _read_json(PROTOCOL_PATH)["protocol_id"],
        "calibration_summary_sha256": sha256_file(OUTPUT_ROOT / "calibration_summary.json"),
        "selected_settings": selected,
        "selected_setting_count": len(selected),
        "tier": chosen,
        "tier_projections": projections,
        "confirmation_authorized": bool(selected and chosen is not None),
        "no_confirmation_reason": None if selected and chosen is not None else ("no calibration setting passed" if not selected else "no predeclared tier fits the cumulative guarded budget"),
    }
    selection["selection_id"] = _canonical_digest(selection)
    _write_json(SELECTION_PATH, selection)
    _write_checksums(PROTOCOL_ROOT)
    _update_status(
        state="calibrated",
        stage="calibrate",
        completed=len(_calibration_tasks()),
        total=len(_calibration_tasks()),
        message=f"passing={calibration_summary['settings_passing_both_candidates']} selected={len(selected)} tier={None if chosen is None else chosen['tier']}",
    )
    print(json.dumps(selection, indent=2))
    return selection


def calibrate(workers: int) -> None:
    verify_protocol()
    _run_tasks(_save_calibration_task, _calibration_tasks(), workers=workers, stage="calibrate-simulate")
    analyze_calibration()


def _load_selection() -> dict[str, Any]:
    if not SELECTION_PATH.is_file():
        raise FileNotFoundError("calibration has not frozen a confirmation selection")
    selection = _read_json(SELECTION_PATH)
    payload = {key: value for key, value in selection.items() if key != "selection_id"}
    if selection.get("selection_id") != _canonical_digest(payload):
        raise ValueError("confirmation selection digest mismatch")
    if selection.get("protocol_id") != _read_json(PROTOCOL_PATH)["protocol_id"]:
        raise ValueError("confirmation selection belongs to another protocol")
    if selection.get("calibration_summary_sha256") != sha256_file(OUTPUT_ROOT / "calibration_summary.json"):
        raise ValueError("calibration summary changed after selection freeze")
    return selection


def _selected_settings() -> list[CarrierSetting]:
    return [
        CarrierSetting(
            k=int(row["k"]),
            half_life=int(row["half_life"]),
            coupling=float(row["coupling"]),
            copy_mode=str(row["copy_mode"]),
        )
        for row in _load_selection()["selected_settings"]
    ]


def _confirmation_design() -> tuple[int, int]:
    tier = _load_selection().get("tier")
    if tier is None:
        return 0, 0
    return int(tier["futures"]), int(tier["horizon"])


def _primary_path(candidate: str, matrix_id: int, setting_id: str) -> Path:
    return PRIMARY_ROOT / setting_id / f"c{candidate}_m{matrix_id:03d}.npz"


def _multiform_path(candidate: str, matrix_id: int, setting_id: str) -> Path:
    return MULTIFORM_ROOT / setting_id / f"c{candidate}_m{matrix_id:03d}.npz"


def _primary_tasks() -> list[tuple[str, int, str]]:
    rules = [rule for rule in _selected_rules() if rule not in ENGINEERING_RULES]
    return [
        (candidate, matrix_id, setting.setting_id)
        for setting in _selected_settings()
        for candidate in CANDIDATE_NAMES
        for matrix_id in rules
    ]


def _multiform_tasks() -> list[tuple[str, int, str]]:
    return [
        (candidate, matrix_id, setting.setting_id)
        for setting in _selected_settings()
        for candidate in CANDIDATE_NAMES
        for matrix_id in EXPECTED_SHARED_MULTIFORM_RULES
    ]


def _setting_by_id(setting_id: str) -> CarrierSetting:
    matches = [setting for setting in _selected_settings() if setting.setting_id == setting_id]
    if len(matches) != 1:
        raise ValueError(f"selected setting lookup failed: {setting_id}")
    return matches[0]


def _simulate_primary_cell(candidate: str, matrix_id: int, setting_id: str) -> dict[str, Any]:
    limiter = threadpool_limits(limits=1)
    try:
        protocol = _read_json(PROTOCOL_PATH)
        selection = _load_selection()
        futures, horizon = _confirmation_design()
        setting = _setting_by_id(setting_id)
        row = _registry_cell(candidate, matrix_id)
        target = row["target"]
        stranger = target[row["stranger_permutation"]].copy()
        beta = _beta(matrix_id)
        mask = _mask_values(candidate, matrix_id, setting.k)
        matched_random_mask = _mask_values(candidate, matrix_id, setting.k, "random")
        shuffled = _shuffled_initial(candidate, matrix_id, setting, target, mask)
        shape = (len(PRIMARY_ARMS), futures)
        arrays = _empty_result_arrays(shape, horizon)
        for arm_index, arm in enumerate(PRIMARY_ARMS):
            arm_beta = beta
            arm_target = target
            arm_start = stranger if arm.startswith("stranger") else target
            arm_mask = matched_random_mask if arm == "native_random_mask" else mask
            override = None
            if arm in {"native_shuffled", "stranger_shuffled"}:
                override = shuffled
            if arm == "joint_relabel_correct":
                order = row["isomorphism_permutation"]
                correct = writer_signal(target, mask)
                arm_beta, arm_start, _, arm_mask = permutation_equivariance(beta, target, correct, mask, order)
                arm_target = target[order].copy()
            start_group = "stranger" if arm.startswith("stranger") else "native"
            for future in range(futures):
                readout, h, target_h, carrier_h = simulate_carrier_future(
                    arm_start,
                    arm_target,
                    None,
                    arm_beta,
                    GARD,
                    CANDIDATES[candidate],
                    setting,
                    arm_mask,
                    _policy(arm),
                    dynamics_seed=derive_seed(MASTER_SEED, "confirmation.dynamics", candidate, matrix_id, start_group, future),
                    carrier_seed=derive_seed(MASTER_SEED, "confirmation.carrier", candidate, matrix_id, setting_id, start_group, future),
                    horizon=horizon,
                    initial_override=override,
                )
                _store_readout(arrays, (arm_index, future), readout, h, target_h, carrier_h)
        return {
            "format": np.asarray(FORMAT),
            "protocol_id": np.asarray(protocol["protocol_id"]),
            "selection_id": np.asarray(selection["selection_id"]),
            "kind": np.asarray("primary"),
            "candidate": np.asarray(candidate),
            "matrix_id": np.asarray(matrix_id, dtype=np.int16),
            "setting_id": np.asarray(setting_id),
            "arms": np.asarray(PRIMARY_ARMS),
            "futures": np.asarray(futures, dtype=np.int16),
            "horizon": np.asarray(horizon, dtype=np.int16),
            **arrays,
        }
    finally:
        limiter.restore_original_limits()


def _save_primary_task(task: tuple[str, int, str]) -> tuple[str, int, str, int]:
    candidate, matrix_id, setting_id = task
    path = _primary_path(candidate, matrix_id, setting_id)
    if path.is_file():
        payload = _load_npz(path)
        if str(payload["selection_id"].item()) != _load_selection()["selection_id"]:
            raise ValueError(f"stale primary checkpoint: {path}")
        return candidate, matrix_id, "existing", int(payload["state_digest"].size)
    payload = _simulate_primary_cell(candidate, matrix_id, setting_id)
    _atomic_npz(path, **payload)
    return candidate, matrix_id, "generated", int(payload["state_digest"].size)


def _simulate_multiform_cell(candidate: str, matrix_id: int, setting_id: str) -> dict[str, Any]:
    limiter = threadpool_limits(limits=1)
    try:
        protocol = _read_json(PROTOCOL_PATH)
        selection = _load_selection()
        futures, horizon = _confirmation_design()
        setting = _setting_by_id(setting_id)
        row = _registry_cell(candidate, matrix_id)
        if not row["multiform"] or row["form_a"] is None or row["form_b"] is None:
            raise ValueError(f"multiform registry missing c{candidate} m{matrix_id:03d}")
        form_a = row["form_a"]
        form_b = row["form_b"]
        beta = _beta(matrix_id)
        mask = _mask_values(candidate, matrix_id, setting.k)
        shape = (len(MULTIFORM_ARMS), futures)
        arrays = _empty_result_arrays(shape, horizon)
        for arm_index, arm in enumerate(MULTIFORM_ARMS):
            start = form_a if arm.startswith("state_a") else form_b
            target = form_a if arm.endswith("carrier_a") else form_b
            other = form_b if arm.endswith("carrier_a") else form_a
            start_label = "a" if arm.startswith("state_a") else "b"
            for future in range(futures):
                readout, h, target_h, carrier_h = simulate_carrier_future(
                    start,
                    target,
                    other,
                    beta,
                    GARD,
                    CANDIDATES[candidate],
                    setting,
                    mask,
                    _policy(arm),
                    dynamics_seed=derive_seed(MASTER_SEED, "multiform.dynamics", candidate, matrix_id, start_label, future),
                    carrier_seed=derive_seed(MASTER_SEED, "multiform.carrier", candidate, matrix_id, setting_id, start_label, future),
                    horizon=horizon,
                )
                _store_readout(arrays, (arm_index, future), readout, h, target_h, carrier_h)
        return {
            "format": np.asarray(FORMAT),
            "protocol_id": np.asarray(protocol["protocol_id"]),
            "selection_id": np.asarray(selection["selection_id"]),
            "kind": np.asarray("multiform"),
            "candidate": np.asarray(candidate),
            "matrix_id": np.asarray(matrix_id, dtype=np.int16),
            "setting_id": np.asarray(setting_id),
            "arms": np.asarray(MULTIFORM_ARMS),
            "futures": np.asarray(futures, dtype=np.int16),
            "horizon": np.asarray(horizon, dtype=np.int16),
            **arrays,
        }
    finally:
        limiter.restore_original_limits()


def _save_multiform_task(task: tuple[str, int, str]) -> tuple[str, int, str, int]:
    candidate, matrix_id, setting_id = task
    path = _multiform_path(candidate, matrix_id, setting_id)
    if path.is_file():
        payload = _load_npz(path)
        if str(payload["selection_id"].item()) != _load_selection()["selection_id"]:
            raise ValueError(f"stale multiform checkpoint: {path}")
        return candidate, matrix_id, "existing", int(payload["state_digest"].size)
    payload = _simulate_multiform_cell(candidate, matrix_id, setting_id)
    _atomic_npz(path, **payload)
    return candidate, matrix_id, "generated", int(payload["state_digest"].size)


def confirm(workers: int) -> None:
    verify_protocol()
    selection = _load_selection()
    if not selection["confirmation_authorized"]:
        _update_status(state="confirmation_not_authorized", stage="confirm", message=selection["no_confirmation_reason"])
        print(f"No confirmation futures generated: {selection['no_confirmation_reason']}")
        return
    _run_tasks(_save_primary_task, _primary_tasks(), workers=workers, stage="confirm-primary")
    _run_tasks(_save_multiform_task, _multiform_tasks(), workers=workers, stage="confirm-multiform")


def _metric_mean(payload: dict[str, np.ndarray], field: str, arm_index: int, horizon: int) -> float:
    values = np.asarray(payload[field][arm_index])
    if field.startswith("terminal8_f"):
        landmark = int(field.rsplit("f", 1)[1])
        if horizon < landmark:
            return float("nan")
        return float(np.mean(values == 1))
    return float(np.mean(values))


def _primary_metric_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate, matrix_id, setting_id in _primary_tasks():
        path = _primary_path(candidate, matrix_id, setting_id)
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = _load_npz(path)
        arms = [str(value) for value in payload["arms"]]
        horizon = int(payload["horizon"].item())
        for arm_index, arm in enumerate(arms):
            rows.append(
                {
                    "candidate": candidate,
                    "matrix_id": matrix_id,
                    "setting_id": setting_id,
                    "arm": arm,
                    "n_futures": int(payload["futures"].item()),
                    "horizon": horizon,
                    "terminal8_f16": _metric_mean(payload, "terminal8_f16", arm_index, horizon),
                    "terminal8_f32": _metric_mean(payload, "terminal8_f32", arm_index, horizon),
                    "terminal8_f64": _metric_mean(payload, "terminal8_f64", arm_index, horizon),
                    "capture_any_f16": _metric_mean(payload, "capture_any_f16", arm_index, horizon),
                    "capture_any_f32": _metric_mean(payload, "capture_any_f32", arm_index, horizon),
                    "capture_any_f64": _metric_mean(payload, "capture_any_f64", arm_index, horizon),
                    "arrival_f16": _metric_mean(payload, "arrival_f16", arm_index, horizon),
                    "occupancy": _metric_mean(payload, "occupancy", arm_index, horizon),
                    "maximum_residence": _metric_mean(payload, "maximum_residence", arm_index, horizon),
                    "departed": _metric_mean(payload, "departed", arm_index, horizon),
                    "reentered": _metric_mean(payload, "reentered", arm_index, horizon),
                    "extinction": _metric_mean(payload, "extinct", arm_index, horizon),
                    "final_target_h": _metric_mean(payload, "final_target_h", arm_index, horizon),
                    "carrier_target_h": _metric_mean(payload, "carrier_target_h", arm_index, horizon),
                }
            )
    return rows


def _multiform_metric_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate, matrix_id, setting_id in _multiform_tasks():
        path = _multiform_path(candidate, matrix_id, setting_id)
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = _load_npz(path)
        arms = [str(value) for value in payload["arms"]]
        horizon = int(payload["horizon"].item())
        for arm_index, arm in enumerate(arms):
            rows.append(
                {
                    "candidate": candidate,
                    "matrix_id": matrix_id,
                    "setting_id": setting_id,
                    "arm": arm,
                    "n_futures": int(payload["futures"].item()),
                    "horizon": horizon,
                    "terminal8_f16": _metric_mean(payload, "terminal8_f16", arm_index, horizon),
                    "terminal8_f32": _metric_mean(payload, "terminal8_f32", arm_index, horizon),
                    "terminal8_f64": _metric_mean(payload, "terminal8_f64", arm_index, horizon),
                    "capture_any_f32": _metric_mean(payload, "capture_any_f32", arm_index, horizon),
                    "occupancy": _metric_mean(payload, "occupancy", arm_index, horizon),
                    "origin_correct": float(np.mean(np.asarray(payload["origin_correct"][arm_index]) == 1)),
                    "carrier_origin_correct": float(np.mean(np.asarray(payload["carrier_origin_correct"][arm_index]) == 1)),
                    "final_target_h": _metric_mean(payload, "final_target_h", arm_index, horizon),
                    "final_other_h": _metric_mean(payload, "final_other_h", arm_index, horizon),
                    "extinction": _metric_mean(payload, "extinct", arm_index, horizon),
                }
            )
    return rows


def _arm_vector(table: pd.DataFrame, candidate: str, setting_id: str, arm: str, column: str) -> NDArray[np.float64]:
    cell = table[
        (table["candidate"] == candidate)
        & (table["setting_id"] == setting_id)
        & (table["arm"] == arm)
    ].sort_values("matrix_id")
    return cell[column].to_numpy(dtype=np.float64)


def _ci_dict(values: NDArray, label: str) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return {"point": None, "lower": None, "upper": None, "n_rules": 0}
    point, lower, upper = bootstrap_mean_ci(
        array,
        seed=derive_seed(MASTER_SEED, "bootstrap", label),
        repetitions=BOOTSTRAPS,
    )
    return {"point": point, "lower": lower, "upper": upper, "n_rules": int(array.size)}


def _paired_ci_dict(left: NDArray, right: NDArray, label: str) -> dict[str, Any]:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b)
    if not np.any(valid):
        return {"point": None, "lower": None, "upper": None, "n_rules": 0}
    point, lower, upper = paired_bootstrap_ci(
        a[valid],
        b[valid],
        seed=derive_seed(MASTER_SEED, "bootstrap", label),
        repetitions=BOOTSTRAPS,
    )
    return {"point": point, "lower": lower, "upper": upper, "n_rules": int(np.sum(valid))}


def _candidate_primary_gates(primary: pd.DataFrame, candidate: str, setting: CarrierSetting) -> dict[str, Any]:
    setting_id = setting.setting_id
    correct = _arm_vector(primary, candidate, setting_id, "native_correct", "terminal8_f32")
    zero = _arm_vector(primary, candidate, setting_id, "native_zero", "terminal8_f32")
    correct_ci = _ci_dict(correct, f"primary.{candidate}.{setting_id}.correct_f32")
    correct_zero = _paired_ci_dict(correct, zero, f"primary.{candidate}.{setting_id}.correct-zero")
    controls: dict[str, Any] = {}
    for arm in (
        "native_shuffled",
        "native_reader_off",
        "native_founder_writer_off",
        "native_renewal_off",
    ):
        controls[arm] = _paired_ci_dict(
            correct,
            _arm_vector(primary, candidate, setting_id, arm, "terminal8_f32"),
            f"primary.{candidate}.{setting_id}.correct-{arm}",
        )
    f64 = _ci_dict(
        _arm_vector(primary, candidate, setting_id, "native_correct", "terminal8_f64"),
        f"primary.{candidate}.{setting_id}.correct_f64",
    )
    erase = _arm_vector(primary, candidate, setting_id, "native_erase_after_g2", "terminal8_f32")
    rescue = _arm_vector(primary, candidate, setting_id, "native_erase_rescue_g3", "terminal8_f32")
    effect = float(np.mean(correct) - np.mean(zero))
    removed = float(np.mean(correct) - np.mean(erase))
    restored = float(np.mean(rescue) - np.mean(erase))
    removal_fraction = None if effect <= 0.0 else removed / effect
    restoration_fraction = None if removed <= 0.0 else restored / removed
    stranger_install = _paired_ci_dict(
        _arm_vector(primary, candidate, setting_id, "stranger_correct", "terminal8_f32"),
        _arm_vector(primary, candidate, setting_id, "stranger_zero", "terminal8_f32"),
        f"primary.{candidate}.{setting_id}.stranger_install",
    )
    iso = _paired_ci_dict(
        _arm_vector(primary, candidate, setting_id, "joint_relabel_correct", "terminal8_f32"),
        correct,
        f"primary.{candidate}.{setting_id}.isomorphism",
    )
    controls_pass = all(item["lower"] is not None and item["lower"] > 0.0 for item in controls.values())
    memory_pass = bool(
        correct_ci["lower"] is not None
        and correct_ci["lower"] > 0.30
        and correct_zero["point"] is not None
        and correct_zero["point"] >= 0.20
        and correct_zero["lower"] > 0.10
        and controls_pass
        and f64["lower"] is not None
        and f64["lower"] > 0.0
        and removal_fraction is not None
        and removal_fraction >= 0.70
        and restoration_fraction is not None
        and restoration_fraction >= 0.70
    )
    equivariance_pass = bool(iso["point"] is not None and abs(float(iso["point"])) <= 0.03)
    return {
        "native_correct_terminal8_f32": correct_ci,
        "correct_minus_zero_terminal8_f32": correct_zero,
        "control_differences": controls,
        "native_correct_terminal8_f64": f64,
        "erasure": {
            "effect_correct_minus_zero": effect,
            "removed_correct_minus_erase": removed,
            "restored_rescue_minus_erase": restored,
            "removal_fraction": removal_fraction,
            "restoration_fraction": restoration_fraction,
        },
        "stranger_installation_difference": stranger_install,
        "joint_relabel_minus_native": iso,
        "memory_gate_passed": memory_pass,
        "equivariance_gate_passed": equivariance_pass,
    }


def _candidate_multiform_gates(multiform: pd.DataFrame, candidate: str, setting: CarrierSetting) -> dict[str, Any]:
    setting_id = setting.setting_id
    aa_capture = _arm_vector(multiform, candidate, setting_id, "state_a_carrier_a", "terminal8_f32")
    ab_capture = _arm_vector(multiform, candidate, setting_id, "state_a_carrier_b", "terminal8_f32")
    ba_capture = _arm_vector(multiform, candidate, setting_id, "state_b_carrier_a", "terminal8_f32")
    bb_capture = _arm_vector(multiform, candidate, setting_id, "state_b_carrier_b", "terminal8_f32")
    form_a_cross = _ci_dict(ba_capture, f"multiform.{candidate}.{setting_id}.form_a_cross")
    form_b_cross = _ci_dict(ab_capture, f"multiform.{candidate}.{setting_id}.form_b_cross")
    aa_origin = _arm_vector(multiform, candidate, setting_id, "state_a_carrier_a", "origin_correct")
    ab_origin = _arm_vector(multiform, candidate, setting_id, "state_a_carrier_b", "origin_correct")
    ba_origin = _arm_vector(multiform, candidate, setting_id, "state_b_carrier_a", "origin_correct")
    bb_origin = _arm_vector(multiform, candidate, setting_id, "state_b_carrier_b", "origin_correct")
    crossover_by_rule = 0.5 * ((ab_origin - (1.0 - aa_origin)) + (ba_origin - (1.0 - bb_origin)))
    crossover = _ci_dict(crossover_by_rule, f"multiform.{candidate}.{setting_id}.crossover")
    origin_by_rule = 0.25 * (aa_origin + ab_origin + ba_origin + bb_origin)
    origin = _ci_dict(origin_by_rule, f"multiform.{candidate}.{setting_id}.origin")
    carrier_decode_by_rule = 0.25 * sum(
        _arm_vector(multiform, candidate, setting_id, arm, "carrier_origin_correct")
        for arm in MULTIFORM_ARMS
    )
    carrier_decode = _ci_dict(carrier_decode_by_rule, f"multiform.{candidate}.{setting_id}.carrier_decode")
    passed = bool(
        form_a_cross["lower"] is not None
        and form_a_cross["lower"] > 0.25
        and form_b_cross["lower"] is not None
        and form_b_cross["lower"] > 0.25
        and crossover["point"] is not None
        and crossover["point"] >= 0.20
        and crossover["lower"] > 0.10
        and origin["lower"] is not None
        and origin["lower"] > 0.75
    )
    return {
        "cross_start_form_a_terminal8_f32": form_a_cross,
        "cross_start_form_b_terminal8_f32": form_b_cross,
        "carrier_crossover": crossover,
        "carrier_origin_accuracy": origin,
        "final_carrier_decoding": carrier_decode,
        "multiform_gate_passed": passed,
        "same_start_capture_context": {
            "a_with_a": _ci_dict(aa_capture, f"multiform.{candidate}.{setting_id}.aa"),
            "b_with_b": _ci_dict(bb_capture, f"multiform.{candidate}.{setting_id}.bb"),
        },
    }


def analyze() -> None:
    verify_protocol()
    selection = _load_selection()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if not selection["confirmation_authorized"]:
        summary = {
            "format": FORMAT,
            "protocol_id": _read_json(PROTOCOL_PATH)["protocol_id"],
            "selection_id": selection["selection_id"],
            "complete": True,
            "confirmation_run": False,
            "classification": "calibration_stopped_before_confirmation",
            "reason": selection["no_confirmation_reason"],
            "scientific_verdict": "No confirmation verdict; the engineering screen did not authorize a frozen confirmation tier.",
            "reporting_boundary": "exploratory; not preprint evidence; not a replication",
        }
        _write_json(OUTPUT_ROOT / "primary_summary.json", summary)
        _update_status(state="analyzed", stage="analyze", message=summary["classification"])
        return
    missing_primary = [task for task in _primary_tasks() if not _primary_path(*task).is_file()]
    missing_multiform = [task for task in _multiform_tasks() if not _multiform_path(*task).is_file()]
    if missing_primary or missing_multiform:
        raise FileNotFoundError(f"confirmation checkpoints incomplete: primary={len(missing_primary)} multiform={len(missing_multiform)}")
    primary = pd.DataFrame(_primary_metric_rows())
    multiform = pd.DataFrame(_multiform_metric_rows())
    primary.to_csv(OUTPUT_ROOT / "primary_rule_metrics.csv", index=False)
    multiform.to_csv(OUTPUT_ROOT / "multiform_rule_metrics.csv", index=False)
    setting_results: list[dict[str, Any]] = []
    for setting in _selected_settings():
        candidate_results: dict[str, Any] = {}
        for candidate in CANDIDATE_NAMES:
            primary_result = _candidate_primary_gates(primary, candidate, setting)
            multiform_result = _candidate_multiform_gates(multiform, candidate, setting)
            candidate_results[candidate] = {"primary": primary_result, "multiform": multiform_result}
        memory_both = all(candidate_results[candidate]["primary"]["memory_gate_passed"] for candidate in CANDIDATE_NAMES)
        multiform_both = all(candidate_results[candidate]["multiform"]["multiform_gate_passed"] for candidate in CANDIDATE_NAMES)
        equivariance_both = all(candidate_results[candidate]["primary"]["equivariance_gate_passed"] for candidate in CANDIDATE_NAMES)
        constructive = bool(memory_both and multiform_both and equivariance_both)
        compressed_noisy = bool(constructive and setting.copy_mode == "nominal" and setting.k <= 32)
        full_register = bool(constructive and setting.k == 100)
        setting_results.append(
            {
                **setting.to_dict(),
                "setting_id": setting.setting_id,
                "candidates": candidate_results,
                "memory_both_candidates": memory_both,
                "multiform_both_candidates": multiform_both,
                "equivariance_both_candidates": equivariance_both,
                "constructive_carrier_pass": constructive,
                "compressed_noisy_pass": compressed_noisy,
                "engineered_full_register_pass": full_register,
            }
        )
    if any(row["compressed_noisy_pass"] for row in setting_results):
        classification = "compressed_noisy_carrier_pass"
        verdict = "A compressed noisy external register constructively maintained and selected forms under both simulator contracts."
    elif any(row["engineered_full_register_pass"] for row in setting_results):
        classification = "engineered_full_register_only"
        verdict = "Only a full molecule-indexed external register passed; this is an engineered side channel, not endogenous GARD heredity."
    elif any(row["constructive_carrier_pass"] for row in setting_results):
        classification = "constructive_noncompressed_carrier_pass"
        verdict = "An engineered carrier passed, but not the registered compressed/noisy tier."
    else:
        classification = "registered_carrier_family_failed_constructive_gate"
        verdict = "The selected carrier settings did not jointly pass maintenance, causal ablation/rescue, multiform selection, and relabeling gates in both contracts."
    summary = {
        "format": FORMAT,
        "protocol_id": _read_json(PROTOCOL_PATH)["protocol_id"],
        "selection_id": selection["selection_id"],
        "complete": True,
        "confirmation_run": True,
        "tier": selection["tier"],
        "classification": classification,
        "scientific_verdict": verdict,
        "settings": setting_results,
        "reporting_boundary": "exploratory hypothesis stress test; not current-preprint evidence; not Wagner or independent GARD replication",
    }
    _write_json(OUTPUT_ROOT / "primary_summary.json", summary)
    _update_status(state="analyzed", stage="analyze", message=classification)
    print(json.dumps({"classification": classification, "scientific_verdict": verdict}, indent=2))


def _percentage(value: Any) -> str:
    return "not evaluated" if value is None else f"{100.0 * float(value):.1f}%"


def report() -> None:
    if not (OUTPUT_ROOT / "primary_summary.json").is_file():
        raise FileNotFoundError("run analyze before report")
    summary = _read_json(OUTPUT_ROOT / "primary_summary.json")
    selection = _load_selection()
    lines = [
        "# Scientific report: clean-room GARD lineage carrier",
        "",
        "## Boundary first",
        "",
        "This is an exploratory reviewer-motivated stress test of a claim the current preprint does not make. It is not preprint evidence, not an independent replication, and not a replication of the Wagner GRN work. Candidates 02 and 03 are alternative reconstruction contracts. Exact replay is computational reproducibility only.",
        "",
        "## Registered outcome",
        "",
        f"**Classification:** `{summary['classification']}`",
        "",
        summary["scientific_verdict"],
        "",
        "The carrier is an added inherited molecule-indexed register. Even a positive result therefore shows what an engineered side channel can do; it does not show that unmodified GARD already contains that mechanism.",
        "",
        "## Engineering and confirmation",
        "",
        f"The engineering grid tested {len(_all_settings())} settings on matrices {', '.join(map(str, ENGINEERING_RULES))} in both contracts. It froze {selection['selected_setting_count']} setting(s) before confirmation.",
    ]
    if selection.get("tier") is not None:
        tier = selection["tier"]
        lines.extend(
            [
                f"The benchmark selected tier {tier['tier']}: {tier['futures']} futures per arm through F{tier['horizon']}.",
                "",
            ]
        )
    if summary.get("confirmation_run"):
        lines.extend(["## Gate detail", ""])
        for setting in summary["settings"]:
            lines.append(
                f"### {setting['setting_id']} (k={setting['k']}, L={setting['half_life']}, coupling={setting['coupling']}, {setting['copy_mode']})"
            )
            lines.append("")
            lines.append(
                f"Overall: carrier memory both contracts={setting['memory_both_candidates']}; multiform both={setting['multiform_both_candidates']}; relabeling both={setting['equivariance_both_candidates']}; constructive pass={setting['constructive_carrier_pass']}."
            )
            lines.append("")
            for candidate in CANDIDATE_NAMES:
                result = setting["candidates"][candidate]
                primary = result["primary"]
                multiform = result["multiform"]
                lines.append(
                    f"- Contract {candidate}: native correct F32 terminal strict-8 {_percentage(primary['native_correct_terminal8_f32']['point'])} (95% rule-bootstrap {_percentage(primary['native_correct_terminal8_f32']['lower'])}–{_percentage(primary['native_correct_terminal8_f32']['upper'])}); correct-minus-zero {_percentage(primary['correct_minus_zero_terminal8_f32']['point'])}; cross-form A {_percentage(multiform['cross_start_form_a_terminal8_f32']['point'])}, B {_percentage(multiform['cross_start_form_b_terminal8_f32']['point'])}; origin accuracy {_percentage(multiform['carrier_origin_accuracy']['point'])}."
                )
            lines.append("")
    lines.extend(
        [
            "## Interpretation limits",
            "",
            "A 100-coordinate pass is reported as an engineered full-register result only. A nominal noisy pass with k≤32 is the stronger compressed tier. A null result limits only this frozen carrier family and completed runtime tier. These outcomes must not be merged into current-preprint evidence tables.",
            "",
        ]
    )
    (OUTPUT_ROOT / "SCIENTIFIC_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    lay = [
        "# Lay summary",
        "",
        "We added a separate, inheritable ‘notebook’ to each simulated GARD assembly. The notebook can bias which molecule types are available as the assembly grows, and the adult assembly can rewrite the notebook before it is passed on.",
        "",
        summary["scientific_verdict"],
        "",
        "This does not change the claims of the current preprint. The notebook is something we engineered into the model, and this follow-up was designed after seeing related exploratory work. It is useful for deciding what to test next, not as confirmation of the original paper.",
        "",
    ]
    (OUTPUT_ROOT / "LAY_SUMMARY.md").write_text("\n".join(lay), encoding="utf-8")
    methods = [
        "# Methods and audit map",
        "",
        "- `../protocol/protocol.json`: sealed scientific design and implementation hashes",
        "- `../protocol/confirmation_selection.json`: outcome-dependent engineering selection frozen before confirmation",
        "- `calibration_rule_metrics.csv`: equal-rule engineering summaries",
        "- `primary_rule_metrics.csv`: single-form confirmation summaries (when authorized)",
        "- `multiform_rule_metrics.csv`: reciprocal two-form summaries (when authorized)",
        "- `primary_summary.json`: registered gates and classification",
        "- `../verification/verification_audit.json`: exact replay and checksum audit",
        "",
        "No full daughter trajectories are discarded without an audit trace: each future checkpoint contains its SHA-256 daughter-trajectory digest plus the complete boundary-H, target-H, and carrier-decoding traces. Full verification regenerates every future and compares those arrays.",
        "",
    ]
    (OUTPUT_ROOT / "METHODS_AND_AUDIT.md").write_text("\n".join(methods), encoding="utf-8")
    boundary_copy = (TASK_ROOT / "REPORTING_BOUNDARY.md").read_text(encoding="utf-8")
    (OUTPUT_ROOT / "READ_ME_FIRST_REPORTING_BOUNDARY.md").write_text(boundary_copy, encoding="utf-8")
    _write_checksums(OUTPUT_ROOT)
    _update_status(state="reported", stage="report", message=summary["classification"])
    print(f"Reports written under {OUTPUT_ROOT}")


def _compare_payload(stored: dict[str, np.ndarray], replay: dict[str, Any]) -> tuple[bool, bool, float]:
    if set(stored) != set(replay):
        return False, False, float("inf")
    discrete_exact = True
    all_exact = True
    maximum_h_error = 0.0
    for key in stored:
        left = np.asarray(stored[key])
        right = np.asarray(replay[key])
        if left.shape != right.shape or left.dtype != right.dtype:
            return False, False, float("inf")
        if key == "boundary_h":
            valid = np.isfinite(left) & np.isfinite(right)
            if np.any(np.isfinite(left) != np.isfinite(right)):
                maximum_h_error = float("inf")
            elif np.any(valid):
                maximum_h_error = max(maximum_h_error, float(np.max(np.abs(left[valid] - right[valid]))))
        if np.issubdtype(left.dtype, np.floating):
            equal = bool(np.array_equal(left, right, equal_nan=True))
        else:
            equal = bool(np.array_equal(left, right))
        all_exact = all_exact and equal
        if key == "state_digest" or np.issubdtype(left.dtype, np.integer) or np.issubdtype(left.dtype, np.bool_):
            discrete_exact = discrete_exact and equal
    return all_exact, discrete_exact, maximum_h_error


def _replay_receipt_path(kind: str, candidate: str, matrix_id: int, setting_id: str) -> Path:
    suffix = "calibration" if kind == "calibration" else setting_id
    return VERIFICATION_ROOT / "replay" / kind / suffix / f"c{candidate}_m{matrix_id:03d}.json"


def _replay_job(job: tuple[str, str, int, str]) -> dict[str, Any]:
    kind, candidate, matrix_id, setting_id = job
    if kind == "calibration":
        path = _calibration_path(candidate, matrix_id)
    elif kind == "primary":
        path = _primary_path(candidate, matrix_id, setting_id)
    elif kind == "multiform":
        path = _multiform_path(candidate, matrix_id, setting_id)
    else:
        raise ValueError(f"unknown replay kind: {kind}")
    checkpoint_sha = sha256_file(path)
    receipt_path = _replay_receipt_path(kind, candidate, matrix_id, setting_id)
    if receipt_path.is_file():
        receipt = _read_json(receipt_path)
        receipt_payload = {key: value for key, value in receipt.items() if key != "receipt_id"}
        if (
            receipt.get("receipt_id") == _canonical_digest(receipt_payload)
            and receipt.get("protocol_id") == _read_json(PROTOCOL_PATH)["protocol_id"]
            and receipt.get("selection_id") == _load_selection()["selection_id"]
            and receipt.get("sha256") == checkpoint_sha
            and receipt.get("all_arrays_exact") is True
            and receipt.get("discrete_trajectory_digests_exact") is True
            and float(receipt.get("maximum_h_error", float("inf"))) == 0.0
        ):
            return receipt
    if kind == "calibration":
        replay = _simulate_calibration_cell(candidate, matrix_id)
    elif kind == "primary":
        replay = _simulate_primary_cell(candidate, matrix_id, setting_id)
    else:
        replay = _simulate_multiform_cell(candidate, matrix_id, setting_id)
    stored = _load_npz(path)
    exact, discrete, maximum_h_error = _compare_payload(stored, replay)
    result = {
        "kind": kind,
        "candidate": candidate,
        "matrix_id": matrix_id,
        "setting_id": setting_id or None,
        "path": str(path.resolve()),
        "sha256": checkpoint_sha,
        "futures": int(stored["state_digest"].size),
        "all_arrays_exact": exact,
        "discrete_trajectory_digests_exact": discrete,
        "maximum_h_error": maximum_h_error,
        "protocol_id": _read_json(PROTOCOL_PATH)["protocol_id"],
        "selection_id": _load_selection()["selection_id"],
    }
    result["receipt_id"] = _canonical_digest(result)
    _write_json(receipt_path, result)
    return result


def _expected_replay_jobs() -> list[tuple[str, str, int, str]]:
    jobs = [("calibration", candidate, matrix_id, "") for candidate, matrix_id in _calibration_tasks()]
    selection = _load_selection()
    if selection["confirmation_authorized"]:
        jobs.extend(("primary", candidate, matrix_id, setting_id) for candidate, matrix_id, setting_id in _primary_tasks())
        jobs.extend(("multiform", candidate, matrix_id, setting_id) for candidate, matrix_id, setting_id in _multiform_tasks())
    return jobs


def verify(*, full_replay: bool, workers: int) -> None:
    protocol = verify_protocol()
    selection = _load_selection()
    protocol_checksums = _verify_checksums(PROTOCOL_ROOT)
    output_checksums = _verify_checksums(OUTPUT_ROOT)
    jobs = _expected_replay_jobs()
    missing = []
    for kind, candidate, matrix_id, setting_id in jobs:
        path = (
            _calibration_path(candidate, matrix_id)
            if kind == "calibration"
            else _primary_path(candidate, matrix_id, setting_id)
            if kind == "primary"
            else _multiform_path(candidate, matrix_id, setting_id)
        )
        if not path.is_file():
            missing.append(str(path))
    if missing:
        raise FileNotFoundError(f"missing expected checkpoints: {len(missing)}")
    started = time.time()
    checkpoint_rows: list[dict[str, Any]] = []
    if full_replay:
        _update_status(state="running", stage="verify-replay", total=len(jobs), started_at=started)
        if workers <= 1:
            iterator: Iterable[dict[str, Any]] = map(_replay_job, jobs)
            for index, result in enumerate(iterator, start=1):
                checkpoint_rows.append(result)
                _update_status(state="running", stage="verify-replay", completed=index, total=len(jobs), started_at=started, message=f"{result['kind']} c{result['candidate']} m{result['matrix_id']:03d}")
                _check_soft_stop()
        else:
            completed = 0
            for offset in range(0, len(jobs), workers):
                batch = jobs[offset : offset + workers]
                with ProcessPoolExecutor(max_workers=min(workers, len(batch))) as executor:
                    results = list(executor.map(_replay_job, batch, chunksize=1))
                for result in results:
                    checkpoint_rows.append(result)
                    completed += 1
                    _update_status(state="running", stage="verify-replay", completed=completed, total=len(jobs), started_at=started, message=f"{result['kind']} c{result['candidate']} m{result['matrix_id']:03d}")
                _check_soft_stop()
    else:
        for job in jobs:
            kind, candidate, matrix_id, setting_id = job
            path = _calibration_path(candidate, matrix_id) if kind == "calibration" else _primary_path(candidate, matrix_id, setting_id) if kind == "primary" else _multiform_path(candidate, matrix_id, setting_id)
            payload = _load_npz(path)
            checkpoint_rows.append(
                {
                    "kind": kind,
                    "candidate": candidate,
                    "matrix_id": matrix_id,
                    "setting_id": setting_id or None,
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                    "futures": int(payload["state_digest"].size),
                    "all_arrays_exact": None,
                    "discrete_trajectory_digests_exact": None,
                    "maximum_h_error": None,
                }
            )
    replay_exact = bool(full_replay and all(row["all_arrays_exact"] for row in checkpoint_rows))
    discrete_exact = bool(full_replay and all(row["discrete_trajectory_digests_exact"] for row in checkpoint_rows))
    maximum_h_error = (
        max(float(row["maximum_h_error"]) for row in checkpoint_rows)
        if full_replay and checkpoint_rows
        else None
    )
    complete = bool(
        full_replay
        and replay_exact
        and discrete_exact
        and maximum_h_error == 0.0
        and all(protocol_checksums.values())
        and all(output_checksums.values())
    )
    audit = {
        "format": FORMAT,
        "protocol_id": protocol["protocol_id"],
        "selection_id": selection["selection_id"],
        "complete": complete,
        "full_replay_requested": full_replay,
        "checkpoint_files": len(checkpoint_rows),
        "replayed_futures": sum(row["futures"] for row in checkpoint_rows) if full_replay else 0,
        "all_arrays_replay_exact": replay_exact,
        "discrete_replay_exact": discrete_exact,
        "maximum_h_error": maximum_h_error,
        "protocol_checksums_verified": all(protocol_checksums.values()),
        "output_checksums_verified": all(output_checksums.values()),
        "cleanroom_firewall_passed": _firewall_audit()["passed"],
        "wagner_code_used": False,
        "scientific_replication_claimed": False,
        "checkpoints": checkpoint_rows,
    }
    _write_json(VERIFICATION_ROOT / "verification_audit.json", audit)
    _write_checksums(VERIFICATION_ROOT)
    if full_replay and not complete:
        raise ValueError("full verification did not replay exactly")
    _update_status(
        state="complete" if complete else "verified_without_full_replay",
        stage="complete" if complete else "verify",
        completed=len(jobs),
        total=len(jobs),
        started_at=started,
        message=f"replayed_futures={audit['replayed_futures']}",
    )
    print(json.dumps({key: value for key, value in audit.items() if key != "checkpoints"}, indent=2))


def smoke() -> None:
    verify_protocol()
    for candidate, matrix_id in (("02", 11), ("03", 54)):
        row = _registry_cell(candidate, matrix_id)
        beta = _beta(matrix_id)
        setting = CarrierSetting(k=100, half_life=4, coupling=2.0, copy_mode="nominal")
        mask = _mask_values(candidate, matrix_id, 100)
        seed = derive_seed(MASTER_SEED, "smoke", candidate, matrix_id)
        readout, h, _, _ = simulate_carrier_future(
            row["target"], row["target"], None, beta, GARD, CANDIDATES[candidate], setting, mask,
            _policy("native_no_carrier"), dynamics_seed=seed, carrier_seed=derive_seed(MASTER_SEED, "smoke", "carrier", candidate, matrix_id), horizon=4,
        )
        rng = np.random.default_rng(seed)
        current = row["target"].astype(np.int64).copy()
        expected_h = []
        for _ in range(4):
            record = advance_fission(current, beta, GARD, CANDIDATES[candidate], rng)
            expected_h.append(record.h)
            current = record.daughter
        if readout.observed != 4 or not np.array_equal(h, np.asarray(expected_h), equal_nan=True):
            raise AssertionError(f"zero-carrier smoke parity failed for candidate {candidate}")
    print("Two-candidate zero-carrier parity smoke passed; production seeds were not consumed.")


def run_tests() -> None:
    command = [sys.executable, "-m", "pytest", str(TASK_ROOT / "test_carrier.py"), "-q"]
    subprocess.run(command, cwd=CODEX_ROOT, check=True)


def status() -> None:
    if not STATUS_PATH.is_file():
        print(json.dumps({"state": "not_started", "stage": "none", "cumulative_elapsed_seconds": _ledger_elapsed()}, indent=2))
        return
    payload = _read_json(STATUS_PATH)
    payload["cumulative_elapsed_seconds"] = _ledger_elapsed()
    payload["soft_remaining_seconds"] = max(0.0, SOFT_LIMIT_SECONDS - _ledger_elapsed())
    payload["hard_remaining_seconds"] = max(0.0, HARD_LIMIT_SECONDS - _ledger_elapsed())
    if _ledger_elapsed() >= HARD_LIMIT_SECONDS and payload.get("state") != "complete":
        payload["state"] = "incomplete_walltime_hard_stop"
        payload["stage"] = "hard-stop"
        payload["message"] = "The cumulative eight-hour hard limit was exhausted; incomplete work has no scientific verdict."
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
        calibrate(workers)
        confirm(workers)
        analyze()
        report()
        verify(full_replay=True, workers=workers)
        complete = True
    except SoftStop as error:
        _stop_meter()
        _update_status(state="incomplete_walltime_soft_stop", stage="soft-stop", message=str(error), error=None)
        print(f"Campaign checkpointed without a verdict: {error}", flush=True)
        return
    except Exception as error:
        _stop_meter()
        _update_status(state="failed", stage="failed", message="pipeline stopped", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        _stop_meter()
    if complete:
        current = _read_json(STATUS_PATH)
        _update_status(
            state="complete",
            stage="complete",
            completed=int(current.get("completed", 0)),
            total=int(current.get("total", 0)),
            message=str(current.get("message", "complete")),
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("prepare")
    subcommands.add_parser("test")
    subcommands.add_parser("smoke")
    benchmark_parser = subcommands.add_parser("benchmark")
    benchmark_parser.add_argument("--workers", type=int, default=16)
    calibration_parser = subcommands.add_parser("calibrate")
    calibration_parser.add_argument("--workers", type=int, default=16)
    confirmation_parser = subcommands.add_parser("confirm")
    confirmation_parser.add_argument("--workers", type=int, default=16)
    subcommands.add_parser("analyze")
    subcommands.add_parser("report")
    verify_parser = subcommands.add_parser("verify")
    verify_parser.add_argument("--workers", type=int, default=16)
    verify_parser.add_argument("--full-replay", action="store_true")
    all_parser = subcommands.add_parser("all")
    all_parser.add_argument("--workers", type=int, default=16)
    subcommands.add_parser("status")
    subcommands.add_parser("remaining-hard", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    if arguments.command == "prepare":
        prepare()
    elif arguments.command == "test":
        run_tests()
    elif arguments.command == "smoke":
        smoke()
    elif arguments.command == "benchmark":
        benchmark(arguments.workers)
    elif arguments.command == "calibrate":
        calibrate(arguments.workers)
    elif arguments.command == "confirm":
        confirm(arguments.workers)
    elif arguments.command == "analyze":
        analyze()
    elif arguments.command == "report":
        report()
    elif arguments.command == "verify":
        verify(full_replay=arguments.full_replay, workers=arguments.workers)
    elif arguments.command == "all":
        run_all(arguments.workers)
    elif arguments.command == "status":
        status()
    elif arguments.command == "remaining-hard":
        remaining_hard()


if __name__ == "__main__":
    main()
