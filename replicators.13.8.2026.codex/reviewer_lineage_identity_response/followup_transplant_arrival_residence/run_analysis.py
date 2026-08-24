"""Reproducible runner for the strict-B transplant and residence follow-up."""

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
import sys
import time
from typing import Any, Iterable, Sequence


TASK_ROOT = Path(__file__).resolve().parent
PARENT_TASK = TASK_ROOT.parent
CODEX_ROOT = PARENT_TASK.parent
WORKSPACE_ROOT = CODEX_ROOT.parent
if str(CODEX_ROOT) not in sys.path:
    sys.path.insert(0, str(CODEX_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(TASK_ROOT / "artifacts" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from threadpoolctl import threadpool_limits

from plastic_heredity.config import CANDIDATES, GardConfig
from plastic_heredity.mechanistic import sha256_file
from plastic_heredity.regime_confirmation import CONFIRMATION_MASTER_SEED
from plastic_heredity.seeds import derive_seed
from plastic_heredity.simulator import SimulationError, advance_fission, generate_beta
from reviewer_lineage_identity_response.followup_transplant_arrival_residence.transplant_core import (
    CAPTURE_WINDOW,
    DEPARTURE_THRESHOLD,
    HORIZON,
    INHERITANCE_THRESHOLD,
    bootstrap_mean_ci,
    bray_curtis_similarity,
    choose_rule_permutation,
    cosine,
    first_capture_class,
    first_target_capture,
    first_target_capture_metric,
    generate_mass_preserving_perturbations,
    inverse_permute_state,
    medoid,
    permute_beta,
    permute_state,
    score_future,
)


ARTIFACT_ROOT = TASK_ROOT / "artifacts"
PROTOCOL_ROOT = ARTIFACT_ROOT / "protocol"
WORK_ROOT = ARTIFACT_ROOT / "work"
PRIMARY_ROOT = WORK_ROOT / "primary"
RARE_ROOT = WORK_ROOT / "rare"
OUTPUT_ROOT = ARTIFACT_ROOT / "output"
VERIFICATION_ROOT = ARTIFACT_ROOT / "verification"
STATUS_PATH = ARTIFACT_ROOT / "STATUS.json"
PROTOCOL_PATH = PROTOCOL_ROOT / "protocol.json"
REGISTRATION_PATH = PROTOCOL_ROOT / "registration.json"
DONOR_PATH = PROTOCOL_ROOT / "donor_registry.csv"
PERMUTATION_PATH = PROTOCOL_ROOT / "permutation_registry.json"
RARE_REGISTRY_PATH = PROTOCOL_ROOT / "rare_start_registry.csv"
SOURCE_MANIFEST_PATH = PROTOCOL_ROOT / "scientific_source_manifest.json"
HYPOTHESIS_MANIFEST_PATH = PROTOCOL_ROOT / "hypothesis_only_manifest.json"
SEED_REGISTRY_PATH = PROTOCOL_ROOT / "seed_registry.json"

PARENT_PROTOCOL = PARENT_TASK / "artifacts" / "protocol" / "protocol.json"
PARENT_SELECTION = PARENT_TASK / "artifacts" / "protocol" / "matrix_selection.csv"
PARENT_B_BANK = PARENT_TASK / "artifacts" / "output" / "b_bank.csv"
PARENT_STABLE_FORMS = PARENT_TASK / "artifacts" / "output" / "stable_forms.csv"
PARENT_AUDIT = PARENT_TASK / "artifacts" / "verification" / "verification_audit.json"
PARENT_LINEAGE_ROOT = PARENT_TASK / "artifacts" / "work" / "lineages"

HYPOTHESIS_PATHS = {
    "ideas": WORKSPACE_ROOT
    / "NewIdeas/preprints/ingressing-minds-v-ruliad-paper-ideas/IDEAS.md",
    "form_atlas_expectations": WORKSPACE_ROOT
    / "NewIdeas/preprints/ingressing-minds-v-ruliad-paper-ideas/form_atlas/EXPECTATIONS.md",
    "causal_heredity_protocol": WORKSPACE_ROOT
    / "NewIdeas/preprints/ingressing-minds-v-ruliad-paper-ideas/codex.reconstructionsAndStressTesting/CAUSAL_HEREDITY_PROTOCOL.md",
}

FORMAT = "strict-b-transplant-arrival-residence-v1"
MASTER_SEED = "202608211604-strict-b-transplant-v1"
RULES = 50
DONORS_PER_CELL = 3
FUTURES = 128
BOOTSTRAPS = 10_000
PERMUTATION_PROPOSALS = 4_096
RARE_RULES = (11, 54, 63)
RARE_PERTURBATIONS = 8
RARE_PROPOSALS = 4_096
RARE_DOSE_LADDER = (4, 8, 12, 16, 20, 24)
ARMS = ("native", "state_only", "rule_only", "joint", "natural_stranger")
GARD = GardConfig(generations=HORIZON)

BASE_SCIENTIFIC_SOURCE_PATHS = {
    "runner": Path(__file__),
    "core": TASK_ROOT / "transplant_core.py",
    "readme": TASK_ROOT / "README.md",
    "review_plan": TASK_ROOT / "REVIEW_AND_PLAN.md",
    "hypothesis_boundary": TASK_ROOT / "HYPOTHESIS_BOUNDARY.md",
    "parent_protocol": PARENT_PROTOCOL,
    "parent_selection": PARENT_SELECTION,
    "parent_b_bank": PARENT_B_BANK,
    "parent_stable_forms": PARENT_STABLE_FORMS,
    "parent_verification_audit": PARENT_AUDIT,
    "simulator": CODEX_ROOT / "plastic_heredity" / "simulator.py",
    "config": CODEX_ROOT / "plastic_heredity" / "config.py",
    "seeds": CODEX_ROOT / "plastic_heredity" / "seeds.py",
    "regime_confirmation": CODEX_ROOT / "plastic_heredity" / "regime_confirmation.py",
    "requirements": CODEX_ROOT / "requirements-lock.txt",
}


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(_json_ready(value), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _assert_local(path: Path) -> None:
    resolved = path.resolve()
    if resolved != TASK_ROOT and TASK_ROOT not in resolved.parents:
        raise ValueError(f"refusing write outside follow-up folder: {resolved}")


def _write_json(path: Path, value: Any) -> None:
    _assert_local(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_json_ready(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    _assert_local(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: _csv_value(row.get(key)) for key in fieldnames} for row in rows])
    temporary.replace(path)


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict, np.ndarray)):
        return json.dumps(_json_ready(value), separators=(",", ":"))
    return _json_ready(value)


def _atomic_npz(path: Path, **values: Any) -> None:
    _assert_local(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **values)
    temporary.replace(path)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as bundle:
        return {name: bundle[name] for name in bundle.files}


def _runtime() -> dict[str, str]:
    packages = ("numpy", "pandas", "scipy", "matplotlib", "threadpoolctl")
    return {
        "python": platform.python_version(),
        **{name: importlib.metadata.version(name) for name in packages},
    }


def _manifest(paths: dict[str, Path], *, classification: str) -> dict[str, Any]:
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing manifest inputs: {missing}")
    return {
        "classification": classification,
        "files": {
            name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for name, path in sorted(paths.items())
        },
    }


def _write_checksums(directory: Path) -> None:
    _assert_local(directory)
    rows = []
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        if path.name == "SHA256SUMS":
            continue
        rows.append(f"{sha256_file(path)}  {path.relative_to(directory)}")
    (directory / "SHA256SUMS").write_text("\n".join(rows) + "\n", encoding="utf-8")


def _verify_checksums(directory: Path) -> dict[str, bool]:
    checksum = directory / "SHA256SUMS"
    if not checksum.is_file():
        raise FileNotFoundError(checksum)
    output: dict[str, bool] = {}
    for line in checksum.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        path = directory / relative
        output[relative] = path.is_file() and sha256_file(path) == digest
    if not output or not all(output.values()):
        raise ValueError(f"checksum verification failed: {[key for key, value in output.items() if not value]}")
    return output


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
    now = time.time()
    start = now if started_at is None else started_at
    elapsed = max(0.0, now - start)
    rate = completed / elapsed if elapsed > 0 and completed else 0.0
    eta = (total - completed) / rate if rate > 0 and total >= completed else None
    payload = {
        "format": FORMAT,
        "state": state,
        "stage": stage,
        "completed": completed,
        "total": total,
        "elapsed_seconds": elapsed,
        "throughput_cells_per_second": rate,
        "eta_seconds": eta,
        "message": message,
        "error": error,
        "pid": os.getpid(),
        "updated_at_epoch": now,
        "protocol_id": _read_json(PROTOCOL_PATH).get("protocol_id") if PROTOCOL_PATH.is_file() else None,
    }
    _write_json(STATUS_PATH, payload)


def _seed_collision_audit() -> dict[str, Any]:
    collisions: list[str] = []
    for path in CODEX_ROOT.rglob("*"):
        if not path.is_file() or TASK_ROOT in path.parents:
            continue
        if any(part in {".venv", ".git", "__pycache__", "artifacts"} for part in path.parts):
            continue
        if path.suffix not in {".py", ".json", ".md", ".sh"}:
            continue
        try:
            if MASTER_SEED in path.read_text(encoding="utf-8"):
                collisions.append(str(path.relative_to(CODEX_ROOT)))
        except (UnicodeDecodeError, OSError):
            continue
    if collisions:
        raise ValueError(f"master seed collision outside package: {collisions}")
    return {"master_seed": MASTER_SEED, "collisions": collisions, "passed": True}


def _selected_rules() -> list[int]:
    table = pd.read_csv(PARENT_SELECTION)
    if "matrix_id" not in table.columns or len(table) != RULES:
        raise ValueError("parent matrix selection is not the frozen 50-rule cohort")
    return [int(value) for value in table["matrix_id"]]


def _scientific_source_paths() -> dict[str, Path]:
    paths = dict(BASE_SCIENTIFIC_SOURCE_PATHS)
    for candidate in ("02", "03"):
        for matrix_id in _selected_rules():
            paths[f"parent_fixed_c{candidate}_m{matrix_id:03d}"] = (
                PARENT_LINEAGE_ROOT / f"c{candidate}_m{matrix_id:03d}_fixed.npz"
            )
    return paths


def _parent_integrity() -> dict[str, Any]:
    audit = _read_json(PARENT_AUDIT)
    required = {
        "complete": True,
        "discrete_replay_exact": True,
        "output_checksums_verified": True,
    }
    failed = [key for key, value in required.items() if audit.get(key) is not value]
    if failed or float(audit.get("maximum_h_error", float("inf"))) != 0.0:
        raise ValueError(f"parent verification audit is not complete: {failed}")
    return {
        "format": audit.get("format"),
        "protocol_id": audit.get("protocol_id"),
        "registration_id": audit.get("registration_id"),
        "replayed_lineages": audit.get("replayed_lineages"),
        "complete": True,
    }


def _strict_bank() -> pd.DataFrame:
    table = pd.read_csv(PARENT_B_BANK, dtype={"candidate": str})
    table = table[table["kind"] == "strict"].copy()
    required = {"candidate", "matrix_id", "bank_index", "lineage", "daughters", "final_B"}
    if not required.issubset(table.columns):
        raise ValueError("parent B bank lacks required columns")
    table["matrix_id"] = table["matrix_id"].astype(int)
    table["bank_index"] = table["bank_index"].astype(int)
    return table.sort_values(["candidate", "matrix_id", "bank_index"])


def _donor_rows(bank: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selected = set(_selected_rules())
    for candidate in ("02", "03"):
        for matrix_id in _selected_rules():
            cell = bank[(bank["candidate"] == candidate) & (bank["matrix_id"] == matrix_id)]
            if len(cell) < DONORS_PER_CELL:
                raise ValueError(f"c{candidate} m{matrix_id:03d} has fewer than three strict B donors")
            all_states = [np.asarray(json.loads(value), dtype=np.uint8) for value in cell["final_B"]]
            cell_records = list(cell.to_dict("records"))
            for donor_index, record in enumerate(cell_records[:DONORS_PER_CELL]):
                state = np.asarray(json.loads(record["final_B"]), dtype=np.uint8)
                daughters = np.asarray(json.loads(record["daughters"]), dtype=np.uint8)
                episode_medoid = medoid(daughters)
                natural_candidates: list[tuple[float, int, np.ndarray]] = []
                for other_record, other_state in zip(cell_records, all_states):
                    if int(other_record["lineage"]) == int(record["lineage"]):
                        continue
                    similarity = cosine(state, other_state)
                    if similarity <= DEPARTURE_THRESHOLD:
                        natural_candidates.append((similarity, int(other_record["bank_index"]), other_state))
                natural_candidates.sort(key=lambda item: (item[0], item[1]))
                natural = natural_candidates[0] if natural_candidates else None
                rows.append(
                    {
                        "candidate": candidate,
                        "matrix_id": matrix_id,
                        "donor_index": donor_index,
                        "bank_index": int(record["bank_index"]),
                        "lineage": int(record["lineage"]),
                        "final_B": state,
                        "episode_medoid": episode_medoid,
                        "natural_stranger_available": natural is not None,
                        "natural_stranger_h": None if natural is None else natural[0],
                        "natural_stranger_bank_index": None if natural is None else natural[1],
                        "natural_stranger": None if natural is None else natural[2],
                    }
                )
    if len(rows) != RULES * 2 * DONORS_PER_CELL or any(row["matrix_id"] not in selected for row in rows):
        raise ValueError("donor registry size or rule membership is invalid")
    return rows


def _permutation_registry(donors: Sequence[dict[str, Any]]) -> dict[str, Any]:
    registry: dict[str, Any] = {"proposal_count": PERMUTATION_PROPOSALS, "rules": {}}
    for matrix_id in _selected_rules():
        states = np.asarray([row["final_B"] for row in donors if row["matrix_id"] == matrix_id])
        permutation, mean_h, values = choose_rule_permutation(
            states,
            seed=derive_seed(MASTER_SEED, "rule_permutation", matrix_id),
            proposals=PERMUTATION_PROPOSALS,
        )
        registry["rules"][str(matrix_id)] = {
            "permutation": permutation,
            "mean_self_h": mean_h,
            "maximum_self_h": float(values.max()),
            "minimum_self_h": float(values.min()),
            "all_six_distinct_at_h085": bool(np.all(values <= DEPARTURE_THRESHOLD)),
        }
    return registry


def _rare_registry() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    table = pd.read_csv(PARENT_STABLE_FORMS, dtype={"candidate": str})
    table["matrix_id"] = table["matrix_id"].astype(int)
    selected = table[(table["selected_distinct"] == True) & table["matrix_id"].isin(RARE_RULES)].copy()  # noqa: E712
    rows: list[dict[str, Any]] = []
    adequacy: dict[str, Any] = {}
    for candidate in ("02", "03"):
        for matrix_id in RARE_RULES:
            cell = selected[(selected["candidate"] == candidate) & (selected["matrix_id"] == matrix_id)].sort_values("cluster_id")
            if len(cell) != 2:
                raise ValueError(f"rare panel c{candidate} m{matrix_id:03d} does not have exactly two selected forms")
            forms = [np.asarray(json.loads(value), dtype=np.uint8) for value in cell["medoid"]]
            clusters = [int(value) for value in cell["cluster_id"]]
            for form_index in (0, 1):
                own = forms[form_index]
                other = forms[1 - form_index]
                rows.append(
                    {
                        "candidate": candidate,
                        "matrix_id": matrix_id,
                        "form_index": form_index,
                        "cluster_id": clusters[form_index],
                        "start_index": 0,
                        "start_kind": "intact",
                        "dose": 0,
                        "own_h": 1.0,
                        "other_h": cosine(other, own),
                        "start": own,
                        "own_target": own,
                        "other_target": other,
                    }
                )
                dose, starts, proposals = generate_mass_preserving_perturbations(
                    own,
                    other,
                    seed=derive_seed(MASTER_SEED, "rare_perturbation", candidate, matrix_id, form_index),
                    required=RARE_PERTURBATIONS,
                    proposals_per_dose=RARE_PROPOSALS,
                    dose_ladder=RARE_DOSE_LADDER,
                )
                key = f"c{candidate}_m{matrix_id:03d}_f{form_index}"
                adequacy[key] = {
                    "dose": dose,
                    "starts_found": len(starts),
                    "proposals_examined": proposals,
                    "passed": len(starts) == RARE_PERTURBATIONS,
                }
                for index, start in enumerate(starts, start=1):
                    rows.append(
                        {
                            "candidate": candidate,
                            "matrix_id": matrix_id,
                            "form_index": form_index,
                            "cluster_id": clusters[form_index],
                            "start_index": index,
                            "start_kind": "perturbed",
                            "dose": dose,
                            "own_h": cosine(own, start),
                            "other_h": cosine(other, start),
                            "start": start,
                            "own_target": own,
                            "other_target": other,
                        }
                    )
    return rows, adequacy


def _bray_calibration() -> dict[str, Any]:
    cosine_values: list[float] = []
    bray_values: list[float] = []
    for candidate in ("02", "03"):
        for matrix_id in _selected_rules():
            path = PARENT_LINEAGE_ROOT / f"c{candidate}_m{matrix_id:03d}_fixed.npz"
            with np.load(path, allow_pickle=False) as bundle:
                lineages = min(8, bundle["parents"].shape[0])
                for lineage in range(lineages):
                    observed = int(bundle["observed"][lineage])
                    parents = bundle["parents"][lineage, :observed].astype(np.float64)
                    daughters = bundle["daughters"][lineage, :observed].astype(np.float64)
                    cosine_values.extend(bundle["boundary_h"][lineage, :observed].tolist())
                    denominator = parents.sum(axis=1) + daughters.sum(axis=1)
                    bray = np.ones(observed, dtype=np.float64)
                    np.divide(
                        np.abs(parents - daughters).sum(axis=1),
                        denominator,
                        out=bray,
                        where=denominator > 0,
                    )
                    bray_values.extend((1.0 - bray).tolist())
    source = np.asarray(cosine_values)
    alternative = np.asarray(bray_values)
    mapping: dict[str, float] = {}
    for threshold in (0.85, 0.90, 0.95):
        quantile = float(np.mean(source <= threshold))
        mapping[str(threshold)] = float(np.quantile(alternative, quantile))
    return {
        "method": "empirical_percentile_match_from_first_eight_sealed_random_start_lineages_per_candidate_rule",
        "pairs": len(source),
        "bray_cutoff_by_cosine": mapping,
    }


def prepare() -> None:
    if PROTOCOL_PATH.is_file():
        verify_protocol()
        print(f"Existing sealed protocol verified: {_read_json(PROTOCOL_PATH)['protocol_id']}")
        return
    _parent_integrity()
    collision = _seed_collision_audit()
    bank = _strict_bank()
    donors = _donor_rows(bank)
    permutations = _permutation_registry(donors)
    rare_rows, rare_adequacy = _rare_registry()
    scientific_manifest = _manifest(_scientific_source_paths(), classification="scientific_input")
    hypothesis_manifest = _manifest(HYPOTHESIS_PATHS, classification="hypothesis_only_non_evidentiary")
    _write_json(SOURCE_MANIFEST_PATH, scientific_manifest)
    _write_json(HYPOTHESIS_MANIFEST_PATH, hypothesis_manifest)
    donor_serialized = [{key: _csv_value(value) for key, value in row.items()} for row in donors]
    rare_serialized = [{key: _csv_value(value) for key, value in row.items()} for row in rare_rows]
    _write_csv(DONOR_PATH, donor_serialized)
    _write_csv(RARE_REGISTRY_PATH, rare_serialized)
    _write_json(PERMUTATION_PATH, permutations)
    seed_registry = {
        "master_seed": MASTER_SEED,
        "collision_audit": collision,
        "domains": [
            "rule_permutation",
            "primary.future",
            "rare_perturbation",
            "rare.future",
            "bootstrap",
            "smoke",
        ],
    }
    _write_json(SEED_REGISTRY_PATH, seed_registry)
    registration = {
        "format": FORMAT,
        "classification": "reviewer_prompted_frozen_before_new_futures",
        "candidates_separate": True,
        "primary_gates": {
            "lineage_identity": {"native_f16_lower_gt": 0.40, "difference_point_ge": 0.20, "difference_lower_gt": 0.10},
            "shared_destination": {"stranger_f16_lower_gt": 0.40, "difference_equivalence_margin": 0.10, "confidence": 0.90},
            "transient": {"native_and_stranger_f32_upper_lt": 0.25},
            "isomorphism": {"margin": 0.03, "confidence": 0.90},
            "rule_rehoming": {"difference_point_ge": 0.20, "difference_lower_gt": 0.10},
        },
        "rare_gate": {"own_lower_gt": 0.50, "cross_upper_lt": 0.20, "origin_accuracy_lower_gt": 0.75},
        "thresholds": {"inheritance_strict_gt": 0.90, "departure_inclusive_le": 0.85},
        "bootstrap_repetitions": BOOTSTRAPS,
    }
    registration["registration_id"] = _canonical_digest(registration)
    _write_json(REGISTRATION_PATH, registration)
    protocol = {
        "format": FORMAT,
        "status": "sealed_before_new_futures",
        "scope": "strict-B transplant, rule rehoming, and rare-two-form challenge",
        "parent_integrity": _parent_integrity(),
        "rules": _selected_rules(),
        "rule_count": RULES,
        "candidates": ["02", "03"],
        "donors_per_cell": DONORS_PER_CELL,
        "futures_per_start_arm": FUTURES,
        "horizon": HORIZON,
        "arms": list(ARMS),
        "permutation_proposals": PERMUTATION_PROPOSALS,
        "rare_rules": list(RARE_RULES),
        "rare_perturbations": RARE_PERTURBATIONS,
        "rare_proposals_per_dose": RARE_PROPOSALS,
        "rare_dose_ladder": list(RARE_DOSE_LADDER),
        "rare_adequacy": rare_adequacy,
        "bray_calibration": _bray_calibration(),
        "registration_id": registration["registration_id"],
        "scientific_source_manifest_sha256": sha256_file(SOURCE_MANIFEST_PATH),
        "hypothesis_only_manifest_sha256": sha256_file(HYPOTHESIS_MANIFEST_PATH),
        "runtime": _runtime(),
        "all_writes_below": str(TASK_ROOT.resolve()),
        "manuscript_modified": False,
    }
    protocol["protocol_id"] = _canonical_digest(protocol)
    _write_json(PROTOCOL_PATH, protocol)
    _write_checksums(PROTOCOL_ROOT)
    _update_status(state="prepared", stage="prepare", message="protocol sealed; no new futures generated")
    print(json.dumps({"protocol_id": protocol["protocol_id"], "rare_adequacy": rare_adequacy}, indent=2))


def verify_protocol() -> dict[str, Any]:
    if not PROTOCOL_PATH.is_file():
        raise FileNotFoundError("run prepare before simulation")
    _verify_checksums(PROTOCOL_ROOT)
    protocol = _read_json(PROTOCOL_PATH)
    if protocol.get("format") != FORMAT or protocol.get("protocol_id") != _canonical_digest({key: value for key, value in protocol.items() if key != "protocol_id"}):
        raise ValueError("protocol digest mismatch")
    current = _manifest(_scientific_source_paths(), classification="scientific_input")
    sealed = _read_json(SOURCE_MANIFEST_PATH)
    if current != sealed:
        raise ValueError("scientific source inputs changed after protocol seal")
    if _parent_integrity()["complete"] is not True:
        raise ValueError("parent audit no longer verifies")
    return protocol


def _beta(matrix_id: int) -> np.ndarray:
    rng = np.random.default_rng(derive_seed(CONFIRMATION_MASTER_SEED, "REGCONF.beta", matrix_id))
    return generate_beta(GARD, rng)


def _load_donors() -> list[dict[str, Any]]:
    table = pd.read_csv(DONOR_PATH, dtype={"candidate": str})
    rows: list[dict[str, Any]] = []
    for record in table.to_dict("records"):
        natural_value = record.get("natural_stranger")
        natural = None
        if isinstance(natural_value, str) and natural_value:
            natural = np.asarray(json.loads(natural_value), dtype=np.uint8)
        rows.append(
            {
                **record,
                "candidate": str(record["candidate"]).zfill(2),
                "matrix_id": int(record["matrix_id"]),
                "donor_index": int(record["donor_index"]),
                "final_B": np.asarray(json.loads(record["final_B"]), dtype=np.uint8),
                "episode_medoid": np.asarray(json.loads(record["episode_medoid"]), dtype=np.uint8),
                "natural_stranger": natural,
            }
        )
    return rows


def _load_rare_starts() -> list[dict[str, Any]]:
    table = pd.read_csv(RARE_REGISTRY_PATH, dtype={"candidate": str})
    rows: list[dict[str, Any]] = []
    for record in table.to_dict("records"):
        rows.append(
            {
                **record,
                "candidate": str(record["candidate"]).zfill(2),
                "matrix_id": int(record["matrix_id"]),
                "form_index": int(record["form_index"]),
                "start_index": int(record["start_index"]),
                "start": np.asarray(json.loads(record["start"]), dtype=np.uint8),
                "own_target": np.asarray(json.loads(record["own_target"]), dtype=np.uint8),
                "other_target": np.asarray(json.loads(record["other_target"]), dtype=np.uint8),
            }
        )
    return rows


def _permutation(matrix_id: int) -> NDArray[np.int16]:
    values = _read_json(PERMUTATION_PATH)["rules"][str(matrix_id)]["permutation"]
    return np.asarray(values, dtype=np.int16)


def _primary_path(candidate: str, matrix_id: int) -> Path:
    return PRIMARY_ROOT / f"c{candidate}_m{matrix_id:03d}.npz"


def _rare_path(candidate: str, matrix_id: int) -> Path:
    return RARE_ROOT / f"c{candidate}_m{matrix_id:03d}.npz"


def _primary_tasks() -> list[tuple[str, int]]:
    return [(candidate, matrix_id) for candidate in ("02", "03") for matrix_id in _selected_rules()]


def _rare_tasks() -> list[tuple[str, int]]:
    return [(candidate, matrix_id) for candidate in ("02", "03") for matrix_id in RARE_RULES]


def _simulate_future_arrays(
    start: NDArray,
    beta: NDArray,
    candidate: str,
    *,
    seed: int,
    horizon: int = HORIZON,
) -> tuple[NDArray[np.uint8], FloatArray, int]:
    rng = np.random.default_rng(seed)
    contract = CANDIDATES[candidate]
    daughters = np.zeros((horizon, GARD.n_types), dtype=np.uint8)
    boundary_h = np.full(horizon, np.nan, dtype=np.float64)
    current = np.asarray(start, dtype=np.int64).copy()
    observed = 0
    for generation in range(horizon):
        try:
            record = advance_fission(current, beta, GARD, contract, rng)
        except SimulationError:
            break
        daughters[generation] = record.daughter.astype(np.uint8)
        boundary_h[generation] = record.h
        current = record.daughter
        observed += 1
    return daughters, boundary_h, observed


def _simulate_primary_cell(candidate: str, matrix_id: int) -> dict[str, Any]:
    limiter = threadpool_limits(limits=1)
    try:
        donor_rows = [
            row
            for row in _load_donors()
            if row["candidate"] == candidate and row["matrix_id"] == matrix_id
        ]
        donor_rows.sort(key=lambda row: row["donor_index"])
        if len(donor_rows) != DONORS_PER_CELL:
            raise ValueError("primary cell donor count differs from protocol")
        permutation = _permutation(matrix_id)
        native_beta = _beta(matrix_id)
        permuted_beta = permute_beta(native_beta, permutation)
        shape = (DONORS_PER_CELL, len(ARMS), FUTURES, HORIZON, GARD.n_types)
        daughters = np.zeros(shape, dtype=np.uint8)
        boundary_h = np.full(shape[:-1], np.nan, dtype=np.float64)
        observed = np.zeros(shape[:3], dtype=np.int8)
        active = np.ones((DONORS_PER_CELL, len(ARMS)), dtype=np.int8)
        starts = np.zeros((DONORS_PER_CELL, len(ARMS), GARD.n_types), dtype=np.uint8)
        original_targets = np.zeros((DONORS_PER_CELL, GARD.n_types), dtype=np.uint8)
        permuted_targets = np.zeros_like(original_targets)
        medoid_targets = np.zeros_like(original_targets)
        natural_targets = np.zeros_like(original_targets)
        for donor_index, donor in enumerate(donor_rows):
            original = donor["final_B"]
            permuted = permute_state(original, permutation).astype(np.uint8)
            original_targets[donor_index] = original
            permuted_targets[donor_index] = permuted
            medoid_targets[donor_index] = donor["episode_medoid"]
            natural = donor["natural_stranger"]
            arm_starts = {
                "native": original,
                "state_only": permuted,
                "rule_only": original,
                "joint": permuted,
                "natural_stranger": np.zeros_like(original) if natural is None else natural,
            }
            if natural is None:
                active[donor_index, ARMS.index("natural_stranger")] = 0
            else:
                natural_targets[donor_index] = natural
            for arm_index, arm in enumerate(ARMS):
                starts[donor_index, arm_index] = arm_starts[arm]
                if not active[donor_index, arm_index]:
                    continue
                beta = permuted_beta if arm in {"rule_only", "joint"} else native_beta
                for future in range(FUTURES):
                    seed = derive_seed(
                        MASTER_SEED,
                        "primary.future",
                        candidate,
                        matrix_id,
                        donor_index,
                        future,
                    )
                    future_daughters, future_h, count = _simulate_future_arrays(
                        arm_starts[arm], beta, candidate, seed=seed
                    )
                    daughters[donor_index, arm_index, future] = future_daughters
                    boundary_h[donor_index, arm_index, future] = future_h
                    observed[donor_index, arm_index, future] = count
        return {
            "candidate": candidate,
            "matrix_id": matrix_id,
            "arms": np.asarray(ARMS),
            "permutation": permutation,
            "active": active,
            "starts": starts,
            "original_targets": original_targets,
            "permuted_targets": permuted_targets,
            "medoid_targets": medoid_targets,
            "natural_targets": natural_targets,
            "daughters": daughters,
            "boundary_h": boundary_h,
            "observed": observed,
        }
    finally:
        limiter.restore_original_limits()


def _save_primary_task(task: tuple[str, int]) -> tuple[str, int, str, int]:
    candidate, matrix_id = task
    path = _primary_path(candidate, matrix_id)
    if path.is_file():
        bundle = _load_npz(path)
        return candidate, matrix_id, "existing", int(np.sum(bundle["active"]) * FUTURES)
    payload = _simulate_primary_cell(candidate, matrix_id)
    _atomic_npz(
        path,
        format=np.asarray(FORMAT),
        protocol_id=np.asarray(_read_json(PROTOCOL_PATH)["protocol_id"]),
        **payload,
    )
    return candidate, matrix_id, "generated", int(np.sum(payload["active"]) * FUTURES)


def _simulate_rare_cell(candidate: str, matrix_id: int) -> dict[str, Any]:
    limiter = threadpool_limits(limits=1)
    try:
        starts = [
            row
            for row in _load_rare_starts()
            if row["candidate"] == candidate and row["matrix_id"] == matrix_id
        ]
        starts.sort(key=lambda row: (row["form_index"], row["start_index"]))
        beta = _beta(matrix_id)
        count = len(starts)
        daughters = np.zeros((count, FUTURES, HORIZON, GARD.n_types), dtype=np.uint8)
        boundary_h = np.full((count, FUTURES, HORIZON), np.nan, dtype=np.float64)
        observed = np.zeros((count, FUTURES), dtype=np.int8)
        start_values = np.asarray([row["start"] for row in starts], dtype=np.uint8)
        own_targets = np.asarray([row["own_target"] for row in starts], dtype=np.uint8)
        other_targets = np.asarray([row["other_target"] for row in starts], dtype=np.uint8)
        form_indices = np.asarray([row["form_index"] for row in starts], dtype=np.int8)
        start_indices = np.asarray([row["start_index"] for row in starts], dtype=np.int8)
        doses = np.asarray([row["dose"] for row in starts], dtype=np.int8)
        for start_offset, row in enumerate(starts):
            for future in range(FUTURES):
                seed = derive_seed(
                    MASTER_SEED,
                    "rare.future",
                    candidate,
                    matrix_id,
                    row["form_index"],
                    row["start_index"],
                    future,
                )
                future_daughters, future_h, completed = _simulate_future_arrays(
                    row["start"], beta, candidate, seed=seed
                )
                daughters[start_offset, future] = future_daughters
                boundary_h[start_offset, future] = future_h
                observed[start_offset, future] = completed
        return {
            "candidate": candidate,
            "matrix_id": matrix_id,
            "start_values": start_values,
            "own_targets": own_targets,
            "other_targets": other_targets,
            "form_indices": form_indices,
            "start_indices": start_indices,
            "doses": doses,
            "daughters": daughters,
            "boundary_h": boundary_h,
            "observed": observed,
        }
    finally:
        limiter.restore_original_limits()


def _save_rare_task(task: tuple[str, int]) -> tuple[str, int, str, int]:
    candidate, matrix_id = task
    path = _rare_path(candidate, matrix_id)
    if path.is_file():
        bundle = _load_npz(path)
        return candidate, matrix_id, "existing", int(bundle["daughters"].shape[0] * FUTURES)
    payload = _simulate_rare_cell(candidate, matrix_id)
    _atomic_npz(
        path,
        format=np.asarray(FORMAT),
        protocol_id=np.asarray(_read_json(PROTOCOL_PATH)["protocol_id"]),
        **payload,
    )
    return candidate, matrix_id, "generated", int(payload["daughters"].shape[0] * FUTURES)


def _run_tasks(
    function: Any,
    tasks: Sequence[tuple[str, int]],
    *,
    workers: int,
    stage: str,
) -> None:
    started = time.time()
    total = len(tasks)
    _update_status(state="running", stage=stage, completed=0, total=total, started_at=started)
    iterator: Iterable[tuple[str, int, str, int]]
    if workers <= 1:
        iterator = map(function, tasks)
        for index, result in enumerate(iterator, start=1):
            message = f"c{result[0]} m{result[1]:03d} {result[2]} futures={result[3]}"
            print(f"[{stage}] {index}/{total} {message}", flush=True)
            _update_status(state="running", stage=stage, completed=index, total=total, started_at=started, message=message)
        return
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for index, result in enumerate(executor.map(function, tasks, chunksize=1), start=1):
            message = f"c{result[0]} m{result[1]:03d} {result[2]} futures={result[3]}"
            print(f"[{stage}] {index}/{total} {message}", flush=True)
            _update_status(state="running", stage=stage, completed=index, total=total, started_at=started, message=message)


def simulate_primary(workers: int) -> None:
    verify_protocol()
    _run_tasks(_save_primary_task, _primary_tasks(), workers=workers, stage="simulate-primary")


def simulate_rare(workers: int) -> None:
    verify_protocol()
    _run_tasks(_save_rare_task, _rare_tasks(), workers=workers, stage="simulate-rare")


def smoke() -> None:
    verify_protocol()
    donors = _load_donors()
    for candidate, matrix_id in (("02", _selected_rules()[0]), ("03", _selected_rules()[1])):
        donor = next(row for row in donors if row["candidate"] == candidate and row["matrix_id"] == matrix_id)
        permutation = _permutation(matrix_id)
        beta = _beta(matrix_id)
        for arm, start, arm_beta in (
            ("native", donor["final_B"], beta),
            ("state_only", permute_state(donor["final_B"], permutation), beta),
            ("rule_only", donor["final_B"], permute_beta(beta, permutation)),
            ("joint", permute_state(donor["final_B"], permutation), permute_beta(beta, permutation)),
        ):
            daughters, h, observed = _simulate_future_arrays(
                start,
                arm_beta,
                candidate,
                seed=derive_seed(MASTER_SEED, "smoke", candidate, matrix_id, arm),
                horizon=3,
            )
            if daughters.shape != (3, GARD.n_types) or h.shape != (3,) or not 0 <= observed <= 3:
                raise AssertionError("smoke future has invalid shape")
    print("Two-candidate/two-rule in-memory smoke passed; production seeds were not consumed.")


def _write_dataframe(path: Path, table: pd.DataFrame) -> None:
    _assert_local(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    table.to_csv(temporary, index=False, compression="gzip" if path.suffix == ".gz" else None)
    temporary.replace(path)


def _prefixed_score(prefix: str, score: Any) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in score.to_dict().items()}


def _primary_observation_rows() -> list[dict[str, Any]]:
    protocol = verify_protocol()
    bray_map = {
        float(key): float(value)
        for key, value in protocol["bray_calibration"]["bray_cutoff_by_cosine"].items()
    }
    rows: list[dict[str, Any]] = []
    for candidate, matrix_id in _primary_tasks():
        path = _primary_path(candidate, matrix_id)
        if not path.is_file():
            raise FileNotFoundError(path)
        bundle = _load_npz(path)
        arms = [str(value) for value in bundle["arms"].tolist()]
        permutation = bundle["permutation"]
        for donor_index in range(DONORS_PER_CELL):
            original_target = bundle["original_targets"][donor_index]
            permuted_target = bundle["permuted_targets"][donor_index]
            original_medoid = bundle["medoid_targets"][donor_index]
            permuted_medoid = permute_state(original_medoid, permutation)
            for arm_index, arm in enumerate(arms):
                if not bool(bundle["active"][donor_index, arm_index]):
                    continue
                if arm in {"native", "state_only", "natural_stranger"}:
                    designated_target = original_target
                    designated_name = "original"
                    designated_medoid = original_medoid
                else:
                    designated_target = permuted_target
                    designated_name = "permuted"
                    designated_medoid = permuted_medoid
                if arm == "state_only":
                    launch_target = permuted_target
                elif arm == "rule_only":
                    launch_target = original_target
                elif arm == "natural_stranger":
                    launch_target = bundle["natural_targets"][donor_index]
                else:
                    launch_target = designated_target
                for future in range(FUTURES):
                    daughters = bundle["daughters"][donor_index, arm_index, future]
                    boundary_h = bundle["boundary_h"][donor_index, arm_index, future]
                    observed = int(bundle["observed"][donor_index, arm_index, future])
                    original_score = score_future(daughters, boundary_h, original_target, observed=observed)
                    permuted_score = score_future(daughters, boundary_h, permuted_target, observed=observed)
                    launch_score = score_future(daughters, boundary_h, launch_target, observed=observed)
                    designated_score = original_score if designated_name == "original" else permuted_score
                    medoid_score = score_future(daughters, boundary_h, designated_medoid, observed=observed)
                    sensitivity: dict[str, Any] = {}
                    for source_threshold in (0.85, 0.90, 0.95):
                        sensitivity[f"designated_capture_f32_cosine_{source_threshold:.2f}"] = bool(
                            first_target_capture_metric(
                                daughters,
                                designated_target,
                                observed=observed,
                                horizon=HORIZON,
                                threshold=source_threshold,
                                metric="cosine",
                            )
                            != -1
                        )
                        sensitivity[f"designated_capture_f32_bray_{source_threshold:.2f}"] = bool(
                            first_target_capture_metric(
                                daughters,
                                designated_target,
                                observed=observed,
                                horizon=HORIZON,
                                threshold=bray_map[source_threshold],
                                metric="bray_curtis",
                            )
                            != -1
                        )
                    row = {
                        "candidate": candidate,
                        "matrix_id": matrix_id,
                        "donor_index": donor_index,
                        "arm": arm,
                        "future": future,
                        "designated_target": designated_name,
                        **_prefixed_score("original", original_score),
                        "permuted_capture_f16": permuted_score.capture_f16,
                        "permuted_capture_f32": permuted_score.capture_f32,
                        "permuted_first_capture": permuted_score.first_capture,
                        "designated_capture_f16": designated_score.capture_f16,
                        "designated_capture_f32": designated_score.capture_f32,
                        "designated_arrival_f8": designated_score.arrival_f8,
                        "designated_occupancy": designated_score.occupancy,
                        "designated_maximum_residence": designated_score.maximum_residence,
                        "designated_reentered": designated_score.reentered,
                        "launch_capture_f16": launch_score.capture_f16,
                        "launch_capture_f32": launch_score.capture_f32,
                        "medoid_capture_f16": medoid_score.capture_f16,
                        "medoid_capture_f32": medoid_score.capture_f32,
                        "first_break_by_f8": bool(original_score.first_break != -1 and original_score.first_break <= 8),
                        **sensitivity,
                    }
                    rows.append(row)
    return rows


def _rule_metrics(primary: pd.DataFrame) -> pd.DataFrame:
    boolean_columns = [
        "original_capture_f16",
        "original_capture_f32",
        "original_arrival_f8",
        "original_reentered",
        "original_coherent8",
        "designated_capture_f16",
        "designated_capture_f32",
        "designated_arrival_f8",
        "designated_reentered",
        "launch_capture_f16",
        "launch_capture_f32",
        "medoid_capture_f16",
        "medoid_capture_f32",
        "first_break_by_f8",
    ]
    boolean_columns.extend(
        column for column in primary.columns if column.startswith("designated_capture_f32_cosine_") or column.startswith("designated_capture_f32_bray_")
    )
    numeric_columns = boolean_columns + [
        "original_occupancy",
        "original_maximum_residence",
        "designated_occupancy",
        "designated_maximum_residence",
        "original_completed",
    ]
    return (
        primary.groupby(["candidate", "matrix_id", "arm"], as_index=False)[numeric_columns]
        .mean()
        .sort_values(["candidate", "matrix_id", "arm"])
    )


def _ci(values: NDArray, *, label: str, confidence: float = 0.95) -> list[float]:
    return list(
        bootstrap_mean_ci(
            values,
            seed=derive_seed(MASTER_SEED, "bootstrap", label),
            repetitions=BOOTSTRAPS,
            confidence=confidence,
        )
    )


def _arm_values(rule_metrics: pd.DataFrame, candidate: str, arm: str, column: str) -> NDArray[np.float64]:
    subset = rule_metrics[(rule_metrics["candidate"] == candidate) & (rule_metrics["arm"] == arm)].sort_values("matrix_id")
    if len(subset) != RULES:
        raise ValueError(f"{candidate}/{arm}/{column} lacks {RULES} rule rows")
    return subset[column].to_numpy(dtype=np.float64)


def _candidate_primary_summary(rule_metrics: pd.DataFrame, candidate: str) -> dict[str, Any]:
    native16 = _arm_values(rule_metrics, candidate, "native", "original_capture_f16")
    native32 = _arm_values(rule_metrics, candidate, "native", "original_capture_f32")
    stranger16 = _arm_values(rule_metrics, candidate, "state_only", "original_capture_f16")
    stranger32 = _arm_values(rule_metrics, candidate, "state_only", "original_capture_f32")
    joint16 = _arm_values(rule_metrics, candidate, "joint", "designated_capture_f16")
    joint32 = _arm_values(rule_metrics, candidate, "joint", "designated_capture_f32")
    native_break = _arm_values(rule_metrics, candidate, "native", "first_break_by_f8")
    joint_break = _arm_values(rule_metrics, candidate, "joint", "first_break_by_f8")
    state_designated = _arm_values(rule_metrics, candidate, "state_only", "designated_capture_f32")
    state_launch = _arm_values(rule_metrics, candidate, "state_only", "launch_capture_f32")
    rule_designated = _arm_values(rule_metrics, candidate, "rule_only", "designated_capture_f32")
    rule_launch = _arm_values(rule_metrics, candidate, "rule_only", "launch_capture_f32")

    native16_ci = _ci(native16, label=f"{candidate}.native16")
    native32_ci = _ci(native32, label=f"{candidate}.native32")
    stranger16_ci = _ci(stranger16, label=f"{candidate}.stranger16")
    stranger32_ci = _ci(stranger32, label=f"{candidate}.stranger32")
    identity_diff = _ci(native16 - stranger16, label=f"{candidate}.identity_diff")
    equivalence_diff = _ci(native16 - stranger16, label=f"{candidate}.shared_equivalence", confidence=0.90)
    iso16 = _ci(native16 - joint16, label=f"{candidate}.iso16", confidence=0.90)
    iso32 = _ci(native32 - joint32, label=f"{candidate}.iso32", confidence=0.90)
    iso_break = _ci(native_break - joint_break, label=f"{candidate}.iso_break", confidence=0.90)
    state_rehome = _ci(state_designated - state_launch, label=f"{candidate}.state_rehome")
    rule_rehome = _ci(rule_designated - rule_launch, label=f"{candidate}.rule_rehome")

    isomorphism = all(item[1] > -0.03 and item[2] < 0.03 for item in (iso16, iso32, iso_break))
    lineage = native16_ci[1] > 0.40 and identity_diff[0] >= 0.20 and identity_diff[1] > 0.10
    shared = stranger16_ci[1] > 0.40 and equivalence_diff[1] > -0.10 and equivalence_diff[2] < 0.10
    transient = native32_ci[2] < 0.25 and stranger32_ci[2] < 0.25
    rehoming = (
        isomorphism
        and state_rehome[0] >= 0.20
        and state_rehome[1] > 0.10
        and rule_rehome[0] >= 0.20
        and rule_rehome[1] > 0.10
    )
    return {
        "native_capture_f16_ci": native16_ci,
        "native_capture_f32_ci": native32_ci,
        "state_only_capture_f16_ci": stranger16_ci,
        "state_only_capture_f32_ci": stranger32_ci,
        "native_minus_state_f16_ci": identity_diff,
        "native_minus_state_f16_equivalence_90ci": equivalence_diff,
        "isomorphism_native_minus_joint_f16_90ci": iso16,
        "isomorphism_native_minus_joint_f32_90ci": iso32,
        "isomorphism_native_minus_joint_break_f8_90ci": iso_break,
        "state_only_rule_target_minus_launch_target_ci": state_rehome,
        "rule_only_rule_target_minus_launch_target_ci": rule_rehome,
        "gates": {
            "lineage_identity": lineage,
            "shared_rule_destination": shared,
            "transient": transient,
            "isomorphism": isomorphism,
            "rule_conditioned_rehoming": rehoming,
        },
    }


def _hazard_tables(primary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for (candidate, arm), group in primary.groupby(["candidate", "arm"]):
        first_break = group["original_first_break"].to_numpy(dtype=int)
        hazards: list[float] = []
        for generation in range(1, HORIZON + 1):
            at_risk = int(np.sum((first_break == -1) | (first_break >= generation)))
            events = int(np.sum(first_break == generation))
            hazard = events / at_risk if at_risk else float("nan")
            hazards.append(hazard)
            rows.append({"candidate": candidate, "arm": arm, "generation": generation, "at_risk": at_risk, "events": events, "hazard": hazard})
        early = float(np.nanmean(hazards[:4]))
        late = float(np.nanmean(hazards[8:]))
        summary_rows.append({"candidate": candidate, "arm": arm, "early_hazard_g1_g4": early, "late_hazard_g9_g32": late, "early_late_ratio": early / late if late > 0 else float("inf")})
    return pd.DataFrame(rows), pd.DataFrame(summary_rows)


def _rare_observation_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate, matrix_id in _rare_tasks():
        path = _rare_path(candidate, matrix_id)
        if not path.is_file():
            raise FileNotFoundError(path)
        bundle = _load_npz(path)
        cell_targets: dict[int, np.ndarray] = {}
        for offset, form_index in enumerate(bundle["form_indices"]):
            cell_targets.setdefault(int(form_index), bundle["own_targets"][offset])
        targets = [cell_targets[0], cell_targets[1]]
        for start_offset in range(bundle["daughters"].shape[0]):
            form_index = int(bundle["form_indices"][start_offset])
            start_index = int(bundle["start_indices"][start_offset])
            start_kind = "intact" if start_index == 0 else "perturbed"
            for future in range(FUTURES):
                daughters = bundle["daughters"][start_offset, future]
                boundary_h = bundle["boundary_h"][start_offset, future]
                observed = int(bundle["observed"][start_offset, future])
                own_score = score_future(daughters, boundary_h, bundle["own_targets"][start_offset], observed=observed)
                other_score = score_future(daughters, boundary_h, bundle["other_targets"][start_offset], observed=observed)
                captured_class = first_capture_class(daughters, targets, observed=observed)
                rows.append(
                    {
                        "candidate": candidate,
                        "matrix_id": matrix_id,
                        "form_index": form_index,
                        "start_index": start_index,
                        "start_kind": start_kind,
                        "dose": int(bundle["doses"][start_offset]),
                        "future": future,
                        "observed": observed,
                        "completed": observed == HORIZON,
                        "own_capture_f16": own_score.capture_f16,
                        "own_capture_f32": own_score.capture_f32,
                        "cross_capture_f16": other_score.capture_f16,
                        "cross_capture_f32": other_score.capture_f32,
                        "captured_class": captured_class,
                        "origin_correct": captured_class == form_index,
                    }
                )
    return rows


def _rare_metrics(rare: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (candidate, matrix_id, form_index), group in rare.groupby(["candidate", "matrix_id", "form_index"]):
        perturbed = group[group["start_kind"] == "perturbed"]
        start_metrics = perturbed.groupby("start_index")[["own_capture_f32", "cross_capture_f32", "origin_correct"]].mean()
        adequate = len(start_metrics) == RARE_PERTURBATIONS
        own_ci = _ci(start_metrics["own_capture_f32"].to_numpy(), label=f"rare.{candidate}.{matrix_id}.{form_index}.own") if adequate else [float("nan")] * 3
        cross_ci = _ci(start_metrics["cross_capture_f32"].to_numpy(), label=f"rare.{candidate}.{matrix_id}.{form_index}.cross") if adequate else [float("nan")] * 3
        accuracy_ci = _ci(start_metrics["origin_correct"].to_numpy(), label=f"rare.{candidate}.{matrix_id}.{form_index}.accuracy") if adequate else [float("nan")] * 3
        intact = group[group["start_kind"] == "intact"]
        passed = adequate and own_ci[1] > 0.50 and cross_ci[2] < 0.20 and accuracy_ci[1] > 0.75
        rows.append(
            {
                "candidate": candidate,
                "matrix_id": int(matrix_id),
                "form_index": int(form_index),
                "adequate": adequate,
                "intact_own_capture_f32": float(intact["own_capture_f32"].mean()),
                "perturbed_own_mean": own_ci[0],
                "perturbed_own_lower": own_ci[1],
                "perturbed_own_upper": own_ci[2],
                "perturbed_cross_mean": cross_ci[0],
                "perturbed_cross_lower": cross_ci[1],
                "perturbed_cross_upper": cross_ci[2],
                "origin_accuracy_mean": accuracy_ci[0],
                "origin_accuracy_lower": accuracy_ci[1],
                "origin_accuracy_upper": accuracy_ci[2],
                "form_gate": passed,
            }
        )
    metrics = pd.DataFrame(rows).sort_values(["candidate", "matrix_id", "form_index"])
    rules: dict[str, Any] = {}
    for matrix_id in RARE_RULES:
        cells = metrics[metrics["matrix_id"] == matrix_id]
        candidate_pass = {
            candidate: bool(len(cells[cells["candidate"] == candidate]) == 2 and cells[cells["candidate"] == candidate]["form_gate"].all())
            for candidate in ("02", "03")
        }
        rules[str(matrix_id)] = {
            "candidate_pass": candidate_pass,
            "exceptional_two_basin_rule": all(candidate_pass.values()),
        }
    return metrics, {"rules": rules, "any_exceptional_rule": any(value["exceptional_two_basin_rule"] for value in rules.values())}


def analyze() -> None:
    verify_protocol()
    missing_primary = [str(_primary_path(*task)) for task in _primary_tasks() if not _primary_path(*task).is_file()]
    missing_rare = [str(_rare_path(*task)) for task in _rare_tasks() if not _rare_path(*task).is_file()]
    if missing_primary or missing_rare:
        raise FileNotFoundError(f"missing checkpoints: primary={len(missing_primary)}, rare={len(missing_rare)}")
    started = time.time()
    _update_status(state="running", stage="analyze", started_at=started, message="scoring primary futures")
    primary = pd.DataFrame(_primary_observation_rows())
    rule_metrics = _rule_metrics(primary)
    hazard, hazard_summary = _hazard_tables(primary)
    _update_status(state="running", stage="analyze", started_at=started, message="scoring rare-form futures")
    rare = pd.DataFrame(_rare_observation_rows())
    rare_metrics, rare_summary = _rare_metrics(rare)
    candidate_summary = {candidate: _candidate_primary_summary(rule_metrics, candidate) for candidate in ("02", "03")}
    gates = {
        gate: all(candidate_summary[candidate]["gates"][gate] for candidate in ("02", "03"))
        for gate in ("lineage_identity", "shared_rule_destination", "transient", "isomorphism", "rule_conditioned_rehoming")
    }
    classifications = [name for name in ("lineage_identity", "shared_rule_destination", "transient") if gates[name]]
    overall = classifications[0] if len(classifications) == 1 else "mixed_or_underdetermined"
    summary = {
        "format": FORMAT,
        "candidate": candidate_summary,
        "all_candidate_gates": gates,
        "primary_classification": overall,
        "rare_panel": rare_summary,
        "natural_stranger_active_fraction": float(
            np.sum(primary["arm"] == "natural_stranger")
            / (RULES * 2 * DONORS_PER_CELL * FUTURES)
        ),
        "scope": "conditional_on_50_previously_strict_capable_rules_and_three_previously_observed_B_donors_per_cell",
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    _write_dataframe(OUTPUT_ROOT / "primary_future_scores.csv.gz", primary)
    _write_dataframe(OUTPUT_ROOT / "primary_rule_metrics.csv", rule_metrics)
    _write_dataframe(OUTPUT_ROOT / "break_hazard.csv", hazard)
    _write_dataframe(OUTPUT_ROOT / "break_hazard_summary.csv", hazard_summary)
    _write_dataframe(OUTPUT_ROOT / "rare_future_scores.csv.gz", rare)
    _write_dataframe(OUTPUT_ROOT / "rare_form_metrics.csv", rare_metrics)
    _write_json(OUTPUT_ROOT / "primary_summary.json", summary)
    _write_json(
        OUTPUT_ROOT / "analysis_manifest.json",
        {
            "format": FORMAT,
            "rows": {
                "primary_future_scores": len(primary),
                "primary_rule_metrics": len(rule_metrics),
                "break_hazard": len(hazard),
                "rare_future_scores": len(rare),
                "rare_form_metrics": len(rare_metrics),
            },
        },
    )
    _update_status(state="analyzed", stage="analyze", started_at=started, completed=1, total=1, message=f"classification={overall}")


def _fmt_ci(values: Sequence[float]) -> str:
    return f"{values[0]:.3f} [{values[1]:.3f}, {values[2]:.3f}]"


def _make_figures() -> list[str]:
    figure_root = OUTPUT_ROOT / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    rule = pd.read_csv(OUTPUT_ROOT / "primary_rule_metrics.csv", dtype={"candidate": str})
    primary = pd.read_csv(OUTPUT_ROOT / "primary_future_scores.csv.gz", dtype={"candidate": str})
    hazard = pd.read_csv(OUTPUT_ROOT / "break_hazard.csv", dtype={"candidate": str})
    rare = pd.read_csv(OUTPUT_ROOT / "rare_form_metrics.csv", dtype={"candidate": str})
    created: list[str] = []

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, constrained_layout=True)
    arm_order = ["native", "state_only", "rule_only", "joint", "natural_stranger"]
    labels = ["native", "state-only", "rule-only", "joint", "natural B"]
    for axis, candidate in zip(axes, ("02", "03")):
        subset = rule[rule["candidate"] == candidate]
        means = [subset[subset["arm"] == arm]["designated_capture_f32"].mean() for arm in arm_order]
        axis.bar(np.arange(len(arm_order)), means, color=["#39568C", "#55C667", "#B8DE29", "#1F968B", "#FDE725"])
        axis.set(title=f"Candidate {candidate}", xticks=np.arange(len(labels)), xticklabels=labels, ylim=(0, 1), ylabel="F32 designated-target capture")
        axis.tick_params(axis="x", rotation=30)
    path = figure_root / "figure_1_transplant_capture.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    created.append(str(path.relative_to(OUTPUT_ROOT)))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, constrained_layout=True)
    for axis, candidate in zip(axes, ("02", "03")):
        for arm, color in (("native", "#39568C"), ("state_only", "#55C667"), ("joint", "#1F968B")):
            values = primary[(primary["candidate"] == candidate) & (primary["arm"] == arm)]["original_first_arrival"].to_numpy()
            observed = np.sort(values[values > 0])
            if observed.size:
                axis.step(observed, np.arange(1, observed.size + 1) / len(values), where="post", label=arm, color=color)
        axis.set(title=f"Candidate {candidate}", xlabel="first arrival generation", ylabel="cumulative arrival probability", xlim=(1, HORIZON), ylim=(0, 1))
        axis.legend(frameon=False)
    path = figure_root / "figure_2_arrival_curves.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    created.append(str(path.relative_to(OUTPUT_ROOT)))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, constrained_layout=True)
    for axis, candidate in zip(axes, ("02", "03")):
        for arm, color in (("native", "#39568C"), ("state_only", "#55C667"), ("rule_only", "#B8DE29"), ("joint", "#1F968B")):
            subset = hazard[(hazard["candidate"] == candidate) & (hazard["arm"] == arm)]
            axis.plot(subset["generation"], subset["hazard"], label=arm, color=color)
        axis.set(title=f"Candidate {candidate}", xlabel="generation", ylabel="first-break hazard", xlim=(1, HORIZON))
        axis.legend(frameon=False, fontsize=8)
    path = figure_root / "figure_3_break_hazards.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    created.append(str(path.relative_to(OUTPUT_ROOT)))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, constrained_layout=True)
    x = np.arange(len(RARE_RULES))
    width = 0.35
    for axis, candidate in zip(axes, ("02", "03")):
        cell = rare[rare["candidate"] == candidate]
        for form_index, offset, color in ((0, -width / 2, "#39568C"), (1, width / 2, "#55C667")):
            values = [float(cell[(cell["matrix_id"] == rule_id) & (cell["form_index"] == form_index)]["perturbed_own_mean"].iloc[0]) for rule_id in RARE_RULES]
            axis.bar(x + offset, values, width, label=f"form {form_index}", color=color)
        axis.axhline(0.5, color="black", linestyle="--", linewidth=1)
        axis.set(title=f"Candidate {candidate}", xticks=x, xticklabels=[str(value) for value in RARE_RULES], xlabel="matrix", ylabel="perturbed own-form capture", ylim=(0, 1))
        axis.legend(frameon=False)
    path = figure_root / "figure_4_rare_form_challenge.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    created.append(str(path.relative_to(OUTPUT_ROOT)))
    return created


def report() -> None:
    verify_protocol()
    summary_path = OUTPUT_ROOT / "primary_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError("run analyze before report")
    started = time.time()
    _update_status(state="running", stage="report", started_at=started)
    summary = _read_json(summary_path)
    figures = _make_figures()
    gates = summary["all_candidate_gates"]
    lines = [
        "# Strict-B transplant, rehoming, and rare-form stress test",
        "",
        "## Outcome",
        "",
        f"Primary classification: **{summary['primary_classification'].replace('_', ' ')}**.",
        "",
        f"- Strong lineage identity: **{'PASS' if gates['lineage_identity'] else 'FAIL'}**.",
        f"- Shared rule destination: **{'PASS' if gates['shared_rule_destination'] else 'FAIL'}**.",
        f"- Strong transient: **{'PASS' if gates['transient'] else 'FAIL'}**.",
        f"- Exact-isomorphism audit: **{'PASS' if gates['isomorphism'] else 'FAIL'}**.",
        f"- Rule-conditioned rehoming: **{'PASS' if gates['rule_conditioned_rehoming'] else 'FAIL'}**.",
        "",
        "## Candidate-separated primary estimates",
        "",
    ]
    for candidate in ("02", "03"):
        item = summary["candidate"][candidate]
        lines.extend(
            [
                f"### Candidate {candidate}",
                "",
                f"- Native B capture by F16: {_fmt_ci(item['native_capture_f16_ci'])}.",
                f"- Spectrum-matched stranger capture by F16: {_fmt_ci(item['state_only_capture_f16_ci'])}.",
                f"- Native minus stranger F16 capture: {_fmt_ci(item['native_minus_state_f16_ci'])}.",
                f"- Native B capture by F32: {_fmt_ci(item['native_capture_f32_ci'])}.",
                f"- Spectrum-matched stranger capture by F32: {_fmt_ci(item['state_only_capture_f32_ci'])}.",
                f"- State-only rule-target minus launch-target capture: {_fmt_ci(item['state_only_rule_target_minus_launch_target_ci'])}.",
                f"- Rule-only rule-target minus launch-target capture: {_fmt_ci(item['rule_only_rule_target_minus_launch_target_ci'])}.",
                "",
            ]
        )
    exceptional = [matrix_id for matrix_id, value in summary["rare_panel"]["rules"].items() if value["exceptional_two_basin_rule"]]
    lines.extend(
        [
            "## Rare two-form challenge",
            "",
            f"Exceptional rules passing both forms in both candidates: **{', '.join(exceptional) if exceptional else 'none'}**.",
            "Matrices 11, 54, and 63 were post-hoc selected case studies; no result here estimates population prevalence.",
            "",
            "## Interpretation boundary",
            "",
            "This campaign is conditional on 50 previously strict-capable catalytic rules and three previously observed strict-B donors per rule and candidate. NewIdeas supplied hypotheses only and no outcome from that folder entered the evidence chain. The finite F32 assay does not establish or exclude an infinite-time mathematical attractor. Sensitivities and natural-stranger results cannot rescue a failed primary gate.",
            "",
        ]
    )
    (OUTPUT_ROOT / "SCIENTIFIC_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    appendix = """# Appendix: transplant and residence definitions

The exact abundance-spectrum-matched stranger is a seeded permutation of the
strict B molecule labels. It preserves mass, occupied counts, and the complete
abundance multiset while breaking alignment to the native catalytic matrix.
The rule-only and joint arms use the exact corresponding row-and-column
permutation of beta. Joint outcomes are scored in the permuted target frame.

Arrival is the first daughter strictly above the target threshold. Target
capture is stronger: eight consecutive descendants must each exceed the
target threshold and all 28 descendant pairs must also exceed it. Departure
uses inclusive H<=0.85; re-entry requires a later complete target capture.
Extinction and incomplete trajectories remain negative.

Rule-level estimates equally weight the 50 matrices. Candidates are never
pooled to rescue a gate. Break-time curves are conditional hazards rather than
raw histograms, avoiding censoring-induced early-count inflation.
"""
    (OUTPUT_ROOT / "APPENDIX_TRANSPLANT.md").write_text(appendix, encoding="utf-8")
    patch_text = f"""# Proposed manuscript and reviewer-response language

This file is proposed language only and does not modify the manuscript.

## Reviewer response

We followed the lineage-identity assays with a prospectively frozen common-
garden transplant test using three strict-B donors from each of 50 previously
capable catalytic rules in both simulator contracts. An exact molecule-label
permutation supplied a mass- and abundance-spectrum-matched stranger, and a
four-arm state/rule permutation factorial separated state incumbency from
rule-conditioned rehoming. The all-candidate classifications were: lineage
identity, **{'pass' if gates['lineage_identity'] else 'fail'}**; shared rule
destination, **{'pass' if gates['shared_rule_destination'] else 'fail'}**;
strong transient, **{'pass' if gates['transient'] else 'fail'}**; and rule-
conditioned rehoming, **{'pass' if gates['rule_conditioned_rehoming'] else 'fail'}**.
These tests are conditional on previously detected strict forms and do not
alter the operational F12 definition.

## Limitation

The transplant assay uses finite F32 futures from previously observed strict-B
states under reconstructed simulator contracts. It distinguishes registered
finite-horizon alternatives but does not prove an infinite-time attractor or
generalize beyond the selected strict-capable cohort.
"""
    (OUTPUT_ROOT / "PROPOSED_MANUSCRIPT_AND_REVIEWER_PATCH.md").write_text(patch_text, encoding="utf-8")
    manifest = _read_json(OUTPUT_ROOT / "analysis_manifest.json")
    manifest["figures"] = figures
    manifest["reports"] = ["SCIENTIFIC_REPORT.md", "APPENDIX_TRANSPLANT.md", "PROPOSED_MANUSCRIPT_AND_REVIEWER_PATCH.md"]
    _write_json(OUTPUT_ROOT / "analysis_manifest.json", manifest)
    _write_checksums(OUTPUT_ROOT)
    _update_status(state="reported", stage="report", started_at=started, completed=1, total=1, message="reports, figures, and output checksums complete")


def _validate_checkpoint(path: Path, *, candidate: str, matrix_id: int) -> dict[str, Any]:
    bundle = _load_npz(path)
    if str(bundle["format"].item()) != FORMAT:
        raise ValueError(f"format mismatch: {path}")
    if str(bundle["protocol_id"].item()) != _read_json(PROTOCOL_PATH)["protocol_id"]:
        raise ValueError(f"protocol mismatch: {path}")
    if str(bundle["candidate"].item()).zfill(2) != candidate or int(bundle["matrix_id"].item()) != matrix_id:
        raise ValueError(f"cell identity mismatch: {path}")
    return bundle


def _compare_payload(stored: dict[str, np.ndarray], replay: dict[str, Any]) -> tuple[bool, float]:
    exact = True
    maximum_h_error = 0.0
    ignored = {"format", "protocol_id"}
    for key, expected in replay.items():
        if key in ignored:
            continue
        if key not in stored:
            return False, float("inf")
        left = np.asarray(stored[key])
        right = np.asarray(expected)
        if key == "boundary_h":
            finite = np.isfinite(left) & np.isfinite(right)
            if np.any(finite):
                maximum_h_error = max(maximum_h_error, float(np.max(np.abs(left[finite] - right[finite]))))
            exact = exact and bool(np.array_equal(np.isnan(left), np.isnan(right))) and maximum_h_error == 0.0
        else:
            exact = exact and bool(np.array_equal(left, right))
    return exact, maximum_h_error


def verify(*, full_replay: bool) -> None:
    protocol = verify_protocol()
    if not (OUTPUT_ROOT / "SHA256SUMS").is_file():
        raise FileNotFoundError("run report before verification")
    output_checks = _verify_checksums(OUTPUT_ROOT)
    primary_tasks = _primary_tasks()
    rare_tasks = _rare_tasks()
    started = time.time()
    total = len(primary_tasks) + len(rare_tasks)
    completed = 0
    replayed_futures = 0
    replay_exact = True
    maximum_h_error = 0.0
    checkpoint_rows: list[dict[str, Any]] = []
    _update_status(state="running", stage="verify-full-replay" if full_replay else "verify", completed=0, total=total, started_at=started)
    for candidate, matrix_id in primary_tasks:
        path = _primary_path(candidate, matrix_id)
        stored = _validate_checkpoint(path, candidate=candidate, matrix_id=matrix_id)
        cell_exact = True
        cell_h = 0.0
        if full_replay:
            replay = _simulate_primary_cell(candidate, matrix_id)
            cell_exact, cell_h = _compare_payload(stored, replay)
            replayed_futures += int(np.sum(replay["active"]) * FUTURES)
        replay_exact = replay_exact and cell_exact
        maximum_h_error = max(maximum_h_error, cell_h)
        checkpoint_rows.append({"kind": "primary", "candidate": candidate, "matrix_id": matrix_id, "path": str(path.resolve()), "sha256": sha256_file(path), "replay_exact": cell_exact, "maximum_h_error": cell_h})
        completed += 1
        _update_status(state="running", stage="verify-full-replay" if full_replay else "verify", completed=completed, total=total, started_at=started, message=f"primary c{candidate} m{matrix_id:03d}")
    for candidate, matrix_id in rare_tasks:
        path = _rare_path(candidate, matrix_id)
        stored = _validate_checkpoint(path, candidate=candidate, matrix_id=matrix_id)
        cell_exact = True
        cell_h = 0.0
        if full_replay:
            replay = _simulate_rare_cell(candidate, matrix_id)
            cell_exact, cell_h = _compare_payload(stored, replay)
            replayed_futures += int(replay["daughters"].shape[0] * FUTURES)
        replay_exact = replay_exact and cell_exact
        maximum_h_error = max(maximum_h_error, cell_h)
        checkpoint_rows.append({"kind": "rare", "candidate": candidate, "matrix_id": matrix_id, "path": str(path.resolve()), "sha256": sha256_file(path), "replay_exact": cell_exact, "maximum_h_error": cell_h})
        completed += 1
        _update_status(state="running", stage="verify-full-replay" if full_replay else "verify", completed=completed, total=total, started_at=started, message=f"rare c{candidate} m{matrix_id:03d}")
    complete = bool(full_replay and replay_exact and maximum_h_error == 0.0 and all(output_checks.values()))
    audit = {
        "format": FORMAT,
        "protocol_id": protocol["protocol_id"],
        "full_replay_requested": full_replay,
        "checkpoint_files": len(checkpoint_rows),
        "replayed_futures": replayed_futures,
        "discrete_replay_exact": replay_exact,
        "maximum_h_error": maximum_h_error,
        "output_checksums_verified": all(output_checks.values()),
        "complete": complete,
        "checkpoints": checkpoint_rows,
    }
    VERIFICATION_ROOT.mkdir(parents=True, exist_ok=True)
    _write_json(VERIFICATION_ROOT / "verification_audit.json", audit)
    _write_checksums(VERIFICATION_ROOT)
    if full_replay and not complete:
        raise ValueError("full verification did not complete exactly")
    _update_status(state="complete" if complete else "verified_without_full_replay", stage="complete" if complete else "verify", started_at=started, completed=total, total=total, message=f"replayed_futures={replayed_futures}")
    print(json.dumps(audit, indent=2))


def status() -> None:
    if not STATUS_PATH.is_file():
        print(json.dumps({"state": "not_started", "stage": "none"}, indent=2))
        return
    print(STATUS_PATH.read_text(encoding="utf-8"), end="")


def run_all(workers: int) -> None:
    try:
        if not PROTOCOL_PATH.is_file():
            prepare()
        verify_protocol()
        smoke()
        simulate_primary(workers)
        simulate_rare(workers)
        analyze()
        report()
        verify(full_replay=True)
    except Exception as error:
        _update_status(state="failed", stage="failed", message="pipeline stopped", error=f"{type(error).__name__}: {error}")
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("prepare")
    subcommands.add_parser("smoke")
    primary = subcommands.add_parser("simulate-primary")
    primary.add_argument("--workers", type=int, default=1)
    rare = subcommands.add_parser("simulate-rare")
    rare.add_argument("--workers", type=int, default=1)
    subcommands.add_parser("analyze")
    subcommands.add_parser("report")
    verify_parser = subcommands.add_parser("verify")
    verify_parser.add_argument("--full-replay", action="store_true")
    all_parser = subcommands.add_parser("all")
    all_parser.add_argument("--workers", type=int, default=16)
    subcommands.add_parser("status")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    if arguments.command == "prepare":
        prepare()
    elif arguments.command == "smoke":
        smoke()
    elif arguments.command == "simulate-primary":
        simulate_primary(arguments.workers)
    elif arguments.command == "simulate-rare":
        simulate_rare(arguments.workers)
    elif arguments.command == "analyze":
        analyze()
    elif arguments.command == "report":
        report()
    elif arguments.command == "verify":
        verify(full_replay=arguments.full_replay)
    elif arguments.command == "all":
        run_all(arguments.workers)
    elif arguments.command == "status":
        status()


if __name__ == "__main__":
    main()
