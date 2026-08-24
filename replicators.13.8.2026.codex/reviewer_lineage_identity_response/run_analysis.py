"""Reproducible runner for reviewer lineage-identity tests 2--4."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any, Iterable

TASK_ROOT = Path(__file__).resolve().parent
CODEX_ROOT = TASK_ROOT.parent
if __package__ in (None, ""):
    sys.path.insert(0, str(CODEX_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(TASK_ROOT / "artifacts" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from plastic_heredity.config import CANDIDATES, GardConfig
from plastic_heredity.mechanistic import sha256_file
from plastic_heredity.regime_confirmation import CONFIRMATION_MASTER_SEED
from plastic_heredity.seeds import derive_seed
from plastic_heredity.simulator import (
    SimulationError,
    advance_fission,
    generate_beta,
    generate_initial_composition,
)
from reviewer_lineage_identity_response.lineage_identity_core import (
    BANK_SIZE,
    BURN_IN,
    DISTINCTNESS_THRESHOLD,
    INHERITANCE_THRESHOLD,
    MAX_LINEAGES,
    PRIMARY_LINEAGES,
    PRIMARY_RESIDENCE,
    PRIMARY_SEPARATION,
    PRIMARY_START_SUPPORT,
    WINDOW,
    Episode,
    bootstrap_mean_ci,
    census_from_clusters,
    cosine,
    empirical_range_overlap,
    find_earliest_episode,
    fork_scores,
    nearest_identity_accuracy,
    probability_superiority,
    residence_clusters,
    select_capable_rules,
    sensitivity_grid,
    sibling_stranger_values,
    split_centroids,
    stranger_literal_distinct_rate,
    strict_literal_fork_rate,
)


ARTIFACT_ROOT = TASK_ROOT / "artifacts"
PROTOCOL_ROOT = ARTIFACT_ROOT / "protocol"
WORK_ROOT = ARTIFACT_ROOT / "work"
LINEAGE_ROOT = WORK_ROOT / "lineages"
FORK_ROOT = WORK_ROOT / "forks"
OUTPUT_ROOT = ARTIFACT_ROOT / "output"
VERIFICATION_ROOT = ARTIFACT_ROOT / "verification"
PROTOCOL_PATH = PROTOCOL_ROOT / "protocol.json"
REGISTRATION_PATH = PROTOCOL_ROOT / "registration.json"
SELECTION_PATH = PROTOCOL_ROOT / "matrix_selection.csv"
SEED_REGISTRY_PATH = PROTOCOL_ROOT / "seed_registry.json"
SOURCE_MANIFEST_PATH = PROTOCOL_ROOT / "source_manifest.json"

CONFIRMATION_STATES = CODEX_ROOT / "results" / "regime_confirmation" / "confirmation_states.csv"
CONFIRMATION_MANIFEST = CODEX_ROOT / "results" / "regime_confirmation" / "manifest.json"
CONFIRMATION_OCCURRENCE = CODEX_ROOT / "results" / "regime_confirmation" / "occurrence_metrics.json"

FORMAT = "reviewer-lineage-identity-v1"
MASTER_SEED = "2026082002403"
RULES = 50
GENERATIONS = 256
FORK_GENERATIONS = 8
BOOTSTRAP_REPETITIONS = 10_000
STRONG_AUC_LOWER = 0.75
STRONG_GAP_LOWER = 0.05
GARD = GardConfig(generations=GENERATIONS)

SOURCE_PATHS = {
    "runner": Path(__file__),
    "core": TASK_ROOT / "lineage_identity_core.py",
    "readme": TASK_ROOT / "README.md",
    "review_plan": TASK_ROOT / "REVIEW_AND_PLAN.md",
    "reviewer_comment": TASK_ROOT / "REVIEWER_COMMENT.md",
    "simulator": CODEX_ROOT / "plastic_heredity" / "simulator.py",
    "config": CODEX_ROOT / "plastic_heredity" / "config.py",
    "seeds": CODEX_ROOT / "plastic_heredity" / "seeds.py",
    "regime_scorer": CODEX_ROOT / "plastic_heredity" / "regime_confirmation.py",
    "confirmation_states": CONFIRMATION_STATES,
    "confirmation_manifest": CONFIRMATION_MANIFEST,
    "confirmation_occurrence": CONFIRMATION_OCCURRENCE,
    "requirements": CODEX_ROOT / "requirements-lock.txt",
}


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(
        _json_ready(value), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    _assert_local(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_ready(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_local(path: Path) -> None:
    resolved = path.resolve()
    if resolved != TASK_ROOT and TASK_ROOT not in resolved.parents:
        raise ValueError(f"refusing write outside reviewer folder: {resolved}")


def _sha_manifest(paths: dict[str, Path]) -> dict[str, dict[str, str]]:
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing source inputs: {missing}")
    return {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for name, path in sorted(paths.items())
    }


def _write_checksums(directory: Path) -> None:
    _assert_local(directory)
    entries: list[str] = []
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        if path.name == "SHA256SUMS":
            continue
        entries.append(f"{sha256_file(path)}  {path.relative_to(directory)}")
    (directory / "SHA256SUMS").write_text("\n".join(entries) + "\n", encoding="utf-8")


def _verify_checksums(directory: Path) -> dict[str, bool]:
    checksum_path = directory / "SHA256SUMS"
    if not checksum_path.is_file():
        raise FileNotFoundError(checksum_path)
    results: dict[str, bool] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        path = directory / relative
        results[relative] = path.is_file() and sha256_file(path) == digest
    if not results or not all(results.values()):
        failed = [name for name, passed in results.items() if not passed]
        raise ValueError(f"checksum verification failed: {failed}")
    return results


def _runtime() -> dict[str, str]:
    packages = ("numpy", "pandas", "scipy", "matplotlib", "threadpoolctl")
    return {
        "python": platform.python_version(),
        **{name: importlib.metadata.version(name) for name in packages},
    }


def _event_counts() -> dict[str, dict[int, int]]:
    table = pd.read_csv(CONFIRMATION_STATES, dtype={"candidate": str})
    required = {"candidate", "matrix_id", "q_primary_all8_all"}
    if not required.issubset(table.columns):
        raise ValueError("REGCONF state table lacks strict-event columns")
    output = {"02": {}, "03": {}}
    for (candidate, matrix_id), group in table.groupby(["candidate", "matrix_id"]):
        events = int(np.rint((group["q_primary_all8_all"] * 128).sum()))
        output[str(candidate)][int(matrix_id)] = events
    return output


def _seed_collision_audit() -> dict[str, Any]:
    collisions: list[str] = []
    for suffix in ("*.py", "*.json", "*.md"):
        for path in CODEX_ROOT.rglob(suffix):
            if TASK_ROOT == path or TASK_ROOT in path.parents or not path.is_file():
                continue
            relative = path.relative_to(CODEX_ROOT)
            if any(part in {".venv", ".git", "__pycache__"} for part in relative.parts):
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            if MASTER_SEED in content:
                collisions.append(str(path.relative_to(CODEX_ROOT)))
    if collisions:
        raise ValueError(f"campaign master seed already appears outside package: {collisions}")
    return {
        "master_seed": MASTER_SEED,
        "scanned_root": str(CODEX_ROOT.resolve()),
        "exact_literal_collisions": collisions,
        "passed": True,
    }


def _protocol(selected: list[int], capable_count: int) -> dict[str, Any]:
    protocol: dict[str, Any] = {
        "format": FORMAT,
        "status": "sealed_before_new_lineage_generation",
        "scope": "reviewer-prompted lineage identity tests 2--4",
        "working_boundary": {
            "all_writes_below": str(TASK_ROOT.resolve()),
            "existing_results_read_only": True,
            "manuscript_edit": False,
            "draft_only_data_excluded": True,
        },
        "cohort": {
            "source": "sealed REGCONF confirmation state table",
            "capable": "at least one primary_all8 event in both candidates",
            "shared_capable_count": capable_count,
            "selection": "first 50 by SHA256(master_seed|matrix_id)",
            "selected_matrix_ids": selected,
            "candidates": ["02", "03"],
            "same_beta_across_candidates": True,
            "conditional_claim_only": True,
        },
        "simulation": {
            "master_seed": MASTER_SEED,
            "primary_random_starts_per_cell": PRIMARY_LINEAGES,
            "maximum_starts_for_b_bank": MAX_LINEAGES,
            "generations": GENERATIONS,
            "burn_in": BURN_IN,
            "analysis_windows": {"strict_F32": 7, "F12_control": 18},
            "window_lengths": {"strict": WINDOW, "F12_control": 12},
            "strict_b_bank": BANK_SIZE,
            "extension": (
                "launch indices 128--255 only when the fixed 128 contain fewer "
                "than 20 strict B episodes; extension excluded from census"
            ),
            "initial_composition_seed_shared_across_candidates": True,
            "dynamics_seed_candidate_specific": True,
        },
        "endpoints": {
            "strict_B": {
                "primary": True,
                "selection": "final daughter of earliest strict episode in earliest qualifying fixed F32 window",
                "break": "first unrounded H<=0.90 in the window",
                "inheritance": "eight consecutive unrounded H>0.90",
                "coherence": "all 28 daughter pairs unrounded H>0.90",
                "distinctness": "every daughter unrounded H<=0.85 from break parent",
            },
            "F12_control": {
                "primary": False,
                "non_rescuing": True,
                "selection": "final daughter of earliest post-break run3 in fixed non-overlapping F12 windows",
            },
        },
        "test_2": {
            "within": "H(first-four centroid, last-four centroid) in one episode",
            "cross": "H(first-four centroid i, last-four centroid j), i!=j, same rule and candidate",
            "bank": "first 20 qualifying lineages in seed order",
            "primary": ["rule-level AUC", "rule-level median within-minus-cross"],
            "strong_gate_each_candidate": {
                "whole_rule_bootstrap_AUC_lower95_above": STRONG_AUC_LOWER,
                "whole_rule_bootstrap_gap_lower95_above": STRONG_GAP_LOWER,
            },
            "literal": ["empirical range overlap", "cross fraction inside within range"],
        },
        "test_3": {
            "forks": 2,
            "generations": FORK_GENERATIONS,
            "noise": "independent; no common random streams",
            "extinction": "incomplete sibling fork scores 0; incomplete stranger comparison scores 1, so extinction cannot create a pass",
            "sibling_score": "minimum corresponding-generation H",
            "stranger_eligibility": "different lineage and initial B H<=0.85",
            "stranger_score": "maximum corresponding-generation H",
            "strong_gate_each_candidate": {
                "whole_rule_bootstrap_AUC_lower95_above": STRONG_AUC_LOWER,
                "whole_rule_bootstrap_gap_lower95_above": STRONG_GAP_LOWER,
            },
            "literal": "sibling minimum H>0.90 and stranger maximum H<=0.90",
            "no_eligible_stranger": "shared-destination evidence, literal failure",
        },
        "test_4": {
            "census_uses_fixed_128_only": True,
            "primary_residence": PRIMARY_RESIDENCE,
            "primary_start_support": PRIMARY_START_SUPPORT,
            "primary_durable_support": 4,
            "durability": "16 continuous resident fissions or re-entry after at least 8 outside fissions",
            "coherent_cluster": "deterministic complete-link, all representative H>0.90",
            "distinct_forms": "greedy support-prioritized medoids all H<=0.85",
            "rule_survival": "at least two stable distinct forms",
            "sensitivities": {
                "residence": [4, 8, 16],
                "start_support": [4, 8, 16],
                "distinctness": [0.80, 0.85, 0.90],
            },
        },
        "inference": {
            "unit": "catalytic rule",
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "candidates_separate": True,
            "candidate_rescue": False,
            "underfilled_cells_retained": True,
            "literal_and_corrected_readouts_both_reported": True,
        },
        "claim_boundary": {
            "tests": "lineage-specific identity and finite-horizon multistability",
            "not_tests": [
                "validity of the operational F12 event",
                "infinite-time mathematical attractors",
                "population selection",
                "biological heredity",
            ],
        },
    }
    protocol["protocol_id"] = _canonical_digest(protocol)
    return protocol


def prepare() -> None:
    if PROTOCOL_ROOT.exists():
        raise FileExistsError(f"refusing to overwrite sealed protocol: {PROTOCOL_ROOT}")
    seed_audit = _seed_collision_audit()
    event_counts = _event_counts()
    capable = sorted(
        matrix_id
        for matrix_id in set(event_counts["02"]) & set(event_counts["03"])
        if event_counts["02"][matrix_id] > 0 and event_counts["03"][matrix_id] > 0
    )
    selected = select_capable_rules(
        event_counts, count=RULES, selection_seed=MASTER_SEED
    )
    PROTOCOL_ROOT.mkdir(parents=True)
    protocol = _protocol(selected, len(capable))
    _write_json(PROTOCOL_PATH, protocol)
    rows = [
        {
            "selection_order": index,
            "matrix_id": matrix_id,
            "regconf_events_02": event_counts["02"][matrix_id],
            "regconf_events_03": event_counts["03"][matrix_id],
            "selection_hash": hashlib.sha256(
                f"{MASTER_SEED}|{matrix_id}".encode("utf-8")
            ).hexdigest(),
        }
        for index, matrix_id in enumerate(selected)
    ]
    pd.DataFrame(rows).to_csv(SELECTION_PATH, index=False)
    seeds = {
        "master": MASTER_SEED,
        "domains": [
            "initial",
            "lineage",
            "strict_fork_a",
            "strict_fork_b",
            "f12_fork_a",
            "f12_fork_b",
            "bootstrap_test2",
            "bootstrap_test3",
            "bootstrap_test4",
        ],
        "existing_regconf_master_read_only": CONFIRMATION_MASTER_SEED,
        "collision_audit": seed_audit,
    }
    _write_json(SEED_REGISTRY_PATH, seeds)
    sources = _sha_manifest(SOURCE_PATHS)
    _write_json(SOURCE_MANIFEST_PATH, sources)
    registration = {
        "format": FORMAT,
        "status": "sealed_before_new_lineage_generation",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256_file(PROTOCOL_PATH),
        "selection_sha256": sha256_file(SELECTION_PATH),
        "seed_registry_sha256": sha256_file(SEED_REGISTRY_PATH),
        "source_manifest_sha256": sha256_file(SOURCE_MANIFEST_PATH),
        "runtime": _runtime(),
    }
    registration["registration_id"] = _canonical_digest(registration)
    _write_json(REGISTRATION_PATH, registration)
    _write_checksums(PROTOCOL_ROOT)
    print(f"Sealed reviewer protocol at {PROTOCOL_ROOT}")
    print(f"Shared capable rules: {len(capable)}; selected: {len(selected)}")


def verify_protocol() -> dict[str, Any]:
    checks = _verify_checksums(PROTOCOL_ROOT)
    protocol = _read_json(PROTOCOL_PATH)
    protocol_id = protocol.pop("protocol_id")
    if _canonical_digest(protocol) != protocol_id:
        raise ValueError("protocol identifier mismatch")
    protocol["protocol_id"] = protocol_id
    registration = _read_json(REGISTRATION_PATH)
    registration_id = registration.pop("registration_id")
    if _canonical_digest(registration) != registration_id:
        raise ValueError("registration identifier mismatch")
    registration["registration_id"] = registration_id
    expected_sources = _read_json(SOURCE_MANIFEST_PATH)
    current_sources = _sha_manifest(SOURCE_PATHS)
    if expected_sources != current_sources:
        changed = [
            name for name in expected_sources if expected_sources[name] != current_sources.get(name)
        ]
        raise ValueError(f"registered source inputs changed: {changed}")
    return {"protocol": protocol, "registration": registration, "checks": checks}


def _selected_rules() -> list[int]:
    return pd.read_csv(SELECTION_PATH)["matrix_id"].astype(int).tolist()


def _checkpoint_path(candidate: str, matrix_id: int, segment: str) -> Path:
    return LINEAGE_ROOT / f"c{candidate}_m{matrix_id:03d}_{segment}.npz"


def _fork_path(candidate: str, matrix_id: int) -> Path:
    return FORK_ROOT / f"c{candidate}_m{matrix_id:03d}.npz"


def _atomic_npz(path: Path, **values: Any) -> None:
    _assert_local(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **values)
    temporary.replace(path)


def _beta(matrix_id: int) -> np.ndarray:
    rng = np.random.default_rng(
        derive_seed(CONFIRMATION_MASTER_SEED, "REGCONF.beta", matrix_id)
    )
    return generate_beta(GARD, rng)


def _simulate_range(candidate: str, matrix_id: int, start: int, stop: int) -> dict[str, Any]:
    limiter = threadpool_limits(limits=1)
    try:
        beta = _beta(matrix_id)
        contract = CANDIDATES[candidate]
        count = stop - start
        parents = np.zeros((count, GENERATIONS, GARD.n_types), dtype=np.uint8)
        daughters = np.zeros_like(parents)
        boundary_h = np.full((count, GENERATIONS), np.nan, dtype=np.float64)
        growth_steps = np.zeros((count, GENERATIONS), dtype=np.uint16)
        observed = np.zeros(count, dtype=np.uint16)
        completed = np.zeros(count, dtype=np.int8)
        for local, lineage in enumerate(range(start, stop)):
            initial_rng = np.random.default_rng(
                derive_seed(MASTER_SEED, "initial", matrix_id, lineage)
            )
            dynamics_rng = np.random.default_rng(
                derive_seed(MASTER_SEED, "lineage", candidate, matrix_id, lineage)
            )
            current = generate_initial_composition(GARD, initial_rng)
            for generation in range(GENERATIONS):
                try:
                    record = advance_fission(current, beta, GARD, contract, dynamics_rng)
                except SimulationError:
                    break
                parents[local, generation] = record.parent.astype(np.uint8)
                daughters[local, generation] = record.daughter.astype(np.uint8)
                boundary_h[local, generation] = record.h
                growth_steps[local, generation] = record.growth_steps
                current = record.daughter
                observed[local] += 1
            completed[local] = int(observed[local] == GENERATIONS)
        return {
            "candidate": candidate,
            "matrix_id": matrix_id,
            "lineage_start": start,
            "lineage_stop": stop,
            "parents": parents,
            "daughters": daughters,
            "boundary_h": boundary_h,
            "growth_steps": growth_steps,
            "observed": observed,
            "completed": completed,
        }
    finally:
        limiter.restore_original_limits()


def _save_simulation(candidate: str, matrix_id: int, segment: str, start: int, stop: int) -> None:
    path = _checkpoint_path(candidate, matrix_id, segment)
    if path.exists():
        return
    payload = _simulate_range(candidate, matrix_id, start, stop)
    protocol_id = _read_json(PROTOCOL_PATH)["protocol_id"]
    _atomic_npz(
        path,
        format=np.asarray(FORMAT),
        protocol_id=np.asarray(protocol_id),
        candidate=np.asarray(candidate),
        matrix_id=np.asarray(matrix_id, dtype=np.int16),
        lineage_start=np.asarray(start, dtype=np.int16),
        lineage_stop=np.asarray(stop, dtype=np.int16),
        parents=payload["parents"],
        daughters=payload["daughters"],
        boundary_h=payload["boundary_h"],
        growth_steps=payload["growth_steps"],
        observed=payload["observed"],
        completed=payload["completed"],
    )


def _cell_tasks() -> list[tuple[str, int]]:
    return [(candidate, matrix_id) for candidate in ("02", "03") for matrix_id in _selected_rules()]


def _simulate_fixed_task(task: tuple[str, int]) -> tuple[str, int, str]:
    candidate, matrix_id = task
    _save_simulation(candidate, matrix_id, "fixed", 0, PRIMARY_LINEAGES)
    return candidate, matrix_id, "fixed"


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as bundle:
        return {name: bundle[name] for name in bundle.files}


def _episodes_from_bundle(bundle: dict[str, np.ndarray], kind: str) -> list[tuple[int, Episode]]:
    start = int(bundle["lineage_start"])
    output: list[tuple[int, Episode]] = []
    for local in range(bundle["daughters"].shape[0]):
        episode = find_earliest_episode(
            bundle["parents"][local],
            bundle["daughters"][local],
            bundle["boundary_h"][local],
            kind=kind,
            window=12 if kind == "f12" else WINDOW,
        )
        if episode is not None:
            output.append((start + local, episode))
    return output


def _needs_extension(candidate: str, matrix_id: int) -> bool:
    fixed = _load_npz(_checkpoint_path(candidate, matrix_id, "fixed"))
    return len(_episodes_from_bundle(fixed, "strict")) < BANK_SIZE


def _simulate_extension_task(task: tuple[str, int]) -> tuple[str, int, str]:
    candidate, matrix_id = task
    if _needs_extension(candidate, matrix_id):
        _save_simulation(
            candidate, matrix_id, "extension", PRIMARY_LINEAGES, MAX_LINEAGES
        )
        return candidate, matrix_id, "extension"
    return candidate, matrix_id, "not-needed"


def simulate(mode: str, workers: int) -> None:
    verify_protocol()
    tasks = _cell_tasks()
    if mode in {"fixed", "all"}:
        _run_tasks(_simulate_fixed_task, tasks, workers, "fixed")
    if mode in {"extension", "all"}:
        missing = [task for task in tasks if not _checkpoint_path(*task, "fixed").is_file()]
        if missing:
            raise FileNotFoundError("fixed simulation must finish before extension")
        _run_tasks(_simulate_extension_task, tasks, workers, "extension")


def _run_tasks(function: Any, tasks: list[tuple[str, int]], workers: int, label: str) -> None:
    if workers <= 1:
        iterator: Iterable[tuple[str, int, str]] = map(function, tasks)
        for index, result in enumerate(iterator, start=1):
            print(f"[{label}] {index}/{len(tasks)} c{result[0]} m{result[1]:03d} {result[2]}", flush=True)
        return
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for index, result in enumerate(executor.map(function, tasks, chunksize=1), start=1):
            print(f"[{label}] {index}/{len(tasks)} c{result[0]} m{result[1]:03d} {result[2]}", flush=True)


def _collect_episodes(candidate: str, matrix_id: int, kind: str) -> list[tuple[int, Episode]]:
    bundles = [_load_npz(_checkpoint_path(candidate, matrix_id, "fixed"))]
    extension = _checkpoint_path(candidate, matrix_id, "extension")
    if extension.is_file():
        bundles.append(_load_npz(extension))
    episodes: list[tuple[int, Episode]] = []
    for bundle in bundles:
        episodes.extend(_episodes_from_bundle(bundle, kind))
    episodes.sort(key=lambda item: item[0])
    return episodes[:BANK_SIZE]


def _simulate_forks_for_kind(
    candidate: str,
    matrix_id: int,
    kind: str,
    episodes: list[tuple[int, Episode]],
) -> dict[str, np.ndarray]:
    beta = _beta(matrix_id)
    contract = CANDIDATES[candidate]
    count = len(episodes)
    starts = np.zeros((count, GARD.n_types), dtype=np.uint8)
    fork_a = np.zeros((count, FORK_GENERATIONS, GARD.n_types), dtype=np.uint8)
    fork_b = np.zeros_like(fork_a)
    completed = np.zeros((count, 2), dtype=np.int8)
    lineages = np.asarray([lineage for lineage, _ in episodes], dtype=np.int16)
    for index, (lineage, episode) in enumerate(episodes):
        starts[index] = episode.final
        for fork_index, label in enumerate((f"{kind}_fork_a", f"{kind}_fork_b")):
            rng = np.random.default_rng(
                derive_seed(MASTER_SEED, label, candidate, matrix_id, lineage)
            )
            current = episode.final.copy()
            target = fork_a if fork_index == 0 else fork_b
            observed = 0
            for generation in range(FORK_GENERATIONS):
                try:
                    record = advance_fission(current, beta, GARD, contract, rng)
                except SimulationError:
                    break
                target[index, generation] = record.daughter.astype(np.uint8)
                current = record.daughter
                observed += 1
            completed[index, fork_index] = int(observed == FORK_GENERATIONS)
    return {
        f"{kind}_lineages": lineages,
        f"{kind}_starts": starts,
        f"{kind}_fork_a": fork_a,
        f"{kind}_fork_b": fork_b,
        f"{kind}_completed": completed,
    }


def _fork_task(task: tuple[str, int]) -> tuple[str, int, str]:
    candidate, matrix_id = task
    path = _fork_path(candidate, matrix_id)
    if path.exists():
        return candidate, matrix_id, "existing"
    strict = _collect_episodes(candidate, matrix_id, "strict")
    f12 = _collect_episodes(candidate, matrix_id, "f12")
    payload = {
        **_simulate_forks_for_kind(candidate, matrix_id, "strict", strict),
        **_simulate_forks_for_kind(candidate, matrix_id, "f12", f12),
    }
    _atomic_npz(
        path,
        format=np.asarray(FORMAT),
        protocol_id=np.asarray(_read_json(PROTOCOL_PATH)["protocol_id"]),
        candidate=np.asarray(candidate),
        matrix_id=np.asarray(matrix_id, dtype=np.int16),
        **payload,
    )
    return candidate, matrix_id, f"strict={len(strict)},f12={len(f12)}"


def fork(workers: int) -> None:
    verify_protocol()
    tasks = _cell_tasks()
    missing = [task for task in tasks if not _checkpoint_path(task[0], task[1], "fixed").is_file()]
    if missing:
        raise FileNotFoundError(f"fixed lineage checkpoints missing for {len(missing)} cells")
    # Every cell needing an extension must have completed it before fork launch.
    extension_missing = [
        task
        for task in tasks
        if _needs_extension(*task)
        and not _checkpoint_path(task[0], task[1], "extension").is_file()
    ]
    if extension_missing:
        raise FileNotFoundError(
            f"B-bank extensions missing for {len(extension_missing)} cells"
        )
    _run_tasks(_fork_task, tasks, workers, "fork")


def _baseline_cell_rows(
    candidate: str,
    matrix_id: int,
    kind: str,
    episodes_with_ids: list[tuple[int, Episode]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    episodes = [episode for _, episode in episodes_with_ids]
    lineages = [lineage for lineage, _ in episodes_with_ids]
    within, cross = sibling_stranger_values(episodes)
    overlap, overlap_fraction = empirical_range_overlap(within, cross)
    rows: list[dict[str, Any]] = []
    halves = [split_centroids(episode) for episode in episodes]
    for index, value in enumerate(within):
        rows.append(
            {
                "candidate": candidate,
                "matrix_id": matrix_id,
                "kind": kind,
                "relation": "within",
                "source_lineage": lineages[index],
                "target_lineage": lineages[index],
                "h": value,
            }
        )
    for i in range(len(episodes)):
        for j in range(len(episodes)):
            if i == j:
                continue
            rows.append(
                {
                    "candidate": candidate,
                    "matrix_id": matrix_id,
                    "kind": kind,
                    "relation": "cross",
                    "source_lineage": lineages[i],
                    "target_lineage": lineages[j],
                    "h": cosine(halves[i][0], halves[j][1]),
                }
            )
    summary = {
        "candidate": candidate,
        "matrix_id": matrix_id,
        "kind": kind,
        "episodes": len(episodes),
        "bank_complete": len(episodes) == BANK_SIZE,
        "within_n": len(within),
        "cross_n": len(cross),
        "within_median": float(np.median(within)) if within.size else np.nan,
        "cross_median": float(np.median(cross)) if cross.size else np.nan,
        "median_gap": (
            float(np.median(within) - np.median(cross))
            if within.size and cross.size
            else np.nan
        ),
        "auc": probability_superiority(within, cross),
        "identity_accuracy": nearest_identity_accuracy(episodes),
        "literal_range_overlap": overlap,
        "literal_cross_inside_within_range": overlap_fraction,
    }
    return summary, rows


def _fork_cell_rows(
    candidate: str,
    matrix_id: int,
    kind: str,
    bundle: dict[str, np.ndarray],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    starts = bundle[f"{kind}_starts"]
    a = bundle[f"{kind}_fork_a"]
    b = bundle[f"{kind}_fork_b"]
    lineages = bundle[f"{kind}_lineages"].astype(int)
    sibling, stranger, pairs = fork_scores(starts, a, b)
    completed = bundle[f"{kind}_completed"].astype(bool)
    if sibling.size:
        sibling[~np.all(completed, axis=1)] = 0.0
    for index, (i, j) in enumerate(pairs):
        if not completed[i, 0] or not completed[j, 1]:
            stranger[index] = 1.0
    scores: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    for index, value in enumerate(sibling):
        scores.append(
            {
                "candidate": candidate,
                "matrix_id": matrix_id,
                "kind": kind,
                "relation": "sibling",
                "source_lineage": lineages[index],
                "target_lineage": lineages[index],
                "score": value,
                "comparison_complete": bool(np.all(completed[index])),
            }
        )
        for generation in range(FORK_GENERATIONS):
            trajectories.append(
                {
                    "candidate": candidate,
                    "matrix_id": matrix_id,
                    "kind": kind,
                    "relation": "sibling",
                    "source_lineage": lineages[index],
                    "target_lineage": lineages[index],
                    "generation": generation + 1,
                    "h": cosine(a[index, generation], b[index, generation]),
                    "comparison_complete": bool(np.all(completed[index])),
                }
            )
    for value, (i, j) in zip(stranger, pairs):
        scores.append(
            {
                "candidate": candidate,
                "matrix_id": matrix_id,
                "kind": kind,
                "relation": "stranger",
                "source_lineage": lineages[i],
                "target_lineage": lineages[j],
                "score": value,
                "comparison_complete": bool(completed[i, 0] and completed[j, 1]),
            }
        )
        for generation in range(FORK_GENERATIONS):
            trajectories.append(
                {
                    "candidate": candidate,
                    "matrix_id": matrix_id,
                    "kind": kind,
                    "relation": "stranger",
                    "source_lineage": lineages[i],
                    "target_lineage": lineages[j],
                    "generation": generation + 1,
                    "h": cosine(a[i, generation], b[j, generation]),
                    "comparison_complete": bool(completed[i, 0] and completed[j, 1]),
                }
            )
    summary = {
        "candidate": candidate,
        "matrix_id": matrix_id,
        "kind": kind,
        "episodes": len(starts),
        "bank_complete": len(starts) == BANK_SIZE,
        "eligible_stranger_pairs": len(pairs),
        "sibling_median": float(np.median(sibling)) if sibling.size else np.nan,
        "stranger_median": float(np.median(stranger)) if stranger.size else np.nan,
        "median_gap": (
            float(np.median(sibling) - np.median(stranger))
            if sibling.size and stranger.size
            else np.nan
        ),
        "auc": probability_superiority(sibling, stranger),
        "literal_sibling_all8_rate": strict_literal_fork_rate(sibling),
        "literal_stranger_distinct_all8_rate": stranger_literal_distinct_rate(stranger),
        "no_distinguishable_stranger": len(pairs) == 0,
    }
    return summary, scores, trajectories


def _bank_rows(
    candidate: str,
    matrix_id: int,
    kind: str,
    episodes: list[tuple[int, Episode]],
) -> list[dict[str, Any]]:
    return [
        {
            "candidate": candidate,
            "matrix_id": matrix_id,
            "kind": kind,
            "bank_index": index,
            "lineage": lineage,
            "window_index": episode.window_index,
            "break_index": episode.break_index,
            "run_start": episode.run_start,
            "anchor": json.dumps(episode.anchor.astype(int).tolist(), separators=(",", ":")),
            "daughters": json.dumps(episode.daughters.astype(int).tolist(), separators=(",", ":")),
            "final_B": json.dumps(episode.final.astype(int).tolist(), separators=(",", ":")),
        }
        for index, (lineage, episode) in enumerate(episodes)
    ]


def _census_cell(
    candidate: str,
    matrix_id: int,
    daughters: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    fixed = daughters[:PRIMARY_LINEAGES]
    cached: dict[int, tuple[list[Any], list[list[int]]]] = {}
    for residence in (4, 8, 16):
        cached[residence] = residence_clusters(
            fixed, residence_length=residence
        )
    primary = census_from_clusters(
        *cached[PRIMARY_RESIDENCE],
        residence_length=PRIMARY_RESIDENCE,
        start_support=PRIMARY_START_SUPPORT,
        durable_support=4,
        separation=PRIMARY_SEPARATION,
    )
    primary_row = {
        "candidate": candidate,
        "matrix_id": matrix_id,
        "residence_length": PRIMARY_RESIDENCE,
        "start_support": PRIMARY_START_SUPPORT,
        "durable_support": 4,
        "separation": PRIMARY_SEPARATION,
        "residence_episodes": primary.residence_episodes,
        "coherent_clusters": primary.coherent_clusters,
        "stable_forms": len(primary.stable_forms),
        "distinct_stable_forms": len(primary.distinct_forms),
        "literal_rule_survives": len(primary.distinct_forms) >= 2,
    }
    form_rows: list[dict[str, Any]] = []
    selected_ids = {item.cluster_id for item in primary.distinct_forms}
    for form in primary.stable_forms:
        form_rows.append(
            {
                "candidate": candidate,
                "matrix_id": matrix_id,
                "cluster_id": form.cluster_id,
                "selected_distinct": form.cluster_id in selected_ids,
                "start_support": len(form.starts),
                "durable_support": len(form.durable_starts),
                "residence_episodes": len(form.episodes),
                "medoid": json.dumps(form.medoid.astype(int).tolist(), separators=(",", ":")),
            }
        )
    sensitivity_rows: list[dict[str, Any]] = []
    for residence, support, durable, separation in sensitivity_grid():
        result = census_from_clusters(
            *cached[residence],
            residence_length=residence,
            start_support=support,
            durable_support=durable,
            separation=separation,
        )
        sensitivity_rows.append(
            {
                "candidate": candidate,
                "matrix_id": matrix_id,
                "residence_length": residence,
                "start_support": support,
                "durable_support": durable,
                "separation": separation,
                "residence_episodes": result.residence_episodes,
                "coherent_clusters": result.coherent_clusters,
                "stable_forms": len(result.stable_forms),
                "distinct_stable_forms": len(result.distinct_forms),
                "rule_survives": len(result.distinct_forms) >= 2,
            }
        )
    return primary_row, form_rows, sensitivity_rows


def _candidate_summary(
    baseline: pd.DataFrame,
    forks: pd.DataFrame,
    census: pd.DataFrame,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for candidate in ("02", "03"):
        strict_baseline = baseline[
            (baseline["candidate"] == candidate) & (baseline["kind"] == "strict")
        ]
        strict_forks = forks[
            (forks["candidate"] == candidate) & (forks["kind"] == "strict")
        ]
        candidate_census = census[census["candidate"] == candidate]
        test2_auc = bootstrap_mean_ci(
            strict_baseline["auc"].to_numpy(),
            repetitions=BOOTSTRAP_REPETITIONS,
            rng=np.random.default_rng(derive_seed(MASTER_SEED, "bootstrap_test2", candidate, "auc")),
        )
        test2_gap = bootstrap_mean_ci(
            strict_baseline["median_gap"].to_numpy(),
            repetitions=BOOTSTRAP_REPETITIONS,
            rng=np.random.default_rng(derive_seed(MASTER_SEED, "bootstrap_test2", candidate, "gap")),
        )
        test3_auc = bootstrap_mean_ci(
            strict_forks["auc"].to_numpy(),
            repetitions=BOOTSTRAP_REPETITIONS,
            rng=np.random.default_rng(derive_seed(MASTER_SEED, "bootstrap_test3", candidate, "auc")),
        )
        test3_gap = bootstrap_mean_ci(
            strict_forks["median_gap"].to_numpy(),
            repetitions=BOOTSTRAP_REPETITIONS,
            rng=np.random.default_rng(derive_seed(MASTER_SEED, "bootstrap_test3", candidate, "gap")),
        )
        multi = (candidate_census["distinct_stable_forms"] >= 2).astype(float).to_numpy()
        test4_fraction = bootstrap_mean_ci(
            multi,
            repetitions=BOOTSTRAP_REPETITIONS,
            rng=np.random.default_rng(derive_seed(MASTER_SEED, "bootstrap_test4", candidate)),
        )
        test2_adequate = bool(
            len(strict_baseline) == RULES and strict_baseline["bank_complete"].all()
        )
        test3_adequate = bool(
            len(strict_forks) == RULES
            and strict_forks["bank_complete"].all()
            and (~strict_forks["no_distinguishable_stranger"]).all()
        )
        output[candidate] = {
            "test_2": {
                "rules": len(strict_baseline),
                "complete_banks": int(strict_baseline["bank_complete"].sum()),
                "auc_mean_ci": test2_auc,
                "median_gap_mean_ci": test2_gap,
                "adequate": test2_adequate,
                "strong_gate": bool(
                    test2_adequate
                    and test2_auc[1] > STRONG_AUC_LOWER
                    and test2_gap[1] > STRONG_GAP_LOWER
                ),
            },
            "test_3": {
                "rules": len(strict_forks),
                "complete_banks": int(strict_forks["bank_complete"].sum()),
                "rules_without_distinguishable_stranger": int(
                    strict_forks["no_distinguishable_stranger"].sum()
                ),
                "auc_mean_ci": test3_auc,
                "median_gap_mean_ci": test3_gap,
                "adequate": test3_adequate,
                "strong_gate": bool(
                    test3_adequate
                    and test3_auc[1] > STRONG_AUC_LOWER
                    and test3_gap[1] > STRONG_GAP_LOWER
                ),
            },
            "test_4": {
                "rules": len(candidate_census),
                "zero_forms": int((candidate_census["distinct_stable_forms"] == 0).sum()),
                "one_form": int((candidate_census["distinct_stable_forms"] == 1).sum()),
                "two_or_more_forms": int((candidate_census["distinct_stable_forms"] >= 2).sum()),
                "multistable_fraction_ci": test4_fraction,
                "universal_all_50_gate": bool(len(candidate_census) == RULES and np.all(multi == 1)),
            },
        }
    output["all_candidate_gates"] = {
        "test_2": all(output[candidate]["test_2"]["strong_gate"] for candidate in ("02", "03")),
        "test_3": all(output[candidate]["test_3"]["strong_gate"] for candidate in ("02", "03")),
        "test_4_universal": all(
            output[candidate]["test_4"]["universal_all_50_gate"] for candidate in ("02", "03")
        ),
    }
    return output


def analyze() -> None:
    verification = verify_protocol()
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"refusing to overwrite analysis output: {OUTPUT_ROOT}")
    tasks = _cell_tasks()
    missing = [task for task in tasks if not _fork_path(*task).is_file()]
    if missing:
        raise FileNotFoundError(f"fork checkpoints missing for {len(missing)} cells")
    temporary = OUTPUT_ROOT.with_name("output.incomplete")
    if temporary.exists():
        raise FileExistsError(f"remove or inspect incomplete output first: {temporary}")
    temporary.mkdir(parents=True)
    baseline_summaries: list[dict[str, Any]] = []
    baseline_values: list[dict[str, Any]] = []
    b_bank_rows: list[dict[str, Any]] = []
    fork_summaries: list[dict[str, Any]] = []
    fork_scores_rows: list[dict[str, Any]] = []
    fork_trajectories: list[dict[str, Any]] = []
    census_rows: list[dict[str, Any]] = []
    form_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    for index, (candidate, matrix_id) in enumerate(tasks, start=1):
        for kind in ("strict", "f12"):
            episode_items = _collect_episodes(candidate, matrix_id, kind)
            b_bank_rows.extend(_bank_rows(candidate, matrix_id, kind, episode_items))
            summary, values = _baseline_cell_rows(candidate, matrix_id, kind, episode_items)
            baseline_summaries.append(summary)
            baseline_values.extend(values)
        fork_bundle = _load_npz(_fork_path(candidate, matrix_id))
        for kind in ("strict", "f12"):
            summary, scores, trajectories = _fork_cell_rows(
                candidate, matrix_id, kind, fork_bundle
            )
            fork_summaries.append(summary)
            fork_scores_rows.extend(scores)
            fork_trajectories.extend(trajectories)
        fixed = _load_npz(_checkpoint_path(candidate, matrix_id, "fixed"))
        census, forms, sensitivities = _census_cell(
            candidate, matrix_id, fixed["daughters"]
        )
        census_rows.append(census)
        form_rows.extend(forms)
        sensitivity_rows.extend(sensitivities)
        print(f"[analyze] {index}/{len(tasks)} c{candidate} m{matrix_id:03d}", flush=True)
    tables = {
        "baseline_rule_metrics.csv": pd.DataFrame(baseline_summaries),
        "baseline_values.csv": pd.DataFrame(baseline_values),
        "b_bank.csv": pd.DataFrame(b_bank_rows),
        "fork_rule_metrics.csv": pd.DataFrame(fork_summaries),
        "fork_scores.csv": pd.DataFrame(fork_scores_rows),
        "fork_trajectories.csv": pd.DataFrame(fork_trajectories),
        "attractor_census.csv": pd.DataFrame(census_rows),
        "stable_forms.csv": pd.DataFrame(form_rows),
        "attractor_sensitivity.csv": pd.DataFrame(sensitivity_rows),
    }
    for name, table in tables.items():
        table.to_csv(temporary / name, index=False)
    summary = _candidate_summary(
        tables["baseline_rule_metrics.csv"],
        tables["fork_rule_metrics.csv"],
        tables["attractor_census.csv"],
    )
    _write_json(temporary / "primary_summary.json", summary)
    manifest = {
        "format": FORMAT,
        "protocol_id": verification["protocol"]["protocol_id"],
        "registration_id": verification["registration"]["registration_id"],
        "runtime": _runtime(),
        "cells": len(tasks),
        "tables": {name: len(table) for name, table in tables.items()},
        "status": "analysis_complete_reports_pending",
    }
    _write_json(temporary / "analysis_manifest.json", manifest)
    temporary.replace(OUTPUT_ROOT)
    print(f"Analysis written to {OUTPUT_ROOT}")


def _fmt_interval(values: list[float] | tuple[float, float, float]) -> str:
    return f"{values[0]:.3f} [{values[1]:.3f}, {values[2]:.3f}]"


def _ecdf(axis: Any, values: np.ndarray, label: str, color: str) -> None:
    finite = np.sort(np.asarray(values, dtype=np.float64))
    finite = finite[np.isfinite(finite)]
    if finite.size:
        axis.step(finite, np.arange(1, finite.size + 1) / finite.size, where="post", label=label, color=color)


def _make_figures() -> None:
    figure_root = OUTPUT_ROOT / "figures"
    figure_root.mkdir(exist_ok=True)
    baseline = pd.read_csv(OUTPUT_ROOT / "baseline_values.csv", dtype={"candidate": str})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for axis, candidate in zip(axes, ("02", "03")):
        selected = baseline[(baseline["candidate"] == candidate) & (baseline["kind"] == "strict")]
        _ecdf(axis, selected.loc[selected["relation"] == "within", "h"].to_numpy(), "within lineage", "#1f77b4")
        _ecdf(axis, selected.loc[selected["relation"] == "cross", "h"].to_numpy(), "cross lineage", "#d62728")
        axis.axvline(0.90, color="black", linestyle="--", linewidth=1)
        axis.set(title=f"Candidate {candidate}", xlabel="split-centroid cosine H", ylabel="ECDF")
        axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_root / "figure_1_sibling_stranger_ecdf.png", dpi=180)
    plt.close(fig)

    trajectories = pd.read_csv(OUTPUT_ROOT / "fork_trajectories.csv", dtype={"candidate": str})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for axis, candidate in zip(axes, ("02", "03")):
        selected = trajectories[
            (trajectories["candidate"] == candidate)
            & (trajectories["kind"] == "strict")
            & (trajectories["comparison_complete"].astype(str) == "True")
        ]
        for relation, color in (("sibling", "#1f77b4"), ("stranger", "#d62728")):
            relation_values = selected[selected["relation"] == relation]
            if relation_values.empty:
                continue
            means = relation_values.groupby("generation")["h"].mean()
            axis.plot(means.index, means.values, marker="o", label=relation, color=color)
        axis.axhline(0.90, color="black", linestyle="--", linewidth=1)
        axis.set(title=f"Candidate {candidate}", xlabel="fork generation", ylabel="mean corresponding H", xticks=range(1, 9))
        axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_root / "figure_2_post_break_forks.png", dpi=180)
    plt.close(fig)

    census = pd.read_csv(OUTPUT_ROOT / "attractor_census.csv", dtype={"candidate": str})
    categories = ["0", "1", "2+"]
    fig, axis = plt.subplots(figsize=(7, 4))
    x = np.arange(len(categories))
    width = 0.35
    for offset, candidate in zip((-width / 2, width / 2), ("02", "03")):
        values = census.loc[census["candidate"] == candidate, "distinct_stable_forms"]
        counts = [int((values == 0).sum()), int((values == 1).sum()), int((values >= 2).sum())]
        axis.bar(x + offset, counts, width, label=f"candidate {candidate}")
    axis.set(xticks=x, xticklabels=categories, xlabel="distinct stable forms per rule", ylabel="rules")
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_root / "figure_3_attractor_census.png", dpi=180)
    plt.close(fig)

    rule_metrics = pd.read_csv(OUTPUT_ROOT / "baseline_rule_metrics.csv", dtype={"candidate": str})
    fig, axis = plt.subplots(figsize=(7, 4))
    positions: list[int] = []
    values: list[np.ndarray] = []
    labels: list[str] = []
    position = 1
    for candidate in ("02", "03"):
        for kind in ("strict", "f12"):
            values.append(rule_metrics.loc[(rule_metrics["candidate"] == candidate) & (rule_metrics["kind"] == kind), "auc"].dropna().to_numpy())
            positions.append(position)
            labels.append(f"{candidate}\n{kind}")
            position += 1
        position += 1
    axis.boxplot(values, positions=positions, widths=0.65, showfliers=False)
    axis.axhline(0.5, color="black", linestyle="--", linewidth=1)
    axis.set(xticks=positions, xticklabels=labels, ylabel="rule-level within > cross AUC")
    fig.tight_layout()
    fig.savefig(figure_root / "figure_4_strict_f12_control.png", dpi=180)
    plt.close(fig)


def report() -> None:
    verify_protocol()
    if not (OUTPUT_ROOT / "primary_summary.json").is_file():
        raise FileNotFoundError("run analyze before report")
    summary = _read_json(OUTPUT_ROOT / "primary_summary.json")
    _make_figures()
    lines = [
        "# Reviewer lineage-identity tests 2--4",
        "",
        "## Outcome",
        "",
        "This reviewer-prompted campaign evaluates lineage-specific identity conditional on a frozen cohort of 50 previously strict-capable catalytic rules. It does not redefine or invalidate the operational F12 event.",
        "",
    ]
    for test in ("test_2", "test_3", "test_4_universal"):
        passed = summary["all_candidate_gates"][test]
        lines.append(f"- **{test.replace('_', ' ').title()}:** {'PASS' if passed else 'FAIL'} across both candidates.")
    lines.extend(["", "## Candidate-separated primary readouts", ""])
    for candidate in ("02", "03"):
        item = summary[candidate]
        lines.extend(
            [
                f"### Candidate {candidate}",
                "",
                f"- Test 2 complete B banks: {item['test_2']['complete_banks']}/50; AUC {_fmt_interval(item['test_2']['auc_mean_ci'])}; median-gap {_fmt_interval(item['test_2']['median_gap_mean_ci'])}; strong gate **{'PASS' if item['test_2']['strong_gate'] else 'FAIL'}**.",
                f"- Test 3 complete B banks: {item['test_3']['complete_banks']}/50; rules without a distinguishable stranger: {item['test_3']['rules_without_distinguishable_stranger']}; AUC {_fmt_interval(item['test_3']['auc_mean_ci'])}; median-gap {_fmt_interval(item['test_3']['median_gap_mean_ci'])}; strong gate **{'PASS' if item['test_3']['strong_gate'] else 'FAIL'}**.",
                f"- Test 4 rules with 0/1/2+ stable forms: {item['test_4']['zero_forms']}/{item['test_4']['one_form']}/{item['test_4']['two_or_more_forms']}; multistable fraction {_fmt_interval(item['test_4']['multistable_fraction_ci'])}; all-50 gate **{'PASS' if item['test_4']['universal_all_50_gate'] else 'FAIL'}**.",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation boundary",
            "",
            "Tests 2 and 3 concern whether a coherent strict-event form carries lineage-discriminating information. Test 4 concerns finite-horizon, cross-start multistability. A failure does not make an observed F12 break-and-renewal sequence a computational false positive; it limits the stronger stable-identity interpretation.",
            "",
            "The ordinary F12 repeats are descriptive controls and cannot rescue a strict-B gate. Every inference is conditional on the selected capable-rule cohort, and candidates 02 and 03 are never pooled to rescue disagreement.",
        ]
    )
    (OUTPUT_ROOT / "SCIENTIFIC_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    appendix = f"""# Appendix: lineage identity, forks, and attractor census

## Design

Fifty catalytic matrices with at least one archived strict coherent-eight event in both simulator candidates were selected by a frozen hash order. Each matrix was evaluated under candidates 02 and 03 using 128 fresh random-start lineages of 256 fissions. Fissions 1--32 were burn-in. The remaining trajectory supplied seven fixed F32 windows for strict B and 18 fixed F12 windows for the non-rescuing control. A preregistered second block of 128 starts was used only where needed to obtain up to 20 strict episodes for identity tests; it never entered the attractor census.

Strict form B was the final daughter of the earliest coherent, old-anchor-distinct eight-fission episode in the earliest qualifying window. Ordinary F12 run-three episodes were non-rescuing controls.

## Test 2

Within-lineage similarity compared the first-four and last-four daughter centroids of each strict episode. Cross-lineage similarity used the identical axes across every different-lineage pairing under the same rule. Rule-level AUC and median gaps were summarized with 10,000 whole-rule bootstrap replicates. The corrected strong gate required lower bounds above {STRONG_AUC_LOWER:.2f} and {STRONG_GAP_LOWER:.2f}, respectively, in both candidates. Empirical distribution overlap was retained as the reviewer's literal sensitivity readout.

## Test 3

Each selected B was cloned into two independent eight-fission futures. The sibling score was the minimum corresponding-generation H. Eligible strangers came from different lineages under the same rule and began at H<=0.85; their score was the maximum corresponding-generation H. An incomplete sibling fork scored 0 and an incomplete stranger comparison scored 1, so extinction could not manufacture a pass. The same corrected gates were applied. The literal readout required sibling H>0.90 throughout and stranger H<=0.90 throughout.

## Test 4

Rolling coherent residence windows were clustered by deterministic complete linkage. A primary stable form required support from at least eight random starts plus 16-fission persistence or departure-and-reentry in at least four starts. Distinct form medoids required H<=0.85. Residence, support, and separation sensitivities are reported in full.

## Scope

These are finite-horizon operational tests, not proofs of infinite-time attractors. Rule selection used archived capability, while all tested lineage outcomes used fresh seed domains. Results generalize only to this frozen capable-rule cohort.
"""
    (OUTPUT_ROOT / "APPENDIX_LINEAGE_IDENTITY.md").write_text(appendix, encoding="utf-8")

    gate2 = "passed" if summary["all_candidate_gates"]["test_2"] else "did not pass"
    gate3 = "passed" if summary["all_candidate_gates"]["test_3"] else "did not pass"
    gate4 = "passed" if summary["all_candidate_gates"]["test_4_universal"] else "did not pass"
    patch = f"""# Proposed manuscript and reviewer-response patch

This file is proposed language only. It does not modify the manuscript.

## Reviewer response

We thank the reviewer for separating local renewal from lineage-specific identity. We prospectively applied the requested sibling--stranger, post-break fork, and attractor-census tests to 50 previously strict-capable catalytic rules under both simulator contracts. We retained the reviewer's literal thresholds and added preregistered whole-rule uncertainty gates. The sibling--stranger gate {gate2}; the post-break-fork gate {gate3}; and the universal two-form census gate {gate4}. Complete distributions, per-rule classifications, F12 controls, and sensitivity analyses are reported in Appendix X. These tests concern the stronger identity interpretation and do not alter the operational definition of F12 as local parent-to-daughter renewal.

## Proposed Results insertion

An isolated reviewer-prompted campaign tested whether coherent post-break episodes carried lineage-specific identity or represented destinations shared under a catalytic rule. Fifty previously strict-capable matrices were frozen by outcome-independent hash ordering and evaluated under both simulator contracts from fresh random starts. Candidate-separated sibling--stranger, independent-fork, and finite-horizon attractor-census results are reported in Appendix X. The all-candidate corrected gates were: test 2, **{'pass' if summary['all_candidate_gates']['test_2'] else 'fail'}**; test 3, **{'pass' if summary['all_candidate_gates']['test_3'] else 'fail'}**; and universal test 4, **{'pass' if summary['all_candidate_gates']['test_4_universal'] else 'fail'}**. This adjudicates lineage identity only within the selected capable-rule cohort.

## Proposed limitation

The lineage-identity assays are reviewer-prompted and conditional on catalytic matrices already known to produce at least one strict event in both contracts. Their finite horizons and operational clustering thresholds do not establish or exclude mathematical infinite-time attractors. Ordinary F12 remains a local adjacency endpoint and should not be interpreted as a stable molecular identity.
"""
    (OUTPUT_ROOT / "PROPOSED_MANUSCRIPT_AND_REVIEWER_PATCH.md").write_text(patch, encoding="utf-8")
    manifest = _read_json(OUTPUT_ROOT / "analysis_manifest.json")
    manifest["status"] = "reports_complete_verification_pending"
    manifest["figures"] = sorted(path.name for path in (OUTPUT_ROOT / "figures").glob("*.png"))
    manifest["reports"] = [
        "SCIENTIFIC_REPORT.md",
        "APPENDIX_LINEAGE_IDENTITY.md",
        "PROPOSED_MANUSCRIPT_AND_REVIEWER_PATCH.md",
    ]
    _write_json(OUTPUT_ROOT / "analysis_manifest.json", manifest)
    _write_checksums(OUTPUT_ROOT)
    print(f"Reports and figures written to {OUTPUT_ROOT}")


def _validate_checkpoint(path: Path, expected_candidate: str, expected_matrix: int) -> dict[str, Any]:
    bundle = _load_npz(path)
    if str(bundle["format"]) != FORMAT:
        raise ValueError(f"checkpoint format mismatch: {path}")
    protocol_id = _read_json(PROTOCOL_PATH)["protocol_id"]
    if str(bundle["protocol_id"]) != protocol_id:
        raise ValueError(f"checkpoint protocol mismatch: {path}")
    if str(bundle["candidate"]) != expected_candidate or int(bundle["matrix_id"]) != expected_matrix:
        raise ValueError(f"checkpoint cell mismatch: {path}")
    return {"path": str(path), "sha256": sha256_file(path)}


def _compare_arrays(left: dict[str, np.ndarray], right: dict[str, Any]) -> tuple[bool, float]:
    exact = True
    maximum = 0.0
    for name in ("parents", "daughters", "growth_steps", "observed", "completed"):
        exact = exact and bool(np.array_equal(left[name], right[name]))
    a = left["boundary_h"]
    b = right["boundary_h"]
    exact = exact and bool(np.array_equal(a, b, equal_nan=True))
    finite = np.isfinite(a) & np.isfinite(b)
    if np.any(finite):
        maximum = float(np.max(np.abs(a[finite] - b[finite])))
    return exact, maximum


def verify(full_replay: bool) -> None:
    protocol = verify_protocol()
    if VERIFICATION_ROOT.exists():
        raise FileExistsError(f"refusing to overwrite verification bundle: {VERIFICATION_ROOT}")
    VERIFICATION_ROOT.mkdir(parents=True)
    checkpoint_rows: list[dict[str, Any]] = []
    replay_exact = True
    maximum_h_error = 0.0
    replayed_lineages = 0
    for candidate, matrix_id in _cell_tasks():
        fixed_path = _checkpoint_path(candidate, matrix_id, "fixed")
        if not fixed_path.is_file():
            raise FileNotFoundError(fixed_path)
        checkpoint_rows.append(_validate_checkpoint(fixed_path, candidate, matrix_id))
        fixed = _load_npz(fixed_path)
        if fixed["daughters"].shape != (PRIMARY_LINEAGES, GENERATIONS, GARD.n_types):
            raise ValueError(f"fixed checkpoint shape mismatch: {fixed_path}")
        if _needs_extension(candidate, matrix_id):
            extension_path = _checkpoint_path(candidate, matrix_id, "extension")
            if not extension_path.is_file():
                raise FileNotFoundError(extension_path)
            checkpoint_rows.append(_validate_checkpoint(extension_path, candidate, matrix_id))
        fork_path = _fork_path(candidate, matrix_id)
        if not fork_path.is_file():
            raise FileNotFoundError(fork_path)
        checkpoint_rows.append(_validate_checkpoint(fork_path, candidate, matrix_id))
        if full_replay:
            regenerated = _simulate_range(candidate, matrix_id, 0, PRIMARY_LINEAGES)
            exact, error = _compare_arrays(fixed, regenerated)
            replay_exact = replay_exact and exact
            maximum_h_error = max(maximum_h_error, error)
            replayed_lineages += PRIMARY_LINEAGES
            extension_path = _checkpoint_path(candidate, matrix_id, "extension")
            if extension_path.is_file():
                extension = _load_npz(extension_path)
                regenerated_extension = _simulate_range(candidate, matrix_id, PRIMARY_LINEAGES, MAX_LINEAGES)
                exact, error = _compare_arrays(extension, regenerated_extension)
                replay_exact = replay_exact and exact
                maximum_h_error = max(maximum_h_error, error)
                replayed_lineages += MAX_LINEAGES - PRIMARY_LINEAGES
            expected_fork = {
                **_simulate_forks_for_kind(candidate, matrix_id, "strict", _collect_episodes(candidate, matrix_id, "strict")),
                **_simulate_forks_for_kind(candidate, matrix_id, "f12", _collect_episodes(candidate, matrix_id, "f12")),
            }
            fork_bundle = _load_npz(fork_path)
            for name, values in expected_fork.items():
                replay_exact = replay_exact and bool(np.array_equal(fork_bundle[name], values))
    output_verified = False
    if (OUTPUT_ROOT / "SHA256SUMS").is_file():
        _verify_checksums(OUTPUT_ROOT)
        output_verified = True
    audit = {
        "format": FORMAT,
        "protocol_id": protocol["protocol"]["protocol_id"],
        "registration_id": protocol["registration"]["registration_id"],
        "checkpoint_files": len(checkpoint_rows),
        "checkpoint_digests": checkpoint_rows,
        "output_checksums_verified": output_verified,
        "full_replay_requested": full_replay,
        "replayed_lineages": replayed_lineages,
        "discrete_replay_exact": replay_exact if full_replay else None,
        "maximum_h_error": maximum_h_error if full_replay else None,
        "complete": bool(output_verified and full_replay and replay_exact and maximum_h_error == 0.0),
    }
    _write_json(VERIFICATION_ROOT / "verification_audit.json", audit)
    _write_checksums(VERIFICATION_ROOT)
    if full_replay and not audit["complete"]:
        raise ValueError("full verification did not meet the completion contract")
    print(json.dumps(_json_ready(audit), indent=2, sort_keys=True))


def status() -> None:
    protocol = PROTOCOL_PATH.is_file()
    selected = _selected_rules() if protocol else []
    cells = [(candidate, matrix_id) for candidate in ("02", "03") for matrix_id in selected]
    fixed = sum(_checkpoint_path(candidate, matrix_id, "fixed").is_file() for candidate, matrix_id in cells)
    extension = sum(_checkpoint_path(candidate, matrix_id, "extension").is_file() for candidate, matrix_id in cells)
    forks = sum(_fork_path(candidate, matrix_id).is_file() for candidate, matrix_id in cells)
    payload = {
        "protocol_sealed": protocol,
        "cells": len(cells),
        "fixed_checkpoints": fixed,
        "extension_checkpoints": extension,
        "fork_checkpoints": forks,
        "analysis_complete": (OUTPUT_ROOT / "primary_summary.json").is_file(),
        "reports_complete": (OUTPUT_ROOT / "SCIENTIFIC_REPORT.md").is_file(),
        "verification_complete": (VERIFICATION_ROOT / "verification_audit.json").is_file(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare", help="freeze sources, rules, seeds, endpoints, and gates")
    simulation = subparsers.add_parser("simulate", help="generate resumable random-start lineage checkpoints")
    simulation.add_argument("--mode", choices=("fixed", "extension", "all"), default="all")
    simulation.add_argument("--workers", type=int, default=1)
    forks = subparsers.add_parser("fork", help="generate independent strict-B and F12-control forks")
    forks.add_argument("--workers", type=int, default=1)
    subparsers.add_parser("analyze", help="compute all registered tests and sensitivity tables")
    subparsers.add_parser("report", help="render reports, proposed text, and figures")
    subparsers.add_parser("status", help="read campaign progress without writing")
    verification = subparsers.add_parser("verify", help="verify seals, checkpoints, outputs, and optional full replay")
    verification.add_argument("--full-replay", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        prepare()
    elif args.command == "simulate":
        simulate(args.mode, args.workers)
    elif args.command == "fork":
        fork(args.workers)
    elif args.command == "analyze":
        analyze()
    elif args.command == "report":
        report()
    elif args.command == "status":
        status()
    elif args.command == "verify":
        verify(args.full_replay)
    else:  # pragma: no cover
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
