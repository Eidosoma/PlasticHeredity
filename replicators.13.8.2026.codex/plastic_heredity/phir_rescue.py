"""R0 replay-and-remeasurement for the Chapter 5 Phi-r rescue program."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from threadpoolctl import threadpool_limits

from .config import CANDIDATES, GardConfig
from .intervention_core import MolecularEdit, _records_digest, edited_snapshot
from .mechanistic import (
    _atomic_destination,
    _canonical_digest,
    sha256_file,
    verify_checksums,
    write_checksums,
)
from .mechanistic_metrics import holm_adjust
from .phir_ch5 import _append_ledger, _json_ready, _runtime_versions, _snapshot_after_record
from .phir_instruments import ATOM_NAMES, advance_fission_traced
from .phir_protocol_adjudication import (
    DEFAULT_OUTPUT as PAB24_OUTPUT,
    DEFAULT_REGISTRATION as PAB24_REGISTRATION,
    DIRECTIONS,
    Segment,
    _future_seed as pab_future_seed,
    _physical_fields,
    _score_trace_fields,
    scientific_spec as pab_scientific_spec,
    trace_representations,
    verify_result as verify_pab24_result,
)
from .phir_rescue_instruments import (
    active_partition,
    atom_mapping,
    beta_physical_partition,
    calibrate_numit,
    close_all_clr,
    full_block_revised,
    generate_numit_library,
    macro_phi_score,
    matched_partition_null,
    rank_gaussianize,
)
from .seeds import derive_seed
from .simulator import FissionRecord, SimulationError, Snapshot


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DOCUMENT = "CODEX_CH5_PHIR_RESCUE_PREREGISTRATION.md"
AMENDMENT_DOCUMENT = "CODEX_CH5_PHIR_RESCUE_PROCEDURAL_AMENDMENT_001.md"
ORIGINAL_REGISTRATION_ID = "d4e26c19d5a39b80c64a048e195ede50d21281092e863cd7e4dddfaa704762b8"

DEFAULT_VALIDATION = RESULTS / "phir_rescue_validation"
DEFAULT_REGISTRATION = RESULTS / "phir_rescue_registration"
DEFAULT_SMOKE = RESULTS / "phir_rescue_smoke"
DEFAULT_OUTPUT = RESULTS / "phir_rescue_r0"
DEFAULT_WORK = RESULTS / ".phir_rescue_r0_work"
DEFAULT_LOG = RESULTS / "phir_rescue_r0.log"

LABEL = "CODEX_CH5_PHIR_RESCUE_R0_V1"
PROGRAM_FORMAT = "codex-ch5-phir-rescue-program-v1"
REGISTRATION_FORMAT = "codex-ch5-phir-rescue-registration-v1"
CHECKPOINT_FORMAT = "codex-ch5-phir-rescue-checkpoint-v1"
RESULT_FORMAT = "codex-ch5-phir-rescue-result-v1"
STATUS_FORMAT = "codex-ch5-phir-rescue-status-v1"
SERVICE_NAME = "codex-phir-rescue-r0-20260819"

MATRICES = 24
REPLICATES = 2
FINAL_START = 31
PHASE_POINTS = 16
PARTITION_NULL_DRAWS = 128
NUMIT_SYSTEMS = 4096
NUMIT_NEIGHBORS = 256
NUMIT_BURN = 512
BOOTSTRAP_REPETITIONS = 4096
RANDOMIZATION_REPETITIONS = 4096
MINIMUM_FREE_DISK_BYTES = 1_500_000_000
CPU_BUDGET_SECONDS = 30 * 60 * 60

ARMS = (
    "FRESH__NOOP",
    "FRESH__EXHAUSTIVE__STABILIZE",
    "FRESH__EXHAUSTIVE__DESTABILIZE",
)
REPRESENTATIONS = ("fable_style", "phase_normalized", "generational")
CANDIDATE_MENU = (
    ("NUMIT_MACRO", "numit_probit"),
    ("PARTITION_NULL_FULL", "partition_null_z"),
    ("FULL_BLOCK_RAW", "full_block_raw"),
    ("COPULA_PUBLIC_RAW", "copula_public_raw"),
)

SOURCE_FILES = (
    DOCUMENT,
    AMENDMENT_DOCUMENT,
    "plastic_heredity/phir_rescue.py",
    "plastic_heredity/phir_rescue_instruments.py",
    "tests/test_phir_rescue.py",
    "plastic_heredity/phir_protocol_adjudication.py",
    "plastic_heredity/phir_instruments.py",
    "plastic_heredity/phir_ch5.py",
    "plastic_heredity/config.py",
    "plastic_heredity/intervention_core.py",
    "plastic_heredity/mechanistic.py",
    "plastic_heredity/mechanistic_metrics.py",
    "plastic_heredity/seeds.py",
    "plastic_heredity/simulator.py",
)

CANONICAL_MODULE = "plastic_heredity.phir_rescue"
if __name__ == "__main__":
    # Pickles written by ``python -m`` must name the importable module rather
    # than the ephemeral ``__main__`` module.
    sys.modules[CANONICAL_MODULE] = sys.modules[__name__]


def _seed_value(name: str) -> str:
    return hashlib.sha256(f"{LABEL}::{name}".encode("utf-8")).hexdigest()


SEED_DOMAINS = {
    name: _seed_value(name)
    for name in (
        "numit",
        "partition_null",
        "bootstrap",
        "randomization",
        "validation",
        "smoke",
        "replay",
    )
}


@dataclass(frozen=True)
class RescueSpec:
    label: str
    matrices: int
    replicates: int
    final_start: int
    phase_points: int
    partition_null_draws: int
    numit_systems: int
    numit_neighbors: int
    numit_burn: int
    bootstrap_repetitions: int
    randomization_repetitions: int
    cpu_budget_seconds: int


@dataclass(frozen=True)
class RescueBatch:
    matrix_id: int
    lineage_rows: tuple[dict[str, Any], ...]
    cpu_seconds: float
    scientific_digest: str


RescueSpec.__module__ = CANONICAL_MODULE
RescueBatch.__module__ = CANONICAL_MODULE


def scientific_spec() -> RescueSpec:
    return RescueSpec(
        "r0",
        MATRICES,
        REPLICATES,
        FINAL_START,
        PHASE_POINTS,
        PARTITION_NULL_DRAWS,
        NUMIT_SYSTEMS,
        NUMIT_NEIGHBORS,
        NUMIT_BURN,
        BOOTSTRAP_REPETITIONS,
        RANDOMIZATION_REPETITIONS,
        CPU_BUDGET_SECONDS,
    )


def smoke_spec() -> RescueSpec:
    return RescueSpec("smoke", 1, 1, 2, 4, 8, 32, 16, 16, 32, 32, 600)


def _source_hashes() -> dict[str, str]:
    return {name: sha256_file(ROOT / name) for name in SOURCE_FILES}


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(_json_ready(value), sort_keys=True, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=5)
    temporary.replace(path)


def _write_checkpoint_fixture(path: Path) -> RescueBatch:
    """Write a tiny non-scientific checkpoint for cross-process validation."""

    provisional = RescueBatch(
        matrix_id=0,
        lineage_rows=({"fixture": "checkpoint-portability", "values": [1, 2, 3]},),
        cpu_seconds=0.0,
        scientific_digest="",
    )
    batch = RescueBatch(
        matrix_id=provisional.matrix_id,
        lineage_rows=provisional.lineage_rows,
        cpu_seconds=provisional.cpu_seconds,
        scientific_digest=_batch_digest(provisional),
    )
    _atomic_pickle(path, batch)
    return batch


def _batch_digest(batch: RescueBatch) -> str:
    # Runtime accounting is operational metadata and is intentionally excluded
    # from the scientific replay identity.
    value = RescueBatch(batch.matrix_id, batch.lineage_rows, 0.0, "")
    return _canonical_digest(_json_ready(asdict(value)))


def protocol() -> dict[str, Any]:
    predecessor = verify_pab24_result()
    value: dict[str, Any] = {
        "format": PROGRAM_FORMAT,
        "program": "Chapter 5 Phi-r strongest-fair-test R0",
        "immutable_predecessor": {
            "registration_id": predecessor["registration_id"],
            "manifest_sha256": sha256_file(PAB24_OUTPUT / "manifest.json"),
            "result": str(PAB24_OUTPUT.relative_to(ROOT)),
        },
        "spec": asdict(scientific_spec()),
        "substrate": {
            "matrices": MATRICES,
            "candidates": list(CANDIDATES),
            "replicates": REPLICATES,
            "launch": "FRESH",
            "arms": list(ARMS),
            "new_scientific_matrices": 0,
        },
        "representations": list(REPRESENTATIONS),
        "candidate_menu": [name for name, _ in CANDIDATE_MENU],
        "selection_order_is_frozen": True,
        "nulls": {
            "numit_systems": NUMIT_SYSTEMS,
            "numit_neighbors": NUMIT_NEIGHBORS,
            "numit_burn": NUMIT_BURN,
            "transition_bucket": "nearest positive multiple of 16, half upward",
            "partition_draws": PARTITION_NULL_DRAWS,
        },
        "inference": {
            "unit": "whole catalytic matrix",
            "bootstrap_draws": BOOTSTRAP_REPETITIONS,
            "sign_randomizations": RANDOMIZATION_REPETITIONS,
            "candidates_pooled": False,
            "replicates_pooled": False,
        },
        "execution": {
            "detached": True,
            "maximum_workers": 12,
            "cpu_budget_seconds": CPU_BUDGET_SECONDS,
            "minimum_free_disk_bytes": MINIMUM_FREE_DISK_BYTES,
            "complete_exact_replay": True,
        },
        "r1_locked_pending_human_authorization": True,
        "no_48_matrix_run": True,
        "procedural_amendment": {
            "number": 1,
            "document": AMENDMENT_DOCUMENT,
            "document_sha256": sha256_file(ROOT / AMENDMENT_DOCUMENT),
            "supersedes_registration_id": ORIGINAL_REGISTRATION_ID,
            "scientific_contract_changed": False,
            "failed_lineage_checkpoints_reused": False,
            "deterministic_numit_libraries_may_be_hash_verified_and_reused": True,
        },
        "external_fable_code_data_models_or_seeds_imported": False,
        "claim_boundary": [
            "R0 is development remeasurement, not confirmation",
            "legacy raw nine-atom negative result is unchanged",
            "no consciousness, agency, biological life, or metaphysical claim",
            "no cross-clean-room rescue before an independent mirrored result",
        ],
    }
    value["protocol_id"] = _canonical_digest(_json_ready(value))
    return value


def _transition_bucket(transitions: int) -> int:
    if transitions < 2:
        raise ValueError("NuMIT requires at least two transitions")
    return max(16, 16 * ((int(transitions) + 8) // 16))


def _library_path(bucket: int) -> Path:
    return DEFAULT_WORK / "numit_libraries" / f"transitions_{bucket:04d}.npz"


def _required_buckets() -> tuple[int, ...]:
    frame = pd.read_csv(PAB24_OUTPUT / "pab24_lineages.csv.gz")
    frame["candidate"] = frame["candidate"].astype(str).str.zfill(2)
    selected = frame[
        (frame["launch"] == "FRESH")
        & (frame["arm"].isin(ARMS))
        & (frame["information_eligible"] == 1)
    ]
    buckets: set[int] = set()
    for representation in REPRESENTATIONS:
        column = f"{representation}_observations"
        for observations in selected[column].dropna().astype(int):
            buckets.add(_transition_bucket(observations - 1))
    return tuple(sorted(buckets))


def _numit_seed(bucket: int, spec: RescueSpec) -> int:
    domain = SEED_DOMAINS["smoke"] if spec.label == "smoke" else SEED_DOMAINS["numit"]
    return derive_seed(domain, LABEL, spec.label, "numit", bucket)


def _partition_seed(
    spec: RescueSpec,
    matrix_id: int,
    candidate: str,
    replicate: int,
    representation: str,
) -> int:
    domain = SEED_DOMAINS["smoke"] if spec.label == "smoke" else SEED_DOMAINS["partition_null"]
    return derive_seed(
        domain,
        LABEL,
        spec.label,
        matrix_id,
        candidate,
        replicate,
        representation,
    )


def _generate_library_task(args: tuple[int, RescueSpec, str]) -> tuple[int, float]:
    bucket, spec, destination_text = args
    destination = Path(destination_text)
    started = time.process_time()
    with threadpool_limits(limits=1):
        library = generate_numit_library(
            bucket,
            spec.numit_systems,
            np.random.default_rng(_numit_seed(bucket, spec)),
            burn=spec.numit_burn,
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **library)
    temporary.replace(destination)
    return bucket, time.process_time() - started


def _write_status(stage: str, completed: int, total: int, **extra: Any) -> None:
    safe = stage.replace("/", "_")
    started_path = DEFAULT_WORK / f"started_at_{safe}.txt"
    if not started_path.exists():
        started_path.parent.mkdir(parents=True, exist_ok=True)
        started_path.write_text(str(time.time()), encoding="ascii")
    elapsed = max(0.0, time.time() - float(started_path.read_text(encoding="ascii")))
    rate = completed / elapsed if completed and elapsed else 0.0
    _atomic_json(
        DEFAULT_WORK / "campaign_status.json",
        {
            "format": STATUS_FORMAT,
            "stage": stage,
            "completed": completed,
            "total": total,
            "fraction": completed / total if total else 1.0,
            "elapsed_seconds": elapsed,
            "eta_seconds": (total - completed) / rate if rate else None,
            "pid": os.getpid(),
            "free_disk_bytes": shutil.disk_usage(ROOT).free,
            **extra,
        },
    )


def _prepare_libraries(spec: RescueSpec, workers: int) -> tuple[tuple[int, ...], float]:
    buckets = _required_buckets() if spec.label == "r0" else (16, 32)
    missing = [bucket for bucket in buckets if not _library_path(bucket).exists()]
    completed = len(buckets) - len(missing)
    cpu_seconds = 0.0
    _write_status("numit_libraries", completed, len(buckets), reused=completed)
    arguments = [(bucket, spec, str(_library_path(bucket))) for bucket in missing]
    pool_workers = min(max(1, workers), 4)
    executor: ProcessPoolExecutor | None = None
    generated: Iterable[tuple[int, float]]
    if pool_workers == 1:
        generated = map(_generate_library_task, arguments)
    else:
        executor = ProcessPoolExecutor(max_workers=pool_workers)
        generated = executor.map(_generate_library_task, arguments, chunksize=1)
    try:
        for bucket, local_cpu in generated:
            cpu_seconds += local_cpu
            completed += 1
            _write_status(
                "numit_libraries",
                completed,
                len(buckets),
                reused=len(buckets) - len(missing),
                cpu_seconds=cpu_seconds,
                latest_bucket=bucket,
            )
            print(f"[NuMIT] {completed}/{len(buckets)} buckets", flush=True)
            if cpu_seconds > spec.cpu_budget_seconds:
                raise RuntimeError("registered R0 CPU pause boundary reached during NuMIT")
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
    for bucket in buckets:
        with np.load(_library_path(bucket)) as library:
            if (
                int(library["transition_count"][0]) != bucket
                or int(library["systems"][0]) != spec.numit_systems
            ):
                raise ValueError(f"invalid NuMIT library {bucket}")
    return buckets, cpu_seconds


def _archive_tables() -> tuple[pd.DataFrame, pd.DataFrame, Mapping[str, NDArray]]:
    lineages = pd.read_csv(PAB24_OUTPUT / "pab24_lineages.csv.gz")
    actions = pd.read_csv(PAB24_OUTPUT / "selected_edits.csv.gz")
    for frame in (lineages, actions):
        frame["candidate"] = frame["candidate"].astype(str).str.zfill(2)
    lineages = lineages[(lineages["launch"] == "FRESH") & lineages["arm"].isin(ARMS)].copy()
    actions = actions[
        (actions["phase"] == "PAB24")
        & (actions["launch"] == "FRESH")
        & actions["arm"].isin(ARMS)
    ].copy()
    matrices = np.load(PAB24_OUTPUT / "matrix_inputs.npz")
    return lineages, actions, matrices


def _library(bucket: int) -> dict[str, NDArray]:
    with np.load(_library_path(bucket)) as values:
        return {name: np.asarray(values[name]).copy() for name in values.files}


def _score_representation(
    counts: NDArray,
    beta: NDArray,
    spec: RescueSpec,
    matrix_id: int,
    candidate: str,
    replicate: int,
    representation: str,
) -> dict[str, Any]:
    clr = close_all_clr(counts)
    data, active = rank_gaussianize(clr)
    physical_a, physical_b = beta_physical_partition(beta)
    first, second = active_partition(active, physical_a, physical_b)
    macro = macro_phi_score(data, first, second)
    full = full_block_revised(data, first, second)
    null, null_values = matched_partition_null(
        data,
        int(first.size),
        spec.partition_null_draws,
        np.random.default_rng(
            _partition_seed(spec, matrix_id, candidate, replicate, representation)
        ),
        full.revised,
    )
    transitions = int(counts.shape[0] - 1)
    bucket = _transition_bucket(transitions)
    calibration = calibrate_numit(
        macro.revised,
        macro.whole_mi,
        _library(bucket),
        neighbors=spec.numit_neighbors,
    )
    output: dict[str, Any] = {
        "copula_observations": int(counts.shape[0]),
        "copula_transitions": transitions,
        "copula_active_dimensions": int(active.size),
        "copula_physical_partition_a": [int(active[index]) for index in first],
        "copula_physical_partition_b": [int(active[index]) for index in second],
        "copula_public_raw": macro.revised,
        "macro_whole_mi": macro.whole_mi,
        "copula_causation": macro.causation,
        "copula_emergence": macro.emergence,
        "copula_synergy": macro.synergy_persistence,
        "full_block_raw": full.revised,
        "full_whole_mi": full.whole_mi,
        "full_aa_mi": full.aa_mi,
        "full_ab_mi": full.ab_mi,
        "full_ba_mi": full.ba_mi,
        "full_bb_mi": full.bb_mi,
        "full_double_redundancy": full.double_redundancy,
        "partition_null_mean": null.null_mean,
        "partition_null_std": null.null_std,
        "partition_null_z": null.z_score,
        "partition_null_percentile": null.percentile,
        "partition_null_values": null_values.tolist(),
        "numit_bucket": bucket,
        "numit_percentile": calibration.percentile,
        "numit_probit": calibration.probit,
        "numit_valid": int(calibration.valid),
        "numit_reference_min_mi": calibration.reference_min_mi,
        "numit_reference_max_mi": calibration.reference_max_mi,
    }
    output.update(
        {f"copula_atom_{name}": value for name, value in atom_mapping(macro).items()}
    )
    return output


def _trace_archived_arm(
    matrix_id: int,
    beta: NDArray,
    initial: NDArray,
    candidate: str,
    replicate: int,
    arm: str,
    action_rows: pd.DataFrame,
    spec: RescueSpec,
) -> dict[str, Any]:
    config = GardConfig()
    pab_spec = pab_scientific_spec()
    rng = np.random.default_rng(pab_future_seed(pab_spec, candidate, matrix_id, replicate))
    snapshot = Snapshot(np.asarray(initial, dtype=np.int64).copy(), 0, (), ())
    records: list[FissionRecord] = []
    segments: list[Segment] = []
    local_actions = action_rows[
        (action_rows["matrix_id"] == matrix_id)
        & (action_rows["candidate"] == candidate)
        & (action_rows["replicate"] == replicate)
        & (action_rows["arm"] == arm)
    ].sort_values("step")
    if len(local_actions) != pab_spec.control_horizon:
        raise ValueError(f"archived action sequence incomplete for {matrix_id}/{candidate}/{replicate}/{arm}")
    for step in range(1, pab_spec.control_horizon + 1):
        pre_growth = snapshot.composition.copy()
        try:
            traced = advance_fission_traced(
                pre_growth, beta, config, CANDIDATES[candidate], rng
            )
        except SimulationError:
            break
        records.append(traced.record)
        snapshot = _snapshot_after_record(snapshot, traced.record)
        action = local_actions.iloc[step - 1]
        if int(action["step"]) != step:
            raise ValueError("archived action ordering changed")
        edit: MolecularEdit | None = None
        if int(action["edit_applied"]):
            if step >= pab_spec.control_horizon:
                raise ValueError("archived PAB24 final-step edit is forbidden")
            edit = MolecularEdit(int(action["remove_type"]), int(action["add_type"]))
            snapshot = edited_snapshot(snapshot, edit)
        segments.append(
            Segment(
                step=step,
                pre_growth=pre_growth,
                growth_observations=tuple(
                    np.asarray(value, dtype=np.int64).copy()
                    for value in traced.growth_observations
                ),
                record=traced.record,
                post_control=snapshot.composition.copy(),
                edit=edit,
            )
        )
    complete = len(segments) == pab_spec.control_horizon
    direction = "NOOP" if arm.endswith("NOOP") else arm.rsplit("__", 1)[1]
    row: dict[str, Any] = {
        "matrix_id": matrix_id,
        "candidate": candidate,
        "replicate": replicate,
        "arm": arm,
        "direction": direction,
        "completed_horizon": int(complete),
        "information_eligible": int(complete),
        "completed_fissions": len(segments),
        "extinct": int(not complete),
        "controlled_record_digest": _records_digest(records),
        "final_rng_state_digest": _canonical_digest(_json_ready(rng.bit_generator.state)),
        "final_composition": snapshot.composition.astype(int).tolist(),
    }
    row.update(_physical_fields(segments, beta, spec.final_start))
    row.update(_score_trace_fields(segments, pab_spec, include_registered=False))
    representations = trace_representations(
        segments, spec.final_start, spec.phase_points, include_registered=False
    )
    if complete:
        for representation in REPRESENTATIONS:
            scores = _score_representation(
                representations[representation],
                beta,
                spec,
                matrix_id,
                candidate,
                replicate,
                representation,
            )
            row.update({f"{representation}_{name}": value for name, value in scores.items()})
    return row


def _run_matrix(args: tuple[int, RescueSpec]) -> RescueBatch:
    matrix_id, spec = args
    started = time.process_time()
    with threadpool_limits(limits=1):
        lineages, actions, matrices = _archive_tables()
        index = int(np.flatnonzero(matrices["pab24_matrix_id"] == matrix_id)[0])
        beta = np.asarray(matrices["pab24_beta"][index], dtype=np.float64)
        initial = np.asarray(matrices["pab24_initial"][index], dtype=np.int64)
        rows = tuple(
            _trace_archived_arm(
                matrix_id, beta, initial, candidate, replicate, arm, actions, spec
            )
            for candidate in CANDIDATES
            for replicate in range(spec.replicates)
            for arm in ARMS
        )
    provisional = RescueBatch(matrix_id, rows, time.process_time() - started, "")
    return RescueBatch(
        provisional.matrix_id,
        provisional.lineage_rows,
        provisional.cpu_seconds,
        _batch_digest(provisional),
    )


def _checkpoint_contract(spec: RescueSpec, registration_id: str, stage: str) -> dict[str, Any]:
    value = {
        "format": CHECKPOINT_FORMAT,
        "registration_id": registration_id,
        "protocol_id": protocol()["protocol_id"],
        "stage": stage,
        "spec": asdict(spec),
        "source_hashes": _source_hashes(),
    }
    value["contract_id"] = _canonical_digest(_json_ready(value))
    return value


def _run_checkpointed(
    spec: RescueSpec,
    registration_id: str,
    directory: Path,
    stage: str,
    workers: int,
    prior_cpu_seconds: float,
) -> tuple[list[RescueBatch], float]:
    directory.mkdir(parents=True, exist_ok=True)
    contract = _checkpoint_contract(spec, registration_id, stage)
    contract_path = directory / "checkpoint_contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text(encoding="utf-8")) != _json_ready(contract):
            raise ValueError(f"checkpoint contract changed: {directory}")
    else:
        _atomic_json(contract_path, contract)
    batches: list[RescueBatch | None] = [None] * spec.matrices
    missing: list[int] = []
    cpu_seconds = prior_cpu_seconds
    for matrix_id in range(spec.matrices):
        path = directory / f"matrix_{matrix_id:04d}.pkl"
        if not path.exists():
            missing.append(matrix_id)
            continue
        with path.open("rb") as handle:
            batch = pickle.load(handle)
        if not isinstance(batch, RescueBatch) or batch.scientific_digest != _batch_digest(batch):
            raise ValueError(f"invalid R0 checkpoint {path}")
        batches[matrix_id] = batch
        cpu_seconds += batch.cpu_seconds
    completed = spec.matrices - len(missing)
    _write_status(stage, completed, spec.matrices, reused=completed, cpu_seconds=cpu_seconds)
    arguments = [(matrix_id, spec) for matrix_id in missing]
    executor: ProcessPoolExecutor | None = None
    generated: Iterable[RescueBatch]
    if workers <= 1:
        generated = map(_run_matrix, arguments)
    else:
        executor = ProcessPoolExecutor(max_workers=min(workers, 12))
        generated = executor.map(_run_matrix, arguments, chunksize=1)
    try:
        for matrix_id, batch in zip(missing, generated, strict=True):
            if batch.matrix_id != matrix_id or batch.scientific_digest != _batch_digest(batch):
                raise AssertionError("R0 worker returned an invalid batch")
            batches[matrix_id] = batch
            _atomic_pickle(directory / f"matrix_{matrix_id:04d}.pkl", batch)
            cpu_seconds += batch.cpu_seconds
            completed += 1
            _write_status(
                stage,
                completed,
                spec.matrices,
                reused=spec.matrices - len(missing),
                cpu_seconds=cpu_seconds,
            )
            print(f"[{stage}] {completed}/{spec.matrices} matrices", flush=True)
            if cpu_seconds > spec.cpu_budget_seconds:
                raise RuntimeError("registered R0 CPU pause boundary reached")
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
    if any(batch is None for batch in batches):
        raise AssertionError(f"R0 checkpoint stage incomplete: {stage}")
    return [batch for batch in batches if batch is not None], cpu_seconds


def _load_stage(directory: Path, spec: RescueSpec) -> list[RescueBatch]:
    output: list[RescueBatch] = []
    for matrix_id in range(spec.matrices):
        path = directory / f"matrix_{matrix_id:04d}.pkl"
        with path.open("rb") as handle:
            batch = pickle.load(handle)
        if not isinstance(batch, RescueBatch) or batch.scientific_digest != _batch_digest(batch):
            raise ValueError(f"invalid R0 checkpoint {path}")
        output.append(batch)
    return output


def _archive_audit(batches: Sequence[RescueBatch]) -> dict[str, Any]:
    observed = pd.DataFrame([row for batch in batches for row in batch.lineage_rows])
    observed["candidate"] = observed["candidate"].astype(str).str.zfill(2)
    archived = pd.read_csv(PAB24_OUTPUT / "pab24_lineages.csv.gz")
    archived["candidate"] = archived["candidate"].astype(str).str.zfill(2)
    archived = archived[(archived["launch"] == "FRESH") & archived["arm"].isin(ARMS)].copy()
    keys = ["matrix_id", "candidate", "replicate", "arm"]
    observed = observed.sort_values(keys).reset_index(drop=True)
    archived = archived.sort_values(keys).reset_index(drop=True)
    key_exact = len(observed) == len(archived) and observed[keys].equals(archived[keys])
    exact_columns = (
        "completed_horizon",
        "information_eligible",
        "completed_fissions",
        "extinct",
        "controlled_record_digest",
        "final_rng_state_digest",
    )
    exact_mismatches: dict[str, int] = {}
    if key_exact:
        for column in exact_columns:
            exact_mismatches[column] = int(
                np.count_nonzero(observed[column].astype(str) != archived[column].astype(str))
            )
    maximum_legacy_error = 0.0
    legacy_mismatches = 0
    suffixes = (
        "revised",
        "full_typeset",
        "macro_typeset",
        "normalized_full",
        "causation",
        "emergence",
        "synergy",
        *(f"atom_{name}" for name in ATOM_NAMES),
    )
    if key_exact:
        for representation in REPRESENTATIONS:
            for suffix in suffixes:
                column = f"{representation}_{suffix}"
                left = observed[column].to_numpy(float)
                right = archived[column].to_numpy(float)
                equal_nan = np.isnan(left) & np.isnan(right)
                finite = np.isfinite(left) & np.isfinite(right)
                errors = np.abs(left[finite] - right[finite])
                if errors.size:
                    maximum_legacy_error = max(maximum_legacy_error, float(errors.max()))
                    legacy_mismatches += int(np.count_nonzero(errors > 1e-12))
                legacy_mismatches += int(np.count_nonzero(~(finite | equal_nan)))
    inherited_error = (
        float(np.max(np.abs(observed["inherited_31_60"] - archived["inherited_31_60"])))
        if key_exact
        else float("inf")
    )
    final_composition_mismatches = 0
    if key_exact:
        for left, right in zip(observed["final_composition"], archived["final_composition"], strict=True):
            parsed = json.loads(right) if isinstance(right, str) else right
            final_composition_mismatches += int(list(left) != list(parsed))
    passed = bool(
        key_exact
        and not any(exact_mismatches.values())
        and legacy_mismatches == 0
        and maximum_legacy_error <= 1e-12
        and inherited_error <= 1e-15
        and final_composition_mismatches == 0
    )
    return {
        "format": "codex-ch5-phir-rescue-archive-audit-v1",
        "lineages": int(len(observed)),
        "key_exact": key_exact,
        "exact_mismatches": exact_mismatches,
        "legacy_score_mismatches": legacy_mismatches,
        "maximum_legacy_score_error": maximum_legacy_error,
        "maximum_inherited_fraction_error": inherited_error,
        "final_composition_mismatches": final_composition_mismatches,
        "passed": passed,
    }


def _replay_audit(
    generated: Sequence[RescueBatch], replayed: Sequence[RescueBatch]
) -> dict[str, Any]:
    rows = [
        {
            "matrix_id": left.matrix_id,
            "generated_digest": left.scientific_digest,
            "replay_digest": right.scientific_digest,
            "exact": left.scientific_digest == right.scientific_digest,
        }
        for left, right in zip(generated, replayed, strict=True)
    ]
    return {
        "format": "codex-ch5-phir-rescue-replay-v1",
        "matrices": rows,
        "complete_exact_replay": bool(len(rows) == MATRICES and all(row["exact"] for row in rows)),
    }


def _seeded_rng(domain: str, *keys: object) -> np.random.Generator:
    return np.random.default_rng(derive_seed(SEED_DOMAINS[domain], LABEL, *keys))


def _summary(values: NDArray, key: str, arrays: dict[str, NDArray], spec: RescueSpec) -> dict[str, Any]:
    vector = np.asarray(values, dtype=np.float64)
    vector = vector[np.isfinite(vector)]
    if not vector.size:
        return {
            "effect": float("nan"), "ci95": [float("nan"), float("nan")],
            "positive_p": float("nan"), "matrices": 0, "matrices_positive": 0,
            "loo_positive": 0,
        }
    bootstrap_indices = _seeded_rng("bootstrap", key).integers(
        0, vector.size, size=(spec.bootstrap_repetitions, vector.size)
    )
    bootstrap = vector[bootstrap_indices].mean(axis=1)
    signs = _seeded_rng("randomization", key).choice(
        (-1.0, 1.0), size=(spec.randomization_repetitions, vector.size)
    )
    randomized = (signs * vector).mean(axis=1)
    observed = float(vector.mean())
    positive_p = float(
        (1 + np.count_nonzero(randomized >= observed)) / (spec.randomization_repetitions + 1)
    )
    loo = np.asarray(
        [(vector.sum() - vector[index]) / (vector.size - 1) for index in range(vector.size)]
        if vector.size > 1 else [observed],
        dtype=np.float64,
    )
    arrays[f"{key}__matrix_values"] = vector
    arrays[f"{key}__bootstrap"] = bootstrap
    arrays[f"{key}__sign_randomization"] = randomized
    arrays[f"{key}__leave_one_out"] = loo
    ci = np.quantile(bootstrap, (0.025, 0.975))
    return {
        "effect": observed,
        "ci95": [float(ci[0]), float(ci[1])],
        "positive_p": positive_p,
        "matrices": int(vector.size),
        "matrices_positive": int(np.count_nonzero(vector > 0.0)),
        "loo_positive": int(np.count_nonzero(loo > 0.0)),
    }


def _effect_series(
    frame: pd.DataFrame, metric: str, candidate: str, replicate: int
) -> pd.Series:
    selected = frame[
        (frame["candidate"] == candidate)
        & (frame["replicate"] == replicate)
        & frame["direction"].isin(DIRECTIONS)
        & (frame["information_eligible"] == 1)
    ]
    pivot = selected.pivot(index="matrix_id", columns="direction", values=metric)
    if not set(DIRECTIONS).issubset(pivot.columns):
        return pd.Series(dtype=float)
    return pivot["STABILIZE"] - pivot["DESTABILIZE"]


def analyze_batches(
    batches: Sequence[RescueBatch], spec: RescueSpec
) -> tuple[dict[str, Any], dict[str, pd.DataFrame], dict[str, NDArray]]:
    raw_rows = [dict(row) for batch in batches for row in batch.lineage_rows]
    arrays: dict[str, NDArray] = {}
    for row in raw_rows:
        for representation in REPRESENTATIONS:
            key = f"{representation}_partition_null_values"
            if key in row:
                array_key = (
                    f"partition_null__m{int(row['matrix_id']):04d}__{row['candidate']}__"
                    f"r{int(row['replicate'])}__{row['arm']}__{representation}"
                )
                arrays[array_key] = np.asarray(row.pop(key), dtype=np.float64)
    frame = pd.DataFrame(raw_rows)
    frame["candidate"] = frame["candidate"].astype(str).str.zfill(2)
    matrix_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    candidate_diagnostics: list[dict[str, Any]] = []
    cell_lookup: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    all_metrics = [
        (name, representation, f"{representation}_{suffix}")
        for name, suffix in CANDIDATE_MENU
        for representation in REPRESENTATIONS
    ]
    all_metrics.append(("HEREDITY", "final30", "inherited_31_60"))
    for name, representation, metric in all_metrics:
        local: list[dict[str, Any]] = []
        for candidate in CANDIDATES:
            for replicate in range(spec.replicates):
                series = _effect_series(frame, metric, candidate, replicate)
                for matrix_id, value in series.items():
                    matrix_rows.append(
                        {
                            "estimator": name,
                            "representation": representation,
                            "metric": metric,
                            "candidate": candidate,
                            "replicate": replicate,
                            "matrix_id": int(matrix_id),
                            "effect": float(value),
                        }
                    )
                key = f"{name}__{representation}__{candidate}__r{replicate}"
                summary = _summary(series.to_numpy(float), key, arrays, spec)
                summary.update(
                    {
                        "estimator": name,
                        "representation": representation,
                        "metric": metric,
                        "candidate": candidate,
                        "replicate": replicate,
                        "finite_paired_fraction": float(
                            np.count_nonzero(np.isfinite(series.to_numpy(float)))
                            / spec.matrices
                        ),
                    }
                )
                local.append(summary)
                cell_lookup[(name, representation, replicate, candidate)] = summary
        finite = [item for item in local if np.isfinite(item["positive_p"])]
        adjusted = holm_adjust([item["positive_p"] for item in finite]) if finite else []
        for item, value in zip(finite, adjusted, strict=True):
            item["holm_positive_p"] = float(value)
        cell_rows.extend(local)

    selected_estimator: str | None = None
    for name, suffix in CANDIDATE_MENU:
        primary = [
            cell_lookup[(name, "fable_style", replicate, candidate)]
            for candidate in CANDIDATES
            for replicate in range(spec.replicates)
        ]
        phase = [
            cell_lookup[(name, "phase_normalized", replicate, candidate)]
            for candidate in CANDIDATES
            for replicate in range(spec.replicates)
        ]
        generation = [
            cell_lookup[(name, "generational", replicate, candidate)]
            for candidate in CANDIDATES
            for replicate in range(spec.replicates)
        ]
        positive_primary = all(item["effect"] > 0.0 for item in primary)
        loo_gate = all(item["loo_positive"] >= 22 for item in primary)
        coverage_gate = all(item["finite_paired_fraction"] >= 0.95 for item in primary)
        phase_positive = sum(item["effect"] > 0.0 for item in phase)
        generation_positive = sum(item["effect"] > 0.0 for item in generation)
        scale_gate = max(phase_positive, generation_positive) >= 3
        qualifies = bool(positive_primary and loo_gate and coverage_gate and scale_gate)
        diagnostic = {
            "estimator": name,
            "metric_suffix": suffix,
            "primary_positive_all_four": positive_primary,
            "loo_at_least_22_all_four": loo_gate,
            "coverage_at_least_95_percent": coverage_gate,
            "phase_positive_cells": phase_positive,
            "generational_positive_cells": generation_positive,
            "secondary_scale_gate": scale_gate,
            "qualifies": qualifies,
        }
        candidate_diagnostics.append(diagnostic)
        if selected_estimator is None and qualifies:
            selected_estimator = name

    selection = {
        "selected_estimator": selected_estimator,
        "r0_development_candidate_selected": selected_estimator is not None,
        "r1_remains_locked_pending_human_authorization": True,
        "candidate_diagnostics": candidate_diagnostics,
    }
    metrics = {
        "format": "codex-ch5-phir-rescue-primary-metrics-v1",
        "registration_id": verify_registration()["registration_id"],
        "selection": selection,
        "cells": cell_rows,
        "gates": {
            "archive_reproduction_exact": True,
            "complete_exact_replay": True,
            "permutation_invariance": True,
            "r0_selected_candidate": selected_estimator is not None,
            "r1_authorized": False,
        },
        "decision_status": (
            "r0_candidate_selected_awaiting_human_review"
            if selected_estimator is not None
            else "r0_no_coherent_candidate_r1_locked"
        ),
        "prior_legacy_result_modified": False,
    }
    frames = {
        "lineages": frame,
        "matrix_effects": pd.DataFrame(matrix_rows),
        "cell_summaries": pd.DataFrame(cell_rows),
        "candidate_diagnostics": pd.DataFrame(candidate_diagnostics),
    }
    return metrics, frames, arrays


def _reports(metrics: dict[str, Any]) -> tuple[str, str]:
    cells = metrics["cells"]
    table_rows = []
    for item in cells:
        if item["representation"] != "fable_style" or item["estimator"] == "HEREDITY":
            continue
        table_rows.append(
            f"| {item['estimator']} | {item['candidate']} | {item['replicate']} | "
            f"{item['effect']:+.5f} [{item['ci95'][0]:+.5f}, {item['ci95'][1]:+.5f}] | "
            f"{item.get('holm_positive_p', float('nan')):.4g} | {item['loo_positive']}/24 |"
        )
    selected = metrics["selection"]["selected_estimator"]
    technical = "\n".join(
        (
            "# Chapter 5 Phi-r strongest-fair-test R0 report",
            "",
            f"Registration: `{metrics['registration_id']}`.",
            "",
            "R0 replayed selected sealed PAB24 trajectories. It generated no new scientific matrix or intervention outcome. Archived lineage and legacy-score reproduction and the complete second replay both passed exactly.",
            "",
            "## Primary final-30 development readings",
            "",
            "All effects are stabilization minus destabilization. R0 is a development/selection cohort; these intervals are descriptive and cannot confirm a new claim.",
            "",
            "| Estimator | Candidate | Replicate | Effect [95% matrix CI] | Holm p(+) | positive LOO |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
            *table_rows,
            "",
            "## Frozen selection",
            "",
            f"Selected estimator: **{selected if selected is not None else 'none'}**.",
            "",
            f"Decision status: `{metrics['decision_status']}`.",
            "",
            "R1 remains locked pending separate human authorization. No 48-matrix continuation is authorized.",
            "",
            "## Claim boundary",
            "",
            "This analysis does not overwrite the negative legacy raw nine-atom result. A selected candidate is a development hypothesis requiring a completely fresh 24-matrix confirmation. Nothing here establishes consciousness, agency, life, or metaphysical organization.",
            "",
        )
    )
    if selected is None:
        plain = (
            "None of the four theory-motivated information gauges behaved consistently enough across both simulators and replicates to justify a fresh confirmation. The heredity-control result remains valid, but this bounded attempt did not rescue a general Phi-r relationship."
        )
    else:
        plain = (
            f"One predefined information gauge, {selected}, moved consistently enough in the old 24-matrix material to earn a genuinely fresh test. This is a promising lead, not a discovery: it was selected on existing data and must now succeed unchanged on new matrices."
        )
    lay = "\n".join(
        (
            "# Lay summary — Phi-r strongest-fair-test R0",
            "",
            "We replayed the same molecular histories but used several more charitable ways of asking whether their remaining information became more integrated. The old negative reading remains on the record.",
            "",
            plain,
            "",
        )
    )
    return technical, lay


def _write_result(
    registration: dict[str, Any],
    spec: RescueSpec,
    generated: Sequence[RescueBatch],
    replayed: Sequence[RescueBatch],
    archive: dict[str, Any],
    replay: dict[str, Any],
    metrics: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    arrays: dict[str, NDArray],
    buckets: Sequence[int],
    cpu_seconds: float,
) -> None:
    technical, lay = _reports(metrics)
    with _atomic_destination(DEFAULT_OUTPUT) as destination:
        _atomic_json(destination / "primary_metrics.json", metrics)
        _atomic_json(destination / "archive_audit.json", archive)
        _atomic_json(destination / "replay_audit.json", replay)
        (destination / "SCIENTIFIC_REPORT.md").write_text(technical, encoding="utf-8")
        (destination / "LAY_SUMMARY.md").write_text(lay, encoding="utf-8")
        _atomic_json(
            destination / "claim_boundaries.json",
            {
                "supported_claims": ["R0 development selection only"],
                "failed_or_pending": [metrics["decision_status"]],
                "prior_results_modified": False,
                "prohibited_interpretations": protocol()["claim_boundary"],
            },
        )
        row_counts: dict[str, int] = {}
        for name, frame in frames.items():
            frame.to_csv(destination / f"{name}.csv.gz", index=False, compression="gzip")
            row_counts[name] = int(len(frame))
        np.savez_compressed(destination / "inference_arrays.npz", **arrays)
        library_arrays: dict[str, NDArray] = {}
        library_manifest: list[dict[str, Any]] = []
        for bucket in buckets:
            path = _library_path(bucket)
            with np.load(path) as library:
                for key in ("whole_mi", "revised"):
                    library_arrays[f"b{bucket:04d}_{key}"] = np.asarray(library[key])
            library_manifest.append(
                {"bucket": bucket, "sha256": sha256_file(path), "systems": spec.numit_systems}
            )
        np.savez_compressed(destination / "numit_libraries.npz", **library_arrays)
        _atomic_json(destination / "numit_library_manifest.json", library_manifest)
        readback_counts = {
            name: int(len(pd.read_csv(destination / f"{name}.csv.gz"))) for name in frames
        }
        readback = {
            "expected_row_counts": row_counts,
            "row_counts": readback_counts,
            "archive_exact": archive["passed"],
            "replay_exact": replay["complete_exact_replay"],
            "complete_readback_exact": bool(
                row_counts == readback_counts
                and archive["passed"]
                and replay["complete_exact_replay"]
            ),
        }
        if not readback["complete_readback_exact"]:
            raise AssertionError(f"R0 readback failed: {readback}")
        _atomic_json(destination / "readback_audit.json", readback)
        _atomic_json(
            destination / "manifest.json",
            {
                "format": RESULT_FORMAT,
                "registration_id": registration["registration_id"],
                "matrices": spec.matrices,
                "new_scientific_matrices": 0,
                "archive_reproduction_exact": True,
                "complete_exact_replay": True,
                "complete_readback_exact": True,
                "selected_estimator": metrics["selection"]["selected_estimator"],
                "r1_authorized": False,
                "no_48_matrix_run": True,
                "cpu_seconds": cpu_seconds,
                "row_counts": row_counts,
                "runtime": _runtime_versions(),
            },
        )
        write_checksums(destination)
    verify_checksums(DEFAULT_OUTPUT)


def validation_checks() -> dict[str, bool]:
    predecessor = verify_pab24_result()
    counts = np.asarray(
        [[4, 2, 1, 0], [3, 3, 1, 0], [2, 4, 1, 0], [1, 5, 1, 0]],
        dtype=np.int64,
    )
    beta = np.exp(np.random.default_rng(31).normal(-4.0, 1.0, size=(4, 4)))
    data, active = rank_gaussianize(close_all_clr(counts))
    pa, pb = beta_physical_partition(beta)
    first, second = active_partition(active, pa, pb)
    full = full_block_revised(data, first, second)
    null, null_values = matched_partition_null(
        data, int(first.size), 8, np.random.default_rng(7), full.revised
    )
    permutation = np.asarray([2, 0, 3, 1])
    pdata, pactive = rank_gaussianize(close_all_clr(counts[:, permutation]))
    ppa, ppb = beta_physical_partition(beta[np.ix_(permutation, permutation)])
    pfirst, psecond = active_partition(pactive, ppa, ppb)
    pfull = full_block_revised(pdata, pfirst, psecond)
    future_a = pab_future_seed(pab_scientific_spec(), "02", 3, 1)
    future_b = pab_future_seed(pab_scientific_spec(), "02", 3, 1)
    legacy_score_names = {
        "revised",
        "full_typeset",
        "macro_typeset",
        "normalized_full",
        "causation",
        "emergence",
        "synergy",
        "active_coordinates",
        "partition_a",
        "partition_b",
        "observations",
        "transitions",
        "digest",
        *(f"atom_{name}" for name in ATOM_NAMES),
    }
    amended_score_names = {
        "copula_observations",
        "copula_transitions",
        "copula_active_dimensions",
        "copula_physical_partition_a",
        "copula_physical_partition_b",
        "copula_public_raw",
        "copula_causation",
        "copula_emergence",
        "copula_synergy",
        *(f"copula_atom_{name}" for name in ATOM_NAMES),
    }
    checks = {
        "01_predecessor_exact_replay": bool(
            predecessor["pab24_complete_exact_replay"]
            and predecessor["archive_reproduction_exact"]
        ),
        "02_zero_new_scientific_matrices": protocol()["substrate"]["new_scientific_matrices"] == 0,
        "03_twenty_four_archived_matrices": scientific_spec().matrices == 24,
        "04_three_frozen_arms": len(ARMS) == 3,
        "05_candidate_order_frozen": tuple(name for name, _ in CANDIDATE_MENU) == (
            "NUMIT_MACRO", "PARTITION_NULL_FULL", "FULL_BLOCK_RAW", "COPULA_PUBLIC_RAW"
        ),
        "06_rank_gaussian_finite": np.isfinite(data).all(),
        "07_all_clr_coordinates_retained": data.shape[0] == counts.shape[1],
        "08_fixed_partition_complete": first.size + second.size == data.shape[0],
        "09_full_formula_identity": abs(
            full.revised - (full.whole_mi - full.aa_mi - full.bb_mi + full.double_redundancy)
        ) < 1e-12,
        "10_partition_null_draws_exact": null_values.shape == (8,) and null.draws == 8,
        "11_simultaneous_permutation_invariant": abs(full.revised - pfull.revised) < 1e-9,
        "12_future_stream_arm_free": future_a == future_b,
        "13_null_stream_separate": _partition_seed(scientific_spec(), 3, "02", 1, "fable_style") != future_a,
        "14_numit_bucket_half_up": _transition_bucket(24) == 32 and _transition_bucket(23) == 16,
        "15_complete_replay_required": protocol()["execution"]["complete_exact_replay"],
        "16_r1_locked": protocol()["r1_locked_pending_human_authorization"],
        "17_no_48": protocol()["no_48_matrix_run"],
        "18_source_files_exist": all((ROOT / name).is_file() for name in SOURCE_FILES),
        "19_external_assets_not_imported": not protocol()["external_fable_code_data_models_or_seeds_imported"],
        "20_new_score_namespace_disjoint": legacy_score_names.isdisjoint(amended_score_names),
        "21_checkpoint_module_canonical": (
            RescueSpec.__module__ == CANONICAL_MODULE
            and RescueBatch.__module__ == CANONICAL_MODULE
        ),
        "22_amendment_is_procedural_only": (
            protocol()["procedural_amendment"]["number"] == 1
            and not protocol()["procedural_amendment"]["scientific_contract_changed"]
            and not protocol()["procedural_amendment"]["failed_lineage_checkpoints_reused"]
        ),
    }
    return checks


def run_validation(output: Path = DEFAULT_VALIDATION) -> dict[str, Any]:
    checks = validation_checks()
    if not all(checks.values()):
        raise AssertionError([name for name, passed in checks.items() if not passed])
    payload = {
        "format": "codex-ch5-phir-rescue-validation-v1",
        "checks": checks,
        "check_count": len(checks),
        "passed_count": sum(checks.values()),
        "all_checks_passed": True,
        "source_hashes": _source_hashes(),
        "new_scientific_matrices": 0,
    }
    with _atomic_destination(output) as destination:
        _atomic_json(destination / "validation.json", payload)
        write_checksums(destination)
    verify_checksums(output)
    print(f"Phi-r rescue validation passed: {len(checks)}/{len(checks)}", flush=True)
    return payload


def register_program() -> dict[str, Any]:
    verify_checksums(DEFAULT_VALIDATION)
    validation = json.loads((DEFAULT_VALIDATION / "validation.json").read_text(encoding="utf-8"))
    if validation["source_hashes"] != _source_hashes():
        raise ValueError("source changed after Phi-r rescue validation")
    for forbidden in (DEFAULT_REGISTRATION, DEFAULT_SMOKE, DEFAULT_OUTPUT, DEFAULT_WORK):
        if forbidden.exists():
            raise FileExistsError(f"pre-R0 artifact already exists: {forbidden}")
    predecessor = verify_pab24_result()
    body: dict[str, Any] = {
        "format": REGISTRATION_FORMAT,
        "protocol": protocol(),
        "protocol_id": protocol()["protocol_id"],
        "source_hashes": _source_hashes(),
        "source_tree_sha256": _canonical_digest(_source_hashes()),
        "seed_registry": SEED_DOMAINS,
        "predecessor_registration_id": predecessor["registration_id"],
        "predecessor_manifest_sha256": sha256_file(PAB24_OUTPUT / "manifest.json"),
        "predecessor_selected_edits_sha256": sha256_file(PAB24_OUTPUT / "selected_edits.csv.gz"),
        "procedural_amendment": {
            "number": 1,
            "document_sha256": sha256_file(ROOT / AMENDMENT_DOCUMENT),
            "supersedes_registration_id": ORIGINAL_REGISTRATION_ID,
        },
        "new_scientific_matrices_at_registration": 0,
        "numeric_environment": _runtime_versions(),
    }
    body["registration_id"] = _canonical_digest(_json_ready(body))
    with _atomic_destination(DEFAULT_REGISTRATION) as destination:
        shutil.copy2(ROOT / DOCUMENT, destination / "preregistration.md")
        shutil.copy2(
            ROOT / AMENDMENT_DOCUMENT,
            destination / "procedural_amendment_001.md",
        )
        shutil.copy2(DEFAULT_VALIDATION / "validation.json", destination / "validation.json")
        _atomic_json(destination / "protocol.json", protocol())
        _atomic_json(destination / "seed_registry.json", SEED_DOMAINS)
        _atomic_json(destination / "registration.json", body)
        write_checksums(destination)
    verify_checksums(DEFAULT_REGISTRATION)
    _append_ledger(
        f"<!-- phir-rescue-registration-{body['registration_id']} -->",
        (
            "## Chapter 5 Phi-r strongest-fair-test R0 amended and re-registered",
            "",
            f"- Registration: `{body['registration_id']}`.",
            f"- This procedural registration supersedes `{ORIGINAL_REGISTRATION_ID}` after a pre-analysis field-namespace/checkpoint-portability failure.",
            "- R0 replays selected sealed PAB24 lineages and generates no new scientific matrix or intervention outcome.",
            "- The legacy negative raw nine-atom result remains immutable.",
            "- R1 is locked pending R0 selection and separate human authorization; no 48-matrix run is authorized.",
        ),
    )
    print(f"Phi-r rescue R0 registered: {body['registration_id']}", flush=True)
    return body


def verify_registration(directory: Path = DEFAULT_REGISTRATION) -> dict[str, Any]:
    verify_checksums(directory)
    registration = json.loads((directory / "registration.json").read_text(encoding="utf-8"))
    body = dict(registration)
    observed = body.pop("registration_id")
    if registration["format"] != REGISTRATION_FORMAT:
        raise ValueError("unsupported Phi-r rescue registration")
    if _canonical_digest(_json_ready(body)) != observed:
        raise ValueError("Phi-r rescue registration identity failed")
    if registration["source_hashes"] != _source_hashes():
        raise ValueError("Phi-r rescue source tree changed")
    if registration["protocol"] != _json_ready(protocol()):
        raise ValueError("Phi-r rescue protocol changed")
    if sha256_file(PAB24_OUTPUT / "manifest.json") != registration["predecessor_manifest_sha256"]:
        raise ValueError("PAB24 predecessor changed")
    if sha256_file(PAB24_OUTPUT / "selected_edits.csv.gz") != registration["predecessor_selected_edits_sha256"]:
        raise ValueError("PAB24 selected edits changed")
    return registration


def run_smoke(output: Path = DEFAULT_SMOKE) -> dict[str, Any]:
    registration = verify_registration()
    spec = smoke_spec()
    rng = np.random.default_rng(81)
    counts = rng.poisson(2.0, size=(48, 12))
    counts[:, 0] += np.arange(48) % 4
    beta = np.exp(rng.normal(-4.0, 1.0, size=(12, 12)))
    data, active = rank_gaussianize(close_all_clr(counts))
    pa, pb = beta_physical_partition(beta)
    first, second = active_partition(active, pa, pb)
    macro = macro_phi_score(data, first, second)
    full = full_block_revised(data, first, second)
    null_a, values_a = matched_partition_null(data, int(first.size), 8, np.random.default_rng(5), full.revised)
    null_b, values_b = matched_partition_null(data, int(first.size), 8, np.random.default_rng(5), full.revised)
    library = generate_numit_library(16, 32, np.random.default_rng(6), burn=16)
    calibration = calibrate_numit(macro.revised, macro.whole_mi, library, neighbors=16)
    payload = {
        "format": "codex-ch5-phir-rescue-smoke-v1",
        "registration_id": registration["registration_id"],
        "artificial_non_scientific_fixture": True,
        "rank_copula_finite": bool(np.isfinite(data).all()),
        "macro_score_finite": bool(np.isfinite(macro.revised)),
        "full_score_finite": bool(np.isfinite(full.revised)),
        "partition_null_deterministic": bool(null_a == null_b and np.array_equal(values_a, values_b)),
        "numit_library_shape": list(library["revised"].shape),
        "numit_calibration_exercised": bool(calibration.neighbors == 16),
        "scientific_effect_sizes_disclosed": False,
        "new_scientific_matrices": 0,
    }
    required = (
        "rank_copula_finite", "macro_score_finite", "full_score_finite",
        "partition_null_deterministic", "numit_calibration_exercised",
    )
    if not all(payload[name] for name in required):
        raise AssertionError(f"Phi-r rescue smoke failed: {payload}")
    with _atomic_destination(output) as destination:
        _atomic_json(destination / "smoke.json", payload)
        write_checksums(destination)
    verify_checksums(output)
    print("Phi-r rescue non-scientific smoke passed", flush=True)
    return payload


def _prepare_work(registration_id: str, spec: RescueSpec) -> None:
    if DEFAULT_OUTPUT.exists():
        raise FileExistsError(f"completed R0 output exists: {DEFAULT_OUTPUT}")
    if shutil.disk_usage(ROOT).free < MINIMUM_FREE_DISK_BYTES:
        raise RuntimeError("Phi-r rescue R0 requires at least 1.5 GB free")
    DEFAULT_WORK.mkdir(parents=True, exist_ok=True)
    contract = {
        "format": "codex-ch5-phir-rescue-work-v1",
        "registration_id": registration_id,
        "protocol_id": protocol()["protocol_id"],
        "spec": asdict(spec),
    }
    path = DEFAULT_WORK / "campaign_contract.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != _json_ready(contract):
            raise ValueError("Phi-r rescue work contract changed")
    else:
        _atomic_json(path, contract)


def run_scientific(workers: int = min(os.cpu_count() or 1, 12)) -> dict[str, Any]:
    registration = verify_registration()
    verify_checksums(DEFAULT_SMOKE)
    spec = scientific_spec()
    _prepare_work(registration["registration_id"], spec)
    buckets, library_cpu = _prepare_libraries(spec, workers)
    generated, generation_cpu = _run_checkpointed(
        spec,
        registration["registration_id"],
        DEFAULT_WORK / "generated",
        "R0 generation",
        workers,
        library_cpu,
    )
    archive = _archive_audit(generated)
    _atomic_json(DEFAULT_WORK / "archive_audit.json", archive)
    if not archive["passed"]:
        raise AssertionError(f"R0 archived replay gate failed: {archive}")
    replayed, total_cpu = _run_checkpointed(
        spec,
        registration["registration_id"],
        DEFAULT_WORK / "replay",
        "R0 replay",
        workers,
        generation_cpu,
    )
    replay = _replay_audit(generated, replayed)
    if not replay["complete_exact_replay"]:
        raise AssertionError("R0 complete replay failed")
    _write_status("analysis", 0, 1, cpu_seconds=total_cpu)
    metrics, frames, arrays = analyze_batches(generated, spec)
    _write_result(
        registration, spec, generated, replayed, archive, replay, metrics, frames,
        arrays, buckets, total_cpu,
    )
    _write_status(
        "awaiting_human_review", 1, 1, cpu_seconds=total_cpu,
        selected_estimator=metrics["selection"]["selected_estimator"],
        output=str(DEFAULT_OUTPUT),
    )
    _append_ledger(
        f"<!-- phir-rescue-result-{sha256_file(DEFAULT_OUTPUT / 'manifest.json')} -->",
        (
            "## Chapter 5 Phi-r strongest-fair-test R0 completed",
            "",
            f"- Result: `{DEFAULT_OUTPUT.relative_to(ROOT)}`.",
            "- Selected sealed PAB24 trajectories and all legacy scores were reproduced before new remeasurement; complete replay passed.",
            f"- Development-selected estimator: `{metrics['selection']['selected_estimator']}`.",
            f"- Decision status: `{metrics['decision_status']}`.",
            "- The prior raw nine-atom result remains unchanged; R1 and every 48-matrix run remain locked.",
        ),
    )
    return metrics


def status_payload() -> dict[str, Any]:
    output: dict[str, Any] = {
        "validation": DEFAULT_VALIDATION.exists(),
        "registration": DEFAULT_REGISTRATION.exists(),
        "smoke": DEFAULT_SMOKE.exists(),
        "complete": DEFAULT_OUTPUT.exists(),
        "service": SERVICE_NAME,
        "free_disk_bytes": shutil.disk_usage(ROOT).free,
        "minimum_free_disk_bytes": MINIMUM_FREE_DISK_BYTES,
        "cpu_budget_seconds": CPU_BUDGET_SECONDS,
        "r1_locked": True,
        "no_48_matrix_run": True,
    }
    status = DEFAULT_WORK / "campaign_status.json"
    if status.exists():
        output["campaign"] = json.loads(status.read_text(encoding="utf-8"))
    launch = DEFAULT_WORK / "detached_launch.json"
    if launch.exists():
        output["detached_launch"] = json.loads(launch.read_text(encoding="utf-8"))
    return output


def launch_detached(workers: int = min(os.cpu_count() or 1, 12)) -> dict[str, Any]:
    registration = verify_registration()
    verify_checksums(DEFAULT_SMOKE)
    if DEFAULT_OUTPUT.exists():
        raise FileExistsError(f"completed output exists: {DEFAULT_OUTPUT}")
    if shutil.disk_usage(ROOT).free < MINIMUM_FREE_DISK_BYTES:
        raise RuntimeError("detached R0 launch refused below the sealed disk floor")
    DEFAULT_WORK.mkdir(parents=True, exist_ok=True)
    command = [
        "systemd-run", "--user", f"--unit={SERVICE_NAME}", "--collect",
        "--property", f"WorkingDirectory={ROOT}",
        "--property", f"StandardOutput=append:{DEFAULT_LOG}",
        "--property", f"StandardError=append:{DEFAULT_LOG}",
        sys.executable, "-m", "plastic_heredity.phir_rescue", "run",
        "--workers", str(min(workers, 12)),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = {
        "format": "codex-ch5-phir-rescue-detached-launch-v1",
        "registration_id": registration["registration_id"],
        "service": SERVICE_NAME,
        "workers": min(workers, 12),
        "launched_at_unix": time.time(),
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }
    _atomic_json(DEFAULT_WORK / "detached_launch.json", payload)
    return payload


def verify_result() -> dict[str, Any]:
    verify_checksums(DEFAULT_OUTPUT)
    manifest = json.loads((DEFAULT_OUTPUT / "manifest.json").read_text(encoding="utf-8"))
    if not all(
        manifest[name]
        for name in ("archive_reproduction_exact", "complete_exact_replay", "complete_readback_exact")
    ):
        raise ValueError("Phi-r rescue R0 result integrity failed")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate")
    subparsers.add_parser("register")
    subparsers.add_parser("smoke")
    run = subparsers.add_parser("run")
    run.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 12))
    launch = subparsers.add_parser("launch")
    launch.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 12))
    subparsers.add_parser("status")
    subparsers.add_parser("verify")
    fixture = subparsers.add_parser("_checkpoint-fixture", help=argparse.SUPPRESS)
    fixture.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    if arguments.command == "validate":
        run_validation()
    elif arguments.command == "register":
        register_program()
    elif arguments.command == "smoke":
        run_smoke()
    elif arguments.command == "run":
        run_scientific(arguments.workers)
    elif arguments.command == "launch":
        print(json.dumps(launch_detached(arguments.workers), sort_keys=True, indent=2))
    elif arguments.command == "status":
        print(json.dumps(status_payload(), sort_keys=True, indent=2))
    elif arguments.command == "verify":
        print(json.dumps(verify_result(), sort_keys=True, indent=2))
    elif arguments.command == "_checkpoint-fixture":
        batch = _write_checkpoint_fixture(arguments.output)
        print(batch.scientific_digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
