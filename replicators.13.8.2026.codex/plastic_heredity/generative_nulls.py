"""Prospective GN1 decomposition of catalytic context and fission geometry.

This module is additive.  It does not modify the sealed simulator, F12 process,
strict-regime endpoint, frozen predictor, or any completed intervention result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.special import expit
from threadpoolctl import threadpool_limits

from .config import CANDIDATES, ExperimentConfig, GardConfig, SimulationContract
from .experiment import StateCase, _json_ready, _runtime_manifest
from .features import history_features, state_graph_features
from .intervention_core import (
    FrozenFullPredictor,
    MolecularEdit,
    apply_molecular_edit,
    enumerate_legal_edits,
)
from .intervention_outgoing_rule import select_outgoing_rule_edits
from .mechanistic import (
    _atomic_destination,
    _canonical_digest,
    sha256_file,
    verify_checksums,
    write_checksums,
)
from .mechanistic_features import H8_INDICES
from .mechanistic_metrics import holm_adjust
from .mechanistic_v2_features import FEATURE_NAMES, STATE_ONLY_INDICES
from .mechanistic_v2_models import fit_block_transform, fit_linear
from .metrics import centered_spearman, spearman
from .processes import evaluate_process
from .regime_prediction_endpoints import evaluate_rich_regime
from .seeds import derive_seed
from .simulator import (
    FissionRecord,
    SimulationError,
    Snapshot,
    _fission,
    cosine_similarity,
    generate_beta,
    generate_initial_composition,
    simulate_future_absorbing,
    simulate_lineage,
)


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results"
DOCUMENT = "CODEX_GENERATIVE_NULLS_PREREGISTRATION.md"
LEDGER = ROOT / "GENERATIVE_NULL_RESULTS_LEDGER.md"

DEFAULT_VALIDATION = RESULT_ROOT / "generative_null_validation"
DEFAULT_REGISTRATION = RESULT_ROOT / "generative_null_registration"
DEFAULT_SMOKE = RESULT_ROOT / "generative_null_smoke"
DEFAULT_OUTPUT = RESULT_ROOT / "generative_null_decomposition"
DEFAULT_WORK = RESULT_ROOT / ".generative_null_work"
DEFAULT_LOG = RESULT_ROOT / "generative_null_decomposition.log"

PROGRAM_FORMAT = "codex-generative-null-decomposition-v1"
VALIDATION_FORMAT = "codex-generative-null-validation-v1"
REGISTRATION_FORMAT = "codex-generative-null-registration-v1"
RESULT_FORMAT = "codex-generative-null-result-v1"
CHECKPOINT_FORMAT = "codex-generative-null-checkpoint-v1"
STATUS_FORMAT = "codex-generative-null-status-v1"
LABEL = "GN1_CATALYTIC_GEOMETRY_V1"

MATRICES = 96
LANDMARKS = (20, 35, 50, 65, 80)
F32_BRANCHES = 64
INTERVENTION_BRANCHES = 32
F12_HORIZON = 12
F32_HORIZON = 32
BOOTSTRAP_REPETITIONS = 4_096
RANDOMIZATION_REPETITIONS = 4_096
TOST_MARGIN = 0.025
FEATURE_ENVELOPE_Z = 5.0
MINIMUM_FREE_DISK_BYTES = 2_000_000_000

MECHANISMS = (
    "NATURAL_GARD",
    "HOMOGENEOUS_GENERATIVE",
    "COUPLING_DERANGED",
    "FISSION_ONLY_GENERATIVE",
)
INTERVENTION_ARMS = (
    "NOOP",
    "SOURCE_RULE_UP",
    "SOURCE_RULE_DOWN",
    "RANDOM",
)
STRICT_ENDPOINTS = ("primary_all8", "secondary_first5", "secondary_centroid")

HOMOGENEOUS_VALUE = float(np.exp(-4.0 + 4.0**2 / 2.0))
STATE_RIDGE = {"02": 0.1, "03": 0.01}
EXPECTED_FROZEN_MODEL_SHA256 = (
    "9b3305a7fed11f432651926d34903443e9413ed299c5d0f1056a0b5fde9990af"
)
UPSTREAM_MODEL = (
    ROOT
    / "results_intervention_replication"
    / "cr1_confirmation_registration"
    / "frozen_full_predictor.npz"
)

SOURCE_FILES = (
    DOCUMENT,
    "plastic_heredity/config.py",
    "plastic_heredity/features.py",
    "plastic_heredity/generative_nulls.py",
    "plastic_heredity/intervention_core.py",
    "plastic_heredity/intervention_outgoing_rule.py",
    "plastic_heredity/mechanistic_v2_features.py",
    "plastic_heredity/mechanistic_v2_models.py",
    "plastic_heredity/processes.py",
    "plastic_heredity/regime_prediction_endpoints.py",
    "plastic_heredity/seeds.py",
    "plastic_heredity/simulator.py",
    "tests/test_generative_nulls.py",
    "pyproject.toml",
    "requirements-lock.txt",
)


def _seed_value(name: str) -> str:
    return hashlib.sha256(
        f"codex-clean-room-generative-nulls-v1::{name}".encode("utf-8")
    ).hexdigest()


SEEDS = {
    name: _seed_value(name)
    for name in (
        "matrix_generation",
        "initial_composition",
        "main_trajectory",
        "coupling_derangement",
        "random_edit_selection",
        "future_simulation",
        "bootstrap",
        "randomization",
        "validation",
        "smoke",
        "replay",
    )
}


@dataclass(frozen=True)
class NullCase:
    state_id: str
    mechanism: str
    candidate: str
    matrix_id: int
    landmark: int
    source_beta: FloatArray
    active_beta: FloatArray
    snapshot: Snapshot
    main_attempt: int
    coupling_permutation: IntArray


@dataclass(frozen=True)
class NullBatch:
    state_id: str
    state_digest: str
    f12_joint: NDArray[np.int8]
    f12_break: NDArray[np.int8]
    f12_inherited_count: NDArray[np.int8]
    f12_survival: NDArray[np.int8]
    f12_growth_updates: NDArray[np.int32]
    f12_final_entropy: FloatArray
    f12_final_occupied: NDArray[np.int16]
    strict_targets: NDArray[np.int8]
    strict_break: NDArray[np.int8]
    strict_run8: NDArray[np.int8]
    f32_survival: NDArray[np.int8]
    intervention_joint: NDArray[np.int8]
    intervention_break: NDArray[np.int8]
    intervention_inherited_count: NDArray[np.int8]
    intervention_survival: NDArray[np.int8]
    edits: tuple[MolecularEdit | None, ...]


def _source_hashes() -> dict[str, str]:
    return {name: sha256_file(ROOT / name) for name in SOURCE_FILES}


def _homogeneous_beta(config: GardConfig) -> FloatArray:
    return np.full(
        (config.n_types, config.n_types), HOMOGENEOUS_VALUE, dtype=np.float64
    )


def fixed_point_free_permutation(size: int, rng: np.random.Generator) -> IntArray:
    """Return one uniformly generated Sattolo cycle, hence no fixed points."""

    if size < 2:
        raise ValueError("a fixed-point-free cycle requires at least two labels")
    values = np.arange(size, dtype=np.int64)
    for index in range(size - 1, 0, -1):
        other = int(rng.integers(0, index))
        values[index], values[other] = values[other], values[index]
    if np.any(values == np.arange(size)) or np.unique(values).size != size:
        raise AssertionError("Sattolo permutation contract failed")
    return values


def derange_beta(beta: FloatArray, permutation: IntArray) -> FloatArray:
    matrix = np.asarray(beta, dtype=np.float64)
    order = np.asarray(permutation, dtype=np.int64)
    if matrix.shape != (order.size, order.size):
        raise ValueError("beta and permutation dimensions differ")
    if np.unique(order).size != order.size or np.any(order == np.arange(order.size)):
        raise ValueError("coupling permutation must be fixed-point-free")
    return matrix[np.ix_(order, order)].copy()


def _entropy(composition: NDArray) -> float:
    values = np.asarray(composition, dtype=np.float64)
    mass = float(values.sum())
    if mass <= 0.0:
        return 0.0
    probability = values[values > 0.0] / mass
    return float(-np.sum(probability * np.log(probability)))


def _fission_only_advance(
    composition: NDArray,
    config: GardConfig,
    contract: SimulationContract,
    rng: np.random.Generator,
) -> FissionRecord:
    current = np.asarray(composition, dtype=np.int64).copy()
    mass = int(current.sum())
    if mass <= 0:
        raise SimulationError("fission-only assembly became extinct")
    if mass > config.n_max:
        raise SimulationError("fission-only launch exceeds parent mass")
    arrivals = config.n_max - mass
    if arrivals:
        current += np.asarray(
            rng.multinomial(
                arrivals,
                np.full(config.n_types, 1.0 / config.n_types, dtype=np.float64),
            ),
            dtype=np.int64,
        )
    if int(current.sum()) != config.n_max:
        raise AssertionError("fission-only parent mass changed")
    daughter = _fission(current, config, contract, rng)
    return FissionRecord(
        parent=current,
        daughter=daughter,
        h=cosine_similarity(current, daughter),
        growth_steps=arrivals,
    )


def simulate_fission_only_lineage(
    initial: NDArray,
    config: GardConfig,
    contract: SimulationContract,
    rng: np.random.Generator,
) -> list[Snapshot]:
    current = np.asarray(initial, dtype=np.int64).copy()
    inheritance: list[bool] = []
    boundary_h: list[float] = []
    snapshots: list[Snapshot] = []
    cumulative = 0
    for generation in range(1, config.generations + 1):
        record = _fission_only_advance(current, config, contract, rng)
        cumulative += record.growth_steps
        boundary_h.append(record.h)
        inheritance.append(record.h > config.inheritance_threshold)
        current = record.daughter
        snapshots.append(
            Snapshot(
                composition=current.copy(),
                generation=generation,
                inheritance=tuple(inheritance),
                boundary_h=tuple(boundary_h),
                previous_growth_steps=record.growth_steps,
                cumulative_growth_steps=cumulative,
            )
        )
    return snapshots


def simulate_fission_only_future(
    snapshot: Snapshot,
    config: GardConfig,
    contract: SimulationContract,
    horizon: int,
    rng: np.random.Generator,
) -> tuple[list[FissionRecord], bool]:
    current = np.asarray(snapshot.composition, dtype=np.int64).copy()
    records: list[FissionRecord] = []
    for _ in range(horizon):
        try:
            record = _fission_only_advance(current, config, contract, rng)
        except SimulationError:
            return records, False
        records.append(record)
        current = record.daughter
    return records, True


def _lineage_with_retry(
    initial: IntArray,
    beta: FloatArray,
    config: GardConfig,
    contract: SimulationContract,
    candidate: str,
    matrix_id: int,
) -> tuple[list[Snapshot], int]:
    for attempt in range(100):
        rng = np.random.default_rng(
            derive_seed(
                SEEDS["main_trajectory"],
                f"{LABEL}.main",
                candidate,
                matrix_id,
                attempt,
            )
        )
        try:
            return simulate_lineage(initial, beta, config, contract, rng), attempt
        except SimulationError:
            continue
    raise SimulationError(
        f"GN1 main trajectory failed for candidate {candidate}, matrix {matrix_id}"
    )


def build_cases(
    matrices: int = MATRICES,
    landmarks: tuple[int, ...] = LANDMARKS,
    config: GardConfig | None = None,
) -> list[NullCase]:
    current_config = config or GardConfig()
    homogeneous = _homogeneous_beta(current_config)
    cases: list[NullCase] = []
    for matrix_id in range(matrices):
        beta = generate_beta(
            current_config,
            np.random.default_rng(
                derive_seed(SEEDS["matrix_generation"], LABEL, matrix_id)
            ),
        )
        initial = generate_initial_composition(
            current_config,
            np.random.default_rng(
                derive_seed(SEEDS["initial_composition"], LABEL, matrix_id)
            ),
        )
        permutation = fixed_point_free_permutation(
            current_config.n_types,
            np.random.default_rng(
                derive_seed(SEEDS["coupling_derangement"], LABEL, matrix_id)
            ),
        )
        shuffled = derange_beta(beta, permutation)
        for candidate, contract in CANDIDATES.items():
            natural, natural_attempt = _lineage_with_retry(
                initial,
                beta,
                current_config,
                contract,
                candidate,
                matrix_id,
            )
            homogeneous_lineage, homogeneous_attempt = _lineage_with_retry(
                initial,
                homogeneous,
                current_config,
                contract,
                candidate,
                matrix_id,
            )
            fission_rng = np.random.default_rng(
                derive_seed(
                    SEEDS["main_trajectory"],
                    f"{LABEL}.main",
                    candidate,
                    matrix_id,
                    0,
                )
            )
            fission_lineage = simulate_fission_only_lineage(
                initial, current_config, contract, fission_rng
            )
            paths = {
                "NATURAL_GARD": (natural, beta, natural_attempt),
                "HOMOGENEOUS_GENERATIVE": (
                    homogeneous_lineage,
                    homogeneous,
                    homogeneous_attempt,
                ),
                "COUPLING_DERANGED": (natural, shuffled, natural_attempt),
                "FISSION_ONLY_GENERATIVE": (fission_lineage, homogeneous, 0),
            }
            for mechanism in MECHANISMS:
                lineage, active_beta, attempt = paths[mechanism]
                by_generation = {item.generation: item for item in lineage}
                for landmark in landmarks:
                    snapshot = by_generation[landmark]
                    cases.append(
                        NullCase(
                            state_id=(
                                f"gn1-{mechanism.lower()}-c{candidate}-"
                                f"m{matrix_id:03d}-g{landmark:03d}"
                            ),
                            mechanism=mechanism,
                            candidate=candidate,
                            matrix_id=matrix_id,
                            landmark=landmark,
                            source_beta=beta,
                            active_beta=active_beta,
                            snapshot=snapshot,
                            main_attempt=attempt,
                            coupling_permutation=permutation,
                        )
                    )
    return cases


def _edited_snapshot(snapshot: Snapshot, edit: MolecularEdit | None) -> Snapshot:
    if edit is None:
        return snapshot
    return Snapshot(
        composition=apply_molecular_edit(snapshot.composition, edit),
        generation=snapshot.generation,
        inheritance=snapshot.inheritance,
        boundary_h=snapshot.boundary_h,
        previous_growth_steps=snapshot.previous_growth_steps,
        cumulative_growth_steps=snapshot.cumulative_growth_steps,
    )


def _simulate_case_future(
    case: NullCase,
    config: GardConfig,
    horizon: int,
    rng: np.random.Generator,
    edit: MolecularEdit | None = None,
) -> tuple[list[FissionRecord], bool]:
    snapshot = _edited_snapshot(case.snapshot, edit)
    contract = CANDIDATES[case.candidate]
    if case.mechanism == "FISSION_ONLY_GENERATIVE":
        return simulate_fission_only_future(snapshot, config, contract, horizon, rng)
    return simulate_future_absorbing(
        snapshot, case.active_beta, config, contract, horizon, rng
    )


def _future_seed(case: NullCase, branch: int) -> int:
    # Mechanism and arm are intentionally absent: common random streams.
    return derive_seed(
        SEEDS["future_simulation"],
        f"{LABEL}.future",
        case.candidate,
        case.matrix_id,
        case.landmark,
        branch,
    )


def _random_edit(case: NullCase) -> MolecularEdit:
    legal = enumerate_legal_edits(case.snapshot.composition)
    rng = np.random.default_rng(
        derive_seed(
            SEEDS["random_edit_selection"],
            f"{LABEL}.random_edit",
            case.mechanism,
            case.candidate,
            case.matrix_id,
            case.landmark,
        )
    )
    return legal[int(rng.integers(0, len(legal)))]


def _state_digest(case: NullCase) -> str:
    digest = hashlib.sha256()
    for value in (
        case.state_id,
        case.mechanism,
        case.candidate,
        str(case.matrix_id),
        str(case.landmark),
        str(case.main_attempt),
    ):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    for value in (
        case.source_beta,
        case.active_beta,
        case.snapshot.composition,
        np.asarray(case.snapshot.inheritance, dtype=np.int8),
        np.asarray(case.snapshot.boundary_h, dtype=np.float64),
        case.coupling_permutation,
    ):
        digest.update(np.ascontiguousarray(value).tobytes())
    digest.update(np.asarray(
        [
            case.snapshot.generation,
            case.snapshot.previous_growth_steps,
            case.snapshot.cumulative_growth_steps,
        ], dtype=np.int64
    ).tobytes())
    return digest.hexdigest()


def _record_digest(records: Iterable[FissionRecord], completed: bool) -> str:
    digest = hashlib.sha256()
    digest.update(bytes([int(completed)]))
    for record in records:
        digest.update(np.ascontiguousarray(record.parent).tobytes())
        digest.update(np.ascontiguousarray(record.daughter).tobytes())
        digest.update(np.asarray([record.h], dtype=np.float64).tobytes())
        digest.update(np.asarray([record.growth_steps], dtype=np.int64).tobytes())
    return digest.hexdigest()


def _summarize_f12(
    records: list[FissionRecord], completed_f32: bool
) -> tuple[int, int, int, int, int, float, int]:
    selected = records[:F12_HORIZON]
    outcome = evaluate_process(selected)
    completed = len(selected) == F12_HORIZON
    inherited = sum(record.h > 0.9 for record in selected)
    growth = sum(record.growth_steps for record in selected)
    final = selected[-1].daughter if selected else np.zeros(100, dtype=np.int64)
    return (
        int(outcome.joint_break_run3),
        int(outcome.break_event),
        int(inherited),
        int(completed),
        int(growth),
        _entropy(final),
        int(np.count_nonzero(final)),
    )


def _worker(arguments: tuple[NullCase, GardConfig]) -> NullBatch:
    case, config = arguments
    limiter = threadpool_limits(limits=1)
    try:
        rules = select_outgoing_rule_edits(
            case.snapshot.composition, case.source_beta
        )
        random_edit = _random_edit(case)
        edits: tuple[MolecularEdit | None, ...] = (
            None,
            rules["RULE_UP"],
            rules["RULE_DOWN"],
            random_edit,
        )
        f12_joint = np.empty(F32_BRANCHES, dtype=np.int8)
        f12_break = np.empty(F32_BRANCHES, dtype=np.int8)
        f12_inherited = np.empty(F32_BRANCHES, dtype=np.int8)
        f12_survival = np.empty(F32_BRANCHES, dtype=np.int8)
        f12_growth = np.empty(F32_BRANCHES, dtype=np.int32)
        f12_entropy = np.empty(F32_BRANCHES, dtype=np.float64)
        f12_occupied = np.empty(F32_BRANCHES, dtype=np.int16)
        strict_targets = np.empty((F32_BRANCHES, 3), dtype=np.int8)
        strict_break = np.empty(F32_BRANCHES, dtype=np.int8)
        strict_run8 = np.empty(F32_BRANCHES, dtype=np.int8)
        f32_survival = np.empty(F32_BRANCHES, dtype=np.int8)
        noop_records: list[list[FissionRecord]] = []
        noop_completed: list[bool] = []
        for branch in range(F32_BRANCHES):
            records, completed = _simulate_case_future(
                case,
                config,
                F32_HORIZON,
                np.random.default_rng(_future_seed(case, branch)),
            )
            noop_records.append(records)
            noop_completed.append(completed)
            summary = _summarize_f12(records, completed)
            (
                f12_joint[branch],
                f12_break[branch],
                f12_inherited[branch],
                f12_survival[branch],
                f12_growth[branch],
                f12_entropy[branch],
                f12_occupied[branch],
            ) = summary
            strict = evaluate_rich_regime(records)
            strict_targets[branch] = np.asarray(strict.targets, dtype=np.int8)
            strict_break[branch] = int(strict.break_event)
            strict_run8[branch] = int(strict.any_run8_after_break)
            f32_survival[branch] = int(completed)

        intervention_joint = np.empty(
            (len(INTERVENTION_ARMS), INTERVENTION_BRANCHES), dtype=np.int8
        )
        intervention_break = np.empty_like(intervention_joint)
        intervention_inherited = np.empty_like(intervention_joint)
        intervention_survival = np.empty_like(intervention_joint)
        for branch in range(INTERVENTION_BRANCHES):
            noop_summary = _summarize_f12(
                noop_records[branch], noop_completed[branch]
            )
            intervention_joint[0, branch] = noop_summary[0]
            intervention_break[0, branch] = noop_summary[1]
            intervention_inherited[0, branch] = noop_summary[2]
            intervention_survival[0, branch] = noop_summary[3]
            for arm_index in range(1, len(INTERVENTION_ARMS)):
                records, completed = _simulate_case_future(
                    case,
                    config,
                    F12_HORIZON,
                    np.random.default_rng(_future_seed(case, branch)),
                    edits[arm_index],
                )
                outcome = evaluate_process(records)
                intervention_joint[arm_index, branch] = int(
                    outcome.joint_break_run3
                )
                intervention_break[arm_index, branch] = int(outcome.break_event)
                intervention_inherited[arm_index, branch] = sum(
                    record.h > 0.9 for record in records
                )
                intervention_survival[arm_index, branch] = int(completed)
        return NullBatch(
            state_id=case.state_id,
            state_digest=_state_digest(case),
            f12_joint=f12_joint,
            f12_break=f12_break,
            f12_inherited_count=f12_inherited,
            f12_survival=f12_survival,
            f12_growth_updates=f12_growth,
            f12_final_entropy=f12_entropy,
            f12_final_occupied=f12_occupied,
            strict_targets=strict_targets,
            strict_break=strict_break,
            strict_run8=strict_run8,
            f32_survival=f32_survival,
            intervention_joint=intervention_joint,
            intervention_break=intervention_break,
            intervention_inherited_count=intervention_inherited,
            intervention_survival=intervention_survival,
            edits=edits,
        )
    finally:
        limiter.restore_original_limits()


def _batch_digest(batch: NullBatch) -> str:
    digest = hashlib.sha256()
    digest.update(batch.state_id.encode("utf-8"))
    digest.update(batch.state_digest.encode("ascii"))
    for value in (
        batch.f12_joint,
        batch.f12_break,
        batch.f12_inherited_count,
        batch.f12_survival,
        batch.f12_growth_updates,
        batch.f12_final_entropy,
        batch.f12_final_occupied,
        batch.strict_targets,
        batch.strict_break,
        batch.strict_run8,
        batch.f32_survival,
        batch.intervention_joint,
        batch.intervention_break,
        batch.intervention_inherited_count,
        batch.intervention_survival,
    ):
        digest.update(np.ascontiguousarray(value).tobytes())
    for edit in batch.edits:
        pair = (-1, -1) if edit is None else (edit.remove_type, edit.add_type)
        digest.update(np.asarray(pair, dtype=np.int16).tobytes())
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(_json_ready(value), handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _atomic_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def _write_status(
    work: Path,
    stage: str,
    completed: int,
    total: int,
    **extra: Any,
) -> None:
    _atomic_json(
        work / "campaign_status.json",
        {
            "format": STATUS_FORMAT,
            "stage": stage,
            "completed": int(completed),
            "total": int(total),
            "fraction": float(completed / total) if total else 1.0,
            **extra,
        },
    )


def _checkpoint_contract(
    cases: list[NullCase], registration_id: str, stage: str
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "registration_id": registration_id,
        "label": LABEL,
        "stage": stage,
        "case_ids": [case.state_id for case in cases],
        "case_digests": [_state_digest(case) for case in cases],
        "mechanisms": list(MECHANISMS),
        "intervention_arms": list(INTERVENTION_ARMS),
        "f32_branches": F32_BRANCHES,
        "intervention_branches": INTERVENTION_BRANCHES,
        "f12_horizon": F12_HORIZON,
        "f32_horizon": F32_HORIZON,
        "future_seed_excludes_mechanism_and_arm": True,
        "source_hashes": _source_hashes(),
    }
    value["contract_id"] = _canonical_digest(_json_ready(value))
    return value


def run_checkpointed_batches(
    cases: list[NullCase],
    config: GardConfig,
    registration_id: str,
    directory: Path,
    work: Path,
    stage: str,
    workers: int,
) -> list[NullBatch]:
    directory.mkdir(parents=True, exist_ok=True)
    contract = _checkpoint_contract(cases, registration_id, stage)
    contract_path = directory / "checkpoint_contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text(encoding="utf-8")) != json.loads(
            json.dumps(_json_ready(contract))
        ):
            raise ValueError(f"GN1 checkpoint contract changed: {directory}")
    else:
        _atomic_json(contract_path, contract)

    batches: list[NullBatch | None] = [None] * len(cases)
    missing: list[int] = []
    for index, case in enumerate(cases):
        path = directory / f"state_{index:04d}.pkl"
        if path.is_file():
            with path.open("rb") as handle:
                batch = pickle.load(handle)
            if (
                not isinstance(batch, NullBatch)
                or batch.state_id != case.state_id
                or batch.state_digest != _state_digest(case)
            ):
                raise ValueError(f"invalid GN1 checkpoint: {path}")
            batches[index] = batch
        else:
            missing.append(index)
    completed = len(cases) - len(missing)
    _write_status(work, stage, completed, len(cases), reused=completed)

    arguments = [(cases[index], config) for index in missing]
    generated: Iterable[NullBatch]
    if workers <= 1:
        generated = map(_worker, arguments)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=workers)
        generated = executor.map(_worker, arguments, chunksize=1)
    try:
        for index, batch in zip(missing, generated, strict=True):
            if batch.state_id != cases[index].state_id:
                raise AssertionError("GN1 worker ordering changed")
            batches[index] = batch
            _atomic_pickle(directory / f"state_{index:04d}.pkl", batch)
            completed += 1
            _write_status(work, stage, completed, len(cases), reused=len(cases) - len(missing))
            print(f"[{stage}] {completed}/{len(cases)} state batches", flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
    if any(batch is None for batch in batches):
        raise AssertionError("GN1 checkpoint stage has missing batches")
    _write_status(work, stage, len(cases), len(cases), reused=len(cases) - len(missing))
    return [batch for batch in batches if batch is not None]


def protocol() -> dict[str, Any]:
    value: dict[str, Any] = {
        "format": PROGRAM_FORMAT,
        "status": "frozen_before_any_gn1_scientific_matrix",
        "scope": "reviewer-response catalytic-versus-geometric decomposition",
        "separate_from": [
            "closed strict-eight prediction program",
            "closed CR1--CR10 intervention program",
        ],
        "cohort": {
            "fresh_matrices": MATRICES,
            "candidates": list(CANDIDATES),
            "landmarks": list(LANDMARKS),
            "states": MATRICES * len(CANDIDATES) * len(LANDMARKS) * len(MECHANISMS),
            "main_path_attempts": 100,
            "matrix_replacement": False,
        },
        "mechanisms": {
            "NATURAL_GARD": "sealed heterogeneous GARD main paths and futures",
            "HOMOGENEOUS_GENERATIVE": {
                "constant": HOMOGENEOUS_VALUE,
                "formula": "exp(-4 + 4^2/2)",
                "generates_own_main_path": True,
            },
            "COUPLING_DERANGED": {
                "operation": "fixed-point-free simultaneous beta row/column relabeling after natural landmark restoration",
                "composition_and_history_unchanged": True,
                "same_permutation_all_landmarks_and_candidates_within_matrix": True,
                "entry_multiset_and_spectrum_preserved": True,
            },
            "FISSION_ONLY_GENERATIVE": {
                "growth": "IID uniform arrivals to exact mass 80",
                "partition_and_selected_daughter": "candidate-specific sealed contracts",
                "beta_used_by_dynamics": False,
                "growth_count": "number of uniform arrivals",
                "generates_own_main_path": True,
            },
        },
        "endpoints": {
            "f12": "JOINT_BREAK_RUN3 using strict unrounded H>0.9",
            "strict_f32": {
                "primary": "primary_all8",
                "secondary": ["secondary_first5", "secondary_centroid"],
                "inheritance": ">0.9",
                "coherence": ">0.9",
                "old_anchor_distinctness": "<=0.85",
            },
            "extinction": "absorbing; uncertified event negative",
        },
        "futures": {
            "untreated_f32_per_state": F32_BRANCHES,
            "f12_from_f32_prefix": True,
            "branch_halves": {"A": [0, 31], "B": [32, 63]},
            "intervention_f12_per_arm": INTERVENTION_BRANCHES,
            "intervention_arms": list(INTERVENTION_ARMS),
            "noop_reuses_f32_branches": [0, 31],
            "complete_replay": True,
            "future_retry": False,
        },
        "source_rule": {
            "quantity": "x @ original_beta == original_beta.T @ x",
            "selection_beta": "original heterogeneous matrix for every mechanism",
            "purpose": "transport registered causal rule after catalytic coupling removal",
            "random_uniform_over_legal_edits": True,
        },
        "prediction": {
            "frozen_archive_sha256": EXPECTED_FROZEN_MODEL_SHA256,
            "transfer_contrast": "FULL_STATE_GRAPH_HISTORY minus DIRECT_HISTORY_PHASE log-loss",
            "feature_envelope_z": FEATURE_ENVELOPE_Z,
            "fission_only_beta_placeholder": HOMOGENEOUS_VALUE,
            "crossfit": {
                "fold": "matrix_id modulo 2",
                "baseline": "unique H10, unpenalized",
                "added": "state/composition only; no beta-derived interactions",
                "candidate_ridge": STATE_RIDGE,
                "training_transforms_only": True,
                "hyperparameter_search": False,
            },
        },
        "inference": {
            "unit": "whole catalytic matrix",
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "randomization_repetitions": RANDOMIZATION_REPETITIONS,
            "holm_families": {
                "natural_null": 6,
                "candidate_mechanism_intervention": 8,
            },
            "directional_ci": 0.95,
            "equivalence_ci": 0.90,
            "tost_probability_margin": TOST_MARGIN,
            "candidates_pooled_for_rescue": False,
            "omnibus_gate": False,
        },
        "randomness": {
            "seed_domains": SEEDS,
            "future_seed_excludes_mechanism_and_arm": True,
            "random_edit_stream_separate": True,
            "common_random_streams_not_identical_futures": True,
        },
        "integrity": {
            "minimum_free_disk_bytes": MINIMUM_FREE_DISK_BYTES,
            "checkpoint_resumable": True,
            "complete_exact_replay": True,
            "readback_audit": True,
            "mandatory_stop_after_seal": True,
        },
        "claim_boundary": {
            "geometric_floor_reported_at_full_prominence": True,
            "prohibited": [
                "life or autonomous agency",
                "biological memory or error correction",
                "installed compotype",
                "real prebiotic chemistry",
                "Phi or PhiID",
                "universal origin-of-life mechanism",
                "evidence outside the reconstructed GARD family",
            ],
        },
    }
    value["protocol_id"] = _canonical_digest(_json_ready(value))
    return value


def _manual_history_prediction(
    arrays: dict[str, NDArray], candidate: str, history: FloatArray
) -> FloatArray:
    prefix = f"c{candidate}__history"
    values = np.atleast_2d(np.asarray(history, dtype=np.float64))
    transformed = (values - arrays[f"{prefix}__scaler_mean"]) / arrays[
        f"{prefix}__scaler_scale"
    ]
    logits = (
        transformed @ arrays[f"{prefix}__classifier_coef"].T
        + arrays[f"{prefix}__classifier_intercept"]
    ).reshape(-1)
    return np.clip(expit(logits), 1e-12, 1.0 - 1e-12)


def frozen_predictions(
    cases: list[NullCase], model_path: Path, config: GardConfig
) -> tuple[FloatArray, FloatArray, NDArray[np.int8]]:
    predictor = FrozenFullPredictor.load(model_path)
    arrays = predictor.arrays
    direct = np.empty(len(cases), dtype=np.float64)
    full = np.empty(len(cases), dtype=np.float64)
    outside = np.empty(len(cases), dtype=np.int8)
    for index, case in enumerate(cases):
        history = history_features(case.snapshot, config)
        state = state_graph_features(case.snapshot.composition, case.active_beta, config)
        direct[index] = _manual_history_prediction(
            arrays, case.candidate, history[None, :]
        )[0]
        full[index] = predictor.predict_features(
            case.candidate, state[None, :], history[None, :]
        )[0]
        base = f"c{case.candidate}"
        state_z = (state - arrays[f"{base}__full_state_scaler_mean"]) / arrays[
            f"{base}__full_state_scaler_scale"
        ]
        history_z = (
            history - arrays[f"{base}__history__scaler_mean"]
        ) / arrays[f"{base}__history__scaler_scale"]
        outside[index] = int(
            np.max(np.abs(state_z)) > FEATURE_ENVELOPE_Z
            or np.max(np.abs(history_z)) > FEATURE_ENVELOPE_Z
        )
    return direct, full, outside


def _h10_state_features(
    cases: list[NullCase], config: GardConfig
) -> tuple[FloatArray, FloatArray]:
    legacy = np.vstack([history_features(case.snapshot, config) for case in cases])
    h8 = legacy[:, H8_INDICES]
    clocks = np.asarray(
        [
            (
                case.snapshot.previous_growth_steps / max(config.max_growth_steps, 1),
                case.snapshot.cumulative_growth_steps
                / max(config.generations * config.max_growth_steps, 1),
            )
            for case in cases
        ],
        dtype=np.float64,
    )
    h10 = np.column_stack((h8, clocks))
    state = np.vstack(
        [
            state_graph_features(case.snapshot.composition, case.active_beta, config)[
                list(STATE_ONLY_INDICES)
            ]
            for case in cases
        ]
    )
    if h10.shape[1] != len(FEATURE_NAMES["h10"]):
        raise AssertionError("GN1 H10 feature contract changed")
    if state.shape[1] != len(FEATURE_NAMES["state"]):
        raise AssertionError("GN1 state feature contract changed")
    return h10, state


def crossfit_state_predictions(
    cases: list[NullCase],
    batches: list[NullBatch],
    config: GardConfig,
) -> tuple[FloatArray, FloatArray, dict[str, Any]]:
    h10, state = _h10_state_features(cases, config)
    baseline = np.full(len(cases), np.nan, dtype=np.float64)
    enhanced = np.full(len(cases), np.nan, dtype=np.float64)
    audit: dict[str, Any] = {"fits": [], "matrix_overlap": False}
    for candidate in CANDIDATES:
        for mechanism in MECHANISMS:
            group = np.asarray(
                [
                    index
                    for index, case in enumerate(cases)
                    if case.candidate == candidate and case.mechanism == mechanism
                ],
                dtype=np.int64,
            )
            matrix_ids = np.asarray([cases[index].matrix_id for index in group])
            successes = np.asarray(
                [batches[index].f12_joint.sum() for index in group], dtype=np.float64
            )
            trials = np.full(group.size, F32_BRANCHES, dtype=np.float64)
            for fold in (0, 1):
                train_local = matrix_ids % 2 != fold
                test_local = ~train_local
                train = group[train_local]
                test = group[test_local]
                if set(matrix_ids[train_local]).intersection(matrix_ids[test_local]):
                    audit["matrix_overlap"] = True
                    raise AssertionError("GN1 cross-fit leaked a catalytic matrix")
                h_transform = fit_block_transform(
                    "h10", h10[train], FEATURE_NAMES["h10"]
                )
                state_transform = fit_block_transform(
                    "state", state[train], FEATURE_NAMES["state"]
                )
                train_h = h_transform.transform(h10[train])
                test_h = h_transform.transform(h10[test])
                train_state = state_transform.transform(state[train])
                test_state = state_transform.transform(state[test])
                base = fit_linear(
                    f"gn1_{candidate}_{mechanism}_fold{fold}_h10",
                    "h10",
                    train_h,
                    successes[train_local],
                    trials[train_local],
                    0.0,
                )
                train_logits = base.correction(train_h)
                test_logits = base.correction(test_h)
                added = fit_linear(
                    f"gn1_{candidate}_{mechanism}_fold{fold}_state",
                    "state",
                    train_state,
                    successes[train_local],
                    trials[train_local],
                    STATE_RIDGE[candidate],
                    train_logits,
                )
                baseline[test] = expit(test_logits)
                enhanced[test] = expit(test_logits + added.correction(test_state))
                audit["fits"].append(
                    {
                        "candidate": candidate,
                        "mechanism": mechanism,
                        "fold": fold,
                        "train_matrices": sorted(set(matrix_ids[train_local].tolist())),
                        "test_matrices": sorted(set(matrix_ids[test_local].tolist())),
                        "h10_features": int(train_h.shape[1]),
                        "state_features": int(train_state.shape[1]),
                        "state_ridge": STATE_RIDGE[candidate],
                        "base_gradient_max_abs": base.gradient_max_abs,
                        "state_gradient_max_abs": added.gradient_max_abs,
                    }
                )
    if not np.isfinite(baseline).all() or not np.isfinite(enhanced).all():
        raise AssertionError("GN1 cross-fitting left missing predictions")
    audit["all_predictions_finite"] = True
    audit["folds"] = 2
    return baseline, enhanced, audit


def build_tables(
    cases: list[NullCase],
    batches: list[NullBatch],
    frozen_direct: FloatArray,
    frozen_full: FloatArray,
    frozen_outside: NDArray[np.int8],
    crossfit_direct: FloatArray,
    crossfit_state: FloatArray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    state_rows: list[dict[str, Any]] = []
    intervention_rows: list[dict[str, Any]] = []
    edit_rows: list[dict[str, Any]] = []
    for index, (case, batch) in enumerate(zip(cases, batches, strict=True)):
        composition = np.asarray(case.snapshot.composition, dtype=np.int64)
        fraction = composition.astype(np.float64) / composition.sum()
        active_throughput = (
            float(fraction @ case.active_beta @ fraction)
            if case.mechanism != "FISSION_ONLY_GENERATIVE"
            else float("nan")
        )
        source_throughput = float(fraction @ case.source_beta @ fraction)
        half = F32_BRANCHES // 2
        state_rows.append(
            {
                "state_id": case.state_id,
                "mechanism": case.mechanism,
                "candidate": case.candidate,
                "matrix_id": case.matrix_id,
                "landmark": case.landmark,
                "main_attempt": case.main_attempt,
                "launch_mass": int(composition.sum()),
                "launch_occupied": int(np.count_nonzero(composition)),
                "launch_entropy": _entropy(composition),
                "source_throughput": source_throughput,
                "active_throughput": active_throughput,
                "q_f12": float(batch.f12_joint.mean()),
                "q_f12_A": float(batch.f12_joint[:half].mean()),
                "q_f12_B": float(batch.f12_joint[half:].mean()),
                "q_break_f12": float(batch.f12_break.mean()),
                "mean_inherited_f12": float(batch.f12_inherited_count.mean()),
                "survival_f12": float(batch.f12_survival.mean()),
                "mean_growth_updates_f12": float(batch.f12_growth_updates.mean()),
                "mean_final_entropy_f12": float(batch.f12_final_entropy.mean()),
                "mean_final_occupied_f12": float(batch.f12_final_occupied.mean()),
                "q_break_f32": float(batch.strict_break.mean()),
                "q_run8_f32": float(batch.strict_run8.mean()),
                "q_strict_all8": float(batch.strict_targets[:, 0].mean()),
                "q_strict_all8_A": float(batch.strict_targets[:half, 0].mean()),
                "q_strict_all8_B": float(batch.strict_targets[half:, 0].mean()),
                "q_strict_first5": float(batch.strict_targets[:, 1].mean()),
                "q_strict_centroid": float(batch.strict_targets[:, 2].mean()),
                "survival_f32": float(batch.f32_survival.mean()),
                "frozen_direct": float(frozen_direct[index]),
                "frozen_full": float(frozen_full[index]),
                "frozen_outside_5sd": int(frozen_outside[index]),
                "crossfit_h10": float(crossfit_direct[index]),
                "crossfit_h10_state": float(crossfit_state[index]),
                "state_digest": batch.state_digest,
                "batch_digest": _batch_digest(batch),
            }
        )
        for arm_index, arm in enumerate(INTERVENTION_ARMS):
            intervention_rows.append(
                {
                    "state_id": case.state_id,
                    "mechanism": case.mechanism,
                    "candidate": case.candidate,
                    "matrix_id": case.matrix_id,
                    "landmark": case.landmark,
                    "arm": arm,
                    "q_f12": float(batch.intervention_joint[arm_index].mean()),
                    "q_break_f12": float(
                        batch.intervention_break[arm_index].mean()
                    ),
                    "mean_inherited_f12": float(
                        batch.intervention_inherited_count[arm_index].mean()
                    ),
                    "survival_f12": float(
                        batch.intervention_survival[arm_index].mean()
                    ),
                }
            )
            edit = batch.edits[arm_index]
            edit_rows.append(
                {
                    "state_id": case.state_id,
                    "mechanism": case.mechanism,
                    "candidate": case.candidate,
                    "matrix_id": case.matrix_id,
                    "landmark": case.landmark,
                    "arm": arm,
                    "remove_type": -1 if edit is None else edit.remove_type,
                    "add_type": -1 if edit is None else edit.add_type,
                }
            )
    return (
        pd.DataFrame.from_records(state_rows),
        pd.DataFrame.from_records(intervention_rows),
        pd.DataFrame.from_records(edit_rows),
    )


def _matrix_means(values: FloatArray, matrix_ids: NDArray) -> FloatArray:
    ids = np.asarray(matrix_ids, dtype=np.int64)
    data = np.asarray(values, dtype=np.float64)
    return np.asarray(
        [data[ids == matrix_id].mean() for matrix_id in np.unique(ids)],
        dtype=np.float64,
    )


def _bootstrap_mean(
    matrix_values: FloatArray, name: str, level: float = 0.95
) -> tuple[float, tuple[float, float], FloatArray]:
    values = np.asarray(matrix_values, dtype=np.float64)
    rng = np.random.default_rng(
        derive_seed(SEEDS["bootstrap"], f"{LABEL}.bootstrap", name)
    )
    indices = rng.integers(
        0, values.size, size=(BOOTSTRAP_REPETITIONS, values.size)
    )
    samples = values[indices].mean(axis=1)
    alpha = (1.0 - level) / 2.0
    interval = np.quantile(samples, (alpha, 1.0 - alpha))
    return (
        float(values.mean()),
        (float(interval[0]), float(interval[1])),
        samples,
    )


def _sign_randomization_p(matrix_values: FloatArray, name: str) -> tuple[float, FloatArray]:
    values = np.asarray(matrix_values, dtype=np.float64)
    observed = float(values.mean())
    rng = np.random.default_rng(
        derive_seed(SEEDS["randomization"], f"{LABEL}.randomization", name)
    )
    signs = rng.integers(
        0,
        2,
        size=(RANDOMIZATION_REPETITIONS, values.size),
        dtype=np.int8,
    ).astype(np.float64)
    signs = signs * 2.0 - 1.0
    null = signs @ values / values.size
    p_value = float(
        (np.count_nonzero(null >= observed) + 1)
        / (RANDOMIZATION_REPETITIONS + 1)
    )
    return p_value, null


def _effect_summary(
    values: FloatArray,
    matrix_ids: NDArray,
    name: str,
    arrays: dict[str, FloatArray],
    equivalence_margin: float | None = None,
) -> dict[str, Any]:
    grouped = _matrix_means(values, matrix_ids)
    observed, interval, samples = _bootstrap_mean(grouped, name, 0.95)
    p_value, null = _sign_randomization_p(grouped, name)
    arrays[f"{name}__bootstrap"] = samples
    arrays[f"{name}__randomization"] = null
    result: dict[str, Any] = {
        "effect": observed,
        "ci95": interval,
        "randomization_p_raw": p_value,
        "matrices": int(grouped.size),
        "matrices_positive": int(np.count_nonzero(grouped > 0.0)),
        "maximum_single_matrix_abs": float(np.max(np.abs(grouped))),
    }
    if equivalence_margin is not None:
        _, interval90, samples90 = _bootstrap_mean(grouped, name + ".tost", 0.90)
        arrays[f"{name}__bootstrap90"] = samples90
        result.update(
            {
                "ci90": interval90,
                "equivalence_margin": equivalence_margin,
                "equivalent": bool(
                    interval90[0] > -equivalence_margin
                    and interval90[1] < equivalence_margin
                ),
            }
        )
    return result


def _proper_brier(q: FloatArray, prediction: FloatArray) -> FloatArray:
    q = np.asarray(q, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    return q * (1.0 - p) ** 2 + (1.0 - q) * p**2


def _log_loss(q: FloatArray, prediction: FloatArray) -> FloatArray:
    q = np.asarray(q, dtype=np.float64)
    p = np.clip(np.asarray(prediction, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    return -(q * np.log(p) + (1.0 - q) * np.log(1.0 - p))


def _reliability_summary(
    left: FloatArray,
    right: FloatArray,
    matrix_ids: NDArray,
    name: str,
    arrays: dict[str, FloatArray],
) -> dict[str, Any]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    ids = np.asarray(matrix_ids, dtype=np.int64)
    unique = np.unique(ids)
    locations = [np.flatnonzero(ids == value) for value in unique]
    rng = np.random.default_rng(
        derive_seed(SEEDS["bootstrap"], f"{LABEL}.reliability", name)
    )
    overall_samples = np.empty(BOOTSTRAP_REPETITIONS, dtype=np.float64)
    centered_samples = np.empty(BOOTSTRAP_REPETITIONS, dtype=np.float64)
    for repetition in range(BOOTSTRAP_REPETITIONS):
        chosen = rng.integers(0, unique.size, size=unique.size)
        selected = np.concatenate([locations[index] for index in chosen])
        groups = np.repeat(
            np.arange(unique.size), [locations[index].size for index in chosen]
        )
        overall_samples[repetition] = spearman(left[selected], right[selected])
        centered_samples[repetition] = centered_spearman(
            left[selected], right[selected], groups
        )
    arrays[f"{name}__overall_bootstrap"] = overall_samples
    arrays[f"{name}__centered_bootstrap"] = centered_samples

    def interval(values: FloatArray) -> tuple[float, float]:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return (float("nan"), float("nan"))
        low, high = np.quantile(finite, (0.025, 0.975))
        return float(low), float(high)

    return {
        "overall": spearman(left, right),
        "overall_ci95": interval(overall_samples),
        "centered_within_matrix": centered_spearman(left, right, ids),
        "centered_ci95": interval(centered_samples),
        "states": int(left.size),
        "matrices": int(unique.size),
    }


def _paired_reliability_difference(
    natural: pd.DataFrame,
    null: pd.DataFrame,
    candidate: str,
    mechanism: str,
    arrays: dict[str, FloatArray],
) -> dict[str, Any]:
    keys = ["matrix_id", "landmark"]
    merged = natural.merge(null, on=keys, suffixes=("_natural", "_null"))
    ids = merged["matrix_id"].to_numpy(dtype=np.int64)
    unique = np.unique(ids)
    locations = [np.flatnonzero(ids == value) for value in unique]
    name = f"reliability_diff.c{candidate}.{mechanism}"
    rng = np.random.default_rng(
        derive_seed(SEEDS["bootstrap"], f"{LABEL}.reliability_diff", candidate, mechanism)
    )
    overall = np.empty(BOOTSTRAP_REPETITIONS, dtype=np.float64)
    centered = np.empty(BOOTSTRAP_REPETITIONS, dtype=np.float64)
    for repetition in range(BOOTSTRAP_REPETITIONS):
        chosen = rng.integers(0, unique.size, size=unique.size)
        selected = np.concatenate([locations[index] for index in chosen])
        groups = np.repeat(
            np.arange(unique.size), [locations[index].size for index in chosen]
        )
        overall[repetition] = spearman(
            merged["q_f12_A_natural"].to_numpy()[selected],
            merged["q_f12_B_natural"].to_numpy()[selected],
        ) - spearman(
            merged["q_f12_A_null"].to_numpy()[selected],
            merged["q_f12_B_null"].to_numpy()[selected],
        )
        centered[repetition] = centered_spearman(
            merged["q_f12_A_natural"].to_numpy()[selected],
            merged["q_f12_B_natural"].to_numpy()[selected],
            groups,
        ) - centered_spearman(
            merged["q_f12_A_null"].to_numpy()[selected],
            merged["q_f12_B_null"].to_numpy()[selected],
            groups,
        )
    arrays[f"{name}__overall_bootstrap"] = overall
    arrays[f"{name}__centered_bootstrap"] = centered
    overall_finite = overall[np.isfinite(overall)]
    centered_finite = centered[np.isfinite(centered)]
    return {
        "candidate": candidate,
        "null": mechanism,
        "natural_minus_null_overall": float(
            spearman(merged["q_f12_A_natural"], merged["q_f12_B_natural"])
            - spearman(merged["q_f12_A_null"], merged["q_f12_B_null"])
        ),
        "overall_ci95": tuple(
            float(value) for value in np.quantile(overall_finite, (0.025, 0.975))
        ) if overall_finite.size else (float("nan"), float("nan")),
        "natural_minus_null_centered": float(
            centered_spearman(
                merged["q_f12_A_natural"].to_numpy(),
                merged["q_f12_B_natural"].to_numpy(),
                ids,
            )
            - centered_spearman(
                merged["q_f12_A_null"].to_numpy(),
                merged["q_f12_B_null"].to_numpy(),
                ids,
            )
        ),
        "centered_ci95": tuple(
            float(value) for value in np.quantile(centered_finite, (0.025, 0.975))
        ) if centered_finite.size else (float("nan"), float("nan")),
    }


def _mean_summary(
    values: FloatArray,
    matrix_ids: NDArray,
    name: str,
    arrays: dict[str, FloatArray],
) -> dict[str, Any]:
    grouped = _matrix_means(values, matrix_ids)
    observed, interval, samples = _bootstrap_mean(grouped, name)
    arrays[f"{name}__bootstrap"] = samples
    return {"mean": observed, "ci95": interval, "matrices": int(grouped.size)}


def _apply_holm(rows: list[dict[str, Any]], key: str = "randomization_p_raw") -> None:
    if not rows:
        return
    adjusted = holm_adjust([float(row[key]) for row in rows])
    for row, value in zip(rows, adjusted, strict=True):
        row["randomization_p_holm"] = float(value)


def compute_inference(
    states: pd.DataFrame, interventions: pd.DataFrame
) -> tuple[dict[str, Any], dict[str, FloatArray]]:
    arrays: dict[str, FloatArray] = {}
    metrics: dict[str, Any] = {
        "format": "codex-generative-null-inference-v1",
        "cell_summaries": [],
        "reliability": [],
        "reliability_natural_minus_null": [],
        "frozen_predictor": [],
        "crossfit_state_predictor": [],
        "natural_minus_null": {"f12": [], "strict_all8": []},
        "intervention": [],
        "rule_attenuation": [],
        "classification": {},
    }

    frozen_rows_by_half: dict[str, list[dict[str, Any]]] = {"A": [], "B": []}
    crossfit_rows_by_half: dict[str, list[dict[str, Any]]] = {"A": [], "B": []}
    intervention_rows: list[dict[str, Any]] = []

    for candidate in CANDIDATES:
        for mechanism in MECHANISMS:
            cell = states[
                (states["candidate"] == candidate)
                & (states["mechanism"] == mechanism)
            ].copy()
            matrix_ids = cell["matrix_id"].to_numpy(dtype=np.int64)
            summary: dict[str, Any] = {
                "candidate": candidate,
                "mechanism": mechanism,
                "states": int(cell.shape[0]),
                "f12": _mean_summary(
                    cell["q_f12"].to_numpy(),
                    matrix_ids,
                    f"mean.f12.c{candidate}.{mechanism}",
                    arrays,
                ),
                "break_f12": _mean_summary(
                    cell["q_break_f12"].to_numpy(),
                    matrix_ids,
                    f"mean.break_f12.c{candidate}.{mechanism}",
                    arrays,
                ),
                "survival_f12": _mean_summary(
                    cell["survival_f12"].to_numpy(),
                    matrix_ids,
                    f"mean.survival_f12.c{candidate}.{mechanism}",
                    arrays,
                ),
                "strict_all8": _mean_summary(
                    cell["q_strict_all8"].to_numpy(),
                    matrix_ids,
                    f"mean.strict_all8.c{candidate}.{mechanism}",
                    arrays,
                ),
                "strict_first5": _mean_summary(
                    cell["q_strict_first5"].to_numpy(),
                    matrix_ids,
                    f"mean.strict_first5.c{candidate}.{mechanism}",
                    arrays,
                ),
                "strict_centroid": _mean_summary(
                    cell["q_strict_centroid"].to_numpy(),
                    matrix_ids,
                    f"mean.strict_centroid.c{candidate}.{mechanism}",
                    arrays,
                ),
                "run8_f32": _mean_summary(
                    cell["q_run8_f32"].to_numpy(),
                    matrix_ids,
                    f"mean.run8.c{candidate}.{mechanism}",
                    arrays,
                ),
                "survival_f32": _mean_summary(
                    cell["survival_f32"].to_numpy(),
                    matrix_ids,
                    f"mean.survival_f32.c{candidate}.{mechanism}",
                    arrays,
                ),
                "mean_inherited_f12": _mean_summary(
                    cell["mean_inherited_f12"].to_numpy(),
                    matrix_ids,
                    f"mean.inherited_f12.c{candidate}.{mechanism}",
                    arrays,
                ),
                "frozen_outside_5sd_fraction": float(
                    cell["frozen_outside_5sd"].mean()
                ),
            }
            metrics["cell_summaries"].append(summary)
            reliability = _reliability_summary(
                cell["q_f12_A"].to_numpy(),
                cell["q_f12_B"].to_numpy(),
                matrix_ids,
                f"reliability.c{candidate}.{mechanism}",
                arrays,
            )
            reliability.update({"candidate": candidate, "mechanism": mechanism})
            metrics["reliability"].append(reliability)

            for half in ("A", "B"):
                q = cell[f"q_f12_{half}"].to_numpy(dtype=np.float64)
                frozen_gain = _effect_summary(
                    _log_loss(q, cell["frozen_direct"].to_numpy())
                    - _log_loss(q, cell["frozen_full"].to_numpy()),
                    matrix_ids,
                    f"frozen_gain.{half}.c{candidate}.{mechanism}",
                    arrays,
                )
                frozen_gain.update(
                    {
                        "candidate": candidate,
                        "mechanism": mechanism,
                        "half": half,
                        "brier_gain": float(
                            (
                                _proper_brier(q, cell["frozen_direct"].to_numpy())
                                - _proper_brier(q, cell["frozen_full"].to_numpy())
                            ).mean()
                        ),
                        "ordinary_branch_level_brier": True,
                    }
                )
                frozen_rows_by_half[half].append(frozen_gain)

                crossfit_gain = _effect_summary(
                    _log_loss(q, cell["crossfit_h10"].to_numpy())
                    - _log_loss(q, cell["crossfit_h10_state"].to_numpy()),
                    matrix_ids,
                    f"crossfit_gain.{half}.c{candidate}.{mechanism}",
                    arrays,
                )
                crossfit_gain.update(
                    {
                        "candidate": candidate,
                        "mechanism": mechanism,
                        "half": half,
                        "brier_gain": float(
                            (
                                _proper_brier(q, cell["crossfit_h10"].to_numpy())
                                - _proper_brier(
                                    q, cell["crossfit_h10_state"].to_numpy()
                                )
                            ).mean()
                        ),
                        "ordinary_branch_level_brier": True,
                    }
                )
                crossfit_rows_by_half[half].append(crossfit_gain)

            arm_cell = interventions[
                (interventions["candidate"] == candidate)
                & (interventions["mechanism"] == mechanism)
            ]
            pivot = arm_cell.pivot(
                index=["state_id", "matrix_id", "landmark"],
                columns="arm",
                values="q_f12",
            ).reset_index()
            ids = pivot["matrix_id"].to_numpy(dtype=np.int64)
            contrasts = {
                "up_minus_down": pivot["SOURCE_RULE_UP"].to_numpy()
                - pivot["SOURCE_RULE_DOWN"].to_numpy(),
                "up_minus_noop": pivot["SOURCE_RULE_UP"].to_numpy()
                - pivot["NOOP"].to_numpy(),
                "noop_minus_down": pivot["NOOP"].to_numpy()
                - pivot["SOURCE_RULE_DOWN"].to_numpy(),
                "random_minus_noop": pivot["RANDOM"].to_numpy()
                - pivot["NOOP"].to_numpy(),
            }
            row: dict[str, Any] = {
                "candidate": candidate,
                "mechanism": mechanism,
                "arm_means": {
                    arm: float(pivot[arm].mean()) for arm in INTERVENTION_ARMS
                },
            }
            for contrast, values in contrasts.items():
                row[contrast] = _effect_summary(
                    values,
                    ids,
                    f"intervention.{contrast}.c{candidate}.{mechanism}",
                    arrays,
                    equivalence_margin=(
                        TOST_MARGIN
                        if contrast == "random_minus_noop"
                        or (
                            contrast == "up_minus_down"
                            and mechanism != "NATURAL_GARD"
                        )
                        else None
                    ),
                )
            intervention_rows.append(row)

    for half in ("A", "B"):
        _apply_holm(frozen_rows_by_half[half])
        _apply_holm(crossfit_rows_by_half[half])
        metrics["frozen_predictor"].extend(frozen_rows_by_half[half])
        metrics["crossfit_state_predictor"].extend(crossfit_rows_by_half[half])
    _apply_holm([row["up_minus_down"] for row in intervention_rows])
    metrics["intervention"] = intervention_rows

    natural_null_f12: list[dict[str, Any]] = []
    natural_null_strict: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        natural = states[
            (states["candidate"] == candidate)
            & (states["mechanism"] == "NATURAL_GARD")
        ]
        natural_intervention = interventions[
            (interventions["candidate"] == candidate)
            & (interventions["mechanism"] == "NATURAL_GARD")
        ].pivot(
            index=["matrix_id", "landmark"], columns="arm", values="q_f12"
        ).reset_index()
        natural_intervention["rule_effect"] = (
            natural_intervention["SOURCE_RULE_UP"]
            - natural_intervention["SOURCE_RULE_DOWN"]
        )
        for mechanism in MECHANISMS[1:]:
            null = states[
                (states["candidate"] == candidate)
                & (states["mechanism"] == mechanism)
            ]
            keys = ["matrix_id", "landmark"]
            joined = natural[keys + ["q_f12", "q_strict_all8"]].merge(
                null[keys + ["q_f12", "q_strict_all8"]],
                on=keys,
                suffixes=("_natural", "_null"),
            )
            ids = joined["matrix_id"].to_numpy(dtype=np.int64)
            f12 = _effect_summary(
                joined["q_f12_natural"].to_numpy()
                - joined["q_f12_null"].to_numpy(),
                ids,
                f"natural_minus_null.f12.c{candidate}.{mechanism}",
                arrays,
            )
            f12.update({"candidate": candidate, "null": mechanism})
            natural_null_f12.append(f12)
            strict = _effect_summary(
                joined["q_strict_all8_natural"].to_numpy()
                - joined["q_strict_all8_null"].to_numpy(),
                ids,
                f"natural_minus_null.strict.c{candidate}.{mechanism}",
                arrays,
            )
            strict.update({"candidate": candidate, "null": mechanism})
            natural_null_strict.append(strict)
            metrics["reliability_natural_minus_null"].append(
                _paired_reliability_difference(
                    natural, null, candidate, mechanism, arrays
                )
            )

            null_intervention = interventions[
                (interventions["candidate"] == candidate)
                & (interventions["mechanism"] == mechanism)
            ].pivot(
                index=["matrix_id", "landmark"], columns="arm", values="q_f12"
            ).reset_index()
            null_intervention["rule_effect"] = (
                null_intervention["SOURCE_RULE_UP"]
                - null_intervention["SOURCE_RULE_DOWN"]
            )
            effect_join = natural_intervention[
                keys + ["rule_effect"]
            ].merge(
                null_intervention[keys + ["rule_effect"]],
                on=keys,
                suffixes=("_natural", "_null"),
            )
            attenuation = _effect_summary(
                effect_join["rule_effect_natural"].to_numpy()
                - effect_join["rule_effect_null"].to_numpy(),
                effect_join["matrix_id"].to_numpy(dtype=np.int64),
                f"rule_attenuation.c{candidate}.{mechanism}",
                arrays,
            )
            attenuation.update({"candidate": candidate, "null": mechanism})
            metrics["rule_attenuation"].append(attenuation)

    _apply_holm(natural_null_f12)
    _apply_holm(natural_null_strict)
    metrics["natural_minus_null"]["f12"] = natural_null_f12
    metrics["natural_minus_null"]["strict_all8"] = natural_null_strict

    cell_lookup = {
        (row["candidate"], row["mechanism"]): row
        for row in metrics["cell_summaries"]
    }
    intervention_lookup = {
        (row["candidate"], row["mechanism"]): row
        for row in metrics["intervention"]
    }
    metrics["classification"] = {
        "nonzero_fission_only_geometric_floor_both_candidates": all(
            cell_lookup[(candidate, "FISSION_ONLY_GENERATIVE")]["f12"]["ci95"][0]
            > 0.0
            for candidate in CANDIDATES
        ),
        "natural_rule_effect_positive_both_candidates": all(
            intervention_lookup[(candidate, "NATURAL_GARD")]["up_minus_down"][
                "ci95"
            ][0]
            > 0.0
            for candidate in CANDIDATES
        ),
        "all_null_rule_effects_equivalent_zero": all(
            intervention_lookup[(candidate, mechanism)]["up_minus_down"].get(
                "equivalent", False
            )
            for candidate in CANDIDATES
            for mechanism in MECHANISMS[1:]
        ),
        "interpretation": (
            "component-wise decomposition only; no omnibus pass gate and no null "
            "may be called equivalent solely because its interval crosses zero"
        ),
    }
    return metrics, arrays


def _format_interval(values: Iterable[float]) -> str:
    low, high = (float(value) for value in values)
    return f"[{low:+.4f}, {high:+.4f}]"


def _scientific_report(metrics: dict[str, Any]) -> str:
    rows = [
        "# GN1 catalytic-versus-geometric null decomposition",
        "",
        "GN1 is a prospectively registered reviewer-response decomposition with no single omnibus pass gate. All rates and effects are candidate-separated and use whole catalytic matrices for inference.",
        "",
        "## Event prevalence and statewise reliability",
        "",
        "| Candidate | Mechanism | F12 | 95% CI | strict all8 F32 | 95% CI | reliability | centered reliability |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    reliability = {
        (item["candidate"], item["mechanism"]): item
        for item in metrics["reliability"]
    }
    for item in metrics["cell_summaries"]:
        rel = reliability[(item["candidate"], item["mechanism"])]
        rows.append(
            "| {candidate} | {mechanism} | {f12:.4f} | {f12_ci} | {strict:.4f} | {strict_ci} | {rel:+.3f} | {centered:+.3f} |".format(
                candidate=item["candidate"],
                mechanism=item["mechanism"],
                f12=item["f12"]["mean"],
                f12_ci=_format_interval(item["f12"]["ci95"]),
                strict=item["strict_all8"]["mean"],
                strict_ci=_format_interval(item["strict_all8"]["ci95"]),
                rel=rel["overall"],
                centered=rel["centered_within_matrix"],
            )
        )
    rows.extend(
        [
            "",
            "Overall reliability includes stable between-matrix propensities. The centered value removes each matrix mean and is the more direct test of landmark-specific state dependence.",
            "",
            "## Natural-minus-null event differences",
            "",
            "| Endpoint | Candidate | Null | Natural-null | 95% CI | Holm p |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for endpoint in ("f12", "strict_all8"):
        for item in metrics["natural_minus_null"][endpoint]:
            rows.append(
                f"| {endpoint} | {item['candidate']} | {item['null']} | {item['effect']:+.4f} | {_format_interval(item['ci95'])} | {item['randomization_p_holm']:.6g} |"
            )
    rows.extend(
        [
            "",
            "## Frozen manuscript-algorithm transfer",
            "",
            "Positive values mean lower branch-level log loss for FULL_STATE_GRAPH_HISTORY than DIRECT_HISTORY_PHASE.",
            "",
            "| Candidate | Mechanism | Half | Log-loss gain | 95% CI | Holm p |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for item in metrics["frozen_predictor"]:
        rows.append(
            f"| {item['candidate']} | {item['mechanism']} | {item['half']} | {item['effect']:+.6f} | {_format_interval(item['ci95'])} | {item['randomization_p_holm']:.6g} |"
        )
    rows.extend(
        [
            "",
            "## Null-specific clean composition predictor",
            "",
            "Positive values mean that a two-way whole-matrix-cross-fitted composition block improves over the unique H10 direct baseline. Penalties were fixed before outcomes and no null-specific tuning occurred.",
            "",
            "| Candidate | Mechanism | Half | Log-loss gain | 95% CI | Holm p |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for item in metrics["crossfit_state_predictor"]:
        rows.append(
            f"| {item['candidate']} | {item['mechanism']} | {item['half']} | {item['effect']:+.6f} | {_format_interval(item['ci95'])} | {item['randomization_p_holm']:.6g} |"
        )
    rows.extend(
        [
            "",
            "## Transported outgoing-rule intervention",
            "",
            "The original heterogeneous beta selected SOURCE_RULE_UP and SOURCE_RULE_DOWN in every mechanism. Positive effects mean more F12 under the risk-raising than the stabilizing edit.",
            "",
            "| Candidate | Mechanism | Up-down | 95% CI | Holm p | Null TOST | Random-noop | Random TOST |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in metrics["intervention"]:
        effect = item["up_minus_down"]
        random = item["random_minus_noop"]
        rows.append(
            f"| {item['candidate']} | {item['mechanism']} | {effect['effect']:+.4f} | {_format_interval(effect['ci95'])} | {effect['randomization_p_holm']:.6g} | {effect.get('equivalent', 'n/a')} | {random['effect']:+.4f} | {random.get('equivalent', False)} |"
        )
    rows.extend(
        [
            "",
            "## Integrity and claim boundary",
            "",
            "Every scientific future was replayed completely. Natural no-op uses the unmodified sealed simulator. No future was retried and no failed or extinct matrix was replaced.",
            "",
            "GN1 can quantify a geometric floor and catalytic contributions within reconstructed GARD. It cannot establish life, biological memory, autonomous agency, an installed compotype, real prebiotic chemistry, Phi/PhiID, or a universal origin-of-life mechanism.",
            "",
        ]
    )
    return "\n".join(rows)


def _lay_summary(metrics: dict[str, Any]) -> str:
    lookup = {
        (item["candidate"], item["mechanism"]): item
        for item in metrics["cell_summaries"]
    }
    natural = [
        lookup[(candidate, "NATURAL_GARD")]["f12"]["mean"]
        for candidate in CANDIDATES
    ]
    fission = [
        lookup[(candidate, "FISSION_ONLY_GENERATIVE")]["f12"]["mean"]
        for candidate in CANDIDATES
    ]
    homogeneous = [
        lookup[(candidate, "HOMOGENEOUS_GENERATIVE")]["f12"]["mean"]
        for candidate in CANDIDATES
    ]
    intervention = {
        (item["candidate"], item["mechanism"]): item
        for item in metrics["intervention"]
    }
    natural_rule = [
        intervention[(candidate, "NATURAL_GARD")]["up_minus_down"]["effect"]
        for candidate in CANDIDATES
    ]
    return "\n".join(
        [
            "# GN1 in plain language",
            "",
            "This experiment asks how much break-and-renewal comes from the mechanics of filling, splitting, and choosing a daughter, and how much depends on catalytic chemistry being aligned with the current assembly.",
            "",
            f"Natural reconstructed GARD produced F12 rates of {natural[0]:.1%} and {natural[1]:.1%}. The completely fission-only process produced {fission[0]:.1%} and {fission[1]:.1%}; homogeneous catalysis produced {homogeneous[0]:.1%} and {homogeneous[1]:.1%}. A nonzero null rate is the geometric floor and is not catalytic heredity.",
            "",
            f"In natural GARD, the transported one-molecule catalytic rule separated risk-raising from stabilizing edits by {natural_rule[0]:.1%} and {natural_rule[1]:.1%}. The full report shows whether this control survived or disappeared when catalytic coupling was removed.",
            "",
            "The state-reliability and predictor comparisons ask a different question: even if events remain common, are the same restored states reproducibly more vulnerable, and does composition add predictive information beyond history?",
            "",
            "This is a decomposition inside simulated GARD-like systems. It does not show life, agency, biological memory, or real prebiotic chemistry.",
            "",
        ]
    )


def _append_seal_notice(registration_id: str) -> None:
    text = LEDGER.read_text(encoding="utf-8").rstrip() + "\n"
    marker = f"<!-- sealed-gn1-{registration_id} -->"
    if marker in text:
        return
    rows = [
        "",
        marker,
        "## GN1 generative-null decomposition sealed",
        "",
        f"- Registration: `{registration_id}`.",
        "- Result: `results/generative_null_decomposition`.",
        "- Natural, homogeneous, post-restoration coupling-deranged, and fission-only mechanisms completed with full replay.",
        "- There is no omnibus pass gate; geometric floor, frequency, reliability, prediction, strict occurrence, and steerability are reported separately.",
        "- GN1 stopped after sealing and did not alter any strict-eight or CR1--CR10 result.",
        "",
    ]
    LEDGER.write_text(text + "\n".join(rows), encoding="utf-8")


def write_result(
    output: Path,
    registration: dict[str, Any],
    cases: list[NullCase],
    batches: list[NullBatch],
    replay_batches: list[NullBatch],
    states: pd.DataFrame,
    interventions: pd.DataFrame,
    edits: pd.DataFrame,
    metrics: dict[str, Any],
    inference_arrays: dict[str, FloatArray],
    crossfit_audit: dict[str, Any],
) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite GN1 result: {output}")
    replay_match = np.asarray(
        [
            _batch_digest(left) == _batch_digest(right)
            for left, right in zip(batches, replay_batches, strict=True)
        ],
        dtype=bool,
    )
    if not bool(replay_match.all()):
        raise ValueError("GN1 exact replay mismatch")
    matrices = sorted({case.matrix_id for case in cases})
    source_betas = np.stack(
        [
            next(case.source_beta for case in cases if case.matrix_id == matrix_id)
            for matrix_id in matrices
        ]
    )
    permutations = np.stack(
        [
            next(
                case.coupling_permutation
                for case in cases
                if case.matrix_id == matrix_id
            )
            for matrix_id in matrices
        ]
    )
    compositions = np.stack([case.snapshot.composition for case in cases])
    branch_arrays = {
        "f12_joint": np.stack([batch.f12_joint for batch in batches]),
        "f12_break": np.stack([batch.f12_break for batch in batches]),
        "f12_inherited_count": np.stack(
            [batch.f12_inherited_count for batch in batches]
        ),
        "f12_survival": np.stack([batch.f12_survival for batch in batches]),
        "strict_targets": np.stack([batch.strict_targets for batch in batches]),
        "strict_break": np.stack([batch.strict_break for batch in batches]),
        "strict_run8": np.stack([batch.strict_run8 for batch in batches]),
        "f32_survival": np.stack([batch.f32_survival for batch in batches]),
        "intervention_joint": np.stack(
            [batch.intervention_joint for batch in batches]
        ),
        "intervention_break": np.stack(
            [batch.intervention_break for batch in batches]
        ),
        "intervention_survival": np.stack(
            [batch.intervention_survival for batch in batches]
        ),
    }
    manifest = {
        "format": RESULT_FORMAT,
        "registration_id": registration["registration_id"],
        "matrices": MATRICES,
        "states": len(cases),
        "mechanisms": list(MECHANISMS),
        "untreated_f32_futures": len(cases) * F32_BRANCHES,
        "nonnoop_intervention_f12_futures": len(cases)
        * (len(INTERVENTION_ARMS) - 1)
        * INTERVENTION_BRANCHES,
        "noop_intervention_reused_from_f32": True,
        "maximum_scientific_fission_boundaries": len(cases)
        * (
            F32_BRANCHES * F32_HORIZON
            + (len(INTERVENTION_ARMS) - 1)
            * INTERVENTION_BRANCHES
            * F12_HORIZON
        ),
        "maximum_replay_fission_boundaries": len(cases)
        * (
            F32_BRANCHES * F32_HORIZON
            + (len(INTERVENTION_ARMS) - 1)
            * INTERVENTION_BRANCHES
            * F12_HORIZON
        ),
        "exact_replay": bool(replay_match.all()),
        "replayed_state_batches": int(replay_match.size),
        "no_future_retry_or_matrix_replacement": True,
        "no_omnibus_gate": True,
        "mandatory_stop": True,
        "runtime": _runtime_manifest(),
    }
    claim_boundaries = {
        "supported_scope": "mechanistic decomposition within reconstructed GARD dynamics",
        "geometric_floor_must_not_be_called_catalytic_heredity": True,
        "prohibited": protocol()["claim_boundary"]["prohibited"],
    }
    replay_audit = {
        "format": "codex-generative-null-replay-v1",
        "registration_id": registration["registration_id"],
        "state_batches": int(replay_match.size),
        "all_exact": bool(replay_match.all()),
        "generation_digests_sha256": hashlib.sha256(
            "".join(_batch_digest(batch) for batch in batches).encode("ascii")
        ).hexdigest(),
        "replay_digests_sha256": hashlib.sha256(
            "".join(_batch_digest(batch) for batch in replay_batches).encode("ascii")
        ).hexdigest(),
    }
    matrix_summaries = (
        states.groupby(["mechanism", "candidate", "matrix_id"], as_index=False)
        .mean(numeric_only=True)
        .sort_values(["mechanism", "candidate", "matrix_id"])
    )
    with _atomic_destination(output) as destination:
        states.to_csv(destination / "state_probabilities.csv.gz", index=False)
        interventions.to_csv(
            destination / "intervention_probabilities.csv.gz", index=False
        )
        edits.to_csv(destination / "selected_edits.csv.gz", index=False)
        matrix_summaries.to_csv(destination / "matrix_summaries.csv", index=False)
        np.savez_compressed(
            destination / "state_and_matrix_arrays.npz",
            source_betas=source_betas,
            coupling_permutations=permutations,
            compositions=compositions,
            state_ids=states["state_id"].to_numpy(dtype="U"),
        )
        np.savez_compressed(destination / "branch_outcomes.npz", **branch_arrays)
        np.savez_compressed(destination / "inference_arrays.npz", **inference_arrays)
        _atomic_json(destination / "inference_metrics.json", metrics)
        _atomic_json(destination / "crossfit_audit.json", crossfit_audit)
        _atomic_json(destination / "replay_audit.json", replay_audit)
        _atomic_json(destination / "manifest.json", manifest)
        _atomic_json(destination / "claim_boundaries.json", claim_boundaries)
        (destination / "SCIENTIFIC_REPORT.md").write_text(
            _scientific_report(metrics), encoding="utf-8"
        )
        (destination / "LAY_SUMMARY.md").write_text(
            _lay_summary(metrics), encoding="utf-8"
        )
        read_states = pd.read_csv(destination / "state_probabilities.csv.gz")
        read_interventions = pd.read_csv(
            destination / "intervention_probabilities.csv.gz"
        )
        with np.load(destination / "branch_outcomes.npz") as read_arrays:
            arrays_exact = all(
                np.array_equal(read_arrays[name], value, equal_nan=True)
                for name, value in branch_arrays.items()
            )
        readback = {
            "state_rows_exact": len(read_states) == len(states),
            "intervention_rows_exact": len(read_interventions) == len(interventions),
            "branch_arrays_exact": arrays_exact,
            "registration_id_exact": json.loads(
                (destination / "manifest.json").read_text(encoding="utf-8")
            )["registration_id"]
            == registration["registration_id"],
        }
        readback["all_exact"] = bool(all(readback.values()))
        if not readback["all_exact"]:
            raise ValueError("GN1 scientific readback failed")
        _atomic_json(destination / "readback_audit.json", readback)
        write_checksums(destination)
    verify_checksums(output)
    _append_seal_notice(registration["registration_id"])


def _fixture_snapshot() -> Snapshot:
    composition = np.zeros(100, dtype=np.int64)
    composition[:20] = 2
    return Snapshot(
        composition=composition,
        generation=20,
        inheritance=(True, False, True, True),
        boundary_h=(0.95, 0.80, 0.93, 0.94),
        previous_growth_steps=17,
        cumulative_growth_steps=311,
    )


def _fixture_record(h: float, shift: int = 0) -> FissionRecord:
    parent = np.zeros(100, dtype=np.int64)
    daughter = np.zeros(100, dtype=np.int64)
    parent[shift % 100] = 80
    daughter[shift % 100] = 40
    return FissionRecord(parent=parent, daughter=daughter, h=h, growth_steps=1)


def _registered_seed_values() -> set[str]:
    values: set[str] = set()
    for root in (ROOT / "results", ROOT / "results_intervention_replication"):
        if not root.exists():
            continue
        for path in root.rglob("registration.json"):
            if DEFAULT_REGISTRATION in path.parents:
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            registry = payload.get("seed_registry", {})
            stack = [registry]
            while stack:
                item = stack.pop()
                if isinstance(item, dict):
                    stack.extend(item.values())
                elif isinstance(item, (list, tuple)):
                    stack.extend(item)
                elif isinstance(item, str):
                    values.add(item)
    return values


def validation_checks() -> dict[str, bool]:
    config = GardConfig(generations=5)
    rng = np.random.default_rng(
        derive_seed(SEEDS["validation"], f"{LABEL}.fixture")
    )
    beta = np.exp(rng.normal(-4.0, 4.0, size=(100, 100)))
    homogeneous = _homogeneous_beta(config)
    permutation = fixed_point_free_permutation(100, rng)
    shuffled = derange_beta(beta, permutation)
    snapshot = _fixture_snapshot()
    natural_case = NullCase(
        "fixture-natural",
        "NATURAL_GARD",
        "02",
        0,
        20,
        beta,
        beta,
        snapshot,
        0,
        permutation,
    )
    coupled_case = NullCase(
        "fixture-coupled",
        "COUPLING_DERANGED",
        "02",
        0,
        20,
        beta,
        shuffled,
        snapshot,
        0,
        permutation,
    )
    common_seed = _future_seed(natural_case, 3)
    natural_records, natural_completed = _simulate_case_future(
        natural_case, config, 4, np.random.default_rng(common_seed)
    )
    direct_records, direct_completed = simulate_future_absorbing(
        snapshot,
        beta,
        config,
        CANDIDATES["02"],
        4,
        np.random.default_rng(common_seed),
    )
    fission_record = _fission_only_advance(
        snapshot.composition,
        config,
        CANDIDATES["02"],
        np.random.default_rng(
            derive_seed(SEEDS["validation"], f"{LABEL}.fission")
        ),
    )
    rules = select_outgoing_rule_edits(snapshot.composition, beta)
    legal = set(enumerate_legal_edits(snapshot.composition))
    endpoint = evaluate_process(
        [
            _fixture_record(0.8),
            _fixture_record(0.91),
            _fixture_record(0.92),
            _fixture_record(0.93),
        ]
    )
    threshold_endpoint = evaluate_process(
        [
            _fixture_record(0.9),
            _fixture_record(np.nextafter(0.9, np.inf)),
            _fixture_record(np.nextafter(0.9, np.inf)),
            _fixture_record(np.nextafter(0.9, np.inf)),
        ]
    )
    predictor = FrozenFullPredictor.load(UPSTREAM_MODEL)
    state = state_graph_features(snapshot.composition, beta, config)
    history = history_features(snapshot, config)
    prediction = predictor.predict_features("02", state[None, :], history[None, :])
    manual = _manual_history_prediction(
        predictor.arrays, "02", history[None, :]
    )
    matrix_ids = np.repeat(np.arange(4), 5)
    values = np.arange(20, dtype=np.float64)
    grouped = _matrix_means(values, matrix_ids)
    protocol_value = protocol()
    checks = {
        "protocol_exact": protocol_value["cohort"]["fresh_matrices"] == 96
        and protocol_value["futures"]["untreated_f32_per_state"] == 64
        and protocol_value["futures"]["intervention_f12_per_arm"] == 32,
        "seed_domains_unique": len(set(SEEDS.values())) == len(SEEDS),
        "seed_domains_disjoint_from_prior_registrations": set(SEEDS.values()).isdisjoint(
            _registered_seed_values()
        ),
        "frozen_model_hash_exact": sha256_file(UPSTREAM_MODEL)
        == EXPECTED_FROZEN_MODEL_SHA256,
        "homogeneous_positive_constant_exact": np.all(
            homogeneous == HOMOGENEOUS_VALUE
        )
        and HOMOGENEOUS_VALUE == float(np.exp(4.0)),
        "homogeneous_boost_type_invariant": np.all(
            (homogeneous @ snapshot.composition) / snapshot.composition.sum()
            == HOMOGENEOUS_VALUE
        ),
        "coupling_permutation_fixed_point_free": np.unique(permutation).size == 100
        and not np.any(permutation == np.arange(100)),
        "coupling_entry_multiset_exact": np.array_equal(
            np.sort(beta, axis=None), np.sort(shuffled, axis=None)
        ),
        "coupling_frobenius_exact": np.allclose(
            np.linalg.norm(beta, ord="fro"),
            np.linalg.norm(shuffled, ord="fro"),
            rtol=2e-15,
            atol=0.0,
        ),
        "coupling_singular_values_preserved": np.allclose(
            np.linalg.svd(beta, compute_uv=False),
            np.linalg.svd(shuffled, compute_uv=False),
            rtol=1e-12,
            atol=1e-12,
        ),
        "coupling_applied_after_restoration": np.array_equal(
            natural_case.snapshot.composition, coupled_case.snapshot.composition
        )
        and natural_case.snapshot.inheritance == coupled_case.snapshot.inheritance
        and natural_case.snapshot.boundary_h == coupled_case.snapshot.boundary_h,
        "fission_only_parent_mass_exact": int(fission_record.parent.sum())
        == config.n_max,
        "fission_only_fixed_daughter_mass_exact": int(fission_record.daughter.sum())
        == config.n_min,
        "fission_only_nonnegative_integer": np.issubdtype(
            fission_record.daughter.dtype, np.integer
        )
        and bool(np.all(fission_record.daughter >= 0)),
        "natural_noop_plain_simulator_bitwise_exact": natural_completed
        == direct_completed
        and _record_digest(natural_records, natural_completed)
        == _record_digest(direct_records, direct_completed),
        "future_seed_excludes_mechanism": _future_seed(natural_case, 3)
        == _future_seed(coupled_case, 3),
        "future_branches_distinct": _future_seed(natural_case, 3)
        != _future_seed(natural_case, 4),
        "random_selection_stream_distinct": SEEDS["random_edit_selection"]
        != SEEDS["future_simulation"],
        "source_rule_edits_legal": rules["RULE_UP"] in legal
        and rules["RULE_DOWN"] in legal,
        "source_rule_orientation_exact": np.array_equal(
            snapshot.composition / snapshot.composition.sum() @ beta,
            beta.T @ (snapshot.composition / snapshot.composition.sum()),
        ),
        "instantaneous_edit_preserves_mass": all(
            int(apply_molecular_edit(snapshot.composition, edit).sum())
            == int(snapshot.composition.sum())
            for edit in rules.values()
        ),
        "f12_endpoint_fixture_exact": endpoint.joint_break_run3,
        "strict_threshold_fixture_exact": threshold_endpoint.joint_break_run3,
        "frozen_predictions_finite": prediction.shape == (1,)
        and manual.shape == (1,)
        and np.isfinite(prediction).all()
        and np.isfinite(manual).all(),
        "matrix_block_reduction_exact": np.array_equal(
            grouped, np.asarray([2.0, 7.0, 12.0, 17.0])
        ),
        "holm_family_sizes_exact": len(CANDIDATES) * (len(MECHANISMS) - 1) == 6
        and len(CANDIDATES) * len(MECHANISMS) == 8,
        "ordinary_brier_fixture_exact": np.allclose(
            _proper_brier(np.asarray([0.0, 1.0]), np.asarray([0.2, 0.8])),
            np.asarray([0.04, 0.04]),
        ),
        "scientific_paths_absent": not DEFAULT_OUTPUT.exists()
        and not DEFAULT_WORK.exists(),
        "strict_eight_prediction_not_targeted": "strict-eight prediction"
        in " ".join(protocol_value["separate_from"]),
        "no_omnibus_gate": not protocol_value["inference"]["omnibus_gate"],
    }
    return {name: bool(value) for name, value in checks.items()}


def validate(output: Path = DEFAULT_VALIDATION) -> None:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite GN1 validation: {output}")
    for path in (DEFAULT_REGISTRATION, DEFAULT_SMOKE, DEFAULT_OUTPUT, DEFAULT_WORK):
        if path.exists():
            raise FileExistsError(f"GN1 artifact exists before validation: {path}")
    checks = validation_checks()
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"GN1 validation checks failed: {failed}")
    command = [str(ROOT / ".venv/bin/python"), "-m", "pytest", "-q"]
    completed = subprocess.run(
        command, cwd=ROOT, check=False, capture_output=True, text=True
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "GN1 full repository validation failed\n"
            + completed.stdout
            + "\n"
            + completed.stderr
        )
    payload = {
        "format": VALIDATION_FORMAT,
        "checks": checks,
        "check_count": len(checks),
        "all_checks_passed": True,
        "pytest": {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        },
        "source_hashes": _source_hashes(),
        "source_tree_sha256": _canonical_digest(_source_hashes()),
        "protocol_id": protocol()["protocol_id"],
        "scientific_gn1_matrices_generated": 0,
        "scientific_gn1_futures_generated": 0,
    }
    with _atomic_destination(output) as destination:
        _atomic_json(destination / "validation.json", payload)
        write_checksums(destination)
    verify_checksums(output)


def _append_registration_notice(registration_id: str) -> None:
    text = LEDGER.read_text(encoding="utf-8").rstrip() + "\n"
    marker = f"<!-- registered-gn1-{registration_id} -->"
    if marker in text:
        return
    rows = [
        "",
        marker,
        "## GN1 catalytic-versus-geometric decomposition registered",
        "",
        f"- Registration: `{registration_id}`.",
        "- Ninety-six fresh matrices, both candidates, five landmarks, four mechanisms, F12/F32 outcomes, prediction comparisons, outgoing-rule transport, and complete replay were sealed before science.",
        "- The coupling derangement occurs only after natural landmark restoration and exactly preserves the beta weight multiset and network invariants.",
        "- GN1 has no omnibus pass gate and cannot rescue or replace any earlier result.",
        "- Scientific GN1 matrices and futures at registration: **0**.",
        "",
    ]
    LEDGER.write_text(text + "\n".join(rows), encoding="utf-8")


def register(
    validation_directory: Path = DEFAULT_VALIDATION,
    output: Path = DEFAULT_REGISTRATION,
) -> None:
    validation_directory = validation_directory.resolve()
    output = output.resolve()
    verify_checksums(validation_directory)
    validation = json.loads(
        (validation_directory / "validation.json").read_text(encoding="utf-8")
    )
    if not validation.get("all_checks_passed"):
        raise ValueError("GN1 validation did not pass")
    if validation.get("source_hashes") != _source_hashes():
        raise ValueError("GN1 source changed after validation")
    for path in (DEFAULT_SMOKE, DEFAULT_OUTPUT, DEFAULT_WORK):
        if path.exists():
            raise FileExistsError(f"GN1 scientific artifact exists: {path}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite GN1 registration: {output}")
    if sha256_file(UPSTREAM_MODEL) != EXPECTED_FROZEN_MODEL_SHA256:
        raise ValueError("GN1 frozen F12 predictor changed")
    payload: dict[str, Any] = {
        "format": REGISTRATION_FORMAT,
        "protocol_id": protocol()["protocol_id"],
        "source_hashes": _source_hashes(),
        "source_tree_sha256": _canonical_digest(_source_hashes()),
        "validation_checksum_manifest_sha256": sha256_file(
            validation_directory / "SHA256SUMS"
        ),
        "frozen_model_sha256": sha256_file(UPSTREAM_MODEL),
        "seed_registry": SEEDS,
        "scientific_gn1_matrices_at_registration": 0,
        "scientific_gn1_futures_at_registration": 0,
    }
    payload["registration_id"] = _canonical_digest(_json_ready(payload))
    with _atomic_destination(output) as destination:
        _atomic_json(destination / "protocol.json", protocol())
        _atomic_json(destination / "seed_registry.json", SEEDS)
        _atomic_json(destination / "registration.json", payload)
        shutil.copy2(ROOT / DOCUMENT, destination / "preregistration.md")
        shutil.copy2(UPSTREAM_MODEL, destination / "frozen_full_predictor.npz")
        shutil.copy2(
            validation_directory / "validation.json", destination / "validation.json"
        )
        write_checksums(destination)
    verify_registration(output)
    _append_registration_notice(payload["registration_id"])
    print(f"GN1 registered: {payload['registration_id']}", flush=True)


def verify_registration(directory: Path = DEFAULT_REGISTRATION) -> dict[str, Any]:
    directory = directory.resolve()
    verify_checksums(directory)
    payload = json.loads(
        (directory / "registration.json").read_text(encoding="utf-8")
    )
    if payload.get("format") != REGISTRATION_FORMAT:
        raise ValueError("invalid GN1 registration format")
    unsigned = dict(payload)
    registration_id = unsigned.pop("registration_id", None)
    if registration_id is None or _canonical_digest(_json_ready(unsigned)) != registration_id:
        raise ValueError("invalid GN1 registration ID")
    if payload["source_hashes"] != _source_hashes():
        raise ValueError("GN1 source changed after registration")
    if json.loads((directory / "protocol.json").read_text()) != protocol():
        raise ValueError("GN1 protocol changed after registration")
    if payload["seed_registry"] != SEEDS:
        raise ValueError("GN1 seed registry changed")
    model = directory / "frozen_full_predictor.npz"
    if sha256_file(model) != EXPECTED_FROZEN_MODEL_SHA256:
        raise ValueError("GN1 registered predictor changed")
    return payload


def smoke(
    registration_directory: Path = DEFAULT_REGISTRATION,
    output: Path = DEFAULT_SMOKE,
) -> None:
    registration = verify_registration(registration_directory)
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite GN1 smoke: {output}")
    if DEFAULT_OUTPUT.exists() or DEFAULT_WORK.exists():
        raise FileExistsError("GN1 scientific artifact exists before smoke")
    config = GardConfig(generations=5)
    rng = np.random.default_rng(
        derive_seed(SEEDS["smoke"], f"{LABEL}.artificial")
    )
    beta = np.exp(rng.normal(-2.0, 1.0, size=(100, 100)))
    permutation = fixed_point_free_permutation(100, rng)
    snapshot = _fixture_snapshot()
    mechanisms = {
        "NATURAL_GARD": beta,
        "HOMOGENEOUS_GENERATIVE": _homogeneous_beta(config),
        "COUPLING_DERANGED": derange_beta(beta, permutation),
    }
    checks: dict[str, bool] = {
        "registration_verified": True,
        "artificial_non_scientific_fixture": True,
        "no_rates_or_effect_directions_disclosed": True,
        "all_four_mechanisms_exercised": False,
        "replay_exact": True,
        "scientific_gn1_matrices_generated_is_zero": True,
    }
    count = 0
    for mechanism, active in mechanisms.items():
        case = NullCase(
            f"smoke-{mechanism}",
            mechanism,
            "02",
            -1,
            5,
            beta,
            active,
            snapshot,
            0,
            permutation,
        )
        seed = derive_seed(SEEDS["smoke"], f"{LABEL}.future", mechanism)
        left = _simulate_case_future(case, config, 3, np.random.default_rng(seed))
        right = _simulate_case_future(case, config, 3, np.random.default_rng(seed))
        checks["replay_exact"] = checks["replay_exact"] and (
            _record_digest(*left) == _record_digest(*right)
        )
        count += 1
    fission_case = NullCase(
        "smoke-fission",
        "FISSION_ONLY_GENERATIVE",
        "02",
        -1,
        5,
        beta,
        _homogeneous_beta(config),
        snapshot,
        0,
        permutation,
    )
    seed = derive_seed(SEEDS["smoke"], f"{LABEL}.future", "fission")
    left = _simulate_case_future(fission_case, config, 3, np.random.default_rng(seed))
    right = _simulate_case_future(fission_case, config, 3, np.random.default_rng(seed))
    checks["replay_exact"] = checks["replay_exact"] and (
        _record_digest(*left) == _record_digest(*right)
    )
    count += 1
    checks["all_four_mechanisms_exercised"] = count == 4
    checks["all_checks_passed"] = bool(all(checks.values()))
    if not checks["all_checks_passed"]:
        raise RuntimeError("GN1 smoke failed")
    with _atomic_destination(output) as destination:
        _atomic_json(
            destination / "smoke.json",
            {
                "format": "codex-generative-null-smoke-v1",
                "registration_id": registration["registration_id"],
                "checks": checks,
                "all_checks_passed": True,
            },
        )
        write_checksums(destination)
    verify_checksums(output)


def run(
    registration_directory: Path = DEFAULT_REGISTRATION,
    output: Path = DEFAULT_OUTPUT,
    work: Path = DEFAULT_WORK,
    workers: int = min(os.cpu_count() or 1, 14),
) -> None:
    registration = verify_registration(registration_directory)
    if not DEFAULT_SMOKE.exists():
        raise FileNotFoundError("GN1 smoke must pass before scientific generation")
    verify_checksums(DEFAULT_SMOKE)
    smoke_payload = json.loads((DEFAULT_SMOKE / "smoke.json").read_text())
    if not smoke_payload.get("all_checks_passed"):
        raise ValueError("GN1 smoke no longer passes")
    if smoke_payload.get("registration_id") != registration["registration_id"]:
        raise ValueError("GN1 smoke belongs to a different registration")
    output = output.resolve()
    work = work.resolve()
    if output.exists():
        raise FileExistsError(f"GN1 result already exists: {output}")
    if workers < 1:
        raise ValueError("workers must be positive")
    free = shutil.disk_usage(ROOT).free
    if free < MINIMUM_FREE_DISK_BYTES:
        raise RuntimeError(
            f"GN1 requires {MINIMUM_FREE_DISK_BYTES} free bytes; found {free}"
        )
    work.mkdir(parents=True, exist_ok=True)
    config = GardConfig()
    _write_status(
        work,
        "building_null_launch_states",
        0,
        MATRICES * len(CANDIDATES) * len(LANDMARKS) * len(MECHANISMS),
    )
    print(
        f"[gn1 1/7] Building {MATRICES} fresh matrices and four null mechanisms",
        flush=True,
    )
    cases = build_cases(config=config)
    expected = MATRICES * len(CANDIDATES) * len(LANDMARKS) * len(MECHANISMS)
    if len(cases) != expected:
        raise AssertionError("GN1 case count changed")
    _write_status(work, "building_null_launch_states", expected, expected)
    print(f"[gn1 2/7] Running {len(cases):,} state batches", flush=True)
    batches = run_checkpointed_batches(
        cases,
        config,
        registration["registration_id"],
        work / "generate",
        work,
        "scientific_generate",
        workers,
    )
    print("[gn1 3/7] Replaying every state batch", flush=True)
    replay = run_checkpointed_batches(
        cases,
        config,
        registration["registration_id"],
        work / "replay",
        work,
        "scientific_replay",
        workers,
    )
    if any(
        _batch_digest(left) != _batch_digest(right)
        for left, right in zip(batches, replay, strict=True)
    ):
        raise ValueError("GN1 complete replay failed")
    print("[gn1 4/7] Scoring frozen and clean cross-fitted predictors", flush=True)
    model_path = registration_directory / "frozen_full_predictor.npz"
    frozen_direct, frozen_full, frozen_outside = frozen_predictions(
        cases, model_path, config
    )
    crossfit_direct, crossfit_state, crossfit_audit = crossfit_state_predictions(
        cases, batches, config
    )
    states, interventions, edits = build_tables(
        cases,
        batches,
        frozen_direct,
        frozen_full,
        frozen_outside,
        crossfit_direct,
        crossfit_state,
    )
    print("[gn1 5/7] Computing whole-matrix decomposition inference", flush=True)
    metrics, inference_arrays = compute_inference(states, interventions)
    print("[gn1 6/7] Writing and reading back scientific artifacts", flush=True)
    write_result(
        output,
        registration,
        cases,
        batches,
        replay,
        states,
        interventions,
        edits,
        metrics,
        inference_arrays,
        crossfit_audit,
    )
    _write_status(
        work,
        "sealed_complete_final_stop",
        len(cases) * 2,
        len(cases) * 2,
        integrity_passed=True,
        output=str(output),
        registration_id=registration["registration_id"],
    )
    print("[gn1 7/7] Result sealed; FINAL STOP", flush=True)


def read_status(work: Path = DEFAULT_WORK) -> dict[str, Any]:
    path = work.resolve() / "campaign_status.json"
    if not path.exists():
        raise FileNotFoundError(f"GN1 status does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Prospective catalytic-versus-geometric GN1 decomposition"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--output", type=Path, default=DEFAULT_VALIDATION)
    register_parser = subparsers.add_parser("register")
    register_parser.add_argument(
        "--validation", type=Path, default=DEFAULT_VALIDATION
    )
    register_parser.add_argument("--output", type=Path, default=DEFAULT_REGISTRATION)
    smoke_parser = subparsers.add_parser("smoke")
    smoke_parser.add_argument(
        "--registration", type=Path, default=DEFAULT_REGISTRATION
    )
    smoke_parser.add_argument("--output", type=Path, default=DEFAULT_SMOKE)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument(
        "--registration", type=Path, default=DEFAULT_REGISTRATION
    )
    run_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run_parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    run_parser.add_argument(
        "--workers", type=int, default=min(os.cpu_count() or 1, 14)
    )
    status_parser = subparsers.add_parser("status")
    status_parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    arguments = parser.parse_args(argv)
    if arguments.command == "validate":
        validate(arguments.output)
    elif arguments.command == "register":
        register(arguments.validation, arguments.output)
    elif arguments.command == "smoke":
        smoke(arguments.registration, arguments.output)
    elif arguments.command == "run":
        run(arguments.registration, arguments.output, arguments.work_dir, arguments.workers)
    elif arguments.command == "status":
        print(json.dumps(read_status(arguments.work_dir), indent=2, sort_keys=True))
    else:  # pragma: no cover
        raise AssertionError(arguments.command)


if __name__ == "__main__":
    main()
