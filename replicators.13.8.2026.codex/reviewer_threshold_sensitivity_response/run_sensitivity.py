from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Sequence

# Keep runtime caches within the task's isolated write boundary and allow both
# ``python -m ...`` and direct-script invocation from the repository root.
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent / "artifacts" / "matplotlib"),
)
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from plastic_heredity.config import CANDIDATES, CohortConfig, ExperimentConfig
from plastic_heredity.episode_coherence import _experiment_from_manifest
from plastic_heredity.experiment import PROCESS_COLUMNS, StateCase
from plastic_heredity.intervention_core import MolecularEdit, simulate_one_shot
from plastic_heredity.intervention_cr1_confirmation import (
    LABEL as CR1_LABEL,
    SEEDS as CR1_SEEDS,
    experiment as cr1_experiment,
    phase_spec as cr1_phase_spec,
)
from plastic_heredity.intervention_replication import _future_seed as cr1_future_seed
from plastic_heredity.mechanistic import verify_checksums, write_checksums
from plastic_heredity.processes import evaluate_process
from plastic_heredity.regime_confirmation import (
    CONFIRMATION_MASTER_SEED as REGIME_CONFIRMATION_MASTER_SEED,
    ENDPOINTS as REGIME_ENDPOINTS,
    GEOMETRY_COLUMNS as REGIME_GEOMETRY_COLUMNS,
    _experiment as regime_experiment,
    evaluate_regime,
)
from plastic_heredity.seeds import derive_seed
from plastic_heredity.simulator import (
    SimulationError,
    cosine_similarity,
    generate_beta,
    generate_initial_composition,
    simulate_future_absorbing,
    simulate_lineage,
)

from reviewer_threshold_sensitivity_response.sensitivity_core import (
    BOOTSTRAP_REPETITIONS,
    F12_BASELINE,
    F12_DEFINITIONS,
    F32_BASELINE,
    F32_DEFINITIONS,
    REFERENCE_MASTER_SEED,
    SENSITIVITY_MASTER_SEED,
    atomic_npz,
    canonical_digest,
    dominant_h_component_centroid,
    f12_definition_table,
    f32_definition_table,
    reference_summary,
    score_f12_grid,
    score_f32_grid,
    sha256_file,
    summarize_cr1_grid,
    summarize_prediction_grid,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = Path(__file__).resolve().parent
ARTIFACT_ROOT = TASK_ROOT / "artifacts"
PROTOCOL_ROOT = ARTIFACT_ROOT / "protocol"
WORK_ROOT = ARTIFACT_ROOT / "work"
REPLAY_ROOT = ARTIFACT_ROOT / "replays"
OUTPUT_ROOT = ARTIFACT_ROOT / "output"

F12_SOURCE = REPOSITORY_ROOT / "results/scaled5"
F32_SOURCE = REPOSITORY_ROOT / "results/regime_confirmation"
F32_ENSEMBLE_SOURCE = REPOSITORY_ROOT / "results/regime_ensemble_confirmation"
CR1_SOURCE = (
    REPOSITORY_ROOT
    / "results_intervention_replication/cr1_model_guided_confirmation"
)
CR1_REGISTRATION = (
    REPOSITORY_ROOT / "results_intervention_replication/cr1_confirmation_registration"
)
L36_CODE_ROOT = Path(
    "/home/robert/Projects/replications/PlasticHeredity/"
    "original.1.8.2026.eidosoma-ai-scientist.code/"
    "arrival-of-self-replicators-eidosoma-groups-42"
)
L36_REPORT_ROOT = Path(
    "/home/robert/Projects/replications/PlasticHeredity/"
    "original.1.8.2026.eidosoma-ai-scientist.stepReports/"
    "artifacts/research_steps/S19/loops/L36"
)
L36_CONFIG = L36_CODE_ROOT / "configs/e01/s19_l36_independent_lineage_basin_transfer.yaml"
L36_SCRIPT = L36_CODE_ROOT / "scripts/e01/run_s19_l36_independent_lineage_basin_transfer.py"
L36_REPORT = L36_REPORT_ROOT / "S19_L36_FULL_RESULTS.md"
L36_CENTROID_SOURCE = (
    L36_CODE_ROOT / "src/e01_onset_discovery/empirical_committor.py"
)
L37_CONFIG = L36_CODE_ROOT / "configs/e01/s19_l37_multilineage_any_attractor.yaml"
L37_SCRIPT = L36_CODE_ROOT / "scripts/e01/run_s19_l37_multilineage_any_attractor.py"
L37_REPORT = L36_REPORT_ROOT.parent / "L37/S19_L37_FULL_RESULTS.md"

F12_MAX_HORIZON = 16
F32_HORIZON = 32
REFERENCE_HORIZON = 100
CR1_ARMS = ("MODEL_UP", "MODEL_DOWN", "RANDOM", "NOOP")
CHECKPOINT_FORMAT = "threshold-sensitivity-checkpoint-v1"
RESULT_FORMAT = "threshold-sensitivity-appendix-v1"
_FROZEN_PROTOCOL_ID: str | None = None


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


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _source_contract() -> dict[str, Any]:
    sources = {
        "task_runner": Path(__file__),
        "task_core": TASK_ROOT / "sensitivity_core.py",
        "f12_checksum_manifest": F12_SOURCE / "SHA256SUMS",
        "f12_manifest": F12_SOURCE / "manifest.json",
        "f32_checksum_manifest": F32_SOURCE / "SHA256SUMS",
        "f32_manifest": F32_SOURCE / "manifest.json",
        "f32_ensemble_checksum_manifest": F32_ENSEMBLE_SOURCE / "SHA256SUMS",
        "cr1_checksum_manifest": CR1_SOURCE / "SHA256SUMS",
        "cr1_manifest": CR1_SOURCE / "manifest.json",
        "cr1_registration_checksum_manifest": CR1_REGISTRATION / "SHA256SUMS",
        "l36_frozen_config": L36_CONFIG,
        "l36_frozen_script": L36_SCRIPT,
        "l36_frozen_centroid_source": L36_CENTROID_SOURCE,
        "l36_report": L36_REPORT,
        "l37_frozen_config": L37_CONFIG,
        "l37_frozen_script": L37_SCRIPT,
        "l37_report": L37_REPORT,
    }
    missing = [str(path) for path in sources.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing required read-only sources: {missing}")
    return {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for name, path in sources.items()
    }


def _verify_source_archives() -> dict[str, int]:
    archives = {
        "f12_scaled5": F12_SOURCE,
        "f32_regime_confirmation": F32_SOURCE,
        "f32_ensemble_confirmation": F32_ENSEMBLE_SOURCE,
        "cr1_confirmation": CR1_SOURCE,
        "cr1_registration": CR1_REGISTRATION,
    }
    return {
        name: len(verify_checksums(directory)) for name, directory in archives.items()
    }


def _protocol() -> dict[str, Any]:
    value: dict[str, Any] = {
        "format": "threshold-sensitivity-exploratory-protocol-v1",
        "status": "post_hoc_exploratory_sensitivity_frozen_before_grid_readout",
        "working_boundary": {
            "all_new_writes_below": str(TASK_ROOT.resolve()),
            "existing_and_external_sources_read_only": True,
            "source_manuscript_modified": False,
        },
        "f12_grid": f12_definition_table().to_dict("records"),
        "f32_grid": f32_definition_table().to_dict("records"),
        "models": {
            "f12": "archived scaled5 prediction_full versus prediction_history",
            "f32": "archived REGCONF prediction_primary_all8_h10_state versus prediction_primary_all8_h10",
            "f32_ensemble": "existing baseline result quoted secondarily; not rescored or used to rescue REGCONF",
            "refitting_or_recalibration": False,
            "threshold_dependent_history_recomputed": False,
        },
        "intervention": {
            "program": "CR1 model-guided confirmation only",
            "arms": CR1_ARMS,
            "contrasts": [
                "MODEL_UP-MODEL_DOWN",
                "MODEL_UP-NOOP",
                "NOOP-MODEL_DOWN",
                "RANDOM-NOOP",
            ],
            "common_random_streams_preserved": True,
            "f16_is_deterministic_extension": True,
        },
        "independent_lineage_reference": {
            "method": "L36 two independently seeded same-beta same-initial-state lineages; H090 dominant-component centroid per lineage; cosine H between centroids",
            "application": "new contextual reference on scaled5 restored states",
            "horizon": REFERENCE_HORIZON,
            "roles": ["REFERENCE_A", "REFERENCE_B"],
            "historical_raw_l36_parquets_available": False,
            "classification": "new_contextual_simulation_not_historical_l36_replay",
            "master_seed": REFERENCE_MASTER_SEED,
        },
        "inference": {
            "independent_unit": "catalytic matrix",
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "bootstrap_master_seed": SENSITIVITY_MASTER_SEED,
            "candidates_separate": True,
            "branch_halves_separate": True,
            "all_grid_cells_reported": True,
            "multiplicity_status": "descriptive exploratory intervals; no cellwise confirmatory claims",
        },
        "inequalities": {
            "inheritance": "strict H > threshold",
            "break": "inclusive H <= threshold",
            "strict_pairwise_coherence": "strict H > threshold",
            "old_anchor_distinctness": "inclusive H <= threshold",
        },
        "source_contract": _source_contract(),
        "claim_boundary": (
            "No alternate definition is confirmatory, no favorable cell changes a "
            "registered verdict, and F12 intervention effects do not establish "
            "control of the strict F32 event."
        ),
    }
    value["protocol_id"] = canonical_digest(value)
    return _json_ready(value)


def prepare() -> None:
    _verify_source_archives()
    protocol = _protocol()
    PROTOCOL_ROOT.mkdir(parents=True, exist_ok=True)
    path = PROTOCOL_ROOT / "analysis_protocol.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != protocol:
            raise ValueError("existing frozen protocol differs from current contract")
        print(f"Protocol already frozen and identical: {path}", flush=True)
        return
    _write_json(path, protocol)
    f12_definition_table().to_csv(PROTOCOL_ROOT / "f12_definitions.csv", index=False)
    f32_definition_table().to_csv(PROTOCOL_ROOT / "f32_definitions.csv", index=False)
    write_checksums(PROTOCOL_ROOT)
    print(f"Frozen exploratory protocol: {path}", flush=True)


def verify_protocol() -> dict[str, Any]:
    verify_checksums(PROTOCOL_ROOT)
    source_archive_files = _verify_source_archives()
    saved = json.loads(
        (PROTOCOL_ROOT / "analysis_protocol.json").read_text(encoding="utf-8")
    )
    current = _protocol()
    if saved != current:
        raise ValueError("frozen sensitivity protocol or an input identity changed")
    return {
        "protocol_id": saved["protocol_id"],
        "source_contract_current": True,
        "checksums_valid": True,
        "source_archive_files_verified": source_archive_files,
    }


def _saved_protocol_id() -> str:
    global _FROZEN_PROTOCOL_ID
    if _FROZEN_PROTOCOL_ID is None:
        protocol_path = PROTOCOL_ROOT / "analysis_protocol.json"
        if not protocol_path.is_file():
            raise FileNotFoundError("run prepare before creating replay checkpoints")
        _FROZEN_PROTOCOL_ID = str(
            json.loads(protocol_path.read_text(encoding="utf-8"))["protocol_id"]
        )
    return _FROZEN_PROTOCOL_ID


def _checkpoint_path(dataset: str, index: int) -> Path:
    return WORK_ROOT / dataset / f"state_{index:04d}.npz"


def _checkpoint_complete(path: Path, dataset: str, index: int) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as archive:
            return (
                str(archive["format"].item()) == CHECKPOINT_FORMAT
                and str(archive["dataset"].item()) == dataset
                and int(archive["state_index"].item()) == index
                and str(archive["protocol_id"].item()) == _saved_protocol_id()
            )
    except Exception:
        return False


def _f12_worker(arguments: tuple[int, StateCase, ExperimentConfig]) -> int:
    index, case, experiment = arguments
    output = _checkpoint_path("f12", index)
    if _checkpoint_complete(output, "f12", index):
        return index
    branches = experiment.confirmation.branches_per_state
    boundary_h = np.full((branches, F12_MAX_HORIZON), np.nan, dtype=np.float64)
    labels = np.zeros((branches, len(F12_DEFINITIONS)), dtype=np.int8)
    baseline_process = np.full(
        (branches, len(PROCESS_COLUMNS)), np.nan, dtype=np.float64
    )
    baseline_completed = np.zeros(branches, dtype=np.int8)
    with threadpool_limits(limits=1):
        for branch in range(branches):
            rng = np.random.default_rng(
                derive_seed(
                    experiment.master_seed,
                    f"{case.cohort}.future",
                    case.candidate,
                    case.matrix_id,
                    case.landmark,
                    branch,
                )
            )
            records, _ = simulate_future_absorbing(
                case.snapshot,
                case.beta,
                experiment.gard,
                CANDIDATES[case.candidate],
                F12_MAX_HORIZON,
                rng,
            )
            for position, record in enumerate(records):
                boundary_h[branch, position] = record.h
            labels[branch] = score_f12_grid(boundary_h[branch])
            first_twelve = records[:12]
            outcome = evaluate_process(
                first_twelve, experiment.gard.inheritance_threshold
            )
            values = outcome.to_dict()
            baseline_process[branch] = [
                float(values[name]) for name in PROCESS_COLUMNS
            ]
            baseline_completed[branch] = int(len(first_twelve) == 12)
    atomic_npz(
        output,
        format=np.asarray(CHECKPOINT_FORMAT),
        dataset=np.asarray("f12"),
        state_index=np.asarray(index, dtype=np.int32),
        protocol_id=np.asarray(_saved_protocol_id()),
        boundary_h=boundary_h,
        labels=labels,
        baseline_process=baseline_process,
        baseline_completed=baseline_completed,
    )
    return index


def _f32_worker(arguments: tuple[int, StateCase, ExperimentConfig]) -> int:
    index, case, experiment = arguments
    output = _checkpoint_path("f32", index)
    if _checkpoint_complete(output, "f32", index):
        return index
    branches = experiment.confirmation.branches_per_state
    boundary_h = np.full((branches, F32_HORIZON), np.nan, dtype=np.float64)
    labels = np.zeros((branches, len(F32_DEFINITIONS)), dtype=np.int8)
    onsets = np.full((branches, len(F32_DEFINITIONS)), -1, dtype=np.int16)
    baseline_targets = np.zeros((branches, len(REGIME_ENDPOINTS)), dtype=np.int8)
    baseline_onsets = np.full(
        (branches, len(REGIME_ENDPOINTS)), -1, dtype=np.int16
    )
    baseline_completed = np.zeros(branches, dtype=np.int8)
    baseline_observed = np.zeros(branches, dtype=np.int16)
    baseline_first_break = np.full(branches, -1, dtype=np.int16)
    baseline_first_run = np.full(branches, -1, dtype=np.int16)
    baseline_geometry = np.full(
        (branches, len(REGIME_GEOMETRY_COLUMNS)), np.nan, dtype=np.float64
    )
    with threadpool_limits(limits=1):
        for branch in range(branches):
            rng = np.random.default_rng(
                derive_seed(
                    experiment.master_seed,
                    f"{case.cohort}.future",
                    case.candidate,
                    case.matrix_id,
                    case.landmark,
                    branch,
                )
            )
            records, complete = simulate_future_absorbing(
                case.snapshot,
                case.beta,
                experiment.gard,
                CANDIDATES[case.candidate],
                F32_HORIZON,
                rng,
            )
            for position, record in enumerate(records):
                boundary_h[branch, position] = record.h
            labels[branch], onsets[branch] = score_f32_grid(records)
            baseline = evaluate_regime(records)
            values = baseline.to_dict()
            baseline_targets[branch] = [
                int(values[name]) for name in REGIME_ENDPOINTS
            ]
            baseline_onsets[branch] = [
                int(values[f"{name}_onset"]) for name in REGIME_ENDPOINTS
            ]
            baseline_completed[branch] = int(complete)
            baseline_observed[branch] = len(records)
            baseline_first_break[branch] = baseline.first_break_index
            baseline_first_run[branch] = baseline.first_run8_start
            baseline_geometry[branch] = [
                float(values[name]) for name in REGIME_GEOMETRY_COLUMNS
            ]
    atomic_npz(
        output,
        format=np.asarray(CHECKPOINT_FORMAT),
        dataset=np.asarray("f32"),
        state_index=np.asarray(index, dtype=np.int32),
        protocol_id=np.asarray(_saved_protocol_id()),
        boundary_h=boundary_h,
        labels=labels,
        onsets=onsets,
        baseline_targets=baseline_targets,
        baseline_onsets=baseline_onsets,
        baseline_completed=baseline_completed,
        baseline_observed=baseline_observed,
        baseline_first_break=baseline_first_break,
        baseline_first_run=baseline_first_run,
        baseline_geometry=baseline_geometry,
    )
    return index


def _cr1_edits() -> dict[tuple[str, str], MolecularEdit | None]:
    table = pd.read_csv(
        CR1_SOURCE / "selected_interventions.csv", dtype={"candidate": str}
    )
    output: dict[tuple[str, str], MolecularEdit | None] = {}
    for row in table.itertuples(index=False):
        key = (str(row.state_id), str(row.arm))
        output[key] = (
            None
            if bool(row.is_noop)
            else MolecularEdit(int(row.remove_type), int(row.add_type))
        )
    return output


def _cr1_worker(
    arguments: tuple[
        int,
        StateCase,
        ExperimentConfig,
        tuple[MolecularEdit | None, ...],
    ]
) -> int:
    index, case, experiment, edits = arguments
    output = _checkpoint_path("cr1", index)
    if _checkpoint_complete(output, "cr1", index):
        return index
    spec = cr1_phase_spec()
    boundary_h = np.full(
        (len(CR1_ARMS), spec.branches, F12_MAX_HORIZON),
        np.nan,
        dtype=np.float64,
    )
    labels = np.zeros(
        (len(CR1_ARMS), spec.branches, len(F12_DEFINITIONS)), dtype=np.int8
    )
    with threadpool_limits(limits=1):
        for branch in range(spec.branches):
            seed = cr1_future_seed(spec, case, branch)
            for arm_index, edit in enumerate(edits):
                outcome = simulate_one_shot(
                    case.snapshot,
                    case.beta,
                    case.candidate,
                    experiment.gard,
                    F12_MAX_HORIZON,
                    np.random.default_rng(seed),
                    edit,
                )
                boundary_h[arm_index, branch] = outcome.boundary_h
                labels[arm_index, branch] = score_f12_grid(outcome.boundary_h)
    atomic_npz(
        output,
        format=np.asarray(CHECKPOINT_FORMAT),
        dataset=np.asarray("cr1"),
        state_index=np.asarray(index, dtype=np.int32),
        protocol_id=np.asarray(_saved_protocol_id()),
        boundary_h=boundary_h,
        labels=labels,
    )
    return index


def _reference_worker(arguments: tuple[int, StateCase, ExperimentConfig]) -> int:
    index, case, experiment = arguments
    output = _checkpoint_path("reference", index)
    if _checkpoint_complete(output, "reference", index):
        return index
    records_by_role: list[list[Any]] = []
    completed: list[int] = []
    with threadpool_limits(limits=1):
        for role in ("REFERENCE_A", "REFERENCE_B"):
            rng = np.random.default_rng(
                derive_seed(
                    REFERENCE_MASTER_SEED,
                    "l36_method_reference",
                    case.candidate,
                    case.matrix_id,
                    case.landmark,
                    role,
                )
            )
            records, complete = simulate_future_absorbing(
                case.snapshot,
                case.beta,
                experiment.gard,
                CANDIDATES[case.candidate],
                REFERENCE_HORIZON,
                rng,
            )
            records_by_role.append(records)
            completed.append(int(complete))
    within_boundary_h = np.full((2, REFERENCE_HORIZON), np.nan, dtype=np.float64)
    between_lineage_centroid_h = np.asarray(np.nan, dtype=np.float64)
    component_sizes = np.zeros(2, dtype=np.int16)
    for role_index, records in enumerate(records_by_role):
        for position, record in enumerate(records):
            within_boundary_h[role_index, position] = record.h
    if all(value == 1 for value in completed):
        centroids: list[np.ndarray] = []
        for role_index, records in enumerate(records_by_role):
            states = np.vstack([record.daughter for record in records])
            centroid, members = dominant_h_component_centroid(states)
            centroids.append(centroid)
            component_sizes[role_index] = len(members)
        between_lineage_centroid_h = np.asarray(
            cosine_similarity(centroids[0], centroids[1]), dtype=np.float64
        )
    atomic_npz(
        output,
        format=np.asarray(CHECKPOINT_FORMAT),
        dataset=np.asarray("reference"),
        state_index=np.asarray(index, dtype=np.int32),
        protocol_id=np.asarray(_saved_protocol_id()),
        within_boundary_h=within_boundary_h,
        between_lineage_centroid_h=between_lineage_centroid_h,
        component_sizes=component_sizes,
        completed=np.asarray(completed, dtype=np.int8),
    )
    return index


def _run_workers(
    dataset: str,
    arguments: Sequence[Any],
    worker: Callable[[Any], int],
    workers: int,
) -> None:
    directory = WORK_ROOT / dataset
    directory.mkdir(parents=True, exist_ok=True)
    pending = [
        item
        for item in arguments
        if not _checkpoint_complete(_checkpoint_path(dataset, int(item[0])), dataset, int(item[0]))
    ]
    print(
        f"[{dataset}] {len(arguments) - len(pending)}/{len(arguments)} checkpoints already complete; {len(pending)} pending",
        flush=True,
    )
    if not pending:
        return
    progress_every = max(1, len(pending) // 40)
    if workers <= 1:
        for count, item in enumerate(pending, start=1):
            worker(item)
            if count % progress_every == 0 or count == len(pending):
                print(f"[{dataset}] completed {count}/{len(pending)} pending states", flush=True)
        return
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(worker, item) for item in pending]
        for count, future in enumerate(as_completed(futures), start=1):
            future.result()
            if count % progress_every == 0 or count == len(pending):
                print(f"[{dataset}] completed {count}/{len(pending)} pending states", flush=True)


def _cohort_matrix_worker(
    arguments: tuple[ExperimentConfig, str, CohortConfig, int]
) -> list[StateCase]:
    experiment, cohort_name, cohort, matrix_id = arguments
    with threadpool_limits(limits=1):
        beta_rng = np.random.default_rng(
            derive_seed(experiment.master_seed, f"{cohort_name}.beta", matrix_id)
        )
        initial_rng = np.random.default_rng(
            derive_seed(experiment.master_seed, f"{cohort_name}.initial", matrix_id)
        )
        beta = generate_beta(experiment.gard, beta_rng)
        initial = generate_initial_composition(experiment.gard, initial_rng)
        cases: list[StateCase] = []
        for candidate, contract in CANDIDATES.items():
            lineage = None
            for attempt in range(100):
                path_rng = np.random.default_rng(
                    derive_seed(
                        experiment.master_seed,
                        f"{cohort_name}.main_path",
                        candidate,
                        matrix_id,
                        attempt,
                    )
                )
                try:
                    lineage = simulate_lineage(
                        initial, beta, experiment.gard, contract, path_rng
                    )
                    break
                except SimulationError:
                    continue
            if lineage is None:
                raise SimulationError(
                    f"failed to obtain a complete {cohort_name} trajectory for "
                    f"candidate {candidate}, matrix {matrix_id} in 100 attempts"
                )
            by_generation = {snapshot.generation: snapshot for snapshot in lineage}
            for landmark in cohort.landmarks:
                cases.append(
                    StateCase(
                        state_id=(
                            f"{cohort_name}-c{candidate}-m{matrix_id:03d}"
                            f"-g{landmark:03d}"
                        ),
                        cohort=cohort_name,
                        candidate=candidate,
                        matrix_id=matrix_id,
                        landmark=landmark,
                        beta=beta,
                        snapshot=by_generation[landmark],
                    )
                )
        return cases


def _build_cohort_parallel(
    experiment: ExperimentConfig,
    cohort_name: str,
    cohort: CohortConfig,
    workers: int,
) -> list[StateCase]:
    arguments = [
        (experiment, cohort_name, cohort, matrix_id)
        for matrix_id in range(cohort.matrices)
    ]
    if workers <= 1:
        batches = [_cohort_matrix_worker(item) for item in arguments]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            batches = list(
                executor.map(_cohort_matrix_worker, arguments, chunksize=1)
            )
    return [case for batch in batches for case in batch]


def _f12_cases(workers: int) -> tuple[ExperimentConfig, list[StateCase]]:
    manifest = json.loads((F12_SOURCE / "manifest.json").read_text(encoding="utf-8"))
    experiment = _experiment_from_manifest(manifest)
    cases = _build_cohort_parallel(
        experiment, "CONF", experiment.confirmation, workers
    )
    return experiment, cases


def _f32_cases(workers: int) -> tuple[ExperimentConfig, list[StateCase]]:
    experiment = regime_experiment(REGIME_CONFIRMATION_MASTER_SEED)
    cases = _build_cohort_parallel(
        experiment, "REGCONF", experiment.confirmation, workers
    )
    return experiment, cases


def _cr1_cases(workers: int) -> tuple[ExperimentConfig, list[StateCase]]:
    experiment = cr1_experiment()
    cases = _build_cohort_parallel(
        experiment, CR1_LABEL, experiment.confirmation, workers
    )
    return experiment, cases


def _metadata(cases: Sequence[StateCase]) -> dict[str, np.ndarray]:
    return {
        "state_ids": np.asarray([case.state_id for case in cases]),
        "candidates": np.asarray([case.candidate for case in cases]),
        "matrix_ids": np.asarray([case.matrix_id for case in cases], dtype=np.int16),
        "landmarks": np.asarray([case.landmark for case in cases], dtype=np.int16),
    }


def _load_checkpoints(
    dataset: str,
    count: int,
    keys: Sequence[str],
) -> dict[str, np.ndarray]:
    output: dict[str, list[np.ndarray]] = {key: [] for key in keys}
    for index in range(count):
        path = _checkpoint_path(dataset, index)
        if not _checkpoint_complete(path, dataset, index):
            raise ValueError(f"missing or invalid {dataset} checkpoint {index}")
        with np.load(path, allow_pickle=False) as archive:
            for key in keys:
                output[key].append(np.asarray(archive[key]))
    return {key: np.stack(values) for key, values in output.items()}


def _f12_baseline_audit(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    baseline_index = next(
        index for index, definition in enumerate(F12_DEFINITIONS) if definition.is_baseline
    )
    with np.load(F12_SOURCE / "analysis_arrays.npz", allow_pickle=False) as archive:
        expected_labels = np.asarray(archive["confirmation_targets"], dtype=np.int8)
    observed_labels = arrays["labels"][:, :, baseline_index]
    branches = pd.read_csv(
        F12_SOURCE / "confirmation_branches.csv.gz", dtype={"candidate": str}
    )
    expected_completed = branches["completed_horizon"].to_numpy(dtype=np.int8).reshape(
        observed_labels.shape
    )
    expected_process = branches.loc[:, PROCESS_COLUMNS].to_numpy(dtype=np.float64).reshape(
        arrays["baseline_process"].shape
    )
    process_error = np.nanmax(
        np.abs(expected_process - arrays["baseline_process"]), initial=0.0
    )
    audit = {
        "baseline_definition": F12_BASELINE,
        "labels_exact": bool(np.array_equal(expected_labels, observed_labels)),
        "completed_horizon_exact": bool(
            np.array_equal(expected_completed, arrays["baseline_completed"])
        ),
        "process_nan_mask_exact": bool(
            np.array_equal(np.isnan(expected_process), np.isnan(arrays["baseline_process"]))
        ),
        "maximum_finite_process_absolute_error": float(process_error),
        "process_within_1e_14": bool(process_error <= 1e-14),
    }
    if not all(
        audit[key]
        for key in (
            "labels_exact",
            "completed_horizon_exact",
            "process_nan_mask_exact",
            "process_within_1e_14",
        )
    ):
        raise AssertionError(f"F12 baseline replay failed: {audit}")
    return audit


def _f32_baseline_audit(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    baseline_index = next(
        index for index, definition in enumerate(F32_DEFINITIONS) if definition.is_baseline
    )
    with np.load(F32_SOURCE / "confirmation_arrays.npz", allow_pickle=False) as archive:
        comparisons = {
            "grid_primary_label_exact": np.array_equal(
                arrays["labels"][:, :, baseline_index], archive["labels_primary_all8"]
            ),
            "registered_targets_exact": np.array_equal(
                arrays["baseline_targets"][:, :, 0], archive["labels_primary_all8"]
            )
            and np.array_equal(
                arrays["baseline_targets"][:, :, 1], archive["labels_secondary_first5"]
            )
            and np.array_equal(
                arrays["baseline_targets"][:, :, 2], archive["labels_secondary_centroid"]
            ),
            "onsets_exact": np.array_equal(arrays["baseline_onsets"], archive["onsets"]),
            "completed_exact": np.array_equal(
                arrays["baseline_completed"], archive["completed_horizon"]
            ),
            "observed_exact": np.array_equal(
                arrays["baseline_observed"], archive["observed_fissions"]
            ),
            "first_break_exact": np.array_equal(
                arrays["baseline_first_break"], archive["first_break_index"]
            ),
            "first_run_exact": np.array_equal(
                arrays["baseline_first_run"], archive["first_run8_start"]
            ),
            "geometry_exact": np.array_equal(
                arrays["baseline_geometry"],
                archive["first_run8_geometry"],
                equal_nan=True,
            ),
        }
    if not all(comparisons.values()):
        raise AssertionError(f"F32 baseline replay failed: {comparisons}")
    return {"baseline_definition": F32_BASELINE, **comparisons}


def _cr1_baseline_audit(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    baseline_index = next(
        index for index, definition in enumerate(F12_DEFINITIONS) if definition.is_baseline
    )
    with np.load(CR1_SOURCE / "branch_arrays.npz", allow_pickle=False) as archive:
        expected_h = np.asarray(archive["boundary_h"], dtype=np.float64)
        expected_targets = np.asarray(archive["targets"], dtype=np.int8)
    observed_h = arrays["boundary_h"][:, :, :, :12]
    observed_targets = arrays["labels"][:, :, :, baseline_index]
    comparison = {
        "first_12_boundary_h_exact": bool(
            np.array_equal(expected_h, observed_h, equal_nan=True)
        ),
        "baseline_targets_exact": bool(np.array_equal(expected_targets, observed_targets)),
    }
    if not all(comparison.values()):
        raise AssertionError(f"CR1 baseline replay failed: {comparison}")
    return {"baseline_definition": F12_BASELINE, **comparison}


def _save_replay(
    dataset: str,
    cases: Sequence[StateCase],
    arrays: dict[str, np.ndarray],
    audit: dict[str, Any],
) -> None:
    REPLAY_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_npz(
        REPLAY_ROOT / f"{dataset}.npz",
        protocol_id=np.asarray(_saved_protocol_id()),
        **_metadata(cases),
        **arrays,
    )
    _write_json(REPLAY_ROOT / f"{dataset}_audit.json", audit)


def replay_f12(workers: int) -> None:
    verify_protocol()
    experiment, cases = _f12_cases(workers)
    arguments = [(index, case, experiment) for index, case in enumerate(cases)]
    _run_workers("f12", arguments, _f12_worker, workers)
    arrays = _load_checkpoints(
        "f12",
        len(cases),
        ("boundary_h", "labels", "baseline_process", "baseline_completed"),
    )
    _save_replay("f12", cases, arrays, _f12_baseline_audit(arrays))
    print(f"F12 replay and extension saved: {REPLAY_ROOT / 'f12.npz'}", flush=True)


def replay_f32(workers: int) -> None:
    verify_protocol()
    experiment, cases = _f32_cases(workers)
    arguments = [(index, case, experiment) for index, case in enumerate(cases)]
    _run_workers("f32", arguments, _f32_worker, workers)
    arrays = _load_checkpoints(
        "f32",
        len(cases),
        (
            "boundary_h",
            "labels",
            "onsets",
            "baseline_targets",
            "baseline_onsets",
            "baseline_completed",
            "baseline_observed",
            "baseline_first_break",
            "baseline_first_run",
            "baseline_geometry",
        ),
    )
    _save_replay("f32", cases, arrays, _f32_baseline_audit(arrays))
    print(f"F32 replay saved: {REPLAY_ROOT / 'f32.npz'}", flush=True)


def replay_cr1(workers: int) -> None:
    verify_protocol()
    experiment, cases = _cr1_cases(workers)
    with np.load(CR1_SOURCE / "state_and_matrix_arrays.npz", allow_pickle=False) as archive:
        expected_ids = np.asarray(archive["state_ids"])
        expected_compositions = np.asarray(archive["compositions"], dtype=np.int64)
    if not np.array_equal(expected_ids, np.asarray([case.state_id for case in cases])):
        raise AssertionError("CR1 rebuilt state IDs differ from the sealed archive")
    if not np.array_equal(
        expected_compositions,
        np.vstack([case.snapshot.composition for case in cases]),
    ):
        raise AssertionError("CR1 rebuilt compositions differ from the sealed archive")
    edits = _cr1_edits()
    arguments = [
        (
            index,
            case,
            experiment,
            tuple(edits[(case.state_id, arm)] for arm in CR1_ARMS),
        )
        for index, case in enumerate(cases)
    ]
    _run_workers("cr1", arguments, _cr1_worker, workers)
    arrays = _load_checkpoints("cr1", len(cases), ("boundary_h", "labels"))
    _save_replay("cr1", cases, arrays, _cr1_baseline_audit(arrays))
    print(f"CR1 replay and extension saved: {REPLAY_ROOT / 'cr1.npz'}", flush=True)


def replay_reference(workers: int) -> None:
    verify_protocol()
    experiment, cases = _f12_cases(workers)
    arguments = [(index, case, experiment) for index, case in enumerate(cases)]
    _run_workers("reference", arguments, _reference_worker, workers)
    arrays = _load_checkpoints(
        "reference",
        len(cases),
        (
            "within_boundary_h",
            "between_lineage_centroid_h",
            "component_sizes",
            "completed",
        ),
    )
    audit = {
        "classification": "new_contextual_l36_method_simulation_not_historical_l36_replay",
        "states": len(cases),
        "lineages": 2 * len(cases),
        "horizon": REFERENCE_HORIZON,
        "all_seed_roles_distinct_by_contract": True,
        "complete_lineages": int(arrays["completed"].sum()),
        "total_lineages": int(arrays["completed"].size),
        "defined_between_lineage_centroid_h": int(
            np.isfinite(arrays["between_lineage_centroid_h"]).sum()
        ),
        "frozen_l36_config_sha256": sha256_file(L36_CONFIG),
        "frozen_l36_script_sha256": sha256_file(L36_SCRIPT),
        "frozen_l36_centroid_source_sha256": sha256_file(L36_CENTROID_SOURCE),
        "frozen_l36_report_sha256": sha256_file(L36_REPORT),
        "frozen_l37_config_sha256": sha256_file(L37_CONFIG),
        "frozen_l37_script_sha256": sha256_file(L37_SCRIPT),
        "frozen_l37_report_sha256": sha256_file(L37_REPORT),
    }
    _save_replay("reference", cases, arrays, audit)
    print(f"Lineage reference saved: {REPLAY_ROOT / 'reference.npz'}", flush=True)


def replay(dataset: str, workers: int) -> None:
    functions = {
        "f12": replay_f12,
        "f32": replay_f32,
        "cr1": replay_cr1,
        "reference": replay_reference,
    }
    selected = tuple(functions) if dataset == "all" else (dataset,)
    for name in selected:
        functions[name](workers)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _read_states(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, dtype={"candidate": str})
    table["candidate"] = table["candidate"].str.zfill(2)
    return table


def analyze() -> None:
    verify_protocol()
    required = [
        REPLAY_ROOT / "f12.npz",
        REPLAY_ROOT / "f32.npz",
        REPLAY_ROOT / "cr1.npz",
        REPLAY_ROOT / "reference.npz",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"replay stages incomplete: {missing}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    f12 = _load_npz(REPLAY_ROOT / "f12.npz")
    f32 = _load_npz(REPLAY_ROOT / "f32.npz")
    cr1 = _load_npz(REPLAY_ROOT / "cr1.npz")
    reference = _load_npz(REPLAY_ROOT / "reference.npz")
    f12_states = _read_states(F12_SOURCE / "confirmation_states.csv")
    f32_states = _read_states(F32_SOURCE / "confirmation_states.csv")
    cr1_states = _read_states(CR1_SOURCE / "state_probabilities.csv")
    if not np.array_equal(f12["state_ids"], f12_states["state_id"].to_numpy()):
        raise AssertionError("F12 state order changed")
    if not np.array_equal(f32["state_ids"], f32_states["state_id"].to_numpy()):
        raise AssertionError("F32 state order changed")
    if not np.array_equal(cr1["state_ids"], cr1_states["state_id"].to_numpy()):
        raise AssertionError("CR1 state order changed")

    f12_definitions = f12_definition_table()
    f32_definitions = f32_definition_table()
    f12_metrics = summarize_prediction_grid(
        f12["labels"],
        f12_states,
        f12_definitions,
        "prediction_history",
        "prediction_full",
        "f12",
    )
    f32_metrics = summarize_prediction_grid(
        f32["labels"],
        f32_states,
        f32_definitions,
        "prediction_primary_all8_h10",
        "prediction_primary_all8_h10_state",
        "f32",
    )
    cr1_metrics = summarize_cr1_grid(
        cr1["labels"], cr1_states, f12_definitions, CR1_ARMS
    )
    h_summary = reference_summary(
        f12["boundary_h"][:, :, :12],
        f12["matrix_ids"],
        f12["candidates"],
        reference["between_lineage_centroid_h"],
        reference["matrix_ids"],
        reference["candidates"],
    )
    f12_metrics.to_csv(OUTPUT_ROOT / "f12_sensitivity.csv", index=False)
    f32_metrics.to_csv(OUTPUT_ROOT / "f32_sensitivity.csv", index=False)
    cr1_metrics.to_csv(OUTPUT_ROOT / "cr1_sensitivity.csv", index=False)
    h_summary.to_csv(OUTPUT_ROOT / "h_reference_summary.csv", index=False)
    f12_definitions.to_csv(OUTPUT_ROOT / "f12_definitions.csv", index=False)
    f32_definitions.to_csv(OUTPUT_ROOT / "f32_definitions.csv", index=False)
    audit = independently_recompute(f12_metrics, f32_metrics, cr1_metrics)
    _write_json(OUTPUT_ROOT / "metric_recomputation_audit.json", audit)
    if not audit["all_checks_passed"]:
        raise AssertionError(f"metric readback audit failed: {audit}")
    print(f"Sensitivity tables written to {OUTPUT_ROOT}", flush=True)


def independently_recompute(
    f12_metrics: pd.DataFrame,
    f32_metrics: pd.DataFrame,
    cr1_metrics: pd.DataFrame,
) -> dict[str, Any]:
    f12 = _load_npz(REPLAY_ROOT / "f12.npz")
    f32 = _load_npz(REPLAY_ROOT / "f32.npz")
    cr1 = _load_npz(REPLAY_ROOT / "cr1.npz")
    checks: dict[str, bool] = {}
    checks["f12_rows"] = len(f12_metrics) == len(F12_DEFINITIONS) * 2
    checks["f32_rows"] = len(f32_metrics) == len(F32_DEFINITIONS) * 2
    checks["cr1_rows"] = len(cr1_metrics) == len(F12_DEFINITIONS) * 2 * 2 * 4
    checks["f12_event_counts"] = int(f12_metrics["events"].sum()) == int(
        sum(
            f12["labels"][f12["candidates"] == candidate, :, definition].sum()
            for candidate in ("02", "03")
            for definition in range(len(F12_DEFINITIONS))
        )
    )
    checks["f32_event_counts"] = int(f32_metrics["events"].sum()) == int(
        sum(
            f32["labels"][f32["candidates"] == candidate, :, definition].sum()
            for candidate in ("02", "03")
            for definition in range(len(F32_DEFINITIONS))
        )
    )
    baseline_index = next(
        index for index, definition in enumerate(F12_DEFINITIONS) if definition.is_baseline
    )
    checks["cr1_baseline_nonempty"] = bool(cr1["labels"][:, :, :, baseline_index].sum() > 0)
    return {
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "f12_grid_cells": len(F12_DEFINITIONS),
        "f32_grid_cells": len(F32_DEFINITIONS),
    }


def _heatmap(
    axis: Any,
    table: pd.DataFrame,
    value: str,
    thresholds: Sequence[float],
    horizons: Sequence[int],
    title: str,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    pivot = table.pivot(
        index="inheritance_threshold_strict", columns="horizon_fissions", values=value
    ).reindex(index=thresholds, columns=horizons)
    image = axis.imshow(
        pivot.to_numpy(), aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax
    )
    axis.set_xticks(range(len(horizons)), [str(item) for item in horizons])
    axis.set_yticks(range(len(thresholds)), [f"{item:.2f}" for item in thresholds])
    axis.set_xlabel("Horizon F")
    axis.set_ylabel("Inheritance threshold")
    axis.set_title(title, fontsize=9)
    for row in range(pivot.shape[0]):
        for column in range(pivot.shape[1]):
            value_at_cell = pivot.iloc[row, column]
            if np.isfinite(value_at_cell):
                axis.text(
                    column,
                    row,
                    f"{value_at_cell:.3f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="black",
                )
    plt.colorbar(image, ax=axis, fraction=0.046, pad=0.04)


def _plot_h_reference(f12: dict[str, np.ndarray], reference: dict[str, np.ndarray]) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for column, candidate in enumerate(("02", "03")):
        selected = f12["candidates"] == candidate
        boundary = f12["boundary_h"][selected, :, :12].ravel()
        boundary = boundary[np.isfinite(boundary)]
        axis = axes[0, column]
        axis.hist(boundary, bins=100, range=(0, 1), density=True, color="#4472c4", alpha=0.8)
        for threshold in (0.85, 0.88, 0.90, 0.92, 0.95):
            axis.axvline(threshold, color="#222222", linewidth=0.8, alpha=0.7)
        axis.set_xlim(0.65, 1.0)
        axis.set_title(f"Candidate {candidate}: parent-to-daughter H")
        axis.set_xlabel("Cosine similarity H")
        axis.set_ylabel("Density")
        selected_reference = reference["candidates"] == candidate
        between = reference["between_lineage_centroid_h"][selected_reference].ravel()
        between = np.sort(between[np.isfinite(between)])
        axis = axes[1, column]
        axis.plot(between, np.arange(1, len(between) + 1) / len(between), color="#c44e52")
        for threshold in (0.80, 0.85, 0.90):
            axis.axvline(threshold, color="#222222", linewidth=0.8, alpha=0.7)
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.set_title(f"Candidate {candidate}: between independent lineages")
        axis.set_xlabel("Matched-fission daughter similarity H")
        axis.set_ylabel("Empirical CDF")
    figure.suptitle(
        "Figure S1. Empirical context for inheritance and distinctness cutoffs",
        fontsize=13,
    )
    figure.savefig(OUTPUT_ROOT / "figure_s1_h_reference.png", dpi=220)
    plt.close(figure)


def _plot_f12_surfaces(f12_metrics: pd.DataFrame) -> None:
    thresholds = (0.85, 0.88, 0.90, 0.92, 0.95)
    horizons = (8, 10, 12, 16)
    figure, axes = plt.subplots(4, 3, figsize=(13, 14), constrained_layout=True)
    for candidate_index, candidate in enumerate(("02", "03")):
        for run_index, run_length in enumerate((2, 3, 4)):
            selected = f12_metrics.loc[
                (f12_metrics["candidate"] == candidate)
                & (f12_metrics["renewal_run_length"] == run_length)
            ]
            _heatmap(
                axes[candidate_index * 2, run_index],
                selected,
                "prevalence",
                thresholds,
                horizons,
                f"c{candidate}, run {run_length}: prevalence",
                "viridis",
                0.0,
                1.0,
            )
            _heatmap(
                axes[candidate_index * 2 + 1, run_index],
                selected,
                "centered_branch_half_reliability",
                thresholds,
                horizons,
                f"c{candidate}, run {run_length}: centered reliability",
                "coolwarm",
                -1.0,
                1.0,
            )
    figure.suptitle(
        "Figure S2. F12-family prevalence and empirical-q reliability",
        fontsize=14,
    )
    figure.savefig(OUTPUT_ROOT / "figure_s2_f12_prevalence_reliability.png", dpi=220)
    plt.close(figure)


def _plot_prediction_and_cr1(
    f12_metrics: pd.DataFrame, cr1_metrics: pd.DataFrame
) -> None:
    thresholds = (0.85, 0.88, 0.90, 0.92, 0.95)
    horizons = (8, 10, 12, 16)
    figure, axes = plt.subplots(6, 3, figsize=(13, 20), constrained_layout=True)
    cr1_selected = cr1_metrics.loc[
        cr1_metrics["contrast"] == "MODEL_UP_minus_MODEL_DOWN"
    ]
    for candidate_index, candidate in enumerate(("02", "03")):
        base_row = candidate_index * 3
        for run_index, run_length in enumerate((2, 3, 4)):
            prediction = f12_metrics.loc[
                (f12_metrics["candidate"] == candidate)
                & (f12_metrics["renewal_run_length"] == run_length)
            ].copy()
            prediction["mean_log_gain"] = 0.5 * (
                prediction["log_loss_gain_A"] + prediction["log_loss_gain_B"]
            )
            _heatmap(
                axes[base_row, run_index],
                prediction,
                "mean_log_gain",
                thresholds,
                horizons,
                f"c{candidate}, run {run_length}: full-history log gain",
                "coolwarm",
            )
            for half_offset, half in enumerate(("A", "B"), start=1):
                intervention = cr1_selected.loc[
                    (cr1_selected["candidate"] == candidate)
                    & (cr1_selected["renewal_run_length"] == run_length)
                    & (cr1_selected["half"] == half)
                ]
                _heatmap(
                    axes[base_row + half_offset, run_index],
                    intervention,
                    "estimate",
                    thresholds,
                    horizons,
                    f"c{candidate}, run {run_length}: CR1 up-down (half {half})",
                    "coolwarm",
                )
    figure.suptitle(
        "Figure S3. Frozen predictor advantage and CR1 intervention direction",
        fontsize=14,
    )
    figure.savefig(OUTPUT_ROOT / "figure_s3_prediction_cr1.png", dpi=220)
    plt.close(figure)


def _plot_f32(f32_metrics: pd.DataFrame) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for row, candidate in enumerate(("02", "03")):
        selected_candidate = f32_metrics.loc[f32_metrics["candidate"] == candidate].copy()
        selected_candidate["mean_log_gain"] = 0.5 * (
            selected_candidate["log_loss_gain_A"]
            + selected_candidate["log_loss_gain_B"]
        )
        for column, metric in enumerate(
            ("prevalence", "centered_branch_half_reliability", "mean_log_gain")
        ):
            axis = axes[row, column]
            for anchor, marker in zip((0.80, 0.85, 0.90), ("o", "s", "^"), strict=True):
                selected = selected_candidate.loc[
                    selected_candidate["old_anchor_threshold_inclusive"] == anchor
                ].sort_values(
                    ["adjacent_and_pairwise_threshold_strict", "strict_run_length"]
                )
                for threshold in (0.88, 0.90, 0.92):
                    line = selected.loc[
                        selected["adjacent_and_pairwise_threshold_strict"] == threshold
                    ]
                    axis.plot(
                        line["strict_run_length"],
                        line[metric],
                        marker=marker,
                        label=f"anchor {anchor:.2f}, H {threshold:.2f}",
                        alpha=0.8,
                    )
            axis.axhline(0.0, color="black", linewidth=0.7)
            axis.set_xticks((7, 8, 9))
            axis.set_xlabel("Strict run length")
            axis.set_title(f"c{candidate}: {metric.replace('_', ' ')}")
            if row == 0 and column == 2:
                axis.legend(fontsize=6, ncol=2)
    figure.suptitle("Figure S4. Strict-F32 endpoint sensitivity", fontsize=14)
    figure.savefig(OUTPUT_ROOT / "figure_s4_f32_sensitivity.png", dpi=220)
    plt.close(figure)


def _format_range(values: pd.Series, digits: int = 3) -> str:
    finite = values[np.isfinite(values)]
    if finite.empty:
        return "undefined"
    return f"{finite.min():.{digits}f}-{finite.max():.{digits}f}"


def report() -> None:
    verify_protocol()
    for name in (
        "f12_sensitivity.csv",
        "f32_sensitivity.csv",
        "cr1_sensitivity.csv",
        "h_reference_summary.csv",
    ):
        if not (OUTPUT_ROOT / name).is_file():
            raise FileNotFoundError(f"analysis output missing: {name}")
    f12_metrics = pd.read_csv(OUTPUT_ROOT / "f12_sensitivity.csv", dtype={"candidate": str})
    f32_metrics = pd.read_csv(OUTPUT_ROOT / "f32_sensitivity.csv", dtype={"candidate": str})
    cr1_metrics = pd.read_csv(OUTPUT_ROOT / "cr1_sensitivity.csv", dtype={"candidate": str})
    h_summary = pd.read_csv(OUTPUT_ROOT / "h_reference_summary.csv", dtype={"candidate": str})
    f12 = _load_npz(REPLAY_ROOT / "f12.npz")
    reference = _load_npz(REPLAY_ROOT / "reference.npz")
    _plot_h_reference(f12, reference)
    _plot_f12_surfaces(f12_metrics)
    _plot_prediction_and_cr1(f12_metrics, cr1_metrics)
    _plot_f32(f32_metrics)

    local_f12 = f12_metrics.loc[
        f12_metrics["inheritance_threshold_strict"].isin((0.88, 0.90, 0.92))
        & f12_metrics["horizon_fissions"].isin((10, 12, 16))
    ]
    local_cr1 = cr1_metrics.loc[
        cr1_metrics["inheritance_threshold_strict"].isin((0.88, 0.90, 0.92))
        & cr1_metrics["horizon_fissions"].isin((10, 12, 16))
        & (cr1_metrics["contrast"] == "MODEL_UP_minus_MODEL_DOWN")
    ]
    baseline_f12 = f12_metrics.loc[f12_metrics["registered_baseline"]]
    baseline_f32 = f32_metrics.loc[f32_metrics["registered_baseline"]]
    baseline_cr1 = cr1_metrics.loc[
        cr1_metrics["registered_baseline"]
        & (cr1_metrics["contrast"] == "MODEL_UP_minus_MODEL_DOWN")
    ]
    report_lines = [
        "# Exploratory endpoint-definition sensitivity appendix",
        "",
        "## Status",
        "",
        "This appendix is post-hoc and exploratory. All definitions, metrics, models, contrasts, and figures were frozen before alternate grid results were opened. No model or feature transform was refitted or recalibrated.",
        "",
        "F16 is a deterministic extension of the archived F12 seed streams, not pure rescoring of retained F12 trajectories. The independent-lineage reference newly applies the frozen L36 two-lineage, dominant-H090-component-centroid design to scaled5 restored states because the historical L36/L37 raw Parquet trajectories were not retained locally.",
        "",
        "## Empirical H context",
        "",
        "| Distribution | Candidate | n | q05 | Median | q95 | Fraction at or below focal cutoff |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in h_summary.itertuples(index=False):
        if row.distribution == "parent_to_selected_daughter":
            label = "Parent-to-selected-daughter (cutoff 0.90)"
            fraction = row.fraction_le_0_90
        else:
            label = "Between independent reference-lineage centroids (cutoff 0.85)"
            fraction = row.fraction_le_0_85
        report_lines.append(
            f"| {label} | {row.candidate} | {int(row.observations):,} | {row.q05:.3f} | {row.median:.3f} | {row.q95:.3f} | {fraction:.3f} |"
        )
    report_lines.extend(
        [
            "",
            "Figure S1 shows the complete empirical reference distributions. The independent-lineage CDF supplies a dataset-specific scale for the 0.85 distinctness cutoff despite the high cosine floor of nonnegative composition vectors. No formal claim that the boundary-H distribution is bimodal, or that 0.90 was originally selected from an empirical antimode, is made.",
            "",
            "## F12-family sensitivity",
            "",
            f"Across the local neighborhood (`H=0.88-0.92`, F10-F16), prevalence spans {_format_range(local_f12['prevalence'])}, centered split-half reliability spans {_format_range(local_f12['centered_branch_half_reliability'])}, and mean frozen full-over-history log-loss gain spans {_format_range(0.5 * (local_f12['log_loss_gain_A'] + local_f12['log_loss_gain_B']), 4)}.",
            "",
            "The complete 60-definition table is retained; stress thresholds 0.85 and 0.95 are not omitted. Figures S2-S3 display all candidates and renewal lengths separately.",
            "",
            "### Local qualitative stability",
            "",
            "| Candidate | Full-history gain > 0, half A | Full-history gain > 0, half B | 95% CI entirely > 0, A/B |",
            "|---|---:|---:|---:|",
        ]
    )
    for candidate in ("02", "03"):
        selected = local_f12.loc[local_f12["candidate"] == candidate]
        report_lines.append(
            f"| {candidate} | {(selected['log_loss_gain_A'] > 0).sum()}/{len(selected)} | {(selected['log_loss_gain_B'] > 0).sum()}/{len(selected)} | {(selected['log_loss_gain_A_ci95_lower'] > 0).sum()}/{len(selected)}; {(selected['log_loss_gain_B_ci95_lower'] > 0).sum()}/{len(selected)} |"
        )
    report_lines.extend(
        [
            "",
            "### Registered F12 baseline readback",
            "",
            "| Candidate | Prevalence | Centered reliability | Log-loss gain A/B |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in baseline_f12.itertuples(index=False):
        report_lines.append(
            f"| {row.candidate} | {row.prevalence:.4f} | {row.centered_branch_half_reliability:.4f} | {row.log_loss_gain_A:.5f} / {row.log_loss_gain_B:.5f} |"
        )
    report_lines.extend(
        [
            "",
            "## CR1 intervention-direction sensitivity",
            "",
            f"Across the same local neighborhood, MODEL_UP minus MODEL_DOWN spans {_format_range(local_cr1['estimate'], 4)} across candidates and fixed branch halves. All four registered contrasts and all 60 endpoint definitions are reported in the machine-readable table.",
            "",
            "| Candidate | Half | MODEL_UP - MODEL_DOWN > 0 | 95% CI entirely > 0 |",
            "|---|---:|---:|---:|",
        ]
    )
    for candidate in ("02", "03"):
        for half in ("A", "B"):
            selected = local_cr1.loc[
                (local_cr1["candidate"] == candidate) & (local_cr1["half"] == half)
            ]
            report_lines.append(
                f"| {candidate} | {half} | {(selected['estimate'] > 0).sum()}/{len(selected)} | {(selected['ci95_lower'] > 0).sum()}/{len(selected)} |"
            )
    report_lines.extend(
        [
            "",
            "### Registered CR1 baseline readback",
            "",
            "| Candidate | Half | MODEL_UP - MODEL_DOWN [95% CI] |",
            "|---|---:|---:|",
        ]
    )
    for row in baseline_cr1.itertuples(index=False):
        report_lines.append(
            f"| {row.candidate} | {row.half} | {row.estimate:.4f} [{row.ci95_lower:.4f}, {row.ci95_upper:.4f}] |"
        )
    report_lines.extend(
        [
            "",
            "## Strict-F32 sensitivity",
            "",
            "The strict grid varies its coupled adjacent/all-pairs threshold, run length, and old-anchor cutoff without changing horizon, inputs, or predictor. It is descriptive and cannot rescue the registered predictor or the later failed ensemble.",
            "",
            "| Candidate | h10+state gain > 0, half A | h10+state gain > 0, half B | Prevalence range | Centered reliability range |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for candidate in ("02", "03"):
        selected = f32_metrics.loc[f32_metrics["candidate"] == candidate]
        report_lines.append(
            f"| {candidate} | {(selected['log_loss_gain_A'] > 0).sum()}/{len(selected)} | {(selected['log_loss_gain_B'] > 0).sum()}/{len(selected)} | {_format_range(selected['prevalence'])} | {_format_range(selected['centered_branch_half_reliability'])} |"
        )
    report_lines.extend(
        [
            "",
            "| Candidate | Baseline prevalence | Centered reliability | h10+state over h10 gain A/B |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in baseline_f32.itertuples(index=False):
        report_lines.append(
            f"| {row.candidate} | {row.prevalence:.4f} | {row.centered_branch_half_reliability:.4f} | {row.log_loss_gain_A:.6f} / {row.log_loss_gain_B:.6f} |"
        )
    report_lines.extend(
        [
            "",
            "The separately registered direct-plus-hurdle ensemble remains a failed all-candidate hypothesis: its existing baseline confirmation passed both candidate-03 halves and failed both candidate-02 halves. It was not refit or used to select an alternate definition here.",
            "",
            "## Interpretation boundary",
            "",
            "The tables describe whether estimates vary continuously around the registered definitions. They do not establish that any one cutoff is natural, prospectively validate an alternate endpoint, or license selection of the most favorable cell. F12 causal effects do not establish causal control of the strict F32 event.",
            "",
        ]
    )
    (OUTPUT_ROOT / "APPENDIX_THRESHOLD_SENSITIVITY.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    patch_lines = [
        "# Proposed manuscript additions (not applied)",
        "",
        "## Methods insertion: Exploratory endpoint-definition sensitivity",
        "",
        "After all registered analyses, we froze a post-hoc no-refit sensitivity protocol. The F12 family crossed strict parent-daughter inheritance thresholds 0.85, 0.88, 0.90, 0.92 and 0.95; horizons 8, 10, 12 and 16; and renewal lengths two, three and four. Archived full and direct-history predictions were applied unchanged. A separate F32 grid crossed coupled adjacent/all-pairs thresholds 0.88, 0.90 and 0.92; run lengths seven, eight and nine; and inclusive old-anchor thresholds 0.80, 0.85 and 0.90. Candidates and fixed branch halves remained separate, and uncertainty resampled whole catalytic matrices. F16 used a deterministic extension of the original seed streams. CR1 arms were rescored without changing their edits or common random streams.",
        "",
        "## Results insertion",
        "",
        *report_lines[report_lines.index("## F12-family sensitivity") + 2 : report_lines.index("## Interpretation boundary")],
        "",
        "## Replacement for Limitation 3",
        "",
        "> The inheritance, horizon, run-length, coherence, and old-anchor choices remain operational rather than uniquely validated. A post-hoc, no-refit replay and deterministic-extension sensitivity across nearby definitions is reported in Appendix X; it tests local qualitative stability but does not convert any alternate definition into a confirmatory endpoint.",
        "",
        "## Reviewer response",
        "",
        "We agree that acknowledging operational choices without showing their local consequences was insufficient. We added empirical parent-daughter and independent-lineage H reference distributions and complete, no-refit F12 and strict-F32 sensitivity grids. The appendix reports prevalence, ordinary and matrix-centered branch-half reliability, frozen predictor advantage, and CR1 intervention direction without selecting favorable combinations. We also distinguish exact rescoring through F12 from deterministic extension to F16 and retain the post-hoc limitation explicitly.",
        "",
    ]
    (OUTPUT_ROOT / "PROPOSED_MANUSCRIPT_AND_REVIEWER_PATCH.md").write_text(
        "\n".join(patch_lines), encoding="utf-8"
    )
    manifest = {
        "format": RESULT_FORMAT,
        "protocol_id": json.loads(
            (PROTOCOL_ROOT / "analysis_protocol.json").read_text(encoding="utf-8")
        )["protocol_id"],
        "f12_definitions": len(F12_DEFINITIONS),
        "f32_definitions": len(F32_DEFINITIONS),
        "cr1_only": True,
        "models_refit_or_recalibrated": False,
        "source_files_modified": False,
        "runtime": {"python": platform.python_version(), "platform": platform.platform()},
    }
    _write_json(OUTPUT_ROOT / "manifest.json", manifest)
    checksum = OUTPUT_ROOT / "SHA256SUMS"
    if checksum.exists():
        checksum.unlink()
    write_checksums(OUTPUT_ROOT)
    print(f"Appendix figures and reports written to {OUTPUT_ROOT}", flush=True)


def verify() -> None:
    protocol_audit = verify_protocol()
    verify_checksums(OUTPUT_ROOT)
    manifest = json.loads((OUTPUT_ROOT / "manifest.json").read_text(encoding="utf-8"))
    metric_audit = json.loads(
        (OUTPUT_ROOT / "metric_recomputation_audit.json").read_text(encoding="utf-8")
    )
    replay_audits = {
        name: json.loads(
            (REPLAY_ROOT / f"{name}_audit.json").read_text(encoding="utf-8")
        )
        for name in ("f12", "f32", "cr1", "reference")
    }
    required_figures = [
        OUTPUT_ROOT / "figure_s1_h_reference.png",
        OUTPUT_ROOT / "figure_s2_f12_prevalence_reliability.png",
        OUTPUT_ROOT / "figure_s3_prediction_cr1.png",
        OUTPUT_ROOT / "figure_s4_f32_sensitivity.png",
    ]
    checks = {
        "protocol_current": protocol_audit["source_contract_current"],
        "metric_readback": metric_audit["all_checks_passed"],
        "f12_baseline": all(
            replay_audits["f12"][key]
            for key in (
                "labels_exact",
                "completed_horizon_exact",
                "process_nan_mask_exact",
                "process_within_1e_14",
            )
        ),
        "f32_baseline": all(
            value
            for key, value in replay_audits["f32"].items()
            if key != "baseline_definition"
        ),
        "cr1_baseline": replay_audits["cr1"]["first_12_boundary_h_exact"]
        and replay_audits["cr1"]["baseline_targets_exact"],
        "figures_present": all(path.stat().st_size > 10_000 for path in required_figures),
        "result_contract": manifest["f12_definitions"] == 60
        and manifest["f32_definitions"] == 27
        and manifest["models_refit_or_recalibrated"] is False,
    }
    audit = {
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "checksum_manifest_regenerated_after_audit": True,
    }
    _write_json(OUTPUT_ROOT / "verification_audit.json", audit)
    if not audit["all_checks_passed"]:
        raise AssertionError(f"final verification failed: {audit}")
    checksum = OUTPUT_ROOT / "SHA256SUMS"
    if checksum.exists():
        checksum.unlink()
    write_checksums(OUTPUT_ROOT)
    verify_checksums(OUTPUT_ROOT)
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Exploratory no-refit endpoint-threshold sensitivity appendix"
    )
    subcommands = value.add_subparsers(dest="command", required=True)
    subcommands.add_parser("prepare")
    replay_parser = subcommands.add_parser("replay")
    replay_parser.add_argument(
        "--dataset", choices=("f12", "f32", "cr1", "reference", "all"), default="all"
    )
    replay_parser.add_argument(
        "--workers", type=int, default=max(1, min(os.cpu_count() or 1, 14))
    )
    subcommands.add_parser("analyze")
    subcommands.add_parser("report")
    subcommands.add_parser("verify")
    return value


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.command == "prepare":
        prepare()
    elif args.command == "replay":
        replay(args.dataset, args.workers)
    elif args.command == "analyze":
        analyze()
    elif args.command == "report":
        report()
    elif args.command == "verify":
        verify()
    else:  # pragma: no cover
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
