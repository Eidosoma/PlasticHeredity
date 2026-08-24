"""Run the reviewer-prompted threshold and similarity-metric sensitivity.

All generated files are confined to this analysis directory. Existing result
archives and simulator code are opened read-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


TASK_ROOT = Path(__file__).resolve().parent
PAPER_ROOT = TASK_ROOT.parent
WORKSPACE_ROOT = PAPER_ROOT.parent
SOURCE_ROOT = WORKSPACE_ROOT / "replicators.13.8.2026.codex"
os.environ.setdefault("MPLCONFIGDIR", str(TASK_ROOT / "artifacts" / "matplotlib"))
sys.path.insert(0, str(SOURCE_ROOT))
sys.path.insert(0, str(TASK_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from plastic_heredity.config import CANDIDATES, ExperimentConfig
from plastic_heredity.experiment import StateCase
from plastic_heredity.mechanistic import verify_checksums, write_checksums
from plastic_heredity.seeds import derive_seed
from plastic_heredity.simulator import simulate_future_absorbing
from reviewer_threshold_sensitivity_response.run_sensitivity import (
    _f12_cases,
    _f32_cases,
)
from reviewer_threshold_sensitivity_response.sensitivity_core import (
    dominant_h_component_centroid,
    summarize_cr1_grid,
    summarize_prediction_grid,
)

from sensitivity_core import (
    F12_DEFINITIONS,
    F12_HORIZONS,
    F12_RUN_LENGTHS,
    F12_THRESHOLDS,
    F32_ANCHOR_THRESHOLDS,
    F32_DEFINITIONS,
    F32_RUN_LENGTHS,
    F32_THRESHOLDS,
    boundary_similarities,
    bray_curtis_similarity,
    cosine_similarity,
    jaccard,
    quantile_matched_cutoffs,
    score_f12_array,
    score_f32_records,
)


ARTIFACT_ROOT = TASK_ROOT / "artifacts"
PROTOCOL_ROOT = ARTIFACT_ROOT / "protocol"
WORK_ROOT = ARTIFACT_ROOT / "work"
REPLAY_ROOT = ARTIFACT_ROOT / "replays"
CALIBRATION_ROOT = ARTIFACT_ROOT / "calibration"
OUTPUT_ROOT = ARTIFACT_ROOT / "output"

OLD_SENSITIVITY_ROOT = SOURCE_ROOT / "reviewer_threshold_sensitivity_response"
OLD_REPLAY_ROOT = OLD_SENSITIVITY_ROOT / "artifacts" / "replays"
F12_SOURCE = SOURCE_ROOT / "results" / "scaled5"
F32_SOURCE = SOURCE_ROOT / "results" / "regime_confirmation"
CR1_SOURCE = (
    SOURCE_ROOT
    / "results_intervention_replication"
    / "cr1_model_guided_confirmation"
)

F12_HORIZON = 16
F32_HORIZON = 32
REFERENCE_HORIZON = 100
REFERENCE_MASTER_SEED = (
    "e354bb648e15692f59bd99e947cddeeb3cbc5643a38045157762313166f86d4b"
)
CR1_ARMS = ("MODEL_UP", "MODEL_DOWN", "RANDOM", "NOOP")
CHECKPOINT_FORMAT = "threshold-metric-sensitivity-checkpoint-v1"
RESULT_FORMAT = "threshold-metric-sensitivity-results-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
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


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _source_contract() -> dict[str, dict[str, str]]:
    paths = {
        "runner": Path(__file__),
        "core": TASK_ROOT / "sensitivity_core.py",
        "simulator": SOURCE_ROOT / "plastic_heredity" / "simulator.py",
        "config": SOURCE_ROOT / "plastic_heredity" / "config.py",
        "seed_derivation": SOURCE_ROOT / "plastic_heredity" / "seeds.py",
        "old_sensitivity_core": OLD_SENSITIVITY_ROOT / "sensitivity_core.py",
        "old_sensitivity_runner": OLD_SENSITIVITY_ROOT / "run_sensitivity.py",
        "f12_manifest": F12_SOURCE / "manifest.json",
        "f12_checksums": F12_SOURCE / "SHA256SUMS",
        "f32_manifest": F32_SOURCE / "manifest.json",
        "f32_checksums": F32_SOURCE / "SHA256SUMS",
        "cr1_manifest": CR1_SOURCE / "manifest.json",
        "cr1_checksums": CR1_SOURCE / "SHA256SUMS",
        "old_f12_replay": OLD_REPLAY_ROOT / "f12.npz",
        "old_f32_replay": OLD_REPLAY_ROOT / "f32.npz",
        "old_cr1_replay": OLD_REPLAY_ROOT / "cr1.npz",
        "old_reference_replay": OLD_REPLAY_ROOT / "reference.npz",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing required read-only sources: {missing}")
    return {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for name, path in paths.items()
    }


def _verify_source_archives() -> dict[str, int]:
    return {
        "f12": len(verify_checksums(F12_SOURCE)),
        "f32": len(verify_checksums(F32_SOURCE)),
        "cr1": len(verify_checksums(CR1_SOURCE)),
    }


def _f12_protocol_table() -> list[dict[str, Any]]:
    return [
        {
            "definition_index": index,
            "definition_key": definition.key,
            "source_cosine_threshold": definition.source_threshold,
            "horizon_fissions": definition.horizon,
            "renewal_run_length": definition.run_length,
            "registered_shape": definition.registered_shape,
        }
        for index, definition in enumerate(F12_DEFINITIONS)
    ]


def _f32_protocol_table() -> list[dict[str, Any]]:
    return [
        {
            "definition_index": index,
            "definition_key": definition.key,
            "source_cosine_threshold": definition.source_threshold,
            "strict_run_length": definition.run_length,
            "source_cosine_anchor_threshold": definition.source_anchor_threshold,
            "horizon_fissions": F32_HORIZON,
            "registered_shape": definition.registered_shape,
        }
        for index, definition in enumerate(F32_DEFINITIONS)
    ]


def _protocol() -> dict[str, Any]:
    value: dict[str, Any] = {
        "format": "reviewer-threshold-metric-sensitivity-protocol-v1",
        "date": "2026-08-19",
        "status": (
            "reviewer_prompted_post_hoc_robustness_locked_before_new_cells_"
            "and_alternative_metric_readout"
        ),
        "working_boundary": {
            "all_new_writes_below": str(TASK_ROOT.resolve()),
            "all_existing_sources_read_only": True,
            "manuscript_modified": False,
        },
        "scientific_question": (
            "Do F12 renewal, strict coherent-regime occurrence, and the frozen "
            "predictor comparisons vary qualitatively across nearby operational "
            "thresholds or when cosine is replaced by a compositional distance?"
        ),
        "metrics": {
            "cosine": "frozen cosine similarity H",
            "bray_curtis": (
                "1 - 0.5*L1(p,q), where p and q are separately normalized "
                "composition profiles; equivalently one minus total-variation distance"
            ),
        },
        "metric_calibration": {
            "purpose": "match empirical strictness, not numerical cutoff values",
            "inheritance_and_coherence": (
                "For each source cosine cutoff, compute its pooled empirical <= "
                "percentile over all finite paired cosine/Bray parent-to-selected-"
                "daughter comparisons in the deterministic scaled5 F16 replay; "
                "use the inverted empirical Bray CDF at that percentile."
            ),
            "old_anchor": (
                "For each source cosine cutoff, compute its pooled empirical <= "
                "percentile over the prior frozen two-independent-lineage reference "
                "centroid comparisons; recompute Bray similarity on those same "
                "cosine-H090-component centroids and use the inverted empirical Bray CDF."
            ),
            "pooling": "both candidates, matrices, landmarks, and finite observations",
            "no_endpoint_or_prediction_use": True,
            "quantile_method": "inverted_cdf",
        },
        "f12_grid": _f12_protocol_table(),
        "f32_grid": _f32_protocol_table(),
        "models": {
            "f12": "archived scaled5 prediction_full versus prediction_history",
            "f32": (
                "archived REGCONF prediction_primary_all8_h10_state versus "
                "prediction_primary_all8_h10"
            ),
            "refitting": False,
            "recalibration": False,
            "threshold_dependent_history_recomputation": False,
        },
        "intervention": {
            "scope": (
                "CR1 is rescored over the extended cosine F12 grid from retained "
                "H trajectories; alternative-metric CR1 is outside scope because "
                "the retained intervention archive lacks compositions."
            ),
            "arms": CR1_ARMS,
            "common_random_streams_unchanged": True,
        },
        "replay": {
            "classification": (
                "exact deterministic replay of already scored registered seed streams; "
                "no added seed, branch, state, matrix, or future"
            ),
            "f12_horizon": F12_HORIZON,
            "f32_horizon": F32_HORIZON,
            "reference_horizon": REFERENCE_HORIZON,
            "checkpointed": True,
            "baseline_exact_readback_required": True,
        },
        "inference": {
            "independent_unit": "catalytic matrix",
            "bootstrap_repetitions": 512,
            "candidates_separate": True,
            "fixed_branch_halves_separate": True,
            "all_cells_reported": True,
            "multiplicity": "descriptive exploratory intervals; no confirmatory gate",
        },
        "inequalities": {
            "inheritance_and_coherence": "strict similarity > cutoff",
            "break_and_old_anchor": "inclusive similarity <= cutoff",
        },
        "planned_outputs": {
            "f12_cells_per_metric": len(F12_DEFINITIONS),
            "f32_cells_per_metric": len(F32_DEFINITIONS),
            "metrics": 2,
            "compact_plots": 3,
        },
        "claim_boundary": (
            "This is a post-hoc robustness analysis. It cannot designate a new "
            "confirmatory endpoint, select a favorable cell, or rescue a failed "
            "registered gate. Quantile matching is distributional, not an assertion "
            "that cosine and Bray-Curtis define identical events."
        ),
        "source_contract": _source_contract(),
    }
    value["protocol_id"] = canonical_digest(value)
    return _json_ready(value)


def prepare() -> None:
    _verify_source_archives()
    protocol = _protocol()
    PROTOCOL_ROOT.mkdir(parents=True, exist_ok=True)
    path = PROTOCOL_ROOT / "analysis_protocol.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved != protocol:
            raise ValueError("the existing frozen protocol differs from current code or inputs")
        print(f"Protocol already frozen and identical: {path}", flush=True)
        return
    _write_json(path, protocol)
    pd.DataFrame(_f12_protocol_table()).to_csv(
        PROTOCOL_ROOT / "f12_grid.csv", index=False
    )
    pd.DataFrame(_f32_protocol_table()).to_csv(
        PROTOCOL_ROOT / "f32_grid.csv", index=False
    )
    write_checksums(PROTOCOL_ROOT)
    print(f"Frozen post-hoc protocol: {path}", flush=True)


def verify_protocol() -> dict[str, Any]:
    verify_checksums(PROTOCOL_ROOT)
    saved = json.loads(
        (PROTOCOL_ROOT / "analysis_protocol.json").read_text(encoding="utf-8")
    )
    current = _protocol()
    if saved != current:
        raise ValueError("the frozen protocol, analysis code, or source identity changed")
    return {
        "protocol_id": saved["protocol_id"],
        "protocol_current": True,
        "source_archives_verified": _verify_source_archives(),
    }


def _protocol_id() -> str:
    path = PROTOCOL_ROOT / "analysis_protocol.json"
    if not path.is_file():
        raise FileNotFoundError("run the prepare stage before replay")
    return str(json.loads(path.read_text(encoding="utf-8"))["protocol_id"])


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
                and str(archive["protocol_id"].item()) == _protocol_id()
            )
    except Exception:
        return False


def _f12_worker(arguments: tuple[int, StateCase, ExperimentConfig]) -> int:
    index, case, experiment = arguments
    output = _checkpoint_path("f12_metric", index)
    if _checkpoint_complete(output, "f12_metric", index):
        return index
    branches = experiment.confirmation.branches_per_state
    boundary_cosine = np.full((branches, F12_HORIZON), np.nan, dtype=np.float64)
    boundary_bray = np.full_like(boundary_cosine, np.nan)
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
                F12_HORIZON,
                rng,
            )
            if records:
                boundary_cosine[branch, : len(records)] = np.asarray(
                    [record.h for record in records], dtype=np.float64
                )
                boundary_bray[branch, : len(records)] = boundary_similarities(
                    records, "bray_curtis"
                )
    _atomic_npz(
        output,
        format=np.asarray(CHECKPOINT_FORMAT),
        dataset=np.asarray("f12_metric"),
        state_index=np.asarray(index, dtype=np.int32),
        protocol_id=np.asarray(_protocol_id()),
        boundary_cosine=boundary_cosine,
        boundary_bray_curtis=boundary_bray,
    )
    return index


def _reference_worker(arguments: tuple[int, StateCase, ExperimentConfig]) -> int:
    index, case, experiment = arguments
    output = _checkpoint_path("reference_metric", index)
    if _checkpoint_complete(output, "reference_metric", index):
        return index
    states_by_role: list[np.ndarray | None] = [None, None]
    completed = np.zeros(2, dtype=np.int8)
    component_sizes = np.zeros(2, dtype=np.int16)
    with threadpool_limits(limits=1):
        for role_index, role in enumerate(("REFERENCE_A", "REFERENCE_B")):
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
            completed[role_index] = int(complete)
            if complete:
                states_by_role[role_index] = np.vstack(
                    [record.daughter for record in records]
                )
    centroid_cosine = np.asarray(np.nan, dtype=np.float64)
    centroid_bray = np.asarray(np.nan, dtype=np.float64)
    if completed.all():
        centroids: list[np.ndarray] = []
        for role_index, states in enumerate(states_by_role):
            if states is None:  # pragma: no cover - guarded by completed.all()
                raise AssertionError("complete reference lineage lacks states")
            centroid, members = dominant_h_component_centroid(states)
            centroids.append(centroid)
            component_sizes[role_index] = len(members)
        centroid_cosine = np.asarray(
            cosine_similarity(centroids[0], centroids[1]), dtype=np.float64
        )
        centroid_bray = np.asarray(
            bray_curtis_similarity(centroids[0], centroids[1]), dtype=np.float64
        )
    _atomic_npz(
        output,
        format=np.asarray(CHECKPOINT_FORMAT),
        dataset=np.asarray("reference_metric"),
        state_index=np.asarray(index, dtype=np.int32),
        protocol_id=np.asarray(_protocol_id()),
        centroid_cosine=centroid_cosine,
        centroid_bray_curtis=centroid_bray,
        component_sizes=component_sizes,
        completed=completed,
    )
    return index


def _load_calibration() -> dict[str, Any]:
    path = CALIBRATION_ROOT / "metric_calibration.json"
    if not path.is_file():
        raise FileNotFoundError("run foundation replay and calibrate before F32 replay")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value["protocol_id"] != _protocol_id():
        raise ValueError("metric calibration belongs to a different protocol")
    return value


def _cutoff_map(section: Mapping[str, Any]) -> dict[float, float]:
    return {float(key): float(value) for key, value in section.items()}


def _f32_worker(
    arguments: tuple[
        int,
        StateCase,
        ExperimentConfig,
        dict[float, float],
        dict[float, float],
    ]
) -> int:
    index, case, experiment, bray_cutoffs, bray_anchor_cutoffs = arguments
    output = _checkpoint_path("f32_metric", index)
    if _checkpoint_complete(output, "f32_metric", index):
        return index
    branches = experiment.confirmation.branches_per_state
    labels_cosine = np.zeros((branches, len(F32_DEFINITIONS)), dtype=np.int8)
    labels_bray = np.zeros_like(labels_cosine)
    onsets_cosine = np.full(labels_cosine.shape, -1, dtype=np.int16)
    onsets_bray = np.full(labels_cosine.shape, -1, dtype=np.int16)
    boundary_cosine = np.full((branches, F32_HORIZON), np.nan, dtype=np.float64)
    boundary_bray = np.full_like(boundary_cosine, np.nan)
    identity = {float(value): float(value) for value in F32_THRESHOLDS}
    identity_anchor = {
        float(value): float(value) for value in F32_ANCHOR_THRESHOLDS
    }
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
                F32_HORIZON,
                rng,
            )
            labels_cosine[branch], onsets_cosine[branch], cosine_values = (
                score_f32_records(
                    records, "cosine", identity, identity_anchor
                )
            )
            labels_bray[branch], onsets_bray[branch], bray_values = score_f32_records(
                records,
                "bray_curtis",
                bray_cutoffs,
                bray_anchor_cutoffs,
            )
            boundary_cosine[branch] = cosine_values
            boundary_bray[branch] = bray_values
    _atomic_npz(
        output,
        format=np.asarray(CHECKPOINT_FORMAT),
        dataset=np.asarray("f32_metric"),
        state_index=np.asarray(index, dtype=np.int32),
        protocol_id=np.asarray(_protocol_id()),
        boundary_cosine=boundary_cosine,
        boundary_bray_curtis=boundary_bray,
        labels_cosine=labels_cosine,
        labels_bray_curtis=labels_bray,
        onsets_cosine=onsets_cosine,
        onsets_bray_curtis=onsets_bray,
    )
    return index


def _run_workers(
    dataset: str,
    arguments: Sequence[Any],
    worker: Callable[[Any], int],
    workers: int,
) -> None:
    (WORK_ROOT / dataset).mkdir(parents=True, exist_ok=True)
    pending = [
        item
        for item in arguments
        if not _checkpoint_complete(
            _checkpoint_path(dataset, int(item[0])), dataset, int(item[0])
        )
    ]
    done = len(arguments) - len(pending)
    print(
        f"[{dataset}] {done}/{len(arguments)} checkpoints present; "
        f"{len(pending)} pending",
        flush=True,
    )
    if not pending:
        return
    progress_every = max(1, len(pending) // 50)
    if workers <= 1:
        for count, item in enumerate(pending, start=1):
            worker(item)
            if count % progress_every == 0 or count == len(pending):
                print(f"[{dataset}] {count}/{len(pending)} new states", flush=True)
        return
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(worker, item) for item in pending]
        for count, future in enumerate(as_completed(futures), start=1):
            future.result()
            if count % progress_every == 0 or count == len(pending):
                print(f"[{dataset}] {count}/{len(pending)} new states", flush=True)


def _assemble(dataset: str, count: int, keys: Sequence[str]) -> dict[str, np.ndarray]:
    values: dict[str, list[np.ndarray]] = {key: [] for key in keys}
    for index in range(count):
        path = _checkpoint_path(dataset, index)
        if not _checkpoint_complete(path, dataset, index):
            raise ValueError(f"missing or invalid {dataset} checkpoint {index}")
        with np.load(path, allow_pickle=False) as archive:
            for key in keys:
                values[key].append(np.asarray(archive[key]))
    return {key: np.stack(items) for key, items in values.items()}


def _metadata(cases: Sequence[StateCase]) -> dict[str, np.ndarray]:
    return {
        "state_ids": np.asarray([case.state_id for case in cases]),
        "candidates": np.asarray([case.candidate for case in cases]),
        "matrix_ids": np.asarray([case.matrix_id for case in cases], dtype=np.int16),
        "landmarks": np.asarray([case.landmark for case in cases], dtype=np.int16),
    }


def replay_foundation(workers: int) -> None:
    verify_protocol()
    experiment, cases = _f12_cases(workers)
    arguments = [(index, case, experiment) for index, case in enumerate(cases)]
    _run_workers("f12_metric", arguments, _f12_worker, workers)
    f12 = _assemble(
        "f12_metric", len(cases), ("boundary_cosine", "boundary_bray_curtis")
    )
    old_f12 = _load_npz(OLD_REPLAY_ROOT / "f12.npz")
    f12_audit = {
        "state_ids_exact": bool(
            np.array_equal(old_f12["state_ids"], _metadata(cases)["state_ids"])
        ),
        "cosine_boundary_exact": bool(
            np.array_equal(
                old_f12["boundary_h"], f12["boundary_cosine"], equal_nan=True
            )
        ),
        "finite_metric_pairs": int(
            np.count_nonzero(
                np.isfinite(f12["boundary_cosine"])
                & np.isfinite(f12["boundary_bray_curtis"])
            )
        ),
    }
    if not all(
        f12_audit[key] for key in ("state_ids_exact", "cosine_boundary_exact")
    ):
        raise AssertionError(f"F12 deterministic replay audit failed: {f12_audit}")
    _atomic_npz(
        REPLAY_ROOT / "f12_metric.npz",
        protocol_id=np.asarray(_protocol_id()),
        **_metadata(cases),
        **f12,
    )
    _write_json(REPLAY_ROOT / "f12_metric_audit.json", f12_audit)

    _run_workers("reference_metric", arguments, _reference_worker, workers)
    reference = _assemble(
        "reference_metric",
        len(cases),
        (
            "centroid_cosine",
            "centroid_bray_curtis",
            "component_sizes",
            "completed",
        ),
    )
    old_reference = _load_npz(OLD_REPLAY_ROOT / "reference.npz")
    reference_audit = {
        "state_ids_exact": bool(
            np.array_equal(old_reference["state_ids"], _metadata(cases)["state_ids"])
        ),
        "cosine_centroid_similarity_exact": bool(
            np.array_equal(
                old_reference["between_lineage_centroid_h"],
                reference["centroid_cosine"],
                equal_nan=True,
            )
        ),
        "component_sizes_exact": bool(
            np.array_equal(
                old_reference["component_sizes"], reference["component_sizes"]
            )
        ),
        "completion_exact": bool(
            np.array_equal(old_reference["completed"], reference["completed"])
        ),
        "finite_metric_pairs": int(
            np.count_nonzero(
                np.isfinite(reference["centroid_cosine"])
                & np.isfinite(reference["centroid_bray_curtis"])
            )
        ),
    }
    if not all(
        value
        for key, value in reference_audit.items()
        if key != "finite_metric_pairs"
    ):
        raise AssertionError(
            f"reference deterministic replay audit failed: {reference_audit}"
        )
    _atomic_npz(
        REPLAY_ROOT / "reference_metric.npz",
        protocol_id=np.asarray(_protocol_id()),
        **_metadata(cases),
        **reference,
    )
    _write_json(REPLAY_ROOT / "reference_metric_audit.json", reference_audit)
    print(f"Foundation replays saved under {REPLAY_ROOT}", flush=True)


def calibrate() -> None:
    verify_protocol()
    f12 = _load_npz(REPLAY_ROOT / "f12_metric.npz")
    reference = _load_npz(REPLAY_ROOT / "reference_metric.npz")
    inheritance_mapping, inheritance_rows = quantile_matched_cutoffs(
        f12["boundary_cosine"],
        f12["boundary_bray_curtis"],
        F12_THRESHOLDS,
    )
    anchor_mapping, anchor_rows = quantile_matched_cutoffs(
        reference["centroid_cosine"],
        reference["centroid_bray_curtis"],
        F32_ANCHOR_THRESHOLDS,
    )
    calibration = {
        "format": "threshold-metric-quantile-calibration-v1",
        "protocol_id": _protocol_id(),
        "inheritance_and_coherence_bray_cutoff_by_source_cosine": {
            f"{key:.3f}": value for key, value in inheritance_mapping.items()
        },
        "old_anchor_bray_cutoff_by_source_cosine": {
            f"{key:.2f}": value for key, value in anchor_mapping.items()
        },
        "inheritance_rows": inheritance_rows,
        "anchor_rows": anchor_rows,
        "outcomes_or_predictions_used": False,
        "candidate_specific_cutoffs": False,
    }
    _write_json(CALIBRATION_ROOT / "metric_calibration.json", calibration)
    pd.DataFrame(inheritance_rows).assign(role="inheritance_and_coherence").to_csv(
        CALIBRATION_ROOT / "inheritance_mapping.csv", index=False
    )
    pd.DataFrame(anchor_rows).assign(role="old_anchor").to_csv(
        CALIBRATION_ROOT / "old_anchor_mapping.csv", index=False
    )
    write_checksums(CALIBRATION_ROOT)
    print(
        "Metric cutoffs calibrated from similarity distributions only: "
        f"{CALIBRATION_ROOT / 'metric_calibration.json'}",
        flush=True,
    )


def replay_f32(workers: int) -> None:
    verify_protocol()
    calibration = _load_calibration()
    bray_cutoffs = _cutoff_map(
        calibration["inheritance_and_coherence_bray_cutoff_by_source_cosine"]
    )
    bray_anchor = _cutoff_map(
        calibration["old_anchor_bray_cutoff_by_source_cosine"]
    )
    experiment, cases = _f32_cases(workers)
    arguments = [
        (index, case, experiment, bray_cutoffs, bray_anchor)
        for index, case in enumerate(cases)
    ]
    _run_workers("f32_metric", arguments, _f32_worker, workers)
    arrays = _assemble(
        "f32_metric",
        len(cases),
        (
            "boundary_cosine",
            "boundary_bray_curtis",
            "labels_cosine",
            "labels_bray_curtis",
            "onsets_cosine",
            "onsets_bray_curtis",
        ),
    )
    old_f32 = _load_npz(OLD_REPLAY_ROOT / "f32.npz")
    baseline_index = next(
        index for index, definition in enumerate(F32_DEFINITIONS)
        if definition.registered_shape
    )
    audit = {
        "state_ids_exact": bool(
            np.array_equal(old_f32["state_ids"], _metadata(cases)["state_ids"])
        ),
        "cosine_boundary_exact": bool(
            np.array_equal(
                old_f32["boundary_h"], arrays["boundary_cosine"], equal_nan=True
            )
        ),
        "registered_cosine_labels_exact": bool(
            np.array_equal(
                old_f32["baseline_targets"][:, :, 0],
                arrays["labels_cosine"][:, :, baseline_index],
            )
        ),
        "registered_cosine_onsets_exact": bool(
            np.array_equal(
                old_f32["baseline_onsets"][:, :, 0],
                arrays["onsets_cosine"][:, :, baseline_index],
            )
        ),
    }
    if not all(audit.values()):
        raise AssertionError(f"F32 deterministic replay audit failed: {audit}")
    _atomic_npz(
        REPLAY_ROOT / "f32_metric.npz",
        protocol_id=np.asarray(_protocol_id()),
        **_metadata(cases),
        **arrays,
    )
    _write_json(REPLAY_ROOT / "f32_metric_audit.json", audit)
    print(f"F32 metric replay saved: {REPLAY_ROOT / 'f32_metric.npz'}", flush=True)


def _definition_tables(
    calibration: Mapping[str, Any]
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    identity = {float(value): float(value) for value in F12_THRESHOLDS}
    identity_anchor = {
        float(value): float(value) for value in F32_ANCHOR_THRESHOLDS
    }
    bray = _cutoff_map(
        calibration["inheritance_and_coherence_bray_cutoff_by_source_cosine"]
    )
    bray_anchor = _cutoff_map(
        calibration["old_anchor_bray_cutoff_by_source_cosine"]
    )
    f12_tables: dict[str, pd.DataFrame] = {}
    f32_tables: dict[str, pd.DataFrame] = {}
    for metric, cutoffs, anchors in (
        ("cosine", identity, identity_anchor),
        ("bray_curtis", bray, bray_anchor),
    ):
        f12_tables[metric] = pd.DataFrame(
            [
                {
                    "definition_index": index,
                    "definition_key": definition.key,
                    "metric": metric,
                    "inheritance_threshold_source_cosine": definition.source_threshold,
                    "metric_threshold_strict": cutoffs[definition.source_threshold],
                    "horizon_fissions": definition.horizon,
                    "renewal_run_length": definition.run_length,
                    "registered_shape": definition.registered_shape,
                    "registered_metric_and_shape": (
                        metric == "cosine" and definition.registered_shape
                    ),
                }
                for index, definition in enumerate(F12_DEFINITIONS)
            ]
        )
        f32_tables[metric] = pd.DataFrame(
            [
                {
                    "definition_index": index,
                    "definition_key": definition.key,
                    "metric": metric,
                    "adjacent_and_pairwise_threshold_source_cosine": (
                        definition.source_threshold
                    ),
                    "metric_threshold_strict": cutoffs[definition.source_threshold],
                    "strict_run_length": definition.run_length,
                    "old_anchor_threshold_source_cosine": (
                        definition.source_anchor_threshold
                    ),
                    "metric_old_anchor_threshold_inclusive": anchors[
                        definition.source_anchor_threshold
                    ],
                    "horizon_fissions": F32_HORIZON,
                    "registered_shape": definition.registered_shape,
                    "registered_metric_and_shape": (
                        metric == "cosine" and definition.registered_shape
                    ),
                }
                for index, definition in enumerate(F32_DEFINITIONS)
            ]
        )
    return f12_tables, f32_tables


def _read_states(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, dtype={"candidate": str})
    table["candidate"] = table["candidate"].astype(str).str.zfill(2)
    return table


def _agreement_table(
    endpoint: str,
    cosine_labels: np.ndarray,
    bray_labels: np.ndarray,
    candidates: np.ndarray,
    definitions: Sequence,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate in ("02", "03"):
        selected = np.asarray(candidates).astype(str) == candidate
        for index, definition in enumerate(definitions):
            left = cosine_labels[selected, :, index]
            right = bray_labels[selected, :, index]
            rows.append(
                {
                    "endpoint": endpoint,
                    "candidate": candidate,
                    "definition_index": index,
                    "definition_key": definition.key,
                    "cosine_events": int(left.sum()),
                    "bray_curtis_events": int(right.sum()),
                    "cosine_prevalence": float(left.mean()),
                    "bray_curtis_prevalence": float(right.mean()),
                    "event_jaccard": jaccard(left, right),
                    "branch_label_agreement": float(np.mean(left == right)),
                    "both_positive": int(np.count_nonzero((left == 1) & (right == 1))),
                }
            )
    return pd.DataFrame(rows)


def analyze() -> None:
    verify_protocol()
    calibration = _load_calibration()
    f12_metric = _load_npz(REPLAY_ROOT / "f12_metric.npz")
    f32_metric = _load_npz(REPLAY_ROOT / "f32_metric.npz")
    old_cr1 = _load_npz(OLD_REPLAY_ROOT / "cr1.npz")
    f12_tables, f32_tables = _definition_tables(calibration)
    identity = {float(value): float(value) for value in F12_THRESHOLDS}
    bray_cutoffs = _cutoff_map(
        calibration["inheritance_and_coherence_bray_cutoff_by_source_cosine"]
    )
    f12_labels = {
        "cosine": score_f12_array(f12_metric["boundary_cosine"], identity),
        "bray_curtis": score_f12_array(
            f12_metric["boundary_bray_curtis"], bray_cutoffs
        ),
    }
    cr1_labels = score_f12_array(old_cr1["boundary_h"], identity)
    f32_labels = {
        "cosine": f32_metric["labels_cosine"],
        "bray_curtis": f32_metric["labels_bray_curtis"],
    }
    f12_states = _read_states(F12_SOURCE / "confirmation_states.csv")
    f32_states = _read_states(F32_SOURCE / "confirmation_states.csv")
    cr1_states = _read_states(CR1_SOURCE / "state_probabilities.csv")
    if not np.array_equal(f12_metric["state_ids"], f12_states["state_id"]):
        raise AssertionError("F12 state order changed")
    if not np.array_equal(f32_metric["state_ids"], f32_states["state_id"]):
        raise AssertionError("F32 state order changed")
    if not np.array_equal(old_cr1["state_ids"], cr1_states["state_id"]):
        raise AssertionError("CR1 state order changed")

    f12_metrics: list[pd.DataFrame] = []
    f32_metrics: list[pd.DataFrame] = []
    for metric in ("cosine", "bray_curtis"):
        f12_metrics.append(
            summarize_prediction_grid(
                f12_labels[metric],
                f12_states,
                f12_tables[metric],
                "prediction_history",
                "prediction_full",
                f"threshold_metric_f12_{metric}",
            )
        )
        f32_metrics.append(
            summarize_prediction_grid(
                f32_labels[metric],
                f32_states,
                f32_tables[metric],
                "prediction_primary_all8_h10",
                "prediction_primary_all8_h10_state",
                f"threshold_metric_f32_{metric}",
            )
        )
    f12_metrics_table = pd.concat(f12_metrics, ignore_index=True)
    f32_metrics_table = pd.concat(f32_metrics, ignore_index=True)
    cr1_metrics = summarize_cr1_grid(
        cr1_labels,
        cr1_states,
        f12_tables["cosine"],
        CR1_ARMS,
    )
    agreement = pd.concat(
        [
            _agreement_table(
                "f12",
                f12_labels["cosine"],
                f12_labels["bray_curtis"],
                f12_metric["candidates"],
                F12_DEFINITIONS,
            ),
            _agreement_table(
                "f32",
                f32_labels["cosine"],
                f32_labels["bray_curtis"],
                f32_metric["candidates"],
                F32_DEFINITIONS,
            ),
        ],
        ignore_index=True,
    )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    f12_metrics_table.to_csv(OUTPUT_ROOT / "f12_sensitivity.csv", index=False)
    f32_metrics_table.to_csv(OUTPUT_ROOT / "f32_sensitivity.csv", index=False)
    cr1_metrics.to_csv(OUTPUT_ROOT / "cr1_cosine_sensitivity.csv", index=False)
    agreement.to_csv(OUTPUT_ROOT / "cross_metric_agreement.csv", index=False)
    pd.concat(f12_tables.values(), ignore_index=True).to_csv(
        OUTPUT_ROOT / "f12_definitions.csv", index=False
    )
    pd.concat(f32_tables.values(), ignore_index=True).to_csv(
        OUTPUT_ROOT / "f32_definitions.csv", index=False
    )

    f12_baseline = next(
        index for index, definition in enumerate(F12_DEFINITIONS)
        if definition.registered_shape
    )
    f32_baseline = next(
        index for index, definition in enumerate(F32_DEFINITIONS)
        if definition.registered_shape
    )
    with np.load(F12_SOURCE / "analysis_arrays.npz", allow_pickle=False) as source:
        source_f12 = np.asarray(source["confirmation_targets"], dtype=np.int8)
    with np.load(CR1_SOURCE / "branch_arrays.npz", allow_pickle=False) as source:
        source_cr1 = np.asarray(source["targets"], dtype=np.int8)
    with np.load(F32_SOURCE / "confirmation_arrays.npz", allow_pickle=False) as source:
        source_f32 = np.asarray(source["labels_primary_all8"], dtype=np.int8)
    audit = {
        "f12_registered_cosine_labels_exact": bool(
            np.array_equal(f12_labels["cosine"][:, :, f12_baseline], source_f12)
        ),
        "cr1_registered_cosine_labels_exact": bool(
            np.array_equal(cr1_labels[:, :, :, f12_baseline], source_cr1)
        ),
        "f32_registered_cosine_labels_exact": bool(
            np.array_equal(f32_labels["cosine"][:, :, f32_baseline], source_f32)
        ),
        "f12_rows": len(f12_metrics_table) == 2 * 2 * len(F12_DEFINITIONS),
        "f32_rows": len(f32_metrics_table) == 2 * 2 * len(F32_DEFINITIONS),
        "cr1_rows": len(cr1_metrics) == 2 * 2 * 4 * len(F12_DEFINITIONS),
        "agreement_rows": len(agreement)
        == 2 * (len(F12_DEFINITIONS) + len(F32_DEFINITIONS)),
    }
    audit["all_checks_passed"] = all(audit.values())
    _write_json(OUTPUT_ROOT / "metric_readback_audit.json", audit)
    if not audit["all_checks_passed"]:
        raise AssertionError(f"analysis readback audit failed: {audit}")
    print(f"Sensitivity tables written under {OUTPUT_ROOT}", flush=True)


def _heatmap(
    axis: Any,
    values: pd.DataFrame,
    row_field: str,
    row_values: Sequence,
    column_field: str,
    column_values: Sequence,
    value_field: str,
    title: str,
    cmap: str,
    center_zero: bool = False,
    percent: bool = False,
) -> None:
    pivot = values.pivot(
        index=row_field, columns=column_field, values=value_field
    ).reindex(index=row_values, columns=column_values)
    matrix = pivot.to_numpy(dtype=np.float64)
    kwargs: dict[str, Any] = {}
    if center_zero:
        maximum = float(np.nanmax(np.abs(matrix), initial=1e-9))
        kwargs = {"vmin": -maximum, "vmax": maximum}
    image = axis.imshow(
        matrix, origin="lower", aspect="auto", cmap=cmap, **kwargs
    )
    axis.set_xticks(range(len(column_values)), [str(item) for item in column_values])
    axis.set_yticks(
        range(len(row_values)),
        [f"{float(item):.3f}" for item in row_values],
    )
    axis.set_xlabel(column_field.replace("_", " "))
    axis.set_ylabel("source cosine threshold")
    axis.set_title(title, fontsize=9)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            if np.isfinite(value):
                label = f"{100 * value:.2f}%" if percent else f"{value:.3f}"
                axis.text(column, row, label, ha="center", va="center", fontsize=6)
    plt.colorbar(image, ax=axis, fraction=0.046, pad=0.04)


def _plot_f12(table: pd.DataFrame) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
    for row, metric in enumerate(("cosine", "bray_curtis")):
        selected_metric = table.loc[table["metric"] == metric].copy()
        selected_metric["worst_half_gain"] = selected_metric[
            ["log_loss_gain_A", "log_loss_gain_B"]
        ].min(axis=1)
        collapsed = (
            selected_metric.groupby(
                [
                    "inheritance_threshold_source_cosine",
                    "horizon_fissions",
                    "renewal_run_length",
                ],
                as_index=False,
            )["worst_half_gain"]
            .min()
        )
        for column, horizon in enumerate(F12_HORIZONS):
            values = collapsed.loc[collapsed["horizon_fissions"] == horizon]
            _heatmap(
                axes[row, column],
                values,
                "inheritance_threshold_source_cosine",
                F12_THRESHOLDS,
                "renewal_run_length",
                F12_RUN_LENGTHS,
                "worst_half_gain",
                f"{metric.replace('_', ' ').title()}, F={horizon}",
                "coolwarm",
                center_zero=True,
            )
    figure.suptitle(
        "F12 robustness: minimum composite-over-history gain across candidates and halves",
        fontsize=12,
    )
    figure.savefig(OUTPUT_ROOT / "figure_1_f12_threshold_metric.png", dpi=220)
    plt.close(figure)


def _plot_f32(table: pd.DataFrame) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for row, metric in enumerate(("cosine", "bray_curtis")):
        selected_metric = table.loc[table["metric"] == metric]
        collapsed = (
            selected_metric.groupby(
                [
                    "adjacent_and_pairwise_threshold_source_cosine",
                    "strict_run_length",
                    "old_anchor_threshold_source_cosine",
                ],
                as_index=False,
            )["prevalence"]
            .mean()
        )
        for column, anchor in enumerate(F32_ANCHOR_THRESHOLDS):
            values = collapsed.loc[
                collapsed["old_anchor_threshold_source_cosine"] == anchor
            ]
            _heatmap(
                axes[row, column],
                values,
                "adjacent_and_pairwise_threshold_source_cosine",
                F32_THRESHOLDS,
                "strict_run_length",
                F32_RUN_LENGTHS,
                "prevalence",
                f"{metric.replace('_', ' ').title()}, anchor equiv. {anchor:.2f}",
                "viridis",
                percent=True,
            )
    figure.suptitle(
        "Strict coherent-event prevalence across windows, cutoffs, and metrics",
        fontsize=12,
    )
    figure.savefig(OUTPUT_ROOT / "figure_2_strict_threshold_metric.png", dpi=220)
    plt.close(figure)


def _plot_agreement(agreement: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for axis, endpoint in zip(axes, ("f12", "f32"), strict=True):
        selected = agreement.loc[agreement["endpoint"] == endpoint]
        for candidate, marker in (("02", "o"), ("03", "s")):
            values = selected.loc[selected["candidate"] == candidate]
            axis.scatter(
                values["cosine_prevalence"],
                values["bray_curtis_prevalence"],
                s=18,
                alpha=0.65,
                marker=marker,
                label=f"candidate {candidate}",
            )
        maximum = float(
            max(
                selected["cosine_prevalence"].max(),
                selected["bray_curtis_prevalence"].max(),
                0.01,
            )
        )
        axis.plot([0, maximum], [0, maximum], color="black", linewidth=0.8)
        axis.set_xlim(0, maximum * 1.03)
        axis.set_ylim(0, maximum * 1.03)
        axis.set_xlabel("Cosine prevalence")
        axis.set_ylabel("Quantile-matched Bray-Curtis prevalence")
        axis.set_title(endpoint.upper())
        axis.legend(fontsize=8)
    figure.suptitle("Cross-metric prevalence agreement (every grid cell)", fontsize=12)
    figure.savefig(OUTPUT_ROOT / "figure_3_cross_metric_agreement.png", dpi=220)
    plt.close(figure)


def _range(series: pd.Series, digits: int = 4) -> str:
    finite = series[np.isfinite(series)]
    if finite.empty:
        return "undefined"
    return f"{finite.min():.{digits}f}–{finite.max():.{digits}f}"


def report() -> None:
    verify_protocol()
    calibration = _load_calibration()
    f12 = pd.read_csv(OUTPUT_ROOT / "f12_sensitivity.csv", dtype={"candidate": str})
    f32 = pd.read_csv(OUTPUT_ROOT / "f32_sensitivity.csv", dtype={"candidate": str})
    cr1 = pd.read_csv(
        OUTPUT_ROOT / "cr1_cosine_sensitivity.csv", dtype={"candidate": str}
    )
    agreement = pd.read_csv(
        OUTPUT_ROOT / "cross_metric_agreement.csv", dtype={"candidate": str}
    )
    for table in (f12, f32, cr1, agreement):
        table["candidate"] = table["candidate"].astype(str).str.zfill(2)
    _plot_f12(f12)
    _plot_f32(f32)
    _plot_agreement(agreement)

    inheritance_mapping = calibration[
        "inheritance_and_coherence_bray_cutoff_by_source_cosine"
    ]
    anchor_mapping = calibration["old_anchor_bray_cutoff_by_source_cosine"]
    f12_shape = f12.loc[f12["registered_shape"]].copy()
    f32_shape = f32.loc[f32["registered_shape"]].copy()
    f12["worst_half_gain"] = f12[["log_loss_gain_A", "log_loss_gain_B"]].min(axis=1)
    f32["worst_half_gain"] = f32[["log_loss_gain_A", "log_loss_gain_B"]].min(axis=1)
    strict_agreement = agreement.loc[
        (agreement["endpoint"] == "f32")
        & agreement["definition_key"].str.contains("h0.900_r8_a0.85", regex=False)
    ]
    f12_positive = bool((f12["worst_half_gain"] > 0).all())
    f12_ci_positive = bool(
        (
            f12[["log_loss_gain_A_ci95_lower", "log_loss_gain_B_ci95_lower"]]
            .min(axis=1)
            > 0
        ).all()
    )
    strict_nonzero = (
        f32.groupby(["metric", "candidate"])["events"].max() > 0
    ).to_dict()

    mapping_rows = [
        "| Source cosine | Bray-Curtis inheritance/coherence |",
        "|---:|---:|",
    ]
    mapping_rows.extend(
        f"| {float(key):.3f} | {float(value):.6f} |"
        for key, value in inheritance_mapping.items()
    )
    anchor_rows = [
        "| Source cosine anchor | Bray-Curtis anchor |",
        "|---:|---:|",
    ]
    anchor_rows.extend(
        f"| {float(key):.2f} | {float(value):.6f} |"
        for key, value in anchor_mapping.items()
    )
    baseline_rows = [
        "| Endpoint | Metric | Candidate | Events | Prevalence | Centered split-half r | Frozen gain A / B (nats) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for endpoint, table in (("F12", f12_shape), ("Strict F32", f32_shape)):
        for row in table.sort_values(["metric", "candidate"]).itertuples(index=False):
            baseline_rows.append(
                f"| {endpoint} | {row.metric} | {row.candidate} | {int(row.events):,} | "
                f"{row.prevalence:.5f} | {row.centered_branch_half_reliability:.3f} | "
                f"{row.log_loss_gain_A:.5f} / {row.log_loss_gain_B:.5f} |"
            )

    report_text = f"""# Threshold and alternative-metric sensitivity: results

**Date:** 2026-08-19  
**Status:** Reviewer-prompted post-hoc robustness analysis  
**Confirmatory status:** None; all cells are descriptive and cannot replace a registered gate.

## Bottom line

The requested grid is now complete: F12 inheritance cutoffs 0.85, 0.875,
0.90, 0.925, and 0.95; renewal runs 2–5; horizons 8, 10, 12, and 16;
strict coherent windows 6, 8, and 10; and old-anchor cutoffs corresponding to
0.80, 0.85, and 0.90. The whole grid was scored with cosine and with a
quantile-matched Bray-Curtis composition similarity.

Across all {len(f12)} metric-by-candidate F12 rows, the frozen composite's
worst branch-half gain was positive: **{str(f12_positive).lower()}**; every
whole-matrix 95% lower bound was positive: **{str(f12_ci_positive).lower()}**.
The worst-half gain range was **{_range(f12['worst_half_gain'], 5)} nats**.
Thus the F12 predictor conclusion is not a single-threshold or cosine-only
artifact within this grid.

Strict-event occurrence remained nonzero in at least one grid cell for every
metric/candidate combination: **{str(all(strict_nonzero.values())).lower()}**.
Because strict cells can be sparse, the complete event counts, event-positive
matrix counts, prevalence, reliability, and frozen predictor scores remain in
the CSV rather than being reduced to a pass/fail claim.

## Alternative metric and cutoff calibration

Bray-Curtis similarity is `1 - 0.5 * ||p-q||_1` for separately normalized
composition profiles. It is one minus total-variation distance. Numeric cosine
thresholds were not reused. Each Bray-Curtis cutoff was fixed at the empirical
percentile corresponding to its cosine cutoff, using similarity distributions
only and before strict-event readout.

{chr(10).join(mapping_rows)}

{chr(10).join(anchor_rows)}

The anchor mapping used the same independently replayed lineage centroids as
the existing reference analysis. Dominant components remained defined by the
frozen cosine-H090 reference procedure so that only the comparison metric—not
the objects being compared—changed.

## Registered-shape readback under both metrics

“Registered shape” means F12/F=12/run=3 or strict F32/run=8/anchor-equivalent
0.85, at the 0.90 source-threshold percentile. Only the cosine rows are the
literal registered endpoints; Bray-Curtis rows are robustness analogues.

{chr(10).join(baseline_rows)}

At the strict registered shape, branch-label Jaccard agreement between cosine
and Bray-Curtis was **{_range(strict_agreement['event_jaccard'], 3)}** across
the two candidates; raw branch agreement was
**{_range(strict_agreement['branch_label_agreement'], 3)}**. Divergence here
is scientifically informative: percentile-matched metrics can preserve event
frequency while changing which individual futures qualify.

## What each figure shows

- `figure_1_f12_threshold_metric.png`: all thresholds, runs, horizons, and both
  metrics; each cell is the minimum composite-over-history gain across the two
  candidates and fixed branch halves.
- `figure_2_strict_threshold_metric.png`: strict-event prevalence for all
  thresholds, windows, anchor cutoffs, and both metrics.
- `figure_3_cross_metric_agreement.png`: cosine versus Bray-Curtis prevalence
  for every F12 and strict grid cell, candidate-separated.

## CR1 intervention sensitivity

The archived CR1 intervention trajectories contain scalar cosine H but not
parent/daughter compositions. They were therefore rescored over the full new
80-cell cosine F12 grid, including run 5 and exact 0.875/0.925 cutoffs. The
complete four contrasts, candidates, and branch halves are in
`cr1_cosine_sensitivity.csv`. No alternative-metric intervention claim is made.

## Interpretation boundary

This result supports qualitative robustness within a deliberately small,
reviewer-motivated neighborhood. It does not identify a uniquely natural
threshold, make Bray-Curtis superior to cosine, prospectively validate any
alternate endpoint, or permit choosing the most favorable cell. The strict
event should continue to be discussed with its absolute event support and
split-half reliability, especially at the most stringent cells.

## Provenance and verification

- No new state, matrix, branch, seed, or model fit was introduced.
- Missing composition-level comparisons were recovered by exact deterministic
  replay of the already scored seed streams.
- Cosine boundary trajectories and registered labels/onsets were required to
  reproduce the sealed archives exactly.
- Every new write is below this analysis directory; the manuscript was not
  modified.
"""
    (OUTPUT_ROOT / "RESULTS_REPORT.md").write_text(report_text, encoding="utf-8")

    suggested = f"""# Suggested manuscript and reviewer-response language

## Supplementary Methods: post-hoc endpoint robustness

After the registered analyses, we locked a reviewer-prompted post-hoc
robustness grid before opening its new cells. For the F12 renewal endpoint we
crossed strict parent-to-selected-daughter inheritance cutoffs 0.85, 0.875,
0.90, 0.925, and 0.95; horizons 8, 10, 12, and 16; and renewed runs of 2–5.
For the strict coherent-regime endpoint we crossed the same source cutoffs,
coherent windows of 6, 8, and 10, and inclusive old-anchor cutoffs 0.80, 0.85,
and 0.90. Archived predictors were applied unchanged, candidates and fixed
branch halves remained separate, and uncertainty resampled whole catalytic
matrices.

As an alternative composition metric we used Bray-Curtis similarity on
separately normalized profiles, `1 - 0.5 ||p-q||_1`. To avoid treating cosine
and Bray-Curtis numbers as commensurate, each alternative cutoff was matched to
the pooled empirical percentile of its cosine counterpart using similarity
distributions only. Composition comparisons absent from the compact archive
were recovered by exact replay of registered seed streams; no states, futures,
or model fits were added. These analyses are descriptive robustness checks,
not new confirmatory gates.

## Results insertion

Across the complete F12 threshold/run/horizon grid and both cosine and
quantile-matched Bray-Curtis definitions, the minimum frozen
composite-over-history log-loss gain across candidates and branch halves was
{f12['worst_half_gain'].min():.5f} nats, and all matrix-bootstrap 95% lower
bounds remained positive. Strict coherent-event prevalence changed with the
cutoff, window length, anchor criterion, and metric, but events remained
observable in the grid for both implementations. We report all cellwise event
counts, event-positive matrix counts, split-half reliability, and predictor
scores in Supplementary Table X rather than selecting a favorable definition.

## Reviewer response

We agree that transparency about operational thresholds should be accompanied
by a direct robustness display. We added compact no-refit sensitivity plots for
the requested threshold and run grids and repeated the endpoint scoring with a
quantile-matched Bray-Curtis composition similarity. The new analysis retains
the registered conclusions as registered, labels every alternate cell post
hoc, and shows both stability and metric-dependent changes in event identity.
"""
    (OUTPUT_ROOT / "SUGGESTED_TEXT.md").write_text(suggested, encoding="utf-8")

    manifest = {
        "format": RESULT_FORMAT,
        "protocol_id": _protocol_id(),
        "f12_definitions_per_metric": len(F12_DEFINITIONS),
        "f32_definitions_per_metric": len(F32_DEFINITIONS),
        "metrics": ["cosine", "bray_curtis"],
        "models_refit_or_recalibrated": False,
        "new_random_futures": False,
        "source_or_manuscript_files_modified": False,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    _write_json(OUTPUT_ROOT / "manifest.json", manifest)
    checksum = OUTPUT_ROOT / "SHA256SUMS"
    if checksum.exists():
        checksum.unlink()
    write_checksums(OUTPUT_ROOT)
    print(f"Figures and reports written under {OUTPUT_ROOT}", flush=True)


def verify() -> None:
    protocol = verify_protocol()
    verify_checksums(CALIBRATION_ROOT)
    verify_checksums(OUTPUT_ROOT)
    f12_audit = json.loads(
        (REPLAY_ROOT / "f12_metric_audit.json").read_text(encoding="utf-8")
    )
    reference_audit = json.loads(
        (REPLAY_ROOT / "reference_metric_audit.json").read_text(encoding="utf-8")
    )
    f32_audit = json.loads(
        (REPLAY_ROOT / "f32_metric_audit.json").read_text(encoding="utf-8")
    )
    readback = json.loads(
        (OUTPUT_ROOT / "metric_readback_audit.json").read_text(encoding="utf-8")
    )
    manifest = json.loads((OUTPUT_ROOT / "manifest.json").read_text(encoding="utf-8"))
    figures = [
        OUTPUT_ROOT / "figure_1_f12_threshold_metric.png",
        OUTPUT_ROOT / "figure_2_strict_threshold_metric.png",
        OUTPUT_ROOT / "figure_3_cross_metric_agreement.png",
    ]
    checks = {
        "protocol_current": protocol["protocol_current"],
        "f12_replay_exact": f12_audit["state_ids_exact"]
        and f12_audit["cosine_boundary_exact"],
        "reference_replay_exact": all(
            reference_audit[key]
            for key in (
                "state_ids_exact",
                "cosine_centroid_similarity_exact",
                "component_sizes_exact",
                "completion_exact",
            )
        ),
        "f32_replay_exact": all(f32_audit.values()),
        "metric_readback": readback["all_checks_passed"],
        "figures_present": all(path.is_file() and path.stat().st_size > 10_000 for path in figures),
        "result_contract": manifest["f12_definitions_per_metric"] == 80
        and manifest["f32_definitions_per_metric"] == 45
        and manifest["models_refit_or_recalibrated"] is False
        and manifest["new_random_futures"] is False,
    }
    audit = {"checks": checks, "all_checks_passed": all(checks.values())}
    _write_json(OUTPUT_ROOT / "verification.json", audit)
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
        description="Run the post-hoc threshold and alternative-metric sensitivity"
    )
    subcommands = value.add_subparsers(dest="command", required=True)
    subcommands.add_parser("prepare")
    foundation = subcommands.add_parser("replay-foundation")
    foundation.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 12))
    subcommands.add_parser("calibrate")
    f32 = subcommands.add_parser("replay-f32")
    f32.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 12))
    subcommands.add_parser("analyze")
    subcommands.add_parser("report")
    subcommands.add_parser("verify")
    all_parser = subcommands.add_parser("all")
    all_parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 12))
    return value


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.command == "prepare":
        prepare()
    elif args.command == "replay-foundation":
        replay_foundation(args.workers)
    elif args.command == "calibrate":
        calibrate()
    elif args.command == "replay-f32":
        replay_f32(args.workers)
    elif args.command == "analyze":
        analyze()
    elif args.command == "report":
        report()
    elif args.command == "verify":
        verify()
    elif args.command == "all":
        prepare()
        replay_foundation(args.workers)
        calibrate()
        replay_f32(args.workers)
        analyze()
        report()
        verify()
    else:  # pragma: no cover
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
