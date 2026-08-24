"""Checkpointed strict-event geometry and target-specific prediction audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import warnings
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

import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits

from plastic_heredity.config import CANDIDATES, ExperimentConfig
from plastic_heredity.experiment import StateCase
from plastic_heredity.mechanistic import verify_checksums, write_checksums
from plastic_heredity.mechanistic_metrics import (
    _paired_gain,
    _reliability_bootstrap,
    _state_brier,
    _state_log_loss,
    holm_adjust,
    paired_matrix_randomization_p,
)
from plastic_heredity.mechanistic_v2_features import MechanisticV2RawFeatures
from plastic_heredity.mechanistic_v2_models import (
    CandidateRegistryV2,
    fit_candidate_registry_v2,
    load_registries_v2,
    predict_candidate_registry_v2,
    save_registries_v2,
)
from plastic_heredity.metrics import centered_spearman, log_loss_from_q, q_brier, spearman
from plastic_heredity.regime_confirmation import (
    BOOTSTRAP_MASTER_SEED,
    CONFIRMATION_MASTER_SEED,
    DEVELOPMENT_MASTER_SEED,
    RANDOMIZATION_MASTER_SEED,
    _experiment as regime_experiment,
)
from plastic_heredity.seeds import derive_seed
from plastic_heredity.simulator import simulate_future_absorbing
from reviewer_threshold_sensitivity_response.run_sensitivity import (
    _build_cohort_parallel,
)

from strict_core import (
    CROSS_EVAL_NAMES,
    GATE_NAMES,
    HORIZON,
    SPEC_BRAY_GLOBAL,
    SPEC_BRAY_RELATION,
    SPEC_COSINE,
    SPEC_NAMES,
    WINDOW_STAT_NAMES,
    EndpointSpec,
    calibration_comparisons,
    match_event_controls,
    quantile_match,
    score_all_specs,
)


ARTIFACT_ROOT = TASK_ROOT / "artifacts"
PROTOCOL_ROOT = ARTIFACT_ROOT / "protocol"
WORK_ROOT = ARTIFACT_ROOT / "work"
CALIBRATION_ROOT = ARTIFACT_ROOT / "calibration"
REPLAY_ROOT = ARTIFACT_ROOT / "replays"
MODEL_ROOT = ARTIFACT_ROOT / "models"
OUTPUT_ROOT = ARTIFACT_ROOT / "output"

DEVELOPMENT_SOURCE = SOURCE_ROOT / "results" / "regime_development"
CONFIRMATION_SOURCE = SOURCE_ROOT / "results" / "regime_confirmation"
PRIOR_AUDIT_ROOT = (
    PAPER_ROOT / "reviewer_threshold_metric_sensitivity_extension" / "artifacts"
)
PRIOR_GLOBAL_CALIBRATION = PRIOR_AUDIT_ROOT / "calibration" / "metric_calibration.json"
PRIOR_CONFIRMATION_REPLAY = PRIOR_AUDIT_ROOT / "replays" / "f32_metric.npz"
PRIOR_F32_DEFINITIONS = PRIOR_AUDIT_ROOT / "output" / "f32_definitions.csv"

CALIBRATION_BRANCHES = tuple(range(0, 64, 8)) + tuple(range(64, 128, 8))
BOOTSTRAP_REPETITIONS = 4096
RANDOMIZATION_REPETITIONS = 4096
MINIMUM_EVENTS = 100
MINIMUM_EVENT_MATRICES = 20
CHECKPOINT_FORMAT = "strict-event-geometry-checkpoint-v1"
RESULT_FORMAT = "strict-event-geometry-audit-v1"
MATCHING_SEED = "913c1dc424b18e707a3924d6eaccf1be68403e70d0c089e46c7e0458bf9db6b"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


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


def _replace_checksums(directory: Path) -> None:
    checksum = directory / "SHA256SUMS"
    if checksum.exists():
        checksum.unlink()
    write_checksums(directory)


def _scientific_source_contract() -> dict[str, dict[str, str]]:
    paths = {
        "strict_core": TASK_ROOT / "strict_core.py",
        "simulator": SOURCE_ROOT / "plastic_heredity" / "simulator.py",
        "seeds": SOURCE_ROOT / "plastic_heredity" / "seeds.py",
        "regime_confirmation": SOURCE_ROOT / "plastic_heredity" / "regime_confirmation.py",
        "features": SOURCE_ROOT / "plastic_heredity" / "mechanistic_v2_features.py",
        "models": SOURCE_ROOT / "plastic_heredity" / "mechanistic_v2_models.py",
        "development_checksums": DEVELOPMENT_SOURCE / "SHA256SUMS",
        "confirmation_checksums": CONFIRMATION_SOURCE / "SHA256SUMS",
        "prior_global_calibration": PRIOR_GLOBAL_CALIBRATION,
        "prior_confirmation_replay": PRIOR_CONFIRMATION_REPLAY,
        "prior_f32_definitions": PRIOR_F32_DEFINITIONS,
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing required read-only sources: {missing}")
    return {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for name, path in paths.items()
    }


def _verify_sources() -> dict[str, int]:
    return {
        "development": len(verify_checksums(DEVELOPMENT_SOURCE)),
        "confirmation": len(verify_checksums(CONFIRMATION_SOURCE)),
        "prior_audit": len(verify_checksums(PRIOR_AUDIT_ROOT / "output")),
    }


def _protocol() -> dict[str, Any]:
    value: dict[str, Any] = {
        "format": "strict-event-geometry-posthoc-protocol-v1",
        "date": "2026-08-19",
        "status": "post_hoc_locked_before_relation_specific_cutoff_or_model_readout",
        "working_boundary": {
            "all_new_writes_below": str(TASK_ROOT.resolve()),
            "existing_sources_read_only": True,
            "manuscript_modified": False,
        },
        "question": (
            "Why does the registered strict event change under Bray-Curtis, "
            "and does state information predict a Bray-specific event when the "
            "models are trained for that target?"
        ),
        "endpoints": [
            {
                "name": SPEC_COSINE,
                "metric": "cosine",
                "inheritance": 0.90,
                "coherence": 0.90,
                "anchor": 0.85,
            },
            {
                "name": SPEC_BRAY_GLOBAL,
                "metric": "bray_curtis",
                "cutoffs": "read-only prior globally percentile-mapped analysis",
            },
            {
                "name": SPEC_BRAY_RELATION,
                "metric": "bray_curtis",
                "cutoffs": "development-only relation-specific percentile mapping",
            },
        ],
        "shared_shape": {
            "horizon": HORIZON,
            "window": 8,
            "selected_lineage": True,
            "inheritance_and_coherence": "strict similarity > cutoff",
            "break_and_anchor": "inclusive similarity <= cutoff",
        },
        "calibration": {
            "cohort": "REGDEV only",
            "branches_per_state": CALIBRATION_BRANCHES,
            "boundary_objects": "all observed parent-to-selected-daughter boundaries",
            "precursor": (
                "registered cosine first break followed by a run of eight "
                "strict H>0.90 boundaries, before coherence or anchor gates"
            ),
            "coherence_objects": (
                "union of unique unordered daughter pairs appearing in any "
                "precursor window within a branch"
            ),
            "anchor_objects": (
                "registered cosine first-break parent versus the union of unique "
                "daughters appearing in precursor windows"
            ),
            "quantile_method": "inverted empirical CDF",
            "event_prevalence_matched": False,
            "confirmation_used": False,
        },
        "failure_gates": list(GATE_NAMES),
        "nondegeneracy": {
            "statistics": list(WINDOW_STAT_NAMES),
            "dominance_share": 0.80,
            "positive_window": "earliest qualifying event window",
            "control_window": "earliest break-plus-inherited-run8 precursor",
            "control_matching": (
                "one negative precursor branch per positive, same state/spec, "
                "without replacement, frozen SHA256 order"
            ),
            "matching_seed": MATCHING_SEED,
        },
        "models": {
            "suite": "original candidate-separated no-PCA nested offset-ridge suite",
            "primary_comparison": "h10_state versus h10",
            "targets": list(SPEC_NAMES),
            "development_only_fit_and_selection": True,
            "seal_before_confirmation_replay": True,
            "transfer_control": "cosine-trained predictions scored on both Bray labels",
        },
        "inference": {
            "unit": "catalytic matrix",
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "randomization_repetitions": RANDOMIZATION_REPETITIONS,
            "holm_family": "four candidate-by-half cells within each target-matched endpoint",
            "power_rule": {
                "events": MINIMUM_EVENTS,
                "event_matrices": MINIMUM_EVENT_MATRICES,
                "required_in": ["development", "confirmation"],
            },
            "confirmatory": False,
        },
        "replay": {
            "classification": (
                "exact deterministic replay of registered development and "
                "confirmation seed streams; no new futures"
            ),
            "checkpointed_by_state": True,
            "cosine_label_and_onset_readback_required": True,
            "prior_global_Bray_confirmation_label_readback_required": True,
        },
        "claim_boundary": (
            "The analysis is mechanistic and predictive diagnosis, not causal "
            "intervention evidence, prospective confirmation, or both-daughter fidelity."
        ),
        "scientific_source_contract": _scientific_source_contract(),
    }
    value["protocol_id"] = canonical_digest(value)
    return _json_ready(value)


def prepare() -> None:
    _verify_sources()
    protocol = _protocol()
    PROTOCOL_ROOT.mkdir(parents=True, exist_ok=True)
    path = PROTOCOL_ROOT / "analysis_protocol.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != protocol:
            raise ValueError("existing protocol differs from the scientific contract")
        print(f"Protocol already frozen and identical: {path}", flush=True)
        return
    _write_json(path, protocol)
    _replace_checksums(PROTOCOL_ROOT)
    print(f"Frozen strict-event audit protocol: {path}", flush=True)


def verify_protocol() -> dict[str, Any]:
    verify_checksums(PROTOCOL_ROOT)
    saved = json.loads((PROTOCOL_ROOT / "analysis_protocol.json").read_text())
    if saved != _protocol():
        raise ValueError("scientific protocol, core, or input identity changed")
    return {
        "protocol_id": saved["protocol_id"],
        "protocol_current": True,
        "source_archives": _verify_sources(),
    }


def _protocol_id() -> str:
    path = PROTOCOL_ROOT / "analysis_protocol.json"
    if not path.is_file():
        raise FileNotFoundError("run prepare before generating replay data")
    return str(json.loads(path.read_text())["protocol_id"])


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


def _cases(cohort: str, workers: int) -> tuple[ExperimentConfig, list[StateCase]]:
    if cohort == "development":
        experiment = regime_experiment(DEVELOPMENT_MASTER_SEED)
        label = "REGDEV"
        config = experiment.development
    elif cohort == "confirmation":
        experiment = regime_experiment(CONFIRMATION_MASTER_SEED)
        label = "REGCONF"
        config = experiment.confirmation
    else:
        raise ValueError(cohort)
    return experiment, _build_cohort_parallel(experiment, label, config, workers)


def _metadata(cases: Sequence[StateCase]) -> dict[str, np.ndarray]:
    return {
        "state_ids": np.asarray([case.state_id for case in cases]),
        "candidates": np.asarray([case.candidate for case in cases]),
        "matrix_ids": np.asarray([case.matrix_id for case in cases], dtype=np.int16),
        "landmarks": np.asarray([case.landmark for case in cases], dtype=np.int16),
    }


def _future(case: StateCase, experiment: ExperimentConfig, branch: int):
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
    return simulate_future_absorbing(
        case.snapshot,
        case.beta,
        experiment.gard,
        CANDIDATES[case.candidate],
        HORIZON,
        rng,
    )


def _calibration_worker(arguments: tuple[int, StateCase, ExperimentConfig]) -> int:
    index, case, experiment = arguments
    dataset = "calibration"
    output = _checkpoint_path(dataset, index)
    if _checkpoint_complete(output, dataset, index):
        return index
    collected: dict[str, list[np.ndarray]] = {
        f"{relation}_{metric}": []
        for relation in ("boundary", "coherence", "anchor")
        for metric in ("cosine", "bray_curtis")
    }
    cosine_labels = np.zeros(len(CALIBRATION_BRANCHES), dtype=np.int8)
    cosine_onsets = np.full(len(CALIBRATION_BRANCHES), -1, dtype=np.int16)
    cosine_spec = EndpointSpec(SPEC_COSINE, "cosine", 0.90, 0.90, 0.85)
    with threadpool_limits(limits=1):
        for position, branch in enumerate(CALIBRATION_BRANCHES):
            records, _ = _future(case, experiment, branch)
            values = calibration_comparisons(records)
            for name, array in values.items():
                collected[name].append(array)
            outcome = score_all_specs(records, (cosine_spec,))[0][0]
            cosine_labels[position] = int(outcome.event)
            cosine_onsets[position] = outcome.onset
    arrays = {
        name: np.concatenate(parts) if parts else np.empty(0, dtype=np.float64)
        for name, parts in collected.items()
    }
    _atomic_npz(
        output,
        format=np.asarray(CHECKPOINT_FORMAT),
        dataset=np.asarray(dataset),
        state_index=np.asarray(index, dtype=np.int32),
        protocol_id=np.asarray(_protocol_id()),
        cosine_labels=cosine_labels,
        cosine_onsets=cosine_onsets,
        **arrays,
    )
    return index


def _global_cutoffs() -> tuple[float, float, float]:
    value = json.loads(PRIOR_GLOBAL_CALIBRATION.read_text())
    inheritance = float(
        value["inheritance_and_coherence_bray_cutoff_by_source_cosine"]["0.900"]
    )
    anchor = float(value["old_anchor_bray_cutoff_by_source_cosine"]["0.85"])
    return inheritance, inheritance, anchor


def _relation_cutoffs() -> tuple[float, float, float]:
    path = CALIBRATION_ROOT / "relation_specific_calibration.json"
    if not path.is_file():
        raise FileNotFoundError("relation-specific calibration is incomplete")
    verify_checksums(CALIBRATION_ROOT)
    value = json.loads(path.read_text())
    if value["protocol_id"] != _protocol_id():
        raise ValueError("calibration belongs to another protocol")
    return (
        float(value["boundary"]["target_cutoff"]),
        float(value["coherence"]["target_cutoff"]),
        float(value["anchor"]["target_cutoff"]),
    )


def _specs() -> tuple[EndpointSpec, ...]:
    global_boundary, global_coherence, global_anchor = _global_cutoffs()
    relation_boundary, relation_coherence, relation_anchor = _relation_cutoffs()
    return (
        EndpointSpec(SPEC_COSINE, "cosine", 0.90, 0.90, 0.85),
        EndpointSpec(
            SPEC_BRAY_GLOBAL,
            "bray_curtis",
            global_boundary,
            global_coherence,
            global_anchor,
        ),
        EndpointSpec(
            SPEC_BRAY_RELATION,
            "bray_curtis",
            relation_boundary,
            relation_coherence,
            relation_anchor,
        ),
    )


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
    print(
        f"[{dataset}] {len(arguments)-len(pending)}/{len(arguments)} present; "
        f"{len(pending)} pending",
        flush=True,
    )
    if not pending:
        return
    progress_every = max(1, len(pending) // 50)
    if workers <= 1:
        for count, item in enumerate(pending, 1):
            worker(item)
            if count % progress_every == 0 or count == len(pending):
                print(f"[{dataset}] {count}/{len(pending)} new states", flush=True)
        return
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(worker, item) for item in pending]
        for count, future in enumerate(as_completed(futures), 1):
            future.result()
            if count % progress_every == 0 or count == len(pending):
                print(f"[{dataset}] {count}/{len(pending)} new states", flush=True)


def calibrate(workers: int) -> None:
    verify_protocol()
    experiment, cases = _cases("development", workers)
    arguments = [(index, case, experiment) for index, case in enumerate(cases)]
    _run_workers("calibration", arguments, _calibration_worker, workers)
    comparisons: dict[str, list[np.ndarray]] = {
        f"{relation}_{metric}": []
        for relation in ("boundary", "coherence", "anchor")
        for metric in ("cosine", "bray_curtis")
    }
    observed_labels: list[np.ndarray] = []
    observed_onsets: list[np.ndarray] = []
    for index in range(len(cases)):
        path = _checkpoint_path("calibration", index)
        if not _checkpoint_complete(path, "calibration", index):
            raise ValueError(f"missing calibration checkpoint {index}")
        with np.load(path, allow_pickle=False) as archive:
            for name in comparisons:
                comparisons[name].append(np.asarray(archive[name]))
            observed_labels.append(np.asarray(archive["cosine_labels"]))
            observed_onsets.append(np.asarray(archive["cosine_onsets"]))
    merged = {name: np.concatenate(parts) for name, parts in comparisons.items()}
    labels = np.stack(observed_labels)
    onsets = np.stack(observed_onsets)
    with np.load(DEVELOPMENT_SOURCE / "development_arrays.npz", allow_pickle=False) as source:
        expected_labels = np.asarray(source["labels_primary_all8"])[
            :, CALIBRATION_BRANCHES
        ]
        expected_onsets = np.asarray(source["onsets"])[
            :, CALIBRATION_BRANCHES, 0
        ]
        expected_ids = np.asarray(source["state_ids"])
    audit = {
        "state_ids_exact": bool(
            np.array_equal(expected_ids, _metadata(cases)["state_ids"])
        ),
        "cosine_labels_exact": bool(np.array_equal(labels, expected_labels)),
        "cosine_onsets_exact": bool(np.array_equal(onsets, expected_onsets)),
        "calibration_states": len(cases),
        "branches_per_state": len(CALIBRATION_BRANCHES),
    }
    if not all(value for key, value in audit.items() if key.endswith("_exact")):
        raise AssertionError(f"calibration replay audit failed: {audit}")
    mapping = {
        "format": "relation-specific-bray-calibration-v1",
        "protocol_id": _protocol_id(),
        "boundary": quantile_match(
            merged["boundary_cosine"], merged["boundary_bray_curtis"], 0.90
        ),
        "coherence": quantile_match(
            merged["coherence_cosine"], merged["coherence_bray_curtis"], 0.90
        ),
        "anchor": quantile_match(
            merged["anchor_cosine"], merged["anchor_bray_curtis"], 0.85
        ),
        "event_prevalence_used": False,
        "confirmation_used": False,
    }
    CALIBRATION_ROOT.mkdir(parents=True, exist_ok=True)
    _write_json(CALIBRATION_ROOT / "relation_specific_calibration.json", mapping)
    _write_json(CALIBRATION_ROOT / "replay_audit.json", audit)
    _atomic_npz(
        CALIBRATION_ROOT / "comparison_distributions.npz",
        protocol_id=np.asarray(_protocol_id()),
        **merged,
    )
    _replace_checksums(CALIBRATION_ROOT)
    print(f"Relation-specific cutoffs frozen at {CALIBRATION_ROOT}", flush=True)


def _replay_worker(
    arguments: tuple[int, StateCase, ExperimentConfig, str, tuple[EndpointSpec, ...]]
) -> int:
    index, case, experiment, dataset, specs = arguments
    output = _checkpoint_path(dataset, index)
    if _checkpoint_complete(output, dataset, index):
        return index
    branches = experiment.confirmation.branches_per_state
    n_specs = len(specs)
    labels = np.zeros((branches, n_specs), dtype=np.int8)
    onsets = np.full((branches, n_specs), -1, dtype=np.int16)
    first_break = np.full_like(onsets, -1)
    first_run = np.full_like(onsets, -1)
    deepest_gate = np.zeros((branches, n_specs), dtype=np.int8)
    eligible_windows = np.zeros((branches, n_specs), dtype=np.int16)
    coherent_windows = np.zeros((branches, n_specs), dtype=np.int16)
    best_pairwise_margin = np.full((branches, n_specs), np.nan, dtype=np.float64)
    best_anchor_margin = np.full((branches, n_specs), np.nan, dtype=np.float64)
    precursor_stats = np.full(
        (branches, n_specs, len(WINDOW_STAT_NAMES)), np.nan, dtype=np.float64
    )
    event_stats = np.full_like(precursor_stats, np.nan)
    cross_eval = np.full(
        (branches, n_specs, n_specs, len(CROSS_EVAL_NAMES)),
        np.nan,
        dtype=np.float64,
    )
    completed = np.zeros(branches, dtype=np.int8)
    observed = np.zeros(branches, dtype=np.int16)
    with threadpool_limits(limits=1):
        for branch in range(branches):
            records, complete = _future(case, experiment, branch)
            outcomes, cross = score_all_specs(records, specs)
            completed[branch] = int(complete)
            observed[branch] = len(records)
            cross_eval[branch] = cross
            for spec_index, outcome in enumerate(outcomes):
                labels[branch, spec_index] = int(outcome.event)
                onsets[branch, spec_index] = outcome.onset
                first_break[branch, spec_index] = outcome.first_break
                first_run[branch, spec_index] = outcome.first_run
                deepest_gate[branch, spec_index] = outcome.deepest_gate
                eligible_windows[branch, spec_index] = outcome.eligible_windows
                coherent_windows[branch, spec_index] = outcome.coherent_windows
                best_pairwise_margin[branch, spec_index] = outcome.best_pairwise_margin
                best_anchor_margin[branch, spec_index] = outcome.best_anchor_margin
                precursor_stats[branch, spec_index] = outcome.precursor_stats
                event_stats[branch, spec_index] = outcome.event_stats
    _atomic_npz(
        output,
        format=np.asarray(CHECKPOINT_FORMAT),
        dataset=np.asarray(dataset),
        state_index=np.asarray(index, dtype=np.int32),
        protocol_id=np.asarray(_protocol_id()),
        labels=labels,
        onsets=onsets,
        first_break=first_break,
        first_run=first_run,
        deepest_gate=deepest_gate,
        eligible_windows=eligible_windows,
        coherent_windows=coherent_windows,
        best_pairwise_margin=best_pairwise_margin,
        best_anchor_margin=best_anchor_margin,
        precursor_stats=precursor_stats,
        event_stats=event_stats,
        cross_eval=cross_eval,
        completed=completed,
        observed=observed,
    )
    return index


REPLAY_KEYS = (
    "labels",
    "onsets",
    "first_break",
    "first_run",
    "deepest_gate",
    "eligible_windows",
    "coherent_windows",
    "best_pairwise_margin",
    "best_anchor_margin",
    "precursor_stats",
    "event_stats",
    "cross_eval",
    "completed",
    "observed",
)


def _assemble_replay(dataset: str, cases: Sequence[StateCase]) -> dict[str, np.ndarray]:
    values: dict[str, list[np.ndarray]] = {key: [] for key in REPLAY_KEYS}
    for index in range(len(cases)):
        path = _checkpoint_path(dataset, index)
        if not _checkpoint_complete(path, dataset, index):
            raise ValueError(f"missing {dataset} checkpoint {index}")
        with np.load(path, allow_pickle=False) as archive:
            for key in REPLAY_KEYS:
                values[key].append(np.asarray(archive[key]))
    return {key: np.stack(parts) for key, parts in values.items()}


def _prior_global_index() -> int:
    table = pd.read_csv(PRIOR_F32_DEFINITIONS)
    selected = table.loc[
        (table["metric"] == "bray_curtis")
        & table["registered_shape"].astype(bool)
    ]
    if len(selected) != 1:
        raise ValueError("cannot identify prior registered-shape Bray definition")
    return int(selected.iloc[0]["definition_index"])


def replay(cohort: str, workers: int) -> None:
    verify_protocol()
    if cohort == "confirmation":
        verify_model_seal()
    specs = _specs()
    experiment, cases = _cases(cohort, workers)
    dataset = f"{cohort}_replay"
    arguments = [
        (index, case, experiment, dataset, specs)
        for index, case in enumerate(cases)
    ]
    _run_workers(dataset, arguments, _replay_worker, workers)
    arrays = _assemble_replay(dataset, cases)
    source_root = DEVELOPMENT_SOURCE if cohort == "development" else CONFIRMATION_SOURCE
    source_file = (
        source_root / "development_arrays.npz"
        if cohort == "development"
        else source_root / "confirmation_arrays.npz"
    )
    with np.load(source_file, allow_pickle=False) as source:
        expected_ids = np.asarray(source["state_ids"])
        expected_labels = np.asarray(source["labels_primary_all8"])
        expected_onsets = np.asarray(source["onsets"])[:, :, 0]
        expected_completed = np.asarray(source["completed_horizon"])
        expected_observed = np.asarray(source["observed_fissions"])
    audit: dict[str, Any] = {
        "state_ids_exact": bool(
            np.array_equal(expected_ids, _metadata(cases)["state_ids"])
        ),
        "cosine_labels_exact": bool(np.array_equal(expected_labels, arrays["labels"][:, :, 0])),
        "cosine_onsets_exact": bool(np.array_equal(expected_onsets, arrays["onsets"][:, :, 0])),
        "completed_exact": bool(np.array_equal(expected_completed, arrays["completed"])),
        "observed_exact": bool(np.array_equal(expected_observed, arrays["observed"])),
    }
    if cohort == "confirmation":
        prior = _load_npz(PRIOR_CONFIRMATION_REPLAY)
        index = _prior_global_index()
        audit["global_bray_labels_exact"] = bool(
            np.array_equal(prior["labels_bray_curtis"][:, :, index], arrays["labels"][:, :, 1])
        )
        audit["global_bray_onsets_exact"] = bool(
            np.array_equal(prior["onsets_bray_curtis"][:, :, index], arrays["onsets"][:, :, 1])
        )
    if not all(value for key, value in audit.items() if key.endswith("_exact")):
        raise AssertionError(f"{cohort} replay audit failed: {audit}")
    REPLAY_ROOT.mkdir(parents=True, exist_ok=True)
    _atomic_npz(
        REPLAY_ROOT / f"{cohort}.npz",
        protocol_id=np.asarray(_protocol_id()),
        spec_names=np.asarray(SPEC_NAMES),
        window_stat_names=np.asarray(WINDOW_STAT_NAMES),
        cross_eval_names=np.asarray(CROSS_EVAL_NAMES),
        **_metadata(cases),
        **arrays,
    )
    _write_json(REPLAY_ROOT / f"{cohort}_audit.json", audit)
    _replace_checksums(REPLAY_ROOT)
    print(f"{cohort.title()} replay saved under {REPLAY_ROOT}", flush=True)


def _raw_features(path: Path) -> tuple[MechanisticV2RawFeatures, dict[str, np.ndarray]]:
    arrays = _load_npz(path)
    raw = MechanisticV2RawFeatures(
        h10=np.asarray(arrays["h10"], dtype=np.float64),
        state=np.asarray(arrays["state_block"], dtype=np.float64),
        beta=np.asarray(arrays["beta_block"], dtype=np.float64),
        interaction=np.asarray(arrays["interaction_block"], dtype=np.float64),
    )
    return raw, arrays


def _fit_registries(
    raw: MechanisticV2RawFeatures,
    candidates: np.ndarray,
    matrix_ids: np.ndarray,
    labels: np.ndarray,
) -> dict[str, CandidateRegistryV2]:
    output: dict[str, CandidateRegistryV2] = {}
    for candidate in ("02", "03"):
        selected = np.asarray(candidates).astype(str) == candidate
        output[candidate] = fit_candidate_registry_v2(
            candidate,
            raw.selected(selected),
            labels[selected],
            np.asarray(matrix_ids[selected], dtype=np.int64),
        )
    return output


def _predict_registries(
    registries: Mapping[str, CandidateRegistryV2],
    raw: MechanisticV2RawFeatures,
    candidates: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    output: dict[str, dict[str, np.ndarray]] = {}
    for candidate in ("02", "03"):
        selected = np.asarray(candidates).astype(str) == candidate
        output[candidate] = predict_candidate_registry_v2(
            registries[candidate], raw.selected(selected)
        )
    return output


def _model_paths(spec: str) -> tuple[Path, Path]:
    return MODEL_ROOT / f"models_{spec}.npz", MODEL_ROOT / f"contract_{spec}.json"


def fit_and_seal() -> None:
    verify_protocol()
    verify_checksums(CALIBRATION_ROOT)
    verify_checksums(REPLAY_ROOT)
    replay_path = REPLAY_ROOT / "development.npz"
    if not replay_path.is_file():
        raise FileNotFoundError("development replay must finish before fitting")
    development = _load_npz(replay_path)
    raw, source = _raw_features(DEVELOPMENT_SOURCE / "development_arrays.npz")
    if not np.array_equal(development["state_ids"], source["state_ids"]):
        raise AssertionError("development feature and replay state order differ")
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    registries_by_spec: dict[str, dict[str, CandidateRegistryV2]] = {}
    for spec_index, spec in enumerate(SPEC_NAMES):
        print(f"[models] fitting {spec}", flush=True)
        registries = _fit_registries(
            raw,
            source["candidates"],
            source["matrix_ids"],
            development["labels"][:, :, spec_index],
        )
        archive, contract = _model_paths(spec)
        save_registries_v2(archive, contract, registries)
        registries_by_spec[spec] = registries

    portable_errors: dict[str, dict[str, dict[str, float]]] = {}
    for spec in SPEC_NAMES:
        archive, contract = _model_paths(spec)
        loaded = load_registries_v2(archive, contract)
        left = _predict_registries(
            registries_by_spec[spec], raw, source["candidates"]
        )
        right = _predict_registries(loaded, raw, source["candidates"])
        portable_errors[spec] = {
            candidate: {
                model: float(np.max(np.abs(left[candidate][model] - right[candidate][model])))
                for model in left[candidate]
            }
            for candidate in ("02", "03")
        }
    portable_max = max(
        value
        for by_candidate in portable_errors.values()
        for by_model in by_candidate.values()
        for value in by_model.values()
    )

    original = load_registries_v2(
        DEVELOPMENT_SOURCE / "models_primary_all8.npz",
        DEVELOPMENT_SOURCE / "model_contract_primary_all8.json",
    )
    original_predictions = _predict_registries(original, raw, source["candidates"])
    refit_predictions = _predict_registries(
        registries_by_spec[SPEC_COSINE], raw, source["candidates"]
    )
    positive_control_errors = {
        candidate: {
            model: float(
                np.max(
                    np.abs(
                        original_predictions[candidate][model]
                        - refit_predictions[candidate][model]
                    )
                )
            )
            for model in original_predictions[candidate]
        }
        for candidate in ("02", "03")
    }
    positive_control_max = max(
        value
        for by_model in positive_control_errors.values()
        for value in by_model.values()
    )
    seal: dict[str, Any] = {
        "format": "strict-event-target-model-seal-v1",
        "status": "sealed_before_relation_specific_confirmation_scoring",
        "protocol_id": _protocol_id(),
        "development_replay_sha256": sha256_file(replay_path),
        "calibration_sha256": sha256_file(
            CALIBRATION_ROOT / "relation_specific_calibration.json"
        ),
        "targets": list(SPEC_NAMES),
        "selected_lambdas": {
            spec: {
                candidate: registries_by_spec[spec][candidate].selected_lambdas
                for candidate in ("02", "03")
            }
            for spec in SPEC_NAMES
        },
        "portable_prediction_max_absolute_error": portable_max,
        "portable_prediction_errors": portable_errors,
        "cosine_refit_positive_control_max_absolute_error": positive_control_max,
        "cosine_refit_positive_control_errors": positive_control_errors,
        "confirmation_labels_or_outcomes_loaded": False,
    }
    seal["seal_id"] = canonical_digest(seal)
    _write_json(MODEL_ROOT / "model_seal.json", seal)
    _replace_checksums(MODEL_ROOT)
    if portable_max > 1e-12 or positive_control_max > 1e-12:
        raise AssertionError(
            f"model portability/control audit failed: {portable_max}, {positive_control_max}"
        )
    print(f"Target-specific models sealed at {MODEL_ROOT}", flush=True)


def verify_model_seal() -> dict[str, Any]:
    verify_checksums(MODEL_ROOT)
    seal = json.loads((MODEL_ROOT / "model_seal.json").read_text())
    seal_id = seal.pop("seal_id")
    if canonical_digest(seal) != seal_id:
        raise ValueError("model seal identifier mismatch")
    seal["seal_id"] = seal_id
    if seal["protocol_id"] != _protocol_id():
        raise ValueError("model seal belongs to another protocol")
    if seal["development_replay_sha256"] != sha256_file(REPLAY_ROOT / "development.npz"):
        raise ValueError("development replay changed after model seal")
    if seal["calibration_sha256"] != sha256_file(
        CALIBRATION_ROOT / "relation_specific_calibration.json"
    ):
        raise ValueError("relation calibration changed after model seal")
    if seal["confirmation_labels_or_outcomes_loaded"] is not False:
        raise ValueError("model seal does not preserve the development-only boundary")
    return seal


def _load_model_predictions(
    raw: MechanisticV2RawFeatures, candidates: np.ndarray
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    output: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for spec in SPEC_NAMES:
        archive, contract = _model_paths(spec)
        output[spec] = _predict_registries(
            load_registries_v2(archive, contract), raw, candidates
        )
    return output


def _power_table(
    cohort: str,
    labels: np.ndarray,
    candidates: np.ndarray,
    matrix_ids: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec_index, spec in enumerate(SPEC_NAMES):
        for candidate in ("02", "03"):
            selected = np.asarray(candidates).astype(str) == candidate
            values = labels[selected, :, spec_index]
            mids = np.asarray(matrix_ids[selected], dtype=np.int64)
            event_matrices = int(
                sum(values[mids == matrix].any() for matrix in np.unique(mids))
            )
            events = int(values.sum())
            rows.append(
                {
                    "cohort": cohort,
                    "spec": spec,
                    "candidate": candidate,
                    "states": int(selected.sum()),
                    "branches": int(values.size),
                    "events": events,
                    "prevalence": float(values.mean()),
                    "event_matrices": event_matrices,
                    "power_adequate": events >= MINIMUM_EVENTS
                    and event_matrices >= MINIMUM_EVENT_MATRICES,
                }
            )
    return pd.DataFrame(rows)


def _prediction_rows(
    labels: np.ndarray,
    candidates: np.ndarray,
    matrix_ids: np.ndarray,
    predictions: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
    development_power: pd.DataFrame,
    confirmation_power: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    for target_index, target in enumerate(SPEC_NAMES):
        for candidate in ("02", "03"):
            selected = np.asarray(candidates).astype(str) == candidate
            values = labels[selected, :, target_index]
            mids = np.asarray(matrix_ids[selected], dtype=np.int64)
            q_values = {
                "A": values[:, : values.shape[1] // 2].mean(axis=1),
                "B": values[:, values.shape[1] // 2 :].mean(axis=1),
            }
            for training_target in SPEC_NAMES:
                model_values = predictions[training_target][candidate]
                for half, q in q_values.items():
                    branch_slice = (
                        slice(0, values.shape[1] // 2)
                        if half == "A"
                        else slice(values.shape[1] // 2, values.shape[1])
                    )
                    seed = (target, training_target, candidate, half)
                    gain, interval = _paired_gain(
                        q,
                        model_values["h10"],
                        model_values["h10_state"],
                        mids,
                        _state_log_loss,
                        BOOTSTRAP_REPETITIONS,
                        np.random.default_rng(
                            derive_seed(BOOTSTRAP_MASTER_SEED, "strict_audit.log", *seed)
                        ),
                    )
                    brier, brier_interval = _paired_gain(
                        q,
                        model_values["h10"],
                        model_values["h10_state"],
                        mids,
                        _state_brier,
                        BOOTSTRAP_REPETITIONS,
                        np.random.default_rng(
                            derive_seed(BOOTSTRAP_MASTER_SEED, "strict_audit.brier", *seed)
                        ),
                    )
                    p_value = paired_matrix_randomization_p(
                        q,
                        model_values["h10"],
                        model_values["h10_state"],
                        mids,
                        RANDOMIZATION_REPETITIONS,
                        np.random.default_rng(
                            derive_seed(RANDOMIZATION_MASTER_SEED, "strict_audit", *seed)
                        ),
                    )
                    dev_ok = bool(
                        development_power.loc[
                            (development_power["spec"] == target)
                            & (development_power["candidate"] == candidate),
                            "power_adequate",
                        ].iloc[0]
                    )
                    conf_ok = bool(
                        confirmation_power.loc[
                            (confirmation_power["spec"] == target)
                            & (confirmation_power["candidate"] == candidate),
                            "power_adequate",
                        ].iloc[0]
                    )
                    rows.append(
                        {
                            "evaluation_target": target,
                            "training_target": training_target,
                            "target_matched": target == training_target,
                            "candidate": candidate,
                            "half": half,
                            "events_in_half": int(values[:, branch_slice].sum()),
                            "development_power_adequate": dev_ok,
                            "confirmation_power_adequate": conf_ok,
                            "log_loss_h10": log_loss_from_q(q, model_values["h10"]),
                            "log_loss_h10_state": log_loss_from_q(q, model_values["h10_state"]),
                            "log_loss_gain": gain,
                            "log_loss_gain_ci95_lower": interval[0],
                            "log_loss_gain_ci95_upper": interval[1],
                            "q_brier_gain": brier,
                            "q_brier_gain_ci95_lower": brier_interval[0],
                            "q_brier_gain_ci95_upper": brier_interval[1],
                            "randomization_p_raw": p_value,
                        }
                    )
                    for model, prediction in model_values.items():
                        inventory.append(
                            {
                                "evaluation_target": target,
                                "training_target": training_target,
                                "candidate": candidate,
                                "half": half,
                                "model": model,
                                "log_loss": log_loss_from_q(q, prediction),
                                "q_brier": q_brier(q, prediction),
                                "spearman": spearman(prediction, q),
                                "centered_spearman": centered_spearman(prediction, q, mids),
                            }
                        )
    table = pd.DataFrame(rows)
    table["randomization_p_holm"] = np.nan
    table["passes_exploratory_gate"] = False
    for target in SPEC_NAMES:
        selected = (table["evaluation_target"] == target) & table["target_matched"]
        adjusted = holm_adjust(table.loc[selected, "randomization_p_raw"].tolist())
        table.loc[selected, "randomization_p_holm"] = adjusted
        table.loc[selected, "passes_exploratory_gate"] = (
            table.loc[selected, "development_power_adequate"]
            & table.loc[selected, "confirmation_power_adequate"]
            & (table.loc[selected, "log_loss_gain"] > 0.0)
            & (table.loc[selected, "log_loss_gain_ci95_lower"] > 0.0)
            & (table.loc[selected, "randomization_p_holm"] < 0.05)
        )
    return table, pd.DataFrame(inventory)


def _gate_table(cohort: str, replay: Mapping[str, np.ndarray]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    stages = (
        ("observed", 0),
        ("break", 1),
        ("inherited_run8", 2),
        ("coherent_window", 3),
        ("strict_event", 4),
    )
    candidates = np.asarray(replay["candidates"]).astype(str)
    for spec_index, spec in enumerate(SPEC_NAMES):
        for candidate in ("02", "03"):
            selected = candidates == candidate
            gates = replay["deepest_gate"][selected, :, spec_index]
            total = int(gates.size)
            for stage, minimum in stages:
                reached = total if stage == "observed" else int(np.count_nonzero(gates >= minimum))
                rows.append(
                    {
                        "cohort": cohort,
                        "spec": spec,
                        "candidate": candidate,
                        "stage": stage,
                        "stage_order": minimum,
                        "branches": total,
                        "reached": reached,
                        "fraction": reached / total,
                    }
                )
            for code, name in enumerate(GATE_NAMES):
                rows.append(
                    {
                        "cohort": cohort,
                        "spec": spec,
                        "candidate": candidate,
                        "stage": f"terminal:{name}",
                        "stage_order": code,
                        "branches": total,
                        "reached": int(np.count_nonzero(gates == code)),
                        "fraction": float(np.mean(gates == code)),
                    }
                )
    return pd.DataFrame(rows)


def _reliability_table(replay: Mapping[str, np.ndarray]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    candidates = np.asarray(replay["candidates"]).astype(str)
    matrix_ids = np.asarray(replay["matrix_ids"], dtype=np.int64)
    for spec_index, spec in enumerate(SPEC_NAMES):
        for candidate in ("02", "03"):
            selected = candidates == candidate
            values = replay["labels"][selected, :, spec_index]
            mids = matrix_ids[selected]
            split = values.shape[1] // 2
            q_a = values[:, :split].mean(axis=1)
            q_b = values[:, split:].mean(axis=1)
            output: dict[str, Any] = {
                "spec": spec,
                "candidate": candidate,
                "ordinary": spearman(q_a, q_b),
                "centered": centered_spearman(q_a, q_b, mids),
            }
            for centered in (False, True):
                name = "centered" if centered else "ordinary"
                try:
                    estimate, interval = _reliability_bootstrap(
                        q_a,
                        q_b,
                        mids,
                        centered,
                        BOOTSTRAP_REPETITIONS,
                        np.random.default_rng(
                            derive_seed(
                                BOOTSTRAP_MASTER_SEED,
                                "strict_audit.reliability",
                                spec,
                                candidate,
                                name,
                            )
                        ),
                    )
                except (IndexError, ValueError):
                    estimate, interval = np.nan, (np.nan, np.nan)
                output[f"{name}_bootstrap_estimate"] = estimate
                output[f"{name}_ci95_lower"] = interval[0]
                output[f"{name}_ci95_upper"] = interval[1]
            rows.append(output)
    return pd.DataFrame(rows)


def _overlap_table(replay: Mapping[str, np.ndarray]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    candidates = np.asarray(replay["candidates"]).astype(str)
    for candidate in ("02", "03"):
        selected = candidates == candidate
        for left_index, left in enumerate(SPEC_NAMES):
            a = replay["labels"][selected, :, left_index].astype(bool)
            for right_index, right in enumerate(SPEC_NAMES):
                b = replay["labels"][selected, :, right_index].astype(bool)
                union = int(np.count_nonzero(a | b))
                intersection = int(np.count_nonzero(a & b))
                rows.append(
                    {
                        "candidate": candidate,
                        "left": left,
                        "right": right,
                        "left_events": int(a.sum()),
                        "right_events": int(b.sum()),
                        "intersection": intersection,
                        "union": union,
                        "jaccard": np.nan if union == 0 else intersection / union,
                        "raw_agreement": float(np.mean(a == b)),
                    }
                )
        labels = replay["labels"][selected].astype(np.int8)
        codes = labels[:, :, 0] * 4 + labels[:, :, 1] * 2 + labels[:, :, 2]
        for code in range(8):
            rows.append(
                {
                    "candidate": candidate,
                    "left": "stratum",
                    "right": f"cosine{(code>>2)&1}_global{(code>>1)&1}_relation{code&1}",
                    "left_events": int(np.count_nonzero(codes == code)),
                    "right_events": int(codes.size),
                    "intersection": np.nan,
                    "union": np.nan,
                    "jaccard": np.nan,
                    "raw_agreement": float(np.mean(codes == code)),
                }
            )
    return pd.DataFrame(rows)


def _cross_evaluation_table(replay: Mapping[str, np.ndarray]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    candidates = np.asarray(replay["candidates"]).astype(str)
    for candidate in ("02", "03"):
        selected = candidates == candidate
        for source_index, source in enumerate(SPEC_NAMES):
            source_positive = replay["labels"][selected, :, source_index].astype(bool)
            for target_index, target in enumerate(SPEC_NAMES):
                values = replay["cross_eval"][selected, :, source_index, target_index]
                evaluated = values[source_positive]
                if evaluated.size == 0:
                    temporal = boundary = pairwise = anchor = all_pass = np.nan
                else:
                    temporal_values = evaluated[:, 0] >= 0.5
                    boundary_values = evaluated[:, 1] > 0.0
                    pairwise_values = evaluated[:, 2] > 0.0
                    anchor_values = evaluated[:, 3] >= 0.0
                    temporal = float(temporal_values.mean())
                    boundary = float(boundary_values.mean())
                    pairwise = float(pairwise_values.mean())
                    anchor = float(anchor_values.mean())
                    all_pass = float(
                        np.mean(
                            temporal_values
                            & boundary_values
                            & pairwise_values
                            & anchor_values
                        )
                    )
                rows.append(
                    {
                        "candidate": candidate,
                        "source_event": source,
                        "target_evaluation": target,
                        "source_events": int(source_positive.sum()),
                        "fraction_starts_after_target_break": temporal,
                        "fraction_boundary_pass": boundary,
                        "fraction_pairwise_pass": pairwise,
                        "fraction_anchor_pass": anchor,
                        "fraction_all_target_conditions_pass": all_pass,
                    }
                )
    return pd.DataFrame(rows)


def _matched_tables(
    replay: Mapping[str, np.ndarray]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matches = match_event_controls(
        replay["labels"],
        replay["first_run"],
        replay["state_ids"],
        SPEC_NAMES,
        MATCHING_SEED,
    )
    rows: list[dict[str, Any]] = []
    for match in matches:
        state = int(match["state_index"])
        spec_index = int(match["spec_index"])
        event_branch = int(match["event_branch"])
        control_branch = int(match["control_branch"])
        row: dict[str, Any] = {
            **match,
            "candidate": str(replay["candidates"][state]),
            "matrix_id": int(replay["matrix_ids"][state]),
            "landmark": int(replay["landmarks"][state]),
        }
        event_values = replay["event_stats"][state, event_branch, spec_index]
        control_values = replay["precursor_stats"][state, control_branch, spec_index]
        for stat_index, stat in enumerate(WINDOW_STAT_NAMES):
            row[f"event_{stat}"] = float(event_values[stat_index])
            row[f"control_{stat}"] = float(control_values[stat_index])
            row[f"difference_{stat}"] = float(
                event_values[stat_index] - control_values[stat_index]
            )
        rows.append(row)
    pairs = pd.DataFrame(rows)

    candidate_values = np.asarray(replay["candidates"]).astype(str)
    summary_rows: list[dict[str, Any]] = []
    for spec_index, spec in enumerate(SPEC_NAMES):
        for candidate in ("02", "03"):
            selected = candidate_values == candidate
            positive = int(replay["labels"][selected, :, spec_index].sum())
            eligible_controls = int(
                np.count_nonzero(
                    (replay["labels"][selected, :, spec_index] == 0)
                    & (replay["first_run"][selected, :, spec_index] >= 0)
                )
            )
            matched = int(
                len(pairs.loc[(pairs["spec"] == spec) & (pairs["candidate"] == candidate)])
                if not pairs.empty
                else 0
            )
            summary_rows.append(
                {
                    "spec": spec,
                    "candidate": candidate,
                    "positive_events": positive,
                    "eligible_negative_precursors": eligible_controls,
                    "matched_pairs": matched,
                    "unmatched_positive_events": positive - matched,
                }
            )
    matching_summary = pd.DataFrame(summary_rows)

    effect_rows: list[dict[str, Any]] = []
    if not pairs.empty:
        for spec in SPEC_NAMES:
            for candidate in ("02", "03"):
                selected = pairs.loc[
                    (pairs["spec"] == spec) & (pairs["candidate"] == candidate)
                ]
                if selected.empty:
                    continue
                unique_matrices = np.sort(selected["matrix_id"].unique())
                for stat in WINDOW_STAT_NAMES:
                    column = f"difference_{stat}"
                    matrix_effect = np.asarray(
                        [
                            selected.loc[selected["matrix_id"] == matrix, column].mean()
                            for matrix in unique_matrices
                        ],
                        dtype=np.float64,
                    )
                    rng = np.random.default_rng(
                        derive_seed(
                            BOOTSTRAP_MASTER_SEED,
                            "strict_audit.matched",
                            spec,
                            candidate,
                            stat,
                        )
                    )
                    draws = rng.integers(
                        0,
                        len(matrix_effect),
                        size=(BOOTSTRAP_REPETITIONS, len(matrix_effect)),
                    )
                    samples = matrix_effect[draws].mean(axis=1)
                    lower, upper = np.quantile(samples, (0.025, 0.975))
                    effect_rows.append(
                        {
                            "spec": spec,
                            "candidate": candidate,
                            "statistic": stat,
                            "matched_pairs": len(selected),
                            "matrices": len(unique_matrices),
                            "event_mean": float(selected[f"event_{stat}"].mean()),
                            "control_mean": float(selected[f"control_{stat}"].mean()),
                            "paired_difference": float(matrix_effect.mean()),
                            "ci95_lower": float(lower),
                            "ci95_upper": float(upper),
                        }
                    )
    return pairs, matching_summary, pd.DataFrame(effect_rows)


def _event_characterization_tables(
    replay: Mapping[str, np.ndarray]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return one row per event and a long all-event summary.

    The event rows are intentionally independent of control availability.  In
    particular, the dominance fractions in the summary therefore describe all
    detected events rather than only the subset that could be matched within a
    state to a precursor-reaching negative branch.
    """

    labels = np.asarray(replay["labels"], dtype=np.int8)
    rows: list[dict[str, Any]] = []
    for state_index in range(labels.shape[0]):
        for branch in range(labels.shape[1]):
            for spec_index, spec in enumerate(SPEC_NAMES):
                if labels[state_index, branch, spec_index] != 1:
                    continue
                stats = replay["event_stats"][state_index, branch, spec_index]
                diagonal = replay["cross_eval"][
                    state_index, branch, spec_index, spec_index
                ]
                row: dict[str, Any] = {
                    "state_index": state_index,
                    "state_id": str(replay["state_ids"][state_index]),
                    "candidate": str(replay["candidates"][state_index]),
                    "matrix_id": int(replay["matrix_ids"][state_index]),
                    "landmark": int(replay["landmarks"][state_index]),
                    "branch": branch,
                    "spec": spec,
                    "onset": int(replay["onsets"][state_index, branch, spec_index]),
                    "first_break": int(
                        replay["first_break"][state_index, branch, spec_index]
                    ),
                    "first_run": int(
                        replay["first_run"][state_index, branch, spec_index]
                    ),
                    "minimum_inheritance_margin": float(diagonal[1]),
                    "minimum_pairwise_margin": float(diagonal[2]),
                    "anchor_distinctness_margin": float(diagonal[3]),
                }
                for stat_index, stat in enumerate(WINDOW_STAT_NAMES):
                    row[stat] = float(stats[stat_index])
                row["all_daughters_top1_ge_0_80"] = bool(
                    row["daughter_fraction_top1_ge_0_80"] == 1.0
                )
                row["any_daughter_top1_ge_0_80"] = bool(
                    row["daughter_fraction_top1_ge_0_80"] > 0.0
                )
                row["all_daughters_top2_ge_0_80"] = bool(
                    row["daughter_fraction_top2_ge_0_80"] == 1.0
                )
                row["any_daughter_top2_ge_0_80"] = bool(
                    row["daughter_fraction_top2_ge_0_80"] > 0.0
                )
                rows.append(row)
    events = pd.DataFrame(rows)

    statistics = list(WINDOW_STAT_NAMES) + [
        "minimum_inheritance_margin",
        "minimum_pairwise_margin",
        "anchor_distinctness_margin",
        "all_daughters_top1_ge_0_80",
        "any_daughter_top1_ge_0_80",
        "all_daughters_top2_ge_0_80",
        "any_daughter_top2_ge_0_80",
    ]
    summary_rows: list[dict[str, Any]] = []
    for spec in SPEC_NAMES:
        for candidate in ("02", "03"):
            selected = events.loc[
                (events["spec"] == spec) & (events["candidate"] == candidate)
            ]
            for statistic in statistics:
                values = selected[statistic].astype(float).to_numpy()
                if values.size:
                    mean = float(np.mean(values))
                    median = float(np.median(values))
                    q025, q975 = (
                        float(value)
                        for value in np.quantile(values, (0.025, 0.975))
                    )
                else:
                    mean = median = q025 = q975 = np.nan
                summary_rows.append(
                    {
                        "spec": spec,
                        "candidate": candidate,
                        "events": len(selected),
                        "event_matrices": int(selected["matrix_id"].nunique()),
                        "statistic": statistic,
                        "mean": mean,
                        "median": median,
                        "q025": q025,
                        "q975": q975,
                    }
                )
    return events, pd.DataFrame(summary_rows)


def analyze() -> None:
    verify_protocol()
    verify_model_seal()
    verify_checksums(CALIBRATION_ROOT)
    verify_checksums(REPLAY_ROOT)
    development = _load_npz(REPLAY_ROOT / "development.npz")
    confirmation = _load_npz(REPLAY_ROOT / "confirmation.npz")
    development_raw, development_source = _raw_features(
        DEVELOPMENT_SOURCE / "development_arrays.npz"
    )
    confirmation_raw, confirmation_source = _raw_features(
        CONFIRMATION_SOURCE / "confirmation_arrays.npz"
    )
    del development_raw
    if not np.array_equal(confirmation["state_ids"], confirmation_source["state_ids"]):
        raise AssertionError("confirmation replay/features order mismatch")
    predictions = _load_model_predictions(
        confirmation_raw, confirmation_source["candidates"]
    )
    development_power = _power_table(
        "development",
        development["labels"],
        development["candidates"],
        development["matrix_ids"],
    )
    confirmation_power = _power_table(
        "confirmation",
        confirmation["labels"],
        confirmation["candidates"],
        confirmation["matrix_ids"],
    )
    prediction, inventory = _prediction_rows(
        confirmation["labels"],
        confirmation["candidates"],
        confirmation["matrix_ids"],
        predictions,
        development_power,
        confirmation_power,
    )
    gates = pd.concat(
        [_gate_table("development", development), _gate_table("confirmation", confirmation)],
        ignore_index=True,
    )
    reliability = _reliability_table(confirmation)
    overlap = _overlap_table(confirmation)
    cross = _cross_evaluation_table(confirmation)
    pairs, matching, effects = _matched_tables(confirmation)
    events, event_summary = _event_characterization_tables(confirmation)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    pd.concat([development_power, confirmation_power], ignore_index=True).to_csv(
        OUTPUT_ROOT / "event_power.csv", index=False
    )
    prediction.to_csv(OUTPUT_ROOT / "prediction_comparisons.csv", index=False)
    inventory.to_csv(OUTPUT_ROOT / "model_inventory.csv", index=False)
    gates.to_csv(OUTPUT_ROOT / "gate_waterfall.csv", index=False)
    reliability.to_csv(OUTPUT_ROOT / "reliability.csv", index=False)
    overlap.to_csv(OUTPUT_ROOT / "event_overlap.csv", index=False)
    cross.to_csv(OUTPUT_ROOT / "cross_metric_window_evaluation.csv", index=False)
    pairs.to_csv(OUTPUT_ROOT / "matched_event_control_pairs.csv.gz", index=False)
    matching.to_csv(OUTPUT_ROOT / "matching_summary.csv", index=False)
    effects.to_csv(OUTPUT_ROOT / "matched_nondegeneracy_effects.csv", index=False)
    events.to_csv(OUTPUT_ROOT / "event_characteristics.csv.gz", index=False)
    event_summary.to_csv(
        OUTPUT_ROOT / "event_nondegeneracy_summary.csv", index=False
    )

    diagonal = prediction.loc[prediction["target_matched"]]
    audit = {
        "event_power_rows": len(development_power) + len(confirmation_power) == 12,
        "prediction_rows": len(prediction) == 3 * 3 * 2 * 2,
        "target_matched_rows": len(diagonal) == 12,
        "holm_values_defined_for_target_matched": bool(
            diagonal["randomization_p_holm"].notna().all()
        ),
        "gate_rows": len(gates) == 2 * 3 * 2 * 10,
        "overlap_nonempty": not overlap.empty,
        "cross_evaluation_nonempty": not cross.empty,
        "matched_pairs_nonempty": not pairs.empty,
        "matched_effects_nonempty": not effects.empty,
        "event_rows_equal_confirmation_labels": len(events)
        == int(confirmation["labels"].sum()),
        "event_summary_rows": len(event_summary)
        == len(SPEC_NAMES) * 2 * (len(WINDOW_STAT_NAMES) + 7),
        "event_characteristics_finite": bool(
            np.isfinite(
                events[
                    list(WINDOW_STAT_NAMES)
                    + [
                        "minimum_inheritance_margin",
                        "minimum_pairwise_margin",
                        "anchor_distinctness_margin",
                    ]
                ].to_numpy(dtype=np.float64)
            ).all()
        ),
        "all_event_stats_finite": bool(
            np.isfinite(
                confirmation["event_stats"][confirmation["labels"].astype(bool)]
            ).all()
        ),
    }
    audit["all_checks_passed"] = all(audit.values())
    _write_json(OUTPUT_ROOT / "analysis_readback_audit.json", audit)
    if not audit["all_checks_passed"]:
        raise AssertionError(f"analysis audit failed: {audit}")
    print(f"Strict-event analysis tables written under {OUTPUT_ROOT}", flush=True)


def report() -> None:
    verify_protocol()
    verify_model_seal()
    verify_checksums(CALIBRATION_ROOT)
    verify_checksums(REPLAY_ROOT)
    audit_path = OUTPUT_ROOT / "analysis_readback_audit.json"
    if not audit_path.is_file():
        raise FileNotFoundError("run analyze before report")
    analysis_audit = json.loads(audit_path.read_text())
    if analysis_audit.get("all_checks_passed") is not True:
        raise ValueError("analysis readback audit has not passed")
    from reporting import build_reports, make_figures

    figure_files = make_figures(OUTPUT_ROOT, CALIBRATION_ROOT)
    classification = build_reports(
        TASK_ROOT, OUTPUT_ROOT, CALIBRATION_ROOT, MODEL_ROOT
    )
    report_audit = {
        "format": "strict-event-report-audit-v1",
        "protocol_id": _protocol_id(),
        "figures_created": len(figure_files),
        "expected_figure_files": 10,
        "all_figures_nonempty": all(Path(path).stat().st_size > 0 for path in figure_files),
        "results_report_nonempty": (TASK_ROOT / "RESULTS_REPORT.md").stat().st_size > 0,
        "lay_summary_nonempty": (TASK_ROOT / "LAY_SUMMARY.md").stat().st_size > 0,
        "suggested_text_nonempty": (TASK_ROOT / "SUGGESTED_TEXT.md").stat().st_size > 0,
    }
    report_audit["all_checks_passed"] = (
        report_audit["figures_created"] == report_audit["expected_figure_files"]
        and all(
            bool(value)
            for key, value in report_audit.items()
            if key.endswith("_nonempty")
        )
    )
    if not report_audit["all_checks_passed"]:
        raise AssertionError(f"report audit failed: {report_audit}")
    _write_json(OUTPUT_ROOT / "result_summary.json", classification)
    _write_json(OUTPUT_ROOT / "report_readback_audit.json", report_audit)
    print(f"Reports and figures written under {TASK_ROOT}", flush=True)


def _required_output_files() -> tuple[str, ...]:
    return (
        "event_power.csv",
        "prediction_comparisons.csv",
        "model_inventory.csv",
        "gate_waterfall.csv",
        "reliability.csv",
        "event_overlap.csv",
        "cross_metric_window_evaluation.csv",
        "matched_event_control_pairs.csv.gz",
        "matching_summary.csv",
        "matched_nondegeneracy_effects.csv",
        "event_characteristics.csv.gz",
        "event_nondegeneracy_summary.csv",
        "analysis_readback_audit.json",
        "result_summary.json",
        "report_readback_audit.json",
        "figures/gate_waterfall_confirmation.png",
        "figures/gate_waterfall_confirmation.pdf",
        "figures/matched_nondegeneracy_effects.png",
        "figures/matched_nondegeneracy_effects.pdf",
        "figures/prediction_gains_and_transfer.png",
        "figures/prediction_gains_and_transfer.pdf",
        "figures/event_overlap_and_strata.png",
        "figures/event_overlap_and_strata.pdf",
        "figures/relation_specific_calibration.png",
        "figures/relation_specific_calibration.pdf",
    )


def verify() -> None:
    protocol = verify_protocol()
    verify_checksums(CALIBRATION_ROOT)
    seal = verify_model_seal()
    verify_checksums(REPLAY_ROOT)
    required = [OUTPUT_ROOT / name for name in _required_output_files()]
    required.extend(
        TASK_ROOT / name
        for name in ("RESULTS_REPORT.md", "LAY_SUMMARY.md", "SUGGESTED_TEXT.md")
    )
    missing = [str(path) for path in required if not path.is_file()]
    empty = [str(path) for path in required if path.is_file() and path.stat().st_size == 0]
    if missing or empty:
        raise FileNotFoundError(f"missing={missing}; empty={empty}")

    calibration = json.loads(
        (CALIBRATION_ROOT / "relation_specific_calibration.json").read_text()
    )
    replay_audits = {
        cohort: json.loads((REPLAY_ROOT / f"{cohort}_audit.json").read_text())
        for cohort in ("development", "confirmation")
    }
    analysis_audit = json.loads(
        (OUTPUT_ROOT / "analysis_readback_audit.json").read_text()
    )
    report_audit = json.loads(
        (OUTPUT_ROOT / "report_readback_audit.json").read_text()
    )
    development = _load_npz(REPLAY_ROOT / "development.npz")
    confirmation = _load_npz(REPLAY_ROOT / "confirmation.npz")
    prediction = pd.read_csv(OUTPUT_ROOT / "prediction_comparisons.csv")
    event_rows = pd.read_csv(OUTPUT_ROOT / "event_characteristics.csv.gz")
    diagonal = prediction.loc[prediction["target_matched"].astype(bool)]
    checks: dict[str, Any] = {
        "protocol_current": protocol["protocol_current"] is True,
        "calibration_protocol_exact": calibration["protocol_id"] == _protocol_id(),
        "calibration_development_only": calibration["confirmation_used"] is False,
        "calibration_not_prevalence_matched": calibration["event_prevalence_used"] is False,
        "calibration_relations_nonempty": all(
            int(calibration[name]["paired_comparisons"]) > 0
            for name in ("boundary", "coherence", "anchor")
        ),
        "development_replay_audit_exact": all(
            value
            for key, value in replay_audits["development"].items()
            if key.endswith("_exact")
        ),
        "confirmation_replay_audit_exact": all(
            value
            for key, value in replay_audits["confirmation"].items()
            if key.endswith("_exact")
        ),
        "development_protocol_exact": str(development["protocol_id"].item())
        == _protocol_id(),
        "confirmation_protocol_exact": str(confirmation["protocol_id"].item())
        == _protocol_id(),
        "spec_order_exact": bool(
            np.array_equal(development["spec_names"], np.asarray(SPEC_NAMES))
            and np.array_equal(confirmation["spec_names"], np.asarray(SPEC_NAMES))
        ),
        "model_sealed_before_confirmation": seal["confirmation_labels_or_outcomes_loaded"]
        is False,
        "model_portability_exact": float(
            seal["portable_prediction_max_absolute_error"]
        )
        <= 1e-12,
        "cosine_refit_positive_control_exact": float(
            seal["cosine_refit_positive_control_max_absolute_error"]
        )
        <= 1e-12,
        "analysis_audit_passed": analysis_audit["all_checks_passed"] is True,
        "report_audit_passed": report_audit["all_checks_passed"] is True,
        "prediction_rows_exact": len(prediction) == 36,
        "target_matched_rows_exact": len(diagonal) == 12,
        "target_matched_holm_complete": bool(
            diagonal["randomization_p_holm"].notna().all()
        ),
        "event_rows_exact": len(event_rows) == int(confirmation["labels"].sum()),
        "top_level_reports_nonempty": all(
            (TASK_ROOT / name).stat().st_size > 0
            for name in ("RESULTS_REPORT.md", "LAY_SUMMARY.md", "SUGGESTED_TEXT.md")
        ),
    }
    checks["all_checks_passed"] = all(checks.values())
    if not checks["all_checks_passed"]:
        raise AssertionError(f"final verification failed: {checks}")

    summary = json.loads((OUTPUT_ROOT / "result_summary.json").read_text())
    manifest = {
        "format": RESULT_FORMAT,
        "status": "complete_and_verified",
        "protocol_id": _protocol_id(),
        "model_seal_id": seal["seal_id"],
        "development_replay_sha256": sha256_file(REPLAY_ROOT / "development.npz"),
        "confirmation_replay_sha256": sha256_file(REPLAY_ROOT / "confirmation.npz"),
        "relation_calibration_sha256": sha256_file(
            CALIBRATION_ROOT / "relation_specific_calibration.json"
        ),
        "results_report_sha256": sha256_file(TASK_ROOT / "RESULTS_REPORT.md"),
        "lay_summary_sha256": sha256_file(TASK_ROOT / "LAY_SUMMARY.md"),
        "suggested_text_sha256": sha256_file(TASK_ROOT / "SUGGESTED_TEXT.md"),
        "scientific_summary": summary,
    }
    _write_json(OUTPUT_ROOT / "verification_audit.json", checks)
    _write_json(OUTPUT_ROOT / "result_manifest.json", manifest)
    _replace_checksums(OUTPUT_ROOT)
    verify_checksums(OUTPUT_ROOT)
    print(f"FINAL VERIFICATION PASSED: {OUTPUT_ROOT}", flush=True)


def status() -> None:
    def state_count(path: Path) -> int:
        if not path.is_file():
            return 0
        with np.load(path, allow_pickle=False) as archive:
            return int(len(archive["state_ids"]))

    expected_development = state_count(DEVELOPMENT_SOURCE / "development_arrays.npz")
    expected_confirmation = state_count(
        CONFIRMATION_SOURCE / "confirmation_arrays.npz"
    )
    expected = {
        "calibration": expected_development,
        "development_replay": expected_development,
        "confirmation_replay": expected_confirmation,
    }
    protocol_present = (PROTOCOL_ROOT / "analysis_protocol.json").is_file()
    checkpoint_counts: dict[str, dict[str, int]] = {}
    for dataset, count in expected.items():
        directory = WORK_ROOT / dataset
        present = len(list(directory.glob("state_*.npz"))) if directory.is_dir() else 0
        valid = (
            sum(
                _checkpoint_complete(_checkpoint_path(dataset, index), dataset, index)
                for index in range(count)
            )
            if protocol_present
            else 0
        )
        checkpoint_counts[dataset] = {
            "valid": int(valid),
            "expected": count,
            "present": present,
        }
    value = {
        "protocol_frozen": protocol_present,
        "checkpoints": checkpoint_counts,
        "relation_calibration_complete": (
            CALIBRATION_ROOT / "relation_specific_calibration.json"
        ).is_file(),
        "development_replay_assembled": (REPLAY_ROOT / "development.npz").is_file(),
        "models_sealed": (MODEL_ROOT / "model_seal.json").is_file(),
        "confirmation_replay_assembled": (REPLAY_ROOT / "confirmation.npz").is_file(),
        "analysis_complete": (OUTPUT_ROOT / "analysis_readback_audit.json").is_file(),
        "report_complete": (OUTPUT_ROOT / "report_readback_audit.json").is_file(),
        "verification_complete": (OUTPUT_ROOT / "verification_audit.json").is_file(),
    }
    print(json.dumps(value, indent=2), flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Strict-event geometry and target-specific prediction audit"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in (
        "calibrate",
        "replay-development",
        "replay-confirmation",
        "all",
    ):
        item = subparsers.add_parser(command)
        item.add_argument(
            "--workers",
            type=int,
            default=min(14, max(1, (os.cpu_count() or 2) - 1)),
        )
    for command in ("prepare", "fit-seal", "analyze", "report", "verify", "status"):
        subparsers.add_parser(command)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    if getattr(arguments, "workers", 1) < 1:
        raise ValueError("workers must be positive")
    if arguments.command == "prepare":
        prepare()
    elif arguments.command == "calibrate":
        calibrate(arguments.workers)
    elif arguments.command == "replay-development":
        replay("development", arguments.workers)
    elif arguments.command == "fit-seal":
        fit_and_seal()
    elif arguments.command == "replay-confirmation":
        replay("confirmation", arguments.workers)
    elif arguments.command == "analyze":
        analyze()
    elif arguments.command == "report":
        report()
    elif arguments.command == "verify":
        verify()
    elif arguments.command == "status":
        status()
    elif arguments.command == "all":
        prepare()
        calibrate(arguments.workers)
        replay("development", arguments.workers)
        fit_and_seal()
        replay("confirmation", arguments.workers)
        analyze()
        report()
        verify()
    else:
        raise AssertionError(arguments.command)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
