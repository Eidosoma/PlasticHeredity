"""Stable on-disk formats for simulation and analysis checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .analysis import AnalyzedRun
from .gard import RunTrace


def save_trace(path: Path, trace: RunTrace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        counts=trace.counts,
        generations=trace.generations,
        phases=trace.phases,
        joins=trace.joins,
        leaves=trace.leaves,
        intervention_species=trace.intervention_species,
        intervention_delta=trace.intervention_delta,
        intervention_score=trace.intervention_score,
        beta=trace.beta,
        seed=np.asarray(trace.seed, dtype=np.int64),
    )


def load_trace(path: Path) -> RunTrace:
    with np.load(path, allow_pickle=False) as data:
        return RunTrace(
            counts=data["counts"],
            generations=data["generations"],
            phases=data["phases"],
            joins=data["joins"],
            leaves=data["leaves"],
            intervention_species=data["intervention_species"],
            intervention_delta=data["intervention_delta"],
            intervention_score=data["intervention_score"],
            beta=data["beta"],
            seed=int(data["seed"]),
        )


def save_analysis(path: Path, run: AnalyzedRun) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        phi=run.causal.values,
        phi_time_indices=run.causal.time_indices,
        partition=run.causal.partition,
        fiedler=run.causal.fiedler,
        grouped=run.causal.grouped,
        replicator_labels=run.replicator.labels,
        replicator_similarity=run.replicator.similarity,
        replicator_reference=run.replicator.reference,
        spike_indices=run.spike_indices,
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot JSON encode {type(value)!r}")


def _sanitize_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_json(item) for item in value]
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, np.ndarray):
        return _sanitize_json(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(
            _sanitize_json(payload),
            stream,
            indent=2,
            sort_keys=True,
            default=_json_default,
            allow_nan=False,
        )
        stream.write("\n")
