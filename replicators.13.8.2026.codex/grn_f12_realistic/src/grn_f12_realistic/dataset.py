from __future__ import annotations

from pathlib import Path

import numpy as np

from .storage import load_npz


REQUIRED_FIELDS = (
    "network_index", "W", "history_features", "structural_features", "node_features",
    "event_count", "break_count", "event_half0", "event_half1",
    "break_half0", "break_half1", "cue_index", "age", "futures",
    "run5_count", "f24_count", "event_count_q025", "event_count_q10",
    "coherence_mean", "old_anchor_separation",
)


def network_paths(run_dir: str | Path, cohort: str, tier: str) -> list[Path]:
    return sorted((Path(run_dir) / "data" / cohort / tier).glob("network_*.npz"))


def load_dataset(paths: list[Path]) -> dict[str, np.ndarray]:
    if not paths:
        raise RuntimeError("dataset has no network shards")
    rows: list[dict[str, np.ndarray]] = []
    for path in paths:
        shard = load_npz(path)
        missing = [name for name in REQUIRED_FIELDS if name not in shard]
        if missing:
            raise RuntimeError(f"{path} is missing {missing}")
        rows.append(shard)
    rows.sort(key=lambda row: int(row["network_index"]))
    indices = np.asarray([int(row["network_index"]) for row in rows], dtype=np.int32)
    if len(np.unique(indices)) != len(indices):
        raise RuntimeError("duplicate network indices")
    result: dict[str, np.ndarray] = {"network_index": indices}
    per_network = (
        "W", "history_features", "structural_features", "node_features", "event_count",
        "break_count", "event_half0", "event_half1", "break_half0", "break_half1",
        "cue_index", "age",
        "run5_count", "f24_count", "event_count_q025", "event_count_q10",
        "coherence_mean", "old_anchor_separation",
    )
    for name in per_network:
        result[name] = np.stack([row[name] for row in rows])
    futures = np.asarray([int(row["futures"]) for row in rows], dtype=np.int32)
    if len(np.unique(futures)) != 1:
        raise RuntimeError("future panel size differs across networks")
    result["futures"] = futures
    return result


def flatten_states(dataset: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    networks, states = dataset["event_count"].shape
    result = {
        "network_index": np.repeat(dataset["network_index"], states),
        "network_row": np.repeat(np.arange(networks, dtype=np.int32), states),
        "state_row": np.tile(np.arange(states, dtype=np.int32), networks),
        "W": np.repeat(dataset["W"], states, axis=0),
    }
    for name in (
        "history_features", "structural_features", "node_features", "event_count", "break_count",
        "event_half0", "event_half1", "break_half0", "break_half1", "cue_index", "age",
        "run5_count", "f24_count", "event_count_q025", "event_count_q10",
        "coherence_mean", "old_anchor_separation",
    ):
        value = dataset[name]
        result[name] = value.reshape((networks * states,) + value.shape[2:])
    result["total"] = np.repeat(dataset["futures"], states)
    result["half_total"] = result["total"] // 2
    return result
