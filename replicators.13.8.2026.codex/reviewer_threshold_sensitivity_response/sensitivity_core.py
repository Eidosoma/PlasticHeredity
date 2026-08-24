from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from plastic_heredity.metrics import (
    centered_spearman,
    log_loss_from_q,
    q_brier,
    spearman,
)
from plastic_heredity.seeds import derive_seed
from plastic_heredity.simulator import FissionRecord, cosine_similarity


F12_THRESHOLDS = (0.85, 0.88, 0.90, 0.92, 0.95)
F12_HORIZONS = (8, 10, 12, 16)
F12_RUN_LENGTHS = (2, 3, 4)
F12_BASELINE = (0.90, 12, 3)

F32_THRESHOLDS = (0.88, 0.90, 0.92)
F32_RUN_LENGTHS = (7, 8, 9)
F32_ANCHOR_THRESHOLDS = (0.80, 0.85, 0.90)
F32_BASELINE = (0.90, 8, 0.85)

BOOTSTRAP_REPETITIONS = 512
SENSITIVITY_MASTER_SEED = (
    "a252d844064bec1441e2f14feee1c1d083f173eca5e060e262691554eade5568"
)
REFERENCE_MASTER_SEED = (
    "e354bb648e15692f59bd99e947cddeeb3cbc5643a38045157762313166f86d4b"
)


@dataclass(frozen=True)
class F12Definition:
    inheritance_threshold: float
    horizon: int
    run_length: int

    @property
    def key(self) -> str:
        return (
            f"h{self.inheritance_threshold:.2f}_f{self.horizon:02d}"
            f"_r{self.run_length}"
        )

    @property
    def is_baseline(self) -> bool:
        return (
            self.inheritance_threshold,
            self.horizon,
            self.run_length,
        ) == F12_BASELINE


@dataclass(frozen=True)
class F32Definition:
    strict_threshold: float
    run_length: int
    old_anchor_threshold: float

    @property
    def key(self) -> str:
        return (
            f"h{self.strict_threshold:.2f}_r{self.run_length}"
            f"_a{self.old_anchor_threshold:.2f}"
        )

    @property
    def is_baseline(self) -> bool:
        return (
            self.strict_threshold,
            self.run_length,
            self.old_anchor_threshold,
        ) == F32_BASELINE


F12_DEFINITIONS = tuple(
    F12Definition(threshold, horizon, run_length)
    for threshold in F12_THRESHOLDS
    for horizon in F12_HORIZONS
    for run_length in F12_RUN_LENGTHS
)
F32_DEFINITIONS = tuple(
    F32Definition(threshold, run_length, anchor)
    for threshold in F32_THRESHOLDS
    for run_length in F32_RUN_LENGTHS
    for anchor in F32_ANCHOR_THRESHOLDS
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def f12_definition_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "definition_index": index,
                "definition_key": definition.key,
                "inheritance_threshold_strict": definition.inheritance_threshold,
                "horizon_fissions": definition.horizon,
                "renewal_run_length": definition.run_length,
                "registered_baseline": definition.is_baseline,
            }
            for index, definition in enumerate(F12_DEFINITIONS)
        ]
    )


def f32_definition_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "definition_index": index,
                "definition_key": definition.key,
                "adjacent_and_pairwise_threshold_strict": definition.strict_threshold,
                "strict_run_length": definition.run_length,
                "old_anchor_threshold_inclusive": definition.old_anchor_threshold,
                "horizon_fissions": 32,
                "registered_baseline": definition.is_baseline,
            }
            for index, definition in enumerate(F32_DEFINITIONS)
        ]
    )


def _first_run(values: NDArray[np.bool_], length: int, start: int) -> int | None:
    if values.size < length:
        return None
    for index in range(start, values.size - length + 1):
        if bool(values[index : index + length].all()):
            return index
    return None


def score_f12_definition(
    boundary_h: NDArray[np.float64], definition: F12Definition
) -> bool:
    values = np.asarray(boundary_h[: definition.horizon], dtype=np.float64)
    valid = np.isfinite(values)
    inherited = valid & (values > definition.inheritance_threshold)
    breaks = np.flatnonzero(valid & ~inherited)
    if breaks.size == 0:
        return False
    first_break = int(breaks[0])
    return _first_run(inherited, definition.run_length, first_break + 1) is not None


def score_f12_grid(boundary_h: NDArray[np.float64]) -> NDArray[np.int8]:
    return np.asarray(
        [score_f12_definition(boundary_h, item) for item in F12_DEFINITIONS],
        dtype=np.int8,
    )


def _pairwise_minimum(vectors: Sequence[NDArray]) -> float:
    return min(
        cosine_similarity(vectors[left], vectors[right])
        for left in range(len(vectors))
        for right in range(left + 1, len(vectors))
    )


def dominant_h_component_centroid(
    post_fission_states: NDArray[np.integer], threshold: float = 0.90
) -> tuple[NDArray[np.float64], tuple[int, ...]]:
    """Reproduce the frozen L36/L23 dominant threshold-component centroid."""

    counts = np.asarray(post_fission_states, dtype=np.float64)
    if counts.ndim != 2 or counts.shape[0] == 0 or np.any(counts < 0):
        raise ValueError("post-fission states must be a nonempty nonnegative matrix")
    masses = counts.sum(axis=1)
    if np.any(masses <= 0):
        raise ValueError("empty post-fission composition is not centroid-evaluable")
    compositions = counts / masses[:, None]
    norms = np.linalg.norm(compositions, axis=1)
    normalized = compositions / norms[:, None]
    similarity = np.clip(normalized @ normalized.T, -1.0, 1.0)
    adjacency = similarity >= float(threshold)
    remaining = set(range(len(compositions)))
    components: list[tuple[int, ...]] = []
    while remaining:
        root = min(remaining)
        stack = [root]
        found: set[int] = set()
        while stack:
            item = stack.pop()
            if item in found:
                continue
            found.add(item)
            stack.extend(
                int(value)
                for value in np.flatnonzero(adjacency[item])
                if int(value) not in found
            )
        remaining.difference_update(found)
        components.append(tuple(sorted(found)))
    dominant = min(components, key=lambda value: (-len(value), min(value)))
    centroid = compositions[list(dominant)].mean(axis=0)
    centroid /= centroid.sum()
    return np.ascontiguousarray(centroid, dtype=np.float64), dominant


def score_f32_definition(
    records: Sequence[FissionRecord], definition: F32Definition
) -> tuple[bool, int]:
    inherited = np.asarray(
        [record.h > definition.strict_threshold for record in records], dtype=bool
    )
    breaks = np.flatnonzero(~inherited)
    if breaks.size == 0:
        return False, -1
    first_break = int(breaks[0])
    anchor = records[first_break].parent
    last_start = len(records) - definition.run_length
    for start in range(first_break + 1, last_start + 1):
        if not bool(inherited[start : start + definition.run_length].all()):
            continue
        daughters = [
            record.daughter
            for record in records[start : start + definition.run_length]
        ]
        if _pairwise_minimum(daughters) <= definition.strict_threshold:
            continue
        maximum_anchor = max(cosine_similarity(anchor, item) for item in daughters)
        if maximum_anchor <= definition.old_anchor_threshold:
            return True, start
    return False, -1


def score_f32_grid(
    records: Sequence[FissionRecord],
) -> tuple[NDArray[np.int8], NDArray[np.int16]]:
    labels = np.zeros(len(F32_DEFINITIONS), dtype=np.int8)
    onsets = np.full(len(F32_DEFINITIONS), -1, dtype=np.int16)
    if not records:
        return labels, onsets

    daughters = np.vstack(
        [np.asarray(record.daughter, dtype=np.float64) for record in records]
    )
    norms = np.linalg.norm(daughters, axis=1)
    denominators = np.outer(norms, norms)
    pairwise = np.divide(
        daughters @ daughters.T,
        denominators,
        out=np.zeros_like(denominators),
        where=denominators != 0.0,
    )
    pairwise = np.clip(pairwise, 0.0, 1.0)
    boundary_h = np.asarray([record.h for record in records], dtype=np.float64)
    definition_index = {
        (
            definition.strict_threshold,
            definition.run_length,
            definition.old_anchor_threshold,
        ): index
        for index, definition in enumerate(F32_DEFINITIONS)
    }

    for threshold in F32_THRESHOLDS:
        inherited = boundary_h > threshold
        breaks = np.flatnonzero(~inherited)
        if breaks.size == 0:
            continue
        first_break = int(breaks[0])
        anchor = np.asarray(records[first_break].parent, dtype=np.float64)
        anchor_norm = float(np.linalg.norm(anchor))
        anchor_denominators = norms * anchor_norm
        anchor_similarity = np.divide(
            daughters @ anchor,
            anchor_denominators,
            out=np.zeros_like(norms),
            where=anchor_denominators != 0.0,
        )
        anchor_similarity = np.clip(anchor_similarity, 0.0, 1.0)
        for run_length in F32_RUN_LENGTHS:
            last_start = len(records) - run_length
            qualifying: list[tuple[int, float]] = []
            for start in range(first_break + 1, last_start + 1):
                stop = start + run_length
                if not bool(inherited[start:stop].all()):
                    continue
                if float(pairwise[start:stop, start:stop].min()) <= threshold:
                    continue
                qualifying.append(
                    (start, float(anchor_similarity[start:stop].max()))
                )
            for anchor_threshold in F32_ANCHOR_THRESHOLDS:
                index = definition_index[(threshold, run_length, anchor_threshold)]
                for start, maximum_anchor in qualifying:
                    if maximum_anchor <= anchor_threshold:
                        labels[index] = 1
                        onsets[index] = start
                        break
    return labels, onsets


def _matrix_blocks(matrix_ids: NDArray[np.int64]) -> tuple[NDArray, list[NDArray]]:
    unique = np.unique(matrix_ids)
    return unique, [np.flatnonzero(matrix_ids == item) for item in unique]


def _bootstrap_statistic(
    arrays: dict[str, NDArray],
    matrix_ids: NDArray[np.int64],
    statistic: Any,
    seed_parts: tuple[Any, ...],
    repetitions: int = BOOTSTRAP_REPETITIONS,
) -> tuple[float, float]:
    unique, blocks = _matrix_blocks(matrix_ids)
    rng = np.random.default_rng(
        derive_seed(SENSITIVITY_MASTER_SEED, "bootstrap", *seed_parts)
    )
    samples = np.full(repetitions, np.nan, dtype=np.float64)
    for repetition in range(repetitions):
        selected = rng.integers(0, len(unique), size=len(unique))
        indices = np.concatenate([blocks[index] for index in selected])
        groups = np.repeat(
            np.arange(len(unique), dtype=np.int64),
            [blocks[index].size for index in selected],
        )
        sampled = {name: np.asarray(value)[indices] for name, value in arrays.items()}
        samples[repetition] = statistic(sampled, groups)
    finite = samples[np.isfinite(samples)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return tuple(float(value) for value in np.quantile(finite, (0.025, 0.975)))


def _prevalence_interval(
    labels: NDArray[np.int8],
    matrix_ids: NDArray[np.int64],
    seed_parts: tuple[Any, ...],
) -> tuple[float, float]:
    unique = np.unique(matrix_ids)
    matrix_rates = np.asarray(
        [labels[matrix_ids == item].mean() for item in unique], dtype=np.float64
    )
    rng = np.random.default_rng(
        derive_seed(SENSITIVITY_MASTER_SEED, "prevalence", *seed_parts)
    )
    draws = rng.integers(
        0, len(unique), size=(BOOTSTRAP_REPETITIONS, len(unique))
    )
    samples = matrix_rates[draws].mean(axis=1)
    return tuple(float(value) for value in np.quantile(samples, (0.025, 0.975)))


def summarize_prediction_grid(
    labels: NDArray[np.int8],
    states: pd.DataFrame,
    definitions: pd.DataFrame,
    baseline_prediction_column: str,
    added_prediction_column: str,
    analysis_name: str,
) -> pd.DataFrame:
    if labels.ndim != 3:
        raise ValueError("labels must have state x branch x definition dimensions")
    if labels.shape[0] != len(states) or labels.shape[2] != len(definitions):
        raise ValueError("label dimensions do not match states/definitions")
    split = labels.shape[1] // 2
    rows: list[dict[str, Any]] = []
    for candidate in ("02", "03"):
        selected = states["candidate"].astype(str).str.zfill(2) == candidate
        selected_indices = np.flatnonzero(selected.to_numpy())
        candidate_labels = labels[selected_indices]
        matrix_ids = states.loc[selected, "matrix_id"].to_numpy(dtype=np.int64)
        baseline_prediction = states.loc[
            selected, baseline_prediction_column
        ].to_numpy(dtype=np.float64)
        added_prediction = states.loc[
            selected, added_prediction_column
        ].to_numpy(dtype=np.float64)
        for definition_index, definition in definitions.iterrows():
            values = candidate_labels[:, :, definition_index]
            q_all = values.mean(axis=1)
            q_a = values[:, :split].mean(axis=1)
            q_b = values[:, split:].mean(axis=1)
            prevalence_ci = _prevalence_interval(
                values.ravel(),
                np.repeat(matrix_ids, values.shape[1]),
                (analysis_name, candidate, int(definition_index)),
            )
            reliability = spearman(q_a, q_b)
            centered_reliability = centered_spearman(q_a, q_b, matrix_ids)
            reliability_ci = _bootstrap_statistic(
                {"a": q_a, "b": q_b},
                matrix_ids,
                lambda sampled, groups: spearman(sampled["a"], sampled["b"]),
                (analysis_name, candidate, int(definition_index), "reliability"),
            )
            centered_reliability_ci = _bootstrap_statistic(
                {"a": q_a, "b": q_b},
                matrix_ids,
                lambda sampled, groups: centered_spearman(
                    sampled["a"], sampled["b"], groups
                ),
                (
                    analysis_name,
                    candidate,
                    int(definition_index),
                    "centered_reliability",
                ),
            )
            row: dict[str, Any] = {
                **definition.to_dict(),
                "candidate": candidate,
                "states": len(selected_indices),
                "branches": int(values.size),
                "events": int(values.sum()),
                "event_positive_matrices": int(
                    sum(values[matrix_ids == item].any() for item in np.unique(matrix_ids))
                ),
                "prevalence": float(values.mean()),
                "prevalence_ci95_lower": prevalence_ci[0],
                "prevalence_ci95_upper": prevalence_ci[1],
                "transition_region_states": int(((q_all > 0.1) & (q_all < 0.9)).sum()),
                "branch_half_reliability": reliability,
                "branch_half_reliability_ci95_lower": reliability_ci[0],
                "branch_half_reliability_ci95_upper": reliability_ci[1],
                "centered_branch_half_reliability": centered_reliability,
                "centered_branch_half_reliability_ci95_lower": centered_reliability_ci[0],
                "centered_branch_half_reliability_ci95_upper": centered_reliability_ci[1],
            }
            for half, q in (("A", q_a), ("B", q_b)):
                base_loss = log_loss_from_q(q, baseline_prediction)
                added_loss = log_loss_from_q(q, added_prediction)
                log_gain = base_loss - added_loss
                brier_gain = q_brier(q, baseline_prediction) - q_brier(
                    q, added_prediction
                )
                log_ci = _bootstrap_statistic(
                    {
                        "q": q,
                        "baseline": baseline_prediction,
                        "added": added_prediction,
                    },
                    matrix_ids,
                    lambda sampled, groups: log_loss_from_q(
                        sampled["q"], sampled["baseline"]
                    )
                    - log_loss_from_q(sampled["q"], sampled["added"]),
                    (
                        analysis_name,
                        candidate,
                        int(definition_index),
                        half,
                        "log_gain",
                    ),
                )
                centered_base = centered_spearman(
                    baseline_prediction, q, matrix_ids
                )
                centered_added = centered_spearman(added_prediction, q, matrix_ids)
                row.update(
                    {
                        f"log_loss_baseline_{half}": base_loss,
                        f"log_loss_added_{half}": added_loss,
                        f"log_loss_gain_{half}": log_gain,
                        f"log_loss_gain_{half}_ci95_lower": log_ci[0],
                        f"log_loss_gain_{half}_ci95_upper": log_ci[1],
                        f"q_brier_gain_{half}": brier_gain,
                        f"centered_spearman_baseline_{half}": centered_base,
                        f"centered_spearman_added_{half}": centered_added,
                        f"centered_spearman_gain_{half}": centered_added
                        - centered_base,
                    }
                )
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_cr1_grid(
    labels: NDArray[np.int8],
    states: pd.DataFrame,
    definitions: pd.DataFrame,
    arms: Sequence[str],
) -> pd.DataFrame:
    if labels.ndim != 4:
        raise ValueError("CR1 labels must be state x arm x branch x definition")
    arm_index = {name: index for index, name in enumerate(arms)}
    contrasts = (
        ("MODEL_UP", "MODEL_DOWN"),
        ("MODEL_UP", "NOOP"),
        ("NOOP", "MODEL_DOWN"),
        ("RANDOM", "NOOP"),
    )
    split = labels.shape[2] // 2
    rows: list[dict[str, Any]] = []
    for candidate in ("02", "03"):
        selected = states["candidate"].astype(str).str.zfill(2) == candidate
        indices = np.flatnonzero(selected.to_numpy())
        candidate_labels = labels[indices]
        matrix_ids = states.loc[selected, "matrix_id"].to_numpy(dtype=np.int64)
        unique = np.unique(matrix_ids)
        for definition_index, definition in definitions.iterrows():
            for half, branch_slice in (
                ("A", slice(0, split)),
                ("B", slice(split, labels.shape[2])),
            ):
                rates = candidate_labels[:, :, branch_slice, definition_index].mean(
                    axis=2
                )
                for left, right in contrasts:
                    state_effect = rates[:, arm_index[left]] - rates[:, arm_index[right]]
                    matrix_effect = np.asarray(
                        [state_effect[matrix_ids == item].mean() for item in unique],
                        dtype=np.float64,
                    )
                    rng = np.random.default_rng(
                        derive_seed(
                            SENSITIVITY_MASTER_SEED,
                            "cr1.bootstrap",
                            candidate,
                            int(definition_index),
                            half,
                            left,
                            right,
                        )
                    )
                    draws = rng.integers(
                        0,
                        len(unique),
                        size=(BOOTSTRAP_REPETITIONS, len(unique)),
                    )
                    samples = matrix_effect[draws].mean(axis=1)
                    lower, upper = np.quantile(samples, (0.025, 0.975))
                    rows.append(
                        {
                            **definition.to_dict(),
                            "candidate": candidate,
                            "half": half,
                            "left_arm": left,
                            "right_arm": right,
                            "contrast": f"{left}_minus_{right}",
                            "estimate": float(state_effect.mean()),
                            "ci95_lower": float(lower),
                            "ci95_upper": float(upper),
                            "matrices": int(len(unique)),
                            "states": int(len(indices)),
                            "expected_positive_direction": (
                                left,
                                right,
                            )
                            in {
                                ("MODEL_UP", "MODEL_DOWN"),
                                ("MODEL_UP", "NOOP"),
                                ("NOOP", "MODEL_DOWN"),
                            },
                        }
                    )
    return pd.DataFrame(rows)


def reference_summary(
    f12_boundary_h: NDArray[np.float64],
    f12_matrix_ids: NDArray[np.int64],
    f12_candidates: NDArray[np.str_],
    between_lineage_h: NDArray[np.float64],
    reference_matrix_ids: NDArray[np.int64],
    reference_candidates: NDArray[np.str_],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    specifications = (
        (
            "parent_to_selected_daughter",
            f12_boundary_h,
            f12_matrix_ids,
            f12_candidates,
            F12_THRESHOLDS,
        ),
        (
            "between_l36_method_independent_lineages",
            between_lineage_h,
            reference_matrix_ids,
            reference_candidates,
            F32_ANCHOR_THRESHOLDS,
        ),
    )
    for distribution, values, matrix_ids, candidates, thresholds in specifications:
        for candidate in ("02", "03"):
            selected = np.asarray(candidates).astype(str) == candidate
            finite = np.asarray(values[selected], dtype=np.float64).ravel()
            finite = finite[np.isfinite(finite)]
            row: dict[str, Any] = {
                "distribution": distribution,
                "candidate": candidate,
                "observations": int(finite.size),
                "matrices": int(np.unique(np.asarray(matrix_ids)[selected]).size),
                "mean": float(finite.mean()),
                "sd": float(finite.std()),
                "q01": float(np.quantile(finite, 0.01)),
                "q05": float(np.quantile(finite, 0.05)),
                "q25": float(np.quantile(finite, 0.25)),
                "median": float(np.quantile(finite, 0.50)),
                "q75": float(np.quantile(finite, 0.75)),
                "q95": float(np.quantile(finite, 0.95)),
                "q99": float(np.quantile(finite, 0.99)),
                "formal_modality_claim": False,
            }
            for threshold in thresholds:
                suffix = f"{threshold:.2f}".replace(".", "_")
                row[f"fraction_le_{suffix}"] = float((finite <= threshold).mean())
                row[f"fraction_gt_{suffix}"] = float((finite > threshold).mean())
                if threshold == 0.85:
                    row["percentile_rank_of_0_85"] = float(
                        100.0 * (finite <= 0.85).mean()
                    )
            rows.append(row)
    return pd.DataFrame(rows)


def atomic_npz(path: Path, **arrays: NDArray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def checkpoint_indices(directory: Path) -> Iterable[int]:
    for path in sorted(directory.glob("state_*.npz")):
        yield int(path.stem.split("_")[-1])
