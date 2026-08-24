"""Pure geometry, scoring, diversity, and matching utilities.

The module deliberately contains no filesystem or simulator orchestration so
that every scientific rule can be unit-tested before the post-hoc protocol is
sealed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


RUN_LENGTH = 8
HORIZON = 32
DOMINANCE_SHARE = 0.80

SPEC_COSINE = "cosine_registered"
SPEC_BRAY_GLOBAL = "bray_global"
SPEC_BRAY_RELATION = "bray_relation_specific"
SPEC_NAMES = (SPEC_COSINE, SPEC_BRAY_GLOBAL, SPEC_BRAY_RELATION)

GATE_NO_BREAK = 0
GATE_BREAK_NO_RUN = 1
GATE_RUN_NO_COHERENCE = 2
GATE_COHERENCE_ANCHOR_FAIL = 3
GATE_EVENT = 4
GATE_NAMES = (
    "no_break",
    "break_no_inherited_run8",
    "inherited_run8_no_coherent_window",
    "coherent_window_anchor_fail",
    "strict_event",
)

WINDOW_STAT_NAMES = (
    "effective_species_mean",
    "effective_species_min",
    "occupied_types_mean",
    "occupied_types_min",
    "top1_share_mean",
    "top1_share_max",
    "top2_share_mean",
    "top2_share_max",
    "daughter_fraction_top1_ge_0_80",
    "daughter_fraction_top2_ge_0_80",
    "adjacent_total_variation_mean",
    "adjacent_total_variation_max",
    "occupied_set_turnover_mean",
    "occupied_set_turnover_max",
    "growth_steps_sum",
    "growth_steps_mean",
    "growth_steps_max",
)

CROSS_EVAL_NAMES = (
    "starts_after_target_break",
    "minimum_boundary_margin",
    "minimum_pairwise_margin",
    "anchor_margin",
)


@dataclass(frozen=True)
class EndpointSpec:
    name: str
    metric: str
    inheritance_cutoff: float
    coherence_cutoff: float
    anchor_cutoff: float


@dataclass(frozen=True)
class MetricGeometry:
    boundary: NDArray[np.float64]
    profiles: NDArray[np.float64]
    bands: tuple[NDArray[np.float64], ...]


@dataclass(frozen=True)
class EndpointOutcome:
    event: bool
    onset: int
    first_break: int
    first_run: int
    deepest_gate: int
    eligible_windows: int
    coherent_windows: int
    best_pairwise_margin: float
    best_anchor_margin: float
    precursor_stats: NDArray[np.float64]
    event_stats: NDArray[np.float64]


def normalized_profile(values: NDArray) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or np.any(array < 0):
        raise ValueError("composition must be a one-dimensional nonnegative vector")
    mass = float(array.sum())
    if mass <= 0:
        raise ValueError("empty composition has no normalized profile")
    return np.ascontiguousarray(array / mass, dtype=np.float64)


def cosine_similarity(left: NDArray, right: NDArray) -> float:
    left_f = np.asarray(left, dtype=np.float64)
    right_f = np.asarray(right, dtype=np.float64)
    denominator = float(np.linalg.norm(left_f) * np.linalg.norm(right_f))
    if denominator == 0.0:
        return 0.0
    return float(np.clip(np.dot(left_f, right_f) / denominator, 0.0, 1.0))


def bray_curtis_similarity(left: NDArray, right: NDArray) -> float:
    left_p = normalized_profile(left)
    right_p = normalized_profile(right)
    return float(np.clip(1.0 - 0.5 * np.abs(left_p - right_p).sum(), 0.0, 1.0))


def similarity(left: NDArray, right: NDArray, metric: str) -> float:
    if metric == "cosine":
        return cosine_similarity(left, right)
    if metric == "bray_curtis":
        return bray_curtis_similarity(left, right)
    raise ValueError(f"unknown metric: {metric}")


def _profiles(records: Sequence, metric: str) -> NDArray[np.float64]:
    daughters = np.vstack(
        [np.asarray(record.daughter, dtype=np.float64) for record in records]
    )
    if metric == "cosine":
        norms = np.linalg.norm(daughters, axis=1)
        return np.divide(
            daughters,
            norms[:, None],
            out=np.zeros_like(daughters),
            where=norms[:, None] != 0.0,
        )
    if metric == "bray_curtis":
        masses = daughters.sum(axis=1)
        return np.divide(
            daughters,
            masses[:, None],
            out=np.zeros_like(daughters),
            where=masses[:, None] != 0.0,
        )
    raise ValueError(f"unknown metric: {metric}")


def _profile(values: NDArray, metric: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if metric == "cosine":
        norm = float(np.linalg.norm(array))
        return array / norm if norm else np.zeros_like(array)
    if metric == "bray_curtis":
        mass = float(array.sum())
        return array / mass if mass else np.zeros_like(array)
    raise ValueError(f"unknown metric: {metric}")


def _row_similarity(
    profiles: NDArray[np.float64], anchor: NDArray[np.float64], metric: str
) -> NDArray[np.float64]:
    if metric == "cosine":
        return np.clip(profiles @ anchor, 0.0, 1.0)
    return np.clip(1.0 - 0.5 * np.abs(profiles - anchor).sum(axis=1), 0.0, 1.0)


def build_geometry(records: Sequence, metric: str) -> MetricGeometry:
    if not records:
        return MetricGeometry(
            boundary=np.empty(0, dtype=np.float64),
            profiles=np.empty((0, 0), dtype=np.float64),
            bands=tuple(np.empty(0, dtype=np.float64) for _ in range(RUN_LENGTH)),
        )
    boundary = np.asarray(
        [similarity(record.parent, record.daughter, metric) for record in records],
        dtype=np.float64,
    )
    profiles = _profiles(records, metric)
    bands: list[NDArray[np.float64]] = [np.empty(0, dtype=np.float64)]
    for offset in range(1, RUN_LENGTH):
        if metric == "cosine":
            values = np.einsum(
                "ij,ij->i", profiles[:-offset], profiles[offset:], optimize=True
            )
        else:
            values = 1.0 - 0.5 * np.abs(
                profiles[:-offset] - profiles[offset:]
            ).sum(axis=1)
        bands.append(np.clip(values, 0.0, 1.0))
    return MetricGeometry(boundary, profiles, tuple(bands))


def window_pairwise_minimum(geometry: MetricGeometry, start: int) -> float:
    return float(
        min(
            np.min(geometry.bands[offset][start : start + RUN_LENGTH - offset])
            for offset in range(1, RUN_LENGTH)
        )
    )


def window_statistics(records: Sequence, start: int) -> NDArray[np.float64]:
    window = records[start : start + RUN_LENGTH]
    if len(window) != RUN_LENGTH:
        raise ValueError("window statistics require eight complete fissions")
    counts = np.vstack(
        [np.asarray(record.daughter, dtype=np.float64) for record in window]
    )
    masses = counts.sum(axis=1)
    profiles = counts / masses[:, None]
    positive = profiles > 0.0
    entropy = -np.sum(
        np.where(positive, profiles * np.log(np.where(positive, profiles, 1.0)), 0.0),
        axis=1,
    )
    effective = np.exp(entropy)
    occupied = positive.sum(axis=1).astype(np.float64)
    sorted_profiles = np.sort(profiles, axis=1)
    top1 = sorted_profiles[:, -1]
    top2 = sorted_profiles[:, -2:].sum(axis=1)
    adjacent_tv = 0.5 * np.abs(profiles[1:] - profiles[:-1]).sum(axis=1)
    occupied_turnover: list[float] = []
    for left, right in zip(positive[:-1], positive[1:], strict=True):
        union = int(np.count_nonzero(left | right))
        intersection = int(np.count_nonzero(left & right))
        occupied_turnover.append(0.0 if union == 0 else 1.0 - intersection / union)
    turnover = np.asarray(occupied_turnover, dtype=np.float64)
    growth = np.asarray([record.growth_steps for record in window], dtype=np.float64)
    return np.asarray(
        [
            effective.mean(),
            effective.min(),
            occupied.mean(),
            occupied.min(),
            top1.mean(),
            top1.max(),
            top2.mean(),
            top2.max(),
            np.mean(top1 >= DOMINANCE_SHARE),
            np.mean(top2 >= DOMINANCE_SHARE),
            adjacent_tv.mean(),
            adjacent_tv.max(),
            turnover.mean(),
            turnover.max(),
            growth.sum(),
            growth.mean(),
            growth.max(),
        ],
        dtype=np.float64,
    )


def score_endpoint(
    records: Sequence,
    geometry: MetricGeometry,
    spec: EndpointSpec,
) -> EndpointOutcome:
    empty_stats = np.full(len(WINDOW_STAT_NAMES), np.nan, dtype=np.float64)
    inherited = geometry.boundary > spec.inheritance_cutoff
    breaks = np.flatnonzero(~inherited)
    if breaks.size == 0:
        return EndpointOutcome(
            False, -1, -1, -1, GATE_NO_BREAK, 0, 0, np.nan, np.nan,
            empty_stats.copy(), empty_stats.copy(),
        )
    first_break = int(breaks[0])
    anchor = _profile(records[first_break].parent, spec.metric)
    anchor_similarity = _row_similarity(geometry.profiles, anchor, spec.metric)
    first_run = -1
    onset = -1
    eligible = 0
    coherent = 0
    best_pair = -np.inf
    best_anchor = -np.inf
    precursor_stats = empty_stats.copy()
    event_stats = empty_stats.copy()
    for start in range(first_break + 1, len(records) - RUN_LENGTH + 1):
        stop = start + RUN_LENGTH
        if not bool(inherited[start:stop].all()):
            continue
        eligible += 1
        if first_run < 0:
            first_run = start
            precursor_stats = window_statistics(records, start)
        pair_margin = window_pairwise_minimum(geometry, start) - spec.coherence_cutoff
        best_pair = max(best_pair, pair_margin)
        if pair_margin <= 0.0:
            continue
        coherent += 1
        anchor_margin = spec.anchor_cutoff - float(anchor_similarity[start:stop].max())
        best_anchor = max(best_anchor, anchor_margin)
        if anchor_margin >= 0.0 and onset < 0:
            onset = start
            event_stats = window_statistics(records, start)
    if onset >= 0:
        gate = GATE_EVENT
    elif coherent > 0:
        gate = GATE_COHERENCE_ANCHOR_FAIL
    elif eligible > 0:
        gate = GATE_RUN_NO_COHERENCE
    else:
        gate = GATE_BREAK_NO_RUN
    return EndpointOutcome(
        event=onset >= 0,
        onset=onset,
        first_break=first_break,
        first_run=first_run,
        deepest_gate=gate,
        eligible_windows=eligible,
        coherent_windows=coherent,
        best_pairwise_margin=float(best_pair) if np.isfinite(best_pair) else np.nan,
        best_anchor_margin=float(best_anchor) if np.isfinite(best_anchor) else np.nan,
        precursor_stats=precursor_stats,
        event_stats=event_stats,
    )


def cross_evaluate_event_window(
    records: Sequence,
    source: EndpointOutcome,
    target: EndpointOutcome,
    target_geometry: MetricGeometry,
    target_spec: EndpointSpec,
) -> NDArray[np.float64]:
    values = np.full(len(CROSS_EVAL_NAMES), np.nan, dtype=np.float64)
    if not source.event or target.first_break < 0:
        return values
    start = source.onset
    stop = start + RUN_LENGTH
    values[0] = float(start > target.first_break)
    values[1] = float(
        target_geometry.boundary[start:stop].min() - target_spec.inheritance_cutoff
    )
    values[2] = float(
        window_pairwise_minimum(target_geometry, start) - target_spec.coherence_cutoff
    )
    anchor = _profile(records[target.first_break].parent, target_spec.metric)
    anchor_similarity = _row_similarity(target_geometry.profiles, anchor, target_spec.metric)
    values[3] = float(
        target_spec.anchor_cutoff - anchor_similarity[start:stop].max()
    )
    return values


def score_all_specs(
    records: Sequence,
    specs: Sequence[EndpointSpec],
) -> tuple[list[EndpointOutcome], NDArray[np.float64]]:
    geometries = {
        metric: build_geometry(records, metric)
        for metric in {spec.metric for spec in specs}
    }
    outcomes = [score_endpoint(records, geometries[spec.metric], spec) for spec in specs]
    cross = np.full(
        (len(specs), len(specs), len(CROSS_EVAL_NAMES)), np.nan, dtype=np.float64
    )
    for source_index, source in enumerate(outcomes):
        for target_index, (target, spec) in enumerate(zip(outcomes, specs, strict=True)):
            cross[source_index, target_index] = cross_evaluate_event_window(
                records, source, target, geometries[spec.metric], spec
            )
    return outcomes, cross


def calibration_comparisons(records: Sequence) -> dict[str, NDArray[np.float64]]:
    """Extract paired relation objects from registered-cosine precursors.

    Boundary comparisons use all observed fissions. Coherence and anchor
    comparisons use the union of unique objects occurring in any post-break
    run of eight strict cosine-H>0.90 boundaries. No coherence or anchor event
    result is used to select these objects.
    """

    cosine = build_geometry(records, "cosine")
    bray = build_geometry(records, "bray_curtis")
    output: dict[str, list[float]] = {
        "boundary_cosine": cosine.boundary.tolist(),
        "boundary_bray_curtis": bray.boundary.tolist(),
        "coherence_cosine": [],
        "coherence_bray_curtis": [],
        "anchor_cosine": [],
        "anchor_bray_curtis": [],
    }
    inherited = cosine.boundary > 0.90
    breaks = np.flatnonzero(~inherited)
    if breaks.size == 0:
        return {name: np.asarray(values, dtype=np.float64) for name, values in output.items()}
    first_break = int(breaks[0])
    starts = [
        start
        for start in range(first_break + 1, len(records) - RUN_LENGTH + 1)
        if bool(inherited[start : start + RUN_LENGTH].all())
    ]
    if not starts:
        return {name: np.asarray(values, dtype=np.float64) for name, values in output.items()}
    pair_indices: set[tuple[int, int]] = set()
    daughter_indices: set[int] = set()
    for start in starts:
        indices = range(start, start + RUN_LENGTH)
        daughter_indices.update(indices)
        pair_indices.update(
            (left, right)
            for left in indices
            for right in range(left + 1, start + RUN_LENGTH)
        )
    for left, right in sorted(pair_indices):
        for metric, geometry in (("cosine", cosine), ("bray_curtis", bray)):
            offset = right - left
            output[f"coherence_{metric}"].append(float(geometry.bands[offset][left]))
    for metric, geometry in (("cosine", cosine), ("bray_curtis", bray)):
        anchor = _profile(records[first_break].parent, metric)
        values = _row_similarity(geometry.profiles, anchor, metric)
        output[f"anchor_{metric}"].extend(float(values[index]) for index in sorted(daughter_indices))
    return {name: np.asarray(values, dtype=np.float64) for name, values in output.items()}


def quantile_match(
    source: NDArray,
    target: NDArray,
    source_cutoff: float,
) -> dict[str, float | int]:
    left = np.asarray(source, dtype=np.float64).ravel()
    right = np.asarray(target, dtype=np.float64).ravel()
    keep = np.isfinite(left) & np.isfinite(right)
    left = left[keep]
    right = right[keep]
    if left.size == 0:
        raise ValueError("quantile matching requires finite paired comparisons")
    percentile = float(np.mean(left <= source_cutoff))
    if percentile <= 0.0:
        cutoff = float(np.min(right))
    elif percentile >= 1.0:
        cutoff = float(np.max(right))
    else:
        cutoff = float(np.quantile(right, percentile, method="inverted_cdf"))
    return {
        "source_cutoff": float(source_cutoff),
        "source_fraction_le": percentile,
        "target_cutoff": cutoff,
        "target_fraction_le": float(np.mean(right <= cutoff)),
        "paired_comparisons": int(left.size),
    }


def deterministic_order(
    values: Iterable[int], seed_parts: Sequence[str | int]
) -> list[int]:
    prefix = "|".join(str(value) for value in seed_parts)
    return sorted(
        (int(value) for value in values),
        key=lambda value: hashlib.sha256(f"{prefix}|{value}".encode()).digest(),
    )


def match_event_controls(
    labels: NDArray,
    first_runs: NDArray,
    state_ids: Sequence[str],
    spec_names: Sequence[str] = SPEC_NAMES,
    matching_seed: str = "strict-event-matching-v1",
) -> list[dict[str, int | str]]:
    """Match positives to precursor-reaching negatives within the same state."""

    event = np.asarray(labels, dtype=np.int8)
    runs = np.asarray(first_runs, dtype=np.int16)
    if event.shape != runs.shape or event.ndim != 3:
        raise ValueError("labels and first_runs must be state x branch x spec")
    if event.shape[0] != len(state_ids) or event.shape[2] != len(spec_names):
        raise ValueError("matching metadata does not match arrays")
    rows: list[dict[str, int | str]] = []
    for state_index, state_id in enumerate(state_ids):
        for spec_index, spec_name in enumerate(spec_names):
            positives = deterministic_order(
                np.flatnonzero(event[state_index, :, spec_index] == 1),
                (matching_seed, "positive", spec_name, state_id),
            )
            controls = deterministic_order(
                np.flatnonzero(
                    (event[state_index, :, spec_index] == 0)
                    & (runs[state_index, :, spec_index] >= 0)
                ),
                (matching_seed, "control", spec_name, state_id),
            )
            for pair_index, (positive, control) in enumerate(zip(positives, controls)):
                rows.append(
                    {
                        "state_index": state_index,
                        "state_id": str(state_id),
                        "spec_index": spec_index,
                        "spec": str(spec_name),
                        "pair_index_within_state": pair_index,
                        "event_branch": positive,
                        "control_branch": control,
                    }
                )
    return rows
