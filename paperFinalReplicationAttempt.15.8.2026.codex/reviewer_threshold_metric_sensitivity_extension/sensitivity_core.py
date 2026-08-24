"""Pure scoring utilities for the post-hoc threshold/metric sensitivity.

The alternative similarity is Bray--Curtis similarity applied to normalized
compositions.  For profiles p and q it is ``1 - 0.5 * ||p - q||_1``.  This is
also one minus total-variation distance and, unlike raw-count Bray--Curtis,
does not confound composition with the parent/daughter mass difference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


F12_THRESHOLDS = (0.85, 0.875, 0.90, 0.925, 0.95)
F12_HORIZONS = (8, 10, 12, 16)
F12_RUN_LENGTHS = (2, 3, 4, 5)

F32_THRESHOLDS = F12_THRESHOLDS
F32_RUN_LENGTHS = (6, 8, 10)
F32_ANCHOR_THRESHOLDS = (0.80, 0.85, 0.90)


@dataclass(frozen=True)
class F12Definition:
    source_threshold: float
    horizon: int
    run_length: int

    @property
    def key(self) -> str:
        return (
            f"h{self.source_threshold:.3f}_f{self.horizon:02d}"
            f"_r{self.run_length}"
        )

    @property
    def registered_shape(self) -> bool:
        return (
            self.source_threshold == 0.90
            and self.horizon == 12
            and self.run_length == 3
        )


@dataclass(frozen=True)
class F32Definition:
    source_threshold: float
    run_length: int
    source_anchor_threshold: float

    @property
    def key(self) -> str:
        return (
            f"h{self.source_threshold:.3f}_r{self.run_length}"
            f"_a{self.source_anchor_threshold:.2f}"
        )

    @property
    def registered_shape(self) -> bool:
        return (
            self.source_threshold == 0.90
            and self.run_length == 8
            and self.source_anchor_threshold == 0.85
        )


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


def normalized_profile(values: NDArray) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or np.any(array < 0):
        raise ValueError("a composition must be a one-dimensional nonnegative vector")
    mass = float(array.sum())
    if mass <= 0:
        raise ValueError("an empty composition has no normalized profile")
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


def boundary_similarities(records: Sequence, metric: str) -> NDArray[np.float64]:
    similarity = {
        "cosine": cosine_similarity,
        "bray_curtis": bray_curtis_similarity,
    }.get(metric)
    if similarity is None:
        raise ValueError(f"unknown metric: {metric}")
    return np.asarray(
        [similarity(record.parent, record.daughter) for record in records],
        dtype=np.float64,
    )


def quantile_matched_cutoffs(
    cosine_values: NDArray,
    alternative_values: NDArray,
    cosine_cutoffs: Sequence[float],
) -> tuple[dict[float, float], list[dict[str, float | int]]]:
    """Map cutoffs by pooled empirical percentile, using an inverted CDF.

    The two metric arrays refer to the same comparison objects, but only their
    pooled marginal distributions are used.  Non-finite paired observations are
    removed before matching.
    """

    cosine = np.asarray(cosine_values, dtype=np.float64).ravel()
    alternative = np.asarray(alternative_values, dtype=np.float64).ravel()
    keep = np.isfinite(cosine) & np.isfinite(alternative)
    cosine = cosine[keep]
    alternative = alternative[keep]
    if cosine.size == 0:
        raise ValueError("no finite paired similarities are available")
    mapping: dict[float, float] = {}
    rows: list[dict[str, float | int]] = []
    for cutoff in cosine_cutoffs:
        percentile = float(np.mean(cosine <= float(cutoff)))
        # The endpoints are explicit because NumPy's quantile does not accept
        # q outside [0, 1] and because the empirical inverse is unambiguous here.
        if percentile <= 0.0:
            mapped = float(np.min(alternative))
        elif percentile >= 1.0:
            mapped = float(np.max(alternative))
        else:
            mapped = float(
                np.quantile(alternative, percentile, method="inverted_cdf")
            )
        mapping[float(cutoff)] = mapped
        rows.append(
            {
                "cosine_cutoff": float(cutoff),
                "cosine_fraction_le": percentile,
                "bray_curtis_cutoff": mapped,
                "bray_curtis_fraction_le": float(np.mean(alternative <= mapped)),
                "paired_observations": int(cosine.size),
            }
        )
    return mapping, rows


def score_f12_array(
    boundary_similarity: NDArray,
    cutoff_by_source_threshold: Mapping[float, float],
) -> NDArray[np.int8]:
    """Score every F12 definition over arbitrary leading dimensions.

    The last dimension is fission time. Extinction padding (NaN) is neither an
    inheritance nor a break, matching the frozen absorbing-future semantics.
    """

    values = np.asarray(boundary_similarity, dtype=np.float64)
    if values.ndim < 2 or values.shape[-1] < max(F12_HORIZONS):
        raise ValueError("boundary similarities must end in at least 16 fissions")
    flat = values.reshape((-1, values.shape[-1]))
    labels = np.zeros((flat.shape[0], len(F12_DEFINITIONS)), dtype=np.int8)
    for definition_index, definition in enumerate(F12_DEFINITIONS):
        cutoff = float(cutoff_by_source_threshold[definition.source_threshold])
        observed = flat[:, : definition.horizon]
        valid = np.isfinite(observed)
        inherited = valid & (observed > cutoff)
        break_mask = valid & ~inherited
        has_break = break_mask.any(axis=1)
        first_break = np.argmax(break_mask, axis=1)
        run_length = definition.run_length
        starts = definition.horizon - run_length + 1
        run_at = np.ones((flat.shape[0], starts), dtype=bool)
        for offset in range(run_length):
            run_at &= inherited[:, offset : offset + starts]
        start_indices = np.arange(starts, dtype=np.int16)[None, :]
        after_break = start_indices > first_break[:, None]
        labels[:, definition_index] = (
            has_break & np.any(run_at & after_break, axis=1)
        ).astype(np.int8)
    return labels.reshape(values.shape[:-1] + (len(F12_DEFINITIONS),))


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


def _row_similarity(profiles: NDArray, anchor: NDArray, metric: str) -> NDArray:
    if metric == "cosine":
        return np.clip(profiles @ anchor, 0.0, 1.0)
    return np.clip(1.0 - 0.5 * np.abs(profiles - anchor).sum(axis=1), 0.0, 1.0)


def _band_similarities(
    profiles: NDArray[np.float64], metric: str, maximum_run: int
) -> list[NDArray[np.float64]]:
    """Return similarities for pairs at offsets 1..maximum_run-1.

    Only pairs that can coexist in a requested window are evaluated. This cuts
    Bray--Curtis work by about half relative to a full 32x32 distance matrix.
    """

    bands: list[NDArray[np.float64]] = [np.empty(0, dtype=np.float64)]
    for offset in range(1, maximum_run):
        if metric == "cosine":
            values = np.einsum(
                "ij,ij->i", profiles[:-offset], profiles[offset:], optimize=True
            )
        else:
            values = 1.0 - 0.5 * np.abs(
                profiles[:-offset] - profiles[offset:]
            ).sum(axis=1)
        bands.append(np.clip(values, 0.0, 1.0))
    return bands


def _window_pairwise_minimum(
    bands: Sequence[NDArray[np.float64]], start: int, run_length: int
) -> float:
    return float(
        min(
            np.min(bands[offset][start : start + run_length - offset])
            for offset in range(1, run_length)
        )
    )


def score_f32_records(
    records: Sequence,
    metric: str,
    cutoff_by_source_threshold: Mapping[float, float],
    anchor_cutoff_by_source_threshold: Mapping[float, float],
) -> tuple[NDArray[np.int8], NDArray[np.int16], NDArray[np.float64]]:
    """Score the full strict-event grid for one absorbing future."""

    labels = np.zeros(len(F32_DEFINITIONS), dtype=np.int8)
    onsets = np.full(len(F32_DEFINITIONS), -1, dtype=np.int16)
    boundary = np.full(32, np.nan, dtype=np.float64)
    observed = boundary_similarities(records, metric)
    boundary[: len(observed)] = observed
    if not records:
        return labels, onsets, boundary

    profiles = _profiles(records, metric)
    bands: list[NDArray[np.float64]] | None = None
    window_minimum: dict[tuple[int, int], float] = {}
    definition_index = {
        (
            definition.source_threshold,
            definition.run_length,
            definition.source_anchor_threshold,
        ): index
        for index, definition in enumerate(F32_DEFINITIONS)
    }

    for source_threshold in F32_THRESHOLDS:
        cutoff = float(cutoff_by_source_threshold[source_threshold])
        inherited = observed > cutoff
        breaks = np.flatnonzero(~inherited)
        if breaks.size == 0:
            continue
        first_break = int(breaks[0])
        anchor = _profile(records[first_break].parent, metric)
        anchor_similarity = _row_similarity(profiles, anchor, metric)
        for run_length in F32_RUN_LENGTHS:
            last_start = len(records) - run_length
            if last_start < first_break + 1:
                continue
            qualifying: list[tuple[int, float]] = []
            for start in range(first_break + 1, last_start + 1):
                stop = start + run_length
                if not bool(inherited[start:stop].all()):
                    continue
                if bands is None:
                    bands = _band_similarities(profiles, metric, max(F32_RUN_LENGTHS))
                key = (start, run_length)
                if key not in window_minimum:
                    window_minimum[key] = _window_pairwise_minimum(
                        bands, start, run_length
                    )
                if window_minimum[key] <= cutoff:
                    continue
                qualifying.append(
                    (start, float(np.max(anchor_similarity[start:stop])))
                )
            for source_anchor in F32_ANCHOR_THRESHOLDS:
                index = definition_index[
                    (source_threshold, run_length, source_anchor)
                ]
                anchor_cutoff = float(
                    anchor_cutoff_by_source_threshold[source_anchor]
                )
                for start, maximum_anchor in qualifying:
                    if maximum_anchor <= anchor_cutoff:
                        labels[index] = 1
                        onsets[index] = start
                        break
    return labels, onsets, boundary


def jaccard(left: NDArray, right: NDArray) -> float:
    left_b = np.asarray(left, dtype=bool)
    right_b = np.asarray(right, dtype=bool)
    union = np.count_nonzero(left_b | right_b)
    if union == 0:
        return float("nan")
    return float(np.count_nonzero(left_b & right_b) / union)
