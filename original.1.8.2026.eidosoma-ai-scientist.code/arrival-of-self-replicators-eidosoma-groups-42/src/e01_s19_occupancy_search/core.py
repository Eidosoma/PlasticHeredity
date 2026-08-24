"""Deterministic primitives for the additive E01/S19-L07 search.

L07 is explicitly adaptive and occupancy-directed.  These helpers therefore
do not implement a confirmatory selection gate.  They do enforce exact
definitions, status-bearing missingness, deterministic statistics, and the
human-waived sole scientific target: closeness to paper occupancy 0.88.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from e01_frozen_timebase_ensemble.core import selected_clock_observations

VERSION = "E01-S19-L07-OCCUPANCY-SETTING-SEARCH-v1.0.0"
LOOP_ID = "S19-L07"
PAPER_OCCUPANCY_TARGET = 0.88
PAPER_OCCUPANCY_TOLERANCE = 0.03
BOOTSTRAP_REPLICATES = 4096


@dataclass(frozen=True, slots=True)
class ExploratoryExposureDefinition:
    """L07-only fixed Poisson exposure beyond the earlier S12F search range.

    The target paper does not state an exposure duration.  This object preserves
    the existing simulator interface and numerical kernel while permitting a
    bounded missing-configuration diagnostic through h=4.0.  It must not be
    interpreted as a recovered author parameter.
    """

    family: str
    h: float
    c: None = None
    h_max: None = None

    def validate(self) -> None:
        if self.family != "FIXED_COMMON_EXPOSURE":
            raise ValueError("L07 exploratory exposure is fixed-only")
        if not np.isfinite(self.h) or not 1.25 < self.h <= 4.0:
            raise ValueError("L07 extended exposure h must be in (1.25, 4.0]")

    @property
    def identity(self) -> str:
        self.validate()
        return f"FIXED-h={self.h:.17g}"


def _state_matrix(observations: Sequence[Any]) -> NDArray[np.float64]:
    values = np.asarray([item.state for item in observations], dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != 100:
        raise ValueError("expected a nonempty observations-by-100 state matrix")
    if np.any(values < 0) or not np.all(np.isfinite(values)):
        raise ValueError("composition states must be finite and nonnegative")
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms <= 0):
        raise ValueError("cosine similarity is undefined for an empty state")
    return values


def cosine_rows(left: NDArray[np.float64], right: NDArray[np.float64]) -> NDArray[np.float64]:
    """Rowwise CPU-float64 cosine, clipped only for roundoff."""

    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("cosine inputs must have the same two-dimensional shape")
    denominator = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    if np.any(denominator <= 0) or not np.all(np.isfinite(denominator)):
        raise ValueError("nonpositive or nonfinite cosine denominator")
    return np.clip(np.sum(a * b, axis=1) / denominator, -1.0, 1.0)


def adjacent_scores(observations: Sequence[Any], *, alignment: str) -> NDArray[np.float64]:
    """Calculate frozen adjacent-incoming or source technique-1-average scores."""

    values = _state_matrix(observations)
    if len(values) < 2:
        raise ValueError("at least two observations are required")
    adjacent = cosine_rows(values[:-1], values[1:])
    if alignment == "INCOMING_DUPLICATE_FIRST":
        return np.concatenate(([adjacent[0]], adjacent)).astype(np.float64, copy=False)
    if alignment == "TWO_NEIGHBOR_AVERAGE":
        return (
            np.concatenate(([adjacent[0]], adjacent))
            + np.concatenate((adjacent, [adjacent[-1]]))
        ) / 2.0
    raise ValueError(f"unsupported adjacent alignment {alignment!r}")


def _generation_boundaries(trajectory: Any) -> tuple[tuple[Any, Any], ...]:
    """Return exact (pre-fission parent, selected post-fission daughter) pairs."""

    observations = tuple(trajectory.observations)
    pairs: list[tuple[Any, Any]] = []
    for index, item in enumerate(observations):
        if item.observation_kind != "post_fission":
            continue
        if index == 0:
            raise ValueError("post-fission observation has no parent")
        parent = observations[index - 1]
        generation = int(item.growth_generation_one_based)
        if (
            parent.observation_kind != "molecular_update"
            or int(parent.growth_generation_one_based) != generation
        ):
            raise ValueError("post-fission observation is not preceded by its parent")
        pairs.append((parent, item))
    if len(pairs) != int(trajectory.completed_fissions):
        raise ValueError("fission boundary cardinality mismatch")
    return tuple(pairs)


def boundary_scores(trajectory: Any, *, boundary_object: str, alignment: str) -> NDArray[np.float64]:
    """Compute one source/paper-plausible generation-boundary score series."""

    pairs = _generation_boundaries(trajectory)
    if boundary_object == "PARENT_TO_SELECTED_DAUGHTER":
        parent = _state_matrix([item[0] for item in pairs])
        daughter = _state_matrix([item[1] for item in pairs])
        return cosine_rows(parent, daughter)
    if boundary_object == "POST_FISSION_TRACE":
        observations = [item[1] for item in pairs]
    elif boundary_object == "PRE_FISSION_TRACE":
        observations = [item[0] for item in pairs]
    else:
        raise ValueError(f"unsupported boundary object {boundary_object!r}")
    return adjacent_scores(observations, alignment=alignment)


def _project_generation_labels(
    trajectory: Any,
    boundary_labels: NDArray[np.bool_],
    *,
    projection: str,
    clock_id: str,
) -> tuple[tuple[Any, ...], list[bool | None]]:
    selected = selected_clock_observations(trajectory, clock_id)
    if len(boundary_labels) != int(trajectory.completed_fissions):
        raise ValueError("boundary label cardinality mismatch")
    output: list[bool | None] = []
    for item in selected:
        generation = int(item.growth_generation_one_based)
        kind = str(item.observation_kind)
        if projection == "INCOMING_INTERVAL_INITIAL_NEGATIVE":
            output.append(False if generation <= 0 else bool(boundary_labels[generation - 1]))
        elif projection in {
            "OUTGOING_INTERVAL_INITIAL_NEGATIVE",
            "OUTGOING_INTERVAL_PREFIX_INELIGIBLE",
        }:
            if kind == "post_fission" and generation >= 1:
                output.append(bool(boundary_labels[generation - 1]))
            elif kind == "molecular_update" and generation >= 2:
                output.append(bool(boundary_labels[generation - 2]))
            elif projection == "OUTGOING_INTERVAL_PREFIX_INELIGIBLE":
                output.append(None)
            else:
                output.append(False)
        else:
            raise ValueError(f"unsupported projection {projection!r}")
    return selected, output


def materialize_frozen_setting(trajectory: Any, setting: dict[str, Any]) -> pd.DataFrame:
    """Apply one pre-registered frozen-trajectory setting to one trajectory."""

    threshold = float(setting["threshold"])
    comparator = str(setting.get("comparator", "STRICT_GT"))
    if comparator not in {"STRICT_GT", "GE"}:
        raise ValueError("comparator must be STRICT_GT or GE")

    family = str(setting["family"])
    clock_id = str(setting.get("clockId", "C1_SELECTED_DAUGHTER_RETAINED"))
    boundary_object = setting.get("boundaryObject")
    projection = str(setting.get("projection", "ALL_OBSERVATIONS"))
    alignment = str(setting.get("alignment", "INCOMING_DUPLICATE_FIRST"))

    if family == "ADJACENT_CLOCK":
        observations = selected_clock_observations(trajectory, clock_id)
        scores = adjacent_scores(observations, alignment=alignment)
        labels = scores > threshold if comparator == "STRICT_GT" else scores >= threshold
        units = tuple(observations)
        labels_object: list[bool | None] = [bool(value) for value in labels]
    elif family == "BOUNDARY_SCORE":
        scores_boundary = boundary_scores(
            trajectory,
            boundary_object=str(boundary_object),
            alignment=alignment,
        )
        labels_boundary = (
            scores_boundary > threshold if comparator == "STRICT_GT" else scores_boundary >= threshold
        )
        if projection == "BOUNDARY_ONLY":
            pairs = _generation_boundaries(trajectory)
            units = tuple(item[1] for item in pairs)
            scores = scores_boundary
            labels_object = [bool(value) for value in labels_boundary]
        else:
            units, labels_object = _project_generation_labels(
                trajectory,
                labels_boundary.astype(bool, copy=False),
                projection=projection,
                clock_id=clock_id,
            )
            scores = np.full(len(units), np.nan, dtype=np.float64)
            for index, item in enumerate(units):
                generation = int(item.growth_generation_one_based)
                kind = str(item.observation_kind)
                if projection == "INCOMING_INTERVAL_INITIAL_NEGATIVE" and generation >= 1:
                    scores[index] = scores_boundary[generation - 1]
                elif projection.startswith("OUTGOING_INTERVAL"):
                    if kind == "post_fission" and generation >= 1:
                        scores[index] = scores_boundary[generation - 1]
                    elif kind == "molecular_update" and generation >= 2:
                        scores[index] = scores_boundary[generation - 2]
    else:
        raise ValueError(f"unsupported setting family {family!r}")

    rows = []
    for local_index, (item, label, score) in enumerate(zip(units, labels_object, scores, strict=True)):
        rows.append(
            {
                "roundId": setting["roundId"],
                "settingId": setting["settingId"],
                "settingPairId": setting["settingPairId"],
                "candidateId": str(trajectory.configuration_id),
                "matrixIndex": int(trajectory.matrix_index),
                "trajectoryId": str(trajectory.trajectory_id),
                "analysisUnitIndex": local_index,
                "rawObservationIndex": int(item.observation_index),
                "generation": int(item.growth_generation_one_based),
                "observationKind": str(item.observation_kind),
                "labelStatus": "ELIGIBLE" if label is not None else "INELIGIBLE_LOCKED_PREFIX",
                "isReplicator": label,
                "score": float(score) if np.isfinite(score) else None,
            }
        )
    return pd.DataFrame(rows)


def _episode_durations(labels: NDArray[np.bool_]) -> NDArray[np.int64]:
    if labels.size == 0:
        return np.asarray([], dtype=np.int64)
    padded = np.concatenate(([False], labels, [False]))
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return (ends - starts).astype(np.int64, copy=False)


def fingerprint(frame: pd.DataFrame) -> dict[str, Any]:
    """Compute a status-bearing temporal fingerprint; occupancy remains primary."""

    ordered = frame.sort_values("analysisUnitIndex", kind="stable")
    valid = ordered["isReplicator"].notna().to_numpy(dtype=bool)
    labels = ordered.loc[valid, "isReplicator"].to_numpy(dtype=bool)
    eligible_indices = ordered.loc[valid, "analysisUnitIndex"].to_numpy(dtype=np.int64)
    if labels.size == 0:
        raise ValueError("setting produced zero eligible analysis units")
    onset_positions = eligible_indices[labels]
    onset = int(onset_positions[0]) if onset_positions.size else None
    consistency: float | None = None
    if labels.size >= 3 and np.ptp(labels.astype(np.int8)) > 0:
        value = float(np.corrcoef(labels[:-1].astype(float), labels[1:].astype(float))[0, 1])
        consistency = value if np.isfinite(value) else None
    durations = _episode_durations(labels)
    entries = int(durations.size)
    cutoff = int(math.floor(0.25 * len(ordered)))
    prefix = ordered.iloc[:cutoff]
    prefix_valid = prefix["isReplicator"].dropna()
    return {
        "analysisUnitCount": int(len(ordered)),
        "eligibleCount": int(labels.size),
        "ineligibleCount": int(len(ordered) - labels.size),
        "persistence": int(np.count_nonzero(labels)),
        "occupancy": float(np.mean(labels)),
        "consistency": consistency,
        "firstOnsetRawIndex0": onset,
        "firstOnsetRawStep1": None if onset is None else onset + 1,
        "firstOnsetNormalized": None if onset is None else float(onset / max(1, len(ordered) - 1)),
        "entryCount": entries,
        "exitCount": int(entries - (1 if labels[-1] else 0)),
        "episodeCount": entries,
        "meanEpisodeDuration": float(np.mean(durations)) if durations.size else None,
        "medianEpisodeDuration": float(np.median(durations)) if durations.size else None,
        "longestEpisode": int(np.max(durations)) if durations.size else 0,
        "nonreplicatingAtCutoff": None if prefix_valid.empty else bool(not bool(prefix_valid.iloc[-1])),
        "noReplicatorThroughCutoff": bool(not prefix_valid.astype(bool).any()),
    }


def deterministic_seed(*parts: object) -> int:
    material = "\x1f".join([VERSION, *map(str, parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:16], "big")


def aggregate_occupancy(
    fingerprints: pd.DataFrame,
    *,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> pd.DataFrame:
    """Aggregate by catalytic matrix and attach deterministic matrix-bootstrap CIs."""

    required = {"roundId", "settingId", "settingPairId", "candidateId", "matrixIndex", "occupancy"}
    missing = required.difference(fingerprints.columns)
    if missing:
        raise ValueError(f"fingerprints missing columns: {sorted(missing)}")
    rows: list[dict[str, Any]] = []
    grouped = fingerprints.groupby(
        ["roundId", "settingId", "settingPairId", "candidateId"], sort=True, dropna=False
    )
    for (round_id, setting_id, pair_id, candidate_id), group in grouped:
        ordered = group.sort_values("matrixIndex", kind="stable")
        values = ordered["occupancy"].to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("occupancy aggregation received nonfinite values")
        rng = np.random.Generator(
            np.random.PCG64DXSM(deterministic_seed("bootstrap", round_id, setting_id, candidate_id))
        )
        indices = rng.integers(0, len(values), size=(bootstrap_replicates, len(values)))
        boot = np.mean(values[indices], axis=1)
        mean = float(np.mean(values))
        rows.append(
            {
                "roundId": round_id,
                "settingId": setting_id,
                "settingPairId": pair_id,
                "candidateId": candidate_id,
                "matrixCount": int(len(values)),
                "meanOccupancy": mean,
                "medianOccupancy": float(np.median(values)),
                "sdOccupancy": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "ci025MeanOccupancy": float(np.quantile(boot, 0.025)),
                "ci975MeanOccupancy": float(np.quantile(boot, 0.975)),
                "targetOccupancy": PAPER_OCCUPANCY_TARGET,
                "absoluteTargetError": abs(mean - PAPER_OCCUPANCY_TARGET),
                "withinPaperApproximateBand": bool(
                    abs(mean - PAPER_OCCUPANCY_TARGET) <= PAPER_OCCUPANCY_TOLERANCE
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_pairs(aggregate: pd.DataFrame) -> pd.DataFrame:
    """Rank settings solely by worst candidate-specific occupancy error."""

    rows: list[dict[str, Any]] = []
    for (round_id, pair_id), group in aggregate.groupby(["roundId", "settingPairId"], sort=True):
        errors = group["absoluteTargetError"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "roundId": round_id,
                "settingPairId": pair_id,
                "candidateCount": int(len(group)),
                "maximumAbsoluteTargetError": float(np.max(errors)),
                "meanAbsoluteTargetError": float(np.mean(errors)),
                "allCandidatesWithinApproximateBand": bool(group["withinPaperApproximateBand"].all()),
                "soleScientificTargetPassed": bool(group["withinPaperApproximateBand"].all()),
            }
        )
    result = pd.DataFrame(rows)
    return result.sort_values(
        ["maximumAbsoluteTargetError", "meanAbsoluteTargetError", "roundId", "settingPairId"],
        kind="stable",
    ).reset_index(drop=True)
