"""Frozen scientific primitives for E01/S19-L08.

L08 is an untouched, non-adaptive comparison of exactly two mechanisms that
were discovered in L07.  This module deliberately exposes no parameter-search
interface.  It fixes the label objects, temporal fingerprints, bootstrap seed
contract, paper-distance calculations, and terminal decision order used by the
pre-outcome lock.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from e01_frozen_timebase_ensemble.core import selected_clock_observations
from e01_s19_occupancy_search.core import (
    boundary_scores,
    materialize_frozen_setting,
)

VERSION = "E01-S19-L08-UNTOUCHED-OCCUPANCY-MECHANISM-DISCRIMINATION-v1.0.0"
LOOP_ID = "S19-L08"
BOOTSTRAP_REPLICATES = 4096
PAPER_OCCUPANCY_TARGET = 0.88
PAPER_OCCUPANCY_LOWER = 0.85
PAPER_OCCUPANCY_UPPER = 0.91

MECHANISM_A = "A_FISSION_BOUNDARY"
MECHANISM_B = "B_HIGH_EXPOSURE"
OBJECT_A_BOUNDARY = "A_BOUNDARY"
OBJECT_A_PROJECTED = "A_PROJECTED_MOLECULAR"
OBJECT_B_MOLECULAR = "B_MOLECULAR"

PAPER_TARGETS: dict[str, tuple[float, float]] = {
    "selectedClockLength": (716.0 / 0.88, 198.0 / 0.88),
    "persistence": (716.0, 198.0),
    "occupancy": (0.88, 0.03),
    "consistency": (0.38, 0.06),
    "firstOnsetRawStep1": (37.0, 27.0),
    "firstOnsetNormalized": (0.37, 0.27),
}

RAW_DISTANCE_METRICS = (
    "selectedClockLength",
    "persistence",
    "occupancy",
    "consistency",
    "firstOnsetRawStep1",
)
NORMALIZED_DISTANCE_METRICS = (
    "selectedClockLength",
    "persistence",
    "occupancy",
    "consistency",
    "firstOnsetNormalized",
)


def _sha256_array(values: NDArray[Any]) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(b"\0")
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def mechanism_setting(mechanism_id: str, analysis_object_id: str) -> dict[str, Any]:
    """Return one of the three and only three locked label materializations."""

    common = {
        "roundId": LOOP_ID,
        "threshold": 0.9,
        "comparator": "STRICT_GT",
        "clockId": "C1_SELECTED_DAUGHTER_RETAINED",
        "alignment": "INCOMING_DUPLICATE_FIRST",
    }
    if mechanism_id == MECHANISM_A and analysis_object_id == OBJECT_A_BOUNDARY:
        return {
            **common,
            "settingId": "L08-A-BOUNDARY",
            "settingPairId": OBJECT_A_BOUNDARY,
            "family": "BOUNDARY_SCORE",
            "boundaryObject": "PARENT_TO_SELECTED_DAUGHTER",
            "projection": "BOUNDARY_ONLY",
        }
    if mechanism_id == MECHANISM_A and analysis_object_id == OBJECT_A_PROJECTED:
        return {
            **common,
            "settingId": "L08-A-PROJECTED",
            "settingPairId": OBJECT_A_PROJECTED,
            "family": "BOUNDARY_SCORE",
            "boundaryObject": "PARENT_TO_SELECTED_DAUGHTER",
            "projection": "OUTGOING_INTERVAL_PREFIX_INELIGIBLE",
        }
    if mechanism_id == MECHANISM_B and analysis_object_id == OBJECT_B_MOLECULAR:
        return {
            **common,
            "settingId": "L08-B-MOLECULAR",
            "settingPairId": OBJECT_B_MOLECULAR,
            "family": "ADJACENT_CLOCK",
            "projection": "ALL_OBSERVATIONS",
        }
    raise ValueError(f"unregistered mechanism/object: {mechanism_id}/{analysis_object_id}")


def materialize_analysis_object(
    trajectory: Any,
    mechanism_id: str,
    analysis_object_id: str,
) -> pd.DataFrame:
    frame = materialize_frozen_setting(
        trajectory,
        mechanism_setting(mechanism_id, analysis_object_id),
    ).copy()
    frame.insert(0, "mechanismId", mechanism_id)
    frame.insert(1, "analysisObjectId", analysis_object_id)
    return frame


def _run_descriptors(labels: NDArray[np.bool_], desired: bool) -> dict[str, Any]:
    """Episode count, duration, and onset spacing for one label polarity."""

    values = np.asarray(labels, dtype=bool)
    mask = values if desired else ~values
    padded = np.concatenate(([False], mask, [False]))
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1).astype(np.int64, copy=False)
    ends = np.flatnonzero(changes == -1).astype(np.int64, copy=False)
    durations = (ends - starts).astype(np.int64, copy=False)
    spacings = np.diff(starts).astype(np.int64, copy=False)
    prefix = "positive" if desired else "negative"
    return {
        f"{prefix}EpisodeCount": int(durations.size),
        f"{prefix}MeanEpisodeDuration": (
            float(np.mean(durations)) if durations.size else None
        ),
        f"{prefix}MedianEpisodeDuration": (
            float(np.median(durations)) if durations.size else None
        ),
        f"{prefix}LongestEpisodeDuration": (
            int(np.max(durations)) if durations.size else 0
        ),
        f"{prefix}MeanEpisodeSpacing": (
            float(np.mean(spacings)) if spacings.size else None
        ),
        f"{prefix}MedianEpisodeSpacing": (
            float(np.median(spacings)) if spacings.size else None
        ),
        f"{prefix}EpisodeStartIndices": starts.tolist(),
        f"{prefix}EpisodeDurations": durations.tolist(),
    }


def label_fingerprint(frame: pd.DataFrame) -> dict[str, Any]:
    """Compute the complete locked boundary or molecular temporal fingerprint."""

    required = {
        "analysisUnitIndex",
        "rawObservationIndex",
        "labelStatus",
        "isReplicator",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"label frame missing columns {sorted(missing)}")
    ordered = frame.sort_values("analysisUnitIndex", kind="stable").reset_index(drop=True)
    eligible_mask = ordered["isReplicator"].notna().to_numpy(dtype=bool)
    labels = ordered.loc[eligible_mask, "isReplicator"].to_numpy(dtype=bool)
    eligible_unit_indices = ordered.loc[
        eligible_mask, "analysisUnitIndex"
    ].to_numpy(dtype=np.int64)
    if labels.size == 0:
        return {
            "fingerprintStatus": "INELIGIBLE_NO_LOCKED_ANALYSIS_UNITS",
            "analysisUnitLength": int(len(ordered)),
            "eligibleLength": 0,
            "ineligibleLength": int(len(ordered)),
            "labelSha256": None,
            "eligibilitySha256": _sha256_array(eligible_mask.astype(np.int8)),
        }

    positives = np.flatnonzero(labels)
    onset_eligible_position = int(positives[0]) if positives.size else None
    onset_index0 = (
        int(eligible_unit_indices[onset_eligible_position])
        if onset_eligible_position is not None
        else None
    )
    onset_step1 = None if onset_index0 is None else onset_index0 + 1
    onset_normalized = (
        None
        if onset_index0 is None
        else float(onset_index0 / max(1, len(ordered) - 1))
    )

    consistency: float | None = None
    consistency_status = "UNDEFINED_CONSTANT_OR_TOO_SHORT"
    if labels.size >= 3 and np.ptp(labels.astype(np.int8)) > 0:
        with np.errstate(invalid="ignore", divide="ignore"):
            value = float(
                np.corrcoef(labels[:-1].astype(float), labels[1:].astype(float))[0, 1]
            )
        if np.isfinite(value):
            consistency = value
            consistency_status = "DEFINED"

    positive = _run_descriptors(labels, True)
    negative = _run_descriptors(labels, False)
    label_codes = labels.astype(np.int8)
    return {
        "fingerprintStatus": "ELIGIBLE",
        "analysisUnitLength": int(len(ordered)),
        "eligibleLength": int(labels.size),
        "ineligibleLength": int(len(ordered) - labels.size),
        "persistence": int(np.count_nonzero(labels)),
        "negativePersistence": int(np.count_nonzero(~labels)),
        "occupancy": float(np.mean(labels)),
        "firstOnsetEligiblePositionIndex0": onset_eligible_position,
        "firstOnsetRawIndex0": onset_index0,
        "firstOnsetRawStep1": onset_step1,
        "firstOnsetNormalized": onset_normalized,
        "consistency": consistency,
        "consistencyStatus": consistency_status,
        "labelSha256": _sha256_array(label_codes),
        "eligibilitySha256": _sha256_array(eligible_mask.astype(np.int8)),
        **positive,
        **negative,
    }


def trajectory_diagnostics(trajectory: Any) -> dict[str, Any]:
    """Calculate the preregistered non-label trajectory discriminants."""

    selected = selected_clock_observations(
        trajectory, "C1_SELECTED_DAUGHTER_RETAINED"
    )
    completed = [row for row in trajectory.generations if row.pre_fission_mass is not None]
    pre = np.asarray([row.pre_fission_mass for row in completed], dtype=np.float64)
    post = np.asarray([row.post_fission_mass for row in completed], dtype=np.float64)
    overshoot = np.asarray(
        [row.overshoot_before_trim for row in completed], dtype=np.float64
    )
    parent_daughter = (
        boundary_scores(
            trajectory,
            boundary_object="PARENT_TO_SELECTED_DAUGHTER",
            alignment="INCOMING_DUPLICATE_FIRST",
        )
        if trajectory.completed_fissions
        else np.asarray([], dtype=np.float64)
    )
    maxstep_count = int(
        sum(row.terminal_status == "max_steps_reached" for row in trajectory.generations)
    )
    attempted_generations = int(len(trajectory.generations))
    complete = bool(
        trajectory.terminal_status == "requested_fissions_completed"
        and trajectory.completed_fissions == 100
    )
    return {
        "trajectoryStatus": "COMPLETE" if complete else "INCOMPLETE_RETAINED",
        "selectedClockLength": int(len(selected)),
        "boundaryUnitLength": int(trajectory.completed_fissions),
        "completedFissions": int(trajectory.completed_fissions),
        "terminalStatus": str(trajectory.terminal_status),
        "extinctionGeneration": trajectory.extinction_generation,
        "attemptedGenerationCount": attempted_generations,
        "maxStepTerminationCount": maxstep_count,
        "maxStepTerminationFraction": (
            float(maxstep_count / attempted_generations) if attempted_generations else None
        ),
        "meanParentDaughterSimilarity": (
            float(np.mean(parent_daughter)) if parent_daughter.size else None
        ),
        "medianParentDaughterSimilarity": (
            float(np.median(parent_daughter)) if parent_daughter.size else None
        ),
        "meanPreFissionMass": float(np.mean(pre)) if pre.size else None,
        "medianPreFissionMass": float(np.median(pre)) if pre.size else None,
        "meanPostFissionMass": float(np.mean(post)) if post.size else None,
        "medianPostFissionMass": float(np.median(post)) if post.size else None,
        "meanPretrimOvershoot": float(np.mean(overshoot)) if overshoot.size else None,
        "q95PretrimOvershoot": (
            float(np.quantile(overshoot, 0.95)) if overshoot.size else None
        ),
        "maximumPretrimOvershoot": float(np.max(overshoot)) if overshoot.size else None,
        "parentDaughterSimilaritySha256": (
            _sha256_array(parent_daughter) if parent_daughter.size else None
        ),
    }


def deterministic_seed(root_hex: str, *parts: object) -> int:
    """Domain-separated 128-bit PCG seed, with the root included literally."""

    material = "\x1f".join([VERSION, root_hex, *map(str, parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:16], "big")


def bootstrap_indices(
    root_hex: str,
    *identity: object,
    matrix_count: int = 100,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> NDArray[np.int64]:
    if matrix_count <= 0 or replicates != BOOTSTRAP_REPLICATES:
        raise ValueError("L08 requires a positive matrix count and exactly 4096 replicates")
    rng = np.random.Generator(
        np.random.PCG64DXSM(deterministic_seed(root_hex, "bootstrap", *identity))
    )
    return rng.integers(0, matrix_count, size=(replicates, matrix_count), dtype=np.int64)


def paper_distance(values: Mapping[str, float], onset_mode: str) -> float:
    metrics = (
        RAW_DISTANCE_METRICS if onset_mode == "RAW_ONSET" else NORMALIZED_DISTANCE_METRICS
    )
    if onset_mode not in {"RAW_ONSET", "NORMALIZED_ONSET"}:
        raise ValueError("unknown onset mode")
    deviations = []
    for metric in metrics:
        value = float(values[metric])
        target, scale = PAPER_TARGETS[metric]
        if not np.isfinite(value):
            return float("nan")
        deviations.append((value - target) / scale)
    return float(math.sqrt(np.mean(np.square(deviations))))


def absolute_scaled_error(metric: str, value: float) -> float:
    target, scale = PAPER_TARGETS[metric]
    return abs(float(value) - target) / scale


def occupancy_in_band(value: float) -> bool:
    return bool(PAPER_OCCUPANCY_LOWER <= float(value) <= PAPER_OCCUPANCY_UPPER)


def terminal_decision(
    *,
    operational_integrity_passed: bool,
    joint_occupancy_gate_passed: bool,
    fission_preference_gates_passed: bool,
    exposure_preference_gates_passed: bool,
) -> str:
    """Apply the frozen resolution order without favorable tie breaking."""

    if not operational_integrity_passed:
        return "LOOP_FAILED_CLOSED"
    if not joint_occupancy_gate_passed:
        return "NEITHER_MECHANISM_REPRODUCES_ON_UNTOUCHED_DATA"
    if fission_preference_gates_passed and not exposure_preference_gates_passed:
        return "EVIDENCE_FAVORS_FISSION_BOUNDARY_MECHANISM"
    if exposure_preference_gates_passed and not fission_preference_gates_passed:
        return "EVIDENCE_FAVORS_HIGH_EXPOSURE_MECHANISM"
    return "NONIDENTIFIABLE_BETWEEN_FROZEN_MECHANISMS"


def exact_fingerprint_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Value-preserving equality for deterministic result regeneration."""

    if set(left) != set(right):
        return False
    for key in left:
        a, b = left[key], right[key]
        if isinstance(a, float) or isinstance(b, float):
            if a is None or b is None:
                if a is not b:
                    return False
            elif math.isnan(float(a)) and math.isnan(float(b)):
                continue
            elif float(a) != float(b):
                return False
        elif a != b:
            return False
    return True


def analysis_objects_for_mechanism(mechanism_id: str) -> Sequence[str]:
    if mechanism_id == MECHANISM_A:
        return (OBJECT_A_BOUNDARY, OBJECT_A_PROJECTED)
    if mechanism_id == MECHANISM_B:
        return (OBJECT_B_MOLECULAR,)
    raise ValueError(mechanism_id)
