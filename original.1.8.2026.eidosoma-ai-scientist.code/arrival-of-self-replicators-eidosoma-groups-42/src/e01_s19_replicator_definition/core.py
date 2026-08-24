"""Outcome-blind contracts and statistics for E01/S19 Loop 2.

The module fixes one implementation for each of the four human-directed label
families.  It contains no emergence calculation and no filesystem I/O.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Iterable, Literal

import numpy as np
from numpy.typing import NDArray

VERSION = "E01-S19-L02-REPLICATOR-DEFINITION-TEMPORAL-FINGERPRINT-v1.0.0"
RESEARCH_STEP_ID = "S19"
LOOP_ID = "S19-L02"
ROOT_SEED_HEX = "56862dd64d12c64bc4cdf2ab6472d65ac0246bcba1089227114fca78764edbad"
CANDIDATE_IDS = ("S12F-CANDIDATE-02", "S12F-CANDIDATE-03")
COMPARATOR_LABEL_ID = "MOL_ADJACENT_INCOMING_H900"
BOOTSTRAP_REPLICATES = 4096


@dataclass(frozen=True, slots=True)
class LabelDefinition:
    """One and only one implementation of an authorized label family."""

    ordinal: int
    label_id: str
    family_name: str
    implementation_id: str
    evidence_class: str
    temporal_scope: str
    compositional_coordinates: str
    distance_or_similarity: str
    recurrence_rule: str
    reference_rule: str
    missing_data_rule: str
    global_reference: bool
    comparator_only: bool
    source_grounding_gate: bool
    unresolved_material_choice: str | None


LABEL_DEFINITIONS = (
    LabelDefinition(
        1,
        COMPARATOR_LABEL_ID,
        "ADJACENT_MOLECULAR_H_GT_0_9",
        "e01_creative_directional_search.label_trajectory/MOLECULAR_ADJACENT_INCOMING",
        "FROZEN_S13Y_PRIMARY_SOURCE_TRANSPLANT_COMPARATOR",
        "INCOMING_LOCAL_EXCEPT_INITIAL_DUPLICATES_FIRST_INCOMING_VALUE",
        "L1_CLOSED_100_COMPONENT_RELATIVE_COMPOSITION",
        "COSINE_H_STRICTLY_GREATER_THAN_0.9",
        "NONE_ADJACENT_SMOOTHNESS_ONLY",
        "IMMEDIATELY_PRECEDING_SELECTED_CLOCK_STATE",
        "ALL_FROZEN_SELECTED_CLOCK_ROWS_ELIGIBLE",
        False,
        True,
        False,
        "NOT_A_RECURRING_ATTRACTOR_DEFINITION",
    ),
    LabelDefinition(
        2,
        "PF_DOMINANT_COMPONENT_CENTROID_H900",
        "DOMINANT_RECURRING_COMPOSITION_CENTROID_MEMBERSHIP",
        "e01_creative_directional_search.label_trajectory/POSTFISSION_DOMINANT_COMPONENT_CENTROID",
        "FROZEN_S13X_PAPER_INFERRED_RECONSTRUCTION",
        "RETROSPECTIVE_COMPLETED_TRAJECTORY",
        "L1_CLOSED_POSTFISSION_AND_SELECTED_CLOCK_RELATIVE_COMPOSITION",
        "POSTFISSION_COSINE_GRAPH_AT_GREATER_THAN_OR_EQUAL_TO_0.9_THEN_SELECTED_TO_CENTROID_COSINE_AT_GREATER_THAN_OR_EQUAL_TO_0.9",
        "LARGEST_SINGLE_LINKAGE_COMPONENT_OVER_100_POSTFISSION_STATES",
        "MEAN_COMPOSITION_OF_DOMINANT_COMPONENT; LARGEST_SIZE_THEN_EARLIEST_MEMBER_TIE",
        "NONEMPTY_STATES_REQUIRED; FAIL_TRAJECTORY_ON_EMPTY_STATE",
        True,
        False,
        False,
        "PAPER_SAYS_EUCLIDEAN_BUT_DOES_NOT_FIX_COSINE_THRESHOLD_LINKAGE_OR_CENTROID_RULE",
    ),
    LabelDefinition(
        3,
        "PF_EUCLIDEAN_KMEANS_DOMINANT",
        "RECURRING_COMPOSITION_CLUSTER_MEMBERSHIP",
        "e01_creative_directional_search.label_trajectory/POSTFISSION_EUCLIDEAN_KMEANS_DOMINANT",
        "FROZEN_S13X_PAPER_INFERRED_RECONSTRUCTION",
        "RETROSPECTIVE_COMPLETED_TRAJECTORY",
        "L1_CLOSED_POSTFISSION_AND_SELECTED_CLOCK_RELATIVE_COMPOSITION",
        "EUCLIDEAN_L2_KMEANS",
        "K_IN_2_TO_10_MAXIMUM_SILHOUETTE; EXACT_TIE_TO_LOWER_K; N_INIT_10; LLOYD",
        "LARGEST_POSTFISSION_CLUSTER; EXACT_SIZE_TIE_TO_LOWEST_SKLEARN_CLUSTER_ID",
        "NONEMPTY_STATES_REQUIRED; FAIL_TRAJECTORY_IF_NO_FINITE_SILHOUETTE_SOLUTION",
        True,
        False,
        False,
        "PAPER_DOES_NOT_FIX_KMEANS_K_RANGE_SILHOUETTE_OR_DOMINANT_CLUSTER_TIE_RULE",
    ),
    LabelDefinition(
        4,
        "PF_HISTORICAL_ADJACENT_AVERAGE_H090",
        "HISTORICAL_GARD_TECHNIQUE1_COMPOTYPE_NONDRIFT",
        "frozen_S13Y_HISTORICAL_H090_REPLICATOR_from_e01_gard_historical.historical_nondrift_technique1",
        "SOURCE_TRACEABLE_HISTORICAL_GARD_V10_RECONSTRUCTION",
        "LOCAL_ADJACENT_WITH_OUTGOING_NEIGHBOR_FOR_INTERIOR_POSTFISSION_GENERATIONS",
        "RAW_POSTFISSION_COUNTS_SOURCE_L2_NORMALIZED",
        "MEAN_OF_INCOMING_AND_OUTGOING_COSINE_H_STRICTLY_GREATER_THAN_0.9",
        "SOURCE_TECHNIQUE1_LOCAL_NONDRIFT",
        "ADJACENT_POSTFISSION_GENERATIONS_PROPAGATED_TO_SELECTED_MOLECULAR_CLOCK",
        "INITIAL_SELECTED_STATE_INELIGIBLE_AND_RETAINED; ALL_GENERATION_ROWS_ELIGIBLE",
        False,
        False,
        True,
        "HISTORICAL_PUBLIC_GARD_V10_IS_NOT_THE_UNAVAILABLE_TARGET_PAPER_CODE",
    ),
)


PAPER_TARGETS = {
    "persistence": (716.0, 198.0),
    "occupancy": (0.88, 0.03),
    "consistency": (0.38, 0.06),
    "firstOnsetRawScore": (37.0, 27.0),
    "firstOnsetNormalizedScore": (0.37, 0.27),
}


def seed_material(*identity: object) -> bytes:
    return "\x1f".join([VERSION, ROOT_SEED_HEX, *map(str, identity)]).encode("utf-8")


def derive_seed128(*identity: object) -> int:
    """Derive one deterministic PCG64DXSM-compatible 128-bit seed."""

    return int.from_bytes(hashlib.sha256(seed_material(*identity)).digest()[:16], "big")


def derive_seed32(*identity: object) -> int:
    """Derive one deterministic legacy 32-bit seed."""

    return int.from_bytes(hashlib.sha256(seed_material(*identity)).digest()[:4], "big")


def _episodes(
    sequence_indices: NDArray[np.int64], labels: NDArray[np.bool_]
) -> list[tuple[int, int, int]]:
    if len(labels) == 0:
        return []
    episodes: list[tuple[int, int, int]] = []
    start: int | None = None
    prior_index: int | None = None
    for index, label in zip(sequence_indices, labels, strict=True):
        current = int(index)
        contiguous = prior_index is not None and current == prior_index + 1
        if bool(label) and (start is None or not contiguous):
            if start is not None and prior_index is not None:
                episodes.append((start, prior_index, prior_index - start + 1))
            start = current
        elif not bool(label) and start is not None:
            assert prior_index is not None
            episodes.append((start, prior_index, prior_index - start + 1))
            start = None
        prior_index = current
    if start is not None and prior_index is not None:
        episodes.append((start, prior_index, prior_index - start + 1))
    return episodes


def consecutive_binary_consistency(
    sequence_indices: NDArray[np.int64], labels: NDArray[np.bool_]
) -> float | None:
    """Pearson correlation of consecutive eligible molecular labels."""

    if len(labels) < 3:
        return None
    contiguous = np.diff(sequence_indices) == 1
    first = labels[:-1][contiguous].astype(np.float64)
    second = labels[1:][contiguous].astype(np.float64)
    if len(first) < 2 or np.ptp(first) == 0 or np.ptp(second) == 0:
        return None
    value = float(np.corrcoef(first, second)[0, 1])
    return value if np.isfinite(value) else None


def fingerprint_from_labels(
    *,
    sequence_indices: Iterable[int],
    labels: Iterable[bool | None],
    total_clock_count: int,
    observation_kinds: Iterable[str],
    global_reference: bool,
) -> dict[str, Any]:
    """Compute the fully locked temporal fingerprint for one trajectory."""

    indices_all = np.asarray(list(sequence_indices), dtype=np.int64)
    labels_object = np.asarray(list(labels), dtype=object)
    kinds_all = np.asarray(list(observation_kinds), dtype=object)
    if not (len(indices_all) == len(labels_object) == len(kinds_all)):
        raise ValueError("label fingerprint inputs have inconsistent lengths")
    if total_clock_count <= 0 or len(indices_all) != total_clock_count:
        raise ValueError("label rows must retain the complete selected molecular clock")
    if not np.array_equal(indices_all, np.arange(total_clock_count, dtype=np.int64)):
        raise ValueError("selected molecular sequence indices must be complete and ordered")
    valid = np.asarray([value is not None and not (isinstance(value, float) and np.isnan(value)) for value in labels_object])
    indices = indices_all[valid]
    binary = np.asarray([bool(value) for value in labels_object[valid]], dtype=bool)
    if len(binary) == 0:
        raise ValueError("trajectory has no eligible label rows")
    onset_positions = indices[binary]
    onset = int(onset_positions[0]) if len(onset_positions) else None
    onset_normalized = (
        float(onset / (total_clock_count - 1))
        if onset is not None and total_clock_count > 1
        else (0.0 if onset is not None else None)
    )
    episodes = _episodes(indices, binary)
    durations = np.asarray([item[2] for item in episodes], dtype=np.float64)
    cutoff = int(math.floor(0.25 * total_clock_count))
    cutoff_index = max(0, cutoff - 1)
    eligible_through_cutoff = valid & (indices_all <= cutoff_index)
    latest_eligible = np.flatnonzero(eligible_through_cutoff)
    at_cutoff: bool | None = None
    if len(latest_eligible):
        at_cutoff = bool(labels_object[int(latest_eligible[-1])])
    no_positive_through_cutoff = not any(
        bool(value)
        for value, keep in zip(labels_object, eligible_through_cutoff, strict=True)
        if keep
    )
    post_mask = valid & (kinds_all == "post_fission")
    post_indices = indices_all[post_mask]
    post_labels = np.asarray([bool(value) for value in labels_object[post_mask]], dtype=bool)
    post_episodes = _episodes(post_indices, post_labels)
    post_positive = post_indices[post_labels]
    reference_span: float | None = None
    if global_reference and len(post_positive):
        reference_span = (
            float((post_positive[-1] - post_positive[0]) / max(1, total_clock_count - 1))
        )
    return {
        "totalClockCount": int(total_clock_count),
        "eligibleCount": int(len(binary)),
        "ineligibleCount": int(total_clock_count - len(binary)),
        "persistence": int(np.count_nonzero(binary)),
        "occupancy": float(np.mean(binary)),
        "consistency": consecutive_binary_consistency(indices, binary),
        "firstOnsetRawIndex0": onset,
        "firstOnsetRawStep1": None if onset is None else onset + 1,
        "firstOnsetNormalized": onset_normalized,
        "firstOnsetRawScore": float(total_clock_count if onset is None else onset),
        "firstOnsetNormalizedScore": float(1.0 if onset is None else onset_normalized),
        "neverReplicator": bool(onset is None),
        "entryCount": int(len(episodes)),
        "exitCount": int(sum(1 for _, end, _ in episodes if end < total_clock_count - 1)),
        "episodeCount": int(len(episodes)),
        "meanEpisodeDuration": float(np.mean(durations)) if len(durations) else None,
        "medianEpisodeDuration": float(np.median(durations)) if len(durations) else None,
        "longestEpisode": int(np.max(durations)) if len(durations) else 0,
        "cutoffCount": cutoff,
        "cutoffIndex0": cutoff_index,
        "isNonreplicatingAtCutoff": None if at_cutoff is None else not at_cutoff,
        "noReplicatorObservedThroughCutoff": bool(no_positive_through_cutoff),
        "postFissionCount": int(np.count_nonzero(post_mask)),
        "postFissionReplicatorCount": int(np.count_nonzero(post_labels)),
        "postFissionReplicatorFraction": float(np.mean(post_labels)) if len(post_labels) else None,
        "postFissionEpisodeCount": int(len(post_episodes)),
        "sameReferenceRecurrenceApplicable": bool(global_reference),
        "sameReferenceReentryCount": int(max(0, len(post_episodes) - 1)) if global_reference else None,
        "sameReferenceTemporalSpanNormalized": reference_span,
    }


def paper_fingerprint_distance(
    summary: dict[str, float | int | None],
    *,
    onset_mode: Literal["RAW", "NORMALIZED"],
) -> float | None:
    """Standardized distance to the paper's control Table-1 fingerprint."""

    onset_key = "firstOnsetRawScore" if onset_mode == "RAW" else "firstOnsetNormalizedScore"
    keys = ("persistence", "occupancy", "consistency", onset_key)
    values: list[float] = []
    for key in keys:
        raw = summary.get(key)
        if raw is None or not np.isfinite(float(raw)):
            return None
        target, scale = PAPER_TARGETS[key]
        values.append((float(raw) - target) / scale)
    return float(np.sqrt(np.mean(np.square(values))))


def closer_dimension_count(
    candidate: dict[str, float | int | None],
    comparator: dict[str, float | int | None],
    *,
    onset_mode: Literal["RAW", "NORMALIZED"],
) -> tuple[int, bool]:
    """Count target dimensions strictly closer than the adjacent-H comparator."""

    onset_key = "firstOnsetRawScore" if onset_mode == "RAW" else "firstOnsetNormalizedScore"
    keys = ("persistence", "occupancy", "consistency", onset_key)
    count = 0
    structure_improved = False
    for key in keys:
        left, right = candidate.get(key), comparator.get(key)
        if left is None or right is None:
            continue
        target, _ = PAPER_TARGETS[key]
        improved = abs(float(left) - target) < abs(float(right) - target)
        count += int(improved)
        if key in ("consistency", onset_key) and improved:
            structure_improved = True
    return count, structure_improved
