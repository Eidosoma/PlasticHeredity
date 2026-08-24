"""Outcome-blind contracts for E01/S19 Loop 6.

The module fixes one past-only post-fission-boundary recurrence label and the
frozen adjacent-H comparator.  It has no simulation, PhiRL, emergence,
prediction, intervention, or scientific filesystem logic.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from e01_clean_directional_confirmation.core import fixed_label_spec
from e01_creative_directional_search.core import (
    label_trajectory as frozen_label_trajectory,
)
from e01_frozen_timebase_ensemble.core import selected_clock_observations

VERSION = "E01-S19-L06-PAST-ONLY-MULTIATTRACTOR-BOUNDARY-RECURRENCE-v1.0.0"
RESEARCH_STEP_ID = "S19"
LOOP_ID = "S19-L06"
ROOT_SEED_HEX = "c44296c170a0aeebcf29c476f415e24788e01da88e2442b66c97f856251c79df"
CANDIDATE_IDS = ("S12F-CANDIDATE-02", "S12F-CANDIDATE-03")
COMPARATOR_LABEL_ID = "MOL_ADJACENT_INCOMING_H900"
STRUCTURAL_LABEL_ID = "PF_PAST_ONLY_MULTIATTRACTOR_BOUNDARY_RECURRENCE_H900"
H_THRESHOLD = 0.9
BOOTSTRAP_REPLICATES = 4096
PERMUTATION_REPLICATES = 4096
SUFFIX_ENDPOINT_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
SUFFIX_VARIANTS = ("DELETE", "SHUFFLE", "REPLACE")


@dataclass(frozen=True, slots=True)
class LabelDefinition:
    """One locked label specification."""

    ordinal: int
    label_id: str
    role: str
    evidence_class: str
    temporal_scope: str
    comparator_only: bool
    promotable_scope: str


LABEL_DEFINITIONS = (
    LabelDefinition(
        1,
        COMPARATOR_LABEL_ID,
        "FROZEN_ADJACENT_MOLECULAR_COMPARATOR",
        "FROZEN_S13Y_SOURCE_TRANSPLANT_COMPARATOR",
        "LOCAL_INCOMING_MOLECULAR_SIMILARITY",
        True,
        "NONE_COMPARATOR",
    ),
    LabelDefinition(
        2,
        STRUCTURAL_LABEL_ID,
        "PAST_ONLY_MULTIATTRACTOR_POST_FISSION_BOUNDARY_RECURRENCE",
        "HUMAN_LOCKED_PAPER_AND_SOURCE_INFORMED_BOUNDARY_RECONSTRUCTION",
        "PAST_ONLY_BOUNDARY_ONLINE_NO_BACKFILL",
        False,
        "RETROSPECTIVE_PAPER_FACING_ONLY_PENDING_UNTOUCHED_CONFIRMATION",
    ),
)
LABEL_BY_ID = {item.label_id: item for item in LABEL_DEFINITIONS}


def seed_material(*identity: object) -> bytes:
    return "\x1f".join([VERSION, ROOT_SEED_HEX, *map(str, identity)]).encode()


def derive_seed128(*identity: object) -> int:
    return int.from_bytes(hashlib.sha256(seed_material(*identity)).digest()[:16], "big")


def compositions(states: NDArray[np.integer[Any]]) -> NDArray[np.float64]:
    values = np.asarray(states, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 100 or np.any(values < 0):
        raise ValueError("states must be nonnegative observations-by-100 counts")
    masses = values.sum(axis=1)
    if np.any(masses <= 0):
        raise ValueError("selected molecular clock contains an empty state")
    return values / masses[:, None]


def cosine_matrix(values: NDArray[np.float64]) -> NDArray[np.float64]:
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms <= 0) or not np.all(np.isfinite(norms)):
        raise ValueError("nonpositive or nonfinite composition norm")
    unit = values / norms[:, None]
    result = np.clip(unit @ unit.T, -1.0, 1.0)
    return (result + result.T) * 0.5


def _validated_clock(
    states: NDArray[np.integer[Any]],
    generations: NDArray[np.integer[Any]],
    observation_kinds: NDArray[np.str_],
    sequence_indices: NDArray[np.integer[Any]],
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.str_], NDArray[np.int64]]:
    values = np.asarray(states, dtype=np.int64)
    generation_array = np.asarray(generations, dtype=np.int64)
    kinds = np.asarray(observation_kinds, dtype=str)
    indices = np.asarray(sequence_indices, dtype=np.int64)
    if not (len(values) == len(generation_array) == len(kinds) == len(indices)):
        raise ValueError("clock arrays must align")
    if not np.array_equal(indices, np.arange(len(indices), dtype=np.int64)):
        raise ValueError("selected sequence indices must be complete and ordered")
    if len(indices) < 4 or generation_array[0] != 0:
        raise ValueError("one leading generation-zero row is required")
    if kinds[0] != "initial_selected_state":
        raise ValueError("the selected clock must begin with its initial state")
    boundary_indices = np.flatnonzero(kinds == "post_fission")
    boundary_generations = generation_array[boundary_indices]
    if len(boundary_indices) == 0 or not np.array_equal(
        boundary_generations,
        np.arange(1, len(boundary_indices) + 1, dtype=np.int64),
    ):
        raise ValueError("post-fission boundaries must be exactly ordered generations 1..G")
    if np.any(np.diff(boundary_indices) <= 0):
        raise ValueError("post-fission boundary indices must increase")
    return values, generation_array, kinds, indices


def boundary_recurrence(
    states: NDArray[np.integer[Any]],
    generations: NDArray[np.integer[Any]],
    observation_kinds: NDArray[np.str_],
    sequence_indices: NDArray[np.integer[Any]],
    *,
    query_stop: int | None = None,
) -> dict[str, Any]:
    """Apply the single locked boundary rule and prospective projection.

    Boundary ``b_g`` is active iff it has strict cosine ``H>0.9`` to at least
    one already observed boundary ``b_h`` with ``0<h<=g-2``.  Its decision is
    assigned to ``b_g`` and subsequent selected observations up to, but not
    including, ``b_(g+1)``.  Rows before ``b_1`` are eligible negative; the
    initial generation-zero row alone is ineligible.
    """

    values, generation_array, kinds, indices = _validated_clock(
        states, generations, observation_kinds, sequence_indices
    )
    stop = len(indices) - 1 if query_stop is None else int(query_stop)
    if stop < 2 or stop >= len(indices):
        raise ValueError("query_stop must be between 2 and the final selected index")
    values = values[: stop + 1]
    generation_array = generation_array[: stop + 1]
    kinds = kinds[: stop + 1]
    indices = indices[: stop + 1]

    boundary_indices = np.flatnonzero(kinds == "post_fission")
    boundary_generations = generation_array[boundary_indices]
    boundary_states = values[boundary_indices]
    boundary_similarity = (
        cosine_matrix(compositions(boundary_states))
        if len(boundary_states)
        else np.empty((0, 0), dtype=np.float64)
    )
    boundary_labels = np.zeros(len(boundary_indices), dtype=bool)
    boundary_scores = np.full(len(boundary_indices), np.nan, dtype=np.float64)
    boundary_distinct_counts = np.zeros(len(boundary_indices), dtype=np.int64)
    boundary_qualifying_counts = np.zeros(len(boundary_indices), dtype=np.int64)
    boundary_first_match = np.full(len(boundary_indices), -1, dtype=np.int64)
    boundary_last_match = np.full(len(boundary_indices), -1, dtype=np.int64)
    boundary_matching_generations: list[tuple[int, ...]] = [
        () for _ in range(len(boundary_indices))
    ]

    for query_ordinal, generation in enumerate(boundary_generations):
        allowed = np.flatnonzero(
            (boundary_generations > 0) & (boundary_generations <= int(generation) - 2)
        )
        if not len(allowed):
            continue
        scores = boundary_similarity[query_ordinal, allowed]
        boundary_scores[query_ordinal] = float(np.max(scores))
        matched = allowed[scores > H_THRESHOLD]
        if len(matched):
            matched_generations = tuple(int(boundary_generations[i]) for i in matched)
            boundary_labels[query_ordinal] = True
            boundary_matching_generations[query_ordinal] = matched_generations
            boundary_qualifying_counts[query_ordinal] = len(matched_generations)
            boundary_distinct_counts[query_ordinal] = len(matched_generations)
            boundary_first_match[query_ordinal] = matched_generations[0]
            boundary_last_match[query_ordinal] = matched_generations[-1]

    count = len(indices)
    labels = np.zeros(count, dtype=bool)
    scores = np.full(count, np.nan, dtype=np.float64)
    distinct = np.zeros(count, dtype=np.int64)
    qualifying = np.zeros(count, dtype=np.int64)
    first_match = np.full(count, -1, dtype=np.int64)
    last_match = np.full(count, -1, dtype=np.int64)
    source_boundary = np.zeros(count, dtype=np.int64)
    matching_generations: list[tuple[int, ...]] = [() for _ in range(count)]

    boundary_ordinal_by_generation = {
        int(generation): ordinal
        for ordinal, generation in enumerate(boundary_generations)
    }
    for t in range(1, count):
        generation = int(generation_array[t])
        if kinds[t] == "post_fission":
            source_generation = generation
        else:
            source_generation = generation - 1
        source_boundary[t] = max(source_generation, 0)
        ordinal = boundary_ordinal_by_generation.get(source_generation)
        if ordinal is None:
            continue
        labels[t] = bool(boundary_labels[ordinal])
        scores[t] = boundary_scores[ordinal]
        distinct[t] = boundary_distinct_counts[ordinal]
        qualifying[t] = boundary_qualifying_counts[ordinal]
        first_match[t] = boundary_first_match[ordinal]
        last_match[t] = boundary_last_match[ordinal]
        matching_generations[t] = boundary_matching_generations[ordinal]

    if labels[0] or source_boundary[0] != 0:
        raise RuntimeError("generation-zero initial row changed")
    if np.any(boundary_labels[:2]):
        raise RuntimeError("the first two boundaries cannot recur under h<=g-2")
    if not np.array_equal(labels[boundary_indices], boundary_labels):
        raise RuntimeError("boundary projection changed its own boundary decision")
    for ordinal, generation in enumerate(boundary_generations):
        matched = boundary_matching_generations[ordinal]
        if any(not (0 < h <= int(generation) - 2) for h in matched):
            raise RuntimeError("a boundary used an ineligible prior generation")

    eligible = np.arange(count) > 0
    activated = np.flatnonzero(boundary_labels)
    diagnostics = {
        "eligibleCount": int(np.count_nonzero(eligible)),
        "positiveCount": int(np.count_nonzero(labels)),
        "boundaryCount": len(boundary_indices),
        "activatedBoundaryCount": int(np.count_nonzero(boundary_labels)),
        "activatedBoundaryFraction": float(np.mean(boundary_labels))
        if len(boundary_labels)
        else 0.0,
        "firstActivatedBoundaryGeneration": int(boundary_generations[activated[0]])
        if len(activated)
        else None,
        "meanDistinctPriorBoundaryCount": float(np.mean(boundary_distinct_counts)),
        "medianDistinctPriorBoundaryCount": float(np.median(boundary_distinct_counts)),
        "maxDistinctPriorBoundaryCount": int(np.max(boundary_distinct_counts))
        if len(boundary_distinct_counts)
        else 0,
        "meanDistinctPriorBoundaryCountPositive": float(
            np.mean(boundary_distinct_counts[boundary_labels])
        )
        if np.any(boundary_labels)
        else None,
        "preFirstBoundaryEligibleNegativeCount": int(boundary_indices[0] - 1)
        if len(boundary_indices)
        else count - 1,
        "futureBoundaryCount": 0,
        "futureSuffixIndependentByConstruction": True,
        "backfillApplied": False,
        "carryAcrossNextBoundaryApplied": False,
    }
    return {
        "labels": labels,
        "scores": scores,
        "distinctPriorBoundaryCount": distinct,
        "qualifyingPriorBoundaryCount": qualifying,
        "firstMatchingBoundaryGeneration": first_match,
        "lastMatchingBoundaryGeneration": last_match,
        "sourceBoundaryGeneration": source_boundary,
        "matchingBoundaryGenerations": tuple(matching_generations),
        "boundaryIndices": boundary_indices.astype(np.int64),
        "boundaryGenerations": boundary_generations.astype(np.int64),
        "boundaryLabels": boundary_labels,
        "boundaryScores": boundary_scores,
        "boundaryDistinctPriorCount": boundary_distinct_counts,
        "boundaryQualifyingPriorCount": boundary_qualifying_counts,
        "boundaryMatchingGenerations": tuple(boundary_matching_generations),
        "diagnostics": diagnostics,
    }


def boundary_recurrence_reference(
    states: NDArray[np.integer[Any]],
    generations: NDArray[np.integer[Any]],
    observation_kinds: NDArray[np.str_],
    sequence_indices: NDArray[np.integer[Any]],
) -> dict[str, Any]:
    """Independent scalar-loop replay of boundary decisions and projection."""

    values, generation_array, kinds, indices = _validated_clock(
        states, generations, observation_kinds, sequence_indices
    )
    boundary_indices = [i for i, kind in enumerate(kinds) if kind == "post_fission"]
    boundary_compositions = compositions(values[np.asarray(boundary_indices)])
    labels = np.zeros(len(indices), dtype=bool)
    scores = np.full(len(indices), np.nan, dtype=np.float64)
    distinct = np.zeros(len(indices), dtype=np.int64)
    qualifying = np.zeros(len(indices), dtype=np.int64)
    first_match = np.full(len(indices), -1, dtype=np.int64)
    last_match = np.full(len(indices), -1, dtype=np.int64)
    source_boundary = np.zeros(len(indices), dtype=np.int64)
    matching: list[tuple[int, ...]] = [() for _ in range(len(indices))]
    decisions: dict[int, tuple[bool, float, tuple[int, ...]]] = {}
    for ordinal, boundary_index in enumerate(boundary_indices):
        generation = int(generation_array[boundary_index])
        allowed = [h for h in range(1, generation - 1)]
        similarities = []
        for h in allowed:
            reference = h - 1
            left = boundary_compositions[ordinal]
            right = boundary_compositions[reference]
            similarities.append(float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right))))
        score = max(similarities) if similarities else np.nan
        matched = tuple(
            h for h, similarity in zip(allowed, similarities, strict=True)
            if similarity > H_THRESHOLD
        )
        decisions[generation] = (bool(matched), score, matched)
    for t in range(1, len(indices)):
        generation = int(generation_array[t])
        source = generation if kinds[t] == "post_fission" else generation - 1
        source_boundary[t] = max(source, 0)
        if source not in decisions:
            continue
        label, score, matched = decisions[source]
        labels[t] = label
        scores[t] = score
        matching[t] = matched
        distinct[t] = len(matched)
        qualifying[t] = len(matched)
        if matched:
            first_match[t] = matched[0]
            last_match[t] = matched[-1]
    return {
        "labels": labels,
        "scores": scores,
        "distinctPriorBoundaryCount": distinct,
        "qualifyingPriorBoundaryCount": qualifying,
        "firstMatchingBoundaryGeneration": first_match,
        "lastMatchingBoundaryGeneration": last_match,
        "sourceBoundaryGeneration": source_boundary,
        "matchingBoundaryGenerations": tuple(matching),
    }


def suffix_endpoint_indices(total_clock_count: int) -> tuple[int, ...]:
    if total_clock_count < 6:
        raise ValueError("selected clock is too short for five suffix sentinels")
    endpoints = {
        min(total_clock_count - 2, max(2, int(np.floor(q * (total_clock_count - 1)))))
        for q in SUFFIX_ENDPOINT_QUANTILES
    }
    if len(endpoints) != len(SUFFIX_ENDPOINT_QUANTILES):
        raise ValueError("suffix endpoint quantiles collapsed")
    return tuple(sorted(endpoints))


def label_trajectory(
    trajectory: Any,
    definition: LabelDefinition,
    *,
    clock_id: str = "C1_SELECTED_DAUGHTER_RETAINED",
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    selected = selected_clock_observations(trajectory, clock_id)
    if definition.comparator_only:
        raw, diagnostic = frozen_label_trajectory(
            trajectory, fixed_label_spec(COMPARATOR_LABEL_ID), clock_id=clock_id
        )
        output = raw.copy()
        output["labelFamily"] = definition.role
        output["labelEvidenceTier"] = definition.evidence_class
        output["temporalScope"] = definition.temporal_scope
        output["labelStatus"] = "ELIGIBLE"
        output["ineligibilityReason"] = None
        for column in (
            "distinctPriorBoundaryCount",
            "qualifyingPriorBoundaryCount",
            "firstMatchingBoundaryGeneration",
            "lastMatchingBoundaryGeneration",
            "sourceBoundaryGeneration",
            "isPostFissionBoundary",
            "isActivatedBoundary",
        ):
            output[column] = None
        return output, {"comparatorOnly": True, **diagnostic}, pd.DataFrame()

    states = np.asarray([item.state for item in selected], dtype=np.int64)
    generations = np.asarray(
        [int(item.growth_generation_one_based) for item in selected], dtype=np.int64
    )
    kinds = np.asarray([str(item.observation_kind) for item in selected], dtype=str)
    indices = np.arange(len(selected), dtype=np.int64)
    result = boundary_recurrence(states, generations, kinds, indices)
    rows = []
    for index, item in enumerate(selected):
        eligible = index > 0
        rows.append(
            {
                "candidateId": str(trajectory.configuration_id),
                "trajectoryId": str(trajectory.trajectory_id),
                "matrixIndex": int(trajectory.matrix_index),
                "labelId": definition.label_id,
                "labelFamily": definition.role,
                "labelEvidenceTier": definition.evidence_class,
                "temporalScope": definition.temporal_scope,
                "selectedSequenceIndex": index,
                "rawObservationIndex": int(item.observation_index),
                "generation": int(generations[index]),
                "observationKind": kinds[index],
                "isReplicator": bool(result["labels"][index]) if eligible else None,
                "labelScore": float(result["scores"][index])
                if eligible and np.isfinite(result["scores"][index])
                else None,
                "labelStatus": "ELIGIBLE" if eligible else "INELIGIBLE",
                "ineligibilityReason": None if eligible else "GENERATION_ZERO_INITIAL_STATE",
                "distinctPriorBoundaryCount": int(result["distinctPriorBoundaryCount"][index])
                if eligible else None,
                "qualifyingPriorBoundaryCount": int(result["qualifyingPriorBoundaryCount"][index])
                if eligible else None,
                "firstMatchingBoundaryGeneration": int(result["firstMatchingBoundaryGeneration"][index])
                if eligible and result["firstMatchingBoundaryGeneration"][index] >= 0 else None,
                "lastMatchingBoundaryGeneration": int(result["lastMatchingBoundaryGeneration"][index])
                if eligible and result["lastMatchingBoundaryGeneration"][index] >= 0 else None,
                "sourceBoundaryGeneration": int(result["sourceBoundaryGeneration"][index])
                if eligible else None,
                "isPostFissionBoundary": bool(kinds[index] == "post_fission")
                if eligible else None,
                "isActivatedBoundary": bool(result["labels"][index])
                if eligible and kinds[index] == "post_fission" else None,
            }
        )
    boundary_rows = []
    for ordinal, selected_index in enumerate(result["boundaryIndices"]):
        boundary_rows.append(
            {
                "candidateId": str(trajectory.configuration_id),
                "trajectoryId": str(trajectory.trajectory_id),
                "matrixIndex": int(trajectory.matrix_index),
                "labelId": definition.label_id,
                "boundaryOrdinal": ordinal,
                "boundaryGeneration": int(result["boundaryGenerations"][ordinal]),
                "selectedSequenceIndex": int(selected_index),
                "rawObservationIndex": int(selected[int(selected_index)].observation_index),
                "observationKind": "post_fission",
                "isActivatedBoundary": bool(result["boundaryLabels"][ordinal]),
                "maximumPriorBoundarySimilarity": float(result["boundaryScores"][ordinal])
                if np.isfinite(result["boundaryScores"][ordinal]) else None,
                "distinctPriorBoundaryCount": int(result["boundaryDistinctPriorCount"][ordinal]),
                "qualifyingPriorBoundaryCount": int(result["boundaryQualifyingPriorCount"][ordinal]),
                "matchingBoundaryGenerations": list(result["boundaryMatchingGenerations"][ordinal]),
            }
        )
    return pd.DataFrame(rows), result["diagnostics"], pd.DataFrame(boundary_rows)


def recomputed_generation_block_metrics(
    states: NDArray[np.integer[Any]],
    generations: NDArray[np.integer[Any]],
    observation_kinds: NDArray[np.str_],
    orders: NDArray[np.integer[Any]],
) -> dict[str, NDArray[np.float64]]:
    """Recompute boundary recurrence after whole growth-fission block shuffles."""

    values, generation_array, kinds, _ = _validated_clock(
        states,
        generations,
        observation_kinds,
        np.arange(len(states), dtype=np.int64),
    )
    observed = np.unique(generation_array[generation_array > 0])
    generation_count = len(observed)
    permutations = np.asarray(orders, dtype=np.int64)
    expected = np.arange(generation_count, dtype=np.int64)
    if permutations.ndim != 2 or permutations.shape[1] != generation_count:
        raise ValueError("orders must be replicates-by-generation_count")
    if not np.array_equal(
        np.sort(permutations, axis=1), np.broadcast_to(expected, permutations.shape)
    ):
        raise ValueError("every generation order must be a permutation")
    blocks = [np.flatnonzero(generation_array == generation) for generation in observed]
    if any(kinds[block[-1]] != "post_fission" for block in blocks):
        raise ValueError("each generation block must end at its post-fission boundary")
    boundary_indices = np.asarray([block[-1] for block in blocks], dtype=np.int64)
    boundary_similarity = cosine_matrix(compositions(values[boundary_indices]))
    replicate_count = len(permutations)
    activations = np.zeros((replicate_count, generation_count), dtype=bool)
    for position in range(2, generation_count):
        current = permutations[:, position]
        prior = permutations[:, : position - 1]
        activations[:, position] = (
            boundary_similarity[current[:, None], prior] > H_THRESHOLD
        ).any(axis=1)

    eligible_count = int(sum(len(block) for block in blocks))
    sequences = np.zeros((replicate_count, eligible_count), dtype=bool)
    for replicate in range(replicate_count):
        cursor = 0
        previous = False
        for position, original_ordinal in enumerate(permutations[replicate]):
            block_length = len(blocks[int(original_ordinal)])
            update_count = block_length - 1
            if update_count:
                sequences[replicate, cursor : cursor + update_count] = previous
            cursor += update_count
            current = bool(activations[replicate, position])
            sequences[replicate, cursor] = current
            cursor += 1
            previous = current
        if cursor != eligible_count:
            raise RuntimeError("permuted generation blocks changed clock cardinality")

    persistence = sequences.sum(axis=1).astype(np.float64)
    occupancy = persistence / eligible_count
    left = sequences[:, :-1]
    right = sequences[:, 1:]
    n = left.shape[1]
    mean_left = left.mean(axis=1)
    mean_right = right.mean(axis=1)
    covariance = (left & right).sum(axis=1) / n - mean_left * mean_right
    denominator = np.sqrt(
        mean_left * (1.0 - mean_left) * mean_right * (1.0 - mean_right)
    )
    consistency = np.full(replicate_count, np.nan, dtype=np.float64)
    valid = denominator > 0
    consistency[valid] = covariance[valid] / denominator[valid]
    has_positive = sequences.any(axis=1)
    onset_eligible = np.argmax(sequences, axis=1).astype(np.float64)
    onset = np.where(has_positive, onset_eligible + 1.0, eligible_count + 1.0)
    normalized = np.where(has_positive, onset / eligible_count, 1.0)
    return {
        "persistence": persistence,
        "occupancy": occupancy,
        "consistency": consistency,
        "firstOnsetRawScore": onset,
        "firstOnsetNormalizedScore": normalized,
    }
