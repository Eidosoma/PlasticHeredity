"""Outcome-blind contracts for E01/S19 Loop 5.

This module fixes one past-only cross-generation recurrence label and the
frozen adjacent-H comparator.  It contains no emergence, prediction,
intervention, simulation, or scientific filesystem logic.
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

VERSION = "E01-S19-L05-PAST-ONLY-CROSS-GENERATION-RECURRENCE-ACTIVATION-v1.0.0"
RESEARCH_STEP_ID = "S19"
LOOP_ID = "S19-L05"
ROOT_SEED_HEX = "3e0a417fd0f71df069160030b8aa2ae8593227eb14ca83df21886346c52eefd6"
CANDIDATE_IDS = ("S12F-CANDIDATE-02", "S12F-CANDIDATE-03")
COMPARATOR_LABEL_ID = "MOL_ADJACENT_INCOMING_H900"
STRUCTURAL_LABEL_ID = "MOL_PAST_ONLY_CROSS_GENERATION_RECURRENCE_H900"
L04_REFERENCE_LABEL_ID = "MOL_CROSS_GENERATION_RECURRENCE_H900"
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
        "PAST_ONLY_CROSS_GENERATION_RECURRENCE_ACTIVATION",
        "HUMAN_LOCKED_DIRECT_FALSIFICATION_OF_L04_FUTURE_DEPENDENCE",
        "PAST_ONLY_PREFIX_CAUSAL_NO_BACKFILL",
        False,
        "EXPLORATORY_PAPER_FACING_AND_ONLINE_LABEL_CONSTRUCTION_ONLY",
    ),
)
LABEL_BY_ID = {item.label_id: item for item in LABEL_DEFINITIONS}


def seed_material(*identity: object) -> bytes:
    return "\x1f".join([VERSION, ROOT_SEED_HEX, *map(str, identity)]).encode()


def derive_seed128(*identity: object) -> int:
    return int.from_bytes(hashlib.sha256(seed_material(*identity)).digest()[:16], "big")


def compositions(states: NDArray[np.integer[Any]]) -> NDArray[np.float64]:
    """Return L1-closed 100-component compositions."""

    values = np.asarray(states, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 100 or np.any(values < 0):
        raise ValueError("states must be nonnegative observations-by-100 counts")
    mass = values.sum(axis=1)
    if np.any(mass <= 0):
        raise ValueError("selected molecular clock contains an empty state")
    return values / mass[:, None]


def cosine_matrix(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Historical cosine-H similarities, explicitly symmetrized."""

    norms = np.linalg.norm(values, axis=1)
    if np.any(norms <= 0) or not np.all(np.isfinite(norms)):
        raise ValueError("nonpositive or nonfinite composition norm")
    unit = values / norms[:, None]
    result = np.clip(unit @ unit.T, -1.0, 1.0)
    return (result + result.T) * 0.5


def _validated_clock(
    states: NDArray[np.integer[Any]],
    generations: NDArray[np.integer[Any]],
    sequence_indices: NDArray[np.integer[Any]],
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    values = np.asarray(states, dtype=np.int64)
    generation_array = np.asarray(generations, dtype=np.int64)
    indices = np.asarray(sequence_indices, dtype=np.int64)
    if len(values) != len(generation_array) or len(values) != len(indices):
        raise ValueError("states, generations, and sequence indices must align")
    if not np.array_equal(indices, np.arange(len(indices), dtype=np.int64)):
        raise ValueError("selected sequence indices must be complete and ordered")
    if len(indices) < 3:
        raise ValueError("trajectory is too short for past-only recurrence")
    positive = generation_array > 0
    if np.count_nonzero(~positive) != 1 or generation_array[0] != 0:
        raise ValueError("exactly one leading generation-zero row is required")
    observed = np.unique(generation_array[positive])
    if not np.array_equal(observed, np.arange(1, int(observed[-1]) + 1)):
        raise ValueError("positive-numbered generations must be consecutive")
    return values, generation_array, indices


def past_only_recurrence(
    states: NDArray[np.integer[Any]],
    generations: NDArray[np.integer[Any]],
    sequence_indices: NDArray[np.integer[Any]],
    *,
    query_stop: int | None = None,
) -> dict[str, Any]:
    """Apply the single locked one-sided recurrence-activation rule.

    For a query at index ``t`` in generation ``g``, only references ``s <= t-2``
    with ``0 < g_s < g`` are eligible.  No future state, backfill, or carried
    persistence can change a label already assigned.
    """

    values, generation_array, indices = _validated_clock(
        states, generations, sequence_indices
    )
    stop = len(indices) - 1 if query_stop is None else int(query_stop)
    if stop < 2 or stop >= len(indices):
        raise ValueError("query_stop must be between 2 and the final selected index")
    values = values[: stop + 1]
    generation_array = generation_array[: stop + 1]
    indices = indices[: stop + 1]
    similarity = cosine_matrix(compositions(values))
    count = len(indices)

    labels = np.zeros(count, dtype=bool)
    scores = np.full(count, np.nan, dtype=np.float64)
    distinct_generations = np.zeros(count, dtype=np.int64)
    qualifying_states = np.zeros(count, dtype=np.int64)
    immediate_match = np.zeros(count, dtype=np.int64)
    immediate_similarity = np.full(count, np.nan, dtype=np.float64)
    same_generation_prior = np.zeros(count, dtype=np.int64)
    first_generation = np.full(count, -1, dtype=np.int64)
    last_generation = np.full(count, -1, dtype=np.int64)
    earliest_sequence = np.full(count, -1, dtype=np.int64)
    latest_sequence = np.full(count, -1, dtype=np.int64)
    matching_reference_indices: list[tuple[int, ...]] = [() for _ in range(count)]

    for t in range(1, count):
        generation = int(generation_array[t])
        if generation <= 0:
            continue
        if t >= 2:
            references = np.arange(t - 1, dtype=np.int64)
            reference_generations = generation_array[references]
            eligible = (reference_generations > 0) & (
                reference_generations < generation
            )
            eligible_indices = references[eligible]
            if len(eligible_indices):
                candidate_scores = similarity[t, eligible_indices]
                scores[t] = float(np.max(candidate_scores))
                matched_indices = eligible_indices[candidate_scores > H_THRESHOLD]
                if len(matched_indices):
                    labels[t] = True
                    matching_reference_indices[t] = tuple(
                        int(value) for value in matched_indices
                    )
                    qualifying_states[t] = len(matched_indices)
                    matched_generations = np.unique(generation_array[matched_indices])
                    distinct_generations[t] = len(matched_generations)
                    first_generation[t] = int(matched_generations[0])
                    last_generation[t] = int(matched_generations[-1])
                    earliest_sequence[t] = int(matched_indices[0])
                    latest_sequence[t] = int(matched_indices[-1])
        if t >= 1 and 0 < generation_array[t - 1] < generation:
            immediate_similarity[t] = float(similarity[t, t - 1])
            immediate_match[t] = int(similarity[t, t - 1] > H_THRESHOLD)
        same_generation_indices = np.flatnonzero(
            (generation_array[:t] == generation) & (similarity[t, :t] > H_THRESHOLD)
        )
        same_generation_prior[t] = len(same_generation_indices)

    eligible_rows = generation_array > 0
    immediate_only = (immediate_match > 0) & ~labels
    if not np.array_equal(labels, distinct_generations > 0):
        raise RuntimeError("past-only labels and distinct earlier generations disagree")
    if np.any(earliest_sequence[labels] > indices[labels] - 2):
        raise RuntimeError("a past-only match violated the nonadjacent index rule")
    if np.any(last_generation[labels] >= generation_array[labels]):
        raise RuntimeError("a past-only match did not come from an earlier generation")

    diagnostics = {
        "eligibleCount": int(np.count_nonzero(eligible_rows)),
        "positiveCount": int(np.count_nonzero(labels)),
        "meanDistinctEarlierGenerationCount": float(
            np.mean(distinct_generations[eligible_rows])
        ),
        "medianDistinctEarlierGenerationCount": float(
            np.median(distinct_generations[eligible_rows])
        ),
        "maxDistinctEarlierGenerationCount": int(np.max(distinct_generations)),
        "fractionEligibleWithAtLeast2EarlierGenerations": float(
            np.mean(distinct_generations[eligible_rows] >= 2)
        ),
        "fractionEligibleWithAtLeast5EarlierGenerations": float(
            np.mean(distinct_generations[eligible_rows] >= 5)
        ),
        "fractionEligibleWithAtLeast10EarlierGenerations": float(
            np.mean(distinct_generations[eligible_rows] >= 10)
        ),
        "immediateOnlyEvidenceCount": int(np.count_nonzero(immediate_only)),
        "sameGenerationPriorMatchCount": int(np.count_nonzero(same_generation_prior)),
        "futureReferenceCount": 0,
        "futureSuffixIndependentByConstruction": True,
        "backfillApplied": False,
        "carryForwardApplied": False,
    }
    return {
        "labels": labels,
        "scores": scores,
        "distinctEarlierGenerationCount": distinct_generations,
        "qualifyingPriorStateCount": qualifying_states,
        "immediatePriorCrossGenerationMatchCount": immediate_match,
        "immediateOnlyEvidence": immediate_only,
        "maximumImmediatePriorSimilarity": immediate_similarity,
        "sameGenerationPriorMatchCount": same_generation_prior,
        "firstMatchingGeneration": first_generation,
        "lastMatchingGeneration": last_generation,
        "earliestMatchingSequenceIndex": earliest_sequence,
        "latestMatchingSequenceIndex": latest_sequence,
        "matchingReferenceIndices": tuple(matching_reference_indices),
        "diagnostics": diagnostics,
    }


def past_only_recurrence_reference(
    states: NDArray[np.integer[Any]],
    generations: NDArray[np.integer[Any]],
    sequence_indices: NDArray[np.integer[Any]],
) -> dict[str, NDArray[Any]]:
    """Independent row-loop replay of the locked label and match identities."""

    values, generation_array, indices = _validated_clock(
        states, generations, sequence_indices
    )
    similarity = cosine_matrix(compositions(values))
    labels = np.zeros(len(indices), dtype=bool)
    scores = np.full(len(indices), np.nan, dtype=np.float64)
    distinct = np.zeros(len(indices), dtype=np.int64)
    qualifying = np.zeros(len(indices), dtype=np.int64)
    earliest = np.full(len(indices), -1, dtype=np.int64)
    latest = np.full(len(indices), -1, dtype=np.int64)
    matching_reference_indices: list[tuple[int, ...]] = [
        () for _ in range(len(indices))
    ]
    for t in range(2, len(indices)):
        generation = int(generation_array[t])
        if generation <= 0:
            continue
        allowed = [s for s in range(t - 1) if 0 < int(generation_array[s]) < generation]
        if not allowed:
            continue
        allowed_array = np.asarray(allowed, dtype=np.int64)
        values_t = similarity[t, allowed_array]
        scores[t] = float(np.max(values_t))
        matched = allowed_array[values_t > H_THRESHOLD]
        if len(matched):
            labels[t] = True
            matching_reference_indices[t] = tuple(int(value) for value in matched)
            qualifying[t] = len(matched)
            distinct[t] = len(np.unique(generation_array[matched]))
            earliest[t] = int(matched[0])
            latest[t] = int(matched[-1])
    return {
        "labels": labels,
        "scores": scores,
        "distinctEarlierGenerationCount": distinct,
        "qualifyingPriorStateCount": qualifying,
        "earliestMatchingSequenceIndex": earliest,
        "latestMatchingSequenceIndex": latest,
        "matchingReferenceIndices": tuple(matching_reference_indices),
    }


def suffix_endpoint_indices(total_clock_count: int) -> tuple[int, ...]:
    """Five frozen interior endpoints based only on selected-clock length."""

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
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Materialize the frozen comparator or the one L05 structural label."""

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
            "distinctEarlierGenerationCount",
            "qualifyingPriorStateCount",
            "immediatePriorCrossGenerationMatchCount",
            "immediateOnlyEvidence",
            "maximumImmediatePriorSimilarity",
            "sameGenerationPriorMatchCount",
            "firstMatchingGeneration",
            "lastMatchingGeneration",
            "earliestMatchingSequenceIndex",
            "latestMatchingSequenceIndex",
        ):
            output[column] = None
        return output, {"comparatorOnly": True, **diagnostic}

    states = np.asarray([item.state for item in selected], dtype=np.int64)
    generations = np.asarray(
        [int(item.growth_generation_one_based) for item in selected], dtype=np.int64
    )
    indices = np.arange(len(selected), dtype=np.int64)
    result = past_only_recurrence(states, generations, indices)
    rows = []
    for index, item in enumerate(selected):
        eligible = generations[index] > 0
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
                "observationKind": str(item.observation_kind),
                "isReplicator": bool(result["labels"][index]) if eligible else None,
                "labelScore": float(result["scores"][index]) if eligible else None,
                "labelStatus": "ELIGIBLE" if eligible else "INELIGIBLE",
                "ineligibilityReason": None
                if eligible
                else "PRE_FIRST_FISSION_GENERATION_ZERO",
                "distinctEarlierGenerationCount": int(
                    result["distinctEarlierGenerationCount"][index]
                )
                if eligible
                else None,
                "qualifyingPriorStateCount": int(
                    result["qualifyingPriorStateCount"][index]
                )
                if eligible
                else None,
                "immediatePriorCrossGenerationMatchCount": int(
                    result["immediatePriorCrossGenerationMatchCount"][index]
                )
                if eligible
                else None,
                "immediateOnlyEvidence": bool(result["immediateOnlyEvidence"][index])
                if eligible
                else None,
                "maximumImmediatePriorSimilarity": float(
                    result["maximumImmediatePriorSimilarity"][index]
                )
                if eligible
                and np.isfinite(result["maximumImmediatePriorSimilarity"][index])
                else None,
                "sameGenerationPriorMatchCount": int(
                    result["sameGenerationPriorMatchCount"][index]
                )
                if eligible
                else None,
                "firstMatchingGeneration": int(result["firstMatchingGeneration"][index])
                if eligible and result["firstMatchingGeneration"][index] >= 0
                else None,
                "lastMatchingGeneration": int(result["lastMatchingGeneration"][index])
                if eligible and result["lastMatchingGeneration"][index] >= 0
                else None,
                "earliestMatchingSequenceIndex": int(
                    result["earliestMatchingSequenceIndex"][index]
                )
                if eligible and result["earliestMatchingSequenceIndex"][index] >= 0
                else None,
                "latestMatchingSequenceIndex": int(
                    result["latestMatchingSequenceIndex"][index]
                )
                if eligible and result["latestMatchingSequenceIndex"][index] >= 0
                else None,
            }
        )
    return pd.DataFrame(rows), result["diagnostics"]


def binary_consistency(labels: NDArray[np.bool_]) -> NDArray[np.float64]:
    """Vectorized consecutive-label Pearson correlation by row."""

    values = np.asarray(labels, dtype=bool)
    left = values[:, :-1]
    right = values[:, 1:]
    n = left.shape[1]
    mean_left = left.mean(axis=1)
    mean_right = right.mean(axis=1)
    covariance = (left & right).sum(axis=1) / n - mean_left * mean_right
    denominator = np.sqrt(
        mean_left * (1.0 - mean_left) * mean_right * (1.0 - mean_right)
    )
    output = np.full(len(values), np.nan, dtype=np.float64)
    valid = denominator > 0
    output[valid] = covariance[valid] / denominator[valid]
    return output


def recomputed_generation_block_metrics(
    states: NDArray[np.integer[Any]],
    generations: NDArray[np.integer[Any]],
    orders: NDArray[np.integer[Any]],
) -> dict[str, NDArray[np.float64]]:
    """Recompute past-only labels after whole-generation order permutations.

    ``orders`` contains zero-based original block ordinals.  The shuffled
    blocks are renumbered 1..G by their new order, while their internal state
    order remains unchanged.  The immediate-neighbor exclusion is reapplied
    at each newly formed block boundary.
    """

    values = np.asarray(states, dtype=np.int64)
    generation_array = np.asarray(generations, dtype=np.int64)
    if generation_array[0] != 0:
        raise ValueError("generation zero must lead the selected clock")
    observed = np.unique(generation_array[generation_array > 0])
    generation_count = len(observed)
    permutations = np.asarray(orders, dtype=np.int64)
    if permutations.ndim != 2 or permutations.shape[1] != generation_count:
        raise ValueError("orders must be replicates-by-generation_count")
    expected = np.arange(generation_count, dtype=np.int64)
    if not np.array_equal(
        np.sort(permutations, axis=1),
        np.broadcast_to(expected, permutations.shape),
    ):
        raise ValueError("every generation order must be a permutation")

    similarity = cosine_matrix(compositions(values))
    block_indices = [
        np.flatnonzero(generation_array == generation) for generation in observed
    ]
    eligible_count = int(np.count_nonzero(generation_array > 0))
    replicate_count = len(permutations)
    ranks = np.empty_like(permutations)
    ranks[np.arange(replicate_count)[:, None], permutations] = expected[None, :]
    labels = np.zeros((replicate_count, len(values)), dtype=bool)
    first_by_block = np.zeros((replicate_count, generation_count), dtype=bool)
    last_by_block = np.zeros((replicate_count, generation_count), dtype=bool)
    first_positive_by_block = np.full(
        (replicate_count, generation_count), -1, dtype=np.int32
    )
    internal_n00 = np.zeros(replicate_count, dtype=np.int64)
    internal_n01 = np.zeros(replicate_count, dtype=np.int64)
    internal_n10 = np.zeros(replicate_count, dtype=np.int64)
    internal_n11 = np.zeros(replicate_count, dtype=np.int64)

    for block_ordinal, indices in enumerate(block_indices):
        query_position = ranks[:, block_ordinal]
        other_blocks = [i for i in range(generation_count) if i != block_ordinal]
        for row_offset, state_index in enumerate(indices):
            matching_blocks = []
            only_last_match: dict[int, bool] = {}
            for reference_block in other_blocks:
                references = block_indices[reference_block]
                matched = similarity[state_index, references] > H_THRESHOLD
                if matched.any():
                    matching_blocks.append(reference_block)
                    only_last_match[reference_block] = bool(
                        matched.sum() == 1 and matched[-1]
                    )
            if not matching_blocks:
                continue
            matching_array = np.asarray(matching_blocks, dtype=np.int64)
            matching_positions = ranks[:, matching_array]
            earlier = matching_positions < query_position[:, None]
            row_labels = earlier.any(axis=1)
            if row_offset == 0:
                earlier_count = earlier.sum(axis=1)
                only_one = earlier_count == 1
                if np.any(only_one):
                    selected_column = np.argmax(earlier, axis=1)
                    selected_block = matching_array[selected_column]
                    adjacent_position = (
                        ranks[np.arange(replicate_count), selected_block]
                        == query_position - 1
                    )
                    only_last = np.asarray(
                        [only_last_match[int(block)] for block in selected_block],
                        dtype=bool,
                    )
                    row_labels &= ~(only_one & adjacent_position & only_last)
            labels[:, state_index] = row_labels

        block_labels = labels[:, indices]
        first_by_block[:, block_ordinal] = block_labels[:, 0]
        last_by_block[:, block_ordinal] = block_labels[:, -1]
        has_positive = block_labels.any(axis=1)
        first_positive = np.argmax(block_labels, axis=1).astype(np.int32)
        first_positive[~has_positive] = -1
        first_positive_by_block[:, block_ordinal] = first_positive
        if len(indices) > 1:
            left = block_labels[:, :-1]
            right = block_labels[:, 1:]
            internal_n00 += (~left & ~right).sum(axis=1)
            internal_n01 += (~left & right).sum(axis=1)
            internal_n10 += (left & ~right).sum(axis=1)
            internal_n11 += (left & right).sum(axis=1)

    persistence = labels[:, 1:].sum(axis=1).astype(np.float64)
    occupancy = persistence / eligible_count
    ordered_first = np.take_along_axis(first_by_block, permutations, axis=1)
    ordered_last = np.take_along_axis(last_by_block, permutations, axis=1)
    left = ordered_last[:, :-1]
    right = ordered_first[:, 1:]
    n00 = internal_n00 + (~left & ~right).sum(axis=1)
    n01 = internal_n01 + (~left & right).sum(axis=1)
    n10 = internal_n10 + (left & ~right).sum(axis=1)
    n11 = internal_n11 + (left & right).sum(axis=1)
    transition_count = n00 + n01 + n10 + n11
    mean_left = (n10 + n11) / transition_count
    mean_right = (n01 + n11) / transition_count
    covariance = n11 / transition_count - mean_left * mean_right
    denominator = np.sqrt(mean_left * (1 - mean_left) * mean_right * (1 - mean_right))
    consistency = np.full(replicate_count, np.nan, dtype=np.float64)
    valid = denominator > 0
    consistency[valid] = covariance[valid] / denominator[valid]

    ordered_first_positive = np.take_along_axis(
        first_positive_by_block, permutations, axis=1
    )
    block_lengths = np.asarray(
        [len(indices) for indices in block_indices], dtype=np.int64
    )
    ordered_lengths = block_lengths[permutations]
    offsets = 1 + np.cumsum(
        np.column_stack(
            [np.zeros(replicate_count, dtype=np.int64), ordered_lengths[:, :-1]]
        ),
        axis=1,
    )
    candidate_onsets = np.where(
        ordered_first_positive >= 0,
        offsets + ordered_first_positive,
        len(values),
    )
    onset = candidate_onsets.min(axis=1).astype(np.float64)
    normalized = np.where(onset < len(values), onset / eligible_count, 1.0)
    return {
        "persistence": persistence,
        "occupancy": occupancy,
        "consistency": consistency,
        "firstOnsetRawScore": onset,
        "firstOnsetNormalizedScore": normalized,
    }
