"""Outcome-blind contracts for E01/S19 Loop 4.

The module fixes exactly one structural label and one frozen comparator.  It
contains no emergence, prediction, intervention, simulation, or filesystem
logic.
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

VERSION = "E01-S19-L04-CROSS-GENERATION-RECURRENCE-MEMBERSHIP-v1.0.0"
RESEARCH_STEP_ID = "S19"
LOOP_ID = "S19-L04"
ROOT_SEED_HEX = "701f9bc9f413b89114e0d5d2e25bb1e4389c8071d8c4e8f305c65d9c75bb2a87"
CANDIDATE_IDS = ("S12F-CANDIDATE-02", "S12F-CANDIDATE-03")
COMPARATOR_LABEL_ID = "MOL_ADJACENT_INCOMING_H900"
STRUCTURAL_LABEL_ID = "MOL_CROSS_GENERATION_RECURRENCE_H900"
H_THRESHOLD = 0.9
BOOTSTRAP_REPLICATES = 4096
PERMUTATION_REPLICATES = 4096


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
        "COMPLETED_RUN_CROSS_GENERATION_RECURRENCE_MEMBERSHIP",
        "DIRECT_PAPER_ACROSS_GENERATIONS_WORDING_PLUS_HUMAN_LOCKED_RECONSTRUCTION",
        "RETROSPECTIVE_COMPLETED_RUN_SYMMETRIC_MEMBERSHIP",
        False,
        "RETROSPECTIVE_PAPER_FACING_ONLY",
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


def cross_generation_recurrence(
    states: NDArray[np.integer[Any]],
    generations: NDArray[np.integer[Any]],
    sequence_indices: NDArray[np.integer[Any]],
) -> dict[str, Any]:
    """Apply the one locked cross-generation recurrence rule.

    A positive row must have strict cosine H > 0.9 to a nonadjacent selected
    molecular-clock row from another positive-numbered fission generation.
    Generation zero is ineligible.  Matching rows from the same reference
    generation contribute only one distinct visit.
    """

    generations_array = np.asarray(generations, dtype=np.int64)
    indices = np.asarray(sequence_indices, dtype=np.int64)
    if len(states) != len(generations_array) or len(states) != len(indices):
        raise ValueError("states, generations, and sequence indices must align")
    if not np.array_equal(indices, np.arange(len(indices), dtype=np.int64)):
        raise ValueError("selected sequence indices must be complete and ordered")
    if len(indices) < 3:
        raise ValueError("trajectory is too short for cross-generation recurrence")
    positive_generation = generations_array > 0
    if np.count_nonzero(~positive_generation) != 1 or generations_array[0] != 0:
        raise ValueError("exactly one leading generation-zero row is required")
    observed_generations = np.unique(generations_array[positive_generation])
    if not np.array_equal(
        observed_generations,
        np.arange(1, int(observed_generations[-1]) + 1, dtype=np.int64),
    ):
        raise ValueError("positive-numbered generations must be consecutive")

    similarity = cosine_matrix(compositions(states))
    different_generation = generations_array[:, None] != generations_array[None, :]
    positive_pair = positive_generation[:, None] & positive_generation[None, :]
    sequence_separation = np.abs(indices[:, None] - indices[None, :])
    nonadjacent = sequence_separation > 1
    eligible_pair = positive_pair & different_generation & nonadjacent
    immediate_pair = positive_pair & different_generation & (sequence_separation == 1)
    qualifying = eligible_pair & (similarity > H_THRESHOLD)
    immediate_matches = immediate_pair & (similarity > H_THRESHOLD)
    if not np.array_equal(qualifying, qualifying.T):
        raise RuntimeError("cross-generation qualifying relation is not symmetric")

    labels = qualifying.any(axis=1)
    labels[~positive_generation] = False
    state_match_count = qualifying.sum(axis=1).astype(np.int64)
    immediate_match_count = immediate_matches.sum(axis=1).astype(np.int64)
    distinct_generation = np.zeros(len(indices), dtype=np.int64)
    first_generation = np.full(len(indices), -1, dtype=np.int64)
    last_generation = np.full(len(indices), -1, dtype=np.int64)
    earliest_sequence = np.full(len(indices), -1, dtype=np.int64)
    latest_sequence = np.full(len(indices), -1, dtype=np.int64)
    for generation in observed_generations:
        generation_matches = qualifying[:, generations_array == generation].any(axis=1)
        distinct_generation += generation_matches.astype(np.int64)
        first_generation[(first_generation < 0) & generation_matches] = int(generation)
        last_generation[generation_matches] = int(generation)
    for row_index in np.flatnonzero(labels):
        matched = np.flatnonzero(qualifying[row_index])
        earliest_sequence[row_index] = int(matched[0])
        latest_sequence[row_index] = int(matched[-1])
    if not np.array_equal(labels, distinct_generation > 0):
        raise RuntimeError("labels and distinct-generation visits disagree")

    score = np.full(len(indices), np.nan, dtype=np.float64)
    for row_index in np.flatnonzero(positive_generation):
        candidates = similarity[row_index, eligible_pair[row_index]]
        if len(candidates):
            score[row_index] = float(np.max(candidates))
    maximum_immediate = np.full(len(indices), np.nan, dtype=np.float64)
    for row_index in np.flatnonzero(positive_generation):
        candidates = similarity[row_index, immediate_pair[row_index]]
        if len(candidates):
            maximum_immediate[row_index] = float(np.max(candidates))
    immediate_only = (immediate_match_count > 0) & ~labels
    same_generation_matches = (
        positive_pair
        & (generations_array[:, None] == generations_array[None, :])
        & (sequence_separation > 0)
        & (similarity > H_THRESHOLD)
    ).sum(axis=1).astype(np.int64)

    upper = np.triu(qualifying, k=1)
    pair_rows, pair_columns = np.nonzero(upper)
    generation_pairs = {
        tuple(sorted((int(generations_array[left]), int(generations_array[right]))))
        for left, right in zip(pair_rows, pair_columns, strict=True)
    }
    possible_generation_pairs = len(observed_generations) * (len(observed_generations) - 1) // 2
    diagnostics = {
        "eligibleCount": int(np.count_nonzero(positive_generation)),
        "positiveCount": int(np.count_nonzero(labels)),
        "relationPairCount": int(np.count_nonzero(upper)),
        "distinctGenerationPairCount": len(generation_pairs),
        "possibleGenerationPairCount": int(possible_generation_pairs),
        "generationPairDensity": float(len(generation_pairs) / possible_generation_pairs),
        "recurrentGenerationCount": len(
            np.unique(generations_array[positive_generation & labels])
        ),
        "meanDistinctOtherGenerationCount": float(
            np.mean(distinct_generation[positive_generation])
        ),
        "medianDistinctOtherGenerationCount": float(
            np.median(distinct_generation[positive_generation])
        ),
        "maxDistinctOtherGenerationCount": int(np.max(distinct_generation)),
        "fractionEligibleWithAtLeast2OtherGenerations": float(
            np.mean(distinct_generation[positive_generation] >= 2)
        ),
        "fractionEligibleWithAtLeast5OtherGenerations": float(
            np.mean(distinct_generation[positive_generation] >= 5)
        ),
        "fractionEligibleWithAtLeast10OtherGenerations": float(
            np.mean(distinct_generation[positive_generation] >= 10)
        ),
        "immediateOnlyEvidenceCount": int(np.count_nonzero(immediate_only)),
        "sameGenerationOnlyOrAdditionalMatchCount": int(
            np.count_nonzero(same_generation_matches)
        ),
        "exactSymmetryPassed": True,
    }
    return {
        "labels": labels,
        "scores": score,
        "distinctOtherGenerationCount": distinct_generation,
        "qualifyingStateCount": state_match_count,
        "immediateCrossGenerationMatchCount": immediate_match_count,
        "immediateOnlyEvidence": immediate_only,
        "maximumImmediateCrossGenerationSimilarity": maximum_immediate,
        "sameGenerationMatchCount": same_generation_matches,
        "firstMatchingGeneration": first_generation,
        "lastMatchingGeneration": last_generation,
        "earliestMatchingSequenceIndex": earliest_sequence,
        "latestMatchingSequenceIndex": latest_sequence,
        "diagnostics": diagnostics,
    }


def label_trajectory(
    trajectory: Any,
    definition: LabelDefinition,
    *,
    clock_id: str = "C1_SELECTED_DAUGHTER_RETAINED",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Materialize one of the two locked labels."""

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
            "distinctOtherGenerationCount",
            "qualifyingStateCount",
            "immediateCrossGenerationMatchCount",
            "immediateOnlyEvidence",
            "maximumImmediateCrossGenerationSimilarity",
            "sameGenerationMatchCount",
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
    result = cross_generation_recurrence(states, generations, indices)
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
                "ineligibilityReason": None if eligible else "PRE_FIRST_FISSION_GENERATION_ZERO",
                "distinctOtherGenerationCount": (
                    int(result["distinctOtherGenerationCount"][index]) if eligible else None
                ),
                "qualifyingStateCount": (
                    int(result["qualifyingStateCount"][index]) if eligible else None
                ),
                "immediateCrossGenerationMatchCount": (
                    int(result["immediateCrossGenerationMatchCount"][index])
                    if eligible
                    else None
                ),
                "immediateOnlyEvidence": (
                    bool(result["immediateOnlyEvidence"][index]) if eligible else None
                ),
                "maximumImmediateCrossGenerationSimilarity": (
                    float(result["maximumImmediateCrossGenerationSimilarity"][index])
                    if eligible
                    and np.isfinite(result["maximumImmediateCrossGenerationSimilarity"][index])
                    else None
                ),
                "sameGenerationMatchCount": (
                    int(result["sameGenerationMatchCount"][index]) if eligible else None
                ),
                "firstMatchingGeneration": (
                    int(result["firstMatchingGeneration"][index])
                    if eligible and result["firstMatchingGeneration"][index] >= 0
                    else None
                ),
                "lastMatchingGeneration": (
                    int(result["lastMatchingGeneration"][index])
                    if eligible and result["lastMatchingGeneration"][index] >= 0
                    else None
                ),
                "earliestMatchingSequenceIndex": (
                    int(result["earliestMatchingSequenceIndex"][index])
                    if eligible and result["earliestMatchingSequenceIndex"][index] >= 0
                    else None
                ),
                "latestMatchingSequenceIndex": (
                    int(result["latestMatchingSequenceIndex"][index])
                    if eligible and result["latestMatchingSequenceIndex"][index] >= 0
                    else None
                ),
            }
        )
    return pd.DataFrame(rows), result["diagnostics"]


def binary_consistency_from_counts(
    n00: int, n01: int, n10: int, n11: int
) -> float | None:
    """Pearson correlation for a binary consecutive-pair contingency table."""

    count = n00 + n01 + n10 + n11
    if count < 2:
        return None
    mean_left = (n10 + n11) / count
    mean_right = (n01 + n11) / count
    variance_left = mean_left * (1.0 - mean_left)
    variance_right = mean_right * (1.0 - mean_right)
    if variance_left <= 0 or variance_right <= 0:
        return None
    covariance = n11 / count - mean_left * mean_right
    value = covariance / np.sqrt(variance_left * variance_right)
    return float(value) if np.isfinite(value) else None
