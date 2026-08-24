"""Outcome-blind label contracts for E01/S19 Loop 3.

The module contains no filesystem access and no emergence, prediction, or
intervention calculation.  It fixes one modal-boundary reference rule and four
nonfactorial structural projections before scientific outcomes are opened.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from e01_clean_directional_confirmation.core import fixed_label_spec
from e01_creative_directional_search.core import (
    label_trajectory as frozen_label_trajectory,
)
from e01_frozen_timebase_ensemble.core import selected_clock_observations

VERSION = "E01-S19-L03-BOUNDARY-COMPOTYPE-MOLECULAR-PROJECTION-v1.0.0"
RESEARCH_STEP_ID = "S19"
LOOP_ID = "S19-L03"
ROOT_SEED_HEX = "f933a93d674227efb7c10f111f47c25d55ba088fb7045a2dc9f8c845b81b2d49"
CANDIDATE_IDS = ("S12F-CANDIDATE-02", "S12F-CANDIDATE-03")
COMPARATOR_LABEL_ID = "MOL_ADJACENT_INCOMING_H900"
BOOTSTRAP_REPLICATES = 4096
H_THRESHOLD = 0.9

BoundarySubstrate = Literal[
    "POST_FISSION_SELECTED_DAUGHTER", "GENERATION_END_PRE_FISSION"
]
ActivationRule = Literal["BACKFILL_ALL_OCCURRENCES", "SECOND_OCCURRENCE_ONWARD"]
ProjectionRule = Literal["INCOMING_BOUNDARY_ENDING", "OUTGOING_BOUNDARY_STARTING"]


@dataclass(frozen=True, slots=True)
class LabelDefinition:
    """One locked label specification."""

    ordinal: int
    label_id: str
    role: str
    boundary_substrate: BoundarySubstrate | None
    activation_rule: ActivationRule | None
    projection_rule: ProjectionRule | None
    evidence_class: str
    temporal_scope: str
    comparator_only: bool
    promotable_scope: str


LABEL_DEFINITIONS = (
    LabelDefinition(
        1,
        COMPARATOR_LABEL_ID,
        "FROZEN_ADJACENT_MOLECULAR_COMPARATOR",
        None,
        None,
        None,
        "FROZEN_S13Y_SOURCE_TRANSPLANT_COMPARATOR",
        "LOCAL_INCOMING_MOLECULAR_SIMILARITY",
        True,
        "NONE_COMPARATOR",
    ),
    LabelDefinition(
        2,
        "PF_MODAL_MEDOID_BACKFILL_INCOMING_H900",
        "POSTFISSION_MODAL_COMPOTYPE_SOURCE_STYLE_BACKFILL",
        "POST_FISSION_SELECTED_DAUGHTER",
        "BACKFILL_ALL_OCCURRENCES",
        "INCOMING_BOUNDARY_ENDING",
        "PAPER_AND_HISTORICAL_SOURCE_INFORMED_RECONSTRUCTION",
        "RETROSPECTIVE_COMPLETED_RUN_REFERENCE_AND_BACKFILL",
        False,
        "RETROSPECTIVE_PAPER_FACING_ONLY",
    ),
    LabelDefinition(
        3,
        "PF_MODAL_MEDOID_ACTIVATED_INCOMING_H900",
        "POSTFISSION_MODAL_COMPOTYPE_SECOND_RECURRENCE_ACTIVATION",
        "POST_FISSION_SELECTED_DAUGHTER",
        "SECOND_OCCURRENCE_ONWARD",
        "INCOMING_BOUNDARY_ENDING",
        "PAPER_AND_HISTORICAL_SOURCE_INFORMED_RECONSTRUCTION",
        "RETROSPECTIVE_REFERENCE_WITH_FORWARD_RECURRENCE_ACTIVATION",
        False,
        "RETROSPECTIVE_PAPER_FACING_ONLY",
    ),
    LabelDefinition(
        4,
        "PF_MODAL_MEDOID_ACTIVATED_OUTGOING_H900",
        "POSTFISSION_MODAL_COMPOTYPE_OUTGOING_PROJECTION",
        "POST_FISSION_SELECTED_DAUGHTER",
        "SECOND_OCCURRENCE_ONWARD",
        "OUTGOING_BOUNDARY_STARTING",
        "PAPER_AND_HISTORICAL_SOURCE_INFORMED_RECONSTRUCTION",
        "RETROSPECTIVE_REFERENCE_WITH_FORWARD_RECURRENCE_ACTIVATION",
        False,
        "RETROSPECTIVE_PAPER_FACING_ONLY",
    ),
    LabelDefinition(
        5,
        "GE_MODAL_MEDOID_ACTIVATED_INCOMING_H900",
        "HISTORICAL_GENERATION_END_MODAL_COMPOTYPE_PROJECTION",
        "GENERATION_END_PRE_FISSION",
        "SECOND_OCCURRENCE_ONWARD",
        "INCOMING_BOUNDARY_ENDING",
        "PINNED_HISTORICAL_TRACE_SEMANTIC_AND_PAPER_INFORMED_REFERENCE",
        "RETROSPECTIVE_REFERENCE_WITH_FORWARD_RECURRENCE_ACTIVATION",
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
    values = np.asarray(states, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 100 or np.any(values < 0):
        raise ValueError("states must be nonnegative observations-by-100 counts")
    masses = values.sum(axis=1)
    if np.any(masses <= 0):
        raise ValueError("boundary substrate contains an empty state")
    return values / masses[:, None]


def cosine_matrix(values: NDArray[np.float64]) -> NDArray[np.float64]:
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms <= 0) or not np.all(np.isfinite(norms)):
        raise ValueError("nonpositive or nonfinite composition norm")
    unit = values / norms[:, None]
    return np.clip(unit @ unit.T, -1.0, 1.0)


def boundary_modal_reference(
    states: NDArray[np.integer[Any]], generations: Iterable[int]
) -> dict[str, Any]:
    """Choose the boundary state with most strict-H>0.9 neighbors.

    Self-membership is included.  Exact frequency ties go to the earliest
    generation and then earliest boundary row.  A compotype is recurrent only
    when at least two boundary states belong to the modal neighborhood.
    """

    values = compositions(states)
    generations_array = np.asarray(list(generations), dtype=np.int64)
    if len(values) != len(generations_array) or len(values) == 0:
        raise ValueError("boundary states and generations must be nonempty and aligned")
    similarities = cosine_matrix(values)
    membership_matrix = similarities > H_THRESHOLD
    frequencies = membership_matrix.sum(axis=1).astype(np.int64)
    best_frequency = int(frequencies.max())
    tied = np.flatnonzero(frequencies == best_frequency)
    reference_index = int(
        min(tied, key=lambda index: (int(generations_array[index]), int(index)))
    )
    scores = similarities[:, reference_index].astype(np.float64, copy=True)
    members = scores > H_THRESHOLD
    if int(np.count_nonzero(members)) != best_frequency:
        raise RuntimeError("modal frequency and reference membership disagree")
    member_indices = np.flatnonzero(members)
    recurrent = best_frequency >= 2
    return {
        "referenceIndex": reference_index,
        "referenceGeneration": int(generations_array[reference_index]),
        "referenceFrequency": best_frequency,
        "recurrent": recurrent,
        "scores": scores,
        "members": members,
        "memberIndices": member_indices.astype(np.int64),
        "firstMemberGeneration": (
            int(generations_array[member_indices[0]]) if len(member_indices) else None
        ),
        "secondMemberGeneration": (
            int(generations_array[member_indices[1]]) if len(member_indices) >= 2 else None
        ),
    }


def activate_membership(
    members: NDArray[np.bool_], rule: ActivationRule, recurrent: bool
) -> NDArray[np.bool_]:
    labels = np.zeros(len(members), dtype=bool)
    if not recurrent:
        return labels
    member_indices = np.flatnonzero(members)
    if rule == "BACKFILL_ALL_OCCURRENCES":
        labels[member_indices] = True
    elif rule == "SECOND_OCCURRENCE_ONWARD":
        labels[member_indices[1:]] = True
    else:  # pragma: no cover - Literal plus locked registry protects this path
        raise ValueError(f"unsupported activation rule: {rule}")
    return labels


def project_boundary_state(
    selected: Iterable[Any],
    boundary_generations: Iterable[int],
    boundary_labels: NDArray[np.bool_],
    boundary_scores: NDArray[np.float64],
    boundary_members: NDArray[np.bool_],
    rule: ProjectionRule,
) -> tuple[list[bool | None], list[float | None], list[bool | None], list[int | None]]:
    """Project one boundary state to the locked molecular clock.

    Incoming projection assigns boundary g to every row carrying generation g,
    including its selected post-fission row.  Outgoing projection assigns a
    post-fission row g to itself and assigns molecular updates in generation
    g+1 to boundary g.  Rows before the first observed boundary are ineligible.
    """

    generations = np.asarray(list(boundary_generations), dtype=np.int64)
    if not (
        len(generations)
        == len(boundary_labels)
        == len(boundary_scores)
        == len(boundary_members)
    ):
        raise ValueError("boundary projection arrays are misaligned")
    index_by_generation = {int(value): index for index, value in enumerate(generations)}
    labels: list[bool | None] = []
    scores: list[float | None] = []
    members: list[bool | None] = []
    sources: list[int | None] = []
    for item in selected:
        generation = int(item.growth_generation_one_based)
        kind = str(item.observation_kind)
        source_generation: int | None
        if generation <= 0:
            source_generation = None
        elif rule == "INCOMING_BOUNDARY_ENDING":
            source_generation = generation
        elif rule == "OUTGOING_BOUNDARY_STARTING":
            source_generation = generation if kind == "post_fission" else generation - 1
            if source_generation <= 0:
                source_generation = None
        else:  # pragma: no cover
            raise ValueError(f"unsupported projection rule: {rule}")
        boundary_index = (
            index_by_generation.get(source_generation)
            if source_generation is not None
            else None
        )
        if boundary_index is None:
            labels.append(None)
            scores.append(None)
            members.append(None)
            sources.append(None)
        else:
            labels.append(bool(boundary_labels[boundary_index]))
            scores.append(float(boundary_scores[boundary_index]))
            members.append(bool(boundary_members[boundary_index]))
            sources.append(int(source_generation))
    return labels, scores, members, sources


def _boundary_observations(trajectory: Any, substrate: BoundarySubstrate) -> tuple[Any, ...]:
    completed = int(trajectory.completed_fissions)
    if substrate == "POST_FISSION_SELECTED_DAUGHTER":
        boundaries = tuple(
            item for item in trajectory.observations if item.observation_kind == "post_fission"
        )
    elif substrate == "GENERATION_END_PRE_FISSION":
        rows = []
        for generation in range(1, completed + 1):
            matches = [
                item
                for item in trajectory.observations
                if item.observation_kind == "molecular_update"
                and int(item.growth_generation_one_based) == generation
            ]
            if not matches:
                raise ValueError(f"generation {generation} has no molecular update")
            rows.append(max(matches, key=lambda item: int(item.observation_index)))
        boundaries = tuple(rows)
    else:  # pragma: no cover
        raise ValueError(f"unsupported boundary substrate: {substrate}")
    if len(boundaries) != completed:
        raise ValueError(f"boundary cardinality mismatch: {len(boundaries)} != {completed}")
    generations = [int(item.growth_generation_one_based) for item in boundaries]
    if generations != list(range(1, completed + 1)):
        raise ValueError("boundary generations are not exactly 1..completed_fissions")
    return boundaries


def label_trajectory(
    trajectory: Any,
    definition: LabelDefinition,
    *,
    clock_id: str = "C1_SELECTED_DAUGHTER_RETAINED",
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Materialize one locked label and its boundary evidence."""

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
        output["boundarySubstrate"] = None
        output["activationRule"] = None
        output["projectionRule"] = None
        output["boundaryMember"] = None
        output["sourceBoundaryGeneration"] = None
        return output, diagnostic, pd.DataFrame()

    assert definition.boundary_substrate is not None
    assert definition.activation_rule is not None
    assert definition.projection_rule is not None
    selected = selected_clock_observations(trajectory, clock_id)
    boundaries = _boundary_observations(trajectory, definition.boundary_substrate)
    boundary_states = np.asarray([item.state for item in boundaries], dtype=np.int64)
    generations = np.asarray(
        [int(item.growth_generation_one_based) for item in boundaries], dtype=np.int64
    )
    reference = boundary_modal_reference(boundary_states, generations)
    boundary_labels = activate_membership(
        reference["members"], definition.activation_rule, bool(reference["recurrent"])
    )
    labels, scores, members, source_generations = project_boundary_state(
        selected,
        generations,
        boundary_labels,
        reference["scores"],
        reference["members"],
        definition.projection_rule,
    )
    rows = []
    for index, (item, label, score, member, source_generation) in enumerate(
        zip(selected, labels, scores, members, source_generations, strict=True)
    ):
        eligible = label is not None
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
                "generation": int(item.growth_generation_one_based),
                "observationKind": str(item.observation_kind),
                "isReplicator": label,
                "labelScore": score,
                "labelStatus": "ELIGIBLE" if eligible else "INELIGIBLE_NO_SOURCE_BOUNDARY",
                "ineligibilityReason": None if eligible else "NO_SOURCE_BOUNDARY_FOR_PROJECTION",
                "boundarySubstrate": definition.boundary_substrate,
                "activationRule": definition.activation_rule,
                "projectionRule": definition.projection_rule,
                "boundaryMember": member,
                "sourceBoundaryGeneration": source_generation,
            }
        )
    boundary_rows = []
    member_indices = {int(value) for value in reference["memberIndices"]}
    for boundary_index, (item, score, active) in enumerate(
        zip(boundaries, reference["scores"], boundary_labels, strict=True)
    ):
        boundary_rows.append(
            {
                "candidateId": str(trajectory.configuration_id),
                "trajectoryId": str(trajectory.trajectory_id),
                "matrixIndex": int(trajectory.matrix_index),
                "labelId": definition.label_id,
                "boundarySubstrate": definition.boundary_substrate,
                "boundaryIndex": boundary_index,
                "boundaryGeneration": int(item.growth_generation_one_based),
                "rawObservationIndex": int(item.observation_index),
                "observationKind": str(item.observation_kind),
                "similarityToReference": float(score),
                "isModalCompotypeMember": boundary_index in member_indices,
                "isActivatedReplicatorBoundary": bool(active),
                "isReferenceBoundary": boundary_index == int(reference["referenceIndex"]),
            }
        )
    diagnostic = {
        "referenceIndex": int(reference["referenceIndex"]),
        "referenceGeneration": int(reference["referenceGeneration"]),
        "referenceFrequency": int(reference["referenceFrequency"]),
        "recurrent": bool(reference["recurrent"]),
        "firstMemberGeneration": reference["firstMemberGeneration"],
        "secondMemberGeneration": reference["secondMemberGeneration"],
        "boundaryCount": len(boundaries),
        "eligibleMolecularCount": sum(value is not None for value in labels),
    }
    return pd.DataFrame(rows), diagnostic, pd.DataFrame(boundary_rows)
