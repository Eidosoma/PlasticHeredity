"""Pure rules for the strict-8 prediction/mechanism diagnosis."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


TRANSITION_NAMES = ("break", "run8_given_break", "coherence_given_run8", "anchor_given_coherence")
CONCENTRATION_NAMES = (
    "shannon_effective_species",
    "simpson_effective_species",
    "occupied_types",
    "top1_share",
    "top2_share",
    "shannon_evenness",
)
ARM_NAMES = (
    "NOOP",
    "EVEN_CONCENTRATE_D1",
    "EVEN_CONCENTRATE_D4",
    "EVEN_FLATTEN_D1",
    "EVEN_FLATTEN_D4",
    "EVEN_RANDOM_D1",
    "EVEN_RANDOM_D4",
    "RICH_CONTRACT_D1",
    "RICH_CONTRACT_D4",
    "RICH_EXPAND_D1",
    "RICH_EXPAND_D4",
)
PRIMARY_INTERVENTION_CONTRASTS = {
    "evenness_concentrate_minus_flatten": ("EVEN_CONCENTRATE_D4", "EVEN_FLATTEN_D4"),
    "richness_contract_minus_expand": ("RICH_CONTRACT_D4", "RICH_EXPAND_D4"),
}


@dataclass(frozen=True)
class EditResult:
    composition: NDArray[np.int64]
    requested_dose: int
    achieved_dose: int
    mass_before: int
    mass_after: int
    occupied_before: int
    occupied_after: int
    simpson_before: float
    simpson_after: float
    steps: tuple[tuple[int, int], ...]


def _probability(composition: NDArray) -> NDArray[np.float64]:
    values = np.asarray(composition, dtype=np.float64)
    if values.ndim != 1 or np.any(values < 0) or values.sum() <= 0:
        raise ValueError("composition must be one-dimensional, nonnegative, and nonempty")
    return values / values.sum()


def concentration_descriptors(composition: NDArray) -> NDArray[np.float64]:
    probability = _probability(composition)
    positive = probability[probability > 0]
    entropy = float(-np.sum(positive * np.log(positive)))
    occupied = int(positive.size)
    ordered = np.sort(probability)
    top1 = float(ordered[-1])
    top2 = float(ordered[-2:].sum())
    evenness = 1.0 if occupied == 1 else entropy / np.log(occupied)
    return np.asarray(
        (
            np.exp(entropy),
            1.0 / float(np.dot(probability, probability)),
            occupied,
            top1,
            top2,
            evenness,
        ),
        dtype=np.float64,
    )


def transition_masks(deepest_gate: NDArray) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Return success and eligibility masks for the four ordered transitions."""

    gates = np.asarray(deepest_gate, dtype=np.int8)
    if gates.ndim < 1 or np.any((gates < 0) | (gates > 4)):
        raise ValueError("deepest gates must be integer codes from zero through four")
    successes = np.stack([gates >= level for level in range(1, 5)], axis=-1)
    eligible = np.stack(
        [np.ones_like(gates, dtype=bool)] + [gates >= level for level in range(1, 4)],
        axis=-1,
    )
    return successes, eligible


def aggregate_transitions(
    deepest_gate: NDArray, branch_slice: slice | None = None
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    gates = np.asarray(deepest_gate)
    if gates.ndim != 2:
        raise ValueError("transition aggregation expects state by branch gates")
    selected = gates if branch_slice is None else gates[:, branch_slice]
    success, eligible = transition_masks(selected)
    return success.sum(axis=1).astype(float), eligible.sum(axis=1).astype(float)


def _seed(parts: Sequence[str | int]) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode()).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _choice(rng: np.random.Generator, values: NDArray[np.int64]) -> int:
    if values.size == 0:
        raise ValueError("cannot choose from an empty set")
    return int(values[int(rng.integers(values.size))])


def _parse_arm(arm: str) -> tuple[str, int]:
    if arm == "NOOP":
        return "NOOP", 0
    if arm not in ARM_NAMES:
        raise ValueError(f"unknown intervention arm: {arm}")
    stem, dose_text = arm.rsplit("_D", 1)
    return stem, int(dose_text)


def apply_intervention(
    composition: NDArray,
    arm: str,
    state_id: str,
    selection_seed: str,
) -> EditResult:
    """Apply one frozen composition policy; D1 is a prefix of D4."""

    values = np.asarray(composition)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("composition must be a one-dimensional integer array")
    if np.any(values < 0) or values.sum() <= 0:
        raise ValueError("composition must be nonnegative and nonempty")
    stem, requested = _parse_arm(arm)
    edited = values.astype(np.int64, copy=True)
    before = concentration_descriptors(edited)
    rng = np.random.default_rng(_seed((selection_seed, state_id, stem)))
    steps: list[tuple[int, int]] = []
    for _ in range(requested):
        present = np.flatnonzero(edited > 0)
        if stem == "EVEN_CONCENTRATE":
            maximum = edited[present].max()
            recipients = present[edited[present] == maximum]
            recipient = _choice(rng, recipients)
            donors = present[(edited[present] >= 2) & (present != recipient)]
            if donors.size == 0:
                break
            minimum = edited[donors].min()
            donor = _choice(rng, donors[edited[donors] == minimum])
        elif stem == "EVEN_FLATTEN":
            maximum = edited[present].max()
            minimum = edited[present].min()
            if maximum - minimum < 2:
                break
            donor = _choice(rng, present[edited[present] == maximum])
            recipients = present[(edited[present] == minimum) & (present != donor)]
            if recipients.size == 0:
                break
            recipient = _choice(rng, recipients)
        elif stem == "EVEN_RANDOM":
            donors = present[edited[present] >= 2]
            if donors.size == 0 or present.size < 2:
                break
            donor = _choice(rng, donors)
            recipient = _choice(rng, present[present != donor])
        elif stem == "RICH_CONTRACT":
            donors = np.flatnonzero(edited == 1)
            if donors.size == 0 or present.size < 2:
                break
            donor = _choice(rng, donors)
            recipient = _choice(rng, present[present != donor])
        elif stem == "RICH_EXPAND":
            donors = present[edited[present] >= 2]
            recipients = np.flatnonzero(edited == 0)
            if donors.size == 0 or recipients.size == 0:
                break
            donor = _choice(rng, donors)
            recipient = _choice(rng, recipients)
        elif stem == "NOOP":
            break
        else:
            raise AssertionError(stem)
        edited[donor] -= 1
        edited[recipient] += 1
        steps.append((donor, recipient))
    after = concentration_descriptors(edited)
    result = EditResult(
        composition=edited,
        requested_dose=requested,
        achieved_dose=len(steps),
        mass_before=int(values.sum()),
        mass_after=int(edited.sum()),
        occupied_before=int(before[2]),
        occupied_after=int(after[2]),
        simpson_before=float(before[1]),
        simpson_after=float(after[1]),
        steps=tuple(steps),
    )
    if result.mass_before != result.mass_after or np.any(result.composition < 0):
        raise AssertionError("intervention violated the mass/nonnegativity contract")
    if stem.startswith("EVEN_") and result.occupied_before != result.occupied_after:
        raise AssertionError("evenness intervention changed the occupied set size")
    if stem == "EVEN_CONCENTRATE" and result.achieved_dose and not result.simpson_after < result.simpson_before:
        raise AssertionError("concentration intervention did not reduce Simpson effective number")
    if stem == "EVEN_FLATTEN" and result.achieved_dose and not result.simpson_after > result.simpson_before:
        raise AssertionError("flattening intervention did not increase Simpson effective number")
    if stem == "RICH_CONTRACT" and result.occupied_after != result.occupied_before - result.achieved_dose:
        raise AssertionError("richness contraction changed the wrong number of types")
    if stem == "RICH_EXPAND" and result.occupied_after != result.occupied_before + result.achieved_dose:
        raise AssertionError("richness expansion changed the wrong number of types")
    return result


def bray_pair_decomposition(left: NDArray, right: NDArray) -> dict[str, float]:
    """Decompose normalized Bray distance by abundance rank of the pair mean."""

    p = _probability(left)
    q = _probability(right)
    absolute = 0.5 * np.abs(p - q)
    order = np.argsort(-(p + q) / 2.0, kind="mergesort")
    total = float(absolute.sum())
    return {
        "bray_distance": total,
        "top1_contribution": float(absolute[order[:1]].sum()),
        "rank2_to5_contribution": float(absolute[order[1:5]].sum()),
        "tail6plus_contribution": float(absolute[order[5:]].sum()),
        "dominant_type_same": float(int(np.argmax(p) == np.argmax(q))),
        "top1_share_left": float(p.max()),
        "top1_share_right": float(q.max()),
    }

