"""Round-2 causal carrier tests for Life-like cellular automata.

The campaign is deliberately separate from :mod:`plastic_ca.causal_heredity`:
the earlier result directory and RNG namespace are immutable, while this module
reuses the validated simulator as a clean-room execution primitive.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Callable, Sequence

import numpy as np

from .causal_heredity import (
    CausalContract,
    _atomic_json,
    _atomic_text,
    _bootstrap_summary,
    _component_spectrum,
    _copy_batch,
    _cosine,
    _donor_record,
    _exact_random_state,
    _hash_seed,
    _observer_vectors,
    _run_recovery,
    _sha256,
    _simulate_batch,
    _site_mask,
    _state_from_hex,
    _state_to_hex,
    _strict_event,
    _summarize_outcomes,
    _write_rows_csv,
    launch_detached,
)
from .e19 import require_pinned_numpy
from .life_family import (
    LifeFamilyContract,
    _mass_support,
    launch_library,
    life_rule_notation,
    live_2x2_counts_batch,
)


PROTOCOL_PATH = Path(__file__).resolve().parent.parent / "LIFE_CARRIER_PROTOCOL.md"
AUDIT_RULES = (124375, 125398)


@dataclass(frozen=True)
class CarrierContract:
    implementation_version: str = "life-carrier-cleanroom-v1"
    namespace: str = "plastic-ca-life-carrier-v1"
    width: int = 16
    height: int = 16
    activity_budget: int = 48
    min_sweeps: int = 4
    max_sweeps: int = 64
    process_noise: float = 0.002
    copy_error: float = 0.005
    donor_horizon: int = 32
    screen_horizon: int = 16
    holdout_horizon: int = 48
    mapping_horizon: int = 16
    density_min: float = 0.05
    density_max: float = 0.95
    pair_density_tolerance: float = 0.05
    assignment_similarity: float = 0.90
    assignment_margin: float = 0.05
    screen_crossover: float = 0.10
    holdout_crossover: float = 0.15
    control_advantage: float = 0.10
    target_probability: float = 0.25
    morphology_neighbor_error: float = 0.02
    morphology_component_cosine: float = 0.95
    morphology_max_swaps: int = 100_000
    doses: tuple[float, ...] = (0.0625, 0.125, 0.25, 0.5, 1.0)
    noise_multipliers: tuple[int, ...] = (0, 1, 2)
    pedigree_depth: int = 8

    def causal(self, namespace_suffix: str = "base", *, width: int | None = None) -> CausalContract:
        extent = self.width if width is None else width
        budget = self.activity_budget if extent == self.width else 4 * extent
        maximum = self.max_sweeps if extent == self.width else 256
        return CausalContract(
            implementation_version=self.implementation_version,
            namespace=f"{self.namespace}:{namespace_suffix}",
            recovery_horizon=self.holdout_horizon,
            life_width=extent,
            life_height=extent,
            life_activity_budget=budget,
            life_min_sweeps=self.min_sweeps,
            life_max_sweeps=maximum,
            life_process_noise=self.process_noise,
            life_copy_error=self.copy_error,
            pedigree_depth=self.pedigree_depth,
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.update(
            {
                "simulator": self.causal().to_dict(),
                "primary_unit": "matched donor pair",
                "recipient_operation": "replace bits under mask; never OR composite",
                "form_id": "0.75 mass support of established primary centroid",
                "assignment": "cosine >=0.90 and best-minus-runner-up >=0.05",
            }
        )
        return value

    @property
    def digest(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class CarrierProfile:
    panel_limit: int
    audit_donors: int
    audit_discovery_cap: int
    audit_replicates: int
    screen_pairs: int
    screen_discovery_cap: int
    screen_replicates: int
    max_candidates: int
    holdout_pairs: int
    holdout_discovery_cap: int
    holdout_replicates: int
    mapping_pairs: int
    mapping_replicates: int
    bootstrap_resamples: int
    morphology_max_swaps: int


CARRIER_PROFILES: dict[str, CarrierProfile] = {
    "smoke": CarrierProfile(2, 2, 64, 2, 1, 96, 2, 1, 1, 96, 2, 1, 1, 100, 250),
    "pilot": CarrierProfile(6, 8, 1_024, 8, 4, 4_096, 4, 2, 4, 4_096, 8, 2, 4, 1_000, 5_000),
    "reference": CarrierProfile(
        24,
        128,
        32_768,
        128,
        16,
        32_768,
        32,
        4,
        64,
        32_768,
        128,
        16,
        32,
        10_000,
        100_000,
    ),
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def build_carrier_panel(atlas: Path, profile: CarrierProfile) -> list[dict[str, Any]]:
    """Return the frozen multi-form development panel."""

    eligible = [
        row
        for row in _read_csv(atlas)
        if int(row["library_size"]) >= 2
        and 0.005 <= float(row["strict"]) <= 0.5
        and float(row["mean_survival"]) >= 16.0
    ]
    ranked = sorted(
        eligible,
        key=lambda row: (-float(row["strict"]), -int(row["library_size"]), int(row["rule"])),
    )[: profile.panel_limit]
    return [
        {
            "rule": int(row["rule"]),
            "notation": str(row["notation"]),
            "development_strict": float(row["strict"]),
            "development_library_size": int(row["library_size"]),
            "development_support_masks": [
                int(value) for value in str(row.get("support_masks", "")).split("|") if value
            ],
        }
        for row in ranked
    ]


def _life_launches(contract: CausalContract) -> tuple[np.ndarray, ...]:
    return launch_library(
        LifeFamilyContract(
            width=contract.life_width,
            height=contract.life_height,
            activity_budget=contract.life_activity_budget,
            min_sweeps=contract.life_min_sweeps,
            max_sweeps=contract.life_max_sweeps,
            flip_noise=contract.life_process_noise,
            copy_error=contract.life_copy_error,
            futures_per_launch=1,
        )
    )


def _donor_density(donor: dict[str, Any]) -> float:
    state = _state_from_hex("life", str(donor["donor_state_hex"]))
    return float(np.mean(state))


def _pair_donors(
    donors: Sequence[dict[str, Any]],
    tolerance: float,
    preferred_forms: Sequence[int] | None = None,
) -> dict[str, Any]:
    by_form: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for donor in donors:
        by_form[int(donor["form_id"])].append(donor)
    forms = (
        [int(form) for form in preferred_forms if int(form) in by_form][:2]
        if preferred_forms is not None
        else sorted(by_form, key=lambda form: (-len(by_form[form]), form))[:2]
    )
    if len(forms) < 2:
        return {"forms": forms, "pairs": []}
    form_a, form_b = forms
    candidates: list[tuple[float, str, str, dict[str, Any], dict[str, Any]]] = []
    for left in by_form[form_a]:
        for right in by_form[form_b]:
            if int(left["launch_index"]) != int(right["launch_index"]):
                continue
            left_target = np.asarray(left["target_compositions"]["primary"], dtype=float)
            right_target = np.asarray(right["target_compositions"]["primary"], dtype=float)
            target_similarity = _cosine(left_target, right_target)
            if target_similarity > 0.80:
                continue
            delta = abs(float(left["density"]) - float(right["density"]))
            if delta <= tolerance + 1e-12:
                candidates.append((delta, str(left["donor_id"]), str(right["donor_id"]), left, right))
    used_a: set[str] = set()
    used_b: set[str] = set()
    pairs: list[dict[str, Any]] = []
    for delta, left_id, right_id, left, right in sorted(candidates, key=lambda row: row[:3]):
        if left_id in used_a or right_id in used_b:
            continue
        used_a.add(left_id)
        used_b.add(right_id)
        pairs.append(
            {
                "pair_id": f"{left['rule']}-{len(pairs):04d}-{left_id}-{right_id}",
                "rule": int(left["rule"]),
                "launch_index": int(left["launch_index"]),
                "form_a": form_a,
                "form_b": form_b,
                "density_delta": delta,
                "target_similarity": _cosine(
                    np.asarray(left["target_compositions"]["primary"], dtype=float),
                    np.asarray(right["target_compositions"]["primary"], dtype=float),
                ),
                "donor_a": left,
                "donor_b": right,
            }
        )
    return {"forms": forms, "pairs": pairs}


def _discover_life_donors(
    rule: int,
    contract: CarrierContract,
    *,
    namespace: str,
    discovery_cap: int,
    target_donors: int | None,
    target_pairs: int | None,
    density_filter: bool,
) -> dict[str, Any]:
    simulator = contract.causal(namespace)
    launches = _life_launches(simulator)
    donors: list[dict[str, Any]] = []
    examined = 0
    strict_seen = 0
    density_rejected = 0
    retention_rejected = 0
    death_counts: Counter[str] = Counter()
    form_counts_seen: Counter[int] = Counter()
    retained_by_form_launch: Counter[tuple[int, int]] = Counter()
    batch_index = 0
    batch_size = 256
    paired: dict[str, Any] = {"forms": [], "pairs": []}
    while examined < discovery_cap:
        size = min(batch_size, discovery_cap - examined)
        launch_indices = [(examined + local) % len(launches) for local in range(size)]
        initial = np.stack([launches[index] for index in launch_indices])
        trace = _simulate_batch(
            "life",
            rule,
            initial,
            simulator,
            horizon=contract.donor_horizon,
            rng_seed=_hash_seed(contract.namespace, namespace, "donor", rule, batch_index),
            observer="primary",
        )
        death_counts.update(reason for reason in trace.death if reason is not None)
        for local in range(size):
            length = int(np.count_nonzero(trace.valid[local]))
            if not length:
                continue
            event = _strict_event(trace.compositions[local, :length], simulator.thresholds)
            if event is None:
                continue
            strict_seen += 1
            donor = _donor_record(
                "life",
                rule,
                "primary",
                launch_indices[local],
                examined + local,
                initial[local],
                trace,
                local,
                "switcher",
                event,
            )
            density = _donor_density(donor)
            if density_filter and not contract.density_min <= density <= contract.density_max:
                density_rejected += 1
                continue
            donor["density"] = density
            donor["form_id"] = int(
                _mass_support(np.asarray(donor["target_compositions"]["primary"]), 0.75)
            )
            form_counts_seen[int(donor["form_id"])] += 1
            if target_pairs is not None:
                retention_key = (int(donor["form_id"]), int(donor["launch_index"]))
                if retained_by_form_launch[retention_key] >= target_pairs:
                    retention_rejected += 1
                    continue
                retained_by_form_launch[retention_key] += 1
            donors.append(donor)
            if target_donors is not None and len(donors) >= target_donors:
                break
        examined += size
        batch_index += 1
        if target_donors is not None and len(donors) >= target_donors:
            break
        if target_pairs is not None:
            preferred = [form for form, _ in sorted(form_counts_seen.items(), key=lambda item: (-item[1], item[0]))[:2]]
            paired = _pair_donors(donors, contract.pair_density_tolerance, preferred)
            if len(paired["pairs"]) >= target_pairs:
                break
    if target_pairs is not None:
        preferred = [form for form, _ in sorted(form_counts_seen.items(), key=lambda item: (-item[1], item[0]))[:2]]
        paired = _pair_donors(donors, contract.pair_density_tolerance, preferred)
    return {
        "entry": {"rule": rule, "notation": life_rule_notation(rule)},
        "namespace": namespace,
        "examined": examined,
        "strict_seen": strict_seen,
        "density_rejected": density_rejected,
        "retention_rejected": retention_rejected,
        "form_counts_seen": {str(key): value for key, value in sorted(form_counts_seen.items())},
        "death_counts": dict(sorted(death_counts.items())),
        "target_donors": target_donors,
        "target_pairs": target_pairs,
        "acquired_donors": len(donors),
        "forms": paired["forms"],
        "pair_count": len(paired["pairs"]),
        "donors": donors[:target_donors] if target_donors is not None else donors,
        "pairs": paired["pairs"][:target_pairs] if target_pairs is not None else [],
    }


def _audit_acquire_task(arguments: tuple[dict[str, Any], CarrierContract, CarrierProfile]) -> dict[str, Any]:
    item, contract, profile = arguments
    return _discover_life_donors(
        int(item["rule"]),
        contract,
        namespace="audit-acquisition",
        discovery_cap=profile.audit_discovery_cap,
        target_donors=profile.audit_donors,
        target_pairs=None,
        density_filter=False,
    )


def _pair_acquire_task(arguments: tuple[dict[str, Any], CarrierContract, CarrierProfile]) -> dict[str, Any]:
    item, contract, profile = arguments
    holdout = bool(item.get("holdout", False))
    return _discover_life_donors(
        int(item["rule"]),
        contract,
        namespace="holdout-acquisition" if holdout else "screen-acquisition",
        discovery_cap=profile.holdout_discovery_cap if holdout else profile.screen_discovery_cap,
        target_donors=None,
        target_pairs=profile.holdout_pairs if holdout else profile.screen_pairs,
        density_filter=True,
    )


def _replace_mask(recipient: np.ndarray, source: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if recipient.shape != source.shape or source.shape != mask.shape:
        raise ValueError("recipient, source, and mask shapes must match")
    result = np.asarray(recipient, dtype=np.bool_).copy()
    result[mask] = np.asarray(source, dtype=np.bool_)[mask]
    return result


def _transform_board(board: np.ndarray, operation: str) -> np.ndarray:
    if operation == "identity":
        return board.copy()
    if operation == "rotate90":
        return np.rot90(board, 1).copy()
    if operation == "reflect":
        return np.fliplr(board).copy()
    if operation == "translate":
        return np.roll(board, shift=(3, 5), axis=(0, 1))
    raise ValueError(f"unknown transform {operation!r}")


def _randomize_within_mask(source: np.ndarray, mask: np.ndarray, key: str) -> np.ndarray:
    positions = np.flatnonzero(mask.ravel())
    live = int(np.count_nonzero(source & mask))
    result = np.zeros(source.size, dtype=np.bool_)
    ranked = sorted(positions, key=lambda position: hashlib.sha256(f"{key}:{position}".encode()).digest())
    result[np.asarray(ranked[:live], dtype=int)] = True
    return result.reshape(source.shape)


def _block_shuffle(source: np.ndarray, mask: np.ndarray, key: str) -> np.ndarray:
    """Deterministically permute 2x2 blocks, then restore exact masked mass."""

    height, width = source.shape
    blocks = [
        source[y : y + 2, x : x + 2].copy()
        for y in range(0, height, 2)
        for x in range(0, width, 2)
    ]
    order = sorted(range(len(blocks)), key=lambda index: hashlib.sha256(f"{key}:{index}".encode()).digest())
    shuffled = np.zeros_like(source)
    cursor = 0
    for y in range(0, height, 2):
        for x in range(0, width, 2):
            shuffled[y : y + 2, x : x + 2] = blocks[order[cursor]]
            cursor += 1
    target = int(np.count_nonzero(source & mask))
    masked = shuffled & mask
    if int(masked.sum()) != target:
        return _randomize_within_mask(source, mask, key + ":mass-repair")
    return masked


def _live_neighbor_hist(board: np.ndarray) -> np.ndarray:
    counts = np.zeros(board.shape, dtype=np.uint8)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx or dy:
                counts += np.roll(board, shift=(dy, dx), axis=(0, 1))
    histogram = np.asarray([np.count_nonzero(board & (counts == value)) for value in range(9)], dtype=float)
    total = histogram.sum()
    return histogram / total if total else histogram


def _morphology_metrics(candidate: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    neighbor_error = float(np.abs(_live_neighbor_hist(candidate) - _live_neighbor_hist(target)).sum())
    component_cosine = _cosine(_component_spectrum(candidate), _component_spectrum(target))
    return neighbor_error, component_cosine


def _morphology_surrogate(
    source: np.ndarray,
    mask: np.ndarray,
    key: str,
    contract: CarrierContract,
    max_swaps: int,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    target = source & mask
    live = int(target.sum())
    available = int(mask.sum())
    if live == 0 or live == available:
        candidate = target.copy()
        error, cosine = _morphology_metrics(candidate, target)
        return candidate, {"swaps": 0, "neighbor_error": error, "component_cosine": cosine}
    candidate = _randomize_within_mask(source, mask, key + ":initial")
    rng = np.random.default_rng(_hash_seed(contract.namespace, "morphology", key))
    positions = np.flatnonzero(mask.ravel())
    best = candidate.copy()
    best_error, best_cosine = _morphology_metrics(best, target)
    best_score = best_error + max(0.0, contract.morphology_component_cosine - best_cosine)
    for proposal in range(1, max_swaps + 1):
        flat = candidate.ravel()
        live_positions = positions[flat[positions]]
        dead_positions = positions[~flat[positions]]
        if not len(live_positions) or not len(dead_positions):
            break
        left = int(live_positions[int(rng.integers(len(live_positions)))])
        right = int(dead_positions[int(rng.integers(len(dead_positions)))])
        flat[left] = False
        flat[right] = True
        error, cosine = _morphology_metrics(candidate, target)
        score = error + max(0.0, contract.morphology_component_cosine - cosine)
        accept = score <= best_score or rng.random() < math.exp(min(0.0, (best_score - score) * 10.0))
        if accept:
            if score < best_score:
                best = candidate.copy()
                best_error, best_cosine, best_score = error, cosine, score
        else:
            flat[left] = True
            flat[right] = False
        if best_error <= contract.morphology_neighbor_error and best_cosine >= contract.morphology_component_cosine:
            return best, {
                "swaps": proposal,
                "neighbor_error": best_error,
                "component_cosine": best_cosine,
            }
    metadata = {"swaps": max_swaps, "neighbor_error": best_error, "component_cosine": best_cosine}
    if best_error <= contract.morphology_neighbor_error and best_cosine >= contract.morphology_component_cosine:
        return best, metadata
    return None, metadata


def _assignment(vector: np.ndarray, target_a: np.ndarray, target_b: np.ndarray, contract: CarrierContract) -> str | None:
    similarity_a = _cosine(vector, target_a)
    similarity_b = _cosine(vector, target_b)
    best = max(similarity_a, similarity_b)
    if best < contract.assignment_similarity or abs(similarity_a - similarity_b) < contract.assignment_margin:
        return None
    return "A" if similarity_a > similarity_b else "B"


def _pair_targets(pair: dict[str, Any], observer: str = "primary") -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray(pair["donor_a"]["target_compositions"][observer], dtype=float),
        np.asarray(pair["donor_b"]["target_compositions"][observer], dtype=float),
    )


def _trace_pair_summary(
    trace: Any,
    pair: dict[str, Any],
    contract: CarrierContract,
    checkpoints: Sequence[int],
) -> dict[str, Any]:
    target_a, target_b = _pair_targets(pair)
    result: dict[str, Any] = {}
    for checkpoint in checkpoints:
        index = checkpoint - 1
        labels: list[str | None] = []
        persistent: list[str | None] = []
        for future in range(len(trace.valid)):
            if index >= trace.valid.shape[1] or not bool(trace.valid[future, index]):
                labels.append(None)
                persistent.append(None)
                continue
            labels.append(_assignment(trace.compositions[future, index], target_a, target_b, contract))
            start = max(0, index - 7)
            sequence = [
                _assignment(trace.compositions[future, position], target_a, target_b, contract)
                if bool(trace.valid[future, position])
                else None
                for position in range(start, index + 1)
            ]
            persistent.append(sequence[0] if len(sequence) == 8 and sequence[0] is not None and len(set(sequence)) == 1 else None)
        result[str(checkpoint)] = {
            "p_a": labels.count("A") / len(labels) if labels else 0.0,
            "p_b": labels.count("B") / len(labels) if labels else 0.0,
            "persistent_p_a": persistent.count("A") / len(persistent) if persistent else 0.0,
            "persistent_p_b": persistent.count("B") / len(persistent) if persistent else 0.0,
            "survival": int(np.count_nonzero(trace.valid[:, index])) / len(labels) if labels else 0.0,
            "valid": int(np.count_nonzero(trace.valid[:, index])) if index < trace.valid.shape[1] else 0,
            "n": len(labels),
        }
    auxiliary: dict[str, Any] = {}
    final_index = checkpoints[-1] - 1
    for observer in ("terminal2x2", "components"):
        target_a_aux, target_b_aux = _pair_targets(pair, observer)
        labels = []
        for future in range(len(trace.valid)):
            if final_index >= trace.valid.shape[1] or not bool(trace.valid[future, final_index]):
                labels.append(None)
                continue
            vector = _observer_vectors("life", trace.terminals[future, final_index][None, ...])[observer][0]
            labels.append(_assignment(vector, target_a_aux, target_b_aux, contract))
        auxiliary[observer] = {
            "p_a": labels.count("A") / len(labels) if labels else 0.0,
            "p_b": labels.count("B") / len(labels) if labels else 0.0,
        }
    result["auxiliary"] = auxiliary
    return result


def _simulate_pair_condition(
    pair: dict[str, Any],
    base: np.ndarray,
    contract: CarrierContract,
    *,
    namespace: str,
    condition: str,
    replicates: int,
    horizon: int,
    process_noise: float | None = None,
    copy_error: float | None = None,
    host_rule: int | None = None,
    width: int | None = None,
) -> dict[str, Any]:
    simulator = contract.causal(namespace, width=width)
    epsilon = simulator.life_copy_error if copy_error is None else copy_error
    initial = _copy_batch(
        base,
        replicates,
        epsilon,
        _hash_seed(contract.namespace, namespace, condition, "initial-copy"),
    )
    trace = _simulate_batch(
        "life",
        int(pair["rule"] if host_rule is None else host_rule),
        initial,
        simulator,
        horizon=horizon,
        rng_seed=_hash_seed(contract.namespace, namespace, condition, "garden"),
        observer="primary",
        process_noise=process_noise,
        copy_error=epsilon,
    )
    checkpoints = tuple(value for value in (1, 6, 12, 16, 24, 48) if value <= horizon)
    return {
        "outcomes": _trace_pair_summary(trace, pair, contract, checkpoints),
        "death_count": int(sum(reason is not None for reason in trace.death)),
    }


def _audit_task(arguments: tuple[dict[str, Any], CarrierContract, CarrierProfile]) -> dict[str, Any]:
    item, contract, profile = arguments
    rule = int(item["entry"]["rule"])
    simulator = contract.causal("audit-garden")
    rows: list[dict[str, Any]] = []
    for donor in item["donors"]:
        donor_id = str(donor["donor_id"])
        source = _state_from_hex("life", str(donor["donor_state_hex"]))
        ancestor = _state_from_hex("life", str(donor["ancestor_state_hex"]))
        launch = _state_from_hex("life", str(donor["initial_state_hex"]))
        mask = _site_mask(source.shape, 0.5, "square", donor_id + ":audit-square")
        fragment = source & mask
        live = int(fragment.sum())
        density_background = _exact_random_state(
            source.shape,
            int(source.sum()),
            donor_id + ":audit-density-background",
        )
        morphology, morphology_meta = _morphology_surrogate(
            source,
            mask,
            donor_id + ":audit-morphology",
            contract,
            profile.morphology_max_swaps,
        )
        conditions: list[tuple[str, np.ndarray, dict[str, Any]]] = [
            ("intact", source, {}),
            ("ancestor", ancestor, {}),
            ("square_empty", fragment, {}),
            (
                "density_random_empty",
                _exact_random_state(source.shape, live, donor_id + ":audit-density-random"),
                {},
            ),
            ("block_shuffle_empty", _block_shuffle(source, mask, donor_id + ":audit-block"), {}),
            ("generic_all_live_empty", mask.copy(), {}),
            ("translated_empty", _transform_board(fragment, "translate"), {}),
            ("rotated_empty", _transform_board(fragment, "rotate90"), {}),
            ("reflected_empty", _transform_board(fragment, "reflect"), {}),
            ("square_launch_recipient", _replace_mask(launch, source, mask), {}),
            (
                "square_density_recipient",
                _replace_mask(density_background, source, mask),
                {},
            ),
        ]
        if morphology is not None:
            conditions.append(("morphology_empty", morphology, morphology_meta))
        else:
            rows.append(
                {
                    "rule": rule,
                    "donor_id": donor_id,
                    "condition": "morphology_empty",
                    "missing": True,
                    "missing_reason": "morphology_tolerance_not_reached",
                    "morphology": morphology_meta,
                }
            )
        for condition, base, metadata in conditions:
            trace, outcomes = _run_recovery(
                donor,
                base,
                simulator,
                replicates=profile.audit_replicates,
                condition_key=f"audit:{rule}:{donor_id}:{condition}",
                horizon=contract.holdout_horizon,
            )
            rows.append(
                {
                    "rule": rule,
                    "notation": life_rule_notation(rule),
                    "donor_id": donor_id,
                    "donor_density": float(donor["density"]),
                    "ancestor_density": float(ancestor.mean()),
                    "condition": condition,
                    "missing": False,
                    "transmitted_live": int(base.sum()),
                    "death_count": int(sum(reason is not None for reason in trace.death)),
                    "morphology": metadata or None,
                    **_summarize_outcomes(outcomes),
                }
            )
    return {"entry": item["entry"], "rows": rows}


def _screen_conditions(
    pair: dict[str, Any],
    all_pairs: Sequence[dict[str, Any]],
    contract: CarrierContract,
    profile: CarrierProfile,
    *,
    phase: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    recipient = _state_from_hex("life", str(pair["donor_a"]["initial_state_hex"]))
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    pair_index = next(
        (index for index, candidate in enumerate(all_pairs) if candidate["pair_id"] == pair["pair_id"]),
        0,
    )
    unrelated = all_pairs[(pair_index + 1) % len(all_pairs)] if all_pairs else pair
    is_holdout = phase == "holdout"
    doses = (0.5,) if is_holdout else contract.doses
    specifications: list[tuple[str, str, float, str, str]] = []
    for label in ("A", "B"):
        for dose in doses:
            specifications.append((label, "donor", dose, "square", "identity"))
        if not is_holdout:
            specifications.extend(
                (
                    (label, "donor", 0.5, "strip", "identity"),
                    (label, "donor", 0.5, "two_lobe", "identity"),
                )
            )
        for control in ("exact_random", "block_shuffle", "morphology", "unrelated"):
            specifications.append((label, control, 0.5, "square", "identity"))
        if is_holdout:
            for operation in ("translate", "rotate90", "reflect"):
                specifications.append((label, "donor", 0.5, "square", operation))

    for label, intervention, dose, geometry, operation in specifications:
        donor_key = "donor_a" if label == "A" else "donor_b"
        donor = pair[donor_key]
        source = _state_from_hex("life", str(donor["donor_state_hex"]))
        key = f"{phase}:{pair['pair_id']}:{label}:{intervention}:{dose}:{geometry}:{operation}"
        mask = _site_mask(source.shape, dose, geometry, key + ":mask")
        morphology_meta: dict[str, Any] | None = None
        if intervention == "donor":
            material = source
        elif intervention == "exact_random":
            material = _randomize_within_mask(source, mask, key)
        elif intervention == "block_shuffle":
            material = _block_shuffle(source, mask, key)
        elif intervention == "unrelated":
            material = _state_from_hex(
                "life",
                str(unrelated[donor_key]["donor_state_hex"]),
            )
        elif intervention == "morphology":
            material, morphology_meta = _morphology_surrogate(
                source,
                mask,
                key,
                contract,
                profile.morphology_max_swaps,
            )
            if material is None:
                missing.append(
                    {
                        "rule": int(pair["rule"]),
                        "pair_id": pair["pair_id"],
                        "source_form": label,
                        "intervention": intervention,
                        "dose": dose,
                        "geometry": geometry,
                        "operation": operation,
                        "missing": True,
                        "missing_reason": "morphology_tolerance_not_reached",
                        "morphology": morphology_meta,
                    }
                )
                continue
        else:
            raise ValueError(intervention)
        if operation != "identity":
            material = _transform_board(material, operation)
            mask = _transform_board(mask, operation)
        base = _replace_mask(recipient, material, mask)
        rows.append(
            {
                "rule": int(pair["rule"]),
                "pair_id": pair["pair_id"],
                "source_form": label,
                "intervention": intervention,
                "dose": dose,
                "geometry": geometry,
                "operation": operation,
                "missing": False,
                "base": base,
                "morphology": morphology_meta,
                "condition_key": key,
            }
        )
    return rows, missing


def _pair_garden_task(arguments: tuple[dict[str, Any], CarrierContract, CarrierProfile]) -> dict[str, Any]:
    item, contract, profile = arguments
    phase = str(item["phase"])
    replicates = profile.holdout_replicates if phase == "holdout" else profile.screen_replicates
    horizon = contract.holdout_horizon if phase == "holdout" else contract.screen_horizon
    all_pairs = item["all_pairs"]
    rows: list[dict[str, Any]] = []
    for pair in item["pairs"]:
        conditions, missing = _screen_conditions(pair, all_pairs, contract, profile, phase=phase)
        rows.extend(missing)
        for condition in conditions:
            simulated = _simulate_pair_condition(
                pair,
                condition.pop("base"),
                contract,
                namespace=f"{phase}-garden",
                condition=str(condition.pop("condition_key")),
                replicates=replicates,
                horizon=horizon,
            )
            rows.append({**condition, **simulated})
    return {
        "entry": item["entry"],
        "chunk": item["chunk"],
        "phase": phase,
        "pair_count": len(item["pairs"]),
        "rows": rows,
    }


def _tile_mask(shape: tuple[int, int], tile: int) -> np.ndarray:
    height, width = shape
    if height % 4 or width % 4 or not 0 <= tile < 16:
        raise ValueError("tile scans require a 4x4 tiling")
    tile_height, tile_width = height // 4, width // 4
    row, column = divmod(tile, 4)
    mask = np.zeros(shape, dtype=np.bool_)
    mask[
        row * tile_height : (row + 1) * tile_height,
        column * tile_width : (column + 1) * tile_width,
    ] = True
    return mask


def _tile_to_32(board: np.ndarray) -> np.ndarray:
    return np.tile(board, (2, 2))


def _pedigree_summary(
    pair: dict[str, Any],
    label: str,
    base: np.ndarray,
    contract: CarrierContract,
    profile: CarrierProfile,
) -> dict[str, Any]:
    simulator = contract.causal("mapping-pedigree")
    states = _copy_batch(
        base,
        profile.mapping_replicates,
        simulator.life_copy_error,
        _hash_seed(contract.namespace, "mapping-pedigree", pair["pair_id"], label, "root"),
    )
    final_trace = None
    for depth in range(1, contract.pedigree_depth + 1):
        children: list[np.ndarray] = []
        for index, state in enumerate(states):
            mask = _site_mask(state.shape, 0.5, "square", f"{pair['pair_id']}:{label}:{depth}:{index}")
            children.extend((state & mask, state & ~mask))
        child_batch = np.stack(children)
        if simulator.life_copy_error:
            rng = np.random.default_rng(
                _hash_seed(contract.namespace, "mapping-pedigree-copy", pair["pair_id"], label, depth)
            )
            child_batch ^= rng.random(child_batch.shape) < simulator.life_copy_error
        final_trace = _simulate_batch(
            "life",
            int(pair["rule"]),
            child_batch,
            simulator,
            horizon=1,
            rng_seed=_hash_seed(contract.namespace, "mapping-pedigree-step", pair["pair_id"], label, depth),
            observer="primary",
            copy_error=0.0,
        )
        states = final_trace.offspring[:, 0].copy()
        invalid = ~final_trace.valid[:, 0]
        states[invalid] = False
    assert final_trace is not None
    target_a, target_b = _pair_targets(pair)
    labels: list[str | None] = []
    for future in range(len(final_trace.valid)):
        labels.append(
            _assignment(final_trace.compositions[future, 0], target_a, target_b, contract)
            if bool(final_trace.valid[future, 0])
            else None
        )
    return {
        "p_a": labels.count("A") / len(labels),
        "p_b": labels.count("B") / len(labels),
        "survival": int(np.count_nonzero(final_trace.valid[:, 0])) / len(labels),
        "n_leaves": len(labels),
    }


def _mapping_task(arguments: tuple[dict[str, Any], CarrierContract, CarrierProfile]) -> dict[str, Any]:
    item, contract, profile = arguments
    rows: list[dict[str, Any]] = []
    for pair in item["pairs"]:
        recipient = _state_from_hex("life", str(pair["donor_a"]["initial_state_hex"]))
        for label in ("A", "B"):
            donor = pair["donor_a" if label == "A" else "donor_b"]
            source = _state_from_hex("life", str(donor["donor_state_hex"]))
            primary_mask = _site_mask(source.shape, 0.5, "square", f"mapping:{pair['pair_id']}:primary")
            primary = _replace_mask(recipient, source, primary_mask)

            for tile in range(16):
                mask = _tile_mask(source.shape, tile)
                bases = {
                    "tile_sufficiency": _replace_mask(recipient, source, mask),
                    "tile_deletion": _replace_mask(source, recipient, mask),
                }
                for intervention, base in bases.items():
                    simulated = _simulate_pair_condition(
                        pair,
                        base,
                        contract,
                        namespace="mapping-tiles",
                        condition=f"{pair['pair_id']}:{label}:{intervention}:{tile}",
                        replicates=profile.mapping_replicates,
                        horizon=contract.mapping_horizon,
                    )
                    rows.append(
                        {
                            "rule": int(pair["rule"]),
                            "pair_id": pair["pair_id"],
                            "source_form": label,
                            "probe": intervention,
                            "tile": tile,
                            **simulated,
                        }
                    )

            for eta_multiplier in contract.noise_multipliers:
                for epsilon_multiplier in contract.noise_multipliers:
                    simulated = _simulate_pair_condition(
                        pair,
                        primary,
                        contract,
                        namespace="mapping-noise",
                        condition=(
                            f"{pair['pair_id']}:{label}:eta{eta_multiplier}:epsilon{epsilon_multiplier}"
                        ),
                        replicates=profile.mapping_replicates,
                        horizon=contract.mapping_horizon,
                        process_noise=contract.process_noise * eta_multiplier,
                        copy_error=contract.copy_error * epsilon_multiplier,
                    )
                    rows.append(
                        {
                            "rule": int(pair["rule"]),
                            "pair_id": pair["pair_id"],
                            "source_form": label,
                            "probe": "noise",
                            "eta_multiplier": eta_multiplier,
                            "epsilon_multiplier": epsilon_multiplier,
                            **simulated,
                        }
                    )

            for operation in ("identity", "translate", "rotate90", "reflect"):
                transformed_source = _transform_board(source, operation)
                transformed_mask = _transform_board(primary_mask, operation)
                base = _replace_mask(recipient, transformed_source, transformed_mask)
                simulated = _simulate_pair_condition(
                    pair,
                    base,
                    contract,
                    namespace="mapping-transform",
                    condition=f"{pair['pair_id']}:{label}:{operation}",
                    replicates=profile.mapping_replicates,
                    horizon=contract.mapping_horizon,
                )
                rows.append(
                    {
                        "rule": int(pair["rule"]),
                        "pair_id": pair["pair_id"],
                        "source_form": label,
                        "probe": "transform",
                        "operation": operation,
                        **simulated,
                    }
                )

            source32 = _tile_to_32(source)
            recipient32 = _tile_to_32(recipient)
            mask32 = _tile_to_32(primary_mask)
            simulated = _simulate_pair_condition(
                pair,
                _replace_mask(recipient32, source32, mask32),
                contract,
                namespace="mapping-scale32",
                condition=f"{pair['pair_id']}:{label}:scale32",
                replicates=profile.mapping_replicates,
                horizon=contract.mapping_horizon,
                width=32,
            )
            rows.append(
                {
                    "rule": int(pair["rule"]),
                    "pair_id": pair["pair_id"],
                    "source_form": label,
                    "probe": "scale",
                    "width": 32,
                    **simulated,
                }
            )

            for bit in range(17):
                neighbor = int(pair["rule"]) ^ (1 << bit)
                simulated = _simulate_pair_condition(
                    pair,
                    primary,
                    contract,
                    namespace="mapping-neighbor",
                    condition=f"{pair['pair_id']}:{label}:bit{bit}",
                    replicates=profile.mapping_replicates,
                    horizon=contract.mapping_horizon,
                    host_rule=neighbor,
                )
                rows.append(
                    {
                        "rule": int(pair["rule"]),
                        "host_rule": neighbor,
                        "bit": bit,
                        "pair_id": pair["pair_id"],
                        "source_form": label,
                        "probe": "neighbor",
                        **simulated,
                    }
                )

            rows.append(
                {
                    "rule": int(pair["rule"]),
                    "pair_id": pair["pair_id"],
                    "source_form": label,
                    "probe": "pedigree",
                    "depth": contract.pedigree_depth,
                    "pedigree": _pedigree_summary(pair, label, primary, contract, profile),
                }
            )
    return {
        "entry": item["entry"],
        "chunk": item["chunk"],
        "pair_count": len(item["pairs"]),
        "rows": rows,
    }


def _flatten(stage: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for result in stage for row in result.get("rows", [])]


def _sign_flip_p(values: Sequence[float], resamples: int, seed: int) -> float:
    data = np.asarray(values, dtype=float)
    if not len(data):
        return 1.0
    observed = float(data.mean())
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(resamples):
        signs = rng.choice(np.asarray((-1.0, 1.0)), size=len(data))
        exceed += int(float(np.mean(data * signs)) >= observed)
    return (exceed + 1) / (resamples + 1)


def _holm(p_values: Sequence[float]) -> list[float]:
    if not p_values:
        return []
    order = sorted(range(len(p_values)), key=lambda index: (p_values[index], index))
    adjusted = [1.0] * len(p_values)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, float(p_values[index]) * (count - rank))
        adjusted[index] = min(1.0, running)
    return adjusted


def _summary(values: Sequence[float], profile: CarrierProfile, contract: CarrierContract, *key: object) -> dict[str, Any]:
    result = _bootstrap_summary(
        values,
        profile.bootstrap_resamples,
        _hash_seed(contract.namespace, "bootstrap", *key),
    )
    result["p_value"] = _sign_flip_p(
        values,
        profile.bootstrap_resamples,
        _hash_seed(contract.namespace, "permutation", *key),
    )
    return result


def _audit_adjudication(
    stage: Sequence[dict[str, Any]], profile: CarrierProfile, contract: CarrierContract
) -> dict[str, Any]:
    rows = [row for row in _flatten(stage) if not row.get("missing", False)]
    results: dict[str, Any] = {}
    for rule in AUDIT_RULES:
        grouped: dict[str, dict[str, float]] = defaultdict(dict)
        for row in rows:
            if int(row["rule"]) == rule:
                grouped[str(row["donor_id"])][str(row["condition"])] = float(row["success_rate"])
        contrast_specs = {
            "square_minus_density_random": ("square_empty", "density_random_empty"),
            "square_minus_generic_all_live": ("square_empty", "generic_all_live_empty"),
            "square_minus_morphology": ("square_empty", "morphology_empty"),
            "intact_minus_ancestor": ("intact", "ancestor"),
        }
        contrasts: dict[str, Any] = {}
        p_values: list[float] = []
        names: list[str] = []
        for name, (left, right) in contrast_specs.items():
            values = [cells[left] - cells[right] for cells in grouped.values() if left in cells and right in cells]
            summary = _summary(values, profile, contract, "audit", rule, name)
            contrasts[name] = summary
            p_values.append(float(summary["p_value"]))
            names.append(name)
        for name, adjusted in zip(names, _holm(p_values), strict=True):
            contrasts[name]["p_holm"] = adjusted
        historical = contrasts["square_minus_density_random"]
        replicated = bool(
            historical["mean"] is not None
            and historical["mean"] >= 0.08
            and historical["ci95"][0] is not None
            and historical["ci95"][0] > 0.0
            and historical["p_holm"] < 0.05
        )
        specificity_names = (
            "square_minus_generic_all_live",
            "square_minus_morphology",
            "intact_minus_ancestor",
        )
        acquired_specific = replicated and all(
            contrasts[name]["mean"] is not None
            and contrasts[name]["mean"] >= contract.control_advantage
            and contrasts[name]["ci95"][0] is not None
            and contrasts[name]["ci95"][0] > 0.0
            and contrasts[name]["p_holm"] < 0.05
            for name in specificity_names
        )
        verdict = (
            "ACQUIRED_FORM_SPECIFIC_AUDIT_LEAD"
            if acquired_specific
            else "SATURATION_NUCLEATION_ONLY"
            if replicated
            else "SATURATION_LEAD_NOT_REPLICATED"
        )
        results[str(rule)] = {
            "n_donors": len(grouped),
            "contrasts": contrasts,
            "historical_effect_replicated": replicated,
            "acquired_form_specific": acquired_specific,
            "verdict": verdict,
        }
    return results


def _row_probability(row: dict[str, Any], checkpoint: int, label: str, *, observer: str | None = None) -> float:
    if observer is None:
        values = row["outcomes"][str(checkpoint)]
    else:
        values = row["outcomes"]["auxiliary"][observer]
    return float(values["p_a" if label == "A" else "p_b"])


def _condition_index(rows: Sequence[dict[str, Any]]) -> dict[tuple[Any, ...], dict[str, Any]]:
    result: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        if row.get("missing", False):
            continue
        key = (
            int(row["rule"]),
            str(row["pair_id"]),
            str(row["source_form"]),
            str(row["intervention"]),
            float(row["dose"]),
            str(row["geometry"]),
            str(row["operation"]),
        )
        result[key] = row
    return result


def _rule_crossovers(
    rows: Sequence[dict[str, Any]],
    rule: int,
    checkpoint: int,
    *,
    intervention: str = "donor",
    operation: str = "identity",
    observer: str | None = None,
) -> tuple[list[float], list[float], list[float], list[float]]:
    index = _condition_index(rows)
    pair_ids = sorted(
        {
            str(row["pair_id"])
            for row in rows
            if not row.get("missing", False) and int(row["rule"]) == rule
        }
    )
    crossovers: list[float] = []
    directions_a: list[float] = []
    directions_b: list[float] = []
    survival: list[float] = []
    for pair_id in pair_ids:
        base = (rule, pair_id)
        key_a = base + ("A", intervention, 0.5, "square", operation)
        key_b = base + ("B", intervention, 0.5, "square", operation)
        if key_a not in index or key_b not in index:
            continue
        row_a, row_b = index[key_a], index[key_b]
        d_a = _row_probability(row_a, checkpoint, "A", observer=observer) - _row_probability(
            row_b, checkpoint, "A", observer=observer
        )
        d_b = _row_probability(row_b, checkpoint, "B", observer=observer) - _row_probability(
            row_a, checkpoint, "B", observer=observer
        )
        directions_a.append(d_a)
        directions_b.append(d_b)
        crossovers.append(min(d_a, d_b))
        if observer is None:
            survival.append(
                0.5
                * (
                    float(row_a["outcomes"][str(checkpoint)]["survival"])
                    + float(row_b["outcomes"][str(checkpoint)]["survival"])
                )
            )
    return crossovers, directions_a, directions_b, survival


def _screen_adjudication(
    stage: Sequence[dict[str, Any]], profile: CarrierProfile, contract: CarrierContract
) -> dict[str, Any]:
    rows = _flatten(stage)
    rules = sorted({int(row["rule"]) for row in rows})
    result: dict[str, Any] = {}
    for rule in rules:
        crossovers, direction_a, direction_b, survival = _rule_crossovers(
            rows, rule, contract.screen_horizon
        )
        auxiliary: dict[str, float | None] = {}
        for observer in ("terminal2x2", "components"):
            values, _, _, _ = _rule_crossovers(
                rows,
                rule,
                contract.screen_horizon,
                observer=observer,
            )
            auxiliary[observer] = float(np.mean(values)) if values else None
        mean_crossover = float(np.mean(crossovers)) if crossovers else None
        mean_survival = float(np.mean(survival)) if survival else 0.0
        eligible = bool(
            len(crossovers) >= min(12, profile.screen_pairs)
            and mean_crossover is not None
            and mean_crossover >= contract.screen_crossover
            and mean_survival >= 0.90
            and all(value is not None and value > 0.0 for value in auxiliary.values())
        )
        result[str(rule)] = {
            "n_pairs": len(crossovers),
            "mean_crossover": mean_crossover,
            "mean_direction_a": float(np.mean(direction_a)) if direction_a else None,
            "mean_direction_b": float(np.mean(direction_b)) if direction_b else None,
            "mean_survival": mean_survival,
            "auxiliary_crossover": auxiliary,
            "eligible_for_holdout": eligible,
        }
    return result


def select_holdout_candidates(screen: dict[str, Any], maximum: int) -> list[int]:
    eligible = [
        (int(rule), values)
        for rule, values in screen.items()
        if bool(values["eligible_for_holdout"])
    ]
    eligible.sort(key=lambda item: (-float(item[1]["mean_crossover"]), item[0]))
    return [rule for rule, _ in eligible[:maximum]]


def _control_advantages(
    rows: Sequence[dict[str, Any]], rule: int, checkpoint: int, control: str
) -> list[float]:
    index = _condition_index(rows)
    pair_ids = sorted({key[1] for key in index if key[0] == rule})
    differences: list[float] = []
    for pair_id in pair_ids:
        donor_a = index.get((rule, pair_id, "A", "donor", 0.5, "square", "identity"))
        donor_b = index.get((rule, pair_id, "B", "donor", 0.5, "square", "identity"))
        control_a = index.get((rule, pair_id, "A", control, 0.5, "square", "identity"))
        control_b = index.get((rule, pair_id, "B", control, 0.5, "square", "identity"))
        if None in (donor_a, donor_b, control_a, control_b):
            continue
        donor_correct = 0.5 * (
            _row_probability(donor_a, checkpoint, "A") + _row_probability(donor_b, checkpoint, "B")
        )
        control_correct = 0.5 * (
            _row_probability(control_a, checkpoint, "A") + _row_probability(control_b, checkpoint, "B")
        )
        differences.append(donor_correct - control_correct)
    return differences


def _mapping_crossovers(
    rows: Sequence[dict[str, Any]], rule: int, probe: str, checkpoint: int, **fields: Any
) -> list[float]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if int(row["rule"]) != rule or row.get("probe") != probe:
            continue
        if any(row.get(key) != value for key, value in fields.items()):
            continue
        grouped[str(row["pair_id"])][str(row["source_form"])] = row
    values: list[float] = []
    for pair_rows in grouped.values():
        if "A" not in pair_rows or "B" not in pair_rows:
            continue
        row_a, row_b = pair_rows["A"], pair_rows["B"]
        if probe == "pedigree":
            d_a = float(row_a["pedigree"]["p_a"]) - float(row_b["pedigree"]["p_a"])
            d_b = float(row_b["pedigree"]["p_b"]) - float(row_a["pedigree"]["p_b"])
        else:
            d_a = _row_probability(row_a, checkpoint, "A") - _row_probability(row_b, checkpoint, "A")
            d_b = _row_probability(row_b, checkpoint, "B") - _row_probability(row_a, checkpoint, "B")
        values.append(min(d_a, d_b))
    return values


def _holdout_adjudication(
    holdout_stage: Sequence[dict[str, Any]],
    mapping_stage: Sequence[dict[str, Any]],
    profile: CarrierProfile,
    contract: CarrierContract,
) -> dict[str, Any]:
    rows = _flatten(holdout_stage)
    mapping = _flatten(mapping_stage)
    rules = sorted({int(row["rule"]) for row in rows})
    results: dict[str, Any] = {}
    family_tests: list[tuple[int, str, float]] = []
    for rule in rules:
        cross48, direction_a, direction_b, _ = _rule_crossovers(rows, rule, 48)
        cross16, _, _, _ = _rule_crossovers(rows, rule, 16)
        primary = _summary(cross48, profile, contract, "holdout", rule, "crossover48")
        controls: dict[str, Any] = {}
        for control in ("exact_random", "block_shuffle", "morphology"):
            values = _control_advantages(rows, rule, 48, control)
            controls[control] = _summary(values, profile, contract, "holdout", rule, control)
            family_tests.append((rule, control, float(controls[control]["p_value"])))
        family_tests.append((rule, "crossover48", float(primary["p_value"])))
        auxiliary = {}
        for observer in ("terminal2x2", "components"):
            values, _, _, _ = _rule_crossovers(rows, rule, 48, observer=observer)
            auxiliary[observer] = _summary(values, profile, contract, "holdout", rule, observer)
        index = _condition_index(rows)
        correct_probabilities: list[float] = []
        improved = 0
        pair_count = 0
        for pair_id in sorted({key[1] for key in index if key[0] == rule}):
            row_a = index.get((rule, pair_id, "A", "donor", 0.5, "square", "identity"))
            row_b = index.get((rule, pair_id, "B", "donor", 0.5, "square", "identity"))
            if row_a is None or row_b is None:
                continue
            correct_probabilities.append(
                0.5 * (_row_probability(row_a, 48, "A") + _row_probability(row_b, 48, "B"))
            )
            d_a = _row_probability(row_a, 48, "A") - _row_probability(row_b, 48, "A")
            d_b = _row_probability(row_b, 48, "B") - _row_probability(row_a, 48, "B")
            improved += int(min(d_a, d_b) > 0.0)
            pair_count += 1
        pedigree = _summary(
            _mapping_crossovers(mapping, rule, "pedigree", contract.mapping_horizon),
            profile,
            contract,
            "mapping",
            rule,
            "pedigree",
        )
        scale = _summary(
            _mapping_crossovers(mapping, rule, "scale", contract.mapping_horizon, width=32),
            profile,
            contract,
            "mapping",
            rule,
            "scale32",
        )
        transform_values: dict[str, Any] = {}
        for operation in ("translate", "rotate90", "reflect"):
            transform_values[operation] = _summary(
                _mapping_crossovers(
                    mapping,
                    rule,
                    "transform",
                    contract.mapping_horizon,
                    operation=operation,
                ),
                profile,
                contract,
                "mapping",
                rule,
                operation,
            )
        results[str(rule)] = {
            "crossover_generation_16": _summary(cross16, profile, contract, "holdout", rule, "crossover16"),
            "crossover_generation_48": primary,
            "direction_a_mean": float(np.mean(direction_a)) if direction_a else None,
            "direction_b_mean": float(np.mean(direction_b)) if direction_b else None,
            "correct_history_probability": float(np.mean(correct_probabilities)) if correct_probabilities else 0.0,
            "fraction_pairs_improved": improved / pair_count if pair_count else 0.0,
            "controls": controls,
            "auxiliary": auxiliary,
            "pedigree_depth_8": pedigree,
            "scale_32": scale,
            "transformations": transform_values,
        }

    adjusted = _holm([item[2] for item in family_tests])
    for (rule, name, _), value in zip(family_tests, adjusted, strict=True):
        target = (
            results[str(rule)]["crossover_generation_48"]
            if name == "crossover48"
            else results[str(rule)]["controls"][name]
        )
        target["p_holm"] = value

    for rule_text, values in results.items():
        primary = values["crossover_generation_48"]
        controls_pass = all(
            summary["mean"] is not None
            and summary["mean"] >= contract.control_advantage
            and summary["ci95"][0] is not None
            and summary["ci95"][0] > 0.0
            and summary.get("p_holm", 1.0) < 0.05
            for summary in values["controls"].values()
        )
        auxiliary_pass = any(
            summary["mean"] is not None and summary["mean"] > 0.0
            for summary in values["auxiliary"].values()
        )
        transformations_pass = all(
            values["transformations"][operation]["mean"] is not None
            and values["transformations"][operation]["mean"] > 0.0
            for operation in ("translate", "rotate90")
        )
        durable = bool(
            primary["mean"] is not None
            and primary["mean"] >= contract.holdout_crossover
            and primary["ci95"][0] is not None
            and primary["ci95"][0] > 0.0
            and primary.get("p_holm", 1.0) < 0.05
            and values["direction_a_mean"] is not None
            and values["direction_a_mean"] > 0.0
            and values["direction_b_mean"] is not None
            and values["direction_b_mean"] > 0.0
            and controls_pass
            and values["correct_history_probability"] >= contract.target_probability
            and values["fraction_pairs_improved"] >= 0.5
            and values["pedigree_depth_8"]["mean"] is not None
            and values["pedigree_depth_8"]["ci95"][0] is not None
            and values["pedigree_depth_8"]["ci95"][0] > 0.0
            and auxiliary_pass
            and values["scale_32"]["mean"] is not None
            and values["scale_32"]["mean"] > 0.0
            and transformations_pass
        )
        transient = bool(
            not durable
            and values["crossover_generation_16"]["mean"] is not None
            and values["crossover_generation_16"]["mean"] >= contract.screen_crossover
            and values["crossover_generation_16"]["ci95"][0] is not None
            and values["crossover_generation_16"]["ci95"][0] > 0.0
        )
        values["durable_gate"] = durable
        values["transient_gate"] = transient
        values["verdict"] = (
            "DURABLE_CAUSAL_CARRIER"
            if durable
            else "FORM_SPECIFIC_TRANSIENT_CARRIER"
            if transient
            else "NO_CAUSAL_CARRIER_FOUND"
        )
    return results


def _chunks(values: Sequence[Any], size: int) -> list[list[Any]]:
    return [list(values[index : index + size]) for index in range(0, len(values), size)]


def _run_stage(
    output: Path,
    stage: str,
    items: Sequence[dict[str, Any]],
    task: Callable[[tuple[dict[str, Any], CarrierContract, CarrierProfile]], dict[str, Any]],
    contract: CarrierContract,
    profile: CarrierProfile,
    *,
    design_digest: str,
    workers: int,
    resume: bool,
    execute: bool,
    deadline: float | None,
    status: Callable[..., None],
) -> tuple[list[dict[str, Any]], bool]:
    root = output / stage
    checkpoints = root / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    missing: list[dict[str, Any]] = []
    for item in items:
        key = str(item["checkpoint"])
        path = checkpoints / f"{key}.json"
        if (resume or not execute) and path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("design_digest") == design_digest:
                results[key] = payload["result"]
                continue
        missing.append(item)
    started = time.time()

    def save(item: dict[str, Any], result: dict[str, Any]) -> None:
        key = str(item["checkpoint"])
        _atomic_json(
            checkpoints / f"{key}.json",
            {"design_digest": design_digest, "stage": stage, "checkpoint": key, "result": result},
        )
        results[key] = result
        elapsed = max(time.time() - started, 1e-9)
        completed_now = max(1, len(results))
        eta = elapsed / completed_now * max(0, len(items) - len(results))
        status("running", stage, completed=len(results), total=len(items), eta_seconds=eta)

    truncated = False
    if execute and missing:
        if workers <= 1:
            for item in missing:
                if deadline is not None and time.time() >= deadline:
                    truncated = True
                    break
                save(item, task((item, contract, profile)))
        else:
            pool = ProcessPoolExecutor(max_workers=min(workers, len(missing)))
            futures = {pool.submit(task, (item, contract, profile)): item for item in missing}
            processed: set[Any] = set()
            try:
                for future in as_completed(futures):
                    item = futures[future]
                    save(item, future.result())
                    processed.add(future)
                    if deadline is not None and time.time() >= deadline:
                        truncated = True
                        for pending in futures:
                            if pending not in processed:
                                pending.cancel()
                        break
            finally:
                pool.shutdown(wait=True, cancel_futures=truncated)
            if truncated:
                for future, item in futures.items():
                    if future in processed or future.cancelled() or not future.done():
                        continue
                    save(item, future.result())
    complete = len(results) == len(items)
    _atomic_json(
        root / "stage_summary.json",
        {
            "stage": stage,
            "design_digest": design_digest,
            "complete": complete,
            "completed": len(results),
            "total": len(items),
            "budget_truncated": truncated or not complete,
        },
    )
    if complete:
        _atomic_text(root / "COMPLETE", "complete\n")
    return [results[key] for key in sorted(results)], complete


def _screen_items(acquisitions: Sequence[dict[str, Any]], phase: str, chunk_size: int) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for result in acquisitions:
        pairs = list(result.get("pairs", []))
        for chunk, selected in enumerate(_chunks(pairs, chunk_size)):
            rule = int(result["entry"]["rule"])
            items.append(
                {
                    "checkpoint": f"rule-{rule:06d}-chunk-{chunk:03d}",
                    "entry": result["entry"],
                    "phase": phase,
                    "chunk": chunk,
                    "pairs": selected,
                    "all_pairs": pairs,
                }
            )
    return items


def _render_report(results: dict[str, Any]) -> str:
    lines = [
        "# Life carrier falsification and discovery campaign",
        "",
        f"State: **{results['state']}**. Profile: `{results['profile']}`.",
        f"Design digest: `{results['design_digest']}`.",
        "",
        "## Saturation audit",
        "",
    ]
    for rule, values in results["adjudication"]["audit"].items():
        effect = values["contrasts"]["square_minus_density_random"]
        lines.append(
            f"- Rule {rule}: **{values['verdict']}**; historical contrast "
            f"`{effect['mean']}` with 95% CI `{effect['ci95']}`."
        )
    lines.extend(("", "## Multi-form screen", ""))
    selected = results["adjudication"]["selected_candidates"]
    lines.append(f"Sealed holdout rules: `{selected}`.")
    for rule, values in results["adjudication"]["screen"].items():
        lines.append(
            f"- Rule {rule}: {values['n_pairs']} pairs, crossover "
            f"`{values['mean_crossover']}`, eligible `{values['eligible_for_holdout']}`."
        )
    lines.extend(("", "## Independent holdout", ""))
    if not results["adjudication"]["holdout"]:
        lines.append("No rule met the frozen screen gate, so no confirmatory holdout was opened.")
    for rule, values in results["adjudication"]["holdout"].items():
        primary = values["crossover_generation_48"]
        lines.append(
            f"- Rule {rule}: **{values['verdict']}**; generation-48 crossover "
            f"`{primary['mean']}` with 95% CI `{primary['ci95']}`."
        )
    lines.extend(
        (
            "",
            "## Evidence boundary",
            "",
            "A dense block that generically nucleates an attractor is not an acquired carrier. A carrier "
            "requires reciprocal A/B information from matched parents to change an identical recipient, "
            "survive controls, and persist through a cue-free pedigree. Screening is candidate selection "
            "only; claims come from the untouched holdout.",
            "",
        )
    )
    return "\n".join(lines)


def _render_lay_summary(results: dict[str, Any]) -> str:
    audit_values = results["adjudication"]["audit"].get("125398", {})
    audit_verdict = audit_values.get("verdict", "not completed")
    holdout = results["adjudication"]["holdout"]
    durable = [rule for rule, values in holdout.items() if values.get("durable_gate")]
    transient = [rule for rule, values in holdout.items() if values.get("transient_gate")]
    return "\n\n".join(
        (
            "# Lay summary",
            (
                "This round asks whether a parent carries a specific acquired pattern, rather than merely "
                "dropping enough live cells into a region that the rule naturally fills in."
            ),
            f"The almost-solid rule-125398 lead was classified as: **{audit_verdict}**.",
            (
                f"Durable form-specific carriers: {', '.join(durable) if durable else 'none'}. "
                f"Transient form-specific leads: {', '.join(transient) if transient else 'none'}."
            ),
            (
                "A durable result means A material and B material drove the same recipient toward different "
                "parental forms, survived matched random and shuffled controls, and remained detectable down "
                "a family tree. A null result still permits ordinary attractors and short-lived continuity."
            ),
        )
    ) + "\n"


def _update_discovery_log(results: dict[str, Any], path: Path) -> None:
    start = "<!-- life-carrier-round-2:start -->"
    end = "<!-- life-carrier-round-2:end -->"
    section = "\n".join(
        (
            start,
            "## Life carrier round 2",
            "",
            f"Completed under design `{results['design_digest']}`.",
            "",
            f"- Rule-125398 audit: `{results['adjudication']['audit'].get('125398', {}).get('verdict')}`",
            f"- Sealed holdout candidates: `{results['adjudication']['selected_candidates']}`",
            f"- Overall verdict: `{results['adjudication']['overall_verdict']}`",
            "",
            "See `results/life-carrier-round-2/REPORT.md` and `LAY_SUMMARY.md`.",
            end,
        )
    )
    existing = path.read_text(encoding="utf-8") if path.exists() else "# Discovery log\n"
    if start in existing and end in existing:
        prefix = existing.split(start, 1)[0].rstrip()
        suffix = existing.split(end, 1)[1].lstrip()
        updated = prefix + "\n\n" + section + ("\n\n" + suffix if suffix else "\n")
    else:
        updated = existing.rstrip() + "\n\n" + section + "\n"
    _atomic_text(path, updated)


def run_life_carrier_campaign(
    output: Path,
    *,
    life_atlas: Path,
    profile_name: str = "reference",
    workers: int = 20,
    max_hours: float = 48.0,
    resume: bool = False,
    selected_stages: Sequence[str] | None = None,
) -> dict[str, Any]:
    require_pinned_numpy()
    if profile_name not in CARRIER_PROFILES:
        raise ValueError(f"unknown carrier profile {profile_name!r}")
    if not life_atlas.exists():
        raise FileNotFoundError(f"frozen Life atlas not found: {life_atlas}")
    profile = CARRIER_PROFILES[profile_name]
    contract = CarrierContract(morphology_max_swaps=profile.morphology_max_swaps)
    panel = build_carrier_panel(life_atlas, profile)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    deadline = started + max_hours * 3600 if max_hours > 0 else None
    active = set(selected_stages or ("audit", "acquire", "screen", "seal", "holdout", "mapping", "adjudication"))
    _atomic_text(output / "RUN.pid", f"{os.getpid()}\n")

    def status(state: str, stage: str, **extra: Any) -> None:
        payload = {
            "state": state,
            "stage": stage,
            "profile": profile_name,
            "pid": os.getpid(),
            "started_unix": started,
            "updated_unix": time.time(),
            "elapsed_seconds": time.time() - started,
            "deadline_unix": deadline,
            **extra,
        }
        _atomic_json(output / "STATUS.json", payload)
        progress = f" {extra['completed']}/{extra['total']}" if "completed" in extra else ""
        print(f"[{state}] {stage}{progress}", flush=True)

    implementation_files = (
        Path(__file__),
        Path(__file__).with_name("causal_heredity.py"),
        Path(__file__).with_name("life_family.py"),
    )
    design = {
        "contract": contract.to_dict(),
        "profile": asdict(profile),
        "audit_rules": list(AUDIT_RULES),
        "screen_panel": panel,
        "protocol_sha256": _sha256(PROTOCOL_PATH),
        "life_atlas_sha256": _sha256(life_atlas),
        "implementation_sha256": {path.name: _sha256(path) for path in implementation_files},
    }
    design_digest = hashlib.sha256(
        json.dumps(design, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _atomic_json(output / "DESIGN.json", {**design, "design_digest": design_digest})
    _atomic_json(
        output / "MANIFEST.json",
        {
            "experiment": "life_carrier_round_2",
            "profile": profile_name,
            "design_digest": design_digest,
            "started_unix": started,
            "workers": workers,
            "max_hours": max_hours,
            "environment": {
                "python": sys.version,
                "numpy": np.__version__,
                "platform": platform.platform(),
            },
        },
    )
    completeness: dict[str, bool] = {}
    data: dict[str, list[dict[str, Any]]] = {}

    try:
        audit_entries = [
            {"checkpoint": f"rule-{rule:06d}", "rule": rule} for rule in AUDIT_RULES
        ]
        status("running", "audit_acquire")
        data["audit_acquire"], completeness["audit_acquire"] = _run_stage(
            output,
            "audit_acquire",
            audit_entries,
            _audit_acquire_task,
            contract,
            profile,
            design_digest=design_digest,
            workers=workers,
            resume=resume,
            execute="audit" in active,
            deadline=deadline,
            status=status,
        )
        audit_items: list[dict[str, Any]] = []
        audit_chunk_size = 1 if profile_name == "smoke" else 2 if profile_name == "pilot" else 8
        for acquisition in data["audit_acquire"]:
            for chunk, donors in enumerate(_chunks(acquisition.get("donors", []), audit_chunk_size)):
                audit_items.append(
                    {
                        "checkpoint": f"rule-{int(acquisition['entry']['rule']):06d}-chunk-{chunk:03d}",
                        "entry": acquisition["entry"],
                        "donors": donors,
                    }
                )
        status("running", "audit")
        data["audit"], completeness["audit"] = _run_stage(
            output,
            "audit",
            audit_items,
            _audit_task,
            contract,
            profile,
            design_digest=design_digest,
            workers=workers,
            resume=resume,
            execute="audit" in active,
            deadline=deadline,
            status=status,
        )

        acquire_items = [
            {"checkpoint": f"rule-{int(entry['rule']):06d}", **entry} for entry in panel
        ]
        status("running", "screen_acquire")
        data["screen_acquire"], completeness["screen_acquire"] = _run_stage(
            output,
            "screen_acquire",
            acquire_items,
            _pair_acquire_task,
            contract,
            profile,
            design_digest=design_digest,
            workers=workers,
            resume=resume,
            execute="acquire" in active,
            deadline=deadline,
            status=status,
        )
        screen_items = _screen_items(data["screen_acquire"], "screen", max(1, profile.screen_pairs))
        status("running", "screen")
        data["screen"], completeness["screen"] = _run_stage(
            output,
            "screen",
            screen_items,
            _pair_garden_task,
            contract,
            profile,
            design_digest=design_digest,
            workers=workers,
            resume=resume,
            execute="screen" in active,
            deadline=deadline,
            status=status,
        )
        audit_adjudication = _audit_adjudication(data["audit"], profile, contract)
        screen_adjudication = _screen_adjudication(data["screen"], profile, contract)
        seal_path = output / "CHILD_SELECTION.json"
        if completeness["screen"] and ("seal" in active or not seal_path.exists()):
            candidates = select_holdout_candidates(screen_adjudication, profile.max_candidates)
            _atomic_json(
                seal_path,
                {
                    "design_digest": design_digest,
                    "sealed_unix": time.time(),
                    "selection_rule": "eligible; decreasing crossover; ascending rule id",
                    "candidates": candidates,
                    "screen": screen_adjudication,
                },
            )
        elif seal_path.exists():
            sealed = json.loads(seal_path.read_text(encoding="utf-8"))
            if sealed.get("design_digest") != design_digest:
                raise ValueError("child-selection manifest has the wrong design digest")
            candidates = [int(value) for value in sealed["candidates"]]
        else:
            candidates = []
        completeness["seal"] = completeness["screen"] and seal_path.exists()

        holdout_acquire_items = [
            {
                "checkpoint": f"rule-{rule:06d}",
                "rule": rule,
                "holdout": True,
            }
            for rule in candidates
        ]
        status("running", "holdout_acquire")
        data["holdout_acquire"], completeness["holdout_acquire"] = _run_stage(
            output,
            "holdout_acquire",
            holdout_acquire_items,
            _pair_acquire_task,
            contract,
            profile,
            design_digest=design_digest,
            workers=workers,
            resume=resume,
            execute="holdout" in active,
            deadline=deadline,
            status=status,
        )
        holdout_chunk = 1 if profile_name == "smoke" else 2 if profile_name == "pilot" else 4
        holdout_items = _screen_items(data["holdout_acquire"], "holdout", holdout_chunk)
        status("running", "holdout")
        data["holdout"], completeness["holdout"] = _run_stage(
            output,
            "holdout",
            holdout_items,
            _pair_garden_task,
            contract,
            profile,
            design_digest=design_digest,
            workers=workers,
            resume=resume,
            execute="holdout" in active,
            deadline=deadline,
            status=status,
        )

        holdout_rows = _flatten(data["holdout"])
        mapping_candidates: list[int] = []
        for rule in candidates:
            crossovers, direction_a, direction_b, _ = _rule_crossovers(holdout_rows, rule, 48)
            summary = _summary(crossovers, profile, contract, "mapping-selection", rule)
            if (
                summary["mean"] is not None
                and summary["mean"] >= contract.holdout_crossover
                and summary["ci95"][0] is not None
                and summary["ci95"][0] > 0.0
                and direction_a
                and direction_b
                and float(np.mean(direction_a)) > 0.0
                and float(np.mean(direction_b)) > 0.0
            ):
                mapping_candidates.append(rule)
        mapping_seal = output / "MAPPING_SELECTION.json"
        if completeness["holdout"]:
            _atomic_json(
                mapping_seal,
                {
                    "design_digest": design_digest,
                    "sealed_unix": time.time(),
                    "candidates": mapping_candidates,
                },
            )
        completeness["mapping_seal"] = completeness["holdout"] and mapping_seal.exists()

        mapping_items: list[dict[str, Any]] = []
        by_rule_acquisition = {
            int(result["entry"]["rule"]): result for result in data["holdout_acquire"]
        }
        for rule in mapping_candidates:
            selected_pairs = by_rule_acquisition[rule].get("pairs", [])[: profile.mapping_pairs]
            for chunk, pairs in enumerate(_chunks(selected_pairs, 1 if profile_name != "reference" else 2)):
                mapping_items.append(
                    {
                        "checkpoint": f"rule-{rule:06d}-chunk-{chunk:03d}",
                        "entry": by_rule_acquisition[rule]["entry"],
                        "chunk": chunk,
                        "pairs": pairs,
                    }
                )
        status("running", "mapping")
        data["mapping"], completeness["mapping"] = _run_stage(
            output,
            "mapping",
            mapping_items,
            _mapping_task,
            contract,
            profile,
            design_digest=design_digest,
            workers=workers,
            resume=resume,
            execute="mapping" in active,
            deadline=deadline,
            status=status,
        )

        holdout_adjudication = _holdout_adjudication(
            data["holdout"], data["mapping"], profile, contract
        )
        verdicts = [values["verdict"] for values in holdout_adjudication.values()]
        overall = (
            "DURABLE_CAUSAL_CARRIER"
            if "DURABLE_CAUSAL_CARRIER" in verdicts
            else "FORM_SPECIFIC_TRANSIENT_CARRIER"
            if "FORM_SPECIFIC_TRANSIENT_CARRIER" in verdicts
            else "NO_CAUSAL_CARRIER_FOUND"
        )
        scheduled = (
            "audit_acquire",
            "audit",
            "screen_acquire",
            "screen",
            "seal",
            "holdout_acquire",
            "holdout",
            "mapping_seal",
            "mapping",
        )
        all_complete = all(completeness.get(stage, False) for stage in scheduled)
        results = {
            "experiment": "life_carrier_round_2",
            "profile": profile_name,
            "state": "complete" if all_complete else "partial_budget_exhausted",
            "design_digest": design_digest,
            "contract_digest": contract.digest,
            "started_unix": started,
            "completed_unix": time.time(),
            "elapsed_seconds": time.time() - started,
            "stage_completeness": completeness,
            "adjudication": {
                "audit": audit_adjudication,
                "screen": screen_adjudication,
                "selected_candidates": candidates,
                "mapping_candidates": mapping_candidates,
                "holdout": holdout_adjudication,
                "overall_verdict": overall,
            },
        }
        status("running", "adjudication")
        for stage in ("audit", "screen", "holdout", "mapping"):
            _write_rows_csv(output / stage / f"{stage}.csv", _flatten(data[stage]))
        _atomic_json(output / "RESULTS.json", results)
        _atomic_text(output / "REPORT.md", _render_report(results))
        _atomic_text(output / "LAY_SUMMARY.md", _render_lay_summary(results))
        if all_complete:
            _atomic_text(output / "COMPLETE", "complete\n")
            partial = output / "PARTIAL"
            if partial.exists():
                partial.unlink()
            if profile_name == "reference":
                _update_discovery_log(results, Path("DISCOVERY_LOG_EIDOSOMA_SCIENTIST.md"))
            status("complete", "campaign")
        else:
            _atomic_text(output / "PARTIAL", "budget exhausted or selected dependency missing; resume supported\n")
            status("partial_budget_exhausted", "campaign")
        return results
    except BaseException as error:
        status("failed", "campaign", error=repr(error))
        raise


__all__ = [
    "AUDIT_RULES",
    "CARRIER_PROFILES",
    "CarrierContract",
    "CarrierProfile",
    "build_carrier_panel",
    "launch_detached",
    "run_life_carrier_campaign",
    "select_holdout_candidates",
]
