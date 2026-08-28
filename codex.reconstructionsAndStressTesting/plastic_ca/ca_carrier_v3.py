"""Round-3 continuous-form carrier tests across Life-like CA and ECA.

This module deliberately has its own contract and RNG namespace.  Round-2
artifacts are development inputs only: they freeze a narrow continuous
hypothesis and outcome-blind control tolerances, never confirmation outcomes.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .causal_heredity import (
    BatchTrace,
    CausalContract,
    _atomic_json,
    _atomic_text,
    _bootstrap_summary,
    _component_spectrum,
    _copy_batch,
    _cosine,
    _donor_record,
    _hash_seed,
    _observer_vectors,
    _sha256,
    _simulate_batch,
    _site_mask,
    _state_from_hex,
    _state_to_hex,
    _strict_event,
    _write_rows_csv,
    launch_detached,
)
from .e19 import LAUNCH_HEX, _hex_to_row, figure_states, final4_counts, require_pinned_numpy
from .life_carrier import (
    _block_shuffle,
    _holm,
    _live_neighbor_hist,
    _replace_mask,
    _transform_board,
)
from .life_family import (
    LifeFamilyContract,
    launch_library,
    life_rule_notation,
    load_rule_registry,
)
from .particle_e19 import build_e19_domain_dictionary


PROTOCOL_PATH = Path(__file__).resolve().parent.parent / "CA_CARRIER_V3_PROTOCOL.md"
ROUND2_ACQUISITION = (
    Path(__file__).resolve().parent.parent
    / "results/life-carrier-round-2/screen_acquire/checkpoints/rule-031649.json"
)
ROUND2_RULE = 31649


@dataclass(frozen=True)
class V3Contract:
    implementation_version: str = "ca-carrier-v3-cleanroom-v1"
    namespace: str = "plastic-ca-carrier-v3"
    life_width: int = 16
    eca_width: int = 64
    activity_budget: int = 48
    min_sweeps: int = 4
    max_sweeps: int = 64
    life_process_noise: float = 0.002
    life_copy_error: float = 0.005
    eca_process_noise: float = 0.01
    eca_copy_error: float = 0.015
    donor_horizon: int = 32
    assignment_similarity: float = 0.90
    assignment_margin: float = 0.05
    prototype_similarity: float = 0.95
    prototype_margin: float = 0.05
    cluster_similarity: float = 0.95
    cluster_minimum: int = 16
    cluster_launches: int = 2
    pair_separation: float = 0.80
    narrow_density_tolerance: float = 0.02
    wide_density_tolerance: float = 0.05
    primary_crossover: float = 0.15
    local_validator_crossover: float = 0.10
    control_advantage: float = 0.10
    survival_gate: float = 0.90
    doses: tuple[float, ...] = (0.0625, 0.125, 0.25, 0.5, 0.75, 1.0)
    checkpoints: tuple[int, ...] = (1, 8, 16, 32, 64)
    pedigree_depth: int = 8

    def causal(
        self,
        suffix: str,
        *,
        substrate: str = "life",
        extent: int | None = None,
    ) -> CausalContract:
        width = self.life_width if extent is None else extent
        if substrate == "life":
            budget = self.activity_budget if width == self.life_width else 4 * width
            maximum = self.max_sweeps if width == self.life_width else 4 * width
            return CausalContract(
                implementation_version=self.implementation_version,
                namespace=f"{self.namespace}:{suffix}",
                recovery_horizon=max(self.checkpoints),
                life_width=width,
                life_height=width,
                life_activity_budget=budget,
                life_min_sweeps=self.min_sweeps,
                life_max_sweeps=maximum,
                life_process_noise=self.life_process_noise,
                life_copy_error=self.life_copy_error,
                pedigree_depth=self.pedigree_depth,
            )
        return CausalContract(
            implementation_version=self.implementation_version,
            namespace=f"{self.namespace}:{suffix}",
            recovery_horizon=max(self.checkpoints),
            eca_width=self.eca_width,
            eca_activity_budget=4 * self.eca_width,
            eca_min_sweeps=4,
            eca_max_sweeps=128,
            eca_process_noise=self.eca_process_noise,
            eca_copy_error=self.eca_copy_error,
            pedigree_depth=self.pedigree_depth,
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.update(
            {
                "narrow_hypothesis": "continuous prototype pair; never pooled support IDs",
                "primary_unit": "matched donor pair",
                "missing_policy": "dead, invalid, and unresolved futures remain in denominator",
                "generation_policy": "alive and valid at the named completed generation",
            }
        )
        return value

    @property
    def digest(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class V3Profile:
    calibration_candidates: int
    morphology_keep: int
    narrow_per_launch: int
    replay_pairs: int
    replay_replicates: int
    confirmation_pairs: int
    confirmation_replicates: int
    transfer_pairs: int
    transfer_replicates: int
    mapping_pairs: int
    mapping_replicates: int
    initial_life_cap: int
    initial_eca_cap: int
    retained_donors: int
    extend_life_rules: int
    extend_eca_rules: int
    extend_cap: int
    screen_life_candidates: int
    screen_eca_candidates: int
    screen_pairs: int
    screen_replicates: int
    holdout_candidates: int
    holdout_pairs: int
    holdout_replicates: int
    bootstrap_resamples: int


V3_PROFILES: dict[str, V3Profile] = {
    "smoke": V3Profile(
        64, 8, 64, 1, 2, 2, 2, 1, 2, 2, 2,
        8, 8, 16, 2, 2, 32, 1, 1, 1, 2, 1, 2, 2, 100,
    ),
    "pilot": V3Profile(
        512, 16, 2_048, 4, 16, 8, 16, 4, 8, 8, 8,
        256, 512, 64, 8, 8, 4_096, 4, 4, 4, 8, 2, 8, 16, 1_000,
    ),
    "reference": V3Profile(
        4_096, 32, 32_768, 16, 256, 64, 128, 32, 64, 32, 32,
        2_048, 4_096, 256, 64, 32, 32_768, 32, 16, 16, 32, 8, 64, 128, 10_000,
    ),
}


def _normalize_vector(vector: Sequence[float] | np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    total = float(value.sum())
    return value / total if total > 0.0 else value


def _cyclic_pattern_counts(states: np.ndarray, k: int) -> np.ndarray:
    """Normalized cyclic binary-word census for 1-D or flattened 2-D states."""

    values = np.asarray(states, dtype=np.bool_)
    if values.ndim != 2:
        raise ValueError("cyclic pattern states must be shaped (sample, width)")
    codes = np.zeros(values.shape, dtype=np.uint16)
    for bit in range(k):
        codes |= np.roll(values, -bit, axis=1).astype(np.uint16) << (k - bit - 1)
    result = np.zeros((len(values), 1 << k), dtype=np.float64)
    for code in range(1 << k):
        result[:, code] = np.count_nonzero(codes == code, axis=1)
    totals = result.sum(axis=1, keepdims=True)
    return np.divide(result, totals, out=np.zeros_like(result), where=totals > 0)


def _life_3x3_counts(states: np.ndarray) -> np.ndarray:
    values = np.asarray(states, dtype=np.bool_)
    if values.ndim != 3:
        raise ValueError("Life states must be shaped (sample, height, width)")
    codes = np.zeros(values.shape, dtype=np.uint16)
    bit = 8
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            codes |= np.roll(values, shift=(-dy, -dx), axis=(1, 2)).astype(np.uint16) << bit
            bit -= 1
    result = np.zeros((len(values), 512), dtype=np.float64)
    for code in range(512):
        result[:, code] = np.count_nonzero(codes == code, axis=(1, 2))
    totals = result.sum(axis=1, keepdims=True)
    return np.divide(result, totals, out=np.zeros_like(result), where=totals > 0)


def _life_autocorrelation(states: np.ndarray) -> np.ndarray:
    values = np.asarray(states, dtype=np.bool_)
    offsets = (
        (0, 1), (1, 0), (1, 1), (1, -1),
        (0, 2), (2, 0), (2, 2), (2, -2),
        (0, 4), (4, 0), (4, 4), (4, -4),
    )
    rows: list[np.ndarray] = []
    for dy, dx in offsets:
        shifted = np.roll(values, shift=(dy, dx), axis=(1, 2))
        p11 = np.mean(values & shifted, axis=(1, 2))
        p00 = np.mean(~values & ~shifted, axis=(1, 2))
        mismatch = 1.0 - p11 - p00
        rows.extend((p11, mismatch, p00))
    return np.stack(rows, axis=1)


def _component_sizes(board: np.ndarray, value: bool) -> list[int]:
    target = np.asarray(board, dtype=np.bool_) == value
    height, width = target.shape
    unseen = set(map(tuple, np.argwhere(target)))
    sizes: list[int] = []
    while unseen:
        root = unseen.pop()
        queue: deque[tuple[int, int]] = deque((root,))
        size = 0
        while queue:
            y, x = queue.popleft()
            size += 1
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if not (dy or dx):
                        continue
                    candidate = ((y + dy) % height, (x + dx) % width)
                    if candidate in unseen:
                        unseen.remove(candidate)
                        queue.append(candidate)
        sizes.append(size)
    return sorted(sizes, reverse=True)


def _size_bins(sizes: Sequence[int]) -> np.ndarray:
    result = np.zeros(8, dtype=np.float64)
    for size in sizes:
        index = (
            0 if size == 1 else 1 if size == 2 else 2 if size <= 4 else
            3 if size <= 8 else 4 if size <= 16 else 5 if size <= 32 else
            6 if size <= 64 else 7
        )
        result[index] += 1.0
    total = result.sum()
    return result / total if total else result


def _life_global(states: np.ndarray) -> np.ndarray:
    rows: list[np.ndarray] = []
    for board in np.asarray(states, dtype=np.bool_):
        live = _component_sizes(board, True)
        dead = _component_sizes(board, False)
        cells = board.size
        geometry = np.concatenate(
            (
                _size_bins(live),
                _size_bins(dead),
                np.asarray(
                    (
                        float(board.mean()),
                        (live[0] / cells) if live else 0.0,
                        (dead[0] / cells) if dead else 0.0,
                        len(live) / cells,
                        len(dead) / cells,
                    )
                ),
            )
        )
        rows.append(geometry)
    return np.stack(rows)


def _structure_factor(states: np.ndarray) -> np.ndarray:
    values = np.asarray(states, dtype=np.float64)
    rows: list[np.ndarray] = []
    if values.ndim == 2:
        for row in values:
            centered = row - row.mean()
            power = np.abs(np.fft.rfft(centered)) ** 2
            rows.append(_normalize_vector(power[1:9]))
    elif values.ndim == 3:
        coordinates = ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (0, 2), (2, 0), (2, 2), (3, 0), (0, 3), (3, 1), (1, 3))
        for board in values:
            centered = board - board.mean()
            power = np.abs(np.fft.fft2(centered)) ** 2
            rows.append(_normalize_vector([power[y, x] for y, x in coordinates]))
    else:
        raise ValueError("structure-factor states must be 1-D or 2-D batches")
    return np.stack(rows)


def _run_length_spectrum(states: np.ndarray) -> np.ndarray:
    rows: list[np.ndarray] = []
    for row in np.asarray(states, dtype=np.bool_):
        if bool(np.all(row)) or bool(np.all(~row)):
            vector = np.zeros(18, dtype=np.float64)
            vector[8 if bool(row[0]) else 17] = 1.0
            rows.append(vector)
            continue
        boundary = next(index for index in range(len(row)) if row[index] != row[index - 1])
        rotated = np.roll(row, -boundary)
        counts = np.zeros((2, 9), dtype=np.float64)
        start = 0
        for index in range(1, len(rotated) + 1):
            if index < len(rotated) and rotated[index] == rotated[start]:
                continue
            length = index - start
            counts[int(rotated[start]), min(length, 9) - 1] += 1.0
            start = index
        rows.append(_normalize_vector(counts.ravel()))
    return np.stack(rows)


@lru_cache(maxsize=256)
def _eca_domain_codes(rule: int) -> frozenset[int]:
    return build_e19_domain_dictionary(rule).codes


def _eca_mesoscale(states: np.ndarray, rule: int) -> np.ndarray:
    """Figure/ground final-4 census from three consecutive terminal rows."""

    values = np.asarray(states, dtype=np.bool_)
    if values.ndim != 3 or values.shape[1] != 3:
        raise ValueError("ECA mesoscale input must have shape (sample, 3, width)")
    figures = figure_states((values[:, 0], values[:, 1], values[:, 2]), _eca_domain_codes(rule))
    counts = final4_counts(figures).astype(np.float64)
    totals = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)


def extra_observer_vectors(substrate: str, states: np.ndarray) -> dict[str, np.ndarray]:
    """Independent local and global observer families used after discovery."""

    if substrate == "life":
        return {
            "local_secondary": _life_3x3_counts(states),
            "local_aux": _life_autocorrelation(states),
            "global": np.concatenate((_life_global(states), _structure_factor(states)), axis=1),
        }
    if substrate == "eca":
        multiscale = np.concatenate(
            tuple(_cyclic_pattern_counts(states, k) / 2.0 for k in range(5, 9)), axis=1
        )
        return {
            "local_secondary": multiscale,
            "local_aux": _run_length_spectrum(states),
            "global": _structure_factor(states),
        }
    raise ValueError(f"unknown substrate {substrate!r}")


def _augment_donor(donor: dict[str, Any], established_states: np.ndarray) -> dict[str, Any]:
    result = dict(donor)
    targets = dict(result["target_compositions"])
    for name, vectors in extra_observer_vectors(str(result["substrate"]), established_states).items():
        targets[name] = np.mean(vectors, axis=0).astype(float).tolist()
    terminal_key = "terminal2x2" if result["substrate"] == "life" else "raw4"
    targets["primary_terminal"] = np.mean(
        _observer_vectors(str(result["substrate"]), established_states)[terminal_key], axis=0
    ).astype(float).tolist()
    if result["substrate"] == "eca" and len(established_states) >= 3:
        windows = np.stack(
            [established_states[index - 2 : index + 1] for index in range(2, len(established_states))]
        )
        targets["mesoscale"] = np.mean(
            _eca_mesoscale(windows, int(result["rule"])), axis=0
        ).astype(float).tolist()
    result["target_compositions"] = targets
    state = _state_from_hex(str(result["substrate"]), str(result["donor_state_hex"]))
    result["density"] = float(np.mean(state))
    result.pop("generation_sweeps", None)
    result.pop("generation_activity", None)
    return result


def _donor_targets_from_state(donor: dict[str, Any]) -> dict[str, list[float]]:
    substrate = str(donor["substrate"])
    state = _state_from_hex(substrate, str(donor["donor_state_hex"]))
    extras = extra_observer_vectors(substrate, state[None, ...])
    targets = {name: values[0].astype(float).tolist() for name, values in extras.items()}
    targets["primary"] = list(donor["target_compositions"]["primary"])
    terminal_key = "terminal2x2" if substrate == "life" else "raw4"
    targets["primary_terminal"] = _observer_vectors(substrate, state[None, ...])[terminal_key][0].astype(float).tolist()
    return targets


def _rounded_outward(value: float, quantum: float = 0.01) -> float:
    return math.ceil((value - 1e-15) / quantum) * quantum


def _masked_random(shape: tuple[int, ...], mask: np.ndarray, live: int, key: str) -> np.ndarray:
    positions = list(map(int, np.flatnonzero(mask.ravel())))
    positions.sort(key=lambda index: hashlib.sha256(f"{key}:{index}".encode()).digest())
    result = np.zeros(math.prod(shape), dtype=np.bool_)
    result[np.asarray(positions[: max(0, min(live, len(positions)))], dtype=int)] = True
    return result.reshape(shape)


def _morphology_distances(substrate: str, candidate: np.ndarray, target: np.ndarray) -> dict[str, float]:
    if substrate == "life":
        neighbor = float(np.abs(_live_neighbor_hist(candidate) - _live_neighbor_hist(target)).sum())
        component = _cosine(_component_spectrum(candidate), _component_spectrum(target))
    else:
        neighbor = float(
            np.abs(_cyclic_pattern_counts(candidate[None, :], 3)[0] - _cyclic_pattern_counts(target[None, :], 3)[0]).sum()
        )
        component = _cosine(_run_length_spectrum(candidate[None, :])[0], _run_length_spectrum(target[None, :])[0])
    structure = float(
        np.abs(_structure_factor(candidate[None, ...])[0] - _structure_factor(target[None, ...])[0]).sum()
    )
    return {"neighbor_error": neighbor, "component_cosine": component, "structure_error": structure}


def conditional_null_ensemble(
    substrate: str,
    source: np.ndarray,
    mask: np.ndarray,
    key: str,
    *,
    candidates: int,
    keep: int,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Return an always-present exact-count conditional morphology null."""

    target = np.asarray(source, dtype=np.bool_) & mask
    live = int(target.sum())
    ranked: list[tuple[float, str, np.ndarray, dict[str, float]]] = []
    for index in range(candidates):
        candidate = _masked_random(tuple(source.shape), mask, live, f"{key}:null:{index}")
        distances = _morphology_distances(substrate, candidate, target)
        score = (
            distances["neighbor_error"]
            + (1.0 - distances["component_cosine"])
            + distances["structure_error"]
        )
        digest = hashlib.sha256(candidate.tobytes() + str(index).encode()).hexdigest()
        ranked.append((score, digest, candidate, distances))
    ranked.sort(key=lambda row: (row[0], row[1]))
    selected = ranked[: min(keep, len(ranked))]
    if not selected:
        selected = [(0.0, "degenerate", target.copy(), _morphology_distances(substrate, target, target))]
    metadata = {
        "generated": candidates,
        "retained": len(selected),
        "target_live": live,
        "best": selected[0][3],
        "mean": {
            name: float(np.mean([row[3][name] for row in selected]))
            for name in ("neighbor_error", "component_cosine", "structure_error")
        },
    }
    return [row[2] for row in selected], metadata


def _prototype_label(vector: Sequence[float], target_a: Sequence[float], target_b: Sequence[float], contract: V3Contract) -> str | None:
    similarity_a = _cosine(vector, target_a)
    similarity_b = _cosine(vector, target_b)
    if max(similarity_a, similarity_b) < contract.prototype_similarity:
        return None
    if abs(similarity_a - similarity_b) < contract.prototype_margin:
        return None
    return "A" if similarity_a > similarity_b else "B"


def select_narrow_prototype(round2_path: Path = ROUND2_ACQUISITION) -> dict[str, Any]:
    payload = json.loads(round2_path.read_text(encoding="utf-8"))["result"]
    pair = min(payload["pairs"], key=lambda row: (float(row["target_similarity"]), str(row["pair_id"])))
    donor_a = dict(pair["donor_a"])
    donor_b = dict(pair["donor_b"])
    donor_a.setdefault("substrate", "life")
    donor_b.setdefault("substrate", "life")
    targets_a = _donor_targets_from_state(donor_a)
    targets_b = _donor_targets_from_state(donor_b)
    pooled: dict[str, Any] = {}
    for form in (2868, 3892):
        vectors = [
            np.asarray(donor["target_compositions"]["primary"], dtype=float)
            for donor in payload["donors"]
            if int(donor.get("form_id", -1)) == form
        ]
        pooled[str(form)] = {
            "n": len(vectors),
            "centroid": np.mean(vectors, axis=0).astype(float).tolist() if vectors else [],
        }
    pooled_cosine = _cosine(pooled["2868"]["centroid"], pooled["3892"]["centroid"])
    contract = V3Contract()
    match_counts: Counter[str] = Counter()
    matches_by_launch: dict[str, Counter[str]] = defaultdict(Counter)
    for donor in payload["donors"]:
        label = _prototype_label(
            donor["target_compositions"]["primary"],
            targets_a["primary"],
            targets_b["primary"],
            contract,
        )
        match_counts[label or "unmatched"] += 1
        matches_by_launch[str(donor["launch_index"])][label or "unmatched"] += 1
    return {
        "rule": ROUND2_RULE,
        "pair_id": str(pair["pair_id"]),
        "target_similarity": float(pair["target_similarity"]),
        "donor_a": donor_a,
        "donor_b": donor_b,
        "targets": {
            name: {"A": targets_a[name], "B": targets_b[name]}
            for name in ("primary", "primary_terminal", "local_secondary", "local_aux", "global")
        },
        "target_basis": {
            "primary": "retained eight-generation established centroid",
            "primary_terminal": "retained donor terminal state",
            "local_secondary": "retained donor terminal state; historical eight terminals unavailable",
            "local_aux": "retained donor terminal state; historical eight terminals unavailable",
            "global": "retained donor terminal state; historical eight terminals unavailable",
        },
        "pooled_support_id_diagnostic": {**pooled, "centroid_cosine": pooled_cosine},
        "historical_prototype_match_diagnostic": {
            "counts": dict(sorted(match_counts.items())),
            "by_launch": {
                launch: dict(sorted(counts.items()))
                for launch, counts in sorted(matches_by_launch.items())
            },
            "confirmatory_use": False,
        },
    }


def calibrate_controls(
    prototype: dict[str, Any], profile: V3Profile, contract: V3Contract
) -> dict[str, Any]:
    payload = json.loads(ROUND2_ACQUISITION.read_text(encoding="utf-8"))["result"]
    distances: list[dict[str, float]] = []
    for pair in payload["pairs"]:
        for label in ("a", "b"):
            donor = pair[f"donor_{label}"]
            source = _state_from_hex("life", str(donor["donor_state_hex"]))
            mask = _site_mask(source.shape, 0.5, "square", f"calibration:{pair['pair_id']}")
            _, metadata = conditional_null_ensemble(
                "life",
                source,
                mask,
                f"calibration:{pair['pair_id']}:{label}",
                candidates=profile.calibration_candidates,
                keep=profile.morphology_keep,
            )
            distances.append(dict(metadata["best"]))
    calipers = {
        "neighbor_error": _rounded_outward(float(np.quantile([row["neighbor_error"] for row in distances], 0.9))),
        "component_cosine_min": math.floor(float(np.quantile([row["component_cosine"] for row in distances], 0.1)) * 100.0) / 100.0,
        "structure_error": _rounded_outward(float(np.quantile([row["structure_error"] for row in distances], 0.9))),
    }
    return {
        "selection_basis": "minimum pre-transplant continuous-target cosine; pair-id tie break",
        "outcome_data_used": False,
        "prototype": prototype,
        "conditional_null": {
            "candidates": profile.calibration_candidates,
            "keep": profile.morphology_keep,
            "historical_fragments": len(distances),
            "calipers": calipers,
        },
        "contract_digest": contract.digest,
    }


def continuous_clusters(
    donors: Sequence[dict[str, Any]],
    observer: str,
    *,
    threshold: float = 0.95,
) -> list[dict[str, Any]]:
    """Deterministic complete-linkage-constrained partition.

    Sorting by donor ID makes the result independent of input order.  A donor
    joins the compatible cluster with the largest worst-member similarity.
    """

    clusters: list[list[dict[str, Any]]] = []
    for donor in sorted(donors, key=lambda row: str(row["donor_id"])):
        vector = np.asarray(donor["target_compositions"][observer], dtype=float)
        compatible: list[tuple[float, str, int]] = []
        for index, members in enumerate(clusters):
            similarities = [
                _cosine(vector, member["target_compositions"][observer]) for member in members
            ]
            if similarities and min(similarities) >= threshold:
                compatible.append((min(similarities), str(members[0]["donor_id"]), index))
        if compatible:
            _, _, selected = max(compatible, key=lambda row: (row[0], tuple(-ord(ch) for ch in row[1])))
            clusters[selected].append(donor)
        else:
            clusters.append([donor])
    result: list[dict[str, Any]] = []
    for members in clusters:
        targets: dict[str, list[float]] = {}
        names = sorted(set.intersection(*(set(member["target_compositions"]) for member in members)))
        for name in names:
            targets[name] = np.mean(
                [member["target_compositions"][name] for member in members], axis=0
            ).astype(float).tolist()
        member_ids = sorted(str(member["donor_id"]) for member in members)
        result.append(
            {
                "cluster_id": hashlib.sha256("|".join(member_ids).encode()).hexdigest()[:16],
                "size": len(members),
                "launches": sorted({int(member["launch_index"]) for member in members}),
                "member_ids": member_ids,
                "targets": targets,
                "members": members,
            }
        )
    return sorted(result, key=lambda row: (-int(row["size"]), str(row["cluster_id"])))


def _density_match(
    donors_a: Sequence[dict[str, Any]],
    donors_b: Sequence[dict[str, Any]],
    tolerance: float,
) -> list[tuple[float, dict[str, Any], dict[str, Any]]]:
    """Maximum-cardinality deterministic matching for an interval caliper."""

    left = sorted(donors_a, key=lambda row: (float(row["density"]), str(row["donor_id"])))
    right = sorted(donors_b, key=lambda row: (float(row["density"]), str(row["donor_id"])))
    pairs: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    i = j = 0
    while i < len(left) and j < len(right):
        density_a = float(left[i]["density"])
        density_b = float(right[j]["density"])
        if abs(density_a - density_b) <= tolerance + 1e-12:
            pairs.append((abs(density_a - density_b), left[i], right[j]))
            i += 1
            j += 1
        elif density_a < density_b:
            i += 1
        else:
            j += 1
    return pairs


def _pair_members(
    left: dict[str, Any], right: dict[str, Any], tolerance: float, rule: int, family: str
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    targets = {
        name: {"A": left["targets"][name], "B": right["targets"][name]}
        for name in sorted(set(left["targets"]) & set(right["targets"]))
    }
    launches = sorted(
        {int(donor["launch_index"]) for donor in left["members"]}
        | {int(donor["launch_index"]) for donor in right["members"]}
    )
    for launch in launches:
        matches = _density_match(
            [donor for donor in left["members"] if int(donor["launch_index"]) == launch],
            [donor for donor in right["members"] if int(donor["launch_index"]) == launch],
            tolerance,
        )
        for delta, donor_a, donor_b in matches:
            pair_id = f"{donor_a['substrate']}-{rule}-{family}-{left['cluster_id']}-{right['cluster_id']}-{len(pairs):04d}"
            pairs.append(
                {
                    "pair_id": pair_id,
                    "substrate": str(donor_a["substrate"]),
                    "rule": rule,
                    "family": family,
                    "launch_index": launch,
                    "density_delta": delta,
                    "donor_a": donor_a,
                    "donor_b": donor_b,
                    "targets": targets,
                }
            )
    return pairs


def discover_continuous_candidates(
    donors: Sequence[dict[str, Any]], contract: V3Contract
) -> dict[str, Any]:
    if not donors:
        return {"families": {}, "best_score": [0, 0, 0.0, 0], "best_candidate": None}
    substrate = str(donors[0]["substrate"])
    rule = int(donors[0]["rule"])
    tolerance = contract.narrow_density_tolerance if rule == ROUND2_RULE and substrate == "life" else contract.wide_density_tolerance
    families: dict[str, Any] = {}
    best: tuple[tuple[int, int, float, int], dict[str, Any]] | None = None
    for family, observer in (("local", "primary"), ("global", "global")):
        clusters = continuous_clusters(donors, observer, threshold=contract.cluster_similarity)
        candidates: list[dict[str, Any]] = []
        for left_index, left in enumerate(clusters):
            for right in clusters[left_index + 1 :]:
                similarity = _cosine(left["targets"][observer], right["targets"][observer])
                if similarity > contract.pair_separation:
                    continue
                pairs = _pair_members(left, right, tolerance, rule, family)
                score = (
                    len(pairs),
                    min(int(left["size"]), int(right["size"])),
                    1.0 - similarity,
                    len(set(left["launches"]) | set(right["launches"])),
                )
                candidate = {
                    "candidate_id": f"{substrate}-{rule:06d}-{family}-{left['cluster_id']}-{right['cluster_id']}",
                    "substrate": substrate,
                    "rule": rule,
                    "family": family,
                    "observer": observer,
                    "cluster_a": {key: value for key, value in left.items() if key != "members"},
                    "cluster_b": {key: value for key, value in right.items() if key != "members"},
                    "centroid_similarity": similarity,
                    "score": list(score),
                    "pair_count": len(pairs),
                    "pairs": pairs[:128],
                }
                candidates.append(candidate)
                if best is None or score > best[0] or (score == best[0] and candidate["candidate_id"] < best[1]["candidate_id"]):
                    best = (score, candidate)
                # A many-singleton rule can have tens of thousands of cluster
                # pairs.  Keep a bounded top set while preserving the exact
                # best score and full donor catalogue in the parent result.
                if len(candidates) > 64:
                    candidates.sort(
                        key=lambda row: (
                            tuple(-float(value) for value in row["score"]),
                            str(row["candidate_id"]),
                        )
                    )
                    del candidates[16:]
        families[family] = {
            "cluster_sizes": [int(cluster["size"]) for cluster in clusters],
            "qualified_clusters": sum(
                int(cluster["size"] >= contract.cluster_minimum and len(cluster["launches"]) >= contract.cluster_launches)
                for cluster in clusters
            ),
            "candidates": sorted(candidates, key=lambda row: (tuple(-float(x) for x in row["score"]), row["candidate_id"]))[:16],
        }
    return {
        "families": families,
        "best_score": list(best[0]) if best else [0, 0, 0.0, 0],
        "best_candidate": (
            {
                key: value
                for key, value in best[1].items()
                if key not in ("pairs", "cluster_a", "cluster_b")
            }
            if best else None
        ),
    }


def _launches(substrate: str, contract: V3Contract, *, extent: int | None = None) -> tuple[np.ndarray, ...]:
    if substrate == "eca":
        return tuple(_hex_to_row(value) for value in LAUNCH_HEX)
    causal = contract.causal("launches", substrate="life", extent=extent)
    return launch_library(
        LifeFamilyContract(
            width=causal.life_width,
            height=causal.life_height,
            activity_budget=causal.life_activity_budget,
            min_sweeps=causal.life_min_sweeps,
            max_sweeps=causal.life_max_sweeps,
            flip_noise=causal.life_process_noise,
            copy_error=causal.life_copy_error,
            futures_per_launch=1,
        )
    )


def _discover_donors(
    *,
    substrate: str,
    rule: int,
    contract: V3Contract,
    namespace: str,
    cap: int,
    retained: int,
    launch_only: int | None = None,
    prototype: dict[str, Any] | None = None,
    analyze: bool = True,
) -> dict[str, Any]:
    causal = contract.causal(namespace, substrate=substrate)
    launches = _launches(substrate, contract)
    donors: list[dict[str, Any]] = []
    strict_seen = 0
    density_rejected = 0
    unmatched = 0
    deaths: Counter[str] = Counter()
    batch_size = 256
    for start in range(0, cap, batch_size):
        size = min(batch_size, cap - start)
        if launch_only is None:
            launch_indices = [(start + local) % len(launches) for local in range(size)]
        else:
            launch_indices = [launch_only] * size
        initial = np.stack([launches[index] for index in launch_indices])
        trace = _simulate_batch(
            substrate,
            rule,
            initial,
            causal,
            horizon=contract.donor_horizon,
            rng_seed=_hash_seed(contract.namespace, namespace, substrate, rule, launch_only, start),
            observer="raw" if substrate == "eca" else "primary",
        )
        deaths.update(reason for reason in trace.death if reason is not None)
        for local in range(size):
            length = int(np.count_nonzero(trace.valid[local]))
            if length < 10:
                continue
            event = _strict_event(trace.compositions[local, :length], causal.thresholds)
            if event is None:
                continue
            strict_seen += 1
            if len(donors) >= retained:
                continue
            record = _donor_record(
                substrate,
                rule,
                "raw" if substrate == "eca" else "primary",
                launch_indices[local],
                start + local,
                initial[local],
                trace,
                local,
                "switcher",
                event,
            )
            states = trace.terminals[local, record["target_generation_indices"]]
            donor = _augment_donor(record, states)
            if not 0.05 <= float(donor["density"]) <= 0.95:
                density_rejected += 1
                continue
            if prototype is not None:
                label = _prototype_label(
                    donor["target_compositions"]["primary"],
                    prototype["targets"]["primary"]["A"],
                    prototype["targets"]["primary"]["B"],
                    contract,
                )
                if label is None:
                    unmatched += 1
                    continue
                donor["prototype_label"] = label
            donors.append(donor)
    discovery = discover_continuous_candidates(donors, contract) if prototype is None and analyze else None
    return {
        "entry": {
            "substrate": substrate,
            "rule": rule,
            "notation": life_rule_notation(rule) if substrate == "life" else None,
        },
        "namespace": namespace,
        "launch_only": launch_only,
        "examined": cap,
        "strict_seen": strict_seen,
        "retained": len(donors),
        "retention_cap": retained,
        "density_rejected": density_rejected,
        "prototype_unmatched": unmatched,
        "death_counts": dict(sorted(deaths.items())),
        "donors": donors,
        "discovery": discovery,
    }


def _narrow_acquire_task(arguments: tuple[dict[str, Any], V3Contract, V3Profile]) -> dict[str, Any]:
    item, contract, profile = arguments
    return _discover_donors(
        substrate="life",
        rule=ROUND2_RULE,
        contract=contract,
        namespace="narrow-fresh-acquisition",
        cap=profile.narrow_per_launch,
        retained=max(512, profile.confirmation_pairs + profile.transfer_pairs + profile.mapping_pairs),
        launch_only=int(item["launch_index"]),
        prototype=item["prototype"],
    )


def _wide_discover_task(arguments: tuple[dict[str, Any], V3Contract, V3Profile]) -> dict[str, Any]:
    item, contract, profile = arguments
    return _discover_donors(
        substrate=str(item["substrate"]),
        rule=int(item["rule"]),
        contract=contract,
        namespace=str(item["namespace"]),
        cap=int(item["cap"]),
        retained=profile.retained_donors,
    )


def pair_prototype_donors(
    acquisitions: Sequence[dict[str, Any]], prototype: dict[str, Any], contract: V3Contract
) -> list[dict[str, Any]]:
    by_launch_label: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for result in acquisitions:
        for donor in result.get("donors", []):
            by_launch_label[(int(donor["launch_index"]), str(donor["prototype_label"]))].append(donor)
    pairs: list[dict[str, Any]] = []
    for launch in sorted({key[0] for key in by_launch_label}):
        matches = _density_match(
            by_launch_label.get((launch, "A"), []),
            by_launch_label.get((launch, "B"), []),
            contract.narrow_density_tolerance,
        )
        for delta, donor_a, donor_b in matches:
            id_a = str(donor_a["donor_id"])
            id_b = str(donor_b["donor_id"])
            pair_id = f"narrow-{len(pairs):04d}-{id_a}-{id_b}"
            pairs.append(
                {
                    "pair_id": pair_id,
                    "substrate": "life",
                    "rule": ROUND2_RULE,
                    "family": "local",
                    "launch_index": launch,
                    "density_delta": delta,
                    "donor_a": donor_a,
                    "donor_b": donor_b,
                    "targets": prototype["targets"],
                }
            )
    # The density-optimal matching above is scientific; this hash ordering is
    # the preregistered cohort assignment and is independent of outcomes.
    return sorted(
        pairs,
        key=lambda pair: hashlib.sha256(f"{contract.namespace}:cohort:{pair['pair_id']}".encode()).hexdigest(),
    )


def _legacy_pairs(prototype: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    payload = json.loads(ROUND2_ACQUISITION.read_text(encoding="utf-8"))["result"]
    result: list[dict[str, Any]] = []
    for old in payload["pairs"][:limit]:
        donor_a = dict(old["donor_a"])
        donor_b = dict(old["donor_b"])
        for donor in (donor_a, donor_b):
            donor.setdefault("substrate", "life")
            donor.setdefault("density", float(np.mean(_state_from_hex("life", donor["donor_state_hex"]))))
        targets_a = _donor_targets_from_state(donor_a)
        targets_b = _donor_targets_from_state(donor_b)
        result.append(
            {
                "pair_id": f"replay-{old['pair_id']}",
                "substrate": "life",
                "rule": ROUND2_RULE,
                "family": "pair_specific",
                "launch_index": int(old["launch_index"]),
                "density_delta": float(old["density_delta"]),
                "donor_a": donor_a,
                "donor_b": donor_b,
                "targets": {
                    name: {"A": targets_a[name], "B": targets_b[name]}
                    for name in ("primary", "primary_terminal", "local_secondary", "local_aux", "global")
                },
            }
        )
    return result


def _assignment(
    vector: Sequence[float] | np.ndarray,
    target_a: Sequence[float],
    target_b: Sequence[float],
    contract: V3Contract,
) -> str | None:
    similarity_a = _cosine(vector, target_a)
    similarity_b = _cosine(vector, target_b)
    if max(similarity_a, similarity_b) < contract.assignment_similarity:
        return None
    if abs(similarity_a - similarity_b) < contract.assignment_margin:
        return None
    return "A" if similarity_a > similarity_b else "B"


def _checkpoint_vectors(
    trace: BatchTrace, substrate: str, rule: int, checkpoint: int
) -> dict[str, np.ndarray]:
    index = checkpoint - 1
    values: dict[str, np.ndarray] = {"primary": trace.compositions[:, index]}
    values.update(extra_observer_vectors(substrate, trace.terminals[:, index]))
    if substrate == "eca" and index >= 2:
        windows = np.stack(
            (trace.terminals[:, index - 2], trace.terminals[:, index - 1], trace.terminals[:, index]),
            axis=1,
        )
        values["mesoscale"] = _eca_mesoscale(windows, rule)
    return values


def _trace_summary(
    trace: BatchTrace,
    substrate: str,
    rule: int,
    targets: dict[str, dict[str, list[float]]],
    contract: V3Contract,
    checkpoints: Sequence[int],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for checkpoint in checkpoints:
        index = checkpoint - 1
        if index >= trace.valid.shape[1]:
            continue
        vectors = _checkpoint_vectors(trace, substrate, rule, checkpoint)
        observers: dict[str, Any] = {}
        for observer in sorted(set(vectors) & set(targets)):
            labels: list[str | None] = []
            for future in range(len(trace.valid)):
                if not bool(trace.valid[future, index]):
                    labels.append(None)
                    continue
                labels.append(
                    _assignment(
                        vectors[observer][future],
                        targets[observer]["A"],
                        targets[observer]["B"],
                        contract,
                    )
                )
            observers[observer] = {
                "p_a": labels.count("A") / len(labels) if labels else 0.0,
                "p_b": labels.count("B") / len(labels) if labels else 0.0,
                "resolved": sum(label is not None for label in labels) / len(labels) if labels else 0.0,
            }
        persistent: list[str | None] = []
        for future in range(len(trace.valid)):
            if checkpoint < 8 or not bool(trace.valid[future, index]):
                persistent.append(None)
                continue
            labels = [
                _assignment(
                    trace.compositions[future, position],
                    targets["primary"]["A"],
                    targets["primary"]["B"],
                    contract,
                )
                if bool(trace.valid[future, position]) else None
                for position in range(index - 7, index + 1)
            ]
            persistent.append(labels[0] if labels[0] is not None and len(set(labels)) == 1 else None)
        result[str(checkpoint)] = {
            "observers": observers,
            "persistent_p_a": persistent.count("A") / len(persistent) if persistent else 0.0,
            "persistent_p_b": persistent.count("B") / len(persistent) if persistent else 0.0,
            "survival": float(np.mean(trace.valid[:, index])) if len(trace.valid) else 0.0,
            "valid": int(np.count_nonzero(trace.valid[:, index])),
            "n": len(trace.valid),
        }
    return result


def _transform_source(substrate: str, source: np.ndarray, operation: str) -> np.ndarray:
    if substrate == "life":
        return _transform_board(source, operation)
    if operation == "identity":
        return source.copy()
    if operation == "translate":
        return np.roll(source, 7)
    if operation in ("reflect", "rotate90"):
        return source[::-1].copy()
    raise ValueError(f"unknown operation {operation!r}")


def _resize_state(state: np.ndarray, extent: int) -> np.ndarray:
    if state.ndim == 1:
        if extent != len(state):
            raise ValueError("ECA scale changes are not registered")
        return state.copy()
    if extent % state.shape[0]:
        raise ValueError("Life scale must be an integer tiling of the 16x16 state")
    factor = extent // state.shape[0]
    return np.tile(state, (factor, factor))


def _block_control(source: np.ndarray, mask: np.ndarray, block: int, key: str) -> np.ndarray:
    if source.ndim == 1:
        chunks = [source[index : index + block].copy() for index in range(0, len(source), block)]
        order = sorted(range(len(chunks)), key=lambda index: hashlib.sha256(f"{key}:{index}".encode()).digest())
        shuffled = np.concatenate([chunks[index] for index in order])[: len(source)]
    elif block == 2:
        shuffled = _block_shuffle(source, np.ones_like(source), key)
    else:
        height, width = source.shape
        chunks = [
            source[y : y + block, x : x + block].copy()
            for y in range(0, height, block)
            for x in range(0, width, block)
        ]
        order = sorted(range(len(chunks)), key=lambda index: hashlib.sha256(f"{key}:{index}".encode()).digest())
        shuffled = np.zeros_like(source)
        cursor = 0
        for y in range(0, height, block):
            for x in range(0, width, block):
                shuffled[y : y + block, x : x + block] = chunks[order[cursor]]
                cursor += 1
    target_live = int(np.count_nonzero(source & mask))
    fragment = shuffled & mask
    if int(fragment.sum()) != target_live:
        fragment = _masked_random(tuple(source.shape), mask, target_live, f"{key}:mass-repair")
    return fragment


def _condition_bases(
    pair: dict[str, Any],
    label: str,
    condition: dict[str, Any],
    contract: V3Contract,
    profile: V3Profile,
) -> tuple[np.ndarray, dict[str, Any]]:
    substrate = str(pair["substrate"])
    donor = pair["donor_a" if label == "A" else "donor_b"]
    extent = int(condition.get("extent", contract.life_width if substrate == "life" else contract.eca_width))
    source_key = {
        "donor": "donor_state_hex",
        "ancestor": "ancestor_state_hex",
        "anchor": "anchor_state_hex",
    }.get(str(condition["intervention"]), "donor_state_hex")
    source = _resize_state(_state_from_hex(substrate, str(donor[source_key])), extent)
    source = _transform_source(substrate, source, str(condition.get("operation", "identity")))
    launch = _resize_state(_state_from_hex(substrate, str(donor["initial_state_hex"])), extent)
    geometry = str(condition.get("geometry", "square" if substrate == "life" else "one_interval"))
    dose = float(condition.get("dose", 0.5))
    if "tile" in condition:
        if substrate != "life" or tuple(source.shape) != (16, 16):
            raise ValueError("4x4 tile probes require the 16x16 Life substrate")
        tile = int(condition["tile"])
        mask = np.zeros((16, 16), dtype=np.bool_)
        mask[(tile // 4) * 4 : (tile // 4 + 1) * 4, (tile % 4) * 4 : (tile % 4 + 1) * 4] = True
    else:
        mask = _site_mask(tuple(source.shape), dose, geometry, f"{pair['pair_id']}:{condition['condition_id']}")
    intervention = str(condition["intervention"])
    metadata: dict[str, Any] = {"mask_cells": int(mask.sum())}
    if intervention in ("donor", "ancestor", "anchor"):
        fragments = [source & mask]
    elif intervention == "tile_sufficient":
        fragments = [source & mask]
    elif intervention == "tile_deleted":
        primary_mask = _site_mask(tuple(source.shape), 0.5, "square", f"{pair['pair_id']}:mapping-primary")
        base = _replace_mask(launch, source, primary_mask)
        bases = _replace_mask(base, launch, mask)[None, ...]
        metadata["transmitted_live"] = [int(np.count_nonzero(source & primary_mask & ~mask))]
        return bases, metadata
    elif intervention == "exact_random":
        live = int(np.count_nonzero(source & mask))
        fragments = [_masked_random(tuple(source.shape), mask, live, f"{pair['pair_id']}:{label}:exact")]
    elif intervention.startswith("block"):
        block = int(intervention.removeprefix("block"))
        fragments = [_block_control(source, mask, block, f"{pair['pair_id']}:{label}:{intervention}")]
        metadata["block"] = block
    elif intervention == "all_live":
        fragments = [mask.copy()]
    elif intervention == "conditional_null":
        fragments, null_meta = conditional_null_ensemble(
            substrate,
            source,
            mask,
            f"{pair['pair_id']}:{label}",
            candidates=profile.calibration_candidates,
            keep=profile.morphology_keep,
        )
        metadata["conditional_null"] = null_meta
    elif intervention in ("same_unrelated", "opposite_unrelated"):
        field = (
            f"same_donor_{label.lower()}"
            if intervention == "same_unrelated"
            else f"opposite_donor_{label.lower()}"
        )
        alternate = pair.get(field)
        if alternate is None:
            return np.empty((0,) + source.shape, dtype=np.bool_), {"missing": "no_unrelated_donor"}
        alternate_source = _resize_state(_state_from_hex(substrate, str(alternate["donor_state_hex"])), extent)
        fragments = [alternate_source & mask]
    else:
        raise ValueError(f"unknown intervention {intervention!r}")
    bases = np.stack([_replace_mask(launch, fragment, mask) for fragment in fragments])
    metadata["transmitted_live"] = [int(np.count_nonzero(fragment)) for fragment in fragments]
    return bases, metadata


def _simulate_condition(
    pair: dict[str, Any],
    label: str,
    condition: dict[str, Any],
    contract: V3Contract,
    profile: V3Profile,
    *,
    replicates: int,
    horizon: int,
    phase: str,
) -> dict[str, Any]:
    substrate = str(pair["substrate"])
    bases, metadata = _condition_bases(pair, label, condition, contract, profile)
    if not len(bases):
        return {"missing": True, "missing_reason": metadata["missing"]}
    extent = int(condition.get("extent", contract.life_width if substrate == "life" else contract.eca_width))
    tiled = np.stack([bases[index % len(bases)] for index in range(replicates)])
    causal = contract.causal(f"{phase}-garden", substrate=substrate, extent=extent if substrate == "life" else None)
    epsilon_multiplier = float(condition.get("copy_multiplier", 1.0))
    eta_multiplier = float(condition.get("process_multiplier", 1.0))
    epsilon = (contract.life_copy_error if substrate == "life" else contract.eca_copy_error) * epsilon_multiplier
    eta = (contract.life_process_noise if substrate == "life" else contract.eca_process_noise) * eta_multiplier
    # Source label is intentionally absent: A and B receive common copy and
    # process random streams for each pair/condition.
    seed_key = f"{phase}:{pair['pair_id']}:{condition['condition_id']}"
    initial = tiled.copy()
    if epsilon > 0.0:
        rng = np.random.default_rng(_hash_seed(contract.namespace, seed_key, "initial-copy"))
        initial ^= rng.random(initial.shape) < epsilon
    trace = _simulate_batch(
        substrate,
        int(condition.get("host_rule", pair["rule"])),
        initial,
        causal,
        horizon=horizon,
        rng_seed=_hash_seed(contract.namespace, seed_key, "trajectory"),
        observer="raw" if substrate == "eca" else "primary",
        process_noise=eta,
        copy_error=epsilon,
    )
    checkpoints = [value for value in contract.checkpoints if value <= horizon]
    if horizon not in checkpoints:
        checkpoints.append(horizon)
    return {
        "missing": False,
        "outcomes": _trace_summary(
            trace,
            substrate,
            int(condition.get("host_rule", pair["rule"])),
            pair["targets"],
            contract,
            checkpoints,
        ),
        "death_count": int(sum(reason is not None for reason in trace.death)),
        "metadata": metadata,
    }


def _transfer_task(arguments: tuple[dict[str, Any], V3Contract, V3Profile]) -> dict[str, Any]:
    item, contract, profile = arguments
    rows: list[dict[str, Any]] = []
    for pair in item["pairs"]:
        for condition in item["conditions"]:
            horizon = int(condition.get("horizon", item["horizon"]))
            for label in ("A", "B"):
                simulation = _simulate_condition(
                    pair,
                    label,
                    condition,
                    contract,
                    profile,
                    replicates=int(item["replicates"]),
                    horizon=horizon,
                    phase=str(item["phase"]),
                )
                rows.append(
                    {
                        "substrate": pair["substrate"],
                        "rule": int(pair["rule"]),
                        "family": pair.get("family", "local"),
                        "pair_id": pair["pair_id"],
                        "source_form": label,
                        **condition,
                        **simulation,
                    }
                )
    return {"entry": item["entry"], "phase": item["phase"], "rows": rows}


def _condition(
    intervention: str,
    *,
    dose: float = 0.5,
    geometry: str = "square",
    operation: str = "identity",
    horizon: int | None = None,
    **extra: Any,
) -> dict[str, Any]:
    pieces = [intervention, f"d{dose:g}", geometry, operation]
    pieces.extend(f"{key}-{extra[key]}" for key in sorted(extra))
    result: dict[str, Any] = {
        "condition_id": ":".join(pieces),
        "intervention": intervention,
        "dose": dose,
        "geometry": geometry,
        "operation": operation,
        **extra,
    }
    if horizon is not None:
        result["horizon"] = horizon
    return result


def _primary_conditions(*, replay: bool = False, wide: bool = False) -> list[dict[str, Any]]:
    if replay:
        names = ("donor", "exact_random", "block2", "conditional_null")
    elif wide:
        names = ("donor", "ancestor", "exact_random", "block2", "conditional_null")
    else:
        names = (
            "donor", "ancestor", "anchor", "exact_random", "block2", "block4",
            "block8", "conditional_null", "all_live", "same_unrelated", "opposite_unrelated",
        )
    return [
        _condition(name, horizon=128 if not replay and not wide and name in ("donor", "ancestor") else None)
        for name in names
    ]


def _mechanism_conditions(contract: V3Contract) -> list[dict[str, Any]]:
    conditions: list[dict[str, Any]] = []
    conditions.extend(_condition("donor", dose=dose) for dose in contract.doses)
    conditions.extend(
        _condition("donor", geometry=geometry)
        for geometry in ("strip", "two_lobe", "dispersed")
    )
    conditions.extend(
        _condition("donor", operation=operation)
        for operation in ("translate", "rotate90", "reflect")
    )
    conditions.extend(
        _condition(
            "donor",
            process_multiplier=multiplier,
            copy_multiplier=multiplier,
        )
        for multiplier in (0.0, 0.5, 2.0)
    )
    conditions.extend(_condition("donor", extent=extent) for extent in (32, 64))
    conditions.extend(
        _condition("donor", host_rule=ROUND2_RULE ^ (1 << bit)) for bit in range(17)
    )
    return conditions


def attach_unrelated_controls(pairs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    if not pairs:
        return []
    result: list[dict[str, Any]] = []
    for index, original in enumerate(pairs):
        pair = dict(original)
        alternate = pairs[(index + 1) % len(pairs)] if len(pairs) > 1 else None
        if alternate is not None:
            pair["same_donor_a"] = alternate["donor_a"]
            pair["same_donor_b"] = alternate["donor_b"]
            pair["opposite_donor_a"] = alternate["donor_b"]
            pair["opposite_donor_b"] = alternate["donor_a"]
        result.append(pair)
    return result


def _tile_mapping_task(arguments: tuple[dict[str, Any], V3Contract, V3Profile]) -> dict[str, Any]:
    item, contract, profile = arguments
    conditions = [_condition("donor")]
    for tile in item.get("tiles", range(16)):
        conditions.append(_condition("tile_sufficient", dose=0.0625, tile=int(tile)))
        conditions.append(_condition("tile_deleted", dose=0.5, tile=int(tile)))
    forwarded = {
        **item,
        "conditions": conditions,
        "replicates": profile.mapping_replicates,
        "horizon": 32,
    }
    return _transfer_task((forwarded, contract, profile))


def _pedigree_label_summary(
    states: np.ndarray,
    valid_count: int,
    expected: int,
    substrate: str,
    targets: dict[str, dict[str, list[float]]],
    contract: V3Contract,
) -> dict[str, Any]:
    observers: dict[str, Any] = {}
    if valid_count:
        vectors = extra_observer_vectors(substrate, states)
        if substrate == "life":
            # A completed one-generation Life primary composition is not
            # recoverable from the terminal alone; terminal 2x2 is the local
            # pedigree readout and is explicitly named as such.
            primary = _observer_vectors("life", states)["terminal2x2"]
        else:
            primary = _observer_vectors("eca", states)["raw4"]
        vectors["primary_terminal"] = primary
        target_map = dict(targets)
        for observer in sorted(set(vectors) & set(target_map)):
            labels = [
                _assignment(vector, target_map[observer]["A"], target_map[observer]["B"], contract)
                for vector in vectors[observer]
            ]
            observers[observer] = {
                "p_a": labels.count("A") / expected,
                "p_b": labels.count("B") / expected,
                "resolved": sum(label is not None for label in labels) / expected,
            }
    return {"expected": expected, "valid": valid_count, "survival": valid_count / expected, "observers": observers}


def _pedigree_task(arguments: tuple[dict[str, Any], V3Contract, V3Profile]) -> dict[str, Any]:
    item, contract, _profile = arguments
    pair = item["pairs"][0]
    substrate = str(pair["substrate"])
    rows: list[dict[str, Any]] = []
    for label in ("A", "B"):
        donor = pair["donor_a" if label == "A" else "donor_b"]
        source = _state_from_hex(substrate, str(donor["donor_state_hex"]))
        launch = _state_from_hex(substrate, str(donor["initial_state_hex"]))
        geometry = "square" if substrate == "life" else "one_interval"
        mask = _site_mask(tuple(source.shape), 0.5, geometry, f"{pair['pair_id']}:pedigree")
        current = _replace_mask(launch, source, mask)[None, ...]
        by_depth: dict[str, Any] = {}
        for depth in range(1, contract.pedigree_depth + 1):
            expected = 2**depth
            if len(current):
                parents = np.repeat(current, 2, axis=0)
                epsilon = contract.life_copy_error if substrate == "life" else contract.eca_copy_error
                rng = np.random.default_rng(
                    _hash_seed(contract.namespace, "pedigree", pair["pair_id"], depth)
                )
                parents ^= rng.random(parents.shape) < epsilon
                causal = contract.causal("pedigree", substrate=substrate)
                trace = _simulate_batch(
                    substrate,
                    int(pair["rule"]),
                    parents,
                    causal,
                    horizon=1,
                    rng_seed=_hash_seed(contract.namespace, "pedigree-step", pair["pair_id"], depth),
                    observer="raw" if substrate == "eca" else "primary",
                    copy_error=0.0,
                )
                valid = trace.valid[:, 0]
                current = trace.terminals[valid, 0]
            by_depth[str(depth)] = _pedigree_label_summary(
                current, len(current), expected, substrate, pair["targets"], contract
            )
        rows.append(
            {
                "substrate": substrate,
                "rule": int(pair["rule"]),
                "pair_id": pair["pair_id"],
                "source_form": label,
                "probe": "both_daughter_pedigree",
                "depths": by_depth,
            }
        )
    return {"entry": item["entry"], "phase": item["phase"], "rows": rows}


def _row_probability(row: dict[str, Any], checkpoint: int, label: str, observer: str = "primary") -> float:
    return float(row["outcomes"][str(checkpoint)]["observers"].get(observer, {}).get("p_a" if label == "A" else "p_b", 0.0))


def _group_rows(rows: Sequence[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        if row.get("missing"):
            continue
        result[(str(row["pair_id"]), str(row["condition_id"]), str(row["source_form"]))] = row
    return result


def pair_crossovers(
    rows: Sequence[dict[str, Any]],
    checkpoint: int,
    condition_id: str,
    *,
    observer: str = "primary",
) -> list[dict[str, float | str]]:
    index = _group_rows(rows)
    pair_ids = sorted({key[0] for key in index if key[1] == condition_id})
    result: list[dict[str, float | str]] = []
    for pair_id in pair_ids:
        row_a = index.get((pair_id, condition_id, "A"))
        row_b = index.get((pair_id, condition_id, "B"))
        if row_a is None or row_b is None:
            continue
        d_a = _row_probability(row_a, checkpoint, "A", observer) - _row_probability(row_b, checkpoint, "A", observer)
        d_b = _row_probability(row_b, checkpoint, "B", observer) - _row_probability(row_a, checkpoint, "B", observer)
        survival = 0.5 * (
            float(row_a["outcomes"][str(checkpoint)]["survival"])
            + float(row_b["outcomes"][str(checkpoint)]["survival"])
        )
        result.append({"pair_id": pair_id, "crossover": min(d_a, d_b), "direction_a": d_a, "direction_b": d_b, "survival": survival})
        result[-1]["correct_probability"] = 0.5 * (
            _row_probability(row_a, checkpoint, "A", observer)
            + _row_probability(row_b, checkpoint, "B", observer)
        )
    return result


def _flatten(stage: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for result in stage for row in result.get("rows", [])]


def select_mapping_tile(stage: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rows = _flatten(stage)
    baseline_id = _condition("donor")["condition_id"]
    baseline = {str(row["pair_id"]): float(row["crossover"]) for row in pair_crossovers(rows, 32, baseline_id)}
    scores: list[dict[str, Any]] = []
    for tile in range(16):
        sufficient_id = _condition("tile_sufficient", dose=0.0625, tile=tile)["condition_id"]
        deleted_id = _condition("tile_deleted", dose=0.5, tile=tile)["condition_id"]
        sufficient = pair_crossovers(rows, 32, sufficient_id)
        deleted = pair_crossovers(rows, 32, deleted_id)
        suff_map = {str(row["pair_id"]): float(row["crossover"]) for row in sufficient}
        delete_map = {str(row["pair_id"]): float(row["crossover"]) for row in deleted}
        shared = sorted(set(suff_map) & set(delete_map) & set(baseline))
        values = [suff_map[key] + baseline[key] - delete_map[key] for key in shared]
        scores.append({"tile": tile, "n_pairs": len(values), "mean_score": float(np.mean(values)) if values else None})
    eligible = [row for row in scores if row["mean_score"] is not None]
    selected = max(eligible, key=lambda row: (float(row["mean_score"]), -int(row["tile"]))) if eligible else {"tile": 0, "n_pairs": 0, "mean_score": None}
    return {"selection_rule": "max mean(sufficiency + baseline - deletion); lower tile tie break", "selected_tile": int(selected["tile"]), "scores": scores}


def rank_extension_rules(
    discoveries: Sequence[dict[str, Any]], profile: V3Profile
) -> dict[str, list[int]]:
    ranked: dict[str, list[tuple[tuple[float, ...], int]]] = {"life": [], "eca": []}
    for result in discoveries:
        entry = result["entry"]
        score = tuple(float(value) for value in result.get("discovery", {}).get("best_score", [0, 0, 0, 0]))
        ranked[str(entry["substrate"])].append((score, int(entry["rule"])))
    output: dict[str, list[int]] = {}
    for substrate, rows in ranked.items():
        rows.sort(key=lambda row: (tuple(-value for value in row[0]), row[1]))
        maximum = profile.extend_life_rules if substrate == "life" else profile.extend_eca_rules
        output[substrate] = [rule for _, rule in rows[:maximum]]
    return output


def select_screen_candidates(
    extensions: Sequence[dict[str, Any]], profile: V3Profile, contract: V3Contract
) -> list[dict[str, Any]]:
    best_by_rule_family: dict[tuple[str, int, str], dict[str, Any]] = {}
    for result in extensions:
        discovery = result.get("discovery") or {}
        for family_values in discovery.get("families", {}).values():
            for candidate in family_values.get("candidates", []):
                if len(candidate["pairs"]) < profile.screen_pairs:
                    continue
                if min(int(candidate["cluster_a"]["size"]), int(candidate["cluster_b"]["size"])) < contract.cluster_minimum:
                    continue
                if min(len(candidate["cluster_a"]["launches"]), len(candidate["cluster_b"]["launches"])) < contract.cluster_launches:
                    continue
                key = (str(candidate["substrate"]), int(candidate["rule"]), str(candidate["family"]))
                previous = best_by_rule_family.get(key)
                if previous is None or tuple(candidate["score"]) > tuple(previous["score"]) or (
                    tuple(candidate["score"]) == tuple(previous["score"])
                    and str(candidate["candidate_id"]) < str(previous["candidate_id"])
                ):
                    best_by_rule_family[key] = candidate
    by_substrate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (substrate, _rule, _family), candidate in best_by_rule_family.items():
        by_substrate[substrate].append(candidate)
    selected: list[dict[str, Any]] = []
    for substrate in ("life", "eca"):
        candidates = sorted(
            by_substrate.get(substrate, []),
            key=lambda row: (tuple(-float(value) for value in row["score"]), str(row["candidate_id"])),
        )
        maximum = profile.screen_life_candidates if substrate == "life" else profile.screen_eca_candidates
        selected.extend(candidates[:maximum])
    return selected


def _candidate_targets(candidate: dict[str, Any]) -> dict[str, dict[str, list[float]]]:
    common = sorted(set(candidate["cluster_a"]["targets"]) & set(candidate["cluster_b"]["targets"]))
    return {
        observer: {
            "A": candidate["cluster_a"]["targets"][observer],
            "B": candidate["cluster_b"]["targets"][observer],
        }
        for observer in common
    }


def _assign_candidate_donors(
    donors: Sequence[dict[str, Any]], candidate: dict[str, Any], contract: V3Contract
) -> list[dict[str, Any]]:
    observer = str(candidate["observer"])
    targets = _candidate_targets(candidate)
    by_launch_label: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for donor in donors:
        label = _assignment(
            donor["target_compositions"][observer],
            targets[observer]["A"],
            targets[observer]["B"],
            contract,
        )
        if label is not None:
            by_launch_label[(int(donor["launch_index"]), label)].append(donor)
    pseudo_left = {
        "cluster_id": "holdout-a",
        "targets": {name: values["A"] for name, values in targets.items()},
        "members": [donor for (launch, label), values in by_launch_label.items() if label == "A" for donor in values],
    }
    pseudo_right = {
        "cluster_id": "holdout-b",
        "targets": {name: values["B"] for name, values in targets.items()},
        "members": [donor for (launch, label), values in by_launch_label.items() if label == "B" for donor in values],
    }
    return _pair_members(
        pseudo_left,
        pseudo_right,
        contract.wide_density_tolerance,
        int(candidate["rule"]),
        str(candidate["family"]),
    )


def _holdout_acquire_task(arguments: tuple[dict[str, Any], V3Contract, V3Profile]) -> dict[str, Any]:
    item, contract, profile = arguments
    candidate = item["candidate"]
    acquisition = _discover_donors(
        substrate=str(candidate["substrate"]),
        rule=int(candidate["rule"]),
        contract=contract,
        namespace=f"wide-holdout-acquisition:{candidate['candidate_id']}",
        cap=profile.extend_cap,
        retained=max(profile.retained_donors, profile.holdout_pairs * 8),
        analyze=False,
    )
    pairs = _assign_candidate_donors(acquisition["donors"], candidate, contract)
    return {
        "entry": acquisition["entry"],
        "candidate": {key: value for key, value in candidate.items() if key != "pairs"},
        "examined": acquisition["examined"],
        "strict_seen": acquisition["strict_seen"],
        "retained": acquisition["retained"],
        "pair_count": len(pairs),
        "pairs": pairs[: profile.holdout_pairs],
    }


def _summary(values: Sequence[float], profile: V3Profile, *seed_parts: object) -> dict[str, Any]:
    result = _bootstrap_summary(
        values,
        profile.bootstrap_resamples,
        _hash_seed("ca-carrier-v3-bootstrap", *seed_parts),
    )
    data = np.asarray(values, dtype=np.float64)
    if not len(data):
        result["p_value"] = 1.0
    else:
        rng = np.random.default_rng(_hash_seed("ca-carrier-v3-sign-flip", *seed_parts))
        observed = float(data.mean())
        exceed = 0
        for _ in range(profile.bootstrap_resamples):
            signs = rng.choice(np.asarray((-1.0, 1.0)), size=len(data))
            exceed += int(float(np.mean(data * signs)) >= observed)
        result["p_value"] = (exceed + 1) / (profile.bootstrap_resamples + 1)
    return result


def _condition_difference(
    rows: Sequence[dict[str, Any]], checkpoint: int, left_id: str, right_id: str,
    *, observer: str = "primary",
) -> list[float]:
    left = {str(row["pair_id"]): float(row["crossover"]) for row in pair_crossovers(rows, checkpoint, left_id, observer=observer)}
    right = {str(row["pair_id"]): float(row["crossover"]) for row in pair_crossovers(rows, checkpoint, right_id, observer=observer)}
    return [left[key] - right[key] for key in sorted(set(left) & set(right))]


def _adjudicate_transfer_rows(
    rows: Sequence[dict[str, Any]],
    checkpoint: int,
    profile: V3Profile,
    contract: V3Contract,
    namespace: str,
    *,
    primary_observer: str = "primary",
) -> dict[str, Any]:
    donor_id = _condition("donor")["condition_id"]
    primary_rows = pair_crossovers(rows, checkpoint, donor_id, observer=primary_observer)
    crossover = [float(row["crossover"]) for row in primary_rows]
    directions_a = [float(row["direction_a"]) for row in primary_rows]
    directions_b = [float(row["direction_b"]) for row in primary_rows]
    survival = [float(row["survival"]) for row in primary_rows]
    correct_probability = [float(row["correct_probability"]) for row in primary_rows]
    observers: dict[str, Any] = {}
    for observer in ("local_secondary", "local_aux", "mesoscale", "global"):
        values = [
            float(row["crossover"])
            for row in pair_crossovers(rows, checkpoint, donor_id, observer=observer)
        ]
        observers[observer] = _summary(values, profile, namespace, checkpoint, observer)
    controls: dict[str, Any] = {}
    for control in ("ancestor", "exact_random", "block2", "conditional_null"):
        control_id = _condition(control)["condition_id"]
        values = _condition_difference(rows, checkpoint, donor_id, control_id, observer=primary_observer)
        controls[control] = _summary(values, profile, namespace, checkpoint, control)
    interventions: dict[str, Any] = {}
    for intervention in (
        "anchor", "block4", "block8", "all_live", "same_unrelated", "opposite_unrelated"
    ):
        condition_id = _condition(intervention)["condition_id"]
        values = [
            float(row["crossover"])
            for row in pair_crossovers(
                rows, checkpoint, condition_id, observer=primary_observer
            )
        ]
        interventions[intervention] = _summary(
            values, profile, namespace, checkpoint, intervention
        )
    return {
        "n_pairs": len(crossover),
        "primary_observer": primary_observer,
        "crossover": _summary(crossover, profile, namespace, checkpoint, "primary"),
        "direction_a_mean": float(np.mean(directions_a)) if directions_a else None,
        "direction_b_mean": float(np.mean(directions_b)) if directions_b else None,
        "survival_mean": float(np.mean(survival)) if survival else 0.0,
        "correct_probability_mean": float(np.mean(correct_probability)) if correct_probability else 0.0,
        "fraction_pairs_positive": float(np.mean(np.asarray(crossover) > 0.0)) if crossover else 0.0,
        "observers": observers,
        "control_advantages": controls,
        "interventions": interventions,
    }


def _gate(summary: dict[str, Any], threshold: float) -> bool:
    return bool(
        summary.get("mean") is not None
        and float(summary["mean"]) >= threshold
        and summary.get("ci95", [None])[0] is not None
        and float(summary["ci95"][0]) > 0.0
    )


def select_holdout_candidates(
    screen_stage: Sequence[dict[str, Any]],
    candidates: Sequence[dict[str, Any]],
    profile: V3Profile,
    contract: V3Contract,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = _flatten(screen_stage)
    by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pair_to_candidate: dict[str, str] = {}
    for candidate in candidates:
        for pair in candidate["pairs"][: profile.screen_pairs]:
            pair_to_candidate[str(pair["pair_id"])] = str(candidate["candidate_id"])
    for row in rows:
        candidate_id = pair_to_candidate.get(str(row["pair_id"]))
        if candidate_id is not None:
            by_candidate[candidate_id].append(row)
    summaries: dict[str, Any] = {}
    eligible: dict[str, list[tuple[float, str, dict[str, Any]]]] = defaultdict(list)
    lookup = {str(candidate["candidate_id"]): candidate for candidate in candidates}
    for candidate_id, candidate_rows in by_candidate.items():
        candidate = lookup[candidate_id]
        summary = _adjudicate_transfer_rows(
            candidate_rows,
            32,
            profile,
            contract,
            f"screen:{candidate_id}",
            primary_observer=str(candidate["observer"]),
        )
        # A global-discovery candidate must agree with an independently local
        # observer; a local-discovery candidate must agree with a second local
        # encoding before it can advance.
        validator_names = ("local_secondary", "local_aux")
        validator_pass = any(_gate(summary["observers"][name], 0.0) for name in validator_names)
        passed = bool(
            summary["n_pairs"] >= profile.screen_pairs
            and summary["crossover"].get("mean") is not None
            and float(summary["crossover"]["mean"]) >= contract.local_validator_crossover
            and summary["survival_mean"] >= contract.survival_gate
            and validator_pass
            and summary["control_advantages"]["exact_random"].get("mean") is not None
            and float(summary["control_advantages"]["exact_random"]["mean"]) >= contract.control_advantage
            and summary["control_advantages"]["block2"].get("mean") is not None
            and float(summary["control_advantages"]["block2"]["mean"]) >= contract.control_advantage
        )
        summary["eligible_for_holdout"] = passed
        summaries[candidate_id] = summary
        if passed:
            eligible[str(candidate["substrate"])].append(
                (float(summary["crossover"]["mean"]), candidate_id, candidate)
            )
    selected: list[dict[str, Any]] = []
    for substrate in ("life", "eca"):
        ordered = sorted(eligible.get(substrate, []), key=lambda row: (-row[0], row[1]))
        selected.extend(candidate for _, _, candidate in ordered[: profile.holdout_candidates])
    return selected, summaries


def _narrow_verdict(
    replay: dict[str, Any],
    confirmation: dict[str, Any],
    pedigree: dict[str, Any],
    robustness: dict[str, Any],
    morphology_balance: float,
    profile: V3Profile,
    contract: V3Contract,
) -> dict[str, Any]:
    pair_specific = bool(
        replay["n_pairs"] >= profile.replay_pairs
        and _gate(replay["crossover"], contract.primary_crossover)
        and replay["direction_a_mean"] is not None
        and replay["direction_a_mean"] > 0
        and replay["direction_b_mean"] is not None
        and replay["direction_b_mean"] > 0
        and replay["fraction_pairs_positive"] >= 0.5
        and replay["survival_mean"] >= contract.survival_gate
        and _gate(replay["control_advantages"]["exact_random"], contract.control_advantage)
        and _gate(replay["control_advantages"]["block2"], contract.control_advantage)
    )
    controls = confirmation["control_advantages"]
    reusable = bool(
        pair_specific
        and confirmation["n_pairs"] >= profile.confirmation_pairs
        and _gate(confirmation["crossover"], contract.primary_crossover)
        and confirmation["direction_a_mean"] is not None
        and confirmation["direction_a_mean"] > 0
        and confirmation["direction_b_mean"] is not None
        and confirmation["direction_b_mean"] > 0
        and confirmation["survival_mean"] >= contract.survival_gate
        and confirmation["fraction_pairs_positive"] >= 0.5
        and confirmation["correct_probability_mean"] >= 0.25
        and _gate(controls["ancestor"], contract.control_advantage)
        and _gate(controls["exact_random"], contract.control_advantage)
        and _gate(controls["block2"], contract.control_advantage)
        and any(_gate(confirmation["observers"][name], 0.0) for name in ("local_secondary", "local_aux"))
        and _gate(
            confirmation["interventions"]["same_unrelated"],
            contract.local_validator_crossover,
        )
    )
    pedigree_gate = bool(pedigree.get("depth8_crossover", {}).get("mean") is not None and pedigree["depth8_crossover"]["ci95"][0] is not None and pedigree["depth8_crossover"]["ci95"][0] > 0)
    durable = reusable and pedigree_gate and bool(confirmation.get("generation_128_pass", False))
    global_gate = _gate(confirmation["observers"]["global"], contract.local_validator_crossover)
    observer_robust = bool(
        durable
        and global_gate
        and morphology_balance >= 0.90
        and robustness.get("geometry_pass", False)
        and robustness.get("scale_pass", False)
        and robustness.get("moderate_noise_pass", False)
    )
    verdict = (
        "OBSERVER_ROBUST_PLASTIC_HEREDITY" if observer_robust else
        "DURABLE_LOCAL_PLASTIC_HEREDITY" if durable else
        "REUSABLE_LOCAL_TEXTURE_FORM" if reusable else
        "CAUSAL_PAIR_TEXTURE_ADDRESS" if pair_specific else
        "PAIR_SPECIFIC_ONLY" if float(replay["crossover"].get("mean") or 0.0) > 0 else
        "NO_CAUSAL_CARRIER_FOUND"
    )
    return {
        "pair_specific_gate": pair_specific,
        "reusable_local_gate": reusable,
        "durable_local_gate": durable,
        "global_observer_gate": global_gate,
        "morphology_balance_fraction": morphology_balance,
        "robustness_gate": bool(
            robustness.get("geometry_pass", False)
            and robustness.get("scale_pass", False)
            and robustness.get("moderate_noise_pass", False)
        ),
        "observer_robust_gate": observer_robust,
        "verdict": verdict,
    }


def _run_stage(
    output: Path,
    stage: str,
    items: Sequence[dict[str, Any]],
    task: Callable[[tuple[dict[str, Any], V3Contract, V3Profile]], dict[str, Any]],
    contract: V3Contract,
    profile: V3Profile,
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
    initial_completed = len(results)

    def save(item: dict[str, Any], result: dict[str, Any]) -> None:
        key = str(item["checkpoint"])
        _atomic_json(
            checkpoints / f"{key}.json",
            {"checkpoint": key, "stage": stage, "design_digest": design_digest, "result": result},
        )
        results[key] = result
        completed_new = max(1, len(results) - initial_completed)
        elapsed = max(time.time() - started, 1e-9)
        remaining = max(0, len(items) - len(results))
        eta = elapsed / completed_new * remaining
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


def _make_transfer_items(
    pairs: Sequence[dict[str, Any]],
    *,
    phase: str,
    conditions: Sequence[dict[str, Any]],
    replicates: int,
    horizon: int,
) -> list[dict[str, Any]]:
    return [
        {
            "checkpoint": f"{phase}-{index:05d}",
            "entry": {
                "substrate": pair["substrate"],
                "rule": int(pair["rule"]),
                "candidate_id": pair.get("candidate_id"),
            },
            "phase": phase,
            "pairs": [pair],
            "conditions": list(conditions),
            "replicates": replicates,
            "horizon": horizon,
        }
        for index, pair in enumerate(pairs)
    ]


def _morphology_balance(
    rows: Sequence[dict[str, Any]], calibration: dict[str, Any]
) -> dict[str, Any]:
    calipers = calibration["conditional_null"]["calipers"]
    checks: list[bool] = []
    metrics: list[dict[str, float]] = []
    for row in rows:
        if row.get("intervention") != "conditional_null" or row.get("missing"):
            continue
        best = row.get("metadata", {}).get("conditional_null", {}).get("best")
        if not best:
            continue
        metrics.append(best)
        checks.append(
            float(best["neighbor_error"]) <= float(calipers["neighbor_error"])
            and float(best["component_cosine"]) >= float(calipers["component_cosine_min"])
            and float(best["structure_error"]) <= float(calipers["structure_error"])
        )
    return {
        "n_fragments": len(checks),
        "passing": int(sum(checks)),
        "fraction": float(np.mean(checks)) if checks else 0.0,
        "calipers": calipers,
        "metrics": metrics,
    }


def _pedigree_adjudication(
    stage: Sequence[dict[str, Any]], profile: V3Profile
) -> dict[str, Any]:
    rows = _flatten(stage)
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[str(row["pair_id"])][str(row["source_form"])] = row
    values: list[float] = []
    survival: list[float] = []
    for pair_id, pair_rows in grouped.items():
        if "A" not in pair_rows or "B" not in pair_rows:
            continue
        a = pair_rows["A"]["depths"]["8"]
        b = pair_rows["B"]["depths"]["8"]
        a_obs = a["observers"].get("primary_terminal", {})
        b_obs = b["observers"].get("primary_terminal", {})
        d_a = float(a_obs.get("p_a", 0.0)) - float(b_obs.get("p_a", 0.0))
        d_b = float(b_obs.get("p_b", 0.0)) - float(a_obs.get("p_b", 0.0))
        values.append(min(d_a, d_b))
        survival.append(0.5 * (float(a["survival"]) + float(b["survival"])))
    return {
        "depth8_crossover": _summary(values, profile, "narrow-pedigree", "depth8"),
        "depth8_survival_mean": float(np.mean(survival)) if survival else 0.0,
    }


def _mechanism_adjudication(
    stage: Sequence[dict[str, Any]], profile: V3Profile
) -> dict[str, Any]:
    rows = _flatten(stage)

    def summarize(condition: dict[str, Any], name: str) -> dict[str, Any]:
        values = [
            float(row["crossover"])
            for row in pair_crossovers(rows, 64, str(condition["condition_id"]))
        ]
        return _summary(values, profile, "narrow-mechanism", name)

    doses = {str(dose): summarize(_condition("donor", dose=dose), f"dose-{dose}") for dose in V3Contract().doses}
    geometries = {
        geometry: summarize(_condition("donor", geometry=geometry), f"geometry-{geometry}")
        for geometry in ("strip", "two_lobe", "dispersed")
    }
    transformations = {
        operation: summarize(_condition("donor", operation=operation), f"transform-{operation}")
        for operation in ("translate", "rotate90", "reflect")
    }
    noise = {
        str(multiplier): summarize(
            _condition("donor", process_multiplier=multiplier, copy_multiplier=multiplier),
            f"noise-{multiplier}",
        )
        for multiplier in (0.0, 0.5, 2.0)
    }
    scales = {
        str(extent): summarize(_condition("donor", extent=extent), f"scale-{extent}")
        for extent in (32, 64)
    }
    neighbors = {
        str(bit): summarize(
            _condition("donor", host_rule=ROUND2_RULE ^ (1 << bit)), f"neighbor-{bit}"
        )
        for bit in range(17)
    }
    return {
        "doses": doses,
        "geometries": geometries,
        "transformations": transformations,
        "noise": noise,
        "scales": scales,
        "rule_neighbors": neighbors,
        "geometry_pass": any(_gate(summary, 0.0) for summary in geometries.values()),
        "scale_pass": any(_gate(summary, 0.0) for summary in scales.values()),
        "moderate_noise_pass": _gate(noise["0.5"], 0.0),
    }


def _mapping_validation_adjudication(
    stage: Sequence[dict[str, Any]], profile: V3Profile
) -> dict[str, Any]:
    rows = _flatten(stage)
    tiles = sorted({int(row["tile"]) for row in rows if "tile" in row})
    if not tiles:
        return {
            "selected_tile": None,
            "sufficiency": _summary([], profile, "mapping-validation", "sufficiency"),
            "necessity": _summary([], profile, "mapping-validation", "necessity"),
        }
    tile = tiles[0]
    baseline_id = _condition("donor")["condition_id"]
    sufficient_id = _condition("tile_sufficient", dose=0.0625, tile=tile)["condition_id"]
    deleted_id = _condition("tile_deleted", dose=0.5, tile=tile)["condition_id"]
    sufficient = [
        float(row["crossover"]) for row in pair_crossovers(rows, 32, sufficient_id)
    ]
    necessity = _condition_difference(rows, 32, baseline_id, deleted_id)
    return {
        "selected_tile": tile,
        "sufficiency": _summary(sufficient, profile, "mapping-validation", tile, "sufficiency"),
        "necessity": _summary(necessity, profile, "mapping-validation", tile, "necessity"),
    }


def adjudicate_campaign(
    stage_data: dict[str, list[dict[str, Any]]],
    calibration: dict[str, Any],
    profile: V3Profile,
    contract: V3Contract,
    screen_candidates: Sequence[dict[str, Any]],
    holdout_acquisitions: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    replay_rows = _flatten(stage_data.get("narrow_replay", []))
    confirm_rows = _flatten(stage_data.get("narrow_confirm", []))
    replay = _adjudicate_transfer_rows(replay_rows, 64, profile, contract, "narrow-replay")
    confirmation = _adjudicate_transfer_rows(confirm_rows, 64, profile, contract, "narrow-confirm")
    donor128 = pair_crossovers(confirm_rows, 128, _condition("donor")["condition_id"])
    generation128 = _summary(
        [float(row["crossover"]) for row in donor128], profile, "narrow-confirm", "generation128"
    )
    confirmation["generation_128"] = generation128
    confirmation["generation_128_pass"] = _gate(generation128, contract.primary_crossover)
    pedigree = _pedigree_adjudication(stage_data.get("narrow_pedigree", []), profile)
    robustness = _mechanism_adjudication(stage_data.get("narrow_mechanism", []), profile)
    mapping_validation = _mapping_validation_adjudication(
        stage_data.get("narrow_map_validate", []), profile
    )
    morphology = _morphology_balance(confirm_rows, calibration)
    narrow = {
        "replay": replay,
        "fresh_confirmation": confirmation,
        "pedigree": pedigree,
        "robustness": robustness,
        "mapping_validation": mapping_validation,
        "morphology_balance": morphology,
    }
    narrow["claim"] = _narrow_verdict(
        replay,
        confirmation,
        pedigree,
        robustness,
        float(morphology["fraction"]),
        profile,
        contract,
    )

    holdout_rows = _flatten(stage_data.get("wide_holdout", []))
    pair_to_candidate: dict[str, dict[str, Any]] = {}
    for acquisition in holdout_acquisitions:
        candidate = acquisition["candidate"]
        for pair in acquisition.get("pairs", []):
            pair_to_candidate[str(pair["pair_id"])] = candidate
    grouped_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    candidate_lookup: dict[str, dict[str, Any]] = {}
    for row in holdout_rows:
        candidate = pair_to_candidate.get(str(row["pair_id"]))
        if candidate:
            candidate_id = str(candidate["candidate_id"])
            grouped_rows[candidate_id].append(row)
            candidate_lookup[candidate_id] = candidate
    wide: dict[str, Any] = {}
    family_tests: dict[tuple[str, str], list[tuple[str, str, float]]] = defaultdict(list)
    for candidate_id, rows in grouped_rows.items():
        candidate = candidate_lookup[candidate_id]
        summary = _adjudicate_transfer_rows(
            rows,
            64,
            profile,
            contract,
            f"holdout:{candidate_id}",
            primary_observer=str(candidate["observer"]),
        )
        summary["candidate"] = candidate
        summary["morphology_balance"] = _morphology_balance(rows, calibration)
        family_key = (str(candidate["substrate"]), str(candidate["family"]))
        family_tests[family_key].append(
            (candidate_id, "crossover", float(summary["crossover"]["p_value"]))
        )
        for control_name, control_summary in summary["control_advantages"].items():
            family_tests[family_key].append(
                (candidate_id, control_name, float(control_summary["p_value"]))
            )
        wide[candidate_id] = summary
    for family, tests in family_tests.items():
        adjusted = _holm([value for _, _, value in tests])
        for (candidate_id, test_name, _), p_holm in zip(tests, adjusted, strict=True):
            target = (
                wide[candidate_id]["crossover"]
                if test_name == "crossover"
                else wide[candidate_id]["control_advantages"][test_name]
            )
            target["p_holm"] = p_holm
    for candidate_id, summary in wide.items():
        primary = summary["crossover"]
        controls = summary["control_advantages"]
        local_agreement = any(_gate(summary["observers"][name], 0.0) for name in ("local_secondary", "local_aux"))
        global_agreement = _gate(summary["observers"]["global"], contract.local_validator_crossover)
        passed = bool(
            _gate(primary, contract.primary_crossover)
            and float(primary.get("p_holm", 1.0)) < 0.05
            and summary["survival_mean"] >= contract.survival_gate
            and summary["fraction_pairs_positive"] >= 0.5
            and summary["correct_probability_mean"] >= 0.25
            and _gate(controls["ancestor"], contract.control_advantage)
            and float(controls["ancestor"].get("p_holm", 1.0)) < 0.05
            and _gate(controls["exact_random"], contract.control_advantage)
            and float(controls["exact_random"].get("p_holm", 1.0)) < 0.05
            and _gate(controls["block2"], contract.control_advantage)
            and float(controls["block2"].get("p_holm", 1.0)) < 0.05
        )
        summary["local_agreement"] = local_agreement
        summary["global_agreement"] = global_agreement
        morphology_pass = bool(
            _gate(controls["conditional_null"], contract.control_advantage)
            and float(controls["conditional_null"].get("p_holm", 1.0)) < 0.05
            and float(summary["morphology_balance"]["fraction"]) >= 0.90
        )
        summary["morphology_pass"] = morphology_pass
        summary["verdict"] = (
            "OBSERVER_ROBUST_PLASTIC_HEREDITY" if passed and local_agreement and global_agreement and morphology_pass else
            "REUSABLE_LOCAL_TEXTURE_FORM" if passed and local_agreement else
            "GENERIC_NUCLEATION_ONLY" if summary["survival_mean"] >= contract.survival_gate else
            "NO_CAUSAL_CARRIER_FOUND"
        )
    return {
        "narrow": narrow,
        "wide": wide,
        "wide_counts": {
            "screen_candidates": len(screen_candidates),
            "holdout_candidates": len(holdout_acquisitions),
            "adjudicated_holdouts": len(wide),
        },
    }


def _render_report(results: dict[str, Any]) -> str:
    narrow = results["adjudication"]["narrow"]
    claim = narrow["claim"]
    lines = [
        "# CA carrier round 3: continuous forms and observer scope",
        "",
        f"State: **{results['state']}**. Profile: `{results['profile']}`.",
        f"Design digest: `{results['design_digest']}`.",
        "",
        "## Narrow rule-31649 result",
        "",
        f"- Verdict: **{claim['verdict']}**.",
        f"- Pair-specific gate: `{claim['pair_specific_gate']}`.",
        f"- Reusable local-form gate: `{claim['reusable_local_gate']}`.",
        f"- Durable local-heredity gate: `{claim['durable_local_gate']}`.",
        f"- Global-observer gate: `{claim['global_observer_gate']}`.",
        f"- Geometry/scale/noise robustness gate: `{claim['robustness_gate']}`.",
        f"- Morphology balance: `{claim['morphology_balance_fraction']}`.",
        "",
        "## Broad search",
        "",
        f"Screen candidates: `{results['adjudication']['wide_counts']['screen_candidates']}`; "
        f"holdouts: `{results['adjudication']['wide_counts']['adjudicated_holdouts']}`.",
    ]
    for candidate_id, values in sorted(results["adjudication"]["wide"].items()):
        lines.append(
            f"- `{candidate_id}`: **{values['verdict']}**, crossover "
            f"`{values['crossover']['mean']}`, CI `{values['crossover']['ci95']}`."
        )
    lines.extend(
        (
            "",
            "## Evidence boundary",
            "",
            "A texture-address result means spatial information causally steered an observer-level future. "
            "Only the highest tier supports a reusable acquired form seen by both local and global observers. "
            "No tier implies metabolism, agency, or biological life.",
            "",
        )
    )
    return "\n".join(lines)


def _render_lay_summary(results: dict[str, Any]) -> str:
    claim = results["adjudication"]["narrow"]["claim"]
    wide = results["adjudication"]["wide"]
    positives = [candidate for candidate, row in wide.items() if row["verdict"] != "NO_CAUSAL_CARRIER_FOUND"]
    state = "The planned run finished." if results["state"] == "complete" else "The 48-hour budget ended with a resumable partial run."
    return "\n\n".join(
        (
            "# Lay summary",
            state,
            (
                "The narrow test asks whether a piece of a parent pattern carries a reproducible local texture, "
                "then asks whether freshly acquired examples carry the same texture rather than merely helping "
                "the recipient survive. Its current round-three verdict is " + claim["verdict"].replace("_", " ").lower() + "."
            ),
            (
                "The wider test searches every one of the 256 elementary rules and every rule in the retained "
                "1,024-rule Life-like catalogue. It only uses the phrase Plastic Heredity when the acquired "
                "state beats its own earlier ancestor and independent local and global descriptions agree."
            ),
            f"Broad candidates with something above a complete null: {len(positives)}.",
        )
    ) + "\n"


def _update_discovery_log(results: dict[str, Any], path: Path) -> None:
    start = "<!-- ca-carrier-round-3:start -->"
    end = "<!-- ca-carrier-round-3:end -->"
    claim = results["adjudication"]["narrow"]["claim"]
    section = "\n".join(
        (
            start,
            "## CA carrier round 3",
            "",
            f"State `{results['state']}` under design `{results['design_digest']}`.",
            f"Narrow rule-31649 verdict: `{claim['verdict']}`.",
            f"Broad holdouts adjudicated: `{results['adjudication']['wide_counts']['adjudicated_holdouts']}`.",
            "",
            "See `results/ca-carrier-round-3/REPORT.md` for the continuous-form evidence ladder.",
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


def run_ca_carrier_v3(
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
    if profile_name not in V3_PROFILES:
        raise ValueError(f"unknown v3 profile {profile_name!r}")
    if not ROUND2_ACQUISITION.exists():
        raise FileNotFoundError(f"round-2 development acquisition not found: {ROUND2_ACQUISITION}")
    if not life_atlas.exists():
        raise FileNotFoundError(f"frozen Life atlas not found: {life_atlas}")
    profile = V3_PROFILES[profile_name]
    contract = V3Contract()
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    deadline = started + max_hours * 3600 if max_hours > 0 else None

    stage_order = (
        "calibrate",
        "narrow_acquire",
        "narrow_replay",
        "narrow_confirm",
        "narrow_pedigree",
        "wide_discover",
        "wide_extend",
        "wide_screen",
        "wide_holdout_acquire",
        "wide_holdout",
        "narrow_mechanism",
        "narrow_map_discover",
        "narrow_map_validate",
        "adjudication",
    )
    stages_to_run = set(selected_stages or stage_order)

    def status(state: str, stage: str, **extra: Any) -> None:
        now = time.time()
        stage_index = stage_order.index(stage) if stage in stage_order else len(stage_order)
        payload = {
            "state": state,
            "stage": stage,
            "stage_index": stage_index,
            "stage_count": len(stage_order),
            "profile": profile_name,
            "pid": os.getpid(),
            "started_unix": started,
            "updated_unix": now,
            "elapsed_seconds": now - started,
            "deadline_unix": deadline,
            "deadline_remaining_seconds": max(0.0, deadline - now) if deadline is not None else None,
            **extra,
        }
        _atomic_json(output / "STATUS.json", payload)
        progress = f" {extra['completed']}/{extra['total']}" if "completed" in extra else ""
        print(f"[{state}] {stage}{progress}", flush=True)

    registry = list(load_rule_registry())
    implementation_files = (
        Path(__file__),
        Path(__file__).with_name("causal_heredity.py"),
        Path(__file__).with_name("life_carrier.py"),
        Path(__file__).with_name("life_family.py"),
        Path(__file__).with_name("e19.py"),
    )
    design_payload = {
        "contract": contract.to_dict(),
        "profile": asdict(profile),
        "coverage": {"eca_rules": 256, "life_registry_rules": len(registry), "life_universe_claim": False},
        "protocol_sha256": _sha256(PROTOCOL_PATH),
        "round2_acquisition_sha256": _sha256(ROUND2_ACQUISITION),
        "life_atlas_sha256": _sha256(life_atlas),
        "life_registry_sha256": hashlib.sha256(json.dumps(registry).encode()).hexdigest(),
        "implementation_sha256": {path.name: _sha256(path) for path in implementation_files},
    }
    design_digest = hashlib.sha256(
        json.dumps(design_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _atomic_json(output / "DESIGN.json", {**design_payload, "design_digest": design_digest})
    _atomic_json(
        output / "MANIFEST.json",
        {
            "experiment": "ca_carrier_round_3",
            "profile": profile_name,
            "design_digest": design_digest,
            "started_unix": started,
            "workers": workers,
            "max_hours": max_hours,
            "environment": {
                "python": sys.version,
                "numpy": np.__version__,
                "platform": platform.platform(),
                "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            },
        },
    )

    completeness: dict[str, bool] = {}
    stage_data: dict[str, list[dict[str, Any]]] = {}

    def execute_stage(
        name: str,
        items: Sequence[dict[str, Any]],
        task: Callable[[tuple[dict[str, Any], V3Contract, V3Profile]], dict[str, Any]],
    ) -> list[dict[str, Any]]:
        execute = name in stages_to_run
        if execute and deadline is not None and time.time() >= deadline:
            status("budget_truncated", name, completed=0, total=len(items), eta_seconds=None)
        else:
            status("running" if execute else "loading", name, completed=0, total=len(items))
        rows, complete = _run_stage(
            output,
            name,
            items,
            task,
            contract,
            profile,
            design_digest=design_digest,
            workers=workers,
            resume=resume or not execute,
            execute=execute and (deadline is None or time.time() < deadline),
            deadline=deadline,
            status=status,
        )
        stage_data[name] = rows
        completeness[name] = complete
        return rows

    try:
        status("running", "calibrate")
        calibration_path = output / "CALIBRATION.json"
        calibration: dict[str, Any]
        if resume and calibration_path.exists():
            existing = json.loads(calibration_path.read_text(encoding="utf-8"))
            if existing.get("design_digest") == design_digest:
                calibration = existing["calibration"]
            else:
                calibration = calibrate_controls(select_narrow_prototype(), profile, contract)
        else:
            calibration = calibrate_controls(select_narrow_prototype(), profile, contract)
        _atomic_json(calibration_path, {"design_digest": design_digest, "calibration": calibration})
        _atomic_json(
            output / "NARROW_HYPOTHESIS.json",
            {
                "design_digest": design_digest,
                "prototype_pair_id": calibration["prototype"]["pair_id"],
                "targets": calibration["prototype"]["targets"],
                "target_basis": calibration["prototype"]["target_basis"],
                "selection_basis": calibration["selection_basis"],
                "pooled_support_id_diagnostic": calibration["prototype"]["pooled_support_id_diagnostic"],
                "historical_prototype_match_diagnostic": calibration["prototype"]["historical_prototype_match_diagnostic"],
            },
        )
        completeness["calibrate"] = True

        narrow_acquisition_items = [
            {
                "checkpoint": f"launch-{launch}",
                "entry": {"substrate": "life", "rule": ROUND2_RULE, "launch_index": launch},
                "launch_index": launch,
                "prototype": calibration["prototype"],
            }
            for launch in range(4)
        ]
        narrow_acquisitions = execute_stage("narrow_acquire", narrow_acquisition_items, _narrow_acquire_task)
        all_narrow_pairs = (
            pair_prototype_donors(narrow_acquisitions, calibration["prototype"], contract)
            if completeness["narrow_acquire"] else []
        )
        confirmation_pairs = attach_unrelated_controls(all_narrow_pairs[: profile.confirmation_pairs])
        transfer_start = profile.confirmation_pairs
        transfer_pairs = attach_unrelated_controls(
            all_narrow_pairs[transfer_start : transfer_start + profile.transfer_pairs]
        )
        mapping_start = transfer_start + profile.transfer_pairs
        mapping_pairs = attach_unrelated_controls(
            all_narrow_pairs[mapping_start : mapping_start + profile.mapping_pairs]
        )
        cohort_payload = {
            "design_digest": design_digest,
            "pair_count": len(all_narrow_pairs),
            "underpowered": len(confirmation_pairs) < profile.confirmation_pairs,
            "confirmation_pair_ids": [pair["pair_id"] for pair in confirmation_pairs],
            "transfer_pair_ids": [pair["pair_id"] for pair in transfer_pairs],
            "mapping_pair_ids": [pair["pair_id"] for pair in mapping_pairs],
        }
        _atomic_json(output / "NARROW_COHORTS.json", cohort_payload)

        replay_pairs = _legacy_pairs(calibration["prototype"], profile.replay_pairs)
        replay_items = _make_transfer_items(
            replay_pairs,
            phase="narrow-replay",
            conditions=_primary_conditions(replay=True),
            replicates=profile.replay_replicates,
            horizon=64,
        )
        execute_stage("narrow_replay", replay_items, _transfer_task)
        confirm_items = _make_transfer_items(
            confirmation_pairs,
            phase="narrow-confirm",
            conditions=_primary_conditions(),
            replicates=profile.confirmation_replicates,
            horizon=64,
        )
        execute_stage("narrow_confirm", confirm_items, _transfer_task)

        pedigree_pairs = mapping_pairs[: max(1, len(mapping_pairs) // 2)]
        pedigree_items = [
            {
                "checkpoint": f"pedigree-{index:04d}",
                "entry": {"substrate": "life", "rule": ROUND2_RULE},
                "phase": "narrow-pedigree",
                "pairs": [pair],
            }
            for index, pair in enumerate(pedigree_pairs)
        ]
        execute_stage("narrow_pedigree", pedigree_items, _pedigree_task)

        wide_items = [
            {
                "checkpoint": f"life-rule-{rule:06d}",
                "substrate": "life",
                "rule": int(rule),
                "namespace": "wide-discovery-life",
                "cap": profile.initial_life_cap,
            }
            for rule in registry
        ] + [
            {
                "checkpoint": f"eca-rule-{rule:06d}",
                "substrate": "eca",
                "rule": rule,
                "namespace": "wide-discovery-eca",
                "cap": profile.initial_eca_cap,
            }
            for rule in range(256)
        ]
        wide_discoveries = execute_stage("wide_discover", wide_items, _wide_discover_task)
        extension_selection = rank_extension_rules(wide_discoveries, profile) if completeness["wide_discover"] else {"life": [], "eca": []}
        _atomic_json(
            output / "WIDE_EXTENSION_SELECTION.json",
            {"design_digest": design_digest, "selection": extension_selection, "source_complete": completeness["wide_discover"]},
        )
        extension_items = [
            {
                "checkpoint": f"{substrate}-rule-{rule:06d}",
                "substrate": substrate,
                "rule": rule,
                "namespace": f"wide-extension-{substrate}",
                "cap": profile.extend_cap,
            }
            for substrate in ("life", "eca")
            for rule in extension_selection[substrate]
        ]
        extensions = execute_stage("wide_extend", extension_items, _wide_discover_task)
        screen_candidates = (
            select_screen_candidates(extensions, profile, contract)
            if completeness["wide_extend"] else []
        )
        _atomic_json(
            output / "WIDE_CANDIDATES.json",
            {
                "design_digest": design_digest,
                "source_complete": completeness["wide_extend"],
                "candidate_count": len(screen_candidates),
                "candidates": [
                    {
                        key: value
                        for key, value in candidate.items()
                        if key not in ("pairs", "cluster_a", "cluster_b")
                    }
                    | {
                        "cluster_a": {key: value for key, value in candidate["cluster_a"].items() if key != "member_ids"},
                        "cluster_b": {key: value for key, value in candidate["cluster_b"].items() if key != "member_ids"},
                        "pair_ids": [pair["pair_id"] for pair in candidate["pairs"][: profile.screen_pairs]],
                    }
                    for candidate in screen_candidates
                ],
            },
        )
        screen_pairs: list[dict[str, Any]] = []
        for candidate in screen_candidates:
            pairs = attach_unrelated_controls(candidate["pairs"][: profile.screen_pairs])
            for pair in pairs:
                pair["candidate_id"] = candidate["candidate_id"]
            screen_pairs.extend(pairs)
        screen_items = _make_transfer_items(
            screen_pairs,
            phase="wide-screen",
            conditions=_primary_conditions(wide=True),
            replicates=profile.screen_replicates,
            horizon=32,
        )
        screen_stage = execute_stage("wide_screen", screen_items, _transfer_task)
        if completeness["wide_screen"]:
            holdout_candidates, screen_summaries = select_holdout_candidates(
                screen_stage, screen_candidates, profile, contract
            )
        else:
            holdout_candidates, screen_summaries = [], {}
        _atomic_json(
            output / "HOLDOUT_CANDIDATES.json",
            {
                "design_digest": design_digest,
                "source_complete": completeness["wide_screen"],
                "candidate_ids": [candidate["candidate_id"] for candidate in holdout_candidates],
                "screen_summaries": screen_summaries,
            },
        )
        holdout_acquire_items = [
            {
                "checkpoint": f"candidate-{index:03d}-{candidate['substrate']}-{int(candidate['rule']):06d}",
                "candidate": candidate,
            }
            for index, candidate in enumerate(holdout_candidates)
        ]
        holdout_acquisitions = execute_stage(
            "wide_holdout_acquire", holdout_acquire_items, _holdout_acquire_task
        )
        holdout_pairs: list[dict[str, Any]] = []
        for acquisition in holdout_acquisitions:
            for pair in acquisition.get("pairs", []):
                pair["candidate_id"] = acquisition["candidate"]["candidate_id"]
                holdout_pairs.append(pair)
        holdout_items = _make_transfer_items(
            holdout_pairs,
            phase="wide-holdout",
            conditions=_primary_conditions(wide=True),
            replicates=profile.holdout_replicates,
            horizon=64,
        )
        execute_stage("wide_holdout", holdout_items, _transfer_task)

        mechanism_items = _make_transfer_items(
            transfer_pairs,
            phase="narrow-mechanism",
            conditions=_mechanism_conditions(contract),
            replicates=profile.transfer_replicates,
            horizon=64,
        )
        execute_stage("narrow_mechanism", mechanism_items, _transfer_task)

        split = min(16, max(1, len(mapping_pairs) // 2)) if mapping_pairs else 0
        mapping_discovery_pairs = mapping_pairs[:split]
        mapping_validation_pairs = mapping_pairs[split:]
        mapping_discovery_items = [
            {
                "checkpoint": f"mapping-discovery-{index:04d}",
                "entry": {"substrate": "life", "rule": ROUND2_RULE},
                "phase": "narrow-map-discover",
                "pairs": [pair],
                "tiles": list(range(16)),
            }
            for index, pair in enumerate(mapping_discovery_pairs)
        ]
        mapping_discovery = execute_stage(
            "narrow_map_discover", mapping_discovery_items, _tile_mapping_task
        )
        mapping_selection = select_mapping_tile(mapping_discovery) if completeness["narrow_map_discover"] else {"selected_tile": 0, "scores": [], "selection_rule": "source incomplete"}
        _atomic_json(
            output / "MAPPING_SELECTION.json",
            {"design_digest": design_digest, "source_complete": completeness["narrow_map_discover"], **mapping_selection},
        )
        mapping_validation_items = [
            {
                "checkpoint": f"mapping-validation-{index:04d}",
                "entry": {"substrate": "life", "rule": ROUND2_RULE},
                "phase": "narrow-map-validate",
                "pairs": [pair],
                "tiles": [int(mapping_selection["selected_tile"])],
            }
            for index, pair in enumerate(mapping_validation_pairs)
        ]
        execute_stage("narrow_map_validate", mapping_validation_items, _tile_mapping_task)

        status("running", "adjudication")
        adjudication = adjudicate_campaign(
            stage_data,
            calibration,
            profile,
            contract,
            screen_candidates,
            holdout_acquisitions,
        )
        for name, values in stage_data.items():
            if any("rows" in value for value in values):
                _write_rows_csv(output / name / f"{name}.csv", _flatten(values))
        required = tuple(name for name in stage_order if name not in ("calibrate", "adjudication"))
        all_complete = completeness.get("calibrate", False) and all(completeness.get(name, False) for name in required)
        completeness["adjudication"] = True
        results = {
            "experiment": "ca_carrier_round_3",
            "profile": profile_name,
            "state": "complete" if all_complete else "partial_budget_exhausted",
            "design_digest": design_digest,
            "contract_digest": contract.digest,
            "started_unix": started,
            "completed_unix": time.time(),
            "elapsed_seconds": time.time() - started,
            "stage_completeness": completeness,
            "narrow_cohorts": cohort_payload,
            "extension_selection": extension_selection,
            "adjudication": adjudication,
        }
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
            status("complete", "adjudication")
        else:
            _atomic_text(output / "PARTIAL", "time budget or selected-stage dependency incomplete; resume supported\n")
            status("partial_budget_exhausted", "adjudication")
        return results
    except BaseException as error:
        status("failed", "adjudication", error=repr(error))
        raise
