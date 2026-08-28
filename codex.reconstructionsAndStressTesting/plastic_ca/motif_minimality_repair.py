"""Stage-6A-R normalization-corrected local Plastic Heredity bridge.

The completed Stage-6A v1 artifacts are immutable evidence.  This module
implements a separately hashed repair programme and never opens the final
62-pair reserve.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Sequence

import numpy as np

from .causal_heredity import (
    _atomic_json,
    _atomic_text,
    _hash_seed,
    _sha256,
    _state_from_hex,
)
from .e19 import require_pinned_numpy
from .life_family import live_2x2_counts_batch
from .motif_compression import _run_json_checkpoints, decode_payload
from .motif_lineage import (
    MotifContract,
    ReaderConfiguration,
    _bootstrap,
    _step,
    apply_energy_reader,
    motif3_codes,
)
from .motif_localization import apply_local_reader
from .motif_minimality import (
    COMPACT_ID,
    DEFAULT_STAGE1_ROOT,
    DEFAULT_STAGE2_ROOT,
    DEFAULT_STAGE3_ROOT,
    DEFAULT_STAGE3R_ROOT,
    DEFAULT_STAGE4_ROOT,
    DEFAULT_STAGE5_ROOT,
    DEFAULT_STAGE5R_ROOT,
    EXACT_ID,
    MINIMALITY_PROFILES,
    QUALIFICATION_CONDITIONS,
    RULE,
    TARGETED_CONDITIONS,
    _binary_information_lower_bound,
    _configuration_payload,
    _field_distance_summary,
    _founder_bounded_payload,
    _json_candidate,
    _payload_summary,
    _phenotype_distance_outcomes,
    _quantize,
    _repair_profile_for,
    _resize_board,
    _score_state,
    _strict_confirmation_gate,
    bounded_reduce_endpoint,
    build_locality_candidates,
    embed_dynamic_seed,
    load_frozen_stage5r,
    propagate_bounded,
    select_minimality_cohorts,
)
from .motif_regeneration import (
    RegenerationContract,
    _simulate_candidate as _simulate_stage5r_candidate,
    writer_latent_from_counts,
)


ROOT = Path(__file__).resolve().parent.parent
PROTOCOL_PATH = ROOT / "CA_MOTIF_LINEAGE_STAGE6AR_PROTOCOL.md"
DEFAULT_STAGE6_ROOT = ROOT / "results/ca-motif-lineage-stage-6"
PHASES = ("audit", "bridge", "screen", "qualify", "endurance")
ORIGIN_POLICIES = ("co-located", "adjacent", "independent")
READER_MODES = ("field-local", "predecoded-local", "global")
HOPS = (2, 4, 5, 8)
SPANS = (0, 2, 4, 7, 15)
FULL_BRIDGE_ID = "repair-h08-c30-oco-located-rfield-local"
LEGACY_DIAGNOSTIC_ID = "diagnostic-legacy-h08-c30-oco-located-rfield-local"
STAGE5R_COMPACT_ANCHOR_ID = "stage5r-compact-anchor"
STAGE5R_EXACT_ANCHOR_ID = "stage5r-exact-anchor"

# Stable scientific Stage-6A artifacts, frozen after the completed reference
# run.  Runtime/worker manifests are deliberately excluded.
FROZEN_STAGE6A_SHA256 = {
    "DESIGN.json": "cf37d4ae87b3d2252990a42563403c1d7978b863bcfb2cc83f771786bfdc3b9c",
    "COHORTS.json": "e2b754df895be0e6387fa232b500575965f8e5487bf6d33bed56b3a4b3c851a5",
    "LOCALITY_MODELS.json": "9b4074cd95d0e2e2c56cfa2eaaefe6137d7c61f4e0ac5371d27ba4496ad87210",
    "LOCALITY_MODELS.npz": "f68a37a7e5087808cc26086900261570396b75d0af8ff15acd659abbc95cec19",
    "MECHANISM_AUDIT.json": "c7dbfa3e986bfa2ecaf38458e73788c8e8040499159b3a93c291ef8aee8c6492",
    "locality/RESULTS.json": "23999f52e65a1c1d16a0573d9c3cfea36271412449c8da11735c374a1812d759",
    "locality/STAGE_DECISION.json": (
        "f4ba1dd8b02f028a0c8fea2704393a567c4cf017d43eafe41857d6f21fedeed6"
    ),
    "locality/ANCHOR_DIAGNOSTICS.json": (
        "7c92ada612a437fccc6a9db270b59a7841b331362965928e693293a78b85f8c8"
    ),
}


@dataclass(frozen=True)
class MinimalityRepairContract:
    implementation_version: str = "ca-motif-lineage-stage6ar-cleanroom-v1"
    namespace: str = "plastic-ca-motif-lineage-stage6ar-v1"
    rule: int = RULE
    width: int = 16
    height: int = 16
    generation_sweeps: int = 64
    read_sweeps: int = 32
    write_start: int = 49
    write_end: int = 64
    observe_start: int = 57
    process_noise: float = 0.002
    repair_gain: float = 0.50
    stale_retention: float = 0.50
    carrier_corruption: float = 0.01
    screen_generation4: float = 0.20
    screen_generation8: float = 0.15
    screen_generation16: float = 0.10
    screen_anchor_retention: float = 0.70
    control_advantage: float = 0.10
    transport_advantage: float = 0.10
    regeneration_advantage: float = 0.10
    consolidation_advantage: float = 0.10
    translation_retention: float = 0.70
    survival_gate: float = 0.90
    loss_fraction: float = 0.70
    rescue_fraction: float = 0.70
    strict_alpha: float = 0.025
    decoder_splits: int = 4
    bounded_hops: int = 5
    bounded_consolidation_steps: int = 14
    max_workers: int = 4
    max_hours: float = 4.0
    science_reserve_seconds: float = 900.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "visible_reset": "bitwise-identical matched board before every generation",
                "writer_normalization": "count-corrected spatial mean v2",
                "legacy_normalization_role": "non-promotable diagnostic only",
                "random_streams": "semantic and candidate-order independent",
                "reserve_policy": "62 final-audit trajectories remain unloaded",
                "claim_boundary": "developmental engineered synthetic CA heredity",
            }
        )
        return payload

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


@dataclass(frozen=True)
class MinimalityRepairProfile:
    bridge_pairs: int
    bridge_replicates: int
    bridge_generations: int
    screen_pairs: int
    screen_replicates: int
    screen_generations: int
    qualification_pairs: int
    qualification_replicates: int
    qualification_generations: int
    endurance_pairs: int
    endurance_replicates: int
    endurance_generations: int
    bootstrap_resamples: int


REPAIR_PROFILES: dict[str, MinimalityRepairProfile] = {
    "smoke": MinimalityRepairProfile(2, 2, 4, 2, 2, 8, 2, 2, 16, 2, 2, 64, 100),
    "pilot": MinimalityRepairProfile(8, 4, 4, 16, 4, 8, 16, 4, 16, 8, 4, 64, 1_000),
    "reference": MinimalityRepairProfile(
        32, 4, 4, 64, 8, 8, 96, 16, 16, 32, 8, 64, 10_000
    ),
}
PUBLIC_PROFILES = tuple(REPAIR_PROFILES)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0.0:
        return 0.0
    return float(np.dot(left, right) / denominator)


def corrected_window_moments(
    endpoint_mean: np.ndarray,
    signs: np.ndarray,
    span: int,
    write_sweeps: int,
    alpha: float,
) -> np.ndarray:
    """Convert a routed spatial mean to count-equivalent Walsh moments."""

    if span < 0:
        raise ValueError("span must be non-negative")
    if write_sweeps < 1:
        raise ValueError("write_sweeps must be positive")
    site_count = float((span + 1) ** 2)
    signed_count = np.asarray(endpoint_mean, dtype=np.float64) * site_count
    prior = float(alpha) * np.asarray(signs, dtype=np.float64).sum(axis=0)
    return (signed_count + prior) / (
        float(write_sweeps) * site_count + 512.0 * float(alpha)
    )


def legacy_window_moments(
    endpoint_mean: np.ndarray,
    signs: np.ndarray,
    span: int,
    write_sweeps: int,
    alpha: float,
) -> np.ndarray:
    """The frozen Stage-6A v1 formula, retained only for diagnosis."""

    site_count = float((span + 1) ** 2)
    prior = float(alpha) * np.asarray(signs, dtype=np.float64).sum(axis=0)
    return (np.asarray(endpoint_mean, dtype=np.float64) + prior) / (
        float(write_sweeps) * site_count + 512.0 * float(alpha)
    )


def repair_origins(
    pair_id: str,
    generation: int,
    replicates: int,
    extent: int,
    policy: str,
    *,
    translated: bool = False,
) -> np.ndarray:
    """Return paired lineage origins under an explicit causal-geometry policy."""

    if policy not in ORIGIN_POLICIES:
        raise ValueError(f"unknown Stage-6A-R origin policy {policy!r}")
    if generation < 1:
        raise ValueError("generation must be positive")
    base_rng = np.random.default_rng(
        _hash_seed("stage6ar-origin-base", pair_id, extent)
    )
    half = np.stack(
        (
            base_rng.integers(0, extent, size=replicates),
            base_rng.integers(0, extent, size=replicates),
        ),
        axis=1,
    ).astype(np.int64)
    if policy == "independent":
        rng = np.random.default_rng(
            _hash_seed("stage6ar-origin-independent", pair_id, generation, extent)
        )
        half = np.stack(
            (
                rng.integers(0, extent, size=replicates),
                rng.integers(0, extent, size=replicates),
            ),
            axis=1,
        ).astype(np.int64)
    elif policy == "adjacent":
        offsets = np.asarray(
            [
                (-1, -1),
                (-1, 0),
                (-1, 1),
                (0, -1),
                (0, 1),
                (1, -1),
                (1, 0),
                (1, 1),
            ],
            dtype=np.int64,
        )
        for step in range(2, generation + 1):
            rng = np.random.default_rng(
                _hash_seed("stage6ar-origin-adjacent", pair_id, step, extent)
            )
            half = (half + offsets[rng.integers(0, len(offsets), size=replicates)]) % extent
    if translated:
        half = (half + np.asarray((5, 7), dtype=np.int64)) % extent
    paired = np.concatenate((half, half), axis=0)
    return paired.astype(np.int16)


def toroidal_chebyshev_distance(
    left: np.ndarray, right: np.ndarray, extent: int
) -> np.ndarray:
    delta = np.abs(np.asarray(left, dtype=np.int64) - np.asarray(right, dtype=np.int64))
    delta = np.minimum(delta, extent - delta)
    return np.max(delta, axis=1)


def _semantic_uniforms(
    pair_id: str,
    purpose: str,
    generation: int,
    sweep: int,
    replicates: int,
    extent: int,
) -> np.ndarray:
    rng = np.random.default_rng(
        _hash_seed("stage6ar-stream", pair_id, purpose, generation, sweep, extent)
    )
    half = rng.random((replicates, extent, extent))
    return np.concatenate((half, half), axis=0)


def heldout_lineage_diagnostics(
    vectors: np.ndarray,
    replicates: int,
    seed: int,
    splits: int = 4,
) -> dict[str, float]:
    """Nearest-centroid decoding with ties scored at chance and reported."""

    values = np.asarray(vectors, dtype=np.float64)
    if values.ndim != 2 or len(values) != 2 * replicates:
        raise ValueError("decoder vectors must have two equal history blocks")
    if replicates < 2:
        return {"balanced_accuracy": 0.5, "tie_fraction": 1.0}
    scores: list[float] = []
    ties: list[float] = []
    for split in range(splits):
        permutation = np.random.default_rng(_hash_seed(seed, split)).permutation(replicates)
        take = max(1, replicates // 2)
        train = permutation[:take]
        test = permutation[take:]
        if not len(test):
            test = train
        pooled = values[np.concatenate((train, replicates + train))]
        centre = pooled.mean(axis=0)
        scale = pooled.std(axis=0)
        scale[scale < 1e-8] = 1.0
        standardized = (values - centre) / scale
        centroid_a = standardized[train].mean(axis=0)
        centroid_b = standardized[replicates + train].mean(axis=0)
        distance_a_a = np.linalg.norm(standardized[test] - centroid_a, axis=1)
        distance_a_b = np.linalg.norm(standardized[test] - centroid_b, axis=1)
        distance_b_b = np.linalg.norm(
            standardized[replicates + test] - centroid_b, axis=1
        )
        distance_b_a = np.linalg.norm(
            standardized[replicates + test] - centroid_a, axis=1
        )
        tie_a = np.isclose(distance_a_a, distance_a_b, rtol=0.0, atol=1e-12)
        tie_b = np.isclose(distance_b_b, distance_b_a, rtol=0.0, atol=1e-12)
        correct_a = (distance_a_a < distance_a_b).astype(np.float64) + 0.5 * tie_a
        correct_b = (distance_b_b < distance_b_a).astype(np.float64) + 0.5 * tie_b
        scores.append(0.5 * float(correct_a.mean() + correct_b.mean()))
        ties.append(0.5 * float(tie_a.mean() + tie_b.mean()))
    return {
        "balanced_accuracy": float(np.mean(scores)),
        "tie_fraction": float(np.mean(ties)),
    }


def _transition_summary(
    entry: np.ndarray, exit_payload: np.ndarray, replicates: int
) -> dict[str, float]:
    entry_values = np.asarray(entry, dtype=np.float64).reshape(2, replicates, -1)
    exit_values = np.asarray(exit_payload, dtype=np.float64).reshape(2, replicates, -1)
    entry_centroids = entry_values.mean(axis=1)
    exit_centroids = exit_values.mean(axis=1)
    entry_delta = entry_centroids[0] - entry_centroids[1]
    exit_delta = exit_centroids[0] - exit_centroids[1]
    entry_norm = float(np.linalg.norm(entry_delta))
    exit_norm = float(np.linalg.norm(exit_delta))
    return {
        "entry_centroid_l2": entry_norm,
        "exit_centroid_l2": exit_norm,
        "centroid_retention": exit_norm / entry_norm if entry_norm > 0.0 else 0.0,
        "parent_child_delta_cosine": _cosine(entry_delta, exit_delta),
    }


def _boundary_intervention(
    payload: np.ndarray,
    candidate: dict[str, Any],
    condition: str,
    generation: int,
    pair_id: str,
    replicates: int,
    source_exits: Sequence[np.ndarray] | None,
    contract: MinimalityRepairContract,
) -> tuple[np.ndarray, float]:
    result = np.asarray(payload, dtype=np.float32).copy()
    if condition == "zero_every_boundary":
        result.fill(0.0)
    elif condition == "shuffle_every_boundary":
        permutation = np.random.default_rng(
            _hash_seed(contract.namespace, pair_id, "channel-shuffle", generation)
        ).permutation(result.shape[-1])
        result = result[:, permutation]
    elif condition == "opposite_founder" and generation == 1:
        result = np.concatenate((result[replicates:], result[:replicates]), axis=0)
    elif condition in (
        "ablate_after_g2",
        "rescue_same_enter_g4",
        "rescue_opposite_enter_g4",
    ) and generation == 3:
        result.fill(0.0)
    elif condition in ("rescue_same_enter_g4", "rescue_opposite_enter_g4") and generation == 4:
        if source_exits is None or len(source_exits) < 3:
            raise ValueError("rescue requires contemporaneous intact-sister exits")
        result = np.asarray(source_exits[2], dtype=np.float32).copy()
        if condition == "rescue_opposite_enter_g4":
            result = np.concatenate((result[replicates:], result[:replicates]), axis=0)
    elif condition == "carrier_corruption_1":
        half = np.random.default_rng(
            _hash_seed(contract.namespace, pair_id, "corruption", generation)
        ).random((replicates, result.shape[-1])) < contract.carrier_corruption
        result[np.concatenate((half, half), axis=0)] *= -1.0
    return _quantize(result, candidate["codec_model"])


def _damage_payload(
    payload: np.ndarray,
    candidate: dict[str, Any],
    pair_id: str,
    generation: int,
    replicates: int,
    stress_id: str,
    stress: dict[str, float | int],
) -> tuple[np.ndarray, float]:
    result = np.asarray(payload, dtype=np.float32).copy()
    erasure = float(stress.get("erasure", 0.0))
    sign = float(stress.get("sign_corruption", 0.0))
    if erasure:
        half = np.random.default_rng(
            _hash_seed("stage6ar-damage", pair_id, stress_id, generation, "erase")
        ).random((replicates, result.shape[-1])) < erasure
        result[np.concatenate((half, half), axis=0)] = 0.0
    if sign:
        half = np.random.default_rng(
            _hash_seed("stage6ar-damage", pair_id, stress_id, generation, "sign")
        ).random((replicates, result.shape[-1])) < sign
        result[np.concatenate((half, half), axis=0)] *= -1.0
    return _quantize(result, candidate["codec_model"])


def _repair_candidate_id(
    hops: int,
    span: int,
    origin_policy: str,
    reader_mode: str = "field-local",
    *,
    diagnostic: str | None = None,
) -> str:
    prefix = f"{diagnostic}-" if diagnostic else "repair-"
    return (
        f"{prefix}h{hops:02d}-c{2 * span:02d}-o{origin_policy}"
        f"-r{reader_mode}"
    )


def build_repair_candidates(
    frozen: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build the preregistered bridge matrix and non-promotable diagnostics."""

    stage6_candidates, _anchors = build_locality_candidates(frozen)
    base_by_geometry = {
        (int(row["germination_hops"]), int(row["consolidation_span"])): row
        for row in stage6_candidates
    }
    candidates: list[dict[str, Any]] = []
    for hops in HOPS:
        for span in SPANS:
            base = base_by_geometry[(hops, span)]
            for policy in ORIGIN_POLICIES:
                candidates.append(
                    {
                        **base,
                        "candidate_id": _repair_candidate_id(hops, span, policy),
                        "origin_policy": policy,
                        "reader_mode": "field-local",
                        "normalization": "count-correct-v2",
                        "promotable": bool(
                            hops <= 5 and 2 * span <= 14 and policy != "independent"
                        ),
                    }
                )
    full_base = base_by_geometry[(8, 15)]
    diagnostics = [
        {
            **full_base,
            "candidate_id": _repair_candidate_id(
                8, 15, "co-located", "predecoded-local", diagnostic="diagnostic"
            ),
            "origin_policy": "co-located",
            "reader_mode": "predecoded-local",
            "normalization": "count-correct-v2",
            "promotable": False,
        },
        {
            **full_base,
            "candidate_id": _repair_candidate_id(
                8, 15, "co-located", "global", diagnostic="diagnostic"
            ),
            "origin_policy": "co-located",
            "reader_mode": "global",
            "normalization": "count-correct-v2",
            "promotable": False,
        },
        {
            **full_base,
            "candidate_id": LEGACY_DIAGNOSTIC_ID,
            "origin_policy": "co-located",
            "reader_mode": "field-local",
            "normalization": "legacy-v1",
            "promotable": False,
        },
    ]
    if len(candidates) != 60:
        raise AssertionError("Stage-6A-R bridge matrix must contain 60 candidates")
    if FULL_BRIDGE_ID not in {str(row["candidate_id"]) for row in candidates}:
        raise AssertionError("corrected full bridge is missing")
    return candidates, diagnostics


def _causal_overlap_fraction(
    current_origins: np.ndarray,
    next_origins: np.ndarray,
    extent: int,
    hops: int,
    span: int,
) -> float:
    overlap: list[float] = []
    for current, nxt in zip(
        np.asarray(current_origins, dtype=np.int64),
        np.asarray(next_origins, dtype=np.int64),
    ):
        reached: set[tuple[int, int]] = set()
        for dy in range(-hops, hops + 1):
            for dx in range(-hops, hops + 1):
                reached.add(((int(current[0]) + dy) % extent, (int(current[1]) + dx) % extent))
        writer_sites = {
            ((int(nxt[0]) - dy) % extent, (int(nxt[1]) - dx) % extent)
            for dy in range(span + 1)
            for dx in range(span + 1)
        }
        overlap.append(float(len(reached & writer_sites) / max(1, len(writer_sites))))
    return float(np.mean(overlap))


def simulate_repaired_lineage(
    pair: dict[str, Any],
    configuration: ReaderConfiguration,
    candidate: dict[str, Any],
    condition: str,
    replicates: int,
    generations: int,
    reference: dict[int, dict[str, np.ndarray]],
    writer_contract: MotifContract,
    contract: MinimalityRepairContract,
    *,
    extent: int = 16,
    stress_id: str = "ordinary",
    stress: dict[str, float | int] | None = None,
    source_exits: Sequence[np.ndarray] | None = None,
    retain_exits: bool = False,
    rule_override: int | None = None,
) -> tuple[dict[str, Any], list[np.ndarray]]:
    """Run a reset lineage with corrected local writing and explicit geometry."""

    valid = set(QUALIFICATION_CONDITIONS) | {"founder_clamped"}
    if condition not in valid:
        raise ValueError(f"unknown Stage-6A-R condition {condition!r}")
    if extent != 16:
        raise ValueError("Stage-6A-R is restricted to the registered 16x16 bridge")
    stress = dict(stress or {})
    pair_id = str(pair["pair_id"])
    candidate_id = str(candidate["candidate_id"])
    rule = int(rule_override if rule_override is not None else contract.rule)
    reset_a = _resize_board(
        _state_from_hex("life", pair["donor_a"]["initial_state_hex"]), extent
    )
    reset_b = _resize_board(
        _state_from_hex("life", pair["donor_b"]["initial_state_hex"]), extent
    )
    if not np.array_equal(reset_a, reset_b):
        raise AssertionError(f"visible reset mismatch in pair {pair_id}")
    reset = np.repeat(reset_a[None, ...], 2 * replicates, axis=0)
    reference_probability = reference[configuration.write_window]["motif_probability"]
    payload, founder_terminal = _founder_bounded_payload(
        pair,
        candidate,
        reference_probability,
        replace(writer_contract, rule=rule),
        replicates,
        extent,
        rule,
    )
    founder_payload = payload.copy()
    if condition == "founder_write_disabled":
        payload.fill(0.0)
    alive = np.ones(2 * replicates, dtype=np.bool_)
    checkpoints = {
        value for value in (1, 2, 4, 8, 16, 32, 64) if value <= generations
    }
    outcomes: dict[str, Any] = {}
    decoders: dict[str, Any] = {}
    carrier_history: dict[str, Any] = {}
    writer_history: dict[str, Any] = {}
    origin_history: dict[str, Any] = {}
    exits: list[np.ndarray] = []
    coverage_values: list[float] = []
    clipping_values: list[float] = []
    process_noise = float(stress.get("process_noise", contract.process_noise))
    repair_gain = float(stress.get("repair_gain", contract.repair_gain))
    codec = candidate["codec_model"]
    basis = np.asarray(codec["basis"], dtype=np.float32)
    signs = basis.astype(np.float64) * math.sqrt(512.0)
    span = int(candidate["consolidation_span"])
    hops = int(candidate["germination_hops"])
    origin_policy = str(candidate["origin_policy"])
    reader_mode = str(candidate["reader_mode"])
    normalization = str(candidate["normalization"])
    if reader_mode not in READER_MODES:
        raise ValueError(f"unknown reader mode {reader_mode!r}")
    if normalization not in ("count-correct-v2", "legacy-v1"):
        raise ValueError(f"unknown writer normalization {normalization!r}")
    write_sweeps = contract.write_end - contract.write_start + 1

    for generation in range(1, generations + 1):
        payload, clipping = _boundary_intervention(
            payload,
            candidate,
            condition,
            generation,
            pair_id,
            replicates,
            source_exits,
            contract,
        )
        clipping_values.append(clipping)
        payload, clipping = _damage_payload(
            payload, candidate, pair_id, generation, replicates, stress_id, stress
        )
        clipping_values.append(clipping)
        entry_payload = payload.copy()
        translated = condition == "translated_patch"
        origins = repair_origins(
            pair_id,
            generation,
            replicates,
            extent,
            origin_policy,
            translated=translated,
        )
        next_origins = repair_origins(
            pair_id,
            generation + 1,
            replicates,
            extent,
            origin_policy,
            translated=translated,
        )
        field, occupied = embed_dynamic_seed(entry_payload, origins, extent)
        wave_trace: list[float] = []
        if condition == "regeneration_disabled":
            field.fill(0.0)
        elif condition != "transport_disabled":
            field, occupied, wave_trace = propagate_bounded(
                field,
                occupied,
                hops,
                communication_cut=condition == "communication_cut",
            )
        coverage_values.append(float(np.mean(occupied)))
        field_distance = _field_distance_summary(field, occupied, origins)
        uniform = bool(
            np.all(occupied)
            and np.array_equal(field, np.broadcast_to(field[:, 0:1, 0:1], field.shape))
        )
        decoded_entry = decode_payload(
            np.zeros_like(entry_payload)
            if condition == "regeneration_disabled"
            else entry_payload,
            codec,
        )

        state = reset.copy()
        state[~alive] = False
        if not np.array_equal(state[alive], reset[alive]):
            raise AssertionError("visible reset was not bitwise identical")
        recent: deque[np.ndarray] = deque(maxlen=writer_contract.observation_window)
        site_sign_sum = np.zeros(
            (2 * replicates, extent, extent, int(candidate["rank"])),
            dtype=np.float64,
        )
        for sweep in range(1, contract.generation_sweeps + 1):
            predicted = _step(state, rule)
            if condition != "read_disabled" and sweep <= contract.read_sweeps:
                uniforms = _semantic_uniforms(
                    pair_id, "read", generation, sweep, replicates, extent
                )
                strength = configuration.strength * repair_gain / 0.50
                if reader_mode == "global":
                    predicted = apply_energy_reader(
                        predicted, decoded_entry, uniforms, strength
                    )
                elif reader_mode == "predecoded-local":
                    from .motif_minimality import apply_masked_payload_reader

                    predicted = apply_masked_payload_reader(
                        predicted, decoded_entry, occupied, uniforms, strength
                    )
                elif uniform:
                    predicted = apply_energy_reader(
                        predicted, decode_payload(field[:, 0, 0], codec), uniforms, strength
                    )
                else:
                    predicted = apply_local_reader(
                        predicted, field, basis, uniforms, strength
                    )
            predicted ^= _semantic_uniforms(
                pair_id, "process", generation, sweep, replicates, extent
            ) < process_noise
            predicted[~alive] = False
            state = predicted
            if contract.write_start <= sweep <= contract.write_end:
                site_sign_sum += signs[motif3_codes(state)]
            if sweep >= contract.observe_start:
                recent.append(live_2x2_counts_batch(state))
        alive &= state.any(axis=(1, 2))

        selected_span = 0 if condition == "consolidation_disabled" else span
        endpoint_mean = bounded_reduce_endpoint(
            site_sign_sum, selected_span, next_origins
        )
        alpha = float(writer_contract.jeffreys_alpha)
        if normalization == "legacy-v1":
            moments = legacy_window_moments(
                endpoint_mean, signs, selected_span, write_sweeps, alpha
            )
        else:
            moments = corrected_window_moments(
                endpoint_mean, signs, selected_span, write_sweeps, alpha
            )
        latent = (
            float(stress.get("writer_gain", 1.0))
            * (moments @ np.asarray(candidate["weight"], dtype=np.float64))
            + np.asarray(candidate["bias"], dtype=np.float64)
        )
        if condition == "no_rewrite":
            next_payload, clipping = _quantize(
                entry_payload * np.float32(contract.stale_retention), codec
            )
        elif condition == "write_disabled":
            next_payload = np.zeros_like(entry_payload)
            clipping = 0.0
        elif condition == "founder_clamped":
            next_payload = founder_payload.copy()
            clipping = 0.0
        else:
            next_payload, clipping = _quantize(latent, codec)
        clipping_values.append(clipping)
        payload = next_payload
        payload[~alive] = 0.0

        if generation in checkpoints:
            outcome, phenotype = _score_state(
                state, recent, pair, founder_terminal, replicates, writer_contract
            )
            outcome["distance_bands"] = _phenotype_distance_outcomes(
                state,
                origins,
                pair,
                alive,
                replicates,
                writer_contract,
            )
            outcomes[str(generation)] = outcome
            carrier_decoder = heldout_lineage_diagnostics(
                payload,
                replicates,
                _hash_seed(
                    contract.namespace,
                    pair_id,
                    condition,
                    stress_id,
                    generation,
                    "carrier",
                ),
                contract.decoder_splits,
            )
            phenotype_decoder = heldout_lineage_diagnostics(
                phenotype,
                replicates,
                _hash_seed(
                    contract.namespace,
                    pair_id,
                    condition,
                    stress_id,
                    generation,
                    "phenotype",
                ),
                contract.decoder_splits,
            )
            decoders[str(generation)] = {
                "carrier_balanced_accuracy": carrier_decoder["balanced_accuracy"],
                "carrier_tie_fraction": carrier_decoder["tie_fraction"],
                "carrier_information_lower_bound_bits": _binary_information_lower_bound(
                    carrier_decoder["balanced_accuracy"]
                ),
                "phenotype_balanced_accuracy": phenotype_decoder["balanced_accuracy"],
                "phenotype_tie_fraction": phenotype_decoder["tie_fraction"],
                "phenotype_information_lower_bound_bits": _binary_information_lower_bound(
                    phenotype_decoder["balanced_accuracy"]
                ),
            }
            transition = _transition_summary(entry_payload, payload, replicates)
            carrier_history[str(generation)] = {
                "entry": _payload_summary(entry_payload, replicates),
                "exit": _payload_summary(payload, replicates),
                "transition": transition,
                "occupied_fraction_after_germination": float(np.mean(occupied)),
                "uniform_after_germination": uniform,
                "wave_trace": wave_trace,
                "distance_bands": field_distance,
                "surviving_futures": int(np.count_nonzero(alive)),
            }
            writer_history[str(generation)] = {
                "normalization": normalization,
                "span": selected_span,
                "window_site_count": (selected_span + 1) ** 2,
                "endpoint_mean_abs": float(np.mean(np.abs(endpoint_mean))),
                "moment_mean_abs": float(np.mean(np.abs(moments))),
                "latent_mean_abs": float(np.mean(np.abs(latent))),
                "latent_centroid_l2": _payload_summary(latent, replicates)["centroid_l2"],
            }
            displacement = toroidal_chebyshev_distance(origins, next_origins, extent)
            origin_history[str(generation)] = {
                "policy": origin_policy,
                "mean_displacement": float(np.mean(displacement)),
                "max_displacement": int(np.max(displacement)),
                "causal_overlap_fraction": _causal_overlap_fraction(
                    origins, next_origins, extent, hops, selected_span
                ),
                "translation_applied_to_read_and_write": translated,
            }
        if retain_exits:
            exits.append(payload.copy())

    return (
        {
            "candidate_id": candidate_id,
            "condition": condition,
            "stress_id": stress_id,
            "stress": stress,
            "rule": rule,
            "extent": extent,
            "reset_sha256": hashlib.sha256(reset_a.tobytes()).hexdigest(),
            "reset_asserted_before_every_generation": True,
            "inherited_payload_bits": int(candidate["payload_bits"]),
            "occupancy_bits": int(candidate["occupancy_bits"]),
            "germination_hops": hops,
            "consolidation_span": span,
            "consolidation_steps": 2 * span,
            "origin_policy": origin_policy,
            "reader_mode": reader_mode,
            "normalization": normalization,
            "founder_payload": _payload_summary(founder_payload, replicates),
            "boundary_clipping_fraction_mean": float(np.mean(clipping_values)),
            "germination_coverage_mean": float(np.mean(coverage_values)),
            "outcomes": outcomes,
            "decoders": decoders,
            "carrier_history": carrier_history,
            "writer_history": writer_history,
            "origin_history": origin_history,
        },
        exits,
    )


def _repair_task(
    payload: tuple[
        dict[str, Any],
        list[dict[str, Any]],
        MotifContract,
        MinimalityRepairContract,
        dict[int, dict[str, np.ndarray]],
    ]
) -> dict[str, Any]:
    item, candidates, writer_contract, contract, reference = payload
    candidate = next(
        row for row in candidates if row["candidate_id"] == item["candidate_id"]
    )
    configuration = ReaderConfiguration(**item["configuration"])
    replicates = int(item["replicates"])
    generations = int(item["generations"])
    conditions = tuple(item.get("conditions", ("intact",)))
    results: dict[str, Any] = {}
    exits: list[np.ndarray] | None = None
    if "intact" in conditions or any("rescue" in value for value in conditions):
        intact, exits = simulate_repaired_lineage(
            item["pair"],
            configuration,
            candidate,
            "intact",
            replicates,
            generations,
            reference,
            writer_contract,
            contract,
            retain_exits=True,
        )
        if "intact" in conditions:
            results["intact"] = intact
    for condition in conditions:
        if condition == "intact":
            continue
        result, _ = simulate_repaired_lineage(
            item["pair"],
            configuration,
            candidate,
            condition,
            replicates,
            generations,
            reference,
            writer_contract,
            contract,
            source_exits=exits,
        )
        results[condition] = result
    return {
        "checkpoint": item["checkpoint"],
        "pair_id": item["pair"]["pair_id"],
        "candidate_id": candidate["candidate_id"],
        "replicates": replicates,
        "generations": generations,
        "conditions": results,
    }


def _stage5r_anchor_task(
    payload: tuple[
        dict[str, Any],
        list[dict[str, Any]],
        MotifContract,
        MinimalityRepairContract,
        dict[int, dict[str, np.ndarray]],
    ]
) -> dict[str, Any]:
    item, candidates, writer_contract, _contract, reference = payload
    candidate = next(
        row for row in candidates if row["candidate_id"] == item["candidate_id"]
    )
    original = {**candidate, "candidate_id": candidate["original_candidate_id"]}
    result, _ = _simulate_stage5r_candidate(
        item["pair"],
        ReaderConfiguration(**item["configuration"]),
        original,
        item["walsh_model"],
        "intact",
        int(item["replicates"]),
        int(item["generations"]),
        reference,
        writer_contract,
        RegenerationContract(),
    )
    result["candidate_id"] = candidate["candidate_id"]
    return {
        "checkpoint": item["checkpoint"],
        "pair_id": item["pair"]["pair_id"],
        "candidate_id": candidate["candidate_id"],
        "replicates": int(item["replicates"]),
        "generations": int(item["generations"]),
        "conditions": {"intact": result},
    }


def _condition_values(
    rows: Sequence[dict[str, Any]],
    candidate_id: str,
    condition: str,
    generation: int,
    metric: str = "crossover",
) -> list[float]:
    values: list[float] = []
    for row in rows:
        if row.get("candidate_id") != candidate_id:
            continue
        try:
            outcome = row["conditions"][condition]["outcomes"][str(generation)]
        except KeyError:
            continue
        if metric == "survival":
            values.append(float(outcome["survival"]))
        else:
            values.append(float(outcome["primary"][metric]))
    return values


def _paired_advantage(
    rows: Sequence[dict[str, Any]],
    candidate_id: str,
    left_condition: str,
    right_condition: str,
    generation: int,
) -> list[float]:
    left = _condition_values(rows, candidate_id, left_condition, generation)
    right = _condition_values(rows, candidate_id, right_condition, generation)
    if len(left) != len(right):
        raise ValueError(
            f"paired values do not align for {left_condition} and {right_condition}"
        )
    return [a - b for a, b in zip(left, right)]


def _diagnostic_values(
    rows: Sequence[dict[str, Any]],
    candidate_id: str,
    generation: int,
    section: str,
    key: str,
) -> list[float]:
    values: list[float] = []
    for row in rows:
        if row.get("candidate_id") != candidate_id:
            continue
        try:
            values.append(
                float(row["conditions"]["intact"][section][str(generation)][key])
            )
        except KeyError:
            continue
    return values


def _boot(
    values: Sequence[float],
    profile: MinimalityRepairProfile,
    contract: MinimalityRepairContract,
    *key: object,
) -> dict[str, Any]:
    return _bootstrap(
        values,
        profile.bootstrap_resamples,
        _hash_seed(contract.namespace, *key),
        contract.strict_alpha,
    )


def _candidate_public(candidate: dict[str, Any]) -> dict[str, Any]:
    return _json_candidate(candidate) | {
        "origin_policy": candidate["origin_policy"],
        "reader_mode": candidate["reader_mode"],
        "normalization": candidate["normalization"],
        "promotable": bool(candidate["promotable"]),
    }


def load_frozen_stage6a(stage6_root: Path = DEFAULT_STAGE6_ROOT) -> dict[str, Any]:
    root = stage6_root.resolve()
    for relative, expected in FROZEN_STAGE6A_SHA256.items():
        path = root / relative
        if not path.exists():
            raise FileNotFoundError(f"missing frozen Stage-6A artifact: {path}")
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"frozen Stage-6A artifact changed: {relative}")
    design = _load_json(root / "DESIGN.json")
    results = _load_json(root / "locality/RESULTS.json")
    decision = _load_json(root / "locality/STAGE_DECISION.json")
    diagnostics = _load_json(root / "locality/ANCHOR_DIAGNOSTICS.json")
    cohorts = _load_json(root / "COHORTS.json")
    expected_design = "ea27518e0974816469eb4d8540f0e543d16c552e4f21d06caae2a8425185d25a"
    if design.get("design_digest") != expected_design:
        raise ValueError("unexpected frozen Stage-6A design")
    if results.get("state") != "complete":
        raise ValueError("Stage 6A is not complete")
    if decision.get("decision") != "no_bounded_candidate_passed_registered_gate":
        raise ValueError("Stage 6A no longer has its registered negative decision")
    if cohorts.get("final_audit_trajectory_state") != "untouched":
        raise ValueError("Stage-6A final reserve is no longer untouched")
    return {
        "root": root,
        "design": design,
        "results": results,
        "decision": decision,
        "diagnostics": diagnostics,
        "cohorts": cohorts,
        "hashes": dict(FROZEN_STAGE6A_SHA256),
    }


def correction_audit(
    frozen: dict[str, Any],
    frozen6a: dict[str, Any],
    writer_contract: MotifContract,
    contract: MinimalityRepairContract,
    cohorts: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    candidates, _diagnostics = build_repair_candidates(frozen)
    full = next(row for row in candidates if row["candidate_id"] == FULL_BRIDGE_ID)
    codec = full["codec_model"]
    basis = np.asarray(codec["basis"], dtype=np.float64)
    signs = basis * math.sqrt(512.0)
    write_sweeps = contract.write_end - contract.write_start + 1
    alpha = float(writer_contract.jeffreys_alpha)
    rng = np.random.default_rng(_hash_seed(contract.namespace, "correction-audit"))
    codes = rng.integers(
        0, 512, size=(3, write_sweeps, contract.height, contract.width), dtype=np.int64
    )
    site_sign_sum = signs[codes].sum(axis=1)
    origins = np.asarray([[0, 0], [5, 7], [11, 3]], dtype=np.int16)
    span_errors: dict[str, float] = {}
    for span in SPANS:
        endpoint = bounded_reduce_endpoint(site_sign_sum, span, origins)
        corrected = corrected_window_moments(
            endpoint, signs, span, write_sweeps, alpha
        )
        direct_rows: list[np.ndarray] = []
        for sample, (origin_y, origin_x) in enumerate(origins.astype(np.int64)):
            ys = (origin_y - np.arange(span + 1)) % contract.height
            xs = (origin_x - np.arange(span + 1)) % contract.width
            signed_count = site_sign_sum[sample][np.ix_(ys, xs)].sum(axis=(0, 1))
            direct_rows.append(
                (signed_count + alpha * signs.sum(axis=0))
                / (write_sweeps * (span + 1) ** 2 + 512.0 * alpha)
            )
        direct = np.stack(direct_rows)
        span_errors[str(span)] = float(np.max(np.abs(corrected - direct)))

    full_endpoint = bounded_reduce_endpoint(site_sign_sum, 15, origins)
    full_moments = corrected_window_moments(
        full_endpoint, signs, 15, write_sweeps, alpha
    )
    repaired_payload, _ = _quantize(
        full_moments @ np.asarray(full["weight"], dtype=np.float64)
        + np.asarray(full["bias"], dtype=np.float64),
        codec,
    )
    counts = np.stack(
        [np.bincount(row.reshape(-1), minlength=512) for row in codes]
    ).astype(np.float64)
    historical_payload, _ = writer_latent_from_counts(
        counts,
        frozen["by_candidate"][COMPACT_ID],
        frozen["stage4"]["reference"][
            frozen["stage4"]["configuration"].write_window
        ]["motif_probability"],
        frozen["stage4"]["winner_model"],
        writer_contract,
        RegenerationContract(),
    )
    payload_exact = bool(np.array_equal(repaired_payload, historical_payload))
    legacy_moments = legacy_window_moments(
        full_endpoint, signs, 15, write_sweeps, alpha
    )
    empirical_correct = full_moments - (
        alpha * signs.sum(axis=0)
        / (write_sweeps * 256.0 + 512.0 * alpha)
    )
    empirical_legacy = legacy_moments - (
        alpha * signs.sum(axis=0)
        / (write_sweeps * 256.0 + 512.0 * alpha)
    )
    attenuation = float(
        np.linalg.norm(empirical_legacy) / max(np.linalg.norm(empirical_correct), 1e-15)
    )

    checkpoint = (
        frozen6a["root"]
        / "locality/qualification/checkpoints/qualification-p0000-c00.json"
    )
    legacy_collapse = False
    legacy_exit_centroid = None
    if checkpoint.exists():
        row = _load_json(checkpoint)["result"]
        try:
            legacy_exit_centroid = float(
                row["conditions"]["intact"]["carrier_history"]["1"]["exit"][
                    "centroid_l2"
                ]
            )
            legacy_collapse = legacy_exit_centroid == 0.0
        except KeyError:
            pass

    tied = heldout_lineage_diagnostics(
        np.zeros((8, 4), dtype=np.float64), 4, _hash_seed(contract.namespace, "ties")
    )
    co1 = repair_origins("audit", 1, 8, 16, "co-located")
    co2 = repair_origins("audit", 2, 8, 16, "co-located")
    adjacent1 = repair_origins("audit", 1, 8, 16, "adjacent")
    adjacent2 = repair_origins("audit", 2, 8, 16, "adjacent")
    co_distance = toroidal_chebyshev_distance(co1, co2, 16)
    adjacent_distance = toroidal_chebyshev_distance(adjacent1, adjacent2, 16)
    reserve_untouched = bool(
        not cohorts["audit"]
        and frozen["cohorts"].get("later_audit_trajectory_state") == "untouched"
        and frozen6a["cohorts"].get("final_audit_trajectory_state") == "untouched"
    )
    maximum_error = max(span_errors.values())
    passed = bool(
        maximum_error <= 1e-12
        and payload_exact
        and legacy_collapse
        and abs(attenuation - 1.0 / 256.0) <= 1e-10
        and tied == {"balanced_accuracy": 0.5, "tie_fraction": 1.0}
        and np.all(co_distance == 0)
        and np.all(adjacent_distance == 1)
        and reserve_untouched
    )
    return {
        "state": "complete",
        "gate": passed,
        "writer_span_max_abs_errors": span_errors,
        "writer_span_max_abs_error": maximum_error,
        "full_span_quantized_payload_exact_match": payload_exact,
        "legacy_empirical_attenuation": attenuation,
        "expected_legacy_attenuation": 1.0 / 256.0,
        "legacy_checkpoint_exit_centroid_l2": legacy_exit_centroid,
        "legacy_checkpoint_collapse_reproduced": legacy_collapse,
        "decoder_all_ties": tied,
        "co_located_displacement_exact_zero": bool(np.all(co_distance == 0)),
        "adjacent_displacement_exact_one": bool(np.all(adjacent_distance == 1)),
        "semantic_stream_candidate_independent": True,
        "reserve_untouched": reserve_untouched,
        "frozen_stage6a_hashes": frozen6a["hashes"],
    }


def _base_candidate_summary(
    rows: Sequence[dict[str, Any]],
    candidate: dict[str, Any],
    generation: int,
    profile: MinimalityRepairProfile,
    contract: MinimalityRepairContract,
    phase: str,
) -> dict[str, Any]:
    candidate_id = str(candidate["candidate_id"])
    values = _condition_values(rows, candidate_id, "intact", generation)
    direction_a = _condition_values(
        rows, candidate_id, "intact", generation, "direction_a"
    )
    direction_b = _condition_values(
        rows, candidate_id, "intact", generation, "direction_b"
    )
    survival = _condition_values(rows, candidate_id, "intact", generation, "survival")
    result: dict[str, Any] = {
        "candidate": _candidate_public(candidate),
        "crossover": _boot(values, profile, contract, phase, candidate_id, generation),
        "direction_a_mean": float(np.mean(direction_a)) if direction_a else 0.0,
        "direction_b_mean": float(np.mean(direction_b)) if direction_b else 0.0,
        "fraction_pairs_positive": float(np.mean(np.asarray(values) > 0.0))
        if values
        else 0.0,
        "survival_mean": float(np.mean(survival)) if survival else 0.0,
    }
    for section, key, output_key in (
        ("decoders", "carrier_balanced_accuracy", "carrier_decoder_mean"),
        ("decoders", "carrier_tie_fraction", "carrier_decoder_tie_mean"),
        ("decoders", "phenotype_balanced_accuracy", "phenotype_decoder_mean"),
        ("carrier_history", "transition", "unused"),
        ("origin_history", "causal_overlap_fraction", "causal_overlap_mean"),
        ("origin_history", "mean_displacement", "origin_displacement_mean"),
        ("writer_history", "moment_mean_abs", "writer_moment_mean_abs"),
        ("writer_history", "latent_centroid_l2", "writer_latent_centroid_mean"),
    ):
        if output_key == "unused":
            continue
        diagnostics = _diagnostic_values(
            rows, candidate_id, generation, section, key
        )
        result[output_key] = float(np.mean(diagnostics)) if diagnostics else None
    transitions: list[dict[str, Any]] = []
    for row in rows:
        if row.get("candidate_id") != candidate_id:
            continue
        try:
            transitions.append(
                row["conditions"]["intact"]["carrier_history"][str(generation)][
                    "transition"
                ]
            )
        except KeyError:
            continue
    for key in (
        "entry_centroid_l2",
        "exit_centroid_l2",
        "centroid_retention",
        "parent_child_delta_cosine",
    ):
        result[f"carrier_{key}_mean"] = (
            float(np.mean([float(row[key]) for row in transitions]))
            if transitions
            else None
        )
    return result


def summarize_bridge(
    rows: Sequence[dict[str, Any]],
    anchor_rows: Sequence[dict[str, Any]],
    candidates: Sequence[dict[str, Any]],
    diagnostics: Sequence[dict[str, Any]],
    profile: MinimalityRepairProfile,
    contract: MinimalityRepairContract,
    *,
    scientific: bool,
) -> dict[str, Any]:
    all_candidates = [*candidates, *diagnostics]
    summaries = {
        str(candidate["candidate_id"]): _base_candidate_summary(
            rows,
            candidate,
            profile.bridge_generations,
            profile,
            contract,
            "bridge",
        )
        for candidate in all_candidates
    }
    anchor_summaries: dict[str, Any] = {}
    for anchor_id in (STAGE5R_COMPACT_ANCHOR_ID, STAGE5R_EXACT_ANCHOR_ID):
        values = _condition_values(
            anchor_rows, anchor_id, "intact", profile.bridge_generations
        )
        anchor_summaries[anchor_id] = {
            "crossover": _boot(
                values, profile, contract, "bridge-anchor", anchor_id
            ),
            "historical_mechanism_only": True,
        }
    anchor_mean = float(
        anchor_summaries[STAGE5R_COMPACT_ANCHOR_ID]["crossover"]["mean"] or 0.0
    )
    full = summaries[FULL_BRIDGE_ID]
    full_mean = float(full["crossover"]["mean"] or 0.0)
    full_lower = full["crossover"]["ci"][0]
    retention = full_mean / anchor_mean if anchor_mean > 0.0 else 0.0
    full_gate = bool(
        not scientific
        or (
            full_lower is not None
            and float(full_lower) > 0.0
            and retention >= contract.screen_anchor_retention
            and full["survival_mean"] >= contract.survival_gate
        )
    )
    promotable = [row for row in candidates if row.get("promotable")]
    ranked = sorted(
        promotable,
        key=lambda row: (
            -float(summaries[row["candidate_id"]]["crossover"]["mean"] or -1.0),
            int(row["germination_hops"]) + int(row["consolidation_steps"]),
            str(row["candidate_id"]),
        ),
    )
    selected: list[str] = []
    for policy in ("co-located", "adjacent"):
        match = next(
            (row for row in ranked if row["origin_policy"] == policy), None
        )
        if match is not None:
            selected.append(str(match["candidate_id"]))
    for candidate in ranked:
        candidate_id = str(candidate["candidate_id"])
        if candidate_id not in selected:
            selected.append(candidate_id)
        if len(selected) == 6:
            break
    return {
        "state": "complete",
        "scientific_gate_applied": scientific,
        "stage_gate": full_gate,
        "corrected_full_bridge_gate": full_gate,
        "corrected_full_bridge_id": FULL_BRIDGE_ID,
        "corrected_full_bridge_anchor_retention": retention,
        "historical_anchors": anchor_summaries,
        "candidate_summaries": summaries,
        "selected_candidate_ids": selected[:6],
        "legacy_diagnostic_id": LEGACY_DIAGNOSTIC_ID,
        "independent_origin_candidates_non_promotable": True,
    }


def summarize_screen(
    rows: Sequence[dict[str, Any]],
    candidates: Sequence[dict[str, Any]],
    profile: MinimalityRepairProfile,
    contract: MinimalityRepairContract,
    *,
    scientific: bool,
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        summary = _base_candidate_summary(
            rows,
            candidate,
            8,
            profile,
            contract,
            "screen",
        )
        active = _boot(
            _paired_advantage(rows, candidate_id, "intact", "no_rewrite", 8),
            profile,
            contract,
            "screen",
            candidate_id,
            "active-rewrite",
        )
        translated = _boot(
            _condition_values(rows, candidate_id, "translated_patch", 8),
            profile,
            contract,
            "screen",
            candidate_id,
            "translation",
        )
        clamped = _boot(
            _condition_values(rows, candidate_id, "founder_clamped", 8),
            profile,
            contract,
            "screen",
            candidate_id,
            "founder-clamped",
        )
        summary.update(
            {
                "active_rewrite_advantage": active,
                "translated": translated,
                "founder_clamped": clamped,
            }
        )
        summaries[candidate_id] = summary
    anchor_mean = float(
        summaries[FULL_BRIDGE_ID]["crossover"]["mean"] or 0.0
    )
    eligible: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        summary = summaries[candidate_id]
        intact_mean = float(summary["crossover"]["mean"] or 0.0)
        retention = intact_mean / anchor_mean if anchor_mean > 0.0 else 0.0
        translated_mean = float(summary["translated"]["mean"] or 0.0)
        translation_retention = (
            translated_mean / intact_mean if intact_mean > 0.0 else 0.0
        )
        lower = summary["crossover"]["ci"][0]
        active_lower = summary["active_rewrite_advantage"]["ci"][0]
        translated_lower = summary["translated"]["ci"][0]
        screen_positive = bool(
            candidate_id != FULL_BRIDGE_ID
            and candidate.get("promotable")
            and (
                not scientific
                or (
                    intact_mean >= contract.screen_generation8
                    and lower is not None
                    and float(lower) > 0.0
                    and summary["survival_mean"] >= contract.survival_gate
                    and summary["direction_a_mean"] > 0.0
                    and summary["direction_b_mean"] > 0.0
                    and summary["fraction_pairs_positive"] >= 0.50
                    and retention >= contract.screen_anchor_retention
                    and active_lower is not None
                    and float(active_lower) > 0.0
                    and translated_lower is not None
                    and float(translated_lower) > 0.0
                    and translation_retention >= contract.translation_retention
                )
            )
        )
        summary["corrected_anchor_retention"] = retention
        summary["translation_retention"] = translation_retention
        summary["screen_positive"] = screen_positive
        if screen_positive:
            eligible.append(candidate)
    ranked = sorted(
        eligible,
        key=lambda row: (
            -float(summaries[row["candidate_id"]]["crossover"]["mean"] or -1.0),
            int(row["germination_hops"]) + int(row["consolidation_steps"]),
            str(row["candidate_id"]),
        ),
    )
    selected = [str(row["candidate_id"]) for row in ranked[:2]]
    return {
        "state": "complete",
        "scientific_gate_applied": scientific,
        "stage_gate": bool(selected),
        "anchor_id": FULL_BRIDGE_ID,
        "anchor_generation8_crossover_mean": anchor_mean,
        "candidate_summaries": summaries,
        "eligible_candidate_ids": [str(row["candidate_id"]) for row in eligible],
        "selected_candidate_ids": selected,
    }


def summarize_qualification(
    rows: Sequence[dict[str, Any]],
    candidates: Sequence[dict[str, Any]],
    profile: MinimalityRepairProfile,
    contract: MinimalityRepairContract,
    *,
    scientific: bool,
) -> dict[str, Any]:
    anchor_values = _condition_values(rows, FULL_BRIDGE_ID, "intact", 16)
    anchor = _boot(
        anchor_values, profile, contract, "qualification", FULL_BRIDGE_ID, 16
    )
    anchor_mean = float(anchor["mean"] or 0.0)
    summaries: dict[str, Any] = {}
    qualified: list[str] = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        if candidate_id == FULL_BRIDGE_ID:
            continue
        candidate_rows = [row for row in rows if row.get("candidate_id") == candidate_id]
        transformed = [
            {
                "pair_id": row["pair_id"],
                "candidates": {candidate_id: {"conditions": row["conditions"]}},
            }
            for row in candidate_rows
        ]
        strict = (
            _strict_confirmation_gate(
                transformed,
                candidate_id,
                _repair_profile_for(
                    profile,  # type: ignore[arg-type]
                    "stage6ar-qualification",
                    len(transformed),
                    profile.qualification_replicates,
                    profile.qualification_generations,
                ),
                contract,  # type: ignore[arg-type]
                contract.strict_alpha,
            )
            if scientific
            else {"verdict": "NOT_ADJUDICATED_SMOKE", "renewed_gate": True}
        )
        intact16 = _boot(
            _condition_values(rows, candidate_id, "intact", 16),
            profile,
            contract,
            "qualification",
            candidate_id,
            16,
        )
        mean16 = float(intact16["mean"] or 0.0)
        anchor_retention = mean16 / anchor_mean if anchor_mean > 0.0 else 0.0
        targeted = {
            control: _boot(
                _paired_advantage(rows, candidate_id, "intact", control, 8),
                profile,
                contract,
                "qualification",
                candidate_id,
                control,
            )
            for control in TARGETED_CONDITIONS
            if control != "translated_patch"
        }
        intact8 = float(
            np.mean(_condition_values(rows, candidate_id, "intact", 8))
        )
        translated = _boot(
            _condition_values(rows, candidate_id, "translated_patch", 8),
            profile,
            contract,
            "qualification",
            candidate_id,
            "translation",
        )
        translation_retention = (
            float(translated["mean"] or 0.0) / intact8 if intact8 > 0.0 else 0.0
        )
        bounded = bool(
            candidate.get("promotable")
            and int(candidate["germination_hops"]) <= contract.bounded_hops
            and int(candidate["consolidation_steps"])
            <= contract.bounded_consolidation_steps
            and candidate["origin_policy"] != "independent"
        )
        local_gate = bool(
            not scientific
            or (
                strict.get("renewed_gate")
                and bounded
                and intact16["ci"][0] is not None
                and float(intact16["ci"][0]) > 0.0
                and anchor_retention >= contract.screen_anchor_retention
                and all(
                    value["ci"][0] is not None
                    and float(value["ci"][0]) > 0.0
                    and float(value["mean"] or 0.0) >= contract.control_advantage
                    for value in targeted.values()
                )
                and translated["ci"][0] is not None
                and float(translated["ci"][0]) > 0.0
                and translation_retention >= contract.translation_retention
            )
        )
        summaries[candidate_id] = {
            "candidate": _candidate_public(candidate),
            "strict": strict,
            "generation16": intact16,
            "corrected_anchor_retention": anchor_retention,
            "targeted_control_advantages_generation8": targeted,
            "translated_generation8": translated,
            "translation_retention": translation_retention,
            "finite_light_cone_audit": bounded,
            "qualification_gate": local_gate,
        }
        if local_gate:
            qualified.append(candidate_id)
    winner = (
        max(
            qualified,
            key=lambda value: float(
                summaries[value]["generation16"]["mean"] or -1.0
            ),
        )
        if qualified
        else None
    )
    return {
        "state": "complete",
        "scientific_gate_applied": scientific,
        "stage_gate": bool(qualified),
        "anchor_id": FULL_BRIDGE_ID,
        "anchor_generation16": anchor,
        "candidate_summaries": summaries,
        "qualified_candidate_ids": qualified,
        "winner_candidate_id": winner,
    }


def summarize_endurance(
    rows: Sequence[dict[str, Any]],
    winner_id: str,
    independent_id: str,
    profile: MinimalityRepairProfile,
    contract: MinimalityRepairContract,
    *,
    scientific: bool,
) -> dict[str, Any]:
    generation32 = _boot(
        _condition_values(rows, winner_id, "intact", 32),
        profile,
        contract,
        "endurance",
        winner_id,
        32,
    )
    generation64 = _boot(
        _condition_values(rows, winner_id, "intact", 64),
        profile,
        contract,
        "endurance",
        winner_id,
        64,
    )
    no_rewrite = _boot(
        _paired_advantage(rows, winner_id, "intact", "no_rewrite", 64),
        profile,
        contract,
        "endurance",
        winner_id,
        "no-rewrite",
    )
    communication = _boot(
        _paired_advantage(rows, winner_id, "intact", "communication_cut", 64),
        profile,
        contract,
        "endurance",
        winner_id,
        "communication-cut",
    )
    independent = _boot(
        _condition_values(rows, independent_id, "intact", 64),
        profile,
        contract,
        "endurance",
        independent_id,
        64,
    )
    intact64 = _condition_values(rows, winner_id, "intact", 64)
    independent64 = _condition_values(rows, independent_id, "intact", 64)
    relocation_advantage = _boot(
        [a - b for a, b in zip(intact64, independent64)],
        profile,
        contract,
        "endurance",
        winner_id,
        "relocation",
    )
    passed = bool(
        not scientific
        or (
            generation32["ci"][0] is not None
            and float(generation32["ci"][0]) > 0.0
            and generation64["ci"][0] is not None
            and float(generation64["ci"][0]) > 0.0
            and no_rewrite["ci"][0] is not None
            and float(no_rewrite["ci"][0]) > 0.0
        )
    )
    return {
        "state": "complete",
        "scientific_gate_applied": scientific,
        "stage_gate": passed,
        "winner_candidate_id": winner_id,
        "independent_relocation_candidate_id": independent_id,
        "generation32": generation32,
        "generation64": generation64,
        "active_rewrite_advantage_generation64": no_rewrite,
        "communication_cut_advantage_generation64": communication,
        "independent_relocation_generation64": independent,
        "continuous_origin_advantage_generation64": relocation_advantage,
    }


def _prepare_stage6ar(
    output: Path,
    profile_name: str,
    *,
    stage6_root: Path,
    stage5r_root: Path,
    stage5_root: Path,
    stage4_root: Path,
    stage3r_root: Path,
    stage3_root: Path,
    stage2_root: Path,
    stage1_root: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, list[dict[str, Any]]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    MinimalityRepairProfile,
    MinimalityRepairContract,
    MotifContract,
    str,
]:
    contract = MinimalityRepairContract()
    writer_contract = MotifContract()
    profile = REPAIR_PROFILES[profile_name]
    frozen = load_frozen_stage5r(
        stage5r_root,
        stage5_root,
        stage4_root,
        stage3r_root,
        stage3_root,
        stage2_root,
        stage1_root,
    )
    frozen6a = load_frozen_stage6a(stage6_root)
    base = select_minimality_cohorts(
        MINIMALITY_PROFILES[profile_name],
        frozen,
        profile_name=profile_name,
        open_audit=False,
    )
    cohorts = {
        "bridge": base["scale"][: profile.bridge_pairs],
        "screen": base["locality_screen"][: profile.screen_pairs],
        "qualify": base["locality_qualification"][: profile.qualification_pairs],
        "endurance": base["locality_endurance"][: profile.endurance_pairs],
        "audit": [],
    }
    used = [
        str(pair["pair_id"])
        for name, rows in cohorts.items()
        if name != "audit"
        for pair in rows
    ]
    if len(used) != len(set(used)):
        raise AssertionError("Stage-6A-R development cohorts overlap")
    if set(used) & set(frozen["later_ids"]):
        raise AssertionError("Stage-6A-R development touched the final reserve")
    candidates, diagnostics = build_repair_candidates(frozen)
    audit_digest = hashlib.sha256("\n".join(frozen["later_ids"]).encode()).hexdigest()
    design_payload = {
        "experiment": "ca_motif_lineage_stage_6ar",
        "contract": contract.to_dict(),
        "writer_contract_digest": writer_contract.digest,
        "profile_name": profile_name,
        "profile": asdict(profile),
        "phases": PHASES,
        "phases_separate_invocations": True,
        "automatic_successor_launch": False,
        "stage5r_design_digest": frozen["design_digest"],
        "stage6a_design_digest": frozen6a["design"]["design_digest"],
        "frozen_stage6a_sha256": frozen6a["hashes"],
        "configuration": frozen["stage4"]["configuration"].to_dict(),
        "bridge_factorial": {
            "hops": HOPS,
            "spans": SPANS,
            "origin_policies": ORIGIN_POLICIES,
            "candidate_ids": [str(row["candidate_id"]) for row in candidates],
            "diagnostic_candidate_ids": [
                str(row["candidate_id"]) for row in diagnostics
            ],
            "full_bridge_id": FULL_BRIDGE_ID,
            "legacy_diagnostic_id": LEGACY_DIAGNOSTIC_ID,
        },
        "development_pair_ids": {
            name: [str(pair["pair_id"]) for pair in rows]
            for name, rows in cohorts.items()
            if name != "audit"
        },
        "final_audit_pair_count": len(frozen["later_ids"]),
        "final_audit_pair_ids_sha256": audit_digest,
        "final_audit_trajectories_loaded": False,
        "input_sha256": {
            "protocol": _sha256(PROTOCOL_PATH),
            **{
                f"stage5r_{key}": _sha256(path)
                for key, path in frozen["paths"].items()
            },
        },
        "implementation_sha256": {
            "motif_minimality_repair.py": _sha256(Path(__file__)),
            "motif_minimality.py": _sha256(
                Path(__file__).with_name("motif_minimality.py")
            ),
            "motif_regeneration.py": _sha256(
                Path(__file__).with_name("motif_regeneration.py")
            ),
        },
        "cleanroom_exclusion": (
            "no Wagner or Fable implementation source is read, imported, hashed, "
            "or executed"
        ),
        "developmental_repair_not_independent_confirmation": True,
    }
    design_digest = hashlib.sha256(
        json.dumps(design_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    design = {**design_payload, "design_digest": design_digest}
    design_path = output / "DESIGN.json"
    if design_path.exists():
        existing = _load_json(design_path)
        if existing.get("design_digest") != design_digest:
            raise ValueError("Stage-6A-R design digest changed; refusing mixed resume")
    else:
        _atomic_json(design_path, design)
    _atomic_json(
        output / "COHORTS.json",
        {
            "design_digest": design_digest,
            "development_pair_ids": design["development_pair_ids"],
            "final_audit_pair_count": len(frozen["later_ids"]),
            "final_audit_pair_ids_sha256": audit_digest,
            "final_audit_trajectory_state": "untouched",
            "final_audit_trajectories_not_loaded": True,
        },
    )
    _atomic_json(
        output / "CANDIDATES.json",
        {
            "design_digest": design_digest,
            "candidate_count": len(candidates),
            "diagnostic_count": len(diagnostics),
            "candidates": [_candidate_public(row) for row in candidates],
            "diagnostics": [_candidate_public(row) for row in diagnostics],
        },
    )
    return (
        frozen,
        frozen6a,
        cohorts,
        candidates,
        diagnostics,
        profile,
        contract,
        writer_contract,
        design_digest,
    )


def _status_writer(
    output: Path,
    phase: str,
    profile_name: str,
    started: float,
    hard_deadline: float,
    science_deadline: float,
):
    def status(state: str, current: str, **extra: Any) -> None:
        now = time.time()
        payload = {
            "state": state,
            "stage": "6ar-corrected-local-bridge",
            "phase": phase,
            "current": current,
            "profile": profile_name,
            "pid": os.getpid(),
            "started_unix": started,
            "updated_unix": now,
            "elapsed_seconds": now - started,
            "hard_deadline_unix": hard_deadline,
            "science_deadline_unix": science_deadline,
            "deadline_remaining_seconds": max(0.0, hard_deadline - now),
            **extra,
        }
        _atomic_json(output / "STATUS.json", payload)
        _atomic_json(output / phase / "STATUS.json", payload)
        progress = (
            f" {extra['completed']}/{extra['total']}" if "completed" in extra else ""
        )
        print(f"[{state}] stage6ar-{phase}:{current}{progress}", flush=True)

    return status


def _prior_gate(output: Path, phase: str, design_digest: str) -> dict[str, Any]:
    path = output / phase / "STAGE_DECISION.json"
    if not path.exists():
        raise FileNotFoundError(f"reviewed prior-phase decision required: {path}")
    decision = _load_json(path)
    if decision.get("design_digest") != design_digest or decision.get("state") != "complete":
        raise ValueError(f"Stage-6A-R {phase} is not complete under this design")
    if not decision.get("stage_gate", False):
        raise ValueError(f"Stage-6A-R {phase} gate failed; successor is blocked")
    return decision


def _write_phase_outputs(
    output: Path,
    phase: str,
    design_digest: str,
    results: dict[str, Any],
    report: str,
    lay: str,
) -> None:
    phase_root = output / phase
    phase_root.mkdir(parents=True, exist_ok=True)
    stage_gate = bool(results.get("stage_gate", results.get("gate", False)))
    next_phase = (
        PHASES[PHASES.index(phase) + 1]
        if results.get("state") == "complete"
        and stage_gate
        and phase != PHASES[-1]
        else None
    )
    decision = {
        "experiment": "ca_motif_lineage_stage_6ar",
        "state": results.get("state"),
        "phase": phase,
        "design_digest": design_digest,
        "stage_gate": stage_gate,
        "decision": (
            f"{next_phase}_may_run_after_review"
            if next_phase
            else ("repair_complete" if stage_gate else "phase_gate_failed")
        ),
        "next_phase": next_phase,
        "automatic_launch": False,
        "review_required": bool(next_phase),
        "final_audit_trajectory_state": "untouched",
    }
    results = {**results, "design_digest": design_digest, "phase": phase}
    _atomic_json(phase_root / "RESULTS.json", results)
    _atomic_json(phase_root / "STAGE_DECISION.json", decision)
    _atomic_text(phase_root / "REPORT.md", report)
    _atomic_text(phase_root / "LAY_SUMMARY.md", lay)
    if results.get("state") == "complete":
        _atomic_text(phase_root / "COMPLETE", "complete\n")
    _atomic_json(output / "RESULTS.json", results)
    _atomic_json(output / "STAGE_DECISION.json", decision)
    _atomic_text(output / "REPORT.md", report)
    _atomic_text(output / "LAY_SUMMARY.md", lay)
    _atomic_json(
        output / "QUEUE.json",
        {
            "experiment": "ca_motif_lineage_stage_6ar",
            "design_digest": design_digest,
            "state": "blocked_pending_human_review" if next_phase else results.get("state"),
            "completed_phase": phase if results.get("state") == "complete" else None,
            "next_phase": next_phase,
            "automatic_launch": False,
            "review_required": bool(next_phase),
        },
    )


def _run_audit_phase(
    output: Path,
    frozen: dict[str, Any],
    frozen6a: dict[str, Any],
    cohorts: dict[str, list[dict[str, Any]]],
    writer_contract: MotifContract,
    contract: MinimalityRepairContract,
    design_digest: str,
) -> dict[str, Any]:
    result = correction_audit(
        frozen, frozen6a, writer_contract, contract, cohorts
    )
    result["stage_gate"] = bool(result["gate"])
    report = (
        "# Stage 6A-R correction audit\n\n"
        f"State: complete. Gate: **{'PASS' if result['gate'] else 'FAIL'}**.\n\n"
        f"Maximum corrected-window error: {result['writer_span_max_abs_error']}. "
        f"Full-span quantized equality: {result['full_span_quantized_payload_exact_match']}. "
        f"Observed legacy attenuation: {result['legacy_empirical_attenuation']}. "
        "The completed Stage 6A evidence remains frozen and the final reserve remains untouched.\n"
    )
    lay = (
        "# Lay summary\n\n"
        "The previous local writer averaged a neighbourhood and then divided by its size again. "
        "This audit checks the corrected arithmetic against the known working global writer, "
        f"and the correction {'passes' if result['gate'] else 'does not pass'} "
        "every registered invariant. "
        "No sealed test cases were opened.\n"
    )
    _write_phase_outputs(output, "audit", design_digest, result, report, lay)
    return result


def _run_bridge_phase(
    output: Path,
    frozen: dict[str, Any],
    cohorts: dict[str, list[dict[str, Any]]],
    candidates: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
    profile: MinimalityRepairProfile,
    contract: MinimalityRepairContract,
    writer_contract: MotifContract,
    design_digest: str,
    *,
    workers: int,
    resume: bool,
    deadline: float,
    status,
    scientific: bool,
) -> dict[str, Any]:
    configuration = _configuration_payload(frozen["stage4"]["configuration"])
    all_candidates = [*candidates, *diagnostics]
    items = [
        {
            "checkpoint": f"bridge-p{pair_index:04d}-c{candidate_index:03d}",
            "pair": pair,
            "candidate_id": candidate["candidate_id"],
            "configuration": configuration,
            "replicates": profile.bridge_replicates,
            "generations": profile.bridge_generations,
            "conditions": ("intact",),
        }
        for pair_index, pair in enumerate(cohorts["bridge"])
        for candidate_index, candidate in enumerate(all_candidates)
    ]
    status("running", "bridge-matrix", completed=0, total=len(items))
    rows, complete = _run_json_checkpoints(
        output,
        "bridge-matrix",
        items,
        all_candidates,
        _repair_task,
        writer_contract,
        contract,  # type: ignore[arg-type]
        frozen["stage4"]["reference"],
        design_digest,
        workers=workers,
        resume=resume,
        deadline=deadline,
        status=status,
    )
    if not complete:
        return {"state": "partial_budget_exhausted", "stage_gate": False}
    anchors = [
        {
            **frozen["by_candidate"][COMPACT_ID],
            "candidate_id": STAGE5R_COMPACT_ANCHOR_ID,
            "original_candidate_id": COMPACT_ID,
        },
        {
            **frozen["by_candidate"][EXACT_ID],
            "candidate_id": STAGE5R_EXACT_ANCHOR_ID,
            "original_candidate_id": EXACT_ID,
        },
    ]
    anchor_items = [
        {
            "checkpoint": f"anchor-p{pair_index:04d}-c{candidate_index:02d}",
            "pair": pair,
            "candidate_id": candidate["candidate_id"],
            "configuration": configuration,
            "replicates": profile.bridge_replicates,
            "generations": profile.bridge_generations,
            "walsh_model": frozen["stage4"]["winner_model"],
        }
        for pair_index, pair in enumerate(cohorts["bridge"])
        for candidate_index, candidate in enumerate(anchors)
    ]
    status("running", "bridge-anchors", completed=0, total=len(anchor_items))
    anchor_rows, complete = _run_json_checkpoints(
        output,
        "bridge-anchors",
        anchor_items,
        anchors,
        _stage5r_anchor_task,
        writer_contract,
        contract,  # type: ignore[arg-type]
        frozen["stage4"]["reference"],
        design_digest,
        workers=workers,
        resume=resume,
        deadline=deadline,
        status=status,
    )
    if not complete:
        return {"state": "partial_budget_exhausted", "stage_gate": False}
    result = summarize_bridge(
        rows,
        anchor_rows,
        candidates,
        diagnostics,
        profile,
        contract,
        scientific=scientific,
    )
    report = (
        "# Stage 6A-R mechanistic bridge\n\n"
        "State: complete. Corrected full-bridge gate: "
        f"**{'PASS' if result['stage_gate'] else 'FAIL'}**.\n\n"
        "Full-bridge compact-anchor retention: "
        f"{result['corrected_full_bridge_anchor_retention']}. "
        f"Selected bounded candidates: {', '.join(result['selected_candidate_ids']) or 'none'}.\n"
    )
    lay = (
        "# Lay summary\n\n"
        "This phase changes the distance that memory may travel, the size of the neighbourhood "
        "used to rewrite it, and whether parent and daughter remain spatial neighbours. "
        f"The corrected global bridge {'worked' if result['stage_gate'] else 'did not work'}, "
        "which determines whether local candidates may be screened.\n"
    )
    _write_phase_outputs(output, "bridge", design_digest, result, report, lay)
    return result


def _run_screen_phase(
    output: Path,
    frozen: dict[str, Any],
    cohorts: dict[str, list[dict[str, Any]]],
    candidates: list[dict[str, Any]],
    profile: MinimalityRepairProfile,
    contract: MinimalityRepairContract,
    writer_contract: MotifContract,
    design_digest: str,
    *,
    workers: int,
    resume: bool,
    deadline: float,
    status,
    scientific: bool,
) -> dict[str, Any]:
    bridge = _load_json(output / "bridge/RESULTS.json")
    selected_ids = list(bridge["selected_candidate_ids"])
    by_id = {str(row["candidate_id"]): row for row in candidates}
    selected = [by_id[FULL_BRIDGE_ID], *[by_id[value] for value in selected_ids]]
    configuration = _configuration_payload(frozen["stage4"]["configuration"])
    conditions = (
        "intact",
        "founder_clamped",
        "no_rewrite",
        "write_disabled",
        "transport_disabled",
        "communication_cut",
        "translated_patch",
    )
    items = [
        {
            "checkpoint": f"screen-p{pair_index:04d}-c{candidate_index:02d}",
            "pair": pair,
            "candidate_id": candidate["candidate_id"],
            "configuration": configuration,
            "replicates": profile.screen_replicates,
            "generations": profile.screen_generations,
            "conditions": conditions,
        }
        for pair_index, pair in enumerate(cohorts["screen"])
        for candidate_index, candidate in enumerate(selected)
    ]
    status("running", "screen", completed=0, total=len(items))
    rows, complete = _run_json_checkpoints(
        output,
        "screen-checkpoints",
        items,
        selected,
        _repair_task,
        writer_contract,
        contract,  # type: ignore[arg-type]
        frozen["stage4"]["reference"],
        design_digest,
        workers=workers,
        resume=resume,
        deadline=deadline,
        status=status,
    )
    if not complete:
        return {"state": "partial_budget_exhausted", "stage_gate": False}
    result = summarize_screen(
        rows, selected, profile, contract, scientific=scientific
    )
    report = (
        "# Stage 6A-R local repair screen\n\n"
        f"State: complete. Gate: **{'PASS' if result['stage_gate'] else 'FAIL'}**.\n\n"
        f"Screen-positive candidates: {', '.join(result['eligible_candidate_ids']) or 'none'}. "
        f"Qualification nominees: {', '.join(result['selected_candidate_ids']) or 'none'}.\n"
    )
    lay = (
        "# Lay summary\n\n"
        "The corrected memory mechanisms were tested while the visible daughter was reset. "
        f"{len(result['eligible_candidate_ids'])} local designs kept enough family-specific "
        "influence, active rewriting, and translation robustness to pass this screen.\n"
    )
    _write_phase_outputs(output, "screen", design_digest, result, report, lay)
    return result


def _run_qualification_phase(
    output: Path,
    frozen: dict[str, Any],
    cohorts: dict[str, list[dict[str, Any]]],
    candidates: list[dict[str, Any]],
    profile: MinimalityRepairProfile,
    contract: MinimalityRepairContract,
    writer_contract: MotifContract,
    design_digest: str,
    *,
    workers: int,
    resume: bool,
    deadline: float,
    status,
    scientific: bool,
) -> dict[str, Any]:
    screen = _load_json(output / "screen/RESULTS.json")
    selected_ids = list(screen["selected_candidate_ids"])
    by_id = {str(row["candidate_id"]): row for row in candidates}
    selected = [by_id[FULL_BRIDGE_ID], *[by_id[value] for value in selected_ids]]
    configuration = _configuration_payload(frozen["stage4"]["configuration"])
    conditions = tuple(dict.fromkeys((*QUALIFICATION_CONDITIONS, "founder_clamped")))
    items: list[dict[str, Any]] = []
    for pair_index, pair in enumerate(cohorts["qualify"]):
        for candidate_index, candidate in enumerate(selected):
            items.append(
                {
                    "checkpoint": f"qualify-p{pair_index:04d}-c{candidate_index:02d}",
                    "pair": pair,
                    "candidate_id": candidate["candidate_id"],
                    "configuration": configuration,
                    "replicates": profile.qualification_replicates,
                    "generations": profile.qualification_generations,
                    "conditions": (
                        ("intact",)
                        if candidate["candidate_id"] == FULL_BRIDGE_ID
                        else conditions
                    ),
                }
            )
    status("running", "qualification", completed=0, total=len(items))
    rows, complete = _run_json_checkpoints(
        output,
        "qualification-checkpoints",
        items,
        selected,
        _repair_task,
        writer_contract,
        contract,  # type: ignore[arg-type]
        frozen["stage4"]["reference"],
        design_digest,
        workers=workers,
        resume=resume,
        deadline=deadline,
        status=status,
    )
    if not complete:
        return {"state": "partial_budget_exhausted", "stage_gate": False}
    result = summarize_qualification(
        rows, selected, profile, contract, scientific=scientific
    )
    report = (
        "# Stage 6A-R strict qualification\n\n"
        f"State: complete. Gate: **{'PASS' if result['stage_gate'] else 'FAIL'}**.\n\n"
        f"Qualified candidates: {', '.join(result['qualified_candidate_ids']) or 'none'}. "
        f"Winner: {result['winner_candidate_id'] or 'none'}. "
        "These are developmental repair results because Stage 6A previously exposed the cohort.\n"
    )
    lay = (
        "# Lay summary\n\n"
        "This phase asks whether a local memory truly causes family-form recovery: removing it "
        "must erase the effect, restoring the correct memory must rescue it, and restoring the "
        "opposite memory must reverse it. "
        f"The strict test {'passed' if result['stage_gate'] else 'did not pass'}.\n"
    )
    _write_phase_outputs(output, "qualify", design_digest, result, report, lay)
    return result


def _run_endurance_phase(
    output: Path,
    frozen: dict[str, Any],
    cohorts: dict[str, list[dict[str, Any]]],
    candidates: list[dict[str, Any]],
    profile: MinimalityRepairProfile,
    contract: MinimalityRepairContract,
    writer_contract: MotifContract,
    design_digest: str,
    *,
    workers: int,
    resume: bool,
    deadline: float,
    status,
    scientific: bool,
) -> dict[str, Any]:
    qualification = _load_json(output / "qualify/RESULTS.json")
    winner_id = str(qualification["winner_candidate_id"])
    by_id = {str(row["candidate_id"]): row for row in candidates}
    winner = by_id[winner_id]
    independent = next(
        row
        for row in candidates
        if row["germination_hops"] == winner["germination_hops"]
        and row["consolidation_span"] == winner["consolidation_span"]
        and row["origin_policy"] == "independent"
        and row["reader_mode"] == winner["reader_mode"]
    )
    selected = [winner, independent]
    configuration = _configuration_payload(frozen["stage4"]["configuration"])
    items: list[dict[str, Any]] = []
    for pair_index, pair in enumerate(cohorts["endurance"]):
        items.extend(
            (
                {
                    "checkpoint": f"endurance-p{pair_index:04d}-winner",
                    "pair": pair,
                    "candidate_id": winner_id,
                    "configuration": configuration,
                    "replicates": profile.endurance_replicates,
                    "generations": profile.endurance_generations,
                    "conditions": (
                        "intact",
                        "no_rewrite",
                        "carrier_corruption_1",
                        "communication_cut",
                    ),
                },
                {
                    "checkpoint": f"endurance-p{pair_index:04d}-independent",
                    "pair": pair,
                    "candidate_id": independent["candidate_id"],
                    "configuration": configuration,
                    "replicates": profile.endurance_replicates,
                    "generations": profile.endurance_generations,
                    "conditions": ("intact",),
                },
            )
        )
    status("running", "endurance", completed=0, total=len(items))
    rows, complete = _run_json_checkpoints(
        output,
        "endurance-checkpoints",
        items,
        selected,
        _repair_task,
        writer_contract,
        contract,  # type: ignore[arg-type]
        frozen["stage4"]["reference"],
        design_digest,
        workers=workers,
        resume=resume,
        deadline=deadline,
        status=status,
    )
    if not complete:
        return {"state": "partial_budget_exhausted", "stage_gate": False}
    result = summarize_endurance(
        rows,
        winner_id,
        str(independent["candidate_id"]),
        profile,
        contract,
        scientific=scientific,
    )
    report = (
        "# Stage 6A-R endurance\n\n"
        f"State: complete. Gate: **{'PASS' if result['stage_gate'] else 'FAIL'}**.\n\n"
        f"Generation-64 crossover mean: {result['generation64']['mean']}. "
        f"Active-rewrite advantage: {result['active_rewrite_advantage_generation64']['mean']}.\n"
    )
    endurance_phrase = (
        "continued to rebuild family form"
        if result["stage_gate"]
        else "did not retain a strictly renewed family signal"
    )
    lay = (
        "# Lay summary\n\n"
        "The best repaired local memory was followed for 64 parent-to-daughter cycles. "
        f"It {endurance_phrase}. "
        "The sealed final audit was not opened.\n"
    )
    _write_phase_outputs(output, "endurance", design_digest, result, report, lay)
    return result


def run_motif_minimality_repair(
    output: Path,
    *,
    phase: str = "audit",
    stage6_root: Path = DEFAULT_STAGE6_ROOT,
    stage5r_root: Path = DEFAULT_STAGE5R_ROOT,
    stage5_root: Path = DEFAULT_STAGE5_ROOT,
    stage4_root: Path = DEFAULT_STAGE4_ROOT,
    stage3r_root: Path = DEFAULT_STAGE3R_ROOT,
    stage3_root: Path = DEFAULT_STAGE3_ROOT,
    stage2_root: Path = DEFAULT_STAGE2_ROOT,
    stage1_root: Path = DEFAULT_STAGE1_ROOT,
    profile_name: str = "reference",
    workers: int = 4,
    max_hours: float = 4.0,
    resume: bool = False,
    authorize_confirmation: bool = False,
) -> dict[str, Any]:
    """Run one manually gated Stage-6A-R phase."""

    require_pinned_numpy()
    if phase not in PHASES:
        raise ValueError(f"unknown Stage-6A-R phase {phase!r}")
    if profile_name not in PUBLIC_PROFILES:
        raise ValueError(f"unknown Stage-6A-R profile {profile_name!r}")
    if workers < 1 or workers > 4:
        raise ValueError("Stage-6A-R workers must be in [1, 4]")
    if max_hours <= 0.0 or max_hours > 4.0:
        raise ValueError("Stage-6A-R max-hours must be in (0, 4]")
    if phase != "audit" and not resume:
        raise ValueError("Stage-6A-R successor phases require --resume after review")
    if phase == "qualify" and not authorize_confirmation:
        raise ValueError("Stage-6A-R qualification requires --authorize-confirmation")
    if phase != "qualify" and authorize_confirmation:
        raise ValueError("confirmation authorization is valid only for qualification")
    if not PROTOCOL_PATH.exists():
        raise FileNotFoundError(PROTOCOL_PATH)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / phase).mkdir(parents=True, exist_ok=True)
    started = time.time()
    hard_deadline = started + max_hours * 3600.0
    reserve = min(
        MinimalityRepairContract().science_reserve_seconds,
        max(60.0, max_hours * 3600.0 * 0.125),
    )
    science_deadline = max(started, hard_deadline - reserve)
    status = _status_writer(
        output, phase, profile_name, started, hard_deadline, science_deadline
    )
    try:
        status("running", "freeze-and-design")
        (
            frozen,
            frozen6a,
            cohorts,
            candidates,
            diagnostics,
            profile,
            contract,
            writer_contract,
            design_digest,
        ) = _prepare_stage6ar(
            output,
            profile_name,
            stage6_root=stage6_root,
            stage5r_root=stage5r_root,
            stage5_root=stage5_root,
            stage4_root=stage4_root,
            stage3r_root=stage3r_root,
            stage3_root=stage3_root,
            stage2_root=stage2_root,
            stage1_root=stage1_root,
        )
        predecessors = {
            "audit": (),
            "bridge": ("audit",),
            "screen": ("audit", "bridge"),
            "qualify": ("audit", "bridge", "screen"),
            "endurance": ("audit", "bridge", "screen", "qualify"),
        }[phase]
        for predecessor in predecessors:
            _prior_gate(output, predecessor, design_digest)
        _atomic_json(
            output / phase / "MANIFEST.json",
            {
                "experiment": "ca_motif_lineage_stage_6ar",
                "phase": phase,
                "profile": profile_name,
                "design_digest": design_digest,
                "contract_digest": contract.digest,
                "workers": workers,
                "max_hours": max_hours,
                "resume": resume,
                "authorize_confirmation": authorize_confirmation,
                "started_unix": started,
                "environment": {
                    "python": sys.version,
                    "numpy": np.__version__,
                    "platform": platform.platform(),
                    "cpu_count": os.cpu_count(),
                    "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                },
            },
        )
        scientific = profile_name == "reference"
        if phase == "audit":
            result = _run_audit_phase(
                output,
                frozen,
                frozen6a,
                cohorts,
                writer_contract,
                contract,
                design_digest,
            )
        elif phase == "bridge":
            result = _run_bridge_phase(
                output,
                frozen,
                cohorts,
                candidates,
                diagnostics,
                profile,
                contract,
                writer_contract,
                design_digest,
                workers=workers,
                resume=resume,
                deadline=science_deadline,
                status=status,
                scientific=scientific,
            )
        elif phase == "screen":
            result = _run_screen_phase(
                output,
                frozen,
                cohorts,
                candidates,
                profile,
                contract,
                writer_contract,
                design_digest,
                workers=workers,
                resume=resume,
                deadline=science_deadline,
                status=status,
                scientific=scientific,
            )
        elif phase == "qualify":
            result = _run_qualification_phase(
                output,
                frozen,
                cohorts,
                candidates,
                profile,
                contract,
                writer_contract,
                design_digest,
                workers=workers,
                resume=resume,
                deadline=science_deadline,
                status=status,
                scientific=scientific,
            )
        else:
            result = _run_endurance_phase(
                output,
                frozen,
                cohorts,
                candidates,
                profile,
                contract,
                writer_contract,
                design_digest,
                workers=workers,
                resume=resume,
                deadline=science_deadline,
                status=status,
                scientific=scientific,
            )
        if result.get("state") != "complete":
            report = (
                f"# Stage 6A-R {phase}\n\nState: {result.get('state')}. "
                "Resume is required.\n"
            )
            lay = (
                f"# Lay summary\n\nThe {phase} phase reached its time budget and "
                "saved its checkpoints.\n"
            )
            _write_phase_outputs(output, phase, design_digest, result, report, lay)
        status(
            str(result.get("state", "unknown")),
            "campaign",
            stage_gate=result.get("stage_gate", result.get("gate")),
            next_phase=(
                PHASES[PHASES.index(phase) + 1]
                if result.get("state") == "complete"
                and result.get("stage_gate", result.get("gate"))
                and phase != PHASES[-1]
                else None
            ),
        )
        return result
    except BaseException as error:
        status("failed", "campaign", error=repr(error))
        raise


__all__ = [
    "FROZEN_STAGE6A_SHA256",
    "FULL_BRIDGE_ID",
    "MinimalityRepairContract",
    "MinimalityRepairProfile",
    "ORIGIN_POLICIES",
    "PHASES",
    "PUBLIC_PROFILES",
    "REPAIR_PROFILES",
    "build_repair_candidates",
    "corrected_window_moments",
    "correction_audit",
    "heldout_lineage_diagnostics",
    "legacy_window_moments",
    "repair_origins",
    "run_motif_minimality_repair",
    "simulate_repaired_lineage",
    "toroidal_chebyshev_distance",
]
