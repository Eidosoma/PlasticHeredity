"""Stage-6B-R renewal, coverage, and scale tests for CA Plastic Heredity.

The module consumes frozen Stage-5R, Stage-6A, and Stage-6A-R artifacts.  It
does not change their decisions.  All developmental search uses exposed pairs;
the later-audit reserve is loaded only after a final design has been sealed.
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
    _cosine_labels,
    _normalize_rows,
    _outcome,
    _step,
    apply_energy_reader,
    motif3_codes,
)
from .motif_localization import apply_local_reader
from .motif_minimality import (
    DEFAULT_STAGE1_ROOT,
    DEFAULT_STAGE2_ROOT,
    DEFAULT_STAGE3_ROOT,
    DEFAULT_STAGE3R_ROOT,
    DEFAULT_STAGE4_ROOT,
    DEFAULT_STAGE5_ROOT,
    DEFAULT_STAGE5R_ROOT,
    MINIMALITY_PROFILES,
    RULE,
    _binary_information_lower_bound,
    _configuration_payload,
    _founder_bounded_payload,
    _payload_summary,
    _quantize,
    _resize_board,
    _score_state,
    bounded_reduce_endpoint,
    load_frozen_stage5r,
    propagate_bounded,
    select_minimality_cohorts,
)
from .motif_minimality_repair import (
    DEFAULT_STAGE6_ROOT,
    FULL_BRIDGE_ID,
    _boundary_intervention,
    _condition_values,
    _paired_advantage,
    _semantic_uniforms,
    _transition_summary,
    build_repair_candidates,
    heldout_lineage_diagnostics,
    load_frozen_stage6a,
    repair_origins,
    toroidal_chebyshev_distance,
)


ROOT = Path(__file__).resolve().parent.parent
PROTOCOL_PATH = ROOT / "CA_MOTIF_LINEAGE_STAGE6BR_PROTOCOL.md"
DEFAULT_STAGE6AR_ROOT = ROOT / "results/ca-motif-lineage-stage-6ar"
PHASES = ("transient", "repair", "coverage", "scale", "adjudicate")
WINDOWS = (
    ("directed-5", "directed", 4),
    ("directed-8", "directed", 7),
    ("centered-5", "centered", 2),
    ("centered-9", "centered", 4),
    ("centered-11", "centered", 5),
)
FIT_MODES = ("frozen-global", "local-one-step", "local-rollout")
TURNOVERS = (0.0, 0.25, 0.50, 0.75)
RADII = (5, 6, 7, 8)
SEED_LAYOUTS = ((1, "single"), (2, "replicated"), (2, "partitioned"),
                (4, "replicated"), (4, "partitioned"))
FULL_ANCHOR_ID = "stage6br-full-h08-c30"

TRANSIENT_IDS = (
    "repair-h05-c08-oadjacent-rfield-local",
    "repair-h05-c14-oadjacent-rfield-local",
    "repair-h05-c14-oco-located-rfield-local",
)

SCREEN_CONDITIONS = (
    "intact",
    "founder_clamped",
    "no_rewrite",
    "carrier_corruption_1",
    "write_disabled",
    "transport_disabled",
    "communication_cut",
    "translated_patch",
)
TRANSIENT_CONDITIONS = (
    "intact",
    "founder_clamped",
    "no_rewrite",
    "write_disabled",
    "transport_disabled",
    "communication_cut",
    "zero_every_boundary",
    "shuffle_every_boundary",
    "opposite_founder",
    "translated_patch",
)
FULL_CONDITIONS = (
    "intact",
    "zero_every_boundary",
    "shuffle_every_boundary",
    "read_disabled",
    "founder_write_disabled",
    "write_disabled",
    "no_rewrite",
    "ablate_after_g2",
    "rescue_same_enter_g4",
    "rescue_opposite_enter_g4",
    "opposite_founder",
    "carrier_corruption_1",
    "transport_disabled",
    "regeneration_disabled",
    "consolidation_disabled",
    "translated_patch",
    "communication_cut",
    "founder_clamped",
)


@dataclass(frozen=True)
class RenewalContract:
    implementation_version: str = "ca-motif-lineage-stage6br-cleanroom-v1"
    namespace: str = "plastic-ca-motif-lineage-stage6br-v1"
    rule: int = RULE
    generation_sweeps: int = 64
    read_sweeps: int = 32
    write_start: int = 49
    write_end: int = 64
    observe_start: int = 57
    process_noise: float = 0.002
    repair_gain: float = 0.50
    bit_corruption: float = 0.01
    strict_alpha: float = 0.025
    transient_generation4: float = 0.15
    screen_generation8: float = 0.15
    qualification_generation16: float = 0.10
    anchor_retention: float = 0.70
    translation_retention: float = 0.70
    survival_gate: float = 0.90
    loss_fraction: float = 0.70
    rescue_fraction: float = 0.70
    decoder_splits: int = 4
    max_workers: int = 4
    max_hours: float = 8.0
    exploration_stop_hours: float = 6.25
    adjudication_reserve_hours: float = 1.75

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {
            "visible_reset": "bitwise-identical matched board before every generation",
            "writer_normalization": "count-corrected window moments",
            "runtime_labels": False,
            "random_streams": "semantic, paired, order and worker independent",
            "final_reserve_policy": "automatic only after sealed passing design",
            "claim_boundary": "engineered synthetic CA heredity",
        }

    @property
    def digest(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class RenewalProfile:
    transient_pairs: int
    transient_replicates: int
    writer_train_pairs: int
    writer_train_replicates: int
    repair_calibration_pairs: int
    repair_calibration_replicates: int
    repair_validation_pairs: int
    repair_validation_replicates: int
    coverage_pilot_pairs: int
    coverage_validation_pairs: int
    coverage_replicates: int
    scale_pairs: int
    scale_replicates: int
    qualification_pairs: int
    qualification_replicates: int
    endurance_pairs: int
    endurance_replicates: int
    final_pairs: int
    final_replicates: int
    bootstrap_resamples: int
    rollout_refits: int


RENEWAL_PROFILES: dict[str, RenewalProfile] = {
    "smoke": RenewalProfile(
        2, 2, 1, 2, 1, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 100, 1
    ),
    "pilot": RenewalProfile(
        16, 4, 16, 2, 8, 2, 16, 4, 4, 16, 4, 8, 2, 16, 4, 8, 4, 8, 4,
        1_000, 2
    ),
    "reference": RenewalProfile(
        96, 16, 96, 4, 32, 4, 64, 8, 16, 64, 8, 32, 4, 96, 16, 32, 8,
        62, 24, 10_000, 3
    ),
}
PUBLIC_PROFILES = tuple(RENEWAL_PROFILES)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def centered_reduce_endpoint(
    values: np.ndarray, radius: int, origins: np.ndarray
) -> np.ndarray:
    """Return the exact centred odd-square mean at each routed endpoint."""

    source = np.asarray(values, dtype=np.float64)
    if source.ndim < 4 or source.shape[1] != source.shape[2]:
        raise ValueError("values require sample, square-y, square-x, channel axes")
    extent = source.shape[1]
    if radius < 0 or 2 * radius + 1 > extent:
        raise ValueError("centred radius does not fit the lattice")
    result = np.zeros((len(source), source.shape[-1]), dtype=np.float64)
    offsets = np.arange(-radius, radius + 1, dtype=np.int64)
    for sample, (origin_y, origin_x) in enumerate(
        np.asarray(origins, dtype=np.int64)
    ):
        ys = (origin_y + offsets) % extent
        xs = (origin_x + offsets) % extent
        result[sample] = source[sample][np.ix_(ys, xs)].mean(axis=(0, 1))
    return result


def centred_causal_overlap(
    current: np.ndarray,
    nxt: np.ndarray,
    extent: int,
    hops: int,
    radius: int,
) -> float:
    """Fraction of a centred writer square contained in the parent light cone."""

    fractions: list[float] = []
    offsets = range(-radius, radius + 1)
    for left, right in zip(
        np.asarray(current, dtype=np.int64), np.asarray(nxt, dtype=np.int64)
    ):
        reached = {
            ((int(left[0]) + dy) % extent, (int(left[1]) + dx) % extent)
            for dy in range(-hops, hops + 1)
            for dx in range(-hops, hops + 1)
        }
        window = {
            ((int(right[0]) + dy) % extent, (int(right[1]) + dx) % extent)
            for dy in offsets
            for dx in offsets
        }
        fractions.append(len(reached & window) / max(1, len(window)))
    return float(np.mean(fractions))


def latch_update(
    parent: np.ndarray,
    proposal: np.ndarray,
    rho: float,
    codec: dict[str, Any],
    *,
    write_enabled: bool = True,
) -> tuple[np.ndarray, float]:
    """Turn over a slow carrier; without writing only the retained part remains."""

    if not 0.0 <= rho <= 1.0:
        raise ValueError("rho must be in [0, 1]")
    previous = np.asarray(parent, dtype=np.float32)
    fresh = np.asarray(proposal, dtype=np.float32)
    if previous.shape != fresh.shape:
        raise ValueError("parent and proposal shapes differ")
    latent = np.float32(rho) * previous
    if write_enabled:
        latent = latent + np.float32(1.0 - rho) * fresh
    return _quantize(latent, codec)


def _payload_codes(payload: np.ndarray, codec: dict[str, Any]) -> np.ndarray:
    scale = np.asarray(codec["quantizer_scale"], dtype=np.float32)
    qmax = (1 << (int(codec["bits"]) - 1)) - 1
    normalized = np.divide(
        np.asarray(payload, dtype=np.float32),
        scale,
        out=np.zeros_like(np.asarray(payload, dtype=np.float32)),
        where=scale > 0,
    )
    signed = np.clip(np.rint(normalized * qmax), -qmax, qmax).astype(np.int16)
    return (signed + qmax).astype(np.uint8)


def _codes_payload(codes: np.ndarray, codec: dict[str, Any]) -> np.ndarray:
    scale = np.asarray(codec["quantizer_scale"], dtype=np.float32)
    qmax = (1 << (int(codec["bits"]) - 1)) - 1
    signed = np.asarray(codes, dtype=np.int16) - qmax
    signed = np.clip(signed, -qmax, qmax)
    return (signed.astype(np.float32) * (scale / np.float32(qmax))).astype(
        np.float32
    )


def hamming84_encode(nibbles: np.ndarray) -> np.ndarray:
    """Systematic SECDED Hamming(8,4), least-significant data bit first."""

    values = np.asarray(nibbles, dtype=np.uint8) & np.uint8(0x0F)
    bits = ((values[..., None] >> np.arange(4, dtype=np.uint8)) & 1).astype(
        np.uint8
    )
    code = np.zeros(values.shape + (8,), dtype=np.uint8)
    code[..., 2] = bits[..., 0]
    code[..., 4] = bits[..., 1]
    code[..., 5] = bits[..., 2]
    code[..., 6] = bits[..., 3]
    code[..., 0] = code[..., 2] ^ code[..., 4] ^ code[..., 6]
    code[..., 1] = code[..., 2] ^ code[..., 5] ^ code[..., 6]
    code[..., 3] = code[..., 4] ^ code[..., 5] ^ code[..., 6]
    code[..., 7] = np.bitwise_xor.reduce(code[..., :7], axis=-1)
    return code


def hamming84_decode(codewords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decode SECDED words and report uncorrectable double-error flags."""

    code = (np.asarray(codewords, dtype=np.uint8) & 1).copy()
    if code.shape[-1] != 8:
        raise ValueError("Hamming codewords require eight bits")
    s1 = code[..., 0] ^ code[..., 2] ^ code[..., 4] ^ code[..., 6]
    s2 = code[..., 1] ^ code[..., 2] ^ code[..., 5] ^ code[..., 6]
    s4 = code[..., 3] ^ code[..., 4] ^ code[..., 5] ^ code[..., 6]
    syndrome = s1 + 2 * s2 + 4 * s4
    parity = np.bitwise_xor.reduce(code, axis=-1)
    single = (syndrome > 0) & (parity == 1)
    parity_only = (syndrome == 0) & (parity == 1)
    flat = code.reshape(-1, 8)
    single_flat = single.reshape(-1)
    syndrome_flat = syndrome.reshape(-1)
    rows = np.flatnonzero(single_flat)
    if len(rows):
        flat[rows, syndrome_flat[rows] - 1] ^= np.uint8(1)
    parity_rows = np.flatnonzero(parity_only.reshape(-1))
    if len(parity_rows):
        flat[parity_rows, 7] ^= np.uint8(1)
    code = flat.reshape(code.shape)
    data = (
        code[..., 2]
        | (code[..., 4] << 1)
        | (code[..., 5] << 2)
        | (code[..., 6] << 3)
    )
    uncorrectable = (syndrome > 0) & (parity == 0)
    return data.astype(np.uint8), uncorrectable


def coded_payload_roundtrip(
    payload: np.ndarray,
    codec: dict[str, Any],
    ecc: str,
    flips: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Quantized raw or Hamming-protected carrier channel round trip."""

    codes = _payload_codes(payload, codec)
    if ecc == "none":
        bits = ((codes[..., None] >> np.arange(4, dtype=np.uint8)) & 1).astype(
            np.uint8
        )
        if flips is not None:
            bits ^= np.asarray(flips, dtype=np.uint8)
        decoded = np.sum(bits << np.arange(4, dtype=np.uint8), axis=-1)
        uncorrectable = np.zeros(decoded.shape, dtype=np.bool_)
    elif ecc == "hamming84":
        bits = hamming84_encode(codes)
        if flips is not None:
            bits ^= np.asarray(flips, dtype=np.uint8)
        decoded, uncorrectable = hamming84_decode(bits)
    else:
        raise ValueError(f"unknown ECC mode {ecc!r}")
    return _codes_payload(decoded, codec), float(np.mean(uncorrectable))


def seed_offsets(extent: int, count: int) -> np.ndarray:
    if count == 1:
        values = ((0, 0),)
    elif count == 2:
        values = ((0, 0), (extent // 2, extent // 2))
    elif count == 4:
        half = extent // 2
        values = ((0, 0), (0, half), (half, 0), (half, half))
    else:
        side = int(math.ceil(math.sqrt(count)))
        grid = tuple(
            (int(round(y * extent / side)) % extent,
             int(round(x * extent / side)) % extent)
            for y in range(side)
            for x in range(side)
        )
        values = grid[:count]
    return np.asarray(values, dtype=np.int16)


def renewal_seed_origins(
    pair_id: str,
    generation: int,
    replicates: int,
    extent: int,
    count: int,
    origin_policy: str,
    *,
    translated: bool = False,
) -> np.ndarray:
    base = repair_origins(
        pair_id,
        generation,
        replicates,
        extent,
        origin_policy,
        translated=translated,
    ).astype(np.int64)
    offsets = seed_offsets(extent, count).astype(np.int64)
    return ((base[:, None, :] + offsets[None, :, :]) % extent).astype(np.int16)


def seedify_payload(
    payload: np.ndarray, count: int, mode: str
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(payload, dtype=np.float32)
    rank = values.shape[-1]
    masks = np.ones((count, rank), dtype=np.float32)
    if mode == "partitioned":
        masks.fill(0.0)
        for channel in range(rank):
            masks[channel % count, channel] = 1.0
    elif mode not in ("single", "replicated"):
        raise ValueError(f"unknown seed mode {mode!r}")
    return values[:, None, :] * masks[None, :, :], masks


def seed_ablation_activity(
    pair_id: str,
    replicates: int,
    count: int,
    fraction: float,
    namespace: str,
) -> np.ndarray:
    """Return a family-paired mask with the exact requested aggregate dose."""

    if not 0.0 <= fraction <= 1.0:
        raise ValueError("seed-ablation fraction must be in [0, 1]")
    active = np.ones((replicates, count), dtype=np.bool_)
    remove = int(round(replicates * count * fraction))
    order = np.random.default_rng(
        _hash_seed(namespace, pair_id, "seed-ablation", fraction)
    ).permutation(replicates * count)
    active.reshape(-1)[order[:remove]] = False
    return np.concatenate((active, active), axis=0)


def _embed_seed_payloads(
    payloads: np.ndarray,
    origins: np.ndarray,
    extent: int,
    channel_masks: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(payloads, dtype=np.float32)
    samples, seeds, rank = values.shape
    masks = (
        np.asarray(channel_masks, dtype=np.bool_)
        if channel_masks is not None
        else np.ones(values.shape, dtype=np.bool_)
    )
    if masks.shape != values.shape:
        raise ValueError("channel masks must match distributed seed payloads")
    field = np.zeros((samples, extent, extent, rank), dtype=np.float32)
    channel_occupied = np.zeros_like(field, dtype=np.bool_)
    for sample in range(samples):
        for seed in range(seeds):
            y, x = np.asarray(origins[sample, seed], dtype=np.int64)
            # A zero-valued encoded channel is still a physical channel.  Its
            # presence is defined by the fixed partition, not by its value.
            active = masks[sample, seed]
            field[sample, y, x, active] = values[sample, seed, active]
            channel_occupied[sample, y, x, active] = True
    return field, channel_occupied


def propagate_channelwise(
    field: np.ndarray,
    channel_occupied: np.ndarray,
    hops: int,
    *,
    communication_cut: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Nearest-neighbour propagation with occupancy tracked per channel."""

    values = np.asarray(field, dtype=np.float32).copy()
    occupied = np.asarray(channel_occupied, dtype=np.bool_).copy()
    extent = values.shape[1]
    trace: list[float] = []
    for _ in range(hops):
        count = np.zeros(occupied.shape, dtype=np.int16)
        total = np.zeros(values.shape, dtype=np.float64)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if not (dy or dx):
                    continue
                shifted_occ = np.roll(occupied, (dy, dx), axis=(1, 2))
                shifted = np.roll(
                    values.astype(np.float64) * occupied,
                    (dy, dx),
                    axis=(1, 2),
                )
                if communication_cut and dx:
                    columns = (0, extent // 2) if dx == 1 else (
                        extent - 1, extent // 2 - 1
                    )
                    for column in columns:
                        shifted_occ[:, :, column] = False
                        shifted[:, :, column] = 0.0
                count += shifted_occ
                total += shifted
        new = (~occupied) & (count > 0)
        numerator = total + values.astype(np.float64) * occupied
        denominator = count + occupied.astype(np.int16)
        averaged = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 0,
        )
        values[occupied | new] = averaged[occupied | new]
        occupied |= new
        trace.append(float(np.mean(np.any(occupied, axis=-1))))
    return values, occupied, trace


def fit_shared_writer(
    moments: np.ndarray, targets: np.ndarray, ridge: float = 1e-6
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Fit one label-free shared ridge map from local moments to carrier latent."""

    x = np.asarray(moments, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y):
        raise ValueError("writer fit matrices must be aligned two-dimensional arrays")
    design = np.concatenate((x, np.ones((len(x), 1))), axis=1)
    penalty = np.eye(design.shape[1], dtype=np.float64) * ridge
    penalty[-1, -1] = ridge * 1e-3
    fitted = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    prediction = design @ fitted
    total = float(np.sum((y - y.mean(axis=0, keepdims=True)) ** 2))
    residual = float(np.sum((y - prediction) ** 2))
    return (
        fitted[:-1].astype(np.float32),
        fitted[-1].astype(np.float32),
        {"r2": 1.0 - residual / total if total > 0.0 else 0.0,
         "sample_count": float(len(x))},
    )


def _window_name(kind: str, extent_value: int) -> str:
    return f"{kind}-{2 * extent_value + 1}" if kind == "centered" else (
        f"directed-{extent_value + 1}"
    )


def build_renewal_candidates(frozen: dict[str, Any]) -> list[dict[str, Any]]:
    repair, _ = build_repair_candidates(frozen)
    base = next(row for row in repair if row["candidate_id"] == FULL_BRIDGE_ID)
    candidates: list[dict[str, Any]] = []
    for window_id, kind, value in WINDOWS:
        for fit_mode in FIT_MODES:
            for rho in TURNOVERS:
                candidate_id = (
                    f"renew-h05-w{window_id}-f{fit_mode}-r{int(rho * 100):02d}"
                )
                candidates.append(
                    {
                        **base,
                        "candidate_id": candidate_id,
                        "germination_hops": 5,
                        "window_id": window_id,
                        "window_kind": kind,
                        "window_value": value,
                        "consolidation_span": value,
                        "consolidation_steps": 2 * value,
                        "writer_fit_mode": fit_mode,
                        "rho": rho,
                        "origin_policy": "co-located",
                        "reader_mode": "field-local",
                        "normalization": "count-correct-v2",
                        "seed_count": 1,
                        "seed_mode": "single",
                        "ecc": "none",
                        "promotable": True,
                        "bounded": True,
                    }
                )
    if len(candidates) != 60:
        raise AssertionError("Stage-6B-R renewal grid must contain 60 candidates")
    return candidates


def candidate_public(candidate: dict[str, Any]) -> dict[str, Any]:
    excluded = {"codec_model", "weight", "bias"}
    payload = {
        key: value
        for key, value in candidate.items()
        if key not in excluded and not isinstance(value, np.ndarray)
    }
    for key in ("weight", "bias"):
        if key in candidate:
            payload[f"{key}_sha256"] = hashlib.sha256(
                np.asarray(candidate[key], dtype=np.float32).tobytes()
            ).hexdigest()
    codec = candidate.get("codec_model")
    if codec is not None:
        payload["codec_id"] = codec["candidate_id"]
        payload["codec_rank"] = int(codec["rank"])
        payload["codec_bits"] = int(codec["bits"])
    payload.update(
        {
            "runtime_label_access": False,
            "runtime_parent_access": False,
            "runtime_target_access": False,
        }
    )
    return payload


def _stage6ar_hashes(root: Path) -> dict[str, str]:
    relatives = (
        "DESIGN.json",
        "COHORTS.json",
        "audit/RESULTS.json",
        "bridge/RESULTS.json",
        "screen/RESULTS.json",
        "screen/STAGE_DECISION.json",
    )
    hashes: dict[str, str] = {}
    for relative in relatives:
        path = root / relative
        if not path.exists():
            raise FileNotFoundError(f"missing frozen Stage-6A-R artifact: {path}")
        hashes[relative] = _sha256(path)
    screen = _load_json(root / "screen/STAGE_DECISION.json")
    if screen.get("decision") != "phase_gate_failed":
        raise ValueError("Stage-6A-R screen no longer has its frozen negative decision")
    if screen.get("final_audit_trajectory_state") != "untouched":
        raise ValueError("Stage-6A-R reserve is not untouched")
    return hashes


def select_renewal_cohorts(
    base: dict[str, list[dict[str, Any]]],
    frozen: dict[str, Any],
    profile: RenewalProfile,
    *,
    open_final: bool,
) -> dict[str, list[dict[str, Any]]]:
    training = base["evolution_training"]
    train = training[: profile.writer_train_pairs]
    calibration = training[
        profile.writer_train_pairs:
        profile.writer_train_pairs + profile.repair_calibration_pairs
    ]
    if len(calibration) < profile.repair_calibration_pairs:
        calibration = training[-profile.repair_calibration_pairs:]
        train = training[: max(1, len(training) - len(calibration))]
    result = {
        "transient": base["locality_qualification"][: profile.transient_pairs],
        "writer_train": train,
        "repair_calibration": calibration,
        "repair_validation": base["evolution_validation"][
            : profile.repair_validation_pairs
        ],
        "coverage": base["compression_screen"][: profile.coverage_validation_pairs],
        "scale": base["ecology"][: profile.scale_pairs],
        "qualification": base["compression_qualification"][
            : profile.qualification_pairs
        ],
        "endurance": base["locality_endurance"][: profile.endurance_pairs],
        "final": [],
    }
    if open_final:
        reserve = base.get("audit", [])
        if len(reserve) < profile.final_pairs:
            raise ValueError("opened Stage-6B-R reserve is incomplete")
        result["final"] = reserve[: profile.final_pairs]
    used = [
        str(pair["pair_id"])
        for name, rows in result.items()
        if name != "final"
        for pair in rows
    ]
    if len(used) != len(set(used)):
        raise AssertionError("Stage-6B-R developmental cohorts overlap")
    if set(used) & set(frozen["later_ids"]):
        raise AssertionError("Stage-6B-R development touched the final reserve")
    return result


def _window_endpoint(
    site_sign_sum: np.ndarray,
    candidate: dict[str, Any],
    origins: np.ndarray,
    *,
    disabled: bool,
) -> tuple[np.ndarray, int, int]:
    kind = str(candidate["window_kind"])
    value = 0 if disabled else int(candidate["window_value"])
    if kind == "centered":
        endpoint = centered_reduce_endpoint(site_sign_sum, value, origins)
        site_count = (2 * value + 1) ** 2
        routing_steps = 2 * value
    elif kind == "directed":
        endpoint = bounded_reduce_endpoint(site_sign_sum, value, origins)
        site_count = (value + 1) ** 2
        routing_steps = 2 * value
    else:
        raise ValueError(f"unknown writer window {kind!r}")
    return endpoint, site_count, routing_steps


def _site_count_moments(
    endpoint_mean: np.ndarray,
    signs: np.ndarray,
    site_count: int,
    write_sweeps: int,
    alpha: float,
) -> np.ndarray:
    signed_count = np.asarray(endpoint_mean, dtype=np.float64) * float(site_count)
    prior = alpha * np.asarray(signs, dtype=np.float64).sum(axis=0)
    return (signed_count + prior) / (
        write_sweeps * float(site_count) + 512.0 * alpha
    )


def _flatten_seed_payload(payload: np.ndarray) -> np.ndarray:
    values = np.asarray(payload, dtype=np.float32)
    return values.reshape(values.shape[0], -1)


def _seed_boundary_intervention(
    payload: np.ndarray,
    candidate: dict[str, Any],
    condition: str,
    generation: int,
    pair_id: str,
    replicates: int,
    source_exits: Sequence[np.ndarray] | None,
    contract: RenewalContract,
) -> tuple[np.ndarray, float, float]:
    values = np.asarray(payload, dtype=np.float32)
    samples, seeds, rank = values.shape
    flat = values.reshape(samples * seeds, rank)
    flat_sources = (
        [np.asarray(row).reshape(samples * seeds, rank) for row in source_exits]
        if source_exits is not None
        else None
    )
    delegated = "intact" if condition == "carrier_corruption_1" else condition
    updated, clipping = _boundary_intervention(
        flat,
        candidate,
        delegated,
        generation,
        pair_id,
        replicates * seeds,
        flat_sources,
        contract,  # type: ignore[arg-type]
    )
    result = updated.reshape(samples, seeds, rank)
    uncorrectable = 0.0
    bit_count = 8 if candidate.get("ecc") == "hamming84" else 4
    flips = None
    if condition == "carrier_corruption_1":
        half = np.random.default_rng(
            _hash_seed(contract.namespace, pair_id, "bit-corruption", generation)
        ).random((replicates, seeds, rank, bit_count)) < contract.bit_corruption
        flips = np.concatenate((half, half), axis=0)
    result, uncorrectable = coded_payload_roundtrip(
        result, candidate["codec_model"], str(candidate.get("ecc", "none")), flips
    )
    return result, clipping, uncorrectable


def _seed_field(
    payloads: np.ndarray,
    origins: np.ndarray,
    extent: int,
    hops: int,
    mode: str,
    *,
    communication_cut: bool,
    seed_active: np.ndarray | None = None,
    channel_masks: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    values = np.asarray(payloads, dtype=np.float32)
    active_seeds = (
        np.asarray(seed_active, dtype=np.bool_)
        if seed_active is not None
        else np.ones(values.shape[:2], dtype=np.bool_)
    )
    if active_seeds.shape != values.shape[:2]:
        raise ValueError("seed activity mask has the wrong shape")
    if mode in ("single", "replicated"):
        samples, seeds, rank = values.shape
        field = np.zeros((samples, extent, extent, rank), dtype=np.float32)
        occupied = np.zeros((samples, extent, extent), dtype=np.bool_)
        for sample in range(samples):
            for seed in range(seeds):
                if not active_seeds[sample, seed]:
                    continue
                y, x = np.asarray(origins[sample, seed], dtype=np.int64)
                if occupied[sample, y, x]:
                    field[sample, y, x] = 0.5 * (
                        field[sample, y, x] + values[sample, seed]
                    )
                else:
                    field[sample, y, x] = values[sample, seed]
                occupied[sample, y, x] = True
        return propagate_bounded(
            field, occupied, hops, communication_cut=communication_cut
        )
    masks = (
        np.asarray(channel_masks, dtype=np.bool_)
        if channel_masks is not None
        else np.ones(values.shape, dtype=np.bool_)
    )
    masks &= active_seeds[..., None]
    field, channel_occupied = _embed_seed_payloads(
        values, origins, extent, masks
    )
    field, channel_occupied, trace = propagate_channelwise(
        field,
        channel_occupied,
        hops,
        communication_cut=communication_cut,
    )
    return field, np.any(channel_occupied, axis=-1), trace


def _writer_overlap(
    current: np.ndarray,
    nxt: np.ndarray,
    extent: int,
    candidate: dict[str, Any],
) -> float:
    hops = int(candidate["germination_hops"])
    values: list[float] = []
    for seed in range(current.shape[1]):
        if candidate["window_kind"] == "centered":
            values.append(
                centred_causal_overlap(
                    current[:, seed],
                    nxt[:, seed],
                    extent,
                    hops,
                    int(candidate["window_value"]),
                )
            )
        else:
            span = int(candidate["window_value"])
            reached_radius = hops
            overlaps: list[float] = []
            for left, right in zip(current[:, seed], nxt[:, seed]):
                reached = {
                    ((int(left[0]) + dy) % extent,
                     (int(left[1]) + dx) % extent)
                    for dy in range(-reached_radius, reached_radius + 1)
                    for dx in range(-reached_radius, reached_radius + 1)
                }
                window = {
                    ((int(right[0]) - dy) % extent,
                     (int(right[1]) - dx) % extent)
                    for dy in range(span + 1)
                    for dx in range(span + 1)
                }
                overlaps.append(len(reached & window) / max(1, len(window)))
            values.append(float(np.mean(overlaps)))
    return float(np.mean(values))


def _nearest_seed_distance(
    extent: int, origins: np.ndarray
) -> np.ndarray:
    """Chebyshev distance to the nearest member of each seed constellation."""

    points = np.asarray(origins, dtype=np.int64)
    if points.ndim != 3 or points.shape[-1] != 2:
        raise ValueError("seed origins require sample, seed, coordinate axes")
    yy, xx = np.indices((extent, extent))
    result = np.full((len(points), extent, extent), extent, dtype=np.int16)
    for sample, constellation in enumerate(points):
        for oy, ox in constellation:
            dy = np.minimum(np.abs(yy - oy), extent - np.abs(yy - oy))
            dx = np.minimum(np.abs(xx - ox), extent - np.abs(xx - ox))
            result[sample] = np.minimum(
                result[sample], np.maximum(dy, dx)
            )
    return result


def _field_distance_summary_multiseed(
    field: np.ndarray,
    occupied: np.ndarray,
    origins: np.ndarray,
) -> dict[str, Any]:
    extent = field.shape[1]
    distances = _nearest_seed_distance(extent, origins)
    bands: dict[str, list[float]] = {
        "0-2": [], "3-5": [], "6-8": [], "9+": []
    }
    signal: dict[str, list[float]] = {key: [] for key in bands}
    for sample, distance in enumerate(distances):
        masks = {
            "0-2": distance <= 2,
            "3-5": (distance >= 3) & (distance <= 5),
            "6-8": (distance >= 6) & (distance <= 8),
            "9+": distance >= 9,
        }
        for key, mask in masks.items():
            if np.any(mask):
                bands[key].append(float(np.mean(occupied[sample][mask])))
                signal[key].append(float(np.mean(np.abs(field[sample][mask]))))
    return {
        key: {
            "occupied_fraction": (
                float(np.mean(bands[key])) if bands[key] else None
            ),
            "mean_abs_signal": (
                float(np.mean(signal[key])) if signal[key] else None
            ),
        }
        for key in bands
    }


def _outside_light_cone_fraction(
    occupied: np.ndarray, origins: np.ndarray, hops: int
) -> float:
    distance = _nearest_seed_distance(occupied.shape[1], origins)
    outside = distance > hops
    if not np.any(outside):
        return 0.0
    return float(np.count_nonzero(occupied & outside) / np.count_nonzero(outside))


def _phenotype_distance_outcomes_multiseed(
    states: np.ndarray,
    origins: np.ndarray,
    pair: dict[str, Any],
    alive: np.ndarray,
    replicates: int,
    writer_contract: MotifContract,
    hops: int,
) -> dict[str, Any]:
    """Score phenotype recovery in bands around the nearest inherited seed."""

    extent = states.shape[1]
    distance = _nearest_seed_distance(extent, origins)
    codes = states.astype(np.uint8)
    codes |= np.roll(states, -1, axis=2).astype(np.uint8) << 1
    codes |= np.roll(states, -1, axis=1).astype(np.uint8) << 2
    codes |= (
        np.roll(np.roll(states, -1, axis=1), -1, axis=2).astype(np.uint8)
        << 3
    )
    masks = {
        "inside-cone": distance <= hops,
        "outside-cone": distance > hops,
        "0-2": distance <= 2,
        "3-5": (distance >= 3) & (distance <= 5),
        "6-8": (distance >= 6) & (distance <= 8),
        "9+": distance >= 9,
    }
    targets = pair["targets"].get(
        "primary_terminal", pair["targets"]["primary"]
    )
    result: dict[str, Any] = {}
    for key, band_masks in masks.items():
        counts = np.zeros((len(states), 15), dtype=np.float64)
        for sample, mask in enumerate(band_masks):
            if not np.any(mask):
                continue
            for code in range(1, 16):
                counts[sample, code - 1] = np.count_nonzero(
                    codes[sample][mask] == code
                )
        labels = _cosine_labels(
            _normalize_rows(counts), targets["A"], targets["B"], writer_contract
        )
        result[key] = _outcome(labels, alive, replicates)
    return result


def simulate_renewal_lineage(
    pair: dict[str, Any],
    configuration: ReaderConfiguration,
    candidate: dict[str, Any],
    condition: str,
    replicates: int,
    generations: int,
    reference: dict[int, dict[str, np.ndarray]],
    writer_contract: MotifContract,
    contract: RenewalContract,
    *,
    extent: int = 16,
    source_exits: Sequence[np.ndarray] | None = None,
    retain_exits: bool = False,
    collect_training: bool = False,
) -> tuple[dict[str, Any], list[np.ndarray]]:
    """Run a visible-reset lineage with a renewable, possibly distributed latch."""

    valid = set(FULL_CONDITIONS) | {"seed_ablation_25", "seed_ablation_50"}
    if condition not in valid:
        raise ValueError(f"unknown Stage-6B-R condition {condition!r}")
    if extent not in (16, 32, 64):
        raise ValueError("Stage-6B-R extents are 16, 32, and 64")
    pair_id = str(pair["pair_id"])
    rule = int(contract.rule)
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
    founder, founder_terminal = _founder_bounded_payload(
        pair,
        candidate,
        reference_probability,
        replace(writer_contract, rule=rule),
        replicates,
        extent,
        rule,
    )
    seed_count = int(candidate.get("seed_count", 1))
    seed_mode = str(candidate.get("seed_mode", "single"))
    payload, seed_masks = seedify_payload(founder, seed_count, seed_mode)
    founder_payload = payload.copy()
    if condition == "founder_write_disabled":
        payload.fill(0.0)
    alive = np.ones(2 * replicates, dtype=np.bool_)
    checkpoints = {
        value
        for value in (1, 2, 4, 6, 8, 16, 32, 64)
        if value <= generations
    }
    outcomes: dict[str, Any] = {}
    decoders: dict[str, Any] = {}
    carrier_history: dict[str, Any] = {}
    writer_history: dict[str, Any] = {}
    origin_history: dict[str, Any] = {}
    exits: list[np.ndarray] = []
    training_moments: list[np.ndarray] = []
    training_targets: list[np.ndarray] = []
    coverage_values: list[float] = []
    clipping_values: list[float] = []
    uncorrectable_values: list[float] = []
    codec = candidate["codec_model"]
    basis = np.asarray(codec["basis"], dtype=np.float32)
    signs = basis.astype(np.float64) * math.sqrt(512.0)
    write_sweeps = contract.write_end - contract.write_start + 1
    alpha = float(writer_contract.jeffreys_alpha)
    hops = int(candidate["germination_hops"])
    origin_policy = str(candidate.get("origin_policy", "co-located"))
    rho = float(candidate.get("rho", 0.0))

    for generation in range(1, generations + 1):
        payload, clipping, uncorrectable = _seed_boundary_intervention(
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
        uncorrectable_values.append(uncorrectable)
        entry_payload = payload.copy()
        translated = condition == "translated_patch"
        origins = renewal_seed_origins(
            pair_id,
            generation,
            replicates,
            extent,
            seed_count,
            origin_policy,
            translated=translated,
        )
        next_origins = renewal_seed_origins(
            pair_id,
            generation + 1,
            replicates,
            extent,
            seed_count,
            origin_policy,
            translated=translated,
        )
        active_payload = entry_payload.copy()
        seed_active = np.ones((2 * replicates, seed_count), dtype=np.bool_)
        if condition == "seed_ablation_25":
            seed_active = seed_ablation_activity(
                pair_id, replicates, seed_count, 0.25, contract.namespace
            )
            active_payload[~seed_active] = 0.0
        elif condition == "seed_ablation_50":
            seed_active = seed_ablation_activity(
                pair_id, replicates, seed_count, 0.50, contract.namespace
            )
            active_payload[~seed_active] = 0.0
        field, occupied, wave_trace = _seed_field(
            active_payload,
            origins,
            extent,
            0 if condition == "transport_disabled" else hops,
            seed_mode,
            communication_cut=condition == "communication_cut",
            seed_active=seed_active,
            channel_masks=np.broadcast_to(
                seed_masks[None, :, :], active_payload.shape
            ),
        )
        if condition == "regeneration_disabled":
            field.fill(0.0)
            occupied.fill(False)
        coverage_values.append(float(np.mean(occupied)))
        field_distance = _field_distance_summary_multiseed(
            field, occupied, origins
        )
        outside_light_cone = _outside_light_cone_fraction(
            occupied, origins, 0 if condition == "transport_disabled" else hops
        )
        state = reset.copy()
        state[~alive] = False
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
                if np.all(occupied) and np.array_equal(
                    field, np.broadcast_to(field[:, :1, :1], field.shape)
                ):
                    predicted = apply_energy_reader(
                        predicted,
                        decode_payload(field[:, 0, 0], codec),
                        uniforms,
                        configuration.strength,
                    )
                else:
                    predicted = apply_local_reader(
                        predicted,
                        field,
                        basis,
                        uniforms,
                        configuration.strength,
                    )
            predicted ^= _semantic_uniforms(
                pair_id, "process", generation, sweep, replicates, extent
            ) < contract.process_noise
            predicted[~alive] = False
            state = predicted
            if contract.write_start <= sweep <= contract.write_end:
                site_sign_sum += signs[motif3_codes(state)]
            if sweep >= contract.observe_start:
                recent.append(live_2x2_counts_batch(state))
        alive &= state.any(axis=(1, 2))

        moment_rows: list[np.ndarray] = []
        site_count = 1
        routing_steps = 0
        for seed in range(seed_count):
            endpoint, site_count, routing_steps = _window_endpoint(
                site_sign_sum,
                candidate,
                next_origins[:, seed],
                disabled=condition == "consolidation_disabled",
            )
            moment_rows.append(
                _site_count_moments(
                    endpoint, signs, site_count, write_sweeps, alpha
                )
            )
        moments = np.stack(moment_rows, axis=1)
        proposal = (
            moments @ np.asarray(candidate["weight"], dtype=np.float64)
            + np.asarray(candidate["bias"], dtype=np.float64)
        ).astype(np.float32)
        proposal *= seed_masks[None, :, :]
        if collect_training:
            training_moments.append(moments.reshape(-1, moments.shape[-1]))
            target = np.repeat(founder[:, None, :], seed_count, axis=1)
            target *= seed_masks[None, :, :]
            training_targets.append(target.reshape(-1, target.shape[-1]))
        if condition == "write_disabled":
            next_payload = np.zeros_like(entry_payload)
            clipping = 0.0
        elif condition == "founder_clamped":
            next_payload = founder_payload.copy()
            clipping = 0.0
        else:
            renewal_rho = (
                float(candidate.get("no_rewrite_rho", rho))
                if condition == "no_rewrite"
                else rho
            )
            next_payload, clipping = latch_update(
                entry_payload,
                proposal,
                renewal_rho,
                codec,
                write_enabled=condition != "no_rewrite",
            )
            next_payload *= seed_masks[None, :, :]
        if condition in ("seed_ablation_25", "seed_ablation_50"):
            next_payload[~seed_active] = 0.0
        clipping_values.append(clipping)
        payload = next_payload
        payload[~alive] = 0.0

        if generation in checkpoints:
            outcome, phenotype = _score_state(
                state, recent, pair, founder_terminal, replicates, writer_contract
            )
            outcome["distance_bands"] = _phenotype_distance_outcomes_multiseed(
                state,
                origins,
                pair,
                alive,
                replicates,
                writer_contract,
                hops,
            )
            outcomes[str(generation)] = outcome
            flattened = _flatten_seed_payload(payload)
            entry_flat = _flatten_seed_payload(entry_payload)
            carrier_decoder = heldout_lineage_diagnostics(
                flattened,
                replicates,
                _hash_seed(
                    contract.namespace,
                    pair_id,
                    condition,
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
                    generation,
                    "phenotype",
                ),
                contract.decoder_splits,
            )
            decoders[str(generation)] = {
                "carrier_balanced_accuracy": carrier_decoder["balanced_accuracy"],
                "carrier_tie_fraction": carrier_decoder["tie_fraction"],
                "carrier_information_lower_bound_bits": (
                    _binary_information_lower_bound(
                        carrier_decoder["balanced_accuracy"]
                    )
                ),
                "phenotype_balanced_accuracy": phenotype_decoder[
                    "balanced_accuracy"
                ],
                "phenotype_tie_fraction": phenotype_decoder["tie_fraction"],
                "phenotype_information_lower_bound_bits": (
                    _binary_information_lower_bound(
                        phenotype_decoder["balanced_accuracy"]
                    )
                ),
            }
            carrier_history[str(generation)] = {
                "entry": _payload_summary(entry_flat, replicates),
                "exit": _payload_summary(flattened, replicates),
                "transition": _transition_summary(
                    entry_flat, flattened, replicates
                ),
                "occupied_fraction_after_germination": float(np.mean(occupied)),
                "wave_trace": wave_trace,
                "distance_bands": field_distance,
                "outside_light_cone_occupied_fraction": outside_light_cone,
            }
            writer_history[str(generation)] = {
                "window_kind": candidate["window_kind"],
                "window_value": int(candidate["window_value"]),
                "window_site_count": site_count,
                "routing_steps": routing_steps,
                "moment_mean_abs": float(np.mean(np.abs(moments))),
                "latent_mean_abs": float(np.mean(np.abs(proposal))),
                "latent_centroid_l2": _payload_summary(
                    _flatten_seed_payload(proposal), replicates
                )["centroid_l2"],
            }
            displacement = toroidal_chebyshev_distance(
                origins.reshape(-1, 2), next_origins.reshape(-1, 2), extent
            )
            origin_history[str(generation)] = {
                "policy": origin_policy,
                "mean_displacement": float(np.mean(displacement)),
                "max_displacement": int(np.max(displacement)),
                "causal_overlap_fraction": _writer_overlap(
                    origins, next_origins, extent, candidate
                ),
                "translation_applied_to_read_and_write": translated,
            }
        if retain_exits:
            exits.append(payload.copy())

    physical_bits = int(candidate["payload_bits"])
    if seed_mode == "replicated":
        physical_bits *= seed_count
    if candidate.get("ecc") == "hamming84":
        physical_bits *= 2
    rank = int(candidate["rank"])
    routing_steps_total = hops + 2 * int(candidate["window_value"])
    result: dict[str, Any] = {
        "candidate_id": str(candidate["candidate_id"]),
        "condition": condition,
        "rule": rule,
        "extent": extent,
        "reset_sha256": hashlib.sha256(reset_a.tobytes()).hexdigest(),
        "reset_asserted_before_every_generation": True,
        "logical_payload_bits": int(candidate["payload_bits"]),
        "physical_inherited_bits": physical_bits,
        "developmental_field_values": extent * extent * rank,
        "developmental_writer_values": extent * extent * rank,
        "routing_site_channel_updates_upper_bound": (
            extent * extent * rank * routing_steps_total
        ),
        "seed_density": seed_count / float(extent * extent),
        "shared_writer_parameter_bits": int(
            (
                np.asarray(candidate["weight"]).size
                + np.asarray(candidate["bias"]).size
            )
            * 32
        ),
        "seed_count": seed_count,
        "seed_mode": seed_mode,
        "seed_ablation_fraction_actual": float(np.mean(~seed_active)),
        "germination_hops": hops,
        "window_kind": candidate["window_kind"],
        "window_value": int(candidate["window_value"]),
        "rho": rho,
        "ecc": candidate.get("ecc", "none"),
        "founder_payload": _payload_summary(
            _flatten_seed_payload(founder_payload), replicates
        ),
        "boundary_clipping_fraction_mean": float(np.mean(clipping_values)),
        "uncorrectable_codeword_fraction_mean": float(
            np.mean(uncorrectable_values)
        ),
        "germination_coverage_mean": float(np.mean(coverage_values)),
        "outcomes": outcomes,
        "decoders": decoders,
        "carrier_history": carrier_history,
        "writer_history": writer_history,
        "origin_history": origin_history,
    }
    if collect_training:
        result["training"] = {
            "moments": np.concatenate(training_moments).astype(np.float32).tolist(),
            "targets": np.concatenate(training_targets).astype(np.float32).tolist(),
        }
    return result, exits


def _renewal_task(
    payload: tuple[
        dict[str, Any],
        list[dict[str, Any]],
        MotifContract,
        RenewalContract,
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
    collect_training = bool(item.get("collect_training", False))
    extent = int(item.get("extent", 16))
    results: dict[str, Any] = {}
    exits: list[np.ndarray] | None = None
    if "intact" in conditions or any("rescue" in value for value in conditions):
        intact, exits = simulate_renewal_lineage(
            item["pair"],
            configuration,
            candidate,
            "intact",
            replicates,
            generations,
            reference,
            writer_contract,
            contract,
            extent=extent,
            retain_exits=True,
            collect_training=collect_training,
        )
        if "intact" in conditions:
            results["intact"] = intact
    for condition in conditions:
        if condition == "intact":
            continue
        row, _ = simulate_renewal_lineage(
            item["pair"],
            configuration,
            candidate,
            condition,
            replicates,
            generations,
            reference,
            writer_contract,
            contract,
            extent=extent,
            source_exits=exits,
            collect_training=collect_training,
        )
        results[condition] = row
    return {
        "checkpoint": item["checkpoint"],
        "pair_id": item["pair"]["pair_id"],
        "candidate_id": candidate["candidate_id"],
        "replicates": replicates,
        "generations": generations,
        "extent": extent,
        "conditions": results,
    }


def _boot(
    values: Sequence[float],
    profile: RenewalProfile,
    contract: RenewalContract,
    *key: object,
) -> dict[str, Any]:
    return _bootstrap(
        values,
        profile.bootstrap_resamples,
        _hash_seed(contract.namespace, *key),
        contract.strict_alpha,
    )


def _diagnostic_mean(
    rows: Sequence[dict[str, Any]],
    candidate_id: str,
    condition: str,
    generation: int,
    section: str,
    key: str,
) -> float | None:
    values: list[float] = []
    for row in rows:
        if row.get("candidate_id") != candidate_id:
            continue
        try:
            values.append(
                float(row["conditions"][condition][section][str(generation)][key])
            )
        except KeyError:
            continue
    return float(np.mean(values)) if values else None


def _diagnostic_values_stage6br(
    rows: Sequence[dict[str, Any]],
    candidate_id: str,
    condition: str,
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
                float(row["conditions"][condition][section][str(generation)][key])
            )
        except KeyError:
            continue
    return values


def summarize_candidate(
    rows: Sequence[dict[str, Any]],
    candidate: dict[str, Any],
    generation: int,
    profile: RenewalProfile,
    contract: RenewalContract,
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
    survival = _condition_values(
        rows, candidate_id, "intact", generation, "survival"
    )
    active = _paired_advantage(
        rows, candidate_id, "intact", "no_rewrite", generation
    ) if _condition_values(rows, candidate_id, "no_rewrite", generation) else []
    translated = _condition_values(
        rows, candidate_id, "translated_patch", generation
    )
    mean = float(np.mean(values)) if values else 0.0
    translated_mean = float(np.mean(translated)) if translated else 0.0
    corrupted = _condition_values(
        rows, candidate_id, "carrier_corruption_1", generation
    )
    corrupted_mean = float(np.mean(corrupted)) if corrupted else 0.0
    result: dict[str, Any] = {
        "candidate": candidate_public(candidate),
        "crossover": _boot(
            values, profile, contract, phase, candidate_id, generation, "intact"
        ),
        "direction_a_mean": float(np.mean(direction_a)) if direction_a else 0.0,
        "direction_b_mean": float(np.mean(direction_b)) if direction_b else 0.0,
        "fraction_pairs_positive": (
            float(np.mean(np.asarray(values) > 0.0)) if values else 0.0
        ),
        "survival_mean": float(np.mean(survival)) if survival else 0.0,
        "active_rewrite_advantage": _boot(
            active, profile, contract, phase, candidate_id, generation, "active"
        ),
        "translated": _boot(
            translated,
            profile,
            contract,
            phase,
            candidate_id,
            generation,
            "translated",
        ),
        "translation_retention": (
            translated_mean / mean if mean > 0.0 else 0.0
        ),
        "carrier_corruption": _boot(
            corrupted,
            profile,
            contract,
            phase,
            candidate_id,
            generation,
            "carrier-corruption",
        ),
        "carrier_corruption_retention": (
            corrupted_mean / mean if mean > 0.0 and corrupted else None
        ),
    }
    diagnostic_fields = (
        ("decoders", "carrier_balanced_accuracy", "carrier_decoder_mean"),
        ("decoders", "phenotype_balanced_accuracy", "phenotype_decoder_mean"),
        (
            "carrier_history",
            "occupied_fraction_after_germination",
            "coverage_mean",
        ),
        ("origin_history", "causal_overlap_fraction", "causal_overlap_mean"),
        ("writer_history", "moment_mean_abs", "writer_moment_mean_abs"),
    )
    for section, key, output_key in diagnostic_fields:
        result[output_key] = _diagnostic_mean(
            rows, candidate_id, "intact", generation, section, key
        )
    transitions: dict[str, float | None] = {}
    for key in ("centroid_retention", "parent_child_delta_cosine"):
        values_diag: list[float] = []
        for row in rows:
            if row.get("candidate_id") != candidate_id:
                continue
            try:
                values_diag.append(
                    float(
                        row["conditions"]["intact"]["carrier_history"][
                            str(generation)
                        ]["transition"][key]
                    )
                )
            except KeyError:
                continue
        transitions[key] = (
            float(np.mean(values_diag)) if values_diag else None
        )
    result.update(
        {
            "carrier_centroid_retention_mean": transitions["centroid_retention"],
            "carrier_parent_child_delta_cosine_mean": transitions[
                "parent_child_delta_cosine"
            ],
        }
    )
    return result


def _loss_retention(
    rows: Sequence[dict[str, Any]],
    candidate_id: str,
    condition: str,
    generation: int,
) -> float:
    intact = _condition_values(rows, candidate_id, "intact", generation)
    control = _condition_values(rows, candidate_id, condition, generation)
    intact_mean = float(np.mean(intact)) if intact else 0.0
    control_mean = float(np.mean(control)) if control else 0.0
    return control_mean / intact_mean if intact_mean > 0.0 else 1.0


def candidate_gate(
    summary: dict[str, Any],
    contract: RenewalContract,
    *,
    minimum: float,
    anchor_mean: float | None = None,
) -> bool:
    lower = summary["crossover"]["ci"][0]
    active_lower = summary["active_rewrite_advantage"]["ci"][0]
    mean = float(summary["crossover"]["mean"] or 0.0)
    anchor_retention = (
        mean / anchor_mean if anchor_mean is not None and anchor_mean > 0.0 else 1.0
    )
    summary["anchor_retention"] = anchor_retention
    return bool(
        mean >= minimum
        and lower is not None
        and float(lower) > 0.0
        and active_lower is not None
        and float(active_lower) > 0.0
        and summary["direction_a_mean"] > 0.0
        and summary["direction_b_mean"] > 0.0
        and summary["fraction_pairs_positive"] >= 0.50
        and summary["survival_mean"] >= contract.survival_gate
        and summary["translation_retention"] >= contract.translation_retention
        and anchor_retention >= contract.anchor_retention
    )


def _write_phase_outputs(
    output: Path,
    phase: str,
    design_digest: str,
    result: dict[str, Any],
    report: str,
    lay: str,
    *,
    next_phase: str | None,
) -> None:
    root = output / phase
    root.mkdir(parents=True, exist_ok=True)
    payload = result | {"phase": phase, "design_digest": design_digest}
    decision = {
        "experiment": "ca_motif_lineage_stage_6br",
        "state": result.get("state"),
        "phase": phase,
        "design_digest": design_digest,
        "stage_gate": bool(result.get("stage_gate")),
        "decision": (
            "campaign_complete"
            if phase == PHASES[-1]
            else "continue_registered_exploration"
        ),
        "next_phase": next_phase,
        "automatic_launch": next_phase is not None,
        "final_audit_trajectory_state": result.get(
            "final_audit_trajectory_state", "untouched"
        ),
    }
    _atomic_json(root / "RESULTS.json", payload)
    _atomic_json(root / "STAGE_DECISION.json", decision)
    _atomic_text(root / "REPORT.md", report)
    _atomic_text(root / "LAY_SUMMARY.md", lay)
    if result.get("state") == "complete":
        _atomic_text(root / "COMPLETE", "complete\n")
    _atomic_json(output / "RESULTS.json", payload)
    _atomic_json(output / "STAGE_DECISION.json", decision)
    _atomic_text(output / "REPORT.md", report)
    _atomic_text(output / "LAY_SUMMARY.md", lay)


def _model_paths(output: Path) -> tuple[Path, Path]:
    return output / "RENEWAL_MODELS.json", output / "RENEWAL_MODELS.npz"


def write_models(
    output: Path,
    candidates: Sequence[dict[str, Any]],
    audits: Sequence[dict[str, Any]],
    design_digest: str,
) -> None:
    metadata_path, array_path = _model_paths(output)
    arrays: dict[str, np.ndarray] = {}
    public: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        prefix = f"candidate_{index:03d}"
        arrays[f"{prefix}_weight"] = np.asarray(
            candidate["weight"], dtype=np.float32
        )
        arrays[f"{prefix}_bias"] = np.asarray(candidate["bias"], dtype=np.float32)
        codec = candidate["codec_model"]
        arrays[f"{prefix}_codec_basis"] = np.asarray(
            codec["basis"], dtype=np.float32
        )
        arrays[f"{prefix}_codec_scale"] = np.asarray(
            codec["quantizer_scale"], dtype=np.float32
        )
        public.append(
            candidate_public(candidate)
            | {
                "codec": {
                    key: codec[key]
                    for key in (
                        "candidate_id",
                        "family",
                        "rank",
                        "bits",
                        "payload_bits",
                    )
                },
                "array_keys": {
                    "weight": f"{prefix}_weight",
                    "bias": f"{prefix}_bias",
                    "codec_basis": f"{prefix}_codec_basis",
                    "codec_scale": f"{prefix}_codec_scale",
                }
            }
        )
    temporary = array_path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, array_path)
    _atomic_json(
        metadata_path,
        {
            "experiment": "ca_motif_lineage_stage_6br",
            "design_digest": design_digest,
            "candidate_count": len(candidates),
            "candidates": public,
            "fit_audits": list(audits),
            "model_sha256": _sha256(array_path),
        },
    )


def load_models(output: Path, design_digest: str) -> list[dict[str, Any]]:
    metadata_path, array_path = _model_paths(output)
    metadata = _load_json(metadata_path)
    if metadata.get("design_digest") != design_digest:
        raise ValueError("Stage-6B-R model design mismatch")
    if _sha256(array_path) != metadata.get("model_sha256"):
        raise ValueError("Stage-6B-R model archive hash mismatch")
    candidates: list[dict[str, Any]] = []
    with np.load(array_path, allow_pickle=False) as arrays:
        for row in metadata["candidates"]:
            candidate = {
                key: value
                for key, value in row.items()
                if key not in ("array_keys", "codec")
            }
            candidate["weight"] = np.asarray(
                arrays[row["array_keys"]["weight"]], dtype=np.float32
            )
            candidate["bias"] = np.asarray(
                arrays[row["array_keys"]["bias"]], dtype=np.float32
            )
            candidate["codec_model"] = dict(row["codec"]) | {
                "basis": np.asarray(
                    arrays[row["array_keys"]["codec_basis"]], dtype=np.float32
                ),
                "quantizer_scale": np.asarray(
                    arrays[row["array_keys"]["codec_scale"]], dtype=np.float32
                ),
                "runtime_label_access": False,
                "runtime_parent_access": False,
                "runtime_target_access": False,
            }
            candidates.append(candidate)
    return candidates


def _collect_fit_arrays(
    rows: Sequence[dict[str, Any]], condition: str
) -> tuple[np.ndarray, np.ndarray]:
    moments: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for row in rows:
        try:
            training = row["conditions"][condition]["training"]
        except KeyError:
            continue
        moments.append(np.asarray(training["moments"], dtype=np.float32))
        targets.append(np.asarray(training["targets"], dtype=np.float32))
    if not moments:
        raise ValueError("writer training produced no local moment samples")
    return np.concatenate(moments), np.concatenate(targets)


def _select_diverse(
    summaries: dict[str, dict[str, Any]],
    candidates: Sequence[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    ranked = sorted(
        candidates,
        key=lambda row: (
            -float(summaries[str(row["candidate_id"])]["crossover"]["mean"] or -1),
            int(row["physical_inherited_bits"])
            if "physical_inherited_bits" in row
            else int(row.get("payload_bits", 64)),
            str(row["candidate_id"]),
        ),
    )
    selected: list[dict[str, Any]] = []
    seen_fit: set[str] = set()
    seen_window: set[str] = set()
    for candidate in ranked:
        fit_mode = str(candidate.get("writer_fit_mode", ""))
        window = str(candidate.get("window_id", ""))
        if fit_mode not in seen_fit or window not in seen_window:
            selected.append(candidate)
            seen_fit.add(fit_mode)
            seen_window.add(window)
        if len(selected) == limit:
            return selected
    selected_ids = {str(row["candidate_id"]) for row in selected}
    for candidate in ranked:
        if str(candidate["candidate_id"]) not in selected_ids:
            selected.append(candidate)
            selected_ids.add(str(candidate["candidate_id"]))
        if len(selected) == limit:
            break
    return selected


def _checkpoint_items(
    cohort: Sequence[dict[str, Any]],
    candidates: Sequence[dict[str, Any]],
    configuration: dict[str, Any],
    *,
    prefix: str,
    replicates: int,
    generations: int,
    conditions: Sequence[str],
    extent: int = 16,
    collect_training: bool = False,
) -> list[dict[str, Any]]:
    return [
        {
            "checkpoint": f"{prefix}-p{pair_index:04d}-c{candidate_index:03d}",
            "pair": pair,
            "candidate_id": candidate["candidate_id"],
            "configuration": configuration,
            "replicates": replicates,
            "generations": generations,
            "conditions": tuple(conditions),
            "extent": extent,
            "collect_training": collect_training,
        }
        for pair_index, pair in enumerate(cohort)
        for candidate_index, candidate in enumerate(candidates)
    ]


def _run_transient(
    output: Path,
    frozen: dict[str, Any],
    cohorts: dict[str, list[dict[str, Any]]],
    profile: RenewalProfile,
    contract: RenewalContract,
    writer_contract: MotifContract,
    design_digest: str,
    *,
    workers: int,
    resume: bool,
    deadline: float,
    status,
    scientific: bool,
) -> dict[str, Any]:
    repair_candidates, _ = build_repair_candidates(frozen)
    by_id = {str(row["candidate_id"]): row for row in repair_candidates}
    selected = [
        {
            **by_id[value],
            "window_id": f"directed-{int(by_id[value]['consolidation_span']) + 1}",
            "window_kind": "directed",
            "window_value": int(by_id[value]["consolidation_span"]),
            "writer_fit_mode": "frozen-global",
            "rho": 0.0,
            "no_rewrite_rho": 0.5,
            "seed_count": 1,
            "seed_mode": "single",
            "ecc": "none",
        }
        for value in TRANSIENT_IDS
    ]
    configuration = _configuration_payload(frozen["stage4"]["configuration"])
    items = [
        {
            "checkpoint": f"transient-p{pair_index:04d}-c{candidate_index:02d}",
            "pair": pair,
            "candidate_id": candidate["candidate_id"],
            "configuration": configuration,
            "replicates": profile.transient_replicates,
            "generations": 6,
            "conditions": TRANSIENT_CONDITIONS,
        }
        for pair_index, pair in enumerate(cohorts["transient"])
        for candidate_index, candidate in enumerate(selected)
    ]
    status("running", "transient-checkpoints", completed=0, total=len(items))
    rows, complete = _run_json_checkpoints(
        output,
        "transient-checkpoints",
        items,
        selected,
        _renewal_task,
        writer_contract,
        contract,
        frozen["stage4"]["reference"],
        design_digest,
        workers=workers,
        resume=resume,
        deadline=deadline,
        status=status,
    )
    if not complete:
        return {"state": "partial_budget_exhausted", "stage_gate": False}
    summaries: dict[str, Any] = {}
    positive: list[str] = []
    for candidate in selected:
        candidate_id = str(candidate["candidate_id"])
        summary = summarize_candidate(
            rows, candidate, 4, profile, contract, "transient"
        )
        controls = {
            condition: _loss_retention(rows, candidate_id, condition, 4)
            for condition in (
                "write_disabled",
                "transport_disabled",
                "zero_every_boundary",
                "shuffle_every_boundary",
            )
        }
        loss_pass = all(value <= 1.0 - contract.loss_fraction for value in controls.values())
        passed = candidate_gate(
            summary, contract, minimum=contract.transient_generation4
        ) and (loss_pass if scientific else True)
        summary.update(
            {
                "control_retention": controls,
                "targeted_loss_pass": loss_pass,
                "transient_gate": passed,
                "timecourse": {
                    str(generation): _boot(
                        _condition_values(
                            rows, candidate_id, "intact", generation
                        ),
                        profile,
                        contract,
                        "transient-timecourse",
                        candidate_id,
                        generation,
                    )
                    for generation in (1, 2, 4, 6)
                },
            }
        )
        summaries[candidate_id] = summary
        if passed:
            positive.append(candidate_id)
    result = {
        "state": "complete",
        "scientific_gate_applied": scientific,
        "stage_gate": bool(positive),
        "transient_positive_candidate_ids": positive,
        "candidate_summaries": summaries,
        "final_audit_trajectory_state": "untouched",
    }
    _write_phase_outputs(
        output,
        "transient",
        design_digest,
        result,
        "# Stage 6B-R transient local heredity\n\n"
        f"Transient-positive candidates: {positive or 'none'}.\n",
        "# Lay summary\n\n"
        f"{len(positive)} local carriers retained a causally renewed family signal "
        "through four generations. The durability repair continues independently.\n",
        next_phase="repair",
    )
    return result


def _fit_geometry_models(
    output: Path,
    frozen: dict[str, Any],
    cohorts: dict[str, list[dict[str, Any]]],
    candidates: Sequence[dict[str, Any]],
    profile: RenewalProfile,
    contract: RenewalContract,
    writer_contract: MotifContract,
    design_digest: str,
    *,
    workers: int,
    resume: bool,
    deadline: float,
    status,
) -> tuple[dict[tuple[str, str], tuple[np.ndarray, np.ndarray]], list[dict[str, Any]]]:
    by_window: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        by_window.setdefault(str(candidate["window_id"]), candidate)
    configuration = _configuration_payload(frozen["stage4"]["configuration"])
    models: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    audits: list[dict[str, Any]] = []
    for window_id, base in sorted(by_window.items()):
        models[(window_id, "frozen-global")] = (
            np.asarray(base["weight"], dtype=np.float32),
            np.asarray(base["bias"], dtype=np.float32),
        )
    one_step_candidates = [
        {
            **base,
            "candidate_id": f"fit-one-step-{window_id}",
            "writer_fit_mode": "frozen-global",
            "rho": 0.0,
        }
        for window_id, base in sorted(by_window.items())
    ]
    one_step_items = _checkpoint_items(
        cohorts["writer_train"],
        one_step_candidates,
        configuration,
        prefix="one-step",
        replicates=profile.writer_train_replicates,
        generations=1,
        conditions=("founder_clamped",),
        collect_training=True,
    )
    status("running", "writer-one-step", completed=0, total=len(one_step_items))
    rows, complete = _run_json_checkpoints(
        output,
        "writer-one-step",
        one_step_items,
        one_step_candidates,
        _renewal_task,
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
        raise TimeoutError("writer one-step fitting reached the campaign deadline")
    current: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for candidate in one_step_candidates:
        candidate_id = str(candidate["candidate_id"])
        selected_rows = [row for row in rows if row["candidate_id"] == candidate_id]
        moments, targets = _collect_fit_arrays(selected_rows, "founder_clamped")
        weight, bias, audit = fit_shared_writer(moments, targets)
        window_id = str(candidate["window_id"])
        models[(window_id, "local-one-step")] = (weight, bias)
        current[window_id] = (weight, bias)
        audits.append(
            {
                "window_id": window_id,
                "fit_mode": "local-one-step",
                "round": 0,
                "label_and_outcome_blind_runtime": True,
                **audit,
            }
        )
    for round_index in range(1, profile.rollout_refits + 1):
        rollout_candidates: list[dict[str, Any]] = []
        for window_id, base in sorted(by_window.items()):
            weight, bias = current[window_id]
            rollout_candidates.append(
                {
                    **base,
                    "candidate_id": f"fit-rollout-{round_index}-{window_id}",
                    "writer_fit_mode": "local-rollout",
                    "rho": 0.50,
                    "weight": weight,
                    "bias": bias,
                }
            )
        phase_name = f"writer-rollout-{round_index}"
        items = _checkpoint_items(
            cohorts["writer_train"],
            rollout_candidates,
            configuration,
            prefix=f"rollout-{round_index}",
            replicates=profile.writer_train_replicates,
            generations=4,
            conditions=("intact",),
            collect_training=True,
        )
        status("running", phase_name, completed=0, total=len(items))
        rollout_rows, complete = _run_json_checkpoints(
            output,
            phase_name,
            items,
            rollout_candidates,
            _renewal_task,
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
            raise TimeoutError("closed-loop writer fitting reached the campaign deadline")
        for candidate in rollout_candidates:
            candidate_id = str(candidate["candidate_id"])
            selected_rows = [
                row for row in rollout_rows if row["candidate_id"] == candidate_id
            ]
            moments, targets = _collect_fit_arrays(selected_rows, "intact")
            weight, bias, audit = fit_shared_writer(moments, targets)
            window_id = str(candidate["window_id"])
            current[window_id] = (weight, bias)
            audits.append(
                {
                    "window_id": window_id,
                    "fit_mode": "local-rollout",
                    "round": round_index,
                    "label_and_outcome_blind_runtime": True,
                    **audit,
                }
            )
    for window_id, value in current.items():
        models[(window_id, "local-rollout")] = value
    return models, audits


def _full_anchor(frozen: dict[str, Any]) -> dict[str, Any]:
    repair, _ = build_repair_candidates(frozen)
    base = next(row for row in repair if row["candidate_id"] == FULL_BRIDGE_ID)
    return {
        **base,
        "candidate_id": FULL_ANCHOR_ID,
        "window_id": "directed-16",
        "window_kind": "directed",
        "window_value": 15,
        "writer_fit_mode": "frozen-global",
        "rho": 0.0,
        "no_rewrite_rho": 0.5,
        "seed_count": 1,
        "seed_mode": "single",
        "ecc": "none",
        "promotable": False,
        "bounded": False,
    }


def _run_repair(
    output: Path,
    frozen: dict[str, Any],
    cohorts: dict[str, list[dict[str, Any]]],
    profile: RenewalProfile,
    contract: RenewalContract,
    writer_contract: MotifContract,
    design_digest: str,
    *,
    workers: int,
    resume: bool,
    deadline: float,
    status,
    scientific: bool,
) -> dict[str, Any]:
    candidates = build_renewal_candidates(frozen)
    try:
        models, fit_audits = _fit_geometry_models(
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
            deadline=deadline,
            status=status,
        )
    except TimeoutError:
        return {"state": "partial_budget_exhausted", "stage_gate": False}
    fitted: list[dict[str, Any]] = []
    for candidate in candidates:
        weight, bias = models[
            (str(candidate["window_id"]), str(candidate["writer_fit_mode"]))
        ]
        fitted.append({**candidate, "weight": weight, "bias": bias})
    configuration = _configuration_payload(frozen["stage4"]["configuration"])
    calibration_items = _checkpoint_items(
        cohorts["repair_calibration"],
        fitted,
        configuration,
        prefix="repair-calibration",
        replicates=profile.repair_calibration_replicates,
        generations=4,
        conditions=(
            "intact",
            "founder_clamped",
            "no_rewrite",
            "translated_patch",
        ),
    )
    status(
        "running",
        "repair-calibration",
        completed=0,
        total=len(calibration_items),
    )
    calibration_rows, complete = _run_json_checkpoints(
        output,
        "repair-calibration",
        calibration_items,
        fitted,
        _renewal_task,
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
    calibration_summaries = {
        str(candidate["candidate_id"]): summarize_candidate(
            calibration_rows,
            candidate,
            4,
            profile,
            contract,
            "repair-calibration",
        )
        for candidate in fitted
    }
    nominees = _select_diverse(calibration_summaries, fitted, 6)
    validation_candidates: list[dict[str, Any]] = []
    for candidate in nominees:
        for ecc in ("none", "hamming84"):
            suffix = "raw" if ecc == "none" else "h84"
            validation_candidates.append(
                {
                    **candidate,
                    "candidate_id": f"{candidate['candidate_id']}-e{suffix}",
                    "ecc": ecc,
                }
            )
    anchor = _full_anchor(frozen)
    validation_with_anchor = [anchor, *validation_candidates]
    validation_items = _checkpoint_items(
        cohorts["repair_validation"],
        validation_with_anchor,
        configuration,
        prefix="repair-validation",
        replicates=profile.repair_validation_replicates,
        generations=8,
        conditions=SCREEN_CONDITIONS,
    )
    status(
        "running", "repair-validation", completed=0, total=len(validation_items)
    )
    validation_rows, complete = _run_json_checkpoints(
        output,
        "repair-validation",
        validation_items,
        validation_with_anchor,
        _renewal_task,
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
    summaries = {
        str(candidate["candidate_id"]): summarize_candidate(
            validation_rows,
            candidate,
            8,
            profile,
            contract,
            "repair-validation",
        )
        for candidate in validation_with_anchor
    }
    anchor_mean = float(summaries[FULL_ANCHOR_ID]["crossover"]["mean"] or 0.0)
    eligible: list[dict[str, Any]] = []
    for candidate in validation_candidates:
        summary = summaries[str(candidate["candidate_id"])]
        passed = candidate_gate(
            summary,
            contract,
            minimum=contract.screen_generation8,
            anchor_mean=anchor_mean,
        )
        summary["repair_gate"] = passed
        if passed:
            eligible.append(candidate)
    ranked_source = eligible or validation_candidates
    promoted = _select_diverse(summaries, ranked_source, min(3, len(ranked_source)))
    write_models(
        output,
        [anchor, *validation_candidates],
        fit_audits,
        design_digest,
    )
    result = {
        "state": "complete",
        "scientific_gate_applied": scientific,
        "stage_gate": bool(eligible),
        "anchor_id": FULL_ANCHOR_ID,
        "anchor_generation8_crossover_mean": anchor_mean,
        "calibration_nominee_ids": [str(row["candidate_id"]) for row in nominees],
        "eligible_candidate_ids": [str(row["candidate_id"]) for row in eligible],
        "promoted_candidate_ids": [str(row["candidate_id"]) for row in promoted],
        "promotion_is_diagnostic_fallback": not bool(eligible),
        "candidate_summaries": summaries,
        "fit_audits": fit_audits,
        "final_audit_trajectory_state": "untouched",
    }
    _write_phase_outputs(
        output,
        "repair",
        design_digest,
        result,
        "# Stage 6B-R renewal repair\n\n"
        f"Strict repair-positive candidates: {result['eligible_candidate_ids'] or 'none'}. "
        f"Coverage nominees: {result['promoted_candidate_ids']}.\n",
        "# Lay summary\n\n"
        "The local writer was centred, trained on its own repeated outputs, and "
        "combined with a slower inherited latch. The next stage maps how much "
        "coverage the best mechanisms require.\n",
        next_phase="coverage",
    )
    return result


def build_coverage_variants(
    promoted: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    for parent in promoted:
        parent_id = str(parent["candidate_id"])
        for radius in RADII:
            for seed_count, seed_mode in SEED_LAYOUTS:
                variants.append(
                    {
                        **parent,
                        "candidate_id": (
                            f"coverage-{parent_id}-h{radius:02d}-"
                            f"s{seed_count:02d}-{seed_mode}"
                        ),
                        "source_candidate_id": parent_id,
                        "germination_hops": radius,
                        "seed_count": seed_count,
                        "seed_mode": seed_mode,
                        "promotable": radius < 8,
                        "bounded": radius < 8,
                    }
                )
    return variants


def _select_coverage_diverse(
    summaries: dict[str, dict[str, Any]],
    candidates: Sequence[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    ranked = sorted(
        candidates,
        key=lambda row: (
            -float(summaries[str(row["candidate_id"])]["crossover"]["mean"] or -1),
            int(row["seed_count"]),
            int(row["germination_hops"]),
            str(row["candidate_id"]),
        ),
    )
    selected: list[dict[str, Any]] = []
    seen_layout: set[tuple[int, str]] = set()
    seen_radius: set[int] = set()
    for candidate in ranked:
        layout = (int(candidate["seed_count"]), str(candidate["seed_mode"]))
        radius = int(candidate["germination_hops"])
        if layout not in seen_layout or radius not in seen_radius:
            selected.append(candidate)
            seen_layout.add(layout)
            seen_radius.add(radius)
        if len(selected) == limit:
            return selected
    selected_ids = {str(row["candidate_id"]) for row in selected}
    for candidate in ranked:
        if str(candidate["candidate_id"]) not in selected_ids:
            selected.append(candidate)
            selected_ids.add(str(candidate["candidate_id"]))
        if len(selected) == limit:
            break
    return selected


def _coverage_items(
    cohort: Sequence[dict[str, Any]],
    candidates: Sequence[dict[str, Any]],
    configuration: dict[str, Any],
    *,
    prefix: str,
    replicates: int,
    generations: int,
    validation: bool,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for pair_index, pair in enumerate(cohort):
        for candidate_index, candidate in enumerate(candidates):
            conditions = list(SCREEN_CONDITIONS) if validation else [
                "intact", "no_rewrite", "translated_patch"
            ]
            if int(candidate.get("seed_count", 1)) > 1:
                conditions.extend(("seed_ablation_25", "seed_ablation_50"))
            items.append(
                {
                    "checkpoint": (
                        f"{prefix}-p{pair_index:04d}-c{candidate_index:03d}"
                    ),
                    "pair": pair,
                    "candidate_id": candidate["candidate_id"],
                    "configuration": configuration,
                    "replicates": replicates,
                    "generations": generations,
                    "conditions": tuple(conditions),
                    "extent": 16,
                }
            )
    return items


def _run_coverage(
    output: Path,
    frozen: dict[str, Any],
    cohorts: dict[str, list[dict[str, Any]]],
    profile: RenewalProfile,
    contract: RenewalContract,
    writer_contract: MotifContract,
    design_digest: str,
    *,
    workers: int,
    resume: bool,
    deadline: float,
    status,
    scientific: bool,
) -> dict[str, Any]:
    repair_result = _load_json(output / "repair/RESULTS.json")
    models = load_models(output, design_digest)
    by_id = {str(row["candidate_id"]): row for row in models}
    promoted = [
        by_id[value] for value in repair_result["promoted_candidate_ids"]
    ]
    variants = build_coverage_variants(promoted)
    configuration = _configuration_payload(frozen["stage4"]["configuration"])
    pilot_cohort = cohorts["coverage"][: profile.coverage_pilot_pairs]
    pilot_items = _coverage_items(
        pilot_cohort,
        variants,
        configuration,
        prefix="coverage-pilot",
        replicates=max(2, profile.coverage_replicates // 2),
        generations=4,
        validation=False,
    )
    status("running", "coverage-pilot", completed=0, total=len(pilot_items))
    pilot_rows, complete = _run_json_checkpoints(
        output,
        "coverage-pilot",
        pilot_items,
        variants,
        _renewal_task,
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
    pilot_summaries = {
        str(candidate["candidate_id"]): summarize_candidate(
            pilot_rows,
            candidate,
            4,
            profile,
            contract,
            "coverage-pilot",
        )
        for candidate in variants
    }
    nominees = _select_coverage_diverse(pilot_summaries, variants, 12)
    anchor = by_id[FULL_ANCHOR_ID]
    validation_candidates = [anchor, *nominees]
    validation_items = _coverage_items(
        cohorts["coverage"],
        validation_candidates,
        configuration,
        prefix="coverage-validation",
        replicates=profile.coverage_replicates,
        generations=8,
        validation=True,
    )
    status(
        "running",
        "coverage-validation",
        completed=0,
        total=len(validation_items),
    )
    rows, complete = _run_json_checkpoints(
        output,
        "coverage-validation",
        validation_items,
        validation_candidates,
        _renewal_task,
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
    summaries = {
        str(candidate["candidate_id"]): summarize_candidate(
            rows, candidate, 8, profile, contract, "coverage-validation"
        )
        for candidate in validation_candidates
    }
    anchor_mean = float(summaries[FULL_ANCHOR_ID]["crossover"]["mean"] or 0.0)
    eligible: list[dict[str, Any]] = []
    for candidate in nominees:
        candidate_id = str(candidate["candidate_id"])
        summary = summaries[candidate_id]
        passed = candidate_gate(
            summary,
            contract,
            minimum=contract.screen_generation8,
            anchor_mean=anchor_mean,
        )
        ablation: dict[str, float] = {}
        if int(candidate["seed_count"]) > 1:
            for condition in ("seed_ablation_25", "seed_ablation_50"):
                ablation[condition] = _loss_retention(
                    rows, candidate_id, condition, 8
                )
        summary["seed_ablation_retention"] = ablation
        summary["coverage_gate"] = passed
        if passed:
            eligible.append(candidate)
    ranked = sorted(
        eligible or nominees,
        key=lambda row: -float(
            summaries[str(row["candidate_id"])]["crossover"]["mean"] or -1
        ),
    )
    fixed = next(
        (
            row
            for row in ranked
            if int(row["seed_count"]) == 1
            and int(row["germination_hops"]) < 8
        ),
        None,
    )
    distributed = next(
        (row for row in ranked if int(row["seed_count"]) > 1), None
    )
    selected = [row for row in (fixed, distributed) if row is not None]
    result = {
        "state": "complete",
        "scientific_gate_applied": scientific,
        "stage_gate": bool(eligible),
        "anchor_id": FULL_ANCHOR_ID,
        "anchor_generation8_crossover_mean": anchor_mean,
        "pilot_nominee_ids": [str(row["candidate_id"]) for row in nominees],
        "eligible_candidate_ids": [str(row["candidate_id"]) for row in eligible],
        "selected_candidate_ids": [str(row["candidate_id"]) for row in selected],
        "selection_is_diagnostic_fallback": not bool(eligible),
        "candidate_summaries": summaries,
        "final_audit_trajectory_state": "untouched",
    }
    _write_phase_outputs(
        output,
        "coverage",
        design_digest,
        result,
        "# Stage 6B-R coverage and distributed carriers\n\n"
        f"Strict coverage-positive candidates: {result['eligible_candidate_ids'] or 'none'}. "
        f"Scale nominees: {result['selected_candidate_ids']}.\n",
        "# Lay summary\n\n"
        "This stage tests whether the memory needs a nearly whole-lattice wave, "
        "or whether several genuinely local seeds can cooperate without increasing "
        "the total inherited message.\n",
        next_phase="scale",
    )
    return result


def _rebuild_coverage_candidates(
    output: Path, design_digest: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    repair_result = _load_json(output / "repair/RESULTS.json")
    models = load_models(output, design_digest)
    by_id = {str(row["candidate_id"]): row for row in models}
    promoted = [by_id[value] for value in repair_result["promoted_candidate_ids"]]
    variants = build_coverage_variants(promoted)
    return variants, by_id[FULL_ANCHOR_ID]


def build_scale_variants(
    sources: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    for source in sources:
        source_id = str(source["candidate_id"])
        for extent in (16, 32, 64):
            variants.append(
                {
                    **source,
                    "candidate_id": f"scale-{source_id}-n{extent:02d}-fixed",
                    "source_candidate_id": source_id,
                    "scale_mode": "fixed",
                    "extent": extent,
                }
            )
            variants.append(
                {
                    **source,
                    "candidate_id": f"scale-{source_id}-n{extent:02d}-diameter",
                    "source_candidate_id": source_id,
                    "scale_mode": "diameter",
                    "extent": extent,
                    "germination_hops": extent // 2,
                    "window_id": f"directed-{extent}",
                    "window_kind": "directed",
                    "window_value": extent - 1,
                    "consolidation_span": extent - 1,
                    "consolidation_steps": 2 * (extent - 1),
                    "seed_count": 1,
                    "seed_mode": "single",
                    "bounded": False,
                }
            )
            if int(source.get("seed_count", 1)) > 1:
                factor = (extent // 16) ** 2
                variants.append(
                    {
                        **source,
                        "candidate_id": f"scale-{source_id}-n{extent:02d}-density",
                        "source_candidate_id": source_id,
                        "scale_mode": "density",
                        "extent": extent,
                        "seed_count": int(source["seed_count"]) * factor,
                    }
                )
    return variants


def _band_crossover(
    rows: Sequence[dict[str, Any]],
    candidate_id: str,
    generation: int,
    hops: int,
) -> list[float]:
    keys = ["0-2"]
    if hops >= 3:
        keys.append("3-5")
    if hops >= 6:
        keys.append("6-8")
    if hops >= 9:
        keys.append("9+")
    values: list[float] = []
    for row in rows:
        if row.get("candidate_id") != candidate_id:
            continue
        try:
            bands = row["conditions"]["intact"]["outcomes"][str(generation)][
                "distance_bands"
            ]
        except KeyError:
            continue
        if "inside-cone" in bands:
            values.append(float(bands["inside-cone"]["crossover"]))
        else:
            present = [float(bands[key]["crossover"]) for key in keys]
            values.append(float(np.mean(present)))
    return values


def summarize_scale(
    rows: Sequence[dict[str, Any]],
    variants: Sequence[dict[str, Any]],
    profile: RenewalProfile,
    contract: RenewalContract,
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for candidate in variants:
        candidate_id = str(candidate["candidate_id"])
        global_values = _condition_values(rows, candidate_id, "intact", 8)
        in_cone = _band_crossover(
            rows,
            candidate_id,
            8,
            int(candidate["germination_hops"]),
        )
        outside_values = _diagnostic_values_stage6br(
            rows,
            candidate_id,
            "intact",
            8,
            "carrier_history",
            "outside_light_cone_occupied_fraction",
        )
        extent = int(candidate["extent"])
        rank = int(candidate["rank"])
        seed_count = int(candidate.get("seed_count", 1))
        physical_bits = int(candidate["payload_bits"])
        if candidate.get("seed_mode") == "replicated":
            physical_bits *= seed_count
        if candidate.get("ecc") == "hamming84":
            physical_bits *= 2
        summaries[candidate_id] = {
            "candidate": candidate_public(candidate),
            "global_crossover": _boot(
                global_values, profile, contract, "scale", candidate_id, "global"
            ),
            "in_cone_crossover": _boot(
                in_cone, profile, contract, "scale", candidate_id, "in-cone"
            ),
            "coverage_mean": _diagnostic_mean(
                rows,
                candidate_id,
                "intact",
                8,
                "carrier_history",
                "occupied_fraction_after_germination",
            ),
            "outside_light_cone_occupied_fraction_max": (
                max(outside_values) if outside_values else None
            ),
            "resource_accounting": {
                "logical_inherited_bits": int(candidate["payload_bits"]),
                "physical_inherited_bits": physical_bits,
                "temporary_field_values": extent * extent * rank,
                "temporary_writer_values": extent * extent * rank,
                "routing_site_channel_updates_upper_bound": (
                    extent
                    * extent
                    * rank
                    * (
                        int(candidate["germination_hops"])
                        + 2 * int(candidate["window_value"])
                    )
                ),
                "seed_count": seed_count,
                "seed_density": seed_count / float(extent * extent),
                "shared_writer_parameter_bits": int(
                    (
                        np.asarray(candidate["weight"]).size
                        + np.asarray(candidate["bias"]).size
                    )
                    * 32
                ),
            },
        }
    source_ids = sorted({str(row["source_candidate_id"]) for row in variants})
    retention: dict[str, Any] = {}
    class_pass: dict[str, dict[str, bool]] = {}
    for source_id in source_ids:
        retention[source_id] = {}
        class_pass[source_id] = {}
        modes = sorted(
            {
                str(row["scale_mode"])
                for row in variants
                if row["source_candidate_id"] == source_id
            }
        )
        for mode in modes:
            baseline_id = f"scale-{source_id}-n16-{mode}"
            key = "in_cone_crossover" if mode == "fixed" else "global_crossover"
            baseline = float(summaries[baseline_id][key]["mean"] or 0.0)
            values: dict[str, float] = {}
            for extent in (16, 32, 64):
                candidate_id = f"scale-{source_id}-n{extent:02d}-{mode}"
                mean = float(summaries[candidate_id][key]["mean"] or 0.0)
                values[str(extent)] = mean / baseline if baseline > 0.0 else 0.0
            retention[source_id][mode] = values
            retention_pass = bool(
                values["32"] >= contract.anchor_retention
                and values["64"] >= contract.anchor_retention
            )
            exact_support = {
                str(extent): (
                    summaries[f"scale-{source_id}-n{extent:02d}-{mode}"][
                        "outside_light_cone_occupied_fraction_max"
                    ]
                    is not None
                    and float(
                        summaries[f"scale-{source_id}-n{extent:02d}-{mode}"][
                            "outside_light_cone_occupied_fraction_max"
                        ]
                    )
                    == 0.0
                )
                for extent in (16, 32, 64)
            }
            retention[source_id][mode] = {
                "relative_recovery": values,
                "retention_pass": retention_pass,
                "exact_light_cone_support": exact_support,
            }
            class_pass[source_id][mode] = bool(
                retention_pass
                and (mode != "fixed" or all(exact_support.values()))
            )
    return {
        "state": "complete",
        "candidate_summaries": summaries,
        "retention_relative_to_n16": retention,
        "class_pass": class_pass,
        "fixed_budget_exact_light_cone_pass": all(
            all(
                value["fixed"]["exact_light_cone_support"].values()
            )
            for value in retention.values()
            if "fixed" in value
        ),
        "nearest_neighbour_only": True,
    }


def _run_scale(
    output: Path,
    frozen: dict[str, Any],
    cohorts: dict[str, list[dict[str, Any]]],
    profile: RenewalProfile,
    contract: RenewalContract,
    writer_contract: MotifContract,
    design_digest: str,
    *,
    workers: int,
    resume: bool,
    deadline: float,
    status,
    scientific: bool,
) -> dict[str, Any]:
    coverage_result = _load_json(output / "coverage/RESULTS.json")
    variants, anchor = _rebuild_coverage_candidates(output, design_digest)
    by_id = {str(row["candidate_id"]): row for row in variants}
    selected = [
        by_id[value] for value in coverage_result["selected_candidate_ids"]
    ]
    sources = [anchor, *selected]
    scale_variants = build_scale_variants(sources)
    configuration = _configuration_payload(frozen["stage4"]["configuration"])
    items: list[dict[str, Any]] = []
    for pair_index, pair in enumerate(cohorts["scale"]):
        for candidate_index, candidate in enumerate(scale_variants):
            items.append(
                {
                    "checkpoint": (
                        f"scale-p{pair_index:04d}-c{candidate_index:03d}"
                    ),
                    "pair": pair,
                    "candidate_id": candidate["candidate_id"],
                    "configuration": configuration,
                    "replicates": profile.scale_replicates,
                    "generations": 8,
                    "conditions": (
                        "intact",
                        "no_rewrite",
                        "transport_disabled",
                        "translated_patch",
                    ),
                    "extent": int(candidate["extent"]),
                }
            )
    status("running", "scale-checkpoints", completed=0, total=len(items))
    rows, complete = _run_json_checkpoints(
        output,
        "scale-checkpoints",
        items,
        scale_variants,
        _renewal_task,
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
    result = summarize_scale(rows, scale_variants, profile, contract)
    passing_sources: list[str] = []
    for source in selected:
        source_id = str(source["candidate_id"])
        if any(result["class_pass"][source_id].values()):
            passing_sources.append(source_id)
    result.update(
        {
            "scientific_gate_applied": scientific,
            "stage_gate": bool(passing_sources),
            "passing_source_candidate_ids": passing_sources,
            "tested_source_candidate_ids": [
                str(row["candidate_id"]) for row in sources
            ],
            "final_audit_trajectory_state": "untouched",
        }
    )
    _write_phase_outputs(
        output,
        "scale",
        design_digest,
        result,
        "# Stage 6B-R scale and causal geometry\n\n"
        f"Passing local-interaction sources: {passing_sources or 'none'}.\n",
        "# Lay summary\n\n"
        "The same carriers were placed in larger worlds. Fixed reach, signalling "
        "time that grows with body size, and constant seed density are reported "
        "as separate mechanisms rather than one pooled claim.\n",
        next_phase="adjudicate",
    )
    return result


def _causal_gate(
    rows: Sequence[dict[str, Any]],
    candidate_id: str,
    generation: int,
    contract: RenewalContract,
) -> tuple[bool, dict[str, Any]]:
    intact_values = _condition_values(rows, candidate_id, "intact", generation)
    intact = float(np.mean(intact_values)) if intact_values else 0.0
    loss_conditions = (
        "zero_every_boundary",
        "shuffle_every_boundary",
        "read_disabled",
        "founder_write_disabled",
        "write_disabled",
        "transport_disabled",
        "regeneration_disabled",
        "communication_cut",
    )
    retention = {
        condition: _loss_retention(rows, candidate_id, condition, generation)
        for condition in loss_conditions
    }
    loss_pass = all(
        value <= 1.0 - contract.loss_fraction for value in retention.values()
    )
    same_values = _condition_values(
        rows, candidate_id, "rescue_same_enter_g4", generation
    )
    opposite_values = _condition_values(
        rows, candidate_id, "rescue_opposite_enter_g4", generation
    )
    opposite_founder = _condition_values(
        rows, candidate_id, "opposite_founder", generation
    )
    same = float(np.mean(same_values)) if same_values else 0.0
    opposite = float(np.mean(opposite_values)) if opposite_values else 0.0
    founder_reversal = (
        float(np.mean(opposite_founder)) if opposite_founder else 0.0
    )
    rescue_pass = bool(
        intact > 0.0
        and same / intact >= contract.rescue_fraction
        and opposite < 0.0
        and founder_reversal < 0.0
    )
    return loss_pass and rescue_pass, {
        "targeted_control_retention": retention,
        "targeted_loss_pass": loss_pass,
        "same_rescue_retention": same / intact if intact > 0.0 else 0.0,
        "opposite_rescue_mean": opposite,
        "opposite_founder_mean": founder_reversal,
        "rescue_and_reversal_pass": rescue_pass,
    }


def _qualification_candidates(
    output: Path, design_digest: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    repair = _load_json(output / "repair/RESULTS.json")
    coverage = _load_json(output / "coverage/RESULTS.json")
    scale = _load_json(output / "scale/RESULTS.json")
    variants, anchor = _rebuild_coverage_candidates(output, design_digest)
    by_id = {str(row["candidate_id"]): row for row in variants}
    coverage_positive = set(coverage["eligible_candidate_ids"])
    scale_positive = set(scale["passing_source_candidate_ids"])
    repair_positive = set(repair["eligible_candidate_ids"])
    ids = sorted(
        candidate_id
        for candidate_id in coverage_positive & scale_positive
        if str(by_id[candidate_id]["source_candidate_id"]) in repair_positive
    )
    ranked = sorted(
        ids,
        key=lambda candidate_id: -float(
            coverage["candidate_summaries"][candidate_id]["crossover"]["mean"]
            or -1
        ),
    )
    return [by_id[value] for value in ranked[:2]], anchor


def _final_design_payload(
    candidates: Sequence[dict[str, Any]],
    anchor: dict[str, Any],
    frozen: dict[str, Any],
    profile: RenewalProfile,
    design_digest: str,
    output: Path,
) -> dict[str, Any]:
    ids = [str(row["candidate_id"]) for row in candidates]
    candidate_order = [*ids, str(anchor["candidate_id"])]
    payload = {
        "experiment": "ca_motif_lineage_stage_6br_final_audit",
        "parent_design_digest": design_digest,
        "candidate_ids": ids,
        "anchor_id": str(anchor["candidate_id"]),
        "candidate_order": candidate_order,
        "candidate_public": [candidate_public(row) for row in [*candidates, anchor]],
        "conditions": FULL_CONDITIONS,
        "generations": 32,
        "replicates_per_history": profile.final_replicates,
        "pair_count": profile.final_pairs,
        "pair_ids_sha256": hashlib.sha256(
            "\n".join(frozen["later_ids"][: profile.final_pairs]).encode()
        ).hexdigest(),
        "model_metadata_sha256": _sha256(output / "RENEWAL_MODELS.json"),
        "model_arrays_sha256": _sha256(output / "RENEWAL_MODELS.npz"),
        "automatic_opening_authorized": True,
        "retuning_after_seal": False,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return payload | {"final_audit_digest": digest}


def _run_adjudicate(
    output: Path,
    frozen: dict[str, Any],
    cohorts: dict[str, list[dict[str, Any]]],
    profile: RenewalProfile,
    profile_name: str,
    contract: RenewalContract,
    writer_contract: MotifContract,
    design_digest: str,
    *,
    workers: int,
    resume: bool,
    deadline: float,
    status,
    scientific: bool,
    auto_final_audit: bool,
) -> dict[str, Any]:
    candidates, anchor = _qualification_candidates(output, design_digest)
    transient = _load_json(output / "transient/RESULTS.json")
    repair = _load_json(output / "repair/RESULTS.json")
    configuration = _configuration_payload(frozen["stage4"]["configuration"])
    if not candidates:
        anchor_pass = bool(
            float(repair["anchor_generation8_crossover_mean"] or 0.0) > 0.0
        )
        verdict = (
            "TRANSIENT_LOCAL_PH_ONLY"
            if transient["stage_gate"]
            else ("WHOLE_LATTICE_COORDINATION_ONLY" if anchor_pass else "NO_CORRECTED_PH")
        )
        result = {
            "state": "complete",
            "stage_gate": False,
            "qualification_candidate_ids": [],
            "qualified_candidate_ids": [],
            "enduring_candidate_ids": [],
            "final_audit_opened": False,
            "final_audit_trajectory_state": "untouched",
            "verdict": verdict,
        }
        _write_phase_outputs(
            output,
            "adjudicate",
            design_digest,
            result,
            "# Stage 6B-R adjudication\n\n"
            f"Verdict: **{verdict}**. No local candidate earned qualification, "
            "so the final reserve remains untouched.\n",
            "# Lay summary\n\n"
            "No repaired local mechanism passed both durability and scaling. "
            "The protected final cases were therefore not opened.\n",
            next_phase=None,
        )
        return result
    qualification_models = [anchor, *candidates]
    qualification_items = _checkpoint_items(
        cohorts["qualification"],
        qualification_models,
        configuration,
        prefix="qualification",
        replicates=profile.qualification_replicates,
        generations=16,
        conditions=FULL_CONDITIONS,
    )
    status(
        "running",
        "qualification",
        completed=0,
        total=len(qualification_items),
    )
    qualification_rows, complete = _run_json_checkpoints(
        output,
        "qualification",
        qualification_items,
        qualification_models,
        _renewal_task,
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
    summaries = {
        str(candidate["candidate_id"]): summarize_candidate(
            qualification_rows,
            candidate,
            16,
            profile,
            contract,
            "qualification",
        )
        for candidate in qualification_models
    }
    anchor_mean = float(summaries[FULL_ANCHOR_ID]["crossover"]["mean"] or 0.0)
    qualified: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        causal_pass, causal = _causal_gate(
            qualification_rows, candidate_id, 16, contract
        )
        summary = summaries[candidate_id]
        recovery_pass = candidate_gate(
            summary,
            contract,
            minimum=contract.qualification_generation16,
            anchor_mean=anchor_mean,
        )
        finite = bool(
            int(candidate["germination_hops"]) < 8
            and int(candidate["consolidation_steps"]) <= 14
        )
        passed = recovery_pass and causal_pass and finite
        summary.update(
            causal
            | {
                "finite_causal_gate": finite,
                "qualification_gate": passed,
            }
        )
        if passed:
            qualified.append(candidate)
    _atomic_json(
        output / "adjudicate/QUALIFICATION.json",
        {
            "design_digest": design_digest,
            "candidate_summaries": summaries,
            "qualified_candidate_ids": [
                str(row["candidate_id"]) for row in qualified
            ],
        },
    )
    if not qualified:
        verdict = (
            "TRANSIENT_LOCAL_PH_ONLY"
            if transient["stage_gate"]
            else "WHOLE_LATTICE_COORDINATION_ONLY"
        )
        result = {
            "state": "complete",
            "stage_gate": False,
            "qualification_candidate_ids": [
                str(row["candidate_id"]) for row in candidates
            ],
            "qualified_candidate_ids": [],
            "enduring_candidate_ids": [],
            "qualification": summaries,
            "final_audit_opened": False,
            "final_audit_trajectory_state": "untouched",
            "verdict": verdict,
        }
        _write_phase_outputs(
            output,
            "adjudicate",
            design_digest,
            result,
            "# Stage 6B-R adjudication\n\n"
            f"Verdict: **{verdict}**. Strict qualification failed.\n",
            "# Lay summary\n\n"
            "The best developmental mechanisms did not survive the complete "
            "causal ladder, so the protected reserve was not opened.\n",
            next_phase=None,
        )
        return result
    endurance_items = _checkpoint_items(
        cohorts["endurance"],
        qualified,
        configuration,
        prefix="endurance",
        replicates=profile.endurance_replicates,
        generations=64,
        conditions=(
            "intact",
            "no_rewrite",
            "carrier_corruption_1",
            "translated_patch",
        ),
    )
    status("running", "endurance", completed=0, total=len(endurance_items))
    endurance_rows, complete = _run_json_checkpoints(
        output,
        "endurance",
        endurance_items,
        qualified,
        _renewal_task,
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
    endurance_summaries: dict[str, Any] = {}
    enduring: list[dict[str, Any]] = []
    for candidate in qualified:
        candidate_id = str(candidate["candidate_id"])
        summary = summarize_candidate(
            endurance_rows,
            candidate,
            64,
            profile,
            contract,
            "endurance",
        )
        lower = summary["crossover"]["ci"][0]
        active_lower = summary["active_rewrite_advantage"]["ci"][0]
        passed = bool(
            lower is not None
            and float(lower) > 0.0
            and active_lower is not None
            and float(active_lower) > 0.0
            and summary["translation_retention"] >= contract.translation_retention
        )
        summary["endurance_gate"] = passed
        endurance_summaries[candidate_id] = summary
        if passed:
            enduring.append(candidate)
    _atomic_json(
        output / "adjudicate/ENDURANCE.json",
        {
            "design_digest": design_digest,
            "candidate_summaries": endurance_summaries,
            "enduring_candidate_ids": [
                str(row["candidate_id"]) for row in enduring
            ],
        },
    )
    if not enduring:
        verdict = (
            "TRANSIENT_LOCAL_PH_ONLY"
            if transient["stage_gate"]
            else "WHOLE_LATTICE_COORDINATION_ONLY"
        )
        result = {
            "state": "complete",
            "stage_gate": False,
            "qualified_candidate_ids": [
                str(row["candidate_id"]) for row in qualified
            ],
            "enduring_candidate_ids": [],
            "qualification": summaries,
            "endurance": endurance_summaries,
            "final_audit_opened": False,
            "final_audit_trajectory_state": "untouched",
            "verdict": verdict,
        }
        _write_phase_outputs(
            output,
            "adjudicate",
            design_digest,
            result,
            "# Stage 6B-R adjudication\n\n"
            f"Verdict: **{verdict}**. Generation-64 endurance failed.\n",
            "# Lay summary\n\n"
            "Some candidates passed qualification but lost renewable identity "
            "during the long lineage. The final reserve stayed closed.\n",
            next_phase=None,
        )
        return result
    if not auto_final_audit:
        raise ValueError("enduring candidates require --auto-final-audit authorization")
    enduring = enduring[:2]
    final_design = _final_design_payload(
        enduring, anchor, frozen, profile, design_digest, output
    )
    final_design_path = output / "FINAL_AUDIT_DESIGN.json"
    if final_design_path.exists():
        if _load_json(final_design_path) != final_design:
            raise ValueError("sealed Stage-6B-R final design changed")
    else:
        _atomic_json(final_design_path, final_design)
    opened_base = select_minimality_cohorts(
        MINIMALITY_PROFILES[profile_name],
        frozen,
        profile_name=profile_name,
        open_audit=True,
    )
    opened = select_renewal_cohorts(
        opened_base, frozen, profile, open_final=True
    )
    cohort_path = output / "COHORTS.json"
    cohort_payload = _load_json(cohort_path)
    cohort_payload.update(
        {
            "final_audit_trajectory_state": "opened",
            "final_audit_trajectories_not_loaded": False,
            "final_audit_opened_unix": time.time(),
            "final_audit_digest": final_design["final_audit_digest"],
        }
    )
    _atomic_json(cohort_path, cohort_payload)
    final_models = [*enduring, anchor]
    final_items = _checkpoint_items(
        opened["final"],
        final_models,
        configuration,
        prefix="final-audit",
        replicates=profile.final_replicates,
        generations=32,
        conditions=FULL_CONDITIONS,
    )
    status("running", "final-audit", completed=0, total=len(final_items))
    final_rows, complete = _run_json_checkpoints(
        output,
        "final-audit",
        final_items,
        final_models,
        _renewal_task,
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
        return {
            "state": "partial_budget_exhausted",
            "stage_gate": False,
            "final_audit_opened": True,
            "final_audit_trajectory_state": "opened",
        }
    final_summaries: dict[str, Any] = {}
    confirmed: list[dict[str, Any]] = []
    final_anchor = summarize_candidate(
        final_rows, anchor, 32, profile, contract, "final-audit"
    )
    final_anchor_mean = float(final_anchor["crossover"]["mean"] or 0.0)
    final_summaries[FULL_ANCHOR_ID] = final_anchor
    for candidate in enduring:
        candidate_id = str(candidate["candidate_id"])
        summary = summarize_candidate(
            final_rows, candidate, 32, profile, contract, "final-audit"
        )
        causal_pass, causal = _causal_gate(
            final_rows, candidate_id, 32, contract
        )
        passed = candidate_gate(
            summary,
            contract,
            minimum=contract.qualification_generation16,
            anchor_mean=final_anchor_mean,
        ) and causal_pass
        summary.update(causal | {"final_gate": passed})
        final_summaries[candidate_id] = summary
        if passed:
            confirmed.append(candidate)
    fixed_confirmed = any(int(row["seed_count"]) == 1 for row in confirmed)
    distributed_confirmed = any(int(row["seed_count"]) > 1 for row in confirmed)
    if fixed_confirmed:
        verdict = "CONFIRMED_DURABLE_FIXED_RADIUS_CA_PLASTIC_HEREDITY"
    elif distributed_confirmed:
        verdict = "CONFIRMED_DISTRIBUTED_LOCAL_INTERACTION_CA_PLASTIC_HEREDITY"
    elif transient["stage_gate"]:
        verdict = "TRANSIENT_LOCAL_PH_ONLY"
    else:
        verdict = "WHOLE_LATTICE_COORDINATION_ONLY"
    cohort_payload.update(
        {
            "final_audit_trajectory_state": "complete",
            "final_audit_completed_unix": time.time(),
        }
    )
    _atomic_json(cohort_path, cohort_payload)
    result = {
        "state": "complete",
        "stage_gate": bool(confirmed),
        "qualified_candidate_ids": [str(row["candidate_id"]) for row in qualified],
        "enduring_candidate_ids": [str(row["candidate_id"]) for row in enduring],
        "confirmed_candidate_ids": [str(row["candidate_id"]) for row in confirmed],
        "qualification": summaries,
        "endurance": endurance_summaries,
        "final_audit": final_summaries,
        "final_audit_opened": True,
        "final_audit_trajectory_state": "complete",
        "final_audit_digest": final_design["final_audit_digest"],
        "verdict": verdict,
    }
    _write_phase_outputs(
        output,
        "adjudicate",
        design_digest,
        result,
        "# Stage 6B-R final adjudication\n\n"
        f"Verdict: **{verdict}**. Confirmed candidates: "
        f"{result['confirmed_candidate_ids'] or 'none'}.\n",
        "# Lay summary\n\n"
        "Every surviving mechanism was frozen before the protected families "
        "were opened. The verdict distinguishes a small fixed-radius memory from "
        "a body-wide pattern built entirely from local interactions.\n",
        next_phase=None,
    )
    return result


def _prepare_stage6br(
    output: Path,
    profile_name: str,
    *,
    auto_final_audit: bool,
    stage6ar_root: Path,
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
    dict[str, list[dict[str, Any]]],
    RenewalProfile,
    RenewalContract,
    MotifContract,
    str,
]:
    """Freeze all developmental inputs without opening the final reserve."""

    profile = RENEWAL_PROFILES[profile_name]
    contract = RenewalContract()
    writer_contract = MotifContract()
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
    stage6ar_hashes = _stage6ar_hashes(stage6ar_root.resolve())
    base = select_minimality_cohorts(
        MINIMALITY_PROFILES[profile_name],
        frozen,
        profile_name=profile_name,
        open_audit=False,
    )
    cohorts = select_renewal_cohorts(
        base, frozen, profile, open_final=False
    )
    candidates = build_renewal_candidates(frozen)
    reserve_digest = hashlib.sha256(
        "\n".join(frozen["later_ids"]).encode()
    ).hexdigest()
    development_pair_ids = {
        name: [str(pair["pair_id"]) for pair in rows]
        for name, rows in cohorts.items()
        if name != "final"
    }
    design_payload = {
        "experiment": "ca_motif_lineage_stage_6br",
        "contract": contract.to_dict(),
        "writer_contract_digest": writer_contract.digest,
        "profile_name": profile_name,
        "profile": asdict(profile),
        "phases": PHASES,
        "automatic_phase_progression": True,
        "negative_developmental_phase_does_not_trigger_retuning": True,
        "automatic_final_audit_user_authorized": bool(auto_final_audit),
        "automatic_final_audit_requires_qualification_and_endurance": True,
        "stage5r_design_digest": frozen["design_digest"],
        "stage6a_design_digest": frozen6a["design"]["design_digest"],
        "frozen_stage6a_sha256": frozen6a["hashes"],
        "stage6ar_design_digest": _load_json(
            stage6ar_root.resolve() / "DESIGN.json"
        )["design_digest"],
        "frozen_stage6ar_sha256": stage6ar_hashes,
        "configuration": frozen["stage4"]["configuration"].to_dict(),
        "renewal_factorial": {
            "windows": WINDOWS,
            "fit_modes": FIT_MODES,
            "turnovers": TURNOVERS,
            "candidate_count": len(candidates),
            "candidate_ids": [str(row["candidate_id"]) for row in candidates],
        },
        "coverage_factorial": {
            "radii": RADII,
            "seed_layouts": SEED_LAYOUTS,
            "fixed_total_partition_for_partitioned_layouts": True,
        },
        "development_pair_ids": development_pair_ids,
        "final_audit_pair_count": len(frozen["later_ids"]),
        "final_audit_pair_ids_sha256": reserve_digest,
        "final_audit_trajectories_loaded": False,
        "input_sha256": {
            "protocol": _sha256(PROTOCOL_PATH),
            **{
                f"stage5r_{key}": _sha256(path)
                for key, path in frozen["paths"].items()
            },
        },
        "implementation_sha256": {
            "motif_renewal_repair.py": _sha256(Path(__file__)),
            "motif_minimality_repair.py": _sha256(
                Path(__file__).with_name("motif_minimality_repair.py")
            ),
            "motif_minimality.py": _sha256(
                Path(__file__).with_name("motif_minimality.py")
            ),
            "motif_lineage.py": _sha256(
                Path(__file__).with_name("motif_lineage.py")
            ),
        },
        "cleanroom_exclusion": (
            "no Wagner or Fable implementation source is read, imported, "
            "hashed, or executed"
        ),
        "claim_boundary": "engineered synthetic cellular-automaton heredity",
        "retuning_on_final_audit": False,
    }
    design_digest = hashlib.sha256(
        json.dumps(design_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    design = design_payload | {"design_digest": design_digest}
    design_path = output / "DESIGN.json"
    if design_path.exists():
        existing = _load_json(design_path)
        if existing.get("design_digest") != design_digest:
            raise ValueError("Stage-6B-R design changed; refusing a mixed resume")
    else:
        _atomic_json(design_path, design)

    cohort_path = output / "COHORTS.json"
    prior_cohorts = _load_json(cohort_path) if cohort_path.exists() else {}
    reserve_state = prior_cohorts.get(
        "final_audit_trajectory_state", "untouched"
    )
    _atomic_json(
        cohort_path,
        {
            "design_digest": design_digest,
            "development_pair_ids": development_pair_ids,
            "development_pair_count": sum(
                len(rows) for rows in development_pair_ids.values()
            ),
            "final_audit_pair_count": len(frozen["later_ids"]),
            "final_audit_pair_ids_sha256": reserve_digest,
            "final_audit_trajectory_state": reserve_state,
            "final_audit_trajectories_not_loaded": reserve_state == "untouched",
            **{
                key: value
                for key, value in prior_cohorts.items()
                if key.startswith("final_audit_")
                and key
                not in {
                    "final_audit_trajectory_state",
                    "final_audit_trajectories_not_loaded",
                }
            },
        },
    )
    _atomic_json(
        output / "CANDIDATES.json",
        {
            "design_digest": design_digest,
            "candidate_count": len(candidates),
            "candidates": [candidate_public(row) for row in candidates],
            "models_fitted_only_on_exposed_developmental_pairs": True,
        },
    )
    provenance_path = output / "CLEANROOM_PROVENANCE.json"
    provenance = {
        "experiment": "ca_motif_lineage_stage_6br",
        "design_digest": design_digest,
        "permitted_inputs": (
            "the published preprint, code-free documentation and result/data "
            "artifacts from prior clean-room CA stages"
        ),
        "excluded_inputs": (
            "Wagner implementation code and Fable implementation code"
        ),
        "excluded_code_read_import_hash_execute": True,
        "implementation_independent": True,
    }
    if provenance_path.exists() and _load_json(provenance_path) != provenance:
        raise ValueError("Stage-6B-R clean-room provenance changed")
    _atomic_json(provenance_path, provenance)
    return (
        frozen,
        cohorts,
        profile,
        contract,
        writer_contract,
        design_digest,
    )


def _campaign_clock(
    output: Path,
    design_digest: str,
    max_hours: float,
    *,
    resume: bool,
) -> dict[str, Any]:
    path = output / "CAMPAIGN.json"
    if path.exists():
        campaign = _load_json(path)
        if not resume:
            raise ValueError("Stage-6B-R output exists; use --resume")
        if campaign.get("design_digest") != design_digest:
            raise ValueError("Stage-6B-R campaign design mismatch")
        if not math.isclose(float(campaign["max_hours"]), max_hours):
            raise ValueError("Stage-6B-R max-hours changed during resume")
        return campaign
    started = time.time()
    hard_deadline = started + max_hours * 3600.0
    # Engineering profiles use a shortened clock but retain the registered
    # 6.25:1.75 exploration/adjudication split.  The reference profile at the
    # eight-hour ceiling therefore receives exactly the preregistered times.
    exploration_fraction = (
        RenewalContract().exploration_stop_hours
        / RenewalContract().max_hours
    )
    exploration_hours = max_hours * exploration_fraction
    campaign = {
        "experiment": "ca_motif_lineage_stage_6br",
        "design_digest": design_digest,
        "started_unix": started,
        "max_hours": max_hours,
        "hard_deadline_unix": hard_deadline,
        "exploration_deadline_unix": started + exploration_hours * 3600.0,
        "adjudication_reserve_hours": max_hours - exploration_hours,
        "deadline_persists_across_resume": True,
    }
    _atomic_json(path, campaign)
    return campaign


def _status_writer_stage6br(
    output: Path,
    profile_name: str,
    campaign: dict[str, Any],
    active_phase: list[str],
):
    def status(state: str, current: str, **extra: Any) -> None:
        now = time.time()
        phase = active_phase[0]
        payload = {
            "state": state,
            "stage": "6br-renewal-coverage-scale",
            "phase": phase,
            "current": current,
            "profile": profile_name,
            "pid": os.getpid(),
            "workers_hard_max": RenewalContract().max_workers,
            "started_unix": float(campaign["started_unix"]),
            "updated_unix": now,
            "elapsed_seconds": now - float(campaign["started_unix"]),
            "hard_deadline_unix": float(campaign["hard_deadline_unix"]),
            "exploration_deadline_unix": float(
                campaign["exploration_deadline_unix"]
            ),
            "deadline_remaining_seconds": max(
                0.0, float(campaign["hard_deadline_unix"]) - now
            ),
            **extra,
        }
        _atomic_json(output / "STATUS.json", payload)
        if phase in PHASES:
            (output / phase).mkdir(parents=True, exist_ok=True)
            _atomic_json(output / phase / "STATUS.json", payload)
        progress = (
            f" {extra['completed']}/{extra['total']}"
            if "completed" in extra
            else ""
        )
        print(f"[{state}] stage6br-{phase}:{current}{progress}", flush=True)

    return status


def _require_phase_complete(
    output: Path, phase: str, design_digest: str
) -> dict[str, Any]:
    path = output / phase / "RESULTS.json"
    if not path.exists() or not (output / phase / "COMPLETE").exists():
        raise FileNotFoundError(f"completed Stage-6B-R predecessor required: {path}")
    result = _load_json(path)
    if result.get("design_digest") != design_digest:
        raise ValueError(f"Stage-6B-R {phase} design mismatch")
    if result.get("state") != "complete":
        raise ValueError(f"Stage-6B-R {phase} is not complete")
    return result


def run_motif_renewal_repair(
    output: Path,
    *,
    phase: str = "all",
    stage6ar_root: Path = DEFAULT_STAGE6AR_ROOT,
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
    max_hours: float = 8.0,
    resume: bool = False,
    auto_final_audit: bool = False,
) -> dict[str, Any]:
    """Run the checkpointed Stage-6B-R campaign under one persisted clock."""

    require_pinned_numpy()
    if phase not in (*PHASES, "all"):
        raise ValueError(f"unknown Stage-6B-R phase {phase!r}")
    if profile_name not in PUBLIC_PROFILES:
        raise ValueError(f"unknown Stage-6B-R profile {profile_name!r}")
    if workers < 1 or workers > RenewalContract().max_workers:
        raise ValueError("Stage-6B-R workers must be in [1, 4]")
    if max_hours <= 0.0 or max_hours > RenewalContract().max_hours:
        raise ValueError("Stage-6B-R max-hours must be in (0, 8]")
    if profile_name == "reference" and not auto_final_audit:
        raise ValueError(
            "reference Stage-6B-R requires the registered automatic final-audit authorization"
        )
    if not PROTOCOL_PATH.exists():
        raise FileNotFoundError(PROTOCOL_PATH)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _atomic_text(output / "RUN.pid", f"{os.getpid()}\n")
    try:
        (
            frozen,
            cohorts,
            profile,
            contract,
            writer_contract,
            design_digest,
        ) = _prepare_stage6br(
            output,
            profile_name,
            auto_final_audit=auto_final_audit,
            stage6ar_root=stage6ar_root,
            stage6_root=stage6_root,
            stage5r_root=stage5r_root,
            stage5_root=stage5_root,
            stage4_root=stage4_root,
            stage3r_root=stage3r_root,
            stage3_root=stage3_root,
            stage2_root=stage2_root,
            stage1_root=stage1_root,
        )
        campaign = _campaign_clock(
            output, design_digest, max_hours, resume=resume
        )
        active_phase = ["prepare"]
        status = _status_writer_stage6br(
            output, profile_name, campaign, active_phase
        )
        status("running", "frozen-design-verified")
        selected = PHASES if phase == "all" else (phase,)
        if phase != "all":
            for predecessor in PHASES[: PHASES.index(phase)]:
                _require_phase_complete(output, predecessor, design_digest)
        scientific = profile_name == "reference"
        result: dict[str, Any] = {
            "state": "not_started",
            "design_digest": design_digest,
        }
        runners = {
            "transient": lambda deadline: _run_transient(
                output,
                frozen,
                cohorts,
                profile,
                contract,
                writer_contract,
                design_digest,
                workers=workers,
                resume=resume,
                deadline=deadline,
                status=status,
                scientific=scientific,
            ),
            "repair": lambda deadline: _run_repair(
                output,
                frozen,
                cohorts,
                profile,
                contract,
                writer_contract,
                design_digest,
                workers=workers,
                resume=resume,
                deadline=deadline,
                status=status,
                scientific=scientific,
            ),
            "coverage": lambda deadline: _run_coverage(
                output,
                frozen,
                cohorts,
                profile,
                contract,
                writer_contract,
                design_digest,
                workers=workers,
                resume=resume,
                deadline=deadline,
                status=status,
                scientific=scientific,
            ),
            "scale": lambda deadline: _run_scale(
                output,
                frozen,
                cohorts,
                profile,
                contract,
                writer_contract,
                design_digest,
                workers=workers,
                resume=resume,
                deadline=deadline,
                status=status,
                scientific=scientific,
            ),
            "adjudicate": lambda deadline: _run_adjudicate(
                output,
                frozen,
                cohorts,
                profile,
                profile_name,
                contract,
                writer_contract,
                design_digest,
                workers=workers,
                resume=resume,
                deadline=deadline,
                status=status,
                scientific=scientific,
                auto_final_audit=auto_final_audit,
            ),
        }
        for current_phase in selected:
            active_phase[0] = current_phase
            phase_root = output / current_phase
            phase_root.mkdir(parents=True, exist_ok=True)
            if resume and (phase_root / "COMPLETE").exists():
                result = _require_phase_complete(
                    output, current_phase, design_digest
                )
                status("skipped", "complete-checkpoint")
                continue
            if current_phase != "transient":
                predecessor = PHASES[PHASES.index(current_phase) - 1]
                _require_phase_complete(output, predecessor, design_digest)
            deadline = float(
                campaign[
                    "hard_deadline_unix"
                    if current_phase == "adjudicate"
                    else "exploration_deadline_unix"
                ]
            )
            if time.time() >= deadline:
                result = {
                    "state": "partial_budget_exhausted",
                    "phase": current_phase,
                    "design_digest": design_digest,
                }
                status("partial_budget_exhausted", "deadline-before-phase")
                break
            manifest = {
                "experiment": "ca_motif_lineage_stage_6br",
                "phase": current_phase,
                "profile": profile_name,
                "design_digest": design_digest,
                "contract_digest": contract.digest,
                "workers": workers,
                "max_hours_total_campaign": max_hours,
                "resume": resume,
                "auto_final_audit": auto_final_audit,
                "campaign_started_unix": campaign["started_unix"],
                "hard_deadline_unix": campaign["hard_deadline_unix"],
                "exploration_deadline_unix": campaign[
                    "exploration_deadline_unix"
                ],
                "phase_started_unix": time.time(),
                "environment": {
                    "python": sys.version,
                    "numpy": np.__version__,
                    "platform": platform.platform(),
                    "cpu_count": os.cpu_count(),
                    "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                },
            }
            manifest_path = phase_root / "MANIFEST.json"
            if not manifest_path.exists():
                _atomic_json(manifest_path, manifest)
            status("running", "phase-start")
            result = runners[current_phase](deadline)
            if result.get("state") != "complete":
                status(
                    str(result.get("state", "partial")),
                    "phase-incomplete",
                )
                break
            _atomic_json(
                output / "QUEUE.json",
                {
                    "experiment": "ca_motif_lineage_stage_6br",
                    "design_digest": design_digest,
                    "state": (
                        "complete"
                        if current_phase == PHASES[-1]
                        else "automatic_progression"
                    ),
                    "completed_phase": current_phase,
                    "next_phase": (
                        PHASES[PHASES.index(current_phase) + 1]
                        if current_phase != PHASES[-1]
                        else None
                    ),
                    "automatic_launch": current_phase != PHASES[-1],
                    "final_audit_trajectory_state": result.get(
                        "final_audit_trajectory_state", "untouched"
                    ),
                },
            )
            status("completed_phase", "phase-complete")
        if result.get("state") == "complete" and selected[-1] == "adjudicate":
            _atomic_text(output / "COMPLETE", "complete\n")
            active_phase[0] = "complete"
            status(
                "complete",
                "campaign-complete",
                verdict=result.get("verdict"),
                final_audit_trajectory_state=result.get(
                    "final_audit_trajectory_state", "untouched"
                ),
            )
        return result
    except BaseException as error:
        now = time.time()
        _atomic_json(
            output / "STATUS.json",
            {
                "state": "failed",
                "stage": "6br-renewal-coverage-scale",
                "pid": os.getpid(),
                "updated_unix": now,
                "error": repr(error),
            },
        )
        raise


__all__ = [
    "DEFAULT_STAGE6AR_ROOT",
    "PHASES",
    "PUBLIC_PROFILES",
    "RENEWAL_PROFILES",
    "RenewalContract",
    "RenewalProfile",
    "build_coverage_variants",
    "build_renewal_candidates",
    "build_scale_variants",
    "candidate_public",
    "centered_reduce_endpoint",
    "centred_causal_overlap",
    "coded_payload_roundtrip",
    "hamming84_decode",
    "hamming84_encode",
    "latch_update",
    "run_motif_renewal_repair",
    "seed_offsets",
    "select_renewal_cohorts",
    "simulate_renewal_lineage",
]
