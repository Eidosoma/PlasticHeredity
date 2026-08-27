"""Stage-5R regenerative localization of the compact CA Walsh carrier.

The inherited object is still the frozen Stage-4 64-bit payload.  Unlike the
failed Stage-5 diffusion layer, an occupied seed locally copies itself through
an eight-step germination wave before it is read.  Daughter observations are
then consolidated by an audited finite nearest-neighbour reduction before the
next seed is written.
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

from .causal_heredity import _atomic_json, _atomic_text, _hash_seed, _sha256, _state_from_hex
from .e19 import require_pinned_numpy
from .life_family import live_2x2_counts_batch
from .lineage_field import load_round3_pairs
from .motif_compression import (
    CompressionContract,
    _configuration_payload,
    _phase_rows,
    _run_json_checkpoints,
    decode_payload,
    encode_payload,
    load_stage3r_fit_matrix,
    quantize_payload,
    simulate_compressed_lineage,
)
from .motif_lineage import (
    MotifContract,
    ReaderConfiguration,
    _bootstrap,
    _founders,
    _paired_uniforms,
    _step,
    apply_energy_reader,
    collect_trajectory_counts,
    motif3_codes,
    write_parent_carriers,
)
from .motif_lineage_stage3 import CHECKPOINT_GENERATIONS, motif_counts_batch, write_energy_from_counts
from .motif_localization import (
    DEFAULT_STAGE1_ROOT,
    DEFAULT_STAGE2_ROOT,
    DEFAULT_STAGE3_ROOT,
    DEFAULT_STAGE3R_ROOT,
    DEFAULT_STAGE4_ROOT,
    GLOBAL_ANCHOR_ID,
    REGISTERED_MODE_IDS,
    _global_candidate,
    _payload_summary,
    apply_local_reader,
    embed_patch,
    load_frozen_stage4,
    patch_origins,
    quantize_channels,
    transcode_audit,
    transport_field,
    walsh_bit_supports,
    walsh_mode_ids,
)
from .motif_repair import RepairProfile, _score_state, _strict_confirmation_gate, heldout_lineage_accuracy


ROOT = Path(__file__).resolve().parent.parent
PROTOCOL_PATH = ROOT / "CA_MOTIF_LINEAGE_STAGE5R_PROTOCOL.md"
DEFAULT_STAGE5_ROOT = ROOT / "results/ca-motif-lineage-stage-5"
RULE = 31649
PHASES = (
    "audit",
    "fit",
    "calibrate",
    "bridge",
    "screen",
    "qualify",
    "transfer",
    "adjudicate",
    "confirm",
)
DEFAULT_PRECONFIRMATION_PHASES = PHASES[:-1]
CORE_CONDITIONS = (
    "intact",
    "zero_every_boundary",
    "shuffle_every_boundary",
    "read_disabled",
    "founder_write_disabled",
    "no_rewrite",
    "ablate_after_g2",
    "rescue_same_enter_g4",
    "rescue_opposite_enter_g4",
    "opposite_founder",
    "carrier_corruption_1",
)
LOCAL_EXTRA_CONDITIONS = (
    "spatial_shuffle_every_boundary",
    "write_disabled",
    "transport_disabled",
    "regeneration_disabled",
    "consolidation_disabled",
    "translated_patch",
    "half_width_bottleneck",
    "recombine_first_half",
    "recombine_second_half",
)
BRIDGE_CONDITIONS = (
    "intact",
    "transport_disabled",
    "regeneration_disabled",
    "consolidation_disabled",
    "no_rewrite",
    "founder_clamped",
)


@dataclass(frozen=True)
class RegenerationContract:
    implementation_version: str = "ca-motif-lineage-stage5r-cleanroom-v1"
    namespace: str = "plastic-ca-motif-lineage-stage5r-v1"
    rule: int = RULE
    width: int = 16
    height: int = 16
    channels: int = 16
    boundary_bits_per_value: int = 4
    generation_sweeps: int = 64
    germination_steps: int = 8
    consolidation_steps: int = 30
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
    confirmation_alpha_per_object: float = 0.005
    decoder_splits: int = 4
    science_reserve_seconds: float = 1800.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "visible_reset": "bitwise-identical native board before every generation",
                "boundary_object": "one- or four-site quantized Walsh seed plus occupancy",
                "germination": "synchronous Moore-neighbour copying; one edge per step",
                "consolidation": "15 horizontal plus 15 vertical nearest-neighbour reductions",
                "writer_access": "local current 3x3 motifs only; no labels or targets",
                "reader_access": "local current motifs and locally regenerated field only",
                "independent_unit": "matched founder pair",
                "missing_policy": "dead and unresolved futures remain in denominators",
                "claim_boundary": "synthetic CA Plastic Heredity; not biological life or agency",
            }
        )
        return payload

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


@dataclass(frozen=True)
class RegenerationProfile:
    bridge_pairs: int
    bridge_replicates: int
    bridge_generations: int
    screen_pairs: int
    screen_replicates: int
    screen_generations: int
    qualification_pairs: int
    qualification_replicates: int
    qualification_generations: int
    transfer_pairs_per_rule: int
    transfer_replicates: int
    transfer_generations: int
    confirmation_pairs: int
    confirmation_replicates: int
    confirmation_generations: int
    bootstrap_resamples: int


REGENERATION_PROFILES: dict[str, RegenerationProfile] = {
    "smoke": RegenerationProfile(2, 2, 4, 2, 2, 4, 2, 2, 4, 2, 2, 4, 2, 2, 4, 100),
    "pilot": RegenerationProfile(8, 4, 8, 16, 4, 8, 16, 8, 16, 16, 4, 16, 16, 8, 16, 1_000),
    "reference": RegenerationProfile(16, 8, 4, 64, 16, 8, 96, 32, 16, 64, 8, 16, 96, 64, 16, 10_000),
}
PUBLIC_PROFILES = tuple(REGENERATION_PROFILES)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_frozen_stage5(
    stage5_root: Path = DEFAULT_STAGE5_ROOT,
    stage4_root: Path = DEFAULT_STAGE4_ROOT,
    stage3r_root: Path = DEFAULT_STAGE3R_ROOT,
    stage3_root: Path = DEFAULT_STAGE3_ROOT,
    stage2_root: Path = DEFAULT_STAGE2_ROOT,
    stage1_root: Path = DEFAULT_STAGE1_ROOT,
) -> dict[str, Any]:
    """Validate the exposed Stage-5 negative result without opening its reserve."""

    stage5_root = stage5_root.resolve()
    names = {
        "results": "PRECONFIRMATION_RESULTS.json",
        "decision": "SELECTION_DECISION.json",
        "design": "DESIGN.json",
        "cohorts": "COHORTS.json",
        "manifest": "MANIFEST.json",
        "qualification": "QUALIFICATION.json",
    }
    paths = {key: stage5_root / name for key, name in names.items()}
    missing = [path for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing frozen Stage-5 artifacts: {missing}")
    payload = {key: _load_json(path) for key, path in paths.items()}
    digest = str(payload["design"].get("design_digest"))
    for key in ("results", "decision", "cohorts", "manifest", "qualification"):
        if str(payload[key].get("design_digest")) != digest:
            raise ValueError(f"Stage-5 {key} design digest does not match its design")
    if payload["results"].get("state") != "awaiting_confirmation":
        raise ValueError("Stage 5 is not frozen at preconfirmation review")
    if payload["decision"].get("confirmation_state") != "awaiting_human_review":
        raise ValueError("Stage-5 decision is not frozen for human review")
    if payload["decision"].get("confirmation_candidate_ids") != [GLOBAL_ANCHOR_ID]:
        raise ValueError("Stage 5 no longer has the registered global-only selection")
    if payload["qualification"].get("qualified_candidate_ids") != [GLOBAL_ANCHOR_ID]:
        raise ValueError("Stage-5 qualification is no longer the registered local negative")
    if payload["cohorts"].get("confirmation_trajectory_state") != "untouched":
        raise ValueError("Stage-5 confirmation reserve has already been opened")
    if payload["cohorts"].get("later_audit_trajectory_state") != "untouched":
        raise ValueError("Stage-5 later-audit reserve has already been opened")

    frozen4 = load_frozen_stage4(
        stage4_root, stage3r_root, stage3_root, stage2_root, stage1_root
    )
    if payload["design"].get("stage4_design_digest") != frozen4["design_digest"]:
        raise ValueError("Stage-5 ancestry does not match the supplied Stage 4")
    return {
        **payload,
        "root": stage5_root,
        "paths": paths,
        "design_digest": digest,
        "stage4": frozen4,
    }


def select_regeneration_cohorts(
    profile: RegenerationProfile,
    frozen: dict[str, Any],
    *,
    profile_name: str,
) -> dict[str, list[dict[str, Any]]]:
    by_id = frozen["stage4"]["by_id"]
    cohort = frozen["cohorts"]
    bridge = [by_id[value] for value in cohort["anatomy_pair_ids"]][: profile.bridge_pairs]
    screen = [by_id[value] for value in cohort["screen_pair_ids"]][: profile.screen_pairs]
    qualification = [by_id[value] for value in cohort["qualification_pair_ids"]][
        : profile.qualification_pairs
    ]
    sealed_ids = list(cohort["confirmation_pair_ids"]) + list(cohort["later_audit_pair_ids"])
    if len(sealed_ids) != len(set(sealed_ids)):
        raise ValueError("Stage-5 reserve identities overlap")
    if profile_name == "reference":
        if len(sealed_ids) != 158:
            raise AssertionError("the frozen Stage-5 reserve must contain 158 pairs")
        confirmation_ids = sealed_ids[: profile.confirmation_pairs]
        later_ids = sealed_ids[profile.confirmation_pairs :]
        if (len(confirmation_ids), len(later_ids)) != (96, 62):
            raise AssertionError("Stage-5R confirmation/audit split changed")
    else:
        confirmation_ids = list(cohort["screen_pair_ids"][: profile.confirmation_pairs])
        later_ids = sealed_ids
    confirmation = [by_id[value] for value in confirmation_ids]
    later_audit = [by_id[value] for value in later_ids]
    if profile_name == "reference":
        exposed = {pair["pair_id"] for pair in bridge + screen + qualification}
        if exposed & set(confirmation_ids):
            raise AssertionError("Stage-5R confirmation overlaps exposed pairs")

    secondary = load_round3_pairs()
    transfer: list[dict[str, Any]] = []
    for rule in (31648, 70366):
        transfer.extend(
            {**pair, "stage5r_transfer_rule": rule}
            for pair in secondary[rule][: profile.transfer_pairs_per_rule]
        )
    return {
        "bridge": bridge,
        "screen": screen,
        "qualification": qualification,
        "confirmation": confirmation,
        "later_audit": later_audit,
        "transfer": transfer,
    }


def _repeat_histories(values: np.ndarray, replicates: int) -> np.ndarray:
    return np.concatenate(
        (np.repeat(values[0:1], replicates, axis=0), np.repeat(values[1:2], replicates, axis=0)),
        axis=0,
    )


def _swap_histories(values: np.ndarray, replicates: int) -> np.ndarray:
    return np.concatenate((values[replicates:], values[:replicates]), axis=0)


def _stage5r_uniforms(
    pair_id: str, purpose: str, generation: int, sweep: int, replicates: int
) -> np.ndarray:
    return _paired_uniforms(
        pair_id, f"stage5r-{purpose}-generation-{generation}", sweep, replicates
    )


def seed_occupancy(
    sample_count: int,
    origins: np.ndarray,
    patch_side: int,
    *,
    height: int = 16,
    width: int = 16,
    translated: bool = False,
) -> np.ndarray:
    occupied = np.zeros((sample_count, height, width), dtype=np.bool_)
    for sample, (origin_y, origin_x) in enumerate(np.asarray(origins, dtype=np.int64)):
        if translated:
            origin_y = (origin_y + 5) % height
            origin_x = (origin_x + 7) % width
        for dy in range(patch_side):
            for dx in range(patch_side):
                occupied[sample, (origin_y + dy) % height, (origin_x + dx) % width] = True
    return occupied


def propagate_regenerative_wave(
    field: np.ndarray,
    occupancy: np.ndarray,
    propagator: str,
    steps: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    """Propagate a field locally; exact zero remains an exact fixed point."""

    values = np.asarray(field, dtype=np.float32).copy()
    occupied = np.asarray(occupancy, dtype=np.bool_).copy()
    if values.ndim != 4 or occupied.shape != values.shape[:3]:
        raise ValueError("field and occupancy shapes do not align")
    if propagator not in ("flood-retain", "flood-consensus", "bistable"):
        raise ValueError(f"unknown regenerative propagator {propagator!r}")
    trace: list[dict[str, float]] = []
    for step in range(1, steps + 1):
        neighbour_count = np.zeros(occupied.shape, dtype=np.int16)
        # Accumulate in float64 so the mean of identical float32 seed values
        # rounds back to that exact float32 value instead of drifting by ulps.
        neighbour_sum = np.zeros(values.shape, dtype=np.float64)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                shifted_occupancy = np.roll(occupied, shift=(dy, dx), axis=(1, 2))
                neighbour_count += shifted_occupancy
                neighbour_sum += np.roll(
                    values.astype(np.float64) * occupied[..., None],
                    shift=(dy, dx),
                    axis=(1, 2),
                )
        new_sites = (~occupied) & (neighbour_count > 0)
        copied = np.divide(
            neighbour_sum,
            neighbour_count[..., None],
            out=np.zeros(values.shape, dtype=np.float64),
            where=neighbour_count[..., None] > 0,
        )
        if propagator == "flood-retain":
            values[new_sites] = copied[new_sites]
        elif propagator == "flood-consensus":
            total = neighbour_sum + values.astype(np.float64) * occupied[..., None]
            divisor = neighbour_count + occupied.astype(np.int16)
            averaged = np.divide(
                total,
                divisor[..., None],
                out=np.zeros(values.shape, dtype=np.float64),
                where=divisor[..., None] > 0,
            )
            values[occupied | new_sites] = averaged[occupied | new_sites]
        else:
            # Local reaction-diffusion with an occupancy-gated copying front.
            mixed = copied
            reacted = np.tanh(mixed * np.float32(1.35)).astype(np.float32)
            values[new_sites] = reacted[new_sites]
            existing = occupied & (neighbour_count > 0)
            values[existing] = (
                np.float32(0.65) * values[existing] + np.float32(0.35) * reacted[existing]
            )
        occupied |= new_sites
        trace.append(
            {
                "step": float(step),
                "occupied_fraction": float(np.mean(occupied)),
                "nonzero_fraction": float(np.mean(values != 0.0)),
            }
        )
    return values, occupied, trace


def germinate_payload(
    payload: np.ndarray,
    origins: np.ndarray,
    propagator: str,
    steps: int,
    *,
    translated: bool = False,
    regeneration_disabled: bool = False,
    transport_disabled: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    values = np.asarray(payload, dtype=np.float32)
    mode = "translated" if translated else "contiguous"
    field = embed_patch(values, origins, mode=mode)
    occupied = seed_occupancy(
        len(values), origins, values.shape[1], translated=translated
    )
    if transport_disabled:
        return field, occupied, []
    if regeneration_disabled:
        trace: list[dict[str, float]] = []
        for step in range(1, steps + 1):
            field = transport_field(field, 0.35)
            trace.append(
                {
                    "step": float(step),
                    "occupied_fraction": float(np.mean(occupied)),
                    "nonzero_fraction": float(np.mean(field != 0.0)),
                }
            )
        return field, occupied, trace
    return propagate_regenerative_wave(field, occupied, propagator, steps)


def ring_reduce_exact(values: np.ndarray) -> np.ndarray:
    """Thirty one-edge shifts producing the exact 16x16 spatial mean everywhere."""

    source = np.asarray(values, dtype=np.float64)
    if source.ndim < 3 or source.shape[1:3] != (16, 16):
        raise ValueError("ring reduction requires 16x16 spatial axes 1 and 2")
    packet = source.copy()
    total = source.copy()
    for _ in range(15):
        packet = np.roll(packet, 1, axis=2)
        total += packet
    row_mean = total / 16.0
    packet = row_mean.copy()
    total = row_mean.copy()
    for _ in range(15):
        packet = np.roll(packet, 1, axis=1)
        total += packet
    return total / 16.0


def fit_regenerative_writers(
    fit_matrix: np.ndarray,
    reference_probability: np.ndarray,
    walsh_model: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fit label-blind moment-to-latent maps from exposed Stage-3R traces."""

    values = np.asarray(fit_matrix, dtype=np.float64)
    basis = np.asarray(walsh_model["basis"], dtype=np.float64)
    signs = basis * math.sqrt(512.0)
    logits = np.clip(values * 2.0, -8.0, 8.0)
    probabilities = np.asarray(reference_probability, dtype=np.float64)[None, :] * np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    moments = probabilities @ signs
    targets = np.float64(0.50) * (values @ basis)

    design = np.concatenate((moments, np.ones((len(moments), 1))), axis=1)
    ridge = np.eye(17, dtype=np.float64) * 1e-6
    ridge[-1, -1] = 1e-9
    fitted = np.linalg.solve(design.T @ design + ridge, design.T @ targets)
    full_weight = fitted[:16].astype(np.float32)
    full_bias = fitted[16].astype(np.float32)

    diagonal_weight = np.zeros(16, dtype=np.float32)
    diagonal_bias = np.zeros(16, dtype=np.float32)
    diagonal_prediction = np.zeros_like(targets)
    diagonal_r2: list[float] = []
    for channel in range(16):
        local_design = np.stack((moments[:, channel], np.ones(len(moments))), axis=1)
        local_ridge = np.diag((1e-6, 1e-9))
        local_fit = np.linalg.solve(
            local_design.T @ local_design + local_ridge,
            local_design.T @ targets[:, channel],
        )
        diagonal_weight[channel] = local_fit[0]
        diagonal_bias[channel] = local_fit[1]
        diagonal_prediction[:, channel] = local_design @ local_fit
        total = float(np.sum((targets[:, channel] - targets[:, channel].mean()) ** 2))
        residual = float(np.sum((targets[:, channel] - diagonal_prediction[:, channel]) ** 2))
        diagonal_r2.append(1.0 - residual / total if total > 0.0 else 0.0)

    full_prediction = design @ fitted
    full_total = float(np.sum((targets - targets.mean(axis=0, keepdims=True)) ** 2))
    full_residual = float(np.sum((targets - full_prediction) ** 2))
    writers = [
        {
            "writer_id": "hist512-exact",
            "writer_kind": "hist512-exact",
            "runtime_label_access": False,
            "runtime_target_access": False,
            "developmental_bins_per_site": 512,
        },
        {
            "writer_id": "moment16-ridge",
            "writer_kind": "moment16-ridge",
            "weight": full_weight,
            "bias": full_bias,
            "runtime_label_access": False,
            "runtime_target_access": False,
            "developmental_bins_per_site": 16,
        },
        {
            "writer_id": "moment16-diagonal",
            "writer_kind": "moment16-diagonal",
            "weight": diagonal_weight,
            "bias": diagonal_bias,
            "runtime_label_access": False,
            "runtime_target_access": False,
            "developmental_bins_per_site": 16,
        },
    ]
    return writers, {
        "label_and_outcome_blind": True,
        "samples": int(len(values)),
        "features": 16,
        "targets": 16,
        "full_ridge_r2": 1.0 - full_residual / full_total if full_total > 0.0 else 0.0,
        "diagonal_r2_by_channel": diagonal_r2,
        "diagonal_r2_median": float(np.median(diagonal_r2)),
        "repair_gain_in_target": 0.50,
    }


def calibrate_regenerative_dynamics(
    scale: np.ndarray,
    contract: RegenerationContract,
) -> dict[str, Any]:
    """Outcome-blind synthetic calibration of the three propagation classes."""

    scales = np.asarray(scale, dtype=np.float32)
    rng = np.random.default_rng(_hash_seed(contract.namespace, "synthetic-calibration"))
    signed = rng.integers(-7, 8, size=(12, 16)).astype(np.float32)
    signed[signed == 0.0] = 1.0
    payload = signed * (scales[None, :] / np.float32(7.0))
    payload = payload[:, None, None, :]
    origins = np.tile(np.asarray([[8, 8]], dtype=np.int16), (len(payload), 1))
    translated = np.tile(np.asarray([[1, 2]], dtype=np.int16), (len(payload), 1))
    rows: list[dict[str, Any]] = []
    for propagator in ("flood-retain", "flood-consensus", "bistable"):
        field, occupied, trace = germinate_payload(
            payload, origins, propagator, contract.germination_steps
        )
        target = np.broadcast_to(payload[:, 0:1, 0:1, :], field.shape)
        relative_error = float(
            np.linalg.norm(field - target) / max(float(np.linalg.norm(target)), 1e-12)
        )
        translated_field, translated_occupied, _ = germinate_payload(
            payload, translated, propagator, contract.germination_steps, translated=True
        )
        translation_error = float(
            np.linalg.norm(translated_field - target)
            / max(float(np.linalg.norm(target)), 1e-12)
        )
        zero = np.zeros((2, 1, 1, 16), dtype=np.float32)
        zero_origins = np.asarray([[3, 4], [3, 4]], dtype=np.int16)
        zero_field, _, _ = germinate_payload(
            zero, zero_origins, propagator, contract.germination_steps
        )
        impulse = np.ones((1, 1, 1, 1), dtype=np.float32)
        impulse_origin = np.asarray([[8, 8]], dtype=np.int16)
        light, _, _ = germinate_payload(impulse, impulse_origin, propagator, 3)
        support = np.argwhere(np.abs(light[0, ..., 0]) > 0.0)
        distances = [
            max(
                min(abs(int(y) - 8), 16 - abs(int(y) - 8)),
                min(abs(int(x) - 8), 16 - abs(int(x) - 8)),
            )
            for y, x in support
        ]
        coverage = float(np.mean(occupied))
        translation_coverage = float(np.mean(translated_occupied))
        light_cone = max(distances, default=0)
        score = (
            coverage
            + translation_coverage
            - relative_error
            - translation_error
            - (0.0 if np.array_equal(zero_field, np.zeros_like(zero_field)) else 10.0)
            - max(0, light_cone - 3)
        )
        rows.append(
            {
                "propagator": propagator,
                "coverage_after_8": coverage,
                "coverage_trace": trace,
                "relative_reconstruction_error": relative_error,
                "translation_coverage_after_8": translation_coverage,
                "translation_reconstruction_error": translation_error,
                "zero_exactly_stable": bool(np.array_equal(zero_field, np.zeros_like(zero_field))),
                "three_step_light_cone_max_chebyshev_distance": light_cone,
                "light_cone_pass": light_cone <= 3,
                "finite": bool(np.isfinite(field).all()),
                "score": score,
            }
        )
    eligible = [
        row
        for row in rows
        if row["finite"] and row["zero_exactly_stable"] and row["light_cone_pass"]
    ]
    selected = sorted(
        eligible,
        key=lambda row: (
            -float(row["score"]),
            ("flood-retain", "flood-consensus", "bistable").index(row["propagator"]),
        ),
    )[:2]
    if len(selected) != 2:
        raise AssertionError("two distinct regenerative propagation classes must calibrate")
    return {
        "label_and_outcome_blind": True,
        "grid": rows,
        "selected": selected,
        "selected_propagators": [row["propagator"] for row in selected],
    }


def build_regenerative_candidates(
    writers: Sequence[dict[str, Any]], calibration: dict[str, Any]
) -> list[dict[str, Any]]:
    by_id = {str(writer["writer_id"]): writer for writer in writers}
    candidates: list[dict[str, Any]] = []
    for propagator in calibration["selected_propagators"]:
        calibration_row = next(
            row for row in calibration["grid"] if row["propagator"] == propagator
        )
        for writer_id in ("hist512-exact", "moment16-ridge", "moment16-diagonal"):
            writer = by_id[writer_id]
            for patch_side in (1, 2):
                candidate_id = f"regen-{writer_id}-p{patch_side}-{propagator}"
                bins = int(writer["developmental_bins_per_site"])
                shared_bits = (
                    0
                    if writer_id == "hist512-exact"
                    else (16 * 16 + 16) * 32
                    if writer_id == "moment16-ridge"
                    else 32 * 16
                )
                candidate = {
                    "candidate_id": candidate_id,
                    "kind": "local",
                    "writer_id": writer_id,
                    "writer_kind": writer["writer_kind"],
                    "propagator": propagator,
                    "patch_side": patch_side,
                    "germination_steps": 8,
                    "rank": patch_side * patch_side * 16,
                    "bits": 4,
                    "payload_bits": patch_side * patch_side * 16 * 4,
                    "occupancy_bits": patch_side * patch_side,
                    "developmental_field_values": 16 * 16 * 16,
                    "developmental_field_storage_bits_float32": 16 * 16 * 16 * 32,
                    "developmental_writer_values": 16 * 16 * bins,
                    "developmental_writer_storage_bits_float32": 16 * 16 * bins * 32,
                    "routing_buffer_values": 2 * 16 * 16 * bins,
                    "shared_stage4_codebook_bits": 656,
                    "shared_writer_parameter_bits": shared_bits,
                    "calibration_score": float(calibration_row["score"]),
                    "runtime_label_access": False,
                    "runtime_parent_access": False,
                    "runtime_target_access": False,
                    "locality_audit": True,
                    "reduction_steps": 30,
                }
                for key in ("weight", "bias"):
                    if key in writer:
                        candidate[key] = writer[key]
                candidates.append(candidate)
    if len(candidates) != 12 or len({row["candidate_id"] for row in candidates}) != 12:
        raise AssertionError("the registered Stage-5R atlas must contain 12 unique candidates")
    return candidates


def _json_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in candidate.items()
        if key not in ("stage4_model", "weight", "bias")
        and not isinstance(value, np.ndarray)
    }


def save_regenerative_models(
    output: Path,
    writers: Sequence[dict[str, Any]],
    candidates: Sequence[dict[str, Any]],
    *,
    design_digest: str,
) -> dict[str, Any]:
    arrays: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for index, writer in enumerate(writers):
        row: dict[str, Any] = {}
        keys: dict[str, str] = {}
        for key, value in writer.items():
            if isinstance(value, np.ndarray):
                array_key = f"writer_{index:02d}__{key}"
                arrays[array_key] = value
                keys[key] = array_key
            else:
                row[key] = value
        row["array_keys"] = keys
        rows.append(row)
    path = output / "REGENERATIVE_MODELS.npz"
    _atomic_npz(path, **arrays)
    manifest = {
        "design_digest": design_digest,
        "model_sha256": _sha256(path),
        "allow_pickle": False,
        "writers": rows,
        "candidates": [_json_candidate(candidate) for candidate in candidates],
    }
    _atomic_json(output / "REGENERATIVE_MODELS.json", manifest)
    return manifest


def load_regenerative_models(
    output: Path, design_digest: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = _load_json(output / "REGENERATIVE_MODELS.json")
    if manifest.get("design_digest") != design_digest:
        raise ValueError("regenerative-model design digest mismatch")
    path = output / "REGENERATIVE_MODELS.npz"
    if _sha256(path) != manifest.get("model_sha256"):
        raise ValueError("regenerative-model archive hash mismatch")
    writers: list[dict[str, Any]] = []
    with np.load(path, allow_pickle=False) as arrays:
        for metadata in manifest["writers"]:
            writer = {key: value for key, value in metadata.items() if key != "array_keys"}
            for key, array_key in metadata["array_keys"].items():
                writer[key] = np.asarray(arrays[array_key])
            writers.append(writer)
    by_id = {str(writer["writer_id"]): writer for writer in writers}
    candidates: list[dict[str, Any]] = []
    for metadata in manifest["candidates"]:
        candidate = dict(metadata)
        writer = by_id[str(candidate["writer_id"])]
        for key in ("weight", "bias"):
            if key in writer:
                candidate[key] = writer[key]
        candidates.append(candidate)
    return writers, candidates


def writer_latent_from_counts(
    counts: np.ndarray,
    candidate: dict[str, Any],
    reference_probability: np.ndarray,
    walsh_model: dict[str, Any],
    writer_contract: MotifContract,
    contract: RegenerationContract,
) -> tuple[np.ndarray, float]:
    values = np.asarray(counts, dtype=np.float64)
    if candidate["writer_kind"] == "hist512-exact":
        energy = write_energy_from_counts(values, reference_probability, writer_contract)
        latent = np.float32(contract.repair_gain) * (
            energy @ np.asarray(walsh_model["basis"], dtype=np.float32)
        )
    else:
        alpha = writer_contract.jeffreys_alpha
        probability = (values + alpha) / (
            values.sum(axis=1, keepdims=True) + 512.0 * alpha
        )
        signs = np.asarray(walsh_model["basis"], dtype=np.float64) * math.sqrt(512.0)
        moments = probability @ signs
        if candidate["writer_kind"] == "moment16-ridge":
            latent = moments @ np.asarray(candidate["weight"], dtype=np.float64)
        else:
            latent = moments * np.asarray(candidate["weight"], dtype=np.float64)
        latent = latent + np.asarray(candidate["bias"], dtype=np.float64)
    return quantize_payload(np.asarray(latent, dtype=np.float32), walsh_model)


def _seed_from_latent(latent: np.ndarray, patch_side: int) -> np.ndarray:
    return np.broadcast_to(
        np.asarray(latent, dtype=np.float32)[:, None, None, :],
        (len(latent), patch_side, patch_side, latent.shape[-1]),
    ).copy()


def _founder_regenerative_payload(
    pair: dict[str, Any],
    candidate: dict[str, Any],
    walsh_model: dict[str, Any],
    reference_probability: np.ndarray,
    writer_contract: MotifContract,
    replicates: int,
    rule: int,
) -> tuple[np.ndarray, np.ndarray]:
    founders = _founders(pair)
    counts = collect_trajectory_counts(
        founders, (32,), rule=rule
    )[32]
    latent, _ = writer_latent_from_counts(
        counts["motif"],
        candidate,
        reference_probability,
        walsh_model,
        replace(writer_contract, rule=rule),
        RegenerationContract(rule=rule),
    )
    payload = _seed_from_latent(_repeat_histories(latent, replicates), int(candidate["patch_side"]))
    return payload, counts["terminal"]


def _apply_seed_intervention(
    payload: np.ndarray,
    condition: str,
    generation: int,
    pair_id: str,
    replicates: int,
    source_exits: Sequence[np.ndarray] | None,
    scale: np.ndarray,
) -> tuple[np.ndarray, float]:
    result = np.asarray(payload, dtype=np.float32).copy()
    if condition == "zero_every_boundary":
        result.fill(0.0)
    elif condition == "shuffle_every_boundary":
        permutation = np.random.default_rng(
            _hash_seed("stage5r-channel-shuffle", pair_id, generation)
        ).permutation(result.shape[-1])
        result = result[..., permutation]
    elif condition == "spatial_shuffle_every_boundary":
        flat = result.reshape(len(result), -1, result.shape[-1])
        permutation = np.random.default_rng(
            _hash_seed("stage5r-spatial-shuffle", pair_id, generation)
        ).permutation(flat.shape[1])
        result = flat[:, permutation].reshape(result.shape)
    elif condition == "opposite_founder" and generation == 1:
        result = _swap_histories(result, replicates)
    elif condition in ("ablate_after_g2", "rescue_same_enter_g4", "rescue_opposite_enter_g4") and generation == 3:
        result.fill(0.0)
    elif condition in ("rescue_same_enter_g4", "rescue_opposite_enter_g4") and generation == 4:
        if source_exits is None or len(source_exits) < 3:
            raise ValueError("rescue requires a contemporaneous intact sister payload")
        result = np.asarray(source_exits[2], dtype=np.float32).copy()
        if condition == "rescue_opposite_enter_g4":
            result = _swap_histories(result, replicates)
    elif condition == "carrier_corruption_1":
        half = np.random.default_rng(
            _hash_seed("stage5r-carrier-corruption", pair_id, generation)
        ).random((replicates, *result.shape[1:])) < 0.01
        result[np.concatenate((half, half), axis=0)] *= -1.0
    elif condition == "half_width_bottleneck":
        result[..., 1::2] = 0.0
    elif condition in ("recombine_first_half", "recombine_second_half"):
        swapped = _swap_histories(result, replicates)
        channel_slice = slice(0, 8) if condition == "recombine_first_half" else slice(8, 16)
        result[..., channel_slice] = swapped[..., channel_slice]
    return quantize_channels(result, scale)


def _damage_seed(
    payload: np.ndarray,
    pair_id: str,
    candidate_id: str,
    generation: int,
    replicates: int,
    stress_id: str,
    stress: dict[str, float | int],
    scale: np.ndarray,
) -> tuple[np.ndarray, float]:
    result = np.asarray(payload, dtype=np.float32).copy()
    erasure = float(stress.get("erasure", 0.0))
    sign = float(stress.get("sign_corruption", 0.0))
    if erasure:
        half = np.random.default_rng(
            _hash_seed("stage5r-damage", pair_id, candidate_id, stress_id, generation, "erase")
        ).random((replicates, *result.shape[1:])) < erasure
        result[np.concatenate((half, half), axis=0)] = 0.0
    if sign:
        half = np.random.default_rng(
            _hash_seed("stage5r-damage", pair_id, candidate_id, stress_id, generation, "sign")
        ).random((replicates, *result.shape[1:])) < sign
        result[np.concatenate((half, half), axis=0)] *= -1.0
    return quantize_channels(result, scale)


def simulate_regenerative_lineage(
    pair: dict[str, Any],
    configuration: ReaderConfiguration,
    candidate: dict[str, Any],
    walsh_model: dict[str, Any],
    condition: str,
    replicates: int,
    generations: int,
    reference: dict[int, dict[str, np.ndarray]],
    writer_contract: MotifContract,
    contract: RegenerationContract,
    *,
    stress_id: str = "ordinary",
    stress: dict[str, float | int] | None = None,
    source_exits: Sequence[np.ndarray] | None = None,
    retain_exits: bool = False,
    rule_override: int | None = None,
) -> tuple[dict[str, Any], list[np.ndarray]]:
    """Simulate a lineage whose local seed regenerates before visible development."""

    valid = CORE_CONDITIONS + LOCAL_EXTRA_CONDITIONS + ("founder_clamped",)
    if condition not in valid:
        raise ValueError(f"unknown Stage-5R condition {condition!r}")
    stress = dict(stress or {})
    pair_id = str(pair["pair_id"])
    rule = int(rule_override if rule_override is not None else contract.rule)
    basis = np.asarray(walsh_model["basis"], dtype=np.float32)
    scale = np.asarray(walsh_model["quantizer_scale"], dtype=np.float32)
    reference_probability = reference[configuration.write_window]["motif_probability"]
    reset_state = _state_from_hex("life", pair["donor_a"]["initial_state_hex"])
    other_reset = _state_from_hex("life", pair["donor_b"]["initial_state_hex"])
    if not np.array_equal(reset_state, other_reset):
        raise AssertionError(f"visible reset mismatch in pair {pair_id}")
    reset = np.repeat(reset_state[None, ...], 2 * replicates, axis=0)
    payload, founder_terminal = _founder_regenerative_payload(
        pair,
        candidate,
        walsh_model,
        reference_probability,
        writer_contract,
        replicates,
        rule,
    )
    founder_payload = payload.copy()
    if condition == "founder_write_disabled":
        payload.fill(0.0)
    alive = np.ones(2 * replicates, dtype=np.bool_)
    checkpoints = {value for value in CHECKPOINT_GENERATIONS if value <= generations}
    outcomes: dict[str, Any] = {}
    decoders: dict[str, Any] = {}
    carrier_history: dict[str, Any] = {}
    exits: list[np.ndarray] = []
    clipping_values: list[float] = []
    coverage_values: list[float] = []
    uniform_values: list[float] = []
    process_noise = float(stress.get("process_noise", contract.process_noise))

    for generation in range(1, generations + 1):
        payload, clipping = _apply_seed_intervention(
            payload, condition, generation, pair_id, replicates, source_exits, scale
        )
        clipping_values.append(clipping)
        payload, clipping = _damage_seed(
            payload,
            pair_id,
            str(candidate["candidate_id"]),
            generation,
            replicates,
            stress_id,
            stress,
            scale,
        )
        clipping_values.append(clipping)
        entry_payload = payload.copy()
        entry_summary = _payload_summary(entry_payload, replicates)
        origins = patch_origins(pair_id, generation, replicates)
        field, occupied, wave_trace = germinate_payload(
            entry_payload,
            origins,
            str(candidate["propagator"]),
            contract.germination_steps,
            translated=condition == "translated_patch",
            regeneration_disabled=condition == "regeneration_disabled",
            transport_disabled=condition == "transport_disabled",
        )
        coverage_values.append(float(np.mean(occupied)))
        uniform = np.array_equal(
            field,
            np.broadcast_to(field[:, 0:1, 0:1, :], field.shape),
        )
        uniform_values.append(float(uniform))
        state = reset.copy()
        state[~alive] = False
        if not np.array_equal(state[alive], reset[alive]):
            raise AssertionError("visible reset was not bitwise identical")
        recent: deque[np.ndarray] = deque(maxlen=writer_contract.observation_window)
        counts = np.zeros((2 * replicates, 512), dtype=np.float64)
        site_counts = np.zeros_like(counts)
        site_origins = patch_origins(pair_id, generation + 1, replicates)
        sample_indices = np.arange(2 * replicates)
        for sweep in range(1, contract.generation_sweeps + 1):
            predicted = _step(state, rule)
            if condition != "read_disabled" and sweep <= contract.read_sweeps:
                uniforms = _stage5r_uniforms(
                    pair_id, "read", generation, sweep, replicates
                )
                if uniform:
                    carrier = decode_payload(field[:, 0, 0, :], walsh_model)
                    predicted = apply_energy_reader(
                        predicted, carrier, uniforms, configuration.strength
                    )
                else:
                    predicted = apply_local_reader(
                        predicted, field, basis, uniforms, configuration.strength
                    )
            predicted ^= (
                _stage5r_uniforms(pair_id, "process", generation, sweep, replicates)
                < process_noise
            )
            predicted[~alive] = False
            state = predicted
            if contract.write_start <= sweep <= contract.write_end:
                codes = motif3_codes(state)
                counts += motif_counts_batch(codes)
                selected_codes = codes[
                    sample_indices, site_origins[:, 0], site_origins[:, 1]
                ].astype(np.int64)
                np.add.at(site_counts, (sample_indices, selected_codes), 1.0)
            if sweep >= contract.observe_start:
                recent.append(live_2x2_counts_batch(state))
        alive &= state.any(axis=(1, 2))
        if condition == "no_rewrite":
            next_payload, clipping = quantize_channels(
                entry_payload * np.float32(contract.stale_retention), scale
            )
        elif condition == "write_disabled":
            next_payload = np.zeros_like(entry_payload)
            clipping = 0.0
        elif condition == "founder_clamped":
            next_payload = founder_payload.copy()
            clipping = 0.0
        else:
            writer_counts = site_counts if condition == "consolidation_disabled" else counts
            latent, clipping = writer_latent_from_counts(
                writer_counts,
                candidate,
                reference_probability,
                walsh_model,
                replace(writer_contract, rule=rule),
                contract,
            )
            next_payload = _seed_from_latent(latent, int(candidate["patch_side"]))
        payload = next_payload
        clipping_values.append(clipping)
        payload[~alive] = 0.0
        if generation in checkpoints:
            outcome, vectors = _score_state(
                state, recent, pair, founder_terminal, replicates, writer_contract
            )
            outcomes[str(generation)] = outcome
            decoders[str(generation)] = {
                "carrier_balanced_accuracy": heldout_lineage_accuracy(
                    payload.reshape(len(payload), -1),
                    replicates,
                    _hash_seed(
                        contract.namespace,
                        pair_id,
                        candidate["candidate_id"],
                        condition,
                        stress_id,
                        generation,
                        "carrier",
                    ),
                    contract.decoder_splits,
                ),
                "phenotype_balanced_accuracy": heldout_lineage_accuracy(
                    vectors,
                    replicates,
                    _hash_seed(
                        contract.namespace,
                        pair_id,
                        candidate["candidate_id"],
                        condition,
                        stress_id,
                        generation,
                        "phenotype",
                    ),
                    contract.decoder_splits,
                ),
            }
            carrier_history[str(generation)] = {
                "entry": entry_summary,
                "exit": _payload_summary(payload, replicates),
                "occupied_fraction_after_germination": float(np.mean(occupied)),
                "uniform_after_germination": uniform,
                "wave_trace": wave_trace,
                "surviving_futures": int(np.count_nonzero(alive)),
            }
        if retain_exits:
            exits.append(payload.copy())

    return (
        {
            "candidate_id": candidate["candidate_id"],
            "condition": condition,
            "stress_id": stress_id,
            "stress": stress,
            "rule": rule,
            "reset_sha256": hashlib.sha256(reset_state.tobytes()).hexdigest(),
            "reset_asserted_before_every_generation": True,
            "boundary_patch_side": int(candidate["patch_side"]),
            "inherited_payload_bits": int(candidate["payload_bits"]),
            "occupancy_bits": int(candidate["occupancy_bits"]),
            "developmental_field_values": int(candidate["developmental_field_values"]),
            "developmental_writer_values": int(candidate["developmental_writer_values"]),
            "founder_payload": _payload_summary(founder_payload, replicates),
            "boundary_clipping_fraction_mean": float(np.mean(clipping_values)) if clipping_values else 0.0,
            "germination_coverage_mean": float(np.mean(coverage_values)) if coverage_values else 0.0,
            "germination_uniform_fraction": float(np.mean(uniform_values)) if uniform_values else 0.0,
            "outcomes": outcomes,
            "decoders": decoders,
            "carrier_history": carrier_history,
        },
        exits,
    )


def _simulate_candidate(
    pair: dict[str, Any],
    configuration: ReaderConfiguration,
    candidate: dict[str, Any],
    walsh_model: dict[str, Any],
    condition: str,
    replicates: int,
    generations: int,
    reference: dict[int, dict[str, np.ndarray]],
    writer_contract: MotifContract,
    contract: RegenerationContract,
    *,
    stress_id: str = "ordinary",
    stress: dict[str, float | int] | None = None,
    source_exits: Sequence[np.ndarray] | None = None,
    retain_exits: bool = False,
    rule_override: int | None = None,
) -> tuple[dict[str, Any], list[np.ndarray]]:
    if candidate["kind"] == "global":
        if condition not in CORE_CONDITIONS:
            raise ValueError(f"global anchor does not define local condition {condition!r}")
        return simulate_compressed_lineage(
            pair,
            configuration,
            candidate["stage4_model"],
            condition,
            replicates,
            generations,
            reference,
            writer_contract,
            CompressionContract(),
            stress_id=stress_id,
            stress=stress,
            source_exits=source_exits,
            retain_exits=retain_exits,
            rule_override=rule_override,
        )
    return simulate_regenerative_lineage(
        pair,
        configuration,
        candidate,
        walsh_model,
        condition,
        replicates,
        generations,
        reference,
        writer_contract,
        contract,
        stress_id=stress_id,
        stress=stress,
        source_exits=source_exits,
        retain_exits=retain_exits,
        rule_override=rule_override,
    )


def _bridge_pair_task(
    payload: tuple[
        dict[str, Any],
        list[dict[str, Any]],
        MotifContract,
        RegenerationContract,
        dict[int, dict[str, np.ndarray]],
    ]
) -> dict[str, Any]:
    item, candidates, writer_contract, contract, reference = payload
    configuration = ReaderConfiguration(**item["configuration"])
    walsh_model = item["walsh_model"]
    local = next(row for row in candidates if row["candidate_id"] == item["local_candidate_id"])
    global_candidate = next(row for row in candidates if row["candidate_id"] == GLOBAL_ANCHOR_ID)
    global_result, _ = _simulate_candidate(
        item["pair"],
        configuration,
        global_candidate,
        walsh_model,
        "intact",
        int(item["replicates"]),
        int(item["generations"]),
        reference,
        writer_contract,
        contract,
    )
    intact, exits = _simulate_candidate(
        item["pair"],
        configuration,
        local,
        walsh_model,
        "intact",
        int(item["replicates"]),
        int(item["generations"]),
        reference,
        writer_contract,
        contract,
        retain_exits=True,
    )
    conditions = {"intact": intact}
    for condition in BRIDGE_CONDITIONS[1:]:
        result, _ = _simulate_candidate(
            item["pair"],
            configuration,
            local,
            walsh_model,
            condition,
            int(item["replicates"]),
            int(item["generations"]),
            reference,
            writer_contract,
            contract,
            source_exits=exits,
        )
        conditions[condition] = result
    return {
        "checkpoint": item["checkpoint"],
        "pair_id": item["pair"]["pair_id"],
        "global_anchor": global_result,
        "local_candidate_id": local["candidate_id"],
        "conditions": conditions,
    }


def _screen_pair_task(
    payload: tuple[
        dict[str, Any],
        list[dict[str, Any]],
        MotifContract,
        RegenerationContract,
        dict[int, dict[str, np.ndarray]],
    ]
) -> dict[str, Any]:
    item, candidates, writer_contract, contract, reference = payload
    configuration = ReaderConfiguration(**item["configuration"])
    candidate = next(row for row in candidates if row["candidate_id"] == item["candidate_id"])
    result, _ = _simulate_candidate(
        item["pair"],
        configuration,
        candidate,
        item["walsh_model"],
        "intact",
        int(item["replicates"]),
        int(item["generations"]),
        reference,
        writer_contract,
        contract,
    )
    return {
        "checkpoint": item["checkpoint"],
        "pair_id": item["pair"]["pair_id"],
        "candidates": {str(candidate["candidate_id"]): result},
    }


def _qualification_pair_task(
    payload: tuple[
        dict[str, Any],
        list[dict[str, Any]],
        MotifContract,
        RegenerationContract,
        dict[int, dict[str, np.ndarray]],
    ]
) -> dict[str, Any]:
    item, candidates, writer_contract, contract, reference = payload
    configuration = ReaderConfiguration(**item["configuration"])
    candidate = next(row for row in candidates if row["candidate_id"] == item["candidate_id"])
    replicates = int(item["replicates"])
    generations = int(item["generations"])
    intact, exits = _simulate_candidate(
        item["pair"],
        configuration,
        candidate,
        item["walsh_model"],
        "intact",
        replicates,
        generations,
        reference,
        writer_contract,
        contract,
        retain_exits=True,
    )
    conditions: dict[str, Any] = {"intact": intact}
    registered = CORE_CONDITIONS[1:]
    if candidate["kind"] == "local":
        registered += LOCAL_EXTRA_CONDITIONS
    for condition in registered:
        result, _ = _simulate_candidate(
            item["pair"],
            configuration,
            candidate,
            item["walsh_model"],
            condition,
            replicates,
            generations,
            reference,
            writer_contract,
            contract,
            source_exits=exits,
        )
        conditions[condition] = result
    return {
        "checkpoint": item["checkpoint"],
        "pair_id": item["pair"]["pair_id"],
        "replicates": replicates,
        "generations": generations,
        "candidates": {
            str(candidate["candidate_id"]): {
                "candidate_id": candidate["candidate_id"],
                "conditions": conditions,
            }
        },
    }


def _transfer_pair_task(
    payload: tuple[
        dict[str, Any],
        list[dict[str, Any]],
        MotifContract,
        RegenerationContract,
        dict[int, dict[str, np.ndarray]],
    ]
) -> dict[str, Any]:
    item, candidates, writer_contract, contract, reference = payload
    configuration = ReaderConfiguration(**item["configuration"])
    candidate = next(row for row in candidates if row["candidate_id"] == item["candidate_id"])
    rule = int(item["pair"]["stage5r_transfer_rule"])
    result, _ = _simulate_candidate(
        item["pair"],
        configuration,
        candidate,
        item["walsh_model"],
        "intact",
        int(item["replicates"]),
        int(item["generations"]),
        reference,
        writer_contract,
        contract,
        rule_override=rule,
    )
    return {
        "checkpoint": item["checkpoint"],
        "pair_id": item["pair"]["pair_id"],
        "rule": rule,
        "candidates": {str(candidate["candidate_id"]): result},
    }


def _confirmation_pair_task(
    payload: tuple[
        dict[str, Any],
        list[dict[str, Any]],
        MotifContract,
        RegenerationContract,
        dict[int, dict[str, np.ndarray]],
    ]
) -> dict[str, Any]:
    item, candidates, writer_contract, contract, reference = payload
    configuration = ReaderConfiguration(**item["configuration"])
    candidate = next(row for row in candidates if row["candidate_id"] == item["candidate_id"])
    environment = str(item["environment"])
    stress = (
        {}
        if environment == "ordinary"
        else {"erasure": 0.10, "sign_corruption": 0.05, "process_noise": 0.004}
    )
    replicates = int(item["replicates"])
    generations = int(item["generations"])
    intact, exits = _simulate_candidate(
        item["pair"],
        configuration,
        candidate,
        item["walsh_model"],
        "intact",
        replicates,
        generations,
        reference,
        writer_contract,
        contract,
        stress_id=environment,
        stress=stress,
        retain_exits=True,
    )
    conditions: dict[str, Any] = {"intact": intact}
    registered = CORE_CONDITIONS[1:]
    if candidate["kind"] == "local":
        registered += LOCAL_EXTRA_CONDITIONS
    for condition in registered:
        result, _ = _simulate_candidate(
            item["pair"],
            configuration,
            candidate,
            item["walsh_model"],
            condition,
            replicates,
            generations,
            reference,
            writer_contract,
            contract,
            stress_id=environment,
            stress=stress,
            source_exits=exits,
        )
        conditions[condition] = result
    return {
        "checkpoint": item["checkpoint"],
        "pair_id": item["pair"]["pair_id"],
        "replicates": replicates,
        "generations": generations,
        "candidates": {
            str(candidate["candidate_id"]): {
                "environments": {environment: {"conditions": conditions}}
            }
        },
    }


def _outcome_values(
    rows: Sequence[dict[str, Any]], candidate_id: str, generation: int, metric: str
) -> list[float]:
    values: list[float] = []
    for row in rows:
        try:
            outcome = row["candidates"][candidate_id]["outcomes"][str(generation)]
            value = outcome["survival"] if metric == "survival" else outcome["primary"][metric]
        except KeyError:
            continue
        values.append(float(value))
    return values


def _condition_values(
    rows: Sequence[dict[str, Any]],
    candidate_id: str,
    condition: str,
    generation: int,
    metric: str = "crossover",
) -> list[float]:
    values: list[float] = []
    for row in rows:
        try:
            outcome = row["candidates"][candidate_id]["conditions"][condition]["outcomes"][
                str(generation)
            ]
            value = outcome["survival"] if metric == "survival" else outcome["primary"][metric]
        except KeyError:
            continue
        values.append(float(value))
    return values


def _condition_advantage(
    rows: Sequence[dict[str, Any]], candidate_id: str, control: str, generation: int
) -> list[float]:
    left = _condition_values(rows, candidate_id, "intact", generation)
    right = _condition_values(rows, candidate_id, control, generation)
    if len(left) != len(right):
        raise ValueError(f"paired Stage-5R metrics do not align for {control}")
    return [a - b for a, b in zip(left, right)]


def _repair_profile(
    profile: RegenerationProfile, role: str, *, confirmation: bool = False
) -> RepairProfile:
    pairs = profile.confirmation_pairs if confirmation else profile.qualification_pairs
    replicates = profile.confirmation_replicates if confirmation else profile.qualification_replicates
    generations = profile.confirmation_generations if confirmation else profile.qualification_generations
    return RepairProfile(
        role,
        pairs,
        replicates,
        generations,
        role,
        pairs,
        replicates,
        generations,
        pairs,
        replicates,
        generations,
        profile.bootstrap_resamples,
    )


def _positive_gate(summary: dict[str, Any], minimum: float) -> bool:
    return bool(
        summary["mean"] is not None
        and float(summary["mean"]) >= minimum
        and summary["ci"][0] is not None
        and float(summary["ci"][0]) > 0.0
    )


def summarize_bridge(
    rows: Sequence[dict[str, Any]],
    local_candidate_id: str,
    profile: RegenerationProfile,
    contract: RegenerationContract,
    complete: bool,
) -> dict[str, Any]:
    if not complete:
        return {"state": "incomplete"}
    generation = min(4, profile.bridge_generations)

    def values(condition: str) -> list[float]:
        result: list[float] = []
        for row in rows:
            try:
                result.append(
                    float(row["conditions"][condition]["outcomes"][str(generation)]["primary"]["crossover"])
                )
            except KeyError:
                continue
        return result

    global_values = [
        float(row["global_anchor"]["outcomes"][str(generation)]["primary"]["crossover"])
        for row in rows
        if str(generation) in row.get("global_anchor", {}).get("outcomes", {})
    ]
    summaries = {
        condition: _bootstrap(
            values(condition),
            profile.bootstrap_resamples,
            _hash_seed(contract.namespace, "bridge", condition),
            contract.strict_alpha,
        )
        for condition in BRIDGE_CONDITIONS
    }
    return {
        "state": "complete",
        "generation": generation,
        "local_candidate_id": local_candidate_id,
        "global_anchor": _bootstrap(
            global_values,
            profile.bootstrap_resamples,
            _hash_seed(contract.namespace, "bridge", "global"),
            contract.strict_alpha,
        ),
        "conditions": summaries,
        "transport_advantage": _bootstrap(
            [a - b for a, b in zip(values("intact"), values("transport_disabled"))],
            profile.bootstrap_resamples,
            _hash_seed(contract.namespace, "bridge", "transport-advantage"),
            contract.strict_alpha,
        ),
        "regeneration_advantage": _bootstrap(
            [a - b for a, b in zip(values("intact"), values("regeneration_disabled"))],
            profile.bootstrap_resamples,
            _hash_seed(contract.namespace, "bridge", "regeneration-advantage"),
            contract.strict_alpha,
        ),
        "consolidation_advantage": _bootstrap(
            [a - b for a, b in zip(values("intact"), values("consolidation_disabled"))],
            profile.bootstrap_resamples,
            _hash_seed(contract.namespace, "bridge", "consolidation-advantage"),
            contract.strict_alpha,
        ),
    }


def adjudicate_regeneration_screen(
    rows: Sequence[dict[str, Any]],
    candidates: Sequence[dict[str, Any]],
    profile: RegenerationProfile,
    contract: RegenerationContract,
    complete: bool,
) -> dict[str, Any]:
    if not complete:
        return {"state": "incomplete", "selected_candidate_ids": []}
    generation = min(8, profile.screen_generations)
    summaries: dict[str, Any] = {}
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        crossover_values = _outcome_values(rows, candidate_id, generation, "crossover")
        crossover = _bootstrap(
            crossover_values,
            profile.bootstrap_resamples,
            _hash_seed(contract.namespace, "screen", candidate_id, generation),
            contract.strict_alpha,
        )
        summaries[candidate_id] = {
            "candidate": _json_candidate(candidate),
            "crossover": crossover,
            "survival_mean": float(np.mean(_outcome_values(rows, candidate_id, generation, "survival"))),
            "direction_a_mean": float(np.mean(_outcome_values(rows, candidate_id, generation, "direction_a"))),
            "direction_b_mean": float(np.mean(_outcome_values(rows, candidate_id, generation, "direction_b"))),
            "fraction_pairs_positive": float(np.mean(np.asarray(crossover_values) > 0.0)),
        }
    anchor_mean = float(summaries[GLOBAL_ANCHOR_ID]["crossover"]["mean"] or 0.0)
    local = [candidate for candidate in candidates if candidate["kind"] == "local"]
    for candidate in local:
        summary = summaries[candidate["candidate_id"]]
        mean = float(summary["crossover"]["mean"] or 0.0)
        lower = summary["crossover"]["ci"][0]
        retention = mean / anchor_mean if anchor_mean > 0.0 else 0.0
        summary["anchor_retention"] = retention
        summary["screen_eligible"] = bool(
            profile.screen_generations >= 8
            and mean >= contract.screen_generation8
            and lower is not None
            and float(lower) > 0.0
            and summary["survival_mean"] >= contract.survival_gate
            and summary["direction_a_mean"] > 0.0
            and summary["direction_b_mean"] > 0.0
            and summary["fraction_pairs_positive"] >= 0.50
            and retention >= contract.screen_anchor_retention
        )
    eligible = (
        [candidate for candidate in local if summaries[candidate["candidate_id"]]["screen_eligible"]]
        if profile.screen_generations >= 8
        else local
    )
    if not eligible:
        eligible = sorted(
            local,
            key=lambda candidate: (
                -float(summaries[candidate["candidate_id"]]["crossover"]["mean"] or -1.0),
                int(candidate["payload_bits"]),
                candidate["candidate_id"],
            ),
        )[:1]

    def effect(candidate: dict[str, Any]) -> float:
        return float(summaries[candidate["candidate_id"]]["crossover"]["mean"] or -1.0)

    exact_one = [
        candidate
        for candidate in eligible
        if candidate["writer_kind"] == "hist512-exact" and int(candidate["patch_side"]) == 1
    ]
    moment = [candidate for candidate in eligible if candidate["writer_kind"].startswith("moment16")]
    choices: list[dict[str, Any]] = []
    if exact_one:
        choices.append(min(exact_one, key=lambda candidate: (-effect(candidate), candidate["candidate_id"])))
    if moment:
        choices.append(
            min(
                moment,
                key=lambda candidate: (
                    int(candidate["payload_bits"]),
                    -effect(candidate),
                    candidate["candidate_id"],
                ),
            )
        )
    if not choices:
        choices.append(min(eligible, key=lambda candidate: (-effect(candidate), candidate["candidate_id"])))
    selected = list(dict.fromkeys(candidate["candidate_id"] for candidate in choices))[:2]
    return {
        "state": "complete",
        "generation": generation,
        "anchor_crossover_mean": anchor_mean,
        "candidate_summaries": summaries,
        "selected_candidate_ids": [GLOBAL_ANCHOR_ID, *selected],
        "scientific_gate_applied": profile.screen_generations >= 8,
        "fallback_nomination_used": not any(
            summaries[candidate["candidate_id"]].get("screen_eligible", False)
            for candidate in local
        ),
    }


def _regeneration_extra_gate(
    rows: Sequence[dict[str, Any]],
    candidate: dict[str, Any],
    profile: RegenerationProfile,
    contract: RegenerationContract,
    alpha: float,
) -> dict[str, Any]:
    candidate_id = str(candidate["candidate_id"])

    def boot(values: Sequence[float], name: str) -> dict[str, Any]:
        return _bootstrap(
            values,
            profile.bootstrap_resamples,
            _hash_seed(contract.namespace, "regenerative-extra", candidate_id, name),
            alpha,
        )

    generation = 8
    channel = boot(
        _condition_advantage(rows, candidate_id, "shuffle_every_boundary", generation),
        "channel",
    )
    transport = boot(
        _condition_advantage(rows, candidate_id, "transport_disabled", generation),
        "transport",
    )
    regeneration = boot(
        _condition_advantage(rows, candidate_id, "regeneration_disabled", generation),
        "regeneration",
    )
    consolidation = boot(
        _condition_advantage(rows, candidate_id, "consolidation_disabled", generation),
        "consolidation",
    )
    intact = boot(_condition_values(rows, candidate_id, "intact", generation), "intact")
    translated = boot(
        _condition_values(rows, candidate_id, "translated_patch", generation), "translated"
    )
    intact_mean = float(intact["mean"] or 0.0)
    translation_retention = (
        float(translated["mean"] or 0.0) / intact_mean if intact_mean > 0.0 else 0.0
    )
    payload_ok = int(candidate["payload_bits"]) == int(candidate["patch_side"]) ** 2 * 16 * 4
    passed = bool(
        _positive_gate(channel, contract.control_advantage)
        and _positive_gate(transport, contract.transport_advantage)
        and _positive_gate(regeneration, contract.regeneration_advantage)
        and _positive_gate(consolidation, contract.consolidation_advantage)
        and translated["ci"][0] is not None
        and float(translated["ci"][0]) > 0.0
        and translation_retention >= contract.translation_retention
        and candidate.get("locality_audit") is True
        and int(candidate.get("reduction_steps", 0)) == contract.consolidation_steps
        and payload_ok
    )
    return {
        "channel_shuffle_advantage": channel,
        "transport_disabled_advantage": transport,
        "regeneration_disabled_advantage": regeneration,
        "consolidation_disabled_advantage": consolidation,
        "translated_patch": translated,
        "translation_retention": translation_retention,
        "light_cone_and_runtime_locality_audit": bool(candidate.get("locality_audit")),
        "thirty_step_reduction_audit": int(candidate.get("reduction_steps", 0)) == 30,
        "payload_accounting_audit": payload_ok,
        "passed": passed,
    }


def adjudicate_regenerative_qualification(
    rows: Sequence[dict[str, Any]],
    candidates: Sequence[dict[str, Any]],
    profile: RegenerationProfile,
    contract: RegenerationContract,
    complete: bool,
) -> dict[str, Any]:
    if not complete:
        return {"state": "incomplete", "qualified_candidate_ids": [], "candidate_summaries": {}}
    gate_applied = profile.qualification_generations >= 16
    summaries: dict[str, Any] = {}
    qualified: list[str] = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        candidate_rows = [row for row in rows if candidate_id in row.get("candidates", {})]
        strict = (
            _strict_confirmation_gate(
                candidate_rows,
                candidate_id,
                _repair_profile(profile, "stage5r-qualification"),
                contract,  # type: ignore[arg-type]
                contract.strict_alpha,
            )
            if gate_applied
            else {"verdict": "NOT_ADJUDICATED_PROFILE", "renewed_gate": False}
        )
        if candidate["kind"] == "local" and gate_applied:
            local = _regeneration_extra_gate(
                candidate_rows, candidate, profile, contract, contract.strict_alpha
            )
        elif candidate["kind"] == "local":
            local = {"passed": False, "scientific_gate_applied": False}
        else:
            local = {"passed": False, "not_applicable_global_anchor": True}
        local_pass = bool(strict.get("renewed_gate") and local.get("passed"))
        summaries[candidate_id] = {
            "candidate": _json_candidate(candidate),
            "strict": strict,
            "regeneration": local,
            "regenerative_renewed_gate": local_pass,
        }
        if (not gate_applied) or (candidate["kind"] == "global" and strict.get("renewed_gate")) or local_pass:
            qualified.append(candidate_id)
    return {
        "state": "complete",
        "scientific_gate_applied": gate_applied,
        "candidate_summaries": summaries,
        "qualified_candidate_ids": qualified,
    }


def select_stage5r_finalists(
    candidates: Sequence[dict[str, Any]],
    qualification: dict[str, Any],
    qualification_rows: Sequence[dict[str, Any]],
    profile: RegenerationProfile,
) -> list[str]:
    qualified_ids = set(qualification.get("qualified_candidate_ids", ()))
    local = [
        candidate
        for candidate in candidates
        if candidate["kind"] == "local" and candidate["candidate_id"] in qualified_ids
    ]
    if profile.qualification_generations < 16:
        local = [candidate for candidate in candidates if candidate["kind"] == "local"]
    generation = min(16, profile.qualification_generations)

    def effect(candidate: dict[str, Any]) -> float:
        values = _condition_values(
            qualification_rows, str(candidate["candidate_id"]), "intact", generation
        )
        return float(np.mean(values)) if values else -1.0

    exact = [
        candidate
        for candidate in local
        if candidate["writer_kind"] == "hist512-exact" and int(candidate["patch_side"]) == 1
    ]
    moments = [candidate for candidate in local if candidate["writer_kind"].startswith("moment16")]
    choices: list[dict[str, Any]] = []
    if exact:
        choices.append(min(exact, key=lambda candidate: (-effect(candidate), candidate["candidate_id"])))
    if moments:
        choices.append(
            min(
                moments,
                key=lambda candidate: (
                    int(candidate["payload_bits"]),
                    int(candidate["shared_writer_parameter_bits"]),
                    -effect(candidate),
                    candidate["candidate_id"],
                ),
            )
        )
    return [GLOBAL_ANCHOR_ID, *list(dict.fromkeys(row["candidate_id"] for row in choices))[:2]]


def summarize_transfer(
    rows: Sequence[dict[str, Any]],
    candidate_ids: Sequence[str],
    profile: RegenerationProfile,
    contract: RegenerationContract,
    complete: bool,
) -> dict[str, Any]:
    if not complete:
        return {"state": "incomplete", "rules": {}}
    generation = min(16, profile.transfer_generations)
    rules: dict[str, Any] = {}
    for rule in (31648, 70366):
        rule_rows = [row for row in rows if int(row["rule"]) == rule]
        rules[str(rule)] = {
            "pairs": len({row["pair_id"] for row in rule_rows}),
            "candidates": {
                candidate_id: {
                    "crossover": _bootstrap(
                        _outcome_values(rule_rows, candidate_id, generation, "crossover"),
                        profile.bootstrap_resamples,
                        _hash_seed(contract.namespace, "transfer", rule, candidate_id),
                        contract.strict_alpha,
                    )
                }
                for candidate_id in candidate_ids
            },
        }
    return {"state": "complete", "generation": generation, "rules": rules, "exploratory_only": True}


def adjudicate_stage5r_confirmation(
    rows: Sequence[dict[str, Any]],
    candidates: Sequence[dict[str, Any]],
    profile: RegenerationProfile,
    contract: RegenerationContract,
    complete: bool,
) -> dict[str, Any]:
    if not complete or profile.confirmation_generations < 16:
        return {"state": "incomplete", "verdict": "INCOMPLETE", "candidates": {}}
    summaries: dict[str, Any] = {}
    passes: dict[str, dict[str, bool]] = {}
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        summaries[candidate_id] = {"candidate": _json_candidate(candidate), "environments": {}}
        passes[candidate_id] = {}
        for environment in ("ordinary", "moderate_joint"):
            transformed = [
                {
                    "candidates": {
                        candidate_id: {
                            "conditions": row["candidates"][candidate_id]["environments"][environment]["conditions"]
                        }
                    }
                }
                for row in rows
                if candidate_id in row.get("candidates", {})
                and environment in row["candidates"][candidate_id].get("environments", {})
            ]
            strict = _strict_confirmation_gate(
                transformed,
                candidate_id,
                _repair_profile(profile, f"stage5r-confirm-{environment}", confirmation=True),
                contract,  # type: ignore[arg-type]
                contract.confirmation_alpha_per_object,
            )
            if candidate["kind"] == "local":
                local = _regeneration_extra_gate(
                    transformed,
                    candidate,
                    profile,
                    contract,
                    contract.confirmation_alpha_per_object,
                )
                passed = bool(strict.get("renewed_gate") and local.get("passed"))
            else:
                local = {"not_applicable_global_anchor": True}
                passed = bool(strict.get("renewed_gate"))
            summaries[candidate_id]["environments"][environment] = {
                "strict": strict,
                "regeneration": local,
                "stage5r_gate": passed,
            }
            passes[candidate_id][environment] = passed
    anchor_pass = passes.get(GLOBAL_ANCHOR_ID, {}).get("ordinary", False)
    local64 = [
        candidate
        for candidate in candidates
        if candidate["kind"] == "local"
        and int(candidate["payload_bits"]) == 64
        and passes[candidate["candidate_id"]]["ordinary"]
    ]
    robust64 = [
        candidate for candidate in local64 if passes[candidate["candidate_id"]]["moderate_joint"]
    ]
    distributed = [
        candidate
        for candidate in candidates
        if candidate["kind"] == "local"
        and int(candidate["payload_bits"]) > 64
        and passes[candidate["candidate_id"]]["ordinary"]
    ]
    verdict = (
        "NO_STAGE4_REPLICATION"
        if not anchor_pass
        else "ROBUST_REGENERATIVE_LOCAL_64BIT_CA_PLASTIC_HEREDITY"
        if robust64
        else "REGENERATIVE_LOCAL_64BIT_CA_PLASTIC_HEREDITY"
        if local64
        else "REGENERATIVE_LOCAL_DISTRIBUTED_CA_PLASTIC_HEREDITY"
        if distributed
        else "GLOBAL_BROADCAST_ONLY"
    )
    return {
        "state": "complete",
        "verdict": verdict,
        "fresh_stage4_anchor_replicated": anchor_pass,
        "robust_regenerative_local_64bit_candidate_ids": [row["candidate_id"] for row in robust64],
        "regenerative_local_64bit_candidate_ids": [row["candidate_id"] for row in local64],
        "regenerative_local_distributed_candidate_ids": [row["candidate_id"] for row in distributed],
        "candidates": summaries,
        "claim_boundary": "synthetic CA lineage memory only; no metabolism, agency, or biological-life claim",
    }


def _selected_candidates(
    candidates: Sequence[dict[str, Any]], candidate_ids: Sequence[str]
) -> list[dict[str, Any]]:
    by_id = {str(candidate["candidate_id"]): candidate for candidate in candidates}
    missing = [candidate_id for candidate_id in candidate_ids if candidate_id not in by_id]
    if missing:
        raise ValueError(f"selected Stage-5R candidates are missing: {missing}")
    return [by_id[candidate_id] for candidate_id in candidate_ids]


def regenerative_mechanism_audit(
    walsh_model: dict[str, Any], calibration: dict[str, Any]
) -> dict[str, Any]:
    rng = np.random.default_rng(_hash_seed("stage5r-mechanism-audit", 1))
    values = rng.normal(size=(3, 16, 16, 7))
    reduced = ring_reduce_exact(values)
    expected = np.broadcast_to(values.mean(axis=(1, 2), keepdims=True), values.shape)
    reduction_error = float(np.max(np.abs(reduced - expected)))
    scale = np.asarray(walsh_model["quantizer_scale"], dtype=np.float32)
    latent = quantize_channels(
        rng.uniform(-1.0, 1.0, size=(3, 16)).astype(np.float32) * scale,
        scale,
    )[0]
    payload = latent[:, None, None, :]
    origins = np.asarray([[0, 0], [5, 7], [11, 3]], dtype=np.int16)
    field, occupied, trace = germinate_payload(payload, origins, "flood-retain", 8)
    expected_field = np.broadcast_to(latent[:, None, None, :], field.shape)
    p1_exact = bool(np.array_equal(field, expected_field) and np.all(occupied))
    transcode = transcode_audit(walsh_model)
    selected_rows = [
        row
        for row in calibration["grid"]
        if row["propagator"] in calibration["selected_propagators"]
    ]
    passed = bool(
        reduction_error <= 1e-12
        and p1_exact
        and transcode["uniform_field_global_equivalence"]
        and transcode["zero_field_exactly_inert"]
        and all(row["light_cone_pass"] and row["zero_exactly_stable"] for row in selected_rows)
    )
    return {
        "passed": passed,
        "p1_flood_retain_exactly_uniform_after_8": p1_exact,
        "p1_coverage_trace": trace,
        "ring_reduction_steps": 30,
        "ring_reduction_max_abs_error": reduction_error,
        "ring_reduction_exact_to_float64_tolerance": reduction_error <= 1e-12,
        "uniform_field_global_equivalence_max_abs_error": transcode[
            "uniform_field_global_equivalence_max_abs_error"
        ],
        "uniform_field_global_equivalence": transcode["uniform_field_global_equivalence"],
        "zero_field_exactly_inert": transcode["zero_field_exactly_inert"],
        "selected_propagation_audits": selected_rows,
        "runtime_vectorization_scope": "only fields proved bitwise uniform; all other fields use explicit local reader",
    }


def _queue(
    design_digest: str,
    state: str,
    *,
    finalists: Sequence[str] = (),
    verdict: str | None = None,
) -> dict[str, Any]:
    return {
        "experiment": "ca_motif_lineage_stage_5r",
        "design_digest": design_digest,
        "state": state,
        "finalists": list(finalists),
        "verdict": verdict,
        "confirmation_requires": ["human review", "--resume", "--authorize-confirmation"],
        "stage6_automatic_launch": False,
    }


def _render_preconfirmation_report(results: dict[str, Any]) -> str:
    bridge = results["bridge"]
    screen = results["screen"]
    qualification = results["qualification"]
    decision = results["selection_decision"]
    lines = [
        "# CA motif-lineage Stage 5R preconfirmation report",
        "",
        f"State: `{results['state']}`.",
        f"Preconfirmation verdict: `{results['preconfirmation_verdict']}`.",
        "",
        "Stage 5R replaces the failed passive diffusive patch with an occupied local copying wave. "
        "Only one or four sites cross the reset boundary; the rest of the field is regenerated by "
        "eight synchronous nearest-neighbour steps.",
        "",
        "## Mechanism audit",
        "",
        f"The audit passed: `{results['mechanism_audit']['passed']}`. A one-site seed became an "
        f"exact uniform 16x16 field after eight steps: "
        f"`{results['mechanism_audit']['p1_flood_retain_exactly_uniform_after_8']}`. The explicit "
        f"30-step reduction differed from the direct spatial mean by "
        f"`{results['mechanism_audit']['ring_reduction_max_abs_error']:.3g}`.",
        "",
        "## Exposed results",
        "",
        f"Bridge local candidate: `{bridge.get('local_candidate_id')}`. "
        f"Its generation-{bridge.get('generation')} intact mean was "
        f"`{(bridge.get('conditions', {}).get('intact', {}).get('mean') or 0.0):.4f}`; "
        f"the frozen global anchor mean was `{(bridge.get('global_anchor', {}).get('mean') or 0.0):.4f}`.",
        f"The screen anchor mean was `{screen.get('anchor_crossover_mean', 0.0):.4f}` and selected "
        f"`{', '.join(screen.get('selected_candidate_ids', ()))}`.",
        f"Qualification selected as passing: "
        f"`{', '.join(qualification.get('qualified_candidate_ids', ())) or 'none'}`.",
        "",
        "## Seal",
        "",
        f"Confirmation candidates: `{', '.join(decision.get('confirmation_candidate_ids', ())) or 'none'}`. "
        f"Confirmation state: `{decision.get('confirmation_state')}`. No sealed Stage-5 reserve "
        "trajectory was opened by this invocation.",
        "",
        "Inherited bits, temporary field storage, writer buffers, occupancy, and shared parameters "
        "are accounted separately. This is a synthetic CA lineage-memory result, not a claim about "
        "life, metabolism, agency, or consciousness.",
    ]
    return "\n".join(lines) + "\n"


def _render_preconfirmation_lay(results: dict[str, Any]) -> str:
    qualified = [
        value
        for value in results["qualification"].get("qualified_candidate_ids", ())
        if value != GLOBAL_ANCHOR_ID
    ]
    if qualified:
        outcome = (
            "At least one regenerative local carrier passed every exposed test. This is promising, "
            "but it is not confirmed until the separately sealed pairs are deliberately opened."
        )
    else:
        outcome = (
            "The local wave could be built, but no local carrier passed the full exposed renewal test. "
            "The sealed reserve therefore remains unnecessary and untouched."
        )
    return (
        "# Stage 5R in plain language\n\n"
        "Stage 5 had put a memory patch into a daughter lattice and let it spread like dye in water. "
        "It faded. Stage 5R instead lets the patch copy itself from cell to neighbouring cell. After "
        "eight small local steps, one inherited seed can cover the whole 16-by-16 lattice without "
        "jumping across space.\n\n"
        "The daughter then grows for 64 CA steps. Its cells record local patterns, pass those records "
        "around a finite neighbour-to-neighbour circuit, and make the next tiny seed. We separately "
        "turn off spreading, self-copying, reading, and rewriting to check that success genuinely "
        "depends on the whole inheritance loop.\n\n"
        f"{outcome}\n\n"
        "Even a positive result would mean engineered plastic heredity in this artificial model. It "
        "would not by itself show that the CA is alive or conscious.\n"
    )


def _render_final_report(results: dict[str, Any]) -> str:
    adjudication = results["adjudication"]
    return (
        "# CA motif-lineage Stage 5R confirmation report\n\n"
        f"State: `{results['state']}`.\n\n"
        f"Verdict: `{adjudication['verdict']}`.\n\n"
        f"Fresh global anchor replicated: `{adjudication.get('fresh_stage4_anchor_replicated')}`. "
        f"Robust regenerative 64-bit candidates: "
        f"`{', '.join(adjudication.get('robust_regenerative_local_64bit_candidate_ids', ())) or 'none'}`.\n\n"
        "All confirmation objects, thresholds, models, and cohorts were frozen before the sealed "
        "trajectories were opened. The claim remains limited to synthetic CA lineage memory.\n"
    )


def _render_final_lay(results: dict[str, Any]) -> str:
    verdict = results["adjudication"]["verdict"]
    return (
        "# Stage 5R confirmation in plain language\n\n"
        f"The sealed test finished with verdict `{verdict}`. The test asked whether a tiny inherited "
        "local seed can regrow, steer an otherwise reset daughter, and be rewritten by that daughter "
        "for many generations. Controls separately broke each link in that loop.\n\n"
        "This result concerns an engineered cellular automaton. It is evidence about a mechanism of "
        "synthetic heredity, not evidence that the automaton is biologically alive or conscious.\n"
    )


def run_motif_regeneration(
    output: Path,
    *,
    stage5_root: Path = DEFAULT_STAGE5_ROOT,
    stage4_root: Path = DEFAULT_STAGE4_ROOT,
    stage3r_root: Path = DEFAULT_STAGE3R_ROOT,
    stage3_root: Path = DEFAULT_STAGE3_ROOT,
    stage2_root: Path = DEFAULT_STAGE2_ROOT,
    stage1_root: Path = DEFAULT_STAGE1_ROOT,
    profile_name: str = "reference",
    workers: int = 20,
    max_hours: float = 8.0,
    resume: bool = False,
    phases: Sequence[str] | None = None,
    authorize_confirmation: bool = False,
) -> dict[str, Any]:
    require_pinned_numpy()
    if profile_name not in PUBLIC_PROFILES:
        raise ValueError(f"unknown Stage-5R profile {profile_name!r}")
    if max_hours <= 0.0 or max_hours > 8.0:
        raise ValueError("Stage-5R max-hours must be in (0, 8]")
    selected_phases = tuple(phases or DEFAULT_PRECONFIRMATION_PHASES)
    unknown = [phase for phase in selected_phases if phase not in PHASES]
    if unknown:
        raise ValueError(f"unknown Stage-5R phases: {unknown}")
    if "confirm" in selected_phases:
        if selected_phases != ("confirm",):
            raise ValueError("confirmation must be a separate invocation")
        if not authorize_confirmation or not resume:
            raise ValueError("confirmation requires --resume and --authorize-confirmation")
    elif authorize_confirmation:
        raise ValueError("confirmation authorization is valid only for the confirm phase")
    if not PROTOCOL_PATH.exists():
        raise FileNotFoundError(PROTOCOL_PATH)

    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    hard_deadline = started + max_hours * 3600.0
    contract = RegenerationContract()
    writer_contract = MotifContract()
    profile = REGENERATION_PROFILES[profile_name]
    reserve_seconds = min(
        contract.science_reserve_seconds, max(60.0, max_hours * 3600.0 * 0.10)
    )
    science_deadline = max(started, hard_deadline - reserve_seconds)

    def status(state: str, phase: str, **extra: Any) -> None:
        now = time.time()
        payload = {
            "state": state,
            "stage": "5R-regenerative-localization",
            "phase": phase,
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
        progress = f" {extra['completed']}/{extra['total']}" if "completed" in extra else ""
        print(f"[{state}] {phase}{progress}", flush=True)

    try:
        status("running", "freeze_and_cohort")
        frozen = load_frozen_stage5(
            stage5_root,
            stage4_root,
            stage3r_root,
            stage3_root,
            stage2_root,
            stage1_root,
        )
        frozen4 = frozen["stage4"]
        cohorts = select_regeneration_cohorts(profile, frozen, profile_name=profile_name)
        walsh_model = frozen4["winner_model"]
        mode_ids = walsh_mode_ids(walsh_model)
        if tuple(mode_ids) != REGISTERED_MODE_IDS:
            raise ValueError("the frozen Stage-4 Walsh mode order changed")
        fit_matrix, fit_groups, trace_hashes = load_stage3r_fit_matrix(frozen4["stage3r"])
        reference_probability = frozen4["reference"][
            frozen4["configuration"].write_window
        ]["motif_probability"]
        writers, fit_audit = fit_regenerative_writers(
            fit_matrix, reference_probability, walsh_model
        )
        calibration = calibrate_regenerative_dynamics(
            walsh_model["quantizer_scale"], contract
        )
        local_candidates = build_regenerative_candidates(writers, calibration)
        global_candidate = _global_candidate(walsh_model)
        all_candidates = [global_candidate, *local_candidates]
        configuration_payload = _configuration_payload(frozen4["configuration"])
        bridge_candidate = next(
            candidate
            for candidate in local_candidates
            if candidate["writer_kind"] == "hist512-exact"
            and int(candidate["patch_side"]) == 1
            and candidate["propagator"] == calibration["selected_propagators"][0]
        )
        mechanism_audit = regenerative_mechanism_audit(walsh_model, calibration)
        design_payload = {
            "experiment": "ca_motif_lineage_stage_5r",
            "contract": contract.to_dict(),
            "writer_contract_digest": writer_contract.digest,
            "profile_name": profile_name,
            "profile": asdict(profile),
            "configuration": frozen4["configuration"].to_dict(),
            "stage5_design_digest": frozen["design_digest"],
            "stage4_design_digest": frozen4["design_digest"],
            "phases_contract": PHASES,
            "confirmation_separate_invocation": True,
            "registered_stage4_winner": "walsh-r016-q04",
            "registered_walsh_mode_ids": mode_ids,
            "registered_walsh_bit_supports": walsh_bit_supports(mode_ids),
            "selected_propagators": calibration["selected_propagators"],
            "local_candidate_ids": [row["candidate_id"] for row in local_candidates],
            "bridge_candidate_id": bridge_candidate["candidate_id"],
            "bridge_pair_ids": [pair["pair_id"] for pair in cohorts["bridge"]],
            "screen_pair_ids": [pair["pair_id"] for pair in cohorts["screen"]],
            "qualification_pair_ids": [pair["pair_id"] for pair in cohorts["qualification"]],
            "confirmation_pair_ids": [pair["pair_id"] for pair in cohorts["confirmation"]],
            "later_audit_pair_ids": [pair["pair_id"] for pair in cohorts["later_audit"]],
            "transfer_pair_ids_by_rule": {
                str(rule): [
                    pair["pair_id"]
                    for pair in cohorts["transfer"]
                    if int(pair["stage5r_transfer_rule"]) == rule
                ]
                for rule in (31648, 70366)
            },
            "fit_trace_groups": len(set(fit_groups)),
            "fit_trace_sha256": trace_hashes,
            "input_sha256": {
                "protocol": _sha256(PROTOCOL_PATH),
                **{f"stage5_{key}": _sha256(path) for key, path in frozen["paths"].items()},
            },
            "implementation_sha256": {
                "motif_regeneration.py": _sha256(Path(__file__)),
                "motif_localization.py": _sha256(Path(__file__).with_name("motif_localization.py")),
                "motif_compression.py": _sha256(Path(__file__).with_name("motif_compression.py")),
            },
            "information_accounting": {
                "primary_inherited_bits": 64,
                "fallback_inherited_bits": 256,
                "primary_occupancy_bits": 1,
                "fallback_occupancy_bits": 4,
                "developmental_field_values": 4096,
                "hist512_writer_values": 16 * 16 * 512,
                "moment16_writer_values": 16 * 16 * 16,
                "routing_steps": 30,
                "categories_not_interchangeable": True,
            },
            "cleanroom_exclusion": "no Wagner or Fable implementation source is read, imported, hashed, or executed",
            "retuning_on_confirmation": False,
        }
        design_digest = hashlib.sha256(
            json.dumps(design_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        design = {**design_payload, "design_digest": design_digest}
        design_path = output / "DESIGN.json"
        if resume and design_path.exists():
            if _load_json(design_path).get("design_digest") != design_digest:
                raise ValueError("Stage-5R resume design digest mismatch")
        elif "confirm" in selected_phases:
            raise FileNotFoundError("confirmation requires a reviewed Stage-5R design")
        _atomic_json(design_path, design)
        _atomic_json(
            output / "MANIFEST.json",
            {
                "experiment": "ca_motif_lineage_stage_5r",
                "profile": profile_name,
                "design_digest": design_digest,
                "contract_digest": contract.digest,
                "workers": workers,
                "max_hours": max_hours,
                "invocation_phases": selected_phases,
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
        existing_cohort = _load_json(output / "COHORTS.json") if (output / "COHORTS.json").exists() else {}
        cohort_payload = {
            "design_digest": design_digest,
            "bridge_pair_ids": design["bridge_pair_ids"],
            "screen_pair_ids": design["screen_pair_ids"],
            "qualification_pair_ids": design["qualification_pair_ids"],
            "confirmation_pair_ids": design["confirmation_pair_ids"],
            "later_audit_pair_ids": design["later_audit_pair_ids"],
            "confirmation_trajectory_state": existing_cohort.get(
                "confirmation_trajectory_state", "untouched"
            ),
            "later_audit_trajectory_state": "untouched",
        }
        _atomic_json(output / "COHORTS.json", cohort_payload)
        _atomic_json(output / "QUEUE.json", _queue(design_digest, "running"))

        audit = {
            "design_digest": design_digest,
            "state": "passed" if mechanism_audit["passed"] else "failed",
            "stage5_state": frozen["results"]["state"],
            "stage5_local_qualified_candidate_ids": [],
            "stage5_global_anchor_only": True,
            "stage5_confirmation_pairs_still_untouched": True,
            "stage5r_confirmation_pairs": len(cohorts["confirmation"]),
            "stage5r_later_audit_pairs": len(cohorts["later_audit"]),
            "mechanism_audit": mechanism_audit,
            "cleanroom_exclusion_upheld": True,
            "confirmation_not_opened": "confirm" not in selected_phases,
        }
        if "audit" in selected_phases:
            _atomic_json(output / "CLEANROOM_AUDIT.json", audit)
        if not mechanism_audit["passed"]:
            raise AssertionError("Stage-5R mechanism audit failed")

        model_consumers = {"bridge", "screen", "qualify", "transfer", "adjudicate", "confirm"}
        if "fit" in selected_phases:
            status("running", "fit")
            fit_audit.update(
                {"design_digest": design_digest, "trace_sha256": trace_hashes}
            )
            _atomic_json(output / "WRITER_FIT_AUDIT.json", fit_audit)
            save_regenerative_models(
                output, writers, local_candidates, design_digest=design_digest
            )
        elif model_consumers.intersection(selected_phases):
            stored_writers, stored_candidates = load_regenerative_models(output, design_digest)
            if [row["writer_id"] for row in stored_writers] != [row["writer_id"] for row in writers]:
                raise ValueError("stored Stage-5R writers changed")
            if [row["candidate_id"] for row in stored_candidates] != [row["candidate_id"] for row in local_candidates]:
                raise ValueError("stored Stage-5R candidate list changed")
            local_candidates = stored_candidates
            all_candidates = [global_candidate, *local_candidates]
            bridge_candidate = next(
                row for row in local_candidates if row["candidate_id"] == design["bridge_candidate_id"]
            )

        if "calibrate" in selected_phases:
            status("running", "calibration")
            calibration["design_digest"] = design_digest
            mechanism_audit["design_digest"] = design_digest
            _atomic_json(output / "CALIBRATION.json", calibration)
            _atomic_json(output / "MECHANISM_AUDIT.json", mechanism_audit)
        elif model_consumers.intersection(selected_phases):
            stored_calibration = _load_json(output / "CALIBRATION.json")
            stored_audit = _load_json(output / "MECHANISM_AUDIT.json")
            if stored_calibration.get("design_digest") != design_digest:
                raise ValueError("Stage-5R calibration belongs to another design")
            if stored_audit.get("design_digest") != design_digest or not stored_audit.get("passed"):
                raise ValueError("a passed Stage-5R mechanism audit is required")

        downstream = {"bridge", "screen", "qualify", "transfer", "adjudicate", "confirm"}
        if not downstream.intersection(selected_phases):
            status("phases_complete", "campaign")
            return {"state": "phases_complete", "completed_phases": selected_phases}

        if "confirm" in selected_phases:
            decision = _load_json(output / "SELECTION_DECISION.json")
            confirmation_design = _load_json(output / "CONFIRMATION_DESIGN.json")
            if decision.get("design_digest") != design_digest or confirmation_design.get("design_digest") != design_digest:
                raise ValueError("Stage-5R confirmation decision belongs to another design")
            if decision.get("confirmation_state") != "awaiting_human_review":
                raise ValueError("Stage-5R confirmation is not awaiting review")
            candidate_ids = list(decision["confirmation_candidate_ids"])
            if not candidate_ids or candidate_ids != confirmation_design.get("candidate_ids"):
                raise ValueError("Stage-5R confirmation candidate list is empty or changed")
            if confirmation_design.get("model_sha256") != _sha256(output / "REGENERATIVE_MODELS.npz"):
                raise ValueError("Stage-5R model archive changed after review")
            confirmation_candidates = _selected_candidates(all_candidates, candidate_ids)
            items = [
                {
                    "checkpoint": f"confirm-{pair_index:04d}-object-{candidate_index:02d}-env-{environment}",
                    "pair": pair,
                    "candidate_id": candidate["candidate_id"],
                    "environment": environment,
                    "replicates": profile.confirmation_replicates,
                    "generations": profile.confirmation_generations,
                    "configuration": configuration_payload,
                    "walsh_model": walsh_model,
                }
                for pair_index, pair in enumerate(cohorts["confirmation"])
                for candidate_index, candidate in enumerate(confirmation_candidates)
                for environment in ("ordinary", "moderate_joint")
            ]
            status("running", "confirmation", completed=0, total=len(items))
            rows, complete = _run_json_checkpoints(
                output,
                "confirmation",
                items,
                confirmation_candidates,
                _confirmation_pair_task,
                writer_contract,
                contract,  # type: ignore[arg-type]
                frozen4["reference"],
                design_digest,
                workers=workers,
                resume=resume,
                deadline=science_deadline,
                status=status,
            )
            adjudication = adjudicate_stage5r_confirmation(
                rows, confirmation_candidates, profile, contract, complete
            )
            state = "complete" if complete else "partial_budget_exhausted"
            results = {
                "experiment": "ca_motif_lineage_stage_5r",
                "state": state,
                "profile": profile_name,
                "design_digest": design_digest,
                "stage5_design_digest": frozen["design_digest"],
                "elapsed_seconds": time.time() - started,
                "adjudication": adjudication,
            }
            _atomic_json(output / "RESULTS.json", results)
            _atomic_text(output / "REPORT.md", _render_final_report(results))
            _atomic_text(output / "LAY_SUMMARY.md", _render_final_lay(results))
            _atomic_json(
                output / "STAGE_DECISION.json",
                {
                    "design_digest": design_digest,
                    "verdict": adjudication["verdict"],
                    "decision": "stage6_may_be_planned_after_review" if complete else "resume_stage5r_confirmation",
                    "automatic_launch": False,
                    "review_required": True,
                },
            )
            if complete:
                cohort_payload["confirmation_trajectory_state"] = "complete"
                _atomic_json(output / "COHORTS.json", cohort_payload)
                _atomic_text(output / "COMPLETE", "complete\n")
            _atomic_json(
                output / "QUEUE.json",
                _queue(design_digest, state, finalists=candidate_ids, verdict=adjudication["verdict"]),
            )
            status(state, "campaign", verdict=adjudication["verdict"])
            return results

        if "bridge" in selected_phases:
            items = [
                {
                    "checkpoint": f"bridge-{index:04d}",
                    "pair": pair,
                    "local_candidate_id": bridge_candidate["candidate_id"],
                    "replicates": profile.bridge_replicates,
                    "generations": profile.bridge_generations,
                    "configuration": configuration_payload,
                    "walsh_model": walsh_model,
                }
                for index, pair in enumerate(cohorts["bridge"])
            ]
            status("running", "bridge", completed=0, total=len(items))
            bridge_rows, complete = _run_json_checkpoints(
                output,
                "bridge",
                items,
                [global_candidate, bridge_candidate],
                _bridge_pair_task,
                writer_contract,
                contract,  # type: ignore[arg-type]
                frozen4["reference"],
                design_digest,
                workers=workers,
                resume=resume,
                deadline=science_deadline,
                status=status,
            )
            bridge = summarize_bridge(
                bridge_rows, bridge_candidate["candidate_id"], profile, contract, complete
            )
            bridge["design_digest"] = design_digest
            _atomic_json(output / "BRIDGE.json", bridge)
            if not complete:
                status("partial_budget_exhausted", "bridge")
                return {"state": "partial_budget_exhausted", "phase": "bridge"}
        else:
            bridge = _load_json(output / "BRIDGE.json")

        if "screen" in selected_phases:
            items = [
                {
                    "checkpoint": f"screen-{pair_index:04d}-object-{candidate_index:02d}",
                    "pair": pair,
                    "candidate_id": candidate["candidate_id"],
                    "replicates": profile.screen_replicates,
                    "generations": profile.screen_generations,
                    "configuration": configuration_payload,
                    "walsh_model": walsh_model,
                }
                for pair_index, pair in enumerate(cohorts["screen"])
                for candidate_index, candidate in enumerate(all_candidates)
            ]
            status("running", "screen", completed=0, total=len(items))
            screen_rows, complete = _run_json_checkpoints(
                output,
                "screen",
                items,
                all_candidates,
                _screen_pair_task,
                writer_contract,
                contract,  # type: ignore[arg-type]
                frozen4["reference"],
                design_digest,
                workers=workers,
                resume=resume,
                deadline=science_deadline,
                status=status,
            )
            screen = adjudicate_regeneration_screen(
                screen_rows, all_candidates, profile, contract, complete
            )
            screen["design_digest"] = design_digest
            _atomic_json(output / "SCREEN.json", screen)
            if not complete:
                status("partial_budget_exhausted", "screen")
                return {"state": "partial_budget_exhausted", "phase": "screen"}
        else:
            screen = _load_json(output / "SCREEN.json")
            screen_rows = _phase_rows(output, "screen", design_digest)
        selected_candidates = _selected_candidates(
            all_candidates, list(screen["selected_candidate_ids"])
        )

        if "qualify" in selected_phases:
            items = [
                {
                    "checkpoint": f"qualify-{pair_index:04d}-object-{candidate_index:02d}",
                    "pair": pair,
                    "candidate_id": candidate["candidate_id"],
                    "replicates": profile.qualification_replicates,
                    "generations": profile.qualification_generations,
                    "configuration": configuration_payload,
                    "walsh_model": walsh_model,
                }
                for pair_index, pair in enumerate(cohorts["qualification"])
                for candidate_index, candidate in enumerate(selected_candidates)
            ]
            status("running", "qualification", completed=0, total=len(items))
            qualification_rows, complete = _run_json_checkpoints(
                output,
                "qualification",
                items,
                selected_candidates,
                _qualification_pair_task,
                writer_contract,
                contract,  # type: ignore[arg-type]
                frozen4["reference"],
                design_digest,
                workers=workers,
                resume=resume,
                deadline=science_deadline,
                status=status,
            )
            qualification = adjudicate_regenerative_qualification(
                qualification_rows, selected_candidates, profile, contract, complete
            )
            qualification["design_digest"] = design_digest
            _atomic_json(output / "QUALIFICATION.json", qualification)
            if not complete:
                status("partial_budget_exhausted", "qualification")
                return {"state": "partial_budget_exhausted", "phase": "qualification"}
        else:
            qualification = _load_json(output / "QUALIFICATION.json")
            qualification_rows = _phase_rows(output, "qualification", design_digest)

        confirmation_ids = select_stage5r_finalists(
            selected_candidates, qualification, qualification_rows, profile
        )
        confirmation_candidates = _selected_candidates(all_candidates, confirmation_ids)
        if "transfer" in selected_phases:
            items = [
                {
                    "checkpoint": f"transfer-{pair_index:04d}-object-{candidate_index:02d}",
                    "pair": pair,
                    "candidate_id": candidate["candidate_id"],
                    "replicates": profile.transfer_replicates,
                    "generations": profile.transfer_generations,
                    "configuration": configuration_payload,
                    "walsh_model": walsh_model,
                }
                for pair_index, pair in enumerate(cohorts["transfer"])
                for candidate_index, candidate in enumerate(confirmation_candidates)
            ]
            status("running", "transfer", completed=0, total=len(items))
            transfer_rows, complete = _run_json_checkpoints(
                output,
                "transfer",
                items,
                confirmation_candidates,
                _transfer_pair_task,
                writer_contract,
                contract,  # type: ignore[arg-type]
                frozen4["reference"],
                design_digest,
                workers=workers,
                resume=resume,
                deadline=science_deadline,
                status=status,
            )
            transfer = summarize_transfer(
                transfer_rows, confirmation_ids, profile, contract, complete
            )
            transfer["design_digest"] = design_digest
            _atomic_json(output / "TRANSFER.json", transfer)
            if not complete:
                status("partial_budget_exhausted", "transfer")
                return {"state": "partial_budget_exhausted", "phase": "transfer"}
        else:
            transfer = _load_json(output / "TRANSFER.json")

        if "adjudicate" not in selected_phases:
            status("phases_complete", "campaign")
            return {"state": "phases_complete", "completed_phases": selected_phases}

        qualified_local_ids = [
            candidate_id
            for candidate_id in qualification.get("qualified_candidate_ids", ())
            if candidate_id != GLOBAL_ANCHOR_ID
        ]
        scientific_gate = profile.qualification_generations >= 16
        if scientific_gate and not qualified_local_ids:
            confirmation_ids = []
            confirmation_state = "not_opened_no_local_qualifier"
            preconfirmation_verdict = (
                "GLOBAL_BROADCAST_ONLY"
                if GLOBAL_ANCHOR_ID in qualification.get("qualified_candidate_ids", ())
                else "NO_STAGE4_REPLICATION"
            )
        else:
            confirmation_state = "awaiting_human_review"
            preconfirmation_verdict = "REGENERATIVE_LOCAL_CANDIDATE_AWAITING_CONFIRMATION"
        candidate_metadata = {
            candidate["candidate_id"]: _json_candidate(candidate)
            for candidate in _selected_candidates(all_candidates, confirmation_ids)
        }
        decision = {
            "design_digest": design_digest,
            "stage5_design_digest": frozen["design_digest"],
            "confirmation_candidate_ids": confirmation_ids,
            "candidate_metadata": candidate_metadata,
            "confirmation_state": confirmation_state,
            "confirmation_pairs": profile.confirmation_pairs,
            "confirmation_replicates": profile.confirmation_replicates,
            "confirmation_generations": profile.confirmation_generations,
            "ordinary_and_moderate_joint": True,
            "retuning_permitted": False,
            "automatic_launch": False,
            "preconfirmation_verdict": preconfirmation_verdict,
        }
        _atomic_json(output / "SELECTION_DECISION.json", decision)
        if confirmation_ids:
            _atomic_json(
                output / "CONFIRMATION_DESIGN.json",
                {
                    "design_digest": design_digest,
                    "candidate_ids": confirmation_ids,
                    "candidate_metadata": candidate_metadata,
                    "model_sha256": _sha256(output / "REGENERATIVE_MODELS.npz"),
                    "cohort_ids_sha256": hashlib.sha256(
                        "\n".join(pair["pair_id"] for pair in cohorts["confirmation"]).encode()
                    ).hexdigest(),
                    "trajectory_state": "untouched",
                    "authorization_required": True,
                },
            )
        results = {
            "experiment": "ca_motif_lineage_stage_5r",
            "state": "awaiting_confirmation" if confirmation_ids else "complete_without_confirmation",
            "profile": profile_name,
            "design_digest": design_digest,
            "stage5_design_digest": frozen["design_digest"],
            "elapsed_seconds": time.time() - started,
            "preconfirmation_verdict": preconfirmation_verdict,
            "mechanism_audit": mechanism_audit,
            "bridge": bridge,
            "screen": screen,
            "qualification": qualification,
            "transfer": transfer,
            "selection_decision": decision,
            "information_accounting": design["information_accounting"],
        }
        _atomic_json(output / "PRECONFIRMATION_RESULTS.json", results)
        _atomic_text(output / "REPORT.md", _render_preconfirmation_report(results))
        _atomic_text(output / "LAY_SUMMARY.md", _render_preconfirmation_lay(results))
        queue_state = "blocked_pending_human_review" if confirmation_ids else "complete_no_confirmation"
        _atomic_json(
            output / "QUEUE.json",
            _queue(
                design_digest,
                queue_state,
                finalists=confirmation_ids,
                verdict=preconfirmation_verdict,
            ),
        )
        status(
            results["state"],
            "campaign",
            confirmation_candidate_ids=confirmation_ids,
            verdict=preconfirmation_verdict,
        )
        return results
    except BaseException as error:
        status("failed", "campaign", error=repr(error))
        raise
