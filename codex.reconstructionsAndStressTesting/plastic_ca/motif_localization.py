"""Stage-5 physical localization of the renewed CA Walsh carrier.

The visible CA and all Stage-4 scientific semantics stay frozen.  This module
places the registered 16-coordinate Walsh message in a second, strictly local
field and lets only a spatial bottleneck cross each reset boundary.
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
    _hadamard,
    _json_model,
    _phase_rows,
    _run_json_checkpoints,
    _selected_models,
    decode_payload,
    load_codec_models,
    load_frozen_stage3r,
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
    motif3_codes,
    motif_energy_advantage,
    write_parent_carriers,
)
from .motif_lineage_stage3 import CHECKPOINT_GENERATIONS
from .motif_repair import RepairProfile, _carrier_summary, _score_state, _strict_confirmation_gate, heldout_lineage_accuracy


ROOT = Path(__file__).resolve().parent.parent
PROTOCOL_PATH = ROOT / "CA_MOTIF_LINEAGE_STAGE5_PROTOCOL.md"
DEFAULT_STAGE4_ROOT = ROOT / "results/ca-motif-lineage-stage-4"
DEFAULT_STAGE3R_ROOT = ROOT / "results/ca-motif-lineage-stage-3r"
DEFAULT_STAGE3_ROOT = ROOT / "results/ca-motif-lineage-stage-3"
DEFAULT_STAGE2_ROOT = ROOT / "results/ca-motif-lineage-stage-2"
DEFAULT_STAGE1_ROOT = ROOT / "results/ca-motif-lineage-stage-1"
RULE = 31649
STAGE4_WINNER_ID = "walsh-r016-q04"
GLOBAL_ANCHOR_ID = "global-walsh-anchor"
REGISTERED_MODE_IDS = (0, 20, 320, 17, 272, 3, 384, 80, 288, 6, 192, 5, 36, 9, 72, 24)
PHASES = (
    "audit",
    "anatomy",
    "calibrate",
    "transcode",
    "localize",
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
    "translated_patch",
    "half_width_bottleneck",
    "dispersed_equal_bits",
    "recombine_first_half",
    "recombine_second_half",
)


@dataclass(frozen=True)
class LocalizationContract:
    implementation_version: str = "ca-motif-lineage-stage5-cleanroom-v1"
    namespace: str = "plastic-ca-motif-lineage-stage5-v1"
    rule: int = RULE
    width: int = 16
    height: int = 16
    channels: int = 16
    boundary_bits_per_value: int = 4
    generation_sweeps: int = 64
    read_sweeps: int = 32
    write_start: int = 49
    write_end: int = 64
    observe_start: int = 57
    process_noise: float = 0.002
    stale_retention: float = 0.50
    carrier_corruption: float = 0.01
    screen_generation4: float = 0.20
    screen_generation8: float = 0.15
    screen_generation16: float = 0.10
    screen_anchor_retention: float = 0.30
    control_advantage: float = 0.10
    transport_advantage: float = 0.10
    translation_retention: float = 0.50
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
                "boundary_object": "quantized spatial patch only",
                "transport": "synchronous nearest-neighbour Moore diffusion on a torus",
                "writer_access": "current local 3x3 motif and current local field only",
                "reader_access": "affected local motif and field sites only",
                "independent_unit": "matched founder pair",
                "missing_policy": "dead and unresolved futures remain in denominators",
                "claim_boundary": "synthetic CA Plastic Heredity; not biological life or agency",
            }
        )
        return payload

    @property
    def digest(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class LocalizationProfile:
    anatomy_pairs: int
    anatomy_replicates: int
    anatomy_generations: int
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


LOCALIZATION_PROFILES: dict[str, LocalizationProfile] = {
    "smoke": LocalizationProfile(2, 2, 4, 2, 2, 4, 2, 2, 4, 2, 2, 4, 2, 2, 4, 100),
    "pilot": LocalizationProfile(16, 4, 8, 16, 4, 8, 16, 8, 16, 16, 4, 16, 16, 8, 16, 1_000),
    "reference": LocalizationProfile(
        16, 8, 8, 64, 16, 8, 96, 32, 16, 64, 8, 16, 128, 64, 16, 10_000
    ),
}
PUBLIC_PROFILES = tuple(LOCALIZATION_PROFILES)


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


def load_frozen_stage4(
    stage4_root: Path = DEFAULT_STAGE4_ROOT,
    stage3r_root: Path = DEFAULT_STAGE3R_ROOT,
    stage3_root: Path = DEFAULT_STAGE3_ROOT,
    stage2_root: Path = DEFAULT_STAGE2_ROOT,
    stage1_root: Path = DEFAULT_STAGE1_ROOT,
) -> dict[str, Any]:
    """Validate Stage 4 and recover its exact winner without exposing reserve trajectories."""

    stage4_root = stage4_root.resolve()
    paths = {
        key: stage4_root / filename
        for key, filename in (
            ("results", "RESULTS.json"),
            ("decision", "STAGE_DECISION.json"),
            ("design", "DESIGN.json"),
            ("cohorts", "COHORTS.json"),
            ("manifest", "MANIFEST.json"),
            ("models", "CODEC_MODELS.json"),
            ("model_arrays", "CODEC_MODELS.npz"),
        )
    }
    missing = [path for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing frozen Stage-4 artifacts: {missing}")
    payload = {
        key: _load_json(path)
        for key, path in paths.items()
        if key != "model_arrays"
    }
    digest = str(payload["design"].get("design_digest"))
    for key in ("results", "decision", "cohorts", "manifest", "models"):
        if str(payload[key].get("design_digest")) != digest:
            raise ValueError(f"Stage-4 {key} design digest does not match its design")
    if payload["results"].get("state") != "complete":
        raise ValueError("Stage 4 is not complete")
    if payload["decision"].get("decision") != "stage5_may_be_planned_after_review":
        raise ValueError("Stage-4 decision does not permit Stage 5")
    if payload["results"].get("adjudication", {}).get("verdict") != "ROBUST_COMPACT_RENEWED_CA_PLASTIC_HEREDITY":
        raise ValueError("Stage 4 did not produce the registered robust compact result")
    if payload["cohorts"].get("confirmation_trajectory_state") != "complete":
        raise ValueError("Stage-4 confirmation ledger is not complete")
    winner = payload["results"]["adjudication"]["candidates"].get(STAGE4_WINNER_ID)
    if not winner:
        raise ValueError("registered Stage-4 Walsh winner is missing")
    for environment in ("ordinary", "moderate_joint"):
        if not winner["environments"][environment]["strict"].get("stage4_renewed_gate"):
            raise ValueError(f"Stage-4 Walsh winner failed {environment}")

    frozen3r = load_frozen_stage3r(stage3r_root, stage3_root, stage2_root, stage1_root)
    if payload["design"].get("stage3r_design_digest") != frozen3r["design_digest"]:
        raise ValueError("Stage-4 ancestry does not match Stage 3R")
    models = load_codec_models(stage4_root, digest)
    matches = [model for model in models if model["candidate_id"] == STAGE4_WINNER_ID]
    if len(matches) != 1:
        raise ValueError("Stage-4 Walsh model is absent or duplicated")
    model = matches[0]
    if int(model["rank"]) != 16 or int(model["bits"]) != 4 or int(model["payload_bits"]) != 64:
        raise ValueError("Stage-4 winner no longer has the registered 16x4-bit payload")
    reserve_ids = list(payload["cohorts"].get("stage5_reserve_pair_ids", ()))
    if len(reserve_ids) != len(set(reserve_ids)):
        raise ValueError("Stage-5 reserve contains duplicate pair identities")
    if any(pair_id not in frozen3r["by_id"] for pair_id in reserve_ids):
        raise ValueError("a Stage-5 reserve identity is absent from the frozen pair bank")
    return {
        **payload,
        "root": stage4_root,
        "paths": paths,
        "design_digest": digest,
        "stage3r": frozen3r,
        "configuration": frozen3r["configuration"],
        "reference": frozen3r["reference"],
        "by_id": frozen3r["by_id"],
        "winner_model": model,
        "reserve_ids": reserve_ids,
    }


def select_localization_cohorts(
    profile: LocalizationProfile,
    frozen: dict[str, Any],
    *,
    profile_name: str,
) -> dict[str, list[dict[str, Any]]]:
    by_id = frozen["by_id"]
    stage4_selection = [by_id[pair_id] for pair_id in frozen["cohorts"]["selection_pair_ids"]]
    stage4_confirmation = [by_id[pair_id] for pair_id in frozen["cohorts"]["confirmation_pair_ids"]]
    reserve = [by_id[pair_id] for pair_id in frozen["reserve_ids"]]
    if profile_name == "reference":
        if len(reserve) != 158:
            raise AssertionError("the registered Stage-5 reserve must contain 158 pairs")
        confirmation = reserve[: profile.confirmation_pairs]
        later_audit = reserve[profile.confirmation_pairs :]
        if len(confirmation) != 128 or len(later_audit) != 30:
            raise AssertionError("Stage-5 confirmation/audit split changed")
    else:
        confirmation = stage4_confirmation[: profile.confirmation_pairs]
        later_audit = reserve
    secondary = load_round3_pairs()
    transfer: list[dict[str, Any]] = []
    for rule in (31648, 70366):
        transfer.extend(
            {**pair, "stage5_transfer_rule": rule}
            for pair in secondary[rule][: profile.transfer_pairs_per_rule]
        )
    result = {
        "anatomy": stage4_selection[: profile.anatomy_pairs],
        "screen": stage4_confirmation[: profile.screen_pairs],
        "qualification": stage4_selection[: profile.qualification_pairs],
        "confirmation": confirmation,
        "later_audit": later_audit,
        "transfer": transfer,
    }
    scientific = [result["screen"], result["qualification"]]
    exposed_ids = {pair["pair_id"] for cohort in scientific for pair in cohort}
    confirmation_ids = {pair["pair_id"] for pair in confirmation}
    if profile_name == "reference" and exposed_ids & confirmation_ids:
        raise AssertionError("Stage-5 confirmation overlaps exposed engineering pairs")
    return result


def walsh_mode_ids(model: dict[str, Any]) -> list[int]:
    """Return exact Hadamard column identities for an archived Walsh model."""

    basis = np.asarray(model["basis"], dtype=np.float32)
    walsh = _hadamard(512)
    identities: list[int] = []
    for column in basis.T:
        matches = np.flatnonzero(np.all(walsh == column[:, None], axis=0))
        if len(matches) != 1:
            raise ValueError("an archived basis column is not an exact canonical Walsh mode")
        identities.append(int(matches[0]))
    if len(set(identities)) != len(identities):
        raise ValueError("Walsh mode identities are not unique")
    return identities


def walsh_bit_supports(mode_ids: Sequence[int]) -> list[list[int]]:
    return [[bit for bit in range(9) if (int(mode) >> bit) & 1] for mode in mode_ids]


def fit_local_writers(
    fit_matrix: np.ndarray,
    reference_probability: np.ndarray,
    walsh_model: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fit the registered label-blind parity-moment-to-Walsh affine writer."""

    values = np.asarray(fit_matrix, dtype=np.float64)
    basis = np.asarray(walsh_model["basis"], dtype=np.float64)
    scale = np.asarray(walsh_model["quantizer_scale"], dtype=np.float64)
    signs = basis * math.sqrt(512.0)
    logits = np.clip(values * 2.0, -8.0, 8.0)
    weights = np.asarray(reference_probability, dtype=np.float64)[None, :] * np.exp(logits)
    weights /= weights.sum(axis=1, keepdims=True)
    moments = weights @ signs
    coefficients = values @ basis
    slopes = np.zeros(16, dtype=np.float64)
    intercepts = np.zeros(16, dtype=np.float64)
    r2: list[float] = []
    for channel in range(16):
        design = np.stack((moments[:, channel], np.ones(len(moments))), axis=1)
        ridge = np.diag((1e-6, 1e-9))
        fitted = np.linalg.solve(design.T @ design + ridge, design.T @ coefficients[:, channel])
        slopes[channel], intercepts[channel] = fitted
        prediction = design @ fitted
        residual = float(np.sum((prediction - coefficients[:, channel]) ** 2))
        total = float(np.sum((coefficients[:, channel] - coefficients[:, channel].mean()) ** 2))
        r2.append(1.0 - residual / total if total > 0.0 else 0.0)
    slope_limit = np.maximum(scale * 64.0, 1.0)
    slopes = np.clip(slopes, -slope_limit, slope_limit)
    intercepts = np.clip(intercepts, -scale, scale)
    writers = [
        {
            "writer_id": "analytic",
            "slope": scale.astype(np.float32),
            "intercept": np.zeros(16, dtype=np.float32),
            "fit_label_access": False,
            "runtime_label_access": False,
            "runtime_parent_access": False,
            "runtime_target_access": False,
        },
        {
            "writer_id": "affine",
            "slope": slopes.astype(np.float32),
            "intercept": intercepts.astype(np.float32),
            "fit_label_access": False,
            "runtime_label_access": False,
            "runtime_parent_access": False,
            "runtime_target_access": False,
        },
    ]
    audit = {
        "label_blind": True,
        "samples": int(len(values)),
        "channels": 16,
        "affine_r2_by_channel": r2,
        "affine_r2_median": float(np.median(r2)),
        "analytic_definition": "frozen quantizer scale times local Walsh parity sign",
        "affine_definition": "per-channel ridge affine map from implied motif parity moment",
    }
    return writers, audit


def save_writer_models(
    output: Path,
    writers: Sequence[dict[str, Any]],
    candidates: Sequence[dict[str, Any]],
    *,
    design_digest: str,
) -> dict[str, Any]:
    arrays: dict[str, np.ndarray] = {}
    writer_rows: list[dict[str, Any]] = []
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
        writer_rows.append(row)
    path = output / "LOCAL_MODELS.npz"
    _atomic_npz(path, **arrays)
    manifest = {
        "design_digest": design_digest,
        "model_sha256": _sha256(path),
        "allow_pickle": False,
        "writers": writer_rows,
        "candidates": [_json_model(candidate) for candidate in candidates],
    }
    _atomic_json(output / "LOCAL_MODELS.json", manifest)
    return manifest


def load_writer_models(output: Path, design_digest: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = _load_json(output / "LOCAL_MODELS.json")
    if manifest.get("design_digest") != design_digest:
        raise ValueError("local-model design digest mismatch")
    path = output / "LOCAL_MODELS.npz"
    if _sha256(path) != manifest.get("model_sha256"):
        raise ValueError("local-model archive hash mismatch")
    writers: list[dict[str, Any]] = []
    with np.load(path, allow_pickle=False) as arrays:
        for metadata in manifest["writers"]:
            writer = {key: value for key, value in metadata.items() if key != "array_keys"}
            for key, array_key in metadata["array_keys"].items():
                writer[key] = np.asarray(arrays[array_key])
            writers.append(writer)
    by_id = {writer["writer_id"]: writer for writer in writers}
    candidates = [
        {**row, "slope": by_id[row["writer_id"]]["slope"], "intercept": by_id[row["writer_id"]]["intercept"]}
        for row in manifest["candidates"]
    ]
    return writers, candidates


def quantize_channels(values: np.ndarray, scale: np.ndarray, bits: int = 4) -> tuple[np.ndarray, float]:
    """Signed per-channel boundary quantizer with an exact zero code."""

    array = np.asarray(values, dtype=np.float32)
    scales = np.asarray(scale, dtype=np.float32)
    if array.shape[-1] != len(scales):
        raise ValueError("the final dimension must match channel scales")
    qmax = (1 << (bits - 1)) - 1
    normalized = np.divide(array, scales, out=np.zeros_like(array), where=scales > 0.0)
    clipping = float(np.mean(np.abs(normalized) > 1.0))
    integers = np.clip(np.rint(normalized * qmax), -qmax, qmax)
    result = integers * (scales / np.float32(qmax))
    result[array == 0.0] = 0.0
    return result.astype(np.float32), clipping


def transport_field(field: np.ndarray, diffusion: float) -> np.ndarray:
    """One synchronous Moore-neighbour diffusion step on the torus."""

    values = np.asarray(field, dtype=np.float32)
    if values.ndim != 4:
        raise ValueError("field must have shape (sample, height, width, channel)")
    if not 0.0 <= diffusion <= 1.0:
        raise ValueError("diffusion must be in [0, 1]")
    if diffusion == 0.0:
        return values.copy()
    neighbours = np.zeros_like(values)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                neighbours += np.roll(values, shift=(dy, dx), axis=(1, 2))
    return ((1.0 - diffusion) * values + (diffusion / 8.0) * neighbours).astype(np.float32)


def write_local_field(
    field: np.ndarray,
    codes: np.ndarray,
    basis: np.ndarray,
    slope: np.ndarray,
    intercept: np.ndarray,
    *,
    retention: float,
    write_gain: float,
    scale: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Apply one local, label-blind parity writer update."""

    signs = np.asarray(basis, dtype=np.float32)[codes] * np.float32(math.sqrt(512.0))
    target = signs * np.asarray(slope, dtype=np.float32) + np.asarray(intercept, dtype=np.float32)
    result = np.float32(retention) * np.asarray(field, dtype=np.float32) + np.float32(write_gain) * target
    scales = np.asarray(scale, dtype=np.float32)
    clipping = float(np.mean(np.abs(result) > scales))
    return np.clip(result, -scales, scales).astype(np.float32), clipping


def local_energy_advantage(predicted: np.ndarray, field: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Local motif-energy gain; a uniform field is exactly the global Walsh reader."""

    states = np.asarray(predicted, dtype=np.bool_)
    values = np.asarray(field, dtype=np.float32)
    basis_values = np.asarray(basis, dtype=np.float32)
    if values.shape[:3] != states.shape or values.shape[-1] != basis_values.shape[1]:
        raise ValueError("visible state, local field, and Walsh basis shapes do not align")
    codes = motif3_codes(states)
    advantage = np.zeros(states.shape, dtype=np.float32)
    offsets = tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1))
    for bit, (dy, dx) in enumerate(offsets):
        affected_codes = np.roll(codes, shift=(dy, dx), axis=(1, 2))
        affected_field = np.roll(values, shift=(dy, dx), axis=(1, 2))
        current = np.einsum("...c,...c->...", affected_field, basis_values[affected_codes], optimize=True)
        flipped = np.einsum(
            "...c,...c->...",
            affected_field,
            basis_values[affected_codes ^ np.uint16(1 << bit)],
            optimize=True,
        )
        advantage += flipped - current
    return advantage


def apply_local_reader(
    predicted: np.ndarray,
    field: np.ndarray,
    basis: np.ndarray,
    uniforms: np.ndarray,
    strength: float,
) -> np.ndarray:
    result = np.asarray(predicted, dtype=np.bool_).copy()
    advantage = local_energy_advantage(result, field, basis)
    probability = np.float32(strength) * np.tanh(np.maximum(advantage, 0.0) / 9.0)
    result[np.asarray(uniforms) < probability] ^= True
    return result


def _repeat_histories(values: np.ndarray, replicates: int) -> np.ndarray:
    return np.concatenate(
        (
            np.repeat(values[0:1], replicates, axis=0),
            np.repeat(values[1:2], replicates, axis=0),
        ),
        axis=0,
    )


def _swap_histories(values: np.ndarray, replicates: int) -> np.ndarray:
    return np.concatenate((values[replicates:], values[:replicates]), axis=0)


def _local_uniforms(
    pair_id: str,
    purpose: str,
    generation: int,
    sweep: int,
    replicates: int,
) -> np.ndarray:
    return _paired_uniforms(
        pair_id,
        f"stage5-{purpose}-generation-{generation}",
        sweep,
        replicates,
    )


def patch_origins(
    pair_id: str,
    generation: int,
    replicates: int,
    *,
    height: int = 16,
    width: int = 16,
) -> np.ndarray:
    rng = np.random.default_rng(_hash_seed("stage5-patch-origin", pair_id, generation))
    half = np.stack(
        (rng.integers(0, height, size=replicates), rng.integers(0, width, size=replicates)),
        axis=1,
    ).astype(np.int16)
    return np.concatenate((half, half), axis=0)


def extract_patch(field: np.ndarray, patch_side: int, origins: np.ndarray) -> np.ndarray:
    values = np.asarray(field, dtype=np.float32)
    if patch_side not in (1, 2, 4):
        raise ValueError("local patch side must be 1, 2, or 4")
    if len(origins) != len(values):
        raise ValueError("one patch origin is required per sample")
    result = np.empty((len(values), patch_side, patch_side, values.shape[-1]), dtype=np.float32)
    for sample, (origin_y, origin_x) in enumerate(np.asarray(origins, dtype=np.int64)):
        ys = (origin_y + np.arange(patch_side)) % values.shape[1]
        xs = (origin_x + np.arange(patch_side)) % values.shape[2]
        result[sample] = values[sample][np.ix_(ys, xs)]
    return result


def embed_patch(
    payload: np.ndarray,
    origins: np.ndarray,
    *,
    height: int = 16,
    width: int = 16,
    mode: str = "contiguous",
) -> np.ndarray:
    values = np.asarray(payload, dtype=np.float32)
    if values.ndim != 4 or values.shape[1] != values.shape[2]:
        raise ValueError("payload must have shape (sample, side, side, channel)")
    if len(origins) != len(values):
        raise ValueError("one patch origin is required per sample")
    result = np.zeros((len(values), height, width, values.shape[-1]), dtype=np.float32)
    side = values.shape[1]
    for sample, (origin_y, origin_x) in enumerate(np.asarray(origins, dtype=np.int64)):
        if mode == "translated":
            origin_y = (origin_y + 5) % height
            origin_x = (origin_x + 7) % width
        if mode in ("contiguous", "translated"):
            for dy in range(side):
                for dx in range(side):
                    result[sample, (origin_y + dy) % height, (origin_x + dx) % width] = values[
                        sample, dy, dx
                    ]
        elif mode == "dispersed":
            for dy in range(side):
                for dx in range(side):
                    cell_index = dy * side + dx
                    for channel in range(values.shape[-1]):
                        position = (cell_index * 37 + channel * 17) % (height * width)
                        y = (origin_y + position // width) % height
                        x = (origin_x + position % width) % width
                        result[sample, y, x, channel] += values[sample, dy, dx, channel]
        else:
            raise ValueError(f"unknown patch embedding mode {mode!r}")
    return result


def _payload_summary(payload: np.ndarray, replicates: int) -> dict[str, float]:
    flat = np.asarray(payload, dtype=np.float32).reshape(2, replicates, -1)
    delta = flat[0].mean(axis=0) - flat[1].mean(axis=0)
    return {
        "mean_abs": float(np.mean(np.abs(payload))),
        "centroid_l2": float(np.linalg.norm(delta)),
        "within_history_variance": float(np.mean(np.var(flat, axis=1))),
        "nonzero_fraction": float(np.mean(np.asarray(payload) != 0.0)),
    }


def _apply_local_boundary_intervention(
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
            _hash_seed("stage5-channel-shuffle", pair_id, generation)
        ).permutation(result.shape[-1])
        result = result[..., permutation]
    elif condition == "spatial_shuffle_every_boundary":
        flat = result.reshape(len(result), -1, result.shape[-1])
        permutation = np.random.default_rng(
            _hash_seed("stage5-spatial-shuffle", pair_id, generation)
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
            _hash_seed("stage5-carrier-corruption", pair_id, generation)
        ).random((replicates, *result.shape[1:])) < 0.01
        result[np.concatenate((half, half), axis=0)] *= -1.0
    elif condition == "half_width_bottleneck":
        flat = result.reshape(len(result), -1)
        flat[:, 1::2] = 0.0
        result = flat.reshape(result.shape)
    elif condition in ("recombine_first_half", "recombine_second_half"):
        swapped = _swap_histories(result, replicates)
        channel_slice = slice(0, 8) if condition == "recombine_first_half" else slice(8, 16)
        result[..., channel_slice] = swapped[..., channel_slice]
    return quantize_channels(result, scale)


def _damage_local_payload(
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
            _hash_seed("stage5-damage", pair_id, candidate_id, stress_id, generation, "erase")
        ).random((replicates, *result.shape[1:])) < erasure
        result[np.concatenate((half, half), axis=0)] = 0.0
    if sign:
        half = np.random.default_rng(
            _hash_seed("stage5-damage", pair_id, candidate_id, stress_id, generation, "sign")
        ).random((replicates, *result.shape[1:])) < sign
        result[np.concatenate((half, half), axis=0)] *= -1.0
    return quantize_channels(result, scale)


def calibrate_local_dynamics(
    writers: Sequence[dict[str, Any]],
    basis: np.ndarray,
    scale: np.ndarray,
) -> dict[str, Any]:
    """Outcome-blind synthetic impulse/mosaic calibration."""

    basis_values = np.asarray(basis, dtype=np.float32)
    scales = np.asarray(scale, dtype=np.float32)
    codes_rng = np.random.default_rng(_hash_seed("stage5-outcome-blind-calibration", 1))
    code_sequence = codes_rng.integers(0, 512, size=(16, 1, 16, 16), dtype=np.uint16)
    rows: list[dict[str, Any]] = []
    for writer in writers:
        for diffusion in (0.05, 0.10, 0.20, 0.35):
            impulse = np.zeros((1, 16, 16, 16), dtype=np.float32)
            impulse[0, 8, 8] = scales
            initial_mass = impulse.sum(axis=(1, 2))
            transported = impulse.copy()
            for _ in range(32):
                transported = transport_field(transported, diffusion)
            mass_error = float(
                np.max(np.abs(transported.sum(axis=(1, 2)) - initial_mass))
                / max(float(np.max(np.abs(initial_mass))), 1e-9)
            )
            site_norm = np.linalg.norm(transported[0] / scales, axis=-1)
            reach_fraction = float(np.mean(site_norm > 1e-3))
            for write_gain in (0.08, 0.16, 0.32):
                for retention in (0.50, 0.75, 1.00):
                    field = np.zeros_like(impulse)
                    clipping: list[float] = []
                    target_means: list[np.ndarray] = []
                    for sweep_codes in code_sequence:
                        field = transport_field(field, diffusion)
                        signs = basis_values[sweep_codes] * np.float32(math.sqrt(512.0))
                        target = signs * writer["slope"] + writer["intercept"]
                        target_means.append(target.mean(axis=(0, 1, 2)))
                        field, clip = write_local_field(
                            field,
                            sweep_codes,
                            basis_values,
                            writer["slope"],
                            writer["intercept"],
                            retention=retention,
                            write_gain=write_gain,
                            scale=scales,
                        )
                        clipping.append(clip)
                    observed = field.mean(axis=(0, 1, 2))
                    expected = np.mean(np.stack(target_means), axis=0)
                    norm_error = float(
                        np.linalg.norm(observed - expected)
                        / max(float(np.linalg.norm(expected)), float(np.mean(scales)), 1e-9)
                    )
                    if np.std(observed) > 0.0 and np.std(expected) > 0.0:
                        correlation = float(np.corrcoef(observed, expected)[0, 1])
                    else:
                        correlation = 0.0
                    clipping_mean = float(np.mean(clipping))
                    score = (
                        correlation
                        - 0.35 * norm_error
                        - 0.75 * clipping_mean
                        + 0.20 * reach_fraction
                        - 2.0 * mass_error
                    )
                    rows.append(
                        {
                            "writer_id": writer["writer_id"],
                            "diffusion": diffusion,
                            "write_gain": write_gain,
                            "retention": retention,
                            "impulse_reach_fraction_32": reach_fraction,
                            "mass_conservation_error": mass_error,
                            "mosaic_channel_correlation": correlation,
                            "mosaic_normalized_error": norm_error,
                            "writer_clipping_fraction": clipping_mean,
                            "finite": bool(np.isfinite(field).all()),
                            "score": score if np.isfinite(score) else -math.inf,
                        }
                    )
    selected: list[dict[str, Any]] = []
    for writer in writers:
        eligible = [row for row in rows if row["writer_id"] == writer["writer_id"] and row["finite"]]
        ranked = sorted(
            eligible,
            key=lambda row: (
                -float(row["score"]),
                float(row["diffusion"]),
                float(row["write_gain"]),
                float(row["retention"]),
            ),
        )
        selected.append(ranked[0])
        selected.append(
            next(row for row in ranked[1:] if row["diffusion"] != ranked[0]["diffusion"])
        )
    return {
        "label_and_outcome_blind": True,
        "grid_size": len(rows),
        "grid": rows,
        "selected": selected,
    }


def build_local_candidates(
    writers: Sequence[dict[str, Any]],
    calibration: dict[str, Any],
) -> list[dict[str, Any]]:
    by_id = {writer["writer_id"]: writer for writer in writers}
    candidates: list[dict[str, Any]] = []
    for row in calibration["selected"]:
        writer = by_id[row["writer_id"]]
        for patch_side in (1, 2, 4):
            candidate_id = (
                f"local-{row['writer_id']}-p{patch_side}"
                f"-d{int(round(row['diffusion'] * 100)):02d}"
                f"-w{int(round(row['write_gain'] * 100)):02d}"
                f"-r{int(round(row['retention'] * 100)):03d}"
            )
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "kind": "local",
                    "writer_id": row["writer_id"],
                    "patch_side": patch_side,
                    "diffusion": float(row["diffusion"]),
                    "write_gain": float(row["write_gain"]),
                    "retention": float(row["retention"]),
                    "calibration_score": float(row["score"]),
                    "rank": patch_side * patch_side * 16,
                    "bits": 4,
                    "payload_bits": patch_side * patch_side * 16 * 4,
                    "developmental_field_values": 16 * 16 * 16,
                    "developmental_field_storage_bits_float32": 16 * 16 * 16 * 32,
                    "shared_stage4_codebook_bits": 656,
                    "shared_writer_parameter_bits": 0 if row["writer_id"] == "analytic" else 16 * 2 * 32,
                    "slope": writer["slope"],
                    "intercept": writer["intercept"],
                    "runtime_label_access": False,
                    "runtime_parent_access": False,
                    "runtime_target_access": False,
                    "locality_audit": True,
                }
            )
    candidates = sorted(
        candidates,
        key=lambda candidate: (
            -float(candidate["calibration_score"]),
            int(candidate["payload_bits"]),
            str(candidate["candidate_id"]),
        ),
    )[:12]
    if len(candidates) > 16:
        raise AssertionError("no more than 16 local candidates may advance")
    return candidates


def _founder_local_payload(
    pair: dict[str, Any],
    candidate: dict[str, Any],
    basis: np.ndarray,
    scale: np.ndarray,
    replicates: int,
    rule: int,
) -> np.ndarray:
    state = _repeat_histories(_founders(pair), replicates)
    field = np.zeros((*state.shape, 16), dtype=np.float32)
    for _sweep in range(1, 33):
        state = _step(state, rule)
        field = transport_field(field, float(candidate["diffusion"]))
        field, _ = write_local_field(
            field,
            motif3_codes(state),
            basis,
            candidate["slope"],
            candidate["intercept"],
            retention=float(candidate["retention"]),
            write_gain=float(candidate["write_gain"]),
            scale=scale,
        )
    origins = patch_origins(str(pair["pair_id"]), 1, replicates)
    payload = extract_patch(field, int(candidate["patch_side"]), origins)
    return quantize_channels(payload, scale)[0]


def simulate_local_lineage(
    pair: dict[str, Any],
    configuration: ReaderConfiguration,
    candidate: dict[str, Any],
    walsh_model: dict[str, Any],
    condition: str,
    replicates: int,
    generations: int,
    reference: dict[int, dict[str, np.ndarray]],
    writer_contract: MotifContract,
    contract: LocalizationContract,
    *,
    stress_id: str = "ordinary",
    stress: dict[str, float | int] | None = None,
    source_exits: Sequence[np.ndarray] | None = None,
    retain_exits: bool = False,
    rule_override: int | None = None,
) -> tuple[dict[str, Any], list[np.ndarray]]:
    """Simulate renewal when only a local Walsh-field patch crosses boundaries."""

    if condition not in CORE_CONDITIONS + LOCAL_EXTRA_CONDITIONS:
        raise ValueError(f"unknown Stage-5 condition {condition!r}")
    stress = dict(stress or {})
    pair_id = str(pair["pair_id"])
    rule = int(rule_override if rule_override is not None else contract.rule)
    basis = np.asarray(walsh_model["basis"], dtype=np.float32)
    scale = np.asarray(walsh_model["quantizer_scale"], dtype=np.float32)
    reset_state = _state_from_hex("life", pair["donor_a"]["initial_state_hex"])
    other_reset = _state_from_hex("life", pair["donor_b"]["initial_state_hex"])
    if not np.array_equal(reset_state, other_reset):
        raise AssertionError(f"visible reset mismatch in pair {pair_id}")
    reset = np.repeat(reset_state[None, ...], 2 * replicates, axis=0)
    founder_writer = replace(writer_contract, rule=rule)
    written = write_parent_carriers(
        _founders(pair), (configuration.write_window,), reference, founder_writer
    )[configuration.write_window]
    founder_terminal = written["terminal"]
    payload = _founder_local_payload(pair, candidate, basis, scale, replicates, rule)
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
    writer_clipping_values: list[float] = []
    process_noise = float(stress.get("process_noise", contract.process_noise))
    diffusion = 0.0 if condition == "transport_disabled" else float(candidate["diffusion"])

    for generation in range(1, generations + 1):
        payload, clipping = _apply_local_boundary_intervention(
            payload, condition, generation, pair_id, replicates, source_exits, scale
        )
        clipping_values.append(clipping)
        payload, clipping = _damage_local_payload(
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
        embedding_mode = (
            "translated"
            if condition == "translated_patch"
            else "dispersed"
            if condition == "dispersed_equal_bits"
            else "contiguous"
        )
        field = embed_patch(entry_payload, origins, mode=embedding_mode)
        state = reset.copy()
        state[~alive] = False
        if not np.array_equal(state[alive], reset[alive]):
            raise AssertionError("visible reset was not bitwise identical")
        recent: deque[np.ndarray] = deque(maxlen=writer_contract.observation_window)
        for sweep in range(1, contract.generation_sweeps + 1):
            if sweep == contract.write_start and condition not in ("no_rewrite",):
                field.fill(0.0)
            field = transport_field(field, diffusion)
            predicted = _step(state, rule)
            if condition != "read_disabled" and sweep <= contract.read_sweeps:
                predicted = apply_local_reader(
                    predicted,
                    field,
                    basis,
                    _local_uniforms(pair_id, "read", generation, sweep, replicates),
                    configuration.strength,
                )
            predicted ^= (
                _local_uniforms(pair_id, "process", generation, sweep, replicates)
                < process_noise
            )
            predicted[~alive] = False
            state = predicted
            if (
                contract.write_start <= sweep <= contract.write_end
                and condition not in ("no_rewrite", "write_disabled")
            ):
                field, writer_clip = write_local_field(
                    field,
                    motif3_codes(state),
                    basis,
                    candidate["slope"],
                    candidate["intercept"],
                    retention=float(candidate["retention"]),
                    write_gain=float(candidate["write_gain"]),
                    scale=scale,
                )
                writer_clipping_values.append(writer_clip)
            if sweep >= contract.observe_start:
                recent.append(live_2x2_counts_batch(state))
        alive &= state.any(axis=(1, 2))
        if condition == "no_rewrite":
            payload, clipping = quantize_channels(
                entry_payload * np.float32(contract.stale_retention), scale
            )
        elif condition == "write_disabled":
            payload = np.zeros_like(entry_payload)
            clipping = 0.0
        else:
            next_origins = patch_origins(pair_id, generation + 1, replicates)
            payload, clipping = quantize_channels(
                extract_patch(field, int(candidate["patch_side"]), next_origins), scale
            )
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
            "developmental_field_values": int(candidate["developmental_field_values"]),
            "developmental_field_storage_bits_float32": int(
                candidate["developmental_field_storage_bits_float32"]
            ),
            "founder_payload": _payload_summary(founder_payload, replicates),
            "boundary_clipping_fraction_mean": float(np.mean(clipping_values)) if clipping_values else 0.0,
            "writer_clipping_fraction_mean": float(np.mean(writer_clipping_values)) if writer_clipping_values else 0.0,
            "outcomes": outcomes,
            "decoders": decoders,
            "carrier_history": carrier_history,
        },
        exits,
    )


def _d4_code_map(rotation: int, reflect: bool) -> np.ndarray:
    result = np.empty(512, dtype=np.int64)
    for code in range(512):
        source = np.asarray([(code >> bit) & 1 for bit in range(9)], dtype=np.uint8).reshape(3, 3)
        target = np.rot90(source, rotation)
        if reflect:
            target = np.fliplr(target)
        result[code] = sum(int(value) << bit for bit, value in enumerate(target.ravel()))
    return result


def build_anatomy_models(walsh_model: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the preregistered, deterministic Walsh lesion/orientation atlas."""

    basis = np.asarray(walsh_model["basis"], dtype=np.float32)
    scale = np.asarray(walsh_model["quantizer_scale"], dtype=np.float32)

    def variant(candidate_id: str, indices: Sequence[int], *, transformed_basis: np.ndarray | None = None, equal_scale: bool = False) -> dict[str, Any]:
        chosen = np.asarray(indices, dtype=np.int64)
        selected_basis = basis[:, chosen] if transformed_basis is None else transformed_basis[:, chosen]
        selected_scale = scale[chosen].copy()
        if equal_scale:
            selected_scale.fill(float(np.median(scale)))
        return {
            "candidate_id": candidate_id,
            "family": "walsh-anatomy",
            "rank": int(len(chosen)),
            "bits": 4,
            "precision": "fixed-scale-signed",
            "payload_bits": int(len(chosen) * 4),
            "codebook_bits": int(len(chosen) * 9 + len(chosen) * 32),
            "interpretable": True,
            "runtime_label_access": False,
            "runtime_parent_access": False,
            "runtime_target_access": False,
            "basis": selected_basis.astype(np.float32),
            "quantizer_scale": selected_scale.astype(np.float32),
        }

    models = [variant("anatomy-full16", range(16))]
    models.extend(
        variant(f"anatomy-loo-{channel:02d}", [index for index in range(16) if index != channel])
        for channel in range(16)
    )
    models.extend(
        variant(f"anatomy-single-{channel:02d}", [channel])
        for channel in range(16)
    )
    rng = np.random.default_rng(_hash_seed("stage5-balanced-walsh-masks", 1))
    masks: set[tuple[int, ...]] = set()
    while len(masks) < 32:
        masks.add(tuple(sorted(int(value) for value in rng.choice(16, size=8, replace=False))))
    models.extend(
        variant(f"anatomy-balanced-{index:02d}", mask)
        for index, mask in enumerate(sorted(masks))
    )
    for rotation in range(4):
        for reflect in (False, True):
            mapping = _d4_code_map(rotation, reflect)
            transformed = basis[mapping]
            models.append(
                variant(
                    f"anatomy-d4-r{rotation}-f{int(reflect)}",
                    range(16),
                    transformed_basis=transformed,
                )
            )
    models.append(variant("anatomy-equal-scale", range(16), equal_scale=True))
    if len(models) != 74 or len({model["candidate_id"] for model in models}) != 74:
        raise AssertionError("the registered anatomy atlas must contain 74 unique variants")
    return models


def transcode_audit(walsh_model: dict[str, Any]) -> dict[str, Any]:
    """Prove uniform local fields reproduce the frozen global Walsh reader."""

    basis = np.asarray(walsh_model["basis"], dtype=np.float32)
    scale = np.asarray(walsh_model["quantizer_scale"], dtype=np.float32)
    rng = np.random.default_rng(_hash_seed("stage5-transcode-audit", 1))
    predicted = rng.random((4, 16, 16)) < 0.37
    latent = rng.uniform(-1.0, 1.0, size=(4, 16)).astype(np.float32) * scale
    latent = quantize_channels(latent, scale)[0]
    field = np.broadcast_to(latent[:, None, None, :], (4, 16, 16, 16)).copy()
    global_carrier = decode_payload(latent, walsh_model)
    global_advantage = motif_energy_advantage(predicted, global_carrier)
    local_advantage = local_energy_advantage(predicted, field, basis)
    maximum_error = float(np.max(np.abs(global_advantage - local_advantage)))

    origins = patch_origins("transcode-audit", 1, 2)
    sample_patch = quantize_channels(
        rng.uniform(-1.0, 1.0, size=(4, 4, 4, 16)).astype(np.float32) * scale,
        scale,
    )[0]
    roundtrip = extract_patch(embed_patch(sample_patch, origins), 4, origins)

    impulse = np.zeros((1, 16, 16, 1), dtype=np.float32)
    impulse[0, 8, 8, 0] = 1.0
    transported = impulse
    for _ in range(3):
        transported = transport_field(transported, 0.2)
    support = np.argwhere(np.abs(transported[0, ..., 0]) > 0.0)
    distances = [
        max(min(abs(int(y) - 8), 16 - abs(int(y) - 8)), min(abs(int(x) - 8), 16 - abs(int(x) - 8)))
        for y, x in support
    ]
    zero_state = np.zeros((2, 16, 16), dtype=np.bool_)
    zero_field = np.zeros((2, 16, 16, 16), dtype=np.float32)
    zero_uniforms = np.zeros((2, 16, 16), dtype=np.float64)
    zero_inert = np.array_equal(
        apply_local_reader(zero_state, zero_field, basis, zero_uniforms, 0.25), zero_state
    )
    return {
        "uniform_field_global_equivalence_max_abs_error": maximum_error,
        "uniform_field_global_equivalence": maximum_error <= 2e-5,
        "boundary_roundtrip_exact": bool(np.array_equal(roundtrip, sample_patch)),
        "zero_field_exactly_inert": bool(zero_inert),
        "three_step_light_cone_max_chebyshev_distance": max(distances, default=0),
        "three_step_light_cone_pass": max(distances, default=0) <= 3,
        "reader_neighbourhood": "nine affected 3x3 motif sites only",
        "writer_neighbourhood": "current 3x3 motif at the same site only",
        "transport_neighbourhood": "eight nearest Moore neighbours only",
        "passed": bool(
            maximum_error <= 2e-5
            and np.array_equal(roundtrip, sample_patch)
            and zero_inert
            and max(distances, default=0) <= 3
        ),
    }


def _global_candidate(walsh_model: dict[str, Any]) -> dict[str, Any]:
    stage4_model = dict(walsh_model)
    stage4_model["candidate_id"] = GLOBAL_ANCHOR_ID
    return {
        "candidate_id": GLOBAL_ANCHOR_ID,
        "kind": "global",
        "rank": 16,
        "bits": 4,
        "payload_bits": 64,
        "shared_stage4_codebook_bits": int(walsh_model["codebook_bits"]),
        "shared_writer_parameter_bits": 0,
        "boundary_scope": "uniform global broadcast diagnostic",
        "locality_audit": False,
        "stage4_model": stage4_model,
    }


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
    contract: LocalizationContract,
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
    return simulate_local_lineage(
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


def _anatomy_pair_task(
    payload: tuple[
        dict[str, Any],
        list[dict[str, Any]],
        MotifContract,
        LocalizationContract,
        dict[int, dict[str, np.ndarray]],
    ]
) -> dict[str, Any]:
    item, models, writer_contract, contract, reference = payload
    configuration = ReaderConfiguration(**item["configuration"])
    candidates: dict[str, Any] = {}
    for model in models:
        result, _ = simulate_compressed_lineage(
            item["pair"],
            configuration,
            model,
            "intact",
            int(item["replicates"]),
            int(item["generations"]),
            reference,
            writer_contract,
            CompressionContract(),
        )
        candidates[str(model["candidate_id"])] = result
    return {
        "checkpoint": item["checkpoint"],
        "pair_id": item["pair"]["pair_id"],
        "candidates": candidates,
    }


def _screen_pair_task(
    payload: tuple[
        dict[str, Any],
        list[dict[str, Any]],
        MotifContract,
        LocalizationContract,
        dict[int, dict[str, np.ndarray]],
    ]
) -> dict[str, Any]:
    item, candidates, writer_contract, contract, reference = payload
    configuration = ReaderConfiguration(**item["configuration"])
    selected = [
        candidate
        for candidate in candidates
        if item.get("candidate_id") is None or candidate["candidate_id"] == item["candidate_id"]
    ]
    results: dict[str, Any] = {}
    walsh_model = item["walsh_model"]
    for candidate in selected:
        result, _ = _simulate_candidate(
            item["pair"],
            configuration,
            candidate,
            walsh_model,
            "intact",
            int(item["replicates"]),
            int(item["generations"]),
            reference,
            writer_contract,
            contract,
        )
        results[str(candidate["candidate_id"])] = result
    return {
        "checkpoint": item["checkpoint"],
        "pair_id": item["pair"]["pair_id"],
        "candidates": results,
    }


def _qualification_pair_task(
    payload: tuple[
        dict[str, Any],
        list[dict[str, Any]],
        MotifContract,
        LocalizationContract,
        dict[int, dict[str, np.ndarray]],
    ]
) -> dict[str, Any]:
    item, candidates, writer_contract, contract, reference = payload
    configuration = ReaderConfiguration(**item["configuration"])
    candidate = next(value for value in candidates if value["candidate_id"] == item["candidate_id"])
    walsh_model = item["walsh_model"]
    replicates = int(item["replicates"])
    generations = int(item["generations"])
    intact, exits = _simulate_candidate(
        item["pair"],
        configuration,
        candidate,
        walsh_model,
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
            walsh_model,
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
        LocalizationContract,
        dict[int, dict[str, np.ndarray]],
    ]
) -> dict[str, Any]:
    item, candidates, writer_contract, contract, reference = payload
    configuration = ReaderConfiguration(**item["configuration"])
    candidate = next(value for value in candidates if value["candidate_id"] == item["candidate_id"])
    rule = int(item["pair"]["stage5_transfer_rule"])
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
        LocalizationContract,
        dict[int, dict[str, np.ndarray]],
    ]
) -> dict[str, Any]:
    item, candidates, writer_contract, contract, reference = payload
    configuration = ReaderConfiguration(**item["configuration"])
    candidate = next(value for value in candidates if value["candidate_id"] == item["candidate_id"])
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
    rows: Sequence[dict[str, Any]],
    candidate_id: str,
    generation: int,
    metric: str,
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


def summarize_anatomy(
    rows: Sequence[dict[str, Any]],
    models: Sequence[dict[str, Any]],
    profile: LocalizationProfile,
    contract: LocalizationContract,
    complete: bool,
) -> dict[str, Any]:
    if not complete:
        return {"state": "incomplete", "variants": {}}
    generation = min(8, profile.anatomy_generations)
    summaries: dict[str, Any] = {}
    for model in models:
        candidate_id = str(model["candidate_id"])
        values = _outcome_values(rows, candidate_id, generation, "crossover")
        summaries[candidate_id] = {
            "rank": int(model["rank"]),
            "payload_bits": int(model["payload_bits"]),
            "crossover": _bootstrap(
                values,
                profile.bootstrap_resamples,
                _hash_seed(contract.namespace, "anatomy", candidate_id, generation),
                contract.strict_alpha,
            ),
            "survival_mean": float(np.mean(_outcome_values(rows, candidate_id, generation, "survival"))),
        }
    full = float(summaries["anatomy-full16"]["crossover"]["mean"] or 0.0)
    leave_one_out = {
        str(channel): full - float(summaries[f"anatomy-loo-{channel:02d}"]["crossover"]["mean"] or 0.0)
        for channel in range(16)
    }
    singles = {
        str(channel): float(summaries[f"anatomy-single-{channel:02d}"]["crossover"]["mean"] or 0.0)
        for channel in range(16)
    }
    d4 = {
        candidate_id: float(summary["crossover"]["mean"] or 0.0)
        for candidate_id, summary in summaries.items()
        if candidate_id.startswith("anatomy-d4-")
    }
    importance = np.asarray(list(leave_one_out.values()), dtype=np.float64)
    concentration = (
        float(np.max(np.abs(importance)) / np.sum(np.abs(importance)))
        if np.sum(np.abs(importance)) > 0.0
        else 0.0
    )
    return {
        "state": "complete",
        "generation": generation,
        "full16_crossover_mean": full,
        "leave_one_out_effect_loss": leave_one_out,
        "single_mode_crossover": singles,
        "d4_crossover": d4,
        "leave_one_out_importance_concentration": concentration,
        "distributed_across_modes": bool(concentration < 0.50),
        "variants": summaries,
    }


def adjudicate_localization_screen(
    rows: Sequence[dict[str, Any]],
    candidates: Sequence[dict[str, Any]],
    profile: LocalizationProfile,
    contract: LocalizationContract,
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
            _hash_seed(contract.namespace, "localize", candidate_id, generation),
            contract.strict_alpha,
        )
        summaries[candidate_id] = {
            "candidate": _candidate_metadata(candidate),
            "crossover": crossover,
            "survival_mean": float(np.mean(_outcome_values(rows, candidate_id, generation, "survival"))),
            "direction_a_mean": float(np.mean(_outcome_values(rows, candidate_id, generation, "direction_a"))),
            "direction_b_mean": float(np.mean(_outcome_values(rows, candidate_id, generation, "direction_b"))),
            "fraction_pairs_positive": float(np.mean(np.asarray(crossover_values) > 0.0)),
        }
    anchor_mean = float(summaries[GLOBAL_ANCHOR_ID]["crossover"]["mean"] or 0.0)
    locals_only = [candidate for candidate in candidates if candidate["kind"] == "local"]
    for candidate in locals_only:
        candidate_id = str(candidate["candidate_id"])
        summary = summaries[candidate_id]
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
    if profile.screen_generations < 8:
        eligible = locals_only
    else:
        eligible = [candidate for candidate in locals_only if summaries[candidate["candidate_id"]]["screen_eligible"]]
    if not eligible:
        eligible = sorted(
            locals_only,
            key=lambda candidate: (
                -float(summaries[candidate["candidate_id"]]["crossover"]["mean"] or -1.0),
                int(candidate["payload_bits"]),
                str(candidate["candidate_id"]),
            ),
        )[:1]

    def effect(candidate: dict[str, Any]) -> float:
        return float(summaries[candidate["candidate_id"]]["crossover"]["mean"] or -1.0)

    choices: list[dict[str, Any]] = []
    one_site = [candidate for candidate in eligible if int(candidate["patch_side"]) == 1]
    if one_site:
        choices.append(min(one_site, key=lambda candidate: (-effect(candidate), candidate["candidate_id"])))
    choices.append(min(eligible, key=lambda candidate: (int(candidate["payload_bits"]), -effect(candidate), candidate["candidate_id"])))
    choices.append(min(eligible, key=lambda candidate: (-effect(candidate), int(candidate["payload_bits"]), candidate["candidate_id"])))
    for writer_id in ("analytic", "affine"):
        writer_candidates = [candidate for candidate in eligible if candidate["writer_id"] == writer_id]
        if writer_candidates:
            choices.append(min(writer_candidates, key=lambda candidate: (-effect(candidate), candidate["candidate_id"])))
    selected_local = list(dict.fromkeys(str(candidate["candidate_id"]) for candidate in choices))[:4]
    return {
        "state": "complete",
        "generation": generation,
        "anchor_crossover_mean": anchor_mean,
        "candidate_summaries": summaries,
        "selected_candidate_ids": [GLOBAL_ANCHOR_ID, *selected_local],
        "scientific_gate_applied": profile.screen_generations >= 8,
        "fallback_nomination_used": not any(
            summaries[candidate["candidate_id"]].get("screen_eligible", False)
            for candidate in locals_only
        ),
    }


def _repair_profile(
    profile: LocalizationProfile,
    role: str,
    *,
    confirmation: bool = False,
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
            outcome = row["candidates"][candidate_id]["conditions"][condition]["outcomes"][str(generation)]
            value = outcome["survival"] if metric == "survival" else outcome["primary"][metric]
        except KeyError:
            continue
        values.append(float(value))
    return values


def _condition_advantage(
    rows: Sequence[dict[str, Any]],
    candidate_id: str,
    control: str,
    generation: int,
) -> list[float]:
    left = _condition_values(rows, candidate_id, "intact", generation)
    right = _condition_values(rows, candidate_id, control, generation)
    if len(left) != len(right):
        raise ValueError(f"paired Stage-5 metrics do not align for {control}")
    return [a - b for a, b in zip(left, right)]


def _positive_gate(summary: dict[str, Any], minimum: float) -> bool:
    return bool(
        summary["mean"] is not None
        and float(summary["mean"]) >= minimum
        and summary["ci"][0] is not None
        and float(summary["ci"][0]) > 0.0
    )


def _local_extra_gate(
    rows: Sequence[dict[str, Any]],
    candidate: dict[str, Any],
    profile: LocalizationProfile,
    contract: LocalizationContract,
    alpha: float,
) -> dict[str, Any]:
    candidate_id = str(candidate["candidate_id"])

    def boot(values: Sequence[float], name: str) -> dict[str, Any]:
        return _bootstrap(
            values,
            profile.bootstrap_resamples,
            _hash_seed(contract.namespace, "local-extra", candidate_id, name),
            alpha,
        )

    generation = 8
    channel = boot(_condition_advantage(rows, candidate_id, "shuffle_every_boundary", generation), "channel")
    transport = boot(_condition_advantage(rows, candidate_id, "transport_disabled", generation), "transport")
    spatial = boot(_condition_advantage(rows, candidate_id, "spatial_shuffle_every_boundary", generation), "spatial")
    intact = boot(_condition_values(rows, candidate_id, "intact", generation), "intact")
    translated = boot(_condition_values(rows, candidate_id, "translated_patch", generation), "translated")
    intact_mean = float(intact["mean"] or 0.0)
    translation_retention = (
        float(translated["mean"] or 0.0) / intact_mean if intact_mean > 0.0 else 0.0
    )
    passed = bool(
        _positive_gate(channel, contract.control_advantage)
        and _positive_gate(transport, contract.transport_advantage)
        and translated["ci"][0] is not None
        and float(translated["ci"][0]) > 0.0
        and translation_retention >= contract.translation_retention
        and candidate.get("locality_audit") is True
        and int(candidate["payload_bits"]) == int(candidate["patch_side"]) ** 2 * 16 * 4
    )
    return {
        "channel_shuffle_advantage": channel,
        "transport_disabled_advantage": transport,
        "spatial_shuffle_advantage_diagnostic": spatial,
        "translated_patch": translated,
        "translation_retention": translation_retention,
        "light_cone_and_runtime_locality_audit": bool(candidate.get("locality_audit")),
        "payload_accounting_audit": int(candidate["payload_bits"]) == int(candidate["patch_side"]) ** 2 * 16 * 4,
        "passed": passed,
    }


def adjudicate_local_qualification(
    rows: Sequence[dict[str, Any]],
    candidates: Sequence[dict[str, Any]],
    profile: LocalizationProfile,
    contract: LocalizationContract,
    complete: bool,
) -> dict[str, Any]:
    if not complete:
        return {"state": "incomplete", "qualified_candidate_ids": [], "candidates": {}}
    gate_applied = profile.qualification_generations >= 16
    summaries: dict[str, Any] = {}
    qualified: list[str] = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        candidate_rows = [row for row in rows if candidate_id in row.get("candidates", {})]
        if gate_applied:
            strict = _strict_confirmation_gate(
                candidate_rows,
                candidate_id,
                _repair_profile(profile, "stage5-qualification"),
                contract,  # type: ignore[arg-type]
                contract.strict_alpha,
            )
        else:
            strict = {"verdict": "NOT_ADJUDICATED_PROFILE", "renewed_gate": False}
        if candidate["kind"] == "local" and gate_applied:
            local = _local_extra_gate(candidate_rows, candidate, profile, contract, contract.strict_alpha)
        elif candidate["kind"] == "local":
            local = {"passed": False, "scientific_gate_applied": False}
        else:
            local = {"passed": False, "not_applicable_global_anchor": True}
        renewed = bool(strict.get("renewed_gate"))
        local_pass = bool(renewed and local.get("passed"))
        summaries[candidate_id] = {
            "candidate": _candidate_metadata(candidate),
            "strict": strict,
            "localization": local,
            "local_renewed_gate": local_pass,
        }
        if (not gate_applied) or (candidate["kind"] == "global" and renewed) or local_pass:
            qualified.append(candidate_id)
    return {
        "state": "complete",
        "scientific_gate_applied": gate_applied,
        "candidate_summaries": summaries,
        "qualified_candidate_ids": qualified,
    }


def select_stage5_finalists(
    candidates: Sequence[dict[str, Any]],
    qualification: dict[str, Any],
    qualification_rows: Sequence[dict[str, Any]],
    profile: LocalizationProfile,
) -> list[str]:
    by_id = {str(candidate["candidate_id"]): candidate for candidate in candidates}
    local_candidates = [candidate for candidate in candidates if candidate["kind"] == "local"]
    qualified_ids = set(qualification.get("qualified_candidate_ids", ()))
    qualified = [candidate for candidate in local_candidates if candidate["candidate_id"] in qualified_ids]
    generation = min(16, profile.qualification_generations)

    def effect(candidate: dict[str, Any]) -> float:
        values = _condition_values(
            qualification_rows, str(candidate["candidate_id"]), "intact", generation
        )
        return float(np.mean(values)) if values else -1.0

    choices: list[dict[str, Any]] = []
    one_site_pool = [candidate for candidate in qualified if int(candidate["patch_side"]) == 1]
    if not one_site_pool:
        one_site_pool = [candidate for candidate in local_candidates if int(candidate["patch_side"]) == 1]
    if one_site_pool:
        choices.append(min(one_site_pool, key=lambda candidate: (-effect(candidate), candidate["candidate_id"])))
    fallback = [candidate for candidate in qualified if candidate not in choices]
    if fallback:
        choices.append(
            min(
                fallback,
                key=lambda candidate: (
                    int(candidate["payload_bits"]),
                    -effect(candidate),
                    candidate["candidate_id"],
                ),
            )
        )
    return [GLOBAL_ANCHOR_ID, *list(dict.fromkeys(candidate["candidate_id"] for candidate in choices))[:2]]


def summarize_transfer(
    rows: Sequence[dict[str, Any]],
    candidate_ids: Sequence[str],
    profile: LocalizationProfile,
    contract: LocalizationContract,
    complete: bool,
) -> dict[str, Any]:
    if not complete:
        return {"state": "incomplete", "rules": {}}
    generation = min(16, profile.transfer_generations)
    rules: dict[str, Any] = {}
    for rule in (31648, 70366):
        rule_rows = [row for row in rows if int(row["rule"]) == rule]
        values: dict[str, Any] = {}
        for candidate_id in candidate_ids:
            crossover = _outcome_values(rule_rows, candidate_id, generation, "crossover")
            values[candidate_id] = {
                "crossover": _bootstrap(
                    crossover,
                    profile.bootstrap_resamples,
                    _hash_seed(contract.namespace, "transfer", rule, candidate_id),
                    contract.strict_alpha,
                )
            }
        rules[str(rule)] = {"pairs": len(rule_rows), "candidates": values}
    return {"state": "complete", "generation": generation, "rules": rules, "exploratory_only": True}


def _candidate_metadata(candidate: dict[str, Any]) -> dict[str, Any]:
    ignored = {"slope", "intercept", "stage4_model"}
    return {key: value for key, value in candidate.items() if key not in ignored and not isinstance(value, np.ndarray)}


def adjudicate_stage5_confirmation(
    rows: Sequence[dict[str, Any]],
    candidates: Sequence[dict[str, Any]],
    profile: LocalizationProfile,
    contract: LocalizationContract,
    complete: bool,
) -> dict[str, Any]:
    if not complete or profile.confirmation_generations < 16:
        return {"state": "incomplete", "verdict": "INCOMPLETE", "candidates": {}}
    summaries: dict[str, Any] = {}
    passes: dict[str, dict[str, bool]] = {}
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        summaries[candidate_id] = {"candidate": _candidate_metadata(candidate), "environments": {}}
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
                _repair_profile(profile, f"stage5-confirm-{environment}", confirmation=True),
                contract,  # type: ignore[arg-type]
                contract.confirmation_alpha_per_object,
            )
            if candidate["kind"] == "local":
                local = _local_extra_gate(
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
                "localization": local,
                "stage5_gate": passed,
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
    robust64 = [candidate for candidate in local64 if passes[candidate["candidate_id"]]["moderate_joint"]]
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
        else "ROBUST_LOCAL_64BIT_RENEWED_CA_PLASTIC_HEREDITY"
        if robust64
        else "LOCAL_64BIT_RENEWED_CA_PLASTIC_HEREDITY"
        if local64
        else "LOCAL_DISTRIBUTED_RENEWED_CA_PLASTIC_HEREDITY"
        if distributed
        else "GLOBAL_BROADCAST_ONLY"
    )
    return {
        "state": "complete",
        "verdict": verdict,
        "fresh_stage4_anchor_replicated": anchor_pass,
        "robust_local_64bit_candidate_ids": [candidate["candidate_id"] for candidate in robust64],
        "local_64bit_candidate_ids": [candidate["candidate_id"] for candidate in local64],
        "local_distributed_candidate_ids": [candidate["candidate_id"] for candidate in distributed],
        "candidates": summaries,
        "claim_boundary": "synthetic CA lineage memory only; no metabolism, agency, or biological-life claim",
    }


def _selected_candidates(
    candidates: Sequence[dict[str, Any]], candidate_ids: Sequence[str]
) -> list[dict[str, Any]]:
    by_id = {str(candidate["candidate_id"]): candidate for candidate in candidates}
    missing = [candidate_id for candidate_id in candidate_ids if candidate_id not in by_id]
    if missing:
        raise ValueError(f"selected Stage-5 candidates are missing: {missing}")
    return [by_id[candidate_id] for candidate_id in candidate_ids]


def _queue(
    design_digest: str,
    state: str,
    *,
    finalists: Sequence[str] = (),
    verdict: str | None = None,
) -> dict[str, Any]:
    stage5: dict[str, Any] = {
        "stage": 5,
        "name": "physical_localization",
        "state": state,
        "confirmation_candidate_ids": list(finalists),
    }
    if verdict is not None:
        stage5["verdict"] = verdict
    return {
        "design_digest": design_digest,
        "automatic_launch": False,
        "stages": [
            {"stage": 1, "name": "motif_carrier_upper_bound", "state": "complete"},
            {"stage": 2, "name": "freeze_and_generalize_reader", "state": "complete"},
            {"stage": 3, "name": "renewed_heredity_causal_ladder", "state": "complete_negative"},
            {"stage": "3R", "name": "semantic_closure_and_repair", "state": "complete_positive"},
            {"stage": 4, "name": "compression_and_robustness", "state": "complete_positive"},
            stage5,
            {"stage": 6, "name": "unregistered_follow_up", "state": "blocked_pending_stage5_review"},
        ],
    }


def _render_preconfirmation_report(results: dict[str, Any]) -> str:
    decision = results["selection_decision"]
    qualification = results["qualification"]
    lines = [
        "# CA motif-lineage Stage 5: preconfirmation report",
        "",
        f"State: `{results['state']}`.",
        "",
        "The registered 16-channel Stage-4 Walsh carrier was rewritten as a local cellular field, passed through spatial bottlenecks, transported only by nearest-neighbour diffusion, and challenged with the full exposed-pair causal ladder. The sealed Stage-5 pairs have not been simulated.",
        "",
        f"Screen-selected objects: {len(results['screen'].get('selected_candidate_ids', []))}.",
        f"Strictly qualified objects: {len(qualification.get('qualified_candidate_ids', []))}.",
        f"Frozen confirmation objects: {', '.join(decision['confirmation_candidate_ids'])}.",
        f"Confirmation state: `{decision['confirmation_state']}`.",
        "",
        "## Frozen objects",
        "",
        "| Object | Scope | Inherited bits | Developmental working bits | Shared parameter bits | Qualification |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for candidate_id in decision["confirmation_candidate_ids"]:
        summary = qualification.get("candidate_summaries", {}).get(candidate_id, {})
        candidate = summary.get("candidate", decision["candidate_metadata"].get(candidate_id, {}))
        scope = "global diagnostic" if candidate.get("kind") == "global" else f"{candidate.get('patch_side')}x{candidate.get('patch_side')} local patch"
        inherited = int(candidate.get("payload_bits", 64))
        developmental = int(candidate.get("developmental_field_storage_bits_float32", 0))
        shared = int(candidate.get("shared_stage4_codebook_bits", 0)) + int(candidate.get("shared_writer_parameter_bits", 0))
        passed = summary.get("local_renewed_gate", summary.get("strict", {}).get("renewed_gate", False))
        lines.append(f"| `{candidate_id}` | {scope} | {inherited} | {developmental} | {shared} | {passed} |")
    lines.extend(
        (
            "",
            "The global object is an upper-bound broadcast and is not evidence of localization. For local objects, inherited boundary bits and temporary developmental storage are deliberately reported separately.",
            "",
            "This remains a synthetic CA experiment, not a claim about biological life, agency, or nonphysical inheritance.",
            "",
        )
    )
    return "\n".join(lines)


def _render_preconfirmation_lay(results: dict[str, Any]) -> str:
    candidates = results["selection_decision"]["confirmation_candidate_ids"]
    return "\n".join(
        (
            "# Stage 5 in plain language",
            "",
            "Stage 4 showed that a tiny 64-bit hidden message could survive a visible reset and help the cellular automaton rebuild its ancestral texture. Stage 5 is asking whether that message can behave more like a physical propagule: start at one small place, spread only to neighbouring sites, act locally, and be rewritten locally by the daughter.",
            "",
            f"The exposed engineering rounds have frozen these final test objects: {', '.join(candidates)}. One is the old nonlocal broadcast used as a safety check; the others are spatially local. A one-site carrier still inherits only 64 bits, although the daughter temporarily grows a much larger working field while it develops.",
            "",
            "The untouched confirmation cases have not been opened. They require a separate, explicitly authorized run, so this is a locked shortlist rather than the final result.",
            "",
        )
    )


def _render_final_report(results: dict[str, Any]) -> str:
    adjudication = results["adjudication"]
    lines = [
        "# CA motif-lineage Stage 5: final report",
        "",
        f"Verdict: `{adjudication['verdict']}`.",
        f"Fresh Stage-4 global-anchor replication: `{adjudication['fresh_stage4_anchor_replicated']}`.",
        "",
        "| Object | Inherited bits | Ordinary Stage-5 gate | Moderate Stage-5 gate |",
        "|---|---:|---:|---:|",
    ]
    for candidate_id, value in adjudication["candidates"].items():
        metadata = value["candidate"]
        ordinary = value["environments"]["ordinary"]["stage5_gate"]
        moderate = value["environments"]["moderate_joint"]["stage5_gate"]
        lines.append(f"| `{candidate_id}` | {metadata.get('payload_bits', 64)} | {ordinary} | {moderate} |")
    lines.extend(
        (
            "",
            "A local positive requires more than form recovery: channel scrambling and transport removal must damage it, translated patches must retain the effect, the visible state must reset exactly, and the daughter must actively rewrite the carrier.",
            "",
            "The inherited boundary message, temporary developmental field, and shared machinery are separate accounting categories. This synthetic result does not establish metabolism, agency, biological life, or a nonphysical memory.",
            "",
        )
    )
    return "\n".join(lines)


def _render_final_lay(results: dict[str, Any]) -> str:
    verdict = results["adjudication"]["verdict"]
    explanations = {
        "ROBUST_LOCAL_64BIT_RENEWED_CA_PLASTIC_HEREDITY": "A one-cell, 64-bit propagule rebuilt the inherited texture through local spreading and local rewriting, and it still worked under moderate copying and process damage.",
        "LOCAL_64BIT_RENEWED_CA_PLASTIC_HEREDITY": "A one-cell, 64-bit propagule worked under ordinary conditions, but its full causal result did not survive the registered moderate damage.",
        "LOCAL_DISTRIBUTED_RENEWED_CA_PLASTIC_HEREDITY": "Localization worked only when the inherited message occupied several neighbouring sites; the one-cell version was not sufficient.",
        "GLOBAL_BROADCAST_ONLY": "The old 64-bit message still works when copied everywhere at once, but we did not make it work as a genuinely local propagule.",
        "NO_STAGE4_REPLICATION": "Even the frozen nonlocal Stage-4 safety check failed on the fresh cases, so the localization result cannot be interpreted.",
    }
    return "\n".join(
        (
            "# Stage 5 final result in plain language",
            "",
            f"The final verdict is `{verdict}`.",
            "",
            explanations.get(verdict, "The spectral carrier was informative, but no registered local object passed the fresh causal test."),
            "",
            "The important distinction is between the small message that crosses generations and the larger workspace the daughter grows while developing. Only the former is inherited. This remains an engineered cellular-automaton result, not proof of life or consciousness.",
            "",
        )
    )


def _update_discovery_log(state: str, verdict: str, elapsed_seconds: float) -> None:
    path = ROOT / "DISCOVERY_LOG_EIDOSOMA_SCIENTIST.md"
    existing = path.read_text(encoding="utf-8") if path.exists() else "# Discovery log\n"
    start = "<!-- STAGE5_LOCALIZATION_START -->"
    end = "<!-- STAGE5_LOCALIZATION_END -->"
    section = "\n".join(
        (
            start,
            "## CA Stage 5 — physical localization",
            "",
            f"State: `{state}`; verdict: `{verdict}`.",
            f"Elapsed `{elapsed_seconds / 3600.0:.3f}` wall hours.",
            "See `results/ca-motif-lineage-stage-5/REPORT.md` and `LAY_SUMMARY.md`.",
            end,
        )
    )
    if start in existing and end in existing:
        prefix, remainder = existing.split(start, 1)
        _, suffix = remainder.split(end, 1)
        updated = prefix.rstrip() + "\n\n" + section + suffix
    else:
        updated = existing.rstrip() + "\n\n" + section + "\n"
    _atomic_text(path, updated)


def run_motif_localization(
    output: Path,
    *,
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
        raise ValueError(f"unknown Stage-5 profile {profile_name!r}")
    if max_hours <= 0.0 or max_hours > 8.0:
        raise ValueError("Stage-5 max-hours must be in (0, 8]")
    selected_phases = tuple(phases or DEFAULT_PRECONFIRMATION_PHASES)
    unknown = [phase for phase in selected_phases if phase not in PHASES]
    if unknown:
        raise ValueError(f"unknown Stage-5 phases: {unknown}")
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
    contract = LocalizationContract()
    writer_contract = MotifContract()
    profile = LOCALIZATION_PROFILES[profile_name]
    reserve_seconds = min(
        contract.science_reserve_seconds, max(60.0, max_hours * 3600.0 * 0.10)
    )
    science_deadline = max(started, hard_deadline - reserve_seconds)

    def status(state: str, phase: str, **extra: Any) -> None:
        now = time.time()
        payload = {
            "state": state,
            "stage": "5-physical-localization",
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
        frozen = load_frozen_stage4(
            stage4_root, stage3r_root, stage3_root, stage2_root, stage1_root
        )
        cohorts = select_localization_cohorts(profile, frozen, profile_name=profile_name)
        walsh_model = frozen["winner_model"]
        mode_ids = walsh_mode_ids(walsh_model)
        if tuple(mode_ids) != REGISTERED_MODE_IDS:
            raise ValueError("the frozen Stage-4 Walsh mode order changed")
        fit_matrix, fit_groups, trace_hashes = load_stage3r_fit_matrix(frozen["stage3r"])
        writers, writer_fit_audit = fit_local_writers(
            fit_matrix,
            frozen["reference"][frozen["configuration"].write_window]["motif_probability"],
            walsh_model,
        )
        calibration = calibrate_local_dynamics(
            writers, walsh_model["basis"], walsh_model["quantizer_scale"]
        )
        local_candidates = build_local_candidates(writers, calibration)
        global_candidate = _global_candidate(walsh_model)
        all_candidates = [global_candidate, *local_candidates]
        anatomy_models = build_anatomy_models(walsh_model)
        configuration_payload = _configuration_payload(frozen["configuration"])
        design_payload = {
            "experiment": "ca_motif_lineage_stage_5",
            "contract": contract.to_dict(),
            "writer_contract_digest": writer_contract.digest,
            "profile_name": profile_name,
            "profile": asdict(profile),
            "configuration": frozen["configuration"].to_dict(),
            "stage4_design_digest": frozen["design_digest"],
            "phases_contract": PHASES,
            "confirmation_separate_invocation": True,
            "registered_stage4_winner": STAGE4_WINNER_ID,
            "registered_walsh_mode_ids": mode_ids,
            "registered_walsh_bit_supports": walsh_bit_supports(mode_ids),
            "local_candidate_ids": [candidate["candidate_id"] for candidate in local_candidates],
            "anatomy_variant_ids": [model["candidate_id"] for model in anatomy_models],
            "anatomy_pair_ids": [pair["pair_id"] for pair in cohorts["anatomy"]],
            "screen_pair_ids": [pair["pair_id"] for pair in cohorts["screen"]],
            "qualification_pair_ids": [pair["pair_id"] for pair in cohorts["qualification"]],
            "confirmation_pair_ids": [pair["pair_id"] for pair in cohorts["confirmation"]],
            "later_audit_pair_ids": [pair["pair_id"] for pair in cohorts["later_audit"]],
            "transfer_pair_ids_by_rule": {
                str(rule): [
                    pair["pair_id"]
                    for pair in cohorts["transfer"]
                    if int(pair["stage5_transfer_rule"]) == rule
                ]
                for rule in (31648, 70366)
            },
            "fit_trace_groups": len(set(fit_groups)),
            "fit_trace_sha256": trace_hashes,
            "input_sha256": {
                "protocol": _sha256(PROTOCOL_PATH),
                **{
                    f"stage4_{key}": _sha256(path)
                    for key, path in frozen["paths"].items()
                },
            },
            "implementation_sha256": {
                "motif_localization.py": _sha256(Path(__file__)),
                "motif_compression.py": _sha256(Path(__file__).with_name("motif_compression.py")),
                "motif_lineage.py": _sha256(Path(__file__).with_name("motif_lineage.py")),
            },
            "information_accounting": {
                "primary_inherited_bits": 64,
                "developmental_field_values": 4096,
                "developmental_field_storage_bits_float32": 131072,
                "stage4_shared_codebook_bits": int(walsh_model["codebook_bits"]),
                "affine_writer_shared_parameter_bits": 16 * 2 * 32,
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
                raise ValueError("Stage-5 resume design digest mismatch")
        elif "confirm" in selected_phases:
            raise FileNotFoundError("confirmation requires a reviewed Stage-5 design")
        _atomic_json(design_path, design)
        _atomic_json(
            output / "MANIFEST.json",
            {
                "experiment": "ca_motif_lineage_stage_5",
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
            "anatomy_pair_ids": design["anatomy_pair_ids"],
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
            "state": "passed",
            "stage4_verdict": frozen["results"]["adjudication"]["verdict"],
            "stage4_winner": STAGE4_WINNER_ID,
            "winner_rank": int(walsh_model["rank"]),
            "winner_bits_per_value": int(walsh_model["bits"]),
            "winner_payload_bits": int(walsh_model["payload_bits"]),
            "winner_codebook_bits": int(walsh_model["codebook_bits"]),
            "walsh_mode_ids": mode_ids,
            "walsh_bit_supports": walsh_bit_supports(mode_ids),
            "stage5_reserve_pairs": len(frozen["reserve_ids"]),
            "confirmation_pairs": len(cohorts["confirmation"]),
            "later_audit_pairs": len(cohorts["later_audit"]),
            "cleanroom_exclusion_upheld": True,
            "confirmation_not_opened": "confirm" not in selected_phases,
        }
        if "audit" in selected_phases:
            _atomic_json(output / "CLEANROOM_AUDIT.json", audit)

        local_model_consumers = {"transcode", "localize", "qualify", "transfer", "adjudicate", "confirm"}
        if "calibrate" in selected_phases:
            status("running", "calibration")
            writer_fit_audit.update(
                {"design_digest": design_digest, "trace_sha256": trace_hashes}
            )
            calibration["design_digest"] = design_digest
            _atomic_json(output / "WRITER_FIT_AUDIT.json", writer_fit_audit)
            _atomic_json(output / "CALIBRATION.json", calibration)
            save_writer_models(
                output, writers, local_candidates, design_digest=design_digest
            )
        elif local_model_consumers.intersection(selected_phases):
            stored_writers, stored_candidates = load_writer_models(output, design_digest)
            if [writer["writer_id"] for writer in stored_writers] != [writer["writer_id"] for writer in writers]:
                raise ValueError("stored Stage-5 writers changed")
            if [candidate["candidate_id"] for candidate in stored_candidates] != [candidate["candidate_id"] for candidate in local_candidates]:
                raise ValueError("stored Stage-5 candidate list changed")
            local_candidates = stored_candidates
            all_candidates = [global_candidate, *local_candidates]

        if "transcode" in selected_phases:
            status("running", "transcode")
            transcode = transcode_audit(walsh_model)
            transcode["design_digest"] = design_digest
            _atomic_json(output / "TRANSCODE_AUDIT.json", transcode)
            if not transcode["passed"]:
                raise AssertionError("Stage-5 local/global transcode audit failed")
        elif set(selected_phases) & {"localize", "qualify", "transfer", "adjudicate", "confirm"}:
            transcode = _load_json(output / "TRANSCODE_AUDIT.json")
            if transcode.get("design_digest") != design_digest or not transcode.get("passed"):
                raise ValueError("a passed Stage-5 transcode audit is required")

        if "anatomy" in selected_phases:
            items = [
                {
                    "checkpoint": f"anatomy-{index:04d}",
                    "pair": pair,
                    "replicates": profile.anatomy_replicates,
                    "generations": profile.anatomy_generations,
                    "configuration": configuration_payload,
                }
                for index, pair in enumerate(cohorts["anatomy"])
            ]
            status("running", "anatomy", completed=0, total=len(items))
            anatomy_rows, complete = _run_json_checkpoints(
                output,
                "anatomy",
                items,
                anatomy_models,
                _anatomy_pair_task,
                writer_contract,
                contract,  # type: ignore[arg-type]
                frozen["reference"],
                design_digest,
                workers=workers,
                resume=resume,
                deadline=science_deadline,
                status=status,
            )
            anatomy = summarize_anatomy(
                anatomy_rows, anatomy_models, profile, contract, complete
            )
            anatomy["design_digest"] = design_digest
            _atomic_json(output / "ANATOMY.json", anatomy)
            if not complete:
                status("partial_budget_exhausted", "anatomy")
                return {"state": "partial_budget_exhausted", "phase": "anatomy"}
        elif {"localize", "qualify", "transfer", "adjudicate", "confirm"}.intersection(selected_phases):
            anatomy = _load_json(output / "ANATOMY.json")
        else:
            anatomy = {"state": "not_requested"}

        downstream = {"localize", "qualify", "transfer", "adjudicate", "confirm"}
        if not downstream.intersection(selected_phases):
            status("phases_complete", "campaign")
            return {"state": "phases_complete", "completed_phases": selected_phases}

        if "confirm" in selected_phases:
            decision = _load_json(output / "SELECTION_DECISION.json")
            confirmation_design = _load_json(output / "CONFIRMATION_DESIGN.json")
            if decision.get("design_digest") != design_digest or confirmation_design.get("design_digest") != design_digest:
                raise ValueError("Stage-5 confirmation decision belongs to another design")
            if decision.get("confirmation_state") != "awaiting_human_review":
                raise ValueError("Stage-5 confirmation is not awaiting review")
            candidate_ids = list(decision["confirmation_candidate_ids"])
            if candidate_ids != confirmation_design.get("candidate_ids"):
                raise ValueError("Stage-5 confirmation candidate list changed after review")
            if confirmation_design.get("local_model_sha256") != _sha256(output / "LOCAL_MODELS.npz"):
                raise ValueError("Stage-5 local model archive changed after review")
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
            confirmation_rows, complete = _run_json_checkpoints(
                output,
                "confirmation",
                items,
                confirmation_candidates,
                _confirmation_pair_task,
                writer_contract,
                contract,  # type: ignore[arg-type]
                frozen["reference"],
                design_digest,
                workers=workers,
                resume=resume,
                deadline=science_deadline,
                status=status,
            )
            adjudication = adjudicate_stage5_confirmation(
                confirmation_rows,
                confirmation_candidates,
                profile,
                contract,
                complete,
            )
            state = "complete" if complete else "partial_budget_exhausted"
            results = {
                "experiment": "ca_motif_lineage_stage_5",
                "state": state,
                "profile": profile_name,
                "design_digest": design_digest,
                "stage4_design_digest": frozen["design_digest"],
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
                    "decision": "stage6_may_be_planned_after_review" if complete else "resume_stage5_confirmation",
                    "automatic_launch": False,
                    "review_required": True,
                },
            )
            if complete:
                cohort_payload["confirmation_trajectory_state"] = "complete"
                _atomic_json(output / "COHORTS.json", cohort_payload)
                _atomic_text(output / "COMPLETE", "complete\n")
                if profile_name == "reference":
                    _update_discovery_log(state, adjudication["verdict"], results["elapsed_seconds"])
            _atomic_json(
                output / "QUEUE.json",
                _queue(
                    design_digest,
                    state,
                    finalists=candidate_ids,
                    verdict=adjudication["verdict"],
                ),
            )
            status(state, "campaign", verdict=adjudication["verdict"])
            return results

        if "localize" in selected_phases:
            items = [
                {
                    "checkpoint": f"localize-{pair_index:04d}-object-{candidate_index:02d}",
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
            status("running", "localize", completed=0, total=len(items))
            screen_rows, complete = _run_json_checkpoints(
                output,
                "localize",
                items,
                all_candidates,
                _screen_pair_task,
                writer_contract,
                contract,  # type: ignore[arg-type]
                frozen["reference"],
                design_digest,
                workers=workers,
                resume=resume,
                deadline=science_deadline,
                status=status,
            )
            screen = adjudicate_localization_screen(
                screen_rows, all_candidates, profile, contract, complete
            )
            screen["design_digest"] = design_digest
            _atomic_json(output / "LOCALIZATION_SCREEN.json", screen)
            if not complete:
                status("partial_budget_exhausted", "localize")
                return {"state": "partial_budget_exhausted", "phase": "localize"}
        else:
            screen = _load_json(output / "LOCALIZATION_SCREEN.json")
            screen_rows = _phase_rows(output, "localize", design_digest)
        selected_ids = list(screen["selected_candidate_ids"])
        selected_candidates = _selected_candidates(all_candidates, selected_ids)

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
                frozen["reference"],
                design_digest,
                workers=workers,
                resume=resume,
                deadline=science_deadline,
                status=status,
            )
            qualification = adjudicate_local_qualification(
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

        confirmation_ids = select_stage5_finalists(
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
                frozen["reference"],
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

        candidate_metadata = {
            candidate["candidate_id"]: _candidate_metadata(candidate)
            for candidate in confirmation_candidates
        }
        decision = {
            "design_digest": design_digest,
            "stage4_design_digest": frozen["design_digest"],
            "confirmation_candidate_ids": confirmation_ids,
            "candidate_metadata": candidate_metadata,
            "confirmation_state": "awaiting_human_review",
            "confirmation_pairs": profile.confirmation_pairs,
            "confirmation_replicates": profile.confirmation_replicates,
            "confirmation_generations": profile.confirmation_generations,
            "ordinary_and_moderate_joint": True,
            "retuning_permitted": False,
            "automatic_launch": False,
        }
        _atomic_json(output / "SELECTION_DECISION.json", decision)
        _atomic_json(
            output / "CONFIRMATION_DESIGN.json",
            {
                "design_digest": design_digest,
                "candidate_ids": confirmation_ids,
                "candidate_metadata": candidate_metadata,
                "local_model_sha256": _sha256(output / "LOCAL_MODELS.npz"),
                "cohort_ids_sha256": hashlib.sha256(
                    "\n".join(pair["pair_id"] for pair in cohorts["confirmation"]).encode()
                ).hexdigest(),
                "trajectory_state": "untouched",
                "authorization_required": True,
            },
        )
        results = {
            "experiment": "ca_motif_lineage_stage_5",
            "state": "awaiting_confirmation",
            "profile": profile_name,
            "design_digest": design_digest,
            "stage4_design_digest": frozen["design_digest"],
            "elapsed_seconds": time.time() - started,
            "anatomy": anatomy,
            "screen": screen,
            "qualification": qualification,
            "transfer": transfer,
            "selection_decision": decision,
            "information_accounting": design["information_accounting"],
        }
        _atomic_json(output / "PRECONFIRMATION_RESULTS.json", results)
        _atomic_text(output / "REPORT.md", _render_preconfirmation_report(results))
        _atomic_text(output / "LAY_SUMMARY.md", _render_preconfirmation_lay(results))
        _atomic_json(
            output / "QUEUE.json",
            _queue(
                design_digest,
                "blocked_pending_human_review",
                finalists=confirmation_ids,
            ),
        )
        status(
            "awaiting_confirmation",
            "campaign",
            confirmation_candidate_ids=confirmation_ids,
        )
        return results
    except BaseException as error:
        status("failed", "campaign", error=repr(error))
        raise
