"""Stage-6 minimal, scalable, and evolvable regenerative CA heredity.

Five separately gated rounds progressively remove the global communication
assumptions of Stage 5R.  The final 62-pair audit reserve is never loaded by a
reference development round.
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
    _configuration_payload,
    _phase_rows,
    _run_json_checkpoints,
    decode_payload,
    load_stage3r_fit_matrix,
    quantize_payload,
)
from .motif_lineage import (
    MotifContract,
    ReaderConfiguration,
    _bootstrap,
    _cosine_labels,
    _founders,
    _normalize_rows,
    _outcome,
    _step,
    apply_energy_reader,
    collect_trajectory_counts,
    motif3_codes,
)
from .motif_localization import (
    DEFAULT_STAGE1_ROOT,
    DEFAULT_STAGE2_ROOT,
    DEFAULT_STAGE3_ROOT,
    DEFAULT_STAGE3R_ROOT,
    DEFAULT_STAGE4_ROOT,
    GLOBAL_ANCHOR_ID,
    REGISTERED_MODE_IDS,
    _global_candidate,
    apply_local_reader,
    walsh_mode_ids,
)
from .motif_regeneration import (
    CORE_CONDITIONS,
    DEFAULT_STAGE5_ROOT,
    LOCAL_EXTRA_CONDITIONS,
    RegenerationContract,
    _payload_summary,
    _simulate_candidate as _simulate_stage5r_candidate,
    fit_regenerative_writers,
    load_frozen_stage5,
    load_regenerative_models,
)
from .motif_repair import RepairProfile, _score_state, _strict_confirmation_gate, heldout_lineage_accuracy


ROOT = Path(__file__).resolve().parent.parent
PROTOCOL_PATH = ROOT / "CA_MOTIF_LINEAGE_STAGE6_PROTOCOL.md"
DEFAULT_STAGE5R_ROOT = ROOT / "results/ca-motif-lineage-stage-5r"
RULE = 31649
ROUNDS = ("locality", "scale", "compression", "ecology", "audit")
EXACT_ID = "regen-hist512-exact-p1-flood-consensus"
COMPACT_ID = "regen-moment16-ridge-p1-flood-consensus"
COMPACT_ANCHOR_ID = "stage5r-compact-anchor"
EXACT_ANCHOR_ID = "stage5r-exact-anchor"
LOCALITY_HOPS = (2, 4, 5, 8)
CONSOLIDATION_SPANS = (0, 2, 4, 7, 15)
COMPRESSION_SHAPES = ((16, 4), (12, 3), (8, 3), (8, 2), (4, 2), (2, 2))
TARGETED_CONDITIONS = (
    "transport_disabled",
    "regeneration_disabled",
    "consolidation_disabled",
    "translated_patch",
    "communication_cut",
)
QUALIFICATION_CONDITIONS = CORE_CONDITIONS + (
    "write_disabled",
    *TARGETED_CONDITIONS,
)


@dataclass(frozen=True)
class MinimalityContract:
    implementation_version: str = "ca-motif-lineage-stage6-cleanroom-v1"
    namespace: str = "plastic-ca-motif-lineage-stage6-v1"
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
    final_alpha_per_object: float = 0.005
    decoder_splits: int = 4
    bounded_hops: int = 5
    bounded_consolidation_steps: int = 14
    retention_fraction: float = 0.70
    max_hours: float = 4.0
    science_reserve_seconds: float = 1800.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "visible_reset": "bitwise-identical matched board before every generation",
                "propagation": "synchronous occupied Moore-neighbour consensus copying",
                "writer_reduction": "registered finite directed shift-and-accumulate rectangle",
                "independent_unit": "matched founder pair",
                "reserve_policy": "62 Stage-5R later-audit pairs unread until authorized final audit",
                "claim_boundary": "engineered synthetic CA heredity only",
            }
        )
        return payload

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


@dataclass(frozen=True)
class MinimalityProfile:
    locality_screen_pairs: int
    locality_screen_replicates: int
    locality_qualification_pairs: int
    locality_qualification_replicates: int
    locality_endurance_pairs: int
    locality_endurance_replicates: int
    scale_pairs: int
    scale_replicates: int
    compression_screen_pairs: int
    compression_screen_replicates: int
    compression_qualification_pairs: int
    compression_qualification_replicates: int
    ecology_pairs: int
    ecology_replicates: int
    evolution_training_pairs: int
    evolution_validation_pairs: int
    evolution_populations: int
    evolution_population_size: int
    evolution_generations: int
    audit_pairs: int
    audit_replicates: int
    bootstrap_resamples: int


MINIMALITY_PROFILES: dict[str, MinimalityProfile] = {
    "smoke": MinimalityProfile(2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 2, 2, 100),
    "pilot": MinimalityProfile(16, 4, 16, 4, 8, 4, 8, 2, 16, 4, 16, 4, 8, 4, 16, 8, 4, 16, 12, 16, 8, 1_000),
    "reference": MinimalityProfile(
        64,
        8,
        96,
        16,
        32,
        8,
        32,
        4,
        64,
        8,
        96,
        16,
        64,
        8,
        128,
        64,
        12,
        32,
        30,
        62,
        48,
        10_000,
    ),
}
PUBLIC_PROFILES = tuple(MINIMALITY_PROFILES)


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


def load_frozen_stage5r(
    stage5r_root: Path = DEFAULT_STAGE5R_ROOT,
    stage5_root: Path = DEFAULT_STAGE5_ROOT,
    stage4_root: Path = DEFAULT_STAGE4_ROOT,
    stage3r_root: Path = DEFAULT_STAGE3R_ROOT,
    stage3_root: Path = DEFAULT_STAGE3_ROOT,
    stage2_root: Path = DEFAULT_STAGE2_ROOT,
    stage1_root: Path = DEFAULT_STAGE1_ROOT,
) -> dict[str, Any]:
    stage5r_root = stage5r_root.resolve()
    names = {
        "results": "RESULTS.json",
        "decision": "STAGE_DECISION.json",
        "design": "DESIGN.json",
        "cohorts": "COHORTS.json",
        "manifest": "MANIFEST.json",
        "models": "REGENERATIVE_MODELS.json",
        "model_arrays": "REGENERATIVE_MODELS.npz",
        "mechanism": "MECHANISM_AUDIT.json",
    }
    paths = {key: stage5r_root / name for key, name in names.items()}
    missing = [path for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing frozen Stage-5R artifacts: {missing}")
    payload = {
        key: _load_json(path)
        for key, path in paths.items()
        if key != "model_arrays"
    }
    digest = str(payload["design"].get("design_digest"))
    for key in ("results", "decision", "cohorts", "manifest", "models", "mechanism"):
        if str(payload[key].get("design_digest")) != digest:
            raise ValueError(f"Stage-5R {key} design digest does not match")
    if payload["results"].get("state") != "complete":
        raise ValueError("Stage 5R is not complete")
    verdict = payload["results"].get("adjudication", {}).get("verdict")
    if verdict != "ROBUST_REGENERATIVE_LOCAL_64BIT_CA_PLASTIC_HEREDITY":
        raise ValueError("Stage 5R does not have the registered robust verdict")
    if payload["decision"].get("decision") != "stage6_may_be_planned_after_review":
        raise ValueError("Stage-5R decision does not permit Stage 6")
    if payload["cohorts"].get("confirmation_trajectory_state") != "complete":
        raise ValueError("Stage-5R confirmation is incomplete")
    if payload["cohorts"].get("later_audit_trajectory_state") != "untouched":
        raise ValueError("Stage-5R later-audit reserve is no longer untouched")
    if _sha256(paths["model_arrays"]) != payload["models"].get("model_sha256"):
        raise ValueError("Stage-5R model archive hash mismatch")
    frozen5 = load_frozen_stage5(
        stage5_root, stage4_root, stage3r_root, stage3_root, stage2_root, stage1_root
    )
    if payload["design"].get("stage5_design_digest") != frozen5["design_digest"]:
        raise ValueError("Stage-5R ancestry does not match supplied Stage 5")
    writers, candidates = load_regenerative_models(stage5r_root, digest)
    by_candidate = {str(row["candidate_id"]): row for row in candidates}
    for candidate_id in (EXACT_ID, COMPACT_ID):
        if candidate_id not in by_candidate:
            raise ValueError(f"confirmed Stage-5R candidate missing: {candidate_id}")
    robust = set(payload["results"]["adjudication"].get("robust_regenerative_local_64bit_candidate_ids", ()))
    if not {EXACT_ID, COMPACT_ID} <= robust:
        raise ValueError("both registered Stage-5R candidates must be robust")
    later_ids = list(payload["cohorts"]["later_audit_pair_ids"])
    if len(later_ids) != 62 or len(set(later_ids)) != 62:
        raise ValueError("Stage-6 final reserve must contain 62 unique pairs")
    return {
        **payload,
        "root": stage5r_root,
        "paths": paths,
        "design_digest": digest,
        "stage5": frozen5,
        "stage4": frozen5["stage4"],
        "writers_loaded": writers,
        "candidates_loaded": candidates,
        "by_candidate": by_candidate,
        "later_ids": later_ids,
    }


def select_minimality_cohorts(
    profile: MinimalityProfile,
    frozen: dict[str, Any],
    *,
    profile_name: str,
    open_audit: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    all_pairs = load_round3_pairs()[RULE]
    by_id = {str(pair["pair_id"]): pair for pair in all_pairs}
    later = set(frozen["later_ids"])
    missing = later - set(by_id)
    if missing:
        raise ValueError("a Stage-6 reserve identity is absent from the frozen pair bank")
    exposed = [pair for pair in all_pairs if pair["pair_id"] not in later]
    exposed = sorted(
        exposed,
        key=lambda pair: (
            hashlib.sha256(f"stage6-cohort:{pair['pair_id']}".encode()).hexdigest(),
            pair["pair_id"],
        ),
    )
    sizes = (
        ("locality_screen", profile.locality_screen_pairs),
        ("locality_qualification", profile.locality_qualification_pairs),
        ("locality_endurance", profile.locality_endurance_pairs),
        ("scale", profile.scale_pairs),
        ("compression_screen", profile.compression_screen_pairs),
        ("compression_qualification", profile.compression_qualification_pairs),
        ("ecology", profile.ecology_pairs),
        ("evolution_training", profile.evolution_training_pairs),
        ("evolution_validation", profile.evolution_validation_pairs),
    )
    result: dict[str, list[dict[str, Any]]] = {}
    cursor = 0
    for name, size in sizes:
        result[name] = exposed[cursor : cursor + size]
        cursor += size
        if len(result[name]) != size:
            raise ValueError("not enough exposed pairs for the registered Stage-6 partitions")
    if profile_name == "reference":
        result["audit"] = (
            [by_id[pair_id] for pair_id in frozen["later_ids"]]
            if open_audit
            else []
        )
    else:
        result["audit"] = exposed[cursor : cursor + profile.audit_pairs]
    used = [pair["pair_id"] for name, rows in result.items() if name != "audit" for pair in rows]
    if len(used) != len(set(used)):
        raise AssertionError("Stage-6 development partitions overlap")
    if profile_name == "reference" and set(used) & later:
        raise AssertionError("Stage-6 development touched the final reserve")
    return result


def _repeat_histories(values: np.ndarray, replicates: int) -> np.ndarray:
    return np.concatenate(
        (np.repeat(values[0:1], replicates, axis=0), np.repeat(values[1:2], replicates, axis=0)),
        axis=0,
    )


def _swap_histories(values: np.ndarray, replicates: int) -> np.ndarray:
    return np.concatenate((values[replicates:], values[:replicates]), axis=0)


def _resize_board(board: np.ndarray, extent: int) -> np.ndarray:
    values = np.asarray(board, dtype=np.bool_)
    if extent % values.shape[0] or values.shape[0] != values.shape[1]:
        raise ValueError("extent must be an integer square tiling of the source")
    factor = extent // values.shape[0]
    return np.tile(values, (factor, factor))


def _uniforms(
    pair_id: str,
    candidate_id: str,
    purpose: str,
    generation: int,
    sweep: int,
    replicates: int,
    extent: int,
) -> np.ndarray:
    rng = np.random.default_rng(
        _hash_seed("stage6", pair_id, candidate_id, purpose, generation, sweep, extent)
    )
    half = rng.random((replicates, extent, extent))
    return np.concatenate((half, half), axis=0)


def dynamic_origins(
    pair_id: str, generation: int, replicates: int, extent: int
) -> np.ndarray:
    rng = np.random.default_rng(
        _hash_seed("stage6-origin", pair_id, generation, extent)
    )
    half = np.stack(
        (
            rng.integers(0, extent, size=replicates),
            rng.integers(0, extent, size=replicates),
        ),
        axis=1,
    ).astype(np.int16)
    return np.concatenate((half, half), axis=0)


def embed_dynamic_seed(
    payload: np.ndarray,
    origins: np.ndarray,
    extent: int,
    *,
    translated: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(payload, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("Stage-6 seed must have shape (sample, channel)")
    field = np.zeros((len(values), extent, extent, values.shape[-1]), dtype=np.float32)
    occupied = np.zeros((len(values), extent, extent), dtype=np.bool_)
    for sample, (y, x) in enumerate(np.asarray(origins, dtype=np.int64)):
        if translated:
            y = (y + 5) % extent
            x = (x + 7) % extent
        field[sample, y, x] = values[sample]
        occupied[sample, y, x] = True
    return field, occupied


def propagate_bounded(
    field: np.ndarray,
    occupancy: np.ndarray,
    hops: int,
    *,
    communication_cut: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    values = np.asarray(field, dtype=np.float32).copy()
    occupied = np.asarray(occupancy, dtype=np.bool_).copy()
    trace: list[float] = []
    extent = values.shape[1]
    for _step_index in range(hops):
        neighbour_count = np.zeros_like(occupied, dtype=np.int16)
        neighbour_sum = np.zeros(values.shape, dtype=np.float64)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                shifted_occupancy = np.roll(occupied, (dy, dx), axis=(1, 2))
                shifted_values = np.roll(
                    values.astype(np.float64) * occupied[..., None],
                    (dy, dx),
                    axis=(1, 2),
                )
                if communication_cut:
                    # Two fixed vertical severances split the torus into halves.
                    blocked: tuple[int, ...] = ()
                    if dx == 1:
                        blocked = (0, extent // 2)
                    elif dx == -1:
                        blocked = (extent - 1, extent // 2 - 1)
                    for column in blocked:
                        shifted_occupancy[:, :, column] = False
                        shifted_values[:, :, column] = 0.0
                neighbour_count += shifted_occupancy
                neighbour_sum += shifted_values
        new_sites = (~occupied) & (neighbour_count > 0)
        total = neighbour_sum + values.astype(np.float64) * occupied[..., None]
        divisor = neighbour_count + occupied.astype(np.int16)
        averaged = np.divide(
            total,
            divisor[..., None],
            out=np.zeros_like(total),
            where=divisor[..., None] > 0,
        )
        values[occupied | new_sites] = averaged[occupied | new_sites]
        occupied |= new_sites
        trace.append(float(np.mean(occupied)))
    return values, occupied, trace


def bounded_shift_reduce(values: np.ndarray, span: int) -> np.ndarray:
    """Explicit 2*span-step directed local rectangular reduction."""

    source = np.asarray(values, dtype=np.float64)
    if source.ndim < 3 or source.shape[1] != source.shape[2]:
        raise ValueError("values require square spatial axes 1 and 2")
    extent = source.shape[1]
    if span < 0 or span >= extent:
        raise ValueError("span must be in [0, extent)")
    packet = source.copy()
    total = source.copy()
    for _ in range(span):
        packet = np.roll(packet, 1, axis=2)
        total += packet
    horizontal = total / float(span + 1)
    packet = horizontal.copy()
    total = horizontal.copy()
    for _ in range(span):
        packet = np.roll(packet, 1, axis=1)
        total += packet
    return total / float(span + 1)


def bounded_reduce_endpoint(
    values: np.ndarray, span: int, origins: np.ndarray
) -> np.ndarray:
    """Endpoint-equivalent local rectangle average without materializing routing buffers."""

    source = np.asarray(values, dtype=np.float64)
    result = np.zeros((len(source), source.shape[-1]), dtype=np.float64)
    extent = source.shape[1]
    for sample, (origin_y, origin_x) in enumerate(np.asarray(origins, dtype=np.int64)):
        ys = (origin_y - np.arange(span + 1)) % extent
        xs = (origin_x - np.arange(span + 1)) % extent
        result[sample] = source[sample][np.ix_(ys, xs)].mean(axis=(0, 1))
    return result


def _codec_model(
    walsh_model: dict[str, Any], rank: int, bits: int
) -> dict[str, Any]:
    if rank < 1 or rank > 16:
        raise ValueError("Stage-6 rank must be between 1 and 16")
    return {
        "candidate_id": f"stage6-walsh-r{rank:02d}-q{bits:02d}",
        "family": walsh_model["family"],
        "rank": rank,
        "bits": bits,
        "payload_bits": rank * bits,
        "basis": np.asarray(walsh_model["basis"], dtype=np.float32)[:, :rank].copy(),
        "quantizer_scale": np.asarray(walsh_model["quantizer_scale"], dtype=np.float32)[:rank].copy(),
        "runtime_label_access": False,
        "runtime_parent_access": False,
        "runtime_target_access": False,
    }


def fit_minimal_writer(
    fit_matrix: np.ndarray,
    reference_probability: np.ndarray,
    codec: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    values = np.asarray(fit_matrix, dtype=np.float64)
    basis = np.asarray(codec["basis"], dtype=np.float64)
    signs = basis * math.sqrt(512.0)
    logits = np.clip(values * 2.0, -8.0, 8.0)
    probability = np.asarray(reference_probability, dtype=np.float64)[None, :] * np.exp(logits)
    probability /= probability.sum(axis=1, keepdims=True)
    moments = probability @ signs
    target = np.float64(0.50) * (values @ basis)
    design = np.concatenate((moments, np.ones((len(moments), 1))), axis=1)
    ridge = np.eye(codec["rank"] + 1, dtype=np.float64) * 1e-6
    ridge[-1, -1] = 1e-9
    fitted = np.linalg.solve(design.T @ design + ridge, design.T @ target)
    prediction = design @ fitted
    total = float(np.sum((target - target.mean(axis=0, keepdims=True)) ** 2))
    residual = float(np.sum((target - prediction) ** 2))
    return (
        fitted[:-1].astype(np.float32),
        fitted[-1].astype(np.float32),
        {
            "label_and_outcome_blind": True,
            "rank": int(codec["rank"]),
            "bits": int(codec["bits"]),
            "r2": 1.0 - residual / total if total > 0.0 else 0.0,
        },
    )


def build_locality_candidates(
    frozen: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    walsh = frozen["stage4"]["winner_model"]
    compact = frozen["by_candidate"][COMPACT_ID]
    codec = _codec_model(walsh, 16, 4)
    base = {
        "kind": "bounded",
        "writer_kind": "moment-ridge",
        "rank": 16,
        "bits": 4,
        "payload_bits": 64,
        "occupancy_bits": 1,
        "codec_model": codec,
        "weight": np.asarray(compact["weight"], dtype=np.float32),
        "bias": np.asarray(compact["bias"], dtype=np.float32),
        "runtime_label_access": False,
        "runtime_parent_access": False,
        "runtime_target_access": False,
    }
    candidates: list[dict[str, Any]] = []
    for hops in LOCALITY_HOPS:
        for span in CONSOLIDATION_SPANS:
            steps = 2 * span
            candidates.append(
                {
                    **base,
                    "candidate_id": f"bounded-h{hops:02d}-c{steps:02d}",
                    "germination_hops": hops,
                    "consolidation_span": span,
                    "consolidation_steps": steps,
                    "bounded": hops <= 5 and steps <= 14,
                    "developmental_field_values_16": 16 * 16 * 16,
                    "developmental_writer_values_16": 16 * 16 * 16,
                    "routing_buffer_values_16": 2 * 16 * 16 * 16,
                    "shared_writer_parameter_bits": (16 * 16 + 16) * 32,
                }
            )
    if len(candidates) != 20:
        raise AssertionError("Stage-6A must contain 20 factorial candidates")
    anchors = {
        "global": _global_candidate(walsh),
        "exact": {**frozen["by_candidate"][EXACT_ID], "candidate_id": EXACT_ANCHOR_ID},
        "compact": next(
            row for row in candidates if row["germination_hops"] == 8 and row["consolidation_span"] == 15
        )
        | {"candidate_id": COMPACT_ANCHOR_ID, "bounded": False},
    }
    anchors["exact"]["original_candidate_id"] = EXACT_ID
    anchors["global"]["candidate_id"] = GLOBAL_ANCHOR_ID
    return candidates, anchors


def build_compression_candidates(
    frozen: dict[str, Any],
    base_candidate: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fit_matrix, _, _ = load_stage3r_fit_matrix(frozen["stage4"]["stage3r"])
    reference_probability = frozen["stage4"]["reference"][32]["motif_probability"]
    walsh = frozen["stage4"]["winner_model"]
    candidates: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for rank, bits in COMPRESSION_SHAPES:
        codec = _codec_model(walsh, rank, bits)
        weight, bias, audit = fit_minimal_writer(fit_matrix, reference_probability, codec)
        candidate = {
            **{key: value for key, value in base_candidate.items() if key not in ("codec_model", "weight", "bias", "candidate_id")},
            "candidate_id": f"compressed-r{rank:02d}-q{bits:02d}-h{base_candidate['germination_hops']:02d}-c{base_candidate['consolidation_steps']:02d}",
            "rank": rank,
            "bits": bits,
            "payload_bits": rank * bits,
            "codec_model": codec,
            "weight": weight,
            "bias": bias,
            "developmental_field_values_16": 16 * 16 * rank,
            "developmental_writer_values_16": 16 * 16 * rank,
            "routing_buffer_values_16": 2 * 16 * 16 * rank,
            "shared_writer_parameter_bits": (rank * rank + rank) * 32,
        }
        candidates.append(candidate)
        audits.append({"candidate_id": candidate["candidate_id"], **audit})
    return candidates, audits


def _quantize(values: np.ndarray, codec: dict[str, Any]) -> tuple[np.ndarray, float]:
    quantized, clipping = quantize_payload(np.asarray(values, dtype=np.float32), codec)
    return quantized, float(clipping)


def _latent_from_counts(
    counts: np.ndarray,
    candidate: dict[str, Any],
    reference_probability: np.ndarray,
    writer_contract: MotifContract,
) -> tuple[np.ndarray, float]:
    """Write a finite payload from motif counts without labels or targets."""

    values = np.asarray(counts, dtype=np.float64)
    codec = candidate["codec_model"]
    basis = np.asarray(codec["basis"], dtype=np.float64)
    signs = basis * math.sqrt(512.0)
    alpha = float(writer_contract.jeffreys_alpha)
    probability = (values + alpha) / (
        values.sum(axis=1, keepdims=True) + 512.0 * alpha
    )
    moments = probability @ signs
    latent = (
        moments @ np.asarray(candidate["weight"], dtype=np.float64)
        + np.asarray(candidate["bias"], dtype=np.float64)
    )
    return _quantize(latent, codec)


def _founder_bounded_payload(
    pair: dict[str, Any],
    candidate: dict[str, Any],
    reference_probability: np.ndarray,
    writer_contract: MotifContract,
    replicates: int,
    extent: int,
    rule: int,
) -> tuple[np.ndarray, np.ndarray]:
    founders = np.stack([_resize_board(board, extent) for board in _founders(pair)])
    counts = collect_trajectory_counts(founders, (32,), rule=rule)[32]
    latent, _ = _latent_from_counts(
        counts["motif"], candidate, reference_probability, writer_contract
    )
    return _repeat_histories(latent, replicates), counts["terminal"]


def _boundary_intervention(
    payload: np.ndarray,
    candidate: dict[str, Any],
    condition: str,
    generation: int,
    pair_id: str,
    replicates: int,
    source_exits: Sequence[np.ndarray] | None,
    contract: MinimalityContract,
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
        result = _swap_histories(result, replicates)
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
            result = _swap_histories(result, replicates)
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
            _hash_seed("stage6-damage", pair_id, candidate["candidate_id"], stress_id, generation, "erase")
        ).random((replicates, result.shape[-1])) < erasure
        result[np.concatenate((half, half), axis=0)] = 0.0
    if sign:
        half = np.random.default_rng(
            _hash_seed("stage6-damage", pair_id, candidate["candidate_id"], stress_id, generation, "sign")
        ).random((replicates, result.shape[-1])) < sign
        result[np.concatenate((half, half), axis=0)] *= -1.0
    return _quantize(result, candidate["codec_model"])


def _field_distance_summary(
    field: np.ndarray,
    occupied: np.ndarray,
    origins: np.ndarray,
) -> dict[str, Any]:
    extent = field.shape[1]
    bands: dict[str, list[float]] = {"0-2": [], "3-5": [], "6-8": [], "9+": []}
    signal: dict[str, list[float]] = {key: [] for key in bands}
    yy, xx = np.indices((extent, extent))
    for sample, (oy, ox) in enumerate(np.asarray(origins, dtype=np.int64)):
        dy = np.minimum(np.abs(yy - oy), extent - np.abs(yy - oy))
        dx = np.minimum(np.abs(xx - ox), extent - np.abs(xx - ox))
        distance = np.maximum(dy, dx)
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
            "occupied_fraction": float(np.mean(bands[key])) if bands[key] else None,
            "mean_abs_signal": float(np.mean(signal[key])) if signal[key] else None,
        }
        for key in bands
    }


def _phenotype_distance_outcomes(
    states: np.ndarray,
    origins: np.ndarray,
    pair: dict[str, Any],
    alive: np.ndarray,
    replicates: int,
    writer_contract: MotifContract,
) -> dict[str, Any]:
    extent = states.shape[1]
    codes = states.astype(np.uint8)
    codes |= np.roll(states, -1, axis=2).astype(np.uint8) << 1
    codes |= np.roll(states, -1, axis=1).astype(np.uint8) << 2
    codes |= (
        np.roll(np.roll(states, -1, axis=1), -1, axis=2).astype(np.uint8) << 3
    )
    yy, xx = np.indices((extent, extent))
    masks_by_band: dict[str, list[np.ndarray]] = {
        "0-2": [],
        "3-5": [],
        "6-8": [],
        "9+": [],
    }
    for oy, ox in np.asarray(origins, dtype=np.int64):
        dy = np.minimum(np.abs(yy - oy), extent - np.abs(yy - oy))
        dx = np.minimum(np.abs(xx - ox), extent - np.abs(xx - ox))
        distance = np.maximum(dy, dx)
        masks_by_band["0-2"].append(distance <= 2)
        masks_by_band["3-5"].append((distance >= 3) & (distance <= 5))
        masks_by_band["6-8"].append((distance >= 6) & (distance <= 8))
        masks_by_band["9+"].append(distance >= 9)
    targets = pair["targets"].get("primary_terminal", pair["targets"]["primary"])
    result: dict[str, Any] = {}
    for key, masks in masks_by_band.items():
        counts = np.zeros((len(states), 15), dtype=np.float64)
        for sample, mask in enumerate(masks):
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


def predecode_local_field(field: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Decode a static local field once; exactly the local Walsh dot product."""

    return np.einsum(
        "...c,mc->...m",
        np.asarray(field, dtype=np.float32),
        np.asarray(basis, dtype=np.float32),
        optimize=True,
    ).astype(np.float32)


def apply_predecoded_local_reader(
    predicted: np.ndarray,
    energy_field: np.ndarray,
    uniforms: np.ndarray,
    strength: float,
) -> np.ndarray:
    result = np.asarray(predicted, dtype=np.bool_).copy()
    codes = motif3_codes(result)
    advantage = np.zeros(result.shape, dtype=np.float32)
    for bit, (dy, dx) in enumerate(
        (offset for offset in ((y, x) for y in (-1, 0, 1) for x in (-1, 0, 1)))
    ):
        affected_codes = np.roll(codes, shift=(dy, dx), axis=(1, 2))
        affected_energy = np.roll(energy_field, shift=(dy, dx), axis=(1, 2))
        current = np.take_along_axis(
            affected_energy, affected_codes[..., None], axis=-1
        )[..., 0]
        flipped = np.take_along_axis(
            affected_energy,
            (affected_codes ^ np.uint16(1 << bit))[..., None],
            axis=-1,
        )[..., 0]
        advantage += flipped - current
    probability = np.float32(strength) * np.tanh(
        np.maximum(advantage, 0.0) / np.float32(9.0)
    )
    result[np.asarray(uniforms) < probability] ^= True
    return result


def apply_masked_payload_reader(
    predicted: np.ndarray,
    carrier: np.ndarray,
    occupied: np.ndarray,
    uniforms: np.ndarray,
    strength: float,
) -> np.ndarray:
    """Fast exact reader for a copied one-seed payload inside its occupied cone."""

    result = np.asarray(predicted, dtype=np.bool_).copy()
    codes = motif3_codes(result)
    values = np.asarray(carrier, dtype=np.float32)
    advantage = np.zeros(result.shape, dtype=np.float32)
    for bit, (dy, dx) in enumerate(
        (offset for offset in ((y, x) for y in (-1, 0, 1) for x in (-1, 0, 1)))
    ):
        affected_codes = np.roll(codes, shift=(dy, dx), axis=(1, 2))
        mask = np.roll(occupied, shift=(dy, dx), axis=(1, 2))
        flat = affected_codes.reshape(len(result), -1)
        current = np.take_along_axis(values, flat, axis=1).reshape(result.shape)
        flipped = np.take_along_axis(
            values, (flat ^ np.uint16(1 << bit)), axis=1
        ).reshape(result.shape)
        advantage += (flipped - current) * mask
    probability = np.float32(strength) * np.tanh(
        np.maximum(advantage, 0.0) / np.float32(9.0)
    )
    result[np.asarray(uniforms) < probability] ^= True
    return result


def simulate_bounded_lineage(
    pair: dict[str, Any],
    configuration: ReaderConfiguration,
    candidate: dict[str, Any],
    condition: str,
    replicates: int,
    generations: int,
    reference: dict[int, dict[str, np.ndarray]],
    writer_contract: MotifContract,
    contract: MinimalityContract,
    *,
    extent: int = 16,
    stress_id: str = "ordinary",
    stress: dict[str, float | int] | None = None,
    source_exits: Sequence[np.ndarray] | None = None,
    retain_exits: bool = False,
    rule_override: int | None = None,
) -> tuple[dict[str, Any], list[np.ndarray]]:
    """Run a reset lineage with finite-hop read and finite-window rewrite."""

    valid = QUALIFICATION_CONDITIONS
    if condition not in valid:
        raise ValueError(f"unknown Stage-6 condition {condition!r}")
    if extent not in (16, 32, 64):
        raise ValueError("Stage-6 extents are registered at 16, 32, and 64")
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
    checkpoints = {value for value in (1, 2, 4, 8, 16, 32, 64) if value <= generations}
    outcomes: dict[str, Any] = {}
    decoders: dict[str, Any] = {}
    carrier_history: dict[str, Any] = {}
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
        origins = dynamic_origins(pair_id, generation, replicates, extent)
        field, occupied = embed_dynamic_seed(
            entry_payload,
            origins,
            extent,
            translated=condition == "translated_patch",
        )
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
        local_carrier = decode_payload(
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
            (2 * replicates, extent, extent, int(candidate["rank"])), dtype=np.float32
        )
        for sweep in range(1, contract.generation_sweeps + 1):
            predicted = _step(state, rule)
            if condition != "read_disabled" and sweep <= contract.read_sweeps:
                uniforms = _uniforms(
                    pair_id, candidate_id, "read", generation, sweep, replicates, extent
                )
                if uniform:
                    carrier = decode_payload(field[:, 0, 0], codec)
                    predicted = apply_energy_reader(
                        predicted, carrier, uniforms, configuration.strength * repair_gain / 0.50
                    )
                else:
                    predicted = apply_masked_payload_reader(
                        predicted,
                        local_carrier,
                        occupied,
                        uniforms,
                        configuration.strength * repair_gain / 0.50,
                    )
            predicted ^= _uniforms(
                pair_id, candidate_id, "process", generation, sweep, replicates, extent
            ) < process_noise
            predicted[~alive] = False
            state = predicted
            if contract.write_start <= sweep <= contract.write_end:
                site_sign_sum += signs[motif3_codes(state)].astype(np.float32)
            if sweep >= contract.observe_start:
                recent.append(live_2x2_counts_batch(state))
        alive &= state.any(axis=(1, 2))

        next_origins = dynamic_origins(pair_id, generation + 1, replicates, extent)
        if condition == "no_rewrite":
            next_payload, clipping = _quantize(
                entry_payload * np.float32(contract.stale_retention), codec
            )
        elif condition == "write_disabled":
            next_payload = np.zeros_like(entry_payload)
            clipping = 0.0
        else:
            selected_span = 0 if condition == "consolidation_disabled" else span
            endpoint_sum = bounded_reduce_endpoint(
                site_sign_sum, selected_span, next_origins
            )
            observations = float(
                (contract.write_end - contract.write_start + 1)
                * (selected_span + 1) ** 2
            )
            alpha = float(writer_contract.jeffreys_alpha)
            moments = (endpoint_sum + alpha * signs.sum(axis=0)) / (
                observations + 512.0 * alpha
            )
            latent = (
                float(stress.get("writer_gain", 1.0))
                * (moments @ np.asarray(candidate["weight"], dtype=np.float64))
                + np.asarray(candidate["bias"], dtype=np.float64)
            )
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
            decoders[str(generation)] = {
                "carrier_balanced_accuracy": heldout_lineage_accuracy(
                    payload,
                    replicates,
                    _hash_seed(
                        contract.namespace,
                        pair_id,
                        candidate_id,
                        condition,
                        stress_id,
                        generation,
                        "carrier",
                    ),
                    contract.decoder_splits,
                ),
                "phenotype_balanced_accuracy": heldout_lineage_accuracy(
                    phenotype,
                    replicates,
                    _hash_seed(
                        contract.namespace,
                        pair_id,
                        candidate_id,
                        condition,
                        stress_id,
                        generation,
                        "phenotype",
                    ),
                    contract.decoder_splits,
                ),
            }
            decoders[str(generation)]["carrier_information_lower_bound_bits"] = (
                _binary_information_lower_bound(
                    decoders[str(generation)]["carrier_balanced_accuracy"]
                )
            )
            decoders[str(generation)]["phenotype_information_lower_bound_bits"] = (
                _binary_information_lower_bound(
                    decoders[str(generation)]["phenotype_balanced_accuracy"]
                )
            )
            carrier_history[str(generation)] = {
                "entry": _payload_summary(entry_payload, replicates),
                "exit": _payload_summary(payload, replicates),
                "occupied_fraction_after_germination": float(np.mean(occupied)),
                "uniform_after_germination": uniform,
                "wave_trace": wave_trace,
                "distance_bands": field_distance,
                "surviving_futures": int(np.count_nonzero(alive)),
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
            "developmental_field_values": extent * extent * int(candidate["rank"]),
            "developmental_writer_values": extent * extent * int(candidate["rank"]),
            "founder_payload": _payload_summary(founder_payload, replicates),
            "boundary_clipping_fraction_mean": float(np.mean(clipping_values)),
            "germination_coverage_mean": float(np.mean(coverage_values)),
            "outcomes": outcomes,
            "decoders": decoders,
            "carrier_history": carrier_history,
        },
        exits,
    )


def _lineage_task(
    payload: tuple[
        dict[str, Any],
        list[dict[str, Any]],
        MotifContract,
        MinimalityContract,
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
    stress_id = str(item.get("stress_id", "ordinary"))
    stress = dict(item.get("stress", {}))
    extent = int(item.get("extent", 16))
    rule = int(item.get("rule", contract.rule))
    results: dict[str, Any] = {}
    exits: list[np.ndarray] | None = None
    if "intact" in conditions or any("rescue" in value for value in conditions):
        intact, exits = simulate_bounded_lineage(
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
            stress_id=stress_id,
            stress=stress,
            retain_exits=True,
            rule_override=rule,
        )
        if "intact" in conditions:
            results["intact"] = intact
    for condition in conditions:
        if condition == "intact":
            continue
        result, _ = simulate_bounded_lineage(
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
            stress_id=stress_id,
            stress=stress,
            source_exits=exits,
            rule_override=rule,
        )
        results[condition] = result
    return {
        "checkpoint": item["checkpoint"],
        "pair_id": item["pair"]["pair_id"],
        "replicates": replicates,
        "generations": generations,
        "extent": extent,
        "rule": rule,
        "candidate_id": candidate["candidate_id"],
        "stress_id": stress_id,
        "conditions": results,
    }


def _stage5r_anchor_task(
    payload: tuple[
        dict[str, Any],
        list[dict[str, Any]],
        MotifContract,
        MinimalityContract,
        dict[int, dict[str, np.ndarray]],
    ]
) -> dict[str, Any]:
    item, candidates, writer_contract, _contract, reference = payload
    candidate = next(
        row for row in candidates if row["candidate_id"] == item["candidate_id"]
    )
    simulation_candidate = candidate
    if candidate["candidate_id"] == EXACT_ANCHOR_ID:
        simulation_candidate = {**candidate, "candidate_id": EXACT_ID}
    result, _ = _simulate_stage5r_candidate(
        item["pair"],
        ReaderConfiguration(**item["configuration"]),
        simulation_candidate,
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
        "extent": 16,
        "rule": RULE,
        "stress_id": "ordinary",
        "conditions": {"intact": result},
    }


def _json_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    codec = candidate["codec_model"]
    return {
        key: value
        for key, value in candidate.items()
        if key not in ("codec_model", "weight", "bias")
    } | {
        "codec_id": codec["candidate_id"],
        "codec_family": codec["family"],
        "codec_rank": int(codec["rank"]),
        "codec_bits": int(codec["bits"]),
        "codec_basis_sha256": hashlib.sha256(
            np.asarray(codec["basis"], dtype=np.float32).tobytes()
        ).hexdigest(),
        "writer_weight_sha256": hashlib.sha256(
            np.asarray(candidate["weight"], dtype=np.float32).tobytes()
        ).hexdigest(),
    }


def save_candidate_models(
    output: Path,
    filename: str,
    candidates: Sequence[dict[str, Any]],
    design_digest: str,
) -> dict[str, Any]:
    arrays: dict[str, np.ndarray] = {}
    metadata: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        prefix = f"c{index:03d}"
        codec = candidate["codec_model"]
        arrays[f"{prefix}_basis"] = np.asarray(codec["basis"], dtype=np.float32)
        arrays[f"{prefix}_scale"] = np.asarray(
            codec["quantizer_scale"], dtype=np.float32
        )
        arrays[f"{prefix}_weight"] = np.asarray(candidate["weight"], dtype=np.float32)
        arrays[f"{prefix}_bias"] = np.asarray(candidate["bias"], dtype=np.float32)
        metadata.append({"array_prefix": prefix, **_json_candidate(candidate)})
    array_path = output / f"{filename}.npz"
    _atomic_npz(array_path, **arrays)
    payload = {
        "design_digest": design_digest,
        "candidate_count": len(candidates),
        "candidate_ids": [str(row["candidate_id"]) for row in candidates],
        "array_sha256": _sha256(array_path),
        "candidates": metadata,
    }
    _atomic_json(output / f"{filename}.json", payload)
    return payload


def load_candidate_models(
    output: Path, filename: str, design_digest: str
) -> list[dict[str, Any]]:
    metadata = _load_json(output / f"{filename}.json")
    array_path = output / f"{filename}.npz"
    if metadata.get("design_digest") != design_digest:
        raise ValueError(f"{filename} belongs to another Stage-6 design")
    if _sha256(array_path) != metadata.get("array_sha256"):
        raise ValueError(f"{filename} array hash mismatch")
    candidates: list[dict[str, Any]] = []
    with np.load(array_path, allow_pickle=False) as arrays:
        for row in metadata["candidates"]:
            prefix = str(row["array_prefix"])
            candidate = {
                key: value
                for key, value in row.items()
                if key
                not in (
                    "array_prefix",
                    "codec_id",
                    "codec_family",
                    "codec_rank",
                    "codec_bits",
                    "codec_basis_sha256",
                    "writer_weight_sha256",
                )
            }
            candidate["codec_model"] = {
                "candidate_id": row["codec_id"],
                "family": row["codec_family"],
                "rank": int(row["codec_rank"]),
                "bits": int(row["codec_bits"]),
                "payload_bits": int(row["codec_rank"] * row["codec_bits"]),
                "basis": arrays[f"{prefix}_basis"].copy(),
                "quantizer_scale": arrays[f"{prefix}_scale"].copy(),
                "runtime_label_access": False,
                "runtime_parent_access": False,
                "runtime_target_access": False,
            }
            candidate["weight"] = arrays[f"{prefix}_weight"].copy()
            candidate["bias"] = arrays[f"{prefix}_bias"].copy()
            candidates.append(candidate)
    if [row["candidate_id"] for row in candidates] != metadata["candidate_ids"]:
        raise ValueError(f"{filename} candidate order changed")
    return candidates


def _repair_profile_for(
    profile: MinimalityProfile,
    role: str,
    pairs: int,
    replicates: int,
    generations: int,
) -> RepairProfile:
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


def _as_strict_rows(
    rows: Sequence[dict[str, Any]], candidate_id: str, *, environment: str | None = None
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        if row.get("candidate_id") != candidate_id:
            continue
        if environment is not None and row.get("stress_id") != environment:
            continue
        result.append(
            {
                "pair_id": row["pair_id"],
                "candidates": {
                    candidate_id: {"conditions": row["conditions"]}
                },
            }
        )
    return result


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
        values.append(
            float(outcome["survival"] if metric == "survival" else outcome["primary"][metric])
        )
    return values


def _paired_advantage(
    rows: Sequence[dict[str, Any]],
    candidate_id: str,
    control: str,
    generation: int,
) -> list[float]:
    intact = _condition_values(rows, candidate_id, "intact", generation)
    other = _condition_values(rows, candidate_id, control, generation)
    if len(intact) != len(other):
        raise ValueError(f"paired Stage-6 values do not align for {control}")
    return [left - right for left, right in zip(intact, other)]


def _decoder_values(
    rows: Sequence[dict[str, Any]],
    candidate_id: str,
    condition: str,
    generation: int,
    key: str,
) -> list[float]:
    values: list[float] = []
    for row in rows:
        if row.get("candidate_id") != candidate_id:
            continue
        try:
            values.append(
                float(row["conditions"][condition]["decoders"][str(generation)][key])
            )
        except KeyError:
            continue
    return values


def compression_pareto_frontier(
    candidates: Sequence[dict[str, Any]], screen: dict[str, Any]
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        points.append(
            {
                "candidate_id": candidate_id,
                "payload_bits": int(candidate["payload_bits"]),
                "writer_values": int(candidate["developmental_writer_values_16"]),
                "shared_parameter_bits": int(candidate["shared_writer_parameter_bits"]),
                "communication_steps": int(candidate["germination_hops"])
                + int(candidate["consolidation_steps"]),
                "generation8_crossover": float(
                    screen["candidate_summaries"][candidate_id]["crossover"]["mean"]
                    or 0.0
                ),
            }
        )
    frontier: list[dict[str, Any]] = []
    for point in points:
        dominated = False
        for other in points:
            if other is point:
                continue
            no_worse = bool(
                other["payload_bits"] <= point["payload_bits"]
                and other["writer_values"] <= point["writer_values"]
                and other["shared_parameter_bits"] <= point["shared_parameter_bits"]
                and other["communication_steps"] <= point["communication_steps"]
                and other["generation8_crossover"] >= point["generation8_crossover"]
            )
            strictly_better = any(
                other[key] != point[key]
                for key in (
                    "payload_bits",
                    "writer_values",
                    "shared_parameter_bits",
                    "communication_steps",
                    "generation8_crossover",
                )
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(point)
    return sorted(frontier, key=lambda row: (row["payload_bits"], row["candidate_id"]))


def _boot(
    values: Sequence[float],
    profile: MinimalityProfile,
    contract: MinimalityContract,
    *key: object,
    alpha: float | None = None,
) -> dict[str, Any]:
    return _bootstrap(
        values,
        profile.bootstrap_resamples,
        _hash_seed(contract.namespace, *key),
        contract.strict_alpha if alpha is None else alpha,
    )


def summarize_screen(
    rows: Sequence[dict[str, Any]],
    candidates: Sequence[dict[str, Any]],
    anchor_id: str,
    profile: MinimalityProfile,
    contract: MinimalityContract,
    *,
    scientific_gate: bool,
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        values = _condition_values(rows, candidate_id, "intact", 8)
        summaries[candidate_id] = {
            "candidate": _json_candidate(candidate),
            "crossover": _boot(values, profile, contract, "screen", candidate_id),
            "survival_mean": float(
                np.mean(_condition_values(rows, candidate_id, "intact", 8, "survival"))
            ),
            "direction_a_mean": float(
                np.mean(_condition_values(rows, candidate_id, "intact", 8, "direction_a"))
            ),
            "direction_b_mean": float(
                np.mean(_condition_values(rows, candidate_id, "intact", 8, "direction_b"))
            ),
            "fraction_pairs_positive": float(np.mean(np.asarray(values) > 0.0)),
        }
    anchor_mean = float(summaries[anchor_id]["crossover"]["mean"] or 0.0)
    eligible: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        summary = summaries[candidate_id]
        mean = float(summary["crossover"]["mean"] or 0.0)
        lower = summary["crossover"]["ci"][0]
        retention = mean / anchor_mean if anchor_mean > 0.0 else 0.0
        summary["anchor_retention"] = retention
        summary["screen_eligible"] = bool(
            candidate_id != anchor_id
            and candidate.get("bounded")
            and mean >= contract.screen_generation8
            and lower is not None
            and float(lower) > 0.0
            and summary["survival_mean"] >= contract.survival_gate
            and summary["direction_a_mean"] > 0.0
            and summary["direction_b_mean"] > 0.0
            and summary["fraction_pairs_positive"] >= 0.50
            and retention >= contract.screen_anchor_retention
        )
        if summary["screen_eligible"]:
            eligible.append(candidate)
    ranked = sorted(
        eligible
        or [row for row in candidates if row.get("bounded")],
        key=lambda row: (
            -float(summaries[row["candidate_id"]]["crossover"]["mean"] or -1.0),
            int(row["germination_hops"]) + int(row["consolidation_steps"]),
            row["candidate_id"],
        ),
    )
    selected: list[str] = []
    if ranked:
        selected.append(str(ranked[0]["candidate_id"]))
        cheapest = min(
            ranked,
            key=lambda row: (
                int(row["germination_hops"]) + int(row["consolidation_steps"]),
                -float(summaries[row["candidate_id"]]["crossover"]["mean"] or -1.0),
            ),
        )
        selected.append(str(cheapest["candidate_id"]))
        deepest_writer = min(
            ranked,
            key=lambda row: (
                -int(row["consolidation_steps"]),
                int(row["germination_hops"]),
                -float(summaries[row["candidate_id"]]["crossover"]["mean"] or -1.0),
            ),
        )
        selected.append(str(deepest_writer["candidate_id"]))
    selected = list(dict.fromkeys(selected))[:2]
    return {
        "state": "complete",
        "scientific_gate_applied": scientific_gate,
        "anchor_id": anchor_id,
        "anchor_crossover_mean": anchor_mean,
        "candidate_summaries": summaries,
        "selected_candidate_ids": [anchor_id, *selected],
        "fallback_nomination_used": not bool(eligible),
    }


def summarize_locality_qualification(
    rows: Sequence[dict[str, Any]],
    candidates: Sequence[dict[str, Any]],
    profile: MinimalityProfile,
    contract: MinimalityContract,
    *,
    anchor_id: str,
    scientific_gate: bool,
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    qualified: list[str] = []
    anchor_generation16 = float(
        np.mean(_condition_values(rows, anchor_id, "intact", 16))
    )
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        transformed = _as_strict_rows(rows, candidate_id)
        strict = (
            _strict_confirmation_gate(
                transformed,
                candidate_id,
                _repair_profile_for(
                    profile,
                    "stage6-locality",
                    len(transformed),
                    profile.locality_qualification_replicates,
                    32,
                ),
                contract,  # type: ignore[arg-type]
                contract.strict_alpha,
            )
            if scientific_gate
            else {"verdict": "NOT_ADJUDICATED_SMOKE", "renewed_gate": False}
        )
        intact32 = _boot(
            _condition_values(rows, candidate_id, "intact", 32),
            profile,
            contract,
            "locality",
            candidate_id,
            "intact32",
        )
        intact16_values = _condition_values(rows, candidate_id, "intact", 16)
        intact16_mean = float(np.mean(intact16_values)) if intact16_values else 0.0
        anchor_retention = (
            intact16_mean / anchor_generation16 if anchor_generation16 > 0.0 else 0.0
        )
        controls = {
            control: _boot(
                _paired_advantage(rows, candidate_id, control, 8),
                profile,
                contract,
                "locality",
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
            "locality",
            candidate_id,
            "translation",
        )
        retention = (
            float(translated["mean"] or 0.0) / intact8 if intact8 > 0.0 else 0.0
        )
        bounded = bool(
            candidate.get("bounded")
            and int(candidate["germination_hops"]) <= contract.bounded_hops
            and int(candidate["consolidation_steps"])
            <= contract.bounded_consolidation_steps
        )
        locality_pass = bool(
            strict.get("renewed_gate")
            and bounded
            and anchor_retention >= contract.screen_anchor_retention
            and intact32["ci"][0] is not None
            and float(intact32["ci"][0]) > 0.0
            and all(
                summary["ci"][0] is not None
                and float(summary["ci"][0]) > 0.0
                and float(summary["mean"] or 0.0) >= contract.control_advantage
                for summary in controls.values()
            )
            and translated["ci"][0] is not None
            and float(translated["ci"][0]) > 0.0
            and retention >= contract.translation_retention
        )
        summaries[candidate_id] = {
            "candidate": _json_candidate(candidate),
            "strict": strict,
            "generation32": intact32,
            "generation16_anchor_retention": anchor_retention,
            "targeted_control_advantages_generation8": controls,
            "translated_generation8": translated,
            "translation_retention": retention,
            "finite_light_cone_audit": bounded,
            "locality_gate": locality_pass,
        }
        if locality_pass or not scientific_gate:
            qualified.append(candidate_id)
    return {
        "state": "complete",
        "scientific_gate_applied": scientific_gate,
        "anchor_id": anchor_id,
        "anchor_generation16_crossover_mean": anchor_generation16,
        "candidate_summaries": summaries,
        "qualified_candidate_ids": qualified,
    }


def stage6_mechanism_audit(contract: MinimalityContract) -> dict[str, Any]:
    rng = np.random.default_rng(_hash_seed(contract.namespace, "mechanism-audit"))
    values = rng.normal(size=(3, 16, 16, 5))
    explicit = bounded_shift_reduce(values, 7)
    origins = np.asarray([[0, 0], [5, 7], [11, 3]], dtype=np.int16)
    endpoint = bounded_reduce_endpoint(values, 7, origins)
    selected = explicit[np.arange(3), origins[:, 0], origins[:, 1]]
    reduction_error = float(np.max(np.abs(selected - endpoint)))
    payload = np.ones((1, 3), dtype=np.float32)
    origin = np.asarray([[8, 8]], dtype=np.int16)
    field, occupied = embed_dynamic_seed(payload, origin, 16)
    field, occupied, _ = propagate_bounded(field, occupied, 3)
    light_cone = True
    for y, x in np.argwhere(occupied[0]):
        dy = min(abs(int(y) - 8), 16 - abs(int(y) - 8))
        dx = min(abs(int(x) - 8), 16 - abs(int(x) - 8))
        light_cone &= max(dy, dx) <= 3
    zero_field, zero_occupied = embed_dynamic_seed(
        np.zeros_like(payload), origin, 16
    )
    zero_field, _, _ = propagate_bounded(zero_field, zero_occupied, 5)
    zero_stable = bool(np.array_equal(zero_field, np.zeros_like(zero_field)))
    impulse = np.zeros((1, 16, 16, 1), dtype=np.float64)
    impulse[0, 0, 0, 0] = 1.0
    endpoint_origin = np.asarray([[8, 8]], dtype=np.int16)
    unreachable_zero = bool(
        bounded_reduce_endpoint(impulse, 7, endpoint_origin)[0, 0] == 0.0
    )
    impulse[0, 0, 0, 0] = 0.0
    impulse[0, 8, 8, 0] = 1.0
    reachable_nonzero = bool(
        bounded_reduce_endpoint(impulse, 7, endpoint_origin)[0, 0] > 0.0
    )
    passed = bool(
        reduction_error <= 1e-12
        and light_cone
        and zero_stable
        and unreachable_zero
        and reachable_nonzero
    )
    return {
        "passed": passed,
        "bounded_reduction_endpoint_max_abs_error": reduction_error,
        "finite_propagation_light_cone": light_cone,
        "exact_zero_stability": zero_stable,
        "unreachable_writer_impulse_exactly_zero": unreachable_zero,
        "reachable_writer_impulse_nonzero": reachable_nonzero,
        "runtime_label_parent_target_access": False,
        "routing_steps_primary_bound": contract.bounded_consolidation_steps,
    }


def build_scale_variants(
    bounded: dict[str, Any], anchor: dict[str, Any]
) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    for source_name, source in (("bounded", bounded), ("anchor", anchor)):
        for extent in (16, 32, 64):
            fixed = {
                **source,
                "candidate_id": f"scale-{source_name}-n{extent:02d}-fixed",
                "source_candidate_id": source["candidate_id"],
                "scale_mode": "fixed",
                "extent": extent,
            }
            scaled = {
                **source,
                "candidate_id": f"scale-{source_name}-n{extent:02d}-diameter",
                "source_candidate_id": source["candidate_id"],
                "scale_mode": "diameter",
                "extent": extent,
                "germination_hops": extent // 2,
                "consolidation_span": extent - 1,
                "consolidation_steps": 2 * (extent - 1),
                "bounded": False,
            }
            variants.extend((fixed, scaled))
    return variants


def summarize_scale(
    rows: Sequence[dict[str, Any]],
    variants: Sequence[dict[str, Any]],
    profile: MinimalityProfile,
    contract: MinimalityContract,
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for candidate in variants:
        candidate_id = str(candidate["candidate_id"])
        values = _condition_values(rows, candidate_id, "intact", 8)
        band_values: dict[str, list[float]] = {key: [] for key in ("0-2", "3-5", "6-8", "9+")}
        carrier_coverage: dict[str, list[float]] = {key: [] for key in band_values}
        for row in rows:
            if row.get("candidate_id") != candidate_id:
                continue
            try:
                outcome_bands = row["conditions"]["intact"]["outcomes"]["8"]["distance_bands"]
                field_bands = row["conditions"]["intact"]["carrier_history"]["8"]["distance_bands"]
            except KeyError:
                continue
            for key in band_values:
                band_values[key].append(float(outcome_bands[key]["crossover"]))
                occupied = field_bands[key]["occupied_fraction"]
                if occupied is not None:
                    carrier_coverage[key].append(float(occupied))
        summaries[candidate_id] = {
            "candidate": _json_candidate(candidate),
            "global_crossover": _boot(values, profile, contract, "scale", candidate_id),
            "distance_band_terminal_crossover": {
                key: _boot(values, profile, contract, "scale", candidate_id, key)
                for key, values in band_values.items()
            },
            "carrier_occupied_fraction_by_band": {
                key: float(np.mean(values)) if values else None
                for key, values in carrier_coverage.items()
            },
        }
    retention: dict[str, Any] = {}
    for source_name in ("bounded", "anchor"):
        for mode in ("fixed", "diameter"):
            baseline_id = f"scale-{source_name}-n16-{mode}"
            baseline = float(summaries[baseline_id]["global_crossover"]["mean"] or 0.0)
            retention[f"{source_name}-{mode}"] = {
                str(extent): (
                    float(
                        summaries[f"scale-{source_name}-n{extent:02d}-{mode}"]["global_crossover"]["mean"]
                        or 0.0
                    )
                    / baseline
                    if baseline > 0.0
                    else 0.0
                )
                for extent in (16, 32, 64)
            }
    bounded_fixed_variants = [
        row
        for row in variants
        if row["candidate_id"].startswith("scale-bounded-")
        and row["scale_mode"] == "fixed"
    ]
    light_cone_pass = True
    fixed_in_cone: dict[str, float] = {}
    for candidate in bounded_fixed_variants:
        candidate_id = str(candidate["candidate_id"])
        hops = int(candidate["germination_hops"])
        inside_keys = ["0-2"] if hops <= 2 else ["0-2", "3-5"]
        outside_keys = (
            ["3-5", "6-8", "9+"] if hops <= 2 else ["6-8", "9+"]
        )
        coverage = summaries[candidate_id]["carrier_occupied_fraction_by_band"]
        light_cone_pass &= all(coverage[key] in (None, 0.0) for key in outside_keys)
        inside_means = [
            summaries[candidate_id]["distance_band_terminal_crossover"][key]["mean"]
            for key in inside_keys
        ]
        fixed_in_cone[str(candidate["extent"])] = float(
            np.mean([float(value or 0.0) for value in inside_means])
        )
    fixed_baseline = fixed_in_cone.get("16", 0.0)
    fixed_in_cone_retention = {
        extent: value / fixed_baseline if fixed_baseline > 0.0 else 0.0
        for extent, value in fixed_in_cone.items()
    }
    fixed_retention_pass = bool(
        fixed_in_cone_retention.get("32", 0.0) >= contract.retention_fraction
        and fixed_in_cone_retention.get("64", 0.0) >= contract.retention_fraction
    )
    diameter_retention = retention["bounded-diameter"]
    scaled_pass = bool(
        diameter_retention["32"] >= contract.retention_fraction
        and diameter_retention["64"] >= contract.retention_fraction
    )
    return {
        "state": "complete",
        "generation": 8,
        "candidate_summaries": summaries,
        "global_retention_relative_to_n16": retention,
        "fixed_budget_exact_light_cone_pass": light_cone_pass,
        "diameter_scaled_retention_pass": scaled_pass,
        "fixed_budget_in_cone_crossover": fixed_in_cone,
        "fixed_budget_in_cone_retention": fixed_in_cone_retention,
        "fixed_budget_in_cone_retention_pass": fixed_retention_pass,
        "scale_gate": bool(light_cone_pass and fixed_retention_pass and scaled_pass),
        "distance_band_observer": "terminal live-2x2 texture; carrier support recorded separately",
    }


def summarize_rule_domain(
    rows: Sequence[dict[str, Any]],
    candidate_id: str,
    profile: MinimalityProfile,
    contract: MinimalityContract,
) -> dict[str, Any]:
    rules: dict[str, Any] = {}
    for rule in sorted({int(row["rule"]) for row in rows}):
        selected = [row for row in rows if int(row["rule"]) == rule]
        values = _condition_values(selected, candidate_id, "intact", 8)
        rules[str(rule)] = {
            "bit_distance_from_31649": (rule ^ RULE).bit_count(),
            "crossover": _boot(values, profile, contract, "rule-domain", rule),
        }
    return {
        "state": "complete",
        "generation": 8,
        "candidate_id": candidate_id,
        "rules": rules,
        "registered_positive_anchor": 31648,
        "registered_negative_anchor": 70366,
        "exploratory_only_no_refitting": True,
    }


def _environment_gate(
    rows: Sequence[dict[str, Any]],
    candidate: dict[str, Any],
    profile: MinimalityProfile,
    contract: MinimalityContract,
    environment: str,
    pairs: int,
    replicates: int,
    generations: int,
    alpha: float,
    scientific_gate: bool,
) -> dict[str, Any]:
    candidate_id = str(candidate["candidate_id"])
    selected = [row for row in rows if row.get("stress_id") == environment]
    transformed = _as_strict_rows(selected, candidate_id)
    strict = (
        _strict_confirmation_gate(
            transformed,
            candidate_id,
            _repair_profile_for(
                profile,
                f"stage6-{environment}",
                pairs,
                replicates,
                generations,
            ),
            contract,  # type: ignore[arg-type]
            alpha,
        )
        if scientific_gate
        else {"verdict": "NOT_ADJUDICATED_PROFILE", "renewed_gate": False}
    )
    targeted: dict[str, Any] = {}
    for control in (
        "transport_disabled",
        "regeneration_disabled",
        "consolidation_disabled",
        "communication_cut",
    ):
        targeted[control] = _boot(
            _paired_advantage(selected, candidate_id, control, 8),
            profile,
            contract,
            "environment",
            candidate_id,
            environment,
            control,
            alpha=alpha,
        )
    intact8 = float(np.mean(_condition_values(selected, candidate_id, "intact", 8)))
    translated = _boot(
        _condition_values(selected, candidate_id, "translated_patch", 8),
        profile,
        contract,
        "environment",
        candidate_id,
        environment,
        "translated",
        alpha=alpha,
    )
    translation_retention = (
        float(translated["mean"] or 0.0) / intact8 if intact8 > 0.0 else 0.0
    )
    targeted_pass = bool(
        all(
            summary["ci"][0] is not None
            and float(summary["ci"][0]) > 0.0
            and float(summary["mean"] or 0.0) >= contract.control_advantage
            for summary in targeted.values()
        )
        and translated["ci"][0] is not None
        and float(translated["ci"][0]) > 0.0
        and translation_retention >= contract.translation_retention
    )
    return {
        "strict": strict,
        "targeted_control_advantages_generation8": targeted,
        "translated_generation8": translated,
        "translation_retention": translation_retention,
        "targeted_locality_gate": targeted_pass,
        "gate": bool(strict.get("renewed_gate") and targeted_pass),
    }


def summarize_compression_qualification(
    rows: Sequence[dict[str, Any]],
    candidates: Sequence[dict[str, Any]],
    anchor_id: str,
    profile: MinimalityProfile,
    contract: MinimalityContract,
    *,
    scientific_gate: bool,
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    anchor_ordinary = float(
        np.mean(
            _condition_values(
                [row for row in rows if row.get("stress_id") == "ordinary"],
                anchor_id,
                "intact",
                16,
            )
        )
    )
    qualified: list[str] = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        environments = {
            environment: _environment_gate(
                rows,
                candidate,
                profile,
                contract,
                environment,
                profile.compression_qualification_pairs,
                profile.compression_qualification_replicates,
                32,
                contract.strict_alpha,
                scientific_gate,
            )
            for environment in ("ordinary", "moderate_joint")
        }
        ordinary_values = _condition_values(
            [row for row in rows if row.get("stress_id") == "ordinary"],
            candidate_id,
            "intact",
            16,
        )
        mean = float(np.mean(ordinary_values)) if ordinary_values else 0.0
        retention = mean / anchor_ordinary if anchor_ordinary > 0.0 else 0.0
        compact_primary = bool(
            int(candidate["payload_bits"]) <= 32
            and int(candidate["rank"]) <= 8
            and retention >= contract.retention_fraction
            and environments["ordinary"]["gate"]
            and environments["moderate_joint"]["gate"]
        )
        summaries[candidate_id] = {
            "candidate": _json_candidate(candidate),
            "environments": environments,
            "generation16_anchor_retention": retention,
            "compact_primary_gate": compact_primary,
            "stretch_8_or_4_bit": bool(
                compact_primary and int(candidate["payload_bits"]) <= 8
            ),
        }
        if compact_primary or (not scientific_gate and candidate_id != anchor_id):
            qualified.append(candidate_id)
    return {
        "state": "complete",
        "scientific_gate_applied": scientific_gate,
        "anchor_id": anchor_id,
        "anchor_generation16_crossover_mean": anchor_ordinary,
        "candidate_summaries": summaries,
        "qualified_candidate_ids": qualified,
        "smallest_qualified_candidate_id": (
            min(
                qualified,
                key=lambda value: (
                    int(summaries[value]["candidate"]["payload_bits"]), value
                ),
            )
            if qualified
            else None
        ),
    }


ECOLOGY_SCENARIOS = (
    "same_history",
    "opposite_history",
    "intact_corrupted",
    "unequal_precision",
    "half_channel_chimera",
)
ECOLOGY_SEPARATIONS = (2, 5, 8)
EVOLUTION_TREATMENTS = (
    "autocorrelated",
    "iid_histories",
    "inheritance_disabled",
    "payload_shuffled",
    "rewrite_disabled",
)


def _cosine_batch(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.sum(left * right, axis=-1)
    denominator = np.linalg.norm(left, axis=-1) * np.linalg.norm(right, axis=-1)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator > 0.0,
    )


def _binary_information_lower_bound(accuracy: float) -> float:
    error = min(0.5, max(0.0, 1.0 - float(accuracy)))
    if error <= 0.0:
        return 1.0
    if error >= 0.5:
        return 0.0
    entropy = -error * math.log2(error) - (1.0 - error) * math.log2(1.0 - error)
    return 1.0 - entropy


def simulate_ecology_assay(
    pair: dict[str, Any],
    configuration: ReaderConfiguration,
    candidate: dict[str, Any],
    scenario: str,
    separation: int,
    replicates: int,
    reference: dict[int, dict[str, np.ndarray]],
    writer_contract: MotifContract,
    contract: MinimalityContract,
) -> dict[str, Any]:
    if scenario not in ECOLOGY_SCENARIOS or separation not in ECOLOGY_SEPARATIONS:
        raise ValueError("unregistered ecology assay")
    pair_id = str(pair["pair_id"])
    extent = 16
    reference_probability = reference[configuration.write_window]["motif_probability"]
    primary, founder_terminal = _founder_bounded_payload(
        pair,
        candidate,
        reference_probability,
        writer_contract,
        replicates,
        extent,
        contract.rule,
    )
    secondary = primary.copy()
    if scenario == "opposite_history":
        secondary = _swap_histories(secondary, replicates)
    elif scenario == "intact_corrupted":
        mask_half = np.random.default_rng(
            _hash_seed(contract.namespace, pair_id, scenario)
        ).random((replicates, secondary.shape[-1])) < 0.25
        secondary[np.concatenate((mask_half, mask_half), axis=0)] *= -1.0
    elif scenario == "unequal_precision":
        secondary[:, 1::2] = 0.0
    elif scenario == "half_channel_chimera":
        swapped = _swap_histories(secondary, replicates)
        secondary[:, : secondary.shape[-1] // 2] = swapped[:, : secondary.shape[-1] // 2]
    secondary, _ = _quantize(secondary, candidate["codec_model"])

    origins_a = dynamic_origins(pair_id, 1, replicates, extent)
    origins_b = origins_a.copy()
    origins_b[:, 1] = (origins_b[:, 1] + separation) % extent
    field_a, occupied_a = embed_dynamic_seed(primary, origins_a, extent)
    field_b, occupied_b = embed_dynamic_seed(secondary, origins_b, extent)
    hops = int(candidate["germination_hops"])
    field_a, occupied_a, trace_a = propagate_bounded(field_a, occupied_a, hops)
    field_b, occupied_b, trace_b = propagate_bounded(field_b, occupied_b, hops)
    denominator = occupied_a.astype(np.int8) + occupied_b.astype(np.int8)
    field = np.divide(
        field_a * occupied_a[..., None] + field_b * occupied_b[..., None],
        denominator[..., None],
        out=np.zeros_like(field_a),
        where=denominator[..., None] > 0,
    )
    occupied = denominator > 0
    overlap = occupied_a & occupied_b

    distance_a = np.linalg.norm(field - primary[:, None, None], axis=-1)
    distance_b = np.linalg.norm(field - secondary[:, None, None], axis=-1)
    domain_a = occupied & (distance_a < distance_b)
    domain_b = occupied & (distance_b < distance_a)
    coexist = occupied & np.isclose(distance_a, distance_b, atol=1e-6)

    reset_state = _resize_board(
        _state_from_hex("life", pair["donor_a"]["initial_state_hex"]), extent
    )
    state = np.repeat(reset_state[None], 2 * replicates, axis=0)
    recent: deque[np.ndarray] = deque(maxlen=writer_contract.observation_window)
    site_sign_sum = np.zeros(
        (2 * replicates, extent, extent, int(candidate["rank"])), dtype=np.float32
    )
    basis = np.asarray(candidate["codec_model"]["basis"], dtype=np.float32)
    signs = basis.astype(np.float64) * math.sqrt(512.0)
    decoded_field = predecode_local_field(field, basis)
    for sweep in range(1, contract.generation_sweeps + 1):
        predicted = _step(state, contract.rule)
        if sweep <= contract.read_sweeps:
            predicted = apply_predecoded_local_reader(
                predicted,
                decoded_field,
                _uniforms(
                    pair_id,
                    str(candidate["candidate_id"]),
                    f"ecology-{scenario}-{separation}",
                    1,
                    sweep,
                    replicates,
                    extent,
                ),
                configuration.strength,
            )
        predicted ^= _uniforms(
            pair_id,
            str(candidate["candidate_id"]),
            f"ecology-process-{scenario}-{separation}",
            1,
            sweep,
            replicates,
            extent,
        ) < contract.process_noise
        state = predicted
        if contract.write_start <= sweep <= contract.write_end:
            site_sign_sum += signs[motif3_codes(state)].astype(np.float32)
        if sweep >= contract.observe_start:
            recent.append(live_2x2_counts_batch(state))
    alive = state.any(axis=(1, 2))
    outcome, _ = _score_state(
        state, recent, pair, founder_terminal, replicates, writer_contract
    )

    span = int(candidate["consolidation_span"])
    endpoint_a = bounded_reduce_endpoint(site_sign_sum, span, origins_a)
    endpoint_b = bounded_reduce_endpoint(site_sign_sum, span, origins_b)
    observations = float((contract.write_end - contract.write_start + 1) * (span + 1) ** 2)
    alpha = writer_contract.jeffreys_alpha
    moment_a = (endpoint_a + alpha * signs.sum(axis=0)) / (observations + 512.0 * alpha)
    moment_b = (endpoint_b + alpha * signs.sum(axis=0)) / (observations + 512.0 * alpha)
    weight = np.asarray(candidate["weight"], dtype=np.float64)
    bias = np.asarray(candidate["bias"], dtype=np.float64)
    offspring_a, _ = _quantize(moment_a @ weight + bias, candidate["codec_model"])
    offspring_b, _ = _quantize(moment_b @ weight + bias, candidate["codec_model"])
    return {
        "pair_id": pair_id,
        "candidate_id": candidate["candidate_id"],
        "scenario": scenario,
        "separation": separation,
        "replicates": replicates,
        "field": {
            "occupied_fraction": float(np.mean(occupied)),
            "overlap_fraction": float(np.mean(overlap)),
            "seed_a_domain_fraction": float(np.mean(domain_a)),
            "seed_b_domain_fraction": float(np.mean(domain_b)),
            "coexistence_fraction": float(np.mean(coexist)),
            "wave_trace_a": trace_a,
            "wave_trace_b": trace_b,
        },
        "descendant_outcome": outcome,
        "offspring": {
            "a_similarity_to_seed_a": float(np.mean(_cosine_batch(offspring_a, primary))),
            "a_similarity_to_seed_b": float(np.mean(_cosine_batch(offspring_a, secondary))),
            "b_similarity_to_seed_a": float(np.mean(_cosine_batch(offspring_b, primary))),
            "b_similarity_to_seed_b": float(np.mean(_cosine_batch(offspring_b, secondary))),
            "a_b_similarity": float(np.mean(_cosine_batch(offspring_a, offspring_b))),
        },
    }


def _ecology_task(
    payload: tuple[
        dict[str, Any],
        list[dict[str, Any]],
        MotifContract,
        MinimalityContract,
        dict[int, dict[str, np.ndarray]],
    ]
) -> dict[str, Any]:
    item, candidates, writer_contract, contract, reference = payload
    candidate = next(
        row for row in candidates if row["candidate_id"] == item["candidate_id"]
    )
    result = simulate_ecology_assay(
        item["pair"],
        ReaderConfiguration(**item["configuration"]),
        candidate,
        str(item["scenario"]),
        int(item["separation"]),
        int(item["replicates"]),
        reference,
        writer_contract,
        contract,
    )
    return {"checkpoint": item["checkpoint"], **result}


def summarize_ecology(
    rows: Sequence[dict[str, Any]],
    profile: MinimalityProfile,
    contract: MinimalityContract,
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    for scenario in ECOLOGY_SCENARIOS:
        summaries[scenario] = {}
        for separation in ECOLOGY_SEPARATIONS:
            selected = [
                row
                for row in rows
                if row["scenario"] == scenario and int(row["separation"]) == separation
            ]
            summaries[scenario][str(separation)] = {
                "pairs": len(selected),
                "descendant_crossover": _boot(
                    [row["descendant_outcome"]["primary"]["crossover"] for row in selected],
                    profile,
                    contract,
                    "ecology",
                    scenario,
                    separation,
                ),
                "overlap_fraction_mean": float(
                    np.mean([row["field"]["overlap_fraction"] for row in selected])
                ),
                "seed_a_domain_fraction_mean": float(
                    np.mean([row["field"]["seed_a_domain_fraction"] for row in selected])
                ),
                "seed_b_domain_fraction_mean": float(
                    np.mean([row["field"]["seed_b_domain_fraction"] for row in selected])
                ),
                "offspring_a_b_similarity_mean": float(
                    np.mean([row["offspring"]["a_b_similarity"] for row in selected])
                ),
            }
    return {
        "state": "complete",
        "candidate_id": rows[0]["candidate_id"] if rows else None,
        "single_generation_mechanistic_assay": True,
        "summaries": summaries,
    }


def _evolution_cache(
    frozen: dict[str, Any], ranks: Sequence[int]
) -> dict[int, dict[str, np.ndarray | float]]:
    fit_matrix, _, _ = load_stage3r_fit_matrix(frozen["stage4"]["stage3r"])
    values = np.asarray(fit_matrix, dtype=np.float64)
    if len(values) > 256:
        order = np.argsort(
            [
                hashlib.sha256(f"stage6-evolution-row:{index}".encode()).hexdigest()
                for index in range(len(values))
            ]
        )[:256]
        values = values[order]
    reference_probability = np.asarray(
        frozen["stage4"]["reference"][32]["motif_probability"], dtype=np.float64
    )
    probability = reference_probability[None] * np.exp(np.clip(values * 2.0, -8.0, 8.0))
    probability /= probability.sum(axis=1, keepdims=True)
    full_basis = np.asarray(frozen["stage4"]["winner_model"]["basis"], dtype=np.float64)
    result: dict[int, dict[str, np.ndarray | float]] = {}
    for rank in ranks:
        basis = full_basis[:, :rank]
        moments = probability @ (basis * math.sqrt(512.0))
        target = np.float64(0.50) * (values @ basis)
        design = np.concatenate((moments, np.ones((len(moments), 1))), axis=1)
        result[rank] = {
            "xtx": design.T @ design,
            "xty": design.T @ target,
            "yty": float(np.sum(target * target)),
            "target_total": float(
                np.sum((target - target.mean(axis=0, keepdims=True)) ** 2)
            ),
            "samples": float(len(target)),
        }
    return result


def _genome_score(
    genome: dict[str, Any],
    cache: dict[int, dict[str, np.ndarray | float]],
    treatment: str,
) -> tuple[float, float]:
    rank = int(genome["rank"])
    weight = np.asarray(genome["weight"], dtype=np.float64)[:rank, :rank]
    bias = np.asarray(genome["bias"], dtype=np.float64)[:rank]
    coefficients = np.concatenate((weight, bias[None]), axis=0)
    values = cache[rank]
    xtx = np.asarray(values["xtx"], dtype=np.float64)
    xty = np.asarray(values["xty"], dtype=np.float64)
    sse = (
        float(values["yty"])
        - 2.0 * float(np.sum(coefficients * xty))
        + float(np.trace(coefficients.T @ xtx @ coefficients))
    )
    r2 = 1.0 - sse / max(float(values["target_total"]), 1e-12)
    fidelity = max(0.0, min(1.0, r2))
    fidelity *= min(1.0, int(genome["bits"]) / 4.0)
    fidelity *= min(1.0, int(genome["hops"]) / 5.0)
    fidelity *= min(1.0, (int(genome["span"]) + 1) / 8.0)
    treatment_factor = {
        "autocorrelated": 1.0,
        "iid_histories": 0.05,
        "inheritance_disabled": 0.0,
        "payload_shuffled": 0.10,
        "rewrite_disabled": 0.05,
    }[treatment]
    memory_proxy = fidelity * treatment_factor
    nonzero = int(np.count_nonzero(weight))
    cost = (
        rank * int(genome["bits"]) / 64.0
        + nonzero / 256.0
        + (int(genome["hops"]) + 2 * int(genome["span"])) / 35.0
    )
    return memory_proxy - 0.04 * cost, memory_proxy


def _mutate_genome(
    genome: dict[str, Any], rng: np.random.Generator
) -> dict[str, Any]:
    result = {
        key: (np.asarray(value).copy() if key in ("weight", "bias") else value)
        for key, value in genome.items()
    }
    if rng.random() < 0.25:
        result["rank"] = int(rng.choice((2, 4, 8, 12, 16)))
    if rng.random() < 0.25:
        result["bits"] = int(rng.choice((2, 3, 4)))
    result["hops"] = int(np.clip(int(result["hops"]) + rng.integers(-1, 2), 1, 8))
    result["span"] = int(np.clip(int(result["span"]) + rng.integers(-2, 3), 0, 15))
    weight = np.asarray(result["weight"], dtype=np.float64)
    mutation = rng.random(weight.shape) < 0.08
    weight[mutation] += rng.normal(0.0, 0.04, size=np.count_nonzero(mutation))
    knockout = rng.random(weight.shape) < 0.015
    weight[knockout] = 0.0
    result["weight"] = np.round(weight * 64.0) / 64.0
    bias = np.asarray(result["bias"], dtype=np.float64)
    bias += rng.normal(0.0, 0.01, size=bias.shape)
    result["bias"] = np.round(bias * 64.0) / 64.0
    result["reader_strength"] = float(
        np.clip(float(result["reader_strength"]) + rng.normal(0.0, 0.02), 0.05, 0.50)
    )
    return result


def run_evolutionary_search(
    frozen: dict[str, Any],
    base_candidate: dict[str, Any],
    profile: MinimalityProfile,
    contract: MinimalityContract,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ranks = (2, 4, 8, 12, 16)
    cache = _evolution_cache(frozen, ranks)
    fitted, _ = build_compression_candidates(frozen, base_candidate)
    fitted_by_rank = {int(row["rank"]): row for row in fitted}
    population_summaries: list[dict[str, Any]] = []
    nominees: list[dict[str, Any]] = []
    for track in ("descent", "de_novo"):
        for treatment in EVOLUTION_TREATMENTS:
            for population_index in range(profile.evolution_populations):
                rng = np.random.default_rng(
                    _hash_seed(
                        contract.namespace,
                        "evolution",
                        track,
                        treatment,
                        population_index,
                    )
                )
                population: list[dict[str, Any]] = []
                for individual in range(profile.evolution_population_size):
                    if track == "descent":
                        source = fitted_by_rank[16]
                        genome = {
                            "rank": 16,
                            "bits": 4,
                            "hops": int(base_candidate["germination_hops"]),
                            "span": int(base_candidate["consolidation_span"]),
                            "weight": np.asarray(source["weight"], dtype=np.float64),
                            "bias": np.asarray(source["bias"], dtype=np.float64),
                            "reader_strength": 0.25,
                        }
                        for _ in range(1 + individual % 3):
                            genome = _mutate_genome(genome, rng)
                    else:
                        weight = rng.normal(0.0, 0.08, size=(16, 16))
                        weight[rng.random((16, 16)) < 0.80] = 0.0
                        genome = {
                            "rank": int(rng.choice(ranks)),
                            "bits": int(rng.choice((2, 3, 4))),
                            "hops": int(rng.integers(1, 9)),
                            "span": int(rng.integers(0, 16)),
                            "weight": np.round(weight * 64.0) / 64.0,
                            "bias": np.zeros(16, dtype=np.float64),
                            "reader_strength": float(rng.uniform(0.10, 0.40)),
                        }
                    population.append(genome)
                best_trace: list[float] = []
                for _generation in range(profile.evolution_generations):
                    scored = [
                        (*_genome_score(genome, cache, treatment), genome)
                        for genome in population
                    ]
                    scored.sort(key=lambda row: (-row[0], -row[1]))
                    best_trace.append(float(scored[0][1]))
                    parents = [row[2] for row in scored[: max(2, len(scored) // 4)]]
                    population = [
                        _mutate_genome(parents[index % len(parents)], rng)
                        for index in range(profile.evolution_population_size)
                    ]
                scored = [
                    (*_genome_score(genome, cache, treatment), genome)
                    for genome in population
                ]
                scored.sort(key=lambda row: (-row[0], -row[1]))
                objective, proxy, winner = scored[0]
                discovered = bool(proxy >= 0.15)
                population_summaries.append(
                    {
                        "track": track,
                        "treatment": treatment,
                        "population": population_index,
                        "best_objective": float(objective),
                        "best_memory_proxy": float(proxy),
                        "discovered_proxy_candidate": discovered,
                        "best_trace": best_trace,
                        "genome": {
                            "rank": int(winner["rank"]),
                            "bits": int(winner["bits"]),
                            "payload_bits": int(winner["rank"] * winner["bits"]),
                            "hops": int(winner["hops"]),
                            "span": int(winner["span"]),
                            "nonzero_weights": int(np.count_nonzero(winner["weight"])),
                            "reader_strength": float(winner["reader_strength"]),
                        },
                    }
                )
                if treatment == "autocorrelated" and discovered:
                    rank = int(winner["rank"])
                    codec = _codec_model(frozen["stage4"]["winner_model"], rank, int(winner["bits"]))
                    nominees.append(
                        {
                            **{
                                key: value
                                for key, value in base_candidate.items()
                                if key not in ("candidate_id", "codec_model", "weight", "bias")
                            },
                            "candidate_id": f"evolved-{track}-p{population_index:02d}-r{rank:02d}-q{int(winner['bits']):02d}",
                            "evolution_track": track,
                            "rank": rank,
                            "bits": int(winner["bits"]),
                            "payload_bits": rank * int(winner["bits"]),
                            "germination_hops": int(winner["hops"]),
                            "consolidation_span": int(winner["span"]),
                            "consolidation_steps": 2 * int(winner["span"]),
                            "bounded": bool(
                                int(winner["hops"]) <= contract.bounded_hops
                                and 2 * int(winner["span"])
                                <= contract.bounded_consolidation_steps
                            ),
                            "codec_model": codec,
                            "weight": np.asarray(winner["weight"], dtype=np.float32)[:rank, :rank],
                            "bias": np.asarray(winner["bias"], dtype=np.float32)[:rank],
                            "reader_strength": float(winner["reader_strength"]),
                            "developmental_field_values_16": 16 * 16 * rank,
                            "developmental_writer_values_16": 16 * 16 * rank,
                            "routing_buffer_values_16": 2 * 16 * 16 * rank,
                            "shared_writer_parameter_bits": (rank * rank + rank) * 32,
                        }
                    )
    discovery_counts = {
        track: {
            treatment: sum(
                bool(row["discovered_proxy_candidate"])
                for row in population_summaries
                if row["track"] == track and row["treatment"] == treatment
            )
            for treatment in EVOLUTION_TREATMENTS
        }
        for track in ("descent", "de_novo")
    }
    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    for candidate in nominees:
        key = (
            candidate["evolution_track"],
            candidate["rank"],
            candidate["bits"],
            candidate["germination_hops"],
            candidate["consolidation_span"],
            hashlib.sha256(np.asarray(candidate["weight"]).tobytes()).hexdigest(),
        )
        unique.setdefault(key, candidate)
    nominated = sorted(
        unique.values(),
        key=lambda row: (
            int(row["payload_bits"]),
            int(row["germination_hops"]) + int(row["consolidation_steps"]),
            row["candidate_id"],
        ),
    )[:4]
    return (
        {
            "state": "complete",
            "training_assay": "label-blind trace-reconstruction proxy; not itself a PH result",
            "tracks": ("descent", "de_novo"),
            "treatments": EVOLUTION_TREATMENTS,
            "populations_per_treatment": profile.evolution_populations,
            "population_size": profile.evolution_population_size,
            "generations": profile.evolution_generations,
            "discovery_counts": discovery_counts,
            "population_summaries": population_summaries,
            "nominated_candidate_ids": [row["candidate_id"] for row in nominated],
            "validation_required_for_any_accessibility_claim": True,
        },
        nominated,
    )


def _prepare_stage6(
    output: Path,
    profile_name: str,
    *,
    open_audit: bool,
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
    list[dict[str, Any]],
    dict[str, Any],
    MinimalityProfile,
    MinimalityContract,
    MotifContract,
    str,
]:
    contract = MinimalityContract()
    writer_contract = MotifContract()
    profile = MINIMALITY_PROFILES[profile_name]
    frozen = load_frozen_stage5r(
        stage5r_root,
        stage5_root,
        stage4_root,
        stage3r_root,
        stage3_root,
        stage2_root,
        stage1_root,
    )
    if tuple(walsh_mode_ids(frozen["stage4"]["winner_model"])) != REGISTERED_MODE_IDS:
        raise ValueError("the registered Walsh order changed")
    cohorts = select_minimality_cohorts(
        profile, frozen, profile_name=profile_name, open_audit=open_audit
    )
    locality_candidates, anchors = build_locality_candidates(frozen)
    audit_ids = (
        list(frozen["later_ids"])
        if profile_name == "reference"
        else [pair["pair_id"] for pair in cohorts["audit"]]
    )
    design_payload = {
        "experiment": "ca_motif_lineage_stage_6",
        "contract": contract.to_dict(),
        "writer_contract_digest": writer_contract.digest,
        "profile_name": profile_name,
        "profile": asdict(profile),
        "round_contract": ROUNDS,
        "rounds_separate_invocations": True,
        "automatic_successor_launch": False,
        "stage5r_design_digest": frozen["design_digest"],
        "configuration": frozen["stage4"]["configuration"].to_dict(),
        "registered_mode_ids": walsh_mode_ids(frozen["stage4"]["winner_model"]),
        "locality_factorial": {
            "hops": LOCALITY_HOPS,
            "consolidation_spans": CONSOLIDATION_SPANS,
            "candidate_ids": [row["candidate_id"] for row in locality_candidates],
            "compact_anchor_id": COMPACT_ANCHOR_ID,
            "diagnostic_anchor_ids": (GLOBAL_ANCHOR_ID, EXACT_ANCHOR_ID),
        },
        "compression_shapes": COMPRESSION_SHAPES,
        "ecology_scenarios": ECOLOGY_SCENARIOS,
        "ecology_separations": ECOLOGY_SEPARATIONS,
        "evolution_treatments": EVOLUTION_TREATMENTS,
        "pair_ids": {
            name: [pair["pair_id"] for pair in rows]
            for name, rows in cohorts.items()
            if name != "audit"
        }
        | {"audit": audit_ids},
        "final_audit_pair_ids_sha256": hashlib.sha256(
            "\n".join(audit_ids).encode()
        ).hexdigest(),
        "input_sha256": {
            "protocol": _sha256(PROTOCOL_PATH),
            **{
                f"stage5r_{key}": _sha256(path)
                for key, path in frozen["paths"].items()
            },
        },
        "implementation_sha256": {
            "motif_minimality.py": _sha256(Path(__file__)),
            "motif_regeneration.py": _sha256(
                Path(__file__).with_name("motif_regeneration.py")
            ),
            "motif_localization.py": _sha256(
                Path(__file__).with_name("motif_localization.py")
            ),
            "motif_compression.py": _sha256(
                Path(__file__).with_name("motif_compression.py")
            ),
        },
        "cleanroom_exclusion": "no Wagner or Fable implementation source is read, imported, hashed, or executed",
        "retuning_on_final_audit": False,
    }
    design_digest = hashlib.sha256(
        json.dumps(design_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    design = {**design_payload, "design_digest": design_digest}
    design_path = output / "DESIGN.json"
    if design_path.exists():
        existing = _load_json(design_path)
        if existing.get("design_digest") != design_digest:
            raise ValueError("Stage-6 design digest changed; refusing mixed resume")
    else:
        _atomic_json(design_path, design)
    cohort_path = output / "COHORTS.json"
    existing_cohort = _load_json(cohort_path) if cohort_path.exists() else {}
    cohort_payload = {
        "design_digest": design_digest,
        "development_pair_ids": design["pair_ids"],
        "final_audit_pair_ids_sha256": design["final_audit_pair_ids_sha256"],
        "final_audit_pair_count": len(audit_ids),
        "final_audit_trajectory_state": existing_cohort.get(
            "final_audit_trajectory_state", "untouched"
        ),
        "final_audit_trajectories_not_loaded": bool(
            profile_name == "reference" and not open_audit
        ),
    }
    _atomic_json(cohort_path, cohort_payload)
    mechanism = stage6_mechanism_audit(contract)
    mechanism["design_digest"] = design_digest
    _atomic_json(output / "MECHANISM_AUDIT.json", mechanism)
    if not mechanism["passed"]:
        raise AssertionError("Stage-6 primitive mechanism audit failed")
    locality_models = [*locality_candidates, anchors["compact"]]
    model_json = output / "LOCALITY_MODELS.json"
    if not model_json.exists():
        save_candidate_models(
            output, "LOCALITY_MODELS", locality_models, design_digest
        )
    else:
        stored = load_candidate_models(output, "LOCALITY_MODELS", design_digest)
        if [row["candidate_id"] for row in stored] != [
            row["candidate_id"] for row in locality_models
        ]:
            raise ValueError("stored Stage-6 locality model order changed")
    return (
        frozen,
        cohorts,
        locality_candidates,
        anchors,
        profile,
        contract,
        writer_contract,
        design_digest,
    )


def _require_prior_round(
    output: Path,
    round_name: str,
    design_digest: str,
    *,
    scientific_profile: bool,
    authorize_gate_override: bool,
) -> dict[str, Any]:
    path = output / round_name / "STAGE_DECISION.json"
    if not path.exists():
        raise FileNotFoundError(f"reviewed prior-round decision required: {path}")
    decision = _load_json(path)
    if decision.get("design_digest") != design_digest or decision.get("state") != "complete":
        raise ValueError(f"Stage-6 {round_name} is not complete under this design")
    if scientific_profile and not decision.get("stage_gate", False) and not authorize_gate_override:
        raise ValueError(
            f"Stage-6 {round_name} gate failed; explicit --authorize-gate-override is required"
        )
    return decision


def _write_round_outputs(
    round_root: Path,
    results: dict[str, Any],
    decision: dict[str, Any],
    report: str,
    lay: str,
) -> None:
    _atomic_json(round_root / "RESULTS.json", results)
    _atomic_json(round_root / "STAGE_DECISION.json", decision)
    _atomic_text(round_root / "REPORT.md", report)
    _atomic_text(round_root / "LAY_SUMMARY.md", lay)
    if results.get("state") == "complete":
        _atomic_text(round_root / "COMPLETE", "complete\n")


def _candidate_effect(
    rows: Sequence[dict[str, Any]], candidate_id: str, generation: int
) -> float:
    values = _condition_values(rows, candidate_id, "intact", generation)
    return float(np.mean(values)) if values else -1.0


def _status_writer(
    output: Path,
    round_root: Path,
    round_name: str,
    profile_name: str,
    started: float,
    hard_deadline: float,
    science_deadline: float,
):
    def status(state: str, phase: str, **extra: Any) -> None:
        now = time.time()
        payload = {
            "state": state,
            "stage": "6-minimality",
            "round": round_name,
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
        _atomic_json(round_root / "STATUS.json", payload)
        _atomic_json(output / "STATUS.json", payload)
        progress = (
            f" {extra['completed']}/{extra['total']}" if "completed" in extra else ""
        )
        print(f"[{state}] stage6-{round_name}:{phase}{progress}", flush=True)

    return status


def _run_locality_round(
    output: Path,
    round_root: Path,
    frozen: dict[str, Any],
    cohorts: dict[str, list[dict[str, Any]]],
    locality_candidates: list[dict[str, Any]],
    anchors: dict[str, Any],
    profile: MinimalityProfile,
    profile_name: str,
    contract: MinimalityContract,
    writer_contract: MotifContract,
    design_digest: str,
    *,
    workers: int,
    resume: bool,
    science_deadline: float,
    status,
) -> dict[str, Any]:
    all_candidates = [*locality_candidates, anchors["compact"]]
    configuration = _configuration_payload(frozen["stage4"]["configuration"])
    screen_items = [
        {
            "checkpoint": f"screen-p{pair_index:04d}-c{candidate_index:02d}",
            "pair": pair,
            "candidate_id": candidate["candidate_id"],
            "configuration": configuration,
            "replicates": profile.locality_screen_replicates,
            "generations": 8,
            "conditions": ("intact",),
        }
        for pair_index, pair in enumerate(cohorts["locality_screen"])
        for candidate_index, candidate in enumerate(all_candidates)
    ]
    status("running", "screen", completed=0, total=len(screen_items))
    screen_rows, complete = _run_json_checkpoints(
        round_root,
        "screen",
        screen_items,
        all_candidates,
        _lineage_task,
        writer_contract,
        contract,  # type: ignore[arg-type]
        frozen["stage4"]["reference"],
        design_digest,
        workers=workers,
        resume=resume,
        deadline=science_deadline,
        status=status,
    )
    if not complete:
        return {"state": "partial_budget_exhausted", "phase": "screen"}
    scientific = profile_name == "reference"
    screen = summarize_screen(
        screen_rows,
        all_candidates,
        COMPACT_ANCHOR_ID,
        profile,
        contract,
        scientific_gate=scientific,
    )
    screen["design_digest"] = design_digest
    _atomic_json(round_root / "SCREEN.json", screen)
    diagnostic_candidates = [anchors["global"], anchors["exact"]]
    diagnostic_items = [
        {
            "checkpoint": f"diagnostic-p{pair_index:04d}-c{candidate_index:02d}",
            "pair": pair,
            "candidate_id": candidate["candidate_id"],
            "configuration": configuration,
            "replicates": profile.locality_screen_replicates,
            "generations": 8,
            "walsh_model": frozen["stage4"]["winner_model"],
        }
        for pair_index, pair in enumerate(cohorts["locality_screen"])
        for candidate_index, candidate in enumerate(diagnostic_candidates)
    ]
    status(
        "running", "anchor_diagnostics", completed=0, total=len(diagnostic_items)
    )
    diagnostic_rows, complete = _run_json_checkpoints(
        round_root,
        "anchor_diagnostics",
        diagnostic_items,
        diagnostic_candidates,
        _stage5r_anchor_task,
        writer_contract,
        contract,  # type: ignore[arg-type]
        frozen["stage4"]["reference"],
        design_digest,
        workers=workers,
        resume=resume,
        deadline=science_deadline,
        status=status,
    )
    if not complete:
        return {"state": "partial_budget_exhausted", "phase": "anchor_diagnostics"}
    anchor_diagnostics = {
        candidate["candidate_id"]: {
            "generation8_crossover": _boot(
                _condition_values(
                    diagnostic_rows, candidate["candidate_id"], "intact", 8
                ),
                profile,
                contract,
                "diagnostic-anchor",
                candidate["candidate_id"],
            ),
            "historical_mechanism_only_not_eligible": True,
        }
        for candidate in diagnostic_candidates
    }
    _atomic_json(
        round_root / "ANCHOR_DIAGNOSTICS.json",
        {"design_digest": design_digest, "anchors": anchor_diagnostics},
    )
    selected_ids = list(screen["selected_candidate_ids"])
    by_id = {row["candidate_id"]: row for row in all_candidates}
    selected = [by_id[value] for value in selected_ids]

    qualification_items = [
        {
            "checkpoint": f"qualification-p{pair_index:04d}-c{candidate_index:02d}",
            "pair": pair,
            "candidate_id": candidate["candidate_id"],
            "configuration": configuration,
            "replicates": profile.locality_qualification_replicates,
            "generations": 32,
            "conditions": QUALIFICATION_CONDITIONS,
        }
        for pair_index, pair in enumerate(cohorts["locality_qualification"])
        for candidate_index, candidate in enumerate(selected)
    ]
    status(
        "running", "qualification", completed=0, total=len(qualification_items)
    )
    qualification_rows, complete = _run_json_checkpoints(
        round_root,
        "qualification",
        qualification_items,
        selected,
        _lineage_task,
        writer_contract,
        contract,  # type: ignore[arg-type]
        frozen["stage4"]["reference"],
        design_digest,
        workers=workers,
        resume=resume,
        deadline=science_deadline,
        status=status,
    )
    if not complete:
        return {"state": "partial_budget_exhausted", "phase": "qualification"}
    qualification = summarize_locality_qualification(
        qualification_rows,
        selected,
        profile,
        contract,
        anchor_id=COMPACT_ANCHOR_ID,
        scientific_gate=scientific,
    )
    qualification["design_digest"] = design_digest
    _atomic_json(round_root / "QUALIFICATION.json", qualification)
    bounded_selected = [row for row in selected if row["candidate_id"] != COMPACT_ANCHOR_ID]
    qualified_ids = set(qualification["qualified_candidate_ids"])
    qualified = [row for row in bounded_selected if row["candidate_id"] in qualified_ids]
    pool = qualified or bounded_selected
    winner = (
        max(
            pool,
            key=lambda row: (
                _candidate_effect(qualification_rows, row["candidate_id"], 32),
                -(int(row["germination_hops"]) + int(row["consolidation_steps"])),
            ),
        )
        if pool
        else None
    )
    if winner is None:
        raise AssertionError("Stage-6A did not retain a bounded diagnostic candidate")

    endurance_items = [
        {
            "checkpoint": f"endurance-p{pair_index:04d}",
            "pair": pair,
            "candidate_id": winner["candidate_id"],
            "configuration": configuration,
            "replicates": profile.locality_endurance_replicates,
            "generations": 64,
            "conditions": (
                "intact",
                "no_rewrite",
                "carrier_corruption_1",
                "communication_cut",
            ),
        }
        for pair_index, pair in enumerate(cohorts["locality_endurance"])
    ]
    status("running", "endurance", completed=0, total=len(endurance_items))
    endurance_rows, complete = _run_json_checkpoints(
        round_root,
        "endurance",
        endurance_items,
        [winner],
        _lineage_task,
        writer_contract,
        contract,  # type: ignore[arg-type]
        frozen["stage4"]["reference"],
        design_digest,
        workers=workers,
        resume=resume,
        deadline=science_deadline,
        status=status,
    )
    if not complete:
        return {"state": "partial_budget_exhausted", "phase": "endurance"}
    endurance = {
        "candidate_id": winner["candidate_id"],
        "generation32": _boot(
            _condition_values(endurance_rows, winner["candidate_id"], "intact", 32),
            profile,
            contract,
            "endurance",
            "generation32",
        ),
        "generation64": _boot(
            _condition_values(endurance_rows, winner["candidate_id"], "intact", 64),
            profile,
            contract,
            "endurance",
            "generation64",
        ),
        "no_rewrite_advantage_generation64": _boot(
            _paired_advantage(endurance_rows, winner["candidate_id"], "no_rewrite", 64),
            profile,
            contract,
            "endurance",
            "no-rewrite",
        ),
        "carrier_corruption_generation64": _boot(
            _condition_values(
                endurance_rows,
                winner["candidate_id"],
                "carrier_corruption_1",
                64,
            ),
            profile,
            contract,
            "endurance",
            "carrier-corruption",
        ),
        "communication_cut_advantage_generation64": _boot(
            _paired_advantage(
                endurance_rows, winner["candidate_id"], "communication_cut", 64
            ),
            profile,
            contract,
            "endurance",
            "communication-cut",
        ),
    }
    endurance["generation64_positive"] = bool(
        endurance["generation64"]["ci"][0] is not None
        and float(endurance["generation64"]["ci"][0]) > 0.0
    )
    _atomic_json(round_root / "ENDURANCE.json", {"design_digest": design_digest, **endurance})
    stage_gate = bool(
        qualification["candidate_summaries"][winner["candidate_id"]]["locality_gate"]
    )
    results = {
        "experiment": "ca_motif_lineage_stage_6a",
        "state": "complete",
        "profile": profile_name,
        "design_digest": design_digest,
        "stage5r_design_digest": frozen["design_digest"],
        "screen": screen,
        "anchor_diagnostics": anchor_diagnostics,
        "qualification": qualification,
        "endurance": endurance,
        "winner_candidate_id": winner["candidate_id"],
        "winner_candidate": _json_candidate(winner),
        "stage_gate": stage_gate,
        "scientific_gate_applied": scientific,
    }
    decision = {
        "experiment": "ca_motif_lineage_stage_6a",
        "state": "complete",
        "design_digest": design_digest,
        "stage_gate": stage_gate,
        "winner_candidate_id": winner["candidate_id"],
        "decision": (
            "bounded_candidate_ready_for_review"
            if stage_gate
            else "no_bounded_candidate_passed_registered_gate"
        ),
        "automatic_launch": False,
        "review_required": True,
    }
    report = (
        "# Stage 6A: bounded locality\n\n"
        f"State: complete. Registered gate: **{'PASS' if stage_gate else 'FAIL'}**.\n\n"
        f"The selected bounded diagnostic is `{winner['candidate_id']}` with "
        f"{winner['germination_hops']} germination hops and "
        f"{winner['consolidation_steps']} local writer-routing steps. "
        "All visible daughters were reset before development; only the finite payload crossed the boundary.\n\n"
        f"Generation-64 crossover mean: {endurance['generation64']['mean']}. "
        "This endurance value is supplementary; the registered gate is the strict generation-16 causal ladder plus positive generation 32 and the locality controls.\n"
    )
    lay = (
        "# Lay summary\n\n"
        f"Stage 6A {'found' if stage_gate else 'did not find'} a carrier that passed every registered test while "
        "information could travel only a few cells and the daughter could summarize only a nearby patch. "
        f"The best tested bounded design was `{winner['candidate_id']}`. "
        "The generation-64 run asks whether that memory continues to be actively rebuilt, rather than merely fading slowly.\n"
    )
    _write_round_outputs(round_root, results, decision, report, lay)
    return results


def _run_scale_round(
    output: Path,
    round_root: Path,
    frozen: dict[str, Any],
    cohorts: dict[str, list[dict[str, Any]]],
    locality_candidates: list[dict[str, Any]],
    anchors: dict[str, Any],
    locality_decision: dict[str, Any],
    profile: MinimalityProfile,
    profile_name: str,
    contract: MinimalityContract,
    writer_contract: MotifContract,
    design_digest: str,
    *,
    workers: int,
    resume: bool,
    science_deadline: float,
    status,
) -> dict[str, Any]:
    winner_id = str(locality_decision["winner_candidate_id"])
    bounded = next(row for row in locality_candidates if row["candidate_id"] == winner_id)
    variants = build_scale_variants(bounded, anchors["compact"])
    save_candidate_models(round_root, "SCALE_MODELS", variants, design_digest)
    configuration = _configuration_payload(frozen["stage4"]["configuration"])
    scale_items = [
        {
            "checkpoint": f"scale-p{pair_index:04d}-c{candidate_index:02d}",
            "pair": pair,
            "candidate_id": candidate["candidate_id"],
            "configuration": configuration,
            "replicates": profile.scale_replicates,
            "generations": 8,
            "conditions": ("intact",),
            "extent": candidate["extent"],
        }
        for pair_index, pair in enumerate(cohorts["scale"])
        for candidate_index, candidate in enumerate(variants)
    ]
    status("running", "scale", completed=0, total=len(scale_items))
    scale_rows, complete = _run_json_checkpoints(
        round_root,
        "scale",
        scale_items,
        variants,
        _lineage_task,
        writer_contract,
        contract,  # type: ignore[arg-type]
        frozen["stage4"]["reference"],
        design_digest,
        workers=workers,
        resume=resume,
        deadline=science_deadline,
        status=status,
    )
    if not complete:
        return {"state": "partial_budget_exhausted", "phase": "scale"}
    scale = summarize_scale(scale_rows, variants, profile, contract)
    scale["design_digest"] = design_digest
    _atomic_json(round_root / "SCALE.json", scale)

    rules = [RULE ^ (1 << bit) for bit in range(18)]
    rules.extend((RULE, 70366))
    rules = list(dict.fromkeys(rules))
    rule_pairs = cohorts["scale"][: min(16, len(cohorts["scale"]))]
    rule_items = [
        {
            "checkpoint": f"rule-p{pair_index:04d}-r{rule:06d}",
            "pair": pair,
            "candidate_id": bounded["candidate_id"],
            "configuration": configuration,
            "replicates": max(2, profile.scale_replicates // 2),
            "generations": 8,
            "conditions": ("intact",),
            "rule": rule,
        }
        for pair_index, pair in enumerate(rule_pairs)
        for rule in rules
    ]
    status("running", "rule_domain", completed=0, total=len(rule_items))
    rule_rows, complete = _run_json_checkpoints(
        round_root,
        "rule_domain",
        rule_items,
        [bounded],
        _lineage_task,
        writer_contract,
        contract,  # type: ignore[arg-type]
        frozen["stage4"]["reference"],
        design_digest,
        workers=workers,
        resume=resume,
        deadline=science_deadline,
        status=status,
    )
    if not complete:
        return {"state": "partial_budget_exhausted", "phase": "rule_domain"}
    rule_domain = summarize_rule_domain(
        rule_rows, bounded["candidate_id"], profile, contract
    )
    rule_domain["design_digest"] = design_digest
    _atomic_json(round_root / "RULE_DOMAIN.json", rule_domain)
    scientific = profile_name == "reference"
    stage_gate = bool(scale["scale_gate"]) if scientific else False
    results = {
        "experiment": "ca_motif_lineage_stage_6b",
        "state": "complete",
        "profile": profile_name,
        "design_digest": design_digest,
        "source_candidate_id": bounded["candidate_id"],
        "scale": scale,
        "rule_domain": rule_domain,
        "stage_gate": stage_gate,
        "scientific_gate_applied": scientific,
    }
    decision = {
        "experiment": "ca_motif_lineage_stage_6b",
        "state": "complete",
        "design_digest": design_digest,
        "stage_gate": stage_gate,
        "source_candidate_id": bounded["candidate_id"],
        "decision": (
            "scale_geometry_gate_passed"
            if stage_gate
            else "scale_geometry_gate_not_passed"
        ),
        "automatic_launch": False,
        "review_required": True,
    }
    report = (
        "# Stage 6B: scale and causal geometry\n\n"
        f"Registered scale gate: **{'PASS' if stage_gate else 'FAIL'}**. "
        f"Exact fixed-budget light-cone audit: {scale['fixed_budget_exact_light_cone_pass']}. "
        f"Diameter-scaled retention audit: {scale['diameter_scaled_retention_pass']}.\n\n"
        "The rule-domain panel is exploratory and never refits or promotes a carrier.\n"
    )
    lay = (
        "# Lay summary\n\n"
        "This round enlarged the world while either keeping the memory's travel budget fixed or letting it grow with the world's diameter. "
        f"The combined registered scale test {'passed' if stage_gate else 'did not pass'}. "
        "The fixed-budget runs also check that influence is literally absent beyond the allowed light cone, rather than merely becoming weak.\n"
    )
    _write_round_outputs(round_root, results, decision, report, lay)
    return results


def _run_compression_round(
    output: Path,
    round_root: Path,
    frozen: dict[str, Any],
    cohorts: dict[str, list[dict[str, Any]]],
    locality_candidates: list[dict[str, Any]],
    locality_decision: dict[str, Any],
    profile: MinimalityProfile,
    profile_name: str,
    contract: MinimalityContract,
    writer_contract: MotifContract,
    design_digest: str,
    *,
    workers: int,
    resume: bool,
    science_deadline: float,
    status,
) -> dict[str, Any]:
    bounded = next(
        row
        for row in locality_candidates
        if row["candidate_id"] == locality_decision["winner_candidate_id"]
    )
    compressed, fit_audits = build_compression_candidates(frozen, bounded)
    anchor = {**bounded, "candidate_id": "compression-full-64-anchor"}
    all_candidates = [anchor, *compressed]
    save_candidate_models(
        round_root, "COMPRESSION_MODELS", all_candidates, design_digest
    )
    _atomic_json(
        round_root / "WRITER_FIT_AUDIT.json",
        {"design_digest": design_digest, "fits": fit_audits},
    )
    configuration = _configuration_payload(frozen["stage4"]["configuration"])
    screen_items = [
        {
            "checkpoint": f"screen-p{pair_index:04d}-c{candidate_index:02d}",
            "pair": pair,
            "candidate_id": candidate["candidate_id"],
            "configuration": configuration,
            "replicates": profile.compression_screen_replicates,
            "generations": 8,
            "conditions": ("intact",),
        }
        for pair_index, pair in enumerate(cohorts["compression_screen"])
        for candidate_index, candidate in enumerate(all_candidates)
    ]
    status("running", "screen", completed=0, total=len(screen_items))
    screen_rows, complete = _run_json_checkpoints(
        round_root,
        "screen",
        screen_items,
        all_candidates,
        _lineage_task,
        writer_contract,
        contract,  # type: ignore[arg-type]
        frozen["stage4"]["reference"],
        design_digest,
        workers=workers,
        resume=resume,
        deadline=science_deadline,
        status=status,
    )
    if not complete:
        return {"state": "partial_budget_exhausted", "phase": "screen"}
    scientific = profile_name == "reference"
    screen = summarize_screen(
        screen_rows,
        all_candidates,
        anchor["candidate_id"],
        profile,
        contract,
        scientific_gate=scientific,
    )
    screen["design_digest"] = design_digest
    _atomic_json(round_root / "SCREEN.json", screen)
    by_id = {row["candidate_id"]: row for row in all_candidates}
    selected = [by_id[value] for value in screen["selected_candidate_ids"]]
    environments = {
        "ordinary": {},
        "moderate_joint": {
            "erasure": 0.10,
            "sign_corruption": 0.05,
            "process_noise": 0.004,
        },
    }
    qualification_items = [
        {
            "checkpoint": f"qualification-p{pair_index:04d}-c{candidate_index:02d}-e{environment}",
            "pair": pair,
            "candidate_id": candidate["candidate_id"],
            "configuration": configuration,
            "replicates": profile.compression_qualification_replicates,
            "generations": 32,
            "conditions": QUALIFICATION_CONDITIONS,
            "stress_id": environment,
            "stress": stress,
        }
        for pair_index, pair in enumerate(cohorts["compression_qualification"])
        for candidate_index, candidate in enumerate(selected)
        for environment, stress in environments.items()
    ]
    status(
        "running", "qualification", completed=0, total=len(qualification_items)
    )
    qualification_rows, complete = _run_json_checkpoints(
        round_root,
        "qualification",
        qualification_items,
        selected,
        _lineage_task,
        writer_contract,
        contract,  # type: ignore[arg-type]
        frozen["stage4"]["reference"],
        design_digest,
        workers=workers,
        resume=resume,
        deadline=science_deadline,
        status=status,
    )
    if not complete:
        return {"state": "partial_budget_exhausted", "phase": "qualification"}
    qualification = summarize_compression_qualification(
        qualification_rows,
        selected,
        anchor["candidate_id"],
        profile,
        contract,
        scientific_gate=scientific,
    )
    qualification["design_digest"] = design_digest
    _atomic_json(round_root / "QUALIFICATION.json", qualification)
    stress_grid: dict[str, dict[str, float]] = {
        "ordinary": {},
        "erasure_005": {"erasure": 0.05},
        "erasure_010": {"erasure": 0.10},
        "erasure_020": {"erasure": 0.20},
        "sign_002": {"sign_corruption": 0.02},
        "sign_005": {"sign_corruption": 0.05},
        "sign_010": {"sign_corruption": 0.10},
        "writer_gain_075": {"writer_gain": 0.75},
        "writer_gain_125": {"writer_gain": 1.25},
    }
    stress_pairs = cohorts["compression_screen"][: min(32, len(cohorts["compression_screen"]))]
    stress_items = [
        {
            "checkpoint": f"stress-p{pair_index:04d}-c{candidate_index:02d}-e{stress_id}",
            "pair": pair,
            "candidate_id": candidate["candidate_id"],
            "configuration": configuration,
            "replicates": max(2, profile.compression_screen_replicates // 2),
            "generations": 8,
            "conditions": ("intact",),
            "stress_id": stress_id,
            "stress": stress,
        }
        for pair_index, pair in enumerate(stress_pairs)
        for candidate_index, candidate in enumerate(selected)
        for stress_id, stress in stress_grid.items()
    ]
    status("running", "stress_curves", completed=0, total=len(stress_items))
    stress_rows, complete = _run_json_checkpoints(
        round_root,
        "stress_curves",
        stress_items,
        selected,
        _lineage_task,
        writer_contract,
        contract,  # type: ignore[arg-type]
        frozen["stage4"]["reference"],
        design_digest,
        workers=workers,
        resume=resume,
        deadline=science_deadline,
        status=status,
    )
    if not complete:
        return {"state": "partial_budget_exhausted", "phase": "stress_curves"}
    stress_curves = {
        candidate["candidate_id"]: {
            stress_id: {
                "crossover": _boot(
                    _condition_values(
                        [row for row in stress_rows if row["stress_id"] == stress_id],
                        candidate["candidate_id"],
                        "intact",
                        8,
                    ),
                    profile,
                    contract,
                    "compression-stress",
                    candidate["candidate_id"],
                    stress_id,
                ),
                "carrier_balanced_accuracy_mean": float(
                    np.mean(
                        _decoder_values(
                            [row for row in stress_rows if row["stress_id"] == stress_id],
                            candidate["candidate_id"],
                            "intact",
                            8,
                            "carrier_balanced_accuracy",
                        )
                    )
                ),
                "carrier_information_lower_bound_bits_mean": float(
                    np.mean(
                        _decoder_values(
                            [row for row in stress_rows if row["stress_id"] == stress_id],
                            candidate["candidate_id"],
                            "intact",
                            8,
                            "carrier_information_lower_bound_bits",
                        )
                    )
                ),
            }
            for stress_id in stress_grid
        }
        for candidate in selected
    }
    pareto = compression_pareto_frontier(all_candidates, screen)
    _atomic_json(
        round_root / "STRESS_CURVES.json",
        {
            "design_digest": design_digest,
            "stress_grid": stress_grid,
            "curves": stress_curves,
            "information_metric": "binary symmetric-channel lower bound from held-out balanced accuracy",
        },
    )
    _atomic_json(
        round_root / "PARETO_FRONTIER.json",
        {"design_digest": design_digest, "frontier": pareto},
    )
    smallest = qualification["smallest_qualified_candidate_id"]
    diagnostic = smallest
    if diagnostic is None:
        nonanchor = [row for row in selected if row["candidate_id"] != anchor["candidate_id"]]
        diagnostic = min(
            nonanchor,
            key=lambda row: (int(row["payload_bits"]), row["candidate_id"]),
        )["candidate_id"] if nonanchor else None
    stage_gate = bool(smallest) if scientific else False
    results = {
        "experiment": "ca_motif_lineage_stage_6c",
        "state": "complete",
        "profile": profile_name,
        "design_digest": design_digest,
        "source_candidate_id": bounded["candidate_id"],
        "screen": screen,
        "qualification": qualification,
        "stress_curves": stress_curves,
        "pareto_frontier": pareto,
        "stage_gate": stage_gate,
        "scientific_gate_applied": scientific,
    }
    decision = {
        "experiment": "ca_motif_lineage_stage_6c",
        "state": "complete",
        "design_digest": design_digest,
        "stage_gate": stage_gate,
        "smallest_qualified_candidate_id": smallest,
        "diagnostic_candidate_id": diagnostic,
        "source_candidate_id": bounded["candidate_id"],
        "decision": (
            "bounded_compressed_candidate_ready_for_review"
            if stage_gate
            else "no_at_most_32bit_candidate_passed_registered_gate"
        ),
        "automatic_launch": False,
        "review_required": True,
    }
    report = (
        "# Stage 6C: carrier and writer compression\n\n"
        f"Registered compression gate: **{'PASS' if stage_gate else 'FAIL'}**. "
        f"Smallest qualified candidate: `{smallest}`.\n\n"
        "Qualification required the ordinary and moderate-damage causal ladders, bounded locality controls, no more than 32 inherited bits, no more than eight moments, and 70% retention of the full 64-bit bounded anchor.\n"
    )
    lay = (
        "# Lay summary\n\n"
        f"This round {'did' if stage_gate else 'did not'} compress the inherited guide to 32 bits or less while keeping the full causal signature under normal and damaged conditions. "
        f"The smallest passing carrier was `{smallest}`. A bit count alone is not enough: zeroing, scrambling, blocking reading, or blocking rewriting still had to destroy the history-specific recovery.\n"
    )
    _write_round_outputs(round_root, results, decision, report, lay)
    return results


def _run_ecology_round(
    output: Path,
    round_root: Path,
    frozen: dict[str, Any],
    cohorts: dict[str, list[dict[str, Any]]],
    locality_candidates: list[dict[str, Any]],
    locality_decision: dict[str, Any],
    compression_decision: dict[str, Any],
    profile: MinimalityProfile,
    profile_name: str,
    contract: MinimalityContract,
    writer_contract: MotifContract,
    design_digest: str,
    *,
    workers: int,
    resume: bool,
    science_deadline: float,
    status,
) -> dict[str, Any]:
    bounded = next(
        row
        for row in locality_candidates
        if row["candidate_id"] == locality_decision["winner_candidate_id"]
    )
    compression_models = load_candidate_models(
        output / "compression", "COMPRESSION_MODELS", design_digest
    )
    ecology_id = (
        compression_decision.get("smallest_qualified_candidate_id")
        or compression_decision.get("diagnostic_candidate_id")
    )
    ecology_candidate = next(
        row for row in compression_models if row["candidate_id"] == ecology_id
    )
    configuration = _configuration_payload(frozen["stage4"]["configuration"])
    ecology_items = [
        {
            "checkpoint": f"ecology-p{pair_index:04d}-s{scenario}-d{separation:02d}",
            "pair": pair,
            "candidate_id": ecology_candidate["candidate_id"],
            "configuration": configuration,
            "replicates": profile.ecology_replicates,
            "scenario": scenario,
            "separation": separation,
        }
        for pair_index, pair in enumerate(cohorts["ecology"])
        for scenario in ECOLOGY_SCENARIOS
        for separation in ECOLOGY_SEPARATIONS
    ]
    status("running", "ecology", completed=0, total=len(ecology_items))
    ecology_rows, complete = _run_json_checkpoints(
        round_root,
        "ecology",
        ecology_items,
        [ecology_candidate],
        _ecology_task,
        writer_contract,
        contract,  # type: ignore[arg-type]
        frozen["stage4"]["reference"],
        design_digest,
        workers=workers,
        resume=resume,
        deadline=science_deadline,
        status=status,
    )
    if not complete:
        return {"state": "partial_budget_exhausted", "phase": "ecology"}
    ecology = summarize_ecology(ecology_rows, profile, contract)
    ecology["design_digest"] = design_digest
    _atomic_json(round_root / "ECOLOGY.json", ecology)

    status("running", "evolutionary_search")
    evolution, nominees = run_evolutionary_search(
        frozen, bounded, profile, contract
    )
    evolution["design_digest"] = design_digest
    _atomic_json(round_root / "EVOLUTION.json", evolution)
    if nominees:
        save_candidate_models(
            round_root, "EVOLVED_MODELS", nominees, design_digest
        )
    else:
        _atomic_json(
            round_root / "EVOLVED_MODELS.json",
            {
                "design_digest": design_digest,
                "candidate_count": 0,
                "candidate_ids": [],
                "candidates": [],
                "array_sha256": None,
            },
        )

    validation_rows: list[dict[str, Any]] = []
    if nominees:
        validation_items = [
            {
                "checkpoint": f"validation-p{pair_index:04d}-c{candidate_index:02d}",
                "pair": pair,
                "candidate_id": candidate["candidate_id"],
                "configuration": {
                    **configuration,
                    "strength": float(candidate.get("reader_strength", configuration["strength"])),
                },
                "replicates": profile.ecology_replicates,
                "generations": 16,
                "conditions": QUALIFICATION_CONDITIONS,
                "stress_id": "ordinary",
            }
            for pair_index, pair in enumerate(cohorts["evolution_validation"])
            for candidate_index, candidate in enumerate(nominees)
        ]
        status(
            "running", "evolution_validation", completed=0, total=len(validation_items)
        )
        validation_rows, complete = _run_json_checkpoints(
            round_root,
            "evolution_validation",
            validation_items,
            nominees,
            _lineage_task,
            writer_contract,
            contract,  # type: ignore[arg-type]
            frozen["stage4"]["reference"],
            design_digest,
            workers=workers,
            resume=resume,
            deadline=science_deadline,
            status=status,
        )
        if not complete:
            return {
                "state": "partial_budget_exhausted",
                "phase": "evolution_validation",
            }
    scientific = profile_name == "reference"
    validation: dict[str, Any] = {}
    validated_ids: list[str] = []
    for candidate in nominees:
        gate = _environment_gate(
            validation_rows,
            candidate,
            profile,
            contract,
            "ordinary",
            profile.evolution_validation_pairs,
            profile.ecology_replicates,
            16,
            contract.strict_alpha,
            scientific,
        )
        architecture_pass = bool(
            candidate.get("bounded") and int(candidate["payload_bits"]) <= 32
        )
        passed = bool(gate["gate"] and architecture_pass)
        validation[candidate["candidate_id"]] = {
            "candidate": _json_candidate(candidate),
            "causal_validation": gate,
            "bounded_at_most_32bit": architecture_pass,
            "validated": passed,
        }
        if passed or not scientific:
            validated_ids.append(candidate["candidate_id"])
    de_novo_auto = int(evolution["discovery_counts"]["de_novo"]["autocorrelated"])
    control_max = max(
        int(evolution["discovery_counts"]["de_novo"][treatment])
        for treatment in EVOLUTION_TREATMENTS
        if treatment != "autocorrelated"
    )
    validated_de_novo = [
        value
        for value in validated_ids
        if validation[value]["candidate"].get("evolution_track") == "de_novo"
    ]
    accessibility = bool(
        scientific
        and de_novo_auto >= math.ceil(profile.evolution_populations / 2)
        and control_max < math.ceil(profile.evolution_populations / 2)
        and validated_de_novo
    )
    validation_summary = {
        "design_digest": design_digest,
        "scientific_gate_applied": scientific,
        "candidates": validation,
        "validated_candidate_ids": validated_ids,
        "validated_de_novo_candidate_ids": validated_de_novo,
        "de_novo_autocorrelated_proxy_discoveries": de_novo_auto,
        "de_novo_max_control_proxy_discoveries": control_max,
        "evolutionary_accessibility_gate": accessibility,
        "proxy_discovery_never_counted_without_full_ca_validation": True,
    }
    _atomic_json(round_root / "EVOLUTION_VALIDATION.json", validation_summary)
    evolved_finalist = (
        min(
            validated_ids,
            key=lambda value: (
                int(validation[value]["candidate"]["payload_bits"]), value
            ),
        )
        if validated_ids
        else None
    )
    results = {
        "experiment": "ca_motif_lineage_stage_6d",
        "state": "complete",
        "profile": profile_name,
        "design_digest": design_digest,
        "ecology": ecology,
        "evolution": evolution,
        "evolution_validation": validation_summary,
        "evolutionary_accessibility_gate": accessibility,
        "evolved_finalist_candidate_id": evolved_finalist,
        "stage_gate": True,
        "scientific_gate_applied": scientific,
    }
    decision = {
        "experiment": "ca_motif_lineage_stage_6d",
        "state": "complete",
        "design_digest": design_digest,
        "stage_gate": True,
        "evolutionary_accessibility_gate": accessibility,
        "evolved_finalist_candidate_id": evolved_finalist,
        "decision": "ecology_and_evolution_complete_ready_for_review",
        "automatic_launch": False,
        "review_required": True,
    }
    report = (
        "# Stage 6D: ecology and evolutionary accessibility\n\n"
        f"Ecology assays completed for `{ecology_candidate['candidate_id']}`. "
        f"The registered evolutionary-accessibility gate: **{'PASS' if accessibility else 'FAIL'}**. "
        f"Validated evolved finalist: `{evolved_finalist}`.\n\n"
        "Evolutionary training used a label-blind reconstruction proxy only. No proxy winner was counted as PH unless it subsequently passed the full CA causal validation on disjoint exposed pairs.\n"
    )
    lay = (
        "# Lay summary\n\n"
        "The ecology test put two inherited memory seeds into the same world and measured where they blended, competed, or produced mixed offspring. "
        f"The separate evolution test {'did' if accessibility else 'did not'} show that random sparse mechanisms could repeatedly evolve into a fully validated hereditary carrier under correlated histories while the control environments could not. "
        "A failed evolution result would not erase the engineered PH result; it would mean we have not shown that such a mechanism is easy to evolve.\n"
    )
    _write_round_outputs(round_root, results, decision, report, lay)
    return results


def _run_audit_round(
    output: Path,
    round_root: Path,
    frozen: dict[str, Any],
    cohorts: dict[str, list[dict[str, Any]]],
    locality_candidates: list[dict[str, Any]],
    anchors: dict[str, Any],
    locality_decision: dict[str, Any],
    scale_decision: dict[str, Any],
    compression_decision: dict[str, Any],
    ecology_decision: dict[str, Any],
    profile: MinimalityProfile,
    profile_name: str,
    contract: MinimalityContract,
    writer_contract: MotifContract,
    design_digest: str,
    *,
    workers: int,
    resume: bool,
    science_deadline: float,
    status,
) -> dict[str, Any]:
    bounded = next(
        row
        for row in locality_candidates
        if row["candidate_id"] == locality_decision["winner_candidate_id"]
    )
    candidates: list[dict[str, Any]] = [anchors["compact"], bounded]
    class_ids: dict[str, str | None] = {
        "anchor": anchors["compact"]["candidate_id"],
        "bounded": bounded["candidate_id"],
        "compressed": None,
        "evolved": None,
    }
    compressed_id = compression_decision.get("smallest_qualified_candidate_id")
    if compressed_id:
        compression_models = load_candidate_models(
            output / "compression", "COMPRESSION_MODELS", design_digest
        )
        compressed = next(
            row for row in compression_models if row["candidate_id"] == compressed_id
        )
        if compressed["candidate_id"] not in {row["candidate_id"] for row in candidates}:
            candidates.append(compressed)
        class_ids["compressed"] = compressed["candidate_id"]
    evolved_id = ecology_decision.get("evolved_finalist_candidate_id")
    if evolved_id:
        evolved_models = load_candidate_models(
            output / "ecology", "EVOLVED_MODELS", design_digest
        )
        evolved = next(row for row in evolved_models if row["candidate_id"] == evolved_id)
        if evolved["candidate_id"] not in {row["candidate_id"] for row in candidates}:
            candidates.append(evolved)
        class_ids["evolved"] = evolved["candidate_id"]
    if len(candidates) > 4:
        raise AssertionError("Stage-6E may audit at most four frozen candidates")
    if (round_root / "FINAL_MODELS.json").exists():
        stored_final = load_candidate_models(
            round_root, "FINAL_MODELS", design_digest
        )
        if [row["candidate_id"] for row in stored_final] != [
            row["candidate_id"] for row in candidates
        ]:
            raise ValueError("sealed final model order changed")
    else:
        save_candidate_models(round_root, "FINAL_MODELS", candidates, design_digest)
    final_design_payload = {
        "experiment": "ca_motif_lineage_stage_6e_final_audit",
        "design_digest": design_digest,
        "candidate_ids": [row["candidate_id"] for row in candidates],
        "candidate_metadata": {
            row["candidate_id"]: _json_candidate(row) for row in candidates
        },
        "candidate_classes": class_ids,
        "model_archive_sha256": _sha256(round_root / "FINAL_MODELS.npz"),
        "pair_count": len(cohorts["audit"]),
        "pair_ids_sha256": hashlib.sha256(
            "\n".join(pair["pair_id"] for pair in cohorts["audit"]).encode()
        ).hexdigest(),
        "replicates": profile.audit_replicates,
        "generations": 16,
        "environments": ("ordinary", "moderate_joint"),
        "conditions": QUALIFICATION_CONDITIONS,
        "alpha_per_object": contract.final_alpha_per_object,
        "models_and_thresholds_frozen": True,
        "retuning_permitted": False,
    }
    final_digest = hashlib.sha256(
        json.dumps(final_design_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    final_design = {**final_design_payload, "final_audit_digest": final_digest}
    final_path = round_root / "FINAL_AUDIT_DESIGN.json"
    if final_path.exists():
        if _load_json(final_path).get("final_audit_digest") != final_digest:
            raise ValueError("final audit design changed after sealing")
    else:
        _atomic_json(final_path, final_design)
    cohort_payload = _load_json(output / "COHORTS.json")
    cohort_payload["final_audit_trajectory_state"] = "opened"
    cohort_payload["final_audit_opened_unix"] = time.time()
    _atomic_json(output / "COHORTS.json", cohort_payload)

    configuration = _configuration_payload(frozen["stage4"]["configuration"])
    environments = {
        "ordinary": {},
        "moderate_joint": {
            "erasure": 0.10,
            "sign_corruption": 0.05,
            "process_noise": 0.004,
        },
    }
    items = [
        {
            "checkpoint": f"audit-p{pair_index:04d}-c{candidate_index:02d}-e{environment}",
            "pair": pair,
            "candidate_id": candidate["candidate_id"],
            "configuration": {
                **configuration,
                "strength": float(candidate.get("reader_strength", configuration["strength"])),
            },
            "replicates": profile.audit_replicates,
            "generations": 16,
            "conditions": QUALIFICATION_CONDITIONS,
            "stress_id": environment,
            "stress": stress,
        }
        for pair_index, pair in enumerate(cohorts["audit"])
        for candidate_index, candidate in enumerate(candidates)
        for environment, stress in environments.items()
    ]
    status("running", "final_audit", completed=0, total=len(items))
    rows, complete = _run_json_checkpoints(
        round_root,
        "final_audit",
        items,
        candidates,
        _lineage_task,
        writer_contract,
        contract,  # type: ignore[arg-type]
        frozen["stage4"]["reference"],
        design_digest,
        workers=workers,
        resume=resume,
        deadline=science_deadline,
        status=status,
    )
    if not complete:
        return {"state": "partial_budget_exhausted", "phase": "final_audit"}
    summaries: dict[str, Any] = {}
    passes: dict[str, bool] = {}
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        environment_results = {
            environment: _environment_gate(
                rows,
                candidate,
                profile,
                contract,
                environment,
                profile.audit_pairs,
                profile.audit_replicates,
                16,
                contract.final_alpha_per_object,
                True,
            )
            for environment in environments
        }
        passed = all(value["gate"] for value in environment_results.values())
        summaries[candidate_id] = {
            "candidate": _json_candidate(candidate),
            "environments": environment_results,
            "robust_final_gate": passed,
        }
        passes[candidate_id] = passed
    anchor_pass = passes.get(str(class_ids["anchor"]), False)
    bounded_pass = passes.get(str(class_ids["bounded"]), False)
    compressed_pass = bool(
        class_ids["compressed"] and passes.get(str(class_ids["compressed"]), False)
    )
    evolved_pass = bool(
        class_ids["evolved"] and passes.get(str(class_ids["evolved"]), False)
    )
    scale_pass = bool(scale_decision.get("stage_gate"))
    if not anchor_pass:
        verdict = "FAILURE_TO_REPLICATE_STAGE5R_ANCHOR"
    elif evolved_pass and compressed_pass and bounded_pass and scale_pass:
        verdict = "ROBUST_EVOLVABLE_BOUNDED_COMPRESSED_CA_PLASTIC_HEREDITY"
    elif compressed_pass and bounded_pass and scale_pass:
        verdict = "ROBUST_BOUNDED_COMPRESSED_CA_PLASTIC_HEREDITY"
    elif bounded_pass and scale_pass:
        verdict = "ROBUST_BOUNDED_LOCAL_CA_PLASTIC_HEREDITY"
    elif bounded_pass:
        verdict = "SCALE_DEPENDENT_REGENERATIVE_BROADCAST"
    else:
        verdict = "STAGE5R_ONLY_REPLICATION"
    adjudication = {
        "state": "complete",
        "verdict": verdict,
        "candidate_classes": class_ids,
        "anchor_pass": anchor_pass,
        "bounded_pass": bounded_pass,
        "compressed_pass": compressed_pass,
        "evolved_pass": evolved_pass,
        "scale_gate_from_stage6b": scale_pass,
        "candidates": summaries,
        "claim_boundary": "engineered synthetic CA lineage memory; not evidence of metabolism, agency, biological life, or human-memory mechanism",
    }
    cohort_payload["final_audit_trajectory_state"] = "complete"
    cohort_payload["final_audit_completed_unix"] = time.time()
    _atomic_json(output / "COHORTS.json", cohort_payload)
    results = {
        "experiment": "ca_motif_lineage_stage_6e",
        "state": "complete",
        "profile": profile_name,
        "design_digest": design_digest,
        "final_audit_digest": final_digest,
        "adjudication": adjudication,
        "stage_gate": anchor_pass,
    }
    decision = {
        "experiment": "ca_motif_lineage_stage_6e",
        "state": "complete",
        "design_digest": design_digest,
        "final_audit_digest": final_digest,
        "stage_gate": anchor_pass,
        "verdict": verdict,
        "decision": "stage6_program_complete",
        "automatic_launch": False,
        "review_required": True,
    }
    report = (
        "# Stage 6E: sealed final audit\n\n"
        f"Final registered verdict: **{verdict}**.\n\n"
        f"Anchor pass: {anchor_pass}; bounded pass: {bounded_pass}; compressed pass: {compressed_pass}; evolved pass: {evolved_pass}; prior scale gate: {scale_pass}.\n\n"
        "All 62 reserved pairs were opened together after the candidate list, models, conditions, thresholds, and candidate order were sealed. No final-audit retuning was permitted.\n"
    )
    lay = (
        "# Lay summary\n\n"
        f"The untouched final test produced the verdict `{verdict}`. "
        "The hierarchy distinguishes merely reproducing the earlier full-communication result from surviving strict locality, compression, scaling, and evolutionary-accessibility tests. "
        "Even the strongest verdict concerns an engineered cellular-automaton memory mechanism, not life or human cognition by itself.\n"
    )
    _write_round_outputs(round_root, results, decision, report, lay)
    return results


def run_motif_minimality(
    output: Path,
    *,
    round_name: str = "locality",
    stage5r_root: Path = DEFAULT_STAGE5R_ROOT,
    stage5_root: Path = DEFAULT_STAGE5_ROOT,
    stage4_root: Path = DEFAULT_STAGE4_ROOT,
    stage3r_root: Path = DEFAULT_STAGE3R_ROOT,
    stage3_root: Path = DEFAULT_STAGE3_ROOT,
    stage2_root: Path = DEFAULT_STAGE2_ROOT,
    stage1_root: Path = DEFAULT_STAGE1_ROOT,
    profile_name: str = "reference",
    workers: int = 20,
    max_hours: float = 4.0,
    resume: bool = False,
    authorize_gate_override: bool = False,
    authorize_final_audit: bool = False,
) -> dict[str, Any]:
    """Run one separately gated Stage-6 round under a four-hour wall contract."""

    require_pinned_numpy()
    if round_name not in ROUNDS:
        raise ValueError(f"unknown Stage-6 round {round_name!r}")
    if profile_name not in PUBLIC_PROFILES:
        raise ValueError(f"unknown Stage-6 profile {profile_name!r}")
    if max_hours <= 0.0 or max_hours > 4.0:
        raise ValueError("Stage-6 max-hours must be in (0, 4]")
    if workers < 1:
        raise ValueError("Stage-6 workers must be positive")
    if workers > 20:
        raise ValueError("Stage-6 workers may not exceed 20")
    if round_name == "audit" and not authorize_final_audit:
        raise ValueError("the final audit requires --authorize-final-audit")
    if round_name == "audit" and not resume:
        raise ValueError("the final audit requires --resume after reviewing prior rounds")
    if round_name != "audit" and authorize_final_audit:
        raise ValueError("final-audit authorization is valid only for the audit round")
    if not PROTOCOL_PATH.exists():
        raise FileNotFoundError(PROTOCOL_PATH)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    round_root = output / round_name
    round_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    hard_deadline = started + max_hours * 3600.0
    reserve = min(
        MinimalityContract().science_reserve_seconds,
        max(60.0, max_hours * 3600.0 * 0.125),
    )
    science_deadline = max(started, hard_deadline - reserve)
    status = _status_writer(
        output,
        round_root,
        round_name,
        profile_name,
        started,
        hard_deadline,
        science_deadline,
    )
    try:
        status("running", "freeze_and_cohort")
        (
            frozen,
            cohorts,
            locality_candidates,
            anchors,
            profile,
            contract,
            writer_contract,
            design_digest,
        ) = _prepare_stage6(
            output,
            profile_name,
            open_audit=round_name == "audit",
            stage5r_root=stage5r_root,
            stage5_root=stage5_root,
            stage4_root=stage4_root,
            stage3r_root=stage3r_root,
            stage3_root=stage3_root,
            stage2_root=stage2_root,
            stage1_root=stage1_root,
        )
        _atomic_json(
            round_root / "MANIFEST.json",
            {
                "experiment": "ca_motif_lineage_stage_6",
                "round": round_name,
                "profile": profile_name,
                "design_digest": design_digest,
                "contract_digest": contract.digest,
                "workers": workers,
                "max_hours": max_hours,
                "resume": resume,
                "authorize_gate_override": authorize_gate_override,
                "authorize_final_audit": authorize_final_audit,
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
        decisions: dict[str, dict[str, Any]] = {}
        predecessors = {
            "locality": (),
            "scale": ("locality",),
            "compression": ("locality", "scale"),
            "ecology": ("locality", "scale", "compression"),
            "audit": ("locality", "scale", "compression", "ecology"),
        }[round_name]
        for predecessor in predecessors:
            decisions[predecessor] = _require_prior_round(
                output,
                predecessor,
                design_digest,
                scientific_profile=scientific,
                authorize_gate_override=authorize_gate_override,
            )
        if round_name == "locality":
            results = _run_locality_round(
                output,
                round_root,
                frozen,
                cohorts,
                locality_candidates,
                anchors,
                profile,
                profile_name,
                contract,
                writer_contract,
                design_digest,
                workers=workers,
                resume=resume,
                science_deadline=science_deadline,
                status=status,
            )
        elif round_name == "scale":
            results = _run_scale_round(
                output,
                round_root,
                frozen,
                cohorts,
                locality_candidates,
                anchors,
                decisions["locality"],
                profile,
                profile_name,
                contract,
                writer_contract,
                design_digest,
                workers=workers,
                resume=resume,
                science_deadline=science_deadline,
                status=status,
            )
        elif round_name == "compression":
            results = _run_compression_round(
                output,
                round_root,
                frozen,
                cohorts,
                locality_candidates,
                decisions["locality"],
                profile,
                profile_name,
                contract,
                writer_contract,
                design_digest,
                workers=workers,
                resume=resume,
                science_deadline=science_deadline,
                status=status,
            )
        elif round_name == "ecology":
            results = _run_ecology_round(
                output,
                round_root,
                frozen,
                cohorts,
                locality_candidates,
                decisions["locality"],
                decisions["compression"],
                profile,
                profile_name,
                contract,
                writer_contract,
                design_digest,
                workers=workers,
                resume=resume,
                science_deadline=science_deadline,
                status=status,
            )
        else:
            results = _run_audit_round(
                output,
                round_root,
                frozen,
                cohorts,
                locality_candidates,
                anchors,
                decisions["locality"],
                decisions["scale"],
                decisions["compression"],
                decisions["ecology"],
                profile,
                profile_name,
                contract,
                writer_contract,
                design_digest,
                workers=workers,
                resume=resume,
                science_deadline=science_deadline,
                status=status,
            )
        state = str(results.get("state", "unknown"))
        next_round = (
            ROUNDS[ROUNDS.index(round_name) + 1]
            if state == "complete" and round_name != "audit"
            else None
        )
        _atomic_json(
            output / "QUEUE.json",
            {
                "experiment": "ca_motif_lineage_stage_6",
                "design_digest": design_digest,
                "state": (
                    "blocked_pending_human_review"
                    if state == "complete" and next_round
                    else state
                ),
                "completed_round": round_name if state == "complete" else None,
                "next_round": next_round,
                "automatic_launch": False,
                "review_required": bool(next_round),
            },
        )
        status(
            state,
            "campaign",
            next_round=next_round,
            stage_gate=results.get("stage_gate"),
        )
        return results
    except BaseException as error:
        status("failed", "campaign", error=repr(error))
        raise
