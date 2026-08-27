"""Stage-4 compression and robustness of the renewed CA motif carrier.

Only the latent payload crosses a generation boundary. Codec fitting is
label-blind and uses already-exposed Stage-3R diagnostic traces.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass, replace
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

from .causal_heredity import _atomic_json, _atomic_text, _hash_seed, _sha256, _state_from_hex
from .e19 import require_pinned_numpy
from .life_family import live_2x2_counts_batch
from .lineage_field import load_round3_pairs
from .motif_lineage import (
    MotifContract,
    ReaderConfiguration,
    _bootstrap,
    _founders,
    _paired_uniforms,
    _step,
    apply_energy_reader,
    motif3_codes,
    write_parent_carriers,
)
from .motif_lineage_stage3 import CHECKPOINT_GENERATIONS, motif_counts_batch, write_energy_from_counts
from .motif_repair import (
    DEFAULT_STAGE3_ROOT,
    RepairProfile,
    _carrier_summary,
    _score_state,
    _strict_confirmation_gate,
    heldout_lineage_accuracy,
    load_frozen_stage3,
)


ROOT = Path(__file__).resolve().parent.parent
PROTOCOL_PATH = ROOT / "CA_MOTIF_LINEAGE_STAGE4_PROTOCOL.md"
DEFAULT_STAGE3R_ROOT = ROOT / "results/ca-motif-lineage-stage-3r"
DEFAULT_STAGE2_ROOT = ROOT / "results/ca-motif-lineage-stage-2"
DEFAULT_STAGE1_ROOT = ROOT / "results/ca-motif-lineage-stage-1"
RULE = 31649
ANCHOR_ID = "identity-r512-f32"
RANKS = (256, 128, 64, 32, 16, 8)
PHASES = ("audit", "fit", "screen", "qualify", "stress", "transfer", "adjudicate", "confirm")
DEFAULT_PRECONFIRMATION_PHASES = PHASES[:-1]
CAUSAL_CONDITIONS = (
    "intact",
    "zero_every_boundary",
    "shuffle_every_boundary",
    "latent_shuffle_every_boundary",
    "read_disabled",
    "founder_write_disabled",
    "no_rewrite",
    "ablate_after_g2",
    "rescue_same_enter_g4",
    "rescue_opposite_enter_g4",
    "opposite_founder",
    "carrier_corruption_1",
)


@dataclass(frozen=True)
class CompressionContract:
    implementation_version: str = "ca-motif-lineage-stage4-cleanroom-v1"
    namespace: str = "plastic-ca-motif-lineage-stage4-v1"
    rule: int = RULE
    generation_sweeps: int = 64
    read_sweeps: int = 32
    write_start: int = 49
    write_end: int = 64
    observe_start: int = 57
    repair_gain: float = 0.50
    stale_retention: float = 0.50
    process_noise: float = 0.002
    carrier_corruption: float = 0.01
    screen_generation4: float = 0.20
    screen_generation8: float = 0.15
    screen_generation16: float = 0.10
    screen_anchor_retention: float = 0.60
    finalist_anchor_retention: float = 0.70
    control_advantage: float = 0.10
    survival_gate: float = 0.90
    loss_fraction: float = 0.70
    rescue_fraction: float = 0.70
    strict_alpha: float = 0.025
    confirmation_alpha_per_codec: float = 0.005
    decoder_splits: int = 4
    science_reserve_seconds: float = 1800.0

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "visible_reset": "bitwise-identical native board before every generation",
                "boundary_object": "quantized latent payload only",
                "writer_access": "daughter sweeps 49-64 motif counts; no lineage labels or targets",
                "reader_access": "decoded latent payload only during sweeps 1-32",
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
class CompressionProfile:
    selection_pairs: int
    screen_replicates: int
    screen_generations: int
    qualification_replicates: int
    qualification_generations: int
    stress_pairs: int
    stress_replicates: int
    stress_generations: int
    transfer_pairs_per_rule: int
    transfer_replicates: int
    transfer_generations: int
    confirmation_pairs: int
    confirmation_replicates: int
    confirmation_generations: int
    bootstrap_resamples: int


COMPRESSION_PROFILES: dict[str, CompressionProfile] = {
    "smoke": CompressionProfile(2, 2, 4, 2, 4, 2, 2, 4, 2, 2, 4, 2, 2, 4, 100),
    "pilot": CompressionProfile(16, 4, 8, 8, 16, 8, 4, 16, 16, 4, 16, 16, 8, 16, 1_000),
    "reference": CompressionProfile(
        96, 16, 8, 32, 16, 32, 16, 16, 64, 8, 16, 128, 64, 16, 10_000
    ),
}
PUBLIC_PROFILES = tuple(COMPRESSION_PROFILES)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_frozen_stage3r(
    stage3r_root: Path = DEFAULT_STAGE3R_ROOT,
    stage3_root: Path = DEFAULT_STAGE3_ROOT,
    stage2_root: Path = DEFAULT_STAGE2_ROOT,
    stage1_root: Path = DEFAULT_STAGE1_ROOT,
) -> dict[str, Any]:
    """Validate the positive Stage-3R ancestry and expose no new pair."""

    stage3r_root = stage3r_root.resolve()
    paths = {
        key: stage3r_root / filename
        for key, filename in (
            ("results", "RESULTS.json"),
            ("decision", "STAGE_DECISION.json"),
            ("design", "DESIGN.json"),
            ("cohorts", "COHORTS.json"),
            ("manifest", "MANIFEST.json"),
            ("models", "REPAIR_MODELS.json"),
        )
    }
    missing = [path for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing frozen Stage-3R artifacts: {missing}")
    payload = {key: _load_json(path) for key, path in paths.items()}
    digest = str(payload["decision"].get("design_digest"))
    for key in ("results", "design", "cohorts", "manifest", "models"):
        if str(payload[key].get("design_digest")) != digest:
            raise ValueError(f"Stage-3R {key} design digest does not match the decision")
    if payload["results"].get("state") != "complete":
        raise ValueError("Stage 3R is not complete")
    if payload["decision"].get("decision") != "advance_to_stage_4_after_review":
        raise ValueError("Stage-3R decision does not permit Stage 4")
    if payload["decision"].get("verdict") != "STRICT_RENEWED_CA_PLASTIC_HEREDITY":
        raise ValueError("Stage 3R did not pass its strict primary gate")
    candidate = payload["results"]["adjudication"]["candidates"].get(
        "simple--strict-49-64--gain-050"
    )
    if not candidate or not candidate.get("strict", {}).get("renewed_gate"):
        raise ValueError("the registered Stage-3R winner is absent or did not pass")
    model = candidate.get("model", {})
    if model.get("window_id") != "strict-49-64" or float(model.get("gain", -1.0)) != 0.5:
        raise ValueError("the Stage-3R winner does not match the frozen Stage-4 mechanism")

    frozen3 = load_frozen_stage3(stage3_root, stage2_root, stage1_root)
    if payload["results"].get("stage3_design_digest") != frozen3["design_digest"]:
        raise ValueError("Stage-3R ancestry does not match Stage 3")
    used_ids = set(frozen3["used_pair_ids"])
    used_ids.update(payload["cohorts"]["selection_pair_ids"])
    used_ids.update(payload["cohorts"]["confirmation_pair_ids"])
    by_id = {pair["pair_id"]: pair for pair in frozen3["all_pairs"]}
    if not used_ids <= set(by_id):
        raise ValueError("a frozen Stage-3R pair is missing from the pair bank")
    trace_paths = sorted((stage3r_root / "diagnostics/traces").glob("*.npz"))
    if len(trace_paths) != len(payload["cohorts"]["diagnostic_pair_ids"]):
        raise ValueError("Stage-3R diagnostic traces are incomplete")
    return {
        **payload,
        "root": stage3r_root,
        "paths": paths,
        "design_digest": digest,
        "stage3": frozen3,
        "configuration": frozen3["configuration"],
        "reference": frozen3["reference"],
        "all_pairs": frozen3["all_pairs"],
        "by_id": by_id,
        "used_pair_ids": used_ids,
        "trace_paths": trace_paths,
    }


def select_compression_cohorts(
    profile: CompressionProfile,
    frozen: dict[str, Any],
    contract: CompressionContract,
    *,
    profile_name: str,
) -> dict[str, list[dict[str, Any]]]:
    unused = [pair for pair in frozen["all_pairs"] if pair["pair_id"] not in frozen["used_pair_ids"]]
    ordered = sorted(
        unused,
        key=lambda pair: (
            hashlib.sha256(f"{contract.namespace}:cohort:{pair['pair_id']}".encode()).hexdigest(),
            pair["pair_id"],
        ),
    )
    if profile_name == "smoke":
        exposed_ids = frozen["cohorts"]["selection_pair_ids"][: profile.selection_pairs]
        selection = [frozen["by_id"][pair_id] for pair_id in exposed_ids]
        confirmation = selection[: profile.confirmation_pairs]
        reserve: list[dict[str, Any]] = []
    else:
        stop = profile.selection_pairs + profile.confirmation_pairs
        if len(ordered) < stop:
            raise ValueError("not enough untouched rule-31649 pairs for Stage 4")
        selection = ordered[: profile.selection_pairs]
        confirmation = ordered[profile.selection_pairs:stop]
        reserve = ordered[stop:]
        selected_ids = {pair["pair_id"] for pair in selection}
        confirmation_ids = {pair["pair_id"] for pair in confirmation}
        if selected_ids & confirmation_ids or (selected_ids | confirmation_ids) & frozen["used_pair_ids"]:
            raise AssertionError("Stage-4 scientific cohorts are not fresh and disjoint")
    secondary = load_round3_pairs()
    transfer: list[dict[str, Any]] = []
    for rule in (31648, 70366):
        rows = secondary[rule]
        count = min(profile.transfer_pairs_per_rule, len(rows))
        transfer.extend({**pair, "stage4_transfer_rule": rule} for pair in rows[:count])
    return {
        "selection": selection,
        "stress": selection[: profile.stress_pairs],
        "confirmation": confirmation,
        "reserve": reserve,
        "transfer": transfer,
    }


def _canonicalize_columns(basis: np.ndarray) -> np.ndarray:
    result = np.asarray(basis, dtype=np.float64).copy()
    indices = np.argmax(np.abs(result), axis=0)
    signs = np.sign(result[indices, np.arange(result.shape[1])])
    signs[signs == 0.0] = 1.0
    result *= signs
    return result.astype(np.float32)


def _hadamard(order: int = 512) -> np.ndarray:
    if order <= 0 or order & (order - 1):
        raise ValueError("Hadamard order must be a positive power of two")
    matrix = np.ones((1, 1), dtype=np.float32)
    while len(matrix) < order:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    return matrix / np.float32(math.sqrt(order))


def _d4_orbits() -> list[list[int]]:
    def transform(code: int, rotation: int, reflect: bool) -> int:
        source = np.asarray([(code >> bit) & 1 for bit in range(9)], dtype=np.uint8).reshape(3, 3)
        target = np.rot90(source, rotation)
        if reflect:
            target = np.fliplr(target)
        return int(sum(int(value) << bit for bit, value in enumerate(target.ravel())))

    unseen = set(range(512))
    groups: list[list[int]] = []
    while unseen:
        code = min(unseen)
        orbit = sorted({transform(code, rotation, reflect) for rotation in range(4) for reflect in (False, True)})
        groups.append(orbit)
        unseen.difference_update(orbit)
    return groups


def _count_groups() -> list[list[int]]:
    groups: list[list[int]] = []
    centre_bit = 4
    for centre in (0, 1):
        for neighbours in range(9):
            groups.append(
                [
                    code
                    for code in range(512)
                    if ((code >> centre_bit) & 1) == centre
                    and code.bit_count() - centre == neighbours
                ]
            )
    return groups


def _pool_basis(groups: Sequence[Sequence[int]]) -> np.ndarray:
    basis = np.zeros((512, len(groups)), dtype=np.float32)
    for column, group in enumerate(groups):
        if not group:
            raise ValueError("pooling group cannot be empty")
        basis[np.asarray(group, dtype=np.int64), column] = 1.0 / math.sqrt(len(group))
    np.testing.assert_allclose(basis.T @ basis, np.eye(len(groups)), atol=1e-6)
    return basis


def load_stage3r_fit_matrix(frozen: dict[str, Any]) -> tuple[np.ndarray, list[str], dict[str, str]]:
    """Load only exposed strict-49-64 centroids; no lineage label is returned."""

    window_index = 3
    samples: list[np.ndarray] = []
    groups: list[str] = []
    hashes: dict[str, str] = {}
    for path in frozen["trace_paths"]:
        with np.load(path, allow_pickle=False) as archive:
            centroids = np.asarray(archive["window_centroids"], dtype=np.float32)
        if centroids.ndim != 4 or centroids.shape[0] <= window_index or centroids.shape[-1] != 512:
            raise ValueError(f"unexpected Stage-3R trace shape: {path}")
        values = centroids[window_index].reshape(-1, 512) * np.float32(0.5)
        samples.append(values)
        groups.extend([path.stem] * len(values))
        hashes[str(path.relative_to(ROOT))] = _sha256(path)
    matrix = np.concatenate(samples, axis=0).astype(np.float32)
    if not np.isfinite(matrix).all():
        raise ValueError("non-finite Stage-3R fit trace")
    return matrix, groups, hashes


def _codebook_bits(family: str, rank: int, bits: int) -> int:
    if family in ("identity", "count18", "d4"):
        basis_bits = 0
    elif family in ("walsh", "sparse"):
        basis_bits = rank * 9
    elif family == "random":
        basis_bits = 64
    else:
        basis_bits = 512 * rank * 32
    scale_bits = rank * 32 if bits < 32 else 0
    return basis_bits + scale_bits


def _model(
    family: str,
    rank: int,
    bits: int,
    basis: np.ndarray | None,
    fit_matrix: np.ndarray,
    *,
    interpretable: bool,
) -> dict[str, Any]:
    suffix = "f32" if bits == 32 else f"q{bits:02d}"
    coefficients = fit_matrix if basis is None else fit_matrix @ basis
    scale = None
    if bits < 32:
        scale = np.max(np.abs(coefficients), axis=0).astype(np.float32)
        scale[scale < 1e-12] = 1.0
    return {
        "candidate_id": f"{family}-r{rank:03d}-{suffix}",
        "family": family,
        "rank": rank,
        "bits": bits,
        "precision": "float32" if bits == 32 else "fixed-scale-signed",
        "payload_bits": rank * bits,
        "codebook_bits": _codebook_bits(family, rank, bits),
        "interpretable": interpretable,
        "runtime_parent_access": False,
        "runtime_label_access": False,
        "runtime_target_access": False,
        **({"basis": np.asarray(basis, dtype=np.float32)} if basis is not None else {}),
        **({"quantizer_scale": scale} if scale is not None else {}),
    }


def quantize_payload(values: np.ndarray, model: dict[str, Any]) -> tuple[np.ndarray, float]:
    latent = np.asarray(values, dtype=np.float32)
    bits = int(model["bits"])
    if bits == 32:
        return latent.copy(), 0.0
    scale = np.asarray(model["quantizer_scale"], dtype=np.float32)
    qmax = (1 << (bits - 1)) - 1
    normalized = np.divide(latent, scale, out=np.zeros_like(latent), where=scale > 0)
    clipping = float(np.mean(np.abs(normalized) > 1.0))
    integers = np.clip(np.rint(normalized * qmax), -qmax, qmax)
    result = integers * (scale / np.float32(qmax))
    result[latent == 0.0] = 0.0
    return result.astype(np.float32), clipping


def encode_payload(carrier: np.ndarray, model: dict[str, Any]) -> tuple[np.ndarray, float]:
    values = np.asarray(carrier, dtype=np.float32)
    latent = values.copy() if model["family"] == "identity" else values @ np.asarray(model["basis"], dtype=np.float32)
    return quantize_payload(latent, model)


def decode_payload(payload: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    latent = np.asarray(payload, dtype=np.float32)
    if model["family"] == "identity":
        return latent.copy()
    return (latent @ np.asarray(model["basis"], dtype=np.float32).T).astype(np.float32)


def fit_codec_atlas(fit_matrix: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fit the registered 50-codec atlas without using history labels."""

    values = np.asarray(fit_matrix, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 512:
        raise ValueError("fit matrix must have shape (sample, 512)")
    models: list[dict[str, Any]] = []
    for bits in (32, 8, 4, 2):
        models.append(_model("identity", 512, bits, None, values, interpretable=True))

    covariance = values.astype(np.float64).T @ values.astype(np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    pca = _canonicalize_columns(eigenvectors[:, order[: max(RANKS)]])
    for rank in RANKS:
        for bits in (32, 8, 4):
            models.append(_model("pca", rank, bits, pca[:, :rank], values, interpretable=False))

    walsh = _hadamard(512)
    walsh_energy = np.sum((values @ walsh) ** 2, axis=0)
    walsh_order = np.argsort(-walsh_energy, kind="stable")
    for rank in RANKS:
        for bits in (8, 4):
            models.append(_model("walsh", rank, bits, walsh[:, walsh_order[:rank]], values, interpretable=True))

    sparse_order = np.argsort(-np.sum(values * values, axis=0), kind="stable")
    for rank in RANKS:
        basis = np.eye(512, dtype=np.float32)[:, sparse_order[:rank]]
        models.append(_model("sparse", rank, 8, basis, values, interpretable=True))

    rng = np.random.default_rng(_hash_seed("stage4-random-orthoprojector", 1))
    random_basis, _ = np.linalg.qr(rng.normal(size=(512, max(RANKS))), mode="reduced")
    random_basis = _canonicalize_columns(random_basis)
    for rank in RANKS:
        models.append(_model("random", rank, 8, random_basis[:, :rank], values, interpretable=False))

    d4_basis = _pool_basis(_d4_orbits())
    count_basis = _pool_basis(_count_groups())
    if d4_basis.shape[1] != 102 or count_basis.shape[1] != 18:
        raise AssertionError("registered structural codec dimensions changed")
    for family, basis in (("d4", d4_basis), ("count18", count_basis)):
        for bits in (8, 4):
            models.append(_model(family, basis.shape[1], bits, basis, values, interpretable=True))

    if len(models) != 50 or len({model["candidate_id"] for model in models}) != 50:
        raise AssertionError("registered codec atlas must contain exactly 50 unique codecs")
    total_energy = float(np.sum(eigenvalues))
    audit = {
        "label_blind": True,
        "fit_samples": len(values),
        "fit_dimensions": 512,
        "model_count": len(models),
        "d4_orbits": d4_basis.shape[1],
        "count_groups": count_basis.shape[1],
        "pca_explained_energy": {
            str(rank): float(np.sum(eigenvalues[order[:rank]]) / total_energy) if total_energy > 0 else 0.0
            for rank in RANKS
        },
        "identity_float32_exact": bool(
            np.array_equal(decode_payload(encode_payload(values, models[0])[0], models[0]), values)
        ),
    }
    return models, audit


def save_codec_models(
    output: Path,
    models: Sequence[dict[str, Any]],
    *,
    design_digest: str,
    fit_trace_digest: str,
) -> dict[str, Any]:
    arrays: dict[str, np.ndarray] = {}
    metadata: list[dict[str, Any]] = []
    for index, model in enumerate(models):
        row: dict[str, Any] = {}
        array_keys: dict[str, str] = {}
        for key, value in model.items():
            if isinstance(value, np.ndarray):
                array_key = f"model_{index:03d}__{key}"
                arrays[array_key] = value
                array_keys[key] = array_key
            else:
                row[key] = value
        row["array_keys"] = array_keys
        metadata.append(row)
    path = output / "CODEC_MODELS.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    manifest = {
        "design_digest": design_digest,
        "fit_trace_digest": fit_trace_digest,
        "allow_pickle": False,
        "model_sha256": _sha256(path),
        "models": metadata,
    }
    _atomic_json(output / "CODEC_MODELS.json", manifest)
    return manifest


def load_codec_models(output: Path, design_digest: str) -> list[dict[str, Any]]:
    manifest = _load_json(output / "CODEC_MODELS.json")
    if manifest.get("design_digest") != design_digest:
        raise ValueError("codec-model design digest mismatch")
    path = output / "CODEC_MODELS.npz"
    if _sha256(path) != manifest.get("model_sha256"):
        raise ValueError("codec-model hash mismatch")
    result: list[dict[str, Any]] = []
    with np.load(path, allow_pickle=False) as archive:
        for metadata in manifest["models"]:
            model = {key: value for key, value in metadata.items() if key != "array_keys"}
            for key, array_key in metadata["array_keys"].items():
                model[key] = np.asarray(archive[array_key])
            result.append(model)
    return result


def stress_scenarios() -> dict[str, dict[str, float | int]]:
    scenarios: dict[str, dict[str, float | int]] = {
        "ordinary": {},
        **{f"erasure_{int(value * 1000):03d}": {"erasure": value} for value in (0.01, 0.05, 0.10, 0.25, 0.50)},
        **{f"sign_{int(value * 1000):03d}": {"sign_corruption": value} for value in (0.01, 0.05, 0.10, 0.20)},
        **{f"process_{int(value * 1000):03d}": {"process_noise": value} for value in (0.002, 0.004, 0.008, 0.016)},
        **{f"gain_{int(value * 1000):03d}": {"repair_gain": value} for value in (0.35, 0.425, 0.50, 0.575, 0.65)},
        **{f"read_{value:02d}": {"read_sweeps": value} for value in (16, 24, 32, 40)},
        "moderate_joint": {"erasure": 0.10, "sign_corruption": 0.05, "process_noise": 0.004},
        "harsh_joint": {"erasure": 0.25, "sign_corruption": 0.10, "process_noise": 0.008},
    }
    return scenarios


def _stage4_uniforms(
    pair_id: str,
    purpose: str,
    generation: int,
    sweep: int,
    replicates: int,
) -> np.ndarray:
    return _paired_uniforms(
        pair_id,
        f"stage4-{purpose}-generation-{generation}",
        sweep,
        replicates,
    )


def _repeat_histories(values: np.ndarray, replicates: int) -> np.ndarray:
    return np.concatenate(
        (np.repeat(values[0:1], replicates, axis=0), np.repeat(values[1:2], replicates, axis=0)),
        axis=0,
    )


def _swap_histories(values: np.ndarray, replicates: int) -> np.ndarray:
    return np.concatenate((values[replicates:], values[:replicates]), axis=0)


def _latent_summary(payload: np.ndarray, replicates: int) -> dict[str, float]:
    shaped = np.asarray(payload, dtype=np.float32).reshape(2, replicates, -1)
    delta = shaped[0].mean(axis=0) - shaped[1].mean(axis=0)
    return {
        "mean_abs": float(np.mean(np.abs(payload))),
        "centroid_l2": float(np.linalg.norm(delta)),
        "within_history_variance": float(np.mean(np.var(shaped, axis=1))),
    }


def _damage_payload(
    payload: np.ndarray,
    model: dict[str, Any],
    pair_id: str,
    generation: int,
    replicates: int,
    stress_id: str,
    stress: dict[str, float | int],
) -> tuple[np.ndarray, float]:
    result = payload.copy()
    rank = result.shape[1]
    erasure = float(stress.get("erasure", 0.0))
    sign = float(stress.get("sign_corruption", 0.0))
    if erasure > 0.0:
        half = np.random.default_rng(
            _hash_seed("stage4-damage", pair_id, stress_id, model["candidate_id"], generation, "erase")
        ).random((replicates, rank)) < erasure
        result[np.concatenate((half, half), axis=0)] = 0.0
    if sign > 0.0:
        half = np.random.default_rng(
            _hash_seed("stage4-damage", pair_id, stress_id, model["candidate_id"], generation, "sign")
        ).random((replicates, rank)) < sign
        result[np.concatenate((half, half), axis=0)] *= -1.0
    return quantize_payload(result, model)


def _apply_payload_intervention(
    payload: np.ndarray,
    model: dict[str, Any],
    condition: str,
    generation: int,
    pair_id: str,
    replicates: int,
    contract: CompressionContract,
    source_exits: Sequence[np.ndarray] | None,
) -> tuple[np.ndarray, float]:
    result = payload.copy()
    clipping = 0.0
    if condition == "zero_every_boundary":
        result.fill(0.0)
    elif condition == "shuffle_every_boundary":
        decoded = decode_payload(result, model)
        permutation = np.random.default_rng(
            _hash_seed(contract.namespace, pair_id, "decoded-shuffle", generation)
        ).permutation(512)
        result, clipping = encode_payload(decoded[:, permutation], model)
    elif condition == "latent_shuffle_every_boundary":
        permutation = np.random.default_rng(
            _hash_seed(contract.namespace, pair_id, "latent-shuffle", generation)
        ).permutation(result.shape[1])
        result, clipping = quantize_payload(result[:, permutation], model)
    elif condition == "opposite_founder" and generation == 1:
        result = _swap_histories(result, replicates)
    elif condition in ("ablate_after_g2", "rescue_same_enter_g4", "rescue_opposite_enter_g4") and generation == 3:
        result.fill(0.0)
    elif condition in ("rescue_same_enter_g4", "rescue_opposite_enter_g4") and generation == 4:
        if source_exits is None or len(source_exits) < 3:
            raise ValueError("rescue requires a contemporaneous intact sister payload")
        result = source_exits[2].copy()
        if condition == "rescue_opposite_enter_g4":
            result = _swap_histories(result, replicates)
    elif condition == "carrier_corruption_1":
        half = np.random.default_rng(
            _hash_seed(contract.namespace, pair_id, "latent-corruption", generation)
        ).random((replicates, result.shape[1])) < contract.carrier_corruption
        result[np.concatenate((half, half), axis=0)] *= -1.0
        result, clipping = quantize_payload(result, model)
    return result, clipping


def simulate_compressed_lineage(
    pair: dict[str, Any],
    configuration: ReaderConfiguration,
    model: dict[str, Any],
    condition: str,
    replicates: int,
    generations: int,
    reference: dict[int, dict[str, np.ndarray]],
    writer_contract: MotifContract,
    contract: CompressionContract,
    *,
    stress_id: str = "ordinary",
    stress: dict[str, float | int] | None = None,
    source_exits: Sequence[np.ndarray] | None = None,
    retain_exits: bool = False,
    rule_override: int | None = None,
) -> tuple[dict[str, Any], list[np.ndarray]]:
    """Run a lineage in which only an encoded payload survives each reset."""

    if condition not in CAUSAL_CONDITIONS:
        raise ValueError(f"unknown Stage-4 condition {condition!r}")
    stress = dict(stress or {})
    pair_id = str(pair["pair_id"])
    rule = int(rule_override if rule_override is not None else contract.rule)
    reset_state = _state_from_hex("life", pair["donor_a"]["initial_state_hex"])
    other_reset = _state_from_hex("life", pair["donor_b"]["initial_state_hex"])
    if not np.array_equal(reset_state, other_reset):
        raise AssertionError(f"visible reset mismatch in pair {pair_id}")
    reset = np.repeat(reset_state[None, ...], 2 * replicates, axis=0)

    founder_writer = replace(writer_contract, rule=rule)
    written = write_parent_carriers(
        _founders(pair), (configuration.write_window,), reference, founder_writer
    )[configuration.write_window]
    founder_carrier = _repeat_histories(written[configuration.family], replicates)
    payload, founder_clipping = encode_payload(founder_carrier, model)
    if condition == "founder_write_disabled":
        payload.fill(0.0)
    founder_terminal = written["terminal"]
    alive = np.ones(2 * replicates, dtype=np.bool_)
    checkpoints = {value for value in CHECKPOINT_GENERATIONS if value <= generations}
    outcomes: dict[str, Any] = {}
    decoders: dict[str, Any] = {}
    carrier_history: dict[str, Any] = {}
    exits: list[np.ndarray] = []
    clipping_values = [founder_clipping]
    reference_probability = reference[configuration.write_window]["motif_probability"]
    read_sweeps = int(stress.get("read_sweeps", contract.read_sweeps))
    process_noise = float(stress.get("process_noise", contract.process_noise))
    repair_gain = float(stress.get("repair_gain", contract.repair_gain))

    for generation in range(1, generations + 1):
        payload, intervention_clipping = _apply_payload_intervention(
            payload, model, condition, generation, pair_id, replicates, contract, source_exits
        )
        clipping_values.append(intervention_clipping)
        payload, damage_clipping = _damage_payload(
            payload, model, pair_id, generation, replicates, stress_id, stress
        )
        clipping_values.append(damage_clipping)
        entry_payload = payload.copy()
        entry = decode_payload(entry_payload, model)
        entry_summary = _carrier_summary(entry, replicates)
        latent_entry_summary = _latent_summary(entry_payload, replicates)

        state = reset.copy()
        state[~alive] = False
        if not np.array_equal(state[alive], reset[alive]):
            raise AssertionError("visible reset was not bitwise identical")
        recent: deque[np.ndarray] = deque(maxlen=writer_contract.observation_window)
        counts = np.zeros((2 * replicates, 512), dtype=np.float64)
        for sweep in range(1, contract.generation_sweeps + 1):
            predicted = _step(state, rule)
            if condition != "read_disabled" and sweep <= read_sweeps:
                predicted = apply_energy_reader(
                    predicted,
                    entry,
                    _stage4_uniforms(pair_id, "read", generation, sweep, replicates),
                    configuration.strength,
                )
            predicted ^= (
                _stage4_uniforms(pair_id, "process", generation, sweep, replicates)
                < process_noise
            )
            predicted[~alive] = False
            state = predicted
            if contract.write_start <= sweep <= contract.write_end:
                counts += motif_counts_batch(motif3_codes(state))
            if sweep >= contract.observe_start:
                recent.append(live_2x2_counts_batch(state))
        alive &= state.any(axis=(1, 2))
        raw = write_energy_from_counts(counts, reference_probability, writer_contract)
        if condition == "no_rewrite":
            payload, clipping = quantize_payload(
                entry_payload * np.float32(contract.stale_retention), model
            )
        else:
            payload, clipping = encode_payload(raw * np.float32(repair_gain), model)
        clipping_values.append(clipping)
        payload[~alive] = 0.0
        decoded_exit = decode_payload(payload, model)

        if generation in checkpoints:
            outcome, vectors = _score_state(
                state, recent, pair, founder_terminal, replicates, writer_contract
            )
            outcomes[str(generation)] = outcome
            decoders[str(generation)] = {
                "carrier_balanced_accuracy": heldout_lineage_accuracy(
                    decoded_exit,
                    replicates,
                    _hash_seed(contract.namespace, pair_id, model["candidate_id"], condition, stress_id, generation, "carrier"),
                    contract.decoder_splits,
                ),
                "latent_balanced_accuracy": heldout_lineage_accuracy(
                    payload,
                    replicates,
                    _hash_seed(contract.namespace, pair_id, model["candidate_id"], condition, stress_id, generation, "latent"),
                    contract.decoder_splits,
                ),
                "phenotype_balanced_accuracy": heldout_lineage_accuracy(
                    vectors,
                    replicates,
                    _hash_seed(contract.namespace, pair_id, model["candidate_id"], condition, stress_id, generation, "phenotype"),
                    contract.decoder_splits,
                ),
            }
            carrier_history[str(generation)] = {
                "entry": entry_summary,
                "exit": _carrier_summary(decoded_exit, replicates),
                "latent_entry": latent_entry_summary,
                "latent_exit": _latent_summary(payload, replicates),
                "surviving_futures": int(np.count_nonzero(alive)),
            }
        if retain_exits:
            exits.append(payload.copy())

    return (
        {
            "candidate_id": model["candidate_id"],
            "condition": condition,
            "stress_id": stress_id,
            "stress": stress,
            "rule": rule,
            "reset_sha256": hashlib.sha256(reset_state.tobytes()).hexdigest(),
            "reset_asserted_before_every_generation": True,
            "founder_carrier": _carrier_summary(founder_carrier, replicates),
            "founder_payload": _latent_summary(encode_payload(founder_carrier, model)[0], replicates),
            "quantizer_clipping_fraction_mean": float(np.mean(clipping_values)),
            "outcomes": outcomes,
            "decoders": decoders,
            "carrier_history": carrier_history,
        },
        exits,
    )


def _screen_pair_task(
    payload: tuple[dict[str, Any], list[dict[str, Any]], MotifContract, CompressionContract, dict[int, dict[str, np.ndarray]]]
) -> dict[str, Any]:
    item, models, writer_contract, contract, reference = payload
    configuration = ReaderConfiguration(**item["configuration"])
    candidates: dict[str, Any] = {}
    for model in models:
        result, _ = simulate_compressed_lineage(
            item["pair"], configuration, model, "intact", int(item["replicates"]),
            int(item["generations"]), reference, writer_contract, contract
        )
        candidates[str(model["candidate_id"])] = result
    return {
        "checkpoint": item["checkpoint"],
        "pair_id": item["pair"]["pair_id"],
        "replicates": int(item["replicates"]),
        "generations": int(item["generations"]),
        "candidates": candidates,
    }


def _qualification_pair_task(
    payload: tuple[dict[str, Any], list[dict[str, Any]], MotifContract, CompressionContract, dict[int, dict[str, np.ndarray]]]
) -> dict[str, Any]:
    item, models, writer_contract, contract, reference = payload
    pair = item["pair"]
    configuration = ReaderConfiguration(**item["configuration"])
    replicates = int(item["replicates"])
    generations = int(item["generations"])
    candidates: dict[str, Any] = {}
    task_models = [
        model for model in models
        if item.get("candidate_id") is None or model["candidate_id"] == item["candidate_id"]
    ]
    for model in task_models:
        intact, exits = simulate_compressed_lineage(
            pair, configuration, model, "intact", replicates, generations,
            reference, writer_contract, contract, retain_exits=True
        )
        conditions: dict[str, Any] = {"intact": intact}
        for condition in CAUSAL_CONDITIONS[1:]:
            result, _ = simulate_compressed_lineage(
                pair, configuration, model, condition, replicates, generations,
                reference, writer_contract, contract, source_exits=exits
            )
            conditions[condition] = result
        candidates[str(model["candidate_id"])] = {
            "candidate_id": model["candidate_id"], "conditions": conditions
        }
    return {
        "checkpoint": item["checkpoint"],
        "pair_id": pair["pair_id"],
        "replicates": replicates,
        "generations": generations,
        "candidates": candidates,
    }


def _stress_pair_task(
    payload: tuple[dict[str, Any], list[dict[str, Any]], MotifContract, CompressionContract, dict[int, dict[str, np.ndarray]]]
) -> dict[str, Any]:
    item, models, writer_contract, contract, reference = payload
    configuration = ReaderConfiguration(**item["configuration"])
    results: dict[str, Any] = {}
    scenarios = stress_scenarios()
    task_models = [
        model for model in models
        if item.get("candidate_id") is None or model["candidate_id"] == item["candidate_id"]
    ]
    for model in task_models:
        by_scenario: dict[str, Any] = {}
        for stress_id, stress in scenarios.items():
            result, _ = simulate_compressed_lineage(
                item["pair"], configuration, model, "intact", int(item["replicates"]),
                int(item["generations"]), reference, writer_contract, contract,
                stress_id=stress_id, stress=stress
            )
            by_scenario[stress_id] = result
        results[str(model["candidate_id"])] = by_scenario
    return {
        "checkpoint": item["checkpoint"],
        "pair_id": item["pair"]["pair_id"],
        "candidates": results,
    }


def _transfer_pair_task(
    payload: tuple[dict[str, Any], list[dict[str, Any]], MotifContract, CompressionContract, dict[int, dict[str, np.ndarray]]]
) -> dict[str, Any]:
    item, models, writer_contract, contract, reference = payload
    configuration = ReaderConfiguration(**item["configuration"])
    rule = int(item["pair"]["stage4_transfer_rule"])
    candidates: dict[str, Any] = {}
    for model in models:
        result, _ = simulate_compressed_lineage(
            item["pair"], configuration, model, "intact", int(item["replicates"]),
            int(item["generations"]), reference, writer_contract, contract,
            rule_override=rule
        )
        candidates[str(model["candidate_id"])] = result
    return {
        "checkpoint": item["checkpoint"],
        "pair_id": item["pair"]["pair_id"],
        "rule": rule,
        "candidates": candidates,
    }


def _confirmation_pair_task(
    payload: tuple[dict[str, Any], list[dict[str, Any]], MotifContract, CompressionContract, dict[int, dict[str, np.ndarray]]]
) -> dict[str, Any]:
    item, models, writer_contract, contract, reference = payload
    pair = item["pair"]
    configuration = ReaderConfiguration(**item["configuration"])
    replicates = int(item["replicates"])
    generations = int(item["generations"])
    environments = {
        "ordinary": {},
        "moderate_joint": stress_scenarios()["moderate_joint"],
    }
    if item.get("environment") is not None:
        environment = str(item["environment"])
        environments = {environment: environments[environment]}
    candidates: dict[str, Any] = {}
    task_models = [
        model for model in models
        if item.get("candidate_id") is None or model["candidate_id"] == item["candidate_id"]
    ]
    for model in task_models:
        candidate_environments: dict[str, Any] = {}
        for stress_id, stress in environments.items():
            intact, exits = simulate_compressed_lineage(
                pair, configuration, model, "intact", replicates, generations,
                reference, writer_contract, contract, stress_id=stress_id,
                stress=stress, retain_exits=True
            )
            conditions: dict[str, Any] = {"intact": intact}
            for condition in CAUSAL_CONDITIONS[1:]:
                result, _ = simulate_compressed_lineage(
                    pair, configuration, model, condition, replicates, generations,
                    reference, writer_contract, contract, stress_id=stress_id,
                    stress=stress, source_exits=exits
                )
                conditions[condition] = result
            candidate_environments[stress_id] = {"conditions": conditions}
        candidates[str(model["candidate_id"])] = {"environments": candidate_environments}
    return {
        "checkpoint": item["checkpoint"],
        "pair_id": pair["pair_id"],
        "replicates": replicates,
        "generations": generations,
        "candidates": candidates,
    }


def _run_json_checkpoints(
    output: Path,
    phase: str,
    items: Sequence[dict[str, Any]],
    models: list[dict[str, Any]],
    task: Callable[..., dict[str, Any]],
    writer_contract: MotifContract,
    contract: CompressionContract,
    reference: dict[int, dict[str, np.ndarray]],
    design_digest: str,
    *,
    workers: int,
    resume: bool,
    deadline: float,
    status: Callable[..., None],
) -> tuple[list[dict[str, Any]], bool]:
    root = output / phase
    checkpoint_root = root / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    missing: list[dict[str, Any]] = []
    for item in items:
        key = str(item["checkpoint"])
        path = checkpoint_root / f"{key}.json"
        if resume and path.exists():
            payload = _load_json(path)
            if payload.get("design_digest") != design_digest:
                raise ValueError(f"{phase} checkpoint design mismatch: {path}")
            results[key] = payload["result"]
        else:
            missing.append(item)
    initial = len(results)
    started = time.monotonic()
    truncated = False

    def save(item: dict[str, Any], result: dict[str, Any]) -> None:
        key = str(item["checkpoint"])
        _atomic_json(
            checkpoint_root / f"{key}.json",
            {
                "design_digest": design_digest,
                "phase": phase,
                "checkpoint": key,
                "result": result,
            },
        )
        results[key] = result
        elapsed = max(time.monotonic() - started, 1e-6)
        completed_new = max(1, len(results) - initial)
        eta = elapsed / completed_new * max(0, len(items) - len(results))
        status(
            "running",
            phase,
            completed=len(results),
            total=len(items),
            eta_seconds=eta,
            latest_checkpoint=key,
        )

    if workers <= 1:
        for item in missing:
            if time.time() >= deadline:
                truncated = True
                break
            save(item, task((item, models, writer_contract, contract, reference)))
    elif missing and time.time() < deadline:
        pool = ProcessPoolExecutor(max_workers=max(1, min(workers, len(missing))))
        iterator = iter(missing)
        pending: dict[Any, dict[str, Any]] = {}

        def submit_one() -> bool:
            if time.time() >= deadline:
                return False
            try:
                item = next(iterator)
            except StopIteration:
                return False
            pending[pool.submit(task, (item, models, writer_contract, contract, reference))] = item
            return True

        for _ in range(min(len(missing), max(1, workers))):
            submit_one()
        try:
            while pending:
                remaining = deadline - time.time()
                if remaining <= 0.0:
                    truncated = True
                    break
                done, _ = wait(tuple(pending), timeout=min(10.0, remaining), return_when=FIRST_COMPLETED)
                for future in done:
                    item = pending.pop(future)
                    save(item, future.result())
                    submit_one()
            if truncated:
                for future in pending:
                    future.cancel()
        finally:
            pool.shutdown(wait=True, cancel_futures=truncated)
    elif missing:
        truncated = True
    complete = len(results) == len(items)
    _atomic_json(
        root / "stage_summary.json",
        {
            "design_digest": design_digest,
            "phase": phase,
            "complete": complete,
            "completed": len(results),
            "total": len(items),
            "budget_truncated": truncated or not complete,
        },
    )
    if complete:
        _atomic_text(root / "COMPLETE", "complete\n")
    return [results[key] for key in sorted(results)], complete


def _json_model(model: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in model.items() if not isinstance(value, np.ndarray)}


def _selected_models(models: Sequence[dict[str, Any]], candidate_ids: Sequence[str]) -> list[dict[str, Any]]:
    by_id = {str(model["candidate_id"]): model for model in models}
    missing = [candidate_id for candidate_id in candidate_ids if candidate_id not in by_id]
    if missing:
        raise ValueError(f"selected codec models are missing: {missing}")
    return [by_id[candidate_id] for candidate_id in candidate_ids]


def _screen_values(
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


def adjudicate_screen(
    rows: Sequence[dict[str, Any]],
    models: Sequence[dict[str, Any]],
    profile: CompressionProfile,
    contract: CompressionContract,
    complete: bool,
) -> dict[str, Any]:
    if not complete:
        return {"state": "incomplete", "selected_candidate_ids": []}
    generation = 8 if profile.screen_generations >= 8 else profile.screen_generations

    def boot(values: Sequence[float], name: str) -> dict[str, Any]:
        return _bootstrap(
            values,
            profile.bootstrap_resamples,
            _hash_seed(contract.namespace, "screen", name),
            contract.strict_alpha,
        )

    raw: dict[str, dict[str, Any]] = {}
    for model in models:
        candidate_id = str(model["candidate_id"])
        crossover_values = _screen_values(rows, candidate_id, generation, "crossover")
        raw[candidate_id] = {
            "model": _json_model(model),
            "generation": generation,
            "crossover": boot(crossover_values, f"{candidate_id}-crossover"),
            "survival_mean": float(np.mean(_screen_values(rows, candidate_id, generation, "survival"))),
            "direction_a_mean": float(np.mean(_screen_values(rows, candidate_id, generation, "direction_a"))),
            "direction_b_mean": float(np.mean(_screen_values(rows, candidate_id, generation, "direction_b"))),
            "fraction_pairs_positive": float(np.mean(np.asarray(crossover_values) > 0.0)),
        }
    anchor_mean = float(raw[ANCHOR_ID]["crossover"]["mean"] or 0.0)
    for candidate_id, summary in raw.items():
        mean = float(summary["crossover"]["mean"] or 0.0)
        retention = mean / anchor_mean if anchor_mean > 0.0 else 0.0
        lower = summary["crossover"]["ci"][0]
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
        smallest = min(models, key=lambda model: (int(model["payload_bits"]), str(model["candidate_id"])))
        strongest = min(
            models,
            key=lambda model: (
                -float(
                    raw[model["candidate_id"]]["crossover"]["mean"]
                    if raw[model["candidate_id"]]["crossover"]["mean"] is not None
                    else -1.0
                ),
                str(model["candidate_id"]),
            ),
        )
        selected = list(dict.fromkeys((ANCHOR_ID, str(smallest["candidate_id"]), str(strongest["candidate_id"]))))
    else:
        selected = [ANCHOR_ID]
        for family in sorted({str(model["family"]) for model in models}):
            eligible = [model for model in models if model["family"] == family and raw[model["candidate_id"]]["screen_eligible"]]
            if not eligible:
                continue
            smallest = min(
                eligible,
                key=lambda model: (
                    int(model["payload_bits"]),
                    -float(
                        raw[model["candidate_id"]]["crossover"]["mean"]
                        if raw[model["candidate_id"]]["crossover"]["mean"] is not None
                        else -1.0
                    ),
                    str(model["candidate_id"]),
                ),
            )
            strongest = min(
                eligible,
                key=lambda model: (
                    -float(
                        raw[model["candidate_id"]]["crossover"]["mean"]
                        if raw[model["candidate_id"]]["crossover"]["mean"] is not None
                        else -1.0
                    ),
                    int(model["payload_bits"]),
                    str(model["candidate_id"]),
                ),
            )
            selected.extend((str(smallest["candidate_id"]), str(strongest["candidate_id"])))
        selected = list(dict.fromkeys(selected))
    return {
        "state": "complete",
        "generation": generation,
        "anchor_candidate_id": ANCHOR_ID,
        "anchor_crossover_mean": anchor_mean,
        "candidate_summaries": raw,
        "selected_candidate_ids": selected,
        "scientific_gate_applied": profile.screen_generations >= 8,
    }


def _repair_profile(profile: CompressionProfile, role: str, *, confirmation: bool = False) -> RepairProfile:
    pairs = profile.confirmation_pairs if confirmation else profile.selection_pairs
    replicates = profile.confirmation_replicates if confirmation else profile.qualification_replicates
    generations = profile.confirmation_generations if confirmation else profile.qualification_generations
    return RepairProfile(role, pairs, replicates, generations, role, pairs, replicates, generations, pairs, replicates, generations, profile.bootstrap_resamples)


def _qualification_difference(
    rows: Sequence[dict[str, Any]], candidate_id: str, right: str, generation: int
) -> list[float]:
    values: list[float] = []
    for row in rows:
        try:
            intact = row["candidates"][candidate_id]["conditions"]["intact"]["outcomes"][str(generation)]["primary"]["crossover"]
            control = row["candidates"][candidate_id]["conditions"][right]["outcomes"][str(generation)]["primary"]["crossover"]
        except KeyError:
            continue
        values.append(float(intact) - float(control))
    return values


def adjudicate_qualification(
    rows: Sequence[dict[str, Any]],
    selected_ids: Sequence[str],
    profile: CompressionProfile,
    contract: CompressionContract,
    complete: bool,
) -> dict[str, Any]:
    if not complete:
        return {"state": "incomplete", "qualified_candidate_ids": [], "candidates": {}}
    gate_applied = profile.qualification_generations >= 16
    summaries: dict[str, Any] = {}
    qualified: list[str] = []
    generation = min(8, profile.qualification_generations)
    for candidate_id in selected_ids:
        if gate_applied:
            strict = _strict_confirmation_gate(
                rows,
                candidate_id,
                _repair_profile(profile, "stage4-selection"),
                contract,  # type: ignore[arg-type]
                contract.strict_alpha,
            )
        else:
            strict = {"verdict": "NOT_ADJUDICATED_PROFILE", "renewed_gate": False}
        latent_values = _qualification_difference(
            rows, candidate_id, "latent_shuffle_every_boundary", generation
        )
        latent = _bootstrap(
            latent_values,
            profile.bootstrap_resamples,
            _hash_seed(contract.namespace, "qualification", candidate_id, "latent-shuffle"),
            contract.strict_alpha,
        )
        latent_passed = bool(
            latent["mean"] is not None
            and float(latent["mean"]) >= contract.control_advantage
            and latent["ci"][0] is not None
            and float(latent["ci"][0]) > 0.0
        )
        summaries[candidate_id] = {
            "strict": strict,
            "latent_shuffle_advantage": latent,
            "latent_shuffle_gate": latent_passed,
        }
        if not gate_applied or (strict.get("renewed_gate") and latent_passed):
            qualified.append(candidate_id)
    return {
        "state": "complete",
        "scientific_gate_applied": gate_applied,
        "candidate_summaries": summaries,
        "qualified_candidate_ids": qualified,
    }


def _stress_metric(
    rows: Sequence[dict[str, Any]], candidate_id: str, scenario: str, generation: int, metric: str
) -> list[float]:
    values: list[float] = []
    for row in rows:
        try:
            outcome = row["candidates"][candidate_id][scenario]["outcomes"][str(generation)]
            value = outcome["survival"] if metric == "survival" else outcome["primary"][metric]
        except KeyError:
            continue
        values.append(float(value))
    return values


def summarize_stress(
    rows: Sequence[dict[str, Any]],
    candidate_ids: Sequence[str],
    profile: CompressionProfile,
    contract: CompressionContract,
    complete: bool,
) -> dict[str, Any]:
    if not complete:
        return {"state": "incomplete", "candidates": {}}
    generation = 16 if profile.stress_generations >= 16 else profile.stress_generations
    summaries: dict[str, Any] = {}
    for candidate_id in candidate_ids:
        scenarios: dict[str, Any] = {}
        for scenario in stress_scenarios():
            values = _stress_metric(rows, candidate_id, scenario, generation, "crossover")
            scenarios[scenario] = {
                "crossover": _bootstrap(
                    values,
                    profile.bootstrap_resamples,
                    _hash_seed(contract.namespace, "stress", candidate_id, scenario),
                    contract.strict_alpha,
                ),
                "survival_mean": float(np.mean(_stress_metric(rows, candidate_id, scenario, generation, "survival"))),
            }
        summaries[candidate_id] = {
            "scenarios": scenarios,
            "worst_case_crossover": min(
                float(value["crossover"]["mean"])
                if value["crossover"]["mean"] is not None
                else -1.0
                for value in scenarios.values()
            ),
            "ordinary_crossover": float(scenarios["ordinary"]["crossover"]["mean"] or 0.0),
            "moderate_joint_crossover": float(scenarios["moderate_joint"]["crossover"]["mean"] or 0.0),
        }
    return {"state": "complete", "generation": generation, "candidates": summaries}


def _qualification_effect(rows: Sequence[dict[str, Any]], candidate_id: str, generation: int) -> float:
    values: list[float] = []
    for row in rows:
        try:
            value = row["candidates"][candidate_id]["conditions"]["intact"]["outcomes"][str(generation)]["primary"]["crossover"]
        except KeyError:
            continue
        values.append(float(value))
    return float(np.mean(values)) if values else 0.0


def select_finalists(
    models: Sequence[dict[str, Any]],
    qualified_ids: Sequence[str],
    qualification_rows: Sequence[dict[str, Any]],
    stress: dict[str, Any],
    profile: CompressionProfile,
    contract: CompressionContract,
) -> list[str]:
    by_id = {str(model["candidate_id"]): model for model in models}
    candidates = [by_id[candidate_id] for candidate_id in qualified_ids if candidate_id != ANCHOR_ID]
    if not candidates:
        return []
    generation = 16 if profile.qualification_generations >= 16 else profile.qualification_generations
    anchor_effect = _qualification_effect(qualification_rows, ANCHOR_ID, generation)
    retained = [
        model for model in candidates
        if anchor_effect <= 0.0
        or _qualification_effect(qualification_rows, model["candidate_id"], generation)
        >= contract.finalist_anchor_retention * anchor_effect
    ]
    if not retained:
        return []

    def key(model: dict[str, Any]) -> tuple[Any, ...]:
        stress_value = stress.get("candidates", {}).get(model["candidate_id"], {}).get("worst_case_crossover", -1.0)
        return (
            int(model["payload_bits"]),
            -float(stress_value),
            not bool(model["interpretable"]),
            str(model["candidate_id"]),
        )

    choices: list[dict[str, Any]] = [min(retained, key=key)]
    interpretable = [model for model in retained if model["interpretable"]]
    if interpretable:
        choices.append(min(interpretable, key=key))
    robust = [model for model in retained if int(model["rank"]) <= 128 and int(model["bits"]) <= 8]
    if robust:
        choices.append(
            min(
                robust,
                key=lambda model: (
                    -float(stress.get("candidates", {}).get(model["candidate_id"], {}).get("worst_case_crossover", -1.0)),
                    int(model["payload_bits"]),
                    not bool(model["interpretable"]),
                    str(model["candidate_id"]),
                ),
            )
        )
    return list(dict.fromkeys(str(model["candidate_id"]) for model in choices))[:3]


def summarize_transfer(
    rows: Sequence[dict[str, Any]],
    candidate_ids: Sequence[str],
    profile: CompressionProfile,
    contract: CompressionContract,
    complete: bool,
) -> dict[str, Any]:
    if not complete:
        return {"state": "incomplete", "rules": {}}
    generation = 16 if profile.transfer_generations >= 16 else profile.transfer_generations
    rules: dict[str, Any] = {}
    for rule in (31648, 70366):
        rule_rows = [row for row in rows if int(row["rule"]) == rule]
        candidates: dict[str, Any] = {}
        for candidate_id in candidate_ids:
            values = [
                float(row["candidates"][candidate_id]["outcomes"][str(generation)]["primary"]["crossover"])
                for row in rule_rows
            ]
            candidates[candidate_id] = {
                "crossover": _bootstrap(
                    values,
                    profile.bootstrap_resamples,
                    _hash_seed(contract.namespace, "transfer", rule, candidate_id),
                    contract.strict_alpha,
                )
            }
        rules[str(rule)] = {"pairs": len(rule_rows), "candidates": candidates}
    return {"state": "complete", "generation": generation, "rules": rules, "exploratory_only": True}


def adjudicate_confirmation(
    rows: Sequence[dict[str, Any]],
    models: Sequence[dict[str, Any]],
    profile: CompressionProfile,
    contract: CompressionContract,
    complete: bool,
) -> dict[str, Any]:
    if not complete or profile.confirmation_generations < 16:
        return {"state": "incomplete", "verdict": "INCOMPLETE", "candidates": {}}
    candidates: dict[str, Any] = {}
    passed: dict[str, dict[str, bool]] = {}
    for model in models:
        candidate_id = str(model["candidate_id"])
        candidate: dict[str, Any] = {"model": _json_model(model), "environments": {}}
        passed[candidate_id] = {}
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
                if candidate_id in row["candidates"]
                and environment in row["candidates"][candidate_id]["environments"]
            ]
            strict = _strict_confirmation_gate(
                transformed,
                candidate_id,
                _repair_profile(profile, f"stage4-confirm-{environment}", confirmation=True),
                contract,  # type: ignore[arg-type]
                contract.confirmation_alpha_per_codec,
            )
            latent_values = _qualification_difference(
                transformed, candidate_id, "latent_shuffle_every_boundary", 8
            )
            latent = _bootstrap(
                latent_values,
                profile.bootstrap_resamples,
                _hash_seed(contract.namespace, "confirmation", candidate_id, environment, "latent-shuffle"),
                contract.confirmation_alpha_per_codec,
            )
            latent_passed = bool(
                latent["mean"] is not None
                and float(latent["mean"]) >= contract.control_advantage
                and latent["ci"][0] is not None
                and float(latent["ci"][0]) > 0.0
            )
            strict["latent_shuffle_advantage"] = latent
            strict["latent_shuffle_gate"] = latent_passed
            strict["stage4_renewed_gate"] = bool(strict["renewed_gate"] and latent_passed)
            candidate["environments"][environment] = {"strict": strict}
            passed[candidate_id][environment] = bool(strict["stage4_renewed_gate"])
        candidates[candidate_id] = candidate

    by_id = {str(model["candidate_id"]): model for model in models}
    anchor_pass = passed.get(ANCHOR_ID, {}).get("ordinary", False)
    compact_ordinary = [
        candidate_id for candidate_id, value in passed.items()
        if candidate_id != ANCHOR_ID and value["ordinary"]
        and int(by_id[candidate_id]["rank"]) <= 64 and int(by_id[candidate_id]["bits"]) <= 8
    ]
    compact_robust = [candidate_id for candidate_id in compact_ordinary if passed[candidate_id]["moderate_joint"]]
    compressed = [
        candidate_id for candidate_id, value in passed.items()
        if candidate_id != ANCHOR_ID and value["ordinary"]
        and int(by_id[candidate_id]["rank"]) <= 128 and int(by_id[candidate_id]["bits"]) <= 8
    ]
    any_compressed = [candidate_id for candidate_id, value in passed.items() if candidate_id != ANCHOR_ID and value["ordinary"]]
    verdict = (
        "NO_FRESH_STAGE3R_REPLICATION"
        if not anchor_pass
        else "ROBUST_COMPACT_RENEWED_CA_PLASTIC_HEREDITY"
        if compact_robust
        else "COMPACT_RENEWED_CA_PLASTIC_HEREDITY"
        if compact_ordinary
        else "COMPRESSED_RENEWED_CA_PLASTIC_HEREDITY"
        if compressed
        else "HIGH_DIMENSIONAL_RENEWED_CA_PLASTIC_HEREDITY"
        if any_compressed
        else "FULL_CARRIER_ONLY"
    )
    return {
        "state": "complete",
        "verdict": verdict,
        "fresh_anchor_replicated": anchor_pass,
        "robust_compact_candidate_ids": compact_robust,
        "compact_candidate_ids": compact_ordinary,
        "compressed_candidate_ids": compressed,
        "candidates": candidates,
        "claim_boundary": "synthetic CA lineage memory only; no metabolism, agency, or biological-life claim",
    }


def _phase_rows(output: Path, phase: str, design_digest: str) -> list[dict[str, Any]]:
    summary = _load_json(output / phase / "stage_summary.json")
    if summary.get("design_digest") != design_digest or not summary.get("complete"):
        raise ValueError(f"{phase} is absent, incomplete, or belongs to another design")
    rows: list[dict[str, Any]] = []
    for path in sorted((output / phase / "checkpoints").glob("*.json")):
        payload = _load_json(path)
        if payload.get("design_digest") != design_digest:
            raise ValueError(f"{phase} checkpoint design mismatch: {path}")
        rows.append(payload["result"])
    return rows


def _configuration_payload(configuration: ReaderConfiguration) -> dict[str, Any]:
    return {
        "family": configuration.family,
        "write_window": configuration.write_window,
        "strength": configuration.strength,
        "read_duration": configuration.read_duration,
    }


def _pareto_frontier(
    models: Sequence[dict[str, Any]],
    candidate_ids: Sequence[str],
    qualification_rows: Sequence[dict[str, Any]],
    stress: dict[str, Any],
    generations: int,
) -> list[dict[str, Any]]:
    generation = 16 if generations >= 16 else generations
    rows: list[dict[str, Any]] = []
    by_id = {str(model["candidate_id"]): model for model in models}
    for candidate_id in candidate_ids:
        model = by_id[candidate_id]
        rows.append(
            {
                "candidate_id": candidate_id,
                "payload_bits": int(model["payload_bits"]),
                "rank": int(model["rank"]),
                "bits": int(model["bits"]),
                "interpretable": bool(model["interpretable"]),
                "intact_crossover": _qualification_effect(qualification_rows, candidate_id, generation),
                "worst_case_crossover": float(
                    stress.get("candidates", {}).get(candidate_id, {}).get("worst_case_crossover", -1.0)
                ),
            }
        )
    frontier: list[dict[str, Any]] = []
    for row in rows:
        dominated = any(
            other["payload_bits"] <= row["payload_bits"]
            and other["intact_crossover"] >= row["intact_crossover"]
            and other["worst_case_crossover"] >= row["worst_case_crossover"]
            and (
                other["payload_bits"] < row["payload_bits"]
                or other["intact_crossover"] > row["intact_crossover"]
                or other["worst_case_crossover"] > row["worst_case_crossover"]
            )
            for other in rows
        )
        if not dominated:
            frontier.append(row)
    return sorted(frontier, key=lambda row: (row["payload_bits"], -row["intact_crossover"], row["candidate_id"]))


def _queue(
    design_digest: str,
    state: str,
    *,
    finalists: Sequence[str] = (),
    verdict: str | None = None,
) -> dict[str, Any]:
    stages: list[dict[str, Any]] = [
        {"stage": 1, "name": "motif_carrier_upper_bound", "state": "complete"},
        {"stage": 2, "name": "freeze_and_generalize_reader", "state": "complete"},
        {"stage": 3, "name": "renewed_heredity_causal_ladder", "state": "complete_negative"},
        {"stage": "3R", "name": "semantic_closure_and_repair", "state": "complete_positive"},
        {
            "stage": 4,
            "name": "compression_and_robustness",
            "state": state,
            "finalist_candidate_ids": list(finalists),
        },
        {"stage": 5, "name": "localize_inheritance", "state": "blocked_pending_stage4_review"},
    ]
    if verdict is not None:
        stages[4]["verdict"] = verdict
        if verdict != "NO_FRESH_STAGE3R_REPLICATION":
            stages[5]["state"] = "blocked_pending_human_review"
    return {
        "design_digest": design_digest,
        "automatic_launch": False,
        "stages": stages,
    }


def _render_preconfirmation_report(results: dict[str, Any]) -> str:
    decision = results["selection_decision"]
    frontier = results.get("pareto_frontier", [])
    lines = [
        "# CA motif-lineage Stage 4: preconfirmation report",
        "",
        f"State: `{results['state']}`.",
        "",
        "The 512-value Stage-3R carrier was challenged with 50 label-blind codecs, a full causal qualification ladder, registered damage curves, and an exploratory two-rule transfer panel. Confirmation has not been opened.",
        "",
        f"Screen-selected codecs: {len(results['screen'].get('selected_candidate_ids', []))}.",
        f"Strictly qualified codecs: {len(results['qualification'].get('qualified_candidate_ids', []))}.",
        f"Frozen compressed finalists: {', '.join(decision['finalist_candidate_ids']) or 'none'}.",
        f"Confirmation state: `{decision['confirmation_state']}`.",
        "",
        "## Pareto frontier",
        "",
        "| Codec | Payload bits | Values | Bits/value | Intact G16 | Worst stress |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in frontier:
        lines.append(
            f"| `{row['candidate_id']}` | {row['payload_bits']} | {row['rank']} | {row['bits']} | {row['intact_crossover']:.3f} | {row['worst_case_crossover']:.3f} |"
        )
    lines.extend(
        (
            "",
            "This is a synthetic CA lineage-memory test, not a claim about biological life, agency, or nonphysical inheritance.",
            "",
        )
    )
    return "\n".join(lines)


def _render_preconfirmation_lay(results: dict[str, Any]) -> str:
    finalists = results["selection_decision"]["finalist_candidate_ids"]
    return "\n".join(
        (
            "# Stage 4 in plain language",
            "",
            "We treated the hidden 512-number texture memory like a message and asked how aggressively it could be shortened or coarsened while still helping a freshly reset cellular automaton rebuild its inherited form.",
            "",
            f"The engineering rounds have frozen {len(finalists)} compressed finalist(s): {', '.join(finalists) or 'none'}. They were chosen using fresh selection cases, carrier ablations and rescues, deliberate copying damage, and rule-transfer probes. The final sealed cases have not yet been run, so this is a shortlist rather than the final scientific answer.",
            "",
            "The decisive next step requires explicit review and a separate confirmation command. Stage 5 remains blocked.",
            "",
        )
    )


def _render_final_report(results: dict[str, Any]) -> str:
    adjudication = results["adjudication"]
    lines = [
        "# CA motif-lineage Stage 4: final report",
        "",
        f"Verdict: `{adjudication['verdict']}`.",
        f"Fresh full-carrier replication: `{adjudication['fresh_anchor_replicated']}`.",
        "",
        "| Codec | Ordinary strict gate | Moderate-joint strict gate |",
        "|---|---:|---:|",
    ]
    for candidate_id, value in adjudication["candidates"].items():
        ordinary = value["environments"]["ordinary"]["strict"]["stage4_renewed_gate"]
        moderate = value["environments"]["moderate_joint"]["strict"]["stage4_renewed_gate"]
        lines.append(f"| `{candidate_id}` | {ordinary} | {moderate} |")
    lines.extend(
        (
            "",
            "Only the latent payload crossed boundaries; the visible lattice was reset before every generation. This remains a synthetic CA result and does not establish biological life or agency.",
            "",
        )
    )
    return "\n".join(lines)


def _render_final_lay(results: dict[str, Any]) -> str:
    adjudication = results["adjudication"]
    return "\n".join(
        (
            "# Stage 4 final result in plain language",
            "",
            f"The final verdict is `{adjudication['verdict']}`.",
            "",
            "We reset the visible cellular automaton at every generation and let only a compressed hidden message pass from parent to daughter. The test then asked whether the right ancestral form returned, whether deleting or scrambling that message destroyed the effect, whether putting the right message back rescued it, and whether the message survived moderate copying damage.",
            "",
            "This tells us how compact and robust this engineered form of Plastic Heredity is. It does not by itself show metabolism, consciousness, agency, or the origin of life.",
            "",
        )
    )


def _update_discovery_log(state: str, verdict: str, elapsed_seconds: float) -> None:
    path = ROOT / "DISCOVERY_LOG_EIDOSOMA_SCIENTIST.md"
    existing = path.read_text(encoding="utf-8") if path.exists() else "# Discovery log\n"
    start = "<!-- STAGE4_COMPRESSION_START -->"
    end = "<!-- STAGE4_COMPRESSION_END -->"
    section = "\n".join(
        (
            start,
            "## CA Stage 4 — compressed texture heredity",
            "",
            f"State: `{state}`; verdict: `{verdict}`.",
            f"Elapsed `{elapsed_seconds / 3600.0:.3f}` wall hours.",
            "See `results/ca-motif-lineage-stage-4/REPORT.md` and `LAY_SUMMARY.md`.",
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


def run_motif_compression(
    output: Path,
    *,
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
        raise ValueError(f"unknown Stage-4 profile {profile_name!r}")
    if max_hours <= 0.0 or max_hours > 8.0:
        raise ValueError("Stage-4 max-hours must be in (0, 8]")
    selected_phases = tuple(phases or DEFAULT_PRECONFIRMATION_PHASES)
    unknown = [phase for phase in selected_phases if phase not in PHASES]
    if unknown:
        raise ValueError(f"unknown Stage-4 phases: {unknown}")
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
    contract = CompressionContract()
    writer_contract = MotifContract()
    profile = COMPRESSION_PROFILES[profile_name]
    reserve = min(contract.science_reserve_seconds, max(60.0, max_hours * 3600.0 * 0.10))
    science_deadline = max(started, hard_deadline - reserve)

    def status(state: str, phase: str, **extra: Any) -> None:
        now = time.time()
        payload = {
            "state": state,
            "stage": "4-compression-and-robustness",
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
        frozen = load_frozen_stage3r(stage3r_root, stage3_root, stage2_root, stage1_root)
        cohorts = select_compression_cohorts(profile, frozen, contract, profile_name=profile_name)
        trace_hashes = {str(path): _sha256(path) for path in frozen["trace_paths"]}
        fit_trace_digest = hashlib.sha256(
            json.dumps(trace_hashes, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        configuration = frozen["configuration"]
        design_payload = {
            "experiment": "ca_motif_lineage_stage_4",
            "contract": contract.to_dict(),
            "writer_contract_digest": writer_contract.digest,
            "profile_name": profile_name,
            "profile": asdict(profile),
            "configuration": configuration.to_dict(),
            "stage3r_design_digest": frozen["design_digest"],
            "phases_contract": PHASES,
            "confirmation_separate_invocation": True,
            "codec_families": ("identity", "pca", "walsh", "sparse", "random", "d4", "count18"),
            "ranks": RANKS,
            "stress_scenarios": stress_scenarios(),
            "selection_pair_ids": [pair["pair_id"] for pair in cohorts["selection"]],
            "stress_pair_ids": [pair["pair_id"] for pair in cohorts["stress"]],
            "confirmation_pair_ids": [pair["pair_id"] for pair in cohorts["confirmation"]],
            "stage5_reserve_pair_ids": [pair["pair_id"] for pair in cohorts["reserve"]],
            "fit_trace_digest": fit_trace_digest,
            "input_sha256": {
                "protocol": _sha256(PROTOCOL_PATH),
                **{f"stage3r_{key}": _sha256(path) for key, path in frozen["paths"].items()},
            },
            "implementation_sha256": {
                "motif_compression.py": _sha256(Path(__file__)),
                "motif_repair.py": _sha256(Path(__file__).with_name("motif_repair.py")),
                "motif_lineage.py": _sha256(Path(__file__).with_name("motif_lineage.py")),
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
                raise ValueError("Stage-4 resume design digest mismatch")
        elif "confirm" in selected_phases:
            raise FileNotFoundError("confirmation requires a reviewed Stage-4 design")
        _atomic_json(design_path, design)
        _atomic_json(
            output / "MANIFEST.json",
            {
                "experiment": "ca_motif_lineage_stage_4",
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
        cohort_payload = {
            "design_digest": design_digest,
            "prior_rule31649_pair_ids_excluded": sorted(frozen["used_pair_ids"]),
            "selection_pair_ids": [pair["pair_id"] for pair in cohorts["selection"]],
            "stress_pair_ids": [pair["pair_id"] for pair in cohorts["stress"]],
            "confirmation_pair_ids": [pair["pair_id"] for pair in cohorts["confirmation"]],
            "stage5_reserve_pair_ids": [pair["pair_id"] for pair in cohorts["reserve"]],
            "transfer_pair_ids_by_rule": {
                str(rule): [pair["pair_id"] for pair in cohorts["transfer"] if pair["stage4_transfer_rule"] == rule]
                for rule in (31648, 70366)
            },
            "confirmation_trajectory_state": "untouched",
        }
        if "confirm" not in selected_phases:
            _atomic_json(output / "COHORTS.json", cohort_payload)
        _atomic_json(output / "QUEUE.json", _queue(design_digest, "running"))

        audit = {
            "design_digest": design_digest,
            "state": "passed",
            "stage3r_winner": "simple--strict-49-64--gain-050",
            "stage3r_strict_gate": True,
            "prior_rule31649_pairs": len(frozen["used_pair_ids"]),
            "unused_before_stage4": len(frozen["all_pairs"]) - len(frozen["used_pair_ids"]),
            "selection_pairs": len(cohorts["selection"]),
            "confirmation_pairs": len(cohorts["confirmation"]),
            "stage5_reserve_pairs": len(cohorts["reserve"]),
            "cleanroom_exclusion_upheld": True,
            "confirmation_not_opened": "confirm" not in selected_phases,
        }
        if profile_name == "reference" and (
            audit["unused_before_stage4"] != 382 or audit["stage5_reserve_pairs"] != 158
        ):
            raise AssertionError("the registered Stage-4 pair ledger changed")
        if "audit" in selected_phases:
            _atomic_json(output / "CLEANROOM_AUDIT.json", audit)

        downstream = {"fit", "screen", "qualify", "stress", "transfer", "adjudicate", "confirm"}
        if not downstream.intersection(selected_phases):
            status("phases_complete", "campaign")
            return {"state": "phases_complete", "completed_phases": selected_phases}

        if "fit" in selected_phases:
            status("running", "fit")
            fit_matrix, groups, verified_hashes = load_stage3r_fit_matrix(frozen)
            if verified_hashes != {str(Path(path).relative_to(ROOT)): value for path, value in trace_hashes.items()}:
                raise ValueError("fit trace changed during Stage-4 startup")
            models, fit_audit = fit_codec_atlas(fit_matrix)
            fit_audit.update(
                {
                    "design_digest": design_digest,
                    "fit_trace_digest": fit_trace_digest,
                    "pair_groups": len(set(groups)),
                    "trace_sha256": verified_hashes,
                }
            )
            _atomic_json(output / "FIT_AUDIT.json", fit_audit)
            save_codec_models(
                output, models, design_digest=design_digest, fit_trace_digest=fit_trace_digest
            )
        else:
            models = load_codec_models(output, design_digest)

        configuration_payload = _configuration_payload(configuration)

        if "confirm" in selected_phases:
            decision = _load_json(output / "SELECTION_DECISION.json")
            confirmation_design = _load_json(output / "CONFIRMATION_DESIGN.json")
            if decision.get("design_digest") != design_digest or confirmation_design.get("design_digest") != design_digest:
                raise ValueError("confirmation decision belongs to another Stage-4 design")
            if decision.get("confirmation_state") != "awaiting_human_review":
                raise ValueError("Stage-4 confirmation is not awaiting review")
            candidate_ids = [ANCHOR_ID, *decision["finalist_candidate_ids"]]
            if candidate_ids != confirmation_design.get("candidate_ids"):
                raise ValueError("confirmation candidate list changed after review")
            if confirmation_design.get("model_sha256") != _sha256(output / "CODEC_MODELS.npz"):
                raise ValueError("confirmation codec archive changed after review")
            confirmation_models = _selected_models(models, candidate_ids)
            items = [
                {
                    "checkpoint": f"confirm-{pair_index:04d}-codec-{model_index:02d}-env-{environment}",
                    "pair": pair,
                    "candidate_id": model["candidate_id"],
                    "environment": environment,
                    "replicates": profile.confirmation_replicates,
                    "generations": profile.confirmation_generations,
                    "configuration": configuration_payload,
                }
                for pair_index, pair in enumerate(cohorts["confirmation"])
                for model_index, model in enumerate(confirmation_models)
                for environment in ("ordinary", "moderate_joint")
            ]
            status("running", "confirmation", completed=0, total=len(items))
            rows, complete = _run_json_checkpoints(
                output, "confirmation", items, confirmation_models, _confirmation_pair_task,
                writer_contract, contract, frozen["reference"], design_digest,
                workers=workers, resume=resume, deadline=science_deadline, status=status
            )
            adjudication = adjudicate_confirmation(rows, confirmation_models, profile, contract, complete)
            state = "complete" if complete else "partial_budget_exhausted"
            results = {
                "experiment": "ca_motif_lineage_stage_4",
                "state": state,
                "profile": profile_name,
                "design_digest": design_digest,
                "stage3r_design_digest": frozen["design_digest"],
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
                    "decision": "stage5_may_be_planned_after_review" if adjudication.get("fresh_anchor_replicated") else "halt_stage5_no_fresh_replication",
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
                _queue(design_digest, state, finalists=decision["finalist_candidate_ids"], verdict=adjudication["verdict"]),
            )
            status(state, "campaign", verdict=adjudication["verdict"])
            return results

        if "screen" in selected_phases:
            items = [
                {
                    "checkpoint": f"screen-{index:04d}", "pair": pair,
                    "replicates": profile.screen_replicates,
                    "generations": profile.screen_generations,
                    "configuration": configuration_payload,
                }
                for index, pair in enumerate(cohorts["selection"])
            ]
            status("running", "screen", completed=0, total=len(items))
            screen_rows, complete = _run_json_checkpoints(
                output, "screen", items, models, _screen_pair_task, writer_contract,
                contract, frozen["reference"], design_digest, workers=workers,
                resume=resume, deadline=science_deadline, status=status
            )
            screen = adjudicate_screen(screen_rows, models, profile, contract, complete)
            screen["design_digest"] = design_digest
            _atomic_json(output / "SCREEN.json", screen)
            if not complete:
                status("partial_budget_exhausted", "screen")
                return {"state": "partial_budget_exhausted", "phase": "screen"}
        else:
            screen = _load_json(output / "SCREEN.json")
            screen_rows = _phase_rows(output, "screen", design_digest)
        selected_ids = list(screen["selected_candidate_ids"])
        selected_models = _selected_models(models, selected_ids)

        if "qualify" in selected_phases:
            items = [
                {
                    "checkpoint": f"qualify-{pair_index:04d}-codec-{model_index:02d}",
                    "pair": pair,
                    "candidate_id": model["candidate_id"],
                    "replicates": profile.qualification_replicates,
                    "generations": profile.qualification_generations,
                    "configuration": configuration_payload,
                }
                for pair_index, pair in enumerate(cohorts["selection"])
                for model_index, model in enumerate(selected_models)
            ]
            status("running", "qualification", completed=0, total=len(items))
            qualification_rows, complete = _run_json_checkpoints(
                output, "qualification", items, selected_models, _qualification_pair_task,
                writer_contract, contract, frozen["reference"], design_digest,
                workers=workers, resume=resume, deadline=science_deadline, status=status
            )
            qualification = adjudicate_qualification(
                qualification_rows, selected_ids, profile, contract, complete
            )
            qualification["design_digest"] = design_digest
            _atomic_json(output / "QUALIFICATION.json", qualification)
            if not complete:
                status("partial_budget_exhausted", "qualification")
                return {"state": "partial_budget_exhausted", "phase": "qualification"}
        else:
            qualification = _load_json(output / "QUALIFICATION.json")
            qualification_rows = _phase_rows(output, "qualification", design_digest)
        qualified_ids = list(qualification["qualified_candidate_ids"])
        qualified_models = _selected_models(models, qualified_ids)

        if "stress" in selected_phases:
            items = [
                {
                    "checkpoint": f"stress-{pair_index:04d}-codec-{model_index:02d}",
                    "pair": pair,
                    "candidate_id": model["candidate_id"],
                    "replicates": profile.stress_replicates,
                    "generations": profile.stress_generations,
                    "configuration": configuration_payload,
                }
                for pair_index, pair in enumerate(cohorts["stress"])
                for model_index, model in enumerate(qualified_models)
            ]
            status("running", "robustness", completed=0, total=len(items))
            stress_rows, complete = _run_json_checkpoints(
                output, "robustness", items, qualified_models, _stress_pair_task,
                writer_contract, contract, frozen["reference"], design_digest,
                workers=workers, resume=resume, deadline=science_deadline, status=status
            )
            robustness = summarize_stress(stress_rows, qualified_ids, profile, contract, complete)
            robustness["design_digest"] = design_digest
            _atomic_json(output / "ROBUSTNESS.json", robustness)
            if not complete:
                status("partial_budget_exhausted", "robustness")
                return {"state": "partial_budget_exhausted", "phase": "robustness"}
        else:
            robustness = _load_json(output / "ROBUSTNESS.json")
            stress_rows = _phase_rows(output, "robustness", design_digest)

        finalists = select_finalists(
            models, qualified_ids, qualification_rows, robustness, profile, contract
        )
        transfer_ids = list(dict.fromkeys((ANCHOR_ID, *finalists)))
        transfer_models = _selected_models(models, transfer_ids)
        if "transfer" in selected_phases:
            items = [
                {
                    "checkpoint": f"transfer-{index:04d}", "pair": pair,
                    "replicates": profile.transfer_replicates,
                    "generations": profile.transfer_generations,
                    "configuration": configuration_payload,
                }
                for index, pair in enumerate(cohorts["transfer"])
            ]
            status("running", "transfer", completed=0, total=len(items))
            transfer_rows, complete = _run_json_checkpoints(
                output, "transfer", items, transfer_models, _transfer_pair_task,
                writer_contract, contract, frozen["reference"], design_digest,
                workers=workers, resume=resume, deadline=science_deadline, status=status
            )
            transfer = summarize_transfer(transfer_rows, transfer_ids, profile, contract, complete)
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

        frontier = _pareto_frontier(
            models, qualified_ids, qualification_rows, robustness,
            profile.qualification_generations
        )
        confirmation_state = "awaiting_human_review"
        decision = {
            "design_digest": design_digest,
            "stage3r_design_digest": frozen["design_digest"],
            "anchor_candidate_id": ANCHOR_ID,
            "screen_selected_candidate_ids": selected_ids,
            "strictly_qualified_candidate_ids": qualified_ids,
            "finalist_candidate_ids": finalists,
            "confirmation_state": confirmation_state,
            "confirmation_requires_separate_invocation": True,
            "automatic_launch": False,
            "review_required": True,
        }
        _atomic_json(output / "SELECTION_DECISION.json", decision)
        confirmation_ids = [ANCHOR_ID, *finalists]
        confirmation_design = {
            "design_digest": design_digest,
            "protocol_sha256": _sha256(PROTOCOL_PATH),
            "model_sha256": _sha256(output / "CODEC_MODELS.npz"),
            "candidate_ids": confirmation_ids,
            "candidate_models": [_json_model(model) for model in _selected_models(models, confirmation_ids)],
            "confirmation_pair_ids": [pair["pair_id"] for pair in cohorts["confirmation"]],
            "replicates": profile.confirmation_replicates,
            "generations": profile.confirmation_generations,
            "environments": ("ordinary", "moderate_joint"),
            "alpha_per_codec": contract.confirmation_alpha_per_codec,
            "trajectory_state": "untouched",
            "authorization_required": True,
        }
        confirmation_design["confirmation_design_digest"] = hashlib.sha256(
            json.dumps(confirmation_design, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        _atomic_json(output / "CONFIRMATION_DESIGN.json", confirmation_design)
        results = {
            "experiment": "ca_motif_lineage_stage_4_preconfirmation",
            "state": confirmation_state,
            "profile": profile_name,
            "design_digest": design_digest,
            "stage3r_design_digest": frozen["design_digest"],
            "elapsed_seconds": time.time() - started,
            "screen": screen,
            "qualification": qualification,
            "robustness": robustness,
            "transfer": transfer,
            "pareto_frontier": frontier,
            "selection_decision": decision,
        }
        _atomic_json(output / "PRECONFIRMATION_RESULTS.json", results)
        report = _render_preconfirmation_report(results)
        lay = _render_preconfirmation_lay(results)
        _atomic_text(output / "PRECONFIRMATION_REPORT.md", report)
        _atomic_text(output / "PRECONFIRMATION_LAY_SUMMARY.md", lay)
        _atomic_text(output / "REPORT.md", report)
        _atomic_text(output / "LAY_SUMMARY.md", lay)
        _atomic_json(output / "QUEUE.json", _queue(design_digest, confirmation_state, finalists=finalists))
        _atomic_text(output / "PRECONFIRMATION_COMPLETE", "complete\n")
        if profile_name == "reference":
            _update_discovery_log(confirmation_state, "PRECONFIRMATION", results["elapsed_seconds"])
        status(confirmation_state, "campaign", finalists=len(finalists))
        return results
    except BaseException as error:
        status("failed", "campaign", error=repr(error))
        raise


__all__ = [
    "ANCHOR_ID",
    "CAUSAL_CONDITIONS",
    "COMPRESSION_PROFILES",
    "CompressionContract",
    "CompressionProfile",
    "DEFAULT_PRECONFIRMATION_PHASES",
    "PHASES",
    "PUBLIC_PROFILES",
    "adjudicate_confirmation",
    "decode_payload",
    "encode_payload",
    "fit_codec_atlas",
    "load_codec_models",
    "load_frozen_stage3r",
    "quantize_payload",
    "run_motif_compression",
    "save_codec_models",
    "select_compression_cohorts",
    "select_finalists",
    "simulate_compressed_lineage",
    "stress_scenarios",
]
