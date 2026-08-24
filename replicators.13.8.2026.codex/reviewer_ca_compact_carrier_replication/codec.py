"""Frozen compact carrier codecs and their validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contract import CANDIDATE_IDS, load_json, sha256_bytes, sha256_file


@dataclass(frozen=True)
class Codec:
    candidate_id: str
    family: str
    rank: int
    bits: int
    precision: str
    payload_bits: int
    codebook_bits: int
    basis: np.ndarray | None = None
    scale: np.ndarray | None = None

    @property
    def latent_dtype(self) -> np.dtype[Any]:
        return np.dtype(np.float32 if self.precision == "float32" else np.int8)

    def encode(self, carrier: np.ndarray) -> np.ndarray:
        value = np.asarray(carrier, dtype=np.float32)
        if value.shape[-1] != 512 or not np.isfinite(value).all():
            raise ValueError("carrier must end in 512 finite coordinates")
        if self.family == "identity":
            return value.copy()
        assert self.basis is not None and self.scale is not None
        coefficients = np.matmul(value, self.basis)
        ratio = np.divide(
            7.0 * coefficients,
            self.scale,
            out=np.zeros_like(coefficients, dtype=np.float32),
            where=self.scale != 0,
        )
        return np.clip(np.rint(ratio), -7, 7).astype(np.int8)

    def decode(self, payload: np.ndarray) -> np.ndarray:
        value = np.asarray(payload)
        if value.shape[-1] != self.rank:
            raise ValueError(f"payload for {self.candidate_id} must have rank {self.rank}")
        if self.family == "identity":
            result = value.astype(np.float32, copy=True)
        else:
            assert self.basis is not None and self.scale is not None
            coefficients = value.astype(np.float32) * (self.scale / np.float32(7.0))
            result = np.matmul(coefficients, self.basis.T).astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError("decoder produced a nonfinite carrier")
        return result

    def zero_payload(self, leading_shape: tuple[int, ...]) -> np.ndarray:
        return np.zeros((*leading_shape, self.rank), dtype=self.latent_dtype)

    def metadata(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "family": self.family,
            "rank": self.rank,
            "bits": self.bits,
            "precision": self.precision,
            "payload_bits": self.payload_bits,
            "codebook_bits": self.codebook_bits,
            "runtime_label_access": False,
            "runtime_parent_access": False,
            "runtime_target_access": False,
        }


def _selected_metadata(document: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    selected = {
        str(model["candidate_id"]): model
        for model in document["models"]
        if str(model["candidate_id"]) in CANDIDATE_IDS
    }
    if tuple(candidate for candidate in CANDIDATE_IDS if candidate in selected) != CANDIDATE_IDS:
        raise ValueError("codec metadata does not contain the three frozen candidates")
    return selected


def load_codecs(input_root: Path) -> dict[str, Codec]:
    json_path = input_root / "hypothesis/CODEC_MODELS.json"
    npz_path = input_root / "hypothesis/CODEC_MODELS.npz"
    document = load_json(json_path)
    if document.get("allow_pickle") is not False:
        raise ValueError("codec archive must prohibit pickle")
    if document.get("model_sha256") != sha256_file(npz_path):
        raise ValueError("codec archive SHA256 mismatch")
    metadata = _selected_metadata(document)
    codecs: dict[str, Codec] = {}
    with np.load(npz_path, allow_pickle=False) as arrays:
        for candidate_id in CANDIDATE_IDS:
            item = metadata[candidate_id]
            keys = item.get("array_keys", {})
            basis = None
            scale = None
            if item["family"] != "identity":
                basis = np.asarray(arrays[str(keys["basis"])], dtype=np.float32).copy()
                scale = np.asarray(
                    arrays[str(keys["quantizer_scale"])], dtype=np.float32
                ).copy()
            codecs[candidate_id] = Codec(
                candidate_id=candidate_id,
                family=str(item["family"]),
                rank=int(item["rank"]),
                bits=int(item["bits"]),
                precision=str(item["precision"]),
                payload_bits=int(item["payload_bits"]),
                codebook_bits=int(item["codebook_bits"]),
                basis=basis,
                scale=scale,
            )
    return codecs


def validate_codecs(input_root: Path) -> dict[str, Any]:
    codecs = load_codecs(input_root)
    audit: dict[str, Any] = {}
    for candidate_id, codec in codecs.items():
        zero = np.zeros((3, 512), dtype=np.float32)
        zero_payload = codec.encode(zero)
        zero_decoded = codec.decode(zero_payload)
        if np.any(zero_payload) or np.any(zero_decoded):
            raise ValueError(f"{candidate_id} fails exact-zero preservation")
        item: dict[str, Any] = {
            **codec.metadata(),
            "zero_preserved_exactly": True,
            "latent_dtype": str(codec.latent_dtype),
        }
        if codec.family == "identity":
            probe = np.linspace(-4.0, 4.0, 512, dtype=np.float32)
            if not np.array_equal(codec.decode(codec.encode(probe)), probe):
                raise ValueError("identity float32 codec is not bitwise exact")
            item["bitwise_identity"] = True
        else:
            assert codec.basis is not None and codec.scale is not None
            if codec.basis.shape != (512, codec.rank):
                raise ValueError(f"{candidate_id} basis shape mismatch")
            if codec.scale.shape != (codec.rank,):
                raise ValueError(f"{candidate_id} scale shape mismatch")
            if not np.isfinite(codec.basis).all() or not np.isfinite(codec.scale).all():
                raise ValueError(f"{candidate_id} has nonfinite machinery")
            if np.any(codec.scale < 0):
                raise ValueError(f"{candidate_id} has negative quantizer scales")
            gram = codec.basis.T.astype(np.float64) @ codec.basis.astype(np.float64)
            orthogonality_error = float(np.max(np.abs(gram - np.eye(codec.rank))))
            if orthogonality_error > 2e-5:
                raise ValueError(f"{candidate_id} basis is not orthonormal")
            item.update(
                {
                    "basis_shape": list(codec.basis.shape),
                    "scale_shape": list(codec.scale.shape),
                    "basis_sha256": sha256_bytes(codec.basis.tobytes(order="C")),
                    "scale_sha256": sha256_bytes(codec.scale.tobytes(order="C")),
                    "orthogonality_max_abs_error": orthogonality_error,
                    "scale_min": float(codec.scale.min()),
                    "scale_max": float(codec.scale.max()),
                }
            )
            if codec.family == "walsh":
                expected = 1.0 / np.sqrt(512.0)
                walsh_error = float(np.max(np.abs(np.abs(codec.basis) - expected)))
                if walsh_error > 2e-7:
                    raise ValueError("Walsh basis entries are not signed 1/sqrt(512)")
                item["walsh_entry_max_abs_error"] = walsh_error
        audit[candidate_id] = item
    return {
        "valid": True,
        "candidate_order": list(CANDIDATE_IDS),
        "quantizer": "q=clip(rint(7*coefficient/scale),-7,7); decode=q*scale/7",
        "codecs": audit,
    }
