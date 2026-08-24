from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from reviewer_ca_compact_carrier_replication.codec import load_codecs, validate_codecs
from reviewer_ca_compact_carrier_replication.contract import (
    CANDIDATE_IDS,
    CONDITIONS,
    DEFAULT_ARTIFACTS,
    SCHEMA_VERSION,
    atomic_write_json,
    read_checkpoint,
    seal_registration,
    verify_registration,
    write_checkpoint,
)
from reviewer_ca_compact_carrier_replication.snapshot import (
    LOCAL_ALLOWLIST,
    SOURCE_ALLOWLIST,
    verify_snapshot,
)


def test_snapshot_adapter_is_data_and_documents_only() -> None:
    paths = [*LOCAL_ALLOWLIST, *SOURCE_ALLOWLIST]
    assert paths
    assert all(Path(path).suffix.lower() in {".json", ".md", ".npz"} for path in paths)
    assert not any(Path(path).suffix.lower() == ".py" for path in paths)
    manifest = verify_snapshot(DEFAULT_ARTIFACTS / "input")
    assert manifest["source_results_or_checkpoints_imported"] is False
    assert manifest["source_code_opened_imported_hashed_or_executed"] is False


def test_exact_candidate_and_causal_panels_are_frozen() -> None:
    assert CANDIDATE_IDS == (
        "identity-r512-f32",
        "pca-r008-q04",
        "walsh-r016-q04",
    )
    assert len(CONDITIONS) == 12
    assert "decoded_shuffle_every_boundary" in CONDITIONS
    assert "latent_shuffle_every_boundary" in CONDITIONS


def test_frozen_codecs_validate_and_preserve_exact_zero() -> None:
    report = validate_codecs(DEFAULT_ARTIFACTS / "input")
    assert report["valid"]
    codecs = load_codecs(DEFAULT_ARTIFACTS / "input")
    assert tuple(codecs) == CANDIDATE_IDS
    zero = np.zeros((2, 3, 512), dtype=np.float32)
    for codec in codecs.values():
        payload = codec.encode(zero)
        assert not np.any(payload)
        assert not np.any(codec.decode(payload))


def test_identity_is_bitwise_and_four_bit_quantizer_is_idempotent() -> None:
    codecs = load_codecs(DEFAULT_ARTIFACTS / "input")
    rng = np.random.default_rng(9182)
    carrier = rng.normal(size=(4, 512)).astype(np.float32)
    identity = codecs["identity-r512-f32"]
    np.testing.assert_array_equal(identity.decode(identity.encode(carrier)), carrier)
    for candidate_id in ("pca-r008-q04", "walsh-r016-q04"):
        codec = codecs[candidate_id]
        first = codec.encode(carrier)
        assert first.dtype == np.int8
        assert int(first.min()) >= -7 and int(first.max()) <= 7
        second = codec.encode(codec.decode(first))
        np.testing.assert_array_equal(second, first)


def test_quantizer_rounds_to_registered_formula() -> None:
    codec = load_codecs(DEFAULT_ARTIFACTS / "input")["walsh-r016-q04"]
    assert codec.basis is not None and codec.scale is not None
    coefficients = np.linspace(-1.2, 1.2, codec.rank, dtype=np.float32) * codec.scale
    carrier = coefficients @ codec.basis.T
    observed = codec.encode(carrier)
    expected = np.clip(np.rint(7.0 * coefficients / codec.scale), -7, 7).astype(np.int8)
    np.testing.assert_array_equal(observed, expected)


def test_registration_and_checkpoint_tamper_detection(tmp_path: Path) -> None:
    registration = seal_registration(
        {"schema_version": SCHEMA_VERSION, "experiment": "fixture"}
    )
    verify_registration(registration)
    with pytest.raises(ValueError, match="digest"):
        verify_registration({**registration, "experiment": "changed"})
    checkpoint = tmp_path / "checkpoint.json"
    write_checkpoint(checkpoint, registration["design_digest"], {"value": 3})
    assert read_checkpoint(checkpoint, registration["design_digest"])["value"] == 3
    checkpoint.write_text(checkpoint.read_text().replace('"value": 3', '"value": 4'))
    with pytest.raises(ValueError, match="checksum"):
        read_checkpoint(checkpoint, registration["design_digest"])
    with pytest.raises(ValueError, match="nonfinite"):
        atomic_write_json(tmp_path / "bad.json", {"x": float("nan")})
