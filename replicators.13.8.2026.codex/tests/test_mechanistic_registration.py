import json
from pathlib import Path

import pytest

from plastic_heredity.config import ExperimentConfig
from plastic_heredity.mechanistic import (
    MECHCONF_MASTER_SEED,
    _atomic_destination,
    verify_checksums,
    write_checksums,
)
from plastic_heredity.mechanistic_v2 import (
    MECHCONF2_MASTER_SEED,
    _confirmation_experiment,
)
from plastic_heredity.memory import MEMORY_CONFIRM_MASTER_SEED


def test_registration_checksum_detects_tampering(tmp_path: Path):
    artifact = tmp_path / "model.bin"
    artifact.write_bytes(b"registered")
    write_checksums(tmp_path)
    assert verify_checksums(tmp_path) == {"model.bin": True}
    artifact.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum verification failed"):
        verify_checksums(tmp_path)


def test_registration_destination_refuses_to_overwrite_existing_bundle(tmp_path: Path):
    destination = tmp_path / "sealed"
    destination.mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        with _atomic_destination(destination):
            pass


def test_mechconf_seed_domain_is_distinct_from_all_prior_cohorts():
    assert MECHCONF_MASTER_SEED != ExperimentConfig.scaled5().master_seed


def test_mechconf2_seed_domain_is_distinct_from_every_prior_campaign():
    seeds = {
        ExperimentConfig.scaled5().master_seed,
        MECHCONF_MASTER_SEED,
        MEMORY_CONFIRM_MASTER_SEED,
        MECHCONF2_MASTER_SEED,
    }
    assert len(seeds) == 4


def test_mechconf2_contract_survives_json_registration_roundtrip():
    contract = _confirmation_experiment().to_dict()
    serialized = json.loads(json.dumps(contract))
    assert serialized == json.loads(json.dumps(_confirmation_experiment().to_dict()))


def test_posthoc_experiment_matches_the_sealed_mechconf_manifest():
    manifest_path = Path("results/mechanistic_confirmation/manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = json.loads(
        json.dumps(_confirmation_experiment(MECHCONF_MASTER_SEED).to_dict())
    )
    assert manifest["experiment"] == expected
