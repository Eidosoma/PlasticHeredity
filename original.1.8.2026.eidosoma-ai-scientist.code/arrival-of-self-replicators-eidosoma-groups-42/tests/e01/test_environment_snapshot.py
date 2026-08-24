from __future__ import annotations

import hashlib
import importlib.util
import zipfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENV = load_module(
    "capture_environment_snapshot", "scripts/e01/capture_environment_snapshot.py"
)
SMOKE = load_module("s03_clean_smoke", "scripts/e01/s03_clean_smoke.py")


def test_lock_parser_normalizes_names_and_collects_hashes(tmp_path: Path) -> None:
    lock = tmp_path / "requirements.lock"
    lock.write_text(
        "Example_Package==1.2.3 \\\n+    --hash=sha256:" + "a" * 64 + "\n",
        encoding="utf-8",
    )
    packages, hashes = ENV.parse_lock(lock)
    assert packages == {"example-package": "1.2.3"}
    assert hashes == {"a" * 64}


def test_wheel_inspection_binds_metadata_and_artifact_hash(tmp_path: Path) -> None:
    wheel = tmp_path / "example_package-1.2.3-py3-none-any.whl"
    metadata_text = """Metadata-Version: 2.4
Name: Example_Package
Version: 1.2.3
Requires-Python: >=3.13
License-Expression: BSD-3-Clause

"""
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("example_package-1.2.3.dist-info/METADATA", metadata_text)
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    result = ENV.inspect_wheel(wheel, {digest}, "test")
    assert result["package"] == "example-package"
    assert result["version"] == "1.2.3"
    assert result["sha256"] == digest
    assert result["hash_in_compiled_lock"] is True
    assert result["license"] == "BSD-3-Clause"


def test_precision_and_smoke_contract_forbid_silent_changes() -> None:
    policy = yaml.safe_load(
        (REPO_ROOT / "configs/e01/s03_precision_policy.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert policy["noSilentPrecisionChanges"] is True
    assert policy["cpuReference"]["arrayDtype"] == "float64"
    assert policy["gpuAssisted"]["referenceDtype"] == "float64"
    assert policy["gpuAssisted"]["tensorFloat32Allowed"] is False
    assert policy["gpuAssisted"]["deviceSelection"].startswith("Explicit")
    for package, version in SMOKE.EXPECTED_VERSIONS.items():
        assert ENV.CORE_EXPECTED[package] == version
    assert ENV.CORE_EXPECTED["phyid"] == "0+untagged.8.g6c5f2e9"
