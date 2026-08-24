"""Workspace-bound identities used by S06 exact-regeneration validation."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from .seed import canonical_json_bytes, sha256_hex
from .trajectory import CaptureIdentity, RegistryBoundary, TrajectoryContractError

EXPECTED_REGISTRY_SHA256 = (
    "aef0e179de6466697540ba10236ed24af37fbda12bd4f1c6b1fb5fe7a27af891"
)
EXPECTED_RUNTIME_FINGERPRINT = (
    "sha256:755207c258f156260e5854db667ae2ba2edf62ffc6a6c1e5cf06009d451a86c0"
)
EXPECTED_NUMERIC_THREAD_ENVIRONMENT = {
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package_source_sha256(package_directory: Path) -> str:
    """Hash a canonical path-to-file-hash manifest for one Python package."""

    files = sorted(
        path
        for path in package_directory.glob("*.py")
        if path.is_file() and path.name != "__pycache__"
    )
    if not files:
        raise TrajectoryContractError(
            f"No Python sources found in {package_directory}."
        )
    manifest = {
        path.name: {"sha256": file_sha256(path), "sizeBytes": path.stat().st_size}
        for path in files
    }
    return sha256_hex(canonical_json_bytes(manifest))


def repository_head(repository_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = result.stdout.strip()
    if len(commit) != 40:
        raise TrajectoryContractError("Repository HEAD is not a full Git commit.")
    return commit


def capture_identity_from_workspace(
    *,
    repository_root: Path,
    artifacts_root: Path,
) -> CaptureIdentity:
    """Build the exact current engine/runtime/schema identity from frozen files."""

    environment_path = (
        artifacts_root
        / "E01_forensic_replication_bundle/provenance/environment_report.json"
    )
    environment = json.loads(environment_path.read_text())
    runtime_fingerprint = environment["runtimeImage"]["replacementRuntimeFingerprint"]
    if runtime_fingerprint != EXPECTED_RUNTIME_FINGERPRINT:
        raise TrajectoryContractError("Unexpected frozen runtime fingerprint.")
    actual_thread_environment = {
        name: os.environ.get(name) for name in EXPECTED_NUMERIC_THREAD_ENVIRONMENT
    }
    if actual_thread_environment != EXPECTED_NUMERIC_THREAD_ENVIRONMENT:
        raise TrajectoryContractError(
            "Numeric thread environment is not frozen to one thread: "
            f"{actual_thread_environment}."
        )
    config_root = repository_root / "configs/e01"
    return CaptureIdentity(
        engine_id="e01_gard_independent@1.0.0",
        engine_package="e01_gard_independent",
        engine_version="1.0.0",
        repository_commit=repository_head(repository_root),
        engine_source_sha256=package_source_sha256(
            repository_root / "src/e01_gard_independent"
        ),
        adapter_source_sha256=package_source_sha256(
            repository_root / "src/e01_gard_reproducibility"
        ),
        python_version=platform.python_version(),
        numpy_version=np.__version__,
        platform=platform.platform(),
        byte_order=sys.byteorder,
        runtime_fingerprint=runtime_fingerprint,
        seed_schema_sha256=file_sha256(config_root / "s06_seed_schema.json"),
        trajectory_schema_sha256=file_sha256(
            config_root / "s06_trajectory_schema.json"
        ),
        precision_contract_sha256=file_sha256(
            config_root / "s06_precision_contract.yaml"
        ),
        numeric_thread_environment=tuple(
            sorted(EXPECTED_NUMERIC_THREAD_ENVIRONMENT.items())
        ),
    )


def registry_boundary_from_workspace(*, artifacts_root: Path) -> RegistryBoundary:
    """Load and fail closed on the unchanged v0.3.0 registry."""

    path = (
        artifacts_root
        / "E01_forensic_replication_bundle/specifications/specification_registry_v0.3.0.yaml"
    )
    digest = file_sha256(path)
    if digest != EXPECTED_REGISTRY_SHA256:
        raise TrajectoryContractError("Registry v0.3.0 hash changed.")
    registry = yaml.safe_load(path.read_text())
    parameters = registry["parameters"]
    gate = registry["executionGate"]
    by_parameter = {item["parameter"]: item for item in parameters}
    if len(by_parameter) != len(parameters):
        raise TrajectoryContractError("Registry parameter identities are not unique.")
    if by_parameter["gard.initial_state.rng_stream"]["value"] != "UNRESOLVED::E01-A020":
        raise TrajectoryContractError("Author RNG sentinel changed.")
    if (
        by_parameter["preprocessing.state_sampling_instant"]["value"]
        != "UNRESOLVED::E01-A025"
    ):
        raise TrajectoryContractError("State-sampling sentinel changed.")
    return RegistryBoundary(
        registry_version=registry["registryVersion"],
        registry_sha256=digest,
        registry_executable=gate["executable"],
        no_silent_defaults=gate["noSilentDefaults"],
        parameter_count=len(parameters),
        unresolved_parameter_count=gate["unresolvedParameterCount"],
        unexpanded_branch_set_count=gate["unexpandedBranchSetCount"],
    )
