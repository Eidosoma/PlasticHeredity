#!/usr/bin/env python3
"""Capture the E01 S03 runtime, dependency, hardware, and license snapshot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import zipfile
from email.parser import Parser
from importlib import metadata
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACTS = Path("/artifacts")
DEFAULT_WHEELHOUSE = Path("/cache/e01_s03/wheelhouse")
DEFAULT_PHYID_WHEEL_DIR = Path("/cache/e01_s03/phyid-wheel")
DEFAULT_PRECISION = REPO_ROOT / "configs/e01/s03_precision_policy.yaml"
DEFAULT_PINS = REPO_ROOT / "configs/e01/s03_source_pins.yaml"

CORE_EXPECTED = {
    "numpy": "2.4.6",
    "scipy": "1.18.0",
    "scikit-learn": "1.9.0",
    "numba": "0.65.1",
    "torch": "2.11.0+cu128",
    "omegaid": "0.2.5",
    "phyid": "0+untagged.8.g6c5f2e9",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_output(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    return {
        "command": command,
        "returnCode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def canonical_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def parse_lock(lock_path: Path) -> tuple[dict[str, str], set[str]]:
    text = lock_path.read_text(encoding="utf-8")
    packages = {
        canonical_name(match.group(1)): match.group(2)
        for match in re.finditer(r"^([A-Za-z0-9_.-]+)==([^ \\\n]+)", text, re.MULTILINE)
    }
    hashes = set(re.findall(r"--hash=sha256:([0-9a-f]{64})", text))
    return packages, hashes


def _license_summary(message: Any) -> tuple[str, str]:
    expression = message.get("License-Expression")
    if expression:
        return expression.strip(), "License-Expression"
    license_text = message.get("License", "")
    first_line = next(
        (line.strip() for line in license_text.splitlines() if line.strip()), ""
    )
    if first_line:
        return first_line[:200], "License"
    classifiers = message.get_all("Classifier", [])
    classifier = next((item for item in classifiers if item.startswith("License ::")), "")
    return classifier or "UNSPECIFIED-IN-WHEEL-METADATA", "Classifier or unavailable"


def inspect_wheel(path: Path, lock_hashes: set[str], source: str) -> dict[str, Any]:
    with zipfile.ZipFile(path) as archive:
        metadata_names = [
            name
            for name in archive.namelist()
            if name.endswith(".dist-info/METADATA") and name.count("/") == 1
        ]
        if len(metadata_names) != 1:
            raise ValueError(f"Expected one METADATA file in {path}")
        message = Parser().parsestr(archive.read(metadata_names[0]).decode("utf-8"))
    digest = sha256_file(path)
    license_value, license_source = _license_summary(message)
    return {
        "package": canonical_name(message["Name"]),
        "version": message["Version"],
        "filename": path.name,
        "artifact_type": "wheel",
        "size_bytes": path.stat().st_size,
        "sha256": digest,
        "hash_in_compiled_lock": digest in lock_hashes,
        "source": source,
        "requires_python": message.get("Requires-Python", ""),
        "license": license_value,
        "license_metadata_source": license_source,
    }


def installed_record_snapshot(package_names: list[str]) -> list[dict[str, Any]]:
    rows = []
    for package_name in package_names:
        distribution = metadata.distribution(package_name)
        record = distribution.read_text("RECORD")
        metadata_text = distribution.read_text("METADATA") or ""
        rows.append(
            {
                "package": canonical_name(distribution.metadata["Name"]),
                "version": distribution.version,
                "distInfoPath": str(distribution._path),
                "recordSha256": (
                    hashlib.sha256(record.encode()).hexdigest() if record else None
                ),
                "metadataSha256": hashlib.sha256(metadata_text.encode()).hexdigest(),
                "installer": (distribution.read_text("INSTALLER") or "").strip(),
                "directUrl": (distribution.read_text("direct_url.json") or "").strip()
                or None,
            }
        )
    return rows


def capture_gpu() -> dict[str, Any]:
    query = command_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    )
    devices = []
    if query["returnCode"] == 0:
        for line in query["stdout"].splitlines():
            index, name, uuid, driver, memory_mib, compute_capability = [
                value.strip() for value in line.split(",")
            ]
            devices.append(
                {
                    "index": int(index),
                    "name": name,
                    "uuid": uuid,
                    "driverVersion": driver,
                    "memoryMiB": int(memory_mib),
                    "computeCapability": compute_capability,
                }
            )
    import torch

    torch_snapshot = {
        "version": torch.__version__,
        "compiledCudaVersion": torch.version.cuda,
        "cudnnVersion": torch.backends.cudnn.version(),
        "cudaAvailable": torch.cuda.is_available(),
        "visibleDeviceCount": torch.cuda.device_count(),
        "float32MatmulPrecision": torch.get_float32_matmul_precision(),
        "matmulAllowTf32AtCapture": torch.backends.cuda.matmul.allow_tf32,
        "cudnnAllowTf32AtCapture": torch.backends.cudnn.allow_tf32,
        "deterministicAlgorithmsAtCapture": torch.are_deterministic_algorithms_enabled(),
    }
    return {"query": query, "devices": devices, "torch": torch_snapshot}


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build(
    artifacts_root: Path,
    *,
    wheelhouse: Path = DEFAULT_WHEELHOUSE,
    phyid_wheel_dir: Path = DEFAULT_PHYID_WHEEL_DIR,
    precision_path: Path = DEFAULT_PRECISION,
    pins_path: Path = DEFAULT_PINS,
) -> dict[str, Any]:
    provenance = artifacts_root / "E01_forensic_replication_bundle/provenance"
    step_dir = artifacts_root / "research_steps/S03"
    provenance.mkdir(parents=True, exist_ok=True)
    step_dir.mkdir(parents=True, exist_ok=True)
    lock_path = provenance / "requirements-s03-py313-cu128.lock"
    clean_freeze_path = provenance / "clean_environment_python_freeze.txt"
    clean_smoke_path = step_dir / "clean_environment_smoke.json"
    if not all(path.is_file() for path in [lock_path, clean_freeze_path, clean_smoke_path]):
        raise FileNotFoundError("Clean lock, freeze, or smoke artifact is missing")

    lock_packages, lock_hashes = parse_lock(lock_path)
    wheel_rows = [
        inspect_wheel(path, lock_hashes, "resolved wheelhouse")
        for path in sorted(wheelhouse.glob("*.whl"))
    ]
    phyid_wheels = list(phyid_wheel_dir.glob("*.whl"))
    if len(phyid_wheels) != 1:
        raise ValueError("Expected exactly one pinned phyid wheel")
    phyid_row = inspect_wheel(phyid_wheels[0], lock_hashes, "pinned git commit build")
    phyid_row["hash_in_compiled_lock"] = False
    wheel_rows.append(phyid_row)
    by_package = {row["package"]: row for row in wheel_rows}

    clean_smoke = json.loads(clean_smoke_path.read_text(encoding="utf-8"))
    clean_versions = {
        canonical_name(name): version for name, version in clean_smoke["packages"].items()
    }
    validation_errors = []
    if len(lock_packages) != 38:
        validation_errors.append(f"Expected 38 locked packages, found {len(lock_packages)}")
    if len(wheel_rows) != 39:
        validation_errors.append(f"Expected 39 wheel artifacts including phyid, found {len(wheel_rows)}")
    if not all(row["hash_in_compiled_lock"] for row in wheel_rows if row["package"] != "phyid"):
        validation_errors.append("One or more resolved wheel hashes are absent from the lock")
    for package, expected in CORE_EXPECTED.items():
        if by_package.get(package, {}).get("version") != expected:
            validation_errors.append(f"Wheel version mismatch for {package}")
        if clean_versions.get(package) != expected:
            validation_errors.append(f"Clean smoke version mismatch for {package}")
    if not clean_smoke.get("success"):
        validation_errors.append("Clean environment smoke did not succeed")

    dpkg = command_output(
        ["dpkg-query", "-W", "-f=${binary:Package}==${Version}\\n"]
    )
    system_lines = sorted(line for line in dpkg["stdout"].splitlines() if line)
    system_lock_path = provenance / "system_packages.lock"
    system_lock_path.write_text("\n".join(system_lines) + "\n", encoding="utf-8")

    pip_freeze = command_output([sys.executable, "-m", "pip", "freeze", "--all"])
    base_freeze_path = provenance / "base_environment_python_freeze.txt"
    base_freeze_path.write_text(pip_freeze["stdout"] + "\n", encoding="utf-8")

    os_release_path = Path("/etc/os-release")
    python_executable = Path(sys.executable).resolve()
    binary_paths = {
        "python": python_executable,
        "nvcc": Path("/usr/local/cuda/bin/nvcc"),
        "nvidiaSmi": Path("/usr/bin/nvidia-smi"),
    }
    binary_hashes = {
        name: {
            "path": str(path),
            "sha256": sha256_file(path) if path.is_file() else None,
            "sizeBytes": path.stat().st_size if path.is_file() else None,
        }
        for name, path in binary_paths.items()
    }
    cpu = {
        "osCpuCount": os.cpu_count(),
        "nproc": command_output(["nproc"]),
        "cpusetEffective": (
            Path("/sys/fs/cgroup/cpuset.cpus.effective").read_text().strip()
            if Path("/sys/fs/cgroup/cpuset.cpus.effective").is_file()
            else None
        ),
        "lscpu": command_output(["lscpu", "--json"]),
        "projectMaximumWorkers": 8,
    }
    gpu = capture_gpu()
    installed = installed_record_snapshot(
        ["numpy", "scipy", "scikit-learn", "numba", "llvmlite", "torch"]
    )
    precision = yaml.safe_load(precision_path.read_text(encoding="utf-8"))
    precision["observedAtSnapshot"] = {
        "visibleGpuCount": len(gpu["devices"]),
        "gpuDevices": gpu["devices"],
        "cleanSmokeArtifact": "$ARTIFACTS_DIR/research_steps/S03/clean_environment_smoke.json",
        "cleanSmokeValid": clean_smoke["success"],
    }
    precision_output = provenance / "precision_policy.yaml"
    precision_output.write_text(
        yaml.safe_dump(precision, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    fingerprint_inputs = {
        "osReleaseSha256": sha256_file(os_release_path),
        "kernel": platform.release(),
        "machine": platform.machine(),
        "pythonVersion": platform.python_version(),
        "binaryHashes": binary_hashes,
        "systemLockSha256": sha256_file(system_lock_path),
        "basePythonFreezeSha256": sha256_file(base_freeze_path),
        "cleanPythonFreezeSha256": sha256_file(clean_freeze_path),
        "compiledLockSha256": sha256_file(lock_path),
        "installedCoreDistributions": installed,
    }
    fingerprint_sha = hashlib.sha256(
        json.dumps(fingerprint_inputs, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    image = {
        "containerRuntimeObserved": "Docker-compatible overlay root in Sysbox-isolated workspace",
        "ociImageReference": "UNAVAILABLE::PARENT_IMAGE_REFERENCE_NOT_EXPOSED",
        "ociImageDigest": "UNAVAILABLE::PARENT_OCI_DIGEST_NOT_EXPOSED",
        "noSilentDefault": True,
        "replacementRuntimeFingerprint": f"sha256:{fingerprint_sha}",
        "limitation": (
            "The parent OCI image reference/digest is not exposed inside this nested "
            "workspace or its inner Docker daemon. The replacement fingerprint covers "
            "OS, kernel, key binaries, system/Python locks, and core distribution records, "
            "but is not an OCI manifest digest."
        ),
    }
    environment = {
        "schema": "eidosoma.e01.s03_environment_snapshot.v1",
        "researchStepId": "S03",
        "capturedOn": "2026-08-01",
        "runtimeImage": image,
        "operatingSystem": {
            "platform": platform.platform(),
            "kernel": platform.release(),
            "machine": platform.machine(),
            "osRelease": platform.freedesktop_os_release(),
            "osReleaseSha256": sha256_file(os_release_path),
            "glibc": platform.libc_ver(),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": str(python_executable),
            "pipVersion": metadata.version("pip"),
            "uvVersion": command_output(["uv", "--version"]),
            "installedCoreDistributions": installed,
        },
        "runtimeBinaries": binary_hashes,
        "cpu": cpu,
        "gpu": gpu,
        "dependencySnapshot": {
            "compiledLockPath": "$ARTIFACTS_DIR/E01_forensic_replication_bundle/provenance/requirements-s03-py313-cu128.lock",
            "compiledLockSha256": sha256_file(lock_path),
            "lockedPackageCount": len(lock_packages),
            "resolvedWheelCount": len(wheel_rows) - 1,
            "phyidCommitWheelCount": 1,
            "wheelhouseCachePath": str(wheelhouse),
            "cleanFreezeSha256": sha256_file(clean_freeze_path),
            "baseFreezeSha256": sha256_file(base_freeze_path),
            "systemLockSha256": sha256_file(system_lock_path),
        },
        "precisionPolicy": "$ARTIFACTS_DIR/E01_forensic_replication_bundle/provenance/precision_policy.yaml",
        "cleanEnvironment": {
            "artifact": "$ARTIFACTS_DIR/research_steps/S03/clean_environment_smoke.json",
            "success": clean_smoke["success"],
            "pythonVersion": clean_smoke["python"]["version"],
            "isVirtualEnvironment": clean_smoke["isolation"]["isVirtualEnvironment"],
            "compatibilityPatchApplied": clean_smoke["compatibilityPatchApplied"],
            "compatibilityNote": clean_smoke["compatibilityNote"],
        },
        "validation": {
            "valid": not validation_errors,
            "errors": validation_errors,
            "coreExpectedVersions": CORE_EXPECTED,
            "allResolvedWheelHashesInLock": all(
                row["hash_in_compiled_lock"]
                for row in wheel_rows
                if row["package"] != "phyid"
            ),
        },
        "runtimeFingerprintInputs": fingerprint_inputs,
    }
    environment_json = provenance / "environment_report.json"
    environment_json.write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    dependency_fields = [
        "package",
        "version",
        "filename",
        "artifact_type",
        "size_bytes",
        "sha256",
        "hash_in_compiled_lock",
        "source",
        "requires_python",
        "license",
        "license_metadata_source",
    ]
    write_csv(provenance / "dependency_artifacts.csv", wheel_rows, dependency_fields)
    write_csv(
        provenance / "dependency_licenses.csv",
        [
            {
                "package": row["package"],
                "version": row["version"],
                "license": row["license"],
                "license_metadata_source": row["license_metadata_source"],
                "artifact_sha256": row["sha256"],
            }
            for row in wheel_rows
        ],
        [
            "package",
            "version",
            "license",
            "license_metadata_source",
            "artifact_sha256",
        ],
    )

    pins = yaml.safe_load(pins_path.read_text(encoding="utf-8"))
    license_lines = [
        "# E01 S03 license notes",
        "",
        "## Top summary",
        "",
        "- **Research step:** S03 — Freeze source and environment snapshots",
        "- **Completion status:** License inventory complete for the S03 source and dependency snapshot.",
        "- **Artifacts written:** This note and `dependency_licenses.csv`; authoritative source identities are in `source_manifest.yaml`.",
        f"- **Validation result:** {'PASS' if not validation_errors else 'FAIL'}; every frozen wheel has license metadata or an explicit unavailable marker.",
        "- **Outcome classification:** Supportive sub-result; this is metadata capture, not legal advice.",
        "- **Caveats or blockers:** Repository-level licensing is absent for historical GARD and absent at the modern GARD root; no redistribution is authorized by this report.",
        "- **Recommended next action:** Keep unlicensed GARD material as commit references/cache-only inputs and consult counsel before redistribution.",
        "",
        "## Source notes",
        "",
        f"- Paper `{pins['paper']['arxivId']}{pins['paper']['version']}`: {pins['paper']['license']}.",
    ]
    for pin in pins["repositories"]:
        license_lines.append(
            f"- `{pin['sourceId']}` at `{pin['commit']}`: {pin['license']}. {pin['redistributionPolicy']}"
        )
    license_lines.extend(
        [
            "",
            "## Dependency notes",
            "",
            f"The clean lock resolved {len(lock_packages)} packages and `dependency_licenses.csv` records the metadata embedded in all {len(wheel_rows)} frozen wheels, including the commit-built `phyid` wheel.",
            "",
            "License strings are transcribed from source or wheel metadata. They may be incomplete for vendored libraries and are not a legal interpretation.",
        ]
    )
    (provenance / "license_notes.md").write_text(
        "\n".join(license_lines) + "\n", encoding="utf-8"
    )

    cpu_model = "Intel Xeon (see lscpu JSON)"
    gpu_summary = ", ".join(
        f"cuda:{item['index']} {item['name']} {item['uuid']}"
        for item in gpu["devices"]
    )
    environment_md = f"""# E01 S03 environment report

## Top summary

- **Research step:** S03 — Freeze source and environment snapshots
- **Completion status:** Environment capture complete.
- **Artifacts written:** JSON environment report, compiled and installed locks, system-package lock, dependency/hash/license tables, precision policy, and clean smoke result.
- **Validation result:** {'PASS' if not validation_errors else 'FAIL'}; {len(lock_packages)} locked packages, {len(wheel_rows)} frozen wheels including `phyid`, and clean Python 3.13 CPU/GPU smoke success.
- **Outcome classification:** Supportive sub-result.
- **Caveats or blockers:** Parent OCI image digest is not exposed and remains an explicit unavailable sentinel; the composite runtime fingerprint is not an OCI digest. Two L4 GPUs were visible although the generic plan described one fast GPU.
- **Recommended next action:** Use the compiled lock and explicit device UUID/precision fields for later environments; do not infer an image digest or GPU selection.

## Runtime and image identity

- OS: {platform.platform()}
- Python: {platform.python_version()} at `{python_executable}`
- CUDA environment: 12.8.1; PyTorch: {CORE_EXPECTED['torch']}
- OCI image digest: `{image['ociImageDigest']}`
- Composite runtime fingerprint: `{image['replacementRuntimeFingerprint']}`

The parent image is outside the nested daemon and its OCI reference/digest is not visible. The fingerprint binds the available OS, kernel, key-binary hashes, system/Python locks, and installed core distribution records without pretending to be an OCI manifest identity.

## CPU, GPU, and precision

- CPU: {cpu_model}; cgroup exposes {cpu['osCpuCount']} logical CPUs, while project policy permits at most 8 workers.
- Visible GPUs: {gpu_summary}
- Frozen numerical reference: CPU float64.
- GPU validation: explicit device index and UUID, float64, TF32 disabled, deterministic validation algorithms, and 1e-10 absolute/relative cross-device tolerance.
- ΩID default backend: NumPy; CuPy is not in the frozen core environment and cannot be selected implicitly.

## Locks and clean-environment result

- Compiled lock SHA-256: `{sha256_file(lock_path)}`
- Clean freeze SHA-256: `{sha256_file(clean_freeze_path)}`
- System lock SHA-256: `{sha256_file(system_lock_path)}`
- Clean environment: Python {clean_smoke['python']['version']}, virtual environment `{clean_smoke['isolation']['prefix']}`, success `{str(clean_smoke['success']).lower()}`.
- Python 3.13 patching: none. Released wheels installed directly; the pinned `phyid` commit built with a missing-README metadata warning but no source modification.

## Validation

All resolved wheel hashes are present in the compiled lock: `{str(environment['validation']['allResolvedWheelHashesInLock']).lower()}`. Dependency consistency and the CPU/GPU smoke passed. Validation errors: `{validation_errors or 'none'}`.
"""
    (provenance / "environment_report.md").write_text(environment_md, encoding="utf-8")

    if validation_errors:
        raise RuntimeError("; ".join(validation_errors))
    return {
        "valid": True,
        "lockedPackageCount": len(lock_packages),
        "wheelArtifactCount": len(wheel_rows),
        "runtimeFingerprint": f"sha256:{fingerprint_sha}",
        "artifacts": [
            str(environment_json),
            str(provenance / "environment_report.md"),
            str(provenance / "dependency_artifacts.csv"),
            str(provenance / "dependency_licenses.csv"),
            str(provenance / "license_notes.md"),
            str(precision_output),
            str(system_lock_path),
            str(base_freeze_path),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--wheelhouse", type=Path, default=DEFAULT_WHEELHOUSE)
    parser.add_argument("--phyid-wheel-dir", type=Path, default=DEFAULT_PHYID_WHEEL_DIR)
    args = parser.parse_args()
    result = build(
        args.artifacts_root,
        wheelhouse=args.wheelhouse,
        phyid_wheel_dir=args.phyid_wheel_dir,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
