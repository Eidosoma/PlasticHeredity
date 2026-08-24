#!/usr/bin/env python3
"""Run the E01 S03 clean-environment CPU/GPU compatibility smoke test."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import traceback
from importlib import metadata
from pathlib import Path
from typing import Any

EXPECTED_VERSIONS = {
    "numpy": "2.4.6",
    "scipy": "1.18.0",
    "scikit-learn": "1.9.0",
    "numba": "0.65.1",
    "torch": "2.11.0+cu128",
    "omegaid": "0.2.5",
}
EXPECTED_ATOMS = {
    "rtr",
    "rtx",
    "rty",
    "rts",
    "xtr",
    "xtx",
    "xty",
    "xts",
    "ytr",
    "ytx",
    "yty",
    "yts",
    "str",
    "stx",
    "sty",
    "sts",
}


def package_versions() -> dict[str, str]:
    result = {
        name: metadata.version(name)
        for name in [*EXPECTED_VERSIONS, "phyid", "llvmlite"]
    }
    for name, expected in EXPECTED_VERSIONS.items():
        if result[name] != expected:
            raise AssertionError(f"{name}: expected {expected}, found {result[name]}")
    return result


def cpu_checks() -> dict[str, Any]:
    import numba
    import numpy as np
    import torch
    from omegaid.core.decomposition import calc_phiid_multivariate
    from omegaid.utils.backend import get_backend_name, set_backend
    from phyid.calculate import calc_PhiID
    from scipy.stats import spearmanr
    from sklearn.linear_model import LinearRegression

    rng = np.random.default_rng(20260801)
    x = np.linspace(-2.0, 2.0, 128, dtype=np.float64)
    design = np.column_stack((x, x**2))
    target = 1.25 * x - 0.75 * x**2 + 0.5
    fit = LinearRegression().fit(design, target)
    score = float(fit.score(design, target))
    rho = float(spearmanr(x, x**3).statistic)

    @numba.njit(cache=False)
    def sum_squares(values):
        total = 0.0
        for value in values:
            total += value * value
        return total

    numba_value = float(sum_squares(x))
    numpy_value = float(np.sum(x * x))

    source = rng.normal(size=512).astype(np.float64)
    target_series = (0.6 * np.roll(source, 1) + rng.normal(scale=0.8, size=512)).astype(
        np.float64
    )
    phyid_checks: dict[str, Any] = {}
    for redundancy in ("MMI", "CCS"):
        atoms, _ = calc_PhiID(
            source, target_series, tau=1, kind="gaussian", redundancy=redundancy
        )
        if set(atoms) != EXPECTED_ATOMS:
            raise AssertionError(f"Unexpected phyid atom keys for {redundancy}")
        finite = all(np.isfinite(value).all() for value in atoms.values())
        if not finite:
            raise AssertionError(f"Non-finite phyid atom for {redundancy}")
        phyid_checks[redundancy] = {
            "atomCount": len(atoms),
            "allFinite": finite,
            "rtrMean": float(np.mean(atoms["rtr"])),
        }

    set_backend("numpy")
    omega_source = np.vstack((source, rng.normal(size=512))).astype(np.float64)
    omega_target = np.vstack((target_series, rng.normal(size=512))).astype(np.float64)
    omega_checks: dict[str, Any] = {}
    for redundancy in ("MMI", "CCS"):
        atoms, _ = calc_phiid_multivariate(
            omega_source,
            omega_target,
            tau=1,
            kind="gaussian",
            redundancy=redundancy,
        )
        if set(atoms) != EXPECTED_ATOMS:
            raise AssertionError(f"Unexpected omegaid atom keys for {redundancy}")
        finite = all(np.isfinite(value).all() for value in atoms.values())
        if not finite:
            raise AssertionError(f"Non-finite omegaid atom for {redundancy}")
        omega_checks[redundancy] = {
            "atomCount": len(atoms),
            "allFinite": finite,
            "rtrMean": float(np.mean(atoms["rtr"])),
        }

    torch.manual_seed(20260801)
    torch.set_default_dtype(torch.float64)
    a = torch.randn((32, 32), dtype=torch.float64, device="cpu")
    cpu_product = a @ a.T
    if cpu_product.dtype != torch.float64 or not torch.isfinite(cpu_product).all():
        raise AssertionError("Invalid PyTorch CPU float64 result")
    if not (
        abs(score - 1.0) < 1e-12
        and abs(rho - 1.0) < 1e-12
        and abs(numba_value - numpy_value) < 1e-12
    ):
        raise AssertionError("CPU scientific-stack numerical smoke failed")
    return {
        "numpyDtype": str(x.dtype),
        "scikitLearnR2": score,
        "scipySpearmanRho": rho,
        "numbaMatchesNumpyAbsoluteError": abs(numba_value - numpy_value),
        "torchDtype": str(cpu_product.dtype),
        "phyid": phyid_checks,
        "omegaidBackend": get_backend_name(),
        "omegaid": omega_checks,
        "valid": True,
        "torchFixture": a.tolist(),
        "torchCpuProduct": cpu_product.tolist(),
    }


def gpu_checks(cpu_result: dict[str, Any], device_index: int) -> dict[str, Any]:
    import torch

    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():
        raise AssertionError("CUDA is not available in the clean environment")
    if device_index >= torch.cuda.device_count():
        raise AssertionError(
            f"Requested cuda:{device_index}; only {torch.cuda.device_count()} devices visible"
        )
    device = torch.device(f"cuda:{device_index}")
    fixture = torch.tensor(cpu_result["torchFixture"], dtype=torch.float64, device=device)
    gpu_product = (fixture @ fixture.T).cpu()
    cpu_product = torch.tensor(cpu_result["torchCpuProduct"], dtype=torch.float64)
    absolute_error = float(torch.max(torch.abs(cpu_product - gpu_product)).item())
    relative_error = float(
        torch.max(
            torch.abs(cpu_product - gpu_product)
            / torch.clamp(torch.abs(cpu_product), min=1e-15)
        ).item()
    )
    if not torch.allclose(cpu_product, gpu_product, rtol=1e-10, atol=1e-10):
        raise AssertionError(
            f"CPU/GPU float64 mismatch: abs={absolute_error}, rel={relative_error}"
        )
    properties = torch.cuda.get_device_properties(device_index)
    return {
        "valid": True,
        "deviceIndex": device_index,
        "deviceName": properties.name,
        "deviceUuid": str(properties.uuid),
        "computeCapability": list(torch.cuda.get_device_capability(device_index)),
        "totalMemoryBytes": properties.total_memory,
        "torchCudaVersion": torch.version.cuda,
        "cudnnVersion": torch.backends.cudnn.version(),
        "dtype": str(gpu_product.dtype),
        "maxAbsoluteErrorVsCpu": absolute_error,
        "maxRelativeErrorVsCpu": relative_error,
        "float32MatmulPrecision": torch.get_float32_matmul_precision(),
        "matmulAllowTf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnnAllowTf32": torch.backends.cudnn.allow_tf32,
        "deterministicAlgorithms": torch.are_deterministic_algorithms_enabled(),
        "visibleDeviceCount": torch.cuda.device_count(),
    }


def execute(device_index: int, cpu_only: bool) -> dict[str, Any]:
    if sys.version_info[:2] != (3, 13):
        raise AssertionError(f"Expected Python 3.13, found {platform.python_version()}")
    versions = package_versions()
    cpu_result = cpu_checks()
    gpu_result = {"skipped": True, "reason": "--cpu-only"}
    if not cpu_only:
        gpu_result = gpu_checks(cpu_result, device_index)
    cpu_result.pop("torchFixture")
    cpu_result.pop("torchCpuProduct")
    return {
        "schema": "eidosoma.e01.s03_clean_environment_smoke.v1",
        "researchStepId": "S03",
        "success": True,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "isolation": {
            "prefix": sys.prefix,
            "basePrefix": sys.base_prefix,
            "isVirtualEnvironment": sys.prefix != sys.base_prefix,
        },
        "threadEnvironment": {
            name: os.environ.get(name)
            for name in [
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMBA_NUM_THREADS",
            ]
        },
        "packages": versions,
        "cpu": cpu_result,
        "gpu": gpu_result,
        "compatibilityPatchApplied": False,
        "compatibilityNote": (
            "Python 3.13-compatible released wheels and the pinned phyid source "
            "installed without source patching."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = execute(args.device_index, args.cpu_only)
    except Exception as error:  # pragma: no cover - exercised in external smoke
        result = {
            "schema": "eidosoma.e01.s03_clean_environment_smoke.v1",
            "researchStepId": "S03",
            "success": False,
            "errorType": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"success": True, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
