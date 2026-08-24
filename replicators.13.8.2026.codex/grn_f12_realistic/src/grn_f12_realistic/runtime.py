from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from .storage import source_manifest


def gpu_inventory() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi", "--query-gpu=index,name,memory.total,driver_version,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    result = []
    for line in completed.stdout.splitlines():
        index, name, memory, driver, capability = [part.strip() for part in line.split(",")]
        result.append({
            "index": int(index), "name": name, "memory_mib": int(memory),
            "driver": driver, "compute_capability": capability,
        })
    return result


def require_gpu(expected_visible: int | None = None) -> list[str]:
    import jax

    devices = jax.devices("gpu")
    if jax.default_backend() != "gpu" or not devices:
        raise RuntimeError("GPU execution is required; CPU fallback is forbidden")
    if expected_visible is not None and len(devices) != expected_visible:
        raise RuntimeError(f"expected {expected_visible} visible GPU(s), found {len(devices)}")
    return [str(device) for device in devices]


def environment_manifest(project_root: str | Path) -> dict[str, Any]:
    import jax
    import numpy
    import optax
    import scipy
    import sklearn

    completed = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"], check=True, capture_output=True, text=True
    )
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "jax": jax.__version__, "jaxlib": jax.lib.__version__,
        "numpy": numpy.__version__, "optax": optax.__version__,
        "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
        "jax_backend": jax.default_backend(), "jax_devices": [str(device) for device in jax.devices()],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpus": gpu_inventory(), "pip_freeze": completed.stdout.splitlines(),
        "source_manifest": source_manifest(project_root),
    }

