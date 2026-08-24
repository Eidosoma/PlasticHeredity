from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .config import PROJECT_DIR, Registration
from .engine import sequential_sweep_numpy, signed_update
from .source import generate_rulebook


def validate(registration: Registration) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    previous = np.asarray([-1, 1, -1], dtype=np.int8)
    checks["ties_retain_previous"] = np.array_equal(signed_update(np.zeros(3), previous), previous)
    weights = np.asarray([[0, 1], [1, 0]], dtype=float)
    state = np.asarray([[1, -1]], dtype=np.int8)
    checks["sequential_update_order"] = np.array_equal(sequential_sweep_numpy(weights, state), np.asarray([[-1, -1]], dtype=np.int8))
    first = generate_rulebook(0, registration.protocol, "validation")
    second = generate_rulebook(0, registration.protocol, "validation")
    checks["source_replay"] = np.array_equal(first.weights, second.weights) and np.array_equal(first.target_a, second.target_a)
    checks["targets_distinct"] = int(np.sum(first.target_a != first.target_b)) >= int(registration.engine["minimum_target_hamming"])
    checks["targets_stable"] = np.array_equal(sequential_sweep_numpy(first.weights, first.target_a[None, :])[0], first.target_a) and np.array_equal(sequential_sweep_numpy(first.weights, first.target_b[None, :])[0], first.target_b)
    checks["k10_duplicate_absent"] = not any("k10" in arm for arm in registration.protocol["carrier"]["arms"])
    forbidden_hits: list[str] = []
    forbidden_parent = ".." + "/" + "New" + "Ideas"
    forbidden_import = "import " + "New" + "Ideas"
    for path in list((PROJECT_DIR / "src").rglob("*.py")) + list((PROJECT_DIR / "protocols").rglob("*.json")):
        text = path.read_text()
        if forbidden_parent in text or forbidden_import in text:
            forbidden_hits.append(str(path.relative_to(PROJECT_DIR)))
    checks["no_forbidden_runtime_path"] = not forbidden_hits
    return {
        "format": "wagner-memory-validation-v1",
        "checks": checks,
        "forbidden_hits": forbidden_hits,
        "valid": all(checks.values()),
    }
