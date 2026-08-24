from __future__ import annotations

import ast
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from .config import PROJECT_DIR, Registration
from .contracts import validate_protocol_counts
from .engine import (
    longest_true_run,
    sequential_sweep_numpy,
    signed_update,
    states_to_int,
)
from .experiment import _update_latch
from .rng import semantic_bytes
from .source import enumerate_landscape, generate_rulebook


def _runtime_import_violations() -> list[str]:
    violations: list[str] = []
    for path in sorted((PROJECT_DIR / "src").rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                compact = name.lower().replace("_", "")
                if "newideas" in compact or name == "wagner_memory_cleanroom" or name.startswith("wagner_memory_cleanroom."):
                    violations.append(f"{path.relative_to(PROJECT_DIR)}:{name}")
    return violations


def validate(registration: Registration) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    previous = np.asarray([-1, 1, -1], dtype=np.int8)
    checks["ties_retain_previous"] = np.array_equal(
        signed_update(np.zeros(3), previous), previous
    )
    weights = np.asarray([[0, 1], [1, 0]], dtype=np.float32)
    state = np.asarray([[1, -1]], dtype=np.int8)
    checks["sequential_update_order"] = np.array_equal(
        sequential_sweep_numpy(weights, state), np.asarray([[-1, -1]], dtype=np.int8)
    )
    checks["strict_run_counter"] = np.array_equal(
        longest_true_run(np.asarray([[1, 0], [1, 1], [0, 1], [1, 1]], dtype=bool)),
        np.asarray([2, 3]),
    )

    first = generate_rulebook(0, registration.protocol, "validation")
    second = generate_rulebook(0, registration.protocol, "validation")
    landscape = enumerate_landscape(
        first.weights, first.target_a.size, int(registration.engine["max_sweeps"])
    )
    target_a = int(states_to_int(first.target_a[None, :])[0])
    target_b = int(states_to_int(first.target_b[None, :])[0])
    midpoint_values = states_to_int(np.stack(first.midpoints))
    target_a_id = int(landscape.attractor_index[target_a])
    target_b_id = int(landscape.attractor_index[target_b])
    checks["source_replay"] = (
        np.array_equal(first.weights, second.weights)
        and first.proposal_log == second.proposal_log
    )
    master_seed = str(registration.protocol["master_seed"])
    domain_digests = {
        domain: semantic_bytes(master_seed, "source", domain, 0, 0).hex()
        for domain in ("full:state", "smoke:state", "benchmark:full:state")
    }
    checks["source_domains_registered"] = (
        registration.engine.get("source_domain_scheme")
        == "profile:stage; benchmark:profile:stage"
    )
    checks["source_domains_disjoint"] = len(set(domain_digests.values())) == len(domain_digests)
    details["source_domain_seed_digests"] = domain_digests
    checks["source_float64"] = first.weights.dtype == np.float64
    checks["dense_diagonal_retained"] = bool(np.all(first.weights != 0) and np.all(np.diag(first.weights) != 0))
    checks["targets_complementary"] = np.array_equal(first.target_a, -first.target_b)
    checks["targets_are_point_attractors"] = (
        int(landscape.successor[target_a]) == target_a
        and int(landscape.successor[target_b]) == target_b
    )
    checks["basins_match_used_matrix"] = bool(
        np.isclose(first.basin_a, np.mean(landscape.attractor_index == target_a_id))
        and np.isclose(first.basin_b, np.mean(landscape.attractor_index == target_b_id))
    )
    checks["midpoints_complementary_and_balanced"] = (
        int(midpoint_values[0] ^ midpoint_values[1]) == (1 << first.target_a.size) - 1
        and all(int((int(value) ^ target_a).bit_count()) == first.target_a.size // 2 for value in midpoint_values)
        and all(int((int(value) ^ target_b).bit_count()) == first.target_a.size // 2 for value in midpoint_values)
    )
    forced_ids = states_to_int(np.stack((first.forced_a, first.forced_b)))
    checks["forced_breaks_outside_target_basins"] = (
        int(landscape.attractor_index[int(forced_ids[0])]) != target_a_id
        and int(landscape.attractor_index[int(forced_ids[1])]) != target_b_id
    )
    accepted_log = first.proposal_log[-1]
    checks["accepted_weight_digest_is_used_matrix"] = (
        accepted_log["weight_sha256"] == sha256(first.weights.tobytes(order="C")).hexdigest()
    )
    checks["proposal_decisions_complete"] = (
        len(first.proposal_log) == first.proposal_count
        and bool(first.proposal_log[-1]["accepted"])
        and all(not row["accepted"] for row in first.proposal_log[:-1])
        and all("successor_sha256" in row and "assignment_sha256" in row for row in first.proposal_log)
    )

    carrier = np.zeros((1, 2), dtype=np.int8)
    ttl = np.zeros((1, 2), dtype=np.int16)
    pending = np.zeros((1, 2), dtype=np.int8)
    streak = np.zeros((1, 2), dtype=np.int16)
    trajectory = np.asarray([[[1, -1]]], dtype=np.int8)
    written, written_ttl, pending, streak = _update_latch(
        carrier, ttl, pending, streak, trajectory,
        retention=16, threshold=1, rewrite=True,
    )
    checks["threshold_one_trajectory_write"] = bool(
        np.array_equal(written, trajectory[-1]) and np.all(written_ttl == 16)
    )
    checks["mark_washout_semantics_registered"] = (
        registration.protocol["slow_mark"].get("washout_age_updates")
        == "expression-and-mark-recurrently"
    )
    not_yet, _, pending2, streak2 = _update_latch(
        carrier, ttl, np.zeros_like(pending), np.zeros_like(streak), trajectory,
        retention=16, threshold=2, rewrite=True,
    )
    written2, _, _, _ = _update_latch(
        not_yet, ttl, pending2, streak2, trajectory,
        retention=16, threshold=2, rewrite=True,
    )
    checks["consecutive_write_threshold"] = np.all(not_yet == 0) and np.array_equal(written2, trajectory[-1])

    count_contract = validate_protocol_counts(registration)
    checks["registered_counts"] = bool(count_contract["valid"])
    details["count_contract"] = count_contract
    violations = _runtime_import_violations()
    checks["cleanroom_runtime_imports"] = not violations
    details["runtime_import_violations"] = violations
    checks["two_gpu_contract"] = int(registration.operations["required_gpu_count"]) == 2
    checks["twelve_hour_hard_guard"] = float(registration.operations["hard_deadline_hours"]) <= 12.0
    return {
        "format": "wagner-memory-validation-v2",
        "profile": registration.profile_name,
        "protocol_digest": registration.protocol_digest,
        "checks": checks,
        "details": details,
        "valid": all(checks.values()),
    }
