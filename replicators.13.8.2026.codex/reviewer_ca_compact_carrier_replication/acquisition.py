"""Prospective acquisition of a fresh, outcome-independent donor bank."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.optimize import linear_sum_assignment

from reviewer_ca_lineage_renewal_replication_v2.engine import (
    decode_state_hex,
    encode_state_hex,
    normalized_counts,
    step_rule31649_batch,
    texture2x2_counts_batch,
)

from .contract import (
    ACQUISITION_NAMESPACE,
    CONTRACT,
    DEFAULT_ARTIFACTS,
    PAIRING_NAMESPACE,
    PROFILE,
    SCHEMA_VERSION,
    atomic_write_json,
    hash_order,
    load_json,
    semantic_seed,
    sha256_json,
)
from .snapshot import verify_snapshot


def _cosine_scores(counts: np.ndarray, targets: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    observation = normalized_counts(counts)
    scores: list[np.ndarray] = []
    for label in ("A", "B"):
        target = np.asarray(targets[label], dtype=np.float64)
        numerator = np.sum(observation * target, axis=-1)
        denominator = np.linalg.norm(observation, axis=-1) * np.linalg.norm(target)
        scores.append(
            np.divide(
                numerator,
                denominator,
                out=np.zeros_like(numerator),
                where=denominator > 0,
            )
        )
    return scores[0], scores[1]


def _candidate_noise(launch: int, count: int) -> np.ndarray:
    noise = np.empty((count, 64, 16, 16), dtype=bool)
    probability = float(CONTRACT["ordinary_process_noise"])
    for candidate in range(count):
        rng = np.random.default_rng(
            semantic_seed(ACQUISITION_NAMESPACE, launch, candidate, "process")
        )
        noise[candidate] = rng.random((64, 16, 16)) < probability
    return noise


def generate_launch_candidates(
    *,
    launch: int,
    reset_state_hex: str,
    targets: Mapping[str, np.ndarray],
    historical_states: set[str],
    count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reset = decode_state_hex(reset_state_hex)
    boards = np.broadcast_to(reset, (count, 16, 16)).copy()
    noise = _candidate_noise(launch, count)
    observation_counts = np.zeros((count, 15), dtype=np.int64)
    for sweep in range(1, 65):
        boards = step_rule31649_batch(boards)
        boards ^= noise[:, sweep - 1].astype(np.uint8)
        if 57 <= sweep <= 64:
            observation_counts += texture2x2_counts_batch(boards)
    score_a, score_b = _cosine_scores(observation_counts, targets)
    choose_a = score_a >= score_b
    best = np.where(choose_a, score_a, score_b)
    other = np.where(choose_a, score_b, score_a)
    resolved = (
        (best >= float(CONTRACT["acquisition_similarity"]))
        & ((best - other) >= float(CONTRACT["acquisition_margin"]))
    )
    alive = np.any(boards, axis=(-2, -1))
    donors: list[dict[str, Any]] = []
    seen_states: set[str] = set()
    rejections = {
        "dead": 0,
        "unresolved": 0,
        "historically_identical": 0,
        "fresh_duplicate": 0,
    }
    for index, board in enumerate(boards):
        state_hex = encode_state_hex(board)
        if not alive[index]:
            rejections["dead"] += 1
            continue
        if not resolved[index]:
            rejections["unresolved"] += 1
            continue
        if state_hex in historical_states:
            rejections["historically_identical"] += 1
            continue
        if state_hex in seen_states:
            rejections["fresh_duplicate"] += 1
            continue
        seen_states.add(state_hex)
        label = "A" if bool(choose_a[index]) else "B"
        identity = hashlib.sha256(
            f"{ACQUISITION_NAMESPACE}\x1f{launch}\x1f{index}\x1f{state_hex}".encode()
        ).hexdigest()[:16]
        donors.append(
            {
                "donor_id": f"fresh-31649-{launch}-{index:04d}-{identity}",
                "launch_index": launch,
                "candidate_index": index,
                "initial_state_hex": reset_state_hex,
                "donor_state_hex": state_hex,
                "prototype_label": label,
                "density": float(np.mean(board)),
                "best_similarity": float(best[index]),
                "other_similarity": float(other[index]),
                "assignment_margin": float(best[index] - other[index]),
                "trajectory_seed": semantic_seed(
                    ACQUISITION_NAMESPACE, launch, index, "process"
                ),
            }
        )
    return donors, {
        "launch_index": launch,
        "generated": count,
        "accepted": len(donors),
        "accepted_A": sum(donor["prototype_label"] == "A" for donor in donors),
        "accepted_B": sum(donor["prototype_label"] == "B" for donor in donors),
        "rejections": rejections,
    }


def historical_state_set(document: Mapping[str, Any]) -> set[str]:
    states: set[str] = set()
    for donor in document["donors"]:
        for key, value in donor.items():
            if key.endswith("_state_hex") and isinstance(value, str):
                states.add(value)
    return states


def _tie_key(value: str) -> tuple[bytes, str]:
    return hashlib.sha256(f"{PAIRING_NAMESPACE}\x1f{value}".encode()).digest(), value


def _match_launch(donors: Iterable[Mapping[str, Any]], launch: int) -> list[dict[str, Any]]:
    left = sorted(
        (
            dict(donor)
            for donor in donors
            if int(donor["launch_index"]) == launch
            and donor["prototype_label"] == "A"
        ),
        key=lambda donor: (float(donor["density"]), _tie_key(str(donor["donor_id"]))),
    )
    right = sorted(
        (
            dict(donor)
            for donor in donors
            if int(donor["launch_index"]) == launch
            and donor["prototype_label"] == "B"
        ),
        key=lambda donor: (float(donor["density"]), _tie_key(str(donor["donor_id"]))),
    )
    if not left or not right:
        return []
    edge_ids = [
        f"{a['donor_id']}|{b['donor_id']}" for a in left for b in right
    ]
    edge_rank = {edge: rank for rank, edge in enumerate(sorted(edge_ids, key=_tie_key))}
    density_scale = 1_000_000
    tie_base = len(edge_ids) + 1
    max_allowed = int(round(float(CONTRACT["density_caliper"]) * density_scale)) * tie_base + len(edge_ids)
    unmatched_penalty = (len(left) + len(right) + 1) * (max_allowed + 1)
    forbidden = 2 * unmatched_penalty
    # Each A donor chooses one B donor or a private dummy column.  The large
    # dummy penalty maximizes cardinality before density and hash tie costs.
    costs = np.full((len(left), len(right) + len(left)), forbidden, dtype=np.int64)
    allowed = np.zeros((len(left), len(right)), dtype=bool)
    for row, a in enumerate(left):
        for column, b in enumerate(right):
            difference = abs(float(a["density"]) - float(b["density"]))
            edge = f"{a['donor_id']}|{b['donor_id']}"
            if difference <= float(CONTRACT["density_caliper"]) + 1e-15:
                allowed[row, column] = True
                costs[row, column] = (
                    int(round(difference * density_scale)) * tie_base + edge_rank[edge]
                )
        costs[row, len(right) + row] = unmatched_penalty + row
    rows, columns = linear_sum_assignment(costs)
    pairs: list[dict[str, Any]] = []
    for row, column in zip(rows, columns, strict=True):
        if column >= len(right) or not allowed[row, column]:
            continue
        a = left[int(row)]
        b = right[int(column)]
        identity = f"{a['donor_id']}|{b['donor_id']}"
        digest = hashlib.sha256(
            f"{PAIRING_NAMESPACE}\x1f{identity}".encode()
        ).hexdigest()
        pairs.append(
            {
                "pair_id": f"fresh-pair-{launch}-{digest[:20]}",
                "a_donor_id": a["donor_id"],
                "b_donor_id": b["donor_id"],
                "launch_index": launch,
                "density_difference": abs(float(a["density"]) - float(b["density"])),
            }
        )
    return pairs


def match_donors(donors: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    values = [dict(donor) for donor in donors]
    pairs = [pair for launch in range(4) for pair in _match_launch(values, launch)]
    index = {str(pair["pair_id"]): pair for pair in pairs}
    return [index[pair_id] for pair_id in hash_order(index, PAIRING_NAMESPACE)]


def acquire(artifacts_root: Path | None = None) -> dict[str, Any]:
    artifacts = (artifacts_root or DEFAULT_ARTIFACTS).resolve()
    if (artifacts / "REGISTRATION.json").exists():
        raise RuntimeError("fresh acquisition is frozen by registration")
    verify_snapshot(artifacts / "input")
    hypothesis = load_json(artifacts / "input/local/HYPOTHESIS.json")
    launches = load_json(artifacts / "input/local/LAUNCH_RESETS.json")
    historical = load_json(artifacts / "input/local/DONORS.json")
    targets = {
        label: np.asarray(hypothesis["targets"]["primary"][label], dtype=np.float64)
        for label in ("A", "B")
    }
    historical_states = historical_state_set(historical)
    donors: list[dict[str, Any]] = []
    launch_audits: list[dict[str, Any]] = []
    count = int(PROFILE["acquisition_candidates_per_launch"])
    for launch in range(int(PROFILE["acquisition_launches"])):
        produced, audit = generate_launch_candidates(
            launch=launch,
            reset_state_hex=str(launches[f"launch{launch}"]),
            targets=targets,
            historical_states=historical_states,
            count=count,
        )
        donors.extend(produced)
        launch_audits.append(audit)
    pairs = match_donors(donors)
    required = int(PROFILE["minimum_fresh_pairs"])
    state = "READY" if len(pairs) >= required else "UNDERPOWERED_FRESH_ACQUISITION"
    allocations = {
        "engineering": pairs[: int(PROFILE["engineering_pairs"])],
        "confirmation": pairs[
            int(PROFILE["engineering_pairs"]): int(PROFILE["engineering_pairs"])
            + int(PROFILE["confirmation_pairs"])
        ],
        "audit_reserve": pairs[
            int(PROFILE["engineering_pairs"]) + int(PROFILE["confirmation_pairs"]):required
        ],
    }
    used = [
        pair[key]
        for cohort in allocations.values()
        for pair in cohort
        for key in ("a_donor_id", "b_donor_id")
    ]
    audit = {
        "schema_version": SCHEMA_VERSION,
        "state": state,
        "namespace": ACQUISITION_NAMESPACE,
        "thresholds_frozen_before_acquisition": True,
        "adaptive_extension_or_threshold_change": False,
        "generated_candidates": count * int(PROFILE["acquisition_launches"]),
        "accepted_donors": len(donors),
        "matched_pairs": len(pairs),
        "required_pairs": required,
        "launches": launch_audits,
        "matched_pairs_by_launch": {
            f"launch{launch}": sum(pair["launch_index"] == launch for pair in pairs)
            for launch in range(4)
        },
        "allocation_counts": {name: len(value) for name, value in allocations.items()},
        "allocated_donors_unique": len(used) == len(set(used)),
        "maximum_allocated_density_difference": max(
            (pair["density_difference"] for cohort in allocations.values() for pair in cohort),
            default=0.0,
        ),
        "historical_state_count": len(historical_states),
        "historical_state_overlap": sum(
            donor["donor_state_hex"] in historical_states for donor in donors
        ),
        "matching_objective": (
            "maximum same-launch A/B cardinality under 0.02 density caliper; "
            "then minimum density distance and deterministic SHA256 tie-break"
        ),
    }
    acquisition = {
        "schema_version": SCHEMA_VERSION,
        "state": state,
        "namespace": ACQUISITION_NAMESPACE,
        "donors": donors,
        "donor_digest": sha256_json(donors),
        "pairs": pairs,
        "pair_digest": sha256_json(pairs),
    }
    atomic_write_json(artifacts / "ACQUISITION.json", acquisition)
    atomic_write_json(
        artifacts / "COHORTS.json",
        {
            "schema_version": SCHEMA_VERSION,
            "state": state,
            "allocation_namespace": PAIRING_NAMESPACE,
            "cohorts": allocations,
            "audit": audit,
        },
    )
    atomic_write_json(artifacts / "ACQUISITION_AUDIT.json", audit)
    return audit
