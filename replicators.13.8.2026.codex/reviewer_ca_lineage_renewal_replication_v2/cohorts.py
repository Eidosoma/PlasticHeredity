"""Donor exclusions and deterministic same-launch matching for v2."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.optimize import linear_sum_assignment

from .contract import CONTRACT, PAIRING_NAMESPACE


NEWIDEAS_PAIR_RE = re.compile(
    r"narrow-[0-9]{4}-(life-31649-[0-3]-[0-9]+)-"
    r"(life-31649-[0-3]-[0-9]+)"
)


def pair_objects(value: Any) -> list[dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            if {"pair_id", "a_donor_id", "b_donor_id"} <= set(item):
                found[str(item["pair_id"])] = dict(item)
            for child in item.values():
                visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return [found[key] for key in sorted(found)]


def parse_newideas_pair(pair_id: str) -> tuple[str, str]:
    match = NEWIDEAS_PAIR_RE.fullmatch(pair_id)
    if match is None:
        raise ValueError(f"invalid NewIdeas pair identifier: {pair_id}")
    return match.group(1), match.group(2)


def newideas_exposed_donors(cohorts: Mapping[str, Any]) -> set[str]:
    pair_ids: set[str] = set()
    for key in (
        "prior_pair_ids_excluded",
        "diagnostic_pair_ids",
        "selection_pair_ids",
        "confirmation_pair_ids",
    ):
        pair_ids.update(str(value) for value in cohorts[key])
    return {donor for pair_id in pair_ids for donor in parse_newideas_pair(pair_id)}


def donor_ids_from_pairs(pairs: Iterable[Mapping[str, Any]]) -> set[str]:
    return {
        str(pair[key])
        for pair in pairs
        for key in ("a_donor_id", "b_donor_id")
    }


def excluded_donors(
    stage1_registration: Mapping[str, Any],
    stage2_registration: Mapping[str, Any],
    source_cohorts: Mapping[str, Any],
    v1_registration: Mapping[str, Any],
) -> tuple[set[str], dict[str, int]]:
    local_pairs = pair_objects([stage1_registration, stage2_registration])
    local = donor_ids_from_pairs(local_pairs)
    source = newideas_exposed_donors(source_cohorts)
    v1_pairs = pair_objects(v1_registration.get("cohorts", {}))
    v1 = donor_ids_from_pairs(v1_pairs)
    union = local | source | v1
    return union, {
        "local_stage1_stage2_pairs": len(local_pairs),
        "local_stage1_stage2_donors": len(local),
        "source_exposed_donors": len(source),
        "sealed_v1_pairs": len(v1_pairs),
        "sealed_v1_donors": len(v1),
        "union_excluded_donors": len(union),
    }


def untouched_donors(
    donors: Iterable[Mapping[str, Any]], excluded: set[str]
) -> list[dict[str, Any]]:
    result = [dict(donor) for donor in donors if str(donor["donor_id"]) not in excluded]
    ids = [str(donor["donor_id"]) for donor in result]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate donor identifiers")
    return result


def _hash_key(value: str) -> tuple[bytes, str]:
    return hashlib.sha256(f"{PAIRING_NAMESPACE}\x1f{value}".encode()).digest(), value


def _match_launch(
    donors_a: list[dict[str, Any]], donors_b: list[dict[str, Any]], launch: int
) -> list[dict[str, Any]]:
    donors_a = sorted(donors_a, key=lambda donor: _hash_key(str(donor["donor_id"])))
    donors_b = sorted(donors_b, key=lambda donor: _hash_key(str(donor["donor_id"])))
    if not donors_a or not donors_b:
        return []
    edge_ids = [
        f"{left['donor_id']}|{right['donor_id']}"
        for left in donors_a
        for right in donors_b
    ]
    ranked = {
        edge_id: rank
        for rank, edge_id in enumerate(sorted(edge_ids, key=_hash_key))
    }
    matched_count = min(len(donors_a), len(donors_b))
    density_scale = 1_000_000
    tie_base = matched_count * len(edge_ids) + 1
    forbidden = (density_scale + 1) * tie_base
    costs = np.empty((len(donors_a), len(donors_b)), dtype=np.int64)
    allowed = np.zeros_like(costs, dtype=bool)
    caliper = float(CONTRACT["density_caliper"])
    for row, left in enumerate(donors_a):
        for column, right in enumerate(donors_b):
            difference = abs(float(left["density"]) - float(right["density"]))
            edge_id = f"{left['donor_id']}|{right['donor_id']}"
            if difference <= caliper + 1e-15:
                allowed[row, column] = True
                density_cost = int(round(difference * density_scale))
                costs[row, column] = density_cost * tie_base + ranked[edge_id]
            else:
                costs[row, column] = forbidden + ranked[edge_id]
    rows, columns = linear_sum_assignment(costs)
    if len(rows) != matched_count or not np.all(allowed[rows, columns]):
        raise ValueError(
            f"launch {launch} cannot match its full minority label under the density caliper"
        )
    pairs: list[dict[str, Any]] = []
    for row, column in zip(rows, columns, strict=True):
        left = donors_a[int(row)]
        right = donors_b[int(column)]
        difference = abs(float(left["density"]) - float(right["density"]))
        identity = f"{left['donor_id']}|{right['donor_id']}"
        digest = hashlib.sha256(
            f"{PAIRING_NAMESPACE}\x1f{identity}".encode()
        ).hexdigest()
        pairs.append(
            {
                "pair_id": f"renewal-v2-{launch}-{digest[:16]}",
                "a_donor_id": str(left["donor_id"]),
                "b_donor_id": str(right["donor_id"]),
                "density_difference": difference,
                "launch_index": launch,
                "pairing_edge_sha256": digest,
            }
        )
    return sorted(pairs, key=lambda pair: _hash_key(str(pair["pair_id"])))


def match_untouched_donors(donors: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    values = [dict(donor) for donor in donors]
    pairs: list[dict[str, Any]] = []
    for launch in range(4):
        launch_donors = [
            donor for donor in values if int(donor["launch_index"]) == launch
        ]
        donors_a = [
            donor for donor in launch_donors if donor["prototype_label"] == "A"
        ]
        donors_b = [
            donor for donor in launch_donors if donor["prototype_label"] == "B"
        ]
        pairs.extend(_match_launch(donors_a, donors_b, launch))
    used = donor_ids_from_pairs(pairs)
    if len(used) != 2 * len(pairs):
        raise AssertionError("a donor was reused by deterministic matching")
    return sorted(pairs, key=lambda pair: _hash_key(str(pair["pair_id"])))


def allocate(
    donors: Iterable[Mapping[str, Any]],
    stage1_registration: Mapping[str, Any],
    stage2_registration: Mapping[str, Any],
    source_cohorts: Mapping[str, Any],
    v1_registration: Mapping[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    all_donors = [dict(donor) for donor in donors]
    excluded, exclusion_audit = excluded_donors(
        stage1_registration, stage2_registration, source_cohorts, v1_registration
    )
    untouched = untouched_donors(all_donors, excluded)
    confirmation = match_untouched_donors(untouched)
    quarantine = pair_objects(v1_registration["cohorts"].get("quarantine", []))
    if len(quarantine) != 2:
        raise ValueError("the sealed v1 engineering cohort must contain exactly two pairs")
    by_launch_label = {
        f"launch{launch}_{label}": sum(
            int(donor["launch_index"]) == launch
            and donor["prototype_label"] == label
            for donor in untouched
        )
        for launch in range(4)
        for label in ("A", "B")
    }
    pairs_by_launch = {
        f"launch{launch}": sum(int(pair["launch_index"]) == launch for pair in confirmation)
        for launch in range(4)
    }
    audit = {
        **exclusion_audit,
        "frozen_donors": len(all_donors),
        "untouched_donors": len(untouched),
        "untouched_by_launch_label": by_launch_label,
        "confirmation_pairs": len(confirmation),
        "confirmation_pairs_by_launch": pairs_by_launch,
        "confirmation_unique_donors": len(donor_ids_from_pairs(confirmation)),
        "maximum_density_difference": max(
            (float(pair["density_difference"]) for pair in confirmation), default=0.0
        ),
        "matching_objective": (
            "maximum same-launch A/B cardinality under density caliper; then minimum "
            "total density difference; SHA256 deterministic tie-break"
        ),
        "quarantine_source": "two already-exposed sealed-v1 engineering pairs",
    }
    return {"quarantine": quarantine, "confirmation": confirmation}, audit
