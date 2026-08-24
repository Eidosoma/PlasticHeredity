"""Donor-level exclusions and deterministic confirmation allocation."""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

from .contract import PAIRING_NAMESPACE, hash_order


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


def newideas_exposed_pairs(cohorts: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    for key in (
        "prior_pair_ids_excluded",
        "diagnostic_pair_ids",
        "selection_pair_ids",
        "confirmation_pair_ids",
    ):
        values.extend(str(value) for value in cohorts[key])
    return sorted(set(values))


def donor_ids_from_pairs(pairs: Iterable[Mapping[str, Any]]) -> set[str]:
    return {
        str(pair[key])
        for pair in pairs
        for key in ("a_donor_id", "b_donor_id")
    }


def eligible_pairs(
    pool: Iterable[Mapping[str, Any]],
    stage1_registration: Mapping[str, Any],
    stage2_registration: Mapping[str, Any],
    newideas_cohorts: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pool = list(pool)
    local_pairs = pair_objects([stage1_registration, stage2_registration])
    local_donors = donor_ids_from_pairs(local_pairs)
    exposed_ids = newideas_exposed_pairs(newideas_cohorts)
    newideas_donors = {
        donor for pair_id in exposed_ids for donor in parse_newideas_pair(pair_id)
    }
    excluded = local_donors | newideas_donors
    eligible = [
        dict(pair)
        for pair in pool
        if pair["a_donor_id"] not in excluded and pair["b_donor_id"] not in excluded
    ]
    eligible_ids = hash_order(
        (str(pair["pair_id"]) for pair in eligible), PAIRING_NAMESPACE
    )
    index = {str(pair["pair_id"]): pair for pair in eligible}
    ordered = [index[pair_id] for pair_id in eligible_ids]
    audit = {
        "frozen_pool_pairs": len(pool),
        "local_used_pairs": len(local_pairs),
        "local_used_donors": len(local_donors),
        "newideas_exposed_pair_ids": len(exposed_ids),
        "newideas_exposed_donors": len(newideas_donors),
        "union_excluded_donors": len(excluded),
        "eligible_pairs": len(ordered),
        "max_density_difference": max(
            (float(pair["density_difference"]) for pair in ordered), default=0.0
        ),
    }
    return ordered, audit


def allocate(eligible: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    if len(eligible) < 98:
        raise ValueError(f"need 98 fully fresh pairs; found {len(eligible)}")
    return {"quarantine": eligible[:2], "confirmation": eligible[2:98]}
