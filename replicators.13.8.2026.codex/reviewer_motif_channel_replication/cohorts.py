"""Outcome-blind fresh-donor pairing and cohort allocation."""

from __future__ import annotations

import hashlib
from typing import Any, Iterable, Mapping

from .contract import PAIRING_NAMESPACE, hash_order, parse_historical_pair_id


DENSITY_CALIPER = 0.02


def historical_donor_ids(pair_ids: Iterable[str]) -> set[str]:
    donors: set[str] = set()
    for pair_id in pair_ids:
        left, right = parse_historical_pair_id(pair_id)
        donors.update((left, right))
    return donors


def _tie_key(left: str, right: str) -> tuple[bytes, str]:
    value = f"{PAIRING_NAMESPACE}\x1f{left}\x1f{right}"
    return hashlib.sha256(value.encode()).digest(), right


def construct_fresh_pair_pool(
    donors: Iterable[Mapping[str, Any]],
    excluded_pair_ids: Iterable[str],
    *,
    density_caliper: float = DENSITY_CALIPER,
) -> list[dict[str, Any]]:
    excluded = historical_donor_ids(excluded_pair_ids)
    eligible = [dict(donor) for donor in donors if donor["donor_id"] not in excluded]
    pairs: list[dict[str, Any]] = []
    for launch in range(4):
        by_label = {
            label: [
                donor
                for donor in eligible
                if int(donor["launch_index"]) == launch and donor["prototype_label"] == label
            ]
            for label in ("A", "B")
        }
        iterate_label = min(("A", "B"), key=lambda label: (len(by_label[label]), label))
        match_label = "B" if iterate_label == "A" else "A"
        ordered = sorted(
            by_label[iterate_label],
            key=lambda donor: _tie_key("iterate", donor["donor_id"]),
        )
        available = {donor["donor_id"]: donor for donor in by_label[match_label]}
        for focal in ordered:
            candidates = [
                donor
                for donor in available.values()
                if abs(float(focal["density"]) - float(donor["density"]))
                <= density_caliper + 1e-12
            ]
            if not candidates:
                continue
            partner = min(
                candidates,
                key=lambda donor: (
                    abs(float(focal["density"]) - float(donor["density"])),
                    _tie_key(focal["donor_id"], donor["donor_id"]),
                ),
            )
            del available[partner["donor_id"]]
            a = focal if focal["prototype_label"] == "A" else partner
            b = focal if focal["prototype_label"] == "B" else partner
            pair_key = hashlib.sha256(
                f"{PAIRING_NAMESPACE}\x1f{a['donor_id']}\x1f{b['donor_id']}".encode()
            ).hexdigest()
            pairs.append(
                {
                    "pair_id": f"fresh-{launch}-{pair_key[:16]}",
                    "launch_index": launch,
                    "a_donor_id": a["donor_id"],
                    "b_donor_id": b["donor_id"],
                    "density_difference": abs(float(a["density"]) - float(b["density"])),
                }
            )
    pair_ids = [pair["pair_id"] for pair in pairs]
    if len(pair_ids) != len(set(pair_ids)):
        raise AssertionError("fresh pair IDs are not unique")
    used = [
        donor_id
        for pair in pairs
        for donor_id in (pair["a_donor_id"], pair["b_donor_id"])
    ]
    if len(used) != len(set(used)):
        raise AssertionError("a donor was reused in the fresh pair pool")
    return sorted(pairs, key=lambda pair: pair["pair_id"])


def allocate_stage1(
    pair_pool: list[dict[str, Any]], namespace: str
) -> dict[str, list[dict[str, Any]]]:
    ordered_ids = hash_order((pair["pair_id"] for pair in pair_pool), namespace)
    index = {pair["pair_id"]: pair for pair in pair_pool}
    required = 64 + 48 + 64
    if len(ordered_ids) < required:
        raise RuntimeError(
            f"UNDERPOWERED_FRESH_POOL: need {required} pairs, found {len(ordered_ids)}"
        )
    return {
        "calibration": [index[pair_id] for pair_id in ordered_ids[:64]],
        "discovery": [index[pair_id] for pair_id in ordered_ids[64:112]],
        "validation": [index[pair_id] for pair_id in ordered_ids[112:176]],
    }


def allocate_stage2(
    pair_pool: list[dict[str, Any]],
    used_stage1: Iterable[str],
    namespace: str,
) -> dict[str, list[dict[str, Any]]]:
    used = set(used_stage1)
    remaining = [pair for pair in pair_pool if pair["pair_id"] not in used]
    ordered_ids = hash_order((pair["pair_id"] for pair in remaining), namespace)
    index = {pair["pair_id"]: pair for pair in remaining}
    required = 2 + 32 + 96
    if len(ordered_ids) < required:
        raise RuntimeError(
            f"UNDERPOWERED_FRESH_POOL: need {required} Stage-2 pairs, found {len(ordered_ids)}"
        )
    return {
        "development_quarantine": [index[pair_id] for pair_id in ordered_ids[:2]],
        "writer_audit": [index[pair_id] for pair_id in ordered_ids[2:34]],
        "outcome": [index[pair_id] for pair_id in ordered_ids[34:130]],
    }
