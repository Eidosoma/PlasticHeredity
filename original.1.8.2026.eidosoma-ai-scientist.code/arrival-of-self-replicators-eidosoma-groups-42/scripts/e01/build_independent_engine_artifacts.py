#!/usr/bin/env python3
"""Validate the S05 independent GARD engine and write compact evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import MISSING, asdict, fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from e01_gard_historical import (
    NumpyUniformSource,
    historical_single_event,
)
from e01_gard_historical import (
    advance_one_generation as historical_advance_generation,
)
from e01_gard_historical import (
    compute_propensities as historical_propensities,
)
from e01_gard_historical import (
    split_fixed_size_without_replacement as historical_fission,
)
from e01_gard_independent import (
    GardSpecification,
    RNGInput,
    RNGStreams,
    SpecificationError,
    advance_generation,
    calculate_propensities,
    fission,
    sample_update,
    specification_from_mapping,
)
from e01_gard_independent import __all__ as public_api
from e01_gard_independent import __version__ as engine_version

PROFILES_PATH = REPOSITORY_ROOT / "configs/e01/s05_specification_profiles.yaml"
CONTRACT_PATH = REPOSITORY_ROOT / "configs/e01/s05_independent_contract.yaml"
PACKAGE_ROOT = REPOSITORY_ROOT / "src/e01_gard_independent"
TEST_PATH = REPOSITORY_ROOT / "tests/e01/test_independent_engine.py"
HISTORICAL_PROFILE = "E01-S05-HISTORICAL-DISTRIBUTION-COMPARISON-v1.0.0"
PACKAGE_FILES = sorted(PACKAGE_ROOT.glob("*.py"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a mapping in {path}.")
    return payload


def to_builtin(value: Any) -> Any:
    if is_dataclass(value):
        return to_builtin(asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_builtin(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def raw_profile(profile_id: str) -> dict[str, Any]:
    profiles = load_yaml(PROFILES_PATH)
    raw = dict(profiles["profiles"][profile_id])
    raw.pop("evidenceBoundary")
    return raw


def load_profile(profile_id: str) -> GardSpecification:
    return specification_from_mapping(raw_profile(profile_id))


def rng_streams(prefix: str, seeds: tuple[int, ...]) -> RNGStreams:
    if len(seeds) != 6:
        raise ValueError("Exactly six literal validation seeds are required.")
    purposes = ("beta", "init", "events", "waiting", "fission", "daughter")
    streams = [
        RNGInput(f"{prefix}-{purpose}-{seed}", np.random.default_rng(seed))
        for purpose, seed in zip(purposes, seeds, strict=True)
    ]
    return RNGStreams(*streams)


def total_variation(left: Counter[Any], right: Counter[Any]) -> float:
    left_total = sum(left.values())
    right_total = sum(right.values())
    support = set(left) | set(right)
    return 0.5 * sum(
        abs(left.get(key, 0) / left_total - right.get(key, 0) / right_total)
        for key in support
    )


def target_total_variation(counts: Counter[int], target: np.ndarray) -> float:
    total = sum(counts.values())
    return 0.5 * sum(
        abs(counts.get(index, 0) / total - float(probability))
        for index, probability in enumerate(target)
    )


def counter_rows(
    independent: Counter[Any], historical: Counter[Any]
) -> list[dict[str, Any]]:
    support = sorted(set(independent) | set(historical), key=repr)
    return [
        {
            "outcome": to_builtin(outcome),
            "independentCount": independent.get(outcome, 0),
            "historicalCount": historical.get(outcome, 0),
        }
        for outcome in support
    ]


def comparison_sizes(quick: bool) -> dict[str, int]:
    configured = load_yaml(PROFILES_PATH)["distributionalValidation"]
    if not quick:
        return {
            "propensity": configured["propensityCases"],
            "event": configured["eventDrawsPerEngine"],
            "fission": configured["fissionDrawsPerEngine"],
            "oddFission": configured["oddFissionDrawsPerEngine"],
            "trajectory": configured["trajectoryDrawsPerEngine"],
        }
    return {
        "propensity": 32,
        "event": 3000,
        "fission": 2000,
        "oddFission": 2000,
        "trajectory": 1000,
    }


def run_propensity_comparison(case_count: int) -> dict[str, Any]:
    started = time.perf_counter()
    specification = load_profile(HISTORICAL_PROFILE)
    generator = np.random.default_rng(50001)
    maxima = {"boost": 0.0, "join": 0.0, "leave": 0.0, "total": 0.0}
    for _ in range(case_count):
        state = generator.integers(0, 8, size=specification.n_species)
        if int(state.sum()) == 0:
            state[0] = 1
        beta = np.exp(generator.normal(-1.0, 1.0, size=(3, 3)))
        independent = calculate_propensities(
            state, beta=beta, specification=specification
        )
        historical = historical_propensities(
            state,
            beta=beta,
            rho=specification.rho,
            k_f=specification.k_f,
            k_b=specification.k_b,
        )
        maxima["boost"] = max(
            maxima["boost"],
            float(np.max(np.abs(np.asarray(independent.boost) - historical.boost))),
        )
        maxima["join"] = max(
            maxima["join"],
            float(np.max(np.abs(np.asarray(independent.join) - historical.join))),
        )
        maxima["leave"] = max(
            maxima["leave"],
            float(np.max(np.abs(np.asarray(independent.leave) - historical.leave))),
        )
        maxima["total"] = max(
            maxima["total"], abs(independent.total - historical.total)
        )
    return {
        "comparisonId": "S05-C01",
        "caseCount": case_count,
        "maximumAbsoluteErrors": maxima,
        "overallMaximumAbsoluteError": max(maxima.values()),
        "elapsedSeconds": time.perf_counter() - started,
    }


def run_event_comparison(draw_count: int) -> dict[str, Any]:
    started = time.perf_counter()
    specification = load_profile(HISTORICAL_PROFILE)
    beta = np.asarray(
        [[1.0, 0.2, 0.1], [0.5, 0.3, 0.4], [0.2, 0.8, 0.1]],
        dtype=np.float64,
    )
    state = (2, 1, 1)
    independent_props = calculate_propensities(
        state, beta=beta, specification=specification
    )
    historical_props = historical_propensities(
        state,
        beta=beta,
        rho=specification.rho,
        k_f=specification.k_f,
        k_b=specification.k_b,
    )
    np.testing.assert_allclose(
        independent_props.concatenated,
        historical_props.concatenated,
        rtol=0.0,
        atol=1e-12,
    )
    target = np.asarray(independent_props.probabilities)
    independent_counts: Counter[int] = Counter()
    historical_counts: Counter[int] = Counter()
    independent_streams = rng_streams(
        "event-independent", (50991, 50992, 51001, 50994, 50995, 50996)
    )
    historical_source = NumpyUniformSource(np.random.default_rng(51002))
    for index in range(1, draw_count + 1):
        independent = sample_update(
            state,
            beta=beta,
            specification=specification,
            rng_streams=independent_streams,
            generation_index_one_based=1,
            step_index_one_based=index,
            model_time_before=None,
        )
        historical = historical_single_event(
            state,
            beta=beta,
            rho=specification.rho,
            k_f=specification.k_f,
            k_b=specification.k_b,
            uniform_source=historical_source,
            event_number=index,
        )
        if independent.selected_event_index_zero_based is None:
            raise AssertionError("Categorical event did not record an event index.")
        independent_counts[independent.selected_event_index_zero_based] += 1
        historical_counts[historical.event_index_zero_based] += 1
    return {
        "comparisonId": "S05-C02",
        "drawsPerEngine": draw_count,
        "targetProbabilities": target.tolist(),
        "twoEngineTotalVariation": total_variation(
            independent_counts, historical_counts
        ),
        "independentTargetTotalVariation": target_total_variation(
            independent_counts, target
        ),
        "historicalTargetTotalVariation": target_total_variation(
            historical_counts, target
        ),
        "counts": counter_rows(independent_counts, historical_counts),
        "elapsedSeconds": time.perf_counter() - started,
    }


def run_fission_comparison(draw_count: int, *, odd: bool) -> dict[str, Any]:
    started = time.perf_counter()
    specification = load_profile(HISTORICAL_PROFILE)
    parent = (3, 2, 2) if odd else (4, 3, 2, 1)
    if len(parent) != specification.n_species:
        raw = raw_profile(HISTORICAL_PROFILE)
        raw.update(
            specification_id=(
                "E01-S05-HISTORICAL-FISSION-EVEN-v1"
                if not odd
                else "E01-S05-HISTORICAL-FISSION-ODD-v1"
            ),
            n_species=len(parent),
            n_min=1,
            n_max=max(sum(parent), 2),
            rho=[1.0 / len(parent)] * len(parent),
        )
        specification = specification_from_mapping(raw)
    seed = 52011 if odd else 52001
    historical_seed = 52012 if odd else 52002
    independent_streams = rng_streams(
        "fission-independent",
        (seed - 41, seed - 31, seed - 21, seed - 11, seed, seed + 11),
    )
    historical_source = NumpyUniformSource(np.random.default_rng(historical_seed))
    independent_counts: Counter[Any] = Counter()
    historical_counts: Counter[Any] = Counter()
    independent_conservation = True
    historical_conservation = True
    for index in range(1, draw_count + 1):
        independent = fission(
            parent,
            specification=specification,
            rng_streams=independent_streams,
            generation_index_one_based=index,
        )
        historical = historical_fission(parent, uniform_source=historical_source)
        independent_key = (
            (independent.child_first, independent.discarded)
            if odd
            else independent.child_first
        )
        historical_key = (
            (historical.child_a, historical.discarded) if odd else historical.child_a
        )
        independent_counts[independent_key] += 1
        historical_counts[historical_key] += 1
        independent_conservation &= bool(
            np.array_equal(
                np.asarray(independent.child_first)
                + np.asarray(independent.child_second)
                + np.asarray(independent.discarded),
                parent,
            )
        )
        historical_conservation &= bool(
            np.array_equal(
                np.asarray(historical.child_a)
                + np.asarray(historical.child_b)
                + np.asarray(historical.discarded),
                parent,
            )
        )
    return {
        "comparisonId": "S05-C04" if odd else "S05-C03",
        "oddParent": odd,
        "parent": parent,
        "drawsPerEngine": draw_count,
        "twoEngineTotalVariation": total_variation(
            independent_counts, historical_counts
        ),
        "independentConservation": independent_conservation,
        "historicalConservation": historical_conservation,
        "supportSize": len(set(independent_counts) | set(historical_counts)),
        "counts": counter_rows(independent_counts, historical_counts),
        "elapsedSeconds": time.perf_counter() - started,
    }


def run_trajectory_comparison(draw_count: int) -> dict[str, Any]:
    started = time.perf_counter()
    specification = load_profile(HISTORICAL_PROFILE)
    beta = np.asarray(
        [[0.4, 0.1, 0.2], [0.2, 0.5, 0.1], [0.1, 0.3, 0.4]],
        dtype=np.float64,
    )
    initial = (1, 1, 1)
    independent_streams = rng_streams(
        "trajectory-independent", (52991, 52992, 53001, 52994, 53002, 52996)
    )
    historical_source = NumpyUniformSource(np.random.default_rng(53003))
    independent_endpoints: Counter[Any] = Counter()
    historical_endpoints: Counter[Any] = Counter()
    independent_event_counts: list[int] = []
    historical_event_counts: list[int] = []
    for index in range(1, draw_count + 1):
        independent = advance_generation(
            initial,
            beta=beta,
            specification=specification,
            rng_streams=independent_streams,
            generation_index_one_based=index,
        )
        historical = historical_advance_generation(
            initial,
            beta=beta,
            rho=specification.rho,
            k_f=specification.k_f,
            k_b=specification.k_b,
            n_max=specification.n_max,
            uniform_source=historical_source,
            event_guard=10000,
        )
        independent_endpoint = (
            ("terminal", independent.terminal_status)
            if independent.next_state is None
            else independent.next_state
        )
        historical_endpoint = (
            ("terminal", historical.terminal_status)
            if historical.next_state is None
            else historical.next_state
        )
        independent_endpoints[independent_endpoint] += 1
        historical_endpoints[historical_endpoint] += 1
        independent_event_counts.append(len(independent.growth.events))
        historical_event_counts.append(len(historical.growth.events))

    independent_values = np.asarray(independent_event_counts, dtype=np.float64)
    historical_values = np.asarray(historical_event_counts, dtype=np.float64)
    standard_error = float(
        np.sqrt(
            independent_values.var(ddof=1) / draw_count
            + historical_values.var(ddof=1) / draw_count
        )
    )
    difference = float(independent_values.mean() - historical_values.mean())
    standardized = (
        0.0
        if standard_error == 0.0 and difference == 0.0
        else abs(difference) / standard_error
    )
    return {
        "comparisonId": "S05-C05",
        "drawsPerEngine": draw_count,
        "endpointTwoEngineTotalVariation": total_variation(
            independent_endpoints, historical_endpoints
        ),
        "endpointCounts": counter_rows(independent_endpoints, historical_endpoints),
        "independentMeanEventCount": float(independent_values.mean()),
        "historicalMeanEventCount": float(historical_values.mean()),
        "meanEventCountDifference": difference,
        "meanEventCountStandardError": standard_error,
        "meanEventCountStandardizedDifference": standardized,
        "elapsedSeconds": time.perf_counter() - started,
    }


def run_comparisons(quick: bool, workers: int) -> list[dict[str, Any]]:
    sizes = comparison_sizes(quick)
    tasks = (
        (run_propensity_comparison, (sizes["propensity"],), {}),
        (run_event_comparison, (sizes["event"],), {}),
        (run_fission_comparison, (sizes["fission"],), {"odd": False}),
        (run_fission_comparison, (sizes["oddFission"],), {"odd": True}),
        (run_trajectory_comparison, (sizes["trajectory"],), {}),
    )
    if workers == 1:
        return [function(*args, **kwargs) for function, args, kwargs in tasks]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(function, *args, **kwargs)
            for function, args, kwargs in tasks
        ]
        return [future.result() for future in futures]


def effective_thresholds(quick: bool) -> dict[str, float]:
    thresholds = dict(
        load_yaml(PROFILES_PATH)["distributionalValidation"]["thresholds"]
    )
    if quick:
        thresholds.update(
            categoricalTwoEngineTotalVariation=0.12,
            categoricalPerEngineTargetTotalVariation=0.12,
            fixedFissionTwoEngineTotalVariation=0.18,
            oddFissionTwoEngineTotalVariation=0.18,
            trajectoryEndpointTwoEngineTotalVariation=0.18,
            trajectoryMeanEventCountStandardizedDifference=8.0,
        )
    return {key: float(value) for key, value in thresholds.items()}


def comparison_rows(
    comparisons: list[dict[str, Any]], quick: bool
) -> list[dict[str, Any]]:
    by_id = {item["comparisonId"]: item for item in comparisons}
    thresholds = effective_thresholds(quick)
    entries = (
        (
            "S05-C01",
            "propensity arrays",
            "overallMaximumAbsoluteError",
            "propensityMaximumAbsoluteError",
        ),
        (
            "S05-C02",
            "categorical event law",
            "twoEngineTotalVariation",
            "categoricalTwoEngineTotalVariation",
        ),
        (
            "S05-C02",
            "independent categorical versus analytical target",
            "independentTargetTotalVariation",
            "categoricalPerEngineTargetTotalVariation",
        ),
        (
            "S05-C02",
            "S04 categorical versus analytical target",
            "historicalTargetTotalVariation",
            "categoricalPerEngineTargetTotalVariation",
        ),
        (
            "S05-C03",
            "even fixed-size fission law",
            "twoEngineTotalVariation",
            "fixedFissionTwoEngineTotalVariation",
        ),
        (
            "S05-C04",
            "odd fixed-size fission/discard law",
            "twoEngineTotalVariation",
            "oddFissionTwoEngineTotalVariation",
        ),
        (
            "S05-C05",
            "one-generation endpoint law",
            "endpointTwoEngineTotalVariation",
            "trajectoryEndpointTwoEngineTotalVariation",
        ),
        (
            "S05-C05",
            "growth event-count mean",
            "meanEventCountStandardizedDifference",
            "trajectoryMeanEventCountStandardizedDifference",
        ),
    )
    rows = []
    for comparison_id, scope, metric, threshold_name in entries:
        observed = float(by_id[comparison_id][metric])
        threshold = thresholds[threshold_name]
        rows.append(
            {
                "comparisonId": comparison_id,
                "scope": scope,
                "branch": HISTORICAL_PROFILE,
                "metric": metric,
                "observed": observed,
                "thresholdName": threshold_name,
                "threshold": threshold,
                "passed": observed <= threshold,
                "comparisonBoundary": (
                    "Matched public historical branch; distributional only; "
                    "no exact trajectory, legacy MATLAB RNG, or author-code claim."
                ),
            }
        )
    return rows


def diagnostic_logs() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    payload = load_yaml(PROFILES_PATH)
    seeds = tuple(payload["validationRandomInputs"]["unitFixtureSeeds"])
    records = []
    checks = []
    for offset, profile_id in enumerate(payload["profiles"]):
        specification = load_profile(profile_id)
        streams = rng_streams(
            f"diagnostic-{offset}", tuple(seed + 10 * offset for seed in seeds)
        )
        beta = np.zeros((specification.n_species, specification.n_species))
        model_time = (
            None if specification.clock_semantics.value == "event_index_only" else 0.0
        )
        event = sample_update(
            [1, 1, 0],
            beta=beta,
            specification=specification,
            rng_streams=streams,
            generation_index_one_based=1,
            step_index_one_based=1,
            model_time_before=model_time,
        )
        split = fission(
            [2, 1, 1],
            specification=specification,
            rng_streams=streams,
            generation_index_one_based=1,
        )
        records.append(
            {
                "profile": profile_id,
                "rngInputs": streams.descriptions(),
                "event": event,
                "fission": split,
            }
        )
        state_valid = all(
            isinstance(value, int) and value >= 0 for value in event.post_state
        )
        conservation = np.array_equal(
            np.asarray(split.child_first)
            + np.asarray(split.child_second)
            + np.asarray(split.discarded),
            split.parent,
        )
        checks.extend(
            [
                {
                    "checkId": f"UNIT-{offset + 1:02d}-STATE",
                    "profile": profile_id,
                    "description": "Post-update state is nonnegative integer-valued.",
                    "passed": state_valid,
                },
                {
                    "checkId": f"UNIT-{offset + 1:02d}-FISSION",
                    "profile": profile_id,
                    "description": "Fission conserves parent including explicit discard.",
                    "passed": bool(conservation),
                },
                {
                    "checkId": f"UNIT-{offset + 1:02d}-LOG",
                    "profile": profile_id,
                    "description": "Event and fission diagnostic schema IDs are explicit.",
                    "passed": (
                        event.record_schema_version == "eidosoma.e01.s05_event_log.v1"
                        and split.record_schema_version
                        == "eidosoma.e01.s05_fission_log.v1"
                    ),
                },
            ]
        )

    hand_raw = raw_profile(HISTORICAL_PROFILE)
    hand_raw.update(
        specification_id="E01-S05-HAND-ORACLE-v1",
        n_species=2,
        n_min=1,
        n_max=4,
        k_f=0.1,
        k_b=0.2,
        rho=[0.25, 0.75],
    )
    hand = specification_from_mapping(hand_raw)
    propensities = calculate_propensities(
        [2, 1], beta=[[1.0, 2.0], [0.5, 0.0]], specification=hand
    )
    hand_passed = bool(
        np.allclose(propensities.boost, [7 / 3, 4 / 3], rtol=0.0, atol=1e-12)
        and np.allclose(propensities.join, [0.175, 0.3], rtol=0.0, atol=1e-12)
        and np.allclose(propensities.leave, [14 / 15, 4 / 15], rtol=0.0, atol=1e-12)
        and np.isclose(propensities.total, 1.675, rtol=0.0, atol=1e-12)
    )
    checks.append(
        {
            "checkId": "UNIT-04-HAND-PROPENSITIES",
            "profile": hand.specification_id,
            "description": "Hand-calculated boost/join/leave/total arrays agree.",
            "passed": hand_passed,
        }
    )

    sentinel_raw = raw_profile(HISTORICAL_PROFILE)
    sentinel_raw["k_f"] = "UNRESOLVED::E01-A009"
    sentinel_rejected = False
    try:
        specification_from_mapping(sentinel_raw)
    except SpecificationError:
        sentinel_rejected = True
    checks.append(
        {
            "checkId": "UNIT-05-SENTINEL-REJECTION",
            "profile": "registry execution gate",
            "description": "Raw unresolved registry sentinel is rejected.",
            "passed": sentinel_rejected,
        }
    )
    return to_builtin(records), checks


def source_independence_checks(contract: dict[str, Any]) -> list[dict[str, Any]]:
    checks = []
    forbidden = contract["implementationBoundary"]["forbiddenImport"]
    for path in PACKAGE_FILES:
        checks.append(
            {
                "checkId": f"INDEPENDENCE-{path.name}",
                "path": str(path.relative_to(REPOSITORY_ROOT)),
                "forbiddenToken": forbidden,
                "passed": forbidden not in path.read_text(encoding="utf-8"),
            }
        )
    checks.append(
        {
            "checkId": "INDEPENDENCE-MODULE-SET",
            "path": str(PACKAGE_ROOT.relative_to(REPOSITORY_ROOT)),
            "forbiddenToken": forbidden,
            "passed": len(PACKAGE_FILES) == 5,
        }
    )
    return checks


def frozen_evidence_checks(
    contract: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checks = []
    for key, record in contract["frozenEvidence"].items():
        path = Path(record["path"])
        actual = sha256(path)
        checks.append(
            {
                "checkId": f"FROZEN-{key}",
                "path": str(path),
                "expectedSha256": record["sha256"],
                "actualSha256": actual,
                "passed": actual == record["sha256"],
            }
        )
    registry_record = contract["frozenEvidence"]["specificationRegistry"]
    registry_path = Path(registry_record["path"])
    registry = load_yaml(registry_path)
    gate = registry["executionGate"]
    preservation = {
        "schema": "eidosoma.e01.s05_registry_preservation.v1",
        "researchStepId": "S05",
        "registryPath": str(registry_path),
        "registryVersion": registry["registryVersion"],
        "expectedSha256": registry_record["sha256"],
        "actualSha256": sha256(registry_path),
        "unchanged": sha256(registry_path) == registry_record["sha256"],
        "parameterCount": len(registry["parameters"]),
        "unresolvedParameterCount": gate["unresolvedParameterCount"],
        "unexpandedBranchSetCount": gate["unexpandedBranchSetCount"],
        "executable": gate["executable"],
        "noSilentDefaults": gate["noSilentDefaults"],
        "s05RegistryUpdates": [],
        "interpretation": (
            "S05 profiles are validation branch instances only; no v0.3.0 "
            "parameter, sentinel, conflict, or branch set was changed."
        ),
    }
    checks.append(
        {
            "checkId": "FROZEN-REGISTRY-GATE",
            "path": str(registry_path),
            "passed": (
                preservation["parameterCount"] == 120
                and preservation["unresolvedParameterCount"] == 64
                and preservation["unexpandedBranchSetCount"] == 21
                and preservation["executable"] is False
                and preservation["noSilentDefaults"] is True
                and preservation["s05RegistryUpdates"] == []
            ),
        }
    )
    return checks, preservation


def write_comparison_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_branch_catalog(path: Path, contract: dict[str, Any]) -> None:
    rows = []
    for dimension, values in contract["implementedBranches"].items():
        for value in values:
            rows.append(
                {
                    "dimension": dimension,
                    "branch": value,
                    "implemented": True,
                    "authorDefaultClaimed": False,
                    "crossS04Eligible": (
                        value
                        in {
                            "historical_reference",
                            "historical_orientation_with_diagonal",
                            "categorical_single_event",
                            "event_index_only",
                            "eventwise_zero_rate",
                            "eventwise_exact_stop",
                            "unbounded_historical_comparison",
                            "fixed_size_without_replacement_odd_discard",
                            "first",
                            "continue_exact_selected",
                            "with_replacement_counts",
                        }
                    ),
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def api_surface() -> dict[str, Any]:
    import e01_gard_independent as package

    functions = {}
    for name in public_api:
        value = getattr(package, name)
        if inspect.isfunction(value):
            functions[name] = str(inspect.signature(value))
    specification_fields = [
        {
            "name": field.name,
            "hasDefault": not (
                field.default is MISSING and field.default_factory is MISSING
            ),
        }
        for field in fields(GardSpecification)
    ]
    return {
        "schema": "eidosoma.e01.s05_api_surface.v1",
        "researchStepId": "S05",
        "engineVersion": engine_version,
        "publicSymbols": sorted(public_api),
        "functionSignatures": functions,
        "specificationFields": specification_fields,
        "allSpecificationFieldsRequired": not any(
            item["hasDefault"] for item in specification_fields
        ),
    }


def build_manifest(
    artifact_root: Path,
    step_dir: Path,
    shared_dir: Path,
    git_commit: str,
) -> dict[str, Any]:
    inputs = [
        Path("/workspace/AGENTS.md"),
        Path("/workspace/FULL_PLAN.md"),
        Path("/workspace/RESEARCH_PLAN.md"),
        Path("/workspace/input-attachments/MANIFEST.json"),
        Path(
            "/workspace/input-attachments/ed5486bf-a043-485b-a233-d88d8d123759/"
            "_metadata/ATTACHMENT.md"
        ),
        Path(
            "/workspace/input-attachments/ed5486bf-a043-485b-a233-d88d8d123759/"
            "pdf-markdown.md"
        ),
        Path("/artifacts/research_steps/S01/research_step_full_results.md"),
        Path("/artifacts/research_steps/S02/research_step_full_results.md"),
        Path("/artifacts/research_steps/S03/research_step_full_results.md"),
        Path("/artifacts/research_steps/S04/research_step_full_results.md"),
    ]
    contract = load_yaml(CONTRACT_PATH)
    inputs.extend(Path(item["path"]) for item in contract["frozenEvidence"].values())
    code = [CONTRACT_PATH, PROFILES_PATH, TEST_PATH, Path(__file__), *PACKAGE_FILES]
    outputs = sorted(
        path
        for directory in (step_dir, shared_dir)
        for path in directory.glob("*")
        if path.is_file() and path.name != "artifact_manifest.json"
    )

    def record(path: Path, role: str) -> dict[str, Any]:
        return {
            "path": str(path),
            "role": role,
            "sizeBytes": path.stat().st_size,
            "sha256": sha256(path),
        }

    return {
        "schema": "eidosoma.e01.s05_artifact_manifest.v1",
        "researchStepId": "S05",
        "generatedOn": "2026-08-01",
        "gitCommit": git_commit,
        "inputs": [record(path, "input") for path in inputs],
        "repositoryCode": [record(path, "repository_code") for path in code],
        "outputs": [record(path, "output") for path in outputs],
        "artifactRoot": str(artifact_root),
        "selfHashExcluded": True,
    }


def build(artifact_root: Path, *, quick: bool, workers: int) -> dict[str, Any]:
    if workers < 1 or workers > 8:
        raise ValueError("workers must be between 1 and 8.")
    step_dir = artifact_root / "research_steps/S05"
    shared_dir = (
        artifact_root / "E01_forensic_replication_bundle/software/independent_engine"
    )
    step_dir.mkdir(parents=True, exist_ok=True)
    shared_dir.mkdir(parents=True, exist_ok=True)
    contract = load_yaml(CONTRACT_PATH)

    comparisons = run_comparisons(quick, workers)
    rows = comparison_rows(comparisons, quick)
    diagnostic, unit_checks = diagnostic_logs()
    independence = source_independence_checks(contract)
    frozen_checks, registry_preservation = frozen_evidence_checks(contract)
    all_checks = unit_checks + independence + frozen_checks
    success = all(row["passed"] for row in rows) and all(
        check["passed"] for check in all_checks
    )

    comparison_csv = step_dir / "distributional_agreement.csv"
    write_comparison_csv(comparison_csv, rows)
    write_json(
        step_dir / "distributional_agreement_details.json",
        {
            "schema": "eidosoma.e01.s05_distributional_agreement.v1",
            "researchStepId": "S05",
            "quickMode": quick,
            "matchedBranch": HISTORICAL_PROFILE,
            "comparisons": comparisons,
            "gates": rows,
            "allPassed": all(row["passed"] for row in rows),
            "claimBoundary": (
                "Distributional agreement only for matched explicit public historical "
                "semantics; no exact cross-RNG trajectory or author-code identity."
            ),
        },
    )
    write_json(
        step_dir / "unit_invariants.json",
        {
            "schema": "eidosoma.e01.s05_unit_invariants.v1",
            "researchStepId": "S05",
            "checks": all_checks,
            "passedCount": sum(check["passed"] for check in all_checks),
            "checkCount": len(all_checks),
            "allPassed": all(check["passed"] for check in all_checks),
        },
    )
    write_json(step_dir / "registry_preservation.json", registry_preservation)
    write_json(shared_dir / "diagnostic_event_log_fixture.json", diagnostic)
    write_json(shared_dir / "api_surface.json", api_surface())
    shutil.copyfile(CONTRACT_PATH, shared_dir / "independent_engine_contract.yaml")
    shutil.copyfile(PROFILES_PATH, shared_dir / "validation_profiles.yaml")
    write_branch_catalog(shared_dir / "branch_catalog.csv", contract)

    git_commit = git_output("rev-parse", "HEAD")
    git_branch = git_output("branch", "--show-current")
    git_dirty = bool(git_output("status", "--porcelain"))
    package_hashes = {
        str(path.relative_to(REPOSITORY_ROOT)): sha256(path) for path in PACKAGE_FILES
    }
    write_json(
        shared_dir / "engine_pointer.json",
        {
            "schema": "eidosoma.e01.s05_engine_pointer.v1",
            "researchStepId": "S05",
            "engineVersion": engine_version,
            "repository": str(REPOSITORY_ROOT),
            "branch": git_branch,
            "commit": git_commit,
            "worktreeDirtyAtGeneration": git_dirty,
            "package": "src/e01_gard_independent",
            "packageFileSha256": package_hashes,
            "contract": "configs/e01/s05_independent_contract.yaml",
            "profiles": "configs/e01/s05_specification_profiles.yaml",
            "authorImplementation": "UNAVAILABLE::NO_AUTHOR_CODE_RELEASE_FOUND",
            "identityBoundary": (
                "Independent E01 implementation; not S04 control flow, legacy MATLAB "
                "RNG, modern GARD source, or unavailable author implementation."
            ),
        },
    )
    write_json(
        shared_dir / "benchmark.json",
        {
            "schema": "eidosoma.e01.s05_benchmark.v1",
            "researchStepId": "S05",
            "quickMode": quick,
            "workerProcesses": workers,
            "threadEnvironment": {
                name: os.environ.get(name, "UNSET")
                for name in (
                    "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMBA_NUM_THREADS",
                )
            },
            "tasks": [
                {
                    "comparisonId": item["comparisonId"],
                    "elapsedSeconds": item["elapsedSeconds"],
                    "workUnits": item.get("drawsPerEngine", item.get("caseCount", 0)),
                    "workUnitsPerSecondPerEngine": (
                        item.get("drawsPerEngine", item.get("caseCount", 0))
                        / item["elapsedSeconds"]
                    ),
                    "interpretation": "Diagnostic runtime only; not a pass criterion.",
                }
                for item in comparisons
            ],
        },
    )

    artifact_paths = [
        str(comparison_csv),
        str(step_dir / "distributional_agreement_details.json"),
        str(step_dir / "unit_invariants.json"),
        str(step_dir / "registry_preservation.json"),
        str(shared_dir / "diagnostic_event_log_fixture.json"),
        str(shared_dir / "api_surface.json"),
        str(shared_dir / "independent_engine_contract.yaml"),
        str(shared_dir / "validation_profiles.yaml"),
        str(shared_dir / "branch_catalog.csv"),
        str(shared_dir / "engine_pointer.json"),
        str(shared_dir / "benchmark.json"),
    ]
    report_path = step_dir / "research_step_full_results.md"
    if report_path.is_file():
        artifact_paths.append(str(report_path))
    validation = {
        "researchStepId": "S05",
        "stepNumber": 5,
        "success": success,
        "status": "complete" if success and not quick else "quick_validation_only",
        "artifactsWritten": artifact_paths,
        "validationResult": (
            f"PASS: {sum(check['passed'] for check in all_checks)}/{len(all_checks)} "
            f"unit/independence/provenance checks and "
            f"{sum(row['passed'] for row in rows)}/{len(rows)} matched-branch "
            "distributional gates passed."
            if success
            else "FAIL: one or more unit, provenance, or distributional gates failed."
        ),
        "outcomeClassification": "supportive"
        if success and not quick
        else "not_classified_quick_mode",
        "caveatsOrBlockers": [
            "No identity claim with the unavailable author implementation.",
            "No exact legacy MATLAB or cross-RNG trajectory claim.",
            "Registry v0.3.0 remains closed; S05 profiles are validation-only branch instances.",
            "Paper vector-Poisson and modern Gillespie fixtures were not compared to S04.",
            "Canonical seed derivation and serialization remain S06 and were not begun.",
        ],
        "recommendedNextAction": (
            "Stop after S05 and return control; execute S06 only if separately authorized."
        ),
        "quickMode": quick,
        "workerProcesses": workers,
        "unitValidation": {
            "passedCount": sum(check["passed"] for check in all_checks),
            "checkCount": len(all_checks),
        },
        "distributionalValidation": {
            "passedCount": sum(row["passed"] for row in rows),
            "gateCount": len(rows),
            "matchedBranch": HISTORICAL_PROFILE,
        },
        "registryValidation": registry_preservation,
    }
    write_json(step_dir / "validation_summary.json", validation)
    validation["artifactsWritten"].append(str(step_dir / "validation_summary.json"))
    write_json(step_dir / "validation_summary.json", validation)
    manifest = build_manifest(artifact_root, step_dir, shared_dir, git_commit)
    write_json(step_dir / "artifact_manifest.json", manifest)
    return validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build(args.artifacts_dir.resolve(), quick=args.quick, workers=args.workers)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
