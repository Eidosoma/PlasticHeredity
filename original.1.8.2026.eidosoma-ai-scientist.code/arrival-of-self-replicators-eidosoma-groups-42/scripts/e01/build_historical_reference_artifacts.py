#!/usr/bin/env python3
"""Validate the S04 historical-reference engine and write compact artifacts."""

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
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from e01_gard_historical import (
    HistoricalSourceDomainError,
    UniformTape,
    advance_one_generation,
    catalytic_matrix_from_standard_normals,
    compute_propensities,
    grow_to_split_size,
    historical_h,
    historical_initial_state_with_replacement,
    historical_nondrift_technique1,
    historical_nondrift_technique2,
    historical_single_event,
    simulate_lineage,
    split_fixed_size_without_replacement,
)
from e01_gard_historical import __version__ as engine_version

CONTRACT_PATH = REPOSITORY_ROOT / "configs/e01/s04_historical_contract.yaml"
FIXTURES_PATH = REPOSITORY_ROOT / "configs/e01/s04_small_cases.yaml"
ENGINE_PATHS = [
    REPOSITORY_ROOT / "src/e01_gard_historical/__init__.py",
    REPOSITORY_ROOT / "src/e01_gard_historical/engine.py",
    REPOSITORY_ROOT / "src/e01_gard_historical/nondrift.py",
]


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


def git_output(*args: str, workdir: Path = REPOSITORY_ROOT) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=workdir,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


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


def compare(actual: Any, expected: Any, path: str = "root") -> list[str]:
    actual = to_builtin(actual)
    expected = to_builtin(expected)
    if isinstance(expected, bool):
        return (
            []
            if isinstance(actual, bool) and actual is expected
            else [f"{path}: {actual!r} != {expected!r}"]
        )
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not isinstance(actual, (int, float)) or isinstance(actual, bool):
            return [f"{path}: nonnumeric actual {actual!r}"]
        if not np.isclose(float(actual), float(expected), rtol=1e-12, atol=1e-12):
            return [f"{path}: {actual!r} != {expected!r}"]
        return []
    if isinstance(expected, str):
        return [] if actual == expected else [f"{path}: {actual!r} != {expected!r}"]
    if expected is None:
        return [] if actual is None else [f"{path}: {actual!r} is not None"]
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return [f"{path}: expected list, got {type(actual).__name__}"]
        errors = []
        if len(actual) != len(expected):
            errors.append(f"{path}: length {len(actual)} != {len(expected)}")
            return errors
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            errors.extend(compare(left, right, f"{path}[{index}]"))
        return errors
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [f"{path}: expected mapping, got {type(actual).__name__}"]
        errors = []
        for key, value in expected.items():
            if key not in actual:
                errors.append(f"{path}: missing key {key!r}")
            else:
                errors.extend(compare(actual[key], value, f"{path}.{key}"))
        return errors
    return [] if actual == expected else [f"{path}: {actual!r} != {expected!r}"]


def camel_result(**values: Any) -> dict[str, Any]:
    return {key: to_builtin(value) for key, value in values.items()}


def run_case(case: dict[str, Any], cases: dict[str, dict[str, Any]]) -> dict[str, Any]:
    kind = case["kind"]
    inputs = case["inputs"]
    if kind == "catalytic_matrix":
        beta = catalytic_matrix_from_standard_normals(
            inputs["standardNormals"], a=inputs["a"], sigma=inputs["sigma"]
        )
        actual = camel_result(
            beta=beta, diagonalRetained=bool(np.all(np.diag(beta) > 0))
        )
    elif kind == "propensities":
        props = compute_propensities(
            inputs["state"],
            beta=inputs["beta"],
            rho=inputs["rho"],
            k_f=inputs["kF"],
            k_b=inputs["kB"],
        )
        actual = camel_result(
            boost=props.boost,
            join=props.join,
            leave=props.leave,
            concatenated=props.concatenated,
            total=props.total,
        )
    elif kind == "weighted_event":
        parent_case = cases[inputs["propensityCase"]]
        parent_inputs = parent_case["inputs"]
        event = historical_single_event(
            parent_inputs["state"],
            beta=parent_inputs["beta"],
            rho=parent_inputs["rho"],
            k_f=parent_inputs["kF"],
            k_b=parent_inputs["kB"],
            uniform_source=UniformTape((inputs["uniformDraw"],)),
        )
        actual = camel_result(
            eventIndexZeroBased=event.event_index_zero_based,
            kind=event.kind,
            postState=event.post_state,
            massDelta=event.mass_delta,
        )
    elif kind == "growth":
        tape = UniformTape(tuple(inputs["uniformDraws"]))
        result = grow_to_split_size(
            inputs["state"],
            beta=inputs["beta"],
            rho=inputs["rho"],
            k_f=inputs["kF"],
            k_b=inputs["kB"],
            n_max=inputs["nMax"],
            uniform_source=tape,
            event_guard=inputs["eventGuard"],
        )
        actual = camel_result(
            finalState=result.final_state,
            terminalStatus=result.terminal_status,
            historicalSignedSpecies=[
                event.historical_signed_species for event in result.events
            ],
            eventMassDeltas=[event.mass_delta for event in result.events],
            eventTotalRates=[event.total_rate for event in result.events],
            legacyDtAccumulator=result.legacy_dt_accumulator,
            legacyInverseRateSum=result.legacy_inverse_rate_sum,
            drawsConsumed=tape.consumed,
        )
    elif kind == "fission":
        tape = UniformTape(tuple(inputs["uniformDraws"]))
        result = split_fixed_size_without_replacement(
            inputs["parent"], uniform_source=tape
        )
        actual = camel_result(
            childA=result.child_a,
            childB=result.child_b,
            discarded=result.discarded,
            childMasses=[sum(result.child_a), sum(result.child_b)],
            followedDaughter=result.followed_daughter,
            daughterSelectionRule=result.daughter_selection_rule,
            selections=result.selections,
            drawsConsumed=tape.consumed,
        )
    elif kind == "generation":
        tape = UniformTape(tuple(inputs["uniformDraws"]))
        result = advance_one_generation(
            inputs["state"],
            beta=inputs["beta"],
            rho=inputs["rho"],
            k_f=inputs["kF"],
            k_b=inputs["kB"],
            n_max=inputs["nMax"],
            uniform_source=tape,
            event_guard=inputs["eventGuard"],
        )
        if result.fission is None:
            raise AssertionError("Generation fixture unexpectedly became extinct.")
        actual = camel_result(
            preFissionState=result.growth.final_state,
            childA=result.fission.child_a,
            childB=result.fission.child_b,
            nextState=result.next_state,
            terminalStatus=result.terminal_status,
            drawsConsumed=tape.consumed,
        )
    elif kind == "lineage":
        tape = UniformTape(tuple(inputs["uniformDraws"]))
        result = simulate_lineage(
            inputs["state"],
            beta=inputs["beta"],
            rho=inputs["rho"],
            k_f=inputs["kF"],
            k_b=inputs["kB"],
            n_max=inputs["nMax"],
            n_generations=inputs["nGenerations"],
            uniform_source=tape,
            event_guard_per_generation=inputs["eventGuardPerGeneration"],
        )
        actual = camel_result(
            preFissionTrace=result.pre_fission_trace,
            finalState=result.final_state,
            completedFissions=result.completed_fissions,
            terminalStatus=result.terminal_status,
            drawsConsumed=tape.consumed,
        )
    elif kind == "h_similarity":
        actual = camel_result(h=historical_h(inputs["set1"], inputs["set2"]))
    elif kind == "nondrift_technique1":
        result = historical_nondrift_technique1(
            inputs["trace"], threshold=inputs["threshold"]
        )
        actual = camel_result(
            angles=result.angles,
            localScores=result.local_scores,
            isNonDrift=result.is_non_drift,
            activeGenerationCount=result.active_generation_count,
            firstZeroSumGenerationOneBased=result.first_zero_sum_generation_one_based,
        )
    elif kind == "nondrift_technique2":
        result = historical_nondrift_technique2(
            inputs["trace"],
            threshold=inputs["threshold"],
            drift_size=inputs["driftSize"],
        )
        actual = camel_result(angles=result.angles, isNonDrift=result.is_non_drift)
    elif kind == "expected_source_domain_error":
        try:
            if inputs["operation"] != "nondrift_technique2":
                raise AssertionError(
                    f"Unknown error fixture operation {inputs['operation']!r}."
                )
            historical_nondrift_technique2(
                inputs["trace"],
                threshold=inputs["threshold"],
                drift_size=inputs["driftSize"],
            )
        except HistoricalSourceDomainError as exc:
            actual = camel_result(
                exception=type(exc).__name__,
                classification="SOURCE_EDGE_CASE_PRESERVED_NOT_REPAIRED",
                message=str(exc),
            )
        else:
            actual = {"exception": "NONE"}
    elif kind == "initializer":
        tape = UniformTape(tuple(inputs["uniformDraws"]))
        state = historical_initial_state_with_replacement(
            n_g=inputs["nG"], n_min=inputs["nMin"], uniform_source=tape
        )
        actual = camel_result(
            state=state,
            distinctTypeCount=sum(value > 0 for value in state),
            totalMass=sum(state),
            drawsConsumed=tape.consumed,
        )
    else:
        raise AssertionError(f"Unknown fixture kind {kind!r}.")

    errors = compare(actual, case["expected"], case["caseId"])
    return {
        "caseId": case["caseId"],
        "kind": kind,
        "passed": not errors,
        "expected": to_builtin(case["expected"]),
        "actual": actual,
        "errors": errors,
    }


def validate_sources(contract: dict[str, Any]) -> dict[str, Any]:
    source = contract["sourceIdentity"]
    source_root = Path(source["localPath"])
    commit = git_output("rev-parse", "HEAD^{commit}", workdir=source_root)
    tree = git_output("rev-parse", "HEAD^{tree}", workdir=source_root)
    status = git_output("status", "--porcelain", workdir=source_root)
    file_results = []
    by_file = {item["path"]: item for item in contract["sourceFiles"]}
    errors = []
    if commit != source["commit"]:
        errors.append(f"historical commit {commit} != {source['commit']}")
    if tree != source["tree"]:
        errors.append(f"historical tree {tree} != {source['tree']}")
    if status:
        errors.append("historical source checkout is dirty")
    for item in contract["sourceFiles"]:
        path = source_root / item["path"]
        observed = sha256(path) if path.is_file() else None
        passed = observed == item["sha256"]
        if not passed:
            errors.append(f"source hash mismatch: {item['path']}")
        file_results.append(
            {
                "path": item["path"],
                "expectedSha256": item["sha256"],
                "observedSha256": observed,
                "passed": passed,
            }
        )

    mapping_results = []
    for mapping in contract["sourceLineMappings"]:
        item = by_file[mapping["sourceFile"]]
        path = source_root / mapping["sourceFile"]
        line_count = len(path.read_text(encoding="utf-8").splitlines())
        bounds_valid = 1 <= mapping["lineStart"] <= mapping["lineEnd"] <= line_count
        if not bounds_valid:
            errors.append(f"invalid source line range: {mapping['behaviorId']}")
        mapping_results.append(
            {
                "behaviorId": mapping["behaviorId"],
                "sourceFile": mapping["sourceFile"],
                "lineStart": mapping["lineStart"],
                "lineEnd": mapping["lineEnd"],
                "sourceFileSha256": item["sha256"],
                "lineCount": line_count,
                "boundsValid": bounds_valid,
                "portSymbols": mapping["portSymbols"],
                "finding": mapping["finding"],
            }
        )
    return {
        "passed": not errors,
        "commit": commit,
        "tree": tree,
        "cleanCheckout": not status,
        "fileResults": file_results,
        "mappingResults": mapping_results,
        "errors": errors,
    }


def validate_registry(contract: dict[str, Any]) -> dict[str, Any]:
    registry_meta = contract["specificationRegistry"]
    registry_path = Path(registry_meta["path"])
    registry = load_yaml(registry_path)
    observed_hash = sha256(registry_path)
    s04_parameters = [
        parameter
        for parameter in registry["parameters"]
        if parameter.get("ownerStep") == "S04"
    ]
    registered = {parameter["parameter"]: parameter for parameter in s04_parameters}
    mapped = {item["parameter"]: item for item in contract["s04RegistryMappings"]}
    exact_mapping = set(registered) == set(mapped)
    sentinel_results = []
    for name, parameter in registered.items():
        action = mapped[name]["registryAction"]
        value = str(parameter["value"])
        if "SENTINEL" in action:
            preserved = value.startswith("UNRESOLVED::")
        elif "BRANCH_SET" in action:
            preserved = value.startswith("BRANCH_SET::")
        else:
            preserved = True
        sentinel_results.append(
            {
                "parameter": name,
                "ambiguityId": parameter.get("ambiguityId"),
                "value": parameter["value"],
                "resolutionStatus": parameter["resolutionStatus"],
                "registryAction": action,
                "historicalFinding": mapped[name]["historicalFinding"],
                "preserved": preserved,
            }
        )
    errors = []
    if observed_hash != registry_meta["sha256"]:
        errors.append("specification registry SHA-256 changed")
    if len(registry["parameters"]) != registry_meta["parameterCount"]:
        errors.append("specification registry parameter count changed")
    if len(s04_parameters) != registry_meta["s04OwnedParameterCount"]:
        errors.append("S04-owned parameter count changed")
    if not exact_mapping:
        errors.append("S04 registry/contract parameter mapping is not exact")
    if not all(item["preserved"] for item in sentinel_results):
        errors.append("one or more registry sentinels/branch sets were not preserved")
    gate = registry["executionGate"]
    if gate["executable"] or not gate["noSilentDefaults"]:
        errors.append("registry execution/no-silent-default gate changed")
    return {
        "passed": not errors,
        "registryPath": str(registry_path),
        "expectedSha256": registry_meta["sha256"],
        "observedSha256": observed_hash,
        "parameterCount": len(registry["parameters"]),
        "s04OwnedParameterCount": len(s04_parameters),
        "mappingExact": exact_mapping,
        "executionGate": gate,
        "sentinelAndBranchPreservation": sentinel_results,
        "errors": errors,
    }


def validate_manifest(contract: dict[str, Any]) -> dict[str, Any]:
    metadata = contract["sourceManifest"]
    path = Path(metadata["path"])
    manifest = load_yaml(path)
    observed_hash = sha256(path)
    historical = next(
        repository
        for repository in manifest["repositories"]
        if repository["sourceId"] == "gard_historical"
    )
    modern = next(
        repository
        for repository in manifest["repositories"]
        if repository["sourceId"] == "gard_modern"
    )
    comparisons = {item["comparisonId"]: item for item in manifest["sourceComparisons"]}
    errors = []
    if observed_hash != metadata["sha256"]:
        errors.append("source manifest SHA-256 changed")
    if historical["commit"] != contract["sourceIdentity"]["commit"]:
        errors.append("historical manifest commit does not match contract")
    if historical["tree"] != contract["sourceIdentity"]["tree"]:
        errors.append("historical manifest tree does not match contract")
    if comparisons["S03-SC01"]["result"] != "CONFLICT_PRESERVED":
        errors.append("historical/modern growth conflict was not preserved")
    if comparisons["S03-SC02"]["result"] != "BYTE_IDENTICAL":
        errors.append("historical/modern non-drift identity changed")
    return {
        "passed": not errors,
        "sourceManifestPath": str(path),
        "expectedSha256": metadata["sha256"],
        "observedSha256": observed_hash,
        "historicalCommit": historical["commit"],
        "historicalTree": historical["tree"],
        "historicalLicense": historical["license"],
        "modernCommit": modern["commit"],
        "growthComparison": comparisons["S03-SC01"],
        "nonDriftComparison": comparisons["S03-SC02"],
        "authorCodeSentinel": manifest["authorCodeSearch"]["result"],
        "errors": errors,
    }


def validate_api() -> dict[str, Any]:
    from e01_gard_historical import engine as engine_module
    from e01_gard_historical import nondrift as nondrift_module

    required_parameters = {
        "catalytic_matrix_from_standard_normals": ["a", "sigma"],
        "catalytic_matrix_from_numpy_rng_explicit": [
            "n_g",
            "a",
            "sigma",
            "generator",
        ],
        "compute_propensities": ["beta", "rho", "k_f", "k_b"],
        "historical_single_event": ["beta", "rho", "k_f", "k_b", "uniform_source"],
        "grow_to_split_size": [
            "beta",
            "rho",
            "k_f",
            "k_b",
            "n_max",
            "uniform_source",
            "event_guard",
        ],
        "split_fixed_size_without_replacement": ["uniform_source"],
        "advance_one_generation": [
            "beta",
            "rho",
            "k_f",
            "k_b",
            "n_max",
            "uniform_source",
            "event_guard",
        ],
        "simulate_lineage": [
            "beta",
            "rho",
            "k_f",
            "k_b",
            "n_max",
            "n_generations",
            "uniform_source",
            "event_guard_per_generation",
        ],
        "historical_initial_state_with_replacement": [
            "n_g",
            "n_min",
            "uniform_source",
        ],
    }
    checks = []
    errors = []
    for symbol, required in required_parameters.items():
        signature = inspect.signature(getattr(engine_module, symbol))
        missing = []
        defaulted = []
        for parameter_name in required:
            if parameter_name not in signature.parameters:
                missing.append(parameter_name)
            elif (
                signature.parameters[parameter_name].default
                is not inspect.Parameter.empty
            ):
                defaulted.append(parameter_name)
        passed = not missing and not defaulted
        if not passed:
            errors.append(f"{symbol}: missing={missing}; defaulted={defaulted}")
        checks.append(
            {
                "symbol": symbol,
                "requiredInputs": required,
                "missing": missing,
                "silentlyDefaulted": defaulted,
                "passed": passed,
            }
        )
    nondrift_required = {
        "historical_nondrift_technique1": ["threshold"],
        "historical_nondrift_technique2": ["threshold", "drift_size"],
    }
    for symbol, required in nondrift_required.items():
        signature = inspect.signature(getattr(nondrift_module, symbol))
        missing = [name for name in required if name not in signature.parameters]
        defaulted = [
            name
            for name in required
            if name in signature.parameters
            and signature.parameters[name].default is not inspect.Parameter.empty
        ]
        passed = not missing and not defaulted
        if not passed:
            errors.append(f"{symbol}: missing={missing}; defaulted={defaulted}")
        checks.append(
            {
                "symbol": symbol,
                "requiredInputs": required,
                "missing": missing,
                "silentlyDefaulted": defaulted,
                "passed": passed,
            }
        )
    return {"passed": not errors, "checks": checks, "errors": errors}


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(row.get(key), sort_keys=True)
                    if isinstance(row.get(key), (list, dict))
                    else row.get(key)
                    for key in fieldnames
                }
            )


def compatibility_markdown(
    contract: dict[str, Any],
    fixture_summary: dict[str, Any],
    validation_passed: bool,
) -> str:
    boundary_rows = "\n".join(
        f"| {item['boundaryId']} | {item['subject']} | {item['historicalBehavior']} | "
        f"{item['paperOrModernBehavior']} |"
        for item in contract["compatibilityBoundaries"]
    )
    return f"""# S04 historical-reference compatibility notes

## Top summary

| Field | Result |
| --- | --- |
| Research step ID | **S04** |
| Completion status | **{"Complete" if validation_passed else "Blocked"}** for the historical-reference compatibility layer; S05 was not begun. |
| Artifacts written | Engine pointer/manifest, source traceability, compatibility matrix, {fixture_summary["caseCount"]} verified small cases, registry-preservation audit, validation summary, artifact manifest, and the canonical full-results report. |
| Validation result | **{"PASS" if validation_passed else "FAIL"}** — {fixture_summary["passedCount"]}/{fixture_summary["caseCount"]} fixtures passed; source, registry, and API no-default checks are included in the S04 validation summary. |
| Outcome classification | **Supportive** for translating the pinned public historical behavior; constraining differences from the paper are preserved. |
| Caveats or blockers | Legacy MATLAB RNG equality is unresolved; historical GARD has no detected license; author code is unavailable; the historical source lacks paper max-steps/vector-Poisson semantics and uses different fission/initialization behavior. |
| Recommended next action | Stop after S04 and return control. If separately authorized, S05 should implement an independent engine without copying this control flow and compare only declared model-level branches. |

## Lay summary

This layer reproduces what the pinned public 2014 GARD v10 source actually does on small, explicit-draw cases. It does not say that the paper's authors used that source. Important differences—especially one-event updates, fixed-size fission, and hidden legacy random-state order—remain visible rather than being smoothed into a single “GARD” implementation.

## Compatibility matrix

| ID | Subject | Pinned historical behavior | Paper, modern, or unresolved boundary |
| --- | --- | --- | --- |
{boundary_rows}

## Operational rules

- All kinetic values, reservoir vectors, split sizes, matrices, and draw sources are required API inputs. There is no implicit paper profile.
- `UniformTape` is the exact fixture path. It supplies draws directly and therefore bypasses—but does not solve—legacy MATLAB RNG identity.
- `NumpyUniformSource` and `catalytic_matrix_from_numpy_rng_explicit` are explicitly labeled distribution-compatible conveniences, not legacy MATLAB stream emulators.
- The validation `event_guard` raises on exhaustion. It is not interpreted as the paper's unresolved `max_steps` terminal rule.
- The public historical source remains in `/cache` at its pinned commit and is not redistributed under artifacts or copied into the repository.
- No MATLAB or GNU Octave executable was present. Small-case validation uses hand-calculated expected values plus explicit draw tapes against the source-traced port; it does not claim original-runtime trajectory equality.

## Source identity

- Commit: `{contract["sourceIdentity"]["commit"]}`
- Tree: `{contract["sourceIdentity"]["tree"]}`
- Contract: `{contract["contractVersion"]}`
- Engine: `{contract["engineVersion"]}`
- License state: `{contract["sourceIdentity"]["license"]}`
- Author implementation: `{contract["paperBoundary"]["authorCodeSentinel"]}`
"""


def write_artifacts(artifacts_root: Path) -> dict[str, Any]:
    contract = load_yaml(CONTRACT_PATH)
    fixtures = load_yaml(FIXTURES_PATH)
    if engine_version != str(contract["engineVersion"]):
        raise AssertionError(
            f"Engine version {engine_version} != contract {contract['engineVersion']}."
        )

    source_validation = validate_sources(contract)
    registry_validation = validate_registry(contract)
    manifest_validation = validate_manifest(contract)
    api_validation = validate_api()

    cases_by_id = {case["caseId"]: case for case in fixtures["cases"]}
    if len(cases_by_id) != len(fixtures["cases"]):
        raise AssertionError("Fixture case IDs are not unique.")
    case_results = [run_case(case, cases_by_id) for case in fixtures["cases"]]
    fixture_errors = [error for result in case_results for error in result["errors"]]
    fixture_summary = {
        "caseCount": len(case_results),
        "passedCount": sum(result["passed"] for result in case_results),
        "failedCount": sum(not result["passed"] for result in case_results),
        "allPassed": not fixture_errors,
        "errors": fixture_errors,
    }
    invariant_checks = {
        "eventMassChangesAreExactlyPlusOrMinusOne": all(
            result["actual"].get("massDelta") in {-1, 1}
            for result in case_results
            if result["kind"] == "weighted_event"
        )
        and all(
            all(delta in {-1, 1} for delta in result["actual"]["eventMassDeltas"])
            for result in case_results
            if result["kind"] == "growth"
        ),
        "fissionConservesParentIncludingOddDiscard": all(
            np.array_equal(
                np.asarray(result["actual"]["childA"])
                + np.asarray(result["actual"]["childB"])
                + np.asarray(result["actual"]["discarded"]),
                np.asarray(cases_by_id[result["caseId"]]["inputs"]["parent"]),
            )
            for result in case_results
            if result["kind"] == "fission"
        ),
        "historicalFissionHasFixedChildMass": all(
            result["actual"]["childMasses"][0] == result["actual"]["childMasses"][1]
            for result in case_results
            if result["kind"] == "fission"
        ),
        "daughterSelectionConsumesNoExtraDraw": next(
            result["actual"]["drawsConsumed"]
            for result in case_results
            if result["caseId"] == "HC09_generation_follows_child_a"
        )
        == 4,
    }
    overall_passed = (
        fixture_summary["allPassed"]
        and source_validation["passed"]
        and registry_validation["passed"]
        and manifest_validation["passed"]
        and api_validation["passed"]
        and all(invariant_checks.values())
    )

    shared_dir = artifacts_root / contract["artifactPaths"]["sharedDirectory"]
    step_dir = artifacts_root / contract["artifactPaths"]["stepDirectory"]
    shared_dir.mkdir(parents=True, exist_ok=True)
    step_dir.mkdir(parents=True, exist_ok=True)

    cases_payload = {
        "schema": "eidosoma.e01.s04_verified_small_cases.v1",
        "researchStepId": "S04",
        "fixtureVersion": fixtures["fixtureVersion"],
        "oraclePolicy": fixtures["oraclePolicy"],
        "rngPolicy": "Explicit draw tapes; no legacy MATLAB RNG equality asserted.",
        "summary": fixture_summary,
        "invariantChecks": invariant_checks,
        "cases": case_results,
    }
    (shared_dir / "verified_small_cases.json").write_text(
        json.dumps(cases_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    trace_rows = source_validation["mappingResults"]
    write_csv(
        shared_dir / "source_traceability.csv",
        trace_rows,
        [
            "behaviorId",
            "sourceFile",
            "lineStart",
            "lineEnd",
            "sourceFileSha256",
            "lineCount",
            "boundsValid",
            "portSymbols",
            "finding",
        ],
    )
    write_csv(
        shared_dir / "compatibility_matrix.csv",
        contract["compatibilityBoundaries"],
        ["boundaryId", "subject", "historicalBehavior", "paperOrModernBehavior"],
    )

    head = git_output("rev-parse", "HEAD")
    branch = git_output("branch", "--show-current")
    dirty = bool(git_output("status", "--porcelain"))
    engine_pointer = {
        "schema": "eidosoma.e01.s04_engine_pointer.v1",
        "researchStepId": "S04",
        "engineVersion": engine_version,
        "contractVersion": contract["contractVersion"],
        "repository": "Eidosoma/arrival-of-self-replicators",
        "branch": branch,
        "commit": head,
        "worktreeDirtyAtGeneration": dirty,
        "enginePaths": [
            str(path.relative_to(REPOSITORY_ROOT)) for path in ENGINE_PATHS
        ],
        "contractPath": str(CONTRACT_PATH.relative_to(REPOSITORY_ROOT)),
        "fixturesPath": str(FIXTURES_PATH.relative_to(REPOSITORY_ROOT)),
        "historicalSourceCommit": contract["sourceIdentity"]["commit"],
        "historicalSourceTree": contract["sourceIdentity"]["tree"],
        "authorCodeIdentity": contract["paperBoundary"]["authorCodeSentinel"],
        "licenseBoundary": contract["sourceIdentity"]["license"],
        "codeStorageRule": "Repository code remains in Git; artifacts contain only pointers, reports, and compact evidence.",
    }
    (shared_dir / "engine_pointer.json").write_text(
        json.dumps(engine_pointer, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    artifact_contract = {
        "schema": "eidosoma.e01.s04_historical_behavior_contract.v1",
        "researchStepId": "S04",
        "contractVersion": contract["contractVersion"],
        "engineVersion": engine_version,
        "sourceIdentity": contract["sourceIdentity"],
        "paperBoundary": contract["paperBoundary"],
        "compatibilityBoundaries": contract["compatibilityBoundaries"],
        "sourceLineMappings": contract["sourceLineMappings"],
        "s04RegistryMappings": contract["s04RegistryMappings"],
        "registryPreservation": {
            "registryPath": registry_validation["registryPath"],
            "registrySha256": registry_validation["observedSha256"],
            "unchanged": registry_validation["passed"],
        },
    }
    (shared_dir / "historical_behavior_contract.yaml").write_text(
        yaml.safe_dump(artifact_contract, sort_keys=False), encoding="utf-8"
    )
    (shared_dir / "compatibility_notes.md").write_text(
        compatibility_markdown(contract, fixture_summary, overall_passed),
        encoding="utf-8",
    )

    registry_path = step_dir / "registry_preservation.json"
    registry_path.write_text(
        json.dumps(registry_validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path = step_dir / "research_step_full_results.md"
    artifact_paths = [
        shared_dir / "compatibility_matrix.csv",
        shared_dir / "compatibility_notes.md",
        shared_dir / "engine_pointer.json",
        shared_dir / "historical_behavior_contract.yaml",
        shared_dir / "source_traceability.csv",
        shared_dir / "verified_small_cases.json",
        registry_path,
        step_dir / "validation_summary.json",
        step_dir / "artifact_manifest.json",
    ]
    if report_path.is_file():
        artifact_paths.append(report_path)
    validation = {
        "schema": "eidosoma.e01.s04_validation_summary.v1",
        "researchStepId": "S04",
        "stepNumber": 4,
        "success": overall_passed,
        "status": "complete" if overall_passed else "blocked",
        "artifactsWritten": [str(path) for path in artifact_paths],
        "validationResult": "PASS" if overall_passed else "FAIL",
        "outcomeClassification": "supportive"
        if overall_passed
        else "constraining/contradictory",
        "fixtureValidation": fixture_summary,
        "invariantChecks": invariant_checks,
        "sourceValidation": source_validation,
        "registryValidation": registry_validation,
        "sourceManifestValidation": manifest_validation,
        "apiNoSilentDefaultsValidation": api_validation,
        "runtimeCompatibility": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "matlabExecutable": shutil.which("matlab"),
            "octaveExecutable": shutil.which("octave"),
            "originalHistoricalMatlabExecuted": False,
            "fixtureOracle": "hand-calculated values plus explicit draw tapes",
            "cpuWorkers": 1,
            "gpuUsed": False,
        },
        "caveatsOrBlockers": [
            "Legacy MATLAB RNG equality remains unresolved; explicit draw tapes bypass but do not resolve it.",
            "Historical GARD has no detected repository license and remains reference-only.",
            "The author implementation remains unavailable; no identity claim is made.",
            "Historical fission, initialization, update, time, and max-step behavior differ from or underdetermine paper prose.",
            "No MATLAB/Octave runtime was present; validation does not claim original-runtime trajectory equality.",
        ],
        "recommendedNextAction": "Stop after S04 and return control; begin S05 only after separate authorization.",
        "s05Started": False,
    }
    validation_path = step_dir / "validation_summary.json"
    validation_path.write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    input_paths = [
        Path("/workspace/AGENTS.md"),
        Path("/workspace/FULL_PLAN.md"),
        Path("/workspace/RESEARCH_PLAN.md"),
        Path("/workspace/input-attachments/MANIFEST.json"),
        Path(
            "/workspace/input-attachments/ed5486bf-a043-485b-a233-d88d8d123759/_metadata/ATTACHMENT.md"
        ),
        Path(
            "/workspace/input-attachments/ed5486bf-a043-485b-a233-d88d8d123759/pdf-markdown.md"
        ),
        Path("/cache/e01_s03/downloads/paper-2607.28250v1.pdf"),
        Path("/artifacts/research_steps/S01/research_step_full_results.md"),
        Path("/artifacts/research_steps/S02/research_step_full_results.md"),
        Path("/artifacts/research_steps/S03/research_step_full_results.md"),
        Path(contract["sourceManifest"]["path"]),
        Path(contract["specificationRegistry"]["path"]),
        CONTRACT_PATH,
        FIXTURES_PATH,
        *ENGINE_PATHS,
        Path(__file__).resolve(),
    ]
    input_paths.extend(
        Path(contract["sourceIdentity"]["localPath"]) / item["path"]
        for item in contract["sourceFiles"]
    )
    output_paths = [
        path for path in artifact_paths if path.name != "artifact_manifest.json"
    ]
    records = []
    for role, paths in (("input_or_code", input_paths), ("output", output_paths)):
        for path in paths:
            if path.is_file():
                records.append(
                    {
                        "role": role,
                        "path": str(path),
                        "sizeBytes": path.stat().st_size,
                        "sha256": sha256(path),
                    }
                )
    manifest = {
        "schema": "eidosoma.e01.s04_artifact_manifest.v1",
        "researchStepId": "S04",
        "repositoryCommit": head,
        "repositoryBranch": branch,
        "repositoryDirtyAtGeneration": dirty,
        "historicalSourceCommit": contract["sourceIdentity"]["commit"],
        "records": records,
        "selfHashExcluded": True,
    }
    (step_dir / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return validation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path(os.environ.get("ARTIFACTS_DIR", "/artifacts")),
    )
    args = parser.parse_args()
    validation = write_artifacts(args.artifacts_dir.resolve())
    print(
        json.dumps(
            {
                "success": validation["success"],
                "validationResult": validation["validationResult"],
            }
        )
    )
    return 0 if validation["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
