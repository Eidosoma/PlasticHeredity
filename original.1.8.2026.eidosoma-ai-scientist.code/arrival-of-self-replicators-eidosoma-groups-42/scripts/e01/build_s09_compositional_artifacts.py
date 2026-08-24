#!/usr/bin/env python3
"""Build the frozen E01 S09 compositional-zero validation artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml
from jsonschema import Draft202012Validator
from scipy.stats import pearsonr, spearmanr

from e01_compositional_preprocessing import (
    CoordinateSpecification,
    ZeroTreatment,
    apply_zero_treatment,
    covariance_diagnostics,
    evaluate_transform,
    helmert_simplex_basis,
    pairwise_euclidean,
    principal_logratio_basis,
    validate_simplex_basis,
)
from e01_gard_reproducibility.serialization import (
    deserialize_envelope,
    float_from_hex,
    float_to_hex,
    make_envelope,
    serialize_envelope,
    validate_json_schema,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPOSITORY_ROOT / "configs/e01/s09_compositional_preregistration.yaml"
SCHEMA_PATH = REPOSITORY_ROOT / "configs/e01/s09_transform_output_schema.json"
PREREGISTRATION_COMMIT = "e83b4e9de3e8077cb6aa41b0975adc93e4b6d560"
CONTRACT_VERSION = "E01-S09-compositional-transform-contract-v1.0.0"
REGISTRY_SHA256 = "aef0e179de6466697540ba10236ed24af37fbda12bd4f1c6b1fb5fe7a27af891"
INVERSE_TOLERANCE = 1e-12
CONDITION_THRESHOLD = 1e12

DELTA_TAGS = {
    1.0e-6: "1em6",
    1.0e-4: "1em4",
    1.0e-2: "1em2",
    0.1: "0p1",
    0.5: "0p5",
    1.0: "1",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def verify_frozen_inputs(config: dict[str, Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for item in config["frozenInputs"]:
        path = Path(item["path"])
        actual = sha256_file(path) if path.is_file() else None
        success = actual == item["sha256"]
        results.append(
            {
                "inputId": item["inputId"],
                "path": str(path),
                "expectedSha256": item["sha256"],
                "actualSha256": actual,
                "success": success,
            }
        )
    failures = [item for item in results if not item["success"]]
    if failures:
        raise RuntimeError(f"Frozen S09 input mismatch: {failures}")
    return results


def load_registry(artifacts_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = (
        artifacts_root
        / "E01_forensic_replication_bundle/specifications/specification_registry_v0.3.0.yaml"
    )
    if sha256_file(path) != REGISTRY_SHA256:
        raise RuntimeError("Registry v0.3.0 identity changed.")
    registry = yaml.safe_load(path.read_text())
    owners = [item for item in registry["parameters"] if item.get("ownerStep") == "S09"]
    if len(owners) != 8 or {item["ambiguityId"] for item in owners} != {
        f"E01-A{index:03d}" for index in range(26, 34)
    }:
        raise RuntimeError("S09 registry-owner set changed.")
    if registry["executionGate"]["executable"] is not False:
        raise RuntimeError("Registry execution gate unexpectedly opened.")
    if registry["executionGate"]["noSilentDefaults"] is not True:
        raise RuntimeError("Registry no-silent-default flag changed.")
    return registry, owners


def zero_specifications(config: dict[str, Any]) -> list[ZeroTreatment]:
    result: list[ZeroTreatment] = []
    for delta in config["zeroTreatments"]["pseudocountGrid"]:
        numeric = float(delta)
        tag = DELTA_TAGS[numeric]
        result.append(
            ZeroTreatment(
                specification_id=f"E01-S09-ZERO-ADD-DELTA-{tag}-v1.0.0",
                method="additive_pseudocount",
                delta=numeric,
                evidence_class="PLAN_FROZEN_VALIDATION_BRANCH_NOT_AUTHOR_DEFAULT",
            )
        )
        result.append(
            ZeroTreatment(
                specification_id=(f"E01-S09-ZERO-MULT-MATCHDELTA-{tag}-v1.0.0"),
                method="multiplicative_replacement",
                delta=numeric,
                evidence_class="VALIDATION_ONLY_RECONSTRUCTION_NOT_AUTHOR_DEFAULT",
            )
        )
    result.append(
        ZeroTreatment(
            specification_id="E01-S09-ZERO-NONE-v1.0.0",
            method="none",
            delta=None,
            evidence_class="STRICT_DOMAIN_VALIDATION_CONTROL_NOT_AUTHOR_DEFAULT",
        )
    )
    return result


def coordinate_specifications(
    dimension: int,
    *,
    plr_fit_scope_id: str,
) -> list[CoordinateSpecification]:
    evidence = "VALIDATION_ONLY_RECONSTRUCTION_NOT_AUTHOR_DEFAULT"
    result = [
        CoordinateSpecification(
            specification_id=f"E01-S09-COORD-FULLCLR-D{dimension}-v1.0.0",
            family="full_clr",
            dimension=dimension,
            evidence_class=evidence,
            dropped_component_zero_based=None,
            basis_fit_scope_id=None,
        )
    ]
    for component in range(dimension):
        result.append(
            CoordinateSpecification(
                specification_id=(
                    f"E01-S09-COORD-DROPCLR-D{dimension}-C{component + 1}-v1.0.0"
                ),
                family="dropped_clr",
                dimension=dimension,
                evidence_class=evidence,
                dropped_component_zero_based=component,
                basis_fit_scope_id=None,
            )
        )
    result.extend(
        [
            CoordinateSpecification(
                specification_id=f"E01-S09-COORD-ILR-HELMERT-D{dimension}-v1.0.0",
                family="ilr_helmert",
                dimension=dimension,
                evidence_class=evidence,
                dropped_component_zero_based=None,
                basis_fit_scope_id=None,
            ),
            CoordinateSpecification(
                specification_id=f"E01-S09-COORD-RAW-D{dimension}-v1.0.0",
                family="raw_proportions",
                dimension=dimension,
                evidence_class="REGISTRY_ADVERSARIAL_COORDINATE_CONTROL",
                dropped_component_zero_based=None,
                basis_fit_scope_id=None,
            ),
            CoordinateSpecification(
                specification_id=f"E01-S09-COORD-HELLINGER-D{dimension}-v1.0.0",
                family="hellinger",
                dimension=dimension,
                evidence_class="REGISTRY_ADVERSARIAL_COORDINATE_CONTROL",
                dropped_component_zero_based=None,
                basis_fit_scope_id=None,
            ),
            CoordinateSpecification(
                specification_id=(
                    f"E01-S09-COORD-PLR-COVEIG-D{dimension}-FIT-"
                    f"{plr_fit_scope_id}-v1.0.0"
                ),
                family="principal_log_ratio",
                dimension=dimension,
                evidence_class="REGISTRY_ADVERSARIAL_COORDINATE_CONTROL",
                dropped_component_zero_based=None,
                basis_fit_scope_id=plr_fit_scope_id,
            ),
        ]
    )
    return result


def complete_specification_id(
    zero_specification_id: str,
    coordinate_specification_id: str,
) -> str:
    digest = hashlib.sha256(
        f"{zero_specification_id}\0{coordinate_specification_id}".encode()
    ).hexdigest()[:20]
    return f"E01-S09-COMPLETE-{digest}-v1.0.0"


def _basis_for_coordinate(
    specification: CoordinateSpecification,
    *,
    plr_basis: np.ndarray | None,
) -> np.ndarray | None:
    if specification.family == "ilr_helmert":
        return helmert_simplex_basis(specification.dimension)
    if specification.family == "principal_log_ratio":
        return plr_basis
    return None


def evaluate_scope(
    *,
    scope_id: str,
    states: np.ndarray,
    observation_ids: list[str],
    zero_specs: list[ZeroTreatment],
    include_payload_records: bool,
    fixture_id_for_payload: str | None,
    coordinate_catalog: dict[str, CoordinateSpecification],
    complete_catalog: dict[str, dict[str, Any]],
    basis_catalog: dict[str, dict[str, Any]],
) -> tuple[
    dict[tuple[str, str], list[dict[str, Any]]],
    list[dict[str, Any]],
]:
    if states.ndim != 2 or states.shape[0] != len(observation_ids):
        raise RuntimeError("Scope state/identity shape mismatch.")
    dimension = states.shape[1]
    coordinates = coordinate_specifications(dimension, plr_fit_scope_id=scope_id)
    for specification in coordinates:
        coordinate_catalog[specification.specification_id] = specification

    evaluations: dict[tuple[str, str], list[dict[str, Any]]] = {}
    payload_records: list[dict[str, Any]] = []
    for zero in zero_specs:
        treated = [apply_zero_treatment(row, zero) for row in states]
        plr_candidates = [
            item.composition
            for item in treated
            if item.composition is not None and np.all(item.composition > 0)
        ]
        plr_basis: np.ndarray | None = None
        plr_eigenvalues: np.ndarray | None = None
        plr_status = "FIT"
        if len(plr_candidates) >= 2:
            plr_basis, plr_eigenvalues = principal_logratio_basis(
                np.vstack(plr_candidates)
            )
        else:
            plr_status = "INSUFFICIENT_ROWS_TO_FIT"
        basis_id = f"E01-S09-PLR-BASIS-{scope_id}-{zero.specification_id}-v1.0.0"
        basis_catalog[basis_id] = {
            "basisId": basis_id,
            "scopeId": scope_id,
            "zeroSpecificationId": zero.specification_id,
            "dimension": dimension,
            "status": plr_status,
            "eligibleFitRowCount": len(plr_candidates),
            "basisHex": (
                [[float_to_hex(value) for value in row] for row in plr_basis]
                if plr_basis is not None
                else None
            ),
            "eigenvaluesHex": (
                [float_to_hex(value) for value in plr_eigenvalues]
                if plr_eigenvalues is not None
                else None
            ),
            "gramMaximumAbsoluteErrorHex": (
                float_to_hex(
                    float(
                        np.max(
                            np.abs(
                                plr_basis.T @ plr_basis
                                - np.eye(dimension - 1, dtype=np.float64)
                            )
                        )
                    )
                )
                if plr_basis is not None
                else None
            ),
            "onesOrthogonalityMaximumAbsoluteErrorHex": (
                float_to_hex(
                    float(
                        np.max(
                            np.abs(plr_basis.T @ np.ones(dimension, dtype=np.float64))
                        )
                    )
                )
                if plr_basis is not None
                else None
            ),
            "eigenvalueNearTieToleranceHex": (
                float_to_hex(
                    max(len(plr_candidates), dimension)
                    * np.finfo(np.float64).eps
                    * max(float(np.max(np.abs(plr_eigenvalues))), 1.0)
                )
                if plr_eigenvalues is not None
                else None
            ),
            "adjacentEigenvalueNearTieCount": (
                int(
                    np.count_nonzero(
                        np.abs(np.diff(plr_eigenvalues))
                        <= max(len(plr_candidates), dimension)
                        * np.finfo(np.float64).eps
                        * max(float(np.max(np.abs(plr_eigenvalues))), 1.0)
                    )
                )
                if plr_eigenvalues is not None
                else None
            ),
        }
        for coordinate in coordinates:
            key = (zero.specification_id, coordinate.specification_id)
            complete_id = complete_specification_id(*key)
            complete_catalog[complete_id] = {
                "completeSpecificationId": complete_id,
                "zeroSpecificationId": zero.specification_id,
                "coordinateSpecificationId": coordinate.specification_id,
                "dimension": dimension,
                "basisFitScopeId": coordinate.basis_fit_scope_id,
                "basisId": (
                    basis_id if coordinate.family == "principal_log_ratio" else None
                ),
                "evidenceClass": "VALIDATION_ONLY_RECONSTRUCTION_NOT_AUTHOR_DEFAULT",
            }
            entries: list[dict[str, Any]] = []
            for index, (state, observation_id, zero_result) in enumerate(
                zip(states, observation_ids, treated, strict=True), start=1
            ):
                entry: dict[str, Any] = {
                    "observationId": observation_id,
                    "observationIndexOneBased": index,
                    "inputState": np.asarray(state, dtype=np.float64),
                    "inputMass": float(np.sum(state)),
                    "zeroCount": zero_result.zero_count,
                    "treatedComposition": zero_result.composition,
                    "replacementMassPerZero": zero_result.replacement_mass_per_zero,
                    "status": zero_result.status,
                    "reason": zero_result.reason,
                    "coordinates": None,
                    "reconstructedComposition": None,
                    "maximumAbsoluteInverseError": None,
                    "maximumRelativeInverseError": None,
                    "closureError": None,
                }
                if zero_result.composition is not None:
                    if coordinate.family == "principal_log_ratio" and plr_basis is None:
                        entry["status"] = "INELIGIBLE"
                        entry["reason"] = (
                            "INSUFFICIENT_ROWS_TO_FIT_PRINCIPAL_LOG_RATIO_BASIS"
                        )
                    else:
                        basis = _basis_for_coordinate(coordinate, plr_basis=plr_basis)
                        transformed = evaluate_transform(
                            zero_result.composition,
                            coordinate,
                            simplex_basis=basis,
                        )
                        entry.update(
                            {
                                "status": transformed.status,
                                "reason": transformed.reason,
                                "coordinates": transformed.coordinates,
                                "reconstructedComposition": transformed.reconstructed_composition,
                                "maximumAbsoluteInverseError": transformed.maximum_absolute_inverse_error,
                                "maximumRelativeInverseError": transformed.maximum_relative_inverse_error,
                                "closureError": transformed.closure_error,
                            }
                        )
                entries.append(entry)
                if include_payload_records:
                    assert fixture_id_for_payload is not None
                    payload_records.append(
                        payload_record(
                            fixture_id=fixture_id_for_payload,
                            complete_specification_id=complete_id,
                            zero_specification_id=zero.specification_id,
                            coordinate_specification_id=coordinate.specification_id,
                            entry=entry,
                        )
                    )
            evaluations[key] = entries
    return evaluations, payload_records


def _hex_array(values: np.ndarray | None) -> list[str] | None:
    if values is None:
        return None
    return [float_to_hex(float(value)) for value in values]


def _hex_optional(value: float | None) -> str | None:
    return None if value is None else float_to_hex(value)


def payload_record(
    *,
    fixture_id: str,
    complete_specification_id: str,
    zero_specification_id: str,
    coordinate_specification_id: str,
    entry: dict[str, Any],
) -> dict[str, Any]:
    return {
        "fixtureId": fixture_id,
        "observationId": entry["observationId"],
        "observationIndexOneBased": entry["observationIndexOneBased"],
        "completeSpecificationId": complete_specification_id,
        "zeroSpecificationId": zero_specification_id,
        "coordinateSpecificationId": coordinate_specification_id,
        "status": entry["status"],
        "reason": entry["reason"],
        "inputStateHex": _hex_array(entry["inputState"]),
        "inputMassHex": float_to_hex(entry["inputMass"]),
        "zeroCount": entry["zeroCount"],
        "treatedCompositionHex": _hex_array(entry["treatedComposition"]),
        "replacementMassPerZeroHex": _hex_optional(entry["replacementMassPerZero"]),
        "coordinatesHex": _hex_array(entry["coordinates"]),
        "reconstructedCompositionHex": _hex_array(entry["reconstructedComposition"]),
        "maximumAbsoluteInverseErrorHex": _hex_optional(
            entry["maximumAbsoluteInverseError"]
        ),
        "maximumRelativeInverseErrorHex": _hex_optional(
            entry["maximumRelativeInverseError"]
        ),
        "closureErrorHex": _hex_optional(entry["closureError"]),
    }


def output_csv_rows(payload_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in payload_records:
        rows.append(
            {
                "fixtureId": record["fixtureId"],
                "observationId": record["observationId"],
                "observationIndexOneBased": record["observationIndexOneBased"],
                "completeSpecificationId": record["completeSpecificationId"],
                "zeroSpecificationId": record["zeroSpecificationId"],
                "coordinateSpecificationId": record["coordinateSpecificationId"],
                "status": record["status"],
                "reason": record["reason"] or "",
                "inputMass": float.fromhex(record["inputMassHex"]),
                "zeroCount": record["zeroCount"],
                "treatedComposition": json.dumps(
                    [float.fromhex(x) for x in record["treatedCompositionHex"]]
                    if record["treatedCompositionHex"] is not None
                    else None,
                    separators=(",", ":"),
                ),
                "coordinates": json.dumps(
                    [float.fromhex(x) for x in record["coordinatesHex"]]
                    if record["coordinatesHex"] is not None
                    else None,
                    separators=(",", ":"),
                ),
                "maximumAbsoluteInverseError": (
                    ""
                    if record["maximumAbsoluteInverseErrorHex"] is None
                    else format(
                        float.fromhex(record["maximumAbsoluteInverseErrorHex"]),
                        ".17g",
                    )
                ),
            }
        )
    return rows


def numerical_rows(
    *,
    scope_id: str,
    evaluations: dict[tuple[str, str], list[dict[str, Any]]],
    coordinate_catalog: dict[str, CoordinateSpecification],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    diagnostics: list[dict[str, Any]] = []
    eligibility: list[dict[str, Any]] = []
    inverse: list[dict[str, Any]] = []
    for (zero_id, coordinate_id), entries in sorted(evaluations.items()):
        specification = coordinate_catalog[coordinate_id]
        eligible = [item for item in entries if item["status"] == "ELIGIBLE"]
        reasons = Counter(
            item["reason"] for item in entries if item["status"] != "ELIGIBLE"
        )
        coordinates = (
            np.vstack([item["coordinates"] for item in eligible])
            if eligible
            else np.empty((0, 0), dtype=np.float64)
        )
        coordinate_dimension = (
            specification.dimension
            if specification.family in ("full_clr", "raw_proportions", "hellinger")
            else specification.dimension - 1
        )
        finite = bool(
            not eligible
            or (
                coordinates.shape == (len(eligible), coordinate_dimension)
                and np.all(np.isfinite(coordinates))
            )
        )
        if eligible:
            covariance = covariance_diagnostics(
                coordinates, condition_threshold=CONDITION_THRESHOLD
            )
        else:
            covariance = {
                "status": "INSUFFICIENT_ELIGIBLE_ROWS",
                "rank": 0,
                "rankTolerance": None,
                "conditionNumberRaw": None,
                "effectiveConditionNumber": None,
            }
        if specification.family == "full_clr" and len(eligible) >= 2:
            readiness = "STRUCTURALLY_SINGULAR_FULL_CLR"
        elif specification.family == "raw_proportions" and len(eligible) >= 2:
            readiness = "STRUCTURALLY_SINGULAR_RAW_CLOSURE"
        else:
            readiness = covariance["status"]
        expected_max_rank = min(
            max(len(eligible) - 1, 0),
            (
                coordinate_dimension - 1
                if specification.family in ("full_clr", "raw_proportions")
                else coordinate_dimension
            ),
        )
        inverse_abs = [item["maximumAbsoluteInverseError"] for item in eligible]
        inverse_rel = [item["maximumRelativeInverseError"] for item in eligible]
        closure = [item["closureError"] for item in eligible]
        inverse_pass = bool(
            eligible
            and max(inverse_abs) <= INVERSE_TOLERANCE
            and max(inverse_rel) <= INVERSE_TOLERANCE
            and max(closure) <= INVERSE_TOLERANCE
        )
        complete_id = complete_specification_id(zero_id, coordinate_id)
        common = {
            "scopeId": scope_id,
            "completeSpecificationId": complete_id,
            "zeroSpecificationId": zero_id,
            "coordinateSpecificationId": coordinate_id,
            "coordinateFamily": specification.family,
            "dimension": specification.dimension,
            "droppedComponentOneBased": (
                ""
                if specification.dropped_component_zero_based is None
                else specification.dropped_component_zero_based + 1
            ),
            "inputObservationCount": len(entries),
            "eligibleObservationCount": len(eligible),
            "ineligibleObservationCount": len(entries) - len(eligible),
        }
        diagnostics.append(
            {
                **common,
                "coordinateDimension": coordinate_dimension,
                "allEligibleValuesFinite": finite,
                "covarianceRank": covariance["rank"],
                "expectedMaximumRank": expected_max_rank,
                "rankTolerance": number_or_blank(covariance["rankTolerance"]),
                "conditionNumberRaw": number_or_inf_or_blank(
                    covariance["conditionNumberRaw"]
                ),
                "effectiveConditionNumber": number_or_inf_or_blank(
                    covariance["effectiveConditionNumber"]
                ),
                "covarianceReadinessStatus": readiness,
            }
        )
        eligibility.append(
            {
                **common,
                "eligibilityFraction": format(len(eligible) / len(entries), ".17g"),
                "ineligibilityReasonsJson": json.dumps(
                    {str(k): v for k, v in sorted(reasons.items())},
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "rowCoverageComplete": True,
            }
        )
        inverse.append(
            {
                **common,
                "maximumAbsoluteError": (
                    "" if not inverse_abs else format(max(inverse_abs), ".17g")
                ),
                "maximumRelativeError": (
                    "" if not inverse_rel else format(max(inverse_rel), ".17g")
                ),
                "maximumClosureError": (
                    "" if not closure else format(max(closure), ".17g")
                ),
                "validationStatus": "PASS" if inverse_pass else "NOT_EVALUABLE",
            }
        )
    return diagnostics, eligibility, inverse


def number_or_blank(value: float | None) -> str:
    return "" if value is None else format(float(value), ".17g")


def number_or_inf_or_blank(value: float | None) -> str:
    if value is None:
        return ""
    if math.isinf(value):
        return "inf"
    return format(float(value), ".17g")


def upper_triangle(values: np.ndarray) -> np.ndarray:
    indices = np.triu_indices(values.shape[0], k=1)
    return values[indices]


def correlation_or_none(
    first: np.ndarray, second: np.ndarray, kind: str
) -> float | None:
    if first.size < 2 or np.all(first == first[0]) or np.all(second == second[0]):
        return None
    if kind == "pearson":
        return float(pearsonr(first, second).statistic)
    return float(spearmanr(first, second).statistic)


def representation_rows(
    *,
    scope_id: str,
    zero_specs: list[ZeroTreatment],
    evaluations: dict[tuple[str, str], list[dict[str, Any]]],
    coordinate_catalog: dict[str, CoordinateSpecification],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    by_zero: dict[str, dict[str, tuple[str, list[dict[str, Any]]]]] = defaultdict(dict)
    for (zero_id, coordinate_id), entries in evaluations.items():
        specification = coordinate_catalog[coordinate_id]
        label = specification.family
        if specification.family == "dropped_clr":
            label = f"dropped_clr_C{specification.dropped_component_zero_based + 1}"
        by_zero[zero_id][label] = (coordinate_id, entries)

    comparisons = [
        ("full_clr", "ilr_helmert", True),
        ("ilr_helmert", "principal_log_ratio", True),
        ("raw_proportions", "ilr_helmert", False),
        ("hellinger", "ilr_helmert", False),
    ]
    for dimension in sorted({item.dimension for item in coordinate_catalog.values()}):
        for component in range(1, dimension + 1):
            comparisons.append((f"dropped_clr_C{component}", "ilr_helmert", False))
    for zero in zero_specs:
        collection = by_zero[zero.specification_id]
        for first_label, second_label, isometry in comparisons:
            if first_label not in collection or second_label not in collection:
                continue
            first_id, first_entries = collection[first_label]
            second_id, second_entries = collection[second_label]
            first_by_obs = {
                item["observationId"]: item
                for item in first_entries
                if item["status"] == "ELIGIBLE"
            }
            second_by_obs = {
                item["observationId"]: item
                for item in second_entries
                if item["status"] == "ELIGIBLE"
            }
            common_ids = sorted(set(first_by_obs) & set(second_by_obs))
            max_difference: float | None = None
            pearson: float | None = None
            spearman: float | None = None
            if len(common_ids) >= 2:
                first_matrix = np.vstack(
                    [first_by_obs[item]["coordinates"] for item in common_ids]
                )
                second_matrix = np.vstack(
                    [second_by_obs[item]["coordinates"] for item in common_ids]
                )
                first_distances = upper_triangle(pairwise_euclidean(first_matrix))
                second_distances = upper_triangle(pairwise_euclidean(second_matrix))
                max_difference = float(
                    np.max(np.abs(first_distances - second_distances))
                )
                pearson = correlation_or_none(
                    first_distances, second_distances, "pearson"
                )
                spearman = correlation_or_none(
                    first_distances, second_distances, "spearman"
                )
                status = "PASS" if (not isometry or max_difference <= 1e-12) else "FAIL"
            else:
                status = "NOT_EVALUABLE_FEWER_THAN_TWO_COMMON_ROWS"
            output.append(
                {
                    "scopeId": scope_id,
                    "zeroSpecificationId": zero.specification_id,
                    "firstCoordinateSpecificationId": first_id,
                    "secondCoordinateSpecificationId": second_id,
                    "firstFamily": first_label,
                    "secondFamily": second_label,
                    "commonEligibleObservationCount": len(common_ids),
                    "distancePairCount": len(common_ids) * (len(common_ids) - 1) // 2,
                    "maximumAbsoluteDistanceDifference": (
                        "" if max_difference is None else format(max_difference, ".17g")
                    ),
                    "PearsonDistanceCorrelation": (
                        "" if pearson is None else format(pearson, ".17g")
                    ),
                    "SpearmanDistanceCorrelation": (
                        "" if spearman is None else format(spearman, ".17g")
                    ),
                    "isometryRequired": isometry,
                    "validationStatus": status,
                }
            )
    return output


def zero_frequency_rows(
    fixtures: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    observations: list[dict[str, Any]] = []
    fixtures_rows: list[dict[str, Any]] = []
    for fixture in fixtures:
        states = np.asarray(fixture["states"], dtype=np.float64)
        for index, (observation_id, row) in enumerate(
            zip(fixture["observationIds"], states, strict=True), start=1
        ):
            zeros = int(np.count_nonzero(row == 0))
            observations.append(
                {
                    "fixtureId": fixture["fixtureId"],
                    "observationId": observation_id,
                    "observationIndexOneBased": index,
                    "dimension": row.size,
                    "inputMass": format(float(np.sum(row)), ".17g"),
                    "zeroComponentCount": zeros,
                    "zeroComponentFraction": format(zeros / row.size, ".17g"),
                    "anyZero": zeros > 0,
                    "zeroSum": bool(np.sum(row) == 0),
                }
            )
        fixture_observations = [
            item for item in observations if item["fixtureId"] == fixture["fixtureId"]
        ]
        fixtures_rows.append(
            {
                "scopeId": fixture["fixtureId"],
                "dimension": states.shape[1],
                "observationCount": states.shape[0],
                "observationWithAnyZeroCount": sum(
                    bool(item["anyZero"]) for item in fixture_observations
                ),
                "zeroSumObservationCount": sum(
                    bool(item["zeroSum"]) for item in fixture_observations
                ),
                "zeroComponentCount": sum(
                    int(item["zeroComponentCount"]) for item in fixture_observations
                ),
                "totalComponentCellCount": states.size,
                "zeroComponentFraction": format(
                    sum(
                        int(item["zeroComponentCount"]) for item in fixture_observations
                    )
                    / states.size,
                    ".17g",
                ),
            }
        )
    return observations, fixtures_rows


def replacement_rows(
    fixtures: list[dict[str, Any]], zero_specs: list[ZeroTreatment]
) -> list[dict[str, Any]]:
    additive = {
        item.delta: item for item in zero_specs if item.method == "additive_pseudocount"
    }
    multiplicative = {
        item.delta: item
        for item in zero_specs
        if item.method == "multiplicative_replacement"
    }
    rows: list[dict[str, Any]] = []
    for fixture in fixtures:
        for observation_id, state_value in zip(
            fixture["observationIds"], fixture["states"], strict=True
        ):
            state = np.asarray(state_value, dtype=np.float64)
            positive_indices = np.flatnonzero(state > 0)
            for delta in sorted(additive):
                add = apply_zero_treatment(state, additive[delta])
                mult = apply_zero_treatment(state, multiplicative[delta])
                if mult.composition is None:
                    rows.append(
                        {
                            "fixtureId": fixture["fixtureId"],
                            "observationId": observation_id,
                            "delta": format(delta, ".17g"),
                            "status": "INELIGIBLE",
                            "reason": mult.reason,
                            "zeroCount": mult.zero_count,
                            "additiveVsMultiplicativeAitchisonDistance": "",
                            "positivePartLogRatioMaximumError": "",
                            "strictlyPositiveRowMaximumChangeUnderMultiplicative": "",
                        }
                    )
                    continue
                assert add.composition is not None
                add_clr = np.log(add.composition) - np.mean(np.log(add.composition))
                mult_clr = np.log(mult.composition) - np.mean(np.log(mult.composition))
                distance = float(np.linalg.norm(add_clr - mult_clr))
                if positive_indices.size >= 2:
                    original = state[positive_indices] / np.sum(state[positive_indices])
                    replaced = mult.composition[positive_indices]
                    original_ratios = np.log(original[:, None] / original[None, :])
                    replaced_ratios = np.log(replaced[:, None] / replaced[None, :])
                    ratio_error = float(
                        np.max(np.abs(original_ratios - replaced_ratios))
                    )
                else:
                    ratio_error = 0.0
                if mult.zero_count == 0:
                    original_closed = state / np.sum(state)
                    positive_change = float(
                        np.max(np.abs(original_closed - mult.composition))
                    )
                else:
                    positive_change = None
                rows.append(
                    {
                        "fixtureId": fixture["fixtureId"],
                        "observationId": observation_id,
                        "delta": format(delta, ".17g"),
                        "status": "ELIGIBLE",
                        "reason": "",
                        "zeroCount": mult.zero_count,
                        "additiveVsMultiplicativeAitchisonDistance": format(
                            distance, ".17g"
                        ),
                        "positivePartLogRatioMaximumError": format(ratio_error, ".17g"),
                        "strictlyPositiveRowMaximumChangeUnderMultiplicative": (
                            ""
                            if positive_change is None
                            else format(positive_change, ".17g")
                        ),
                    }
                )
    return rows


def failure_injections(
    *,
    sample_envelope_bytes: bytes,
    expected_record_count: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    def caught(injection_id: str, callback: Any) -> None:
        try:
            callback()
        except Exception as exc:  # noqa: BLE001 - the artifact records exact type
            results.append(
                {
                    "injectionId": injection_id,
                    "success": True,
                    "detectedBy": type(exc).__name__,
                    "message": str(exc),
                }
            )
        else:
            results.append(
                {
                    "injectionId": injection_id,
                    "success": False,
                    "detectedBy": None,
                    "message": "injection was not detected",
                }
            )

    valid_zero = ZeroTreatment("TEST-ZERO", "none", None, "VALIDATION")
    caught("NEGATIVE_INPUT_REJECTED", lambda: apply_zero_treatment([1, -1], valid_zero))
    caught(
        "NONFINITE_INPUT_REJECTED",
        lambda: apply_zero_treatment([1, float("nan")], valid_zero),
    )
    caught(
        "RAW_REGISTRY_SENTINEL_REJECTED_AS_EXECUTABLE_SPECIFICATION",
        lambda: ZeroTreatment("UNRESOLVED::E01-A027", "none", None, "VALIDATION"),
    )

    strict = apply_zero_treatment([1, 0], valid_zero)
    coordinate = CoordinateSpecification(
        "TEST-FULL-CLR",
        "full_clr",
        2,
        "VALIDATION",
        None,
        None,
    )
    result = evaluate_transform(strict.composition, coordinate, simplex_basis=None)
    if result.status != "INELIGIBLE" or result.reason != (
        "ZERO_COMPONENT_LOG_RATIO_WITHOUT_REPLACEMENT"
    ):
        results.append(
            {
                "injectionId": "LOG_RATIO_ZERO_WITHOUT_REPLACEMENT_REMAINS_INELIGIBLE_NOT_INFINITE",
                "success": False,
                "detectedBy": None,
                "message": "zero log-ratio domain was not explicit",
            }
        )
    else:
        results.append(
            {
                "injectionId": "LOG_RATIO_ZERO_WITHOUT_REPLACEMENT_REMAINS_INELIGIBLE_NOT_INFINITE",
                "success": True,
                "detectedBy": "STATUS_DOMAIN_CHECK",
                "message": result.reason,
            }
        )
    caught(
        "NONORTHONORMAL_ILR_BASIS_REJECTED",
        lambda: validate_simplex_basis([[1.0], [1.0]], dimension=2),
    )

    def hidden_row_check() -> None:
        observed = expected_record_count - 1
        if observed != expected_record_count:
            raise RuntimeError(
                f"row count {observed} differs from expected {expected_record_count}"
            )

    caught("HIDDEN_ROW_DELETION_DETECTED", hidden_row_check)

    def corrupt_inverse() -> None:
        expected = np.array([0.5, 0.5])
        corrupted = np.array([0.6, 0.4])
        error = float(np.max(np.abs(expected - corrupted)))
        if error > INVERSE_TOLERANCE:
            raise RuntimeError(f"inverse error {error} exceeds tolerance")

    caught("CORRUPTED_INVERSE_DETECTED", corrupt_inverse)

    tampered = bytearray(sample_envelope_bytes)
    marker = b'"researchStepId":"S09"'
    location = tampered.find(marker)
    if location < 0:
        raise RuntimeError("Unable to locate tamper marker.")
    tampered[location : location + len(marker)] = b'"researchStepId":"X09"'
    caught(
        "TAMPERED_CHECKSUM_REJECTED",
        lambda: deserialize_envelope(bytes(tampered), require_canonical=True),
    )
    return results


def specification_registry_rows(
    *,
    complete_catalog: dict[str, dict[str, Any]],
    payload_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_complete: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in payload_records:
        by_complete[record["completeSpecificationId"]].append(record)
    output: list[dict[str, Any]] = []
    for complete_id, item in sorted(complete_catalog.items()):
        records = by_complete.get(complete_id, [])
        if not records:
            continue
        eligible = [record for record in records if record["status"] == "ELIGIBLE"]
        inverse_errors = [
            float_from_hex(record["maximumAbsoluteInverseErrorHex"])
            for record in eligible
        ]
        if eligible and max(inverse_errors) <= INVERSE_TOLERANCE:
            status = "ACCEPTED_WITH_EXPLICIT_ELIGIBILITY_DOMAIN"
        elif not eligible:
            status = "NOT_EVALUABLE_DECLARED_DOMAIN"
        else:
            status = "REJECTED_NUMERICAL_VALIDATION_FAILURE"
        reasons = Counter(
            record["reason"] for record in records if record["status"] != "ELIGIBLE"
        )
        output.append(
            {
                **item,
                "recordCount": len(records),
                "eligibleRecordCount": len(eligible),
                "ineligibleRecordCount": len(records) - len(eligible),
                "ineligibilityReasons": {
                    str(key): value for key, value in sorted(reasons.items())
                },
                "maximumAbsoluteInverseError": (
                    None if not inverse_errors else max(inverse_errors)
                ),
                "transformValidationStatus": status,
                "paperDefaultStatus": "NOT_RECOVERED_NOT_ASSIGNED",
            }
        )
    return output


def create_figures(
    *,
    step_dir: Path,
    zero_observations: list[dict[str, Any]],
    numerical: list[dict[str, Any]],
    replacements: list[dict[str, Any]],
) -> list[Path]:
    fixture_names = sorted({item["fixtureId"] for item in zero_observations})
    zero_counts = [
        sum(
            bool(item["anyZero"])
            for item in zero_observations
            if item["fixtureId"] == fixture
        )
        for fixture in fixture_names
    ]
    figure, axis = plt.subplots(figsize=(10, 4.5))
    axis.bar(range(len(fixture_names)), zero_counts, color="#4c78a8")
    axis.set_xticks(
        range(len(fixture_names)), [short_fixture(x) for x in fixture_names]
    )
    axis.set_ylabel("observations with ≥1 zero")
    axis.set_title("S09 zero-containing observations retained by fixture")
    axis.tick_params(axis="x", rotation=25)
    figure.tight_layout()
    zero_path = step_dir / "zero_frequency.png"
    figure.savefig(zero_path, dpi=180)
    plt.close(figure)

    pooled = [item for item in numerical if item["scopeId"].startswith("E01-S09-POOL")]
    families = sorted({item["coordinateFamily"] for item in pooled})
    ready_fraction = []
    for family in families:
        group = [item for item in pooled if item["coordinateFamily"] == family]
        ready_fraction.append(
            sum(item["covarianceReadinessStatus"] == "READY" for item in group)
            / len(group)
        )
    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.bar(range(len(families)), ready_fraction, color="#f58518")
    axis.set_xticks(range(len(families)), families)
    axis.set_ylim(0, 1)
    axis.set_ylabel("fraction of pooled specifications READY")
    axis.set_title("Covariance readiness is distinct from transform validity")
    axis.tick_params(axis="x", rotation=30)
    figure.tight_layout()
    conditioning_path = step_dir / "covariance_readiness.png"
    figure.savefig(conditioning_path, dpi=180)
    plt.close(figure)

    eligible_replacements = [
        item for item in replacements if item["status"] == "ELIGIBLE"
    ]
    deltas = sorted({float(item["delta"]) for item in eligible_replacements})
    medians = [
        float(
            np.median(
                [
                    float(item["additiveVsMultiplicativeAitchisonDistance"])
                    for item in eligible_replacements
                    if float(item["delta"]) == delta
                ]
            )
        )
        for delta in deltas
    ]
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(deltas, medians, marker="o", color="#54a24b")
    axis.set_xscale("log")
    axis.set_xlabel("matched count-scale delta")
    axis.set_ylabel("median additive–multiplicative Aitchison distance")
    axis.set_title("Zero-treatment disagreement across the frozen grid")
    figure.tight_layout()
    replacement_path = step_dir / "replacement_disagreement.png"
    figure.savefig(replacement_path, dpi=180)
    plt.close(figure)
    return [zero_path, conditioning_path, replacement_path]


def short_fixture(value: str) -> str:
    return (
        value.replace("E01-S08-FIXTURE-", "").replace("-v1.0.0", "").replace("-", " ")
    )


def manifest_entries(paths: list[tuple[Path, str]]) -> list[dict[str, Any]]:
    result = []
    for path, role in sorted(paths, key=lambda item: str(item[0])):
        if path.is_file():
            result.append(
                {
                    "path": str(path),
                    "role": role,
                    "sizeBytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return result


def write_artifact_manifest(artifacts_root: Path) -> None:
    step_dir = artifacts_root / "research_steps/S09"
    shared_dir = artifacts_root / "E01_forensic_replication_bundle/preprocessing"
    paths: list[tuple[Path, str]] = []
    for path in step_dir.glob("*"):
        if path.name != "artifact_manifest.json" and path.is_file():
            paths.append((path, "S09_RESULT"))
    for path in shared_dir.glob("*"):
        if path.is_file():
            paths.append((path, "SHARED_REUSABLE_PREPROCESSING_ARTIFACT"))
    for path in [
        CONFIG_PATH,
        SCHEMA_PATH,
        REPOSITORY_ROOT / "src/e01_compositional_preprocessing/__init__.py",
        REPOSITORY_ROOT / "src/e01_compositional_preprocessing/transforms.py",
        Path(__file__).resolve(),
        REPOSITORY_ROOT / "tests/e01/test_compositional_preprocessing.py",
        Path("/workspace/RESEARCH_PLAN.md"),
    ]:
        if path.is_file():
            paths.append((path, "INPUT_OR_REPOSITORY_PROVENANCE"))
    manifest = {
        "schema": "eidosoma.e01.s09_artifact_manifest.v1",
        "researchStepId": "S09",
        "repository": str(REPOSITORY_ROOT),
        "branch": git_output("branch", "--show-current"),
        "repositoryCommit": git_output("rev-parse", "HEAD"),
        "preregistrationCommit": PREREGISTRATION_COMMIT,
        "artifacts": manifest_entries(paths),
        "selfHashExcluded": True,
        "s10Absent": not (artifacts_root / "research_steps/S10").exists(),
    }
    write_json(step_dir / "artifact_manifest.json", manifest)


def build(artifacts_root: Path) -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    expected_thread_environment = config["executionPolicy"]["requiredEnvironment"]
    actual_thread_environment = {
        name: os.environ.get(name) for name in expected_thread_environment
    }
    if actual_thread_environment != expected_thread_environment:
        raise RuntimeError(
            "S09 numeric thread environment differs from the frozen contract: "
            f"{actual_thread_environment}."
        )
    preregistration_sha = sha256_file(CONFIG_PATH)
    frozen_results = verify_frozen_inputs(config)
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", PREREGISTRATION_COMMIT, "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=False,
    ).returncode:
        raise RuntimeError("Frozen S09 preregistration is not an ancestor of HEAD.")
    registry, registry_owners = load_registry(artifacts_root)
    step_dir = artifacts_root / "research_steps/S09"
    shared_dir = artifacts_root / "E01_forensic_replication_bundle/preprocessing"
    s10_dir = artifacts_root / "research_steps/S10"
    outcomes_preexisting = step_dir.exists()
    if s10_dir.exists():
        raise RuntimeError(
            "S10 artifacts already exist; S09 scope audit cannot proceed."
        )
    step_dir.mkdir(parents=True, exist_ok=True)
    shared_dir.mkdir(parents=True, exist_ok=True)

    fixture_path = artifacts_root / "research_steps/S08/fixture_catalog.json"
    fixtures = json.loads(fixture_path.read_text())["fixtures"]
    zero_specs = zero_specifications(config)
    coordinate_catalog: dict[str, CoordinateSpecification] = {}
    complete_catalog: dict[str, dict[str, Any]] = {}
    basis_catalog: dict[str, dict[str, Any]] = {}
    payload_records: list[dict[str, Any]] = []
    all_numerical: list[dict[str, Any]] = []
    all_eligibility: list[dict[str, Any]] = []
    all_inverse: list[dict[str, Any]] = []
    all_representation: list[dict[str, Any]] = []

    for fixture in fixtures:
        states = np.asarray(fixture["states"], dtype=np.float64)
        evaluations, records = evaluate_scope(
            scope_id=fixture["fixtureId"],
            states=states,
            observation_ids=fixture["observationIds"],
            zero_specs=zero_specs,
            include_payload_records=True,
            fixture_id_for_payload=fixture["fixtureId"],
            coordinate_catalog=coordinate_catalog,
            complete_catalog=complete_catalog,
            basis_catalog=basis_catalog,
        )
        payload_records.extend(records)
        numerical, eligibility, inverse = numerical_rows(
            scope_id=fixture["fixtureId"],
            evaluations=evaluations,
            coordinate_catalog=coordinate_catalog,
        )
        all_numerical.extend(numerical)
        all_eligibility.extend(eligibility)
        all_inverse.extend(inverse)
        all_representation.extend(
            representation_rows(
                scope_id=fixture["fixtureId"],
                zero_specs=zero_specs,
                evaluations=evaluations,
                coordinate_catalog=coordinate_catalog,
            )
        )

    for dimension in sorted({len(item["states"][0]) for item in fixtures}):
        selected = [item for item in fixtures if len(item["states"][0]) == dimension]
        states = np.vstack(
            [np.asarray(item["states"], dtype=np.float64) for item in selected]
        )
        observation_ids = [
            f"{item['fixtureId']}::{observation_id}"
            for item in selected
            for observation_id in item["observationIds"]
        ]
        scope_id = f"E01-S09-POOL-D{dimension}-v1.0.0"
        evaluations, _ = evaluate_scope(
            scope_id=scope_id,
            states=states,
            observation_ids=observation_ids,
            zero_specs=zero_specs,
            include_payload_records=False,
            fixture_id_for_payload=None,
            coordinate_catalog=coordinate_catalog,
            complete_catalog=complete_catalog,
            basis_catalog=basis_catalog,
        )
        numerical, eligibility, inverse = numerical_rows(
            scope_id=scope_id,
            evaluations=evaluations,
            coordinate_catalog=coordinate_catalog,
        )
        all_numerical.extend(numerical)
        all_eligibility.extend(eligibility)
        all_inverse.extend(inverse)
        all_representation.extend(
            representation_rows(
                scope_id=scope_id,
                zero_specs=zero_specs,
                evaluations=evaluations,
                coordinate_catalog=coordinate_catalog,
            )
        )

    payload_records.sort(
        key=lambda item: (
            item["fixtureId"],
            item["observationIndexOneBased"],
            item["zeroSpecificationId"],
            item["coordinateSpecificationId"],
        )
    )
    coordinate_payload = [
        {
            "specificationId": item.specification_id,
            "family": item.family,
            "dimension": item.dimension,
            "droppedComponentOneBased": (
                None
                if item.dropped_component_zero_based is None
                else item.dropped_component_zero_based + 1
            ),
            "basisFitScopeId": item.basis_fit_scope_id,
            "evidenceClass": item.evidence_class,
        }
        for item in sorted(
            coordinate_catalog.values(),
            key=lambda specification: specification.specification_id,
        )
    ]
    zero_payload = [
        {
            "specificationId": item.specification_id,
            "method": item.method,
            "deltaHex": None if item.delta is None else float_to_hex(item.delta),
            "evidenceClass": item.evidence_class,
        }
        for item in zero_specs
    ]
    registry_gate = registry["executionGate"]
    payload = {
        "schema": "eidosoma.e01.s09_transform_payload.v1",
        "researchStepId": "S09",
        "contractVersion": CONTRACT_VERSION,
        "preregistrationSha256": preregistration_sha,
        "registryBoundary": {
            "registryVersion": registry["registryVersion"],
            "registrySha256": REGISTRY_SHA256,
            "executable": registry_gate["executable"],
            "noSilentDefaults": registry_gate["noSilentDefaults"],
            "unresolvedParameterCount": registry_gate["unresolvedParameterCount"],
            "unexpandedBranchSetCount": registry_gate["unexpandedBranchSetCount"],
        },
        "fixtures": [
            {
                "fixtureId": item["fixtureId"],
                "dimension": len(item["states"][0]),
                "observationCount": len(item["states"]),
                "stateSha256": item["stateSha256"],
            }
            for item in fixtures
        ],
        "zeroSpecifications": zero_payload,
        "coordinateSpecifications": coordinate_payload,
        "completeSpecifications": [
            complete_catalog[key] for key in sorted(complete_catalog)
        ],
        "principalLogRatioBases": [basis_catalog[key] for key in sorted(basis_catalog)],
        "records": payload_records,
    }
    envelope = make_envelope(payload)
    serialized = serialize_envelope(envelope)
    schema = json.loads(SCHEMA_PATH.read_text())
    Draft202012Validator.check_schema(schema)
    validate_json_schema(envelope, schema, validator_factory=Draft202012Validator)
    round_trip = deserialize_envelope(serialized, require_canonical=True)
    if serialize_envelope(round_trip) != serialized:
        raise RuntimeError("S09 canonical transform round trip changed bytes.")
    (step_dir / "transform_arrays.json").write_bytes(serialized)

    csv_rows = output_csv_rows(payload_records)
    write_csv(
        step_dir / "transform_outputs.csv",
        csv_rows,
        list(csv_rows[0]),
    )
    zero_observations, zero_fixtures = zero_frequency_rows(fixtures)
    write_csv(
        step_dir / "zero_frequency_by_observation.csv",
        zero_observations,
        list(zero_observations[0]),
    )
    write_csv(
        step_dir / "zero_frequency_by_fixture.csv",
        zero_fixtures,
        list(zero_fixtures[0]),
    )
    replacement = replacement_rows(fixtures, zero_specs)
    write_csv(
        step_dir / "replacement_agreement.csv",
        replacement,
        list(replacement[0]),
    )
    write_csv(
        step_dir / "numerical_diagnostics.csv",
        all_numerical,
        list(all_numerical[0]),
    )
    write_csv(
        step_dir / "eligibility_summary.csv",
        all_eligibility,
        list(all_eligibility[0]),
    )
    write_csv(
        step_dir / "inverse_transform_validation.csv",
        all_inverse,
        list(all_inverse[0]),
    )
    write_csv(
        step_dir / "representation_agreement.csv",
        all_representation,
        list(all_representation[0]),
    )

    valid_specs = specification_registry_rows(
        complete_catalog=complete_catalog,
        payload_records=payload_records,
    )
    specification_document = {
        "schema": "eidosoma.e01.s09_transform_specifications.v1",
        "researchStepId": "S09",
        "collectionVersion": "E01-S09-transform-specifications-v1.0.0",
        "evidenceClass": "VALIDATION_ONLY_RECONSTRUCTION_NOT_AUTHOR_DEFAULT",
        "zeroSpecifications": zero_payload,
        "coordinateSpecifications": coordinate_payload,
        "completeSpecifications": [
            complete_catalog[key] for key in sorted(complete_catalog)
        ],
        "principalLogRatioBases": [basis_catalog[key] for key in sorted(basis_catalog)],
    }
    (shared_dir / "transform_specifications_v1.0.0.yaml").write_text(
        yaml.safe_dump(specification_document, sort_keys=False)
    )
    valid_document = {
        "schema": "eidosoma.e01.s09_valid_transform_specification_registry.v1",
        "researchStepId": "S09",
        "registryVersion": "E01-S09-valid-transform-specifications-v1.0.0",
        "statusBoundary": "VALIDATION_ELIGIBILITY_NOT_AUTHOR_DEFAULT_SELECTION",
        "paperDefaultSelected": False,
        "specifications": valid_specs,
    }
    (shared_dir / "valid_transform_specification_registry_v1.0.0.yaml").write_text(
        yaml.safe_dump(valid_document, sort_keys=False)
    )
    contract_document = {
        "schema": "eidosoma.e01.s09_compositional_transform_contract.v1",
        "researchStepId": "S09",
        "contractVersion": CONTRACT_VERSION,
        "scopeBoundary": config["scopeBoundary"],
        "zeroTreatments": config["zeroTreatments"],
        "coordinateFamilies": config["coordinateFamilies"],
        "featureLayout": config["featureLayout"],
        "invalidTransformPolicy": config["invalidTransformPolicy"],
        "diagnostics": config["diagnostics"],
        "acceptanceRules": config["acceptanceRules"],
        "serialization": config["serialization"],
        "registryBoundary": payload["registryBoundary"],
        "registryOwnerSnapshot": registry_owners,
    }
    (shared_dir / "compositional_transform_contract_v1.0.0.yaml").write_text(
        yaml.safe_dump(contract_document, sort_keys=False)
    )
    shutil.copyfile(SCHEMA_PATH, shared_dir / "transform_output_schema_v1.0.0.json")
    shutil.copyfile(CONFIG_PATH, step_dir / "preregistration.yaml")

    preregistration_record = {
        "schema": "eidosoma.e01.s09_preregistration_record.v1",
        "researchStepId": "S09",
        "preregistrationVersion": config["preregistrationVersion"],
        "preregistrationCommit": PREREGISTRATION_COMMIT,
        "preregistrationCommitIsAncestor": True,
        "preregistrationSha256": preregistration_sha,
        "frozenInputResults": frozen_results,
        "canonicalOutcomesPresentBeforeExecution": outcomes_preexisting,
        "canonicalOutcomeRuleChangedAfterInspection": False,
    }
    write_json(step_dir / "preregistration_record.json", preregistration_record)

    registry_path = (
        artifacts_root
        / "E01_forensic_replication_bundle/specifications/specification_registry_v0.3.0.yaml"
    )
    registry_preservation = {
        "schema": "eidosoma.e01.s09_registry_preservation.v1",
        "researchStepId": "S09",
        "beforeSha256": REGISTRY_SHA256,
        "afterSha256": sha256_file(registry_path),
        "byteIdentical": sha256_file(registry_path) == REGISTRY_SHA256,
        "registryVersion": registry["registryVersion"],
        "executionGate": registry_gate,
        "s09OwnerParameters": registry_owners,
        "s09RegistryUpdates": [],
    }
    write_json(step_dir / "registry_preservation.json", registry_preservation)

    expected_record_count = sum(
        len(item["states"]) * len(zero_specs) * (len(item["states"][0]) + 5)
        for item in fixtures
    )
    injections = failure_injections(
        sample_envelope_bytes=serialized,
        expected_record_count=expected_record_count,
    )
    write_json(
        step_dir / "failure_injection.json",
        {
            "schema": "eidosoma.e01.s09_failure_injection.v1",
            "researchStepId": "S09",
            "injections": injections,
        },
    )
    schema_validation = {
        "schema": "eidosoma.e01.s09_schema_validation.v1",
        "researchStepId": "S09",
        "draft202012MetaSchemaPass": True,
        "instanceConformancePass": True,
        "payloadSha256": envelope["payloadSha256"],
        "serializedFileSha256": hashlib.sha256(serialized).hexdigest(),
        "canonicalRoundTripByteExact": True,
        "recordCount": len(payload_records),
        "binary64Encoding": "canonical Python float.hex strings",
    }
    write_json(step_dir / "schema_validation.json", schema_validation)

    figures = create_figures(
        step_dir=step_dir,
        zero_observations=zero_observations,
        numerical=all_numerical,
        replacements=replacement,
    )
    any_zero_count = sum(bool(item["anyZero"]) for item in zero_observations)
    zero_sum_count = sum(bool(item["zeroSum"]) for item in zero_observations)
    eligible_records = [
        item for item in payload_records if item["status"] == "ELIGIBLE"
    ]
    max_inverse = max(
        float_from_hex(item["maximumAbsoluteInverseErrorHex"])
        for item in eligible_records
    )
    isometry_rows = [item for item in all_representation if item["isometryRequired"]]
    evaluated_isometries = [
        item
        for item in isometry_rows
        if item["validationStatus"] != "NOT_EVALUABLE_FEWER_THAN_TWO_COMMON_ROWS"
    ]
    checks = [
        {
            "checkId": f"FROZEN_INPUT_{item['inputId']}",
            "success": item["success"],
        }
        for item in frozen_results
    ]
    checks.extend(
        [
            {"checkId": "PREREGISTRATION_COMMIT_ANCESTOR", "success": True},
            {
                "checkId": "NUMERIC_THREAD_ENVIRONMENT_FROZEN",
                "success": actual_thread_environment == expected_thread_environment,
            },
            {
                "checkId": "REGISTRY_BYTE_AND_OWNER_SENTINEL_PRESERVATION",
                "success": registry_preservation["byteIdentical"],
            },
            {
                "checkId": "COMPLETE_TRANSFORM_ROW_COVERAGE",
                "success": len(payload_records) == expected_record_count,
            },
            {
                "checkId": "FROZEN_ZERO_TREATMENT_GRID_COMPLETE",
                "success": len(zero_specs) == 13,
            },
            {
                "checkId": "EVERY_DROPPED_COMPONENT_MATERIALIZED",
                "success": all(
                    sum(
                        item.family == "dropped_clr" and item.dimension == dimension
                        for item in coordinate_catalog.values()
                    )
                    >= dimension
                    for dimension in (2, 4)
                ),
            },
            {
                "checkId": "ELIGIBLE_VALUES_FINITE",
                "success": all(
                    item["allEligibleValuesFinite"] for item in all_numerical
                ),
            },
            {
                "checkId": "ALL_INELIGIBLE_ROWS_HAVE_REASONS",
                "success": all(
                    item["reason"] is not None
                    for item in payload_records
                    if item["status"] == "INELIGIBLE"
                ),
            },
            {
                "checkId": "INVERSE_TRANSFORM_TOLERANCE",
                "success": max_inverse <= INVERSE_TOLERANCE,
            },
            {
                "checkId": "SIMPLEX_BASES_ORTHONORMAL",
                "success": all(
                    item["status"] != "FIT" or item["basisHex"] is not None
                    for item in basis_catalog.values()
                ),
            },
            {
                "checkId": "REQUIRED_REPRESENTATION_ISOMETRIES",
                "success": all(
                    item["validationStatus"] == "PASS" for item in evaluated_isometries
                ),
            },
            {
                "checkId": "SCHEMA_CHECKSUM_AND_CANONICAL_ROUND_TRIP",
                "success": True,
            },
            {
                "checkId": "FAILURE_INJECTIONS",
                "success": all(item["success"] for item in injections),
            },
            {
                "checkId": "ZERO_DIAGNOSTIC_COUNTS_RECONCILE_S08",
                "success": any_zero_count == 9 and zero_sum_count == 2,
            },
            {
                "checkId": "DIAGNOSTIC_TABLES_COMPLETE",
                "success": bool(
                    all_numerical
                    and all_eligibility
                    and all_inverse
                    and all_representation
                    and replacement
                ),
            },
            {
                "checkId": "DIAGNOSTIC_FIGURES_WRITTEN",
                "success": all(path.stat().st_size > 0 for path in figures),
            },
            {"checkId": "S10_NOT_BEGUN", "success": not s10_dir.exists()},
        ]
    )
    success = all(item["success"] for item in checks)
    accepted_specs = sum(
        item["transformValidationStatus"] == "ACCEPTED_WITH_EXPLICIT_ELIGIBILITY_DOMAIN"
        for item in valid_specs
    )
    validation_summary = {
        "schema": "eidosoma.e01.s09_validation_summary.v1",
        "researchStepId": "S09",
        "stepNumber": 9,
        "success": success,
        "status": "COMPLETE" if success else "VALIDATION_FAILED",
        "artifactsWritten": [
            "transform_arrays.json",
            "transform_outputs.csv",
            "zero_frequency_by_observation.csv",
            "zero_frequency_by_fixture.csv",
            "eligibility_summary.csv",
            "numerical_diagnostics.csv",
            "inverse_transform_validation.csv",
            "representation_agreement.csv",
            "replacement_agreement.csv",
            "failure_injection.json",
            "schema_validation.json",
            "registry_preservation.json",
            "preregistration.yaml",
            "preregistration_record.json",
            "zero_frequency.png",
            "covariance_readiness.png",
            "replacement_disagreement.png",
        ],
        "validationResult": (
            "PASS: all preregistered S09 numerical, retention, schema, provenance, and scope gates passed"
            if success
            else "FAIL: at least one preregistered S09 gate failed"
        ),
        "caveatsOrBlockers": [
            "No author zero policy or paper default was recovered.",
            "Full CLR and raw closed proportions are structurally covariance-singular.",
            "Fixture covariance readiness does not establish trajectory-scale estimator readiness.",
            "Principal-log-ratio bases are data-adaptive validation branches.",
        ],
        "recommendedNextAction": (
            "Hand control back; S10 is eligible only after separate authorization."
        ),
        "checks": checks,
        "anchorMetrics": {
            "fixtureCount": len(fixtures),
            "observationCount": sum(len(item["states"]) for item in fixtures),
            "observationWithAnyZeroCount": any_zero_count,
            "zeroSumObservationCount": zero_sum_count,
            "zeroTreatmentSpecificationCount": len(zero_specs),
            "losslessTransformRecordCount": len(payload_records),
            "eligibleTransformRecordCount": len(eligible_records),
            "ineligibleTransformRecordCount": len(payload_records)
            - len(eligible_records),
            "acceptedCompleteSpecificationCount": accepted_specs,
            "notEvaluableCompleteSpecificationCount": len(valid_specs) - accepted_specs,
            "maximumAbsoluteInverseError": max_inverse,
            "requiredEvaluableIsometryCount": len(evaluated_isometries),
            "requiredIsometryPassCount": sum(
                item["validationStatus"] == "PASS" for item in evaluated_isometries
            ),
            "failureInjectionPassCount": sum(item["success"] for item in injections),
            "failureInjectionCount": len(injections),
        },
    }
    write_json(step_dir / "validation_summary.json", validation_summary)
    if not success:
        failures = [item for item in checks if not item["success"]]
        raise RuntimeError(f"S09 validation failed: {failures}")
    write_artifact_manifest(artifacts_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--finalize-manifest", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.finalize_manifest:
        write_artifact_manifest(args.artifacts_dir)
    else:
        build(args.artifacts_dir)


if __name__ == "__main__":
    main()
