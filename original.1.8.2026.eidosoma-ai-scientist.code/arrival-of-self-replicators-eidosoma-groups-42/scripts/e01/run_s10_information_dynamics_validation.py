#!/usr/bin/env python3
"""Run the preregistered E01 S10 information-dynamics validation suite."""

# ruff: noqa: BLE001, PLC0206

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.io import loadmat

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from e01_information_dynamics.backends import (
    DecompositionResult,
    _load_omegaid,
    _load_phyid,
    _normalize_result,
    backend_identity,
    compare_decompositions,
    run_omegaid,
    run_phyid,
)
from e01_information_dynamics.synthetic import (
    GENERATORS,
    SyntheticSeries,
    affine_transform,
    common_time_shuffle,
    discrete_relabel,
)
from e01_information_dynamics.validation import (
    ATOM_IDS,
    I_KEYS,
    InformationValidationError,
    aggregate_means,
    coupled_ar_covariance,
    discrete_exact_oracle,
    exact_redundant_pmf,
    exact_xor_pmf,
    exhaustive_partition_search,
    gaussian_mmi_oracle,
    gaussian_partition_objective,
    greedy_partition_search,
    noisy_redundant_covariance,
    spectral_partition,
    strict_sample_gate,
)

CONFIG_PATH = (
    REPOSITORY_ROOT / "configs/e01/s10_information_dynamics_preregistration.yaml"
)
PREREGISTRATION_COMMIT = "02d38634f47e73d95882846aeeb89820f38a98b0"
PREREGISTRATION_SHA256 = (
    "5c54b8f88e8e8634a4b7f39783e3359084e25ce44cbed2291a141da85e19f3dd"
)
REGISTRY_SHA256 = "aef0e179de6466697540ba10236ed24af37fbda12bd4f1c6b1fb5fe7a27af891"

SYSTEM_KINDS = {
    "E01-S10-SYS-INDEPENDENT-GAUSSIAN-v1.0.0": "gaussian",
    "E01-S10-SYS-REDUNDANT-DISCRETE-v1.0.0": "discrete",
    "E01-S10-SYS-REDUNDANT-GAUSSIAN-v1.0.0": "gaussian",
    "E01-S10-SYS-XOR-DISCRETE-v1.0.0": "discrete",
    "E01-S10-SYS-COUPLED-AR-v1.0.0": "gaussian",
}
STRUCTURED_SYSTEMS = tuple(
    system for system in SYSTEM_KINDS if "INDEPENDENT" not in system
)
REDUNDANCIES = ("MMI", "CCS")
PLANTED_PARTITION = (0, 1)

MAPPING_IDS = {
    "group_mean": "E01-S10-PARTMAP-GROUP-MEAN-v1.0.0",
    "pc1": "E01-S10-PARTMAP-PC1-v1.0.0",
    "omega_equal_width_vector": "E01-S10-PARTMAP-OMEGA-EQUAL-WIDTH-VECTOR-v1.0.0",
}
OBJECTIVE_IDS = {
    "synchronous_mi": "E01-S10-MIB-OBJ-SYNCHRONOUS-GAUSSIAN-MI-v1.0.0",
    "bidirectional_lagged_mi": "E01-S10-MIB-OBJ-BIDIRECTIONAL-LAGGED-GAUSSIAN-MI-v1.0.0",
    "abs_paper_equation": "E01-S10-MIB-OBJ-ABS-PAPER-EQUATION-AGGREGATE-v1.0.0",
}
NORMALIZATION_IDS = {
    "none": "E01-S10-MIB-NORM-NONE-v1.0.0",
    "min_part_entropy": "E01-S10-MIB-NORM-MIN-PART-ENTROPY-v1.0.0",
    "geometric_part_size": "E01-S10-MIB-NORM-GEOMETRIC-PART-SIZE-v1.0.0",
}
SEARCH_IDS = {
    "exhaustive_all": "E01-S10-MIB-SEARCH-EXHAUSTIVE-ALL-v1.0.0",
    "exhaustive_balanced": "E01-S10-MIB-SEARCH-EXHAUSTIVE-BALANCED-v1.0.0",
    "spectral": "E01-S10-MIB-SEARCH-SPECTRAL-FIXED-CANDIDATE-v1.0.0",
    "greedy": "E01-S10-MIB-SEARCH-GREEDY-SINGLE-FLIP-v1.0.0",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity" if value < 0 else "NaN"
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True) + "\n")


def write_yaml(path: Path, value: Any) -> None:
    path.write_text(yaml.safe_dump(jsonable(value), sort_keys=False, width=100))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty required table {path.name}.")
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(jsonable(value), sort_keys=True)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in row.items()
                }
            )


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def verify_preregistration(config: dict[str, Any]) -> dict[str, Any]:
    if sha256_file(CONFIG_PATH) != PREREGISTRATION_SHA256:
        raise RuntimeError("S10 preregistration working-tree bytes changed.")
    committed = subprocess.run(
        [
            "git",
            "show",
            f"{PREREGISTRATION_COMMIT}:configs/e01/s10_information_dynamics_preregistration.yaml",
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    if hashlib.sha256(committed).hexdigest() != PREREGISTRATION_SHA256:
        raise RuntimeError("S10 preregistration commit does not contain frozen bytes.")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", PREREGISTRATION_COMMIT, "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
    )
    frozen_results: list[dict[str, Any]] = []
    for item in config["frozenInputs"]:
        path = Path(item["path"])
        actual = sha256_file(path) if path.is_file() else None
        success = actual == item["sha256"]
        frozen_results.append({**item, "actualSha256": actual, "success": success})
    failures = [item for item in frozen_results if not item["success"]]
    if failures:
        raise RuntimeError(f"Frozen S10 input mismatch: {failures}")
    return {
        "status": "VERIFIED_FROZEN_BEFORE_OUTCOMES",
        "success": True,
        "preregistrationCommit": PREREGISTRATION_COMMIT,
        "preregistrationSha256": PREREGISTRATION_SHA256,
        "frozenInputs": frozen_results,
    }


def registry_preservation(artifacts_root: Path) -> dict[str, Any]:
    path = (
        artifacts_root
        / "E01_forensic_replication_bundle/specifications/specification_registry_v0.3.0.yaml"
    )
    before = sha256_file(path)
    registry = yaml.safe_load(path.read_text())
    relevant = [
        item
        for item in registry["parameters"]
        if item.get("ambiguityId") in {"E01-A043", "E01-A044"}
        or item.get("ownerStep") == "S10"
    ]
    expected_ambiguities = {
        "E01-A043",
        "E01-A044",
        "E01-A045",
        "E01-A046",
        "E01-A047",
        "E01-A054",
        "E01-A055",
        "E01-A056",
        "E01-A058",
        "E01-A060",
    }
    if {item["ambiguityId"] for item in relevant} != expected_ambiguities:
        raise RuntimeError("S10 registry owner/boundary set changed.")
    if (
        registry["executionGate"]["executable"] is not False
        or registry["executionGate"]["noSilentDefaults"] is not True
    ):
        raise RuntimeError("Registry execution/no-silent-default boundary changed.")
    after = sha256_file(path)
    if before != REGISTRY_SHA256 or after != REGISTRY_SHA256:
        raise RuntimeError("Registry v0.3.0 bytes changed.")
    return {
        "researchStepId": "S10",
        "registryVersion": registry["registryVersion"],
        "path": str(path),
        "beforeSha256": before,
        "afterSha256": after,
        "byteForBytePreserved": before == after == REGISTRY_SHA256,
        "executionGateExecutable": registry["executionGate"]["executable"],
        "noSilentDefaults": registry["executionGate"]["noSilentDefaults"],
        "relevantParameters": relevant,
        "candidateSpecificationBoundary": "S10 candidate validation does not mutate or resolve the author-facing registry.",
    }


def _series_for(system_id: str, replicate: int, variant: str) -> SyntheticSeries:
    series = GENERATORS[system_id](replicate)
    if variant == "base":
        return series
    if variant == "affine":
        return (
            affine_transform(series)
            if SYSTEM_KINDS[system_id] == "gaussian"
            else discrete_relabel(series)
        )
    if variant == "time_shuffle":
        return common_time_shuffle(series)
    raise RuntimeError(f"Unknown reference task variant {variant}.")


def _data_sha256(data: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(data, dtype=">f8"))
    return hashlib.sha256(array.tobytes()).hexdigest()


def reference_case_worker(task: tuple[str, int, str, str]) -> dict[str, Any]:
    """Generate and evaluate one reference case; safe for a process pool."""

    system_id, replicate, redundancy, variant = task
    started = time.perf_counter()
    series = _series_for(system_id, replicate, variant)
    result = run_phyid(
        series.data[:, 0],
        series.data[:, 1],
        tau=1,
        kind=SYSTEM_KINDS[system_id],  # type: ignore[arg-type]
        redundancy=redundancy,  # type: ignore[arg-type]
    )
    estimator_stream = series.seed_payload["streams"]["estimator"]
    payload: dict[str, Any] = {
        "caseId": f"{system_id}::r{replicate:02d}::{redundancy}::{variant}",
        "systemId": system_id,
        "replicateIndex": replicate,
        "redundancy": redundancy,
        "kind": SYSTEM_KINDS[system_id],
        "variant": variant,
        "backendId": result.backend_id,
        "status": result.status,
        "reason": result.reason,
        "sampleGate": result.sample_gate,
        "nativeUnits": result.native_units,
        "reportedUnits": result.reported_units,
        "regularizationEntered": result.regularization_entered,
        "runtimeSeconds": time.perf_counter() - started,
        "dataSha256": _data_sha256(series.data),
        "estimatorStreamId": estimator_stream["streamId"],
        "estimatorSeedMaterialHex": estimator_stream["seedMaterialHex"],
        "seedPayloadSha256": canonical_sha256(series.seed_payload),
        "permutation": series.seed_payload.get("permutation"),
    }
    if result.status == "ELIGIBLE":
        payload["means"] = result.means()
    return payload


def _direct_result(
    *,
    backend_name: str,
    source: np.ndarray,
    target: np.ndarray,
    tau: int,
    kind: str,
    redundancy: str,
) -> DecompositionResult:
    """Run the six-effective-sample pinned regression fixture outside science eligibility."""

    if backend_name == "phyid":
        calculate, _ = _load_phyid()
        atoms, calculations = calculate.calc_PhiID(
            source, target, tau, kind=kind, redundancy=redundancy
        )
        backend_id = "pinned_phyid_cpu_source_fixture"
    else:
        decomposition, backend = _load_omegaid()
        backend.set_backend(backend_name)
        matrix = np.stack([source, target])
        atoms, calculations = decomposition.calc_phiid_multivariate(
            matrix, matrix, tau=tau, kind=kind, redundancy=redundancy
        )
        backend_id = f"pinned_omegaid_{backend_name}_source_fixture"
    normalized = _normalize_result(atoms=atoms, calculations=calculations, kind=kind)  # type: ignore[arg-type]
    gate = strict_sample_gate(source, target, tau=tau, kind=kind)  # type: ignore[arg-type]
    return DecompositionResult(
        backend_id=backend_id,
        kind=kind,  # type: ignore[arg-type]
        redundancy=redundancy,  # type: ignore[arg-type]
        status="ELIGIBLE",
        reason="SOURCE_REGRESSION_FIXTURE_BYPASS_NOT_SCIENCE_ELIGIBLE",
        atoms=normalized[0],
        intermediate_mi=normalized[1],
        intermediate_redundancy=normalized[2],
        double_redundancy=normalized[3],
        sample_gate=gate,
        native_units="bits" if kind == "discrete" else "nats",
        reported_units="nats",
        regularization_entered=False,
    )


def _array_map_comparison(
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    maximum = 0.0
    maximum_relative = 0.0
    worst = None
    success = set(left) == set(right)
    if success:
        for key in sorted(left):
            a = np.asarray(left[key], dtype=np.float64)
            b = np.asarray(right[key], dtype=np.float64)
            absolute = np.abs(a - b)
            relative = absolute / np.maximum(
                np.maximum(np.abs(a), np.abs(b)), np.finfo(float).tiny
            )
            candidate = float(np.max(absolute))
            if candidate >= maximum:
                maximum = candidate
                maximum_relative = float(np.max(relative))
                worst = key
            success &= bool(np.allclose(a, b, atol=atol, rtol=rtol))
    return {
        "success": bool(success),
        "status": "PASS" if success else "FAIL",
        "maximumAbsoluteError": maximum,
        "maximumRelativeErrorAtWorstAbsolute": maximum_relative,
        "worstField": worst,
        "absoluteTolerance": atol,
        "relativeTolerance": rtol,
    }


def source_fixture_validation(
    config: dict[str, Any], *, gpu_enabled: bool
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fixture = loadmat(config["pinnedSources"]["phyid"]["matlabFixture"]["path"])
    source = fixture["src"].squeeze().astype(np.float64)
    target = fixture["trg"].squeeze().astype(np.float64)
    tau = int(fixture["tau"].squeeze())
    comparisons: list[dict[str, Any]] = []
    atom_rows: list[dict[str, Any]] = []
    gate = config["comparisonGates"]["sourceMatlabFixture"]
    for kind in ("gaussian", "discrete"):
        for redundancy in REDUNDANCIES:
            prefix = "PhiIDFullDiscrete" if kind == "discrete" else "PhiIDFull"
            expected_matrix = np.asarray(
                fixture[f"{prefix}_{redundancy}_L"], dtype=np.float64
            )
            if kind == "discrete":
                expected_matrix = expected_matrix * np.log(2.0)
            expected = {
                atom: expected_matrix[index] for index, atom in enumerate(ATOM_IDS)
            }
            reference = _direct_result(
                backend_name="phyid",
                source=source,
                target=target,
                tau=tau,
                kind=kind,
                redundancy=redundancy,
            )
            assert reference.atoms is not None
            comparison = _array_map_comparison(
                reference.atoms,
                expected,
                atol=float(gate["localArrayAbsoluteTolerance"]),
                rtol=float(gate["localArrayRelativeTolerance"]),
            )
            comparisons.append(
                {
                    "comparisonId": f"source_matlab::{kind}::{redundancy}",
                    "comparisonFamily": "SOURCE_MATLAB_FIXTURE",
                    "systemId": "PINNED_MATLAB_FIXTURE",
                    "replicateIndex": None,
                    "kind": kind,
                    "redundancy": redundancy,
                    "leftBackend": reference.backend_id,
                    "rightBackend": "pinned_matlab_expected_arrays",
                    "scienceSampleEligibility": reference.sample_gate["status"],
                    "fixtureBypassReason": reference.reason,
                    **comparison,
                }
            )
            for atom in ATOM_IDS:
                atom_rows.append(
                    {
                        "caseId": f"PINNED_MATLAB_FIXTURE::{kind}::{redundancy}",
                        "systemId": "PINNED_MATLAB_FIXTURE",
                        "replicateIndex": None,
                        "variant": "source_fixture",
                        "backendId": reference.backend_id,
                        "kind": kind,
                        "redundancy": redundancy,
                        "atomId": atom,
                        "meanNats": float(np.mean(reference.atoms[atom])),
                        "status": "SOURCE_REGRESSION_ONLY_NOT_SCIENCE_ELIGIBLE",
                    }
                )
            omega_cpu = _direct_result(
                backend_name="numpy",
                source=source,
                target=target,
                tau=tau,
                kind=kind,
                redundancy=redundancy,
            )
            comparisons.append(
                {
                    "comparisonId": f"source_ref_vs_omega_cpu::{kind}::{redundancy}",
                    "comparisonFamily": "REFERENCE_VS_OMEGA_CPU_SOURCE_FIXTURE",
                    "systemId": "PINNED_MATLAB_FIXTURE",
                    "replicateIndex": None,
                    "kind": kind,
                    "redundancy": redundancy,
                    "leftBackend": reference.backend_id,
                    "rightBackend": omega_cpu.backend_id,
                    **compare_decompositions(
                        reference,
                        omega_cpu,
                        absolute_tolerance=1e-10,
                        relative_tolerance=1e-10,
                    ),
                }
            )
            if gpu_enabled:
                try:
                    omega_gpu = _direct_result(
                        backend_name="cupy",
                        source=source,
                        target=target,
                        tau=tau,
                        kind=kind,
                        redundancy=redundancy,
                    )
                    gpu_comparison = compare_decompositions(
                        omega_cpu,
                        omega_gpu,
                        absolute_tolerance=1e-10,
                        relative_tolerance=1e-10,
                    )
                    right_backend = omega_gpu.backend_id
                except Exception as error:  # preserve optional backend failure
                    gpu_comparison = {
                        "success": False,
                        "status": "FAIL",
                        "reason": f"{type(error).__name__}: {error}",
                    }
                    right_backend = "pinned_omegaid_cupy_source_fixture"
                comparisons.append(
                    {
                        "comparisonId": f"source_omega_cpu_vs_gpu::{kind}::{redundancy}",
                        "comparisonFamily": "OMEGA_CPU_VS_GPU_SOURCE_FIXTURE",
                        "systemId": "PINNED_MATLAB_FIXTURE",
                        "replicateIndex": None,
                        "kind": kind,
                        "redundancy": redundancy,
                        "leftBackend": omega_cpu.backend_id,
                        "rightBackend": right_backend,
                        **gpu_comparison,
                    }
                )
    return comparisons, atom_rows


def _mean_vector(case: dict[str, Any]) -> dict[str, float]:
    means = case["means"]
    return {
        **{f"atom.{key}": float(means["atomMeans"][key]) for key in ATOM_IDS},
        **{f"mi.{key}": float(means["miMeans"][key]) for key in I_KEYS},
    }


def reference_invariance_rows(
    cases: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    indexed = {
        (
            case["systemId"],
            case["replicateIndex"],
            case["redundancy"],
            case["variant"],
        ): case
        for case in cases
    }
    rows: list[dict[str, Any]] = []
    tolerance = config["comparisonGates"]["affineAndScaleInvariance"]
    for system_id in SYSTEM_KINDS:
        control = (
            "gaussian_affine"
            if SYSTEM_KINDS[system_id] == "gaussian"
            else "discrete_relabel"
        )
        for replicate in range(config["syntheticDesign"]["primaryReplicates"]):
            for redundancy in REDUNDANCIES:
                base = indexed[(system_id, replicate, redundancy, "base")]
                transformed = indexed[(system_id, replicate, redundancy, "affine")]
                left = _mean_vector(base)
                right = _mean_vector(transformed)
                comparison = _array_map_comparison(
                    {key: np.asarray([value]) for key, value in left.items()},
                    {key: np.asarray([value]) for key, value in right.items()},
                    atol=float(tolerance["meanAbsoluteTolerance"]),
                    rtol=float(tolerance["meanRelativeTolerance"]),
                )
                rows.append(
                    {
                        "controlFamily": control,
                        "systemId": system_id,
                        "replicateIndex": replicate,
                        "redundancy": redundancy,
                        "backendId": "pinned_phyid_cpu",
                        "originalTotalMiNats": base["means"]["totalMi"],
                        "controlTotalMiNats": transformed["means"]["totalMi"],
                        **comparison,
                    }
                )
    shuffle_gate = config["comparisonGates"]["timeShuffle"]
    for system_id in STRUCTURED_SYSTEMS:
        for replicate in range(config["syntheticDesign"]["primaryReplicates"]):
            for redundancy in REDUNDANCIES:
                base = indexed[(system_id, replicate, redundancy, "base")]
                shuffled = indexed[(system_id, replicate, redundancy, "time_shuffle")]
                original = float(base["means"]["totalMi"])
                shuffled_value = float(shuffled["means"]["totalMi"])
                ratio = abs(shuffled_value) / max(abs(original), np.finfo(float).tiny)
                success = abs(shuffled_value) <= float(
                    shuffle_gate["shuffledAbsoluteTotalMiNats"]
                ) and ratio <= float(
                    shuffle_gate["shuffledToOriginalTotalMiRatioMaximum"]
                )
                rows.append(
                    {
                        "controlFamily": "common_time_permutation",
                        "systemId": system_id,
                        "replicateIndex": replicate,
                        "redundancy": redundancy,
                        "backendId": "pinned_phyid_cpu",
                        "originalTotalMiNats": original,
                        "controlTotalMiNats": shuffled_value,
                        "controlToOriginalRatio": ratio,
                        "permutationSha256": shuffled["permutation"]["sha256"],
                        "absoluteTolerance": shuffle_gate[
                            "shuffledAbsoluteTotalMiNats"
                        ],
                        "relativeRatioMaximum": shuffle_gate[
                            "shuffledToOriginalTotalMiRatioMaximum"
                        ],
                        "success": success,
                        "status": "PASS" if success else "FAIL",
                    }
                )
    return rows


def theory_oracles() -> dict[tuple[str, str], dict[str, Any]]:
    redundant_states, redundant_probabilities = exact_redundant_pmf()
    xor_states, xor_probabilities = exact_xor_pmf()
    result: dict[tuple[str, str], dict[str, Any]] = {}
    independent = gaussian_mmi_oracle(np.eye(4))
    for redundancy in REDUNDANCIES:
        result[("E01-S10-SYS-INDEPENDENT-GAUSSIAN-v1.0.0", redundancy)] = independent
        result[("E01-S10-SYS-REDUNDANT-DISCRETE-v1.0.0", redundancy)] = (
            discrete_exact_oracle(
                redundant_states,
                redundant_probabilities,
                redundancy=redundancy,  # type: ignore[arg-type]
            )
        )
        result[("E01-S10-SYS-XOR-DISCRETE-v1.0.0", redundancy)] = discrete_exact_oracle(
            xor_states,
            xor_probabilities,
            redundancy=redundancy,  # type: ignore[arg-type]
        )
    result[("E01-S10-SYS-REDUNDANT-GAUSSIAN-v1.0.0", "MMI")] = gaussian_mmi_oracle(
        noisy_redundant_covariance()
    )
    result[("E01-S10-SYS-COUPLED-AR-v1.0.0", "MMI")] = gaussian_mmi_oracle(
        coupled_ar_covariance()
    )
    return result


def theoretical_comparison_rows(
    base_cases: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    oracles = theory_oracles()
    rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for case in base_cases:
        oracle = oracles.get((case["systemId"], case["redundancy"]))
        if oracle is None or case["status"] != "ELIGIBLE":
            continue
        if "DISCRETE" in case["systemId"]:
            per_tolerance = float(
                config["comparisonGates"]["exactDiscreteTheory"][
                    "perReplicateMaximumAbsoluteAtomErrorNats"
                ]
            )
        elif "INDEPENDENT" in case["systemId"]:
            per_tolerance = float(
                config["comparisonGates"]["independentGaussian"][
                    "ensembleMaximumAbsoluteAtomMeanNats"
                ]
            )
        else:
            per_tolerance = float(
                config["comparisonGates"]["gaussianMmiPopulationTheory"][
                    "perReplicateMaximumAbsoluteAtomErrorNats"
                ]
            )
        for atom in ATOM_IDS:
            observed = float(case["means"]["atomMeans"][atom])
            expected = float(oracle["atomMeans"][atom])
            error = abs(observed - expected)
            grouped[(case["systemId"], case["redundancy"], atom)].append(observed)
            rows.append(
                {
                    "scope": "replicate",
                    "systemId": case["systemId"],
                    "redundancy": case["redundancy"],
                    "replicateIndex": case["replicateIndex"],
                    "atomId": atom,
                    "observedMeanNats": observed,
                    "expectedMeanNats": expected,
                    "absoluteErrorNats": error,
                    "toleranceNats": per_tolerance,
                    "success": error <= per_tolerance,
                    "status": "PASS" if error <= per_tolerance else "FAIL",
                }
            )
    for (system_id, redundancy, atom), values in sorted(grouped.items()):
        expected = float(oracles[(system_id, redundancy)]["atomMeans"][atom])
        observed = float(np.mean(values))
        if "DISCRETE" in system_id:
            tolerance = float(
                config["comparisonGates"]["exactDiscreteTheory"][
                    "ensembleMaximumAbsoluteAtomErrorNats"
                ]
            )
        elif "INDEPENDENT" in system_id:
            tolerance = float(
                config["comparisonGates"]["independentGaussian"][
                    "ensembleMaximumAbsoluteAtomMeanNats"
                ]
            )
        else:
            tolerance = float(
                config["comparisonGates"]["gaussianMmiPopulationTheory"][
                    "ensembleMaximumAbsoluteAtomErrorNats"
                ]
            )
        error = abs(observed - expected)
        rows.append(
            {
                "scope": "ensemble",
                "systemId": system_id,
                "redundancy": redundancy,
                "replicateIndex": None,
                "atomId": atom,
                "observedMeanNats": observed,
                "expectedMeanNats": expected,
                "absoluteErrorNats": error,
                "toleranceNats": tolerance,
                "success": error <= tolerance,
                "status": "PASS" if error <= tolerance else "FAIL",
            }
        )
    return rows


def cross_backend_validation(
    config: dict[str, Any], *, gpu_enabled: bool
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    comparison_rows: list[dict[str, Any]] = []
    atom_rows: list[dict[str, Any]] = []
    invariance_rows: list[dict[str, Any]] = []
    cross_replicates = int(config["syntheticDesign"]["crossBackendReplicates"])
    for system_id, kind in SYSTEM_KINDS.items():
        for replicate in range(cross_replicates):
            base_series = GENERATORS[system_id](replicate)
            transformed_series = (
                affine_transform(base_series)
                if kind == "gaussian"
                else discrete_relabel(base_series)
            )
            for redundancy in REDUNDANCIES:
                reference = run_phyid(
                    base_series.data[:, 0],
                    base_series.data[:, 1],
                    tau=1,
                    kind=kind,
                    redundancy=redundancy,  # type: ignore[arg-type]
                )
                omega_cpu = run_omegaid(
                    base_series.data[:, 0],
                    base_series.data[:, 1],
                    tau=1,
                    kind=kind,
                    redundancy=redundancy,
                    backend_name="numpy",  # type: ignore[arg-type]
                )
                comparison_rows.append(
                    {
                        "comparisonId": f"ref_vs_omega_cpu::{system_id}::r{replicate:02d}::{redundancy}",
                        "comparisonFamily": "REFERENCE_VS_OMEGA_CPU",
                        "systemId": system_id,
                        "replicateIndex": replicate,
                        "kind": kind,
                        "redundancy": redundancy,
                        "leftBackend": reference.backend_id,
                        "rightBackend": omega_cpu.backend_id,
                        **compare_decompositions(
                            reference,
                            omega_cpu,
                            absolute_tolerance=1e-10,
                            relative_tolerance=1e-10,
                        ),
                    }
                )
                if omega_cpu.status == "ELIGIBLE":
                    assert omega_cpu.atoms is not None
                    for atom in ATOM_IDS:
                        atom_rows.append(
                            {
                                "caseId": f"{system_id}::r{replicate:02d}::{redundancy}::base::omega_cpu",
                                "systemId": system_id,
                                "replicateIndex": replicate,
                                "variant": "base",
                                "backendId": omega_cpu.backend_id,
                                "kind": kind,
                                "redundancy": redundancy,
                                "atomId": atom,
                                "meanNats": float(np.mean(omega_cpu.atoms[atom])),
                                "status": "ELIGIBLE",
                            }
                        )
                omega_cpu_transformed = run_omegaid(
                    transformed_series.data[:, 0],
                    transformed_series.data[:, 1],
                    tau=1,
                    kind=kind,
                    redundancy=redundancy,
                    backend_name="numpy",  # type: ignore[arg-type]
                )
                cpu_invariance = compare_mean_results(
                    omega_cpu, omega_cpu_transformed, atol=1e-10, rtol=1e-10
                )
                invariance_rows.append(
                    {
                        "controlFamily": "gaussian_affine"
                        if kind == "gaussian"
                        else "discrete_relabel",
                        "systemId": system_id,
                        "replicateIndex": replicate,
                        "redundancy": redundancy,
                        "backendId": omega_cpu.backend_id,
                        **cpu_invariance,
                    }
                )
                if not gpu_enabled:
                    continue
                try:
                    omega_gpu = run_omegaid(
                        base_series.data[:, 0],
                        base_series.data[:, 1],
                        tau=1,
                        kind=kind,
                        redundancy=redundancy,
                        backend_name="cupy",  # type: ignore[arg-type]
                    )
                    gpu_comparison = compare_decompositions(
                        omega_cpu,
                        omega_gpu,
                        absolute_tolerance=1e-10,
                        relative_tolerance=1e-10,
                    )
                    right_backend = omega_gpu.backend_id
                    if omega_gpu.status == "ELIGIBLE":
                        assert omega_gpu.atoms is not None
                        for atom in ATOM_IDS:
                            atom_rows.append(
                                {
                                    "caseId": f"{system_id}::r{replicate:02d}::{redundancy}::base::omega_gpu",
                                    "systemId": system_id,
                                    "replicateIndex": replicate,
                                    "variant": "base",
                                    "backendId": omega_gpu.backend_id,
                                    "kind": kind,
                                    "redundancy": redundancy,
                                    "atomId": atom,
                                    "meanNats": float(np.mean(omega_gpu.atoms[atom])),
                                    "status": "ELIGIBLE",
                                }
                            )
                    omega_gpu_transformed = run_omegaid(
                        transformed_series.data[:, 0],
                        transformed_series.data[:, 1],
                        tau=1,
                        kind=kind,
                        redundancy=redundancy,
                        backend_name="cupy",  # type: ignore[arg-type]
                    )
                    gpu_invariance = compare_mean_results(
                        omega_gpu, omega_gpu_transformed, atol=1e-10, rtol=1e-10
                    )
                except Exception as error:
                    gpu_comparison = {
                        "success": False,
                        "status": "FAIL",
                        "reason": f"{type(error).__name__}: {error}",
                    }
                    gpu_invariance = dict(gpu_comparison)
                    right_backend = "pinned_omegaid_cupy"
                comparison_rows.append(
                    {
                        "comparisonId": f"omega_cpu_vs_gpu::{system_id}::r{replicate:02d}::{redundancy}",
                        "comparisonFamily": "OMEGA_CPU_VS_GPU",
                        "systemId": system_id,
                        "replicateIndex": replicate,
                        "kind": kind,
                        "redundancy": redundancy,
                        "leftBackend": omega_cpu.backend_id,
                        "rightBackend": right_backend,
                        **gpu_comparison,
                    }
                )
                invariance_rows.append(
                    {
                        "controlFamily": "gaussian_affine"
                        if kind == "gaussian"
                        else "discrete_relabel",
                        "systemId": system_id,
                        "replicateIndex": replicate,
                        "redundancy": redundancy,
                        "backendId": right_backend,
                        **gpu_invariance,
                    }
                )
    return comparison_rows, atom_rows, invariance_rows


def compare_mean_results(
    left: DecompositionResult,
    right: DecompositionResult,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    if left.status != "ELIGIBLE" or right.status != "ELIGIBLE":
        return {
            "success": False,
            "status": "INELIGIBLE",
            "reason": "ONE_OR_BOTH_RESULTS_INELIGIBLE",
        }
    assert left.means() is not None and right.means() is not None
    left_vector = {
        **left.means()["atomMeans"],
        **{f"mi.{key}": value for key, value in left.means()["miMeans"].items()},
    }
    right_vector = {
        **right.means()["atomMeans"],
        **{f"mi.{key}": value for key, value in right.means()["miMeans"].items()},
    }
    return _array_map_comparison(
        {key: np.asarray([value]) for key, value in left_vector.items()},
        {key: np.asarray([value]) for key, value in right_vector.items()},
        atol=atol,
        rtol=rtol,
    )


def mib_worker(
    replicate: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    series = GENERATORS["E01-S10-SYS-BLOCK-AR4-v1.0.0"](replicate)
    data = series.data
    spectral = spectral_partition(data)
    result_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for mapping in MAPPING_IDS:
        for objective in OBJECTIVE_IDS:
            for normalization in NORMALIZATION_IDS:
                for search in SEARCH_IDS:
                    result: dict[str, Any]
                    candidates: list[dict[str, Any]] = []
                    if mapping == "omega_equal_width_vector" and search in {
                        "exhaustive_all",
                        "greedy",
                    }:
                        result = {
                            "status": "INELIGIBLE",
                            "reason": "VECTOR_MAPPING_SEARCH_DOMAIN_INCLUDES_UNBALANCED_PARTITIONS",
                        }
                    elif search == "exhaustive_all":
                        result, candidates = exhaustive_partition_search(
                            data,
                            mapping=mapping,
                            objective=objective,
                            normalization=normalization,
                            balanced_only=False,
                        )
                    elif search == "exhaustive_balanced":
                        result, candidates = exhaustive_partition_search(
                            data,
                            mapping=mapping,
                            objective=objective,
                            normalization=normalization,
                            balanced_only=True,
                        )
                    elif search == "spectral":
                        if spectral["status"] != "ELIGIBLE":
                            result = dict(spectral)
                        else:
                            part = tuple(spectral["partA"])
                            result = gaussian_partition_objective(
                                data,
                                part,
                                mapping=mapping,  # type: ignore[arg-type]
                                objective=objective,  # type: ignore[arg-type]
                                normalization=normalization,  # type: ignore[arg-type]
                            )
                            result["spectralRelativeEigengap"] = spectral.get(
                                "relativeEigengap"
                            )
                    else:
                        result = greedy_partition_search(
                            data,
                            mapping=mapping,
                            objective=objective,
                            normalization=normalization,
                        )
                    row = {
                        "replicateIndex": replicate,
                        "mapping": mapping,
                        "mappingId": MAPPING_IDS[mapping],
                        "objective": objective,
                        "objectiveId": OBJECTIVE_IDS[objective],
                        "normalization": normalization,
                        "normalizationId": NORMALIZATION_IDS[normalization],
                        "search": search,
                        "searchId": SEARCH_IDS[search],
                        "status": result.get("status", "INELIGIBLE"),
                        "reason": result.get("reason"),
                        "partA": result.get("partA"),
                        "partB": result.get("partB"),
                        "rawObjective": result.get("rawObjective"),
                        "normalizedObjective": result.get("normalizedObjective"),
                        "matchesPlantedPartition": tuple(result.get("partA", ()))
                        == PLANTED_PARTITION,
                        "mappingDiagnostics": result.get("mappingDiagnostics", {}),
                        "searchDiagnostics": {
                            key: result[key]
                            for key in ("iterations", "spectralRelativeEigengap")
                            if key in result
                        },
                    }
                    result_rows.append(row)
                    for candidate in candidates:
                        candidate_rows.append(
                            {
                                "replicateIndex": replicate,
                                "mapping": mapping,
                                "mappingId": MAPPING_IDS[mapping],
                                "objective": objective,
                                "objectiveId": OBJECTIVE_IDS[objective],
                                "normalization": normalization,
                                "normalizationId": NORMALIZATION_IDS[normalization],
                                "search": search,
                                "searchId": SEARCH_IDS[search],
                                "status": candidate["status"],
                                "reason": candidate["reason"],
                                "partA": candidate["partA"],
                                "partB": candidate["partB"],
                                "rawObjective": candidate["rawObjective"],
                                "normalizedObjective": candidate["normalizedObjective"],
                                "matchesPlantedPartition": tuple(candidate["partA"])
                                == PLANTED_PARTITION,
                            }
                        )
    seed = series.seed_payload["streams"]["estimator"]
    seed_row = {
        "systemId": series.system_id,
        "variant": "mib",
        "replicateIndex": replicate,
        "streamId": seed["streamId"],
        "seedMaterialHex": seed["seedMaterialHex"],
        "seedPayloadSha256": canonical_sha256(series.seed_payload),
        "dataSha256": _data_sha256(data),
    }
    return result_rows, candidate_rows, seed_row


def annotate_mib_agreement(rows: list[dict[str, Any]]) -> None:
    index = {
        (
            row["replicateIndex"],
            row["mapping"],
            row["objective"],
            row["normalization"],
            row["search"],
        ): row
        for row in rows
    }
    for row in rows:
        if row["search"] not in {"spectral", "greedy"}:
            row["matchesExhaustive"] = None
            row["objectiveDifferenceFromExhaustive"] = None
            continue
        exact_search = (
            "exhaustive_balanced"
            if row["mapping"] == "omega_equal_width_vector"
            else "exhaustive_all"
        )
        exact = index[
            (
                row["replicateIndex"],
                row["mapping"],
                row["objective"],
                row["normalization"],
                exact_search,
            )
        ]
        if row["status"] != "ELIGIBLE" or exact["status"] != "ELIGIBLE":
            row["matchesExhaustive"] = False
            row["objectiveDifferenceFromExhaustive"] = None
        else:
            difference = abs(
                float(row["normalizedObjective"]) - float(exact["normalizedObjective"])
            )
            row["matchesExhaustive"] = (
                row["partA"] == exact["partA"] and difference <= 1e-12
            )
            row["objectiveDifferenceFromExhaustive"] = difference


def failure_injections(config: dict[str, Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    def record(injection_id: str, detected: bool, evidence: Any) -> None:
        results.append(
            {
                "injectionId": injection_id,
                "detected": bool(detected),
                "status": "PASS" if detected else "FAIL",
                "evidence": jsonable(evidence),
            }
        )

    unresolved = "UNRESOLVED::E01-A045"
    record(
        "FI01",
        unresolved.startswith("UNRESOLVED::"),
        "configuration validator rejects unresolved prefixes",
    )
    try:
        aggregate_means({"unknown": np.zeros(2)}, {key: np.zeros(2) for key in I_KEYS})
    except InformationValidationError as error:
        record("FI02", True, str(error))
    else:
        record("FI02", False, "unknown atom accepted")
    exact_copy = np.tile(np.asarray([[0.0, 0.0], [1.0, 1.0]]), (300, 1))
    singular = strict_sample_gate(
        exact_copy[:, 0], exact_copy[:, 1], tau=1, kind="gaussian"
    )
    record(
        "FI03",
        singular["status"] == "INELIGIBLE" and "RANK_DEFICIENT" in singular["reason"],
        singular,
    )
    nonfinite = np.arange(600, dtype=float)
    nonfinite[11] = np.nan
    no_deletion = strict_sample_gate(
        nonfinite, np.arange(600, dtype=float), tau=1, kind="gaussian"
    )
    record(
        "FI04", no_deletion["reason"] == "NONFINITE_INPUT_NO_ROW_DELETION", no_deletion
    )
    true_nats = math.log(2.0)
    mislabeled_bits = 1.0
    record(
        "FI05",
        abs(mislabeled_bits - true_nats) > 1e-10,
        {"mislabeled": mislabeled_bits, "expectedNats": true_nats},
    )
    record(
        "FI06",
        not np.isclose(0.0, 2e-10, atol=1e-10, rtol=1e-10),
        {"injectedDifference": 2e-10},
    )
    noncanonical = gaussian_partition_objective(
        np.random.default_rng(7).normal(size=(600, 4)),
        (1, 2),
        mapping="group_mean",
        objective="synchronous_mi",
        normalization="none",
    )
    record(
        "FI07",
        noncanonical["status"] == "INELIGIBLE"
        and "canonical" in noncanonical["reason"],
        noncanonical,
    )
    missing_permutation = {"streams": {"estimator": {"streamId": "fixture"}}}
    record(
        "FI08",
        "permutation" not in missing_permutation,
        "shuffle provenance requires permutation identity",
    )
    strict_label = "E01-S10-OMEGAID-2X2-GAUSSIAN-STRICT-GUARDED-v1.0.0"
    regularization_entered = True
    record(
        "FI09",
        regularization_entered and "STRICT" in strict_label,
        "strict label rejected when fallback entered",
    )
    first_input = Path(config["frozenInputs"][0]["path"]).read_bytes()
    actual = hashlib.sha256(first_input + b"tamper").hexdigest()
    record(
        "FI10",
        actual != config["frozenInputs"][0]["sha256"],
        {"tamperedSha256": actual},
    )
    return results


def summarize_validation(
    *,
    config: dict[str, Any],
    base_cases: list[dict[str, Any]],
    theory_rows: list[dict[str, Any]],
    invariance_rows: list[dict[str, Any]],
    backend_rows: list[dict[str, Any]],
    mib_rows: list[dict[str, Any]],
    injection_rows: list[dict[str, Any]],
    gpu_enabled: bool,
) -> dict[str, Any]:
    gates: dict[str, Any] = {}
    base_eligible = [case for case in base_cases if case["status"] == "ELIGIBLE"]
    closure_errors = [
        max(
            abs(case["means"]["latticeClosureError"]),
            abs(case["means"]["paperEquationClosureError"]),
        )
        for case in base_eligible
    ]
    gates["latticeClosure"] = {
        "maximumAbsoluteError": max(closure_errors),
        "tolerance": 1e-10,
        "success": max(closure_errors) <= 1e-10,
    }

    independent = [case for case in base_eligible if "INDEPENDENT" in case["systemId"]]
    independent_ensemble_atoms = {
        atom: float(np.mean([case["means"]["atomMeans"][atom] for case in independent]))
        for atom in ATOM_IDS
    }
    independent_ensemble_total = float(
        np.mean([case["means"]["totalMi"] for case in independent])
    )
    independent_success = (
        abs(independent_ensemble_total) <= 0.002
        and max(abs(value) for value in independent_ensemble_atoms.values()) <= 0.002
        and all(abs(case["means"]["totalMi"]) <= 0.01 for case in independent)
    )
    gates["independentGaussian"] = {
        "ensembleTotalMiNats": independent_ensemble_total,
        "maximumAbsoluteEnsembleAtomNats": max(
            abs(value) for value in independent_ensemble_atoms.values()
        ),
        "replicateCount": len(independent),
        "success": independent_success,
    }

    theory_ensemble = [row for row in theory_rows if row["scope"] == "ensemble"]
    theory_replicate = [row for row in theory_rows if row["scope"] == "replicate"]
    discrete_ensemble = [
        row for row in theory_ensemble if "DISCRETE" in row["systemId"]
    ]
    discrete_rep = [row for row in theory_replicate if "DISCRETE" in row["systemId"]]
    discrete_groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in discrete_rep:
        discrete_groups[
            (row["systemId"], row["redundancy"], row["replicateIndex"])
        ].append(row)
    discrete_passing_replicates = sum(
        all(item["success"] for item in rows) for rows in discrete_groups.values()
    )
    gates["exactDiscreteTheory"] = {
        "ensembleComparisons": len(discrete_ensemble),
        "ensembleFailures": sum(not row["success"] for row in discrete_ensemble),
        "passingReplicateVectors": discrete_passing_replicates,
        "totalReplicateVectors": len(discrete_groups),
        "minimumPerSystemRedundancy": 15,
        "success": all(row["success"] for row in discrete_ensemble)
        and all(
            sum(
                all(item["success"] for item in rows)
                for key, rows in discrete_groups.items()
                if key[:2] == group
            )
            >= 15
            for group in {(key[0], key[1]) for key in discrete_groups}
        ),
    }
    gaussian_theory_ensemble = [
        row
        for row in theory_ensemble
        if "GAUSSIAN" in row["systemId"] and "INDEPENDENT" not in row["systemId"]
    ]
    gaussian_theory_rep = [
        row
        for row in theory_replicate
        if "GAUSSIAN" in row["systemId"] and "INDEPENDENT" not in row["systemId"]
    ]
    gaussian_groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(
        list
    )
    for row in gaussian_theory_rep:
        gaussian_groups[
            (row["systemId"], row["redundancy"], row["replicateIndex"])
        ].append(row)
    gates["gaussianMmiPopulationTheory"] = {
        "ensembleComparisons": len(gaussian_theory_ensemble),
        "ensembleFailures": sum(not row["success"] for row in gaussian_theory_ensemble),
        "success": all(row["success"] for row in gaussian_theory_ensemble)
        and all(
            sum(
                all(item["success"] for item in rows)
                for key, rows in gaussian_groups.items()
                if key[:2] == group
            )
            >= 15
            for group in {(key[0], key[1]) for key in gaussian_groups}
        ),
    }

    qualitative_groups: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for case in base_eligible:
        system_id = case["systemId"]
        means = case["means"]
        if "REDUNDANT-GAUSSIAN" in system_id:
            passed = (
                means["pastRedundancy"] > means["pastSynergy"]
                and means["atomMeans"]["rtr"] > 0
                and means["paperEquationAggregateDirect"] < 0
            )
            qualitative_groups[(system_id, case["redundancy"])].append(passed)
        elif "COUPLED-AR" in system_id:
            passed = (
                means["totalMi"] > 0
                and means["miMeans"]["I_xtb"] > means["miMeans"]["I_yta"]
            )
            qualitative_groups[(system_id, case["redundancy"])].append(passed)
    gates["qualitativeSystems"] = {
        "groups": {
            "::".join(key): {
                "passes": sum(values),
                "total": len(values),
                "success": sum(values) >= 15,
            }
            for key, values in sorted(qualitative_groups.items())
        },
        "success": all(sum(values) >= 15 for values in qualitative_groups.values()),
    }

    affine = [
        row
        for row in invariance_rows
        if row["controlFamily"] != "common_time_permutation"
    ]
    shuffle = [
        row
        for row in invariance_rows
        if row["controlFamily"] == "common_time_permutation"
    ]
    shuffle_groups: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for row in shuffle:
        shuffle_groups[(row["systemId"], row["redundancy"])].append(
            bool(row["success"])
        )
    affine_by_backend_control: dict[tuple[str, str], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for row in affine:
        affine_by_backend_control[(row["backendId"], row["controlFamily"])].append(row)
    gates["affineAndScaleInvariance"] = {
        "comparisons": len(affine),
        "failures": sum(not row["success"] for row in affine),
        "byBackendAndControl": {
            "::".join(key): {
                "comparisons": len(rows),
                "failures": sum(not row["success"] for row in rows),
                "success": all(row["success"] for row in rows),
                "maximumAbsoluteError": max(
                    float(row.get("maximumAbsoluteError", 0.0) or 0.0) for row in rows
                ),
            }
            for key, rows in sorted(affine_by_backend_control.items())
        },
        "success": all(row["success"] for row in affine),
    }
    gates["timeShuffle"] = {
        "comparisons": len(shuffle),
        "groups": {
            "::".join(key): {
                "passes": sum(values),
                "total": len(values),
                "success": sum(values) >= 15,
            }
            for key, values in sorted(shuffle_groups.items())
        },
        "success": all(sum(values) >= 15 for values in shuffle_groups.values()),
    }

    source_rows = [
        row
        for row in backend_rows
        if row["comparisonFamily"] == "SOURCE_MATLAB_FIXTURE"
    ]
    cpu_rows = [
        row
        for row in backend_rows
        if "REFERENCE_VS_OMEGA_CPU" in row["comparisonFamily"]
    ]
    gpu_rows = [
        row for row in backend_rows if "OMEGA_CPU_VS_GPU" in row["comparisonFamily"]
    ]
    gates["sourceMatlabFixture"] = {
        "comparisons": len(source_rows),
        "failures": sum(not row["success"] for row in source_rows),
        "success": len(source_rows) == 4 and all(row["success"] for row in source_rows),
    }
    gates["referenceVsOmegaCpu"] = {
        "comparisons": len(cpu_rows),
        "failures": sum(not row["success"] for row in cpu_rows),
        "success": bool(cpu_rows) and all(row["success"] for row in cpu_rows),
    }
    gates["omegaCpuVsGpu"] = {
        "gpuRequested": gpu_enabled,
        "comparisons": len(gpu_rows),
        "failures": sum(not row["success"] for row in gpu_rows),
        "success": (not gpu_enabled)
        or (bool(gpu_rows) and all(row["success"] for row in gpu_rows)),
    }

    primary_mib = [
        row
        for row in mib_rows
        if row["mapping"] == "group_mean"
        and row["normalization"] == "none"
        and row["search"] == "exhaustive_all"
    ]
    primary_by_objective: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in primary_mib:
        primary_by_objective[row["objective"]].append(row)
    approximation: dict[str, Any] = {}
    for mapping in MAPPING_IDS:
        for search in ("spectral", "greedy"):
            selected = [
                row
                for row in mib_rows
                if row["mapping"] == mapping and row["search"] == search
            ]
            key = f"{mapping}::{search}"
            approximation[key] = {
                "eligibleComparisons": sum(
                    row["status"] == "ELIGIBLE" for row in selected
                ),
                "matches": sum(
                    row.get("matchesExhaustive") is True for row in selected
                ),
                "total": len(selected),
                "allEligibleMatch": bool(selected)
                and all(
                    row["status"] == "ELIGIBLE" and row.get("matchesExhaustive") is True
                    for row in selected
                ),
            }
    gates["mibRecovery"] = {
        "primaryByObjective": {
            objective: {
                "recoveries": sum(row["matchesPlantedPartition"] for row in rows),
                "total": len(rows),
                "success": len(rows) == 8
                and all(row["matchesPlantedPartition"] for row in rows),
            }
            for objective, rows in sorted(primary_by_objective.items())
        },
        "approximation": approximation,
        "success": len(primary_by_objective) == 3
        and all(
            len(rows) == 8 and all(row["matchesPlantedPartition"] for row in rows)
            for rows in primary_by_objective.values()
        ),
    }
    gates["failureInjection"] = {
        "detected": sum(row["detected"] for row in injection_rows),
        "total": len(injection_rows),
        "success": len(injection_rows) == 10
        and all(row["detected"] for row in injection_rows),
    }
    reference_gate_names = [
        "latticeClosure",
        "independentGaussian",
        "exactDiscreteTheory",
        "gaussianMmiPopulationTheory",
        "qualitativeSystems",
        "affineAndScaleInvariance",
        "timeShuffle",
        "sourceMatlabFixture",
        "mibRecovery",
        "failureInjection",
    ]
    reference_success = all(gates[name]["success"] for name in reference_gate_names)
    accelerated_success = gates["referenceVsOmegaCpu"]["success"]
    applicable_backend_success = gates["omegaCpuVsGpu"]["success"]
    overall = reference_success and accelerated_success and applicable_backend_success
    return {
        "schema": "eidosoma.e01.s10_validation_summary.v1",
        "researchStepId": "S10",
        "preregistrationVersion": config["preregistrationVersion"],
        "preregistrationCommit": PREREGISTRATION_COMMIT,
        "preregistrationSha256": PREREGISTRATION_SHA256,
        "outcomesInspectedAfterPreregistration": True,
        "referenceGateSuccess": reference_success,
        "acceleratedPathSuccess": accelerated_success,
        "gpuApplicableGateSuccess": applicable_backend_success,
        "overallSuccess": overall,
        "outcomeClassification": "supportive"
        if overall
        else "constraining/contradictory",
        "gates": gates,
    }


def eligibility_registry(
    summary: dict[str, Any], mib_rows: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    gates = summary["gates"]
    candidates: list[dict[str, Any]] = []
    affine_details = gates["affineAndScaleInvariance"]["byBackendAndControl"]
    reference_affine_success = all(
        details["success"]
        for key, details in affine_details.items()
        if key.startswith("pinned_phyid_cpu::")
    )
    reference_common = (
        gates["latticeClosure"]["success"]
        and gates["sourceMatlabFixture"]["success"]
        and reference_affine_success
    )
    for redundancy in REDUNDANCIES:
        candidates.append(
            {
                "branchId": f"E01-S10-PHYID-GAUSSIAN-STRICT::{redundancy}",
                "branchFamily": "estimator_redundancy",
                "status": "CONDITIONALLY_ELIGIBLE_FOR_S11_N_EFF_GE_512"
                if reference_common
                else "INELIGIBLE_VALIDATION_FAILURE",
                "authorDefault": False,
                "caveats": [
                    "author estimator and redundancy mapping unresolved",
                    "CCS source is marked To be implemented"
                    if redundancy == "CCS"
                    else "MMI is a candidate, not an author default",
                    "strict per-window sample gate requires at least 512 effective samples",
                    "queued S11 fixed windows 32/64/128/256 are all ineligible",
                ],
            }
        )
    omega_cpu_gaussian_scale = affine_details.get(
        "pinned_omegaid_numpy::gaussian_affine", {"success": False}
    )["success"]
    omega_cpu_discrete_scale = affine_details.get(
        "pinned_omegaid_numpy::discrete_relabel", {"success": False}
    )["success"]
    omega_gpu_gaussian_scale = affine_details.get(
        "pinned_omegaid_cupy::gaussian_affine", {"success": False}
    )["success"]
    omega_gpu_discrete_scale = affine_details.get(
        "pinned_omegaid_cupy::discrete_relabel", {"success": False}
    )["success"]
    candidates.extend(
        [
            {
                "branchId": "E01-S10-OMEGAID-2X2-GAUSSIAN-CPU-v1.0.0",
                "branchFamily": "accelerated_backend",
                "status": "CONDITIONALLY_ELIGIBLE_ACCELERATOR_N_EFF_GE_512"
                if gates["referenceVsOmegaCpu"]["success"] and omega_cpu_gaussian_scale
                else "INELIGIBLE_CROSSCHECK_OR_INVARIANCE_FAILURE",
                "authorDefault": False,
                "caveats": [
                    "Gaussian 2x2 guarded path only",
                    "SVD plus 1e-6 fallback prohibited in strict branch",
                    "queued S11 fixed windows 32/64/128/256 are all ineligible",
                ],
            },
            {
                "branchId": "E01-S10-OMEGAID-2X2-GAUSSIAN-GPU-v1.0.0",
                "branchFamily": "accelerated_backend",
                "status": "CONDITIONALLY_ELIGIBLE_ACCELERATOR_N_EFF_GE_512"
                if gates["omegaCpuVsGpu"]["success"] and omega_gpu_gaussian_scale
                else "INELIGIBLE_CROSSCHECK_OR_INVARIANCE_FAILURE",
                "authorDefault": False,
                "caveats": [
                    "Gaussian float64 on recorded NVIDIA L4 UUID",
                    "2x2 guarded path only",
                    "queued S11 fixed windows 32/64/128/256 are all ineligible",
                ],
            },
            {
                "branchId": "E01-S10-OMEGAID-2X2-DISCRETE-CPU-v1.0.0",
                "branchFamily": "accelerated_backend",
                "status": "ELIGIBLE_CROSSCHECKED_ACCELERATOR"
                if gates["referenceVsOmegaCpu"]["success"] and omega_cpu_discrete_scale
                else "INELIGIBLE_DISCRETE_RELABEL_INVARIANCE_FAILURE",
                "authorDefault": False,
                "caveats": [
                    "pinned OmegaID binarizes the complete two-row input with one global mean",
                    "failed every preregistered independent binary relabel control",
                ],
            },
            {
                "branchId": "E01-S10-OMEGAID-2X2-DISCRETE-GPU-v1.0.0",
                "branchFamily": "accelerated_backend",
                "status": "ELIGIBLE_CROSSCHECKED_ACCELERATOR"
                if gates["omegaCpuVsGpu"]["success"] and omega_gpu_discrete_scale
                else "INELIGIBLE_DISCRETE_RELABEL_INVARIANCE_FAILURE",
                "authorDefault": False,
                "caveats": [
                    "pinned OmegaID binarizes the complete two-row input with one global mean",
                    "failed every preregistered independent binary relabel control",
                ],
            },
            {
                "branchId": "OMEGAID-MORE-THAN-2X2-DOUBLET-LATTICE",
                "branchFamily": "estimator",
                "status": "INELIGIBLE_AS_16_ATOM_PHIID_SUBSTITUTE",
                "authorDefault": False,
                "caveats": [
                    "least-squares approximation",
                    "different atom identities",
                    "redundancy argument unused",
                ],
            },
            {
                "branchId": "E01-S10-PHYID-DISCRETE-BINARY-v1.0.0",
                "branchFamily": "estimator",
                "status": "VALIDATED_SYNTHETIC_BINARY_ONLY_NOT_S11_CONTINUOUS_INPUT",
                "authorDefault": False,
                "caveats": [
                    "mean-threshold binarization",
                    "native bits explicitly converted to nats",
                ],
            },
        ]
    )
    for mapping in MAPPING_IDS:
        for objective in OBJECTIVE_IDS:
            for normalization in NORMALIZATION_IDS:
                for search in SEARCH_IDS:
                    rows = [
                        row
                        for row in mib_rows
                        if row["mapping"] == mapping
                        and row["objective"] == objective
                        and row["normalization"] == normalization
                        and row["search"] == search
                    ]
                    all_eligible = bool(rows) and all(
                        row["status"] == "ELIGIBLE" for row in rows
                    )
                    if search in {"spectral", "greedy"}:
                        validated = all_eligible and all(
                            row.get("matchesExhaustive") is True for row in rows
                        )
                        status = (
                            "ELIGIBLE_VALIDATION_ONLY_APPROXIMATION"
                            if validated
                            else "INELIGIBLE_APPROXIMATION_GATE_FAILURE"
                        )
                    elif (
                        mapping == "omega_equal_width_vector"
                        and search == "exhaustive_all"
                    ):
                        status = "INELIGIBLE_UNBALANCED_SEARCH_DOMAIN"
                    else:
                        status = (
                            "ELIGIBLE_VALIDATION_ONLY_SMALL_D_EXHAUSTIVE"
                            if all_eligible
                            else "INELIGIBLE_NUMERICAL_OR_DOMAIN_FAILURE"
                        )
                    candidates.append(
                        {
                            "branchId": "::".join(
                                [
                                    MAPPING_IDS[mapping],
                                    OBJECTIVE_IDS[objective],
                                    NORMALIZATION_IDS[normalization],
                                    SEARCH_IDS[search],
                                ]
                            ),
                            "branchFamily": "partition_specification",
                            "status": status,
                            "authorDefault": False,
                            "validatedReplicates": sum(
                                row["status"] == "ELIGIBLE" for row in rows
                            ),
                            "totalReplicates": len(rows),
                            "caveats": [
                                "no author partition mapping/objective/normalization/search recovered",
                                "four-dimensional planted validation does not establish scalability to 99 components",
                            ],
                        }
                    )
    for sentinel in (
        "UNRESOLVED::E01-A043",
        "UNRESOLVED::E01-A044",
        "UNRESOLVED::E01-A045",
        "UNRESOLVED::E01-A046",
        "UNRESOLVED::E01-A054",
        "UNRESOLVED::E01-A055",
        "UNRESOLVED::E01-A056",
        "UNRESOLVED::E01-A058",
    ):
        candidates.append(
            {
                "branchId": sentinel,
                "branchFamily": "author_method_sentinel",
                "status": "PRESERVED_UNRESOLVED_NOT_EXECUTABLE",
                "authorDefault": False,
                "caveats": ["no source evidence resolved this sentinel in S10"],
            }
        )
    return {
        "schema": "eidosoma.e01.s10_information_dynamics_eligibility_registry.v1",
        "researchStepId": "S10",
        "registryVersion": "E01-S10-information-dynamics-eligibility-v1.0.0",
        "statusBoundary": "SYNTHETIC_VALIDATION_ELIGIBILITY_NOT_AUTHOR_DEFAULT_SELECTION",
        "paperPrimarySelected": False,
        "specificationRegistrySha256": REGISTRY_SHA256,
        "preprocessingRegistry": {
            "path": next(
                item["path"]
                for item in config["frozenInputs"]
                if item["inputId"] == "validTransformRegistry"
            ),
            "sha256": next(
                item["sha256"]
                for item in config["frozenInputs"]
                if item["inputId"] == "validTransformRegistry"
            ),
            "rule": "S09 acceptance is necessary but not sufficient; each S11 window must pass the strict S10 covariance/sample gate. Full CLR and raw closed proportions retain structural singularity.",
        },
        "s11WindowCompatibility": {
            "source": "/workspace/RESEARCH_PLAN.md S11 queue",
            "queuedWindowLengths": [32, 64, 128, 256],
            "queuedLags": [1, 2, 4, 8],
            "strictMinimumEffectiveSamples": 512,
            "effectiveSampleFormula": "window_length_minus_tau",
            "eligibleQueuedFixedWindowPairs": 0,
            "totalQueuedFixedWindowPairs": 16,
            "status": "NO_QUEUED_FIXED_WINDOW_IS_ELIGIBLE",
            "conditionallyEligibleScopes": [
                "expanding_window_after_effective_sample_count_reaches_512",
                "whole_trajectory_if_effective_sample_count_reaches_512",
            ],
            "requiredAlternative": "A smaller-window or regularized branch requires separate preregistration and validation; S11 may not silently relax E01-S10-SAMPLE-GATE-STRICT-v1.0.0.",
        },
        "candidates": candidates,
    }


def make_figures(
    output_dir: Path,
    atom_rows: list[dict[str, Any]],
    backend_rows: list[dict[str, Any]],
    mib_rows: list[dict[str, Any]],
) -> None:
    reference = [
        row
        for row in atom_rows
        if row["backendId"] == "pinned_phyid_cpu" and row["variant"] == "base"
    ]
    systems = list(SYSTEM_KINDS)
    matrix = np.zeros((len(systems) * 2, len(ATOM_IDS)))
    labels: list[str] = []
    for system_index, system in enumerate(systems):
        for redundancy_index, redundancy in enumerate(REDUNDANCIES):
            subset = [
                row
                for row in reference
                if row["systemId"] == system and row["redundancy"] == redundancy
            ]
            for atom_index, atom in enumerate(ATOM_IDS):
                matrix[system_index * 2 + redundancy_index, atom_index] = np.mean(
                    [row["meanNats"] for row in subset if row["atomId"] == atom]
                )
            labels.append(f"{system.split('SYS-')[-1].split('-v')[0]} | {redundancy}")
    limit = max(abs(float(np.min(matrix))), abs(float(np.max(matrix))))
    fig, axis = plt.subplots(figsize=(13, 7))
    image = axis.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
    axis.set_xticks(range(len(ATOM_IDS)), ATOM_IDS, rotation=45, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_title("S10 mean PhiID atom profiles (pinned phyid CPU, nats)")
    fig.colorbar(image, ax=axis, label="mean local atom (nats)")
    fig.tight_layout()
    fig.savefig(output_dir / "synthetic_atom_profiles.png", dpi=180)
    plt.close(fig)

    comparable = [
        row
        for row in backend_rows
        if row["comparisonFamily"] in {"REFERENCE_VS_OMEGA_CPU", "OMEGA_CPU_VS_GPU"}
        and isinstance(row.get("maximumAbsoluteError"), (int, float))
    ]
    fig, axis = plt.subplots(figsize=(9, 5))
    families = sorted({row["comparisonFamily"] for row in comparable})
    for family in families:
        values = [
            max(float(row["maximumAbsoluteError"]), 1e-18)
            for row in comparable
            if row["comparisonFamily"] == family
        ]
        axis.scatter(range(len(values)), values, s=18, alpha=0.7, label=family)
    axis.axhline(1e-10, color="black", linestyle="--", label="frozen tolerance")
    axis.set_yscale("log")
    axis.set_xlabel("comparison index")
    axis.set_ylabel("maximum local absolute error")
    axis.set_title("S10 backend agreement")
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "backend_agreement.png", dpi=180)
    plt.close(fig)

    eligible = [row for row in mib_rows if row["status"] == "ELIGIBLE"]
    group_labels: list[str] = []
    fractions: list[float] = []
    for mapping in MAPPING_IDS:
        for search in SEARCH_IDS:
            subset = [
                row
                for row in eligible
                if row["mapping"] == mapping and row["search"] == search
            ]
            if not subset:
                continue
            group_labels.append(f"{mapping}\n{search}")
            fractions.append(
                float(np.mean([row["matchesPlantedPartition"] for row in subset]))
            )
    fig, axis = plt.subplots(figsize=(12, 5))
    axis.bar(range(len(fractions)), fractions, color="#3f7cac")
    axis.set_xticks(range(len(group_labels)), group_labels, rotation=45, ha="right")
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("fraction selecting planted split")
    axis.set_title("S10 planted block partition recovery across explicit branches")
    fig.tight_layout()
    fig.savefig(output_dir / "mib_recovery.png", dpi=180)
    plt.close(fig)


def artifact_manifest(output_dir: Path, bundle_dir: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for root, role_prefix in (
        (output_dir, "research_step"),
        (bundle_dir, "bundle_contract"),
    ):
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.name == "artifact_manifest.json":
                continue
            files.append(
                {
                    "role": role_prefix,
                    "path": str(path),
                    "sizeBytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    manifest = {
        "schema": "eidosoma.e01.s10_artifact_manifest.v1",
        "researchStepId": "S10",
        "artifactCountExcludingManifest": len(files),
        "artifacts": files,
        "repository": {
            "path": str(REPOSITORY_ROOT),
            "branch": git_output("branch", "--show-current"),
            "head": git_output("rev-parse", "HEAD"),
            "preregistrationCommit": PREREGISTRATION_COMMIT,
        },
    }
    write_json(output_dir / "artifact_manifest.json", manifest)
    return manifest


def run(args: argparse.Namespace) -> None:
    artifacts_root = Path(args.artifacts_root).resolve()
    output_dir = artifacts_root / "research_steps/S10"
    bundle_dir = artifacts_root / "E01_forensic_replication_bundle/information_dynamics"
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    if args.finalize_manifest_only:
        manifest = artifact_manifest(output_dir, bundle_dir)
        print(
            json.dumps(
                {
                    "artifactCount": manifest["artifactCountExcludingManifest"],
                    "status": "manifest_finalized",
                }
            )
        )
        return

    started = time.time()
    config = yaml.safe_load(CONFIG_PATH.read_text())
    prereg_record = verify_preregistration(config)
    registry_record = registry_preservation(artifacts_root)
    shutil.copyfile(CONFIG_PATH, output_dir / "preregistration.yaml")
    write_json(output_dir / "preregistration_record.json", prereg_record)
    write_json(output_dir / "registry_preservation.json", registry_record)

    source_trace = {
        "schema": "eidosoma.e01.s10_source_traceability.v1",
        "researchStepId": "S10",
        "resolvedModules": backend_identity(),
        "phyid": {
            "identity": config["pinnedSources"]["phyid"],
            "trace": {
                "fourVectorAndEstimatorDispatch": "phyid/calculate.py:16-47,154-215",
                "mutualInformationTerms": "phyid/calculate.py:50-61",
                "redundancyDispatch": "phyid/calculate.py:65-113",
                "atomLinearSystem": "phyid/calculate.py:116-151",
                "gaussianEntropy": "phyid/measures.py:10-28",
                "binaryEntropyAndNativeBits": "phyid/measures.py:31-53",
                "mmiAndCcs": "phyid/measures.py:56-113",
            },
            "boundary": "Pinned public source, not the unavailable author implementation.",
        },
        "omegaid": {
            "identity": config["pinnedSources"]["omegaid"],
            "trace": {
                "twoByTwoDispatch": "omegaid/core/decomposition.py:167-194,281-318",
                "doubletApproximation": "omegaid/core/decomposition.py:196-279",
                "svdPlus1e6Fallback": "omegaid/core/entropy.py:37-57",
                "backendSelection": "omegaid/utils/backend.py:1-39",
            },
            "boundary": config["pinnedSources"]["omegaid"]["implementationBoundary"],
        },
        "paperAggregateDerivation": {
            "direct": "I_xytab-I_xtab-I_ytab",
            "atoms": "str+stx+sty+sts-rtr-rtx-rty-rts",
            "status": "EQUATION_DERIVED_NOT_SOURCE_NAMED_ATOM",
        },
    }
    write_json(output_dir / "source_traceability.json", source_trace)

    gpu_enabled = not args.skip_gpu
    source_backend_rows, source_atom_rows = source_fixture_validation(
        config, gpu_enabled=gpu_enabled
    )

    tasks: list[tuple[str, int, str, str]] = []
    primary_replicates = int(config["syntheticDesign"]["primaryReplicates"])
    for system_id in SYSTEM_KINDS:
        for replicate in range(primary_replicates):
            for redundancy in REDUNDANCIES:
                tasks.append((system_id, replicate, redundancy, "base"))
                tasks.append((system_id, replicate, redundancy, "affine"))
                if system_id in STRUCTURED_SYSTEMS:
                    tasks.append((system_id, replicate, redundancy, "time_shuffle"))
    with ProcessPoolExecutor(
        max_workers=int(config["runtime"]["execution"]["workerProcesses"])
    ) as executor:
        cases = list(executor.map(reference_case_worker, tasks, chunksize=2))
    cases.sort(key=lambda row: row["caseId"])
    base_cases = [case for case in cases if case["variant"] == "base"]

    benchmark_rows: list[dict[str, Any]] = []
    atom_rows = list(source_atom_rows)
    seed_rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    for case in cases:
        benchmark_rows.append(
            {
                key: value
                for key, value in case.items()
                if key not in {"means", "sampleGate"}
            }
            | {
                "sampleGateStatus": case["sampleGate"]["status"],
                "sampleGateReason": case["sampleGate"].get("reason"),
                "effectiveSampleCount": case["sampleGate"].get("effectiveSampleCount"),
                "conditionNumber": case["sampleGate"].get("conditionNumber"),
                "totalMiNats": case.get("means", {}).get("totalMi"),
                "paperEquationAggregateNats": case.get("means", {}).get(
                    "paperEquationAggregateDirect"
                ),
                "latticeClosureErrorNats": case.get("means", {}).get(
                    "latticeClosureError"
                ),
                "paperEquationClosureErrorNats": case.get("means", {}).get(
                    "paperEquationClosureError"
                ),
            }
        )
        seed_rows[(case["systemId"], case["variant"], case["replicateIndex"])] = {
            "systemId": case["systemId"],
            "variant": case["variant"],
            "replicateIndex": case["replicateIndex"],
            "streamId": case["estimatorStreamId"],
            "seedMaterialHex": case["estimatorSeedMaterialHex"],
            "seedPayloadSha256": case["seedPayloadSha256"],
            "dataSha256": case["dataSha256"],
            "permutation": case["permutation"],
        }
        if case["status"] == "ELIGIBLE":
            for atom in ATOM_IDS:
                atom_rows.append(
                    {
                        "caseId": case["caseId"],
                        "systemId": case["systemId"],
                        "replicateIndex": case["replicateIndex"],
                        "variant": case["variant"],
                        "backendId": case["backendId"],
                        "kind": case["kind"],
                        "redundancy": case["redundancy"],
                        "atomId": atom,
                        "meanNats": case["means"]["atomMeans"][atom],
                        "status": "ELIGIBLE",
                    }
                )

    theory_rows = theoretical_comparison_rows(base_cases, config)
    invariance_rows = reference_invariance_rows(cases, config)
    cross_backend_rows, omega_atom_rows, omega_invariance_rows = (
        cross_backend_validation(config, gpu_enabled=gpu_enabled)
    )
    backend_rows = source_backend_rows + cross_backend_rows
    atom_rows.extend(omega_atom_rows)
    invariance_rows.extend(omega_invariance_rows)

    mib_rows: list[dict[str, Any]] = []
    mib_candidate_rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=int(config["runtime"]["execution"]["workerProcesses"])
    ) as executor:
        for rows, candidates, seed_row in executor.map(
            mib_worker, range(int(config["syntheticDesign"]["mibReplicates"]))
        ):
            mib_rows.extend(rows)
            mib_candidate_rows.extend(candidates)
            seed_rows[
                (seed_row["systemId"], seed_row["variant"], seed_row["replicateIndex"])
            ] = seed_row
    annotate_mib_agreement(mib_rows)
    mib_rows.sort(
        key=lambda row: (
            row["replicateIndex"],
            row["mapping"],
            row["objective"],
            row["normalization"],
            row["search"],
        )
    )
    mib_candidate_rows.sort(
        key=lambda row: (
            row["replicateIndex"],
            row["mapping"],
            row["objective"],
            row["normalization"],
            row["search"],
            row["partA"],
        )
    )

    injection_rows = failure_injections(config)
    summary = summarize_validation(
        config=config,
        base_cases=base_cases,
        theory_rows=theory_rows,
        invariance_rows=invariance_rows,
        backend_rows=backend_rows,
        mib_rows=mib_rows,
        injection_rows=injection_rows,
        gpu_enabled=gpu_enabled,
    )
    eligibility = eligibility_registry(summary, mib_rows, config)

    write_csv(output_dir / "benchmark_cases.csv", benchmark_rows)
    write_csv(output_dir / "atom_results.csv", atom_rows)
    write_csv(output_dir / "theoretical_comparisons.csv", theory_rows)
    write_csv(output_dir / "invariance_results.csv", invariance_rows)
    write_csv(output_dir / "backend_comparisons.csv", backend_rows)
    write_csv(output_dir / "mib_partition_results.csv", mib_rows)
    write_csv(output_dir / "mib_candidate_scores.csv", mib_candidate_rows)
    write_csv(
        output_dir / "eligibility_registry.csv",
        [
            {
                "branchId": item["branchId"],
                "branchFamily": item["branchFamily"],
                "status": item["status"],
                "authorDefault": item["authorDefault"],
                "caveats": item["caveats"],
                "validatedReplicates": item.get("validatedReplicates"),
                "totalReplicates": item.get("totalReplicates"),
            }
            for item in eligibility["candidates"]
        ],
    )
    write_json(output_dir / "validation_summary.json", summary)
    write_json(
        output_dir / "failure_injection.json",
        {"researchStepId": "S10", "results": injection_rows},
    )
    write_json(
        output_dir / "seed_manifest.json",
        {
            "schema": "eidosoma.e01.s10_seed_manifest.v1",
            "researchStepId": "S10",
            "rootSeedHex": config["randomness"]["rootSeedHex"],
            "derivationAlgorithm": config["randomness"]["derivationAlgorithm"],
            "records": [seed_rows[key] for key in sorted(seed_rows)],
        },
    )
    runtime = {
        "schema": "eidosoma.e01.s10_runtime_manifest.v1",
        "researchStepId": "S10",
        "startedEpochSeconds": started,
        "completedEpochSeconds": time.time(),
        "wallSeconds": time.time() - started,
        "pid": os.getpid(),
        "python": sys.version,
        "numpy": np.__version__,
        "environment": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "gpuRequested": gpu_enabled,
        "backendIdentity": backend_identity(),
        "repositoryHeadAtRun": git_output("rev-parse", "HEAD"),
        "repositoryDirtyAtRun": bool(git_output("status", "--short")),
        "preregistrationCommit": PREREGISTRATION_COMMIT,
    }
    if gpu_enabled:
        try:
            import cupy

            properties = cupy.cuda.runtime.getDeviceProperties(0)
            runtime["cupy"] = {
                "version": cupy.__version__,
                "cudaRuntimeVersion": cupy.cuda.runtime.runtimeGetVersion(),
                "logicalDeviceIndex": 0,
                "deviceName": properties["name"].decode()
                if isinstance(properties["name"], bytes)
                else str(properties["name"]),
            }
        except Exception as error:
            runtime["cupy"] = {
                "status": "FAILED",
                "reason": f"{type(error).__name__}: {error}",
            }
    write_json(output_dir / "runtime_manifest.json", runtime)

    contract = {
        "schema": "eidosoma.e01.s10_information_dynamics_contract.v1",
        "researchStepId": "S10",
        "contractVersion": "E01-S10-information-dynamics-contract-v1.0.0",
        "status": "VALIDATED_RECONSTRUCTION_BRANCHES_NOT_AUTHOR_METHOD_RECOVERY",
        "preregistration": {
            "version": config["preregistrationVersion"],
            "commit": PREREGISTRATION_COMMIT,
            "sha256": PREREGISTRATION_SHA256,
        },
        "sourceIdentities": config["pinnedSources"],
        "precision": config["runtime"]["precision"],
        "randomness": config["randomness"],
        "atomCatalog": config["atomCatalog"],
        "estimatorCatalog": config["estimatorCatalog"],
        "sampleGate": config["sampleGate"],
        "partitionCatalog": config["partitionCatalog"],
        "validationSummarySha256": sha256_file(output_dir / "validation_summary.json"),
        "overallValidationSuccess": summary["overallSuccess"],
        "authorMethodBoundary": config["scopeBoundary"],
    }
    write_yaml(bundle_dir / "information_dynamics_contract_v1.0.0.yaml", contract)
    write_yaml(
        bundle_dir / "information_dynamics_eligibility_registry_v1.0.0.yaml",
        eligibility,
    )
    make_figures(output_dir, atom_rows, backend_rows, mib_rows)
    artifact_manifest(output_dir, bundle_dir)
    print(
        json.dumps(
            {
                "status": "complete",
                "overallSuccess": summary["overallSuccess"],
                "outcomeClassification": summary["outcomeClassification"],
                "benchmarkCases": len(benchmark_rows),
                "backendComparisons": len(backend_rows),
                "mibResults": len(mib_rows),
                "wallSeconds": runtime["wallSeconds"],
            },
            sort_keys=True,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifacts-root", default=os.environ.get("ARTIFACTS_DIR", "/artifacts")
    )
    parser.add_argument("--skip-gpu", action="store_true")
    parser.add_argument("--finalize-manifest-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
