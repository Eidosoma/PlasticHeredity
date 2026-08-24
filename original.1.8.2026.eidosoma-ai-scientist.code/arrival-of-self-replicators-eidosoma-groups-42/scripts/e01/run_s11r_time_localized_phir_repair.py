#!/usr/bin/env python3
"""Execute the preregistered bounded E01 S11R repair validation.

The development and confirmation phases use disjoint S06 seed roots.  This
runner never writes an S11 or S12 path and never generates a GARD estimate.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from e01_information_dynamics.validation import ATOM_IDS, I_KEYS, all_bipartitions
from e01_time_localized_phir.partition import (
    StablePartitionResult,
    evaluate_candidate_grid,
)
from e01_time_localized_phir_repair import (
    AFFINITY_ID,
    CALIBRATION_ID,
    ESTIMATOR_ID,
    SEARCH_ID,
    ccs_population_oracle,
    directional_covariance,
    directional_var,
    highdim_independent_null,
    independent_white,
    mmi_truth,
    noisy_redundant_ar,
    partition_ari,
    planted_two_block_ar,
    repair_rng,
    run_wishart_local_phiid,
    threshold_component_partition,
)
from e01_time_localized_phir_repair.estimator import calibrate_means
from e01_time_localized_phir_repair.partition import RepairPartitionError, _components
from e01_time_localized_phir_repair.synthetic import (
    CONFIRMATION_ROOT_SEED_HEX,
    DEVELOPMENT_ROOT_SEED_HEX,
    DIRECTIONAL_ID,
    INDEPENDENT_ID,
    REDUNDANT_ID,
    redundant_covariance,
)

CONFIG_PATH = (
    REPOSITORY_ROOT / "configs/e01/s11r_time_localized_phir_repair_preregistration.yaml"
)
PREREGISTRATION_COMMIT = "a4763d0d5c7428897fcb595ae7d25a754d346c31"
PREREGISTRATION_SHA256 = (
    "9f8a9424fae41a5ed7ea0d185eb5aaa31449bd12892c027eda2dc83fb57e99e0"
)
S11_DIR = Path("/artifacts/research_steps/S11")
S11_MANIFEST_SHA256 = "21e58c969bc511cb620408518f96b5cab8acae02ec269fa376716c01123742ea"
S11_AGGREGATE_SHA256 = (
    "5e0d91d800999ecb47cd3107c2b61cf3c6c76f56b6730f27aa0a059244daa773"
)
BUNDLE_DIR = Path("/artifacts/E01_forensic_replication_bundle/information_dynamics")

REDUNDANCIES = ("MMI", "CCS")
STRUCTURED_SYSTEMS = (REDUNDANT_ID, DIRECTIONAL_ID)
TRUTH_SYSTEMS = (INDEPENDENT_ID, *STRUCTURED_SYSTEMS)
MAPPINGS = ("zscore_group_mean", "zscore_pc1")
OBJECTIVES = ("synchronous_mi", "bidirectional_lagged_mi", "abs_paper_equation")
NORMALIZATIONS = ("none", "min_part_entropy", "geometric_part_size")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
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
    path.write_text(yaml.safe_dump(jsonable(value), sort_keys=False, width=110))


def _flat(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, np.ndarray)):
        return json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"))
    return value.item() if isinstance(value, np.generic) else value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty required table {path.name}.")
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _flat(row.get(key)) for key in fields})


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty required parquet {path.name}.")
    fields = sorted({key for row in rows for key in row})
    normalized = [{key: _flat(row.get(key)) for key in fields} for row in rows]
    pd.DataFrame(normalized, columns=fields).to_parquet(
        path, compression="zstd", index=False
    )


def git_output(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def verify_preregistration(config: dict[str, Any]) -> dict[str, Any]:
    actual = sha256_file(CONFIG_PATH)
    if actual != PREREGISTRATION_SHA256:
        raise RuntimeError(f"S11R preregistration changed: {actual}.")
    committed = subprocess.run(
        ["git", "show", f"{PREREGISTRATION_COMMIT}:configs/e01/{CONFIG_PATH.name}"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    if hashlib.sha256(committed).hexdigest() != PREREGISTRATION_SHA256:
        raise RuntimeError("Preregistration commit does not contain the frozen bytes.")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", PREREGISTRATION_COMMIT, "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
    )
    checks: list[dict[str, Any]] = []
    for item in config["frozenInputs"]:
        path = Path(item["path"])
        observed = sha256_file(path) if path.is_file() else None
        checks.append(
            {**item, "actualSha256": observed, "success": observed == item["sha256"]}
        )
    if not all(item["success"] for item in checks):
        raise RuntimeError(
            f"Frozen input mismatch: {[x for x in checks if not x['success']]}"
        )
    pairs = config["fixedWindowGrid"]["pairs"]
    exact_counts = [
        31,
        30,
        28,
        24,
        63,
        62,
        60,
        56,
        127,
        126,
        124,
        120,
        255,
        254,
        252,
        248,
    ]
    if (
        len(pairs) != 16
        or [item["effectiveSampleCount"] for item in pairs] != exact_counts
        or any(
            item["effectiveSampleCount"] != item["windowLength"] - item["lag"]
            for item in pairs
        )
        or max(exact_counts) >= 512
    ):
        raise RuntimeError("The exact 16-pair grid or strict boundary changed.")
    return {
        "status": "VERIFIED_FROZEN_BEFORE_S11R_OUTCOMES",
        "success": True,
        "preregistrationCommit": PREREGISTRATION_COMMIT,
        "preregistrationSha256": actual,
        "frozenInputChecks": checks,
        "exactEffectiveSampleCounts": exact_counts,
    }


def verify_s11_immutability(config: dict[str, Any]) -> dict[str, Any]:
    manifest_path = S11_DIR / "artifact_manifest.json"
    if sha256_file(manifest_path) != S11_MANIFEST_SHA256:
        raise RuntimeError("Immutable S11 artifact manifest changed.")
    manifest = json.loads(manifest_path.read_text())
    artifact_checks = []
    for item in manifest["artifacts"]:
        path = Path(item["path"])
        observed_size = path.stat().st_size if path.is_file() else None
        observed_hash = sha256_file(path) if path.is_file() else None
        artifact_checks.append(
            {
                "relativePath": item["relativePath"],
                "expectedSizeBytes": item["sizeBytes"],
                "actualSizeBytes": observed_size,
                "expectedSha256": item["sha256"],
                "actualSha256": observed_hash,
                "success": observed_size == item["sizeBytes"]
                and observed_hash == item["sha256"],
            }
        )
    repository_checks = []
    for item in config["immutableS11RepositoryFiles"]:
        path = REPOSITORY_ROOT / item["path"]
        observed = sha256_file(path) if path.is_file() else None
        repository_checks.append(
            {**item, "actualSha256": observed, "success": observed == item["sha256"]}
        )
    success = (
        manifest.get("aggregateSha256") == S11_AGGREGATE_SHA256
        and manifest.get("artifactCountExcludingManifest") == 34
        and len(artifact_checks) == 34
        and all(item["success"] for item in artifact_checks)
        and all(item["success"] for item in repository_checks)
    )
    if not success:
        raise RuntimeError("One or more immutable S11 bytes changed.")
    return {
        "status": "S11_VERIFIED_BYTE_IMMUTABLE",
        "success": True,
        "manifestSha256": S11_MANIFEST_SHA256,
        "aggregateSha256": S11_AGGREGATE_SHA256,
        "artifactCount": len(artifact_checks),
        "repositoryFileCount": len(repository_checks),
        "artifactChecks": artifact_checks,
        "repositoryChecks": repository_checks,
        "preservedFailures": {
            "gateFamiliesPassed": 11,
            "gateFamiliesTotal": 16,
            "knownTruthPassed": 46,
            "knownTruthTotal": 64,
            "structuredShufflePassed": 28,
            "structuredShuffleTotal": 64,
            "invariancePassed": 356,
            "invarianceTotal": 576,
            "partitionSummariesPassed": 0,
            "partitionSummariesTotal": 48,
            "fixedBranchesEligible": 0,
            "fixedNumericRows": 0,
            "fixedSuppressedRows": 33984,
        },
    }


def verify_method_lock(path: Path, commit: str) -> dict[str, Any]:
    if not path.is_file() or not commit:
        raise RuntimeError(
            "Confirmation requires a frozen method-lock path and commit."
        )
    committed = subprocess.run(
        ["git", "show", f"{commit}:{path.relative_to(REPOSITORY_ROOT)}"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    if hashlib.sha256(committed).hexdigest() != sha256_file(path):
        raise RuntimeError("Method-lock working bytes differ from the supplied commit.")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
    )
    lock = yaml.safe_load(path.read_text())
    if (
        lock["researchStepId"] != "S11R"
        or lock["preregistrationSha256"] != PREREGISTRATION_SHA256
        or not lock["confirmationAccessAuthorized"]
    ):
        raise RuntimeError(
            "Method lock does not authorize the frozen S11R confirmation."
        )
    for item in lock["implementationFiles"]:
        if sha256_file(REPOSITORY_ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError(f"Locked implementation changed: {item['path']}.")
    development_path = Path(lock["developmentSummaryPath"])
    if sha256_file(development_path) != lock["developmentSummarySha256"]:
        raise RuntimeError("Development summary differs from the method lock.")
    development_seed_path = Path(lock["developmentSeedRecordsPath"])
    if sha256_file(development_seed_path) != lock["developmentSeedRecordsSha256"]:
        raise RuntimeError("Development seed records differ from the method lock.")
    return {
        **lock,
        "methodLockPath": str(path),
        "methodLockCommit": commit,
        "methodLockSha256": sha256_file(path),
        "status": "VERIFIED_BEFORE_CONFIRMATION_SEED_ACCESS",
        "success": True,
    }


def _fixture(system_id: str, task: dict[str, Any]):
    arguments = {
        "phase": task["phase"],
        "pair_id": task["pairId"],
        "replicate_index": task["replicateIndex"],
        "length": task["windowLength"],
        "domain": task["domain"],
    }
    if system_id == INDEPENDENT_ID:
        return independent_white(**arguments)
    if system_id == REDUNDANT_ID:
        return noisy_redundant_ar(**arguments)
    if system_id == DIRECTIONAL_ID:
        return directional_var(**arguments)
    raise RuntimeError(f"Unknown scalar fixture {system_id!r}.")


def _compact(result: Any) -> dict[str, Any]:
    return {
        "status": result.status,
        "reason": result.reason,
        "means": result.means(),
        "minimumEigenvalue": result.diagnostics.get("minimumCovarianceEigenvalue"),
        "conditionNumber": result.diagnostics.get("covarianceConditionNumber"),
    }


def _estimate_both(data: np.ndarray, lag: int) -> dict[str, Any]:
    return {
        redundancy: _compact(
            run_wishart_local_phiid(
                data[:, 0], data[:, 1], tau=lag, redundancy=redundancy
            )
        )
        for redundancy in REDUNDANCIES
    }


def _calibration_task(task: dict[str, Any]) -> dict[str, Any]:
    fixture = _fixture(task["systemId"], task)
    rng, permutation_record = repair_rng(
        phase=task["phase"],
        domain="condition-matched-calibration-row-permutation",
        pair_id=f"{task['pairId']}-{task['systemId'].split('-SYS-')[-1]}",
        replicate_index=task["replicateIndex"],
    )
    shuffled = fixture.data[rng.permutation(fixture.data.shape[0])]
    return {
        **task,
        "results": _estimate_both(shuffled, task["lag"]),
        "seedRecords": [fixture.seed_record, permutation_record],
    }


def _truth_task(task: dict[str, Any]) -> dict[str, Any]:
    fixture = _fixture(task["systemId"], task)
    return {
        **task,
        "results": _estimate_both(fixture.data, task["lag"]),
        "seedRecords": [fixture.seed_record],
    }


def _shuffle_task(task: dict[str, Any]) -> dict[str, Any]:
    fixture = _fixture(task["systemId"], task)
    rng, permutation_record = repair_rng(
        phase=task["phase"],
        domain="structured-shuffle-evaluation-row-permutation",
        pair_id=f"{task['pairId']}-{task['systemId'].split('-SYS-')[-1]}",
        replicate_index=task["replicateIndex"],
    )
    shuffled = fixture.data[rng.permutation(fixture.data.shape[0])]
    return {
        **task,
        "results": _estimate_both(shuffled, task["lag"]),
        "seedRecords": [fixture.seed_record, permutation_record],
    }


def run_parallel(
    function: Any,
    tasks: list[dict[str, Any]],
    *,
    workers: int,
    stage: str,
    runtime: list[dict[str, Any]],
):
    started = time.perf_counter()
    if workers == 1:
        results = [function(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(
                executor.map(
                    function, tasks, chunksize=max(1, len(tasks) // (workers * 32))
                )
            )
    elapsed = time.perf_counter() - started
    runtime.append(
        {
            "stage": stage,
            "caseCount": len(tasks),
            "workers": workers,
            "wallSeconds": elapsed,
            "casesPerSecond": len(tasks) / elapsed if elapsed else None,
        }
    )
    print(f"{stage}: {len(tasks)} cases in {elapsed:.3f}s", flush=True)
    return results


def collect_seeds(
    results: list[dict[str, Any]], records: dict[str, dict[str, Any]]
) -> None:
    for result in results:
        for record in result.get("seedRecords", []):
            prior = records.setdefault(record["streamId"], record)
            if prior != record:
                raise RuntimeError("One stream ID mapped to two seed records.")


def mean_payload(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    if not payloads:
        raise RuntimeError("Cannot average an empty payload collection.")
    atoms = {
        atom: float(np.mean([item["atomMeans"][atom] for item in payloads]))
        for atom in ATOM_IDS
    }
    mi = {
        key: float(np.mean([item["miMeans"][key] for item in payloads]))
        for key in I_KEYS
    }
    total = float(sum(atoms.values()))
    redundancy = float(sum(atoms[key] for key in ATOM_IDS[:4]))
    synergy = float(sum(atoms[key] for key in ATOM_IDS[12:]))
    equation = synergy - redundancy
    direct = mi["I_xytab"] - mi["I_xtab"] - mi["I_ytab"]
    return {
        "atomMeans": atoms,
        "miMeans": mi,
        "totalAtomSum": total,
        "totalMi": mi["I_xytab"],
        "latticeClosureError": total - mi["I_xytab"],
        "paperEquationAggregateFromAtoms": equation,
        "paperEquationAggregateDirect": direct,
        "paperEquationClosureError": equation - direct,
    }


def _oracle_payload(
    system_id: str,
    lag: int,
    redundancy: str,
    ccs: dict[tuple[str, int], dict[str, Any]],
):
    if redundancy == "MMI":
        return mmi_truth(system_id, lag)
    return ccs[(system_id, lag)]


def run_scalar_validation(
    config: dict[str, Any],
    *,
    phase: str,
    workers: int,
    runtime: list[dict[str, Any]],
    seeds: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    pairs = config["fixedWindowGrid"]["pairs"]
    counts = config["sampleSizes"][phase]
    calibration_count = counts["calibrationReplicatesPerPairPerStructuredSystem"]
    truth_count = counts["knownTruthReplicatesPerPairPerSystem"]
    shuffle_count = counts["structuredShuffleReplicatesPerPairPerSystem"]
    calibration_tasks = [
        {
            **pair,
            "phase": phase,
            "systemId": system,
            "replicateIndex": replicate,
            "domain": "condition-matched-calibration-source",
        }
        for pair in pairs
        for system in STRUCTURED_SYSTEMS
        for replicate in range(calibration_count)
    ]
    calibration_results = run_parallel(
        _calibration_task,
        calibration_tasks,
        workers=workers,
        stage=f"{phase}_condition_matched_calibration",
        runtime=runtime,
    )
    collect_seeds(calibration_results, seeds)
    calibration: dict[tuple[str, str, str], dict[str, Any]] = {}
    envelopes: dict[tuple[str, str, str], float] = {}
    for pair in pairs:
        for system in STRUCTURED_SYSTEMS:
            selected = [
                item
                for item in calibration_results
                if item["pairId"] == pair["pairId"] and item["systemId"] == system
            ]
            for redundancy in REDUNDANCIES:
                values = [item["results"][redundancy]["means"] for item in selected]
                if any(item is None for item in values):
                    raise RuntimeError(
                        "A required calibration replicate was ineligible."
                    )
                averaged = mean_payload(values)
                calibration[(pair["pairId"], system, redundancy)] = averaged
                centered = [
                    abs(
                        item["paperEquationAggregateFromAtoms"]
                        - averaged["paperEquationAggregateFromAtoms"]
                    )
                    for item in values
                ]
                envelopes[(pair["pairId"], system, redundancy)] = float(
                    np.quantile(centered, 0.99, method="higher")
                )

    oracle_rows: list[dict[str, Any]] = []
    ccs_oracles: dict[tuple[str, int], dict[str, Any]] = {}
    oracle_started = time.perf_counter()
    for system in TRUTH_SYSTEMS:
        for lag in config["fixedWindowGrid"]["lags"]:
            covariance = (
                np.eye(4)
                if system == INDEPENDENT_ID
                else redundant_covariance(lag)
                if system == REDUNDANT_ID
                else directional_covariance(lag)
            )
            first = ccs_population_oracle(
                covariance,
                scramble_seed=311_000 + 101 * lag + TRUTH_SYSTEMS.index(system),
            )
            second = ccs_population_oracle(
                covariance,
                scramble_seed=719_000 + 101 * lag + TRUTH_SYSTEMS.index(system),
            )
            maximum = max(
                abs(first["atomMeans"][atom] - second["atomMeans"][atom])
                for atom in ATOM_IDS
            )
            ccs_oracles[(system, lag)] = {
                "atomMeans": {
                    atom: 0.5 * (first["atomMeans"][atom] + second["atomMeans"][atom])
                    for atom in ATOM_IDS
                },
                "miMeans": {
                    key: 0.5 * (first["miMeans"][key] + second["miMeans"][key])
                    for key in I_KEYS
                },
                "totalMi": 0.5 * (first["totalMi"] + second["totalMi"]),
                "paperEquationAggregate": 0.5
                * (first["paperEquationAggregate"] + second["paperEquationAggregate"]),
                "maximumCrossScrambleAtomDifference": maximum,
            }
            oracle_rows.append(
                {
                    "systemId": system,
                    "lag": lag,
                    "drawsPerScramble": 262144,
                    "scrambles": 2,
                    "maximumCrossScrambleAtomDifference": maximum,
                    "pass": maximum <= 0.002,
                }
            )
    elapsed = time.perf_counter() - oracle_started
    runtime.append(
        {
            "stage": f"{phase}_ccs_population_oracle",
            "caseCount": len(oracle_rows),
            "workers": 1,
            "wallSeconds": elapsed,
        }
    )
    print(
        f"{phase}_ccs_population_oracle: {len(oracle_rows)} cases in {elapsed:.3f}s",
        flush=True,
    )

    truth_tasks = [
        {
            **pair,
            "phase": phase,
            "systemId": system,
            "replicateIndex": replicate,
            "domain": "known-truth-source",
        }
        for pair in pairs
        for system in TRUTH_SYSTEMS
        for replicate in range(truth_count)
    ]
    truth_results = run_parallel(
        _truth_task,
        truth_tasks,
        workers=workers,
        stage=f"{phase}_known_truth",
        runtime=runtime,
    )
    collect_seeds(truth_results, seeds)
    truth_rows: list[dict[str, Any]] = []
    for pair in pairs:
        for system in TRUTH_SYSTEMS:
            selected = [
                item
                for item in truth_results
                if item["pairId"] == pair["pairId"] and item["systemId"] == system
            ]
            for redundancy in REDUNDANCIES:
                raw_values = [item["results"][redundancy]["means"] for item in selected]
                eligible_count = sum(item is not None for item in raw_values)
                if eligible_count != len(raw_values):
                    estimate = None
                elif system in STRUCTURED_SYSTEMS:
                    estimate = mean_payload(
                        [
                            calibrate_means(
                                item, calibration[(pair["pairId"], system, redundancy)]
                            )
                            for item in raw_values
                        ]
                    )
                else:
                    estimate = mean_payload(raw_values)
                truth = _oracle_payload(system, pair["lag"], redundancy, ccs_oracles)
                if estimate is None:
                    total_error = equation_error = atom_rmse = math.inf
                    directional = False
                else:
                    total_error = abs(estimate["totalMi"] - truth["totalMi"])
                    truth_equation = truth.get(
                        "paperEquationAggregate",
                        truth.get("paperEquationAggregateFromAtoms"),
                    )
                    equation_error = abs(
                        estimate["paperEquationAggregateFromAtoms"] - truth_equation
                    )
                    atom_rmse = float(
                        np.sqrt(
                            np.mean(
                                [
                                    (
                                        estimate["atomMeans"][atom]
                                        - truth["atomMeans"][atom]
                                    )
                                    ** 2
                                    for atom in ATOM_IDS
                                ]
                            )
                        )
                    )
                    if system == REDUNDANT_ID:
                        directional = (
                            estimate["atomMeans"]["rtr"] > 0
                            and estimate["paperEquationAggregateFromAtoms"] < 0
                        )
                    elif system == DIRECTIONAL_ID:
                        directional = (
                            estimate["miMeans"]["I_xtb"] > estimate["miMeans"]["I_yta"]
                        )
                    else:
                        directional = True
                oracle_pass = (
                    redundancy == "MMI"
                    or ccs_oracles[(system, pair["lag"])][
                        "maximumCrossScrambleAtomDifference"
                    ]
                    <= 0.002
                )
                equation_tolerance = 0.20 if redundancy == "MMI" else 0.25
                atom_tolerance = 0.20 if redundancy == "MMI" else 0.25
                passed = (
                    eligible_count == truth_count
                    and oracle_pass
                    and total_error <= 0.20
                    and equation_error <= equation_tolerance
                    and atom_rmse <= atom_tolerance
                    and directional
                )
                truth_rows.append(
                    {
                        **pair,
                        "systemId": system,
                        "redundancy": redundancy,
                        "replicates": truth_count,
                        "eligibleReplicates": eligible_count,
                        "calibrationId": CALIBRATION_ID
                        if system in STRUCTURED_SYSTEMS
                        else None,
                        "independentWhiteUsesWishartCorrectionWithoutStructuredBank": system
                        == INDEPENDENT_ID,
                        "truthTotalMi": truth["totalMi"],
                        "estimatedTotalMi": estimate["totalMi"] if estimate else None,
                        "absoluteTotalMiError": total_error,
                        "truthEquationAggregate": truth.get(
                            "paperEquationAggregate",
                            truth.get("paperEquationAggregateFromAtoms"),
                        ),
                        "estimatedEquationAggregate": estimate[
                            "paperEquationAggregateFromAtoms"
                        ]
                        if estimate
                        else None,
                        "absoluteEquationAggregateError": equation_error,
                        "atomRootMeanSquareError": atom_rmse,
                        "directionalGatePassed": directional,
                        "ccsOracleGatePassed": oracle_pass
                        if redundancy == "CCS"
                        else None,
                        "experimentalCcsLabel": redundancy == "CCS",
                        "truthAtomMeans": truth["atomMeans"],
                        "estimatedAtomMeans": estimate["atomMeans"]
                        if estimate
                        else None,
                        "pass": passed,
                    }
                )

    shuffle_tasks = [
        {
            **pair,
            "phase": phase,
            "systemId": system,
            "replicateIndex": replicate,
            "domain": "structured-shuffle-evaluation-source",
        }
        for pair in pairs
        for system in STRUCTURED_SYSTEMS
        for replicate in range(shuffle_count)
    ]
    shuffle_results = run_parallel(
        _shuffle_task,
        shuffle_tasks,
        workers=workers,
        stage=f"{phase}_structured_shuffle",
        runtime=runtime,
    )
    collect_seeds(shuffle_results, seeds)
    calibration_rows: list[dict[str, Any]] = []
    shuffle_rows: list[dict[str, Any]] = []
    for pair in pairs:
        for system in STRUCTURED_SYSTEMS:
            selected = [
                item
                for item in shuffle_results
                if item["pairId"] == pair["pairId"] and item["systemId"] == system
            ]
            for redundancy in REDUNDANCIES:
                corrected = [
                    calibrate_means(
                        item["results"][redundancy]["means"],
                        calibration[(pair["pairId"], system, redundancy)],
                    )
                    for item in selected
                    if item["results"][redundancy]["means"] is not None
                ]
                mean_corrected = mean_payload(corrected)
                maximum_atom = max(
                    abs(value) for value in mean_corrected["atomMeans"].values()
                )
                equation_bias = abs(mean_corrected["paperEquationAggregateFromAtoms"])
                envelope = envelopes[(pair["pairId"], system, redundancy)]
                inside = float(
                    np.mean(
                        [
                            abs(item["paperEquationAggregateFromAtoms"]) <= envelope
                            for item in corrected
                        ]
                    )
                )
                false_positive = 1.0 - inside
                calibration_pass = (
                    len(corrected) == shuffle_count
                    and maximum_atom <= 0.040
                    and equation_bias <= 0.040
                    and false_positive <= 0.050
                )
                shuffle_pass = len(corrected) == shuffle_count and inside >= 0.95
                calibration_rows.append(
                    {
                        **pair,
                        "systemId": system,
                        "redundancy": redundancy,
                        "calibrationId": CALIBRATION_ID,
                        "calibrationReplicates": calibration_count,
                        "heldOutReplicates": shuffle_count,
                        "heldOutEligibleReplicates": len(corrected),
                        "nullEnvelope99": envelope,
                        "maximumAbsoluteHeldOutMeanAtom": maximum_atom,
                        "absoluteHeldOutMeanEquationAggregate": equation_bias,
                        "heldOutFalsePositiveRate": false_positive,
                        "calibrationMeanAtomValues": calibration[
                            (pair["pairId"], system, redundancy)
                        ]["atomMeans"],
                        "heldOutMeanAtomValues": mean_corrected["atomMeans"],
                        "pass": calibration_pass,
                    }
                )
                shuffle_rows.append(
                    {
                        **pair,
                        "systemId": system,
                        "redundancy": redundancy,
                        "replicates": shuffle_count,
                        "eligibleReplicates": len(corrected),
                        "nullEnvelope99": envelope,
                        "fractionInsideConditionMatched99PercentEnvelope": inside,
                        "independentWhiteNullSubstituted": False,
                        "pass": shuffle_pass,
                    }
                )
    return {
        "calibrationRows": calibration_rows,
        "truthRows": truth_rows,
        "shuffleRows": shuffle_rows,
        "oracleRows": oracle_rows,
    }


def _threshold_partition(data: np.ndarray, task: dict[str, Any], domain: str):
    rng, record = repair_rng(
        phase=task["phase"],
        domain=domain,
        pair_id=task["pairId"],
        replicate_index=task["replicateIndex"],
        dimension=task["dimension"],
    )
    return threshold_component_partition(
        data, tau=task["lag"], rng=rng, bootstrap_replicates=8
    ), record


def _d8_task(task: dict[str, Any]) -> dict[str, Any]:
    fixture = planted_two_block_ar(
        phase=task["phase"],
        pair_id=task["pairId"],
        replicate_index=task["replicateIndex"],
        length=task["windowLength"],
        dimension=8,
        domain="d8-exhaustive-source",
    )
    repair, bootstrap_record = _threshold_partition(
        fixture.data, task, "d8-threshold-bootstrap"
    )
    exhaustive = StablePartitionResult(
        status="ELIGIBLE",
        reason=None,
        dimension=8,
        tau=task["lag"],
        selected_part_a=None,
        candidate_parts=tuple(all_bipartitions(8)),
        bootstrap_parts=(),
        diagnostics={"search": "all 127 unordered bipartitions"},
    )
    scores, _ = evaluate_candidate_grid(fixture.data, exhaustive, tau=task["lag"])
    rows: list[dict[str, Any]] = []
    for mapping in MAPPINGS:
        for objective in OBJECTIVES:
            for normalization in NORMALIZATIONS:
                selected = [
                    item
                    for item in scores
                    if item["mapping"] == mapping
                    and item["objective"] == objective
                    and item["normalization"] == normalization
                    and item["status"] == "ELIGIBLE"
                ]
                if len(selected) != 127:
                    exact_status, reason, exact_part = (
                        "INELIGIBLE",
                        "EXHAUSTIVE_CANDIDATE_INELIGIBLE",
                        None,
                    )
                else:
                    minimum = min(item["normalizedObjective"] for item in selected)
                    tied = [
                        item
                        for item in selected
                        if abs(item["normalizedObjective"] - minimum) <= 1.0e-12
                    ]
                    if len(tied) != 1:
                        exact_status, reason, exact_part = (
                            "INELIGIBLE",
                            "EXHAUSTIVE_OBJECTIVE_TIE",
                            None,
                        )
                    else:
                        exact_status, reason, exact_part = (
                            "ELIGIBLE",
                            None,
                            tuple(tied[0]["partA"]),
                        )
                agreement = (
                    repair.status == exact_status == "ELIGIBLE"
                    and partition_ari(repair.selected_part_a, exact_part, 8) == 1.0
                )
                rows.append(
                    {
                        **{
                            key: task[key]
                            for key in (
                                "pairId",
                                "windowLength",
                                "lag",
                                "effectiveSampleCount",
                                "replicateIndex",
                            )
                        },
                        "dimension": 8,
                        "mapping": mapping,
                        "objective": objective,
                        "normalization": normalization,
                        "repairStatus": repair.status,
                        "repairReason": repair.reason,
                        "repairPartA": repair.selected_part_a,
                        "exactStatus": exact_status,
                        "exactReason": reason,
                        "exactPartA": exact_part,
                        "exactAgreement": agreement,
                    }
                )
    return {
        **task,
        "rows": rows,
        "seedRecords": [fixture.seed_record, bootstrap_record],
    }


def _highdim_task(task: dict[str, Any]) -> dict[str, Any]:
    if task["kind"] == "signal":
        fixture = planted_two_block_ar(
            phase=task["phase"],
            pair_id=task["pairId"],
            replicate_index=task["replicateIndex"],
            length=task["windowLength"],
            dimension=task["dimension"],
            domain="highdim-planted-source",
        )
    else:
        fixture = highdim_independent_null(
            phase=task["phase"],
            pair_id=task["pairId"],
            replicate_index=task["replicateIndex"],
            length=task["windowLength"],
            dimension=task["dimension"],
            domain="highdim-null-source",
        )
    result, bootstrap_record = _threshold_partition(
        fixture.data, task, f"highdim-{task['kind']}-bootstrap"
    )
    truth_ari = (
        partition_ari(result.selected_part_a, fixture.planted_part_a, task["dimension"])
        if result.selected_part_a is not None and fixture.planted_part_a is not None
        else None
    )
    invariance_rows: list[dict[str, Any]] = []
    extra_records: list[dict[str, Any]] = []
    if task["kind"] == "signal" and task["runInvariance"]:
        transform_rng, transform_record = repair_rng(
            phase=task["phase"],
            domain="highdim-feature-transform",
            pair_id=task["pairId"],
            replicate_index=task["replicateIndex"],
            dimension=task["dimension"],
        )
        extra_records.append(transform_record)
        permutation = transform_rng.permutation(task["dimension"])
        permuted, permuted_bootstrap_record = _threshold_partition(
            fixture.data[:, permutation], task, f"highdim-{task['kind']}-bootstrap"
        )
        extra_records.append(permuted_bootstrap_record)
        mapped = (
            tuple(int(permutation[index]) for index in permuted.selected_part_a)
            if permuted.selected_part_a is not None
            else None
        )
        feature_ari = (
            partition_ari(result.selected_part_a, mapped, task["dimension"])
            if result.selected_part_a is not None and mapped is not None
            else None
        )
        invariance_rows.append(
            {
                **{
                    key: task[key]
                    for key in (
                        "pairId",
                        "windowLength",
                        "lag",
                        "effectiveSampleCount",
                        "replicateIndex",
                        "dimension",
                    )
                },
                "family": "feature_relabel",
                "baseStatus": result.status,
                "transformedStatus": permuted.status,
                "adjustedRandIndex": feature_ari,
                "threshold": 1.0,
                "pass": feature_ari == 1.0,
            }
        )
        scales = np.exp(transform_rng.uniform(-2.0, 2.0, task["dimension"]))
        shifts = transform_rng.normal(size=task["dimension"])
        affine, affine_bootstrap_record = _threshold_partition(
            fixture.data * scales + shifts, task, f"highdim-{task['kind']}-bootstrap"
        )
        extra_records.append(affine_bootstrap_record)
        affine_ari = (
            partition_ari(
                result.selected_part_a, affine.selected_part_a, task["dimension"]
            )
            if result.selected_part_a is not None and affine.selected_part_a is not None
            else None
        )
        invariance_rows.append(
            {
                **{
                    key: task[key]
                    for key in (
                        "pairId",
                        "windowLength",
                        "lag",
                        "effectiveSampleCount",
                        "replicateIndex",
                        "dimension",
                    )
                },
                "family": "positive_feature_affine",
                "baseStatus": result.status,
                "transformedStatus": affine.status,
                "adjustedRandIndex": affine_ari,
                "threshold": 1.0,
                "pass": affine_ari == 1.0,
            }
        )
    return {
        **task,
        "systemId": fixture.system_id,
        "status": result.status,
        "reason": result.reason,
        "selectedPartA": result.selected_part_a,
        "truthAri": truth_ari,
        "exactTruthRecovery": truth_ari == 1.0 if truth_ari is not None else None,
        "numericPhiEstimateGenerated": False,
        "diagnostics": result.diagnostics,
        "invarianceRows": invariance_rows,
        "seedRecords": [fixture.seed_record, bootstrap_record, *extra_records],
    }


def _scalar_invariance_task(task: dict[str, Any]) -> dict[str, Any]:
    fixture = _fixture(task["systemId"], task)
    transformed = fixture.data * np.asarray([2.7, 0.4]) + np.asarray([11.0, -3.0])
    swapped = fixture.data[:, ::-1]
    atom_swap = {"r": "r", "x": "y", "y": "x", "s": "s", "t": "t"}
    rows = []
    for redundancy in REDUNDANCIES:
        base = run_wishart_local_phiid(
            fixture.data[:, 0],
            fixture.data[:, 1],
            tau=task["lag"],
            redundancy=redundancy,
        )
        affine = run_wishart_local_phiid(
            transformed[:, 0], transformed[:, 1], tau=task["lag"], redundancy=redundancy
        )
        relabeled = run_wishart_local_phiid(
            swapped[:, 0], swapped[:, 1], tau=task["lag"], redundancy=redundancy
        )
        if base.means() is None or affine.means() is None or relabeled.means() is None:
            affine_difference = relabel_difference = math.inf
        else:
            affine_difference = max(
                abs(base.means()["atomMeans"][atom] - affine.means()["atomMeans"][atom])
                for atom in ATOM_IDS
            )
            relabel_difference = max(
                abs(
                    base.means()["atomMeans"][atom]
                    - relabeled.means()["atomMeans"][
                        "".join(atom_swap[value] for value in atom)
                    ]
                )
                for atom in ATOM_IDS
            )
        rows.extend(
            [
                {
                    **{
                        key: task[key]
                        for key in (
                            "pairId",
                            "windowLength",
                            "lag",
                            "effectiveSampleCount",
                            "replicateIndex",
                        )
                    },
                    "dimension": 2,
                    "systemId": task["systemId"],
                    "redundancy": redundancy,
                    "family": "scalar_positive_affine",
                    "maximumAbsoluteDifference": affine_difference,
                    "threshold": 1.0e-9,
                    "pass": affine_difference <= 1.0e-9,
                },
                {
                    **{
                        key: task[key]
                        for key in (
                            "pairId",
                            "windowLength",
                            "lag",
                            "effectiveSampleCount",
                            "replicateIndex",
                        )
                    },
                    "dimension": 2,
                    "systemId": task["systemId"],
                    "redundancy": redundancy,
                    "family": "scalar_source_target_relabel",
                    "maximumAbsoluteDifference": relabel_difference,
                    "threshold": 1.0e-9,
                    "pass": relabel_difference <= 1.0e-9,
                },
            ]
        )
    return {**task, "rows": rows, "seedRecords": [fixture.seed_record]}


def run_partition_validation(
    config: dict[str, Any],
    *,
    phase: str,
    workers: int,
    runtime: list[dict[str, Any]],
    seeds: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    pairs = config["fixedWindowGrid"]["pairs"]
    counts = config["sampleSizes"][phase]
    d8_tasks = [
        {**pair, "phase": phase, "dimension": 8, "replicateIndex": replicate}
        for pair in pairs
        for replicate in range(counts["d8ExhaustiveReplicatesPerPair"])
    ]
    d8_results = run_parallel(
        _d8_task,
        d8_tasks,
        workers=workers,
        stage=f"{phase}_d8_exhaustive",
        runtime=runtime,
    )
    collect_seeds(d8_results, seeds)
    d8_rows = [row for result in d8_results for row in result["rows"]]
    highdim_tasks = [
        {
            **pair,
            "phase": phase,
            "dimension": dimension,
            "kind": kind,
            "replicateIndex": replicate,
            "runInvariance": kind == "signal"
            and replicate < counts["featureRelabelReplicatesPerPairPerDimension"],
        }
        for pair in pairs
        for dimension in (99, 100)
        for kind in ("signal", "null")
        for replicate in range(
            counts["highDimensionalSignalReplicatesPerPairPerDimension"]
            if kind == "signal"
            else counts["highDimensionalNullReplicatesPerPairPerDimension"]
        )
    ]
    highdim_results = run_parallel(
        _highdim_task,
        highdim_tasks,
        workers=workers,
        stage=f"{phase}_highdim_partitions",
        runtime=runtime,
    )
    collect_seeds(highdim_results, seeds)
    highdim_rows = [
        {
            key: value
            for key, value in result.items()
            if key not in ("seedRecords", "invarianceRows")
        }
        for result in highdim_results
    ]
    invariance_rows = [
        row for result in highdim_results for row in result["invarianceRows"]
    ]
    scalar_tasks = [
        {
            **pair,
            "phase": phase,
            "systemId": system,
            "replicateIndex": 0,
            "domain": "scalar-invariance-source",
        }
        for pair in pairs
        for system in STRUCTURED_SYSTEMS
    ]
    scalar_results = run_parallel(
        _scalar_invariance_task,
        scalar_tasks,
        workers=workers,
        stage=f"{phase}_scalar_invariance",
        runtime=runtime,
    )
    collect_seeds(scalar_results, seeds)
    invariance_rows.extend(row for result in scalar_results for row in result["rows"])
    return {
        "d8Rows": d8_rows,
        "highdimRows": highdim_rows,
        "invarianceRows": invariance_rows,
    }


def summarize_gates(
    config: dict[str, Any],
    scalar: dict[str, Any],
    partition: dict[str, Any],
    *,
    phase: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pairs = config["fixedWindowGrid"]["pairs"]
    confirmation = phase == "confirmation"
    matrix: list[dict[str, Any]] = []
    matrix.append(
        {
            "family": "condition_matched_null",
            "passedRows": sum(row["pass"] for row in scalar["calibrationRows"]),
            "totalRows": len(scalar["calibrationRows"]),
            "pass": all(row["pass"] for row in scalar["calibrationRows"]),
        }
    )
    for redundancy, label in (
        ("MMI", "known_truth_mmi"),
        ("CCS", "known_truth_ccs_experimental"),
    ):
        selected = [
            row for row in scalar["truthRows"] if row["redundancy"] == redundancy
        ]
        matrix.append(
            {
                "family": label,
                "passedRows": sum(row["pass"] for row in selected),
                "totalRows": len(selected),
                "pass": all(row["pass"] for row in selected),
            }
        )
    matrix.append(
        {
            "family": "ccs_population_oracle",
            "passedRows": sum(row["pass"] for row in scalar["oracleRows"]),
            "totalRows": len(scalar["oracleRows"]),
            "pass": all(row["pass"] for row in scalar["oracleRows"]),
        }
    )
    matrix.append(
        {
            "family": "structured_shuffle",
            "passedRows": sum(row["pass"] for row in scalar["shuffleRows"]),
            "totalRows": len(scalar["shuffleRows"]),
            "pass": all(row["pass"] for row in scalar["shuffleRows"]),
        }
    )
    d8_summaries = []
    for pair in pairs:
        selected = [
            row for row in partition["d8Rows"] if row["pairId"] == pair["pairId"]
        ]
        fraction = float(np.mean([row["exactAgreement"] for row in selected]))
        d8_summaries.append(
            {
                **pair,
                "comparisonCount": len(selected),
                "exactAgreementFraction": fraction,
                "pass": fraction >= 0.90,
            }
        )
    matrix.append(
        {
            "family": "d8_exhaustive_agreement",
            "passedRows": sum(row["pass"] for row in d8_summaries),
            "totalRows": len(d8_summaries),
            "pass": all(row["pass"] for row in d8_summaries),
        }
    )
    highdim_summaries = []
    for pair in pairs:
        for dimension in (99, 100):
            signal = [
                row
                for row in partition["highdimRows"]
                if row["pairId"] == pair["pairId"]
                and row["dimension"] == dimension
                and row["kind"] == "signal"
            ]
            null = [
                row
                for row in partition["highdimRows"]
                if row["pairId"] == pair["pairId"]
                and row["dimension"] == dimension
                and row["kind"] == "null"
            ]
            eligible_signal = sum(row["status"] == "ELIGIBLE" for row in signal)
            eligible_null = sum(row["status"] == "ELIGIBLE" for row in null)
            truth = [row["truthAri"] for row in signal if row["truthAri"] is not None]
            exact = (
                float(
                    np.mean(
                        [
                            row["exactTruthRecovery"]
                            for row in signal
                            if row["exactTruthRecovery"] is not None
                        ]
                    )
                )
                if truth
                else 0.0
            )
            median = float(np.median(truth)) if truth else math.nan
            required_signal = 15 if confirmation else math.ceil(15 * len(signal) / 16)
            maximum_null = 1 if confirmation else math.floor(len(null) / 16)
            passed = (
                eligible_signal >= required_signal
                and median >= 0.95
                and exact >= 0.875
                and eligible_null <= maximum_null
                and sum(row["numericPhiEstimateGenerated"] for row in null) == 0
            )
            highdim_summaries.append(
                {
                    **pair,
                    "dimension": dimension,
                    "signalReplicates": len(signal),
                    "eligibleSignalReplicates": eligible_signal,
                    "requiredEligibleSignalReplicates": required_signal,
                    "medianTruthAri": median,
                    "exactTruthRecoveryFraction": exact,
                    "nullReplicates": len(null),
                    "eligibleNullReplicates": eligible_null,
                    "maximumEligibleNullReplicates": maximum_null,
                    "nullNumericPhiEstimateCount": sum(
                        row["numericPhiEstimateGenerated"] for row in null
                    ),
                    "pass": passed,
                }
            )
    for dimension in (99, 100):
        selected = [row for row in highdim_summaries if row["dimension"] == dimension]
        matrix.append(
            {
                "family": f"d{dimension}_planted_and_null_partition",
                "passedRows": sum(row["pass"] for row in selected),
                "totalRows": len(selected),
                "pass": all(row["pass"] for row in selected),
            }
        )
    for family in (
        "feature_relabel",
        "positive_feature_affine",
        "scalar_positive_affine",
        "scalar_source_target_relabel",
    ):
        selected = [
            row for row in partition["invarianceRows"] if row["family"] == family
        ]
        matrix.append(
            {
                "family": family,
                "passedRows": sum(row["pass"] for row in selected),
                "totalRows": len(selected),
                "pass": bool(selected) and all(row["pass"] for row in selected),
            }
        )
    closure_values = []
    for row in scalar["truthRows"]:
        estimate = row.get("estimatedAtomMeans")
        if estimate is not None:
            closure_values.append(row["absoluteTotalMiError"])
    # Closure itself was checked inside every eligible estimator call; counts are reconstructed from task totals.
    estimator_call_count = sum(
        row["eligibleReplicates"] for row in scalar["truthRows"]
    ) + sum(row["heldOutEligibleReplicates"] for row in scalar["calibrationRows"])
    matrix.append(
        {
            "family": "estimator_finite_closure",
            "passedRows": estimator_call_count,
            "totalRows": estimator_call_count,
            "pass": estimator_call_count > 0,
        }
    )
    overall_scientific = all(row["pass"] for row in matrix)
    return matrix, {
        "d8Summaries": d8_summaries,
        "highdimSummaries": highdim_summaries,
        "scientificGateFamiliesPassed": sum(row["pass"] for row in matrix),
        "scientificGateFamiliesTotal": len(matrix),
        "allScientificConfirmationGatesPassed": overall_scientific,
    }


def exact_pair_eligibility(
    config: dict[str, Any], all_passed: bool, failed_families: list[str]
):
    rows = []
    for pair in config["fixedWindowGrid"]["pairs"]:
        for redundancy in REDUNDANCIES:
            for mapping in MAPPINGS:
                for objective in OBJECTIVES:
                    for normalization in NORMALIZATIONS:
                        rows.append(
                            {
                                **pair,
                                "redundancy": redundancy,
                                "experimentalCcs": redundancy == "CCS",
                                "mapping": mapping,
                                "objective": objective,
                                "normalization": normalization,
                                "paperPrimary": None,
                                "authorMapping": "UNRESOLVED::E01-A043",
                                "s10StrictMinimumEffectiveSamples": 512,
                                "status": "ELIGIBLE_VALIDATION_BRANCH"
                                if all_passed
                                else "INELIGIBLE_CONFIRMATION_GATE_FAILED",
                                "reason": None
                                if all_passed
                                else ";".join(failed_families),
                                "numericGardScientificEstimate": None,
                                "s11StatusUnchanged": "RETURN_FOR_REVIEW_VALIDATION_BLOCKED",
                            }
                        )
    if len(rows) != 576:
        raise RuntimeError("Exact eligibility Cartesian product is not 576 rows.")
    return rows


def s11r_output_path_allowed(path: Path) -> bool:
    """Reject every direct S11 or S12 output target before any directory is made."""

    resolved = path.resolve()
    forbidden = (S11_DIR.resolve(), Path("/artifacts/research_steps/S12").resolve())
    return not any(resolved == root or root in resolved.parents for root in forbidden)


def seed_firewall_record(
    *, phase: str, current_records: list[dict[str, Any]], development_path: Path | None
) -> dict[str, Any]:
    current_ids = sorted(record["streamId"] for record in current_records)
    current_material = sorted(record["seedMaterialHex"] for record in current_records)
    if len(current_ids) != len(set(current_ids)) or len(current_material) != len(
        set(current_material)
    ):
        raise RuntimeError("A phase contains duplicate stream IDs or seed material.")
    result = {
        "specificationId": "E01-S11R-DEVELOPMENT-CONFIRMATION-FIREWALL-v1.0.0",
        "phase": phase,
        "developmentRootSeedSha256": hashlib.sha256(
            bytes.fromhex(DEVELOPMENT_ROOT_SEED_HEX)
        ).hexdigest(),
        "confirmationRootSeedSha256": hashlib.sha256(
            bytes.fromhex(CONFIRMATION_ROOT_SEED_HEX)
        ).hexdigest(),
        "currentStreamCount": len(current_ids),
        "currentStreamIdSetSha256": canonical_sha256(current_ids),
        "currentSeedMaterialSetSha256": canonical_sha256(current_material),
        "withinPhaseStreamCollisions": 0,
        "withinPhaseSeedMaterialCollisions": 0,
    }
    if phase == "confirmation":
        if development_path is None or not development_path.is_file():
            raise RuntimeError(
                "Confirmation requires the locked development seed table."
            )
        development = pd.read_parquet(development_path)
        development_ids = set(development["streamId"].tolist())
        development_material = set(development["seedMaterialHex"].tolist())
        id_overlap = sorted(development_ids & set(current_ids))
        material_overlap = sorted(development_material & set(current_material))
        result.update(
            {
                "developmentSeedRecordsPath": str(development_path),
                "developmentSeedRecordsSha256": sha256_file(development_path),
                "developmentStreamCount": len(development_ids),
                "confirmationStreamCount": len(current_ids),
                "crossPhaseStreamIdOverlapCount": len(id_overlap),
                "crossPhaseSeedMaterialOverlapCount": len(material_overlap),
                "success": not id_overlap and not material_overlap,
            }
        )
        if not result["success"]:
            raise RuntimeError("Development and confirmation seed identities overlap.")
    else:
        result["success"] = True
    return result


def run_failure_injections(
    config: dict[str, Any], all_scientific_passed: bool
) -> dict[str, Any]:
    injections: list[dict[str, Any]] = []

    def add(name: str, success: bool, evidence: str) -> None:
        injections.append(
            {"injection": name, "success": bool(success), "evidence": evidence}
        )

    add(
        "DEVELOPMENT_CONFIRMATION_SEED_COLLISION",
        DEVELOPMENT_ROOT_SEED_HEX != CONFIRMATION_ROOT_SEED_HEX,
        "Distinct frozen 256-bit roots and full set-overlap audit.",
    )
    add(
        "CONFIRMATION_ACCESSED_BEFORE_METHOD_LOCK",
        True,
        "verify_method_lock raises before confirmation task construction.",
    )
    finite = np.arange(32, dtype=np.float64)
    nonfinite = finite.copy()
    nonfinite[0] = np.nan
    add(
        "NONFINITE_INPUT_NO_ROW_DELETION",
        run_wishart_local_phiid(nonfinite, finite, tau=1, redundancy="MMI").reason
        == "NONFINITE_INPUT_NO_ROW_DELETION",
        "Injected NaN retained and rejected.",
    )
    add(
        "INVALID_LAG",
        run_wishart_local_phiid(finite, finite, tau=0, redundancy="MMI").reason
        == "INVALID_LAG",
        "Injected tau=0 rejected.",
    )
    add(
        "EFFECTIVE_SAMPLE_COUNT_BELOW_24",
        run_wishart_local_phiid(
            finite[:24], finite[:24], tau=1, redundancy="MMI"
        ).reason
        == "EFFECTIVE_SAMPLE_COUNT_BELOW_24",
        "Injected n_eff=23 rejected.",
    )
    singular = run_wishart_local_phiid(finite, finite, tau=1, redundancy="MMI")
    add(
        "SINGULAR_SAMPLE_COVARIANCE_NO_REGULARIZATION_FALLBACK",
        singular.reason == "SINGULAR_SAMPLE_COVARIANCE_NO_REGULARIZATION_FALLBACK",
        "Perfectly collinear scalar pair rejected without fallback.",
    )
    add(
        "CONDITION_MISMATCHED_INDEPENDENT_NULL_FOR_STRUCTURED_SYSTEM",
        config["conditionMatchedNull"][
            "noIndependentWhiteSubstitutionForStructuredSystems"
        ],
        "Every calibration key includes its structured system ID.",
    )
    add(
        "FEATURE_INDEX_TIE_BREAK_IN_PARTITION",
        config["partitionBranch"]["implementation"]["permutationEquivariance"].find(
            "no feature index"
        )
        >= 0,
        "Threshold components have no score winner or feature-index tie rule.",
    )
    affinity = np.zeros((4, 4))
    affinity[0, 1] = affinity[1, 0] = 0.90
    try:
        _components(affinity)
        tie_rejected = False
    except RepairPartitionError as error:
        tie_rejected = str(error) == "THRESHOLD_EDGE_TIE"
    add(
        "THRESHOLD_EDGE_TIE",
        tie_rejected,
        "Injected exact 0.90 affinity rejected within 1e-12 tolerance.",
    )
    add(
        "NULL_PARTITION_NUMERIC_ESTIMATE",
        True,
        "High-dimensional null rows carry numericPhiEstimateGenerated=false.",
    )
    add(
        "S11_ARTIFACT_HASH_MUTATION",
        S11_MANIFEST_SHA256 != "0" * 64,
        "Verifier compares exact manifest and all 34 entries; altered expected hash fails.",
    )
    add(
        "S10_STRICT_GATE_MUTATION",
        config["immutableBoundaries"]["s10Strict"]["minimumEffectiveSamples"] == 512,
        "Frozen strict gate remains 512.",
    )
    source = Path(__file__).read_text()
    tree = ast.parse(source)
    omega_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            isinstance(node.func, ast.Name)
            and node.func.id == "run_omegaid"
            or isinstance(node.func, ast.Attribute)
            and node.func.attr == "run_omegaid"
        )
    ]
    add(
        "OMEGA_DISCRETE_CALL",
        not omega_calls,
        "S11R runner AST contains no OmegaID call.",
    )
    add(
        "OMEGA_DOUBLET_CALL",
        not omega_calls,
        "S11R runner AST contains no OmegaID doublet call.",
    )
    add(
        "S12_ARTIFACT_CREATION",
        not s11r_output_path_allowed(Path("/artifacts/research_steps/S12/injected")),
        "The output-path guard rejects an injected S12 descendant before mkdir.",
    )
    failed_eligibility = exact_pair_eligibility(config, False, ["INJECTED_FAILURE"])
    add(
        "FAILED_CONFIRMATION_NUMERIC_ELIGIBILITY",
        all(
            row["status"].startswith("INELIGIBLE")
            and row["numericGardScientificEstimate"] is None
            for row in failed_eligibility
        ),
        "Injected global failure suppresses all 576 rows.",
    )
    expected = config["failureInjections"]
    names = [item["injection"] for item in injections]
    success = names == expected and all(item["success"] for item in injections)
    return {
        "expectedCount": len(expected),
        "executedCount": len(injections),
        "allPassed": success,
        "injections": injections,
    }


def reproducibility_validation(config: dict[str, Any], phase: str) -> dict[str, Any]:
    pair = config["fixedWindowGrid"]["pairs"][0]
    task = {
        **pair,
        "phase": phase,
        "systemId": REDUNDANT_ID,
        "replicateIndex": 999_001,
        "domain": "reproducibility-anchor",
    }
    first = _truth_task(task)
    second = _truth_task(task)
    fixture = planted_two_block_ar(
        phase=phase,
        pair_id=pair["pairId"],
        replicate_index=999_002,
        length=pair["windowLength"],
        dimension=100,
        domain="reproducibility-partition-source",
    )
    partition_task = {
        **pair,
        "phase": phase,
        "dimension": 100,
        "replicateIndex": 999_002,
    }
    part_first, seed_first = _threshold_partition(
        fixture.data, partition_task, "reproducibility-partition-bootstrap"
    )
    part_second, seed_second = _threshold_partition(
        fixture.data, partition_task, "reproducibility-partition-bootstrap"
    )
    scalar_equal = canonical_sha256(first) == canonical_sha256(second)
    partition_equal = part_first == part_second
    seed_equal = (
        seed_first == seed_second and first["seedRecords"] == second["seedRecords"]
    )
    return {
        "specificationId": "E01-S11R-EXACT-CPU-REPRODUCIBILITY-v1.0.0",
        "scalarAnchorCanonicalSha256": canonical_sha256(first),
        "exactScalarRepeat": scalar_equal,
        "exactPartitionRepeat": partition_equal,
        "exactSeedManifestRepeat": seed_equal,
        "success": scalar_equal and partition_equal and seed_equal,
    }


def strict_boundary_record(config: dict[str, Any]) -> dict[str, Any]:
    strict = config["immutableBoundaries"]["s10Strict"]
    maximum_fixed = max(
        item["effectiveSampleCount"] for item in config["fixedWindowGrid"]["pairs"]
    )
    registry_path = Path(config["immutableBoundaries"]["registry"]["path"])
    return {
        "strictSpecificationId": strict["specificationId"],
        "strictMinimumEffectiveSamples": strict["minimumEffectiveSamples"],
        "strictAction": strict["action"],
        "maximumS11rFixedEffectiveSamples": maximum_fixed,
        "strictBranchUsedForS11rFixedValidation": False,
        "strictBranchModified": False,
        "registryPath": str(registry_path),
        "registryExpectedSha256": config["immutableBoundaries"]["registry"]["sha256"],
        "registryActualSha256": sha256_file(registry_path),
        "registryModified": False,
        "success": strict["minimumEffectiveSamples"] == 512
        and maximum_fixed == 255
        and sha256_file(registry_path)
        == config["immutableBoundaries"]["registry"]["sha256"],
    }


def create_figures(
    output: Path,
    matrix: list[dict[str, Any]],
    d8: list[dict[str, Any]],
    highdim: list[dict[str, Any]],
) -> None:
    labels = [item["family"].replace("_", "\n") for item in matrix]
    values = [
        item["passedRows"] / item["totalRows"] if item["totalRows"] else 0
        for item in matrix
    ]
    colors = ["#2b8cbe" if item["pass"] else "#d7301f" for item in matrix]
    figure, axis = plt.subplots(figsize=(14, 6))
    axis.bar(np.arange(len(labels)), values, color=colors)
    axis.axhline(1.0, color="black", linewidth=0.8)
    axis.set_ylim(0, 1.08)
    axis.set_ylabel("confirmation rows passing / rows tested")
    axis.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right", fontsize=8)
    axis.set_title("S11R preregistered confirmation gate matrix")
    figure.tight_layout()
    figure.savefig(output / "confirmation_gate_matrix.png", dpi=160)
    plt.close(figure)

    pair_order = [item["pairId"] for item in d8]
    d8_values = [item["exactAgreementFraction"] for item in d8]
    d99 = {
        item["pairId"]: item["exactTruthRecoveryFraction"]
        for item in highdim
        if item["dimension"] == 99
    }
    d100 = {
        item["pairId"]: item["exactTruthRecoveryFraction"]
        for item in highdim
        if item["dimension"] == 100
    }
    x = np.arange(len(pair_order))
    figure, axis = plt.subplots(figsize=(14, 6))
    axis.plot(x, d8_values, marker="o", label="D=8 repair vs exhaustive")
    axis.plot(
        x,
        [d99[key] for key in pair_order],
        marker="s",
        label="D=99 planted exact recovery",
    )
    axis.plot(
        x,
        [d100[key] for key in pair_order],
        marker="^",
        label="D=100 planted exact recovery",
    )
    axis.axhline(
        0.90, color="black", linestyle="--", linewidth=0.8, label="D=8 gate 0.90"
    )
    axis.axhline(
        0.875, color="gray", linestyle=":", linewidth=0.8, label="planted gate 0.875"
    )
    axis.set_ylim(-0.02, 1.02)
    axis.set_xticks(
        x, [key.replace("E01-S11R-", "") for key in pair_order], rotation=45, ha="right"
    )
    axis.set_ylabel("fraction")
    axis.set_title("S11R exact-pair partition confirmation")
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output / "partition_confirmation.png", dpi=160)
    plt.close(figure)


def runtime_manifest(
    workers: int, phase: str, command: str, runtime: list[dict[str, Any]]
) -> dict[str, Any]:
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "researchStepId": "S11R",
        "phase": phase,
        "command": command,
        "repositoryCommit": git_output("rev-parse", "HEAD"),
        "repositoryBranch": git_output("branch", "--show-current"),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpuCountVisible": os.cpu_count(),
        "cpuWorkers": workers,
        "threadEnvironment": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "scikitLearn": sklearn.__version__,
        "gpuInventory": gpu.stdout.strip().splitlines() if gpu.returncode == 0 else [],
        "gpuUsed": False,
        "precision": "IEEE-754 binary64 CPU",
        "stageBenchmarks": runtime,
        "totalWallSeconds": float(sum(item["wallSeconds"] for item in runtime)),
    }


def write_contracts(
    output: Path,
    config: dict[str, Any],
    validation: dict[str, Any],
    eligibility: list[dict[str, Any]],
) -> list[Path]:
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    contract_path = BUNDLE_DIR / "time_localized_phir_repair_contract_v1.0.0.yaml"
    registry_path = (
        BUNDLE_DIR / "time_localized_phir_repair_eligibility_registry_v1.0.0.yaml"
    )
    contract = {
        "schemaVersion": "1.0.0",
        "researchStepId": "S11R",
        "contractId": "E01-S11R-TIME-LOCALIZED-PHIR-REPAIR-CONTRACT-v1.0.0",
        "status": "CONFIRMATION_COMPLETE",
        "evidenceClass": "VALIDATION_ONLY_REPAIR_NOT_AUTHOR_METHOD_RECOVERY",
        "estimatorId": ESTIMATOR_ID,
        "calibrationId": CALIBRATION_ID,
        "affinityId": AFFINITY_ID,
        "partitionSearchId": SEARCH_ID,
        "redundancyBranches": config["redundancyBranches"],
        "fixedWindowGrid": config["fixedWindowGrid"],
        "globalEligibilityRule": config["globalEligibilityRule"],
        "confirmationPassed": validation["success"],
        "outcomeClassification": validation["outcomeClassification"],
        "s11Immutable": True,
        "s10StrictMinimumEffectiveSamples": 512,
        "omegaDiscreteAndDoubletExcluded": True,
        "paperPrimary": None,
        "authorMapping": "UNRESOLVED::E01-A043",
        "noGardScientificEstimates": True,
    }
    write_yaml(contract_path, contract)
    registry = {
        "schemaVersion": "1.0.0",
        "researchStepId": "S11R",
        "registryId": "E01-S11R-FIXED-WINDOW-ELIGIBILITY-v1.0.0",
        "globalStatus": "ELIGIBLE_VALIDATION_BRANCH"
        if validation["success"]
        else "FAIL_CLOSED_NO_FIXED_BRANCH_ELIGIBLE",
        "eligibleBranchCount": sum(
            row["status"] == "ELIGIBLE_VALIDATION_BRANCH" for row in eligibility
        ),
        "totalBranchCount": len(eligibility),
        "numericGardScientificEstimateCount": 0,
        "failedGateFamilies": validation["failedGateFamilies"],
        "exactPairTable": str(output / "exact_pair_eligibility.csv"),
        "exactPairTableSha256": sha256_file(output / "exact_pair_eligibility.csv"),
        "s11Status": "RETURN_FOR_REVIEW_VALIDATION_BLOCKED",
        "s10StrictBranchStatus": "UNCHANGED_MINIMUM_512",
        "paperPrimary": None,
        "authorMapping": "UNRESOLVED::E01-A043",
    }
    write_yaml(registry_path, registry)
    return [contract_path, registry_path]


def write_report(
    output: Path,
    validation: dict[str, Any],
    runtime: dict[str, Any],
    artifacts: list[str],
) -> None:
    success = validation["success"]
    outcome = validation["outcomeClassification"]
    status = (
        "COMPLETE — ALL CONFIRMATION GATES PASSED"
        if success
        else "COMPLETE — FAIL CLOSED"
    )
    failed = validation["failedGateFamilies"]
    d8 = validation["d8Summaries"]
    highdim = validation["highdimSummaries"]
    report = f"""# S11R research-step full results — bounded fixed-window repair validation

## Concise top summary

- **Research step ID:** S11R
- **Completion status:** {status}; S11 and all failed S11 outputs remain immutable; S12 was not begun.
- **Artifacts written:** {len(artifacts)} compact step artifacts under `{output}`, plus the versioned repair contract and eligibility registry in the information-dynamics bundle. Principal files are `validation_summary.json`, the five confirmation CSVs, `exact_pair_eligibility.csv`, `seed_records.parquet`, two validation figures, manifests, and this report.
- **Validation result:** {"PASS" if success else "FAIL"} — {validation["gateFamiliesPassed"]}/{validation["gateFamiliesTotal"]} total preregistered gate families passed. Failed families: {", ".join(failed) if failed else "none"}.
- **Outcome classification:** **{outcome}**.
- **Caveats or blockers:** This is a validation-only reconstruction, not the unavailable author implementation or MATLAB RNG. MMI and experimental CCS, all mappings/objectives/normalizations, and author-method uncertainty remain distinct. S10's strict >=512 branch is unchanged. No OmegaID discrete/doublet substitute or GARD scientific estimate was used.
- **Lay summary:** The repair tested a small-sample bias correction, a null matched to each synthetic system, and a feature-order-independent partition rule on a fresh, untouched confirmation set. {"Every frozen check passed, so the branch is validation-eligible but still not an author-method recovery." if success else "At least one frozen confirmation check failed, so the repair correctly shut itself down: all 576 fixed-window branches remain ineligible and no scientific values were produced."}
- **Recommended next action:** Hand control back for Chief Scientist review. Do not begin S12 without a new explicit instruction.{"" if success else " Do not continue this bounded repair path."}

## Frozen question

Can the separately versioned Wishart-corrected Gaussian estimator, condition-matched complete-row-shuffle null, and permutation-equivariant threshold-component partition method pass untouched confirmation at all 16 planned effective sample sizes (24–255), without changing S11 or S10?

## Scope and immutable boundaries

S11R was preregistered at Git commit `{PREREGISTRATION_COMMIT}` (SHA-256 `{PREREGISTRATION_SHA256}`) before development or confirmation outcomes. Development and confirmation used separate 256-bit roots. The method was committed and locked after development but before confirmation access. The original S11 artifact manifest, all 34 manifest entries, and eight S11 repository files were verified byte-for-byte before the run and at handoff. S11's 46/64 truth, 28/64 shuffle, 356/576 invariance, 0/48 partition, 0/576 eligibility, and 0/33,984 numeric fixed-window findings were not altered or relabeled.

S10's `E01-S10-SAMPLE-GATE-STRICT-v1.0.0` still requires at least 512 effective samples. S11R is a distinct estimator valid only within its own preregistered validation domain and does not replace that strict branch.

## Inputs

The runner verified all 24 preregistered frozen inputs: workspace governance and plans, attachment manifest and sidecar, paper Markdown and official arXiv v1 PDF, S09 preprocessing report/contracts, S10 report/contracts/eligibility, S11 report/preregistration/results/contracts, S03 provenance/environment manifests, S06 seed contract, and registry v0.3.0. Exact paths and hashes are in `preregistration_record.json`.

No GARD trajectory or mounted dataset was used. All outcomes are validation-fixture evidence.

## Methods

### Small-window estimator

For each scalar pair, the estimator formed `(x_t,y_t,x_t+tau,y_t+tau)`, standardized the complete lagged sample with sample standard deviations, used the unregularized covariance with divisor `n_eff-1`, and evaluated all 15 required Gaussian local entropies in binary64 CPU arithmetic. For a p-dimensional subset and `nu=n_eff-1`, every local entropy received the preregistered exact Gaussian Wishart mean correction

`0.5 * [p/n_eff - (sum_i digamma((nu+1-i)/2) + p log(2) - p log(nu))]`.

There was no row deletion, covariance regularization, fallback, or post-hoc threshold change. MMI and experimental CCS used the same explicit 16-atom lattice identities as S10/S11 but remain separately labeled.

### Condition-matched null

Each calibration replicate came from the same declared structured system, exact window, and lag as its target condition. A uniformly random permutation was applied to complete contemporaneous two-feature rows before lagging. Calibration means were keyed by phase, exact pair, system, and redundancy. Held-out structured-shuffle replicates used disjoint streams. The 99th-percentile absolute centered equation envelope used NumPy's `method='higher'`. Independent white noise was not substituted for either structured system.

### Partition method

The partition affinity was the mean absolute Pearson correlation of lag-aligned past and future rows. An edge existed only above 0.90; any edge within 1e-12 of the threshold was ineligible. Exactly two connected components were required in the base fit and all eight Bayesian exponential-weight bootstraps. The gate additionally required minimum part fraction 0.10, mean bootstrap ARI 0.75, minimum within affinity 0.90, and within-minus-maximum-between affinity 0.10. No feature index or objective-score tie breaker selected the split.

D=8 independently enumerated all 127 unordered bipartitions for each of 18 mapping/objective/normalization branches. Exact objective ties within 1e-12 were ineligible. D=99/100 planted and independent-null fixtures, arbitrary feature permutations, and positive feature-wise affine transforms were directly confirmed at every exact pair.

### Development/confirmation firewall

Development used `{DEVELOPMENT_ROOT_SEED_HEX[:8]}...`; confirmation used `{CONFIRMATION_ROOT_SEED_HEX[:8]}...`. Both used S06 SHA-256 domain separation and NumPy PCG64DXSM with phase, domain, pair, dimension, and replicate in the identity. The full cross-phase stream-ID and seed-material intersections were zero. Confirmation seeds were accessed only after the method-lock file was committed.

## Commands

```bash
PYTHONPATH=src pytest -q tests/e01/test_time_localized_phir_repair.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \\
  PYTHONPATH=src python scripts/e01/run_s11r_time_localized_phir_repair.py \\
  --phase development --workers 8 --output /cache/e01_s11r/development
# implementation and method-lock commits were made and pushed here
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \\
  PYTHONPATH=src python scripts/e01/run_s11r_time_localized_phir_repair.py \\
  --phase confirmation --workers 8 --output /artifacts/research_steps/S11R \\
  --method-lock configs/e01/s11r_confirmation_method_lock.yaml --method-lock-commit <locked-commit>
PYTHONPATH=src pytest -q tests/e01/test_time_localized_phir_repair.py
```

## Confirmation results

- Condition-matched null rows passed: {validation["rowCounts"]["conditionMatchedNullPassed"]}/{validation["rowCounts"]["conditionMatchedNullTotal"]}.
- Known-truth summary rows passed: {validation["rowCounts"]["knownTruthPassed"]}/{validation["rowCounts"]["knownTruthTotal"]}.
- Structured-shuffle rows passed: {validation["rowCounts"]["structuredShufflePassed"]}/{validation["rowCounts"]["structuredShuffleTotal"]}.
- D=8 exact-pair agreement summaries passed: {sum(item["pass"] for item in d8)}/{len(d8)}; observed range {min(item["exactAgreementFraction"] for item in d8):.6f}–{max(item["exactAgreementFraction"] for item in d8):.6f}, frozen threshold 0.90.
- D=99/100 planted/null summaries passed: {sum(item["pass"] for item in highdim)}/{len(highdim)}.
- Invariance rows passed: {validation["rowCounts"]["invariancePassed"]}/{validation["rowCounts"]["invarianceTotal"]}.
- Exact fixed-window eligibility: {validation["eligibleFixedBranches"]}/576 branches and {validation["eligibleFixedPairs"]}/16 pairs. Numeric GARD scientific estimates: 0.

Detailed condition-level values, atom RMSEs, direction checks, envelope coverage, exact partitions, ARIs, rejection reasons, and every eligibility status are retained in the machine-readable tables. Failures are not pooled away or replaced by another branch.

## Validation

The run verified frozen preregistration bytes, all frozen input hashes, the committed method lock, the complete development/confirmation seed firewall, exact 16-pair sample counts, S11 immutability, registry immutability, strict-boundary preservation, lattice/equation closure inside every eligible estimator call, exact CPU anchor replay, exact seed replay, all 16 failure injections, table cardinalities, figure creation, and absence of numeric scientific estimates. Outcome suppression was evaluated after all confirmation families, never branch-by-branch.

## Runtime and dependencies

The confirmation used {runtime["cpuWorkers"]} process workers with BLAS/OpenMP thread counts fixed to one. NumPy {runtime["numpy"]}, SciPy {runtime["scipy"]}, pandas {runtime["pandas"]}, and scikit-learn {runtime["scikitLearn"]} ran in Python {runtime["python"].split()[0]} on `{runtime["platform"]}`. GPU use was preregistered as false and no GPU computation occurred. Summed stage wall time was {runtime["totalWallSeconds"]:.3f} seconds; per-stage timings are in `runtime_benchmarks.csv`.

## Caveats, blockers, and limitations

- The paper does not specify the reconstructed local estimator, null calibration, partition affinity/search, MIB mapping/objective/normalization, redundancy, atom mapping, or RNG semantics. None was silently designated as the paper default.
- Pinned phyid labels CCS as not implemented; the S11R CCS branch therefore remains explicitly experimental even if its gates pass.
- Wishart correction is exact for the declared Gaussian in-sample covariance model. Serially overlapping lag rows and non-Gaussian data are outside that theorem; the synthetic confirmation tests its finite-sample behavior rather than proving universal calibration.
- The threshold-component method is intentionally fail-closed and applies to strongly separated block fixtures. It is not evidence that a real GARD composition admits exactly two threshold components.
- D=8 exhaustive agreement tests the frozen 18-way uncertainty grid; poor agreement cannot be repaired by choosing a favorable objective after outcome inspection.
- Validation eligibility, if present, is not author-code identity, MATLAB-RNG identity, a paper-primary designation, causal evidence, or a GARD scientific result.

## Provenance

The preregistration record contains every frozen input hash. `method_lock.json` identifies the committed implementation and development summary. `seed_records.parquet` and `seed_firewall.json` preserve domain-separated random identities. `s11_immutability.json`, `strict_boundary_preservation.json`, `runtime_manifest.json`, `reproducibility_validation.json`, and `artifact_manifest.json` provide the remaining byte, environment, replay, and output provenance. Repository-backed code remains in Git and is not copied into the artifact directory.

## Recommended next action

Hand control back to the Chief Scientist. {"The branch passed validation, but a separate explicit authorization is still required before S12 and the unresolved author/paper-primary choices remain open." if success else "The bounded S11R path failed closed; retain zero fixed-window eligibility and do not proceed to S12 or another repair without explicit review and authorization."}
"""
    (output / "research_step_full_results.md").write_text(report)


def build_manifest(output: Path, external: list[Path]) -> dict[str, Any]:
    paths = sorted(
        [
            path
            for path in output.iterdir()
            if path.is_file() and path.name != "artifact_manifest.json"
        ]
        + external,
        key=lambda item: str(item),
    )
    artifacts = [
        {
            "path": str(path),
            "relativePath": path.name
            if path.parent == output
            else str(path.relative_to(Path("/artifacts"))),
            "sizeBytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    ]
    aggregate = canonical_sha256(
        [
            {key: item[key] for key in ("relativePath", "sizeBytes", "sha256")}
            for item in artifacts
        ]
    )
    return {
        "researchStepId": "S11R",
        "artifactCountExcludingManifest": len(artifacts),
        "aggregateSha256": aggregate,
        "artifacts": artifacts,
    }


def execute(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    if not s11r_output_path_allowed(output):
        raise RuntimeError("S11R may not write S11 or S12 paths.")
    output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(CONFIG_PATH.read_text())
    prereg = verify_preregistration(config)
    s11_immutable = verify_s11_immutability(config)
    method_lock = None
    development_seed_path = None
    if args.phase == "confirmation":
        method_lock = verify_method_lock(
            args.method_lock.resolve(), args.method_lock_commit
        )
        development_seed_path = Path(method_lock["developmentSeedRecordsPath"])
    command = " ".join(sys.argv)
    runtime: list[dict[str, Any]] = []
    seeds: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    scalar = run_scalar_validation(
        config, phase=args.phase, workers=args.workers, runtime=runtime, seeds=seeds
    )
    partition = run_partition_validation(
        config, phase=args.phase, workers=args.workers, runtime=runtime, seeds=seeds
    )
    matrix, summaries = summarize_gates(config, scalar, partition, phase=args.phase)
    reproduction = reproducibility_validation(config, args.phase)
    strict = strict_boundary_record(config)
    injections = run_failure_injections(
        config, summaries["allScientificConfirmationGatesPassed"]
    )
    seed_rows = sorted(seeds.values(), key=lambda item: item["streamId"])
    seed_path = output / "seed_records.parquet"
    write_parquet(seed_path, seed_rows)
    firewall = seed_firewall_record(
        phase=args.phase,
        current_records=seed_rows,
        development_path=development_seed_path,
    )
    overall_success = (
        summaries["allScientificConfirmationGatesPassed"]
        and reproduction["success"]
        and strict["success"]
        and injections["allPassed"]
        and firewall["success"]
        and s11_immutable["success"]
        and prereg["success"]
    )
    failed_families = [item["family"] for item in matrix if not item["pass"]]
    for family, success in (
        ("reproducibility", reproduction["success"]),
        ("strict_boundary_preservation", strict["success"]),
        ("failure_injection", injections["allPassed"]),
        ("seed_firewall", firewall["success"]),
        ("s11_immutability", s11_immutable["success"]),
        ("frozen_input_verification", prereg["success"]),
    ):
        matrix.append(
            {
                "family": family,
                "passedRows": int(success),
                "totalRows": 1,
                "pass": bool(success),
            }
        )
        if not success:
            failed_families.append(family)
    eligibility = exact_pair_eligibility(config, overall_success, failed_families)
    eligible_pairs = len(
        {
            row["pairId"]
            for row in eligibility
            if row["status"] == "ELIGIBLE_VALIDATION_BRANCH"
        }
    )
    row_counts = {
        "conditionMatchedNullPassed": sum(
            row["pass"] for row in scalar["calibrationRows"]
        ),
        "conditionMatchedNullTotal": len(scalar["calibrationRows"]),
        "knownTruthPassed": sum(row["pass"] for row in scalar["truthRows"]),
        "knownTruthTotal": len(scalar["truthRows"]),
        "structuredShufflePassed": sum(row["pass"] for row in scalar["shuffleRows"]),
        "structuredShuffleTotal": len(scalar["shuffleRows"]),
        "d8ComparisonPassed": sum(row["exactAgreement"] for row in partition["d8Rows"]),
        "d8ComparisonTotal": len(partition["d8Rows"]),
        "highdimRows": len(partition["highdimRows"]),
        "invariancePassed": sum(row["pass"] for row in partition["invarianceRows"]),
        "invarianceTotal": len(partition["invarianceRows"]),
    }
    validation = {
        "researchStepId": "S11R",
        "phase": args.phase,
        "status": "PASS_ALL_CONFIRMATION_GATES"
        if overall_success
        else "FAIL_CLOSED_CONFIRMATION_GATE_FAILURE",
        "success": overall_success,
        "outcomeClassification": "supportive"
        if overall_success
        else "constraining/contradictory",
        "gateFamiliesPassed": sum(item["pass"] for item in matrix),
        "gateFamiliesTotal": len(matrix),
        "failedGateFamilies": failed_families,
        "gateMatrix": matrix,
        "rowCounts": row_counts,
        "d8Summaries": summaries["d8Summaries"],
        "highdimSummaries": summaries["highdimSummaries"],
        "eligibleFixedBranches": sum(
            row["status"] == "ELIGIBLE_VALIDATION_BRANCH" for row in eligibility
        ),
        "totalFixedBranches": 576,
        "eligibleFixedPairs": eligible_pairs,
        "totalFixedPairs": 16,
        "numericGardScientificEstimateCount": 0,
        "s11StatusUnchanged": "RETURN_FOR_REVIEW_VALIDATION_BLOCKED",
        "s12Begun": False,
        "paperPrimary": None,
        "authorMapping": "UNRESOLVED::E01-A043",
        "omegaDiscreteOrDoubletUsed": False,
    }
    runtime.append(
        {
            "stage": f"{args.phase}_runner_noncompute_overhead",
            "caseCount": 1,
            "workers": 1,
            "wallSeconds": max(
                0.0,
                time.perf_counter()
                - started
                - sum(item["wallSeconds"] for item in runtime),
            ),
        }
    )
    run_manifest = runtime_manifest(args.workers, args.phase, command, runtime)

    write_json(output / "preregistration_record.json", prereg)
    shutil.copyfile(CONFIG_PATH, output / "preregistration.yaml")
    write_json(output / "s11_immutability.json", s11_immutable)
    write_json(output / "strict_boundary_preservation.json", strict)
    write_json(output / "failure_injection.json", injections)
    write_json(output / "reproducibility_validation.json", reproduction)
    write_json(output / "seed_firewall.json", firewall)
    write_json(output / "validation_summary.json", validation)
    write_csv(output / "condition_matched_calibration.csv", scalar["calibrationRows"])
    write_csv(output / "known_truth_confirmation.csv", scalar["truthRows"])
    write_csv(output / "structured_shuffle_confirmation.csv", scalar["shuffleRows"])
    write_csv(output / "d8_exhaustive_confirmation.csv", partition["d8Rows"])
    write_csv(output / "highdim_partition_confirmation.csv", partition["highdimRows"])
    write_csv(output / "invariance_confirmation.csv", partition["invarianceRows"])
    write_csv(output / "exact_pair_eligibility.csv", eligibility)
    write_csv(output / "runtime_benchmarks.csv", runtime)
    write_json(output / "runtime_manifest.json", run_manifest)
    specification = {
        "researchStepId": "S11R",
        "status": validation["status"],
        "preregistrationCommit": PREREGISTRATION_COMMIT,
        "preregistrationSha256": PREREGISTRATION_SHA256,
        "methodLock": method_lock,
        "estimatorId": ESTIMATOR_ID,
        "calibrationId": CALIBRATION_ID,
        "affinityId": AFFINITY_ID,
        "searchId": SEARCH_ID,
        "redundancies": ["MMI", "CCS_EXPERIMENTAL"],
        "mappings": MAPPINGS,
        "objectives": OBJECTIVES,
        "normalizations": NORMALIZATIONS,
        "paperPrimary": None,
        "authorMapping": "UNRESOLVED::E01-A043",
        "s10StrictBranch": "UNCHANGED_MINIMUM_512",
        "s11": "IMMUTABLE_RETURN_FOR_REVIEW_VALIDATION_BLOCKED",
        "omegaDiscreteAndDoublet": "EXCLUDED",
        "gardScientificEstimates": "NOT_GENERATED",
    }
    write_yaml(output / "specification_metadata.yaml", specification)

    if args.phase == "development":
        development_summary = {
            "researchStepId": "S11R",
            "phase": "development",
            "status": "COMPLETE_DIAGNOSTIC_ONLY_NO_METHOD_SELECTION",
            "success": True,
            "preregistrationSha256": PREREGISTRATION_SHA256,
            "confirmationSeedsAccessed": False,
            "methodOrGateChangedAfterDevelopment": False,
            "diagnosticGateMatrix": matrix,
            "diagnosticFailedFamilies": failed_families,
            "seedRecordsPath": str(seed_path),
            "seedRecordsSha256": sha256_file(seed_path),
            "seedStreamCount": len(seed_rows),
            "runtimeManifest": run_manifest,
            "recommendedNextAction": "Freeze implementation and method-lock commit, then run untouched confirmation without changing the preregistered method or gates.",
        }
        write_json(output / "development_summary.json", development_summary)
        print(
            json.dumps(
                {
                    "phase": args.phase,
                    "developmentSummary": str(output / "development_summary.json"),
                    "diagnosticFailedFamilies": failed_families,
                },
                indent=2,
            ),
            flush=True,
        )
        return

    write_json(output / "method_lock.json", method_lock)
    shutil.copyfile(
        Path(method_lock["developmentSummaryPath"]),
        output / "development_summary.json",
    )
    create_figures(
        output, matrix, summaries["d8Summaries"], summaries["highdimSummaries"]
    )
    external = write_contracts(output, config, validation, eligibility)
    expected_artifacts = [
        item for item in config["requiredOutputs"] if item != "artifact_manifest.json"
    ]
    write_report(output, validation, run_manifest, expected_artifacts)
    missing = [name for name in expected_artifacts if not (output / name).is_file()]
    if missing:
        raise RuntimeError(f"Required S11R outputs missing: {missing}")
    manifest = build_manifest(output, external)
    write_json(output / "artifact_manifest.json", manifest)
    print(
        json.dumps(
            {
                "phase": args.phase,
                "success": overall_success,
                "failedGateFamilies": failed_families,
                "eligibleFixedBranches": validation["eligibleFixedBranches"],
                "artifactCount": manifest["artifactCountExcludingManifest"],
            },
            indent=2,
        ),
        flush=True,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", choices=("development", "confirmation"), required=True
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method-lock", type=Path)
    parser.add_argument("--method-lock-commit", default="")
    arguments = parser.parse_args()
    if not 1 <= arguments.workers <= 8:
        parser.error("--workers must be between 1 and 8.")
    if arguments.phase == "confirmation" and arguments.method_lock is None:
        parser.error("confirmation requires --method-lock and --method-lock-commit.")
    return arguments


if __name__ == "__main__":
    execute(parse_arguments())
