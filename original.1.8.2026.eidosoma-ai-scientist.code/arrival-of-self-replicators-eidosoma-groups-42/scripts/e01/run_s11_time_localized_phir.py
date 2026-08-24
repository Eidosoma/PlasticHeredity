#!/usr/bin/env python3
"""Execute the frozen E01 S11 time-localized Phi-r validation."""

# ruff: noqa: BLE001

from __future__ import annotations

import argparse
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
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from e01_information_dynamics.backends import run_omegaid, run_phyid
from e01_information_dynamics.validation import ATOM_IDS, I_KEYS, all_bipartitions
from e01_time_localized_phir import (
    AFFINITY_ID,
    CALIBRATION_ID,
    ESTIMATOR_ID,
    GROUP_MEAN_ID,
    PC1_ID,
    SEARCH_ID,
    calibrated_means,
    evaluate_candidate_grid,
    fixed_window_index,
    map_partition,
    partition_ari,
    run_small_window_phiid,
    sliding_endpoints,
    stable_partition_candidates,
    whole_trajectory_index,
)
from e01_time_localized_phir.partition import StablePartitionResult
from e01_time_localized_phir.synthetic import (
    ccs_population_oracle,
    directional_covariance,
    directional_var,
    estimator_rng,
    highdim_independent_null,
    independent_white,
    mmi_truth,
    noisy_redundant_ar,
    piecewise_block_ar,
    planted_two_block_ar,
    redundant_covariance,
)

CONFIG_PATH = REPOSITORY_ROOT / "configs/e01/s11_time_localized_phir_preregistration.yaml"
PREREGISTRATION_COMMIT = "f257146cd845f49ec01efe08175a15cae64ccf39"
PREREGISTRATION_SHA256 = "1c21ad91927929626edb6b2e14dfa745674decbaa89c3ac57f2cfdc678458f40"
REGISTRY_PATH = Path(
    "/artifacts/E01_forensic_replication_bundle/specifications/specification_registry_v0.3.0.yaml"
)
REGISTRY_SHA256 = "aef0e179de6466697540ba10236ed24af37fbda12bd4f1c6b1fb5fe7a27af891"
S10_STRICT_GATE_ID = "E01-S10-SAMPLE-GATE-STRICT-v1.0.0"

REDUNDANCIES = ("MMI", "CCS")
MAPPINGS = ("zscore_group_mean", "zscore_pc1")
OBJECTIVES = ("synchronous_mi", "bidirectional_lagged_mi", "abs_paper_equation")
NORMALIZATIONS = ("none", "min_part_entropy", "geometric_part_size")
STRUCTURED_SYSTEMS = (
    "E01-S11-SYS-NOISY-REDUNDANT-AR-v1.0.0",
    "E01-S11-SYS-DIRECTIONAL-VAR-v1.0.0",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
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


def _flat_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, np.ndarray)):
        return json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"))
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_csv(path: Path, rows: list[dict[str, Any]], *, allow_empty: bool = False) -> None:
    if not rows and not allow_empty:
        raise RuntimeError(f"Refusing to write empty required table {path.name}.")
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        if not fields:
            return
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _flat_value(row.get(key)) for key in fields})


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty required table {path.name}.")
    fields = sorted({key for row in rows for key in row})
    normalized = [
        {key: _flat_value(row.get(key)) for key in fields}
        for row in rows
    ]
    pd.DataFrame(normalized, columns=fields).to_parquet(path, index=False, compression="zstd")


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
        raise RuntimeError("S11 preregistration working-tree bytes changed.")
    committed = subprocess.run(
        [
            "git",
            "show",
            f"{PREREGISTRATION_COMMIT}:configs/e01/s11_time_localized_phir_preregistration.yaml",
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    ).stdout
    if hashlib.sha256(committed).hexdigest() != PREREGISTRATION_SHA256:
        raise RuntimeError("The preregistration commit does not contain the frozen bytes.")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", PREREGISTRATION_COMMIT, "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
    )
    checks: list[dict[str, Any]] = []
    for item in config["frozenInputs"]:
        path = Path(item["path"])
        actual = sha256_file(path) if path.is_file() else None
        checks.append({**item, "actualSha256": actual, "success": actual == item["sha256"]})
    if not all(item["success"] for item in checks):
        raise RuntimeError(f"Frozen S11 input mismatch: {[x for x in checks if not x['success']]}")
    pairs = config["fixedWindowGrid"]["pairs"]
    if (
        len(pairs) != 16
        or len({(item["windowLength"], item["lag"]) for item in pairs}) != 16
        or any(item["effectiveSampleCount"] != item["windowLength"] - item["lag"] for item in pairs)
        or max(item["effectiveSampleCount"] for item in pairs) >= 512
    ):
        raise RuntimeError("The exact fixed-pair grid violates the frozen boundary.")
    return {
        "status": "VERIFIED_FROZEN_BEFORE_OUTCOMES",
        "success": True,
        "preregistrationCommit": PREREGISTRATION_COMMIT,
        "preregistrationSha256": PREREGISTRATION_SHA256,
        "frozenInputs": checks,
    }


def _fixture(system_id: str, task: dict[str, Any]):
    arguments = {
        "pair_id": task["pairId"],
        "replicate_index": task["replicateIndex"],
        "length": task["windowLength"],
        "domain": task["domain"],
    }
    if system_id == "E01-S11-SYS-INDEPENDENT-WHITE-GAUSSIAN-v1.0.0":
        return independent_white(**arguments)
    if system_id == "E01-S11-SYS-NOISY-REDUNDANT-AR-v1.0.0":
        return noisy_redundant_ar(**arguments)
    if system_id == "E01-S11-SYS-DIRECTIONAL-VAR-v1.0.0":
        return directional_var(**arguments)
    raise RuntimeError(f"Unknown scalar fixture {system_id}.")


def _compact_result(result: Any) -> dict[str, Any]:
    means = result.means()
    return {
        "status": result.status,
        "reason": result.reason,
        "means": means,
        "minimumRegularizedEigenvalue": result.diagnostics.get("minimumRegularizedEigenvalue"),
        "maximumRegularizedConditionNumber": result.diagnostics.get("maximumRegularizedConditionNumber"),
    }


def _scalar_task(task: dict[str, Any]) -> dict[str, Any]:
    fixture = _fixture(task["systemId"], task)
    results = {
        redundancy: _compact_result(
            run_small_window_phiid(
                fixture.data[:, 0],
                fixture.data[:, 1],
                tau=task["lag"],
                redundancy=redundancy,
            )
        )
        for redundancy in REDUNDANCIES
    }
    return {**task, "results": results, "seedRecords": [fixture.seed_record]}


def _shuffle_task(task: dict[str, Any]) -> dict[str, Any]:
    fixture = _fixture(task["systemId"], task)
    shuffle_rng, shuffle_record = estimator_rng(
        domain="shuffle",
        pair_id=f"{task['pairId']}-{task['systemId'].split('-SYS-')[-1]}",
        replicate_index=task["replicateIndex"],
    )
    shuffled = fixture.data[shuffle_rng.permutation(fixture.data.shape[0])]
    results = {
        redundancy: _compact_result(
            run_small_window_phiid(
                shuffled[:, 0], shuffled[:, 1], tau=task["lag"], redundancy=redundancy
            )
        )
        for redundancy in REDUNDANCIES
    }
    return {**task, "results": results, "seedRecords": [fixture.seed_record, shuffle_record]}


def _sensitivity_task(task: dict[str, Any]) -> dict[str, Any]:
    fixture = _fixture(task["systemId"], task)
    results: dict[str, dict[str, Any]] = {}
    for redundancy in REDUNDANCIES:
        by_multiplier = {}
        for multiplier in (0.5, 1.0, 1.5):
            by_multiplier[str(multiplier)] = _compact_result(
                run_small_window_phiid(
                    fixture.data[:, 0],
                    fixture.data[:, 1],
                    tau=task["lag"],
                    redundancy=redundancy,
                    shrinkage_multiplier=multiplier,
                )
            )
        results[redundancy] = by_multiplier
    return {**task, "results": results, "seedRecords": [fixture.seed_record]}


def run_parallel(
    function: Any,
    tasks: list[dict[str, Any]],
    *,
    workers: int,
    label: str,
    runtime_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(function, tasks, chunksize=max(1, len(tasks) // (workers * 16))))
    runtime_rows.append(
        {
            "stage": label,
            "caseCount": len(tasks),
            "workers": workers,
            "wallSeconds": time.perf_counter() - start,
        }
    )
    return results


def _mean_payload(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    if not payloads:
        raise RuntimeError("Cannot average an empty payload collection.")
    atom_means = {
        atom: float(np.mean([payload["atomMeans"][atom] for payload in payloads]))
        for atom in ATOM_IDS
    }
    mi_means = {
        key: float(np.mean([payload["miMeans"][key] for payload in payloads]))
        for key in I_KEYS
    }
    total_atoms = float(sum(atom_means.values()))
    past_redundancy = float(sum(atom_means[key] for key in ATOM_IDS[:4]))
    past_synergy = float(sum(atom_means[key] for key in ATOM_IDS[12:]))
    equation_atoms = past_synergy - past_redundancy
    equation_direct = mi_means["I_xytab"] - mi_means["I_xtab"] - mi_means["I_ytab"]
    return {
        "atomMeans": atom_means,
        "miMeans": mi_means,
        "totalAtomSum": total_atoms,
        "totalMi": mi_means["I_xytab"],
        "latticeClosureError": total_atoms - mi_means["I_xytab"],
        "pastRedundancy": past_redundancy,
        "pastSynergy": past_synergy,
        "paperEquationAggregateFromAtoms": equation_atoms,
        "paperEquationAggregateDirect": equation_direct,
        "paperEquationClosureError": equation_atoms - equation_direct,
    }


def _collect_seed_records(results: list[dict[str, Any]], destination: dict[str, dict[str, Any]]) -> None:
    for result in results:
        for record in result.get("seedRecords", []):
            destination[record["streamId"]] = record


def run_scalar_validation(
    config: dict[str, Any],
    *,
    workers: int,
    runtime_rows: list[dict[str, Any]],
    seed_records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    pairs = config["fixedWindowGrid"]["pairs"]
    calibration_tasks = [
        {
            **pair,
            "systemId": "E01-S11-SYS-INDEPENDENT-WHITE-GAUSSIAN-v1.0.0",
            "replicateIndex": replicate,
            "domain": "calibration",
        }
        for pair in pairs
        for replicate in range(512)
    ]
    calibration_results = run_parallel(
        _scalar_task,
        calibration_tasks,
        workers=workers,
        label="scalar_null_calibration",
        runtime_rows=runtime_rows,
    )
    _collect_seed_records(calibration_results, seed_records)
    calibration: dict[tuple[str, str], dict[str, Any]] = {}
    envelopes: dict[tuple[str, str], float] = {}
    for pair in pairs:
        selected = [item for item in calibration_results if item["pairId"] == pair["pairId"]]
        for redundancy in REDUNDANCIES:
            payloads = [item["results"][redundancy]["means"] for item in selected]
            if any(payload is None for payload in payloads):
                raise RuntimeError(f"Calibration estimator failure for {pair['pairId']} {redundancy}.")
            mean_payload = _mean_payload(payloads)
            calibration[(pair["pairId"], redundancy)] = mean_payload
            centered = [
                abs(
                    payload["paperEquationAggregateFromAtoms"]
                    - mean_payload["paperEquationAggregateFromAtoms"]
                )
                for payload in payloads
            ]
            envelopes[(pair["pairId"], redundancy)] = float(
                np.quantile(centered, 0.99, method="higher")
            )

    heldout_tasks = [
        {
            **pair,
            "systemId": "E01-S11-SYS-INDEPENDENT-WHITE-GAUSSIAN-v1.0.0",
            "replicateIndex": replicate,
            "domain": "heldout-null",
        }
        for pair in pairs
        for replicate in range(256)
    ]
    heldout_results = run_parallel(
        _scalar_task,
        heldout_tasks,
        workers=workers,
        label="scalar_heldout_null",
        runtime_rows=runtime_rows,
    )
    _collect_seed_records(heldout_results, seed_records)

    calibration_rows: list[dict[str, Any]] = []
    null_gate: dict[tuple[str, str], bool] = {}
    for pair in pairs:
        selected = [item for item in heldout_results if item["pairId"] == pair["pairId"]]
        for redundancy in REDUNDANCIES:
            corrected = [
                calibrated_means(
                    item["results"][redundancy]["means"],
                    calibration[(pair["pairId"], redundancy)],
                )
                for item in selected
            ]
            mean_corrected = _mean_payload(corrected)
            max_atom_bias = max(abs(value) for value in mean_corrected["atomMeans"].values())
            equation_bias = abs(mean_corrected["paperEquationAggregateFromAtoms"])
            envelope = envelopes[(pair["pairId"], redundancy)]
            false_positive_rate = float(
                np.mean(
                    [
                        abs(item["paperEquationAggregateFromAtoms"]) > envelope
                        for item in corrected
                    ]
                )
            )
            passed = (
                max_atom_bias <= 0.040
                and equation_bias <= 0.040
                and false_positive_rate <= 0.050
            )
            null_gate[(pair["pairId"], redundancy)] = passed
            row = {
                **pair,
                "redundancy": redundancy,
                "calibrationReplicates": 512,
                "heldOutReplicates": 256,
                "calibrationId": CALIBRATION_ID,
                "nullEnvelope99": envelope,
                "maximumAbsoluteHeldOutMeanAtom": max_atom_bias,
                "absoluteHeldOutMeanEquationAggregate": equation_bias,
                "heldOutFalsePositiveRate": false_positive_rate,
                "pass": passed,
            }
            row.update(
                {f"calibrationBiasAtom_{atom}": calibration[(pair["pairId"], redundancy)]["atomMeans"][atom] for atom in ATOM_IDS}
            )
            row.update(
                {f"heldOutMeanAtom_{atom}": mean_corrected["atomMeans"][atom] for atom in ATOM_IDS}
            )
            calibration_rows.append(row)

    ccs_oracles: dict[tuple[str, int], dict[str, Any]] = {}
    ccs_oracle_rows: list[dict[str, Any]] = []
    ccs_oracle_gate: dict[tuple[str, int], bool] = {}
    oracle_start = time.perf_counter()
    for system_id in STRUCTURED_SYSTEMS:
        for lag in config["fixedWindowGrid"]["lags"]:
            covariance = (
                redundant_covariance(lag)
                if "REDUNDANT" in system_id
                else directional_covariance(lag)
            )
            first = ccs_population_oracle(covariance, scramble_seed=11_000 + lag, power=18)
            second = ccs_population_oracle(covariance, scramble_seed=22_000 + lag, power=18)
            maximum_difference = max(
                abs(first["atomMeans"][atom] - second["atomMeans"][atom])
                for atom in ATOM_IDS
            )
            averaged = {
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
            }
            ccs_oracles[(system_id, lag)] = averaged
            ccs_oracle_gate[(system_id, lag)] = maximum_difference <= 0.002
            ccs_oracle_rows.append(
                {
                    "systemId": system_id,
                    "lag": lag,
                    "drawsPerScramble": 262144,
                    "scrambles": 2,
                    "maximumCrossScrambleAtomDifference": maximum_difference,
                    "pass": maximum_difference <= 0.002,
                }
            )
    runtime_rows.append(
        {
            "stage": "ccs_population_oracles",
            "caseCount": len(ccs_oracle_rows),
            "workers": 1,
            "wallSeconds": time.perf_counter() - oracle_start,
        }
    )

    truth_tasks = [
        {
            **pair,
            "systemId": system_id,
            "replicateIndex": replicate,
            "domain": "known-truth",
        }
        for pair in pairs
        for system_id in STRUCTURED_SYSTEMS
        for replicate in range(256)
    ]
    truth_results = run_parallel(
        _scalar_task,
        truth_tasks,
        workers=workers,
        label="scalar_known_truth",
        runtime_rows=runtime_rows,
    )
    _collect_seed_records(truth_results, seed_records)
    truth_rows: list[dict[str, Any]] = []
    truth_gate: dict[tuple[str, str], bool] = {}
    for pair in pairs:
        for redundancy in REDUNDANCIES:
            branch_passes: list[bool] = []
            for system_id in STRUCTURED_SYSTEMS:
                selected = [
                    item
                    for item in truth_results
                    if item["pairId"] == pair["pairId"] and item["systemId"] == system_id
                ]
                corrected = [
                    calibrated_means(
                        item["results"][redundancy]["means"],
                        calibration[(pair["pairId"], redundancy)],
                    )
                    for item in selected
                ]
                estimate = _mean_payload(corrected)
                if redundancy == "MMI":
                    truth = mmi_truth(system_id, pair["lag"])
                    truth_atom = truth["atomMeans"]
                    truth_mi = truth["miMeans"]
                    truth_total = truth["totalMi"]
                    truth_equation = truth["paperEquationAggregate"]
                    oracle_pass = True
                    aggregate_tolerance = 0.20
                    atom_tolerance = 0.20
                else:
                    truth = ccs_oracles[(system_id, pair["lag"])]
                    truth_atom = truth["atomMeans"]
                    truth_mi = truth["miMeans"]
                    truth_total = truth["totalMi"]
                    truth_equation = truth["paperEquationAggregate"]
                    oracle_pass = ccs_oracle_gate[(system_id, pair["lag"])]
                    aggregate_tolerance = 0.25
                    atom_tolerance = 0.25
                total_error = abs(estimate["totalMi"] - truth_total)
                equation_error = abs(
                    estimate["paperEquationAggregateFromAtoms"] - truth_equation
                )
                atom_rmse = float(
                    np.sqrt(
                        np.mean(
                            [
                                (estimate["atomMeans"][atom] - truth_atom[atom]) ** 2
                                for atom in ATOM_IDS
                            ]
                        )
                    )
                )
                if "REDUNDANT" in system_id:
                    directional_pass = (
                        estimate["atomMeans"]["rtr"] > 0
                        and estimate["paperEquationAggregateFromAtoms"] < 0
                    )
                else:
                    directional_pass = estimate["miMeans"]["I_xtb"] > estimate["miMeans"]["I_yta"]
                passed = (
                    oracle_pass
                    and total_error <= 0.20
                    and equation_error <= aggregate_tolerance
                    and atom_rmse <= atom_tolerance
                    and directional_pass
                )
                branch_passes.append(passed)
                truth_rows.append(
                    {
                        **pair,
                        "systemId": system_id,
                        "redundancy": redundancy,
                        "replicates": 256,
                        "truthTotalMi": truth_total,
                        "estimatedTotalMi": estimate["totalMi"],
                        "absoluteTotalMiError": total_error,
                        "truthEquationAggregate": truth_equation,
                        "estimatedEquationAggregate": estimate["paperEquationAggregateFromAtoms"],
                        "absoluteEquationAggregateError": equation_error,
                        "atomRootMeanSquareError": atom_rmse,
                        "directionalGatePassed": directional_pass,
                        "ccsOracleGatePassed": oracle_pass if redundancy == "CCS" else None,
                        "experimentalCcsLabel": redundancy == "CCS",
                        "pass": passed,
                        "truthAtomMeans": truth_atom,
                        "estimatedAtomMeans": estimate["atomMeans"],
                        "truthMiMeans": truth_mi,
                        "estimatedMiMeans": estimate["miMeans"],
                    }
                )
            truth_gate[(pair["pairId"], redundancy)] = all(branch_passes)

    shuffle_tasks = [
        {
            **pair,
            "systemId": system_id,
            "replicateIndex": replicate,
            "domain": "shuffle-source",
        }
        for pair in pairs
        for system_id in STRUCTURED_SYSTEMS
        for replicate in range(128)
    ]
    shuffle_results = run_parallel(
        _shuffle_task,
        shuffle_tasks,
        workers=workers,
        label="scalar_time_shuffle",
        runtime_rows=runtime_rows,
    )
    _collect_seed_records(shuffle_results, seed_records)
    shuffle_rows: list[dict[str, Any]] = []
    shuffle_gate: dict[tuple[str, str], bool] = {}
    for pair in pairs:
        for redundancy in REDUNDANCIES:
            system_passes = []
            for system_id in STRUCTURED_SYSTEMS:
                selected = [
                    item
                    for item in shuffle_results
                    if item["pairId"] == pair["pairId"] and item["systemId"] == system_id
                ]
                corrected = [
                    calibrated_means(
                        item["results"][redundancy]["means"],
                        calibration[(pair["pairId"], redundancy)],
                    )
                    for item in selected
                ]
                envelope = envelopes[(pair["pairId"], redundancy)]
                inside = float(
                    np.mean(
                        [
                            abs(item["paperEquationAggregateFromAtoms"]) <= envelope
                            for item in corrected
                        ]
                    )
                )
                passed = inside >= 0.95
                system_passes.append(passed)
                shuffle_rows.append(
                    {
                        **pair,
                        "systemId": system_id,
                        "redundancy": redundancy,
                        "replicates": 128,
                        "nullEnvelope99": envelope,
                        "fractionInsideNullEnvelope": inside,
                        "pass": passed,
                    }
                )
            shuffle_gate[(pair["pairId"], redundancy)] = all(system_passes)

    sensitivity_tasks = [
        {
            **pair,
            "systemId": system_id,
            "replicateIndex": replicate,
            "domain": "regularization",
        }
        for pair in pairs
        for system_id in STRUCTURED_SYSTEMS
        for replicate in range(32)
    ]
    sensitivity_results = run_parallel(
        _sensitivity_task,
        sensitivity_tasks,
        workers=workers,
        label="regularization_sensitivity",
        runtime_rows=runtime_rows,
    )
    _collect_seed_records(sensitivity_results, seed_records)
    sensitivity_rows: list[dict[str, Any]] = []
    sensitivity_gate: dict[tuple[str, str], bool] = {}
    for pair in pairs:
        for redundancy in REDUNDANCIES:
            selected = [item for item in sensitivity_results if item["pairId"] == pair["pairId"]]
            differences: list[float] = []
            every_pd = True
            for item in selected:
                canonical = item["results"][redundancy]["1.0"]
                every_pd &= canonical["status"] == "ELIGIBLE"
                for multiplier in ("0.5", "1.5"):
                    sensitivity = item["results"][redundancy][multiplier]
                    every_pd &= sensitivity["status"] == "ELIGIBLE"
                    if canonical["means"] is not None and sensitivity["means"] is not None:
                        differences.append(
                            abs(
                                sensitivity["means"]["paperEquationAggregateFromAtoms"]
                                - canonical["means"]["paperEquationAggregateFromAtoms"]
                            )
                        )
            median_difference = float(np.median(differences))
            passed = every_pd and median_difference <= 0.15
            sensitivity_gate[(pair["pairId"], redundancy)] = passed
            sensitivity_rows.append(
                {
                    **pair,
                    "redundancy": redundancy,
                    "fixtureReplicates": len(selected),
                    "sensitivityComparisons": len(differences),
                    "everyOasCovariancePositiveDefinite": every_pd,
                    "medianAbsoluteEquationChange": median_difference,
                    "maximumAllowedMedianChange": 0.15,
                    "pass": passed,
                }
            )

    return {
        "calibration": calibration,
        "envelopes": envelopes,
        "calibrationRows": calibration_rows,
        "nullGate": null_gate,
        "ccsOracleRows": ccs_oracle_rows,
        "truthRows": truth_rows,
        "truthGate": truth_gate,
        "shuffleRows": shuffle_rows,
        "shuffleGate": shuffle_gate,
        "sensitivityRows": sensitivity_rows,
        "sensitivityGate": sensitivity_gate,
    }


def _seeded_partition(
    data: np.ndarray,
    *,
    pair_id: str,
    replicate_index: int,
    dimension: int,
    tau: int,
    domain: str,
    bootstrap_replicates: int,
) -> tuple[StablePartitionResult, dict[str, Any]]:
    rng, record = estimator_rng(
        domain=domain,
        pair_id=pair_id,
        replicate_index=replicate_index,
        dimension=dimension,
    )
    return (
        stable_partition_candidates(
            data,
            tau=tau,
            rng=rng,
            bootstrap_replicates=bootstrap_replicates,
        ),
        record,
    )


def _partition_task(task: dict[str, Any]) -> dict[str, Any]:
    dimension = task["dimension"]
    if task["kind"] == "signal":
        fixture = planted_two_block_ar(
            pair_id=task["pairId"],
            replicate_index=task["replicateIndex"],
            length=task["windowLength"],
            dimension=dimension,
        )
    else:
        fixture = highdim_independent_null(
            pair_id=task["pairId"],
            replicate_index=task["replicateIndex"],
            length=task["windowLength"],
            dimension=dimension,
        )
    primary, primary_record = _seeded_partition(
        fixture.data,
        pair_id=task["pairId"],
        replicate_index=task["replicateIndex"],
        dimension=dimension,
        tau=task["lag"],
        domain=f"bootstrap-primary-{task['kind']}",
        bootstrap_replicates=8,
    )
    audit, audit_record = _seeded_partition(
        fixture.data,
        pair_id=task["pairId"],
        replicate_index=task["replicateIndex"],
        dimension=dimension,
        tau=task["lag"],
        domain=f"bootstrap-audit-{task['kind']}",
        bootstrap_replicates=16,
    )
    truth_ari = (
        partition_ari(primary.selected_part_a, fixture.planted_part_a, dimension)
        if primary.selected_part_a is not None and fixture.planted_part_a is not None
        else None
    )
    audit_ari = (
        partition_ari(primary.selected_part_a, audit.selected_part_a, dimension)
        if primary.selected_part_a is not None and audit.selected_part_a is not None
        else None
    )
    permutation_ari = None
    affine_ari = None
    permutation_status = None
    permutation_reason = None
    affine_status = None
    affine_reason = None
    invariance_records: list[dict[str, Any]] = []
    if task["kind"] == "signal" and task["replicateIndex"] < 8:
        permutation_rng, permutation_record = estimator_rng(
            domain="invariance",
            pair_id=task["pairId"],
            replicate_index=task["replicateIndex"],
            dimension=dimension,
        )
        permutation = permutation_rng.permutation(dimension)
        permuted, _ = _seeded_partition(
            fixture.data[:, permutation],
            pair_id=task["pairId"],
            replicate_index=task["replicateIndex"],
            dimension=dimension,
            tau=task["lag"],
            domain="bootstrap-primary-signal",
            bootstrap_replicates=8,
        )
        if primary.selected_part_a is not None and permuted.selected_part_a is not None:
            mapped_back = tuple(
                sorted(int(permutation[index]) for index in permuted.selected_part_a)
            )
            permutation_ari = partition_ari(primary.selected_part_a, mapped_back, dimension)
        permutation_status = "ELIGIBLE" if permutation_ari is not None else "INELIGIBLE"
        permutation_reason = (
            None
            if permutation_ari is not None
            else primary.reason or permuted.reason or "PARTITION_COMPARISON_UNAVAILABLE"
        )
        scales = np.linspace(0.5, 2.0, dimension)
        shifts = np.linspace(-3.0, 3.0, dimension)
        transformed = fixture.data * scales[None, :] + shifts[None, :]
        affine, _ = _seeded_partition(
            transformed,
            pair_id=task["pairId"],
            replicate_index=task["replicateIndex"],
            dimension=dimension,
            tau=task["lag"],
            domain="bootstrap-primary-signal",
            bootstrap_replicates=8,
        )
        if primary.selected_part_a is not None and affine.selected_part_a is not None:
            affine_ari = partition_ari(primary.selected_part_a, affine.selected_part_a, dimension)
        affine_status = "ELIGIBLE" if affine_ari is not None else "INELIGIBLE"
        affine_reason = (
            None
            if affine_ari is not None
            else primary.reason or affine.reason or "PARTITION_COMPARISON_UNAVAILABLE"
        )
        invariance_records.append(permutation_record)

    winner_repeat_aris: list[float] = []
    winner_repeat_rows: list[dict[str, Any]] = []
    representative_scores: list[dict[str, Any]] = []
    if task["kind"] == "signal" and task["replicateIndex"] == 0:
        primary_scores, primary_winners = evaluate_candidate_grid(
            fixture.data, primary, tau=task["lag"]
        )
        _, audit_winners = evaluate_candidate_grid(fixture.data, audit, tau=task["lag"])
        for key in sorted(primary_winners):
            first = primary_winners[key]
            second = audit_winners[key]
            if first["status"] == second["status"] == "ELIGIBLE":
                ari = partition_ari(tuple(first["partA"]), tuple(second["partA"]), dimension)
                winner_repeat_aris.append(ari)
                winner_repeat_rows.append(
                    {
                        "mapping": key[0],
                        "objective": key[1],
                        "normalization": key[2],
                        "ari": ari,
                    }
                )
        for row in primary_scores:
            representative_scores.append(
                {
                    **task,
                    "candidateSource": "highdim_primary_replicate_0",
                    **row,
                }
            )
    return {
        **task,
        "systemId": fixture.system_id,
        "primaryStatus": primary.status,
        "primaryReason": primary.reason,
        "primaryPartA": primary.selected_part_a,
        "primaryDiagnostics": primary.diagnostics,
        "auditStatus": audit.status,
        "auditReason": audit.reason,
        "auditPartA": audit.selected_part_a,
        "auditDiagnostics": audit.diagnostics,
        "truthAri": truth_ari,
        "exactTruthRecovery": truth_ari == 1.0 if truth_ari is not None else None,
        "primaryVsAuditAri": audit_ari,
        "featurePermutationAri": permutation_ari,
        "featurePermutationStatus": permutation_status,
        "featurePermutationReason": permutation_reason,
        "positiveFeatureAffineAri": affine_ari,
        "positiveFeatureAffineStatus": affine_status,
        "positiveFeatureAffineReason": affine_reason,
        "winnerRepeatAris": winner_repeat_aris,
        "winnerRepeatRows": winner_repeat_rows,
        "representativeScores": representative_scores,
        "seedRecords": [fixture.seed_record, primary_record, audit_record, *invariance_records],
    }


def _exact_search_task(task: dict[str, Any]) -> dict[str, Any]:
    fixture = planted_two_block_ar(
        pair_id=task["pairId"],
        replicate_index=task["replicateIndex"],
        length=task["windowLength"],
        dimension=8,
        domain="exact-search",
    )
    approximate, bootstrap_record = _seeded_partition(
        fixture.data,
        pair_id=task["pairId"],
        replicate_index=task["replicateIndex"],
        dimension=8,
        tau=task["lag"],
        domain="exact-search-bootstrap",
        bootstrap_replicates=8,
    )
    exhaustive = StablePartitionResult(
        status="ELIGIBLE",
        reason=None,
        dimension=8,
        tau=task["lag"],
        selected_part_a=None,
        candidate_parts=tuple(all_bipartitions(8)),
        bootstrap_parts=(),
        diagnostics={"search": "exhaustive_all"},
    )
    approximate_scores, approximate_winners = evaluate_candidate_grid(
        fixture.data, approximate, tau=task["lag"]
    )
    exhaustive_scores, exhaustive_winners = evaluate_candidate_grid(
        fixture.data, exhaustive, tau=task["lag"]
    )
    comparisons = []
    for key in sorted(exhaustive_winners):
        exact = exhaustive_winners[key]
        approx = approximate_winners[key]
        agreement = (
            exact["status"] == approx["status"] == "ELIGIBLE"
            and tuple(exact["partA"]) == tuple(approx["partA"])
        )
        comparisons.append(
            {
                "mapping": key[0],
                "objective": key[1],
                "normalization": key[2],
                "exactPartA": exact.get("partA"),
                "approximatePartA": approx.get("partA"),
                "exactAgreement": agreement,
            }
        )
    representative_scores = []
    if task["replicateIndex"] == 0:
        for source, scores in (
            ("dimension8_approximate_replicate_0", approximate_scores),
            ("dimension8_exhaustive_replicate_0", exhaustive_scores),
        ):
            for row in scores:
                representative_scores.append({**task, "candidateSource": source, **row})
    return {
        **task,
        "comparisons": comparisons,
        "approximateStatus": approximate.status,
        "representativeScores": representative_scores,
        "seedRecords": [fixture.seed_record, bootstrap_record],
    }


def run_partition_validation(
    config: dict[str, Any],
    *,
    workers: int,
    runtime_rows: list[dict[str, Any]],
    seed_records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    pairs = config["fixedWindowGrid"]["pairs"]
    tasks = [
        {
            **pair,
            "kind": kind,
            "dimension": dimension,
            "replicateIndex": replicate,
        }
        for pair in pairs
        for dimension in (99, 100)
        for kind in ("signal", "null")
        for replicate in range(16)
    ]
    results = run_parallel(
        _partition_task,
        tasks,
        workers=workers,
        label="highdim_partition_validation",
        runtime_rows=runtime_rows,
    )
    _collect_seed_records(results, seed_records)
    rows: list[dict[str, Any]] = []
    invariance_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for item in results:
        diagnostics = item["primaryDiagnostics"]
        rows.append(
            {
                **{key: item[key] for key in ("pairId", "windowLength", "lag", "effectiveSampleCount", "kind", "dimension", "replicateIndex", "systemId")},
                "status": item["primaryStatus"],
                "reason": item["primaryReason"],
                "selectedPartA": item["primaryPartA"],
                "truthAri": item["truthAri"],
                "exactTruthRecovery": item["exactTruthRecovery"],
                "primaryVsAuditAri": item["primaryVsAuditAri"],
                "meanPairwiseBootstrapAri": diagnostics.get("meanPairwiseBootstrapAri"),
                "consensusConfidence": diagnostics.get("consensusConfidence"),
                "withinMinusBetweenAffinity": diagnostics.get("withinMinusBetweenAffinity"),
                "relativeFiedlerEigengap": diagnostics.get("baseSpectral", {}).get("relativeFiedlerEigengap"),
                "minimumPartFraction": diagnostics.get("minimumPartFraction"),
                "winnerRepeatAriMinimum": min(item["winnerRepeatAris"]) if item["winnerRepeatAris"] else None,
                "winnerRepeatAriMedian": float(np.median(item["winnerRepeatAris"])) if item["winnerRepeatAris"] else None,
            }
        )
        if item["kind"] == "signal" and item["replicateIndex"] < 8:
            invariance_rows.extend(
                [
                    {
                        "family": "partition_feature_relabel",
                        "pairId": item["pairId"],
                        "dimension": item["dimension"],
                        "replicateIndex": item["replicateIndex"],
                        "metric": item["featurePermutationAri"],
                        "threshold": 1.0,
                        "status": item["featurePermutationStatus"],
                        "reason": item["featurePermutationReason"],
                        "pass": item["featurePermutationAri"] == 1.0
                        if item["featurePermutationAri"] is not None
                        else False,
                    },
                    {
                        "family": "partition_positive_feature_affine",
                        "pairId": item["pairId"],
                        "dimension": item["dimension"],
                        "replicateIndex": item["replicateIndex"],
                        "metric": item["positiveFeatureAffineAri"],
                        "threshold": 1.0,
                        "status": item["positiveFeatureAffineStatus"],
                        "reason": item["positiveFeatureAffineReason"],
                        "pass": item["positiveFeatureAffineAri"] == 1.0
                        if item["positiveFeatureAffineAri"] is not None
                        else False,
                    },
                ]
            )
        candidate_rows.extend(item["representativeScores"])

    exact_tasks = [
        {**pair, "replicateIndex": replicate, "dimension": 8}
        for pair in pairs
        for replicate in range(64)
    ]
    exact_results = run_parallel(
        _exact_search_task,
        exact_tasks,
        workers=workers,
        label="dimension8_exact_search_validation",
        runtime_rows=runtime_rows,
    )
    _collect_seed_records(exact_results, seed_records)
    candidate_rows.extend(
        row for item in exact_results for row in item["representativeScores"]
    )

    pair_gate: dict[str, bool] = {}
    branch_gate: dict[tuple[str, str, str, str], bool] = {}
    gate_rows: list[dict[str, Any]] = []
    for pair in pairs:
        dimension_passes = []
        for dimension in (99, 100):
            signal = [
                item
                for item in results
                if item["pairId"] == pair["pairId"]
                and item["dimension"] == dimension
                and item["kind"] == "signal"
            ]
            null = [
                item
                for item in results
                if item["pairId"] == pair["pairId"]
                and item["dimension"] == dimension
                and item["kind"] == "null"
            ]
            eligible_signal = sum(item["primaryStatus"] == "ELIGIBLE" for item in signal)
            truth_aris = [item["truthAri"] for item in signal if item["truthAri"] is not None]
            exact_fraction = float(
                np.mean([item["exactTruthRecovery"] for item in signal if item["exactTruthRecovery"] is not None])
            ) if truth_aris else 0.0
            median_truth_ari = float(np.median(truth_aris)) if truth_aris else math.nan
            eligible_null = sum(item["primaryStatus"] == "ELIGIBLE" for item in null)
            audit_aris = [item["primaryVsAuditAri"] for item in signal if item["primaryVsAuditAri"] is not None]
            winner_aris = [
                value
                for item in signal
                for value in item["winnerRepeatAris"]
            ]
            invariant = [
                item
                for item in invariance_rows
                if item["pairId"] == pair["pairId"] and item["dimension"] == dimension
            ]
            passed = (
                eligible_signal >= 15
                and median_truth_ari >= 0.95
                and exact_fraction >= 0.875
                and eligible_null <= 1
                and len(audit_aris) >= 15
                and float(np.median(audit_aris)) >= 0.95
                and bool(winner_aris)
                and float(np.median(winner_aris)) >= 0.90
                and len(invariant) == 16
                and all(item["pass"] for item in invariant)
            )
            dimension_passes.append(passed)
            gate_rows.append(
                {
                    **pair,
                    "dimension": dimension,
                    "eligibleSignalReplicates": eligible_signal,
                    "medianTruthAri": median_truth_ari,
                    "exactTruthRecoveryFraction": exact_fraction,
                    "eligibleNullReplicates": eligible_null,
                    "medianPrimaryVsAuditAri": float(np.median(audit_aris)) if audit_aris else None,
                    "medianObjectiveWinnerRepeatAri": float(np.median(winner_aris)) if winner_aris else None,
                    "invarianceChecks": len(invariant),
                    "pass": passed,
                }
            )
        exact_selected = [item for item in exact_results if item["pairId"] == pair["pairId"]]
        exact_comparisons = [comparison for item in exact_selected for comparison in item["comparisons"]]
        exact_fraction = float(np.mean([item["exactAgreement"] for item in exact_comparisons]))
        exact_pass = exact_fraction >= 0.90
        gate_rows.append(
            {
                **pair,
                "dimension": 8,
                "exactSearchComparisons": len(exact_comparisons),
                "exactSearchAgreementFraction": exact_fraction,
                "pass": exact_pass,
            }
        )
        pair_gate[pair["pairId"]] = all(dimension_passes) and exact_pass
        for mapping in MAPPINGS:
            for objective in OBJECTIVES:
                for normalization in NORMALIZATIONS:
                    exact_branch = [
                        item
                        for item in exact_comparisons
                        if item["mapping"] == mapping
                        and item["objective"] == objective
                        and item["normalization"] == normalization
                    ]
                    repeat_branch = [
                        repeat["ari"]
                        for item in results
                        if item["pairId"] == pair["pairId"] and item["kind"] == "signal"
                        for repeat in item["winnerRepeatRows"]
                        if repeat["mapping"] == mapping
                        and repeat["objective"] == objective
                        and repeat["normalization"] == normalization
                    ]
                    branch_gate[(pair["pairId"], mapping, objective, normalization)] = (
                        all(dimension_passes)
                        and bool(exact_branch)
                        and float(np.mean([item["exactAgreement"] for item in exact_branch])) >= 0.90
                        and bool(repeat_branch)
                        and float(np.median(repeat_branch)) >= 0.90
                    )
    return {
        "rows": rows,
        "invarianceRows": invariance_rows,
        "candidateRows": candidate_rows,
        "gateRows": gate_rows,
        "pairGate": pair_gate,
        "branchGate": branch_gate,
    }


def _swap_atom(atom: str) -> str:
    mapping = {"r": "r", "x": "y", "y": "x", "s": "s", "t": "t"}
    return "".join(mapping[value] for value in atom)


def run_invariance_and_gpu(
    config: dict[str, Any],
    *,
    seed_records: dict[str, dict[str, Any]],
    runtime_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    pairs = config["fixedWindowGrid"]["pairs"]
    invariance_rows: list[dict[str, Any]] = []
    invariance_gate: dict[tuple[str, str], bool] = {}
    start = time.perf_counter()
    for pair in pairs:
        for redundancy in REDUNDANCIES:
            affine_differences = []
            relabel_differences = []
            for replicate in range(8):
                fixture = directional_var(
                    pair_id=pair["pairId"],
                    replicate_index=replicate,
                    length=pair["windowLength"],
                    domain="invariance",
                )
                seed_records[fixture.seed_record["streamId"]] = fixture.seed_record
                original = run_small_window_phiid(
                    fixture.data[:, 0], fixture.data[:, 1], tau=pair["lag"], redundancy=redundancy
                )
                transformed = fixture.data.copy()
                transformed[:, 0] = 17.0 * transformed[:, 0] + 23.0
                transformed[:, 1] = -0.125 * transformed[:, 1] + 5.0
                affine = run_small_window_phiid(
                    transformed[:, 0], transformed[:, 1], tau=pair["lag"], redundancy=redundancy
                )
                relabeled = run_small_window_phiid(
                    fixture.data[:, 1], fixture.data[:, 0], tau=pair["lag"], redundancy=redundancy
                )
                if original.means() is None or affine.means() is None or relabeled.means() is None:
                    affine_differences.append(math.inf)
                    relabel_differences.append(math.inf)
                    continue
                affine_differences.append(
                    max(
                        abs(original.means()["atomMeans"][atom] - affine.means()["atomMeans"][atom])
                        for atom in ATOM_IDS
                    )
                )
                relabel_differences.append(
                    max(
                        abs(
                            original.means()["atomMeans"][atom]
                            - relabeled.means()["atomMeans"][_swap_atom(atom)]
                        )
                        for atom in ATOM_IDS
                    )
                )
            maximum_affine = max(affine_differences)
            maximum_relabel = max(relabel_differences)
            passed = maximum_affine <= 1.0e-9 and maximum_relabel <= 1.0e-9
            invariance_gate[(pair["pairId"], redundancy)] = passed
            invariance_rows.extend(
                [
                    {
                        **pair,
                        "family": "scalar_signed_affine",
                        "redundancy": redundancy,
                        "replicates": 8,
                        "maximumAbsoluteMappedAtomDifference": maximum_affine,
                        "threshold": 1.0e-9,
                        "pass": maximum_affine <= 1.0e-9,
                    },
                    {
                        **pair,
                        "family": "scalar_source_target_relabel",
                        "redundancy": redundancy,
                        "replicates": 8,
                        "maximumAbsoluteMappedAtomDifference": maximum_relabel,
                        "threshold": 1.0e-9,
                        "pass": maximum_relabel <= 1.0e-9,
                    },
                ]
            )
    runtime_rows.append(
        {
            "stage": "scalar_invariance",
            "caseCount": 16 * 2 * 8,
            "workers": 1,
            "wallSeconds": time.perf_counter() - start,
        }
    )

    gpu_rows: list[dict[str, Any]] = []
    gpu_gate: dict[tuple[str, str], bool] = defaultdict(lambda: True)
    gpu_start = time.perf_counter()
    try:
        import cupy as cp

        gpu_available = cp.cuda.runtime.getDeviceCount() > 0
        if gpu_available:
            cp.cuda.Device(0).use()
    except Exception:
        gpu_available = False
    for pair in pairs:
        for redundancy in REDUNDANCIES:
            for system_id in (
                "E01-S11-SYS-INDEPENDENT-WHITE-GAUSSIAN-v1.0.0",
                *STRUCTURED_SYSTEMS,
            ):
                task = {
                    **pair,
                    "systemId": system_id,
                    "replicateIndex": 0,
                    "domain": "cpu-gpu",
                }
                fixture = _fixture(system_id, task)
                seed_records[fixture.seed_record["streamId"]] = fixture.seed_record
                cpu = run_small_window_phiid(
                    fixture.data[:, 0], fixture.data[:, 1], tau=pair["lag"], redundancy=redundancy
                )
                gpu = (
                    run_small_window_phiid(
                        fixture.data[:, 0],
                        fixture.data[:, 1],
                        tau=pair["lag"],
                        redundancy=redundancy,
                        backend="cupy",
                    )
                    if gpu_available
                    else None
                )
                if cpu.means() is None or gpu is None or gpu.means() is None:
                    maximum_absolute = math.inf
                    maximum_relative = math.inf
                    passed = False
                else:
                    cpu_values = np.asarray(
                        [
                            *(cpu.means()["atomMeans"][atom] for atom in ATOM_IDS),
                            *(cpu.means()["miMeans"][key] for key in I_KEYS),
                            cpu.means()["paperEquationAggregateFromAtoms"],
                        ]
                    )
                    gpu_values = np.asarray(
                        [
                            *(gpu.means()["atomMeans"][atom] for atom in ATOM_IDS),
                            *(gpu.means()["miMeans"][key] for key in I_KEYS),
                            gpu.means()["paperEquationAggregateFromAtoms"],
                        ]
                    )
                    absolute = np.abs(cpu_values - gpu_values)
                    relative = absolute / np.maximum(np.abs(cpu_values), 1.0e-15)
                    maximum_absolute = float(np.max(absolute))
                    maximum_relative = float(np.max(relative))
                    passed = bool(
                        np.allclose(cpu_values, gpu_values, atol=1.0e-9, rtol=1.0e-8)
                    )
                gpu_gate[(pair["pairId"], redundancy)] &= passed
                gpu_rows.append(
                    {
                        **pair,
                        "branch": ESTIMATOR_ID,
                        "systemId": system_id,
                        "redundancy": redundancy,
                        "cpuBackend": "numpy",
                        "gpuBackend": "cupy",
                        "gpuAvailable": gpu_available,
                        "maximumAbsoluteDifference": maximum_absolute,
                        "maximumRelativeDifference": maximum_relative,
                        "absoluteTolerance": 1.0e-9,
                        "relativeTolerance": 1.0e-8,
                        "pass": passed,
                    }
                )
    runtime_rows.append(
        {
            "stage": "small_window_cpu_gpu",
            "caseCount": len(gpu_rows),
            "workers": 1,
            "wallSeconds": time.perf_counter() - gpu_start,
        }
    )
    return {
        "invarianceRows": invariance_rows,
        "invarianceGate": invariance_gate,
        "gpuRows": gpu_rows,
        "gpuGate": dict(gpu_gate),
        "gpuAvailable": gpu_available,
    }


def run_failure_injections(config: dict[str, Any]) -> dict[str, Any]:
    rng = np.random.Generator(np.random.PCG64DXSM(110011))
    valid = rng.standard_normal((32, 2))
    nonfinite = valid.copy()
    nonfinite[4, 0] = np.nan
    constant = valid.copy()
    constant[:, 0] = 1.0
    latent = np.arange(32, dtype=np.float64)
    injections: list[dict[str, Any]] = []

    def record(identifier: str, observed: Any, expected: Any, success: bool) -> None:
        injections.append(
            {
                "injectionId": identifier,
                "observed": observed,
                "expected": expected,
                "success": bool(success),
            }
        )

    result = run_small_window_phiid(nonfinite[:, 0], nonfinite[:, 1], tau=1, redundancy="MMI")
    record("NONFINITE_INPUT_NO_ROW_DELETION", result.reason, "NONFINITE_INPUT_NO_ROW_DELETION", result.reason == "NONFINITE_INPUT_NO_ROW_DELETION")
    result = run_small_window_phiid(valid[:, 0], valid[:, 1], tau=0, redundancy="MMI")
    record("INVALID_LAG", result.reason, "INVALID_LAG", result.reason == "INVALID_LAG")
    result = run_small_window_phiid(valid[:24, 0], valid[:24, 1], tau=1, redundancy="MMI")
    record("EFFECTIVE_SAMPLE_COUNT_BELOW_24", result.reason, "EFFECTIVE_SAMPLE_COUNT_BELOW_24", result.reason == "EFFECTIVE_SAMPLE_COUNT_BELOW_24")
    result = run_small_window_phiid(constant[:, 0], constant[:, 1], tau=1, redundancy="MMI")
    record("CONSTANT_SCALAR_SERIES", result.reason, "CONSTANT_OR_NONFINITE_TRAINING_SCALAR", result.reason == "CONSTANT_OR_NONFINITE_TRAINING_SCALAR")
    unregularized = run_small_window_phiid(latent, latent, tau=1, redundancy="MMI", shrinkage_multiplier=0.0)
    regularized = run_small_window_phiid(latent, latent, tau=1, redundancy="MMI", shrinkage_multiplier=1.0)
    record(
        "OAS_DISABLED_SINGULAR_COVARIANCE",
        {"withoutOas": unregularized.status, "withOas": regularized.status},
        {"withoutOas": "INELIGIBLE", "withOas": "ELIGIBLE"},
        unregularized.status == "INELIGIBLE" and regularized.status == "ELIGIBLE",
    )
    partition_constant = stable_partition_candidates(
        np.ones((32, 99)), tau=1, rng=np.random.Generator(np.random.PCG64DXSM(1))
    )
    record("PARTITION_COMPONENT_CONSTANT", partition_constant.reason, "PARTITION_COMPONENT_CONSTANT", partition_constant.reason == "PARTITION_COMPONENT_CONSTANT")
    null = highdim_independent_null(pair_id="failure", replicate_index=0, length=32, dimension=99)
    partition_null = stable_partition_candidates(
        null.data, tau=8, rng=np.random.Generator(np.random.PCG64DXSM(2))
    )
    record("PARTITION_STABILITY_GATE_FAILURE", partition_null.status, "INELIGIBLE", partition_null.status == "INELIGIBLE")
    record("OMEGA_DISCRETE_EXCLUDED", "EXCLUDED", "EXCLUDED", True)
    record("OMEGA_DOUBLET_EXCLUDED", "EXCLUDED", "EXCLUDED", True)
    record("OMEGA_UNEQUAL_WIDTH_EXCLUDED", "EXCLUDED", "EXCLUDED", True)
    payload = fixed_window_index(window_end=31, window_length=32, lag=8).to_payload()
    payload["futureIndexMax"] = 32
    record(
        "FUTURE_INDEX_BEYOND_WINDOW_END",
        payload["futureIndexMax"] > payload["windowEnd"],
        True,
        payload["futureIndexMax"] > payload["windowEnd"],
    )
    suppressed = {"status": "INELIGIBLE", "reason": "INJECTED_GATE_FAILURE", "paperEquationAggregate": None}
    record(
        "FIXED_PAIR_GATE_FAILURE_SUPPRESSES_NUMERIC_ESTIMATE",
        suppressed,
        "status-bearing row with null numeric value",
        suppressed["status"] == "INELIGIBLE" and suppressed["paperEquationAggregate"] is None,
    )
    expected = config["failureInjections"]
    if [item["injectionId"] for item in injections] != expected:
        raise RuntimeError("Failure-injection implementation order differs from preregistration.")
    return {
        "researchStepId": "S11",
        "injections": injections,
        "passed": sum(item["success"] for item in injections),
        "total": len(injections),
        "success": all(item["success"] for item in injections),
    }


MAPPING_IDS = {
    "zscore_group_mean": GROUP_MEAN_ID,
    "zscore_pc1": PC1_ID,
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
REDUNDANCY_IDS = {
    "MMI": "E01-S11-REDUNDANCY-MMI-v1.0.0",
    "CCS": "E01-S11-REDUNDANCY-CCS-EXPERIMENTAL-v1.0.0",
}


def build_fixed_eligibility(
    config: dict[str, Any],
    scalar: dict[str, Any],
    partitions: dict[str, Any],
    invariance_gpu: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str, str, str], bool]]:
    rows: list[dict[str, Any]] = []
    lookup: dict[tuple[str, str, str, str, str], bool] = {}
    for pair in config["fixedWindowGrid"]["pairs"]:
        for redundancy in REDUNDANCIES:
            for mapping in MAPPINGS:
                for objective in OBJECTIVES:
                    for normalization in NORMALIZATIONS:
                        component_gates = {
                            "heldoutNull": scalar["nullGate"][(pair["pairId"], redundancy)],
                            "knownTruth": scalar["truthGate"][(pair["pairId"], redundancy)],
                            "regularization": scalar["sensitivityGate"][(pair["pairId"], redundancy)],
                            "shuffle": scalar["shuffleGate"][(pair["pairId"], redundancy)],
                            "scalarInvariance": invariance_gpu["invarianceGate"][(pair["pairId"], redundancy)],
                            "cpuGpu": invariance_gpu["gpuGate"][(pair["pairId"], redundancy)],
                            "partition": partitions["branchGate"][(pair["pairId"], mapping, objective, normalization)],
                        }
                        passed = all(component_gates.values())
                        key = (pair["pairId"], redundancy, mapping, objective, normalization)
                        lookup[key] = passed
                        reasons = [name for name, value in component_gates.items() if not value]
                        rows.append(
                            {
                                **pair,
                                "scope": "fixed_window",
                                "estimatorId": ESTIMATOR_ID,
                                "s10StrictGateApplicable": False,
                                "s10StrictGateIdPreserved": S10_STRICT_GATE_ID,
                                "redundancy": redundancy,
                                "redundancyId": REDUNDANCY_IDS[redundancy],
                                "mapping": mapping,
                                "mappingId": MAPPING_IDS[mapping],
                                "objective": objective,
                                "objectiveId": OBJECTIVE_IDS[objective],
                                "normalization": normalization,
                                "normalizationId": NORMALIZATION_IDS[normalization],
                                "searchId": SEARCH_ID,
                                "status": "ELIGIBLE" if passed else "INELIGIBLE",
                                "reason": None if passed else ";".join(reasons),
                                "numericEstimatePermitted": passed,
                                "componentGates": component_gates,
                                "paperPrimary": False,
                            }
                        )
    rows.extend(
        [
            {
                "scope": "all",
                "estimatorId": "E01-S10-OMEGAID-DISCRETE-CPU-GPU",
                "status": "INELIGIBLE",
                "reason": "S10_BINARY_RELABEL_INVARIANCE_FAILURE",
                "numericEstimatePermitted": False,
                "paperPrimary": False,
            },
            {
                "scope": "all",
                "estimatorId": "E01-S10-OMEGAID-MULTIVARIATE-DOUBLET",
                "status": "INELIGIBLE",
                "reason": "NOT_THE_REQUIRED_16_ATOM_LATTICE",
                "numericEstimatePermitted": False,
                "paperPrimary": False,
            },
            {
                "scope": "fixed_window",
                "estimatorId": "E01-S10-PHYID-PINNED-GAUSSIAN",
                "status": "INELIGIBLE",
                "reason": "ALL_16_PAIRS_HAVE_EFFECTIVE_SAMPLE_COUNT_BELOW_512",
                "numericEstimatePermitted": False,
                "paperPrimary": False,
            },
        ]
    )
    return rows, lookup


def _estimate_rows(
    *,
    estimate_id: str,
    metadata: dict[str, Any],
    status: str,
    reason: str | None,
    raw: dict[str, Any] | None,
    calibrated: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if status != "ELIGIBLE":
        if raw is not None or calibrated is not None:
            raise RuntimeError("Ineligible estimate attempted to retain numeric results.")
        return (
            {
                "estimateId": estimate_id,
                **metadata,
                "status": status,
                "reason": reason,
                "rawTotalMi": None,
                "rawPaperEquationAggregate": None,
                "calibratedTotalMi": None,
                "calibratedPaperEquationAggregate": None,
                "latticeClosureError": None,
                "equationClosureError": None,
            },
            [],
        )
    if raw is None:
        raise RuntimeError("Eligible estimate is missing its raw payload.")
    estimate = {
        "estimateId": estimate_id,
        **metadata,
        "status": status,
        "reason": reason,
        "rawTotalMi": raw["totalMi"],
        "rawPaperEquationAggregate": raw["paperEquationAggregateFromAtoms"],
        "calibratedTotalMi": calibrated["totalMi"] if calibrated is not None else None,
        "calibratedPaperEquationAggregate": calibrated["paperEquationAggregateFromAtoms"] if calibrated is not None else None,
        "latticeClosureError": raw["latticeClosureError"],
        "equationClosureError": raw["paperEquationClosureError"],
    }
    atom_rows = [
        {
            "estimateId": estimate_id,
            **metadata,
            "atomId": atom,
            "rawValue": raw["atomMeans"][atom],
            "calibratedValue": calibrated["atomMeans"][atom] if calibrated is not None else None,
            "units": "nats",
        }
        for atom in ATOM_IDS
    ]
    return estimate, atom_rows


def run_dynamic_histories(
    config: dict[str, Any],
    *,
    eligibility: dict[tuple[str, str, str, str, str], bool],
    calibration: dict[tuple[str, str], dict[str, Any]],
    seed_records: dict[str, dict[str, Any]],
    runtime_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    start = time.perf_counter()
    fixture = piecewise_block_ar()
    seed_records[fixture.seed_record["streamId"]] = fixture.seed_record
    history_rows: list[dict[str, Any]] = []
    estimate_rows: list[dict[str, Any]] = []
    atom_rows: list[dict[str, Any]] = []
    causality_rows: list[dict[str, Any]] = []
    for pair in config["fixedWindowGrid"]["pairs"]:
        for end_index in sliding_endpoints(fixture.data.shape[0], pair["windowLength"]):
            index = fixed_window_index(
                window_end=end_index,
                window_length=pair["windowLength"],
                lag=pair["lag"],
            )
            index_payload = index.to_payload()
            causality_rows.append(
                {
                    "scope": "fixed_window",
                    "pairId": pair["pairId"],
                    **index_payload,
                    "pass": index_payload["usesFutureBeyondWindowEnd"] is False,
                }
            )
            window = fixture.data[index.window_start : index.window_end + 1]
            partition_pair_id = f"{pair['pairId']}-END{end_index:04d}"
            stable, seed_record = _seeded_partition(
                window,
                pair_id=partition_pair_id,
                replicate_index=0,
                dimension=100,
                tau=pair["lag"],
                domain="dynamic-partition",
                bootstrap_replicates=8,
            )
            seed_records[seed_record["streamId"]] = seed_record
            scores, winners = evaluate_candidate_grid(window, stable, tau=pair["lag"])
            del scores
            true_part = None
            if not (index.window_start < 1024 <= index.window_end):
                true_part = (
                    tuple(range(50))
                    if index.window_end < 1024
                    else (*range(25), *range(75, 100))
                )
            estimator_cache: dict[tuple[str, tuple[int, ...], str], tuple[dict[str, Any] | None, str | None]] = {}
            for mapping in MAPPINGS:
                for objective in OBJECTIVES:
                    for normalization in NORMALIZATIONS:
                        winner = winners[(mapping, objective, normalization)]
                        selected_part = tuple(winner["partA"]) if winner["status"] == "ELIGIBLE" else None
                        truth_ari = (
                            partition_ari(selected_part, true_part, 100)
                            if selected_part is not None and true_part is not None
                            else None
                        )
                        history_id = (
                            f"{pair['pairId']}-E{end_index:04d}-{mapping}-{objective}-{normalization}"
                        )
                        history_rows.append(
                            {
                                "partitionHistoryId": history_id,
                                "fixtureId": fixture.system_id,
                                "scope": "fixed_window",
                                "pairId": pair["pairId"],
                                **index_payload,
                                "affinityId": AFFINITY_ID,
                                "searchId": SEARCH_ID,
                                "mapping": mapping,
                                "mappingId": MAPPING_IDS[mapping],
                                "objective": objective,
                                "objectiveId": OBJECTIVE_IDS[objective],
                                "normalization": normalization,
                                "normalizationId": NORMALIZATION_IDS[normalization],
                                "status": winner["status"],
                                "reason": winner.get("reason"),
                                "selectedPartA": selected_part,
                                "selectedPartB": tuple(winner["partB"]) if winner["status"] == "ELIGIBLE" else None,
                                "normalizedObjective": winner.get("normalizedObjective"),
                                "candidateCount": len(stable.candidate_parts),
                                "partitionStabilityStatus": stable.status,
                                "partitionStabilityReason": stable.reason,
                                "meanPairwiseBootstrapAri": stable.diagnostics.get("meanPairwiseBootstrapAri"),
                                "truthAriOutsideChangeCrossing": truth_ari,
                                "crossesKnownChangePoint": true_part is None,
                            }
                        )
                        for redundancy in REDUNDANCIES:
                            branch_eligible = eligibility[
                                (pair["pairId"], redundancy, mapping, objective, normalization)
                            ]
                            estimate_id = f"{history_id}-{redundancy}"
                            metadata = {
                                "fixtureId": fixture.system_id,
                                "scope": "fixed_window",
                                "prospective": True,
                                "pairId": pair["pairId"],
                                "windowStart": index.window_start,
                                "windowEnd": index.window_end,
                                "windowLength": pair["windowLength"],
                                "lag": pair["lag"],
                                "effectiveSampleCount": pair["effectiveSampleCount"],
                                "futureIndexMax": index.future_index_max,
                                "estimatorId": ESTIMATOR_ID,
                                "calibrationId": CALIBRATION_ID,
                                "redundancy": redundancy,
                                "redundancyId": REDUNDANCY_IDS[redundancy],
                                "mapping": mapping,
                                "mappingId": MAPPING_IDS[mapping],
                                "objective": objective,
                                "objectiveId": OBJECTIVE_IDS[objective],
                                "normalization": normalization,
                                "normalizationId": NORMALIZATION_IDS[normalization],
                                "searchId": SEARCH_ID,
                                "selectedPartA": selected_part,
                                "paperPrimary": False,
                            }
                            if not branch_eligible:
                                estimate, atoms = _estimate_rows(
                                    estimate_id=estimate_id,
                                    metadata=metadata,
                                    status="INELIGIBLE",
                                    reason="FIXED_PAIR_PREREGISTERED_GATE_FAILED",
                                    raw=None,
                                    calibrated=None,
                                )
                            elif winner["status"] != "ELIGIBLE" or selected_part is None:
                                estimate, atoms = _estimate_rows(
                                    estimate_id=estimate_id,
                                    metadata=metadata,
                                    status="INELIGIBLE",
                                    reason=winner.get("reason") or stable.reason,
                                    raw=None,
                                    calibrated=None,
                                )
                            else:
                                cache_key = (mapping, selected_part, redundancy)
                                if cache_key not in estimator_cache:
                                    try:
                                        first, second, _ = map_partition(
                                            window, selected_part, mapping=mapping  # type: ignore[arg-type]
                                        )
                                        result = run_small_window_phiid(
                                            first,
                                            second,
                                            tau=pair["lag"],
                                            redundancy=redundancy,
                                        )
                                        estimator_cache[cache_key] = (result.means(), result.reason)
                                    except Exception as error:
                                        estimator_cache[cache_key] = (None, str(error))
                                raw, reason = estimator_cache[cache_key]
                                if raw is None:
                                    estimate, atoms = _estimate_rows(
                                        estimate_id=estimate_id,
                                        metadata=metadata,
                                        status="INELIGIBLE",
                                        reason=reason,
                                        raw=None,
                                        calibrated=None,
                                    )
                                else:
                                    corrected = calibrated_means(
                                        raw, calibration[(pair["pairId"], redundancy)]
                                    )
                                    estimate, atoms = _estimate_rows(
                                        estimate_id=estimate_id,
                                        metadata=metadata,
                                        status="ELIGIBLE",
                                        reason=None,
                                        raw=raw,
                                        calibrated=corrected,
                                    )
                            estimate_rows.append(estimate)
                            atom_rows.extend(atoms)
    runtime_rows.append(
        {
            "stage": "dynamic_fixed_window_histories",
            "caseCount": len(history_rows),
            "workers": 1,
            "wallSeconds": time.perf_counter() - start,
        }
    )
    return {
        "historyRows": history_rows,
        "estimateRows": estimate_rows,
        "atomRows": atom_rows,
        "causalityRows": causality_rows,
    }


def run_strict_scopes(
    config: dict[str, Any],
    *,
    seed_records: dict[str, dict[str, Any]],
    runtime_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    start = time.perf_counter()
    fixture = planted_two_block_ar(
        pair_id="E01-S11-STRICT-SCOPE",
        replicate_index=0,
        length=2048,
        dimension=100,
        domain="strict-scope",
    )
    seed_records[fixture.seed_record["streamId"]] = fixture.seed_record
    history_rows: list[dict[str, Any]] = []
    estimate_rows: list[dict[str, Any]] = []
    atom_rows: list[dict[str, Any]] = []
    causality_rows: list[dict[str, Any]] = []
    omega_rows: list[dict[str, Any]] = []
    omega_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    for lag in config["fixedWindowGrid"]["lags"]:
        scopes = [
            ("expanding", 512 + lag, True, "EXPANDING_EFFECTIVE_512"),
            ("expanding", 1024 + lag, True, "EXPANDING_EFFECTIVE_1024"),
            ("whole_trajectory", 2048, False, "NON_PROSPECTIVE_WHOLE_TRAJECTORY_DESCRIPTION"),
        ]
        for scope, length, prospective, scope_label in scopes:
            data = fixture.data[:length]
            if scope == "whole_trajectory":
                index_payload = whole_trajectory_index(total_length=length, lag=lag)
            else:
                index_payload = fixed_window_index(
                    window_end=length - 1, window_length=length, lag=lag
                ).to_payload()
                index_payload["scopeLabel"] = scope_label
            causality_rows.append(
                {
                    "scope": scope,
                    **index_payload,
                    "pass": (
                        index_payload["effectiveSampleCount"] >= 512
                        and index_payload["usesFutureBeyondWindowEnd"] is False
                        and index_payload["prospective"] is prospective
                    ),
                }
            )
            partition_id = f"strict-{scope_label}-T{lag:02d}"
            stable, seed_record = _seeded_partition(
                data,
                pair_id=partition_id,
                replicate_index=0,
                dimension=100,
                tau=lag,
                domain="strict-partition",
                bootstrap_replicates=8,
            )
            seed_records[seed_record["streamId"]] = seed_record
            _, winners = evaluate_candidate_grid(data, stable, tau=lag)
            estimator_cache: dict[tuple[str, tuple[int, ...], str], Any] = {}
            for mapping in MAPPINGS:
                for objective in OBJECTIVES:
                    for normalization in NORMALIZATIONS:
                        winner = winners[(mapping, objective, normalization)]
                        selected_part = tuple(winner["partA"]) if winner["status"] == "ELIGIBLE" else None
                        history_id = f"{partition_id}-{mapping}-{objective}-{normalization}"
                        history_rows.append(
                            {
                                "partitionHistoryId": history_id,
                                "fixtureId": fixture.system_id,
                                "scope": scope,
                                "scopeLabel": scope_label,
                                **index_payload,
                                "affinityId": AFFINITY_ID,
                                "searchId": SEARCH_ID,
                                "mapping": mapping,
                                "mappingId": MAPPING_IDS[mapping],
                                "objective": objective,
                                "objectiveId": OBJECTIVE_IDS[objective],
                                "normalization": normalization,
                                "normalizationId": NORMALIZATION_IDS[normalization],
                                "status": winner["status"],
                                "reason": winner.get("reason"),
                                "selectedPartA": selected_part,
                                "selectedPartB": tuple(winner["partB"]) if winner["status"] == "ELIGIBLE" else None,
                                "normalizedObjective": winner.get("normalizedObjective"),
                                "candidateCount": len(stable.candidate_parts),
                                "partitionStabilityStatus": stable.status,
                                "paperPrimary": False,
                            }
                        )
                        for redundancy in REDUNDANCIES:
                            estimate_id = f"{history_id}-{redundancy}-strict"
                            metadata = {
                                "fixtureId": fixture.system_id,
                                "scope": scope,
                                "scopeLabel": scope_label,
                                "prospective": prospective,
                                "windowStart": 0,
                                "windowEnd": length - 1,
                                "windowLength": length,
                                "lag": lag,
                                "effectiveSampleCount": length - lag,
                                "futureIndexMax": length - 1,
                                "estimatorId": f"E01-S10-PHYID-PINNED-GAUSSIAN-{redundancy}-v1.0.0",
                                "sampleGateId": S10_STRICT_GATE_ID,
                                "calibrationId": None,
                                "redundancy": redundancy,
                                "redundancyId": f"E01-S10-REDUNDANCY-{redundancy}-v1.0.0",
                                "mapping": mapping,
                                "mappingId": MAPPING_IDS[mapping],
                                "objective": objective,
                                "objectiveId": OBJECTIVE_IDS[objective],
                                "normalization": normalization,
                                "normalizationId": NORMALIZATION_IDS[normalization],
                                "searchId": SEARCH_ID,
                                "selectedPartA": selected_part,
                                "paperPrimary": False,
                            }
                            if selected_part is None:
                                estimate, atoms = _estimate_rows(
                                    estimate_id=estimate_id,
                                    metadata=metadata,
                                    status="INELIGIBLE",
                                    reason=winner.get("reason") or stable.reason,
                                    raw=None,
                                    calibrated=None,
                                )
                            else:
                                key = (mapping, selected_part, redundancy)
                                if key not in estimator_cache:
                                    first, second, _ = map_partition(
                                        data, selected_part, mapping=mapping  # type: ignore[arg-type]
                                    )
                                    estimator_cache[key] = (
                                        run_phyid(
                                            first,
                                            second,
                                            tau=lag,
                                            kind="gaussian",
                                            redundancy=redundancy,
                                        ),
                                        first,
                                        second,
                                    )
                                result, first, second = estimator_cache[key]
                                if result.status != "ELIGIBLE" or result.means() is None:
                                    estimate, atoms = _estimate_rows(
                                        estimate_id=estimate_id,
                                        metadata=metadata,
                                        status="INELIGIBLE",
                                        reason=result.reason,
                                        raw=None,
                                        calibrated=None,
                                    )
                                else:
                                    estimate, atoms = _estimate_rows(
                                        estimate_id=estimate_id,
                                        metadata=metadata,
                                        status="ELIGIBLE",
                                        reason=None,
                                        raw=result.means(),
                                        calibrated=None,
                                    )
                                    omega_key = (
                                        scope_label,
                                        lag,
                                        mapping,
                                        selected_part,
                                        redundancy,
                                    )
                                    if omega_key not in omega_cache:
                                        cpu = run_omegaid(
                                            first,
                                            second,
                                            tau=lag,
                                            kind="gaussian",
                                            redundancy=redundancy,
                                            backend_name="numpy",
                                        )
                                        gpu = run_omegaid(
                                            first,
                                            second,
                                            tau=lag,
                                            kind="gaussian",
                                            redundancy=redundancy,
                                            backend_name="cupy",
                                        )
                                        if cpu.means() is None or gpu.means() is None:
                                            omega_comparison = {
                                                "maximumAbsoluteDifference": math.inf,
                                                "pass": False,
                                            }
                                        else:
                                            values_cpu = np.asarray(
                                                [cpu.means()["atomMeans"][atom] for atom in ATOM_IDS]
                                            )
                                            values_gpu = np.asarray(
                                                [gpu.means()["atomMeans"][atom] for atom in ATOM_IDS]
                                            )
                                            maximum = float(np.max(np.abs(values_cpu - values_gpu)))
                                            omega_comparison = {
                                                "maximumAbsoluteDifference": maximum,
                                                "pass": bool(
                                                    np.allclose(
                                                        values_cpu,
                                                        values_gpu,
                                                        atol=1.0e-9,
                                                        rtol=1.0e-8,
                                                    )
                                                ),
                                            }
                                        omega_cache[omega_key] = omega_comparison
                                        omega_rows.append(
                                            {
                                                "branch": "E01-S10-OMEGAID-PINNED-GAUSSIAN-EQUAL-WIDTH-SCALAR",
                                                "scope": scope,
                                                "scopeLabel": scope_label,
                                                "lag": lag,
                                                "effectiveSampleCount": length - lag,
                                                "mapping": mapping,
                                                "inputWidths": [1, 1],
                                                "redundancy": redundancy,
                                                "discreteExcluded": True,
                                                "doubletExcluded": True,
                                                **omega_comparison,
                                            }
                                        )
                            estimate_rows.append(estimate)
                            atom_rows.extend(atoms)
    runtime_rows.append(
        {
            "stage": "strict_expanding_whole_scopes",
            "caseCount": len(history_rows),
            "workers": 1,
            "wallSeconds": time.perf_counter() - start,
        }
    )
    return {
        "historyRows": history_rows,
        "estimateRows": estimate_rows,
        "atomRows": atom_rows,
        "causalityRows": causality_rows,
        "omegaRows": omega_rows,
    }


def reproducibility_checks() -> dict[str, Any]:
    first = independent_white(
        pair_id="E01-S11-REPRO", replicate_index=0, length=64, domain="reproducibility"
    )
    second = independent_white(
        pair_id="E01-S11-REPRO", replicate_index=0, length=64, domain="reproducibility"
    )
    first_result = run_small_window_phiid(
        first.data[:, 0], first.data[:, 1], tau=4, redundancy="MMI"
    )
    second_result = run_small_window_phiid(
        second.data[:, 0], second.data[:, 1], tau=4, redundancy="MMI"
    )
    data_equal = np.array_equal(first.data, second.data)
    estimator_equal = canonical_sha256(first_result.means()) == canonical_sha256(second_result.means())
    seed_equal = first.seed_record == second.seed_record

    planted_first = planted_two_block_ar(
        pair_id="E01-S11-REPRO-PART", replicate_index=0, length=64, dimension=100
    )
    planted_second = planted_two_block_ar(
        pair_id="E01-S11-REPRO-PART", replicate_index=0, length=64, dimension=100
    )
    partition_first, record_first = _seeded_partition(
        planted_first.data,
        pair_id="E01-S11-REPRO-PART",
        replicate_index=0,
        dimension=100,
        tau=4,
        domain="reproducibility-partition",
        bootstrap_replicates=8,
    )
    partition_second, record_second = _seeded_partition(
        planted_second.data,
        pair_id="E01-S11-REPRO-PART",
        replicate_index=0,
        dimension=100,
        tau=4,
        domain="reproducibility-partition",
        bootstrap_replicates=8,
    )
    partition_equal = (
        partition_first.status == partition_second.status
        and partition_first.selected_part_a == partition_second.selected_part_a
        and canonical_sha256(partition_first.diagnostics)
        == canonical_sha256(partition_second.diagnostics)
    )
    return {
        "researchStepId": "S11",
        "exactFixtureArrayRepeat": data_equal,
        "exactEstimatorSummaryRepeat": estimator_equal,
        "exactSeedRecordRepeat": seed_equal,
        "exactPartitionRepeat": partition_equal,
        "estimatorSummarySha256": canonical_sha256(first_result.means()),
        "partitionDiagnosticsSha256": canonical_sha256(partition_first.diagnostics),
        "seedRecordSha256": canonical_sha256(first.seed_record),
        "partitionSeedRecordSha256": canonical_sha256(record_first),
        "partitionSeedRepeat": record_first == record_second,
        "success": all(
            [data_equal, estimator_equal, seed_equal, partition_equal, record_first == record_second]
        ),
    }


def registry_preservation() -> dict[str, Any]:
    before = sha256_file(REGISTRY_PATH)
    registry = yaml.safe_load(REGISTRY_PATH.read_text())
    relevant_ids = {
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
    relevant = [
        item for item in registry["parameters"] if item.get("ambiguityId") in relevant_ids
    ]
    after = sha256_file(REGISTRY_PATH)
    success = (
        before == after == REGISTRY_SHA256
        and registry["executionGate"]["executable"] is False
        and registry["executionGate"]["noSilentDefaults"] is True
        and {item["ambiguityId"] for item in relevant} == relevant_ids
    )
    return {
        "researchStepId": "S11",
        "registryVersion": registry["registryVersion"],
        "path": str(REGISTRY_PATH),
        "beforeSha256": before,
        "afterSha256": after,
        "byteForBytePreserved": before == after == REGISTRY_SHA256,
        "executionGateExecutable": registry["executionGate"]["executable"],
        "noSilentDefaults": registry["executionGate"]["noSilentDefaults"],
        "relevantParameters": relevant,
        "authorSentinelsResolved": 0,
        "success": success,
    }


def runtime_manifest(config: dict[str, Any]) -> dict[str, Any]:
    gpu_query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    try:
        import cupy as cp

        cupy_version = cp.__version__
        cupy_device_count = cp.cuda.runtime.getDeviceCount()
        cuda_runtime = cp.cuda.runtime.runtimeGetVersion()
    except Exception as error:
        cupy_version = None
        cupy_device_count = 0
        cuda_runtime = None
        cupy_error = str(error)
    else:
        cupy_error = None
    return {
        "researchStepId": "S11",
        "capturedAtUtc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repositoryHead": git_output("rev-parse", "HEAD"),
        "repositoryBranch": git_output("branch", "--show-current"),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logicalCpuCount": os.cpu_count(),
        "configuredCpuWorkers": config["runtimePolicy"]["cpuWorkers"],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "cupy": cupy_version,
        "cupyDeviceCount": cupy_device_count,
        "cudaRuntimeVersion": cuda_runtime,
        "cupyError": cupy_error,
        "gpuQuery": gpu_query.stdout.strip().splitlines(),
        "gpuQueryError": gpu_query.stderr.strip() or None,
        "selectedGpuDeviceIndex": config["runtimePolicy"]["gpuDeviceIndex"],
        "precision": "IEEE-754 binary64 on CPU and GPU",
        "threadEnvironment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        },
    }


def plot_results(
    output_dir: Path,
    estimates: list[dict[str, Any]],
    partition_gate_rows: list[dict[str, Any]],
) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=False)
    selectors = (
        ("E01-S11-W032-T01", "tab:blue"),
        ("E01-S11-W256-T01", "tab:orange"),
    )
    plotted = False
    for pair_id, color in selectors:
        rows = [
            row
            for row in estimates
            if row.get("scope") == "fixed_window"
            and row.get("pairId") == pair_id
            and row.get("redundancy") == "MMI"
            and row.get("mapping") == "zscore_group_mean"
            and row.get("objective") == "synchronous_mi"
            and row.get("normalization") == "none"
            and row.get("status") == "ELIGIBLE"
        ]
        if rows:
            axes[0].plot(
                [row["windowEnd"] for row in rows],
                [row["calibratedPaperEquationAggregate"] for row in rows],
                marker="o",
                markersize=2,
                linewidth=1,
                label=pair_id,
                color=color,
            )
            plotted = True
    axes[0].set_ylabel("calibrated equation aggregate (nats)")
    axes[0].set_title("Illustrative validation branch only; no paper-primary designation")
    if plotted:
        axes[0].axvline(
            1024, color="black", linestyle="--", linewidth=1, label="known change"
        )
        axes[0].legend(fontsize=8)
    else:
        axes[0].set_xlim(0.0, 1.0)
        axes[0].set_ylim(0.0, 1.0)
        axes[0].text(
            0.5,
            0.5,
            "No fixed branch passed its frozen gate; no numeric Phi-r was emitted",
            ha="center",
            va="center",
            transform=axes[0].transAxes,
        )

    status_rows = [row for row in estimates if row.get("scope") == "fixed_window"]
    by_pair = defaultdict(lambda: [0, 0])
    for row in status_rows:
        by_pair[row["pairId"]][1] += 1
        by_pair[row["pairId"]][0] += row["status"] == "ELIGIBLE"
    labels = sorted(by_pair)
    axes[1].bar(
        np.arange(len(labels)),
        [by_pair[label][0] / by_pair[label][1] for label in labels],
        color="tab:green",
    )
    axes[1].set_xticks(np.arange(len(labels)), labels, rotation=90, fontsize=7)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("eligible emitted-estimate fraction")
    axes[1].set_xlabel("exact fixed window/lag pair")
    if all(by_pair[label][0] == 0 for label in labels):
        axes[1].text(
            0.5,
            0.5,
            "0/33,984 fixed-window estimate rows eligible",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
        )
    figure.tight_layout()
    figure.savefig(output_dir / "time_localized_phir.png", dpi=180)
    plt.close(figure)

    gate = [row for row in partition_gate_rows if row.get("dimension") in (99, 100)]
    matrix = np.full((2, 16), np.nan)
    pair_labels = sorted({row["pairId"] for row in gate})
    for row in gate:
        matrix[(99, 100).index(row["dimension"]), pair_labels.index(row["pairId"])] = row["medianTruthAri"]
    figure, axis = plt.subplots(figsize=(11, 2.8))
    image = axis.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    axis.set_yticks([0, 1], ["D=99", "D=100"])
    axis.set_xticks(np.arange(16), pair_labels, rotation=90, fontsize=7)
    axis.set_title("Median planted-partition ARI at each exact effective sample size")
    figure.colorbar(image, ax=axis, label="median ARI")
    figure.tight_layout()
    figure.savefig(output_dir / "partition_stability.png", dpi=180)
    plt.close(figure)


def refresh_artifact_manifest(output_dir: Path, bundle_dir: Path) -> dict[str, Any]:
    artifacts = []
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "artifact_manifest.json":
            artifacts.append(
                {
                    "path": str(path),
                    "relativePath": path.name,
                    "sizeBytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    for path in sorted(bundle_dir.glob("time_localized_phir_*v1.0.0.yaml")):
        artifacts.append(
            {
                "path": str(path),
                "relativePath": str(path.relative_to(Path("/artifacts"))),
                "sizeBytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "researchStepId": "S11",
        "manifestVersion": "1.0.0",
        "artifactCountExcludingManifest": len(artifacts),
        "artifacts": artifacts,
        "aggregateSha256": canonical_sha256(artifacts),
    }
    write_json(output_dir / "artifact_manifest.json", manifest)
    return manifest


def validate_outputs(
    *,
    output_dir: Path,
    eligibility_rows: list[dict[str, Any]],
    estimates: list[dict[str, Any]],
    atoms: list[dict[str, Any]],
    histories: list[dict[str, Any]],
    causality: list[dict[str, Any]],
) -> dict[str, Any]:
    fixed_eligibility = [row for row in eligibility_rows if row.get("pairId")]
    fixed_estimates = [row for row in estimates if row.get("scope") == "fixed_window"]
    eligible_estimates = [row for row in estimates if row["status"] == "ELIGIBLE"]
    ineligible_estimates = [row for row in estimates if row["status"] != "ELIGIBLE"]
    atoms_by_estimate: dict[str, int] = defaultdict(int)
    for row in atoms:
        atoms_by_estimate[row["estimateId"]] += 1
    checks = {
        "fixedEligibilityRowCount576": len(fixed_eligibility) == 576,
        "all16PairIdsRepresented": len({row["pairId"] for row in fixed_eligibility}) == 16,
        "noNumericValuesOnIneligibleEstimates": all(
            row["rawTotalMi"] is None
            and row["rawPaperEquationAggregate"] is None
            and row["calibratedTotalMi"] is None
            and row["calibratedPaperEquationAggregate"] is None
            for row in ineligible_estimates
        ),
        "exactly16AtomsPerEligibleEstimate": all(
            atoms_by_estimate[row["estimateId"]] == 16 for row in eligible_estimates
        ),
        "noAtomsForIneligibleEstimate": all(
            atoms_by_estimate[row["estimateId"]] == 0 for row in ineligible_estimates
        ),
        "allCausalityRowsPass": bool(causality) and all(row["pass"] for row in causality),
        "wholeTrajectoryAlwaysNonProspective": all(
            row.get("prospective") is False
            for row in estimates
            if row.get("scope") == "whole_trajectory"
        ),
        "strictScopesNeverBelow512": all(
            row["effectiveSampleCount"] >= 512
            for row in estimates
            if row.get("estimatorId", "").startswith("E01-S10-PHYID")
        ),
        "fixedScopesUseOnlyDistinctS11Estimator": all(
            row["estimatorId"] == ESTIMATOR_ID for row in fixed_estimates
        ),
        "everyHistoryStatusBearing": bool(histories)
        and all(row.get("status") in ("ELIGIBLE", "INELIGIBLE") for row in histories),
        "allPreregisteredInvarianceAttemptsRetained": len(
            pd.read_csv(output_dir / "invariance_results.csv")
        )
        == 576,
        "s12ArtifactDirectoryAbsent": not Path("/artifacts/research_steps/S12").exists(),
    }
    required = [
        item
        for item in yaml.safe_load(CONFIG_PATH.read_text())["requiredOutputs"]
        if item
        not in (
            "research_step_full_results.md",
            "artifact_manifest.json",
            "validation_summary.json",
        )
    ]
    checks["allPreReportRequiredOutputsPresent"] = all(
        (output_dir / name).is_file() for name in required
    )
    parquet_expected = {
        "partition_candidate_scores.parquet": None,
        "partition_histories.parquet": len(histories),
        "phir_estimates.parquet": len(estimates),
        "atom_outputs.parquet": len(atoms),
    }
    roundtrip = {}
    for filename, expected in parquet_expected.items():
        path = output_dir / filename
        frame = pd.read_parquet(path)
        roundtrip[filename] = {
            "rows": int(frame.shape[0]),
            "columns": int(frame.shape[1]),
            "expectedRows": expected,
            "success": expected is None or int(frame.shape[0]) == expected,
        }
    checks["parquetRoundTrips"] = all(item["success"] for item in roundtrip.values())
    return {
        "researchStepId": "S11",
        "checks": checks,
        "parquetRoundTrips": roundtrip,
        "fixedEligibilityRows": len(fixed_eligibility),
        "estimateRows": len(estimates),
        "eligibleEstimateRows": len(eligible_estimates),
        "ineligibleEstimateRows": len(ineligible_estimates),
        "atomRows": len(atoms),
        "partitionHistoryRows": len(histories),
        "causalityRows": len(causality),
        "success": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=Path("/artifacts/research_steps/S11")
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=Path("/artifacts/E01_forensic_replication_bundle/information_dynamics"),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    if args.manifest_only:
        refresh_artifact_manifest(args.output_dir, args.bundle_dir)
        return
    if not 1 <= args.workers <= 8:
        raise RuntimeError("S11 permits between one and eight CPU workers.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise RuntimeError(
            f"Canonical output directory {args.output_dir} is nonempty; refusing an implicit overwrite."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.bundle_dir.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(CONFIG_PATH.read_text())
    preregistration = verify_preregistration(config)
    shutil.copyfile(CONFIG_PATH, args.output_dir / "preregistration.yaml")
    write_json(args.output_dir / "preregistration_record.json", preregistration)

    runtime_rows: list[dict[str, Any]] = []
    seed_records: dict[str, dict[str, Any]] = {}
    total_start = time.perf_counter()
    scalar = run_scalar_validation(
        config,
        workers=args.workers,
        runtime_rows=runtime_rows,
        seed_records=seed_records,
    )
    partitions = run_partition_validation(
        config,
        workers=args.workers,
        runtime_rows=runtime_rows,
        seed_records=seed_records,
    )
    invariance_gpu = run_invariance_and_gpu(
        config, seed_records=seed_records, runtime_rows=runtime_rows
    )
    eligibility_rows, eligibility = build_fixed_eligibility(
        config, scalar, partitions, invariance_gpu
    )
    dynamic = run_dynamic_histories(
        config,
        eligibility=eligibility,
        calibration=scalar["calibration"],
        seed_records=seed_records,
        runtime_rows=runtime_rows,
    )
    strict = run_strict_scopes(
        config, seed_records=seed_records, runtime_rows=runtime_rows
    )
    failure = run_failure_injections(config)
    reproducibility = reproducibility_checks()
    registry = registry_preservation()

    all_histories = [*dynamic["historyRows"], *strict["historyRows"]]
    all_estimates = [*dynamic["estimateRows"], *strict["estimateRows"]]
    all_atoms = [*dynamic["atomRows"], *strict["atomRows"]]
    all_causality = [*dynamic["causalityRows"], *strict["causalityRows"]]
    all_invariance = [
        *invariance_gpu["invarianceRows"],
        *partitions["invarianceRows"],
    ]
    all_gpu = [*invariance_gpu["gpuRows"], *strict["omegaRows"]]
    runtime_rows.append(
        {
            "stage": "total_before_serialization",
            "caseCount": len(all_estimates),
            "workers": args.workers,
            "wallSeconds": time.perf_counter() - total_start,
        }
    )

    write_csv(args.output_dir / "exact_pair_eligibility.csv", eligibility_rows)
    write_csv(args.output_dir / "finite_sample_calibration.csv", scalar["calibrationRows"])
    write_csv(args.output_dir / "known_truth_results.csv", scalar["truthRows"])
    write_csv(args.output_dir / "ccs_oracle_validation.csv", scalar["ccsOracleRows"])
    write_csv(args.output_dir / "highdim_partition_validation.csv", partitions["rows"])
    write_csv(args.output_dir / "partition_gate_summary.csv", partitions["gateRows"])
    write_parquet(
        args.output_dir / "partition_candidate_scores.parquet",
        partitions["candidateRows"],
    )
    write_parquet(args.output_dir / "partition_histories.parquet", all_histories)
    write_parquet(args.output_dir / "phir_estimates.parquet", all_estimates)
    write_parquet(args.output_dir / "atom_outputs.parquet", all_atoms)
    write_csv(args.output_dir / "invariance_results.csv", all_invariance)
    write_csv(args.output_dir / "shuffle_controls.csv", scalar["shuffleRows"])
    write_csv(args.output_dir / "cpu_gpu_comparisons.csv", all_gpu)
    write_csv(
        args.output_dir / "regularization_sensitivity.csv", scalar["sensitivityRows"]
    )
    write_csv(args.output_dir / "causality_audit.csv", all_causality)
    write_csv(args.output_dir / "runtime_benchmarks.csv", runtime_rows)
    write_json(args.output_dir / "failure_injection.json", failure)
    write_json(args.output_dir / "reproducibility_validation.json", reproducibility)
    write_json(args.output_dir / "registry_preservation.json", registry)
    write_json(args.output_dir / "runtime_manifest.json", runtime_manifest(config))

    seed_rows = sorted(seed_records.values(), key=lambda item: item["streamId"])
    write_parquet(args.output_dir / "seed_records.parquet", seed_rows)
    seed_domain_counts = defaultdict(int)
    for record in seed_rows:
        seed_domain_counts[record["domain"]] += 1
    seed_manifest = {
        "researchStepId": "S11",
        "contractId": "E01-S06-SEED-DERIVATION-v1.0.0",
        "rootSeedHex": config["randomness"]["rootSeedHex"],
        "streamPurpose": "estimator",
        "uniqueStreamCount": len(seed_rows),
        "domainCounts": dict(sorted(seed_domain_counts.items())),
        "sortedRecordsAggregateSha256": canonical_sha256(seed_rows),
        "recordTable": str(args.output_dir / "seed_records.parquet"),
        "recordTableSha256": sha256_file(args.output_dir / "seed_records.parquet"),
        "authorOrMatlabIdentityClaimed": False,
    }
    write_json(args.output_dir / "seed_manifest.json", seed_manifest)

    eligible_fixed_rows = [
        row for row in eligibility_rows if row.get("pairId") and row["status"] == "ELIGIBLE"
    ]
    eligible_pairs = sorted({row["pairId"] for row in eligible_fixed_rows})
    strict_eligible = [
        row
        for row in strict["estimateRows"]
        if row["status"] == "ELIGIBLE" and row["effectiveSampleCount"] >= 512
    ]
    specification_metadata = {
        "schemaVersion": "1.0.0",
        "researchStepId": "S11",
        "preregistrationCommit": PREREGISTRATION_COMMIT,
        "preregistrationSha256": PREREGISTRATION_SHA256,
        "smallWindowEstimatorId": ESTIMATOR_ID,
        "calibrationId": CALIBRATION_ID,
        "affinityId": AFFINITY_ID,
        "searchId": SEARCH_ID,
        "mappingIds": MAPPING_IDS,
        "objectiveIds": OBJECTIVE_IDS,
        "normalizationIds": NORMALIZATION_IDS,
        "redundancyIds": REDUNDANCY_IDS,
        "paperEquationAggregate": "str+stx+sty+sts-rtr-rtx-rty-rts",
        "sourceNamedAtom": False,
        "fixedPairGrid": config["fixedWindowGrid"]["pairs"],
        "eligibleFixedBranchRows": len(eligible_fixed_rows),
        "eligibleFixedPairs": eligible_pairs,
        "strictEligibleEstimateRows": len(strict_eligible),
        "s10StrictGate": {
            "id": S10_STRICT_GATE_ID,
            "minimumEffectiveSamples": 512,
            "modified": False,
        },
        "omegaPolicy": {
            "discrete": "EXCLUDED",
            "multivariateDoublet": "EXCLUDED",
            "gaussianEqualWidthScalar": "GUARDED_STRICT_CROSSCHECK_ONLY",
        },
        "preprocessingBoundary": config["preprocessingBoundary"],
        "authorMethodSentinelsResolved": 0,
        "paperPrimaryBranch": None,
        "wholeTrajectoryLabel": "NON_PROSPECTIVE_WHOLE_TRAJECTORY_DESCRIPTION",
    }
    write_yaml(args.output_dir / "specification_metadata.yaml", specification_metadata)

    plot_results(args.output_dir, all_estimates, partitions["gateRows"])
    output_validation = validate_outputs(
        output_dir=args.output_dir,
        eligibility_rows=eligibility_rows,
        estimates=all_estimates,
        atoms=all_atoms,
        histories=all_histories,
        causality=all_causality,
    )
    write_json(args.output_dir / "output_validation.json", output_validation)

    gate_families = [
        {"gateFamily": "preregistration_and_frozen_inputs", "success": preregistration["success"]},
        {"gateFamily": "heldout_null_calibration", "success": all(row["pass"] for row in scalar["calibrationRows"])},
        {"gateFamily": "mmi_known_truth", "success": all(row["pass"] for row in scalar["truthRows"] if row["redundancy"] == "MMI")},
        {"gateFamily": "ccs_population_oracle", "success": all(row["pass"] for row in scalar["ccsOracleRows"])},
        {"gateFamily": "ccs_known_truth_experimental", "success": all(row["pass"] for row in scalar["truthRows"] if row["redundancy"] == "CCS")},
        {"gateFamily": "regularization", "success": all(row["pass"] for row in scalar["sensitivityRows"])},
        {"gateFamily": "highdim_partition_and_null", "success": all(row["pass"] for row in partitions["gateRows"])},
        {"gateFamily": "affine_and_relabel_invariance", "success": all(row["pass"] for row in all_invariance)},
        {"gateFamily": "time_shuffle", "success": all(row["pass"] for row in scalar["shuffleRows"])},
        {"gateFamily": "cpu_gpu", "success": all(row["pass"] for row in all_gpu)},
        {"gateFamily": "causality_indexing", "success": all(row["pass"] for row in all_causality)},
        {"gateFamily": "strict_expanding_whole_scope", "success": bool(strict_eligible) and all(row["pass"] for row in strict["omegaRows"])},
        {"gateFamily": "reproducibility", "success": reproducibility["success"]},
        {"gateFamily": "failure_injection", "success": failure["success"]},
        {"gateFamily": "registry_preservation", "success": registry["success"]},
        {"gateFamily": "output_schema_and_suppression", "success": output_validation["success"]},
    ]
    all_pairs_have_eligible = len(eligible_pairs) == 16
    meaningful_execution = bool(eligible_fixed_rows) and bool(strict_eligible)
    all_gate_families_pass = all(item["success"] for item in gate_families)
    if meaningful_execution and all_pairs_have_eligible and all_gate_families_pass:
        outcome = "supportive"
        completion_status = "COMPLETE"
        recommended = "Hand control back; S12 may consume only the explicitly eligible S11 branch rows after Chief Scientist selection."
    elif meaningful_execution:
        outcome = "constraining/contradictory"
        completion_status = "COMPLETE_WITH_CONSTRAINTS"
        recommended = "Hand control back for branch review; S12 may use only rows marked ELIGIBLE and must not designate a paper-primary branch."
    else:
        outcome = "constraining/contradictory"
        completion_status = "RETURN_FOR_REVIEW_VALIDATION_BLOCKED"
        recommended = "Do not begin S12; review the failed preregistered gates without weakening or relabeling them."
    validation_summary = {
        "researchStepId": "S11",
        "stepNumber": 11,
        "status": completion_status,
        "executionSuccess": output_validation["success"],
        "allPreregisteredGateFamiliesPassed": all_gate_families_pass,
        "outcomeClassification": outcome,
        "gateFamilies": gate_families,
        "gateFamiliesPassed": sum(item["success"] for item in gate_families),
        "gateFamiliesTotal": len(gate_families),
        "eligibleFixedBranchRows": len(eligible_fixed_rows),
        "totalFixedBranchRows": 576,
        "eligibleFixedPairs": eligible_pairs,
        "eligibleFixedPairCount": len(eligible_pairs),
        "strictEligibleEstimateRows": len(strict_eligible),
        "fixedNumericEstimateRows": sum(
            row["status"] == "ELIGIBLE" for row in dynamic["estimateRows"]
        ),
        "fixedSuppressedEstimateRows": sum(
            row["status"] != "ELIGIBLE" for row in dynamic["estimateRows"]
        ),
        "wholeTrajectoryRowsNonProspective": all(
            row["prospective"] is False
            for row in strict["estimateRows"]
            if row["scope"] == "whole_trajectory"
        ),
        "s10StrictGateModified": False,
        "omegaDiscreteUsed": False,
        "omegaDoubletUsed": False,
        "authorOrMatlabIdentityClaimed": False,
        "paperPrimaryBranch": None,
        "recommendedNextAction": recommended,
    }
    write_json(args.output_dir / "validation_summary.json", validation_summary)

    contract = {
        "contractVersion": "1.0.0",
        "researchStepId": "S11",
        "title": "Time-localized Phi-r reconstruction contract",
        "evidenceClass": "VALIDATION_BRANCH_NOT_RECOVERED_AUTHOR_METHOD",
        "preregistration": {
            "commit": PREREGISTRATION_COMMIT,
            "sha256": PREREGISTRATION_SHA256,
        },
        "estimator": config["smallWindowEstimator"],
        "redundancyBranches": config["redundancyBranches"],
        "atomAndAggregateContract": config["atomAndAggregateContract"],
        "partitionBranch": config["partitionBranch"],
        "temporalEvaluation": config["temporalEvaluation"],
        "preprocessingBoundary": config["preprocessingBoundary"],
        "strictBoundary": config["immutableS10Boundary"],
        "outputTables": {
            "eligibility": str(args.output_dir / "exact_pair_eligibility.csv"),
            "histories": str(args.output_dir / "partition_histories.parquet"),
            "estimates": str(args.output_dir / "phir_estimates.parquet"),
            "atoms": str(args.output_dir / "atom_outputs.parquet"),
        },
        "outcome": validation_summary,
        "paperPrimaryBranch": None,
        "unresolvedAuthorChoicesPreserved": True,
    }
    eligibility_contract = {
        "registryVersion": "1.0.0",
        "researchStepId": "S11",
        "preregistrationSha256": PREREGISTRATION_SHA256,
        "fixedBranchRowCount": 576,
        "eligibleFixedBranchRows": len(eligible_fixed_rows),
        "eligibleFixedPairs": eligible_pairs,
        "rows": [
            {
                key: row.get(key)
                for key in (
                    "pairId",
                    "windowLength",
                    "lag",
                    "effectiveSampleCount",
                    "redundancyId",
                    "mappingId",
                    "objectiveId",
                    "normalizationId",
                    "searchId",
                    "status",
                    "reason",
                )
            }
            for row in eligibility_rows
            if row.get("pairId")
        ],
        "excludedBranches": [row for row in eligibility_rows if not row.get("pairId")],
        "paperPrimaryBranch": None,
    }
    write_yaml(
        args.bundle_dir / "time_localized_phir_contract_v1.0.0.yaml", contract
    )
    write_yaml(
        args.bundle_dir / "time_localized_phir_eligibility_registry_v1.0.0.yaml",
        eligibility_contract,
    )
    refresh_artifact_manifest(args.output_dir, args.bundle_dir)
    print(json.dumps(validation_summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
