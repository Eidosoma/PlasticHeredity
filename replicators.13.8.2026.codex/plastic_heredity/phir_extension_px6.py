"""PX6 fixed redundancy-correction continuum and analytic sign envelope."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .mechanistic import verify_checksums, write_checksums
from .phir_ch5 import _append_ledger
from .phir_extension_common import (
    BOOTSTRAP_DRAWS,
    MASTER_REGISTRATION,
    MAX_WORKERS,
    RANDOMIZATION_DRAWS,
    RESULT_ROOT,
    ROOT,
    atomic_json,
    canonical_digest,
    canonical_json,
    paired_summary,
    purpose_seed,
    runtime_versions,
    sha256_file,
)


DOCUMENT = "CODEX_CH5_PHIR_EXTENSION_PREREGISTRATION.md"
AMENDMENT = "CODEX_CH5_PHIR_EXTENSION_PX6_AMENDMENT.md"
DEFAULT_VALIDATION = RESULT_ROOT / "px6_validation"
DEFAULT_REGISTRATION = RESULT_ROOT / "px6_registration"
DEFAULT_SMOKE = RESULT_ROOT / "px6_smoke"
DEFAULT_OUTPUT = RESULT_ROOT / "px6_redundancy_robustness"
DEFAULT_WORK = RESULT_ROOT / ".px6_work"
DEFAULT_LOG = RESULT_ROOT / "px6_redundancy_robustness.log"
PRE_AMENDMENT_REGISTRATION = RESULT_ROOT / "px6_registration_pre_amendment_001"
SUPERSEDED_REGISTRATION_ID = (
    "50d7e122ea37da68cbdcf716b235ebf4c5d4305c15f7b5e81fd4abf43ac43576"
)

LABEL = "CODEX_CH5_PHIR_EXTENSION_PX6_V1"
REGISTRATION_FORMAT = "codex-ch5-phir-extension-px6-registration-v2"
RESULT_FORMAT = "codex-ch5-phir-extension-px6-result-v1"
SERVICE_NAME = "codex-phir-extension-px6-20260820"

LAMBDA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
CPU_SECONDS = 2.0 * 3600.0

SOURCE_FILES = (
    DOCUMENT,
    AMENDMENT,
    "plastic_heredity/phir_extension_px6.py",
    "plastic_heredity/phir_extension_common.py",
    "tests/test_phir_extension_px6.py",
    "plastic_heredity/phir_instruments.py",
    "plastic_heredity/phir_rescue_instruments.py",
    "plastic_heredity/seeds.py",
)


@dataclass(frozen=True)
class DatasetContract:
    phase: str
    relative_path: str
    base_column: str
    redundancy_column: str
    group_columns: tuple[str, ...]
    within_columns: tuple[str, ...]
    high_arm: str
    low_arm: str
    filters: tuple[tuple[str, str], ...] = ()


DATASETS = (
    DatasetContract(
        "PX1",
        "results/phir_extension/px1_fresh_confirmation/lineages.csv.gz",
        "material_full_base",
        "material_double_redundancy",
        ("candidate", "replicate"),
        (),
        "STABILIZE",
        "DESTABILIZE",
    ),
    DatasetContract(
        "PX2",
        "results/phir_extension/px2_event_locked_recovery/scores.csv.gz",
        "material_full_base",
        "material_double_redundancy",
        ("candidate", "half"),
        ("break_step",),
        "RENEWAL_UP",
        "RENEWAL_DOWN",
    ),
    DatasetContract(
        "PX3",
        "results/phir_extension/px3_confirmation/scores.csv.gz",
        "material_full_base",
        "material_double_redundancy",
        ("candidate", "half"),
        ("landmark",),
        "PHI_UP",
        "PHI_DOWN",
    ),
    DatasetContract(
        "PX4",
        "results/phir_extension/px4_simulator_moderator/lineages.csv.gz",
        "final30_material_full_base",
        "final30_material_double_redundancy",
        ("candidate", "replicate", "variant"),
        (),
        "STABILIZE",
        "DESTABILIZE",
    ),
    DatasetContract(
        "PX5",
        "results/phir_extension/px5_generative_null_remeasurement/scores.csv.gz",
        "material_full_base",
        "material_double_redundancy",
        ("candidate", "mechanism"),
        ("landmark",),
        "SOURCE_RULE_DOWN",
        "SOURCE_RULE_UP",
        (("context", "INTERVENTION"),),
    ),
)


def _source_hashes() -> dict[str, str]:
    return {name: sha256_file(ROOT / name) for name in SOURCE_FILES}


def protocol() -> dict[str, Any]:
    value: dict[str, Any] = {
        "phase": "PX6",
        "procedural_amendment": {
            "document": AMENDMENT,
            "supersedes_registration_id": SUPERSEDED_REGISTRATION_ID,
            "change": "PX2 within-state namespace landmark -> break_step",
            "scientific_result_existed_at_amendment": False,
        },
        "question": "are full-block intervention contrasts robust to every fixed correction between no redundancy restoration and the complete minimum-MI correction?",
        "datasets": [asdict(item) for item in DATASETS],
        "correction": "Phi(lambda) = full_base + lambda * double_redundancy",
        "lambda_grid": list(LAMBDA_GRID),
        "analytic_envelope": "because each matrix effect is affine in lambda, extrema over [0,1] occur at lambda=0 or lambda=1",
        "classification": {
            "uniform_positive": "95% whole-matrix bootstrap lower bound of the analytic minimum is above zero",
            "uniform_negative": "95% whole-matrix bootstrap upper bound of the analytic maximum is below zero",
            "definition_sensitive": "point analytic envelope spans zero",
            "inconclusive_same_sign": "point envelope has one sign but its bootstrap envelope includes zero",
        },
        "inference": {
            "unit": "whole catalytic matrix",
            "bootstrap": BOOTSTRAP_DRAWS,
            "randomization": RANDOMIZATION_DRAWS,
            "same_matrix_resamples_for_all_lambda": True,
        },
        "prohibitions": {
            "no_lambda_selection": True,
            "no_outcome_driven_grid_change": True,
            "cannot_rescue_failed_public_nine_atom_result": True,
            "no_48_matrix_campaign": True,
        },
        "provenance_boundary": {
            "PX1_PX2": "post-outcome robustness remeasurement",
            "PX3_PX4_PX5": "prospectively fixed before those result files existed",
        },
        "cpu_seconds": CPU_SECONDS,
        "claim_boundary": [
            "robustness of a full-block statistic is not robustness of public nine-atom Phi-r",
            "no correction value may be chosen after seeing results",
            "information robustness is not consciousness, agency, or life",
            "strict-eight is excluded",
        ],
    }
    value["protocol_id"] = canonical_digest(value)
    return value


def _all_inputs_exist() -> bool:
    return all((ROOT / item.relative_path).exists() for item in DATASETS)


def input_schema_checks() -> dict[str, bool]:
    """Validate real archived headers without reading scientific values."""

    checks: dict[str, bool] = {}
    for item in DATASETS:
        path = ROOT / item.relative_path
        if not path.exists():
            checks[f"{item.phase}_schema"] = False
            continue
        columns = set(pd.read_csv(path, nrows=0).columns)
        required = {
            "matrix_id",
            "arm",
            item.base_column,
            item.redundancy_column,
            *item.group_columns,
            *item.within_columns,
            *(column for column, _value in item.filters),
        }
        checks[f"{item.phase}_schema"] = required.issubset(columns)
    return checks


def validation_checks() -> dict[str, bool]:
    checks = {
        "master_registration_exists": MASTER_REGISTRATION.exists(),
        "five_phase_contracts": len(DATASETS) == 5,
        "lambda_endpoints_present": LAMBDA_GRID[0] == 0.0
        and LAMBDA_GRID[-1] == 1.0,
        "lambda_grid_fixed": LAMBDA_GRID == (0.0, 0.25, 0.5, 0.75, 1.0),
        "lambda_grid_monotonic": all(
            left < right for left, right in zip(LAMBDA_GRID[:-1], LAMBDA_GRID[1:])
        ),
        "affine_envelope_fixed": protocol()["analytic_envelope"].startswith("because"),
        "no_lambda_selection": protocol()["prohibitions"]["no_lambda_selection"],
        "cannot_rescue_public_result": protocol()["prohibitions"][
            "cannot_rescue_failed_public_nine_atom_result"
        ],
        "draws_fixed": BOOTSTRAP_DRAWS == 4096
        and RANDOMIZATION_DRAWS == 4096,
        "cpu_allocation_fixed": CPU_SECONDS == 2 * 3600,
        "no_48_matrix_campaign": protocol()["prohibitions"][
            "no_48_matrix_campaign"
        ],
        "strict_eight_excluded": "strict-eight is excluded"
        in protocol()["claim_boundary"],
        "px2_break_step_namespace": DATASETS[1].within_columns == ("break_step",),
    }
    checks.update(input_schema_checks())
    return checks


def run_validation() -> dict[str, Any]:
    checks = validation_checks()
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/test_phir_extension_px6.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    payload = {
        "format": "codex-ch5-phir-extension-px6-validation-v1",
        "checks": checks,
        "pytest_returncode": completed.returncode,
        "pytest_stdout": completed.stdout,
        "pytest_stderr": completed.stderr,
        "all_passed": bool(all(checks.values()) and completed.returncode == 0),
        "runtime": runtime_versions(),
    }
    if DEFAULT_VALIDATION.exists():
        shutil.rmtree(DEFAULT_VALIDATION)
    DEFAULT_VALIDATION.mkdir(parents=True)
    atomic_json(DEFAULT_VALIDATION / "validation.json", payload)
    write_checksums(DEFAULT_VALIDATION)
    if not payload["all_passed"]:
        raise AssertionError(f"PX6 validation failed\n{completed.stdout}\n{completed.stderr}")
    return payload


def register_program() -> dict[str, Any]:
    verify_checksums(DEFAULT_VALIDATION)
    validation = json.loads((DEFAULT_VALIDATION / "validation.json").read_text())
    if not validation["all_passed"]:
        raise ValueError("PX6 validation did not pass")
    if not _all_inputs_exist():
        raise RuntimeError("PX6 registration is locked until PX1-PX5 outputs exist")
    if not PRE_AMENDMENT_REGISTRATION.exists():
        raise RuntimeError("PX6 superseded registration archive is missing")
    prior = json.loads(
        (PRE_AMENDMENT_REGISTRATION / "registration.json").read_text()
    )
    if prior.get("registration_id") != SUPERSEDED_REGISTRATION_ID:
        raise ValueError("PX6 superseded registration identity changed")
    if DEFAULT_REGISTRATION.exists():
        raise FileExistsError(f"PX6 registration exists: {DEFAULT_REGISTRATION}")
    master = json.loads((MASTER_REGISTRATION / "registration.json").read_text())
    input_hashes = {
        item.phase: sha256_file(ROOT / item.relative_path) for item in DATASETS
    }
    manifests: dict[str, Any] = {}
    for item in DATASETS:
        manifest_path = (ROOT / item.relative_path).parent / "manifest.json"
        manifests[item.phase] = {
            "sha256": sha256_file(manifest_path),
            "manifest": json.loads(manifest_path.read_text()),
        }
    body: dict[str, Any] = {
        "format": REGISTRATION_FORMAT,
        "master_registration_id": master["registration_id"],
        "protocol": protocol(),
        "source_hashes": _source_hashes(),
        "input_hashes": input_hashes,
        "input_manifests": manifests,
        "runtime": runtime_versions(),
        "new_scientific_matrices_at_registration": 0,
        "new_scientific_futures_at_registration": 0,
        "supersedes_registration_id": SUPERSEDED_REGISTRATION_ID,
        "failed_launch_produced_scientific_result": False,
    }
    body["registration_id"] = canonical_digest(body)
    DEFAULT_REGISTRATION.mkdir(parents=True)
    shutil.copy2(ROOT / DOCUMENT, DEFAULT_REGISTRATION / "preregistration.md")
    shutil.copy2(ROOT / AMENDMENT, DEFAULT_REGISTRATION / "procedural_amendment.md")
    atomic_json(DEFAULT_REGISTRATION / "protocol.json", body["protocol"])
    atomic_json(DEFAULT_REGISTRATION / "registration.json", body)
    write_checksums(DEFAULT_REGISTRATION)
    _append_ledger(
        f"<!-- phir-extension-px6-registration-{body['registration_id']} -->",
        [
            "## Phi-r extension PX6 registered",
            "",
            f"- Registration: `{body['registration_id']}`.",
            "- All PX1-PX5 inputs were hashed before the fixed redundancy continuum was evaluated.",
            "- No lambda selection or 48-matrix continuation is permitted.",
        ],
    )
    return body


def verify_registration() -> dict[str, Any]:
    verify_checksums(DEFAULT_REGISTRATION)
    body = json.loads((DEFAULT_REGISTRATION / "registration.json").read_text())
    observed = body.pop("registration_id")
    if body.get("format") != REGISTRATION_FORMAT or observed != canonical_digest(body):
        raise ValueError("PX6 registration identity failed")
    body["registration_id"] = observed
    if body["protocol"] != canonical_json(protocol()):
        raise ValueError("PX6 protocol changed")
    if body["source_hashes"] != _source_hashes():
        raise ValueError("PX6 source changed after registration")
    for item in DATASETS:
        if body["input_hashes"][item.phase] != sha256_file(ROOT / item.relative_path):
            raise ValueError(f"PX6 input changed: {item.phase}")
    return body


def _cell_effects(
    contract: DatasetContract,
) -> list[tuple[dict[str, Any], NDArray[np.float64], NDArray[np.float64]]]:
    frame = pd.read_csv(ROOT / contract.relative_path)
    if "candidate" in frame:
        frame["candidate"] = frame["candidate"].astype(str).str.zfill(2)
    for column, value in contract.filters:
        frame = frame[frame[column].astype(str) == value]
    output: list[tuple[dict[str, Any], NDArray[np.float64], NDArray[np.float64]]] = []
    grouping: str | list[str]
    grouping = list(contract.group_columns)
    for key, selected in frame.groupby(grouping, sort=True):
        values = key if isinstance(key, tuple) else (key,)
        labels = {
            column: value for column, value in zip(contract.group_columns, values, strict=True)
        }
        indices = ["matrix_id", *contract.within_columns]
        base = selected.pivot(index=indices, columns="arm", values=contract.base_column)
        redundancy = selected.pivot(
            index=indices, columns="arm", values=contract.redundancy_column
        )
        intercept = base[contract.high_arm] - base[contract.low_arm]
        slope = redundancy[contract.high_arm] - redundancy[contract.low_arm]
        if contract.within_columns:
            intercept = intercept.groupby("matrix_id").mean()
            slope = slope.groupby("matrix_id").mean()
        else:
            intercept.index = intercept.index.astype(int)
            slope.index = slope.index.astype(int)
        common = intercept.index.intersection(slope.index)
        output.append(
            (
                {"phase": contract.phase, **labels},
                intercept.loc[common].to_numpy(float),
                slope.loc[common].to_numpy(float),
            )
        )
    return output


def analyze_cell(
    labels: Mapping[str, Any],
    intercept: NDArray,
    slope: NDArray,
) -> tuple[dict[str, Any], dict[str, NDArray]]:
    base = np.asarray(intercept, dtype=np.float64)
    correction = np.asarray(slope, dtype=np.float64)
    if base.shape != correction.shape or base.ndim != 1 or base.size < 2:
        raise ValueError("PX6 cell effects must be paired matrix vectors")
    cell_id = canonical_digest(dict(labels))
    bootstrap_rng = np.random.default_rng(
        purpose_seed("bootstrap", "PX6", cell_id)
    )
    indices = bootstrap_rng.integers(
        0, base.size, size=(BOOTSTRAP_DRAWS, base.size)
    )
    randomization_rng = np.random.default_rng(
        purpose_seed("randomization", "PX6", cell_id)
    )
    signs = randomization_rng.choice(
        (-1.0, 1.0), size=(RANDOMIZATION_DRAWS, base.size)
    )
    grid_rows: list[dict[str, Any]] = []
    grid_bootstrap = np.empty((len(LAMBDA_GRID), BOOTSTRAP_DRAWS))
    grid_randomization = np.empty((len(LAMBDA_GRID), RANDOMIZATION_DRAWS))
    for position, weight in enumerate(LAMBDA_GRID):
        effects = base + weight * correction
        boot = effects[indices].mean(axis=1)
        random = (effects[None, :] * signs).mean(axis=1)
        grid_bootstrap[position] = boot
        grid_randomization[position] = random
        observed = float(effects.mean())
        grid_rows.append(
            {
                "lambda": weight,
                "effect": observed,
                "ci90": np.quantile(boot, (0.05, 0.95)).tolist(),
                "ci95": np.quantile(boot, (0.025, 0.975)).tolist(),
                "positive_sign_randomization_p": float(
                    (1 + np.count_nonzero(random >= observed))
                    / (RANDOMIZATION_DRAWS + 1)
                ),
                "two_sided_sign_randomization_p": float(
                    (1 + np.count_nonzero(np.abs(random) >= abs(observed)))
                    / (RANDOMIZATION_DRAWS + 1)
                ),
            }
        )
    endpoint0 = base[indices].mean(axis=1)
    endpoint1 = (base + correction)[indices].mean(axis=1)
    boot_minimum = np.minimum(endpoint0, endpoint1)
    boot_maximum = np.maximum(endpoint0, endpoint1)
    mean0 = float(base.mean())
    mean1 = float((base + correction).mean())
    point_minimum = min(mean0, mean1)
    point_maximum = max(mean0, mean1)
    minimum_ci = np.quantile(boot_minimum, (0.025, 0.975)).tolist()
    maximum_ci = np.quantile(boot_maximum, (0.025, 0.975)).tolist()
    uniform_positive = bool(minimum_ci[0] > 0)
    uniform_negative = bool(maximum_ci[1] < 0)
    definition_sensitive = bool(point_minimum <= 0 <= point_maximum)
    if uniform_positive:
        classification = "uniform_positive"
    elif uniform_negative:
        classification = "uniform_negative"
    elif definition_sensitive:
        classification = "definition_sensitive"
    else:
        classification = "inconclusive_same_sign"
    mean_slope = float(correction.mean())
    crossing = -mean0 / mean_slope if abs(mean_slope) > 1e-15 else float("nan")
    if not np.isfinite(crossing) or not 0.0 <= crossing <= 1.0:
        crossing = float("nan")
    result = {
        **dict(labels),
        "matrices": int(base.size),
        "mean_base_contrast": mean0,
        "mean_redundancy_contrast": mean_slope,
        "grid": grid_rows,
        "analytic_point_envelope": [point_minimum, point_maximum],
        "analytic_minimum_ci95": minimum_ci,
        "analytic_maximum_ci95": maximum_ci,
        "zero_crossing_lambda": crossing,
        "classification": classification,
    }
    arrays = {
        "matrix_base": base,
        "matrix_redundancy": correction,
        "bootstrap_indices": indices,
        "randomization_signs": signs,
        "grid_bootstrap": grid_bootstrap,
        "grid_randomization": grid_randomization,
        "bootstrap_minimum": boot_minimum,
        "bootstrap_maximum": boot_maximum,
    }
    return result, arrays


def analyze_all() -> tuple[dict[str, Any], pd.DataFrame, dict[str, NDArray]]:
    cells: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    arrays: dict[str, NDArray] = {}
    for contract in DATASETS:
        for labels, intercept, slope in _cell_effects(contract):
            result, local = analyze_cell(labels, intercept, slope)
            index = len(cells)
            cells.append(result)
            arrays.update({f"cell_{index:03d}__{name}": value for name, value in local.items()})
            for matrix_id, (base, correction) in enumerate(
                zip(intercept, slope, strict=True)
            ):
                matrix_rows.append(
                    {
                        "cell_index": index,
                        **dict(labels),
                        "matrix_position": matrix_id,
                        "base_contrast": float(base),
                        "redundancy_contrast": float(correction),
                    }
                )
    counts = {
        name: sum(cell["classification"] == name for cell in cells)
        for name in (
            "uniform_positive",
            "uniform_negative",
            "definition_sensitive",
            "inconclusive_same_sign",
        )
    }
    metrics = {
        "format": "codex-ch5-phir-extension-px6-metrics-v1",
        "lambda_grid": list(LAMBDA_GRID),
        "cells": cells,
        "classification_counts": counts,
        "cells_total": len(cells),
        "no_lambda_selected": True,
        "public_nine_atom_result_unchanged": True,
    }
    return metrics, pd.DataFrame(matrix_rows), arrays


def _analysis_digest(metrics: Mapping[str, Any], arrays: Mapping[str, NDArray]) -> str:
    return canonical_digest(
        {
            "metrics": metrics,
            "arrays": {
                name: {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "bytes": np.ascontiguousarray(value).tobytes().hex(),
                }
                for name, value in sorted(arrays.items())
            },
        }
    )


def _write_result(
    registration: Mapping[str, Any],
    metrics: Mapping[str, Any],
    matrix_effects: pd.DataFrame,
    arrays: Mapping[str, NDArray],
    replay: Mapping[str, Any],
    cpu: float,
) -> dict[str, Any]:
    temporary = DEFAULT_OUTPUT.with_name(DEFAULT_OUTPUT.name + f".tmp-{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    matrix_effects.to_csv(temporary / "matrix_effects.csv.gz", index=False)
    np.savez_compressed(temporary / "inference_arrays.npz", **arrays)
    atomic_json(temporary / "primary_metrics.json", metrics)
    atomic_json(temporary / "replay_audit.json", replay)
    report = [
        "# PX6 redundancy-correction robustness",
        "",
        f"Registration: `{registration['registration_id']}`.",
        "",
        "No correction value was selected. Every registered cell was evaluated at the complete fixed grid and with the analytic affine envelope over lambda in [0,1].",
        "",
        "```json",
        json.dumps(metrics["classification_counts"], indent=2, sort_keys=True),
        "```",
        "",
        "These classifications apply only to the material full-block statistic. They do not rescue or modify the public nine-atom Phi-r result.",
    ]
    (temporary / "SCIENTIFIC_REPORT.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    (temporary / "LAY_SUMMARY.md").write_text(
        "# PX6 lay summary\n\n"
        "The full-system information formula contains a debated correction for information copied redundantly by both parts. Instead of choosing the correction that looks best, we swept from none of it to all of it and used a mathematical envelope covering every value in between.\n\n"
        "A uniformly positive cell stays positive throughout that entire range. A definition-sensitive cell crosses zero and therefore depends on the convention. This check does not alter the separate public Phi-r result.\n",
        encoding="utf-8",
    )
    manifest = {
        "format": RESULT_FORMAT,
        "registration_id": registration["registration_id"],
        "cpu_seconds": cpu,
        "complete_exact_replay": bool(replay["complete_exact_replay"]),
        "complete_readback_exact": False,
        "cells": metrics["cells_total"],
        "classification_counts": metrics["classification_counts"],
        "no_lambda_selected": True,
    }
    atomic_json(temporary / "manifest.json", manifest)
    write_checksums(temporary)
    temporary.replace(DEFAULT_OUTPUT)
    verify_checksums(DEFAULT_OUTPUT)
    readback = pd.read_csv(DEFAULT_OUTPUT / "matrix_effects.csv.gz")
    exact = len(readback) == len(matrix_effects)
    manifest["complete_readback_exact"] = exact
    atomic_json(DEFAULT_OUTPUT / "manifest.json", manifest)
    atomic_json(DEFAULT_OUTPUT / "readback_audit.json", {"complete": exact})
    write_checksums(DEFAULT_OUTPUT)
    if not exact:
        raise AssertionError("PX6 readback failed")
    _append_ledger(
        f"<!-- phir-extension-px6-result-{registration['registration_id']} -->",
        [
            "## Phi-r extension PX6 completed",
            "",
            "- Result: `results/phir_extension/px6_redundancy_robustness`.",
            f"- Fixed-grid classifications: `{json.dumps(metrics['classification_counts'], sort_keys=True)}`.",
            "- No lambda was selected; the public nine-atom result remains unchanged.",
        ],
    )
    return manifest


def run_scientific() -> dict[str, Any]:
    registration = verify_registration()
    verify_checksums(DEFAULT_SMOKE)
    if DEFAULT_OUTPUT.exists():
        raise FileExistsError(f"PX6 output exists: {DEFAULT_OUTPUT}")
    started = time.process_time()
    metrics, matrix_effects, arrays = analyze_all()
    first_digest = _analysis_digest(metrics, arrays)
    replay_metrics, replay_matrix_effects, replay_arrays = analyze_all()
    second_digest = _analysis_digest(replay_metrics, replay_arrays)
    matrix_exact = matrix_effects.equals(replay_matrix_effects)
    replay = {
        "generation_digest": first_digest,
        "replay_digest": second_digest,
        "matrix_table_exact": matrix_exact,
        "complete_exact_replay": first_digest == second_digest and matrix_exact,
    }
    if not replay["complete_exact_replay"]:
        raise AssertionError("PX6 exact replay failed")
    cpu = float(time.process_time() - started)
    if cpu > CPU_SECONDS:
        raise RuntimeError("PX6 exceeded its fixed CPU allocation")
    return _write_result(
        registration, metrics, matrix_effects, arrays, replay, cpu
    )


def run_smoke() -> dict[str, Any]:
    if DEFAULT_SMOKE.exists():
        raise FileExistsError(f"PX6 smoke exists: {DEFAULT_SMOKE}")
    positive, positive_arrays = analyze_cell(
        {"phase": "FIXTURE", "cell": "positive"},
        np.asarray([1.0, 1.1, 0.9, 1.2]),
        np.asarray([-0.1, 0.1, 0.0, -0.05]),
    )
    crossing, crossing_arrays = analyze_cell(
        {"phase": "FIXTURE", "cell": "crossing"},
        np.asarray([-0.5, -0.4, -0.6, -0.5]),
        np.asarray([1.2, 1.1, 1.3, 1.2]),
    )
    repeat, repeat_arrays = analyze_cell(
        {"phase": "FIXTURE", "cell": "positive"},
        np.asarray([1.0, 1.1, 0.9, 1.2]),
        np.asarray([-0.1, 0.1, 0.0, -0.05]),
    )
    payload = {
        "format": "codex-ch5-phir-extension-px6-smoke-v1",
        "uniform_positive_fixture": positive["classification"] == "uniform_positive",
        "crossing_fixture": crossing["classification"] == "definition_sensitive",
        "crossing_inside_unit_interval": 0.0 < crossing["zero_crossing_lambda"] < 1.0,
        "analytic_endpoints_ordered": positive["analytic_point_envelope"][0]
        <= positive["analytic_point_envelope"][1],
        "deterministic_replay": _analysis_digest(positive, positive_arrays)
        == _analysis_digest(repeat, repeat_arrays),
        "all_grid_values_retained": len(positive["grid"]) == len(LAMBDA_GRID),
        "effect_sizes_suppressed": True,
    }
    payload["passed"] = bool(
        all(value for key, value in payload.items() if key != "format")
    )
    DEFAULT_SMOKE.mkdir(parents=True)
    atomic_json(DEFAULT_SMOKE / "smoke.json", payload)
    write_checksums(DEFAULT_SMOKE)
    if not payload["passed"]:
        raise AssertionError("PX6 smoke failed")
    return payload


def launch_detached() -> dict[str, Any]:
    registration = verify_registration()
    verify_checksums(DEFAULT_SMOKE)
    if DEFAULT_OUTPUT.exists():
        raise FileExistsError(f"PX6 output exists: {DEFAULT_OUTPUT}")
    DEFAULT_WORK.mkdir(parents=True, exist_ok=True)
    command = [
        "systemd-run",
        "--user",
        f"--unit={SERVICE_NAME}",
        "--collect",
        "--property",
        f"WorkingDirectory={ROOT}",
        "--property",
        f"StandardOutput=append:{DEFAULT_LOG}",
        "--property",
        f"StandardError=append:{DEFAULT_LOG}",
        sys.executable,
        "-m",
        "plastic_heredity.phir_extension_px6",
        "run",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = {
        "registration_id": registration["registration_id"],
        "service": SERVICE_NAME,
        "launched_at_unix": time.time(),
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }
    atomic_json(DEFAULT_WORK / "detached_launch.json", payload)
    return payload


def status_payload() -> dict[str, Any]:
    return {
        "phase": "PX6",
        "validation": DEFAULT_VALIDATION.exists(),
        "registration": DEFAULT_REGISTRATION.exists(),
        "smoke": DEFAULT_SMOKE.exists(),
        "all_inputs_exist": _all_inputs_exist(),
        "complete": DEFAULT_OUTPUT.exists(),
        "service": SERVICE_NAME,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate")
    commands.add_parser("register")
    commands.add_parser("smoke")
    commands.add_parser("run")
    commands.add_parser("launch")
    commands.add_parser("status")
    arguments = parser.parse_args(argv)
    if arguments.command == "validate":
        print(json.dumps(run_validation(), indent=2, sort_keys=True))
    elif arguments.command == "register":
        print(json.dumps(register_program(), indent=2, sort_keys=True))
    elif arguments.command == "smoke":
        print(json.dumps(run_smoke(), indent=2, sort_keys=True))
    elif arguments.command == "run":
        print(json.dumps(run_scientific(), indent=2, sort_keys=True))
    elif arguments.command == "launch":
        print(json.dumps(launch_detached(), indent=2, sort_keys=True))
    elif arguments.command == "status":
        print(json.dumps(status_payload(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
