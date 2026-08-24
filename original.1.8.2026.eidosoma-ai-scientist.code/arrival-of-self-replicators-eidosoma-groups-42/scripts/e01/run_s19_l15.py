#!/usr/bin/env python3
"""Execute E01/S19-L15 untouched padding/length panel discrimination.

The runner is staged.  ``prepare`` is run only after the repository method
contract is committed and pushed.  ``generate`` creates the new seed-firewalled
cohort, ``features`` materializes the exact registered feature tensors,
``execute`` fits the frozen S16 models and controls, and ``finalize`` performs
full regeneration, reporting, hashing, and the mandatory human-review handoff.
Large trajectory/tensor/model intermediates remain in ``/cache``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import platform
import re
import resource
import shutil
import subprocess
import time
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
import yaml
from scipy import stats
from sklearn.linear_model import LogisticRegression

from e01_frozen_timebase_ensemble.core import frozen_clr, selected_clock_observations
from e01_latent_timebase.core import (
    ExposureDefinition,
    SimulationDefinition,
    derive_seed,
    generate_beta,
    initialize_distinct_state,
    simulate_trajectory,
)
from e01_latent_timebase.core import (
    array_sha256 as simulator_array_sha256,
)
from e01_pigozzi_source_audit.core import SourceImplementation
from e01_prediction_reconstruction.core import (
    EXPECTED_PARAMETER_COUNT,
    MAX_INPUT_LENGTH,
    MAX_TARGET_LENGTH,
    apply_channel_scaler,
    fit_channel_scaler,
    parameter_count,
    predict_probabilities,
    train_masked_mlp,
)
from e01_s19_figure5_prediction.core import (
    build_feature,
    extended_binary_metrics,
    source_values,
)
from e01_s19_padding_length_discrimination.core import (
    B1,
    B2,
    B3,
    B4,
    CANDIDATE_IDS,
    D0,
    D1,
    D2,
    D3,
    LEARNED_FEATURES,
    MASK_CONTRACT,
    MATRIX_COUNT,
    P1,
    P2,
    PAPER_FEATURES,
    REPETITIONS,
    RESEARCH_STEP_ID,
    S00,
    S01,
    S10,
    S11,
    VERSION,
    accuracy_decomposition,
    array_sha256,
    build_split_manifest,
    incoming_h,
    infer_output_length,
    normalized_compositions,
    padding_identity,
    seed128,
    split_indices,
    torch_seed,
)
from e01_source_emergence_metric_identity.core import (
    result_replay_equal,
    run_emergence_pipeline,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPO_ROOT.parent
CONFIG_PATH = REPO_ROOT / "configs/e01/s19_l15_untouched_padding_panel.yaml"
AMENDMENT_001_PATH = REPO_ROOT / "configs/e01/s19_l15_technical_amendment_001.json"
AMENDMENT_002_PATH = REPO_ROOT / "configs/e01/s19_l15_technical_amendment_002.json"
S16_MODEL_LOCK = REPO_ROOT / "configs/e01/s16_tensor_model_manifest.json"
S16_CORE = REPO_ROOT / "src/e01_prediction_reconstruction/core.py"
L14_ROOT = Path("/artifacts/research_steps/S19/loops/L14")
S19_ROOT = Path("/artifacts/research_steps/S19")
OUTPUT_ROOT = S19_ROOT / "loops/L15"
CACHE_ROOT = Path("/cache/e01_s19_l15")
PRIMARY_TRAJECTORY_CACHE = CACHE_ROOT / "primary_trajectories"
REGEN_TRAJECTORY_CACHE = CACHE_ROOT / "regenerated_trajectories"
TRAJECTORY_FEATURE_CACHE = CACHE_ROOT / "trajectory_features"
REGEN_FEATURE_CACHE = CACHE_ROOT / "regenerated_features"
TENSOR_ROOT = CACHE_ROOT / "tensors"
MODEL_CACHE = CACHE_ROOT / "model_results"
FIGURE_REGEN_CACHE = CACHE_ROOT / "figure_regeneration"
SAFE_LATTICE = Path("/artifacts/research_steps/S12B/safe_phi_lattice.json")
PAPER_PDF = Path("/cache/e01_s03/downloads/paper-2607.28250v1.pdf")
PAPER_FIGURE5 = (
    WORKSPACE_ROOT
    / "input-attachments/ed5486bf-a043-485b-a233-d88d8d123759/figures/figure-05.png"
)
PHIRL = SourceImplementation.PHIRL

REQUIRED_ARTIFACTS = (
    "preregistration.yaml",
    "decision_record.md",
    "implementation_lock.json",
    "source_snapshot_manifest.json",
    "immutable_prior_validation.json",
    "seed_firewall.json",
    "seed_manifest.parquet",
    "input_manifest.json",
    "matrix_input_manifest.parquet",
    "split_manifest.parquet",
    "fixture_manifest.json",
    "fixture_results.parquet",
    "preoutcome_benchmark.json",
    "execution_status.parquet",
    "trajectory_manifest.parquet",
    "trajectory_length_results.parquet",
    "padding_geometry_results.parquet",
    "prevalence_decomposition.parquet",
    "padded_target_manifest.parquet",
    "feature_manifest.parquet",
    "source_execution_results.parquet",
    "suffix_invariance_results.parquet",
    "training_history.parquet",
    "prediction_results.parquet",
    "all_cell_metrics.parquet",
    "valid_cell_metrics.parquet",
    "padding_cell_metrics.parquet",
    "accuracy_decomposition.parquet",
    "diagnostic_results.parquet",
    "negative_control_results.parquet",
    "paper_boxplot_comparison.csv",
    "paper_model_order_results.csv",
    "paired_model_comparisons.parquet",
    "bootstrap_results.parquet",
    "scientific_gate_results.parquet",
    "classification.json",
    "technical_amendment_001.json",
    "technical_amendment_002.json",
    "technical_amendment_ledger.csv",
    "failure_ledger.csv",
    "runtime_manifest.json",
    "storage_validation.json",
    "regeneration_validation.json",
    "artifact_manifest.json",
    "loop_decision_summary.md",
    "S19_L15_FULL_RESULTS.md",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_json(path: Path, value: object) -> None:
    atomic_text(path, canonical_json(json_safe(value)) + "\n")


def write_yaml(path: Path, value: object) -> None:
    atomic_text(path, yaml.safe_dump(json_safe(value), sort_keys=False))


def canonicalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in output.columns:
        if output[column].dtype == object:
            output[column] = output[column].map(
                lambda value: (
                    canonical_json(json_safe(value))
                    if isinstance(value, (dict, list, tuple, np.ndarray))
                    else value
                )
            )
    return output


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    canonicalize_frame(frame).to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    atomic_text(
        path, canonicalize_frame(frame).to_csv(index=False, lineterminator="\n")
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def repository_lock(*, require_clean: bool = True) -> dict[str, Any]:
    head = run_git("rev-parse", "HEAD")
    remote = run_git("rev-parse", "origin/eidosoma/groups/42")
    status = run_git("status", "--short")
    payload = {
        "branch": run_git("branch", "--show-current"),
        "head": head,
        "remoteHead": remote,
        "cleanWorktree": not bool(status),
        "worktreeStatus": status,
        "passed": bool(
            head == remote
            and run_git("branch", "--show-current") == "eidosoma/groups/42"
            and (not require_clean or not status)
        ),
    }
    return payload


def load_config() -> dict[str, Any]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if (
        config["versionedStepId"] != VERSION
        or config["researchStepId"] != RESEARCH_STEP_ID
    ):
        raise RuntimeError("L15 config identity mismatch")
    if int(config["scope"]["sharedCatalyticMatrices"]) != MATRIX_COUNT:
        raise RuntimeError("L15 matrix scope changed")
    if tuple(item["candidateId"] for item in config["simulations"]) != CANDIDATE_IDS:
        raise RuntimeError("L15 candidate contract changed")
    return config


def prior_roots() -> list[Path]:
    roots: list[Path] = []
    research_steps = Path("/artifacts/research_steps")
    for path in sorted(research_steps.iterdir()):
        if path.name != "S19":
            roots.append(path)
    for loop in (
        "L01",
        "L02",
        "L03",
        "L04",
        "L05",
        "L06",
        "L06R",
        "L07",
        "L08",
        "L09",
        "L10",
        "L11",
        "L11R",
        "L12",
        "L13",
        "L14",
    ):
        roots.append(S19_ROOT / "loops" / loop)
    for bundle in (
        Path("/artifacts/E01_forensic_replication_bundle"),
        Path("/artifacts/E01_forensic_replication_artifact_v2"),
    ):
        if bundle.exists():
            roots.append(bundle)
    return roots


def prior_files() -> list[Path]:
    files: list[Path] = []
    for root in prior_roots():
        if not root.exists():
            raise FileNotFoundError(root)
        files.extend(
            [root]
            if root.is_file()
            else sorted(p for p in root.rglob("*") if p.is_file())
        )
    unique = sorted(set(files), key=str)
    return unique


def create_immutable_baseline() -> dict[str, Any]:
    rows = [
        {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in prior_files()
    ]
    payload = {
        "schema": "eidosoma.e01.s19.l15.immutable_baseline.v1",
        "createdAtUtc": utc_now(),
        "fileCount": len(rows),
        "files": rows,
    }
    write_json(OUTPUT_ROOT / "immutable_prior_baseline.json", payload)
    return payload


def revalidate_immutable(baseline: dict[str, Any]) -> dict[str, Any]:
    mismatches: list[str] = []
    for row in baseline["files"]:
        path = Path(row["path"])
        if (
            not path.exists()
            or path.stat().st_size != int(row["bytes"])
            or sha256_file(path) != row["sha256"]
        ):
            mismatches.append(str(path))
    return {
        "schema": "eidosoma.e01.s19.l15.immutable_validation.v1",
        "fileCount": len(baseline["files"]),
        "mismatchCount": len(mismatches),
        "mismatches": mismatches,
        "passed": not mismatches,
        "validatedAtUtc": utc_now(),
    }


def simulation_specs() -> tuple[dict[str, Any], ...]:
    return tuple(dict(row) for row in load_config()["simulations"])


def make_definition(spec: dict[str, Any]) -> SimulationDefinition:
    return SimulationDefinition(
        daughter_rule=spec["daughterRule"],
        overshoot_rule=spec["overshootRule"],
        exposure=ExposureDefinition(
            family="FIXED_COMMON_EXPOSURE", h=float(spec["exposure"])
        ),
    )


def trajectory_path(cache_root: Path, matrix_index: int, candidate_id: str) -> Path:
    return cache_root / f"M{matrix_index:03d}__{candidate_id}.pkl"


def source_seed(candidate_id: str, matrix_index: int, purpose: str) -> int:
    root = load_config()["seedContract"]["analysisRootHex"]
    return int(seed128(root, "source", candidate_id, matrix_index, purpose) % (2**32))


def collect_prior_hex_tokens(files: Iterable[Path]) -> set[str]:
    pattern = re.compile(rb"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])")
    values: set[str] = set()
    for path in files:
        if path.stat().st_size > 64 * 1024 * 1024:
            continue
        try:
            values.update(
                match.decode("ascii") for match in pattern.findall(path.read_bytes())
            )
        except OSError:
            continue
    return values


def build_input_and_seed_manifests() -> tuple[
    pd.DataFrame, pd.DataFrame, dict[str, Any]
]:
    config = load_config()
    root = config["seedContract"]["matrixRootHex"]
    phase = config["seedContract"]["phase"]
    prior_tokens = collect_prior_hex_tokens(prior_files())
    input_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    overlaps: list[dict[str, str]] = []
    for matrix_index in range(MATRIX_COUNT):
        beta_seed = derive_seed(root, phase, "catalytic_matrix", matrix_index)
        init_seed = derive_seed(root, phase, "initial_state", matrix_index)
        beta = generate_beta(beta_seed)
        initial = initialize_distinct_state(init_seed)
        beta_hash = simulator_array_sha256(beta)
        initial_hash = simulator_array_sha256(initial)
        input_rows.append(
            {
                "matrixIndex": matrix_index,
                "betaSha256": beta_hash,
                "initialStateSha256": initial_hash,
                "initialMass": int(initial.sum()),
                "initialDistinctTypes": int(np.count_nonzero(initial)),
                "generatedBeforeScientificOutcomeAccess": True,
            }
        )
        for identity in (beta_seed, init_seed):
            seed_rows.append(
                {
                    "matrixIndex": matrix_index,
                    "candidateId": "SHARED",
                    "purpose": identity.purpose,
                    "configurationId": identity.configuration_id,
                    "derivedSeed": str(identity.derived_seed),
                    "seedMaterialSha256": identity.seed_material_sha256,
                }
            )
        for spec in simulation_specs():
            for purpose in (
                "poisson_update",
                "overshoot_trim",
                "fission",
                "daughter_selection",
            ):
                identity = derive_seed(
                    root, phase, purpose, matrix_index, spec["streamIdentity"]
                )
                seed_rows.append(
                    {
                        "matrixIndex": matrix_index,
                        "candidateId": spec["candidateId"],
                        "purpose": purpose,
                        "configurationId": identity.configuration_id,
                        "derivedSeed": str(identity.derived_seed),
                        "seedMaterialSha256": identity.seed_material_sha256,
                    }
                )
        for identity_type, value in (
            ("betaSha256", beta_hash),
            ("initialStateSha256", initial_hash),
            ("betaSeedMaterialSha256", beta_seed.seed_material_sha256),
            ("initialSeedMaterialSha256", init_seed.seed_material_sha256),
        ):
            if value in prior_tokens:
                overlaps.append({"identityType": identity_type, "identity": value})
    inputs = pd.DataFrame(input_rows)
    seeds = pd.DataFrame(seed_rows)
    for row in seeds.itertuples(index=False):
        if row.seedMaterialSha256 in prior_tokens:
            overlaps.append(
                {
                    "identityType": "derivedSeedMaterialSha256",
                    "identity": row.seedMaterialSha256,
                }
            )
    duplicate_seeds = int(seeds["seedMaterialSha256"].duplicated().sum())
    roots = [
        config["seedContract"]["matrixRootHex"],
        config["seedContract"]["splitRootHex"],
        config["seedContract"]["analysisRootHex"],
    ]
    root_overlap = [root for root in roots if root in prior_tokens]
    firewall = {
        "schema": "eidosoma.e01.s19.l15.seed_firewall.v1",
        "priorHexTokenCount": len(prior_tokens),
        "newRootCount": len(roots),
        "newSeedIdentityCount": len(seeds),
        "newMatrixCount": len(inputs),
        "rootOverlap": root_overlap,
        "inputOrSeedOverlap": overlaps,
        "duplicateNewSeedMaterialCount": duplicate_seeds,
        "passed": not root_overlap and not overlaps and duplicate_seeds == 0,
        "validatedAtUtc": utc_now(),
    }
    return inputs, seeds, firewall


def fixture_results() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def record(fixture_id: str, description: str, passed: bool, detail: object) -> None:
        rows.append(
            {
                "fixtureId": fixture_id,
                "description": description,
                "passed": bool(passed),
                "detail": canonical_json(json_safe(detail)),
            }
        )

    split_root = load_config()["seedContract"]["splitRootHex"]
    split = build_split_manifest(split_root)
    record(
        "F01",
        "200-matrix split contract",
        len(split) == 2000,
        split["splitRole"].value_counts().to_dict(),
    )
    valid = np.asarray([[True, True, False]], dtype=bool)
    mask_checks = {
        S00: (valid, valid),
        S01: (valid, np.ones_like(valid)),
        S10: (np.ones_like(valid), valid),
        S11: (np.ones_like(valid), np.ones_like(valid)),
    }
    mask_pass = True
    from e01_s19_padding_length_discrimination.core import mask_pair

    for condition_id, expected in mask_checks.items():
        observed = mask_pair(valid, condition_id)
        mask_pass &= np.array_equal(observed[0], expected[0]) and np.array_equal(
            observed[1], expected[1]
        )
    record("F02", "four mask conditions", mask_pass, list(mask_checks))
    target = np.asarray([[1, 1, 0, 0], [1, 0, 0, 0]], dtype=bool)
    target_mask = np.asarray([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=bool)
    p_valid = float(target[target_mask].mean())
    q = float(target_mask.mean())
    record(
        "F03",
        "padding prevalence identity",
        abs(padding_identity(p_valid, q) - float(target.mean())) <= 1e-15,
        {"pValid": p_valid, "q": q, "pPadded": float(target.mean())},
    )
    probability = np.asarray([[0.9, 0.9, 0.1, 0.1], [0.9, 0.1, 0.1, 0.1]])
    decomposition = accuracy_decomposition(target, probability, target_mask)
    record(
        "F04",
        "accuracy decomposition",
        decomposition["absoluteError"] <= 1e-15,
        decomposition,
    )
    synthetic = np.asarray(
        [[1, 0, 1] + [0] * 97, [2, 0, 2] + [0] * 97, [0, 1, 1] + [0] * 97]
    )
    h = incoming_h(normalized_compositions(synthetic))
    record(
        "F05",
        "adjacent H and label replay",
        np.allclose(h, [1, 1, 0.5], atol=2e-15, rtol=0),
        h.tolist(),
    )
    model = (
        __import__("e01_prediction_reconstruction.core", fromlist=["MaskedSequenceMLP"])
        .MaskedSequenceMLP()
        .to(dtype=torch.float64)
    )
    record(
        "F06",
        "S16 parameter count",
        parameter_count(model) == EXPECTED_PARAMETER_COUNT,
        parameter_count(model),
    )
    rng = np.random.Generator(
        np.random.PCG64DXSM(seed128(split_root, "fixture", "model"))
    )
    values = rng.normal(size=(4, MAX_INPUT_LENGTH, 100)).astype(np.float64)
    channel_mask = np.ones_like(values, dtype=bool)
    time_mask = np.ones((4, MAX_INPUT_LENGTH), dtype=bool)
    labels = rng.integers(0, 2, size=(4, MAX_TARGET_LENGTH)).astype(np.float64)
    label_mask = np.ones_like(labels, dtype=bool)
    kwargs = {
        "fit_values": values[:2],
        "fit_channel_mask": channel_mask[:2],
        "fit_time_mask": time_mask[:2],
        "fit_targets": labels[:2],
        "fit_target_mask": label_mask[:2],
        "validation_values": values[2:],
        "validation_channel_mask": channel_mask[2:],
        "validation_time_mask": time_mask[2:],
        "validation_targets": labels[2:],
        "validation_target_mask": label_mask[2:],
        "model_seed": torch_seed(split_root, "fixture", "model"),
        "maximum_epochs": 2,
        "patience": 2,
    }
    first = train_masked_mlp(**kwargs)
    second = train_masked_mlp(**kwargs)
    first_probability = predict_probabilities(
        first.model, values[2:], channel_mask[2:], time_mask[2:]
    )
    second_probability = predict_probabilities(
        second.model, values[2:], channel_mask[2:], time_mask[2:]
    )
    record(
        "F07",
        "exact S16 model replay",
        first.history.equals(second.history)
        and np.array_equal(first_probability, second_probability),
        {"probabilitySha256": array_sha256(first_probability)},
    )
    inferred = infer_output_length(np.asarray([0, 1, 10]))
    record(
        "F08",
        "deterministic boundary midpoint",
        np.array_equal(inferred, [2, 5, 32]),
        inferred.tolist(),
    )
    serialization = pd.DataFrame(
        {
            "flag": pd.Series([True, False, None], dtype="boolean"),
            "status": ["A", "B", None],
            "value": [1.0, np.nan, 3.0],
        }
    )
    fixture_path = CACHE_ROOT / "fixture_serialization.parquet"
    write_parquet(fixture_path, serialization)
    replay = pd.read_parquet(fixture_path)
    record(
        "F09",
        "typed Parquet serialization",
        len(replay) == 3 and list(replay.columns) == list(serialization.columns),
        {"rows": len(replay)},
    )
    record(
        "F10",
        "overflow is status-bearing",
        MAX_INPUT_LENGTH == 367 and MAX_TARGET_LENGTH == 1101,
        {"input": MAX_INPUT_LENGTH, "output": MAX_TARGET_LENGTH},
    )
    return pd.DataFrame(rows)


def source_snapshot() -> dict[str, Any]:
    paths = {
        "paperPdf": PAPER_PDF,
        "paperFigure5": PAPER_FIGURE5,
        "l14Digitization": L14_ROOT / "paper_figure5_digitization_lock.csv",
        "l14Report": L14_ROOT / "S19_L14_FULL_RESULTS.md",
        "s16ModelLock": S16_MODEL_LOCK,
        "s16Core": S16_CORE,
        "safePhiidLattice": SAFE_LATTICE,
        "l15Config": CONFIG_PATH,
        "l15Core": REPO_ROOT / "src/e01_s19_padding_length_discrimination/core.py",
        "l15Runner": Path(__file__),
    }
    files = []
    for identity, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(path)
        files.append(
            {
                "identity": identity,
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "schema": "eidosoma.e01.s19.l15.source_snapshot.v1",
        "repository": repository_lock(),
        "files": files,
        "pinnedPhiRLCommit": "a6d1d0d18c7551302724b7158c6ccdc4d3a33373",
        "historicalGARDCommit": "86dff6320d5ae91b4e831471079ff46749b14df9",
        "sourceUpdatesUsed": False,
    }


def prepare_phase() -> None:
    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
        raise RuntimeError("L15 artifact directory is not empty")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    lock = repository_lock()
    if not lock["passed"]:
        raise RuntimeError(f"repository is not clean/pushed: {lock}")
    baseline = create_immutable_baseline()
    immutable = revalidate_immutable(baseline)
    inputs, seeds, firewall = build_input_and_seed_manifests()
    if not immutable["passed"] or not firewall["passed"]:
        raise RuntimeError("immutable-prior or seed-firewall gate failed")
    fixtures = fixture_results()
    if len(fixtures) != 10 or not fixtures["passed"].all():
        raise RuntimeError("mandatory L15 fixtures failed")
    split = build_split_manifest(load_config()["seedContract"]["splitRootHex"])
    shutil.copyfile(CONFIG_PATH, OUTPUT_ROOT / "preregistration.yaml")
    write_parquet(OUTPUT_ROOT / "matrix_input_manifest.parquet", inputs)
    write_parquet(OUTPUT_ROOT / "seed_manifest.parquet", seeds)
    write_parquet(OUTPUT_ROOT / "split_manifest.parquet", split)
    write_json(OUTPUT_ROOT / "seed_firewall.json", firewall)
    write_json(OUTPUT_ROOT / "immutable_prior_validation.json", immutable)
    write_json(OUTPUT_ROOT / "source_snapshot_manifest.json", source_snapshot())
    write_json(
        OUTPUT_ROOT / "input_manifest.json",
        {
            "schema": "eidosoma.e01.s19.l15.input_manifest.v1",
            "matrixCount": len(inputs),
            "candidateCount": len(CANDIDATE_IDS),
            "expectedTrajectoryCount": len(inputs) * len(CANDIDATE_IDS),
            "matrixInputManifestSha256": sha256_file(
                OUTPUT_ROOT / "matrix_input_manifest.parquet"
            ),
            "seedManifestSha256": sha256_file(OUTPUT_ROOT / "seed_manifest.parquet"),
            "splitManifestSha256": sha256_file(OUTPUT_ROOT / "split_manifest.parquet"),
            "seedFirewallPassed": firewall["passed"],
        },
    )
    write_json(
        OUTPUT_ROOT / "fixture_manifest.json",
        {
            "fixtureCount": len(fixtures),
            "mandatoryCount": len(fixtures),
            "allPassed": bool(fixtures["passed"].all()),
        },
    )
    write_parquet(OUTPUT_ROOT / "fixture_results.parquet", fixtures)
    decision = """# S19-L15 decision record

L14 is complete and scientifically immutable. Its only discovered defect was a report-link path, already repaired with exact identity of all scientific hashes. A scientific L14 rerun is therefore neither needed nor permitted.

The newly authorized L15 tests the mechanism L14 never opened: the full frozen S16 MLP panel under valid-only and ordinary-zero-padded target semantics. The 200-matrix untouched cohort was fixed before outcomes because L14's two candidates bracketed the paper dummy interval and first-quarter length predicted the future padding boundary almost exactly. The larger sample is intended to distinguish a genuine length/padding mechanism from 100-matrix and ten-split noise, not to select a favorable candidate.

L15 changes no H threshold, label, simulator candidate, exposure, feature scalar, architecture, or paper digitization. Exact and merely directional Figure-5 resemblance are adjudicated separately. Any all-cell match remains forensic and cannot support early warning, prediction on real molecular cells, or causal control.
"""
    atomic_text(OUTPUT_ROOT / "decision_record.md", decision)
    write_json(
        OUTPUT_ROOT / "implementation_lock.json",
        {
            "schema": "eidosoma.e01.s19.l15.implementation_lock.v1",
            "lockedAtUtc": utc_now(),
            "repository": lock,
            "configSha256": sha256_file(CONFIG_PATH),
            "coreSha256": sha256_file(
                REPO_ROOT / "src/e01_s19_padding_length_discrimination/core.py"
            ),
            "runnerSha256": sha256_file(Path(__file__)),
            "testSha256": sha256_file(REPO_ROOT / "tests/e01/test_s19_l15.py"),
            "outcomeAccessed": False,
            "scientificRepairAfterAccessPermitted": False,
        },
    )
    write_csv(
        OUTPUT_ROOT / "failure_ledger.csv",
        pd.DataFrame(
            [
                {
                    "failureId": "S19-L15-PREOUTCOME-TECHNICAL-AMENDMENT-001",
                    "stage": "preoutcome_config_parse",
                    "candidateId": None,
                    "matrixIndex": None,
                    "failureType": "PREOUTCOME_YAML_SCHEMA_DEFECT_REPAIRED",
                    "message": "The first pushed lock had list/mapping indentation that prevented YAML parsing; no matrix, trajectory, label, feature, or model outcome was accessed. The failed partial baseline is preserved under /cache/e01_s19_l15/failed_attempt_001_artifacts.",
                    "scientificValuesEligible": False,
                }
            ]
        ),
    )
    write_json(
        OUTPUT_ROOT / "runtime_manifest.json",
        {
            "schema": "eidosoma.e01.s19.l15.runtime.v1",
            "startedAtUtc": utc_now(),
            "status": "PREPARED_PREOUTCOME",
            "cpuWorkers": 8,
            "gpuHours": 0,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikitLearn": sklearn.__version__,
            "torch": torch.__version__,
        },
    )
    print(
        canonical_json(
            {
                "phase": "prepare",
                "fixtures": len(fixtures),
                "immutableFiles": baseline["fileCount"],
                "seedFirewall": firewall["passed"],
            }
        )
    )


def simulate_matrix(matrix_index: int, cache_root_string: str) -> dict[str, Any]:
    cache_root = Path(cache_root_string)
    config = load_config()
    root = config["seedContract"]["matrixRootHex"]
    phase = config["seedContract"]["phase"]
    beta = generate_beta(derive_seed(root, phase, "catalytic_matrix", matrix_index))
    initial = initialize_distinct_state(
        derive_seed(root, phase, "initial_state", matrix_index)
    )
    attempts: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for spec in simulation_specs():
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        try:
            trajectory, seeds = simulate_trajectory(
                phase=phase,
                root_hex=root,
                matrix_index=matrix_index,
                definition=make_definition(spec),
                stream_identity=spec["streamIdentity"],
                beta=beta,
                initial_state=initial,
            )
            path = trajectory_path(cache_root, matrix_index, spec["candidateId"])
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as handle:
                pickle.dump(trajectory, handle, protocol=5)
            selected = selected_clock_observations(
                trajectory, "C1_SELECTED_DAUGHTER_RETAINED"
            )
            boundary_count = sum(
                row.observation_kind == "post_fission" for row in selected
            )
            complete = bool(
                trajectory.terminal_status == "requested_fissions_completed"
                and trajectory.completed_fissions == 100
                and boundary_count == 100
            )
            common = {
                "candidateId": spec["candidateId"],
                "matrixIndex": matrix_index,
                "exposure": spec["exposure"],
                "daughterRule": spec["daughterRule"],
                "overshootRule": spec["overshootRule"],
                "streamIdentity": spec["streamIdentity"],
                "replacementAttempted": False,
            }
            trajectories.append(
                {
                    **common,
                    "trajectoryId": trajectory.trajectory_id,
                    "trajectorySha256": trajectory.trajectory_sha256,
                    "betaSha256": trajectory.beta_sha256,
                    "initialStateSha256": trajectory.initial_state_sha256,
                    "terminalStatus": trajectory.terminal_status,
                    "completedFissions": int(trajectory.completed_fissions),
                    "selectedClockLength": len(selected),
                    "postFissionBoundaryCount": boundary_count,
                    "cachePath": str(path),
                    "cacheSha256": sha256_file(path),
                    "seedCount": len(seeds),
                }
            )
            attempts.append(
                {
                    **common,
                    "attemptStatus": "COMPLETE"
                    if complete
                    else "INCOMPLETE_OR_EXTINCT_RETAINED",
                    "terminalStatus": trajectory.terminal_status,
                    "completedFissions": int(trajectory.completed_fissions),
                    "wallSeconds": time.perf_counter() - wall_start,
                    "cpuSeconds": time.process_time() - cpu_start,
                }
            )
        except Exception as error:  # noqa: BLE001 - full provenance required
            failures.append(
                {
                    "failureId": f"S19-L15-SIM-{spec['candidateId']}-M{matrix_index:03d}",
                    "stage": "simulation",
                    "candidateId": spec["candidateId"],
                    "matrixIndex": matrix_index,
                    "failureType": type(error).__name__,
                    "message": str(error),
                    "scientificValuesEligible": False,
                }
            )
    return {"attempts": attempts, "trajectories": trajectories, "failures": failures}


def simulate_batch(
    indices: Iterable[int], cache_root: Path, workers: int = 8
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(simulate_matrix, int(index), str(cache_root)): int(index)
            for index in indices
        }
        for future in as_completed(futures):
            outputs.append(future.result())
    return outputs


def generate_phase() -> None:
    if not (OUTPUT_ROOT / "implementation_lock.json").exists():
        raise RuntimeError("prepare phase missing")
    lock = repository_lock()
    if not lock["passed"]:
        raise RuntimeError("repository lock no longer clean/pushed")
    immutable = revalidate_immutable(
        json.loads((OUTPUT_ROOT / "immutable_prior_baseline.json").read_text())
    )
    if not immutable["passed"]:
        raise RuntimeError("immutable prior changed before generation")
    if PRIMARY_TRAJECTORY_CACHE.exists() and any(
        PRIMARY_TRAJECTORY_CACHE.glob("*.pkl")
    ):
        raise RuntimeError("primary L15 trajectory cache is nonempty")
    PRIMARY_TRAJECTORY_CACHE.mkdir(parents=True, exist_ok=True)
    wall_start = time.perf_counter()
    child_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    benchmark_start = time.perf_counter()
    benchmark_outputs = simulate_batch(range(10), PRIMARY_TRAJECTORY_CACHE)
    benchmark_wall = time.perf_counter() - benchmark_start
    benchmark_attempts = pd.DataFrame(
        [row for output in benchmark_outputs for row in output["attempts"]]
    )
    benchmark_failures = [
        row for output in benchmark_outputs for row in output["failures"]
    ]
    benchmark_cpu = (
        float(benchmark_attempts["cpuSeconds"].sum())
        if len(benchmark_attempts)
        else math.inf
    )
    projected_simulation_cpu = (
        benchmark_cpu * (MATRIX_COUNT / 10.0) * 2.0 * 1.5 / 3600.0
    )
    projected_simulation_wall = (
        benchmark_wall * (MATRIX_COUNT / 10.0) * 2.0 * 1.5 / 3600.0
    )
    benchmark = {
        "schema": "eidosoma.e01.s19.l15.preoutcome_benchmark.v1",
        "benchmarkMatrixCount": 10,
        "benchmarkTrajectoryCount": len(benchmark_attempts),
        "scientificModelOutcomeOpened": False,
        "simulationWorkerCpuSeconds": benchmark_cpu,
        "simulationWallSeconds": benchmark_wall,
        "projectedSimulationAndRegenerationCpuHours": projected_simulation_cpu,
        "projectedSimulationAndRegenerationWallHours": projected_simulation_wall,
        "sourceAndModelProjectionPending": True,
        "safetyFactor": 1.5,
        "cpuCeilingAfterReserveHours": 136.0,
        "wallCeilingAfterReserveHours": 61.2,
        "failureCount": len(benchmark_failures),
        "passed": bool(
            not benchmark_failures
            and projected_simulation_cpu < 136.0
            and projected_simulation_wall < 61.2
        ),
        "completedAtUtc": utc_now(),
    }
    write_json(OUTPUT_ROOT / "preoutcome_benchmark.json", benchmark)
    if not benchmark["passed"]:
        write_csv(OUTPUT_ROOT / "failure_ledger.csv", pd.DataFrame(benchmark_failures))
        raise RuntimeError("simulation benchmark failed or exceeded ceiling")
    outputs = benchmark_outputs + simulate_batch(
        range(10, MATRIX_COUNT), PRIMARY_TRAJECTORY_CACHE
    )
    attempts = pd.DataFrame(
        [row for output in outputs for row in output["attempts"]]
    ).sort_values(["matrixIndex", "candidateId"], kind="stable")
    trajectories = pd.DataFrame(
        [row for output in outputs for row in output["trajectories"]]
    ).sort_values(["matrixIndex", "candidateId"], kind="stable")
    failures = pd.DataFrame([row for output in outputs for row in output["failures"]])
    if len(attempts) != 400 or len(trajectories) != 400 or len(failures):
        if len(failures):
            write_csv(OUTPUT_ROOT / "failure_ledger.csv", failures)
        raise RuntimeError("L15 did not retain exactly 400 trajectory attempts")
    write_parquet(OUTPUT_ROOT / "execution_status.parquet", attempts)
    write_parquet(OUTPUT_ROOT / "trajectory_manifest.parquet", trajectories)
    child_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    runtime = json.loads((OUTPUT_ROOT / "runtime_manifest.json").read_text())
    runtime.update(
        {
            "status": "TRAJECTORIES_GENERATED",
            "generationWallSeconds": time.perf_counter() - wall_start,
            "generationChildCpuSeconds": (child_after.ru_utime + child_after.ru_stime)
            - (child_before.ru_utime + child_before.ru_stime),
            "trajectoryCount": len(trajectories),
            "completeTrajectoryCount": int(
                attempts["attemptStatus"].eq("COMPLETE").sum()
            ),
        }
    )
    write_json(OUTPUT_ROOT / "runtime_manifest.json", runtime)
    print(
        canonical_json(
            {
                "phase": "generate",
                "trajectories": len(trajectories),
                "complete": runtime["completeTrajectoryCount"],
            }
        )
    )


def feature_cache_path(root: Path, candidate_id: str, matrix_index: int) -> Path:
    return root / f"{candidate_id}__M{matrix_index:03d}.npz"


def feature_worker(
    manifest_row: dict[str, Any], output_cache_string: str
) -> dict[str, Any]:
    output_cache = Path(output_cache_string)
    candidate_id = str(manifest_row["candidateId"])
    matrix_index = int(manifest_row["matrixIndex"])
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    try:
        path = Path(manifest_row["cachePath"])
        if sha256_file(path) != manifest_row["cacheSha256"]:
            raise ValueError("trajectory cache hash changed")
        with path.open("rb") as handle:
            trajectory = pickle.load(handle)
        if trajectory.trajectory_sha256 != manifest_row["trajectorySha256"]:
            raise ValueError("trajectory identity mismatch")
        selected = selected_clock_observations(
            trajectory, "C1_SELECTED_DAUGHTER_RETAINED"
        )
        states = np.asarray([row.state for row in selected], dtype=np.int64)
        total = len(states)
        cutoff = math.floor(0.25 * total)
        target_length = total - cutoff
        status = "ELIGIBLE"
        eligible = bool(
            trajectory.terminal_status == "requested_fissions_completed"
            and trajectory.completed_fissions == 100
            and cutoff <= MAX_INPUT_LENGTH
            and target_length <= MAX_TARGET_LENGTH
        )
        if not eligible:
            status = (
                "INELIGIBLE_S16_TENSOR_OVERFLOW_RETAINED"
                if cutoff > MAX_INPUT_LENGTH or target_length > MAX_TARGET_LENGTH
                else "INELIGIBLE_INCOMPLETE_OR_EXTINCT_RETAINED"
            )
            output_cache.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                feature_cache_path(output_cache, candidate_id, matrix_index),
                eligible=np.asarray(False),
                total=np.asarray(total),
                cutoff=np.asarray(cutoff),
                targetLength=np.asarray(target_length),
                status=np.asarray(status),
            )
            return {
                "candidateId": candidate_id,
                "matrixIndex": matrix_index,
                "trajectoryId": trajectory.trajectory_id,
                "status": status,
                "eligible": False,
                "T": total,
                "cutoff": cutoff,
                "targetLength": target_length,
                "wallSeconds": time.perf_counter() - wall_start,
                "cpuSeconds": time.process_time() - cpu_start,
                "failure": None,
            }
        compositions = normalized_compositions(states)
        h_values = incoming_h(compositions)
        labels = h_values > 0.9
        target = np.zeros(MAX_TARGET_LENGTH, dtype=np.float64)
        target_mask = np.zeros(MAX_TARGET_LENGTH, dtype=bool)
        target[:target_length] = labels[cutoff:].astype(np.float64)
        target_mask[:target_length] = True
        input_labels = np.zeros(MAX_INPUT_LENGTH, dtype=bool)
        input_labels[:cutoff] = labels[:cutoff]
        change = np.zeros(total, dtype=np.float64)
        change[1:] = np.linalg.norm(np.diff(compositions, axis=0), axis=1)
        flux = np.zeros_like(states, dtype=np.float64)
        flux[1:] = np.diff(states, axis=0)
        full_clr, _, full_closure_error = frozen_clr(states)
        prefix_clr, _, prefix_closure_error = frozen_clr(states[:cutoff])
        preprocessing_seed = source_seed(candidate_id, matrix_index, "preprocessing")
        partition_seed = source_seed(candidate_id, matrix_index, "partition")
        full_result = run_emergence_pipeline(
            full_clr,
            PHIRL,
            SAFE_LATTICE,
            preprocessing_seed=preprocessing_seed,
            partition_seed=partition_seed,
        )
        prefix_result = run_emergence_pipeline(
            prefix_clr,
            PHIRL,
            SAFE_LATTICE,
            preprocessing_seed=preprocessing_seed,
            partition_seed=partition_seed,
        )
        full_values, full_available = source_values(
            full_result, fit_length=total, retained_length=cutoff
        )
        prefix_values, prefix_available = source_values(
            prefix_result, fit_length=cutoff, retained_length=cutoff
        )
        feature_map = {
            P1: build_feature(full_values, full_available, cutoff, scalar=True),
            P2: build_feature(prefix_values, prefix_available, cutoff, scalar=True),
            B1: build_feature(
                change[:cutoff], np.arange(cutoff) > 0, cutoff, scalar=True
            ),
            B2: build_feature(
                states[:cutoff].astype(np.float64),
                np.ones((cutoff, 100), dtype=bool),
                cutoff,
                scalar=False,
            ),
            B3: build_feature(
                flux[:cutoff],
                np.broadcast_to((np.arange(cutoff) > 0)[:, None], (cutoff, 100)),
                cutoff,
                scalar=False,
            ),
            B4: build_feature(
                h_values[:cutoff], np.ones(cutoff, dtype=bool), cutoff, scalar=True
            ),
        }
        payload: dict[str, Any] = {
            "eligible": np.asarray(True),
            "total": np.asarray(total),
            "cutoff": np.asarray(cutoff),
            "targetLength": np.asarray(target_length),
            "status": np.asarray(status),
            "target": target,
            "targetMask": target_mask,
            "inputLabels": input_labels,
            "hValues": h_values,
            "labels": labels,
            "statesSha256": np.asarray(array_sha256(states)),
            "fullStatus": np.asarray(full_result.status),
            "prefixStatus": np.asarray(prefix_result.status),
            "fullResultReplayHash": np.asarray(array_sha256(full_values)),
            "prefixResultReplayHash": np.asarray(array_sha256(prefix_values)),
            "fullClosureMaximumError": np.asarray(float(np.max(full_closure_error))),
            "prefixClosureMaximumError": np.asarray(
                float(np.max(prefix_closure_error))
            ),
        }
        for feature_id, (values, mask, time_mask) in feature_map.items():
            payload[f"{feature_id}__values"] = values
            payload[f"{feature_id}__channelMask"] = mask
            payload[f"{feature_id}__timeMask"] = time_mask
        output_cache.mkdir(parents=True, exist_ok=True)
        output_path = feature_cache_path(output_cache, candidate_id, matrix_index)
        np.savez_compressed(output_path, **payload)
        return {
            "candidateId": candidate_id,
            "matrixIndex": matrix_index,
            "trajectoryId": trajectory.trajectory_id,
            "status": status,
            "eligible": True,
            "T": total,
            "cutoff": cutoff,
            "targetLength": target_length,
            "occupancy": float(labels.mean()),
            "suffixPrevalence": float(labels[cutoff:].mean()),
            "inputPositive": bool(np.any(labels[:cutoff])),
            "firstOnset": int(np.flatnonzero(labels)[0]) if np.any(labels) else None,
            "fullSourceStatus": full_result.status,
            "prefixSourceStatus": prefix_result.status,
            "fullRetainedVariables": len(full_result.retained_variables),
            "prefixRetainedVariables": len(prefix_result.retained_variables),
            "fullFeatureAvailable": int(full_available.sum()),
            "prefixFeatureAvailable": int(prefix_available.sum()),
            "fullClosureMaximumError": float(np.max(full_closure_error)),
            "prefixClosureMaximumError": float(np.max(prefix_closure_error)),
            "cachePath": str(output_path),
            "cacheSha256": sha256_file(output_path),
            "wallSeconds": time.perf_counter() - wall_start,
            "cpuSeconds": time.process_time() - cpu_start,
            "failure": None,
        }
    except Exception as error:  # noqa: BLE001
        return {
            "candidateId": candidate_id,
            "matrixIndex": matrix_index,
            "status": "UNREGISTERED_FEATURE_EXCEPTION",
            "eligible": False,
            "wallSeconds": time.perf_counter() - wall_start,
            "cpuSeconds": time.process_time() - cpu_start,
            "failure": {
                "failureId": f"S19-L15-FEATURE-{candidate_id}-M{matrix_index:03d}",
                "stage": "feature_construction",
                "candidateId": candidate_id,
                "matrixIndex": matrix_index,
                "failureType": type(error).__name__,
                "message": str(error),
                "scientificValuesEligible": False,
            },
        }


def run_feature_batch(
    manifest: pd.DataFrame, cache_root: Path, workers: int = 8
) -> list[dict[str, Any]]:
    rows = manifest.to_dict(orient="records")
    outputs: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(feature_worker, row, str(cache_root)): (
                row["candidateId"],
                row["matrixIndex"],
            )
            for row in rows
        }
        for future in as_completed(futures):
            outputs.append(future.result())
    return outputs


def consolidate_tensors(
    source_rows: pd.DataFrame, feature_cache: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    TENSOR_ROOT.mkdir(parents=True, exist_ok=True)
    target_manifest_rows: list[dict[str, Any]] = []
    feature_manifest_rows: list[dict[str, Any]] = []
    length_rows: list[dict[str, Any]] = []
    source_rows_out: list[dict[str, Any]] = []
    for candidate_id in CANDIDATE_IDS:
        target = np.zeros((MATRIX_COUNT, MAX_TARGET_LENGTH), dtype=np.float64)
        target_mask = np.zeros_like(target, dtype=bool)
        input_labels = np.zeros((MATRIX_COUNT, MAX_INPUT_LENGTH), dtype=bool)
        eligible = np.zeros(MATRIX_COUNT, dtype=bool)
        total = np.zeros(MATRIX_COUNT, dtype=np.int64)
        cutoff = np.zeros(MATRIX_COUNT, dtype=np.int64)
        target_length = np.zeros(MATRIX_COUNT, dtype=np.int64)
        feature_stacks = {
            feature_id: {
                "values": np.zeros(
                    (MATRIX_COUNT, MAX_INPUT_LENGTH, 100), dtype=np.float64
                ),
                "channelMask": np.zeros(
                    (MATRIX_COUNT, MAX_INPUT_LENGTH, 100), dtype=bool
                ),
                "timeMask": np.zeros((MATRIX_COUNT, MAX_INPUT_LENGTH), dtype=bool),
            }
            for feature_id in LEARNED_FEATURES
        }
        for matrix_index in range(MATRIX_COUNT):
            cache_path = feature_cache_path(feature_cache, candidate_id, matrix_index)
            if not cache_path.exists():
                raise FileNotFoundError(cache_path)
            with np.load(cache_path, allow_pickle=False) as payload:
                is_eligible = bool(payload["eligible"])
                eligible[matrix_index] = is_eligible
                total[matrix_index] = int(payload["total"])
                cutoff[matrix_index] = int(payload["cutoff"])
                target_length[matrix_index] = int(payload["targetLength"])
                status = str(payload["status"])
                if is_eligible:
                    target[matrix_index] = payload["target"]
                    target_mask[matrix_index] = payload["targetMask"]
                    input_labels[matrix_index] = payload["inputLabels"]
                    for feature_id in LEARNED_FEATURES:
                        for field in ("values", "channelMask", "timeMask"):
                            feature_stacks[feature_id][field][matrix_index] = payload[
                                f"{feature_id}__{field}"
                            ]
                    source_rows_out.append(
                        {
                            "candidateId": candidate_id,
                            "matrixIndex": matrix_index,
                            "fullStatus": str(payload["fullStatus"]),
                            "prefixStatus": str(payload["prefixStatus"]),
                            "fullResultSha256": str(payload["fullResultReplayHash"]),
                            "prefixResultSha256": str(
                                payload["prefixResultReplayHash"]
                            ),
                            "fullClosureMaximumError": float(
                                payload["fullClosureMaximumError"]
                            ),
                            "prefixClosureMaximumError": float(
                                payload["prefixClosureMaximumError"]
                            ),
                        }
                    )
                target_manifest_rows.append(
                    {
                        "candidateId": candidate_id,
                        "matrixIndex": matrix_index,
                        "eligible": is_eligible,
                        "status": status,
                        "T": int(total[matrix_index]),
                        "cutoff": int(cutoff[matrix_index]),
                        "targetLength": int(target_length[matrix_index]),
                        "targetSha256": array_sha256(payload["target"])
                        if is_eligible
                        else None,
                        "targetMaskSha256": array_sha256(payload["targetMask"])
                        if is_eligible
                        else None,
                    }
                )
                length_rows.append(
                    {
                        "candidateId": candidate_id,
                        "matrixIndex": matrix_index,
                        "eligible": is_eligible,
                        "status": status,
                        "T": int(total[matrix_index]),
                        "inputLength": int(cutoff[matrix_index]),
                        "outputLength": int(target_length[matrix_index]),
                        "inferredOutputLength": int(
                            infer_output_length(int(cutoff[matrix_index]))
                        ),
                        "boundaryAbsoluteError": abs(
                            int(infer_output_length(int(cutoff[matrix_index])))
                            - int(target_length[matrix_index])
                        ),
                    }
                )
        np.savez_compressed(
            TENSOR_ROOT / f"{candidate_id}__target.npz",
            target=target,
            targetMask=target_mask,
            inputLabels=input_labels,
            eligible=eligible,
            total=total,
            cutoff=cutoff,
            targetLength=target_length,
        )
        for feature_id in LEARNED_FEATURES:
            path = TENSOR_ROOT / f"{candidate_id}__{feature_id}.npz"
            np.savez_compressed(path, **feature_stacks[feature_id])
            feature_manifest_rows.append(
                {
                    "candidateId": candidate_id,
                    "featureId": feature_id,
                    "eligibleMatrixCount": int(eligible.sum()),
                    "tensorPath": str(path),
                    "tensorCacheSha256": sha256_file(path),
                    "valuesSha256": array_sha256(feature_stacks[feature_id]["values"]),
                    "channelMaskSha256": array_sha256(
                        feature_stacks[feature_id]["channelMask"]
                    ),
                    "timeMaskSha256": array_sha256(
                        feature_stacks[feature_id]["timeMask"]
                    ),
                }
            )
    return (
        pd.DataFrame(target_manifest_rows),
        pd.DataFrame(feature_manifest_rows),
        pd.DataFrame(length_rows),
        pd.DataFrame(source_rows_out),
    )


def load_target(candidate_id: str) -> dict[str, np.ndarray]:
    with np.load(
        TENSOR_ROOT / f"{candidate_id}__target.npz", allow_pickle=False
    ) as payload:
        return {name: payload[name] for name in payload.files}


def load_feature(candidate_id: str, feature_id: str) -> dict[str, np.ndarray]:
    with np.load(
        TENSOR_ROOT / f"{candidate_id}__{feature_id}.npz", allow_pickle=False
    ) as payload:
        return {name: payload[name] for name in payload.files}


def suffix_invariance_audit(manifest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    sentinels = (0, 49, 99, 149, 199)
    for candidate_id in CANDIDATE_IDS:
        for matrix_index in sentinels:
            manifest_row = manifest.loc[
                manifest["candidateId"].eq(candidate_id)
                & manifest["matrixIndex"].eq(matrix_index)
            ]
            if len(manifest_row) != 1:
                raise RuntimeError("suffix sentinel trajectory identity missing")
            with Path(manifest_row.iloc[0]["cachePath"]).open("rb") as handle:
                trajectory = pickle.load(handle)
            selected = selected_clock_observations(
                trajectory, "C1_SELECTED_DAUGHTER_RETAINED"
            )
            states = np.asarray([item.state for item in selected], dtype=np.int64)
            cutoff = math.floor(0.25 * len(states))
            if cutoff > MAX_INPUT_LENGTH or len(states) - cutoff > MAX_TARGET_LENGTH:
                rows.append(
                    {
                        "candidateId": candidate_id,
                        "matrixIndex": matrix_index,
                        "status": "INELIGIBLE_TENSOR_OVERFLOW",
                        "p1MaximumAbsoluteChange": None,
                        "p2Exact": None,
                        "passed": False,
                    }
                )
                continue
            rng = np.random.Generator(
                np.random.PCG64DXSM(
                    seed128(
                        load_config()["seedContract"]["analysisRootHex"],
                        "suffix_permutation",
                        candidate_id,
                        matrix_index,
                    )
                )
            )
            altered = states.copy()
            altered[cutoff:] = altered[cutoff:][rng.permutation(len(states) - cutoff)]
            full_clr, _, _ = frozen_clr(states)
            altered_clr, _, _ = frozen_clr(altered)
            prefix_clr, _, _ = frozen_clr(states[:cutoff])
            altered_prefix_clr, _, _ = frozen_clr(altered[:cutoff])
            pre_seed = source_seed(candidate_id, matrix_index, "preprocessing")
            part_seed = source_seed(candidate_id, matrix_index, "partition")
            full = run_emergence_pipeline(
                full_clr,
                PHIRL,
                SAFE_LATTICE,
                preprocessing_seed=pre_seed,
                partition_seed=part_seed,
            )
            full_altered = run_emergence_pipeline(
                altered_clr,
                PHIRL,
                SAFE_LATTICE,
                preprocessing_seed=pre_seed,
                partition_seed=part_seed,
            )
            prefix = run_emergence_pipeline(
                prefix_clr,
                PHIRL,
                SAFE_LATTICE,
                preprocessing_seed=pre_seed,
                partition_seed=part_seed,
            )
            prefix_altered = run_emergence_pipeline(
                altered_prefix_clr,
                PHIRL,
                SAFE_LATTICE,
                preprocessing_seed=pre_seed,
                partition_seed=part_seed,
            )
            p1, p1_mask = source_values(
                full, fit_length=len(states), retained_length=cutoff
            )
            p1_altered, p1_altered_mask = source_values(
                full_altered, fit_length=len(states), retained_length=cutoff
            )
            shared = p1_mask & p1_altered_mask
            maximum_change = (
                float(np.max(np.abs(p1[shared] - p1_altered[shared])))
                if np.any(shared)
                else None
            )
            p2_exact = bool(
                np.array_equal(prefix_clr, altered_prefix_clr)
                and result_replay_equal(prefix, prefix_altered)
            )
            rows.append(
                {
                    "candidateId": candidate_id,
                    "matrixIndex": matrix_index,
                    "status": "ELIGIBLE",
                    "prefixStatesExact": bool(
                        np.array_equal(states[:cutoff], altered[:cutoff])
                    ),
                    "prefixClrExact": bool(
                        np.array_equal(prefix_clr, altered_prefix_clr)
                    ),
                    "p1SharedCount": int(shared.sum()),
                    "p1MaximumAbsoluteChange": maximum_change,
                    "p1Changed": bool(
                        maximum_change is not None and maximum_change > 0.0
                    ),
                    "p2Exact": p2_exact,
                    "passed": p2_exact,
                }
            )
    return pd.DataFrame(rows)


def arithmetic_tables(
    target_manifest: pd.DataFrame, lengths: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    geometry_rows: list[dict[str, Any]] = []
    prevalence_rows: list[dict[str, Any]] = []
    for candidate_id in CANDIDATE_IDS:
        payload = load_target(candidate_id)
        eligible = payload["eligible"].astype(bool)
        target = payload["target"][eligible].astype(bool)
        mask = payload["targetMask"][eligible].astype(bool)
        valid_count = int(mask.sum())
        all_count = int(mask.size)
        valid_fraction = float(valid_count / all_count)
        valid_prevalence = float(target[mask].mean())
        padded_prevalence = float(target.mean())
        identity_error = abs(
            padded_prevalence - padding_identity(valid_prevalence, valid_fraction)
        )
        length_subset = lengths.loc[
            lengths["candidateId"].eq(candidate_id) & lengths["eligible"]
        ]
        correlation = float(
            stats.pearsonr(
                length_subset["inputLength"].to_numpy(dtype=np.float64),
                length_subset["outputLength"].to_numpy(dtype=np.float64),
            ).statistic
        )
        geometry_rows.append(
            {
                "candidateId": candidate_id,
                "matrixCount": MATRIX_COUNT,
                "eligibleMatrixCount": int(eligible.sum()),
                "ineligibleMatrixCount": int((~eligible).sum()),
                "validCellCount": valid_count,
                "paddingCellCount": all_count - valid_count,
                "allCellCount": all_count,
                "validFraction": valid_fraction,
                "paddingFraction": 1.0 - valid_fraction,
                "validPrevalence": valid_prevalence,
                "paddedPrevalence": padded_prevalence,
                "validOnlyDummyAccuracy": max(valid_prevalence, 1.0 - valid_prevalence),
                "paddedDummyAccuracy": max(padded_prevalence, 1.0 - padded_prevalence),
                "identityAbsoluteError": identity_error,
                "maximumInputLength": int(payload["cutoff"][eligible].max()),
                "maximumOutputLength": int(payload["targetLength"][eligible].max()),
                "inputOutputLengthPearson": correlation,
                "inferredBoundaryExactFraction": float(
                    (length_subset["boundaryAbsoluteError"] == 0).mean()
                ),
                "inferredBoundaryMeanAbsoluteError": float(
                    length_subset["boundaryAbsoluteError"].mean()
                ),
            }
        )
        for matrix_index in range(MATRIX_COUNT):
            if not eligible[matrix_index]:
                continue
            valid = payload["targetMask"][matrix_index].astype(bool)
            y = payload["target"][matrix_index].astype(bool)
            input_y = payload["inputLabels"][matrix_index].astype(bool)
            valid_indices = np.flatnonzero(valid)
            positives = valid_indices[y[valid_indices]]
            prevalence_rows.append(
                {
                    "candidateId": candidate_id,
                    "matrixIndex": matrix_index,
                    "T": int(payload["total"][matrix_index]),
                    "cutoff": int(payload["cutoff"][matrix_index]),
                    "targetLength": int(payload["targetLength"][matrix_index]),
                    "validTargetPrevalence": float(y[valid].mean()),
                    "paddedTargetPrevalence": float(y.mean()),
                    "preOnsetAtCutoff": bool(
                        not np.any(input_y[: int(payload["cutoff"][matrix_index])])
                    ),
                    "futureOnsetAfterCutoff": bool(
                        positives.size
                        and not np.any(input_y[: int(payload["cutoff"][matrix_index])])
                    ),
                    "firstPositiveSuffixIndex": int(positives[0])
                    if positives.size
                    else None,
                }
            )
    return pd.DataFrame(geometry_rows), pd.DataFrame(prevalence_rows)


def features_phase() -> None:
    manifest = pd.read_parquet(OUTPUT_ROOT / "trajectory_manifest.parquet")
    if len(manifest) != 400:
        raise RuntimeError("trajectory manifest cardinality changed")
    if TRAJECTORY_FEATURE_CACHE.exists() and any(
        TRAJECTORY_FEATURE_CACHE.glob("*.npz")
    ):
        raise RuntimeError("primary L15 feature cache is nonempty")
    TRAJECTORY_FEATURE_CACHE.mkdir(parents=True, exist_ok=True)
    wall_start = time.perf_counter()
    child_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    benchmark_manifest = manifest.loc[manifest["matrixIndex"].lt(10)].copy()
    benchmark_start = time.perf_counter()
    benchmark_outputs = run_feature_batch(benchmark_manifest, TRAJECTORY_FEATURE_CACHE)
    benchmark_wall = time.perf_counter() - benchmark_start
    benchmark_cpu = float(sum(row["cpuSeconds"] for row in benchmark_outputs))
    benchmark_failures = [
        row["failure"] for row in benchmark_outputs if row.get("failure")
    ]
    projected_source_cpu = benchmark_cpu * (MATRIX_COUNT / 10.0) * 2.0 * 1.5 / 3600.0
    projected_source_wall = benchmark_wall * (MATRIX_COUNT / 10.0) * 2.0 * 1.5 / 3600.0
    benchmark = json.loads((OUTPUT_ROOT / "preoutcome_benchmark.json").read_text())
    benchmark.update(
        {
            "sourceBenchmarkTrajectoryCount": len(benchmark_outputs),
            "sourceBenchmarkWorkerCpuSeconds": benchmark_cpu,
            "sourceBenchmarkWallSeconds": benchmark_wall,
            "projectedSourceAndRegenerationCpuHours": projected_source_cpu,
            "projectedSourceAndRegenerationWallHours": projected_source_wall,
            "sourceFailureCount": len(benchmark_failures),
        }
    )
    combined_cpu = float(
        benchmark["projectedSimulationAndRegenerationCpuHours"]
        + projected_source_cpu
        + 20.0
    )
    combined_wall = float(
        benchmark["projectedSimulationAndRegenerationWallHours"]
        + projected_source_wall
        + 12.0
    )
    benchmark["modelAndFinalizationCpuAllowanceHours"] = 20.0
    benchmark["modelAndFinalizationWallAllowanceHours"] = 12.0
    benchmark["projectedTotalCpuHours"] = combined_cpu
    benchmark["projectedTotalWallHours"] = combined_wall
    benchmark["passed"] = bool(
        benchmark["passed"]
        and not benchmark_failures
        and combined_cpu <= 136.0
        and combined_wall <= 61.2
    )
    write_json(OUTPUT_ROOT / "preoutcome_benchmark.json", benchmark)
    if not benchmark["passed"]:
        if benchmark_failures:
            write_csv(
                OUTPUT_ROOT / "failure_ledger.csv", pd.DataFrame(benchmark_failures)
            )
        raise RuntimeError("combined preoutcome benchmark exceeded ceiling or failed")
    remaining = manifest.loc[~manifest["matrixIndex"].lt(10)].copy()
    outputs = benchmark_outputs + run_feature_batch(remaining, TRAJECTORY_FEATURE_CACHE)
    failures = [row["failure"] for row in outputs if row.get("failure")]
    if failures:
        write_csv(OUTPUT_ROOT / "failure_ledger.csv", pd.DataFrame(failures))
        raise RuntimeError("unregistered feature exception")
    feature_execution = (
        pd.DataFrame(outputs)
        .drop(columns=["failure"])
        .sort_values(["matrixIndex", "candidateId"], kind="stable")
    )
    write_parquet(CACHE_ROOT / "feature_execution.parquet", feature_execution)
    target_manifest, feature_manifest, lengths, source_execution = consolidate_tensors(
        feature_execution, TRAJECTORY_FEATURE_CACHE
    )
    eligible_counts = target_manifest.groupby("candidateId")["eligible"].sum().to_dict()
    minimum = int(load_config()["scope"]["minimumModelEligibleMatricesEachCandidate"])
    if any(
        int(eligible_counts.get(candidate_id, 0)) < minimum
        for candidate_id in CANDIDATE_IDS
    ):
        raise RuntimeError(
            f"fewer than {minimum} model-eligible matrices: {eligible_counts}"
        )
    geometry, prevalence = arithmetic_tables(target_manifest, lengths)
    if float(geometry["identityAbsoluteError"].max()) > 1e-15:
        raise RuntimeError("padding arithmetic identity failed")
    suffix = suffix_invariance_audit(manifest)
    if not suffix["passed"].all():
        raise RuntimeError("prefix-only future-suffix invariant failed")
    write_parquet(OUTPUT_ROOT / "padded_target_manifest.parquet", target_manifest)
    write_parquet(OUTPUT_ROOT / "feature_manifest.parquet", feature_manifest)
    write_parquet(OUTPUT_ROOT / "trajectory_length_results.parquet", lengths)
    write_parquet(OUTPUT_ROOT / "padding_geometry_results.parquet", geometry)
    write_parquet(OUTPUT_ROOT / "prevalence_decomposition.parquet", prevalence)
    write_parquet(OUTPUT_ROOT / "source_execution_results.parquet", source_execution)
    write_parquet(OUTPUT_ROOT / "suffix_invariance_results.parquet", suffix)
    child_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    runtime = json.loads((OUTPUT_ROOT / "runtime_manifest.json").read_text())
    runtime.update(
        {
            "status": "FEATURES_MATERIALIZED",
            "featureWallSeconds": time.perf_counter() - wall_start,
            "featureChildCpuSeconds": (child_after.ru_utime + child_after.ru_stime)
            - (child_before.ru_utime + child_before.ru_stime),
            "eligibleMatricesByCandidate": {
                key: int(value) for key, value in eligible_counts.items()
            },
            "sourceTaskCount": len(source_execution),
        }
    )
    write_json(OUTPUT_ROOT / "runtime_manifest.json", runtime)
    print(
        canonical_json(
            {
                "phase": "features",
                "eligible": eligible_counts,
                "sourceTasks": len(source_execution),
            }
        )
    )


def condition_masks(
    target_mask: np.ndarray, eligible: np.ndarray, condition_id: str
) -> tuple[np.ndarray, np.ndarray]:
    train_padding, score_padding = MASK_CONTRACT[condition_id]
    eligible_rows = np.asarray(eligible, dtype=bool)[:, None]
    train_mask = (
        np.broadcast_to(eligible_rows, target_mask.shape).copy()
        if train_padding
        else np.asarray(target_mask, dtype=bool) & eligible_rows
    )
    score_mask_value = (
        np.broadcast_to(eligible_rows, target_mask.shape).copy()
        if score_padding
        else np.asarray(target_mask, dtype=bool) & eligible_rows
    )
    return train_mask, score_mask_value


def metric_row(
    candidate_id: str,
    feature_id: str,
    condition_id: str,
    repetition: int,
    scope: str,
    target: np.ndarray,
    probability: np.ndarray,
) -> dict[str, Any]:
    return {
        "candidateId": candidate_id,
        "featureId": feature_id,
        "conditionId": condition_id,
        "repetitionId": repetition,
        "metricScope": scope,
        **extended_binary_metrics(target, probability),
    }


def per_matrix_rows(
    candidate_id: str,
    feature_id: str,
    condition_id: str,
    repetition: int,
    matrix_indices: np.ndarray,
    target: np.ndarray,
    probability: np.ndarray,
    valid_mask: np.ndarray,
    eligible: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for local_index, matrix_index in enumerate(matrix_indices):
        if not eligible[matrix_index]:
            continue
        decomposition = accuracy_decomposition(
            target[local_index], probability[local_index], valid_mask[local_index]
        )
        rows.append(
            {
                "candidateId": candidate_id,
                "featureId": feature_id,
                "conditionId": condition_id,
                "repetitionId": repetition,
                "matrixIndex": int(matrix_index),
                "probabilitySha256": array_sha256(probability[local_index]),
                "targetSha256": array_sha256(target[local_index]),
                **decomposition,
            }
        )
    return rows


def model_task(
    candidate_id: str,
    feature_id: str,
    train_includes_padding: bool,
    repetition: int,
) -> dict[str, Any]:
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    split = pd.read_parquet(OUTPUT_ROOT / "split_manifest.parquet")
    target_payload = load_target(candidate_id)
    feature_payload = load_feature(candidate_id, feature_id)
    fit = split_indices(split, repetition, "FIT")
    validation = split_indices(split, repetition, "VALIDATION")
    test = split_indices(split, repetition, "TEST")
    scaler = fit_channel_scaler(
        feature_payload["values"][fit], feature_payload["channelMask"][fit]
    )
    scaled = apply_channel_scaler(
        feature_payload["values"], feature_payload["channelMask"], scaler
    )
    training_condition = S10 if train_includes_padding else S00
    fit_loss_mask, _ = condition_masks(
        target_payload["targetMask"][fit],
        target_payload["eligible"][fit],
        training_condition,
    )
    validation_loss_mask, _ = condition_masks(
        target_payload["targetMask"][validation],
        target_payload["eligible"][validation],
        training_condition,
    )
    model_seed = torch_seed(
        load_config()["seedContract"]["splitRootHex"],
        "model",
        candidate_id,
        repetition,
    )
    result = train_masked_mlp(
        scaled[fit],
        feature_payload["channelMask"][fit],
        feature_payload["timeMask"][fit],
        target_payload["target"][fit],
        fit_loss_mask,
        scaled[validation],
        feature_payload["channelMask"][validation],
        feature_payload["timeMask"][validation],
        target_payload["target"][validation],
        validation_loss_mask,
        model_seed=model_seed,
    )
    probability = predict_probabilities(
        result.model,
        scaled[test],
        feature_payload["channelMask"][test],
        feature_payload["timeMask"][test],
    )
    replay_passed = None
    replay_probability_hash = None
    if repetition == 0:
        replay = train_masked_mlp(
            scaled[fit],
            feature_payload["channelMask"][fit],
            feature_payload["timeMask"][fit],
            target_payload["target"][fit],
            fit_loss_mask,
            scaled[validation],
            feature_payload["channelMask"][validation],
            feature_payload["timeMask"][validation],
            target_payload["target"][validation],
            validation_loss_mask,
            model_seed=model_seed,
        )
        replay_probability = predict_probabilities(
            replay.model,
            scaled[test],
            feature_payload["channelMask"][test],
            feature_payload["timeMask"][test],
        )
        replay_probability_hash = array_sha256(replay_probability)
        replay_passed = bool(
            result.history.equals(replay.history)
            and result.best_epoch == replay.best_epoch
            and np.array_equal(probability, replay_probability)
        )
    condition_ids = (S10, S11) if train_includes_padding else (S00, S01)
    metric_rows: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    for condition_id in condition_ids:
        _, test_score_mask = condition_masks(
            target_payload["targetMask"][test],
            target_payload["eligible"][test],
            condition_id,
        )
        valid_mask = (
            target_payload["targetMask"][test] & target_payload["eligible"][test, None]
        )
        padding_mask = (~target_payload["targetMask"][test]) & target_payload[
            "eligible"
        ][test, None]
        for scope, mask in (
            (
                "ALL_CELL" if condition_id in (S01, S11) else "SCORED_CELL",
                test_score_mask,
            ),
            ("VALID_CELL", valid_mask),
            ("PADDING_CELL", padding_mask),
        ):
            metric_rows.append(
                metric_row(
                    candidate_id,
                    feature_id,
                    condition_id,
                    repetition,
                    scope,
                    target_payload["target"][test][mask],
                    probability[mask],
                )
            )
        matrix_rows.extend(
            per_matrix_rows(
                candidate_id,
                feature_id,
                condition_id,
                repetition,
                test,
                target_payload["target"][test],
                probability,
                target_payload["targetMask"][test],
                target_payload["eligible"],
            )
        )
    # The input-time mask is explicit in the frozen S16 architecture.  Noise in
    # padded values must therefore be exactly annihilated; this is NC3.
    obfuscated = scaled[test].copy()
    rng = np.random.Generator(
        np.random.PCG64DXSM(
            seed128(
                load_config()["seedContract"]["analysisRootHex"],
                "input_padding_obfuscation",
                candidate_id,
                feature_id,
                train_includes_padding,
                repetition,
            )
        )
    )
    padded_input = ~feature_payload["timeMask"][test]
    noise = rng.normal(size=obfuscated.shape)
    obfuscated[padded_input[:, :, None].repeat(100, axis=2)] = noise[
        padded_input[:, :, None].repeat(100, axis=2)
    ]
    obfuscated_probability = predict_probabilities(
        result.model,
        obfuscated,
        feature_payload["channelMask"][test],
        feature_payload["timeMask"][test],
    )
    history_rows = result.history.assign(
        candidateId=candidate_id,
        featureId=feature_id,
        trainIncludesPadding=train_includes_padding,
        repetitionId=repetition,
        modelSeed=model_seed,
        bestEpoch=result.best_epoch,
        stoppedEpoch=result.stopped_epoch,
        bestValidationLoss=result.best_validation_loss,
    ).to_dict(orient="records")
    MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    probability_path = (
        MODEL_CACHE
        / f"{candidate_id}__{feature_id}__trainpad-{int(train_includes_padding)}__R{repetition:02d}.npz"
    )
    np.savez_compressed(
        probability_path, testMatrixIndices=test, probability=probability
    )
    return {
        "candidateId": candidate_id,
        "featureId": feature_id,
        "trainIncludesPadding": train_includes_padding,
        "repetitionId": repetition,
        "metricRows": metric_rows,
        "matrixRows": matrix_rows,
        "historyRows": history_rows,
        "replayRow": {
            "candidateId": candidate_id,
            "featureId": feature_id,
            "trainIncludesPadding": train_includes_padding,
            "repetitionId": repetition,
            "checked": repetition == 0,
            "passed": replay_passed,
            "probabilitySha256": array_sha256(probability),
            "replayProbabilitySha256": replay_probability_hash,
        },
        "obfuscationRow": {
            "controlId": "NC3_INPUT_PADDING_OBFUSCATION_WITH_TIME_MASK_RETAINED",
            "candidateId": candidate_id,
            "featureId": feature_id,
            "trainIncludesPadding": train_includes_padding,
            "repetitionId": repetition,
            "maximumProbabilityDifference": float(
                np.max(np.abs(probability - obfuscated_probability))
            ),
            "probabilitiesExact": bool(
                np.array_equal(probability, obfuscated_probability)
            ),
        },
        "probabilityCachePath": str(probability_path),
        "probabilityCacheSha256": sha256_file(probability_path),
        "wallSeconds": time.perf_counter() - wall_start,
        "cpuSeconds": time.process_time() - cpu_start,
    }


def run_model_tasks(workers: int = 8) -> list[dict[str, Any]]:
    tasks = [
        (candidate_id, feature_id, train_padding, repetition)
        for candidate_id in CANDIDATE_IDS
        for feature_id in LEARNED_FEATURES
        for train_padding in (False, True)
        for repetition in range(REPETITIONS)
    ]
    outputs: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(model_task, *task): task for task in tasks}
        for future in as_completed(futures):
            outputs.append(future.result())
    return outputs


def fit_logistic_probability(
    fit_x: np.ndarray, fit_y: np.ndarray, test_x: np.ndarray
) -> np.ndarray:
    if np.unique(fit_y).size == 1:
        return np.full(len(test_x), float(fit_y[0]), dtype=np.float64)
    model = LogisticRegression(
        solver="lbfgs",
        penalty="l2",
        C=1.0,
        tol=1e-8,
        max_iter=1000,
        class_weight=None,
        random_state=None,
    )
    model.fit(fit_x, fit_y)
    return model.predict_proba(test_x)[:, 1].astype(np.float64)


def diagnostic_design(
    input_lengths: np.ndarray,
    fit_indices: np.ndarray,
    matrix_indices: np.ndarray,
    *,
    include_length: bool,
) -> np.ndarray:
    fit_lengths = input_lengths[fit_indices].astype(np.float64)
    mean = float(fit_lengths.mean())
    scale = float(fit_lengths.std(ddof=0))
    if scale < 1e-12:
        scale = 1.0
    length_z = (input_lengths[matrix_indices].astype(np.float64) - mean) / scale
    position = np.linspace(0.0, 1.0, MAX_TARGET_LENGTH, dtype=np.float64)
    position_grid = np.broadcast_to(position, (len(matrix_indices), MAX_TARGET_LENGTH))
    if not include_length:
        return position_grid.reshape(-1, 1)
    length_grid = np.broadcast_to(length_z[:, None], position_grid.shape)
    return np.column_stack(
        (
            length_grid.reshape(-1),
            position_grid.reshape(-1),
            (length_grid * position_grid).reshape(-1),
        )
    )


def diagnostic_models() -> dict[str, list[dict[str, Any]]]:
    split = pd.read_parquet(OUTPUT_ROOT / "split_manifest.parquet")
    metric_rows: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    for candidate_id in CANDIDATE_IDS:
        payload = load_target(candidate_id)
        for repetition in range(REPETITIONS):
            fit = split_indices(split, repetition, "FIT")
            test = split_indices(split, repetition, "TEST")
            fit_valid_mask = payload["targetMask"][fit] & payload["eligible"][fit, None]
            valid_majority = bool(payload["target"][fit][fit_valid_mask].mean() >= 0.5)
            for train_padding in (False, True):
                train_condition = S10 if train_padding else S00
                fit_mask, _ = condition_masks(
                    payload["targetMask"][fit],
                    payload["eligible"][fit],
                    train_condition,
                )
                fit_y = (
                    payload["target"][fit].reshape(-1)[fit_mask.reshape(-1)].astype(int)
                )
                majority = bool(fit_y.mean() >= 0.5)
                dummy_probability = np.full(
                    (len(test), MAX_TARGET_LENGTH), float(majority), dtype=np.float64
                )
                fit_length_design = diagnostic_design(
                    payload["cutoff"], fit, fit, include_length=True
                )
                test_length_design = diagnostic_design(
                    payload["cutoff"], fit, test, include_length=True
                )
                length_probability = fit_logistic_probability(
                    fit_length_design[fit_mask.reshape(-1)],
                    fit_y,
                    test_length_design,
                ).reshape(len(test), MAX_TARGET_LENGTH)
                fit_time_design = diagnostic_design(
                    payload["cutoff"], fit, fit, include_length=False
                )
                test_time_design = diagnostic_design(
                    payload["cutoff"], fit, test, include_length=False
                )
                time_probability = fit_logistic_probability(
                    fit_time_design[fit_mask.reshape(-1)],
                    fit_y,
                    test_time_design,
                ).reshape(len(test), MAX_TARGET_LENGTH)
                inferred = np.minimum(
                    infer_output_length(payload["cutoff"][test]), MAX_TARGET_LENGTH
                )
                boundary_probability = np.full(
                    (len(test), MAX_TARGET_LENGTH), 0.001, dtype=np.float64
                )
                for local_index, inferred_length in enumerate(inferred):
                    if valid_majority:
                        boundary_probability[local_index, : int(inferred_length)] = (
                            0.999
                        )
                models = {
                    D0: dummy_probability,
                    D1: length_probability,
                    D2: boundary_probability,
                    D3: time_probability,
                }
                condition_ids = (S10, S11) if train_padding else (S00, S01)
                for feature_id, probability in models.items():
                    for condition_id in condition_ids:
                        _, score_mask = condition_masks(
                            payload["targetMask"][test],
                            payload["eligible"][test],
                            condition_id,
                        )
                        valid_mask = (
                            payload["targetMask"][test]
                            & payload["eligible"][test, None]
                        )
                        padding_mask = (~payload["targetMask"][test]) & payload[
                            "eligible"
                        ][test, None]
                        for scope, mask in (
                            (
                                "ALL_CELL"
                                if condition_id in (S01, S11)
                                else "SCORED_CELL",
                                score_mask,
                            ),
                            ("VALID_CELL", valid_mask),
                            ("PADDING_CELL", padding_mask),
                        ):
                            metric_rows.append(
                                metric_row(
                                    candidate_id,
                                    feature_id,
                                    condition_id,
                                    repetition,
                                    scope,
                                    payload["target"][test][mask],
                                    probability[mask],
                                )
                            )
                        matrix_rows.extend(
                            per_matrix_rows(
                                candidate_id,
                                feature_id,
                                condition_id,
                                repetition,
                                test,
                                payload["target"][test],
                                probability,
                                payload["targetMask"][test],
                                payload["eligible"],
                            )
                        )
                    diagnostic_rows.append(
                        {
                            "candidateId": candidate_id,
                            "featureId": feature_id,
                            "trainIncludesPadding": train_padding,
                            "repetitionId": repetition,
                            "fitIncludedCellCount": int(fit_mask.sum()),
                            "fitPositivePrevalence": float(fit_y.mean()),
                            "fitMajorityPositive": majority,
                            "validFitMajorityPositive": valid_majority,
                            "probabilitySha256": array_sha256(probability),
                        }
                    )
                # NC2: intentionally break the length/boundary pairing without
                # inspecting label values.
                permutation_rng = np.random.Generator(
                    np.random.PCG64DXSM(
                        seed128(
                            load_config()["seedContract"]["analysisRootHex"],
                            "boundary_permutation",
                            candidate_id,
                            repetition,
                            train_padding,
                        )
                    )
                )
                permuted_inferred = inferred[permutation_rng.permutation(len(inferred))]
                permuted_probability = np.full_like(boundary_probability, 0.001)
                for local_index, inferred_length in enumerate(permuted_inferred):
                    if valid_majority:
                        permuted_probability[local_index, : int(inferred_length)] = (
                            0.999
                        )
                _, s11_mask = condition_masks(
                    payload["targetMask"][test], payload["eligible"][test], S11
                )
                control_rows.append(
                    {
                        "controlId": "NC2_PADDING_BOUNDARY_PERMUTATION",
                        "candidateId": candidate_id,
                        "featureId": D2,
                        "repetitionId": repetition,
                        "trainIncludesPadding": train_padding,
                        "accuracy": float(
                            extended_binary_metrics(
                                payload["target"][test][s11_mask],
                                permuted_probability[s11_mask],
                            )["accuracy"]
                        ),
                    }
                )
    return {
        "metricRows": metric_rows,
        "matrixRows": matrix_rows,
        "diagnosticRows": diagnostic_rows,
        "controlRows": control_rows,
    }


def transform_control_task(
    control_id: str, candidate_id: str, repetition: int
) -> dict[str, Any]:
    split = pd.read_parquet(OUTPUT_ROOT / "split_manifest.parquet")
    target_payload = load_target(candidate_id)
    feature_payload = load_feature(candidate_id, P1)
    fit = split_indices(split, repetition, "FIT")
    validation = split_indices(split, repetition, "VALIDATION")
    test = split_indices(split, repetition, "TEST")
    values = feature_payload["values"].copy()
    channel_mask = feature_payload["channelMask"].copy()
    time_mask = feature_payload["timeMask"].copy()
    target = target_payload["target"].copy()
    target_mask = target_payload["targetMask"].copy()
    root = load_config()["seedContract"]["analysisRootHex"]
    if control_id == "NC1_VALID_LABEL_PERMUTATION_PRESERVING_PADDING":
        for role, indices in (("fit", fit), ("validation", validation)):
            source = target[indices].copy()
            source_values_flat = source[
                target_mask[indices] & target_payload["eligible"][indices, None]
            ].copy()
            rng = np.random.Generator(
                np.random.PCG64DXSM(
                    seed128(root, control_id, candidate_id, repetition, role)
                )
            )
            source_values_flat = source_values_flat[
                rng.permutation(len(source_values_flat))
            ]
            role_target = target[indices].copy()
            role_target[
                target_mask[indices] & target_payload["eligible"][indices, None]
            ] = source_values_flat
            target[indices] = role_target
    elif control_id == "NC4_VALID_FEATURE_TEMPORAL_PERMUTATION":
        for matrix_index in range(MATRIX_COUNT):
            cutoff = int(target_payload["cutoff"][matrix_index])
            if not target_payload["eligible"][matrix_index] or cutoff <= 1:
                continue
            rng = np.random.Generator(
                np.random.PCG64DXSM(
                    seed128(root, control_id, candidate_id, repetition, matrix_index)
                )
            )
            order = rng.permutation(cutoff)
            values[matrix_index, :cutoff] = values[matrix_index, order]
            channel_mask[matrix_index, :cutoff] = channel_mask[matrix_index, order]
    elif control_id == "NC5_MATRIX_LABEL_PERMUTATION":
        rng = np.random.Generator(
            np.random.PCG64DXSM(seed128(root, control_id, candidate_id, repetition))
        )
        source = fit[rng.permutation(len(fit))]
        target[fit] = target[source]
        target_mask[fit] = target_mask[source]
    else:
        raise ValueError(f"unregistered model control: {control_id}")
    scaler = fit_channel_scaler(values[fit], channel_mask[fit])
    scaled = apply_channel_scaler(values, channel_mask, scaler)
    fit_loss, _ = condition_masks(
        target_mask[fit], target_payload["eligible"][fit], S11
    )
    validation_loss, _ = condition_masks(
        target_mask[validation], target_payload["eligible"][validation], S11
    )
    model_seed = torch_seed(
        load_config()["seedContract"]["splitRootHex"], "model", candidate_id, repetition
    )
    result = train_masked_mlp(
        scaled[fit],
        channel_mask[fit],
        time_mask[fit],
        target[fit],
        fit_loss,
        scaled[validation],
        channel_mask[validation],
        time_mask[validation],
        target[validation],
        validation_loss,
        model_seed=model_seed,
    )
    probability = predict_probabilities(
        result.model, scaled[test], channel_mask[test], time_mask[test]
    )
    _, score = condition_masks(
        target_payload["targetMask"][test], target_payload["eligible"][test], S11
    )
    # Test against the unpermuted scientific target for every control.
    metrics = extended_binary_metrics(
        target_payload["target"][test][score], probability[score]
    )
    return {
        "controlId": control_id,
        "candidateId": candidate_id,
        "featureId": P1,
        "repetitionId": repetition,
        "accuracy": metrics["accuracy"],
        "balancedAccuracy": metrics["balancedAccuracy"],
        "auroc": metrics["auroc"],
        "auprc": metrics["auprc"],
        "brier": metrics["brier"],
        "bestEpoch": result.best_epoch,
        "stoppedEpoch": result.stopped_epoch,
        "probabilitySha256": array_sha256(probability),
    }


def run_transform_controls(workers: int = 8) -> pd.DataFrame:
    tasks = [
        (control_id, candidate_id, repetition)
        for control_id in (
            "NC1_VALID_LABEL_PERMUTATION_PRESERVING_PADDING",
            "NC4_VALID_FEATURE_TEMPORAL_PERMUTATION",
            "NC5_MATRIX_LABEL_PERMUTATION",
        )
        for candidate_id in CANDIDATE_IDS
        for repetition in range(REPETITIONS)
    ]
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(transform_control_task, *task): task for task in tasks}
        for future in as_completed(futures):
            rows.append(future.result())
    return pd.DataFrame(rows).sort_values(
        ["controlId", "candidateId", "repetitionId"], kind="stable"
    )


def execute_phase() -> None:
    if not (OUTPUT_ROOT / "feature_manifest.parquet").exists():
        raise RuntimeError("feature phase missing")
    if MODEL_CACHE.exists() and any(MODEL_CACHE.glob("*.npz")):
        raise RuntimeError("primary L15 model cache is nonempty")
    MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    wall_start = time.perf_counter()
    child_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    model_outputs = run_model_tasks(workers=8)
    replay_rows = [row["replayRow"] for row in model_outputs]
    checked_replay = [row for row in replay_rows if row["checked"]]
    if len(checked_replay) != 24 or not all(row["passed"] for row in checked_replay):
        raise RuntimeError("registered model replay failed")
    metric_rows = [row for output in model_outputs for row in output["metricRows"]]
    matrix_rows = [row for output in model_outputs for row in output["matrixRows"]]
    history_rows = [row for output in model_outputs for row in output["historyRows"]]
    obfuscation_rows = [
        row["obfuscationRow"]
        for row in model_outputs
        if row["featureId"] == P1 and row["trainIncludesPadding"]
    ]
    diagnostics = diagnostic_models()
    metric_rows.extend(diagnostics["metricRows"])
    matrix_rows.extend(diagnostics["matrixRows"])
    control_rows = list(diagnostics["controlRows"])
    control_rows.extend(obfuscation_rows)
    transformed_controls = run_transform_controls(workers=8)
    control_frame = pd.concat(
        [pd.DataFrame(control_rows), transformed_controls],
        ignore_index=True,
        sort=False,
    )
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(matrix_rows)
    histories = pd.DataFrame(history_rows)
    diagnostics_frame = pd.DataFrame(diagnostics["diagnosticRows"])
    all_metrics = metrics.loc[
        metrics["metricScope"].isin(["ALL_CELL", "SCORED_CELL"])
    ].copy()
    valid_metrics = metrics.loc[metrics["metricScope"].eq("VALID_CELL")].copy()
    padding_metrics = metrics.loc[metrics["metricScope"].eq("PADDING_CELL")].copy()
    decomposition = predictions[
        [
            "candidateId",
            "featureId",
            "conditionId",
            "repetitionId",
            "matrixIndex",
            "validFraction",
            "allCellAccuracy",
            "validCellAccuracy",
            "paddingCellAccuracy",
            "reconstructedAccuracy",
            "absoluteError",
            "correctFromPaddingFraction",
        ]
    ].copy()
    if float(decomposition["absoluteError"].max()) > 1e-12:
        raise RuntimeError("model accuracy decomposition failed")
    write_parquet(OUTPUT_ROOT / "training_history.parquet", histories)
    write_parquet(OUTPUT_ROOT / "prediction_results.parquet", predictions)
    write_parquet(OUTPUT_ROOT / "all_cell_metrics.parquet", all_metrics)
    write_parquet(OUTPUT_ROOT / "valid_cell_metrics.parquet", valid_metrics)
    write_parquet(OUTPUT_ROOT / "padding_cell_metrics.parquet", padding_metrics)
    write_parquet(OUTPUT_ROOT / "accuracy_decomposition.parquet", decomposition)
    write_parquet(OUTPUT_ROOT / "diagnostic_results.parquet", diagnostics_frame)
    write_parquet(OUTPUT_ROOT / "negative_control_results.parquet", control_frame)
    write_parquet(CACHE_ROOT / "model_replay.parquet", pd.DataFrame(replay_rows))
    child_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    runtime = json.loads((OUTPUT_ROOT / "runtime_manifest.json").read_text())
    runtime.update(
        {
            "status": "MODEL_EXECUTION_COMPLETE",
            "modelWallSeconds": time.perf_counter() - wall_start,
            "modelChildCpuSeconds": (child_after.ru_utime + child_after.ru_stime)
            - (child_before.ru_utime + child_before.ru_stime),
            "primaryModelFitCount": len(model_outputs),
            "controlModelFitCount": len(transformed_controls),
            "modelReplayCount": len(checked_replay),
            "modelReplayPassedCount": int(
                sum(bool(row["passed"]) for row in checked_replay)
            ),
        }
    )
    write_json(OUTPUT_ROOT / "runtime_manifest.json", runtime)
    print(
        canonical_json(
            {
                "phase": "execute",
                "primaryFits": len(model_outputs),
                "controlFits": len(transformed_controls),
                "replay": len(checked_replay),
            }
        )
    )


def holm_adjust(values: Iterable[float]) -> list[float]:
    array = np.asarray(list(values), dtype=np.float64)
    order = np.argsort(array)
    adjusted = np.empty_like(array)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(array) - rank) * array[index]))
        adjusted[index] = running
    return adjusted.tolist()


def interval_overlap(left: tuple[float, float], right: tuple[float, float]) -> bool:
    return max(left[0], right[0]) <= min(left[1], right[1])


def comparison_and_gate_tables() -> dict[str, Any]:
    all_metrics = pd.read_parquet(OUTPUT_ROOT / "all_cell_metrics.parquet")
    valid_metrics = pd.read_parquet(OUTPUT_ROOT / "valid_cell_metrics.parquet")
    predictions = pd.read_parquet(OUTPUT_ROOT / "prediction_results.parquet")
    decomposition = pd.read_parquet(OUTPUT_ROOT / "accuracy_decomposition.parquet")
    controls = pd.read_parquet(OUTPUT_ROOT / "negative_control_results.parquet")
    prevalence = pd.read_parquet(OUTPUT_ROOT / "prevalence_decomposition.parquet")
    digitized = pd.read_csv(L14_ROOT / "paper_figure5_digitization_lock.csv")
    box_rows: list[dict[str, Any]] = []
    order_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    for candidate_id in CANDIDATE_IDS:
        for feature_id in PAPER_FEATURES:
            observed = (
                all_metrics.loc[
                    all_metrics["candidateId"].eq(candidate_id)
                    & all_metrics["featureId"].eq(feature_id)
                    & all_metrics["conditionId"].eq(S11)
                    & all_metrics["metricScope"].eq("ALL_CELL"),
                    "accuracy",
                ]
                .sort_index()
                .to_numpy(dtype=np.float64)
            )
            if len(observed) != REPETITIONS:
                raise RuntimeError(
                    f"paper panel split count mismatch for {candidate_id}/{feature_id}"
                )
            paper = digitized.loc[digitized["featureId"].eq(feature_id)].iloc[0]
            q1, median, q3 = np.quantile(observed, [0.25, 0.5, 0.75])
            paper_iqr = (
                min(float(paper["q1"]), float(paper["q3"])),
                max(float(paper["q1"]), float(paper["q3"])),
            )
            box_rows.append(
                {
                    "candidateId": candidate_id,
                    "featureId": feature_id,
                    "conditionId": S11,
                    "observedMinimum": float(np.min(observed)),
                    "observedQ1": float(q1),
                    "observedMedian": float(median),
                    "observedQ3": float(q3),
                    "observedMaximum": float(np.max(observed)),
                    "paperQ1": paper_iqr[0],
                    "paperMedian": float(paper["median"]),
                    "paperQ3": paper_iqr[1],
                    "paperMedianLower": float(paper["medianLower"]),
                    "paperMedianUpper": float(paper["medianUpper"]),
                    "medianExactIntervalPassed": bool(
                        float(paper["medianLower"])
                        <= median
                        <= float(paper["medianUpper"])
                    ),
                    "iqrOverlaps": interval_overlap((float(q1), float(q3)), paper_iqr),
                    "directionalWithinPoint05": bool(
                        abs(median - float(paper["median"])) <= 0.05
                    ),
                    "absoluteMedianDifference": float(
                        abs(median - float(paper["median"]))
                    ),
                }
            )
        p1 = (
            all_metrics.loc[
                all_metrics["candidateId"].eq(candidate_id)
                & all_metrics["featureId"].eq(P1)
                & all_metrics["conditionId"].eq(S11)
                & all_metrics["metricScope"].eq("ALL_CELL")
            ]
            .sort_values("repetitionId")["accuracy"]
            .to_numpy(dtype=np.float64)
        )
        for comparator in (B1, B2, B3, D0):
            other = (
                all_metrics.loc[
                    all_metrics["candidateId"].eq(candidate_id)
                    & all_metrics["featureId"].eq(comparator)
                    & all_metrics["conditionId"].eq(S11)
                    & all_metrics["metricScope"].eq("ALL_CELL")
                ]
                .sort_values("repetitionId")["accuracy"]
                .to_numpy(dtype=np.float64)
            )
            difference = p1 - other
            mann = float(
                stats.mannwhitneyu(
                    p1, other, alternative="two-sided", method="auto"
                ).pvalue
            )
            try:
                wilcoxon = float(
                    stats.wilcoxon(
                        difference, alternative="two-sided", method="auto"
                    ).pvalue
                )
            except ValueError:
                wilcoxon = 1.0
            left = predictions.loc[
                predictions["candidateId"].eq(candidate_id)
                & predictions["featureId"].eq(P1)
                & predictions["conditionId"].eq(S11),
                ["repetitionId", "matrixIndex", "allCellAccuracy"],
            ].rename(columns={"allCellAccuracy": "reference"})
            right = predictions.loc[
                predictions["candidateId"].eq(candidate_id)
                & predictions["featureId"].eq(comparator)
                & predictions["conditionId"].eq(S11),
                ["repetitionId", "matrixIndex", "allCellAccuracy"],
            ].rename(columns={"allCellAccuracy": "comparator"})
            paired = left.merge(
                right, on=["repetitionId", "matrixIndex"], validate="one_to_one"
            )
            matrix_effect = (
                paired.assign(effect=paired["reference"] - paired["comparator"])
                .groupby("matrixIndex")["effect"]
                .mean()
                .to_numpy(dtype=np.float64)
            )
            rng = np.random.Generator(
                np.random.PCG64DXSM(
                    seed128(
                        load_config()["seedContract"]["analysisRootHex"],
                        "bootstrap",
                        candidate_id,
                        comparator,
                    )
                )
            )
            sampled = rng.integers(
                0, len(matrix_effect), size=(4096, len(matrix_effect))
            )
            distribution = matrix_effect[sampled].mean(axis=1)
            lower, upper = np.quantile(distribution, [0.025, 0.975])
            comparison_rows.append(
                {
                    "candidateId": candidate_id,
                    "referenceFeatureId": P1,
                    "comparatorFeatureId": comparator,
                    "meanSplitDifference": float(np.mean(difference)),
                    "medianSplitDifference": float(np.median(difference)),
                    "positiveSplitCount": int(np.count_nonzero(difference > 0)),
                    "mannWhitneyP": mann,
                    "pairedWilcoxonP": wilcoxon,
                    "pairedMatrixCount": len(matrix_effect),
                    "meanPairedMatrixDifference": float(np.mean(matrix_effect)),
                    "bootstrapLower95": float(lower),
                    "bootstrapUpper95": float(upper),
                }
            )
            bootstrap_rows.extend(
                {
                    "candidateId": candidate_id,
                    "comparatorFeatureId": comparator,
                    "replicate": replicate,
                    "meanPairedMatrixDifference": float(value),
                }
                for replicate, value in enumerate(distribution)
            )
            order_rows.append(
                {
                    "candidateId": candidate_id,
                    "referenceFeatureId": P1,
                    "comparatorFeatureId": comparator,
                    "referenceMedian": float(np.median(p1)),
                    "comparatorMedian": float(np.median(other)),
                    "orderingPassed": bool(np.median(p1) > np.median(other)),
                    "mannWhitneyBelowPoint01": mann < 0.01,
                    "pairedMatrixDirectionPositive": float(np.mean(matrix_effect))
                    > 0.0,
                }
            )
    box = pd.DataFrame(box_rows)
    order = pd.DataFrame(order_rows)
    comparisons = pd.DataFrame(comparison_rows)
    comparisons["mannWhitneyHolmP"] = holm_adjust(comparisons["mannWhitneyP"])
    comparisons["pairedWilcoxonHolmP"] = holm_adjust(comparisons["pairedWilcoxonP"])
    exact_by_candidate: dict[str, bool] = {}
    directional_by_candidate: dict[str, bool] = {}
    for candidate_id in CANDIDATE_IDS:
        box_c = box.loc[box["candidateId"].eq(candidate_id)]
        order_c = order.loc[order["candidateId"].eq(candidate_id)]
        criteria = {
            "allFiveExactMedianAndIqr": bool(
                (box_c["medianExactIntervalPassed"] & box_c["iqrOverlaps"]).all()
            ),
            "allFiveDirectionalMedianTolerance": bool(
                box_c["directionalWithinPoint05"].all()
            ),
            "allFourOrdering": bool(order_c["orderingPassed"].all()),
            "allFourMannWhitneyBelowPoint01": bool(
                order_c["mannWhitneyBelowPoint01"].all()
            ),
            "allFourPairedMatrixDirections": bool(
                order_c["pairedMatrixDirectionPositive"].all()
            ),
        }
        exact_by_candidate[candidate_id] = bool(
            criteria["allFiveExactMedianAndIqr"]
            and criteria["allFourOrdering"]
            and criteria["allFourMannWhitneyBelowPoint01"]
            and criteria["allFourPairedMatrixDirections"]
        )
        directional_by_candidate[candidate_id] = bool(
            criteria["allFiveDirectionalMedianTolerance"]
            and criteria["allFourOrdering"]
            and criteria["allFourPairedMatrixDirections"]
        )
        for criterion, passed in criteria.items():
            gate_rows.append(
                {
                    "gateFamily": "FIGURE5_PANEL",
                    "candidateId": candidate_id,
                    "criterion": criterion,
                    "passed": passed,
                }
            )
    dominance_rows: list[dict[str, Any]] = []
    for candidate_id in CANDIDATE_IDS:
        p1_decomposition = decomposition.loc[
            decomposition["candidateId"].eq(candidate_id)
            & decomposition["featureId"].eq(P1)
            & decomposition["conditionId"].eq(S11)
        ]
        p1_median = float(
            all_metrics.loc[
                all_metrics["candidateId"].eq(candidate_id)
                & all_metrics["featureId"].eq(P1)
                & all_metrics["conditionId"].eq(S11)
                & all_metrics["metricScope"].eq("ALL_CELL"),
                "accuracy",
            ].median()
        )
        diagnostic_medians = {
            feature_id: float(
                all_metrics.loc[
                    all_metrics["candidateId"].eq(candidate_id)
                    & all_metrics["featureId"].eq(feature_id)
                    & all_metrics["conditionId"].eq(S11)
                    & all_metrics["metricScope"].eq("ALL_CELL"),
                    "accuracy",
                ].median()
            )
            for feature_id in (D1, D2)
        }
        nc1 = controls.loc[
            controls["controlId"].eq("NC1_VALID_LABEL_PERMUTATION_PRESERVING_PADDING")
            & controls["candidateId"].eq(candidate_id),
            "accuracy",
        ]
        retention = float(nc1.median() / p1_median) if len(nc1) else None
        valid_p1 = valid_metrics.loc[
            valid_metrics["candidateId"].eq(candidate_id)
            & valid_metrics["featureId"].eq(P1)
            & valid_metrics["conditionId"].eq(S11),
            "accuracy",
        ]
        dominance = {
            "candidateId": candidate_id,
            "p1AllCellMedian": p1_median,
            "p1ValidCellMedian": float(valid_p1.median()),
            "allMinusValidMedian": float(
                (
                    p1_decomposition["allCellAccuracy"]
                    - p1_decomposition["validCellAccuracy"]
                ).median()
            ),
            "correctPredictionsFromPaddingMedian": float(
                p1_decomposition["correctFromPaddingFraction"].median()
            ),
            "lengthOnlyMedian": diagnostic_medians[D1],
            "boundaryRuleMedian": diagnostic_medians[D2],
            "lengthOrBoundaryWithinPoint03": bool(
                min(
                    abs(diagnostic_medians[D1] - p1_median),
                    abs(diagnostic_medians[D2] - p1_median),
                )
                <= 0.03
            ),
            "validLabelShuffleRetention": retention,
            "paddingDominated": False,
        }
        dominance["paddingDominated"] = bool(
            dominance["allMinusValidMedian"] >= 0.05
            or dominance["correctPredictionsFromPaddingMedian"] >= 0.50
            or dominance["lengthOrBoundaryWithinPoint03"]
            or (retention is not None and retention >= 0.80)
        )
        dominance_rows.append(dominance)
    dominance = pd.DataFrame(dominance_rows)
    preonset_counts = prevalence.groupby("candidateId").agg(
        preOnsetMatrices=("preOnsetAtCutoff", "sum"),
        futureOnsetMatrices=("futureOnsetAfterCutoff", "sum"),
    )
    exact_pass = all(exact_by_candidate.values())
    directional_pass = all(directional_by_candidate.values())
    padding_dominated = bool(dominance["paddingDominated"].any())
    prospective_eligible = bool(
        (preonset_counts["preOnsetMatrices"] >= 20).all()
        and (preonset_counts["futureOnsetMatrices"] >= 20).all()
    )
    if exact_pass:
        primary = "EXPLORATORY_PAPER_MATCH"
    elif directional_pass:
        primary = "EXPLORATORY_DIRECTIONAL_MATCH"
    else:
        primary = "EXPLORATORY_NON_SUPPORT"
    classifications = [primary]
    if padding_dominated:
        classifications.append("POSSIBLE_PIPELINE_ARTIFACT")
    if exact_pass or directional_pass:
        classifications.append("RETROSPECTIVE_ONLY_LEAD")
    classifications.append("NOT_PROMOTABLE")
    gate_rows.extend(
        [
            {
                "gateFamily": "GLOBAL",
                "candidateId": "BOTH",
                "criterion": "exactPanelBothCandidates",
                "passed": exact_pass,
            },
            {
                "gateFamily": "GLOBAL",
                "candidateId": "BOTH",
                "criterion": "directionalPanelBothCandidates",
                "passed": directional_pass,
            },
            {
                "gateFamily": "GLOBAL",
                "candidateId": "BOTH",
                "criterion": "paddingDominancePresent",
                "passed": padding_dominated,
            },
            {
                "gateFamily": "GLOBAL",
                "candidateId": "BOTH",
                "criterion": "prospectiveInitialAppearanceEligibility",
                "passed": prospective_eligible,
            },
        ]
    )
    return {
        "box": box,
        "order": order,
        "comparisons": comparisons,
        "bootstrap": pd.DataFrame(bootstrap_rows),
        "gates": pd.DataFrame(gate_rows),
        "dominance": dominance,
        "classification": {
            "schema": "eidosoma.e01.s19.l15.classification.v1",
            "researchStepId": RESEARCH_STEP_ID,
            "versionedStepId": VERSION,
            "completionStatus": "COMPLETE_AWAITING_MANDATORY_HUMAN_REVIEW",
            "primaryClassification": primary,
            "classifications": classifications,
            "exactPanelPassedBothCandidates": exact_pass,
            "directionalPanelPassedBothCandidates": directional_pass,
            "paddingDominated": padding_dominated,
            "prospectiveInitialAppearanceEligible": prospective_eligible,
            "promotionStatus": "NOT_PROMOTABLE",
            "authorCodeIdentityClaimed": False,
            "s18ProspectiveStatusChanged": False,
            "s18CausalStatusChanged": False,
        },
    }


def directory_bytes(root: Path) -> int:
    return int(sum(path.stat().st_size for path in root.rglob("*") if path.is_file()))


def artifact_manifest(path: Path, root: Path) -> None:
    rows = []
    for item in sorted(root.rglob("*")):
        if not item.is_file() or item == path:
            continue
        rows.append(
            {
                "path": str(item.relative_to(root)),
                "bytes": item.stat().st_size,
                "sha256": sha256_file(item),
            }
        )
    write_json(
        path,
        {
            "schema": "eidosoma.e01.s19.l15.artifact_manifest.v1",
            "researchStepId": RESEARCH_STEP_ID,
            "versionedStepId": VERSION,
            "createdAtUtc": utc_now(),
            "fileCount": len(rows),
            "files": rows,
        },
    )


def scientific_artifact_hashes() -> dict[str, str]:
    names = (
        "execution_status.parquet",
        "trajectory_manifest.parquet",
        "trajectory_length_results.parquet",
        "padding_geometry_results.parquet",
        "prevalence_decomposition.parquet",
        "padded_target_manifest.parquet",
        "feature_manifest.parquet",
        "source_execution_results.parquet",
        "suffix_invariance_results.parquet",
        "training_history.parquet",
        "prediction_results.parquet",
        "all_cell_metrics.parquet",
        "valid_cell_metrics.parquet",
        "padding_cell_metrics.parquet",
        "accuracy_decomposition.parquet",
        "diagnostic_results.parquet",
        "negative_control_results.parquet",
        "paper_boxplot_comparison.csv",
        "paper_model_order_results.csv",
        "paired_model_comparisons.parquet",
        "bootstrap_results.parquet",
        "scientific_gate_results.parquet",
        "padding_dominance_results.parquet",
        "classification.json",
    )
    return {name: sha256_file(OUTPUT_ROOT / name) for name in names}


def write_s19_root_manifest() -> None:
    path = S19_ROOT / "artifact_manifest.json"
    rows = []
    for item in sorted(S19_ROOT.rglob("*")):
        if not item.is_file() or item == path:
            continue
        rows.append(
            {
                "path": str(item.relative_to(S19_ROOT)),
                "bytes": item.stat().st_size,
                "sha256": sha256_file(item),
            }
        )
    write_json(
        path,
        {
            "schema": "eidosoma.e01.s19.root_artifact_manifest.v3",
            "currentLoop": "L15",
            "createdAtUtc": utc_now(),
            "fileCount": len(rows),
            "files": rows,
        },
    )


def regenerate_and_validate() -> dict[str, Any]:
    """Regenerate every untouched unit and scientific feature from its seed."""

    wall_start = time.perf_counter()
    child_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    if REGEN_TRAJECTORY_CACHE.exists():
        shutil.rmtree(REGEN_TRAJECTORY_CACHE)
    if REGEN_FEATURE_CACHE.exists():
        shutil.rmtree(REGEN_FEATURE_CACHE)
    REGEN_TRAJECTORY_CACHE.mkdir(parents=True)
    REGEN_FEATURE_CACHE.mkdir(parents=True)
    primary_manifest = pd.read_parquet(OUTPUT_ROOT / "trajectory_manifest.parquet")
    outputs = simulate_batch(range(MATRIX_COUNT), REGEN_TRAJECTORY_CACHE)
    failures = [row for output in outputs for row in output["failures"]]
    regenerated = pd.DataFrame(
        [row for output in outputs for row in output["trajectories"]]
    ).sort_values(["matrixIndex", "candidateId"], kind="stable")
    if failures or len(regenerated) != MATRIX_COUNT * len(CANDIDATE_IDS):
        raise RuntimeError("L15 trajectory regeneration failed")
    keys = ["matrixIndex", "candidateId"]
    comparison = primary_manifest.merge(
        regenerated,
        on=keys,
        suffixes=("Primary", "Regenerated"),
        validate="one_to_one",
    )
    trajectory_fields = (
        "trajectoryId",
        "trajectorySha256",
        "betaSha256",
        "initialStateSha256",
        "terminalStatus",
        "completedFissions",
        "selectedClockLength",
        "postFissionBoundaryCount",
    )
    comparison_rows: list[dict[str, Any]] = []
    for row in comparison.to_dict(orient="records"):
        passed = all(
            row[f"{field}Primary"] == row[f"{field}Regenerated"]
            for field in trajectory_fields
        )
        comparison_rows.append(
            {
                "candidateId": row["candidateId"],
                "matrixIndex": int(row["matrixIndex"]),
                "trajectoryFieldsExact": passed,
                "primaryTrajectorySha256": row["trajectorySha256Primary"],
                "regeneratedTrajectorySha256": row["trajectorySha256Regenerated"],
            }
        )
    trajectory_validation = pd.DataFrame(comparison_rows)
    if not trajectory_validation["trajectoryFieldsExact"].all():
        raise RuntimeError("L15 trajectory exact regeneration mismatch")

    regenerated_for_features = regenerated.copy()
    regenerated_for_features["cachePath"] = regenerated_for_features[
        "cachePath"
    ].astype(str)
    feature_outputs = run_feature_batch(
        regenerated_for_features, REGEN_FEATURE_CACHE, workers=8
    )
    feature_failures = [row["failure"] for row in feature_outputs if row.get("failure")]
    if feature_failures or len(feature_outputs) != MATRIX_COUNT * len(CANDIDATE_IDS):
        raise RuntimeError("L15 feature regeneration failed")
    feature_rows: list[dict[str, Any]] = []
    for candidate_id in CANDIDATE_IDS:
        for matrix_index in range(MATRIX_COUNT):
            primary_path = feature_cache_path(
                TRAJECTORY_FEATURE_CACHE, candidate_id, matrix_index
            )
            regenerated_path = feature_cache_path(
                REGEN_FEATURE_CACHE, candidate_id, matrix_index
            )
            with (
                np.load(primary_path, allow_pickle=False) as primary,
                np.load(regenerated_path, allow_pickle=False) as regenerated_payload,
            ):
                names_exact = set(primary.files) == set(regenerated_payload.files)
                array_exact = names_exact and all(
                    np.array_equal(
                        primary[name],
                        regenerated_payload[name],
                        equal_nan=True,
                    )
                    if np.issubdtype(primary[name].dtype, np.inexact)
                    else np.array_equal(primary[name], regenerated_payload[name])
                    for name in primary.files
                )
                feature_rows.append(
                    {
                        "candidateId": candidate_id,
                        "matrixIndex": matrix_index,
                        "fieldSetExact": names_exact,
                        "scientificArraysExact": array_exact,
                        "fieldCount": len(primary.files),
                    }
                )
    feature_validation = pd.DataFrame(feature_rows)
    if not feature_validation["scientificArraysExact"].all():
        raise RuntimeError("L15 feature exact regeneration mismatch")
    write_parquet(
        CACHE_ROOT / "trajectory_regeneration_validation.parquet", trajectory_validation
    )
    write_parquet(
        CACHE_ROOT / "feature_regeneration_validation.parquet", feature_validation
    )
    model_replay = pd.read_parquet(CACHE_ROOT / "model_replay.parquet")
    checked = model_replay.loc[model_replay["checked"]]
    child_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    return {
        "schema": "eidosoma.e01.s19.l15.regeneration_validation.v1",
        "validatedAtUtc": utc_now(),
        "trajectoryUnitsCompared": len(trajectory_validation),
        "trajectoryUnitsExact": int(
            trajectory_validation["trajectoryFieldsExact"].sum()
        ),
        "featureUnitsCompared": len(feature_validation),
        "featureUnitsExact": int(feature_validation["scientificArraysExact"].sum()),
        "modelReplaysCompared": len(checked),
        "modelReplaysExact": int(checked["passed"].sum()),
        "wallSeconds": time.perf_counter() - wall_start,
        "childCpuSeconds": (child_after.ru_utime + child_after.ru_stime)
        - (child_before.ru_utime + child_before.ru_stime),
        "trajectoryRegenerationCache": str(REGEN_TRAJECTORY_CACHE),
        "featureRegenerationCache": str(REGEN_FEATURE_CACHE),
        "passed": bool(
            len(trajectory_validation) == 400
            and trajectory_validation["trajectoryFieldsExact"].all()
            and len(feature_validation) == 400
            and feature_validation["scientificArraysExact"].all()
            and len(checked) == 24
            and checked["passed"].all()
        ),
    }


def save_figure(path: Path) -> Path:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    return path


def generate_figures(classification: dict[str, Any]) -> list[Path]:
    figure_root = OUTPUT_ROOT / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    for old in figure_root.glob("*.png"):
        old.unlink()
    paths: list[Path] = []
    lengths = pd.read_parquet(OUTPUT_ROOT / "trajectory_length_results.parquet")
    geometry = pd.read_parquet(OUTPUT_ROOT / "padding_geometry_results.parquet")
    metrics = pd.read_parquet(OUTPUT_ROOT / "all_cell_metrics.parquet")
    valid_metrics = pd.read_parquet(OUTPUT_ROOT / "valid_cell_metrics.parquet")
    decomposition = pd.read_parquet(OUTPUT_ROOT / "accuracy_decomposition.parquet")
    controls = pd.read_parquet(OUTPUT_ROOT / "negative_control_results.parquet")
    suffix = pd.read_parquet(OUTPUT_ROOT / "suffix_invariance_results.parquet")
    order = pd.read_parquet(OUTPUT_ROOT / "paired_model_comparisons.parquet")
    dominance = pd.read_parquet(OUTPUT_ROOT / "padding_dominance_results.parquet")

    plt.figure(figsize=(7.4, 4.4))
    for candidate_id, group in lengths.loc[lengths["eligible"]].groupby("candidateId"):
        plt.hist(group["outputLength"], bins=24, alpha=0.55, label=candidate_id)
    plt.axvline(MAX_TARGET_LENGTH, color="black", linestyle="--", label="tensor extent")
    plt.xlabel("Valid output length (molecular steps)")
    plt.ylabel("Matrices")
    plt.legend()
    paths.append(
        save_figure(figure_root / "01_trajectory_length_and_padding_extent.png")
    )

    plt.figure(figsize=(7.0, 4.4))
    x = np.arange(len(geometry))
    width = 0.25
    plt.bar(x - width, geometry["validFraction"], width, label="valid fraction")
    plt.bar(x, geometry["validPrevalence"], width, label="valid prevalence")
    plt.bar(x + width, geometry["paddedPrevalence"], width, label="padded prevalence")
    plt.xticks(x, geometry["candidateId"], rotation=10)
    plt.ylabel("Fraction")
    plt.legend()
    paths.append(save_figure(figure_root / "02_padding_prevalence_geometry.png"))

    plt.figure(figsize=(8.5, 4.8))
    feature_order = list(PAPER_FEATURES)
    positions = np.arange(len(feature_order))
    for offset, candidate_id in zip((-0.15, 0.15), CANDIDATE_IDS, strict=True):
        values = [
            metrics.loc[
                metrics["candidateId"].eq(candidate_id)
                & metrics["featureId"].eq(feature_id)
                & metrics["conditionId"].eq(S11)
                & metrics["metricScope"].eq("ALL_CELL"),
                "accuracy",
            ].to_numpy()
            for feature_id in feature_order
        ]
        bp = plt.boxplot(
            values, positions=positions + offset, widths=0.24, patch_artist=True
        )
        for patch in bp["boxes"]:
            patch.set_alpha(0.5)
        plt.plot([], [], label=candidate_id)
    paper = pd.read_csv(L14_ROOT / "paper_figure5_digitization_lock.csv")
    plt.scatter(
        positions,
        [
            float(paper.loc[paper["featureId"].eq(feature), "median"].iloc[0])
            for feature in feature_order
        ],
        marker="x",
        s=55,
        color="black",
        label="paper median",
    )
    plt.xticks(positions, ["PhiRL", "change", "raw", "flux", "dummy"])
    plt.ylabel("All-cell accuracy")
    plt.legend(ncol=3, fontsize=8)
    paths.append(save_figure(figure_root / "03_figure5_all_cell_reconstruction.png"))

    plt.figure(figsize=(8.5, 4.8))
    for offset, candidate_id in zip((-0.15, 0.15), CANDIDATE_IDS, strict=True):
        values = [
            valid_metrics.loc[
                valid_metrics["candidateId"].eq(candidate_id)
                & valid_metrics["featureId"].eq(feature_id)
                & valid_metrics["conditionId"].eq(S11),
                "accuracy",
            ].to_numpy()
            for feature_id in feature_order
        ]
        plt.boxplot(values, positions=positions + offset, widths=0.24)
        plt.plot([], [], label=candidate_id)
    plt.xticks(positions, ["PhiRL", "change", "raw", "flux", "dummy"])
    plt.ylabel("Valid molecular-cell accuracy")
    plt.legend()
    paths.append(save_figure(figure_root / "04_valid_cell_model_panel.png"))

    plt.figure(figsize=(7.2, 4.5))
    summary = (
        metrics.loc[metrics["featureId"].eq(P1)]
        .groupby(["candidateId", "conditionId"])["accuracy"]
        .median()
        .unstack()
        .reindex(columns=[S00, S01, S10, S11])
    )
    summary.T.plot(kind="bar", ax=plt.gca())
    plt.ylabel("Median accuracy")
    plt.xlabel("Train/score padding condition")
    plt.legend(title="")
    paths.append(save_figure(figure_root / "05_four_mask_conditions.png"))

    plt.figure(figsize=(7.0, 4.5))
    dec = decomposition.loc[
        decomposition["featureId"].eq(P1) & decomposition["conditionId"].eq(S11)
    ]
    dec.groupby("candidateId")[
        ["allCellAccuracy", "validCellAccuracy", "paddingCellAccuracy"]
    ].mean().plot(kind="bar", ax=plt.gca())
    plt.ylabel("Mean accuracy")
    plt.xticks(rotation=0)
    paths.append(save_figure(figure_root / "06_accuracy_decomposition.png"))

    plt.figure(figsize=(7.4, 4.4))
    diagnostic_plot = metrics.loc[
        metrics["featureId"].isin([P1, D0, D1, D2, D3])
        & metrics["conditionId"].eq(S11)
        & metrics["metricScope"].eq("ALL_CELL")
    ]
    diagnostic_plot.groupby(["featureId", "candidateId"])[
        "accuracy"
    ].median().unstack().plot(kind="bar", ax=plt.gca())
    plt.ylabel("Median all-cell accuracy")
    plt.xticks(rotation=20, ha="right")
    paths.append(save_figure(figure_root / "07_length_time_and_dummy_diagnostics.png"))

    plt.figure(figsize=(7.4, 4.4))
    control_plot = controls.loc[controls["accuracy"].notna()].copy()
    control_plot.groupby(["controlId", "candidateId"])[
        "accuracy"
    ].median().unstack().plot(kind="bar", ax=plt.gca())
    plt.ylabel("Median all-cell accuracy")
    plt.xticks(rotation=30, ha="right", fontsize=7)
    paths.append(save_figure(figure_root / "08_negative_controls.png"))

    plt.figure(figsize=(7.0, 4.4))
    plt.scatter(lengths["inputLength"], lengths["outputLength"], s=8, alpha=0.4)
    grid = np.arange(
        max(1, int(lengths["inputLength"].min())), int(lengths["inputLength"].max()) + 1
    )
    plt.plot(grid, infer_output_length(grid), color="black", linewidth=1.2)
    plt.xlabel("First-quarter valid length")
    plt.ylabel("Final-three-quarter valid length")
    paths.append(save_figure(figure_root / "09_length_boundary_determinability.png"))

    plt.figure(figsize=(7.0, 4.4))
    suffix.groupby("candidateId")[["p1MaximumAbsoluteChange"]].median().plot(
        kind="bar", ax=plt.gca()
    )
    plt.ylabel("Median |completed-fit change| under suffix perturbation")
    plt.xticks(rotation=0)
    paths.append(
        save_figure(figure_root / "10_future_dependence_and_suffix_invariance.png")
    )

    plt.figure(figsize=(7.0, 4.4))
    order_plot = order.pivot(
        index="comparatorFeatureId",
        columns="candidateId",
        values="medianSplitDifference",
    )
    order_plot.plot(kind="bar", ax=plt.gca())
    plt.axhline(0, color="black", linewidth=0.8)
    plt.ylabel("Median PhiRL − comparator accuracy")
    plt.xticks(rotation=20, ha="right")
    paths.append(save_figure(figure_root / "11_model_order_effects.png"))

    plt.figure(figsize=(7.2, 3.8))
    table = dominance.set_index("candidateId")[
        [
            "allMinusValidMedian",
            "correctPredictionsFromPaddingMedian",
            "validLabelShuffleRetention",
        ]
    ]
    image_values = table.to_numpy(dtype=np.float64)
    image_values = np.nan_to_num(image_values, nan=0.0)
    plt.imshow(
        image_values,
        aspect="auto",
        cmap="magma",
        vmin=0,
        vmax=max(1.0, float(image_values.max())),
    )
    plt.xticks(
        range(table.shape[1]),
        ["all−valid", "correct from pad", "shuffle retention"],
        rotation=15,
    )
    plt.yticks(range(table.shape[0]), table.index)
    plt.colorbar(label="Value")
    paths.append(save_figure(figure_root / "12_padding_dominance_matrix.png"))

    plt.figure(figsize=(7.4, 3.8))
    gate = pd.read_parquet(OUTPUT_ROOT / "scientific_gate_results.parquet")
    gate_plot = gate.assign(passedNumeric=gate["passed"].astype(int)).pivot_table(
        index="criterion",
        columns="candidateId",
        values="passedNumeric",
        aggfunc="max",
        fill_value=0,
    )
    plt.imshow(gate_plot.to_numpy(), aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    plt.xticks(range(gate_plot.shape[1]), gate_plot.columns)
    plt.yticks(range(gate_plot.shape[0]), gate_plot.index, fontsize=7)
    plt.colorbar(ticks=[0, 1], label="Gate")
    paths.append(save_figure(figure_root / "13_final_decision_matrix.png"))
    return paths


def render_reports(
    classification: dict[str, Any], figures: list[Path]
) -> tuple[str, str]:
    geometry = pd.read_parquet(OUTPUT_ROOT / "padding_geometry_results.parquet")
    box = pd.read_csv(OUTPUT_ROOT / "paper_boxplot_comparison.csv")
    dominance = pd.read_parquet(OUTPUT_ROOT / "padding_dominance_results.parquet")
    valid = pd.read_parquet(OUTPUT_ROOT / "valid_cell_metrics.parquet")
    suffix = pd.read_parquet(OUTPUT_ROOT / "suffix_invariance_results.parquet")
    geometry_lines = "\n".join(
        f"- {row.candidateId}: {int(row.eligibleMatrixCount)}/200 eligible; q={row.validFraction:.4f}; real-label prevalence={row.validPrevalence:.4f}; padded prevalence={row.paddedPrevalence:.4f}; padded dummy={row.paddedDummyAccuracy:.4f}."
        for row in geometry.itertuples(index=False)
    )
    panel_lines = []
    for candidate_id in CANDIDATE_IDS:
        values = box.loc[box["candidateId"].eq(candidate_id)]
        panel_lines.append(
            "- "
            + candidate_id
            + ": "
            + "; ".join(
                f"{row.featureId}={row.observedMedian:.4f} (paper≈{row.paperMedian:.4f})"
                for row in values.itertuples(index=False)
            )
            + "."
        )
    order_lines = "\n".join(
        f"- {row.candidateId}, PhiRL vs {row.comparatorFeatureId}: median Δ={row.medianSplitDifference:+.4f}; ordering={bool(row.medianSplitDifference > 0)}; Mann–Whitney p={row.mannWhitneyP:.4g}; matrix-bootstrap 95% CI [{row.bootstrapLower95:+.4f}, {row.bootstrapUpper95:+.4f}]."
        for row in pd.read_parquet(
            OUTPUT_ROOT / "paired_model_comparisons.parquet"
        ).itertuples(index=False)
    )
    valid_lines = "\n".join(
        f"- {candidate_id}: P1 valid-cell accuracy median={group['accuracy'].median():.4f}, balanced accuracy median={group['balancedAccuracy'].median():.4f}, AUPRC median={group['auprc'].median():.4f}."
        for candidate_id, group in valid.loc[
            valid["featureId"].eq(P1) & valid["conditionId"].eq(S11)
        ].groupby("candidateId")
    )
    dominance_lines = "\n".join(
        f"- {row.candidateId}: all−valid={row.allMinusValidMedian:+.4f}; fraction correct from padding={row.correctPredictionsFromPaddingMedian:.4f}; length/boundary within 0.03={row.lengthOrBoundaryWithinPoint03}; shuffled-label retention={row.validLabelShuffleRetention:.4f}; padding-dominated={row.paddingDominated}."
        for row in dominance.itertuples(index=False)
    )
    figure_markdown = "\n\n".join(
        f"![{path.stem}](figures/{path.name})\n\n*Figure {index}. {path.stem.replace('_', ' ')}.*"
        for index, path in enumerate(figures, 1)
    )
    report = f"""# S19-L15 Full Results — Untouched Padding/Length Panel Discrimination

## Top summary

- **Research step:** `{VERSION}`
- **Completion status:** `COMPLETE_AWAITING_MANDATORY_HUMAN_REVIEW`
- **Outcome classification:** `{classification["primaryClassification"]}`; `{", ".join(classification["classifications"])}`
- **Artifacts written:** the full registered machine-readable tables, 13 figures, immutable/source/seed/runtime/storage/regeneration manifests, this report, and the S19 handoff.
- **Validation:** all fixtures passed; 400/400 untouched trajectories and 400/400 complete per-trajectory feature payloads regenerated exactly; 24/24 registered model replays passed; prior artifacts remained immutable.
- **Central boundary:** all-cell padding resemblance is forensic only. Padding cells are not molecular observations, completed-fit PhiRL is future-dependent, and S18 prospective prediction and causal-control conclusions are unchanged.
- **Recommended next action:** mandatory human review; do not activate L16, S20, E02, author contact, intervention work, or report generation automatically.

## Frozen question and why the larger run was warranted

L14 established exact tensor replay but stopped before any MLP because its 100-matrix candidate-specific padding arithmetic straddled the digitized dummy interval. It also showed that first-quarter length almost exactly identifies the padded suffix boundary. L15 therefore used a prospectively frozen, new 200-matrix paired cohort to test the previously unopened mechanism: whether the exact S16 MLP, trained and scored over ordinary zero padding, produces the complete Figure 5 panel. The larger scope was set before outcomes to distinguish a stable mechanism from ten-split noise.

## Immutable methods

Candidate 2 used h=0.6031526490073492 and first-daughter continuation. Candidate 3 used h=0.5613315384859516 and random-nonempty continuation. Both used the frozen overshoot rule and 100-fission selected molecular clock. The sole target was strict adjacent-incoming `H>0.9`. The exact S16 288,789-parameter CPU-float64 MLP, quarter cutoff, right-zero padding, fixed feature construction, ten 128/32/40 matrix splits, and paper digitization were locked and pushed before any outcome. No threshold, model, feature, candidate, or digitization was selected from the result.

## Cohort and padding geometry

{geometry_lines}

The algebraic identity `p_padded = p_valid × q_valid` passed. Target zeros beyond the valid suffix were made fully visible in every all-cell result and were never described as physical states.

## Paper-facing all-cell panel

{chr(10).join(panel_lines)}

{order_lines}

Exact paper-panel and broader directional gates were frozen separately. The primary machine classification above follows those gates in both candidates; favorable pooling was prohibited.

## Valid molecular-cell performance

{valid_lines}

The adjacent-H target was already positive by the quarter cutoff in nearly every matrix. Consequently, even accuracy on valid suffix cells is future-state occupancy rather than a scientifically eligible test of initial appearance. Completed-fit P1 also remains explicitly future-dependent; suffix perturbation changed P1 while P2 was invariant on every sentinel (`{int(suffix["passed"].sum())}/{len(suffix)}`).

## Padding and length diagnostics

{dominance_lines}

The registered majority, time, input-length, deterministic-boundary and transformed controls separate class prevalence, output position, boundary inference and molecular features. A padding-dominated result cannot support prediction of self-replication even if it resembles the paper's numerical boxplots.

## Figures

{figure_markdown}

## Validation, provenance and limitations

The method contract was committed and pushed before cohort generation. A new domain-separated seed root had zero detected overlap with prior seed material or input hashes. Incomplete/extinct/overflow units were retained under registered statuses and never replaced. All 400 trajectory identities, clocks, fission counts, matrix/initial-state hashes and complete feature arrays were independently regenerated from fresh caches. Model probability replay was exact for every registered sentinel fit. Temporary trajectories, tensors and model outputs remained under `/cache/e01_s19_l15`; only compact evidence was promoted. Technical amendment 001 repaired only Figure 11's input-table selection; amendment 002 repaired only a report Boolean-field lookup by rendering its already registered defining expression. Both failed partial assemblies remain quarantined, and all {len(scientific_artifact_hashes())} frozen scientific hashes were exact before and after each amendment.

This remains adaptive forensic work on a preprint with missing padding, target and implementation semantics. Numerical resemblance cannot identify author code and cannot rescue failure on real cells, future independence, incremental value beyond H/stability, or causal control.

## Mandatory handoff

Stop here. No downstream step is active. The human reviewer may decide whether the result warrants a separately locked untouched confirmation, another narrow forensic question, author-code wait, S20 closeout/confirmation, E02 preparation, or pause.
"""
    summary = f"""# S19-L15 Decision Summary

**Status:** complete; mandatory human review required.
**Primary classification:** `{classification["primaryClassification"]}`.
**Additional classifications:** `{", ".join(classification["classifications"])}`.

The new 200-matrix paired run executed the full S16 all-cell padding mechanism that L14 never opened.

{geometry_lines}

{chr(10).join(panel_lines)}

Padding-dominance status: `{classification["paddingDominated"]}`. Exact full-panel gate in both candidates: `{classification["exactPanelPassedBothCandidates"]}`. Directional full-panel gate in both candidates: `{classification["directionalPanelPassedBothCandidates"]}`.

This is forensic only. Return for mandatory human review; no L16, S20, E02, author contact, intervention, or report bundle is active.
"""
    return report, summary


def append_root_ledgers(classification: dict[str, Any]) -> None:
    now = utc_now()
    ledger_path = S19_ROOT / "self_improvement_ledger.parquet"
    ledger = pd.read_parquet(ledger_path)
    sequence = int(pd.to_numeric(ledger["ledgerSequence"]).max()) + 1
    new = pd.DataFrame(
        [
            {
                "appendOnly": True,
                "beliefBeforeLoop": "L14's candidate-bracketing dummy arithmetic and nearly deterministic length-boundary relation left the unexecuted all-cell MLP panel as the highest-leverage Figure-5 mechanism.",
                "failureOrAmbiguityTargeted": "Whether unmasked zero padding plus trajectory-length information can explain the complete Figure-5 model panel rather than the dummy alone.",
                "informationGainRationale": "A new 200-matrix paired cohort tests the already locked convention with greater precision and without changing the target, metric, architecture or simulator candidates.",
                "learned": f"{classification['primaryClassification']}; {','.join(classification['classifications'])}.",
                "ledgerSequence": sequence,
                "loopId": "S19-L15",
                "motivatingEvidence": "L14 stopped before model fitting; q and dummy results bracketed the paper while input length almost exactly determined the output padding boundary.",
                "proposedNextTest": "None active; mandatory human review determines the next action.",
                "recordPhase": "POST_LOOP_RESULT_AND_HUMAN_REVIEW_HANDOFF",
                "remainingPlausibleHypotheses": "Author-specific target, padding/truncation, feature preprocessing, dataset or implementation semantics remain distinguishable only through a separately locked step or code release.",
                "selectedHypotheses": "Exact S16 adjacent-H target, six registered features, four mask conditions and length diagnostics on 200 untouched matrices.",
                "timestampUtc": now,
                "weakenedHypotheses": "The frozen full-panel and control results identify which padding/length explanations fail or remain viable without changing earlier evidence.",
            }
        ]
    )
    combined = pd.concat([ledger, new], ignore_index=True)
    if len(combined) != len(ledger) + 1:
        raise RuntimeError("S19 self-improvement ledger append failed")
    write_parquet(ledger_path, combined)
    with (S19_ROOT / "SELF_IMPROVEMENT_LEDGER.md").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write(
            f"\n\n## S19-L15 — post-loop result ({now})\n\n"
            f"- Belief before: the unexecuted all-cell padding/length mechanism warranted a larger untouched test.\n"
            f"- Learned: `{classification['primaryClassification']}`; `{', '.join(classification['classifications'])}`.\n"
            "- Boundary: result is forensic; every prior verdict remains immutable.\n"
            "- Next: mandatory human review; no next loop is active.\n"
        )

    candidate_path = S19_ROOT / "candidate_registry.parquet"
    candidate = pd.read_parquet(candidate_path)
    registry_order = int(pd.to_numeric(candidate["registryOrder"]).max()) + 1
    row = pd.DataFrame(
        [
            {
                "branchCount": 4,
                "bundleId": "L15_UNTOUCHED_PADDING_PANEL",
                "candidateId": "S19-L15-UNTOUCHED-PADDING-LENGTH-PANEL",
                "candidateSpecificSuccess": 0,
                "completedFitLeakage": 1,
                "computeEfficiency": 3,
                "crossCandidateDiscriminability": 5,
                "deterministicHReuse": 1,
                "explanatoryLeverage": 5,
                "frozenRank": 1,
                "independenceFromPriorOutcomeSelection": 3,
                "outcomeGuidedThresholdSelection": 0,
                "paperFingerprintSpecificity": 5,
                "proposedSpecification": "New 200-matrix paired cohort; exact S16 target/features/model; full 2x2 padding factorial and registered controls",
                "rankingScore": 27.0,
                "registryOrder": registry_order,
                "selected": True,
                "selectionReason": "Explicit human authorization to try a larger L15 after L14's technical validation and arithmetic-only stop",
                "sourceGrounding": 2,
                "testability": 5,
                "undefinedAuthorSemantics": 1,
            }
        ]
    )
    write_parquet(candidate_path, pd.concat([candidate, row], ignore_index=True))

    loop_path = S19_ROOT / "loop_registry.yaml"
    registry = yaml.safe_load(loop_path.read_text())
    if any(item.get("loopId") == "S19-L15" for item in registry["loops"]):
        raise RuntimeError("L15 loop registry entry already exists")
    registry["loops"].append(
        {
            "loopId": "S19-L15",
            "versionedLoopId": VERSION,
            "status": "COMPLETE_AWAITING_MANDATORY_HUMAN_REVIEW",
            "authorized": True,
            "outcomeAccessed": True,
            "humanReviewRequiredAfter": True,
            "completed": True,
            "classification": classification["primaryClassification"],
            "nextStepActive": False,
        }
    )
    write_yaml(loop_path, registry)

    history_path = S19_ROOT / "human_review_history.json"
    history = json.loads(history_path.read_text())
    history["history"].append(
        {
            "decision": "AUTHORIZE_L15_AND_ALLOW_LARGER_UNTOUCHED_RUN",
            "loopId": "S19-L15",
            "scope": VERSION,
            "source": "explicit_human_direction",
            "recordedAtUtc": now,
            "result": classification["primaryClassification"],
            "nextLoopAuthorized": False,
            "s20Activated": False,
            "status": "CONSUMED_AND_RETURNED_FOR_MANDATORY_REVIEW",
        }
    )
    history["pendingDecision"] = "POST_S19_L15_MANDATORY_HUMAN_REVIEW_REQUIRED"
    write_json(history_path, history)


def write_root_handoff(classification: dict[str, Any], report: str) -> None:
    write_json(
        S19_ROOT / "s19_status.json",
        {
            "researchStepId": "S19-L15",
            "status": "COMPLETE_AWAITING_MANDATORY_HUMAN_REVIEW",
            "outcomeClassification": classification["primaryClassification"],
            "validationResult": "PASS_IMMUTABLE_SEED_FIXTURE_TRAJECTORY_FEATURE_MODEL_STORAGE_REGENERATION",
            "artifactsWritten": [
                str(OUTPUT_ROOT / "S19_L15_FULL_RESULTS.md"),
                str(OUTPUT_ROOT / "classification.json"),
                str(OUTPUT_ROOT / "artifact_manifest.json"),
                str(S19_ROOT / "research_step_full_results.md"),
            ],
            "caveatsOrBlockers": [
                "adaptive_exploratory_forensic_reconstruction",
                "padding_cells_are_not_molecular_observations",
                "completed_fit_phirl_is_future_dependent",
                "adjacent_H_target_is_label_coupled",
                "author_tensor_semantics_unavailable",
                "S18_statuses_unchanged",
            ],
            "recommendedNextAction": "MANDATORY_HUMAN_REVIEW_KEEP_L16_S20_E02_AUTHOR_CONTACT_INTERVENTIONS_AND_REPORT_BUNDLE_INACTIVE",
        },
    )
    root_report = report.replace(
        "# S19-L15 Full Results — Untouched Padding/Length Panel Discrimination",
        "# S19 Current-Step Handoff — S19-L15",
    ).replace("](figures/", "](loops/L15/figures/")
    atomic_text(S19_ROOT / "research_step_full_results.md", root_report)


def finalize_phase() -> None:
    start = time.perf_counter()
    if not (OUTPUT_ROOT / "prediction_results.parquet").exists():
        raise RuntimeError("L15 model execution missing")
    amendment = json.loads(AMENDMENT_001_PATH.read_text(encoding="utf-8"))
    if amendment["amendmentId"] != "S19-L15-TECHNICAL-AMENDMENT-001":
        raise RuntimeError("unexpected L15 technical amendment identity")
    amendment_002 = json.loads(AMENDMENT_002_PATH.read_text(encoding="utf-8"))
    if amendment_002["amendmentId"] != "S19-L15-TECHNICAL-AMENDMENT-002":
        raise RuntimeError("unexpected L15 technical amendment 002 identity")
    before_science = (
        scientific_artifact_hashes()
        if (OUTPUT_ROOT / "classification.json").is_file()
        else None
    )
    tables = comparison_and_gate_tables()
    write_csv(OUTPUT_ROOT / "paper_boxplot_comparison.csv", tables["box"])
    write_csv(OUTPUT_ROOT / "paper_model_order_results.csv", tables["order"])
    write_parquet(
        OUTPUT_ROOT / "paired_model_comparisons.parquet", tables["comparisons"]
    )
    write_parquet(OUTPUT_ROOT / "bootstrap_results.parquet", tables["bootstrap"])
    write_parquet(OUTPUT_ROOT / "scientific_gate_results.parquet", tables["gates"])
    write_parquet(
        OUTPUT_ROOT / "padding_dominance_results.parquet", tables["dominance"]
    )
    write_json(OUTPUT_ROOT / "classification.json", tables["classification"])
    classification = tables["classification"]
    after_science = scientific_artifact_hashes()
    if before_science is not None and before_science != after_science:
        raise RuntimeError("L15 technical amendment changed scientific artifacts")
    shutil.copyfile(AMENDMENT_001_PATH, OUTPUT_ROOT / "technical_amendment_001.json")
    shutil.copyfile(AMENDMENT_002_PATH, OUTPUT_ROOT / "technical_amendment_002.json")
    write_csv(
        OUTPUT_ROOT / "technical_amendment_ledger.csv",
        pd.DataFrame(
            [
                {
                    "amendmentId": amendment["amendmentId"],
                    "failureStage": amendment["failureStage"],
                    "scope": amendment["scope"],
                    "scientificValueChanged": False,
                    "scientificArtifactCountCompared": len(after_science),
                    "scientificHashesExact": before_science == after_science,
                    "failedAttemptPreservedPath": amendment["freshFigureCache"],
                    "status": "COMPLETE_VALUE_PRESERVING_REPORTING_REPAIR",
                },
                {
                    "amendmentId": amendment_002["amendmentId"],
                    "failureStage": amendment_002["failureStage"],
                    "scope": amendment_002["scope"],
                    "scientificValueChanged": False,
                    "scientificArtifactCountCompared": len(after_science),
                    "scientificHashesExact": before_science == after_science,
                    "failedAttemptPreservedPath": amendment_002["freshFigureCache"],
                    "status": "COMPLETE_VALUE_PRESERVING_REPORTING_REPAIR",
                },
            ]
        ),
    )

    baseline = json.loads((OUTPUT_ROOT / "immutable_prior_baseline.json").read_text())
    immutable = revalidate_immutable(baseline)
    write_json(OUTPUT_ROOT / "immutable_prior_validation.json", immutable)
    if not immutable["passed"]:
        raise RuntimeError("prior artifact changed during L15")
    regeneration = regenerate_and_validate()
    write_json(OUTPUT_ROOT / "regeneration_validation.json", regeneration)
    if not regeneration["passed"]:
        raise RuntimeError("L15 exact regeneration failed")

    figures = generate_figures(classification)
    report, summary = render_reports(classification, figures)
    atomic_text(OUTPUT_ROOT / "S19_L15_FULL_RESULTS.md", report)
    atomic_text(OUTPUT_ROOT / "loop_decision_summary.md", summary)
    failures = pd.read_csv(OUTPUT_ROOT / "failure_ledger.csv")
    unresolved_failures = failures.loc[
        ~failures["failureType"].eq("PREOUTCOME_YAML_SCHEMA_DEFECT_REPAIRED")
    ]
    if len(unresolved_failures):
        raise RuntimeError("unexpected unresolved L15 failure ledger entries")

    cache_bytes = directory_bytes(CACHE_ROOT)
    artifact_bytes = directory_bytes(OUTPUT_ROOT)
    storage = {
        "schema": "eidosoma.e01.s19.l15.storage_validation.v1",
        "retainedBytes": artifact_bytes,
        "retainedGiB": artifact_bytes / 2**30,
        "temporaryBytes": cache_bytes,
        "temporaryGiB": cache_bytes / 2**30,
        "retainedLimitGiB": 40,
        "temporaryLimitGiB": 120,
        "compiledOrBulkCacheUnderArtifacts": False,
        "passed": bool(artifact_bytes <= 40 * 2**30 and cache_bytes <= 120 * 2**30),
    }
    write_json(OUTPUT_ROOT / "storage_validation.json", storage)
    if not storage["passed"]:
        raise RuntimeError("L15 storage ceiling exceeded")

    runtime = json.loads((OUTPUT_ROOT / "runtime_manifest.json").read_text())
    observed_cpu_seconds = float(
        runtime.get("generationChildCpuSeconds", 0.0)
        + runtime.get("featureChildCpuSeconds", 0.0)
        + runtime.get("modelChildCpuSeconds", 0.0)
        + regeneration["childCpuSeconds"]
    )
    observed_wall_seconds = float(
        runtime.get("generationWallSeconds", 0.0)
        + runtime.get("featureWallSeconds", 0.0)
        + runtime.get("modelWallSeconds", 0.0)
        + regeneration["wallSeconds"]
        + (time.perf_counter() - start)
    )
    if observed_cpu_seconds > 160 * 3600 or observed_wall_seconds > 72 * 3600:
        raise RuntimeError("L15 runtime ceiling exceeded")
    runtime.update(
        {
            "status": "COMPLETE_AWAITING_MANDATORY_HUMAN_REVIEW",
            "completedAtUtc": utc_now(),
            "finalizationWallSecondsAtWrite": time.perf_counter() - start,
            "regeneratedTrajectoryCount": regeneration["trajectoryUnitsCompared"],
            "regeneratedFeatureUnitCount": regeneration["featureUnitsCompared"],
            "observedTrackedCpuSeconds": observed_cpu_seconds,
            "observedTrackedCpuHours": observed_cpu_seconds / 3600.0,
            "observedTrackedWallSeconds": observed_wall_seconds,
            "observedTrackedWallHours": observed_wall_seconds / 3600.0,
            "cpuHourCeiling": 160,
            "wallHourCeiling": 72,
            "reserveFraction": 0.15,
            "scopeReducedAfterOutcomeAccess": False,
            "runtimeDrivenShortcutUsed": False,
        }
    )
    write_json(OUTPUT_ROOT / "runtime_manifest.json", runtime)
    artifact_manifest(OUTPUT_ROOT / "artifact_manifest.json", OUTPUT_ROOT)
    append_root_ledgers(classification)
    write_root_handoff(classification, report)
    artifact_manifest(OUTPUT_ROOT / "artifact_manifest.json", OUTPUT_ROOT)
    write_s19_root_manifest()
    missing = [
        name for name in REQUIRED_ARTIFACTS if not (OUTPUT_ROOT / name).is_file()
    ]
    if missing:
        raise RuntimeError(f"required L15 artifacts missing: {missing}")
    root_links = [
        line.split("](loops/L15/", 1)[1].rstrip(")")
        for line in (S19_ROOT / "research_step_full_results.md")
        .read_text()
        .splitlines()
        if "](loops/L15/" in line
    ]
    if len(root_links) != len(figures) or not all(
        (S19_ROOT / "loops/L15" / link).is_file() for link in root_links
    ):
        raise RuntimeError("L15 root handoff figure links failed")
    write_json(
        CACHE_ROOT / "finalize_status.json",
        {
            "stage": "finalize",
            "completedAtUtc": utc_now(),
            "passed": True,
            "classification": classification["primaryClassification"],
            "requiredArtifactCount": len(REQUIRED_ARTIFACTS),
            "figureCount": len(figures),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=["prepare", "generate", "features", "execute", "finalize", "all"],
    )
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    if args.stage in ("prepare", "all"):
        prepare_phase()
    if args.stage in ("generate", "all"):
        generate_phase()
    if args.stage in ("features", "all"):
        features_phase()
    if args.stage in ("execute", "all"):
        execute_phase()
    if args.stage in ("finalize", "all"):
        finalize_phase()


if __name__ == "__main__":
    main()
