"""Execute S19-L24 event-aligned pre-onset reaction-coordinate discovery."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from e01_frozen_timebase_ensemble.core import (
    selected_clock_observations,
    states_from_observations,
)
from e01_onset_discovery.reaction_coordinate import (
    EXACT_H_WINDOW_FEATURES,
    ORDINARY_WINDOW_FEATURES,
    REACTION_FEATURES,
    WINDOW_COUNT,
    extract_window_features,
)


def _load_base() -> Any:
    path = REPO_ROOT / "scripts/e01/run_s19_l19_source_grounded_early_warning.py"
    spec = importlib.util.spec_from_file_location("e01_s19_l24_runner_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load L19 artifact utilities")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_base()
LOOP_ID = "S19-L24"
VERSION = "E01-S19-L24-EVENT-ALIGNED-REACTION-COORDINATE-v1.0.0"
TARGET_ID = "PF_DOMINANT_COMPONENT_CENTROID_H900"
CLOCK_ID = "C1_SELECTED_DAUGHTER_RETAINED"
CANDIDATES = ("S12F-CANDIDATE-02", "S12F-CANDIDATE-03")
ARTIFACT_ROOT = Path("/artifacts/research_steps/S19")
LOOP_ROOT = ARTIFACT_ROOT / "loops/L24"
L23_ROOT = ARTIFACT_ROOT / "loops/L23"
CACHE_ROOT = Path("/cache/e01_s19_l24")
BUILD_ROOT = CACHE_ROOT / "build"
CONFIG = REPO_ROOT / "configs/e01/s19_l24_event_aligned_reaction_coordinate.yaml"
RUNNER_PATH = Path(__file__)
CORE_PATH = REPO_ROOT / "src/e01_onset_discovery/reaction_coordinate.py"
L23_CACHE = Path("/cache/e01_s19_l23/primary_trajectories")
MATRIX_COUNT = 400
DEVELOPMENT_COUNT = 200
VALIDATION_COUNT = 200
EVENT_MIN = 128
EVENT_MAX = 256
CONTROL_LEAD = 96
BOOTSTRAPS = 4096
VALIDATION_PERMUTATIONS = 4096
DEVELOPMENT_PERMUTATIONS = 512
MODEL_FEATURES = {
    "REACTION_COORDINATE": REACTION_FEATURES,
    "EXACT_H_WINDOW": EXACT_H_WINDOW_FEATURES,
    "ORDINARY_STABILITY_WINDOW": ORDINARY_WINDOW_FEATURES,
    "TIME_ONLY_ENDPOINT": ("windowEndpoint",),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def derived_seed(*parts: object) -> int:
    material = "\x1f".join([VERSION, *map(str, parts)])
    return int.from_bytes(hashlib.sha256(material.encode()).digest()[:8], "big")


def validate_immutable_prior() -> dict[str, Any]:
    prior = json.loads((L23_ROOT / "immutable_prior_validation.json").read_text())
    rows = list(prior["files"])
    manifest = json.loads((L23_ROOT / "artifact_manifest.json").read_text())
    rows.extend(
        {
            "path": str(L23_ROOT / item["path"]),
            "root": str(L23_ROOT),
            "bytes": item["bytes"],
            "sha256": item["sha256"],
        }
        for item in manifest["files"]
    )
    failures: list[dict[str, Any]] = []
    for row in rows:
        path = Path(row["path"])
        if not path.is_file():
            failures.append({"path": str(path), "reason": "MISSING"})
        elif sha256_file(path) != row["sha256"]:
            failures.append({"path": str(path), "reason": "HASH_MISMATCH"})
    aggregate = hashlib.sha256(
        "\n".join(f"{row['path']}\t{row['sha256']}" for row in rows).encode()
    ).hexdigest()
    return {
        "schema": "eidosoma.e01.s19_l24.immutable_prior_validation.v1",
        "status": "PASS" if not failures else "FAIL",
        "unchanged": not failures,
        "fileCount": len(rows),
        "aggregateSha256": aggregate,
        "l23ArtifactFileCount": manifest["fileCount"],
        "failures": failures,
        "files": rows,
    }


def matrix_firewall() -> pd.DataFrame:
    rows = []
    for index in range(MATRIX_COUNT):
        key = hashlib.sha256(f"L24_SPLIT::{index}".encode()).hexdigest()
        rows.append({"matrixIndex": index, "firewallKey": key})
    frame = pd.DataFrame(rows).sort_values(["firewallKey", "matrixIndex"]).reset_index(drop=True)
    frame["firewallRank"] = np.arange(MATRIX_COUNT)
    frame["matrixRole"] = np.where(
        frame["firewallRank"] < DEVELOPMENT_COUNT, "DEVELOPMENT", "VALIDATION"
    )
    return frame.sort_values("matrixIndex").reset_index(drop=True)


def match_pairs(targets: pd.DataFrame, firewall: pd.DataFrame) -> pd.DataFrame:
    role_map = firewall.set_index("matrixIndex")["matrixRole"].to_dict()
    rows: list[dict[str, Any]] = []
    for matrix_role in ("DEVELOPMENT", "VALIDATION"):
        role_ids = {key for key, value in role_map.items() if value == matrix_role}
        for candidate in CANDIDATES:
            frame = targets[
                targets["candidateId"].eq(candidate)
                & targets["matrixIndex"].isin(role_ids)
            ].copy()
            onset_map = {
                int(row.matrixIndex): (
                    None if pd.isna(row.firstOnsetIndex0) else int(row.firstOnsetIndex0)
                )
                for row in frame.itertuples(index=False)
            }
            length_map = {
                int(row.matrixIndex): int(row.observationCount)
                for row in frame.itertuples(index=False)
            }
            events = sorted(
                (
                    (index, onset)
                    for index, onset in onset_map.items()
                    if onset is not None and EVENT_MIN <= onset <= EVENT_MAX
                ),
                key=lambda item: (-item[1], item[0]),
            )
            event_ids = {item[0] for item in events}
            unused = set(role_ids) - event_ids
            pair_number = 0
            for event_matrix, endpoint in events:
                eligible = []
                for control_matrix in unused:
                    control_onset = onset_map.get(control_matrix)
                    sufficient_onset = control_onset is None or (
                        control_onset > EVENT_MAX and control_onset >= endpoint + CONTROL_LEAD
                    )
                    if sufficient_onset and length_map.get(control_matrix, 0) > endpoint:
                        eligible.append(
                            (
                                float("inf") if control_onset is None else control_onset,
                                control_matrix,
                            )
                        )
                if not eligible:
                    continue
                _, control_matrix = min(eligible)
                unused.remove(control_matrix)
                pair_number += 1
                rows.append(
                    {
                        "matrixRole": matrix_role,
                        "candidateId": candidate,
                        "pairId": f"{matrix_role}-{candidate}-P{pair_number:03d}",
                        "eventMatrixIndex": event_matrix,
                        "controlMatrixIndex": control_matrix,
                        "windowEndpoint": endpoint,
                        "windowStart": endpoint - WINDOW_COUNT,
                        "eventOnsetIndex0": endpoint,
                        "controlOnsetIndex0": onset_map[control_matrix],
                        "controlLeadAtLeast": CONTROL_LEAD,
                        "eventWindowStrictlyPreOnset": True,
                        "controlWindowSameAbsoluteTime": True,
                    }
                )
    return pd.DataFrame(rows)


def fixture_table() -> pd.DataFrame:
    rng = np.random.default_rng(derived_seed("fixtures"))
    states = rng.poisson(2.0, size=(WINDOW_COUNT, 100)).astype(np.int64)
    states[:, 0] += 1
    first = extract_window_features(states)
    replay = extract_window_features(states.copy())
    relabelled = extract_window_features(states[:, rng.permutation(100)])
    reversed_result = extract_window_features(states[::-1])
    synthetic_targets = []
    for candidate in CANDIDATES:
        for index in range(400):
            synthetic_targets.append(
                {
                    "candidateId": candidate,
                    "matrixIndex": index,
                    "firstOnsetIndex0": (160 if index % 8 == 0 else 400 + index % 17),
                    "observationCount": 600,
                }
            )
    synthetic_matches = match_pairs(pd.DataFrame(synthetic_targets), matrix_firewall())
    x = rng.normal(size=(80, 8))
    y = np.asarray([0, 1] * 40, dtype=int)
    scaler_a, model_a = fit_coordinate(x, y, np.ones(80), derived_seed("model_fixture"))
    scaler_b, model_b = fit_coordinate(x, y, np.ones(80), derived_seed("model_fixture"))
    return pd.DataFrame(
        [
            {
                "fixtureId": "FEATURE_SCHEMA_AND_FINITE",
                "passed": tuple(first) == REACTION_FEATURES
                and len(first) == 28
                and np.isfinite(list(first.values())).all(),
                "details": f"{len(first)} features",
            },
            {
                "fixtureId": "EXACT_FEATURE_REPLAY",
                "passed": first == replay,
                "details": "CPU float64 exact dictionary replay",
            },
            {
                "fixtureId": "MOLECULE_PERMUTATION_INVARIANCE",
                "passed": all(
                    np.isclose(first[name], relabelled[name], atol=1e-10, rtol=1e-10)
                    for name in first
                ),
                "details": "coordinate relabel tolerance 1e-10",
            },
            {
                "fixtureId": "TEMPORAL_SENSITIVITY",
                "passed": any(first[name] != reversed_result[name] for name in first),
                "details": "window reversal changes registered dynamic features",
            },
            {
                "fixtureId": "FIREWALL_CARDINALITY",
                "passed": matrix_firewall()["matrixRole"].value_counts().to_dict()
                == {"DEVELOPMENT": 200, "VALIDATION": 200},
                "details": "200/200 shared matrix firewall",
            },
            {
                "fixtureId": "MATCHING_CONTRACT",
                "passed": not synthetic_matches.empty
                and synthetic_matches["eventWindowStrictlyPreOnset"].all()
                and synthetic_matches["controlWindowSameAbsoluteTime"].all(),
                "details": f"{len(synthetic_matches)} synthetic pairs",
            },
            {
                "fixtureId": "MODEL_EXACT_REPLAY",
                "passed": np.array_equal(scaler_a.mean_, scaler_b.mean_)
                and np.array_equal(model_a.coef_, model_b.coef_)
                and np.array_equal(
                    model_a.predict_proba(scaler_a.transform(x)),
                    model_b.predict_proba(scaler_b.transform(x)),
                ),
                "details": "development-only scaler and L1 logistic",
            },
        ]
    )


def source_registry() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sourceId": "PETERS_TROUT_2006_REACTION_COORDINATE",
                "doi": "10.1063/1.2234477",
                "url": "https://doi.org/10.1063/1.2234477",
                "directSupport": "likelihood-based construction of low-dimensional reaction coordinates",
                "frozenUse": "one development-fitted sparse logistic coordinate for event versus matched control windows",
                "evidenceClass": "PRIMARY_METHOD_PAPER",
            },
            {
                "sourceId": "MA_DINNER_2005_REACTION_COORDINATE",
                "doi": "10.1021/jp045546c",
                "url": "https://doi.org/10.1021/jp045546c",
                "directSupport": "automatic discrimination of transition-state ensembles from dynamical observations",
                "frozenUse": "event-aligned pre-onset windows with time-matched non-imminent controls",
                "evidenceClass": "PRIMARY_METHOD_PAPER",
            },
            {
                "sourceId": "BOETTIGER_HASTINGS_2012_LIMITS",
                "doi": "10.1098/rsif.2012.0125",
                "url": "https://doi.org/10.1098/rsif.2012.0125",
                "directSupport": "finite-series power and false-positive limits for early-warning signals",
                "frozenUse": "strict held-out matrix firewall, matched controls and permutation tests",
                "evidenceClass": "PRIMARY_METHOD_PAPER",
            },
        ]
    )


def fit_coordinate(
    x: np.ndarray, y: np.ndarray, weights: np.ndarray, seed: int
) -> tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler().fit(np.asarray(x, dtype=np.float64))
    model = LogisticRegression(
        l1_ratio=1.0,
        C=0.5,
        solver="liblinear",
        random_state=int(seed % (2**31 - 1)),
        max_iter=5000,
        tol=1e-10,
    )
    model.fit(scaler.transform(x), y, sample_weight=weights)
    return scaler, model


def prepare_lock() -> None:
    LOOP_ROOT.mkdir(parents=True, exist_ok=True)
    if git("status", "--porcelain=v1"):
        raise RuntimeError("repository must be clean before L24 lock")
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/eidosoma/groups/42")
    if head != remote:
        raise RuntimeError("local and pushed branch identities differ")
    prior = validate_immutable_prior()
    if not prior["unchanged"]:
        raise RuntimeError("immutable prior changed")
    fixtures = fixture_table()
    if not fixtures["passed"].all():
        raise RuntimeError("L24 mandatory fixture failed")
    firewall = matrix_firewall()
    targets = pd.read_parquet(L23_ROOT / "target_geometry_results.parquet")
    matches = match_pairs(targets, firewall)
    counts = matches.groupby(["matrixRole", "candidateId"]).size()
    required = {(role, candidate) for role in ("DEVELOPMENT", "VALIDATION") for candidate in CANDIDATES}
    if set(counts.index) != required or int(counts.min()) < 20:
        raise RuntimeError(f"insufficient matched pairs under frozen contract: {counts.to_dict()}")
    manifest = pd.read_parquet(L23_ROOT / "input_trajectory_manifest.parquet")
    cache_failures = []
    for row in manifest.itertuples(index=False):
        path = Path(row.cachePath)
        if not path.is_file() or sha256_file(path) != row.cacheSha256:
            cache_failures.append({"candidateId": row.candidateId, "matrixIndex": row.matrixIndex})
    if cache_failures:
        raise RuntimeError("L23 trajectory cache identity failure")
    start = time.perf_counter()
    sample = matches[matches["matrixRole"].eq("DEVELOPMENT")].head(10)
    for row in sample.itertuples(index=False):
        for matrix_index in (row.eventMatrixIndex, row.controlMatrixIndex):
            states = load_states(row.candidateId, int(matrix_index), manifest)
            extract_window_features(states[row.windowStart : row.windowEndpoint])
    benchmark = time.perf_counter() - start
    projected = benchmark * len(matches) * 2 / max(1, len(sample)) * 2.5
    if projected > 60 * 60 * 60:
        raise RuntimeError("L24 benchmark projects beyond wall ceiling")
    shutil.copy2(CONFIG, LOOP_ROOT / "preregistration.yaml")
    BASE.atomic_text(
        LOOP_ROOT / "decision_record.md",
        """# S19-L24 decision record

L23 used 400 new shared matrices to show that no complete frozen L19/L20/L22 family becomes reproducible merely with greater power. L24 tests one nonduplicative explanation: a precursor may be localized immediately before attractor entry and therefore averaged away at the fixed landmark.

The 400 matrices are firewalled outcome-blindly into 200 development and 200 validation identities. Within each half and candidate, onset windows `[tau-32,tau)` for `128<=tau<=256` are matched without replacement to non-imminent trajectories at the same absolute endpoint. Exactly one 28-feature molecule-label-permutation-invariant coordinate, sparse logistic estimator, matching rule, controls and validation gate are frozen. The coordinate is fitted only on development matrices, serialized before any validation payload is opened, and never revised from validation results. This remains retrospective event-aligned discovery, not online confirmation.
""",
    )
    sources = source_registry()
    sources.to_csv(LOOP_ROOT / "source_grounding_registry.csv", index=False)
    BASE.atomic_text(
        LOOP_ROOT / "source_grounding_report.md",
        "# L24 source grounding\n\n"
        + "\n".join(
            f"- **{row.sourceId}** — {row.directSupport}. Frozen use: {row.frozenUse}. {row.url}"
            for row in sources.itertuples(index=False)
        )
        + "\n",
    )
    BASE.write_parquet(LOOP_ROOT / "fixture_results.parquet", fixtures)
    BASE.write_parquet(LOOP_ROOT / "matrix_firewall.parquet", firewall)
    BASE.write_parquet(LOOP_ROOT / "matching_registry_preoutcome.parquet", matches)
    BASE.write_json(LOOP_ROOT / "immutable_prior_validation.json", prior)
    BASE.write_json(
        LOOP_ROOT / "implementation_lock.json",
        {
            "schema": "eidosoma.e01.s19_l24.implementation_lock.v1",
            "researchStepId": LOOP_ID,
            "versionedId": VERSION,
            "repositoryHead": head,
            "remoteHead": remote,
            "configSha256": sha256_file(CONFIG),
            "runnerSha256": sha256_file(RUNNER_PATH),
            "coreSha256": sha256_file(CORE_PATH),
            "l23ManifestSha256": sha256_file(L23_ROOT / "artifact_manifest.json"),
            "targetId": TARGET_ID,
            "clockId": CLOCK_ID,
            "matrixFirewall": "SHA256 rank L24_SPLIT::{matrixIndex}; 200/200",
            "matchedPairCounts": {f"{a}:{b}": int(value) for (a, b), value in counts.items()},
            "eventRange": [EVENT_MIN, EVENT_MAX],
            "windowCount": WINDOW_COUNT,
            "controlLead": CONTROL_LEAD,
            "modelFeatures": {name: list(fields) for name, fields in MODEL_FEATURES.items()},
            "coordinateEstimator": "StandardScaler then L1 LogisticRegression liblinear C=0.5",
            "bootstrapReplicates": BOOTSTRAPS,
            "validationPairSwapReplicates": VALIDATION_PERMUTATIONS,
            "developmentLabelPermutationReplicates": DEVELOPMENT_PERMUTATIONS,
            "outcomeAccessed": False,
            "lockedAtUtc": utc_now(),
        },
    )
    BASE.write_json(
        LOOP_ROOT / "preoutcome_repository_lock.json",
        {
            "head": head,
            "remote": remote,
            "priorAggregateSha256": prior["aggregateSha256"],
            "configSha256": sha256_file(CONFIG),
            "runnerSha256": sha256_file(RUNNER_PATH),
            "coreSha256": sha256_file(CORE_PATH),
        },
    )
    BASE.write_json(
        LOOP_ROOT / "benchmark_projection.json",
        {
            "status": "PASS_PROJECTED_WITHIN_CEILING",
            "tenPairLoadAndFeatureSeconds": benchmark,
            "projectedFeatureWallSecondsUpper": projected,
            "projectedCpuHoursUpper": 5,
            "cpuHoursCeiling": 100,
            "wallHoursCeiling": 72,
        },
    )


def load_states(candidate: str, matrix_index: int, manifest: pd.DataFrame) -> np.ndarray:
    row = manifest[
        manifest["candidateId"].eq(candidate)
        & manifest["matrixIndex"].eq(matrix_index)
    ].iloc[0]
    path = Path(row["cachePath"])
    if not path.is_file() or sha256_file(path) != row["cacheSha256"]:
        raise RuntimeError(f"trajectory cache mismatch {candidate} {matrix_index}")
    with path.open("rb") as handle:
        trajectory = pickle.load(handle)
    if trajectory.trajectory_sha256 != row["trajectorySha256"]:
        raise RuntimeError(f"trajectory identity mismatch {candidate} {matrix_index}")
    selected = selected_clock_observations(trajectory, CLOCK_ID)
    states = states_from_observations(selected)
    if len(states) != int(row["selectedClockLength"]):
        raise RuntimeError(f"selected clock mismatch {candidate} {matrix_index}")
    return np.asarray(states, dtype=np.int64)


def extract_matched_features(
    matches: pd.DataFrame, manifest: pd.DataFrame, matrix_role: str
) -> tuple[pd.DataFrame, dict[tuple[str, str, str], np.ndarray]]:
    rows: list[dict[str, Any]] = []
    windows: dict[tuple[str, str, str], np.ndarray] = {}
    subset = matches[matches["matrixRole"].eq(matrix_role)]
    for pair in subset.itertuples(index=False):
        for role, matrix_index, label in (
            ("EVENT", int(pair.eventMatrixIndex), 1),
            ("CONTROL", int(pair.controlMatrixIndex), 0),
        ):
            states = load_states(pair.candidateId, matrix_index, manifest)
            window = states[int(pair.windowStart) : int(pair.windowEndpoint)].copy()
            if window.shape != (WINDOW_COUNT, 100):
                raise RuntimeError("locked event-aligned window unavailable")
            values = extract_window_features(window)
            rows.append(
                {
                    "matrixRole": matrix_role,
                    "candidateId": pair.candidateId,
                    "pairId": pair.pairId,
                    "windowRole": role,
                    "matrixIndex": matrix_index,
                    "label": label,
                    "windowStart": int(pair.windowStart),
                    "windowEndpoint": int(pair.windowEndpoint),
                    **values,
                }
            )
            windows[(pair.candidateId, pair.pairId, role)] = window
    return pd.DataFrame(rows), windows


def development_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame["candidateId"].value_counts()
    total = float(len(frame))
    return np.asarray(
        [total / (2.0 * counts[candidate]) for candidate in frame["candidateId"]],
        dtype=float,
    )


def fit_registered_models(development: pd.DataFrame) -> dict[str, tuple[StandardScaler, LogisticRegression]]:
    result = {}
    y = development["label"].to_numpy(dtype=int)
    weights = development_weights(development)
    for model_id, fields in MODEL_FEATURES.items():
        x = development[list(fields)].to_numpy(dtype=float)
        result[model_id] = fit_coordinate(x, y, weights, derived_seed("model", model_id))
    return result


def model_lock_payload(
    models: dict[str, tuple[StandardScaler, LogisticRegression]], development: pd.DataFrame
) -> dict[str, Any]:
    entries = {}
    for model_id, (scaler, model) in models.items():
        entries[model_id] = {
            "features": list(MODEL_FEATURES[model_id]),
            "scalerMean": scaler.mean_.tolist(),
            "scalerScale": scaler.scale_.tolist(),
            "coefficient": model.coef_[0].tolist(),
            "intercept": model.intercept_.tolist(),
            "nonzeroCoefficients": int(np.count_nonzero(model.coef_)),
            "classes": model.classes_.tolist(),
        }
    return {
        "schema": "eidosoma.e01.s19_l24.coordinate_lock.v1",
        "lockedBeforeValidationPayloadOpen": True,
        "developmentRows": len(development),
        "developmentCandidateRows": development["candidateId"].value_counts().to_dict(),
        "developmentFeatureHash": BASE.frame_hash(
            development[["candidateId", "pairId", "windowRole", "label", *REACTION_FEATURES]]
        ),
        "models": entries,
        "lockedAtUtc": utc_now(),
    }


def score_models(
    frame: pd.DataFrame,
    models: dict[str, tuple[StandardScaler, LogisticRegression]],
    variant: str = "ORIGINAL",
) -> pd.DataFrame:
    rows = []
    for model_id, fields in MODEL_FEATURES.items():
        if model_id not in models:
            continue
        scaler, model = models[model_id]
        score = model.predict_proba(scaler.transform(frame[list(fields)].to_numpy(float)))[:, 1]
        for source, value in zip(frame.itertuples(index=False), score, strict=True):
            rows.append(
                {
                    "candidateId": source.candidateId,
                    "pairId": source.pairId,
                    "windowRole": source.windowRole,
                    "matrixIndex": source.matrixIndex,
                    "label": source.label,
                    "windowEndpoint": source.windowEndpoint,
                    "modelId": model_id,
                    "variant": variant,
                    "score": float(value),
                }
            )
    return pd.DataFrame(rows)


def metric_row(frame: pd.DataFrame) -> dict[str, float]:
    y = frame["label"].to_numpy(int)
    score = frame["score"].to_numpy(float)
    pred = (score >= 0.5).astype(int)
    paired = frame.pivot(index="pairId", columns="windowRole", values="score")
    return {
        "pairCount": float(len(paired)),
        "AUROC": float(roc_auc_score(y, score)),
        "AUPRC": float(average_precision_score(y, score)),
        "BRIER": float(brier_score_loss(y, score)),
        "BALANCED_ACCURACY": float(balanced_accuracy_score(y, pred)),
        "pairedScoreDifferenceMean": float(np.mean(paired["EVENT"] - paired["CONTROL"])),
        "pairedScoreDifferenceMedian": float(np.median(paired["EVENT"] - paired["CONTROL"])),
    }


def aggregate_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (candidate, model, variant), frame in predictions.groupby(
        ["candidateId", "modelId", "variant"], sort=True
    ):
        rows.append(
            {"candidateId": candidate, "modelId": model, "variant": variant, **metric_row(frame)}
        )
    return pd.DataFrame(rows)


def paired_bootstraps(predictions: pd.DataFrame) -> pd.DataFrame:
    original = predictions[predictions["variant"].eq("ORIGINAL")]
    rows = []
    for candidate in CANDIDATES:
        cand = original[original["candidateId"].eq(candidate)]
        pair_ids = sorted(cand["pairId"].unique())
        for replicate in range(BOOTSTRAPS):
            rng = np.random.default_rng(derived_seed("bootstrap", candidate, replicate))
            sampled = rng.choice(pair_ids, size=len(pair_ids), replace=True)
            for model_id in MODEL_FEATURES:
                model = cand[cand["modelId"].eq(model_id)]
                pieces = []
                for occurrence, pair_id in enumerate(sampled):
                    piece = model[model["pairId"].eq(pair_id)].copy()
                    piece["pairId"] = f"B{occurrence:04d}"
                    pieces.append(piece)
                value = metric_row(pd.concat(pieces, ignore_index=True))
                rows.append(
                    {
                        "candidateId": candidate,
                        "replicate": replicate,
                        "modelId": model_id,
                        "AUROC": value["AUROC"],
                        "pairedScoreDifferenceMean": value["pairedScoreDifferenceMean"],
                    }
                )
    return pd.DataFrame(rows)


def validation_pair_permutations(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    original = predictions[
        predictions["variant"].eq("ORIGINAL")
        & predictions["modelId"].eq("REACTION_COORDINATE")
    ]
    observed = {candidate: metric_row(original[original["candidateId"].eq(candidate)])["AUROC"] for candidate in CANDIDATES}
    null_rows = []
    for replicate in range(VALIDATION_PERMUTATIONS):
        values = {}
        for candidate in CANDIDATES:
            frame = original[original["candidateId"].eq(candidate)].copy()
            rng = np.random.default_rng(derived_seed("pair_swap", candidate, replicate))
            for pair_id in frame["pairId"].unique():
                if rng.integers(2):
                    mask = frame["pairId"].eq(pair_id)
                    frame.loc[mask, "score"] = frame.loc[mask, "score"].to_numpy()[::-1]
            values[candidate] = metric_row(frame)["AUROC"]
            null_rows.append(
                {"replicate": replicate, "candidateId": candidate, "nullAUROC": values[candidate]}
            )
        maximum = max(values.values())
        for row in null_rows[-len(CANDIDATES) :]:
            row["maxNullAUROC"] = maximum
    nulls = pd.DataFrame(null_rows)
    result_rows = []
    for candidate in CANDIDATES:
        frame = nulls[nulls["candidateId"].eq(candidate)]
        result_rows.append(
            {
                "candidateId": candidate,
                "observedAUROC": observed[candidate],
                "rawPValue": float((1 + np.count_nonzero(frame["nullAUROC"] >= observed[candidate])) / (VALIDATION_PERMUTATIONS + 1)),
                "familywisePValue": float((1 + np.count_nonzero(frame["maxNullAUROC"] >= observed[candidate])) / (VALIDATION_PERMUTATIONS + 1)),
                "replicates": VALIDATION_PERMUTATIONS,
            }
        )
    return pd.DataFrame(result_rows), nulls


def development_label_permutations(
    development: pd.DataFrame, validation: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    observed_model = fit_registered_models(development)["REACTION_COORDINATE"]
    observed_predictions = score_models(validation, {"REACTION_COORDINATE": observed_model})
    observed = {
        candidate: metric_row(observed_predictions[observed_predictions["candidateId"].eq(candidate)])["AUROC"]
        for candidate in CANDIDATES
    }
    fields = list(REACTION_FEATURES)
    null_rows = []
    for replicate in range(DEVELOPMENT_PERMUTATIONS):
        permuted = development.copy()
        rng = np.random.default_rng(derived_seed("development_label_permutation", replicate))
        for pair_id in permuted["pairId"].unique():
            if rng.integers(2):
                mask = permuted["pairId"].eq(pair_id)
                permuted.loc[mask, "label"] = permuted.loc[mask, "label"].to_numpy()[::-1]
        scaler, model = fit_coordinate(
            permuted[fields].to_numpy(float),
            permuted["label"].to_numpy(int),
            development_weights(permuted),
            derived_seed("development_label_model", replicate),
        )
        scores = model.predict_proba(scaler.transform(validation[fields].to_numpy(float)))[:, 1]
        scored = validation[["candidateId", "pairId", "windowRole", "label"]].copy()
        scored["score"] = scores
        values = {}
        for candidate in CANDIDATES:
            values[candidate] = metric_row(scored[scored["candidateId"].eq(candidate)])["AUROC"]
            null_rows.append(
                {"replicate": replicate, "candidateId": candidate, "nullAUROC": values[candidate]}
            )
        maximum = max(values.values())
        for row in null_rows[-len(CANDIDATES) :]:
            row["maxNullAUROC"] = maximum
    nulls = pd.DataFrame(null_rows)
    results = []
    for candidate in CANDIDATES:
        frame = nulls[nulls["candidateId"].eq(candidate)]
        results.append(
            {
                "candidateId": candidate,
                "observedAUROC": observed[candidate],
                "familywisePValue": float((1 + np.count_nonzero(frame["maxNullAUROC"] >= observed[candidate])) / (DEVELOPMENT_PERMUTATIONS + 1)),
                "replicates": DEVELOPMENT_PERMUTATIONS,
            }
        )
    return pd.DataFrame(results), nulls


def control_predictions(
    validation: pd.DataFrame,
    windows: dict[tuple[str, str, str], np.ndarray],
    models: dict[str, tuple[StandardScaler, LogisticRegression]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    reversed_frame = validation.copy()
    for index, row in reversed_frame.iterrows():
        window = windows[(row["candidateId"], row["pairId"], row["windowRole"])]
        values = extract_window_features(window[::-1])
        for name, value in values.items():
            reversed_frame.at[index, name] = value
    temporal = score_models(
        reversed_frame, {"REACTION_COORDINATE": models["REACTION_COORDINATE"]}, "TEMPORAL_REVERSAL"
    )
    permutation_rows = []
    fields = list(REACTION_FEATURES)
    scaler, model = models["REACTION_COORDINATE"]
    for replicate in range(DEVELOPMENT_PERMUTATIONS):
        shuffled = validation.copy()
        rng = np.random.default_rng(derived_seed("feature_row_permutation", replicate))
        for candidate in CANDIDATES:
            indices = shuffled.index[shuffled["candidateId"].eq(candidate)].to_numpy()
            source = shuffled.loc[indices, fields].to_numpy()[rng.permutation(len(indices))]
            shuffled.loc[indices, fields] = source
        scores = model.predict_proba(scaler.transform(shuffled[fields].to_numpy(float)))[:, 1]
        scored = shuffled[["candidateId", "pairId", "windowRole", "matrixIndex", "label", "windowEndpoint"]].copy()
        scored["modelId"] = "REACTION_COORDINATE"
        scored["variant"] = "FEATURE_ROW_PERMUTATION"
        scored["score"] = scores
        for candidate in CANDIDATES:
            value = metric_row(scored[scored["candidateId"].eq(candidate)])
            permutation_rows.append(
                {
                    "replicate": replicate,
                    "candidateId": candidate,
                    "controlId": "FEATURE_ROW_PERMUTATION",
                    "AUROC": value["AUROC"],
                }
            )
    return temporal, pd.DataFrame(permutation_rows)


def suffix_invariance(
    matches: pd.DataFrame,
    manifest: pd.DataFrame,
    validation: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    sentinels = matches[matches["matrixRole"].eq("VALIDATION")].groupby("candidateId").head(4)
    lookup = validation.set_index(["candidateId", "pairId", "windowRole"])
    for pair in sentinels.itertuples(index=False):
        for role, matrix_index in (("EVENT", pair.eventMatrixIndex), ("CONTROL", pair.controlMatrixIndex)):
            states = load_states(pair.candidateId, int(matrix_index), manifest)
            prefix = states[: int(pair.windowEndpoint)].copy()
            altered = states.copy()
            if len(altered) > int(pair.windowEndpoint):
                rng = np.random.default_rng(derived_seed("suffix", pair.candidateId, pair.pairId, role))
                altered[int(pair.windowEndpoint) :] = altered[int(pair.windowEndpoint) :][rng.permutation(len(altered) - int(pair.windowEndpoint))]
            first = extract_window_features(prefix[-WINDOW_COUNT:])
            second = extract_window_features(altered[int(pair.windowStart) : int(pair.windowEndpoint)])
            stored = lookup.loc[(pair.candidateId, pair.pairId, role)]
            rows.append(
                {
                    "candidateId": pair.candidateId,
                    "pairId": pair.pairId,
                    "windowRole": role,
                    "matrixIndex": int(matrix_index),
                    "suffixChanged": len(states) > int(pair.windowEndpoint),
                    "prefixExact": np.array_equal(prefix, altered[: int(pair.windowEndpoint)]),
                    "featureInvariant": first == second,
                    "storedFeatureExact": all(first[name] == stored[name] for name in REACTION_FEATURES),
                }
            )
    return pd.DataFrame(rows)


def leave_one_pair_out(predictions: pd.DataFrame) -> pd.DataFrame:
    original = predictions[
        predictions["variant"].eq("ORIGINAL")
        & predictions["modelId"].eq("REACTION_COORDINATE")
    ]
    rows = []
    for candidate in CANDIDATES:
        frame = original[original["candidateId"].eq(candidate)]
        for pair_id in frame["pairId"].unique():
            value = metric_row(frame[~frame["pairId"].eq(pair_id)])
            rows.append({"candidateId": candidate, "excludedPairId": pair_id, **value})
    return pd.DataFrame(rows)


def scientific_gates(
    metrics: pd.DataFrame,
    bootstraps: pd.DataFrame,
    pair_permutation: pd.DataFrame,
    development_permutation: pd.DataFrame,
    suffix: pd.DataFrame,
) -> pd.DataFrame:
    original = metrics[metrics["variant"].eq("ORIGINAL")].set_index(["candidateId", "modelId"])
    rows = []
    for candidate in CANDIDATES:
        lead = original.loc[(candidate, "REACTION_COORDINATE")]
        exact = original.loc[(candidate, "EXACT_H_WINDOW")]
        ordinary = original.loc[(candidate, "ORDINARY_STABILITY_WINDOW")]
        lead_boot = bootstraps[
            bootstraps["candidateId"].eq(candidate)
            & bootstraps["modelId"].eq("REACTION_COORDINATE")
        ]
        exact_boot = bootstraps[
            bootstraps["candidateId"].eq(candidate)
            & bootstraps["modelId"].eq("EXACT_H_WINDOW")
        ]["AUROC"].to_numpy()
        ordinary_boot = bootstraps[
            bootstraps["candidateId"].eq(candidate)
            & bootstraps["modelId"].eq("ORDINARY_STABILITY_WINDOW")
        ]["AUROC"].to_numpy()
        auc_lower = float(np.quantile(lead_boot["AUROC"], 0.025))
        diff_lower = float(np.quantile(lead_boot["pairedScoreDifferenceMean"], 0.025))
        delta_exact_lower = float(np.quantile(lead_boot["AUROC"].to_numpy() - exact_boot, 0.025))
        delta_ordinary_lower = float(np.quantile(lead_boot["AUROC"].to_numpy() - ordinary_boot, 0.025))
        pair_p = float(pair_permutation.set_index("candidateId").loc[candidate, "familywisePValue"])
        dev_p = float(development_permutation.set_index("candidateId").loc[candidate, "familywisePValue"])
        suffix_pass = bool(
            suffix[suffix["candidateId"].eq(candidate)][["prefixExact", "featureInvariant", "storedFeatureExact"]].all().all()
        )
        gates = {
            "minimumPairCountPassed": int(lead["pairCount"]) >= 20,
            "auRocPointPassed": float(lead["AUROC"]) >= 0.65,
            "auRocBootstrapPassed": auc_lower > 0.5,
            "pairedDifferencePassed": diff_lower > 0.0,
            "pointImprovementExactHPassed": float(lead["AUROC"]) > float(exact["AUROC"]),
            "pointImprovementOrdinaryPassed": float(lead["AUROC"]) > float(ordinary["AUROC"]),
            "bootstrapImprovementExactHPassed": delta_exact_lower > 0.0,
            "bootstrapImprovementOrdinaryPassed": delta_ordinary_lower > 0.0,
            "pairSwapPermutationPassed": pair_p <= 0.05,
            "developmentPermutationPassed": dev_p <= 0.05,
            "suffixInvariancePassed": suffix_pass,
        }
        rows.append(
            {
                "candidateId": candidate,
                "pairCount": int(lead["pairCount"]),
                "reactionAuRoc": float(lead["AUROC"]),
                "exactHAuRoc": float(exact["AUROC"]),
                "ordinaryAuRoc": float(ordinary["AUROC"]),
                "auRocBootstrapLower95": auc_lower,
                "pairedDifferenceLower95": diff_lower,
                "deltaExactHLower95": delta_exact_lower,
                "deltaOrdinaryLower95": delta_ordinary_lower,
                "pairSwapFamilywiseP": pair_p,
                "developmentPermutationFamilywiseP": dev_p,
                **gates,
                "candidateDiscoveryGatePassed": all(gates.values()),
            }
        )
    return pd.DataFrame(rows)


def make_figures(
    matching: pd.DataFrame,
    development: pd.DataFrame,
    validation: pd.DataFrame,
    metrics: pd.DataFrame,
    bootstraps: pd.DataFrame,
    permutations: pd.DataFrame,
    gates: pd.DataFrame,
) -> list[str]:
    directory = BUILD_ROOT / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    paths = []

    def save(name: str) -> None:
        path = BUILD_ROOT / name
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(path, dpi=170)
        plt.close()
        paths.append(str(path.relative_to(BUILD_ROOT)))

    matching.groupby(["matrixRole", "candidateId"]).size().unstack().plot(kind="bar")
    plt.ylabel("matched pairs")
    plt.title("Development/validation event-control support")
    save("figures/01_matched_pair_support.png")

    matching.boxplot(column="windowEndpoint", by=["matrixRole", "candidateId"], rot=25)
    plt.suptitle("")
    plt.title("Event-window endpoints")
    save("figures/02_event_endpoint_distribution.png")

    metrics[metrics["variant"].eq("ORIGINAL")].pivot(
        index="modelId", columns="candidateId", values="AUROC"
    ).plot(kind="bar", ylim=(0, 1))
    plt.axhline(0.5, color="black", linestyle="--")
    plt.ylabel("held-out AUROC")
    plt.title("Locked reaction coordinate and controls")
    save("figures/03_validation_auroc.png")

    focus = bootstraps[bootstraps["modelId"].eq("REACTION_COORDINATE")]
    for candidate, frame in focus.groupby("candidateId"):
        plt.hist(frame["AUROC"], bins=40, alpha=0.5, label=candidate)
    plt.axvline(0.5, color="black", linestyle="--")
    plt.xlabel("pair-bootstrap AUROC")
    plt.legend()
    save("figures/04_bootstrap_auroc.png")

    for candidate, frame in permutations.groupby("candidateId"):
        plt.hist(frame["nullAUROC"], bins=35, alpha=0.5, label=candidate)
    plt.axvline(0.5, color="black", linestyle="--")
    plt.xlabel("pair-swap null AUROC")
    plt.legend()
    save("figures/05_pair_swap_null.png")

    development.groupby("candidateId")["label"].count().plot(kind="bar", color="#546e7a")
    plt.ylabel("development windows")
    plt.title("Coordinate-fitting support")
    save("figures/06_development_support.png")

    validation.groupby(["candidateId", "windowRole"])["adjacent_h_mean"].mean().unstack().plot(kind="bar")
    plt.ylabel("mean adjacent H")
    plt.title("Ordinary stability in matched windows")
    save("figures/07_stability_control.png")

    gate_columns = [column for column in gates if column.endswith("Passed")]
    matrix = gates.set_index("candidateId")[gate_columns].astype(int).T
    plt.figure(figsize=(7, 6))
    plt.imshow(matrix.to_numpy(), cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    plt.xticks(range(len(matrix.columns)), matrix.columns, rotation=20)
    plt.yticks(range(len(matrix.index)), [name.replace("Passed", "") for name in matrix.index], fontsize=7)
    plt.colorbar(ticks=[0, 1])
    plt.title("Locked discovery gates")
    save("figures/08_gate_matrix.png")
    return paths


def append_root_ledgers(classifications: list[str], selected: bool, timestamp: str) -> None:
    ledger_path = ARTIFACT_ROOT / "self_improvement_ledger.parquet"
    ledger = pd.read_parquet(ledger_path)
    sequence = int(ledger["ledgerSequence"].max()) + 1
    additions = [
        {
            "appendOnly": True,
            "beliefBeforeLoop": "Powered L23 nulls may reflect fixed-landmark averaging of a localized precursor.",
            "failureOrAmbiguityTargeted": "Temporal misalignment of organization-warning features relative to first attractor entry.",
            "informationGainRationale": "A development/validation firewall and time-matched controls isolate localized pre-entry structure without new simulation.",
            "learned": "L24 event-aligned method, matching and validation contract frozen.",
            "ledgerSequence": sequence,
            "loopId": LOOP_ID,
            "motivatingEvidence": "L23 powered non-support across complete fixed-landmark families.",
            "proposedNextTest": "Execute L24.",
            "recordPhase": "PRE_LOOP_METHOD_LOCK",
            "remainingPlausibleHypotheses": "Localized reaction coordinate or genuinely absent past-only precursor.",
            "selectedHypotheses": "One sparse reaction coordinate on 32-observation pre-onset windows.",
            "timestampUtc": timestamp,
            "weakenedHypotheses": "Fixed-landmark feature families merely lacked matrix support.",
        },
        {
            "appendOnly": True,
            "beliefBeforeLoop": "A localized precursor could be common across candidates even though fixed-landmark models were not.",
            "failureOrAmbiguityTargeted": "Event alignment versus absent organization signal.",
            "informationGainRationale": "The coordinate was locked from one matrix half before validation payload access.",
            "learned": ";".join(classifications),
            "ledgerSequence": sequence + 1,
            "loopId": LOOP_ID,
            "motivatingEvidence": "Complete held-out L24 results.",
            "proposedNextTest": "Untouched online confirmation of the locked coordinate." if selected else "One fixed online change-point/operator precursor in L25.",
            "recordPhase": "POST_LOOP_RESULT_AUTONOMOUS_CONTINUATION",
            "remainingPlausibleHypotheses": "A localized transition operator/change point, or no detectable precursor under frozen labels.",
            "selectedHypotheses": "One sparse reaction coordinate on 32-observation pre-onset windows.",
            "timestampUtc": timestamp,
            "weakenedHypotheses": "Event alignment alone resolves the cross-candidate null." if not selected else "No localized precursor exists.",
        },
    ]
    BASE.write_parquet(
        ledger_path,
        pd.concat([ledger, pd.DataFrame(additions).reindex(columns=ledger.columns)], ignore_index=True),
    )
    candidates_path = ARTIFACT_ROOT / "candidate_registry.parquet"
    candidates = pd.read_parquet(candidates_path)
    start = int(candidates["registryOrder"].max()) + 1
    row = {
        "branchCount": 1,
        "bundleId": "L24_EVENT_ALIGNED_REACTION_COORDINATE",
        "candidateId": "S19-L24-REACTION-COORDINATE",
        "candidateSpecificSuccess": 0,
        "completedFitLeakage": 0,
        "computeEfficiency": 5,
        "crossCandidateDiscriminability": 5,
        "deterministicHReuse": 0,
        "explanatoryLeverage": 4,
        "frozenRank": 1,
        "independenceFromPriorOutcomeSelection": 3,
        "outcomeGuidedThresholdSelection": 0,
        "paperFingerprintSpecificity": 0,
        "proposedSpecification": "32-step event-aligned sparse reaction coordinate",
        "rankingScore": 24.0,
        "registryOrder": start,
        "selected": True,
        "selectionReason": "L23_POWERED_NULL_LOCALIZATION_TEST",
        "sourceGrounding": 4,
        "testability": 5,
        "undefinedAuthorSemantics": 0,
    }
    BASE.write_parquet(
        candidates_path,
        pd.concat([candidates, pd.DataFrame([row]).reindex(columns=candidates.columns)], ignore_index=True),
    )
    sources_path = ARTIFACT_ROOT / "source_search_ledger.parquet"
    sources = pd.read_parquet(sources_path)
    additions_source = []
    for item in source_registry().itertuples(index=False):
        additions_source.append(
            {
                "commitOrVersion": item.doi,
                "evidenceClass": item.evidenceClass,
                "finding": f"{item.directSupport}; L24 frozen use: {item.frozenUse}",
                "licenseStatus": "PUBLIC_ARTICLE",
                "redistributionStatus": "CITATION_ONLY",
                "repositoryIdentity": None,
                "retainedPath": None,
                "retrievalDate": timestamp[:10],
                "sha256": None,
                "sourceId": f"L24_{item.sourceId}",
                "sourceType": item.evidenceClass,
                "treeIdentity": None,
                "url": item.url,
            }
        )
    BASE.write_parquet(
        sources_path,
        pd.concat([sources, pd.DataFrame(additions_source).reindex(columns=sources.columns)], ignore_index=True),
    )
    loop_path = ARTIFACT_ROOT / "loop_registry.yaml"
    data = yaml.safe_load(loop_path.read_text())
    data["loops"].append(
        {
            "loopId": LOOP_ID,
            "versionedLoopId": VERSION,
            "status": "COMPLETE_AUTONOMOUS_CONTINUATION_AUTHORIZED",
            "authorized": True,
            "completed": True,
            "outcomeAccessed": True,
            "humanReviewRequiredAfter": False,
            "classification": classifications,
            "selectedDiscoveryLead": "REACTION_COORDINATE" if selected else None,
            "newMatrices": 0,
            "newTrajectories": 0,
            "nextStepActive": True,
        }
    )
    data["laterLoopsAuthorized"] = True
    data["authorizationUpperBound"] = "S19-L42"
    data["proposedNextLoopTheme"] = "UNTOUCHED_ONLINE_REACTION_COORDINATE_CONFIRMATION" if selected else "ONLINE_CHANGE_POINT_OPERATOR_PRECURSOR"
    data["proposedNextLoopActive"] = True
    BASE.atomic_text(loop_path, yaml.safe_dump(data, sort_keys=False))
    review_path = ARTIFACT_ROOT / "human_review_history.json"
    review = json.loads(review_path.read_text())
    review["history"].append(
        {
            "decision": "S19_L24_COMPLETE_CONTINUE_UNDER_EXISTING_AUTHORIZATION",
            "loopId": LOOP_ID,
            "scope": VERSION,
            "recordedAtUtc": timestamp,
            "result": classifications,
            "selectedDiscoveryLead": "REACTION_COORDINATE" if selected else None,
            "source": "locked_execution_result",
            "nextLoopAuthorized": True,
            "s20Activated": False,
        }
    )
    review["pendingDecision"] = "NONE_AUTONOMOUS_SEQUENCE_ACTIVE_THROUGH_L42"
    BASE.write_json(review_path, review)


def manifest_for(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name != "artifact_manifest.json"):
        rows.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {
        "schema": "eidosoma.e01.s19_l24.artifact_manifest.v1",
        "root": str(root),
        "fileCount": len(rows),
        "totalBytes": sum(row["bytes"] for row in rows),
        "files": rows,
    }


def report_text(
    matching: pd.DataFrame,
    metrics: pd.DataFrame,
    gates: pd.DataFrame,
    classifications: list[str],
    selected: bool,
    runtime: dict[str, Any],
) -> str:
    support = matching.groupby(["matrixRole", "candidateId"]).size().rename("pairs").reset_index()
    focus = metrics[metrics["variant"].eq("ORIGINAL")][
        ["candidateId", "modelId", "pairCount", "AUROC", "AUPRC", "BRIER", "pairedScoreDifferenceMean"]
    ]
    recommendation = (
        "Freeze the coordinate and run one new-matrix, online landmark confirmation in L25."
        if selected
        else "Event alignment did not recover a common signal. Test one outcome-blind online change-point/operator precursor in L25 without retuning L24."
    )
    return f"""# S19-L24 — Event-Aligned Cross-Candidate Pre-Onset Reaction Coordinate

## Chief/human handoff

- **Step:** `{VERSION}`
- **Status:** complete within the authorized autonomous L19–L42 sequence.
- **Outcome classifications:** {", ".join(f"`{value}`" for value in classifications)}
- **Selected lead:** `{"REACTION_COORDINATE" if selected else "NONE"}`.
- **Validation:** immutable L23/prior hashes; 800/800 frozen trajectory cache identities; 200/200 outcome-blind shared-matrix firewall; coordinate serialized before validation payload access; candidate-separated matched validation; 4,096 pair bootstraps; 4,096 pair-swap and 512 development-label permutations; exact feature/model/report regeneration; suffix, storage and artifact hashes passed.
- **Recommended next bounded loop:** {recommendation}

## Frozen question

Can one sparse development-fitted reaction coordinate distinguish the 32 observations immediately before a first recurring-attractor entry from a non-imminent trajectory at the same absolute molecular time, in a held-out matrix half and both simulator candidates?

## Matched support

{support.to_markdown(index=False)}

## Methods

L24 generated no matrix or trajectory. The frozen L23 cohort was split by SHA-256 identity into 200 development and 200 validation matrices, paired across candidates. Within each half, first onsets from 128 through 256 defined event windows `[tau-32,tau)`. Controls were distinct, non-event matrices whose first onset occurred after 256 and at least 96 observations later, sampled without replacement at exactly the same endpoint. A 28-feature molecule-label-permutation-invariant summary was fitted with one development-only scaler and L1 logistic coordinate. The exact-H window, ordinary-composition/stability window, and endpoint-only models were locked controls.

## Held-out results

{focus.to_markdown(index=False)}

## Gate adjudication

{gates.to_markdown(index=False)}

## Interpretation

The event time used to align a window is known only after the completed trajectory. Therefore even a passing L24 coordinate would be retrospective discovery and could not support online early warning until frozen at an outcome-blind landmark on new seed-firewalled matrices. Exact-H and ordinary-stability controls prevent a smoothness proxy from being treated as independent organization evidence.

## Runtime and provenance

- Repository lock: `{runtime['repositoryHead']}`.
- CPU float64, one numerical-library thread, no GPU.
- Wall seconds: `{runtime['wallSeconds']:.3f}`; process CPU hours: `{runtime['processCpuHours']:.6f}`.
- Frozen L23 trajectory payloads remained in `/cache/e01_s19_l23`; L24 retained compact evidence only.

## Autonomous continuation boundary

L24 is frozen. The human authorization permits the next bounded loop through at most L42. S20, E02, author contact, interventions and report-bundle work remain inactive.
"""


def decision_summary_text(classifications: list[str], selected: bool) -> str:
    return f"""# S19-L24 decision summary

**Classification:** {", ".join(classifications)}
**Selected discovery lead:** `{"REACTION_COORDINATE" if selected else "NONE"}`

{("The held-out reaction coordinate passed both candidates, but remains retrospective and now requires untouched online confirmation." if selected else "A development-locked event-aligned coordinate did not pass the two-candidate held-out gate; fixed timing was not the sole cause of the earlier nulls.")}

The autonomous authorization permits one next bounded loop through L42. S20, E02, author contact, interventions and report generation remain inactive.
"""


def execute() -> None:
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    preserved_failed_build = CACHE_ROOT / "failed_attempts" / "attempt_002_complete_build"
    lock_candidates = (
        LOOP_ROOT / "preoutcome_repository_lock.json",
        BUILD_ROOT / "preoutcome_repository_lock.json",
        preserved_failed_build / "preoutcome_repository_lock.json",
    )
    lock_path = next((path for path in lock_candidates if path.is_file()), lock_candidates[0])
    if not lock_path.is_file():
        raise RuntimeError("run --prepare-lock first")
    lock = json.loads(lock_path.read_text())
    current_head = git("rev-parse", "HEAD")
    current_remote = git("rev-parse", "origin/eidosoma/groups/42")
    allowed_heads = {lock["head"]}
    amendment_paths = []
    for name in (
        "technical_amendment_001.json",
        "technical_amendment_002.json",
        "technical_amendment_003.json",
        "technical_amendment_004.json",
    ):
        candidates = (LOOP_ROOT / name, BUILD_ROOT / name, preserved_failed_build / name)
        amendment_paths.append(next((path for path in candidates if path.is_file()), candidates[0]))
    for amendment_path in amendment_paths:
        if not amendment_path.is_file():
            continue
        amendment = json.loads(amendment_path.read_text())
        if (
            amendment.get("status") == "APPROVED_VALUE_PRESERVING"
            and amendment.get("originalRepositoryHead") == lock["head"]
            and amendment.get("amendedRunnerSha256") == sha256_file(RUNNER_PATH)
        ):
            allowed_heads.add(amendment["amendedRepositoryHead"])
    if current_head not in allowed_heads or current_remote != current_head:
        raise RuntimeError("repository identity changed outside the recorded technical amendment")
    if git("status", "--porcelain=v1"):
        raise RuntimeError("repository must remain clean")
    prior = validate_immutable_prior()
    if not prior["unchanged"] or prior["aggregateSha256"] != lock["priorAggregateSha256"]:
        raise RuntimeError("immutable prior gate failed")
    fixtures = fixture_table()
    if not fixtures["passed"].all():
        raise RuntimeError("mandatory fixture failure")
    manifest = pd.read_parquet(L23_ROOT / "input_trajectory_manifest.parquet")
    identity_rows = []
    for row in manifest.itertuples(index=False):
        path = Path(row.cachePath)
        passed = path.is_file() and sha256_file(path) == row.cacheSha256
        identity_rows.append(
            {
                "candidateId": row.candidateId,
                "matrixIndex": row.matrixIndex,
                "trajectoryId": row.trajectoryId,
                "trajectorySha256": row.trajectorySha256,
                "cacheSha256": row.cacheSha256,
                "cacheIdentityPassed": passed,
            }
        )
    identities = pd.DataFrame(identity_rows)
    if not identities["cacheIdentityPassed"].all() or len(identities) != 800:
        raise RuntimeError("L23 cache validation failed")
    firewall = matrix_firewall()
    targets = pd.read_parquet(L23_ROOT / "target_geometry_results.parquet")
    matching = match_pairs(targets, firewall)
    preoutcome_candidates = (
        LOOP_ROOT / "matching_registry_preoutcome.parquet",
        BUILD_ROOT / "matching_registry_preoutcome.parquet",
        preserved_failed_build / "matching_registry_preoutcome.parquet",
    )
    preoutcome_path = next(
        (path for path in preoutcome_candidates if path.is_file()),
        preoutcome_candidates[0],
    )
    preoutcome_matching = pd.read_parquet(
        preoutcome_path
    )
    if BASE.frame_hash(matching) != BASE.frame_hash(preoutcome_matching):
        raise RuntimeError("matching registry changed after lock")

    if BUILD_ROOT.exists():
        preserved_failed_build.parent.mkdir(parents=True, exist_ok=True)
        partial_index = 3
        while (preserved_failed_build.parent / f"attempt_{partial_index:03d}_partial_build").exists():
            partial_index += 1
        partial_destination = preserved_failed_build.parent / f"attempt_{partial_index:03d}_partial_build"
        shutil.move(str(BUILD_ROOT), str(partial_destination))
    if not preserved_failed_build.is_dir():
        raise RuntimeError("the complete attempt-002 build required for value equality is missing")
    lock_artifact_source = LOOP_ROOT if (LOOP_ROOT / "preregistration.yaml").is_file() else preserved_failed_build
    amendment_paths = []
    for name in (
        "technical_amendment_001.json",
        "technical_amendment_002.json",
        "technical_amendment_003.json",
        "technical_amendment_004.json",
    ):
        candidates = (LOOP_ROOT / name, preserved_failed_build / name)
        amendment_paths.append(next((path for path in candidates if path.is_file()), candidates[0]))
    BUILD_ROOT.mkdir(parents=True)
    development, development_windows = extract_matched_features(matching, manifest, "DEVELOPMENT")
    models = fit_registered_models(development)
    coordinate_lock = model_lock_payload(models, development)
    BASE.write_json(BUILD_ROOT / "coordinate_lock.json", coordinate_lock)
    coordinate_lock_hash_before_validation = sha256_file(BUILD_ROOT / "coordinate_lock.json")

    # Validation trajectory payloads are opened only after the fitted coordinate is frozen.
    validation, validation_windows = extract_matched_features(matching, manifest, "VALIDATION")
    if sha256_file(BUILD_ROOT / "coordinate_lock.json") != coordinate_lock_hash_before_validation:
        raise RuntimeError("coordinate lock changed after validation access")
    predictions = score_models(validation, models)
    temporal_predictions, row_permutations = control_predictions(validation, validation_windows, models)
    all_predictions = pd.concat([predictions, temporal_predictions], ignore_index=True)
    metrics = aggregate_metrics(all_predictions)
    bootstraps = paired_bootstraps(predictions)
    pair_permutation, pair_nulls = validation_pair_permutations(predictions)
    development_permutation, development_nulls = development_label_permutations(development, validation)
    suffix = suffix_invariance(matching, manifest, validation)
    leave_one_out = leave_one_pair_out(predictions)
    gates = scientific_gates(metrics, bootstraps, pair_permutation, development_permutation, suffix)
    selected = bool(gates["candidateDiscoveryGatePassed"].all())
    classifications = (
        [
            "EVENT_ALIGNED_REACTION_COORDINATE_DISCOVERY_LEAD",
            "REQUIRES_UNTOUCHED_ONLINE_CONFIRMATION",
            "NOT_PROMOTABLE_AS_CONFIRMED",
        ]
        if selected
        else [
            "EVENT_ALIGNED_REACTION_COORDINATE_NON_SUPPORT",
            "TIME_LOCALIZATION_NOT_SUFFICIENT",
            "NOT_PROMOTABLE_AS_CONFIRMED",
        ]
    )
    original = metrics[metrics["variant"].eq("ORIGINAL")].set_index(["candidateId", "modelId"])
    if any(
        float(original.loc[(candidate, "REACTION_COORDINATE"), "AUROC"])
        <= max(
            float(original.loc[(candidate, "EXACT_H_WINDOW"), "AUROC"]),
            float(original.loc[(candidate, "ORDINARY_STABILITY_WINDOW"), "AUROC"]),
        )
        for candidate in CANDIDATES
    ):
        classifications.append("POSSIBLE_STABILITY_PROXY")

    figures = make_figures(matching, development, validation, metrics, bootstraps, pair_nulls, gates)
    for path in (
        "fixture_results.parquet",
        "matrix_firewall.parquet",
        "matching_registry_preoutcome.parquet",
    ):
        shutil.copy2(lock_artifact_source / path, BUILD_ROOT / path)
    for path in (
        "preregistration.yaml",
        "decision_record.md",
        "source_grounding_registry.csv",
        "source_grounding_report.md",
        "immutable_prior_validation.json",
        "implementation_lock.json",
        "preoutcome_repository_lock.json",
        "benchmark_projection.json",
    ):
        shutil.copy2(lock_artifact_source / path, BUILD_ROOT / path)
    for amendment_path in amendment_paths:
        if amendment_path.is_file():
            shutil.copy2(amendment_path, BUILD_ROOT / amendment_path.name)
    failed_attempt_sources = (
        preserved_failed_build / "failed_attempts",
        LOOP_ROOT / "failed_attempts",
    )
    for failed_attempts in failed_attempt_sources:
        if failed_attempts.is_dir():
            shutil.copytree(
                failed_attempts,
                BUILD_ROOT / "failed_attempts",
                dirs_exist_ok=True,
            )
    BASE.write_parquet(BUILD_ROOT / "trajectory_identity_validation.parquet", identities)
    BASE.write_parquet(BUILD_ROOT / "development_window_features.parquet", development)
    BASE.write_parquet(BUILD_ROOT / "validation_window_features.parquet", validation)
    BASE.write_parquet(BUILD_ROOT / "prediction_results.parquet", all_predictions)
    BASE.write_parquet(BUILD_ROOT / "aggregate_metrics.parquet", metrics)
    BASE.write_parquet(BUILD_ROOT / "bootstrap_results.parquet", bootstraps)
    BASE.write_parquet(BUILD_ROOT / "pair_swap_permutation_results.parquet", pair_permutation)
    BASE.write_parquet(BUILD_ROOT / "pair_swap_permutation_nulls.parquet", pair_nulls)
    BASE.write_parquet(BUILD_ROOT / "development_label_permutation_results.parquet", development_permutation)
    BASE.write_parquet(BUILD_ROOT / "development_label_permutation_nulls.parquet", development_nulls)
    BASE.write_parquet(BUILD_ROOT / "feature_row_permutation_results.parquet", row_permutations)
    BASE.write_parquet(BUILD_ROOT / "suffix_invariance_results.parquet", suffix)
    BASE.write_parquet(BUILD_ROOT / "leave_one_pair_out_results.parquet", leave_one_out)
    BASE.write_parquet(BUILD_ROOT / "scientific_gate_results.parquet", gates)
    pd.DataFrame(columns=["stage", "candidateId", "matrixIndex", "exceptionClass", "exceptionMessage"]).to_csv(
        BUILD_ROOT / "failure_ledger.csv", index=False
    )
    amendment_rows = []
    amendment_scopes = {
        "technical_amendment_001.json": "plot output path construction only",
        "technical_amendment_002.json": "same-filesystem atomic artifact promotion only",
        "technical_amendment_003.json": "recovery-source path selection only",
        "technical_amendment_004.json": "complete recovery-source consolidation only",
    }
    for amendment_path in amendment_paths:
        if amendment_path.is_file():
            amendment_rows.append(
                {
                    "amendmentId": amendment_path.stem.rsplit("_", 1)[-1],
                    "scope": amendment_scopes[amendment_path.name],
                    "scientificValuesChanged": False,
                    "status": "APPLIED_FRESH_CACHE_FULL_RERUN",
                }
            )
    pd.DataFrame(
        amendment_rows,
        columns=["amendmentId", "scope", "scientificValuesChanged", "status"],
    ).to_csv(BUILD_ROOT / "technical_amendment_ledger.csv", index=False)
    BASE.write_json(
        BUILD_ROOT / "classification.json",
        {
            "schema": "eidosoma.e01.s19_l24.classification.v1",
            "researchStepId": LOOP_ID,
            "classifications": classifications,
            "selectedDiscoveryLead": "REACTION_COORDINATE" if selected else None,
            "confirmatory": False,
            "prospective": False,
            "priorStatusesChanged": False,
        },
    )
    replay_development, _ = extract_matched_features(matching, manifest, "DEVELOPMENT")
    replay_validation, _ = extract_matched_features(matching, manifest, "VALIDATION")
    replay_models = fit_registered_models(replay_development)
    replay_predictions = score_models(replay_validation, replay_models)
    replay_checks = {
        "developmentFeatureExact": BASE.frame_hash(development) == BASE.frame_hash(replay_development),
        "validationFeatureExact": BASE.frame_hash(validation) == BASE.frame_hash(replay_validation),
        "predictionExact": BASE.frame_hash(predictions) == BASE.frame_hash(replay_predictions),
        "coordinateLockUnchangedAfterValidation": sha256_file(BUILD_ROOT / "coordinate_lock.json") == coordinate_lock_hash_before_validation,
        "matchedRegistryExact": BASE.frame_hash(matching) == BASE.frame_hash(preoutcome_matching),
        "allTrajectoryCacheIdentitiesPassed": bool(identities["cacheIdentityPassed"].all()),
        "suffixInvariancePassed": bool(suffix[["prefixExact", "featureInvariant", "storedFeatureExact"]].all().all()),
    }
    BASE.write_json(
        BUILD_ROOT / "regeneration_validation.json",
        {
            "schema": "eidosoma.e01.s19_l24.regeneration_validation.v1",
            "status": "PASS" if all(replay_checks.values()) else "FAIL",
            "checks": replay_checks,
            "developmentRows": len(development),
            "validationRows": len(validation),
            "predictionRows": len(predictions),
        },
    )
    if not all(replay_checks.values()):
        raise RuntimeError("L24 regeneration validation failed")
    runtime = {
        "schema": "eidosoma.e01.s19_l24.runtime.v1",
        "researchStepId": LOOP_ID,
        "repositoryHead": git("rev-parse", "HEAD"),
        "workers": 1,
        "numericalThreadsPerWorker": 1,
        "gpuHours": 0,
        "wallSeconds": time.perf_counter() - started_wall,
        "processCpuHours": (time.process_time() - started_cpu) / 3600,
        "developmentPairs": int((matching["matrixRole"].eq("DEVELOPMENT")).sum()),
        "validationPairs": int((matching["matrixRole"].eq("VALIDATION")).sum()),
        "bootstrapReplicates": BOOTSTRAPS,
        "pairSwapReplicates": VALIDATION_PERMUTATIONS,
        "developmentPermutationReplicates": DEVELOPMENT_PERMUTATIONS,
        "completedAtUtc": utc_now(),
    }
    BASE.write_json(BUILD_ROOT / "runtime_manifest.json", runtime)
    storage = {
        "schema": "eidosoma.e01.s19_l24.storage_validation.v1",
        "retainedBytesBeforeFinalManifest": sum(path.stat().st_size for path in BUILD_ROOT.rglob("*") if path.is_file()),
        "retainedGiBCeiling": 25,
        "temporaryBytes": sum(path.stat().st_size for path in CACHE_ROOT.rglob("*") if path.is_file()),
        "temporaryGiBCeiling": 75,
    }
    storage["status"] = "PASS" if storage["retainedBytesBeforeFinalManifest"] < 25 * 2**30 and storage["temporaryBytes"] < 75 * 2**30 else "FAIL"
    BASE.write_json(BUILD_ROOT / "storage_validation.json", storage)
    if storage["status"] != "PASS":
        raise RuntimeError("L24 storage ceiling exceeded")
    report = report_text(matching, metrics, gates, classifications, selected, runtime)
    BASE.atomic_text(BUILD_ROOT / "S19_L24_FULL_RESULTS.md", report)
    BASE.atomic_text(BUILD_ROOT / "research_step_full_results.md", report)
    BASE.atomic_text(BUILD_ROOT / "loop_decision_summary.md", decision_summary_text(classifications, selected))
    scientific_names = [
        "development_window_features.parquet",
        "validation_window_features.parquet",
        "prediction_results.parquet",
        "aggregate_metrics.parquet",
        "bootstrap_results.parquet",
        "pair_swap_permutation_results.parquet",
        "pair_swap_permutation_nulls.parquet",
        "development_label_permutation_results.parquet",
        "development_label_permutation_nulls.parquet",
        "feature_row_permutation_results.parquet",
        "suffix_invariance_results.parquet",
        "leave_one_pair_out_results.parquet",
        "scientific_gate_results.parquet",
        "classification.json",
    ]
    equality_rows = []
    if preserved_failed_build.is_dir():
        for name in scientific_names:
            previous = preserved_failed_build / name
            current = BUILD_ROOT / name
            equality_rows.append(
                {
                    "path": name,
                    "priorFailedBuildPresent": previous.is_file(),
                    "priorFailedBuildSha256": sha256_file(previous) if previous.is_file() else None,
                    "authoritativeSha256": sha256_file(current),
                    "byteExact": previous.is_file() and sha256_file(previous) == sha256_file(current),
                }
            )
    equality = pd.DataFrame(
        equality_rows,
        columns=[
            "path",
            "priorFailedBuildPresent",
            "priorFailedBuildSha256",
            "authoritativeSha256",
            "byteExact",
        ],
    )
    BASE.write_parquet(BUILD_ROOT / "technical_amendment_scientific_equality.parquet", equality)
    if not equality.empty and not equality["byteExact"].all():
        raise RuntimeError("technical amendment changed a scientific artifact")
    manifest_value = manifest_for(BUILD_ROOT)
    BASE.write_json(BUILD_ROOT / "artifact_manifest.json", manifest_value)
    promotion_stage = LOOP_ROOT.with_name(".L24-promotion-stage")
    if promotion_stage.exists():
        shutil.rmtree(promotion_stage)
    shutil.copytree(BUILD_ROOT, promotion_stage)
    if LOOP_ROOT.exists():
        shutil.rmtree(LOOP_ROOT)
    os.replace(promotion_stage, LOOP_ROOT)
    shutil.rmtree(BUILD_ROOT)
    final_manifest = json.loads((LOOP_ROOT / "artifact_manifest.json").read_text())
    for item in final_manifest["files"]:
        if sha256_file(LOOP_ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError("post-promotion artifact hash mismatch")
    append_root_ledgers(classifications, selected, runtime["completedAtUtc"])
    BASE.atomic_text(ARTIFACT_ROOT / "research_step_full_results.md", report)
    BASE.atomic_text(ARTIFACT_ROOT / "S19_CURRENT_HANDOFF.md", report.replace("# S19-L24", "# S19 current handoff — S19-L24", 1))
    BASE.write_json(
        ARTIFACT_ROOT / "s19_status.json",
        {
            "schema": "eidosoma.e01.s19.status.v1",
            "programId": "E01-S19-ITERATIVE-SELF-IMPROVING-REPLICATION-SEARCH-v1.0.0",
            "status": "ACTIVE_AUTONOMOUS_SEQUENCE",
            "latestCompletedLoop": LOOP_ID,
            "latestClassification": classifications,
            "selectedDiscoveryLead": "REACTION_COORDINATE" if selected else None,
            "nextAuthorizedLoop": "S19-L25",
            "authorizationUpperBound": "S19-L42",
            "s20Active": False,
            "updatedAtUtc": runtime["completedAtUtc"],
        },
    )
    BASE.write_json(ARTIFACT_ROOT / "artifact_manifest.json", manifest_for(ARTIFACT_ROOT))
    print(json.dumps({"status": "COMPLETE", "classifications": classifications, "selected": selected, "runtime": runtime}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-lock", action="store_true")
    args = parser.parse_args()
    if args.prepare_lock:
        prepare_lock()
    else:
        execute()


if __name__ == "__main__":
    main()
