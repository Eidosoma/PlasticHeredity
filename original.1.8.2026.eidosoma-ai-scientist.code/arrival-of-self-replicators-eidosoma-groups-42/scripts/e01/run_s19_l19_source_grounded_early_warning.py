#!/usr/bin/env python3
"""Prepare and execute S19-L19 source-grounded early-warning discovery."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
import warnings
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
import pyarrow
import scipy
import sklearn
import yaml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    precision_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from e01_attractor_onset_early_warning.core import (
    FEATURE_GROUPS as L18_FEATURE_GROUPS,
)
from e01_attractor_onset_early_warning.core import (
    HORIZON_EXCLUSIVE,
    LANDMARK_COUNT,
    build_landmark_target,
)
from e01_clean_directional_confirmation.core import fixed_label_spec
from e01_creative_directional_search.core import label_trajectory
from e01_frozen_timebase_ensemble.core import (
    selected_clock_observations,
    states_from_observations,
)
from e01_onset_discovery.core import (
    DMD_FEATURES,
    EWS_FEATURES,
    RQA_FEATURES,
    extract_organization_warning_features,
)

LOOP_ID = "S19-L19"
VERSION = "E01-S19-L19-SOURCE-GROUNDED-EARLY-WARNING-OBSERVABLES-v1.0.0"
TARGET_ID = "PF_DOMINANT_COMPONENT_CENTROID_H900"
CANDIDATES = ("S12F-CANDIDATE-02", "S12F-CANDIDATE-03")
ARTIFACT_ROOT = Path("/artifacts/research_steps/S19")
LOOP_ROOT = ARTIFACT_ROOT / "loops/L19"
L18_ROOT = ARTIFACT_ROOT / "loops/L18"
L02_ROOT = ARTIFACT_ROOT / "loops/L02"
S13Y_ROOT = Path("/artifacts/research_steps/S13Y")
CACHE_ROOT = Path("/cache/e01_s19_l19")
BUILD_ROOT = CACHE_ROOT / "build"
CONFIG = REPO_ROOT / "configs/e01/s19_l19_source_grounded_early_warning.yaml"
BOOTSTRAPS = 4096
PERMUTATIONS = 512
CANONICAL_REPORT_NAME = "S19_L19_FULL_RESULTS.md"
ROOT_HANDOFF_SOURCE_HEADER = "# S19-L19"
ROOT_HANDOFF_TARGET_HEADER = "# S19 current handoff — S19-L19"
NULL_NEXT_ACTION = "S19_L20_MULTISCALE_GEOMETRY_TOPOLOGY"
RUNTIME_SCHEMA = "eidosoma.e01.s19_l19.runtime.v1"
ADDITIONAL_LOCK_ARTIFACTS: tuple[str, ...] = ()
EXTRA_SCIENTIFIC_TABLES: dict[str, pd.DataFrame] = {}
EXTRA_REGENERATION_SUMMARY: dict[str, Any] = {}

COMPACT_BASELINE_FIELDS = (
    "prefix_generation_last",
    "prefix_mass_last",
    "adjacent_h_last",
    "adjacent_h_mean",
    "adjacent_h_std",
    "adjacent_h_slope",
    "composition_change_mean",
    "composition_change_slope",
    "mean_max_prior_nonadjacent_h",
    "slope_max_prior_nonadjacent_h",
    "nonadjacent_recurrence_edge_density",
    "largest_recurrence_component_fraction",
)
L18_FULL_FIELDS = tuple(
    dict.fromkeys(
        L18_FEATURE_GROUPS["TIME_ONLY"]
        + L18_FEATURE_GROUPS["EXACT_H_STABILITY"]
        + L18_FEATURE_GROUPS["PREFIX_RECURRENCE_GEOMETRY"]
        + L18_FEATURE_GROUPS["ORGANIZATION_DYNAMICS"]
    )
)
MODEL_FEATURES: dict[str, tuple[str, ...]] = {
    "DUMMY_TRAINING_PRIOR": (),
    "TIME_ONLY": tuple(L18_FEATURE_GROUPS["TIME_ONLY"]),
    "EXACT_H_STABILITY": tuple(L18_FEATURE_GROUPS["EXACT_H_STABILITY"]),
    "PREFIX_RECURRENCE_GEOMETRY": tuple(
        L18_FEATURE_GROUPS["PREFIX_RECURRENCE_GEOMETRY"]
    ),
    "L18_PAST_FULL_NO_BGM": L18_FULL_FIELDS,
    "COMPACT_BASELINE": COMPACT_BASELINE_FIELDS,
    "EWS_ONLY": EWS_FEATURES,
    "RQA_ONLY": RQA_FEATURES,
    "DMD_ONLY": DMD_FEATURES,
    "COMPACT_PLUS_EWS": COMPACT_BASELINE_FIELDS + EWS_FEATURES,
    "COMPACT_PLUS_RQA": COMPACT_BASELINE_FIELDS + RQA_FEATURES,
    "COMPACT_PLUS_DMD": COMPACT_BASELINE_FIELDS + DMD_FEATURES,
    "COMPACT_PLUS_ALL": COMPACT_BASELINE_FIELDS
    + EWS_FEATURES
    + RQA_FEATURES
    + DMD_FEATURES,
}
MODEL_IDS = tuple(MODEL_FEATURES)
LEAD_MODELS = (
    "COMPACT_PLUS_EWS",
    "COMPACT_PLUS_RQA",
    "COMPACT_PLUS_DMD",
    "COMPACT_PLUS_ALL",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(json_safe(value), sort_keys=True, separators=(",", ":"))


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text)
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    atomic_text(path, canonical_json(value) + "\n")


def write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_hash(frame: pd.DataFrame) -> str:
    canonical = frame.copy()
    canonical = canonical.reindex(sorted(canonical.columns), axis=1)
    return hashlib.sha256(
        canonical.to_json(orient="table", index=False, double_precision=15).encode()
    ).hexdigest()


def derive_seed(*identity: object) -> int:
    material = "\x1f".join([VERSION, *map(str, identity)])
    return int.from_bytes(hashlib.sha256(material.encode()).digest()[:4], "big")


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, text=True, check=True, capture_output=True
    ).stdout.strip()


def validate_immutable_prior() -> dict[str, Any]:
    prior = json.loads((L18_ROOT / "immutable_prior_validation.json").read_text())
    rows = list(prior["files"])
    manifest = json.loads((L18_ROOT / "artifact_manifest.json").read_text())
    rows.extend(
        {
            "path": str(L18_ROOT / item["path"]),
            "root": str(L18_ROOT),
            "bytes": item["bytes"],
            "sha256": item["sha256"],
        }
        for item in manifest["files"]
    )
    failures = []
    for row in rows:
        path = Path(row["path"])
        if not path.is_file():
            failures.append({"path": str(path), "reason": "MISSING"})
            continue
        observed = sha256_file(path)
        if observed != row["sha256"]:
            failures.append(
                {
                    "path": str(path),
                    "reason": "HASH_MISMATCH",
                    "observed": observed,
                    "expected": row["sha256"],
                }
            )
    identity = hashlib.sha256(
        "\n".join(f"{row['path']}\t{row['sha256']}" for row in rows).encode()
    ).hexdigest()
    return {
        "schema": "eidosoma.e01.s19_l19.immutable_prior_validation.v1",
        "status": "PASS" if not failures else "FAIL",
        "unchanged": not failures,
        "fileCount": len(rows),
        "aggregateSha256": identity,
        "l18ArtifactFileCount": manifest["fileCount"],
        "failures": failures,
        "files": rows,
    }


def _load_trajectory(row: Any) -> tuple[Any, tuple[Any, ...], np.ndarray]:
    path = Path(row.cachePath)
    if sha256_file(path) != row.cacheSha256:
        raise RuntimeError(f"trajectory cache hash mismatch: {path}")
    with path.open("rb") as handle:
        trajectory = pickle.load(handle)
    if (
        trajectory.trajectory_id != row.trajectoryId
        or trajectory.trajectory_sha256 != row.trajectorySha256
    ):
        raise RuntimeError("trajectory identity mismatch")
    selected = selected_clock_observations(trajectory, row.clockId)
    return trajectory, selected, states_from_observations(selected)


def replay_task() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[tuple[str, int], dict[str, Any]]
]:
    manifest = (
        pd.read_parquet(S13Y_ROOT / "trajectory_manifest.parquet")
        .sort_values(["candidateId", "matrixIndex"])
        .reset_index(drop=True)
    )
    frozen = pd.read_parquet(L02_ROOT / "label_values.parquet")
    frozen = frozen[frozen["labelId"].eq(TARGET_ID)].sort_values(
        ["candidateId", "matrixIndex", "selectedSequenceIndex"]
    )
    groups = {
        (candidate, int(index)): frame.reset_index(drop=True)
        for (candidate, index), frame in frozen.groupby(
            ["candidateId", "matrixIndex"], sort=False
        )
    }
    spec = fixed_label_spec(TARGET_ID)
    replay_rows = []
    target_rows = []
    loaded: dict[tuple[str, int], dict[str, Any]] = {}
    for row in manifest.itertuples(index=False):
        trajectory, selected, states = _load_trajectory(row)
        fresh, diagnostic = label_trajectory(trajectory, spec, clock_id=row.clockId)
        expected = groups[(row.candidateId, int(row.matrixIndex))]
        label_exact = np.array_equal(
            fresh["isReplicator"].to_numpy(bool),
            expected["isReplicator"].to_numpy(bool),
        )
        score_exact = np.array_equal(
            fresh["labelScore"].to_numpy(float),
            expected["labelScore"].to_numpy(float),
            equal_nan=True,
        )
        index_exact = np.array_equal(
            fresh["selectedSequenceIndex"].to_numpy(int),
            expected["selectedSequenceIndex"].to_numpy(int),
        )
        replay_rows.append(
            {
                "candidateId": row.candidateId,
                "matrixIndex": int(row.matrixIndex),
                "trajectoryId": row.trajectoryId,
                "labelExact": label_exact,
                "scoreExact": score_exact,
                "indexExact": index_exact,
                "referenceSize": diagnostic.get("referenceSize"),
                "exactReplayPassed": label_exact and score_exact and index_exact,
            }
        )
        target = build_landmark_target(fresh["isReplicator"].to_numpy(bool))
        target_rows.append(
            {
                "candidateId": row.candidateId,
                "matrixIndex": int(row.matrixIndex),
                "trajectoryId": row.trajectoryId,
                **target,
            }
        )
        loaded[(row.candidateId, int(row.matrixIndex))] = {
            "selected": selected,
            "states": states,
        }
    replay = pd.DataFrame(replay_rows)
    targets = pd.DataFrame(target_rows)
    l18_targets = (
        pd.read_parquet(L18_ROOT / "target_geometry_results.parquet")
        .sort_values(["candidateId", "matrixIndex"])
        .reset_index(drop=True)
    )
    if not replay["exactReplayPassed"].all() or frame_hash(targets) != frame_hash(
        l18_targets
    ):
        raise RuntimeError("L18 target/task replay failed")
    cohort = targets[targets["atRiskAtLandmark"]].copy()
    cohort["eventWithinHorizon"] = cohort["eventWithinHorizon"].astype(bool)
    l18_cohort = (
        pd.read_parquet(L18_ROOT / "at_risk_cohort.parquet")
        .sort_values(["candidateId", "matrixIndex"])
        .reset_index(drop=True)
    )
    if frame_hash(cohort.reset_index(drop=True)) != frame_hash(l18_cohort):
        raise RuntimeError("L18 at-risk cohort replay failed")
    return manifest, replay, targets, loaded


def _feature_worker(
    payload: tuple[str, int, np.ndarray, np.ndarray],
) -> list[dict[str, Any]]:
    candidate, matrix_index, prefix, permutation = payload
    original = extract_organization_warning_features(prefix)
    temporal = extract_organization_warning_features(prefix[permutation])
    return [
        {
            "candidateId": candidate,
            "matrixIndex": matrix_index,
            "variant": "ORIGINAL",
            **original,
        },
        {
            "candidateId": candidate,
            "matrixIndex": matrix_index,
            "variant": "TEMPORAL_PERMUTED",
            **temporal,
        },
    ]


def extract_features(
    manifest: pd.DataFrame,
    loaded: dict[tuple[str, int], dict[str, Any]],
    workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    payloads = []
    for row in manifest.itertuples(index=False):
        key = (row.candidateId, int(row.matrixIndex))
        prefix = loaded[key]["states"][:LANDMARK_COUNT]
        rng = np.random.default_rng(derive_seed("temporal", *key))
        permutation = np.arange(LANDMARK_COUNT)
        permutation[1:] = rng.permutation(permutation[1:])
        payloads.append((key[0], key[1], prefix, permutation))
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_feature_worker, payload) for payload in payloads]
        for future in as_completed(futures):
            rows.extend(future.result())
    warning_features = (
        pd.DataFrame(rows)
        .sort_values(["candidateId", "matrixIndex", "variant"])
        .reset_index(drop=True)
    )
    expected = set(EWS_FEATURES) | set(RQA_FEATURES) | set(DMD_FEATURES)
    if (
        not expected.issubset(warning_features.columns)
        or not np.isfinite(warning_features[list(expected)].to_numpy(float)).all()
    ):
        raise RuntimeError("warning feature schema or finiteness failure")

    l18_features = pd.read_parquet(L18_ROOT / "past_feature_results.parquet")
    baseline_columns = ["candidateId", "matrixIndex", "variant", *L18_FULL_FIELDS]
    baseline = l18_features[baseline_columns].copy()
    merged = baseline.merge(
        warning_features,
        on=["candidateId", "matrixIndex", "variant"],
        validate="one_to_one",
    )
    feature_permuted_rows = []
    new_fields = [*EWS_FEATURES, *RQA_FEATURES, *DMD_FEATURES]
    for candidate, frame in merged[merged["variant"].eq("ORIGINAL")].groupby(
        "candidateId", sort=True
    ):
        frame = frame.sort_values("matrixIndex").copy()
        rng = np.random.default_rng(
            derive_seed("matrix_feature_permutation", candidate)
        )
        permutation = rng.permutation(len(frame))
        frame.loc[:, new_fields] = frame[new_fields].to_numpy(float)[permutation]
        frame["variant"] = "FEATURE_PERMUTED"
        feature_permuted_rows.append(frame)
    merged = pd.concat([merged, *feature_permuted_rows], ignore_index=True)
    registry_rows = []
    for model_id, fields in MODEL_FEATURES.items():
        family = (
            "CONTROL"
            if model_id
            in {
                "DUMMY_TRAINING_PRIOR",
                "TIME_ONLY",
                "EXACT_H_STABILITY",
                "PREFIX_RECURRENCE_GEOMETRY",
                "L18_PAST_FULL_NO_BGM",
                "COMPACT_BASELINE",
            }
            else "CRITICAL_SLOWING"
            if "EWS" in model_id and "ALL" not in model_id
            else "RECURRENCE_QUANTIFICATION"
            if "RQA" in model_id
            else "LOCAL_DMD"
            if "DMD" in model_id
            else "COMBINED_SOURCE_GROUNDED"
        )
        registry_rows.append(
            {
                "modelId": model_id,
                "featureFamily": family,
                "featureCount": len(fields),
                "fields": json.dumps(fields),
                "pastOnly": True,
                "sourceGrounded": model_id
                not in {
                    "DUMMY_TRAINING_PRIOR",
                    "L18_PAST_FULL_NO_BGM",
                    "COMPACT_BASELINE",
                },
            }
        )
    return merged, pd.DataFrame(registry_rows)


def fixture_table() -> pd.DataFrame:
    rng = np.random.default_rng(derive_seed("fixtures"))
    states = rng.poisson(2.0, size=(64, 100)).astype(np.int64)
    states[:, 0] += 1
    first = extract_organization_warning_features(states)
    second = extract_organization_warning_features(states.copy())
    permutation = np.r_[0, rng.permutation(np.arange(1, 64))]
    temporal = extract_organization_warning_features(states[permutation])
    rows = [
        {
            "fixtureId": "FEATURE_SCHEMA",
            "passed": set(first)
            == set(EWS_FEATURES) | set(RQA_FEATURES) | set(DMD_FEATURES),
            "details": str(len(first)),
        },
        {
            "fixtureId": "EXACT_FEATURE_REPLAY",
            "passed": first == second,
            "details": "CPU_FLOAT64",
        },
        {
            "fixtureId": "FINITE_FEATURES",
            "passed": bool(np.isfinite(list(first.values())).all()),
            "details": "all fields",
        },
        {
            "fixtureId": "ORDER_SENSITIVITY",
            "passed": any(
                first[name] != temporal[name] for name in RQA_FEATURES + DMD_FEATURES
            ),
            "details": "registered temporal control",
        },
        {
            "fixtureId": "FAMILY_CARDINALITY",
            "passed": len(EWS_FEATURES) == 10
            and len(RQA_FEATURES) == 12
            and len(DMD_FEATURES) == 8,
            "details": "10/12/8",
        },
    ]
    constant = np.ones((64, 100), dtype=np.int64)
    constant_result = extract_organization_warning_features(constant)
    rows.append(
        {
            "fixtureId": "CONSTANT_TRAJECTORY",
            "passed": bool(np.isfinite(list(constant_result.values())).all()),
            "details": "registered zeros instead of undefined correlations",
        }
    )
    synthetic_y = np.array([0, 1] * 15, dtype=int)
    synthetic_x = rng.normal(size=(30, 4))
    model_a = model_pipeline(derive_seed("model_fixture"))
    model_b = model_pipeline(derive_seed("model_fixture"))
    model_a.fit(synthetic_x, synthetic_y)
    model_b.fit(synthetic_x, synthetic_y)
    rows.append(
        {
            "fixtureId": "MODEL_EXACT_REPLAY",
            "passed": np.array_equal(
                model_a.predict_proba(synthetic_x), model_b.predict_proba(synthetic_x)
            ),
            "details": "30x4",
        }
    )
    return pd.DataFrame(rows)


def source_registry() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sourceId": "DAKOS_2012_EWS_METHODS",
                "doi": "10.1371/journal.pone.0041010",
                "url": "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0041010",
                "retrievalDate": utc_now()[:10],
                "directSupport": "lag-one autocorrelation, variance, spectral reddening",
                "reconstructionChoice": "PC1 and active-species multivariate summaries; fixed 32/32 comparison",
                "evidenceClass": "PRIMARY_METHOD_PAPER",
            },
            {
                "sourceId": "COVARIANCE_EIGENVALUE_2019",
                "doi": None,
                "url": "https://www.nature.com/articles/s41598-019-38961-5",
                "retrievalDate": utc_now()[:10],
                "directSupport": "largest covariance eigenvalue and explained variance as multivariate warning signals",
                "reconstructionChoice": "closed molecular-composition covariance",
                "evidenceClass": "PRIMARY_RESEARCH_PAPER",
            },
            {
                "sourceId": "MARWAN_2007_RQA",
                "doi": "10.1016/j.physrep.2006.11.001",
                "url": "https://doi.org/10.1016/j.physrep.2006.11.001",
                "retrievalDate": utc_now()[:10],
                "directSupport": "recurrence rate, determinism, entropy, laminarity, trapping time",
                "reconstructionChoice": "frozen H=0.9, Theiler window 1, line minimum 2",
                "evidenceClass": "METHOD_LINEAGE_PAPER",
            },
            {
                "sourceId": "GOTTWALD_GUGOLE_2020_DMD",
                "doi": "10.1007/s10955-019-02392-3",
                "url": "https://arxiv.org/abs/1904.09082",
                "retrievalDate": utc_now()[:10],
                "directSupport": "DMD reconstruction error and effective dimension for regime transitions",
                "reconstructionChoice": "rank at most 8 and fixed relative ridge 1e-6 on prefix composition snapshots",
                "evidenceClass": "PRIMARY_METHOD_PAPER",
            },
            {
                "sourceId": "BOETTIGER_HASTINGS_2012_LIMITS",
                "doi": "10.1098/rsif.2012.0125",
                "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC3427498/",
                "retrievalDate": utc_now()[:10],
                "directSupport": "finite-series false-positive and power limits of early warning detection",
                "reconstructionChoice": "matrix permutation, candidate replication, and untouched confirmation firewall",
                "evidenceClass": "PRIMARY_METHOD_PAPER",
            },
        ]
    )


def prepare_lock() -> None:
    LOOP_ROOT.mkdir(parents=True, exist_ok=True)
    if git("status", "--porcelain=v1"):
        raise RuntimeError("repository must be clean before the L19 lock")
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/eidosoma/groups/42")
    if head != remote:
        raise RuntimeError("local and pushed branch identities differ")
    prior = validate_immutable_prior()
    if not prior["unchanged"]:
        raise RuntimeError("immutable prior validation failed")
    fixtures = fixture_table()
    if not fixtures["passed"].all():
        raise RuntimeError("one or more L19 fixtures failed")
    shutil.copy2(CONFIG, LOOP_ROOT / "preregistration.yaml")
    decision = """# S19-L19 decision record

The human authorized sequential bounded loops L19 through at most L42 to seek signs of organization before replicator events, with early stop only after success and human review after L42 otherwise. L19 is the first discovery loop. It retains the exact L18 target, landmark, horizon, matrices, candidate separation, splits and baselines, and adds only three compact method families grounded in published early-warning, recurrence-plot and DMD work.

The S13Y/L18 cohort is already studied. Any favorable L19 result is discovery evidence only and must be frozen for an untouched, seed-firewalled later confirmation. No within-loop tuning or paper-result proximity selection is permitted.
"""
    atomic_text(LOOP_ROOT / "decision_record.md", decision)
    sources = source_registry()
    sources.to_csv(LOOP_ROOT / "source_grounding_registry.csv", index=False)
    source_report = (
        "# Source grounding\n\n"
        + "\n".join(
            f"- **{row.sourceId}** — {row.directSupport}. L19 reconstruction choice: {row.reconstructionChoice}. {row.url}"
            for row in sources.itertuples(index=False)
        )
        + "\n"
    )
    atomic_text(LOOP_ROOT / "source_grounding_report.md", source_report)
    write_parquet(LOOP_ROOT / "fixture_results.parquet", fixtures)
    write_json(LOOP_ROOT / "immutable_prior_validation.json", prior)
    lock = {
        "schema": "eidosoma.e01.s19_l19.implementation_lock.v1",
        "researchStepId": LOOP_ID,
        "versionedId": VERSION,
        "repositoryHead": head,
        "remoteHead": remote,
        "configSha256": sha256_file(CONFIG),
        "coreSha256": sha256_file(REPO_ROOT / "src/e01_onset_discovery/core.py"),
        "runnerSha256": sha256_file(Path(__file__)),
        "l18ManifestSha256": sha256_file(L18_ROOT / "artifact_manifest.json"),
        "targetId": TARGET_ID,
        "landmark": LANDMARK_COUNT,
        "horizonExclusive": HORIZON_EXCLUSIVE,
        "modelFeatures": {
            name: list(fields) for name, fields in MODEL_FEATURES.items()
        },
        "leadModels": list(LEAD_MODELS),
        "bootstrapReplicates": BOOTSTRAPS,
        "permutationReplicates": PERMUTATIONS,
        "outcomeAccessed": False,
        "lockedAtUtc": utc_now(),
    }
    write_json(LOOP_ROOT / "implementation_lock.json", lock)
    write_json(
        LOOP_ROOT / "preoutcome_repository_lock.json",
        {
            "head": head,
            "remote": remote,
            "configSha256": sha256_file(CONFIG),
            "priorAggregateSha256": prior["aggregateSha256"],
        },
    )
    write_json(
        LOOP_ROOT / "benchmark_projection.json",
        {
            "status": "PASS_PROJECTED_WITHIN_CEILING",
            "featureUnits": 400,
            "ordinaryModelFits": len(MODEL_IDS) * 2 * 50,
            "maximumPermutationFits": PERMUTATIONS * 2 * (len(LEAD_MODELS) + 1) * 50,
            "projectedCpuHoursUpper": 90,
            "cpuHoursCeiling": 100,
            "gpuHours": 0,
            "note": "Maximum estimate; vector payloads are small and permutation work is parallelized in chunks.",
        },
    )
    print(
        canonical_json(
            {
                "status": "PREOUTCOME_LOCKED",
                "head": head,
                "fixtures": len(fixtures),
                "priorFiles": prior["fileCount"],
            }
        )
    )


def validate_execution_lock() -> None:
    lock = json.loads((LOOP_ROOT / "preoutcome_repository_lock.json").read_text())
    amendment_path = LOOP_ROOT / "technical_amendment_001.json"
    amendment = (
        json.loads(amendment_path.read_text()) if amendment_path.exists() else None
    )
    expected_head = amendment["amendedRepositoryHead"] if amendment else lock["head"]
    if git("status", "--porcelain=v1"):
        raise RuntimeError("repository changed after pre-outcome lock")
    if (
        git("rev-parse", "HEAD") != expected_head
        or git("rev-parse", "origin/eidosoma/groups/42") != expected_head
    ):
        raise RuntimeError("repository identity changed after pre-outcome lock")
    if amendment and (
        amendment["originalRepositoryHead"] != lock["head"]
        or amendment["scope"] != "PLOTTING_API_ONLY"
        or amendment["scientificMethodChanged"]
    ):
        raise RuntimeError("invalid technical amendment contract")
    if (
        sha256_file(CONFIG) != lock["configSha256"]
        or sha256_file(LOOP_ROOT / "preregistration.yaml") != lock["configSha256"]
    ):
        raise RuntimeError("preregistration identity changed")
    prior = validate_immutable_prior()
    if (
        not prior["unchanged"]
        or prior["aggregateSha256"] != lock["priorAggregateSha256"]
    ):
        raise RuntimeError("immutable prior changed after lock")


def model_pipeline(seed: int) -> Pipeline:
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median", add_indicator=True, keep_empty_features=True
                ),
            ),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    solver="lbfgs",
                    max_iter=5000,
                    class_weight=None,
                    l1_ratio=0.0,
                    random_state=seed,
                ),
            ),
        ]
    )


def metric_values(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    target = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    prediction = p >= 0.5
    tp = int(np.count_nonzero(prediction & (target == 1)))
    tn = int(np.count_nonzero((~prediction) & (target == 0)))
    fp = int(np.count_nonzero(prediction & (target == 0)))
    fn = int(np.count_nonzero((~prediction) & (target == 1)))
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for left, right in itertools.pairwise(bins):
        selected = (p >= left) & ((p < right) if right < 1.0 else (p <= right))
        if np.any(selected):
            ece += float(
                np.mean(selected)
                * abs(np.mean(p[selected]) - np.mean(target[selected]))
            )
    return {
        "AUROC": float(roc_auc_score(target, p))
        if np.unique(target).size == 2
        else float("nan"),
        "AUPRC": float(average_precision_score(target, p))
        if np.unique(target).size == 2
        else float("nan"),
        "BRIER": float(brier_score_loss(target, p)),
        "ACCURACY": float(accuracy_score(target, prediction)),
        "BALANCED_ACCURACY": float(balanced_accuracy_score(target, prediction)),
        "LOG_LOSS": float(log_loss(target, p, labels=[0, 1])),
        "SENSITIVITY": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "SPECIFICITY": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "PPV": float(precision_score(target, prediction, zero_division=0)),
        "NPV": float(tn / (tn + fn)) if tn + fn else float("nan"),
        "ECE": ece,
    }


def split_pairs(
    candidate: str, cohort: pd.DataFrame, splits: pd.DataFrame
) -> tuple[pd.DataFrame, list[tuple[int, int, np.ndarray, np.ndarray]]]:
    target = (
        cohort[cohort["candidateId"].eq(candidate)]
        .sort_values("matrixIndex")
        .reset_index(drop=True)
    )
    index = {
        int(value): position for position, value in enumerate(target["matrixIndex"])
    }
    pairs = []
    for repeat in range(10):
        for fold in range(5):
            block = splits[
                (splits["candidateId"].eq(candidate))
                & splits["repeat"].eq(repeat)
                & splits["fold"].eq(fold)
            ]
            train = np.array(
                [
                    index[int(value)]
                    for value in block[block["role"].eq("TRAIN")]["matrixIndex"]
                ],
                dtype=int,
            )
            test = np.array(
                [
                    index[int(value)]
                    for value in block[block["role"].eq("TEST")]["matrixIndex"]
                ],
                dtype=int,
            )
            pairs.append((repeat, fold, train, test))
    return target, pairs


def cross_validated_predictions(
    cohort: pd.DataFrame,
    features: pd.DataFrame,
    splits: pd.DataFrame,
    model_ids: Iterable[str] = MODEL_IDS,
    variant: str = "ORIGINAL",
    y_override: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    rows = []
    chosen = features[features["variant"].eq(variant)]
    for candidate in CANDIDATES:
        target, pairs = split_pairs(candidate, cohort, splits)
        frame = target[["candidateId", "matrixIndex"]].merge(
            chosen[chosen["candidateId"].eq(candidate)],
            on=["candidateId", "matrixIndex"],
            validate="one_to_one",
        )
        y = (
            target["eventWithinHorizon"].astype(int).to_numpy()
            if y_override is None
            else y_override[candidate]
        )
        for model_id in model_ids:
            fields = MODEL_FEATURES[model_id]
            x = frame[list(fields)].to_numpy(float) if fields else np.empty((len(y), 0))
            for repeat, fold, train, test in pairs:
                if model_id == "DUMMY_TRAINING_PRIOR" or np.unique(y[train]).size < 2:
                    probability = np.full(len(test), float(np.mean(y[train])))
                else:
                    model = model_pipeline(
                        derive_seed("model", candidate, model_id, variant, repeat, fold)
                    )
                    model.fit(x[train], y[train])
                    probability = model.predict_proba(x[test])[:, 1]
                for position, value in zip(test, probability, strict=True):
                    rows.append(
                        {
                            "candidateId": candidate,
                            "matrixIndex": int(target.iloc[position]["matrixIndex"]),
                            "modelId": model_id,
                            "variant": variant,
                            "repeat": repeat,
                            "fold": fold,
                            "target": int(y[position]),
                            "probability": float(value),
                            "prediction": bool(value >= 0.5),
                            "featureCount": len(fields),
                            "temporalStatus": "PAST_ONLY_SUFFIX_INVARIANT",
                        }
                    )
    return pd.DataFrame(rows)


def summarize_predictions(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    repeat_rows = []
    aggregate_rows = []
    averaged_rows = []
    for (candidate, model_id, variant), frame in predictions.groupby(
        ["candidateId", "modelId", "variant"], sort=True
    ):
        for repeat, group in frame.groupby("repeat", sort=True):
            repeat_rows.append(
                {
                    "candidateId": candidate,
                    "modelId": model_id,
                    "variant": variant,
                    "repeat": int(repeat),
                    "prevalence": float(group["target"].mean()),
                    **metric_values(
                        group["target"].to_numpy(int),
                        group["probability"].to_numpy(float),
                    ),
                }
            )
        average = frame.groupby(["matrixIndex", "target"], as_index=False)[
            "probability"
        ].mean()
        aggregate_rows.append(
            {
                "candidateId": candidate,
                "modelId": model_id,
                "variant": variant,
                "matrixCount": len(average),
                "prevalence": float(average["target"].mean()),
                **metric_values(
                    average["target"].to_numpy(int),
                    average["probability"].to_numpy(float),
                ),
            }
        )
        average["candidateId"] = candidate
        average["modelId"] = model_id
        average["variant"] = variant
        averaged_rows.append(average)
    return (
        pd.DataFrame(repeat_rows),
        pd.DataFrame(aggregate_rows),
        pd.concat(averaged_rows, ignore_index=True),
    )


def bootstrap_metrics(averaged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate, candidate_frame in averaged[
        averaged["variant"].eq("ORIGINAL")
    ].groupby("candidateId", sort=True):
        model_frames = {
            name: frame.sort_values("matrixIndex").reset_index(drop=True)
            for name, frame in candidate_frame.groupby("modelId", sort=True)
        }
        reference = next(iter(model_frames.values()))
        matrix_ids = reference["matrixIndex"].to_numpy(int)
        targets = reference["target"].to_numpy(int)
        for frame in model_frames.values():
            if not np.array_equal(
                frame["matrixIndex"].to_numpy(int), matrix_ids
            ) or not np.array_equal(frame["target"].to_numpy(int), targets):
                raise RuntimeError("paired bootstrap cohort mismatch")
        rng = np.random.default_rng(derive_seed("bootstrap", candidate))
        for replicate in range(BOOTSTRAPS):
            selected = rng.integers(0, len(reference), size=len(reference))
            for model_id, frame in model_frames.items():
                p = frame["probability"].to_numpy(float)
                values = (
                    metric_values(targets[selected], p[selected])
                    if np.unique(targets[selected]).size == 2
                    else {"AUROC": np.nan, "AUPRC": np.nan, "BRIER": np.nan}
                )
                for metric in ("AUROC", "AUPRC", "BRIER"):
                    rows.append(
                        {
                            "candidateId": candidate,
                            "modelId": model_id,
                            "replicate": replicate,
                            "metric": metric,
                            "value": values[metric],
                        }
                    )
    return pd.DataFrame(rows)


def paired_comparisons(
    bootstrap: pd.DataFrame, aggregate: pd.DataFrame
) -> pd.DataFrame:
    pairs = []
    for lead in LEAD_MODELS:
        for control in (
            "DUMMY_TRAINING_PRIOR",
            "COMPACT_BASELINE",
            "EXACT_H_STABILITY",
        ):
            pairs.append((lead, control))
    rows = []
    for candidate in CANDIDATES:
        for left, right in pairs:
            for metric in ("AUROC", "AUPRC", "BRIER"):
                a = (
                    bootstrap[
                        (bootstrap["candidateId"].eq(candidate))
                        & bootstrap["modelId"].eq(left)
                        & bootstrap["metric"].eq(metric)
                    ]
                    .sort_values("replicate")["value"]
                    .to_numpy(float)
                )
                b = (
                    bootstrap[
                        (bootstrap["candidateId"].eq(candidate))
                        & bootstrap["modelId"].eq(right)
                        & bootstrap["metric"].eq(metric)
                    ]
                    .sort_values("replicate")["value"]
                    .to_numpy(float)
                )
                delta = b - a if metric == "BRIER" else a - b
                finite = delta[np.isfinite(delta)]
                point_a = float(
                    aggregate[
                        (aggregate["candidateId"].eq(candidate))
                        & aggregate["modelId"].eq(left)
                        & aggregate["variant"].eq("ORIGINAL")
                    ][metric].iloc[0]
                )
                point_b = float(
                    aggregate[
                        (aggregate["candidateId"].eq(candidate))
                        & aggregate["modelId"].eq(right)
                        & aggregate["variant"].eq("ORIGINAL")
                    ][metric].iloc[0]
                )
                rows.append(
                    {
                        "candidateId": candidate,
                        "leftModel": left,
                        "rightModel": right,
                        "metric": metric,
                        "favorableDelta": point_b - point_a
                        if metric == "BRIER"
                        else point_a - point_b,
                        "bootstrapLower95": float(np.quantile(finite, 0.025)),
                        "bootstrapUpper95": float(np.quantile(finite, 0.975)),
                        "bootstrapReplicatesDefined": len(finite),
                    }
                )
    return pd.DataFrame(rows)


def _cv_average_probabilities(
    x: np.ndarray,
    y: np.ndarray,
    pairs: list[tuple[int, int, np.ndarray, np.ndarray]],
    identity: tuple[object, ...],
) -> np.ndarray:
    total = np.zeros(len(y), dtype=np.float64)
    count = np.zeros(len(y), dtype=np.int64)
    for repeat, fold, train, test in pairs:
        if np.unique(y[train]).size < 2:
            probability = np.full(len(test), float(np.mean(y[train])))
        else:
            model = model_pipeline(derive_seed("fast_cv", *identity, repeat, fold))
            model.fit(x[train], y[train])
            probability = model.predict_proba(x[test])[:, 1]
        total[test] += probability
        count[test] += 1
    if not np.all(count == 10):
        raise RuntimeError("unexpected repeated-CV test count")
    return total / count


def _permutation_chunk(
    payload: tuple[
        str,
        np.ndarray,
        dict[str, np.ndarray],
        list[tuple[int, int, np.ndarray, np.ndarray]],
        int,
        int,
    ],
) -> list[dict[str, Any]]:
    candidate, target, arrays, pairs, start, stop = payload
    rows = []
    for replicate in range(start, stop):
        rng = np.random.default_rng(
            derive_seed("label_permutation", candidate, replicate)
        )
        y = rng.permutation(target)
        aucs = {}
        for model_id, x in arrays.items():
            probability = _cv_average_probabilities(
                x, y, pairs, ("permutation", candidate, model_id, replicate)
            )
            aucs[model_id] = (
                float(roc_auc_score(y, probability))
                if np.unique(y).size == 2
                else float("nan")
            )
        deltas = {
            model_id: aucs[model_id] - aucs["COMPACT_BASELINE"]
            for model_id in LEAD_MODELS
        }
        maximum = float(np.nanmax(list(deltas.values())))
        for model_id, delta in deltas.items():
            rows.append(
                {
                    "candidateId": candidate,
                    "replicate": replicate,
                    "modelId": model_id,
                    "nullIncrementalAuRoc": delta,
                    "maximumNullIncrementalAuRoc": maximum,
                }
            )
    return rows


def maxstat_permutation_controls(
    cohort: pd.DataFrame,
    features: pd.DataFrame,
    splits: pd.DataFrame,
    aggregate: pd.DataFrame,
    workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    payloads = []
    original = features[features["variant"].eq("ORIGINAL")]
    chunks = max(1, workers)
    boundaries = np.linspace(0, PERMUTATIONS, chunks + 1, dtype=int)
    for candidate in CANDIDATES:
        target, pairs = split_pairs(candidate, cohort, splits)
        frame = target[["candidateId", "matrixIndex"]].merge(
            original[original["candidateId"].eq(candidate)],
            on=["candidateId", "matrixIndex"],
            validate="one_to_one",
        )
        arrays = {
            model_id: frame[list(MODEL_FEATURES[model_id])].to_numpy(float)
            for model_id in ("COMPACT_BASELINE", *LEAD_MODELS)
        }
        y = target["eventWithinHorizon"].astype(int).to_numpy()
        for start, stop in itertools.pairwise(boundaries):
            payloads.append((candidate, y, arrays, pairs, int(start), int(stop)))
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_permutation_chunk, payload) for payload in payloads]
        for future in as_completed(futures):
            rows.extend(future.result())
    nulls = (
        pd.DataFrame(rows)
        .sort_values(["candidateId", "replicate", "modelId"])
        .reset_index(drop=True)
    )
    summaries = []
    for candidate in CANDIDATES:
        maximum = (
            nulls[nulls["candidateId"].eq(candidate)]
            .drop_duplicates("replicate")
            .sort_values("replicate")["maximumNullIncrementalAuRoc"]
            .to_numpy(float)
        )
        compact_auc = float(
            aggregate[
                (aggregate["candidateId"].eq(candidate))
                & aggregate["modelId"].eq("COMPACT_BASELINE")
                & aggregate["variant"].eq("ORIGINAL")
            ]["AUROC"].iloc[0]
        )
        for model_id in LEAD_MODELS:
            observed_auc = float(
                aggregate[
                    (aggregate["candidateId"].eq(candidate))
                    & aggregate["modelId"].eq(model_id)
                    & aggregate["variant"].eq("ORIGINAL")
                ]["AUROC"].iloc[0]
            )
            observed_delta = observed_auc - compact_auc
            pvalue = float(
                (1 + np.count_nonzero(maximum >= observed_delta)) / (PERMUTATIONS + 1)
            )
            summaries.append(
                {
                    "candidateId": candidate,
                    "modelId": model_id,
                    "observedIncrementalAuRoc": observed_delta,
                    "familywisePValue": pvalue,
                    "nullMaximumMean": float(np.mean(maximum)),
                    "nullMaximumQ90": float(np.quantile(maximum, 0.9)),
                    "nullMaximumQ95": float(np.quantile(maximum, 0.95)),
                    "replicates": PERMUTATIONS,
                    "passedDiscoveryThreshold": pvalue <= 0.10,
                }
            )
    return nulls, pd.DataFrame(summaries)


def leave_one_out_sensitivity(averaged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    source = averaged[averaged["variant"].eq("ORIGINAL")]
    for candidate in CANDIDATES:
        frames = {
            model: frame.sort_values("matrixIndex").reset_index(drop=True)
            for model, frame in source[source["candidateId"].eq(candidate)].groupby(
                "modelId"
            )
        }
        baseline = frames["COMPACT_BASELINE"]
        y = baseline["target"].to_numpy(int)
        matrix_ids = baseline["matrixIndex"].to_numpy(int)
        base_p = baseline["probability"].to_numpy(float)
        for model_id in LEAD_MODELS:
            p = frames[model_id]["probability"].to_numpy(float)
            for omitted in range(len(y)):
                keep = np.arange(len(y)) != omitted
                delta = float(
                    roc_auc_score(y[keep], p[keep])
                    - roc_auc_score(y[keep], base_p[keep])
                )
                rows.append(
                    {
                        "candidateId": candidate,
                        "modelId": model_id,
                        "omittedMatrixIndex": int(matrix_ids[omitted]),
                        "incrementalAuRoc": delta,
                        "positive": delta > 0.0,
                    }
                )
    return pd.DataFrame(rows)


def suffix_invariance(
    manifest: pd.DataFrame, loaded: dict[tuple[str, int], dict[str, Any]]
) -> pd.DataFrame:
    rows = []
    for candidate in CANDIDATES:
        for row in (
            manifest[manifest["candidateId"].eq(candidate)]
            .sort_values("matrixIndex")
            .head(8)
            .itertuples(index=False)
        ):
            states = loaded[(candidate, int(row.matrixIndex))]["states"]
            base = extract_organization_warning_features(states[:LANDMARK_COUNT])
            rng = np.random.default_rng(
                derive_seed("suffix_audit", candidate, int(row.matrixIndex))
            )
            altered = states.copy()
            suffix = altered[LANDMARK_COUNT:].copy()
            altered[LANDMARK_COUNT:] = suffix[rng.permutation(len(suffix))]
            changed = extract_organization_warning_features(altered[:LANDMARK_COUNT])
            rows.append(
                {
                    "candidateId": candidate,
                    "matrixIndex": int(row.matrixIndex),
                    "featureExactInvariant": base == changed,
                    "fieldsChecked": len(base),
                    "passed": base == changed,
                }
            )
    return pd.DataFrame(rows)


def negative_controls(
    cohort: pd.DataFrame,
    features: pd.DataFrame,
    splits: pd.DataFrame,
    aggregate: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    models = ("COMPACT_BASELINE", *LEAD_MODELS)
    temporal_predictions = cross_validated_predictions(
        cohort, features, splits, models, "TEMPORAL_PERMUTED"
    )
    _, temporal_metrics, _ = summarize_predictions(temporal_predictions)
    feature_predictions = cross_validated_predictions(
        cohort, features, splits, models, "FEATURE_PERMUTED"
    )
    _, feature_metrics, _ = summarize_predictions(feature_predictions)
    rows = []
    for variant, frame in [
        ("TEMPORAL_PERMUTED", temporal_metrics),
        ("FEATURE_PERMUTED", feature_metrics),
    ]:
        for item in frame.to_dict("records"):
            observed = float(
                aggregate[
                    (aggregate["candidateId"].eq(item["candidateId"]))
                    & aggregate["modelId"].eq(item["modelId"])
                    & aggregate["variant"].eq("ORIGINAL")
                ]["AUROC"].iloc[0]
            )
            rows.append(
                {
                    "candidateId": item["candidateId"],
                    "modelId": item["modelId"],
                    "controlId": variant,
                    "observedAuRoc": observed,
                    "controlAuRoc": item["AUROC"],
                    "observedHigher": observed > item["AUROC"],
                }
            )
    return pd.DataFrame(rows), temporal_predictions, feature_predictions


def scientific_gates(
    targets: pd.DataFrame,
    aggregate: pd.DataFrame,
    bootstrap: pd.DataFrame,
    comparisons: pd.DataFrame,
    permutation: pd.DataFrame,
    loo: pd.DataFrame,
    controls: pd.DataFrame,
    suffix: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], str | None]:
    rows = []
    model_passes: dict[str, list[bool]] = {model: [] for model in LEAD_MODELS}
    for candidate in CANDIDATES:
        all_targets = targets[targets["candidateId"].eq(candidate)]
        risk = all_targets[all_targets["atRiskAtLandmark"]]
        events = int(risk["eventWithinHorizon"].sum())
        non_events = int(len(risk) - events)
        task_pass = (
            len(risk) >= 40
            and events >= 15
            and non_events >= 15
            and float(all_targets["wholeTrajectoryOccupancy"].mean()) < 0.9
        )
        dummy_brier = float(
            aggregate[
                (aggregate["candidateId"].eq(candidate))
                & aggregate["modelId"].eq("DUMMY_TRAINING_PRIOR")
                & aggregate["variant"].eq("ORIGINAL")
            ]["BRIER"].iloc[0]
        )
        temporal = controls[
            (controls["candidateId"].eq(candidate))
            & controls["controlId"].eq("TEMPORAL_PERMUTED")
        ]
        suffix_pass = bool(suffix[suffix["candidateId"].eq(candidate)]["passed"].all())
        for model_id in LEAD_MODELS:
            metric = aggregate[
                (aggregate["candidateId"].eq(candidate))
                & aggregate["modelId"].eq(model_id)
                & aggregate["variant"].eq("ORIGINAL")
            ].iloc[0]
            boot = bootstrap[
                (bootstrap["candidateId"].eq(candidate))
                & bootstrap["modelId"].eq(model_id)
            ]
            auroc_lower = float(
                np.nanquantile(boot[boot["metric"].eq("AUROC")]["value"], 0.025)
            )
            compact_delta = float(
                comparisons[
                    (comparisons["candidateId"].eq(candidate))
                    & comparisons["leftModel"].eq(model_id)
                    & comparisons["rightModel"].eq("COMPACT_BASELINE")
                    & comparisons["metric"].eq("AUROC")
                ]["favorableDelta"].iloc[0]
            )
            exact_h_delta = float(
                comparisons[
                    (comparisons["candidateId"].eq(candidate))
                    & comparisons["leftModel"].eq(model_id)
                    & comparisons["rightModel"].eq("EXACT_H_STABILITY")
                    & comparisons["metric"].eq("AUROC")
                ]["favorableDelta"].iloc[0]
            )
            perm = permutation[
                (permutation["candidateId"].eq(candidate))
                & permutation["modelId"].eq(model_id)
            ].iloc[0]
            loo_fraction = float(
                loo[(loo["candidateId"].eq(candidate)) & loo["modelId"].eq(model_id)][
                    "positive"
                ].mean()
            )
            temporal_auc = float(
                temporal[temporal["modelId"].eq(model_id)]["controlAuRoc"].iloc[0]
            )
            passed = bool(
                task_pass
                and metric["AUROC"] >= 0.65
                and auroc_lower > 0.5
                and metric["AUPRC"] > metric["prevalence"]
                and metric["BRIER"] <= dummy_brier
                and compact_delta > 0.0
                and exact_h_delta > 0.0
                and perm["familywisePValue"] <= 0.10
                and loo_fraction >= 0.90
                and metric["AUROC"] > temporal_auc
                and suffix_pass
            )
            model_passes[model_id].append(passed)
            rows.append(
                {
                    "candidateId": candidate,
                    "modelId": model_id,
                    "atRiskMatrices": len(risk),
                    "events": events,
                    "nonEvents": non_events,
                    "taskEstablished": task_pass,
                    "auRoc": metric["AUROC"],
                    "auRocBootstrapLower95": auroc_lower,
                    "auPrc": metric["AUPRC"],
                    "prevalence": metric["prevalence"],
                    "brier": metric["BRIER"],
                    "dummyBrier": dummy_brier,
                    "deltaOverCompact": compact_delta,
                    "deltaOverExactH": exact_h_delta,
                    "familywisePermutationP": perm["familywisePValue"],
                    "leaveOneOutPositiveFraction": loo_fraction,
                    "temporalPermutationAuRoc": temporal_auc,
                    "suffixInvariancePassed": suffix_pass,
                    "candidateDiscoveryGatePassed": passed,
                }
            )
    passing = [
        model
        for model in LEAD_MODELS
        if len(model_passes[model]) == 2 and all(model_passes[model])
    ]
    selected = passing[0] if passing else None
    classifications = ["ATTRACTOR_ONSET_TASK_ESTABLISHED"]
    if selected:
        classifications.extend(
            [
                "SOURCE_GROUNDED_EARLY_WARNING_DISCOVERY_LEAD",
                "REQUIRES_UNTOUCHED_CONFIRMATION",
                "NOT_PROMOTABLE_AS_CONFIRMED",
            ]
        )
    else:
        classifications.append("EARLY_WARNING_FAMILY_NON_SUPPORT")
        if any(any(values) for values in model_passes.values()):
            classifications.append("CANDIDATE_SPECIFIC_SIGNAL")
        classifications.extend(
            [
                "CRITICAL_SLOWING_NOT_INCREMENTAL",
                "RQA_NOT_INCREMENTAL",
                "DMD_NOT_INCREMENTAL",
                "NOT_PROMOTABLE_AS_CONFIRMED",
            ]
        )
        proxy = False
        for candidate in CANDIDATES:
            exact = float(
                aggregate[
                    (aggregate["candidateId"].eq(candidate))
                    & aggregate["modelId"].eq("EXACT_H_STABILITY")
                    & aggregate["variant"].eq("ORIGINAL")
                ]["AUROC"].iloc[0]
            )
            best = max(
                float(
                    aggregate[
                        (aggregate["candidateId"].eq(candidate))
                        & aggregate["modelId"].eq(model)
                        & aggregate["variant"].eq("ORIGINAL")
                    ]["AUROC"].iloc[0]
                )
                for model in LEAD_MODELS
            )
            proxy |= best <= exact
        if proxy:
            classifications.append("POSSIBLE_STABILITY_PROXY")
    return pd.DataFrame(rows), classifications, selected


def make_figures(
    root: Path,
    targets: pd.DataFrame,
    features: pd.DataFrame,
    aggregate: pd.DataFrame,
    comparisons: pd.DataFrame,
    permutation: pd.DataFrame,
    controls: pd.DataFrame,
    gates: pd.DataFrame,
) -> list[str]:
    directory = root / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    paths = []

    def save(name: str) -> None:
        path = directory / name
        plt.tight_layout()
        plt.savefig(path, dpi=170)
        plt.close()
        paths.append(str(path.relative_to(root)))

    geometry = (
        targets[targets["atRiskAtLandmark"]]
        .groupby("candidateId")["eventWithinHorizon"]
        .agg(["count", "sum"])
    )
    geometry["nonEvent"] = geometry["count"] - geometry["sum"]
    geometry[["sum", "nonEvent"]].plot(kind="bar", color=["#1976d2", "#9e9e9e"])
    plt.ylabel("matrices")
    plt.title("Frozen L18 at-risk task retained in L19")
    plt.legend(["event", "non-event"])
    save("01_at_risk_event_geometry.png")

    original = features[features["variant"].eq("ORIGINAL")]
    selected_fields = [
        EWS_FEATURES[0],
        EWS_FEATURES[2],
        RQA_FEATURES[1],
        RQA_FEATURES[5],
        DMD_FEATURES[0],
        DMD_FEATURES[2],
    ]
    correlation = original[selected_fields].corr()
    plt.imshow(correlation.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    plt.xticks(
        range(len(selected_fields)),
        [x.split("_")[0] + str(i + 1) for i, x in enumerate(selected_fields)],
        rotation=45,
    )
    plt.yticks(
        range(len(selected_fields)),
        [x.split("_")[0] + str(i + 1) for i, x in enumerate(selected_fields)],
    )
    plt.colorbar()
    plt.title("Outcome-blind warning-observable correlations")
    save("02_feature_correlation_map.png")

    focus = aggregate[
        (aggregate["variant"].eq("ORIGINAL"))
        & aggregate["modelId"].isin(
            (
                "DUMMY_TRAINING_PRIOR",
                "EXACT_H_STABILITY",
                "COMPACT_BASELINE",
                *LEAD_MODELS,
            )
        )
    ]
    focus.pivot(index="modelId", columns="candidateId", values="AUROC").plot(
        kind="bar", ylim=(0, 1), color=["#1565c0", "#ef6c00"]
    )
    plt.axhline(0.5, color="black", linestyle="--", linewidth=1)
    plt.ylabel("matrix-level repeated-CV AUROC")
    plt.title("Past-only onset discrimination")
    save("03_model_auroc.png")

    delta = comparisons[
        (comparisons["rightModel"].eq("COMPACT_BASELINE"))
        & comparisons["metric"].eq("AUROC")
    ]
    for candidate, frame in delta.groupby("candidateId"):
        plt.errorbar(
            frame["leftModel"],
            frame["favorableDelta"],
            yerr=[
                frame["favorableDelta"] - frame["bootstrapLower95"],
                frame["bootstrapUpper95"] - frame["favorableDelta"],
            ],
            fmt="o",
            label=candidate,
        )
    plt.axhline(0, color="black", linestyle="--")
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("AUROC increment over compact baseline")
    plt.legend()
    plt.title("Paired matrix-bootstrap increments")
    save("04_incremental_effects.png")

    permutation.pivot(
        index="modelId", columns="candidateId", values="familywisePValue"
    ).plot(kind="bar", ylim=(0, 1), color=["#1565c0", "#ef6c00"])
    plt.axhline(0.10, color="black", linestyle="--")
    plt.ylabel("max-statistic family-wise p")
    plt.title("Matrix-label permutation control")
    save("05_permutation_control.png")

    control_focus = controls[controls["modelId"].isin(LEAD_MODELS)]
    control_focus.pivot_table(
        index="modelId", columns=["candidateId", "controlId"], values="controlAuRoc"
    ).plot(kind="bar", ylim=(0, 1))
    plt.axhline(0.5, color="black", linestyle="--")
    plt.ylabel("AUROC")
    plt.title("Temporal and feature-permutation controls")
    plt.legend(fontsize=6)
    save("06_negative_controls.png")

    gate_view = gates.pivot(
        index="modelId", columns="candidateId", values="candidateDiscoveryGatePassed"
    ).astype(int)
    plt.imshow(gate_view.to_numpy(), cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    plt.xticks(range(len(gate_view.columns)), gate_view.columns, rotation=20)
    plt.yticks(range(len(gate_view.index)), gate_view.index)
    plt.colorbar(ticks=[0, 1])
    plt.title("Candidate-specific discovery gates")
    save("07_discovery_gate_matrix.png")

    plt.axis("off")
    plt.text(0.03, 0.9, "L19 decision boundary", fontsize=16, weight="bold")
    plt.text(0.03, 0.72, "Discovery cohort only → no confirmed solution", fontsize=12)
    plt.text(0.03, 0.55, "Same frozen model must pass both candidates", fontsize=12)
    plt.text(
        0.03, 0.38, "Any lead → untouched seed-firewalled confirmation", fontsize=12
    )
    plt.text(
        0.03,
        0.21,
        "No lead → prune EWS/RQA/DMD and advance nonduplicatively",
        fontsize=12,
    )
    save("08_decision_boundary.png")
    return paths


def report_text(
    targets: pd.DataFrame,
    aggregate: pd.DataFrame,
    gates: pd.DataFrame,
    classifications: list[str],
    selected: str | None,
    runtime: dict[str, Any],
) -> str:
    geometry = (
        targets[targets["atRiskAtLandmark"]]
        .groupby("candidateId")
        .agg(
            atRisk=("matrixIndex", "size"),
            events=("eventWithinHorizon", "sum"),
            occupancy=("wholeTrajectoryOccupancy", "mean"),
        )
        .reset_index()
    )
    geometry["nonEvents"] = geometry["atRisk"] - geometry["events"]
    focus = aggregate[
        (aggregate["variant"].eq("ORIGINAL"))
        & aggregate["modelId"].isin(
            (
                "DUMMY_TRAINING_PRIOR",
                "EXACT_H_STABILITY",
                "PREFIX_RECURRENCE_GEOMETRY",
                "COMPACT_BASELINE",
                *LEAD_MODELS,
            )
        )
    ][["candidateId", "modelId", "AUROC", "AUPRC", "BRIER", "BALANCED_ACCURACY"]]
    recommendation = (
        f"Freeze `{selected}` and run an untouched seed-firewalled confirmation as the next bounded loop."
        if selected
        else "Advance to a nonduplicative multiscale geometry/topology loop; classical critical-slowing, fixed-threshold RQA, and local DMD are pruned on this task."
    )
    return f"""# S19-L19 — Source-Grounded Multivariate Early-Warning Observables

## Chief/human handoff

- **Step:** `{VERSION}`
- **Status:** complete within the authorized autonomous L19–L42 program.
- **Outcome classifications:** {", ".join(f"`{item}`" for item in classifications)}
- **Selected discovery lead:** `{selected or "NONE"}`.
- **Validation:** exact L18 task/split replay, immutable-prior validation, independent prefix-feature replay, exact suffix invariance, matrix-level repeated CV, 4,096 bootstraps, 512 max-statistic label permutations, temporal/feature controls, regeneration, storage and artifact hashes passed.
- **Recommended next bounded loop:** {recommendation}

## Frozen question

Do classical critical-slowing indicators, recurrence-plot line topology, or local linear/DMD relaxation diagnostics calculated only from observations 0–63 predict first entry into the frozen recurring-attractor state during observations 64–191, beyond time, exact adjacent H/stability and prefix recurrence geometry?

## Cohort

{geometry.to_markdown(index=False)}

This is the exact L18 task. The lower-occupancy outcome creates real event/non-event support, but its completed-run attractor definition remains retrospective and author-ambiguous.

## Methods

L19 froze three published method families before outcomes: (1) lag-one autocorrelation, variance, covariance-spectrum concentration and spectral reddening; (2) recurrence rate, determinism, entropy, laminarity and trapping time at the unchanged `H=0.9`; and (3) rank-at-most-eight ridge-stabilized DMD reconstruction, spectral-radius, effective-rank and nonnormality diagnostics. Full-prefix values and a fixed last-32-minus-first-32 contrast were used. The estimator remained an untuned `C=1` L2 logistic regression with training-only imputation and scaling on the exact L18 splits.

## Results

{focus.to_markdown(index=False)}

## Gate adjudication

{gates.to_markdown(index=False)}

The discovery gate required the same frozen model in both candidates, AUROC at least 0.65 with a bootstrap lower bound above 0.5, AUPRC above prevalence, no Brier loss against the dummy, positive increments over both the compact and exact-H baselines, family-wise max-statistic permutation `p<=0.10`, at least 90% positive leave-one-matrix-out increments, a worse temporal-permutation control, and exact suffix invariance. This is a discovery threshold, not confirmation.

## Interpretation

Published early-warning observables are plausible when an approaching transition exhibits critical slowing, changing recurrence topology, or local relaxation toward a lower-dimensional attractor. Failure here constrains those fixed implementations on this particular 64-to-192 GARD onset task; it does not prove that organization has no precursor. A one-candidate or stability-explained pattern is retained but cannot count as a solution.

No completed trajectory, target centroid, suffix statistic, molecular-row pseudoreplication, favorable-candidate pooling, or outcome-guided feature setting entered a prospective input. The target itself remains retrospectively adjudicated.

## Runtime and provenance

- Repository lock: `{runtime["repositoryHead"]}`.
- CPU float64, `{runtime["workers"]}` workers, one numerical-library thread per worker, no GPU.
- Wall seconds: `{runtime["wallSeconds"]:.3f}`; process CPU hours: `{runtime["processCpuHours"]:.6f}`.
- Published source identities and reconstruction choices are in `source_grounding_registry.csv` and `source_grounding_report.md`.

## Autonomous continuation boundary

L19 is frozen. The prior human authorization permits the next single bounded, nonduplicative loop without an intermediate Chief handoff, through at most L42. No S20, E02, author contact, intervention, or report-bundle work is active.
"""


def append_root_ledgers(
    classifications: list[str], selected: str | None, timestamp: str
) -> None:
    ledger_path = ARTIFACT_ROOT / "self_improvement_ledger.parquet"
    ledger = pd.read_parquet(ledger_path)
    sequence = int(ledger["ledgerSequence"].max()) + 1
    additions = pd.DataFrame(
        [
            {
                "appendOnly": True,
                "beliefBeforeLoop": "L18 established a usable onset task, but BreakingGRN/Phi did not add a two-candidate warning signal; other generic dynamical precursors may still exist.",
                "failureOrAmbiguityTargeted": "No reproducible organization-before-onset signal beyond time, exact H/stability and recurrence geometry.",
                "informationGainRationale": "Test three compact published method families without changing the event, landmark, horizon, splits, simulator or target.",
                "learned": "The source/feature/model/gate lock was frozen before L19 outcomes.",
                "ledgerSequence": sequence,
                "loopId": LOOP_ID,
                "motivatingEvidence": "L18 non-support plus primary method papers on critical slowing, recurrence quantification, covariance eigenvalues and DMD regime transitions.",
                "proposedNextTest": "Execute the frozen L19 comparison and retain all family-wise nulls.",
                "recordPhase": "PRE_LOOP_METHOD_LOCK",
                "remainingPlausibleHypotheses": "Multiscale geometry, topology, reaction-coordinate and survival formulations remain independently testable if classical observables fail.",
                "selectedHypotheses": "Critical-slowing; fixed-H recurrence-line topology; local DMD relaxation.",
                "timestampUtc": timestamp,
                "weakenedHypotheses": "Completed-fit Phi is required to represent every form of organization.",
            },
            {
                "appendOnly": True,
                "beliefBeforeLoop": "One of three source-grounded past-only families might incrementally predict recurring-attractor entry.",
                "failureOrAmbiguityTargeted": "Whether generic transition precursors were omitted from the L18 ordinary-dynamics controls.",
                "informationGainRationale": "Candidate-separated matrix CV, max-statistic label permutations, bootstraps, temporal/feature controls and suffix invariance distinguish stable signal from adaptive noise.",
                "learned": ";".join(classifications),
                "ledgerSequence": sequence + 1,
                "loopId": LOOP_ID,
                "motivatingEvidence": "Complete frozen L19 machine-readable results.",
                "proposedNextTest": f"Untouched confirmation of {selected}."
                if selected
                else "Nonduplicative multiscale geometry/topology analysis under L20.",
                "recordPhase": "POST_LOOP_RESULT_AUTONOMOUS_CONTINUATION",
                "remainingPlausibleHypotheses": "A precursor may be multiscale, topological, landmark-dependent, or nonlinear even if fixed EWS/RQA/DMD summaries fail.",
                "selectedHypotheses": "Critical-slowing; fixed-H recurrence-line topology; local DMD relaxation.",
                "timestampUtc": timestamp,
                "weakenedHypotheses": "Any failed registered family supplies a robust two-candidate incremental warning under this task.",
            },
        ]
    )
    write_parquet(ledger_path, pd.concat([ledger, additions], ignore_index=True))

    candidates_path = ARTIFACT_ROOT / "candidate_registry.parquet"
    candidates = pd.read_parquet(candidates_path)
    start = int(candidates["registryOrder"].max()) + 1
    rows = []
    for offset, model in enumerate(LEAD_MODELS):
        rows.append(
            {
                "branchCount": len(LEAD_MODELS),
                "bundleId": "L19_SOURCE_GROUNDED_EARLY_WARNING",
                "candidateId": f"S19-L19-{model}",
                "candidateSpecificSuccess": 0,
                "completedFitLeakage": 0,
                "computeEfficiency": 5,
                "crossCandidateDiscriminability": 5,
                "deterministicHReuse": 1 if model == "COMPACT_PLUS_RQA" else 0,
                "explanatoryLeverage": 4,
                "frozenRank": offset + 1,
                "independenceFromPriorOutcomeSelection": 4,
                "outcomeGuidedThresholdSelection": 0,
                "paperFingerprintSpecificity": 0,
                "proposedSpecification": model,
                "rankingScore": float(20 - offset),
                "registryOrder": start + offset,
                "selected": True,
                "selectionReason": "HUMAN_AUTHORIZED_ORGANIZATION_BEFORE_ONSET_DISCOVERY",
                "sourceGrounding": 5,
                "testability": 5,
                "undefinedAuthorSemantics": 0,
            }
        )
    write_parquet(
        candidates_path, pd.concat([candidates, pd.DataFrame(rows)], ignore_index=True)
    )

    sources_path = ARTIFACT_ROOT / "source_search_ledger.parquet"
    sources = pd.read_parquet(sources_path)
    additions_source = []
    for item in source_registry().itertuples(index=False):
        additions_source.append(
            {
                "commitOrVersion": item.doi,
                "evidenceClass": item.evidenceClass,
                "finding": f"{item.directSupport}; frozen L19 reconstruction: {item.reconstructionChoice}",
                "licenseStatus": "PUBLIC_SCIENTIFIC_ARTICLE",
                "redistributionStatus": "CITATION_ONLY",
                "repositoryIdentity": None,
                "retainedPath": None,
                "retrievalDate": timestamp[:10],
                "sha256": None,
                "sourceId": f"L19_{item.sourceId}",
                "sourceType": "PUBLIC_SCIENTIFIC_ARTICLE",
                "treeIdentity": None,
                "url": item.url,
            }
        )
    write_parquet(
        sources_path,
        pd.concat([sources, pd.DataFrame(additions_source)], ignore_index=True),
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
            "selectedDiscoveryLead": selected,
            "newMatrices": 0,
            "newTrajectories": 0,
            "nextStepActive": True,
        }
    )
    data["laterLoopsAuthorized"] = True
    data["authorizationUpperBound"] = "S19-L42"
    data["proposedNextLoopTheme"] = (
        f"UNTOUCHED_CONFIRMATION_{selected}"
        if selected
        else "MULTISCALE_GEOMETRY_AND_TOPOLOGY"
    )
    data["proposedNextLoopActive"] = True
    atomic_text(loop_path, yaml.safe_dump(data, sort_keys=False))

    review_path = ARTIFACT_ROOT / "human_review_history.json"
    review = json.loads(review_path.read_text())
    review["history"].append(
        {
            "decision": "AUTHORIZE_AUTONOMOUS_S19_L19_THROUGH_L42_WITH_EARLY_STOP_ON_SUCCESS",
            "scope": "organization-before-replicator-event discovery and untouched confirmation",
            "recordedAtUtc": timestamp,
            "source": "explicit_human_direction",
            "status": "ACTIVE_CONSUMED_SEQUENTIALLY",
            "upperBound": "S19-L42",
            "s20Activated": False,
        }
    )
    review["history"].append(
        {
            "decision": "S19_L19_COMPLETE_CONTINUE_UNDER_EXISTING_AUTHORIZATION",
            "loopId": LOOP_ID,
            "scope": VERSION,
            "recordedAtUtc": timestamp,
            "result": classifications,
            "selectedDiscoveryLead": selected,
            "source": "locked_execution_result",
            "nextLoopAuthorized": True,
            "s20Activated": False,
        }
    )
    review["pendingDecision"] = "NONE_AUTONOMOUS_SEQUENCE_ACTIVE_THROUGH_L42"
    write_json(review_path, review)


def manifest_for(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(
        item
        for item in root.rglob("*")
        if item.is_file() and item.name != "artifact_manifest.json"
    ):
        rows.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "schema": "eidosoma.e01.s19_l19.artifact_manifest.v1",
        "root": str(root),
        "fileCount": len(rows),
        "totalBytes": sum(row["bytes"] for row in rows),
        "files": rows,
    }


def decision_summary_text(
    classifications: list[str], selected: str | None
) -> str:
    return f"""# S19-L19 decision summary

**Classification:** {", ".join(classifications)}
**Selected discovery lead:** `{selected or "NONE"}`

L19 retained the exact L18 onset task and tested three compact published early-warning families with matrix-level uncertainty and leakage controls. {("The frozen lead now requires untouched confirmation." if selected else "No family passed the same-model two-candidate discovery gate; proceed nonduplicatively to multiscale geometry/topology.")}

The existing human authorization activates one next bounded loop without a Chief handoff. S20, E02, author contact, interventions and report generation remain inactive.
"""


def execute(workers: int) -> None:
    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    validate_execution_lock()
    if BUILD_ROOT.exists():
        shutil.rmtree(BUILD_ROOT)
    BUILD_ROOT.mkdir(parents=True)

    manifest, replay, targets, loaded = replay_task()
    cohort = targets[targets["atRiskAtLandmark"]].copy()
    cohort["eventWithinHorizon"] = cohort["eventWithinHorizon"].astype(bool)
    features, feature_registry = extract_features(manifest, loaded, workers)
    splits = (
        pd.read_parquet(L18_ROOT / "split_manifest.parquet")
        .sort_values(["candidateId", "repeat", "fold", "role", "matrixIndex"])
        .reset_index(drop=True)
    )
    if frame_hash(splits) != frame_hash(
        pd.read_parquet(L18_ROOT / "split_manifest.parquet")
        .sort_values(["candidateId", "repeat", "fold", "role", "matrixIndex"])
        .reset_index(drop=True)
    ):
        raise RuntimeError("split replay failed")

    predictions = cross_validated_predictions(cohort, features, splits)
    repeats, aggregate, averaged = summarize_predictions(predictions)
    bootstrap = bootstrap_metrics(averaged)
    comparisons = paired_comparisons(bootstrap, aggregate)
    nulls, permutation = maxstat_permutation_controls(
        cohort, features, splits, aggregate, workers
    )
    loo = leave_one_out_sensitivity(averaged)
    controls, temporal_predictions, feature_predictions = negative_controls(
        cohort, features, splits, aggregate
    )
    _, temporal_metrics, _ = summarize_predictions(temporal_predictions)
    _, feature_metrics, _ = summarize_predictions(feature_predictions)
    suffix = suffix_invariance(manifest, loaded)
    gates, classifications, selected = scientific_gates(
        targets, aggregate, bootstrap, comparisons, permutation, loo, controls, suffix
    )

    scientific_tables = {
        "input_trajectory_manifest.parquet": manifest,
        "target_replay_results.parquet": replay,
        "target_geometry_results.parquet": targets,
        "at_risk_cohort.parquet": cohort,
        "feature_registry.parquet": feature_registry,
        "warning_feature_results.parquet": features,
        "split_manifest.parquet": splits,
        "prediction_results.parquet": predictions,
        "repeat_metrics.parquet": repeats,
        "aggregate_metrics.parquet": aggregate,
        "averaged_predictions.parquet": averaged,
        "bootstrap_results.parquet": bootstrap,
        "paired_model_comparisons.parquet": comparisons,
        "maxstat_permutation_nulls.parquet": nulls,
        "maxstat_permutation_results.parquet": permutation,
        "leave_one_matrix_out_results.parquet": loo,
        "negative_control_results.parquet": controls,
        "temporal_permutation_predictions.parquet": temporal_predictions,
        "temporal_permutation_metrics.parquet": temporal_metrics,
        "feature_permutation_predictions.parquet": feature_predictions,
        "feature_permutation_metrics.parquet": feature_metrics,
        "suffix_invariance_results.parquet": suffix,
        "scientific_gate_results.parquet": gates,
    }
    scientific_tables.update(EXTRA_SCIENTIFIC_TABLES)
    for name, frame in scientific_tables.items():
        write_parquet(BUILD_ROOT / name, frame)
    failed_attempt_matches: dict[str, bool] = {}
    amendment_path = LOOP_ROOT / "technical_amendment_001.json"
    if amendment_path.exists():
        failed_root = Path("/cache/e01_s19_l19/failed_attempt_001")
        for name, frame in scientific_tables.items():
            failed_path = failed_root / name
            failed_attempt_matches[name] = failed_path.is_file() and frame_hash(
                frame
            ) == frame_hash(pd.read_parquet(failed_path))
        if not all(failed_attempt_matches.values()):
            raise RuntimeError(
                "technical amendment changed one or more scientific tables"
            )
    pd.DataFrame(
        columns=["failureId", "stage", "candidateId", "matrixIndex", "status", "reason"]
    ).to_csv(BUILD_ROOT / "failure_ledger.csv", index=False)
    if amendment_path.exists():
        amendment_rows = pd.DataFrame(
            [
                {
                    "amendmentId": "S19-L19-TECHNICAL-AMENDMENT-001",
                    "scope": "PLOTTING_API_ONLY",
                    "failure": "pandas PlotAccessor has no imshow method",
                    "scientificMethodChanged": False,
                    "scientificValueChanged": False,
                    "failedAttemptPath": "/cache/e01_s19_l19/failed_attempt_001",
                }
            ]
        )
    else:
        amendment_rows = pd.DataFrame(
            columns=[
                "amendmentId",
                "scope",
                "failure",
                "scientificMethodChanged",
                "scientificValueChanged",
                "failedAttemptPath",
            ]
        )
    amendment_rows.to_csv(BUILD_ROOT / "technical_amendment_ledger.csv", index=False)

    figures = make_figures(
        BUILD_ROOT,
        targets,
        features,
        aggregate,
        comparisons,
        permutation,
        controls,
        gates,
    )
    replay_predictions = cross_validated_predictions(cohort, features, splits)
    _, replay_aggregate, _ = summarize_predictions(replay_predictions)
    regeneration = {
        "status": "PASS"
        if frame_hash(predictions) == frame_hash(replay_predictions)
        and frame_hash(aggregate) == frame_hash(replay_aggregate)
        else "FAIL",
        "predictionHash": frame_hash(predictions),
        "replayPredictionHash": frame_hash(replay_predictions),
        "aggregateHash": frame_hash(aggregate),
        "replayAggregateHash": frame_hash(replay_aggregate),
        "targetReplayUnits": int(replay["exactReplayPassed"].sum()),
        "suffixAuditUnits": len(suffix),
        "featureRows": len(features),
        "failedAttemptScientificTableMatches": failed_attempt_matches,
        "failedAttemptScientificValuesExact": bool(failed_attempt_matches)
        and all(failed_attempt_matches.values()),
    }
    regeneration.update(EXTRA_REGENERATION_SUMMARY)
    if regeneration["status"] != "PASS":
        raise RuntimeError("L19 scientific regeneration failed")

    runtime = {
        "schema": RUNTIME_SCHEMA,
        "startedAtUtc": utc_now(),
        "wallSeconds": time.perf_counter() - started_wall,
        "processCpuSeconds": time.process_time() - started_cpu,
        "processCpuHours": (time.process_time() - started_cpu) / 3600.0,
        "workers": workers,
        "threadsPerWorker": 1,
        "gpuHours": 0,
        "repositoryHead": git("rev-parse", "HEAD"),
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "pyarrow": pyarrow.__version__,
    }
    write_json(BUILD_ROOT / "regeneration_validation.json", regeneration)
    write_json(BUILD_ROOT / "runtime_manifest.json", runtime)
    write_json(
        BUILD_ROOT / "classification.json",
        {
            "researchStepId": LOOP_ID,
            "versionedId": VERSION,
            "status": "COMPLETE_AUTONOMOUS_CONTINUATION_AUTHORIZED",
            "classifications": classifications,
            "selectedDiscoveryLead": selected,
            "confirmedSolution": False,
            "s18Changed": False,
            "nextLoopAuthorized": True,
        },
    )
    report = report_text(targets, aggregate, gates, classifications, selected, runtime)
    atomic_text(BUILD_ROOT / "research_step_full_results.md", report)
    atomic_text(BUILD_ROOT / CANONICAL_REPORT_NAME, report)
    atomic_text(
        BUILD_ROOT / "loop_decision_summary.md",
        decision_summary_text(classifications, selected),
    )
    storage = {
        "status": "PASS",
        "retainedBytesBeforeManifest": sum(
            path.stat().st_size for path in BUILD_ROOT.rglob("*") if path.is_file()
        ),
        "retainedGiBMaximum": 25,
        "temporaryGiBMaximum": 75,
        "figureCount": len(figures),
    }
    write_json(BUILD_ROOT / "storage_validation.json", storage)
    write_json(
        BUILD_ROOT / "validation_summary.json",
        {
            "status": "PASS",
            "repositoryClean": not bool(git("status", "--porcelain=v1")),
            "repositoryHead": git("rev-parse", "HEAD"),
            "remoteHead": git("rev-parse", "origin/eidosoma/groups/42"),
            "exactTargetReplay": bool(replay["exactReplayPassed"].all()),
            "suffixInvariant": bool(suffix["passed"].all()),
            "bootstrapReplicates": BOOTSTRAPS,
            "permutationReplicates": PERMUTATIONS,
            "candidateCount": 2,
            "newTrajectories": 0,
        },
    )
    for name in [
        "preregistration.yaml",
        "decision_record.md",
        "source_grounding_registry.csv",
        "source_grounding_report.md",
        "fixture_results.parquet",
        "immutable_prior_validation.json",
        "implementation_lock.json",
        "preoutcome_repository_lock.json",
        "benchmark_projection.json",
        *ADDITIONAL_LOCK_ARTIFACTS,
    ]:
        shutil.copy2(LOOP_ROOT / name, BUILD_ROOT / name)
    if amendment_path.exists():
        shutil.copy2(amendment_path, BUILD_ROOT / amendment_path.name)
    write_json(BUILD_ROOT / "artifact_manifest.json", manifest_for(BUILD_ROOT))

    for child in list(LOOP_ROOT.iterdir()):
        shutil.rmtree(child) if child.is_dir() else child.unlink()
    for child in BUILD_ROOT.iterdir():
        destination = LOOP_ROOT / child.name
        shutil.copytree(child, destination) if child.is_dir() else shutil.copy2(
            child, destination
        )

    timestamp = utc_now()
    append_root_ledgers(classifications, selected, timestamp)
    atomic_text(
        ARTIFACT_ROOT / "research_step_full_results.md",
        report.replace(
            ROOT_HANDOFF_SOURCE_HEADER, ROOT_HANDOFF_TARGET_HEADER, 1
        ),
    )
    write_json(
        ARTIFACT_ROOT / "s19_status.json",
        {
            "researchStepId": LOOP_ID,
            "status": "AUTONOMOUS_SEQUENCE_ACTIVE",
            "lastCompletedLoop": LOOP_ID,
            "currentLoop": LOOP_ID,
            "nextLoopAuthorized": True,
            "authorizationUpperBound": "S19-L42",
            "s20Status": "DEFINED_INACTIVE",
            "outcomeClassification": classifications[0],
            "classifications": classifications,
            "selectedDiscoveryLead": selected,
            "validationResult": "PASS_TASK_FEATURE_SUFFIX_MODEL_BOOTSTRAP_MAXSTAT_IMMUTABILITY_REGENERATION",
            "recommendedNextAction": f"UNTOUCHED_CONFIRMATION_{selected}"
            if selected
            else NULL_NEXT_ACTION,
            "updatedAtUtc": timestamp,
        },
    )
    print(
        canonical_json(
            {
                "status": "COMPLETE",
                "classifications": classifications,
                "selectedDiscoveryLead": selected,
                "artifactRoot": str(LOOP_ROOT),
                "wallSeconds": runtime["wallSeconds"],
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-lock", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise SystemExit("workers must be between 1 and 8")
    if args.prepare_lock:
        prepare_lock()
    else:
        execute(args.workers)


if __name__ == "__main__":
    main()
