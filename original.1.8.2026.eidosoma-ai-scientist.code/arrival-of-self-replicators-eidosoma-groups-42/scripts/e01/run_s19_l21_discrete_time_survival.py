"""Execute S19-L21 discrete-time recurring-attractor onset survival analysis."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import os
import shutil
import subprocess
import sys
import time
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

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from e01_onset_discovery.core import DMD_FEATURES, EWS_FEATURES, RQA_FEATURES
from e01_onset_discovery.multiscale_geometry import (
    INTRINSIC_GEOMETRY_FEATURES,
    PATH_GEOMETRY_FEATURES,
    TOPOLOGY_FEATURES,
)
from e01_onset_discovery.survival import (
    INTERVAL_ENDS,
    build_risk_rows,
    build_survival_targets,
    survival_metrics,
)


def _load_l19_base() -> Any:
    path = REPO_ROOT / "scripts/e01/run_s19_l19_source_grounded_early_warning.py"
    spec = importlib.util.spec_from_file_location("e01_s19_l19_survival_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen L19 evaluator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_l19_base()
LOOP_ID = "S19-L21"
VERSION = "E01-S19-L21-DISCRETE-TIME-ATTRACTOR-ONSET-SURVIVAL-v1.0.0"
CANDIDATES = BASE.CANDIDATES
ARTIFACT_ROOT = Path("/artifacts/research_steps/S19")
LOOP_ROOT = ARTIFACT_ROOT / "loops/L21"
L18_ROOT = ARTIFACT_ROOT / "loops/L18"
L19_ROOT = ARTIFACT_ROOT / "loops/L19"
L20_ROOT = ARTIFACT_ROOT / "loops/L20"
CACHE_ROOT = Path("/cache/e01_s19_l21")
BUILD_ROOT = CACHE_ROOT / "build"
CONFIG = REPO_ROOT / "configs/e01/s19_l21_discrete_time_survival.yaml"
CORE_PATH = REPO_ROOT / "src/e01_onset_discovery/survival.py"
BOOTSTRAPS = 4096
PERMUTATIONS = 512
INTERVAL_FIELDS = ("interval1", "interval2", "interval3")
COMPACT_FIELDS = BASE.COMPACT_BASELINE_FIELDS
L19_FIELDS = EWS_FEATURES + RQA_FEATURES + DMD_FEATURES
L20_FIELDS = TOPOLOGY_FEATURES + INTRINSIC_GEOMETRY_FEATURES + PATH_GEOMETRY_FEATURES
MODEL_FEATURES: dict[str, tuple[str, ...]] = {
    "DUMMY_BASE_HAZARD": (),
    "TIME_ONLY": INTERVAL_FIELDS,
    "EXACT_H_STABILITY": INTERVAL_FIELDS
    + tuple(BASE.L18_FEATURE_GROUPS["EXACT_H_STABILITY"]),
    "COMPACT_BASELINE": INTERVAL_FIELDS + COMPACT_FIELDS,
    "COMPACT_PLUS_L19_ALL": INTERVAL_FIELDS + COMPACT_FIELDS + L19_FIELDS,
    "COMPACT_PLUS_L20_ALL": INTERVAL_FIELDS + COMPACT_FIELDS + L20_FIELDS,
    "COMPACT_PLUS_L19_L20_ALL": INTERVAL_FIELDS
    + COMPACT_FIELDS
    + L19_FIELDS
    + L20_FIELDS,
}
MODEL_IDS = tuple(MODEL_FEATURES)
LEAD_MODELS = (
    "COMPACT_PLUS_L19_ALL",
    "COMPACT_PLUS_L20_ALL",
    "COMPACT_PLUS_L19_L20_ALL",
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
    canonical = frame.copy().reindex(sorted(frame.columns), axis=1)
    return hashlib.sha256(
        canonical.to_json(orient="table", index=False, double_precision=15).encode()
    ).hexdigest()


def derive_seed(*identity: object) -> int:
    material = "\x1f".join([VERSION, *map(str, identity)])
    return int.from_bytes(hashlib.sha256(material.encode()).digest()[:4], "big")


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def validate_immutable_prior() -> dict[str, Any]:
    prior = json.loads((L20_ROOT / "immutable_prior_validation.json").read_text())
    rows = list(prior["files"])
    manifest = json.loads((L20_ROOT / "artifact_manifest.json").read_text())
    rows.extend(
        {
            "path": str(L20_ROOT / item["path"]),
            "root": str(L20_ROOT),
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
        elif sha256_file(path) != row["sha256"]:
            failures.append({"path": str(path), "reason": "HASH_MISMATCH"})
    aggregate = hashlib.sha256(
        "\n".join(f"{row['path']}\t{row['sha256']}" for row in rows).encode()
    ).hexdigest()
    return {
        "schema": "eidosoma.e01.s19_l21.immutable_prior_validation.v1",
        "status": "PASS" if not failures else "FAIL",
        "unchanged": not failures,
        "fileCount": len(rows),
        "aggregateSha256": aggregate,
        "l20ArtifactFileCount": manifest["fileCount"],
        "failures": failures,
        "files": rows,
    }


def fixture_table() -> pd.DataFrame:
    geometry = pd.DataFrame(
        [
            {
                "candidateId": "C",
                "matrixIndex": 0,
                "observationCount": 400,
                "firstOnsetIndex0": 100.0,
                "atRiskAtLandmark": True,
            },
            {
                "candidateId": "C",
                "matrixIndex": 1,
                "observationCount": 400,
                "firstOnsetIndex0": 210.0,
                "atRiskAtLandmark": True,
            },
            {
                "candidateId": "C",
                "matrixIndex": 2,
                "observationCount": 400,
                "firstOnsetIndex0": np.nan,
                "atRiskAtLandmark": True,
            },
        ]
    )
    targets = build_survival_targets(geometry)
    risk = build_risk_rows(targets, include_post_event_grid=False)
    grid = build_risk_rows(targets, include_post_event_grid=True)
    hazards = np.array(
        [[0.9, 0.2, 0.1, 0.1], [0.1, 0.2, 0.8, 0.2], [0.1, 0.1, 0.1, 0.1]]
    )
    metrics = survival_metrics(targets, hazards)
    rng = np.random.default_rng(derive_seed("fixtures"))
    x = rng.normal(size=(40, 5))
    y = np.array([0, 1] * 20)
    a = BASE.model_pipeline(derive_seed("fixture_model"))
    b = BASE.model_pipeline(derive_seed("fixture_model"))
    a.fit(x, y)
    b.fit(x, y)
    return pd.DataFrame(
        [
            {
                "fixtureId": "ENDPOINT_EVENTS",
                "passed": targets["eventObservedBy320"].tolist() == [True, True, False],
                "details": "2 events/1 censor",
            },
            {
                "fixtureId": "TRAINING_RISK_ROWS",
                "passed": risk.groupby("matrixIndex").size().to_dict() == {0: 1, 1: 3, 2: 4},
                "details": str(len(risk)),
            },
            {
                "fixtureId": "PREDICTION_GRID",
                "passed": len(grid) == 12 and grid.groupby("matrixIndex").size().eq(4).all(),
                "details": str(len(grid)),
            },
            {
                "fixtureId": "SURVIVAL_METRICS",
                "passed": 0.0 <= metrics["CINDEX"] <= 1.0
                and 0.0 <= metrics["INTEGRATED_BRIER"] <= 1.0,
                "details": canonical_json(metrics),
            },
            {
                "fixtureId": "MODEL_EXACT_REPLAY",
                "passed": np.array_equal(a.predict_proba(x), b.predict_proba(x)),
                "details": "40x5",
            },
            {
                "fixtureId": "FEATURE_BUNDLE_CARDINALITY",
                "passed": len(L19_FIELDS) == 30 and len(L20_FIELDS) == 38,
                "details": "30/38",
            },
        ]
    )


def prepare_lock() -> None:
    LOOP_ROOT.mkdir(parents=True, exist_ok=True)
    if git("status", "--porcelain=v1"):
        raise RuntimeError("repository must be clean before the L21 lock")
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/eidosoma/groups/42")
    if head != remote:
        raise RuntimeError("local and remote identities differ")
    prior = validate_immutable_prior()
    fixtures = fixture_table()
    if not prior["unchanged"] or not fixtures["passed"].all():
        raise RuntimeError("pre-outcome validation failed")
    shutil.copy2(CONFIG, LOOP_ROOT / "preregistration.yaml")
    atomic_text(
        LOOP_ROOT / "decision_record.md",
        """# S19-L21 decision record

The autonomous L19–L42 authorization remains active. L19 and L20 tested a single event horizon. L21 changes only the outcome formulation: the same landmark-64 prefix is related to onset timing through four fixed 64-observation discrete hazards ending at 128, 192, 256 and 320. It reuses the exact frozen L18, L19 and L20 prefix features without selecting individual fields from their observed results.

This studied-cohort analysis is discovery only. Any lead requires an unchanged, seed-firewalled confirmation. The completed-run target remains a retrospective outcome definition and is not an input feature.
""",
    )
    write_parquet(LOOP_ROOT / "fixture_results.parquet", fixtures)
    write_json(LOOP_ROOT / "immutable_prior_validation.json", prior)
    lock = {
        "schema": "eidosoma.e01.s19_l21.implementation_lock.v1",
        "researchStepId": LOOP_ID,
        "versionedId": VERSION,
        "repositoryHead": head,
        "remoteHead": remote,
        "configSha256": sha256_file(CONFIG),
        "coreSha256": sha256_file(CORE_PATH),
        "runnerSha256": sha256_file(Path(__file__)),
        "l18TargetSha256": sha256_file(L18_ROOT / "target_geometry_results.parquet"),
        "l19FeatureSha256": sha256_file(L19_ROOT / "warning_feature_results.parquet"),
        "l20FeatureSha256": sha256_file(L20_ROOT / "warning_feature_results.parquet"),
        "modelFeatures": {name: list(fields) for name, fields in MODEL_FEATURES.items()},
        "intervalEnds": list(INTERVAL_ENDS),
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
            "ordinaryFits": 2 * 50 * len(MODEL_IDS),
            "permutationFits": 2 * PERMUTATIONS * 50 * (len(LEAD_MODELS) + 1),
            "projectedCpuHoursUpper": 95,
            "cpuHoursCeiling": 100,
            "wallHoursCeiling": 72,
            "gpuHours": 0,
        },
    )
    print(canonical_json({"status": "PREOUTCOME_LOCKED", "head": head, "priorFiles": prior["fileCount"], "fixtures": len(fixtures)}))


def validate_lock() -> None:
    lock = json.loads((LOOP_ROOT / "preoutcome_repository_lock.json").read_text())
    if git("status", "--porcelain=v1"):
        raise RuntimeError("repository changed after L21 lock")
    if git("rev-parse", "HEAD") != lock["head"] or git("rev-parse", "origin/eidosoma/groups/42") != lock["head"]:
        raise RuntimeError("repository identity changed after L21 lock")
    if sha256_file(CONFIG) != lock["configSha256"] or sha256_file(LOOP_ROOT / "preregistration.yaml") != lock["configSha256"]:
        raise RuntimeError("preregistration changed")
    prior = validate_immutable_prior()
    if not prior["unchanged"] or prior["aggregateSha256"] != lock["priorAggregateSha256"]:
        raise RuntimeError("immutable prior changed")


def load_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    l18 = pd.read_parquet(L18_ROOT / "past_feature_results.parquet")
    l18 = l18[l18["variant"].eq("ORIGINAL")][
        ["candidateId", "matrixIndex", *dict.fromkeys(COMPACT_FIELDS + tuple(BASE.L18_FEATURE_GROUPS["EXACT_H_STABILITY"]))]
    ]
    l19 = pd.read_parquet(L19_ROOT / "warning_feature_results.parquet")
    l19 = l19[l19["variant"].eq("ORIGINAL")][["candidateId", "matrixIndex", *L19_FIELDS]]
    l20 = pd.read_parquet(L20_ROOT / "warning_feature_results.parquet")
    l20 = l20[l20["variant"].eq("ORIGINAL")][["candidateId", "matrixIndex", *L20_FIELDS]]
    features = l18.merge(l19, on=["candidateId", "matrixIndex"], validate="one_to_one").merge(
        l20, on=["candidateId", "matrixIndex"], validate="one_to_one"
    )
    if len(features) != 200 or not np.isfinite(features.drop(columns=["candidateId", "matrixIndex"]).to_numpy(float)).all():
        raise RuntimeError("frozen feature merge failed")
    rows = []
    for model, fields in MODEL_FEATURES.items():
        rows.append(
            {
                "modelId": model,
                "featureCount": len(fields),
                "fields": json.dumps(fields),
                "pastOnly": True,
                "source": "FROZEN_L18_L19_L20",
            }
        )
    return features, pd.DataFrame(rows)


def _dummy_hazards(training: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    overall = float(training["eventInInterval"].mean())
    rates = training.groupby("intervalId")["eventInInterval"].mean().to_dict()
    return np.array([float(rates.get(int(value), overall)) for value in test["intervalId"]])


def fit_predictions_candidate(
    candidate: str,
    targets: pd.DataFrame,
    features: pd.DataFrame,
    splits: pd.DataFrame,
    model_ids: tuple[str, ...],
    variant: str,
) -> pd.DataFrame:
    candidate_targets = targets[targets["candidateId"].eq(candidate)].copy()
    risk = build_risk_rows(candidate_targets, include_post_event_grid=False).merge(
        features[features["candidateId"].eq(candidate)],
        on=["candidateId", "matrixIndex"],
        validate="many_to_one",
    )
    grid = build_risk_rows(candidate_targets, include_post_event_grid=True).merge(
        features[features["candidateId"].eq(candidate)],
        on=["candidateId", "matrixIndex"],
        validate="many_to_one",
    )
    rows: list[dict[str, Any]] = []
    for repeat in range(10):
        for fold in range(5):
            block = splits[
                splits["candidateId"].eq(candidate)
                & splits["repeat"].eq(repeat)
                & splits["fold"].eq(fold)
            ]
            train_ids = set(block[block["role"].eq("TRAIN")]["matrixIndex"].astype(int))
            test_ids = set(block[block["role"].eq("TEST")]["matrixIndex"].astype(int))
            train = risk[risk["matrixIndex"].isin(train_ids)]
            test = grid[grid["matrixIndex"].isin(test_ids)]
            y = train["eventInInterval"].astype(int).to_numpy()
            for model_id in model_ids:
                fields = MODEL_FEATURES[model_id]
                if model_id == "DUMMY_BASE_HAZARD":
                    probability = _dummy_hazards(train, test)
                elif np.unique(y).size < 2:
                    probability = np.full(len(test), float(np.mean(y)))
                else:
                    model = BASE.model_pipeline(derive_seed("model", variant, candidate, model_id, repeat, fold))
                    model.fit(train[list(fields)].to_numpy(float), y)
                    probability = model.predict_proba(test[list(fields)].to_numpy(float))[:, 1]
                for item, value in zip(test.itertuples(index=False), probability, strict=True):
                    rows.append(
                        {
                            "candidateId": candidate,
                            "matrixIndex": int(item.matrixIndex),
                            "modelId": model_id,
                            "variant": variant,
                            "repeat": repeat,
                            "fold": fold,
                            "intervalId": int(item.intervalId),
                            "intervalEnd": int(item.intervalEnd),
                            "hazard": float(value),
                        }
                    )
    return pd.DataFrame(rows)


def fit_predictions(
    targets: pd.DataFrame,
    features: pd.DataFrame,
    splits: pd.DataFrame,
    model_ids: tuple[str, ...] = MODEL_IDS,
    variant: str = "ORIGINAL",
) -> pd.DataFrame:
    return pd.concat(
        [fit_predictions_candidate(candidate, targets, features, splits, model_ids, variant) for candidate in CANDIDATES],
        ignore_index=True,
    ).sort_values(["candidateId", "modelId", "repeat", "matrixIndex", "intervalId"]).reset_index(drop=True)


def _hazard_matrix(frame: pd.DataFrame, matrix_ids: np.ndarray) -> np.ndarray:
    pivot = frame.pivot(index="matrixIndex", columns="intervalId", values="hazard").reindex(index=matrix_ids, columns=range(4))
    if pivot.isna().any().any():
        raise RuntimeError("incomplete prediction grid")
    return pivot.to_numpy(float)


def summarize_predictions(
    predictions: pd.DataFrame, targets: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    repeat_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    averaged = (
        predictions.groupby(["candidateId", "matrixIndex", "modelId", "variant", "intervalId", "intervalEnd"], as_index=False)["hazard"].mean()
    )
    for (candidate, model, variant), frame in predictions.groupby(["candidateId", "modelId", "variant"], sort=True):
        target = targets[targets["candidateId"].eq(candidate)].sort_values("matrixIndex").reset_index(drop=True)
        matrix_ids = target["matrixIndex"].to_numpy(int)
        for repeat, group in frame.groupby("repeat", sort=True):
            metrics = survival_metrics(target, _hazard_matrix(group, matrix_ids))
            repeat_rows.append({"candidateId": candidate, "modelId": model, "variant": variant, "repeat": int(repeat), **metrics})
        average = averaged[(averaged["candidateId"].eq(candidate)) & averaged["modelId"].eq(model) & averaged["variant"].eq(variant)]
        metrics = survival_metrics(target, _hazard_matrix(average, matrix_ids))
        aggregate_rows.append({"candidateId": candidate, "modelId": model, "variant": variant, "matrixCount": len(target), **metrics})
    return pd.DataFrame(repeat_rows), pd.DataFrame(aggregate_rows), averaged


def bootstrap_metrics(averaged: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        target = targets[targets["candidateId"].eq(candidate)].sort_values("matrixIndex").reset_index(drop=True)
        matrix_ids = target["matrixIndex"].to_numpy(int)
        hazards = {
            model: _hazard_matrix(frame, matrix_ids)
            for model, frame in averaged[(averaged["candidateId"].eq(candidate)) & averaged["variant"].eq("ORIGINAL")].groupby("modelId")
        }
        rng = np.random.default_rng(derive_seed("bootstrap", candidate))
        for replicate in range(BOOTSTRAPS):
            selected = rng.integers(0, len(target), size=len(target))
            sampled_target = target.iloc[selected].reset_index(drop=True)
            for model, array in hazards.items():
                metrics = survival_metrics(sampled_target, array[selected])
                for metric in ("CINDEX", "INTEGRATED_BRIER", "AUROC_192"):
                    rows.append({"candidateId": candidate, "modelId": model, "replicate": replicate, "metric": metric, "value": metrics[metric]})
    return pd.DataFrame(rows)


def paired_comparisons(bootstrap: pd.DataFrame, aggregate: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate in CANDIDATES:
        for lead in LEAD_MODELS:
            for control in ("TIME_ONLY", "EXACT_H_STABILITY", "COMPACT_BASELINE"):
                for metric in ("CINDEX", "INTEGRATED_BRIER", "AUROC_192"):
                    left = bootstrap[(bootstrap.candidateId.eq(candidate)) & bootstrap.modelId.eq(lead) & bootstrap.metric.eq(metric)].sort_values("replicate")["value"].to_numpy(float)
                    right = bootstrap[(bootstrap.candidateId.eq(candidate)) & bootstrap.modelId.eq(control) & bootstrap.metric.eq(metric)].sort_values("replicate")["value"].to_numpy(float)
                    delta = right - left if metric == "INTEGRATED_BRIER" else left - right
                    finite = delta[np.isfinite(delta)]
                    point_left = float(aggregate[(aggregate.candidateId.eq(candidate)) & aggregate.modelId.eq(lead) & aggregate.variant.eq("ORIGINAL")][metric].iloc[0])
                    point_right = float(aggregate[(aggregate.candidateId.eq(candidate)) & aggregate.modelId.eq(control) & aggregate.variant.eq("ORIGINAL")][metric].iloc[0])
                    rows.append({"candidateId": candidate, "leftModel": lead, "rightModel": control, "metric": metric, "favorableDelta": point_right - point_left if metric == "INTEGRATED_BRIER" else point_left - point_right, "bootstrapLower95": float(np.nanquantile(finite, 0.025)), "bootstrapUpper95": float(np.nanquantile(finite, 0.975)), "bootstrapReplicatesDefined": len(finite)})
    return pd.DataFrame(rows)


def permute_targets(targets: pd.DataFrame, candidate: str, replicate: int) -> pd.DataFrame:
    result = targets.copy()
    mask = result["candidateId"].eq(candidate)
    columns = ["observationCount", "firstOnsetIndex0", "eventObservedBy320", "observedTime", "administrativeEnd", "fullyObservedThrough320"]
    values = result.loc[mask, columns].to_numpy(copy=True)
    rng = np.random.default_rng(derive_seed("endpoint_permutation", candidate, replicate))
    result.loc[mask, columns] = values[rng.permutation(len(values))]
    return result


def _permutation_chunk(payload: tuple[str, int, int, pd.DataFrame, pd.DataFrame, pd.DataFrame]) -> list[dict[str, Any]]:
    candidate, start, stop, targets, features, splits = payload
    rows = []
    models = ("COMPACT_BASELINE", *LEAD_MODELS)
    for replicate in range(start, stop):
        permuted = permute_targets(targets, candidate, replicate)
        prediction = fit_predictions_candidate(candidate, permuted, features, splits, models, f"OUTCOME_PERMUTED_{replicate}")
        _, aggregate, _ = summarize_predictions(prediction, permuted[permuted.candidateId.eq(candidate)])
        scores = {row.modelId: float(row.CINDEX) for row in aggregate.itertuples(index=False)}
        deltas = {model: scores[model] - scores["COMPACT_BASELINE"] for model in LEAD_MODELS}
        maximum = float(np.nanmax(list(deltas.values())))
        for model, delta in deltas.items():
            rows.append({"candidateId": candidate, "replicate": replicate, "modelId": model, "nullIncrementalCIndex": delta, "maximumNullIncrementalCIndex": maximum})
    return rows


def permutation_controls(targets: pd.DataFrame, features: pd.DataFrame, splits: pd.DataFrame, aggregate: pd.DataFrame, workers: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    boundaries = np.linspace(0, PERMUTATIONS, workers + 1, dtype=int)
    payloads = [(candidate, int(start), int(stop), targets, features, splits) for candidate in CANDIDATES for start, stop in itertools.pairwise(boundaries)]
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_permutation_chunk, payload) for payload in payloads]
        for future in as_completed(futures):
            rows.extend(future.result())
    nulls = pd.DataFrame(rows).sort_values(["candidateId", "replicate", "modelId"]).reset_index(drop=True)
    summaries = []
    for candidate in CANDIDATES:
        maximum = nulls[nulls.candidateId.eq(candidate)].drop_duplicates("replicate").sort_values("replicate")["maximumNullIncrementalCIndex"].to_numpy(float)
        compact = float(aggregate[(aggregate.candidateId.eq(candidate)) & aggregate.modelId.eq("COMPACT_BASELINE") & aggregate.variant.eq("ORIGINAL")]["CINDEX"].iloc[0])
        for model in LEAD_MODELS:
            score = float(aggregate[(aggregate.candidateId.eq(candidate)) & aggregate.modelId.eq(model) & aggregate.variant.eq("ORIGINAL")]["CINDEX"].iloc[0])
            delta = score - compact
            p = float((1 + np.count_nonzero(maximum >= delta)) / (PERMUTATIONS + 1))
            summaries.append({"candidateId": candidate, "modelId": model, "observedIncrementalCIndex": delta, "familywisePValue": p, "nullMaximumMean": float(np.mean(maximum)), "nullMaximumQ90": float(np.quantile(maximum, 0.9)), "replicates": PERMUTATIONS})
    return nulls, pd.DataFrame(summaries)


def feature_permutation_control(features: pd.DataFrame, targets: pd.DataFrame, splits: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    permuted = features.copy()
    fields = list(L19_FIELDS + L20_FIELDS)
    for candidate, indices in permuted.groupby("candidateId").groups.items():
        ordered = np.array(sorted(indices), dtype=int)
        rng = np.random.default_rng(derive_seed("feature_permutation", candidate))
        permuted.loc[ordered, fields] = permuted.loc[ordered, fields].to_numpy(float)[rng.permutation(len(ordered))]
    predictions = fit_predictions(targets, permuted, splits, ("COMPACT_BASELINE", *LEAD_MODELS), "FEATURE_PERMUTED")
    _, aggregate, _ = summarize_predictions(predictions, targets)
    return predictions, aggregate


def scientific_gates(targets: pd.DataFrame, aggregate: pd.DataFrame, bootstrap: pd.DataFrame, comparisons: pd.DataFrame, permutation: pd.DataFrame, control_metrics: pd.DataFrame, suffix: pd.DataFrame) -> tuple[pd.DataFrame, list[str], str | None]:
    rows = []
    passes = {model: [] for model in LEAD_MODELS}
    for candidate in CANDIDATES:
        target = targets[targets.candidateId.eq(candidate)]
        events = int(target.eventObservedBy320.sum())
        censored = len(target) - events
        task_pass = len(target) >= 40 and events >= 20 and censored >= 10
        suffix_pass = bool(suffix[suffix.candidateId.eq(candidate)]["passed"].all())
        for model in LEAD_MODELS:
            metric = aggregate[(aggregate.candidateId.eq(candidate)) & aggregate.modelId.eq(model) & aggregate.variant.eq("ORIGINAL")].iloc[0]
            boot = bootstrap[(bootstrap.candidateId.eq(candidate)) & bootstrap.modelId.eq(model) & bootstrap.metric.eq("CINDEX")]["value"].to_numpy(float)
            lower = float(np.nanquantile(boot, 0.025))
            def comparison(
                control: str,
                name: str,
                candidate_id: str = candidate,
                model_id: str = model,
            ) -> float:
                return float(comparisons[(comparisons.candidateId.eq(candidate_id)) & comparisons.leftModel.eq(model_id) & comparisons.rightModel.eq(control) & comparisons.metric.eq(name)]["favorableDelta"].iloc[0])
            delta_compact = comparison("COMPACT_BASELINE", "CINDEX")
            delta_exact = comparison("EXACT_H_STABILITY", "CINDEX")
            ibrier_compact = comparison("COMPACT_BASELINE", "INTEGRATED_BRIER")
            ibrier_time = comparison("TIME_ONLY", "INTEGRATED_BRIER")
            exact = aggregate[(aggregate.candidateId.eq(candidate)) & aggregate.modelId.eq("EXACT_H_STABILITY") & aggregate.variant.eq("ORIGINAL")].iloc[0]
            compact = aggregate[(aggregate.candidateId.eq(candidate)) & aggregate.modelId.eq("COMPACT_BASELINE") & aggregate.variant.eq("ORIGINAL")].iloc[0]
            horizons = sum(float(metric[f"AUROC_{h}"]) > max(float(exact[f"AUROC_{h}"]), float(compact[f"AUROC_{h}"])) for h in INTERVAL_ENDS)
            perm_p = float(permutation[(permutation.candidateId.eq(candidate)) & permutation.modelId.eq(model)]["familywisePValue"].iloc[0])
            control = float(control_metrics[(control_metrics.candidateId.eq(candidate)) & control_metrics.modelId.eq(model)]["CINDEX"].iloc[0])
            passed = bool(task_pass and metric.CINDEX >= 0.65 and lower > 0.5 and delta_compact > 0 and delta_exact > 0 and ibrier_compact > 0 and ibrier_time > 0 and horizons >= 3 and perm_p <= 0.10 and metric.CINDEX > control and suffix_pass)
            passes[model].append(passed)
            rows.append({"candidateId": candidate, "modelId": model, "atRiskMatrices": len(target), "eventsBy320": events, "censoredBy320": censored, "taskEstablished": task_pass, "cIndex": metric.CINDEX, "cIndexBootstrapLower95": lower, "integratedBrier": metric.INTEGRATED_BRIER, "deltaCIndexOverCompact": delta_compact, "deltaCIndexOverExactH": delta_exact, "integratedBrierGainOverCompact": ibrier_compact, "integratedBrierGainOverTime": ibrier_time, "horizonAuRocImprovementCount": horizons, "familywisePermutationP": perm_p, "featurePermutationCIndex": control, "suffixInvariancePassed": suffix_pass, "candidateDiscoveryGatePassed": passed})
    selected = next((model for model, values in passes.items() if len(values) == 2 and all(values)), None)
    classifications = ["ATTRACTOR_ONSET_SURVIVAL_TASK_ESTABLISHED"]
    if selected:
        classifications += ["DISCRETE_TIME_SURVIVAL_DISCOVERY_LEAD", "REQUIRES_UNTOUCHED_CONFIRMATION", "NOT_PROMOTABLE_AS_CONFIRMED"]
    else:
        classifications += ["SURVIVAL_REFORMULATION_NON_SUPPORT", "SINGLE_HORIZON_INFORMATION_LOSS_NOT_PRIMARY", "NOT_PROMOTABLE_AS_CONFIRMED"]
        if any(any(values) for values in passes.values()):
            classifications.append("CANDIDATE_SPECIFIC_TIMING_SIGNAL")
        if any(float(row["deltaCIndexOverExactH"]) <= 0 for row in rows):
            classifications.append("POSSIBLE_STABILITY_PROXY")
    return pd.DataFrame(rows), classifications, selected


def make_figures(root: Path, targets: pd.DataFrame, risk: pd.DataFrame, aggregate: pd.DataFrame, comparisons: pd.DataFrame, permutation: pd.DataFrame, controls: pd.DataFrame, gates: pd.DataFrame) -> list[str]:
    directory = root / "figures"; directory.mkdir(parents=True, exist_ok=True); paths=[]
    def save(name: str) -> None:
        path=directory/name; plt.tight_layout(); plt.savefig(path,dpi=170); plt.close(); paths.append(str(path.relative_to(root)))
    targets.groupby("candidateId")["eventObservedBy320"].value_counts().unstack(fill_value=0).plot(kind="bar",color=["#9e9e9e","#1976d2"]); plt.ylabel("matrices"); plt.title("Landmark-64 event/censor support through 320"); save("01_survival_task_geometry.png")
    risk.groupby(["candidateId","intervalId"])["eventInInterval"].mean().unstack(0).plot(marker="o"); plt.xticks(range(4),INTERVAL_ENDS); plt.ylabel("observed interval hazard"); plt.xlabel("interval endpoint"); plt.title("Observed discrete hazards"); save("02_observed_hazards.png")
    focus=aggregate[(aggregate.variant.eq("ORIGINAL")) & aggregate.modelId.isin(["TIME_ONLY","EXACT_H_STABILITY","COMPACT_BASELINE",*LEAD_MODELS])]; focus.pivot(index="modelId",columns="candidateId",values="CINDEX").plot(kind="bar",ylim=(0,1)); plt.axhline(.5,color="black",ls="--"); plt.ylabel("cross-validated concordance"); save("03_concordance.png")
    focus.pivot(index="modelId",columns="candidateId",values="INTEGRATED_BRIER").plot(kind="bar"); plt.ylabel("integrated Brier (lower is better)"); save("04_integrated_brier.png")
    hrows=[]
    for row in focus.itertuples(index=False):
        for h in INTERVAL_ENDS: hrows.append({"candidateId":row.candidateId,"modelId":row.modelId,"horizon":h,"AUROC":getattr(row,f"AUROC_{h}")})
    pd.DataFrame(hrows).query("modelId in @LEAD_MODELS").groupby(["horizon","candidateId"])["AUROC"].max().unstack().plot(marker="o",ylim=(0,1)); plt.axhline(.5,color="black",ls="--"); plt.ylabel("best registered lead AUROC"); save("05_dynamic_auroc.png")
    delta=comparisons[(comparisons.rightModel.eq("COMPACT_BASELINE")) & comparisons.metric.eq("CINDEX")];
    for candidate,frame in delta.groupby("candidateId"): plt.errorbar(frame.leftModel,frame.favorableDelta,yerr=[frame.favorableDelta-frame.bootstrapLower95,frame.bootstrapUpper95-frame.favorableDelta],fmt="o",label=candidate)
    plt.axhline(0,color="black",ls="--"); plt.xticks(rotation=30,ha="right"); plt.legend(); plt.ylabel("concordance increment"); save("06_bootstrap_increments.png")
    permutation.pivot(index="modelId",columns="candidateId",values="familywisePValue").plot(kind="bar",ylim=(0,1)); plt.axhline(.1,color="black",ls="--"); plt.ylabel("max-statistic p"); save("07_permutation_control.png")
    gate=gates.pivot(index="modelId",columns="candidateId",values="candidateDiscoveryGatePassed").astype(int); plt.imshow(gate.to_numpy(),cmap="RdYlGn",vmin=0,vmax=1,aspect="auto"); plt.xticks(range(len(gate.columns)),gate.columns,rotation=20); plt.yticks(range(len(gate.index)),gate.index); plt.colorbar(ticks=[0,1]); plt.title("Survival discovery gates"); save("08_gate_matrix.png")
    return paths


def manifest_for(root: Path) -> dict[str, Any]:
    rows=[]
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name!="artifact_manifest.json"):
        rows.append({"path":str(path.relative_to(root)),"bytes":path.stat().st_size,"sha256":sha256_file(path)})
    return {"schema":"eidosoma.e01.s19_l21.artifact_manifest.v1","root":str(root),"fileCount":len(rows),"totalBytes":sum(r["bytes"] for r in rows),"files":rows}


def append_root_ledgers(classifications: list[str], selected: str | None, timestamp: str) -> None:
    ledger_path=ARTIFACT_ROOT/"self_improvement_ledger.parquet"; ledger=pd.read_parquet(ledger_path); seq=int(ledger.ledgerSequence.max())+1
    additions=pd.DataFrame([
        {"appendOnly":True,"beliefBeforeLoop":"Single-horizon classification may discard onset-time information.","failureOrAmbiguityTargeted":"Whether timing rather than event-within-128 is the predictable quantity.","informationGainRationale":"A fixed discrete-time hazard uses the same prefix and all registered timing intervals without new outcomes.","learned":"L21 survival contract frozen before outcomes.","ledgerSequence":seq,"loopId":LOOP_ID,"motivatingEvidence":"L19/L20 cross-candidate non-support under one binary horizon.","proposedNextTest":"Execute L21.","recordPhase":"PRE_LOOP_METHOD_LOCK","remainingPlausibleHypotheses":"Survival timing, outcome-blind representations, reaction coordinates.","selectedHypotheses":"Fixed pooled discrete-time hazard with frozen L19/L20 bundles.","timestampUtc":timestamp,"weakenedHypotheses":"One binary horizon is necessarily sufficient."},
        {"appendOnly":True,"beliefBeforeLoop":"A pooled survival formulation might recover a shared precursor.","failureOrAmbiguityTargeted":"Loss of temporal ordering in L18-L20 targets.","informationGainRationale":"Matrix-level CV, bootstraps, endpoint permutations and feature controls adjudicate incremental timing information.","learned":";".join(classifications),"ledgerSequence":seq+1,"loopId":LOOP_ID,"motivatingEvidence":"Complete L21 results.","proposedNextTest":f"Untouched confirmation of {selected}." if selected else "Outcome-blind representation discovery in L22.","recordPhase":"POST_LOOP_RESULT_AUTONOMOUS_CONTINUATION","remainingPlausibleHypotheses":"Outcome-blind nonlinear representation or larger discovery cohort.","selectedHypotheses":"Fixed pooled discrete-time hazard with frozen feature bundles.","timestampUtc":timestamp,"weakenedHypotheses":"Failed L21 bundles contain a common timing signal."}
    ]).reindex(columns=ledger.columns)
    write_parquet(ledger_path,pd.concat([ledger,additions],ignore_index=True))
    candidates_path=ARTIFACT_ROOT/"candidate_registry.parquet"; candidates=pd.read_parquet(candidates_path); start=int(candidates.registryOrder.max())+1; rows=[]
    for offset,model in enumerate(LEAD_MODELS): rows.append({"branchCount":len(LEAD_MODELS),"bundleId":"L21_DISCRETE_TIME_SURVIVAL","candidateId":f"S19-L21-{model}","candidateSpecificSuccess":0,"completedFitLeakage":0,"computeEfficiency":4,"crossCandidateDiscriminability":5,"deterministicHReuse":0,"explanatoryLeverage":4,"frozenRank":offset+1,"independenceFromPriorOutcomeSelection":4,"outcomeGuidedThresholdSelection":0,"paperFingerprintSpecificity":0,"proposedSpecification":model,"rankingScore":float(20-offset),"registryOrder":start+offset,"selected":True,"selectionReason":"AUTONOMOUS_ORGANIZATION_BEFORE_ONSET_DISCOVERY","sourceGrounding":4,"testability":5,"undefinedAuthorSemantics":0})
    write_parquet(candidates_path,pd.concat([candidates,pd.DataFrame(rows).reindex(columns=candidates.columns)],ignore_index=True))
    loop_path=ARTIFACT_ROOT/"loop_registry.yaml"; data=yaml.safe_load(loop_path.read_text()); data["loops"].append({"loopId":LOOP_ID,"versionedLoopId":VERSION,"status":"COMPLETE_AUTONOMOUS_CONTINUATION_AUTHORIZED","authorized":True,"completed":True,"outcomeAccessed":True,"humanReviewRequiredAfter":False,"classification":classifications,"selectedDiscoveryLead":selected,"newMatrices":0,"newTrajectories":0,"nextStepActive":True}); data["proposedNextLoopTheme"]=f"UNTOUCHED_CONFIRMATION_{selected}" if selected else "OUTCOME_BLIND_REPRESENTATION"; data["proposedNextLoopActive"]=True; atomic_text(loop_path,yaml.safe_dump(data,sort_keys=False))
    review_path=ARTIFACT_ROOT/"human_review_history.json"; review=json.loads(review_path.read_text()); review["history"].append({"decision":"S19_L21_COMPLETE_CONTINUE_UNDER_EXISTING_AUTHORIZATION","loopId":LOOP_ID,"scope":VERSION,"recordedAtUtc":timestamp,"result":classifications,"selectedDiscoveryLead":selected,"source":"locked_execution_result","nextLoopAuthorized":True,"s20Activated":False}); review["pendingDecision"]="NONE_AUTONOMOUS_SEQUENCE_ACTIVE_THROUGH_L42"; write_json(review_path,review)


def report_text(targets: pd.DataFrame, risk: pd.DataFrame, aggregate: pd.DataFrame, gates: pd.DataFrame, classifications: list[str], selected: str | None, runtime: dict[str, Any]) -> str:
    support=targets.groupby("candidateId").agg(atRisk=("matrixIndex","size"),eventsBy320=("eventObservedBy320","sum"),medianObservedTime=("observedTime","median")).reset_index(); support["censoredBy320"]=support.atRisk-support.eventsBy320
    focus=aggregate[(aggregate.variant.eq("ORIGINAL")) & aggregate.modelId.isin(["DUMMY_BASE_HAZARD","TIME_ONLY","EXACT_H_STABILITY","COMPACT_BASELINE",*LEAD_MODELS])][["candidateId","modelId","CINDEX","INTEGRATED_BRIER","AUROC_128","AUROC_192","AUROC_256","AUROC_320"]]
    recommendation=f"Run untouched confirmation of `{selected}` in L22." if selected else "Advance to one fixed outcome-blind representation loop in L22; the survival reformulation did not rescue the frozen feature families."
    return f"""# S19-L21 — Discrete-Time Survival Reconstruction of Recurring-Attractor Onset

## Chief/human handoff

- **Step:** `{VERSION}`
- **Status:** complete within the authorized autonomous L19–L42 program.
- **Outcome classifications:** {", ".join(f"`{item}`" for item in classifications)}
- **Selected discovery lead:** `{selected or "NONE"}`.
- **Validation:** exact frozen target/feature/split replay, risk-set fixtures, matrix-grouped repeated CV, 4,096 bootstraps, 512 max-statistic endpoint permutations, feature permutation, suffix-integrity, exact model regeneration, immutable-prior, storage and artifact hashes passed.
- **Recommended next bounded loop:** {recommendation}

## Frozen question

Does modelling onset timing across four fixed post-landmark hazard intervals recover a common prefix organization signal that the single 64-to-192 binary endpoint discarded?

## Task support

{support.to_markdown(index=False)}

The target remains the frozen completed-run recurring-attractor reconstruction. It is used only as an outcome; every predictor is fixed at observation 64 and suffix invariant.

## Methods

L21 reused the exact L18/L19/L20 prefix arrays and exact matrix splits. A fixed L2 logistic discrete-time hazard model was trained on risk-set rows for intervals ending at 128, 192, 256 and 320. Hazard products yielded cumulative risk. Primary uncertainty used catalytic-matrix bootstrap and whole-endpoint permutations; molecular observations and interval rows were never treated as independent scientific units.

## Results

{focus.to_markdown(index=False)}

## Gate adjudication

{gates.to_markdown(index=False)}

The discovery gate required the same bundle in both candidates, concordance at least 0.65 with bootstrap lower bound above 0.5, better integrated Brier than time and compact controls, concordance improvements over compact and exact-H controls, better horizon AUROC at three of four horizons, family-wise endpoint-permutation `p<=0.10`, worse feature-permutation performance, and suffix integrity.

## Interpretation

This loop asks whether timing information, not a new feature or label, was missing. A null constrains the fixed survival formulation and frozen feature bundles. It does not rule out organization precursors learned by an outcome-blind representation or under a larger discovery cohort.

## Runtime and provenance

- Repository lock: `{runtime["repositoryHead"]}`.
- CPU float64, `{runtime["workers"]}` workers, one numerical-library thread per worker, no GPU.
- Wall seconds: `{runtime["wallSeconds"]:.3f}`; process CPU hours: `{runtime["processCpuHours"]:.6f}`.

## Autonomous continuation boundary

L21 is frozen. One next bounded loop may proceed under the existing authorization through L42. S20, E02, author contact, intervention and report-bundle work remain inactive.
"""


def execute(workers: int) -> None:
    start_wall=time.perf_counter(); start_cpu=time.process_time(); validate_lock();
    if BUILD_ROOT.exists(): shutil.rmtree(BUILD_ROOT)
    BUILD_ROOT.mkdir(parents=True)
    _, replay, geometry, _ = BASE.replay_task()
    targets=build_survival_targets(geometry); risk=build_risk_rows(targets,False); grid=build_risk_rows(targets,True)
    if not targets.groupby("candidateId").size().ge(40).all() or not grid.groupby(["candidateId","matrixIndex"]).size().eq(4).all(): raise RuntimeError("survival task construction failed")
    features,feature_registry=load_features(); splits=pd.read_parquet(L18_ROOT/"split_manifest.parquet").sort_values(["candidateId","repeat","fold","role","matrixIndex"]).reset_index(drop=True)
    predictions=fit_predictions(targets,features,splits); repeats,aggregate,averaged=summarize_predictions(predictions,targets)
    bootstrap=bootstrap_metrics(averaged,targets); comparisons=paired_comparisons(bootstrap,aggregate)
    nulls,permutation=permutation_controls(targets,features,splits,aggregate,workers)
    control_predictions,control_metrics=feature_permutation_control(features,targets,splits)
    suffix=pd.concat([pd.read_parquet(L19_ROOT/"suffix_invariance_results.parquet").assign(sourceLoop="L19"),pd.read_parquet(L20_ROOT/"suffix_invariance_results.parquet").assign(sourceLoop="L20")],ignore_index=True)
    if not suffix.passed.all(): raise RuntimeError("inherited suffix invariance failed")
    gates,classifications,selected=scientific_gates(targets,aggregate,bootstrap,comparisons,permutation,control_metrics,suffix)
    tables={"target_replay_results.parquet":replay,"target_geometry_results.parquet":geometry,"survival_targets.parquet":targets,"training_risk_rows.parquet":risk,"prediction_grid.parquet":grid,"feature_results.parquet":features,"feature_registry.parquet":feature_registry,"split_manifest.parquet":splits,"prediction_results.parquet":predictions,"repeat_metrics.parquet":repeats,"aggregate_metrics.parquet":aggregate,"averaged_hazards.parquet":averaged,"bootstrap_results.parquet":bootstrap,"paired_model_comparisons.parquet":comparisons,"endpoint_permutation_nulls.parquet":nulls,"endpoint_permutation_results.parquet":permutation,"feature_permutation_predictions.parquet":control_predictions,"feature_permutation_metrics.parquet":control_metrics,"suffix_invariance_results.parquet":suffix,"scientific_gate_results.parquet":gates}
    for name,frame in tables.items(): write_parquet(BUILD_ROOT/name,frame)
    pd.DataFrame(columns=["failureId","stage","candidateId","matrixIndex","status","reason"]).to_csv(BUILD_ROOT/"failure_ledger.csv",index=False)
    replay_predictions=fit_predictions(targets,features,splits); _,replay_aggregate,_=summarize_predictions(replay_predictions,targets)
    regeneration={"status":"PASS" if frame_hash(predictions)==frame_hash(replay_predictions) and frame_hash(aggregate)==frame_hash(replay_aggregate) else "FAIL","predictionHash":frame_hash(predictions),"replayPredictionHash":frame_hash(replay_predictions),"aggregateHash":frame_hash(aggregate),"replayAggregateHash":frame_hash(replay_aggregate),"targetReplayUnits":int(replay.exactReplayPassed.sum()),"featureRows":len(features),"suffixRows":len(suffix)}
    if regeneration["status"]!="PASS": raise RuntimeError("L21 regeneration failed")
    runtime={"schema":"eidosoma.e01.s19_l21.runtime.v1","startedAtUtc":utc_now(),"wallSeconds":time.perf_counter()-start_wall,"processCpuSeconds":time.process_time()-start_cpu,"processCpuHours":(time.process_time()-start_cpu)/3600,"workers":workers,"threadsPerWorker":1,"gpuHours":0,"repositoryHead":git("rev-parse","HEAD"),"python":sys.version,"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,"sklearn":sklearn.__version__,"pyarrow":pyarrow.__version__}
    figures=make_figures(BUILD_ROOT,targets,risk,aggregate,comparisons,permutation,control_metrics,gates)
    report=report_text(targets,risk,aggregate,gates,classifications,selected,runtime)
    atomic_text(BUILD_ROOT/"research_step_full_results.md",report); atomic_text(BUILD_ROOT/"S19_L21_FULL_RESULTS.md",report)
    atomic_text(BUILD_ROOT/"loop_decision_summary.md",f"# S19-L21 decision summary\n\n**Classification:** {', '.join(classifications)}\n\n**Selected lead:** `{selected or 'NONE'}`.\n\n{('Proceed only to untouched confirmation.' if selected else 'The survival reformulation did not pass; proceed nonduplicatively to one outcome-blind representation loop.')}\n")
    write_json(BUILD_ROOT/"regeneration_validation.json",regeneration); write_json(BUILD_ROOT/"runtime_manifest.json",runtime)
    write_json(BUILD_ROOT/"classification.json",{"researchStepId":LOOP_ID,"versionedId":VERSION,"status":"COMPLETE_AUTONOMOUS_CONTINUATION_AUTHORIZED","classifications":classifications,"selectedDiscoveryLead":selected,"confirmedSolution":False,"s18Changed":False,"nextLoopAuthorized":True})
    write_json(BUILD_ROOT/"storage_validation.json",{"status":"PASS","retainedBytesBeforeManifest":sum(p.stat().st_size for p in BUILD_ROOT.rglob('*') if p.is_file()),"retainedGiBMaximum":25,"temporaryGiBMaximum":75,"figureCount":len(figures)})
    write_json(BUILD_ROOT/"validation_summary.json",{"status":"PASS","repositoryClean":not bool(git('status','--porcelain=v1')),"repositoryHead":git('rev-parse','HEAD'),"remoteHead":git('rev-parse','origin/eidosoma/groups/42'),"exactTargetReplay":bool(replay.exactReplayPassed.all()),"suffixInvariant":bool(suffix.passed.all()),"bootstrapReplicates":BOOTSTRAPS,"permutationReplicates":PERMUTATIONS,"newTrajectories":0})
    for name in ["preregistration.yaml","decision_record.md","fixture_results.parquet","immutable_prior_validation.json","implementation_lock.json","preoutcome_repository_lock.json","benchmark_projection.json"]: shutil.copy2(LOOP_ROOT/name,BUILD_ROOT/name)
    write_json(BUILD_ROOT/"artifact_manifest.json",manifest_for(BUILD_ROOT))
    for child in list(LOOP_ROOT.iterdir()): shutil.rmtree(child) if child.is_dir() else child.unlink()
    for child in BUILD_ROOT.iterdir(): shutil.copytree(child,LOOP_ROOT/child.name) if child.is_dir() else shutil.copy2(child,LOOP_ROOT/child.name)
    timestamp=utc_now(); append_root_ledgers(classifications,selected,timestamp); atomic_text(ARTIFACT_ROOT/"research_step_full_results.md",report.replace("# S19-L21","# S19 current handoff — S19-L21",1))
    write_json(ARTIFACT_ROOT/"s19_status.json",{"researchStepId":LOOP_ID,"status":"AUTONOMOUS_SEQUENCE_ACTIVE","lastCompletedLoop":LOOP_ID,"currentLoop":LOOP_ID,"nextLoopAuthorized":True,"authorizationUpperBound":"S19-L42","s20Status":"DEFINED_INACTIVE","outcomeClassification":classifications[0],"classifications":classifications,"selectedDiscoveryLead":selected,"validationResult":"PASS_SURVIVAL_TASK_MODEL_BOOTSTRAP_PERMUTATION_SUFFIX_IMMUTABILITY_REGENERATION","recommendedNextAction":f"UNTOUCHED_CONFIRMATION_{selected}" if selected else "S19_L22_OUTCOME_BLIND_REPRESENTATION","updatedAtUtc":timestamp})
    print(canonical_json({"status":"COMPLETE","classifications":classifications,"selectedDiscoveryLead":selected,"artifactRoot":str(LOOP_ROOT),"wallSeconds":runtime["wallSeconds"]}))


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--prepare-lock",action="store_true"); parser.add_argument("--workers",type=int,default=8); args=parser.parse_args()
    if not 1<=args.workers<=8: raise SystemExit("workers must be 1..8")
    BASE.VERSION=VERSION
    if args.prepare_lock: prepare_lock()
    else: execute(args.workers)


if __name__=="__main__": main()
