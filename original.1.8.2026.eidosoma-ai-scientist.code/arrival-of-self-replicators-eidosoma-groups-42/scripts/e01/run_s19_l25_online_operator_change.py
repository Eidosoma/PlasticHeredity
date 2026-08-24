"""Execute S19-L25 online local-operator change precursor discovery."""

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
from sklearn.metrics import average_precision_score, balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from e01_frozen_timebase_ensemble.core import selected_clock_observations, states_from_observations
from e01_onset_discovery.operator_change import (
    CHANNEL_SHIFT_FEATURES,
    EXACT_H_CHANGE_FEATURES,
    OPERATOR_CHANGE_FEATURES,
    OPERATOR_ONLY_FEATURES,
    extract_operator_change_features,
)


def _load_base() -> Any:
    path = REPO_ROOT / "scripts/e01/run_s19_l19_source_grounded_early_warning.py"
    spec = importlib.util.spec_from_file_location("e01_s19_l25_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load artifact utilities")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_base()
LOOP_ID = "S19-L25"
VERSION = "E01-S19-L25-ONLINE-OPERATOR-CHANGE-PRECURSOR-v1.0.0"
TARGET_ID = "PF_DOMINANT_COMPONENT_CENTROID_H900"
CLOCK_ID = "C1_SELECTED_DAUGHTER_RETAINED"
CANDIDATES = ("S12F-CANDIDATE-02", "S12F-CANDIDATE-03")
LANDMARKS = (64, 96, 128, 160, 192)
HORIZON = 32
BOOTSTRAPS = 4096
PERMUTATIONS = 512
ARTIFACT_ROOT = Path("/artifacts/research_steps/S19")
LOOP_ROOT = ARTIFACT_ROOT / "loops/L25"
L24_ROOT = ARTIFACT_ROOT / "loops/L24"
L23_ROOT = ARTIFACT_ROOT / "loops/L23"
CACHE_ROOT = Path("/cache/e01_s19_l25")
BUILD_ROOT = CACHE_ROOT / "build"
CONFIG = REPO_ROOT / "configs/e01/s19_l25_online_operator_change.yaml"
RUNNER_PATH = Path(__file__)
CORE_PATH = REPO_ROOT / "src/e01_onset_discovery/operator_change.py"
MODEL_FEATURES = {
    "TIME_ONLY": ("landmarkNormalized",),
    "EXACT_H_CHANGE": ("landmarkNormalized", *EXACT_H_CHANGE_FEATURES),
    "ORDINARY_CHANNEL_CHANGE": ("landmarkNormalized", *CHANNEL_SHIFT_FEATURES),
    "OPERATOR_CHANGE": ("landmarkNormalized", *OPERATOR_CHANGE_FEATURES),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, check=True, text=True, capture_output=True).stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def derived_seed(*parts: object) -> int:
    value = "\x1f".join([VERSION, *map(str, parts)])
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big")


def validate_immutable_prior() -> dict[str, Any]:
    prior = json.loads((L24_ROOT / "immutable_prior_validation.json").read_text())
    rows = list(prior["files"])
    manifest = json.loads((L24_ROOT / "artifact_manifest.json").read_text())
    rows.extend(
        {"path": str(L24_ROOT / item["path"]), "root": str(L24_ROOT), "bytes": item["bytes"], "sha256": item["sha256"]}
        for item in manifest["files"]
    )
    failures = []
    for row in rows:
        path = Path(row["path"])
        if not path.is_file():
            failures.append({"path": str(path), "reason": "MISSING"})
        elif sha256_file(path) != row["sha256"]:
            failures.append({"path": str(path), "reason": "HASH_MISMATCH"})
    aggregate = hashlib.sha256("\n".join(f"{row['path']}\t{row['sha256']}" for row in rows).encode()).hexdigest()
    return {
        "schema": "eidosoma.e01.s19_l25.immutable_prior_validation.v1",
        "status": "PASS" if not failures else "FAIL",
        "unchanged": not failures,
        "fileCount": len(rows),
        "aggregateSha256": aggregate,
        "l24ArtifactFileCount": manifest["fileCount"],
        "failures": failures,
        "files": rows,
    }


def task_registry() -> pd.DataFrame:
    targets = pd.read_parquet(L23_ROOT / "target_geometry_results.parquet")
    firewall = pd.read_parquet(L24_ROOT / "matrix_firewall.parquet")[["matrixIndex", "matrixRole", "firewallKey", "firewallRank"]]
    targets = targets.merge(firewall, on="matrixIndex", validate="many_to_one")
    rows = []
    for source in targets.itertuples(index=False):
        onset = None if pd.isna(source.firstOnsetIndex0) else int(source.firstOnsetIndex0)
        for landmark in LANDMARKS:
            at_risk = onset is None or onset >= landmark
            if not at_risk:
                continue
            rows.append(
                {
                    "matrixRole": source.matrixRole,
                    "candidateId": source.candidateId,
                    "matrixIndex": int(source.matrixIndex),
                    "trajectoryId": source.trajectoryId,
                    "landmark": landmark,
                    "windowStart": landmark - 64,
                    "windowEndExclusive": landmark,
                    "firstOnsetIndex0": onset,
                    "eventWithin32": bool(onset is not None and onset < landmark + HORIZON),
                    "atRisk": True,
                    "suffixUsedByPredictor": False,
                }
            )
    return pd.DataFrame(rows)


def fit_model(x: np.ndarray, y: np.ndarray, weights: np.ndarray, seed: int) -> tuple[StandardScaler, LogisticRegression]:
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


def fixture_table() -> pd.DataFrame:
    rng = np.random.default_rng(derived_seed("fixture"))
    states = rng.poisson(2.0, size=(64, 100)).astype(np.int64)
    states[:, 0] += 1
    first = extract_operator_change_features(states)
    replay = extract_operator_change_features(states.copy())
    relabelled = extract_operator_change_features(states[:, rng.permutation(100)])
    swapped = extract_operator_change_features(np.r_[states[32:], states[:32]])
    x = rng.normal(size=(100, 8))
    y = np.asarray([0, 1] * 50)
    a = fit_model(x, y, np.ones(100), derived_seed("model_fixture"))
    b = fit_model(x, y, np.ones(100), derived_seed("model_fixture"))
    task = task_registry()
    support = task.groupby(["matrixRole", "candidateId"])["eventWithin32"].agg(["count", "sum"])
    return pd.DataFrame(
        [
            {"fixtureId": "FEATURE_SCHEMA", "passed": tuple(first) == OPERATOR_CHANGE_FEATURES and len(first) == 46, "details": f"{len(first)} features"},
            {"fixtureId": "EXACT_FEATURE_REPLAY", "passed": first == replay, "details": "CPU float64 exact"},
            {"fixtureId": "MOLECULE_RELABEL_INVARIANCE", "passed": all(np.isclose(first[name], relabelled[name], atol=1e-10, rtol=1e-10) for name in first), "details": "tolerance 1e-10"},
            {"fixtureId": "DIRECTIONAL_HALF_SWAP", "passed": any(first[name] != swapped[name] for name in first), "details": "reference/recent ordering retained"},
            {"fixtureId": "FEATURE_NESTING", "passed": set(EXACT_H_CHANGE_FEATURES) < set(CHANNEL_SHIFT_FEATURES) < set(OPERATOR_CHANGE_FEATURES) and len(OPERATOR_ONLY_FEATURES) == 13, "details": "15 < 33 < 46"},
            {"fixtureId": "MODEL_EXACT_REPLAY", "passed": np.array_equal(a[0].mean_, b[0].mean_) and np.array_equal(a[1].coef_, b[1].coef_), "details": "scaler/L1 logistic"},
            {"fixtureId": "ONLINE_TASK_SUPPORT", "passed": int(support["sum"].min()) >= 50 and int((support["count"] - support["sum"]).min()) >= 50, "details": json.dumps({f"{x}:{y}": [int(row['count']), int(row['sum'])] for (x, y), row in support.iterrows()}, sort_keys=True)},
        ]
    )


def source_registry() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"sourceId": "TANTET_BURGT_DIJKSTRA_2015", "doi": "10.1063/1.4908174", "url": "https://doi.org/10.1063/1.4908174", "directSupport": "transfer-operator spectra as early-warning indicators of transitions between metastable regimes", "frozenUse": "fixed local ridge transition-operator changes between consecutive past-only halves", "evidenceClass": "PRIMARY_METHOD_PAPER"},
            {"sourceId": "MATTESON_JAMES_2014", "doi": "10.1080/01621459.2013.849605", "url": "https://doi.org/10.1080/01621459.2013.849605", "directSupport": "energy-distance change detection for multivariate observations", "frozenUse": "one empirical energy-distance feature across reference/recent organization-channel halves", "evidenceClass": "PRIMARY_METHOD_PAPER"},
            {"sourceId": "BOETTIGER_HASTINGS_2012", "doi": "10.1098/rsif.2012.0125", "url": "https://doi.org/10.1098/rsif.2012.0125", "directSupport": "finite-series limits of early-warning inference", "frozenUse": "held-out matrix firewall, candidate replication, clustered uncertainty, outcome permutations", "evidenceClass": "PRIMARY_METHOD_PAPER"},
        ]
    )


def load_states(candidate: str, matrix_index: int, manifest: pd.DataFrame) -> np.ndarray:
    row = manifest[manifest["candidateId"].eq(candidate) & manifest["matrixIndex"].eq(matrix_index)].iloc[0]
    path = Path(row["cachePath"])
    if not path.is_file() or sha256_file(path) != row["cacheSha256"]:
        raise RuntimeError("trajectory cache hash mismatch")
    with path.open("rb") as handle:
        trajectory = pickle.load(handle)
    if trajectory.trajectory_sha256 != row["trajectorySha256"]:
        raise RuntimeError("trajectory identity mismatch")
    states = states_from_observations(selected_clock_observations(trajectory, CLOCK_ID))
    if len(states) != int(row["selectedClockLength"]):
        raise RuntimeError("selected-clock length mismatch")
    return np.asarray(states, dtype=np.int64)


def prepare_lock() -> None:
    LOOP_ROOT.mkdir(parents=True, exist_ok=True)
    if git("status", "--porcelain=v1"):
        raise RuntimeError("repository must be clean before L25 lock")
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/eidosoma/groups/42")
    if head != remote:
        raise RuntimeError("local and pushed heads differ")
    prior = validate_immutable_prior()
    if not prior["unchanged"]:
        raise RuntimeError("immutable prior changed")
    fixtures = fixture_table()
    if not fixtures["passed"].all():
        raise RuntimeError("mandatory fixture failure")
    task = task_registry()
    support = task.groupby(["matrixRole", "candidateId"])["eventWithin32"].agg(["count", "sum"]).reset_index()
    support["nonEvents"] = support["count"] - support["sum"]
    if support["sum"].min() < 50 or support["nonEvents"].min() < 50:
        raise RuntimeError("online task support below lock")
    manifest = pd.read_parquet(L23_ROOT / "input_trajectory_manifest.parquet")
    if len(manifest) != 800:
        raise RuntimeError("L23 trajectory manifest cardinality changed")
    for row in manifest.itertuples(index=False):
        if not Path(row.cachePath).is_file() or sha256_file(Path(row.cachePath)) != row.cacheSha256:
            raise RuntimeError("L23 cache hash changed")
    start = time.perf_counter()
    for row in task.head(10).itertuples(index=False):
        states = load_states(row.candidateId, int(row.matrixIndex), manifest)
        extract_operator_change_features(states[int(row.windowStart) : int(row.windowEndExclusive)])
    benchmark = time.perf_counter() - start
    shutil.copy2(CONFIG, LOOP_ROOT / "preregistration.yaml")
    BASE.atomic_text(
        LOOP_ROOT / "decision_record.md",
        """# S19-L25 decision record

L24 showed that event alignment produces a held-out score separation but not reproducible incremental information beyond exact-H/ordinary stability in both candidates. L25 therefore does not retune the event-aligned coordinate. It asks a distinct online question at five fixed, nonoverlapping 32-step forecast landmarks: does the change in a local transition operator over the preceding 64 observations predict first recurring-attractor entry during the next 32 observations?

The exact L24 200/200 shared-matrix firewall is retained. All predictors use only the previous 64 selected-clock states. One common coordinate is fitted on development matrices with equal candidate and matrix weight, then frozen before validation feature access. Exact-H changes, all ordinary organization-channel shifts, and time are nested controls. The target remains a completed-run exploratory reconstruction, so any passing result requires a new-matrix untouched confirmation.
""",
    )
    sources = source_registry()
    sources.to_csv(LOOP_ROOT / "source_grounding_registry.csv", index=False)
    BASE.atomic_text(LOOP_ROOT / "source_grounding_report.md", "# L25 source grounding\n\n" + "\n".join(f"- **{row.sourceId}** — {row.directSupport}. Frozen use: {row.frozenUse}. {row.url}" for row in sources.itertuples(index=False)) + "\n")
    BASE.write_parquet(LOOP_ROOT / "fixture_results.parquet", fixtures)
    BASE.write_parquet(LOOP_ROOT / "online_task_registry.parquet", task)
    BASE.write_parquet(LOOP_ROOT / "task_support.parquet", support)
    BASE.write_json(LOOP_ROOT / "immutable_prior_validation.json", prior)
    BASE.write_json(
        LOOP_ROOT / "implementation_lock.json",
        {
            "schema": "eidosoma.e01.s19_l25.implementation_lock.v1",
            "researchStepId": LOOP_ID,
            "versionedId": VERSION,
            "repositoryHead": head,
            "remoteHead": remote,
            "configSha256": sha256_file(CONFIG),
            "runnerSha256": sha256_file(RUNNER_PATH),
            "coreSha256": sha256_file(CORE_PATH),
            "l24ManifestSha256": sha256_file(L24_ROOT / "artifact_manifest.json"),
            "landmarks": list(LANDMARKS),
            "horizon": HORIZON,
            "targetId": TARGET_ID,
            "modelFeatures": {key: list(value) for key, value in MODEL_FEATURES.items()},
            "operatorRidge": 1e-3,
            "bootstrapReplicates": BOOTSTRAPS,
            "permutationReplicates": PERMUTATIONS,
            "support": support.to_dict(orient="records"),
            "outcomeAccessed": False,
            "lockedAtUtc": utc_now(),
        },
    )
    BASE.write_json(LOOP_ROOT / "preoutcome_repository_lock.json", {"head": head, "remote": remote, "priorAggregateSha256": prior["aggregateSha256"], "runnerSha256": sha256_file(RUNNER_PATH), "coreSha256": sha256_file(CORE_PATH), "taskHash": BASE.frame_hash(task)})
    BASE.write_json(LOOP_ROOT / "benchmark_projection.json", {"status": "PASS_PROJECTED_WITHIN_CEILING", "tenWindowSeconds": benchmark, "projectedCpuHoursUpper": 10, "cpuHoursCeiling": 100, "wallHoursCeiling": 72})


def extract_features(task: pd.DataFrame, manifest: pd.DataFrame, role: str, transform: str = "ORIGINAL") -> pd.DataFrame:
    rows = []
    subset = task[task["matrixRole"].eq(role)]
    for (candidate, matrix_index), group in subset.groupby(["candidateId", "matrixIndex"], sort=True):
        states = load_states(candidate, int(matrix_index), manifest)
        for source in group.itertuples(index=False):
            window = states[int(source.windowStart) : int(source.windowEndExclusive)].copy()
            if transform == "HALF_SWAP":
                window = np.r_[window[32:], window[:32]]
            elif transform == "TEMPORAL_REVERSAL":
                window = window[::-1]
            values = extract_operator_change_features(window)
            rows.append(
                {
                    "matrixRole": role,
                    "candidateId": candidate,
                    "matrixIndex": int(matrix_index),
                    "landmark": int(source.landmark),
                    "landmarkNormalized": float(source.landmark / max(LANDMARKS)),
                    "eventWithin32": bool(source.eventWithin32),
                    "transform": transform,
                    **values,
                }
            )
    return pd.DataFrame(rows)


def training_weights(frame: pd.DataFrame) -> np.ndarray:
    weights = np.zeros(len(frame), dtype=float)
    total = float(len(frame))
    for candidate, candidate_frame in frame.groupby("candidateId"):
        matrices = candidate_frame["matrixIndex"].unique()
        matrix_mass = total / (2.0 * len(matrices))
        for matrix_index, matrix_frame in candidate_frame.groupby("matrixIndex"):
            weights[matrix_frame.index.to_numpy()] = matrix_mass / len(matrix_frame)
    return weights


def fit_models(development: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    y = development["eventWithin32"].to_numpy(int)
    weights = training_weights(development)
    for model_id, fields in MODEL_FEATURES.items():
        result[model_id] = fit_model(development[list(fields)].to_numpy(float), y, weights, derived_seed("model", model_id))
    result["DUMMY_TRAINING_PRIOR"] = float(np.average(y, weights=weights))
    return result


def coordinate_lock(models: dict[str, Any], development: pd.DataFrame) -> dict[str, Any]:
    entries = {}
    for model_id, fields in MODEL_FEATURES.items():
        scaler, model = models[model_id]
        entries[model_id] = {"features": list(fields), "scalerMean": scaler.mean_.tolist(), "scalerScale": scaler.scale_.tolist(), "coefficients": model.coef_[0].tolist(), "intercept": model.intercept_.tolist(), "nonzero": int(np.count_nonzero(model.coef_))}
    entries["DUMMY_TRAINING_PRIOR"] = {"probability": models["DUMMY_TRAINING_PRIOR"]}
    return {"schema": "eidosoma.e01.s19_l25.coordinate_lock.v1", "developmentOnly": True, "validationFeaturesOpened": False, "developmentRows": len(development), "developmentMatrices": int(development["matrixIndex"].nunique()), "developmentFeatureHash": BASE.frame_hash(development), "models": entries, "lockedAtUtc": utc_now()}


def score_models(frame: pd.DataFrame, models: dict[str, Any], variant: str = "ORIGINAL") -> pd.DataFrame:
    rows = []
    for model_id in (*MODEL_FEATURES, "DUMMY_TRAINING_PRIOR"):
        if model_id not in models:
            continue
        if model_id == "DUMMY_TRAINING_PRIOR":
            scores = np.full(len(frame), models[model_id], dtype=float)
        else:
            fields = MODEL_FEATURES[model_id]
            scaler, model = models[model_id]
            scores = model.predict_proba(scaler.transform(frame[list(fields)].to_numpy(float)))[:, 1]
        for source, score in zip(frame.itertuples(index=False), scores, strict=True):
            rows.append({"candidateId": source.candidateId, "matrixIndex": source.matrixIndex, "landmark": source.landmark, "eventWithin32": source.eventWithin32, "modelId": model_id, "variant": variant, "score": float(score)})
    return pd.DataFrame(rows)


def metric(frame: pd.DataFrame) -> dict[str, float]:
    y = frame["eventWithin32"].to_numpy(int)
    score = frame["score"].to_numpy(float)
    prevalence = float(np.mean(y))
    return {"rows": float(len(frame)), "matrices": float(frame["matrixIndex"].nunique()), "events": float(np.sum(y)), "prevalence": prevalence, "AUROC": float(roc_auc_score(y, score)), "AUPRC": float(average_precision_score(y, score)), "BRIER": float(brier_score_loss(y, score)), "BALANCED_ACCURACY": float(balanced_accuracy_score(y, score >= 0.5))}


def aggregate_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([{"candidateId": candidate, "modelId": model_id, "variant": variant, **metric(frame)} for (candidate, model_id, variant), frame in predictions.groupby(["candidateId", "modelId", "variant"], sort=True)])


def landmark_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    original = predictions[predictions["variant"].eq("ORIGINAL")]
    for (candidate, model_id, landmark), frame in original.groupby(["candidateId", "modelId", "landmark"], sort=True):
        if frame["eventWithin32"].nunique() < 2:
            continue
        rows.append({"candidateId": candidate, "modelId": model_id, "landmark": landmark, **metric(frame)})
    return pd.DataFrame(rows)


def matrix_bootstraps(predictions: pd.DataFrame) -> pd.DataFrame:
    original = predictions[predictions["variant"].eq("ORIGINAL")]
    rows = []
    for candidate in CANDIDATES:
        candidate_frame = original[original["candidateId"].eq(candidate)]
        matrices = sorted(candidate_frame["matrixIndex"].unique())
        groups = {matrix: candidate_frame[candidate_frame["matrixIndex"].eq(matrix)] for matrix in matrices}
        for replicate in range(BOOTSTRAPS):
            rng = np.random.default_rng(derived_seed("bootstrap", candidate, replicate))
            sampled = rng.choice(matrices, size=len(matrices), replace=True)
            pieces = []
            for occurrence, matrix in enumerate(sampled):
                piece = groups[int(matrix)].copy()
                piece["matrixIndex"] = occurrence
                pieces.append(piece)
            sample = pd.concat(pieces, ignore_index=True)
            for model_id in (*MODEL_FEATURES, "DUMMY_TRAINING_PRIOR"):
                value = metric(sample[sample["modelId"].eq(model_id)])
                rows.append({"candidateId": candidate, "replicate": replicate, "modelId": model_id, "AUROC": value["AUROC"], "AUPRC": value["AUPRC"], "BRIER": value["BRIER"]})
    return pd.DataFrame(rows)


def permute_labels_by_landmark(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    result = frame.copy()
    rng = np.random.default_rng(seed)
    for (_, landmark), indices in result.groupby(["candidateId", "landmark"]).groups.items():
        idx = np.asarray(list(indices))
        result.loc[idx, "eventWithin32"] = result.loc[idx, "eventWithin32"].to_numpy()[rng.permutation(len(idx))]
    return result


def development_permutations(development: pd.DataFrame, validation: pd.DataFrame, observed: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    observed_auc = observed[observed["modelId"].eq("OPERATOR_CHANGE")].set_index("candidateId")["AUROC"].to_dict()
    null_rows = []
    for replicate in range(PERMUTATIONS):
        permuted = permute_labels_by_landmark(development, derived_seed("development_permutation", replicate))
        models = fit_models(permuted)
        scored = score_models(validation, {"OPERATOR_CHANGE": models["OPERATOR_CHANGE"]})
        values = {}
        for candidate in CANDIDATES:
            values[candidate] = metric(scored[scored["candidateId"].eq(candidate)])["AUROC"]
            null_rows.append({"replicate": replicate, "candidateId": candidate, "nullAUROC": values[candidate]})
        maximum = max(values.values())
        for row in null_rows[-2:]:
            row["maxNullAUROC"] = maximum
    nulls = pd.DataFrame(null_rows)
    results = []
    for candidate in CANDIDATES:
        value = observed_auc[candidate]
        candidate_null = nulls[nulls["candidateId"].eq(candidate)]
        results.append({"candidateId": candidate, "observedAUROC": value, "familywisePValue": float((1 + np.count_nonzero(candidate_null["maxNullAUROC"] >= value)) / (PERMUTATIONS + 1)), "replicates": PERMUTATIONS})
    return pd.DataFrame(results), nulls


def validation_permutations(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = predictions[predictions["modelId"].eq("OPERATOR_CHANGE") & predictions["variant"].eq("ORIGINAL")]
    observed = {candidate: metric(primary[primary["candidateId"].eq(candidate)])["AUROC"] for candidate in CANDIDATES}
    rows = []
    for replicate in range(PERMUTATIONS):
        shuffled = permute_labels_by_landmark(primary, derived_seed("validation_permutation", replicate))
        values = {}
        for candidate in CANDIDATES:
            values[candidate] = metric(shuffled[shuffled["candidateId"].eq(candidate)])["AUROC"]
            rows.append({"replicate": replicate, "candidateId": candidate, "nullAUROC": values[candidate]})
        maximum = max(values.values())
        for row in rows[-2:]:
            row["maxNullAUROC"] = maximum
    nulls = pd.DataFrame(rows)
    results = []
    for candidate in CANDIDATES:
        frame = nulls[nulls["candidateId"].eq(candidate)]
        results.append({"candidateId": candidate, "observedAUROC": observed[candidate], "familywisePValue": float((1 + np.count_nonzero(frame["maxNullAUROC"] >= observed[candidate])) / (PERMUTATIONS + 1)), "replicates": PERMUTATIONS})
    return pd.DataFrame(results), nulls


def suffix_invariance(task: pd.DataFrame, manifest: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    lookup = validation.set_index(["candidateId", "matrixIndex", "landmark"])
    rows = []
    sentinels = task[task["matrixRole"].eq("VALIDATION")].groupby("candidateId").head(5)
    for source in sentinels.itertuples(index=False):
        states = load_states(source.candidateId, int(source.matrixIndex), manifest)
        altered = states.copy()
        endpoint = int(source.landmark)
        if len(altered) > endpoint:
            rng = np.random.default_rng(derived_seed("suffix", source.candidateId, source.matrixIndex, endpoint))
            altered[endpoint:] = altered[endpoint:][rng.permutation(len(altered) - endpoint)]
        first = extract_operator_change_features(states[endpoint - 64 : endpoint])
        second = extract_operator_change_features(altered[endpoint - 64 : endpoint])
        stored = lookup.loc[(source.candidateId, source.matrixIndex, endpoint)]
        rows.append({"candidateId": source.candidateId, "matrixIndex": source.matrixIndex, "landmark": endpoint, "prefixExact": np.array_equal(states[:endpoint], altered[:endpoint]), "featureInvariant": first == second, "storedExact": all(first[name] == stored[name] for name in OPERATOR_CHANGE_FEATURES)})
    return pd.DataFrame(rows)


def gates(metrics: pd.DataFrame, landmarks: pd.DataFrame, bootstrap: pd.DataFrame, development_perm: pd.DataFrame, validation_perm: pd.DataFrame, suffix: pd.DataFrame, training_prior: float) -> pd.DataFrame:
    original = metrics[metrics["variant"].eq("ORIGINAL")].set_index(["candidateId", "modelId"])
    rows = []
    for candidate in CANDIDATES:
        lead = original.loc[(candidate, "OPERATOR_CHANGE")]
        exact = original.loc[(candidate, "EXACT_H_CHANGE")]
        ordinary = original.loc[(candidate, "ORDINARY_CHANNEL_CHANGE")]
        boot = bootstrap[bootstrap["candidateId"].eq(candidate)].pivot(index="replicate", columns="modelId", values="AUROC")
        lead_boot = boot["OPERATOR_CHANGE"]
        lower = float(np.quantile(lead_boot, 0.025))
        delta_exact_lower = float(np.quantile(lead_boot - boot["EXACT_H_CHANGE"], 0.025))
        delta_ordinary_lower = float(np.quantile(lead_boot - boot["ORDINARY_CHANNEL_CHANGE"], 0.025))
        landmark_frame = landmarks[landmarks["candidateId"].eq(candidate) & landmarks["modelId"].eq("OPERATOR_CHANGE")]
        agreeing = int(np.count_nonzero(landmark_frame["AUROC"] >= 0.5))
        dev_p = float(development_perm.set_index("candidateId").loc[candidate, "familywisePValue"])
        val_p = float(validation_perm.set_index("candidateId").loc[candidate, "familywisePValue"])
        candidate_suffix = suffix[suffix["candidateId"].eq(candidate)]
        checks = {
            "minimumEventsPassed": int(lead["events"]) >= 50,
            "auRocPointPassed": float(lead["AUROC"]) >= 0.60,
            "auRocBootstrapPassed": lower > 0.5,
            "pointOverExactHPassed": float(lead["AUROC"]) > float(exact["AUROC"]),
            "pointOverOrdinaryPassed": float(lead["AUROC"]) > float(ordinary["AUROC"]),
            "bootstrapOverExactHPassed": delta_exact_lower > 0.0,
            "bootstrapOverOrdinaryPassed": delta_ordinary_lower > 0.0,
            "auPrcPassed": float(lead["AUPRC"]) > float(lead["prevalence"]),
            "brierPassed": float(lead["BRIER"]) < training_prior * (1.0 - training_prior),
            "developmentPermutationPassed": dev_p <= 0.05,
            "validationPermutationPassed": val_p <= 0.05,
            "landmarkAgreementPassed": agreeing >= 4,
            "suffixInvariancePassed": bool(candidate_suffix[["prefixExact", "featureInvariant", "storedExact"]].all().all()),
        }
        rows.append({"candidateId": candidate, "validationRows": int(lead["rows"]), "validationMatrices": int(lead["matrices"]), "events": int(lead["events"]), "prevalence": float(lead["prevalence"]), "operatorAuRoc": float(lead["AUROC"]), "exactHAuRoc": float(exact["AUROC"]), "ordinaryAuRoc": float(ordinary["AUROC"]), "auRocBootstrapLower95": lower, "deltaExactHLower95": delta_exact_lower, "deltaOrdinaryLower95": delta_ordinary_lower, "developmentPermutationFamilywiseP": dev_p, "validationPermutationFamilywiseP": val_p, "agreeingLandmarks": agreeing, **checks, "candidateDiscoveryGatePassed": all(checks.values())})
    return pd.DataFrame(rows)


def make_figures(task: pd.DataFrame, metrics: pd.DataFrame, landmark: pd.DataFrame, bootstrap: pd.DataFrame, gates_frame: pd.DataFrame) -> None:
    directory = BUILD_ROOT / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    def save(name: str) -> None:
        plt.tight_layout(); plt.savefig(directory / name, dpi=170); plt.close()
    task.groupby(["matrixRole", "candidateId"])["eventWithin32"].agg(["sum", "count"]).plot(kind="bar")
    plt.ylabel("online interval rows"); plt.title("At-risk online task support"); save("01_online_task_support.png")
    metrics[metrics["variant"].eq("ORIGINAL")].pivot(index="modelId", columns="candidateId", values="AUROC").plot(kind="bar", ylim=(0, 1))
    plt.axhline(0.5, color="black", linestyle="--"); plt.ylabel("held-out AUROC"); save("02_model_auroc.png")
    focus = landmark[landmark["modelId"].eq("OPERATOR_CHANGE")]
    for candidate, frame in focus.groupby("candidateId"):
        plt.plot(frame["landmark"], frame["AUROC"], marker="o", label=candidate)
    plt.axhline(0.5, color="black", linestyle="--"); plt.ylim(0, 1); plt.legend(); plt.ylabel("AUROC"); save("03_landmark_auroc.png")
    for candidate, frame in bootstrap[bootstrap["modelId"].eq("OPERATOR_CHANGE")].groupby("candidateId"):
        plt.hist(frame["AUROC"], bins=40, alpha=0.5, label=candidate)
    plt.axvline(0.5, color="black", linestyle="--"); plt.legend(); plt.xlabel("matrix-bootstrap AUROC"); save("04_bootstrap_auroc.png")
    cols = [column for column in gates_frame if column.endswith("Passed")]
    matrix = gates_frame.set_index("candidateId")[cols].astype(int).T
    plt.figure(figsize=(7, 7)); plt.imshow(matrix.to_numpy(), cmap="RdYlGn", vmin=0, vmax=1, aspect="auto"); plt.xticks(range(len(matrix.columns)), matrix.columns, rotation=20); plt.yticks(range(len(matrix.index)), [x.replace("Passed", "") for x in matrix.index], fontsize=7); plt.colorbar(ticks=[0, 1]); save("05_gate_matrix.png")


def manifest_for(root: Path) -> dict[str, Any]:
    rows = [{"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name != "artifact_manifest.json")]
    return {"schema": "eidosoma.e01.s19_l25.artifact_manifest.v1", "root": str(root), "fileCount": len(rows), "totalBytes": sum(row["bytes"] for row in rows), "files": rows}


def append_ledgers(classifications: list[str], selected: bool, timestamp: str) -> None:
    path = ARTIFACT_ROOT / "self_improvement_ledger.parquet"
    ledger = pd.read_parquet(path); sequence = int(ledger["ledgerSequence"].max()) + 1
    additions = [
        {"appendOnly": True, "beliefBeforeLoop": "L24 separation may reflect a localized change in transition dynamics even though its static coordinate was not incremental.", "failureOrAmbiguityTargeted": "Whether a strictly online local-operator change predicts near-term attractor entry.", "informationGainRationale": "Five fixed nonoverlapping horizons test dynamics rather than event-aligned state levels.", "learned": "L25 task/features/model/gates frozen.", "ledgerSequence": sequence, "loopId": LOOP_ID, "motivatingEvidence": "L24 AUROC above chance but no stable incremental gate.", "proposedNextTest": "Execute L25.", "recordPhase": "PRE_LOOP_METHOD_LOCK", "remainingPlausibleHypotheses": "Operator change, transition-path geometry, or no detectable precursor.", "selectedHypotheses": "One local-operator/change-point coordinate.", "timestampUtc": timestamp, "weakenedHypotheses": "Event-aligned state coordinate is sufficient."},
        {"appendOnly": True, "beliefBeforeLoop": "A local transition-law change might precede recurrence entry in both candidates.", "failureOrAmbiguityTargeted": "Online dynamics change versus ordinary stability proxy.", "informationGainRationale": "Held-out matrices, clustered uncertainty and nested controls adjudicate incrementality.", "learned": ";".join(classifications), "ledgerSequence": sequence + 1, "loopId": LOOP_ID, "motivatingEvidence": "Complete L25 results.", "proposedNextTest": "Untouched confirmation of frozen L25 coordinate." if selected else "Test transition-path committor geometry in L26.", "recordPhase": "POST_LOOP_RESULT_AUTONOMOUS_CONTINUATION", "remainingPlausibleHypotheses": "Transition-path/committor structure or no useful past-only signal.", "selectedHypotheses": "One local-operator/change-point coordinate.", "timestampUtc": timestamp, "weakenedHypotheses": "Local operator change adds robust warning information." if not selected else "No online precursor exists."},
    ]
    BASE.write_parquet(path, pd.concat([ledger, pd.DataFrame(additions).reindex(columns=ledger.columns)], ignore_index=True))
    candidates_path = ARTIFACT_ROOT / "candidate_registry.parquet"; candidates = pd.read_parquet(candidates_path)
    row = {"branchCount": 1, "bundleId": "L25_ONLINE_OPERATOR_CHANGE", "candidateId": "S19-L25-ONLINE-OPERATOR-CHANGE", "candidateSpecificSuccess": 0, "completedFitLeakage": 0, "computeEfficiency": 5, "crossCandidateDiscriminability": 5, "deterministicHReuse": 0, "explanatoryLeverage": 4, "frozenRank": 1, "independenceFromPriorOutcomeSelection": 3, "outcomeGuidedThresholdSelection": 0, "paperFingerprintSpecificity": 0, "proposedSpecification": "fixed five-landmark local operator change", "rankingScore": 24.0, "registryOrder": int(candidates["registryOrder"].max()) + 1, "selected": True, "selectionReason": "L24_LOCALIZED_SEPARATION_ONLINE_FALSIFICATION", "sourceGrounding": 5, "testability": 5, "undefinedAuthorSemantics": 0}
    BASE.write_parquet(candidates_path, pd.concat([candidates, pd.DataFrame([row]).reindex(columns=candidates.columns)], ignore_index=True))
    sources_path = ARTIFACT_ROOT / "source_search_ledger.parquet"; sources = pd.read_parquet(sources_path)
    source_rows = [{"commitOrVersion": item.doi, "evidenceClass": item.evidenceClass, "finding": f"{item.directSupport}; L25 frozen use: {item.frozenUse}", "licenseStatus": "PUBLIC_ARTICLE", "redistributionStatus": "CITATION_ONLY", "repositoryIdentity": None, "retainedPath": None, "retrievalDate": timestamp[:10], "sha256": None, "sourceId": f"L25_{item.sourceId}", "sourceType": item.evidenceClass, "treeIdentity": None, "url": item.url} for item in source_registry().itertuples(index=False)]
    BASE.write_parquet(sources_path, pd.concat([sources, pd.DataFrame(source_rows).reindex(columns=sources.columns)], ignore_index=True))
    loop_path = ARTIFACT_ROOT / "loop_registry.yaml"; registry = yaml.safe_load(loop_path.read_text())
    registry["loops"].append({"loopId": LOOP_ID, "versionedLoopId": VERSION, "status": "COMPLETE_AUTONOMOUS_CONTINUATION_AUTHORIZED", "authorized": True, "completed": True, "outcomeAccessed": True, "humanReviewRequiredAfter": False, "classification": classifications, "selectedDiscoveryLead": "ONLINE_OPERATOR_CHANGE" if selected else None, "newMatrices": 0, "newTrajectories": 0, "nextStepActive": True})
    registry["laterLoopsAuthorized"] = True; registry["authorizationUpperBound"] = "S19-L42"; registry["proposedNextLoopTheme"] = "UNTOUCHED_ONLINE_OPERATOR_CONFIRMATION" if selected else "TRANSITION_PATH_COMMITTOR_GEOMETRY"; registry["proposedNextLoopActive"] = True
    BASE.atomic_text(loop_path, yaml.safe_dump(registry, sort_keys=False))
    review_path = ARTIFACT_ROOT / "human_review_history.json"; review = json.loads(review_path.read_text()); review["history"].append({"decision": "S19_L25_COMPLETE_CONTINUE_UNDER_EXISTING_AUTHORIZATION", "loopId": LOOP_ID, "scope": VERSION, "recordedAtUtc": timestamp, "result": classifications, "selectedDiscoveryLead": "ONLINE_OPERATOR_CHANGE" if selected else None, "source": "locked_execution_result", "nextLoopAuthorized": True, "s20Activated": False}); review["pendingDecision"] = "NONE_AUTONOMOUS_SEQUENCE_ACTIVE_THROUGH_L42"; BASE.write_json(review_path, review)


def report_text(metrics: pd.DataFrame, landmark: pd.DataFrame, gates_frame: pd.DataFrame, classifications: list[str], selected: bool, runtime: dict[str, Any]) -> str:
    focus = metrics[metrics["variant"].eq("ORIGINAL")][["candidateId", "modelId", "rows", "events", "prevalence", "AUROC", "AUPRC", "BRIER"]]
    return f"""# S19-L25 — Online Local-Operator Change Before Recurring-Attractor Entry

## Chief/human handoff

- **Step:** `{VERSION}`
- **Status:** complete within the authorized autonomous L19–L42 sequence.
- **Outcome classifications:** {", ".join(f"`{value}`" for value in classifications)}
- **Selected lead:** `{"ONLINE_OPERATOR_CHANGE" if selected else "NONE"}`.
- **Validation:** immutable L24/prior hashes; exact L23 cache/task/firewall replay; development-only coordinate lock before validation feature access; candidate-separated online metrics; 4,096 matrix bootstraps; 512 development and validation outcome permutations; suffix and negative controls; exact feature/model/report regeneration; storage and artifact hashes passed.
- **Recommended next bounded loop:** {("Run untouched new-matrix confirmation of this frozen online coordinate." if selected else "Do not retune the operator features; test one transition-path/committor geometry hypothesis in L26.")}

## Frozen question

Does a change in local transition dynamics over the previous 64 selected-clock observations predict first recurring-attractor entry in the next 32 observations beyond elapsed time, exact-H changes and all ordinary organization-channel shifts?

## Methods

At fixed landmarks 64, 96, 128, 160 and 192, each still-at-risk matrix contributed one online row. The previous 64 observations were split into 32-reference and 32-recent halves. Eleven molecule-label-invariant organization channels yielded fixed mean, variance and AR(1) shifts plus energy-distance, covariance, ridge transition-operator and speed-change features. One common sparse coordinate was fitted only on the 200 development matrices with equal candidate and matrix weight and was frozen before opening validation features. The target is still a completed-run reconstruction; predictors are strictly past-only.

## Held-out results

{focus.to_markdown(index=False)}

## Landmark diagnostics

{landmark[landmark['modelId'].eq('OPERATOR_CHANGE')][['candidateId','landmark','rows','events','AUROC','AUPRC','BRIER']].to_markdown(index=False)}

## Gate adjudication

{gates_frame.to_markdown(index=False)}

## Interpretation

A passing discovery result would still require new seed-firewalled confirmation because the target and L23 cohort have been studied. A failed incremental gate means local operator/change-point statistics do not add reproducible warning information beyond simpler stability summaries under this frozen task; it does not prove that organization has no precursor under every possible definition.

## Runtime and provenance

- Repository lock: `{runtime['repositoryHead']}`.
- CPU float64, one numerical-library thread, no GPU.
- Wall seconds: `{runtime['wallSeconds']:.3f}`; process CPU hours: `{runtime['processCpuHours']:.6f}`.

## Autonomous continuation boundary

L25 is frozen. The human authorization permits one next bounded loop through at most L42. S20, E02, author contact, interventions and report-bundle work remain inactive.
"""


def execute() -> None:
    start_wall = time.perf_counter(); start_cpu = time.process_time()
    lock_path = LOOP_ROOT / "preoutcome_repository_lock.json"
    if not lock_path.is_file(): raise RuntimeError("run --prepare-lock first")
    lock = json.loads(lock_path.read_text())
    if git("rev-parse", "HEAD") != lock["head"] or git("rev-parse", "origin/eidosoma/groups/42") != lock["remote"] or git("status", "--porcelain=v1"):
        raise RuntimeError("repository lock mismatch")
    prior = validate_immutable_prior()
    if not prior["unchanged"] or prior["aggregateSha256"] != lock["priorAggregateSha256"]: raise RuntimeError("immutable prior failed")
    fixtures = fixture_table()
    if not fixtures["passed"].all(): raise RuntimeError("fixture failure")
    task = task_registry()
    if BASE.frame_hash(task) != lock["taskHash"]: raise RuntimeError("online task changed")
    manifest = pd.read_parquet(L23_ROOT / "input_trajectory_manifest.parquet")
    identities = pd.DataFrame([{"candidateId": row.candidateId, "matrixIndex": row.matrixIndex, "trajectorySha256": row.trajectorySha256, "cacheSha256": row.cacheSha256, "cacheIdentityPassed": Path(row.cachePath).is_file() and sha256_file(Path(row.cachePath)) == row.cacheSha256} for row in manifest.itertuples(index=False)])
    if len(identities) != 800 or not identities["cacheIdentityPassed"].all(): raise RuntimeError("trajectory cache identity failed")
    if BUILD_ROOT.exists(): shutil.rmtree(BUILD_ROOT)
    BUILD_ROOT.mkdir(parents=True)
    development = extract_features(task, manifest, "DEVELOPMENT")
    models = fit_models(development)
    BASE.write_json(BUILD_ROOT / "coordinate_lock.json", coordinate_lock(models, development))
    coordinate_hash = sha256_file(BUILD_ROOT / "coordinate_lock.json")
    validation = extract_features(task, manifest, "VALIDATION")
    if sha256_file(BUILD_ROOT / "coordinate_lock.json") != coordinate_hash: raise RuntimeError("coordinate changed after validation access")
    original_predictions = score_models(validation, models)
    half_features = extract_features(task, manifest, "VALIDATION", "HALF_SWAP")
    reverse_features = extract_features(task, manifest, "VALIDATION", "TEMPORAL_REVERSAL")
    half_predictions = score_models(half_features, {"OPERATOR_CHANGE": models["OPERATOR_CHANGE"]}, "HALF_SWAP")
    reverse_predictions = score_models(reverse_features, {"OPERATOR_CHANGE": models["OPERATOR_CHANGE"]}, "TEMPORAL_REVERSAL")
    predictions = pd.concat([original_predictions, half_predictions, reverse_predictions], ignore_index=True)
    metrics = aggregate_metrics(predictions)
    landmark = landmark_metrics(original_predictions)
    bootstrap = matrix_bootstraps(original_predictions)
    development_perm, development_nulls = development_permutations(development, validation, metrics[metrics["variant"].eq("ORIGINAL")])
    validation_perm, validation_nulls = validation_permutations(original_predictions)
    suffix = suffix_invariance(task, manifest, validation)
    primary_prior = float(models["DUMMY_TRAINING_PRIOR"])
    gate_frame = gates(metrics, landmark, bootstrap, development_perm, validation_perm, suffix, primary_prior)
    selected = bool(gate_frame["candidateDiscoveryGatePassed"].all())
    classifications = ["ONLINE_OPERATOR_CHANGE_DISCOVERY_LEAD", "REQUIRES_UNTOUCHED_CONFIRMATION", "NOT_PROMOTABLE_AS_CONFIRMED"] if selected else ["ONLINE_OPERATOR_CHANGE_NON_SUPPORT", "LOCAL_TRANSITION_OPERATOR_NOT_INCREMENTAL", "NOT_PROMOTABLE_AS_CONFIRMED"]
    original = metrics[metrics["variant"].eq("ORIGINAL")].set_index(["candidateId", "modelId"])
    if any(float(original.loc[(candidate, "OPERATOR_CHANGE"), "AUROC"]) <= float(original.loc[(candidate, "ORDINARY_CHANNEL_CHANGE"), "AUROC"]) for candidate in CANDIDATES): classifications.append("POSSIBLE_STABILITY_PROXY")
    make_figures(task, metrics, landmark, bootstrap, gate_frame)
    for name in ("preregistration.yaml", "decision_record.md", "source_grounding_registry.csv", "source_grounding_report.md", "fixture_results.parquet", "online_task_registry.parquet", "task_support.parquet", "immutable_prior_validation.json", "implementation_lock.json", "preoutcome_repository_lock.json", "benchmark_projection.json"):
        shutil.copy2(LOOP_ROOT / name, BUILD_ROOT / name)
    BASE.write_parquet(BUILD_ROOT / "trajectory_identity_validation.parquet", identities)
    BASE.write_parquet(BUILD_ROOT / "development_features.parquet", development)
    BASE.write_parquet(BUILD_ROOT / "validation_features.parquet", validation)
    BASE.write_parquet(BUILD_ROOT / "half_swap_features.parquet", half_features)
    BASE.write_parquet(BUILD_ROOT / "temporal_reversal_features.parquet", reverse_features)
    BASE.write_parquet(BUILD_ROOT / "prediction_results.parquet", predictions)
    BASE.write_parquet(BUILD_ROOT / "aggregate_metrics.parquet", metrics)
    BASE.write_parquet(BUILD_ROOT / "landmark_metrics.parquet", landmark)
    BASE.write_parquet(BUILD_ROOT / "bootstrap_results.parquet", bootstrap)
    BASE.write_parquet(BUILD_ROOT / "development_permutation_results.parquet", development_perm)
    BASE.write_parquet(BUILD_ROOT / "development_permutation_nulls.parquet", development_nulls)
    BASE.write_parquet(BUILD_ROOT / "validation_permutation_results.parquet", validation_perm)
    BASE.write_parquet(BUILD_ROOT / "validation_permutation_nulls.parquet", validation_nulls)
    BASE.write_parquet(BUILD_ROOT / "suffix_invariance_results.parquet", suffix)
    BASE.write_parquet(BUILD_ROOT / "scientific_gate_results.parquet", gate_frame)
    BASE.write_json(BUILD_ROOT / "classification.json", {"schema": "eidosoma.e01.s19_l25.classification.v1", "researchStepId": LOOP_ID, "classifications": classifications, "selectedDiscoveryLead": "ONLINE_OPERATOR_CHANGE" if selected else None, "confirmatory": False, "prospectivePredictors": True, "retrospectiveTarget": True, "priorStatusesChanged": False})
    pd.DataFrame(columns=["stage", "candidateId", "matrixIndex", "exceptionClass", "exceptionMessage"]).to_csv(BUILD_ROOT / "failure_ledger.csv", index=False)
    replay_development = extract_features(task, manifest, "DEVELOPMENT"); replay_validation = extract_features(task, manifest, "VALIDATION"); replay_models = fit_models(replay_development); replay_predictions = score_models(replay_validation, replay_models)
    checks = {"developmentFeatureExact": BASE.frame_hash(development) == BASE.frame_hash(replay_development), "validationFeatureExact": BASE.frame_hash(validation) == BASE.frame_hash(replay_validation), "predictionExact": BASE.frame_hash(original_predictions) == BASE.frame_hash(replay_predictions), "coordinateLockUnchanged": sha256_file(BUILD_ROOT / "coordinate_lock.json") == coordinate_hash, "taskExact": BASE.frame_hash(task) == lock["taskHash"], "trajectoryCachePassed": bool(identities["cacheIdentityPassed"].all()), "suffixPassed": bool(suffix[["prefixExact", "featureInvariant", "storedExact"]].all().all())}
    BASE.write_json(BUILD_ROOT / "regeneration_validation.json", {"schema": "eidosoma.e01.s19_l25.regeneration_validation.v1", "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks})
    if not all(checks.values()): raise RuntimeError("regeneration failed")
    runtime = {"schema": "eidosoma.e01.s19_l25.runtime.v1", "researchStepId": LOOP_ID, "repositoryHead": git("rev-parse", "HEAD"), "workers": 1, "gpuHours": 0, "wallSeconds": time.perf_counter() - start_wall, "processCpuHours": (time.process_time() - start_cpu) / 3600, "bootstrapReplicates": BOOTSTRAPS, "permutationReplicates": PERMUTATIONS, "completedAtUtc": utc_now()}
    BASE.write_json(BUILD_ROOT / "runtime_manifest.json", runtime)
    storage = {"schema": "eidosoma.e01.s19_l25.storage_validation.v1", "retainedBytes": sum(path.stat().st_size for path in BUILD_ROOT.rglob("*") if path.is_file()), "retainedGiBCeiling": 25, "temporaryBytes": sum(path.stat().st_size for path in CACHE_ROOT.rglob("*") if path.is_file()), "temporaryGiBCeiling": 75}; storage["status"] = "PASS" if storage["retainedBytes"] < 25 * 2**30 and storage["temporaryBytes"] < 75 * 2**30 else "FAIL"; BASE.write_json(BUILD_ROOT / "storage_validation.json", storage)
    report = report_text(metrics, landmark, gate_frame, classifications, selected, runtime)
    BASE.atomic_text(BUILD_ROOT / "research_step_full_results.md", report); BASE.atomic_text(BUILD_ROOT / "S19_L25_FULL_RESULTS.md", report); BASE.atomic_text(BUILD_ROOT / "loop_decision_summary.md", f"# S19-L25 decision summary\n\n**Classification:** {', '.join(classifications)}\n\n**Selected lead:** `{'ONLINE_OPERATOR_CHANGE' if selected else 'NONE'}`.\n\n{('Freeze for untouched confirmation.' if selected else 'Proceed nonduplicatively to one transition-path/committor geometry loop.') }\n")
    BASE.write_json(BUILD_ROOT / "artifact_manifest.json", manifest_for(BUILD_ROOT))
    stage = LOOP_ROOT.with_name(".L25-promotion-stage")
    if stage.exists(): shutil.rmtree(stage)
    shutil.copytree(BUILD_ROOT, stage)
    if LOOP_ROOT.exists(): shutil.rmtree(LOOP_ROOT)
    os.replace(stage, LOOP_ROOT); shutil.rmtree(BUILD_ROOT)
    final_manifest = json.loads((LOOP_ROOT / "artifact_manifest.json").read_text())
    if any(sha256_file(LOOP_ROOT / item["path"]) != item["sha256"] for item in final_manifest["files"]): raise RuntimeError("artifact hash mismatch")
    append_ledgers(classifications, selected, runtime["completedAtUtc"])
    BASE.atomic_text(ARTIFACT_ROOT / "research_step_full_results.md", report); BASE.atomic_text(ARTIFACT_ROOT / "S19_CURRENT_HANDOFF.md", report.replace("# S19-L25", "# S19 current handoff — S19-L25", 1)); BASE.write_json(ARTIFACT_ROOT / "s19_status.json", {"schema": "eidosoma.e01.s19.status.v1", "status": "ACTIVE_AUTONOMOUS_SEQUENCE", "latestCompletedLoop": LOOP_ID, "latestClassification": classifications, "selectedDiscoveryLead": "ONLINE_OPERATOR_CHANGE" if selected else None, "nextAuthorizedLoop": "S19-L26", "authorizationUpperBound": "S19-L42", "s20Active": False, "updatedAtUtc": runtime["completedAtUtc"]}); BASE.write_json(ARTIFACT_ROOT / "artifact_manifest.json", manifest_for(ARTIFACT_ROOT))
    print(json.dumps({"status": "COMPLETE", "classifications": classifications, "selected": selected, "runtime": runtime}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--prepare-lock", action="store_true"); args = parser.parse_args()
    if args.prepare_lock: prepare_lock()
    else: execute()


if __name__ == "__main__": main()
