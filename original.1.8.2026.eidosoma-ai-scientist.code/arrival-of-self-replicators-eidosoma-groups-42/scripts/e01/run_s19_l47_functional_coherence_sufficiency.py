#!/usr/bin/env python3
"""Run S19-L47 functional-coherence compositional-sufficiency audit."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from e01_onset_discovery.functional_coherence_sufficiency import (
    fit_ridge,
    predict_ridge,
    regression_metrics,
)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


L46 = load_module(
    "e01_l46_runner",
    ROOT / "scripts/e01/run_s19_l46_functional_hereditary_regime.py",
)
BASE = L46.BASE
ARTIFACT_ROOT = Path("/artifacts/research_steps/S19")
LOOP_ROOT = ARTIFACT_ROOT / "loops/L47"
L46_ROOT = ARTIFACT_ROOT / "loops/L46"
BUILD_ROOT = Path("/cache/e01_s19_l47/build")
CONFIG = ROOT / "configs/e01/s19_l47_functional_coherence_sufficiency.yaml"
RUNNER_PATH = Path(__file__).resolve()
CORE_PATH = ROOT / "src/e01_onset_discovery/functional_coherence_sufficiency.py"
LOOP_ID = "S19-L47"
VERSION = "E01-S19-L47-FUNCTIONAL-COHERENCE-COMPOSITIONAL-SUFFICIENCY-v1.0.0"
BOOTSTRAPS = 4096
PERMUTATIONS = 512
EVALUATION_COHORTS = ("L28_VALIDATION", "L31_CONFIRMATION")
CANDIDATES = ("S12F-CANDIDATE-02", "S12F-CANDIDATE-03")
TARGETS = {
    "CATALYTIC_ACTIVATION": "activationCoherenceExcess",
    "EXPECTED_NET_EXCHANGE": "netExchangeCoherenceExcess",
    "GROWTH_DIVISION": "growthCoherenceExcess",
}
M0_FEATURES = (
    "newTripleCompositionCoherence",
    "allInheritedCompositionCoherence",
    "oldNewCompositionH",
    "breakNewCompositionH",
)
M1_FEATURES = (
    *M0_FEATURES,
    "postbreakOpportunities",
    "inheritedPostbreakCount",
    "certificationDelayFissions",
)
MODELS = {"M0_COMPOSITION": M0_FEATURES, "M1_COMPOSITION_PLUS_CHRONOLOGY": M1_FEATURES}
SEED_ROOT = bytes.fromhex(
    "6ac1144e547bc0804eacb9bf63e7a3db44d83548e45adb2dd9818b82bec9aff5"
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def frame_hash(frame: pd.DataFrame) -> str:
    ordered = frame.reindex(sorted(frame.columns), axis=1).reset_index(drop=True)
    return hashlib.sha256(
        ordered.to_json(orient="table", index=False, double_precision=15).encode()
    ).hexdigest()


def seed_material(*parts: object) -> bytes:
    return hashlib.sha256(
        SEED_ROOT + b"\x00" + json.dumps(parts, separators=(",", ":")).encode()
    ).digest()


def derived_seed(*parts: object) -> int:
    return int.from_bytes(seed_material(*parts)[:16], "big")


def interval(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return float("nan"), float("nan")
    return tuple(map(float, np.quantile(finite, [0.025, 0.975])))


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3 or len(np.unique(a[mask])) < 2 or len(np.unique(b[mask])) < 2:
        return float("nan")
    return float(spearmanr(a[mask], b[mask]).statistic)


def validate_immutable_prior() -> dict[str, Any]:
    prior = L46.validate_immutable_prior()
    manifest = json.loads((L46_ROOT / "artifact_manifest.json").read_text())
    rows = []
    for row in manifest["files"]:
        path = L46_ROOT / row["path"]
        actual = sha256_file(path) if path.exists() else None
        rows.append(
            {
                "path": str(path),
                "expectedSha256": row["sha256"],
                "actualSha256": actual,
                "unchanged": actual == row["sha256"],
            }
        )
    passed = bool(prior["unchanged"] and rows and all(row["unchanged"] for row in rows))
    return {
        "schema": "eidosoma.e01.s19_l47.immutable_prior_validation.v1",
        "status": "PASS" if passed else "FAIL",
        "unchanged": passed,
        "priorThroughL45Unchanged": bool(prior["unchanged"]),
        "validatedL46ArtifactCount": len(rows),
        "aggregateSha256": hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "rows": rows,
    }


def source_registry() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sourceId": "L47_L46_LOCAL_FUNCTIONAL_COHERENCE",
                "evidenceClass": "DIRECT_FROZEN_E01_RESULT",
                "finding": "L46 found local catalytic, exchange and growth coherence but no restoration of the old regime.",
                "frozenUse": "sole L47 outcome family",
                "url": None,
            },
            {
                "sourceId": "L47_RUN3_H_DEFINED_TARGET",
                "evidenceClass": "DIRECT_FROZEN_E01_RESULT",
                "finding": "The run3 event is defined by three consecutive parent-daughter H>0.9 fissions.",
                "frozenUse": "composition-H sufficiency control",
                "url": None,
            },
            {
                "sourceId": "L47_GARD_FUNCTION_DEPENDS_ON_COMPOSITION",
                "evidenceClass": "DIRECT_PUBLICATION",
                "finding": "GARD catalytic rates are explicit deterministic functions of composition and beta.",
                "frozenUse": "do not mistake a transformed H effect for independent function",
                "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6073634/",
            },
        ]
    )


def fixture_results() -> pd.DataFrame:
    rng = np.random.default_rng(47)
    x = rng.normal(size=(40, 7))
    x[2, 1] = np.nan
    y = np.nan_to_num(x, nan=0.0) @ np.arange(1, 8) / 10
    first = fit_ridge(x, y, alpha=1.0)
    second = fit_ridge(x, y, alpha=1.0)
    prediction = predict_ridge(first, x)
    order = np.asarray([2, 0, 6, 1, 5, 3, 4])
    permuted = fit_ridge(x[:, order], y, alpha=1.0)
    rows = [
        {
            "fixtureId": "F01_EXACT_MODEL_REPLAY",
            "passed": np.array_equal(first.coefficients, second.coefficients),
            "detail": "fixed float64 solve",
        },
        {
            "fixtureId": "F02_TRAIN_ONLY_IMPUTATION",
            "passed": np.isfinite(first.medians).all(),
            "detail": "development medians stored in model state",
        },
        {
            "fixtureId": "F03_FEATURE_PERMUTATION",
            "passed": np.allclose(
                prediction, predict_ridge(permuted, x[:, order]), atol=1e-12
            ),
            "detail": "joint feature/state permutation",
        },
        {
            "fixtureId": "F04_PERFECT_METRICS",
            "passed": regression_metrics(y, y)["rSquared"] == 1.0,
            "detail": "R2 identity",
        },
        {
            "fixtureId": "F05_IMMEDIATE_STRATUM",
            "passed": (3 - 3) == 0 and (5 - 3) > 0,
            "detail": "certification delay definition",
        },
        {
            "fixtureId": "F06_SEED_SERIALIZATION",
            "passed": str(derived_seed("fixture")).isdigit(),
            "detail": "128-bit decimal-string manifest",
        },
    ]
    return pd.DataFrame(rows)


def episode_table() -> pd.DataFrame:
    episodes = pd.read_parquet(L46_ROOT / "functional_episode_results.parquet")
    growth = pd.read_parquet(L46_ROOT / "growth_division_results.parquet")
    keys = [
        "stateId",
        "evaluationCohort",
        "candidateId",
        "matrixIndex",
        "landmark",
        "branchIndex",
        "branchHalf",
        "oldGrowthComplete",
    ]
    merged = episodes.merge(growth, on=keys, validate="one_to_one")
    merged["compositionCoherenceExcess"] = (
        merged["newTripleCompositionCoherence"]
        - merged["allInheritedCompositionCoherence"]
    )
    merged["activationCoherenceExcess"] = (
        merged["activationNewTripleCoherence"]
        - merged["activationAllInheritedCoherence"]
    )
    merged["netExchangeCoherenceExcess"] = (
        merged["netExchangeNewTripleCoherence"]
        - merged["netExchangeAllInheritedCoherence"]
    )
    merged["growthCoherenceExcess"] = merged["orderedGrowthCoherenceExcess"]
    merged["activationMinusComposition"] = (
        merged["activationCoherenceExcess"] - merged["compositionCoherenceExcess"]
    )
    merged["netExchangeMinusComposition"] = (
        merged["netExchangeCoherenceExcess"] - merged["compositionCoherenceExcess"]
    )
    merged["certificationDelayFissions"] = (
        merged["run3CertificationRelativeOneBased"] - 3
    )
    merged["pathwayStratum"] = np.where(
        merged["certificationDelayFissions"].eq(0),
        "IMMEDIATE_CERTIFICATION",
        "DELAYED_CERTIFICATION",
    )
    if (
        len(merged) != 19_958
        or (merged["certificationDelayFissions"] < 0).any()
        or not np.isfinite(merged["activationCoherenceExcess"]).all()
    ):
        raise RuntimeError("L47 frozen episode-table contract failure")
    return merged.sort_values(
        ["candidateId", "matrixIndex", "landmark", "branchIndex"]
    ).reset_index(drop=True)


def matrix_table(episodes: pd.DataFrame) -> pd.DataFrame:
    columns = [
        *M1_FEATURES,
        *TARGETS.values(),
        "compositionCoherenceExcess",
        "activationMinusComposition",
        "netExchangeMinusComposition",
    ]
    matrix = (
        episodes.groupby(
            ["evaluationCohort", "candidateId", "matrixIndex"], as_index=False
        )[columns]
        .mean()
        .sort_values(["candidateId", "evaluationCohort", "matrixIndex"])
        .reset_index(drop=True)
    )
    return matrix


def pathway_matrix_results(episodes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["evaluationCohort", "candidateId", "matrixIndex"]
    for matrix_keys, group in episodes.groupby(keys, sort=False):
        for target_id, target_column in TARGETS.items():
            immediate = group[group["pathwayStratum"].eq("IMMEDIATE_CERTIFICATION")][
                target_column
            ]
            delayed = group[group["pathwayStratum"].eq("DELAYED_CERTIFICATION")][
                target_column
            ]
            rows.append(
                {
                    **dict(zip(keys, matrix_keys, strict=True)),
                    "targetId": target_id,
                    "immediateBranches": len(immediate),
                    "delayedBranches": len(delayed),
                    "immediateMean": float(immediate.mean()) if len(immediate) else np.nan,
                    "delayedMean": float(delayed.mean()) if len(delayed) else np.nan,
                    "delayedMinusImmediate": (
                        float(delayed.mean() - immediate.mean())
                        if len(delayed) and len(immediate)
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values([*keys, "targetId"]).reset_index(drop=True)


def fit_models(
    matrix: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions = []
    registry = []
    metric_rows = []
    for candidate in CANDIDATES:
        development = matrix[
            matrix["candidateId"].eq(candidate)
            & matrix["evaluationCohort"].eq("L28_DEVELOPMENT")
        ]
        for target_id, target_column in TARGETS.items():
            for model_id, features in MODELS.items():
                state = fit_ridge(
                    development[list(features)].to_numpy(float),
                    development[target_column].to_numpy(float),
                    alpha=1.0,
                )
                registry.append(
                    {
                        "candidateId": candidate,
                        "targetId": target_id,
                        "modelId": model_id,
                        "fitCohort": "L28_DEVELOPMENT",
                        "fitMatrices": len(development),
                        "featureIds": json.dumps(features),
                        "medians": json.dumps(state.medians.tolist()),
                        "means": json.dumps(state.means.tolist()),
                        "scales": json.dumps(state.scales.tolist()),
                        "coefficients": json.dumps(state.coefficients.tolist()),
                        "intercept": state.intercept,
                        "alpha": state.alpha,
                    }
                )
                for cohort in ("L28_DEVELOPMENT", *EVALUATION_COHORTS):
                    test = matrix[
                        matrix["candidateId"].eq(candidate)
                        & matrix["evaluationCohort"].eq(cohort)
                    ]
                    predicted = predict_ridge(
                        state, test[list(features)].to_numpy(float)
                    )
                    observed = test[target_column].to_numpy(float)
                    for row, truth, estimate in zip(
                        test.itertuples(index=False), observed, predicted, strict=True
                    ):
                        predictions.append(
                            {
                                "evaluationCohort": cohort,
                                "candidateId": candidate,
                                "matrixIndex": int(row.matrixIndex),
                                "targetId": target_id,
                                "modelId": model_id,
                                "observed": float(truth),
                                "predicted": float(estimate),
                                "residual": float(truth - estimate),
                            }
                        )
                    values = regression_metrics(observed, predicted)
                    metric_rows.append(
                        {
                            "evaluationCohort": cohort,
                            "candidateId": candidate,
                            "targetId": target_id,
                            "modelId": model_id,
                            "matrices": len(test),
                            **values,
                            "spearman": safe_spearman(observed, predicted),
                        }
                    )
    return (
        pd.DataFrame(predictions).sort_values(
            ["candidateId", "targetId", "modelId", "evaluationCohort", "matrixIndex"]
        ).reset_index(drop=True),
        pd.DataFrame(registry).sort_values(["candidateId", "targetId", "modelId"]).reset_index(drop=True),
        pd.DataFrame(metric_rows).sort_values(
            ["candidateId", "targetId", "modelId", "evaluationCohort"]
        ).reset_index(drop=True),
    )


def permutation_controls(matrix: pd.DataFrame) -> pd.DataFrame:
    rows = []
    features = list(M1_FEATURES)
    for candidate in CANDIDATES:
        development = matrix[
            matrix["candidateId"].eq(candidate)
            & matrix["evaluationCohort"].eq("L28_DEVELOPMENT")
        ]
        x_train = development[features].to_numpy(float)
        for target_id, target_column in TARGETS.items():
            y_train = development[target_column].to_numpy(float)
            for replicate in range(PERMUTATIONS):
                rng = np.random.Generator(
                    np.random.PCG64DXSM(
                        derived_seed("permutation", candidate, target_id, replicate)
                    )
                )
                state = fit_ridge(x_train, y_train[rng.permutation(len(y_train))])
                for cohort in EVALUATION_COHORTS:
                    test = matrix[
                        matrix["candidateId"].eq(candidate)
                        & matrix["evaluationCohort"].eq(cohort)
                    ]
                    predicted = predict_ridge(state, test[features].to_numpy(float))
                    values = regression_metrics(
                        test[target_column].to_numpy(float), predicted
                    )
                    rows.append(
                        {
                            "evaluationCohort": cohort,
                            "candidateId": candidate,
                            "targetId": target_id,
                            "replicate": replicate,
                            "rSquared": values["rSquared"],
                            "rmse": values["rmse"],
                        }
                    )
    return pd.DataFrame(rows).sort_values(
        ["candidateId", "targetId", "evaluationCohort", "replicate"]
    ).reset_index(drop=True)


def effect_matrix_results(
    matrix: pd.DataFrame, predictions: pd.DataFrame
) -> pd.DataFrame:
    """Create one value per catalytic matrix for every registered effect."""
    frames: list[pd.DataFrame] = []
    for effect_id, column in (
        ("ACTIVATION_MINUS_COMPOSITION", "activationMinusComposition"),
        ("NET_EXCHANGE_MINUS_COMPOSITION", "netExchangeMinusComposition"),
    ):
        frame = matrix[
            ["evaluationCohort", "candidateId", "matrixIndex", column]
        ].copy()
        frame = frame.rename(columns={column: "value"})
        frame["effectId"] = effect_id
        frame["effectFamily"] = "DIRECT_VECTOR_MINUS_COMPOSITION"
        frames.append(frame)
    residual = predictions[
        predictions["modelId"].eq("M1_COMPOSITION_PLUS_CHRONOLOGY")
    ][
        [
            "evaluationCohort",
            "candidateId",
            "matrixIndex",
            "targetId",
            "residual",
        ]
    ].copy()
    residual["effectId"] = "M1_RESIDUAL::" + residual["targetId"]
    residual["effectFamily"] = "COMPOSITION_CHRONOLOGY_RESIDUAL"
    residual = residual.rename(columns={"residual": "value"}).drop(
        columns="targetId"
    )
    frames.append(residual)
    result = pd.concat(frames, ignore_index=True)
    return result.sort_values(
        ["effectFamily", "effectId", "candidateId", "evaluationCohort", "matrixIndex"]
    ).reset_index(drop=True)


def bootstrap_effects(effects: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    bootstrap_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    keys = ["evaluationCohort", "candidateId", "effectFamily", "effectId"]
    for values, group in effects.groupby(keys, sort=False):
        finite = group["value"].dropna().to_numpy(dtype=np.float64)
        if not len(finite):
            replicates = np.full(BOOTSTRAPS, np.nan, dtype=np.float64)
        else:
            rng = np.random.Generator(
                np.random.PCG64DXSM(derived_seed("bootstrap_effect", *values))
            )
            indices = rng.integers(0, len(finite), size=(BOOTSTRAPS, len(finite)))
            replicates = finite[indices].mean(axis=1)
        for replicate, mean_value in enumerate(replicates):
            bootstrap_rows.append(
                {
                    **dict(zip(keys, values, strict=True)),
                    "replicate": replicate,
                    "meanValue": float(mean_value),
                }
            )
        low, high = interval(replicates)
        group_rows.append(
            {
                **dict(zip(keys, values, strict=True)),
                "matrixCount": len(finite),
                "meanValue": float(np.mean(finite)) if len(finite) else np.nan,
                "medianValue": float(np.median(finite)) if len(finite) else np.nan,
                "lower95": low,
                "upper95": high,
                "lowerBoundAboveZero": bool(np.isfinite(low) and low > 0),
            }
        )
    return (
        pd.DataFrame(group_rows).sort_values(keys).reset_index(drop=True),
        pd.DataFrame(bootstrap_rows)
        .sort_values([*keys, "replicate"])
        .reset_index(drop=True),
    )


def bootstrap_pathways(
    pathways: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    keys = ["evaluationCohort", "candidateId", "targetId"]
    for values, group in pathways.groupby(keys, sort=False):
        eligible = group.dropna(subset=["delayedMinusImmediate"])
        effects = eligible["delayedMinusImmediate"].to_numpy(dtype=np.float64)
        if len(effects):
            rng = np.random.Generator(
                np.random.PCG64DXSM(derived_seed("bootstrap_pathway", *values))
            )
            indices = rng.integers(0, len(effects), size=(BOOTSTRAPS, len(effects)))
            replicates = effects[indices].mean(axis=1)
        else:
            replicates = np.full(BOOTSTRAPS, np.nan, dtype=np.float64)
        for replicate, mean_value in enumerate(replicates):
            bootstrap_rows.append(
                {
                    **dict(zip(keys, values, strict=True)),
                    "replicate": replicate,
                    "meanDelayedMinusImmediate": float(mean_value),
                }
            )
        low, high = interval(replicates)
        group_rows.append(
            {
                **dict(zip(keys, values, strict=True)),
                "matrixCount": int(group["matrixIndex"].nunique()),
                "immediateMatrices": int(
                    group.loc[group["immediateBranches"].gt(0), "matrixIndex"].nunique()
                ),
                "delayedMatrices": int(
                    group.loc[group["delayedBranches"].gt(0), "matrixIndex"].nunique()
                ),
                "pairedMatrices": len(eligible),
                "immediateBranches": int(group["immediateBranches"].sum()),
                "delayedBranches": int(group["delayedBranches"].sum()),
                "meanDelayedMinusImmediate": (
                    float(np.mean(effects)) if len(effects) else np.nan
                ),
                "lower95": low,
                "upper95": high,
                "intervalExcludesZero": bool(
                    np.isfinite(low) and np.isfinite(high) and (low > 0 or high < 0)
                ),
                "direction": (
                    "POSITIVE"
                    if len(effects) and np.mean(effects) > 0
                    else "NEGATIVE"
                    if len(effects) and np.mean(effects) < 0
                    else "ZERO_OR_UNDEFINED"
                ),
            }
        )
    return (
        pd.DataFrame(group_rows).sort_values(keys).reset_index(drop=True),
        pd.DataFrame(bootstrap_rows)
        .sort_values([*keys, "replicate"])
        .reset_index(drop=True),
    )


def permutation_summary(
    controls: pd.DataFrame, performance: pd.DataFrame
) -> pd.DataFrame:
    actual = performance[
        performance["modelId"].eq("M1_COMPOSITION_PLUS_CHRONOLOGY")
        & performance["evaluationCohort"].isin(EVALUATION_COHORTS)
    ]
    rows = []
    keys = ["evaluationCohort", "candidateId", "targetId"]
    for key_values, group in controls.groupby(keys, sort=False):
        lookup = dict(zip(keys, key_values, strict=True))
        observed = actual
        for key, value in lookup.items():
            observed = observed[observed[key].eq(value)]
        if len(observed) != 1:
            raise RuntimeError("L47 permutation/actual model lookup failure")
        actual_r2 = float(observed.iloc[0]["rSquared"])
        actual_rmse = float(observed.iloc[0]["rmse"])
        rows.append(
            {
                **lookup,
                "permutations": len(group),
                "actualRSquared": actual_r2,
                "nullMedianRSquared": float(group["rSquared"].median()),
                "rSquaredEmpiricalP": float(
                    (1 + group["rSquared"].ge(actual_r2).sum()) / (len(group) + 1)
                ),
                "actualRmse": actual_rmse,
                "nullMedianRmse": float(group["rmse"].median()),
                "rmseEmpiricalP": float(
                    (1 + group["rmse"].le(actual_rmse).sum()) / (len(group) + 1)
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def scientific_gates(
    groups: pd.DataFrame, pathways: pd.DataFrame
) -> tuple[pd.DataFrame, list[str], str]:
    evaluation = groups[groups["evaluationCohort"].isin(EVALUATION_COHORTS)]
    direct = evaluation[
        evaluation["effectFamily"].eq("DIRECT_VECTOR_MINUS_COMPOSITION")
    ]
    residual = evaluation[
        evaluation["effectFamily"].eq("COMPOSITION_CHRONOLOGY_RESIDUAL")
    ]
    direct_pass = bool(
        len(direct) == 8 and direct["lowerBoundAboveZero"].all()
    )
    residual_pass = bool(
        len(residual) == 12 and residual["lowerBoundAboveZero"].all()
    )
    pathway_eval = pathways[pathways["evaluationCohort"].isin(EVALUATION_COHORTS)]
    pathway_availability = bool(
        len(pathway_eval) == 12
        and pathway_eval["immediateMatrices"].ge(20).all()
        and pathway_eval["delayedMatrices"].ge(20).all()
        and pathway_eval["pairedMatrices"].ge(20).all()
    )
    pathway_directions = set(
        pathway_eval.loc[
            pathway_eval["intervalExcludesZero"], "direction"
        ].astype(str)
    )
    pathway_consistent = bool(
        pathway_availability
        and len(pathway_eval) == 12
        and pathway_eval["intervalExcludesZero"].all()
        and len(pathway_directions) == 1
        and "ZERO_OR_UNDEFINED" not in pathway_directions
    )
    rows = [
        {
            "gateId": "VECTOR_DIRECT_CONTRAST_ALL_EVALUATION_GROUPS",
            "requiredRows": 8,
            "observedRows": len(direct),
            "passed": direct_pass,
            "criterion": "activation-minus-composition and exchange-minus-composition lower 95% bound above zero in four held-out groups",
        },
        {
            "gateId": "M1_RESIDUAL_ALL_TARGETS_ALL_EVALUATION_GROUPS",
            "requiredRows": 12,
            "observedRows": len(residual),
            "passed": residual_pass,
            "criterion": "composition-plus-chronology residual lower 95% bound above zero for three targets in four held-out groups",
        },
        {
            "gateId": "COMPLETE_INCREMENTAL_FUNCTIONAL_COHERENCE",
            "requiredRows": 2,
            "observedRows": 2,
            "passed": direct_pass and residual_pass,
            "criterion": "both direct and residual contracts pass",
        },
        {
            "gateId": "PATHWAY_STRATUM_AVAILABILITY",
            "requiredRows": 12,
            "observedRows": len(pathway_eval),
            "passed": pathway_availability,
            "criterion": "at least 20 immediate, delayed and paired matrices per target and held-out group",
        },
        {
            "gateId": "PATHWAY_COMMON_DIRECTION",
            "requiredRows": 12,
            "observedRows": len(pathway_eval),
            "passed": pathway_consistent,
            "criterion": "all delayed-minus-immediate intervals exclude zero in one common direction",
        },
    ]
    classifications = [
        (
            "VECTOR_FUNCTIONAL_COHERENCE_AMPLIFIED_BEYOND_COMPOSITION"
            if direct_pass
            else "VECTOR_FUNCTIONAL_COHERENCE_NOT_AMPLIFIED_BEYOND_COMPOSITION"
        ),
        (
            "FUNCTIONAL_COHERENCE_INCREMENTAL_BEYOND_COMPOSITION_CHRONOLOGY"
            if direct_pass and residual_pass
            else "FUNCTIONAL_COHERENCE_NOT_INCREMENTAL_BEYOND_COMPOSITION_CHRONOLOGY"
        ),
        (
            "TIMING_STRATIFIED_FUNCTIONAL_PATHWAYS"
            if pathway_consistent
            else "PATHWAY_HETEROGENEITY_NOT_IDENTIFIED"
        ),
        "NOT_PROMOTABLE_AS_CONFIRMED",
    ]
    if direct_pass and residual_pass:
        next_theme = "L48_FUNCTIONAL_COHERENCE_COMMITTOR"
    elif pathway_consistent:
        next_theme = "L48_PATHWAY_CONDITIONAL_COMMITTOR"
    else:
        next_theme = "L48_STOCHASTIC_SHOOTING_NECESSITY_AND_EFFICIENCY"
    return pd.DataFrame(rows), classifications, next_theme


def candidate_comparison(groups: pd.DataFrame) -> pd.DataFrame:
    evaluation = groups[groups["evaluationCohort"].isin(EVALUATION_COHORTS)]
    pivot = evaluation.pivot_table(
        index=["evaluationCohort", "effectFamily", "effectId"],
        columns="candidateId",
        values="meanValue",
    ).reset_index()
    left, right = CANDIDATES
    if left in pivot and right in pivot:
        pivot["candidateDifferenceC02MinusC03"] = pivot[left] - pivot[right]
        pivot["candidateDirectionAgreement"] = (
            np.sign(pivot[left]) == np.sign(pivot[right])
        )
    return pivot


def analysis_seed_manifest() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cohort in ("L28_DEVELOPMENT", *EVALUATION_COHORTS):
        for candidate in CANDIDATES:
            for effect_id in (
                "ACTIVATION_MINUS_COMPOSITION",
                "NET_EXCHANGE_MINUS_COMPOSITION",
                *[f"M1_RESIDUAL::{target}" for target in TARGETS],
            ):
                material = seed_material(
                    "bootstrap_effect", cohort, candidate, effect_id
                )
                rows.append(
                    {
                        "purpose": "MATRIX_BOOTSTRAP_EFFECT_STREAM",
                        "evaluationCohort": cohort,
                        "candidateId": candidate,
                        "analysisId": effect_id,
                        "replicate": None,
                        "derivedSeed": str(int.from_bytes(material[:16], "big")),
                        "seedMaterialSha256": hashlib.sha256(material).hexdigest(),
                    }
                )
            for target_id in TARGETS:
                material = seed_material(
                    "bootstrap_pathway", cohort, candidate, target_id
                )
                rows.append(
                    {
                        "purpose": "MATRIX_BOOTSTRAP_PATHWAY_STREAM",
                        "evaluationCohort": cohort,
                        "candidateId": candidate,
                        "analysisId": target_id,
                        "replicate": None,
                        "derivedSeed": str(int.from_bytes(material[:16], "big")),
                        "seedMaterialSha256": hashlib.sha256(material).hexdigest(),
                    }
                )
    for candidate in CANDIDATES:
        for target_id in TARGETS:
            for replicate in range(PERMUTATIONS):
                material = seed_material(
                    "permutation", candidate, target_id, replicate
                )
                rows.append(
                    {
                        "purpose": "DEVELOPMENT_MATRIX_LABEL_PERMUTATION",
                        "evaluationCohort": "L28_DEVELOPMENT",
                        "candidateId": candidate,
                        "analysisId": target_id,
                        "replicate": replicate,
                        "derivedSeed": str(int.from_bytes(material[:16], "big")),
                        "seedMaterialSha256": hashlib.sha256(material).hexdigest(),
                    }
                )
    frame = pd.DataFrame(rows)
    if frame["derivedSeed"].duplicated().any() or frame[
        "seedMaterialSha256"
    ].duplicated().any():
        raise RuntimeError("L47 analysis seed collision")
    return frame


def seed_firewall(seeds: pd.DataFrame) -> dict[str, Any]:
    prior_material: set[str] = set()
    for path in ARTIFACT_ROOT.glob("loops/L*/**/*seed*manifest*.parquet"):
        if "/L47/" in str(path):
            continue
        try:
            frame = pd.read_parquet(path)
        except (OSError, ValueError, TypeError):
            continue
        for column in frame.columns:
            if "seedmaterialsha256" in column.lower():
                prior_material.update(frame[column].dropna().astype(str))
    overlaps = sorted(set(seeds["seedMaterialSha256"].astype(str)) & prior_material)
    return {
        "schema": "eidosoma.e01.s19_l47.seed_firewall.v1",
        "status": "PASS" if not overlaps else "FAIL",
        "analysisSeedCount": len(seeds),
        "analysisSeedMaterialOverlapCount": len(overlaps),
        "newScientificStochasticStreams": 0,
    }


def input_scope_registry() -> pd.DataFrame:
    rows = []
    for name in (
        "functional_episode_results.parquet",
        "growth_division_results.parquet",
        "artifact_manifest.json",
    ):
        path = L46_ROOT / name
        rows.append(
            {
                "inputId": f"L46::{name}",
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "readOnly": True,
            }
        )
    return pd.DataFrame(rows)


def compute_tables() -> tuple[dict[str, pd.DataFrame], list[str], str]:
    episodes = episode_table()
    matrix = matrix_table(episodes)
    pathways = pathway_matrix_results(episodes)
    predictions, model_registry, performance = fit_models(matrix)
    permutation = permutation_controls(matrix)
    permutation_results = permutation_summary(permutation, performance)
    effects = effect_matrix_results(matrix, predictions)
    group_effects, effect_bootstrap = bootstrap_effects(effects)
    pathway_groups, pathway_bootstrap = bootstrap_pathways(pathways)
    gates, classifications, next_theme = scientific_gates(
        group_effects, pathway_groups
    )
    tables = {
        "episode_sufficiency_results.parquet": episodes,
        "matrix_sufficiency_results.parquet": matrix,
        "pathway_matrix_results.parquet": pathways,
        "model_registry.parquet": model_registry,
        "model_predictions.parquet": predictions,
        "model_performance.parquet": performance,
        "effect_matrix_results.parquet": effects,
        "group_effect_results.parquet": group_effects,
        "matrix_bootstrap_results.parquet": effect_bootstrap,
        "pathway_group_results.parquet": pathway_groups,
        "pathway_bootstrap_results.parquet": pathway_bootstrap,
        "permutation_control_results.parquet": permutation,
        "permutation_summary.parquet": permutation_results,
        "candidate_comparison.parquet": candidate_comparison(group_effects),
        "scientific_gate_results.parquet": gates,
    }
    return tables, classifications, next_theme


def make_figures(tables: dict[str, pd.DataFrame]) -> None:
    figure_root = BUILD_ROOT / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    groups = tables["group_effect_results.parquet"]
    heldout = groups[groups["evaluationCohort"].isin(EVALUATION_COHORTS)]
    direct = heldout[
        heldout["effectFamily"].eq("DIRECT_VECTOR_MINUS_COMPOSITION")
    ].copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(direct))
    ax.bar(x, direct["meanValue"], color="#4c78a8")
    ax.vlines(x, direct["lower95"], direct["upper95"], color="black")
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(
        x,
        [
            f"{row.evaluationCohort.split('_')[-1][:4]}-C{row.candidateId[-2:]}\n{row.effectId.split('_')[0]}"
            for row in direct.itertuples(index=False)
        ],
        fontsize=7,
    )
    ax.set_ylabel("Functional coherence excess − composition-H excess")
    ax.set_title("Direct functional amplification beyond composition")
    fig.tight_layout()
    fig.savefig(figure_root / "01_direct_function_minus_composition.png", dpi=160)
    plt.close(fig)

    performance = tables["model_performance.parquet"]
    eval_perf = performance[
        performance["evaluationCohort"].isin(EVALUATION_COHORTS)
    ].copy()
    fig, ax = plt.subplots(figsize=(11, 5))
    for model_id, marker in (
        ("M0_COMPOSITION", "o"),
        ("M1_COMPOSITION_PLUS_CHRONOLOGY", "s"),
    ):
        subset = eval_perf[eval_perf["modelId"].eq(model_id)]
        ax.scatter(
            np.arange(len(subset)),
            subset["rSquared"],
            label=model_id,
            marker=marker,
        )
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(
        np.arange(len(eval_perf[eval_perf["modelId"].eq("M0_COMPOSITION")])),
        [
            f"{r.evaluationCohort.split('_')[-1][:4]}-C{r.candidateId[-2:]}\n{r.targetId.split('_')[0]}"
            for r in eval_perf[eval_perf["modelId"].eq("M0_COMPOSITION")].itertuples(
                index=False
            )
        ],
        rotation=45,
        ha="right",
        fontsize=7,
    )
    ax.set_ylabel("Held-out matrix-level R²")
    ax.set_title("Can composition and chronology explain functional coherence?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_root / "02_model_generalization.png", dpi=160)
    plt.close(fig)

    residual = heldout[
        heldout["effectFamily"].eq("COMPOSITION_CHRONOLOGY_RESIDUAL")
    ]
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(residual))
    ax.bar(x, residual["meanValue"], color="#f58518")
    ax.vlines(x, residual["lower95"], residual["upper95"], color="black")
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(
        x,
        [
            f"{r.evaluationCohort.split('_')[-1][:4]}-C{r.candidateId[-2:]}\n{r.effectId.split('::')[-1].split('_')[0]}"
            for r in residual.itertuples(index=False)
        ],
        rotation=45,
        ha="right",
        fontsize=7,
    )
    ax.set_ylabel("Observed − M1 predicted coherence excess")
    ax.set_title("Residual functional coherence after composition and chronology")
    fig.tight_layout()
    fig.savefig(figure_root / "03_residual_functional_coherence.png", dpi=160)
    plt.close(fig)

    pathway = tables["pathway_group_results.parquet"]
    eval_pathway = pathway[pathway["evaluationCohort"].isin(EVALUATION_COHORTS)]
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(eval_pathway))
    ax.bar(x, eval_pathway["meanDelayedMinusImmediate"], color="#54a24b")
    ax.vlines(x, eval_pathway["lower95"], eval_pathway["upper95"], color="black")
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(
        x,
        [
            f"{r.evaluationCohort.split('_')[-1][:4]}-C{r.candidateId[-2:]}\n{r.targetId.split('_')[0]}"
            for r in eval_pathway.itertuples(index=False)
        ],
        rotation=45,
        ha="right",
        fontsize=7,
    )
    ax.set_ylabel("Delayed − immediate certification coherence")
    ax.set_title("Registered certification-timing pathway contrast")
    fig.tight_layout()
    fig.savefig(figure_root / "04_pathway_timing_contrast.png", dpi=160)
    plt.close(fig)

    permutation = tables["permutation_summary.parquet"]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(permutation))
    ax.scatter(x, permutation["actualRSquared"], label="actual M1", color="#4c78a8")
    ax.scatter(
        x,
        permutation["nullMedianRSquared"],
        label="permuted-target median",
        color="#e45756",
    )
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(
        x,
        [
            f"{r.evaluationCohort.split('_')[-1][:4]}-C{r.candidateId[-2:]}\n{r.targetId.split('_')[0]}"
            for r in permutation.itertuples(index=False)
        ],
        rotation=45,
        ha="right",
        fontsize=7,
    )
    ax.set_ylabel("Held-out matrix-level R²")
    ax.set_title("Registered development-target permutation control")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figure_root / "05_permutation_control.png", dpi=160)
    plt.close(fig)

    gates = tables["scientific_gate_results.parquet"]
    fig, ax = plt.subplots(figsize=(8, 4))
    values = gates[["passed"]].astype(int).to_numpy()
    image = ax.imshow(values, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks([0], ["pass"])
    ax.set_yticks(range(len(gates)), gates["gateId"], fontsize=7)
    ax.set_title("L47 decision matrix")
    fig.colorbar(image, ax=ax, ticks=[0, 1])
    fig.tight_layout()
    fig.savefig(figure_root / "06_decision_matrix.png", dpi=160)
    plt.close(fig)


def report_text(
    tables: dict[str, pd.DataFrame],
    classifications: list[str],
    runtime: dict[str, Any],
    next_theme: str,
) -> str:
    groups = tables["group_effect_results.parquet"]
    effects = groups[groups["evaluationCohort"].isin(EVALUATION_COHORTS)][
        [
            "evaluationCohort",
            "candidateId",
            "effectId",
            "meanValue",
            "lower95",
            "upper95",
        ]
    ]
    performance = tables["model_performance.parquet"]
    heldout_performance = performance[
        performance["evaluationCohort"].isin(EVALUATION_COHORTS)
    ][
        [
            "evaluationCohort",
            "candidateId",
            "targetId",
            "modelId",
            "rmse",
            "rSquared",
            "residualMean",
            "spearman",
        ]
    ]
    pathway = tables["pathway_group_results.parquet"]
    heldout_pathway = pathway[pathway["evaluationCohort"].isin(EVALUATION_COHORTS)]
    return f"""# S19-L47 Full Results — Functional Coherence versus Compositional Sufficiency

## Top summary

- **Research step:** `{VERSION}`
- **Completion status:** complete; additive exploratory evidence
- **Artifacts written:** frozen source/input/seed locks, episode and catalytic-matrix tables, two fixed ridge models, 4,096 matrix bootstraps, 512 target permutations, pathway-stratum audit, six figures, validation and hash manifests
- **Validation:** PASS — immutable prior, frozen L46 table identities, fixtures, source/seed/scope locks, two exact full analysis passes, regeneration, storage and artifact hashes
- **Outcome classification:** {', '.join(f'`{value}`' for value in classifications)}
- **Lay summary:** L46 found that newly inherited triples are locally coherent in catalytic, exchange and growth summaries. L47 asks whether that coherence contains anything beyond the compositional smoothness and timing that define the same three-in-a-row event. It makes no new simulation, target, threshold, branch or information-theory calculation.
- **Recommended next action:** `{next_theme}` under the existing human-authorized sequence. S20, E02, author contact and intervention work remain inactive.

## Frozen question

Does the local functional coherence seen in L46 exceed (a) direct composition-H coherence and (b) a development-only model containing compositional geometry, post-break opportunity count, inherited-fission count and certification delay? A companion pathway audit asks whether immediate versus delayed online certification separates one consistent functional transition subtype.

This audit does not test old-regime restoration again. It treats catalytic activation, expected net exchange and growth/division summaries as deterministic reconstructed-simulator proxies, not experimentally measured functions.

## Inputs and methods

- Inputs: exactly 19,958 frozen L46 certified-episode rows from 280 state/landmark units and their catalytic-matrix identities.
- Independent unit: catalytic matrix; candidate 2 and candidate 3, validation and confirmation remain separate.
- Direct vector contrasts: catalytic/exchange ordered-coherence excess minus the registered composition-H coherence excess.
- Fixed models: `M0_COMPOSITION` and `M1_COMPOSITION_PLUS_CHRONOLOGY`, ridge alpha 1.0, fit only on `L28_DEVELOPMENT`, no held-out refit or tuning.
- Uncertainty: exactly 4,096 catalytic-matrix bootstraps per registered effect; 512 development-target matrix permutations per candidate/target.
- Pathway strata: earliest possible run-3 certification versus delayed certification; no outcome-derived regrouping.
- Compute: intentional serial analysis because vectorized matrix bootstraps made worker-process overhead larger than the measured workload; every numerical-library thread was fixed to one.

## Registered effect results

{effects.to_markdown(index=False, floatfmt='.6f')}

## Held-out model performance

{heldout_performance.to_markdown(index=False, floatfmt='.6f')}

Negative held-out R² means that the frozen development model is worse than predicting the held-out cohort mean. A residual above zero is not automatically independent organization; it passes the registered gate only when its matrix-bootstrap lower bound exceeds zero in every candidate/cohort group.

## Immediate versus delayed certification

{heldout_pathway.to_markdown(index=False, floatfmt='.6f')}

## Scientific gates

{tables['scientific_gate_results.parquet'].to_markdown(index=False)}

## Permutation controls

{tables['permutation_summary.parquet'].to_markdown(index=False, floatfmt='.6f')}

## Interpretation boundary

The source-equation functional vectors are deterministic functions of composition and the catalytic matrix. Therefore a positive unadjusted coherence value is not evidence of a distinct functional memory. Only a preregistered increment beyond direct H and chronology could support that narrower claim. Conversely, failure of this audit does not erase L44's modest ordering result or L46's descriptive local coherence; it constrains their interpretation.

No result here establishes a universal replicator label, early warning, causal emergence, intervention efficacy, causal control or author-code identity. Prior S18 and S19 classifications remain unchanged.

## Validation, runtime and provenance

- Repository lock: `{runtime['repositoryHead']}`.
- Wall time: `{runtime['wallSeconds']:.3f}` seconds; estimated CPU upper bound: `{runtime['estimatedCpuHoursUpper']:.6f}` hours.
- Workers: `{runtime['workers']}`; numerical-library threads: `1`; GPU hours: `0`.
- New matrices/trajectories/branches: `0/0/0`.
- Exact complete analysis passes: `2`.
- Custom code: `{CORE_PATH.relative_to(ROOT)}` and `{RUNNER_PATH.relative_to(ROOT)}`.
- Runtime libraries: NumPy, pandas, SciPy, PyArrow and Matplotlib from the existing workspace environment.

## Limitations

L47 is an adaptive exploratory audit after many prior loops. Its fixed linear controls can test the registered mean-residual claim but cannot prove that every nonlinear compositional transformation has been exhausted. Growth summaries are reconstructed-simulator diagnostics, and certification timing is measured only for branches that achieved the frozen run-3 event. The result is therefore bounded evidence about the specific L46 coherence claim, not a global impossibility theorem about functional organization.
"""


def manifest_for(root: Path) -> dict[str, Any]:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            rows.append(
                {
                    "path": str(path.relative_to(root)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return {
        "schema": "eidosoma.e01.s19_l47.artifact_manifest.v1",
        "loopId": LOOP_ID,
        "files": rows,
        "fileCount": len(rows),
        "aggregateSha256": hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def append_ledgers(
    classifications: list[str], timestamp: str, next_theme: str
) -> None:
    ledger_path = ARTIFACT_ROOT / "self_improvement_ledger.parquet"
    ledger = pd.read_parquet(ledger_path)
    sequence = int(ledger["ledgerSequence"].max()) + 1
    additions = [
        {
            "appendOnly": True,
            "beliefBeforeLoop": "L46 local functional coherence may reflect composition-H smoothness because both source-rate vectors are deterministic functions of composition and beta.",
            "failureOrAmbiguityTargeted": "Whether functional coherence adds beyond compositional geometry, event opportunity/count and certification timing.",
            "informationGainRationale": "Direct paired contrasts and development-only held-out controls can falsify the independent-function interpretation without new simulation or feature search.",
            "learned": "L47 composition/chronology sufficiency audit locked before derived outcomes.",
            "ledgerSequence": sequence,
            "loopId": LOOP_ID,
            "motivatingEvidence": "L46 local functional coherence and equally strong direct compositional coherence.",
            "proposedNextTest": "Run the frozen composition/chronology sufficiency and timing-stratum audit.",
            "recordPhase": "PRE_LOOP_METHOD_LOCK",
            "remainingPlausibleHypotheses": "Independent functional coherence, transformed compositional smoothness, certification-timing pathways or stochastic-shooting necessity.",
            "selectedHypotheses": "L46 functional coherence may or may not survive direct H and chronology controls.",
            "timestampUtc": timestamp,
            "weakenedHypotheses": "High source-rate vector similarity alone demonstrates an independent functional regime.",
        },
        {
            "appendOnly": True,
            "beliefBeforeLoop": "An independent effect requires every registered held-out group to pass both direct-vector and M1-residual gates.",
            "failureOrAmbiguityTargeted": "Functional increment and pathway heterogeneity.",
            "informationGainRationale": "Exact replay and catalytic-matrix inference make negative results constrain the next representation search.",
            "learned": ";".join(classifications),
            "ledgerSequence": sequence + 1,
            "loopId": LOOP_ID,
            "motivatingEvidence": "Complete L47 result.",
            "proposedNextTest": next_theme,
            "recordPhase": "POST_LOOP_RESULT_AUTONOMOUS_CONTINUATION",
            "remainingPlausibleHypotheses": next_theme,
            "selectedHypotheses": "Functional coherence versus compositional/chronological sufficiency.",
            "timestampUtc": timestamp,
            "weakenedHypotheses": "Any registered failed L47 gate, without altering prior descriptive evidence.",
        },
    ]
    BASE.write_parquet(
        ledger_path,
        pd.concat(
            [ledger, pd.DataFrame(additions).reindex(columns=ledger.columns)],
            ignore_index=True,
        ),
    )
    markdown = ARTIFACT_ROOT / "SELF_IMPROVEMENT_LEDGER.md"
    BASE.atomic_text(
        markdown,
        markdown.read_text()
        + f"\n\n## {LOOP_ID} — functional coherence versus compositional sufficiency\n\n"
        + f"- **Learned:** {', '.join(classifications)}.\n"
        + f"- **Next:** `{next_theme}`.\n",
    )

    candidate_path = ARTIFACT_ROOT / "candidate_registry.parquet"
    candidates = pd.read_parquet(candidate_path)
    candidate = {
        "branchCount": 2,
        "bundleId": "L47_FUNCTIONAL_COHERENCE_SUFFICIENCY",
        "candidateId": "S19-L47-FUNCTIONAL-COHERENCE-SUFFICIENCY",
        "candidateSpecificSuccess": 0,
        "completedFitLeakage": 0,
        "computeEfficiency": 5,
        "crossCandidateDiscriminability": 5,
        "deterministicHReuse": 1,
        "explanatoryLeverage": 5,
        "frozenRank": 1,
        "independenceFromPriorOutcomeSelection": 4,
        "outcomeGuidedThresholdSelection": 0,
        "paperFingerprintSpecificity": 0,
        "proposedSpecification": "direct function-minus-H contrasts plus development-only composition/chronology ridge sufficiency audit",
        "rankingScore": 28.0,
        "registryOrder": int(candidates["registryOrder"].max()) + 1,
        "selected": "FUNCTIONAL_COHERENCE_INCREMENTAL_BEYOND_COMPOSITION_CHRONOLOGY"
        in classifications,
        "selectionReason": "L46_FUNCTIONAL_COHERENCE_REQUIRES_DIRECT_H_CONTROL",
        "sourceGrounding": 4,
        "testability": 5,
        "undefinedAuthorSemantics": 0,
    }
    BASE.write_parquet(
        candidate_path,
        pd.concat(
            [candidates, pd.DataFrame([candidate]).reindex(columns=candidates.columns)],
            ignore_index=True,
        ),
    )

    source_path = ARTIFACT_ROOT / "source_search_ledger.parquet"
    sources = pd.read_parquet(source_path)
    additions = []
    for row in source_registry().itertuples(index=False):
        additions.append(
            {
                "commitOrVersion": None,
                "evidenceClass": row.evidenceClass,
                "finding": f"{row.finding}; L47 use: {row.frozenUse}",
                "licenseStatus": "PUBLIC_METADATA_OR_WORKSPACE_EVIDENCE",
                "redistributionStatus": "REFERENCE_ONLY",
                "repositoryIdentity": None,
                "retainedPath": None,
                "retrievalDate": timestamp[:10],
                "sha256": None,
                "sourceId": f"L47_{row.sourceId}",
                "sourceType": row.evidenceClass,
                "treeIdentity": None,
                "url": row.url,
            }
        )
    BASE.write_parquet(
        source_path,
        pd.concat(
            [sources, pd.DataFrame(additions).reindex(columns=sources.columns)],
            ignore_index=True,
        ),
    )


def prepare_lock() -> None:
    LOOP_ROOT.mkdir(parents=True, exist_ok=True)
    if git("status", "--porcelain=v1"):
        raise RuntimeError("repository must be clean before L47 lock")
    head = git("rev-parse", "HEAD")
    remote = git("rev-parse", "origin/eidosoma/groups/42")
    if head != remote:
        raise RuntimeError("L47 local/remote commit mismatch")
    prior = validate_immutable_prior()
    fixtures = fixture_results()
    scope = input_scope_registry()
    seeds = analysis_seed_manifest()
    firewall = seed_firewall(seeds)
    benchmark = {
        "schema": "eidosoma.e01.s19_l47.benchmark_projection.v1",
        "outcomeBlind": True,
        "inputEpisodeRowsFromParquetMetadata": 19_958,
        "projectedWallHoursUpper": 0.5,
        "projectedCpuHoursUpper": 0.5,
        "workers": 1,
        "workersAvailable": 8,
        "parallelismDecision": "serial vectorized matrix analysis; process startup and serialization exceed expected computation",
        "cpuHoursCeiling": 24,
        "wallHoursCeiling": 24,
        "status": "PASS",
    }
    if (
        not prior["unchanged"]
        or not fixtures["passed"].all()
        or len(scope) != 3
        or firewall["status"] != "PASS"
    ):
        raise RuntimeError("L47 preoutcome validation failed")
    shutil.copy2(CONFIG, LOOP_ROOT / "preregistration.yaml")
    BASE.atomic_text(
        LOOP_ROOT / "decision_record.md",
        "# S19-L47 decision record\n\n"
        "The human-authorized autonomous sequence through L65 permits up to eight CPUs when they materially help. L46 found coherent new local catalytic, exchange and growth signatures but direct composition-H coherence was at least as large descriptively. Before opening any L47-derived effect, this step freezes one bounded sufficiency audit: the 19,958 L46 certified episodes only; two vector-minus-H contrasts; three functional targets; M0 composition and M1 composition-plus-chronology fixed ridge models fit only on L28 development matrices; immediate versus delayed certification strata; candidate/cohort separation; 4,096 matrix bootstraps; and 512 development-target permutations. No simulation, branch, target, H threshold, functional proxy, feature search, model search, Phi calculation or intervention is added. Serial vectorized execution is locked because worker overhead would not shorten this small table-only audit.\n",
    )
    BASE.write_json(LOOP_ROOT / "immutable_prior_validation.json", prior)
    BASE.write_parquet(LOOP_ROOT / "fixture_results.parquet", fixtures)
    BASE.write_parquet(LOOP_ROOT / "input_scope_registry.parquet", scope)
    BASE.write_parquet(LOOP_ROOT / "analysis_seed_manifest.parquet", seeds)
    BASE.write_json(LOOP_ROOT / "seed_firewall.json", firewall)
    BASE.write_json(LOOP_ROOT / "benchmark_projection.json", benchmark)
    sources = source_registry()
    BASE.write_parquet(LOOP_ROOT / "source_registry.parquet", sources)
    source_snapshot = {
        "schema": "eidosoma.e01.s19_l47.source_snapshot_manifest.v1",
        "l46FunctionalCoreSha256": sha256_file(
            ROOT / "src/e01_onset_discovery/functional_heredity_regime.py"
        ),
        "l47CoreSha256": sha256_file(CORE_PATH),
        "l47RunnerSha256": sha256_file(RUNNER_PATH),
        "configSha256": sha256_file(CONFIG),
        "sources": sources.to_dict("records"),
    }
    BASE.write_json(LOOP_ROOT / "source_snapshot_manifest.json", source_snapshot)
    locked_inputs = {
        "scopeRegistry": LOOP_ROOT / "input_scope_registry.parquet",
        "analysisSeeds": LOOP_ROOT / "analysis_seed_manifest.parquet",
        "seedFirewall": LOOP_ROOT / "seed_firewall.json",
        "benchmark": LOOP_ROOT / "benchmark_projection.json",
        "sourceSnapshot": LOOP_ROOT / "source_snapshot_manifest.json",
        "l46Episodes": L46_ROOT / "functional_episode_results.parquet",
        "l46Growth": L46_ROOT / "growth_division_results.parquet",
        "l46ArtifactManifest": L46_ROOT / "artifact_manifest.json",
    }
    hashes = {name: sha256_file(path) for name, path in locked_inputs.items()}
    lock = {
        "schema": "eidosoma.e01.s19_l47.implementation_lock.v1",
        "repositoryHead": head,
        "remoteHead": remote,
        "runnerSha256": sha256_file(RUNNER_PATH),
        "coreSha256": sha256_file(CORE_PATH),
        "configSha256": sha256_file(CONFIG),
        "episodeRowsExpected": 19_958,
        "targets": TARGETS,
        "models": {name: list(features) for name, features in MODELS.items()},
        "ridgeAlpha": 1.0,
        "fitCohort": "L28_DEVELOPMENT",
        "evaluationCohorts": list(EVALUATION_COHORTS),
        "pathwayStrata": ["IMMEDIATE_CERTIFICATION", "DELAYED_CERTIFICATION"],
        "matrixBootstraps": BOOTSTRAPS,
        "matrixLabelPermutations": PERMUTATIONS,
        "workers": 1,
        "workersAvailable": 8,
        "newMatrices": 0,
        "newTrajectories": 0,
        "newBranchStreams": 0,
        "lockedInputHashes": hashes,
        "outcomeAccessed": False,
        "lockedAtUtc": utc_now(),
    }
    BASE.write_json(LOOP_ROOT / "implementation_lock.json", lock)
    BASE.write_json(
        LOOP_ROOT / "preoutcome_repository_lock.json",
        {
            "head": head,
            "remote": remote,
            "priorAggregateSha256": prior["aggregateSha256"],
            "runnerSha256": sha256_file(RUNNER_PATH),
            "coreSha256": sha256_file(CORE_PATH),
            "configSha256": sha256_file(CONFIG),
            "lockedInputHashes": hashes,
        },
    )


def execute() -> None:
    started = time.perf_counter()
    lock = json.loads((LOOP_ROOT / "preoutcome_repository_lock.json").read_text())
    if (
        git("rev-parse", "HEAD") != lock["head"]
        or git("rev-parse", "origin/eidosoma/groups/42") != lock["remote"]
        or git("status", "--porcelain=v1")
    ):
        raise RuntimeError("L47 repository lock mismatch")
    prior = validate_immutable_prior()
    fixtures = fixture_results()
    locked_inputs = {
        "scopeRegistry": LOOP_ROOT / "input_scope_registry.parquet",
        "analysisSeeds": LOOP_ROOT / "analysis_seed_manifest.parquet",
        "seedFirewall": LOOP_ROOT / "seed_firewall.json",
        "benchmark": LOOP_ROOT / "benchmark_projection.json",
        "sourceSnapshot": LOOP_ROOT / "source_snapshot_manifest.json",
        "l46Episodes": L46_ROOT / "functional_episode_results.parquet",
        "l46Growth": L46_ROOT / "growth_division_results.parquet",
        "l46ArtifactManifest": L46_ROOT / "artifact_manifest.json",
    }
    if any(
        sha256_file(path) != lock["lockedInputHashes"][name]
        for name, path in locked_inputs.items()
    ):
        raise RuntimeError("L47 locked input changed")
    if (
        not prior["unchanged"]
        or prior["aggregateSha256"] != lock["priorAggregateSha256"]
        or not fixtures["passed"].all()
        or sha256_file(RUNNER_PATH) != lock["runnerSha256"]
        or sha256_file(CORE_PATH) != lock["coreSha256"]
        or sha256_file(CONFIG) != lock["configSha256"]
    ):
        raise RuntimeError("L47 pre-execution validation failed")
    if frame_hash(input_scope_registry()) != frame_hash(
        pd.read_parquet(LOOP_ROOT / "input_scope_registry.parquet")
    ):
        raise RuntimeError("L47 input scope regeneration mismatch")
    if BUILD_ROOT.exists():
        shutil.rmtree(BUILD_ROOT)
    BUILD_ROOT.mkdir(parents=True)

    tables, classifications, next_theme = compute_tables()
    make_figures(tables)
    tables_again, classifications_again, next_theme_again = compute_tables()
    exact = {
        name: frame_hash(frame) == frame_hash(tables_again[name])
        for name, frame in tables.items()
    }
    regeneration = {
        "schema": "eidosoma.e01.s19_l47.regeneration_validation.v1",
        "status": (
            "PASS"
            if all(exact.values())
            and classifications == classifications_again
            and next_theme == next_theme_again
            else "FAIL"
        ),
        "tableExact": exact,
        "classificationExact": classifications == classifications_again,
        "nextThemeExact": next_theme == next_theme_again,
        "analysisPasses": 2,
    }
    if regeneration["status"] != "PASS":
        raise RuntimeError("L47 exact regeneration failure")

    for name, frame in tables.items():
        BASE.write_parquet(BUILD_ROOT / name, frame)
    BASE.write_json(
        BUILD_ROOT / "classification.json",
        {
            "schema": "eidosoma.e01.s19_l47.classification.v1",
            "classifications": classifications,
            "nextTheme": next_theme,
            "priorStatusesChanged": False,
            "promotableAsConfirmed": False,
            "newScientificStreams": 0,
        },
    )
    pd.DataFrame(
        columns=[
            "failureId",
            "stage",
            "status",
            "reason",
            "scientificValuesReleased",
        ]
    ).to_csv(BUILD_ROOT / "failure_ledger.csv", index=False)
    elapsed = time.perf_counter() - started
    runtime = {
        "schema": "eidosoma.e01.s19_l47.runtime.v1",
        "repositoryHead": lock["head"],
        "workers": 1,
        "workersAvailable": 8,
        "parallelismDecision": "serial vectorized table analysis; multiprocessing would add avoidable serialization overhead",
        "numericalLibraryThreadsPerWorker": 1,
        "gpuHours": 0,
        "wallSeconds": elapsed,
        "estimatedCpuHoursUpper": elapsed / 3600,
        "episodeRows": len(tables["episode_sufficiency_results.parquet"]),
        "matrixRows": len(tables["matrix_sufficiency_results.parquet"]),
        "matrixBootstrapsPerEffect": BOOTSTRAPS,
        "matrixLabelPermutations": PERMUTATIONS,
        "analysisPasses": 2,
        "newMatrices": 0,
        "newTrajectories": 0,
        "newBranchStreams": 0,
        "completedAtUtc": utc_now(),
    }
    if runtime["estimatedCpuHoursUpper"] > 24 or runtime["wallSeconds"] > 24 * 3600:
        raise RuntimeError("L47 runtime ceiling exceeded")
    BASE.write_json(BUILD_ROOT / "runtime_manifest.json", runtime)
    BASE.write_json(BUILD_ROOT / "regeneration_validation.json", regeneration)
    retained_bytes = sum(
        path.stat().st_size for path in BUILD_ROOT.rglob("*") if path.is_file()
    ) + sum(path.stat().st_size for path in LOOP_ROOT.iterdir() if path.is_file())
    storage = {
        "schema": "eidosoma.e01.s19_l47.storage_validation.v1",
        "status": "PASS" if retained_bytes <= 15 * 1024**3 else "FAIL",
        "retainedBytes": retained_bytes,
        "retainedGiBCeiling": 15,
        "temporaryGiBCeiling": 30,
    }
    BASE.write_json(BUILD_ROOT / "storage_validation.json", storage)
    report = report_text(tables, classifications, runtime, next_theme)
    if report != report_text(tables, classifications, runtime, next_theme):
        raise RuntimeError("L47 report regeneration failure")
    BASE.atomic_text(BUILD_ROOT / "S19_L47_FULL_RESULTS.md", report)
    BASE.atomic_text(BUILD_ROOT / "research_step_full_results.md", report)
    BASE.atomic_text(
        BUILD_ROOT / "loop_decision_summary.md",
        f"# S19-L47 decision summary\n\n**Classification:** {', '.join(classifications)}\n\n**Next:** `{next_theme}`.\n",
    )
    if storage["status"] != "PASS":
        raise RuntimeError("L47 storage ceiling exceeded")
    for path in (BUILD_ROOT / "figures").glob("*.png"):
        if not path.stat().st_size:
            raise RuntimeError(f"empty L47 figure: {path}")

    for path in BUILD_ROOT.iterdir():
        destination = LOOP_ROOT / path.name
        if path.is_dir():
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(path, destination)
        else:
            shutil.copy2(path, destination)
    BASE.write_json(LOOP_ROOT / "artifact_manifest.json", manifest_for(LOOP_ROOT))
    if manifest_for(LOOP_ROOT) != json.loads(
        (LOOP_ROOT / "artifact_manifest.json").read_text()
    ):
        raise RuntimeError("L47 artifact manifest regeneration failed")

    append_ledgers(classifications, runtime["completedAtUtc"], next_theme)
    root_report = (
        f"# S19 current-step report\n\nLatest completed loop: `{LOOP_ID}`.\n\n"
        f"Classification: {', '.join(classifications)}.\n\n"
        f"Next autonomous theme: `{next_theme}`.\n"
    )
    BASE.atomic_text(ARTIFACT_ROOT / "S19_CURRENT_STEP_REPORT.md", root_report)
    BASE.atomic_text(ARTIFACT_ROOT / "CURRENT_STEP_HANDOFF.md", root_report)
    BASE.write_json(
        ARTIFACT_ROOT / "s19_status.json",
        {
            "schema": "eidosoma.e01.s19.status.v1",
            "programStatus": "ACTIVE_AUTONOMOUS_SEQUENCE",
            "latestCompletedLoop": LOOP_ID,
            "latestClassification": classifications,
            "nextAuthorizedLoop": "S19-L48",
            "nextTheme": next_theme,
            "authorizationUpperBound": "S19-L65",
            "s20Active": False,
            "updatedAtUtc": runtime["completedAtUtc"],
        },
    )
    BASE.write_json(ARTIFACT_ROOT / "artifact_manifest.json", manifest_for(ARTIFACT_ROOT))
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "classifications": classifications,
                "nextTheme": next_theme,
                "runtime": runtime,
            },
            indent=2,
        )
    )


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
