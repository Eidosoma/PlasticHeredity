#!/usr/bin/env python3
"""Run the isolated nonlinear history-only reviewer control."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd

from nonlinear_core import (
    BOOTSTRAP_REPETITIONS,
    CV_FOLDS,
    HISTORY_CLIP,
    MASTER_SEED,
    PCA_COMPONENTS,
    RANDOMIZATION_REPETITIONS,
    RIDGE_C,
    SPLINE_QUANTILES,
    TREE_ITERATIONS,
    TREE_L2,
    TREE_LEAF_GRID,
    TREE_LEARNING_RATE,
    bootstrap_interval,
    canonical_digest,
    derived_seed,
    fit_boosted_history,
    fit_capacity_matched_ridge,
    grouped_development_cv,
    holm_adjust,
    predict_boosted_history,
    predict_capacity_matched_ridge,
    sha256_file,
    sign_randomization_p,
    state_log_loss,
)
from reporting import write_reports


TASK_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TASK_ROOT.parent.parent
MATCHED_ROOT = TASK_ROOT.parent / "reviewer_matched_dimension_noise_control"
MATCHED_REPLAYS = MATCHED_ROOT / "artifacts" / "replays"
if str(MATCHED_ROOT) not in sys.path:
    sys.path.insert(0, str(MATCHED_ROOT))

from adapters import COHORTS, confirmation_targets, load_npz  # noqa: E402


ARTIFACTS = TASK_ROOT / "artifacts"
PROTOCOL_DIR = ARTIFACTS / "protocol"
MODEL_DIR = ARTIFACTS / "models"
OUTPUT_DIR = ARTIFACTS / "output"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_joblib(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    joblib.dump(value, temporary)
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def source_paths() -> dict[str, Path]:
    paths: dict[str, Path] = {
        "analysis_runner": TASK_ROOT / "run_analysis.py",
        "analysis_core": TASK_ROOT / "nonlinear_core.py",
        "analysis_reporting": TASK_ROOT / "reporting.py",
        "analysis_tests": TASK_ROOT / "test_nonlinear_history.py",
        "matched_adapter": MATCHED_ROOT / "adapters.py",
        "matched_replay_audit": MATCHED_REPLAYS / "replay_audit.json",
        "codex_headline_outcomes": PROJECT_ROOT / "replicators.13.8.2026.codex/results/full/analysis_arrays.npz",
        "codex_primary_outcomes": PROJECT_ROOT / "replicators.13.8.2026.codex/results/scaled5/analysis_arrays.npz",
        "fable_headline_outcomes": PROJECT_ROOT / "replicators.13.8.2026.fable/replication/results/conf_data.pkl",
        "fable_primary_outcomes": PROJECT_ROOT / "replicators.13.8.2026.fable/replication/results_sensitivity/v2_cohort.pkl",
    }
    for cohort in COHORTS:
        for candidate in ("02", "03"):
            paths[f"replay_{cohort}_c{candidate}"] = MATCHED_REPLAYS / f"{cohort}_c{candidate}.npz"
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing source artifacts: {missing}")
    return paths


def source_contract() -> dict[str, dict[str, str]]:
    return {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for name, path in source_paths().items()
    }


def runtime_contract() -> dict[str, str]:
    packages = ("numpy", "pandas", "scipy", "scikit-learn", "joblib")
    return {
        "python": platform.python_version(),
        **{package: importlib.metadata.version(package) for package in packages},
    }


def protocol_contract() -> dict[str, Any]:
    body: dict[str, Any] = {
        "title": "Capacity-matched and nonlinear history-only reviewer control",
        "date_frozen": "2026-08-19",
        "status": "reviewer-prompted post-hoc retained-outcome analysis",
        "question": (
            "Does the frozen composite retain confirmation information beyond a "
            "development-selected expressive model of observable hereditary history?"
        ),
        "cohorts": {key: spec.to_json() for key, spec in COHORTS.items()},
        "models": {
            "capacity_matched": {
                "direct_block": "registered implementation-specific H9 or H8",
                "nonlinear_library": (
                    "per-variable square and cube, truncated-cubic terms at development "
                    "quantiles 0.25/0.50/0.75, and all pairwise products"
                ),
                "history_clip_after_standardization": HISTORY_CLIP,
                "library_scaling": "development StandardScaler",
                "compression": f"development PCA with {PCA_COMPONENTS} components",
                "final_inputs": "12 plus registered history dimension (21 for H9; 20 for H8)",
                "final_model": f"implementation-matched L2 logistic ridge, C={RIDGE_C}",
            },
            "expressive_tree": {
                "model": "HistGradientBoostingClassifier with log loss",
                "history_only": True,
                "max_leaf_nodes_grid": list(TREE_LEAF_GRID),
                "iterations": TREE_ITERATIONS,
                "learning_rate": TREE_LEARNING_RATE,
                "l2_regularization": TREE_L2,
                "min_samples_leaf": "max(100, five times development branches per state)",
                "early_stopping": False,
            },
        },
        "selection": {
            "folds": CV_FOLDS,
            "group": "development matrix",
            "tree_selection": "minimum weighted development CV log loss; smaller tree wins ties",
            "family_selection": (
                "minimum weighted development CV log loss between fixed spline/interactions "
                "ridge and selected tree; spline wins exact ties"
            ),
            "confirmation_selection_or_recalibration": "none",
        },
        "inference": {
            "primary_cohorts": ["codex_primary", "fable_primary"],
            "secondary_cohorts": ["codex_headline", "fable_headline"],
            "branch_halves": {"A": [0, 32], "B": [32, 64]},
            "score": "branch log loss in nats, averaged to state then matrix",
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "randomization_repetitions": RANDOMIZATION_REPETITIONS,
            "multiplicity": "Holm across eight primary selected-history-versus-composite cells",
            "strong_gate": (
                "positive composite gain, positive whole-matrix 95% lower bound, and "
                "Holm-adjusted one-sided matrix-sign p<0.05 in all eight primary cells"
            ),
        },
        "master_seed_label": MASTER_SEED,
        "prohibitions": [
            "no new principal lineages",
            "no new confirmation futures",
            "no confirmation-based model or hyperparameter selection",
            "no confirmation recalibration",
            "no pooling across candidates or fixed branch halves",
        ],
        "limitations": [
            "The reviewer concern and prior confirmation results were already known.",
            "This is protocol-locked post-hoc evidence, not prospective preregistration.",
            "Tested nonlinear families cannot rule out every possible history-only model.",
            "Effective capacity of a boosted ensemble is not exactly equated to ridge capacity.",
            "The originating L53/L54 row-level machine-readable artifacts are unavailable.",
        ],
        "sources": source_contract(),
        "runtime": runtime_contract(),
    }
    body["protocol_id"] = canonical_digest(body)
    return body


def prepare() -> None:
    protocol = protocol_contract()
    json_path = PROTOCOL_DIR / "protocol.json"
    if json_path.exists():
        existing = json.loads(json_path.read_text(encoding="utf-8"))
        if existing != protocol:
            raise RuntimeError("existing frozen protocol differs from the current source contract")
    else:
        atomic_json(json_path, protocol)
    markdown = f"""# Frozen protocol: nonlinear history-only comparator

**Protocol ID:** `{protocol['protocol_id']}`  
**Frozen:** 2026-08-19  
**Status:** Reviewer-prompted post-hoc retained-outcome analysis

## Question

Does the frozen composite retain confirmation information beyond a
development-selected expressive model of observable hereditary history?

## Fixed challengers

1. **Exact input match:** registered H9/H8 plus twelve development-fitted
   components of a truncated-cubic-spline and all-pairwise-interaction history
   library, followed by the implementation-matched final ridge with `C=0.1`.
2. **Expressive stress test:** history-only histogram gradient boosting with
   3, 7, or 15 leaves per stage; all other settings are fixed.

Five-fold development-matrix-grouped cross-validation selects tree size and
then the lower-development-loss family. The selected model is frozen before
confirmation scoring. Confirmation never selects, tunes, or recalibrates a
model.

## Inference

The scaled 200-matrix clean-room cohorts are primary. The matched 40-matrix
cohorts are secondary. Candidate and branch half remain separate. Whole-matrix
bootstrap and paired sign-randomization use 4,096 draws; the eight primary
selected-history comparisons receive Holm adjustment.

No new lineage or future is generated. The originating workflow is excluded
because its row-level machine-readable artifacts are unavailable. This is
post-hoc robustness evidence, not a prospective confirmation.
"""
    markdown_path = PROTOCOL_DIR / "PROTOCOL.md"
    if not markdown_path.exists():
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(markdown, encoding="utf-8")
    print(f"protocol frozen: {protocol['protocol_id']}", flush=True)


def select_models() -> None:
    protocol = json.loads((PROTOCOL_DIR / "protocol.json").read_text(encoding="utf-8"))
    fold_records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for cohort, spec in COHORTS.items():
        for candidate in ("02", "03"):
            print(f"development CV: {cohort} c{candidate}", flush=True)
            data = load_npz(MATCHED_REPLAYS / f"{cohort}_c{candidate}.npz")
            records, summary = grouped_development_cv(
                data["dev_history"],
                data["dev_targets"],
                data["dev_matrix"],
                spec.pipeline,
                cohort,
                candidate,
            )
            fold_records.extend(records)
            summaries.append(summary)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fold_records).to_csv(MODEL_DIR / "cv_fold_scores.csv", index=False)
    flat_summaries = []
    for summary in summaries:
        flat_summaries.append(
            {
                **{key: value for key, value in summary.items() if key != "tree_cv_log_loss_by_leaves"},
                **{
                    f"tree_{leaves}_cv_log_loss": score
                    for leaves, score in summary["tree_cv_log_loss_by_leaves"].items()
                },
            }
        )
    pd.DataFrame(flat_summaries).to_csv(MODEL_DIR / "model_selection.csv", index=False)
    atomic_json(
        MODEL_DIR / "selection_manifest.json",
        {
            "protocol_id": protocol["protocol_id"],
            "confirmation_outcomes_opened": False,
            "selection": summaries,
        },
    )


def fit_models() -> None:
    protocol = json.loads((PROTOCOL_DIR / "protocol.json").read_text(encoding="utf-8"))
    selections = pd.read_csv(MODEL_DIR / "model_selection.csv")
    manifest: dict[str, Any] = {"protocol_id": protocol["protocol_id"], "models": {}}
    for cohort, spec in COHORTS.items():
        for candidate in ("02", "03"):
            print(f"full development fit: {cohort} c{candidate}", flush=True)
            data = load_npz(MATCHED_REPLAYS / f"{cohort}_c{candidate}.npz")
            selection = selections[
                (selections["cohort"] == cohort)
                & (selections["candidate"].astype(str).str.zfill(2) == candidate)
            ]
            if len(selection) != 1:
                raise AssertionError("model selection row is not unique")
            row = selection.iloc[0]
            leaves = int(row["selected_tree_leaves"])
            ridge = fit_capacity_matched_ridge(
                data["dev_history"], data["dev_targets"], spec.pipeline
            )
            tree = fit_boosted_history(
                data["dev_history"],
                data["dev_targets"],
                leaves,
                derived_seed(cohort, candidate, "full_tree", leaves),
            )
            spline_prediction = predict_capacity_matched_ridge(ridge, data["conf_history"])
            tree_prediction = predict_boosted_history(tree, data["conf_history"])
            selected_family = str(row["selected_family"])
            selected_prediction = (
                spline_prediction
                if selected_family == "spline_interaction_pca12_ridge"
                else tree_prediction
            )
            atomic_npz(
                MODEL_DIR / f"{cohort}_c{candidate}_predictions.npz",
                spline=spline_prediction,
                tree=tree_prediction,
                selected=selected_prediction,
            )
            atomic_joblib(
                MODEL_DIR / f"{cohort}_c{candidate}_models.joblib",
                {"spline": ridge, "tree": tree, "selected_family": selected_family},
            )
            manifest["models"][f"{cohort}_c{candidate}"] = {
                "history_dimension": int(data["dev_history"].shape[1]),
                "capacity_matched_inputs": int(data["dev_history"].shape[1] + PCA_COMPONENTS),
                "selected_tree_leaves": leaves,
                "selected_family": selected_family,
                "spline_transformer": ridge.transformer.audit(),
                "ridge_iterations": int(ridge.classifier.n_iter_.max()),
                "tree_iterations": int(tree.n_iter_),
                "prediction_rows": int(selected_prediction.size),
            }
    atomic_json(MODEL_DIR / "fit_manifest.json", manifest)


def analyze() -> None:
    protocol = json.loads((PROTOCOL_DIR / "protocol.json").read_text(encoding="utf-8"))
    selections = pd.read_csv(MODEL_DIR / "model_selection.csv")
    cell_records: list[dict[str, Any]] = []
    comparison_records: list[dict[str, Any]] = []
    primary_positions: list[int] = []
    primary_p_values: list[float] = []

    for cohort, spec in COHORTS.items():
        role = "primary" if cohort in ("codex_primary", "fable_primary") else "secondary_headline"
        for candidate in ("02", "03"):
            data = load_npz(MATCHED_REPLAYS / f"{cohort}_c{candidate}.npz")
            with np.load(MODEL_DIR / f"{cohort}_c{candidate}_predictions.npz", allow_pickle=False) as archive:
                predictions = {name: np.asarray(archive[name]) for name in archive.files}
            targets = confirmation_targets(spec, candidate)
            if targets.shape != (data["conf_history"].shape[0], 64):
                raise AssertionError(f"{cohort} c{candidate}: confirmation target shape mismatch")
            selection = selections[
                (selections["cohort"] == cohort)
                & (selections["candidate"].astype(str).str.zfill(2) == candidate)
            ].iloc[0]
            selected_family = str(selection["selected_family"])
            for half, bounds in (("A", (0, 32)), ("B", (32, 64))):
                half_targets = targets[:, bounds[0] : bounds[1]]
                losses = {
                    "direct": state_log_loss(data["conf_direct"], half_targets),
                    "composite": state_log_loss(data["conf_aligned"], half_targets),
                    "spline": state_log_loss(predictions["spline"], half_targets),
                    "tree": state_log_loss(predictions["tree"], half_targets),
                    "selected_history": state_log_loss(predictions["selected"], half_targets),
                }
                cell_records.append(
                    {
                        "cohort": cohort,
                        "implementation": spec.implementation,
                        "role": role,
                        "candidate": candidate,
                        "half": half,
                        "selected_family": selected_family,
                        "matrices": int(np.unique(data["conf_matrix"]).size),
                        "states": int(targets.shape[0]),
                        **{f"{name}_log_loss": float(value.mean()) for name, value in losses.items()},
                    }
                )
                for baseline in ("spline", "tree", "selected_history"):
                    gain_state = losses[baseline] - losses["composite"]
                    history_gain_state = losses["direct"] - losses[baseline]
                    interval = bootstrap_interval(
                        gain_state,
                        data["conf_matrix"],
                        derived_seed(cohort, candidate, half, baseline, "bootstrap"),
                    )
                    p_value = sign_randomization_p(
                        gain_state,
                        data["conf_matrix"],
                        derived_seed(cohort, candidate, half, baseline, "randomization"),
                    )
                    record = {
                        "cohort": cohort,
                        "implementation": spec.implementation,
                        "role": role,
                        "candidate": candidate,
                        "half": half,
                        "baseline": baseline,
                        "selected_family": selected_family,
                        "matrices": int(np.unique(data["conf_matrix"]).size),
                        "states": int(targets.shape[0]),
                        "baseline_log_loss": float(losses[baseline].mean()),
                        "composite_log_loss": float(losses["composite"].mean()),
                        "gain_nats": float(gain_state.mean()),
                        "ci95_lower": interval[0],
                        "ci95_upper": interval[1],
                        "randomization_p": p_value,
                        "history_minus_direct_gain": float(history_gain_state.mean()),
                        "holm_p": np.nan,
                    }
                    comparison_records.append(record)
                    if role == "primary" and baseline == "selected_history":
                        primary_positions.append(len(comparison_records) - 1)
                        primary_p_values.append(p_value)

    for position, adjusted in zip(primary_positions, holm_adjust(primary_p_values), strict=True):
        comparison_records[position]["holm_p"] = adjusted
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cell_records).to_csv(OUTPUT_DIR / "cell_results.csv", index=False)
    pd.DataFrame(comparison_records).to_csv(OUTPUT_DIR / "comparisons.csv", index=False)
    selections.to_csv(OUTPUT_DIR / "model_selection.csv", index=False)
    pd.read_csv(MODEL_DIR / "cv_fold_scores.csv").to_csv(
        OUTPUT_DIR / "cv_fold_scores.csv", index=False
    )
    atomic_json(
        OUTPUT_DIR / "analysis_manifest.json",
        {
            "protocol_id": protocol["protocol_id"],
            "confirmation_futures_generated": 0,
            "cells": len(cell_records),
            "comparisons": len(comparison_records),
            "primary_selected_history_tests": len(primary_positions),
        },
    )


def verify() -> None:
    protocol = json.loads((PROTOCOL_DIR / "protocol.json").read_text(encoding="utf-8"))
    selection_manifest = json.loads((MODEL_DIR / "selection_manifest.json").read_text(encoding="utf-8"))
    fit_manifest = json.loads((MODEL_DIR / "fit_manifest.json").read_text(encoding="utf-8"))
    analysis_manifest = json.loads((OUTPUT_DIR / "analysis_manifest.json").read_text(encoding="utf-8"))
    cells = pd.read_csv(OUTPUT_DIR / "cell_results.csv")
    comparisons = pd.read_csv(OUTPUT_DIR / "comparisons.csv")
    primary = comparisons[
        (comparisons["role"] == "primary")
        & (comparisons["baseline"] == "selected_history")
    ]
    checks = {
        "source_hashes_unchanged": all(
            sha256_file(Path(record["path"])) == record["sha256"]
            for record in protocol["sources"].values()
        ),
        "selection_protocol_identity": selection_manifest["protocol_id"] == protocol["protocol_id"],
        "fit_protocol_identity": fit_manifest["protocol_id"] == protocol["protocol_id"],
        "analysis_protocol_identity": analysis_manifest["protocol_id"] == protocol["protocol_id"],
        "selection_did_not_open_confirmation_outcomes": not selection_manifest["confirmation_outcomes_opened"],
        "no_confirmation_futures_generated": analysis_manifest["confirmation_futures_generated"] == 0,
        "sixteen_candidate_half_cells": len(cells) == 16,
        "eight_primary_selected_tests": len(primary) == 8,
        "all_primary_holm_values_present": bool(primary["holm_p"].notna().all()),
        "all_scores_finite": bool(np.isfinite(cells.select_dtypes(include=[np.number]).to_numpy()).all()),
        "prediction_probabilities_valid": True,
        "capacity_dimensions_matched": all(
            record["capacity_matched_inputs"] in (20, 21)
            for record in fit_manifest["models"].values()
        ),
    }
    for cohort in COHORTS:
        for candidate in ("02", "03"):
            with np.load(MODEL_DIR / f"{cohort}_c{candidate}_predictions.npz", allow_pickle=False) as archive:
                for name in archive.files:
                    values = np.asarray(archive[name])
                    checks["prediction_probabilities_valid"] &= bool(
                        np.isfinite(values).all() and (values > 0).all() and (values < 1).all()
                    )
    atomic_json(OUTPUT_DIR / "verification.json", checks)
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"verification failed: {failed}")


def report() -> None:
    write_reports(OUTPUT_DIR)


def checksums() -> None:
    files = sorted(
        path for path in ARTIFACTS.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    lines = [f"{sha256_file(path)}  {path.relative_to(TASK_ROOT)}" for path in files]
    (OUTPUT_DIR / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=("prepare", "select", "fit", "analyze", "verify", "report", "checksums", "all"),
    )
    args = parser.parse_args()
    stages = (
        ("prepare", prepare),
        ("select", select_models),
        ("fit", fit_models),
        ("analyze", analyze),
        ("verify", verify),
        ("report", report),
        ("checksums", checksums),
    )
    if args.stage == "all":
        for _, function in stages:
            function()
    else:
        dict(stages)[args.stage]()


if __name__ == "__main__":
    main()

