#!/usr/bin/env python3
"""Run the isolated matched-dimension nuisance-PCA reviewer control."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from adapters import (
    ARTIFACTS,
    COHORTS,
    REPLAY_DIR,
    atomic_npz,
    confirmation_targets,
    load_npz,
    prepare_codex_replay,
    prepare_fable_replay,
    source_contract,
)
from nuisance_core import (
    BOOTSTRAP_REPETITIONS,
    DERANGEMENT_REPETITIONS,
    MASTER_SEED,
    PCA_COMPONENTS,
    RANDOMIZATION_REPETITIONS,
    RIDGE_C,
    bootstrap_interval,
    canonical_digest,
    derived_seed,
    fit_composite,
    grouped_derangement,
    holm_adjust,
    predict_composite,
    sha256_file,
    sign_randomization_p,
    state_log_loss,
)
from reporting import write_reports


TASK_ROOT = Path(__file__).resolve().parent
PROTOCOL_DIR = ARTIFACTS / "protocol"
MODEL_DIR = ARTIFACTS / "models"
OUTPUT_DIR = ARTIFACTS / "output"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def runtime_contract() -> dict[str, str]:
    packages = ("numpy", "pandas", "scipy", "scikit-learn")
    return {
        "python": platform.python_version(),
        **{package: importlib.metadata.version(package) for package in packages},
    }


def protocol_contract() -> dict[str, Any]:
    body: dict[str, Any] = {
        "title": "Matched-dimension nuisance-PCA reviewer control",
        "date_frozen": "2026-08-19",
        "status": "reviewer-prompted post-hoc retained-outcome rescore",
        "question": (
            "Can history plus twelve fitted but row-misaligned state components reproduce "
            "the aligned composite's confirmation gain?"
        ),
        "cohorts": {key: spec.to_json() for key, spec in COHORTS.items()},
        "fixed_design": {
            "ridge_C": RIDGE_C,
            "state_components": PCA_COMPONENTS,
            "derangements": DERANGEMENT_REPETITIONS,
            "primary_derangement": 0,
            "development_pairing": "candidate- and generation-stratified Sattolo cycle",
            "confirmation_pairing": (
                "independent candidate- and landmark-stratified Sattolo cycle"
            ),
            "preserved": [
                "history variables",
                "development targets",
                "confirmation outcomes",
                "input dimension",
                "ridge model and C",
                "state-block multiset and covariance",
                "state scaler, PCA basis, and component marginals",
            ],
            "broken": "matrix-state alignment of the twelve state components",
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "randomization_repetitions": RANDOMIZATION_REPETITIONS,
            "multiplicity": (
                "Holm across aligned-versus-primary-nuisance tests in all 16 "
                "cohort/candidate/half cells"
            ),
            "master_seed_label": MASTER_SEED,
        },
        "interpretation_rule": (
            "The generic extra-capacity explanation is disfavored when "
            "aligned-minus-nuisance gain is positive with a positive whole-matrix "
            "interval, while nuisance gains remain materially below aligned gains "
            "across the 32 frozen pairings."
        ),
        "limitations": [
            "The confirmation outcomes and motivating reviewer concern were already known.",
            "This tests exact-marginal row-misaligned nuisance components, not every possible noise family.",
            "The originating L53/L54 machine-readable artifacts are unavailable.",
            (
                "Fable v2 is a 20-input 12+8 revision; the two headline cohorts "
                "and both Codex cohorts are exact 21-input 12+9 controls."
            ),
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
            raise RuntimeError("existing frozen protocol differs from current source contract")
    else:
        atomic_json(json_path, protocol)
    markdown = f"""# Frozen protocol: matched-dimension nuisance-PCA control

**Protocol ID:** `{protocol['protocol_id']}`  
**Frozen:** 2026-08-19  
**Status:** Reviewer-prompted post-hoc retained-outcome rescore

## Question

Can the registered history block plus twelve fitted but incorrectly aligned
state components reproduce the confirmation gain of the aligned composite?

## Fixed control

- Candidate and phase/generation remain separate.
- Within every phase group, a Sattolo cycle assigns each row the state block
  from a different matrix.
- Development and confirmation use independently derived frozen cycles.
- The history block and outcomes remain attached to their original rows.
- The state-block reassignment is an exact permutation, so its marginal
  distribution, covariance, development scaler, PCA-12 basis, dimensionality,
  and PCA-score marginals are exactly preserved.
- The final ridge is refit on development with `C=0.1`; no confirmation refit
  or recalibration is allowed.
- Replicate 0 is primary. Replicates 1-31 are fixed pairing sensitivity.
- Whole-matrix bootstrap and sign randomization use 4,096 draws. The 16
  aligned-versus-primary-nuisance cell tests receive Holm adjustment.

## Cohorts

- Codex headline: 40 development and 40 confirmation matrices, 12+9 inputs.
- Fable headline: 40 development and 40 confirmation matrices, 12+9 inputs.
- Codex scaled: 200 development and 200 confirmation matrices, 12+9 inputs.
- Fable v2: 1,000 development and 200 confirmation matrices, 12+8 inputs.

No confirmation futures are generated. Fable development main paths are
deterministically replayed only to recover the original development features
and targets. The originating L53/L54 artifacts cannot be included because the
required machine-readable files are absent.
"""
    markdown_path = PROTOCOL_DIR / "PROTOCOL.md"
    if not markdown_path.exists():
        atomic_text(markdown_path, markdown)
    print(f"protocol frozen: {protocol['protocol_id']}", flush=True)


def replay(workers: int) -> None:
    protocol = json.loads((PROTOCOL_DIR / "protocol.json").read_text(encoding="utf-8"))
    audits: dict[str, Any] = {"protocol_id": protocol["protocol_id"], "cohorts": {}}
    for key in ("codex_headline", "codex_primary"):
        print(f"replaying retained {key} features", flush=True)
        audits["cohorts"][key] = prepare_codex_replay(COHORTS[key])
    for key in ("fable_headline", "fable_primary"):
        print(f"reconstructing {key} development features", flush=True)
        audits["cohorts"][key] = prepare_fable_replay(COHORTS[key], workers)
    atomic_json(REPLAY_DIR / "replay_audit.json", audits)


def fit() -> None:
    protocol = json.loads((PROTOCOL_DIR / "protocol.json").read_text(encoding="utf-8"))
    manifest: dict[str, Any] = {"protocol_id": protocol["protocol_id"], "models": {}}
    for key, spec in COHORTS.items():
        for candidate in ("02", "03"):
            print(f"fitting {key} c{candidate}: aligned audit + 32 nuisance ridges", flush=True)
            data = load_npz(REPLAY_DIR / f"{key}_c{candidate}.npz")
            aligned_scaler, aligned_model = fit_composite(
                data["dev_history"],
                data["dev_components"],
                data["dev_targets"],
                spec.pipeline,
            )
            aligned_replay = predict_composite(
                data["conf_history"],
                data["conf_components"],
                aligned_scaler,
                aligned_model,
            )
            aligned_error = float(np.max(np.abs(aligned_replay - data["conf_aligned"])))
            if aligned_error > 5e-9:
                raise AssertionError(
                    f"{key} c{candidate}: aligned training replay error {aligned_error}"
                )

            predictions = np.empty(
                (DERANGEMENT_REPETITIONS, data["conf_history"].shape[0]),
                dtype=np.float64,
            )
            coefficients = np.empty(
                (DERANGEMENT_REPETITIONS, PCA_COMPONENTS + spec.history_dimension),
                dtype=np.float64,
            )
            intercepts = np.empty(DERANGEMENT_REPETITIONS, dtype=np.float64)
            n_iterations = np.empty(DERANGEMENT_REPETITIONS, dtype=np.int32)
            development_digests: list[str] = []
            confirmation_digests: list[str] = []

            for replicate in range(DERANGEMENT_REPETITIONS):
                dev_donor = grouped_derangement(
                    data["dev_matrix"],
                    data["dev_group"],
                    derived_seed(key, candidate, "development", replicate),
                )
                conf_donor = grouped_derangement(
                    data["conf_matrix"],
                    data["conf_group"],
                    derived_seed(key, candidate, "confirmation", replicate),
                )
                scaler, classifier = fit_composite(
                    data["dev_history"],
                    data["dev_components"][dev_donor],
                    data["dev_targets"],
                    spec.pipeline,
                )
                predictions[replicate] = predict_composite(
                    data["conf_history"],
                    data["conf_components"][conf_donor],
                    scaler,
                    classifier,
                )
                coefficients[replicate] = classifier.coef_[0]
                intercepts[replicate] = float(classifier.intercept_[0])
                n_iterations[replicate] = int(classifier.n_iter_.max())
                development_digests.append(
                    hashlib.sha256(np.ascontiguousarray(dev_donor).tobytes()).hexdigest()
                )
                confirmation_digests.append(
                    hashlib.sha256(np.ascontiguousarray(conf_donor).tobytes()).hexdigest()
                )

            atomic_npz(
                MODEL_DIR / f"{key}_c{candidate}_nuisance_predictions.npz",
                predictions=predictions,
                coefficients=coefficients,
                intercepts=intercepts,
                n_iterations=n_iterations,
                aligned_replay=aligned_replay,
            )
            manifest["models"][f"{key}_c{candidate}"] = {
                "aligned_prediction_max_abs_error": aligned_error,
                "development_permutation_sha256": development_digests,
                "confirmation_permutation_sha256": confirmation_digests,
                "all_development_pairings_fixed_point_free": True,
                "all_confirmation_pairings_fixed_point_free": True,
                "fitted_inputs": PCA_COMPONENTS + spec.history_dimension,
                "nuisance_models": DERANGEMENT_REPETITIONS,
                "aligned_model_iterations": int(aligned_model.n_iter_.max()),
                "max_nuisance_model_iterations": int(n_iterations.max()),
            }
    atomic_json(MODEL_DIR / "fit_manifest.json", manifest)


def analyze() -> None:
    protocol = json.loads((PROTOCOL_DIR / "protocol.json").read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    sensitivity: list[dict[str, Any]] = []
    p_values: list[float] = []

    for key, spec in COHORTS.items():
        for candidate in ("02", "03"):
            data = load_npz(REPLAY_DIR / f"{key}_c{candidate}.npz")
            model = load_npz(MODEL_DIR / f"{key}_c{candidate}_nuisance_predictions.npz")
            targets = confirmation_targets(spec, candidate)
            if targets.shape[0] != data["conf_history"].shape[0]:
                raise AssertionError(f"{key} c{candidate}: confirmation outcome alignment failed")

            for half, bounds in (("A", (0, 32)), ("B", (32, 64))):
                half_targets = targets[:, bounds[0] : bounds[1]]
                direct_loss = state_log_loss(data["conf_direct"], half_targets)
                aligned_loss = state_log_loss(data["conf_aligned"], half_targets)
                primary_loss = state_log_loss(model["predictions"][0], half_targets)
                aligned_gain_state = direct_loss - aligned_loss
                nuisance_gain_state = direct_loss - primary_loss
                aligned_minus_nuisance_state = primary_loss - aligned_loss
                aligned_ci = bootstrap_interval(
                    aligned_gain_state,
                    data["conf_matrix"],
                    derived_seed(key, candidate, half, "aligned_bootstrap"),
                )
                nuisance_ci = bootstrap_interval(
                    nuisance_gain_state,
                    data["conf_matrix"],
                    derived_seed(key, candidate, half, "nuisance_bootstrap"),
                )
                difference_ci = bootstrap_interval(
                    aligned_minus_nuisance_state,
                    data["conf_matrix"],
                    derived_seed(key, candidate, half, "difference_bootstrap"),
                )
                p_value = sign_randomization_p(
                    aligned_minus_nuisance_state,
                    data["conf_matrix"],
                    derived_seed(key, candidate, half, "difference_randomization"),
                )
                p_values.append(p_value)
                records.append(
                    {
                        "cohort": key,
                        "implementation": spec.implementation,
                        "role": spec.role,
                        "candidate": candidate,
                        "half": half,
                        "matrices": int(np.unique(data["conf_matrix"]).size),
                        "states": int(targets.shape[0]),
                        "fitted_inputs": PCA_COMPONENTS + spec.history_dimension,
                        "aligned_gain": float(aligned_gain_state.mean()),
                        "aligned_ci_low": aligned_ci[0],
                        "aligned_ci_high": aligned_ci[1],
                        "nuisance_gain": float(nuisance_gain_state.mean()),
                        "nuisance_ci_low": nuisance_ci[0],
                        "nuisance_ci_high": nuisance_ci[1],
                        "aligned_minus_nuisance": float(aligned_minus_nuisance_state.mean()),
                        "difference_ci_low": difference_ci[0],
                        "difference_ci_high": difference_ci[1],
                        "randomization_p": p_value,
                    }
                )
                for replicate in range(DERANGEMENT_REPETITIONS):
                    nuisance_loss = state_log_loss(model["predictions"][replicate], half_targets)
                    sensitivity.append(
                        {
                            "cohort": key,
                            "candidate": candidate,
                            "half": half,
                            "replicate": replicate,
                            "nuisance_gain": float((direct_loss - nuisance_loss).mean()),
                            "aligned_minus_nuisance": float(
                                (nuisance_loss - aligned_loss).mean()
                            ),
                        }
                    )

    for record, adjusted in zip(records, holm_adjust(p_values), strict=True):
        record["holm_p"] = adjusted
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(OUTPUT_DIR / "cell_results.csv", index=False)
    pd.DataFrame(sensitivity).to_csv(
        OUTPUT_DIR / "derangement_sensitivity.csv", index=False
    )
    atomic_json(
        OUTPUT_DIR / "analysis_manifest.json",
        {
            "protocol_id": protocol["protocol_id"],
            "cells": len(records),
            "sensitivity_rows": len(sensitivity),
            "confirmation_futures_generated": 0,
            "primary_derangement": 0,
        },
    )


def report() -> None:
    write_reports(OUTPUT_DIR)


def verify() -> None:
    protocol = json.loads((PROTOCOL_DIR / "protocol.json").read_text(encoding="utf-8"))
    replay_audit = json.loads((REPLAY_DIR / "replay_audit.json").read_text(encoding="utf-8"))
    fit_manifest = json.loads((MODEL_DIR / "fit_manifest.json").read_text(encoding="utf-8"))
    cells = pd.read_csv(OUTPUT_DIR / "cell_results.csv")
    sensitivity = pd.read_csv(OUTPUT_DIR / "derangement_sensitivity.csv")
    checks = {
        "source_hashes_unchanged": all(
            sha256_file(Path(record["path"])) == record["sha256"]
            for record in protocol["sources"].values()
        ),
        "protocol_identity_replay": replay_audit["protocol_id"] == protocol["protocol_id"],
        "protocol_identity_fit": fit_manifest["protocol_id"] == protocol["protocol_id"],
        "sixteen_cells": len(cells) == 16,
        "complete_sensitivity": len(sensitivity) == 16 * DERANGEMENT_REPETITIONS,
        "all_finite": bool(
            np.isfinite(cells.select_dtypes(include=[np.number]).to_numpy()).all()
            and np.isfinite(
                sensitivity.select_dtypes(include=[np.number]).to_numpy()
            ).all()
        ),
        "aligned_replays_match": all(
            float(record["aligned_prediction_max_abs_error"]) <= 5e-9
            for record in fit_manifest["models"].values()
        ),
        "dimensions_match_contract": all(
            int(record["fitted_inputs"])
            == PCA_COMPONENTS + COHORTS[name.rsplit("_c", 1)[0]].history_dimension
            for name, record in fit_manifest["models"].items()
        ),
        "no_confirmation_futures": json.loads(
            (OUTPUT_DIR / "analysis_manifest.json").read_text(encoding="utf-8")
        )["confirmation_futures_generated"]
        == 0,
        "required_outputs_present": all(
            path.is_file()
            for path in (
                PROTOCOL_DIR / "PROTOCOL.md",
                OUTPUT_DIR / "RESULTS_REPORT.md",
                OUTPUT_DIR / "SUGGESTED_TEXT.md",
                OUTPUT_DIR / "cell_results.csv",
                OUTPUT_DIR / "derangement_sensitivity.csv",
            )
        ),
    }
    verification = {
        "protocol_id": protocol["protocol_id"],
        "checks": checks,
        "passed": bool(all(checks.values())),
    }
    atomic_json(OUTPUT_DIR / "verification.json", verification)
    if not verification["passed"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"verification failed: {failed}")

    targets = sorted(
        path
        for path in ARTIFACTS.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    lines = [f"{sha256_file(path)}  {path.relative_to(ARTIFACTS)}" for path in targets]
    atomic_text(OUTPUT_DIR / "SHA256SUMS", "\n".join(lines) + "\n")
    print(f"verification passed: {len(checks)}/{len(checks)} checks", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=("prepare", "replay", "fit", "analyze", "report", "verify", "all"),
    )
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 12))
    arguments = parser.parse_args()
    stages: tuple[tuple[str, Callable[[], None]], ...] = (
        ("prepare", prepare),
        ("replay", lambda: replay(max(1, arguments.workers))),
        ("fit", fit),
        ("analyze", analyze),
        ("report", report),
        ("verify", verify),
    )
    for name, function in stages:
        if arguments.stage in (name, "all"):
            print(f"[{name}]", flush=True)
            function()


if __name__ == "__main__":
    main()
