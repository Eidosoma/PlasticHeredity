from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Iterable

# Permit direct-script and module invocation while keeping caches isolated.
TASK_ROOT = Path(__file__).resolve().parent
CODEX_ROOT = TASK_ROOT.parent
if __package__ in (None, ""):
    sys.path.insert(0, str(CODEX_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(TASK_ROOT / "artifacts" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from reviewer_sequence_history_response.implementation_adapters import (
    COHORTS,
    REPLAY_ROOT,
    CohortSpec,
    atomic_npz,
    development_audit,
    load_confirmation_outcomes,
    load_confirmation_replay,
    load_development_replay,
    replay_cohort,
    source_contract,
)
from reviewer_sequence_history_response.sequence_core import (
    BOOTSTRAP_REPETITIONS,
    C_GRID,
    CV_FOLDS,
    HORIZON,
    LAG_GRID,
    RANDOMIZATION_REPETITIONS,
    RENEWAL_RUN,
    THRESHOLD,
    LaggedRidgeModel,
    TransitionModel,
    brier_score,
    branch_losses,
    canonical_digest,
    fit_lagged_ridge,
    fit_transition_model,
    holm_adjust,
    paired_matrix_inference,
    rank_metrics,
    sha256_file,
    state_branch_log_loss,
    transition_predictions,
)


ARTIFACT_ROOT = TASK_ROOT / "artifacts"
PROTOCOL_ROOT = ARTIFACT_ROOT / "protocol"
MODEL_ROOT = ARTIFACT_ROOT / "models"
OUTPUT_ROOT = ARTIFACT_ROOT / "output"
PROTOCOL_PATH = PROTOCOL_ROOT / "protocol.json"
FIT_MANIFEST_PATH = MODEL_ROOT / "fit_manifest.json"
ANALYSIS_MANIFEST_PATH = OUTPUT_ROOT / "analysis_manifest.json"
FABLE_ROOT = CODEX_ROOT.parent / "replicators.13.8.2026.fable" / "replication"


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _runtime() -> dict[str, str]:
    packages = ("numpy", "scipy", "pandas", "scikit-learn", "matplotlib", "threadpoolctl")
    return {
        "python": platform.python_version(),
        **{package: importlib.metadata.version(package) for package in packages},
    }


def _analysis_code_contract() -> dict[str, dict[str, str]]:
    paths = {
        "runner": Path(__file__),
        "core": TASK_ROOT / "sequence_core.py",
        "adapters": TASK_ROOT / "implementation_adapters.py",
        "review_plan": TASK_ROOT / "REVIEW_AND_PLAN.md",
    }
    return {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for name, path in paths.items()
    }


def _protocol_value() -> dict[str, Any]:
    value: dict[str, Any] = {
        "format": "reviewer-sequence-history-protocol-v1",
        "status": "reviewer_prompted_post_hoc_rescore",
        "working_boundary": {
            "all_new_writes_below": str(TASK_ROOT.resolve()),
            "read_only_sources": True,
            "new_confirmation_futures": False,
            "manuscript_edit": False,
        },
        "protocol_amendment": {
            "id": "001",
            "timing": "before model fitting or confirmation-outcome loading",
            "change": "exclude the independently edited manuscript from analysis-input hash enforcement",
            "scientific_contract_changed": False,
        },
        "excluded_non_inputs": [
            "PRE_PRINT_PAPER_DRAFT.md (never read by replay, fitting, scoring, or verification)"
        ],
        "cohorts": {key: spec.to_json() for key, spec in COHORTS.items()},
        "endpoint": {
            "inheritance": f"strict unrounded H > {THRESHOLD}",
            "horizon": HORIZON,
            "event": "a future break followed strictly later by three consecutive inherited fissions",
            "renewal_run": RENEWAL_RUN,
            "extinction": "absorbing negative unless certification already occurred",
        },
        "models": {
            "markov": "P(next outcome | latest inheritance state)",
            "semimarkov": "P(next outcome | latest state, duration bin 1,2,3,4,5+)",
            "transition_outcomes": ["break", "inherit", "terminal"],
            "transition_smoothing": "symmetric Dirichlet(1,1,1)",
            "transition_to_f12": "exact dynamic programming at launch",
            "lagged_primary": {
                "base": "cohort's registered direct-history columns",
                "ordered_features_per_lag": ["continuous_H", "strict_H_indicator", "observed_mask"],
                "lag_grid": LAG_GRID,
                "c_grid": C_GRID,
                "folds": CV_FOLDS,
                "grouping": "development catalytic matrix",
                "classifier": "unweighted L2 logistic with intercept, fold-fitted standardization",
                "selection": "minimum mean fold branch log loss; ties smaller lag then smaller C",
            },
        },
        "scoring": {
            "primary": "branch log loss in natural nats",
            "secondary": ["branch Brier", "state-level overall Spearman", "matrix-centered Spearman"],
            "unit": "one retained stochastic future",
            "candidates_separate": True,
            "branch_halves_separate": True,
        },
        "inference": {
            "cluster": "catalytic matrix",
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "randomization_repetitions": RANDOMIZATION_REPETITIONS,
            "randomization": "paired matrix-sign, one-sided positive gain",
            "family": "eight primary composite-versus-lagged cells",
            "multiplicity": "Holm",
            "strong_gate": "all eight gains > 0, bootstrap lower95 > 0, Holm p < 0.05",
            "secondary_headline_rescue": False,
        },
        "originating_l53_l54": {
            "included": False,
            "reason": "machine-readable state, prediction, and branch artifacts absent",
        },
        "runtime_at_freeze": _runtime(),
        "analysis_code": _analysis_code_contract(),
        "read_only_sources": source_contract(),
    }
    normalized = _json_ready(value)
    normalized["protocol_id"] = canonical_digest(normalized)
    return normalized


def prepare() -> None:
    PROTOCOL_ROOT.mkdir(parents=True, exist_ok=True)
    protocol = _protocol_value()
    if PROTOCOL_PATH.is_file():
        existing = _read_json(PROTOCOL_PATH)
        if existing != protocol:
            raise RuntimeError(
                "an existing frozen protocol differs from the current contract; "
                "it will not be overwritten"
            )
        print(f"protocol already frozen and identical: {protocol['protocol_id']}")
        return
    _write_json(PROTOCOL_PATH, protocol)
    (PROTOCOL_ROOT / "PROTOCOL.md").write_text(
        "# Frozen sequence-history analysis protocol\n\n"
        f"Protocol ID: `{protocol['protocol_id']}`\n\n"
        "The complete machine-readable contract is in `protocol.json`. It was "
        "written before sequence-model confirmation scoring. Existing artifacts "
        "are read-only and no new confirmation futures are permitted.\n",
        encoding="utf-8",
    )
    print(f"frozen protocol: {protocol['protocol_id']}")


def _protocol() -> dict[str, Any]:
    if not PROTOCOL_PATH.is_file():
        raise RuntimeError("run prepare before this stage")
    protocol = _read_json(PROTOCOL_PATH)
    if canonical_digest({key: value for key, value in protocol.items() if key != "protocol_id"}) != protocol["protocol_id"]:
        raise AssertionError("frozen protocol digest is invalid")
    return protocol


def _selected_specs(dataset: str) -> list[CohortSpec]:
    if dataset == "all":
        keys = ("codex_primary", "fable_primary", "codex_headline", "fable_headline")
    elif dataset == "primary":
        keys = ("codex_primary", "fable_primary")
    elif dataset == "headline":
        keys = ("codex_headline", "fable_headline")
    elif dataset in COHORTS:
        keys = (dataset,)
    else:
        raise ValueError(dataset)
    return [COHORTS[key] for key in keys]


def replay(dataset: str, workers: int) -> None:
    protocol = _protocol()
    audit: dict[str, Any] = {
        "protocol_id": protocol["protocol_id"],
        "confirmation_futures_generated": False,
        "cohorts": {},
    }

    def progress(cohort: str, completed: int, total: int) -> None:
        print(f"[{cohort}] replay checkpoints {completed}/{total}", flush=True)

    for spec in _selected_specs(dataset):
        print(f"replaying natural paths: {spec.key}", flush=True)
        audit["cohorts"][spec.key] = replay_cohort(
            spec,
            protocol_id=protocol["protocol_id"],
            workers=workers,
            progress=progress,
        )
    existing = REPLAY_ROOT / "replay_manifest.json"
    if existing.is_file():
        previous = _read_json(existing)
        previous.setdefault("cohorts", {}).update(audit["cohorts"])
        audit = previous
    _write_json(existing, audit)
    print(f"replay manifest: {existing}")


def _transition_path(spec: CohortSpec, candidate: str) -> Path:
    return MODEL_ROOT / f"{spec.key}_c{candidate}_transitions.json"


def _lag_model_path(spec: CohortSpec, candidate: str) -> Path:
    return MODEL_ROOT / f"{spec.key}_c{candidate}_lagged_ridge.npz"


def fit() -> None:
    protocol = _protocol()
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "format": "reviewer-sequence-history-fitted-models-v1",
        "protocol_id": protocol["protocol_id"],
        "confirmation_outcomes_loaded": False,
        "cohorts": {},
    }
    cv_rows: list[dict[str, Any]] = []
    with threadpool_limits(limits=1):
        for spec in _selected_specs("all"):
            print(f"fitting development-only models: {spec.key}", flush=True)
            development = load_development_replay(spec)
            audit = development_audit(spec, development)
            cohort_models: dict[str, Any] = {"development_audit": audit, "candidates": {}}
            for candidate in ("02", "03"):
                state_selected = development["candidate"] == candidate
                trajectory_selected = development["trajectory_candidate"] == candidate
                markov = fit_transition_model(
                    development["trajectory_h"][trajectory_selected],
                    development["trajectory_length"][trajectory_selected],
                    development["trajectory_died"][trajectory_selected].astype(bool),
                    duration_aware=False,
                )
                semimarkov = fit_transition_model(
                    development["trajectory_h"][trajectory_selected],
                    development["trajectory_length"][trajectory_selected],
                    development["trajectory_died"][trajectory_selected].astype(bool),
                    duration_aware=True,
                )
                _write_json(
                    _transition_path(spec, candidate),
                    {
                        "protocol_id": protocol["protocol_id"],
                        "cohort": spec.key,
                        "candidate": candidate,
                        "markov": markov.to_json(),
                        "semimarkov": semimarkov.to_json(),
                    },
                )
                model, candidate_cv = fit_lagged_ridge(
                    development["direct"][state_selected],
                    development["history_h"][state_selected],
                    development["history_length"][state_selected],
                    development["targets"][state_selected],
                    development["matrix_id"][state_selected],
                    direct_columns=spec.direct_columns,
                )
                atomic_npz(
                    _lag_model_path(spec, candidate),
                    protocol_id=np.asarray([protocol["protocol_id"]]),
                    cohort=np.asarray([spec.key]),
                    candidate=np.asarray([candidate]),
                    **model.arrays(),
                )
                for row in candidate_cv:
                    cv_rows.append(
                        {
                            "cohort": spec.key,
                            "role": spec.role,
                            "implementation": spec.implementation,
                            "candidate": candidate,
                            "selected": row["lag"] == model.lag and row["c_value"] == model.c_value,
                            "lag": row["lag"],
                            "c_value": row["c_value"],
                            "mean_log_loss": row["mean_log_loss"],
                            **{
                                f"fold_{index + 1}_log_loss": value
                                for index, value in enumerate(row["fold_losses"])
                            },
                        }
                    )
                cohort_models["candidates"][candidate] = {
                    "selected_lag": model.lag,
                    "selected_c": model.c_value,
                    "features": int(model.coefficient.size),
                    "transition_path": str(_transition_path(spec, candidate).relative_to(TASK_ROOT)),
                    "lag_model_path": str(_lag_model_path(spec, candidate).relative_to(TASK_ROOT)),
                }
                print(
                    f"  c{candidate}: lag={model.lag}, C={model.c_value:g}, "
                    f"features={model.coefficient.size}",
                    flush=True,
                )
            manifest["cohorts"][spec.key] = cohort_models
    pd.DataFrame(cv_rows).to_csv(MODEL_ROOT / "model_selection.csv", index=False)
    manifest["model_selection_sha256"] = sha256_file(MODEL_ROOT / "model_selection.csv")
    _write_json(FIT_MANIFEST_PATH, manifest)
    print(f"fit manifest: {FIT_MANIFEST_PATH}")


def _load_models(
    spec: CohortSpec, candidate: str, protocol_id: str
) -> tuple[TransitionModel, TransitionModel, LaggedRidgeModel]:
    transition = _read_json(_transition_path(spec, candidate))
    if transition["protocol_id"] != protocol_id:
        raise AssertionError("transition model protocol mismatch")
    with np.load(_lag_model_path(spec, candidate), allow_pickle=False) as archive:
        if str(archive["protocol_id"][0]) != protocol_id:
            raise AssertionError("lag model protocol mismatch")
        lagged = LaggedRidgeModel.from_archive(archive)
    return (
        TransitionModel.from_json(transition["markov"]),
        TransitionModel.from_json(transition["semimarkov"]),
        lagged,
    )


def _seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def analyze() -> None:
    protocol = _protocol()
    if not FIT_MANIFEST_PATH.is_file():
        raise RuntimeError("run fit before analyze")
    fit_manifest = _read_json(FIT_MANIFEST_PATH)
    if fit_manifest["protocol_id"] != protocol["protocol_id"]:
        raise AssertionError("fit manifest protocol mismatch")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    score_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    replay_audit: dict[str, Any] = {}
    for spec in _selected_specs("all"):
        print(f"scoring retained confirmation outcomes: {spec.key}", flush=True)
        confirmation = load_confirmation_replay(spec)
        # This is the first stage allowed to load retained confirmation futures.
        retained = load_confirmation_outcomes(spec)
        if retained["targets"].shape[0] != confirmation["direct"].shape[0]:
            raise AssertionError("confirmation target/state mismatch")
        predictions: dict[str, NDArray[np.float64]] = {
            "direct": retained["prediction_direct"].copy(),
            "composite": retained["prediction_composite"].copy(),
            "markov": np.empty(confirmation["direct"].shape[0], dtype=np.float64),
            "semimarkov": np.empty(confirmation["direct"].shape[0], dtype=np.float64),
            "lagged": np.empty(confirmation["direct"].shape[0], dtype=np.float64),
        }
        for candidate in ("02", "03"):
            selected = confirmation["candidate"] == candidate
            markov, semimarkov, lagged = _load_models(
                spec, candidate, protocol["protocol_id"]
            )
            predictions["markov"][selected] = transition_predictions(
                markov,
                confirmation["history_h"][selected],
                confirmation["history_length"][selected],
            )
            predictions["semimarkov"][selected] = transition_predictions(
                semimarkov,
                confirmation["history_h"][selected],
                confirmation["history_length"][selected],
            )
            predictions["lagged"][selected] = lagged.predict(
                confirmation["direct"][selected],
                confirmation["history_h"][selected],
                confirmation["history_length"][selected],
            )
        atomic_npz(
            OUTPUT_ROOT / f"predictions_{spec.key}.npz",
            protocol_id=np.asarray([protocol["protocol_id"]]),
            state_keys=confirmation["state_keys"],
            candidate=confirmation["candidate"],
            matrix_id=confirmation["matrix_id"],
            landmark=confirmation["landmark"],
            targets=retained["targets"].astype(np.int8),
            **{f"prediction_{name}": values for name, values in predictions.items()},
        )
        replay_audit[spec.key] = {
            "states": int(confirmation["direct"].shape[0]),
            "targets": int(retained["targets"].size),
            "direct_feature_max_abs_error": float(
                np.max(np.abs(confirmation["direct"] - retained["source_direct"]))
            ),
            "confirmation_future_source": "retained artifact",
            "new_confirmation_futures": False,
        }
        for candidate in ("02", "03"):
            selected = confirmation["candidate"] == candidate
            candidate_targets = retained["targets"][selected]
            matrices = confirmation["matrix_id"][selected]
            for half, branch_slice in (("A", slice(0, 32)), ("B", slice(32, 64))):
                y_half = candidate_targets[:, branch_slice]
                for model_name, probability in predictions.items():
                    p = probability[selected]
                    overall_rank, centered_rank = rank_metrics(p, y_half, matrices)
                    score_rows.append(
                        {
                            "cohort": spec.key,
                            "implementation": spec.implementation,
                            "role": spec.role,
                            "candidate": candidate,
                            "half": half,
                            "model": model_name,
                            "log_loss_nats": state_branch_log_loss(y_half, p),
                            "brier": brier_score(y_half, p),
                            "overall_spearman": overall_rank,
                            "matrix_centered_spearman": centered_rank,
                            "states": int(selected.sum()),
                            "branches": int(y_half.size),
                        }
                    )
                for baseline_name in ("direct", "markov", "semimarkov", "lagged"):
                    result = paired_matrix_inference(
                        y_half,
                        predictions[baseline_name][selected],
                        predictions["composite"][selected],
                        matrices,
                        repetitions=BOOTSTRAP_REPETITIONS,
                        seed=_seed(protocol["protocol_id"], spec.key, candidate, half, baseline_name),
                    )
                    comparison_rows.append(
                        {
                            "cohort": spec.key,
                            "implementation": spec.implementation,
                            "role": spec.role,
                            "candidate": candidate,
                            "half": half,
                            "baseline": baseline_name,
                            "challenger": "composite",
                            "gain_nats": result["gain_nats"],
                            "ci95_lower": result["ci95"][0],
                            "ci95_upper": result["ci95"][1],
                            "randomization_p_one_sided": result["randomization_p_one_sided"],
                            "matrices": result["matrices"],
                            "branches": result["branches"],
                        }
                    )
    scores = pd.DataFrame(score_rows)
    comparisons = pd.DataFrame(comparison_rows)
    archived_score_audit: dict[str, Any] = {}
    for spec in _selected_specs("all"):
        cohort_scores = scores[scores["cohort"] == spec.key]
        archived_score_audit[spec.key] = {}
        if spec.key.startswith("codex"):
            stored = _read_json(Path(spec.source_directory) / "metrics.json")
            for candidate in ("02", "03"):
                errors = []
                for half in ("A", "B"):
                    cell = cohort_scores[
                        (cohort_scores["candidate"] == candidate)
                        & (cohort_scores["half"] == half)
                    ].set_index("model")
                    errors.extend(
                        [
                            abs(float(cell.loc["direct", "log_loss_nats"]) - stored[candidate]["directions"][half]["log_loss_history"]),
                            abs(float(cell.loc["composite", "log_loss_nats"]) - stored[candidate]["directions"][half]["log_loss_full"]),
                        ]
                    )
                maximum = max(errors)
                if maximum > 2e-12:
                    raise AssertionError(f"{spec.key} c{candidate}: archived score mismatch {maximum}")
                archived_score_audit[spec.key][candidate] = {
                    "max_abs_log_loss_error": maximum,
                    "tolerance": 2e-12,
                    "exact_retained_branch_targets": True,
                }
        elif spec.key == "fable_headline":
            stored = _read_json(FABLE_ROOT / "results" / "confirmation_metrics.json")
            for candidate in ("02", "03"):
                cell = cohort_scores[cohort_scores["candidate"] == candidate]
                direct = float(cell[cell["model"] == "direct"]["log_loss_nats"].mean())
                composite = float(cell[cell["model"] == "composite"]["log_loss_nats"].mean())
                maximum = max(
                    abs(direct - stored[candidate]["logloss"]["direct"]),
                    abs(composite - stored[candidate]["logloss"]["full"]),
                )
                if maximum > 2e-12:
                    raise AssertionError(f"{spec.key} c{candidate}: archived score mismatch {maximum}")
                archived_score_audit[spec.key][candidate] = {
                    "max_abs_log_loss_error": maximum,
                    "tolerance": 2e-12,
                    "exact_retained_branch_targets": True,
                }
        else:
            stored = _read_json(FABLE_ROOT / "results_v2" / "v2_results.json")
            for candidate in ("02", "03"):
                cell = cohort_scores[cohort_scores["candidate"] == candidate]
                direct = float(cell[cell["model"] == "direct"]["log_loss_nats"].mean())
                composite = float(cell[cell["model"] == "composite"]["log_loss_nats"].mean())
                maximum = max(
                    abs(direct - stored[candidate]["direct8"]["logloss"]),
                    abs(composite - stored[candidate]["v2"]["logloss"]),
                )
                # The persisted H64 array is float32; candidate 02 has a tiny
                # near-threshold reconstruction difference from the original
                # in-memory float64 branch flags.  It is retained and disclosed.
                if maximum > 2e-5:
                    raise AssertionError(f"{spec.key} c{candidate}: archived score mismatch {maximum}")
                archived_score_audit[spec.key][candidate] = {
                    "max_abs_log_loss_error": maximum,
                    "tolerance": 2e-5,
                    "target_reconstruction": "retained float32 H64 and retained branch lengths",
                    "new_confirmation_futures": False,
                }
    replay_audit["archived_score_reproduction"] = archived_score_audit
    family = (comparisons["role"] == "primary") & (comparisons["baseline"] == "lagged")
    if int(family.sum()) != 8:
        raise AssertionError(f"expected 8 primary lagged comparisons, got {family.sum()}")
    comparisons["holm_p"] = np.nan
    comparisons.loc[family, "holm_p"] = holm_adjust(
        comparisons.loc[family, "randomization_p_one_sided"].tolist()
    )
    comparisons["cell_pass"] = False
    comparisons.loc[family, "cell_pass"] = (
        (comparisons.loc[family, "gain_nats"] > 0)
        & (comparisons.loc[family, "ci95_lower"] > 0)
        & (comparisons.loc[family, "holm_p"] < 0.05)
    )
    strong_gate = bool(comparisons.loc[family, "cell_pass"].all())
    scores.to_csv(OUTPUT_ROOT / "scores.csv", index=False)
    comparisons.to_csv(OUTPUT_ROOT / "comparisons.csv", index=False)
    _write_json(OUTPUT_ROOT / "replay_audit.json", replay_audit)
    manifest = {
        "format": "reviewer-sequence-history-analysis-v1",
        "protocol_id": protocol["protocol_id"],
        "confirmation_outcomes_loaded_only_in_analyze": True,
        "strong_cross_clean_room_gate": strong_gate,
        "primary_cells_passed": int(comparisons.loc[family, "cell_pass"].sum()),
        "primary_cells_total": 8,
        "scores_sha256": sha256_file(OUTPUT_ROOT / "scores.csv"),
        "comparisons_sha256": sha256_file(OUTPUT_ROOT / "comparisons.csv"),
    }
    _write_json(ANALYSIS_MANIFEST_PATH, manifest)
    print(
        f"analysis complete: primary cells {manifest['primary_cells_passed']}/8; "
        f"strong gate={strong_gate}"
    )


def _format_number(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def _markdown_table(rows: Iterable[Iterable[Any]], headers: Iterable[str]) -> str:
    header = list(headers)
    values = [list(row) for row in rows]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in values)
    return "\n".join(lines)


def _make_figure(primary: pd.DataFrame) -> None:
    figure_rows = primary.sort_values(["implementation", "candidate", "half"])
    labels = [
        f"{'IT1' if 'test_1' in row.implementation else 'IT2'} c{row.candidate}{row.half}"
        for row in figure_rows.itertuples()
    ]
    gains = figure_rows["gain_nats"].to_numpy()
    lower = gains - figure_rows["ci95_lower"].to_numpy()
    upper = figure_rows["ci95_upper"].to_numpy() - gains
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    colors = ["#4878A8" if "test_1" in value else "#D17A22" for value in figure_rows["implementation"]]
    ax.errorbar(
        np.arange(len(gains)),
        gains,
        yerr=np.vstack((lower, upper)),
        fmt="none",
        ecolor="#33383D",
        capsize=3,
        linewidth=1.1,
    )
    ax.scatter(np.arange(len(gains)), gains, c=colors, s=38, zorder=3)
    ax.axhline(0.0, color="#33383D", linestyle="--", linewidth=0.9)
    ax.set_xticks(np.arange(len(gains)), labels, rotation=35, ha="right")
    ax.set_ylabel("Log-loss gain over lagged history (nats/future)")
    ax.set_title("Frozen composite versus development-selected sequence history")
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "figure_sequence_baseline_gain.png", dpi=180)
    fig.savefig(OUTPUT_ROOT / "figure_sequence_baseline_gain.pdf")
    plt.close(fig)


def report() -> None:
    protocol = _protocol()
    if not ANALYSIS_MANIFEST_PATH.is_file():
        raise RuntimeError("run analyze before report")
    manifest = _read_json(ANALYSIS_MANIFEST_PATH)
    scores = pd.read_csv(OUTPUT_ROOT / "scores.csv")
    comparisons = pd.read_csv(OUTPUT_ROOT / "comparisons.csv")
    selection = pd.read_csv(MODEL_ROOT / "model_selection.csv")
    selected = selection[selection["selected"].astype(bool)].copy()
    primary = comparisons[(comparisons["role"] == "primary") & (comparisons["baseline"] == "lagged")].copy()
    _make_figure(primary)

    if manifest["strong_cross_clean_room_gate"]:
        finding = (
            "The frozen composite retained a statistically supported log-loss advantage over "
            "the development-selected ordered-history ridge in every candidate, branch half, "
            "and clean-room implementation."
        )
        interpretation = (
            "This supports incremental predictive content beyond the tested ordered sequence "
            "history, while remaining a post-hoc robustness result rather than a new prospective test."
        )
    else:
        finding = (
            f"The strict cross-clean-room gate did not pass: {manifest['primary_cells_passed']} "
            "of 8 primary candidate-by-half cells met all three criteria."
        )
        interpretation = (
            "The manuscript should retain the narrow claim against the registered direct ridge "
            "and report the sequence comparison explicitly; failed or mixed cells cannot be rescued "
            "by pooling or by the secondary headline cohorts."
        )

    primary_rows = []
    for row in primary.sort_values(["implementation", "candidate", "half"]).itertuples():
        primary_rows.append(
            [
                "IT1 Codex" if "test_1" in row.implementation else "IT2 Fable",
                row.candidate,
                row.half,
                _format_number(row.gain_nats, 5),
                f"[{_format_number(row.ci95_lower, 5)}, {_format_number(row.ci95_upper, 5)}]",
                _format_number(row.randomization_p_one_sided, 5),
                _format_number(row.holm_p, 5),
                "PASS" if bool(row.cell_pass) else "NO",
            ]
        )
    selection_rows = [
        [
            row.cohort,
            row.candidate,
            int(row.lag),
            _format_number(row.c_value, 2),
            _format_number(row.mean_log_loss, 5),
        ]
        for row in selected.sort_values(["cohort", "candidate"]).itertuples()
    ]
    score_pivot = scores[
        (scores["role"] == "primary") & scores["model"].isin(["direct", "markov", "semimarkov", "lagged", "composite"])
    ]
    score_rows = [
        [
            row.cohort,
            row.candidate,
            row.half,
            row.model,
            _format_number(row.log_loss_nats, 5),
            _format_number(row.brier, 5),
            _format_number(row.overall_spearman, 3),
            _format_number(row.matrix_centered_spearman, 3),
        ]
        for row in score_pivot.sort_values(["cohort", "candidate", "half", "model"]).itertuples()
    ]
    report_text = f"""# Sequence-history comparator analysis report

**Status:** Reviewer-prompted post-hoc rescore.  No new confirmation futures,
model recalibration, manuscript edit, or pooling across candidates/halves was
performed.  Protocol ID: `{protocol['protocol_id']}`.

## Executive finding

{finding}

{interpretation}

The Appendix C transition gains are not numerically commensurate with this
analysis: Appendix C scores bits per realized post-break transition, whereas
this report scores nats per complete F12 future from information available at
launch.

## Primary composite-versus-lagged comparison

{_markdown_table(primary_rows, ['Implementation', 'Candidate', 'Half', 'Gain (nats)', '95% matrix CI', 'Raw p', 'Holm p', 'Cell gate'])}

Positive gain favors the frozen composite.  The strong claim requires all eight
cells to have positive gain, positive CI lower bound, and Holm-adjusted
`p < 0.05`.

## Development-only model selection

{_markdown_table(selection_rows, ['Cohort', 'Candidate', 'Selected lags', 'C', 'CV log loss'])}

Each lag supplies continuous H, strict-H status, and an observation mask, in
addition to that cohort's registered direct-history variables.  Selection used
five-fold development-matrix-grouped cross-validation only.

## Primary score inventory

{_markdown_table(score_rows, ['Cohort', 'Candidate', 'Half', 'Model', 'Log loss', 'Brier', 'Spearman', 'Centered Spearman'])}

Markov and semi-Markov probabilities were estimated from natural development
paths with an absorbing terminal outcome and integrated exactly over the F12
event.  They are diagnostics motivated by Appendix C; the ordered lagged ridge
is the predeclared primary sequence comparator.

## Secondary and scope qualifications

- The matched 40-matrix headline results are retained in `scores.csv` and
  `comparisons.csv` as secondary replication checks; they cannot rescue the
  primary gate.
- Independent test 1 retains its registered H9 direct block.  Independent test
  2 v2 retains its deduplicated H8 block.  The analysis does not silently force
  one clean room into the other's representation.
- The originating L53/L54 workflow is excluded because its state-level frozen
  predictions and branch outcomes are not present locally.
- Fable v2 did not retain its original float64 branch flags.  Its outcomes were
  reconstructed from the retained float32 H64 arrays and branch lengths.  The
  resulting archived-score discrepancy is at most the value recorded in
  `replay_audit.json` (tolerance `2e-5` nats); no confirmation branch was
  resimulated to remove that representation-level discrepancy.
- This is a robustness rescore on already-observed outcomes, not a new untouched
  confirmation.

## Reproducibility

The isolated folder contains the frozen protocol, per-matrix replay checkpoints,
development-fitted models, complete CV audit, retained-outcome prediction files,
matrix-aware inference, replay audit, figure, verification report, and SHA-256
manifest.  All source artifacts were read-only.
"""
    (OUTPUT_ROOT / "SEQUENCE_HISTORY_ANALYSIS_REPORT.md").write_text(report_text, encoding="utf-8")

    if manifest["strong_cross_clean_room_gate"]:
        proposed = """# Proposed manuscript and reviewer-response language

## Methods addition

As a reviewer-prompted post-hoc robustness analysis, we compared the frozen F12
composite with development-fitted sequence-history models.  Candidate-specific
first-order and duration-aware transition laws were integrated over the F12
endpoint before observing each retained confirmation future.  A stronger
history comparator augmented the registered direct variables with ordered
pre-launch continuous H values, strict-H indicators, and padding masks; lag
length and ridge strength were selected by development-matrix-grouped
cross-validation.  All models were then scored without recalibration on the
already-observed confirmation branches, separately by candidate and frozen
half.

## Results addition

The frozen composite improved branch log loss over the selected ordered-history
model in all eight implementation-by-candidate-by-half cells; every
whole-matrix 95% interval excluded zero and every Holm-adjusted paired
randomization p value was below 0.05.  Thus, within the tested model family, the
composite advantage was not explained solely by first-order, duration-aware, or
ordered recent inheritance history.  Because this analysis was prompted after
the confirmation results existed, it is supportive post-hoc evidence rather
than a new prospective confirmation.

## Reviewer response

We agree that Appendix C motivated a stronger launch-time history baseline.  We
added both generative Markov/semi-Markov controls and an ordered-history ridge,
fit exclusively on development matrices.  These were converted to or directly
estimated as F12 launch probabilities and rescored on the existing confirmation
outcomes; no new futures were generated.  The frozen composite retained its
advantage in every primary clean-room cell.  We also clarified that Appendix C
uses bits per transition after part of the future is observed, whereas the
headline task uses nats per complete future at launch.
"""
    else:
        proposed = """# Proposed manuscript and reviewer-response language

## Predictor discussion/limitation addition

The registered comparison establishes improvement over its frozen direct-history
ridge, not over every possible history-only predictor.  In a reviewer-prompted
post-hoc rescore, we fitted first-order, duration-aware, and ordered-prefix
history models on development matrices and evaluated them on the already-observed
confirmation outcomes.  The strict cross-clean-room sequence-baseline gate did
not pass uniformly.  Current composition and catalytic context may therefore
encode aspects of ordered history omitted by the registered scalar summary, and
the source of the composite's predictive advantage remains unresolved.

## Reviewer response

We agree that Appendix C motivates a stronger history baseline.  We performed
the requested retained-outcome rescore without generating new confirmation
futures.  We report every candidate and frozen half separately, including the
mixed or negative cells, and have limited the manuscript claim to improvement
over the registered direct-history ridge.  We also clarify that Appendix C's
bits-per-transition task conditions on realized future symbols and is not
numerically equivalent to launch-time F12 log loss.
"""
    (OUTPUT_ROOT / "PROPOSED_MANUSCRIPT_AND_REVIEWER_RESPONSE.md").write_text(
        proposed, encoding="utf-8"
    )
    print(f"report: {OUTPUT_ROOT / 'SEQUENCE_HISTORY_ANALYSIS_REPORT.md'}")


def _output_checksums() -> list[str]:
    paths = sorted(
        path
        for path in OUTPUT_ROOT.iterdir()
        if path.is_file() and path.name != "SHA256SUMS"
    )
    lines = [f"{sha256_file(path)}  {path.name}" for path in paths]
    (OUTPUT_ROOT / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


def verify() -> None:
    protocol = _protocol()
    if not (OUTPUT_ROOT / "comparisons.csv").is_file():
        raise RuntimeError("run report before verify")
    failures: list[str] = []
    current_sources = source_contract()
    if current_sources != protocol["read_only_sources"]:
        failures.append("one or more read-only source artifacts changed after prepare")
    if _analysis_code_contract() != protocol["analysis_code"]:
        failures.append("analysis code changed after prepare")
    comparisons = pd.read_csv(OUTPUT_ROOT / "comparisons.csv")
    score_lookup = pd.read_csv(OUTPUT_ROOT / "scores.csv")
    recomputed = 0
    for spec in _selected_specs("all"):
        with np.load(OUTPUT_ROOT / f"predictions_{spec.key}.npz", allow_pickle=False) as archive:
            if str(archive["protocol_id"][0]) != protocol["protocol_id"]:
                failures.append(f"{spec.key}: prediction protocol mismatch")
                continue
            targets = archive["targets"]
            candidate_values = archive["candidate"]
            for name in ("direct", "markov", "semimarkov", "lagged", "composite"):
                probability = archive[f"prediction_{name}"]
                if not np.all(np.isfinite(probability) & (probability > 0) & (probability < 1)):
                    failures.append(f"{spec.key}: invalid {name} probabilities")
            for candidate in ("02", "03"):
                selected = candidate_values == candidate
                for half, branch_slice in (("A", slice(0, 32)), ("B", slice(32, 64))):
                    y = targets[selected, branch_slice]
                    for baseline in ("direct", "markov", "semimarkov", "lagged"):
                        observed = float(
                            (
                                branch_losses(y, archive[f"prediction_{baseline}"][selected])
                                - branch_losses(y, archive["prediction_composite"][selected])
                            ).mean()
                        )
                        row = comparisons[
                            (comparisons["cohort"] == spec.key)
                            & (comparisons["candidate"].astype(str).str.zfill(2) == candidate)
                            & (comparisons["half"] == half)
                            & (comparisons["baseline"] == baseline)
                        ]
                        if len(row) != 1 or abs(observed - float(row.iloc[0]["gain_nats"])) > 2e-12:
                            failures.append(
                                f"{spec.key} c{candidate}{half} {baseline}: gain recomputation failed"
                            )
                        recomputed += 1
    primary = comparisons[(comparisons["role"] == "primary") & (comparisons["baseline"] == "lagged")]
    if len(primary) != 8:
        failures.append("primary family does not contain eight cells")
    verification = {
        "format": "reviewer-sequence-history-verification-v1",
        "protocol_id": protocol["protocol_id"],
        "passed": not failures,
        "failures": failures,
        "recomputed_comparisons": recomputed,
        "source_hashes_unchanged": current_sources == protocol["read_only_sources"],
        "analysis_code_unchanged": _analysis_code_contract() == protocol["analysis_code"],
        "new_confirmation_futures": False,
        "originating_l53_l54_scored": False,
    }
    _write_json(OUTPUT_ROOT / "verification.json", verification)
    lines = _output_checksums()
    # Immediate manifest readback.
    for line in lines:
        expected, name = line.split("  ", 1)
        if sha256_file(OUTPUT_ROOT / name) != expected:
            failures.append(f"checksum readback failed: {name}")
    if failures:
        verification["passed"] = False
        verification["failures"] = failures
        _write_json(OUTPUT_ROOT / "verification.json", verification)
        _output_checksums()
        raise AssertionError("verification failed: " + "; ".join(failures))
    print(f"verification passed; {len(lines)} output files checksummed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    subparsers.add_parser("prepare")
    replay_parser = subparsers.add_parser("replay")
    replay_parser.add_argument(
        "--dataset",
        default="all",
        choices=("all", "primary", "headline", *COHORTS.keys()),
    )
    replay_parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 12))
    subparsers.add_parser("fit")
    subparsers.add_parser("analyze")
    subparsers.add_parser("report")
    subparsers.add_parser("verify")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "prepare":
        prepare()
    elif args.stage == "replay":
        replay(args.dataset, args.workers)
    elif args.stage == "fit":
        fit()
    elif args.stage == "analyze":
        analyze()
    elif args.stage == "report":
        report()
    elif args.stage == "verify":
        verify()
    else:  # pragma: no cover
        raise AssertionError(args.stage)


if __name__ == "__main__":
    main()
