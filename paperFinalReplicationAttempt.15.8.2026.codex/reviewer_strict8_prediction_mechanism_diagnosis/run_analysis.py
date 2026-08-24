"""Checkpointed strict-8 prediction and causal mechanism diagnosis."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pickle
import platform
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


TASK_ROOT = Path(__file__).resolve().parent
PAPER_ROOT = TASK_ROOT.parent
WORKSPACE_ROOT = PAPER_ROOT.parent
SOURCE_ROOT = WORKSPACE_ROOT / "replicators.13.8.2026.codex"
PREVIOUS_ROOT = PAPER_ROOT / "reviewer_strict_event_geometry_audit"
os.environ.setdefault("MPLCONFIGDIR", str(TASK_ROOT / "artifacts" / "matplotlib"))
sys.path.insert(0, str(SOURCE_ROOT))
sys.path.insert(0, str(PREVIOUS_ROOT))
sys.path.insert(0, str(TASK_ROOT))

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

from plastic_heredity.config import CANDIDATES
from plastic_heredity.mechanistic import verify_checksums, write_checksums
from plastic_heredity.mechanistic_metrics import holm_adjust
from plastic_heredity.mechanistic_v2_features import (
    FEATURE_NAMES,
    MechanisticV2RawFeatures,
)
from plastic_heredity.mechanistic_v2_models import (
    RIDGE_LAMBDAS,
    fit_block_transform,
    fit_linear,
    matrix_cv_fold,
)
from plastic_heredity.metrics import centered_spearman
from plastic_heredity.seeds import derive_seed
from plastic_heredity.simulator import Snapshot, simulate_future_absorbing

from core import (
    ARM_NAMES,
    CONCENTRATION_NAMES,
    PRIMARY_INTERVENTION_CONTRASTS,
    TRANSITION_NAMES,
    aggregate_transitions,
    apply_intervention,
    bray_pair_decomposition,
    concentration_descriptors,
)
from strict_core import (
    GATE_NAMES,
    SPEC_NAMES,
    build_geometry,
    score_all_specs,
    window_pairwise_minimum,
)


def _load_previous_runner():
    path = PREVIOUS_ROOT / "run_analysis.py"
    spec = importlib.util.spec_from_file_location("strict_geometry_runner", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


previous = _load_previous_runner()

ARTIFACT_ROOT = TASK_ROOT / "artifacts"
PROTOCOL_ROOT = ARTIFACT_ROOT / "protocol"
FEATURE_ROOT = ARTIFACT_ROOT / "features"
MODEL_ROOT = ARTIFACT_ROOT / "models"
GEOMETRY_ROOT = ARTIFACT_ROOT / "geometry"
INTERVENTION_ROOT = ARTIFACT_ROOT / "intervention"
WORK_ROOT = ARTIFACT_ROOT / "work"
OUTPUT_ROOT = ARTIFACT_ROOT / "output"

PREVIOUS_REPLAY_ROOT = PREVIOUS_ROOT / "artifacts" / "replays"
PREVIOUS_OUTPUT_ROOT = PREVIOUS_ROOT / "artifacts" / "output"
DEVELOPMENT_SOURCE = SOURCE_ROOT / "results" / "regime_development"
CONFIRMATION_SOURCE = SOURCE_ROOT / "results" / "regime_confirmation"

BOOTSTRAP_REPETITIONS = 4096
RANDOMIZATION_REPETITIONS = 4096
MIN_SUCCESSES = 100
MIN_FAILURES = 100
MIN_OUTCOME_MATRICES = 20
INTERVENTION_BRANCHES = 64
INTERVENTION_HORIZON = 32
FULL_DOSE_VALIDITY = 0.90
INTERVENTION_SELECTION_SEED = hashlib.sha256(
    b"strict8-mechanism-intervention-v1::selection"
).hexdigest()
INTERVENTION_FUTURE_SEED = hashlib.sha256(
    b"strict8-mechanism-intervention-v1::future"
).hexdigest()
BOOTSTRAP_SEED = hashlib.sha256(b"strict8-mechanism-v1::bootstrap").hexdigest()
RANDOMIZATION_SEED = hashlib.sha256(b"strict8-mechanism-v1::randomization").hexdigest()
CHECKPOINT_FORMAT = "strict8-mechanism-checkpoint-v1"
RESULT_FORMAT = "strict8-prediction-mechanism-diagnosis-v1"

VARIANT_NAMES = ("H", "HC", "HS", "HCS")
VARIANT_LABELS = {
    "H": "history",
    "HC": "history+concentration",
    "HS": "history+state26",
    "HCS": "history+concentration+state26",
}
PREDICTION_CONTRASTS = {
    "concentration_beyond_history": ("H", "HC"),
    "state_beyond_history": ("H", "HS"),
    "residual_state_beyond_concentration": ("HC", "HCS"),
}
MARGIN_NAMES = ("pairwise_given_run8", "anchor_given_coherence")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
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


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _atomic_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=5)
    temporary.replace(path)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _replace_checksums(directory: Path) -> None:
    checksum = directory / "SHA256SUMS"
    if checksum.exists():
        checksum.unlink()
    write_checksums(directory)


def _source_contract() -> dict[str, dict[str, str]]:
    paths = {
        "new_core": TASK_ROOT / "core.py",
        "analysis_runner": TASK_ROOT / "run_analysis.py",
        "unit_tests": TASK_ROOT / "test_core.py",
        "previous_core": PREVIOUS_ROOT / "strict_core.py",
        "previous_protocol": PREVIOUS_ROOT / "artifacts" / "protocol" / "analysis_protocol.json",
        "previous_replay_checksums": PREVIOUS_REPLAY_ROOT / "SHA256SUMS",
        "previous_output_checksums": PREVIOUS_OUTPUT_ROOT / "SHA256SUMS",
        "development_checksums": DEVELOPMENT_SOURCE / "SHA256SUMS",
        "confirmation_checksums": CONFIRMATION_SOURCE / "SHA256SUMS",
        "simulator": SOURCE_ROOT / "plastic_heredity" / "simulator.py",
        "model_features": SOURCE_ROOT / "plastic_heredity" / "mechanistic_v2_features.py",
        "model_fitting": SOURCE_ROOT / "plastic_heredity" / "mechanistic_v2_models.py",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    return {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for name, path in paths.items()
    }


def _verify_inputs() -> dict[str, int]:
    return {
        "previous_replays": len(verify_checksums(PREVIOUS_REPLAY_ROOT)),
        "previous_outputs": len(verify_checksums(PREVIOUS_OUTPUT_ROOT)),
        "development": len(verify_checksums(DEVELOPMENT_SOURCE)),
        "confirmation": len(verify_checksums(CONFIRMATION_SOURCE)),
    }


def _protocol() -> dict[str, Any]:
    value: dict[str, Any] = {
        "format": "strict8-prediction-mechanism-protocol-v1",
        "date": "2026-08-19",
        "status": "post_hoc_locked_before_new_gate_readout_or_intervention_futures",
        "scope": {
            "all_new_writes_below": str(TASK_ROOT.resolve()),
            "manuscript_modified": False,
            "deliverable": "internal verified diagnosis",
            "endpoints_equal_primary_status": list(SPEC_NAMES),
        },
        "retained_diagnosis": {
            "transitions": list(TRANSITION_NAMES),
            "margins": list(MARGIN_NAMES),
            "variants": VARIANT_LABELS,
            "contrasts": PREDICTION_CONTRASTS,
            "concentration_features": list(CONCENTRATION_NAMES),
            "development_fit_only": True,
            "seal_before_confirmation_scoring": True,
            "conditional_power": {
                "successes": MIN_SUCCESSES,
                "failures": MIN_FAILURES,
                "matrices_each_outcome": MIN_OUTCOME_MATRICES,
            },
            "reliability_branch_budgets": [8, 16, 32, 64],
        },
        "geometry_replay": {
            "selection": "union of confirmation events and frozen same-state precursor controls",
            "pairwise_comparisons_per_window": 28,
            "bray_rank_bands": ["top1", "rank2_to5", "tail6plus"],
            "original_future_seeds": True,
            "new_random_futures": False,
        },
        "intervention": {
            "states": 2000,
            "source": "retained REGCONF natural states",
            "arms": list(ARM_NAMES),
            "branches_per_state_arm": INTERVENTION_BRANCHES,
            "horizon": INTERVENTION_HORIZON,
            "future_count": 2000 * len(ARM_NAMES) * INTERVENTION_BRANCHES,
            "doses_are_nested": True,
            "mass_history_matrix_clocks_preserved": True,
            "common_random_streams": True,
            "future_seed_excludes_arm": True,
            "selection_seed": INTERVENTION_SELECTION_SEED,
            "future_seed": INTERVENTION_FUTURE_SEED,
            "full_dose_validity_fraction": FULL_DOSE_VALIDITY,
            "primary_contrasts": PRIMARY_INTERVENTION_CONTRASTS,
            "primary_holm_family": "12 endpoint-by-candidate-by-half cells per contrast",
            "primary_dose": 4,
            "dose1": "direction/dose diagnostic",
            "estimand": "intent-to-treat over retained surviving observable REGCONF states",
        },
        "inference": {
            "unit": "catalytic matrix",
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "randomization_repetitions": RANDOMIZATION_REPETITIONS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "randomization_seed": RANDOMIZATION_SEED,
        },
        "claim_boundary": (
            "Post-hoc diagnosis; the fresh intervention supports only effects of the "
            "specified composition policies on reused observable selected-lineage states."
        ),
        "source_contract": _source_contract(),
    }
    value["protocol_id"] = canonical_digest(value)
    return _json_ready(value)


def prepare() -> None:
    _verify_inputs()
    value = _protocol()
    path = PROTOCOL_ROOT / "analysis_protocol.json"
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError("existing protocol differs from current scientific contract")
        print(f"Protocol already frozen and identical: {path}", flush=True)
        return
    _write_json(path, value)
    _replace_checksums(PROTOCOL_ROOT)
    print(f"Frozen strict-8 mechanism protocol: {path}", flush=True)


def verify_protocol() -> dict[str, Any]:
    verify_checksums(PROTOCOL_ROOT)
    saved = json.loads((PROTOCOL_ROOT / "analysis_protocol.json").read_text())
    if saved != _protocol():
        raise ValueError("protocol, scientific core, or input identity changed")
    _verify_inputs()
    return saved


_PROTOCOL_ID_CACHE: str | None = None
_CONFIRMATION_CASE_CACHE: tuple[Any, list[Any]] | None = None


def _protocol_id() -> str:
    global _PROTOCOL_ID_CACHE
    if _PROTOCOL_ID_CACHE is None:
        _PROTOCOL_ID_CACHE = str(verify_protocol()["protocol_id"])
    return _PROTOCOL_ID_CACHE


def _cases(cohort: str, workers: int) -> tuple[Any, list[Any]]:
    global _CONFIRMATION_CASE_CACHE
    if cohort == "confirmation" and _CONFIRMATION_CASE_CACHE is not None:
        return _CONFIRMATION_CASE_CACHE
    value = previous._cases(cohort, workers)
    if cohort == "confirmation":
        _CONFIRMATION_CASE_CACHE = value
    return value


def build_features(workers: int) -> None:
    verify_protocol()
    FEATURE_ROOT.mkdir(parents=True, exist_ok=True)
    for cohort in ("development", "confirmation"):
        output = FEATURE_ROOT / f"{cohort}_concentration.npz"
        if output.is_file():
            continue
        source = _source_arrays(cohort)
        values = np.vstack(
            [concentration_descriptors(composition) for composition in source["compositions"]]
        )
        _atomic_npz(
            output,
            protocol_id=np.asarray(_protocol_id()),
            state_ids=np.asarray(source["state_ids"]),
            concentration_names=np.asarray(CONCENTRATION_NAMES),
            concentration=values,
        )
    _replace_checksums(FEATURE_ROOT)
    print(f"Starting-state concentration features written to {FEATURE_ROOT}", flush=True)


def _source_arrays(cohort: str) -> dict[str, np.ndarray]:
    root = DEVELOPMENT_SOURCE if cohort == "development" else CONFIRMATION_SOURCE
    name = "development_arrays.npz" if cohort == "development" else "confirmation_arrays.npz"
    return _load_npz(root / name)


def _replay_arrays(cohort: str) -> dict[str, np.ndarray]:
    return _load_npz(PREVIOUS_REPLAY_ROOT / f"{cohort}.npz")


def _concentration_arrays(cohort: str) -> dict[str, np.ndarray]:
    verify_checksums(FEATURE_ROOT)
    return _load_npz(FEATURE_ROOT / f"{cohort}_concentration.npz")


def _designs(cohort: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    source = _source_arrays(cohort)
    concentration = _concentration_arrays(cohort)
    if not np.array_equal(source["state_ids"], concentration["state_ids"]):
        raise AssertionError("source and concentration state order differ")
    h = np.asarray(source["h10"], dtype=float)
    c = np.asarray(concentration["concentration"], dtype=float)
    s = np.asarray(source["state_block"], dtype=float)
    return {
        "H": h,
        "HC": np.column_stack((h, c)),
        "HS": np.column_stack((h, s)),
        "HCS": np.column_stack((h, c, s)),
    }, source


def _variant_names() -> dict[str, tuple[str, ...]]:
    h = tuple(FEATURE_NAMES["h10"])
    c = tuple(f"concentration__{name}" for name in CONCENTRATION_NAMES)
    s = tuple(f"state__{name}" for name in FEATURE_NAMES["state"])
    return {"H": h, "HC": h + c, "HS": h + s, "HCS": h + c + s}


def _binomial_loss(successes: np.ndarray, trials: np.ndarray, probability: np.ndarray) -> float:
    p = np.clip(np.asarray(probability, dtype=float), 1e-12, 1 - 1e-12)
    return float(
        -np.sum(successes * np.log(p) + (trials - successes) * np.log(1 - p))
        / np.sum(trials)
    )


def _fit_binomial_model(
    design: np.ndarray,
    names: tuple[str, ...],
    successes: np.ndarray,
    trials: np.ndarray,
    matrix_ids: np.ndarray,
    label: str,
) -> dict[str, Any]:
    keep = np.asarray(trials) > 0
    x = np.asarray(design, dtype=float)[keep]
    y = np.asarray(successes, dtype=float)[keep]
    n = np.asarray(trials, dtype=float)[keep]
    mids = np.asarray(matrix_ids, dtype=np.int64)[keep]
    scores: dict[str, float] = {}
    for ridge in RIDGE_LAMBDAS:
        numerator = 0.0
        denominator = 0.0
        for fold in range(5):
            validation = matrix_cv_fold(mids) == fold
            train = ~validation
            transform = fit_block_transform(label, x[train], names)
            train_x = transform.transform(x[train])
            validation_x = transform.transform(x[validation])
            fit = fit_linear(label, label, train_x, y[train], n[train], ridge)
            probability = expit(fit.correction(validation_x))
            weight = float(n[validation].sum())
            numerator += _binomial_loss(y[validation], n[validation], probability) * weight
            denominator += weight
        scores[f"{ridge:g}"] = numerator / denominator
    minimum = min(scores.values())
    selected = max(
        ridge for ridge in RIDGE_LAMBDAS if scores[f"{ridge:g}"] <= minimum + 1e-12
    )
    transform = fit_block_transform(label, x, names)
    fit = fit_linear(label, label, transform.transform(x), y, n, selected)
    return {
        "label": label,
        "transform": transform,
        "fit": fit,
        "selected_lambda": selected,
        "cv_scores": scores,
        "training_rows": int(keep.sum()),
        "training_successes": float(y.sum()),
        "training_trials": float(n.sum()),
    }


def _predict_binomial(model: Mapping[str, Any], design: np.ndarray) -> np.ndarray:
    return expit(model["fit"].correction(model["transform"].transform(design)))


def _transition_power(
    cohort: str,
    gates: np.ndarray,
    candidates: np.ndarray,
    matrix_ids: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for spec_index, spec in enumerate(SPEC_NAMES):
        for candidate in ("02", "03"):
            selected = np.asarray(candidates).astype(str) == candidate
            success, trials = aggregate_transitions(gates[selected, :, spec_index])
            mids = np.asarray(matrix_ids[selected], dtype=int)
            for transition_index, transition in enumerate(TRANSITION_NAMES):
                successes = int(success[:, transition_index].sum())
                total = int(trials[:, transition_index].sum())
                failures = total - successes
                success_matrices = int(
                    sum(success[mids == key, transition_index].sum() > 0 for key in np.unique(mids))
                )
                failure_matrices = int(
                    sum(
                        (trials[mids == key, transition_index] - success[mids == key, transition_index]).sum() > 0
                        for key in np.unique(mids)
                    )
                )
                rows.append(
                    {
                        "cohort": cohort,
                        "spec": spec,
                        "candidate": candidate,
                        "transition": transition,
                        "successes": successes,
                        "failures": failures,
                        "trials": total,
                        "rate": successes / total,
                        "success_matrices": success_matrices,
                        "failure_matrices": failure_matrices,
                        "power_adequate": successes >= MIN_SUCCESSES
                        and failures >= MIN_FAILURES
                        and success_matrices >= MIN_OUTCOME_MATRICES
                        and failure_matrices >= MIN_OUTCOME_MATRICES,
                    }
                )
    return pd.DataFrame(rows)


def fit_stage_models() -> None:
    verify_protocol()
    verify_checksums(FEATURE_ROOT)
    designs, source = _designs("development")
    replay = _replay_arrays("development")
    if not np.array_equal(source["state_ids"], replay["state_ids"]):
        raise AssertionError("development source/replay order differs")
    names = _variant_names()
    models: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    candidates = np.asarray(source["candidates"]).astype(str)
    matrix_ids = np.asarray(source["matrix_ids"], dtype=int)
    for spec_index, spec in enumerate(SPEC_NAMES):
        for candidate in ("02", "03"):
            selected = candidates == candidate
            success, trials = aggregate_transitions(
                replay["deepest_gate"][selected, :, spec_index]
            )
            for transition_index, transition in enumerate(TRANSITION_NAMES):
                for variant in VARIANT_NAMES:
                    key = "|".join((spec, candidate, transition, variant))
                    print(f"[stage-model] {key}", flush=True)
                    model = _fit_binomial_model(
                        designs[variant][selected],
                        names[variant],
                        success[:, transition_index],
                        trials[:, transition_index],
                        matrix_ids[selected],
                        key,
                    )
                    models[key] = model
                    metadata[key] = {
                        "selected_lambda": model["selected_lambda"],
                        "cv_scores": model["cv_scores"],
                        "training_rows": model["training_rows"],
                        "training_successes": model["training_successes"],
                        "training_trials": model["training_trials"],
                    }
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    path = MODEL_ROOT / "stage_models.pkl"
    _atomic_pickle(path, models)
    _write_json(MODEL_ROOT / "stage_model_metadata.json", metadata)
    seal = {
        "format": "strict8-stage-model-seal-v1",
        "protocol_id": _protocol_id(),
        "development_replay_sha256": sha256_file(PREVIOUS_REPLAY_ROOT / "development.npz"),
        "development_features_sha256": sha256_file(FEATURE_ROOT / "development_concentration.npz"),
        "stage_models_sha256": sha256_file(path),
        "confirmation_gate_or_margin_arrays_loaded": False,
    }
    seal["seal_id"] = canonical_digest(seal)
    _write_json(MODEL_ROOT / "stage_model_seal.json", seal)
    _replace_checksums(MODEL_ROOT)
    print(f"Development stage models sealed at {MODEL_ROOT}", flush=True)


def verify_stage_seal() -> dict[str, Any]:
    verify_checksums(MODEL_ROOT)
    seal = json.loads((MODEL_ROOT / "stage_model_seal.json").read_text())
    seal_id = seal.pop("seal_id")
    if canonical_digest(seal) != seal_id:
        raise ValueError("stage model seal identifier mismatch")
    seal["seal_id"] = seal_id
    if seal["protocol_id"] != _protocol_id():
        raise ValueError("stage model seal protocol mismatch")
    if seal["stage_models_sha256"] != sha256_file(MODEL_ROOT / "stage_models.pkl"):
        raise ValueError("sealed stage models changed")
    if seal["confirmation_gate_or_margin_arrays_loaded"] is not False:
        raise ValueError("development-only model boundary failed")
    return seal


def _group_values(values: np.ndarray, matrix_ids: np.ndarray) -> np.ndarray:
    return np.asarray(
        [np.mean(values[matrix_ids == key]) for key in np.unique(matrix_ids)], dtype=float
    )


def _gain_inference(
    loss_left: np.ndarray,
    loss_right: np.ndarray,
    matrix_ids: np.ndarray,
    seed_parts: Sequence[str],
) -> tuple[float, float, float, float]:
    differences = np.asarray(loss_left) - np.asarray(loss_right)
    groups = _group_values(differences, np.asarray(matrix_ids))
    observed = float(groups.mean())
    rng = np.random.default_rng(derive_seed(BOOTSTRAP_SEED, *seed_parts))
    indices = rng.integers(0, len(groups), size=(BOOTSTRAP_REPETITIONS, len(groups)))
    boot = groups[indices].mean(axis=1)
    lower, upper = np.quantile(boot, (0.025, 0.975))
    rng = np.random.default_rng(derive_seed(RANDOMIZATION_SEED, *seed_parts))
    signs = rng.integers(0, 2, size=(RANDOMIZATION_REPETITIONS, len(groups))) * 2 - 1
    null = (signs @ groups) / len(groups)
    p = float((np.count_nonzero(null >= observed) + 1) / (RANDOMIZATION_REPETITIONS + 1))
    return observed, float(lower), float(upper), p


def _state_binomial_loss(successes: np.ndarray, trials: np.ndarray, probability: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probability), 1e-12, 1 - 1e-12)
    return -(successes * np.log(p) + (trials - successes) * np.log(1 - p)) / trials


def score_stage_models() -> None:
    verify_complete_model_seal()
    designs, source = _designs("confirmation")
    replay = _replay_arrays("confirmation")
    with (MODEL_ROOT / "stage_models.pkl").open("rb") as handle:
        models = pickle.load(handle)
    candidates = np.asarray(source["candidates"]).astype(str)
    matrix_ids = np.asarray(source["matrix_ids"], dtype=int)
    dev_replay = _replay_arrays("development")
    dev_source = _source_arrays("development")
    dev_power = _transition_power(
        "development",
        dev_replay["deepest_gate"],
        dev_source["candidates"],
        dev_source["matrix_ids"],
    )
    conf_power = _transition_power(
        "confirmation", replay["deepest_gate"], candidates, matrix_ids
    )
    rows: list[dict[str, Any]] = []
    predictions: dict[str, np.ndarray] = {}
    for spec_index, spec in enumerate(SPEC_NAMES):
        for candidate in ("02", "03"):
            selected = candidates == candidate
            mids = matrix_ids[selected]
            for transition_index, transition in enumerate(TRANSITION_NAMES):
                for variant in VARIANT_NAMES:
                    key = "|".join((spec, candidate, transition, variant))
                    predictions[key] = _predict_binomial(
                        models[key], designs[variant][selected]
                    )
                for half, branch_slice in (
                    ("A", slice(0, 64)),
                    ("B", slice(64, 128)),
                ):
                    success, trials = aggregate_transitions(
                        replay["deepest_gate"][selected, :, spec_index], branch_slice
                    )
                    y = success[:, transition_index]
                    n = trials[:, transition_index]
                    keep = n > 0
                    for contrast, (left, right) in PREDICTION_CONTRASTS.items():
                        left_loss = _state_binomial_loss(y[keep], n[keep], predictions["|".join((spec, candidate, transition, left))][keep])
                        right_loss = _state_binomial_loss(y[keep], n[keep], predictions["|".join((spec, candidate, transition, right))][keep])
                        gain, lower, upper, p = _gain_inference(
                            left_loss,
                            right_loss,
                            mids[keep],
                            ("stage", spec, candidate, transition, half, contrast),
                        )
                        power_dev = bool(
                            dev_power.loc[
                                (dev_power.spec == spec)
                                & (dev_power.candidate.astype(str).str.zfill(2) == candidate)
                                & (dev_power.transition == transition),
                                "power_adequate",
                            ].iloc[0]
                        )
                        power_conf = bool(
                            conf_power.loc[
                                (conf_power.spec == spec)
                                & (conf_power.candidate.astype(str).str.zfill(2) == candidate)
                                & (conf_power.transition == transition),
                                "power_adequate",
                            ].iloc[0]
                        )
                        rows.append(
                            {
                                "spec": spec,
                                "candidate": candidate,
                                "transition": transition,
                                "half": half,
                                "contrast": contrast,
                                "baseline_variant": left,
                                "enhanced_variant": right,
                                "eligible_states": int(keep.sum()),
                                "successes": int(y.sum()),
                                "trials": int(n.sum()),
                                "development_power_adequate": power_dev,
                                "confirmation_power_adequate": power_conf,
                                "log_loss_baseline": float(left_loss.mean()),
                                "log_loss_enhanced": float(right_loss.mean()),
                                "log_loss_gain": gain,
                                "ci95_lower": lower,
                                "ci95_upper": upper,
                                "randomization_p_raw": p,
                            }
                        )
    table = pd.DataFrame(rows)
    table["randomization_p_holm"] = np.nan
    table["passes_exploratory_gate"] = False
    for transition in TRANSITION_NAMES:
        for contrast in PREDICTION_CONTRASTS:
            selected = (table.transition == transition) & (table.contrast == contrast)
            table.loc[selected, "randomization_p_holm"] = holm_adjust(
                table.loc[selected, "randomization_p_raw"].tolist()
            )
    table["passes_exploratory_gate"] = (
        table["development_power_adequate"]
        & table["confirmation_power_adequate"]
        & (table["log_loss_gain"] > 0)
        & (table["ci95_lower"] > 0)
        & (table["randomization_p_holm"] < 0.05)
    )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    pd.concat((dev_power, conf_power), ignore_index=True).to_csv(
        OUTPUT_ROOT / "transition_power.csv", index=False
    )
    table.to_csv(OUTPUT_ROOT / "stage_prediction.csv", index=False)
    print("Confirmation stage predictions scored", flush=True)


def _weighted_ridge_fit(
    design: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    ridge: float,
) -> tuple[float, np.ndarray]:
    x = np.asarray(design, dtype=float)
    y = np.asarray(target, dtype=float)
    w = np.asarray(weights, dtype=float)
    augmented = np.column_stack((np.ones(len(x)), x))
    root = np.sqrt(w / w.sum())
    weighted_x = augmented * root[:, None]
    weighted_y = y * root
    penalty = np.eye(augmented.shape[1]) * ridge
    penalty[0, 0] = 0.0
    coefficient = np.linalg.solve(
        weighted_x.T @ weighted_x + penalty,
        weighted_x.T @ weighted_y,
    )
    return float(coefficient[0]), coefficient[1:]


def _fit_margin_model(
    design: np.ndarray,
    names: tuple[str, ...],
    target: np.ndarray,
    weights: np.ndarray,
    matrix_ids: np.ndarray,
    label: str,
) -> dict[str, Any]:
    keep = np.isfinite(target) & (np.asarray(weights) > 0)
    x = np.asarray(design, dtype=float)[keep]
    y = np.asarray(target, dtype=float)[keep]
    w = np.asarray(weights, dtype=float)[keep]
    mids = np.asarray(matrix_ids, dtype=int)[keep]
    scores: dict[str, float] = {}
    for ridge in RIDGE_LAMBDAS:
        numerator = 0.0
        denominator = 0.0
        for fold in range(5):
            validation = matrix_cv_fold(mids) == fold
            train = ~validation
            transform = fit_block_transform(label, x[train], names)
            train_x = transform.transform(x[train])
            validation_x = transform.transform(x[validation])
            intercept, coefficient = _weighted_ridge_fit(
                train_x, y[train], w[train], ridge
            )
            prediction = intercept + validation_x @ coefficient
            numerator += float(np.sum(w[validation] * (y[validation] - prediction) ** 2))
            denominator += float(w[validation].sum())
        scores[f"{ridge:g}"] = numerator / denominator
    minimum = min(scores.values())
    selected = max(
        ridge for ridge in RIDGE_LAMBDAS if scores[f"{ridge:g}"] <= minimum + 1e-12
    )
    transform = fit_block_transform(label, x, names)
    intercept, coefficient = _weighted_ridge_fit(
        transform.transform(x), y, w, selected
    )
    return {
        "label": label,
        "transform": transform,
        "intercept": intercept,
        "coefficient": coefficient,
        "selected_lambda": selected,
        "cv_scores": scores,
        "training_rows": int(keep.sum()),
        "training_weight": float(w.sum()),
    }


def _predict_margin(model: Mapping[str, Any], design: np.ndarray) -> np.ndarray:
    return model["intercept"] + model["transform"].transform(design) @ model["coefficient"]


def _margin_branch_values(
    replay: Mapping[str, np.ndarray], spec_index: int, margin_index: int
) -> np.ndarray:
    if margin_index == 0:
        cutoff = previous._specs()[spec_index].coherence_cutoff
        return np.asarray(replay["best_pairwise_margin"][:, :, spec_index], dtype=float) + cutoff
    return np.asarray(replay["best_anchor_margin"][:, :, spec_index], dtype=float)


def _aggregate_margin(
    values: np.ndarray, branch_slice: slice | None = None
) -> tuple[np.ndarray, np.ndarray]:
    selected = values if branch_slice is None else values[:, branch_slice]
    finite = np.isfinite(selected)
    count = finite.sum(axis=1).astype(float)
    total = np.where(finite, selected, 0.0).sum(axis=1)
    mean = np.divide(total, count, out=np.full(len(count), np.nan), where=count > 0)
    return mean, count


def fit_margin_models() -> None:
    verify_stage_seal()
    designs, source = _designs("development")
    replay = _replay_arrays("development")
    names = _variant_names()
    candidates = np.asarray(source["candidates"]).astype(str)
    matrix_ids = np.asarray(source["matrix_ids"], dtype=int)
    models: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    for spec_index, spec in enumerate(SPEC_NAMES):
        for margin_index, margin in enumerate(MARGIN_NAMES):
            values = _margin_branch_values(replay, spec_index, margin_index)
            target, weights = _aggregate_margin(values)
            for candidate in ("02", "03"):
                selected = candidates == candidate
                for variant in VARIANT_NAMES:
                    key = "|".join((spec, candidate, margin, variant))
                    print(f"[margin-model] {key}", flush=True)
                    model = _fit_margin_model(
                        designs[variant][selected],
                        names[variant],
                        target[selected],
                        weights[selected],
                        matrix_ids[selected],
                        key,
                    )
                    models[key] = model
                    metadata[key] = {
                        "selected_lambda": model["selected_lambda"],
                        "cv_scores": model["cv_scores"],
                        "training_rows": model["training_rows"],
                        "training_weight": model["training_weight"],
                    }
    path = MODEL_ROOT / "margin_models.pkl"
    _atomic_pickle(path, models)
    _write_json(MODEL_ROOT / "margin_model_metadata.json", metadata)
    seal = {
        "format": "strict8-complete-model-seal-v1",
        "protocol_id": _protocol_id(),
        "stage_model_seal_sha256": sha256_file(MODEL_ROOT / "stage_model_seal.json"),
        "stage_models_sha256": sha256_file(MODEL_ROOT / "stage_models.pkl"),
        "margin_models_sha256": sha256_file(path),
        "confirmation_gate_or_margin_arrays_loaded": False,
    }
    seal["seal_id"] = canonical_digest(seal)
    _write_json(MODEL_ROOT / "complete_model_seal.json", seal)
    _replace_checksums(MODEL_ROOT)
    print(f"Continuous-margin models sealed at {MODEL_ROOT}", flush=True)


def verify_complete_model_seal() -> dict[str, Any]:
    verify_checksums(MODEL_ROOT)
    verify_stage_seal()
    seal = json.loads((MODEL_ROOT / "complete_model_seal.json").read_text())
    seal_id = seal.pop("seal_id")
    if canonical_digest(seal) != seal_id:
        raise ValueError("complete model seal identifier mismatch")
    seal["seal_id"] = seal_id
    if seal["protocol_id"] != _protocol_id():
        raise ValueError("complete model seal protocol mismatch")
    for key, name in (
        ("stage_models_sha256", "stage_models.pkl"),
        ("margin_models_sha256", "margin_models.pkl"),
    ):
        if seal[key] != sha256_file(MODEL_ROOT / name):
            raise ValueError(f"sealed model changed: {name}")
    return seal


def score_margin_models() -> None:
    verify_complete_model_seal()
    designs, source = _designs("confirmation")
    replay = _replay_arrays("confirmation")
    dev_replay = _replay_arrays("development")
    dev_source = _source_arrays("development")
    with (MODEL_ROOT / "margin_models.pkl").open("rb") as handle:
        models = pickle.load(handle)
    candidates = np.asarray(source["candidates"]).astype(str)
    matrix_ids = np.asarray(source["matrix_ids"], dtype=int)
    dev_candidates = np.asarray(dev_source["candidates"]).astype(str)
    rows: list[dict[str, Any]] = []
    for spec_index, spec in enumerate(SPEC_NAMES):
        for margin_index, margin in enumerate(MARGIN_NAMES):
            confirmation_values = _margin_branch_values(replay, spec_index, margin_index)
            development_values = _margin_branch_values(dev_replay, spec_index, margin_index)
            development_target, development_weight = _aggregate_margin(development_values)
            for candidate in ("02", "03"):
                selected = candidates == candidate
                mids = matrix_ids[selected]
                dev_selected = dev_candidates == candidate
                dev_rows = int(np.count_nonzero(development_weight[dev_selected] > 0))
                dev_matrices = int(
                    np.unique(np.asarray(dev_source["matrix_ids"])[dev_selected][development_weight[dev_selected] > 0]).size
                )
                predictions = {
                    variant: _predict_margin(
                        models["|".join((spec, candidate, margin, variant))],
                        designs[variant][selected],
                    )
                    for variant in VARIANT_NAMES
                }
                for half, branch_slice in (("A", slice(0, 64)), ("B", slice(64, 128))):
                    target, weight = _aggregate_margin(confirmation_values[selected], branch_slice)
                    keep = np.isfinite(target) & (weight > 0)
                    for contrast, (left, right) in PREDICTION_CONTRASTS.items():
                        left_loss = (target[keep] - predictions[left][keep]) ** 2
                        right_loss = (target[keep] - predictions[right][keep]) ** 2
                        gain, lower, upper, p = _gain_inference(
                            left_loss,
                            right_loss,
                            mids[keep],
                            ("margin", spec, candidate, margin, half, contrast),
                        )
                        rows.append(
                            {
                                "spec": spec,
                                "candidate": candidate,
                                "margin": margin,
                                "half": half,
                                "contrast": contrast,
                                "baseline_variant": left,
                                "enhanced_variant": right,
                                "eligible_states": int(keep.sum()),
                                "eligible_matrices": int(np.unique(mids[keep]).size),
                                "development_eligible_states": dev_rows,
                                "development_eligible_matrices": dev_matrices,
                                "mse_baseline": float(left_loss.mean()),
                                "mse_enhanced": float(right_loss.mean()),
                                "mse_gain": gain,
                                "ci95_lower": lower,
                                "ci95_upper": upper,
                                "randomization_p_raw": p,
                                "enhanced_spearman": float(spearmanr(predictions[right][keep], target[keep]).statistic),
                                "enhanced_centered_spearman": centered_spearman(
                                    predictions[right][keep], target[keep], mids[keep]
                                ),
                                "power_adequate": dev_rows >= 100
                                and dev_matrices >= 20
                                and int(keep.sum()) >= 100
                                and int(np.unique(mids[keep]).size) >= 20,
                            }
                        )
    table = pd.DataFrame(rows)
    table["randomization_p_holm"] = np.nan
    for margin in MARGIN_NAMES:
        for contrast in PREDICTION_CONTRASTS:
            selected = (table.margin == margin) & (table.contrast == contrast)
            table.loc[selected, "randomization_p_holm"] = holm_adjust(
                table.loc[selected, "randomization_p_raw"].tolist()
            )
    table["passes_exploratory_gate"] = (
        table["power_adequate"]
        & (table["mse_gain"] > 0)
        & (table["ci95_lower"] > 0)
        & (table["randomization_p_holm"] < 0.05)
    )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_ROOT / "margin_prediction.csv", index=False)
    print("Confirmation continuous-margin predictions scored", flush=True)


def _safe_rank(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 3 or np.unique(left).size < 2 or np.unique(right).size < 2:
        return np.nan
    return float(spearmanr(left, right).statistic)


def compute_reliability() -> None:
    replay = _replay_arrays("confirmation")
    candidates = np.asarray(replay["candidates"]).astype(str)
    matrix_ids = np.asarray(replay["matrix_ids"], dtype=int)
    rows: list[dict[str, Any]] = []
    for spec_index, spec in enumerate(SPEC_NAMES):
        for candidate in ("02", "03"):
            selected = candidates == candidate
            gates = replay["deepest_gate"][selected, :, spec_index]
            mids = matrix_ids[selected]
            for transition_index, transition in enumerate(TRANSITION_NAMES):
                for budget in (8, 16, 32, 64):
                    chunk_count = gates.shape[1] // budget
                    for pair_index in range(chunk_count // 2):
                        left_slice = slice(2 * pair_index * budget, (2 * pair_index + 1) * budget)
                        right_slice = slice((2 * pair_index + 1) * budget, (2 * pair_index + 2) * budget)
                        left_success, left_trials = aggregate_transitions(gates, left_slice)
                        right_success, right_trials = aggregate_transitions(gates, right_slice)
                        left_n = left_trials[:, transition_index]
                        right_n = right_trials[:, transition_index]
                        keep = (left_n > 0) & (right_n > 0)
                        left_q = left_success[keep, transition_index] / left_n[keep]
                        right_q = right_success[keep, transition_index] / right_n[keep]
                        rows.append(
                            {
                                "spec": spec,
                                "candidate": candidate,
                                "transition": transition,
                                "branch_budget_per_half": budget,
                                "pair_index": pair_index,
                                "eligible_states": int(keep.sum()),
                                "ordinary_spearman": _safe_rank(left_q, right_q),
                                "centered_spearman": centered_spearman(
                                    left_q, right_q, mids[keep]
                                ) if keep.sum() >= 3 else np.nan,
                            }
                        )
    pairs = pd.DataFrame(rows)
    summary = (
        pairs.groupby(
            ["spec", "candidate", "transition", "branch_budget_per_half"],
            as_index=False,
        )
        .agg(
            chunk_pairs=("pair_index", "count"),
            eligible_states_mean=("eligible_states", "mean"),
            ordinary_spearman_mean=("ordinary_spearman", "mean"),
            ordinary_spearman_min=("ordinary_spearman", "min"),
            ordinary_spearman_max=("ordinary_spearman", "max"),
            centered_spearman_mean=("centered_spearman", "mean"),
            centered_spearman_min=("centered_spearman", "min"),
            centered_spearman_max=("centered_spearman", "max"),
        )
    )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(OUTPUT_ROOT / "reliability_chunk_pairs.csv", index=False)
    summary.to_csv(OUTPUT_ROOT / "reliability_by_budget.csv", index=False)
    print("Gate reliability curves written", flush=True)


# ---------------------------------------------------------------------------
# Exact geometry replay of strict-event windows and frozen same-state controls


def _checkpoint_path(dataset: str, index: int) -> Path:
    return WORK_ROOT / dataset / f"state_{index:04d}.npz"


def _checkpoint_complete(dataset: str, index: int) -> bool:
    path = _checkpoint_path(dataset, index)
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as archive:
            return (
                str(archive["format"].item()) == CHECKPOINT_FORMAT
                and str(archive["dataset"].item()) == dataset
                and int(archive["state_index"].item()) == index
                and str(archive["protocol_id"].item()) == _protocol_id()
            )
    except Exception:
        return False


def _run_workers(
    dataset: str,
    arguments: Sequence[Any],
    worker: Callable[[Any], int],
    workers: int,
) -> None:
    (WORK_ROOT / dataset).mkdir(parents=True, exist_ok=True)
    pending = [item for item in arguments if not _checkpoint_complete(dataset, int(item[0]))]
    print(
        f"[{dataset}] {len(arguments) - len(pending)}/{len(arguments)} present; "
        f"{len(pending)} pending",
        flush=True,
    )
    if not pending:
        return
    progress_every = max(1, len(pending) // 50)
    if workers <= 1:
        for count, item in enumerate(pending, 1):
            worker(item)
            if count % progress_every == 0 or count == len(pending):
                print(f"[{dataset}] {count}/{len(pending)} new states", flush=True)
        return
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(worker, item) for item in pending]
        for count, future in enumerate(as_completed(futures), 1):
            future.result()
            if count % progress_every == 0 or count == len(pending):
                print(f"[{dataset}] {count}/{len(pending)} new states", flush=True)


def _geometry_selection() -> tuple[dict[int, list[tuple[int, int, int, int]]], dict[str, int]]:
    """Return state -> (branch, spec, role, frozen pair id) requests.

    Role one is an event window and role zero is its frozen same-state control.
    Events lacking a matched control remain in the event union with pair id -1.
    """

    replay = _replay_arrays("confirmation")
    labels = np.asarray(replay["labels"], dtype=np.int8)
    selected: dict[int, list[tuple[int, int, int, int]]] = {
        index: [] for index in range(labels.shape[0])
    }
    event_keys: set[tuple[int, int, int]] = set()
    for state_index, branch, spec_index in np.argwhere(labels == 1):
        event_keys.add((int(state_index), int(branch), int(spec_index)))

    matches = pd.read_csv(PREVIOUS_OUTPUT_ROOT / "matched_event_control_pairs.csv.gz")
    matched_events: set[tuple[int, int, int]] = set()
    for pair_id, row in matches.reset_index(drop=True).iterrows():
        state_index = int(row["state_index"])
        spec_index = int(row["spec_index"])
        event_branch = int(row["event_branch"])
        control_branch = int(row["control_branch"])
        key = (state_index, event_branch, spec_index)
        if key not in event_keys:
            raise AssertionError("frozen match refers to a non-event")
        matched_events.add(key)
        selected[state_index].append((event_branch, spec_index, 1, int(pair_id)))
        selected[state_index].append((control_branch, spec_index, 0, int(pair_id)))
    for state_index, branch, spec_index in sorted(event_keys - matched_events):
        selected[state_index].append((branch, spec_index, 1, -1))
    selected = {key: value for key, value in selected.items() if value}
    return selected, {
        "event_union": len(event_keys),
        "matched_events": len(matched_events),
        "matched_controls": len(matches),
        "states_selected": len(selected),
    }


GEOMETRY_COLUMNS = (
    "request_index",
    "branch",
    "spec_index",
    "role",
    "frozen_pair_id",
    "start",
    "deepest_gate",
    "left_offset",
    "right_offset",
    "cosine_similarity",
    "bray_similarity",
    "bray_distance",
    "top1_contribution",
    "rank2_to5_contribution",
    "tail6plus_contribution",
    "dominant_type_same",
    "top1_share_left",
    "top1_share_right",
    "cosine_pass",
    "bray_global_pass",
    "bray_relation_pass",
)


def _geometry_worker(
    arguments: tuple[
        int,
        Any,
        Any,
        tuple[Any, ...],
        list[tuple[int, int, int, int]],
        dict[str, np.ndarray],
        str,
    ]
) -> int:
    state_index, case, experiment, specs, requests, prior, protocol_id = arguments
    dataset = "geometry_replay"
    output = _checkpoint_path(dataset, state_index)
    if output.is_file():
        try:
            with np.load(output, allow_pickle=False) as archive:
                if (
                    str(archive["format"].item()) == CHECKPOINT_FORMAT
                    and str(archive["dataset"].item()) == dataset
                    and str(archive["protocol_id"].item()) == protocol_id
                ):
                    return state_index
        except Exception:
            pass
    branch_records: dict[int, Sequence[Any]] = {}
    mismatches = 0
    with threadpool_limits(limits=1):
        for branch in sorted({request[0] for request in requests}):
            records, _ = previous._future(case, experiment, branch)
            outcomes, _ = score_all_specs(records, specs)
            for spec_index, outcome in enumerate(outcomes):
                if (
                    int(outcome.event) != int(prior["labels"][branch, spec_index])
                    or outcome.onset != int(prior["onsets"][branch, spec_index])
                    or outcome.first_run != int(prior["first_run"][branch, spec_index])
                    or outcome.deepest_gate != int(prior["deepest_gate"][branch, spec_index])
                ):
                    mismatches += 1
            branch_records[branch] = records
    if mismatches:
        raise AssertionError(f"geometry replay mismatch in state {state_index}: {mismatches}")

    rows: list[list[float]] = []
    for request_index, (branch, spec_index, role, pair_id) in enumerate(requests):
        start = int(
            prior["onsets"][branch, spec_index]
            if role == 1
            else prior["first_run"][branch, spec_index]
        )
        records = branch_records[branch]
        window = records[start : start + 8]
        if start < 0 or len(window) != 8:
            raise AssertionError("selected geometry request lacks a complete window")
        relation_cutoff = float(specs[2].coherence_cutoff)
        global_cutoff = float(specs[1].coherence_cutoff)
        for left_offset in range(8):
            left = np.asarray(window[left_offset].daughter, dtype=float)
            for right_offset in range(left_offset + 1, 8):
                right = np.asarray(window[right_offset].daughter, dtype=float)
                denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
                cosine = 0.0 if denominator == 0 else float(np.dot(left, right) / denominator)
                decomposition = bray_pair_decomposition(left, right)
                bray = 1.0 - decomposition["bray_distance"]
                rows.append(
                    [
                        request_index,
                        branch,
                        spec_index,
                        role,
                        pair_id,
                        start,
                        int(prior["deepest_gate"][branch, spec_index]),
                        left_offset,
                        right_offset,
                        cosine,
                        bray,
                        decomposition["bray_distance"],
                        decomposition["top1_contribution"],
                        decomposition["rank2_to5_contribution"],
                        decomposition["tail6plus_contribution"],
                        decomposition["dominant_type_same"],
                        decomposition["top1_share_left"],
                        decomposition["top1_share_right"],
                        float(cosine > 0.90),
                        float(bray > global_cutoff),
                        float(bray > relation_cutoff),
                    ]
                )
    _atomic_npz(
        output,
        format=np.asarray(CHECKPOINT_FORMAT),
        dataset=np.asarray(dataset),
        state_index=np.asarray(state_index, dtype=np.int32),
        protocol_id=np.asarray(protocol_id),
        columns=np.asarray(GEOMETRY_COLUMNS),
        rows=np.asarray(rows, dtype=np.float64),
        replay_mismatches=np.asarray(mismatches, dtype=np.int32),
        branches_replayed=np.asarray(len(branch_records), dtype=np.int16),
    )
    return state_index


def replay_geometry(workers: int) -> None:
    protocol_id = _protocol_id()
    verify_checksums(PREVIOUS_OUTPUT_ROOT)
    selections, selection_summary = _geometry_selection()
    experiment, cases = _cases("confirmation", workers)
    specs = previous._specs()
    replay = _replay_arrays("confirmation")
    replay_keys = ("labels", "onsets", "first_run", "deepest_gate")
    arguments = [
        (
            index,
            cases[index],
            experiment,
            specs,
            requests,
            {key: np.asarray(replay[key][index]) for key in replay_keys},
            protocol_id,
        )
        for index, requests in sorted(selections.items())
    ]
    _run_workers("geometry_replay", arguments, _geometry_worker, workers)

    frames: list[pd.DataFrame] = []
    mismatch_total = 0
    branches_total = 0
    for state_index in sorted(selections):
        path = _checkpoint_path("geometry_replay", state_index)
        if not _checkpoint_complete("geometry_replay", state_index):
            raise ValueError(f"missing geometry checkpoint {state_index}")
        with np.load(path, allow_pickle=False) as archive:
            frame = pd.DataFrame(np.asarray(archive["rows"]), columns=GEOMETRY_COLUMNS)
            mismatch_total += int(archive["replay_mismatches"].item())
            branches_total += int(archive["branches_replayed"].item())
        frame.insert(0, "state_index", state_index)
        frame.insert(1, "state_id", str(cases[state_index].state_id))
        frame.insert(2, "candidate", str(cases[state_index].candidate))
        frame.insert(3, "matrix_id", int(cases[state_index].matrix_id))
        frame.insert(4, "landmark", int(cases[state_index].landmark))
        frame["spec"] = frame["spec_index"].astype(int).map(dict(enumerate(SPEC_NAMES)))
        frame["role_name"] = frame["role"].astype(int).map({0: "control", 1: "event"})
        frame["window_id"] = (
            frame["state_index"].astype(str)
            + "|"
            + frame["request_index"].astype(int).astype(str)
        )
        frames.append(frame)
    pairs = pd.concat(frames, ignore_index=True)
    GEOMETRY_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(GEOMETRY_ROOT / "window_pair_geometry.csv.gz", index=False)

    contributions = []
    for keys, group in pairs.groupby(["spec", "candidate", "role_name"], sort=False):
        total = float(group["bray_distance"].sum())
        for band, column in (
            ("top1", "top1_contribution"),
            ("rank2_to5", "rank2_to5_contribution"),
            ("tail6plus", "tail6plus_contribution"),
        ):
            value = float(group[column].sum())
            contributions.append(
                {
                    "spec": keys[0],
                    "candidate": keys[1],
                    "role": keys[2],
                    "rank_band": band,
                    "pair_comparisons": len(group),
                    "mean_contribution": float(group[column].mean()),
                    "share_of_total_bray_distance": value / total if total else np.nan,
                }
            )
    pd.DataFrame(contributions).to_csv(
        OUTPUT_ROOT / "geometry_rank_band_summary.csv", index=False
    )

    window_minima = (
        pairs.groupby(
            ["window_id", "spec", "candidate", "role_name"], as_index=False
        )
        .agg(
            cosine_min=("cosine_similarity", "min"),
            bray_min=("bray_similarity", "min"),
            cosine_all_pass=("cosine_pass", "min"),
            bray_global_all_pass=("bray_global_pass", "min"),
            bray_relation_all_pass=("bray_relation_pass", "min"),
            dominant_same_fraction=("dominant_type_same", "mean"),
        )
    )
    window_minima.to_csv(OUTPUT_ROOT / "geometry_window_minima.csv.gz", index=False)
    pass_summary = (
        window_minima.groupby(["spec", "candidate", "role_name"], as_index=False)
        .agg(
            windows=("window_id", "count"),
            cosine_min_mean=("cosine_min", "mean"),
            bray_min_mean=("bray_min", "mean"),
            cosine_all_pass_fraction=("cosine_all_pass", "mean"),
            bray_global_all_pass_fraction=("bray_global_all_pass", "mean"),
            bray_relation_all_pass_fraction=("bray_relation_all_pass", "mean"),
            dominant_same_fraction=("dominant_same_fraction", "mean"),
        )
    )
    pass_summary.to_csv(OUTPUT_ROOT / "geometry_window_summary.csv", index=False)
    audit = {
        "format": "strict8-geometry-replay-audit-v1",
        "protocol_id": protocol_id,
        **selection_summary,
        "branches_replayed": branches_total,
        "window_requests": int(pairs["window_id"].nunique()),
        "pair_comparisons": len(pairs),
        "comparisons_per_window": 28,
        "replay_mismatches": mismatch_total,
        "all_selected_windows_complete": True,
    }
    _write_json(OUTPUT_ROOT / "geometry_replay_audit.json", audit)
    _replace_checksums(GEOMETRY_ROOT)
    print(
        f"Exact geometry replay complete: {audit['window_requests']} windows, "
        f"{len(pairs)} pair comparisons",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Fresh common-random-stream composition intervention


def _intervention_seed(case: Any, branch: int) -> int:
    return derive_seed(
        INTERVENTION_FUTURE_SEED,
        "REGCONF.intervention.future",
        case.candidate,
        case.matrix_id,
        case.landmark,
        branch,
    )


def _edited_snapshot(snapshot: Snapshot, composition: np.ndarray) -> Snapshot:
    return Snapshot(
        composition=np.asarray(composition, dtype=np.int64),
        generation=snapshot.generation,
        inheritance=snapshot.inheritance,
        boundary_h=snapshot.boundary_h,
        previous_growth_steps=snapshot.previous_growth_steps,
        cumulative_growth_steps=snapshot.cumulative_growth_steps,
    )


def _assert_snapshot_preservation(original: Snapshot, edited: Snapshot) -> None:
    if int(original.composition.sum()) != int(edited.composition.sum()):
        raise AssertionError("intervention changed assembly mass")
    for field in (
        "generation",
        "inheritance",
        "boundary_h",
        "previous_growth_steps",
        "cumulative_growth_steps",
    ):
        if getattr(original, field) != getattr(edited, field):
            raise AssertionError(f"intervention changed snapshot field {field}")


def _intervention_worker(
    arguments: tuple[int, Any, Any, tuple[Any, ...], str]
) -> int:
    state_index, case, experiment, specs, protocol_id = arguments
    dataset = "intervention"
    output = _checkpoint_path(dataset, state_index)
    if output.is_file():
        try:
            with np.load(output, allow_pickle=False) as archive:
                if (
                    str(archive["format"].item()) == CHECKPOINT_FORMAT
                    and str(archive["dataset"].item()) == dataset
                    and str(archive["protocol_id"].item()) == protocol_id
                ):
                    return state_index
        except Exception:
            pass

    n_arms = len(ARM_NAMES)
    n_specs = len(specs)
    labels = np.zeros((n_arms, INTERVENTION_BRANCHES, n_specs), dtype=np.int8)
    gates = np.zeros_like(labels)
    completed = np.zeros((n_arms, INTERVENTION_BRANCHES), dtype=np.int8)
    observed = np.zeros((n_arms, INTERVENTION_BRANCHES), dtype=np.int16)
    requested = np.zeros(n_arms, dtype=np.int8)
    achieved = np.zeros(n_arms, dtype=np.int8)
    mass_before = np.zeros(n_arms, dtype=np.int16)
    mass_after = np.zeros(n_arms, dtype=np.int16)
    occupied_before = np.zeros(n_arms, dtype=np.int16)
    occupied_after = np.zeros(n_arms, dtype=np.int16)
    simpson_before = np.zeros(n_arms, dtype=np.float64)
    simpson_after = np.zeros(n_arms, dtype=np.float64)
    edited_compositions = np.zeros(
        (n_arms, len(case.snapshot.composition)), dtype=np.int16
    )
    concentration_after = np.zeros(
        (n_arms, len(CONCENTRATION_NAMES)), dtype=np.float64
    )
    with threadpool_limits(limits=1):
        for arm_index, arm in enumerate(ARM_NAMES):
            edit = apply_intervention(
                case.snapshot.composition,
                arm,
                case.state_id,
                INTERVENTION_SELECTION_SEED,
            )
            snapshot = _edited_snapshot(case.snapshot, edit.composition)
            _assert_snapshot_preservation(case.snapshot, snapshot)
            requested[arm_index] = edit.requested_dose
            achieved[arm_index] = edit.achieved_dose
            mass_before[arm_index] = edit.mass_before
            mass_after[arm_index] = edit.mass_after
            occupied_before[arm_index] = edit.occupied_before
            occupied_after[arm_index] = edit.occupied_after
            simpson_before[arm_index] = edit.simpson_before
            simpson_after[arm_index] = edit.simpson_after
            edited_compositions[arm_index] = edit.composition
            concentration_after[arm_index] = concentration_descriptors(edit.composition)
            for branch in range(INTERVENTION_BRANCHES):
                rng = np.random.default_rng(_intervention_seed(case, branch))
                records, complete = simulate_future_absorbing(
                    snapshot,
                    case.beta,
                    experiment.gard,
                    CANDIDATES[case.candidate],
                    INTERVENTION_HORIZON,
                    rng,
                )
                outcomes, _ = score_all_specs(records, specs)
                completed[arm_index, branch] = int(complete)
                observed[arm_index, branch] = len(records)
                for spec_index, outcome in enumerate(outcomes):
                    labels[arm_index, branch, spec_index] = int(outcome.event)
                    gates[arm_index, branch, spec_index] = int(outcome.deepest_gate)
    _atomic_npz(
        output,
        format=np.asarray(CHECKPOINT_FORMAT),
        dataset=np.asarray(dataset),
        state_index=np.asarray(state_index, dtype=np.int32),
        protocol_id=np.asarray(protocol_id),
        state_id=np.asarray(case.state_id),
        labels=labels,
        gates=gates,
        completed=completed,
        observed=observed,
        requested=requested,
        achieved=achieved,
        mass_before=mass_before,
        mass_after=mass_after,
        occupied_before=occupied_before,
        occupied_after=occupied_after,
        simpson_before=simpson_before,
        simpson_after=simpson_after,
        edited_compositions=edited_compositions,
        concentration_before=concentration_descriptors(case.snapshot.composition),
        concentration_after=concentration_after,
    )
    return state_index


INTERVENTION_KEYS = (
    "labels",
    "gates",
    "completed",
    "observed",
    "requested",
    "achieved",
    "mass_before",
    "mass_after",
    "occupied_before",
    "occupied_after",
    "simpson_before",
    "simpson_after",
    "edited_compositions",
    "concentration_before",
    "concentration_after",
)


def run_intervention(workers: int) -> None:
    protocol_id = _protocol_id()
    experiment, cases = _cases("confirmation", workers)
    specs = previous._specs()
    if len(cases) != 2000:
        raise AssertionError(f"expected 2000 confirmation states, found {len(cases)}")
    arguments = [
        (index, case, experiment, specs, protocol_id)
        for index, case in enumerate(cases)
    ]
    _run_workers("intervention", arguments, _intervention_worker, workers)

    values: dict[str, list[np.ndarray]] = {key: [] for key in INTERVENTION_KEYS}
    for index in range(len(cases)):
        if not _checkpoint_complete("intervention", index):
            raise ValueError(f"missing intervention checkpoint {index}")
        with np.load(_checkpoint_path("intervention", index), allow_pickle=False) as archive:
            for key in INTERVENTION_KEYS:
                values[key].append(np.asarray(archive[key]))
    arrays = {key: np.stack(parts) for key, parts in values.items()}
    metadata = previous._metadata(cases)
    INTERVENTION_ROOT.mkdir(parents=True, exist_ok=True)
    _atomic_npz(
        INTERVENTION_ROOT / "intervention_replay.npz",
        protocol_id=np.asarray(protocol_id),
        arm_names=np.asarray(ARM_NAMES),
        spec_names=np.asarray(SPEC_NAMES),
        **metadata,
        **arrays,
    )

    validation_rows: list[dict[str, Any]] = []
    for arm_index, arm in enumerate(ARM_NAMES):
        requested = arrays["requested"][:, arm_index]
        achieved = arrays["achieved"][:, arm_index]
        validation_rows.append(
            {
                "arm": arm,
                "states": len(cases),
                "requested_dose": int(requested[0]),
                "full_dose_states": int(np.count_nonzero(achieved == requested)),
                "full_dose_fraction": float(np.mean(achieved == requested)),
                "achieved_dose_mean": float(achieved.mean()),
                "mass_preserved_fraction": float(
                    np.mean(
                        arrays["mass_before"][:, arm_index]
                        == arrays["mass_after"][:, arm_index]
                    )
                ),
                "occupied_change_mean": float(
                    np.mean(
                        arrays["occupied_after"][:, arm_index]
                        - arrays["occupied_before"][:, arm_index]
                    )
                ),
                "simpson_change_mean": float(
                    np.mean(
                        arrays["simpson_after"][:, arm_index]
                        - arrays["simpson_before"][:, arm_index]
                    )
                ),
                "completed_horizon_fraction": float(
                    arrays["completed"][:, arm_index].mean()
                ),
                "observed_fissions_mean": float(
                    arrays["observed"][:, arm_index].mean()
                ),
            }
        )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(validation_rows).to_csv(
        OUTPUT_ROOT / "intervention_validation.csv", index=False
    )
    _replace_checksums(INTERVENTION_ROOT)
    print(
        f"Fresh intervention complete: {len(cases) * len(ARM_NAMES) * INTERVENTION_BRANCHES:,} futures",
        flush=True,
    )


def _load_intervention() -> dict[str, np.ndarray]:
    verify_checksums(INTERVENTION_ROOT)
    values = _load_npz(INTERVENTION_ROOT / "intervention_replay.npz")
    if str(values["protocol_id"].item()) != _protocol_id():
        raise ValueError("intervention replay protocol mismatch")
    if tuple(values["arm_names"].astype(str)) != ARM_NAMES:
        raise ValueError("intervention arm order mismatch")
    return values


def _effect_inference(
    state_effect: np.ndarray,
    matrix_ids: np.ndarray,
    seed_parts: Sequence[str],
) -> tuple[float, float, float, float]:
    values = np.asarray(state_effect, dtype=float)
    mids = np.asarray(matrix_ids, dtype=int)
    finite = np.isfinite(values)
    values = values[finite]
    mids = mids[finite]
    groups = _group_values(values, mids)
    observed = float(groups.mean())
    rng = np.random.default_rng(derive_seed(BOOTSTRAP_SEED, *seed_parts))
    indices = rng.integers(0, len(groups), size=(BOOTSTRAP_REPETITIONS, len(groups)))
    bootstrap = groups[indices].mean(axis=1)
    lower, upper = np.quantile(bootstrap, (0.025, 0.975))
    rng = np.random.default_rng(derive_seed(RANDOMIZATION_SEED, *seed_parts))
    signs = rng.integers(0, 2, size=(RANDOMIZATION_REPETITIONS, len(groups))) * 2 - 1
    null = signs @ groups / len(groups)
    p = float((np.count_nonzero(null >= observed) + 1) / (RANDOMIZATION_REPETITIONS + 1))
    return observed, float(lower), float(upper), p


def _intervention_q(
    gates: np.ndarray,
    arm_index: int,
    spec_index: int,
    branch_slice: slice,
    transition_index: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    selected = gates[:, arm_index, branch_slice, spec_index]
    if transition_index is None:
        return (selected >= 4).mean(axis=1), np.full(selected.shape[0], selected.shape[1])
    success, trials = aggregate_transitions(selected)
    numerator = success[:, transition_index]
    denominator = trials[:, transition_index]
    return np.divide(
        numerator,
        denominator,
        out=np.full(len(denominator), np.nan),
        where=denominator > 0,
    ), denominator


def _intervention_power_row(
    labels: np.ndarray,
    mids: np.ndarray,
    left_index: int,
    right_index: int,
) -> dict[str, Any]:
    left_events = int(labels[:, left_index].sum())
    right_events = int(labels[:, right_index].sum())
    left_matrices = int(
        sum(labels[mids == key, left_index].sum() > 0 for key in np.unique(mids))
    )
    right_matrices = int(
        sum(labels[mids == key, right_index].sum() > 0 for key in np.unique(mids))
    )
    discordant = int(np.count_nonzero(labels[:, left_index] != labels[:, right_index]))
    return {
        "left_events": left_events,
        "right_events": right_events,
        "left_event_matrices": left_matrices,
        "right_event_matrices": right_matrices,
        "discordant_branch_pairs": discordant,
        "power_adequate": left_events >= MIN_SUCCESSES
        and right_events >= MIN_SUCCESSES
        and left_matrices >= MIN_OUTCOME_MATRICES
        and right_matrices >= MIN_OUTCOME_MATRICES
        and discordant >= MIN_SUCCESSES,
    }


def analyze_intervention() -> None:
    data = _load_intervention()
    arm_index = {name: index for index, name in enumerate(ARM_NAMES)}
    candidates = np.asarray(data["candidates"]).astype(str)
    matrix_ids = np.asarray(data["matrix_ids"], dtype=int)
    labels = np.asarray(data["labels"], dtype=np.int8)
    gates = np.asarray(data["gates"], dtype=np.int8)
    achieved = np.asarray(data["achieved"], dtype=int)
    primary_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    descriptive_rows: list[dict[str, Any]] = []
    dose_rows: list[dict[str, Any]] = []
    branch_halves = (("A", slice(0, 32)), ("B", slice(32, 64)))

    for contrast, (left_arm, right_arm) in PRIMARY_INTERVENTION_CONTRASTS.items():
        left_index, right_index = arm_index[left_arm], arm_index[right_arm]
        for spec_index, spec in enumerate(SPEC_NAMES):
            for candidate in ("02", "03"):
                state_selection = candidates == candidate
                mids = matrix_ids[state_selection]
                full = (
                    (achieved[state_selection, left_index] == 4)
                    & (achieved[state_selection, right_index] == 4)
                )
                for half, branch_slice in branch_halves:
                    left_q, _ = _intervention_q(
                        gates[state_selection], left_index, spec_index, branch_slice, None
                    )
                    right_q, _ = _intervention_q(
                        gates[state_selection], right_index, spec_index, branch_slice, None
                    )
                    effect, lower, upper, p = _effect_inference(
                        left_q - right_q,
                        mids,
                        ("intervention", contrast, spec, candidate, half, "itt"),
                    )
                    power = _intervention_power_row(
                        labels[state_selection, :, branch_slice, spec_index],
                        mids,
                        left_index,
                        right_index,
                    )
                    if np.count_nonzero(full) >= 2:
                        sensitivity = _effect_inference(
                            left_q[full] - right_q[full],
                            mids[full],
                            (
                                "intervention",
                                contrast,
                                spec,
                                candidate,
                                half,
                                "full-dose",
                            ),
                        )
                    else:
                        sensitivity = (np.nan, np.nan, np.nan, np.nan)
                    primary_rows.append(
                        {
                            "contrast": contrast,
                            "left_arm": left_arm,
                            "right_arm": right_arm,
                            "spec": spec,
                            "candidate": candidate,
                            "half": half,
                            "states": int(state_selection.sum()),
                            "matrices": int(np.unique(mids).size),
                            "left_event_rate": float(left_q.mean()),
                            "right_event_rate": float(right_q.mean()),
                            "event_rate_effect": effect,
                            "ci95_lower": lower,
                            "ci95_upper": upper,
                            "randomization_p_raw": p,
                            "full_dose_states": int(full.sum()),
                            "full_dose_fraction": float(full.mean()),
                            "full_dose_effect": sensitivity[0],
                            "full_dose_ci95_lower": sensitivity[1],
                            "full_dose_ci95_upper": sensitivity[2],
                            "full_dose_randomization_p": sensitivity[3],
                            **power,
                        }
                    )
                    for transition_index, transition in enumerate(TRANSITION_NAMES):
                        left_t, left_trials = _intervention_q(
                            gates[state_selection],
                            left_index,
                            spec_index,
                            branch_slice,
                            transition_index,
                        )
                        right_t, right_trials = _intervention_q(
                            gates[state_selection],
                            right_index,
                            spec_index,
                            branch_slice,
                            transition_index,
                        )
                        keep = np.isfinite(left_t) & np.isfinite(right_t)
                        if keep.sum() >= 2:
                            transition_effect = _effect_inference(
                                left_t[keep] - right_t[keep],
                                mids[keep],
                                (
                                    "intervention-gate",
                                    contrast,
                                    spec,
                                    candidate,
                                    half,
                                    transition,
                                ),
                            )
                        else:
                            transition_effect = (np.nan, np.nan, np.nan, 1.0)
                        gate_rows.append(
                            {
                                "contrast": contrast,
                                "spec": spec,
                                "candidate": candidate,
                                "half": half,
                                "transition": transition,
                                "eligible_states": int(keep.sum()),
                                "left_trials": int(np.nansum(left_trials)),
                                "right_trials": int(np.nansum(right_trials)),
                                "left_rate": float(np.nanmean(left_t)),
                                "right_rate": float(np.nanmean(right_t)),
                                "rate_effect": transition_effect[0],
                                "ci95_lower": transition_effect[1],
                                "ci95_upper": transition_effect[2],
                                "randomization_p_raw": transition_effect[3],
                            }
                        )

    primary = pd.DataFrame(primary_rows)
    primary["randomization_p_holm"] = np.nan
    for contrast in PRIMARY_INTERVENTION_CONTRASTS:
        selected = primary["contrast"] == contrast
        if int(selected.sum()) != 12:
            raise AssertionError("primary Holm family must contain 12 cells")
        primary.loc[selected, "randomization_p_holm"] = holm_adjust(
            primary.loc[selected, "randomization_p_raw"].tolist()
        )
    primary["passes_primary_gate"] = (
        primary["power_adequate"]
        & (primary["full_dose_fraction"] >= FULL_DOSE_VALIDITY)
        & (primary["event_rate_effect"] > 0)
        & (primary["ci95_lower"] > 0)
        & (primary["randomization_p_holm"] < 0.05)
    )

    gate_table = pd.DataFrame(gate_rows)
    gate_table["randomization_p_holm"] = np.nan
    for contrast in PRIMARY_INTERVENTION_CONTRASTS:
        for transition in TRANSITION_NAMES:
            selected = (
                (gate_table["contrast"] == contrast)
                & (gate_table["transition"] == transition)
            )
            gate_table.loc[selected, "randomization_p_holm"] = holm_adjust(
                gate_table.loc[selected, "randomization_p_raw"].tolist()
            )

    noop_index = arm_index["NOOP"]
    for arm in ARM_NAMES[1:]:
        test_index = arm_index[arm]
        for spec_index, spec in enumerate(SPEC_NAMES):
            for candidate in ("02", "03"):
                selected = candidates == candidate
                mids = matrix_ids[selected]
                for half, branch_slice in branch_halves:
                    test_q, _ = _intervention_q(
                        gates[selected], test_index, spec_index, branch_slice, None
                    )
                    noop_q, _ = _intervention_q(
                        gates[selected], noop_index, spec_index, branch_slice, None
                    )
                    effect = _effect_inference(
                        test_q - noop_q,
                        mids,
                        ("descriptive", arm, spec, candidate, half),
                    )
                    descriptive_rows.append(
                        {
                            "arm": arm,
                            "reference": "NOOP",
                            "spec": spec,
                            "candidate": candidate,
                            "half": half,
                            "arm_event_rate": float(test_q.mean()),
                            "noop_event_rate": float(noop_q.mean()),
                            "event_rate_effect": effect[0],
                            "ci95_lower": effect[1],
                            "ci95_upper": effect[2],
                            "randomization_p_raw": effect[3],
                        }
                    )

    dose_contrasts = {
        ("evenness", 1): ("EVEN_CONCENTRATE_D1", "EVEN_FLATTEN_D1"),
        ("evenness", 4): ("EVEN_CONCENTRATE_D4", "EVEN_FLATTEN_D4"),
        ("richness", 1): ("RICH_CONTRACT_D1", "RICH_EXPAND_D1"),
        ("richness", 4): ("RICH_CONTRACT_D4", "RICH_EXPAND_D4"),
    }
    for (axis, dose), (left_arm, right_arm) in dose_contrasts.items():
        left_index, right_index = arm_index[left_arm], arm_index[right_arm]
        for spec_index, spec in enumerate(SPEC_NAMES):
            for candidate in ("02", "03"):
                selected = candidates == candidate
                mids = matrix_ids[selected]
                for half, branch_slice in branch_halves:
                    left_q, _ = _intervention_q(
                        gates[selected], left_index, spec_index, branch_slice, None
                    )
                    right_q, _ = _intervention_q(
                        gates[selected], right_index, spec_index, branch_slice, None
                    )
                    effect = _effect_inference(
                        left_q - right_q,
                        mids,
                        ("dose", axis, str(dose), spec, candidate, half),
                    )
                    dose_rows.append(
                        {
                            "axis": axis,
                            "dose": dose,
                            "left_arm": left_arm,
                            "right_arm": right_arm,
                            "spec": spec,
                            "candidate": candidate,
                            "half": half,
                            "left_event_rate": float(left_q.mean()),
                            "right_event_rate": float(right_q.mean()),
                            "event_rate_effect": effect[0],
                            "ci95_lower": effect[1],
                            "ci95_upper": effect[2],
                            "randomization_p_raw": effect[3],
                        }
                    )

    primary.to_csv(OUTPUT_ROOT / "intervention_primary_effects.csv", index=False)
    gate_table.to_csv(OUTPUT_ROOT / "intervention_gate_effects.csv", index=False)
    pd.DataFrame(descriptive_rows).to_csv(
        OUTPUT_ROOT / "intervention_arm_vs_noop.csv", index=False
    )
    pd.DataFrame(dose_rows).to_csv(
        OUTPUT_ROOT / "intervention_dose_diagnostics.csv", index=False
    )
    print("Fresh intervention effects and gate localization scored", flush=True)


# ---------------------------------------------------------------------------
# Readout, figures, verification, and command-line orchestration


def _result_classification() -> dict[str, Any]:
    stage = pd.read_csv(OUTPUT_ROOT / "stage_prediction.csv")
    margins = pd.read_csv(OUTPUT_ROOT / "margin_prediction.csv")
    primary = pd.read_csv(OUTPUT_ROOT / "intervention_primary_effects.csv")
    validation = pd.read_csv(OUTPUT_ROOT / "intervention_validation.csv")
    geometry = pd.read_csv(OUTPUT_ROOT / "geometry_rank_band_summary.csv")
    gate = pd.read_csv(OUTPUT_ROOT / "intervention_gate_effects.csv")

    stage_summary = (
        stage.groupby(["transition", "contrast"], as_index=False)
        .agg(
            cells=("log_loss_gain", "size"),
            power_adequate_cells=("confirmation_power_adequate", "sum"),
            mean_gain=("log_loss_gain", "mean"),
            positive_cells=("log_loss_gain", lambda value: int((value > 0).sum())),
            passing_cells=("passes_exploratory_gate", "sum"),
        )
    )
    margin_summary = (
        margins.groupby(["margin", "contrast"], as_index=False)
        .agg(
            cells=("mse_gain", "size"),
            power_adequate_cells=("power_adequate", "sum"),
            mean_gain=("mse_gain", "mean"),
            positive_cells=("mse_gain", lambda value: int((value > 0).sum())),
            passing_cells=("passes_exploratory_gate", "sum"),
        )
    )
    primary_summary = (
        primary.groupby("contrast", as_index=False)
        .agg(
            cells=("event_rate_effect", "size"),
            power_adequate_cells=("power_adequate", "sum"),
            mean_effect=("event_rate_effect", "mean"),
            positive_cells=("event_rate_effect", lambda value: int((value > 0).sum())),
            passing_cells=("passes_primary_gate", "sum"),
            minimum_full_dose_fraction=("full_dose_fraction", "min"),
        )
    )
    gate_summary = (
        gate.groupby(["contrast", "transition"], as_index=False)
        .agg(
            mean_effect=("rate_effect", "mean"),
            positive_cells=("rate_effect", lambda value: int((value > 0).sum())),
            holm_significant_cells=(
                "randomization_p_holm", lambda value: int((value < 0.05).sum())
            ),
        )
    )
    geometry_summary = (
        geometry.groupby(["role", "rank_band"], as_index=False)[
            "share_of_total_bray_distance"
        ]
        .mean()
        .rename(columns={"share_of_total_bray_distance": "mean_distance_share"})
    )
    value = {
        "format": "strict8-mechanism-result-classification-v1",
        "protocol_id": _protocol_id(),
        "claim_status": "post_hoc_internal_diagnosis",
        "stage_prediction": stage_summary.to_dict(orient="records"),
        "continuous_margins": margin_summary.to_dict(orient="records"),
        "causal_intervention": primary_summary.to_dict(orient="records"),
        "causal_gate_localization": gate_summary.to_dict(orient="records"),
        "geometry_rank_bands": geometry_summary.to_dict(orient="records"),
        "intervention_minimum_full_dose_fraction": float(
            validation.loc[validation["requested_dose"] == 4, "full_dose_fraction"].min()
        ),
        "interpretation_rule": (
            "A positive cell is directional only. A passing prediction cell additionally "
            "requires adequate development and confirmation support, a positive 95% "
            "matrix-bootstrap lower bound, and Holm-adjusted one-sided p<0.05. A passing "
            "causal cell additionally requires >=90% joint full-dose feasibility."
        ),
    }
    return _json_ready(value)


def _make_figures() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_root = OUTPUT_ROOT / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)

    stage = pd.read_csv(OUTPUT_ROOT / "stage_prediction.csv")
    stage_plot = stage.pivot_table(
        index="transition", columns="contrast", values="log_loss_gain", aggfunc="mean"
    ).reindex(index=TRANSITION_NAMES, columns=PREDICTION_CONTRASTS)
    fig, axis = plt.subplots(figsize=(9.2, 4.5))
    image_handle = axis.imshow(stage_plot.to_numpy(), cmap="RdBu_r", aspect="auto")
    axis.set_xticks(range(len(stage_plot.columns)), stage_plot.columns, rotation=25, ha="right")
    axis.set_yticks(range(len(stage_plot.index)), stage_plot.index)
    axis.set_title("Held-out log-loss gain by strict-8 transition")
    for row in range(stage_plot.shape[0]):
        for column in range(stage_plot.shape[1]):
            value = stage_plot.iloc[row, column]
            axis.text(column, row, f"{value:.4f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image_handle, ax=axis, label="baseline loss − enhanced loss")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(figure_root / f"stage_prediction_gains.{extension}", dpi=180)
    plt.close(fig)

    reliability = pd.read_csv(OUTPUT_ROOT / "reliability_by_budget.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for axis, spec in zip(axes, SPEC_NAMES, strict=True):
        selected = reliability[reliability["spec"] == spec]
        for (candidate, transition), group in selected.groupby(["candidate", "transition"]):
            group = group.sort_values("branch_budget_per_half")
            axis.plot(
                group["branch_budget_per_half"],
                group["centered_spearman_mean"],
                marker="o",
                label=f"c{str(candidate).zfill(2)} {transition}",
            )
        axis.set_title(spec)
        axis.set_xlabel("branches per estimate")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("matrix-centered split-half Spearman")
    handles, labels_text = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels_text, loc="lower center", ncol=4, fontsize=7)
    fig.suptitle("Reliability of the four strict-8 transitions")
    fig.tight_layout(rect=(0, 0.15, 1, 0.94))
    for extension in ("png", "pdf"):
        fig.savefig(figure_root / f"transition_reliability.{extension}", dpi=180)
    plt.close(fig)

    geometry = pd.read_csv(OUTPUT_ROOT / "geometry_rank_band_summary.csv")
    geometry["group"] = (
        geometry["role"]
        + " | "
        + geometry["spec"]
        + " | c"
        + geometry["candidate"].astype(str).str.zfill(2)
    )
    geometry_plot = geometry.pivot_table(
        index="group",
        columns="rank_band",
        values="share_of_total_bray_distance",
        aggfunc="mean",
    ).fillna(0.0)
    geometry_plot = geometry_plot.reindex(columns=["top1", "rank2_to5", "tail6plus"])
    fig, axis = plt.subplots(figsize=(10, 6))
    geometry_plot.plot(kind="barh", stacked=True, ax=axis)
    axis.set_xlabel("share of summed Bray–Curtis distance")
    axis.set_ylabel("")
    axis.set_title("Where pairwise compositional distance resides")
    axis.legend(title="abundance rank")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(figure_root / f"geometry_rank_contributions.{extension}", dpi=180)
    plt.close(fig)

    primary = pd.read_csv(OUTPUT_ROOT / "intervention_primary_effects.csv")
    primary["label"] = (
        primary["contrast"].str.replace("_", " ")
        + " | "
        + primary["spec"]
        + " | c"
        + primary["candidate"].astype(str).str.zfill(2)
        + " | "
        + primary["half"]
    )
    primary = primary.sort_values(["contrast", "spec", "candidate", "half"]).reset_index(drop=True)
    fig, axis = plt.subplots(figsize=(10, 9))
    positions = np.arange(len(primary))
    effects = primary["event_rate_effect"].to_numpy()
    lower = effects - primary["ci95_lower"].to_numpy()
    upper = primary["ci95_upper"].to_numpy() - effects
    colors = np.where(primary["passes_primary_gate"].astype(bool), "#087E8B", "#6C757D")
    axis.errorbar(effects, positions, xerr=np.vstack((lower, upper)), fmt="none", ecolor=colors, capsize=2)
    axis.scatter(effects, positions, c=colors, s=28)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_yticks(positions, primary["label"], fontsize=7)
    axis.set_xlabel("event-probability effect (left policy − right policy)")
    axis.set_title("Fresh composition intervention: matrix-bootstrap 95% intervals")
    axis.invert_yaxis()
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(figure_root / f"intervention_primary_effects.{extension}", dpi=180)
    plt.close(fig)


def report() -> None:
    classification = _result_classification()
    _write_json(OUTPUT_ROOT / "result_classification.json", classification)
    _make_figures()

    stage = pd.DataFrame(classification["stage_prediction"])
    margin = pd.DataFrame(classification["continuous_margins"])
    causal = pd.DataFrame(classification["causal_intervention"])
    gate = pd.DataFrame(classification["causal_gate_localization"])
    stage_lines = "\n".join(
        f"- {row.transition}, {row.contrast}: mean held-out gain {row.mean_gain:.5f}; "
        f"{int(row.positive_cells)}/{int(row.cells)} positive, "
        f"{int(row.passing_cells)} pass the frozen exploratory gate."
        for row in stage.itertuples()
    )
    margin_lines = "\n".join(
        f"- {row.margin}, {row.contrast}: mean MSE gain {row.mean_gain:.6g}; "
        f"{int(row.positive_cells)}/{int(row.cells)} positive, "
        f"{int(row.passing_cells)} pass."
        for row in margin.itertuples()
    )
    causal_lines = "\n".join(
        f"- {row.contrast}: mean event-rate effect {row.mean_effect:+.4f}; "
        f"{int(row.positive_cells)}/{int(row.cells)} directional positives, "
        f"{int(row.passing_cells)} primary passing cells; minimum joint full-dose "
        f"fraction {row.minimum_full_dose_fraction:.3f}."
        for row in causal.itertuples()
    )
    strongest_gate = gate.iloc[int(np.nanargmax(np.abs(gate["mean_effect"].to_numpy())))]
    technical = f"""# Strict-8 prediction-mechanism diagnosis

Status: post-hoc internal diagnosis. This report does not edit or supply wording for the manuscript.

## Question

Why is the frozen predictor weak or unstable for the strict coherent-eight endpoint, and do starting-composition concentration, residual state/network information, finite-branch label noise, or the endpoint's multi-gate geometry explain it?

## Frozen design

- All three endpoint implementations have equal status: cosine registered, globally calibrated Bray–Curtis, and relation-specific Bray–Curtis.
- Models were fitted on development only and sealed before confirmation scoring.
- Four conditional transitions were evaluated: first break, run of eight after a break, mutual coherence after a run, and old-anchor separation after coherence.
- H, H+C, H+S, and H+C+S compare the retained 10-variable `h10` history block, six exact concentration summaries, and the 26-dimensional state block.
- The fresh intervention reuses all 2,000 retained confirmation states, 11 frozen arms, 64 common-random-stream branches per arm, and 32 future fissions: {2000 * len(ARM_NAMES) * INTERVENTION_BRANCHES:,} newly scored futures.

## Conditional prediction

{stage_lines}

## Continuous margins

{margin_lines}

## Finite-branch reliability

See `reliability_by_budget.csv` and `figures/transition_reliability.png`. These quantify whether apparent between-state risk differences stabilize as the branch budget rises from 8 to 64, rather than treating a noisy 128-branch label as ground truth.

## Exact strict-window geometry

Every selected event window and its frozen same-state control was exactly replayed. `geometry_window_summary.csv`, `geometry_rank_band_summary.csv`, and the compressed pair table identify which of the 28 pairwise comparisons bind and whether Bray–Curtis distance is carried by the dominant type, ranks 2–5, or the tail.

## Fresh causal perturbation

{causal_lines}

The largest absolute mean localized gate effect is currently {strongest_gate.contrast} at {strongest_gate.transition} ({strongest_gate.mean_effect:+.4f}). This localization is diagnostic: it distinguishes effects on reaching a renewed run from effects on mutual coherence or separation from the old anchor.

## Interpretation boundary

Prediction gains are held-out associations among retained surviving/observable selected-lineage states. The intervention supplies causal evidence only for the exact one- or four-molecule editing policies, on those same retained states, under the simulator and common random streams. A null result can mean either that the proposed axis is not causal at these doses or that the strict event is too rare/noisy for the available branch budget; the power and reliability tables distinguish those cases where possible.

## Audit trail

The protocol, model seals, exact replay audit, full-dose validation, deterministic replay audit, checksums, and result manifest are all under this subfolder. No new campaign states or new landmark futures were generated; only futures from already retained states were rescored.
"""
    (TASK_ROOT / "DIAGNOSTIC_REPORT.md").write_text(technical, encoding="utf-8")

    total_passes = int(causal["passing_cells"].sum())
    stage_passes = int(stage["passing_cells"].sum())
    lay = f"""# Strict-eight diagnosis — lay summary

We broke the rare “strict eight” outcome into four hurdles instead of asking only whether the whole event happened. We then asked whether a state's recent history, how concentrated its composition is, and the remaining detailed state/network information help predict each hurdle.

Across those hurdle tests, {stage_passes} cells met the deliberately strict evidence rule. The detailed table matters more than this count because a late hurdle can have too few failures to estimate reliably even when the earlier hurdles are well measured.

We also reran the exact eight-daughter windows and inspected all 28 daughter-to-daughter comparisons. This shows whether a window fails because of one troublesome daughter pair and whether its apparent similarity is mainly caused by one dominant molecule or is spread across the composition.

Finally, we deliberately moved either one or four molecules in each retained starting assembly. One pair of edits made the composition more versus less even; another removed versus added occupied molecule types. The same random future stream was used for each competing edit. Across the two causal contrasts, {total_passes} of 24 endpoint/candidate/replicate cells passed the full frozen rule. The direction, uncertainty, dose comparison, and exact gate affected are recorded in the intervention tables; the technical report avoids turning isolated positive cells into a general mechanism claim.

This is a diagnosis of why prediction succeeds or fails on the available states. It does not change the original confirmatory result, and it does not by itself show that every assembly generated from scratch behaves the same way.
"""
    (TASK_ROOT / "LAY_FINDINGS.md").write_text(lay, encoding="utf-8")
    print("Technical report, lay summary, figures, and classification written", flush=True)


REQUIRED_OUTPUT_FILES = (
    "transition_power.csv",
    "stage_prediction.csv",
    "margin_prediction.csv",
    "reliability_chunk_pairs.csv",
    "reliability_by_budget.csv",
    "geometry_rank_band_summary.csv",
    "geometry_window_minima.csv.gz",
    "geometry_window_summary.csv",
    "geometry_replay_audit.json",
    "intervention_validation.csv",
    "intervention_primary_effects.csv",
    "intervention_gate_effects.csv",
    "intervention_arm_vs_noop.csv",
    "intervention_dose_diagnostics.csv",
    "result_classification.json",
)


def _deterministic_intervention_audit(workers: int = 4) -> dict[str, Any]:
    data = _load_intervention()
    experiment, cases = _cases("confirmation", workers)
    specs = previous._specs()
    state_indices = (0, 5, 1000, 1005)
    branches = (0, 63)
    mismatches = 0
    edit_mismatches = 0
    futures = 0
    for state_index in state_indices:
        case = cases[state_index]
        for arm_index, arm in enumerate(ARM_NAMES):
            edit = apply_intervention(
                case.snapshot.composition,
                arm,
                case.state_id,
                INTERVENTION_SELECTION_SEED,
            )
            if not np.array_equal(
                edit.composition,
                data["edited_compositions"][state_index, arm_index],
            ):
                edit_mismatches += 1
            snapshot = _edited_snapshot(case.snapshot, edit.composition)
            _assert_snapshot_preservation(case.snapshot, snapshot)
            for branch in branches:
                rng = np.random.default_rng(_intervention_seed(case, branch))
                records, complete = simulate_future_absorbing(
                    snapshot,
                    case.beta,
                    experiment.gard,
                    CANDIDATES[case.candidate],
                    INTERVENTION_HORIZON,
                    rng,
                )
                outcomes, _ = score_all_specs(records, specs)
                expected_labels = np.asarray([int(value.event) for value in outcomes])
                expected_gates = np.asarray([value.deepest_gate for value in outcomes])
                if (
                    not np.array_equal(
                        expected_labels,
                        data["labels"][state_index, arm_index, branch],
                    )
                    or not np.array_equal(
                        expected_gates,
                        data["gates"][state_index, arm_index, branch],
                    )
                    or int(complete)
                    != int(data["completed"][state_index, arm_index, branch])
                    or len(records)
                    != int(data["observed"][state_index, arm_index, branch])
                ):
                    mismatches += 1
                futures += 1
    return {
        "state_indices": list(state_indices),
        "branches": list(branches),
        "arms": list(ARM_NAMES),
        "futures_replayed": futures,
        "edit_mismatches": edit_mismatches,
        "future_mismatches": mismatches,
        "exact": edit_mismatches == 0 and mismatches == 0,
    }


def verify(workers: int = 4) -> None:
    protocol = verify_protocol()
    verify_checksums(FEATURE_ROOT)
    verify_complete_model_seal()
    verify_checksums(GEOMETRY_ROOT)
    verify_checksums(INTERVENTION_ROOT)
    missing = [name for name in REQUIRED_OUTPUT_FILES if not (OUTPUT_ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing outputs: {missing}")
    if not (TASK_ROOT / "DIAGNOSTIC_REPORT.md").is_file() or not (
        TASK_ROOT / "LAY_FINDINGS.md"
    ).is_file():
        raise FileNotFoundError("reports are incomplete")

    stage = pd.read_csv(OUTPUT_ROOT / "stage_prediction.csv")
    margin = pd.read_csv(OUTPUT_ROOT / "margin_prediction.csv")
    primary = pd.read_csv(OUTPUT_ROOT / "intervention_primary_effects.csv")
    gates = pd.read_csv(OUTPUT_ROOT / "intervention_gate_effects.csv")
    doses = pd.read_csv(OUTPUT_ROOT / "intervention_dose_diagnostics.csv")
    expected_rows = {"stage": 144, "margin": 72, "primary": 24, "gates": 96, "doses": 48}
    observed_rows = {
        "stage": len(stage),
        "margin": len(margin),
        "primary": len(primary),
        "gates": len(gates),
        "doses": len(doses),
    }
    if observed_rows != expected_rows:
        raise AssertionError(f"table dimensions differ: {observed_rows}")
    if primary["randomization_p_holm"].isna().any():
        raise AssertionError("primary Holm family is incomplete")

    geometry_audit = json.loads((OUTPUT_ROOT / "geometry_replay_audit.json").read_text())
    if geometry_audit["replay_mismatches"] != 0:
        raise AssertionError("original-future geometry replay was not exact")
    if geometry_audit["pair_comparisons"] != 28 * geometry_audit["window_requests"]:
        raise AssertionError("geometry replay does not contain 28 pairs per window")

    intervention = _load_intervention()
    expected_shape = (2000, len(ARM_NAMES), INTERVENTION_BRANCHES, len(SPEC_NAMES))
    if tuple(intervention["labels"].shape) != expected_shape:
        raise AssertionError(f"intervention shape {intervention['labels'].shape} != {expected_shape}")
    if not np.array_equal(intervention["mass_before"], intervention["mass_after"]):
        raise AssertionError("mass preservation failed")
    if np.any(intervention["edited_compositions"] < 0):
        raise AssertionError("negative edited composition found")
    validation = pd.read_csv(OUTPUT_ROOT / "intervention_validation.csv")
    if not np.allclose(validation["mass_preserved_fraction"], 1.0):
        raise AssertionError("validation table reports a mass violation")

    deterministic = _deterministic_intervention_audit(workers)
    if not deterministic["exact"]:
        raise AssertionError(f"deterministic replay failed: {deterministic}")
    _write_json(OUTPUT_ROOT / "deterministic_replay_audit.json", deterministic)

    audit = {
        "format": "strict8-mechanism-verification-v1",
        "protocol_id": protocol["protocol_id"],
        "verified": True,
        "expected_table_rows": expected_rows,
        "geometry": geometry_audit,
        "intervention_shape": list(expected_shape),
        "fresh_futures": int(np.prod(expected_shape[:3])),
        "deterministic_replay": deterministic,
        "model_seal_verified": True,
        "input_checksum_contract_verified": True,
        "no_manuscript_files_modified": True,
    }
    _write_json(OUTPUT_ROOT / "verification_audit.json", audit)

    manifest_files = sorted(
        path
        for path in OUTPUT_ROOT.rglob("*")
        if path.is_file() and path.name not in {"SHA256SUMS", "result_manifest.json"}
    )
    manifest = {
        "format": RESULT_FORMAT,
        "protocol_id": protocol["protocol_id"],
        "files": [
            {
                "path": str(path.relative_to(TASK_ROOT)),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in manifest_files
        ],
    }
    manifest["manifest_id"] = canonical_digest(manifest)
    _write_json(OUTPUT_ROOT / "result_manifest.json", manifest)
    _replace_checksums(OUTPUT_ROOT)
    verified_files = len(verify_checksums(OUTPUT_ROOT))
    print(
        f"VERIFIED: {verified_files} output files; {audit['fresh_futures']:,} fresh futures; "
        "deterministic replay exact",
        flush=True,
    )


def status() -> None:
    selections, _ = _geometry_selection()
    value = {
        "protocol_frozen": (PROTOCOL_ROOT / "analysis_protocol.json").is_file(),
        "features_complete": all(
            (FEATURE_ROOT / f"{cohort}_concentration.npz").is_file()
            for cohort in ("development", "confirmation")
        ),
        "stage_models_sealed": (MODEL_ROOT / "stage_model_seal.json").is_file(),
        "margin_models_sealed": (MODEL_ROOT / "complete_model_seal.json").is_file(),
        "stage_scored": (OUTPUT_ROOT / "stage_prediction.csv").is_file(),
        "margins_scored": (OUTPUT_ROOT / "margin_prediction.csv").is_file(),
        "reliability_complete": (OUTPUT_ROOT / "reliability_by_budget.csv").is_file(),
        "geometry_checkpoints": sum(
            _checkpoint_path("geometry_replay", index).is_file() for index in selections
        ),
        "geometry_checkpoints_expected": len(selections),
        "intervention_checkpoints": sum(
            _checkpoint_path("intervention", index).is_file() for index in range(2000)
        ),
        "intervention_checkpoints_expected": 2000,
        "reports_complete": (TASK_ROOT / "DIAGNOSTIC_REPORT.md").is_file()
        and (TASK_ROOT / "LAY_FINDINGS.md").is_file(),
        "verified": (OUTPUT_ROOT / "verification_audit.json").is_file(),
    }
    print(json.dumps(value, indent=2), flush=True)


def run_all(workers: int) -> None:
    prepare()
    build_features(workers)
    fit_stage_models()
    fit_margin_models()
    score_stage_models()
    score_margin_models()
    compute_reliability()
    replay_geometry(workers)
    run_intervention(workers)
    analyze_intervention()
    report()
    verify(min(workers, 4))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Checkpointed strict-8 prediction and causal mechanism diagnosis"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    worker_commands = ("features", "geometry", "intervention", "verify", "all")
    for command in worker_commands:
        item = subparsers.add_parser(command)
        item.add_argument(
            "--workers",
            type=int,
            default=min(14, max(1, (os.cpu_count() or 2) - 1)),
        )
    for command in (
        "prepare",
        "fit-stage",
        "fit-margin",
        "score-stage",
        "score-margin",
        "reliability",
        "analyze-intervention",
        "report",
        "status",
    ):
        subparsers.add_parser(command)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    workers = getattr(arguments, "workers", 1)
    if workers < 1:
        raise ValueError("workers must be positive")
    commands: dict[str, Callable[[], None]] = {
        "prepare": prepare,
        "features": lambda: build_features(workers),
        "fit-stage": fit_stage_models,
        "fit-margin": fit_margin_models,
        "score-stage": score_stage_models,
        "score-margin": score_margin_models,
        "reliability": compute_reliability,
        "geometry": lambda: replay_geometry(workers),
        "intervention": lambda: run_intervention(workers),
        "analyze-intervention": analyze_intervention,
        "report": report,
        "verify": lambda: verify(workers),
        "status": status,
        "all": lambda: run_all(workers),
    }
    commands[arguments.command]()


if __name__ == "__main__":
    main()
