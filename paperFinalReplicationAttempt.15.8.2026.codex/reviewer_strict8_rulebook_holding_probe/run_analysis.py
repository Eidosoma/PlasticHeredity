"""Checkpointed post-hoc rulebook and holding-capacity probe for strict-8.

This follow-up is intentionally separate from the original confirmation and
from the quarantined NewIdeas program.  It derives deterministic forms only
from the frozen simulator equations and beta matrices, cross-fits empirical
holding scores across candidates/landmarks/branch halves, and runs a fresh
common-random-stream alignment intervention.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence


TASK_ROOT = Path(__file__).resolve().parent
PAPER_ROOT = TASK_ROOT.parent
WORKSPACE_ROOT = PAPER_ROOT.parent
SOURCE_ROOT = WORKSPACE_ROOT / "replicators.13.8.2026.codex"
GEOMETRY_ROOT = PAPER_ROOT / "reviewer_strict_event_geometry_audit"
MECHANISM_ROOT = PAPER_ROOT / "reviewer_strict8_prediction_mechanism_diagnosis"
os.environ.setdefault("MPLCONFIGDIR", str(TASK_ROOT / "artifacts" / "matplotlib"))
sys.path.insert(0, str(SOURCE_ROOT))
sys.path.insert(0, str(GEOMETRY_ROOT))
sys.path.insert(0, str(TASK_ROOT))

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

from plastic_heredity.config import CANDIDATES
from plastic_heredity.mechanistic import verify_checksums, write_checksums
from plastic_heredity.mechanistic_metrics import holm_adjust
from plastic_heredity.mechanistic_v2_models import (
    RIDGE_LAMBDAS,
    fit_block_transform,
    fit_linear,
    matrix_cv_fold,
)
from plastic_heredity.regime_confirmation import (
    CONFIRMATION_MASTER_SEED,
    DEVELOPMENT_MASTER_SEED,
    _experiment as regime_experiment,
)
from plastic_heredity.seeds import derive_seed
from plastic_heredity.simulator import (
    Snapshot,
    generate_beta,
    simulate_future_absorbing,
)

from rulebook_core import (
    EDIT_ARMS,
    RULEBOOK_FEATURE_NAMES,
    aggregate_transitions,
    apply_rulebook_edit,
    cosine,
    nearest_form,
    rulebook_features,
    smoothed_rate,
    solve_rulebook,
    tangent_stability_margin,
)
from strict_core import (
    RUN_LENGTH,
    SPEC_NAMES,
    build_geometry,
    score_all_specs,
    window_pairwise_minimum,
)


def _load_geometry_runner():
    path = GEOMETRY_ROOT / "run_analysis.py"
    spec = importlib.util.spec_from_file_location("rulebook_geometry_source", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


geometry_source = _load_geometry_runner()

ARTIFACT_ROOT = TASK_ROOT / "artifacts"
PROTOCOL_ROOT = ARTIFACT_ROOT / "protocol"
RULEBOOK_ROOT = ARTIFACT_ROOT / "rulebooks"
HOLDING_ROOT = ARTIFACT_ROOT / "holding"
MODEL_ROOT = ARTIFACT_ROOT / "models"
INTERVENTION_ROOT = ARTIFACT_ROOT / "intervention"
WORK_ROOT = ARTIFACT_ROOT / "work"
OUTPUT_ROOT = ARTIFACT_ROOT / "output"

DEVELOPMENT_SOURCE = SOURCE_ROOT / "results" / "regime_development"
CONFIRMATION_SOURCE = SOURCE_ROOT / "results" / "regime_confirmation"
GEOMETRY_REPLAY_ROOT = GEOMETRY_ROOT / "artifacts" / "replays"
CONCENTRATION_ROOT = MECHANISM_ROOT / "artifacts" / "features"

TRANSITION_NAMES = (
    "break",
    "run8_given_break",
    "coherence_given_run8",
    "anchor_given_coherence",
)
PREDICTION_TARGETS = ("coherence_given_run8", "strict8")
MODEL_VARIANTS = ("HCS", "HCSR", "HCSX", "HCSRX")
PREDICTION_CONTRASTS = {
    "rulebook_beyond_hcs": ("HCS", "HCSR"),
    "cross_candidate_hold_beyond_hcs": ("HCS", "HCSX"),
    "rulebook_beyond_empirical_hold": ("HCSX", "HCSRX"),
    "empirical_hold_beyond_rulebook": ("HCSR", "HCSRX"),
}

RULEBOOK_STARTS = 16
RULEBOOK_MAXIMUM_ITERATIONS = 1_000
RULEBOOK_TOLERANCE = 1e-11
RULEBOOK_DAMPING = 0.5
RULEBOOK_MERGE_COSINE = 0.95

INTERVENTION_BRANCHES = 64
INTERVENTION_HORIZON = 32
INTERVENTION_PRIMARY_SPEC = "cosine_registered"
INTERVENTION_PRIMARY_ENDPOINTS = (
    "break_by8_away_minus_toward",
    "hold8_toward_minus_away",
    "coherent8_toward_minus_away",
)
FULL_DOSE_VALIDITY = 0.90
MIN_SUCCESSES = 100
MIN_FAILURES = 100
MIN_OUTCOME_MATRICES = 20
BOOTSTRAP_REPETITIONS = 4_096
RANDOMIZATION_REPETITIONS = 4_096
CHECKPOINT_FORMAT = "strict8-rulebook-intervention-checkpoint-v1"
RESULT_FORMAT = "strict8-rulebook-holding-probe-v1"

RULEBOOK_SEED = hashlib.sha256(b"strict8-rulebook-v1::forms").hexdigest()
HOLDING_SEED = hashlib.sha256(b"strict8-rulebook-v1::holding").hexdigest()
INTERVENTION_SELECTION_SEED = hashlib.sha256(
    b"strict8-rulebook-v1::edit-selection"
).hexdigest()
INTERVENTION_FUTURE_SEED = hashlib.sha256(
    b"strict8-rulebook-v1::fresh-futures"
).hexdigest()
BOOTSTRAP_SEED = hashlib.sha256(b"strict8-rulebook-v1::bootstrap").hexdigest()
RANDOMIZATION_SEED = hashlib.sha256(
    b"strict8-rulebook-v1::randomization"
).hexdigest()


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
        json.dumps(_json_ready(value), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
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
        "rulebook_core": TASK_ROOT / "rulebook_core.py",
        "analysis_runner": TASK_ROOT / "run_analysis.py",
        "unit_tests": TASK_ROOT / "test_rulebook_core.py",
        "geometry_runner": GEOMETRY_ROOT / "run_analysis.py",
        "geometry_core": GEOMETRY_ROOT / "strict_core.py",
        "geometry_calibration": GEOMETRY_ROOT / "artifacts" / "calibration" / "SHA256SUMS",
        "geometry_relation_calibration": (
            GEOMETRY_ROOT
            / "artifacts"
            / "calibration"
            / "relation_specific_calibration.json"
        ),
        "global_metric_calibration": (
            PAPER_ROOT
            / "reviewer_threshold_metric_sensitivity_extension"
            / "artifacts"
            / "calibration"
            / "metric_calibration.json"
        ),
        "cohort_reconstruction": (
            SOURCE_ROOT
            / "reviewer_threshold_sensitivity_response"
            / "run_sensitivity.py"
        ),
        "simulator": SOURCE_ROOT / "plastic_heredity" / "simulator.py",
        "config": SOURCE_ROOT / "plastic_heredity" / "config.py",
        "experiment": SOURCE_ROOT / "plastic_heredity" / "experiment.py",
        "seeds": SOURCE_ROOT / "plastic_heredity" / "seeds.py",
        "regime_confirmation": SOURCE_ROOT / "plastic_heredity" / "regime_confirmation.py",
        "binomial_models": (
            SOURCE_ROOT / "plastic_heredity" / "mechanistic_v2_models.py"
        ),
        "inference_helpers": (
            SOURCE_ROOT / "plastic_heredity" / "mechanistic_metrics.py"
        ),
        "development_inputs": DEVELOPMENT_SOURCE / "SHA256SUMS",
        "confirmation_inputs": CONFIRMATION_SOURCE / "SHA256SUMS",
        "development_arrays": DEVELOPMENT_SOURCE / "development_arrays.npz",
        "confirmation_arrays": CONFIRMATION_SOURCE / "confirmation_arrays.npz",
        "geometry_replays": GEOMETRY_REPLAY_ROOT / "SHA256SUMS",
        "development_geometry_replay": GEOMETRY_REPLAY_ROOT / "development.npz",
        "confirmation_geometry_replay": GEOMETRY_REPLAY_ROOT / "confirmation.npz",
        "concentration_features": CONCENTRATION_ROOT / "SHA256SUMS",
        "development_concentration": (
            CONCENTRATION_ROOT / "development_concentration.npz"
        ),
        "confirmation_concentration": (
            CONCENTRATION_ROOT / "confirmation_concentration.npz"
        ),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing source-contract files: {missing}")
    return {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for name, path in paths.items()
    }


def _protocol() -> dict[str, Any]:
    value: dict[str, Any] = {
        "format": "strict8-rulebook-holding-protocol-v1",
        "date": "2026-08-19",
        "status": "post_hoc_hypothesis_prospectively_frozen_before_new_futures",
        "scope": {
            "writes_below": str(TASK_ROOT.resolve()),
            "manuscript_edits": False,
            "newideas_inputs_or_outputs_used": False,
            "starting_states": "2,000 retained REGCONF surviving/observable states",
        },
        "rulebook": {
            "definition": (
                "fixed points of the normalized expected join-minus-leave flow "
                "derived directly from the frozen simulator equations"
            ),
            "starts": RULEBOOK_STARTS,
            "maximum_iterations": RULEBOOK_MAXIMUM_ITERATIONS,
            "tolerance": RULEBOOK_TOLERANCE,
            "damping": RULEBOOK_DAMPING,
            "merge_cosine": RULEBOOK_MERGE_COSINE,
            "features": list(RULEBOOK_FEATURE_NAMES),
            "outcome_blind": True,
        },
        "cross_fitted_holding": {
            "donor": "opposite candidate, same matrix, all five landmarks",
            "donor_branches": "opposite 64-branch half",
            "target": "current candidate/state and nonoverlapping branch half",
            "development_fit_confirmation_score": True,
            "prediction_targets": list(PREDICTION_TARGETS),
            "variants": list(MODEL_VARIANTS),
            "contrasts": PREDICTION_CONTRASTS,
            "primary_metric": INTERVENTION_PRIMARY_SPEC,
            "other_metrics": "equal-status sensitivity readouts",
            "not_launch_time_predictor": True,
        },
        "fresh_intervention": {
            "arms": list(EDIT_ARMS),
            "branches_per_state_arm": INTERVENTION_BRANCHES,
            "horizon": INTERVENTION_HORIZON,
            "fresh_futures": 2000 * len(EDIT_ARMS) * INTERVENTION_BRANCHES,
            "common_random_streams": True,
            "future_seed_excludes_arm": True,
            "mass_preserved": True,
            "occupied_set_preserved": True,
            "doses_nested": True,
            "target": "nearest beta-derived rulebook form",
            "primary_spec": INTERVENTION_PRIMARY_SPEC,
            "primary_endpoints": list(INTERVENTION_PRIMARY_ENDPOINTS),
            "strict8_net_effect": "secondary direction-agnostic readout",
            "full_dose_validity": FULL_DOSE_VALIDITY,
        },
        "inference": {
            "unit": "catalytic matrix",
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "randomization_repetitions": RANDOMIZATION_REPETITIONS,
            "primary_holm_family": "candidate by independent branch half within endpoint",
            "minimum_successes": MIN_SUCCESSES,
            "minimum_failures": MIN_FAILURES,
            "minimum_outcome_matrices": MIN_OUTCOME_MATRICES,
        },
        "seeds": {
            "rulebook": RULEBOOK_SEED,
            "holding": HOLDING_SEED,
            "intervention_selection": INTERVENTION_SELECTION_SEED,
            "intervention_future": INTERVENTION_FUTURE_SEED,
            "bootstrap": BOOTSTRAP_SEED,
            "randomization": RANDOMIZATION_SEED,
        },
        "claim_boundary": (
            "Post-hoc mechanistic follow-up. Causal interpretation is limited to "
            "the frozen alignment edits on reused observable selected-lineage states."
        ),
        "source_contract": _source_contract(),
    }
    value["protocol_id"] = canonical_digest(value)
    return _json_ready(value)


def prepare() -> None:
    value = _protocol()
    path = PROTOCOL_ROOT / "analysis_protocol.json"
    if path.exists():
        verify_checksums(PROTOCOL_ROOT)
        if json.loads(path.read_text()) != value:
            raise ValueError("existing frozen protocol differs from current source contract")
        print(f"Protocol already frozen and identical: {path}", flush=True)
        return
    _write_json(path, value)
    _replace_checksums(PROTOCOL_ROOT)
    print(f"Frozen post-hoc rulebook protocol: {path}", flush=True)


def verify_protocol() -> dict[str, Any]:
    verify_checksums(PROTOCOL_ROOT)
    saved = json.loads((PROTOCOL_ROOT / "analysis_protocol.json").read_text())
    if saved != _protocol():
        raise ValueError("protocol, source, or input identity changed")
    return saved


_PROTOCOL_ID_CACHE: str | None = None


def _protocol_id() -> str:
    global _PROTOCOL_ID_CACHE
    if _PROTOCOL_ID_CACHE is None:
        _PROTOCOL_ID_CACHE = str(verify_protocol()["protocol_id"])
    return _PROTOCOL_ID_CACHE


def _cohort_paths(cohort: str) -> tuple[Path, Path, str, str]:
    if cohort == "development":
        return (
            DEVELOPMENT_SOURCE / "development_arrays.npz",
            GEOMETRY_REPLAY_ROOT / "development.npz",
            DEVELOPMENT_MASTER_SEED,
            "REGDEV",
        )
    if cohort == "confirmation":
        return (
            CONFIRMATION_SOURCE / "confirmation_arrays.npz",
            GEOMETRY_REPLAY_ROOT / "confirmation.npz",
            CONFIRMATION_MASTER_SEED,
            "REGCONF",
        )
    raise ValueError(cohort)


def _source_arrays(cohort: str) -> dict[str, np.ndarray]:
    source, _, _, _ = _cohort_paths(cohort)
    return _load_npz(source)


def _replay_arrays(cohort: str) -> dict[str, np.ndarray]:
    _, replay, _, _ = _cohort_paths(cohort)
    values = _load_npz(replay)
    if tuple(values["spec_names"].astype(str)) != SPEC_NAMES:
        raise ValueError("strict endpoint order differs")
    return values


def _beta_for_matrix(cohort: str, matrix_id: int) -> tuple[Any, np.ndarray]:
    _, _, master_seed, label = _cohort_paths(cohort)
    experiment = regime_experiment(master_seed)
    rng = np.random.default_rng(
        derive_seed(experiment.master_seed, f"{label}.beta", matrix_id)
    )
    return experiment, generate_beta(experiment.gard, rng)


def build_rulebooks() -> None:
    verify_protocol()
    RULEBOOK_ROOT.mkdir(parents=True, exist_ok=True)
    for cohort in ("development", "confirmation"):
        output = RULEBOOK_ROOT / f"{cohort}.npz"
        if output.is_file():
            continue
        source = _source_arrays(cohort)
        _, _, master_seed, _ = _cohort_paths(cohort)
        cohort_experiment = regime_experiment(master_seed)
        matrix_ids = np.asarray(source["matrix_ids"], dtype=int)
        matrices = np.unique(matrix_ids)
        solutions = []
        betas: dict[int, np.ndarray] = {}
        margins: dict[int, np.ndarray] = {}
        for matrix_id in matrices:
            experiment, beta = _beta_for_matrix(cohort, int(matrix_id))
            solution = solve_rulebook(
                beta,
                experiment.gard.k_join,
                experiment.gard.k_leave,
                RULEBOOK_STARTS,
                derive_seed(RULEBOOK_SEED, cohort, int(matrix_id)),
                RULEBOOK_MAXIMUM_ITERATIONS,
                RULEBOOK_TOLERANCE,
                RULEBOOK_DAMPING,
                RULEBOOK_MERGE_COSINE,
            )
            betas[int(matrix_id)] = beta
            solutions.append(solution)
            margins[int(matrix_id)] = np.asarray(
                [
                    tangent_stability_margin(
                        form,
                        beta,
                        experiment.gard.k_join,
                        experiment.gard.k_leave,
                    )
                    for form in solution.forms
                ]
            )
        maximum_forms = max(len(solution.forms) for solution in solutions)
        n_types = int(np.asarray(source["compositions"]).shape[1])
        forms = np.full((len(matrices), maximum_forms, n_types), np.nan)
        stability = np.full((len(matrices), maximum_forms), np.nan)
        form_counts = np.empty(len(matrices), dtype=np.int16)
        iterations = np.empty(len(matrices), dtype=np.int16)
        maximum_updates = np.empty(len(matrices), dtype=np.float64)
        residuals = np.empty(len(matrices), dtype=np.float64)
        lookup: dict[int, int] = {}
        for position, (matrix_id, solution) in enumerate(zip(matrices, solutions, strict=True)):
            lookup[int(matrix_id)] = position
            count = len(solution.forms)
            form_counts[position] = count
            forms[position, :count] = solution.forms
            stability[position, :count] = margins[int(matrix_id)]
            iterations[position] = solution.iterations
            maximum_updates[position] = solution.maximum_update
            residuals[position] = solution.maximum_flow_residual

        compositions = np.asarray(source["compositions"], dtype=np.int64)
        features = np.empty((len(compositions), len(RULEBOOK_FEATURE_NAMES)))
        targets = np.empty_like(compositions, dtype=np.float64)
        nearest_indices = np.empty(len(compositions), dtype=np.int16)
        for index, (composition, matrix_id) in enumerate(zip(compositions, matrix_ids, strict=True)):
            position = lookup[int(matrix_id)]
            available = forms[position, : form_counts[position]]
            nearest, target, _ = nearest_form(composition, available)
            targets[index] = target
            nearest_indices[index] = nearest
            features[index] = rulebook_features(
                composition,
                betas[int(matrix_id)],
                available,
                cohort_experiment.gard.k_join,
                cohort_experiment.gard.k_leave,
                stability_margin=stability[position, nearest],
            )
        _atomic_npz(
            output,
            protocol_id=np.asarray(_protocol_id()),
            state_ids=np.asarray(source["state_ids"]),
            matrix_ids=matrices.astype(np.int16),
            form_counts=form_counts,
            forms=forms,
            stability_margins=stability,
            solver_iterations=iterations,
            solver_maximum_updates=maximum_updates,
            solver_flow_residuals=residuals,
            feature_names=np.asarray(RULEBOOK_FEATURE_NAMES),
            state_features=features,
            state_target_forms=targets,
            state_nearest_form_indices=nearest_indices,
        )
        print(
            f"{cohort} rulebooks: {len(matrices)} matrices, "
            f"{int(form_counts.sum())} forms",
            flush=True,
        )
    _replace_checksums(RULEBOOK_ROOT)


HOLDING_FEATURE_NAMES = tuple(
    f"xcan__{spec}__{transition}"
    for spec in SPEC_NAMES
    for transition in TRANSITION_NAMES
) + tuple(f"xcan__{spec}__strict8" for spec in SPEC_NAMES)
SIBLING_FEATURE_NAMES = tuple(name.replace("xcan__", "sibling__") for name in HOLDING_FEATURE_NAMES)


def _holding_features_for_half(
    replay: Mapping[str, np.ndarray], target_half: str, donor_mode: str
) -> np.ndarray:
    candidates = np.asarray(replay["candidates"]).astype(str)
    matrix_ids = np.asarray(replay["matrix_ids"], dtype=int)
    landmarks = np.asarray(replay["landmarks"], dtype=int)
    gates = np.asarray(replay["deepest_gate"], dtype=np.int8)
    labels = np.asarray(replay["labels"], dtype=np.int8)
    donor_slice = slice(64, 128) if target_half == "A" else slice(0, 64)
    per_state_success = np.empty((len(candidates), len(SPEC_NAMES), 4), dtype=np.int64)
    per_state_trials = np.empty_like(per_state_success)
    per_state_strict = np.empty((len(candidates), len(SPEC_NAMES)), dtype=np.int64)
    for spec_index in range(len(SPEC_NAMES)):
        success, trials = aggregate_transitions(gates[:, donor_slice, spec_index])
        per_state_success[:, spec_index] = success
        per_state_trials[:, spec_index] = trials
        per_state_strict[:, spec_index] = labels[:, donor_slice, spec_index].sum(axis=1)
    grouped = {
        (int(matrix_id), candidate): np.flatnonzero(
            (matrix_ids == matrix_id) & (candidates == candidate)
        )
        for matrix_id in np.unique(matrix_ids)
        for candidate in ("02", "03")
    }
    values = np.empty((len(candidates), len(HOLDING_FEATURE_NAMES)), dtype=np.float64)
    for index, (candidate, matrix_id, landmark) in enumerate(
        zip(candidates, matrix_ids, landmarks, strict=True)
    ):
        if donor_mode == "cross_candidate":
            other = "03" if candidate == "02" else "02"
            donors = grouped[(int(matrix_id), other)]
        elif donor_mode == "sibling":
            donors = grouped[(int(matrix_id), candidate)]
            donors = donors[landmarks[donors] != landmark]
        else:
            raise ValueError(donor_mode)
        cursor = 0
        for spec_index in range(len(SPEC_NAMES)):
            success = per_state_success[donors, spec_index].sum(axis=0)
            trials = per_state_trials[donors, spec_index].sum(axis=0)
            values[index, cursor : cursor + 4] = smoothed_rate(success, trials)
            cursor += 4
        for spec_index in range(len(SPEC_NAMES)):
            success = int(per_state_strict[donors, spec_index].sum())
            trials = len(donors) * 64
            values[index, cursor] = float(smoothed_rate(success, trials))
            cursor += 1
    return values


def build_holding_features() -> None:
    verify_protocol()
    HOLDING_ROOT.mkdir(parents=True, exist_ok=True)
    for cohort in ("development", "confirmation"):
        replay = _replay_arrays(cohort)
        source = _source_arrays(cohort)
        if not np.array_equal(source["state_ids"], replay["state_ids"]):
            raise AssertionError("source/replay state order differs")
        for half in ("A", "B"):
            output = HOLDING_ROOT / f"{cohort}_{half}.npz"
            if output.is_file():
                continue
            cross = _holding_features_for_half(replay, half, "cross_candidate")
            sibling = _holding_features_for_half(replay, half, "sibling")
            _atomic_npz(
                output,
                protocol_id=np.asarray(_protocol_id()),
                state_ids=np.asarray(source["state_ids"]),
                target_half=np.asarray(half),
                donor_half=np.asarray("B" if half == "A" else "A"),
                cross_candidate_names=np.asarray(HOLDING_FEATURE_NAMES),
                sibling_names=np.asarray(SIBLING_FEATURE_NAMES),
                cross_candidate=cross,
                sibling=sibling,
            )
    _replace_checksums(HOLDING_ROOT)
    print("Leakage-safe cross-candidate and sibling holding features written", flush=True)


def _concentration_arrays(cohort: str) -> np.ndarray:
    values = _load_npz(CONCENTRATION_ROOT / f"{cohort}_concentration.npz")
    if str(values["protocol_id"].item()) == "":
        raise ValueError("missing concentration protocol identity")
    source = _source_arrays(cohort)
    if not np.array_equal(source["state_ids"], values["state_ids"]):
        raise AssertionError("concentration/source state order differs")
    return np.asarray(values["concentration"], dtype=np.float64)


def _designs(cohort: str, half: str) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    source = _source_arrays(cohort)
    rulebook = _load_npz(RULEBOOK_ROOT / f"{cohort}.npz")
    holding = _load_npz(HOLDING_ROOT / f"{cohort}_{half}.npz")
    for values in (rulebook, holding):
        if not np.array_equal(source["state_ids"], values["state_ids"]):
            raise AssertionError("derived feature order differs from source")
    hcs = np.column_stack(
        (
            np.asarray(source["h10"], dtype=np.float64),
            _concentration_arrays(cohort),
            np.asarray(source["state_block"], dtype=np.float64),
        )
    )
    rule = np.asarray(rulebook["state_features"], dtype=np.float64)
    cross = np.asarray(holding["cross_candidate"], dtype=np.float64)
    designs = {
        "HCS": hcs,
        "HCSR": np.column_stack((hcs, rule)),
        "HCSX": np.column_stack((hcs, cross)),
        "HCSRX": np.column_stack((hcs, rule, cross)),
    }
    hcs_names = tuple(f"hcs_{index:03d}" for index in range(hcs.shape[1]))
    names = {
        "HCS": hcs_names,
        "HCSR": hcs_names + RULEBOOK_FEATURE_NAMES,
        "HCSX": hcs_names + HOLDING_FEATURE_NAMES,
        "HCSRX": hcs_names + RULEBOOK_FEATURE_NAMES + HOLDING_FEATURE_NAMES,
    }
    return designs, {"source": source, "names": names}


def _target_counts(
    replay: Mapping[str, np.ndarray], half: str, spec_index: int, target: str
) -> tuple[np.ndarray, np.ndarray]:
    branch_slice = slice(0, 64) if half == "A" else slice(64, 128)
    if target == "strict8":
        successes = np.asarray(replay["labels"][:, branch_slice, spec_index]).sum(axis=1)
        return successes.astype(np.int64), np.full(len(successes), 64, dtype=np.int64)
    transition = TRANSITION_NAMES.index(target)
    success, trials = aggregate_transitions(
        np.asarray(replay["deepest_gate"][:, branch_slice, spec_index], dtype=np.int8)
    )
    return success[:, transition], trials[:, transition]


def _binomial_loss(
    successes: np.ndarray, trials: np.ndarray, probability: np.ndarray
) -> float:
    p = np.clip(np.asarray(probability, dtype=float), 1e-12, 1 - 1e-12)
    return float(
        -np.sum(successes * np.log(p) + (trials - successes) * np.log(1 - p))
        / np.sum(trials)
    )


def _state_binomial_loss(
    successes: np.ndarray, trials: np.ndarray, probability: np.ndarray
) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=float), 1e-12, 1 - 1e-12)
    numerator = -successes * np.log(p) - (trials - successes) * np.log(1 - p)
    return np.divide(
        numerator,
        trials,
        out=np.full(len(trials), np.nan, dtype=np.float64),
        where=np.asarray(trials) > 0,
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
    x = np.asarray(design, dtype=np.float64)[keep]
    y = np.asarray(successes, dtype=np.float64)[keep]
    n = np.asarray(trials, dtype=np.float64)[keep]
    mids = np.asarray(matrix_ids, dtype=np.int64)[keep]
    scores: dict[str, float] = {}
    for ridge in RIDGE_LAMBDAS:
        numerator = 0.0
        denominator = 0.0
        for fold in range(5):
            validation = matrix_cv_fold(mids) == fold
            train = ~validation
            transform = fit_block_transform(label, x[train], names)
            fitted = fit_linear(
                label,
                label,
                transform.transform(x[train]),
                y[train],
                n[train],
                ridge,
            )
            probability = expit(fitted.correction(transform.transform(x[validation])))
            weight = float(n[validation].sum())
            numerator += _binomial_loss(
                y[validation], n[validation], probability
            ) * weight
            denominator += weight
        scores[f"{ridge:g}"] = numerator / denominator
    minimum = min(scores.values())
    selected = max(
        ridge for ridge in RIDGE_LAMBDAS if scores[f"{ridge:g}"] <= minimum + 1e-12
    )
    transform = fit_block_transform(label, x, names)
    fitted = fit_linear(
        label, label, transform.transform(x), y, n, selected
    )
    return {
        "label": label,
        "transform": transform,
        "fit": fitted,
        "selected_lambda": selected,
        "cv_scores": scores,
        "training_rows": int(keep.sum()),
        "training_successes": int(y.sum()),
        "training_trials": int(n.sum()),
    }


def _predict_binomial(model: Mapping[str, Any], design: np.ndarray) -> np.ndarray:
    return expit(model["fit"].correction(model["transform"].transform(design)))


def fit_models() -> None:
    verify_protocol()
    verify_checksums(RULEBOOK_ROOT)
    verify_checksums(HOLDING_ROOT)
    replay = _replay_arrays("development")
    models: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    for half in ("A", "B"):
        designs, values = _designs("development", half)
        source = values["source"]
        names = values["names"]
        candidates = np.asarray(source["candidates"]).astype(str)
        matrix_ids = np.asarray(source["matrix_ids"], dtype=int)
        for spec_index, spec in enumerate(SPEC_NAMES):
            for candidate in ("02", "03"):
                selected = candidates == candidate
                for target in PREDICTION_TARGETS:
                    success, trials = _target_counts(replay, half, spec_index, target)
                    for variant in MODEL_VARIANTS:
                        key = f"{spec}|c{candidate}|{half}|{target}|{variant}"
                        model = _fit_binomial_model(
                            designs[variant][selected],
                            names[variant],
                            success[selected],
                            trials[selected],
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
                            "feature_count": len(model["transform"].output_names),
                        }
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    with (MODEL_ROOT / "models.pkl").open("wb") as handle:
        pickle.dump(models, handle, protocol=5)
    _write_json(
        MODEL_ROOT / "model_seal.json",
        {
            "format": "strict8-rulebook-model-seal-v1",
            "protocol_id": _protocol_id(),
            "development_only": True,
            "confirmation_outcomes_loaded": False,
            "models": metadata,
        },
    )
    _replace_checksums(MODEL_ROOT)
    print(f"Sealed {len(models)} development-only rulebook models", flush=True)


def _group_values(values: np.ndarray, matrix_ids: np.ndarray) -> np.ndarray:
    return np.asarray(
        [np.nanmean(values[matrix_ids == key]) for key in np.unique(matrix_ids)],
        dtype=np.float64,
    )


def _effect_inference(
    values: np.ndarray, matrix_ids: np.ndarray, seed_parts: Sequence[str]
) -> tuple[float, float, float, float, float]:
    finite = np.isfinite(values)
    groups = _group_values(np.asarray(values)[finite], np.asarray(matrix_ids)[finite])
    observed = float(groups.mean())
    rng = np.random.default_rng(derive_seed(BOOTSTRAP_SEED, *seed_parts))
    indices = rng.integers(0, len(groups), size=(BOOTSTRAP_REPETITIONS, len(groups)))
    bootstrap = groups[indices].mean(axis=1)
    lower, upper = np.quantile(bootstrap, (0.025, 0.975))
    rng = np.random.default_rng(derive_seed(RANDOMIZATION_SEED, *seed_parts))
    signs = rng.integers(0, 2, size=(RANDOMIZATION_REPETITIONS, len(groups))) * 2 - 1
    null = signs @ groups / len(groups)
    greater = float((np.count_nonzero(null >= observed) + 1) / (len(null) + 1))
    two_sided = float(
        (np.count_nonzero(np.abs(null) >= abs(observed)) + 1) / (len(null) + 1)
    )
    return observed, float(lower), float(upper), greater, two_sided


def _power_row(
    success: np.ndarray, trials: np.ndarray, matrix_ids: np.ndarray
) -> dict[str, Any]:
    successes = int(success.sum())
    failures = int((trials - success).sum())
    success_matrices = int(
        sum(success[matrix_ids == key].sum() > 0 for key in np.unique(matrix_ids))
    )
    failure_matrices = int(
        sum(
            (trials[matrix_ids == key] - success[matrix_ids == key]).sum() > 0
            for key in np.unique(matrix_ids)
        )
    )
    return {
        "successes": successes,
        "failures": failures,
        "success_matrices": success_matrices,
        "failure_matrices": failure_matrices,
        "power_adequate": successes >= MIN_SUCCESSES
        and failures >= MIN_FAILURES
        and success_matrices >= MIN_OUTCOME_MATRICES
        and failure_matrices >= MIN_OUTCOME_MATRICES,
    }


def _spearman_bootstrap(
    predictor: np.ndarray,
    outcome: np.ndarray,
    seed_parts: Sequence[str],
) -> tuple[float, float, float]:
    finite = np.isfinite(predictor) & np.isfinite(outcome)
    x = np.asarray(predictor)[finite]
    y = np.asarray(outcome)[finite]
    observed = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(derive_seed(BOOTSTRAP_SEED, *seed_parts))
    indices = rng.integers(0, len(x), size=(BOOTSTRAP_REPETITIONS, len(x)))
    samples = np.empty(BOOTSTRAP_REPETITIONS, dtype=np.float64)
    for index, selected in enumerate(indices):
        samples[index] = float(spearmanr(x[selected], y[selected]).statistic)
    lower, upper = np.nanquantile(samples, (0.025, 0.975))
    return observed, float(lower), float(upper)


def score_models() -> None:
    verify_protocol()
    verify_checksums(MODEL_ROOT)
    replay = _replay_arrays("confirmation")
    with (MODEL_ROOT / "models.pkl").open("rb") as handle:
        models = pickle.load(handle)
    prediction_rows: list[dict[str, Any]] = []
    correlation_rows: list[dict[str, Any]] = []
    for half in ("A", "B"):
        designs, values = _designs("confirmation", half)
        source = values["source"]
        candidates = np.asarray(source["candidates"]).astype(str)
        matrix_ids = np.asarray(source["matrix_ids"], dtype=int)
        holding = _load_npz(HOLDING_ROOT / f"confirmation_{half}.npz")[
            "cross_candidate"
        ]
        for spec_index, spec in enumerate(SPEC_NAMES):
            for candidate in ("02", "03"):
                selected = candidates == candidate
                mids = matrix_ids[selected]
                predictions: dict[tuple[str, str], np.ndarray] = {}
                for target in PREDICTION_TARGETS:
                    success, trials = _target_counts(replay, half, spec_index, target)
                    power = _power_row(success[selected], trials[selected], mids)
                    for variant in MODEL_VARIANTS:
                        key = f"{spec}|c{candidate}|{half}|{target}|{variant}"
                        predictions[(target, variant)] = _predict_binomial(
                            models[key], designs[variant][selected]
                        )
                    for contrast, (left, right) in PREDICTION_CONTRASTS.items():
                        left_loss = _state_binomial_loss(
                            success[selected], trials[selected], predictions[(target, left)]
                        )
                        right_loss = _state_binomial_loss(
                            success[selected], trials[selected], predictions[(target, right)]
                        )
                        effect = _effect_inference(
                            left_loss - right_loss,
                            mids,
                            ("prediction", contrast, spec, candidate, half, target),
                        )
                        prediction_rows.append(
                            {
                                "contrast": contrast,
                                "spec": spec,
                                "candidate": candidate,
                                "half": half,
                                "target": target,
                                "states": int(selected.sum()),
                                "matrices": int(np.unique(mids).size),
                                "log_loss_gain": effect[0],
                                "ci95_lower": effect[1],
                                "ci95_upper": effect[2],
                                "randomization_p_raw": effect[3],
                                **power,
                            }
                        )

                branch_slice = slice(0, 64) if half == "A" else slice(64, 128)
                success_all, trials_all = aggregate_transitions(
                    np.asarray(replay["deepest_gate"][:, branch_slice, spec_index])
                )
                target_rates = smoothed_rate(success_all, trials_all)
                strict_rate = smoothed_rate(
                    np.asarray(replay["labels"][:, branch_slice, spec_index]).sum(axis=1),
                    np.full(len(candidates), 64),
                )
                for target_index, target in enumerate(TRANSITION_NAMES + ("strict8",)):
                    column = spec_index * 4 + target_index if target_index < 4 else 12 + spec_index
                    matrix_predictor = []
                    matrix_outcome = []
                    for matrix_id in np.unique(mids):
                        rows = selected & (matrix_ids == matrix_id)
                        matrix_predictor.append(float(np.mean(holding[rows, column])))
                        if target_index < 4:
                            matrix_outcome.append(float(np.mean(target_rates[rows, target_index])))
                        else:
                            matrix_outcome.append(float(np.mean(strict_rate[rows])))
                    correlation = _spearman_bootstrap(
                        np.asarray(matrix_predictor),
                        np.asarray(matrix_outcome),
                        ("holding-correlation", spec, candidate, half, target),
                    )
                    correlation_rows.append(
                        {
                            "spec": spec,
                            "candidate": candidate,
                            "half": half,
                            "target": target,
                            "matrices": len(matrix_predictor),
                            "spearman": correlation[0],
                            "ci95_lower": correlation[1],
                            "ci95_upper": correlation[2],
                        }
                    )

    predictions = pd.DataFrame(prediction_rows)
    predictions["randomization_p_holm"] = np.nan
    for contrast in PREDICTION_CONTRASTS:
        for spec in SPEC_NAMES:
            for target in PREDICTION_TARGETS:
                chosen = (
                    (predictions["contrast"] == contrast)
                    & (predictions["spec"] == spec)
                    & (predictions["target"] == target)
                )
                predictions.loc[chosen, "randomization_p_holm"] = holm_adjust(
                    predictions.loc[chosen, "randomization_p_raw"].tolist()
                )
    predictions["passes_gate"] = (
        predictions["power_adequate"]
        & (predictions["log_loss_gain"] > 0)
        & (predictions["ci95_lower"] > 0)
        & (predictions["randomization_p_holm"] < 0.05)
    )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUTPUT_ROOT / "prediction_effects.csv", index=False)
    pd.DataFrame(correlation_rows).to_csv(
        OUTPUT_ROOT / "cross_candidate_holding_correlations.csv", index=False
    )
    print("Rulebook and cross-candidate holding predictors scored", flush=True)


def _checkpoint_path(index: int) -> Path:
    return WORK_ROOT / "intervention" / f"state_{index:04d}.npz"


def _checkpoint_complete(index: int) -> bool:
    path = _checkpoint_path(index)
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as archive:
            return (
                str(archive["format"].item()) == CHECKPOINT_FORMAT
                and str(archive["protocol_id"].item()) == _protocol_id()
                and int(archive["state_index"].item()) == index
            )
    except Exception:
        return False


def _edit_seed(case: Any, arm: str) -> int:
    stem = arm.rsplit("_D", 1)[0] if "_D" in arm else arm
    return derive_seed(INTERVENTION_SELECTION_SEED, case.state_id, stem)


def _future_seed(case: Any, branch: int) -> int:
    return derive_seed(
        INTERVENTION_FUTURE_SEED,
        "REGCONF.rulebook.future",
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


def _first8_outcomes(records: Sequence, specs: Sequence) -> tuple[np.ndarray, ...]:
    break_by8 = np.zeros(len(specs), dtype=np.int8)
    hold8 = np.zeros(len(specs), dtype=np.int8)
    coherent8 = np.zeros(len(specs), dtype=np.int8)
    complete8 = int(len(records) >= RUN_LENGTH)
    for index, spec in enumerate(specs):
        geometry = build_geometry(records, spec.metric)
        observed = geometry.boundary[:RUN_LENGTH]
        break_by8[index] = int(np.any(observed <= spec.inheritance_cutoff))
        if complete8:
            hold = bool(np.all(observed > spec.inheritance_cutoff))
            hold8[index] = int(hold)
            coherent8[index] = int(
                hold
                and window_pairwise_minimum(geometry, 0) > spec.coherence_cutoff
            )
    return break_by8, hold8, coherent8, np.full(len(specs), complete8, dtype=np.int8)


def _intervention_worker(arguments: tuple[int, Any, np.ndarray, tuple, str]) -> int:
    state_index, case, target_form, specs, protocol_id = arguments
    output = _checkpoint_path(state_index)
    if output.is_file():
        try:
            with np.load(output, allow_pickle=False) as archive:
                if (
                    str(archive["format"].item()) == CHECKPOINT_FORMAT
                    and str(archive["protocol_id"].item()) == protocol_id
                    and int(archive["state_index"].item()) == state_index
                ):
                    return state_index
        except Exception:
            pass

    n_arms = len(EDIT_ARMS)
    n_specs = len(specs)
    shape = (n_arms, INTERVENTION_BRANCHES, n_specs)
    strict_labels = np.zeros(shape, dtype=np.int8)
    strict_gates = np.zeros(shape, dtype=np.int8)
    break_by8 = np.zeros(shape, dtype=np.int8)
    hold8 = np.zeros(shape, dtype=np.int8)
    coherent8 = np.zeros(shape, dtype=np.int8)
    complete8 = np.zeros(shape, dtype=np.int8)
    completed32 = np.zeros((n_arms, INTERVENTION_BRANCHES), dtype=np.int8)
    observed = np.zeros((n_arms, INTERVENTION_BRANCHES), dtype=np.int16)
    book_cosine = np.full((n_arms, INTERVENTION_BRANCHES, 4), np.nan)
    requested = np.zeros(n_arms, dtype=np.int8)
    achieved = np.zeros(n_arms, dtype=np.int8)
    occupied_before = np.zeros(n_arms, dtype=np.int16)
    occupied_after = np.zeros(n_arms, dtype=np.int16)
    cosine_before = np.zeros(n_arms, dtype=np.float64)
    cosine_after = np.zeros(n_arms, dtype=np.float64)
    edited = np.zeros((n_arms, len(target_form)), dtype=np.int16)

    gard = regime_experiment(CONFIRMATION_MASTER_SEED).gard
    with threadpool_limits(limits=1):
        for arm_index, arm in enumerate(EDIT_ARMS):
            edit = apply_rulebook_edit(
                case.snapshot.composition,
                target_form,
                arm,
                _edit_seed(case, arm),
            )
            snapshot = _edited_snapshot(case.snapshot, edit.composition)
            requested[arm_index] = edit.requested_dose
            achieved[arm_index] = edit.achieved_dose
            occupied_before[arm_index] = edit.occupied_before
            occupied_after[arm_index] = edit.occupied_after
            cosine_before[arm_index] = edit.cosine_before
            cosine_after[arm_index] = edit.cosine_after
            edited[arm_index] = edit.composition
            for branch in range(INTERVENTION_BRANCHES):
                records, complete = simulate_future_absorbing(
                    snapshot,
                    case.beta,
                    gard,
                    CANDIDATES[case.candidate],
                    INTERVENTION_HORIZON,
                    np.random.default_rng(_future_seed(case, branch)),
                )
                outcomes, _ = score_all_specs(records, specs)
                first8 = _first8_outcomes(records, specs)
                completed32[arm_index, branch] = int(complete)
                observed[arm_index, branch] = len(records)
                break_by8[arm_index, branch] = first8[0]
                hold8[arm_index, branch] = first8[1]
                coherent8[arm_index, branch] = first8[2]
                complete8[arm_index, branch] = first8[3]
                for spec_index, outcome in enumerate(outcomes):
                    strict_labels[arm_index, branch, spec_index] = int(outcome.event)
                    strict_gates[arm_index, branch, spec_index] = int(outcome.deepest_gate)
                for time_index, generation in enumerate((1, 4, 8, 32)):
                    if len(records) >= generation:
                        book_cosine[arm_index, branch, time_index] = cosine(
                            records[generation - 1].daughter, target_form
                        )

    _atomic_npz(
        output,
        format=np.asarray(CHECKPOINT_FORMAT),
        protocol_id=np.asarray(protocol_id),
        state_index=np.asarray(state_index, dtype=np.int32),
        state_id=np.asarray(case.state_id),
        target_form=np.asarray(target_form, dtype=np.float64),
        strict_labels=strict_labels,
        strict_gates=strict_gates,
        break_by8=break_by8,
        hold8=hold8,
        coherent8=coherent8,
        complete8=complete8,
        completed32=completed32,
        observed=observed,
        book_cosine=book_cosine,
        requested=requested,
        achieved=achieved,
        occupied_before=occupied_before,
        occupied_after=occupied_after,
        cosine_before=cosine_before,
        cosine_after=cosine_after,
        edited_compositions=edited,
    )
    return state_index


INTERVENTION_KEYS = (
    "strict_labels",
    "strict_gates",
    "break_by8",
    "hold8",
    "coherent8",
    "complete8",
    "completed32",
    "observed",
    "book_cosine",
    "requested",
    "achieved",
    "occupied_before",
    "occupied_after",
    "cosine_before",
    "cosine_after",
    "edited_compositions",
)


def _run_workers(arguments: list[tuple], workers: int) -> None:
    completed = 0
    if workers <= 1:
        for argument in arguments:
            _intervention_worker(argument)
            completed += 1
            if completed % 40 == 0:
                print(f"[intervention] {completed}/{len(arguments)} states", flush=True)
        return
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_intervention_worker, argument) for argument in arguments]
        for future in as_completed(futures):
            future.result()
            completed += 1
            if completed % 40 == 0:
                print(f"[intervention] {completed}/{len(arguments)} states", flush=True)


def run_intervention(workers: int) -> None:
    verify_protocol()
    verify_checksums(RULEBOOK_ROOT)
    experiment, cases = geometry_source._cases("confirmation", workers)
    specs = geometry_source._specs()
    rulebook = _load_npz(RULEBOOK_ROOT / "confirmation.npz")
    source = _source_arrays("confirmation")
    if len(cases) != 2000 or not np.array_equal(
        np.asarray([case.state_id for case in cases]), source["state_ids"]
    ):
        raise AssertionError("confirmation cases differ from frozen state order")
    reconstructed = {
        "candidates": np.asarray([case.candidate for case in cases]),
        "matrix_ids": np.asarray([case.matrix_id for case in cases]),
        "landmarks": np.asarray([case.landmark for case in cases]),
        "compositions": np.vstack([case.snapshot.composition for case in cases]),
    }
    for name, values in reconstructed.items():
        if not np.array_equal(values, np.asarray(source[name])):
            raise AssertionError(f"reconstructed confirmation {name} differ")
    targets = np.asarray(rulebook["state_target_forms"], dtype=np.float64)
    protocol_id = _protocol_id()
    arguments = [
        (index, case, targets[index], specs, protocol_id)
        for index, case in enumerate(cases)
    ]
    _run_workers(arguments, workers)

    arrays: dict[str, list[np.ndarray]] = {key: [] for key in INTERVENTION_KEYS}
    for index in range(len(cases)):
        if not _checkpoint_complete(index):
            raise ValueError(f"missing intervention checkpoint {index}")
        with np.load(_checkpoint_path(index), allow_pickle=False) as archive:
            for key in INTERVENTION_KEYS:
                arrays[key].append(np.asarray(archive[key]))
    stacked = {key: np.stack(parts) for key, parts in arrays.items()}
    INTERVENTION_ROOT.mkdir(parents=True, exist_ok=True)
    _atomic_npz(
        INTERVENTION_ROOT / "intervention_replay.npz",
        protocol_id=np.asarray(protocol_id),
        arm_names=np.asarray(EDIT_ARMS),
        spec_names=np.asarray(SPEC_NAMES),
        state_ids=np.asarray([case.state_id for case in cases]),
        candidates=np.asarray([case.candidate for case in cases]),
        matrix_ids=np.asarray([case.matrix_id for case in cases], dtype=np.int16),
        landmarks=np.asarray([case.landmark for case in cases], dtype=np.int16),
        target_forms=targets,
        **stacked,
    )
    validation_rows = []
    for arm_index, arm in enumerate(EDIT_ARMS):
        requested = stacked["requested"][:, arm_index]
        achieved = stacked["achieved"][:, arm_index]
        validation_rows.append(
            {
                "arm": arm,
                "states": len(cases),
                "requested_dose": int(requested[0]),
                "full_dose_fraction": float(np.mean(achieved == requested)),
                "achieved_dose_mean": float(achieved.mean()),
                "mass_preserved_fraction": float(
                    np.mean(
                        stacked["edited_compositions"][:, arm_index].sum(axis=1)
                        == np.asarray(source["compositions"]).sum(axis=1)
                    )
                ),
                "occupied_preserved_fraction": float(
                    np.mean(
                        stacked["occupied_before"][:, arm_index]
                        == stacked["occupied_after"][:, arm_index]
                    )
                ),
                "target_cosine_change_mean": float(
                    np.mean(
                        stacked["cosine_after"][:, arm_index]
                        - stacked["cosine_before"][:, arm_index]
                    )
                ),
                "complete8_fraction": float(
                    stacked["complete8"][:, arm_index, :, 0].mean()
                ),
                "complete32_fraction": float(
                    stacked["completed32"][:, arm_index].mean()
                ),
            }
        )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(validation_rows).to_csv(
        OUTPUT_ROOT / "intervention_validation.csv", index=False
    )
    _replace_checksums(INTERVENTION_ROOT)
    print(
        f"Fresh rulebook intervention complete: "
        f"{len(cases) * len(EDIT_ARMS) * INTERVENTION_BRANCHES:,} futures",
        flush=True,
    )


def _load_intervention() -> dict[str, np.ndarray]:
    verify_checksums(INTERVENTION_ROOT)
    values = _load_npz(INTERVENTION_ROOT / "intervention_replay.npz")
    if str(values["protocol_id"].item()) != _protocol_id():
        raise ValueError("intervention protocol differs")
    if tuple(values["arm_names"].astype(str)) != EDIT_ARMS:
        raise ValueError("intervention arm order differs")
    return values


def _state_q(values: np.ndarray, arm: int, branch_slice: slice, spec: int) -> np.ndarray:
    return np.asarray(values[:, arm, branch_slice, spec], dtype=np.float64).mean(axis=1)


def _intervention_power(
    left: np.ndarray,
    right: np.ndarray,
    matrix_ids: np.ndarray,
) -> dict[str, Any]:
    left_flat = np.asarray(left, dtype=np.int8).reshape(-1)
    right_flat = np.asarray(right, dtype=np.int8).reshape(-1)
    left_events = int(left_flat.sum())
    right_events = int(right_flat.sum())
    left_matrices = int(
        sum(left[matrix_ids == key].sum() > 0 for key in np.unique(matrix_ids))
    )
    right_matrices = int(
        sum(right[matrix_ids == key].sum() > 0 for key in np.unique(matrix_ids))
    )
    discordant = int(np.count_nonzero(left != right))
    total = left_flat.size
    return {
        "left_events": left_events,
        "right_events": right_events,
        "left_failures": total - left_events,
        "right_failures": total - right_events,
        "left_event_matrices": left_matrices,
        "right_event_matrices": right_matrices,
        "discordant_branch_pairs": discordant,
        "power_adequate": left_events >= MIN_SUCCESSES
        and right_events >= MIN_SUCCESSES
        and total - left_events >= MIN_FAILURES
        and total - right_events >= MIN_FAILURES
        and left_matrices >= MIN_OUTCOME_MATRICES
        and right_matrices >= MIN_OUTCOME_MATRICES
        and discordant >= MIN_SUCCESSES,
    }


def analyze_intervention() -> None:
    data = _load_intervention()
    arm_index = {name: index for index, name in enumerate(EDIT_ARMS)}
    spec_index = {name: index for index, name in enumerate(SPEC_NAMES)}
    candidates = np.asarray(data["candidates"]).astype(str)
    matrix_ids = np.asarray(data["matrix_ids"], dtype=int)
    achieved = np.asarray(data["achieved"], dtype=int)
    primary_definitions = {
        "break_by8_away_minus_toward": (
            "break_by8",
            "AWAY_BOOK_D4",
            "TOWARD_BOOK_D4",
        ),
        "hold8_toward_minus_away": (
            "hold8",
            "TOWARD_BOOK_D4",
            "AWAY_BOOK_D4",
        ),
        "coherent8_toward_minus_away": (
            "coherent8",
            "TOWARD_BOOK_D4",
            "AWAY_BOOK_D4",
        ),
    }
    primary_rows = []
    secondary_rows = []
    restoration_rows = []
    dose_rows = []
    halves = (("A", slice(0, 32)), ("B", slice(32, 64)))
    cosine_spec = spec_index[INTERVENTION_PRIMARY_SPEC]

    for endpoint, (array_name, left_arm, right_arm) in primary_definitions.items():
        left_index, right_index = arm_index[left_arm], arm_index[right_arm]
        for candidate in ("02", "03"):
            selected = candidates == candidate
            mids = matrix_ids[selected]
            full = (
                (achieved[selected, left_index] == 4)
                & (achieved[selected, right_index] == 4)
            )
            for half, branch_slice in halves:
                array = np.asarray(data[array_name])[selected]
                left_trials = array[:, left_index, branch_slice, cosine_spec]
                right_trials = array[:, right_index, branch_slice, cosine_spec]
                left_q = left_trials.mean(axis=1)
                right_q = right_trials.mean(axis=1)
                effect = _effect_inference(
                    left_q - right_q,
                    mids,
                    ("causal-primary", endpoint, candidate, half),
                )
                primary_rows.append(
                    {
                        "endpoint": endpoint,
                        "array": array_name,
                        "left_arm": left_arm,
                        "right_arm": right_arm,
                        "candidate": candidate,
                        "half": half,
                        "states": int(selected.sum()),
                        "matrices": int(np.unique(mids).size),
                        "left_rate": float(left_q.mean()),
                        "right_rate": float(right_q.mean()),
                        "rate_effect": effect[0],
                        "ci95_lower": effect[1],
                        "ci95_upper": effect[2],
                        "randomization_p_raw": effect[3],
                        "full_dose_fraction": float(full.mean()),
                        **_intervention_power(left_trials, right_trials, mids),
                    }
                )

    primary = pd.DataFrame(primary_rows)
    primary["randomization_p_holm"] = np.nan
    for endpoint in INTERVENTION_PRIMARY_ENDPOINTS:
        selected = primary["endpoint"] == endpoint
        primary.loc[selected, "randomization_p_holm"] = holm_adjust(
            primary.loc[selected, "randomization_p_raw"].tolist()
        )
    primary["passes_gate"] = (
        primary["power_adequate"]
        & (primary["full_dose_fraction"] >= FULL_DOSE_VALIDITY)
        & (primary["rate_effect"] > 0)
        & (primary["ci95_lower"] > 0)
        & (primary["randomization_p_holm"] < 0.05)
    )

    contrast_definitions = {
        "toward_minus_away_d4": ("TOWARD_BOOK_D4", "AWAY_BOOK_D4"),
        "toward_minus_noop_d4": ("TOWARD_BOOK_D4", "NOOP"),
        "away_minus_noop_d4": ("AWAY_BOOK_D4", "NOOP"),
        "random_minus_noop_d4": ("RANDOM_MATCHED_D4", "NOOP"),
    }
    endpoint_arrays = {
        "break_by8": np.asarray(data["break_by8"]),
        "hold8": np.asarray(data["hold8"]),
        "coherent8": np.asarray(data["coherent8"]),
        "strict8": np.asarray(data["strict_labels"]),
    }
    for contrast, (left_arm, right_arm) in contrast_definitions.items():
        li, ri = arm_index[left_arm], arm_index[right_arm]
        for endpoint, array in endpoint_arrays.items():
            for spec, si in spec_index.items():
                for candidate in ("02", "03"):
                    selected = candidates == candidate
                    mids = matrix_ids[selected]
                    for half, branch_slice in halves:
                        left = _state_q(array[selected], li, branch_slice, si)
                        right = _state_q(array[selected], ri, branch_slice, si)
                        effect = _effect_inference(
                            left - right,
                            mids,
                            ("secondary", contrast, endpoint, spec, candidate, half),
                        )
                        secondary_rows.append(
                            {
                                "contrast": contrast,
                                "endpoint": endpoint,
                                "spec": spec,
                                "candidate": candidate,
                                "half": half,
                                "left_rate": float(left.mean()),
                                "right_rate": float(right.mean()),
                                "effect": effect[0],
                                "ci95_lower": effect[1],
                                "ci95_upper": effect[2],
                                "two_sided_randomization_p": effect[4],
                            }
                        )

    for contrast, (left_arm, right_arm) in contrast_definitions.items():
        li, ri = arm_index[left_arm], arm_index[right_arm]
        for time_index, generation in enumerate((1, 4, 8, 32)):
            for candidate in ("02", "03"):
                selected = candidates == candidate
                mids = matrix_ids[selected]
                for half, branch_slice in halves:
                    left = np.nanmean(
                        data["book_cosine"][selected, li, branch_slice, time_index], axis=1
                    )
                    right = np.nanmean(
                        data["book_cosine"][selected, ri, branch_slice, time_index], axis=1
                    )
                    effect = _effect_inference(
                        left - right,
                        mids,
                        ("restoration", contrast, str(generation), candidate, half),
                    )
                    restoration_rows.append(
                        {
                            "contrast": contrast,
                            "generation": generation,
                            "candidate": candidate,
                            "half": half,
                            "effect": effect[0],
                            "ci95_lower": effect[1],
                            "ci95_upper": effect[2],
                            "two_sided_randomization_p": effect[4],
                        }
                    )

    for axis, dose, left_arm, right_arm in (
        ("alignment", 1, "TOWARD_BOOK_D1", "AWAY_BOOK_D1"),
        ("alignment", 4, "TOWARD_BOOK_D4", "AWAY_BOOK_D4"),
    ):
        li, ri = arm_index[left_arm], arm_index[right_arm]
        for endpoint, array_name, direction in (
            ("break_by8", "break_by8", -1),
            ("hold8", "hold8", 1),
            ("coherent8", "coherent8", 1),
            ("strict8", "strict_labels", 0),
        ):
            array = np.asarray(data[array_name])
            for candidate in ("02", "03"):
                selected = candidates == candidate
                mids = matrix_ids[selected]
                for half, branch_slice in halves:
                    left = _state_q(array[selected], li, branch_slice, cosine_spec)
                    right = _state_q(array[selected], ri, branch_slice, cosine_spec)
                    effect = _effect_inference(
                        left - right,
                        mids,
                        ("dose", str(dose), endpoint, candidate, half),
                    )
                    dose_rows.append(
                        {
                            "axis": axis,
                            "dose": dose,
                            "endpoint": endpoint,
                            "candidate": candidate,
                            "half": half,
                            "expected_direction": direction,
                            "effect": effect[0],
                            "ci95_lower": effect[1],
                            "ci95_upper": effect[2],
                            "two_sided_randomization_p": effect[4],
                        }
                    )

    primary.to_csv(OUTPUT_ROOT / "intervention_primary_effects.csv", index=False)
    pd.DataFrame(secondary_rows).to_csv(
        OUTPUT_ROOT / "intervention_secondary_effects.csv", index=False
    )
    pd.DataFrame(restoration_rows).to_csv(
        OUTPUT_ROOT / "rulebook_restoration_effects.csv", index=False
    )
    pd.DataFrame(dose_rows).to_csv(
        OUTPUT_ROOT / "intervention_dose_diagnostics.csv", index=False
    )
    print("Fresh rulebook intervention scored", flush=True)


def _result_classification() -> dict[str, Any]:
    prediction = pd.read_csv(OUTPUT_ROOT / "prediction_effects.csv")
    primary = pd.read_csv(OUTPUT_ROOT / "intervention_primary_effects.csv")
    correlations = pd.read_csv(
        OUTPUT_ROOT / "cross_candidate_holding_correlations.csv"
    )
    prediction_summary = (
        prediction.groupby(["contrast", "spec", "target"], sort=True)
        .agg(
            cells=("passes_gate", "size"),
            positive_cells=("log_loss_gain", lambda values: int((values > 0).sum())),
            passing_cells=("passes_gate", "sum"),
            mean_gain=("log_loss_gain", "mean"),
        )
        .reset_index()
    )
    causal_summary = (
        primary.groupby("endpoint", sort=True)
        .agg(
            cells=("passes_gate", "size"),
            positive_cells=("rate_effect", lambda values: int((values > 0).sum())),
            passing_cells=("passes_gate", "sum"),
            mean_effect=("rate_effect", "mean"),
            minimum_full_dose_fraction=("full_dose_fraction", "min"),
        )
        .reset_index()
    )
    correlation_summary = (
        correlations.groupby(["spec", "target"], sort=True)
        .agg(
            cells=("spearman", "size"),
            positive_cells=("spearman", lambda values: int((values > 0).sum())),
            robust_positive_cells=("ci95_lower", lambda values: int((values > 0).sum())),
            mean_spearman=("spearman", "mean"),
        )
        .reset_index()
    )
    return _json_ready(
        {
            "format": "strict8-rulebook-result-classification-v1",
            "protocol_id": _protocol_id(),
            "claim_status": "post_hoc_mechanistic_followup",
            "prediction": prediction_summary.to_dict(orient="records"),
            "cross_candidate_holding": correlation_summary.to_dict(orient="records"),
            "causal_intervention": causal_summary.to_dict(orient="records"),
            "causal_claim_gate": {
                row.endpoint: bool(row.passing_cells == row.cells)
                for row in causal_summary.itertuples()
            },
            "interpretation": (
                "A causal endpoint is globally supported only if all four candidate-by-half "
                "cells pass power, feasibility, positive interval, and Holm-adjusted tests."
            ),
        }
    )


def _make_figures() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_root = OUTPUT_ROOT / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)

    prediction = pd.read_csv(OUTPUT_ROOT / "prediction_effects.csv")
    chosen = prediction[prediction["spec"] == INTERVENTION_PRIMARY_SPEC].copy()
    chosen["cell"] = (
        chosen["target"]
        + " | c"
        + chosen["candidate"].astype(str).str.zfill(2)
        + " | "
        + chosen["half"]
    )
    table = chosen.pivot_table(
        index="cell", columns="contrast", values="log_loss_gain", aggfunc="mean"
    ).reindex(columns=PREDICTION_CONTRASTS)
    fig, axis = plt.subplots(figsize=(10, 5.5))
    image = axis.imshow(table.to_numpy(), cmap="RdBu_r", aspect="auto")
    axis.set_xticks(
        np.arange(len(table.columns)), table.columns.to_numpy(), rotation=25, ha="right"
    )
    axis.set_yticks(np.arange(len(table.index)), table.index.to_numpy(), fontsize=8)
    axis.set_title("Held-out rulebook/holding log-loss gains (registered cosine)")
    fig.colorbar(image, ax=axis, label="baseline loss - enhanced loss")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(figure_root / f"prediction_gains.{extension}", dpi=180)
    plt.close(fig)

    primary = pd.read_csv(OUTPUT_ROOT / "intervention_primary_effects.csv")
    primary["label"] = (
        primary["endpoint"]
        + " | c"
        + primary["candidate"].astype(str).str.zfill(2)
        + " | "
        + primary["half"]
    )
    primary = primary.sort_values(["endpoint", "candidate", "half"]).reset_index(drop=True)
    fig, axis = plt.subplots(figsize=(10, 6.5))
    for position, row in enumerate(primary.itertuples()):
        color = "#087E8B" if bool(row.passes_gate) else "#6C757D"
        axis.errorbar(
            [row.rate_effect],
            [position],
            xerr=np.asarray(
                [[row.rate_effect - row.ci95_lower], [row.ci95_upper - row.rate_effect]]
            ),
            fmt="o",
            color=color,
            ecolor=color,
            capsize=2,
        )
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.set_yticks(np.arange(len(primary)), primary["label"].to_numpy(), fontsize=7)
    axis.set_xlabel("effect in preregistered positive direction")
    axis.set_title("Fresh rulebook-alignment intervention")
    axis.invert_yaxis()
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(figure_root / f"intervention_primary.{extension}", dpi=180)
    plt.close(fig)

    restoration = pd.read_csv(OUTPUT_ROOT / "rulebook_restoration_effects.csv")
    chosen = restoration[restoration["contrast"] == "toward_minus_away_d4"]
    fig, axis = plt.subplots(figsize=(8, 4.5))
    for (candidate, half), group in chosen.groupby(["candidate", "half"]):
        group = group.sort_values("generation")
        axis.plot(
            group["generation"].to_numpy(),
            group["effect"].to_numpy(),
            marker="o",
            label=f"c{str(candidate).zfill(2)} {half}",
        )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("future selected-lineage fission")
    axis.set_ylabel("toward - away cosine to beta-derived form")
    axis.set_title("Persistence of the rulebook-alignment edit")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(figure_root / f"rulebook_restoration.{extension}", dpi=180)
    plt.close(fig)


def report() -> None:
    classification = _result_classification()
    _write_json(OUTPUT_ROOT / "result_classification.json", classification)
    _make_figures()
    prediction = pd.DataFrame(classification["prediction"])
    causal = pd.DataFrame(classification["causal_intervention"])
    holding = pd.DataFrame(classification["cross_candidate_holding"])
    prediction_lines = "\n".join(
        f"- {row.spec}, {row.target}, {row.contrast}: mean gain {row.mean_gain:.6f}; "
        f"{int(row.passing_cells)}/{int(row.cells)} cells pass."
        for row in prediction.itertuples()
    )
    causal_lines = "\n".join(
        f"- {row.endpoint}: mean directed effect {row.mean_effect:.6f}; "
        f"{int(row.passing_cells)}/{int(row.cells)} cells pass; minimum full-dose "
        f"fraction {row.minimum_full_dose_fraction:.3f}."
        for row in causal.itertuples()
    )
    holding_lines = "\n".join(
        f"- {row.spec}, {row.target}: mean cross-candidate matrix Spearman "
        f"{row.mean_spearman:.3f}; {int(row.robust_positive_cells)}/{int(row.cells)} "
        "cells have a positive 95% lower bound."
        for row in holding.itertuples()
    )
    technical = f"""# Strict-8 rulebook and holding-capacity probe

Status: post-hoc mechanistic follow-up, prospectively frozen before its fresh futures.
No manuscript file was edited, and no quarantined NewIdeas artifact was an analysis input.

## Question

Does the catalytic matrix define a composition-space form and a transferable
holding propensity that explain strict-8 better than generic concentration?

## Deterministic rulebook

For each development and confirmation beta matrix, the analysis solved the
fixed points of the normalized expected join-minus-leave flow implied by the
frozen simulator. Sixteen outcome-blind simplex starts were used, and state
features quantify distance, local flow, self-support, and tangent stability.

## Leakage-safe empirical holding

For each target state and branch half, the holding score uses only the opposite
candidate under the same beta matrix, all five landmarks, and the opposite
branch half. Models were fitted on development matrices and sealed before
confirmation scoring. This is a calibrated matrix diagnostic, not a launch-time
predictor available without sibling futures.

{prediction_lines}

### Direct cross-candidate correlations

{holding_lines}

## Fresh causal alignment intervention

Each retained confirmation state was moved zero, one, or four molecules toward,
away from, or randomly relative to its nearest beta-derived form. Mass and the
occupied set were preserved. Competing arms shared future random streams.

{causal_lines}

The standard strict-8 net effects, all metric sensitivities, one-molecule dose
diagnostics, and relaxation toward the beta-derived form are in the CSV tables.

## Interpretation boundary

This is a post-hoc test of a hypothesis developed after the original strict-8
result. Fresh intervention futures support causal statements only about these
specific edits on retained surviving/observable selected-lineage states. The
cross-candidate score is an empirical world calibration, not an intrinsic
single-state measurement.
"""
    (TASK_ROOT / "DIAGNOSTIC_REPORT.md").write_text(technical, encoding="utf-8")

    causal_passes = int(causal["passing_cells"].sum())
    prediction_passes = int(prediction["passing_cells"].sum())
    lay = f"""# Strict-eight rulebook probe — lay summary

This follow-up asks whether a catalytic matrix behaves like a landscape that
supports particular compositions. We computed that landscape directly from the
simulator equations, checked whether one candidate's independent futures reveal
the other candidate's holding capacity, and then moved molecules toward or away
from the matrix-derived composition before running fresh futures.

Across the held-out predictor comparisons, {prediction_passes} cells passed the
frozen evidence rule. Across the three causal break/hold/coherence questions,
{causal_passes} candidate-by-replicate cells passed. The direction and
replication pattern matter more than either total; they are reported in the
technical tables and figures.

The analysis is explicitly post hoc. It can clarify how strict-eight works, but
it does not become part of the original preregistration and it does not measure
both-daughter reproductive fidelity.
"""
    (TASK_ROOT / "LAY_FINDINGS.md").write_text(lay, encoding="utf-8")
    print("Rulebook technical report, lay summary, figures, and classification written", flush=True)


REQUIRED_OUTPUT_FILES = (
    "prediction_effects.csv",
    "cross_candidate_holding_correlations.csv",
    "intervention_validation.csv",
    "intervention_primary_effects.csv",
    "intervention_secondary_effects.csv",
    "rulebook_restoration_effects.csv",
    "intervention_dose_diagnostics.csv",
    "result_classification.json",
)


def _deterministic_audit(workers: int = 4) -> dict[str, Any]:
    data = _load_intervention()
    experiment, cases = geometry_source._cases("confirmation", workers)
    specs = geometry_source._specs()
    state_indices = (0, 5, 1000, 1005)
    branches = (0, 63)
    mismatches = 0
    edit_mismatches = 0
    futures = 0
    for state_index in state_indices:
        case = cases[state_index]
        target = data["target_forms"][state_index]
        for arm_index, arm in enumerate(EDIT_ARMS):
            edit = apply_rulebook_edit(
                case.snapshot.composition, target, arm, _edit_seed(case, arm)
            )
            if not np.array_equal(
                edit.composition,
                data["edited_compositions"][state_index, arm_index],
            ):
                edit_mismatches += 1
            snapshot = _edited_snapshot(case.snapshot, edit.composition)
            for branch in branches:
                records, complete = simulate_future_absorbing(
                    snapshot,
                    case.beta,
                    experiment.gard,
                    CANDIDATES[case.candidate],
                    INTERVENTION_HORIZON,
                    np.random.default_rng(_future_seed(case, branch)),
                )
                outcomes, _ = score_all_specs(records, specs)
                first8 = _first8_outcomes(records, specs)
                observed_labels = np.asarray([int(outcome.event) for outcome in outcomes])
                observed_gates = np.asarray([outcome.deepest_gate for outcome in outcomes])
                expected = (
                    data["strict_labels"][state_index, arm_index, branch],
                    data["strict_gates"][state_index, arm_index, branch],
                    data["break_by8"][state_index, arm_index, branch],
                    data["hold8"][state_index, arm_index, branch],
                    data["coherent8"][state_index, arm_index, branch],
                    data["complete8"][state_index, arm_index, branch],
                )
                actual = (observed_labels, observed_gates, *first8)
                if any(not np.array_equal(left, right) for left, right in zip(actual, expected)):
                    mismatches += 1
                if int(complete) != int(data["completed32"][state_index, arm_index, branch]):
                    mismatches += 1
                if len(records) != int(data["observed"][state_index, arm_index, branch]):
                    mismatches += 1
                futures += 1
    return {
        "state_indices": list(state_indices),
        "branches": list(branches),
        "arms": list(EDIT_ARMS),
        "futures_replayed": futures,
        "edit_mismatches": edit_mismatches,
        "future_mismatches": mismatches,
        "exact": edit_mismatches == 0 and mismatches == 0,
    }


def verify(workers: int = 4) -> None:
    protocol = verify_protocol()
    verify_checksums(RULEBOOK_ROOT)
    verify_checksums(HOLDING_ROOT)
    verify_checksums(MODEL_ROOT)
    verify_checksums(INTERVENTION_ROOT)
    missing = [name for name in REQUIRED_OUTPUT_FILES if not (OUTPUT_ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing outputs: {missing}")
    if not (TASK_ROOT / "DIAGNOSTIC_REPORT.md").is_file() or not (
        TASK_ROOT / "LAY_FINDINGS.md"
    ).is_file():
        raise FileNotFoundError("reports are incomplete")

    expected_rows = {
        "prediction": 96,
        "correlation": 60,
        "primary": 12,
        "secondary": 192,
        "restoration": 64,
        "dose": 32,
    }
    observed_rows = {
        "prediction": len(pd.read_csv(OUTPUT_ROOT / "prediction_effects.csv")),
        "correlation": len(
            pd.read_csv(OUTPUT_ROOT / "cross_candidate_holding_correlations.csv")
        ),
        "primary": len(pd.read_csv(OUTPUT_ROOT / "intervention_primary_effects.csv")),
        "secondary": len(
            pd.read_csv(OUTPUT_ROOT / "intervention_secondary_effects.csv")
        ),
        "restoration": len(
            pd.read_csv(OUTPUT_ROOT / "rulebook_restoration_effects.csv")
        ),
        "dose": len(pd.read_csv(OUTPUT_ROOT / "intervention_dose_diagnostics.csv")),
    }
    if observed_rows != expected_rows:
        raise AssertionError(f"output dimensions differ: {observed_rows}")

    for cohort in ("development", "confirmation"):
        rulebook = _load_npz(RULEBOOK_ROOT / f"{cohort}.npz")
        if np.max(rulebook["solver_flow_residuals"]) >= 1e-8:
            raise AssertionError("rulebook fixed-point residual is too large")
        if not np.all(np.isfinite(rulebook["state_features"])):
            raise AssertionError("nonfinite rulebook features")

    intervention = _load_intervention()
    expected_shape = (2000, len(EDIT_ARMS), INTERVENTION_BRANCHES, len(SPEC_NAMES))
    if intervention["strict_labels"].shape != expected_shape:
        raise AssertionError("intervention shape differs")
    source = _source_arrays("confirmation")
    if not np.all(
        intervention["edited_compositions"].sum(axis=2)
        == np.asarray(source["compositions"]).sum(axis=1)[:, None]
    ):
        raise AssertionError("intervention mass differs")
    if not np.array_equal(
        intervention["occupied_before"], intervention["occupied_after"]
    ):
        raise AssertionError("intervention occupied set differs")
    audit = _deterministic_audit(workers)
    if not audit["exact"]:
        raise AssertionError(f"deterministic replay failed: {audit}")
    _write_json(OUTPUT_ROOT / "deterministic_replay_audit.json", audit)

    verification = {
        "format": "strict8-rulebook-verification-v1",
        "protocol_id": protocol["protocol_id"],
        "verified": True,
        "expected_table_rows": expected_rows,
        "intervention_shape": list(expected_shape),
        "fresh_futures": int(np.prod(expected_shape[:3])),
        "deterministic_replay": audit,
        "source_contract_verified": True,
        "newideas_inputs_used": False,
        "no_manuscript_files_modified": True,
    }
    _write_json(OUTPUT_ROOT / "verification_audit.json", verification)
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
    print(
        f"VERIFIED: {len(verify_checksums(OUTPUT_ROOT))} outputs; "
        f"{verification['fresh_futures']:,} fresh futures; deterministic replay exact",
        flush=True,
    )


def status() -> None:
    value = {
        "protocol_frozen": (PROTOCOL_ROOT / "analysis_protocol.json").is_file(),
        "rulebooks_complete": all(
            (RULEBOOK_ROOT / f"{cohort}.npz").is_file()
            for cohort in ("development", "confirmation")
        ),
        "holding_features_complete": all(
            (HOLDING_ROOT / f"{cohort}_{half}.npz").is_file()
            for cohort in ("development", "confirmation")
            for half in ("A", "B")
        ),
        "models_sealed": (MODEL_ROOT / "model_seal.json").is_file(),
        "prediction_scored": (OUTPUT_ROOT / "prediction_effects.csv").is_file(),
        "intervention_checkpoints": sum(
            _checkpoint_path(index).is_file() for index in range(2000)
        ),
        "intervention_expected": 2000,
        "intervention_scored": (
            OUTPUT_ROOT / "intervention_primary_effects.csv"
        ).is_file(),
        "reports_complete": (TASK_ROOT / "DIAGNOSTIC_REPORT.md").is_file()
        and (TASK_ROOT / "LAY_FINDINGS.md").is_file(),
        "verified": (OUTPUT_ROOT / "verification_audit.json").is_file(),
    }
    print(json.dumps(value, indent=2), flush=True)


def run_all(workers: int) -> None:
    prepare()
    build_rulebooks()
    build_holding_features()
    fit_models()
    score_models()
    run_intervention(workers)
    analyze_intervention()
    report()
    verify(min(workers, 4))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-hoc strict-8 rulebook and holding-capacity probe"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("intervention", "verify", "all"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--workers", type=int, default=12)
    for command in (
        "prepare",
        "rulebooks",
        "holding",
        "fit",
        "score",
        "analyze-intervention",
        "report",
        "status",
    ):
        subparsers.add_parser(command)
    arguments = parser.parse_args()
    commands = {
        "prepare": prepare,
        "rulebooks": build_rulebooks,
        "holding": build_holding_features,
        "fit": fit_models,
        "score": score_models,
        "intervention": lambda: run_intervention(arguments.workers),
        "analyze-intervention": analyze_intervention,
        "report": report,
        "verify": lambda: verify(arguments.workers),
        "status": status,
        "all": lambda: run_all(arguments.workers),
    }
    commands[arguments.command]()


if __name__ == "__main__":
    main()
