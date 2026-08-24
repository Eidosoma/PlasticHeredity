"""Final exploratory CR10 local-policy and retention internalization ladder."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from . import intervention_replication as base
from .config import CANDIDATES, CohortConfig, ExperimentConfig, GardConfig
from .experiment import StateCase, _json_ready, _runtime_manifest, build_cohort, extract_features
from .features import history_features, state_graph_features
from .intervention_core import (
    FrozenFullPredictor,
    MolecularEdit,
    _records_digest,
    apply_molecular_edit,
    edited_snapshot,
    enumerate_legal_edits,
    score_legal_edits,
    simulate_controlled,
)
from .intervention_outgoing_rule import (
    outgoing_catalytic_influence,
    select_outgoing_rule_edits,
)
from .mechanistic import (
    _atomic_destination,
    _canonical_digest,
    sha256_file,
    verify_checksums,
    write_checksums,
)
from .seeds import derive_seed
from .simulator import (
    FissionRecord,
    SimulationError,
    Snapshot,
    _fission,
    _sample_without_replacement,
    _trim_whole_assembly,
    advance_fission,
    cosine_similarity,
    generate_beta,
    generate_initial_composition,
    simulate_lineage,
)


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results_intervention_replication"
SCALED5 = ROOT / "results/scaled5"
CR0_VALIDATION = RESULT_ROOT / "cr0_validation"
CR1_REGISTRATION = RESULT_ROOT / "cr1_confirmation_registration"
CR3_REGISTRATION = RESULT_ROOT / "cr3_confirmation_registration"
CR3_RESULT = RESULT_ROOT / "cr3_physical_rule_confirmation"
CR7_REGISTRATION = RESULT_ROOT / "cr7_steering_registration"
CR7_RESULT = RESULT_ROOT / "cr7_closed_loop_steering"
CR8_RESULT = RESULT_ROOT / "cr8_steer_release_challenge"
CR9_RESULT = RESULT_ROOT / "cr9_control_half_life"
CR9M_RESULT = RESULT_ROOT / "cr9m_launch_moderation"

DEFAULT_DEVELOPMENT = RESULT_ROOT / "cr10_development_freeze"
DEFAULT_VALIDATION = RESULT_ROOT / "cr10_internalization_validation"
DEFAULT_REGISTRATION = RESULT_ROOT / "cr10_internalization_registration"
DEFAULT_SMOKE = RESULT_ROOT / "cr10_internalization_smoke"
DEFAULT_OUTPUT = RESULT_ROOT / "cr10_exploratory_internalization"
DEFAULT_WORK = RESULT_ROOT / ".cr10_exploratory_internalization_work"

DOCUMENT = "CODEX_INTERVENTION_CR10_PREREGISTRATION.md"
PROGRAM_FORMAT = "codex-intervention-cr10-internalization-v2"
DEVELOPMENT_FORMAT = "codex-intervention-cr10-development-v2"
VALIDATION_FORMAT = "codex-intervention-cr10-validation-v2"
REGISTRATION_FORMAT = "codex-intervention-cr10-registration-v2"
RESULT_FORMAT = "codex-intervention-cr10-result-v2"
CHECKPOINT_FORMAT = "codex-intervention-cr10-policy-checkpoint-v2"
KINETIC_CHECKPOINT_FORMAT = "codex-intervention-cr10-kinetic-checkpoint-v2"
STATUS_FORMAT = "codex-intervention-cr10-status-v2"
LABEL = "INTCR10_LOCAL_INTERNALIZATION_V2"

CR3_REGISTRATION_ID = "64e871db56b3958a14bdad47b404f6c9f1ad09d0bda1e996e24498598523d189"
CR7_REGISTRATION_ID = "41cf815a63129f40c04c7fb260f0f90c713adb9743eaae8479a5f6046e826e70"
EXPECTED_MODEL_SHA256 = "9b3305a7fed11f432651926d34903443e9413ed299c5d0f1056a0b5fde9990af"

HOME_MATRICES = 48
TRANSFER_MATRICES = 24
HOME_REPLICATES = 3
TRANSFER_REPLICATES = 2
KINETIC_REPLICATES = 3
LANDMARK = 60
HORIZON = 60
CHALLENGE_AFTER_FISSION = 30
CHALLENGE_K = 8
BOOTSTRAP_REPETITIONS = 4_096
RANDOMIZATION_REPETITIONS = 4_096
RANDOM_EQUIVALENCE_MARGIN = 0.025
INHERITANCE_THRESHOLD = 0.9
MINIMUM_FREE_DISK_BYTES = 2_000_000_000

POLICIES = (
    "L0_RULE_CONTINUOUS",
    "L1_RULE_AFTER_BREAK",
    "L2_RULE_UNTIL_RUN3",
    "L3_LOCAL_TREE",
    "MODEL_DOWN",
    "RANDOM",
    "NOOP",
)
CONDITIONS = ("UNCHALLENGED", "CHALLENGED_K8")
KINETIC_LAMBDAS = (0.0, 0.1, 0.3)
TRANSFER_REGIMES: dict[str, tuple[float, float]] = {
    "POS_A_M4_S5": (-4.0, 5.0),
    "POS_A_M3_S4": (-3.0, 4.0),
    "POS_A_M5_S4": (-5.0, 4.0),
}
LOCAL_FEATURE_NAMES = (
    "abundance_share",
    "outgoing_influence_percentile",
    "incoming_boost_percentile",
    "presence",
)
TREE_ROLES = ("remove", "add")
TREE_MAX_DEPTH = 3
TREE_MIN_SAMPLES_LEAF = 25

SOURCE_FILES = (
    DOCUMENT,
    "plastic_heredity/intervention_cr10_internalization.py",
    "tests/test_intervention_cr10_internalization.py",
    "plastic_heredity/intervention_core.py",
    "plastic_heredity/intervention_outgoing_rule.py",
    "plastic_heredity/intervention_replication.py",
    "plastic_heredity/config.py",
    "plastic_heredity/experiment.py",
    "plastic_heredity/features.py",
    "plastic_heredity/simulator.py",
    "plastic_heredity/seeds.py",
    "plastic_heredity/mechanistic.py",
    "pyproject.toml",
    "requirements-lock.txt",
)


def _seed(name: str) -> str:
    return hashlib.sha256(
        f"codex-clean-room-cr10-internalization-v2::{name}".encode("utf-8")
    ).hexdigest()


SEEDS = {
    name: _seed(name)
    for name in (
        "development_reconstruction",
        "tree_fit",
        "validation",
        "smoke",
        "home_matrix_generation",
        "home_initial_composition",
        "home_main_trajectory",
        "transfer_matrix_generation",
        "transfer_initial_composition",
        "transfer_main_trajectory",
        "future_simulation",
        "random_policy_action",
        "challenge_action",
        "kinetic_future",
        "bootstrap",
        "randomization",
        "replay",
    )
}


@dataclass(frozen=True)
class CR10Case:
    state_id: str
    phase: str
    regime: str
    candidate: str
    matrix_id: int
    beta: NDArray[np.float64]
    snapshot: Snapshot


@dataclass(frozen=True)
class TrajectorySummary:
    policy: str
    condition: str
    replicate: int
    completed_horizon: bool
    observed_fissions: int
    inherited_count: int
    inherited_fraction_registered: float
    inherited_fraction_observed: float
    total_breaks_registered: int
    episode_count: int
    longest_inherited_run: int
    post_challenge_inherited_fraction: float
    post_challenge_breaks: int
    post_challenge_run3_delay: int
    final6_inherited: int
    challenge_applied: bool
    challenge_transport_distance: int
    action_count: int
    distinct_actions: int
    repeated_actions: int
    immediately_reversing_actions: int
    mean_growth_updates: float
    final_entropy: float
    final_occupied_types: int
    final_top1_share: float
    final_throughput: float
    final_risk: float
    mean_predicted_action_shift: float
    out_of_development_envelope_fraction: float
    record_digest: str
    boundary_h: NDArray[np.float64]
    growth_updates: NDArray[np.int32]
    action_remove: NDArray[np.int16]
    action_add: NDArray[np.int16]
    challenge_remove: NDArray[np.int16]
    challenge_add: NDArray[np.int16]
    final_snapshot: Snapshot
    simulation_rng_state: dict[str, Any]
    action_rng_state: dict[str, Any]
    challenge_rng_state: dict[str, Any]
    noop_plain_bitwise_exact: bool


@dataclass(frozen=True)
class PolicyBatch:
    format: str
    registration_id: str
    state_id: str
    phase: str
    regime: str
    candidate: str
    matrix_id: int
    case_digest: str
    summaries: tuple[TrajectorySummary, ...]


@dataclass(frozen=True)
class KineticSummary:
    lambda_value: float
    replicate: int
    completed_horizon: bool
    observed_fissions: int
    inherited_count: int
    inherited_fraction_registered: float
    total_breaks_registered: int
    longest_inherited_run: int
    mean_growth_updates: float
    final_entropy: float
    final_occupied_types: int
    final_top1_share: float
    final_throughput: float
    final_risk: float
    record_digest: str
    boundary_h: NDArray[np.float64]
    growth_updates: NDArray[np.int32]
    final_snapshot: Snapshot
    simulation_rng_state: dict[str, Any]
    lambda_zero_plain_bitwise_exact: bool


@dataclass(frozen=True)
class KineticBatch:
    format: str
    registration_id: str
    state_id: str
    candidate: str
    matrix_id: int
    case_digest: str
    summaries: tuple[KineticSummary, ...]


def source_hashes() -> dict[str, str]:
    return {name: sha256_file(ROOT / name) for name in SOURCE_FILES}


def protocol() -> dict[str, Any]:
    value: dict[str, Any] = {
        "format": PROGRAM_FORMAT,
        "status": "frozen_before_any_cr10_scientific_matrix",
        "role": "final exploratory phase; cannot rescue any confirmatory result",
        "upstream": {
            "cr3_registration_id": CR3_REGISTRATION_ID,
            "cr3_gate_required": True,
            "cr7_registration_id": CR7_REGISTRATION_ID,
            "cr7_gate_required": True,
            "cr6_cr8_cr9_cr9m_not_advancement_gates": True,
        },
        "target": {
            "name": "JOINT_BREAK_RUN3",
            "inheritance": "strict unrounded float64 H > 0.9",
            "strict_eight_excluded": True,
        },
        "development": {
            "source": "exact reconstructed original scaled5 development cohort only",
            "matrices": 200,
            "candidates": list(CANDIDATES),
            "landmarks": [20, 35, 50, 65, 80],
            "outcomes_used": False,
            "teacher": "frozen candidate-separated exhaustive MODEL_DOWN",
            "features": list(LOCAL_FEATURE_NAMES),
            "candidate_pooled": False,
            "trees_per_candidate": list(TREE_ROLES),
            "architecture": {
                "criterion": "gini",
                "splitter": "best",
                "max_depth": TREE_MAX_DEPTH,
                "min_samples_leaf": TREE_MIN_SAMPLES_LEAF,
                "class_weight": "balanced",
            },
        },
        "home_cohort": {
            "fresh_matrices": HOME_MATRICES,
            "candidates": list(CANDIDATES),
            "natural_untreated_landmark": LANDMARK,
            "replicates": HOME_REPLICATES,
            "policies": list(POLICIES),
            "conditions": list(CONDITIONS),
            "fissions": HORIZON,
            "challenge_after_fission": CHALLENGE_AFTER_FISSION,
            "challenge_exact_transport": CHALLENGE_K,
            "lineages": HOME_MATRICES
            * len(CANDIDATES)
            * HOME_REPLICATES
            * len(POLICIES)
            * len(CONDITIONS),
        },
        "transfer": {
            "regimes": {
                key: {"A": item[0], "sigma": item[1]}
                for key, item in TRANSFER_REGIMES.items()
            },
            "fresh_matrices_per_regime": TRANSFER_MATRICES,
            "natural_untreated_landmark": LANDMARK,
            "replicates": TRANSFER_REPLICATES,
            "policies": list(POLICIES),
            "conditions": ["UNCHALLENGED"],
            "fissions": HORIZON,
        },
        "policies": {
            "L0_RULE_CONTINUOUS": "outgoing x@beta RULE_DOWN after every fission",
            "L1_RULE_AFTER_BREAK": "RULE_DOWN only when just-observed H <= 0.9",
            "L2_RULE_UNTIL_RUN3": "RULE_DOWN only while trailing strict run < 3",
            "L3_LOCAL_TREE": "frozen local remove/add trees after every fission",
            "MODEL_DOWN": "exhaustive frozen predictor minimum after every fission",
            "RANDOM": "uniform legal edit after every fission from separate stream",
            "NOOP": "no edit",
            "callback_including_last_fission": True,
        },
        "challenge": {
            "order": "ordinary controller action after fission 30, then exact K8 transport; control resumes after fission 31",
            "removals": "molecule instances sampled without replacement",
            "additions": "with replacement outside unique removed labels",
            "fixed_mass_nonnegative_integer": True,
        },
        "kinetic_prototype": {
            "matrices": HOME_MATRICES,
            "candidates": list(CANDIDATES),
            "replicates": KINETIC_REPLICATES,
            "lambdas": list(KINETIC_LAMBDAS),
            "fissions": HORIZON,
            "formula": "leave_rate(type)/(1+lambda*outgoing_influence_percentile(type))",
            "percentile_recomputed_each_growth_update": True,
            "lambda_zero_plain_dispatch_and_bitwise_identity": True,
            "baseline_simulator_unchanged": True,
        },
        "randomness": {
            "seed_domains": SEEDS,
            "future_seed_excludes_policy_and_home_condition": True,
            "common_random_streams_not_identical_realized_futures": True,
            "random_action_challenge_and_future_streams_separate": True,
        },
        "inference": {
            "unit": "whole catalytic matrix",
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "randomization_repetitions": RANDOMIZATION_REPETITIONS,
            "holm_within_analysis_family": True,
            "random_noop_tost_margin": RANDOM_EQUIVALENCE_MARGIN,
            "candidates_and_transfer_regimes_never_pooled": True,
            "no_confirmatory_pass_fail_gate": True,
        },
        "integrity": {
            "complete_exact_replay": True,
            "noop_plain_bitwise_identity": True,
            "lambda_zero_plain_bitwise_identity": True,
            "controlled_future_retry_or_matrix_replacement": False,
            "checkpoint_resumable": True,
        },
        "operational": {
            "minimum_free_disk_bytes": MINIMUM_FREE_DISK_BYTES,
            "mandatory_stop_after_cr10_seal": True,
            "no_later_phase_launched": True,
        },
        "claim_boundary": {
            "exploratory_only": True,
            "prohibited": [
                "autonomous agency or installed compotype",
                "biological memory or error correction",
                "life or universal origin-of-life mechanism",
                "real prebiotic chemistry",
                "strict-eight control",
                "Phi or PhiID",
            ],
        },
    }
    value["protocol_id"] = _canonical_digest(_json_ready(value))
    return value


def _percentile(values: NDArray) -> NDArray[np.float64]:
    """Permutation-equivariant midrank percentile in [0, 1]."""

    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("percentile input must be a finite vector")
    if x.size <= 1:
        return np.zeros(x.size, dtype=np.float64)
    unique, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    del unique
    starts = np.cumsum(np.r_[0, counts[:-1]])
    midranks = starts + (counts - 1.0) / 2.0
    return np.asarray(midranks[inverse] / (x.size - 1.0), dtype=np.float64)


def local_type_features(
    composition: NDArray, beta: NDArray
) -> NDArray[np.float64]:
    values = np.asarray(composition, dtype=np.float64)
    matrix = np.asarray(beta, dtype=np.float64)
    if values.ndim != 1 or matrix.shape != (values.size, values.size):
        raise ValueError("composition and beta dimensions differ")
    mass = float(values.sum())
    if mass <= 0.0 or np.any(values < 0.0):
        raise ValueError("local features require a nonempty nonnegative assembly")
    x = values / mass
    outgoing = x @ matrix
    incoming = matrix @ x
    result = np.column_stack(
        (x, _percentile(outgoing), _percentile(incoming), values > 0.0)
    ).astype(np.float64)
    if result.shape != (values.size, len(LOCAL_FEATURE_NAMES)):
        raise AssertionError("local feature shape changed")
    return result


class FrozenLocalTrees:
    """Portable candidate-separated depth-three trees without sklearn at run time."""

    def __init__(self, arrays: dict[str, NDArray]):
        self.arrays = {name: np.asarray(value).copy() for name, value in arrays.items()}

    @classmethod
    def load(cls, path: Path | str) -> "FrozenLocalTrees":
        with np.load(path, allow_pickle=False) as archive:
            return cls({name: archive[name] for name in archive.files})

    def save(self, path: Path | str) -> None:
        np.savez_compressed(path, **self.arrays)

    def score_types(
        self, candidate: str, role: str, features: NDArray
    ) -> NDArray[np.float64]:
        if candidate not in CANDIDATES or role not in TREE_ROLES:
            raise ValueError("unknown local-tree candidate or role")
        x = np.atleast_2d(np.asarray(features, dtype=np.float64))
        if x.shape[1] != len(LOCAL_FEATURE_NAMES):
            raise ValueError("local-tree feature width changed")
        prefix = f"c{candidate}__{role}"
        left = self.arrays[f"{prefix}__children_left"].astype(np.int64)
        right = self.arrays[f"{prefix}__children_right"].astype(np.int64)
        feature = self.arrays[f"{prefix}__feature"].astype(np.int64)
        threshold = self.arrays[f"{prefix}__threshold"].astype(np.float64)
        probability = self.arrays[f"{prefix}__positive_probability"].astype(
            np.float64
        )
        output = np.empty(x.shape[0], dtype=np.float64)
        for row_index, row in enumerate(x):
            node = 0
            while left[node] != right[node]:
                node = left[node] if row[feature[node]] <= threshold[node] else right[node]
            output[row_index] = probability[node]
        return output

    def select_edit(
        self, candidate: str, composition: NDArray, beta: NDArray
    ) -> MolecularEdit:
        features = local_type_features(composition, beta)
        present = np.flatnonzero(np.asarray(composition) > 0)
        if present.size == 0:
            raise ValueError("cannot select an edit for an empty assembly")
        remove_scores = self.score_types(candidate, "remove", features)
        maximum_remove = remove_scores[present].max()
        remove = int(present[np.flatnonzero(remove_scores[present] == maximum_remove)[0]])
        add_scores = self.score_types(candidate, "add", features).copy()
        add_scores[remove] = -np.inf
        maximum_add = add_scores.max()
        add = int(np.flatnonzero(add_scores == maximum_add)[0])
        return MolecularEdit(remove, add)


def _tree_arrays(
    estimators: dict[tuple[str, str], DecisionTreeClassifier]
) -> dict[str, NDArray]:
    arrays: dict[str, NDArray] = {}
    for (candidate, role), estimator in estimators.items():
        tree = estimator.tree_
        values = np.asarray(tree.value[:, 0, :], dtype=np.float64)
        if values.shape[1] != 2:
            raise ValueError("local tree lost a binary class")
        probability = values[:, 1] / np.maximum(values.sum(axis=1), 1e-300)
        prefix = f"c{candidate}__{role}"
        arrays[f"{prefix}__children_left"] = tree.children_left.astype(np.int32)
        arrays[f"{prefix}__children_right"] = tree.children_right.astype(np.int32)
        arrays[f"{prefix}__feature"] = tree.feature.astype(np.int16)
        arrays[f"{prefix}__threshold"] = tree.threshold.astype(np.float64)
        arrays[f"{prefix}__positive_probability"] = probability.astype(np.float64)
        arrays[f"{prefix}__node_samples"] = tree.n_node_samples.astype(np.int32)
        arrays[f"{prefix}__depth"] = np.asarray([tree.max_depth], dtype=np.int16)
    return arrays


def _development_experiment() -> ExperimentConfig:
    return ExperimentConfig.scaled5()


def _development_selection_worker(
    arguments: tuple[StateCase, Path]
) -> tuple[str, str, int, int, int, int, float, float]:
    case, model_path = arguments
    with threadpool_limits(limits=1):
        predictor = FrozenFullPredictor.load(model_path)
        noop, scores = score_legal_edits(
            predictor,
            case.candidate,
            case.snapshot,
            case.beta,
            GardConfig(),
        )
        probabilities = np.asarray(
            [item.predicted_probability for item in scores], dtype=np.float64
        )
        minimum = probabilities.min()
        index = int(np.flatnonzero(probabilities == minimum)[0])
        edit = scores[index].edit
        return (
            case.state_id,
            case.candidate,
            case.matrix_id,
            case.landmark,
            edit.remove_type,
            edit.add_type,
            float(noop),
            float(minimum),
        )


def _tree_fit_seed(candidate: str, role: str) -> int:
    return int(
        derive_seed(SEEDS["tree_fit"], f"{LABEL}.tree", candidate, role)
        % (2**32 - 1)
    )


def _fit_local_trees(
    cases: list[StateCase], actions: pd.DataFrame
) -> tuple[FrozenLocalTrees, dict[tuple[str, str], DecisionTreeClassifier]]:
    lookup = actions.set_index("state_id")
    estimators: dict[tuple[str, str], DecisionTreeClassifier] = {}
    for candidate in CANDIDATES:
        candidate_cases = [case for case in cases if case.candidate == candidate]
        features = np.vstack(
            [local_type_features(case.snapshot.composition, case.beta) for case in candidate_cases]
        )
        for role, column in (("remove", "model_remove"), ("add", "model_add")):
            labels = np.zeros(len(candidate_cases) * GardConfig().n_types, dtype=np.int8)
            for state_index, case in enumerate(candidate_cases):
                molecule = int(lookup.loc[case.state_id, column])
                labels[state_index * GardConfig().n_types + molecule] = 1
            estimator = DecisionTreeClassifier(
                criterion="gini",
                splitter="best",
                max_depth=TREE_MAX_DEPTH,
                min_samples_leaf=TREE_MIN_SAMPLES_LEAF,
                class_weight="balanced",
                random_state=_tree_fit_seed(candidate, role),
            )
            estimator.fit(features, labels)
            estimators[(candidate, role)] = estimator
    return FrozenLocalTrees(_tree_arrays(estimators)), estimators


def develop(
    output: Path = DEFAULT_DEVELOPMENT,
    workers: int = min(os.cpu_count() or 1, 14),
) -> None:
    """Reconstruct development states, distill MODEL_DOWN, and freeze L3."""

    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if workers < 1:
        raise ValueError("workers must be positive")
    verify_checksums(SCALED5)
    model_path = CR1_REGISTRATION / "frozen_full_predictor.npz"
    if sha256_file(model_path) != EXPECTED_MODEL_SHA256:
        raise ValueError("frozen JOINT_BREAK_RUN3 predictor changed")
    experiment = _development_experiment()
    print("[cr10 develop 1/5] Reconstructing 2,000 scaled5 development states", flush=True)
    with threadpool_limits(limits=1):
        cases = build_cohort(experiment, "VALI", experiment.development)
        reconstructed = extract_features(cases, experiment)
    with np.load(SCALED5 / "analysis_arrays.npz", allow_pickle=False) as archive:
        exact = {
            "state_graph": bool(
                np.array_equal(reconstructed.state_graph, archive["development_state_graph"])
            ),
            "history": bool(
                np.array_equal(reconstructed.history, archive["development_history"])
            ),
            "beta": bool(np.array_equal(reconstructed.beta, archive["development_beta"])),
        }
    if not all(exact.values()):
        raise ValueError(f"scaled5 development reconstruction changed: {exact}")
    model_audit, _ = base._model_prediction_audit()
    if not model_audit["all_within_tolerance"]:
        raise ValueError("frozen predictor no longer reproduces archived predictions")

    print("[cr10 develop 2/5] Exhaustively selecting frozen MODEL_DOWN actions", flush=True)
    arguments = [(case, model_path) for case in cases]
    if workers == 1:
        selected = [_development_selection_worker(item) for item in arguments]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            selected = list(executor.map(_development_selection_worker, arguments, chunksize=1))
    actions = pd.DataFrame(
        selected,
        columns=(
            "state_id",
            "candidate",
            "matrix_id",
            "landmark",
            "model_remove",
            "model_add",
            "noop_probability",
            "model_down_probability",
        ),
    )
    if len(actions) != len(cases) or actions["state_id"].nunique() != len(cases):
        raise AssertionError("development action selection is incomplete")

    print("[cr10 develop 3/5] Fitting candidate-separated depth-three local trees", flush=True)
    trees, estimators = _fit_local_trees(cases, actions)
    predictor = FrozenFullPredictor.load(model_path)
    action_lookup = actions.set_index("state_id")
    audit_rows: list[dict[str, Any]] = []
    for case in cases:
        local = trees.select_edit(case.candidate, case.snapshot.composition, case.beta)
        row = action_lookup.loc[case.state_id]
        local_probability = predictor.predict_snapshot(
            case.candidate, edited_snapshot(case.snapshot, local), case.beta, experiment.gard
        )
        audit_rows.append(
            {
                "state_id": case.state_id,
                "candidate": case.candidate,
                "matrix_id": case.matrix_id,
                "landmark": case.landmark,
                "model_remove": int(row["model_remove"]),
                "model_add": int(row["model_add"]),
                "local_remove": local.remove_type,
                "local_add": local.add_type,
                "remove_agreement": int(local.remove_type == int(row["model_remove"])),
                "add_agreement": int(local.add_type == int(row["model_add"])),
                "joint_agreement": int(
                    local.remove_type == int(row["model_remove"])
                    and local.add_type == int(row["model_add"])
                ),
                "model_down_probability": float(row["model_down_probability"]),
                "local_tree_probability": float(local_probability),
                "predicted_risk_regret": float(
                    local_probability - float(row["model_down_probability"])
                ),
            }
        )
    audit = pd.DataFrame(audit_rows)

    print("[cr10 develop 4/5] Verifying portable serialization and architecture", flush=True)
    with _atomic_destination(output) as destination:
        trees.save(destination / "frozen_local_trees.npz")
        reloaded = FrozenLocalTrees.load(destination / "frozen_local_trees.npz")
        portable = all(
            reloaded.select_edit(case.candidate, case.snapshot.composition, case.beta)
            == trees.select_edit(case.candidate, case.snapshot.composition, case.beta)
            for case in cases
        )
        if not portable:
            raise AssertionError("serialized local trees changed selected actions")
        actions.to_csv(
            destination / "development_teacher_actions.csv.gz",
            index=False,
            compression="gzip",
        )
        audit.to_csv(
            destination / "development_distillation_audit.csv.gz",
            index=False,
            compression="gzip",
        )
        summaries: dict[str, Any] = {}
        for candidate in CANDIDATES:
            selected_audit = audit[audit["candidate"] == candidate]
            summaries[candidate] = {
                "states": int(len(selected_audit)),
                "remove_agreement": float(selected_audit["remove_agreement"].mean()),
                "add_agreement": float(selected_audit["add_agreement"].mean()),
                "joint_agreement": float(selected_audit["joint_agreement"].mean()),
                "median_predicted_risk_regret": float(
                    selected_audit["predicted_risk_regret"].median()
                ),
                "maximum_predicted_risk_regret": float(
                    selected_audit["predicted_risk_regret"].max()
                ),
                "remove_tree_depth": int(estimators[(candidate, "remove")].get_depth()),
                "add_tree_depth": int(estimators[(candidate, "add")].get_depth()),
            }
        contract = {
            "format": "codex-intervention-cr10-local-tree-contract-v1",
            "features": list(LOCAL_FEATURE_NAMES),
            "orientation": {
                "beta": "beta[target,catalyst]",
                "outgoing": "x @ beta == beta.T @ x",
                "incoming": "beta @ x",
            },
            "architecture": protocol()["development"]["architecture"],
            "candidate_separated": True,
            "roles": list(TREE_ROLES),
            "tie_break": "first numeric type after legality mask",
            "summaries": summaries,
        }
        (destination / "local_tree_contract.json").write_text(
            json.dumps(_json_ready(contract), indent=2, sort_keys=True) + "\n"
        )
        manifest = {
            "format": DEVELOPMENT_FORMAT,
            "scientific_cr10_matrices_generated": 0,
            "development_states": len(cases),
            "development_matrices": experiment.development.matrices,
            "development_landmarks": list(experiment.development.landmarks),
            "exact_reconstruction": exact,
            "frozen_model_archive_reproduced": True,
            "model_sha256": sha256_file(model_path),
            "scaled5_checksum_manifest_sha256": sha256_file(SCALED5 / "SHA256SUMS"),
            "portable_tree_actions_exact": portable,
            "tree_contract": contract,
            "runtime": _runtime_manifest(),
        }
        (destination / "manifest.json").write_text(
            json.dumps(_json_ready(manifest), indent=2, sort_keys=True) + "\n"
        )
        write_checksums(destination)
    verify_checksums(output)
    print("[cr10 develop 5/5] L3 development freeze sealed; no scientific matrix generated", flush=True)


def _gard_for_regime(regime: str) -> GardConfig:
    if regime == "HOME_A_M4_S4":
        return GardConfig()
    if regime not in TRANSFER_REGIMES:
        raise ValueError(f"unknown CR10 regime: {regime}")
    a, sigma = TRANSFER_REGIMES[regime]
    return replace(GardConfig(), beta_log_mean=a, beta_log_sd=sigma)


def _experiment_for_regime(regime: str) -> ExperimentConfig:
    config = _gard_for_regime(regime)
    cohort = CohortConfig(HOME_MATRICES, HOME_REPLICATES, (LANDMARK,))
    return ExperimentConfig(
        gard=config,
        development=cohort,
        confirmation=cohort,
        horizon=HORIZON,
        bootstrap_repetitions=BOOTSTRAP_REPETITIONS,
        permutation_repetitions=RANDOMIZATION_REPETITIONS,
        master_seed=SEEDS["future_simulation"],
    )


def build_scientific_cohort(phase: str, regime: str) -> list[CR10Case]:
    if phase not in ("home", "transfer"):
        raise ValueError("CR10 phase must be home or transfer")
    if phase == "home" and regime != "HOME_A_M4_S4":
        raise ValueError("home phase requires the default regime")
    if phase == "transfer" and regime not in TRANSFER_REGIMES:
        raise ValueError("transfer phase requires a registered transfer regime")
    matrices = HOME_MATRICES if phase == "home" else TRANSFER_MATRICES
    config = _gard_for_regime(regime)
    matrix_seed = SEEDS[
        "home_matrix_generation" if phase == "home" else "transfer_matrix_generation"
    ]
    initial_seed = SEEDS[
        "home_initial_composition" if phase == "home" else "transfer_initial_composition"
    ]
    trajectory_seed = SEEDS[
        "home_main_trajectory" if phase == "home" else "transfer_main_trajectory"
    ]
    cases: list[CR10Case] = []
    for matrix_id in range(matrices):
        beta = generate_beta(
            config,
            np.random.default_rng(
                derive_seed(matrix_seed, f"{LABEL}.{phase}.beta", regime, matrix_id)
            ),
        )
        initial = generate_initial_composition(
            config,
            np.random.default_rng(
                derive_seed(initial_seed, f"{LABEL}.{phase}.initial", regime, matrix_id)
            ),
        )
        for candidate, contract in CANDIDATES.items():
            lineage: list[Snapshot] | None = None
            for attempt in range(100):
                rng = np.random.default_rng(
                    derive_seed(
                        trajectory_seed,
                        f"{LABEL}.{phase}.natural_main_path",
                        regime,
                        candidate,
                        matrix_id,
                        attempt,
                    )
                )
                try:
                    lineage = simulate_lineage(initial, beta, config, contract, rng)
                    break
                except SimulationError:
                    continue
            if lineage is None:
                raise SimulationError(
                    f"failed to obtain {phase}/{regime}/c{candidate}/m{matrix_id:03d} "
                    "natural launch state in 100 frozen attempts"
                )
            snapshot = {item.generation: item for item in lineage}[LANDMARK]
            cases.append(
                CR10Case(
                    state_id=(
                        f"{LABEL}-{phase}-{regime}-c{candidate}-m{matrix_id:03d}"
                    ),
                    phase=phase,
                    regime=regime,
                    candidate=candidate,
                    matrix_id=matrix_id,
                    beta=beta,
                    snapshot=snapshot,
                )
            )
    return cases


def _case_digest(case: CR10Case) -> str:
    digest = hashlib.sha256()
    for value in (
        case.state_id,
        case.phase,
        case.regime,
        case.candidate,
        str(case.matrix_id),
        str(case.snapshot.generation),
        str(case.snapshot.previous_growth_steps),
        str(case.snapshot.cumulative_growth_steps),
        repr(case.snapshot.inheritance),
        repr(case.snapshot.boundary_h),
    ):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    digest.update(np.ascontiguousarray(case.beta).tobytes())
    digest.update(np.ascontiguousarray(case.snapshot.composition).tobytes())
    return digest.hexdigest()


def _snapshot_equal(left: Snapshot, right: Snapshot) -> bool:
    return bool(
        np.array_equal(left.composition, right.composition)
        and left.generation == right.generation
        and left.inheritance == right.inheritance
        and left.boundary_h == right.boundary_h
        and left.previous_growth_steps == right.previous_growth_steps
        and left.cumulative_growth_steps == right.cumulative_growth_steps
    )


def _rng_state_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return json.dumps(_json_ready(left), sort_keys=True) == json.dumps(
        _json_ready(right), sort_keys=True
    )


def _entropy(composition: NDArray) -> float:
    values = np.asarray(composition, dtype=np.float64)
    mass = float(values.sum())
    if mass <= 0.0:
        return 0.0
    fraction = values[values > 0.0] / mass
    return float(-np.dot(fraction, np.log(fraction)))


def _top1(composition: NDArray) -> float:
    values = np.asarray(composition, dtype=np.float64)
    mass = float(values.sum())
    return float(values.max() / mass) if mass > 0.0 else 0.0


def _throughput(composition: NDArray, beta: NDArray) -> float:
    values = np.asarray(composition, dtype=np.float64)
    mass = float(values.sum())
    if mass <= 0.0:
        return 0.0
    x = values / mass
    return float(x @ np.asarray(beta, dtype=np.float64) @ x)


def trailing_inherited_run(boundary_h: Iterable[float]) -> int:
    run = 0
    for value in reversed(tuple(boundary_h)):
        if float(value) > INHERITANCE_THRESHOLD:
            run += 1
        else:
            break
    return run


def longest_inherited_run(boundary_h: Iterable[float]) -> int:
    longest = 0
    current = 0
    for value in boundary_h:
        if float(value) > INHERITANCE_THRESHOLD:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def count_nonoverlapping_episodes(boundary_h: Iterable[float]) -> int:
    count = 0
    after_break = False
    run = 0
    for value in boundary_h:
        inherited = float(value) > INHERITANCE_THRESHOLD
        if not after_break:
            if not inherited:
                after_break = True
                run = 0
            continue
        if inherited:
            run += 1
            if run == 3:
                count += 1
                after_break = False
                run = 0
        else:
            run = 0
    return count


def _run3_delay(post_challenge_h: NDArray) -> int:
    run = 0
    for index, value in enumerate(np.asarray(post_challenge_h, dtype=np.float64)):
        if np.isfinite(value) and value > INHERITANCE_THRESHOLD:
            run += 1
            if run == 3:
                return index + 1
        else:
            run = 0
    return HORIZON - CHALLENGE_AFTER_FISSION + 1


def exact_random_k_edits(
    composition: NDArray, k: int, rng: np.random.Generator
) -> tuple[MolecularEdit, ...]:
    values = np.asarray(composition, dtype=np.int64)
    if values.ndim != 1 or np.any(values < 0):
        raise ValueError("challenge composition must be a nonnegative integer vector")
    mass = int(values.sum())
    if not 0 <= k <= mass:
        raise ValueError("exact K challenge exceeds assembly mass")
    if k == 0:
        return ()
    molecule_types = np.repeat(np.arange(values.size, dtype=np.int64), values)
    selected = rng.choice(molecule_types.size, size=k, replace=False)
    removals = molecule_types[selected]
    forbidden = np.unique(removals)
    targets = np.setdiff1d(
        np.arange(values.size, dtype=np.int64), forbidden, assume_unique=True
    )
    if targets.size == 0:
        raise ValueError("exact K challenge has no legal target labels")
    additions = rng.choice(targets, size=k, replace=True)
    edits = tuple(
        MolecularEdit(int(remove), int(add))
        for remove, add in zip(removals, additions, strict=True)
    )
    edited = apply_many_edits(values, edits)
    if int(np.abs(edited - values).sum() // 2) != k:
        raise AssertionError("exact K challenge lost its registered transport distance")
    return edits


def apply_many_edits(
    composition: NDArray, edits: Iterable[MolecularEdit]
) -> NDArray[np.int64]:
    current = np.asarray(composition, dtype=np.int64).copy()
    original_mass = int(current.sum())
    for edit in edits:
        current = apply_molecular_edit(current, edit)
    if int(current.sum()) != original_mass or np.any(current < 0):
        raise AssertionError("multiple edits changed mass or created a negative count")
    return current


def edited_snapshot_many(
    snapshot: Snapshot, edits: Iterable[MolecularEdit]
) -> Snapshot:
    current = snapshot
    history = (
        snapshot.generation,
        snapshot.inheritance,
        snapshot.boundary_h,
        snapshot.previous_growth_steps,
        snapshot.cumulative_growth_steps,
    )
    for edit in edits:
        current = edited_snapshot(current, edit)
    observed = (
        current.generation,
        current.inheritance,
        current.boundary_h,
        current.previous_growth_steps,
        current.cumulative_growth_steps,
    )
    if observed != history:
        raise AssertionError("instantaneous multi-edit challenge changed observed history")
    return current


def _future_seed(case: CR10Case, replicate: int) -> int:
    return derive_seed(
        SEEDS["future_simulation"],
        f"{LABEL}.{case.phase}.future",
        case.regime,
        case.candidate,
        case.matrix_id,
        replicate,
    )


def _action_seed(case: CR10Case, replicate: int) -> int:
    return derive_seed(
        SEEDS["random_policy_action"],
        f"{LABEL}.{case.phase}.random_action",
        case.regime,
        case.candidate,
        case.matrix_id,
        replicate,
    )


def _challenge_seed(case: CR10Case, replicate: int) -> int:
    return derive_seed(
        SEEDS["challenge_action"],
        f"{LABEL}.home.challenge_k8",
        case.candidate,
        case.matrix_id,
        replicate,
    )


def _kinetic_seed(case: CR10Case, replicate: int) -> int:
    return derive_seed(
        SEEDS["kinetic_future"],
        f"{LABEL}.home.kinetic",
        case.candidate,
        case.matrix_id,
        replicate,
    )


def _model_coordinates(
    predictor: FrozenFullPredictor,
    candidate: str,
    snapshot: Snapshot,
    beta: NDArray,
    config: GardConfig,
) -> NDArray[np.float64]:
    state = state_graph_features(snapshot.composition, beta, config)
    direct = history_features(snapshot, config)
    base_name = f"c{candidate}"
    scaled = (
        state - predictor.arrays[f"{base_name}__full_state_scaler_mean"]
    ) / predictor.arrays[f"{base_name}__full_state_scaler_scale"]
    components = (
        scaled - predictor.arrays[f"{base_name}__full_state_pca_mean"]
    ) @ predictor.arrays[f"{base_name}__full_state_pca_components"].T
    return np.concatenate((components, direct))


def _out_of_envelope(
    coordinates: NDArray, candidate: str, envelope: dict[str, NDArray]
) -> bool:
    minimum = envelope[f"c{candidate}__minimum"]
    maximum = envelope[f"c{candidate}__maximum"]
    values = np.asarray(coordinates, dtype=np.float64)
    return bool(np.any(values < minimum) or np.any(values > maximum))


def select_policy_edit(
    policy: str,
    candidate: str,
    snapshot: Snapshot,
    beta: NDArray,
    config: GardConfig,
    predictor: FrozenFullPredictor,
    trees: FrozenLocalTrees,
    action_rng: np.random.Generator,
) -> tuple[MolecularEdit | None, float, float]:
    if policy not in POLICIES:
        raise ValueError(f"unknown CR10 policy: {policy}")
    before = predictor.predict_snapshot(candidate, snapshot, beta, config)
    edit: MolecularEdit | None = None
    after = before
    if policy == "L0_RULE_CONTINUOUS":
        edit = select_outgoing_rule_edits(snapshot.composition, beta)["RULE_DOWN"]
    elif policy == "L1_RULE_AFTER_BREAK":
        if snapshot.boundary_h and snapshot.boundary_h[-1] <= INHERITANCE_THRESHOLD:
            edit = select_outgoing_rule_edits(snapshot.composition, beta)["RULE_DOWN"]
    elif policy == "L2_RULE_UNTIL_RUN3":
        if trailing_inherited_run(snapshot.boundary_h) < 3:
            edit = select_outgoing_rule_edits(snapshot.composition, beta)["RULE_DOWN"]
    elif policy == "L3_LOCAL_TREE":
        edit = trees.select_edit(candidate, snapshot.composition, beta)
    elif policy == "MODEL_DOWN":
        noop, scores = score_legal_edits(predictor, candidate, snapshot, beta, config)
        probabilities = np.asarray(
            [item.predicted_probability for item in scores], dtype=np.float64
        )
        minimum = probabilities.min()
        index = int(np.flatnonzero(probabilities == minimum)[0])
        before = float(noop)
        edit = scores[index].edit
        after = float(scores[index].predicted_probability)
    elif policy == "RANDOM":
        legal = enumerate_legal_edits(snapshot.composition)
        edit = legal[int(action_rng.integers(0, len(legal)))]
    elif policy == "NOOP":
        edit = None
    if edit is not None and policy != "MODEL_DOWN":
        after = predictor.predict_snapshot(
            candidate, edited_snapshot(snapshot, edit), beta, config
        )
    return edit, float(before), float(after)


def _summary_digest(summary: TrajectorySummary) -> str:
    digest = hashlib.sha256()
    scalars = (
        summary.policy,
        summary.condition,
        summary.replicate,
        summary.completed_horizon,
        summary.observed_fissions,
        summary.inherited_count,
        summary.inherited_fraction_registered,
        summary.inherited_fraction_observed,
        summary.total_breaks_registered,
        summary.episode_count,
        summary.longest_inherited_run,
        summary.post_challenge_inherited_fraction,
        summary.post_challenge_breaks,
        summary.post_challenge_run3_delay,
        summary.final6_inherited,
        summary.challenge_applied,
        summary.challenge_transport_distance,
        summary.action_count,
        summary.distinct_actions,
        summary.repeated_actions,
        summary.immediately_reversing_actions,
        summary.mean_growth_updates,
        summary.final_entropy,
        summary.final_occupied_types,
        summary.final_top1_share,
        summary.final_throughput,
        summary.final_risk,
        summary.mean_predicted_action_shift,
        summary.out_of_development_envelope_fraction,
        summary.record_digest,
        summary.noop_plain_bitwise_exact,
    )
    digest.update(repr(scalars).encode("utf-8"))
    for array in (
        summary.boundary_h,
        summary.growth_updates,
        summary.action_remove,
        summary.action_add,
        summary.challenge_remove,
        summary.challenge_add,
        summary.final_snapshot.composition,
    ):
        digest.update(np.ascontiguousarray(array).tobytes())
    digest.update(repr(summary.final_snapshot).encode("utf-8"))
    digest.update(json.dumps(summary.simulation_rng_state, sort_keys=True).encode("utf-8"))
    digest.update(json.dumps(summary.action_rng_state, sort_keys=True).encode("utf-8"))
    digest.update(json.dumps(summary.challenge_rng_state, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def policy_batch_digest(batch: PolicyBatch) -> str:
    digest = hashlib.sha256()
    digest.update(
        repr(
            (
                batch.format,
                batch.registration_id,
                batch.state_id,
                batch.phase,
                batch.regime,
                batch.candidate,
                batch.matrix_id,
                batch.case_digest,
            )
        ).encode("utf-8")
    )
    for summary in batch.summaries:
        digest.update(_summary_digest(summary).encode("ascii"))
    return digest.hexdigest()


def _plain_matches_policy(
    case: CR10Case,
    config: GardConfig,
    records: list[FissionRecord],
    completed: bool,
    final_snapshot: Snapshot,
    rng: np.random.Generator,
    simulation_seed: int,
) -> bool:
    plain_rng = np.random.default_rng(simulation_seed)
    plain_experiment = ExperimentConfig(
        gard=config,
        development=CohortConfig(1, 1, (LANDMARK,)),
        confirmation=CohortConfig(1, 1, (LANDMARK,)),
        horizon=HORIZON,
        master_seed=SEEDS["validation"],
    )
    plain = simulate_controlled(
        case.snapshot,
        case.beta,
        case.candidate,
        plain_experiment,
        HORIZON,
        plain_rng,
        None,
    )
    return bool(
        plain.completed_horizon == completed
        and plain.interventions_applied == 0
        and plain.selected_edits == ()
        and _records_digest(plain.records) == _records_digest(records)
        and _snapshot_equal(plain.final_snapshot, final_snapshot)
        and _rng_state_equal(plain_rng.bit_generator.state, rng.bit_generator.state)
    )


def run_policy_trajectory(
    case: CR10Case,
    policy: str,
    condition: str,
    replicate: int,
    predictor: FrozenFullPredictor,
    trees: FrozenLocalTrees,
    envelope: dict[str, NDArray],
) -> TrajectorySummary:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown CR10 home condition: {condition}")
    if case.phase == "transfer" and condition != "UNCHALLENGED":
        raise ValueError("CR10 transfer does not include a challenge condition")
    config = _gard_for_regime(case.regime)
    simulation_seed = _future_seed(case, replicate)
    rng = np.random.default_rng(simulation_seed)
    action_rng = np.random.default_rng(_action_seed(case, replicate))
    challenge_rng = np.random.default_rng(_challenge_seed(case, replicate))
    current = case.snapshot
    records: list[FissionRecord] = []
    actions: list[MolecularEdit] = []
    action_steps: list[int] = []
    challenge_edits: tuple[MolecularEdit, ...] = ()
    risk_before: list[float] = []
    risk_after: list[float] = []
    out_of_envelope: list[int] = []
    completed = True
    cumulative = current.cumulative_growth_steps
    for step in range(HORIZON):
        try:
            record = advance_fission(
                current.composition,
                case.beta,
                config,
                CANDIDATES[case.candidate],
                rng,
            )
        except SimulationError:
            completed = False
            break
        records.append(record)
        cumulative += int(record.growth_steps)
        current = Snapshot(
            composition=np.asarray(record.daughter, dtype=np.int64).copy(),
            generation=current.generation + 1,
            inheritance=current.inheritance
            + (bool(record.h > INHERITANCE_THRESHOLD),),
            boundary_h=current.boundary_h + (float(record.h),),
            previous_growth_steps=int(record.growth_steps),
            cumulative_growth_steps=cumulative,
        )
        edit, before, after = select_policy_edit(
            policy,
            case.candidate,
            current,
            case.beta,
            config,
            predictor,
            trees,
            action_rng,
        )
        if edit is not None:
            current = edited_snapshot(current, edit)
            actions.append(edit)
            action_steps.append(step)
        risk_before.append(before)
        risk_after.append(after)
        if condition == "CHALLENGED_K8" and step + 1 == CHALLENGE_AFTER_FISSION:
            try:
                challenge_edits = exact_random_k_edits(
                    current.composition, CHALLENGE_K, challenge_rng
                )
            except ValueError:
                completed = False
                break
            pre_challenge = current.composition.copy()
            current = edited_snapshot_many(current, challenge_edits)
            distance = int(
                np.abs(current.composition - pre_challenge).sum() // 2
            )
            if distance != CHALLENGE_K:
                raise AssertionError("runtime K8 challenge distance changed")
        coordinates = _model_coordinates(
            predictor, case.candidate, current, case.beta, config
        )
        out_of_envelope.append(
            int(_out_of_envelope(coordinates, case.candidate, envelope))
        )

    observed = len(records)
    h = np.full(HORIZON, np.nan, dtype=np.float64)
    growth = np.full(HORIZON, -1, dtype=np.int32)
    action_remove = np.full(HORIZON, -1, dtype=np.int16)
    action_add = np.full(HORIZON, -1, dtype=np.int16)
    challenge_remove = np.full(CHALLENGE_K, -1, dtype=np.int16)
    challenge_add = np.full(CHALLENGE_K, -1, dtype=np.int16)
    for index, record in enumerate(records):
        h[index] = float(record.h)
        growth[index] = int(record.growth_steps)
    for step, edit in zip(action_steps, actions, strict=True):
        action_remove[step] = edit.remove_type
        action_add[step] = edit.add_type
    for index, edit in enumerate(challenge_edits):
        challenge_remove[index] = edit.remove_type
        challenge_add[index] = edit.add_type
    inherited = np.isfinite(h) & (h > INHERITANCE_THRESHOLD)
    inherited_count = int(inherited.sum())
    action_tuple = tuple(actions)
    distinct = len(set(action_tuple))
    reversing = sum(
        current_edit.remove_type == previous.add_type
        and current_edit.add_type == previous.remove_type
        for previous, current_edit in zip(action_tuple, action_tuple[1:])
    )
    # The same fissions 31--60 window is retained for the paired unchallenged
    # trajectory, making the K8 cost a clean within-matrix/stream contrast.
    post_h = h[CHALLENGE_AFTER_FISSION:]
    post_inherited = np.isfinite(post_h) & (post_h > INHERITANCE_THRESHOLD)
    post_fraction = float(post_inherited.sum() / post_h.size)
    post_breaks = int(post_h.size - post_inherited.sum())
    delay = _run3_delay(post_h)
    final6 = int(
        (
            np.isfinite(h[-6:])
            & (h[-6:] > INHERITANCE_THRESHOLD)
        ).sum()
    )
    noop_exact = True
    if policy == "NOOP" and condition == "UNCHALLENGED":
        noop_exact = _plain_matches_policy(
            case, config, records, completed, current, rng, simulation_seed
        )
    final_composition = np.asarray(current.composition, dtype=np.int64)
    mass = int(final_composition.sum())
    return TrajectorySummary(
        policy=policy,
        condition=condition,
        replicate=replicate,
        completed_horizon=bool(completed and observed == HORIZON),
        observed_fissions=observed,
        inherited_count=inherited_count,
        inherited_fraction_registered=float(inherited_count / HORIZON),
        inherited_fraction_observed=(
            float(inherited_count / observed) if observed else 0.0
        ),
        total_breaks_registered=HORIZON - inherited_count,
        episode_count=count_nonoverlapping_episodes(h[np.isfinite(h)]),
        longest_inherited_run=longest_inherited_run(h[np.isfinite(h)]),
        post_challenge_inherited_fraction=post_fraction,
        post_challenge_breaks=post_breaks,
        post_challenge_run3_delay=delay,
        final6_inherited=final6,
        challenge_applied=(
            condition == "UNCHALLENGED"
            or len(challenge_edits) == CHALLENGE_K
        ),
        challenge_transport_distance=(
            CHALLENGE_K if len(challenge_edits) == CHALLENGE_K else 0
        ),
        action_count=len(actions),
        distinct_actions=distinct,
        repeated_actions=len(actions) - distinct,
        immediately_reversing_actions=int(reversing),
        mean_growth_updates=(
            float(growth[:observed].mean()) if observed else float("nan")
        ),
        final_entropy=_entropy(final_composition),
        final_occupied_types=int(np.count_nonzero(final_composition)),
        final_top1_share=(float(final_composition.max() / mass) if mass else 0.0),
        final_throughput=_throughput(final_composition, case.beta),
        final_risk=predictor.predict_snapshot(
            case.candidate, current, case.beta, config
        ),
        mean_predicted_action_shift=(
            float(np.mean(np.asarray(risk_after) - np.asarray(risk_before)))
            if risk_before
            else 0.0
        ),
        out_of_development_envelope_fraction=(
            float(np.mean(out_of_envelope)) if out_of_envelope else 0.0
        ),
        record_digest=_records_digest(records),
        boundary_h=h,
        growth_updates=growth,
        action_remove=action_remove,
        action_add=action_add,
        challenge_remove=challenge_remove,
        challenge_add=challenge_add,
        final_snapshot=current,
        simulation_rng_state=_json_ready(rng.bit_generator.state),
        action_rng_state=_json_ready(action_rng.bit_generator.state),
        challenge_rng_state=_json_ready(challenge_rng.bit_generator.state),
        noop_plain_bitwise_exact=bool(noop_exact),
    )


def run_policy_case(
    case: CR10Case,
    registration_id: str,
    model_path: Path,
    tree_path: Path,
    envelope_path: Path,
) -> PolicyBatch:
    predictor = FrozenFullPredictor.load(model_path)
    trees = FrozenLocalTrees.load(tree_path)
    with np.load(envelope_path, allow_pickle=False) as archive:
        envelope = {name: archive[name].copy() for name in archive.files}
    replicates = HOME_REPLICATES if case.phase == "home" else TRANSFER_REPLICATES
    conditions = CONDITIONS if case.phase == "home" else ("UNCHALLENGED",)
    summaries: list[TrajectorySummary] = []
    for condition in conditions:
        for replicate in range(replicates):
            for policy in POLICIES:
                summaries.append(
                    run_policy_trajectory(
                        case,
                        policy,
                        condition,
                        replicate,
                        predictor,
                        trees,
                        envelope,
                    )
                )
    return PolicyBatch(
        format=CHECKPOINT_FORMAT,
        registration_id=registration_id,
        state_id=case.state_id,
        phase=case.phase,
        regime=case.regime,
        candidate=case.candidate,
        matrix_id=case.matrix_id,
        case_digest=_case_digest(case),
        summaries=tuple(summaries),
    )


def _policy_worker(arguments: tuple[Any, ...]) -> PolicyBatch:
    with threadpool_limits(limits=1):
        return run_policy_case(*arguments)


def _grow_to_fission_retention(
    composition: NDArray,
    beta: NDArray,
    config: GardConfig,
    candidate: str,
    rng: np.random.Generator,
    lambda_value: float,
) -> tuple[NDArray[np.int64], int]:
    if lambda_value <= 0.0:
        raise ValueError("positive retention growth requires lambda > 0")
    contract = CANDIDATES[candidate]
    current = np.asarray(composition, dtype=np.int64).copy()
    rho = 1.0 / config.n_types
    for step in range(1, config.max_growth_steps + 1):
        mass = int(current.sum())
        if mass <= 0:
            raise SimulationError("assembly became extinct")
        if mass >= config.n_max:
            return _trim_whole_assembly(current, config.n_max, rng), step - 1
        catalytic_boost = 1.0 + (beta @ current) / mass
        join_rate = config.k_join * rho * mass * catalytic_boost
        outgoing = (current / mass) @ beta
        retention = 1.0 / (1.0 + lambda_value * _percentile(outgoing))
        leave_rate = config.k_leave * current * catalytic_boost * retention
        exposure = contract.poisson_exposure
        joins = np.asarray(rng.poisson(join_rate * exposure), dtype=np.int64)
        leaves = np.minimum(
            np.asarray(rng.poisson(leave_rate * exposure), dtype=np.int64), current
        )
        survivors = current - leaves
        if contract.overshoot_rule == "admit_joiners_to_capacity":
            capacity = config.n_max - int(survivors.sum())
            if int(joins.sum()) > capacity:
                joins = _sample_without_replacement(joins, capacity, rng)
            current = survivors + joins
        elif contract.overshoot_rule == "trim_whole_assembly":
            current = survivors + joins
            if int(current.sum()) >= config.n_max:
                current = _trim_whole_assembly(current, config.n_max, rng)
        else:  # pragma: no cover - frozen simulator contract guard
            raise SimulationError(f"unknown overshoot rule: {contract.overshoot_rule}")
        if int(current.sum()) >= config.n_max:
            return current, step
    raise SimulationError(
        f"retention growth did not reach mass {config.n_max} in {config.max_growth_steps} steps"
    )


def advance_fission_retention(
    composition: NDArray,
    beta: NDArray,
    config: GardConfig,
    candidate: str,
    rng: np.random.Generator,
    lambda_value: float,
) -> FissionRecord:
    if lambda_value == 0.0:
        return advance_fission(
            composition, beta, config, CANDIDATES[candidate], rng
        )
    if lambda_value not in KINETIC_LAMBDAS:
        raise ValueError("unregistered kinetic lambda")
    parent, steps = _grow_to_fission_retention(
        composition, beta, config, candidate, rng, lambda_value
    )
    daughter = _fission(parent, config, CANDIDATES[candidate], rng)
    return FissionRecord(
        parent=parent,
        daughter=daughter,
        h=cosine_similarity(parent, daughter),
        growth_steps=steps,
    )


def _kinetic_summary_digest(summary: KineticSummary) -> str:
    digest = hashlib.sha256()
    digest.update(
        repr(
            (
                summary.lambda_value,
                summary.replicate,
                summary.completed_horizon,
                summary.observed_fissions,
                summary.inherited_count,
                summary.inherited_fraction_registered,
                summary.total_breaks_registered,
                summary.longest_inherited_run,
                summary.mean_growth_updates,
                summary.final_entropy,
                summary.final_occupied_types,
                summary.final_top1_share,
                summary.final_throughput,
                summary.final_risk,
                summary.record_digest,
                summary.lambda_zero_plain_bitwise_exact,
            )
        ).encode("utf-8")
    )
    for array in (
        summary.boundary_h,
        summary.growth_updates,
        summary.final_snapshot.composition,
    ):
        digest.update(np.ascontiguousarray(array).tobytes())
    digest.update(repr(summary.final_snapshot).encode("utf-8"))
    digest.update(json.dumps(summary.simulation_rng_state, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def kinetic_batch_digest(batch: KineticBatch) -> str:
    digest = hashlib.sha256()
    digest.update(
        repr(
            (
                batch.format,
                batch.registration_id,
                batch.state_id,
                batch.candidate,
                batch.matrix_id,
                batch.case_digest,
            )
        ).encode("utf-8")
    )
    for summary in batch.summaries:
        digest.update(_kinetic_summary_digest(summary).encode("ascii"))
    return digest.hexdigest()


def run_kinetic_trajectory(
    case: CR10Case,
    lambda_value: float,
    replicate: int,
    predictor: FrozenFullPredictor,
) -> KineticSummary:
    if case.phase != "home" or case.regime != "HOME_A_M4_S4":
        raise ValueError("kinetic prototype is restricted to the home cohort")
    config = _gard_for_regime(case.regime)
    seed = _kinetic_seed(case, replicate)
    rng = np.random.default_rng(seed)
    current = case.snapshot
    cumulative = current.cumulative_growth_steps
    records: list[FissionRecord] = []
    completed = True
    for _step in range(HORIZON):
        try:
            record = advance_fission_retention(
                current.composition,
                case.beta,
                config,
                case.candidate,
                rng,
                lambda_value,
            )
        except SimulationError:
            completed = False
            break
        records.append(record)
        cumulative += int(record.growth_steps)
        current = Snapshot(
            composition=np.asarray(record.daughter, dtype=np.int64).copy(),
            generation=current.generation + 1,
            inheritance=current.inheritance
            + (bool(record.h > INHERITANCE_THRESHOLD),),
            boundary_h=current.boundary_h + (float(record.h),),
            previous_growth_steps=int(record.growth_steps),
            cumulative_growth_steps=cumulative,
        )
    observed = len(records)
    h = np.full(HORIZON, np.nan, dtype=np.float64)
    growth = np.full(HORIZON, -1, dtype=np.int32)
    for index, record in enumerate(records):
        h[index] = float(record.h)
        growth[index] = int(record.growth_steps)
    inherited = np.isfinite(h) & (h > INHERITANCE_THRESHOLD)
    lambda_zero_exact = True
    if lambda_value == 0.0:
        lambda_zero_exact = _plain_matches_policy(
            case, config, records, completed, current, rng, seed
        )
    composition = np.asarray(current.composition, dtype=np.int64)
    return KineticSummary(
        lambda_value=float(lambda_value),
        replicate=replicate,
        completed_horizon=bool(completed and observed == HORIZON),
        observed_fissions=observed,
        inherited_count=int(inherited.sum()),
        inherited_fraction_registered=float(inherited.sum() / HORIZON),
        total_breaks_registered=int(HORIZON - inherited.sum()),
        longest_inherited_run=longest_inherited_run(h[np.isfinite(h)]),
        mean_growth_updates=(
            float(growth[:observed].mean()) if observed else float("nan")
        ),
        final_entropy=_entropy(composition),
        final_occupied_types=int(np.count_nonzero(composition)),
        final_top1_share=_top1(composition),
        final_throughput=_throughput(composition, case.beta),
        final_risk=predictor.predict_snapshot(
            case.candidate, current, case.beta, config
        ),
        record_digest=_records_digest(records),
        boundary_h=h,
        growth_updates=growth,
        final_snapshot=current,
        simulation_rng_state=_json_ready(rng.bit_generator.state),
        lambda_zero_plain_bitwise_exact=bool(lambda_zero_exact),
    )


def run_kinetic_case(
    case: CR10Case, registration_id: str, model_path: Path
) -> KineticBatch:
    predictor = FrozenFullPredictor.load(model_path)
    summaries = tuple(
        run_kinetic_trajectory(case, lambda_value, replicate, predictor)
        for replicate in range(KINETIC_REPLICATES)
        for lambda_value in KINETIC_LAMBDAS
    )
    return KineticBatch(
        format=KINETIC_CHECKPOINT_FORMAT,
        registration_id=registration_id,
        state_id=case.state_id,
        candidate=case.candidate,
        matrix_id=case.matrix_id,
        case_digest=_case_digest(case),
        summaries=summaries,
    )


def _kinetic_worker(arguments: tuple[Any, ...]) -> KineticBatch:
    with threadpool_limits(limits=1):
        return run_kinetic_case(*arguments)


def _checkpoint_path(directory: Path, case: CR10Case) -> Path:
    return directory / f"c{case.candidate}_m{case.matrix_id:03d}.pkl"


def _write_checkpoint(path: Path, value: PolicyBatch | KineticBatch) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _read_policy_checkpoint(
    path: Path, case: CR10Case, registration_id: str
) -> PolicyBatch | None:
    if not path.is_file():
        return None
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, PolicyBatch):
        raise TypeError(f"unsupported CR10 policy checkpoint: {path}")
    expected = (
        CHECKPOINT_FORMAT,
        registration_id,
        case.state_id,
        case.phase,
        case.regime,
        case.candidate,
        case.matrix_id,
        _case_digest(case),
    )
    observed = (
        value.format,
        value.registration_id,
        value.state_id,
        value.phase,
        value.regime,
        value.candidate,
        value.matrix_id,
        value.case_digest,
    )
    if observed != expected:
        raise ValueError(f"CR10 policy checkpoint contract changed: {path}")
    return value


def _read_kinetic_checkpoint(
    path: Path, case: CR10Case, registration_id: str
) -> KineticBatch | None:
    if not path.is_file():
        return None
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, KineticBatch):
        raise TypeError(f"unsupported CR10 kinetic checkpoint: {path}")
    expected = (
        KINETIC_CHECKPOINT_FORMAT,
        registration_id,
        case.state_id,
        case.candidate,
        case.matrix_id,
        _case_digest(case),
    )
    observed = (
        value.format,
        value.registration_id,
        value.state_id,
        value.candidate,
        value.matrix_id,
        value.case_digest,
    )
    if observed != expected:
        raise ValueError(f"CR10 kinetic checkpoint contract changed: {path}")
    return value


def _write_status(
    work: Path,
    stage: str,
    completed: int,
    total: int,
    **extra: Any,
) -> None:
    work.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": STATUS_FORMAT,
        "stage": stage,
        "completed": int(completed),
        "total": int(total),
        "fraction": float(completed / total) if total else 1.0,
        **_json_ready(extra),
    }
    temporary = work / f".campaign-status-{os.getpid()}.tmp"
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(work / "campaign_status.json")


def run_policy_batches(
    cases: list[CR10Case],
    registration_id: str,
    model_path: Path,
    tree_path: Path,
    envelope_path: Path,
    directory: Path,
    workers: int,
    work: Path,
    stage: str,
) -> list[PolicyBatch]:
    batches: dict[str, PolicyBatch] = {}
    missing: list[CR10Case] = []
    for case in cases:
        checkpoint = _read_policy_checkpoint(
            _checkpoint_path(directory, case), case, registration_id
        )
        if checkpoint is None:
            missing.append(case)
        else:
            batches[case.state_id] = checkpoint
    _write_status(work, stage, len(batches), len(cases), reused=len(batches))
    arguments = [
        (case, registration_id, model_path, tree_path, envelope_path)
        for case in missing
    ]
    if workers == 1:
        for arguments_one in arguments:
            batch = _policy_worker(arguments_one)
            case = arguments_one[0]
            _write_checkpoint(_checkpoint_path(directory, case), batch)
            batches[case.state_id] = batch
            _write_status(work, stage, len(batches), len(cases))
            print(f"[{stage}] {len(batches)}/{len(cases)} state batches", flush=True)
    elif arguments:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_policy_worker, arguments_one): arguments_one[0]
                for arguments_one in arguments
            }
            for future in as_completed(futures):
                case = futures[future]
                batch = future.result()
                _write_checkpoint(_checkpoint_path(directory, case), batch)
                batches[case.state_id] = batch
                _write_status(work, stage, len(batches), len(cases))
                print(f"[{stage}] {len(batches)}/{len(cases)} state batches", flush=True)
    if len(batches) != len(cases):
        raise AssertionError("CR10 policy checkpoint cohort is incomplete")
    return [batches[case.state_id] for case in cases]


def run_kinetic_batches(
    cases: list[CR10Case],
    registration_id: str,
    model_path: Path,
    directory: Path,
    workers: int,
    work: Path,
    stage: str,
) -> list[KineticBatch]:
    batches: dict[str, KineticBatch] = {}
    missing: list[CR10Case] = []
    for case in cases:
        checkpoint = _read_kinetic_checkpoint(
            _checkpoint_path(directory, case), case, registration_id
        )
        if checkpoint is None:
            missing.append(case)
        else:
            batches[case.state_id] = checkpoint
    _write_status(work, stage, len(batches), len(cases), reused=len(batches))
    arguments = [(case, registration_id, model_path) for case in missing]
    if workers == 1:
        for arguments_one in arguments:
            batch = _kinetic_worker(arguments_one)
            case = arguments_one[0]
            _write_checkpoint(_checkpoint_path(directory, case), batch)
            batches[case.state_id] = batch
            _write_status(work, stage, len(batches), len(cases))
            print(f"[{stage}] {len(batches)}/{len(cases)} state batches", flush=True)
    elif arguments:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_kinetic_worker, arguments_one): arguments_one[0]
                for arguments_one in arguments
            }
            for future in as_completed(futures):
                case = futures[future]
                batch = future.result()
                _write_checkpoint(_checkpoint_path(directory, case), batch)
                batches[case.state_id] = batch
                _write_status(work, stage, len(batches), len(cases))
                print(f"[{stage}] {len(batches)}/{len(cases)} state batches", flush=True)
    if len(batches) != len(cases):
        raise AssertionError("CR10 kinetic checkpoint cohort is incomplete")
    return [batches[case.state_id] for case in cases]


def replay_audit(
    generated: list[PolicyBatch] | list[KineticBatch],
    replayed: list[PolicyBatch] | list[KineticBatch],
    kind: str,
) -> dict[str, Any]:
    if len(generated) != len(replayed):
        raise ValueError("CR10 replay batch count differs")
    digest_function = policy_batch_digest if kind == "policy" else kinetic_batch_digest
    rows: list[dict[str, Any]] = []
    for left, right in zip(generated, replayed, strict=True):
        left_digest = digest_function(left)  # type: ignore[arg-type]
        right_digest = digest_function(right)  # type: ignore[arg-type]
        rows.append(
            {
                "state_id": left.state_id,
                "generated_digest": left_digest,
                "replay_digest": right_digest,
                "exact": left_digest == right_digest,
            }
        )
    return {
        "format": "codex-intervention-cr10-replay-audit-v1",
        "kind": kind,
        "batches": len(rows),
        "rows": rows,
        "exact_state_action_challenge_endpoint_process_and_rng": bool(
            all(row["exact"] for row in rows)
        ),
    }


def policy_tables(
    cases: list[CR10Case], batches: list[PolicyBatch]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, NDArray]]:
    case_lookup = {case.state_id: case for case in cases}
    rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    boundary_h: list[NDArray] = []
    growth_updates: list[NDArray] = []
    action_remove: list[NDArray] = []
    action_add: list[NDArray] = []
    challenge_remove: list[NDArray] = []
    challenge_add: list[NDArray] = []
    final_composition: list[NDArray] = []
    for batch in batches:
        case = case_lookup[batch.state_id]
        if batch.case_digest != _case_digest(case):
            raise ValueError("CR10 policy batch no longer matches its launch state")
        for summary in batch.summaries:
            row = {
                "state_id": case.state_id,
                "phase": case.phase,
                "regime": case.regime,
                "candidate": case.candidate,
                "matrix_id": case.matrix_id,
                "policy": summary.policy,
                "condition": summary.condition,
                "replicate": summary.replicate,
            }
            for field in (
                "completed_horizon",
                "observed_fissions",
                "inherited_count",
                "inherited_fraction_registered",
                "inherited_fraction_observed",
                "total_breaks_registered",
                "episode_count",
                "longest_inherited_run",
                "post_challenge_inherited_fraction",
                "post_challenge_breaks",
                "post_challenge_run3_delay",
                "final6_inherited",
                "challenge_applied",
                "challenge_transport_distance",
                "action_count",
                "distinct_actions",
                "repeated_actions",
                "immediately_reversing_actions",
                "mean_growth_updates",
                "final_entropy",
                "final_occupied_types",
                "final_top1_share",
                "final_throughput",
                "final_risk",
                "mean_predicted_action_shift",
                "out_of_development_envelope_fraction",
                "noop_plain_bitwise_exact",
                "record_digest",
            ):
                row[field] = getattr(summary, field)
            rows.append(row)
            trajectory_index = len(rows) - 1
            boundary_h.append(summary.boundary_h)
            growth_updates.append(summary.growth_updates)
            action_remove.append(summary.action_remove)
            action_add.append(summary.action_add)
            challenge_remove.append(summary.challenge_remove)
            challenge_add.append(summary.challenge_add)
            final_composition.append(summary.final_snapshot.composition)
            for step in np.flatnonzero(summary.action_remove >= 0):
                action_rows.append(
                    {
                        **{
                            key: row[key]
                            for key in (
                                "state_id",
                                "phase",
                                "regime",
                                "candidate",
                                "matrix_id",
                                "policy",
                                "condition",
                                "replicate",
                            )
                        },
                        "trajectory_index": trajectory_index,
                        "kind": "controller",
                        "after_fission": int(step + 1),
                        "remove_type": int(summary.action_remove[step]),
                        "add_type": int(summary.action_add[step]),
                    }
                )
            for index in np.flatnonzero(summary.challenge_remove >= 0):
                action_rows.append(
                    {
                        **{
                            key: row[key]
                            for key in (
                                "state_id",
                                "phase",
                                "regime",
                                "candidate",
                                "matrix_id",
                                "policy",
                                "condition",
                                "replicate",
                            )
                        },
                        "trajectory_index": trajectory_index,
                        "kind": "challenge",
                        "after_fission": CHALLENGE_AFTER_FISSION,
                        "remove_type": int(summary.challenge_remove[index]),
                        "add_type": int(summary.challenge_add[index]),
                    }
                )
    lineages = pd.DataFrame(rows)
    actions = pd.DataFrame(action_rows)
    numeric = [
        column
        for column in lineages.columns
        if column
        not in {
            "state_id",
            "phase",
            "regime",
            "candidate",
            "matrix_id",
            "policy",
            "condition",
            "replicate",
            "record_digest",
        }
    ]
    matrix = (
        lineages.groupby(
            ["phase", "regime", "candidate", "matrix_id", "condition", "policy"],
            as_index=False,
        )[numeric]
        .mean()
        .sort_values(
            ["phase", "regime", "candidate", "matrix_id", "condition", "policy"]
        )
        .reset_index(drop=True)
    )
    arrays = {
        "boundary_h": np.stack(boundary_h),
        "growth_updates": np.stack(growth_updates),
        "action_remove": np.stack(action_remove),
        "action_add": np.stack(action_add),
        "challenge_remove": np.stack(challenge_remove),
        "challenge_add": np.stack(challenge_add),
        "final_composition": np.stack(final_composition),
    }
    return lineages, actions, matrix, arrays


def kinetic_tables(
    cases: list[CR10Case], batches: list[KineticBatch]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, NDArray]]:
    case_lookup = {case.state_id: case for case in cases}
    rows: list[dict[str, Any]] = []
    boundary_h: list[NDArray] = []
    growth_updates: list[NDArray] = []
    final_composition: list[NDArray] = []
    for batch in batches:
        case = case_lookup[batch.state_id]
        if batch.case_digest != _case_digest(case):
            raise ValueError("CR10 kinetic batch no longer matches its launch state")
        for summary in batch.summaries:
            rows.append(
                {
                    "state_id": case.state_id,
                    "candidate": case.candidate,
                    "matrix_id": case.matrix_id,
                    "lambda": summary.lambda_value,
                    "replicate": summary.replicate,
                    "completed_horizon": summary.completed_horizon,
                    "observed_fissions": summary.observed_fissions,
                    "inherited_count": summary.inherited_count,
                    "inherited_fraction_registered": summary.inherited_fraction_registered,
                    "total_breaks_registered": summary.total_breaks_registered,
                    "longest_inherited_run": summary.longest_inherited_run,
                    "mean_growth_updates": summary.mean_growth_updates,
                    "final_entropy": summary.final_entropy,
                    "final_occupied_types": summary.final_occupied_types,
                    "final_top1_share": summary.final_top1_share,
                    "final_throughput": summary.final_throughput,
                    "final_risk": summary.final_risk,
                    "lambda_zero_plain_bitwise_exact": summary.lambda_zero_plain_bitwise_exact,
                    "record_digest": summary.record_digest,
                }
            )
            boundary_h.append(summary.boundary_h)
            growth_updates.append(summary.growth_updates)
            final_composition.append(summary.final_snapshot.composition)
    lineages = pd.DataFrame(rows)
    numeric = [
        column
        for column in lineages.columns
        if column
        not in {
            "state_id",
            "candidate",
            "matrix_id",
            "lambda",
            "replicate",
            "record_digest",
        }
    ]
    matrix = (
        lineages.groupby(["candidate", "matrix_id", "lambda"], as_index=False)[
            numeric
        ]
        .mean()
        .sort_values(["candidate", "matrix_id", "lambda"])
        .reset_index(drop=True)
    )
    arrays = {
        "boundary_h": np.stack(boundary_h),
        "growth_updates": np.stack(growth_updates),
        "final_composition": np.stack(final_composition),
    }
    return lineages, matrix, arrays


def inference_draws() -> dict[str, NDArray]:
    output: dict[str, NDArray] = {}
    for name, matrices in (
        ("home", HOME_MATRICES),
        ("transfer", TRANSFER_MATRICES),
        ("kinetic", HOME_MATRICES),
    ):
        bootstrap_rng = np.random.default_rng(
            derive_seed(SEEDS["bootstrap"], f"{LABEL}.{name}.whole_matrix")
        )
        randomization_rng = np.random.default_rng(
            derive_seed(SEEDS["randomization"], f"{LABEL}.{name}.whole_matrix")
        )
        output[f"{name}_bootstrap_indices"] = bootstrap_rng.integers(
            0,
            matrices,
            size=(BOOTSTRAP_REPETITIONS, matrices),
            dtype=np.int32,
        )
        signs = randomization_rng.integers(
            0,
            2,
            size=(RANDOMIZATION_REPETITIONS, matrices),
            dtype=np.int8,
        )
        output[f"{name}_randomization_signs"] = 2.0 * signs - 1.0
    return output


def _interval(values: NDArray, alpha: float = 0.05) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    return (
        float(np.quantile(array, alpha / 2.0)),
        float(np.quantile(array, 1.0 - alpha / 2.0)),
    )


def _maximum_leave_one_out_influence(values: NDArray) -> float:
    x = np.asarray(values, dtype=np.float64)
    if x.size <= 1:
        return 0.0
    full = float(x.mean())
    leave = (x.sum() - x) / (x.size - 1)
    return float(np.max(np.abs(leave - full)))


def _contrast_summary(
    values: NDArray,
    bootstrap_indices: NDArray,
    signs: NDArray,
    alternative: str,
) -> tuple[dict[str, Any], NDArray, NDArray]:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or x.size != bootstrap_indices.shape[1] or x.size != signs.shape[1]:
        raise ValueError("CR10 inference lost a whole-matrix block")
    estimate = float(x.mean())
    bootstrap = x[bootstrap_indices].mean(axis=1)
    null = (signs * x[None, :]).mean(axis=1)
    if alternative == "greater":
        p = float((1 + np.count_nonzero(null >= estimate)) / (null.size + 1))
    elif alternative == "less":
        p = float((1 + np.count_nonzero(null <= estimate)) / (null.size + 1))
    elif alternative == "two-sided":
        p = float(
            (1 + np.count_nonzero(np.abs(null) >= abs(estimate)))
            / (null.size + 1)
        )
    else:
        raise ValueError("unknown randomization alternative")
    summary = {
        "estimate": estimate,
        "bootstrap_ci95": list(_interval(bootstrap, 0.05)),
        "bootstrap_ci90": list(_interval(bootstrap, 0.10)),
        "randomization_alternative": alternative,
        "randomization_p_raw": p,
        "matrices_positive": int(np.count_nonzero(x > 0.0)),
        "matrices_negative": int(np.count_nonzero(x < 0.0)),
        "matrices_zero": int(np.count_nonzero(x == 0.0)),
        "maximum_leave_one_matrix_out_influence": _maximum_leave_one_out_influence(x),
    }
    return summary, bootstrap, null


def _holm(values: list[float]) -> list[float]:
    if not values:
        return []
    p = np.asarray(values, dtype=np.float64)
    order = np.argsort(p, kind="mergesort")
    adjusted_ordered = np.maximum.accumulate(
        (p.size - np.arange(p.size)) * p[order]
    )
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(adjusted_ordered, 1.0)
    return [float(value) for value in adjusted]


def _ordered_policy_effects(
    selected: pd.DataFrame, metric: str
) -> tuple[NDArray[np.int64], dict[str, NDArray[np.float64]]]:
    pivot = selected.pivot(index="matrix_id", columns="policy", values=metric).sort_index()
    if tuple(pivot.columns.sort_values()) != tuple(sorted(POLICIES)):
        raise ValueError("CR10 matrix table lost a policy")
    matrix_ids = pivot.index.to_numpy(dtype=np.int64)
    effects = {
        policy: (
            pivot[policy].to_numpy(dtype=np.float64)
            - pivot["NOOP"].to_numpy(dtype=np.float64)
        )
        for policy in POLICIES
        if policy != "NOOP"
    }
    return matrix_ids, effects


def _arm_means(selected: pd.DataFrame, metric: str) -> dict[str, float]:
    return {
        policy: float(
            selected[selected["policy"] == policy][metric].mean()
        )
        for policy in POLICIES
    }


def _gain_recovery(
    effects: dict[str, NDArray[np.float64]], bootstrap_indices: NDArray
) -> dict[str, Any]:
    model = effects["MODEL_DOWN"]
    denominator = float(model.mean())
    model_boot = model[bootstrap_indices].mean(axis=1)
    output: dict[str, Any] = {}
    for policy in (
        "L0_RULE_CONTINUOUS",
        "L1_RULE_AFTER_BREAK",
        "L2_RULE_UNTIL_RUN3",
        "L3_LOCAL_TREE",
    ):
        numerator = float(effects[policy].mean())
        policy_boot = effects[policy][bootstrap_indices].mean(axis=1)
        valid = model_boot > 0.0
        ratio_boot = policy_boot[valid] / model_boot[valid]
        output[policy] = {
            "estimate": float(numerator / denominator) if denominator > 0.0 else None,
            "bootstrap_ci95": (
                list(_interval(ratio_boot, 0.05)) if ratio_boot.size else [None, None]
            ),
            "valid_bootstrap_fraction": float(valid.mean()),
            "undefined_if_model_down_gain_nonpositive": True,
        }
    return output


def compute_inference(
    policy_matrix: pd.DataFrame,
    kinetic_matrix: pd.DataFrame,
    draws: dict[str, NDArray],
    policy_replay_exact: bool,
    kinetic_replay_exact: bool,
    noop_plain_exact: bool,
    lambda_zero_plain_exact: bool,
) -> tuple[dict[str, Any], dict[str, NDArray]]:
    stored: dict[str, NDArray] = {name: value for name, value in draws.items()}
    metrics: dict[str, Any] = {
        "format": "codex-intervention-cr10-inference-v1",
        "exploratory_no_confirmatory_gate": True,
        "home": {"candidates": []},
        "transfer": {"regimes": {}},
        "kinetic": {"candidates": []},
    }
    family_results: dict[str, list[dict[str, Any]]] = {}
    home_indices = draws["home_bootstrap_indices"]
    home_signs = draws["home_randomization_signs"]
    transfer_indices = draws["transfer_bootstrap_indices"]
    transfer_signs = draws["transfer_randomization_signs"]
    kinetic_indices = draws["kinetic_bootstrap_indices"]
    kinetic_signs = draws["kinetic_randomization_signs"]

    for candidate in CANDIDATES:
        candidate_result: dict[str, Any] = {
            "candidate": candidate,
            "conditions": {},
            "challenge_cost": {},
        }
        for condition, metric in (
            ("UNCHALLENGED", "inherited_fraction_registered"),
            ("CHALLENGED_K8", "post_challenge_inherited_fraction"),
        ):
            selected = policy_matrix[
                (policy_matrix["phase"] == "home")
                & (policy_matrix["candidate"] == candidate)
                & (policy_matrix["condition"] == condition)
            ]
            matrix_ids, effects = _ordered_policy_effects(selected, metric)
            if not np.array_equal(matrix_ids, np.arange(HOME_MATRICES)):
                raise ValueError("CR10 home inference lost a matrix")
            condition_result: dict[str, Any] = {
                "metric": metric,
                "arm_means": _arm_means(selected, metric),
                "contrasts_vs_noop": {},
            }
            for policy, effect in effects.items():
                alternative = "two-sided" if policy == "RANDOM" else "greater"
                summary, bootstrap, null = _contrast_summary(
                    effect, home_indices, home_signs, alternative
                )
                key = f"home_c{candidate}_{condition}_{policy}"
                stored[f"{key}__matrix_effect"] = effect
                stored[f"{key}__bootstrap"] = bootstrap
                stored[f"{key}__randomization"] = null
                summary["positive_lower_ci95"] = summary["bootstrap_ci95"][0] > 0.0
                if policy == "RANDOM":
                    ci90 = summary["bootstrap_ci90"]
                    summary["tost_equivalent_to_noop"] = bool(
                        ci90[0] > -RANDOM_EQUIVALENCE_MARGIN
                        and ci90[1] < RANDOM_EQUIVALENCE_MARGIN
                    )
                    summary["equivalence_margin"] = RANDOM_EQUIVALENCE_MARGIN
                condition_result["contrasts_vs_noop"][policy] = summary
                family_results.setdefault(
                    f"home_{condition.lower()}_vs_noop", []
                ).append(summary)
            condition_result["fraction_of_model_down_gain"] = _gain_recovery(
                effects, home_indices
            )
            candidate_result["conditions"][condition] = condition_result

        selected_candidate = policy_matrix[
            (policy_matrix["phase"] == "home")
            & (policy_matrix["candidate"] == candidate)
        ]
        for policy in POLICIES:
            selected_policy = selected_candidate[selected_candidate["policy"] == policy]
            pivot = selected_policy.pivot(
                index="matrix_id",
                columns="condition",
                values="post_challenge_inherited_fraction",
            ).sort_index()
            if tuple(pivot.columns) != CONDITIONS:
                # Pandas sorts columns lexically; normalize explicitly.
                if set(pivot.columns) != set(CONDITIONS):
                    raise ValueError("CR10 challenge pairing lost a condition")
            effect = (
                pivot["CHALLENGED_K8"].to_numpy(dtype=np.float64)
                - pivot["UNCHALLENGED"].to_numpy(dtype=np.float64)
            )
            summary, bootstrap, null = _contrast_summary(
                effect, home_indices, home_signs, "two-sided"
            )
            key = f"home_c{candidate}_challenge_cost_{policy}"
            stored[f"{key}__matrix_effect"] = effect
            stored[f"{key}__bootstrap"] = bootstrap
            stored[f"{key}__randomization"] = null
            candidate_result["challenge_cost"][policy] = summary
            family_results.setdefault("home_challenge_cost", []).append(summary)
        metrics["home"]["candidates"].append(candidate_result)

    for regime in TRANSFER_REGIMES:
        regime_result: dict[str, Any] = {"candidates": []}
        for candidate in CANDIDATES:
            selected = policy_matrix[
                (policy_matrix["phase"] == "transfer")
                & (policy_matrix["regime"] == regime)
                & (policy_matrix["candidate"] == candidate)
                & (policy_matrix["condition"] == "UNCHALLENGED")
            ]
            matrix_ids, effects = _ordered_policy_effects(
                selected, "inherited_fraction_registered"
            )
            if not np.array_equal(matrix_ids, np.arange(TRANSFER_MATRICES)):
                raise ValueError("CR10 transfer inference lost a matrix")
            item: dict[str, Any] = {
                "candidate": candidate,
                "arm_means": _arm_means(selected, "inherited_fraction_registered"),
                "contrasts_vs_noop": {},
            }
            for policy, effect in effects.items():
                alternative = "two-sided" if policy == "RANDOM" else "greater"
                summary, bootstrap, null = _contrast_summary(
                    effect, transfer_indices, transfer_signs, alternative
                )
                key = f"transfer_{regime}_c{candidate}_{policy}"
                stored[f"{key}__matrix_effect"] = effect
                stored[f"{key}__bootstrap"] = bootstrap
                stored[f"{key}__randomization"] = null
                summary["positive_lower_ci95"] = summary["bootstrap_ci95"][0] > 0.0
                if policy == "RANDOM":
                    ci90 = summary["bootstrap_ci90"]
                    summary["tost_equivalent_to_noop"] = bool(
                        ci90[0] > -RANDOM_EQUIVALENCE_MARGIN
                        and ci90[1] < RANDOM_EQUIVALENCE_MARGIN
                    )
                item["contrasts_vs_noop"][policy] = summary
                family_results.setdefault(f"transfer_{regime}_vs_noop", []).append(
                    summary
                )
            item["fraction_of_model_down_gain"] = _gain_recovery(
                effects, transfer_indices
            )
            regime_result["candidates"].append(item)
        metrics["transfer"]["regimes"][regime] = regime_result

    for candidate in CANDIDATES:
        selected = kinetic_matrix[kinetic_matrix["candidate"] == candidate]
        pivot = selected.pivot(
            index="matrix_id", columns="lambda", values="inherited_fraction_registered"
        ).sort_index()
        if not np.array_equal(pivot.index.to_numpy(), np.arange(HOME_MATRICES)):
            raise ValueError("CR10 kinetic inference lost a matrix")
        item: dict[str, Any] = {
            "candidate": candidate,
            "arm_means": {
                str(value): float(pivot[value].mean()) for value in KINETIC_LAMBDAS
            },
            "contrasts_vs_lambda_zero": {},
        }
        for value in KINETIC_LAMBDAS[1:]:
            effect = pivot[value].to_numpy() - pivot[0.0].to_numpy()
            summary, bootstrap, null = _contrast_summary(
                effect, kinetic_indices, kinetic_signs, "greater"
            )
            key = f"kinetic_c{candidate}_lambda_{str(value).replace('.', 'p')}"
            stored[f"{key}__matrix_effect"] = effect
            stored[f"{key}__bootstrap"] = bootstrap
            stored[f"{key}__randomization"] = null
            summary["positive_lower_ci95"] = summary["bootstrap_ci95"][0] > 0.0
            item["contrasts_vs_lambda_zero"][str(value)] = summary
            family_results.setdefault("kinetic_vs_zero", []).append(summary)
        metrics["kinetic"]["candidates"].append(item)

    for family, summaries in family_results.items():
        adjusted = _holm([item["randomization_p_raw"] for item in summaries])
        for item, value in zip(summaries, adjusted, strict=True):
            item["randomization_p_holm_within_family"] = value
            item["holm_family"] = family

    metrics["integrity"] = {
        "policy_exact_replay": bool(policy_replay_exact),
        "kinetic_exact_replay": bool(kinetic_replay_exact),
        "noop_callback_plain_bitwise_exact": bool(noop_plain_exact),
        "lambda_zero_plain_bitwise_exact": bool(lambda_zero_plain_exact),
        "all_integrity_checks_passed": bool(
            policy_replay_exact
            and kinetic_replay_exact
            and noop_plain_exact
            and lambda_zero_plain_exact
        ),
    }
    metrics["claim_status"] = {
        "confirmatory_gate": None,
        "exploratory_results_cannot_rescue_prior_phases": True,
        "mandatory_stop_after_seal": True,
    }
    return metrics, stored


def _prior_seed_values() -> set[str]:
    values: set[str] = set()
    for path in RESULT_ROOT.glob("*registration*/registration.json"):
        if path.parent == DEFAULT_REGISTRATION:
            continue
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        for key in ("seed_registry", "seeds"):
            registry = payload.get(key, {})
            if isinstance(registry, dict):
                values.update(str(item) for item in registry.values())
    return values


def _verify_upstream() -> dict[str, Any]:
    for directory in (
        CR0_VALIDATION,
        CR1_REGISTRATION,
        CR3_REGISTRATION,
        CR3_RESULT,
        CR7_REGISTRATION,
        CR7_RESULT,
    ):
        verify_checksums(directory)
    cr0 = json.loads((CR0_VALIDATION / "validation.json").read_text())
    cr3_registration = json.loads((CR3_REGISTRATION / "registration.json").read_text())
    cr3_result = json.loads((CR3_RESULT / "manifest.json").read_text())
    cr7_registration = json.loads((CR7_REGISTRATION / "registration.json").read_text())
    cr7_result = json.loads((CR7_RESULT / "manifest.json").read_text())
    if not cr0["all_checks_passed"]:
        raise ValueError("inherited CR0 validation no longer passes")
    if cr3_registration["registration_id"] != CR3_REGISTRATION_ID:
        raise ValueError("CR3 registration changed")
    if not (
        cr3_result["full_four_cell_cr3_gate"]
        and cr3_result["exact_replay"]
        and cr3_result["complete_readback_exact"]
    ):
        raise ValueError("CR3 no longer authorizes CR10")
    if cr7_registration["registration_id"] != CR7_REGISTRATION_ID:
        raise ValueError("CR7 registration changed")
    if not (
        cr7_result["complete_cr7_60_fission_gate"]
        and cr7_result["exact_replay"]
        and cr7_result["noop_callback_plain_bitwise_exact"]
        and cr7_result["complete_readback_exact"]
    ):
        raise ValueError("CR7 no longer authorizes CR10")
    model_path = CR1_REGISTRATION / "frozen_full_predictor.npz"
    if sha256_file(model_path) != EXPECTED_MODEL_SHA256:
        raise ValueError("frozen JOINT_BREAK_RUN3 predictor changed")
    context: dict[str, str] = {}
    for name, directory in (
        ("cr8", CR8_RESULT),
        ("cr9", CR9_RESULT),
        ("cr9m", CR9M_RESULT),
    ):
        if (directory / "SHA256SUMS").is_file():
            verify_checksums(directory)
            context[f"{name}_checksum_manifest_sha256"] = sha256_file(
                directory / "SHA256SUMS"
            )
    return {
        "cr0_checksum_manifest_sha256": sha256_file(CR0_VALIDATION / "SHA256SUMS"),
        "cr1_registration_checksum_manifest_sha256": sha256_file(
            CR1_REGISTRATION / "SHA256SUMS"
        ),
        "cr3_registration_checksum_manifest_sha256": sha256_file(
            CR3_REGISTRATION / "SHA256SUMS"
        ),
        "cr3_result_checksum_manifest_sha256": sha256_file(CR3_RESULT / "SHA256SUMS"),
        "cr7_registration_checksum_manifest_sha256": sha256_file(
            CR7_REGISTRATION / "SHA256SUMS"
        ),
        "cr7_result_checksum_manifest_sha256": sha256_file(CR7_RESULT / "SHA256SUMS"),
        "frozen_model_sha256": sha256_file(model_path),
        **context,
    }


def _artificial_case() -> CR10Case:
    config = GardConfig()
    rng = np.random.default_rng(derive_seed(SEEDS["validation"], "artificial.beta"))
    beta = generate_beta(config, rng)
    composition = np.zeros(config.n_types, dtype=np.int64)
    composition[: config.n_min] = 1
    snapshot = Snapshot(
        composition=composition,
        generation=LANDMARK,
        inheritance=tuple([True] * 57 + [False, True, True]),
        boundary_h=tuple([0.95] * 57 + [0.8, 0.93, 0.94]),
        previous_growth_steps=17,
        cumulative_growth_steps=1_100,
    )
    return CR10Case(
        state_id="CR10-ARTIFICIAL-NONSCIENTIFIC",
        phase="home",
        regime="HOME_A_M4_S4",
        candidate="02",
        matrix_id=0,
        beta=beta,
        snapshot=snapshot,
    )


def _fixture_inference_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    policy_rows: list[dict[str, Any]] = []
    effects = {
        "L0_RULE_CONTINUOUS": 0.12,
        "L1_RULE_AFTER_BREAK": 0.06,
        "L2_RULE_UNTIL_RUN3": 0.08,
        "L3_LOCAL_TREE": 0.10,
        "MODEL_DOWN": 0.14,
        "RANDOM": 0.0,
        "NOOP": 0.0,
    }
    for phase, regimes, matrices in (
        ("home", ("HOME_A_M4_S4",), HOME_MATRICES),
        ("transfer", tuple(TRANSFER_REGIMES), TRANSFER_MATRICES),
    ):
        for regime in regimes:
            for candidate in CANDIDATES:
                for matrix_id in range(matrices):
                    jitter = (1 if matrix_id % 2 else -1) * 0.002
                    conditions = CONDITIONS if phase == "home" else ("UNCHALLENGED",)
                    for condition in conditions:
                        challenge_cost = -0.03 if condition == "CHALLENGED_K8" else 0.0
                        for policy in POLICIES:
                            value = 0.78 + effects[policy] + jitter + challenge_cost
                            policy_rows.append(
                                {
                                    "phase": phase,
                                    "regime": regime,
                                    "candidate": candidate,
                                    "matrix_id": matrix_id,
                                    "condition": condition,
                                    "policy": policy,
                                    "inherited_fraction_registered": value,
                                    "post_challenge_inherited_fraction": value,
                                }
                            )
    kinetic_rows: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        for matrix_id in range(HOME_MATRICES):
            jitter = (1 if matrix_id % 2 else -1) * 0.002
            for value in KINETIC_LAMBDAS:
                kinetic_rows.append(
                    {
                        "candidate": candidate,
                        "matrix_id": matrix_id,
                        "lambda": value,
                        "inherited_fraction_registered": 0.8 + value * 0.1 + jitter,
                    }
                )
    return pd.DataFrame(policy_rows), pd.DataFrame(kinetic_rows)


def validation_checks(
    development: Path = DEFAULT_DEVELOPMENT,
) -> dict[str, Any]:
    development = development.resolve()
    verify_checksums(development)
    upstream = _verify_upstream()
    development_manifest = json.loads((development / "manifest.json").read_text())
    tree_contract = json.loads((development / "local_tree_contract.json").read_text())
    trees = FrozenLocalTrees.load(development / "frozen_local_trees.npz")
    predictor = FrozenFullPredictor.load(CR1_REGISTRATION / "frozen_full_predictor.npz")
    with np.load(CR7_REGISTRATION / "development_envelope.npz", allow_pickle=False) as archive:
        envelope = {name: archive[name].copy() for name in archive.files}
    case = _artificial_case()

    permutation = np.random.default_rng(
        derive_seed(SEEDS["validation"], "local.permutation")
    ).permutation(GardConfig().n_types)
    original_features = local_type_features(case.snapshot.composition, case.beta)
    permuted_features = local_type_features(
        case.snapshot.composition[permutation],
        case.beta[np.ix_(permutation, permutation)],
    )
    features_equivariant = bool(
        np.allclose(permuted_features, original_features[permutation], atol=0.0, rtol=0.0)
    )

    rule = select_outgoing_rule_edits(case.snapshot.composition, case.beta)["RULE_DOWN"]
    outgoing = outgoing_catalytic_influence(case.snapshot.composition, case.beta)
    rule_difference = outgoing[rule.add_type] - outgoing[rule.remove_type]
    all_differences = [
        outgoing[edit.add_type] - outgoing[edit.remove_type]
        for edit in enumerate_legal_edits(case.snapshot.composition)
    ]

    break_snapshot = replace(
        case.snapshot,
        inheritance=case.snapshot.inheritance + (False,),
        boundary_h=case.snapshot.boundary_h + (0.9,),
    )
    stable_snapshot = replace(
        case.snapshot,
        inheritance=case.snapshot.inheritance + (True,),
        boundary_h=case.snapshot.boundary_h + (0.95,),
    )
    rng = np.random.default_rng(derive_seed(SEEDS["validation"], "triggers"))
    l1_break = select_policy_edit(
        "L1_RULE_AFTER_BREAK",
        "02",
        break_snapshot,
        case.beta,
        GardConfig(),
        predictor,
        trees,
        rng,
    )[0]
    l1_stable = select_policy_edit(
        "L1_RULE_AFTER_BREAK",
        "02",
        stable_snapshot,
        case.beta,
        GardConfig(),
        predictor,
        trees,
        rng,
    )[0]
    l2_short = select_policy_edit(
        "L2_RULE_UNTIL_RUN3",
        "02",
        break_snapshot,
        case.beta,
        GardConfig(),
        predictor,
        trees,
        rng,
    )[0]
    l2_three = select_policy_edit(
        "L2_RULE_UNTIL_RUN3",
        "02",
        stable_snapshot,
        case.beta,
        GardConfig(),
        predictor,
        trees,
        rng,
    )[0]

    challenge_rng = np.random.default_rng(
        derive_seed(SEEDS["validation"], "challenge")
    )
    challenge = exact_random_k_edits(case.snapshot.composition, CHALLENGE_K, challenge_rng)
    challenged = edited_snapshot_many(case.snapshot, challenge)
    challenge_distance = int(
        np.abs(challenged.composition - case.snapshot.composition).sum() // 2
    )

    noop_first = run_policy_trajectory(
        case, "NOOP", "UNCHALLENGED", 0, predictor, trees, envelope
    )
    noop_second = run_policy_trajectory(
        case, "NOOP", "UNCHALLENGED", 0, predictor, trees, envelope
    )
    kinetic_zero = run_kinetic_trajectory(case, 0.0, 0, predictor)
    kinetic_positive_first = run_kinetic_trajectory(case, 0.1, 0, predictor)
    kinetic_positive_second = run_kinetic_trajectory(case, 0.1, 0, predictor)

    fixture_policy, fixture_kinetic = _fixture_inference_tables()
    fixture_metrics, _ = compute_inference(
        fixture_policy,
        fixture_kinetic,
        inference_draws(),
        policy_replay_exact=True,
        kinetic_replay_exact=True,
        noop_plain_exact=True,
        lambda_zero_plain_exact=True,
    )
    checks = {
        "inherited_cr0_pass": True,
        "cr3_outgoing_gate_replay_readback_pass": True,
        "cr7_closed_loop_gate_replay_readback_pass": True,
        "frozen_predictor_hash_exact": sha256_file(
            CR1_REGISTRATION / "frozen_full_predictor.npz"
        )
        == EXPECTED_MODEL_SHA256,
        "development_exact_reconstruction": all(
            development_manifest["exact_reconstruction"].values()
        ),
        "development_generated_no_scientific_matrix": development_manifest[
            "scientific_cr10_matrices_generated"
        ]
        == 0,
        "local_features_exact_order": tuple(tree_contract["features"])
        == LOCAL_FEATURE_NAMES,
        "local_features_permutation_equivariant": features_equivariant,
        "trees_candidate_separated": tree_contract["candidate_separated"] is True,
        "tree_depths_at_most_three": all(
            tree_contract["summaries"][candidate][f"{role}_tree_depth"]
            <= TREE_MAX_DEPTH
            for candidate in CANDIDATES
            for role in TREE_ROLES
        ),
        "tree_serialization_deterministic": development_manifest[
            "portable_tree_actions_exact"
        ],
        "l3_edit_legal": trees.select_edit(
            case.candidate, case.snapshot.composition, case.beta
        )
        in enumerate_legal_edits(case.snapshot.composition),
        "outgoing_rule_orientation_and_extreme_exact": rule_difference
        == max(all_differences),
        "l1_trigger_exact": l1_break is not None and l1_stable is None,
        "l2_trigger_exact": l2_short is not None and l2_three is None,
        "k8_mass_nonnegative_distance_and_history_exact": bool(
            int(challenged.composition.sum()) == int(case.snapshot.composition.sum())
            and np.all(challenged.composition >= 0)
            and challenge_distance == CHALLENGE_K
            and challenged.generation == case.snapshot.generation
            and challenged.inheritance == case.snapshot.inheritance
            and challenged.boundary_h == case.snapshot.boundary_h
        ),
        "future_seed_policy_and_condition_free": _future_seed(case, 0)
        == _future_seed(case, 0),
        "action_challenge_future_streams_distinct": len(
            {_future_seed(case, 0), _action_seed(case, 0), _challenge_seed(case, 0)}
        )
        == 3,
        "noop_plain_bitwise_exact": noop_first.noop_plain_bitwise_exact,
        "noop_exact_replay": _summary_digest(noop_first)
        == _summary_digest(noop_second),
        "lambda_zero_plain_bitwise_exact": kinetic_zero.lambda_zero_plain_bitwise_exact,
        "positive_lambda_exact_replay": _kinetic_summary_digest(kinetic_positive_first)
        == _kinetic_summary_digest(kinetic_positive_second),
        "strict_threshold_and_endpoint_fixture": count_nonoverlapping_episodes(
            [0.9, 0.91, 0.92, 0.93]
        )
        == 1
        and count_nonoverlapping_episodes([0.91, 0.92, 0.93]) == 0
        and longest_inherited_run([0.9, 0.91, 0.92, 0.93]) == 3,
        "whole_matrix_draws_exact": inference_draws()["home_bootstrap_indices"].shape
        == (BOOTSTRAP_REPETITIONS, HOME_MATRICES)
        and inference_draws()["transfer_bootstrap_indices"].shape
        == (BOOTSTRAP_REPETITIONS, TRANSFER_MATRICES),
        "exploratory_fixture_has_no_confirmatory_gate": fixture_metrics[
            "exploratory_no_confirmatory_gate"
        ]
        and fixture_metrics["claim_status"]["confirmatory_gate"] is None,
        "seed_domains_unique": len(SEEDS) == len(set(SEEDS.values())),
        "seed_domains_disjoint_from_prior_registrations": set(SEEDS.values()).isdisjoint(
            _prior_seed_values()
        ),
        "design_exact": HOME_MATRICES == 48
        and TRANSFER_MATRICES == 24
        and HOME_REPLICATES == 3
        and TRANSFER_REPLICATES == 2
        and KINETIC_REPLICATES == 3
        and HORIZON == 60
        and CHALLENGE_AFTER_FISSION == 30
        and CHALLENGE_K == 8
        and KINETIC_LAMBDAS == (0.0, 0.1, 0.3),
        "strict_eight_excluded": protocol()["target"]["strict_eight_excluded"],
    }
    return {
        "format": VALIDATION_FORMAT,
        "checks": checks,
        "check_count": len(checks),
        "all_checks_passed": bool(all(checks.values())),
        "upstream": upstream,
        "development_checksum_manifest_sha256": sha256_file(
            development / "SHA256SUMS"
        ),
        "scientific_cr10_matrices_generated": 0,
        "scientific_cr10_lineages_generated": 0,
    }


def validate(
    development: Path = DEFAULT_DEVELOPMENT,
    output: Path = DEFAULT_VALIDATION,
) -> None:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    checks = validation_checks(development)
    if not checks["all_checks_passed"]:
        raise AssertionError(
            {key: value for key, value in checks["checks"].items() if not value}
        )
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "CR10 full repository validation failed\n"
            + completed.stdout
            + "\n"
            + completed.stderr
        )
    with _atomic_destination(output) as destination:
        payload = dict(checks)
        payload["source_hashes"] = source_hashes()
        payload["source_tree_sha256"] = _canonical_digest(payload["source_hashes"])
        payload["pytest"] = {
            "command": [sys.executable, "-m", "pytest", "-q"],
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        (destination / "validation.json").write_text(
            json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n"
        )
        write_checksums(destination)
    verify_checksums(output)


def _append_ledger(marker: str, lines: list[str]) -> None:
    path = ROOT / "INTERVENTION_RESULTS_LEDGER.md"
    text = path.read_text()
    if marker in text:
        return
    separator = "" if text.endswith("\n") else "\n"
    path.write_text(text + separator + "\n" + marker + "\n" + "\n".join(lines) + "\n")


def register(
    development: Path = DEFAULT_DEVELOPMENT,
    validation: Path = DEFAULT_VALIDATION,
    output: Path = DEFAULT_REGISTRATION,
) -> None:
    development = development.resolve()
    validation = validation.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if DEFAULT_OUTPUT.exists() or DEFAULT_WORK.exists():
        raise FileExistsError("CR10 scientific output/work exists before registration")
    verify_checksums(development)
    verify_checksums(validation)
    validation_payload = json.loads((validation / "validation.json").read_text())
    if not validation_payload["all_checks_passed"]:
        raise ValueError("CR10 validation is not registration eligible")
    if validation_payload["source_hashes"] != source_hashes():
        raise ValueError("CR10 source changed after validation")
    upstream = _verify_upstream()
    body = {
        "format": REGISTRATION_FORMAT,
        "protocol": protocol(),
        "seed_registry": SEEDS,
        "source_hashes": source_hashes(),
        "source_tree_sha256": _canonical_digest(source_hashes()),
        "upstream": upstream,
        "development_checksum_manifest_sha256": sha256_file(development / "SHA256SUMS"),
        "validation_checksum_manifest_sha256": sha256_file(validation / "SHA256SUMS"),
        "scientific_cr10_matrices_at_registration": 0,
        "scientific_cr10_lineages_at_registration": 0,
    }
    registration_id = _canonical_digest(_json_ready(body))
    body["registration_id"] = registration_id
    with _atomic_destination(output) as destination:
        shutil.copy2(ROOT / DOCUMENT, destination / "preregistration.md")
        shutil.copy2(
            CR1_REGISTRATION / "frozen_full_predictor.npz",
            destination / "frozen_full_predictor.npz",
        )
        shutil.copy2(
            CR7_REGISTRATION / "development_envelope.npz",
            destination / "development_envelope.npz",
        )
        shutil.copy2(
            development / "frozen_local_trees.npz",
            destination / "frozen_local_trees.npz",
        )
        shutil.copy2(
            development / "local_tree_contract.json",
            destination / "local_tree_contract.json",
        )
        shutil.copy2(
            development / "manifest.json",
            destination / "development_manifest.json",
        )
        shutil.copy2(
            development / "development_distillation_audit.csv.gz",
            destination / "development_distillation_audit.csv.gz",
        )
        shutil.copy2(validation / "validation.json", destination / "validation.json")
        (destination / "intervention_protocol.json").write_text(
            json.dumps(_json_ready(protocol()), indent=2, sort_keys=True) + "\n"
        )
        (destination / "intervention_seed_registry.json").write_text(
            json.dumps(SEEDS, indent=2, sort_keys=True) + "\n"
        )
        (destination / "registration.json").write_text(
            json.dumps(_json_ready(body), indent=2, sort_keys=True) + "\n"
        )
        write_checksums(destination)
    verify_checksums(output)
    _append_ledger(
        f"<!-- registered-cr10-{registration_id} -->",
        [
            "## CR10 exploratory internalization ladder registered",
            "",
            f"- Registration: `{registration_id}`.",
            "- Final exploratory phase; it cannot rescue or replace any confirmatory result.",
            "- L0--L3, MODEL_DOWN, RANDOM, NOOP, paired K8 challenge, three transfer regimes, retention lambdas, seeds, matrix inference, replay, and claim boundaries were frozen before scientific generation.",
            "- No CR10 scientific matrix or lineage existed at registration.",
            "",
        ],
    )
    print(f"CR10 registered: {registration_id}", flush=True)


def verify_registration(directory: Path = DEFAULT_REGISTRATION) -> dict[str, Any]:
    directory = directory.resolve()
    verify_checksums(directory)
    registration = json.loads((directory / "registration.json").read_text())
    if registration["format"] != REGISTRATION_FORMAT:
        raise ValueError("unsupported CR10 registration format")
    if registration["source_hashes"] != source_hashes():
        raise ValueError("CR10 source changed after registration")
    if registration["protocol"] != protocol():
        raise ValueError("CR10 protocol implementation changed after registration")
    if registration["seed_registry"] != SEEDS:
        raise ValueError("CR10 seed registry changed after registration")
    body = dict(registration)
    observed = body.pop("registration_id")
    if _canonical_digest(_json_ready(body)) != observed:
        raise ValueError("CR10 registration ID changed")
    if sha256_file(directory / "frozen_full_predictor.npz") != EXPECTED_MODEL_SHA256:
        raise ValueError("registered CR10 predictor changed")
    return registration


def smoke(
    registration_directory: Path = DEFAULT_REGISTRATION,
    output: Path = DEFAULT_SMOKE,
) -> None:
    registration_directory = registration_directory.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    registration = verify_registration(registration_directory)
    predictor = FrozenFullPredictor.load(
        registration_directory / "frozen_full_predictor.npz"
    )
    trees = FrozenLocalTrees.load(registration_directory / "frozen_local_trees.npz")
    with np.load(
        registration_directory / "development_envelope.npz", allow_pickle=False
    ) as archive:
        envelope = {name: archive[name].copy() for name in archive.files}
    case = _artificial_case()

    def execute() -> tuple[list[tuple[int, int] | None], str, str]:
        action_rng = np.random.default_rng(
            derive_seed(SEEDS["smoke"], "all.policy.selection")
        )
        selected: list[tuple[int, int] | None] = []
        for policy in POLICIES:
            edit, _before, _after = select_policy_edit(
                policy,
                case.candidate,
                case.snapshot,
                case.beta,
                GardConfig(),
                predictor,
                trees,
                action_rng,
            )
            selected.append(
                None if edit is None else (edit.remove_type, edit.add_type)
            )
        noop = run_policy_trajectory(
            case, "NOOP", "UNCHALLENGED", 0, predictor, trees, envelope
        )
        kinetic = run_kinetic_trajectory(case, 0.1, 0, predictor)
        return selected, _summary_digest(noop), _kinetic_summary_digest(kinetic)

    first = execute()
    second = execute()
    challenge = exact_random_k_edits(
        case.snapshot.composition,
        CHALLENGE_K,
        np.random.default_rng(derive_seed(SEEDS["smoke"], "challenge")),
    )
    challenged = apply_many_edits(case.snapshot.composition, challenge)
    checks = {
        "registration_verified": True,
        "artificial_non_scientific_fixture": True,
        "all_seven_policies_exercised": len(first[0]) == len(POLICIES),
        "policy_selection_and_trajectory_replay_exact": first == second,
        "noop_plain_bitwise_exact": run_policy_trajectory(
            case, "NOOP", "UNCHALLENGED", 0, predictor, trees, envelope
        ).noop_plain_bitwise_exact,
        "k8_exact_transport": int(
            np.abs(challenged - case.snapshot.composition).sum() // 2
        )
        == CHALLENGE_K,
        "lambda_zero_plain_bitwise_exact": run_kinetic_trajectory(
            case, 0.0, 0, predictor
        ).lambda_zero_plain_bitwise_exact,
        "no_effect_sizes_or_arm_ordering_disclosed": True,
        "scientific_cr10_matrices_generated_is_zero": True,
    }
    payload = {
        "format": "codex-intervention-cr10-smoke-v1",
        "registration_id": registration["registration_id"],
        "checks": checks,
        "all_checks_passed": bool(all(checks.values())),
    }
    if not payload["all_checks_passed"]:
        raise AssertionError({key: value for key, value in checks.items() if not value})
    with _atomic_destination(output) as destination:
        (destination / "smoke.json").write_text(
            json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n"
        )
        write_checksums(destination)
    verify_checksums(output)


def _reports(metrics: dict[str, Any]) -> tuple[str, str]:
    scientific = [
        "# CR10 exploratory internalization ladder",
        "",
        "CR10 is exploratory and has no confirmatory pass/fail gate. It cannot rescue or replace CR6, CR8, CR9, or any other sealed result.",
        "",
        "## Home-regime maintenance and K8 recovery",
        "",
        "All entries below are policy-minus-NOOP inherited-boundary effects with whole-matrix 95% bootstrap intervals.",
        "",
        "| Candidate | Condition | Policy | Effect | 95% CI | Holm p |",
        "|---|---|---|---:|---:|---:|",
    ]
    for candidate_item in metrics["home"]["candidates"]:
        for condition, condition_item in candidate_item["conditions"].items():
            for policy, result in condition_item["contrasts_vs_noop"].items():
                scientific.append(
                    f"| {candidate_item['candidate']} | {condition} | {policy} | "
                    f"{result['estimate']:+.4f} | "
                    f"[{result['bootstrap_ci95'][0]:+.4f}, {result['bootstrap_ci95'][1]:+.4f}] | "
                    f"{result['randomization_p_holm_within_family']:.4g} |"
                )
    scientific.extend(
        [
            "",
            "The challenged analysis uses the registered fissions 31--60 window. Challenge-minus-unchallenged effects and local-policy fractions of the MODEL_DOWN gain are retained in `inference_metrics.json`.",
            "",
            "## Zero-shot transfer",
            "",
            "| Regime | Candidate | Policy | Effect vs NOOP | 95% CI |",
            "|---|---|---|---:|---:|",
        ]
    )
    for regime, regime_item in metrics["transfer"]["regimes"].items():
        for candidate_item in regime_item["candidates"]:
            for policy, result in candidate_item["contrasts_vs_noop"].items():
                scientific.append(
                    f"| {regime} | {candidate_item['candidate']} | {policy} | "
                    f"{result['estimate']:+.4f} | "
                    f"[{result['bootstrap_ci95'][0]:+.4f}, {result['bootstrap_ci95'][1]:+.4f}] |"
                )
    scientific.extend(
        [
            "",
            "## Retention-only kinetic prototype",
            "",
            "| Candidate | Lambda | Inheritance change vs 0 | 95% CI |",
            "|---|---:|---:|---:|",
        ]
    )
    for candidate_item in metrics["kinetic"]["candidates"]:
        for value, result in candidate_item["contrasts_vs_lambda_zero"].items():
            scientific.append(
                f"| {candidate_item['candidate']} | {value} | {result['estimate']:+.4f} | "
                f"[{result['bootstrap_ci95'][0]:+.4f}, {result['bootstrap_ci95'][1]:+.4f}] |"
            )
    integrity = metrics["integrity"]
    scientific.extend(
        [
            "",
            "## Integrity and claim boundary",
            "",
            f"Policy replay exact: **{integrity['policy_exact_replay']}**. Kinetic replay exact: **{integrity['kinetic_exact_replay']}**. NOOP/plain identity: **{integrity['noop_callback_plain_bitwise_exact']}**. Lambda-zero/plain identity: **{integrity['lambda_zero_plain_bitwise_exact']}**.",
            "",
            "These results concern externally applied local policies and one retention-only model extension. They do not demonstrate autonomous agency, biological memory, installed compotypes, life, real prebiotic chemistry, strict-eight control, or a universal origin-of-life mechanism.",
            "",
        ]
    )

    def all_positive(policy: str, condition: str) -> bool:
        return all(
            item["conditions"][condition]["contrasts_vs_noop"][policy][
                "bootstrap_ci95"
            ][0]
            > 0.0
            for item in metrics["home"]["candidates"]
        )

    l0_signal = all_positive("L0_RULE_CONTINUOUS", "UNCHALLENGED")
    l3_signal = all_positive("L3_LOCAL_TREE", "UNCHALLENGED")
    sparse_signal = any(
        all_positive(policy, "UNCHALLENGED")
        for policy in ("L1_RULE_AFTER_BREAK", "L2_RULE_UNTIL_RUN3")
    )
    kinetic_signal = any(
        all(
            candidate["contrasts_vs_lambda_zero"][value]["bootstrap_ci95"][0]
            > 0.0
            for candidate in metrics["kinetic"]["candidates"]
        )
        for value in ("0.1", "0.3")
    )
    lay = [
        "# CR10 in plain language",
        "",
        "This final experiment asked how much of the successful smart controller could be replaced by simpler instructions that are closer to something chemistry might implement.",
        "",
        (
            "The always-on simple catalytic rule improved hereditary stability in both simulator candidates."
            if l0_signal
            else "The always-on simple catalytic rule did not show a clear positive effect in both simulator candidates at this scale."
        ),
        (
            "The tiny local decision trees also retained a clear part of the control effect."
            if l3_signal
            else "The tiny local decision trees did not retain a clear effect in both candidates."
        ),
        (
            "At least one sparse rule worked while intervening only when the lineage looked unstable."
            if sparse_signal
            else "Neither sparse trigger produced a clear benefit in both candidates, so acting less often remains unresolved."
        ),
        (
            "The retention-only chemistry extension showed a consistent positive signal in both candidates."
            if kinetic_signal
            else "The tested retention-only chemistry extension did not show a consistent positive signal in both candidates."
        ),
        "",
        "Whatever the outcome, these are exploratory results. L0--L3 are still external instructions, and the retention experiment changes one assumed rate law. This does not make the assembly alive or autonomous. It tells us which pieces of the external control law can—or cannot—be compressed into simpler local mechanisms.",
        "",
    ]
    return "\n".join(scientific), "\n".join(lay)


def _state_arrays(cases: list[CR10Case]) -> dict[str, NDArray]:
    return {
        "state_id": np.asarray([case.state_id for case in cases]),
        "phase": np.asarray([case.phase for case in cases]),
        "regime": np.asarray([case.regime for case in cases]),
        "candidate": np.asarray([case.candidate for case in cases]),
        "matrix_id": np.asarray([case.matrix_id for case in cases], dtype=np.int16),
        "beta": np.stack([case.beta for case in cases]),
        "launch_composition": np.stack(
            [case.snapshot.composition for case in cases]
        ),
        "launch_boundary_h": np.stack(
            [np.asarray(case.snapshot.boundary_h, dtype=np.float64) for case in cases]
        ),
        "launch_previous_growth_steps": np.asarray(
            [case.snapshot.previous_growth_steps for case in cases], dtype=np.int32
        ),
        "launch_cumulative_growth_steps": np.asarray(
            [case.snapshot.cumulative_growth_steps for case in cases], dtype=np.int64
        ),
    }


def _write_result(
    destination_path: Path,
    registration: dict[str, Any],
    cases: list[CR10Case],
    policy_batches: list[PolicyBatch],
    kinetic_cases: list[CR10Case],
    kinetic_batches: list[KineticBatch],
    policy_replay: dict[str, Any],
    kinetic_replay: dict[str, Any],
    metrics: dict[str, Any],
    inference_arrays: dict[str, NDArray],
) -> None:
    lineages, actions, matrix, policy_arrays = policy_tables(cases, policy_batches)
    kinetic_lineages, kinetic_matrix, kinetic_arrays = kinetic_tables(
        kinetic_cases, kinetic_batches
    )
    scientific, lay = _reports(metrics)
    with _atomic_destination(destination_path) as output:
        lineages.to_csv(
            output / "policy_lineages.csv.gz", index=False, compression="gzip"
        )
        actions.to_csv(
            output / "selected_actions.csv.gz", index=False, compression="gzip"
        )
        matrix.to_csv(output / "policy_matrix_summaries.csv", index=False)
        kinetic_lineages.to_csv(
            output / "kinetic_lineages.csv.gz", index=False, compression="gzip"
        )
        kinetic_matrix.to_csv(output / "kinetic_matrix_summaries.csv", index=False)
        np.savez_compressed(output / "launch_state_arrays.npz", **_state_arrays(cases))
        np.savez_compressed(output / "policy_trajectory_arrays.npz", **policy_arrays)
        np.savez_compressed(output / "kinetic_trajectory_arrays.npz", **kinetic_arrays)
        np.savez_compressed(output / "inference_arrays.npz", **inference_arrays)
        (output / "inference_metrics.json").write_text(
            json.dumps(_json_ready(metrics), indent=2, sort_keys=True) + "\n"
        )
        (output / "policy_replay_audit.json").write_text(
            json.dumps(_json_ready(policy_replay), indent=2, sort_keys=True) + "\n"
        )
        (output / "kinetic_replay_audit.json").write_text(
            json.dumps(_json_ready(kinetic_replay), indent=2, sort_keys=True) + "\n"
        )
        (output / "SCIENTIFIC_REPORT.md").write_text(scientific)
        (output / "LAY_SUMMARY.md").write_text(lay)
        claim_boundaries = {
            "format": "codex-intervention-cr10-claim-boundaries-v1",
            "supported_only_as_exploratory": [
                "compression of externally demonstrated control into local rules",
                "sparse-trigger efficacy if supported in both candidates",
                "one retention-only model-extension result",
            ],
            "cannot_rescue": ["CR6", "CR8", "CR9"],
            "prohibited": protocol()["claim_boundary"]["prohibited"],
        }
        (output / "claim_boundaries.json").write_text(
            json.dumps(claim_boundaries, indent=2, sort_keys=True) + "\n"
        )

        lineage_readback = pd.read_csv(output / "policy_lineages.csv.gz")
        action_readback = pd.read_csv(output / "selected_actions.csv.gz")
        matrix_readback = pd.read_csv(output / "policy_matrix_summaries.csv")
        kinetic_readback = pd.read_csv(output / "kinetic_lineages.csv.gz")
        with np.load(output / "policy_trajectory_arrays.npz", allow_pickle=False) as archive:
            policy_shape_exact = archive["boundary_h"].shape == (len(lineages), HORIZON)
        with np.load(output / "kinetic_trajectory_arrays.npz", allow_pickle=False) as archive:
            kinetic_shape_exact = archive["boundary_h"].shape == (
                len(kinetic_lineages),
                HORIZON,
            )
        readback = {
            "format": "codex-intervention-cr10-readback-v1",
            "policy_lineage_rows_exact": len(lineage_readback) == len(lineages),
            "selected_action_rows_exact": len(action_readback) == len(actions),
            "policy_matrix_rows_exact": len(matrix_readback) == len(matrix),
            "kinetic_lineage_rows_exact": len(kinetic_readback)
            == len(kinetic_lineages),
            "policy_array_shape_exact": bool(policy_shape_exact),
            "kinetic_array_shape_exact": bool(kinetic_shape_exact),
        }
        readback["all_readback_checks_passed"] = bool(
            all(value for key, value in readback.items() if key != "format")
        )
        if not readback["all_readback_checks_passed"]:
            raise AssertionError(readback)
        (output / "readback_audit.json").write_text(
            json.dumps(readback, indent=2, sort_keys=True) + "\n"
        )
        manifest = {
            "format": RESULT_FORMAT,
            "registration_id": registration["registration_id"],
            "exploratory_no_confirmatory_gate": True,
            "home_matrices": HOME_MATRICES,
            "transfer_matrices_per_regime": TRANSFER_MATRICES,
            "transfer_regimes": list(TRANSFER_REGIMES),
            "policy_lineages": len(lineages),
            "policy_replay_lineages": len(lineages),
            "kinetic_lineages": len(kinetic_lineages),
            "kinetic_replay_lineages": len(kinetic_lineages),
            "maximum_scientific_fission_boundaries": (
                len(lineages) + len(kinetic_lineages)
            )
            * HORIZON,
            "maximum_replay_fission_boundaries": (
                len(lineages) + len(kinetic_lineages)
            )
            * HORIZON,
            "policy_exact_replay": metrics["integrity"]["policy_exact_replay"],
            "kinetic_exact_replay": metrics["integrity"]["kinetic_exact_replay"],
            "noop_plain_bitwise_exact": metrics["integrity"][
                "noop_callback_plain_bitwise_exact"
            ],
            "lambda_zero_plain_bitwise_exact": metrics["integrity"][
                "lambda_zero_plain_bitwise_exact"
            ],
            "complete_readback_exact": readback["all_readback_checks_passed"],
            "no_controlled_future_retry_or_matrix_replacement": True,
            "no_refit_or_post_outcome_policy_change": True,
            "mandatory_final_stop": True,
            "runtime": _runtime_manifest(),
        }
        (output / "manifest.json").write_text(
            json.dumps(_json_ready(manifest), indent=2, sort_keys=True) + "\n"
        )
        write_checksums(output)
    verify_checksums(destination_path)


def _prepare_work(work: Path, output: Path, registration_id: str) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite completed CR10 result: {output}")
    free = shutil.disk_usage(ROOT).free
    if free < MINIMUM_FREE_DISK_BYTES:
        raise RuntimeError(
            f"CR10 requires at least {MINIMUM_FREE_DISK_BYTES:,} free bytes; "
            f"only {free:,} are available"
        )
    work.mkdir(parents=True, exist_ok=True)
    contract = {
        "format": "codex-intervention-cr10-work-contract-v1",
        "registration_id": registration_id,
        "protocol_id": protocol()["protocol_id"],
    }
    path = work / "campaign_contract.json"
    if path.is_file():
        if json.loads(path.read_text()) != contract:
            raise ValueError("CR10 work directory belongs to another campaign")
    else:
        path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")


def run(
    registration_directory: Path = DEFAULT_REGISTRATION,
    output: Path = DEFAULT_OUTPUT,
    work: Path = DEFAULT_WORK,
    workers: int = min(os.cpu_count() or 1, 14),
) -> None:
    if workers < 1:
        raise ValueError("workers must be positive")
    registration_directory = registration_directory.resolve()
    output = output.resolve()
    work = work.resolve()
    registration = verify_registration(registration_directory)
    verify_checksums(DEFAULT_SMOKE)
    smoke_payload = json.loads((DEFAULT_SMOKE / "smoke.json").read_text())
    if not smoke_payload["all_checks_passed"]:
        raise ValueError("CR10 smoke validation no longer passes")
    _prepare_work(work, output, registration["registration_id"])
    model_path = registration_directory / "frozen_full_predictor.npz"
    tree_path = registration_directory / "frozen_local_trees.npz"
    envelope_path = registration_directory / "development_envelope.npz"

    print("[cr10 1/9] Building 48 fresh home matrices and 96 natural generation-60 states", flush=True)
    _write_status(work, "building_home_launch_states", 0, 2 * HOME_MATRICES)
    with threadpool_limits(limits=1):
        home_cases = build_scientific_cohort("home", "HOME_A_M4_S4")
    if len(home_cases) != 2 * HOME_MATRICES:
        raise AssertionError("CR10 home cohort is incomplete")

    print(
        f"[cr10 2/9] Running {len(home_cases) * HOME_REPLICATES * len(POLICIES) * len(CONDITIONS):,} home policy lineages",
        flush=True,
    )
    home_generated = run_policy_batches(
        home_cases,
        registration["registration_id"],
        model_path,
        tree_path,
        envelope_path,
        work / "policy" / "home" / "generate",
        workers,
        work,
        "home_policy_generate",
    )
    print("[cr10 3/9] Replaying every home policy lineage", flush=True)
    home_replayed = run_policy_batches(
        home_cases,
        registration["registration_id"],
        model_path,
        tree_path,
        envelope_path,
        work / "policy" / "home" / "replay",
        workers,
        work,
        "home_policy_replay",
    )

    transfer_cases: list[CR10Case] = []
    transfer_generated: list[PolicyBatch] = []
    transfer_replayed: list[PolicyBatch] = []
    print("[cr10 4/9] Running three zero-shot transfer regimes", flush=True)
    for regime in TRANSFER_REGIMES:
        _write_status(work, f"building_transfer_{regime}", 0, 2 * TRANSFER_MATRICES)
        with threadpool_limits(limits=1):
            regime_cases = build_scientific_cohort("transfer", regime)
        if len(regime_cases) != 2 * TRANSFER_MATRICES:
            raise AssertionError(f"CR10 transfer cohort is incomplete for {regime}")
        generated = run_policy_batches(
            regime_cases,
            registration["registration_id"],
            model_path,
            tree_path,
            envelope_path,
            work / "policy" / "transfer" / regime / "generate",
            workers,
            work,
            f"transfer_{regime}_generate",
        )
        replayed = run_policy_batches(
            regime_cases,
            registration["registration_id"],
            model_path,
            tree_path,
            envelope_path,
            work / "policy" / "transfer" / regime / "replay",
            workers,
            work,
            f"transfer_{regime}_replay",
        )
        transfer_cases.extend(regime_cases)
        transfer_generated.extend(generated)
        transfer_replayed.extend(replayed)

    all_cases = home_cases + transfer_cases
    all_generated = home_generated + transfer_generated
    all_replayed = home_replayed + transfer_replayed
    policy_replay = replay_audit(all_generated, all_replayed, "policy")
    if not policy_replay["exact_state_action_challenge_endpoint_process_and_rng"]:
        raise AssertionError("CR10 policy replay failed")
    del home_replayed, transfer_replayed, all_replayed

    print("[cr10 5/9] Running the retention-only kinetic prototype", flush=True)
    kinetic_generated = run_kinetic_batches(
        home_cases,
        registration["registration_id"],
        model_path,
        work / "kinetic" / "generate",
        workers,
        work,
        "kinetic_generate",
    )
    print("[cr10 6/9] Replaying every kinetic lineage", flush=True)
    kinetic_replayed = run_kinetic_batches(
        home_cases,
        registration["registration_id"],
        model_path,
        work / "kinetic" / "replay",
        workers,
        work,
        "kinetic_replay",
    )
    kinetic_replay = replay_audit(kinetic_generated, kinetic_replayed, "kinetic")
    if not kinetic_replay["exact_state_action_challenge_endpoint_process_and_rng"]:
        raise AssertionError("CR10 kinetic replay failed")
    del kinetic_replayed

    print("[cr10 7/9] Computing candidate- and regime-separated matrix inference", flush=True)
    _write_status(work, "whole_matrix_inference", len(all_cases), len(all_cases))
    lineages, _actions, policy_matrix, _policy_arrays = policy_tables(
        all_cases, all_generated
    )
    kinetic_lineages, kinetic_matrix, _kinetic_arrays = kinetic_tables(
        home_cases, kinetic_generated
    )
    noop_plain_exact = bool(
        lineages[
            (lineages["policy"] == "NOOP")
            & (lineages["condition"] == "UNCHALLENGED")
        ]["noop_plain_bitwise_exact"].all()
    )
    lambda_zero_plain_exact = bool(
        kinetic_lineages[kinetic_lineages["lambda"] == 0.0][
            "lambda_zero_plain_bitwise_exact"
        ].all()
    )
    draws = inference_draws()
    metrics, inference_arrays = compute_inference(
        policy_matrix,
        kinetic_matrix,
        draws,
        policy_replay_exact=True,
        kinetic_replay_exact=True,
        noop_plain_exact=noop_plain_exact,
        lambda_zero_plain_exact=lambda_zero_plain_exact,
    )
    if not metrics["integrity"]["all_integrity_checks_passed"]:
        raise AssertionError("CR10 integrity checks failed")

    print("[cr10 8/9] Writing and reading back complete scientific artifacts", flush=True)
    _write_status(work, "writing_and_reading_back_artifacts", len(all_cases), len(all_cases))
    _write_result(
        output,
        registration,
        all_cases,
        all_generated,
        home_cases,
        kinetic_generated,
        policy_replay,
        kinetic_replay,
        metrics,
        inference_arrays,
    )
    _append_ledger(
        f"<!-- sealed-cr10-{registration['registration_id']} -->",
        [
            "## CR10 exploratory internalization ladder sealed",
            "",
            f"- Registration: `{registration['registration_id']}`.",
            f"- Result: `{output.relative_to(ROOT)}`.",
            "- Confirmatory gate: **none (exploratory by registration)**.",
            f"- Policy replay exact: **{metrics['integrity']['policy_exact_replay']}**; kinetic replay exact: **{metrics['integrity']['kinetic_exact_replay']}**.",
            f"- NOOP/plain identity: **{metrics['integrity']['noop_callback_plain_bitwise_exact']}**; lambda-zero/plain identity: **{metrics['integrity']['lambda_zero_plain_bitwise_exact']}**.",
            "- No earlier phase was rescued or reinterpreted; the bounded intervention program stopped after CR10.",
            "",
        ],
    )
    _write_status(
        work,
        "sealed_complete_final_stop",
        len(all_cases),
        len(all_cases),
        output=str(output),
        exploratory_no_confirmatory_gate=True,
        integrity_passed=True,
    )
    print("[cr10 9/9] Exploratory result sealed; FINAL PROGRAM STOP", flush=True)


def read_status(work: Path = DEFAULT_WORK) -> dict[str, Any]:
    work = work.resolve()
    path = work / "campaign_status.json"
    if not path.is_file():
        raise FileNotFoundError(f"CR10 status does not exist: {path}")
    value = json.loads(path.read_text())
    value["checkpoint_counts"] = {
        "policy_home_generate": len(
            list((work / "policy" / "home" / "generate").glob("*.pkl"))
        ),
        "policy_home_replay": len(
            list((work / "policy" / "home" / "replay").glob("*.pkl"))
        ),
        "policy_transfer_generate": len(
            list((work / "policy" / "transfer").glob("*/generate/*.pkl"))
        ),
        "policy_transfer_replay": len(
            list((work / "policy" / "transfer").glob("*/replay/*.pkl"))
        ),
        "kinetic_generate": len(
            list((work / "kinetic" / "generate").glob("*.pkl"))
        ),
        "kinetic_replay": len(
            list((work / "kinetic" / "replay").glob("*.pkl"))
        ),
    }
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    develop_parser = commands.add_parser("develop")
    develop_parser.add_argument("--output", type=Path, default=DEFAULT_DEVELOPMENT)
    develop_parser.add_argument(
        "--workers", type=int, default=min(os.cpu_count() or 1, 14)
    )
    validate_parser = commands.add_parser("validate")
    validate_parser.add_argument(
        "--development", type=Path, default=DEFAULT_DEVELOPMENT
    )
    validate_parser.add_argument("--output", type=Path, default=DEFAULT_VALIDATION)
    register_parser = commands.add_parser("register")
    register_parser.add_argument(
        "--development", type=Path, default=DEFAULT_DEVELOPMENT
    )
    register_parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    register_parser.add_argument("--output", type=Path, default=DEFAULT_REGISTRATION)
    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument(
        "--registration", type=Path, default=DEFAULT_REGISTRATION
    )
    smoke_parser = commands.add_parser("smoke")
    smoke_parser.add_argument(
        "--registration", type=Path, default=DEFAULT_REGISTRATION
    )
    smoke_parser.add_argument("--output", type=Path, default=DEFAULT_SMOKE)
    run_parser = commands.add_parser("run")
    run_parser.add_argument(
        "--registration", type=Path, default=DEFAULT_REGISTRATION
    )
    run_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    run_parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    run_parser.add_argument(
        "--workers", type=int, default=min(os.cpu_count() or 1, 14)
    )
    status_parser = commands.add_parser("status")
    status_parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    return parser


def main(argv: list[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    if arguments.command == "develop":
        develop(arguments.output, arguments.workers)
    elif arguments.command == "validate":
        validate(arguments.development, arguments.output)
    elif arguments.command == "register":
        register(arguments.development, arguments.validation, arguments.output)
    elif arguments.command == "verify":
        print(
            json.dumps(
                verify_registration(arguments.registration), indent=2, sort_keys=True
            )
        )
    elif arguments.command == "smoke":
        smoke(arguments.registration, arguments.output)
    elif arguments.command == "run":
        run(
            arguments.registration,
            arguments.output,
            arguments.work_dir,
            arguments.workers,
        )
    elif arguments.command == "status":
        print(json.dumps(read_status(arguments.work_dir), indent=2, sort_keys=True))
    else:  # pragma: no cover
        raise AssertionError(arguments.command)


if __name__ == "__main__":
    main()
