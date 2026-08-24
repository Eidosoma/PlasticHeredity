"""PX10 Codex-only multiscale causal-information adjudication.

The program reuses the sealed PX9 simulator, break-acquisition, edit-selection,
and branch contracts under a new seed namespace.  It adds a pre-scientific
instrument calibration, a 48-matrix temporal-echo confirmation, a multiscale
PhiID atom audit, and a held-out intervention-to-renewal information channel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits

from . import intervention_cr5 as cr5
from . import phir_extension_px7 as px7
from . import phir_extension_px9 as px9
from .config import CANDIDATES, GardConfig
from .intervention_core import apply_molecular_edit, state_graph_features_many
from .mechanistic import sha256_file, verify_checksums, write_checksums
from .mechanistic_metrics import holm_adjust
from .phir_ch5 import _append_ledger
from .phir_instruments import (
    ANTICHAINS,
    ATOM_ORDER,
    ATOM_NAMES,
    PHIR_ATOMS,
    SYNERGISTIC,
    UNIQUE_0,
    UNIQUE_1,
    _atom_leq,
    atom_name,
    local_phi_id_atoms,
)
from .phir_rescue_instruments import beta_physical_partition
from .seeds import derive_seed
from .simulator import generate_beta, generate_initial_composition


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results" / "phir_extension"
DOCUMENT = "CODEX_CH5_PHIR_PX10_PREREGISTRATION.md"
LEDGER = ROOT / "PHIR_RESULTS_LEDGER.md"

DEFAULT_VALIDATION = RESULT_ROOT / "px10_validation"
DEFAULT_CALIBRATION = RESULT_ROOT / "px10_calibration"
DEFAULT_REGISTRATION = RESULT_ROOT / "px10_registration"
DEFAULT_SMOKE = RESULT_ROOT / "px10_smoke"
DEFAULT_OUTPUT = RESULT_ROOT / "px10_multiscale_causal_information48"
DEFAULT_LOG = RESULT_ROOT / "px10_multiscale_causal_information48.log"
EXTERNAL_WORK = Path(
    "/mnt/bioIce1/PlasticHeredityArchivedWorkfiles/"
    "replicators.13.8.2026.codex/px10_multiscale_causal_information_work"
)
DEFAULT_WORK = EXTERNAL_WORK

MODEL_SOURCE = px9.MODEL_SOURCE
MODEL_CONTRACT_SOURCE = px9.MODEL_CONTRACT_SOURCE
DEVELOPMENT_ARRAYS = (
    ROOT
    / "results_intervention_replication"
    / "cr5_development_freeze"
    / "development_arrays.npz"
)
DEVELOPMENT_STATES = (
    ROOT
    / "results_intervention_replication"
    / "cr5_development_freeze"
    / "renewal_development_states.csv"
)

PROGRAM_FORMAT = "codex-ch5-phir-px10-multiscale-causal-information-v1"
REGISTRATION_FORMAT = "codex-ch5-phir-px10-registration-v1"
RESULT_FORMAT = "codex-ch5-phir-px10-result-v1"
CHECKPOINT_FORMAT = "codex-ch5-phir-px10-checkpoint-v1"
STATUS_FORMAT = "codex-ch5-phir-px10-status-v1"
LABEL = "CODEX_CH5_PHIR_PX10_MULTISCALE_CAUSAL_INFORMATION_V1"

MATRICES = 48
LANDMARKS = px9.LANDMARKS
BRANCHES = 256
HALVES = px9.HALVES
HORIZON = 8
ACQUISITION_LIMIT = 60
MINIMUM_ELIGIBLE_MATRICES = 40
BOOTSTRAP_DRAWS = 4096
RANDOMIZATION_DRAWS = 4096
MAX_WORKERS = 8
MAX_CPU_HOURS = 30.0
MINIMUM_FREE_DISK_BYTES = 800_000_000
OUTCOME_EQUIVALENCE_MARGIN = 0.025
INFORMATION_EQUIVALENCE_MARGIN_BITS = 0.0005
CALIBRATION_NULL_MARGIN_BITS = 0.01

QUANTILE_ARMS = px9.QUANTILE_ARMS
ARMS = px9.ARMS
ATOM_ARMS = ("Q00", "Q100")
ATOM_SUPPORT_BRANCHES = 64
PRIMARY_LAGS = (1, 2, 3, 4)
SECONDARY_LAGS = (8,)
ALL_LAGS = (*PRIMARY_LAGS, *SECONDARY_LAGS)
TEMPORAL_SHIFTS = px9.TEMPORAL_SHIFTS
RANDOM_PARTITIONS = 8
ATOM_GROUPS = ("causation", "synergy_persistence", "emergence")
ATOM_DIRECTIONS = {
    "causation": 1.0,
    "synergy_persistence": -1.0,
    "emergence": 1.0,
}

SOURCE_FILES = (
    DOCUMENT,
    "plastic_heredity/phir_extension_px10.py",
    "tests/test_phir_extension_px10.py",
    "plastic_heredity/phir_extension_px9.py",
    "plastic_heredity/phir_extension_px7.py",
    "plastic_heredity/phir_instruments.py",
    "plastic_heredity/phir_rescue_instruments.py",
    "plastic_heredity/intervention_cr5.py",
    "plastic_heredity/intervention_core.py",
    "plastic_heredity/simulator.py",
    "plastic_heredity/config.py",
    "plastic_heredity/seeds.py",
    "pyproject.toml",
    "requirements-lock.txt",
)


def _seed_domain(name: str) -> str:
    return hashlib.sha256(f"{LABEL}::{name}".encode()).hexdigest()


SEED_DOMAINS = {
    name: _seed_domain(name)
    for name in (
        "matrix",
        "initial",
        "main_path",
        "acquisition",
        "random_action",
        "future",
        "partition",
        "temporal_shuffle",
        "calibration",
        "bootstrap",
        "randomization",
        "macro_kernel",
        "replay",
        "validation",
        "smoke",
    )
}


def _json_ready(value: Any) -> Any:
    return px7._json_ready(value)


def _digest(value: Any) -> str:
    return px7._digest(value)


def _array_digest(*arrays: NDArray) -> str:
    return px7._array_digest(*arrays)


def _atomic_json(path: Path, value: Any) -> None:
    px7._atomic_json(path, value)


def _atomic_pickle(path: Path, value: Any) -> None:
    px7._atomic_pickle(path, value)


def _source_hashes() -> dict[str, str]:
    return {name: sha256_file(ROOT / name) for name in SOURCE_FILES}


@dataclass(frozen=True)
class PX10Spec:
    label: str
    matrices: int
    landmarks: tuple[int, ...]
    branches: int
    horizon: int
    acquisition_limit: int
    bootstrap_draws: int
    randomization_draws: int

    def as_px9(self) -> px9.PX9Spec:
        return px9.PX9Spec(
            self.label,
            self.matrices,
            self.landmarks,
            self.branches,
            self.horizon,
            self.acquisition_limit,
            self.bootstrap_draws,
            self.randomization_draws,
        )


def scientific_spec() -> PX10Spec:
    return PX10Spec(
        "scientific",
        MATRICES,
        LANDMARKS,
        BRANCHES,
        HORIZON,
        ACQUISITION_LIMIT,
        BOOTSTRAP_DRAWS,
        RANDOMIZATION_DRAWS,
    )


def smoke_spec() -> PX10Spec:
    return PX10Spec("smoke", 1, (20,), 16, 4, 60, 32, 32)


def _seed(spec: PX10Spec, domain: str, *keys: object) -> int:
    selected = "smoke" if spec.label == "smoke" else domain
    return derive_seed(SEED_DOMAINS[selected], LABEL, spec.label, domain, *keys)


@contextmanager
def _px9_seed_context() -> Iterable[None]:
    """Run sealed PX9 primitives under the independently sealed PX10 streams."""

    old_label = px9.LABEL
    old_domains = px9.SEED_DOMAINS
    old_matrices = px9.MATRICES
    try:
        px9.LABEL = LABEL
        px9.SEED_DOMAINS = SEED_DOMAINS
        px9.MATRICES = MATRICES
        yield
    finally:
        px9.LABEL = old_label
        px9.SEED_DOMAINS = old_domains
        px9.MATRICES = old_matrices


def protocol(spec: PX10Spec | None = None) -> dict[str, Any]:
    spec = scientific_spec() if spec is None else spec
    value: dict[str, Any] = {
        "format": PROGRAM_FORMAT,
        "question": "Codex-only multiscale and causal identity of the PX9 echo",
        "predecessors_immutable": True,
        "fable_artifacts_used": False,
        "cohort": {
            "matrices": spec.matrices,
            "candidates": list(CANDIDATES),
            "landmarks": list(spec.landmarks),
            "branches_per_arm": spec.branches,
            "halves": {key: list(value) for key, value in _halves(spec).items()},
            "horizon": spec.horizon,
            "acquisition_limit": spec.acquisition_limit,
            "minimum_eligible_matrices": MINIMUM_ELIGIBLE_MATRICES,
            "replacement": False,
        },
        "arms": {
            "all": list(ARMS),
            "uniform_channel": list(QUANTILE_ARMS),
            "atom_scoring": list(ATOM_ARMS),
            "quantiles": list(px9.QUANTILES),
            "exhaustive_scoring": True,
        },
        "temporal_echo": {
            "frozen_px9_target": True,
            "support_branches": 128,
            "derangements": list(TEMPORAL_SHIFTS),
            "incremental_equivalence_bits": INFORMATION_EQUIVALENCE_MARGIN_BITS,
        },
        "atoms": {
            "groups": list(ATOM_GROUPS),
            "directions": ATOM_DIRECTIONS,
            "arms": list(ATOM_ARMS),
            "branches_per_half": ATOM_SUPPORT_BRANCHES,
            "primary_lags": list(PRIMARY_LAGS),
            "secondary_lags": list(SECONDARY_LAGS),
            "matrix_pooled_landmarks": True,
            "random_partitions": RANDOM_PARTITIONS,
            "calibration_gated": True,
        },
        "intervention_channel": {
            "input": "uniform six registered edit-dose arms",
            "output": "renewal run3 within F8",
            "source_target_halves": True,
            "jeffreys_smoothing": 0.5,
            "equivalence_bits": INFORMATION_EQUIVALENCE_MARGIN_BITS,
            "label_derangements": 16,
        },
        "macro_micro": {
            "exploratory_only": True,
            "development_only_transform": True,
            "micro_bits": 4,
            "macro_bits": 2,
            "lags": [1, 3],
        },
        "inference": {
            "unit": "whole catalytic matrix",
            "bootstrap_draws": spec.bootstrap_draws,
            "randomization_draws": spec.randomization_draws,
            "holm_within_named_families": True,
            "outcome_equivalence_margin": OUTCOME_EQUIVALENCE_MARGIN,
        },
        "randomness": {
            "domains": SEED_DOMAINS,
            "arm_in_future_seed": False,
            "common_random_streams": True,
            "random_action_separate": True,
        },
        "operational": {
            "workers_max": MAX_WORKERS,
            "cpu_hours_max": MAX_CPU_HOURS,
            "detached_science": True,
            "matrix_checkpointing": True,
            "complete_replay": True,
            "automatic_continuation": False,
        },
    }
    value["protocol_id"] = _digest(value)
    return value


def _halves(spec: PX10Spec) -> dict[str, tuple[int, ...]]:
    midpoint = spec.branches // 2
    return {"A": tuple(range(midpoint)), "B": tuple(range(midpoint, spec.branches))}


@dataclass(frozen=True)
class GrainModel:
    means: dict[str, NDArray[np.float64]]
    scales: dict[str, NDArray[np.float64]]
    components: dict[str, NDArray[np.float64]]
    medians: dict[str, NDArray[np.float64]]

    def classify(self, candidate: str, features: NDArray) -> tuple[NDArray, NDArray]:
        values = np.atleast_2d(np.asarray(features, dtype=np.float64))
        scores = ((values - self.means[candidate]) / self.scales[candidate]) @ self.components[candidate].T
        bits = scores > self.medians[candidate]
        micro = (
            bits[:, 0].astype(np.int16) * 8
            + bits[:, 1].astype(np.int16) * 4
            + bits[:, 2].astype(np.int16) * 2
            + bits[:, 3].astype(np.int16)
        )
        macro = bits[:, 0].astype(np.int16) * 2 + bits[:, 1].astype(np.int16)
        return micro, macro


def fit_grain_model() -> GrainModel:
    with np.load(DEVELOPMENT_ARRAYS, allow_pickle=False) as archive:
        features = np.asarray(archive["renewal_state_graph"], dtype=np.float64)
    states = pd.read_csv(DEVELOPMENT_STATES)
    if len(states) != len(features):
        raise ValueError("CR5 development state/features row mismatch")
    candidates = states["candidate"].astype(int).map(lambda value: f"{value:02d}")
    means: dict[str, NDArray[np.float64]] = {}
    scales: dict[str, NDArray[np.float64]] = {}
    components: dict[str, NDArray[np.float64]] = {}
    medians: dict[str, NDArray[np.float64]] = {}
    for candidate in CANDIDATES:
        local = features[candidates.to_numpy() == candidate]
        mean = local.mean(axis=0)
        scale = local.std(axis=0)
        scale[~np.isfinite(scale) | (scale <= 1e-12)] = 1.0
        standardized = (local - mean) / scale
        pca = PCA(n_components=4, svd_solver="full").fit(standardized)
        component = np.asarray(pca.components_, dtype=np.float64)
        for index in range(component.shape[0]):
            pivot = int(np.argmax(np.abs(component[index])))
            if component[index, pivot] < 0:
                component[index] *= -1.0
        score = standardized @ component.T
        means[candidate] = mean
        scales[candidate] = scale
        components[candidate] = component
        medians[candidate] = np.median(score, axis=0)
    return GrainModel(means, scales, components, medians)


def save_grain_model(model: GrainModel, path: Path) -> None:
    arrays: dict[str, NDArray] = {}
    for candidate in CANDIDATES:
        arrays[f"c{candidate}__mean"] = model.means[candidate]
        arrays[f"c{candidate}__scale"] = model.scales[candidate]
        arrays[f"c{candidate}__components"] = model.components[candidate]
        arrays[f"c{candidate}__medians"] = model.medians[candidate]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_grain_model(path: Path) -> GrainModel:
    with np.load(path, allow_pickle=False) as archive:
        return GrainModel(
            {candidate: archive[f"c{candidate}__mean"] for candidate in CANDIDATES},
            {candidate: archive[f"c{candidate}__scale"] for candidate in CANDIDATES},
            {candidate: archive[f"c{candidate}__components"] for candidate in CANDIDATES},
            {candidate: archive[f"c{candidate}__medians"] for candidate in CANDIDATES},
        )


def _entropy(values: Sequence[object]) -> float:
    _, counts = np.unique(np.asarray(values), axis=0, return_counts=True)
    probabilities = counts / counts.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))


def _mutual_information_discrete(left: NDArray, right: NDArray) -> float:
    x = np.asarray(left)
    y = np.asarray(right)
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    return _entropy(x) + _entropy(y) - _entropy(np.column_stack((x, y)))


def _discrete_local_surprisal(values: NDArray) -> NDArray[np.float64]:
    array = np.asarray(values)
    if array.ndim == 1:
        array = array[:, None]
    _, inverse, counts = np.unique(
        array, axis=0, return_inverse=True, return_counts=True
    )
    return -np.log2(counts[inverse] / len(array))


def _exact_discrete_atoms(past: NDArray, future: NDArray) -> dict[Any, float]:
    p = np.asarray(past)
    f = np.asarray(future)
    if p.shape != f.shape or p.ndim != 2 or p.shape[1] != 2:
        raise ValueError("the exact discrete PhiID reference requires matching Nx2 arrays")
    source_surprisal = {
        source: _discrete_local_surprisal(p[:, np.asarray(source, dtype=int)])
        for antichain in ANTICHAINS
        for source in antichain
    }
    target_surprisal = {
        target: _discrete_local_surprisal(f[:, np.asarray(target, dtype=int)])
        for antichain in ANTICHAINS
        for target in antichain
    }
    joint_surprisal = {
        (source, target): _discrete_local_surprisal(
            np.column_stack(
                (
                    p[:, np.asarray(source, dtype=int)],
                    f[:, np.asarray(target, dtype=int)],
                )
            )
        )
        for source in source_surprisal
        for target in target_surprisal
    }
    cumulative: dict[Any, float] = {}
    partial: dict[Any, float] = {}
    for atom in ATOM_ORDER:
        informative = np.min(
            np.vstack([source_surprisal[source] for source in atom[0]]), axis=0
        )
        conditional = np.min(
            np.vstack(
                [
                    joint_surprisal[(source, target)]
                    - target_surprisal[target]
                    for source in atom[0]
                    for target in atom[1]
                ]
            ),
            axis=0,
        )
        cumulative[atom] = float(np.mean(informative - conditional))
        lower = [
            partial[other]
            for other in partial
            if other != atom and _atom_leq(other, atom)
        ]
        partial[atom] = cumulative[atom] - float(sum(lower))
    return partial


def _canonical_samples(name: str, samples: int, rng: np.random.Generator) -> tuple[NDArray, NDArray]:
    if name == "independent":
        if samples % 16:
            raise ValueError("the exact independent fixture requires support divisible by 16")
        states = np.asarray(
            [
                (past0, past1, future0, future1)
                for past0 in (0, 1)
                for past1 in (0, 1)
                for future0 in (0, 1)
                for future1 in (0, 1)
            ],
            dtype=np.int8,
        )
        rows = np.tile(states, (samples // 16, 1))
        past = rows[:, :2]
        future = rows[:, 2:]
    elif name in {"copy", "cross_transfer"}:
        if samples % 4:
            raise ValueError("copy fixtures require support divisible by 4")
        states = np.asarray(
            [(first, second) for first in (0, 1) for second in (0, 1)],
            dtype=np.int8,
        )
        past = np.tile(states, (samples // 4, 1))
        future = past.copy() if name == "copy" else past[:, ::-1].copy()
    elif name in {"downward_xor", "parity_preserving"}:
        if samples % 8:
            raise ValueError("XOR fixtures require support divisible by 8")
        states = np.asarray(
            [
                (first, second, auxiliary)
                for first in (0, 1)
                for second in (0, 1)
                for auxiliary in (0, 1)
            ],
            dtype=np.int8,
        )
        rows = np.tile(states, (samples // 8, 1))
        past = rows[:, :2]
        parity = past[:, 0] ^ past[:, 1]
        if name == "downward_xor":
            future = np.column_stack((parity, rows[:, 2])).astype(np.int8)
        else:
            future = np.column_stack((rows[:, 2], rows[:, 2] ^ parity)).astype(
                np.int8
            )
    elif name == "common_driver":
        if samples % 2:
            raise ValueError("common-driver fixture requires even support")
        driver = np.tile(np.asarray((0, 1), dtype=np.int8), samples // 2)
        past = np.column_stack((driver, driver))
        future = past.copy()
    else:
        raise ValueError(name)
    order = rng.permutation(samples)
    return past[order], future[order]


def _dominant_atom(atoms: Mapping[Any, float]) -> str:
    atom = max(atoms, key=lambda key: abs(float(atoms[key])))
    return atom_name(atom)


CANONICAL_ATOM_SIGNATURES: dict[str, dict[str, float]] = {
    "copy": {
        "r_to_r": 1.0,
        "u0_to_u1": -1.0,
        "u1_to_u0": -1.0,
        "s_to_u0": 1.0,
        "s_to_u1": 1.0,
        "u0_to_s": 1.0,
        "u1_to_s": 1.0,
        "s_to_s": -1.0,
    },
    "cross_transfer": {
        "r_to_r": 1.0,
        "u0_to_u0": -1.0,
        "u1_to_u1": -1.0,
        "s_to_u0": 1.0,
        "s_to_u1": 1.0,
        "u0_to_s": 1.0,
        "u1_to_s": 1.0,
        "s_to_s": -1.0,
    },
    "downward_xor": {"s_to_r": 1.0, "s_to_u1": -1.0, "s_to_s": 1.0},
    "parity_preserving": {"s_to_s": 1.0},
    "common_driver": {"r_to_r": 1.0},
}


def _signature_error(name: str, atoms: Mapping[Any, float]) -> float:
    expected = CANONICAL_ATOM_SIGNATURES[name]
    return max(
        abs(float(value) - expected.get(atom_name(atom), 0.0))
        for atom, value in atoms.items()
    )


def _gaussian_mi_from_cov(covariance: NDArray, left: Sequence[int], right: Sequence[int]) -> float:
    first = np.asarray(left, dtype=int)
    second = np.asarray(right, dtype=int)
    joint = np.concatenate((first, second))
    _, ld_first = np.linalg.slogdet(covariance[np.ix_(first, first)])
    _, ld_second = np.linalg.slogdet(covariance[np.ix_(second, second)])
    sign, ld_joint = np.linalg.slogdet(covariance[np.ix_(joint, joint)])
    if sign <= 0:
        raise ValueError("non-positive analytic covariance")
    return float(0.5 * (ld_first + ld_second - ld_joint))


def _gaussian_fixture(dimensions: int = 8) -> tuple[NDArray, float, tuple[NDArray, NDArray]]:
    split = dimensions // 2
    transition = np.eye(dimensions) * 0.45
    transition[:split, split:] = 0.15
    transition[split:, :split] = -0.1125
    noise = np.eye(dimensions) * 0.8
    joint = np.block(
        [
            [np.eye(dimensions), transition.T],
            [transition, transition @ transition.T + noise],
        ]
    )
    left = np.arange(split, dtype=int)
    right = np.arange(split, dimensions, dtype=int)
    past = np.arange(dimensions, dtype=int)
    future = np.arange(dimensions, 2 * dimensions, dtype=int)
    whole = _gaussian_mi_from_cov(joint, past, future)
    aa = _gaussian_mi_from_cov(joint, left, dimensions + left)
    bb = _gaussian_mi_from_cov(joint, right, dimensions + right)
    return joint, float(whole - aa - bb), (left, right)


def _sample_wms(
    covariance: NDArray,
    partition: tuple[NDArray, NDArray],
    samples: int,
    rng: np.random.Generator,
) -> float:
    dimensions = covariance.shape[0] // 2
    data = rng.multivariate_normal(np.zeros(2 * dimensions), covariance, size=samples)
    past = data[:, :dimensions].T
    future = data[:, dimensions:].T
    try:
        whole = px9.gaussian_mutual_information(past, future)
        aa = px9.gaussian_mutual_information(past[partition[0]], future[partition[0]])
        bb = px9.gaussian_mutual_information(past[partition[1]], future[partition[1]])
        return float(whole - aa - bb)
    except (ValueError, np.linalg.LinAlgError):
        return float("nan")


def _sample_wms_from_arrays(
    past: NDArray,
    future: NDArray,
    partition: tuple[NDArray, NDArray],
) -> float:
    try:
        whole = px9.gaussian_mutual_information(past, future)
        aa = px9.gaussian_mutual_information(
            past[partition[0]], future[partition[0]]
        )
        bb = px9.gaussian_mutual_information(
            past[partition[1]], future[partition[1]]
        )
        return float(whole - aa - bb)
    except (ValueError, np.linalg.LinAlgError):
        return float("nan")


def _sample_clr_surrogate(
    samples: int, rng: np.random.Generator
) -> tuple[float, float]:
    """Score a 99-dimensional Gaussian process in the CLR subspace."""

    dimensions = GardConfig().n_types
    past_full = rng.normal(size=(dimensions, samples))
    past_full -= past_full.mean(axis=0, keepdims=True)
    noise = rng.normal(size=(dimensions, samples))
    noise -= noise.mean(axis=0, keepdims=True)
    future_full = 0.45 * past_full + 0.15 * np.roll(
        past_full, dimensions // 2, axis=0
    ) + 0.8 * noise
    future_full -= future_full.mean(axis=0, keepdims=True)
    past = past_full[:-1]
    future = future_full[:-1]
    split = past.shape[0] // 2
    partition = (
        np.arange(split, dtype=int),
        np.arange(split, past.shape[0], dtype=int),
    )
    paired = _sample_wms_from_arrays(past, future, partition)
    shuffled = _sample_wms_from_arrays(
        past, future[:, rng.permutation(samples)], partition
    )
    return paired, shuffled


def run_calibration(output: Path = DEFAULT_CALIBRATION) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    repetitions = 100
    for support in (128, 1024):
        for name, signature in CANONICAL_ATOM_SIGNATURES.items():
            matches = 0
            identity_errors: list[float] = []
            signature_errors: list[float] = []
            for repeat in range(repetitions):
                rng = np.random.default_rng(
                    _seed(scientific_spec(), "calibration", "discrete", name, support, repeat)
                )
                past, future = _canonical_samples(name, support, rng)
                atoms = _exact_discrete_atoms(past, future)
                signature_error = _signature_error(name, atoms)
                matches += int(signature_error <= 1e-10)
                signature_errors.append(signature_error)
                total = _mutual_information_discrete(past, future)
                identity_errors.append(abs(sum(atoms.values()) - total))
            rows.append(
                {
                    "family": "discrete_reference",
                    "fixture": name,
                    "support": support,
                    "expected_nonzero_signature": signature,
                    "signature_recovery_rate": matches / repetitions,
                    "maximum_signature_error": max(signature_errors),
                    "maximum_identity_error": max(identity_errors),
                }
            )

    independent_rows: list[dict[str, Any]] = []
    for support in (128, 1024):
        maximum_atoms: list[float] = []
        for repeat in range(repetitions):
            rng = np.random.default_rng(
                _seed(
                    scientific_spec(),
                    "calibration",
                    "discrete",
                    "independent",
                    support,
                    repeat,
                )
            )
            past, future = _canonical_samples("independent", support, rng)
            atoms = _exact_discrete_atoms(past, future)
            maximum_atoms.append(max(abs(float(value)) for value in atoms.values()))
        independent_rows.append(
            {
                "family": "discrete_independent",
                "support": support,
                "maximum_absolute_atom": max(maximum_atoms),
            }
        )

    covariance, truth, partition = _gaussian_fixture()
    gaussian_rows: list[dict[str, Any]] = []
    for support in (128, 256, 512, 1024, 2048):
        estimates = []
        for repeat in range(repetitions):
            rng = np.random.default_rng(
                _seed(scientific_spec(), "calibration", "gaussian", support, repeat)
            )
            estimates.append(_sample_wms(covariance, partition, support, rng))
        values = np.asarray(estimates, dtype=float)
        finite = values[np.isfinite(values)]
        gaussian_rows.append(
            {
                "family": "analytic_gaussian",
                "support": support,
                "truth": truth,
                "median": float(np.median(finite)),
                "median_relative_error": float(
                    np.median(np.abs(finite - truth) / max(abs(truth), 1e-12))
                ),
                "sign_rate": float(np.mean(np.sign(finite) == np.sign(truth))),
                "finite_rate": float(finite.size / repetitions),
            }
        )

    surrogate_rows: list[dict[str, Any]] = []
    for support in (128, 256, 512, 1024, 2048):
        paired_values: list[float] = []
        shuffled_values: list[float] = []
        for repeat in range(10):
            rng = np.random.default_rng(
                _seed(
                    scientific_spec(),
                    "calibration",
                    "clr_surrogate",
                    support,
                    repeat,
                )
            )
            paired, shuffled = _sample_clr_surrogate(support, rng)
            paired_values.append(paired)
            shuffled_values.append(shuffled)
        paired_array = np.asarray(paired_values, dtype=float)
        shuffled_array = np.asarray(shuffled_values, dtype=float)
        finite = np.isfinite(paired_array) & np.isfinite(shuffled_array)
        surrogate_rows.append(
            {
                "family": "gaussian_clr_99d",
                "support": support,
                "replicates": len(paired_values),
                "finite_rate": float(finite.mean()),
                "median_paired": float(np.median(paired_array[finite])),
                "median_deranged": float(np.median(shuffled_array[finite])),
                "median_temporal_excess": float(
                    np.median(paired_array[finite] - shuffled_array[finite])
                ),
            }
        )

    null_values: list[float] = []
    deranged_values: list[float] = []
    for repeat in range(200):
        rng = np.random.default_rng(
            _seed(scientific_spec(), "calibration", "null", repeat)
        )
        past = rng.normal(size=(2, 1024))
        future = rng.normal(size=(2, 1024))
        atoms = local_phi_id_atoms(past, future)
        null_values.append(float(sum(np.mean(atoms[atom]) for atom in PHIR_ATOMS)))
        related = 0.5 * past + rng.normal(size=(2, 1024))
        deranged = related[:, rng.permutation(related.shape[1])]
        atoms = local_phi_id_atoms(past, deranged)
        deranged_values.append(
            float(sum(np.mean(atoms[atom]) for atom in PHIR_ATOMS))
        )
    null_mean = float(np.mean(null_values))
    deranged_mean = float(np.mean(deranged_values))

    discrete_pass = all(
        row["maximum_identity_error"] <= 1e-10
        and row["maximum_signature_error"] <= 1e-10
        and row["signature_recovery_rate"]
        >= (0.80 if row["support"] == 128 else 0.95)
        for row in rows
    ) and all(row["maximum_absolute_atom"] <= 1e-10 for row in independent_rows)
    gaussian_128 = next(row for row in gaussian_rows if row["support"] == 128)
    gaussian_1024 = next(row for row in gaussian_rows if row["support"] == 1024)
    gaussian_pass = bool(
        gaussian_128["sign_rate"] >= 0.90
        and gaussian_1024["median_relative_error"] <= 0.10
    )
    null_pass = bool(
        abs(null_mean) <= CALIBRATION_NULL_MARGIN_BITS
        and abs(deranged_mean) <= CALIBRATION_NULL_MARGIN_BITS
    )
    payload = {
        "format": "codex-ch5-phir-px10-calibration-v1",
        "source_hashes": _source_hashes(),
        "discrete_reference": rows,
        "discrete_independent": independent_rows,
        "analytic_gaussian": gaussian_rows,
        "gaussian_clr_99d": surrogate_rows,
        "null": {
            "replicates": len(null_values),
            "mean_independent_revised_phi_r": null_mean,
            "mean_deranged_revised_phi_r": deranged_mean,
            "margin_bits": CALIBRATION_NULL_MARGIN_BITS,
            "pass": null_pass,
        },
        "gates": {
            "discrete_reference": discrete_pass,
            "analytic_gaussian": gaussian_pass,
            "null": null_pass,
            "atom_instrument_eligible": bool(discrete_pass and gaussian_pass and null_pass),
        },
    }
    _atomic_json(output / "calibration.json", payload)
    write_checksums(output)
    return payload


@dataclass(frozen=True)
class AtomCasePayload:
    case: px9.ResilienceCase
    initial_by_arm: Mapping[str, NDArray[np.int64]]
    blocks_by_arm: Mapping[str, tuple[px9.PairBlock, ...]]


@dataclass(frozen=True)
class PX10Batch:
    matrix_id: int
    beta: NDArray[np.float64]
    initial: NDArray[np.int16]
    acquisition_rows: tuple[dict[str, Any], ...]
    edit_rows: tuple[dict[str, Any], ...]
    branch_rows: tuple[dict[str, Any], ...]
    score_rows: tuple[dict[str, Any], ...]
    atom_rows: tuple[dict[str, Any], ...]
    cpu_seconds: float
    scientific_digest: str

    def as_px9(self) -> px9.PX9Batch:
        return px9.PX9Batch(
            self.matrix_id,
            self.beta,
            self.initial,
            self.acquisition_rows,
            self.edit_rows,
            self.branch_rows,
            self.score_rows,
            self.cpu_seconds,
            self.scientific_digest,
        )


def _classify_compositions(
    model: GrainModel,
    candidate: str,
    compositions: NDArray,
    beta: NDArray,
) -> tuple[NDArray, NDArray]:
    features = state_graph_features_many(
        np.asarray(compositions, dtype=np.int64), beta, GardConfig()
    )
    return model.classify(candidate, features)


def _run_case(
    case: px9.ResilienceCase,
    students: Mapping[tuple[str, str], cr5.FrozenCR5Student],
    grain: GrainModel,
    spec: PX10Spec,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    AtomCasePayload,
]:
    px9_spec = spec.as_px9()
    selection_rows, edits = px9._select_edits(case, students, px9_spec)
    branch_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    edit_rows: list[dict[str, Any]] = []
    blocks_by_arm: dict[str, list[px9.PairBlock]] = {}
    rows_by_arm: dict[str, list[dict[str, Any]]] = {}
    initial_by_arm: dict[str, NDArray[np.int64]] = {}
    for selected, edit in zip(selection_rows, edits, strict=True):
        arm = str(selected["arm"])
        composition = (
            np.asarray(case.snapshot.composition, dtype=np.int64).copy()
            if edit is None
            else apply_molecular_edit(case.snapshot.composition, edit)
        )
        initial_by_arm[arm] = composition
        initial_micro, initial_macro = _classify_compositions(
            grain, case.candidate, composition[None, :], case.beta
        )
        summary = px9._composition_summary(composition, case.beta)
        edit_rows.append(
            {
                "state_id": case.state_id,
                "axis": "resilience",
                "candidate": case.candidate,
                "matrix_id": case.matrix_id,
                "landmark": case.landmark,
                **selected,
                "remove_type": -1 if edit is None else edit.remove_type,
                "add_type": -1 if edit is None else edit.add_type,
                "history_digest": _array_digest(case.history_counts),
                "initial_micro_bin": int(initial_micro[0]),
                "initial_macro_bin": int(initial_macro[0]),
                **{f"state_{key}": value for key, value in summary.items()},
            }
        )
        blocks: list[px9.PairBlock] = []
        arm_rows: list[dict[str, Any]] = []
        for branch in range(spec.branches):
            row, block = px9._simulate_branch(case, edit, branch, px9_spec)
            row.update(
                {
                    "arm": arm,
                    "prediction": float(selected["prediction"]),
                    "empirical_quantile": float(selected["empirical_quantile"]),
                    "initial_micro_bin": int(initial_micro[0]),
                    "initial_macro_bin": int(initial_macro[0]),
                    "future_micro_lag1": -1,
                    "future_macro_lag1": -1,
                    "future_micro_lag3": -1,
                    "future_macro_lag3": -1,
                }
            )
            arm_rows.append(row)
            blocks.append(block)
        for lag in (1, 3):
            valid = [
                index
                for index, block in enumerate(blocks)
                if len(block.generational_future) >= lag
            ]
            if valid:
                compositions = np.asarray(
                    [blocks[index].generational_future[lag - 1] for index in valid],
                    dtype=np.int64,
                )
                micro, macro = _classify_compositions(
                    grain, case.candidate, compositions, case.beta
                )
                for offset, index in enumerate(valid):
                    arm_rows[index][f"future_micro_lag{lag}"] = int(micro[offset])
                    arm_rows[index][f"future_macro_lag{lag}"] = int(macro[offset])
        branch_rows.extend(arm_rows)
        blocks_by_arm[arm] = blocks
        rows_by_arm[arm] = arm_rows
    for arm in ARMS:
        score_rows.extend(
            px9._score_arm_halves(arm, case, blocks_by_arm[arm], px9_spec)
        )
    score_rows.extend(
        px9._score_concordant_extremes(case, blocks_by_arm, rows_by_arm, px9_spec)
    )
    return (
        branch_rows,
        score_rows,
        edit_rows,
        AtomCasePayload(
            case,
            initial_by_arm,
            {arm: tuple(values) for arm, values in blocks_by_arm.items()},
        ),
    )


def _daughter_sequence(
    initial: NDArray, block: px9.PairBlock
) -> NDArray[np.int16]:
    if len(block.generational_future):
        return np.vstack(
            (
                np.asarray(initial, dtype=np.int16)[None, :],
                np.asarray(block.generational_future, dtype=np.int16),
            )
        )
    return np.asarray(initial, dtype=np.int16)[None, :]


def _lag_pairs(
    payload: AtomCasePayload,
    arm: str,
    indices: Sequence[int],
    lag: int,
    shift: int | None = None,
) -> tuple[NDArray[np.int16], NDArray[np.int16], int]:
    sequences = [
        _daughter_sequence(payload.initial_by_arm[arm], payload.blocks_by_arm[arm][index])
        for index in indices
    ]
    past_rows: list[NDArray[np.int16]] = []
    future_rows: list[NDArray[np.int16]] = []
    self_pairs = 0
    maximum_depth = max((len(sequence) - lag for sequence in sequences), default=0)
    for depth in range(maximum_depth):
        eligible = [sequence for sequence in sequences if len(sequence) > depth + lag]
        if not eligible:
            continue
        futures = [sequence[depth + lag] for sequence in eligible]
        if shift is None or len(eligible) < 2:
            assigned = futures
        else:
            amount = int(shift) % len(eligible)
            if amount == 0:
                amount = 1
            assigned = futures[amount:] + futures[:amount]
        past_rows.extend(sequence[depth] for sequence in eligible)
        future_rows.extend(assigned)
    n_types = GardConfig().n_types
    if not past_rows:
        return (
            np.empty((0, n_types), dtype=np.int16),
            np.empty((0, n_types), dtype=np.int16),
            self_pairs,
        )
    return np.asarray(past_rows), np.asarray(future_rows), self_pairs


def _matrix_random_partitions(
    beta: NDArray, matrix_id: int, spec: PX10Spec
) -> tuple[tuple[NDArray[np.int64], NDArray[np.int64]], ...]:
    first, second = beta_physical_partition(beta)
    split = len(first)
    output: list[tuple[NDArray[np.int64], NDArray[np.int64]]] = []
    seen: set[tuple[int, ...]] = set()
    control = 0
    while len(output) < RANDOM_PARTITIONS:
        rng = np.random.default_rng(
            _seed(spec, "partition", "matrix", matrix_id, control)
        )
        order = rng.permutation(GardConfig().n_types)
        left = np.sort(order[:split]).astype(np.int64)
        key = tuple(int(value) for value in left)
        control += 1
        if key in seen:
            continue
        seen.add(key)
        output.append((left, np.sort(order[split:]).astype(np.int64)))
    return tuple(output)


def _atom_score(
    past_counts: NDArray,
    future_counts: NDArray,
    first_species: Sequence[int],
    second_species: Sequence[int],
) -> dict[str, Any]:
    try:
        past, future, active = px7._explicit_transform(past_counts, future_counts)
        first, second = px7._map_partition(active, first_species, second_species)
        past_macro = np.vstack((past[first].mean(axis=0), past[second].mean(axis=0)))
        future_macro = np.vstack(
            (future[first].mean(axis=0), future[second].mean(axis=0))
        )
        atom_series = local_phi_id_atoms(past_macro, future_macro)
        means = {atom: float(np.mean(values)) for atom, values in atom_series.items()}
        causation = means[(SYNERGISTIC, UNIQUE_0)] + means[(SYNERGISTIC, UNIQUE_1)]
        synergy = means[(SYNERGISTIC, SYNERGISTIC)]
        row: dict[str, Any] = {
            "causation": float(causation),
            "synergy_persistence": float(synergy),
            "emergence": float(causation + synergy),
            "revised_phi_r": float(sum(means[atom] for atom in PHIR_ATOMS)),
            "transitions": int(past.shape[1]),
            "active_dimensions": int(active.size),
            "part_a_dimensions": int(first.size),
            "part_b_dimensions": int(second.size),
            "partition_digest": _array_digest(active, first, second),
        }
        for source in ANTICHAINS:
            for target in ANTICHAINS:
                atom = (source, target)
                row[f"atom_{atom_name(atom)}"] = means[atom]
        return row
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        row = {
            "causation": float("nan"),
            "synergy_persistence": float("nan"),
            "emergence": float("nan"),
            "revised_phi_r": float("nan"),
            "transitions": int(len(past_counts)),
            "active_dimensions": 0,
            "part_a_dimensions": 0,
            "part_b_dimensions": 0,
            "partition_digest": "invalid",
        }
        row.update({f"atom_{name}": float("nan") for name in ATOM_NAMES})
        return row


def _pooled_atom_rows(
    matrix_id: int,
    beta: NDArray,
    payloads: Sequence[AtomCasePayload],
    spec: PX10Spec,
) -> list[dict[str, Any]]:
    beta_first, beta_second = beta_physical_partition(beta)
    random_partitions = _matrix_random_partitions(beta, matrix_id, spec)
    rows: list[dict[str, Any]] = []
    halves = _halves(spec)
    for candidate in CANDIDATES:
        local_cases = [item for item in payloads if item.case.candidate == candidate]
        if not local_cases:
            continue
        for arm in ATOM_ARMS:
            for half, full_indices in halves.items():
                indices = full_indices[: min(ATOM_SUPPORT_BRANCHES, len(full_indices))]
                for lag in ALL_LAGS:
                    real_past: list[NDArray] = []
                    real_future: list[NDArray] = []
                    shuffled: dict[int, tuple[list[NDArray], list[NDArray]]] = {
                        control: ([], []) for control in range(len(TEMPORAL_SHIFTS))
                    }
                    for payload in local_cases:
                        left, right, _ = _lag_pairs(payload, arm, indices, lag)
                        if len(left):
                            real_past.append(left)
                            real_future.append(right)
                        for control, shift in enumerate(TEMPORAL_SHIFTS):
                            left_s, right_s, self_pairs = _lag_pairs(
                                payload, arm, indices, lag, shift
                            )
                            if self_pairs and len(indices) >= 2:
                                raise AssertionError(
                                    "PX10 lag derangement retained a self-pair"
                                )
                            if len(left_s):
                                shuffled[control][0].append(left_s)
                                shuffled[control][1].append(right_s)
                    if not real_past:
                        continue
                    past = np.vstack(real_past)
                    future = np.vstack(real_future)
                    key = {
                        "matrix_id": matrix_id,
                        "candidate": candidate,
                        "arm": arm,
                        "source_half": half,
                        "lag": lag,
                        "support_branches": len(indices),
                    }
                    rows.append(
                        {
                            **key,
                            "score_kind": "paired_beta",
                            "control_id": -1,
                            **_atom_score(
                                past, future, beta_first, beta_second
                            ),
                        }
                    )
                    for control in range(len(TEMPORAL_SHIFTS)):
                        left = np.vstack(shuffled[control][0])
                        right = np.vstack(shuffled[control][1])
                        rows.append(
                            {
                                **key,
                                "score_kind": "shuffled_beta",
                                "control_id": control,
                                **_atom_score(
                                    left, right, beta_first, beta_second
                                ),
                            }
                        )
                    for control, (first, second) in enumerate(random_partitions):
                        rows.append(
                            {
                                **key,
                                "score_kind": "random_partition",
                                "control_id": control,
                                **_atom_score(past, future, first, second),
                            }
                        )
    return rows


def _matrix_digest(batch: Mapping[str, Any]) -> str:
    return _digest(batch)


def _run_matrix(
    arguments: tuple[int, PX10Spec, str, str, str]
) -> PX10Batch:
    matrix_id, spec, model_path, contract_path, grain_path = arguments
    started = time.process_time()
    with threadpool_limits(limits=1), _px9_seed_context():
        config = GardConfig()
        beta = generate_beta(
            config, np.random.default_rng(_seed(spec, "matrix", matrix_id))
        )
        initial = generate_initial_composition(
            config, np.random.default_rng(_seed(spec, "initial", matrix_id))
        ).astype(np.int16)
        students = cr5.load_students(Path(model_path), Path(contract_path))
        grain = load_grain_model(Path(grain_path))
        px9_spec = spec.as_px9()
        natural: list[px9.ResilienceCase] = []
        for candidate in CANDIDATES:
            natural.extend(
                px9._run_natural_candidate(
                    matrix_id, beta, initial, candidate, px9_spec
                )
            )
        cases: list[px9.ResilienceCase] = []
        acquisition_rows: list[dict[str, Any]] = []
        for source in natural:
            broken, acquisition = px9._acquire_break(source, px9_spec)
            acquisition_rows.append(acquisition)
            if broken is not None:
                cases.append(broken)
        branch_rows: list[dict[str, Any]] = []
        score_rows: list[dict[str, Any]] = []
        edit_rows: list[dict[str, Any]] = []
        atom_payloads: list[AtomCasePayload] = []
        for case in cases:
            branches, scores, edits, payload = _run_case(
                case, students, grain, spec
            )
            branch_rows.extend(branches)
            score_rows.extend(scores)
            edit_rows.extend(edits)
            atom_payloads.append(payload)
        atom_rows = _pooled_atom_rows(matrix_id, beta, atom_payloads, spec)
        scientific_digest = _matrix_digest(
            {
                "matrix_id": matrix_id,
                "beta": _array_digest(beta),
                "initial": _array_digest(initial),
                "acquisition": acquisition_rows,
                "edits": edit_rows,
                "branches": branch_rows,
                "scores": score_rows,
                "atoms": atom_rows,
            }
        )
        return PX10Batch(
            matrix_id,
            beta,
            initial,
            tuple(acquisition_rows),
            tuple(edit_rows),
            tuple(branch_rows),
            tuple(score_rows),
            tuple(atom_rows),
            float(time.process_time() - started),
            scientific_digest,
        )


def _bootstrap_summary(
    series: pd.Series,
    repetitions: int,
    key: str,
    arrays: dict[str, NDArray],
    direction: float = 1.0,
) -> dict[str, Any]:
    local = series.replace([np.inf, -np.inf], np.nan).dropna().sort_index()
    values = np.asarray(local, dtype=np.float64)
    matrix_ids = np.asarray(local.index, dtype=np.int64)
    safe = key.replace("/", "__")
    if not values.size:
        return {
            "effect": float("nan"),
            "ci95": [float("nan"), float("nan")],
            "ci90": [float("nan"), float("nan")],
            "one_sided_p": 1.0,
            "matrices": 0,
            "matrices_positive": 0,
        }
    spec = scientific_spec()
    rng = np.random.default_rng(_seed(spec, "bootstrap", key))
    indices = rng.integers(0, len(values), size=(repetitions, len(values)))
    bootstrap = values[indices].mean(axis=1)
    sign_rng = np.random.default_rng(_seed(spec, "randomization", key))
    signs = sign_rng.choice((-1.0, 1.0), size=(RANDOMIZATION_DRAWS, len(values)))
    observed_aligned = direction * float(values.mean())
    randomized = direction * (signs * values).mean(axis=1)
    output = {
        "effect": float(values.mean()),
        "aligned_effect": observed_aligned,
        "ci95": [float(value) for value in np.quantile(bootstrap, (0.025, 0.975))],
        "ci90": [float(value) for value in np.quantile(bootstrap, (0.05, 0.95))],
        "one_sided_p": float((1 + np.count_nonzero(randomized >= observed_aligned)) / (len(randomized) + 1)),
        "matrices": int(len(values)),
        "matrices_positive": int(np.count_nonzero(direction * values > 0)),
        "maximum_absolute_matrix_effect": float(np.max(np.abs(values))),
    }
    arrays[f"{safe}__matrix_ids"] = matrix_ids
    arrays[f"{safe}__matrix_values"] = values
    arrays[f"{safe}__bootstrap"] = bootstrap
    arrays[f"{safe}__randomization"] = randomized
    return output


def _adjust_family(items: list[dict[str, Any]], directions: Sequence[float] | None = None) -> None:
    if not items:
        return
    adjusted = holm_adjust([float(item["one_sided_p"]) for item in items])
    if directions is None:
        directions = [1.0] * len(items)
    for item, p_value, direction in zip(items, adjusted, directions, strict=True):
        lower, upper = item["ci95"]
        ci_directional = lower > 0 if direction > 0 else upper < 0
        item["holm_adjusted_p"] = float(p_value)
        item["pass"] = bool(
            direction * float(item["effect"]) > 0
            and ci_directional
            and p_value < 0.05
        )


def _derive_atom_scores(atoms: pd.DataFrame) -> pd.DataFrame:
    keys = ["matrix_id", "candidate", "arm", "source_half", "lag"]
    rows: list[dict[str, Any]] = []
    for key, local in atoms.groupby(keys, sort=True):
        paired = local[local["score_kind"] == "paired_beta"]
        shuffled = local[local["score_kind"] == "shuffled_beta"]
        random = local[local["score_kind"] == "random_partition"]
        if len(paired) != 1:
            continue
        base = dict(zip(keys, key, strict=True))
        row = {**base}
        for group in (*ATOM_GROUPS, "revised_phi_r"):
            real = float(paired.iloc[0][group])
            shuffled_mean = float(shuffled[group].mean()) if len(shuffled) else float("nan")
            random_median = float(random[group].median()) if len(random) else float("nan")
            row[f"{group}_paired"] = real
            row[f"{group}_temporal"] = real - shuffled_mean
            row[f"{group}_topology"] = real - random_median
        row["temporal_controls"] = int(len(shuffled))
        row["random_partitions"] = int(len(random))
        row["transitions"] = int(paired.iloc[0]["transitions"])
        rows.append(row)
    return pd.DataFrame(rows)


def _atom_family(
    derived: pd.DataFrame,
    value_suffix: str,
    spec: PX10Spec,
    arrays: dict[str, NDArray],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    directions: list[float] = []
    primary = derived[derived["lag"].isin(PRIMARY_LAGS)]
    for candidate in CANDIDATES:
        for half in _halves(spec):
            for group in ATOM_GROUPS:
                column = f"{group}_{value_suffix}"
                local = primary[
                    (primary["candidate"] == candidate)
                    & (primary["source_half"] == half)
                    & (primary["arm"].isin(("Q00", "Q100")))
                ]
                lag_mean = (
                    local.groupby(["matrix_id", "arm"], sort=True)[column]
                    .mean()
                    .unstack("arm")
                )
                effect = lag_mean.get("Q100", pd.Series(dtype=float)) - lag_mean.get(
                    "Q00", pd.Series(dtype=float)
                )
                direction = ATOM_DIRECTIONS[group]
                item = _bootstrap_summary(
                    effect,
                    spec.bootstrap_draws,
                    f"atom/{value_suffix}/{candidate}/{half}/{group}",
                    arrays,
                    direction,
                )
                item.update(
                    {
                        "candidate": candidate,
                        "source_half": half,
                        "group": group,
                        "value": value_suffix,
                        "registered_direction": "positive" if direction > 0 else "negative",
                    }
                )
                items.append(item)
                directions.append(direction)
    _adjust_family(items, directions)
    return items


def _log_loss_bits(outcome: NDArray, probability: NDArray) -> NDArray:
    y = np.asarray(outcome, dtype=np.float64)
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-12, 1 - 1e-12)
    return -(y * np.log2(p) + (1 - y) * np.log2(1 - p))


def _derangement(size: int, rng: np.random.Generator) -> NDArray[np.int64]:
    if size < 2:
        raise ValueError("a derangement needs at least two labels")
    original = np.arange(size, dtype=np.int64)
    while True:
        candidate = rng.permutation(size).astype(np.int64)
        if np.all(candidate != original):
            return candidate


def _derange_source_arm_labels(
    source: pd.DataFrame,
    candidate: str,
    source_half: str,
    control: int,
    arms: Sequence[str],
) -> pd.DataFrame:
    """Destroy a stable action/outcome map while preserving every arm count."""

    arm_order = tuple(str(value) for value in arms)
    arm_index = {arm: index for index, arm in enumerate(arm_order)}
    output = source.copy()
    for (state_id, branch), indices in output.groupby(
        ["state_id", "branch"], sort=True
    ).groups.items():
        local = output.loc[indices]
        if set(local["arm"].astype(str)) != set(arm_order):
            raise ValueError("source arm-label control lacks a complete action panel")
        rng = np.random.default_rng(
            _seed(
                scientific_spec(),
                "randomization",
                "channel_label",
                candidate,
                source_half,
                state_id,
                int(branch),
                control,
            )
        )
        mapping = _derangement(len(arm_order), rng)
        output.loc[indices, "arm"] = [
            arm_order[int(mapping[arm_index[str(value)]])]
            for value in local["arm"]
        ]
    return output


def _channel_matrix_values(
    branches: pd.DataFrame,
    candidate: str,
    source_half: str,
    target_half: str,
    arms: Sequence[str],
    label_control: int | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    local = branches[
        (branches["candidate"] == candidate) & (branches["arm"].isin(arms))
    ].copy()
    source = local[local["half"] == source_half]
    if label_control is not None:
        source = _derange_source_arm_labels(
            source, candidate, source_half, label_control, arms
        )
    target = local[local["half"] == target_half].copy()
    arm_counts = (
        source.groupby(["state_id", "arm"], sort=True)["primary"]
        .agg(["sum", "count"])
        .reset_index()
    )
    arm_counts["p_arm"] = (arm_counts["sum"] + 0.5) / (arm_counts["count"] + 1.0)
    state_counts = (
        source.groupby("state_id", sort=True)["primary"]
        .agg(["sum", "count"])
        .reset_index()
    )
    state_counts["p_state"] = (state_counts["sum"] + 0.5) / (
        state_counts["count"] + 1.0
    )
    prediction = arm_counts[["state_id", "arm", "p_arm"]]
    target = target.merge(prediction, on=["state_id", "arm"], how="inner")
    target = target.merge(state_counts[["state_id", "p_state"]], on="state_id", how="inner")
    target["gain_bits"] = _log_loss_bits(
        target["primary"].to_numpy(), target["p_state"].to_numpy()
    ) - _log_loss_bits(target["primary"].to_numpy(), target["p_arm"].to_numpy())
    return target.groupby("matrix_id", sort=True)["gain_bits"].mean(), arm_counts


def _binary_channel_capacity(probabilities: Sequence[float]) -> float:
    values = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1 - 1e-12)
    channel = np.column_stack((1.0 - values, values))
    weights = np.full(len(values), 1.0 / len(values))
    for _ in range(10000):
        marginal = np.clip(weights @ channel, 1e-15, 1.0)
        divergence = np.sum(channel * np.log(channel / marginal), axis=1)
        updated = weights * np.exp(divergence)
        updated /= updated.sum()
        if np.max(np.abs(updated - weights)) < 1e-12:
            weights = updated
            break
        weights = updated
    marginal = np.clip(weights @ channel, 1e-15, 1.0)
    return float(
        np.sum(weights[:, None] * channel * np.log2(channel / marginal))
    )


def _uniform_binary_channel_information(probabilities: Sequence[float]) -> float:
    values = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1 - 1e-12)
    channel = np.column_stack((1.0 - values, values))
    weights = np.full(len(values), 1.0 / len(values))
    marginal = np.clip(weights @ channel, 1e-15, 1.0)
    return float(
        np.sum(weights[:, None] * channel * np.log2(channel / marginal))
    )


def _channel_analysis(
    branches: pd.DataFrame,
    spec: PX10Spec,
    arrays: dict[str, NDArray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], bool]:
    primary: list[dict[str, Any]] = []
    shuffled_controls: list[dict[str, Any]] = []
    random_controls: list[dict[str, Any]] = []
    directions = (("A", "B"), ("B", "A"))
    for candidate in CANDIDATES:
        for source, target in directions:
            values, probabilities = _channel_matrix_values(
                branches, candidate, source, target, QUANTILE_ARMS
            )
            item = _bootstrap_summary(
                values,
                spec.bootstrap_draws,
                f"channel/{candidate}/{source}_to_{target}",
                arrays,
            )
            state_capacities = probabilities.groupby("state_id", sort=True)["p_arm"].apply(
                _binary_channel_capacity
            )
            state_information = probabilities.groupby("state_id", sort=True)["p_arm"].apply(
                _uniform_binary_channel_information
            )
            item.update(
                {
                    "candidate": candidate,
                    "direction": f"{source}_to_{target}",
                    "mean_source_channel_capacity_bits": float(state_capacities.mean()),
                    "mean_source_uniform_information_bits": float(
                        state_information.mean()
                    ),
                }
            )
            primary.append(item)

            shuffled_matrix: list[pd.Series] = []
            for control_id in range(16):
                shuffled, _ = _channel_matrix_values(
                    branches,
                    candidate,
                    source,
                    target,
                    QUANTILE_ARMS,
                    control_id,
                )
                shuffled_matrix.append(shuffled)
            shuffled_frame = pd.concat(shuffled_matrix, axis=1)
            shuffled_mean = shuffled_frame.mean(axis=1)
            control = _bootstrap_summary(
                shuffled_mean,
                spec.bootstrap_draws,
                f"channel_null/shuffled/{candidate}/{source}_to_{target}",
                arrays,
            )
            control.update(
                {
                    "candidate": candidate,
                    "direction": f"{source}_to_{target}",
                    "control": "arm_label_derangement",
                    "equivalence_margin_bits": INFORMATION_EQUIVALENCE_MARGIN_BITS,
                    "equivalent": bool(
                        control["ci90"][0] > -INFORMATION_EQUIVALENCE_MARGIN_BITS
                        and control["ci90"][1] < INFORMATION_EQUIVALENCE_MARGIN_BITS
                    ),
                }
            )
            shuffled_controls.append(control)

            random_value, _ = _channel_matrix_values(
                branches, candidate, source, target, ("RANDOM", "NOOP")
            )
            random_item = _bootstrap_summary(
                random_value,
                spec.bootstrap_draws,
                f"channel_null/random_noop/{candidate}/{source}_to_{target}",
                arrays,
            )
            random_item.update(
                {
                    "candidate": candidate,
                    "direction": f"{source}_to_{target}",
                    "control": "random_vs_noop",
                    "equivalence_margin_bits": INFORMATION_EQUIVALENCE_MARGIN_BITS,
                    "equivalent": bool(
                        random_item["ci90"][0] > -INFORMATION_EQUIVALENCE_MARGIN_BITS
                        and random_item["ci90"][1] < INFORMATION_EQUIVALENCE_MARGIN_BITS
                    ),
                }
            )
            random_controls.append(random_item)
    _adjust_family(primary)
    passed = bool(
        len(primary) == 4
        and all(item["pass"] for item in primary)
        and all(item["equivalent"] for item in shuffled_controls)
        and all(item["equivalent"] for item in random_controls)
    )
    return primary, shuffled_controls, random_controls, passed


def _kernel_metrics(conditional: NDArray, input_rows: Sequence[int]) -> dict[str, float]:
    selected = np.asarray(input_rows, dtype=int)
    if not selected.size:
        return {
            "effective_information_bits": float("nan"),
            "effectiveness": float("nan"),
            "determinism_bits": float("nan"),
            "degeneracy_bits": float("nan"),
        }
    channel = np.asarray(conditional[selected], dtype=np.float64)
    weights = np.full(len(selected), 1.0 / len(selected))
    marginal = np.clip(weights @ channel, 1e-15, 1.0)
    entropy_conditional = -np.sum(channel * np.log2(np.clip(channel, 1e-15, 1.0)), axis=1)
    entropy_output = -float(np.sum(marginal * np.log2(marginal)))
    maximum = math.log2(channel.shape[1])
    determinism = maximum - float(entropy_conditional.mean())
    degeneracy = maximum - entropy_output
    effective = determinism - degeneracy
    return {
        "effective_information_bits": float(effective),
        "effectiveness": float(effective / max(math.log2(len(selected)), 1.0)),
        "determinism_bits": float(determinism),
        "degeneracy_bits": float(degeneracy),
    }


def _macro_from_micro(conditional: NDArray, observed_inputs: Sequence[int]) -> tuple[NDArray, list[int]]:
    observed = set(int(value) for value in observed_inputs)
    macro = np.zeros((4, 4), dtype=np.float64)
    active_macro: list[int] = []
    for source_macro in range(4):
        children = [value for value in observed if value // 4 == source_macro]
        if not children:
            continue
        active_macro.append(source_macro)
        row = np.mean(conditional[children], axis=0)
        for target_micro, probability in enumerate(row):
            macro[source_macro, target_micro // 4] += probability
    return macro, active_macro


def _macro_micro_analysis(branches: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in CANDIDATES:
        for arm in ARMS:
            local = branches[
                (branches["candidate"] == candidate) & (branches["arm"] == arm)
            ]
            for lag in (1, 3):
                target = f"future_micro_lag{lag}"
                valid = local[local[target] >= 0]
                counts = np.zeros((16, 16), dtype=np.float64)
                for source, future in zip(
                    valid["initial_micro_bin"].astype(int),
                    valid[target].astype(int),
                    strict=True,
                ):
                    counts[source, future] += 1.0
                observed = np.flatnonzero(counts.sum(axis=1) > 0).tolist()
                conditional = (counts + 0.5) / (counts.sum(axis=1, keepdims=True) + 8.0)
                micro = _kernel_metrics(conditional, observed)
                macro_channel, active_macro = _macro_from_micro(conditional, observed)
                macro = _kernel_metrics(macro_channel, active_macro)
                rows.append(
                    {
                        "candidate": candidate,
                        "arm": arm,
                        "lag": lag,
                        "branches": int(len(valid)),
                        "observed_micro_inputs": int(len(observed)),
                        "observed_macro_inputs": int(len(active_macro)),
                        **{f"micro_{key}": value for key, value in micro.items()},
                        **{f"macro_{key}": value for key, value in macro.items()},
                        "macro_minus_micro_effective_information_bits": float(
                            macro["effective_information_bits"]
                            - micro["effective_information_bits"]
                        ),
                    }
                )
    return rows


def analyze_batches(
    batches: Sequence[PX10Batch],
    spec: PX10Spec,
    calibration: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, pd.DataFrame], dict[str, NDArray]]:
    with _px9_seed_context():
        echo_metrics, echo_tables, echo_arrays = px9.analyze_batches(
            [batch.as_px9() for batch in batches], spec.as_px9()
        )
    arrays = dict(echo_arrays)
    acquisitions = echo_tables["acquisition"]
    branches = echo_tables["branches"]
    atoms = pd.DataFrame([row for batch in batches for row in batch.atom_rows])
    derived_atoms = _derive_atom_scores(atoms)

    eligibility: dict[str, dict[str, Any]] = {}
    for candidate in CANDIDATES:
        local = acquisitions[
            (acquisitions["candidate"] == candidate)
            & (acquisitions["eligible"] == 1)
        ]
        matrices = int(local["matrix_id"].nunique())
        eligibility[candidate] = {
            "eligible_states": int(len(local)),
            "eligible_matrices": matrices,
            "minimum": MINIMUM_ELIGIBLE_MATRICES,
            "pass": matrices >= MINIMUM_ELIGIBLE_MATRICES,
        }
    eligibility_pass = all(item["pass"] for item in eligibility.values())
    manipulation = bool(
        eligibility_pass
        and len(echo_metrics["outcome_extreme"]) == 4
        and len(echo_metrics["outcome_dose"]) == 4
        and all(item["pass"] for item in echo_metrics["outcome_extreme"])
        and all(item["pass"] for item in echo_metrics["outcome_dose"])
    )

    nonredundancy_bits: list[dict[str, Any]] = []
    for source in echo_metrics["nonredundancy_log_loss"]:
        item = dict(source)
        for key in ("effect", "aligned_effect", "maximum_absolute_matrix_effect"):
            if key in item:
                item[key] = float(item[key]) / math.log(2.0)
        for key in ("ci95", "ci90"):
            item[key] = [float(value) / math.log(2.0) for value in item[key]]
        item["unit"] = "bits_per_branch"
        item["equivalence_margin"] = INFORMATION_EQUIVALENCE_MARGIN_BITS
        item["equivalent_to_zero"] = bool(
            item["ci90"][0] > -INFORMATION_EQUIVALENCE_MARGIN_BITS
            and item["ci90"][1] < INFORMATION_EQUIVALENCE_MARGIN_BITS
        )
        nonredundancy_bits.append(item)

    temporal_authenticity = bool(
        manipulation and echo_metrics["gates"]["temporal_authenticity"]
    )
    positive_nonredundancy = bool(
        len(nonredundancy_bits) == 4 and all(item["pass"] for item in nonredundancy_bits)
    )
    equivalent_redundancy = bool(
        len(nonredundancy_bits) == 4
        and all(item["equivalent_to_zero"] for item in nonredundancy_bits)
    )
    if not temporal_authenticity:
        echo_classification = "temporal_echo_not_confirmed"
    elif positive_nonredundancy:
        echo_classification = "reliable_nonredundant_temporal_gauge"
    elif equivalent_redundancy:
        echo_classification = "reliable_redundant_temporal_echo"
    else:
        echo_classification = "temporal_echo_reliability_unresolved"

    atom_temporal = _atom_family(derived_atoms, "temporal", spec, arrays)
    atom_topology = _atom_family(derived_atoms, "topology", spec, arrays)
    atom_eligible = bool(calibration["gates"]["atom_instrument_eligible"])
    atom_fingerprint = bool(
        atom_eligible
        and manipulation
        and len(atom_temporal) == 12
        and all(item["pass"] for item in atom_temporal)
    )
    atom_specificity = bool(
        atom_fingerprint
        and len(atom_topology) == 12
        and all(item["pass"] for item in atom_topology)
    )

    channel, channel_shuffled, channel_random, channel_pass = _channel_analysis(
        branches, spec, arrays
    )
    macro_micro = _macro_micro_analysis(branches)
    gates = {
        "eligibility": bool(eligibility_pass),
        "plastic_heredity_manipulation_valid": manipulation,
        "temporal_authenticity": temporal_authenticity,
        "temporal_echo_classification": echo_classification,
        "behaviorally_nonredundant": positive_nonredundancy,
        "behaviorally_equivalent_to_zero": equivalent_redundancy,
        "atom_instrument_eligible": atom_eligible,
        "atomic_temporal_fingerprint": atom_fingerprint,
        "atomic_beta_specificity": atom_specificity,
        "intervention_channel": bool(manipulation and channel_pass),
        "public_revised_can_win": False,
        "automatic_continuation_authorized": False,
    }
    metrics = {
        "format": "codex-ch5-phir-px10-primary-metrics-v1",
        "eligibility": eligibility,
        "outcome_extreme": echo_metrics["outcome_extreme"],
        "outcome_dose": echo_metrics["outcome_dose"],
        "temporal_response": echo_metrics["temporal_response"],
        "temporal_reliability": echo_metrics["temporal_reliability"],
        "temporal_forecast": echo_metrics["temporal_forecast"],
        "temporal_dose_concordance": echo_metrics["temporal_dose_concordance"],
        "nonredundancy_bits": nonredundancy_bits,
        "public_revised_negative_control": echo_metrics[
            "public_revised_negative_control"
        ],
        "atomic_temporal_response": atom_temporal,
        "atomic_topology_response": atom_topology,
        "intervention_channel": channel,
        "channel_label_derangement": channel_shuffled,
        "channel_random_noop": channel_random,
        "macro_micro_exploratory": macro_micro,
        "gates": gates,
    }
    tables = {
        **echo_tables,
        "atom_scores": atoms,
        "derived_atom_scores": derived_atoms,
        "macro_micro": pd.DataFrame(macro_micro),
    }
    return metrics, tables, arrays


def validate(output: Path = DEFAULT_VALIDATION) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any = None) -> None:
        checks.append({"name": name, "pass": bool(passed), "detail": _json_ready(detail)})

    spec = scientific_spec()
    check("scientific_shape", spec.matrices == 48 and spec.branches == 256 and spec.horizon == 8)
    check("candidate_separation", tuple(CANDIDATES) == ("02", "03"))
    check("fresh_seed_namespace", LABEL != px9.LABEL and SEED_DOMAINS != px9.SEED_DOMAINS)
    check("no_fable_source", not any("fable" in value.lower() for value in SOURCE_FILES))
    check("dose_arms", QUANTILE_ARMS == ("Q00", "Q20", "Q40", "Q60", "Q80", "Q100"))
    check("atom_lags", PRIMARY_LAGS == (1, 2, 3, 4) and SECONDARY_LAGS == (8,))
    check("matrix_inference", protocol()["inference"]["unit"] == "whole catalytic matrix")

    grain = fit_grain_model()
    with np.load(DEVELOPMENT_ARRAYS, allow_pickle=False) as archive:
        example = np.asarray(archive["renewal_state_graph"][:12], dtype=float)
    micro, macro = grain.classify("02", example)
    check("grain_dimensions", all(grain.components[candidate].shape == (4, 195) for candidate in CANDIDATES))
    check("grain_nested", bool(np.all(macro == micro // 4)))
    check("grain_ranges", bool(np.all((micro >= 0) & (micro < 16)) and np.all((macro >= 0) & (macro < 4))))

    rng = np.random.default_rng(_seed(smoke_spec(), "validation", "discrete"))
    past, future = _canonical_samples("downward_xor", 4096, rng)
    atoms = _exact_discrete_atoms(past, future)
    total = _mutual_information_discrete(past, future)
    check("discrete_atom_identity", abs(sum(atoms.values()) - total) <= 1e-10)
    check(
        "discrete_xor_signature",
        _signature_error("downward_xor", atoms) <= 1e-10,
    )

    continuous_past = rng.normal(size=(2, 2048))
    continuous_future = 0.4 * continuous_past + rng.normal(size=(2, 2048))
    continuous_atoms = local_phi_id_atoms(continuous_past, continuous_future)
    atom_sum = sum(np.mean(value) for value in continuous_atoms.values())
    whole = px9.gaussian_mutual_information(continuous_past, continuous_future)
    check(
        "continuous_atom_gaussian_crosscheck",
        np.isclose(atom_sum, whole, atol=1e-6, rtol=0),
        {"atom_sum": atom_sum, "gaussian_mi": whole},
    )

    config = GardConfig()
    beta = generate_beta(config, rng)
    composition = np.zeros(config.n_types, dtype=np.int64)
    composition[:40] = 1
    case = px9.ResilienceCase(
        "PX10-validation",
        "02",
        0,
        20,
        beta,
        px9.Snapshot(composition, 20, (True,) * 20, (0.95,) * 20),
        np.tile(composition, (32, 1)).astype(np.int16),
    )
    blocks = []
    for branch in range(4):
        daughters = np.tile(composition, (4, 1)).astype(np.int16)
        daughters[:, (branch + 1) % 40] += np.arange(4, dtype=np.int16)
        blocks.append(
            px9.PairBlock(
                daughters[:-1], daughters[1:], daughters[:-1], daughters[1:]
            )
        )
    payload = AtomCasePayload(case, {"Q100": composition}, {"Q100": tuple(blocks)})
    left, right, self_pairs = _lag_pairs(payload, "Q100", range(4), 2, 1)
    check("lag_pairs_match", left.shape == right.shape and len(left) > 0)
    check("lag_derangement_no_self", self_pairs == 0)
    check("lag_derangement_marginals", sorted(row.tobytes() for row in right) == sorted(row.tobytes() for row in _lag_pairs(payload, "Q100", range(4), 2)[1]))

    partitions = _matrix_random_partitions(beta, 0, smoke_spec())
    beta_parts = beta_physical_partition(beta)
    check("random_partition_count", len(partitions) == RANDOM_PARTITIONS)
    check("random_partition_size", all(len(first) == len(beta_parts[0]) for first, _ in partitions))

    check("constant_channel_capacity", abs(_binary_channel_capacity([0.4, 0.4, 0.4])) < 1e-10)
    check("deterministic_channel_capacity", abs(_binary_channel_capacity([0.0, 1.0]) - 1.0) < 1e-8)
    derangement = _derangement(6, np.random.default_rng(401))
    check(
        "arm_derangement",
        sorted(derangement.tolist()) == list(range(6))
        and bool(np.all(derangement != np.arange(6))),
    )
    check("model_hashes_present", MODEL_SOURCE.exists() and MODEL_CONTRACT_SOURCE.exists())
    check("development_hashes_present", DEVELOPMENT_ARRAYS.exists() and DEVELOPMENT_STATES.exists())
    check("source_files_present", all((ROOT / name).exists() for name in SOURCE_FILES))

    payload_out = {
        "format": "codex-ch5-phir-px10-validation-v1",
        "checks": checks,
        "passed": bool(checks and all(item["pass"] for item in checks)),
    }
    _atomic_json(output / "validation.json", payload_out)
    write_checksums(output)
    if not payload_out["passed"]:
        failed = [item["name"] for item in checks if not item["pass"]]
        raise AssertionError(f"PX10 validation failed: {failed}")
    return payload_out


def register(
    directory: Path = DEFAULT_REGISTRATION,
    calibration_directory: Path = DEFAULT_CALIBRATION,
) -> dict[str, Any]:
    if directory.exists():
        raise FileExistsError(directory)
    verify_checksums(DEFAULT_VALIDATION)
    verify_checksums(calibration_directory)
    calibration = json.loads((calibration_directory / "calibration.json").read_text())
    directory.mkdir(parents=True)
    model = fit_grain_model()
    grain_path = directory / "grain_model.npz"
    save_grain_model(model, grain_path)
    source_hashes = _source_hashes()
    proto = protocol()
    seed_registry = {
        "format": "codex-ch5-phir-px10-seed-registry-v1",
        "label": LABEL,
        "domains": SEED_DOMAINS,
        "fresh_from_px9": all(SEED_DOMAINS[key] != px9.SEED_DOMAINS.get(key) for key in SEED_DOMAINS),
    }
    model_contract = {
        "format": "codex-ch5-phir-px10-model-contract-v1",
        "renewal_student_sha256": sha256_file(MODEL_SOURCE),
        "renewal_contract_sha256": sha256_file(MODEL_CONTRACT_SOURCE),
        "development_arrays_sha256": sha256_file(DEVELOPMENT_ARRAYS),
        "development_states_sha256": sha256_file(DEVELOPMENT_STATES),
        "grain_model_sha256": sha256_file(grain_path),
        "outcome_used_for_grain": False,
    }
    registration_id = _digest(
        {
            "protocol": proto,
            "sources": source_hashes,
            "calibration": calibration,
            "seeds": seed_registry,
            "models": model_contract,
        }
    )
    registration = {
        "format": REGISTRATION_FORMAT,
        "registration_id": registration_id,
        "protocol_id": proto["protocol_id"],
        "source_hashes": source_hashes,
        "calibration_sha256": sha256_file(calibration_directory / "calibration.json"),
        "model_contract": model_contract,
        "registered_at_unix": time.time(),
        "scientific_matrices_generated": False,
    }
    shutil.copy2(ROOT / DOCUMENT, directory / "preregistration.md")
    _atomic_json(directory / "protocol.json", proto)
    _atomic_json(directory / "seed_registry.json", seed_registry)
    _atomic_json(directory / "model_contract.json", model_contract)
    _atomic_json(directory / "registration.json", registration)
    shutil.copy2(calibration_directory / "calibration.json", directory / "calibration.json")
    write_checksums(directory)
    return registration


def verify_registration(directory: Path = DEFAULT_REGISTRATION) -> dict[str, Any]:
    verify_checksums(directory)
    registration = json.loads((directory / "registration.json").read_text())
    if registration["format"] != REGISTRATION_FORMAT:
        raise ValueError("unexpected PX10 registration format")
    for name, expected in registration["source_hashes"].items():
        if sha256_file(ROOT / name) != expected:
            raise ValueError(f"PX10 sealed source changed: {name}")
    contract = registration["model_contract"]
    checks = {
        MODEL_SOURCE: contract["renewal_student_sha256"],
        MODEL_CONTRACT_SOURCE: contract["renewal_contract_sha256"],
        DEVELOPMENT_ARRAYS: contract["development_arrays_sha256"],
        DEVELOPMENT_STATES: contract["development_states_sha256"],
        directory / "grain_model.npz": contract["grain_model_sha256"],
    }
    for path, expected in checks.items():
        if sha256_file(path) != expected:
            raise ValueError(f"PX10 sealed model changed: {path}")
    if sha256_file(directory / "calibration.json") != registration["calibration_sha256"]:
        raise ValueError("PX10 calibration hash mismatch")
    return registration


def smoke(output: Path = DEFAULT_SMOKE) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    registration = verify_registration()
    spec = smoke_spec()
    arguments = (
        2,
        spec,
        str(MODEL_SOURCE),
        str(MODEL_CONTRACT_SOURCE),
        str(DEFAULT_REGISTRATION / "grain_model.npz"),
    )
    first = _run_matrix(arguments)
    second = _run_matrix(arguments)
    score_kinds = {str(row["score_kind"]) for row in first.score_rows}
    atom_kinds = {str(row["score_kind"]) for row in first.atom_rows}
    required_scores = {
        "paired_beta",
        "shuffled_beta",
        "random_partition",
        "cross_beta_partition",
        "public_revised",
        "concordant_outcome_beta",
    }
    required_atoms = {"paired_beta", "shuffled_beta", "random_partition"}
    projected_cpu = float(first.cpu_seconds * 2.0 * MATRICES * (BRANCHES / spec.branches) * (len(LANDMARKS) / len(spec.landmarks)) * (HORIZON / spec.horizon))
    payload = {
        "format": "codex-ch5-phir-px10-smoke-v1",
        "registration_id": registration["registration_id"],
        "exact_replay": first.scientific_digest == second.scientific_digest,
        "branches_created": len(first.branch_rows),
        "scores_created": len(first.score_rows),
        "atom_scores_created": len(first.atom_rows),
        "all_score_kinds_exercised": required_scores.issubset(score_kinds),
        "all_atom_kinds_exercised": required_atoms.issubset(atom_kinds),
        "all_arms_exercised": {str(row["arm"]) for row in first.edit_rows}
        == set(ARMS),
        "both_candidates_exercised": {
            str(row["candidate"]) for row in first.edit_rows
        }
        == set(CANDIDATES),
        "smoke_cpu_seconds": first.cpu_seconds,
        "conservative_projected_cpu_hours": projected_cpu / 3600.0,
        "projection_is_upper_bound": True,
    }
    payload["passed"] = bool(
        payload["exact_replay"]
        and payload["branches_created"] > 0
        and payload["scores_created"] > 0
        and payload["atom_scores_created"] > 0
        and payload["all_score_kinds_exercised"]
        and payload["all_atom_kinds_exercised"]
        and payload["all_arms_exercised"]
        and payload["both_candidates_exercised"]
    )
    output.mkdir(parents=True)
    _atomic_json(output / "smoke.json", payload)
    write_checksums(output)
    if not payload["passed"]:
        raise AssertionError("PX10 smoke failed")
    return payload


def _checkpoint_contract(registration_id: str, spec: PX10Spec) -> dict[str, Any]:
    return {
        "format": CHECKPOINT_FORMAT,
        "registration_id": registration_id,
        "spec": _json_ready(spec.__dict__),
        "grain_model_sha256": sha256_file(DEFAULT_REGISTRATION / "grain_model.npz"),
    }


def _status_write(work: Path, payload: Mapping[str, Any]) -> None:
    _atomic_json(work / "status.json", {"format": STATUS_FORMAT, **payload, "updated_at_unix": time.time()})


class _PX10Unpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module == "__main__" and name in {"PX10Batch", "PX10Spec", "AtomCasePayload", "GrainModel"}:
            return globals()[name]
        return super().find_class(module, name)


def _load_checkpoint(path: Path) -> PX10Batch:
    with path.open("rb") as handle:
        value = _PX10Unpickler(handle).load()
    if not isinstance(value, PX10Batch):
        raise TypeError(f"unexpected PX10 checkpoint type: {path}")
    return value


def _prepare_work(work: Path, registration_id: str, spec: PX10Spec) -> None:
    work.mkdir(parents=True, exist_ok=True)
    expected = _checkpoint_contract(registration_id, spec)
    path = work / "checkpoint_contract.json"
    if path.exists():
        if json.loads(path.read_text()) != expected:
            raise ValueError("PX10 checkpoint contract mismatch")
    else:
        _atomic_json(path, expected)


def _run_checkpoint_stage(
    work: Path,
    spec: PX10Spec,
    workers: int,
    stage: str,
    cpu_budget_seconds: float,
    prior_cpu_seconds: float = 0.0,
) -> list[PX10Batch]:
    directory = work / stage
    directory.mkdir(parents=True, exist_ok=True)
    batches: dict[int, PX10Batch] = {}
    for matrix_id in range(spec.matrices):
        path = directory / f"matrix_{matrix_id:03d}.pkl"
        if path.exists():
            batch = _load_checkpoint(path)
            if batch.matrix_id != matrix_id:
                raise ValueError("PX10 checkpoint matrix mismatch")
            batches[matrix_id] = batch
    consumed = prior_cpu_seconds + sum(batch.cpu_seconds for batch in batches.values())
    _status_write(
        work,
        {
            "state": "running",
            "stage": stage,
            "completed_matrices": len(batches),
            "total_matrices": spec.matrices,
            "cpu_seconds": consumed,
        },
    )
    pending = [matrix_id for matrix_id in range(spec.matrices) if matrix_id not in batches]
    arguments = [
        (
            matrix_id,
            spec,
            str(MODEL_SOURCE),
            str(MODEL_CONTRACT_SOURCE),
            str(DEFAULT_REGISTRATION / "grain_model.npz"),
        )
        for matrix_id in pending
    ]
    if workers == 1:
        results = ((_run_matrix(argument), argument[0]) for argument in arguments)
        for batch, matrix_id in results:
            _atomic_pickle(directory / f"matrix_{matrix_id:03d}.pkl", batch)
            batches[matrix_id] = batch
            consumed += batch.cpu_seconds
            _status_write(work, {"state": "running", "stage": stage, "completed_matrices": len(batches), "total_matrices": spec.matrices, "cpu_seconds": consumed})
            if consumed > cpu_budget_seconds:
                raise TimeoutError("PX10 CPU budget reached after checkpoint")
    elif arguments:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_run_matrix, argument): argument[0] for argument in arguments}
            for future in as_completed(futures):
                matrix_id = futures[future]
                batch = future.result()
                _atomic_pickle(directory / f"matrix_{matrix_id:03d}.pkl", batch)
                batches[matrix_id] = batch
                consumed += batch.cpu_seconds
                _status_write(work, {"state": "running", "stage": stage, "completed_matrices": len(batches), "total_matrices": spec.matrices, "cpu_seconds": consumed})
                if consumed > cpu_budget_seconds:
                    for item in futures:
                        item.cancel()
                    raise TimeoutError("PX10 CPU budget reached after checkpoint")
    if len(batches) != spec.matrices:
        raise AssertionError("PX10 checkpoint stage incomplete")
    return [batches[matrix_id] for matrix_id in range(spec.matrices)]


def _metric_table(items: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> list[str]:
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for item in items:
        values = []
        for column in columns:
            value = item.get(column, "")
            if isinstance(value, float):
                value = f"{value:+.5f}"
            values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return rows


def _reports(metrics: Mapping[str, Any], registration_id: str) -> tuple[str, str]:
    gates = metrics["gates"]
    scientific = [
        "# PX10 multiscale causal-information adjudication",
        "",
        f"Registration: `{registration_id}`.",
        "",
        "## Registered gates",
        "",
        *[f"- {key}: **{value}**" for key, value in gates.items()],
        "",
        "## Temporal echo",
        "",
        *_metric_table(metrics["temporal_response"], ("candidate", "source_half", "effect", "ci95", "holm_adjusted_p", "pass")),
        "",
        "## Atomic multiscale response",
        "",
        *_metric_table(metrics["atomic_temporal_response"], ("candidate", "source_half", "group", "effect", "ci95", "holm_adjusted_p", "pass")),
        "",
        "## Intervention channel",
        "",
        *_metric_table(metrics["intervention_channel"], ("candidate", "direction", "effect", "ci95", "holm_adjusted_p", "pass")),
        "",
        "## Claim boundary",
        "",
        "PX10 cannot make Phi-r causal, overwrite prior nulls, or establish consciousness, life, agency, biological memory, origin-of-life universality, Platonic space, or the Ruliad.",
    ]
    if gates["intervention_channel"]:
        channel_sentence = "The identity of the registered molecular edit dose carried reproducible held-out causal information about renewal."
    else:
        channel_sentence = "The registered held-out intervention-information channel did not pass every cell and control."
    if gates["temporal_echo_classification"] == "reliable_redundant_temporal_echo":
        echo_sentence = "The pairing-corrected score behaved as a reliable but behaviorally redundant echo of renewal."
    elif gates["temporal_authenticity"]:
        echo_sentence = "The pairing-corrected score was temporally authentic, but its incremental status remained nonredundant or unresolved."
    else:
        echo_sentence = "The pairing-corrected score did not confirm as a reliable temporal echo."
    atom_sentence = (
        "The calibrated multiscale atom fingerprint passed."
        if gates["atomic_temporal_fingerprint"]
        else "The multiscale atom fingerprint did not pass its complete calibrated gate."
    )
    lay = [
        "# Lay summary — PX10",
        "",
        "PX10 used one fresh, replayed experiment to ask whether information measurements genuinely track hereditary recovery and how much causal information is carried by tiny molecular edits.",
        "",
        echo_sentence,
        atom_sentence,
        channel_sentence,
        "",
        "These are measurements inside the Codex reconstruction. They do not imply consciousness, life, or a universal information law.",
    ]
    return "\n".join(scientific) + "\n", "\n".join(lay) + "\n"


def run(
    output: Path = DEFAULT_OUTPUT,
    work: Path = DEFAULT_WORK,
    workers: int = MAX_WORKERS,
    cpu_budget_hours: float = MAX_CPU_HOURS,
) -> dict[str, Any]:
    if workers < 1 or workers > MAX_WORKERS:
        raise ValueError(f"PX10 workers must be in [1,{MAX_WORKERS}]")
    if output.exists():
        raise FileExistsError(output)
    registration = verify_registration()
    verify_checksums(DEFAULT_SMOKE)
    verify_checksums(DEFAULT_CALIBRATION)
    spec = scientific_spec()
    _prepare_work(work, registration["registration_id"], spec)
    work.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(work).free < MINIMUM_FREE_DISK_BYTES:
        raise OSError("PX10 work volume lacks required free space")
    started = time.time()
    budget_seconds = cpu_budget_hours * 3600.0
    try:
        generated = _run_checkpoint_stage(work, spec, workers, "generation", budget_seconds)
        generation_cpu = float(sum(batch.cpu_seconds for batch in generated))
        replayed = _run_checkpoint_stage(work, spec, workers, "replay", budget_seconds, generation_cpu)
        replay_cpu = float(sum(batch.cpu_seconds for batch in replayed))
        replay_rows = [
            {
                "matrix_id": left.matrix_id,
                "generation_digest": left.scientific_digest,
                "replay_digest": right.scientific_digest,
                "exact": left.scientific_digest == right.scientific_digest,
            }
            for left, right in zip(generated, replayed, strict=True)
        ]
        replay_audit = {
            "format": "codex-ch5-phir-px10-replay-v1",
            "matrices": replay_rows,
            "complete_exact_replay": bool(len(replay_rows) == spec.matrices and all(item["exact"] for item in replay_rows)),
        }
        if not replay_audit["complete_exact_replay"]:
            raise AssertionError("PX10 exact replay failed")
        calibration = json.loads((DEFAULT_CALIBRATION / "calibration.json").read_text())
        metrics, tables, arrays = analyze_batches(generated, spec, calibration)
        staging = output.with_name(f".{output.name}.staging")
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        _atomic_json(staging / "primary_metrics.json", metrics)
        _atomic_json(staging / "replay_audit.json", replay_audit)
        for name, table in tables.items():
            table.to_csv(staging / f"{name}.csv.gz", index=False, compression="gzip")
        np.savez_compressed(staging / "inference_arrays.npz", **arrays)
        scientific_report, lay_summary = _reports(metrics, registration["registration_id"])
        (staging / "SCIENTIFIC_REPORT.md").write_text(scientific_report)
        (staging / "LAY_SUMMARY.md").write_text(lay_summary)
        claim_boundaries = {
            "supported": [key for key, value in metrics["gates"].items() if value is True],
            "failed_or_unresolved": [key for key, value in metrics["gates"].items() if value is False],
            "prohibited": ["Phi-r as cause", "consciousness", "life", "agency", "biological memory", "origin-of-life universality", "Platonic space", "Ruliad"],
        }
        _atomic_json(staging / "claim_boundaries.json", claim_boundaries)
        manifest = {
            "format": RESULT_FORMAT,
            "registration_id": registration["registration_id"],
            "matrices": spec.matrices,
            "branches_per_arm": spec.branches,
            "workers": workers,
            "generation_cpu_seconds": generation_cpu,
            "replay_cpu_seconds": replay_cpu,
            "wall_seconds": time.time() - started,
            "work_directory": str(work),
            "scientific_digest": _digest([batch.scientific_digest for batch in generated]),
            "gates": metrics["gates"],
        }
        _atomic_json(staging / "manifest.json", manifest)
        write_checksums(staging)
        verify_checksums(staging)
        staging.rename(output)
        _append_ledger(
            LEDGER,
            "PX10 multiscale causal-information adjudication",
            [
                f"Registration `{registration['registration_id']}`.",
                f"Temporal echo classification: `{metrics['gates']['temporal_echo_classification']}`.",
                f"Atomic fingerprint: `{metrics['gates']['atomic_temporal_fingerprint']}`.",
                f"Intervention channel: `{metrics['gates']['intervention_channel']}`.",
                "Complete exact replay passed for all 48 matrices.",
            ],
        )
        _status_write(work, {"state": "complete", "stage": "sealed", "completed_matrices": spec.matrices, "total_matrices": spec.matrices, "output": str(output), "gates": metrics["gates"]})
        return manifest
    except Exception as error:
        _status_write(work, {"state": "failed", "stage": "error", "error": f"{type(error).__name__}: {error}"})
        raise


def verify_result(directory: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    verify_checksums(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    replay = json.loads((directory / "replay_audit.json").read_text())
    if manifest["format"] != RESULT_FORMAT or not replay["complete_exact_replay"]:
        raise ValueError("PX10 result verification failed")
    return manifest


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def launch_detached(
    workers: int = MAX_WORKERS,
    cpu_budget_hours: float = MAX_CPU_HOURS,
    work: Path = DEFAULT_WORK,
) -> dict[str, Any]:
    registration = verify_registration()
    verify_checksums(DEFAULT_SMOKE)
    if DEFAULT_OUTPUT.exists():
        raise FileExistsError(DEFAULT_OUTPUT)
    work.mkdir(parents=True, exist_ok=True)
    launch_path = work / "detached_launch.json"
    if launch_path.exists():
        old = json.loads(launch_path.read_text())
        if _pid_alive(int(old.get("pid", -1))):
            raise RuntimeError(f"PX10 already runs as PID {old['pid']}")
    command = [
        sys.executable,
        "-m",
        "plastic_heredity.phir_extension_px10",
        "run",
        "--workers",
        str(workers),
        "--cpu-budget-hours",
        str(cpu_budget_hours),
        "--work-dir",
        str(work),
    ]
    with DEFAULT_LOG.open("ab", buffering=0) as handle:
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    payload = {
        "format": "codex-ch5-phir-px10-detached-launch-v1",
        "registration_id": registration["registration_id"],
        "pid": process.pid,
        "workers": workers,
        "cpu_budget_hours": cpu_budget_hours,
        "work": str(work),
        "log": str(DEFAULT_LOG),
        "command": command,
        "launched_at_unix": time.time(),
    }
    _atomic_json(launch_path, payload)
    return payload


def status(work: Path = DEFAULT_WORK) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "format": "codex-ch5-phir-px10-status-report-v1",
        "validation": DEFAULT_VALIDATION.exists(),
        "calibration": DEFAULT_CALIBRATION.exists(),
        "registration": DEFAULT_REGISTRATION.exists(),
        "smoke": DEFAULT_SMOKE.exists(),
        "output": DEFAULT_OUTPUT.exists(),
    }
    launch = work / "detached_launch.json"
    if launch.exists():
        value = json.loads(launch.read_text())
        value["alive"] = _pid_alive(int(value.get("pid", -1)))
        payload["launch"] = value
    state = work / "status.json"
    if state.exists():
        payload["work_status"] = json.loads(state.read_text())
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate")
    sub.add_parser("calibrate")
    sub.add_parser("register")
    sub.add_parser("smoke")
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    run_parser.add_argument("--cpu-budget-hours", type=float, default=MAX_CPU_HOURS)
    run_parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    launch_parser = sub.add_parser("launch")
    launch_parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    launch_parser.add_argument("--cpu-budget-hours", type=float, default=MAX_CPU_HOURS)
    launch_parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    status_parser = sub.add_parser("status")
    status_parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK)
    sub.add_parser("verify")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate":
        value = validate()
    elif args.command == "calibrate":
        value = run_calibration()
    elif args.command == "register":
        value = register()
    elif args.command == "smoke":
        value = smoke()
    elif args.command == "run":
        value = run(work=args.work_dir, workers=args.workers, cpu_budget_hours=args.cpu_budget_hours)
    elif args.command == "launch":
        value = launch_detached(args.workers, args.cpu_budget_hours, args.work_dir)
    elif args.command == "status":
        value = status(args.work_dir)
    elif args.command == "verify":
        value = verify_result()
    else:
        raise AssertionError(args.command)
    print(json.dumps(_json_ready(value), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
