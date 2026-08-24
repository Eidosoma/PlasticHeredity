"""Shared, additive contracts for the Chapter 5 Phi-r extension.

The sealed Chapter 5 modules are deliberately not modified.  This module
provides prospective sequence and explicit-transition-pair instruments,
functional flux features, matrix-level inference, and durable I/O helpers for
the new PX phases.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy
import sklearn
from numpy.typing import NDArray

from .config import GardConfig
from .mechanistic_metrics import holm_adjust
from .phir_instruments import (
    ANTICHAINS,
    ATOM_NAMES,
    PHIR_ATOMS,
    SYNERGISTIC,
    UNIQUE_0,
    UNIQUE_1,
    gaussian_mutual_information,
)
from .phir_rescue_instruments import (
    _cached_local_phi_id_atoms,
    active_partition,
    beta_physical_partition,
    close_all_clr,
    full_block_revised,
    macro_phi_score,
    rank_gaussianize,
)
from .seeds import derive_seed


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results" / "phir_extension"
MASTER_DOCUMENT = ROOT / "CODEX_CH5_PHIR_EXTENSION_PREREGISTRATION.md"
MASTER_REGISTRATION = RESULT_ROOT / "registration"

BOOTSTRAP_DRAWS = 4096
RANDOMIZATION_DRAWS = 4096
MAX_WORKERS = 12
CPU_BUDGET_HOURS = 80.0
MINIMUM_FREE_DISK_BYTES = 1_500_000_000

PROGRAM_LABEL = "CODEX_CH5_PHIR_EXTENSION_V1"
PROGRAM_FORMAT = "codex-ch5-phir-extension-program-v1"
MASTER_REGISTRATION_FORMAT = "codex-ch5-phir-extension-registration-v1"


def canonical_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): canonical_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def canonical_digest(value: Any) -> str:
    payload = json.dumps(
        canonical_json(value), sort_keys=True, separators=(",", ":"), allow_nan=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(canonical_json(value), sort_keys=True, indent=2, allow_nan=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=5)
    temporary.replace(path)


def runtime_versions() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
    }


def seed_domain(name: str) -> str:
    return hashlib.sha256(f"{PROGRAM_LABEL}::{name}".encode("utf-8")).hexdigest()


SEED_DOMAINS = {
    name: seed_domain(name)
    for name in (
        "matrix",
        "initial",
        "main_path",
        "action",
        "random_action",
        "future",
        "acquisition",
        "screen",
        "bootstrap",
        "randomization",
        "replay",
        "smoke",
        "validation",
    )
}


def purpose_seed(domain: str, phase: str, *keys: object) -> int:
    if domain not in SEED_DOMAINS:
        raise KeyError(f"unknown Phi-r extension seed domain: {domain}")
    return derive_seed(SEED_DOMAINS[domain], PROGRAM_LABEL, phase, *keys)


@dataclass(frozen=True)
class InformationScore:
    """Full-block and public macro-PhiID readings for one sample."""

    full_revised: float
    full_base: float
    whole_mi: float
    aa_mi: float
    ab_mi: float
    ba_mi: float
    bb_mi: float
    double_redundancy: float
    public_revised: float
    causation: float
    emergence: float
    synergy_persistence: float
    atoms: FloatArray
    active_dimensions: int
    part_a_dimensions: int
    part_b_dimensions: int
    transitions: int

    def fields(self, prefix: str) -> dict[str, Any]:
        output: dict[str, Any] = {
            f"{prefix}_full_revised": self.full_revised,
            f"{prefix}_full_base": self.full_base,
            f"{prefix}_whole_mi": self.whole_mi,
            f"{prefix}_aa_mi": self.aa_mi,
            f"{prefix}_ab_mi": self.ab_mi,
            f"{prefix}_ba_mi": self.ba_mi,
            f"{prefix}_bb_mi": self.bb_mi,
            f"{prefix}_double_redundancy": self.double_redundancy,
            f"{prefix}_public_revised": self.public_revised,
            f"{prefix}_causation": self.causation,
            f"{prefix}_emergence": self.emergence,
            f"{prefix}_synergy_persistence": self.synergy_persistence,
            f"{prefix}_active_dimensions": self.active_dimensions,
            f"{prefix}_part_a_dimensions": self.part_a_dimensions,
            f"{prefix}_part_b_dimensions": self.part_b_dimensions,
            f"{prefix}_transitions": self.transitions,
        }
        output.update(
            {
                f"{prefix}_atom_{name}": float(value)
                for name, value in zip(ATOM_NAMES, self.atoms, strict=True)
            }
        )
        return output


def nan_information_score(transitions: int = 0) -> InformationScore:
    return InformationScore(
        *(float("nan"),) * 12,
        atoms=np.full(len(ATOM_NAMES), np.nan, dtype=np.float64),
        active_dimensions=0,
        part_a_dimensions=0,
        part_b_dimensions=0,
        transitions=int(transitions),
    )


def _macro_from_pairs(
    past: FloatArray,
    future: FloatArray,
    part_a: IntArray,
    part_b: IntArray,
) -> tuple[float, float, float, float, FloatArray]:
    past_macro = np.vstack(
        (past[part_a].mean(axis=0), past[part_b].mean(axis=0))
    )
    future_macro = np.vstack(
        (future[part_a].mean(axis=0), future[part_b].mean(axis=0))
    )
    local = _cached_local_phi_id_atoms(past_macro, future_macro)
    means = {atom: float(np.mean(values)) for atom, values in local.items()}
    atoms = np.asarray(
        [means[(source, target)] for source in ANTICHAINS for target in ANTICHAINS],
        dtype=np.float64,
    )
    revised = float(sum(means[atom] for atom in PHIR_ATOMS))
    synergy = float(means[(SYNERGISTIC, SYNERGISTIC)])
    causation = float(
        means[(SYNERGISTIC, UNIQUE_0)] + means[(SYNERGISTIC, UNIQUE_1)]
    )
    return revised, causation, causation + synergy, synergy, atoms


def _score_transformed_pairs(
    past: FloatArray,
    future: FloatArray,
    part_a: IntArray,
    part_b: IntArray,
) -> InformationScore:
    left = np.asarray(past, dtype=np.float64)
    right = np.asarray(future, dtype=np.float64)
    first = np.asarray(part_a, dtype=np.int64)
    second = np.asarray(part_b, dtype=np.int64)
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] < 2:
        raise ValueError("explicit Phi pairs require matching dimensions and >=2 pairs")
    if not first.size or not second.size:
        raise ValueError("both Phi partitions must be nonempty")
    whole = gaussian_mutual_information(left, right)
    aa = gaussian_mutual_information(left[first], right[first])
    ab = gaussian_mutual_information(left[first], right[second])
    ba = gaussian_mutual_information(left[second], right[first])
    bb = gaussian_mutual_information(left[second], right[second])
    redundancy = min(aa, ab, ba, bb)
    public, causation, emergence, synergy, atoms = _macro_from_pairs(
        left, right, first, second
    )
    base = whole - aa - bb
    return InformationScore(
        full_revised=float(base + redundancy),
        full_base=float(base),
        whole_mi=float(whole),
        aa_mi=float(aa),
        ab_mi=float(ab),
        ba_mi=float(ba),
        bb_mi=float(bb),
        double_redundancy=float(redundancy),
        public_revised=public,
        causation=causation,
        emergence=emergence,
        synergy_persistence=synergy,
        atoms=atoms,
        active_dimensions=int(left.shape[0]),
        part_a_dimensions=int(first.size),
        part_b_dimensions=int(second.size),
        transitions=int(left.shape[1]),
    )


def _material_transform(
    observations: NDArray,
    beta: NDArray,
) -> tuple[FloatArray, IntArray, IntArray]:
    counts = np.asarray(observations, dtype=np.float64)
    if counts.ndim != 2 or counts.shape[0] < 3:
        raise ValueError("material information requires >=3 observations by types")
    data, active = rank_gaussianize(close_all_clr(counts))
    physical_a, physical_b = beta_physical_partition(beta)
    first, second = active_partition(active, physical_a, physical_b)
    return data, first, second


def expected_flux_observations(
    observations: NDArray,
    beta: NDArray,
    config: GardConfig | None = None,
) -> FloatArray:
    """Return module-A join/leave and module-B join/leave expected fluxes."""

    cfg = GardConfig() if config is None else config
    counts = np.asarray(observations, dtype=np.float64)
    matrix = np.asarray(beta, dtype=np.float64)
    if counts.ndim != 2 or counts.shape[1] != cfg.n_types:
        raise ValueError("flux observations must be observations by n_types")
    if matrix.shape != (cfg.n_types, cfg.n_types):
        raise ValueError("beta has the wrong shape for flux observations")
    masses = counts.sum(axis=1)
    if np.any(masses <= 0.0):
        raise ValueError("functional flux is undefined for an empty assembly")
    boost = 1.0 + (counts @ matrix.T) / masses[:, None]
    join = cfg.k_join * (1.0 / cfg.n_types) * masses[:, None] * boost
    leave = cfg.k_leave * counts * boost
    module_a, module_b = beta_physical_partition(matrix)
    return np.column_stack(
        (
            join[:, module_a].sum(axis=1),
            leave[:, module_a].sum(axis=1),
            join[:, module_b].sum(axis=1),
            leave[:, module_b].sum(axis=1),
        )
    ).astype(np.float64)


def _functional_transform(
    observations: NDArray,
    beta: NDArray,
    config: GardConfig | None = None,
) -> tuple[FloatArray, IntArray, IntArray]:
    flux = expected_flux_observations(observations, beta, config)
    data, active = rank_gaussianize(np.log1p(flux).T)
    lookup = {int(original): index for index, original in enumerate(active)}
    first = np.asarray([lookup[index] for index in (0, 1) if index in lookup], dtype=np.int64)
    second = np.asarray([lookup[index] for index in (2, 3) if index in lookup], dtype=np.int64)
    if not first.size or not second.size or first.size + second.size != active.size:
        raise ValueError("functional flux lost a complete active module")
    return data, first, second


def score_sequence(
    observations: NDArray,
    beta: NDArray,
    representation: str,
    config: GardConfig | None = None,
) -> InformationScore:
    """Score adjacent transitions after transforming each unique state once."""

    if representation == "material":
        data, first, second = _material_transform(observations, beta)
    elif representation == "functional_flux":
        data, first, second = _functional_transform(observations, beta, config)
    else:
        raise ValueError(f"unknown information representation: {representation}")
    full = full_block_revised(data, first, second)
    macro = macro_phi_score(data, first, second)
    return InformationScore(
        full_revised=full.revised,
        full_base=float(full.whole_mi - full.aa_mi - full.bb_mi),
        whole_mi=full.whole_mi,
        aa_mi=full.aa_mi,
        ab_mi=full.ab_mi,
        ba_mi=full.ba_mi,
        bb_mi=full.bb_mi,
        double_redundancy=full.double_redundancy,
        public_revised=macro.revised,
        causation=macro.causation,
        emergence=macro.emergence,
        synergy_persistence=macro.synergy_persistence,
        atoms=macro.atoms,
        active_dimensions=int(data.shape[0]),
        part_a_dimensions=int(first.size),
        part_b_dimensions=int(second.size),
        transitions=int(data.shape[1] - 1),
    )


def score_explicit_pairs(
    past_observations: NDArray,
    future_observations: NDArray,
    beta: NDArray,
    representation: str,
    config: GardConfig | None = None,
) -> InformationScore:
    """Score explicit transitions without inventing cross-branch transitions."""

    past = np.asarray(past_observations)
    future = np.asarray(future_observations)
    if past.shape != future.shape or past.ndim != 2 or past.shape[0] < 2:
        raise ValueError("past and future must be matching pair-by-coordinate arrays")
    combined = np.vstack((past, future))
    if representation == "material":
        data, first, second = _material_transform(combined, beta)
    elif representation == "functional_flux":
        data, first, second = _functional_transform(combined, beta, config)
    else:
        raise ValueError(f"unknown information representation: {representation}")
    pairs = past.shape[0]
    return _score_transformed_pairs(data[:, :pairs], data[:, pairs:], first, second)


def paired_matrix_effects(
    frame: pd.DataFrame,
    metric: str,
    high: str,
    low: str,
    *,
    filters: Mapping[str, Any],
    within: Sequence[str] = (),
) -> pd.Series:
    selected = frame.copy()
    for column, value in filters.items():
        selected = selected[selected[column] == value]
    group = ["matrix_id", *within, "arm"]
    means = selected.groupby(group, sort=True)[metric].mean().unstack("arm")
    if high not in means.columns or low not in means.columns:
        return pd.Series(dtype=float)
    differences = means[high] - means[low]
    if within:
        differences = differences.groupby(level="matrix_id").mean()
    return differences.dropna().sort_index()


def paired_summary(
    values: Sequence[float] | NDArray,
    key: str,
    *,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
    randomization_draws: int = RANDOMIZATION_DRAWS,
    equivalence_margin: float | None = None,
) -> tuple[dict[str, Any], dict[str, FloatArray]]:
    vector = np.asarray(values, dtype=np.float64)
    vector = vector[np.isfinite(vector)]
    if vector.size:
        boot_rng = np.random.default_rng(purpose_seed("bootstrap", "inference", key))
        indices = boot_rng.integers(0, vector.size, size=(bootstrap_draws, vector.size))
        bootstrap = vector[indices].mean(axis=1)
        random_rng = np.random.default_rng(
            purpose_seed("randomization", "inference", key)
        )
        signs = random_rng.choice((-1.0, 1.0), size=(randomization_draws, vector.size))
        randomized = (signs * vector).mean(axis=1)
        observed = float(vector.mean())
        positive_p = float(
            (1 + np.count_nonzero(randomized >= observed))
            / (randomization_draws + 1)
        )
        negative_p = float(
            (1 + np.count_nonzero(randomized <= observed))
            / (randomization_draws + 1)
        )
        ci95 = np.quantile(bootstrap, (0.025, 0.975))
        ci90 = np.quantile(bootstrap, (0.05, 0.95))
        loo = (
            (vector.sum() - vector) / (vector.size - 1)
            if vector.size > 1
            else np.asarray([np.nan])
        )
    else:
        bootstrap = np.full(bootstrap_draws, np.nan)
        randomized = np.full(randomization_draws, np.nan)
        observed = positive_p = negative_p = float("nan")
        ci95 = ci90 = np.asarray((np.nan, np.nan))
        loo = np.asarray([np.nan])
    result: dict[str, Any] = {
        "effect": observed,
        "ci95": [float(ci95[0]), float(ci95[1])],
        "ci90": [float(ci90[0]), float(ci90[1])],
        "positive_sign_randomization_p": positive_p,
        "negative_sign_randomization_p": negative_p,
        "two_sided_sign_randomization_p": float(
            min(1.0, 2.0 * min(positive_p, negative_p))
        ),
        "matrices": int(vector.size),
        "matrices_positive": int(np.count_nonzero(vector > 0.0)),
        "matrices_negative": int(np.count_nonzero(vector < 0.0)),
        "maximum_absolute_matrix_effect": (
            float(np.max(np.abs(vector))) if vector.size else float("nan")
        ),
        "leave_one_matrix_out_all_positive": bool(
            vector.size > 1 and np.all(loo > 0.0)
        ),
    }
    if equivalence_margin is not None:
        result.update(
            {
                "equivalence_margin": float(equivalence_margin),
                "tost_via_90ci": bool(
                    np.isfinite(ci90).all()
                    and ci90[0] > -equivalence_margin
                    and ci90[1] < equivalence_margin
                ),
            }
        )
    return result, {
        "matrix_values": vector,
        "bootstrap": np.asarray(bootstrap, dtype=np.float64),
        "sign_randomization": np.asarray(randomized, dtype=np.float64),
        "leave_one_out": np.asarray(loo, dtype=np.float64),
    }


def apply_holm(
    rows: Sequence[dict[str, Any]],
    source: str = "positive_sign_randomization_p",
    destination: str = "holm_adjusted_p",
) -> None:
    finite = [row for row in rows if np.isfinite(row.get(source, np.nan))]
    if not finite:
        return
    adjusted = holm_adjust([float(row[source]) for row in finite])
    for row, value in zip(finite, adjusted, strict=True):
        row[destination] = float(value)


def matrix_block_hotelling(
    values: NDArray,
    key: str,
    *,
    draws: int = RANDOMIZATION_DRAWS,
) -> tuple[dict[str, Any], dict[str, FloatArray]]:
    """Paired whole-matrix sign-randomized ridge Hotelling statistic."""

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or not np.isfinite(matrix).all():
        raise ValueError("Hotelling input must be finite matrices by components")

    def statistic(sample: FloatArray) -> float:
        mean = sample.mean(axis=0)
        covariance = np.atleast_2d(np.cov(sample, rowvar=False, ddof=0))
        ridge = 1e-6 * max(float(np.trace(covariance)) / covariance.shape[0], 1e-12)
        inverse = np.linalg.pinv(covariance + np.eye(covariance.shape[0]) * ridge)
        return float(mean @ inverse @ mean)

    observed = statistic(matrix)
    rng = np.random.default_rng(purpose_seed("randomization", "hotelling", key))
    null = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        signs = rng.choice((-1.0, 1.0), size=matrix.shape[0])
        null[index] = statistic(matrix * signs[:, None])
    p_value = float((1 + np.count_nonzero(null >= observed)) / (draws + 1))
    return (
        {
            "statistic": observed,
            "randomization_p": p_value,
            "matrices": int(matrix.shape[0]),
            "components": int(matrix.shape[1]),
        },
        {"matrix_vectors": matrix, "sign_randomization": null},
    )


def master_protocol() -> dict[str, Any]:
    value: dict[str, Any] = {
        "format": PROGRAM_FORMAT,
        "title": "Chapter 5 Phi-r extension",
        "document_sha256": sha256_file(MASTER_DOCUMENT),
        "matrix_scale": 24,
        "no_48_matrix_campaign": True,
        "run_all_phases_without_evidence_gating": True,
        "max_workers": MAX_WORKERS,
        "cpu_budget_hours": CPU_BUDGET_HOURS,
        "minimum_free_disk_bytes": MINIMUM_FREE_DISK_BYTES,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "randomization_draws": RANDOMIZATION_DRAWS,
        "inference_unit": "whole catalytic matrix",
        "candidates_pooled": False,
        "phase_allocations_cpu_hours": {
            "PX0": 2,
            "PX1": 8,
            "PX2": 10,
            "PX3_DEVELOPMENT": 20,
            "PX3_CONFIRMATION": 12,
            "PX4": 14,
            "PX5": 12,
            "PX6": 2,
        },
        "phases": ["PX1", "PX2", "PX3", "PX4", "PX5", "PX6"],
        "seed_domains": SEED_DOMAINS,
        "claim_boundaries": [
            "prior sealed negative results remain unchanged",
            "no consciousness, agency, life, or metaphysical inference",
            "no 48-matrix continuation",
            "cross-clean-room claims require exactly matched tests",
        ],
    }
    value["protocol_id"] = canonical_digest(value)
    return value


def register_master(source_files: Sequence[str]) -> dict[str, Any]:
    if MASTER_REGISTRATION.exists():
        raise FileExistsError(f"master registration exists: {MASTER_REGISTRATION}")
    hashes = {name: sha256_file(ROOT / name) for name in source_files}
    body: dict[str, Any] = {
        "format": MASTER_REGISTRATION_FORMAT,
        "protocol": master_protocol(),
        "source_hashes": hashes,
        "runtime": runtime_versions(),
        "new_scientific_matrices_at_registration": 0,
    }
    body["registration_id"] = canonical_digest(body)
    MASTER_REGISTRATION.mkdir(parents=True, exist_ok=False)
    atomic_json(MASTER_REGISTRATION / "protocol.json", body["protocol"])
    atomic_json(MASTER_REGISTRATION / "seed_registry.json", SEED_DOMAINS)
    atomic_json(MASTER_REGISTRATION / "registration.json", body)
    return body


def verify_master(source_files: Sequence[str]) -> dict[str, Any]:
    body = json.loads(
        (MASTER_REGISTRATION / "registration.json").read_text(encoding="utf-8")
    )
    observed = body.pop("registration_id")
    if body.get("format") != MASTER_REGISTRATION_FORMAT:
        raise ValueError("unsupported Phi-r extension master registration")
    if observed != canonical_digest(body):
        raise ValueError("Phi-r extension master registration identity failed")
    body["registration_id"] = observed
    if body["protocol"] != canonical_json(master_protocol()):
        raise ValueError("Phi-r extension master protocol changed")
    expected = {name: sha256_file(ROOT / name) for name in source_files}
    if body["source_hashes"] != expected:
        raise ValueError("Phi-r extension master source hashes changed")
    return body


def dataclass_digest(value: Any, *, blank_field: str = "scientific_digest") -> str:
    fields = asdict(value)
    if blank_field in fields:
        fields[blank_field] = ""
    return canonical_digest(fields)


def safe_score_sequence(
    observations: NDArray,
    beta: NDArray,
    representation: str,
    config: GardConfig | None = None,
) -> InformationScore:
    try:
        return score_sequence(observations, beta, representation, config)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        transitions = max(0, int(np.asarray(observations).shape[0]) - 1)
        return nan_information_score(transitions)


def safe_score_pairs(
    past: NDArray,
    future: NDArray,
    beta: NDArray,
    representation: str,
    config: GardConfig | None = None,
) -> InformationScore:
    try:
        return score_explicit_pairs(past, future, beta, representation, config)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        return nan_information_score(int(np.asarray(past).shape[0]))
