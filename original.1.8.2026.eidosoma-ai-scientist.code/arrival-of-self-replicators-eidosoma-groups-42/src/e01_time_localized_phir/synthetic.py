"""Deterministic known-truth and dimensional stress fixtures for E01 S11."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from scipy.special import ndtri
from scipy.stats import qmc

from e01_gard_reproducibility import (
    CANONICAL_STREAM_PURPOSES,
    CouplingPolicy,
    SeedRequest,
    StreamPurpose,
    derive_seed_bundle,
    isolated_stream_namespace,
)
from e01_information_dynamics.validation import gaussian_mmi_oracle

from .estimator import decompose_local_entropies, population_local_entropies

ROOT_SEED_HEX = "11" * 32
BURN_IN = 512


@dataclass(frozen=True, slots=True)
class Fixture:
    system_id: str
    data: np.ndarray
    seed_record: dict[str, Any]
    planted_part_a: tuple[int, ...] | None = None


def estimator_rng(
    *,
    domain: str,
    pair_id: str,
    replicate_index: int,
    dimension: int | None = None,
) -> tuple[np.random.Generator, dict[str, Any]]:
    """Derive one S06-domain-separated estimator stream for an S11 fixture."""

    specification_id = f"E01-S11-{domain}"
    trajectory_id = pair_id if dimension is None else f"{pair_id}-D{dimension:03d}"
    namespace = isolated_stream_namespace(
        experiment_id="E01",
        specification_id=specification_id,
        trajectory_id=trajectory_id,
        replicate_index=replicate_index,
    )
    request = SeedRequest(
        experiment_id="E01",
        specification_id=specification_id,
        trajectory_id=trajectory_id,
        replicate_index=replicate_index,
        engine_id="e01_time_localized_phir",
        root_seed_hex=ROOT_SEED_HEX,
        coupling_policy=CouplingPolicy.TRAJECTORY_ISOLATED,
        coupling_reason=None,
        stream_namespaces={purpose: namespace for purpose in CANONICAL_STREAM_PURPOSES},
    )
    bundle = derive_seed_bundle(request)
    stream = bundle.streams[StreamPurpose.ESTIMATOR]
    record = {
        "domain": domain,
        "pairId": pair_id,
        "dimension": dimension,
        "replicateIndex": replicate_index,
        **stream.to_payload(),
    }
    return stream.generator(), record


def independent_white(
    *, pair_id: str, replicate_index: int, length: int, domain: str
) -> Fixture:
    rng, record = estimator_rng(
        domain=domain, pair_id=pair_id, replicate_index=replicate_index
    )
    return Fixture(
        "E01-S11-SYS-INDEPENDENT-WHITE-GAUSSIAN-v1.0.0",
        rng.standard_normal((length, 2), dtype=np.float64),
        record,
    )


def noisy_redundant_ar(
    *, pair_id: str, replicate_index: int, length: int, domain: str
) -> Fixture:
    rng, record = estimator_rng(
        domain=domain, pair_id=pair_id, replicate_index=replicate_index
    )
    latent = np.zeros(length + BURN_IN, dtype=np.float64)
    innovations = rng.standard_normal(latent.size)
    for index in range(1, latent.size):
        latent[index] = 0.80 * latent[index - 1] + innovations[index]
    data = latent[BURN_IN:, None] + 0.35 * rng.standard_normal((length, 2))
    return Fixture("E01-S11-SYS-NOISY-REDUNDANT-AR-v1.0.0", data, record)


_DIRECTIONAL_TRANSITION = np.asarray([[0.10, 0.00], [0.75, 0.20]], dtype=np.float64)
_DIRECTIONAL_INNOVATION_SD = np.asarray([1.0, 0.5], dtype=np.float64)


def directional_var(
    *, pair_id: str, replicate_index: int, length: int, domain: str
) -> Fixture:
    rng, record = estimator_rng(
        domain=domain, pair_id=pair_id, replicate_index=replicate_index
    )
    data = np.zeros((length + BURN_IN, 2), dtype=np.float64)
    innovations = rng.standard_normal(data.shape) * _DIRECTIONAL_INNOVATION_SD
    for index in range(1, data.shape[0]):
        data[index] = _DIRECTIONAL_TRANSITION @ data[index - 1] + innovations[index]
    return Fixture("E01-S11-SYS-DIRECTIONAL-VAR-v1.0.0", data[BURN_IN:], record)


def redundant_covariance(tau: int) -> np.ndarray:
    latent_variance = 1.0 / (1.0 - 0.80**2)
    same = latent_variance + 0.35**2
    cross_same = latent_variance
    cross_lag = (0.80**tau) * latent_variance
    return np.asarray(
        [
            [same, cross_same, cross_lag, cross_lag],
            [cross_same, same, cross_lag, cross_lag],
            [cross_lag, cross_lag, same, cross_same],
            [cross_lag, cross_lag, cross_same, same],
        ],
        dtype=np.float64,
    )


def directional_covariance(tau: int) -> np.ndarray:
    stationary = solve_discrete_lyapunov(
        _DIRECTIONAL_TRANSITION, np.diag(_DIRECTIONAL_INNOVATION_SD**2)
    )
    past_future = stationary @ np.linalg.matrix_power(_DIRECTIONAL_TRANSITION, tau).T
    return np.block([[stationary, past_future], [past_future.T, stationary]])


def mmi_truth(system_id: str, tau: int) -> dict[str, Any]:
    if system_id == "E01-S11-SYS-NOISY-REDUNDANT-AR-v1.0.0":
        covariance = redundant_covariance(tau)
    elif system_id == "E01-S11-SYS-DIRECTIONAL-VAR-v1.0.0":
        covariance = directional_covariance(tau)
    elif system_id == "E01-S11-SYS-INDEPENDENT-WHITE-GAUSSIAN-v1.0.0":
        covariance = np.eye(4)
    else:
        raise ValueError(f"Unknown scalar system {system_id!r}.")
    return gaussian_mmi_oracle(covariance)


def ccs_population_oracle(
    covariance: np.ndarray,
    *,
    scramble_seed: int,
    power: int = 18,
) -> dict[str, Any]:
    """Approximate the population CCS atom means with a frozen Sobol design."""

    covariance = np.asarray(covariance, dtype=np.float64)
    sampler = qmc.Sobol(d=4, scramble=True, seed=scramble_seed)
    uniforms = sampler.random_base2(power)
    epsilon = np.finfo(np.float64).eps
    normals = ndtri(np.clip(uniforms, epsilon, 1.0 - epsilon))
    samples = normals @ np.linalg.cholesky(covariance).T
    entropies = population_local_entropies(samples, covariance)
    atoms, mi, _, _ = decompose_local_entropies(entropies, redundancy="CCS")
    atom_means = {key: float(np.mean(value)) for key, value in atoms.items()}
    mi_means = {key: float(np.mean(value)) for key, value in mi.items()}
    past_redundancy = float(sum(atom_means[key] for key in ("rtr", "rtx", "rty", "rts")))
    past_synergy = float(sum(atom_means[key] for key in ("str", "stx", "sty", "sts")))
    return {
        "atomMeans": atom_means,
        "miMeans": mi_means,
        "totalMi": mi_means["I_xytab"],
        "paperEquationAggregate": past_synergy - past_redundancy,
        "sobolPower": power,
        "drawCount": 2**power,
        "scrambleSeed": scramble_seed,
    }


def planted_two_block_ar(
    *,
    pair_id: str,
    replicate_index: int,
    length: int,
    dimension: int,
    domain: str = "highdim-signal",
) -> Fixture:
    if dimension not in (8, 99, 100):
        raise ValueError("The frozen planted fixture dimensions are 8, 99, and 100.")
    rng, record = estimator_rng(
        domain=domain,
        pair_id=pair_id,
        replicate_index=replicate_index,
        dimension=dimension,
    )
    first_size = dimension // 2
    latent = np.zeros((length + BURN_IN, 2), dtype=np.float64)
    innovations = rng.standard_normal(latent.shape)
    for index in range(1, latent.shape[0]):
        latent[index] = 0.80 * latent[index - 1] + innovations[index]
    latent = latent[BURN_IN:]
    membership = np.asarray([0] * first_size + [1] * (dimension - first_size))
    data = latent[:, membership] + 0.10 * rng.standard_normal((length, dimension))
    return Fixture(
        "E01-S11-SYS-PLANTED-TWO-BLOCK-AR-v1.0.0",
        data,
        record,
        tuple(range(first_size)),
    )


def highdim_independent_null(
    *,
    pair_id: str,
    replicate_index: int,
    length: int,
    dimension: int,
) -> Fixture:
    rng, record = estimator_rng(
        domain="highdim-null",
        pair_id=pair_id,
        replicate_index=replicate_index,
        dimension=dimension,
    )
    data = np.zeros((length + BURN_IN, dimension), dtype=np.float64)
    innovations = rng.standard_normal(data.shape)
    for index in range(1, data.shape[0]):
        data[index] = 0.20 * data[index - 1] + innovations[index]
    return Fixture(
        "E01-S11-SYS-HIGHDIM-INDEPENDENT-NULL-v1.0.0",
        data[BURN_IN:],
        record,
    )


def piecewise_block_ar(*, dimension: int = 100, length: int = 2048) -> Fixture:
    """Generate the frozen changing-membership prospective-history fixture."""

    if dimension != 100 or length != 2048:
        raise ValueError("The frozen piecewise fixture is exactly D=100 and T=2048.")
    rng, record = estimator_rng(
        domain="dynamic-history",
        pair_id="E01-S11-DYNAMIC",
        replicate_index=0,
        dimension=dimension,
    )
    latent = np.zeros((length + BURN_IN, 2), dtype=np.float64)
    innovations = rng.standard_normal(latent.shape)
    for index in range(1, latent.shape[0]):
        latent[index] = 0.80 * latent[index - 1] + innovations[index]
    latent = latent[BURN_IN:]
    before = np.asarray([0] * 50 + [1] * 50)
    after = before.copy()
    after[25:50] = 1
    after[75:100] = 0
    membership = np.tile(before, (length, 1))
    membership[1024:] = after
    row = np.arange(length)[:, None]
    data = latent[row, membership] + 0.10 * rng.standard_normal((length, dimension))
    return Fixture(
        "E01-S11-SYS-PIECEWISE-BLOCK-AR-v1.0.0",
        data,
        record,
        tuple(range(50)),
    )


def true_piecewise_part(end_index: int) -> tuple[int, ...] | None:
    """Return the frozen regime partition, or None for windows crossing the change."""

    if end_index < 1024:
        return tuple(range(50))
    return (*range(25), *range(75, 100))
