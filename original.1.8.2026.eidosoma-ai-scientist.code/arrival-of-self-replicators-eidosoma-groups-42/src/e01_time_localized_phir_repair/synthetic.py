"""Phase-isolated validation fixtures for the bounded E01 S11R step."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy.linalg import solve_discrete_lyapunov

from e01_gard_reproducibility import (
    CANONICAL_STREAM_PURPOSES,
    CouplingPolicy,
    SeedRequest,
    StreamPurpose,
    derive_seed_bundle,
    isolated_stream_namespace,
)
from e01_information_dynamics.validation import gaussian_mmi_oracle
from e01_time_localized_phir.synthetic import ccs_population_oracle as _s11_ccs_oracle

DEVELOPMENT_ROOT_SEED_HEX = "d1" * 32
CONFIRMATION_ROOT_SEED_HEX = "c1" * 32
BURN_IN = 512

INDEPENDENT_ID = "E01-S11-SYS-INDEPENDENT-WHITE-GAUSSIAN-v1.0.0"
REDUNDANT_ID = "E01-S11-SYS-NOISY-REDUNDANT-AR-v1.0.0"
DIRECTIONAL_ID = "E01-S11-SYS-DIRECTIONAL-VAR-v1.0.0"
PLANTED_ID = "E01-S11-SYS-PLANTED-TWO-BLOCK-AR-v1.0.0"
HIGHDIM_NULL_ID = "E01-S11-SYS-HIGHDIM-INDEPENDENT-NULL-v1.0.0"

_DIRECTIONAL_TRANSITION = np.asarray([[0.10, 0.00], [0.75, 0.20]], dtype=np.float64)
_DIRECTIONAL_INNOVATION_SD = np.asarray([1.0, 0.5], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class Fixture:
    system_id: str
    data: np.ndarray
    seed_record: dict[str, Any]
    planted_part_a: tuple[int, ...] | None = None


def repair_rng(
    *,
    phase: Literal["development", "confirmation"],
    domain: str,
    pair_id: str,
    replicate_index: int,
    dimension: int | None = None,
) -> tuple[np.random.Generator, dict[str, Any]]:
    """Derive one S06 PCG64DXSM stream with phase in its immutable identity."""

    root = (
        DEVELOPMENT_ROOT_SEED_HEX
        if phase == "development"
        else CONFIRMATION_ROOT_SEED_HEX
    )
    specification_id = f"E01-S11R-{phase}-{domain}"
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
        engine_id="e01_time_localized_phir_repair",
        root_seed_hex=root,
        coupling_policy=CouplingPolicy.TRAJECTORY_ISOLATED,
        coupling_reason=None,
        stream_namespaces={purpose: namespace for purpose in CANONICAL_STREAM_PURPOSES},
    )
    stream = derive_seed_bundle(request).streams[StreamPurpose.ESTIMATOR]
    record = {
        "phase": phase,
        "domain": domain,
        "pairId": pair_id,
        "dimension": dimension,
        "replicateIndex": replicate_index,
        **stream.to_payload(),
    }
    return stream.generator(), record


def independent_white(
    *, phase: str, pair_id: str, replicate_index: int, length: int, domain: str
) -> Fixture:
    rng, record = repair_rng(
        phase=phase, domain=domain, pair_id=pair_id, replicate_index=replicate_index
    )
    return Fixture(INDEPENDENT_ID, rng.standard_normal((length, 2)), record)


def noisy_redundant_ar(
    *, phase: str, pair_id: str, replicate_index: int, length: int, domain: str
) -> Fixture:
    rng, record = repair_rng(
        phase=phase, domain=domain, pair_id=pair_id, replicate_index=replicate_index
    )
    latent = np.zeros(length + BURN_IN, dtype=np.float64)
    innovations = rng.standard_normal(latent.size)
    for index in range(1, latent.size):
        latent[index] = 0.80 * latent[index - 1] + innovations[index]
    data = latent[BURN_IN:, None] + 0.35 * rng.standard_normal((length, 2))
    return Fixture(REDUNDANT_ID, data, record)


def directional_var(
    *, phase: str, pair_id: str, replicate_index: int, length: int, domain: str
) -> Fixture:
    rng, record = repair_rng(
        phase=phase, domain=domain, pair_id=pair_id, replicate_index=replicate_index
    )
    data = np.zeros((length + BURN_IN, 2), dtype=np.float64)
    innovations = rng.standard_normal(data.shape) * _DIRECTIONAL_INNOVATION_SD
    for index in range(1, data.shape[0]):
        data[index] = _DIRECTIONAL_TRANSITION @ data[index - 1] + innovations[index]
    return Fixture(DIRECTIONAL_ID, data[BURN_IN:], record)


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
    covariance = {
        REDUNDANT_ID: redundant_covariance,
        DIRECTIONAL_ID: directional_covariance,
    }.get(system_id)
    if covariance is None:
        if system_id == INDEPENDENT_ID:
            return gaussian_mmi_oracle(np.eye(4))
        raise ValueError(f"Unknown scalar system {system_id!r}.")
    return gaussian_mmi_oracle(covariance(tau))


def ccs_population_oracle(
    covariance: np.ndarray, *, scramble_seed: int
) -> dict[str, Any]:
    """Reuse the frozen S11 population oracle, never its failed finite-sample estimator."""

    return _s11_ccs_oracle(covariance, scramble_seed=scramble_seed, power=18)


def planted_two_block_ar(
    *,
    phase: str,
    pair_id: str,
    replicate_index: int,
    length: int,
    dimension: int,
    domain: str,
) -> Fixture:
    if dimension not in (8, 99, 100):
        raise ValueError("S11R planted dimensions are 8, 99, and 100.")
    rng, record = repair_rng(
        phase=phase,
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
    return Fixture(PLANTED_ID, data, record, tuple(range(first_size)))


def highdim_independent_null(
    *,
    phase: str,
    pair_id: str,
    replicate_index: int,
    length: int,
    dimension: int,
    domain: str,
) -> Fixture:
    rng, record = repair_rng(
        phase=phase,
        domain=domain,
        pair_id=pair_id,
        replicate_index=replicate_index,
        dimension=dimension,
    )
    data = np.zeros((length + BURN_IN, dimension), dtype=np.float64)
    innovations = rng.standard_normal(data.shape)
    for index in range(1, data.shape[0]):
        data[index] = 0.20 * data[index - 1] + innovations[index]
    return Fixture(HIGHDIM_NULL_ID, data[BURN_IN:], record)
