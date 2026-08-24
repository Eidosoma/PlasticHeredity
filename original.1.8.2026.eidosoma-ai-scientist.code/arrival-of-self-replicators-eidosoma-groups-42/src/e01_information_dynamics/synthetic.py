"""Deterministic preregistered synthetic systems for E01 S10."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from e01_gard_reproducibility import (
    CANONICAL_STREAM_PURPOSES,
    CouplingPolicy,
    SeedRequest,
    StreamPurpose,
    derive_seed_bundle,
    isolated_stream_namespace,
)

ROOT_SEED_HEX = "10" * 32
RAW_SAMPLE_COUNT = 32_769
BURN_IN = 4_096


@dataclass(frozen=True, slots=True)
class SyntheticSeries:
    """One generated scalar pair or multivariate component matrix."""

    system_id: str
    replicate_index: int
    data: np.ndarray
    seed_payload: dict


def estimator_generator(
    *,
    system_id: str,
    trajectory_id: str,
    replicate_index: int,
) -> tuple[np.random.Generator, dict]:
    """Create an S06-domain-separated estimator stream for one fixture identity."""

    namespace = isolated_stream_namespace(
        experiment_id="E01",
        specification_id=system_id,
        trajectory_id=trajectory_id,
        replicate_index=replicate_index,
    )
    request = SeedRequest(
        experiment_id="E01",
        specification_id=system_id,
        trajectory_id=trajectory_id,
        replicate_index=replicate_index,
        engine_id="e01_information_dynamics",
        root_seed_hex=ROOT_SEED_HEX,
        coupling_policy=CouplingPolicy.TRAJECTORY_ISOLATED,
        coupling_reason=None,
        stream_namespaces={purpose: namespace for purpose in CANONICAL_STREAM_PURPOSES},
    )
    bundle = derive_seed_bundle(request)
    generator = bundle.streams[StreamPurpose.ESTIMATOR].generator()
    return generator, bundle.to_payload()


def independent_gaussian(
    replicate_index: int, *, n: int = RAW_SAMPLE_COUNT
) -> SyntheticSeries:
    system_id = "E01-S10-SYS-INDEPENDENT-GAUSSIAN-v1.0.0"
    rng, payload = estimator_generator(
        system_id=system_id,
        trajectory_id="primary",
        replicate_index=replicate_index,
    )
    data = rng.standard_normal((n, 2), dtype=np.float64)
    return SyntheticSeries(system_id, replicate_index, data, payload)


def redundant_discrete(
    replicate_index: int, *, n: int = RAW_SAMPLE_COUNT
) -> SyntheticSeries:
    system_id = "E01-S10-SYS-REDUNDANT-DISCRETE-v1.0.0"
    rng, payload = estimator_generator(
        system_id=system_id,
        trajectory_id="primary",
        replicate_index=replicate_index,
    )
    latent = np.empty(n, dtype=np.int8)
    latent[0] = int(rng.integers(0, 2))
    flips = rng.random(n - 1) < 0.1
    for index, flip in enumerate(flips, start=1):
        latent[index] = latent[index - 1] ^ int(flip)
    data = np.column_stack([latent, latent]).astype(np.float64)
    return SyntheticSeries(system_id, replicate_index, data, payload)


def redundant_gaussian(
    replicate_index: int, *, n: int = RAW_SAMPLE_COUNT
) -> SyntheticSeries:
    system_id = "E01-S10-SYS-REDUNDANT-GAUSSIAN-v1.0.0"
    rng, payload = estimator_generator(
        system_id=system_id,
        trajectory_id="primary",
        replicate_index=replicate_index,
    )
    latent = np.empty(n + BURN_IN, dtype=np.float64)
    latent[0] = rng.normal(scale=np.sqrt(1.0 / (1.0 - 0.9**2)))
    innovation = rng.standard_normal(n + BURN_IN - 1)
    for index in range(1, latent.size):
        latent[index] = 0.9 * latent[index - 1] + innovation[index - 1]
    latent = latent[BURN_IN:]
    observations = rng.standard_normal((n, 2)) * 0.35
    data = latent[:, None] + observations
    return SyntheticSeries(system_id, replicate_index, data, payload)


def xor_discrete(replicate_index: int, *, n: int = RAW_SAMPLE_COUNT) -> SyntheticSeries:
    system_id = "E01-S10-SYS-XOR-DISCRETE-v1.0.0"
    rng, payload = estimator_generator(
        system_id=system_id,
        trajectory_id="primary",
        replicate_index=replicate_index,
    )
    data = np.empty((n, 2), dtype=np.int8)
    data[0] = rng.integers(0, 2, size=2)
    future_y = rng.integers(0, 2, size=n - 1)
    for index in range(1, n):
        data[index, 0] = data[index - 1, 0] ^ data[index - 1, 1]
        data[index, 1] = future_y[index - 1]
    return SyntheticSeries(system_id, replicate_index, data.astype(np.float64), payload)


def coupled_ar(replicate_index: int, *, n: int = RAW_SAMPLE_COUNT) -> SyntheticSeries:
    system_id = "E01-S10-SYS-COUPLED-AR-v1.0.0"
    rng, payload = estimator_generator(
        system_id=system_id,
        trajectory_id="primary",
        replicate_index=replicate_index,
    )
    transition = np.asarray([[0.0, 0.0], [0.85, 0.25]], dtype=np.float64)
    innovation_sd = np.asarray([1.0, 0.5], dtype=np.float64)
    data = np.zeros((n + BURN_IN, 2), dtype=np.float64)
    innovations = rng.standard_normal(data.shape) * innovation_sd
    for index in range(1, data.shape[0]):
        data[index] = transition @ data[index - 1] + innovations[index]
    return SyntheticSeries(system_id, replicate_index, data[BURN_IN:], payload)


def block_ar4(replicate_index: int, *, n: int = RAW_SAMPLE_COUNT) -> SyntheticSeries:
    system_id = "E01-S10-SYS-BLOCK-AR4-v1.0.0"
    rng, payload = estimator_generator(
        system_id=system_id,
        trajectory_id="mib",
        replicate_index=replicate_index,
    )
    latent = np.zeros((n + BURN_IN, 2), dtype=np.float64)
    innovations = rng.standard_normal(latent.shape)
    for index in range(1, latent.shape[0]):
        latent[index] = 0.82 * latent[index - 1] + innovations[index]
    latent = latent[BURN_IN:]
    noise = rng.standard_normal((n, 4)) * 0.20
    data = (
        np.column_stack([latent[:, 0], latent[:, 0], latent[:, 1], latent[:, 1]])
        + noise
    )
    return SyntheticSeries(system_id, replicate_index, data, payload)


GENERATORS: dict[str, Callable[[int], SyntheticSeries]] = {
    "E01-S10-SYS-INDEPENDENT-GAUSSIAN-v1.0.0": independent_gaussian,
    "E01-S10-SYS-REDUNDANT-DISCRETE-v1.0.0": redundant_discrete,
    "E01-S10-SYS-REDUNDANT-GAUSSIAN-v1.0.0": redundant_gaussian,
    "E01-S10-SYS-XOR-DISCRETE-v1.0.0": xor_discrete,
    "E01-S10-SYS-COUPLED-AR-v1.0.0": coupled_ar,
    "E01-S10-SYS-BLOCK-AR4-v1.0.0": block_ar4,
}


def common_time_shuffle(series: SyntheticSeries) -> SyntheticSeries:
    """Apply one named common row permutation without changing row contents."""

    rng, payload = estimator_generator(
        system_id=series.system_id,
        trajectory_id="time-shuffle",
        replicate_index=series.replicate_index,
    )
    permutation = rng.permutation(series.data.shape[0])
    shuffled_payload = dict(payload)
    shuffled_payload["permutation"] = {
        "algorithm": "numpy.Generator.permutation",
        "length": int(permutation.size),
        "sha256": __import__("hashlib")
        .sha256(permutation.astype(">u8").tobytes())
        .hexdigest(),
    }
    return SyntheticSeries(
        series.system_id,
        series.replicate_index,
        series.data[permutation],
        shuffled_payload,
    )


def affine_transform(series: SyntheticSeries) -> SyntheticSeries:
    """Apply the frozen invertible scalar affine/relabeling control."""

    _, payload = estimator_generator(
        system_id=series.system_id,
        trajectory_id="gaussian-affine-control",
        replicate_index=series.replicate_index,
    )
    data = np.asarray(series.data, dtype=np.float64).copy()
    data[:, 0] = 17.0 * data[:, 0] + 23.0
    data[:, 1] = -0.125 * data[:, 1] + 5.0
    return SyntheticSeries(series.system_id, series.replicate_index, data, payload)


def discrete_relabel(series: SyntheticSeries) -> SyntheticSeries:
    """Apply the frozen positive/negative binary affine relabeling control."""

    _, payload = estimator_generator(
        system_id=series.system_id,
        trajectory_id="discrete-relabel-control",
        replicate_index=series.replicate_index,
    )
    data = np.asarray(series.data, dtype=np.float64).copy()
    data[:, 0] = 3.0 * data[:, 0] + 11.0
    data[:, 1] = -2.0 * data[:, 1] + 7.0
    return SyntheticSeries(series.system_id, series.replicate_index, data, payload)
