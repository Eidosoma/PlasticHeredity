"""Fixed outcome-blind representation for S19-L22.

The representation follows the random-convolution principle of ROCKET, but
operates on a prospectively frozen sequence of molecule-label-permutation-
invariant organization channels.  The kernel bank is generated solely from a
domain-separated seed and never sees outcomes or cohort values.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

LANDMARK_COUNT = 64
MOLECULE_TYPES = 100
CHANNEL_NAMES = (
    "log_mass",
    "shannon_diversity",
    "simpson_concentration",
    "maximum_fraction",
    "occupied_fraction",
    "effective_diversity_fraction",
    "adjacent_h",
    "chord_step",
    "maximum_prior_nonadjacent_h",
    "prior_recurrence_fraction_h090",
    "running_centroid_h",
)
KERNEL_COUNT = 64
KERNEL_LENGTHS = (7, 9, 11)
KERNEL_DILATIONS = (1, 2, 4, 8)
KERNEL_SEED = int.from_bytes(
    hashlib.sha256(b"E01-S19-L22-FIXED-RANDOM-CONVOLUTION-v1").digest()[:8],
    "big",
)


@dataclass(frozen=True)
class RandomKernel:
    length: int
    dilation: int
    padding: bool
    bias: float
    weights: NDArray[np.float64]


def _cosine_rows(left: NDArray[np.float64], right: NDArray[np.float64]) -> NDArray[np.float64]:
    numerator = left @ right.T
    denominator = np.linalg.norm(left, axis=1)[:, None] * np.linalg.norm(
        right, axis=1
    )[None, :]
    if np.any(denominator <= 0.0):
        raise ValueError("composition has nonpositive cosine norm")
    return np.clip(numerator / denominator, -1.0, 1.0)


def organization_channel_sequence(
    states: NDArray[np.integer],
) -> NDArray[np.float64]:
    """Return the fixed 64-by-11 past-only organization-channel sequence."""

    counts = np.asarray(states, dtype=np.float64)
    if counts.shape != (LANDMARK_COUNT, MOLECULE_TYPES):
        raise ValueError("representation input must be 64-by-100")
    if np.any(counts < 0.0):
        raise ValueError("representation input contains negative counts")
    mass = np.sum(counts, axis=1)
    if np.any(mass <= 0.0):
        raise ValueError("representation input contains an empty composition")
    composition = counts / mass[:, None]
    positive = composition > 0.0
    entropy = -np.sum(
        np.where(positive, composition * np.log(np.where(positive, composition, 1.0)), 0.0),
        axis=1,
    )
    entropy /= np.log(MOLECULE_TYPES)
    simpson = np.sum(composition * composition, axis=1)
    maximum = np.max(composition, axis=1)
    occupied = np.count_nonzero(positive, axis=1) / MOLECULE_TYPES
    effective = np.exp(entropy * np.log(MOLECULE_TYPES)) / MOLECULE_TYPES
    similarity = _cosine_rows(composition, composition)
    adjacent = np.ones(LANDMARK_COUNT, dtype=np.float64)
    adjacent[1:] = similarity[np.arange(1, LANDMARK_COUNT), np.arange(LANDMARK_COUNT - 1)]
    chord = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * adjacent))
    max_prior = np.zeros(LANDMARK_COUNT, dtype=np.float64)
    recurrence = np.zeros(LANDMARK_COUNT, dtype=np.float64)
    running_centroid = np.ones(LANDMARK_COUNT, dtype=np.float64)
    for time in range(1, LANDMARK_COUNT):
        prior_mean = np.mean(composition[:time], axis=0, keepdims=True)
        running_centroid[time] = _cosine_rows(
            composition[time : time + 1], prior_mean
        )[0, 0]
        if time >= 2:
            eligible = similarity[time, : time - 1]
            max_prior[time] = float(np.max(eligible))
            recurrence[time] = float(np.mean(eligible > 0.9))
    channels = np.column_stack(
        [
            np.log(mass),
            entropy,
            simpson,
            maximum,
            occupied,
            effective,
            adjacent,
            chord,
            max_prior,
            recurrence,
            running_centroid,
        ]
    ).astype(np.float64, copy=False)
    if not np.isfinite(channels).all():
        raise ValueError("organization channels contain nonfinite values")
    return channels


def standardized_channels(states: NDArray[np.integer]) -> NDArray[np.float64]:
    channels = organization_channel_sequence(states)
    mean = np.mean(channels, axis=0, keepdims=True)
    scale = np.std(channels, axis=0, ddof=0, keepdims=True)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return (channels - mean) / scale


def build_kernel_bank() -> tuple[RandomKernel, ...]:
    """Build the one frozen bank without inspecting data or outcomes."""

    rng = np.random.default_rng(KERNEL_SEED)
    kernels: list[RandomKernel] = []
    for _ in range(KERNEL_COUNT):
        length = int(rng.choice(KERNEL_LENGTHS))
        eligible = [
            dilation
            for dilation in KERNEL_DILATIONS
            if (length - 1) * dilation < LANDMARK_COUNT
        ]
        dilation = int(rng.choice(eligible))
        padding = bool(rng.integers(0, 2))
        weights = rng.normal(size=(len(CHANNEL_NAMES), length)).astype(np.float64)
        weights -= np.mean(weights)
        norm = float(np.linalg.norm(weights))
        if norm <= 0.0:
            raise RuntimeError("degenerate random kernel")
        weights /= norm
        kernels.append(
            RandomKernel(
                length=length,
                dilation=dilation,
                padding=padding,
                bias=float(rng.uniform(-1.0, 1.0)),
                weights=weights,
            )
        )
    return tuple(kernels)


KERNEL_BANK = build_kernel_bank()
RANDOM_CONV_FEATURES = tuple(
    name
    for index in range(KERNEL_COUNT)
    for name in (f"rocket_ppv_{index:03d}", f"rocket_max_{index:03d}")
)


def _kernel_response(
    channels: NDArray[np.float64], kernel: RandomKernel
) -> NDArray[np.float64]:
    receptive = (kernel.length - 1) * kernel.dilation + 1
    if kernel.padding:
        pad = (receptive - 1) // 2
        values = np.pad(channels, ((pad, pad), (0, 0)), mode="constant")
    else:
        values = channels
    output_length = values.shape[0] - receptive + 1
    if output_length <= 0:
        raise RuntimeError("kernel receptive field exceeds prefix")
    response = np.empty(output_length, dtype=np.float64)
    offsets = np.arange(kernel.length) * kernel.dilation
    for position in range(output_length):
        window = values[position + offsets].T
        response[position] = float(np.sum(window * kernel.weights) + kernel.bias)
    return response


def extract_outcome_blind_representation(
    states: NDArray[np.integer],
) -> dict[str, float]:
    channels = standardized_channels(states)
    result: dict[str, float] = {}
    for index, kernel in enumerate(KERNEL_BANK):
        response = _kernel_response(channels, kernel)
        result[f"rocket_ppv_{index:03d}"] = float(np.mean(response > 0.0))
        result[f"rocket_max_{index:03d}"] = float(np.max(response))
    if tuple(result) != RANDOM_CONV_FEATURES:
        raise RuntimeError("random-convolution feature order changed")
    if not np.isfinite(list(result.values())).all():
        raise RuntimeError("random-convolution representation is nonfinite")
    return result


def kernel_bank_fingerprint() -> str:
    digest = hashlib.sha256()
    for kernel in KERNEL_BANK:
        digest.update(np.asarray([kernel.length, kernel.dilation], dtype=np.int64).tobytes())
        digest.update(np.asarray([kernel.padding], dtype=np.bool_).tobytes())
        digest.update(np.asarray([kernel.bias], dtype=np.float64).tobytes())
        digest.update(kernel.weights.tobytes(order="C"))
    return digest.hexdigest()
