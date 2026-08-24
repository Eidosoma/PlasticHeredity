from __future__ import annotations

import numpy as np

from e01_onset_discovery.outcome_blind_representation import (
    CHANNEL_NAMES,
    KERNEL_BANK,
    KERNEL_COUNT,
    RANDOM_CONV_FEATURES,
    extract_outcome_blind_representation,
    kernel_bank_fingerprint,
    organization_channel_sequence,
)


def fixture() -> np.ndarray:
    rng = np.random.default_rng(2201)
    values = rng.poisson(2.0, size=(64, 100)).astype(np.int64)
    values[:, 0] += 1
    return values


def test_schema_and_replay() -> None:
    values = fixture()
    first = extract_outcome_blind_representation(values)
    second = extract_outcome_blind_representation(values.copy())
    assert tuple(first) == RANDOM_CONV_FEATURES
    assert len(first) == 2 * KERNEL_COUNT == 128
    assert first == second
    assert len(kernel_bank_fingerprint()) == 64


def test_molecule_permutation_invariance() -> None:
    values = fixture()
    permutation = np.random.default_rng(2202).permutation(100)
    first = extract_outcome_blind_representation(values)
    second = extract_outcome_blind_representation(values[:, permutation])
    assert all(
        np.isclose(first[name], second[name], atol=1e-12, rtol=1e-12)
        for name in first
    )


def test_temporal_sensitivity_and_channels() -> None:
    values = fixture()
    channels = organization_channel_sequence(values)
    assert channels.shape == (64, len(CHANNEL_NAMES))
    reversed_values = values[::-1].copy()
    first = extract_outcome_blind_representation(values)
    second = extract_outcome_blind_representation(reversed_values)
    assert any(first[name] != second[name] for name in RANDOM_CONV_FEATURES)


def test_positive_scaling_invariance_after_standardization() -> None:
    values = fixture()
    first = extract_outcome_blind_representation(values)
    second = extract_outcome_blind_representation(values * 3)
    assert all(np.isclose(first[name], second[name], atol=1e-12, rtol=1e-12) for name in first)


def test_kernel_contract() -> None:
    assert len(KERNEL_BANK) == KERNEL_COUNT
    assert all(kernel.length in {7, 9, 11} for kernel in KERNEL_BANK)
    assert all(kernel.dilation in {1, 2, 4, 8} for kernel in KERNEL_BANK)
    assert all(np.isclose(np.linalg.norm(kernel.weights), 1.0) for kernel in KERNEL_BANK)
    assert all(np.isclose(np.mean(kernel.weights), 0.0, atol=1e-16) for kernel in KERNEL_BANK)
