from __future__ import annotations

from typing import Any

import numpy as np

from .network import Network


HISTORY_FEATURES = (
    "cue_A", "cue_B", "age_0", "age_2", "age_4", "age_8", "age_12",
    "age_scaled", "cue_age", "intercept_context",
)
NODE_FEATURES = (
    "expression", "velocity", "field", "activation_derivative",
    "signed_in_strength", "signed_out_strength", "cue_sign", "partition_cv",
)
STRUCTURAL_FEATURES = (
    "expression_mean", "expression_std", "velocity_l2", "field_mean", "field_std",
    "derivative_mean", "derivative_std", "weight_spectral_radius", "jacobian_max_real",
    "weight_frobenius", "positive_edge_fraction", "negative_edge_fraction",
    "mean_abs_in_strength", "std_abs_in_strength", "linear_sensitivity", "jacobian_condition",
)


def history_panel(cue_index: int, age: int) -> np.ndarray:
    landmarks = (0, 2, 4, 8, 12)
    values = [float(cue_index == 0), float(cue_index == 1)]
    values.extend(float(age == value) for value in landmarks)
    scaled = age / 12.0
    values.extend((scaled, (1.0 if cue_index == 0 else -1.0) * scaled, 1.0))
    return np.asarray(values, dtype=np.float32)


def _network_summaries(expression: np.ndarray, velocity: np.ndarray, field: np.ndarray, derivative: np.ndarray, W: np.ndarray, jacobian: np.ndarray) -> np.ndarray:
    eigen_w = np.linalg.eigvals(W.astype(np.float64))
    eigen_j = np.linalg.eigvals(jacobian.astype(np.float64))
    nonzero = W != 0
    edge_count = max(int(nonzero.sum()), 1)
    identity = np.eye(W.shape[0], dtype=np.float64)
    try:
        sensitivity = float(np.linalg.norm(np.linalg.pinv(identity - jacobian), ord=2))
    except np.linalg.LinAlgError:
        sensitivity = 1e6
    try:
        condition = float(np.linalg.cond(identity - jacobian))
    except np.linalg.LinAlgError:
        condition = 1e12
    values = np.asarray([
        np.mean(expression), np.std(expression), np.linalg.norm(velocity) / np.sqrt(len(velocity)),
        np.mean(field), np.std(field), np.mean(derivative), np.std(derivative),
        np.max(np.abs(eigen_w)), np.max(np.real(eigen_j)), np.linalg.norm(W),
        np.sum(W > 0) / edge_count, np.sum(W < 0) / edge_count,
        np.mean(np.sum(np.abs(W), axis=1)), np.std(np.sum(np.abs(W), axis=1)),
        np.clip(sensitivity, 0.0, 1e6), np.clip(condition, 0.0, 1e12),
    ], dtype=np.float64)
    return values.astype(np.float32)


def continuous_features(network: Network, state: np.ndarray, cue_index: int, age: int, protocol: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cfg = protocol["tiers"]["continuous"]
    expression = np.asarray(state, dtype=np.float64)
    field = float(cfg["gain"]) * (network.W @ (2.0 * expression - 1.0)) + network.bias
    activation = 1.0 / (1.0 + np.exp(-field))
    velocity = activation - expression
    derivative = activation * (1.0 - activation)
    signed_in = network.W.sum(axis=1)
    signed_out = network.W.sum(axis=0)
    cue = network.cue_a if cue_index == 0 else network.cue_b
    partition_cv = 1.0 / np.sqrt(np.maximum(2.0 * float(cfg["partition_scale"]) * expression, 1.0))
    nodes = np.stack((expression, velocity, field, derivative, signed_in, signed_out, cue, partition_cv), axis=1)
    jacobian = -np.eye(len(expression)) + (2.0 * float(cfg["gain"]) * derivative)[:, None] * network.W
    return history_panel(cue_index, age), _network_summaries(expression, velocity, field, derivative, network.W, jacobian), nodes.astype(np.float32)


def molecular_features(network: Network, mrna: np.ndarray, protein: np.ndarray, cue_index: int, age: int, protocol: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cfg = protocol["tiers"]["molecular"]
    expression = np.asarray(protein, dtype=np.float64) / float(cfg["protein_scale"])
    field = float(cfg["gain"]) * (network.W @ (expression - 0.5)) + network.bias
    activation = 1.0 / (1.0 + np.exp(-field))
    velocity = float(cfg["translation"]) * np.asarray(mrna) / float(cfg["protein_scale"]) - float(cfg["protein_decay"]) * expression
    derivative = activation * (1.0 - activation)
    signed_in = network.W.sum(axis=1)
    signed_out = network.W.sum(axis=0)
    cue = network.cue_a if cue_index == 0 else network.cue_b
    partition_cv = 1.0 / np.sqrt(np.maximum(np.asarray(protein, dtype=np.float64), 1.0))
    nodes = np.stack((expression, velocity, field, derivative, signed_in, signed_out, cue, partition_cv), axis=1)
    regulatory_scale = float(cfg["gain"]) * derivative / float(cfg["protein_scale"])
    jacobian = -float(cfg["protein_decay"]) * np.eye(len(expression)) + regulatory_scale[:, None] * network.W
    return history_panel(cue_index, age), _network_summaries(expression, velocity, field, derivative, network.W, jacobian), nodes.astype(np.float32)

