from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .rng import generator, jax_key, stable_seed
from .storage import load_npz, write_npz_atomic


@dataclass
class GNNModel:
    params: dict[str, Any]
    node_mean: np.ndarray
    node_scale: np.ndarray
    history_mean: np.ndarray
    history_scale: np.ndarray
    width: int
    layers: int
    fold: int
    best_epoch: int
    best_validation_loss: float


def _init_dense(key, inputs: int, outputs: int) -> dict[str, jax.Array]:
    limit = np.sqrt(6.0 / (inputs + outputs))
    return {
        "kernel": jax.random.uniform(key, (inputs, outputs), minval=-limit, maxval=limit),
        "bias": jnp.zeros((outputs,), dtype=jnp.float32),
    }


def init_params(key, node_features: int, history_features: int, width: int, layers: int) -> dict[str, Any]:
    keys = jax.random.split(key, layers + 4)
    message_layers = []
    current = node_features
    for layer in range(layers):
        message_layers.append(_init_dense(keys[layer], current * 3, width))
        current = width
    pooled = width * 3 + history_features
    return {
        "message": tuple(message_layers),
        "shared": _init_dense(keys[layers], pooled, width),
        "break_head": _init_dense(keys[layers + 1], width, 1),
        "recovery_head": _init_dense(keys[layers + 2], width, 1),
    }


def forward(params: dict[str, Any], nodes, weights, history):
    hidden = nodes
    positive = jnp.maximum(weights, 0.0)
    negative = jnp.maximum(-weights, 0.0)
    for layer in params["message"]:
        positive_message = jnp.einsum("bij,bjf->bif", positive, hidden)
        negative_message = jnp.einsum("bij,bjf->bif", negative, hidden)
        combined = jnp.concatenate((hidden, positive_message, negative_message), axis=-1)
        hidden = jax.nn.silu(combined @ layer["kernel"] + layer["bias"])
    pooled = jnp.concatenate(
        (jnp.mean(hidden, axis=1), jnp.std(hidden, axis=1), jnp.max(hidden, axis=1), history), axis=-1
    )
    shared = jax.nn.silu(pooled @ params["shared"]["kernel"] + params["shared"]["bias"])
    break_logit = (shared @ params["break_head"]["kernel"] + params["break_head"]["bias"])[:, 0]
    recovery_logit = (shared @ params["recovery_head"]["kernel"] + params["recovery_head"]["bias"])[:, 0]
    return break_logit, recovery_logit


def _loss(params, batch):
    break_logit, recovery_logit = forward(params, batch["nodes"], batch["W"], batch["history"])
    break_success = batch["break_count"]
    event_success = batch["event_count"]
    total = batch["total"]
    break_nll = -break_success * jax.nn.log_sigmoid(break_logit) - (total - break_success) * jax.nn.log_sigmoid(-break_logit)
    recovery_nll = -event_success * jax.nn.log_sigmoid(recovery_logit) - (break_success - event_success) * jax.nn.log_sigmoid(-recovery_logit)
    denominator = jnp.maximum(jnp.sum(total + break_success), 1.0)
    return jnp.sum(break_nll + recovery_nll) / denominator


def _standardization(flat: dict[str, np.ndarray], mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    node_values = flat["node_features"][mask].reshape(-1, flat["node_features"].shape[-1]).astype(np.float64)
    history_values = flat["history_features"][mask].astype(np.float64)
    node_mean = node_values.mean(axis=0)
    node_scale = np.maximum(node_values.std(axis=0), 1e-6)
    history_mean = history_values.mean(axis=0)
    history_scale = np.maximum(history_values.std(axis=0), 1e-6)
    return node_mean, node_scale, history_mean, history_scale


def _batch(flat: dict[str, np.ndarray], indices: np.ndarray, model: GNNModel | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> dict[str, jax.Array]:
    if isinstance(model, GNNModel):
        node_mean, node_scale = model.node_mean, model.node_scale
        history_mean, history_scale = model.history_mean, model.history_scale
    else:
        node_mean, node_scale, history_mean, history_scale = model
    nodes = (flat["node_features"][indices] - node_mean) / node_scale
    history = (flat["history_features"][indices] - history_mean) / history_scale
    return {
        "nodes": jnp.asarray(nodes, dtype=jnp.float32),
        "history": jnp.asarray(history, dtype=jnp.float32),
        "W": jnp.asarray(flat["W"][indices], dtype=jnp.float32),
        "event_count": jnp.asarray(flat["event_count"][indices], dtype=jnp.float32),
        "break_count": jnp.asarray(flat["break_count"][indices], dtype=jnp.float32),
        "total": jnp.asarray(flat["total"][indices], dtype=jnp.float32),
    }


def fit_fold(flat: dict[str, np.ndarray], protocol: dict[str, Any], tier: str, fold: int) -> GNNModel:
    predictor = protocol["predictor"]
    folds = int(predictor["folds"])
    master = str(protocol["master_seed_label"])
    if "assigned_fold" in flat:
        network_folds = np.asarray(flat["assigned_fold"], dtype=np.int32)
    else:
        network_folds = np.asarray([
            stable_seed(master, "predictor-fold", tier, int(index)) % folds
            for index in flat["network_index"]
        ])
    train_indices = np.flatnonzero(network_folds != fold)
    validation_indices = np.flatnonzero(network_folds == fold)
    if not len(train_indices) or not len(validation_indices):
        raise RuntimeError(f"empty train/validation partition for fold {fold}")
    scaling = _standardization(flat, train_indices)
    width = int(predictor["width"])
    layers = int(predictor["message_layers"])
    params = init_params(
        jax_key(master, "gnn-init", tier, fold), flat["node_features"].shape[-1],
        flat["history_features"].shape[-1], width, layers,
    )
    optimizer = optax.adamw(float(predictor["learning_rate"]), weight_decay=float(predictor["weight_decay"]))
    optimizer_state = optimizer.init(params)

    @jax.jit
    def train_step(current_params, current_state, batch):
        loss_value, gradients = jax.value_and_grad(_loss)(current_params, batch)
        updates, next_state = optimizer.update(gradients, current_state, current_params)
        return optax.apply_updates(current_params, updates), next_state, loss_value

    validation_batch = _batch(flat, validation_indices, scaling)
    validation_loss = jax.jit(_loss)
    rng = generator(master, "gnn-batches", tier, fold)
    batch_size = int(predictor["batch_states"])
    best_params = jax.tree.map(lambda value: np.asarray(value), params)
    best_loss = float("inf")
    best_epoch = -1
    stale = 0
    for epoch in range(int(predictor["max_epochs"])):
        shuffled = rng.permutation(train_indices)
        for start in range(0, len(shuffled), batch_size):
            indices = shuffled[start : start + batch_size]
            params, optimizer_state, _ = train_step(params, optimizer_state, _batch(flat, indices, scaling))
        current_loss = float(validation_loss(params, validation_batch))
        if current_loss < best_loss - 1e-6:
            best_loss = current_loss
            best_epoch = epoch
            best_params = jax.tree.map(lambda value: np.asarray(value), params)
            stale = 0
        else:
            stale += 1
        if stale >= int(predictor["patience"]):
            break
    return GNNModel(
        params=best_params,
        node_mean=scaling[0], node_scale=scaling[1], history_mean=scaling[2], history_scale=scaling[3],
        width=width, layers=layers, fold=fold, best_epoch=best_epoch, best_validation_loss=best_loss,
    )


def predict_model(model: GNNModel, flat: dict[str, np.ndarray], batch_size: int = 1024) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    outputs_break, outputs_recovery = [], []
    compiled = jax.jit(forward)
    for start in range(0, len(flat["event_count"]), batch_size):
        indices = np.arange(start, min(start + batch_size, len(flat["event_count"])))
        batch = _batch(flat, indices, model)
        break_logit, recovery_logit = compiled(model.params, batch["nodes"], batch["W"], batch["history"])
        outputs_break.append(np.asarray(jax.nn.sigmoid(break_logit)))
        outputs_recovery.append(np.asarray(jax.nn.sigmoid(recovery_logit)))
    break_probability = np.concatenate(outputs_break)
    recovery_probability = np.concatenate(outputs_recovery)
    return break_probability, recovery_probability, break_probability * recovery_probability


def predict_ensemble(models: list[GNNModel], flat: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predictions = [predict_model(model, flat) for model in models]
    return tuple(np.mean(np.stack([value[index] for value in predictions]), axis=0) for index in range(3))  # type: ignore[return-value]


def save_model(model: GNNModel, path: str | Path) -> None:
    arrays: dict[str, np.ndarray] = {
        "node_mean": model.node_mean, "node_scale": model.node_scale,
        "history_mean": model.history_mean, "history_scale": model.history_scale,
        "metadata": np.asarray(json.dumps({
            "width": model.width, "layers": model.layers, "fold": model.fold,
            "best_epoch": model.best_epoch, "best_validation_loss": model.best_validation_loss,
        }, sort_keys=True)),
    }
    for index, layer in enumerate(model.params["message"]):
        arrays[f"message_{index}_kernel"] = np.asarray(layer["kernel"])
        arrays[f"message_{index}_bias"] = np.asarray(layer["bias"])
    for name in ("shared", "break_head", "recovery_head"):
        arrays[f"{name}_kernel"] = np.asarray(model.params[name]["kernel"])
        arrays[f"{name}_bias"] = np.asarray(model.params[name]["bias"])
    write_npz_atomic(path, **arrays)


def load_model(path: str | Path) -> GNNModel:
    arrays = load_npz(path)
    metadata = json.loads(str(arrays["metadata"]))
    params: dict[str, Any] = {"message": tuple(
        {"kernel": arrays[f"message_{index}_kernel"], "bias": arrays[f"message_{index}_bias"]}
        for index in range(int(metadata["layers"]))
    )}
    for name in ("shared", "break_head", "recovery_head"):
        params[name] = {"kernel": arrays[f"{name}_kernel"], "bias": arrays[f"{name}_bias"]}
    return GNNModel(
        params=params, node_mean=arrays["node_mean"], node_scale=arrays["node_scale"],
        history_mean=arrays["history_mean"], history_scale=arrays["history_scale"],
        width=int(metadata["width"]), layers=int(metadata["layers"]), fold=int(metadata["fold"]),
        best_epoch=int(metadata["best_epoch"]), best_validation_loss=float(metadata["best_validation_loss"]),
    )
