from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .baselines import HurdleRidge, fit_hurdle_ridge
from .dataset import flatten_states, load_dataset, network_paths
from .gnn import GNNModel, fit_fold, load_model, predict_ensemble, predict_model, save_model
from .rng import generator
from .storage import sha256_file, write_json_atomic, write_npz_atomic


def balanced_network_folds(master: str, tier: str, network_indices: np.ndarray, folds: int) -> dict[int, int]:
    unique = np.unique(network_indices).astype(int)
    order = generator(master, "predictor-fold-order", tier).permutation(unique)
    return {int(network): int(position % folds) for position, network in enumerate(order)}


def _model_paths(run_dir: Path, tier: str) -> tuple[Path, Path, list[Path]]:
    root = run_dir / "models" / tier
    return root / "history_ridge.npz", root / "structural_ridge.npz", sorted(root.glob("gnn_fold_*.npz"))


def train_models(run_dir: str | Path, tier: str, protocol: dict[str, Any]) -> dict[str, Any]:
    root = Path(run_dir)
    paths = network_paths(root, "development", tier)
    expected = int(protocol["tiers"][tier]["development_networks"])
    if len(paths) != expected:
        raise RuntimeError(f"development data incomplete for {tier}: {len(paths)}/{expected}")
    dataset = load_dataset(paths)
    flat = flatten_states(dataset)
    history_x = flat["history_features"]
    structural_x = np.concatenate((history_x, flat["structural_features"]), axis=1)
    c_value = float(protocol["predictor"]["ridge_c"])
    history_model = fit_hurdle_ridge(
        history_x, flat["event_count"], flat["break_count"], flat["total"], c_value
    )
    structural_model = fit_hurdle_ridge(
        structural_x, flat["event_count"], flat["break_count"], flat["total"], c_value
    )
    model_dir = root / "models" / tier
    model_dir.mkdir(parents=True, exist_ok=True)
    history_path = model_dir / "history_ridge.npz"
    structural_path = model_dir / "structural_ridge.npz"
    history_model.save(history_path)
    structural_model.save(structural_path)

    fold_count = int(protocol["predictor"]["folds"])
    folds = balanced_network_folds(
        str(protocol["master_seed_label"]), tier, dataset["network_index"], fold_count
    )
    flat["assigned_fold"] = np.asarray([folds[int(value)] for value in flat["network_index"]])
    models: list[GNNModel] = []
    oof_break = np.full(len(flat["event_count"]), np.nan, dtype=np.float64)
    oof_recovery = np.full_like(oof_break, np.nan)
    oof_event = np.full_like(oof_break, np.nan)
    # fit_fold consumes this explicit balanced assignment when present.
    for fold in range(fold_count):
        model = fit_fold(flat, protocol, tier, fold)
        models.append(model)
        path = model_dir / f"gnn_fold_{fold}.npz"
        save_model(model, path)
        held_out = np.flatnonzero(flat["assigned_fold"] == fold)
        pb, pr, pe = predict_model(model, {name: value[held_out] if len(value) == len(oof_event) else value for name, value in flat.items() if name != "assigned_fold"})
        oof_break[held_out], oof_recovery[held_out], oof_event[held_out] = pb, pr, pe
    if np.any(~np.isfinite(oof_event)):
        raise RuntimeError("development out-of-fold predictions are incomplete")
    history_prediction = history_model.predict(history_x)
    structural_prediction = structural_model.predict(structural_x)
    states = dataset["event_count"].shape[1]
    write_npz_atomic(
        model_dir / "development_oof_predictions.npz",
        network_index=dataset["network_index"],
        history_event=history_prediction[2].reshape(expected, states),
        structural_event=structural_prediction[2].reshape(expected, states),
        full_break=oof_break.reshape(expected, states),
        full_recovery=oof_recovery.reshape(expected, states),
        full_event=oof_event.reshape(expected, states),
    )
    manifest = {
        "format": "grn-f12-model-manifest-v1",
        "tier": tier,
        "development_networks": expected,
        "development_shards": {path.name: sha256_file(path) for path in paths},
        "confirmation_data_accessed": False,
        "fold_assignment": {str(key): value for key, value in sorted(folds.items())},
        "gnn": [
            {"fold": model.fold, "best_epoch": model.best_epoch, "best_validation_loss": model.best_validation_loss}
            for model in models
        ],
    }
    write_json_atomic(model_dir / "training_manifest.json", manifest)
    return manifest


def load_models(run_dir: str | Path, tier: str) -> tuple[HurdleRidge, HurdleRidge, list[GNNModel]]:
    history_path, structural_path, gnn_paths = _model_paths(Path(run_dir), tier)
    if not history_path.exists() or not structural_path.exists() or len(gnn_paths) != 5:
        raise RuntimeError(f"frozen {tier} models are incomplete")
    return HurdleRidge.load(history_path), HurdleRidge.load(structural_path), [load_model(path) for path in gnn_paths]


def predict_confirmation(run_dir: str | Path, tier: str, protocol: dict[str, Any]) -> dict[str, Any]:
    root = Path(run_dir)
    paths = network_paths(root, "confirmation", tier)
    expected = int(protocol["tiers"][tier]["confirmation_networks"])
    if len(paths) != expected:
        raise RuntimeError(f"confirmation data incomplete for {tier}: {len(paths)}/{expected}")
    dataset = load_dataset(paths)
    flat = flatten_states(dataset)
    history_model, structural_model, models = load_models(root, tier)
    history = history_model.predict(flat["history_features"])
    structural_x = np.concatenate((flat["history_features"], flat["structural_features"]), axis=1)
    structural = structural_model.predict(structural_x)
    full = predict_ensemble(models, flat)
    networks, states = dataset["event_count"].shape
    output = root / "predictions" / f"{tier}.npz"
    arrays = {
        "network_index": dataset["network_index"],
        "cue_index": dataset["cue_index"], "age": dataset["age"],
        "event_count": dataset["event_count"], "break_count": dataset["break_count"],
        "event_half0": dataset["event_half0"], "event_half1": dataset["event_half1"],
        "break_half0": dataset["break_half0"], "break_half1": dataset["break_half1"],
        "run5_count": dataset["run5_count"], "f24_count": dataset["f24_count"],
        "event_count_q025": dataset["event_count_q025"], "event_count_q10": dataset["event_count_q10"],
        "coherence_mean": dataset["coherence_mean"],
        "old_anchor_separation": dataset["old_anchor_separation"],
        "futures": dataset["futures"],
    }
    for label, values in (("history", history), ("structural", structural), ("full", full)):
        arrays[f"{label}_break"] = values[0].reshape(networks, states)
        arrays[f"{label}_recovery"] = values[1].reshape(networks, states)
        arrays[f"{label}_event"] = values[2].reshape(networks, states)
    if tier == "molecular":
        _, _, continuous_models = load_models(root, "continuous")
        zero_shot = predict_ensemble(continuous_models, flat)
        arrays["continuous_zero_shot_break"] = zero_shot[0].reshape(networks, states)
        arrays["continuous_zero_shot_recovery"] = zero_shot[1].reshape(networks, states)
        arrays["continuous_zero_shot_event"] = zero_shot[2].reshape(networks, states)
    write_npz_atomic(output, **arrays)
    report = {
        "format": "grn-f12-confirmation-predictions-v1",
        "tier": tier, "networks": networks, "states": networks * states,
        "output": str(output.relative_to(root)), "sha256": sha256_file(output),
    }
    write_json_atomic(root / "predictions" / f"{tier}.json", report)
    return report
