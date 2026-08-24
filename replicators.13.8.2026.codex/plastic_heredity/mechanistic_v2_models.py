"""No-PCA nested offset-ridge models for the beta-completeness correction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import expit

from .mechanistic_v2_features import FEATURE_NAMES, MechanisticV2RawFeatures

FloatArray = NDArray[np.float64]
RIDGE_LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
CV_FOLDS = 5


def matrix_cv_fold(matrix_ids: NDArray[np.int64]) -> NDArray[np.int64]:
    """Return the frozen whole-matrix fold assignment."""

    values = np.asarray(matrix_ids, dtype=np.int64)
    if np.any(values < 0):
        raise ValueError("matrix identifiers must be nonnegative")
    return values % CV_FOLDS


@dataclass(frozen=True)
class BlockTransform:
    name: str
    source_names: tuple[str, ...]
    raw_kept_indices: NDArray[np.int64]
    raw_kept_names: tuple[str, ...]
    dropped: dict[str, str]
    raw_mean: FloatArray
    raw_scale: FloatArray
    output_indices: NDArray[np.int64]
    output_names: tuple[str, ...]
    output_mean: FloatArray
    output_scale: FloatArray
    residual_coefficient: FloatArray | None = None

    @property
    def output_features(self) -> int:
        return int(self.output_indices.size)

    def transform(
        self, values: FloatArray, residual_base: FloatArray | None = None
    ) -> FloatArray:
        selected = np.asarray(values, dtype=np.float64)[:, self.raw_kept_indices]
        transformed = (selected - self.raw_mean) / self.raw_scale
        if self.residual_coefficient is not None:
            if residual_base is None:
                raise ValueError(f"{self.name} requires its registered residual base")
            augmented = np.column_stack(
                (np.ones(residual_base.shape[0], dtype=np.float64), residual_base)
            )
            transformed = transformed - augmented @ self.residual_coefficient
        transformed = transformed[:, self.output_indices]
        return (transformed - self.output_mean) / self.output_scale


@dataclass(frozen=True)
class LinearFit:
    name: str
    block: str
    coefficient: FloatArray
    intercept: float
    ridge_lambda: float
    objective: float
    gradient_max_abs: float
    iterations: int

    def correction(self, block: FloatArray) -> FloatArray:
        return self.intercept + np.asarray(block, dtype=np.float64) @ self.coefficient


@dataclass(frozen=True)
class CandidateRegistryV2:
    candidate: str
    transforms: dict[str, BlockTransform]
    fits: dict[str, LinearFit]
    selected_lambdas: dict[str, float]
    cv_scores: dict[str, dict[str, float]]


def _unique_columns(
    values: FloatArray,
    names: tuple[str, ...],
    tolerance: float = 1e-12,
) -> tuple[NDArray[np.int64], dict[str, str]]:
    values = np.asarray(values, dtype=np.float64)
    standard_deviation = values.std(axis=0)
    dropped = {
        names[index]: "zero_variance"
        for index, value in enumerate(standard_deviation)
        if value <= tolerance
    }
    kept: list[int] = []
    signatures: dict[bytes, list[tuple[int, FloatArray]]] = {}
    for index, deviation in enumerate(standard_deviation):
        if deviation <= tolerance:
            continue
        column = (values[:, index] - values[:, index].mean()) / deviation
        rounded = np.round(column, decimals=10)
        rounded[rounded == 0.0] = 0.0
        signature = rounded.tobytes()
        negative_rounded = np.round(-column, decimals=10)
        negative_rounded[negative_rounded == 0.0] = 0.0
        negative_signature = negative_rounded.tobytes()
        duplicate_of: str | None = None
        for previous_index, previous in signatures.get(signature, []):
            if np.allclose(column, previous, rtol=0.0, atol=tolerance):
                duplicate_of = names[previous_index]
                break
        if duplicate_of is None:
            for previous_index, previous in signatures.get(negative_signature, []):
                if np.allclose(column, -previous, rtol=0.0, atol=tolerance):
                    duplicate_of = f"negative_of:{names[previous_index]}"
                    break
        if duplicate_of is not None:
            dropped[names[index]] = f"affine_duplicate_of:{duplicate_of}"
            continue
        kept.append(index)
        signatures.setdefault(signature, []).append((index, column))
    if not kept:
        raise ValueError("feature block has no nonconstant unique columns")
    return np.asarray(kept, dtype=np.int64), dropped


def fit_block_transform(
    name: str, values: FloatArray, names: tuple[str, ...]
) -> BlockTransform:
    kept, dropped = _unique_columns(values, names)
    selected = np.asarray(values, dtype=np.float64)[:, kept]
    mean = selected.mean(axis=0)
    scale = selected.std(axis=0)
    output = np.arange(kept.size, dtype=np.int64)
    return BlockTransform(
        name=name,
        source_names=names,
        raw_kept_indices=kept,
        raw_kept_names=tuple(names[index] for index in kept),
        dropped=dropped,
        raw_mean=mean,
        raw_scale=scale,
        output_indices=output,
        output_names=tuple(names[index] for index in kept),
        output_mean=np.zeros(kept.size, dtype=np.float64),
        output_scale=np.ones(kept.size, dtype=np.float64),
    )


def fit_interaction_transform(
    values: FloatArray, residual_base: FloatArray
) -> BlockTransform:
    initial = fit_block_transform("interaction", values, FEATURE_NAMES["interaction"])
    standardized = initial.transform(values)
    augmented = np.column_stack(
        (np.ones(residual_base.shape[0], dtype=np.float64), residual_base)
    )
    coefficient = np.linalg.lstsq(augmented, standardized, rcond=None)[0]
    residual = standardized - augmented @ coefficient
    output_indices, residual_dropped = _unique_columns(
        residual, initial.raw_kept_names
    )
    dropped = dict(initial.dropped)
    dropped.update(
        {f"residual:{name}": reason for name, reason in residual_dropped.items()}
    )
    selected = residual[:, output_indices]
    return BlockTransform(
        name="interaction",
        source_names=initial.source_names,
        raw_kept_indices=initial.raw_kept_indices,
        raw_kept_names=initial.raw_kept_names,
        dropped=dropped,
        raw_mean=initial.raw_mean,
        raw_scale=initial.raw_scale,
        output_indices=output_indices,
        output_names=tuple(initial.raw_kept_names[index] for index in output_indices),
        output_mean=selected.mean(axis=0),
        output_scale=selected.std(axis=0),
        residual_coefficient=coefficient,
    )


def fit_transform_suite(raw: MechanisticV2RawFeatures) -> dict[str, BlockTransform]:
    transforms = {
        name: fit_block_transform(name, getattr(raw, name), FEATURE_NAMES[name])
        for name in ("h10", "state", "beta")
    }
    preliminary = {
        name: transforms[name].transform(getattr(raw, name))
        for name in ("h10", "state", "beta")
    }
    residual_base = np.column_stack(
        (preliminary["h10"], preliminary["state"], preliminary["beta"])
    )
    transforms["interaction"] = fit_interaction_transform(
        raw.interaction, residual_base
    )
    return transforms


def transform_blocks(
    transforms: dict[str, BlockTransform], raw: MechanisticV2RawFeatures
) -> dict[str, FloatArray]:
    blocks = {
        name: transforms[name].transform(getattr(raw, name))
        for name in ("h10", "state", "beta")
    }
    residual_base = np.column_stack((blocks["h10"], blocks["state"], blocks["beta"]))
    blocks["interaction"] = transforms["interaction"].transform(
        raw.interaction, residual_base
    )
    return blocks


def _objective_and_gradient(
    parameters: FloatArray,
    design: FloatArray,
    successes: FloatArray,
    trials: FloatArray,
    offset: FloatArray,
    ridge_lambda: float,
) -> tuple[float, FloatArray]:
    intercept = parameters[0]
    coefficient = parameters[1:]
    logits = offset + intercept + design @ coefficient
    total_trials = float(np.sum(trials))
    loss = float(
        np.sum(trials * np.logaddexp(0.0, logits) - successes * logits)
        / total_trials
    )
    loss += float(0.5 * ridge_lambda * np.dot(coefficient, coefficient))
    residual = trials * expit(logits) - successes
    gradient = np.empty_like(parameters)
    gradient[0] = residual.sum() / total_trials
    gradient[1:] = design.T @ residual / total_trials + ridge_lambda * coefficient
    return loss, gradient


def fit_linear(
    name: str,
    block_name: str,
    design: FloatArray,
    successes: FloatArray,
    trials: FloatArray,
    ridge_lambda: float,
    offset: FloatArray | None = None,
) -> LinearFit:
    design = np.asarray(design, dtype=np.float64)
    successes = np.asarray(successes, dtype=np.float64)
    trials = np.asarray(trials, dtype=np.float64)
    if offset is None:
        offset = np.zeros(design.shape[0], dtype=np.float64)
    else:
        offset = np.asarray(offset, dtype=np.float64)
    prior = float(np.clip(successes.sum() / trials.sum(), 1e-8, 1.0 - 1e-8))
    initial = np.zeros(design.shape[1] + 1, dtype=np.float64)
    if np.allclose(offset, 0.0):
        initial[0] = np.log(prior / (1.0 - prior))

    def objective(parameters: FloatArray) -> tuple[float, FloatArray]:
        return _objective_and_gradient(
            parameters, design, successes, trials, offset, ridge_lambda
        )

    fitted = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 5_000, "ftol": 1e-13, "gtol": 1e-8, "maxls": 50},
    )
    value, gradient = objective(fitted.x)
    gradient_max = float(np.max(np.abs(gradient)))
    if not fitted.success and gradient_max > 1e-5:
        raise RuntimeError(f"model {name} failed to converge: {fitted.message}")
    return LinearFit(
        name=name,
        block=block_name,
        coefficient=fitted.x[1:].copy(),
        intercept=float(fitted.x[0]),
        ridge_lambda=float(ridge_lambda),
        objective=float(value),
        gradient_max_abs=gradient_max,
        iterations=int(fitted.nit),
    )


def _binomial_loss(
    successes: FloatArray, trials: FloatArray, logits: FloatArray
) -> float:
    return float(
        np.sum(trials * np.logaddexp(0.0, logits) - successes * logits)
        / np.sum(trials)
    )


@dataclass
class _FoldData:
    train_blocks: dict[str, FloatArray]
    validation_blocks: dict[str, FloatArray]
    train_successes: FloatArray
    validation_successes: FloatArray
    train_trials: FloatArray
    validation_trials: FloatArray
    train_logits: FloatArray
    validation_logits: FloatArray


def _development_folds(
    raw: MechanisticV2RawFeatures,
    successes: FloatArray,
    trials: FloatArray,
    matrix_ids: NDArray[np.int64],
) -> list[_FoldData]:
    folds: list[_FoldData] = []
    for fold in range(CV_FOLDS):
        validation = matrix_cv_fold(matrix_ids) == fold
        train = ~validation
        if not train.any() or not validation.any():
            raise ValueError("whole-matrix CV produced an empty fold")
        transforms = fit_transform_suite(raw.selected(train))
        train_blocks = transform_blocks(transforms, raw.selected(train))
        validation_blocks = transform_blocks(transforms, raw.selected(validation))
        base = fit_linear(
            f"cv{fold}_h10",
            "h10",
            train_blocks["h10"],
            successes[train],
            trials[train],
            ridge_lambda=0.0,
        )
        folds.append(
            _FoldData(
                train_blocks=train_blocks,
                validation_blocks=validation_blocks,
                train_successes=successes[train],
                validation_successes=successes[validation],
                train_trials=trials[train],
                validation_trials=trials[validation],
                train_logits=base.correction(train_blocks["h10"]),
                validation_logits=base.correction(validation_blocks["h10"]),
            )
        )
    return folds


def _select_lambda(
    stage: str, folds: list[_FoldData]
) -> tuple[float, dict[str, float]]:
    scores: dict[str, float] = {}
    for ridge_lambda in RIDGE_LAMBDAS:
        numerator = 0.0
        denominator = 0.0
        for fold_index, fold in enumerate(folds):
            fit = fit_linear(
                f"cv{fold_index}_{stage}_{ridge_lambda:g}",
                stage,
                fold.train_blocks[stage],
                fold.train_successes,
                fold.train_trials,
                ridge_lambda,
                fold.train_logits,
            )
            logits = fold.validation_logits + fit.correction(
                fold.validation_blocks[stage]
            )
            fold_trials = float(fold.validation_trials.sum())
            numerator += _binomial_loss(
                fold.validation_successes, fold.validation_trials, logits
            ) * fold_trials
            denominator += fold_trials
        scores[f"{ridge_lambda:g}"] = numerator / denominator
    minimum = min(scores.values())
    tied = [
        value
        for value in RIDGE_LAMBDAS
        if scores[f"{value:g}"] <= minimum + 1e-12
    ]
    return float(max(tied)), scores


def _advance_folds(stage: str, ridge_lambda: float, folds: list[_FoldData]) -> None:
    for fold_index, fold in enumerate(folds):
        fit = fit_linear(
            f"cv{fold_index}_{stage}_selected",
            stage,
            fold.train_blocks[stage],
            fold.train_successes,
            fold.train_trials,
            ridge_lambda,
            fold.train_logits,
        )
        fold.train_logits = fold.train_logits + fit.correction(fold.train_blocks[stage])
        fold.validation_logits = fold.validation_logits + fit.correction(
            fold.validation_blocks[stage]
        )


def fit_candidate_registry_v2(
    candidate: str,
    raw: MechanisticV2RawFeatures,
    branch_labels: NDArray[np.int8],
    matrix_ids: NDArray[np.int64],
) -> CandidateRegistryV2:
    successes = branch_labels.sum(axis=1).astype(np.float64)
    trials = np.full(branch_labels.shape[0], branch_labels.shape[1], dtype=np.float64)
    matrix_ids = np.asarray(matrix_ids, dtype=np.int64)
    folds = _development_folds(raw, successes, trials, matrix_ids)
    selected: dict[str, float] = {}
    cv_scores: dict[str, dict[str, float]] = {}
    for stage in ("state", "beta", "interaction"):
        selected[stage], cv_scores[stage] = _select_lambda(stage, folds)
        _advance_folds(stage, selected[stage], folds)

    transforms = fit_transform_suite(raw)
    blocks = transform_blocks(transforms, raw)
    base = fit_linear("h10", "h10", blocks["h10"], successes, trials, 0.0)
    h10_logits = base.correction(blocks["h10"])
    state = fit_linear(
        "state", "state", blocks["state"], successes, trials, selected["state"], h10_logits
    )
    state_logits = h10_logits + state.correction(blocks["state"])
    beta = fit_linear(
        "beta", "beta", blocks["beta"], successes, trials, selected["beta"], state_logits
    )
    beta_logits = state_logits + beta.correction(blocks["beta"])
    interaction = fit_linear(
        "interaction",
        "interaction",
        blocks["interaction"],
        successes,
        trials,
        selected["interaction"],
        beta_logits,
    )
    fits = {
        "h10": base,
        "state": state,
        "beta": beta,
        "interaction": interaction,
        "state_only": fit_linear(
            "state_only",
            "state",
            blocks["state"],
            successes,
            trials,
            selected["state"],
        ),
        "beta_only": fit_linear(
            "beta_only",
            "beta",
            blocks["beta"],
            successes,
            trials,
            selected["beta"],
        ),
        "h10_beta": fit_linear(
            "h10_beta",
            "beta",
            blocks["beta"],
            successes,
            trials,
            selected["beta"],
            h10_logits,
        ),
    }
    return CandidateRegistryV2(
        candidate=candidate,
        transforms=transforms,
        fits=fits,
        selected_lambdas=selected,
        cv_scores=cv_scores,
    )


def predict_candidate_registry_v2(
    registry: CandidateRegistryV2, raw: MechanisticV2RawFeatures
) -> dict[str, FloatArray]:
    blocks = transform_blocks(registry.transforms, raw)
    fits = registry.fits
    h10 = fits["h10"].correction(blocks["h10"])
    h10_state = h10 + fits["state"].correction(blocks["state"])
    h10_state_beta = h10_state + fits["beta"].correction(blocks["beta"])
    complete = h10_state_beta + fits["interaction"].correction(blocks["interaction"])
    logits = {
        "h10": h10,
        "h10_state": h10_state,
        "h10_state_beta": h10_state_beta,
        "h10_state_beta_interaction": complete,
        "state_only": fits["state_only"].correction(blocks["state"]),
        "beta_only": fits["beta_only"].correction(blocks["beta"]),
        "h10_beta": h10 + fits["h10_beta"].correction(blocks["beta"]),
    }
    return {name: expit(value) for name, value in logits.items()}


def _transform_metadata(transform: BlockTransform) -> dict[str, Any]:
    return {
        "name": transform.name,
        "source_names": transform.source_names,
        "raw_kept_names": transform.raw_kept_names,
        "output_names": transform.output_names,
        "dropped": transform.dropped,
        "output_features": transform.output_features,
        "uses_pca": False,
        "residualized": transform.residual_coefficient is not None,
    }


def save_registries_v2(
    archive_path: Path,
    contract_path: Path,
    registries: dict[str, CandidateRegistryV2],
) -> None:
    arrays: dict[str, NDArray] = {}
    metadata: dict[str, Any] = {
        "format": "mechanistic-beta-complete-models-v2",
        "ridge_lambdas": RIDGE_LAMBDAS,
        "cv_folds": CV_FOLDS,
        "candidates": {},
    }
    for candidate, registry in registries.items():
        prefix = f"c{candidate}"
        candidate_meta: dict[str, Any] = {
            "transforms": {},
            "fits": {},
            "selected_lambdas": registry.selected_lambdas,
            "cv_scores": registry.cv_scores,
        }
        for name, transform in registry.transforms.items():
            key = f"{prefix}__block__{name}"
            arrays[f"{key}__raw_kept_indices"] = transform.raw_kept_indices
            arrays[f"{key}__raw_mean"] = transform.raw_mean
            arrays[f"{key}__raw_scale"] = transform.raw_scale
            arrays[f"{key}__output_indices"] = transform.output_indices
            arrays[f"{key}__output_mean"] = transform.output_mean
            arrays[f"{key}__output_scale"] = transform.output_scale
            if transform.residual_coefficient is not None:
                arrays[f"{key}__residual_coefficient"] = transform.residual_coefficient
            candidate_meta["transforms"][name] = _transform_metadata(transform)
        for name, fit in registry.fits.items():
            key = f"{prefix}__fit__{name}"
            arrays[f"{key}__coefficient"] = fit.coefficient
            arrays[f"{key}__intercept"] = np.asarray([fit.intercept])
            candidate_meta["fits"][name] = {
                "name": fit.name,
                "block": fit.block,
                "ridge_lambda": fit.ridge_lambda,
                "objective": fit.objective,
                "gradient_max_abs": fit.gradient_max_abs,
                "iterations": fit.iterations,
            }
        metadata["candidates"][candidate] = candidate_meta
    np.savez_compressed(archive_path, **arrays)
    contract_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_registries_v2(
    archive_path: Path, contract_path: Path
) -> dict[str, CandidateRegistryV2]:
    metadata = json.loads(contract_path.read_text(encoding="utf-8"))
    if metadata.get("format") != "mechanistic-beta-complete-models-v2":
        raise ValueError("unsupported v2 model archive")
    registries: dict[str, CandidateRegistryV2] = {}
    with np.load(archive_path) as arrays:
        for candidate, candidate_meta in metadata["candidates"].items():
            prefix = f"c{candidate}"
            transforms: dict[str, BlockTransform] = {}
            for name, item in candidate_meta["transforms"].items():
                key = f"{prefix}__block__{name}"
                residual_key = f"{key}__residual_coefficient"
                transforms[name] = BlockTransform(
                    name=name,
                    source_names=tuple(item["source_names"]),
                    raw_kept_indices=arrays[f"{key}__raw_kept_indices"].copy(),
                    raw_kept_names=tuple(item["raw_kept_names"]),
                    dropped=dict(item["dropped"]),
                    raw_mean=arrays[f"{key}__raw_mean"].copy(),
                    raw_scale=arrays[f"{key}__raw_scale"].copy(),
                    output_indices=arrays[f"{key}__output_indices"].copy(),
                    output_names=tuple(item["output_names"]),
                    output_mean=arrays[f"{key}__output_mean"].copy(),
                    output_scale=arrays[f"{key}__output_scale"].copy(),
                    residual_coefficient=(
                        arrays[residual_key].copy() if residual_key in arrays else None
                    ),
                )
            fits: dict[str, LinearFit] = {}
            for name, item in candidate_meta["fits"].items():
                key = f"{prefix}__fit__{name}"
                fits[name] = LinearFit(
                    name=item["name"],
                    block=item["block"],
                    coefficient=arrays[f"{key}__coefficient"].copy(),
                    intercept=float(arrays[f"{key}__intercept"][0]),
                    ridge_lambda=float(item["ridge_lambda"]),
                    objective=float(item["objective"]),
                    gradient_max_abs=float(item["gradient_max_abs"]),
                    iterations=int(item["iterations"]),
                )
            registries[candidate] = CandidateRegistryV2(
                candidate=candidate,
                transforms=transforms,
                fits=fits,
                selected_lambdas={
                    key: float(value)
                    for key, value in candidate_meta["selected_lambdas"].items()
                },
                cv_scores={
                    key: {score: float(value) for score, value in values.items()}
                    for key, values in candidate_meta["cv_scores"].items()
                },
            )
    return registries
